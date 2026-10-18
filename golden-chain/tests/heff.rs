//! The planar environments and two-site effective Hamiltonian against the
//! symtn-independent fusion-path ED.

use std::sync::Arc;

use golden_chain::ed::{Boundary, Chain, Fib, phi};
use golden_chain::model::{ONE, Space, Tensor, bond_term, physical, virtual_spaces};
use golden_chain::mps::{
    Heff, amplitudes, contract_chain, left_isometry, left_operator, right_isometry, right_operator,
    rot,
};
use num_complex::Complex64;
use symtn::krylov::{self, Lanczos, Which, Workspace};
use symtn::sector::TableRule;
use symtn::typed::{ContractSpec, Runtime, TensorMap};

mod common;

const TOL: f64 = 1e-12;

/// An isometric MPS around bond `(pos, pos+1)` with the environments of `H`.
struct Setup {
    rt: Runtime,
    v: Vec<Space>,
    p: Space,
    al: Vec<Tensor>,
    b: Vec<Tensor>,
    heff: Heff,
    or: Tensor,
    chain: Chain,
    j: f64,
}

fn setup(len: usize, total: Fib, cap: usize, pos: usize, j: f64) -> Setup {
    let rule = Arc::new(TableRule::fibonacci());
    let v = virtual_spaces(&rule, len, total, cap).unwrap();
    setup_with(rule, v, total, pos, j)
}

/// Bond spaces for which canonical isometries embed the *whole* path space
/// around bond `(pos, pos+1)`: every left bond is the full left-reachable
/// space and every right bond the full right-reachable one. The `min` of
/// [`virtual_spaces`] is not enough here: a partial identity onto a smaller
/// space selects an arbitrary subspace, which the Galerkin test covers.
fn exact_spaces(rule: &Arc<TableRule>, len: usize, total: Fib, pos: usize) -> Vec<Space> {
    let step = |[one, tau]: [usize; 2]| [tau, one + tau];
    let mut left = vec![[1, 0]];
    for i in 0..len {
        left.push(step(left[i]));
    }
    let mut right = vec![[0, 0]; len + 1];
    right[len] = if total == Fib::One { [1, 0] } else { [0, 1] };
    for i in (0..len).rev() {
        right[i] = step(right[i + 1]);
    }
    (0..=len)
        .map(|i| {
            let counts = if i < pos { left[i] } else { right[i] };
            let pairs = [Fib::One, Fib::Tau]
                .into_iter()
                .zip(counts)
                .filter(|&(_, n)| n > 0)
                .map(|(x, n)| (golden_chain::model::sector(x), n));
            Space::try_new(Arc::clone(rule), pairs).unwrap()
        })
        .collect()
}

fn setup_with(rule: Arc<TableRule>, v: Vec<Space>, total: Fib, pos: usize, j: f64) -> Setup {
    let len = v.len() - 1;
    let rt = Runtime::builder().build().unwrap();
    let p = physical(&rule).unwrap();
    let h = bond_term(&rt, &p, j).unwrap();
    let al: Vec<_> = (1..pos)
        .map(|k| left_isometry(&rt, &v[k - 1], &p, &v[k]).unwrap())
        .collect();
    let b: Vec<_> = (pos + 2..=len)
        .map(|k| right_isometry(&rt, &v[k - 1], &p, &v[k]).unwrap())
        .collect();
    let ol = left_operator(&rt, &v[0], &p, &h, &al).unwrap();
    let or = right_operator(&rt, &v[len], &p, &h, &b).unwrap();
    let heff = Heff::new(ol, &or, h).unwrap();
    Setup {
        rt,
        v,
        p,
        al,
        b,
        heff,
        or,
        chain: Chain::new(len, Boundary::Open(total)),
        j,
    }
}

impl Setup {
    fn pos(&self) -> usize {
        self.al.len() + 1
    }

    fn theta(&self, seed: u64) -> Tensor {
        let (l, r) = (&self.v[self.pos() - 1], &self.v[self.pos() + 1]);
        TensorMap::rand_with_seed(
            &self.rt,
            [l, &self.p],
            [r, &self.p.try_dual().unwrap()],
            seed,
        )
        .unwrap()
    }

    /// Fusion-path vector of the full state with `theta` at the centre.
    fn embed(&self, theta: &Tensor) -> Vec<Complex64> {
        let mut pieces: Vec<Tensor> = self.al.iter().map(|a| a.repartition(1).unwrap()).collect();
        pieces.push(theta.repartition(1).unwrap());
        pieces.extend(self.b.iter().cloned());
        let psi = contract_chain(&self.rt, &self.v[0], &pieces).unwrap();
        let entries = amplitudes(&psi).unwrap();
        assert_eq!(entries.len(), self.chain.dim(), "state space = path space");
        let mut out = vec![Complex64::new(0.0, 0.0); self.chain.dim()];
        for (path, value) in entries {
            out[self.chain.index(&path).expect("admissible path")] = value;
        }
        out
    }

    fn ed_apply(&self, psi: &[Complex64]) -> Vec<Complex64> {
        let n = self.chain.dim();
        let h = self.chain.hamiltonian(self.j);
        (0..n)
            .map(|row| (0..n).map(|col| psi[col] * h[row * n + col]).sum())
            .collect()
    }
}

fn dot(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter().zip(b).map(|(x, y)| x.conj() * y).sum()
}

fn max_diff(a: &Tensor, b: &Tensor) -> f64 {
    a.axpby(ONE, b, -ONE)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .map(|x| x.norm())
        .fold(0.0, f64::max)
}

/// At full bond dimension: the isometric MPS spans the whole path space,
/// so `Ψ(H_eff θ) = H Ψ(θ)` exactly, at every bond, for both couplings and
/// both total charges.
#[test]
fn full_rank_effective_hamiltonian_is_the_ed_hamiltonian() {
    for total in [Fib::One, Fib::Tau] {
        for j in [1.0, -1.0] {
            for pos in 1..7 {
                let rule = Arc::new(TableRule::fibonacci());
                let v = exact_spaces(&rule, 7, total, pos);
                let s = setup_with(rule, v, total, pos, j);
                let theta = s.theta(11 + pos as u64);
                let lhs = s.embed(&s.heff.apply(&theta).unwrap());
                let rhs = s.ed_apply(&s.embed(&theta));
                for (k, (x, y)) in lhs.iter().zip(&rhs).enumerate() {
                    assert!(
                        (x - y).norm() <= TOL,
                        "total {total:?} J {j} pos {pos} path {k}: {x} != {y}"
                    );
                }
            }
        }
    }
}

/// With truncated bonds: `H_eff` is the Galerkin projection,
/// `⟨θ', H_eff θ⟩ = d_total ⟨Ψ(θ'), H Ψ(θ)⟩_ED`. symtn's `inner` weights
/// each coupled sector by its quantum dimension, and every path of `Ψ` ends
/// in the coupled sector `x_L = total`, so the isometric embedding scales the
/// ED dot product by `d_total` (1 or φ).
#[test]
fn truncated_effective_hamiltonian_is_the_projected_ed_hamiltonian() {
    for (total, d) in [(Fib::One, 1.0), (Fib::Tau, phi())] {
        for pos in 1..8 {
            let s = setup(8, total, 2, pos, 1.0);
            let (theta, other) = (s.theta(3), s.theta(4));
            let bra = s.embed(&other);
            let overlap = other.inner(&theta).unwrap();
            let ed_overlap = d * dot(&bra, &s.embed(&theta));
            assert!(
                (overlap - ed_overlap).norm() <= TOL * ed_overlap.norm().max(1.0),
                "total {total:?} pos {pos}: {overlap} != {ed_overlap}"
            );
            let lhs = other.inner(&s.heff.apply(&theta).unwrap()).unwrap();
            let rhs = d * dot(&bra, &s.ed_apply(&s.embed(&theta)));
            assert!(
                (lhs - rhs).norm() <= TOL * rhs.norm().max(1.0),
                "total {total:?} pos {pos}: {lhs} != {rhs}"
            );
        }
    }
}

/// The Jacobi oracle on the TensorKit-pinned three-anyon matrix (`tests/ed.rs`).
#[test]
fn jacobi_oracle_matches_the_three_anyon_ground_energy() {
    let chain = Chain::new(3, Boundary::Open(Fib::Tau));
    let e0 = common::jacobi_ground(chain.hamiltonian(1.0), chain.dim()).0;
    // Lowest eigenvalue of -[[a, b], [b, c]] (the three-anyon matrix of `tests/ed.rs`), by hand.
    let (a, b, c) = (1.0 + phi().powi(-2), phi().powf(-1.5), phi().recip());
    let hand = -((a + c) + ((a - c).powi(2) + 4.0 * b * b).sqrt()) / 2.0;
    assert!((e0 - hand).abs() <= 1e-13, "{e0} != {hand}");
}

/// Exact embedding: with full bond spaces `H_eff` at the centre bond is
/// the whole ED Hamiltonian, so `symtn::krylov::eigsolve` on [`Heff`] as a
/// `LinearMap` returns the Jacobi ED ground energy, for both couplings and
/// both total charges. No truncation and no SVD are involved.
#[test]
fn centre_bond_lanczos_finds_the_ed_ground_energy() {
    // L = 12 has 89 / 144 paths; the dense Jacobi oracle bounds the range.
    let mut max_numiter = 0;
    for len in 6..=12 {
        let pos = len / 2;
        for total in [Fib::One, Fib::Tau] {
            for j in [1.0, -1.0] {
                let chain = Chain::new(len, Boundary::Open(total));
                let e0 = common::jacobi_ground(chain.hamiltonian(j), chain.dim()).0;
                let rule = Arc::new(TableRule::fibonacci());
                let v = exact_spaces(&rule, len, total, pos);
                let mut s = setup_with(rule, v, total, pos, j);
                let x0 = s.theta(21 + len as u64);
                // MPSKit's DMRG krylovdim; restarts cover dim > 30; a 1e-12
                // residual bounds the eigenvalue error far below 1e-10.
                let alg = Lanczos::new(30, 100, 1e-12);
                let mut ws = Workspace::with_capacity(&x0, alg.get_krylovdim()).unwrap();
                let (values, _, info) =
                    krylov::eigsolve(&mut s.heff, &x0, 1, Which::SmallestReal, &alg, &mut ws)
                        .unwrap();
                assert!(info.converged >= 1, "L {len} total {total:?} J {j}");
                max_numiter = max_numiter.max(info.numiter);
                assert!(
                    (values[0] - e0).abs() <= 1e-10,
                    "L {len} total {total:?} J {j}: {} != {e0}",
                    values[0]
                );
            }
        }
    }
    assert!(max_numiter > 1, "no case needed a thick restart");
}

/// The planar lowering round-trips and refuses a cut that would braid.
#[test]
fn planar_steps_round_trip_and_reject_non_planar_cuts() {
    let s = setup(6, Fib::Tau, 3, 3, 1.0);
    let theta = s.theta(5);
    assert!(max_diff(&rot(&rot(&theta).unwrap()).unwrap(), &theta) <= TOL);
    // The effective Hamiltonian lands on θ's exact space: `axpby_into` would
    // refuse a different space or block layout.
    let mut out = theta.zeros_like().unwrap();
    s.heff
        .apply(&theta)
        .unwrap()
        .axpby_into(&mut out, ONE, ONE)
        .unwrap();
    // h's planar order is (a, b, d, c): (a, d) is not contiguous.
    let h = bond_term(&s.rt, &s.p, 1.0).unwrap();
    let cut = |lhs, rhs| ContractSpec {
        lhs,
        rhs,
        codomain: &[2, 0],
        domain: &[3, 1],
    };
    assert!(h.planar_contract(&theta, &cut(&[0, 3], &[1, 3])).is_err());
    // θ's planar order is (l, p1, p2, r): pairing (c, d) with (p2, p1) twists.
    assert!(h.planar_contract(&theta, &cut(&[3, 2], &[1, 3])).is_err());
}

/// The right term's shortcut: `θ ∘ rot(OR̃) = rot(OR̃ ∘ rot(θ))`, so the
/// hot path needs no transpose of θ.
#[test]
fn rotated_right_operator_acts_from_the_domain() {
    for pos in 1..6 {
        let s = setup(7, Fib::One, 3, pos, 1.0);
        let theta = s.theta(7);
        let shortcut = theta.compose(&rot(&s.or).unwrap()).unwrap();
        let rotated = rot(&s.or.compose(&rot(&theta).unwrap()).unwrap()).unwrap();
        assert!(max_diff(&shortcut, &rotated) <= TOL, "pos {pos}");
    }
}

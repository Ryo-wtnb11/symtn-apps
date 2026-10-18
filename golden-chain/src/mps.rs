//! Open-chain anyonic MPS, its environments and the two-site effective
//! Hamiltonian.
//!
//! Layouts (flat symtn axes; planar order = codomain, then domain reversed):
//! - `AL_i ∈ [V_{i-1}, P] <- [V_i]` (MPSKit `AL[-1 -2; -3]`), left canonical;
//! - `B_i ∈ [V_{i-1}] <- [V_i, P*]`, right canonical (MPSKit `AR` before
//!   `_transpose_front`; `AR.repartition(1)`);
//! - `θ ∈ [V_{i-1}, P] <- [V_{i+1}, P*] = AL_i ∘ B_{i+1}` (MPSKit DMRG2 `ac2`).
//!
//! Why no MPO, unlike MPSKit's `FiniteMPOHamiltonian`: its Jordan MPO comes
//! from an SVD of the bond term, and for nearest-neighbour terms the three
//! nonzero channels of its environment are kept explicitly instead — the
//! block Hamiltonian `HL` and the open bond `EL`, pre-summed into
//! `OL = HL ⊗ id_P + EL` (MPSKit's prepared `GL_O`). Mirrored on the right.

use symtn::krylov::LinearMap;
use symtn::typed::{ContractSpec, Error, Runtime, TensorMap};

use crate::ed::Fib;
use crate::model::{ONE, Space, Tensor, ZERO, label};

/// `AL ∈ [v_prev, p] <- [v]`, the canonical (partial-identity) isometry.
pub fn left_isometry(rt: &Runtime, v_prev: &Space, p: &Space, v: &Space) -> Result<Tensor, Error> {
    TensorMap::isometry(rt, [v_prev, p], [v])
}

/// `B ∈ [v_prev] <- [v, p*]` with `B ∘ B† = id`, without a QR.
pub fn right_isometry(rt: &Runtime, v_prev: &Space, p: &Space, v: &Space) -> Result<Tensor, Error> {
    TensorMap::isometry(rt, [v, &p.try_dual()?], [v_prev])?.adjoint()
}

/// The full planar rotation of a rank-4 tensor with two codomain legs,
/// TensorKit `transpose(t)`; its own inverse.
pub fn rot(t: &Tensor) -> Result<Tensor, Error> {
    t.transpose(&[3, 2], &[1, 0])
}

/// `EL ∈ [V_i, P] <- [V_i, P]`: bond `(i, i+1)` with site `i = a` contracted,
/// `conj(A[1 2; -1]) h[2 -2; 3 -4] A[1 3; -3]`.
fn left_bond(h: &Tensor, a: &Tensor) -> Result<Tensor, Error> {
    // h: a=0 b=1 | c=2 d=3; A: l=0 c=1 | i=2.
    let z = h.planar_contract(a, &spec(&[2], &[1], &[0, 1, 2], &[3, 4]))?; // [a,b,d] <- [l,i]
    // Open legs i'=0 | b=1 d=2 i=3.
    a.adjoint()?
        .planar_contract(&z, &spec(&[2, 1], &[0, 3], &[0, 1], &[3, 2])) // [i',b] <- [i,d]
}

/// `ER̃ ∈ [P, V_{k-1}*] <- [P, V_{k-1}*]`: bond `(k-1, k)` with the rotated
/// right tensor `bt = B̃_k ∈ [P, V_k*] <- [V_{k-1}*]` contracted.
fn right_bond(h: &Tensor, bt: &Tensor) -> Result<Tensor, Error> {
    // bt: q=0 y=1 | x=2.
    let z = h.planar_contract(bt, &spec(&[3], &[0], &[2, 0, 1], &[4, 3]))?; // [c,a,b] <- [x,y]
    // Open legs x'=0 | c=1 a=2 x=3.
    bt.adjoint()?
        .planar_contract(&z, &spec(&[2, 1], &[4, 2], &[2, 0], &[1, 3])) // [a,x'] <- [c,x]
}

fn spec<'a>(
    lhs: &'a [usize],
    rhs: &'a [usize],
    codomain: &'a [usize],
    domain: &'a [usize],
) -> ContractSpec<'a> {
    ContractSpec {
        lhs,
        rhs,
        codomain,
        domain,
    }
}

/// `OL_pos ∈ [V_{pos-1}, P] <- [V_{pos-1}, P]` from `al = AL_1..AL_{pos-1}`.
pub fn left_operator(
    rt: &Runtime,
    v0: &Space,
    p: &Space,
    h: &Tensor,
    al: &[Tensor],
) -> Result<Tensor, Error> {
    let id_p = TensorMap::isomorphism(rt, [p], [p])?;
    let mut ol = TensorMap::zeros(rt, [v0, p], [v0, p])?;
    for a in al {
        let hl = a.adjoint()?.compose(&ol.compose(a)?)?;
        ol = hl.otimes(&id_p)?.axpby(ONE, &left_bond(h, a)?, ONE)?;
    }
    Ok(ol)
}

/// `OR̃_{pos+1} ∈ [P, V_{pos+1}*] <- [P, V_{pos+1}*]` (rotated layout) from
/// `b = B_{pos+2}..B_L`; `vl` is `V_L`.
pub fn right_operator(
    rt: &Runtime,
    vl: &Space,
    p: &Space,
    h: &Tensor,
    b: &[Tensor],
) -> Result<Tensor, Error> {
    let id_p = TensorMap::isomorphism(rt, [p], [p])?;
    let vl_dual = vl.try_dual()?;
    let mut or = TensorMap::zeros(rt, [p, &vl_dual], [p, &vl_dual])?;
    for bk in b.iter().rev() {
        let bt = bk.transpose(&[2, 1], &[0])?; // [P, V_k*] <- [V_{k-1}*]
        let hr = bt.adjoint()?.compose(&or.compose(&bt)?)?;
        or = id_p.otimes(&hr)?.axpby(ONE, &right_bond(h, &bt)?, ONE)?;
    }
    Ok(or)
}

/// `X ∘ Y` for rank-4 `X`, `Y` with two codomain legs, as a planar
/// contraction so it can accumulate into a destination (symtn has no
/// `compose_into`).
const COMPOSE: ContractSpec<'static> = ContractSpec {
    lhs: &[2, 3],
    rhs: &[0, 1],
    codomain: &[0, 1],
    domain: &[2, 3],
};

/// `h_{(p1,p2)} θ`: h's domain `(c, d)` meets θ's `(p1, p2)`. With θ:
/// l=0 p1=1 | r=2 p2=3, the open legs are a=0 b=1 | l=2 r=3, returned in
/// θ's layout `[l, a] <- [r, b]`.
const BOND: ContractSpec<'static> = ContractSpec {
    lhs: &[3, 2],
    rhs: &[3, 1],
    codomain: &[2, 0],
    domain: &[3, 1],
};

/// Two-site effective Hamiltonian `θ ↦ OL∘θ + θ∘rot(OR̃) + h_{(p1,p2)} θ`,
/// the `symtn::krylov::LinearMap` of the centre-bond eigenproblem.
pub struct Heff {
    ol: Tensor,
    or_dom: Tensor,
    h: Tensor,
}

impl Heff {
    /// `or` is `OR̃` in the rotated layout; it is rotated into θ's domain once
    /// so the hot path needs no transpose for the right term.
    pub fn new(ol: Tensor, or: &Tensor, h: Tensor) -> Result<Self, Error> {
        Ok(Self {
            ol,
            or_dom: rot(or)?,
            h,
        })
    }

    pub fn apply(&self, theta: &Tensor) -> Result<Tensor, Error> {
        let mut out = self.h.planar_contract(theta, &BOND)?;
        self.add_sides(theta, &mut out)?;
        Ok(out)
    }

    /// `out ← H_eff θ`; `out` must have θ's space and is not read.
    pub fn apply_into(&self, theta: &Tensor, out: &mut Tensor) -> Result<(), Error> {
        self.h.planar_contract_into(theta, &BOND, out, ONE, ZERO)?;
        self.add_sides(theta, out)
    }

    /// `out += OL∘θ + θ∘rot(OR̃)`.
    fn add_sides(&self, theta: &Tensor, out: &mut Tensor) -> Result<(), Error> {
        self.ol
            .planar_contract_into(theta, &COMPOSE, out, ONE, ONE)?;
        theta.planar_contract_into(&self.or_dom, &COMPOSE, out, ONE, ONE)
    }
}

impl LinearMap<Tensor> for Heff {
    fn apply_into(&mut self, x: &Tensor, y: &mut Tensor) -> Result<(), Error> {
        Heff::apply_into(self, x, y)
    }
}

/// Contracts `pieces`, each `[V] <- [V', …]` (an `AL.repartition(1)`, a θ
/// `.repartition(1)` or a `B`), into `Ψ ∈ [V_0, P, …, P] <- [V_L]`.
pub fn contract_chain(rt: &Runtime, v0: &Space, pieces: &[Tensor]) -> Result<Tensor, Error> {
    let mut state = TensorMap::isomorphism(rt, [v0], [v0])?;
    for piece in pieces {
        state = state.compose(piece)?;
        state = state.repartition(state.rank() - 1)?;
    }
    Ok(state)
}

/// Fusion-path amplitudes `(x_0..=x_L, ψ)` of a full state
/// `Ψ ∈ [V_0, P^L] <- [V_L]` with one-dimensional `V_0`, `V_L`.
///
/// The left-associated codomain tree's inner lines are `x_1..x_{L-1}` and its
/// coupled sector is `x_L`; all trees share that coupled sector, so the
/// dim-weighted inner product is a global factor of the path sum.
pub fn amplitudes(psi: &Tensor) -> Result<Vec<(Vec<Fib>, num_complex::Complex64)>, Error> {
    let zeros = vec![0; psi.rank()];
    psi.subblocks()?
        .map(|(trees, block)| {
            let mut path = vec![Fib::One];
            path.extend(trees.codomain_innerlines().iter().map(label));
            path.push(label(trees.coupled()));
            let value = *block.get(&zeros).ok_or_else(|| {
                Error::InvalidArgument("state legs must be one-dimensional".into())
            })?;
            Ok((path, value))
        })
        .collect()
}

/// Von Neumann entanglement entropy of a Schmidt spectrum given per sector,
/// in MPSKit's convention (0.13.13 `src/algorithms/toolbox.jl:entropy`):
/// `S = -Σ_s d_s Σ_k λ²_{s,k} ln λ²_{s,k}`, with values `λ < ε` dropped.
/// The caller must pass a normalized spectrum, `Σ_s d_s Σ_k λ²_{s,k} = 1`;
/// like MPSKit, this function does not normalize.
pub fn entropy<'a>(spectrum: impl IntoIterator<Item = (Fib, &'a [f64])>) -> f64 {
    -spectrum
        .into_iter()
        .map(|(s, values)| {
            let sum: f64 = values
                .iter()
                .filter(|&&x| x >= f64::EPSILON)
                .map(|&x| x * x * (x * x).ln())
                .sum();
            crate::ed::quantum_dim(s) * sum
        })
        .sum::<f64>()
}

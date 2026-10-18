//! Eigensolvers for the fusion-path ED oracle. Neither touches a symtn
//! tensor: a dense cyclic Jacobi for small chains, and krylov's Lanczos on a
//! plain `Vec<f64>` for chains too long for a dense matrix (krylov is
//! validated against KrylovKit on its own; `symtn::krylov` is only its
//! re-export, so the app resolves the one krylov symtn links).

#![allow(dead_code)]

use std::convert::Infallible;

use golden_chain::dmrg::Dmrg2;
use golden_chain::ed::Chain;
use symtn::krylov::{self, Lanczos, Which, Workspace};

/// Eigenvalues and row-major eigenvectors (column `k` belongs to value `k`)
/// of a dense row-major real symmetric matrix, by cyclic Jacobi rotations.
pub fn jacobi(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v: Vec<f64> = (0..n * n)
        .map(|k| if k / n == k % n { 1.0 } else { 0.0 })
        .collect();
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j].powi(2))
            .sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p * n + q].abs() < 1e-300 {
                    continue;
                }
                let tau = (a[q * n + q] - a[p * n + p]) / (2.0 * a[p * n + q]);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let t = if tau == 0.0 { 1.0 } else { t };
                let (c, sn) = (1.0 / (1.0 + t * t).sqrt(), t / (1.0 + t * t).sqrt());
                for m in [&mut a, &mut v] {
                    for k in 0..n {
                        let (mkp, mkq) = (m[k * n + p], m[k * n + q]);
                        m[k * n + p] = c * mkp - sn * mkq;
                        m[k * n + q] = sn * mkp + c * mkq;
                    }
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - sn * aqk;
                    a[q * n + k] = sn * apk + c * aqk;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

/// Lowest eigenvalue and its unit eigenvector, by [`jacobi`].
pub fn jacobi_ground(a: Vec<f64>, n: usize) -> (f64, Vec<f64>) {
    let (values, vectors) = jacobi(a, n);
    let k = (0..n)
        .min_by(|&x, &y| values[x].total_cmp(&values[y]))
        .expect("n > 0");
    (values[k], (0..n).map(|i| vectors[i * n + k]).collect())
}

/// A plain real vector for krylov (orphan rule: neither `Vec` nor `Vector`
/// is ours).
pub struct EdVec(pub Vec<f64>);

impl krylov::Vector for EdVec {
    type Scalar = f64;
    type Error = Infallible;
    fn similar(&self) -> Result<Self, Infallible> {
        Ok(EdVec(vec![0.0; self.0.len()]))
    }
    fn copy_from(&mut self, x: &Self) -> Result<(), Infallible> {
        self.0.copy_from_slice(&x.0);
        Ok(())
    }
    fn inner(&self, y: &Self) -> Result<f64, Infallible> {
        Ok(self.0.iter().zip(&y.0).map(|(a, b)| a * b).sum())
    }
    fn norm(&self) -> Result<f64, Infallible> {
        Ok(self.inner(self)?.sqrt())
    }
    fn axpby(&mut self, a: f64, x: &Self, b: f64) -> Result<(), Infallible> {
        for (s, x) in self.0.iter_mut().zip(&x.0) {
            // `b == 0` must not read `self` (krylov's contract).
            *s = if b == 0.0 { a * x } else { a * x + b * *s };
        }
        Ok(())
    }
}

/// ED ground energy of `H = -J Σ P^(1)` by Lanczos on the sparse path-basis
/// `H` ([`Chain::apply_hamiltonian`]).
pub fn lanczos_ground_energy(chain: &Chain, j: f64) -> f64 {
    let n = chain.dim();
    // A start vector with no symmetry: every path weighted differently.
    let x0 = EdVec((0..n).map(|k| 1.0 + (k as f64 * 0.618).sin()).collect());
    // krylovdim 30 with many restarts; a 1e-12 residual bounds the
    // eigenvalue error by ~1e-24 / gap, far below the 1e-9 DMRG gate.
    let alg = Lanczos::new(30, 1000, 1e-12);
    let mut ws = Workspace::with_capacity(&x0, alg.get_krylovdim()).unwrap();
    let mut op = |x: &EdVec, y: &mut EdVec| -> Result<(), Infallible> {
        chain.apply_hamiltonian(j, &x.0, &mut y.0);
        Ok(())
    };
    let (values, _, info) =
        krylov::eigsolve(&mut op, &x0, 1, Which::SmallestReal, &alg, &mut ws).unwrap();
    assert!(info.converged >= 1, "ED Lanczos did not converge");
    values[0]
}

/// MPSKit 0.13.13 `DMRG2` and `Defaults` values, spelled out (`tol = 1e-10`,
/// `krylovdim = 30`, eigensolver `maxiter = 200`, `updatetol` floor `1e-14`
/// and factor `1e-3`), with the fixture generator's two changes:
/// - the `updatetol` ceiling is 1e-12, not 1e-4: at 1e-4 the second sweep's
///   eigensolves return their start vectors unchanged, so `ϵ ≈ 1e-16` stops
///   the run ~1e-9 above the ground energy (MPSKit does the same);
/// - at most 50 sweeps, not 200: every case converges in a few, and one that
///   does not is a failure, not something to wait for.
pub fn mpskit_dmrg2(schedule: Vec<usize>) -> Dmrg2 {
    Dmrg2 {
        schedule,
        tol: 1e-10,
        maxiter: 50,
        krylovdim: 30,
        eig_maxiter: 200,
        eig_tol_min: 1e-14,
        eig_tol_max: 1e-12,
        eig_tol_factor: 1e-3,
        seed: 7,
    }
}

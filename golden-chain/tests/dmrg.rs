//! Two-site DMRG against the symtn-independent fusion-path ED (ground energy) and the
//! entanglement-spectrum relation between the two.

mod common;

use std::collections::BTreeMap;

use golden_chain::dmrg::{GroundState, find_groundstate};
use golden_chain::ed::{Boundary, Chain, Fib, quantum_dim};

/// The sparse-`H` Lanczos ED oracle equals the dense Jacobi oracle wherever
/// both run, so it can stand in for Jacobi at L = 20 (6765 paths).
#[test]
fn sparse_lanczos_ed_matches_dense_jacobi() {
    for len in 4..=12 {
        for total in [Fib::One, Fib::Tau] {
            for j in [1.0, -1.0] {
                let chain = Chain::new(len, Boundary::Open(total));
                let (dense, _) = common::jacobi_ground(chain.hamiltonian(j), chain.dim());
                let sparse = common::lanczos_ground_energy(&chain, j);
                assert!(
                    (dense - sparse).abs() <= 1e-11,
                    "L {len} total {total:?} J {j}: {dense} != {sparse}"
                );
            }
        }
    }
}

/// DMRG with a bond dimension ramp reaches the ED ground energy to 1e-9
/// for both couplings and both total charges, up to L = 20. At D = 128 the
/// centre bond of L = 20 (total quantum dimension 34 + 55φ ≈ 123) is kept
/// whole, so the gate tests the sweep and the eigensolver, not truncation.
#[test]
fn dmrg_ground_energy_matches_ed() {
    for len in [6, 11, 16, 20] {
        for total in [Fib::One, Fib::Tau] {
            for j in [1.0, -1.0] {
                let chain = Chain::new(len, Boundary::Open(total));
                let ed = common::lanczos_ground_energy(&chain, j);
                let gs = find_groundstate(len, j, total, &common::mpskit_dmrg2(vec![32, 64, 128]))
                    .unwrap();
                assert!(
                    gs.converged,
                    "L {len} total {total:?} J {j}: {:?}",
                    gs.sweeps
                );
                assert!(
                    (gs.energy - ed).abs() <= 1e-9,
                    "L {len} total {total:?} J {j}: DMRG {} != ED {ed}",
                    gs.energy
                );
            }
        }
    }
}

/// Squared Schmidt values per sector `x_bond` of a unit ED vector: the
/// eigenvalues of `M_x M_xᵀ`, `M_x[left path, right path]` the amplitudes
/// with `x_bond = x`.
fn ed_schmidt(chain: &Chain, psi: &[f64], bond: usize) -> BTreeMap<Fib, Vec<f64>> {
    // Per sector: left prefixes (rows), right suffixes (columns), entries.
    type Block<'a> = (Vec<&'a [Fib]>, Vec<&'a [Fib]>, Vec<(usize, usize, f64)>);
    let mut blocks: BTreeMap<Fib, Block> = BTreeMap::new();
    for (path, &value) in chain.paths().iter().zip(psi) {
        let (rows, cols, entries) = blocks.entry(path[bond]).or_default();
        fn index<'a>(list: &mut Vec<&'a [Fib]>, key: &'a [Fib]) -> usize {
            list.iter().position(|k| *k == key).unwrap_or_else(|| {
                list.push(key);
                list.len() - 1
            })
        }
        let r = index(rows, &path[..bond]);
        let c = index(cols, &path[bond + 1..]);
        entries.push((r, c, value));
    }
    blocks
        .into_iter()
        .map(|(x, (rows, cols, entries))| {
            let (m, n) = (rows.len(), cols.len());
            let mut a = vec![0.0; m * n];
            for (r, c, v) in entries {
                a[r * n + c] = v;
            }
            let gram: Vec<f64> = (0..m * m)
                .map(|k| {
                    (0..n)
                        .map(|l| a[(k / m) * n + l] * a[(k % m) * n + l])
                        .sum()
                })
                .collect();
            (x, common::jacobi(gram, m).0)
        })
        .collect()
}

/// Sorted descending, values below `cut` dropped.
fn significant(values: impl IntoIterator<Item = f64>, cut: f64) -> Vec<f64> {
    let mut v: Vec<f64> = values.into_iter().filter(|&x| x > cut).collect();
    v.sort_by(|a, b| b.total_cmp(a));
    v
}

/// The design's derived relation between the MPS and ED Schmidt values,
/// checked at every bond: for a unit ED vector with singular values `σ_{x,k}`
/// of `M_x`, MPSKit's normalized spectrum is `λ_{x,k} = σ_{x,k} / √d_x`.
/// (In the design's normalization, `Σ|ψ|² = 1/d_c` because the TensorKit
/// norm of the state is `d_c` times the path norm, this is
/// `λ = σ √(d_c/d_x)`.) The sum rule `Σ_x d_x Σ_k λ² = 1` follows.
#[test]
fn entanglement_spectrum_matches_ed_schmidt_values() {
    let len = 9;
    for total in [Fib::One, Fib::Tau] {
        for j in [1.0, -1.0] {
            let chain = Chain::new(len, Boundary::Open(total));
            let (e0, psi) = common::jacobi_ground(chain.hamiltonian(j), chain.dim());
            let gs: GroundState =
                find_groundstate(len, j, total, &common::mpskit_dmrg2(vec![64])).unwrap();
            assert!((gs.energy - e0).abs() <= 1e-10);
            for (bond, spectrum) in (1..len).zip(&gs.spectra) {
                let ed = ed_schmidt(&chain, &psi, bond);
                let weight: f64 = spectrum
                    .iter()
                    .map(|(x, v)| quantum_dim(*x) * v.iter().map(|l| l * l).sum::<f64>())
                    .sum();
                assert!(
                    (weight - 1.0).abs() <= 1e-12,
                    "bond {bond}: Σ d λ² = {weight}"
                );
                for x in [Fib::One, Fib::Tau] {
                    let values = spectrum
                        .iter()
                        .filter(|(y, _)| *y == x)
                        .flat_map(|(_, v)| v);
                    let mps = significant(values.map(|l| l * l), 1e-12);
                    let ed = ed.get(&x).into_iter().flatten();
                    let expected = significant(ed.map(|s2| s2 / quantum_dim(x)), 1e-12);
                    assert_eq!(mps.len(), expected.len(), "bond {bond} sector {x:?}");
                    for (a, b) in mps.iter().zip(&expected) {
                        assert!(
                            (a - b).abs() <= 1e-9,
                            "total {total:?} J {j} bond {bond} sector {x:?}: {a} != {b}"
                        );
                    }
                }
            }
        }
    }
}

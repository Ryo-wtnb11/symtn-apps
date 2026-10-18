//! Slow (`#[ignore]`): the central charge from the open-chain
//! entanglement profile `S(l) = (c/6) ln[(2L/π) sin(πl/L)] + s'`.
//!
//! Feiguin et al., PRL 98, 160409 (2007), p. 3: the chain favouring the
//! trivial channel (here `J = +1`) is the k = 3 RSOS model, the tricritical
//! Ising CFT with `c = 7/10`; the ferromagnetic chain favouring `τ`
//! (`J = -1`) is the 3-state Potts CFT with `c = 4/5`. Their open-chain fit
//! gives `c = 0.70 ± 0.01`.
//!
//! `cargo test --release --test central_charge -- --ignored --nocapture`

mod common;

use golden_chain::dmrg::find_groundstate;
use golden_chain::ed::Fib;

/// Least-squares slope of `y` against `x`.
fn slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let (mx, my) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), (x, y)| (a + x / n, b + y / n));
    let sxy: f64 = points.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = points.iter().map(|(x, _)| (x - mx).powi(2)).sum();
    sxy / sxx
}

/// `c` from bonds `L/8 ≤ l ≤ 7L/8`, fitted separately on each residue of
/// `l mod 6` and averaged: the open chain's entropy alternates with bond
/// parity, and at `J = -1` it also has period 3 (the Potts `Z_3`).
fn central_charge(entropies: &[f64], len: usize) -> f64 {
    let fit = |residue: usize| {
        let points: Vec<(f64, f64)> = (len / 8..=7 * len / 8)
            .filter(|l| l % 6 == residue)
            .map(|l| {
                let chord = (2.0 * len as f64 / std::f64::consts::PI)
                    * (std::f64::consts::PI * l as f64 / len as f64).sin();
                (chord.ln() / 6.0, entropies[l - 1])
            })
            .collect();
        slope(&points)
    };
    (0..6).map(fit).sum::<f64>() / 6.0
}

#[test]
#[ignore = "slow: L = 96 DMRG; run in release"]
fn central_charge_from_the_entanglement_profile() {
    // L ≡ 0 mod 6: at J = -1 a length not divisible by 3 frustrates the
    // Potts order at the open ends and adds an L-dependent entropy bump
    // (half-chain S of the J = -1 ground state: 0.56, 0.95, 0.99, 0.67 at L = 12, 16,
    // 20, 24). D = 128 in the last sweep keeps the truncation error ≲ 1e-8.
    const LEN: usize = 96;
    for (j, expected) in [(1.0, 0.7), (-1.0, 0.8)] {
        let gs =
            find_groundstate(LEN, j, Fib::One, &common::mpskit_dmrg2(vec![32, 64, 128])).unwrap();
        let c = central_charge(&gs.entropies(), LEN);
        let last = gs.sweeps.last().unwrap();
        eprintln!(
            "J {j}: c = {c:.4} (expected {expected}), E = {}, sweeps {}, converged {}, truncation error {:.1e}",
            gs.energy,
            gs.sweeps.len(),
            gs.converged,
            last.max_truncation_error
        );
        // A loose band: finite size, boundary terms and MPSKit's entropy
        // convention (no `Σ p_s ln d_s` term) all shift a 96-site fit.
        assert!((c - expected).abs() <= 0.05, "J {j}: c = {c}");
    }
}

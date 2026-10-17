//! The entanglement entropy in MPSKit's convention, against hand values.

use golden_chain::ed::{Fib, phi};
use golden_chain::mps::entropy;

/// A product state: one Schmidt value 1 in the vacuum sector.
#[test]
fn product_state_has_zero_entropy() {
    assert_eq!(entropy([(Fib::One, &[1.0][..])]), 0.0);
}

/// Two sectors with equal weight, `λ²_1 = 1/2` and `d_τ λ²_τ = 1/2`:
/// `S = -(½ ln ½ + ½ ln(1/(2φ))) = ln 2 + ½ ln φ`. The `d_τ` weight is what
/// distinguishes MPSKit's convention from a plain sum over Schmidt values.
#[test]
fn two_sector_state_is_weighted_by_quantum_dimension() {
    let one = [0.5_f64.sqrt()];
    let tau = [(0.5 / phi()).sqrt(), 0.0, 1e-17];
    let s = entropy([(Fib::One, &one[..]), (Fib::Tau, &tau[..])]);
    let hand = 2.0_f64.ln() + 0.5 * phi().ln();
    assert!((s - hand).abs() <= 1e-15, "{s} != {hand}");
}

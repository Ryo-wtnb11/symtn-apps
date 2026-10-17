//! The golden-chain model on symtn tensors: the physical leg, the bond
//! term, and the virtual spaces of an open-chain MPS.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{TableRule, TableSector};
use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};

use crate::ed::Fib;

pub type Tensor = TensorMap<TableRule, Complex64>;
pub type Space = GradedSpace<TableRule>;

pub const ONE: Complex64 = Complex64::new(1.0, 0.0);
pub const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// `TableRule::fibonacci()` labels: `1` is `TableSector::new(1)`, `τ` is `2`.
pub fn sector(x: Fib) -> TableSector {
    match x {
        Fib::One => TableSector::new(1),
        Fib::Tau => TableSector::new(2),
    }
}

pub fn label(s: &TableSector) -> Fib {
    if *s == sector(Fib::One) {
        Fib::One
    } else {
        Fib::Tau
    }
}

/// The physical leg `τ` (degeneracy 1).
pub fn physical(rule: &Arc<TableRule>) -> Result<Space, Error> {
    GradedSpace::try_new(Arc::clone(rule), [(sector(Fib::Tau), 1)])
}

/// `h = -J P^(1)` on `[P, P] <- [P, P]`: `-J` on the coupled vacuum, `0` on
/// coupled `τ`. Same construction as MPSKitModels' `hard_hexagon` (`block(O, I)`).
pub fn bond_term(rt: &Runtime, p: &Space, j: f64) -> Result<Tensor, Error> {
    TensorMap::from_subblock_fn(rt, [p, p], [p, p], |trees, _| {
        if *trees.coupled() == sector(Fib::One) {
            Complex64::new(-j, 0.0)
        } else {
            ZERO
        }
    })
}

/// Virtual spaces `V_0..=V_L` of the open chain with `x_0 = 1`, `x_L = total`.
///
/// Sector `s` at bond `i` gets `min(#paths 0→i ending in s, #paths i→L
/// starting at s, cap)`, which is exactly the reachable space for a large
/// `cap`. The minimum keeps `V_i ≾ V_{i-1} ⊗ τ` and `V_{i-1} ≾ V_i ⊗ τ̄`
/// sectorwise, so left and right isometries exist at every site.
pub fn virtual_spaces(
    rule: &Arc<TableRule>,
    len: usize,
    total: Fib,
    cap: usize,
) -> Result<Vec<Space>, Error> {
    // counts[i] = [#One, #Tau]
    let mut left = vec![[1usize, 0]];
    for i in 0..len {
        let [one, tau] = left[i];
        left.push([tau, one.saturating_add(tau)]);
    }
    let mut right = vec![[0usize; 2]; len + 1];
    right[len] = match total {
        Fib::One => [1, 0],
        Fib::Tau => [0, 1],
    };
    for i in (0..len).rev() {
        let [one, tau] = right[i + 1];
        right[i] = [tau, one.saturating_add(tau)];
    }
    (0..=len)
        .map(|i| {
            let pairs = [Fib::One, Fib::Tau]
                .into_iter()
                .enumerate()
                .map(|(k, x)| (sector(x), left[i][k].min(right[i][k]).min(cap)))
                .filter(|&(_, n)| n > 0);
            GradedSpace::try_new(Arc::clone(rule), pairs)
        })
        .collect()
}

//! Interim planar contraction, until symtn #2284 names one.
//!
//! Anyonic tensors admit no ordinary `contract` (symtn refuses it, as
//! TensorKit's `blas_contract!` does), so every contraction is TensorKit's
//! `planarcontract!` lowering by hand: transpose both operands so the
//! contracted legs meet at the domain/codomain boundary, then `compose`.
//! This module does that rotation arithmetic once; delete it when #2284
//! lands.
//!
//! Evidence for #2284 collected while writing the golden chain:
//! - Each `@plansor` line of MPSKit becomes 2 transposes + `compose` per
//!   pairwise contraction plus a final transpose; the left environment term is
//!   two contractions and a repartition, the right one two contractions and a
//!   transpose.
//! - The axes are flat source numbers whose meaning depends on the planar
//!   order (codomain, then domain reversed), so every call site needed its
//!   cyclic order derived by hand; a valid-but-wrong rotation is not an error.
//! - Dual flags flip on every bend; whether a result lands on the intended
//!   space is only observable at the next `compose`/`axpby`.

use symtn::typed::Error;

use crate::model::Tensor;

/// Planar order of `t`'s legs: codomain, then domain reversed.
fn cyclic(t: &Tensor) -> Vec<usize> {
    let cod = t.codomain_rank();
    (0..cod).chain((cod..t.rank()).rev()).collect()
}

/// The legs following the contiguous cyclic run `run` of `order`, in order.
fn after_run(order: &[usize], run: &[usize]) -> Result<Vec<usize>, Error> {
    let n = order.len();
    let start = run
        .first()
        .and_then(|first| order.iter().position(|x| x == first))
        .ok_or_else(|| Error::InvalidArgument(format!("empty or unknown cut {run:?}")))?;
    if run.len() > n
        || run
            .iter()
            .enumerate()
            .any(|(k, &x)| order[(start + k) % n] != x)
    {
        return Err(Error::InvalidArgument(format!(
            "cut {run:?} is not a contiguous run of the planar order {order:?}"
        )));
    }
    Ok((run.len()..n).map(|k| order[(start + k) % n]).collect())
}

/// Contracts `a_cut[k]` of `a` with `b_cut[k]` of `b`.
///
/// `a_cut` must be a contiguous run of `a`'s planar order and `reverse(b_cut)`
/// one of `b`'s (planar pairing reverses orientation); anything else would
/// need a braid and is rejected. The result's planar order is `a`'s open legs
/// (cyclically after `a_cut`) followed by `b`'s (cyclically after
/// `b_cut[0]`): codomain = `a`'s open legs, domain = `b`'s reversed.
pub fn contract(a: &Tensor, a_cut: &[usize], b: &Tensor, b_cut: &[usize]) -> Result<Tensor, Error> {
    if a_cut.len() != b_cut.len() {
        return Err(Error::InvalidArgument(format!(
            "cut lengths differ: {a_cut:?} vs {b_cut:?}"
        )));
    }
    let a_open = after_run(&cyclic(a), a_cut)?;
    let b_rev: Vec<usize> = b_cut.iter().rev().copied().collect();
    let b_open = after_run(&cyclic(b), &b_rev)?;
    let a_dom: Vec<usize> = a_cut.iter().rev().copied().collect();
    let b_dom: Vec<usize> = b_open.iter().rev().copied().collect();
    a.transpose(&a_open, &a_dom)?
        .compose(&b.transpose(&b_rev, &b_dom)?)
}

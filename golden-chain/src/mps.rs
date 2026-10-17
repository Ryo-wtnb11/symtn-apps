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

use symtn::typed::{Error, Runtime, TensorMap};

use crate::ed::Fib;
use crate::model::{ONE, Space, Tensor, ZERO, label};
use crate::planar;

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
    let z = planar::contract(h, &[2], a, &[1])?; // [a,b,d] <- [l,i]
    let y = planar::contract(&a.adjoint()?, &[2, 1], &z, &[0, 3])?; // [i'] <- [i,d,b]
    y.repartition(2)
}

/// `ER̃ ∈ [P, V_{k-1}*] <- [P, V_{k-1}*]`: bond `(k-1, k)` with the rotated
/// right tensor `bt = B̃_k ∈ [P, V_k*] <- [V_{k-1}*]` contracted.
fn right_bond(h: &Tensor, bt: &Tensor) -> Result<Tensor, Error> {
    // bt: q=0 y=1 | x=2.
    let z = planar::contract(h, &[3], bt, &[0])?; // [c,a,b] <- [x,y]
    let y = planar::contract(&bt.adjoint()?, &[2, 1], &z, &[4, 2])?; // [x'] <- [a,c,x]
    y.transpose(&[1, 0], &[2, 3])
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

/// Two-site effective Hamiltonian `θ ↦ OL∘θ + θ∘rot(OR̃) + h_{(p1,p2)} θ`.
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
        // θ: l=0 p1=1 | r=2 p2=3. h's domain (c, d) meets (p1, p2).
        let bond = planar::contract(&self.h, &[3, 2], theta, &[3, 1])? // [a,b] <- [l,r]
            .transpose(&[2, 0], &[3, 1])?;
        self.ol
            .compose(theta)?
            .axpby(ONE, &theta.compose(&self.or_dom)?, ONE)?
            .axpby(ONE, &bond, ONE)
    }

    /// `out ← H_eff θ`, the shape of a `symtn::krylov::LinearMap` closure:
    /// `|x, y| heff.apply_into(x, y)`.
    pub fn apply_into(&self, theta: &Tensor, out: &mut Tensor) -> Result<(), Error> {
        self.apply(theta)?.axpby_into(out, ONE, ZERO)
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
/// `S = -Σ_s d_s Σ_k λ²_{s,k} ln λ²_{s,k}`, with values `λ < ε` dropped and
/// the spectrum normalized as `Σ_s d_s Σ_k λ²_{s,k} = 1`.
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

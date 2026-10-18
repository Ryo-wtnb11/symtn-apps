//! Two-site DMRG on the open golden chain, MPSKit 0.13.13's `DMRG2`
//! (`src/algorithms/groundstate/dmrg.jl:find_groundstate!(…, ::DMRG2)`).
//!
//! Per bond: Lanczos on [`Heff`] (`fixedpoint(Hac2, ac2, :SR, …)`), then
//! `svd_compact` + `find_truncated` + `restrict_leg` (`svd_trunc!`), then
//! `normalize!(c)`. The sweep is `pos = 1..L-1` left to right, then
//! `L-2..1` right to left; after each split one environment side is updated.
//!
//! Deviations from MPSKit, both gauge-only:
//! - the initial state is the canonical right isometry on every site plus a
//!   seeded random first `θ`, not a random MPS (no QR needed);
//! - the first sweep's eigensolver tolerance uses `ϵ = 1` in place of the
//!   Galerkin error of the initial state. For a random or isometric start that
//!   error is O(1) as well, and either way the clamp gives `tol_max`.

use std::sync::Arc;
use std::time::Instant;

use num_complex::Complex64;
use symtn::krylov::{self, Lanczos, Which, Workspace};
use symtn::sector::TableRule;
use symtn::typed::{Error, Runtime, Svd, TensorMap, Truncation};

use crate::ed::{Fib, quantum_dim};
use crate::model::{Tensor, bond_term, label, physical, virtual_spaces};
use crate::mps::{Heff, left_step, right_isometry, right_step};

/// Every numerical constant of a run; there are no defaults.
#[derive(Clone, Debug)]
pub struct Dmrg2 {
    /// Bond dimension (total quantum dimension, `Truncation::Rank`) per
    /// sweep; the last value repeats. The first one also sizes the initial
    /// virtual spaces. Convergence is only tested from the sweep that uses
    /// the last value on, so a run never stops short of the requested D.
    pub schedule: Vec<usize>,
    /// Stop once the largest per-bond `ϵ` of a sweep is at or below this
    /// (MPSKit `DMRG2.tol`).
    pub tol: f64,
    /// Maximal number of sweeps (MPSKit `DMRG2.maxiter`); at least
    /// `schedule.len()`.
    pub maxiter: usize,
    /// Lanczos Krylov dimension (MPSKit `Defaults.krylovdim`).
    pub krylovdim: usize,
    /// Lanczos restarts (MPSKit `Defaults.maxiter` for `alg_eigsolve`).
    pub eig_maxiter: usize,
    /// Lanczos convergence check after every Krylov expansion instead of
    /// only at full `krylovdim` (MPSKit `Defaults.alg_eigsolve` `eager`).
    pub eig_eager: bool,
    /// `updatetol` clamp and factor: `clamp(ϵ·factor/√iter, min, max)`
    /// (MPSKit `Defaults.{tol_min, tol_max, eigs_tolfactor}`).
    pub eig_tol_min: f64,
    pub eig_tol_max: f64,
    pub eig_tol_factor: f64,
    /// Seed of the first two-site centre.
    pub seed: u64,
}

/// One sweep (left to right and back).
#[derive(Clone, Debug)]
pub struct SweepLog {
    /// Bond dimension (`Truncation::Rank`) of this sweep.
    pub bond_dimension: usize,
    /// Energy of the truncated state at the end of the sweep.
    pub energy: f64,
    /// Change of `energy` against the previous sweep (`NaN` for the first).
    pub delta_energy: f64,
    /// Largest truncation error, the norm of the discarded part
    /// `sqrt(Σ_c d_c Σ_discarded σ²)` of the normalized two-site eigenvector
    /// (MatrixAlgebraKit `truncation_error`).
    pub max_truncation_error: f64,
    /// Largest `ϵ_bond = |1 - |⟨θ_old, u s vh⟩||` (MPSKit `v`).
    pub epsilon: f64,
    /// Bond eigensolves whose residual did not reach the tolerance
    /// (MPSKit only warns, so this is reported, not an error).
    pub unconverged: usize,
    /// `H_eff` applications of all bond eigensolves (krylov `numops`).
    pub matvecs: usize,
    /// Wall time of the sweep, and its parts: the bond eigensolves, the
    /// truncated SVDs with normalization and `ϵ_bond`, and the environment
    /// steps with the next centre and `H_eff` construction. The rest (the
    /// energy at the end of the sweep) is in `seconds` only.
    pub seconds: f64,
    pub eigsolve_seconds: f64,
    pub svd_seconds: f64,
    pub environment_seconds: f64,
    /// The part of `eigsolve_seconds` spent applying `H_eff`; the rest is
    /// the Lanczos vector work (orthogonalization, workspace, Ritz vector).
    pub apply_seconds: f64,
}

/// A Schmidt spectrum per sector, `(x, λ_x)`, normalized as
/// `Σ_x d_x Σ_k λ²_{x,k} = 1`.
pub type Spectrum = Vec<(Fib, Vec<f64>)>;

#[derive(Clone, Debug)]
pub struct GroundState {
    pub energy: f64,
    /// Wall time from the start of the call to the first sweep: virtual
    /// spaces, initial sites, all right environments and the first centre.
    pub setup_seconds: f64,
    pub converged: bool,
    pub sweeps: Vec<SweepLog>,
    /// Entanglement spectra of bonds `1..L-1` (MPSKit
    /// `entanglement_spectrum(ψ, i)`).
    pub spectra: Vec<Spectrum>,
}

impl GroundState {
    /// MPSKit `entropy(ψ, i)` for bonds `1..L-1`.
    pub fn entropies(&self) -> Vec<f64> {
        self.spectra
            .iter()
            .map(|s| crate::mps::entropy(s.iter().map(|(x, v)| (*x, v.as_slice()))))
            .collect()
    }
}

/// Ground state of `H = -J Σ P^(1)` on `len` anyons with `x_L = total`.
/// Every tensor is created on `rt`, so its CPU pool sets the thread count.
pub fn find_groundstate(
    rt: &Runtime,
    len: usize,
    j: f64,
    total: Fib,
    alg: &Dmrg2,
) -> Result<GroundState, Error> {
    let start = Instant::now();
    if len < 3 || alg.schedule.is_empty() || alg.maxiter < alg.schedule.len() {
        return Err(Error::InvalidArgument(
            "DMRG2 needs L >= 3, a nonempty schedule and maxiter >= schedule.len()".into(),
        ));
    }
    let rule = Arc::new(TableRule::fibonacci());
    let p = physical(&rule)?;
    let h = bond_term(rt, &p, j)?;
    let id_p = TensorMap::isomorphism(rt, [&p], [&p])?;
    // MPSKit's hard-hexagon `virtual_space`: the bond dimension split evenly
    // over the sectors, so the total quantum dimension is about `D`.
    let share = alg.schedule[0] as f64 / (quantum_dim(Fib::One) + quantum_dim(Fib::Tau));
    let v = virtual_spaces(&rule, len, total, share.round() as usize)?;

    // Site k (1-based) at index k-1. Left of the centre: AL form; right: B.
    let mut sites: Vec<Tensor> = (1..=len)
        .map(|k| right_isometry(rt, &v[k - 1], &p, &v[k]))
        .collect::<Result<_, _>>()?;
    // ol[i] = OL_{i+1}; or[i] = OR̃_i (OR̃_L = 0 on [P, V_L*]).
    let mut ol = vec![TensorMap::zeros(rt, [&v[0], &p], [&v[0], &p])?];
    let vl_dual = v[len].try_dual()?;
    let mut or = vec![TensorMap::zeros(rt, [&p, &vl_dual], [&p, &vl_dual])?; len + 1];
    for k in (2..=len).rev() {
        or[k - 1] = right_step(&or[k], &sites[k - 1], &h, &id_p)?;
    }

    let mut theta = TensorMap::rand_with_seed(rt, [&v[0], &p], [&v[2], &p.try_dual()?], alg.seed)?;
    let setup_seconds = start.elapsed().as_secs_f64();
    let mut epsilon = 1.0;
    let mut sweeps: Vec<SweepLog> = Vec::new();
    let mut converged = false;
    let mut last = None;
    for iter in 1..=alg.maxiter {
        let sweep_start = Instant::now();
        let rank = alg.schedule[(iter - 1).min(alg.schedule.len() - 1)];
        let eig = Lanczos::new(
            alg.krylovdim,
            alg.eig_maxiter,
            (epsilon * alg.eig_tol_factor / (iter as f64).sqrt())
                .clamp(alg.eig_tol_min, alg.eig_tol_max),
        )
        .eager(alg.eig_eager);
        let mut log = SweepLog {
            bond_dimension: rank,
            energy: f64::NAN,
            delta_energy: f64::NAN,
            max_truncation_error: 0.0,
            epsilon: 0.0,
            unconverged: 0,
            matvecs: 0,
            seconds: 0.0,
            eigsolve_seconds: 0.0,
            svd_seconds: 0.0,
            environment_seconds: 0.0,
            apply_seconds: 0.0,
        };
        let forward = (1..len).map(|pos| (pos, true));
        let backward = (1..len - 1).rev().map(|pos| (pos, false));
        for (pos, right) in forward.chain(backward) {
            let clock = Instant::now();
            let heff = Heff::new(ol[pos - 1].clone(), &or[pos + 1], h.clone())?;
            log.environment_seconds += clock.elapsed().as_secs_f64();
            let bond = update_bond(heff, &theta, rank, &eig)?;
            log.max_truncation_error = log.max_truncation_error.max(bond.truncation_error);
            log.epsilon = log.epsilon.max(bond.epsilon);
            log.unconverged += usize::from(!bond.converged);
            log.matvecs += bond.matvecs;
            log.eigsolve_seconds += bond.eigsolve_seconds;
            log.svd_seconds += bond.svd_seconds;
            log.apply_seconds += bond.heff.apply_seconds();
            let clock = Instant::now();
            let Svd { u, s, vh } = bond.factors;
            if right {
                ol.truncate(pos);
                ol.push(left_step(&ol[pos - 1], &u, &h, &id_p)?);
                sites[pos - 1] = u.clone();
                sites[pos] = vh.clone();
                theta = if pos + 1 < len {
                    s.compose(&vh)?.repartition(2)?.compose(&sites[pos + 1])?
                } else {
                    // Turn: the right-to-left pass starts at L-2.
                    or[pos] = right_step(&or[pos + 1], &vh, &h, &id_p)?;
                    sites[pos - 2].compose(&u.compose(&s)?.repartition(1)?)?
                };
            } else {
                or[pos] = right_step(&or[pos + 1], &vh, &h, &id_p)?;
                sites[pos] = vh.clone();
                theta = if pos > 1 {
                    sites[pos - 2].compose(&u.compose(&s)?.repartition(1)?)?
                } else {
                    u.compose(&s)?.compose(&vh)?
                };
            }
            log.environment_seconds += clock.elapsed().as_secs_f64();
            if !right && pos == 1 {
                let energy = energy(&bond.heff, &theta)?;
                log.delta_energy = energy - sweeps.last().map_or(f64::NAN, |l| l.energy);
                log.energy = energy;
                last = Some(s);
            }
        }
        epsilon = log.epsilon;
        log.seconds = sweep_start.elapsed().as_secs_f64();
        sweeps.push(log);
        if iter >= alg.schedule.len() && epsilon <= alg.tol {
            converged = true;
            break;
        }
    }
    let s = last.expect("maxiter >= schedule.len() >= 1 runs a sweep");
    Ok(GroundState {
        energy: sweeps.last().map_or(f64::NAN, |l| l.energy),
        setup_seconds,
        converged,
        sweeps,
        spectra: spectra(s, &sites[1..len - 1])?,
    })
}

struct BondUpdate {
    heff: Heff,
    factors: Svd<Tensor>,
    truncation_error: f64,
    epsilon: f64,
    converged: bool,
    matvecs: usize,
    eigsolve_seconds: f64,
    svd_seconds: f64,
}

/// Lanczos, truncated SVD and normalization at one bond.
fn update_bond(
    mut heff: Heff,
    theta: &Tensor,
    rank: usize,
    alg: &Lanczos,
) -> Result<BondUpdate, Error> {
    let clock = Instant::now();
    let mut ws = Workspace::with_capacity(theta, alg.get_krylovdim())?;
    let (_, mut vectors, info) =
        krylov::eigsolve(&mut heff, theta, 1, Which::SmallestReal, alg, &mut ws)
            .map_err(krylov_error)?;
    let eigsolve_seconds = clock.elapsed().as_secs_f64();
    let clock = Instant::now();
    let new = vectors.swap_remove(0);
    let Svd { u, s, vh } = new.svd_compact(&[0, 1], &[2, 3])?;
    let found = s.domain()[0].find_truncated(&s.diagview()?, &Truncation::Rank(rank))?;
    let u = u.restrict_leg(&[(u.codomain_rank(), &found.selection)])?;
    let s = s.restrict_leg(&[(0, &found.selection), (1, &found.selection)])?;
    let vh = vh.restrict_leg(&[(0, &found.selection)])?;
    let s = s.scale(Complex64::new(1.0 / s.norm(2.0)?, 0.0))?;
    let overlap = theta.inner(&u.compose(&s)?.compose(&vh)?)?.norm() / theta.norm(2.0)?;
    Ok(BondUpdate {
        svd_seconds: clock.elapsed().as_secs_f64(),
        eigsolve_seconds,
        matvecs: info.numops,
        heff,
        factors: Svd { u, s, vh },
        truncation_error: found.error,
        epsilon: (1.0 - overlap).abs(),
        converged: info.converged >= 1,
    })
}

/// `⟨θ, H_eff θ⟩ / ⟨θ, θ⟩`, the energy of the whole state with centre `θ`.
fn energy(heff: &Heff, theta: &Tensor) -> Result<f64, Error> {
    Ok((theta.inner(&heff.apply(theta)?)? / theta.inner(theta)?).re)
}

/// Schmidt spectra of bonds `1..L-1` of `AL_1 ∘ s ∘ B_2 ∘ … ∘ B_L`, by moving
/// the centre right with untruncated SVDs; `b` is `B_2..B_{L-1}`. MPSKit reads
/// the same values from its gauge's `C`.
fn spectra(s: Tensor, b: &[Tensor]) -> Result<Vec<Spectrum>, Error> {
    let mut out = vec![spectrum(&s)?];
    let mut centre = s;
    for bk in b {
        let m = centre.compose(bk)?.repartition(2)?; // [W, P] <- [V_k]
        let Svd { s, vh, .. } = m.svd_compact(&[0, 1], &[2])?;
        out.push(spectrum(&s)?);
        centre = s.compose(&vh)?;
    }
    Ok(out)
}

fn spectrum(s: &Tensor) -> Result<Spectrum, Error> {
    Ok(s.diagview()?
        .into_iter()
        .map(|entry| {
            (
                label(&entry.sector),
                entry.values.iter().map(|x| x.re).collect(),
            )
        })
        .collect())
}

fn krylov_error(error: krylov::Error<Error>) -> Error {
    match error {
        krylov::Error::Vector(e) => e,
        other => Error::InvalidArgument(format!("bond eigensolve: {other}")),
    }
}

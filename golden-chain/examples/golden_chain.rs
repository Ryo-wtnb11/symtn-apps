//! Ground state of the open golden chain: per-sweep log, then a CSV of the
//! entanglement profile.
//!
//! `cargo run --release --example golden_chain > profile.csv`

use golden_chain::dmrg::{Dmrg2, find_groundstate};
use golden_chain::ed::Fib;

/// Chain length; a multiple of 6 suits both couplings (see T9).
const LEN: usize = 96;
/// `+1`: tricritical Ising (c = 7/10); `-1`: 3-state Potts (c = 4/5).
const J: f64 = 1.0;
const TOTAL: Fib = Fib::One;

fn main() -> Result<(), symtn::typed::Error> {
    let alg = Dmrg2 {
        // Ramp: cheap sweeps first, then the target D.
        schedule: vec![32, 64, 128],
        // MPSKit DMRG2 defaults, except the eigensolver ceiling (1e-12, not
        // 1e-4), which otherwise stops the run on unchanged start vectors.
        tol: 1e-10,
        maxiter: 50,
        krylovdim: 30,
        eig_maxiter: 200,
        eig_tol_min: 1e-14,
        eig_tol_max: 1e-12,
        eig_tol_factor: 1e-3,
        seed: 7,
    };
    let gs = find_groundstate(LEN, J, TOTAL, &alg)?;
    for (k, sweep) in gs.sweeps.iter().enumerate() {
        println!(
            "# sweep {}: E = {:.12}, dE = {:.2e}, max discarded = {:.2e}, eps = {:.2e}",
            k + 1,
            sweep.energy,
            sweep.delta_energy,
            sweep.max_truncation_error,
            sweep.epsilon
        );
    }
    println!("# E = {:.12}, converged = {}", gs.energy, gs.converged);
    println!("l,S");
    for (l, s) in gs.entropies().iter().enumerate() {
        println!("{},{s:.12}", l + 1);
    }
    Ok(())
}

//! Data for the README figures, as CSV files in `golden-chain/out/`
//! (gitignored); `scripts/plot.py` draws them.
//!
//! `cargo run --release --example golden_chain`
//!
//! - `profile.csv`: `J,l,S` for the open chain of length [`PROFILE_LEN`];
//! - `sweeps.csv`: `J,sweep,E,dE,discarded,epsilon` of the same runs;
//! - `energies.csv`: `L,J,total,E_dmrg,E_ed` against the sparse-Lanczos ED.

// The ED oracle and the MPSKit-like settings are the tests' own, reused
// rather than copied.
#[path = "../tests/common/mod.rs"]
mod common;

use std::fmt::Write as _;
use std::path::Path;

use golden_chain::dmrg::find_groundstate;
use golden_chain::ed::{Boundary, Chain, Fib};

/// A multiple of 6, as T9 needs for `J = -1` (period-3 Potts structure).
const PROFILE_LEN: usize = 96;
/// Ramp: cheap sweeps first, then the target D.
const PROFILE_SCHEDULE: [usize; 3] = [32, 64, 128];
/// Lengths for the ED comparison (L = 24 has 46368 paths) and their bond
/// dimension, which keeps the whole centre bond up to L = 20.
const ED_LENS: [usize; 4] = [12, 16, 20, 24];
const ED_D: usize = 128;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&out)?;

    let (mut profile, mut sweeps) = (
        String::from("J,l,S\n"),
        String::from("J,sweep,E,dE,discarded,epsilon\n"),
    );
    for j in [1.0, -1.0] {
        let alg = common::mpskit_dmrg2(PROFILE_SCHEDULE.to_vec());
        let gs = find_groundstate(PROFILE_LEN, j, Fib::One, &alg)?;
        for (l, s) in gs.entropies().iter().enumerate() {
            writeln!(profile, "{j},{},{s:.15e}", l + 1)?;
        }
        for (k, s) in gs.sweeps.iter().enumerate() {
            writeln!(
                sweeps,
                "{j},{},{:.15e},{:.6e},{:.6e},{:.6e}",
                k + 1,
                s.energy,
                s.delta_energy,
                s.max_truncation_error,
                s.epsilon
            )?;
        }
        eprintln!("L {PROFILE_LEN} J {j}: E = {}", gs.energy);
    }
    std::fs::write(out.join("profile.csv"), profile)?;
    std::fs::write(out.join("sweeps.csv"), sweeps)?;

    let mut energies = String::from("L,J,total,E_dmrg,E_ed\n");
    for len in ED_LENS {
        for j in [1.0, -1.0] {
            for (total, name) in [(Fib::One, "1"), (Fib::Tau, "tau")] {
                let alg = common::mpskit_dmrg2(vec![ED_D]);
                let gs = find_groundstate(len, j, total, &alg)?;
                let ed = common::lanczos_ground_energy(&Chain::new(len, Boundary::Open(total)), j);
                writeln!(energies, "{len},{j},{name},{:.15e},{ed:.15e}", gs.energy)?;
            }
        }
        eprintln!("L {len} done");
    }
    std::fs::write(out.join("energies.csv"), energies)?;
    Ok(())
}

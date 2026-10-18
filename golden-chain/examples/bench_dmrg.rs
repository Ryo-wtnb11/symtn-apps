//! DMRG2 timings for the comparison with MPSKit (`bench/mpskit/bench.jl`
//! writes the same rows); `bench/run.sh` drives both and adds the header.
//!
//! ```sh
//! cargo run --release --example bench_dmrg -- cold|first|warm <threads> <L> <D> <J>
//! ```
//!
//! Rows (CSV, no header): `side,threads,kind,L,D,J,rep,sweep,seconds,
//! eigsolve_s,svd_s,env_s,apply_s,matvecs,energy,epsilon,sweeps,converged,
//! reported` (the thread settings in effect), where
//! `kind` is
//! - `cold`: setup plus the first sweep of the first run in this process
//!   (Rust has no JIT, so a fresh process is the compiled steady code);
//! - `first_after_compile`: the same for the target case after a one-sweep
//!   run of a tiny different case (L = 6, D = 8, same J) in the process;
//! - `full`: a warm run to convergence at fixed D, wall time of the call;
//! - `sweep`: one sweep of a warm run of exactly [`FIXED_SWEEPS`] sweeps
//!   (sweep 1 includes the setup).

#[path = "../tests/common/mod.rs"]
mod common;

use std::ffi::c_int;
use std::sync::OnceLock;
use std::time::Instant;

use golden_chain::dmrg::{Dmrg2, SweepLog, find_groundstate};
use golden_chain::ed::Fib;
use symtn::typed::Runtime;

/// Sweep count of the per-sweep runs, equal on both sides.
const FIXED_SWEEPS: usize = 4;
/// Warm repetitions of the full and the fixed-sweep runs; a first full run
/// beyond [`LONG_RUN`] seconds selects the `_LONG` counts (each fixed-sweep
/// run already gives `FIXED_SWEEPS` samples).
const FULL_REPEATS: [usize; 2] = [5, 3];
const FIXED_REPEATS: [usize; 2] = [3, 1];
const LONG_RUN: f64 = 20.0;
/// The warm-up case of `first`, as on the MPSKit side.
const WARMUP_LEN: usize = 6;
const WARMUP_D: usize = 8;

unsafe extern "C" {
    /// OpenBLAS's own thread count (libopenblas, linked by openblas-src).
    fn openblas_get_num_threads() -> c_int;
}

/// The thread settings in effect, for the `reported` column. symtn's
/// `Runtime` has no accessor for its pool sizes, so those are the requested
/// values; OpenBLAS reports its own.
static REPORTED: OnceLock<String> = OnceLock::new();

fn reported() -> &'static str {
    REPORTED.get().map_or("", String::as_str)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: bench_dmrg cold|first|warm <threads> <L> <D> <J>";
    let [threads, len, d, j] = [1, 2, 3, 4].map(|k| args.get(k).ok_or(usage));
    let (threads, len, d, j): (usize, usize, usize, f64) =
        (threads?.parse()?, len?.parse()?, d?.parse()?, j?.parse()?);
    // Every thread knob symtn has: the runtime CPU pool (dense kernels,
    // tree-transform replay, strided kernels) and the recoupling workers.
    // BLAS threads are OpenBLAS's own (`OPENBLAS_NUM_THREADS`, 1 in run.sh).
    let rt = Runtime::builder()
        .dense_threads(threads)
        .recoupling_threads(threads)
        .build()?;
    // SAFETY: a getter without arguments; libopenblas is initialized when
    // loaded, and the call only reads its thread count.
    let blas = unsafe { openblas_get_num_threads() };
    REPORTED.get_or_init(|| {
        format!("dense_threads={threads} (requested); recoupling_threads={threads} (requested); openblas_get_num_threads={blas}")
    });
    match args[0].as_str() {
        "cold" | "first" => {
            let one_sweep = |d| Dmrg2 {
                maxiter: 1,
                ..common::mpskit_dmrg2(vec![d])
            };
            let kind = if args[0] == "first" {
                // A different, tiny case first: the target's spaces are new.
                find_groundstate(&rt, WARMUP_LEN, j, Fib::One, &one_sweep(WARMUP_D))?;
                "first_after_compile"
            } else {
                "cold"
            };
            let gs = find_groundstate(&rt, len, j, Fib::One, &one_sweep(d))?;
            let s = &gs.sweeps[0];
            row(
                threads,
                kind,
                len,
                d,
                j,
                0,
                1,
                gs.setup_seconds + s.seconds,
                s,
            );
        }
        "warm" => warm(&rt, threads, len, d, j)?,
        _ => return Err(usage.into()),
    }
    Ok(())
}

fn warm(
    rt: &Runtime,
    threads: usize,
    len: usize,
    d: usize,
    j: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let full = common::mpskit_dmrg2(vec![d]);
    let fixed = Dmrg2 {
        // Never met, so every run sweeps exactly `maxiter` times.
        tol: -1.0,
        maxiter: FIXED_SWEEPS,
        ..full.clone()
    };
    // Warm-up: one sweep fills symtn's structure caches for these spaces.
    find_groundstate(
        rt,
        len,
        j,
        Fib::One,
        &Dmrg2 {
            maxiter: 1,
            ..full.clone()
        },
    )?;
    let mut long = 0;
    let mut rep = 1;
    while rep <= FULL_REPEATS[long] {
        let clock = Instant::now();
        let gs = find_groundstate(rt, len, j, Fib::One, &full)?;
        let seconds = clock.elapsed().as_secs_f64();
        if rep == 1 && seconds > LONG_RUN {
            long = 1;
        }
        println!(
            "symtn,{threads},full,{len},{d},{j},{rep},,{seconds:.6e},,,,,{},{:.15e},{:.3e},{},{},{}",
            gs.sweeps.iter().map(|s| s.matvecs).sum::<usize>(),
            gs.energy,
            gs.sweeps.last().map_or(f64::NAN, |s| s.epsilon),
            gs.sweeps.len(),
            gs.converged,
            reported()
        );
        rep += 1;
    }
    for rep in 1..=FIXED_REPEATS[long] {
        let gs = find_groundstate(rt, len, j, Fib::One, &fixed)?;
        for (k, s) in gs.sweeps.iter().enumerate() {
            // Sweep 1 includes the setup, as MPSKit's `finalize` stamps do.
            let seconds = s.seconds + if k == 0 { gs.setup_seconds } else { 0.0 };
            row(threads, "sweep", len, d, j, rep, k + 1, seconds, s);
        }
    }
    eprintln!("L {len} D {d} J {j} done");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn row(
    threads: usize,
    kind: &str,
    len: usize,
    d: usize,
    j: f64,
    rep: usize,
    sweep: usize,
    seconds: f64,
    s: &SweepLog,
) {
    println!(
        "symtn,{threads},{kind},{len},{d},{j},{rep},{sweep},{seconds:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{},{:.15e},{:.3e},,,{}",
        s.eigsolve_seconds,
        s.svd_seconds,
        s.environment_seconds,
        s.apply_seconds,
        s.matvecs,
        s.energy,
        s.epsilon,
        reported()
    );
}

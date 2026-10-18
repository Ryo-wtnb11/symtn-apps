#!/bin/sh
# DMRG2 timings of symtn (examples/bench_dmrg.rs) and MPSKit
# (bench/mpskit/bench.jl), one side at a time, into
# golden-chain/bench/results/dmrg2_t<threads>.csv.
#
#     golden-chain/bench/run.sh <threads> [L...]
#
# Before every measurement it waits until no other cargo, rustc or julia
# process runs and the 1-minute load average is below 4, so neither side
# competes with other work or with the other side.
set -eu
threads=${1:?usage: run.sh <threads> [L...]}
shift
lens=${*:-"24 48 96"}
app=$(cd "$(dirname "$0")/.." && pwd)
out="$app/bench/results/dmrg2_t$threads.csv"
mkdir -p "$app/bench/results"

export OPENBLAS_NUM_THREADS="$threads" RAYON_NUM_THREADS="$threads"
cargo build --release --locked --example bench_dmrg --manifest-path "$app/Cargo.toml"
julia --project="$app/bench/mpskit" -e 'using Pkg; Pkg.instantiate()'

wait_idle() {
    while pgrep -x cargo >/dev/null || pgrep -x rustc >/dev/null || pgrep -x julia >/dev/null ||
        [ "$(sysctl -n vm.loadavg | awk '{print ($2 < 4) ? 0 : 1}')" -ne 0 ]; do
        sleep 120
    done
}
rust() { cargo run -q --release --locked --example bench_dmrg --manifest-path "$app/Cargo.toml" -- "$@"; }
mpskit() { julia --project="$app/bench/mpskit" -t "$threads" "$app/bench/mpskit/bench.jl" "$@"; }

{
    echo "# DMRG2 on the open golden chain, H = -J sum P^(1), total charge 1; golden-chain/bench/run.sh $threads $lens"
    echo "# machine: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.physicalcpu) cores ($(sysctl -n hw.perflevel0.physicalcpu) performance + $(sysctl -n hw.perflevel1.physicalcpu) efficiency), $(sysctl -n hw.memsize | awk '{print $1 / 2^30}') GiB, macOS $(sw_vers -productVersion)"
    echo "# symtn: $(rustc --version), symtn-apps $(git -C "$app" rev-parse --short HEAD), symtn $(grep -o 'rev = "[0-9a-f]*"' "$app/../Cargo.toml" | cut -c8-14), OpenBLAS 0.3.32 (openblas-src 0.10.16, built from source) for GEMM and LAPACK, Runtime::dense_threads($threads), OPENBLAS_NUM_THREADS=$threads"
    echo "# MPSKit: julia $(julia --version | awk '{print $3}') -t $threads, MPSKit 0.13.13, TensorKit 0.17.1, TensorKitSectors 0.3.10, KrylovKit $(grep -A6 '^\[\[deps.KrylovKit\]\]' "$app/bench/mpskit/Manifest.toml" | awk -F'"' '/^version/{print $2}'), $(julia -e 'using LinearAlgebra; print(BLAS.get_config().loaded_libs[1].libname |> basename)') (OpenBLAS_jll 0.3.27), BLAS.set_num_threads($threads)"
    echo "# settings (both sides): D = truncrank(D) on the total quantum dimension; tol 1e-10, maxiter 50; Lanczos krylovdim 30, maxiter 200, eager; updatetol clamp(eps*1e-3/sqrt(sweep), 1e-14, 1e-12); initial virtual spaces D/(1+phi) per sector capped at the reachable space"
    echo "# initial state: symtn right-isometric sites + seeded random first centre; MPSKit seeded random FiniteMPS (Random.seed!(1234))"
    echo "# kinds: cold = setup + first sweep of the first run in a fresh process; full = warm run to convergence; sweep = one sweep of a warm run of exactly 4 sweeps (sweep 1 includes setup); profile = MPSKit's DMRG2 loop copied with timers (split and matvecs only)"
    echo "side,threads,kind,L,D,J,rep,sweep,seconds,eigsolve_s,svd_s,env_s,apply_s,matvecs,energy,epsilon,sweeps,converged"
} >"$out"

for len in $lens; do
    for d in 32 64 128 256; do
        for j in 1 -1; do
            wait_idle
            rust cold "$threads" "$len" "$d" "$j" >>"$out"
            wait_idle
            mpskit cold "$threads" "$len" "$d" "$j" >>"$out"
        done
    done
done
for len in $lens; do
    wait_idle
    rust warm "$threads" "$len" >>"$out"
    wait_idle
    mpskit warm "$threads" "$len" >>"$out"
done

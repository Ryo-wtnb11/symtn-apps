#!/bin/sh
# DMRG2 timings of symtn (examples/bench_dmrg.rs) and MPSKit
# (bench/mpskit/bench.jl), one side and one case at a time; non-interactive,
# so it runs as a batch job as well as on a workstation.
#
#     golden-chain/bench/run.sh <mode> <out dir> <threads> "<L...>" "<D...>" ["<J...>"]
#
# mode:
# - first-run: per case, each in a fresh process, setup + first sweep:
#   cold (Julia including JIT; Rust has no JIT), cold_sysimage (Julia with
#   the image of bench/build_sysimage.sh, when BENCH_IMAGE names it), and
#   first_after_compile (after a tiny different case in the same process);
# - warm: per case, warm full runs to convergence and fixed four-sweep runs.
#
# Writes <out dir>/<mode>_t<threads>_L<L...>_D<D...>.csv with a header that
# records machine, versions, thread settings and the image build time.
# Environment: JULIA (default `julia`), BENCH_IMAGE (optional sysimage),
# BENCH_WAIT_IDLE=1 to wait before every measurement until no other cargo,
# rustc or julia process runs and the 1-minute load is below 4 (for a shared
# workstation; a batch node is exclusive).
set -eu
usage='usage: run.sh first-run|warm <out dir> <threads> "<L...>" "<D...>" ["<J...>"]'
mode=${1:?$usage}
outdir=${2:?$usage}
threads=${3:?$usage}
lens=${4:?$usage}
ds=${5:?$usage}
js=${6:-"1 -1"}
julia=${JULIA:-julia}
app=$(cd "$(dirname "$0")/.." && pwd)
case $mode in first-run | warm) ;; *) echo "$usage" >&2; exit 2 ;; esac
mkdir -p "$outdir"
out="$outdir/${mode}_t${threads}_L$(echo $lens | tr ' ' -)_D$(echo $ds | tr ' ' -).csv"

# One BLAS pool per side of `threads` workers; nothing else spawns threads.
export OPENBLAS_NUM_THREADS="$threads" OMP_NUM_THREADS="$threads" RAYON_NUM_THREADS="$threads"
cargo build -q --release --locked --example bench_dmrg --manifest-path "$app/Cargo.toml"
"$julia" --project="$app/bench/mpskit" -e 'using Pkg; Pkg.instantiate()'

wait_idle() {
    [ "${BENCH_WAIT_IDLE:-0}" = 1 ] || return 0
    while pgrep -x cargo >/dev/null || pgrep -x rustc >/dev/null || pgrep -x julia >/dev/null ||
        [ "$(uptime | awk -F'load averages?: ' '{split($2, a, /[ ,]+/); print (a[1] < 4) ? 0 : 1}')" -ne 0 ]; do
        sleep 120
    done
}
target=${CARGO_TARGET_DIR:-$app/../target}
openblas_dir=$(ls -d "$target"/release/build/openblas-src-*/out/OpenBLAS-* 2>/dev/null | head -1)
export LD_LIBRARY_PATH="$openblas_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="$openblas_dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
# Runs one benchmark process under GNU time and appends its CPU utilisation
# and peak RSS to every row it prints; time's full report goes to
# <out>.time.log. Without GNU time (macOS) both columns stay empty.
measure() {
    rows=$(mktemp)
    if /usr/bin/time -v true >/dev/null 2>&1; then
        report=$(mktemp)
        /usr/bin/time -v -o "$report" "$@" >"$rows"
        cpu=$(awk -F': ' '/Percent of CPU/{sub("%", "", $2); print $2}' "$report")
        rss=$(awk -F': ' '/Maximum resident set size/{print $2}' "$report")
        cat "$report" >>"$out.time.log"
        rm -f "$report"
    else
        "$@" >"$rows"
        cpu="" rss=""
    fi
    sed "s/\$/,$cpu,$rss/" "$rows" >>"$out"
    rm -f "$rows"
}
rust() { measure "$target/release/examples/bench_dmrg" "$@"; }
mpskit() { measure "$julia" --project="$app/bench/mpskit" -t "$threads" "$app/bench/mpskit/bench.jl" "$@"; }

if [ "$(uname)" = Darwin ]; then
    machine="$(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.physicalcpu) cores ($(sysctl -n hw.perflevel0.physicalcpu) performance + $(sysctl -n hw.perflevel1.physicalcpu) efficiency), $(sysctl -n hw.memsize | awk '{print $1 / 2^30}') GiB, macOS $(sw_vers -productVersion)"
else
    machine="$(lscpu | awk -F': +' '/^Model name/{print $2; exit}'), $(lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l) cores, $(awk '/MemTotal/{print $2 / 2^20}' /proc/meminfo) GiB, $(uname -sr), host $(hostname -s)"
fi
openblas=$(echo "$openblas_dir" | sed 's/.*OpenBLAS-/OpenBLAS /')
if [ -n "${BENCH_IMAGE:-}" ]; then
    image_note="$BENCH_IMAGE: $(cat "$BENCH_IMAGE.txt")"
else
    image_note="none (BENCH_IMAGE unset, no cold_sysimage rows)"
fi

{
    echo "# DMRG2 on the open golden chain, H = -J sum P^(1), total charge 1; golden-chain/bench/run.sh $mode $threads L=($lens) D=($ds) J=($js)"
    echo "# machine: $machine"
    echo "# symtn: $(rustc --version), symtn-apps $(git -C "$app" rev-parse --short HEAD), symtn $(grep -o 'rev = "[0-9a-f]*"' "$app/../Cargo.toml" | cut -c8-14), ${openblas:-OpenBLAS (openblas-src)} built from source by openblas-src, for GEMM and LAPACK"
    echo "# symtn threads: Runtime::builder().dense_threads($threads).recoupling_threads($threads), OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=RAYON_NUM_THREADS=$threads"
    echo "# columns: reported = thread settings in effect as each process reports them; cpu_percent, max_rss_kb = GNU /usr/bin/time -v of that whole process (warm-up and every row it printed)"
    echo "# MPSKit: julia $("$julia" --version | awk '{print $3}'), MPSKit 0.13.13, TensorKit 0.17.1, TensorKitSectors 0.3.10, KrylovKit $(grep -A6 '^\[\[deps.KrylovKit\]\]' "$app/bench/mpskit/Manifest.toml" | awk -F'"' '/^version/{print $2}'), $("$julia" -e 'using LinearAlgebra; print(BLAS.get_config().loaded_libs[1].libname |> basename)') (Julia's OpenBLAS_jll)"
    echo "# MPSKit threads: julia -t $threads, BLAS.set_num_threads($threads), MPSKit Defaults.scheduler default (Serial at 1 thread, Dynamic otherwise), TensorKit blockscheduler default (SerialScheduler)"
    echo "# settings (both sides): D = truncrank(D) on the total quantum dimension; tol 1e-10, maxiter 50; Lanczos krylovdim 30, maxiter 200, eager; updatetol clamp(eps*1e-3/sqrt(sweep), 1e-14, 1e-12); initial virtual spaces D/(1+phi) per sector capped at the reachable space"
    echo "# differences: Lanczos orthogonalization is krylov Cgs2 (two full passes per step) in symtn and KrylovKit ModifiedGramSchmidt2 (three-term recurrence plus one full pass) in MPSKit; the initial state is right-isometric sites + a seeded random first centre in symtn and a seeded random FiniteMPS (Random.seed!(1234)) in MPSKit"
    if [ "$mode" = first-run ]; then
        echo "# kinds: cold = fresh process, setup + first sweep of the target case (Julia: includes JIT compilation; Rust: compiled ahead of time, no JIT); cold_sysimage = the same with julia --sysimage, image $image_note; first_after_compile = setup + first sweep of the target case after a one-sweep run of L = 6, D = 8, same J, in the same process (compilation excluded, nothing of the target cached)"
    else
        echo "# kinds: full = warm run to convergence; sweep = one sweep of a warm run of exactly 4 sweeps (sweep 1 includes setup); profile = MPSKit's DMRG2 loop copied with timers (split and matvecs only)"
    fi
    echo "side,threads,kind,L,D,J,rep,sweep,seconds,eigsolve_s,svd_s,env_s,apply_s,matvecs,energy,epsilon,sweeps,converged,reported,cpu_percent,max_rss_kb"
} >"$out"
: >"$out.time.log"

for len in $lens; do
    for d in $ds; do
        for j in $js; do
            if [ "$mode" = first-run ]; then
                wait_idle; rust cold "$threads" "$len" "$d" "$j"
                wait_idle; rust first "$threads" "$len" "$d" "$j"
                wait_idle; mpskit cold "$threads" "$len" "$d" "$j"
                if [ -n "${BENCH_IMAGE:-}" ]; then
                    wait_idle
                    measure "$julia" --sysimage="$BENCH_IMAGE" --project="$app/bench/mpskit" -t "$threads" \
                        "$app/bench/mpskit/bench.jl" cold "$threads" "$len" "$d" "$j" cold_sysimage
                fi
                wait_idle; mpskit first "$threads" "$len" "$d" "$j"
            else
                wait_idle; rust warm "$threads" "$len" "$d" "$j"
                wait_idle; mpskit warm "$threads" "$len" "$d" "$j"
            fi
        done
    done
done
echo "$out"

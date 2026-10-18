#!/bin/sh
# Builds the precompiled Julia image of the bench path (bench/mpskit/
# sysimage.jl) for run.sh's cold_sysimage rows:
#
#     golden-chain/bench/build_sysimage.sh <scratch dir>           # prints the image path
#     golden-chain/bench/build_sysimage.sh <scratch dir> install   # only installs PackageCompiler
#
# PackageCompiler goes into a private environment and depot under the
# scratch directory, so neither the committed Manifest nor ~/.julia changes.
# <image>.txt records the PackageCompiler version and the build time.
set -eu
scratch=${1:?usage: build_sysimage.sh <scratch dir>}
julia=${JULIA:-julia}
app=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$scratch/depot" "$scratch/env"
scratch=$(cd "$scratch" && pwd)
case $(uname) in Darwin) ext=dylib ;; *) ext=so ;; esac
image="$scratch/golden-chain-bench.$ext"
if [ "${2:-}" = install ]; then
    JULIA_DEPOT_PATH="$scratch/depot:${JULIA_DEPOT_PATH:-$HOME/.julia}" "$julia" --project="$scratch/env" \
        -e 'using Pkg; Pkg.add("PackageCompiler"; io = devnull)'
    exit 0
fi
"$julia" --project="$app/bench/mpskit" -e 'using Pkg; Pkg.instantiate()'
JULIA_DEPOT_PATH="$scratch/depot:${JULIA_DEPOT_PATH:-$HOME/.julia}" "$julia" --project="$scratch/env" -e '
    using Pkg
    # Installed once (on a host with network); later builds reuse it.
    isfile(joinpath(dirname(Base.active_project()), "Manifest.toml")) || Pkg.add("PackageCompiler"; io = devnull)
    include(ARGS[1])' "$app/bench/mpskit/sysimage.jl" "$app/bench/mpskit" "$image" >"$image.log" 2>&1
test -f "$image" || { echo "sysimage build failed, see $image.log" >&2; exit 1; }
tail -1 "$image.log" >"$image.txt"
echo "$image"

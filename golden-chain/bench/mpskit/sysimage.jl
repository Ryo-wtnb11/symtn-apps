# Builds the precompiled image for the `cold_sysimage` rows:
#
#     JULIA_DEPOT_PATH=<private depot>: julia --project=<private env> sysimage.jl <bench project> <image path>
#
# PackageCompiler is not a dependency of the bench project; run.sh installs
# it into a private environment and depot, so neither the committed Manifest
# nor ~/.julia changes.
using PackageCompiler
project, image = ARGS
seconds = @elapsed create_sysimage(
    [:MPSKit, :TensorKit, :TensorKitSectors];
    project, sysimage_path = image,
    precompile_execution_file = joinpath(@__DIR__, "precompile.jl"),
)
println("PackageCompiler ", pkgversion(PackageCompiler), ", image build ", round(seconds; digits = 1), " s")

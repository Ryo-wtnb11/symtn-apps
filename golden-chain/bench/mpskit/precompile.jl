# Precompile workload of `sysimage.jl`: the bench's code path on tiny cases,
# both signs of J (the bench functions themselves live in Main, so only the
# package methods they reach are kept in the image).
include(joinpath(@__DIR__, "bench.jl"))
for J in (1.0, -1.0)
    H = golden_chain(6, J)
    timed_run(H, 6, 8; maxiter = 1)
    timed_run(H, 6, 8; tol = -1.0, maxiter = 2)
end

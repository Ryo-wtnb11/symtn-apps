# MPSKit DMRG2 timings for the comparison with `examples/bench_dmrg.rs`;
# `bench/run.sh` drives both and adds the CSV header.
#
#     julia --project=golden-chain/bench/mpskit -t <threads> golden-chain/bench/mpskit/bench.jl cold|first|warm <threads> <L> <D> <J> [label]
#
# Rows as on the Rust side: side,threads,kind,L,D,J,rep,sweep,seconds,
# eigsolve_s,svd_s,env_s,apply_s,matvecs,energy,epsilon,sweeps,converged,
# reported (the thread settings in effect), where kind is
# - cold: state construction, environments and the first sweep of the first
#   run in this process (includes JIT compilation); `cold_sysimage` is the
#   same with the precompiled image of `sysimage.jl` (`julia --sysimage`);
# - first_after_compile: the same for the target case after a one-sweep run of
#   a tiny different case (L = 6, D = 8, same J) in this process;
# - full: a warm run to convergence at fixed D, wall time from the state
#   construction to the return of `find_groundstate`;
# - sweep: one sweep of a warm `find_groundstate` run of exactly FIXED_SWEEPS
#   sweeps, timed by the `finalize` callback (sweep 1 includes the setup);
# - profile: one sweep of `profiled_dmrg2`, a copy of MPSKit's DMRG2 loop
#   with timers, timed H_eff applications and the eigensolver's `numops`
#   (the library loop exposes none of them); not used for the timing tables.

using MPSKit, TensorKit, TensorKitSectors, Random, LinearAlgebra
using MPSKit: KrylovKit, AC2_hamiltonian, calc_galerkin, updatetol, _transpose_front

const P = Vect[FibonacciAnyon](:τ => 1)

# The Rust side's `model::bond_term`: h = -J P^(1), the block of the coupled vacuum.
function golden_chain(L, J)
    h = zeros(ComplexF64, P ⊗ P ← P ⊗ P)
    block(h, FibonacciAnyon(:I)) .= -J
    return FiniteMPOHamiltonian(fill(P, L), (i, i + 1) => h for i in 1:(L - 1))
end

# D split evenly over the sectors (MPSKit's hard-hexagon `virtual_space`);
# FiniteMPS caps each bond at its reachable space, as `model::virtual_spaces`.
virtual_space(D) = Vect[FibonacciAnyon](s => round(Int, D / (1 + dim(FibonacciAnyon(:τ)))) for s in (:I, :τ))

const FIXED_SWEEPS = 4
# As `bench_dmrg.rs`: a first full run beyond LONG_RUN seconds selects the
# second counts.
const FULL_REPEATS = (5, 3)
const FIXED_REPEATS = (3, 1)
const LONG_RUN = 20.0
# The warm-up case of `first`, as on the Rust side.
const WARMUP_LEN = 6
const WARMUP_D = 8

# `tests/common/mod.rs:mpskit_dmrg2`, every value spelled out.
function dmrg2(D; tol = 1.0e-10, maxiter = 50, finalize = MPSKit.Defaults._finalize)
    return DMRG2(;
        trscheme = truncrank(D), tol, maxiter, verbosity = 0, finalize,
        alg_eigsolve = (;
            krylovdim = 30, maxiter = 200, eager = true,
            tol_min = 1.0e-14, tol_max = 1.0e-12, tol_factor = 1.0e-3,
        ),
    )
end

initial_state(L, D) = (Random.seed!(1234); FiniteMPS(L, P, virtual_space(D); right = Vect[FibonacciAnyon](:I => 1)))

# Times of the end of every sweep, relative to `start`.
function timed_run(H, L, D; kwargs...)
    stamps = UInt64[]
    finalize = (iter, ψ, H, envs) -> (push!(stamps, time_ns()); (ψ, envs))
    start = time_ns()
    ψ, envs, ϵ = find_groundstate(initial_state(L, D), H, dmrg2(D; finalize, kwargs...))
    stop = time_ns()
    return ψ, envs, ϵ, (stop - start) / 1.0e9, (stamps .- start) ./ 1.0e9
end

row(threads, kind, L, D, J, rep, sweep, seconds; eig = "", svd = "", env = "", apply = "", matvecs = "", energy = "", epsilon = "", sweeps = "", converged = "") =
    (println(join(("MPSKit", threads, kind, L, D, Int(J), rep, sweep, seconds, eig, svd, env, apply, matvecs, energy, epsilon, sweeps, converged, reported()), ",")); flush(stdout))

# The thread settings in effect, for the `reported` column.
reported() = join((
    "nthreads=$(Threads.nthreads())", "blas=$(BLAS.get_num_threads())",
    "mpskit_scheduler=$(nameof(typeof(MPSKit.Defaults.scheduler[])))",
    "tensorkit_blockscheduler=$(nameof(typeof(TensorKit.blockscheduler[])))",
), "; ")

energy(ψ, H, envs) = real(expectation_value(ψ, H, envs))

function warm(threads, L, D, J)
    H = golden_chain(L, J)
    timed_run(H, L, D; maxiter = 1)  # warm-up
    long = 1
    for rep in 1:maximum(FULL_REPEATS)
        rep > FULL_REPEATS[long] && break
        ψ, envs, ϵ, seconds, stamps = timed_run(H, L, D)
        rep == 1 && seconds > LONG_RUN && (long = 2)
        row(threads, "full", L, D, J, rep, "", seconds; energy = energy(ψ, H, envs), epsilon = ϵ,
            sweeps = length(stamps), converged = ϵ <= 1.0e-10)
    end
    for rep in 1:FIXED_REPEATS[long]
        ψ, envs, ϵ, _, stamps = timed_run(H, L, D; tol = -1.0, maxiter = FIXED_SWEEPS)
        length(stamps) == FIXED_SWEEPS || error("fixed run stopped after $(length(stamps)) sweeps")
        e = energy(ψ, H, envs)
        for (k, t) in enumerate(stamps)
            row(threads, "sweep", L, D, J, rep, k, k == 1 ? t : t - stamps[k - 1];
                energy = k == FIXED_SWEEPS ? e : "", epsilon = k == FIXED_SWEEPS ? ϵ : "")
        end
    end
    profiled_dmrg2(threads, H, L, D, J)
    @info "done" L D J
    return nothing
end

# MPSKit 0.13.13 `find_groundstate!(ψ, H, ::DMRG2)` (src/algorithms/groundstate/dmrg.jl)
# for FIXED_SWEEPS sweeps, with `fixedpoint` expanded into its `eigsolve` call
# so `info.numops` is visible, and `leftenv`/`rightenv` forced before the
# eigensolve so the environment updates are timed apart from it.
function profiled_dmrg2(threads, H, L, D, J)
    alg = dmrg2(D)
    ψ = initial_state(L, D)
    envs = environments(ψ, H)
    ϵs = map(pos -> calc_galerkin(pos, ψ, H, ψ, envs), 1:L)
    ϵ = maximum(ϵs)
    for iter in 1:FIXED_SWEEPS
        alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
        zerovector!(ϵs)
        t = zeros(3)  # eigsolve, svd, env
        matvecs = 0
        apply = Ref(0.0)
        sweep_start = time_ns()
        for (pos, right) in Iterators.flatten((((p, true) for p in 1:(L - 1)), ((p, false) for p in (L - 2):-1:1)))
            t0 = time_ns()
            if right
                @plansor ac2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos + 1][1 -4; -3]
            else
                @plansor ac2[-1 -2; -3 -4] := ψ.AL[pos][-1 -2; 1] * ψ.AC[pos + 1][1 -4; -3]
            end
            leftenv(envs, pos, ψ)
            rightenv(envs, pos + 1, ψ)
            Hac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
            timed = x -> (start = time_ns(); y = Hac2(x); apply[] += (time_ns() - start) / 1.0e9; y)
            t1 = time_ns()
            _, vecs, info = KrylovKit.eigsolve(timed, ac2, 1, :SR, alg_eigsolve)
            matvecs += info.numops
            t2 = time_ns()
            al, c, ar = svd_trunc!(vecs[1]; trunc = alg.trscheme, alg = alg.alg_svd)
            normalize!(c)
            v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) * conj(ar[6; 3 4])
            ϵs[pos] = max(ϵs[pos], abs(1 - abs(v)))
            t3 = time_ns()
            if right
                ψ.AC[pos] = (al, complex(c))
                ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
            else
                ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
                ψ.AC[pos] = (al, complex(c))
            end
            t4 = time_ns()
            t .+= ((t2 - t1), (t3 - t2), (t1 - t0) + (t4 - t3)) ./ 1.0e9
        end
        ϵ = maximum(ϵs)
        row(threads, "profile", L, D, J, 1, iter, (time_ns() - sweep_start) / 1.0e9;
            eig = t[1], svd = t[2], env = t[3], apply = apply[], matvecs, energy = energy(ψ, H, envs), epsilon = ϵ)
    end
    return nothing
end

function main(args)
    usage = "usage: bench.jl cold <threads> <L> <D> <J> [label] | first|warm <threads> <L> <D> <J>"
    length(args) >= 5 || error(usage)
    threads = parse(Int, args[2])
    Threads.nthreads() == threads || error("start julia with -t $threads")
    # Threading knobs: julia -t, the BLAS pool, and MPSKit's default scheduler
    # (`Defaults.scheduler`: Serial for one thread, Dynamic otherwise);
    # TensorKit's `blockscheduler` stays at its default, SerialScheduler.
    BLAS.set_num_threads(threads)
    if args[1] in ("cold", "first")
        L, D, J = parse(Int, args[3]), parse(Int, args[4]), parse(Float64, args[5])
        kind = if args[1] == "first"
            # A different, tiny case first: compiles the code path, caches
            # nothing of the target's spaces.
            timed_run(golden_chain(WARMUP_LEN, J), WARMUP_LEN, WARMUP_D; maxiter = 1)
            "first_after_compile"
        else
            # `cold_sysimage` when started with the image of sysimage.jl.
            get(args, 6, "cold")
        end
        H = golden_chain(L, J)
        _, _, ϵ, _, stamps = timed_run(H, L, D; maxiter = 1)
        row(threads, kind, L, D, J, 0, 1, stamps[1]; epsilon = ϵ)
    elseif args[1] == "warm"
        warm(threads, parse(Int, args[3]), parse(Int, args[4]), parse(Float64, args[5]))
    else
        error(usage)
    end
    return nothing
end

# `precompile.jl` includes this file for its definitions only.
abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)

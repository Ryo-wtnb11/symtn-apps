# symtn-apps

Applications built on [symtn](https://github.com/Ryo-wtnb11/symtn-rs), the
symmetric tensor library. Applications live here, not in symtn: symtn owns
general categorical tensor operations, and models, algorithms such as DMRG,
and their physics validation belong to the applications that use them.

## Crates

| Crate | What |
| --- | --- |
| [`golden-chain`](golden-chain) | Two-site DMRG for the Fibonacci golden chain on anyonic (`TableRule::fibonacci()`) tensors |

## Golden chain

`L` Fibonacci anyons with `H = -J sum_i P^(1)_{i,i+1}`, the projector onto the
trivial fusion channel of neighbouring anyons. `J = +1` is critical with the
tricritical Ising CFT (`c = 7/10`), `J = -1` with the 3-state Potts CFT
(`c = 4/5`).

Validation does not trust the tensors under test: an exact diagonalization in
the fusion-path basis is written independently of symtn, and MPSKit results
are checked in as fixtures (CI never runs Julia).

## Dependencies

symtn is a git dependency pinned by revision; `Cargo.lock` is committed and
CI builds with `--locked`.

| Dependency | Pin | Features |
| --- | --- | --- |
| symtn | [`2b7c4ef`](https://github.com/Ryo-wtnb11/symtn-rs/commit/2b7c4ef151fc6e1b0e78eabf96656b6f0cd5d1eb) | `krylov`, `blas-openblas` (no default `cpu-faer`) |
| krylov | via symtn's re-export `symtn::krylov` (krylov 0.2 from crates.io) | — |

The Lanczos eigensolver is reached only through `symtn::krylov`, so the app
can never resolve a second krylov whose `Vector` trait would not unify with
symtn's `TensorMap` impl. krylov declares LAPACK; symtn's `blas-openblas`
links OpenBLAS, built from source by `openblas-src` (no system library is
needed, the first build takes several minutes).

## References

- A. Feiguin, S. Trebst, A. W. W. Ludwig, M. Troyer, A. Kitaev, Z. Wang,
  M. H. Freedman, "Interacting anyons in topological quantum liquids: the
  golden chain", Phys. Rev. Lett. 98, 160409 (2007).
- TensorKit.jl 0.17.1 and MPSKit.jl 0.13.13 (reference implementations).

## License

MIT OR Apache-2.0.

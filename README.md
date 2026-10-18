# symtn-apps

Applications built on [symtn](https://github.com/Ryo-wtnb11/symtn-rs), the
symmetric tensor library. Applications live here, not in symtn: symtn owns
general categorical tensor operations, and models, algorithms such as DMRG,
and their physics validation belong to the applications that use them.

## Applications

- [`golden-chain`](golden-chain/README.md): two-site DMRG for the Fibonacci golden chain on anyonic (`TableRule::fibonacci()`) tensors, validated against an independent exact diagonalization.

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

## CI

One job on `ubuntu-latest` runs `cargo fmt --check`,
`cargo clippy --all-targets -D warnings` and `cargo test --release`. The DMRG
tests take about 10 s optimized and minutes in a debug build. Slow physics
runs are `#[ignore]` tests or examples, not CI gates.

## License

MIT OR Apache-2.0.

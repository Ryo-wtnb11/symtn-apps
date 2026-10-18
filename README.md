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

### Validation

| Test | Oracle |
| --- | --- |
| `tests/ed.rs` | ED against TensorKit's 3-anyon matrix and the Temperley–Lieb relations |
| `tests/heff.rs` | `H_eff` against ED; centre-bond Lanczos = ED ground energy |
| `tests/dmrg.rs` | DMRG ground energy = ED (sparse Lanczos, cross-checked by Jacobi) to 1e-9, L ≤ 20, both `J`, both total charges; per-bond Schmidt values = ED's (`λ = σ/√d_x` for a unit ED vector) |
| `tests/mpskit.rs` | E₀, entanglement spectra and entropies per bond against MPSKit `DMRG2`, L ∈ {12, 16, 20, 24} |
| `tests/central_charge.rs` (`#[ignore]`) | c from `S(l)` at L = 96: ≈ 0.67 (`J = +1`, TCI 0.7), ≈ 0.78 (`J = -1`, Potts 0.8) |

The MPSKit fixture is regenerated with
`julia --project=reference/mpskit reference/mpskit/golden_chain.jl` (MPSKit
0.13.13, TensorKit 0.17.1, TensorKitSectors 0.3.10, pinned by the checked-in
`Manifest.toml`). The slow fit runs with
`cargo test --release --test central_charge -- --ignored --nocapture`, and
`cargo run --release --example golden_chain` prints a CSV profile.

### Stages

| Stage | Content | Status |
| --- | --- | --- |
| S0 | ED oracle, isometric MPS, environments, two-site `H_eff` | done |
| S1 | `H_eff` checked through symtn's `inner`; entropy | done |
| S2 | `H_eff` as a krylov `LinearMap`; centre-bond Lanczos equals the ED ground energy (L ≤ 12) | done |
| S3 | Two-site DMRG (`svd_compact` + `find_truncated` + `restrict_leg`), sweeps, convergence; DMRG = ED to 1e-9 up to L = 20; MPSKit fixtures; central-charge fit | done |
| S4 | environments and `H_eff` on symtn's `planar_contract` (no app-side planar helper) | done |

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

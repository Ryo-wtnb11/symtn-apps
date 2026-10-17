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

symtn (and krylov-rs, once the eigensolver lands) are git dependencies pinned
by revision; `Cargo.lock` is
committed and CI builds with `--locked`.

## References

- A. Feiguin, S. Trebst, A. W. W. Ludwig, M. Troyer, A. Kitaev, Z. Wang,
  M. H. Freedman, "Interacting anyons in topological quantum liquids: the
  golden chain", Phys. Rev. Lett. 98, 160409 (2007).
- TensorKit.jl 0.17.1 and MPSKit.jl 0.13.13 (reference implementations).

## License

MIT OR Apache-2.0.

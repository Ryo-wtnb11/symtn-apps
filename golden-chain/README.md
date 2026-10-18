# golden-chain

Two-site DMRG for the Fibonacci golden chain on symtn's anyonic tensors
(`TableRule::fibonacci()`).

## Model

`L` Fibonacci anyons `τ` (`τ ⊗ τ = 1 ⊕ τ`, `d_τ = φ`) on an open chain, in the
fusion-path basis `|x_0 = 1, x_1, …, x_L = total⟩`:

```
H = -J Σ_{i=1}^{L-1} P^(1)_{i,i+1}
```

where `P^(1)` projects neighbouring anyons onto the trivial fusion channel.
In symtn this is the bond term `h ∈ [P, P] ← [P, P]` with block `-J` on the
coupled vacuum and `0` on coupled `τ` (`src/model.rs:bond_term`).

Feiguin et al. (PRL 98, 160409 (2007), p. 3) identify the two critical
points:
- `J = +1` favours the trivial channel. It is the k = 3 RSOS model, the
  tricritical Ising CFT with `c = 7/10`.
- `J = -1` favours the `τ` channel. It is the 3-state Potts CFT with
  `c = 4/5`.

## Algorithm

The algorithmic model is MPSKit 0.13.13's two-site `DMRG2`
(`find_groundstate!(…, ::DMRG2)`); `dmrg::find_groundstate` follows it bond
by bond:
- Lanczos on the two-site `H_eff` (via `symtn::krylov`);
- `svd_compact`, then `find_truncated(Truncation::Rank(D))`, then
  `restrict_leg`, then normalization;
- `ϵ_bond = |1 − |⟨θ_old, u s vh⟩||`, and the eigensolver tolerance follows
  MPSKit's `updatetol`, `clamp(ϵ·10⁻³/√sweep, tol_min, tol_max)`;
- a sweep runs left to right over `1..L-1`, then right to left over `L-2..1`;
  it stops when `max ϵ ≤ tol`.

There is no MPO. The block Hamiltonian and the open bond are kept as
explicit environments (`src/mps.rs`), which is exact for nearest-neighbour
terms. Every numerical constant is a field of `Dmrg2`; there are no defaults.

The tests and the example use MPSKit's default values with one exception:
`tol_max = 1e-12` instead of `1e-4`. At `1e-4`, the second sweep's
eigensolves are already below tolerance at their start vectors and return
them unchanged. The run then stops with `ϵ ≈ 1e-14`, yet the energy is
still above the ED ground energy, by 2.8e-9 (total 1) and 3.7e-9 (total τ)
at L = 20, `J = -1`.

## Validation

The tests do not trust the tensors under test. `src/ed.rs` is an exact
diagonalization in the fusion-path basis with a closed-form F-symbol and no
symtn dependency.

- **ED oracle.** The ED reproduces TensorKit's 3-anyon matrix to 1e-13 and
  satisfies the Temperley–Lieb relations to 1e-13 (`tests/ed.rs`).
- **Environments and `H_eff`.** At full bond dimension `H_eff` equals the ED
  Hamiltonian to 1e-12. At truncated bonds it is the Galerkin projection of
  the ED Hamiltonian, to 1e-12 under symtn's quantum-dimension-weighted
  `inner` (`tests/heff.rs`).
- **Centre-bond Lanczos.** On `H_eff` it gives the ED ground energy to 1e-10
  for L ≤ 12 (`tests/heff.rs`).
- **DMRG ground energy.** DMRG `E₀` equals ED to 1e-9 for L ∈ {6, 11, 16, 20},
  both `J` and both total charges (`tests/dmrg.rs`). The ED uses a sparse
  Lanczos, which matches dense Jacobi to 1e-11 for L ≤ 12.
- **Schmidt values.** At every bond (L = 9), DMRG's Schmidt values equal ED's
  to 1e-9 in `λ²`. The relation is `λ_{x,k} = σ_{x,k}/√d_x` for a unit ED
  vector, with `Σ_x d_x Σ_k λ² = 1` to 1e-12 (`tests/dmrg.rs`).
- **Central charge** (`tests/central_charge.rs`, `#[ignore]`). Fitted from the
  entanglement profile at L = 96, it must lie within ±0.05 of 7/10 and 4/5.
  This is a physics check, not an exact gate.

## Running

```sh
cargo test --release -p golden-chain                       # all checks, about 10 s
cargo test --release --test central_charge -- --ignored    # central-charge fit, about 30 s
cargo run --release --example golden_chain                 # CSV data in golden-chain/out/
uv run golden-chain/scripts/plot.py                        # figures in golden-chain/docs/figures/
```

## Results

The figures come from `cargo run --release --example golden_chain` and
`uv run golden-chain/scripts/plot.py`, with the `x_L = 1` sector and `D` as
stated in each title.

**Entanglement profile.** The fit is
`S(l) = (c/6) ln[(2L/π) sin(πl/L)] + s′` over `L/8 ≤ l ≤ 7L/8`, one fit per
residue of `l mod 6` (the open chain alternates with bond parity, and the
`J = -1` profile also has period 3), with `c` averaged. The results are
`c ≈ 0.67` for `J = +1` (exact value 7/10) and `c ≈ 0.78` for `J = -1`
(exact value 4/5).

![Entanglement profile at L = 96 with the CFT fit](docs/figures/entropy_profile.png)

**Convergence.** The plot shows `|ΔE|` between sweeps and the largest
discarded weight in each sweep, for the D ramp 32 → 64 → 128 at L = 96.
Both runs stop after three sweeps at `max ϵ ≤ 1e-10`.

![Per-sweep |ΔE| and discarded weight](docs/figures/convergence.png)

**Ground energy against ED.** At D = 128 every case agrees with the
sparse-Lanczos ED to ≤ 1.1e-13. That holds for L = 24 (46368 paths) too,
where the centre bond is truncated.

![|E_DMRG − E_ED| against L](docs/figures/energy_error.png)

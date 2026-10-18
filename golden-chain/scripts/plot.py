# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""README figures for the golden chain.

    cargo run --release --example golden_chain   # writes golden-chain/out/*.csv
    uv run golden-chain/scripts/plot.py           # writes golden-chain/docs/figures/*.png
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

APP = Path(__file__).resolve().parent.parent
OUT = APP / "out"
FIGURES = APP / "docs" / "figures"
LABEL = {1.0: "J = +1 (tricritical Ising, c = 7/10)", -1.0: "J = -1 (3-state Potts, c = 4/5)"}


def rows(name):
    with open(OUT / name) as f:
        return list(csv.DictReader(f))


def chord(l, n):
    return np.log(2 * n / np.pi * np.sin(np.pi * l / n)) / 6


def fit(l, s, n):
    """c and s' per residue of l mod 6 over L/8 <= l <= 7L/8 (as tests/central_charge.rs),
    c averaged; returns c and the per-residue offsets."""
    cs, offsets = [], {}
    for r in range(6):
        m = (l % 6 == r) & (l >= n // 8) & (l <= 7 * n // 8)
        c, s0 = np.polyfit(chord(l[m], n), s[m], 1)
        cs.append(c)
        offsets[r] = s0
    return float(np.mean(cs)), offsets


def profile():
    data = rows("profile.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, j in zip(axes, (1.0, -1.0)):
        l = np.array([int(r["l"]) for r in data if float(r["J"]) == j])
        s = np.array([float(r["S"]) for r in data if float(r["J"]) == j])
        n = len(l) + 1
        c, offsets = fit(l, s, n)
        ax.plot(l, s, ".", color="black", ms=4, label="DMRG")
        # The fitted curve per residue class (the profile alternates).
        x = np.linspace(1, n - 1, 400)
        for r, s0 in offsets.items():
            ax.plot(x, c * chord(x, n) + s0, "-", color="tab:blue", lw=0.6, alpha=0.6,
                    label=f"(c/6) ln[(2L/π) sin(πl/L)] + s′, c = {c:.3f}" if r == 0 else None)
        ax.set_title(LABEL[j])
        ax.set_xlabel("bond l")
        ax.legend(loc="lower center", fontsize=8)
    axes[0].set_ylabel("entanglement entropy S(l)")
    fig.suptitle(f"Open chain, L = {n}, D = 128")
    fig.tight_layout()
    fig.savefig(FIGURES / "entropy_profile.png", dpi=150)


def sweeps():
    data = rows("sweeps.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for j, color in ((1.0, "tab:blue"), (-1.0, "tab:red")):
        sel = [r for r in data if float(r["J"]) == j]
        k = [int(r["sweep"]) for r in sel]
        de = [abs(float(r["dE"])) for r in sel]
        tr = [float(r["discarded"]) for r in sel]
        ax.semilogy(k[1:], de[1:], "o-", color=color, label=f"|ΔE|, J = {j:+.0f}")
        ax.semilogy(k, tr, "s--", color=color, mfc="none", label=f"max discarded weight, J = {j:+.0f}")
    ax.set_xlabel("sweep")
    ax.set_xticks(sorted({int(r["sweep"]) for r in data}))
    ax.set_title("DMRG2 convergence, L = 96, D = 32, 64, 128")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "convergence.png", dpi=150)


def energies():
    data = rows("energies.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    markers = {(1.0, "1"): "o", (1.0, "tau"): "s", (-1.0, "1"): "^", (-1.0, "tau"): "v"}
    for (j, total), marker in markers.items():
        sel = [r for r in data if float(r["J"]) == j and r["total"] == total]
        ls = [int(r["L"]) for r in sel]
        # Floor at 1e-16 so an exact agreement stays on the log axis.
        err = [max(abs(float(r["E_dmrg"]) - float(r["E_ed"])), 1e-16) for r in sel]
        ax.semilogy(ls, err, marker + "-", label=f"J = {j:+.0f}, total {'τ' if total == 'tau' else '1'}")
    ax.set_xlabel("L")
    ax.set_xticks(sorted({int(r["L"]) for r in data}))
    ax.set_ylabel("|E_DMRG − E_ED|")
    ax.set_title("Ground energy against ED, D = 128")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "energy_error.png", dpi=150)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    profile()
    sweeps()
    energies()

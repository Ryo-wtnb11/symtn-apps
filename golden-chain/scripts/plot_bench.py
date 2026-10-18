# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""DMRG2 timing tables and figure, symtn against MPSKit.

    golden-chain/bench/run.sh 1                 # writes golden-chain/bench/results/dmrg2_t1.csv
    uv run golden-chain/scripts/plot_bench.py   # summary CSV, Markdown tables, docs/figures/dmrg2_timing.png

Per case (side, threads, L, D, J):
- cold: setup plus the first sweep in a fresh process;
- full: median [min, max] over the warm runs to convergence, with the sweep
  count and E0;
- sweep: median [min, max] over sweeps 2..4 of the fixed four-sweep runs
  (sweep 1 starts from a different state on each side and includes setup);
- matvecs per sweep and the eigsolve / SVD / environment split over the same
  sweeps (symtn: the timed runs; MPSKit: its profiled copy of the loop).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

APP = Path(__file__).resolve().parent.parent
RESULTS = APP / "bench" / "results"
FIGURES = APP / "docs" / "figures"
SIDES = ("symtn", "MPSKit")
STEADY = (2, 3, 4)


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def key(r):
    return (r["side"], int(r["threads"]), int(r["L"]), int(r["D"]), int(float(r["J"])))


def summarize(data):
    groups = defaultdict(lambda: defaultdict(list))
    for r in data:
        groups[key(r)][r["kind"]].append(r)
    out = {}
    for k, g in groups.items():
        side = k[0]
        full = [float(r["seconds"]) for r in g["full"]]
        steady = [r for r in g["sweep"] if int(r["sweep"]) in STEADY]
        sweep = [float(r["seconds"]) for r in steady]
        split = steady if side == "symtn" else [r for r in g["profile"] if int(r["sweep"]) in STEADY]
        n = max(len(split), 1)
        mean = lambda f: sum(float(r[f]) for r in split) / n
        out[k] = {
            "cold_s": float(g["cold"][0]["seconds"]) if g["cold"] else float("nan"),
            "full_s": np.median(full),
            "full_min_s": min(full),
            "full_max_s": max(full),
            "sweeps": int(g["full"][0]["sweeps"]),
            "converged": g["full"][0]["converged"],
            "E0": float(g["full"][0]["energy"]),
            "sweep_s": np.median(sweep),
            "sweep_min_s": min(sweep),
            "sweep_max_s": max(sweep),
            "matvecs": mean("matvecs"),
            "eigsolve_s": mean("eigsolve_s"),
            "svd_s": mean("svd_s"),
            "env_s": mean("env_s"),
            "apply_s": mean("apply_s"),
            "profile_s": mean("seconds"),
        }
    return out


def tables(s, threads):
    cases = sorted({k[2:] for k in s if k[1] == threads})
    print(f"\n### {threads} thread{'s' if threads > 1 else ''}\n")
    print("| L | D | J | E0 symtn | ΔE0 | sweeps symtn / MPSKit | cold s symtn / MPSKit | full s symtn / MPSKit | sweep s symtn / MPSKit | MPSKit / symtn per sweep |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in cases:
        a, b = s.get(("symtn", threads, *c)), s.get(("MPSKit", threads, *c))
        if a is None or b is None:
            continue
        rng = lambda x, f: f"{x[f + '_s']:.3g} [{x[f + '_min_s']:.3g}, {x[f + '_max_s']:.3g}]"
        print(
            f"| {c[0]} | {c[1]} | {c[2]:+d} | {a['E0']:.12f} | {abs(a['E0'] - b['E0']):.1e} "
            f"| {a['sweeps']} / {b['sweeps']} | {a['cold_s']:.3g} / {b['cold_s']:.3g} "
            f"| {rng(a, 'full')} / {rng(b, 'full')} | {rng(a, 'sweep')} / {rng(b, 'sweep')} "
            f"| {b['sweep_s'] / a['sweep_s']:.2f} |"
        )
    print("\nPer steady sweep (sweeps 2..4, mean): H_eff applications and time split (s)\n")
    print("| L | D | J | matvecs symtn / MPSKit | eigsolve symtn / MPSKit | of which H_eff apply | SVD symtn / MPSKit | env symtn / MPSKit | ms per H_eff apply symtn / MPSKit |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in cases:
        a, b = s.get(("symtn", threads, *c)), s.get(("MPSKit", threads, *c))
        if a is None or b is None:
            continue
        pm = lambda x: 1e3 * x["apply_s"] / x["matvecs"]
        print(
            f"| {c[0]} | {c[1]} | {c[2]:+d} | {a['matvecs']:.0f} / {b['matvecs']:.0f} "
            f"| {a['eigsolve_s']:.3g} / {b['eigsolve_s']:.3g} | {a['apply_s']:.3g} / {b['apply_s']:.3g} | {a['svd_s']:.3g} / {b['svd_s']:.3g} "
            f"| {a['env_s']:.3g} / {b['env_s']:.3g} | {pm(a):.3g} / {pm(b):.3g} |"
        )


def figure(s, threads):
    lens = sorted({k[2] for k in s if k[1] == threads})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colors = dict(zip(lens, ("tab:blue", "tab:orange", "tab:green")))
    for ax, j in zip(axes, (1, -1)):
        for side, style in zip(SIDES, ("o-", "s--")):
            for l in lens:
                ks = sorted(k for k in s if k[0] == side and k[1] == threads and k[2] == l and k[4] == j)
                if not ks:
                    continue
                d = [k[3] for k in ks]
                t = [s[k]["sweep_s"] for k in ks]
                err = [[s[k]["sweep_s"] - s[k]["sweep_min_s"] for k in ks], [s[k]["sweep_max_s"] - s[k]["sweep_s"] for k in ks]]
                ax.errorbar(d, t, yerr=err, fmt=style, color=colors[l], mfc="none" if side == "MPSKit" else None,
                            capsize=2, label=f"{side}, L = {l}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([32, 64, 128, 256], ["32", "64", "128", "256"])
        ax.set_xlabel("D")
        ax.set_title(f"J = {j:+d}")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("seconds per sweep (median of sweeps 2-4)")
    fig.suptitle(f"DMRG2 per-sweep wall time, golden chain, {threads} thread{'s' if threads > 1 else ''}")
    fig.tight_layout()
    fig.savefig(FIGURES / f"dmrg2_timing_t{threads}.png", dpi=150)


def write_summary(s, threads):
    fields = ["side", "threads", "L", "D", "J"] + list(next(iter(s.values())).keys())
    with open(RESULTS / f"dmrg2_summary_t{threads}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for k in sorted(k for k in s if k[1] == threads):
            w.writerow(list(k) + [s[k][f] for f in fields[5:]])


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = sorted(RESULTS.glob("dmrg2_t*.csv"))
    if not paths:
        sys.exit("no golden-chain/bench/results/dmrg2_t*.csv; run golden-chain/bench/run.sh first")
    s = summarize([r for p in paths for r in rows(p)])
    for threads in sorted({k[1] for k in s}):
        write_summary(s, threads)
        tables(s, threads)
        figure(s, threads)

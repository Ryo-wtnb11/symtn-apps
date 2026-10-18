# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""DMRG2 timing tables and figures, symtn against MPSKit.

    golden-chain/bench/run.sh first-run|warm <dir> <threads> "<L...>" "<D...>"   # CSV in <dir>
    uv run golden-chain/scripts/plot_bench.py [<dir>]   # default golden-chain/bench/results

Prints Markdown tables, writes <dir>/summary.csv and the figures
docs/figures/dmrg2_first_run.png and docs/figures/dmrg2_threads_J{+1,-1}.png.

Per case (side, threads, L, D, J):
- first run (fresh process, setup plus first sweep): cold, cold_sysimage,
  first_after_compile;
- full: median [min, max] over the warm runs to convergence, sweep count, E0;
- sweep: median [min, max] over sweeps 2..4 of the fixed four-sweep runs
  (sweep 1 starts from a different state on each side and includes setup);
- matvecs and the eigsolve / H_eff apply / SVD / environment split, mean over
  the same sweeps (symtn: the timed runs; MPSKit: its profiled loop copy);
- cpu_percent of the warm process; a run whose CPU use stays below half of
  `threads` cores is flagged.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

APP = Path(__file__).resolve().parent.parent
FIGURES = APP / "docs" / "figures"
STEADY = (2, 3, 4)
THREADS = (1, 2, 4, 8, 16, 32)
FIRST_KINDS = (
    ("symtn", "cold", "symtn, fresh process"),
    ("symtn", "first_after_compile", "symtn, after warm-up"),
    ("MPSKit", "cold", "MPSKit, fresh process with JIT"),
    ("MPSKit", "cold_sysimage", "MPSKit, fresh process with sysimage"),
    ("MPSKit", "first_after_compile", "MPSKit, after warm-up"),
)
STYLE = {
    "symtn": dict(marker="*", linestyle="-", markerfacecolor="none", markersize=11),
    "MPSKit": dict(marker="o", linestyle=":", markerfacecolor="none", markersize=7),
}
D_COLORS = {64: "tab:blue", 128: "tab:orange", 256: "tab:green"}


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def key(r):
    return (r["side"], int(r["threads"]), int(r["L"]), int(r["D"]), int(float(r["J"])))


def num(x):
    return float(x) if x not in ("", None) else float("nan")


def summarize(data):
    groups = defaultdict(lambda: defaultdict(list))
    for r in data:
        groups[key(r)][r["kind"]].append(r)
    out = {}
    for k, g in groups.items():
        s = {f"first_{kind}_s": num(g[kind][0]["seconds"]) for kind in ("cold", "cold_sysimage", "first_after_compile") if g[kind]}
        if g["full"]:
            full = [num(r["seconds"]) for r in g["full"]]
            steady = [r for r in g["sweep"] if int(r["sweep"]) in STEADY]
            sweep = [num(r["seconds"]) for r in steady]
            split = steady if k[0] == "symtn" else [r for r in g["profile"] if int(r["sweep"]) in STEADY]
            mean = lambda f: float(np.mean([num(r[f]) for r in split]))
            cpu = num(g["full"][0]["cpu_percent"])
            s.update(
                full_s=float(np.median(full)), full_min_s=min(full), full_max_s=max(full),
                sweeps=int(g["full"][0]["sweeps"]), converged=g["full"][0]["converged"],
                E0=num(g["full"][0]["energy"]),
                sweep_s=float(np.median(sweep)), sweep_min_s=min(sweep), sweep_max_s=max(sweep),
                matvecs=mean("matvecs"), eigsolve_s=mean("eigsolve_s"), apply_s=mean("apply_s"),
                svd_s=mean("svd_s"), env_s=mean("env_s"),
                cpu_percent=cpu, threads_unused=bool(k[1] > 1 and cpu < 50 * k[1]),
                reported=g["full"][0]["reported"], binding=g["full"][0].get("binding", ""),
            )
        out[k] = s
    return out


def first_run_table(s):
    cases = sorted({k[2:] for k in s if k[1] == 1 and "first_cold_s" in s[k]})
    print("\n### First result: setup + first sweep, 1 thread (seconds)\n")
    print("| L | D | J | " + " | ".join(label for _, _, label in FIRST_KINDS) + " |")
    print("|---|---|---|" + "---|" * len(FIRST_KINDS))
    for c in cases:
        cells = [s.get((side, 1, *c), {}).get(f"first_{kind}_s", float("nan")) for side, kind, _ in FIRST_KINDS]
        print(f"| {c[0]} | {c[1]} | {c[2]:+d} | " + " | ".join(f"{x:.3g}" for x in cells) + " |")


def warm_tables(s, threads):
    cases = sorted({k[2:] for k in s if k[1] == threads and "full_s" in s[k]})
    if not cases:
        return
    pair = [(c, s.get(("symtn", threads, *c)), s.get(("MPSKit", threads, *c))) for c in cases]
    pair = [(c, a, b) for c, a, b in pair if a and b and "full_s" in a and "full_s" in b]
    rng = lambda x, f: f"{x[f + '_s']:.3g} [{x[f + '_min_s']:.3g}, {x[f + '_max_s']:.3g}]"
    print(f"\n### {threads} thread{'s' if threads > 1 else ''}\n")
    print("| L | D | J | E0 symtn | ΔE0 | sweeps symtn / MPSKit | full s symtn / MPSKit | sweep s symtn / MPSKit | MPSKit / symtn per sweep | CPU % symtn / MPSKit |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c, a, b in pair:
        flag = lambda x: f"{x['cpu_percent']:.0f}" + (" (threads unused)" if x["threads_unused"] else "")
        print(
            f"| {c[0]} | {c[1]} | {c[2]:+d} | {a['E0']:.12f} | {abs(a['E0'] - b['E0']):.1e} "
            f"| {a['sweeps']} / {b['sweeps']} | {rng(a, 'full')} / {rng(b, 'full')} "
            f"| {rng(a, 'sweep')} / {rng(b, 'sweep')} | {b['sweep_s'] / a['sweep_s']:.2f} | {flag(a)} / {flag(b)} |"
        )
    print("\nPer steady sweep (sweeps 2..4, mean): H_eff applications and time split (s)\n")
    print("| L | D | J | matvecs symtn / MPSKit | eigsolve symtn / MPSKit | of which H_eff apply | SVD symtn / MPSKit | env symtn / MPSKit | ms per H_eff apply symtn / MPSKit |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c, a, b in pair:
        pm = lambda x: 1e3 * x["apply_s"] / x["matvecs"]
        print(
            f"| {c[0]} | {c[1]} | {c[2]:+d} | {a['matvecs']:.0f} / {b['matvecs']:.0f} "
            f"| {a['eigsolve_s']:.3g} / {b['eigsolve_s']:.3g} | {a['apply_s']:.3g} / {b['apply_s']:.3g} "
            f"| {a['svd_s']:.3g} / {b['svd_s']:.3g} | {a['env_s']:.3g} / {b['env_s']:.3g} | {pm(a):.3g} / {pm(b):.3g} |"
        )


def first_run_figure(s, length=48, d=64):
    if not any(k[2:4] == (length, d) and "first_cold_s" in v for k, v in s.items()):
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.16
    for i, (side, kind, label) in enumerate(FIRST_KINDS):
        t = [s.get((side, 1, length, d, j), {}).get(f"first_{kind}_s", np.nan) for j in (1, -1)]
        ax.bar(np.arange(2) + (i - 2) * width, t, width, label=label, color=f"C{i}")
    ax.set_xticks(range(2), ["J = +1", "J = -1"])
    ax.set_yscale("log")
    ax.set_ylabel("seconds to the first sweep's end")
    ax.set_title(f"First result: setup + first sweep, L = {length}, D = {d}, 1 thread")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.savefig(FIGURES / "dmrg2_first_run.png", dpi=150, bbox_inches="tight")


def threads_figure(s, j, length=96):
    if not any(k[2] == length and k[4] == j and "sweep_s" in v for k, v in s.items()):
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for side, style in STYLE.items():
        for d, color in D_COLORS.items():
            pts = [(t, s[k]["sweep_s"]) for t in THREADS if "sweep_s" in s.get(k := (side, t, length, d, j), {})]
            if pts:
                ax.plot(*zip(*pts), color=color, **style)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(THREADS, [str(t) for t in THREADS])
    ax.minorticks_off()
    ax.set_xlabel("threads")
    ax.set_ylabel("seconds per sweep (median of sweeps 2-4)")
    ax.set_title(f"DMRG2 per-sweep wall time, L = {length}, J = {j:+d}")
    colors = [Line2D([], [], color=c, lw=2, label=f"D = {d}") for d, c in D_COLORS.items()]
    sides = [Line2D([], [], color="black", label=side, **st) for side, st in STYLE.items()]
    first = ax.legend(handles=colors, title="bond dimension", fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.add_artist(first)
    ax.legend(handles=sides, title="library", fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(FIGURES / f"dmrg2_threads_J{j:+d}.png", dpi=150, bbox_inches="tight")


def write_summary(s, path):
    fields = sorted({f for v in s.values() for f in v})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "threads", "L", "D", "J"] + fields)
        for k in sorted(s):
            w.writerow(list(k) + [s[k].get(x, "") for x in fields])


if __name__ == "__main__":
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else APP / "bench" / "results"
    paths = sorted(p for p in results.glob("*.csv") if p.name != "summary.csv")
    if not paths:
        sys.exit(f"no CSV in {results}; run golden-chain/bench/run.sh first")
    FIGURES.mkdir(parents=True, exist_ok=True)
    s = summarize([r for p in paths for r in rows(p)])
    write_summary(s, results / "summary.csv")
    first_run_table(s)
    for threads in sorted({k[1] for k in s}):
        warm_tables(s, threads)
    first_run_figure(s)
    for j in (1, -1):
        threads_figure(s, j)

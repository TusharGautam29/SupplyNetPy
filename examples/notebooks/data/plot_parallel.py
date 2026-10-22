"""Plots for the parallel speedup benchmark (reads parallel_raw.csv from benchmark_parallel.py).

Writes three PNGs next to this file:
  parallel_speedup.png   speedup over the one-process run vs worker count, one line per network size
  parallel_time.png      wall-clock time vs network size: one process vs the best parallel run
  parallel_profit.png    profit of every run relative to the one-process mean (model fidelity)

In the first two, each point is the mean of the repeats; whiskers show the lowest
and highest repeat.

Run:
  python plot_parallel.py
"""
import csv
import os
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed series order: smallest N first
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def load():
    runs = defaultdict(list)  # (N, workers) -> rows
    with open(os.path.join(HERE, "parallel_raw.csv")) as f:
        for row in csv.DictReader(f):
            runs[int(row["N"]), int(row["workers"])].append(
                {"time": float(row["time_s"]), "profit": float(row["profit"])})
    return runs


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def errorbar(ax, x, values, color, label):
    mids = [mean(v) for v in values]
    low = [m - min(v) for m, v in zip(mids, values)]
    high = [max(v) - m for m, v in zip(mids, values)]
    ax.errorbar(x, mids, yerr=[low, high], color=color, linewidth=2, marker="o", markersize=6,
                capsize=3, label=label)
    return mids


def main():
    runs = load()
    sizes = sorted({n for n, _ in runs})
    workers = sorted({k for _, k in runs if k > 0})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(workers, workers, color=MUTED, linestyle="--", linewidth=1, label="ideal (speedup = workers)")
    top = 0
    for color, n in zip(COLORS, sizes):
        serial = mean(r["time"] for r in runs[n, 0])
        speedups = [[serial / r["time"] for r in runs[n, k]] for k in workers]
        errorbar(ax, workers, speedups, color, f"N = {n:,}")
        top = max(top, max(max(s) for s in speedups))
    ax.set_ylim(0, top * 1.4)  # the ideal line runs off the top; the measured lines stay readable
    ax.set_xticks(workers)
    style(ax, "Speedup over the one-process run", "worker processes", "speedup (×)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "parallel_speedup.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    errorbar(ax, sizes, [[r["time"] for r in runs[n, 0]] for n in sizes], COLORS[0], "one process")
    best = [min(workers, key=lambda k: mean(r["time"] for r in runs[n, k])) for n in sizes]
    errorbar(ax, sizes, [[r["time"] for r in runs[n, k]] for n, k in zip(sizes, best)], COLORS[1],
             "best parallel run")
    for n, k in zip(sizes, best):
        ax.annotate(f"{k} workers", (n, mean(r["time"] for r in runs[n, k])), textcoords="offset points",
                    xytext=(8, -14), ha="left", color=MUTED, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_xticks(sizes)
    style(ax, "Wall-clock time to simulate 500 time units", "network size N (nodes)", "seconds")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "parallel_time.png"), dpi=150)

    # One group per network size: the one-process repeats (their own spread) beside
    # every parallel run (all worker counts), each as % away from the one-process mean.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color=MUTED, linewidth=1)
    for i, n in enumerate(sizes):
        base = mean(r["profit"] for r in runs[n, 0])
        one = [100 * (r["profit"] - base) / abs(base) for r in runs[n, 0]]
        par = [100 * (r["profit"] - base) / abs(base) for k in workers for r in runs[n, k]]
        ax.plot([i - 0.15] * len(one), one, linestyle="none", marker="o", markersize=7, color=MUTED,
                label="one process" if i == 0 else None)
        ax.plot([i + 0.15] * len(par), par, linestyle="none", marker="o", markersize=7, color=COLORS[0],
                alpha=0.6, label="parallel (all worker counts)" if i == 0 else None)
        ax.plot([i - 0.25, i - 0.05], [mean(one)] * 2, color=INK, linewidth=2)  # zero by construction
        ax.plot([i + 0.05, i + 0.25], [mean(par)] * 2, color=INK, linewidth=2)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"N = {n:,}" for n in sizes])
    style(ax, "Profit of each run, vs the one-process mean (random network)", "",
          "difference in profit (%)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "parallel_profit.png"), dpi=150)
    print("wrote parallel_speedup.png, parallel_time.png, parallel_profit.png")


if __name__ == "__main__":
    main()

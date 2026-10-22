"""Parallel speedup benchmark for SupplyNetPy (PDES phase 6).

Runs the benchmark network from benchmark_performance.py (fan-in version: each
node buys from 3 upstream nodes) in one process, the way users run it today,
and then split over several processes with simulate_sc_net_parallel. Each
parallel configuration is repeated REPS times, the one-process run
SERIAL_REPS times. Parallel times include starting the
worker processes and building the pieces in them.

The parallel model is slightly different: the links between pieces get a
message delay (cut_delay). The networks are also random (exponential lead
times and demand), and each process draws from its own random generator, so
the parallel results match the one-process result only statistically.

Writes parallel_raw.csv (one row per run; workers = 0 means the one-process
run, which is timed inside a fresh process just like the parallel workers).
plot_parallel.py turns it into plots.

Run:
  python benchmark_parallel.py                  # full sweep (a few minutes)
  python benchmark_parallel.py 1000 4 8         # one N, chosen worker counts
"""
import csv
import functools
import multiprocessing
import os
import random
import sys
import time

import SupplyNetPy.Components as scm

from benchmark_performance import build_network

SIM_TIME = 500
CUT_DELAY = 1
FANIN = 3
REPS = 3
SERIAL_REPS = 15
SIZES = [250, 500, 1000, 2000]
WORKERS = [1, 2, 4, 8, 12, 16]
HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["N", "workers", "rep", "time_s", "profit", "cut_links", "windows"]


def _serial(n, rep):
    # Runs in a fresh process, like the parallel workers: a long-lived parent
    # process carries leftovers from earlier runs that made it slower and noisier.
    random.seed(rep)
    net = scm.create_sc_net(*build_network(n, FANIN))
    t0 = time.perf_counter()
    scm.simulate_sc_net(net, SIM_TIME, logging=False)
    return time.perf_counter() - t0, net["profit"]


def run(sizes, workers):
    rows = []
    ctx = multiprocessing.get_context("spawn")
    for n in sizes:
        factory = functools.partial(build_network, n, FANIN)
        # More one-process repeats than parallel ones: they set the model's own
        # run-to-run spread of profit, which the parallel profits are judged against.
        with ctx.Pool(1, maxtasksperchild=1) as pool:  # a fresh process for every run
            for rep, (took, profit) in enumerate(pool.starmap(_serial, [(n, r) for r in range(SERIAL_REPS)])):
                rows.append({"N": n, "workers": 0, "rep": rep, "time_s": took,
                             "profit": profit, "cut_links": 0, "windows": 0})
                print(rows[-1], flush=True)
        for rep in range(REPS):
            for k in workers:
                t0 = time.perf_counter()
                par = scm.simulate_sc_net_parallel(factory, SIM_TIME, n_parts=k, cut_delay=CUT_DELAY, logging=False)
                rows.append({"N": n, "workers": k, "rep": rep, "time_s": time.perf_counter() - t0,
                             "profit": par["profit"], "cut_links": len(par["parallel"]["cut_links"]),
                             "windows": par["parallel"]["windows"]})
                print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        run([int(sys.argv[1])], [int(k) for k in sys.argv[2:]])
    else:
        rows = run(SIZES, WORKERS)
        path = os.path.join(HERE, "parallel_raw.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")

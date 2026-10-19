#!/usr/bin/env python3
"""Timing of `ml fit --model boost` (docs/models-boost.md section 10): the minimum of N runs, with the machine, the load and the settings printed. Not a gate.

    python3 scripts/bench_boost.py [--rows 100000] [--cols 20] [--rounds 50] [--runs 5] [--sklearn]

The data are seeded (numpy, seed 7): a regression target that is a nonlinear function of the columns plus noise, or a noisy threshold of it for a classification, with 5 percent of
the cells of two columns missing. `ml` is timed as a process (`ml fit`, which reads the CSV, bins, fits and writes the file); scikit-learn (with `--sklearn`) is timed on `fit` alone
with the matrix already in memory and OMP_NUM_THREADS=1: the two do not do the same work, so the comparison favours scikit-learn by the CSV read, and **the algorithms differ**
(section 8.1 of the document says how), so a ratio is the ratio of these two programs on these inputs and nothing more."""

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
BIN = os.environ.get("ML_BIN", str(ROOT / "build" / "ml"))


def data(n, d, task):
    rng = np.random.RandomState(7)
    X = rng.randn(n, d)
    z = X[:, 0] * 2 + np.sin(X[:, 1] * 2) + X[:, 2] * X[:, 3] + 0.5 * (X[:, 4] > 0) + rng.randn(n) * 0.5
    y = z if task == "regression" else (z > np.median(z)).astype(int)
    M = np.zeros((n, d), dtype=bool)
    for c in (1, 2):
        M[rng.rand(n) < 0.05, c] = True
    return X, y, M


def write_csv(path, X, y, M, task):
    n, d = X.shape
    with open(path, "w") as f:
        f.write(",".join("f%d" % j for j in range(d)) + ",y\n")
        for i in range(n):
            f.write(",".join("" if M[i, j] else "%.4f" % X[i, j] for j in range(d)) + "," + ("%.4f" % y[i] if task == "regression" else str(y[i])) + "\n")


def load():
    try:
        return os.getloadavg()
    except OSError:
        return (0, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100000)
    ap.add_argument("--cols", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--sklearn", action="store_true")
    a = ap.parse_args()
    print("machine %s %s, %d cpus, load average before %.2f %.2f %.2f; %d rows x %d columns, %d rounds, depth %d, minimum of %d runs" % (
        os.uname().sysname, os.uname().machine, os.cpu_count(), *load(), a.rows, a.cols, a.rounds, a.depth, a.runs))
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for task in ("regression", "classification"):
            X, y, M = data(a.rows, a.cols, task)
            write_csv(tmp / "d.csv", X, y, M, task)
            for threads in (1, 8):
                best = None
                for _ in range(a.runs):
                    t0 = time.perf_counter()
                    p = subprocess.run([BIN, "--root", str(tmp), "fit", "--model", "boost", "--task", task, "--target", "y", "--rounds", str(a.rounds), "--max-depth", str(a.depth),
                                        "--min-leaf", "10", "--bins", "64", "--seed", "1", "--threads", str(threads), "--output", "m.cml", "--overwrite", "d.csv"], capture_output=True)
                    dt = time.perf_counter() - t0
                    assert p.returncode == 0, p.stdout[:300]
                    best = dt if best is None else min(best, dt)
                print("ml fit boost %-14s threads %d: %.3f s (process, CSV read and file written included)" % (task, threads, best))
            if a.sklearn:
                os.environ["OMP_NUM_THREADS"] = "1"
                from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
                Xn = np.where(M, np.nan, X)
                kw = dict(max_iter=a.rounds, learning_rate=0.1, max_depth=a.depth, max_leaf_nodes=None, min_samples_leaf=10, l2_regularization=1.0, max_bins=64, early_stopping=False, random_state=0)
                best = None
                for _ in range(a.runs):
                    m = (HistGradientBoostingRegressor if task == "regression" else HistGradientBoostingClassifier)(**kw)
                    t0 = time.perf_counter()
                    m.fit(Xn, y)
                    dt = time.perf_counter() - t0
                    best = dt if best is None else min(best, dt)
                print("scikit-learn HistGradientBoosting %-14s (1 OpenMP thread, fit only): %.3f s" % (task, best))
    print("load average after %.2f %.2f %.2f" % load())


if __name__ == "__main__":
    sys.exit(main())

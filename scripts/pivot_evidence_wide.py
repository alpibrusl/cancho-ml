#!/usr/bin/env python3
"""The measurement behind the claim that the relative pivot test of docs/models-linear.md 3.1 needs no tolerance that depends on the size of the matrix (docs/refusals-fit.md D1,
6.1): the smallest relative pivot s_jj / a_jj of the matrix linear regression factors, at the sizes the estimators take (p = d + 1 up to 257 columns, the intercept included),
for data that must fit and for exactly dependent columns whose relative pivot is rounding noise. The factorisation is `oracle_linalg.chol_o`, the Python twin of lib/linalg/chol.cho
that the gates check bit for bit; the sums are `oracle_linalg.syrk_o`. Nothing here is a gate: the numbers are quoted in the document with the seed.
usage: pivot_evidence_wide.py [--seed N]"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_linalg as ol  # noqa: E402

TOL = ol.TOL
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 20261015


def rel_pivots(X):
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    C = ol.syrk_o(Xa)
    G = np.triu(C) + np.triu(C, 1).T
    piv = []
    ol.chol_o(G, piv)
    return [s / a if a != 0.0 else (0.0 if s == 0.0 else float("inf")) for s, a in piv]


rng = np.random.default_rng(SEED)
print("tol = 2^-40 = %.4e, seed %d, numpy %s" % (TOL, SEED, np.__version__))
print("\n-- A. well conditioned: standard normal features, the smallest relative pivot over 20 draws (all columns, the intercept included)")
print("%5s %6s %14s %12s" % ("d", "n", "smallest s/a", "over tol"))
for d in (8, 32, 64, 128, 256):
    for n in (500, 5000):
        if n < 2 * d:
            continue
        worst = min(min(rel_pivots(rng.standard_normal((n, d)))) for _ in range(20 if d <= 64 else 5))
        print("%5d %6d %14.3e %11.1ex" % (d, n, worst, worst / TOL))
print("\n-- B. exactly dependent: the last feature is a random combination of all the others (dependent in exact arithmetic; the pivot of the last column is rounding noise)")
print("%5s %6s %9s %16s %16s %10s" % ("d", "n", "fits", "largest s/a", "smallest s/a", "|noise| x"))
worst_noise = 0.0
for d in (3, 8, 32, 64, 128, 256):
    for n in (500, 5000):
        if n < 2 * d:
            continue
        hi, lo, k = -1.0, 1.0, 0
        for _ in range(10 if d <= 64 else 3):
            X = rng.standard_normal((n, d))
            X[:, d - 1] = X[:, :d - 1] @ rng.uniform(-2, 2, d - 1)
            r = rel_pivots(X)[d - 1]
            hi, lo, k = max(hi, r), min(lo, r), k + 1
        m = max(abs(hi), abs(lo))
        worst_noise = max(worst_noise, m)
        print("%5d %6d %9d %16.3e %16.3e %9.1ex" % (d, n, k, hi, lo, m / TOL))
print("\nthe largest |noise| over all sizes: %.3e = %.1e x tol (tol is %.1ex above it)" % (worst_noise, worst_noise / TOL, TOL / worst_noise))
print("\n-- C. the intercept as the dependent column: a constant column (or the full one-hot block) beside it; the pivot of the intercept")
for d in (1, 8, 64, 256):
    n = 2000
    X = rng.standard_normal((n, d))
    X[:, d - 1] = 4.0
    r = rel_pivots(X)[d]
    print("d = %3d  n = %d  relative pivot of the intercept %+.3e  (%s)" % (d, n, r, "refused" if not r > TOL else "ACCEPTED"))

#!/usr/bin/env python3
"""The measurement behind the tolerance of the relative pivot test (docs/models-linear.md 3.1, docs/lanes.md section 3): the smallest relative
Cholesky pivot s_jj / a_jj of the normal equations of linear regression (Gram matrix of the features and a column of ones, the matrix that
`lib/models/linear/linreg.cho` factors), for data that must fit and for data that must be refused. The factorisation is `oracle_linalg.chol_o`, the
Python re-implementation of lib/linalg/chol.cho that the gates check bit for bit; the sums are `oracle_linalg.syrk_o`. Nothing here is a gate: the
numbers it prints are quoted in the document, with the seed. usage: pivot_evidence.py [--seed N]"""
import sys, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_linalg as ol

TOL = ol.TOL
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 20261013


def rel_pivots(X, l2=0.0):
    """relative pivots s_jj / a_jj of [X | 1]'[X | 1] (+ l2 on the feature diagonal), up to and including the first that is not positive"""
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    C = ol.syrk_o(Xa)
    G = np.triu(C) + np.triu(C, 1).T
    for j in range(d):
        G[j, j] += l2
    piv = []
    ol.chol_o(G, piv)
    return [s / a if a != 0.0 else (0.0 if s == 0.0 else float("inf")) for s, a in piv]


def standardise(X):
    X = X - X.mean(axis=0)
    return X / X.std(axis=0)


def row(label, rp):
    m = min(rp)
    print("%-62s min s/a = %+.3e  (%s)" % (label, m, "accepted" if all(r > TOL for r in rp) and len(rp) else "REFUSED"))
    return m


rng = np.random.default_rng(SEED)
print("tol = 2^-40 = %.4e, seed %d, n = 1000 unless said" % (TOL, SEED))
print("\n-- A. well-conditioned data (uniform features, the generator of the gates), minimum over 50 draws")
for d in (1, 4, 20):
    worst = min(min(rel_pivots(ol.elements(ol.derive(s, 1), 1000 * d).reshape(1000, d))) for s in range(1, 51))
    print("d = %-2d  smallest relative pivot over 50 seeds: %.3e   (margin over tol: %.1e x)" % (d, worst, worst / TOL))
print("\n-- B. standardised pair with correlation rho (legitimate, nearly dependent): 20 draws each, the smallest")
for rho in (0.9, 0.999, 0.999999, 1 - 1e-9, 1 - 1e-10, 1 - 1e-11, 1 - 1e-12, 1 - 1e-13):
    vals = []
    for _ in range(20):
        z = rng.standard_normal((1000, 2))
        z = standardise(z)
        # make the two sample columns exactly uncorrelated first (Gram-Schmidt), so the sample correlation is rho
        z[:, 1] = z[:, 1] - z[:, 0] * (z[:, 0] @ z[:, 1]) / (z[:, 0] @ z[:, 0])
        z = standardise(z)
        x1 = rho * z[:, 0] + np.sqrt(1 - rho * rho) * z[:, 1]
        vals.append(min(rel_pivots(np.column_stack([z[:, 0], x1]))))
    r = min(vals)
    print("rho = 1 - %.0e   smallest relative pivot %.3e   expected 1 - rho^2 = %.3e   %s" % (1 - rho, r, 1 - rho * rho, "accepted" if r > TOL else "REFUSED"))
print("\n-- C. dependent columns with l2 = 0: the relative pivot is rounding noise")
for label, mk, dep in (("two equal columns, uniform", lambda n, s: (lambda E: np.column_stack([E, E]))(ol.elements(ol.derive(s, 1), n).reshape(n, 1)), 1),
                       ("two equal columns, standardised normal", lambda n, s: (lambda E: np.column_stack([E, E]))(standardise(rng.standard_normal((n, 1)))), 1),
                       ("a column and 3 times it, uniform", lambda n, s: (lambda E: np.column_stack([E, 3.0 * E]))(ol.elements(ol.derive(s, 1), n).reshape(n, 1)), 1),
                       ("a column, another, and their sum, normal", lambda n, s: (lambda E: np.column_stack([E, E[:, 0] + E[:, 1]]))(rng.standard_normal((n, 2))), 2)):
    best, worst, pos, tot = -1.0, 1.0, 0, 0
    for n in (7, 60, 255, 1000, 5000, 20000):
        for s in range(1, 41):
            r = rel_pivots(mk(n, s))[dep]  # the pivot of the dependent column (the first pivots are those of the independent ones)
            tot += 1
            pos += r > 0
            best, worst = max(best, r), min(worst, r)
    print("%-48s %4d fits: largest relative pivot %+.3e, smallest %+.3e; positive in %d (the old test `s > 0` accepts those)" % (label, tot, best, worst, pos))
print("an all-zero column:")
X = ol.elements(ol.derive(5, 1), 4000).reshape(1000, 4)
X[:, 2] = 0.0
rp = rel_pivots(X)
print("   relative pivots %s  (the pivot is exactly 0.0 and a_jj = 0.0)" % ["%.3e" % r for r in rp])
print("\n-- D. one feature with a large mean: the intercept's pivot is (spread / mean)^2 of the feature, relative to n")
for ratio in (1e2, 1e3, 1e4, 1e5, 5e5, 1e6, 1e7):
    x = ratio + rng.standard_normal((1000, 1))
    r = min(rel_pivots(x))
    print("mean / sd = %.0e   smallest relative pivot %.3e   %s" % (ratio, r, "accepted" if r > TOL else "REFUSED"))
print("\n-- E. the same equal columns with a ridge: the smallest relative pivot is about l2 / a_jj")
E = standardise(rng.standard_normal((1000, 1)))
for l2 in (0.0, 1e-9, 1e-6, 1e-3, 1.0):
    r = min(rel_pivots(np.column_stack([E, E]), l2))
    print("l2 = %.0e  (a_jj = n = 1000 + l2)  smallest relative pivot %+.3e   %s" % (l2, r, "accepted" if r > TOL else "REFUSED"))

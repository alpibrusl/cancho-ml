#!/usr/bin/env python3
"""C1a (the scikit-learn half of gate WG5, docs/weights.md section 5) -- the weighted linear and logistic fits of lib/models against `Ridge` and `LogisticRegression` with
`sample_weight`, with the bounds fixed BEFORE this script was run:

    linear     every coefficient and the intercept within 1e-9 relative (max |a - b| / max(1, |b|)) of Ridge(alpha=l2, solver="cholesky").fit(sample_weight=w)
    logistic   every coefficient and the intercept within 1e-6 relative of LogisticRegression(C = 1 / (l2 * sum(w)), tol=1e-12, lbfgs).fit(sample_weight=w): ml minimises
               (1 / sum w) sum_i w_i loss_i + (l2 / 2) |coef|^2, scikit-learn C sum_i w_i loss_i + |coef|^2 / 2, the same function up to a factor

The data and the weights are those of the driver (scripts/oracle_models.py gen, gen_weights), so ml and scikit-learn fit the same bytes. The control: scikit-learn with the weights
dropped must exceed the bound on every case (a gate that cannot fail judges nothing). Exit 1 when a bound is exceeded or the control stays inside it; the bounds are never widened.

    MODELS_CHECK=build/gate-models/models-check python3 scripts/gate_weights_sklearn.py
"""
import os, sys, pathlib, warnings
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_models as om

LIN_BOUND, LOG_BOUND = 1e-9, 1e-6
# (n, d, seed, l2 in thousandths, wt kind)
LIN = [(1000, 5, 7, 500, 3), (4097, 8, 10, 1000, 4), (2000, 3, 21, 100, 2), (20000, 10, 12, 250, 3)]
LOG = [(1000, 5, 6, 2000, 3, 12), (4097, 8, 10, 1000, 4, 12), (2000, 3, 21, 500, 2, 12), (20000, 10, 12, 100, 3, 12)]


def maxrel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))


def driver(model, n, d, seed, l2m, wt, iters=None):
    a = dict(n=n, d=d, seed=seed, l2m=l2m, wt=wt)
    if iters:
        a["iters"] = iters
    got = om.run_driver(model, a, 1)
    return np.array([int(x, 16) for x in got["params"].split()], dtype=np.uint64).view(np.float64)


def main():
    ok = True
    for (n, d, seed, l2m, wt) in LIN:
        X, y = om.gen("linreg", n, d, seed)
        w = om.gen_weights(seed, n, wt)
        p = driver("linreg", n, d, seed, l2m, wt)
        r = Ridge(alpha=l2m / 1000.0, solver="cholesky").fit(X, y, sample_weight=w)
        ref = np.append(r.coef_, r.intercept_)
        c = Ridge(alpha=l2m / 1000.0, solver="cholesky").fit(X, y)
        ctl = np.append(c.coef_, c.intercept_)
        e, ce = maxrel(p, ref), maxrel(p, ctl)
        good = e <= LIN_BOUND and ce > LIN_BOUND
        ok &= good
        print("linear   n=%-6d d=%-2d wt=%d  rel diff %.3g (bound %.0e)  control %.3g  %s" % (n, d, wt, e, LIN_BOUND, ce, "ok" if good else "FAIL"))
    for (n, d, seed, l2m, wt, iters) in LOG:
        X, y = om.gen("logreg", n, d, seed)
        w = om.gen_weights(seed, n, wt)
        l2 = l2m / 1000.0
        p = driver("logreg", n, d, seed, l2m, wt, iters)
        C = 1.0 / (l2 * w.sum())
        r = LogisticRegression(C=C, tol=1e-12, max_iter=20000).fit(X, y, sample_weight=w)
        ref = np.append(r.coef_[0], r.intercept_[0])
        c = LogisticRegression(C=C * w.sum() / n, tol=1e-12, max_iter=20000).fit(X, y)
        ctl = np.append(c.coef_[0], c.intercept_[0])
        e, ce = maxrel(p, ref), maxrel(p, ctl)
        good = e <= LOG_BOUND and ce > LOG_BOUND
        ok &= good
        print("logistic n=%-6d d=%-2d wt=%d  rel diff %.3g (bound %.0e)  control %.3g  %s" % (n, d, wt, e, LOG_BOUND, ce, "ok" if good else "FAIL"))
    print("gate_weights_sklearn:", "ok" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

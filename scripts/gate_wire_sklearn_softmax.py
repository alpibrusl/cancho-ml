#!/usr/bin/env python3
"""M3 (the scikit-learn half) -- `ml --model softmax` against scikit-learn's multinomial `LogisticRegression`, with the bounds fixed BEFORE this script was run (docs/multiclass.md section 8).

Data: seeded, the class of a row drawn from the softmax of a known linear score (so no problem is separable), 3,000 rows, 5 features, K = 3, 3, 4, 5, 7 and 10 classes, the labels
integers that are not 0 .. K - 1. Both libraries fit without a penalty: `ml` with l2 = 0 and 100 Newton iterations, scikit-learn with penalty=None, lbfgs, tol 1e-12 and 20,000
iterations. The two parametrise the same model differently (a reference class with zero coefficients against K symmetric sets), so what is compared is what both determine, the
probabilities of the K classes of every row, and the metrics:

    probabilities   |p_ml - p_sk| at most 1e-6 for every row and class
    labels          equal on every row whose largest and second largest probability differ by more than 1e-6
    log-loss        relative 1e-6 of scikit-learn's

The control: scikit-learn with the L2 penalty it applies by default (C = 1.0), which is not the fit above, must exceed the probability bound on at least one of the cases (a gate that
cannot fail judges nothing). Exit 1 when a bound is exceeded or the control stays inside it; the bounds are never widened here.

    python3 scripts/gate_wire_sklearn_softmax.py
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
BIN = os.environ.get("ML_BIN", str(ROOT / "build" / "ml"))
N, D = 3000, 5
CASES = [(1, 3), (2, 3), (3, 4), (4, 5), (5, 7), (6, 10)]
P_BOUND, GAP, LL_BOUND = 1e-6, 1e-6, 1e-6


def data(seed, k):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, D)) * np.array([1, 2, 0.5, 1, 1.5])
    W = rng.normal(0, 0.8, (D, k))
    z = X @ W
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    cls = np.array([rng.choice(k, p=row) for row in p])
    labels = np.sort(rng.choice(np.arange(-5, 40), size=k, replace=False))
    return X, labels[cls], labels


def ml(*args):
    r = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True)
    return json.loads(r.stdout)


def run(seed, k, tmp):
    X, y, labels = data(seed, k)
    with open(tmp / "d.csv", "w") as f:
        f.write(",".join("f%d" % j for j in range(D)) + ",y\n")
        for row, t in zip(X, y):
            f.write(",".join(repr(float(v)) for v in row) + ",%d\n" % t)
    r = ml("--root", tmp, "fit", "--model", "softmax", "--target", "y", "--iterations", "100", "--output", "m.cml", "--overwrite", "d.csv")
    if not r.get("ok"):
        return None, "ml refused: %s" % json.dumps(r)[:300]
    q = ml("--root", tmp, "predict", "--model-file", "m.cml", "--proba", "--limit", str(N), "d.csv")
    pm = np.array([[float(v) for v in cell.values()] for _, cell in q["data"]["rows"]])
    lm = np.array([int(lab) for lab, _ in q["data"]["rows"]])
    out = {"labels": labels, "y": y, "pm": pm, "lm": lm}
    sk = LogisticRegression(penalty=None, solver="lbfgs", tol=1e-12, max_iter=20000).fit(X, y)
    assert list(sk.classes_) == list(labels), (sk.classes_, labels)
    out["ps"] = sk.predict_proba(X)
    out["ls"] = sk.predict(X)
    out["pc"] = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", tol=1e-12, max_iter=20000).fit(X, y).predict_proba(X)
    return out, ""


def main():
    bad = 0
    control_exceeds = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        print("scikit-learn %s, numpy %s, %s" % (__import__("sklearn").__version__, np.__version__, BIN))
        for seed, k in CASES:
            out, why = run(seed, k, tmp)
            if out is None:
                print("K=%-2d seed %d: %s" % (k, seed, why))
                bad += 1
                continue
            dp = float(np.max(np.abs(out["pm"] - out["ps"])))
            top2 = np.sort(out["ps"], axis=1)[:, -2:]
            sure = (top2[:, 1] - top2[:, 0]) > GAP
            dl = int(np.sum((out["lm"] != out["ls"]) & sure))
            ll_m = log_loss(out["y"], out["pm"], labels=out["labels"])
            ll_s = log_loss(out["y"], out["ps"], labels=out["labels"])
            rel = abs(ll_m - ll_s) / ll_s
            ctrl = float(np.max(np.abs(out["pm"] - out["pc"])))
            control_exceeds += ctrl > P_BOUND
            ok = dp <= P_BOUND and dl == 0 and rel <= LL_BOUND
            bad += not ok
            print("K=%-2d seed %d: max|dp| %.2e (bound %.0e)  label differences on sure rows %d  log-loss relative %.2e (bound %.0e)  control (C=1) max|dp| %.2e  %s"
                  % (k, seed, dp, P_BOUND, dl, rel, LL_BOUND, ctrl, "ok" if ok else "EXCEEDED"))
    if control_exceeds == 0:
        print("the control stayed inside the bound on every case: the gate cannot fail")
        bad += 1
    print("gate_wire_sklearn_softmax: %s" % ("all bounds hold, the control exceeds the probability bound on %d of %d cases" % (control_exceeds, len(CASES)) if not bad else "%d problem(s)" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

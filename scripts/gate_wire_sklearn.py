#!/usr/bin/env python3
"""W4 -- the end-to-end result of `ml` against scikit-learn, with the tolerances fixed in docs/wiring.md section 7 BEFORE this script was run.

Data: seeded, 2,000 rows, 6 features, the target a known linear rule plus noise (regression) or the sign of one (classification), with 5 percent of the cells of two
columns missing; both libraries impute the mean and standardise (ddof 0). `LinearRegression` against `ml --model linear`, `LogisticRegression` against
`ml --model logistic` (no penalty: l2 = 0 against C=inf, Newton-CG to 1e-12 against 100 Newton iterations). Compared: the coefficients in the standardised space,
the intercept, the RMSE (regression), the ROC-AUC and the log-loss (classification), each with its own bound:

    linear coefficients and intercept  relative 1e-9 of the largest coefficient     rmse relative 1e-9
    logistic coefficients/intercept    relative 1e-5 of the largest coefficient     roc_auc absolute 1e-6     log_loss relative 1e-5

Exit 1 when a bound is exceeded (the bounds are never widened here; a failure is reported as it is).

    python3 scripts/gate_wire_sklearn.py [model]
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
BIN = os.environ.get("ML_BIN", str(ROOT / "build" / "ml"))
N, D = 2000, 6
BOUNDS = {"linear": {"coef": 1e-9, "rmse": 1e-9}, "logistic": {"coef": 1e-5, "auc": 1e-6, "logloss": 1e-5}}


def data(seed, model):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, D)) * np.array([1, 3, 0.5, 10, 2, 1]) + np.array([0, 5, -2, 100, 1, 0])
    beta = np.array([1.5, -0.7, 2.0, 0.05, -1.0, 0.3])
    z = X @ beta + rng.normal(0, 1.0, N)
    y = z if model == "linear" else (z + rng.logistic(0, 1, N) > np.median(z)).astype(int)
    M = X.copy()
    for j in (1, 4):
        M[rng.random(N) < 0.05, j] = np.nan
    return M, y


def write_csv(path, M, y, target):
    with open(path, "w") as f:
        f.write(",".join("f%d" % j for j in range(D)) + "," + target + "\n")
        for row, t in zip(M, y):
            f.write(",".join("" if np.isnan(v) else repr(float(v)) for v in row) + ",%s\n" % (repr(float(t)) if target == "y" else int(t)))


def ml(*args):
    p = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True)
    return json.loads(p.stdout)


def relmax(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def gate(model, seed):
    M, y = data(seed, model)
    imp = SimpleImputer(strategy="mean")
    Xs = StandardScaler().fit_transform(imp.fit_transform(M))
    with tempfile.TemporaryDirectory() as td:
        write_csv(td + "/d.csv", M, y, "y" if model == "linear" else "lab")
        steps = "impute column=f1,f4 strategy=mean;scale column=* kind=standard"
        tgt = "y" if model == "linear" else "lab"
        extra = ["--iterations", "100"] if model == "logistic" else []
        d = ml("--root", td, "fit", "--model", model, "--step", steps, "--target", tgt, "--output", "m.cml", *extra, "d.csv")
        assert d["ok"], d
        s = d["data"]["summary"]
        w = [float(c["value"]) for c in s["coefficients"]]
        b = float(s["intercept"])
        e = ml("--root", td, "eval", "--model-file", "m.cml", "d.csv")
        assert e["ok"], e
        m = e["data"]["metrics"]
    rows = []
    bounds = BOUNDS[model]
    if model == "linear":
        sk = LinearRegression().fit(Xs, y)
        coef = relmax(list(w) + [b], list(sk.coef_) + [sk.intercept_])
        rmse = abs(float(m["rmse"]) - mean_squared_error(y, sk.predict(Xs)) ** 0.5) / (mean_squared_error(y, sk.predict(Xs)) ** 0.5)
        rows = [("coefficients+intercept", coef, bounds["coef"]), ("rmse", rmse, bounds["rmse"])]
    else:
        sk = LogisticRegression(C=np.inf, solver="newton-cg", tol=1e-12, max_iter=1000).fit(Xs, y)
        coef = relmax(list(w) + [b], list(sk.coef_[0]) + [sk.intercept_[0]])
        p = sk.predict_proba(Xs)[:, 1]
        auc = abs(float(m["roc_auc"]) - roc_auc_score(y, p))
        ll = abs(float(m["log_loss"]) - log_loss(y, p)) / log_loss(y, p)
        rows = [("coefficients+intercept", coef, bounds["coef"]), ("roc_auc (absolute)", auc, bounds["auc"]), ("log_loss", ll, bounds["logloss"])]
    ok = True
    for name, err, bound in rows:
        verdict = "PASS" if err <= bound else "FAIL"
        ok &= err <= bound
        print("%-9s seed %d  %-24s observed %.3e  bound %.0e  %s" % (model, seed, name, err, bound, verdict))
    return ok


def main():
    models = sys.argv[1:] or ["linear", "logistic"]
    ok = True
    for model in models:
        for seed in (1, 2, 3):
            ok &= gate(model, seed)
    print("sklearn gate: %s" % ("all PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

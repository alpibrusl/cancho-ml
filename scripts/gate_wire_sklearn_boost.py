#!/usr/bin/env python3
"""GB4 -- `ml --model boost` against scikit-learn's HistGradientBoostingClassifier / Regressor on the shipped train/test files (docs/models-boost.md section 8.1).

The bounds below are the ones of section 8.1, written in that document before this script was first run; they are not changed to fit a result:

    test accuracy within 0.05 absolute, ROC-AUC within 0.03 absolute, log-loss within 15% relative, RMSE (diabetes) within 8% relative

Data: docs/examples/data/{breast_cancer,iris,wine,diabetes}_{train,test}.csv, the same two files to both programs. Settings:
ml --model boost --rounds 100 --learning-rate 0.1 --max-depth 3 --min-leaf 10 --bins 64 --l2 1.0 --seed 0 against
HistGradientBoosting*(max_iter=100, learning_rate=0.1, max_depth=3, max_leaf_nodes=None, min_samples_leaf=10, l2_regularization=1.0, max_bins=64, early_stopping=False, random_state=0).

WHAT IS ALIGNED: the loss, the base score, the Newton leaf times the learning rate, the number of rounds, the depth limit, the minimum rows per leaf, the L2 term, the number of
bins, a learned side for missing cells. WHAT IS NOT: the bins (scikit-learn puts an edge between distinct values or takes quantiles of a subsample; here an edge is a data value at
a fixed rank), the tie rule, the gain test, the growth order, the float summation order. So the trees are not the same trees, equality is not claimed and only the quality of the
result is compared. The report prints the metrics of both libraries.

The controls (they must be able to fail): C1, `--max-depth 1` against scikit-learn at depth 3, must fall outside a bound on at least one data set per task, or it is reported that
it cannot fail on these files (not a failure: the tree gate's C1 did the same); C2, the train target permuted (seed 0), must fall outside the accuracy bound (classification) or the
RMSE bound (regression) on every data set, or the gate fails.

    python3 scripts/gate_wire_sklearn_boost.py        exit 1 when a bound is exceeded, or when a control that must fail stays inside
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
BIN = os.environ.get("ML_BIN", str(ROOT / "build" / "ml"))
DATA = ROOT / "docs" / "examples" / "data"
SETS = [("breast_cancer", "benign", "classification"), ("iris", "virginica", "classification"), ("wine", "class_0", "classification"), ("diabetes", "progression", "regression")]
FLOAT = {"wine": "color_intensity"}
BOUND_ACC, BOUND_AUC, BOUND_LL, BOUND_RMSE = 0.05, 0.03, 0.15, 0.08


def load(name):
    out = []
    for part in ("train", "test"):
        lines = (DATA / ("%s_%s.csv" % (name, part))).read_text().strip().split("\n")
        rows = np.array([[float(c) for c in l.split(",")] for l in lines[1:]])
        out.append((lines[0].split(","), rows))
    return out


def ml(*args):
    p = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True)
    return json.loads(p.stdout)


def run_ml(name, target, task, depth, tmp, train="train"):
    fl = ["--float", FLOAT[name]] if name in FLOAT else []
    d = ml("--root", tmp, "fit", "--model", "boost", "--task", task, "--rounds", 100, "--learning-rate", "0.1", "--max-depth", depth, "--min-leaf", 10, "--bins", 64, "--l2", "1.0", *fl,
           "--seed", "0", "--target", target, "--output", "m.cml", "--overwrite", "%s_%s.csv" % (name, train))
    assert d["ok"], d
    e = ml("--root", tmp, "eval", "--model-file", "m.cml", "%s_test.csv" % name)
    assert e["ok"], e
    return {k: float(v) for k, v in e["data"]["metrics"].items()}


def run_sklearn(task, depth, Xtr, ytr, Xte, yte):
    kw = dict(max_iter=100, learning_rate=0.1, max_depth=depth, max_leaf_nodes=None, min_samples_leaf=10, l2_regularization=1.0, max_bins=64, early_stopping=False, random_state=0)
    if task == "classification":
        c = HistGradientBoostingClassifier(**kw)
        c.fit(Xtr, ytr.astype(int))
        p = c.predict_proba(Xte)[:, 1]
        return {"accuracy": accuracy_score(yte.astype(int), c.predict(Xte)), "roc_auc": roc_auc_score(yte.astype(int), p), "log_loss": log_loss(yte.astype(int), p)}
    r = HistGradientBoostingRegressor(**kw)
    r.fit(Xtr, ytr)
    return {"rmse": mean_squared_error(yte, r.predict(Xte)) ** 0.5}


def permute(tmp, name, target):
    lines = (DATA / ("%s_train.csv" % name)).read_text().strip().split("\n")
    head, rows = lines[0].split(","), [l.split(",") for l in lines[1:]]
    t = head.index(target)
    col = [r[t] for r in rows]
    np.random.RandomState(0).shuffle(col)
    for r, v in zip(rows, col):
        r[t] = v
    (tmp / ("%s_permuted.csv" % name)).write_text(lines[0] + "\n" + "".join(",".join(r) + "\n" for r in rows))


def compare(name, target, task, depth, tmp, report, train="train", only_primary=False):
    (names, tr), (_, te) = load(name)
    t = names.index(target)
    cols = [j for j in range(len(names)) if j != t]
    m = run_ml(name, target, task, depth, tmp, train)
    s = run_sklearn(task, 3, tr[:, cols], tr[:, t], te[:, cols], te[:, t])
    if task == "classification":
        rows = [("accuracy", abs(m["accuracy"] - s["accuracy"]), BOUND_ACC, m["accuracy"], s["accuracy"]), ("roc_auc", abs(m["roc_auc"] - s["roc_auc"]), BOUND_AUC, m["roc_auc"], s["roc_auc"]),
                ("log_loss (relative)", abs(m["log_loss"] - s["log_loss"]) / s["log_loss"], BOUND_LL, m["log_loss"], s["log_loss"])]
    else:
        rows = [("rmse (relative)", abs(m["rmse"] - s["rmse"]) / s["rmse"], BOUND_RMSE, m["rmse"], s["rmse"])]
    ok = True
    primary_ok = True
    for what, dist, bound, a, b in rows:
        within = dist <= bound
        ok &= within
        if what in ("accuracy", "rmse (relative)"):
            primary_ok &= within
        if report:
            print("boost  %-13s %-20s observed %.4f  bound %.2f  ml %.4f  scikit-learn %.4f  %s" % (name, what, dist, bound, a, b, "PASS" if within else "FAIL"))
    return primary_ok if only_primary else ok


def main():
    ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for f in DATA.glob("*.csv"):
            (tmp / f.name).write_bytes(f.read_bytes())
        n = 0
        for name, target, task in SETS:
            r = compare(name, target, task, 3, tmp, True)
            ok &= r
            n += 3 if task == "classification" else 1
        print("boost  %d comparisons" % n)
        for task in ("classification", "regression"):
            outside = [name for name, target, tk in SETS if tk == task and not compare(name, target, tk, 1, tmp, False)]
            print("boost  control C1 (depth 1, %s): outside the bounds on %s" % (task, ", ".join(outside) or "NO DATA SET (reported: this control cannot make the gate fail here)"))
        for name, target, tk in SETS:
            permute(tmp, name, target)
        outside = [name for name, target, tk in SETS if not compare(name, target, tk, 3, tmp, False, train="permuted", only_primary=True)]
        print("boost  control C2 (the train target permuted): outside the bounds on %s" % (", ".join(outside) or "NO DATA SET"))
        if sorted(outside) != sorted(name for name, _, _ in SETS):
            ok = False
            print("FAIL: a model trained on a permuted target stays inside the accuracy or RMSE bound on %s" % ", ".join(sorted(set(n for n, _, _ in SETS) - set(outside))))
    print("sklearn gate (boost): %s" % ("all PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""WT8 -- `ml --model tree|forest` against scikit-learn's DecisionTree and RandomForest on the shipped train/test files (docs/wiring-trees.md section 9).

The tolerances below are the ones of section 9, written in that document before this script was first run, and they are not changed to fit a result:

    forest  test accuracy within 0.03 absolute, ROC-AUC within 0.02 absolute, RMSE within 5% relative
    tree    test accuracy within 0.06 absolute, ROC-AUC within 0.06 absolute, RMSE within 12% relative

Data: docs/examples/data/{breast_cancer,iris,wine,diabetes}_{train,test}.csv, the same two files to both programs. Settings: ml --trees 100 --max-depth 8 --min-leaf 1
--bins 64 --seed 0 against RandomForest*(n_estimators=100, max_depth=8, min_samples_leaf=1, bootstrap=True, random_state=0, n_jobs=1; max_features sqrt for a
classification and 1/3 for a regression, the library's own defaults for --mtry 0), and ml --model tree --max-depth 8 against DecisionTree*(max_depth=8, random_state=0).
A different algorithm (exact thresholds, its own random generator, a float criterion), so only the quality of the result is compared.

The controls. C1, pre-registered (section 9): the same ml model at depth 1 against the reference at depth 8 must fall outside the bound on at least one data set per task,
"or the gate cannot fail and is reported as such". Its first run found that it does not (on these small, easy data a depth-1 forest of 100 stumps is as good as one of depth 8):
that is REPORTED here as a finding and is not a failure. C2 and C3 were added AFTER that run, for that reason, and are labelled as such: C2, a single stump (a tree of depth
1) against the forest reference; C3, the target column of the train file permuted (seed 0), so that the model learns nothing, against the reference. (A first C3, the
positive label swapped, was tried and dropped: the AUC of the other label scored with the other probability is the same number.) Each must be outside the bound on at least
one data set where it applies, or the gate fails. C3 is the one that separates the tree's bounds on these data; the stump alone does not (it is 6.5% from scikit-learn's tree
on diabetes, inside 12%).

The library's own pre-registered gate G-ref (docs/models-tree.md section 10.8) failed on its regression configuration R1 (8.1% against 3%): that result stands, and this
gate neither replaces it nor moves its tolerance.

    python3 scripts/gate_wire_sklearn_trees.py [tree|forest ...]       exit 1 when a bound is exceeded, or when the control stays inside
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
BIN = os.environ.get("ML_BIN", str(ROOT / "build" / "ml"))
DATA = ROOT / "docs" / "examples" / "data"
SETS = [("breast_cancer", "benign", "classification"), ("iris", "virginica", "classification"), ("wine", "class_0", "classification"), ("diabetes", "progression", "regression")]
# a column whose digits differ between the train and the test file is typed per file; --float at fit time is the fix docs/examples.md section 6 found
FLOAT = {"wine": "color_intensity"}
# model -> (accuracy, auc, rmse relative): fixed in docs/wiring-trees.md section 9 before the first run
BOUNDS = {"forest": (0.03, 0.02, 0.05), "tree": (0.06, 0.06, 0.12)}


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


def run_ml(model, name, target, task, depth, tmp, positive=None, train="train"):
    extra = ["--trees", "100", "--bins", "64", "--min-leaf", "1"] if model == "forest" else []
    fl = ["--float", FLOAT[name]] if name in FLOAT else []
    if positive is not None:
        fl += ["--positive", positive]
    d = ml("--root", tmp, "fit", "--model", model, "--task", task, "--max-depth", depth, *extra, *fl, "--seed", "0", "--target", target, "--output", "m.cml", "--overwrite", "%s_%s.csv" % (name, train))
    assert d["ok"], d
    e = ml("--root", tmp, "eval", "--model-file", "m.cml", "%s_test.csv" % name)
    assert e["ok"], e
    return {k: float(v) for k, v in e["data"]["metrics"].items()}


def run_sklearn(model, task, depth, Xtr, ytr, Xte, yte):
    if task == "classification":
        c = (RandomForestClassifier(n_estimators=100, max_depth=depth, min_samples_leaf=1, bootstrap=True, random_state=0, n_jobs=1) if model == "forest"
             else DecisionTreeClassifier(max_depth=depth, random_state=0))
        c.fit(Xtr, ytr.astype(int))
        p = c.predict_proba(Xte)[:, 1]
        return {"accuracy": accuracy_score(yte.astype(int), c.predict(Xte)), "roc_auc": roc_auc_score(yte.astype(int), p)}
    r = (RandomForestRegressor(n_estimators=100, max_depth=depth, min_samples_leaf=1, bootstrap=True, max_features=1 / 3, random_state=0, n_jobs=1) if model == "forest"
         else DecisionTreeRegressor(max_depth=depth, random_state=0))
    r.fit(Xtr, ytr)
    return {"rmse": mean_squared_error(yte, r.predict(Xte)) ** 0.5}


def permute(tmp, name, target):
    """<name>_permuted.csv: the train file with its target column permuted by a seeded shuffle"""
    lines = (DATA / ("%s_train.csv" % name)).read_text().strip().split("\n")
    head, rows = lines[0].split(","), [l.split(",") for l in lines[1:]]
    t = head.index(target)
    col = [r[t] for r in rows]
    np.random.RandomState(0).shuffle(col)
    for r, v in zip(rows, col):
        r[t] = v
    (tmp / ("%s_permuted.csv" % name)).write_text(lines[0] + "\n" + "".join(",".join(r) + "\n" for r in rows))


def compare(model, name, target, task, depth, tmp, report, sk_depth=8, positive=None, ml_model=None, train="train"):
    (names, tr), (_, te) = load(name)
    t = names.index(target)
    cols = [j for j in range(len(names)) if j != t]
    m = run_ml(ml_model or model, name, target, task, depth, tmp, positive, train)
    s = run_sklearn(model, task, sk_depth, tr[:, cols], tr[:, t], te[:, cols], te[:, t])
    acc, auc, rel = BOUNDS[model]
    rows = []
    if task == "classification":
        rows = [("accuracy", abs(m["accuracy"] - s["accuracy"]), acc, m["accuracy"], s["accuracy"]), ("roc_auc", abs(m["roc_auc"] - s["roc_auc"]), auc, m["roc_auc"], s["roc_auc"])]
    else:
        rows = [("rmse (relative)", abs(m["rmse"] - s["rmse"]) / s["rmse"], rel, m["rmse"], s["rmse"])]
    ok = True
    for what, dist, bound, a, b in rows:
        within = dist <= bound
        ok &= within
        if report:
            print("%-6s %-13s %-14s observed %.4f  bound %.2f  ml %.4f  scikit-learn %.4f  %s" % (model, name, what, dist, bound, a, b, "PASS" if within else "FAIL"))
    return ok


def main():
    models = sys.argv[1:] or ["forest", "tree"]
    ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for f in DATA.glob("*.csv"):
            (tmp / f.name).write_bytes(f.read_bytes())
        for model in models:
            for name, target, task in SETS:
                ok &= compare(model, name, target, task, 8, tmp, True)
            # C1 (pre-registered): the same model at depth 1 against the reference at depth 8
            for task in ("classification", "regression"):
                outside = [name for name, target, tk in SETS if tk == task and not compare(model, name, target, tk, 1, tmp, False)]
                print("%-6s control C1 (depth 1, %s): outside the bounds on %s" % (model, task, ", ".join(outside) or "NO DATA SET (reported: this control cannot make the gate fail here)"))
            # C2 (added after the first run): a stump against the forest reference
            if model == "forest":
                for task in ("classification", "regression"):
                    outside = [name for name, target, tk in SETS if tk == task and not compare(model, name, target, tk, 1, tmp, False, ml_model="tree")]
                    print("%-6s control C2 (a stump, %s): outside the bounds on %s" % (model, task, ", ".join(outside) or "NO DATA SET"))
                    if not outside:
                        ok = False
                        print("FAIL: the stump stays inside the bounds: this gate could not fail on forest %s" % task)
            # C3 (added after the first run): the target of the train file permuted
            for name, target, tk in SETS:
                permute(tmp, name, target)
            outside = [name for name, target, tk in SETS if not compare(model, name, target, tk, 8, tmp, False, train="permuted")]
            print("%-6s control C3 (the train target permuted): outside the bounds on %s" % (model, ", ".join(outside) or "NO DATA SET"))
            if not outside:
                ok = False
                print("FAIL: a model trained on a permuted target stays inside the bounds on %s" % model)
    print("sklearn gate (trees): %s" % ("all PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

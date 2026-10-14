#!/usr/bin/env python3
"""S11 -- the model file of a categorical fit against scikit-learn, end to end (docs/stringtable.md section 9), with the tolerances fixed in that document BEFORE the first
run:

    logistic (OneHotEncoder + StandardScaler + LogisticRegression(C = 1 / (l2 n_train), lbfgs, tol 1e-12)), the same train and test FILES:
        predicted labels identical on every test row; probability within 1e-6 absolute; accuracy equal; ROC-AUC within 1e-6 absolute; log-loss within 1e-6 relative
    random forest (OneHotEncoder + RandomForestClassifier): the libraries grow different trees, so equality is not claimed:
        held-out accuracy within 0.05 absolute and ROC-AUC within 0.05 absolute of scikit-learn's
    both: `ml predict` and `ml eval` read the MODEL FILE written by `ml fit` on the train file (no --split, no state shared with the fit), and the file's predictions on
        the test rows equal the in-process held-out scores of `ml fit --split` on the same rows (S2, here once more for the generated data)

The split is the library's (train_test_split with a seed), written as two CSV files. The data have 3 categorical and 2 numeric columns, and a rare category that the
train file may not hold (an unseen category at predict, counted).

    python3 scripts/gate_st_sklearn.py
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
ML = os.environ.get("ML", str(ROOT / "build" / "ml"))
N, L2 = 1500, 0.1


def data(dseed):
    rng = np.random.default_rng(dseed)
    c1 = rng.choice(["amber", "blue", "green", "red"], N, p=[0.1, 0.4, 0.3, 0.2])
    c2 = rng.choice(["c0", "c1", "c2", "c3", "c4", "c5"], N, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05])
    c3 = rng.choice(["x", "y", "z", "rare"], N, p=[0.5, 0.25, 0.245, 0.005])
    a = rng.normal(10, 3, N)
    b = rng.uniform(-2, 2, N)
    e1 = {"amber": 0.8, "blue": -0.4, "green": 0.1, "red": 0.5}
    e2 = {"c0": 0.0, "c1": 0.6, "c2": -0.8, "c3": 1.0, "c4": -0.2, "c5": 0.3}
    e3 = {"x": 0.0, "y": 0.5, "z": -0.5, "rare": 2.0}
    logit = np.array([e1[u] + e2[v] + e3[w] for u, v, w in zip(c1, c2, c3)]) + 0.4 * (a - 10) / 3 - 0.7 * b + rng.logistic(0, 1, N) * 0.5
    y = (logit > 0).astype(int)
    return c1, c2, c3, a, b, y


def write_csv(path, idx, c1, c2, c3, a, b, y):
    with open(path, "w") as f:
        f.write("c1,c2,c3,a,b,y\n")
        for i in idx:
            f.write("%s,%s,%s,%r,%r,%d\n" % (c1[i], c2[i], c3[i], float(a[i]), float(b[i]), y[i]))


def ml(td, *args):
    p = subprocess.run([ML, "--root", td, *map(str, args)], capture_output=True, text=True)
    d = json.loads(p.stdout)
    if not d["ok"]:
        raise SystemExit("ml %s: %s" % (args, d["error"]))
    return d["data"]


def main():
    bad = 0
    for dseed in (1, 2, 3):
        c1, c2, c3, a, b, y = data(dseed)
        idx = np.arange(N)
        tr, te = train_test_split(idx, test_size=0.25, random_state=100 + dseed)
        if dseed == 3:
            # the rare category occurs in the test rows only: unseen at predict, from the model file alone
            for i in tr:
                if c3[i] == "rare":
                    c3[i] = "z"
            for i in te[:3]:
                c3[i] = "rare"
        with tempfile.TemporaryDirectory() as td:
            write_csv(pathlib.Path(td) / "train.csv", tr, c1, c2, c3, a, b, y)
            write_csv(pathlib.Path(td) / "test.csv", te, c1, c2, c3, a, b, y)
            cats = np.column_stack([c1, c2, c3])
            num = np.column_stack([a, b])
            unseen = int(sum(1 for i in te if c3[i] not in set(c3[tr])))
            # ---------------------------------------------------------------- logistic
            step = "scale column=a,b kind=standard; encode column=c1,c2,c3 kind=onehot unknown=ignore"
            ml(td, "fit", "train.csv", "--target", "y", "--model", "logistic", "--l2", L2, "--iterations", 100, "--cat", "c1,c2,c3", "--step", step, "--output", "lr.cml")
            pred = ml(td, "predict", "test.csv", "--model-file", "lr.cml", "--limit", 1000, "--proba")
            ev = ml(td, "eval", "test.csv", "--model-file", "lr.cml")
            sc = StandardScaler().fit(num[tr])
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cats[tr])
            Xtr = np.hstack([sc.transform(num[tr]), enc.transform(cats[tr])])
            Xte = np.hstack([sc.transform(num[te]), enc.transform(cats[te])])
            sk = LogisticRegression(penalty="l2", C=1.0 / (L2 * len(tr)), solver="lbfgs", tol=1e-12, max_iter=20000).fit(Xtr, y[tr])
            proba = sk.predict_proba(Xte)[:, 1]
            labels = (proba >= 0.5).astype(int)
            got_l = np.array([int(r[0]) for r in pred["rows"]])
            got_p = np.array([float(r[1]) for r in pred["rows"]])
            m = ev["metrics"]
            rows = [
                ("predicted labels that differ (count)", float(np.sum(got_l != labels)), 0.0),
                ("probability (max abs)", float(np.max(np.abs(got_p - proba))), 1e-6),
                ("accuracy (difference)", abs(float(m["accuracy"]) - accuracy_score(y[te], labels)), 0.0),
                ("roc_auc (absolute)", abs(float(m["roc_auc"]) - roc_auc_score(y[te], proba)), 1e-6),
                ("log_loss (relative)", abs(float(m["log_loss"]) - log_loss(y[te], proba)) / log_loss(y[te], proba), 1e-6),
                ("unseen categories counted (difference)", float(abs(pred["unknown_categories"] - unseen)), 0.0),
            ]
            for name, err, bound in rows:
                ok = err <= bound
                bad += 0 if ok else 1
                print("logistic, data seed %d  %-42s observed %.3e  bound %.0e  %s" % (dseed, name, err, bound, "PASS" if ok else "FAIL"))
            print("logistic, data seed %d  test rows %d, train rows %d, unseen categories %d; ml accuracy %s roc_auc %s log_loss %s; sklearn %.16g %.16g %.16g" % (
                dseed, len(te), len(tr), unseen, m["accuracy"], m["roc_auc"], m["log_loss"], accuracy_score(y[te], labels), roc_auc_score(y[te], proba), log_loss(y[te], proba)))
            # ---------------------------------------------------------------- random forest
            ml(td, "fit", "train.csv", "--target", "y", "--model", "forest", "--task", "classification", "--trees", 100, "--max-depth", 12, "--cat", "c1,c2,c3",
               "--step", "encode column=c1,c2,c3 kind=onehot", "--seed", 5, "--output", "rf.cml")
            fev = ml(td, "eval", "test.csv", "--model-file", "rf.cml")
            frow = ml(td, "predict", "test.csv", "--model-file", "rf.cml", "--limit", 1000, "--proba")
            enc2 = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cats[tr])
            X2tr = np.hstack([num[tr], enc2.transform(cats[tr])])
            X2te = np.hstack([num[te], enc2.transform(cats[te])])
            rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=5).fit(X2tr, y[tr])
            p2 = rf.predict_proba(X2te)[:, 1]
            fm = fev["metrics"]
            rrows = [
                ("accuracy (absolute)", abs(float(fm["accuracy"]) - accuracy_score(y[te], (p2 >= 0.5).astype(int))), 0.05),
                ("roc_auc (absolute)", abs(float(fm["roc_auc"]) - roc_auc_score(y[te], p2)), 0.05),
            ]
            for name, err, bound in rrows:
                ok = err <= bound
                bad += 0 if ok else 1
                print("forest,   data seed %d  %-42s observed %.3e  bound %.0e  %s" % (dseed, name, err, bound, "PASS" if ok else "FAIL"))
            print("forest,   data seed %d  ml accuracy %s roc_auc %s; sklearn %.6f %.6f; predict and eval from the file agree on the labels: %s" % (
                dseed, fm["accuracy"], fm["roc_auc"], accuracy_score(y[te], (p2 >= 0.5).astype(int)), roc_auc_score(y[te], p2),
                int(np.sum([int(r[0]) == y[i] for r, i in zip(frow["rows"], te)])) / len(te) == float(fm["accuracy"])))
            # ---------------------------------------------------------------- the file against the in-process held-out scores of --split, on the same rows
            for name, flags in (("logistic", ["--model", "logistic", "--l2", L2, "--iterations", 100, "--step", step]),
                                ("forest", ["--model", "forest", "--task", "classification", "--trees", 20, "--step", "encode column=c1,c2,c3 kind=onehot"])):
                fit = ml(td, "fit", "train.csv", "--target", "y", "--cat", "c1,c2,c3", *flags, "--split", "train=0.8", "--seed", 9, "--overwrite", "--output", "s.cml")
                e2 = ml(td, "eval", "train.csv", "--model-file", "s.cml", "--split", "train=0.8", "--seed", 9)
                same = fit["summary"]["holdout"]["metrics"] == e2["metrics"]
                bad += 0 if same else 1
                print("%-8s data seed %d  eval --split on the file equals the held-out metrics of the fit, string for string: %s" % (name, dseed, "PASS" if same else "FAIL"))
    print("S11 scikit-learn %s, numpy %s, machine %s: %s" % (__import__("sklearn").__version__, np.__version__, os.uname().machine, "all PASS" if not bad else "%d FAILED" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

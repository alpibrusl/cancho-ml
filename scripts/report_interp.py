#!/usr/bin/env python3
"""The figures of docs/interpret.md section 10: the comparisons of the gates of tests/conformance/test_interp_sklearn.py and test_interp_oracle.py, run again and PRINTED (the tests
only judge them against the bounds fixed in the document). Needs numpy, scipy and scikit-learn (build/venv) and the built binary (ML_BIN or build/ml).

    build/venv/bin/python scripts/report_interp.py
"""

import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
sys.path.insert(0, str(ROOT / "scripts"))
import test_interp_sklearn as T  # noqa: E402
import oracle_interp as O  # noqa: E402
from harness import Sandbox  # noqa: E402


def rel(a, b):
    m = b > 0
    return float(np.max(np.abs(a[m] - b[m]) / b[m])) if m.any() else 0.0


def main():
    sb = Sandbox()
    try:
        out = []
        # I4a: sklearn's own computation on our trees
        for name, task, model, extra, bound in (("cls", "classification", "tree", ["--max-depth", "7"], 1e-12), ("reg", "regression", "tree", ["--max-depth", "7"], 1e-9),
                                                ("cls", "classification", "forest", ["--trees", "20", "--max-depth", "6", "--bootstrap", "0", "--seed", "2"], 1e-12),
                                                ("reg", "regression", "forest", ["--trees", "20", "--max-depth", "6", "--bootstrap", "0", "--seed", "2"], 1e-9)):
            text, X, y = T.data_of(name, 1500, 21)
            sb.write(name + ".csv", text)
            r = sb.run("fit", "--model", model, "--task", task, "--target", "y", "--overwrite", "--output", "m.ml", name + ".csv", *extra)
            assert r.status == 0, r
            M = O.read_model(sb.read("m.ml"))
            ours = np.array([float(c["importance"]) for c in sb.run("importance", "--model-file", "m.ml").data()["importance"]["columns"]])
            theirs = T.sk_importances_of_our_model(M, X, y)
            wrong = T.sk_importances_of_our_model(M, X, y, flip=True)
            out.append("I4a %-4s %-6s nodes %5d  worst relative difference %.2e (bound %.0e)  control (rows routed the wrong side) largest absolute difference %.3f" % (
                name, model, M["tree"]["total"], rel(ours, theirs), bound, float(np.max(np.abs(ours - wrong)))))
        # I4b: a different algorithm
        for name, task in (("cls", "classification"), ("reg", "regression")):
            for shuffled in (False, True):
                text, X, y = T.data_of(name, 2000, 31)
                sb.write(name + ".csv", text)
                r = sb.run("fit", "--model", "forest", "--task", task, "--trees", "100", "--max-depth", "8", "--seed", "1", "--target", "y", "--overwrite", "--output", "m.ml", name + ".csv")
                assert r.status == 0, r
                ours = np.array([float(c["importance"]) for c in sb.run("importance", "--model-file", "m.ml").data()["importance"]["columns"]])
                ys = np.random.RandomState(0).permutation(y) if shuffled else y
                cls = T.RandomForestClassifier if task == "classification" else T.RandomForestRegressor
                theirs = cls(n_estimators=100, max_depth=8, random_state=0, n_jobs=1).fit(X, ys).feature_importances_
                out.append("I4b %-4s %-8s top feature %s/%s  spearman %.3f  largest share difference %.3f  (bounds: same top, >= 0.9, <= 0.1)" % (
                    name, "control" if shuffled else "forest", T.FEATURES[int(np.argmax(ours))], T.FEATURES[int(np.argmax(theirs))], T.rank_corr(ours, theirs), float(np.max(np.abs(ours - theirs)))))
        # I5: the noise column
        for name, task in (("cls", "classification"), ("reg", "regression")):
            sb.write(name + ".csv", O.to_csv(*O.gen_dataset(name, 2000, 11)))
            sb.write(name + "_new.csv", O.to_csv(*O.gen_dataset(name, 1000, 12)))
            r = sb.run("fit", "--model", "forest", "--task", task, "--trees", "100", "--max-depth", "8", "--seed", "1", "--target", "y", "--overwrite", "--output", "n.ml", name + ".csv")
            assert r.status == 0, r
            d = sb.run("importance", "--model-file", "n.ml", "--data", name + "_new.csv", "--repeats", "10", "--seed", "3").data()
            mdi = {c["column"]: (float(c["importance"]), c["rank"]) for c in d["importance"]["columns"]}
            per = {c["column"]: (float(c["importance"]), float(c["std"]), c["rank"]) for c in d["permutation"]["columns"]}
            weakest = min(v[0] for k, v in per.items() if k != "NOISE")
            out.append("I5 %-4s forest of 100 trees: NOISE impurity share %.4f rank %d of %d; permutation %s %.5f +- %.5f rank %d of %d, weakest signal %.5f" % (
                name, mdi["NOISE"][0], mdi["NOISE"][1], len(mdi), d["permutation"]["metric"], per["NOISE"][0], per["NOISE"][1], per["NOISE"][2], len(per), weakest))
        # I7: scikit-learn's permutation importance on the same fitted model
        R = T.PermutationAgainstScikitLearn.R
        t = T.PermutationAgainstScikitLearn()
        t.sb = sb
        for name, model, task, metric, scoring, classification in (("reg", "forest", "regression", "rmse", "neg_root_mean_squared_error", False), ("reg", "linear", "regression", "r2", "r2", False),
                                                                    ("cls", "forest", "classification", "roc_auc", "roc_auc", True), ("cls", "logistic", "classification", "accuracy", "accuracy", True)):
            X2, y2 = t.prepare(name, model, task)
            d, mo, so = t.ours(name, model, metric)
            mt, st = t.theirs(name, model, X2, y2, scoring, classification)
            base = float(d["baseline"])
            bound = 4.0 * np.sqrt((so ** 2 + st ** 2) / R) + 1e-3 * abs(base)
            worst = float(np.max(np.abs(mo - mt) / bound))
            mc, sc = t.theirs(name, model, X2, y2, scoring, classification, drop_signal=True)
            bound_c = 4.0 * np.sqrt((so ** 2 + sc ** 2) / R) + 1e-3 * abs(base)
            out.append("I7 %-4s %-9s %-8s worst |ours - theirs| / bound %.3f (inside if <= 1); control (target shuffled) %.1f times the bound" % (
                name, model, metric, worst, float(np.max(np.abs(mo - mc) / bound_c))))
        print("\n".join(out))
    finally:
        sb.cleanup()


if __name__ == "__main__":
    main()

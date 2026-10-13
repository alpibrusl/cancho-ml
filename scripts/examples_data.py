#!/usr/bin/env python3
"""Write the data of the worked examples and scikit-learn's results on them (docs/examples.md).

    python3 scripts/examples_data.py                 # self-test of the --l2 mapping, then write the files
    python3 scripts/examples_data.py --selftest      # the self-test only
    python3 scripts/examples_data.py --bin build/ml  # the built ml used by the self-test (default build/ml)

NEEDS scikit-learn (and numpy, scipy). It is the only script of the repository that does, and nothing else runs it: the
documentation check (scripts/check_docs.sh) reads what this script wrote and never re-runs scikit-learn.

Writes, from the datasets bundled with scikit-learn (load_*, never fetch_*, no network):
    docs/examples/data/<name>_train.csv, <name>_test.csv   a fixed seeded split, stratified for a classification
    docs/examples/reference.json                           scikit-learn's results on those very files, with the versions
                                                           of scikit-learn, numpy, scipy and Python and this command line
Float cells are written with repr (the shortest decimal that reads back to the same double).

The mapping of ml's --l2 to scikit-learn's parameters (docs/examples.md section 2):
    linear:    Ridge(alpha = l2)            (LinearRegression() at l2 = 0)
    logistic:  LogisticRegression(C = 1 / (l2 * n_train))   (C = inf, no penalty, at l2 = 0)
The self-test checks it on a toy file and also checks that the wrong mapping (C = 1 / l2) is caught.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import examples_common as common  # noqa: E402

SPLIT = dict(test_size=0.25, random_state=0)


def clean(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load(key):
    from sklearn import datasets
    import numpy as np
    if key == "breast_cancer":
        d = datasets.load_breast_cancer()
        return [clean(n) for n in d.feature_names], d.data, d.target.astype(int), True
    if key == "diabetes":
        d = datasets.load_diabetes()
        return [clean(n) for n in d.feature_names], d.data, d.target.astype(float), False
    if key == "iris":
        d = datasets.load_iris()
        keep = d.target != 0  # versicolor (1) and virginica (2)
        return [clean(n) for n in d.feature_names], d.data[keep], (d.target[keep] == 2).astype(int), True
    if key == "wine":
        d = datasets.load_wine()
        return [clean(n) for n in d.feature_names], d.data, (d.target == 0).astype(int), True
    raise KeyError(key)


def write_csv(path, names, target, X, y, classification):
    lines = [",".join(names + [target])]
    for row, t in zip(X, y):
        cells = [repr(float(v)) for v in row]
        cells.append(str(int(t)) if classification else repr(float(t)))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- scikit-learn side --------------------------------------------------------------------------------------------


def sk_fit_linear(X, y, l2):
    from sklearn.linear_model import LinearRegression, Ridge
    if l2 == 0:
        return LinearRegression(), "LinearRegression()"
    return Ridge(alpha=l2, solver="cholesky"), 'Ridge(alpha=%r, solver="cholesky")' % l2


def sk_logistic(l2, n, solver="lbfgs", mapping="right"):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    if l2 == 0:
        c = np.inf
    elif mapping == "right":
        c = 1.0 / (l2 * n)
    else:
        c = 1.0 / l2
    return LogisticRegression(C=c, solver=solver, tol=1e-12, max_iter=100000), \
        "LogisticRegression(C=%r, solver=%r, tol=1e-12, max_iter=100000)" % (c, solver)


def fit_quiet(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a convergence warning is a failure, not something to hide
        warnings.filterwarnings("ignore", message="Setting penalty=None")
        model.fit(X, y)
    return model


def class_metrics(y, p):
    import numpy as np
    from sklearn import metrics as m
    lab = (p > 0.5).astype(int)
    return {"accuracy": float(m.accuracy_score(y, lab)), "precision": float(m.precision_score(y, lab)),
            "recall": float(m.recall_score(y, lab)), "f1": float(m.f1_score(y, lab)),
            "roc_auc": float(m.roc_auc_score(y, p)), "log_loss": float(m.log_loss(y, p))}


def reg_metrics(y, p):
    from sklearn import metrics as m
    return {"rmse": float(m.mean_squared_error(y, p)) ** 0.5, "mae": float(m.mean_absolute_error(y, p)),
            "r2": float(m.r2_score(y, p))}


def reference_for(ex, run, names, Xtr, ytr, Xte, yte):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    if run["scale"]:
        sc = StandardScaler().fit(Xtr)  # population variance, as ml's scale kind=standard (ddof=0)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    l2 = float(run["l2"])
    out = {"l2": run["l2"], "model": run["model"]}
    if run["model"] == "linear":
        model, desc = sk_fit_linear(Xtr, ytr, l2)
        fit_quiet(model, Xtr, ytr)
        coef, icpt = model.coef_, float(model.intercept_)
        out.update(sklearn_params=desc, metrics=reg_metrics(yte, model.predict(Xte)),
                   pred_first=[float(v) for v in model.predict(Xte)[:common.PROBA_ROWS]])
    else:
        n = len(ytr)
        model, desc = sk_logistic(l2, n)
        fit_quiet(model, Xtr, ytr)
        newton, ndesc = sk_logistic(l2, n, solver="newton-cholesky")
        fit_quiet(newton, Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        coef, icpt = model.coef_[0], float(model.intercept_[0])
        out.update(sklearn_params=desc, metrics=class_metrics(yte, p),
                   proba_first=[float(v) for v in p[:common.PROBA_ROWS]],
                   label_first=[int(v) for v in model.predict(Xte)[:common.PROBA_ROWS]],
                   n_iter=int(model.n_iter_[0]),
                   newton_params=ndesc, newton_coef=[float(v) for v in newton.coef_[0]],
                   newton_intercept=float(newton.intercept_[0]), newton_n_iter=int(newton.n_iter_[0]),
                   scores_min_abs=float(np.min(np.abs(model.decision_function(Xte)))))
    out["coef"] = [{"name": n_, "value": float(v)} for n_, v in zip(names, coef)]
    out["intercept"] = icpt
    return out


# --- the self-test of the mapping ----------------------------------------------------------------------------------


def run_ml(binary, args, cwd):
    r = subprocess.run([str(binary)] + args[1:], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return json.loads(r.stdout.decode())


def selftest(binary):
    import numpy as np
    rng = np.random.default_rng(1)
    n = 40
    X = rng.normal(size=(n, 3))
    yc = (X @ [1, -2, 0.5] + rng.normal(size=n) * 1.5 > 0).astype(int)
    yr = X @ [1, -2, 0.5] + 3 + rng.normal(size=n)
    names = ["a", "b", "c"]
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        write_csv(Path(tmp) / "c.csv", names, "y", X, yc, True)
        write_csv(Path(tmp) / "r.csv", names, "y", X, yr, False)

        def compare(tag, doc, coef, icpt, kind):
            s = doc["data"]["summary"]
            ml = [float(c["value"]) for c in s["coefficients"]] + [float(s["intercept"])]
            sk = list(coef) + [icpt]
            worst = max(abs(a - b) / max(1.0, abs(b)) for a, b in zip(ml, sk))
            tol = common.COEF_TOL[kind][0]
            print("selftest %-34s worst relative difference %.2e (tolerance %.0e)" % (tag, worst, tol))
            return worst <= tol

        for l2 in (0.0, 2.0):
            doc = run_ml(binary, ["ml", "fit", "--model", "linear", "--target", "y", "--l2", repr(l2), "--output", "r.ml", "r.csv", "--overwrite"], tmp)
            model, _ = sk_fit_linear(X, yr, l2)
            model.fit(X, yr)
            if not compare("linear l2=%g" % l2, doc, model.coef_, float(model.intercept_), "linear"):
                bad.append("linear l2=%g" % l2)
        for l2 in (0.0, 0.05):
            doc = run_ml(binary, ["ml", "fit", "--model", "logistic", "--target", "y", "--iterations", "50", "--l2", repr(l2), "--output", "c.ml", "c.csv", "--overwrite"], tmp)
            model, _ = sk_logistic(l2, n)
            fit_quiet(model, X, yc)
            if not compare("logistic l2=%g, C=1/(l2 n)" % l2, doc, model.coef_[0], float(model.intercept_[0]), "logistic"):
                bad.append("logistic l2=%g" % l2)
            if l2 > 0:
                wrong, _ = sk_logistic(l2, n, mapping="wrong")
                fit_quiet(wrong, X, yc)
                if compare("logistic l2=%g, WRONG C=1/l2" % l2, doc, wrong.coef_[0], float(wrong.intercept_[0]), "logistic"):
                    bad.append("the wrong mapping C = 1/l2 was not caught")
    if bad:
        sys.exit("examples_data.py: the self-test of the mapping failed: " + "; ".join(bad))
    print("selftest: the mapping agrees where it must and the wrong one is caught")


# --- main ---------------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", default=str(ROOT / "build" / "ml"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    import numpy
    import scipy
    import sklearn
    from sklearn.model_selection import train_test_split
    if not Path(args.bin).exists():
        sys.exit("examples_data.py: %s does not exist: build ml first (README quick start)" % args.bin)
    selftest(args.bin)
    if args.selftest:
        return 0
    (ROOT / common.DATA_DIR).mkdir(parents=True, exist_ok=True)
    ref = {
        "generated_by": {
            "command": "python3 scripts/examples_data.py" + "".join(" " + a for a in sys.argv[1:]),
            "python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "split": "train_test_split(test_size=0.25, random_state=0), stratified by the target for a classification",
            "note": "scikit-learn's results on the committed CSVs; scripts/site.py and check_docs.sh read this file and do not re-run scikit-learn",
        },
        "examples": {},
    }
    for ex in common.EXAMPLES:
        names, X, y, classification = load(ex["key"])
        assert classification == (ex["task"] == "classification")
        Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y if classification else None, **SPLIT)
        files = {}
        for side, XX, yy in (("train", Xtr, ytr), ("test", Xte, yte)):
            rel = "%s/%s_%s.csv" % (common.DATA_DIR, ex["key"], side)
            sha = write_csv(ROOT / rel, names, ex["target"], XX, yy, classification)
            files[side] = {"path": rel, "rows": int(len(yy)), "sha256": sha}
        entry = {"files": files, "features": len(names), "runs": {}}
        if classification:
            entry["positive_share_train"] = float(ytr.mean())
        for run in ex["runs"]:
            if run.get("ref"):
                entry["runs"][run["id"]] = reference_for(ex, run, names, Xtr, ytr, Xte, yte)
        ref["examples"][ex["key"]] = entry
        print("%-14s train %d test %d, %d reference run(s)" % (ex["key"], len(ytr), len(yte), len(entry["runs"])))
    (ROOT / common.REFERENCE).write_text(json.dumps(ref, indent=1, sort_keys=False) + "\n")
    print("wrote", common.REFERENCE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

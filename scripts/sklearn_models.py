#!/usr/bin/env python3
"""G-sklearn of docs/models-linear.md section 2: the models of lib/models against scikit-learn, with the tolerances written there
before the first run. Data: the M0 sets (ml-data), fit on the first n - n//5 rows, test on the rest (as spike S). Prints one line
per comparison with the measured difference, the tolerance and PASS / FAIL; exit 1 on any FAIL.

    python3 scripts/sklearn_models.py [--sets n1000,n100000,n1000000] [--only linreg,ridge,logreg,logreg-l2,kmeans]
Environment: MODELS_CHECK (driver), ML_DATA (data directory)."""
import os, subprocess, sys, tempfile, warnings, pathlib
import numpy as np
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_models as om
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

BIN = os.environ.get("MODELS_CHECK", str(HERE.parent / "build" / "models-check"))
DATA = pathlib.Path(os.environ.get("ML_DATA", "/Users/alfonso/Workspace/alpibrusl/lexsys/ml-data"))
FAILED = []


def floats(line):
    return np.array([int(h, 16) for h in line.split()], dtype=np.uint64).view(np.float64)


def driver(model, xf, yf, n, d, ntr, **kw):
    args = [BIN, model, xf, yf, "n=%d" % n, "d=%d" % d, "ntr=%d" % ntr] + ["%s=%s" % kv for kv in kw.items()]
    r = subprocess.run(args, capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        k, _, v = line.partition(" ")
        out[k] = v
    if "params" not in out:
        raise SystemExit("driver failed: %s %s %s" % (" ".join(args), r.stdout[:200], r.stderr[:200]))
    return out


def verdict(name, got, tol, extra=""):
    ok = got <= tol
    print("%-46s %12.3e  tol %-9.1e %s %s" % (name, got, tol, "PASS" if ok else "FAIL", extra), flush=True)
    if not ok:
        FAILED.append(name)


def logloss_objective(X, y, w, b, l2):
    z = X @ w + b
    return float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * (w @ w))


def main():
    sets = ["n1000", "n100000", "n1000000"]
    only = None
    if "--sets" in sys.argv:
        sets = sys.argv[sys.argv.index("--sets") + 1].split(",")
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    want = lambda m: only is None or m in only
    for s in sets:
        n = int(s[1:])
        d = 20
        ntr = n - n // 5
        xf, yf = str(DATA / s / "x.bin"), str(DATA / s / "y.bin")
        X = np.fromfile(xf, dtype="<f8").reshape(n, d)
        y = np.fromfile(yf, dtype="<f8")
        Xt, yt, Xe, ye = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
        print("== %s: n=%d, fit on %d rows, test on %d" % (s, n, ntr, n - ntr))
        # ---- linear and ridge: a generated regression target on the same X
        if want("linreg") or want("ridge"):
            rng = np.random.default_rng(7)
            wtrue = rng.standard_normal(d)
            ylin = X @ wtrue + 3.0 + 0.5 * rng.standard_normal(n)
            with tempfile.TemporaryDirectory() as td:
                yl = os.path.join(td, "ylin.bin")
                ylin.astype("<f8").tofile(yl)
                for name, l2m in (("linreg", 0), ("ridge", 25000)):
                    if not want(name):
                        continue
                    out = driver("linreg", xf, yl, n, d, ntr, l2m=l2m)
                    w = floats(out["params"])
                    sk = (LinearRegression() if l2m == 0 else Ridge(alpha=l2m / 1000.0, solver="cholesky")).fit(Xt, ylin[:ntr])
                    ws = np.concatenate([sk.coef_, [sk.intercept_]])
                    verdict("%s coefficients (max |dw| / max(1,|w|))" % name, float(np.max(np.abs(w - ws))) / max(1.0, float(np.max(np.abs(ws)))), 1e-8)
                    rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
                    r1 = rm(Xe @ w[:d] + w[d], ylin[ntr:])
                    r2 = rm(sk.predict(Xe), ylin[ntr:])
                    verdict("%s test RMSE (relative)" % name, abs(r1 - r2) / r2, 1e-9, "rmse cancho %.9f sklearn %.9f" % (r1, r2))
        # ---- logistic
        for name, l2m in (("logreg", 0), ("logreg-l2", 10)):
            if not want(name):
                continue
            l2 = l2m / 1000.0
            out = driver("logreg", xf, yf, n, d, ntr, l2m=l2m, iters=8)
            w = floats(out["params"])
            out12 = driver("logreg", xf, yf, n, d, ntr, l2m=l2m, iters=12)
            w12 = floats(out12["params"])
            if l2m == 0:
                sk = LogisticRegression(penalty=None, solver="lbfgs", tol=1e-12, max_iter=10000).fit(Xt, yt)
            else:
                sk = LogisticRegression(penalty="l2", C=1.0 / (l2 * ntr), solver="lbfgs", tol=1e-12, max_iter=10000).fit(Xt, yt)
            ws = np.concatenate([sk.coef_[0], sk.intercept_])
            verdict("%s coefficients (max |dw|)" % name, float(np.max(np.abs(w - ws))), 1e-6, "K=8 vs K=12 max |dw| %.2e" % float(np.max(np.abs(w - w12))))
            f1 = logloss_objective(Xt, yt, w[:d], w[d], l2)
            f2 = logloss_objective(Xt, yt, ws[:d], ws[d], l2)
            verdict("%s train objective |df|" % name, abs(f1 - f2), 1e-9, "f cancho %.12f sklearn %.12f (cancho - sklearn %+.2e)" % (f1, f2, f1 - f2))
            npos = int(ye.sum())
            nneg = len(ye) - npos
            a1 = roc_auc_score(ye, Xe @ w[:d] + w[d])
            a2 = roc_auc_score(ye, Xe @ ws[:d] + ws[d])
            verdict("%s test AUC |dAUC|" % name, abs(a1 - a2), max(1e-5, 1.0 / (npos * nneg)), "AUC %.6f / %.6f" % (a1, a2))
        # ---- k-means: sklearn gets cancho's own initial centroids
        if want("kmeans"):
            k, iters, seed = 8, 10, 3
            ch = om.chosen_rows(Xt, k, seed, 1)
            out = driver("kmeans", xf, yf, n, d, ntr, k=k, iters=iters, seed=seed, init=1)
            cen = floats(out["params"]).reshape(k, d)
            km = KMeans(n_clusters=k, init=Xt[ch].copy(), n_init=1, algorithm="lloyd", max_iter=iters, tol=0.0).fit(Xt)
            verdict("kmeans centroids (max |dc| / max(1,|c|))", float(np.max(np.abs(cen - km.cluster_centers_) / np.maximum(1.0, np.abs(km.cluster_centers_)))), 1e-9)
            lab, mn, _ = om.assign(Xt, cen)
            inertia = float(sum(mn))        # numpy pairwise sum: only a reporting value
            verdict("kmeans inertia (relative)", abs(inertia - km.inertia_) / km.inertia_, 1e-9, "inertia cancho %.6f sklearn %.6f" % (inertia, km.inertia_))
            differ = int(np.sum(lab != km.labels_))
            verdict("kmeans labels differing (rows)", float(differ), 1e-5 * ntr, "%d of %d rows; empty clusters: %d" % (differ, ntr, int(np.sum(np.bincount(lab, minlength=k) == 0))))
    print("sklearn gate:", "all PASS" if not FAILED else "FAILED: " + ", ".join(FAILED))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()

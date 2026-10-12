#!/usr/bin/env python3
"""G-ref (docs/models-tree.md section 8): the quality of the library's forests against scikit-learn's RandomForest on the same
train/test split, at the tolerances fixed in the document BEFORE this script ran: classification test AUC within 0.005,
regression test RMSE within 3% (relative). scikit-learn's algorithm is different (exact thresholds, its own RNG, a feature subset is
redrawn when no split exists, float criterion), so this compares what the models achieve, not the trees. Missing values: scikit-learn
(>= 1.4) learns a default side from NaN as well, so the missing-value configurations use the NaN matrix on its side.
A control shows the gate can fail: a forest of depth 1 must be outside the tolerance.
usage: gate_sklearn.py [--big]   (--big adds the 1,000,000-row classification, 13 s of scikit-learn)"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_tree as G

TOL_AUC, TOL_RMSE = 0.005, 0.03


def cancho_run(exe, cfg, size, **over):
    c = dict(cfg, **over)
    code, out = G.run(exe, c, size, threads=1, out="full")
    assert code == 0, out[:200]
    pred = np.array([int(l.split()[2], 16) for l in out.splitlines() if l.startswith("P ")], dtype=np.uint64).view("<f8")
    return pred, out


def cancho_pred(exe, cfg, size, **over):
    return cancho_run(exe, cfg, size, **over)[0]


def binned(X, out, d, nbins):
    """X with every present value replaced by its bin index under the library's own edges (E lines); NaN stays NaN"""
    E = np.zeros((d, nbins - 1))
    for l in out.splitlines():
        if l.startswith("E "):
            t = l.split(); E[int(t[1]), int(t[2]) - 1] = np.uint64(int(t[3], 16)).view(np.float64)
    Z = np.empty_like(X)
    for f in range(d):
        col = X[:, f]
        Z[:, f] = np.where(np.isnan(col), np.nan, np.searchsorted(E[f], np.nan_to_num(col), side="right"))
    return Z


def load(cfg, size):
    i = G.inputs(cfg, size)
    n, d = i["n"], i["d"]
    X = np.fromfile(i["x"], "<f8").reshape(n, d)
    y = np.fromfile(i["y"], "<f8")
    ntr = n - n // 5
    return X, y, ntr


def run_one(exe, cid, size, rows):
    cfg = G.CONFIGS[cid]
    X, y, ntr = load(cfg, size)
    mtry = cfg["mtry"]
    pc, out = cancho_run(exe, cfg, size)
    Z = binned(X, out, X.shape[1], cfg["bins"])
    if cfg["task"] == 0:
        m = RandomForestClassifier(n_estimators=cfg["trees"], max_depth=cfg["depth"], max_features=mtry, min_samples_leaf=cfg["leaf"], bootstrap=bool(cfg["boot"]), n_jobs=1, random_state=0)
        m.fit(X[:ntr], y[:ntr])
        ps = m.predict_proba(X[ntr:])[:, 1]
        a, b = roc_auc_score(y[ntr:], pc), roc_auc_score(y[ntr:], ps)
        pk = cancho_pred(exe, cfg, size, depth=1)
        ctl = abs(roc_auc_score(y[ntr:], pk) - b)
        ok = abs(a - b) <= TOL_AUC
        m.fit(Z[:ntr], y[:ntr])
        bb = roc_auc_score(y[ntr:], m.predict_proba(Z[ntr:])[:, 1])
        rows.append((cid, size, "AUC", a, b, abs(a - b), TOL_AUC, ok, ctl, ctl > TOL_AUC, bb, abs(a - bb), abs(a - bb) <= TOL_AUC))
    else:
        m = RandomForestRegressor(n_estimators=cfg["trees"], max_depth=cfg["depth"], max_features=mtry, min_samples_leaf=cfg["leaf"], bootstrap=bool(cfg["boot"]), n_jobs=1, random_state=0)
        m.fit(X[:ntr], y[:ntr])
        ps = m.predict(X[ntr:])
        rm = lambda p: float(np.sqrt(mean_squared_error(y[ntr:], p)))
        a, b = rm(pc), rm(ps)
        pk = cancho_pred(exe, cfg, size, depth=1)
        ctl = abs(rm(pk) - b) / b
        ok = abs(a - b) / b <= TOL_RMSE
        m.fit(Z[:ntr], y[:ntr])
        bb = rm(m.predict(Z[ntr:]))
        rows.append((cid, size, "RMSE", a, b, abs(a - b) / b, TOL_RMSE, ok, ctl, ctl > TOL_RMSE, bb, abs(a - bb) / bb, abs(a - bb) / bb <= TOL_RMSE))


def main():
    exe = f"{G.ROOT}/build/tree-check"
    if not os.path.exists(exe):
        G.build(G.ROOT, exe)
    rows = []
    big = "--big" in sys.argv
    jobs = [("C1", "100000"), ("C2", "100000"), ("C3", "100000"), ("R1", "100000"), ("R2", "100000"), ("C1", "1000"), ("R1", "1000")]
    for cid, size in jobs:
        run_one(exe, cid, size, rows)
    if big:
        os.makedirs(f"{G.ROOT}/build/tree-data/n1000000", exist_ok=True)
        run_one(exe, "C1", "1000000", rows)
    print("scikit-learn %s, numpy %s; cancho tree-check, 1 thread" % (sklearn.__version__, np.__version__))
    print("PRE-REGISTERED (docs/models-tree.md section 8, fixed before the first run): tolerance AUC %.3f, RMSE %.0f%% relative, scikit-learn on the raw features" % (TOL_AUC, 100 * TOL_RMSE))
    print("%-4s %-8s %-6s %-10s %-10s %-10s %-7s %-6s | control (depth 1 forest): distance, outside the tolerance?" % ("cfg", "rows", "metric", "cancho", "sklearn", "distance", "tol", "within"))
    bad = 0
    for cid, size, met, a, b, dist, tol, ok, ctl, ctlout, bb, bdist, bok in rows:
        if size == "1000":
            note, ok_g = "  (1,000 rows: reported, not part of the gate)", True
        else:
            note, ok_g = "", ok
        print("%-4s %-8s %-6s %-10.6f %-10.6f %-10.6f %-7.3f %-6s | %.4f %s%s" % (cid, size, met, a, b, dist, tol, "yes" if ok else "NO", ctl, "outside" if ctlout else "INSIDE (the gate could not fail)", note))
        bad += (not ok_g) or (not ctlout and size != "1000")
    print("G-ref (pre-registered)", "FAILED" if bad else "passed")
    print()
    print("DIAGNOSTIC ADDED AFTER THE FIRST RUN (not pre-registered): scikit-learn on the bin indices under the library's own edges (the same")
    print("candidate thresholds), same tolerances. It separates 'different thresholds' from 'something else': the remaining difference is")
    print("the RNG, the redrawn feature subsets and the float criterion.")
    print("%-4s %-8s %-6s %-10s %-12s %-10s %-6s" % ("cfg", "rows", "metric", "cancho", "sklearn-bins", "distance", "within"))
    bad2 = 0
    for cid, size, met, a, b, dist, tol, ok, ctl, ctlout, bb, bdist, bok in rows:
        print("%-4s %-8s %-6s %-10.6f %-12.6f %-10.6f %-6s" % (cid, size, met, a, bb, bdist, "yes" if bok else "NO"))
        bad2 += (not bok) and size != "1000"
    print("G-ref (diagnostic on bin indices)", "FAILED" if bad2 else "passed")
    return 1 if bad else 0


sys.exit(main())

#!/usr/bin/env python3
"""Gate P1 of docs/metrics.md: ROC-AUC and log-loss on 1,000,000 predictions against sklearn and a thin numpy version.

One thread (OMP/OPENBLAS/VECLIB = 1 in this process; cancho is single threaded), the same data on both sides (the generated inputs of
tools/metrics-check/gen.cho, rebuilt by numpy), minimum of 7 interleaved samples (one Python call, one cancho process, in turn), spread
= max / min, the load average read at the start and at the end of each cell. A cancho sample is a separate process of
`metrics-check metric ...` timing REPS calls of the metric alone with clock_ms (1 ms resolution; the data are generated before the
clock starts; allocation inside the metric IS timed). A Python sample is one call timed with perf_counter.

    python3 scripts/bench_metrics.py [--n 1000000] [--samples 7]
    python3 scripts/bench_metrics.py --sum-cost       the cost and the error of the tree sum against a left fold and math.fsum"""
import os, sys, time, subprocess, pathlib, math, json
os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import oracle_metrics as om

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load():
    return float(os.getloadavg()[0])


def cancho_ms(args, reps):
    r = subprocess.run([om.BIN] + [str(a) for a in args] + [reps], capture_output=True, text=True)
    out = dict(l.split(" ", 1) for l in r.stdout.splitlines() if " " in l)
    return float(out["ms"]) / reps, out["value"]


def thin_auc(y, s):
    o = np.argsort(s, kind="quicksort")
    ss = s[o]
    yy = (y[o] == 1)
    idx = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]
    tps = np.cumsum(yy)[idx]
    fps = (idx + 1) - tps
    tp0 = np.r_[0, tps[:-1]]
    fp0 = np.r_[0, fps[:-1]]
    two_u = int(np.sum((fps - fp0).astype(np.int64) * (tps + tp0).astype(np.int64)))
    P, N = int(tps[-1]), int(fps[-1])
    return two_u / (2 * P * N)


def thin_logloss(y, p):
    eps = 2.0 ** -52
    x = np.where(y == 1, p, 1.0 - p)
    x = np.clip(x, eps, 1.0 - eps)
    return float(-np.log(x).sum() / len(x))


def cell(name, pyfun, cargs, reps, samples):
    from sklearn import metrics as sm
    l0 = load()
    py, ca = [], []
    pyfun()                       # warm up
    cancho_ms(cargs, 1)
    pyv = cav = None
    for _ in range(samples):
        t = time.perf_counter(); pyv = pyfun(); py.append((time.perf_counter() - t) * 1000)
        ms, cav = cancho_ms(cargs, reps); ca.append(ms)
    l1 = load()
    return {"name": name, "py_min": min(py), "py_spread": max(py) / min(py), "c_min": min(ca), "c_spread": max(ca) / min(ca),
            "load0": l0, "load1": l1, "py_value": pyv, "c_value": om.unbits(cav)}


def main():
    a = sys.argv[1:]
    n = int(a[a.index("--n") + 1]) if "--n" in a else 1000000
    samples = int(a[a.index("--samples") + 1]) if "--samples" in a else 7
    import sklearn
    from sklearn import metrics as sm
    print("machine: %s; numpy %s, sklearn %s, python %s; load at start %.2f; n = %d, min of %d interleaved samples; one thread"
          % (os.uname().machine + " " + os.uname().sysname + " " + os.uname().release, np.__version__, sklearn.__version__, sys.version.split()[0], load(), n, samples))
    ls, ss = 3, 8
    rows = []
    for label, mode in (("uniform scores", 0), ("16 distinct values (ties)", 1)):
        y = om.gen_labels(n, ls, 500)
        s = om.gen_floats(mode, n, ss)
        yspec, sspec = "GL:%d:%d:500" % (n, ls), "GF:%d:%d:%d:0" % (mode, n, ss)
        cargs = ("metric", "auc", yspec, sspec, 1, 10000000)
        rows.append(dict(cell("AUC %s, sklearn.roc_auc_score" % label, lambda: float(sm.roc_auc_score(y, s)), cargs, 3, samples), what="auc"))
        rows.append(dict(cell("AUC %s, thin numpy (argsort + cumsum)" % label, lambda: thin_auc(y, s), cargs, 3, samples), what="auc"))
    for label, mode in (("uniform probabilities", 0), ("16 distinct values", 1)):
        y = om.gen_labels(n, ls, 300)
        p = om.gen_floats(mode, n, ss)
        yspec, pspec = "GL:%d:%d:300" % (n, ls), "GF:%d:%d:%d:0" % (mode, n, ss)
        cargs = ("metric", "logloss", yspec, pspec, 1, 10000000)
        rows.append(dict(cell("log-loss %s, sklearn.log_loss" % label, lambda: float(sm.log_loss(y, p, labels=[0, 1])), cargs, 3, samples), what="ll"))
        rows.append(dict(cell("log-loss %s, thin numpy (clip, log, sum)" % label, lambda: thin_logloss(y, p), cargs, 3, samples), what="ll"))
    print("%-62s %10s %10s %9s %7s %7s %11s" % ("cell", "reference", "cancho", "c / ref", "spread", "spread", "load"))
    print("%-62s %10s %10s %9s %7s %7s %11s" % ("", "ms", "ms", "", "ref", "cancho", "start/end"))
    for r in rows:
        ratio = r["c_min"] / r["py_min"]
        verdict = "MATCH" if ratio <= 1.05 else "NOT MATCH"
        print("%-62s %10.2f %10.2f %8.2fx %7.2f %7.2f %5.1f/%-5.1f %s" % (r["name"], r["py_min"], r["c_min"], ratio, r["py_spread"], r["c_spread"], r["load0"], r["load1"], verdict))
        if r["what"] == "auc":
            d = abs(r["py_value"] - r["c_value"]) / r["c_value"]
            print("    values: reference %.17g, cancho %.17g, relative gap %.2e" % (r["py_value"], r["c_value"], d))
        else:
            print("    values: reference %.17g, cancho %.17g, relative gap %.2e" % (r["py_value"], r["c_value"], abs(r["py_value"] - r["c_value"]) / r["c_value"]))
    json.dump(rows, open(ROOT / "docs" / "bench" / "metrics-1thread.json", "w"), indent=1)


if __name__ == "__main__":
    main()

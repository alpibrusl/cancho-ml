#!/usr/bin/env python3
"""Timing of lib/models/tree (docs/models-tree.md section 9). Minimum of N interleaved rounds; each round runs every entry once, in
turn (cancho process, scikit-learn call, ...). The load average (1 minute) is read at the start and at the end of every round.
usage: bench_tree.py SIZE [--rounds 7] [--out FILE.jsonl]        SIZE = 100000 or 1000000 (ml-data/nSIZE)
Entries (classification, depth 8, 64 bins, mtry 4, bootstrap):
  cancho T=10 threads 1,2,4,8,16        cancho T=32 threads 1,2,4,8,16
  sklearn RandomForestClassifier T=10, n_jobs 1 and 16 (EXACT splits: a different algorithm)
  sklearn HistGradientBoostingClassifier, 1 iteration, 1 thread (a histogram tree, but gradient boosting: leaf-wise, second-order
  leaf values, no bagging, no feature subsets; CONTEXT ONLY, not an equivalent)
cancho times are read inside the program (clock_ms): fit = from the loaded arrays to the packed forest (edges, bins, thread copies,
merge included), predict = the test rows to probabilities (piece copies included); repeated `reps` times and divided (reps printed).
sklearn times are perf_counter around fit and around predict_proba. Run with the machine quiet; every environment variable below is
set to 1 so that no library adds threads of its own."""
import json, os, platform, subprocess, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.environ.get("ML_DATA", os.path.join(os.path.dirname(ROOT), "ml-data"))
EXE = f"{ROOT}/build/tree-check"
THREADS = (1, 2, 4, 8, 16)


def cancho(size, trees, threads, reps, preps, phase=0):
    d = f"{ML}/n{size}"
    a = [EXE, "mode=fit", f"x={d}/x.bin", f"y={d}/y.bin", f"n={size}", "d=20", "task=0", "seed=20261012", f"trees={trees}", "depth=8", "leaf=1", "bins=64",
         "mtry=4", "boot=1", f"threads={threads}", "out=hash", f"reps={reps}", f"preps={preps}", f"phase={phase}"]
    r = subprocess.run(a, capture_output=True, text=True, check=True).stdout
    kv = {l.split()[0][1:]: int(l.split()[1]) for l in r.splitlines() if l.startswith("#") and len(l.split()) == 2}
    h = {l.split()[1]: l.split()[2] for l in r.splitlines() if l.startswith("H ")}
    return dict(fit=kv["fit_ms_total"] / reps / 1000.0, predict=kv["predict_ms_total"] / preps / 1000.0, edges=kv["edges_ms"] / 1000.0, bins=kv["bins_ms"] / 1000.0, hash=h["forest"][:12])


def main():
    size = sys.argv[1]
    rounds = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 7
    outf = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    n = int(size)
    ntr = n - n // 5
    X = np.fromfile(f"{ML}/n{size}/x.bin", "<f8").reshape(n, 20)
    y = np.fromfile(f"{ML}/n{size}/y.bin", "<f8")
    reps, preps = (10, 10) if n <= 100000 else (1, 3)
    entries = []
    for trees in (10, 32):
        for t in THREADS:
            entries.append(("cancho", trees, t))
    entries += [("sklearn-rf", 10, 1), ("sklearn-rf", 10, 16), ("sklearn-hgb-1iter", 1, 1)]
    res = {e: dict(fit=[], predict=[], extra=[]) for e in entries}
    loads = []
    log = []
    print("machine %s %s, %d cores, sklearn %s numpy %s, size %s (train %d, test %d), rounds %d, cancho reps %d/%d" % (platform.machine(), platform.platform(terse=True), os.cpu_count(), sklearn.__version__, np.__version__, size, ntr, n - ntr, rounds, reps, preps), flush=True)
    for rd in range(rounds):
        l0 = os.getloadavg()[0]
        for e in entries:
            kind, trees, t = e
            if kind == "cancho":
                r = cancho(size, trees, t, reps, preps, phase=1 if (t == 1 and rd == 0) else 0)
                res[e]["fit"].append(r["fit"]); res[e]["predict"].append(r["predict"]); res[e]["extra"].append((r["edges"], r["bins"], r["hash"]))
            elif kind == "sklearn-rf":
                m = RandomForestClassifier(n_estimators=trees, max_depth=8, max_features=4, bootstrap=True, n_jobs=t, random_state=0)
                t0 = time.perf_counter(); m.fit(X[:ntr], y[:ntr]); t1 = time.perf_counter(); m.predict_proba(X[ntr:]); t2 = time.perf_counter()
                res[e]["fit"].append(t1 - t0); res[e]["predict"].append(t2 - t1)
            else:
                m = HistGradientBoostingClassifier(max_iter=1, max_depth=8, max_leaf_nodes=None, max_bins=63, min_samples_leaf=1, learning_rate=1.0, early_stopping=False, random_state=0)
                t0 = time.perf_counter(); m.fit(X[:ntr], y[:ntr]); t1 = time.perf_counter(); m.predict_proba(X[ntr:]); t2 = time.perf_counter()
                res[e]["fit"].append(t1 - t0); res[e]["predict"].append(t2 - t1)
        l1 = os.getloadavg()[0]
        loads.append((l0, l1))
        print("round %d done, load %.2f -> %.2f" % (rd + 1, l0, l1), flush=True)
    if outf:
        with open(outf, "w") as o:
            for e, v in res.items():
                o.write(json.dumps(dict(entry=list(e), size=size, fit=v["fit"], predict=v["predict"], extra=v["extra"], loads=loads)) + "\n")
    lo = min(min(a, b) for a, b in loads); hi = max(max(a, b) for a, b in loads)
    print("load average (1 min) over the rounds: min %.2f max %.2f; start of first round %.2f, end of last %.2f%s" % (lo, hi, loads[0][0], loads[-1][1], "   FLAGGED: load >= 2" if hi >= 2 or loads[0][0] >= 2 or loads[-1][1] >= 2 else ""))
    print("%-18s %5s %7s | %10s %7s | %10s %7s" % ("entry", "trees", "threads", "fit min s", "spread", "pred min s", "spread"))
    for e in entries:
        v = res[e]
        sf, sp = min(v["fit"]), min(v["predict"])
        print("%-18s %5d %7d | %10.4f %7.3f | %10.4f %7.3f" % (e[0], e[1], e[2], sf, max(v["fit"]) / sf, sp, max(v["predict"]) / sp))
    e1 = ("cancho", 10, 1)
    ex = [x for x in res[e1]["extra"] if x[0] or x[1]]
    if ex:
        print("serial part of fit at 1 thread (measured once, T-independent): edges %.3f s, bin matrix %.3f s" % (ex[0][0], ex[0][1]))
    hs = {e: set(x[2] for x in v["extra"]) for e, v in res.items() if e[0] == "cancho"}
    for trees in (10, 32):
        allh = set().union(*[hs[("cancho", trees, t)] for t in THREADS])
        print("forest hash at T=%d over threads %s: %s" % (trees, THREADS, "IDENTICAL " + next(iter(allh)) if len(allh) == 1 else "DIFFERENT " + str(allh)))


main()

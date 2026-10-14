#!/usr/bin/env python3
"""Spike F timing (docs/spikes.md 13.4): cancho spike_f vs scikit-learn RandomForestClassifier, N interleaved rounds.
usage: bench_f.py SET N EXE     (run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1)
cancho: fit_ms / predict_ms printed by the program (clock_ms around the region). sklearn: perf_counter around fit and around
predict_proba (n_jobs=1, random_state=0). Reports min, max/min, load at start and end, test AUC of both."""
import os, platform, subprocess, sys, time
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
st, N, exe = sys.argv[1], int(sys.argv[2]), sys.argv[3]; n = int(st[1:])
d = f"/Users/alfonso/Workspace/alpibrusl/lexsys/ml-data/{st}"
X = np.fromfile(d + "/x.bin", "<f8").reshape(n, 20); y = np.fromfile(d + "/y.bin", "<f8"); ntr = n - n // 5
ct = {"fit": [], "predict": [], "grow": [], "edges": [], "bins": []}; sk = {"fit": [], "predict": []}
la0 = os.getloadavg()
for _ in range(N):
    out = subprocess.run([exe, d + "/x.bin", d + "/y.bin", str(n), "20261012"], capture_output=True, text=True, check=True).stdout.splitlines()
    for l in out:
        if l.startswith("#"):
            a, b = l[1:].split()
            for k in ct:
                if a == k + "_ms": ct[k].append(int(b) / 1000)
    pc = np.array([int(l.split()[2], 16) for l in out if l.startswith("P ")], dtype=np.uint64).view("<f8")
    m = RandomForestClassifier(n_estimators=10, max_depth=8, max_features=4, bootstrap=True, n_jobs=1, random_state=0)
    t = time.perf_counter(); m.fit(X[:ntr], y[:ntr]); sk["fit"].append(time.perf_counter() - t)
    t = time.perf_counter(); p = m.predict_proba(X[ntr:])[:, 1]; sk["predict"].append(time.perf_counter() - t)
la1 = os.getloadavg()
print("machine %s %s, set %s n=%d ntr=%d, N=%d, 1 thread, sklearn %s numpy %s" % (platform.machine(), platform.platform(terse=True), st, n, ntr, N, sklearn.__version__, np.__version__))
print("load_start %.2f %.2f %.2f  load_end %.2f %.2f %.2f" % (la0 + la1))
for name, t in (("cancho fit", ct["fit"]), ("cancho predict", ct["predict"]), ("  cancho edges", ct["edges"]), ("  cancho bins", ct["bins"]), ("  cancho grow", ct["grow"]), ("sklearn fit", sk["fit"]), ("sklearn predict", sk["predict"])):
    print("%-16s min %.4fs spread %.3f all %s" % (name, min(t), max(t) / min(t), " ".join("%.3f" % v for v in t)))
print("test AUC cancho %.6f sklearn %.6f diff %.6f" % (roc_auc_score(y[ntr:], pc), roc_auc_score(y[ntr:], p), abs(roc_auc_score(y[ntr:], pc) - roc_auc_score(y[ntr:], p))))
print("ratio cancho/sklearn: fit %.4f  predict %.4f" % (min(ct["fit"]) / min(sk["fit"]), min(ct["predict"]) / min(sk["predict"])))

#!/usr/bin/env python3
"""sklearn LogisticRegression on the same split: test AUC and fit time (min of N).
usage: sklearn_l.py DATADIR [N=5]     C=inf (no regularisation), lbfgs, tol/max_iter at sklearn defaults except max_iter=1000.
Prints machine-readable lines; the load average is read at start and end (os.getloadavg)."""
import sys, time, os, platform
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sklearn
from oracle_l import load, split

d = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
X, y = load(d); ntr = split(y.size)
la0 = os.getloadavg(); ts = []
for _ in range(N):
    m = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    t = time.perf_counter(); m.fit(X[:ntr], y[:ntr]); ts.append(time.perf_counter() - t)
la1 = os.getloadavg()
auc = roc_auc_score(y[ntr:], m.decision_function(X[ntr:]))
print("sklearn", sklearn.__version__, "numpy", np.__version__, platform.machine(), "n_iter", int(m.n_iter_[0]))
print("test_auc %.6f fit_s_min %.4f fit_s_all %s" % (auc, min(ts), " ".join("%.4f" % t for t in ts)))
print("load_start %.2f %.2f %.2f load_end %.2f %.2f %.2f" % (la0 + la1))

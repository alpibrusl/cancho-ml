#!/usr/bin/env python3
"""Interleaved fit timing for spike S (docs/spikes.md section 10.3) and the pass counts (G-passes).
usage: bench_s.py --set n100000 -N 7 --cranelift EXE [--llvm EXE]
Each round runs: cancho Newton, cancho L-BFGS (Cranelift, and LLVM if given), numpy Newton, numpy L-BFGS (same algorithms as
oracle_s.py, BLAS sums), scikit-learn LogisticRegression(C=inf, lbfgs). Fit only. Prints min, spread (max/min), all times,
load average at start and end, passes. scikit-learn's passes: n_iter_ and the nfev of the scipy L-BFGS-B result of the actual
fit call (scipy.optimize.minimize is wrapped to record it)."""
import argparse, os, platform, subprocess, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oracle_l, oracle_s
from check_oracle_s import NEWTON, LBFGS, of_bits
import scipy, scipy.optimize, sklearn
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model._logistic as slog

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/Users/alfonso/Workspace/alpibrusl/lexsys/ml-data")
ap.add_argument("--set", default="n100000"); ap.add_argument("-N", type=int, default=7)
ap.add_argument("--cranelift", required=True); ap.add_argument("--llvm")
a = ap.parse_args()
d = os.path.join(a.data, a.set); n = int(a.set[1:])
X, y = oracle_l.load(d); ntr = oracle_l.split(n); Xt, yt = X[:ntr], y[:ntr]; A = oracle_s.augment(Xt)

def cancho(exe, p):
    out = subprocess.run([exe, d + "/x.bin", d + "/y.bin", str(n)] + [str(p[k]) for k in ("algo", "iters", "m", "trials", "t0e")],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    info = {l.split()[0][1:]: int(l.split()[1]) for l in out if l.startswith("#") and len(l.split()) == 2}
    return info["fit_ms"] / 1000.0, info["passes"]

def timed(f):
    t = time.perf_counter(); r = f(); return time.perf_counter() - t, r

# scikit-learn's pass count, from the real fit
rec = {}
_orig = scipy.optimize.minimize
def _wrap(*args, **kw):
    r = _orig(*args, **kw); rec["nfev"] = int(r.nfev); rec["nit"] = int(r.nit); rec["method"] = kw.get("method"); return r
slog.optimize.minimize = _wrap
def sk():
    m = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    t = time.perf_counter(); m.fit(Xt, yt); t = time.perf_counter() - t
    rec["n_iter_"] = int(m.n_iter_[0]); return t, rec["nfev"]

variants = [("cancho Newton (K=6)", lambda: cancho(a.cranelift, NEWTON)), ("cancho L-BFGS (K=8, m=5)", lambda: cancho(a.cranelift, LBFGS))]
if a.llvm:
    variants += [("cancho-llvm Newton", lambda: cancho(a.llvm, NEWTON)), ("cancho-llvm L-BFGS", lambda: cancho(a.llvm, LBFGS))]
variants += [("numpy Newton (same alg.)", lambda: timed(lambda: oracle_s.newton(A, yt, 6))[0:1] + (6,)),
             ("numpy L-BFGS (same alg.)", lambda: (lambda r: (r[0], r[1][1]))(timed(lambda: oracle_s.lbfgs(A, yt, 8, 5)))),
             ("scikit-learn lbfgs C=inf", sk)]
times = {k: [] for k, _ in variants}; passes = {}
la0 = os.getloadavg()
for _ in range(a.N):
    for k, f in variants:
        t, ps = f(); times[k].append(t); passes[k] = ps
la1 = os.getloadavg()
print("machine %s %s, set %s n=%d (train %d), N=%d, numpy %s scipy %s sklearn %s" % (platform.machine(), platform.platform(terse=True), a.set, n, ntr, a.N, np.__version__, scipy.__version__, sklearn.__version__))
print("load_start %.2f %.2f %.2f  load_end %.2f %.2f %.2f" % (la0 + la1))
print("sklearn: n_iter_ %d, scipy %s nit %d nfev %d" % (rec["n_iter_"], rec["method"], rec["nit"], rec["nfev"]))
for k, _ in variants:
    t = times[k]
    print("%-26s min %.4fs  spread %.3f  passes %s  all %s" % (k, min(t), max(t) / min(t), passes[k], " ".join("%.4f" % v for v in t)))

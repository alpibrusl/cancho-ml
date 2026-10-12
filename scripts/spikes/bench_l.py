#!/usr/bin/env python3
"""Interleaved timing for spike L (docs/spikes.md section 4, G-noise): N rounds, each round runs every variant once.
usage: bench_l.py --set n100000 --iters 100 --lr 0.5 -N 7 --cranelift EXE [--llvm EXE]
Variants: cancho (fit time printed by the program itself, clock_ms around the iterations only), cancho-llvm, numpy-blas
(oracle_l.fit, wall time of the fit). Prints min, max/min spread, all times, load average at start and end, test AUC
of the cancho coefficients and of numpy's. Resolution of cancho's clock is 1 ms."""
import argparse, os, platform, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oracle_l
from check_oracle import bits_of, of_bits
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/Users/alfonso/Workspace/alpibrusl/lexsys/ml-data")
ap.add_argument("--set", default="n100000"); ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--lr", type=float, default=0.5); ap.add_argument("-N", type=int, default=7)
ap.add_argument("--cranelift", required=True); ap.add_argument("--llvm")
a = ap.parse_args()
d = os.path.join(a.data, a.set); n = int(a.set[1:])
X, y = oracle_l.load(d); ntr = oracle_l.split(n)

def cancho(exe):
    out = subprocess.run([exe, d + "/x.bin", d + "/y.bin", str(n), "%016x" % bits_of(a.lr), str(a.iters)], capture_output=True, text=True, check=True).stdout.splitlines()
    coef = np.array([of_bits(int(l, 16)) for l in out if not l.startswith("#")])
    ms = [int(l.split()[1]) for l in out if l.startswith("#fit_ms")][0]
    return ms / 1000.0, coef

def numpy_fit():
    t = time.perf_counter(); w, b = oracle_l.fit(X[:ntr], y[:ntr], a.lr, a.iters, "blas"); t = time.perf_counter() - t
    return t, np.concatenate([w, [b]])

variants = [("cancho-cranelift", lambda: cancho(a.cranelift))]
if a.llvm: variants.append(("cancho-llvm", lambda: cancho(a.llvm)))
variants.append(("numpy-blas", numpy_fit))
times = {k: [] for k, _ in variants}; coefs = {}
la0 = os.getloadavg()
for _ in range(a.N):
    for k, f in variants:
        t, c = f(); times[k].append(t); coefs[k] = c
la1 = os.getloadavg()
print("machine %s %s, set %s, n=%d, iters=%d, lr=%g, N=%d, numpy %s" % (platform.machine(), platform.platform(terse=True), a.set, n, a.iters, a.lr, a.N, np.__version__))
print("load_start %.2f %.2f %.2f  load_end %.2f %.2f %.2f" % (la0 + la1))
for k, _ in variants:
    t = times[k]; c = coefs[k]
    auc = roc_auc_score(y[ntr:], X[ntr:] @ c[:20] + c[20])
    print("%-17s min %.3fs  spread(max/min) %.3f  test_auc %.6f  all %s" % (k, min(t), max(t) / min(t), auc, " ".join("%.3f" % v for v in t)))

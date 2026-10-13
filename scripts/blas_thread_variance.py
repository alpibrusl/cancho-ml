#!/usr/bin/env python3
"""Does the installed BLAS (numpy/scipy, Accelerate here) give the same bits at every thread count? docs/linalg.md section 1.3.
Runs the same computations in fresh processes with VECLIB_MAXIMUM_THREADS = OPENBLAS_NUM_THREADS = OMP_NUM_THREADS = t and
prints the hash of each result (the hash of scripts/oracle_linalg.py). Equal hashes for every t would mean the library is
thread-count independent on these inputs; different ones show it is not. A property of one library on one machine."""
import os, subprocess, sys, json

CODE = r"""
import sys, json, numpy as np
sys.path.insert(0, %r)
import oracle_linalg as O
from scipy.linalg import blas
s1, s2 = O.derive(1, 1), O.derive(1, 2)
out = {}
x, y = O.elements(s1, 1000000), O.elements(s2, 1000000)
out["dot 1e6"] = O.hash_bits(np.array([np.dot(x, y)]))
X = O.elements(s1, 100000 * 64).reshape(100000, 64); r = O.elements(s2, 100000)
out["gemv_t 1e5x64"] = O.hash_bits(np.dot(r, X))
out["gemv_n 1e5x64"] = O.hash_bits(np.dot(X, O.elements(s2, 64)))
X2 = O.elements(s1, 200000 * 20).reshape(200000, 20)
out["syrk 2e5x20"] = O.hash_bits(np.triu(blas.dsyrk(1.0, X2.T, trans=0)))
A, B = O.elements(s1, 512 * 512).reshape(512, 512), O.elements(s2, 512 * 512).reshape(512, 512)
out["gemm 512"] = O.hash_bits(np.dot(A, B))
print(json.dumps(out))
""" % os.path.dirname(os.path.abspath(__file__))

def main():
    threads = [int(t) for t in (sys.argv[1] if len(sys.argv) > 1 else "1,2,4,8,16").split(",")]
    rows = {}
    for t in threads:
        env = dict(os.environ, VECLIB_MAXIMUM_THREADS=str(t), OPENBLAS_NUM_THREADS=str(t), OMP_NUM_THREADS=str(t))
        r = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(r.stderr[-800:])
        rows[t] = json.loads(r.stdout)
    for k in rows[threads[0]]:
        hs = [rows[t][k] for t in threads]
        print("%-16s %s  -> %s" % (k, " ".join("%016x" % h for h in hs), "identical at every thread count" if len(set(hs)) == 1 else "DIFFERS between thread counts (%d distinct)" % len(set(hs))))

if __name__ == "__main__":
    main()

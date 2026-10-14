#!/usr/bin/env python3
"""Interleaved timing of several spike_l executables (fit_ms printed by each), min of N, spread, load at start/end.
usage: bench_variants.py --set n100000 -N 7 EXE [EXE ...]   (also prints whether the 21 coefficient lines equal the first EXE's)"""
import argparse, os, subprocess, platform
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/Users/alfonso/Workspace/alpibrusl/lexsys/ml-data")
ap.add_argument("--set", default="n100000"); ap.add_argument("-N", type=int, default=7)
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("exes", nargs="+")
a = ap.parse_args()
d = os.path.join(a.data, a.set); n = int(a.set[1:])
def run(exe):
    out = subprocess.run([exe, d + "/x.bin", d + "/y.bin", str(n), "3fe0000000000000", str(a.iters)], capture_output=True, text=True, check=True).stdout.splitlines()
    return int([l for l in out if l.startswith("#fit_ms")][0].split()[1]) / 1000.0, [l for l in out if not l.startswith("#")]
ts = {e: [] for e in a.exes}; co = {}
la0 = os.getloadavg()
for _ in range(a.N):
    for e in a.exes:
        t, c = run(e); ts[e].append(t); co[e] = c
la1 = os.getloadavg()
print("%s set %s N=%d iters=%d load_start %.2f %.2f %.2f load_end %.2f %.2f %.2f" % ((platform.machine(), a.set, a.N, a.iters) + la0 + la1))
for e in a.exes:
    t = ts[e]
    print("%-22s min %.3fs spread %.3f same_bits_as_first %s all %s" % (os.path.basename(e), min(t), max(t) / min(t), co[e] == co[a.exes[0]], " ".join("%.3f" % v for v in t)))

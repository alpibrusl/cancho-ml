#!/usr/bin/env python3
"""Spike F gate (docs/spikes.md 13.3): G-F-ident and G-F-mutants.
usage: check_f.py [--sizes 1000,100000] [--mutants] [--cancho PATH]
Builds spike_f.cho (and each mutant: one asserted-unique source edit) into build/, runs it and compares every E/N/P line,
byte for byte, with the oracle's output (computed by oracle_f.py and cached in out/). Exit 1 if the real program differs
from the oracle or if any mutant is accepted."""
import argparse, os, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(os.path.dirname(H))
DATA = "/Users/alfonso/Workspace/alpibrusl/lexsys/ml-data"; SEED = "20261012"
MUT = {
 "tie":   ("found == false || gt(nn, dd, bn, bd)", "found == false || gt(bn, bd, nn, dd) == false"),
 "bin":   ("if e[t - 1] <= x {", "if e[t - 1] < x {"),
 "boot":  ("let r = uni(s, ntr);", "var r = uni(s, ntr);\n        if t == 0 && k == 0 {\n            r = (r + 1) % ntr;\n        }"),
 "dir":   None,   # two sites
 "depth": ("if dep < 8 &&", "if dep < 7 &&"),
 "mtry":  ("while q < 4 {", "while q < 3 {"),
}
DIR2 = [("if int_of(col[r]) <= bb {", "if int_of(col[r]) > bb {"), ("if row[wk[2048 + slot]] < fl[1280 + slot] {", "if row[wk[2048 + slot]] >= fl[1280 + slot] {")]

def build(cancho, src, exe):
    r = subprocess.run([cancho, "build", src, "--std", "-o", exe], capture_output=True, text=True)
    if r.returncode: print(r.stdout, r.stderr); sys.exit(2)

def run(exe, s):
    d = f"{DATA}/n{s}"
    out = subprocess.run([exe, d + "/x.bin", d + "/y.bin", str(s), SEED], capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if not l.startswith("#")]

def oracle(s):
    p = f"{R}/out/oracle_n{s}.txt"
    if not os.path.exists(p):
        subprocess.run([sys.executable, f"{H}/oracle_f.py", f"{DATA}/n{s}", SEED, p], check=True, capture_output=True)
    return open(p).read().splitlines()

def diff(a, b):
    if len(a) != len(b): return "line counts %d vs %d" % (len(a), len(b))
    bad = [i for i in range(len(a)) if a[i] != b[i]]
    return None if not bad else "%d differing lines, first: %r vs %r" % (len(bad), a[bad[0]], b[bad[0]])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes", default="1000,100000"); ap.add_argument("--mutants", action="store_true")
    ap.add_argument("--cancho", default="/Users/alfonso/Workspace/alpibrusl/lexsys/cancho-compiler-a4572ea/target/release/cancho")
    a = ap.parse_args(); sizes = a.sizes.split(","); os.makedirs(f"{R}/build", exist_ok=True)
    src = open(f"{H}/spike_f.cho").read(); fail = False
    exe = f"{R}/build/spike_f"; build(a.cancho, f"{H}/spike_f.cho", exe)
    for s in sizes:
        d = diff(run(exe, s), oracle(s)); print("ident  n=%-8s %s" % (s, "IDENTICAL" if d is None else "DIFFERENT: " + d)); fail |= d is not None
    if a.mutants:
        for name, edit in MUT.items():
            edits = DIR2 if name == "dir" else [edit]; m = src
            for f, t in edits:
                assert m.count(f) == 1, (name, f, m.count(f)); m = m.replace(f, t)
            p = f"{R}/build/mut_{name}.cho"; open(p, "w").write(m); build(a.cancho, p, f"{R}/build/mut_{name}")
            res = []
            for s in sizes:
                d = diff(run(f"{R}/build/mut_{name}", s), oracle(s)); res.append("n=%s %s" % (s, "same" if d is None else "differs"))
            rej = any("differs" in r for r in res)
            print("mutant %-6s %s   (%s)" % (name, "REJECTED" if rej else "SURVIVED", "; ".join(res))); fail |= not rej
    sys.exit(1 if fail else 0)
main()

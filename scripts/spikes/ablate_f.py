#!/usr/bin/env python3
"""Profile of spike F by ablation (docs/spikes.md 14.5). Every variant is spike_f.cho with one piece done TWICE (the result is
unchanged, the time grows by that piece's cost) or, for `float_cmp`, with the exact comparison replaced by a float cross
product (the trees may then differ). Min of N interleaved runs of each, phase times as printed by the program itself.
usage: ablate_f.py SET N"""
import os, re, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(os.path.dirname(H))
CH = "/Users/alfonso/Workspace/alpibrusl/lexsys/cancho-compiler-0207556/target/release/cancho"
DATA = "/Users/alfonso/Workspace/alpibrusl/lexsys/ml-data"
src = open(H + "/spike_f.cho").read()

def sub(s, a, b):
    assert s.count(a) == 1, (a, s.count(a)); return s.replace(a, b)

def v_hist(s):
    s = sub(s, "            k = lo;\n            while k < hi {\n                let r = idx[k];\n                let yy",
               "            var rep = 0;\n            while rep < 2 {\n            j = 0;\n            while j < 512 {\n                wk[j] = 0;\n                j = j + 1;\n            }\n            k = lo;\n            while k < hi {\n                let r = idx[k];\n                let yy")
    return sub(s, "                wk[i3] = wk[i3] + 1;\n                k = k + 1;\n            }\n", "                wk[i3] = wk[i3] + 1;\n                k = k + 1;\n            }\n            rep = rep + 1;\n            }\n")
def v_scan(s):
    s = sub(s, "            var q = 0;\n            while q < 4 {\n                var nl = 0;", "            var rep2 = 0;\n            while rep2 < 2 {\n            var q = 0;\n            while q < 4 {\n                var nl = 0;")
    return sub(s, "                q = q + 1;\n            }\n        }\n        if found == false {", "                q = q + 1;\n            }\n            rep2 = rep2 + 1;\n            }\n        }\n        if found == false {")
def v_float(s): return sub(s, "found == false || gt(nn, dd, bn, bd)", "found == false || float_of(nn) * float_of(bd) > float_of(bn) * float_of(dd)")
def v_part(s):
    s = sub(s, "            var a = lo;\n            var z = hi - 1;\n            while a <= z {", "            var a = lo;\n            var rep3 = 0;\n            while rep3 < 2 {\n            a = lo;\n            var z = hi - 1;\n            while a <= z {")
    return sub(s, "                    z = z - 1;\n                }\n            }\n", "                    z = z - 1;\n                }\n            }\n            rep3 = rep3 + 1;\n            }\n")
def v_boot(s):
    s = sub(s, "    var k = 0;\n    var root1 = 0;\n    while k < ntr {", "    var k = 0;\n    var root1 = 0;\n    var rb = 0;\n    while rb < 2 {\n    s = derive1(derive1(seed, 1), t);\n    k = 0;\n    root1 = 0;\n    while k < ntr {")
    return sub(s, "        root1 = root1 + int_of(yb[r]);\n        k = k + 1;\n    }\n", "        root1 = root1 + int_of(yb[r]);\n        k = k + 1;\n    }\n    rb = rb + 1;\n    }\n")
def v_radix(s): return sub(s, "        radix(ka, kb, cnt, ntr);\n", "        radix(ka, kb, cnt, ntr);\n        radix(ka, kb, cnt, ntr);\n")
def v_bins(s): return sub(s, "                make_bins(contents(xr), contents(fr), ntr, contents(bw));\n", "                make_bins(contents(xr), contents(fr), ntr, contents(bw));\n                make_bins(contents(xr), contents(fr), ntr, contents(bw));\n")
VAR = {"base": lambda s: s, "hist_x2": v_hist, "scan_x2": v_scan, "float_cmp": v_float, "part_x2": v_part, "boot_x2": v_boot, "radix_x2": v_radix, "bins_x2": v_bins}

def main():
    st, N = sys.argv[1], int(sys.argv[2]); n = int(st[1:]); d = f"{DATA}/{st}"; os.makedirs(R + "/build", exist_ok=True)
    exes = {}
    for k, f in VAR.items():
        p = f"{R}/build/abl_{k}.cho"; open(p, "w").write(f(src))
        r = subprocess.run([CH, "build", p, "--std", "-o", f"{R}/build/abl_{k}"], capture_output=True, text=True)
        assert r.returncode == 0, (k, r.stderr); exes[k] = f"{R}/build/abl_{k}"
    res = {k: {} for k in VAR}; la0 = os.getloadavg()
    for _ in range(N):
        for k, e in exes.items():
            out = subprocess.run([e, d + "/x.bin", d + "/y.bin", str(n), "20261012"], capture_output=True, text=True, check=True).stdout
            for l in out.splitlines():
                if l.startswith("#"):
                    a, b = l[1:].split(); res[k].setdefault(a, []).append(int(b))
    la1 = os.getloadavg()
    print("set", st, "N", N, "load start %.2f %.2f %.2f end %.2f %.2f %.2f" % (la0 + la1))
    keys = ["edges_ms", "bins_ms", "grow_ms", "fit_ms", "predict_ms"]
    print("%-10s" % "variant", " ".join("%-14s" % k for k in keys))
    for k in VAR:
        print("%-10s" % k, " ".join("%-14s" % ("%d (x%.2f)" % (min(res[k][q]), max(res[k][q]) / max(1, min(res[k][q])))) for q in keys))
main()

#!/usr/bin/env python3
"""Markdown tables for docs/simd-ceiling.md from docs/bench/simd-gate.jsonl (scripts/simd/bench_simd.py) and the copied cancho
figures of docs/bench/gate-1thread.jsonl.   python3 scripts/simd/report.py [jsonl]  > tables.md"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "bench", "simd-gate.jsonl")
rows = [json.loads(l) for l in open(src)]
cancho = {}
for l in open(os.path.join(ROOT, "docs", "bench", "gate-1thread.jsonl")):
    j = json.loads(l); cancho[(j["kernel"], j["n"], j["d"])] = j
ORDER = ["dot", "axpy", "gemv_n", "gemv_t", "syrk", "chol"]
rows.sort(key=lambda r: (ORDER.index(r["kernel"]), r["n"], r["d"]))


def us(s):
    if s >= 1e-1: return "%.3f s" % s
    if s >= 1e-3: return "%.3f ms" % (s * 1e3)
    return "%.2f us" % (s * 1e6)


def shape(r):
    return ("d=%d" % r["n"]) if r["kernel"] == "chol" else (("n=%d" % r["n"]) + (" d=%d" % r["d"] if r["d"] else ""))


def best(r, pre):
    c = [(k, v["min"]) for k, v in r["variants"].items() if k[0] == pre and k[1:].isdigit()]
    return min(c, key=lambda t: t[1]) if c else (None, None)


def flags(r):
    f = []
    if r["worst_spread"] > 1.25: f.append("spread %.2f" % r["worst_spread"])
    if r["flag_load"]: f.append("LOAD>=2")
    return ", ".join(f)


def bytes_of(r):
    return {"gemv_n": r["n"] * r["d"] * 8, "gemv_t": r["n"] * r["d"] * 8, "dot": 2 * r["n"] * 8, "axpy": 3 * r["n"] * 8}.get(r["kernel"])


for kern in ORDER:
    rs = [r for r in rows if r["kernel"] == kern]
    if not rs: continue
    Ls = sorted({int(k[1:]) for r in rs for k in r["variants"] if k[0] == "N" and k[1:].isdigit()})
    print("\n#### %s (unfused, bits identical to the scalar reference of each order)\n" % kern)
    if kern == "axpy":
        print("| shape | C scalar (CS) | C NEON | numpy | cancho (copied) | NEON / numpy | CS / numpy | cancho / numpy | verdict | notes |\n|---|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            v = r["variants"]; n = v["N"]["min"]; c = cancho.get((kern, r["n"], r["d"]))
            print("| %s | %s | %s | %s | %s | %.2f | %.2f | %.2f | %s | %s |" % (shape(r), us(v["CS"]["min"]), us(n), us(r["numpy_s"]), us(c["cancho_s"]) if c else "-",
                  n / r["numpy_s"], v["CS"]["min"] / r["numpy_s"], c["cancho_s"] / r["numpy_s"] if c else 0,
                  "MATCH" if n <= r["numpy_s"] else "NOT MATCH (%.2fx)" % (n / r["numpy_s"]), flags(r)))
        continue
    hdr = "| shape | CS | " + " | ".join("N%d" % L for L in Ls) + " | best L | numpy | cancho (copied) | N(best) / numpy | CS / numpy | cancho / numpy | verdict | bits | notes |"
    print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
    for r in rs:
        v = r["variants"]; bk, bt = best(r, "N"); c = cancho.get((kern, r["n"], r["d"]))
        cells = " | ".join(us(v["N%d" % L]["min"]) if ("N%d" % L) in v else "-" for L in Ls)
        ver = {v["N%d" % L]["ver"] for L in Ls if ("N%d" % L) in v}
        verdict = "MATCH" if bt <= r["numpy_s"] else "NOT MATCH (%.2fx)" % (bt / r["numpy_s"])
        extra = ""
        if kern == "chol" and "raw_s" in r:
            extra = "; vs raw dpotrf %s: %.2fx" % (us(r["raw_s"]), bt / r["raw_s"])
        print("| %s | %s | %s | %s | %s | %s | %.2f | %.2f | %.2f | %s | %s | %s |" % (
            shape(r), us(v["CS"]["min"]), cells, bk, us(r["numpy_s"]), us(c["cancho_s"]) if c else "-", bt / r["numpy_s"], v["CS"]["min"] / r["numpy_s"],
            c["cancho_s"] / r["numpy_s"] if c else 0, verdict, "/".join(sorted(ver)), (flags(r) + extra).strip("; ")))
    print("\n#### %s, fma variant (CHANGES BITS; not in the verdict)\n" % kern)
    hdr = "| shape | " + " | ".join("F%d" % L for L in Ls) + " | best | F(best) / numpy | F(best) / N(best) | bits |"
    print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
    for r in rs:
        v = r["variants"]; fk, ft = best(r, "F"); bk, bt = best(r, "N")
        cells = " | ".join(us(v["F%d" % L]["min"]) if ("F%d" % L) in v else "-" for L in Ls)
        ver = {v["F%d" % L]["ver"] for L in Ls if ("F%d" % L) in v}
        print("| %s | %s | %s | %.2f | %.2f | %s |" % (shape(r), cells, fk, ft / r["numpy_s"], ft / bt, "/".join(sorted(ver))))

import re, math
probe = {}   # N -> best GB/s over the load styles of scripts/simd/bw.c, and the compute peak of scripts/simd/peak.c
pk = {}
for l in open(os.path.join(ROOT, "docs", "bench", "simd-probes.txt")):
    m = re.match(r"N=(\d+) .*?\s([\d.]+) GB/s", l)
    if m: probe[int(m.group(1))] = max(probe.get(int(m.group(1)), 0), float(m.group(2)))
    m = re.match(r"(vfma|unfused pair).*?=\s+([\d.]+) G vector ops/s", l)
    if m: pk[m.group(1)] = float(m.group(2))
PEAK_U = pk["unfused pair"] / 2 * 2   # vector MAC/s (G) x 2 lanes = GMAC/s
PEAK_F = pk["vfma"] * 2
def stream_bound(b):
    N = min(probe, key=lambda n: abs(math.log(b / (n * 8))))
    return probe[N], N
print("\n#### Bandwidth of the memory-bound cells (bytes of X read / time; dot counts both vectors, axpy counts x read, y read and written)\n")
print("| kernel | shape | bytes | N(best) GB/s | CS GB/s | numpy GB/s | cancho GB/s (copied) | best read probe GB/s (size class) | time at probe rate / numpy |\n|---|---|---|---|---|---|---|---|---|")
for r in rows:
    b = bytes_of(r)
    if not b or r["kernel"] in ("dot", "axpy") and False: continue
    v = r["variants"]; bk, bt = best(r, "N")
    if bk is None: bt = v["N"]["min"]; bk = "N"
    c = cancho.get((r["kernel"], r["n"], r["d"]))
    pg, pn = stream_bound(b)
    print("| %s | %s | %.1f MB | %.1f (%s) | %.1f | %.1f | %s | %.1f (N=%d) | %.2f |" % (r["kernel"], shape(r), b / 1e6, b / bt / 1e9, bk, b / v["CS"]["min"] / 1e9, b / r["numpy_s"] / 1e9,
          ("%.1f" % (b / c["cancho_s"] / 1e9)) if c else "-", pg, pn, (b / (pg * 1e9)) / r["numpy_s"]))

print("\n#### What the fixed order costs on dot: N(L) with blocks and a lane tree per block vs 16 free lanes with no blocks\n")
for r in rows:
    if r["kernel"] == "dot":
        v = r["variants"]; bk, bt = best(r, "N")
        print("* dot n=%d: free16 %s, N(best=%s) %s, ratio fixed/free = %.2f; N by L: %s" % (r["n"], us(v["free16"]["min"]), bk, us(bt), bt / v["free16"]["min"],
              ", ".join("%s %s" % (k, us(v[k]["min"])) for k in sorted((k for k in v if k[0] == "N"), key=lambda k: int(k[1:])))))
print("\n#### Spreads and load per cell\n")
print("| kernel | shape | spread numpy | spread CS | spread N(best) | load start / end | attempts |\n|---|---|---|---|---|---|---|")
for r in rows:
    v = r["variants"]; bk, bt = best(r, "N"); bk = bk or "N"
    print("| %s | %s | %.2f | %.2f | %.2f | %.1f / %.1f | %d |" % (r["kernel"], shape(r), r["numpy_spread"], v["CS"]["spread"], v[bk]["spread"], r["load_start"], r["load_end"], r["attempt"]))

print("\n#### Compute-bound cells against the measured register-only peak (scripts/simd/peak.c: unfused %.1f GMAC/s, fma %.1f GMAC/s)\n" % (PEAK_U, PEAK_F))
print("| kernel | shape | useful MACs | N(best) | GMAC/s | % of unfused peak | F(best) | GMAC/s | % of fma peak | CS GMAC/s |\n|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if r["kernel"] not in ("syrk", "chol"): continue
    macs = r["n"] * r["d"] * (r["d"] + 1) / 2 if r["kernel"] == "syrk" else r["n"] ** 3 / 6.0
    v = r["variants"]; bk, bt = best(r, "N"); fk, ft = best(r, "F")
    print("| %s | %s | %.3g | %s | %.2f | %.0f%% | %s | %.2f | %.0f%% | %.2f |" % (r["kernel"], shape(r), macs, bk, macs / bt / 1e9, 100 * macs / bt / 1e9 / PEAK_U, fk, macs / ft / 1e9, 100 * macs / ft / 1e9 / PEAK_F, macs / v["CS"]["min"] / 1e9))

print("\n#### Verdict table: would a SIMD cancho at this ceiling MATCH numpy? (unfused, bit-identical to the scalar reference; fma shown apart)\n")
print("| kernel | shape | numpy | N(best L) | N / numpy | verdict | min(CS,N) / numpy | F(best) / numpy (bits change) | cancho (copied) / N(best) | notes |\n|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    v = r["variants"]; bk, bt = best(r, "N"); fk, ft = best(r, "F"); c = cancho.get((r["kernel"], r["n"], r["d"]))
    if bk is None: bk, bt, fk, ft = "N", v["N"]["min"], "F", v["F"]["min"]
    verdict = "MATCH" if bt <= r["numpy_s"] else "NOT MATCH"
    note = flags(r)
    if r["kernel"] == "chol" and "raw_s" in r: note = (note + "; vs raw dpotrf: %.2f" % (bt / r["raw_s"])).strip("; ")
    print("| %s | %s | %s | %s %s | %.2f | %s | %.2f | %.2f | %s | %s |" % (r["kernel"], shape(r), us(r["numpy_s"]), bk, us(bt), bt / r["numpy_s"], verdict, min(v["CS"]["min"], bt) / r["numpy_s"], ft / r["numpy_s"],
          ("%.2fx" % (c["cancho_s"] / bt)) if c else "-", note))

print("\n#### Which L: time of N(L) over the time of the best L, per cell (1.00 = best); worst case and geometric mean over the cells of each kernel\n")
Ls_all = [1, 2, 4, 8, 16]
print("| kernel | " + " | ".join("L=%d worst / geomean" % L for L in Ls_all) + " |\n|---|" + "---|" * len(Ls_all))
allr = {L: [] for L in Ls_all}
for kern in ORDER:
    if kern == "axpy": continue
    rs = [r for r in rows if r["kernel"] == kern]
    out = []
    for L in Ls_all:
        xs = []
        for r in rs:
            v = r["variants"]; key = "N%d" % L
            if key not in v: continue
            b = min(vv["min"] for k, vv in v.items() if k[0] == "N" and k[1:].isdigit())
            xs.append(v[key]["min"] / b)
        if xs and len(xs) == len(rs):
            out.append("%.2f / %.2f" % (max(xs), math.exp(sum(math.log(x) for x in xs) / len(xs)))); allr[L] += xs
        else: out.append("n/a")
    print("| %s | %s |" % (kern, " | ".join(out)))
print("| all cells where L exists for every kernel (dot, gemv_n, gemv_t, syrk, chol) | " + " | ".join(("%.2f / %.2f" % (max(allr[L]), math.exp(sum(math.log(x) for x in allr[L]) / len(allr[L])))) if L > 1 else "n/a (no dot, gemv_n)" for L in Ls_all) + " |")
print("\n#### The lane order against the current cancho order, vectorised or not: CS (current order, plain C) / N(best L) and N(1) (current order for gemv_t, syrk, vectorised across outputs) / N(best L)\n")
print("| kernel | shape | CS / N(best) | N1 / N(best) |\n|---|---|---|---|")
for r in rows:
    if r["kernel"] in ("axpy",): continue
    v = r["variants"]; bk, bt = best(r, "N")
    print("| %s | %s | %.2f | %s |" % (r["kernel"], shape(r), v["CS"]["min"] / bt, ("%.2f" % (v["N1"]["min"] / bt)) if "N1" in v else "-"))

#!/usr/bin/env python3
"""JSON lines of scripts/bench_linalg.py -> the markdown table of docs/linalg.md section 5.
    python3 scripts/bench_report.py docs/bench/gate-1thread.jsonl > table.md
Verdict (docs/linalg.md 1.3): MATCH if cancho/numpy <= 1.0 on a gate cell, else NOT MATCH with the factor; gemm is reference only
(no verdict, the ratio is printed). A cell whose spread (max/min of the 7 samples) exceeds 1.25 on either side, or whose load
average was >= 2 at either end, is marked."""
import json, sys

def fmt_t(s):
    if s >= 1: return "%.3f s" % s
    if s >= 1e-3: return "%.3f ms" % (s * 1e3)
    return "%.2f us" % (s * 1e6)

def shape(r):
    k, n, d = r["kernel"], r["n"], r["d"]
    if k in ("dot", "axpy"): return "n=%s" % format(n, ",")
    if k in ("chol", "fwd", "bwd"): return "d=%d" % n
    if k == "gemm": return "%d^3" % n
    return "n=%s d=%d" % (format(n, ","), d)

def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    meta = rows[0]["meta"] if rows else {}
    print("machine: %s; compiler %s; numpy %s; env %s; samples per cell %d (min of N, interleaved)\n" % (
        meta.get("machine"), meta.get("compiler"), meta.get("numpy"), meta.get("env"), rows[0]["samples"] if rows else 0))
    with_c = any("c_s" in r for r in rows)
    head = "| kernel | shape | cancho | numpy / scipy | cancho / numpy | verdict |" + (" C (same algorithm) | cancho / C |" if with_c else "") + " spread c / n | load start / end | R c / n |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for r in rows:
        ratio = r["ratio"]
        if r["kernel"] == "gemm":
            verdict = "reference only"
        else:
            verdict = "MATCH" if ratio <= 1.0 else "NOT MATCH (%.2fx)" % ratio
        marks = ""
        if max(r["spread_cancho"], r["spread_numpy"]) > 1.25: marks += " spread>1.25"
        if r["flag_load"]: marks += " LOAD>=2"
        line = "| %s | %s | %s | %s | %.2f | %s |" % (r["kernel"], shape(r), fmt_t(r["cancho_s"]), fmt_t(r["numpy_s"]), ratio, verdict + marks)
        if with_c:
            line += (" %s | %.2f |" % (fmt_t(r["c_s"]), r["cancho_over_c"])) if "c_s" in r else " | |"
        line += " %.2f / %.2f | %.1f / %.1f | %d / %d |" % (r["spread_cancho"], r["spread_numpy"], r["load_start"], r["load_end"], r["reps_cancho"], r["reps_numpy"])
        print(line)

if __name__ == "__main__":
    main()

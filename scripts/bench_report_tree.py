#!/usr/bin/env python3
"""Markdown tables from docs/bench/tree_<SIZE>.jsonl (written by scripts/bench_tree.py): minimum of the rounds, spread, and the
scaling of cancho against its own 1-thread time. usage: bench_report_tree.py SIZE"""
import json, sys
size = sys.argv[1]
rows = [json.loads(l) for l in open(f"docs/bench/tree_{size}.jsonl")]
loads = rows[0]["loads"]
print("size %s; %d rounds; load average (1 min) at the start / end of each round: %s" % (size, len(rows[0]["fit"]), " ".join("%.1f/%.1f" % (a, b) for a, b in loads)))
def cell(v):
    return min(v), max(v) / min(v)
print("(* spread above the 1.25 of the protocol; the cell was not rerun)")
base = {}
print("\n| entry | trees | threads | fit min (s) | spread | fit speedup | predict min (s) | spread | predict speedup |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    kind, trees, t = r["entry"]
    f, fs = cell(r["fit"]); p, ps = cell(r["predict"])
    if kind == "cancho" and t == 1:
        base[trees] = (f, p)
    sf = sp = ""
    if kind == "cancho" and trees in base:
        sf = "%.2fx" % (base[trees][0] / f); sp = "%.2fx" % (base[trees][1] / p)
    name = {"cancho": "cancho", "sklearn-rf": "scikit-learn RandomForestClassifier (exact splits)", "sklearn-hgb-1iter": "scikit-learn HistGradientBoosting, 1 iteration (context only)"}[kind]
    mark = lambda v: "%.2f%s" % (v, " *" if v > 1.25 else "")
    print("| %s | %d | %d | %.4f | %s | %s | %.4f | %s | %s |" % (name, trees, t, f, mark(fs), sf, p, mark(ps), sp))

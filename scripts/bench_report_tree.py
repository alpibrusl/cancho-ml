#!/usr/bin/env python3
"""Markdown tables from docs/bench/tree_<SIZE>.jsonl (written by scripts/bench_tree.py): minimum of the rounds, spread, and the
scaling of cancho against its own 1-thread time. usage: bench_report_tree.py SIZE [SUFFIX]      (SUFFIX "_b" for the replicate session)"""
import json, sys
compact = "--compact" in sys.argv
sys.argv = [a for a in sys.argv if a != "--compact"]
size = sys.argv[1]
suffix = sys.argv[2] if len(sys.argv) > 2 else ""
rows = [json.loads(l) for l in open(f"docs/bench/tree_{size}{suffix}.jsonl")]
loads = rows[0]["loads"]
if compact:
    # one line per entry: the fit minimum (s) and the predict minimum (s) by thread count, and the highest spread of the line
    print("size %s%s; %d rounds; load average (1 min) at the start / end of each round: %s" % (size, suffix, len(rows[0]["fit"]), " ".join("%.1f/%.1f" % (a, b) for a, b in rows[0]["loads"])))
    print("\n| entry | trees | what | 1 thread | 2 | 4 | 8 | 16 | worst spread |\n|---|---|---|---|---|---|---|---|---|")
    by = {}
    for r in rows:
        by.setdefault((r["entry"][0], r["entry"][1]), []).append(r)
    for (kind, trees), rs in by.items():
        for what in ("fit", "predict"):
            cells = {r["entry"][2]: min(r[what]) for r in rs}
            sp = max(max(r[what]) / min(r[what]) for r in rs)
            name = {"cancho": "cancho", "sklearn-rf": "scikit-learn RF", "sklearn-hgb-1iter": "scikit-learn HGB 1 iteration"}[kind]
            print("| %s | %d | %s (s) | %s | %.2f |" % (name, trees, what, " | ".join(("%.4f" % cells[t]) if t in cells else "" for t in (1, 2, 4, 8, 16)), sp))
    sys.exit(0)
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

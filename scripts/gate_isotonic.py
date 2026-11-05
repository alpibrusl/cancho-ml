#!/usr/bin/env python3
"""Gates I1 to I4 of docs/calibration.md for the isotonic fit (lib/metrics/isotonic.cho), through the compiled driver:

    METRICS_CHECK=build/metrics-check python3 scripts/gate_isotonic.py [--cases 400] [--seed 1] [--no-controls]

Four independent checks of every fit. (1) EXACT: pool-adjacent-violators in rational arithmetic (fractions) on the pooled ties, written from section 2.1 of the document: the driver's fitted values must equal the exact ones to a RELATIVE 1e-12
and the blocks (which points share a value) must be the same. (2) THE ARITHMETIC: the same algorithm in doubles, operation for operation as section 2.1 states it, must give the BITS of the driver. (3) OPTIMAL: for inputs of up to 8
points the weighted sum of squares of the fit equals the minimum over every monotone sequence that is constant on a partition into consecutive blocks (brute force over the 2^(n-1) partitions), and the fit is monotone. (4) scikit-learn,
if it is installed: its IsotonicRegression gives the same values within a relative 1e-9 (the library is the oracle's, not a dependency). Ties in x are pooled; the non-increasing fit is the non-decreasing fit of the negated values.
CONTROLS (they must change the answer on at least a tenth of the cases that can show it): ties not pooled, weights ignored, and a stack that merges only when the value below is greater by more than 0.05. Every refusal is reached with its tag.

Exit 0: every number is the expected one. Exit 1: a difference. Exit 2: the harness failed."""
import argparse
import itertools
import os
import pathlib
import struct
import subprocess
import sys
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = os.environ.get("METRICS_CHECK", str(ROOT / "build" / "metrics-check"))
sys.path.insert(0, str(ROOT / "scripts"))
from gate_calibrate import SM64, hx, unhx, fvec, drive  # noqa: E402


def parse(out):
    d = {"info": None, "points": [], "refused": None}
    for line in out.strip().split("\n"):
        w = line.split()
        if not w:
            continue
        if w[0] == "info":
            d["info"] = [int(v) for v in w[1:]]
        elif w[0] == "point":
            d["points"].append(w[2:])
        elif w[0] == "refused":
            d["refused"] = w[1]
    return d


def pooled(x, y, w):
    """the points after pooling ties: stable sort by x, then (sum w y, sum w) in that order, as doubles (what the driver does) and as fractions"""
    order = sorted(range(len(x)), key=lambda i: x[i])
    pts = []
    i = 0
    while i < len(order):
        r = order[i]
        swy, sw = w[r] * y[r], w[r]
        fwy, fw = Fraction(w[r]) * Fraction(y[r]), Fraction(w[r])
        j = i + 1
        while j < len(order) and x[order[j]] == x[r]:
            q = order[j]
            swy, sw = swy + w[q] * y[q], sw + w[q]
            fwy, fw = fwy + Fraction(w[q]) * Fraction(y[q]), fw + Fraction(w[q])
            j = j + 1
        pts.append((x[r], swy, sw, fwy, fw))
        i = j
    return pts


def pav_double(pts, decreasing, strict_gap=0.0):
    """blocks of (sum, weight, count); the merge test is the one of section 2.1"""
    blocks = []
    for _, swy, sw, _, _ in pts:
        blocks.append([swy, sw, 1])
        while len(blocks) > 1:
            below, top = blocks[-2][0] / blocks[-2][1], blocks[-1][0] / blocks[-1][1]
            violation = (below < top - strict_gap) if decreasing else (below > top + strict_gap)
            if not violation:
                break
            b = blocks.pop()
            blocks[-1][0] += b[0]
            blocks[-1][1] += b[1]
            blocks[-1][2] += b[2]
    out = []
    for s, wt, c in blocks:
        out += [s / wt] * c
    return out, [b[2] for b in blocks]


def pav_exact(pts, decreasing):
    blocks = []
    for _, _, _, fwy, fw in pts:
        blocks.append([fwy, fw, 1])
        while len(blocks) > 1:
            below, top = blocks[-2][0] / blocks[-2][1], blocks[-1][0] / blocks[-1][1]
            if not ((below < top) if decreasing else (below > top)):
                break
            b = blocks.pop()
            blocks[-1][0] += b[0]
            blocks[-1][1] += b[1]
            blocks[-1][2] += b[2]
    out = []
    for s, wt, c in blocks:
        out += [s / wt] * c
    return out, [b[2] for b in blocks]


def brute_best(vals, weights, decreasing):
    """the least weighted sum of squares over every monotone step sequence on consecutive blocks (the means of the blocks of a partition, kept if monotone)"""
    n = len(vals)
    best = None
    for cuts in itertools.product([0, 1], repeat=n - 1):
        blocks, start = [], 0
        for i, c in enumerate(cuts + (1,)):
            if c:
                blocks.append((start, i + 1))
                start = i + 1
        means = []
        for lo, hi in blocks:
            sw = sum(weights[lo:hi])
            means.append(sum(weights[k] * vals[k] for k in range(lo, hi)) / sw)
        mono = all((means[k] >= means[k + 1]) if decreasing else (means[k] <= means[k + 1]) for k in range(len(means) - 1))
        if not mono:
            continue
        sse = sum(weights[k] * (vals[k] - m) ** 2 for (lo, hi), m in zip(blocks, means) for k in range(lo, hi))
        best = sse if best is None else min(best, sse)
    return best


def make(rng):
    n = rng.pick([1, 2, 3, 5, 8, 8, 20, 60, 300])
    kind = rng.pick(["distinct", "ties", "trend", "flat", "steps"])
    if kind == "ties":
        x = [float(rng.next() % max(n // 3, 1)) for _ in range(n)]
    else:
        x = [rng.u() for _ in range(n)]
    xs_sorted = sorted(x)
    if kind == "trend":
        y = [xs_sorted[i] + (rng.u() - 0.5) * 0.4 for i in range(n)]
        x = xs_sorted
    elif kind == "flat":
        y = [0.5] * n
    elif kind == "steps":
        y = [float(i // max(n // 4, 1)) + (rng.u() - 0.5) * 2 for i in range(n)]
        x = xs_sorted
    else:
        y = [rng.u() for _ in range(n)]
    w = [rng.pick([1.0, 1.0, 0.001, 5.0, 1e-6, 1000.0]) * (0.5 + rng.u()) for _ in range(n)]
    return x, y, w, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-controls", action="store_true")
    a = ap.parse_args()
    rng = SM64(a.seed)
    bad = []
    differs = {"no-pooling": 0, "no-weights": 0, "gap": 0}
    could = {"no-pooling": 0, "no-weights": 0, "gap": 0}
    done = sk = brute = 0
    try:
        import numpy as np
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        IsotonicRegression = None
    for k in range(a.cases):
        x, y, w, kind = make(rng)
        decr = rng.next() % 3 == 0
        code, out = drive("isotonic", fvec(x), fvec(y), fvec(w), int(decr), 100000)
        d = parse(out)
        if d["refused"]:
            bad.append((k, "refused", d["refused"]))
            continue
        pts = pooled(x, y, w)
        ref, rb = pav_double(pts, decr)
        exa, eb = pav_exact(pts, decr)
        tag = "case %d (%s n=%d %s)" % (k, kind, len(x), "decreasing" if decr else "increasing")
        if d["info"] != [len(pts), len(rb), len(x)] or len(d["points"]) != len(pts):
            bad.append((tag, "info", d["info"], len(pts), len(rb)))
            continue
        for i, (r, p) in enumerate(zip(d["points"], pts)):
            if unhx(r[0]) != p[0] or unhx(r[2]) != p[2]:
                bad.append((tag, "x or weight", i))
            if r[1] != hx(ref[i]):
                bad.append((tag, "fitted bits", i, unhx(r[1]), ref[i]))
            if abs(unhx(r[1]) - float(exa[i])) > 1e-12 * abs(float(exa[i])) + 1e-300:
                bad.append((tag, "fitted vs exact", i, unhx(r[1]), float(exa[i])))
        blocks_got = [int(r[3]) for r in d["points"]]
        want_blocks = [b for b, c in enumerate(rb) for _ in range(c)]
        if blocks_got != want_blocks:
            bad.append((tag, "blocks", blocks_got[:10], want_blocks[:10]))
        fit = [unhx(r[1]) for r in d["points"]]
        if not all((fit[i] >= fit[i + 1]) if decr else (fit[i] <= fit[i + 1]) for i in range(len(fit) - 1)):
            bad.append((tag, "not monotone"))
        if len(pts) <= 8:
            vals = [float(p[3] / p[4]) for p in pts]
            wts = [p[4] for p in pts]
            vals = [p[3] / p[4] for p in pts]
            best = brute_best(vals, wts, decr)
            got = sum(wts[i] * (vals[i] - Fraction(fit[i])) ** 2 for i in range(len(pts)))
            brute += 1
            floor = Fraction(1, 10 ** 20) * (1 + sum(wts[i] * vals[i] ** 2 for i in range(len(pts))))      # a perfect fit has an exact sum of squares of 0 and the doubles' rounding is not
            if got > best * (1 + Fraction(1, 10 ** 9)) + floor:
                bad.append((tag, "not optimal", float(got), float(best)))
        if IsotonicRegression is not None and len(set(x)) == len(x) and not decr:
            m = IsotonicRegression(increasing=True).fit(np.array(x), np.array(y), sample_weight=np.array(w))
            sx = sorted(x)
            ref_sk = m.predict(np.array(sx))
            if max(abs(ref_sk[i] - fit[i]) / (abs(fit[i]) + 1e-300) for i in range(len(fit))) > 1e-9:
                bad.append((tag, "scikit-learn differs"))
            sk += 1
        if not a.no_controls:
            if len(set(x)) < len(x):
                could["no-pooling"] += 1
                sep = [(xx, w_ * y_, w_, None, None) for xx, y_, w_ in sorted(zip(x, y, w), key=lambda t: t[0])]
                differs["no-pooling"] += [hx(v) for v in pav_double(sep, decr)[0]] != [hx(v) for v in ref] if len(sep) == len(ref) else True
            could["no-weights"] += 1
            unit = pooled(x, y, [1.0] * len(x))
            differs["no-weights"] += [hx(v) for v in pav_double(unit, decr)[0]] != [hx(v) for v in ref]
            could["gap"] += 1
            differs["gap"] += [hx(v) for v in pav_double(pts, decr, 5e-2)[0]] != [hx(v) for v in ref]
        done += 1
    if not a.no_controls:
        for name in differs:
            if could[name] and differs[name] * 10 < could[name]:
                bad.append(("control %s changed the answer on %d of %d cases" % (name, differs[name], could[name]),))
    # ---- refusals
    ok = fvec([0.1, 0.2, 0.3])
    refusals = [
        ("ml.metric.undefined", fvec([]), fvec([]), fvec([]), 0, 100000),
        ("ml.metric.length-mismatch", ok, fvec([1.0, 2.0]), ok, 0, 100000),
        ("ml.metric.length-mismatch", ok, ok, fvec([1.0, 1.0]), 0, 100000),
        ("ml.input.non-finite", "H:7ff8000000000000,3fc999999999999a,3fd3333333333333", ok, ok, 0, 100000),
        ("ml.input.non-finite", ok, ok, "H:7ff0000000000000,3fc999999999999a,3fd3333333333333", 0, 100000),
        ("ml.param.out-of-range", ok, ok, fvec([1.0, 0.0, 1.0]), 0, 100000),
        ("ml.param.out-of-range", ok, ok, fvec([1.0, -1.0, 1.0]), 0, 100000),
        ("ml.input.too-many-rows", ok, ok, fvec([1.0, 1.0, 1.0]), 0, 2),
    ]
    for tag, *argv in refusals:
        code, out = drive("isotonic", *argv)
        if parse(out)["refused"] != tag:
            bad.append(("refusal %s" % tag, out.strip()[:80]))
    print("isotonic: %d cases (%d brute-forced for optimality, %d against scikit-learn %s); controls changed the answer on %s of %s" % (
        done, brute, sk, "" if IsotonicRegression else "(not installed)", differs, could))
    if bad:
        for b in bad[:20]:
            print("FAIL", *b)
        print("%d differences" % len(bad))
        return 1
    print("gate_isotonic: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

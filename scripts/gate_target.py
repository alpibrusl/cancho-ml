#!/usr/bin/env python3
"""Gates T1 to T3 of docs/calibration.md for the out-of-fold target encoding (lib/metrics/targetenc.cho), through the compiled driver:

    METRICS_CHECK=build/metrics-check python3 scripts/gate_target.py [--cases 300] [--seed 1] [--no-controls]

An independent implementation written from section 3.1 of the document in exact rational arithmetic: the counts of each category over the fit rows, p0 = positives / max(fit rows, 1), te = (b + k p0) / (n + k) and logn = log1p(n) for scoring, and
for every row the same from the fit rows OUT OF ITS FOLD (fold mod folds, never negative), p0 and 0 for a row outside the vocabulary. The driver's counts and integers must be equal and its floats within a relative 1e-12 of the exact values
(log1p is the library's, fdlibm written in cancho, which differs from the platform's by a few units in the last place). T2, the path of a label, as a PROPERTY: flipping the label of a fit row changes the encoding of a row of ITS OWN FOLD only through p0, by exactly
k (p0' - p0) / (n + k), and changes the encoding of some row of the same category in another fold (whose counts contain it) when there is one. CONTROLS (they must change the answer on at least a tenth of the cases that can show it): the full-data encoding in place of the out-of-fold one, p0 over every row
rather than the fit rows, and a fold of fold mod (folds + 1). Every refusal is reached with its tag.

Exit 0: every number is the expected one. Exit 1: a difference. Exit 2: the harness failed."""
import argparse
import math
import os
import pathlib
import sys
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gate_calibrate import SM64, hx, unhx, fvec, ivec, drive  # noqa: E402


def parse(out):
    d = {"info": None, "rows": [], "cats": [], "refused": None}
    for line in out.strip().split("\n"):
        w = line.split()
        if not w:
            continue
        if w[0] == "info":
            d["info"] = w[1:]
        elif w[0] == "row":
            d["rows"].append(w[2:])
        elif w[0] == "cat":
            d["cats"].append(w[2:])
        elif w[0] == "refused":
            d["refused"] = w[1]
    return d


def oracle(cat, label, fit, fold, ncat, folds, k, naive=False, p0_all=False, fold_mod=None):
    n = len(cat)
    fm = folds if fold_mod is None else fold_mod
    fits = [i for i in range(n) if fit[i] == 1]
    tot_n = len(fits)
    tot_b = sum(label[i] for i in fits)
    if p0_all:
        p0 = Fraction(sum(label), max(n, 1))
    else:
        p0 = Fraction(tot_b, max(tot_n, 1))
    kf = Fraction(k)
    cn = [0] * ncat
    cb = [0] * ncat
    fn_ = [[0] * ncat for _ in range(fm)]
    fb = [[0] * ncat for _ in range(fm)]
    for i in fits:
        if cat[i] >= 0:
            f = fold[i] % fm
            cn[cat[i]] += 1
            cb[cat[i]] += label[i]
            fn_[f][cat[i]] += 1
            fb[f][cat[i]] += label[i]
    rows = []
    for i in range(n):
        if cat[i] < 0:
            rows.append((p0, 0))
            continue
        f = fold[i] % fm
        nn = cn[cat[i]] - (0 if naive else fn_[f][cat[i]])
        bb = cb[cat[i]] - (0 if naive else fb[f][cat[i]])
        rows.append(((bb + kf * p0) / (nn + kf), nn))
    cats = [(cn[c], cb[c], (cb[c] + kf * p0) / (cn[c] + kf)) for c in range(ncat)]
    return p0, tot_n, tot_b, rows, cats


def close(got, want_fraction, log=False):
    w = math.log1p(float(want_fraction)) if log else float(want_fraction)
    return abs(got - w) <= 1e-12 * abs(w) + 1e-300


def make(rng, big=False):
    n = rng.pick([1, 2, 5, 12, 40, 100, 400, 1000])
    ncat = rng.pick([1, 2, 3, 7, 30])
    folds = rng.pick([2, 3, 5, 5, 7])
    k = rng.pick([0.5, 1.0, 20.0, 50.0, 1000.0])
    outside = rng.next() % 4 == 0
    cat = [(rng.next() % (ncat + (1 if outside else 0))) - (1 if outside else 0) for _ in range(n)]
    rate = rng.pick([0.05, 0.3, 0.7])
    label = [1 if rng.u() < rate else 0 for _ in range(n)]
    fit = [1 if rng.u() < rng.pick([1.0, 0.8, 0.5, 0.0]) else 0 for _ in range(n)]
    base = rng.pick([0, 100, -7])
    fold = [base + int(rng.next() % 1000) * rng.pick([1, 1, -1]) for _ in range(n)]
    return cat, label, fit, fold, ncat, folds, k


def run(cat, label, fit, fold, ncat, folds, k, cap=1000000):
    return parse(drive("target", ivec(cat), ivec(label), ivec(fit), ivec(fold), ncat, folds, fvec([k]), cap)[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-controls", action="store_true")
    a = ap.parse_args()
    rng = SM64(a.seed)
    bad = []
    names = ("naive", "p0-all", "fold-modulus")
    differs = dict.fromkeys(names, 0)
    could = dict.fromkeys(names, 0)
    leak_checked = leak_moved = 0
    done = 0
    for kk in range(a.cases):
        cat, label, fit, fold, ncat, folds, k = make(rng)
        d = run(cat, label, fit, fold, ncat, folds, k)
        tag = "case %d (n=%d ncat=%d folds=%d k=%g)" % (kk, len(cat), ncat, folds, k)
        if d["refused"]:
            bad.append((tag, "refused", d["refused"]))
            continue
        p0, tot_n, tot_b, rows, cats = oracle(cat, label, fit, fold, ncat, folds, k)
        if not close(unhx(d["info"][0]), p0) or int(d["info"][1]) != tot_n or int(d["info"][2]) != tot_b:
            bad.append((tag, "info", d["info"], float(p0), tot_n, tot_b))
        for i, (r, (te, nn)) in enumerate(zip(d["rows"], rows)):
            if not close(unhx(r[0]), te):
                bad.append((tag, "te", i, unhx(r[0]), float(te)))
            if not close(unhx(r[1]), nn, log=True):
                bad.append((tag, "logn", i, unhx(r[1]), math.log1p(nn)))
        for c, (r, (cn, cb, cte)) in enumerate(zip(d["cats"], cats)):
            if int(r[0]) != cn or int(r[1]) != cb or not close(unhx(r[2]), cte) or not close(unhx(r[3]), cn, log=True):
                bad.append((tag, "category", c, r, cn, cb, float(cte)))
        # T2: flip the label of one fit row with a category. The counts of every row of ITS OWN FOLD are taken without that row, so the only path from the label to those encodings is p0 (the base rate of all fit rows,
        # which the flip moves by 1 / (fit rows)): te' - te = k (p0' - p0) / (n + k) for each such row, exactly, and no other row of the fold moves otherwise. A row of the same category in another fold moves by more (its counts
        # contain the flipped row).
        cands = [i for i in range(len(cat)) if fit[i] == 1 and cat[i] >= 0]
        if cands:
            i = cands[rng.next() % len(cands)]
            label2 = list(label)
            label2[i] = 1 - label2[i]
            d2 = run(cat, label2, fit, fold, ncat, folds, k)
            p0b, _, _, rows2, _ = oracle(cat, label2, fit, fold, ncat, folds, k)
            fi = fold[i] % folds
            moved_other = False
            for j in range(len(cat)):
                got = unhx(d2["rows"][j][0]) - unhx(d["rows"][j][0])
                if fold[j] % folds == fi:
                    want = float(Fraction(k) * (p0b - p0) / (rows[j][1] + Fraction(k))) if cat[j] >= 0 else float(p0b - p0)
                    if abs(got - want) > 8e-16 * (1.0 + abs(unhx(d["rows"][j][0]))):
                        bad.append((tag, "LEAK: the only path from the label to a row of its fold is p0", i, j, got, want))
                elif cat[j] == cat[i] and abs(got) > 0:
                    moved_other = True
            leak_checked += 1
            leak_moved += moved_other
        if not a.no_controls:
            for name, kw in (("naive", {"naive": True}), ("p0-all", {"p0_all": True}), ("fold-modulus", {"fold_mod": folds + 1})):
                alt = oracle(cat, label, fit, fold, ncat, folds, k, **kw)
                could[name] += 1
                differs[name] += [float(r[0]) for r in alt[3]] != [float(r[0]) for r in rows]
            done += 0
        done += 1
    if not a.no_controls:
        for name in names:
            if could[name] and differs[name] * 10 < could[name]:
                bad.append(("control %s changed the answer on %d of %d cases" % (name, differs[name], could[name]),))
        if leak_moved * 10 < leak_checked:
            bad.append(("the flipped label moved another fold's encoding on only %d of %d cases: the leakage property is not tested" % (leak_moved, leak_checked),))
    # the naive encoding LEAKS (its own row moves when the label flips): the property test is able to fail
    cat, label, fit, fold = [0, 0, 0, 0], [1, 0, 0, 1], [1, 1, 1, 1], [0, 1, 0, 1]
    base = oracle(cat, label, fit, fold, 1, 2, 1.0, naive=True)[3][0][0]
    flipped = oracle(cat, [0, 0, 0, 1], fit, fold, 1, 2, 1.0, naive=True)[3][0][0]
    if base == flipped:
        bad.append(("the naive encoding does not move when its own label flips",))
    # a category seen once: its row is encoded from nothing (out of fold), te = p0
    d = run([0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0], 2, 2, 2.0)
    p0 = Fraction(2, 3)
    if d["refused"] or not close(unhx(d["rows"][0][0]), p0) or unhx(d["rows"][0][1]) != 0.0:
        bad.append(("a category of one row", d["rows"][:1], float(p0)))
    # ---- refusals
    ok = ([0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1], [0, 1, 0, 1])
    refusals = [
        ("ml.metric.undefined", ([], [], [], []), 2, 2, 1.0, 100),
        ("ml.metric.length-mismatch", ([0, 1], [1], [1, 1], [0, 1]), 2, 2, 1.0, 100),
        ("ml.metric.length-mismatch", ([0, 1], [1, 0], [1], [0, 1]), 2, 2, 1.0, 100),
        ("ml.metric.length-mismatch", ([0, 1], [1, 0], [1, 1], [0]), 2, 2, 1.0, 100),
        ("ml.param.out-of-range", ([0, 2, 0, 1], ok[1], ok[2], ok[3]), 2, 2, 1.0, 100),
        ("ml.param.out-of-range", ([0, -2, 0, 1], ok[1], ok[2], ok[3]), 2, 2, 1.0, 100),
        ("ml.param.out-of-range", ok, 0, 2, 1.0, 100),
        ("ml.param.out-of-range", ([-1, -1, -1, -1], ok[1], ok[2], ok[3]), 0, 2, 1.0, 100),
        ("ml.param.out-of-range", ok, 2, 1, 1.0, 100),
        ("ml.param.out-of-range", ok, 2, 1001, 1.0, 100),
        ("ml.param.out-of-range", ok, 2, 2, 0.0, 100),
        ("ml.param.out-of-range", ok, 2, 2, -1.0, 100),
        ("ml.param.out-of-range", ok, 2, 2, float("inf"), 100),
        ("ml.label.not-binary", (ok[0], [1, 2, 0, 1], ok[2], ok[3]), 2, 2, 1.0, 100),
        ("ml.label.not-binary", (ok[0], ok[1], [1, 2, 1, 1], ok[3]), 2, 2, 1.0, 100),
        ("ml.input.too-many-rows", ok, 2, 2, 1.0, 3),
    ]
    for tag, (c_, l_, f_, o_), ncat, folds, k, cap in refusals:
        out = drive("target", ivec(c_), ivec(l_), ivec(f_), ivec(o_), ncat, folds, "H:" + hx(k), cap)[1]
        if parse(out)["refused"] != tag:
            bad.append(("refusal %s" % tag, out.strip()[:80]))
    # a label other than 0 or 1 on a row that is not a fit row is not read
    d = run([0, 1, 0, 1], [1, 7, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], 2, 2, 1.0)
    if d["refused"]:
        bad.append(("a label of a row that is not fit was read", d["refused"]))
    print("target: %d cases; leakage property checked on %d (another fold moved on %d); controls changed the answer on %s of %s" % (done, leak_checked, leak_moved, differs, could))
    if bad:
        for b in bad[:20]:
            print("FAIL", *b)
        print("%d differences" % len(bad))
        return 1
    print("gate_target: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

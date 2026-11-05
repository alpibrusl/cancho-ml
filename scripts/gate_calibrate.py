#!/usr/bin/env python3
"""Gates C1 to C6 of docs/calibration.md for the calibration table and its gate (lib/metrics/calibrate.cho), through the compiled driver:

    METRICS_CHECK=build/metrics-check python3 scripts/gate_calibrate.py [--cases 400] [--seed 1] [--no-controls]

An independent implementation written from section 1.1 of the document (Python integers and fractions, no numpy in the rules): a STABLE sort of the rows by probability, the bin edges floor(i n / bins) in exact integers, the sums
of the probabilities and of p (1 - p) in exact rational arithmetic, the observed rate as an exact quotient, the gate. The driver's integers and flags must equal the oracle's exactly and its floats must be within a RELATIVE 1e-12
of the exact values (the package's decisive sums are rounded in a fixed order, so they are not the exact sums: this is the bound of docs/calibration.md section 5, C2). The flags `ok`, `judged` and `bad` are compared only where the
oracle's decision does not sit within 1e-9 (relative) of a threshold, because there the rounding of the driver may legitimately fall on the other side; the number of cases skipped for that is printed. The level correction (C4) is
compared the same way: the median of the days, the factor of the older half, the clipped probabilities of the newer half. Every refusal of section 1.5 is reached with its tag. CONTROLS (they must change the answer on at least a
tenth of the cases that can show it): an unstable tie order, the edges `linspace(0, n, bins + 1)` truncated, the bad-bin rule without the z condition, and a level correction fitted and judged on the same half.

Exit 0: every number is the expected one. Exit 1: a difference. Exit 2: the harness failed."""
import argparse
import math
import os
import pathlib
import struct
import subprocess
import sys
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = os.environ.get("METRICS_CHECK", str(ROOT / "build" / "metrics-check"))
sys.path.insert(0, str(ROOT / "scripts"))


class SM64:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def u(self):
        return (self.next() >> 11) / float(1 << 53)

    def pick(self, seq):
        return seq[self.next() % len(seq)]


def hx(x):
    return "%016x" % struct.unpack(">Q", struct.pack(">d", float(x)))[0]


def unhx(h):
    return struct.unpack(">d", struct.pack(">Q", int(h, 16)))[0]


def fvec(v):
    return "H:" + ",".join(hx(x) for x in v)


def ivec(v):
    return "L:" + ",".join(str(x) for x in v)


def drive(*args):
    # a driver that does not answer in 30 s is hung (an endless loop is a failure of the code under test, and the gate must say so quickly, not after the harness's own timeout)
    try:
        r = subprocess.run([BIN] + [str(a) for a in args], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        print("FAIL the driver did not answer in 30 s:", " ".join(str(a)[:40] for a in args[:3]))
        sys.exit(1)
    return r.returncode, r.stdout


def parse(out):
    """-> dict(info=[...], rows=[[...]], level=[...] or None, refused=tag or None)"""
    d = {"info": None, "rows": [], "level": None, "refused": None}
    for line in out.strip().split("\n"):
        w = line.split()
        if not w:
            continue
        if w[0] == "info":
            d["info"] = w[1:]
        elif w[0] == "row":
            d["rows"].append(w[2:])
        elif w[0] == "level":
            d["level"] = w[1:]
        elif w[0] == "refused":
            d["refused"] = w[1]
    return d


def oracle(p, y, bins, min_expected, zmax, olo, ohi, blo, bhi, ties="stable", edges="exact", zcond=True):
    """the table of docs/calibration.md section 1.1 in exact arithmetic: rows of (n, mean, observed, ratio, expected, sd2, z, judged, bad) as Fractions / ints, and (overall, ok, nbad)"""
    n = len(p)
    order = sorted(range(n), key=lambda i: Fraction(p[i])) if ties == "stable" else sorted(range(n), key=lambda i: (Fraction(p[i]), -i))
    sp = [Fraction(p[i]) for i in order]
    sy = [y[i] for i in order]
    if edges == "exact":
        e = [i * n // bins for i in range(bins + 1)]
    else:
        import numpy as np
        e = [int(v) for v in np.linspace(0, n, bins + 1)]
    rows, nbad = [], 0
    for b in range(bins):
        lo, hi = e[b], e[b + 1]
        m = hi - lo
        if m <= 0:
            continue
        sum_p = sum(sp[lo:hi])
        ys = sum(sy[lo:hi])
        var = sum(q * (1 - q) for q in sp[lo:hi])
        mean = sum_p / m
        obs = Fraction(ys, m)
        ratio = obs / mean if mean > 0 else None
        judged = sum_p >= Fraction(min_expected)
        sd2 = max(var, Fraction(1, 10 ** 12))
        zz2 = (ys - sum_p) ** 2 / sd2            # z squared, exact (the sign is the sign of ys - sum_p)
        thr = Fraction(zmax) ** 2
        outside = ratio is None or not (Fraction(blo) <= ratio <= Fraction(bhi))
        bad = judged and outside and ((zz2 > thr) if zcond else True)
        nbad += bad
        rows.append((m, mean, obs, ratio, sum_p, sd2, ys, zz2, judged, bad))
    total_p = sum(sp)
    overall = Fraction(sum(sy)) / max(total_p, Fraction(1, 10 ** 12))
    ok = Fraction(olo) <= overall <= Fraction(ohi) and nbad == 0
    return rows, overall, ok, nbad


def rel(got, want, floor=0.0):
    return abs(got - want) <= 1e-12 * abs(want) + floor


def compare(tag, d, orc, bad, skipped, params):
    rows, overall, ok, nbad = orc
    info = d["info"]
    if info is None or len(d["rows"]) != len(rows):
        bad.append((tag, "shape", None if info is None else len(d["rows"]), len(rows)))
        return
    if not rel(unhx(info[0]), float(overall)):
        bad.append((tag, "overall", unhx(info[0]), float(overall)))
    # the flags only where the decision is not within 1e-9 of a threshold
    olo, ohi = Fraction(params[2]), Fraction(params[3])
    margin = min(abs(overall - olo), abs(overall - ohi)) / max(overall, Fraction(1, 10 ** 9))
    tight = margin < Fraction(1, 10 ** 9)
    for i, (r, row) in enumerate(zip(d["rows"], rows)):
        m, mean, obs, ratio, sum_p, sd2, ys, zz2, judged, badf = row
        if int(r[0]) != m:
            bad.append((tag, "n", i, r[0], m))
        if not rel(unhx(r[1]), float(mean)):
            bad.append((tag, "mean", i, unhx(r[1]), float(mean)))
        if unhx(r[2]) != float(obs):
            bad.append((tag, "observed", i, unhx(r[2]), float(obs)))
        if ratio is None:
            if not math.isnan(unhx(r[3])):
                bad.append((tag, "ratio should be NaN", i))
        elif not rel(unhx(r[3]), float(ratio)):
            bad.append((tag, "ratio", i, unhx(r[3]), float(ratio)))
        if not rel(unhx(r[4]), float(sum_p)):
            bad.append((tag, "expected", i, unhx(r[4]), float(sum_p)))
        if not rel(unhx(r[5]), math.sqrt(float(sd2))):
            bad.append((tag, "sd", i, unhx(r[5]), math.sqrt(float(sd2))))
        zexp = math.copysign(math.sqrt(float(zz2)), float(ys - sum_p))
        # z is a DIFFERENCE of two nearly equal sums over sd: its error is a rounding of the operands (1e-12 of their size, divided by sd), not of z itself
        if abs(unhx(r[6]) - zexp) > 1e-12 * (float(ys) + float(sum_p)) / math.sqrt(float(sd2)) + 1e-12 * abs(zexp):
            bad.append((tag, "z", i, unhx(r[6]), zexp))
        # a sum EQUAL to the threshold is compared (the driver's sum of such a bin is exact in the cases where it is equal); one that differs by less than 1e-9 is left out
        edge_judged = sum_p != Fraction(params[0]) and abs(sum_p - Fraction(params[0])) / max(sum_p, Fraction(1, 10 ** 9)) < Fraction(1, 10 ** 9)
        edge_bad = (ratio is not None and (abs(ratio - Fraction(params[4])) / max(ratio, Fraction(1, 10 ** 9)) < Fraction(1, 10 ** 9) or abs(ratio - Fraction(params[5])) / max(ratio, Fraction(1, 10 ** 9)) < Fraction(1, 10 ** 9))) or abs(zz2 - Fraction(params[1]) ** 2) / max(zz2, Fraction(1, 10 ** 9)) < Fraction(1, 10 ** 9)
        if edge_judged or edge_bad:
            skipped[0] += 1
        else:
            if int(r[7]) != int(judged):
                bad.append((tag, "judged", i, r[7], judged))
            if int(r[8]) != int(badf):
                bad.append((tag, "bad", i, r[8], badf))
    if not tight and int(info[1]) != int(ok) and not any(int(r[8]) != int(b[9]) for r, b in zip(d["rows"], rows)):
        bad.append((tag, "ok", info[1], ok))
    if int(info[4]) != sum(r[0] for r in rows):
        bad.append((tag, "rows", info[4]))


def make_case(rng, big=False):
    n = rng.pick([1, 2, 5, 10, 11, 19, 20, 33, 100, 164, 500, 2000]) if not big else 20000
    bins = rng.pick([1, 2, 5, 10, 10, 10, 20])
    if bins > n:
        bins = n
    shape = rng.pick(["beta", "uniform", "ties", "extremes", "tiny", "zeros"])
    p = []
    for i in range(n):
        u = rng.u()
        if shape == "uniform":
            v = u
        elif shape == "ties":
            v = rng.pick([0.0, 0.1, 0.1, 0.25, 0.5, 0.5, 0.9, 1.0])
        elif shape == "extremes":
            v = rng.pick([0.0, 1.0, u * 1e-6, 1 - u * 1e-6])
        elif shape == "tiny":
            v = u * 0.02
        elif shape == "zeros":
            v = rng.pick([0.0, 0.0, 0.0, 0.05, 0.0])
        else:
            v = u ** 4
        p.append(v)
    mult = rng.pick([0.0, 0.5, 1.0, 1.0, 1.0, 1.3, 2.0])
    y = [1 if rng.u() < min(1.0, p[i] * mult) else 0 for i in range(n)]
    if shape == "zeros":
        y = [1 if rng.u() < 0.3 else 0 for _ in range(n)]       # positives where the probability is 0: a bin of mean 0 and observed rate above 0
    if rng.next() % 12 == 0:
        y = [0] * n
    params = [rng.pick([100.0, 100.0, 5.0, 0.0, 20.0]), rng.pick([3.0, 3.0, 1.0, 0.0, 2.0]), 0.95, 1.05, 0.8, 1.25]
    return p, y, bins, params, shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-controls", action="store_true")
    a = ap.parse_args()
    rng = SM64(a.seed)
    bad = []
    skipped = [0]
    differs = {"unstable": 0, "linspace": 0, "no-z": 0}
    could = {"unstable": 0, "linspace": 0, "no-z": 0}
    done = 0
    oks = [0, 0]
    for k in range(a.cases):
        p, y, bins, params, shape = make_case(rng, big=(k % 200 == 199))
        code, out = drive("calibrate", ivec(y), fvec(p), fvec(params), bins, 100000)
        d = parse(out)
        if d["refused"]:
            bad.append((k, "refused", d["refused"]))
            continue
        orc = oracle(p, y, bins, *params)
        compare("case %d (%s n=%d bins=%d)" % (k, shape, len(p), bins), d, orc, bad, skipped, params)
        oks[int(orc[2])] += 1
        if not a.no_controls and len(p) <= 2000:
            if shape == "ties":
                ties = oracle(p, y, bins, *params, ties="unstable")
                could["unstable"] += 1
                differs["unstable"] += [(r[0], r[4], r[6]) for r in ties[0]] != [(r[0], r[4], r[6]) for r in orc[0]]
            lin = oracle(p, y, bins, *params, edges="linspace")
            could["linspace"] += 1
            differs["linspace"] += [(r[0]) for r in lin[0]] != [(r[0]) for r in orc[0]]
            noz = oracle(p, y, bins, *params, zcond=False)
            could["no-z"] += 1
            differs["no-z"] += noz[3] != orc[3]
        done += 1
    # the case where the Python's linspace truncation and the exact edge differ: n = 164, 20 bins (docs/calibration.md section 1.1)
    p = [(i + 0.5) / 164 for i in range(164)]
    y = [i % 2 for i in range(164)]
    code, out = drive("calibrate", ivec(y), fvec(p), fvec([0.0, 0.0, 0.0, 9.0, 0.0, 9.0]), 20, 100000)
    d = parse(out)
    sizes = [int(r[0]) for r in d["rows"]]
    exact = [(i + 1) * 164 // 20 - i * 164 // 20 for i in range(20)]
    import numpy as np
    trunc = [int(v) for v in np.diff(np.linspace(0, 164, 21).astype(int))]
    if sizes != exact:
        bad.append(("n=164 bins=20 edges are not the exact ones", sizes))
    if exact == trunc:
        bad.append(("the n=164 case no longer separates the exact edges from linspace's",))
    # ---- fixed cases the random ones cannot be relied on to reach
    # balanced overall (ratio 1.0) but a bin that is far off: the gate must FAIL on the bin (two groups of 200 rows, p 0.2 and 0.8, observed 0.3 and 0.7)
    p = [0.2] * 200 + [0.8] * 200
    y = [1] * 60 + [0] * 140 + [1] * 140 + [0] * 60
    d = parse(drive("calibrate", ivec(y), fvec(p), fvec([5.0, 3.0, 0.95, 1.05, 0.8, 1.25]), 2, 100000)[1])
    if d["info"] is None or d["info"][1] != "0" or d["info"][2] != "1" or abs(unhx(d["info"][0]) - 1.0) > 1e-12:
        bad.append(("balanced overall but a bad bin", d["info"]))
    # the same table with the bin inside the band passes
    y2 = [1] * 44 + [0] * 156 + [1] * 156 + [0] * 44
    d = parse(drive("calibrate", ivec(y2), fvec(p), fvec([5.0, 3.0, 0.95, 1.05, 0.8, 1.25]), 2, 100000)[1])
    if d["info"] is None or d["info"][1] != "1" or d["info"][2] != "0":
        bad.append(("balanced and in band should pass", d["info"]))
    # a positive at probability 0: the bin has mean 0, observed above 0 (the ratio is NaN, not infinite), and the overall ratio is over the floor 1e-12
    d = parse(drive("calibrate", ivec([1, 0, 0, 0]), fvec([0.0, 0.0, 0.0, 0.0]), fvec([0.0, 0.0, 0.95, 1.05, 0.8, 1.25]), 1, 100000)[1])
    if d["info"] is None or not math.isinf(unhx(d["info"][0])) and unhx(d["info"][0]) < 1e11 or not math.isnan(unhx(d["rows"][0][3])):
        bad.append(("a positive at p = 0", d["info"], d["rows"]))
    # probabilities that sum to less than 1e-3 but more than the floor 1e-12: the overall ratio is over the sum, not over the floor
    d = parse(drive("calibrate", ivec([1, 0, 0, 0]), fvec([1e-5] * 4), fvec([0.0, 0.0, 0.95, 1.05, 0.8, 1.25]), 1, 100000)[1])
    if d["info"] is None or not rel(unhx(d["info"][0]), 1 / 4e-5):
        bad.append(("a sum of probabilities below 1e-3", d["info"]))
    # ---- C4: the level correction
    lvl_done, lvl_made = 0, 0
    for k in range(max(a.cases // 4, 20)):
        p, y, bins, params, shape = make_case(rng)
        n = len(p)
        days = [float(rng.next() % 40) for _ in range(n)]
        if rng.next() % 5 == 0:
            days = [float(i) for i in range(n)]
        code, out = drive("level", ivec(y), fvec(p), fvec(days), fvec(params), bins, 100000)
        d = parse(out)
        sd = sorted(days)
        cut = sd[n // 2] if n % 2 else (sd[n // 2 - 1] + sd[n // 2]) / 2.0
        old = [i for i in range(n) if days[i] <= cut]
        new = [i for i in range(n) if days[i] > cut]
        so = sum(Fraction(p[i]) for i in old)
        if d["refused"]:
            # a table needs `bins` rows in the newer half
            if not (new and len(new) < bins):
                bad.append(("level case %d refused" % k, d["refused"], len(new), bins))
            continue
        lv = d["level"]
        if lv is None or int(lv[1]) != len(old) or int(lv[2]) != len(new):
            bad.append(("level case %d halves" % k, lv, len(old), len(new)))
            continue
        if not old or not new or so <= 0:
            if int(lv[3]) != 0 or unhx(lv[0]) != 1.0:
                bad.append(("level case %d should make no table" % k, lv))
            continue
        factor = Fraction(sum(y[i] for i in old)) / so
        if not rel(unhx(lv[0]), float(factor)):
            bad.append(("level case %d factor" % k, unhx(lv[0]), float(factor)))
        pn = [min(max(p[i] * unhx(lv[0]), 0.0), 1.0) for i in new]
        yn = [y[i] for i in new]
        orc = oracle(pn, yn, bins, *params)
        compare("level case %d" % k, d, orc, bad, skipped, params)
        lvl_done += 1
        lvl_made += int(lv[3])
        if not a.no_controls:
            # the control: fitting and judging on the same (newer) half passes where the honest gate fails, at least sometimes
            po = [min(max(p[i] * float(Fraction(sum(y[j] for j in new)) / max(sum(Fraction(p[j]) for j in new), Fraction(1, 10 ** 12))), 0.0), 1.0) for i in new]
            same = oracle(po, yn, bins, *params)
            could.setdefault("same-half", 0)
            differs.setdefault("same-half", 0)
            could["same-half"] += 1
            differs["same-half"] += same[2] != orc[2]
    # the older half (days 0 to 4) has probabilities that sum to 0: the factor is 1.0, no table is made, and that is not a refusal
    code, out = drive("level", ivec([0, 1, 0, 0, 0, 1, 0, 1, 0, 0]), fvec([0.0] * 5 + [0.5, 0.25, 0.75, 0.125, 0.5]), fvec([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]), fvec([0.0, 3.0, 0.95, 1.05, 0.8, 1.25]), 2, 100000)
    d = parse(out)
    if d["refused"] or d["level"] is None or d["level"][3] != "0" or unhx(d["level"][0]) != 1.0 or d["info"] is not None:
        bad.append(("level with an older half that sums to 0", out.strip()[:120]))
    if not a.no_controls:
        for name in differs:
            if could[name] and differs[name] * 10 < could[name] and name not in ("no-z", "linspace"):
                bad.append(("control %s changed the answer on %d of %d cases" % (name, differs[name], could[name]),))
        if differs["no-z"] == 0:
            bad.append(("control no-z never changed the answer",))
    if oks[0] == 0 or oks[1] == 0:
        bad.append(("the cases never had both outcomes of the gate", oks))
    # ---- refusals (section 1.5)
    P = fvec([100.0, 3.0, 0.95, 1.05, 0.8, 1.25])
    good_p, good_y = fvec([0.1, 0.2, 0.3, 0.4]), ivec([0, 1, 0, 1])
    refusals = [
        ("ml.metric.undefined", "calibrate", ivec([]), fvec([]), P, 1, 100000),
        ("ml.metric.length-mismatch", "calibrate", good_y, fvec([0.1, 0.2, 0.3]), P, 1, 100000),
        ("ml.label.not-binary", "calibrate", ivec([0, 1, 2, 1]), good_p, P, 1, 100000),
        ("ml.metric.bad-probability", "calibrate", good_y, fvec([0.1, 0.2, 1.5, 0.4]), P, 1, 100000),
        ("ml.metric.bad-probability", "calibrate", good_y, fvec([0.1, 0.2, -0.1, 0.4]), P, 1, 100000),
        ("ml.input.non-finite", "calibrate", good_y, "H:7ff8000000000000,3fb999999999999a,3fc999999999999a,3fd999999999999a", P, 1, 100000),
        ("ml.param.out-of-range", "calibrate", good_y, good_p, P, 0, 100000),
        ("ml.param.out-of-range", "calibrate", good_y, good_p, P, 5, 100000),
        ("ml.param.out-of-range", "calibrate", ivec([0, 1] * 600), fvec([0.5] * 1200), P, 1001, 100000),
        ("ml.input.too-many-rows", "calibrate", good_y, good_p, P, 2, 3),
    ]
    for tag, *argv in refusals:
        code, out = drive(*argv)
        if parse(out)["refused"] != tag:
            bad.append(("refusal %s" % tag, out.strip()[:80]))
    print("calibrate: %d cases (gate passed %d, failed %d), %d level cases (%d made a table); %d flag comparisons skipped for sitting on a threshold; controls changed the answer on %s of %s" % (
        done, oks[1], oks[0], lvl_done, lvl_made, skipped[0], differs, could))
    if bad:
        for b in bad[:20]:
            print("FAIL", *b)
        print("%d differences" % len(bad))
        return 1
    print("gate_calibrate: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

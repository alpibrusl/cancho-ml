#!/usr/bin/env python3
"""The oracle of the recommendation metrics (docs/recommend.md section 3, gate RC1): hit rate@k, recall@k, NDCG@k and MRR per user and averaged, from the definitions, in Python (no cancho
code). `log` is the port of std/math.cho that oracle_models.py holds; every sum is a sequential left fold from +0.0.

    python3 scripts/oracle_recommend.py --metrics [N]   N seeded cases (default 300) against build/metrics-check ("recommend" mode), bits compared
    python3 scripts/oracle_recommend.py --controls      the tie rule, the history, the cut at k and the undefined mean, checked on hand-made users
    python3 scripts/oracle_recommend.py --driver        the same controls and the refusals through the compiled driver
Exit 1 on any mismatch. METRICS_CHECK names the driver."""

import math
import os
import pathlib
import random
import struct
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from oracle_models import log_c  # noqa: E402

ROOT = HERE.parent
BIN = os.environ.get("METRICS_CHECK", str(ROOT / "build" / "metrics-check"))
LN2 = float(log_c(np.array([2.0]))[0])


def log(x):
    return float(log_c(np.array([x], dtype=np.float64))[0])


def disc(r):
    """1 / log2(r + 1), as 1.0 / (log(float(r + 1)) / log(2.0)): three roundings, no fused multiply-add"""
    return 1.0 / (log(float(r + 1)) / LN2)


def bits(x):
    return "%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]


def user_values(scores, hist, test, k):
    """(hit, recall, ndcg, mrr) of one user with a non-empty test set; scores: one per item"""
    hs = set(hist)
    ts = set(test)
    order = sorted((i for i in range(len(scores)) if i not in hs), key=lambda i: -scores[i])     # stable: ties by item number; -0.0 == 0.0
    m = len(order)
    kk = min(k, m)
    dcg = 0.0
    found = 0
    first = 0
    for r in range(1, m + 1):
        if order[r - 1] in ts:
            if first == 0:
                first = r
            if r <= kk:
                dcg = dcg + disc(r)
                found += 1
    idcg = 0.0
    for r in range(1, min(k, len(ts)) + 1):
        idcg = idcg + disc(r)
    return (1.0 if found else 0.0, float(found) / float(len(ts)), dcg / idcg, (1.0 / float(first)) if first else 0.0)


def evaluate(scores, hists, tests, k):
    """scores: one list of floats per user; -> dict(users, scored, hit, recall, ndcg, mrr), the means None when no user has a test item"""
    sums = [0.0] * 4
    scored = 0
    for u in range(len(scores)):
        if not tests[u]:
            continue
        v = user_values(scores[u], hists[u], tests[u], k)
        scored += 1
        for j in range(4):
            sums[j] = sums[j] + v[j]
    out = {"users": len(scores), "scored": scored}
    for name, j in (("hit", 0), ("recall", 1), ("ndcg", 2), ("mrr", 3)):
        out[name] = sums[j] / float(scored) if scored else None
    return out


# ---------------------------------------------------------------- the cases
def gen_case(seed):
    rng = random.Random(seed)
    nu = rng.choice([1, 2, 3, 7, 20, 60])
    ni = rng.choice([1, 2, 3, 5, 12, 40, 100])
    style = rng.choice(["random", "ties", "flat", "dense"])
    scores, hists, tests = [], [], []
    for _ in range(nu):
        if style == "ties":
            sc = [float(rng.randint(0, 2)) for _ in range(ni)]
        elif style == "flat":
            sc = [0.0 if rng.random() < 0.5 else -0.0 for _ in range(ni)]
        else:
            sc = [rng.uniform(-3, 3) * 10 ** rng.randint(-2, 2) for _ in range(ni)]
        items = list(range(ni))
        rng.shuffle(items)
        nh = rng.randint(0, ni // 2) if style != "dense" else rng.randint(0, ni)
        hist = sorted(items[:nh])
        rest = items[nh:]
        nt = 0 if (not rest or rng.random() < 0.2) else rng.randint(1, min(len(rest), rng.choice([1, 1, 2, 5, 30])))
        test = sorted(rest[:nt])
        scores.append(sc)
        hists.append(hist)
        tests.append(test)
    k = rng.choice([1, 1, 2, 3, 5, 10, 10, 100])
    k = min(k, ni)
    return scores, hists, tests, k


def spec_ints(v):
    return "L:" + ",".join(str(x) for x in v)


def spec_floats(v):
    return "H:" + ",".join(bits(x) for x in v)


def offsets(lists):
    off = [0]
    flat = []
    for l in lists:
        flat += l
        off.append(len(flat))
    return off, flat


def run_driver(scores, hists, tests, k, cap=100000000, ni=None):
    nu = len(scores)
    ni = len(scores[0]) if ni is None else ni
    ho, hf = offsets(hists)
    to, tf = offsets(tests)
    flat = [x for row in scores for x in row]
    r = subprocess.run([BIN, "recommend", str(nu), str(ni), spec_floats(flat), spec_ints(ho), spec_ints(hf), spec_ints(to), spec_ints(tf), str(k), str(cap)],
                       capture_output=True, text=True, timeout=120)
    out = {}
    for ln in r.stdout.splitlines():
        parts = ln.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return r.returncode, out, r.stdout


def check(seed):
    scores, hists, tests, k = gen_case(seed)
    want = evaluate(scores, hists, tests, k)
    rc, got, text = run_driver(scores, hists, tests, k)
    if rc != 0:
        return "seed %d: driver exit %d: %s" % (seed, rc, text[:200])
    if int(got.get("users", -1)) != want["users"] or int(got.get("scored", -1)) != want["scored"]:
        return "seed %d: users %s scored %s, expected %d %d" % (seed, got.get("users"), got.get("scored"), want["users"], want["scored"])
    for name in ("hit", "recall", "ndcg", "mrr"):
        w = want[name]
        exp = "undefined" if w is None else bits(w)
        if got.get(name) != exp:
            return "seed %d (users=%d, items=%d, k=%d): %s %s, expected %s" % (seed, len(scores), len(scores[0]), k, name, got.get(name), exp)
    return None


# ---------------------------------------------------------------- the controls
def controls():
    """hand-made users whose answer is known without the oracle's arithmetic: if these fail, the definitions above are not the ones in the document"""
    bad = []

    def near(a, b):
        return abs(a - b) < 1e-12

    # the test item is ranked first: everything is 1
    v = user_values([3.0, 2.0, 1.0], [], [0], 1)
    if v != (1.0, 1.0, 1.0, 1.0):
        bad.append("first: %r" % (v,))
    # the test item is ranked second, k = 1: no hit, nothing found, MRR over the whole ranking is 1/2
    v = user_values([3.0, 2.0, 1.0], [], [1], 1)
    if v[0] != 0.0 or v[1] != 0.0 or v[2] != 0.0 or v[3] != 0.5:
        bad.append("second at k=1: %r" % (v,))
    # k = 2: found, NDCG = disc(2) / disc(1) = 1 / log2(3)
    v = user_values([3.0, 2.0, 1.0], [], [1], 2)
    if v[0] != 1.0 or not near(v[2], 1.0 / math.log2(3)):
        bad.append("second at k=2: %r" % (v,))
    # the history is excluded: the best item is in the history, so the test item is first
    v = user_values([9.0, 2.0, 1.0], [0], [1], 1)
    if v != (1.0, 1.0, 1.0, 1.0):
        bad.append("history excluded: %r" % (v,))
    # ties: equal scores rank the smaller item first
    v1 = user_values([1.0, 1.0], [], [0], 1)
    v2 = user_values([1.0, 1.0], [], [1], 1)
    if not (v1[0] == 1.0 and v2[0] == 0.0 and v2[3] == 0.5):
        bad.append("tie rule: %r %r" % (v1, v2))
    # -0.0 and +0.0 are one score
    v = user_values([0.0, -0.0], [], [1], 1)
    if v[0] != 0.0:
        bad.append("-0.0 is not a tie with 0.0: %r" % (v,))
    # two test items at ranks 1 and 3 of 4, k = 4: recall 1, DCG = d1 + d3, IDCG = d1 + d2
    v = user_values([4.0, 3.0, 2.0, 1.0], [], [0, 2], 4)
    d = lambda r: 1.0 / math.log2(r + 1)  # noqa: E731
    if not near(v[2], (d(1) + d(3)) / (d(1) + d(2))) or v[1] != 1.0:
        bad.append("two test items: %r" % (v,))
    # recall: one of two found at k = 1
    v = user_values([4.0, 3.0, 2.0, 1.0], [], [0, 3], 1)
    if v[1] != 0.5:
        bad.append("recall: %r" % (v,))
    # a user without a test item is counted and not scored; no user scored: every mean undefined
    e = evaluate([[1.0, 2.0], [2.0, 1.0]], [[], []], [[], [0]], 1)
    if e["users"] != 2 or e["scored"] != 1 or e["hit"] != 1.0:
        bad.append("user without a test item: %r" % (e,))
    e = evaluate([[1.0, 2.0]], [[]], [[]], 1)
    if e["scored"] != 0 or e["hit"] is not None or e["ndcg"] is not None or e["mrr"] is not None or e["recall"] is not None:
        bad.append("no user scored: %r" % (e,))
    return bad


def driver_controls():
    """the same through the compiled driver, and the refusals of the library (gate RC10)"""
    bad = []
    rc, a, _ = run_driver([[1.0, 1.0]], [[]], [[0]], 1)
    rc, b, _ = run_driver([[1.0, 1.0]], [[]], [[1]], 1)
    if a.get("hit") != bits(1.0) or b.get("hit") != bits(0.0) or b.get("mrr") != bits(0.5):
        bad.append("tie rule: %r %r" % (a, b))
    rc, c, _ = run_driver([[0.0, -0.0]], [[]], [[1]], 1)
    if c.get("hit") != bits(0.0):
        bad.append("-0.0 is not a tie with 0.0: %r" % (c,))
    rc, d, _ = run_driver([[9.0, 2.0, 1.0]], [[0]], [[1]], 1)
    if d.get("hit") != bits(1.0) or d.get("mrr") != bits(1.0):
        bad.append("history excluded: %r" % (d,))
    rc, e, _ = run_driver([[1.0, 2.0]], [[]], [[]], 1)
    if rc != 0 or e.get("hit") != "undefined" or e.get("mrr") != "undefined" or e.get("scored") != "0":
        bad.append("no user scored: %r (exit %d)" % (e, rc))
    f = bits(1.0)

    def refuse(name, args, tag, code):
        r = subprocess.run([BIN, "recommend"] + args, capture_output=True, text=True, timeout=60)
        if ("refused " + tag) not in r.stdout or r.returncode != code:
            bad.append("%s: expected %s exit %d, got %r exit %d" % (name, tag, code, r.stdout.strip()[:80], r.returncode))

    two = "H:" + f + "," + f
    refuse("length", ["1", "2", "H:" + f, "L:0,0", "L:", "L:0,1", "L:0", "1", "100"], "ml.metric.length-mismatch", 8)
    refuse("no users", ["0", "2", "H:", "L:0", "L:", "L:0", "L:", "1", "100"], "ml.input.empty", 8)
    refuse("no items", ["1", "0", "H:", "L:0,0", "L:", "L:0,0", "L:", "1", "100"], "ml.input.empty", 8)
    refuse("non-finite", ["1", "2", "H:7ff8000000000000," + f, "L:0,0", "L:", "L:0,1", "L:0", "1", "100"], "ml.input.non-finite", 8)
    refuse("k zero", ["1", "2", two, "L:0,0", "L:", "L:0,1", "L:0", "0", "100"], "ml.param.out-of-range", 2)
    refuse("k above the items", ["1", "2", two, "L:0,0", "L:", "L:0,1", "L:0", "3", "100"], "ml.param.out-of-range", 2)
    refuse("more scores than the cap", ["1", "2", two, "L:0,0", "L:", "L:0,1", "L:0", "1", "1"], "ml.input.too-many-rows", 8)
    refuse("an item out of range", ["1", "2", two, "L:0,0", "L:", "L:0,1", "L:2", "1", "100"], "ml.param.out-of-range", 2)
    refuse("a test list not ascending", ["1", "3", "H:" + ",".join([f] * 3), "L:0,0", "L:", "L:0,2", "L:2,1", "1", "100"], "ml.param.out-of-range", 2)
    refuse("a test item in the history", ["1", "2", two, "L:0,1", "L:0", "L:0,1", "L:0", "1", "100"], "ml.param.out-of-range", 2)
    refuse("a decreasing offset", ["2", "2", "H:" + ",".join([f] * 4), "L:0,2,1", "L:0", "L:0,0,0", "L:", "1", "100"], "ml.param.out-of-range", 2)
    refuse("a decreasing offset inside the list", ["3", "2", "H:" + ",".join([f] * 6), "L:0,2,1,2", "L:0,1", "L:0,0,0,0", "L:", "1", "100"], "ml.param.out-of-range", 2)
    refuse("an offset past the list", ["2", "2", "H:" + ",".join([f] * 4), "L:0,5,1", "L:0", "L:0,0,0", "L:", "1", "100"], "ml.param.out-of-range", 2)
    refuse("offsets that do not end at the list", ["1", "2", two, "L:0,1", "L:", "L:0,1", "L:0", "1", "100"], "ml.metric.length-mismatch", 8)
    return bad


def main():
    argv = sys.argv[1:]
    if not any(a in argv for a in ("--controls", "--driver", "--metrics")):
        print(__doc__)
        return 2
    rc = 0
    if "--controls" in argv:
        bad = controls()
        for b in bad:
            print("FAIL control:", b)
        print("oracle_recommend controls: %s" % ("all as defined" if not bad else "%d failed" % len(bad)))
        rc |= 1 if bad else 0
    if "--driver" in argv:
        bad = driver_controls()
        for b in bad:
            print("FAIL driver:", b)
        print("oracle_recommend driver controls: %s" % ("all as defined" if not bad else "%d failed" % len(bad)))
        rc |= 1 if bad else 0
    if "--metrics" in argv:
        at = argv.index("--metrics") + 1
        n = int(argv[at]) if len(argv) > at and argv[at].isdigit() else 300
        bad = 0
        for seed in range(1, n + 1):
            msg = check(seed)
            if msg:
                bad += 1
                print("MISMATCH", msg)
                if bad > 10:
                    break
        print("oracle_recommend: %d cases, %s" % (n, "all agree bit for bit" if not bad else "%d mismatches" % bad))
        rc |= 1 if bad else 0
    return rc


if __name__ == "__main__":
    sys.exit(main())

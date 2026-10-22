#!/usr/bin/env python3
"""The oracle of the ranking metrics (docs/ranking.md section 3, gate RK1): NDCG@k, MAP@k, MRR@k, precision@k and recall@k per group and averaged, from the
definitions, in Python (no cancho code). `log` is the port of std/math.cho that oracle_models.py holds; every sum is a sequential left fold from +0.0.

    python3 scripts/oracle_rank.py --metrics [N]    N seeded cases (default 300) against build/metrics-check ("rank" mode), bits compared
    python3 scripts/oracle_rank.py --controls       the tie rule, the undefined groups and the discount, checked on hand-made groups
Exit 1 on any mismatch. METRICS_CHECK names the driver."""

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
R_MAX = 30
G_MAX = 4096
LN2 = float(log_c(np.array([2.0]))[0])


def log(x):
    return float(log_c(np.array([x], dtype=np.float64))[0])


def disc(r):
    """1 / log2(r + 1), as 1.0 / (log(float(r + 1)) / log(2.0)): three roundings, no fused multiply-add"""
    return 1.0 / (log(float(r + 1)) / LN2)


def gain_of(rel, gain):
    return float((1 << rel) - 1) if gain == "exp" else float(rel)


def ranking(scores):
    """the rows of a group (indices in file order) ordered by score descending, ties by file order"""
    return sorted(range(len(scores)), key=lambda i: -scores[i])        # sorted is stable; -0.0 == 0.0


def group_values(rel, sc, k, gain):
    """the values of one group: (ndcg, ap, rr, prec, rec) with None for a value the group does not define"""
    g = len(rel)
    kk = g if k == 0 else min(k, g)
    order = ranking(sc)
    dcg = 0.0
    for r in range(1, kk + 1):
        dcg = dcg + gain_of(rel[order[r - 1]], gain) * disc(r)
    ideal = sorted(rel, reverse=True)
    idcg = 0.0
    for r in range(1, kk + 1):
        idcg = idcg + gain_of(ideal[r - 1], gain) * disc(r)
    nrel = sum(1 for v in rel if v > 0)
    hits = 0
    ap_sum = 0.0
    first = 0
    for r in range(1, kk + 1):
        if rel[order[r - 1]] > 0:
            hits += 1
            ap_sum = ap_sum + float(hits) / float(r)
            if first == 0:
                first = r
    pden = g if k == 0 else k
    prec = float(hits) / float(pden)
    if nrel == 0:
        return (None, None, None, prec, None)
    ndcg = dcg / idcg
    ap = ap_sum / float(min(kk, nrel))
    rr = 1.0 / float(first) if first else 0.0
    rec = float(hits) / float(nrel)
    return (ndcg, ap, rr, prec, rec)


def evaluate(gid, rel, score, k, gain):
    """-> dict(groups, scored, ndcg, map, mrr, prec, recall); the means are sequential sums in group order divided once; None when no group is scored"""
    groups = sorted(set(gid))
    sums = [0.0] * 5
    scored = 0
    for gname in groups:
        idx = [i for i in range(len(gid)) if gid[i] == gname]
        v = group_values([rel[i] for i in idx], [score[i] for i in idx], k, gain)
        sums[3] = sums[3] + v[3]
        if v[0] is not None:
            scored += 1
            for j in (0, 1, 2, 4):
                sums[j] = sums[j] + v[j]
    out = {"groups": len(groups), "scored": scored, "prec": sums[3] / float(len(groups))}
    for name, j in (("ndcg", 0), ("map", 1), ("mrr", 2), ("recall", 4)):
        out[name] = sums[j] / float(scored) if scored else None
    return out


def bits(x):
    return "%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]


# ---------------------------------------------------------------- the cases
def gen_case(seed):
    rng = random.Random(seed)
    ng = rng.choice([1, 2, 3, 5, 12, 40])
    gid, rel, score = [], [], []
    style = rng.choice(["graded", "binary", "ties", "flat", "mixed"])
    for g in range(ng):
        size = rng.choice([1, 1, 2, 3, 5, 9, 20, 60, 200])
        top = {"graded": rng.choice([1, 4, 30]), "binary": 1, "ties": 3, "flat": 0, "mixed": rng.choice([1, 2, 30])}[style]
        if style == "mixed" and rng.random() < 0.3:
            top = 0
        for _ in range(size):
            gid.append(g)
            rel.append(rng.randint(0, top))
            if style == "ties":
                score.append(float(rng.randint(0, 2)))
            elif rng.random() < 0.05:
                score.append(rng.choice([0.0, -0.0]))
            else:
                score.append(rng.uniform(-3, 3) * 10 ** rng.randint(-2, 2))
    order = list(range(len(gid)))
    if rng.random() < 0.6:
        rng.shuffle(order)                  # the rows of a group need not be next to each other
    gid, rel, score = [gid[i] for i in order], [rel[i] for i in order], [score[i] for i in order]
    k = rng.choice([0, 1, 2, 3, 5, 10, 10, 100, 4096])
    gain = rng.choice(["exp", "exp", "linear"])
    return gid, rel, score, k, gain


def spec_ints(v):
    return "L:" + ",".join(str(x) for x in v)


def spec_floats(v):
    return "H:" + ",".join(bits(x) for x in v)


def run_driver(gid, rel, score, k, gain):
    r = subprocess.run([BIN, "rank", spec_ints(gid), spec_ints(rel), spec_floats(score), str(k), gain, "1000000"], capture_output=True, text=True, timeout=120)
    out = {}
    for ln in r.stdout.splitlines():
        parts = ln.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return r.returncode, out, r.stdout


def check(seed):
    gid, rel, score, k, gain = gen_case(seed)
    want = evaluate(gid, rel, score, k, gain)
    rc, got, text = run_driver(gid, rel, score, k, gain)
    if rc != 0:
        return "seed %d: driver exit %d: %s" % (seed, rc, text[:200])
    if int(got.get("groups", -1)) != want["groups"] or int(got.get("scored", -1)) != want["scored"]:
        return "seed %d: groups %s scored %s, expected %d %d" % (seed, got.get("groups"), got.get("scored"), want["groups"], want["scored"])
    for name in ("ndcg", "map", "mrr", "prec", "recall"):
        w = want[name]
        exp = "undefined" if w is None else bits(w)
        if got.get(name) != exp:
            return "seed %d (n=%d, k=%d, %s): %s %s, expected %s" % (seed, len(gid), k, gain, name, got.get(name), exp)
    return None


# ---------------------------------------------------------------- the controls
def controls():
    """hand-made groups whose answer is known without the oracle's arithmetic: if these fail, the definitions above are not the ones in the document"""
    bad = []

    def near(a, b):
        return abs(a - b) < 1e-12

    # one group, relevance 3 2 3 0 1 2 in the order of the scores (the example of the DCG literature), exponential gain, k = 6
    rel = [3, 2, 3, 0, 1, 2]
    sc = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    v = group_values(rel, sc, 6, "exp")
    import math
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(sorted(rel, reverse=True)))
    if not near(v[0], dcg / idcg):
        bad.append("ndcg of the textbook group: %r vs %r" % (v[0], dcg / idcg))
    # the discount is 1 at rank 1 and 1/log2(3) at rank 2
    if not near(disc(1), 1.0) or not near(disc(2), 1 / math.log2(3)):
        bad.append("discount")
    # ties: relevant row first or last in the file, equal scores: the tie rule decides, so the two NDCG differ
    a = group_values([1, 0], [0.0, 0.0], 1, "exp")[0]
    b = group_values([0, 1], [0.0, 0.0], 1, "exp")[0]
    if not (a == 1.0 and b == 0.0):
        bad.append("tie rule: %r %r" % (a, b))
    # a group with no relevant row is left out of the mean and counted
    e = evaluate([0, 0, 1, 1], [1, 0, 0, 0], [2.0, 1.0, 2.0, 1.0], 10, "exp")
    if e["groups"] != 2 or e["scored"] != 1 or e["ndcg"] != 1.0:
        bad.append("empty group: %r" % e)
    e = evaluate([0, 0], [0, 0], [1.0, 2.0], 10, "exp")
    if e["scored"] != 0 or e["ndcg"] is not None or e["map"] is not None or e["mrr"] is not None or e["recall"] is not None or e["prec"] != 0.0:
        bad.append("all groups empty: %r" % e)
    # precision@k divides by k even when the group is smaller; MRR is 0 when the first relevant row is past k
    v = group_values([1, 1], [2.0, 1.0], 4, "exp")
    if v[3] != 0.5:
        bad.append("precision of a group smaller than k: %r" % (v[3],))
    v = group_values([0, 0, 1], [3.0, 2.0, 1.0], 2, "exp")
    if v[2] != 0.0:
        bad.append("mrr past k: %r" % (v[2],))
    # AP: relevant at ranks 1 and 3 of 4, k = 4: (1/1 + 2/3) / 2
    v = group_values([1, 0, 1, 0], [4.0, 3.0, 2.0, 1.0], 4, "exp")
    if not near(v[1], (1.0 + 2.0 / 3.0) / 2.0):
        bad.append("ap: %r" % (v[1],))
    return bad


def driver_controls():
    """the same controls through the compiled driver, and the refusals of the library (gate RK2 and the refusals of RK7)"""
    bad = []

    def run(gid, rel, score, k, gain="exp", cap=1000000):
        return run_driver(gid, rel, score, k, gain)

    # ties: the relevant row first in the file is ranked first, last is ranked last
    rc, a, _ = run([0, 0], [1, 0], [0.0, 0.0], 1)
    rc, b, _ = run([0, 0], [0, 1], [0.0, 0.0], 1)
    if a.get("ndcg") != bits(1.0) or b.get("ndcg") != bits(0.0):
        bad.append("tie rule: %r %r" % (a.get("ndcg"), b.get("ndcg")))
    # -0.0 and +0.0 are one score
    rc, c, _ = run([0, 0], [0, 1], [0.0, -0.0], 1)
    if c.get("ndcg") != bits(0.0):
        bad.append("-0.0 is not a tie with 0.0: %r" % (c.get("ndcg"),))
    # a group with no relevant row among others: scored = groups - 1
    rc, d, _ = run([0, 0, 1, 1], [1, 0, 0, 0], [2.0, 1.0, 2.0, 1.0], 10)
    if d.get("groups") != "2" or d.get("scored") != "1" or d.get("ndcg") != bits(1.0):
        bad.append("an empty group among others: %r" % d)
    # every group empty: undefined, not a refusal
    rc, e, _ = run([0, 0], [0, 0], [1.0, 2.0], 10)
    if rc != 0 or e.get("ndcg") != "undefined" or e.get("map") != "undefined" or e.get("mrr") != "undefined" or e.get("recall") != "undefined" or e.get("prec") != bits(0.0):
        bad.append("all groups empty: %r (exit %d)" % (e, rc))
    # the refusals, each with its tag and exit code
    cases = [
        ("length", (["L:0,0", "L:1", "H:" + bits(1.0) + "," + bits(2.0), "5", "exp", "100"]), "ml.metric.length-mismatch", 8),
        ("empty", (["L:", "L:", "H:", "5", "exp", "100"]), "ml.input.empty", 8),
        ("non-finite", (["L:0,0", "L:1,0", "H:7ff8000000000000," + bits(1.0), "5", "exp", "100"]), "ml.input.non-finite", 8),
        ("relevance 31", (["L:0,0", "L:31,0", "H:" + bits(1.0) + "," + bits(2.0), "5", "exp", "100"]), "ml.param.out-of-range", 2),
        ("relevance -1", (["L:0,0", "L:-1,0", "H:" + bits(1.0) + "," + bits(2.0), "5", "exp", "100"]), "ml.param.out-of-range", 2),
        ("k above the group cap", (["L:0,0", "L:1,0", "H:" + bits(1.0) + "," + bits(2.0), "4097", "exp", "100"]), "ml.param.out-of-range", 2),
        ("negative k", (["L:0,0", "L:1,0", "H:" + bits(1.0) + "," + bits(2.0), "-1", "exp", "100"]), "ml.param.out-of-range", 2),
        ("more rows than the cap", (["L:0,0,0", "L:1,0,1", "H:" + ",".join([bits(1.0)] * 3), "5", "exp", "2"]), "ml.input.too-many-rows", 8),
        ("a group number with no row", (["L:0,2,2", "L:1,0,1", "H:" + ",".join([bits(1.0)] * 3), "5", "exp", "100"]), "ml.param.out-of-range", 2),
        ("more groups than rows", (["L:0,2", "L:1,0", "H:" + bits(1.0) + "," + bits(2.0), "5", "exp", "100"]), "ml.param.out-of-range", 2),
    ]
    for name, args, tag, code in cases:
        r = subprocess.run([BIN, "rank"] + args, capture_output=True, text=True, timeout=60)
        if ("refused " + tag) not in r.stdout or r.returncode != code:
            bad.append("%s: expected %s exit %d, got %r exit %d" % (name, tag, code, r.stdout.strip()[:80], r.returncode))
    # a group of 4097 rows is refused before anything else is done
    n = 4097
    r = subprocess.run([BIN, "rank", spec_ints([0] * n), spec_ints([1] * n), spec_floats([1.0] * n), "5", "exp", "100000"], capture_output=True, text=True, timeout=60)
    if "refused ml.param.out-of-range" not in r.stdout:
        bad.append("a group of %d rows: %r" % (n, r.stdout.strip()[:80]))
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
        print("oracle_rank controls: %s" % ("all as defined" if not bad else "%d failed" % len(bad)))
        rc |= 1 if bad else 0
    if "--driver" in argv:
        bad = driver_controls()
        for b in bad:
            print("FAIL driver:", b)
        print("oracle_rank driver controls: %s" % ("all as defined" if not bad else "%d failed" % len(bad)))
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
        print("oracle_rank: %d cases, %s" % (n, "all agree bit for bit" if not bad else "%d mismatches" % bad))
        rc |= 1 if bad else 0
    return rc


if __name__ == "__main__":
    sys.exit(main())

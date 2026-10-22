#!/usr/bin/env python3
"""The oracle of the recommendation metrics (docs/recommend.md section 3, gate RC1): hit rate@k, recall@k, NDCG@k and MRR per user and averaged, from the definitions, in Python (no cancho
code). `log` is the port of std/math.cho that oracle_models.py holds; every sum is a sequential left fold from +0.0.

    python3 scripts/oracle_recommend.py --metrics [N]   N seeded cases (default 300) against build/metrics-check ("recommend" mode), bits compared
    python3 scripts/oracle_recommend.py --controls      the tie rule, the history, the cut at k and the undefined mean, checked on hand-made users
    python3 scripts/oracle_recommend.py --driver        the same controls and the refusals through the compiled driver
    python3 scripts/oracle_recommend.py --log [N]       N seeded logs (default 300) against build/recommend-check ("log" and "split" modes): the two compressed forms, the sums of duplicates,
                                                        the refusals with their reasons and rows, the rows held out per user; compared exactly (floats as bits)
    python3 scripts/oracle_recommend.py --log-controls  hand-made logs and the refusals through the compiled driver
Exit 1 on any mismatch. METRICS_CHECK names the metrics driver, RECOMMEND_CHECK the log driver."""

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
LOG_BIN = os.environ.get("RECOMMEND_CHECK", str(ROOT / "build" / "recommend-check"))
MAX_USERS = 1048576
MAX_ITEMS = 1048576
MAX_NNZ = 33554432
MAX_VALUE = 1000000.0
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


# ---------------------------------------------------------------- the log (docs/recommend.md section 2) and the split per user (section 7)
TAGS = {"empty": "ml.input.empty", "rows": "ml.input.too-many-rows", "range": "ml.param.out-of-range", "finite": "ml.input.non-finite"}
WHY = {"bad_rating": 1, "duplicate": 2, "users": 3, "items": 4, "interactions": 5, "shape": 6}


def log_oracle(uid, iid, vals, nusers, nitems, implicit, cap):
    """-> ("ok", dict) or ("refused", tag, why, row, row2). The checks in the order the library makes them"""
    n = len(uid)
    if len(iid) != n or len(vals) != n:
        return ("refused", TAGS["range"], WHY["shape"], 0, 0)
    if n == 0:
        return ("refused", TAGS["empty"], WHY["shape"], 0, 0)
    if n > MAX_NNZ or n > cap:
        return ("refused", TAGS["rows"], WHY["interactions"], 0, 0)
    if nusers > MAX_USERS:
        return ("refused", TAGS["range"], WHY["users"], 0, 0)
    if nitems > MAX_ITEMS:
        return ("refused", TAGS["range"], WHY["items"], 0, 0)
    if nusers < 1 or nitems < 1 or nusers > n or nitems > n:
        return ("refused", TAGS["range"], WHY["shape"], 0, 0)
    for r in range(n):
        if not (0 <= uid[r] < nusers and 0 <= iid[r] < nitems):
            return ("refused", TAGS["range"], WHY["shape"], r, 0)
        x = vals[r]
        if x != x or x in (float("inf"), float("-inf")):
            return ("refused", TAGS["finite"], WHY["bad_rating"], r, 0)
        if x > MAX_VALUE or x < -MAX_VALUE or (implicit and not x > 0.0):
            return ("refused", TAGS["range"], WHY["bad_rating"], r, 0)
    if set(uid) != set(range(nusers)) or set(iid) != set(range(nitems)):
        return ("refused", TAGS["range"], WHY["shape"], 0, 0)
    order = sorted(range(n), key=lambda r: (uid[r], iid[r], r))
    pairs = {}
    first_dup = None
    for k, r in enumerate(order):
        key = (uid[r], iid[r])
        if key in pairs:
            if first_dup is None:
                first_dup = (order[k - 1], r)
            pairs[key] = pairs[key] + vals[r]
        else:
            pairs[key] = vals[r]
    if first_dup is not None and not implicit:
        return ("refused", TAGS["range"], WHY["duplicate"], first_dup[0], first_dup[1])
    keys = sorted(pairs)
    uoff = [0] * (nusers + 1)
    for (u, i) in keys:
        uoff[u + 1] += 1
    for u in range(nusers):
        uoff[u + 1] += uoff[u]
    uitem = [i for (u, i) in keys]
    uval = [pairs[k] for k in keys]
    by_item = sorted(keys, key=lambda k: (k[1], k[0]))
    ioff = [0] * (nitems + 1)
    for (u, i) in by_item:
        ioff[i + 1] += 1
    for i in range(nitems):
        ioff[i + 1] += ioff[i]
    return ("ok", {"nnz": len(keys), "uoff": uoff, "uitem": uitem, "uval": uval, "ioff": ioff, "iuser": [u for (u, i) in by_item], "ival": [pairs[k] for k in by_item]})


def split_oracle(uid, tkey, nusers, last):
    """the 0/1 mask of the rows held out: each user's `last` latest by (time key, row), only for a user with more than `last` rows"""
    mask = [0] * len(uid)
    for u in range(nusers):
        rows = sorted((r for r in range(len(uid)) if uid[r] == u), key=lambda r: (tkey[r], r))
        if len(rows) > last:
            for r in rows[len(rows) - last:]:
                mask[r] = 1
    return mask


def run_log_driver(args):
    r = subprocess.run([LOG_BIN] + [str(a) for a in args], capture_output=True, text=True, timeout=120)
    out = {}
    for ln in r.stdout.splitlines():
        parts = ln.split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return r.returncode, out, r.stdout


def parse_ints(t):
    return [int(x) for x in t.split(",")] if t.strip() else []


def parse_bits(t):
    return t.split(",") if t.strip() else []


def gen_log(seed):
    rng = random.Random(seed)
    nu = rng.choice([1, 2, 3, 6, 20, 40])
    ni = rng.choice([1, 2, 3, 5, 15, 30])
    implicit = rng.random() < 0.5
    style = rng.choice(["sparse", "dense", "dups", "hostile"])
    rows = []
    for u in range(nu):
        for i in range(ni):
            if style == "dense" or rng.random() < 0.3:
                rows.append((u, i))
    for u in range(nu):
        rows.append((u, rng.randrange(ni)))        # every user has a row
    for i in range(ni):
        rows.append((rng.randrange(nu), i))        # every item has a row
    if style == "dups":
        for _ in range(rng.randint(1, 5)):
            rows.append(rng.choice(rows))
    if style != "dups" and not implicit:
        seen, uniq = set(), []
        for r in rows:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        rows = uniq
    rng.shuffle(rows)
    uid = [r[0] for r in rows]
    iid = [r[1] for r in rows]
    vals = []
    for _ in rows:
        if implicit:
            vals.append(float(rng.randint(1, 5)) if rng.random() < 0.8 else rng.uniform(0.001, 50.0))
        else:
            vals.append(float(rng.randint(1, 5)) if rng.random() < 0.7 else rng.uniform(-5, 5))
    if style == "hostile" and rows:
        k = rng.randrange(len(rows))
        which = rng.choice(["nan", "inf", "big", "zero", "neg", "uid", "iid"])
        if which == "nan":
            vals[k] = float("nan")
        elif which == "inf":
            vals[k] = float("inf")
        elif which == "big":
            vals[k] = 2.0e6
        elif which == "zero":
            vals[k] = 0.0
        elif which == "neg":
            vals[k] = -3.0
        elif which == "uid":
            uid[k] = nu + rng.randint(0, 2)
        else:
            iid[k] = -1
    return uid, iid, vals, nu, ni, implicit


def check_log(seed):
    uid, iid, vals, nu, ni, implicit = gen_log(seed)
    want = log_oracle(uid, iid, vals, nu, ni, implicit, 10 ** 9)
    rc, got, text = run_log_driver(["log", nu, ni, 1 if implicit else 0, spec_ints(uid), spec_ints(iid), spec_floats(vals), 10 ** 9])
    if want[0] == "refused":
        line = text.strip()
        exp = "refused %s why %d row %d row2 %d" % want[1:]
        if line != exp:
            return "seed %d: expected %r, got %r" % (seed, exp, line[:160])
        return None
    if rc != 0:
        return "seed %d: driver exit %d: %s" % (seed, rc, text[:160])
    w = want[1]
    if int(got.get("nnz", -1)) != w["nnz"]:
        return "seed %d: nnz %s vs %d" % (seed, got.get("nnz"), w["nnz"])
    for name in ("uoff", "uitem", "ioff", "iuser"):
        if parse_ints(got.get(name, "")) != w[name]:
            return "seed %d: %s differs" % (seed, name)
    for name in ("uval", "ival"):
        if parse_bits(got.get(name, "")) != [bits(x) for x in w[name]]:
            return "seed %d: %s differs" % (seed, name)
    return None


def check_split(seed):
    rng = random.Random(seed * 31 + 7)
    nu = rng.choice([1, 2, 5, 12])
    n = rng.randint(nu, 60)
    uid = list(range(nu)) + [rng.randrange(nu) for _ in range(n - nu)]
    rng.shuffle(uid)
    style = rng.choice(["distinct", "ties", "allsame", "negative"])
    if style == "distinct":
        tkey = rng.sample(range(10 * n), n)
    elif style == "ties":
        tkey = [rng.randint(0, 3) for _ in range(n)]
    elif style == "allsame":
        tkey = [5] * n
    else:
        tkey = [rng.randint(-50, 50) for _ in range(n)]
    last = rng.choice([1, 1, 2, 3, 10])
    want = split_oracle(uid, tkey, nu, last)
    rc, got, text = run_log_driver(["split", spec_ints(uid), spec_ints(tkey), nu, last])
    if rc != 0:
        return "split seed %d: driver exit %d: %s" % (seed, rc, text[:160])
    if parse_ints(got.get("mask", "")) != want:
        return "split seed %d (style %s, last %d): mask differs" % (seed, style, last)
    return None


def log_controls():
    """hand-made logs whose answer is known without the oracle's arithmetic, and the refusals through the compiled driver"""
    bad = []
    f = bits(1.0)

    def run(*args):
        return run_log_driver(["log"] + list(args))

    # two users, two items, a full 2 x 2 log: users by item and items by user, ascending
    rc, g, t = run(2, 2, 0, "L:1,0,1,0", "L:1,1,0,0", "H:" + ",".join(bits(x) for x in (4.0, 3.0, 2.0, 1.0)), 100)
    if (rc != 0 or parse_ints(g.get("uoff", "")) != [0, 2, 4] or parse_ints(g.get("uitem", "")) != [0, 1, 0, 1]
            or parse_bits(g.get("uval", "")) != [bits(1.0), bits(3.0), bits(2.0), bits(4.0)]
            or parse_ints(g.get("iuser", "")) != [0, 1, 0, 1] or parse_bits(g.get("ival", "")) != [bits(1.0), bits(2.0), bits(3.0), bits(4.0)]):
        bad.append("2 x 2 log: %r" % t[:200])
    # an implicit pair twice: summed in file order (0.1 + 0.2 + 0.3 is not 0.6 in doubles: the order is the answer)
    rc, g, t = run(1, 1, 1, "L:0,0,0", "L:0,0,0", "H:" + ",".join(bits(x) for x in (0.1, 0.2, 0.3)), 100)
    if rc != 0 or parse_bits(g.get("uval", "")) != [bits((0.1 + 0.2) + 0.3)]:
        bad.append("implicit sum: %r" % t[:200])
    # an explicit pair twice: refused, with the two rows (file order), the smallest (user, item) first
    rc, g, t = run(2, 1, 0, "L:1,0,1,0", "L:0,0,0,0", "H:" + ",".join([f] * 4), 100)
    if t.strip() != "refused ml.param.out-of-range why 2 row 1 row2 3" or rc != 2:
        bad.append("explicit duplicate: %r exit %d" % (t.strip()[:100], rc))

    def refuse(name, args, tag, why, code, row=0, row2=0):
        rc, g, t = run(*args)
        exp = "refused %s why %d row %d row2 %d" % (tag, why, row, row2)
        if t.strip() != exp or rc != code:
            bad.append("%s: expected %r exit %d, got %r exit %d" % (name, exp, code, t.strip()[:100], rc))

    two = "H:" + ",".join([f] * 2)
    refuse("length", [1, 1, 0, "L:0,0", "L:0", two, 100], "ml.param.out-of-range", 6, 2)
    refuse("empty", [1, 1, 0, "L:", "L:", "H:", 100], "ml.input.empty", 6, 8)
    refuse("rows over the cap", [1, 1, 1, "L:0,0", "L:0,0", two, 1], "ml.input.too-many-rows", 5, 8)
    refuse("users over the rows", [3, 1, 1, "L:0,0", "L:0,0", two, 100], "ml.param.out-of-range", 6, 2)
    refuse("a user number out of range", [1, 1, 1, "L:0,1", "L:0,0", two, 100], "ml.param.out-of-range", 6, 2, 1)
    refuse("a negative item", [1, 1, 1, "L:0,0", "L:0,-1", two, 100], "ml.param.out-of-range", 6, 2, 1)
    refuse("a NaN rating", [1, 2, 0, "L:0,0", "L:0,1", "H:7ff8000000000000," + f, 100], "ml.input.non-finite", 1, 8)
    refuse("an infinite rating", [1, 2, 0, "L:0,0", "L:0,1", "H:" + f + ",7ff0000000000000", 100], "ml.input.non-finite", 1, 8, 1)
    refuse("a rating above the bound", [1, 2, 0, "L:0,0", "L:0,1", "H:" + bits(2.0e6) + "," + f, 100], "ml.param.out-of-range", 1, 2)
    refuse("a rating below the bound", [1, 2, 0, "L:0,0", "L:0,1", "H:" + bits(-2.0e6) + "," + f, 100], "ml.param.out-of-range", 1, 2)
    refuse("an implicit zero", [1, 2, 1, "L:0,0", "L:0,1", "H:" + f + "," + bits(0.0), 100], "ml.param.out-of-range", 1, 2, 1)
    refuse("an implicit negative", [1, 2, 1, "L:0,0", "L:0,1", "H:" + bits(-1.0) + "," + f, 100], "ml.param.out-of-range", 1, 2)
    refuse("an item with no row", [1, 2, 0, "L:0,0", "L:0,0", "H:" + f + "," + f, 100], "ml.param.out-of-range", 6, 2)
    refuse("a user with no row", [2, 1, 0, "L:0,0", "L:0,0", "H:" + f + "," + f, 100], "ml.param.out-of-range", 6, 2)
    # the limits are checked before anything is allocated
    refuse("users over the limit", [MAX_USERS + 1, 1, 0, "L:0", "L:0", "H:" + f, 100], "ml.param.out-of-range", 3, 2)
    refuse("items over the limit", [1, MAX_ITEMS + 1, 0, "L:0", "L:0", "H:" + f, 100], "ml.param.out-of-range", 4, 2)

    def split(uid, tkey, nu, last):
        rc, g, t = run_log_driver(["split", spec_ints(uid), spec_ints(tkey), nu, last])
        return rc, g, t

    # one user, four rows with times 3 1 2 1: the latest is row 0; with last = 2, rows 0 and 2; ties: the later row is the later
    rc, g, t = split([0, 0, 0, 0], [3, 1, 2, 1], 1, 1)
    if parse_ints(g.get("mask", "")) != [1, 0, 0, 0]:
        bad.append("last 1: %r" % t)
    rc, g, t = split([0, 0, 0, 0], [3, 1, 2, 1], 1, 2)
    if parse_ints(g.get("mask", "")) != [1, 0, 1, 0]:
        bad.append("last 2: %r" % t)
    rc, g, t = split([0, 0, 0, 0], [1, 1, 1, 1], 1, 2)
    if parse_ints(g.get("mask", "")) != [0, 0, 1, 1]:
        bad.append("ties cut in file order: %r" % t)
    # a user with N rows or fewer stays in training
    rc, g, t = split([0, 0, 1], [1, 2, 3], 2, 2)
    if parse_ints(g.get("mask", "")) != [0, 0, 0]:
        bad.append("user with last rows: %r" % t)
    rc, g, t = split([0, 0, 1, 1, 1], [1, 2, 3, 4, 5], 2, 1)
    if parse_ints(g.get("mask", "")) != [0, 1, 0, 0, 1]:
        bad.append("two users: %r" % t)
    for name, args, code in (("a user out of range", ([0, 2], [1, 2], 2, 1), 2), ("last 0", ([0, 0], [1, 2], 1, 0), 2), ("length", ([0, 0], [1], 1, 1), 2), ("empty", ([], [], 1, 1), 2)):
        rc, g, t = split(*args)
        if rc != code or "refused ml.param.out-of-range" not in t:
            bad.append("split %s: %r exit %d" % (name, t.strip()[:80], rc))
    return bad


def main():
    argv = sys.argv[1:]
    if not any(a in argv for a in ("--controls", "--driver", "--metrics", "--log", "--log-controls")):
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
    if "--log-controls" in argv:
        bad = log_controls()
        for b in bad:
            print("FAIL log control:", b)
        print("oracle_recommend log controls: %s" % ("all as defined" if not bad else "%d failed" % len(bad)))
        rc |= 1 if bad else 0
    if "--log" in argv:
        at = argv.index("--log") + 1
        n = int(argv[at]) if len(argv) > at and argv[at].isdigit() else 300
        bad = 0
        for seed in range(1, n + 1):
            for msg in (check_log(seed), check_split(seed)):
                if msg:
                    bad += 1
                    print("MISMATCH", msg)
            if bad > 10:
                break
        print("oracle_recommend log: %d logs and %d splits, %s" % (n, n, "all agree" if not bad else "%d mismatches" % bad))
        rc |= 1 if bad else 0
    return rc


if __name__ == "__main__":
    sys.exit(main())

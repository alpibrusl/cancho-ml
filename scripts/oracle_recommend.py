#!/usr/bin/env python3
"""The oracle of the recommendation metrics (docs/recommend.md section 3, gate RC1): hit rate@k, recall@k, NDCG@k and MRR per user and averaged, from the definitions, in Python (no cancho
code). `log` is the port of std/math.cho that oracle_models.py holds; every sum is a sequential left fold from +0.0.

    python3 scripts/oracle_recommend.py --metrics [N]   N seeded cases (default 300) against build/metrics-check ("recommend" mode), bits compared
    python3 scripts/oracle_recommend.py --controls      the tie rule, the history, the cut at k and the undefined mean, checked on hand-made users
    python3 scripts/oracle_recommend.py --driver        the same controls and the refusals through the compiled driver
    python3 scripts/oracle_recommend.py --log [N]       N seeded logs (default 300) against build/recommend-check ("log" and "split" modes): the two compressed forms, the sums of duplicates,
                                                        the refusals with their reasons and rows, the rows held out per user; compared exactly (floats as bits)
    python3 scripts/oracle_recommend.py --log-controls  hand-made logs and the refusals through the compiled driver
    python3 scripts/oracle_recommend.py --als [N]       N seeded fits (default 120) against build/recommend-check ("als" mode): P, Q and the loss after every half-pass equal the oracle's bits; the loss
                                                        does not rise from one half-pass to the next; with one pass the factors equal the dense closed form (gates RC3 to RC5)
    python3 scripts/oracle_recommend.py --als-controls  a one-user one-item fit worked by hand, and the refusals of the parameters
    python3 scripts/oracle_recommend.py --payload [N]   N seeded model payloads (default 40): the bytes equal the oracle's writer; the reader gives them back byte for byte; every truncation, an
                                                        extra byte, thirty named corruptions and every byte flipped (stride 1 on the first five, 7 on the rest) are refused or read back canonical
Exit 1 on any mismatch. METRICS_CHECK names the metrics driver, RECOMMEND_CHECK the log driver."""

import math
import os
import pathlib
import random
import struct
import subprocess
import sys

import numpy as np
import oracle_linalg as ol
import struct as _st

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


# ---------------------------------------------------------------- ALS (docs/recommend.md section 4, gates RC3 to RC5)
MASK64 = (1 << 64) - 1


def mix64(z):
    z &= MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def derive64(h, c):
    return mix64(((h ^ c) & MASK64) + 0x9E3779B97F4A7C15)


def init_q(ni, f, seed):
    """draw c of item i: the c-th output of the stream derive(derive(seed, 12), i), unit(z) / sqrt(f)"""
    q = []
    for i in range(ni):
        st = derive64(derive64(seed & MASK64, 12), i)
        row = []
        for _ in range(f):
            st = (st + 0x9E3779B97F4A7C15) & MASK64
            z = mix64(st)
            row.append((float((z >> 11) & ((1 << 53) - 1)) / 4503599627370496.0 - 1.0) / math.sqrt(float(f)))
        q.append(row)
    return q


def gram_o(fac, f):
    g = [[0.0] * f for _ in range(f)]
    for y in fac:
        for a in range(f):
            ya = y[a]
            for b in range(a + 1):
                g[a][b] = g[a][b] + ya * y[b]
    return g


def half_pass_o(off, idx, vals, other, gm, n, f, lam, alpha, implicit):
    out = []
    for r in range(n):
        a_ = [[(gm[p][q] if implicit else 0.0) if q <= p else 0.0 for q in range(f)] for p in range(f)]
        bv = [0.0] * f
        for k in range(off[r], off[r + 1]):
            y = other[idx[k]]
            if implicit:
                coef = alpha * vals[k]
                w = 1.0 + coef
            else:
                coef = 1.0
                w = vals[k]
            for p in range(f):
                yp = y[p]
                for q in range(p + 1):
                    a_[p][q] = a_[p][q] + (coef * (yp * y[q]) if implicit else yp * y[q])
                bv[p] = bv[p] + w * yp
        for s_ in range(f):
            a_[s_][s_] = a_[s_][s_] + lam
        lmat = ol.chol_o(np.array(a_, dtype=np.float64))
        if lmat is None:
            return None
        z = ol.fwd_o(lmat, np.array(bv, dtype=np.float64))
        x = ol.bwd_o(lmat, z)
        out.append([float(v) for v in x])
    return out


def loss_o(w, pf, qf, gm, nu, ni, f, lam, alpha, implicit):
    data = 0.0
    for u in range(nu):
        if implicit:
            quad = 0.0
            for a in range(f):
                t = 0.0
                for b in range(f):
                    gab = gm[a][b] if b <= a else gm[b][a]
                    t = t + gab * pf[u][b]
                quad = quad + pf[u][a] * t
            data = data + quad
        for k in range(w["uoff"][u], w["uoff"][u + 1]):
            s = 0.0
            for c in range(f):
                s = s + pf[u][c] * qf[w["uitem"][k]][c]
            if implicit:
                cc = 1.0 + alpha * w["uval"][k]
                data = data + (cc * ((1.0 - s) * (1.0 - s)) - s * s)
            else:
                data = data + (w["uval"][k] - s) * (w["uval"][k] - s)
    sp = 0.0
    for row in pf:
        for v in row:
            sp = sp + v * v
    sq = 0.0
    for row in qf:
        for v in row:
            sq = sq + v * v
    return data + lam * (sp + sq)


def als_oracle(w, nu, ni, f, lam, alpha, implicit, iterations, seed, trace):
    """-> ("ok", P, Q, losses) or ("refused", tag); w is the output of log_oracle"""
    qf = init_q(ni, f, seed)
    pf = None
    losses = []
    for it in range(iterations):
        g = gram_o(qf, f) if implicit else None
        pf = half_pass_o(w["uoff"], w["uitem"], w["uval"], qf, g, nu, f, lam, alpha, implicit)
        if pf is None:
            return ("refused", "ml.linalg.not-positive-definite")
        if trace:
            g = gram_o(qf, f) if implicit else None
            losses.append(loss_o(w, pf, qf, g, nu, ni, f, lam, alpha, implicit))
        g = gram_o(pf, f) if implicit else None
        qf = half_pass_o(w["ioff"], w["iuser"], w["ival"], pf, g, ni, f, lam, alpha, implicit)
        if qf is None:
            return ("refused", "ml.linalg.not-positive-definite")
        if trace or it == iterations - 1:
            g = gram_o(qf, f) if implicit else None
            ls = loss_o(w, pf, qf, g, nu, ni, f, lam, alpha, implicit)
            losses.append(ls)
    return ("ok", pf, qf, losses)


def run_als_driver(uid, iid, vals, nu, ni, implicit, f, lam, alpha, iters, seed, trace):
    return run_log_driver(["als", nu, ni, 1 if implicit else 0, spec_ints(uid), spec_ints(iid), spec_floats(vals), 10 ** 9, f, spec_floats([lam]), spec_floats([alpha]), iters, seed, 1 if trace else 0])


def gen_als(seed):
    """an accepted log (explicit or implicit) and parameters; None when the log of this seed is refused"""
    uid, iid, vals, nu, ni, implicit = gen_log(seed)
    want = log_oracle(uid, iid, vals, nu, ni, implicit, 10 ** 9)
    if want[0] != "ok":
        return None
    rng = random.Random(seed * 101 + 5)
    f = rng.choice([1, 2, 3, 4, 6])
    lam = rng.choice([0.01, 0.1, 1.0])
    alpha = rng.choice([1.0, 5.0, 40.0])
    iters = rng.randint(1, 4)
    rseed = rng.randrange(-(1 << 62), 1 << 62)
    trace = rng.random() < 0.6
    return uid, iid, vals, nu, ni, implicit, want[1], f, lam, alpha, iters, rseed, trace


def dense_user_step(w, other, nu, f, lam, alpha, implicit):
    """the closed form of the implicit user step over ALL items (docs/recommend.md 4.2), solved densely with numpy: x_u = (Y^T C_u Y + lam I)^-1 Y^T C_u p_u"""
    y = np.array(other, dtype=np.float64)
    ni = y.shape[0]
    out = []
    for u in range(nu):
        if not implicit:
            ks = list(range(w["uoff"][u], w["uoff"][u + 1]))
            yo = y[[w["uitem"][k] for k in ks]]
            r = np.array([w["uval"][k] for k in ks])
            out.append(np.linalg.solve(yo.T @ yo + lam * np.eye(f), yo.T @ r))
            continue
        c = np.ones(ni)
        pref = np.zeros(ni)
        for k in range(w["uoff"][u], w["uoff"][u + 1]):
            c[w["uitem"][k]] = 1.0 + alpha * w["uval"][k]
            pref[w["uitem"][k]] = 1.0
        a_ = y.T @ (y * c[:, None]) + lam * np.eye(f)
        b = y.T @ (c * pref)
        out.append(np.linalg.solve(a_, b))
    return np.array(out)


def check_als(seed):
    g = gen_als(seed)
    if g is None:
        return None
    uid, iid, vals, nu, ni, implicit, w, f, lam, alpha, iters, rseed, trace = g
    want = als_oracle(w, nu, ni, f, lam, alpha, implicit, iters, rseed, trace)
    rc, got, text = run_als_driver(uid, iid, vals, nu, ni, implicit, f, lam, alpha, iters, rseed, trace)
    tag = "seed %d (users=%d items=%d f=%d lam=%g alpha=%g iters=%d implicit=%s trace=%s)" % (seed, nu, ni, f, lam, alpha, iters, implicit, trace)
    if want[0] == "refused":
        if ("refused " + want[1]) not in text:
            return "%s: expected %s, got %r" % (tag, want[1], text[:120])
        return None
    if rc != 0:
        return "%s: driver exit %d: %s" % (tag, rc, text[:160])
    _, pf, qf, losses = want
    if parse_bits(got.get("p", "")) != [bits(x) for row in pf for x in row]:
        return "%s: P differs" % tag
    if parse_bits(got.get("q", "")) != [bits(x) for row in qf for x in row]:
        return "%s: Q differs" % tag
    if parse_bits(got.get("loss", "")) != [bits(x) for x in losses]:
        return "%s: losses differ" % tag
    # RC5: the loss does not rise from one half-pass to the next (a half-pass minimises over its block)
    for a, b in zip(losses, losses[1:]):
        if b > a + 1e-9 * max(1.0, abs(a)):
            return "%s: the loss rose from %r to %r" % (tag, a, b)
    return None


def check_dense(seed):
    """RC4: with one pass, the factors of the item step equal the closed form solved densely (explicit: the observed items only; implicit: all items with the confidences) from the P the fit
    returned, and P equals the dense step from the initial Q"""
    g = gen_als(seed)
    if g is None:
        return None
    uid, iid, vals, nu, ni, implicit, w, f, lam, alpha, iters, rseed, trace = g
    rc, got, text = run_als_driver(uid, iid, vals, nu, ni, implicit, f, lam, alpha, 1, rseed, False)
    if rc != 0:
        return None
    from struct import unpack, pack
    def floats(t):
        return [unpack("<d", pack("<Q", int(h, 16)))[0] for h in parse_bits(t)]
    p_out = np.array(floats(got["p"])).reshape(nu, f)
    q_out = np.array(floats(got["q"])).reshape(ni, f)
    q0 = init_q(ni, f, rseed)
    p_dense = dense_user_step(w, q0, nu, f, lam, alpha, implicit)
    # the item step: the transposed log
    wt = {"uoff": w["ioff"], "uitem": w["iuser"], "uval": w["ival"]}
    q_dense = dense_user_step(wt, p_out.tolist(), ni, f, lam, alpha, implicit)
    tol = 1e-7
    if np.max(np.abs(p_dense - p_out)) > tol * max(1.0, np.max(np.abs(p_dense))):
        return "dense user step, seed %d: worst difference %g" % (seed, np.max(np.abs(p_dense - p_out)))
    if np.max(np.abs(q_dense - q_out)) > tol * max(1.0, np.max(np.abs(q_dense))):
        return "dense item step, seed %d: worst difference %g" % (seed, np.max(np.abs(q_dense - q_out)))
    return None


def als_controls():
    bad = []
    # one user, one item, explicit r = 2, f = 1, lambda = 1, one pass: p = 2 q0 / (q0^2 + 1), then q = 2 p / (p^2 + 1)
    uid, iid, vals = [0], [0], [2.0]
    q0 = init_q(1, 1, 7)[0][0]
    p1 = 2.0 * q0 / (q0 * q0 + 1.0)
    q1 = 2.0 * p1 / (p1 * p1 + 1.0)
    rc, got, text = run_als_driver(uid, iid, vals, 1, 1, False, 1, 1.0, 1.0, 1, 7, False)
    from struct import unpack, pack
    def fl(t):
        return [unpack("<d", pack("<Q", int(h, 16)))[0] for h in parse_bits(t)]
    if rc != 0 or abs(fl(got.get("p", ""))[0] - p1) > 1e-12 or abs(fl(got.get("q", ""))[0] - q1) > 1e-12:
        bad.append("hand-worked 1 x 1 fit: %r" % text[:160])
    # the initial factor is in [-1, 1) / sqrt(f): f = 4 divides by 2
    if not all(-0.5 <= v < 0.5 for row in init_q(5, 4, 3) for v in row):
        bad.append("initial factors out of range")
    f = bits(1.0)
    base = ["als", 1, 1, 0, "L:0", "L:0", "H:" + bits(2.0), 100]

    def refuse(name, tail, tag, code):
        r = subprocess.run([LOG_BIN] + [str(a) for a in base + tail], capture_output=True, text=True, timeout=60)
        if ("refused " + tag) not in r.stdout or r.returncode != code:
            bad.append("%s: expected %s exit %d, got %r exit %d" % (name, tag, code, r.stdout.strip()[:80], r.returncode))

    one = "H:" + f
    refuse("f = 0", [0, one, one, 1, 1, 0], "ml.param.out-of-range", 2)
    refuse("f above the limit", [257, one, one, 1, 1, 0], "ml.param.out-of-range", 2)
    refuse("iterations 0", [1, one, one, 0, 1, 0], "ml.param.out-of-range", 2)
    refuse("iterations above the limit", [1, one, one, 1001, 1, 0], "ml.param.out-of-range", 2)
    refuse("lambda 0", [1, "H:" + bits(0.0), one, 1, 1, 0], "ml.param.out-of-range", 2)
    refuse("lambda negative", [1, "H:" + bits(-1.0), one, 1, 1, 0], "ml.param.out-of-range", 2)
    refuse("lambda NaN", [1, "H:7ff8000000000000", one, 1, 1, 0], "ml.param.out-of-range", 2)
    refuse("lambda infinite", [1, "H:7ff0000000000000", one, 1, 1, 0], "ml.param.out-of-range", 2)
    # a tiny lambda on a single interaction leaves the system singular to working precision: the Cholesky refuses it, and the fit says so
    refuse("a singular system", [2, "H:" + bits(1e-300), one, 1, 1, 0], "ml.linalg.not-positive-definite", 8)
    # alpha is read only by the implicit form
    r = subprocess.run([LOG_BIN] + [str(a) for a in base + [1, one, "H:" + bits(0.0), 1, 1, 0]], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        bad.append("alpha 0 refused by the explicit form: %r" % r.stdout[:80])
    r = subprocess.run([LOG_BIN] + [str(a) for a in ["als", 1, 1, 1, "L:0", "L:0", "H:" + bits(2.0), 100, 1, one, "H:" + bits(0.0), 1, 1, 0]], capture_output=True, text=True, timeout=60)
    if "refused ml.param.out-of-range" not in r.stdout:
        bad.append("alpha 0 accepted by the implicit form: %r" % r.stdout[:80])
    return bad


# ---------------------------------------------------------------- the model payload (docs/wiring-recommend.md section 5, gate WR9)
def w64(v):
    return _st.pack("<Q", v & MASK64)


def wf(x):
    return _st.pack("<d", x)


def payload_oracle(w, nu, ni, implicit, f, lam, alpha, iters, seed, uids, iids, pf, qf):
    """the bytes of the payload, from the definitions of the document (no cancho code)"""
    out = b""
    for v in (10, 1 if implicit else 0, f, iters, nu, ni, w["nnz"]):
        out += w64(v)
    out += wf(lam) + wf(alpha if implicit else 0.0) + w64(seed)
    for ids in (uids, iids):
        out += w64(sum(len(x) for x in ids))
        off = 0
        out += w64(0)
        for x in ids:
            off += len(x)
            out += w64(off)
        out += b"".join(ids)
    out += b"".join(w64(v) for v in w["uoff"])
    out += b"".join(w64(v) for v in w["uitem"])
    out += b"".join(wf(v) for v in w["uval"])
    out += b"".join(wf(v) for row in pf for v in row)
    out += b"".join(wf(v) for row in qf for v in row)
    return out


def gen_ids(rng, n):
    """n distinct non-empty byte strings of 1 to 12 bytes, sorted bytewise: prefixes of one another, high bytes, zero bytes"""
    seen = set()
    while len(seen) < n:
        ln = rng.choice([1, 1, 2, 3, 5, 12])
        seen.add(bytes(rng.choice([0, 1, 48, 49, 97, 98, 127, 128, 255]) for _ in range(ln)))
    return sorted(seen)


def run_payload_driver(uid, iid, vals, nu, ni, implicit, f, lam, alpha, iters, seed, uids, iids):
    spec_u = "X:" + ",".join(x.hex() for x in uids)
    spec_i = "X:" + ",".join(x.hex() for x in iids)
    return run_log_driver(["payload", nu, ni, 1 if implicit else 0, spec_ints(uid), spec_ints(iid), spec_floats(vals), 10 ** 9, f, spec_floats([lam]), spec_floats([alpha]), iters, seed, spec_u, spec_i])


def read_driver(data):
    r = subprocess.run([LOG_BIN, "read", data.hex()], capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout.strip()


def payload_layout(nu, ni, nnz, f, ub, ib):
    o = {}
    o["hdr"] = 0
    o["ub"] = 80
    o["uoffs"] = 88
    o["ubytes"] = 88 + 8 * (nu + 1)
    o["ib"] = o["ubytes"] + ub
    o["ioffs"] = o["ib"] + 8
    o["ibytes"] = o["ioffs"] + 8 * (ni + 1)
    o["hoff"] = o["ibytes"] + ib
    o["hitem"] = o["hoff"] + 8 * (nu + 1)
    o["hval"] = o["hitem"] + 8 * nnz
    o["p"] = o["hval"] + 8 * nnz
    o["q"] = o["p"] + 8 * nu * f
    o["end"] = o["q"] + 8 * ni * f
    return o


def set64(b, at, v):
    return b[:at] + w64(v) + b[at + 8:]


def setf(b, at, x):
    return b[:at] + wf(x) + b[at + 8:]


def check_payload(seed, stride_all):
    g = gen_als(seed * 3 + 1)
    tries = 0
    while g is None and tries < 50:
        tries += 1
        g = gen_als(seed * 3 + 1 + tries * 1000)
    uid, iid, vals, nu, ni, implicit, w, f, lam, alpha, iters, rseed, trace = g
    rng = random.Random(seed)
    uids = gen_ids(rng, nu)
    iids = gen_ids(rng, ni)
    want = als_oracle(w, nu, ni, f, lam, alpha, implicit, iters, rseed, False)
    if want[0] != "ok":
        return None
    data = payload_oracle(w, nu, ni, implicit, f, lam, alpha, iters, rseed, uids, iids, want[1], want[2])
    rc, got, text = run_payload_driver(uid, iid, vals, nu, ni, implicit, f, lam, alpha, iters, rseed, uids, iids)
    if rc != 0 or got.get("payload") != data.hex():
        return "seed %d: the payload differs from the oracle's (exit %d)" % (seed, rc)
    rc, text = read_driver(data)
    if rc != 0 or not text.startswith("ok users %d items %d nnz %d f %d" % (nu, ni, w["nnz"], f)) or not text.endswith("roundtrip same"):
        return "seed %d: reading the payload back: %r" % (seed, text[:100])
    lay = payload_layout(nu, ni, w["nnz"], f, sum(len(x) for x in uids), sum(len(x) for x in iids))
    bad = []

    def expect(name, blob, code):
        rc, t = read_driver(blob)
        if code == 0:
            if rc != 0 or not t.endswith("roundtrip same"):
                bad.append("%s: expected a canonical read, got %r" % (name, t[:80]))
        elif rc != 8 or t != "refused %d" % code:
            bad.append("%s: expected refused %d, got %r (exit %d)" % (name, code, t[:80], rc))

    # every truncation is truncated, and one byte too many is malformed (a stride on the long payloads keeps the run short)
    step = 1 if len(data) < 600 else 5
    for ln in list(range(0, len(data), step)) + [len(data) - 1]:
        expect("truncated at %d" % ln, data[:ln], 2)
    expect("one byte too many", data + b"\0", 1)
    named = [
        ("kind", set64(data, 0, 9), 1), ("method 2", set64(data, 8, 2), 1), ("f 0", set64(data, 16, 0), 1), ("f 257", set64(data, 16, 257), 1),
        ("iterations 0", set64(data, 24, 0), 1), ("iterations 1001", set64(data, 24, 1001), 1), ("users 0", set64(data, 32, 0), 1),
        ("items 0", set64(data, 40, 0), 1), ("nnz 0", set64(data, 48, 0), 1), ("nnz over the limit", set64(data, 48, 1 << 40), 1),
        ("lambda NaN", set64(data, 56, 0x7ff8000000000000), 1), ("lambda 0", setf(data, 56, 0.0), 1), ("lambda negative", setf(data, 56, -1.0), 1),
        ("alpha NaN", set64(data, 64, 0x7ff8000000000000), 1),
        ("user bytes too few", set64(data, 80, nu - 1), 1), ("user bytes too many", set64(data, 80, nu * 255 + 1), 1),
        ("item bytes too few", set64(data, lay["ib"], ni - 1), 1), ("item bytes too many", set64(data, lay["ib"], ni * 255 + 1), 1),
        ("first user offset", set64(data, lay["uoffs"], 1), 1),
        ("item offset not ending at the bytes", set64(data, lay["ioffs"] + 8 * ni, sum(len(x) for x in iids) - 1), 1),
        ("history offset end", set64(data, lay["hoff"] + 8 * nu, w["nnz"] - 1), 1),
        ("history offset start", set64(data, lay["hoff"], 1), 1),
        ("an item number out of range", set64(data, lay["hitem"], ni), 1), ("a negative item number", set64(data, lay["hitem"], -1), 1),
        ("a value NaN", set64(data, lay["hval"], 0x7ff8000000000000), 1), ("a value above the bound", setf(data, lay["hval"], 2.0e6), 1),
        ("P with an infinity", set64(data, lay["p"], 0x7ff0000000000000), 1), ("Q with a NaN", set64(data, lay["q"], 0x7ff8000000000000), 1),
    ]
    if implicit:
        named.append(("an implicit zero value", setf(data, lay["hval"], 0.0), 1))
        named.append(("alpha 0", setf(data, 64, 0.0), 1))
    else:
        named.append(("an explicit alpha", setf(data, 64, 1.0), 1))
    if nu >= 2:
        # two user ids swapped (not ascending) and an id made equal to its neighbour
        a, b = lay["ubytes"], lay["ubytes"]
        ids = list(uids)
        ids[0], ids[1] = ids[1], ids[0]
        if len(ids[0]) == len(uids[0]):
            named.append(("user ids not ascending", data[:lay["ubytes"]] + b"".join(ids) + data[lay["ubytes"] + len(b"".join(ids)):], 1))
    if nu >= 2 and len(uids[0]) == len(uids[1]):
        dup = list(uids)
        dup[1] = dup[0]
        named.append(("equal user ids", data[:lay["ubytes"]] + b"".join(dup) + data[lay["ubytes"] + len(b"".join(dup)):], 1))
        sw = list(uids)
        sw[0], sw[1] = sw[1], sw[0]
        named.append(("user ids swapped", data[:lay["ubytes"]] + b"".join(sw) + data[lay["ubytes"] + len(b"".join(sw)):], 1))
    named.append(("an empty user id", set64(data, lay["uoffs"] + 8, 0), 1))
    named.append(("a value below the bound", setf(data, lay["hval"], -2.0e6), 1))
    # an id of 256 bytes: a payload the writer of the oracle makes with a 256-byte first user id (everything else valid)
    if nu >= 1:
        big = [bytes([1]) * 256] + [bytes([2 + i]) * 2 for i in range(nu - 1)]
        big = sorted(big)
        named.append(("a user id of 256 bytes", payload_oracle(w, nu, ni, implicit, f, lam, alpha, iters, rseed, big, iids, want[1], want[2]), 1))
    # a user with an empty history range: the second offset made the first
    if nu >= 2:
        named.append(("a user with no history", set64(data, lay["hoff"] + 8, 0), 1))
    # two items of one user swapped (not ascending)
    for u in range(nu):
        lo, hi = w["uoff"][u], w["uoff"][u + 1]
        if hi - lo >= 2:
            blob = set64(set64(data, lay["hitem"] + 8 * lo, w["uitem"][lo + 1]), lay["hitem"] + 8 * (lo + 1), w["uitem"][lo])
            named.append(("history items swapped", blob, 1))
            break
    # an item that no history holds: replace one entry by another number that keeps the user's list ascending and leaves some item uncovered
    done = False
    for u in range(nu):
        lo, hi = w["uoff"][u], w["uoff"][u + 1]
        for j in range(lo, hi):
            for v in range(ni):
                prev = w["uitem"][j - 1] if j > lo else -1
                nxt = w["uitem"][j + 1] if j + 1 < hi else ni
                if prev < v < nxt and v != w["uitem"][j]:
                    items = list(w["uitem"])
                    items[j] = v
                    if set(items) != set(range(ni)):
                        named.append(("an item in no history", set64(data, lay["hitem"] + 8 * j, v), 1))
                        done = True
                        break
            if done:
                break
        if done:
            break
    # a positive finite lambda of any bit pattern is read back as it was written (the decoder reproduces every finite pattern): the smallest and largest subnormals, the smallest normal,
    # the largest finite, and random patterns of every exponent
    pr = random.Random(seed + 77)
    for pat in [1, 0x000FFFFFFFFFFFFF, 0x0010000000000000, 0x7FEFFFFFFFFFFFFF] + [pr.randrange(1 << 52) | (pr.randrange(1, 2047) << 52) for _ in range(12)]:
        named.append(("lambda bits %016x" % pat, set64(data, 56, pat), 0))
    for name, blob, code in named:
        expect(name, blob, code)
    # every byte flipped: the result is refused cleanly or read back canonical (a flipped value of P, say, is another valid model)
    flips = 1 if stride_all else 7
    for at in range(0, len(data), flips):
        blob = data[:at] + bytes([data[at] ^ 0xFF]) + data[at + 1:]
        rc, t = read_driver(blob)
        if not ((rc == 8 and t in ("refused 1", "refused 2")) or (rc == 0 and t.endswith("roundtrip same"))):
            bad.append("flip at %d: %r (exit %d)" % (at, t[:80], rc))
            break
    return "; ".join(bad[:4]) if bad else None


def check_find(seed):
    rng = random.Random(seed * 13 + 1)
    n = rng.randint(1, 30)
    ids = gen_ids(rng, n)
    spec = "X:" + ",".join(x.hex() for x in ids)
    keys = [rng.choice(ids) for _ in range(3)] + [ids[0][:1] + b"\x7e", ids[-1] + b"\0", b"\0", b"\xff" * 3, ids[0][:-1] if len(ids[0]) > 1 else b"\x01\x02"]
    for key in keys:
        r = subprocess.run([LOG_BIN, "find", spec, key.hex()], capture_output=True, text=True, timeout=60)
        want = ids.index(key) if key in ids else -1
        if r.stdout.strip() != "index %d" % want:
            return "find seed %d: key %s expected %d, got %r" % (seed, key.hex(), want, r.stdout.strip())
    return None


def main():
    argv = sys.argv[1:]
    if not any(a in argv for a in ("--controls", "--driver", "--metrics", "--log", "--log-controls", "--als", "--als-controls", "--payload")):
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
    if "--als-controls" in argv:
        bad = als_controls()
        for b in bad:
            print("FAIL als control:", b)
        print("oracle_recommend als controls: %s" % ("all as defined" if not bad else "%d failed" % len(bad)))
        rc |= 1 if bad else 0
    if "--als" in argv:
        at = argv.index("--als") + 1
        n = int(argv[at]) if len(argv) > at and argv[at].isdigit() else 120
        bad = 0
        ran = 0
        for seed in range(1, n + 1):
            for msg in (check_als(seed), check_dense(seed)):
                if msg:
                    bad += 1
                    print("MISMATCH", msg)
            ran += 1
            if bad > 10:
                break
        print("oracle_recommend als: %d fits, %s" % (ran, "all agree bit for bit; the loss never rises; each step equals the dense closed form" if not bad else "%d mismatches" % bad))
        rc |= 1 if bad else 0
    if "--payload" in argv:
        at = argv.index("--payload") + 1
        n = int(argv[at]) if len(argv) > at and argv[at].isdigit() else 40
        bad = 0
        for seed in range(1, n + 1):
            msg = check_payload(seed, seed <= 5) or check_find(seed)
            if msg:
                bad += 1
                print("MISMATCH", msg)
        print("oracle_recommend payload: %d payloads, %s" % (n, "bytes equal the oracle's; every truncation, corruption and flip is refused or read back canonical" if not bad else "%d mismatches" % bad))
        rc |= 1 if bad else 0
    return rc


if __name__ == "__main__":
    sys.exit(main())

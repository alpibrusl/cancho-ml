#!/usr/bin/env python3
"""Independent oracle for the split strategies and cross-validation of `ml` (docs/splits.md).

Nothing here imports or calls cancho code. The stream (SplitMix64, rejection sampling, Fisher-Yates), the CSV dialect, the group and
time rules, every assignment of docs/splits.md section 3 and the aggregate of section 6 are written again in Python from the text of
the document, and compared with the bytes the tool prints (`--emit-folds`) and with the metrics it computes (by fitting and
evaluating the real binary on the files this oracle cuts).

    python3 scripts/oracle_split.py --assign       A1: the assignment of every strategy, bytes
    python3 scripts/oracle_split.py --invariants   B1: properties on generated and adversarial data
    python3 scripts/oracle_split.py --e2e          C1: per fold, `ml fit` + `ml eval` on the files this oracle writes == `ml cv`
    python3 scripts/oracle_split.py --threads      D1: threads 1 and 8 give the same bytes
    python3 scripts/oracle_split.py --sklearn      S1: scikit-learn on the folds of --emit-folds
    python3 scripts/oracle_split.py --all
Exit 1 on any mismatch. ML_BIN names the binary (default build/ml)."""
import csv
import io
import json
import math
import os
import pathlib
import random
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = os.environ.get("ML_BIN") or str(ROOT / "build" / "ml")
M64 = (1 << 64) - 1
GOLD = 0x9E3779B97F4A7C15
FAILS = []


# ---------------------------------------------------------------- the stream (docs/rng.md, docs/metrics.md 2.1 and 2.2)
def mix(z):
    z &= M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def derive(h, c):
    return mix((((h & M64) ^ (c & M64)) + GOLD) & M64)


def stream(seed, kind, index):
    return derive(derive(seed, kind), index)


class Stream:
    def __init__(self, s0):
        self.s = s0 & M64

    def draw(self):
        self.s = (self.s + GOLD) & M64
        return mix(self.s)

    def bounded(self, m):
        r = (1 << 63) % m
        while True:
            v = self.draw() >> 1
            if v < (1 << 63) - r:
                return v % m


def shuffle(a, s0):
    st = Stream(s0)
    for i in range(len(a) - 1, 0, -1):
        j = st.bounded(i + 1)
        a[i], a[j] = a[j], a[i]
    return a


def fold_start(n, k, f):
    return f * (n // k) + min(f, n % k)


# ---------------------------------------------------------------- the CSV dialect (docs/reader.md section 2) and the two columns
def read_csv(path):
    """(header, rows) with every cell as str: BOM dropped, CR before LF dropped, blank lines are not records, RFC 4180 quotes."""
    raw = pathlib.Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    text = raw.decode("utf-8")
    recs = [r for r in csv.reader(io.StringIO(text, newline="")) if r != []]
    return recs[0], recs[1:]


DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?(Z)?)?$")
NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
OFFSET = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d{1,9})?)?[+-]\d{2}(:?\d{2})?$")
FORMLIKE = re.compile(r"^\d{4}-")


def days_in(y, m):
    if m == 2:
        return 29 if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0 else 28
    return 30 if m in (4, 6, 9, 11) else 31


def time_key(cell):
    """(kind, key) or (None, reason): kind 'num' key (float,), kind 'iso' key (YYYYMMDDhhmmss, nanos); docs/splits.md section 4"""
    m = DATE.match(cell)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hh = int(m.group(4) or 0)
        mi = int(m.group(5) or 0)
        ss = int(m.group(6) or 0)
        frac = m.group(7) or ""
        if m.group(8) and m.group(4) is None:
            return None, "form"
        if not (1 <= mo <= 12 and 1 <= d <= days_in(y, mo) and hh <= 23 and mi <= 59 and ss <= 59):
            return None, "range"
        nanos = int(frac.ljust(9, "0")) if frac else 0
        return "iso", (((((y * 100 + mo) * 100 + d) * 100 + hh) * 100 + mi) * 100 + ss, nanos)
    if OFFSET.match(cell):
        return None, "offset"
    if NUMBER.match(cell):
        return "num", (float(cell), 0)
    if FORMLIKE.match(cell):
        return None, "form"
    return None, "not-a-time"


class Refusal(Exception):
    def __init__(self, rule, **detail):
        self.rule = rule
        self.detail = detail


def column(header, rows, name):
    if name not in header:
        raise Refusal("ml.input.unknown-column")
    return header.index(name)


def group_ranks(cells):
    """ranks of the distinct texts sorted bytewise, and the rank of each row"""
    for i, c in enumerate(cells):
        if c == "":
            raise Refusal("ml.split.missing-cell", row=i + 1)
    distinct = sorted(set(cells), key=lambda s: s.encode("utf-8"))
    rank = {g: r for r, g in enumerate(distinct)}
    return len(distinct), [rank[c] for c in cells]


def time_keys(cells):
    kind = None
    keys = []
    for i, c in enumerate(cells):
        if c == "":
            raise Refusal("ml.split.missing-cell", row=i + 1)
        k, key = time_key(c)
        if k is None:
            raise Refusal("ml.split.time-format", row=i + 1, reason=key)
        if kind is None:
            kind = k
        elif kind != k:
            raise Refusal("ml.split.time-format", row=i + 1, reason="mixed")
        keys.append(key)
    return keys


def time_order(keys):
    return sorted(range(len(keys)), key=lambda i: (keys[i], i))


# ---------------------------------------------------------------- the assignments (docs/splits.md section 3)
def parse_train(text):
    """'train=F' -> (num, den) of the TEST fraction, and the canonical F"""
    m = re.match(r"^train=(\d*)\.(\d{1,6})$", text)
    assert m, text
    whole, frac = m.group(1), m.group(2)
    den = 10 ** len(frac)
    train = int(whole or "0") * den + int(frac)
    assert 0 < train < den
    canon = ("0." + str(train).rjust(len(frac), "0")).rstrip("0")
    return den - train, den, canon


def n_test_for(n, num, den):
    return (n * num + den - 1) // den


def classes_of(labels):
    return sorted(set(labels))


def holdout(strategy, n, train, seed, labels=None, gids=None, G=0, tkeys=None):
    num, den, _ = parse_train(train)
    nt = n_test_for(n, num, den)
    if nt < 1 or nt > n - 1:
        raise Refusal("ml.split.empty-fold")
    out = [0] * n
    if strategy == "plain":
        perm = shuffle(list(range(n)), stream(seed, 4, 0))
        for p in perm[:nt]:
            out[p] = 1
    elif strategy == "stratify":
        cl = classes_of(labels)
        L = []
        for c, lab in enumerate(cl):
            mem = [i for i in range(n) if labels[i] == lab]
            if len(mem) < 2:
                raise Refusal("ml.split.stratum-too-small")
            L += shuffle(mem, stream(seed, 7, c))
        for t, i in enumerate(L):
            out[i] = ((t + 1) * nt) // n - (t * nt) // n
        for lab in cl:
            if all(out[i] == 1 for i in range(n) if labels[i] == lab):
                raise Refusal("ml.split.stratum-too-small")
    elif strategy == "group":
        if G < 2:
            raise Refusal("ml.split.too-few-groups")
        size = [0] * G
        for g in gids:
            size[g] += 1
        perm = shuffle(list(range(G)), stream(seed, 8, 0))
        test = set()
        rows = 0
        for g in perm:
            if rows < nt and len(test) < G - 1:
                test.add(g)
                rows += size[g]
        out = [1 if g in test else 0 for g in gids]
    elif strategy == "time":
        order = time_order(tkeys)
        for i in order[n - nt:]:
            out[i] = 1
    return out


def cv(strategy, n, K, R, seed, labels=None, gids=None, G=0, tkeys=None):
    """a list of R assignments (fold per row; -1 for the initial block of a time split)"""
    if strategy == "time" and R > 1:
        raise Refusal("ml.split.combination")
    res = []
    for r in range(R):
        out = [0] * n
        if strategy == "plain":
            if K > n:
                raise Refusal("ml.split.empty-fold")
            perm = shuffle(list(range(n)), stream(seed, 5, r))
            for f in range(K):
                for p in range(fold_start(n, K, f), fold_start(n, K, f + 1)):
                    out[perm[p]] = f
        elif strategy == "stratify":
            cl = classes_of(labels)
            L = []
            for c, lab in enumerate(cl):
                mem = [i for i in range(n) if labels[i] == lab]
                if len(mem) < K:
                    raise Refusal("ml.split.stratum-too-small")
                L += shuffle(mem, stream(seed, 6, 2 * r + c))
            for t, i in enumerate(L):
                out[i] = t % K
        elif strategy == "group":
            if G < K:
                raise Refusal("ml.split.too-few-groups")
            size = [0] * G
            for g in gids:
                size[g] += 1
            perm = shuffle(list(range(G)), stream(seed, 8, r))
            load = [0] * K
            where = [0] * G
            for g in perm:
                f = min(range(K), key=lambda x: (load[x], x))
                where[g] = f
                load[f] += size[g]
            out = [where[g] for g in gids]
        elif strategy == "time":
            if n < K + 1:
                raise Refusal("ml.split.empty-fold")
            order = time_order(tkeys)
            for b in range(K + 1):
                for p in range(fold_start(n, K + 1, b), fold_start(n, K + 1, b + 1)):
                    out[order[p]] = b - 1
        res.append(out)
    return res


# ---------------------------------------------------------------- the canonical text (docs/splits.md section 2)
def enc_name(name):
    return "".join(chr(b) if (chr(b).isascii() and (chr(b).isalnum() or chr(b) in "_.-")) else "%%%02X" % b for b in name.encode("utf-8"))


def spec_text(kind, strategy, seed, column=None, train=None, K=None, R=None):
    parts = [kind]
    parts.append("train=" + parse_train(train)[2] if kind == "holdout" else "folds=%d repeats=%d" % (K, R))
    if strategy == "stratify":
        parts.append("stratify")
    elif strategy == "group":
        parts.append("group=" + enc_name(column))
    elif strategy == "time":
        parts.append("time=" + enc_name(column))
    parts.append("seed=%d" % seed)
    return " ".join(parts)


# ---------------------------------------------------------------- the aggregate (docs/splits.md section 6, docs/metrics.md 2.4)
def ordered_sum(t):
    """blocks of 256 terms, four lanes (term i of the block to lane i mod 4, from +0.0), ((l0 + l1) + (l2 + l3)), the fixed tree over blocks"""
    parts = []
    for b in range(0, len(t), 256):
        blk = t[b:b + 256]
        lane = [0.0] * 4
        for i, v in enumerate(blk):
            lane[i % 4] = lane[i % 4] + v
        parts.append((lane[0] + lane[1]) + (lane[2] + lane[3]))
    while len(parts) > 1:
        nxt = [parts[i] + parts[i + 1] for i in range(0, len(parts) - 1, 2)]
        if len(parts) % 2:
            nxt.append(parts[-1])
        parts = nxt
    return parts[0]


def aggregate(values):
    m = len(values)
    mean = ordered_sum(values) / m
    ss = ordered_sum([(v - mean) * (v - mean) for v in values])
    return mean, math.sqrt(ss / (m - 1))


def shortest(x):
    """the text the tool prints for a double: the shortest decimal that reads back (docs/cli.md); repr() agrees for the values used here"""
    return repr(x)


# ---------------------------------------------------------------- generated data
def gen_table(rnd, n, task="cls", groups=12, imbalance=0.5, missing=0.0, dates=True, tie_dates=False):
    """(header, rows): x1, x2 (floats, some empty when `missing`), x3 int, cat text, grp text, day ISO date, t number, y"""
    header = ["x1", "x2", "x3", "cat", "grp", "day", "t", "y"]
    rows = []
    for i in range(n):
        x1 = round(rnd.gauss(0, 1), 4)
        x2 = "" if rnd.random() < missing else str(round(rnd.gauss(1, 2), 3))
        x3 = rnd.randint(0, 9)
        cat = rnd.choice(["red", "green", "blue", "grey"])
        grp = "g%02d" % rnd.randrange(groups)
        if tie_dates:
            day = "2024-%02d-%02d" % (1 + rnd.randrange(3), 1 + rnd.randrange(3))
        else:
            day = "2024-%02d-%02d" % (1 + rnd.randrange(12), 1 + rnd.randrange(28))
        t = rnd.randint(1, 50) if tie_dates else rnd.randint(1, 100000)
        lin = 1.5 * x1 + 0.3 * (0.0 if x2 == "" else float(x2)) + 0.2 * x3 + rnd.gauss(0, 1)
        if task == "cls":
            y = "1" if (lin + rnd.gauss(0, 0.5) > math.log(imbalance / (1 - imbalance)) * -1.0 + 0.4) else "0"
        else:
            y = str(round(lin * 10, 3))
        rows.append([str(x1), x2, str(x3), cat, grp, day, str(t), y])
    return header, rows


def write_csv(path, header, rows, crlf=False, bom=False):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n" if crlf else "\n")
    w.writerow(header)
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    pathlib.Path(path).write_bytes((b"\xef\xbb\xbf" if bom else b"") + data)


# ---------------------------------------------------------------- the binary
WORK = [None]


def run(argv, cwd=None):
    """argv without --root: the scratch directory is the root and every path in argv is relative to it"""
    argv = [str(a) for a in argv]
    p = subprocess.run([BIN, "--root", WORK[0]] + argv, capture_output=True, cwd=cwd, timeout=300)
    out = p.stdout.decode("utf-8", "replace")
    try:
        doc = json.loads(out.splitlines()[0]) if out.strip() else None
    except Exception:
        doc = None
    return p.returncode, doc, p.stderr


def check(ok, what):
    if not ok:
        FAILS.append(what)
        print("MISMATCH", what, flush=True)
    return ok


def strategy_flags(strategy):
    if strategy == "stratify":
        return ["--stratify"]
    if strategy == "group":
        return ["--split-group", "grp"]
    if strategy == "time":
        return ["--split-by", "time:day"]
    return []


def cv_argv(path, strategy, K, R, seed, model, extra=()):
    return ["cv", os.path.basename(path), "--target", "y", "--drop", "grp,day,t,cat", "--model", model, "--folds", K, "--repeats", R, "--seed", seed,
            "--emit-folds"] + strategy_flags(strategy) + list(extra)


def oracle_inputs(header, rows, strategy):
    n = len(rows)
    try:
        labels = [int(r[header.index("y")]) for r in rows]
    except ValueError:
        labels = None
    gids, G, tkeys = None, 0, None
    if strategy == "group":
        G, gids = group_ranks([r[header.index("grp")] for r in rows])
    if strategy == "time":
        tkeys = time_keys([r[header.index("day")] for r in rows])
    return n, labels, gids, G, tkeys


def model_flags(model, task):
    if model in ("tree", "forest"):
        return ["--task", "classification" if task == "cls" else "regression", "--max-depth", "3"] + (["--trees", "5"] if model == "forest" else [])
    if model == "logistic":
        return ["--step", "impute column=x2 strategy=mean;scale column=x1,x2,x3 kind=standard"]
    if model == "linear":
        return ["--step", "impute column=x2 strategy=mean;scale column=x1,x2,x3 kind=standard"]
    return []


def one_class_fold(assigns, labels, K, strategy):
    """whether some held-out side of the folds has one class only: the metric refusal ml.label.single-class of `ml eval` on that side (docs/splits.md section 8)"""
    if labels is None:
        return False
    for a in assigns:
        for f in range(K):
            if len({labels[i] for i in range(len(a)) if a[i] == f}) < 2:
                return True
            tr = {labels[i] for i in range(len(a)) if (a[i] < f if strategy == "time" else a[i] != f)}
            if len(tr) < 2:
                return True
    return False


# ---------------------------------------------------------------- A1: the assignment, bytes
def gate_assign(tmp, quick=False):
    rnd = random.Random(42)
    seeds = [0, 1, 7, 123456789] if not quick else [0, 3]
    cases = 0
    for task, imb in (("cls", 0.5), ("cls", 0.15), ("reg", 0.5)):
        n = rnd.choice([60, 97, 150])
        header, rows = gen_table(rnd, n, task, groups=14, imbalance=imb, dates=True)
        path = os.path.join(tmp, "a1_%s_%s_%d.csv" % (task, imb, n))
        write_csv(path, header, rows, crlf=(imb == 0.15), bom=(imb == 0.15))
        header2, rows2 = read_csv(path)
        assert header2 == header and rows2 == rows
        model = "tree"
        for strategy in ("plain", "stratify", "group", "time"):
            if strategy == "stratify" and task == "reg":
                continue
            for seed in seeds:
                for K, R in ((2, 1), (5, 1), (3, 2)):
                    if strategy == "time" and R > 1:
                        continue
                    argv = cv_argv(os.path.basename(path), strategy, K, R, seed, model, model_flags(model, task))
                    code, doc, _ = run(argv)
                    n_, labels, gids, G, tkeys = oracle_inputs(header, rows, strategy)
                    try:
                        want = cv(strategy, n_, K, R, seed, labels, gids, G, tkeys)
                        if doc is not None and doc.get("ok") is False and doc["error"]["rule"] == "ml.label.single-class" and one_class_fold(want, labels, K, strategy):
                            continue
                    except Refusal as r:
                        check(doc is not None and doc.get("ok") is False and doc["error"]["rule"] == r.rule,
                              "A1 %s %s K=%d R=%d seed=%d: oracle refuses with %s, tool says %s" % (task, strategy, K, R, seed, r.rule, doc and (doc.get("error") or {}).get("rule")))
                        continue
                    ok = check(doc is not None and doc.get("ok"), "A1 %s %s K=%d R=%d seed=%d: tool refused: %s" % (task, strategy, K, R, seed, doc and doc.get("error")))
                    if not ok:
                        continue
                    got = doc["data"]["assignment"]
                    check(got["folds"] == want, "A1 %s %s K=%d R=%d seed=%d: assignment differs" % (task, strategy, K, R, seed))
                    col = {"group": "grp", "time": "day"}.get(strategy)
                    check(doc["data"]["split"] == spec_text("cv", strategy, seed, col, K=K, R=R), "A1 spec text %s" % doc["data"]["split"])
                    cases += 1
                # holdout
                for train in ("train=0.8", "train=0.5", "train=.75", "train=0.33"):
                    argv = ["fit", os.path.basename(path), "--target", "y", "--drop", "grp,day,t,cat", "--model", model, "--split", train, "--seed", seed,
                            "--emit-folds", "--output", "h.model", "--overwrite"] + strategy_flags(strategy) + model_flags(model, task)
                    code, doc, _ = run(argv)
                    n_, labels, gids, G, tkeys = oracle_inputs(header, rows, strategy)
                    try:
                        want = holdout(strategy, n_, train, seed, labels, gids, G, tkeys)
                        if doc is not None and doc.get("ok") is False and doc["error"]["rule"] == "ml.label.single-class" and labels is not None and len({labels[i] for i in range(n_) if want[i] == 1}) < 2:
                            continue
                    except Refusal as r:
                        check(doc is not None and doc.get("ok") is False and doc["error"]["rule"] == r.rule,
                              "A1 holdout %s %s %s seed=%d: oracle refuses with %s, tool says %s" % (task, strategy, train, seed, r.rule, doc and (doc.get("error") or {}).get("rule")))
                        continue
                    ok = check(doc is not None and doc.get("ok"), "A1 holdout %s %s %s seed=%d: tool refused: %s" % (task, strategy, train, seed, doc and doc.get("error")))
                    if not ok:
                        continue
                    sp = doc["data"]["summary"]["split"]
                    check(sp["assignment"] == want, "A1 holdout %s %s %s seed=%d: assignment differs" % (task, strategy, train, seed))
                    check(sp["train"] == want.count(0) and sp["test"] == want.count(1), "A1 holdout sizes")
                    col = {"group": "grp", "time": "day"}.get(strategy)
                    check(sp["spec"] == spec_text("holdout", strategy, seed, col, train=train), "A1 holdout spec %s" % sp["spec"])
                    cases += 1
    print("A1: %d cases compared" % cases)


# ---------------------------------------------------------------- B1: invariants on generated and adversarial data
def gate_invariants(tmp, quick=False):
    rnd = random.Random(7)
    cases = 0
    for trial in range(6 if quick else 20):
        n = rnd.choice([30, 51, 100, 233])
        imb = rnd.choice([0.5, 0.2, 0.08])
        header, rows = gen_table(rnd, n, "cls", groups=rnd.choice([3, 7, 20]), imbalance=imb, tie_dates=(trial % 3 == 0))
        path = os.path.join(tmp, "b1_%d.csv" % trial)
        write_csv(path, header, rows)
        ylab = [int(r[7]) for r in rows]
        gcol = [r[4] for r in rows]
        for K in (2, 3, 5):
            seed = rnd.randrange(1 << 40)
            # stratified: per class the fold counts are floor or ceil of n_c / K
            code, doc, _ = run(cv_argv(os.path.basename(path), "stratify", K, 1, seed, "tree", model_flags("tree", "cls")))
            if doc and doc.get("ok"):
                f = doc["data"]["assignment"]["folds"][0]
                for lab in set(ylab):
                    nc = ylab.count(lab)
                    for k in range(K):
                        c = sum(1 for i in range(n) if ylab[i] == lab and f[i] == k)
                        check(nc // K <= c <= -(-nc // K), "B1 stratified trial=%d K=%d class %s fold %d count %d of %d" % (trial, K, lab, k, c, nc))
                check(sorted(set(f)) == list(range(K)), "B1 stratified folds used")
                cases += 1
            else:
                check(min(ylab.count(0), ylab.count(1)) < K and doc and doc["error"]["rule"] == "ml.split.stratum-too-small", "B1 stratified refusal trial=%d K=%d %s" % (trial, K, doc and doc.get("error")))
            # group: no group in two folds, every fold used
            code, doc, _ = run(cv_argv(os.path.basename(path), "group", K, 1, seed, "tree", model_flags("tree", "cls")))
            if doc and doc.get("ok"):
                f = doc["data"]["assignment"]["folds"][0]
                seen = {}
                for i in range(n):
                    check(seen.setdefault(gcol[i], f[i]) == f[i], "B1 group %s in two folds (trial=%d K=%d)" % (gcol[i], trial, K))
                check(sorted(set(f)) == list(range(K)), "B1 group folds used")
                cases += 1
            else:
                check(len(set(gcol)) < K and doc["error"]["rule"] == "ml.split.too-few-groups", "B1 group refusal %s" % (doc and doc.get("error")))
            # time: every fold's train (blocks <= i) is no later than its test (block i + 1); the blocks are contiguous
            code, doc, _ = run(cv_argv(os.path.basename(path), "time", K, 1, seed, "tree", model_flags("tree", "cls")))
            if doc and doc.get("ok"):
                f = doc["data"]["assignment"]["folds"][0]
                day = [r[5] for r in rows]
                for i in range(K):
                    tr = [day[j] for j in range(n) if f[j] < i]
                    te = [day[j] for j in range(n) if f[j] == i]
                    check(not tr or not te or max(tr) <= min(te), "B1 time fold %d: max train %s > min test %s" % (i, max(tr) if tr else None, min(te) if te else None))
                    check(len(te) > 0, "B1 time fold %d empty" % i)
                cases += 1
            # k-fold: every row tested exactly once, sizes floor/ceil
            code, doc, _ = run(cv_argv(os.path.basename(path), "plain", K, 2, seed, "tree", model_flags("tree", "cls")))
            if doc and doc.get("ok"):
                for f in doc["data"]["assignment"]["folds"]:
                    sizes = [f.count(k) for k in range(K)]
                    check(sum(sizes) == n and all(n // K <= s <= -(-n // K) for s in sizes), "B1 kfold sizes %s" % sizes)
                rs = doc["data"]["results"]
                check(sum(r["test"] for r in rs[:K]) == n, "B1 every row tested once")
                cases += 1
        # holdout stratified: per class the test count is floor or ceil of n_c * n_test / n
        for train in ("train=0.8", "train=0.6", "train=0.9"):
            seed = rnd.randrange(1 << 40)
            argv = ["fit", os.path.basename(path), "--target", "y", "--drop", "grp,day,t,cat", "--model", "tree", "--split", train, "--seed", seed, "--emit-folds", "--stratify",
                    "--output", "b.model", "--overwrite"] + model_flags("tree", "cls")
            code, doc, _ = run(argv)
            if doc and doc.get("ok"):
                a = doc["data"]["summary"]["split"]["assignment"]
                num, den, _ = parse_train(train)
                nt = n_test_for(n, num, den)
                check(sum(a) == nt, "B1 holdout test size")
                for lab in set(ylab):
                    nc = ylab.count(lab)
                    c = sum(1 for i in range(n) if ylab[i] == lab and a[i] == 1)
                    exact = nc * nt / n
                    check(abs(c - exact) < 1.0, "B1 stratified holdout class %s: %d vs exact %.3f" % (lab, c, exact))
                cases += 1
            argv = ["fit", os.path.basename(path), "--target", "y", "--drop", "grp,day,t,cat", "--model", "tree", "--split", train, "--seed", seed, "--emit-folds", "--split-group", "grp",
                    "--output", "b.model", "--overwrite"] + model_flags("tree", "cls")
            code, doc, _ = run(argv)
            if doc and doc.get("ok"):
                a = doc["data"]["summary"]["split"]["assignment"]
                side = {}
                for i in range(n):
                    check(side.setdefault(gcol[i], a[i]) == a[i], "B1 holdout group on both sides")
                check(0 in a and 1 in a, "B1 holdout both sides non-empty")
                cases += 1
            argv = ["fit", os.path.basename(path), "--target", "y", "--drop", "grp,day,t,cat", "--model", "tree", "--split", train, "--seed", seed, "--emit-folds", "--split-by", "time:day",
                    "--output", "b.model", "--overwrite"] + model_flags("tree", "cls")
            code, doc, _ = run(argv)
            if doc and doc.get("ok"):
                a = doc["data"]["summary"]["split"]["assignment"]
                day = [r[5] for r in rows]
                check(max(day[i] for i in range(n) if a[i] == 0) <= min(day[i] for i in range(n) if a[i] == 1), "B1 holdout time order")
                cases += 1
    print("B1: %d cases checked" % cases)


# ---------------------------------------------------------------- C1: end to end, per fold, fit + eval on the files the oracle writes
E2E_PLANS = [
    ("linear", "reg", ["--step", "impute column=x2 strategy=mean;scale column=x1,x2,x3 kind=standard"]),
    ("linear", "reg", ["--step", "impute column=x2 strategy=median;bucket column=x3 kind=quantile bins=3;scale column=* kind=standard"]),
    ("logistic", "cls", ["--step", "impute column=x2 strategy=mean;scale column=x1,x2,x3 kind=standard"]),
    ("tree", "cls", ["--task", "classification", "--max-depth", "3"]),
    ("forest", "reg", ["--task", "regression", "--trees", "6", "--max-depth", "3"]),
    ("forest", "cls", ["--task", "classification", "--trees", "5", "--max-depth", "3"]),
    ("kmeans", "reg", ["--k", "3", "--iterations", "5", "--step", "impute column=x2 strategy=mean;scale column=* kind=standard"]),
    ("linear", "reg", ["--cat", "cat", "--l2", "1.0", "--step", "impute column=x2 strategy=mean;encode column=cat kind=onehot;scale column=x1,x2,x3 kind=standard"]),
    ("logistic", "cls", ["--cat", "cat", "--l2", "1.0", "--step", "impute column=x2 strategy=mean;encode column=cat kind=onehot;scale column=x1,x2,x3 kind=standard"]),
    ("linear", "reg", ["--cat", "cat", "--step", "impute column=x2 strategy=mean;freq column=cat;scale column=* kind=minmax"]),
    ("tree", "reg", ["--cat", "cat", "--task", "regression", "--max-depth", "3", "--step", "hash column=cat buckets=8"]),
]


def fold_files(tmp, header, rows, assign_row, f, mode, tag):
    """the training and held-out files of fold f: mode 'fold' (fold == f is held out), 'time' (blocks <= f train, block f + 1 held out)"""
    if mode == "time":
        tr = [r for r, a in zip(rows, assign_row) if a < f]
        te = [r for r, a in zip(rows, assign_row) if a == f]
    else:
        tr = [r for r, a in zip(rows, assign_row) if a != f]
        te = [r for r, a in zip(rows, assign_row) if a == f]
    p_tr = os.path.join(tmp, "%s_train_%d.csv" % (tag, f))
    p_te = os.path.join(tmp, "%s_test_%d.csv" % (tag, f))
    write_csv(p_tr, header, tr)
    write_csv(p_te, header, te)
    return p_tr, p_te, len(tr), len(te)


def strip_flags(flags, remove):
    out, skip = [], 0
    for i, a in enumerate(flags):
        if skip:
            skip -= 1
            continue
        if a in remove:
            skip = remove[a]
            continue
        out.append(a)
    return out


def gate_e2e(tmp, quick=False, planted=False):
    rnd = random.Random(2026)
    cases = 0
    plans = E2E_PLANS if not quick else E2E_PLANS[:2] + E2E_PLANS[6:8]
    for pi, (model, task, flags) in enumerate(plans):
        header, rows = gen_table(rnd, 120, task, groups=10, imbalance=0.4, missing=0.08)
        path = os.path.join(tmp, "c1_%d.csv" % pi)
        write_csv(path, header, rows)
        for strategy in ("plain", "stratify", "group", "time"):
            if strategy == "stratify" and task != "cls":
                continue
            K, R, seed = (3, 1, 11 + pi) if strategy == "time" else (3, 2, 11 + pi)
            drop = "grp,day,t" if "--cat" in flags else "grp,day,t,cat"
            base = ["--target", "y", "--drop", drop, "--model", model, "--seed", seed] + flags
            if model == "kmeans":
                base = strip_flags(base, {"--target": 1})
                base = [a if a != "y" else a for a in base]
                base = strip_flags(base, {"--drop": 1}) + ["--drop", "y," + drop]
            argv = ["cv", os.path.basename(path), "--folds", K, "--repeats", R, "--emit-folds"] + base + strategy_flags(strategy)
            code, doc, err = run(argv)
            if not check(doc is not None and doc.get("ok"), "C1 plan %d %s: cv refused %s %s" % (pi, strategy, doc and doc.get("error"), err[:200])):
                continue
            data = doc["data"]
            folds = data["assignment"]["folds"]
            check(len(data["results"]) == K * R, "C1 result count")
            for r in range(R):
                for f in range(K):
                    res = data["results"][r * K + f]
                    p_tr, p_te, ntr, nte = fold_files(tmp, header, rows, folds[r], f, "time" if strategy == "time" else "fold", "c1")
                    check((res["train"], res["test"]) == (ntr, nte), "C1 plan %d %s r=%d f=%d sizes %s vs %d %d" % (pi, strategy, r, f, (res["train"], res["test"]), ntr, nte))
                    mp = "c1.model"
                    fit = ["fit", os.path.basename(p_tr), "--output", mp, "--overwrite"] + base
                    c1, d1, e1 = run(fit)
                    if not check(d1 is not None and d1.get("ok"), "C1 plan %d %s r=%d f=%d: fit on the fold file refused %s" % (pi, strategy, r, f, d1 and d1.get("error"))):
                        continue
                    ev = ["eval", os.path.basename(p_te), "--model-file", mp, "--drop", drop if model != "kmeans" else "y," + drop]
                    if model == "kmeans":
                        ev = ["eval", os.path.basename(p_te), "--model-file", mp, "--drop", "y," + drop]
                    c2, d2, e2 = run(ev)
                    if not check(d2 is not None and d2.get("ok"), "C1 plan %d %s r=%d f=%d: eval refused %s" % (pi, strategy, r, f, d2 and d2.get("error"))):
                        continue
                    want = d2["data"]["metrics"]
                    check(res["metrics"] == want, "C1 plan %d (%s %s) %s r=%d f=%d: cv %s != fit+eval %s" % (pi, model, flags[:2], strategy, r, f, json.dumps(res["metrics"])[:200], json.dumps(want)[:200]))
                    cases += 1
            # the aggregate, from the strings the tool printed, with the oracle's sums
            summ = data["summary"]["metrics"]
            for name, agg in summ.items():
                vals = [r["metrics"].get(name) for r in data["results"]]
                if any(not isinstance(v, str) for v in vals):
                    check(agg["mean"] is None, "C1 aggregate of a missing metric")
                    continue
                mean, std = aggregate([float(v) for v in vals])
                check(float(agg["mean"]) == mean and float(agg["std"]) == std, "C1 aggregate %s: tool %s oracle %s %s" % (name, agg, shortest(mean), shortest(std)))
    print("C1: %d fold(s) compared string for string%s" % (cases, " (planted leak)" if planted else ""))


# ---------------------------------------------------------------- D1: thread invariance
def gate_threads(tmp, quick=False):
    rnd = random.Random(5)
    header, rows = gen_table(rnd, 150, "cls", groups=9, missing=0.05)
    path = os.path.join(tmp, "d1.csv")
    write_csv(path, header, rows)
    cases = 0
    for model, task, flags in (E2E_PLANS[0], E2E_PLANS[2], E2E_PLANS[3], E2E_PLANS[5], E2E_PLANS[6]):
        for strategy in ("plain", "stratify", "group", "time"):
            if strategy == "stratify" and task != "cls":
                continue
            outs = []
            for th in (1, 8):
                base = ["--target", "y", "--drop", "grp,day,t,cat", "--model", model, "--seed", 5, "--threads", th] + flags
                if model == "kmeans":
                    base = strip_flags(base, {"--target": 1, "--drop": 1}) + ["--drop", "y,grp,day,t,cat"]
                c, d, _ = run(["cv", os.path.basename(path), "--folds", 3, "--emit-folds"] + base + strategy_flags(strategy))
                outs.append(json.dumps(d, sort_keys=True))
            check(outs[0] == outs[1], "D1 %s %s: threads 1 and 8 differ" % (model, strategy))
            cases += 1
    print("D1: %d answers equal at 1 and 8 threads" % cases)


# ---------------------------------------------------------------- S1: scikit-learn on the same explicit folds
def gate_sklearn(tmp, quick=False):
    import numpy as np
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error
    rnd = random.Random(99)
    worst = {"reg": 0.0, "acc": 0.0, "ll": 0.0}
    for task, model in (("reg", "linear"), ("cls", "logistic")):
        header, rows = gen_table(rnd, 240, task, groups=12, imbalance=0.45)
        path = os.path.join(tmp, "s1_%s.csv" % task)
        write_csv(path, header, rows)
        for strategy in ("plain", "stratify", "group", "time"):
            if strategy == "stratify" and task == "reg":
                continue
            argv = ["cv", os.path.basename(path), "--target", "y", "--drop", "grp,day,t,cat", "--model", model, "--folds", 4, "--seed", 3, "--emit-folds", "--step", "scale column=x1,x2,x3 kind=standard"] + strategy_flags(strategy)
            rows_ok = [r for r in rows if r[1] != ""]
            # no missing cells in this table (missing=0), so the fold files are the table's rows
            code, doc, _ = run(argv)
            if not check(doc is not None and doc.get("ok"), "S1 refused %s" % (doc and doc.get("error"))):
                continue
            folds = doc["data"]["assignment"]["folds"][0]
            X = np.array([[float(r[0]), float(r[1]), float(r[2])] for r in rows])
            y = np.array([float(r[7]) for r in rows])
            for f in range(4):
                if strategy == "time":
                    tr = np.array([a < f for a in folds])
                else:
                    tr = np.array([a != f for a in folds])
                te = np.array([a == f for a in folds])
                mu, sd = X[tr].mean(0), X[tr].std(0)
                Xs = (X - mu) / sd
                res = doc["data"]["results"][f]["metrics"]
                if task == "reg":
                    m = LinearRegression().fit(Xs[tr], y[tr])
                    p = m.predict(Xs[te])
                    rm = math.sqrt(mean_squared_error(y[te], p))
                    mae = mean_absolute_error(y[te], p)
                    for key, want in (("rmse", rm), ("mae", mae)):
                        got = float(res[key])
                        rel = abs(got - want) / max(abs(want), 1e-12)
                        worst["reg"] = max(worst["reg"], rel)
                        check(rel <= 1e-6, "S1 linear %s fold %d %s: %r vs sklearn %r" % (strategy, f, key, got, want))
                else:
                    m = LogisticRegression(C=1e10, max_iter=10000, tol=1e-12).fit(Xs[tr], y[tr])
                    p = m.predict_proba(Xs[te])[:, 1]
                    acc = accuracy_score(y[te], (p > 0.5).astype(float))
                    ll = log_loss(y[te], p, labels=[0.0, 1.0])
                    ga, gl = float(res["accuracy"]), float(res["log_loss"])
                    worst["acc"] = max(worst["acc"], abs(ga - acc))
                    worst["ll"] = max(worst["ll"], abs(gl - ll))
                    check(abs(ga - acc) <= 1e-4 + 1.5 / te.sum(), "S1 logistic %s fold %d accuracy %r vs sklearn %r" % (strategy, f, ga, acc))
                    check(abs(gl - ll) <= 1e-4 or abs(gl - ll) / abs(ll) <= 1e-3, "S1 logistic %s fold %d log_loss %r vs sklearn %r" % (strategy, f, gl, ll))
    print("S1: worst relative error of the linear metrics %.3g, worst |accuracy| gap %.3g, worst |log_loss| gap %.3g" % (worst["reg"], worst["acc"], worst["ll"]))


GATES = {"--assign": gate_assign, "--invariants": gate_invariants, "--e2e": gate_e2e, "--threads": gate_threads, "--sklearn": gate_sklearn}


def main():
    want = [a for a in sys.argv[1:] if a in GATES or a == "--all"]
    if not want:
        print(__doc__)
        return 2
    quick = "--quick" in sys.argv
    names = list(GATES) if "--all" in want else want
    with tempfile.TemporaryDirectory() as tmp:
        WORK[0] = tmp
        for g in names:
            GATES[g](tmp, quick)
    if FAILS:
        print("%d mismatches" % len(FAILS))
        return 1
    print("oracle_split: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

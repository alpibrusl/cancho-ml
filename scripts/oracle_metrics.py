#!/usr/bin/env python3
"""Independent oracles for lib/metrics (docs/metrics.md sections 1 to 3).

Nothing here imports or calls the cancho code except through the compiled driver `build/metrics-check`; every rule is re-implemented
from the written spec in Python (own SplitMix64 on Python ints, own rejection sampling, numpy performing the ordered sums,
`fractions.Fraction` for the exact AUC), and the bits are compared.

    python3 scripts/oracle_metrics.py --splits      S1: train/test, k-fold, stratified against the spec, whole vectors
    python3 scripts/oracle_metrics.py --draw        S3: the bounded draw (threshold and sequence) for many moduli
    python3 scripts/oracle_metrics.py --ratio       the correctly rounded division against Python's int / int
    python3 scripts/oracle_metrics.py --metrics     M1: accuracy, precision, recall, F1 against Python a / b
    python3 scripts/oracle_metrics.py --auc         M2: AUC against Fraction from scipy midranks
    python3 scripts/oracle_metrics.py --sums        M3: rmse, mae, r2, logloss against numpy in the written order
    python3 scripts/oracle_metrics.py --sklearn     R1: all against sklearn.metrics with the stated bounds
    python3 scripts/oracle_metrics.py --multiclass  MC1 to MC4: the multiclass metrics of docs/multiclass.md section 4 against Python integers, Fractions, scipy midranks and numpy
    python3 scripts/oracle_metrics.py --multiclass-sklearn   MC5: the same against sklearn.metrics with bounds fixed in advance
    python3 scripts/oracle_metrics.py --all
Exit 1 on any mismatch. METRICS_CHECK names the driver."""
import math, os, subprocess, sys, pathlib
from fractions import Fraction
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = os.environ.get("METRICS_CHECK", str(ROOT / "build" / "metrics-check"))
M64 = (1 << 64) - 1
GOLD = 0x9E3779B97F4A7C15
TOP = (1 << 63) - 1
U = 2.0 ** -53
FAILS = []


# ---------------------------------------------------------------- the driver
def run(*args):
    r = subprocess.run([BIN] + [str(a) for a in args], capture_output=True, text=True, timeout=600)
    out = {}
    for line in r.stdout.splitlines():
        k, _, v = line.partition(" ")
        out[k] = v
    out["_exit"] = r.returncode
    out["_stderr"] = r.stderr
    return out


def check(ok, what):
    if not ok:
        FAILS.append(what)
        print("MISMATCH", what)
    return ok


# ---------------------------------------------------------------- SplitMix64 on Python ints (docs/rng.md, written again)
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
        """docs/metrics.md 2.1: rejection on the top 63 bits"""
        assert 2 <= m <= (1 << 63) - 1
        self.rejected = getattr(self, 'rejected', 0)
        r = (1 << 63) % m
        while True:
            v = self.draw() >> 1
            if v < (1 << 63) - r:
                return v % m
            self.rejected += 1


def shuffle(a, st):
    for i in range(len(a) - 1, 0, -1):
        j = st.bounded(i + 1)
        a[i], a[j] = a[j], a[i]


def py_train_test(n, n_test, seed, rep):
    perm = list(range(n))
    shuffle(perm, Stream(stream(seed, 4, rep)))
    out = [0] * n
    for p in range(n_test):
        out[perm[p]] = 1
    return out


def py_kfold(n, k, seed, rep):
    perm = list(range(n))
    shuffle(perm, Stream(stream(seed, 5, rep)))
    out = [0] * n
    q, r = divmod(n, k)
    p = 0
    for f in range(k):
        size = q + (1 if f < r else 0)
        for _ in range(size):
            out[perm[p]] = f
            p += 1
    return out


def py_strat(labels, k, seed, rep):
    vals = sorted(set(labels))
    assert len(vals) == 2
    L = []
    for c, v in enumerate(vals):
        members = [i for i, x in enumerate(labels) if x == v]
        shuffle(members, Stream(stream(seed, 6, 2 * rep + c)))
        L += members
    out = [0] * len(labels)
    for t, row in enumerate(L):
        out[row] = t % k
    return out


# ---------------------------------------------------------------- generated inputs (tools/metrics-check/gen.cho, written again)
def seq(seed, n):
    """mix(seed + (i + 1) * gold) for i < n, as uint64"""
    i = np.arange(1, n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed & M64) + i * np.uint64(GOLD)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def element(seed, n):
    z = seq(seed, n)
    return ((z >> np.uint64(11)).astype(np.float64)) / 4503599627370496.0 - 1.0


def gen_labels(n, seed, pm):
    d = (seq(seed, n) >> np.uint64(1)) % np.uint64(1000)
    return (d < np.uint64(pm)).astype(np.int64)


EXTREME = np.array([0.0, 1.0, 2.0 ** -52, 1.0 - 2.0 ** -53, 1e-300, 5e-324, 0.5, 1.0 - 2.0 ** -52])
SIGNED = np.array([-0.0, 0.0, 1.0, -1.0])


def gen_floats(mode, n, seed, seed2=0):
    e = element(seed, n)
    if mode == 0:
        return (e + 1.0) * 0.5
    if mode == 1:
        return np.trunc((e + 1.0) * 0.5 * 16.0) / 16.0
    if mode == 2:
        return np.full(n, 0.5)
    d = seq(seed, n) >> np.uint64(1)
    if mode == 3:
        return EXTREME[(d % np.uint64(8)).astype(np.int64)]
    if mode == 4:
        k = (d % np.uint64(41)).astype(np.int64) - 20
        return e * np.ldexp(1.0, k)
    if mode == 5:
        return SIGNED[(d % np.uint64(4)).astype(np.int64)]
    if mode == 6:
        return e
    if mode == 10:
        return (e + 1.0) * 0.5 + 0.0625 * element(seed2, n)
    raise ValueError(mode)


def spec_int(a):
    return "L:" + ",".join(str(int(x)) for x in a)


def spec_hex(a):
    return "H:" + ",".join("%016x" % int(x) for x in np.asarray(a, dtype=np.float64).view(np.uint64))


def bits(x):
    return "%016x" % int(np.float64(x).view(np.uint64))


def unbits(h):
    return float(np.uint64(int(h, 16)).view(np.float64))


# ---------------------------------------------------------------- the written order of a decisive sum (docs/metrics.md 2.4)
K = 256
LANES = 4


def tree(p):
    while len(p) > 1:
        m = len(p)
        h = m // 2
        q = p[0:2 * h:2] + p[1:2 * h:2]
        if m % 2:
            q = np.concatenate([q, p[m - 1:m]])
        p = q
    return p[0]


def ordered_sum(t):
    t = np.asarray(t, dtype=np.float64)
    n = len(t)
    assert n >= 1
    nb = -(-n // K)
    pad = np.zeros(nb * K)
    pad[:n] = t          # zero padding: adding +0.0 to a lane that is not -0.0 changes nothing
    P = pad.reshape(nb, K // LANES, LANES)
    lanes = np.zeros((nb, LANES))
    for j in range(K // LANES):
        lanes = lanes + P[:, j, :]
    part = (lanes[:, 0] + lanes[:, 1]) + (lanes[:, 2] + lanes[:, 3])
    return tree(part)


# ---------------------------------------------------------------- S1
def case_tag(c):
    return " ".join(str(x) for x in c)


def oracle_splits():
    cases = 0
    sizes = [2, 3, 4, 5, 7, 10, 16, 17, 31, 32, 33, 64, 100, 255, 256, 257, 1000, 1023, 1024, 1025, 4097]
    seeds = [0, 1, 42, 1234567, -1, 2 ** 62, -(2 ** 63)]
    for n in sizes:
        for seed in seeds[:4] if n > 300 else seeds:
            for rep in (0, 1, 5):
                for n_test in sorted({1, n // 2, n - 1}):
                    if 1 <= n_test <= n - 1:
                        got = run("split", "train_test", n, n_test, seed, rep, 100000, "p")
                        want = py_train_test(n, n_test, seed, rep)
                        cases += 1
                        check(got.get("assign", "").split() == [str(x) for x in want], "train_test " + case_tag((n, n_test, seed, rep)))
                for k in sorted({2, 3, 5, 10, n}):
                    if 2 <= k <= n:
                        got = run("split", "kfold", n, k, seed, rep, 100000, "p")
                        want = py_kfold(n, k, seed, rep)
                        cases += 1
                        check(got.get("assign", "").split() == [str(x) for x in want], "kfold " + case_tag((n, k, seed, rep)))
    for n, pm in ((10, 500), (20, 300), (100, 100), (257, 500), (1000, 17), (1000, 900)):
        for seed in (0, 7, 99):
            labels = gen_labels(n, 5 + seed, pm)
            if len(set(labels.tolist())) < 2:
                continue
            for k in (2, 3, 5, 7, 10):
                m = min(np.bincount(labels))
                if m < k:
                    continue
                for rep in (0, 3):
                    got = run("split", "strat", "GL:%d:%d:%d" % (n, 5 + seed, pm), k, seed, rep, 100000, "p")
                    want = py_strat(labels.tolist(), k, seed, rep)
                    cases += 1
                    check(got.get("assign", "").split() == [str(x) for x in want], "strat " + case_tag((n, pm, seed, k, rep)))
    # labels other than 0 and 1: the smaller value is class 0
    lab = [5, 5, 9, 9, 5, 9, 5, 9, 9, 5, 5, 9]
    for k in (2, 3):
        got = run("split", "strat", spec_int(lab), k, 3, 0, 100, "p")
        cases += 1
        check(got.get("assign", "").split() == [str(x) for x in py_strat(lab, k, 3, 0)], "strat labels 5/9 k=%d" % k)
    lab = [-3, 4, 4, -3, 4, -3, -3, 4]
    got = run("split", "strat", spec_int(lab), 2, 11, 1, 100, "p")
    cases += 1
    check(got.get("assign", "").split() == [str(x) for x in py_strat(lab, 2, 11, 1)], "strat labels -3/4")
    # the three vectors written into docs/metrics.md section 5
    for name, want in REFERENCE_VECTORS.items():
        kind, args = name[0], name[1:]
        if kind == "strat":
            got = run("split", "strat", args[0], args[1], args[2], args[3], 1000, "p")
            labs = [int(x) for x in args[0][2:].split(",")]
            cases += 1
            check(got.get("assign", "").split() == [str(x) for x in want] == [str(x) for x in py_strat(labs, args[1], args[2], args[3])], "reference vector %s" % (name,))
            continue
        if kind == "kfold":
            got = run("split", "kfold", *args, 1000, "p")
            exp = py_kfold(args[0], args[1], args[2], args[3])
        elif kind == "train_test":
            got = run("split", "train_test", *args, 1000, "p")
            exp = py_train_test(args[0], args[1], args[2], args[3])
        cases += 1
        check(got.get("assign", "").split() == [str(x) for x in want] == [str(x) for x in exp], "reference vector %s" % (name,))
    # the property gate on the Python side as well: the written vectors are what the spec says, not only what the code says
    print("splits: %d cases against the Python spec" % cases)


# the reference vectors of docs/metrics.md (hand-checkable sizes); recomputed by the Python functions above when the stream kinds were renumbered to 4, 5, 6 (docs/rng.md), compared here with both sides
REFERENCE_VECTORS = {
    ("kfold", 10, 3, 42, 0): [0, 0, 2, 2, 0, 2, 1, 1, 0, 1],
    ("train_test", 10, 3, 42, 0): [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
    ("strat", "L:0,0,0,0,0,0,1,1,1,1", 3, 7, 0): [1, 2, 0, 2, 1, 0, 0, 2, 1, 0],
}


# ---------------------------------------------------------------- S3
def oracle_draw():
    mods = list(range(2, 70)) + [100, 255, 256, 257, 1000, 65535, 65536, 65537, 2 ** 31 - 1, 2 ** 31, 2 ** 31 + 1, 2 ** 32 + 7,
                                 3 * 2 ** 40 + 1, 2 ** 61 - 1, 2 ** 61 + 5, 2 ** 62 - 1, 2 ** 62]
    seeds = [0, 1, 2 ** 62, -5]
    bad = 0
    n = 0
    for m in mods:
        for s in seeds:
            st = Stream(s)
            # the driver prints the values of 40 successive calls: the Python stream is advanced by the draws each call used
            want = []
            for _ in range(40):
                want.append(st.bounded(m))
            out = subprocess.run([BIN, "bounded", str(s), str(m), "40"], capture_output=True, text=True).stdout.split()
            n += 1
            if out != [str(x) for x in want]:
                bad += 1
                check(False, "bounded seed=%d m=%d" % (s, m))
    # a modulus whose rejection is frequent: a threshold off by one or a plain modulus changes the sequence here
    st_cases = [(s, m) for m in (2 ** 62 + 1, 3 * 2 ** 61, 5 * 2 ** 60 + 3, 2 ** 63 - 1, 2 ** 63 - 2 ** 40) for s in range(30)]
    rej = draws = 0
    for s, m in st_cases:
        out = subprocess.run([BIN, "bounded", str(s), str(m), "60"], capture_output=True, text=True).stdout.split()
        st = Stream(s)
        want = [str(st.bounded(m)) for _ in range(60)]
        rej += st.rejected
        draws += 60 + st.rejected
        n += 1
        check(out == want, "bounded(heavy rejection) seed=%d m=%d" % (s, m))
    # the threshold itself: the first draw is exactly the largest accepted value, then exactly one above it (rejected)
    m = (1 << 63) - 1
    for s0, want_rej in ((7257538407534371759, 0), (5697289922173604375, 1)):
        st = Stream(s0)
        v = st.bounded(m)
        out = subprocess.run([BIN, "bounded", str(s0), str(m), "1"], capture_output=True, text=True).stdout.split()
        n += 1
        check(st.rejected == want_rej and out == [str(v)], "bounded at the threshold s0=%d: python rejected %d, driver %s, python %d" % (s0, st.rejected, out, v))
    check(rej > 100, "the heavy-rejection moduli did not reject often enough to test the rule (%d)" % rej)
    print("draw: %d (seed, modulus) sequences; the last %d have moduli above 2^62 where %d of %d attempts were rejected" % (n, len(st_cases), rej, draws))


# ---------------------------------------------------------------- ratio
def oracle_ratio():
    import random
    rnd = random.Random(7)
    pairs = [(0, 1), (1, 1), (1, 3), (2, 3), (1, 7), (5, 8), (1, 2 ** 61), (2 ** 61 - 1, 2 ** 61), (2 ** 61 - 2, 2 ** 61 - 1)]
    for _ in range(3000):
        b = rnd.randrange(1, 1 << rnd.randrange(1, 62))
        a = rnd.randrange(0, b + 1)
        pairs.append((a, b))
    # exact halves: a / b = m / 2^53 midway cases (ties to even)
    for e in range(1, 20):
        b = 2 ** (53 + e)
        for a in (2 ** e + 1, 3 * 2 ** e + 1, (2 ** 53 + 1) * 2 ** (e - 1) if e > 0 else 1):
            if a <= b:
                pairs.append((a, b))
    bad = 0
    for a, b in pairs:
        if b >= 1 << 62:
            continue
        got = run("ratio", a, b).get("value")
        check(got == bits(a / b), "ratio %d/%d: got %s want %s" % (a, b, got, bits(a / b)))
    print("ratio: %d (a, b) pairs, bit equal to Python int / int" % len(pairs))


# ---------------------------------------------------------------- M1
def bin_counts(yt, yp, pos):
    tp = sum(1 for t, p in zip(yt, yp) if t == pos and p == pos)
    fp = sum(1 for t, p in zip(yt, yp) if t != pos and p == pos)
    fn = sum(1 for t, p in zip(yt, yp) if t == pos and p != pos)
    return tp, fp, fn


def oracle_metrics():
    n_cases = 0
    for n, s1, s2, pm1, pm2 in ((1, 1, 2, 500, 500), (2, 1, 2, 500, 500), (7, 3, 4, 400, 600), (100, 5, 6, 300, 300), (1000, 7, 8, 100, 150),
                                (4097, 9, 10, 500, 520), (50000, 11, 12, 20, 25)):
        yt = gen_labels(n, s1, pm1).tolist()
        yp = gen_labels(n, s2, pm2).tolist()
        for pos in (1, 0):
            tp, fp, fn = bin_counts(yt, yp, pos)
            exp = {"accuracy": (sum(1 for a, b in zip(yt, yp) if a == b), n)}
            exp["precision"] = (tp, tp + fp)
            exp["recall"] = (tp, tp + fn)
            exp["f1"] = (2 * tp, 2 * tp + fp + fn)
            for name, (a, b) in exp.items():
                got = run("metric", name, "GL:%d:%d:%d" % (n, s1, pm1), "GL:%d:%d:%d" % (n, s2, pm2), pos, 1000000, 1)
                n_cases += 1
                if b == 0:
                    check(got.get("refused") == "ml.metric.undefined", "%s n=%d pos=%d: expected undefined, got %s" % (name, n, pos, got))
                else:
                    check(got.get("value") == bits(a / b), "%s n=%d pos=%d: got %s want %s" % (name, n, pos, got.get("value"), bits(a / b)))
    # explicit edge cases (also the fixtures of T1, scripts/gate_metrics.py)
    edge = [
        ("accuracy", [1, 0, 2, 2], [1, 0, 2, 1], 1, (3, 4)),                 # accuracy needs no binary labels
        ("precision", [1, 1, 0, 0], [1, 0, 1, 0], 1, (1, 2)),
        ("recall", [1, 1, 0, 0], [1, 0, 1, 0], 1, (1, 2)),
        ("f1", [1, 1, 0, 0], [1, 0, 1, 0], 1, (2, 4)),
        ("precision", [1, 1, 0, 0], [1, 0, 1, 0], 0, (1, 2)),                # the other class as positive
        ("recall", [1, 1, 1, 0], [1, 1, 1, 0], 0, (1, 1)),
        ("f1", [0, 0, 0], [0, 0, 0], 1, None),                               # nothing positive anywhere: 0 / 0
        ("precision", [0, 1], [0, 0], 1, None),                              # nothing predicted positive
        ("recall", [0, 0], [0, 1], 1, None),                                 # nothing truly positive
        ("recall", [0, 1, 1], [0, 1, 0], 1, (1, 2)),
        ("f1", [-5, 7, 7, -5], [7, 7, -5, -5], 7, (2, 4)),                   # labels other than 0 and 1
        ("f1", [-5, 7, 7, -5], [7, 7, -5, -5], -5, (2, 4)),
    ]
    for name, yt, yp, pos, ab in edge:
        got = run("metric", name, spec_int(yt), spec_int(yp), pos, 1000, 1)
        n_cases += 1
        if ab is None:
            check(got.get("refused") == "ml.metric.undefined", "edge %s %s %s: %s" % (name, yt, yp, got))
        else:
            check(got.get("value") == bits(ab[0] / ab[1]), "edge %s %s %s pos=%d: %s want %s" % (name, yt, yp, pos, got, ab))
    # a wrong positive class: a third value is refused, and so is a pos that is in neither array when two labels are present
    got = run("metric", "f1", spec_int([0, 1, 2]), spec_int([0, 1, 1]), 1, 1000, 1)
    check(got.get("refused") == "ml.label.not-binary", "f1 three labels")
    got = run("metric", "recall", spec_int([0, 1, 0]), spec_int([0, 1, 1]), 5, 1000, 1)
    check(got.get("refused") == "ml.label.not-binary", "recall with a positive label that is not a label")
    print("metrics (counts): %d cases bit equal to Python a / b" % n_cases)


# ---------------------------------------------------------------- M2
def exact_auc(yt, score, pos):
    """float(Fraction(2U, 2PN)) with the midranks of scipy.stats.rankdata (an independent route to the same number)"""
    from scipy.stats import rankdata
    yt = np.asarray(yt)
    sc = np.asarray(score, dtype=np.float64)
    isp = yt == pos
    P = int(isp.sum())
    N = len(yt) - P
    r2 = np.rint(rankdata(sc, method="average") * 2).astype(np.int64)    # doubled midranks: exact ints
    two_rplus = int(r2[isp].sum())
    two_u = two_rplus - P * (P + 1)
    return Fraction(two_u, 2 * P * N), P, N


def auc_cases():
    out = []
    for n in (2, 3, 10, 100, 1000, 4097, 20000):
        for mode in (0, 1, 2, 5, 6, 3):
            out.append((n, mode, 31 + n % 7, 17 + mode, 500))
        out.append((n, 0, 5, 9, 30))       # few positives
        out.append((n, 1, 5, 9, 970))      # few negatives, heavy ties
    return out


def oracle_auc():
    n_cases = 0
    for n, mode, ls, ss, pm in auc_cases():
        y = gen_labels(n, ls, pm)
        s = gen_floats(mode, n, ss)
        if y.min() == y.max():
            got = run("metric", "auc", "GL:%d:%d:%d" % (n, ls, pm), "GF:%d:%d:%d:0" % (mode, n, ss), 1, 1000000, 1)
            check(got.get("refused") == "ml.label.single-class", "auc single class n=%d" % n)
            n_cases += 1
            continue
        for pos in (1, 0):
            f, P, N = exact_auc(y, s, pos)
            got = run("metric", "auc", "GL:%d:%d:%d" % (n, ls, pm), "GF:%d:%d:%d:0" % (mode, n, ss), pos, 1000000, 1)
            n_cases += 1
            check(got.get("value") == bits(float(f)), "auc n=%d mode=%d pos=%d: got %s want %s" % (n, mode, pos, got.get("value"), bits(float(f))))
    # explicit tie cases with the answer written by hand
    explicit = [
        ([1, 0], [0.5, 0.5], Fraction(1, 2)),                          # one tie: half a point
        ([1, 0], [0.7, 0.2], Fraction(1)),
        ([1, 0], [0.2, 0.7], Fraction(0)),
        ([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5], Fraction(1, 2)),         # all tied
        ([1, 0, 1, 0], [0.9, 0.8, 0.8, 0.1], Fraction(7, 8)),          # a tie in the middle: 3 + 0.5 + 0.5... of 4
        ([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], Fraction(1, 2)),          # single positive, all tied
        ([0, 0, 0, 1], [-0.0, 0.0, 0.0, -0.0], Fraction(1, 2)),        # -0.0 equals +0.0
    ]
    for yt, sc, want in explicit:
        got = run("metric", "auc", spec_int(yt), spec_hex(sc), 1, 1000, 1)
        n_cases += 1
        check(got.get("value") == bits(float(want)) , "auc explicit %s %s: got %s want %s" % (yt, sc, got, want))
        f, _, _ = exact_auc(yt, sc, 1)
        check(f == want, "auc explicit table entry is wrong for %s %s: scipy says %s, table says %s" % (yt, sc, f, want))
    # extreme magnitudes and signs, subnormals, huge
    for sc in ([1e308, -1e308, 5e-324, -5e-324, 0.0], [5e-324, 0.0, -5e-324, 1.7976931348623157e308, -1.7976931348623157e308]):
        yt = [1, 0, 1, 0, 0] if len(sc) == 5 else [0, 1, 0, 1, 1]
        got = run("metric", "auc", spec_int(yt), spec_hex(sc), 1, 1000, 1)
        f, _, _ = exact_auc(yt, sc, 1)
        n_cases += 1
        check(got.get("value") == bits(float(f)), "auc extremes %s" % sc)
    print("auc: %d cases bit equal to float(Fraction) from scipy midranks" % n_cases)


# ---------------------------------------------------------------- M3
def py_sq_terms(y, p):
    e = y - p
    return e * e


def sums_cases():
    cs = []
    for n in (1, 2, 3, 4, 5, 7, 8, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4097, 65535, 65536, 65537, 100003):
        cs.append((n, 4, 4, 21 + n % 5, 40 + n % 3))        # wide magnitude both
        cs.append((n, 0, 10, 3, 5))                          # y uniform, p = y + noise: the informative pair
    return cs


def oracle_sums():
    n_cases = 0
    for n, ym, pm_, s1, s2 in sums_cases():
        if pm_ == 10:
            y = gen_floats(0, n, s1)
            p = gen_floats(10, n, s1, s2)
            yspec, pspec = "GF:0:%d:%d:0" % (n, s1), "GF:10:%d:%d:%d" % (n, s1, s2)
        else:
            y = gen_floats(ym, n, s1)
            p = gen_floats(pm_, n, s2)
            yspec, pspec = "GF:%d:%d:%d:0" % (ym, n, s1), "GF:%d:%d:%d:0" % (pm_, n, s2)
        sse = ordered_sum(py_sq_terms(y, p))
        want = {"rmse": float(np.sqrt(sse / float(n))), "mae": float(ordered_sum(np.abs(y - p)) / float(n))}
        if np.all(y == y[0]):
            want["r2"] = None
        else:
            mean = ordered_sum(y) / float(n)
            sst = ordered_sum(py_sq_terms(y, np.full(n, mean)))
            want["r2"] = None if sst == 0.0 else float(1.0 - sse / sst)
        for name, w in want.items():
            got = run("metric", name, yspec, pspec, 0, 1000000, 1)
            n_cases += 1
            if w is None or not math.isfinite(w):
                check(got.get("refused") == "ml.metric.undefined", "%s n=%d expected undefined, got %s" % (name, n, got))
            else:
                check(got.get("value") == bits(w), "%s n=%d ym=%d: got %s want %s" % (name, n, ym, got.get("value"), bits(w)))
    # sensitivity: sizes 1..70 and the neighbourhoods of block edges, terms of one magnitude (mode 6 against mode 6), so that a
    # change of lane or of association shows in the last bit (wide-magnitude data hides it: one term dominates the sum)
    sens = list(range(1, 71)) + list(range(250, 262)) + list(range(508, 520)) + [1020, 1023, 1024, 1027, 2047, 2051]
    for n in sens:
        for s1 in (3, 4, 5):
            y = gen_floats(6, n, s1)
            p = gen_floats(6, n, s1 + 50)
            yspec, pspec = "GF:6:%d:%d:0" % (n, s1), "GF:6:%d:%d:0" % (n, s1 + 50)
            sse = ordered_sum(py_sq_terms(y, p))
            want = {"rmse": float(np.sqrt(sse / float(n))), "mae": float(ordered_sum(np.abs(y - p)) / float(n))}
            if n > 1:
                mean = ordered_sum(y) / float(n)
                want["r2"] = float(1.0 - sse / ordered_sum(py_sq_terms(y, np.full(n, mean))))
            for name, w in want.items():
                got = run("metric", name, yspec, pspec, 0, 1000000, 1)
                n_cases += 1
                check(got.get("value") == bits(w), "sensitivity %s n=%d seed=%d: got %s want %s" % (name, n, s1, got.get("value"), bits(w)))
    # a ratio that overflows although SSE and SST are finite: undefined
    got = run("metric", "r2", "H:0000000000000000,1eb67e9c127b6e74", "H:5f138d352e5096af,5f138d352e5096af", 0, 100, 1)
    n_cases += 1
    check(got.get("refused") == "ml.metric.undefined", "r2 overflow: %s" % got)
    # log-loss: the printed terms are the library's own (std.math.log), the ordered sum is ours
    n_exact = n_terms = worst = 0
    for n, mode, ls, ss, pm in ((1, 0, 1, 2, 500), (3, 0, 1, 2, 500), (255, 0, 4, 5, 500), (256, 0, 4, 5, 500), (257, 0, 4, 5, 500),
                                (1025, 1, 6, 7, 300), (3000, 3, 8, 9, 500), (5000, 0, 10, 11, 100), (4099, 3, 12, 13, 800)):
        y = gen_labels(n, ls, pm)
        p = gen_floats(mode, n, ss)
        yspec, pspec = "GL:%d:%d:%d" % (n, ls, pm), "GF:%d:%d:%d:0" % (mode, n, ss)
        t = run("terms", yspec, pspec, 1)
        terms = np.array([unbits(h) for h in t["terms"].split()])
        # the terms: within 2 ulp of Python's, same rule (clip, 1 - p, -log)
        eps = 2.0 ** -52
        x = np.where(y == 1, p, 1.0 - p)
        x = np.minimum(np.maximum(x, eps), 1.0 - eps)
        py_terms = np.array([-math.log(v) for v in x])
        for a, b in zip(terms, py_terms):
            n_terms += 1
            if a == b:
                n_exact += 1
            else:
                ulp = abs(a - b) / math.ulp(b)
                worst = max(worst, ulp)
                check(ulp <= 2, "logloss term differs by %.1f ulp" % ulp)
        got = run("metric", "logloss", yspec, pspec, 1, 1000000, 1)
        n_cases += 1
        check(got.get("value") == bits(float(ordered_sum(terms) / float(n))), "logloss n=%d mode=%d: not the ordered sum of its own terms" % (n, mode))
    print("sums: %d cases bit equal to numpy in the written order; log-loss terms: %d of %d bit equal to Python's math.log terms, worst %.2f ulp"
          % (n_cases, n_exact, n_terms, worst))


# ---------------------------------------------------------------- R1
def ulps_rel(a, b):
    return abs(a - b) / (U * max(abs(a), abs(b), 1e-300) * 2)


def oracle_sklearn():
    import sklearn
    from sklearn import metrics as sm
    import warnings
    warnings.simplefilter("ignore")
    print("sklearn %s, numpy %s" % (sklearn.__version__, np.__version__))
    rows = []

    def report(name, case, mine, ref, bound_rel, kind):
        d = abs(mine - ref)
        rel = d / max(abs(ref), 1e-300)
        eq = mine == ref
        # R-squared is 1 - SSE / SST: two relative errors of the sums, scaled by the ratio, so the bound is absolute
        ok = eq or d <= 2 * bound_rel * U * max(1.0, 1.0 - ref) if kind == "abs" else eq or rel <= bound_rel * U
        check(ok, "sklearn %s %s: mine %r ref %r rel %.3g (bound %g u)" % (name, case, mine, ref, rel, bound_rel))
        rows.append((name, case, eq, rel / U if not eq else 0.0))

    def val(*a):
        r = run(*a)
        assert "value" in r, (a, r)
        return unbits(r["value"])

    BOUND = 130      # docs/metrics.md section 3: (88 + 32 + 10) u
    for n in (2, 10, 257, 5000, 100000):
        for pm1, pm2, lab in ((500, 500, "balanced"), (50, 80, "rare")):
            yt = gen_labels(n, 3, pm1)
            yp = gen_labels(n, 4, pm2)
            if yt.min() == yt.max() or yp.min() == yp.max():
                continue
            ys, ps = "GL:%d:3:%d" % (n, pm1), "GL:%d:4:%d" % (n, pm2)
            for name, f in (("accuracy", sm.accuracy_score), ("precision", sm.precision_score), ("recall", sm.recall_score), ("f1", sm.f1_score)):
                ref = float(f(yt, yp)) if name == "accuracy" else float(f(yt, yp, pos_label=1))
                report(name, "n=%d %s" % (n, lab), val("metric", name, ys, ps, 1, 10000000, 1), ref, BOUND, "rel")
    for n in (2, 10, 257, 5000, 100000):
        for mode, label in ((0, "uniform"), (1, "ties"), (2, "all tied"), (5, "signed zeros"), (6, "negative")):
            for pm in (500, 20):
                y = gen_labels(n, 3, pm)
                if y.min() == y.max():
                    continue
                s = gen_floats(mode, n, 8)
                ref = float(sm.roc_auc_score(y, s))
                report("auc", "n=%d %s pm=%d" % (n, label, pm), val("metric", "auc", "GL:%d:3:%d" % (n, pm), "GF:%d:%d:8:0" % (mode, n), 1, 10000000, 1), ref, BOUND, "rel")
    for n in (3, 10, 257, 5000, 100000):
        for ym, pmode, label in ((4, 4, "wide"), (0, 10, "informative"), (6, 6, "uniform"), (6, 2, "constant prediction")):
            y = gen_floats(ym, n, 21)
            p = gen_floats(pmode, n, 22, 23) if pmode != 10 else gen_floats(10, n, 21, 23)
            yspec = "GF:%d:%d:21:0" % (ym, n)
            pspec = "GF:%d:%d:22:23" % (pmode, n) if pmode != 10 else "GF:10:%d:21:23" % n
            report("rmse", "n=%d %s" % (n, label), val("metric", "rmse", yspec, pspec, 0, 10000000, 1), float(sm.root_mean_squared_error(y, p)), BOUND, "rel")
            report("mae", "n=%d %s" % (n, label), val("metric", "mae", yspec, pspec, 0, 10000000, 1), float(sm.mean_absolute_error(y, p)), BOUND, "rel")
            report("r2", "n=%d %s" % (n, label), val("metric", "r2", yspec, pspec, 0, 10000000, 1), float(sm.r2_score(y, p)), BOUND, "abs")
    for n in (2, 10, 257, 5000, 100000):
        for mode, label in ((0, "uniform"), (1, "ties"), (3, "extreme"), (2, "all equal")):
            for pm in (500, 100):
                y = gen_labels(n, 3, pm)
                if y.min() == y.max():
                    continue
                p = gen_floats(mode, n, 8)
                ref = float(sm.log_loss(y, p, labels=[0, 1]))
                report("logloss", "n=%d %s pm=%d" % (n, label, pm), val("metric", "logloss", "GL:%d:3:%d" % (n, pm), "GF:%d:%d:8:0" % (mode, n), 1, 10000000, 1), ref, BOUND + 2, "rel")
    # what sklearn does where this library refuses (printed for docs/metrics.md section 5.2; the refusals themselves are gate T1)
    def what(f):
        try:
            v = f()
            return "returns %r" % (float(v),)
        except Exception as ex:
            return "raises %s" % type(ex).__name__
    print("where this library refuses, sklearn:")
    for label, f in (
            ("roc_auc_score, one class", lambda: sm.roc_auc_score([1, 1, 1], [0.1, 0.2, 0.3])),
            ("precision_score, nothing predicted positive", lambda: sm.precision_score([0, 1], [0, 0])),
            ("recall_score, nothing truly positive", lambda: sm.recall_score([0, 0], [0, 1])),
            ("f1_score, nothing positive", lambda: sm.f1_score([0, 0, 0], [0, 0, 0])),
            ("r2_score, constant target, wrong prediction", lambda: sm.r2_score([1.0, 1.0, 1.0], [0.5, 0.5, 0.5])),
            ("r2_score, constant target, perfect prediction", lambda: sm.r2_score([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])),
            ("accuracy_score, empty", lambda: sm.accuracy_score([], [])),
            ("root_mean_squared_error, empty", lambda: sm.root_mean_squared_error([], [])),
            ("log_loss, probability 1.5", lambda: sm.log_loss([1, 0], [1.5, 0.5], labels=[0, 1])),
            ("roc_auc_score, a NaN score", lambda: sm.roc_auc_score([1, 0], [float("nan"), 0.5])),
            ("accuracy_score, lengths 3 and 4", lambda: sm.accuracy_score([1, 0, 1], [1, 0, 1, 0]))):
        print("  %-48s %s" % (label, what(f)))
    # the refusals sklearn turns into numbers: 0.0 with a warning, nan, or a ValueError
    refusals = [
        ("precision", "no predicted positive", sm.precision_score([0, 1], [0, 0], zero_division=np.nan)),
        ("recall", "no true positive", sm.recall_score([0, 0], [0, 1], zero_division=np.nan)),
        ("f1", "nothing positive", sm.f1_score([0, 0, 0], [0, 0, 0], zero_division=np.nan)),
    ]
    for name, why, ref in refusals:
        check(isinstance(ref, float) and math.isnan(ref), "sklearn zero_division=nan for %s (%s)" % (name, why))
    # the table
    worst = {}
    exact = {}
    for name, case, eq, rel in rows:
        worst[name] = max(worst.get(name, 0.0), rel)
        e = exact.setdefault(name, [0, 0])
        e[0] += 1 if eq else 0
        e[1] += 1
    print("%-10s %-14s %-24s" % ("metric", "bit equal", "worst relative gap if not (in u = 2^-53)"))
    for name in worst:
        print("%-10s %3d of %-6d %.2f" % (name, exact[name][0], exact[name][1], worst[name]))



# ---------------------------------------------------------------- MC1 to MC5 (docs/multiclass.md section 4, stage A)
def run_lines(*args):
    """the whole stdout of the driver, line by line, and its exit code (the multiclass commands print one line per class)"""
    r = subprocess.run([BIN] + [str(a) for a in args], capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout.splitlines(), r.stderr


def gen_mlabels(n, seed, k):
    """label(seed, i) mod k: GM:n:seed:K of tools/metrics-check/gen.cho"""
    return ((seq(seed, n) >> np.uint64(1)) % np.uint64(k)).astype(np.int64)


def gen_probs(mode, n, k, seed):
    """GQ:mode:n:seed:K: n * k floats of mode `mode`, every row divided by its sequential sum (a zero sum gives 1 / k)"""
    raw = gen_floats(mode, n * k, seed).reshape(n, k)
    s = np.zeros(n)
    for j in range(k):
        s = s + raw[:, j]
    pos = s > 0.0
    return np.where(pos[:, None], raw / np.where(pos, s, 1.0)[:, None], 1.0 / k)


def mc_confusion(yt, yp, classes):
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    conf = [[0] * k for _ in range(k)]
    for t, p in zip(yt, yp):
        conf[idx[int(t)]][idx[int(p)]] += 1
    return conf


def mc_value(conf, which, c):
    """precision (1), recall (2), f1 (3) of class index c as Python's correctly rounded int / int, or None for 0 / 0"""
    k = len(conf)
    tp = conf[c][c]
    row = sum(conf[c])
    col = sum(conf[a][c] for a in range(k))
    num, den = {1: (tp, col), 2: (tp, row), 3: (2 * tp, row + col)}[which]
    return None if den == 0 else num / den


def mc_average(conf, which, weighted):
    """(value, classes left out) with the sums in the order of the classes; value None when no class has one"""
    k = len(conf)
    s = 0.0
    wsum = 0
    have = 0
    for c in range(k):
        v = mc_value(conf, which, c)
        if v is None:
            continue
        sup = sum(conf[c])
        if weighted:
            s = s + float(sup) * v
            wsum += sup
        else:
            s = s + v
        have += 1
    left = k - have
    if have == 0 or (weighted and wsum == 0):
        return None, left
    return (s / float(wsum) if weighted else s / float(have)), left


def mc_args(name, yt, second, classes, cap=1000000):
    return ("mc", name, yt, second, classes, cap)


def mc_check_all(tag, yspec, pspec, yt, yp, classes):
    """every confusion-based output of the driver for one case against the Python computation"""
    cspec = spec_int(classes)
    conf = mc_confusion(yt, yp, classes)
    k = len(classes)
    n = len(yt)
    code, lines, _ = run_lines(*mc_args("confusion", yspec, pspec, cspec))
    rows = [[int(x) for x in l.split()[2:]] for l in lines if l.startswith("row ")]
    check(code == 0 and rows == conf, "%s: confusion differs: %s vs %s" % (tag, rows, conf))
    got = run(*mc_args("accuracy", yspec, pspec, cspec))
    trace = sum(conf[i][i] for i in range(k))
    check(got.get("value") == bits(trace / n), "%s: accuracy %s want %s" % (tag, got.get("value"), bits(trace / n)))
    # the same number from the binary-agnostic accuracy of metrics.classify (any labels): the matrix and the direct count agree
    got2 = run("metric", "accuracy", yspec, pspec, 0, 1000000, 1)
    check(got2.get("value") == got.get("value"), "%s: accuracy of the matrix differs from classify.accuracy" % tag)
    cases = 2
    for which, name in ((1, "precision"), (2, "recall"), (3, "f1")):
        code, lines, _ = run_lines(*mc_args(name, yspec, pspec, cspec))
        per = {}
        for l in lines:
            f = l.split()
            if f[0] == "class":
                per[int(f[1])] = f[2]
        want = {c: ("undefined" if mc_value(conf, which, c) is None else bits(mc_value(conf, which, c))) for c in range(k)}
        check(code == 0 and per == want, "%s: per-class %s differs: %s vs %s" % (tag, name, per, want))
        for weighted, pre in ((False, "macro_"), (True, "weighted_")):
            v, left = mc_average(conf, which, weighted)
            code, lines, _ = run_lines(*mc_args(pre + name, yspec, pspec, cspec))
            d = dict(l.split(" ", 1) for l in lines)
            if v is None:
                check(d.get("refused") == "ml.metric.undefined", "%s: %s%s expected undefined, got %s" % (tag, pre, name, d))
            else:
                check(d.get("value") == bits(v) and d.get("left_out") == str(left), "%s: %s%s got %s want %s left %d" % (tag, pre, name, d, bits(v), left))
            cases += 1
        cases += 1
    return cases


def oracle_multiclass():
    n_cases = 0
    # MC1: generated labels, the classes 0 .. K-1
    for n, k, s1, s2 in ((1, 3, 1, 2), (2, 3, 1, 2), (7, 3, 3, 4), (100, 5, 5, 6), (1000, 7, 7, 8), (4097, 11, 9, 10), (20000, 64, 11, 12), (300, 2, 13, 14)):
        classes = list(range(k))
        yt = gen_mlabels(n, s1, k).tolist()
        yp = gen_mlabels(n, s2, k).tolist()
        n_cases += mc_check_all("GM n=%d K=%d" % (n, k), "GM:%d:%d:%d" % (n, s1, k), "GM:%d:%d:%d" % (n, s2, k), yt, yp, classes)
    # MC1: correlated predictions and labels that are not 0 .. K-1 (negative, large, gaps), generated here and passed as lists
    rs = np.random.RandomState(20261020)
    for n, classes in ((50, [-7, 0, 3]), (300, [-5, 2, 3, 100]), (2000, [-1000000007, -1, 0, 1, 9223372036854775806, 9223372036854775807]),
                       (1500, list(range(-4, 6))), (800, [4, 5])):
        k = len(classes)
        yt = [classes[i] for i in rs.randint(0, k, size=n)]
        flip = rs.random_sample(n) < 0.25
        other = [classes[i] for i in rs.randint(0, k, size=n)]
        yp = [o if f else t for t, o, f in zip(yt, other, flip)]
        n_cases += mc_check_all("explicit n=%d K=%d" % (n, k), spec_int(yt), spec_int(yp), yt, yp, classes)
    # MC1: the classes with no row, a class never predicted, nothing predicted right, one class everywhere
    edge = [
        ([5], [5], [3, 5, 8]),                                   # two classes absent: undefined, left out
        ([0, 1], [0, 0], [0, 1]),                                # class 1 never predicted: precision undefined, recall 0
        ([0, 0, 1, 1], [1, 1, 0, 0], [0, 1]),                    # nothing right: every value 0, F1 0 / 4
        ([2, 2, 2], [2, 2, 2], [1, 2, 3]),                       # one class: precision, recall, F1 of the other two undefined
        ([0, 1, 2], [0, 1, 2], [0, 1, 2]),                       # perfect
        ([0, 1, 2, 2], [2, 2, 2, 0], [0, 1, 2]),
    ]
    for yt, yp, classes in edge:
        n_cases += mc_check_all("edge %s %s" % (yt, yp), spec_int(yt), spec_int(yp), yt, yp, classes)
    # MC2: log-loss: the library's own terms are within 2 ulp of Python's, the sum is the written order
    n_terms = n_exact = 0
    worst = 0.0
    for n, k, mode, ls, ss in ((1, 3, 0, 1, 2), (3, 3, 0, 1, 2), (255, 4, 0, 4, 5), (256, 4, 0, 4, 5), (257, 4, 0, 4, 5), (1025, 7, 1, 6, 7), (3000, 5, 3, 8, 9), (5000, 10, 0, 10, 11), (4099, 64, 3, 12, 13)):
        y = gen_mlabels(n, ls, k)
        p = gen_probs(mode, n, k, ss)
        yspec, pspec, cspec = "GM:%d:%d:%d" % (n, ls, k), "GQ:%d:%d:%d:%d" % (mode, n, ss, k), spec_int(list(range(k)))
        code, lines, _ = run_lines("mc", "terms", yspec, pspec, cspec, 1000000)
        terms = np.array([unbits(h) for h in lines[0].split()[1:]])
        eps = 2.0 ** -52
        x = np.minimum(np.maximum(p[np.arange(n), y], eps), 1.0 - eps)
        for a, b in zip(terms, np.array([-math.log(v) for v in x])):
            n_terms += 1
            if a == b:
                n_exact += 1
            else:
                ulp = abs(a - b) / math.ulp(b)
                worst = max(worst, ulp)
                check(ulp <= 2, "multiclass logloss term differs by %.1f ulp" % ulp)
        got = run("mc", "logloss", yspec, pspec, cspec, 1000000)
        n_cases += 1
        check(got.get("value") == bits(float(ordered_sum(terms) / float(n))), "multiclass logloss n=%d K=%d mode=%d: not the ordered sum of its own terms" % (n, k, mode))
    # MC3: one-versus-rest AUC: each class against the rest is float(Fraction) from scipy midranks, the mean in the order of the classes
    for n, k, mode, ls, ss in ((2, 2, 0, 1, 2), (10, 3, 0, 3, 4), (100, 4, 1, 5, 6), (1000, 7, 0, 7, 8), (4097, 5, 2, 9, 10), (20000, 10, 0, 11, 12), (300, 64, 1, 13, 14)):
        y = gen_mlabels(n, ls, k)
        p = gen_probs(mode, n, k, ss)
        yspec, pspec, cspec = "GM:%d:%d:%d" % (n, ls, k), "GQ:%d:%d:%d:%d" % (mode, n, ss, k), spec_int(list(range(k)))
        code, lines, _ = run_lines("mc", "auc_ovr", yspec, pspec, cspec, 1000000)
        per = {}
        d = {}
        for l in lines:
            f = l.split()
            if f[0] == "class":
                per[int(f[1])] = f[2]
            else:
                d[f[0]] = f[1] if len(f) > 1 else ""
        want = {}
        s = 0.0
        have = 0
        for c in range(k):
            isp = y == c
            if isp.all() or not isp.any():
                want[c] = "undefined"
                continue
            f, _, _ = exact_auc(isp.astype(np.int64), p[:, c], 1)
            want[c] = bits(float(f))
            s = s + float(f)
            have += 1
        n_cases += 1
        check(per == want, "auc_ovr n=%d K=%d: per-class values differ" % (n, k))
        if have == 0:
            check(d.get("refused") == "ml.metric.undefined", "auc_ovr n=%d K=%d: expected undefined" % (n, k))
        else:
            check(d.get("value") == bits(s / float(have)) and d.get("left_out") == str(k - have), "auc_ovr n=%d K=%d: macro %s want %s" % (n, k, d, bits(s / float(have))))
    # MC3: explicit values written by hand (the tie rule is the binary one: a tie counts half)
    code, lines, _ = run_lines("mc", "auc_ovr", spec_int([0, 1, 2]), spec_hex([0.5, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.5]), spec_int([0, 1, 2]), 1000)
    d = dict(l.split(" ", 1) for l in lines if not l.startswith("class"))
    check(d.get("value") == bits(1.0) and d.get("left_out") == "0", "auc_ovr perfect: %s" % d)
    code, lines, _ = run_lines("mc", "auc_ovr", spec_int([0, 1]), spec_hex([0.5, 0.25, 0.25, 0.5, 0.0, 0.0]), spec_int([0, 1, 2]), 1000)
    check(code == 0 and "class 2 undefined" in lines, "auc_ovr: an absent class is undefined and left out")
    n_cases += 2
    print("multiclass: %d cases: confusion exact, per-class and averaged values bit equal to Python int / int, log-loss terms %d of %d bit equal to math.log (worst %.2f ulp) and summed in the written order, "
          "one-versus-rest AUC bit equal to float(Fraction) from scipy midranks" % (n_cases, n_exact, n_terms, worst))


# MC5: sklearn, with the bounds written here before the numbers were looked at: counts-based values and their averages 130 u (the bound of the binary metrics, docs/metrics.md section 3) + 20 u for the mean of up to 64
# values taken in another order, the one-versus-rest AUC 150 u, the log-loss 1e-12 absolute (sklearn renormalises the rows, ours sum to 1 within K 2^-52)
MC_BOUND_COUNTS = 150
MC_BOUND_AUC = 150
MC_LOGLOSS_ABS = 1e-12


def oracle_multiclass_sklearn():
    import sklearn
    from sklearn import metrics as sm
    import warnings
    warnings.simplefilter("ignore")
    print("sklearn %s, numpy %s (multiclass)" % (sklearn.__version__, np.__version__))
    worst = {}

    def note(name, mine, ref, bound_u, what):
        rel = abs(mine - ref) / max(abs(ref), 1e-300)
        ok = mine == ref or rel <= bound_u * U
        check(ok, "sklearn %s %s: mine %r ref %r rel %.3g u (bound %g u)" % (name, what, mine, ref, rel / U, bound_u))
        worst[name] = max(worst.get(name, 0.0), 0.0 if mine == ref else rel / U)

    for n, k, s1, s2 in ((10, 3, 3, 4), (257, 4, 5, 6), (5000, 7, 7, 8), (20000, 12, 9, 10), (4000, 64, 11, 12)):
        classes = list(range(k))
        yt = gen_mlabels(n, s1, k)
        yp = gen_mlabels(n, s2, k)
        # a mostly right prediction as well: the diagonal is dense
        ys, ps, cs = "GM:%d:%d:%d" % (n, s1, k), "GM:%d:%d:%d" % (n, s2, k), spec_int(classes)
        got = run("mc", "accuracy", ys, ps, cs, 10000000)
        note("accuracy", unbits(got["value"]), float(sm.accuracy_score(yt, yp)), MC_BOUND_COUNTS, "n=%d K=%d" % (n, k))
        pr, rc, f1, _ = sm.precision_recall_fscore_support(yt, yp, labels=classes, average=None, zero_division=np.nan)
        for name, ref in (("precision", pr), ("recall", rc), ("f1", f1)):
            code, lines, _e = run_lines("mc", name, ys, ps, cs, 10000000)
            for l in lines:
                f = l.split()
                if f[0] == "class" and f[2] != "undefined":
                    note(name, unbits(f[2]), float(ref[int(f[1])]), MC_BOUND_COUNTS, "n=%d K=%d class %s" % (n, k, f[1]))
                elif f[0] == "class":
                    check(math.isnan(float(ref[int(f[1])])), "sklearn %s: undefined here, a number there (class %s)" % (name, f[1]))
            for avg, pre in (("macro", "macro_"), ("weighted", "weighted_")):
                d = dict(l.split(" ", 1) for l in run_lines("mc", pre + name, ys, ps, cs, 10000000)[1])
                if d.get("left_out") == "0":
                    p2, r2, f2, _ = sm.precision_recall_fscore_support(yt, yp, labels=classes, average=avg)
                    ref2 = {"precision": p2, "recall": r2, "f1": f2}[name]
                    note(pre + name, unbits(d["value"]), float(ref2), MC_BOUND_COUNTS, "n=%d K=%d" % (n, k))
    for n, k, mode, ls, ss in ((50, 3, 0, 3, 4), (1000, 5, 0, 5, 6), (5000, 10, 1, 7, 8), (20000, 7, 0, 9, 10), (400, 64, 0, 11, 12)):
        classes = list(range(k))
        y = gen_mlabels(n, ls, k)
        p = gen_probs(mode, n, k, ss)
        ys, ps, cs = "GM:%d:%d:%d" % (n, ls, k), "GQ:%d:%d:%d:%d" % (mode, n, ss, k), spec_int(classes)
        mine = unbits(run("mc", "logloss", ys, ps, cs, 10000000)["value"])
        ref = float(sm.log_loss(y, p, labels=classes))
        check(abs(mine - ref) <= MC_LOGLOSS_ABS, "sklearn logloss n=%d K=%d: mine %r ref %r" % (n, k, mine, ref))
        worst["logloss (absolute)"] = max(worst.get("logloss (absolute)", 0.0), abs(mine - ref))
        d = dict(l.split(" ", 1) for l in run_lines("mc", "auc_ovr", ys, ps, cs, 10000000)[1] if not l.startswith("class"))
        if d.get("left_out") == "0":
            ref = float(sm.roc_auc_score(y, p, multi_class="ovr", average="macro", labels=classes))
            note("auc_ovr", unbits(d["value"]), ref, MC_BOUND_AUC, "n=%d K=%d" % (n, k))
    print("%-18s worst gap" % "metric")
    for name, w in worst.items():
        print("%-18s %.3g%s" % (name, w, " (absolute)" if "absolute" in name else " u"))

def main():
    a = sys.argv[1:]
    if not os.path.exists(BIN):
        print("no driver at %s (build it: sh scripts/build_metrics.sh)" % BIN)
        return 1
    steps = {"--splits": oracle_splits, "--draw": oracle_draw, "--ratio": oracle_ratio, "--metrics": oracle_metrics, "--auc": oracle_auc,
             "--sums": oracle_sums, "--sklearn": oracle_sklearn, "--multiclass": oracle_multiclass, "--multiclass-sklearn": oracle_multiclass_sklearn}
    todo = list(steps) if "--all" in a or not a else [x for x in a if x in steps]
    for k in todo:
        steps[k]()
    if FAILS:
        print("%d MISMATCHES" % len(FAILS))
        return 1
    print("all oracles agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())

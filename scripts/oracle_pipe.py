#!/usr/bin/env python3
"""Independent oracles for lib/transforms and lib/pipeline (docs/transforms.md section 1, docs/pipeline.md section 1).

Nothing here imports or calls the cancho code except through the compiled driver `build/pipe-check`; every rule is re-implemented from the
written spec in plain Python (IEEE doubles, `math.sqrt`, an `ordered_sum` of its own), and the bits are compared.

    python3 scripts/oracle_pipe.py --sums         the decisive-sum order of the package (the ONE place this file names it) against the driver
    python3 scripts/oracle_pipe.py --plans        P1: canonical plan text, random spellings, tags with line and column
    python3 scripts/oracle_pipe.py --transforms   X3: every transform, fit on train and apply to test, parameters and outputs, bit for bit
    python3 scripts/oracle_pipe.py --pipeline     P2/P3/P6: chains of steps, the schema alignment by name
    python3 scripts/oracle_pipe.py --file         P4: the model file read by an independent reader; the stored parameters applied in Python
    python3 scripts/oracle_pipe.py --sklearn      X4: against scikit-learn with the stated bounds
    python3 scripts/oracle_pipe.py --all
Exit 1 on any mismatch. PIPE_CHECK names the driver. A case count is printed with every verdict."""
import hashlib, math, os, pathlib, random, struct, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = os.environ.get("PIPE_CHECK", str(ROOT / "build" / "pipe-check"))
FAILS = []
CASES = {}


def run(*args, timeout=120):
    r = subprocess.run([BIN] + [str(a) for a in args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def check(ok, what):
    if not ok:
        FAILS.append(what)
        if len(FAILS) <= 25:
            print("MISMATCH", what)
    return ok


def count(label, n=1):
    CASES[label] = CASES.get(label, 0) + n


def bits(x):
    return struct.unpack("<q", struct.pack("<d", x))[0]


def hx(x):
    return "%016x" % (bits(x) & ((1 << 64) - 1))


def unbits(h):
    return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]


def ulps(a, b):
    """the distance in units in the last place between two finite doubles (0 when equal), on the order-preserving integer map"""
    def key(x):
        k = bits(x)
        return k if k >= 0 else -(k & 0x7FFFFFFFFFFFFFFF)
    return abs(key(a) - key(b))


# ---------------------------------------------------------------- the decisive sums (the package's order, written again)
K = 256  # block length of the reduction
LANES = 4  # accumulators inside a block (linalg.order.lanes)


def ordered_sum(t):
    """THE order of the package (lib/linalg/order.cho, docs/linalg.md section 2; reached by `transforms.dsum` through `metrics.sums.reduce`): blocks of K terms, LANES accumulators in a block (term i to lane
    i mod LANES), block partial (l0 + l1) + (l2 + l3), partials combined by a stack that merges equal levels, then right to left. When the package's
    order changes this is the one function to follow (and `--sums` fails until it does; mutants of order.cho are in mutants_pipe.py)."""
    n = len(t)
    stack = []  # (level, value)
    for b in range(0, n, K):
        blk = t[b:b + K]
        lanes = [0.0] * LANES
        m = len(blk)
        j = 0
        while j + LANES <= m:
            for l in range(LANES):
                lanes[l] = lanes[l] + blk[j + l]
            j += LANES
        for l in range(m - j):
            lanes[l] = lanes[l] + blk[j + l]
        part = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
        stack.append((0, part))
        while len(stack) >= 2 and stack[-1][0] == stack[-2][0]:
            (la, va), (lb, vb) = stack[-2], stack[-1]
            stack.pop(); stack.pop()
            stack.append((la + 1, va + vb))
    while len(stack) >= 2:
        (la, va), (lb, vb) = stack[-2], stack[-1]
        stack.pop(); stack.pop()
        stack.append((la, va + vb))
    return stack[0][1] if stack else 0.0


def total(v):
    return ordered_sum(v)


def sq_dev(v, c):
    return ordered_sum([(x - c) * (x - c) for x in v])


def num(x):
    """a value as the text the driver reads (shortest round trip), no negative zero"""
    if x == 0:
        x = 0.0
    return repr(float(x))


def sums_gate():
    rng = random.Random(7)
    sizes = list(range(1, 40)) + [63, 64, 65, 127, 255, 256, 257, 258, 511, 512, 513, 767, 768, 769, 1000, 1023, 1024, 1025, 2047, 2048, 2049, 4097, 5000]
    for n in sizes:
        for mode in range(3):
            if mode == 0:
                v = [rng.uniform(-1, 1) for _ in range(n)]
            elif mode == 1:
                v = [rng.choice([-1, 1]) * 10.0 ** rng.randint(-8, 12) * rng.random() for _ in range(n)]
            else:
                v = [rng.randint(-1000, 1000) / 8.0 for _ in range(n)]
            c = rng.choice([0.0, 0.5, v[0]])
            rc, out, err = run("sums", num(c), ",".join(num(x) for x in v))
            got = dict(l.split() for l in out.splitlines())
            check(rc == 0 and got.get("total") == hx(total(v)), "sums total n=%d mode=%d" % (n, mode))
            check(got.get("sqdev") == hx(sq_dev(v, c)), "sums sqdev n=%d mode=%d" % (n, mode))
            count("sums")


# ---------------------------------------------------------------- the reference transforms
class Refuse(Exception):
    def __init__(self, tag, a=0, b=0, c=0):
        self.tag, self.a, self.b, self.c = tag, a, b, c


def finite(x):
    return x - x == 0.0


def gather(X, c):
    v = [row[c] for row in X if row[c] is not None]
    return v


def ref_fit(step, X, ncols, step_no):
    """step: dict(kind, a, b, value, cols). Returns (ints, floats) in the layout of docs/transforms.md section 4."""
    kind, a, b, value, cols = step["kind"], step["a"], step["b"], step["value"], step["cols"]
    if len(X) < 1:
        raise Refuse("ml.input.empty", step_no)
    k = len(cols)
    ints = [k, ncols] + list(cols)
    fl = []
    counts = []
    for j, c in enumerate(cols):
        v = [x + 0.0 for x in gather(X, c)]
        m = len(v)
        if m == 0 and not (kind == 1 and a == 2):
            raise Refuse("ml.fit.all-missing", step_no, c)
        lo = min(v) if v else 0.0
        hi = max(v) if v else 0.0
        const = lo == hi
        if kind == 1:
            if a == 0:
                fill = lo if const else total(v) / float(m)
            elif a == 1:
                s = sorted(v)
                fill = s[m // 2] if m % 2 == 1 else (s[m // 2 - 1] + s[m // 2]) / 2.0
            else:
                fill = value
            if not finite(fill):
                raise Refuse("ml.fit.non-finite", step_no, c)
            fl.append(fill)
        elif kind == 2:
            d = hi - lo
            if not finite(d):
                raise Refuse("ml.fit.non-finite", step_no, c)
            if d == 0.0:
                d = 1.0
            fl += [lo, d]
        elif kind == 3:
            mean, d = lo, 1.0
            if not const:
                mean = total(v) / float(m)
                if not finite(mean):
                    raise Refuse("ml.fit.non-finite", step_no, c)
                den = m - a
                if den >= 1:
                    var = sq_dev(v, mean) / float(den)
                    sd = math.sqrt(var) if finite(var) and var >= 0 else float("nan")
                    if not finite(sd):
                        raise Refuse("ml.fit.non-finite", step_no, c)
                    if sd > 0.0:
                        d = sd
            fl += [mean, d]
        elif kind in (4, 5):
            cats = sorted(set(v))
            if len(cats) > b:
                raise Refuse("ml.limit.categories", step_no, c)
            counts.append(len(cats))
            fl += cats
        else:
            bins = b
            s = sorted(v)
            edges = []
            for i in range(1, bins):
                if a == 0:
                    rng_ = hi - lo
                    if not finite(rng_):
                        raise Refuse("ml.fit.non-finite", step_no, c)
                    e = lo + rng_ * float(i) / float(bins)
                else:
                    num_ = (m - 1) * i
                    q, r = num_ // bins, num_ % bins
                    e = s[q] if r == 0 else s[q] + (s[q + 1] - s[q]) * (float(r) / float(bins))
                    if not finite(e):
                        raise Refuse("ml.fit.non-finite", step_no, c)
                if edges and e < edges[-1]:
                    e = edges[-1]
                edges.append(e)
            fl += edges
    if kind == 1:
        ints.append(a)
    elif kind == 3:
        ints.append(a)
    elif kind in (4, 5):
        ints += [a] + counts
    elif kind == 6:
        ints += [b, a]
    return ints, fl


def ref_apply(kind, ints, fl, X):
    """Apply stored parameters (and nothing else) to X (list of rows of float or None). Returns (Y, unknown)."""
    k, ncols = ints[0], ints[1]
    cols = ints[2:2 + k]
    base = 2 + k
    sel = {c: j for j, c in enumerate(cols)}
    unknown = 0
    out = []
    if kind in (4, 5):
        cnts = ints[base + 1:base + 1 + k]
        offs = [sum(cnts[:j]) for j in range(k)]
    for ri, row in enumerate(X):
        new = []
        for c in range(ncols):
            x = row[c]
            j = sel.get(c)
            if j is None:
                new.append(x)
            elif x is None:
                if kind == 1:
                    new.append(fl[j])
                elif kind == 4:
                    new += [None] * cnts[j]
                else:
                    new.append(None)
            elif kind == 1:
                new.append(x)
            elif kind in (2, 3):
                new.append((x - fl[2 * j]) / fl[2 * j + 1])
            elif kind in (4, 5):
                cats = fl[offs[j]:offs[j] + cnts[j]]
                x0 = x + 0.0
                if x0 in cats:
                    idx = cats.index(x0)
                    if kind == 5:
                        new.append(float(idx))
                    else:
                        new += [1.0 if t == idx else 0.0 for t in range(cnts[j])]
                else:
                    if ints[base] == 1:
                        raise Refuse("ml.input.unknown-category", 0, c, ri)
                    unknown += 1
                    if kind == 5:
                        new.append(-1.0)
                    else:
                        new += [0.0] * cnts[j]
            else:
                w = ints[base] - 1
                edges = fl[j * w:(j + 1) * w]
                new.append(float(sum(1 for e in edges if e <= x)))
        out.append(new)
    return out, unknown


def width_after(kind, ints):
    if kind == 4:
        k = ints[0]
        return ints[1] - k + sum(ints[3 + k:3 + 2 * k])
    return ints[1]


# ---------------------------------------------------------------- the driver's output
def mat_text(X):
    return ";".join(",".join("?" if v is None else num(v) for v in row) for row in X)


def parse_matrix(lines, i):
    head = lines[i].split()
    r, c = int(head[1]), int(head[2])
    X = []
    for t in range(r):
        cells = lines[i + 1 + t].split()
        X.append([None if x == "?" else unbits(x) for x in cells])
    return X, i + 1 + r


def parse_steps(hexs):
    """An independent decoder of the steps payload (docs/transforms.md section 4): [(kind, ints, floats)]."""
    b = bytes.fromhex(hexs)
    n = struct.unpack_from("<I", b, 0)[0]
    pos = 4
    out = []
    for _ in range(n):
        ln = struct.unpack_from("<Q", b, pos)[0]
        pos += 8
        p = b[pos:pos + ln]
        pos += ln
        kind, d1, r1 = struct.unpack_from("<III", p, 0)
        assert (d1, r1) == (1, 1)
        ni = struct.unpack_from("<Q", p, 12)[0]
        ints = list(struct.unpack_from("<%dq" % ni, p, 20))
        at = 20 + 8 * ni
        d2, r2 = struct.unpack_from("<II", p, at)
        assert d2 == 2
        shape = struct.unpack_from("<%dQ" % r2, p, at + 8)
        cnt = 1
        for s in shape:
            cnt *= s
        fl = list(struct.unpack_from("<%dd" % cnt, p, at + 8 + 8 * r2))
        assert at + 8 + 8 * r2 + 8 * cnt == len(p)
        out.append((kind, ints, fl, shape))
    assert pos == len(b)
    return out


def drive_run(plan, names, train, test):
    rc, out, err = run("run", plan, ",".join(names), mat_text(train), mat_text(test))
    lines = out.splitlines()
    res = {"rc": rc, "steps": None, "names": None, "train": None, "test": None, "unknown": None, "refused": None, "test_refused": None}
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("steps "):
            res["steps"] = parse_steps(l.split()[1])
        elif l.startswith("names"):
            res["names"] = l.split()[1:]
        elif l.startswith("train"):
            res["train"], i = parse_matrix(lines, i)
            continue
        elif l.startswith("test refused"):
            res["test_refused"] = l.split()[2:]
        elif l.startswith("test "):
            res["test"], i = parse_matrix(lines, i)
            continue
        elif l.startswith("unknown"):
            res["unknown"] = int(l.split()[1])
        elif l.startswith("refused"):
            res["refused"] = l.split()[1:]
        i += 1
    return res


# ---------------------------------------------------------------- the generator
def gen_value(rng, kind, scale):
    if kind == "int":
        return float(rng.randint(-5, 9))
    if kind == "dec":
        return round(rng.uniform(-50, 50), 2) + 0.0
    if kind == "wide":
        return rng.choice([-1, 1]) * rng.random() * 10.0 ** rng.randint(-6, 9)
    if kind == "cat":
        return float(rng.randint(0, 4))
    if kind == "unit":
        return rng.random()
    return 0.0


def gen_matrix(rng, nrows, spec, miss_rate):
    """spec: list of (column kind, constant value or None, all_missing)"""
    cols = []
    for (kind, const, allmiss) in spec:
        col = []
        for _ in range(nrows):
            if allmiss or rng.random() < miss_rate:
                col.append(None)
            elif const is not None:
                col.append(const)
            else:
                col.append(gen_value(rng, kind, 1))
        cols.append(col)
    return [[cols[c][r] for c in range(len(spec))] for r in range(nrows)]


def random_spec(rng, ncols, exotic):
    spec = []
    for _ in range(ncols):
        kind = rng.choice(["int", "dec", "wide", "cat", "unit"])
        const, allmiss = None, False
        if exotic:
            r = rng.random()
            if r < 0.15:
                const = rng.choice([0.0, 1.0, 0.1, 3.5, -2.0, 1e-3, 123456.789])
            elif r < 0.25:
                allmiss = True
        spec.append((kind, const, allmiss))
    return spec


# ---------------------------------------------------------------- X3: single transforms
KIND_NAMES = {1: "impute", 2: "scale", 3: "scale", 4: "encode", 5: "encode", 6: "bucket"}


def plan_line(step, names, rng):
    cols = [names[c] for c in step["cols"]]
    kind = step["kind"]
    colv = ",".join(cols)
    if kind == 1:
        s = ["mean", "median", "constant"][step["a"]]
        kv = ["column=" + colv, "strategy=" + s] + (["value=" + num(step["value"])] if step["a"] == 2 else [])
    elif kind == 2:
        kv = ["column=" + colv, "kind=minmax"]
    elif kind == 3:
        kv = ["column=" + colv, "kind=standard", "ddof=%d" % step["a"]]
    elif kind in (4, 5):
        kv = ["column=" + colv, "kind=" + ("onehot" if kind == 4 else "label"), "max-categories=%d" % step["b"], "unknown=" + ["ignore", "refuse"][step["a"]]]
    else:
        kv = ["column=" + colv, "kind=" + ["width", "quantile"][step["a"]], "bins=%d" % step["b"]]
    return KIND_NAMES[kind] + " " + " ".join(kv)


def random_step(rng, ncols, kinds=(1, 2, 3, 4, 5, 6)):
    kind = rng.choice(kinds)
    k = rng.randint(1, ncols)
    cols = sorted(rng.sample(range(ncols), k))
    step = {"kind": kind, "a": 0, "b": 0, "value": 0.0, "cols": cols}
    if kind == 1:
        step["a"] = rng.randint(0, 2)
        step["value"] = rng.choice([0.0, 1.5, -2.0, 7.0, 0.1, 1e-3])
    elif kind == 3:
        step["a"] = rng.randint(0, 1)
    elif kind in (4, 5):
        step["a"] = rng.choice([0, 0, 1])
        step["b"] = rng.choice([2, 3, 64, 1024])
    elif kind == 6:
        step["a"] = rng.randint(0, 1)
        step["b"] = rng.choice([2, 3, 4, 7, 16, 256])
    return step


def ref_pipeline(steps, names, Xtrain, Xtest):
    """fit each step on the output of the one before (training rows only), apply the stored parameters to both; returns
    (list of (kind, ints, floats), train out, test out, unknown, names) or raises Refuse"""
    stored = []
    cur, tst = Xtrain, Xtest
    cur_names = list(names)
    unknown = 0
    for i, st in enumerate(steps):
        ints, fl = ref_fit(st, cur, len(cur_names), i + 1)
        stored.append((st["kind"], ints, fl))
        cur, _ = ref_apply(st["kind"], ints, fl, cur)
        try:
            tst, u = ref_apply(st["kind"], ints, fl, tst)
            unknown += u
        except Refuse as r:
            r.a = i + 1
            raise r
        k = ints[0]
        new = []
        for c, nm in enumerate(cur_names):
            if st["kind"] == 4 and c in ints[2:2 + k]:
                j = ints[2:2 + k].index(c)
                new += ["%s#%d" % (nm, t) for t in range(ints[3 + k + j])]
            else:
                new.append(nm)
        cur_names = new
    return stored, cur, tst, unknown, cur_names


def same_matrix(a, b):
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for x, y in zip(ra, rb):
            if (x is None) != (y is None):
                return False
            if x is not None and bits(x) != bits(y):
                return False
    return True


def compare_run(label, steps, names, Xtr, Xte, maxsteps=None):
    plan = "\n".join(plan_line(s, [n for n in names_after(steps, names, i)], None) for i, s in enumerate(steps)) + "\nmodel kind=constant task=regression"
    try:
        stored, tr, te, unknown, onames = ref_pipeline(steps, names, Xtr, Xte)
        want = None
    except Refuse as r:
        want = r
    got = drive_run(plan, names, Xtr, Xte)
    count(label + (" (refused: %s)" % want.tag if want is not None else " (fitted)"))
    if want is not None:
        if want.tag == "ml.input.unknown-category":
            return check(got["test_refused"] is not None and got["test_refused"][0] == want.tag and [int(x) for x in got["test_refused"][1:4]] == [want.a, want.b, want.c],
                         "%s: expected test refusal %s %s %s %s, got %s | %s" % (label, want.tag, want.a, want.b, want.c, got["test_refused"], plan))
        return check(got["refused"] is not None and got["refused"][0] == want.tag and int(got["refused"][1]) == want.a and (want.b == 0 or int(got["refused"][2]) == want.b),
                     "%s: expected %s step %s col %s, got %s | %s" % (label, want.tag, want.a, want.b, got["refused"], plan))
    ok = check(got["refused"] is None and got["test_refused"] is None and got["steps"] is not None, "%s: refused %s %s | %s" % (label, got["refused"], got["test_refused"], plan))
    if not ok:
        return False
    # parameters
    ok &= check(len(got["steps"]) == len(stored), "%s: step count" % label)
    for (gk, gi, gf, gs), (wk, wi, wf) in zip(got["steps"], stored):
        ok &= check(gk == wk and gi == wi and len(gf) == len(wf) and all(bits(a) == bits(b) for a, b in zip(gf, wf)), "%s: parameters of kind %d differ\n got %s %s\n want %s %s | %s" % (label, wk, gi, gf, wi, wf, plan))
    ok &= check(got["names"] == onames, "%s: names %s != %s" % (label, got["names"], onames))
    ok &= check(same_matrix(got["train"], tr), "%s: train output differs | %s" % (label, plan))
    ok &= check(same_matrix(got["test"], te), "%s: test output differs | %s" % (label, plan))
    ok &= check(got["unknown"] == unknown, "%s: unknown count %s != %s" % (label, got["unknown"], unknown))
    return ok


def names_after(steps, names, upto):
    """the column names the step number `upto` receives (a one-hot step renames its columns c#i)"""
    cur = list(names)
    for i, st in enumerate(steps[:upto]):
        # only the structure of the names matters; recompute widths from the categories is not possible here, so one-hot steps are never
        # followed by a step in the generated chains that names columns by position beyond the original ones
        pass
    return cur


def transforms_gate(n_cases):
    rng = random.Random(20260413)
    for case in range(n_cases):
        ncols = rng.randint(1, 5)
        names = ["c%d" % i for i in range(ncols)]
        exotic = case % 3 != 0
        spec = random_spec(rng, ncols, exotic)
        big = case % 17 == 0
        ntr = rng.choice([1, 2, 3, 5, 12, 40]) if not big else rng.choice([255, 256, 257, 600, 1100])
        nte = rng.randint(1, 30)
        miss = rng.choice([0.0, 0.0, 0.1, 0.3])
        Xtr = gen_matrix(rng, ntr, spec, miss)
        # the test rows: the same distributions, a different seed, and a few values nobody trained on
        Xte = gen_matrix(rng, nte, spec, rng.choice([0.0, 0.2]))
        if rng.random() < 0.3:
            for row in Xte:
                if rng.random() < 0.3:
                    row[rng.randrange(ncols)] = rng.choice([100.0, -100.0, 1e7, 2.5])
        step = random_step(rng, ncols)
        compare_run("transform kind %d" % step["kind"], [step], names, Xtr, Xte)


def pipeline_gate(n_cases):
    """chains: impute then scale (step 2 is fitted on the imputed column), impute then encode, bucket then encode ..."""
    rng = random.Random(99)
    for case in range(n_cases):
        ncols = rng.randint(2, 5)
        names = ["c%d" % i for i in range(ncols)]
        spec = random_spec(rng, ncols, case % 2 == 0)
        ntr = rng.choice([2, 5, 20, 300])
        Xtr = gen_matrix(rng, ntr, spec, rng.choice([0.05, 0.2]))
        Xte = gen_matrix(rng, rng.randint(1, 20), spec, 0.1)
        # step 1: impute some columns; step 2: scale or bucket them; step 3: encode the last column (one-hot renames it, so it comes last)
        k1 = rng.randint(1, ncols)
        cols1 = sorted(rng.sample(range(ncols), k1))
        s1 = {"kind": 1, "a": rng.randint(0, 2), "b": 0, "value": rng.choice([0.0, 2.0]), "cols": cols1}
        s2 = random_step(rng, ncols, kinds=(2, 3, 6, 2, 3))
        steps = [s1, s2]
        if rng.random() < 0.5:
            last = ncols - 1
            s3 = {"kind": rng.choice([4, 5]), "a": rng.randint(0, 1), "b": rng.choice([3, 64]), "value": 0.0, "cols": [last]}
            steps.append(s3)
        compare_run("chain of %d" % len(steps), steps, names, Xtr, Xte)


# ---------------------------------------------------------------- P1: the plan text
def canon_float(x):
    """flt.put_float: the shortest decimal that reads back, positional from 1e-6 up to 1e21 with a point, scientific outside"""
    if x == 0:
        return "0.0"
    r = repr(abs(x))
    mant, _, ex = r.partition("e")
    e10 = int(ex) if ex else 0
    ip, _, fp = mant.partition(".")
    digits = (ip + fp).lstrip("0")
    # position of the decimal point relative to the digits: value = 0.DIGITS * 10^pt
    lead = len(ip.lstrip("0")) if ip.strip("0") else -(len(fp) - len(fp.lstrip("0")))
    pt = lead + e10
    digits = digits.rstrip("0") or "0"
    sign = "-" if x < 0 else ""
    sci = pt - 1  # exponent in d.ddd form
    if -6 <= sci < 21:
        if pt <= 0:
            body = "0." + "0" * (-pt) + digits
        elif pt >= len(digits):
            body = digits + "0" * (pt - len(digits)) + ".0"
        else:
            body = digits[:pt] + "." + digits[pt:]
        return sign + body
    d = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
    return "%s%se%d" % (sign, d, sci)


STEP_TABLE = {
    "impute": [("column", "cols", None, None), ("strategy", "enum:mean,median,constant", None, None), ("value", "float", None, ("strategy", "constant"))],
    "scale": [("column", "cols", None, None), ("kind", "enum:minmax,standard", None, None), ("ddof", "int:0,1", "0", ("kind", "standard"))],
    # min-count and missing (stage 2 of issue 21) are written only when they are not their defaults: a default that starts with `~` (so a numeric line keeps its text)
    "encode": [("column", "cols", None, None), ("kind", "enum:onehot,label", None, None), ("max-categories", "int:1,1024", "64", None), ("unknown", "enum:ignore,refuse", "ignore", None),
               ("min-count", "int:1,1000000", "~1", None), ("missing", "enum:pass,zero", "~pass", None)],
    "freq": [("column", "cols", None, None), ("max-categories", "int:1,1024", "64", None), ("unknown", "enum:ignore,refuse", "ignore", None), ("min-count", "int:1,1000000", "1", None),
             ("missing", "enum:pass,zero", "pass", None)],
    "hash": [("column", "cols", None, None), ("buckets", "int:2,1024", None, None), ("missing", "enum:pass,zero", "pass", None)],
    "bucket": [("column", "cols", None, None), ("kind", "enum:width,quantile", None, None), ("bins", "int:2,256", None, None)],
}
MODEL_TABLE = {
    "constant": [("task", "enum:regression,classification", None, None)],
    "linear": [("l2", "fnn", "0.0", None)],
    "logistic": [("iterations", "int:1,1000", "100", None), ("l2", "fnn", "0.0", None)],
    "softmax": [("iterations", "int:1,1000", "100", None), ("l2", "fnn", "0.0", None)],
    "kmeans": [("k", "int:1,1024", None, None), ("iterations", "int:1,1000", "10", None), ("init", "enum:random,kmeans++", "kmeans++", None)],
    "tree": [("task", "enum:classification,regression", None, None), ("max-depth", "int:1,1000", "8", None), ("min-leaf", "int:1,1000000", "1", None), ("bins", "int:2,255", "64", None),
             ("max-nodes", "int:1,999999999", "4194304", None)],
    "forest": [("task", "enum:classification,regression", None, None), ("trees", "int:1,1024", "100", None), ("max-depth", "int:1,1000", "8", None), ("min-leaf", "int:1,1000000", "1", None),
               ("bins", "int:2,255", "64", None), ("mtry", "int:0,4096", "0", None), ("bootstrap", "int:0,1", "1", None), ("max-nodes", "int:1,999999999", "4194304", None)],
    "boost": [("task", "enum:classification,regression,ranking,lambdarank", None, None), ("rounds", "int:1,10000", "100", None), ("learning-rate", "fnn", "0.1", None), ("max-depth", "int:1,1000", "4", None),
              ("min-leaf", "int:1,1000000", "10", None), ("bins", "int:2,255", "64", None), ("l2", "fnn", "1.0", None), ("subsample", "fnn", "1.0", None),
              ("early-stopping", "int:0,1000", "0", None), ("max-nodes", "int:1,999999999", "4194304", None)],
}


class PlanErr(Exception):
    def __init__(self, tag, a, b):
        self.tag, self.a, self.b = tag, a, b


def parse_float_ref(t):
    """a plan float: the grammar of the reader; python float() agrees on what it accepts except spellings the reader refuses (inf, nan, underscores, blanks)"""
    import re
    if not re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", t):
        return None
    x = float(t)
    if not finite(x):
        return None
    return x + 0.0


def canon_plan(text):
    """The canonical plan of a text, by the written grammar (docs/pipeline.md section 2); raises PlanErr(tag, line-or-limit, col)."""
    if len(text.encode()) > 65536:
        raise PlanErr("ml.limit.plan", 1, 0)
    raw = text.split("\n")
    if raw and raw[-1] == "":
        raw = raw[:-1]
    out = []
    model_seen = False
    lines = 0
    n = 0
    for ln, line in enumerate(raw, 1):
        n = ln
        if line.endswith("\r"):
            line = line[:-1]
        stripped = line.lstrip(" \t")
        if stripped == "" or stripped.startswith("#"):
            continue
        lines += 1
        if lines > 64:
            raise PlanErr("ml.limit.plan", 2, ln)
        if model_seen:
            raise PlanErr("ml.plan.model-line", ln, len(line) - len(stripped) + 1)
        toks = []
        i = 0
        while i < len(line):
            if line[i] in " \t":
                i += 1
                continue
            j = i
            while j < len(line) and line[j] not in " \t":
                if not (33 <= ord(line[j]) <= 126):
                    raise PlanErr("ml.plan.syntax", ln, j + 1)
                j += 1
            toks.append((i, line[i:j]))
            i = j
        if len(toks) > 17:
            raise PlanErr("ml.plan.syntax", ln, toks[17][0] + 1)
        verb_col, verb = toks[0]
        if "=" in verb:
            raise PlanErr("ml.plan.syntax", ln, verb_col + verb.index("=") + 1)
        kvs = []
        for col, t in toks[1:]:
            if "=" not in t:
                raise PlanErr("ml.plan.syntax", ln, col + 1)
            k, _, v = t.partition("=")
            if k == "" or v == "" or "=" in v:
                raise PlanErr("ml.plan.syntax", ln, col + 1)
            kvs.append((col, k, v))
        if verb not in STEP_TABLE and verb != "model":
            raise PlanErr("ml.plan.unknown-step", ln, verb_col + 1)
        if verb == "model":
            kt = [kv for kv in kvs if kv[1] == "kind"]
            if not kt:
                raise PlanErr("ml.plan.missing-key", ln, verb_col + 1)
            if kt[0][2] not in MODEL_TABLE:
                raise PlanErr("ml.plan.bad-value", ln, kt[0][0] + 6)
            defs = [("kind", "enum:" + ",".join(MODEL_TABLE), None, None)] + MODEL_TABLE[kt[0][2]]
        else:
            defs = STEP_TABLE[verb]
        names = {d[0] for d in defs}
        for col, k, v in kvs:
            if k not in names:
                raise PlanErr("ml.plan.unknown-key", ln, col + 1)
        vals = {}
        parts = [verb]
        for name, ty, default, cond in defs:
            ts = [kv for kv in kvs if kv[1] == name]
            if len(ts) > 1:
                raise PlanErr("ml.plan.duplicate-key", ln, ts[1][0] + 1)
            allowed = True
            if cond is not None:
                allowed = vals.get(cond[0]) == cond[1]
            if not allowed:
                if ts:
                    raise PlanErr("ml.plan.bad-value", ln, ts[0][0] + len(name) + 2)
                continue
            if not ts and default is None:
                raise PlanErr("ml.plan.missing-key", ln, verb_col + 1)
            if not ts and default.startswith("~"):
                vals[name] = default[1:]
                continue
            if ts:
                col, k, v = ts[0]
                vcol = col + len(name) + 2
            else:
                v, vcol = default, 0
            if ty == "cols":
                if v == "*":
                    cv = "*"
                else:
                    names_ = v.split(",")
                    if len(names_) > 4096:
                        raise PlanErr("ml.limit.plan", 3, 0)
                    pos = 0
                    for nm in names_:
                        if nm == "" or any(not (33 <= ord(ch) <= 126) or ch in "=#*" for ch in nm):
                            raise PlanErr("ml.plan.bad-value", ln, vcol + pos)
                        if len(nm) > 255:
                            raise PlanErr("ml.limit.plan", 4, 0)
                        pos += len(nm) + 1
                    if len(set(names_)) != len(names_):
                        # the column of the later occurrence of the first duplicate pair found in sorted order
                        s = sorted(names_)
                        pos2 = 0
                        starts = []
                        for nm in names_:
                            starts.append(pos2)
                            pos2 += len(nm) + 1
                        dup = max(i for i in range(len(names_)) if names_.count(names_[i]) > 1)
                        raise PlanErr("ml.plan.bad-value", ln, vcol + starts[dup])
                    cv = ",".join(sorted(names_))
            elif ty.startswith("enum:"):
                if v not in ty[5:].split(","):
                    raise PlanErr("ml.plan.bad-value", ln, vcol)
                cv = v
            elif ty.startswith("int:"):
                lo, hi = [int(x) for x in ty[4:].split(",")]
                if not v.isdigit() or len(v) > 9 or not v.isascii():
                    raise PlanErr("ml.plan.bad-value", ln, vcol)
                if not lo <= int(v) <= hi:
                    raise PlanErr("ml.param.out-of-range", ln, vcol)
                cv = str(int(v))
            else:
                x = parse_float_ref(v)
                if x is None or (ty == "fnn" and x < 0):
                    raise PlanErr("ml.plan.bad-value", ln, vcol)
                cv = canon_float(x)
            vals[name] = cv
            if default is not None and default.startswith("~") and cv == default[1:]:
                continue
            parts.append("%s=%s" % (name, cv))
        out.append(" ".join(parts))
        if verb == "model":
            model_seen = True
    if not model_seen:
        raise PlanErr("ml.plan.model-line", n + 1, 1)
    return "\n".join(out) + "\n"


def drive_plan(text):
    rc, out, err = run("plan", text)
    if out.startswith("refused"):
        t = out.split()
        return ("refused", t[1], int(t[2]), int(t[3]))
    assert out.startswith("plan\n") and out.endswith("end\n"), (out, err)
    return ("ok", out[len("plan\n"):-len("end\n")])


def spell(rng, line):
    """a random re-spelling of a canonical line: key order, blanks, leading zeros, equivalent floats"""
    toks = line.split(" ")
    verb, kvs = toks[0], toks[1:]
    if not kvs:
        return line
    rng.shuffle(kvs)
    # the model line's kind may come anywhere
    for i, kv in enumerate(kvs):
        k, _, v = kv.partition("=")
        if k in ("bins", "max-categories", "ddof", "iterations", "k", "trees", "max-depth", "min-leaf", "min-count", "buckets") and rng.random() < 0.4:
            kvs[i] = "%s=%s%s" % (k, "0" * min(rng.randint(1, 3), 9 - len(v)), v)
        elif k in ("value", "l2"):
            x = float(v)
            alt = rng.choice([repr(x), "%.17g" % x, ("%e" % x) if x != 0 else "0", v])
            if parse_float_ref(alt) == x:
                kvs[i] = "%s=%s" % (k, alt)
        elif k == "column" and v != "*" and rng.random() < 0.7:
            n_ = v.split(",")
            rng.shuffle(n_)
            kvs[i] = "column=" + ",".join(n_)
    sep = lambda: rng.choice([" ", "  ", "\t", " \t "])
    s = rng.choice(["", " ", "\t"]) + verb + sep() + sep().join(kvs) + rng.choice(["", " ", "\t"])
    return s


def plans_gate(n_cases):
    rng = random.Random(5150)
    names_pool = ["a", "b", "c", "income", "age", "x1", "x_2", "Z", "col-9", "q.r"]
    for case in range(n_cases):
        lines = []
        nsteps = rng.randint(0, 5)
        for _ in range(nsteps):
            verb = rng.choice(list(STEP_TABLE))
            cols = "*" if rng.random() < 0.2 else ",".join(sorted(rng.sample(names_pool, rng.randint(1, 4))))
            if verb == "impute":
                s = rng.choice(["mean", "median", "constant"])
                line = "impute column=%s strategy=%s" % (cols, s) + (" value=%s" % canon_float(rng.choice([0.0, 1.5, -3.25, 1e-7, 1e21, 123456789.125, 0.1, 5.0, 2.5e-300]))) if s == "constant" else "impute column=%s strategy=%s" % (cols, s)
            elif verb == "scale":
                kd = rng.choice(["minmax", "standard"])
                line = "scale column=%s kind=%s" % (cols, kd) + (" ddof=%d" % rng.randint(0, 1) if kd == "standard" else "")
            elif verb == "encode":
                line = "encode column=%s kind=%s max-categories=%d unknown=%s" % (cols, rng.choice(["onehot", "label"]), rng.randint(1, 1024), rng.choice(["ignore", "refuse"]))
                if rng.random() < 0.4:
                    line += " min-count=%d" % rng.choice([2, 3, 1000000])
                if rng.random() < 0.4:
                    line += " missing=zero"
            elif verb == "freq":
                line = "freq column=%s max-categories=%d unknown=%s min-count=%d missing=%s" % (cols, rng.randint(1, 1024), rng.choice(["ignore", "refuse"]), rng.choice([1, 2, 77]), rng.choice(["pass", "zero"]))
            elif verb == "hash":
                line = "hash column=%s buckets=%d missing=%s" % (cols, rng.randint(2, 1024), rng.choice(["pass", "zero"]))
            else:
                line = "bucket column=%s kind=%s bins=%d" % (cols, rng.choice(["width", "quantile"]), rng.randint(2, 256))
            lines.append(line)
        kind = rng.choice(list(MODEL_TABLE))
        mdefs = MODEL_TABLE[kind]
        mparts = ["model", "kind=" + kind]
        for name, ty, default, cond in mdefs:
            if ty.startswith("enum:"):
                mparts.append("%s=%s" % (name, rng.choice(ty[5:].split(","))))
            elif ty.startswith("int:"):
                lo, hi = [int(x) for x in ty[4:].split(",")]
                mparts.append("%s=%d" % (name, rng.randint(lo, min(hi, 999))))
            else:
                mparts.append("%s=%s" % (name, canon_float(rng.choice([0.0, 0.1, 1.0, 0.001, 12.5]))))
        lines.append(" ".join(mparts))
        canon = "\n".join(lines) + "\n"
        # a spelling that differs from the canonical one in every way the grammar allows
        sp = []
        for l in lines:
            if rng.random() < 0.15:
                sp.append("")
            if rng.random() < 0.15:
                sp.append("# a comment " + l)
            sp.append(spell(rng, l))
        eol = rng.choice(["\n", "\r\n"])
        text = eol.join(sp) + rng.choice(["", eol, eol + eol])
        count("plan spellings")
        want = canon_plan(text)
        check(want == canon, "plan reference: canonical %r != reference %r of %r" % (canon, want, text))
        got = drive_plan(text)
        check(got == ("ok", canon), "plan: %r -> %r, wanted %r" % (text, got, canon))
        # fixed point
        again = drive_plan(canon)
        check(again == ("ok", canon), "plan fixed point: %r -> %r" % (canon, again))
        count("plan fixed points")
    # floats alone: the canonical spelling of many doubles, read through the reader and written by the driver
    for i in range(400):
        x = rng.choice([rng.uniform(-1, 1), rng.uniform(-1e6, 1e6), rng.random() * 10.0 ** rng.randint(-30, 30), float(rng.randint(-10, 10)), rng.randint(1, 10 ** 9) / 1000.0, 10.0 ** rng.randint(-8, 22)])
        if x == 0:
            continue
        for spelling in (repr(x), "%.17g" % x, "%e" % x):
            t = "model kind=linear l2=%s" % spelling.lstrip("-")
            ax = abs(x)
            if parse_float_ref(spelling.lstrip("-")) is None:
                continue
            want = "model kind=linear l2=%s\n" % canon_float(float(spelling.lstrip("-")))
            got = drive_plan(t)
            check(got == ("ok", want), "float %s -> %r wanted %r" % (spelling, got, want))
            count("float spellings")
    # random garbage: the reference and the driver must agree on the verdict and the position
    alphabet = list("abcdeimnopqrstuvw=, \t\n#*0123456789.-+e")
    words = ["impute", "scale", "encode", "bucket", "model", "column", "kind", "strategy", "value", "bins", "ddof", "task", "l2", "unknown", "max-categories",
             "mean", "standard", "onehot", "width", "constant", "regression", "linear", "refuse", "*", "a,b", "a,a", "1", "0", "257", "1e3", "-1", "nan", "=",
             "freq", "hash", "buckets", "min-count", "missing", "zero", "pass", "1000001", "1024", "1025"]
    for i in range(1500):
        t = []
        for _ in range(rng.randint(1, 4)):
            line = " ".join(rng.choice(words) + ("=" + rng.choice(words) if rng.random() < 0.6 else "") for _ in range(rng.randint(1, 5)))
            t.append(line)
        text = "\n".join(t) if rng.random() < 0.9 else "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        try:
            want = ("ok", canon_plan(text))
        except PlanErr as e:
            want = ("refused", e.tag, e.a, e.b)
        got = drive_plan(text)
        check(got == want, "garbage plan %r: driver %r, reference %r" % (text, got, want))
        count("garbage plans (verdict and position)")


# ---------------------------------------------------------------- P4: the file
def u32(b, at):
    return struct.unpack_from("<I", b, at)[0]


def u64(b, at):
    return struct.unpack_from("<Q", b, at)[0]


def parse_file(data):
    """An independent reader of CML1 with steps, from design.md 4.1 and docs/pipeline.md 5."""
    assert data[:4] == b"CML1" and u32(data, 4) == 1 and u32(data, 8) == 5
    secs, pos = {}, 12
    for want in range(1, 6):
        tag, n = u32(data, pos), u64(data, pos + 4)
        assert tag == want
        secs[tag] = data[pos + 12:pos + 12 + n]
        pos += 12 + n
    assert pos == len(data) - 32 and hashlib.sha256(data[:-32]).digest() == data[-32:]
    sc = secs[2]
    ncols, target = u32(sc, 0), u32(sc, 4)
    cols, p = [], 8
    for _ in range(ncols):
        n = u32(sc, p)
        cols.append((sc[p + 4:p + 4 + n].decode(), u32(sc, p + 4 + n), u32(sc, p + 8 + n)))
        p += 12 + n
    assert p == len(sc)
    st = secs[3]
    ns = u32(st, 0)
    steps, q = [], 4
    for _ in range(ns):
        ln = u64(st, q)
        steps.append(st[q + 8:q + 8 + ln])
        q += 8 + ln
    assert q == len(st)
    return {"plan": secs[1].decode(), "columns": cols, "target": target, "steps": steps, "steps_raw": st, "model": secs[4], "meta": secs[5]}


def decode_step(p):
    kind, d1, r1 = struct.unpack_from("<III", p, 0)
    ni = struct.unpack_from("<Q", p, 12)[0]
    ints = list(struct.unpack_from("<%dq" % ni, p, 20))
    at = 20 + 8 * ni
    d2, r2 = struct.unpack_from("<II", p, at)
    shape = struct.unpack_from("<%dQ" % r2, p, at + 8)
    cnt = 1
    for s in shape:
        cnt *= s
    fl = list(struct.unpack_from("<%dd" % cnt, p, at + 8 + 8 * r2))
    return kind, ints, fl


def file_gate(n_cases):
    sys.path.insert(0, str(ROOT / "tests" / "conformance"))
    import importlib
    mf = importlib.import_module("test_model_file")  # the independent parser of tools/ml/mfile.cho's files
    rng = random.Random(4242)
    # a file with no steps is the file mfile.cho writes: its independent reader accepts it, and the bytes are those of the layout
    for names in (["x"], ["a", "b", "c"]):
        rc, out, err = run("file", "model kind=constant task=regression", ",".join(names), ";".join(",".join(["1"] * len(names)) for _ in range(4)))
        data = bytes.fromhex(out.split()[1])
        parsed = mf.parse(data)
        check(parsed["plan"] == "model kind=constant task=regression\n" and parsed["kind"] == 1 and len(parsed["columns"]) == len(names) + 1 and parsed["rows"] == 4, "zero-step file: mfile's reader: %r" % parsed)
        f = parse_file(data)
        check(f["steps_raw"] == b"\0\0\0\0", "zero-step file: steps section is u32 0")
        # the same bytes from the layout written in Python
        def sec(t, p): return struct.pack("<IQ", t, len(p)) + p
        sch = struct.pack("<II", len(names) + 1, 0) + b"".join(struct.pack("<I", len(n)) + n.encode() + struct.pack("<II", 2, 0) for n in ["y"] + names)
        md = struct.pack("<IIIQ", 1, 1, 1, 4) + struct.pack("<qqqq", 1, 0, 0, 0)
        me = struct.pack("<QQQII", 4, len(names) + 1, 7, 0, 3) + b"pin"
        body = b"CML1" + struct.pack("<II", 1, 5) + sec(1, b"model kind=constant task=regression\n") + sec(2, sch) + sec(3, b"\0\0\0\0") + sec(4, md) + sec(5, me)
        want = body + hashlib.sha256(body).digest()
        check(want == data, "zero-step file differs from the layout of mfile.cho written in Python (%d vs %d bytes)" % (len(want), len(data)))
        count("zero-step files vs mfile layout")
    # files with steps: parse, check the sections, apply the STORED parameters in Python to a test matrix and compare with the driver's predict
    for case in range(n_cases):
        ncols = rng.randint(2, 4)
        names = ["f%d" % i for i in range(ncols)]
        spec = random_spec(rng, ncols, False)
        Xtr = gen_matrix(rng, rng.choice([3, 10, 50]), spec, 0.1)
        Xte = gen_matrix(rng, rng.randint(1, 15), spec, 0.1)
        s1 = {"kind": 1, "a": rng.randint(0, 1), "b": 0, "value": 0.0, "cols": sorted(rng.sample(range(ncols), rng.randint(1, ncols)))}
        s2 = random_step(rng, ncols, kinds=(2, 3, 6))
        steps = [s1, s2]
        if rng.random() < 0.5:
            steps.append({"kind": rng.choice([4, 5]), "a": 0, "b": 64, "value": 0.0, "cols": [ncols - 1]})
        plan = "\n".join(plan_line(s, names, None) for s in steps) + "\nmodel kind=constant task=regression"
        try:
            ref_pipeline(steps, names, Xtr, Xte)
        except Refuse:
            continue
        rc, out, err = run("file", plan, ",".join(names), mat_text(Xtr))
        if not check(out.startswith("file "), "file: %s %s" % (out, err)):
            continue
        data = bytes.fromhex(out.split()[1])
        f = parse_file(data)
        check(f["plan"] == canon_plan(plan), "file: plan section")
        check(len(f["steps"]) == len(steps) and f["steps_raw"][:4] == struct.pack("<I", len(steps)), "file: step count")
        cur = Xte
        try:
            for p in f["steps"]:
                kind, ints, fl = decode_step(p)
                cur, _ = ref_apply(kind, ints, fl, cur)
        except Refuse:
            continue
        rc, out2, err2 = run("predict", out.split()[1], mat_text(Xte))
        lines = out2.splitlines()
        ok = check(lines and lines[0].startswith("test "), "file predict: %s" % out2[:200])
        if ok:
            got, _ = parse_matrix(lines, 0)
            check(same_matrix(got, cur), "file: the driver's predict differs from the stored parameters applied in Python")
        count("files with steps")


# ---------------------------------------------------------------- P6: the schema
def align_gate():
    rng = random.Random(77)
    base = ["a", "b", "c", "d", "e"]
    for case in range(300):
        nf = rng.randint(1, 5)
        feats = base[:nf]
        types = "".join(rng.choice("012") for _ in feats)
        r = rng.random()
        given = list(feats)
        gtypes = list(types)
        if r < 0.5:
            order = list(range(nf))
            rng.shuffle(order)
            given = [feats[i] for i in order]
            gtypes = [types[i] for i in order]
        else:
            if rng.random() < 0.5 and given:
                i = rng.randrange(len(given))
                del given[i]; del gtypes[i]
            if rng.random() < 0.5:
                given.append(rng.choice(["x", "y", "z"])); gtypes.append("2")
            if rng.random() < 0.4 and given:
                i = rng.randrange(len(given)); gtypes[i] = rng.choice("012")
            if rng.random() < 0.3 and given:
                given.append(rng.choice(given)); gtypes.append("2")
            rng.shuffle(given)  # types follow below
        # types must follow the names: rebuild
        tmap = {n: t for n, t in zip(feats, types)}
        gt = []
        for n in given:
            gt.append(tmap.get(n, "2") if r < 0.5 else (rng.choice("012") if rng.random() < 0.1 else tmap.get(n, "2")))
        rc, out, err = run("align", ",".join(feats), types, ",".join(given), "".join(gt))
        # reference
        missing = [n for n in feats if n not in given]
        extra = [n for n in given if n not in feats]
        dups = [n for n in feats if given.count(n) > 1]
        mistyped = [n for n in feats if n in given and gt[given.index(n)] != tmap[n]]
        exp = []
        if missing: exp.append("missing: " + ",".join(missing))
        if extra: exp.append("extra: " + ",".join(extra))
        if mistyped: exp.append("mistyped: " + ",".join(mistyped))
        if dups: exp.append("duplicate: " + ",".join(dups))
        if exp:
            want = "refused ml.schema.mismatch " + "; ".join(exp)
        else:
            want = "perm " + " ".join(str(given.index(n)) for n in feats)
        check(out.strip() == want, "align feats=%s given=%s types=%s gt=%s: %r != %r" % (feats, given, types, gt, out.strip(), want))
        count("schema alignments")


# ---------------------------------------------------------------- X4: scikit-learn
EPS = 2.0 ** -52


def sklearn_gate(n_cases):
    """X4. The bounds are in units of eps * (the largest magnitude M in the training column), because the two libraries add the same numbers in
    different orders (numpy's pairwise sums or an incremental update; the 4-lane block tree here), and the rounding errors of a sum are
    absolute errors of the size of its terms: an error of a few eps * M on a value near zero is a large number of ulps of that value and is not
    a larger error. The coefficient 32 for a mean or an edge is the analytic bound 2 * (log2 n + 4) for n <= 500 (each order has a rounding
    error of at most (log2 n + 4) eps * sum |x| / n on the mean), rounded up to a power of two; 64 for a transformed value, which also
    divides by a scale that carries its own error. Exact comparisons (median, constant, encoders, bucket indices away from an edge) have bound 0."""
    import numpy as np
    import sklearn
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
    print("scikit-learn", sklearn.__version__, "numpy", np.__version__)
    rng = random.Random(31337)
    worst = {}

    def note(name, a, b, k, scale):
        err = abs(a - b)
        ratio = err / (EPS * scale) if scale > 0 else (0.0 if err == 0 else float("inf"))
        w = worst.setdefault(name, [0.0, k, 0, 0])
        w[0] = max(w[0], ratio)
        w[2] += 1
        w[3] += 1 if ulps(a, b) == 0 else 0
        check(ratio <= k, "sklearn %s: %r vs %r differ by %.2f eps*M (bound %s)" % (name, a, b, ratio, k))

    for case in range(n_cases):
        ncols = rng.randint(1, 4)
        names = ["c%d" % i for i in range(ncols)]
        spec = [(rng.choice(["dec", "wide", "unit", "int"]), None, False) for _ in range(ncols)]
        ntr = rng.choice([8, 30, 120, 500])
        Xtr = gen_matrix(rng, ntr, spec, 0.0)
        Xte = gen_matrix(rng, 20, spec, 0.0)
        Xtr_m = gen_matrix(rng, ntr, spec, 0.2)
        Xte_m = gen_matrix(rng, 20, spec, 0.2)
        # keep every column with at least 3 distinct values
        for c in range(ncols):
            if len({r[c] for r in Xtr}) < 3:
                for r in range(min(3, ntr)):
                    Xtr[r][c] = float(r) + 0.25
            if len({r[c] for r in Xtr_m if r[c] is not None}) < 3:
                for r in range(min(3, ntr)):
                    Xtr_m[r][c] = float(r) + 0.25
        A, B = np.array(Xtr), np.array(Xte)
        Am = np.array([[np.nan if v is None else v for v in r] for r in Xtr_m])
        Bm = np.array([[np.nan if v is None else v for v in r] for r in Xte_m])
        allcols = list(range(ncols))
        Mm = [max(abs(r[c]) for r in Xtr_m if r[c] is not None) for c in range(ncols)]
        M = [max(abs(r[c]) for r in Xtr) for c in range(ncols)]

        def ours(kind, a, b, X, T, value=0.0):
            step = {"kind": kind, "a": a, "b": b, "value": value, "cols": allcols}
            plan = plan_line(step, names, None) + "\nmodel kind=constant task=regression"
            g = drive_run(plan, names, X, T)
            check(not (g["refused"] or g["test_refused"]), "sklearn gate: our run refused %s %s" % (g["refused"], g["test_refused"]))
            return g

        # min-max (with missing cells: sklearn passes NaN through)
        g = ours(2, 0, 0, Xtr_m, Xte_m)
        sk = MinMaxScaler().fit(Am)
        want = sk.transform(Bm)
        for r in range(len(Xte_m)):
            for c in range(ncols):
                d = g["steps"][0][2][2 * c + 1]
                if Xte_m[r][c] is None:
                    check(g["test"][r][c] is None and np.isnan(want[r][c]), "minmax missing pass-through")
                else:
                    note("MinMaxScaler transform", g["test"][r][c], float(want[r][c]), 64, EPS and (max(Mm[c], abs(Xte_m[r][c])) / d) * (1 + abs(float(want[r][c]))))
        # standard, population
        g = ours(3, 0, 0, Xtr_m, Xte_m)
        sk = StandardScaler().fit(Am)
        want = sk.transform(Bm)
        for c in range(ncols):
            mean_o, sd_o = g["steps"][0][2][2 * c], g["steps"][0][2][2 * c + 1]
            note("StandardScaler mean_", mean_o, float(sk.mean_[c]), 32, Mm[c])
            note("StandardScaler scale_ (relative)", sd_o, float(sk.scale_[c]), 16, sd_o)
            for r in range(len(Xte_m)):
                if Xte_m[r][c] is None:
                    check(g["test"][r][c] is None, "standard missing pass-through")
                else:
                    note("StandardScaler transform", g["test"][r][c], float(want[r][c]), 64, (max(Mm[c], abs(Xte_m[r][c])) / sd_o) * (1 + abs(float(want[r][c]))))
        # imputer: mean (eps * M), median and constant (exact)
        for strat, a in (("mean", 0), ("median", 1), ("constant", 2)):
            g = ours(1, a, 0, Xtr_m, Xte_m, value=1.5)
            sk = SimpleImputer(strategy=strat, fill_value=1.5 if strat == "constant" else None).fit(Am)
            want = sk.transform(Bm)
            for r in range(len(Xte_m)):
                for c in range(ncols):
                    note("SimpleImputer " + strat, g["test"][r][c], float(want[r][c]), 32 if strat == "mean" else 0, Mm[c])
        # buckets: uniform and quantile; indices equal unless a value lies within a few eps*M of an edge
        for strat, a in (("uniform", 0), ("quantile", 1)):
            for nb in (2, 5, 10):
                if ntr < nb * 2:
                    continue
                g = ours(6, a, nb, Xtr, Xte)
                kb = KBinsDiscretizer(n_bins=nb, encode="ordinal", strategy=strat, subsample=None, **({"quantile_method": "linear"} if strat == "quantile" else {})).fit(A)
                want = kb.transform(B)
                edges_ours = g["steps"][0][2]
                for c in range(ncols):
                    e_sk = kb.bin_edges_[c][1:-1]
                    if len(e_sk) != nb - 1:
                        continue
                    for i in range(nb - 1):
                        note("KBinsDiscretizer %s edges" % strat, edges_ours[c * (nb - 1) + i], float(e_sk[i]), 32, M[c])
                for r in range(len(Xte)):
                    for c in range(ncols):
                        if len(kb.bin_edges_[c]) - 1 != nb:
                            # scikit-learn drops a bin whose edges are less than 1e-8 apart (ties in the data) and so renumbers the buckets above it;
                            # this library keeps every edge (docs/transforms.md 2.4), so the indices are not comparable on such a column
                            w = worst.setdefault("KBinsDiscretizer columns with merged bins (not compared)", [0.0, 0, 0, 0])
                            w[2] += 1
                            continue
                        eq = g["test"][r][c] == float(want[r][c])
                        e_sk = kb.bin_edges_[c][1:-1]
                        near = any(abs(Xte[r][c] - float(e)) <= 64 * EPS * M[c] for e in e_sk)
                        check(eq or near, "KBinsDiscretizer %s index differs away from an edge: %r vs %r (x=%r)" % (strat, g["test"][r][c], want[r][c], Xte[r][c]))
                        w = worst.setdefault("KBinsDiscretizer %s index (exact away from an edge)" % strat, [0.0, 0, 0, 0])
                        w[2] += 1
                        w[3] += 1 if eq else 0
        # encoders on category columns
        cat_spec = [("cat", None, False)] * ncols
        Ctr = gen_matrix(rng, ntr, cat_spec, 0.0)
        Cte = gen_matrix(rng, 20, cat_spec, 0.0)
        for r in Cte[:3]:
            r[0] = 9.0  # unknown at fit
        g = ours(4, 0, 64, Ctr, Cte)
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(np.array(Ctr))
        want = oh.transform(np.array(Cte))
        ok = check(len(g["test"][0]) == want.shape[1], "OneHotEncoder width")
        if ok:
            for r in range(len(Cte)):
                for c in range(want.shape[1]):
                    check(g["test"][r][c] == float(want[r][c]), "OneHotEncoder value (%d,%d)" % (r, c))
            w = worst.setdefault("OneHotEncoder (exact)", [0.0, 0, 0, 0])
            w[2] += want.size
            w[3] += want.size
        g = ours(5, 0, 64, Ctr, Cte)
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(np.array(Ctr))
        want = oe.transform(np.array(Cte))
        for r in range(len(Cte)):
            for c in range(ncols):
                check(g["test"][r][c] == float(want[r][c]), "OrdinalEncoder value")
        w = worst.setdefault("OrdinalEncoder (exact)", [0.0, 0, 0, 0])
        w[2] += len(Cte) * ncols
        w[3] += len(Cte) * ncols
    print("sklearn: per comparison, the observed maximum error in units of eps * scale, the asserted bound, the values compared, and how many were bit-equal")
    for k in sorted(worst):
        print("  %-48s max %8.2f   bound %4s   n=%-6d bit-equal %d" % (k, worst[k][0], worst[k][1], worst[k][2], worst[k][3]))
    count("sklearn values", sum(w[2] for w in worst.values()))


# ---------------------------------------------------------------- main
def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    allg = "--all" in args
    if allg or "--sums" in args:
        sums_gate()
    if allg or "--plans" in args:
        plans_gate(int(os.environ.get("PIPE_PLANS", "1500")))
    if allg or "--transforms" in args:
        transforms_gate(int(os.environ.get("PIPE_CASES", "2500")))
    if allg or "--pipeline" in args:
        pipeline_gate(int(os.environ.get("PIPE_CHAINS", "600")))
        align_gate()
    if allg or "--file" in args:
        file_gate(int(os.environ.get("PIPE_FILES", "150")))
    if allg or "--sklearn" in args:
        sklearn_gate(int(os.environ.get("PIPE_SKLEARN", "120")))
    print("cases: " + ", ".join("%s %d" % kv for kv in sorted(CASES.items())))
    if FAILS:
        print("%d MISMATCHES" % len(FAILS))
        return 1
    print("oracle_pipe: all cases agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())

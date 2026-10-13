#!/usr/bin/env python3
"""The gates of docs/categorical-encoders.md section 1 that run `enc-check` and `ml` on files: E1 and E4 to E6 (the transformed matrices, the payload bytes
and every refusal against scripts/oracle_enc.py, bit for bit), E2 (row order), E3 (thread count), E6 (the hash against hashlib), E9 (no file before
stage 3), E11 (no trap) and E13 (scikit-learn, end to end). The driver is `build/enc-check` (scripts/build_enc.sh) and the CLI `build/ml`; the expected
values come from the oracle, written from the document and sharing no code with the sources under test (the reading of the CSV is the stage-1 oracle's,
itself judged against the reader by gate C1).

    python3 scripts/gate_enc.py diff [--files N] [--seed S]
    python3 scripts/gate_enc.py order | hash | threads | files | fuzz [--runs N] | sklearn
    python3 scripts/gate_enc.py all

Exit 0 when every count of differences is 0. ENC and ML override the binaries. Machine, load and counts are printed.
"""
import hashlib
import json
import os
import pathlib
import random
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gate_cat as G  # noqa: E402
import oracle_cat as O  # noqa: E402
import oracle_enc as E  # noqa: E402

ENC = os.environ.get("ENC", str(ROOT / "build" / "enc-check"))
ML = os.environ.get("ML", str(ROOT / "build" / "ml"))
TRAPS = {132, 133, 134, 136, 139}
TMP = tempfile.TemporaryDirectory()
WORK = pathlib.Path(TMP.name)
FAST = bool(os.environ.get("ENC_FAST"))
SIMPLE = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def load_info():
    try:
        return os.getloadavg()[0]
    except OSError:
        return -1.0


# ------------------------------------------------------------------ running

def enc_args(cfg, plan_text, extra=()):
    a = ["--target", cfg.target, "--cat", ",".join(cfg.cats), "--max-categories", str(cfg.max_cats), "--max-string-bytes", str(cfg.max_str), "--plan", plan_text]
    if cfg.na:
        a += ["--na-values", b",".join(cfg.na).decode("latin-1")]
    return a + list(extra)


def run_enc(cfg, plan_text, files, timeout=120):
    paths = []
    for i, data in enumerate(files):
        p = WORK / ("e%d.csv" % i)
        p.write_bytes(data)
        paths.append(str(p))
    argv = [ENC] + enc_args(cfg, plan_text)
    if len(paths) == 2:
        argv += ["--predict", paths[1]]
    argv.append(paths[0])
    p = subprocess.run(argv, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.decode("latin-1")


# ------------------------------------------------------------------ plans

def render_step(st, rnd):
    """the text of a step with a random spelling: key order, blanks, defaults written or left out"""
    verb = st["verb"]
    cols = st["cols"] if st["cols"] == "*" else ",".join(c.decode("latin-1") for c in st["cols"])
    keys = [("column", cols)]
    if verb == "scale":
        keys.append(("kind", "minmax"))
    elif verb == "encode":
        keys.append(("kind", st["kind"]))
        if st.get("maxcat", 64) != 64 or rnd.random() < 0.3:
            keys.append(("max-categories", str(st.get("maxcat", 64))))
        if st.get("unknown", 0) or rnd.random() < 0.3:
            keys.append(("unknown", "refuse" if st.get("unknown", 0) else "ignore"))
        if st.get("minc", 1) != 1 or rnd.random() < 0.2:
            keys.append(("min-count", ("%d" % st.get("minc", 1)) if rnd.random() < 0.7 else "0%d" % st.get("minc", 1)))
        if st.get("missing", 0) or rnd.random() < 0.2:
            keys.append(("missing", "zero" if st.get("missing", 0) else "pass"))
    elif verb == "freq":
        for k, v, d in (("max-categories", str(st.get("maxcat", 64)), "64"), ("unknown", "refuse" if st.get("unknown", 0) else "ignore", "ignore"),
                        ("min-count", str(st.get("minc", 1)), "1"), ("missing", "zero" if st.get("missing", 0) else "pass", "pass")):
            if v != d or rnd.random() < 0.5:
                keys.append((k, v))
    elif verb == "hash":
        keys.append(("buckets", str(st["buckets"])))
        if st.get("missing", 0) or rnd.random() < 0.5:
            keys.append(("missing", "zero" if st.get("missing", 0) else "pass"))
    rnd.shuffle(keys)
    sep = rnd.choice([" ", "  ", "\t", " \t "])
    return verb + sep + sep.join("%s=%s" % kv for kv in keys) + rnd.choice(["", " "])


def render_plan(plan, rnd):
    return ";".join(render_step(st, rnd) for st in plan.steps)


def gen_plan(rnd, cnames, nnames, wild=False):
    """a plan over the categorical columns (and now and then a numeric min-max scale and a deliberate mistake)"""
    cn = [c for c in cnames if rnd.random() < 0.85] or list(cnames[:1])
    rnd.shuffle(cn)
    steps = []
    groups = []
    while cn:
        k = rnd.randint(1, len(cn))
        groups.append(sorted(cn[:k]))
        cn = cn[k:]
    for g in groups:
        verb = rnd.choice(["encode", "encode", "freq", "hash"])
        st = {"verb": verb, "cols": g}
        if verb == "encode":
            st["kind"] = rnd.choice(["onehot", "onehot", "label"])
        if verb == "hash":
            st["buckets"] = rnd.choice([2, 3, 7, 16, 100])
        else:
            st["maxcat"] = rnd.choice([64] * 12 + [1, 3, 10])
            st["unknown"] = rnd.choice([0, 0, 0, 1])
            st["minc"] = rnd.choice([1] * 4 + [2, 3, 5, 50])
        st["missing"] = rnd.choice([0, 0, 1])
        steps.append(st)
    if nnames and rnd.random() < 0.3:
        st = {"verb": "scale", "cols": "*"}
        steps.insert(rnd.randint(0, len(steps)), st)
    if wild and rnd.random() < 0.25:
        bad = rnd.choice(["scale-cat", "freq-num", "mixed", "unknown", "reencode", "star-none"])
        if bad == "scale-cat":
            steps.insert(rnd.randint(0, len(steps)), {"verb": "scale", "cols": [rnd.choice(cnames)]})
        elif bad == "freq-num" and nnames:
            steps.insert(rnd.randint(0, len(steps)), {"verb": "freq", "cols": [rnd.choice(nnames)]})
        elif bad == "mixed" and nnames:
            steps.insert(rnd.randint(0, len(steps)), {"verb": "freq", "cols": sorted([rnd.choice(cnames), rnd.choice(nnames)])})
        elif bad == "unknown":
            steps.insert(rnd.randint(0, len(steps)), {"verb": "hash", "cols": [b"nope"], "buckets": 4})
        elif bad == "reencode":
            steps.append({"verb": "freq", "cols": [rnd.choice(cnames)]})
        else:
            steps.append({"verb": "hash", "cols": "*", "buckets": 3})
    return E.Plan(steps)


# ------------------------------------------------------------------ the tables

def plain_names(cfg):
    return all(set(c.encode("latin-1")) <= SIMPLE for c in cfg.cats)


def gen_pair(rnd):
    """(header, rows, cfg, data) of a table whose categorical columns can be named in a plan, and the second file read through the vocabulary of the first"""
    while True:
        header, rows, cfg, data = G.gen_file(rnd)
        if not G.cfg_ok(header, cfg) or not plain_names(cfg) or any(b'"' in h for h in header):
            continue
        cfg.max_pool = 1048576
        cfg.max_catcols = 128
        if rnd.random() < 0.85:
            cfg.max_cats = 1024
            cfg.max_str = 255
        return header, rows, cfg, data


def predict_file(rnd, header, rows, cfg):
    """the categorical columns in another order (by name), the numeric ones in their own order, rows drawn from the first file and strings it does not hold"""
    cats = [h for h in header if h.decode("latin-1") in cfg.cats]
    nums = [h for h in header if h not in cats and h != b"y"]
    order = nums + cats
    rnd.shuffle(cats)
    # numeric columns keep their relative order; the categorical ones go anywhere
    pos = sorted(rnd.sample(range(len(nums) + len(cats)), len(cats)))
    order, ni, ci = [], iter(nums), iter(cats)
    for i in range(len(nums) + len(cats)):
        order.append(next(ci) if i in pos else next(ni))
    idx = {h: i for i, h in enumerate(header)}
    out = []
    for _ in range(rnd.randint(1, 40)):
        base = rnd.choice(rows) if rows else None
        row = []
        for h in order:
            j = idx[h]
            if base is not None and rnd.random() < 0.75:
                row.append(base[j])
            elif h in cats:
                row.append(rnd.choice([b"", b"never-seen", b"NA", "新".encode(), b"x" * 5, base[j] if base is not None else b"q", b"__other__"]))
            else:
                row.append(rnd.choice([b"", b"1", b"-2", b"3.25", b"0.5"]))
        out.append(row)
    return G.write_csv(order, out, random.Random(rnd.random()))


class Count(G.Count):
    pass


def compare_plan(count, label, data, cfg, plan, data2=None, rnd=None):
    rnd = rnd or random.Random(0)
    text = render_plan(plan, rnd)
    lines, status = E.run_plan(data, cfg, plan, data2)
    want = "\n".join(lines) + "\n"
    got_status, got = run_enc(cfg, text, [data] if data2 is None else [data, data2])
    count.runs += 1
    if got_status != status or got != want:
        gl, wl = got.splitlines(), want.splitlines()
        k = next((i for i in range(min(len(gl), len(wl))) if gl[i] != wl[i]), min(len(gl), len(wl)))
        count.note(label, "status %d vs oracle %d; plan %r; first difference at line %d: got %r want %r; data %r" % (got_status, status, text, k, gl[k:k + 1], wl[k:k + 1], data[:100]))
        return False
    return True


def corner_files():
    """(label, data, cfg, plans): the corners of the rules of sections 3.1 to 3.5"""
    out = []

    def cfg_of(cats=("c",), **kw):
        return O.Cfg(cats=list(cats), target="y", max_cats=kw.pop("max_cats", 1024), **kw)

    def table(rows, header=b"id,c,y"):
        return header + b"\n" + b"".join(r + b"\n" for r in rows)

    def P(*steps):
        return E.Plan(list(steps))

    def enc(kind, minc=1, **kw):
        d = {"verb": "encode", "cols": [b"c"], "kind": kind, "minc": minc}
        d.update(kw)
        return d

    # counts N-1, N and N+1 at the threshold, and the bucket's place among the kept strings
    base = table([b"%d,%s,1" % (i, s) for i, s in enumerate([b"x"] * 3 + [b"y"] * 2 + [b"z"] * 1 + [b"A"] * 2 + [b"a"] * 4 + [b"_"] * 3)])
    for m in (1, 2, 3, 4, 5, 6, 7, 100):
        out.append(("rare threshold %d" % m, base, cfg_of(), [P(enc("onehot", m)), P(enc("label", m)), P({"verb": "freq", "cols": [b"c"], "minc": m})]))
    # the real __other__
    ob = table([b"1,__other__,1", b"2,a,1", b"3,a,1", b"4,b,1"])
    for m in (1, 2):
        out.append(("__other__ at min-count %d" % m, ob, cfg_of(), [P(enc("onehot", m)), P({"verb": "freq", "cols": [b"c"], "minc": m})]))
    out.append(("one category", table([b"1,only,1", b"2,only,2"]), cfg_of(), [P(enc("onehot")), P(enc("label")), P({"verb": "freq", "cols": [b"c"]}), P({"verb": "hash", "cols": [b"c"], "buckets": 2})]))
    out.append(("all missing", table([b"1,,1", b"2,,2"]), cfg_of(), [P(enc("onehot")), P(enc("label", missing=1)), P({"verb": "freq", "cols": [b"c"]}), P({"verb": "hash", "cols": [b"c"], "buckets": 5, "missing": 1}), P({"verb": "hash", "cols": [b"c"], "buckets": 5})]))
    out.append(("unicode, nfc and nfd are two", table([b"1,\xc3\xa9,1", b"2,e\xcc\x81,2", b"3,\xc3\xa9,3", "4,日本語,1".encode(), "5,😀,1".encode(), b"6,e,1", "7,а,1".encode(), b"8,a,1"]), cfg_of(), [P(enc("onehot")), P(enc("label")), P({"verb": "freq", "cols": [b"c"]}), P({"verb": "hash", "cols": [b"c"], "buckets": 7})]))
    out.append(("embedded NUL and quotes", table([b'1,"a""b",1', b"2,a\x00b,1", b'3,"x,y",1', b"4,a,1", b'5,"l1\nl2",1']), cfg_of(), [P(enc("onehot")), P({"verb": "hash", "cols": [b"c"], "buckets": 16})]))
    # markers and empty cells
    out.append(("markers", table([b"1,NA,1", b"2,x,1", b"3,,1", b"4,?,1", b"5,y,1", b"6,x,1"]), cfg_of(na=[b"NA", b"?"]), [P(enc("onehot")), P(enc("onehot", missing=1)), P(enc("label", missing=1)), P({"verb": "freq", "cols": [b"c"], "missing": 1}), P({"verb": "freq", "cols": [b"c"]})]))
    # the order of the columns and several columns in one step, in two steps, and a numeric scale in between
    t2 = table([b"1,b,z,1", b"2,a,y,2", b"3,b,y,3", b"4,c,x,4"], header=b"id,c,d,y")
    cfg2 = cfg_of(cats=("c", "d"))
    out.append(("two columns", t2, cfg2, [P({"verb": "encode", "cols": [b"c", b"d"], "kind": "onehot"}), P({"verb": "encode", "cols": [b"d"], "kind": "label"}, {"verb": "scale", "cols": "*"}, {"verb": "freq", "cols": [b"c"]}), P({"verb": "hash", "cols": "*", "buckets": 4}), P({"verb": "freq", "cols": "*"}, {"verb": "scale", "cols": "*"})]))
    # column-kind and unknown column
    out.append(("column kinds", t2, cfg2, [P({"verb": "scale", "cols": [b"c"]}), P({"verb": "freq", "cols": [b"id"]}), P({"verb": "freq", "cols": [b"c", b"id"]}), P({"verb": "encode", "cols": [b"id", b"c"], "kind": "label"}), P({"verb": "hash", "cols": [b"nope"], "buckets": 3}), P({"verb": "encode", "cols": [b"c"], "kind": "label"}, {"verb": "freq", "cols": [b"c"]}), P({"verb": "scale", "cols": [b"c"]})]))
    # limits of the step: the bound on the effective categories
    mc = table([b"%d,v%d,1" % (i, i % 9) for i in range(30)])
    for mx in (8, 9, 10):
        out.append(("max-categories %d" % mx, mc, cfg_of(), [P(enc("onehot", maxcat=mx)), P(enc("onehot", 4, maxcat=mx))]))
    # the reader's bound
    big = table([b"%d,v%d,1" % (i, i) for i in range(65)])
    out.append(("65 distinct strings, reader bound 64", big, cfg_of(max_cats=64), [P(enc("onehot"))]))
    out.append(("65 distinct strings, reader bound raised, step bound 64", big, cfg_of(), [P(enc("onehot")), P(enc("onehot", 2)), P({"verb": "hash", "cols": [b"c"], "buckets": 8})]))
    big2 = table([b"%d,w%04d,1" % (i, i) for i in range(1025)])
    out.append(("1025 distinct strings", big2, cfg_of(), [P(enc("onehot"))]))
    out.append(("1024 distinct strings", table([b"%d,w%04d,1" % (i, i) for i in range(1024)]), cfg_of(), [P(enc("onehot", maxcat=1024)), P({"verb": "hash", "cols": [b"c"], "buckets": 1024}), P(enc("label", maxcat=1024))]))
    return out


def gate_diff(files, seed):
    """E1, E4, E5: the matrices, names, payload bytes, unknown counts and refusals of random and adversarial plans against the oracle"""
    cnt = Count("E1/E4/E5 diff")
    rnd = random.Random(seed)
    refused = fitted = 0
    for label, data, cfg, plans in corner_files():
        for k, plan in enumerate(plans):
            compare_plan(cnt, "%s #%d" % (label, k), data, cfg, plan, None, random.Random(k))
    # the corners with a predict file: unseen strings, markers, empty cells, other order, a hash step, unknown=refuse
    pf = b"id,c,y\n1,x,1\n2,NEVER,1\n3,,1\n4,z,1\n5,__other__,1\n"
    pd = b"id,c\n1,x\n2,NEVER\n3,\n4,z\n5,__other__\n6,y\n"
    for label, data, cfg, plans in corner_files()[:9]:
        if cfg.cats != ["c"]:
            continue
        for k, plan in enumerate(plans):
            compare_plan(cnt, "%s #%d with a predict file" % (label, k), data, cfg, plan, pd, random.Random(k))
    # all-unseen predict files, and the refusal of unknown=refuse (the first cell in column then row order)
    base = b"id,c,d,y\n1,a,p,1\n2,b,q,2\n3,a,p,3\n"
    cfg = O.Cfg(cats=["c", "d"], target="y", max_cats=1024)
    for p2 in (b"id,c,d\n1,z,r\n2,y,s\n", b"id,c,d\n1,a,r\n2,z,p\n", b"id,d,c\n1,p,a\n2,q,b\n", b"id,c,d\n"):
        for plan in (E.Plan([{"verb": "encode", "cols": [b"c", b"d"], "kind": "onehot", "unknown": 0}]), E.Plan([{"verb": "encode", "cols": [b"c", b"d"], "kind": "label", "unknown": 1}]),
                     E.Plan([{"verb": "freq", "cols": "*", "unknown": 1}]), E.Plan([{"verb": "hash", "cols": "*", "buckets": 6}])):
            compare_plan(cnt, "all-unseen predict %r" % p2, base, cfg, plan, p2)
    n = 0
    while n < files:
        header, rows, cfg, data = gen_pair(rnd)
        n += 1
        order = [c.encode("latin-1") for c in cfg.cats]
        nnames = [h for h in header if h.decode("latin-1") not in cfg.cats and h != b"y"]
        plan = gen_plan(rnd, order, nnames, wild=True)
        data2 = predict_file(rnd, header, rows, cfg) if rnd.random() < 0.8 else None
        ok = compare_plan(cnt, "table %d" % n, data, cfg, plan, data2, random.Random(n))
        lines, status = E.run_plan(data, cfg, plan, data2)
        if status == 0:
            fitted += 1
        else:
            refused += 1
    print("E1/E4/E5: %d random tables (%d answered, %d refused by the oracle's verdict), the corner families and the all-unseen predict files, machine %s, load %.2f" % (files, fitted, refused, os.uname().machine, load_info()))
    return cnt.report()


def parse_matrix(text, label):
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if l.startswith(label + " "):
            _, r, c = l.split()
            return lines[i + 1:i + 1 + int(r)]
    return None


def gate_order(files, seed):
    """E2: the payload bytes do not depend on the order of the rows, and the transformed row of a row is the same wherever it is"""
    cnt = Count("E2 row order")
    rnd = random.Random(seed)
    made = accepted = 0
    while made < files:
        header, rows, cfg, _ = gen_pair(rnd)
        if len(rows) < 2:
            continue
        made += 1
        order = [c.encode("latin-1") for c in cfg.cats]
        nnames = [h for h in header if h.decode("latin-1") not in cfg.cats and h != b"y"]
        plan = gen_plan(rnd, order, nnames)
        text = render_plan(plan, random.Random(made))
        base = G.write_csv(header, rows, random.Random(1))
        s0, o0 = run_enc(cfg, text, [base])
        for k in range(5):
            perm = list(range(len(rows)))
            rnd.shuffle(perm)
            data = G.write_csv(header, [rows[p] for p in perm], random.Random(2 + k))
            s1, o1 = run_enc(cfg, text, [data])
            cnt.runs += 1
            if (s0 == 0) != (s1 == 0):
                cnt.note("accepted in one order and refused in another", (text, o0[:80], o1[:80]))
                continue
            if s0 != 0:
                continue
            pay0 = [l for l in o0.splitlines() if l.startswith("payload ")]
            pay1 = [l for l in o1.splitlines() if l.startswith("payload ")]
            if pay0 != pay1:
                cnt.note("payload bytes", (text, pay0[0][:60], pay1[0][:60]))
                continue
            m0, m1 = parse_matrix(o0, "train"), parse_matrix(o1, "train")
            for i, p in enumerate(perm):
                if m1[i] != m0[p]:
                    cnt.note("the transformed row of a row", (text, i, p, m1[i], m0[p]))
                    break
        if s0 == 0:
            accepted += 1
    print("E2: %d tables (%d accepted), 5 seeded permutations each, machine %s, load %.2f" % (made, accepted, os.uname().machine, load_info()))
    return cnt.report()


def gate_hash(seed):
    """E6: the buckets of many strings against hashlib, at several B, including strings whose first 8 bytes of SHA-256 have the top bit set"""
    cnt = Count("E6 hash")
    rnd = random.Random(seed)
    alphabet = ["a", "b", "é", "日", "😀", "z", "Z", " ", "0", " "]
    for B in (2, 3, 7, 16, 1000, 1024):
        for rep in range(2 if FAST else 6):
            strings = set()
            while len(strings) < 900:
                n = rnd.randint(1, 12)
                strings.add("".join(rnd.choice(alphabet) for _ in range(n)).encode())
            base = sorted(strings)[0]
            # strings that differ in one byte
            for s in sorted(strings)[:50]:
                strings.add(s[:-1] + bytes([s[-1] ^ 1]) if (s[-1] ^ 1) < 128 and s[-1] < 128 else s + b"!")
            strings = sorted(strings)
            rows = [b"%d,%s,1" % (i, G.enc_cell(s, rnd, True)) for i, s in enumerate(strings)]
            data = b"id,c,y\n" + b"\n".join(rows) + b"\n"
            cfg = O.Cfg(cats=["c"], target="y", max_cats=1024)
            plan = E.Plan([{"verb": "hash", "cols": [b"c"], "buckets": B}])
            compare_plan(cnt, "B=%d" % B, data, cfg, plan, None, rnd)
            # the oracle's buckets are hashlib's: check the one-hot positions directly as well
            s, out = run_enc(cfg, "hash column=c buckets=%d" % B, [data])
            m = parse_matrix(out, "train")
            cnt.runs += 1
            for i, st in enumerate(strings):
                want = int.from_bytes(hashlib.sha256(st).digest()[:8], "big") % B
                cells = m[i].split()[1:]  # the first column is the numeric id
                got = [j for j, c in enumerate(cells) if c == "3ff0000000000000"]
                if got != [want] or any(c not in ("0000000000000000", "3ff0000000000000") for c in cells):
                    cnt.note("bucket of %r" % st, (got, want))
                    break
    print("E6: hashing against hashlib at B in {2,3,7,16,1000,1024}, 900+ strings a table, machine %s, load %.2f" % (os.uname().machine, load_info()))
    return cnt.report()


def ml_run(args, cwd=None, timeout=120):
    p = subprocess.run([ML] + args, capture_output=True, timeout=timeout, cwd=cwd)
    return p.returncode, p.stdout


def gate_threads():
    """E3: `ml fit --cat ... --split --threads T` for T in 1 2 4 8 16 writes the same standard output"""
    cnt = Count("E3 threads")
    rnd = random.Random(11)
    made = 0
    while made < (3 if FAST else 8):
        header, rows, cfg, _ = gen_pair(rnd)
        if len(rows) < 40:
            continue
        made += 1
        p = WORK / ("t%d.csv" % made)
        p.write_bytes(G.write_csv(header, rows, random.Random(made)))
        cats = ",".join(cfg.cats)
        step = "; ".join("encode column=%s kind=onehot unknown=ignore min-count=%d missing=zero" % (c, rnd.choice([1, 2, 3])) for c in cfg.cats)
        outs = set()
        for t in (1, 2, 4, 8, 16):
            args = ["fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", cats, "--max-categories", "1024", "--step", step + "; scale column=* kind=standard", "--split", "train=0.7", "--seed", str(made), "--threads", str(t), "--output", str(WORK / "never.cml"), str(p)]
            if cfg.na:
                args += ["--na-values", b",".join(cfg.na).decode("latin-1")]
            code, out = ml_run(args)
            outs.add((code, out))
            cnt.runs += 1
        if len(outs) != 1:
            cnt.note("the answer depends on the thread count", (made, len(outs)))
    print("E3: %d tables, 5 thread counts each, machine %s, load %.2f" % (made, os.uname().machine, load_info()))
    return cnt.report()


# ------------------------------------------------------------------ E13: scikit-learn, end to end

def gate_sklearn():
    """E13: generated data with 3 categorical and 2 numeric columns; `ml fit --cat ... --split` against StandardScaler + OneHotEncoder + LogisticRegression on the same
    train and test rows (the rows of the library's seeded split, computed by scripts/oracle_metrics.py). The tolerances were fixed in docs/categorical-encoders.md
    section 1 (E13) before the first run: accuracy equal, ROC-AUC within 1e-6 absolute, log-loss within 1e-6 relative, coefficients within 1e-5 absolute."""
    import warnings
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import oracle_metrics as OM
    warnings.filterwarnings("ignore")
    bad = 0
    n, l2, seed, frac = 1500, 0.1, 7, "0.75"
    for dseed in (1, 2, 3):
        rng = np.random.default_rng(dseed)
        c1 = rng.choice(["amber", "blue", "green", "red"], n, p=[0.1, 0.4, 0.3, 0.2])
        c2 = rng.choice(["c0", "c1", "c2", "c3", "c4", "c5"], n, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05])
        c3 = rng.choice(["x", "y", "z"], n)
        a = rng.normal(10, 3, n)
        b = rng.uniform(-2, 2, n)
        eff1 = {"amber": 0.8, "blue": -0.4, "green": 0.1, "red": 0.5}
        eff2 = {"c0": 0.0, "c1": 0.6, "c2": -0.8, "c3": 1.0, "c4": -0.2, "c5": 0.3}
        eff3 = {"x": 0.0, "y": 0.5, "z": -0.5}
        logit = np.array([eff1[u] + eff2[v] + eff3[w] for u, v, w in zip(c1, c2, c3)]) + 0.4 * (a - 10) / 3 - 0.7 * b + rng.logistic(0, 1, n) * 0.5
        y = (logit > 0).astype(int)
        with tempfile.TemporaryDirectory() as td:
            csvp = pathlib.Path(td) / "d.csv"
            with open(csvp, "w") as f:
                f.write("c1,c2,c3,a,b,y\n")
                for i in range(n):
                    f.write("%s,%s,%s,%r,%r,%d\n" % (c1[i], c2[i], c3[i], float(a[i]), float(b[i]), y[i]))
            n_test = -(-n * 25 // 100)
            mask = np.array(OM.py_train_test(n, n_test, seed, 0))
            tr, te = mask == 0, mask == 1
            step = "scale column=* kind=standard; encode column=c1,c2,c3 kind=onehot unknown=ignore"
            p = subprocess.run([ML, "--root", td, "fit", "--target", "y", "--model", "logistic", "--l2", str(l2), "--iterations", "100", "--cat", "c1,c2,c3", "--step", step,
                                "--split", "train=" + frac, "--seed", str(seed), "--output", "never.cml", "d.csv"], capture_output=True, text=True)
            d = json.loads(p.stdout)
            assert d["ok"], d
            summ = d["data"]["summary"]
            assert not (pathlib.Path(td) / "never.cml").exists()
            names = [c["name"] for c in summ["coefficients"]]
            w = np.array([float(c["value"]) for c in summ["coefficients"]])
            b0 = float(summ["intercept"])
            hold = summ["holdout"]["metrics"]
        num = np.column_stack([a, b])
        sc = StandardScaler().fit(num[tr])
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(np.column_stack([c1, c2, c3])[tr])
        def feats(m):
            return np.hstack([sc.transform(num[m]), enc.transform(np.column_stack([c1, c2, c3])[m])])
        Xtr, Xte = feats(tr), feats(te)
        sk = LogisticRegression(penalty="l2", C=1.0 / (l2 * tr.sum()), solver="lbfgs", tol=1e-12, max_iter=20000).fit(Xtr, y[tr])
        proba = sk.predict_proba(Xte)[:, 1]
        rows = []
        rows.append(("train rows / test rows", float(summ["split"]["train"] != tr.sum() or summ["split"]["test"] != te.sum()), 0.0))
        rows.append(("coefficients and intercept (max abs)", float(np.max(np.abs(np.append(w, b0) - np.append(sk.coef_[0], sk.intercept_[0])))), 1e-5))
        rows.append(("accuracy (difference)", abs(float(hold["accuracy"]) - accuracy_score(y[te], (proba >= 0.5).astype(int))), 0.0))
        rows.append(("roc_auc (absolute)", abs(float(hold["roc_auc"]) - roc_auc_score(y[te], proba)), 1e-6))
        rows.append(("log_loss (relative)", abs(float(hold["log_loss"]) - log_loss(y[te], proba)) / log_loss(y[te], proba), 1e-6))
        for name, err, bound in rows:
            ok = err <= bound
            bad += 0 if ok else 1
            print("data seed %d  %-38s observed %.3e  bound %.0e  %s" % (dseed, name, err, bound, "PASS" if ok else "FAIL"))
        print("data seed %d  held-out: accuracy %s roc_auc %s log_loss %s; sklearn accuracy %.16g roc_auc %.16g log_loss %.16g; %d coefficients (%s ...)" % (
            dseed, hold["accuracy"], hold["roc_auc"], hold["log_loss"], accuracy_score(y[te], (proba >= 0.5).astype(int)), roc_auc_score(y[te], proba), log_loss(y[te], proba), len(w), ",".join(names[:4])))
    print("E13 scikit-learn %s, numpy %s, machine %s: %s" % (__import__("sklearn").__version__, np.__version__, os.uname().machine, "all PASS" if not bad else "%d FAILED" % bad))
    return bad


# ------------------------------------------------------------------ E11: no trap

def garble(rnd, text):
    alphabet = list("abcdefghimnopqrstuvwxyz=,;*#\t 0123456789-_.é")
    t = list(text)
    for _ in range(rnd.randint(1, 4)):
        i = rnd.randrange(len(t) + 1)
        k = rnd.randint(0, 3)
        if k == 0 and t:
            del t[min(i, len(t) - 1)]
        elif k == 1:
            t.insert(i, rnd.choice(alphabet))
        elif k == 2 and t:
            t[min(i, len(t) - 1)] = rnd.choice(alphabet)
        else:
            j = rnd.randrange(len(t) + 1)
            t[i:i] = t[j:j + rnd.randint(1, 12)]
    return "".join(t)


def gate_fuzz(runs, seed):
    """E11: no plan and no file reaches a trap; every answer of `ml` is one valid JSON line, every answer of `enc-check` a known status"""
    cnt = Count("E11 no trap")
    rnd = random.Random(seed)
    valid = []
    for _ in range(30):
        header, rows, cfg, data = gen_pair(rnd)
        valid.append((header, rows, cfg, data))
    bad = []
    n = 0
    (WORK / "f.csv").write_bytes(b"")
    for i in range(runs):
        header, rows, cfg, data = rnd.choice(valid)
        order = [c.encode("latin-1") for c in cfg.cats]
        nnames = [h for h in header if h.decode("latin-1") not in cfg.cats and h != b"y"]
        plan = gen_plan(rnd, order, nnames, wild=True)
        text = render_plan(plan, rnd)
        if rnd.random() < 0.4:
            text = garble(rnd, text)
        t = rnd.random()
        if t < 0.6:
            body = data
        elif t < 0.9:
            body = G.mutate(rnd, data)
        else:
            body = rnd.randbytes(rnd.randint(0, 300))
        (WORK / "f.csv").write_bytes(body)
        cats = ",".join(cfg.cats)
        try:
            p = subprocess.run([ENC, "--target", "y", "--cat", cats, "--max-categories", "1024", "--plan", text, "--predict", str(WORK / "f.csv"), str(WORK / "f.csv")], capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            bad.append(("enc-check timeout", text[:60]))
            continue
        n += 1
        if p.returncode in TRAPS or p.returncode < 0 or p.returncode not in (0, 2, 3, 8):
            bad.append(("enc-check", p.returncode, text[:80], p.stdout[:60]))
        if i % 2 == 0:
            extra = rnd.choice([["--split", "train=0.6"], ["--dry-run"], []])
            model = rnd.choice(["linear", "logistic"])
            try:
                q = subprocess.run([ML, "--root", str(WORK), "fit", "--target", "y", "--model", model, "--l2", "1", "--cat", cats, "--max-categories", "1024", "--step", text, "--output", "m.cml", "--overwrite"] + extra + ["f.csv"], capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                bad.append(("ml timeout", text[:60]))
                continue
            n += 1
            if not G.ml_ok(q.returncode, q.stdout):
                bad.append(("ml", q.returncode, q.stdout[:100], text[:80]))
    big = [("4097 categorical columns hashed", b",".join(b"c%d" % i for i in range(4097)) + b",y\n" + b",".join(b"v" for _ in range(4098)) + b"\n", ",".join("c%d" % i for i in range(4097)), "hash column=* buckets=2", ["--max-cat-columns", "4096"]),
           ("a cell of 70000 bytes", b"c0,y\n" + b"z" * 70000 + b",1\n", "c0", "encode column=c0 kind=onehot", ["--max-string-bytes", "65536"]),
           ("2^20 distinct values", b"c0,y\n" + b"".join(b"%d,1\n" % i for i in range(1 << 20)), "c0", "encode column=c0 kind=onehot", []),
           ("1025 distinct values at the ceiling", b"c0,y\n" + b"".join(b"%d,1\n" % i for i in range(1025)), "c0", "encode column=c0 kind=onehot", ["--max-categories", "1024"]),
           ("1024 distinct values and 1024 buckets", b"c0,y\n" + b"".join(b"%d,%d\n" % (i, i % 2) for i in range(1024)), "c0", "hash column=c0 buckets=1024", ["--max-categories", "1024"]),
           ("all cells missing", b"c0,y\n" + b",1\n" * 50, "c0", "encode column=c0 kind=onehot", []),
           ("one category, 50 buckets", b"c0,y\n" + b"a,1\n" * 50, "c0", "hash column=c0 buckets=50", []),
           ("invalid UTF-8", b"c0,y\n\xff,1\n", "c0", "freq column=c0", []),
           ("a quoted cell with a million doubled quotes", b'c0,y\n"' + b'""' * 500000 + b'",1\n', "c0", "freq column=c0", ["--max-string-bytes", "65536"]),
           ("a plan of 64 lines", b"c0,y\n" + b"a,1\nb,2\n", "c0", ";".join(["scale column=nope kind=minmax"] * 64), []),
           ("a plan over the line limit", b"c0,y\n" + b"a,1\nb,2\n", "c0", ";".join(["freq column=c0"] * 70), []),
           ("a million empty cells", b"c0,y\n" + b",1\n" * 1000000, "c0", "encode column=c0 kind=onehot missing=zero", [])]
    for label, data, cats, text, fl in big:
        (WORK / "f.csv").write_bytes(data)
        q = subprocess.run([ML, "--root", str(WORK), "fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", cats, "--step", text, "--output", "m.cml", "--overwrite", "--dry-run", "--max-line-bytes", "16777216"] + fl + ["f.csv"], capture_output=True, timeout=300)
        r = subprocess.run([ENC, "--target", "y", "--cat", cats, "--plan", text, "--max-categories", "1024"] + [x for x in fl if x != "--max-cat-columns" and x != "4096"] + [str(WORK / "f.csv")], capture_output=True, timeout=300)
        n += 2
        if not G.ml_ok(q.returncode, q.stdout):
            bad.append((label, "ml", q.returncode, q.stdout[:100]))
        if r.returncode in TRAPS or r.returncode < 0 or r.returncode not in (0, 2, 3, 8):
            bad.append((label, "enc-check", r.returncode, r.stdout[:60]))
    cnt.runs = n
    for b in bad[:15]:
        cnt.note("TRAP OR BAD ANSWER", b)
    print("E11: %d runs (enc-check and ml fit --cat with hostile plans and files), machine %s, load %.2f" % (n, os.uname().machine, load_info()))
    return cnt.report()


def main():
    args = sys.argv[1:]
    mode = args[0] if args else "all"
    files = 60 if FAST else 400
    seed = 1
    runs = 300 if FAST else 2000
    if "--files" in args:
        files = int(args[args.index("--files") + 1])
    if "--seed" in args:
        seed = int(args[args.index("--seed") + 1])
    if "--runs" in args:
        runs = int(args[args.index("--runs") + 1])
    bad = 0
    table = {"diff": lambda: gate_diff(files, seed), "order": lambda: gate_order(max(10, files // 8), seed), "hash": lambda: gate_hash(seed), "threads": gate_threads}
    for extra in ("files", "fuzz", "sklearn", "plan"):
        if extra in globals() or ("gate_" + extra) in globals():
            pass
    if "gate_files" in globals():
        table["files"] = globals()["gate_files"]
    if "gate_fuzz" in globals():
        table["fuzz"] = lambda: globals()["gate_fuzz"](runs, seed)
    if "gate_sklearn" in globals():
        table["sklearn"] = globals()["gate_sklearn"]
    if "gate_plan" in globals():
        table["plan"] = globals()["gate_plan"]
    names = list(table) if mode == "all" else [mode]
    for n in names:
        bad += table[n]()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The gates of docs/categorical.md section 1 that run the reader and `ml` on files: C1 and C6 (vocabulary, ids and refusals against scripts/oracle_cat.py),
C2 (row order), C3 (thread count), C4 (limits, before allocation), C5 (UTF-8), C7 (round trip through a stored vocabulary) and C8 (no trap). The reader is
`build/reader-check` (scripts/build_reader.sh) and the CLI `build/ml`; the expected values come from the oracle, written from the document and sharing no code
with the sources under test.

    python3 scripts/gate_cat.py diff [--files N] [--seed S]
    python3 scripts/gate_cat.py order | threads | limits | utf8 | roundtrip | fuzz [--runs N]
    python3 scripts/gate_cat.py all

Exit 0 when every count of differences is 0. READER and ML override the binaries. Machine, load and counts are printed.
"""
import hashlib
import json
import os
import pathlib
import random
import resource
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import oracle_cat as O  # noqa: E402

READER = os.environ.get("READER", str(ROOT / "build" / "reader-check"))
ML = os.environ.get("ML", str(ROOT / "build" / "ml"))
TRAPS = {132, 133, 134, 136, 139}
TMP = tempfile.TemporaryDirectory()
WORK = pathlib.Path(TMP.name)


def load_info():
    try:
        return os.getloadavg()[0]
    except OSError:
        return -1.0


# ------------------------------------------------------------------ running

def rc_args(cfg, extra=()):
    a = ["--mode", "cat"]
    if cfg.cats:
        a += ["--cat", ",".join(cfg.cats)]
    if cfg.na:
        a += ["--na-values", b",".join(cfg.na).decode("latin-1")]
    a += ["--max-categories", str(cfg.max_cats), "--max-string-bytes", str(cfg.max_str), "--max-pool-bytes", str(cfg.max_pool), "--max-cat-columns", str(cfg.max_catcols)]
    if cfg.drops:
        a += ["--drop", ",".join(cfg.drops)]
    if cfg.floats:
        a += ["--float", ",".join(cfg.floats)]
    if cfg.target:
        a += ["--target", cfg.target]
    return a + list(extra)


def rc(args, files, timeout=120):
    """run reader-check on the files (bytes); (status, stdout as latin-1 text)"""
    paths = []
    for i, data in enumerate(files):
        p = WORK / ("f%d.csv" % i)
        p.write_bytes(data)
        paths.append(str(p))
    argv = [READER] + args
    if len(paths) == 2:
        argv += ["--predict", paths[1]]
    argv.append(paths[0])
    p = subprocess.run(argv, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.decode("latin-1")


class Count:
    def __init__(self, label):
        self.label, self.runs, self.diffs, self.first = label, 0, 0, []

    def note(self, what, detail):
        self.diffs += 1
        if len(self.first) < 6:
            self.first.append((what, detail))

    def report(self):
        print("%s: %d runs, %d differences" % (self.label, self.runs, self.diffs))
        for what, detail in self.first:
            print("   DIFF", what, detail)
        return self.diffs


def compare(count, label, data, cfg, data2=None, skip2=False):
    """one file (and a second one through the vocabulary of the first): the reader's lines against the oracle's"""
    lines, status = O.run(data, cfg, data2, skip2)
    want = "\n".join(lines) + "\n"
    extra = ["--predict-skip"] if skip2 else []
    got_status, got = rc(rc_args(cfg, extra), [data] if data2 is None else [data, data2])
    count.runs += 1
    if got_status != status or got != want:
        gl, wl = got.splitlines(), want.splitlines()
        k = next((i for i in range(min(len(gl), len(wl))) if gl[i] != wl[i]), min(len(gl), len(wl)))
        count.note(label, "status %d vs oracle %d; first difference at line %d: got %r want %r; data %r" % (got_status, status, k, gl[k:k + 1], wl[k:k + 1], data[:120]))
        return False
    return True


# ------------------------------------------------------------------ the files

WORDS = ["male", "female", "S", "C", "Q", "NA", "none", "Pclass", "A", "B", "a", "b", "yes", "no", "x" * 40]
UNI = ["café", "café", "日本語", "😀", "👨‍👩‍👧", "Ωmega", "Привет", "a", "а", "ａ", "é", "é", "نص", " ", "﻿bom", "ß", "ǅ"]
AWKWARD = ["a,b", 'say "hi"', "two\nlines", "crlf\r\nbreak", "tab\there", 'q"', '"', '""', "a\rb", ",", "\n", " ", "  ", " a", "a ", " a ", "007", "7", "7.0", "1e5", "-1", "NA ", " NA", "na", "?", "N/A"]
NUL = ["a\x00b", "\x00", "x\x00"]


def needs_quote(s):
    return b"," in s or b'"' in s or b"\n" in s or b"\r" in s or s == b""


def enc_cell(s, rnd, force=False):
    if needs_quote(s) or force or (rnd.random() < 0.15 and not s.startswith(b" ")):
        return b'"' + s.replace(b'"', b'""') + b'"'
    return s


def write_csv(header, rows, rnd, eol=b"\n", bom=False, trailing=True, blanks=False):
    out = bytearray(b"\xef\xbb\xbf" if bom else b"")
    lines = [b",".join(enc_cell(h, rnd) for h in header)]
    for r in rows:
        lines.append(b",".join(enc_cell(c, rnd) for c in r))
        if blanks and rnd.random() < 0.1:
            lines.append(b"")
    out += eol.join(lines)
    if trailing:
        out += eol
    # a quoted newline is written as the file's own line break only where the cell had one: the cell bytes are exactly what the generator put
    return bytes(out)


def pool_for(rnd):
    kind = rnd.choice(["words", "uni", "awkward", "mixed", "mixed", "nul"])
    base = {"words": WORDS, "uni": UNI, "awkward": AWKWARD, "mixed": WORDS + UNI + AWKWARD, "nul": WORDS + NUL}[kind]
    return [s.encode() for s in base]


def gen_table(rnd):
    """(header, rows, cfg, extras): 1 to 4 categorical columns, a numeric id, a numeric x, a target y"""
    ncat = rnd.randint(1, 4)
    names = [("c%d" % i).encode() for i in range(ncat)]
    if rnd.random() < 0.2:
        names = [rnd.choice([b"caf\xc3\xa9", b"\xe6\x97\xa5", b"a b", b"x,y", b'q"q']) + str(i).encode() for i in range(ncat)]
    pools = [pool_for(rnd) for _ in range(ncat)]
    card = [rnd.choice([1, 2, 3, 5, 12, 70]) for _ in range(ncat)]
    na = [m for m in (b"NA", b"?", b"na", b"N/A") if rnd.random() < 0.3]
    rows = []
    n = rnd.choice([0, 1, 2, 5, 30, 120, 300])
    emptyc = rnd.random() < 0.15
    for r in range(n):
        row = [str(r).encode()]
        for c in range(ncat):
            u = rnd.random()
            if emptyc and c == 0:
                row.append(b"")
            elif u < 0.1:
                row.append(b"")
            elif u < 0.15 and na:
                row.append(rnd.choice(na))
            else:
                p = pools[c]
                row.append(p[rnd.randrange(min(card[c], len(p)))] if card[c] <= len(p) else p[rnd.randrange(len(p))] + b"%d" % rnd.randrange(card[c]))
        row.append(rnd.choice([b"", b"1", b"-2", b"3.25", b"10.5", b"0.001"]))
        row.append(rnd.choice([b"1", b"2.5", b"-0.5", b"100"]))
        rows.append(row)
    header = [b"id"] + names + [b"x", b"y"]
    cfg = O.Cfg(cats=[n.decode("latin-1") for n in names], na=na, target="y",
                max_cats=rnd.choice([64] * 7 + [2, 4, 1024]), max_str=rnd.choice([255] * 7 + [3, 6, 65536]),
                max_pool=rnd.choice([1048576] * 7 + [30, 100, 300]), max_catcols=rnd.choice([128] * 8 + [1, 2]))
    return header, rows, cfg


def csv_names(cfg):
    return cfg


def gen_file(rnd):
    header, rows, cfg = gen_table(rnd)
    eol = rnd.choice([b"\n", b"\n", b"\r\n"])
    data = write_csv(header, rows, rnd, eol=eol, bom=rnd.random() < 0.1, trailing=rnd.random() < 0.9, blanks=rnd.random() < 0.2)
    return header, rows, cfg, data


# CSV with names that need quoting cannot be named on the command line by comma lists: such tables use cats without a comma in the name
def cfg_ok(header, cfg):
    for n in cfg.cats:
        if "," in n:
            return False
    return True


def adversarial():
    """(label, data, cfg): the files the issue names and the corners of the cell rules"""
    out = []

    def add(label, rows, na=(), header=b"id,c,y", cats=("c",), **kw):
        body = b"".join(r + b"\n" for r in rows)
        out.append((label, header + b"\n" + body, O.Cfg(cats=cats, na=list(na), target="y", **kw)))

    add("unicode and emoji", [b"1,caf\xc3\xa9,1", "2,日本語,2".encode(), "3,😀,3".encode(), b"4,e\xcc\x81,4", b"5,\xc3\xa9,5", b"6,\xef\xbb\xbfbom,6"])
    add("nfc and nfd are two", [b"1,\xc3\xa9,1", b"2,e\xcc\x81,2", b"3,\xc3\xa9,3"])
    add("embedded NUL", [b"1,a\x00b,1", b'2,"x\x00",2', b"3,\x00,3", b"4,a,4"])
    add("quotes", [b'1,"a""b",1', b'2,"a,b",2', b'3,ab"c,3', b'4,"""",4', b'5,"",5', b'6,"x""",6', b'7,"""x",7'])
    add("quoted line breaks", [b'1,"l1\nl2",1', b'2,"l1\r\nl2",2', b'3,"a\rb",3', b'4,"\n",4', b'5,"\r\n",5', b'6,l1,6'])
    add("duplicate-looking strings", [b"1,a,1", b"2,a ,2", b"3, a,3", b"4,A,4", "5,ａ,5".encode(), "6,а,6".encode(), b"7,\xc2\xa0,7", b"8,a\t,8"])
    add("whitespace is kept", [b"1, ,1", b"2,  ,2", b'3," ",3', b"4,\t,4", b'5," x ",5', b"6, x ,6"])
    add("all empty", [b"1,,1", b'2,"",2', b"3,,3"])
    add("one category", [b"1,only,1", b"2,only,2", b"3,only,3"])
    add("markers", [b"1,NA,1", b"2,?,2", b"3,na,3", b"4, NA,4", b'5,"NA",5', b"6,NAN,6", b"7,N,7", b"8,x,8"], na=[b"NA", b"?"])
    add("a marker with a space", [b"1,not available,1", b"2,x,2"], na=[b"not available"])
    add("numeric-looking categories", [b"1,007,1", b"2,7,2", b"3,7.0,3", b"4,1e5,4", b"5,-1,5", b"6,+7,6", b"7,0x7,7"])
    add("bom in the file", [b"1,a,1", b"2,b,2"], header=b"\xef\xbb\xbfid,c,y")
    add("two boms", [b"1,a,1"], header=b"\xef\xbb\xbf\xef\xbb\xbfid,c,y", cats=("\xef\xbb\xbfid".encode("latin-1").decode("latin-1"),) if False else ("c",))
    add("limit exactly on categories", [b"%d,v%d,1" % (i, i) for i in range(4)], max_cats=4)
    add("limit one over on categories", [b"%d,v%d,1" % (i, i) for i in range(5)], max_cats=4)
    add("string at the byte limit", [b"1," + b"x" * 8 + b",1"], max_str=8)
    add("string one over the byte limit", [b"1," + b"x" * 9 + b",1"], max_str=8)
    add("multibyte string at the limit", [b"1,\xe6\x97\xa5\xe6\x9c\xac,1"], max_str=6)
    add("multibyte string over the limit", [b"1,\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e,1"], max_str=6)
    add("pool exactly full", [b"1,aaaa,1", b"2,bbbb,1", b"3,aaaa,1"], max_pool=8)
    add("pool one over", [b"1,aaaa,1", b"2,bbbb,1", b"3,c,1"], max_pool=8)
    add("long strings", [b"1," + b"y" * 255 + b",1", b"2," + b"y" * 254 + b"z,1"])
    add("a string of 256 bytes", [b"1," + b"y" * 256 + b",1"])
    add("a string of 4096 bytes with the limit raised", [b"1," + b"y" * 4096 + b",1"], max_str=4096)
    add("a string of 65536 bytes at the ceiling", [b"1," + b"y" * 65536 + b",1"], max_str=65536)
    add("a cell both too long and not UTF-8 is a long cell", [b"1," + b"\xff" * 300 + b",1"])
    add("a cell both too long and not UTF-8, limit raised, is invalid", [b"1," + b"\xff" * 300 + b",1"], max_str=400)
    add("a 70000-byte string", [b"1," + b"y" * 70000 + b",1"], max_str=65536)
    add("last line without a newline", [b"1,a,1", b"2,b,2"])
    out[-1] = (out[-1][0], out[-1][1][:-1], out[-1][2])
    add("invalid utf8 late", [b"1,a,1", b"2,\xff,2"])
    add("blank lines between rows", [b"1,a,1", b"", b"2,b,2", b"", b""])
    add("unterminated quote", [b'1,"abc,1'])
    add("bad quote", [b'1,"abc"x,1'])
    add("ragged row", [b"1,a,1", b"2,b"])
    add("an undeclared text column", [b"1,abc,1"], cats=())
    add("empty file body", [])
    add("cat names a missing column", [b"1,a,1"], cats=("c", "nope"))
    add("cat names the target", [b"1,a,1"], cats=("y",))
    add("cat names a dropped column", [b"1,a,1"], cats=("c",), drops=["c"])
    add("cat names a float column", [b"1,a,1"], cats=("c",), floats=["c"])
    add("two cat columns, the second breaks the pool", [b"1,aaaa,bbbb,1"], header=b"id,c,d,y", cats=("c", "d"), max_pool=7)
    add("more cat columns than the limit", [b"1,a,b,1"], header=b"id,c,d,y", cats=("c", "d"), max_catcols=1)
    add("cat columns in a file order that differs from the flag order", [b"1,a,z,1", b"2,b,y,2"], header=b"id,c,d,y", cats=("d", "c"))
    return out


# ------------------------------------------------------------------ the gates

def gate_diff(files, seed):
    """C1 and C6: vocabulary bytes, ids, the missing set, and every refusal with its line, column and aux, against the oracle"""
    cnt = Count("C1/C6 diff")
    rnd = random.Random(seed)
    for label, data, cfg in adversarial():
        compare(cnt, "adversarial " + label, data, cfg)
    n_acc = n_ref = 0
    made = 0
    while made < files:
        header, rows, cfg, data = gen_file(rnd)
        if not cfg_ok(header, cfg):
            continue
        made += 1
        ok = compare(cnt, "random file %d" % made, data, cfg)
        lines, status = O.run(data, cfg)
        if status == 0:
            n_acc += 1
        else:
            n_ref += 1
    print("C1/C6: %d adversarial files and %d seeded random files (%d accepted, %d refused by an oracle verdict), machine %s, load %.2f" % (len(adversarial()), files, n_acc, n_ref, os.uname().machine, load_info()))
    return cnt.report()


def parse_out(text):
    """the vocabulary lines, the ids by row and the numeric cells by row of a `--mode cat` output (fit part only)"""
    vocab, ids, cells = [], {}, {}
    refused = False
    for line in text.splitlines():
        if line.startswith("p-"):
            break
        if line.startswith(("vocab", "vcol", "cat ", "dump")):
            vocab.append(line)
        elif line.startswith("id "):
            _, r, c, v, b = line.split()
            ids.setdefault(int(r), {})[int(c)] = (int(v), int(b))
        elif line.startswith("cell "):
            _, r, c, v = line.split()
            cells.setdefault(int(r), {})[int(c)] = v
        elif line.startswith("refused"):
            refused = True
    return refused, vocab, ids, cells


def gate_order(files, seed):
    """C2: the vocabulary and the id of every row do not depend on the order of the rows"""
    cnt = Count("C2 row order")
    rnd = random.Random(seed)
    made = 0
    accepted = 0
    while made < files:
        header, rows, cfg, _ = gen_file(rnd)
        if not cfg_ok(header, cfg) or len(rows) < 2:
            continue
        made += 1
        base = write_csv(header, rows, random.Random(1))
        s0, o0 = rc(rc_args(cfg), [base])
        r0, v0, i0, c0 = parse_out(o0)
        for k in range(5):
            perm = list(range(len(rows)))
            rnd.shuffle(perm)
            data = write_csv(header, [rows[p] for p in perm], random.Random(2 + k))
            s1, o1 = rc(rc_args(cfg), [data])
            r1, v1, i1, c1 = parse_out(o1)
            cnt.runs += 1
            if r0 != r1:
                cnt.note("accepted in one order and refused in another", (cfg.max_cats, cfg.max_str, cfg.max_pool, cfg.max_catcols, o0[:80], o1[:80]))
                continue
            if r0:
                continue
            if v0 != v1:
                cnt.note("vocabulary", (v0[:3], v1[:3]))
            for i, p in enumerate(perm):
                if i1.get(i, {}) != i0.get(p, {}) or c1.get(i, {}) != c0.get(p, {}):
                    cnt.note("ids or cells of a row", (i, p, i1.get(i), i0.get(p)))
                    break
        if not r0:
            accepted += 1
    print("C2: %d tables (%d accepted at every order), 5 seeded permutations each, machine %s, load %.2f" % (made, accepted, os.uname().machine, load_info()))
    return cnt.report()


def gate_threads():
    """C3: `ml fit --cat` writes the same bytes at every thread count, the digest in them is the oracle's, and the reader's dumps are the same on repeated runs"""
    cnt = Count("C3 threads")
    rnd = random.Random(7)
    done = 0
    for n in range(12):
        header, rows, cfg = gen_table(rnd)
        names = [h.decode("latin-1") for h in header[1:-2]]
        if any("," in x or '"' in x for x in names):
            continue
        cfg = O.Cfg(cats=names, na=cfg.na, target="y")
        data = write_csv(header, rows, random.Random(3))
        if not rows:
            continue
        (WORK / "t.csv").write_bytes(data)
        outs = set()
        digest = None
        for t in (1, 2, 4, 8, 16):
            argv = [ML, "--root", str(WORK), "fit", "--model", rnd.choice(["linear", "logistic"]) if False else "linear", "--target", "y", "--output", "m.cml", "--overwrite", "--cat", ",".join(names), "--threads", str(t)]
            if cfg.na:
                argv += ["--na-values", b",".join(cfg.na).decode()]
            argv.append("t.csv")
            p = subprocess.run(argv, capture_output=True)
            cnt.runs += 1
            outs.add((p.returncode, p.stdout))
        if len(outs) != 1:
            cnt.note("thread count changed the answer", len(outs))
        (code, out), = list(outs)[:1]
        lines, status = O.run(data, cfg)
        doc = json.loads(out.decode())
        if status == 0:
            vocab = None
            L = O.load(data, cfg)
            want = hashlib.sha256(O.dump(L.vocab)).hexdigest()
            got = doc["error"]["detail"].get("vocabulary_sha256") if not doc["ok"] else None
            if doc["error"]["rule"] != "ml.input.categorical-not-encoded" or got != want:
                cnt.note("digest", (doc["error"]["rule"], got, want))
        done += 1
        # the reader, five runs
        a = set()
        for _ in range(5):
            a.add(rc(rc_args(cfg), [data]))
            cnt.runs += 1
        if len(a) != 1:
            cnt.note("reader dumps differ between runs", len(a))
    print("C3: %d tables at 5 thread counts through `ml fit --cat` and 5 runs of the reader each, machine %s, load %.2f" % (done, os.uname().machine, load_info()))
    return cnt.report()


def peak_rss_mb(args, path):
    code = ("import resource, subprocess, sys; p = subprocess.run(sys.argv[1:], capture_output=True); "
            "r = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss; print(p.returncode, r, p.stdout.decode('latin-1').splitlines()[:1])")
    out = subprocess.run([sys.executable, "-c", code, READER] + args + [str(path)], capture_output=True).stdout.decode().strip()
    rcode, rss, first = out.split(" ", 2)
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return int(rcode), int(rss) * scale / 1e6, first


def ml_tag(args, td=None):
    p = subprocess.run([ML, "--root", str(td or WORK)] + args, capture_output=True)
    doc = json.loads(p.stdout.decode())
    return p.returncode, (doc["error"]["rule"] if not doc["ok"] else None), doc


def gate_limits():
    """C4: each limit is exact (at the limit accepted, one over refused with the right tag, line and column), and a refusal is reached before anything sized by
    the file exists: the peak memory of the refusal of a 200 MB file"""
    cnt = Count("C4 limits")
    # exactness, against the oracle and through the CLI
    cases = []
    for k in (1, 2, 3, 64, 65, 1024, 1025):
        rows = [b"%d,v%05d,1" % (i, i) for i in range(k)]
        cases.append(("categories %d" % k, b"id,c,y\n" + b"\n".join(rows) + b"\n", O.Cfg(cats=["c"], target="y", max_cats=k - 1 if k > 1 else 1), {"max-categories": str(k - 1 if k > 1 else 1)}))
        cases.append(("categories %d at the limit" % k, b"id,c,y\n" + b"\n".join(rows) + b"\n", O.Cfg(cats=["c"], target="y", max_cats=k if k <= 1024 else 1024), {"max-categories": str(k if k <= 1024 else 1024)}))
    for n in (1, 5, 255, 256, 1000, 65535, 65536):
        for lim in (n - 1, n):
            if lim < 1:
                continue
            data = b"id,c,y\n1," + b"s" * n + b",1\n"
            cases.append(("string %d limit %d" % (n, lim), data, O.Cfg(cats=["c"], target="y", max_str=lim), {"max-string-bytes": str(lim)}))
    for pool in (7, 8, 9):
        data = b"id,c,d,y\n1,aaaa,bbbb,1\n2,aaaa,bbbb,1\n"
        cases.append(("pool %d" % pool, data, O.Cfg(cats=["c", "d"], target="y", max_pool=pool), {"max-pool-bytes": str(pool)}))
    for ncols, lim in ((3, 2), (3, 3), (4, 3), (1, 1)):
        header = b"id," + b",".join(b"c%d" % i for i in range(ncols)) + b",y"
        data = header + b"\n1," + b",".join(b"v" for _ in range(ncols)) + b",1\n"
        cases.append(("catcols %d limit %d" % (ncols, lim), data, O.Cfg(cats=["c%d" % i for i in range(ncols)], target="y", max_catcols=lim), {"max-cat-columns": str(lim)}))
    expect_tag = {22: "ml.limit.categories", 21: "ml.limit.string-bytes", 23: "ml.limit.string-pool", 24: "ml.limit.categorical-columns"}
    for label, data, cfg, flags in cases:
        compare(cnt, label, data, cfg)
        lines, status = O.run(data, cfg)
        (WORK / "l.csv").write_bytes(data)
        argv = ["fit", "--target", "y", "--output", "m.cml", "--overwrite", "--cat", ",".join(cfg.cats)]
        for f, v in flags.items():
            argv += ["--" + f, v]
        code, rule, doc = ml_tag(argv + ["l.csv"])
        cnt.runs += 1
        want_rule = "ml.input.categorical-not-encoded"
        if status != 0:
            want_rule = expect_tag[int(lines[0].split()[1])]
        if rule != want_rule:
            cnt.note("CLI tag", (label, rule, want_rule))
    # before allocation: a refusal of a very large file stays small
    big = WORK / "big.csv"
    n = 2800000
    t0 = time.time()
    with open(big, "wb") as f:
        f.write(b"id,c,y\n")
        for start in range(0, n, 100000):
            f.write(b"".join(b"%d,value-%032d-padding-padding-padding,1\n" % (i, i) for i in range(start, min(n, start + 100000))))
    size = big.stat().st_size
    code, mb, first = peak_rss_mb(["--mode", "cat", "--cat", "c", "--target", "y"], big)
    cnt.runs += 1
    print("   a %.0f MB file of %d distinct values in one categorical column: exit %d, %s, peak %.1f MB (16 allowed)" % (size / 1e6, n, code, first, mb))
    if code != 8 or mb > 16 or " 22 " not in first:
        cnt.note("200 MB file", (code, mb, first))
    code, mb, first = peak_rss_mb(["--mode", "cat", "--cat", "c", "--target", "y", "--max-categories", "1024"], big)
    cnt.runs += 1
    print("   the same with --max-categories 1024: exit %d, %s, peak %.1f MB" % (code, first, mb))
    if code != 8 or mb > 16:
        cnt.note("200 MB file at 1024", (code, mb, first))
    # the oracle agrees on the line of the refusal of the first 70 rows (the whole file is not read by Python)
    head = b"id,c,y\n" + b"".join(b"%d,value-%032d-padding,1\n" % (i, i) for i in range(2000))
    compare(cnt, "2000 distinct", head, O.Cfg(cats=["c"], target="y"))
    # 2^20 distinct values
    with open(big, "wb") as f:
        f.write(b"id,c,y\n")
        f.write(b"".join(b"%d,%d,1\n" % (i, i) for i in range(1 << 20)))
    s, out = rc(rc_args(O.Cfg(cats=["c"], target="y")), [big.read_bytes()])
    cnt.runs += 1
    if s != 8 or not out.startswith("refused 22 66 1 64"):
        cnt.note("2^20 distinct values", (s, out[:60]))
    s, out = rc(rc_args(O.Cfg(cats=["c"], target="y", max_cats=1024)), [big.read_bytes()])
    cnt.runs += 1
    if s != 8 or not out.startswith("refused 22 1026 1 1024"):
        cnt.note("2^20 distinct values at 1024", (s, out[:60]))
    # the shape caps count the ids (every header column counts, dropped ones too): 3 rows of 4 columns, two of them categorical, is 12 cells
    data = b"a,b,c,d\n1,x,2,y\n3,x,4,y\n5,x,6,y\n"
    for cells, want in ((12, 0), (11, 8)):
        p = subprocess.run([READER, "--mode", "cat", "--cat", "b,d", "--max-cells", str(cells), str(big)], capture_output=True)
        (WORK / "s.csv").write_bytes(data)
        p = subprocess.run([READER, "--mode", "cat", "--cat", "b,d", "--max-cells", str(cells), str(WORK / "s.csv")], capture_output=True)
        cnt.runs += 1
        if p.returncode != want or (want and not p.stdout.startswith(b"refused 13 ")):
            cnt.note("cells count the ids", (cells, p.returncode, p.stdout[:60]))
    try:
        big.unlink()
    except OSError:
        pass
    print("C4: machine %s, load %.2f, %.0f s" % (os.uname().machine, load_info(), time.time() - t0))
    return cnt.report()


def gate_utf8(seed):
    """C5: well-formed or not, and the offset of the first bad byte, against Python's strict decoder"""
    cnt = Count("C5 utf8")
    rnd = random.Random(seed)
    cands = []
    for cp in (0x00, 0x01, 0x7F, 0x80, 0x7FF, 0x800, 0xD7FF, 0xE000, 0xFFFD, 0xFFFF, 0x10000, 0x10FFFF):
        cands.append(chr(cp).encode("utf-8", "surrogatepass"))
    raw = [b"\x80", b"\xbf", b"\xc0\x80", b"\xc1\xbf", b"\xe0\x80\x80", b"\xe0\x9f\xbf", b"\xed\xa0\x80", b"\xed\xbf\xbf", b"\xf0\x80\x80\x80", b"\xf0\x8f\xbf\xbf",
           b"\xf4\x90\x80\x80", b"\xf5\x80\x80\x80", b"\xf8\x88\x80\x80\x80", b"\xfe", b"\xff", b"\xc3", b"\xe2\x82", b"\xf0\x9f\x98", b"caf\xe9", b"ok\xc3(", b"a\xe2\x28\xa1",
           b"\xf0\x28\x8c\xbc", b"\xc2\x41", b"\xe1\x80\xe2", b"\xef\xbb\xbf", b"\xef\xbf\xbe", b"\xf4\x8f\xbf\xbf"]
    cands += raw
    for lead in range(0x80, 0x100):
        for second in (0x00, 0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0):
            cands.append(bytes([lead, second]))
            cands.append(bytes([lead, second, 0x80]))
            cands.append(bytes([lead, second, 0x80, 0x80]))
    for _ in range(1500):
        cands.append(bytes(rnd.choice([rnd.randrange(256), rnd.randrange(0x80, 0xC0), rnd.randrange(0xC2, 0xF5)]) for _ in range(rnd.randint(1, 6))))
    seen = set()
    cands = [c for c in cands if not (c in seen or seen.add(c)) and b"\n" not in c and b"\r" not in c and b"," not in c and b'"' not in c and len(c) > 0]
    valid = [c for c in cands if O.utf8_offset(c) < 0]
    invalid = [c for c in cands if O.utf8_offset(c) >= 0]
    # the valid ones together (many categories), the invalid ones one at a time
    for i in range(0, len(valid), 40):
        chunk = valid[i:i + 40]
        data = b"id,c,y\n" + b"".join(b"%d," % j + c + b",1\n" for j, c in enumerate(chunk))
        compare(cnt, "valid chunk %d" % i, data, O.Cfg(cats=["c"], target="y", max_cats=64))
    for c in invalid:
        data = b"id,c,y\n1,ok,1\n2," + c + b",2\n"
        compare(cnt, "invalid %r" % c, data, O.Cfg(cats=["c"], target="y"))
        data = b"id,c,y\n1," + b"a" * 3 + c + b",1\n"
        compare(cnt, "invalid after ascii %r" % c, data, O.Cfg(cats=["c"], target="y"))
    # an invalid cell in a column that is not declared is the old refusal, and in a dropped column no refusal at all
    s, out = rc(["--mode", "cat", "--drop", "c", "--target", "y"], [b"id,c,y\n1,\xff,1\n"])
    cnt.runs += 1
    if s != 0:
        cnt.note("dropped column read", out[:60])
    print("C5: %d valid strings (in chunks) and %d invalid ones (each alone and after ASCII), %d runs, machine %s, load %.2f" % (len(valid), len(invalid), cnt.runs, os.uname().machine, load_info()))
    return cnt.report()


def gen_predict(rnd, header, rows, cfg):
    """a second file with the columns of the first in another order, rows drawn from the first and strings it does not have"""
    names = [h for h in header if h not in (b"y",)]
    order = names[:]
    rnd.shuffle(order)
    idx = {h: i for i, h in enumerate(header)}
    out = []
    for _ in range(rnd.randint(1, 40)):
        base = rnd.choice(rows) if rows else None
        row = []
        for h in order:
            j = idx[h]
            if base is not None and rnd.random() < 0.7:
                row.append(base[j])
            elif h.startswith(b"c") or True:
                row.append(rnd.choice([b"", b"never-seen", b"NA", "新".encode(), b"x" * 5, base[j] if base is not None else b"q"]))
        out.append(row)
    return order, out


def gate_roundtrip(files, seed):
    """C7: a vocabulary built from one file and given to the read of another"""
    cnt = Count("C7 round trip")
    rnd = random.Random(seed)
    made = 0
    selfok = 0
    while made < files:
        header, rows, cfg = gen_table(rnd)
        if not rows or not cfg_ok(header, cfg) or any(b'"' in h for h in header):
            continue
        cfg = O.Cfg(cats=cfg.cats, na=cfg.na, target="y", max_cats=1024, max_pool=1048576)
        data = write_csv(header, rows, random.Random(5))
        lines, status = O.run(data, cfg)
        if status != 0:
            continue
        made += 1
        order, prow = gen_predict(rnd, header, rows, cfg)
        d2 = write_csv(order, prow, random.Random(6))
        compare(cnt, "predict file %d" % made, data, cfg, d2, skip2=True)
        # without the skip, the target column is read as numbers: the file has none, so it is a column like the others
        # the first file through its own vocabulary: the ids of the fit
        s, out = rc(rc_args(cfg, ["--predict-skip"]), [data, data])
        cnt.runs += 1
        fit_ids = [l for l in out.splitlines() if l.startswith("id ")]
        pre_ids = [l[2:] for l in out.splitlines() if l.startswith("p-id ")]
        if fit_ids != pre_ids:
            cnt.note("a file read through its own vocabulary", (fit_ids[:3], pre_ids[:3]))
        else:
            selfok += 1
        # a vocabulary column the predict file lacks
        if cfg.cats:
            drop = cfg.cats[0].encode()
            hh = [h for h in order if h != drop]
            pr = [[c for h, c in zip(order, r) if h != drop] for r in prow]
            compare(cnt, "predict file lacks a column", data, cfg, write_csv(hh, pr, random.Random(7)), skip2=True)
    # a predict file is checked as a fit file is: the cells of a vocabulary column are refused when too long or not UTF-8, a marker is missing, and nothing is added
    fit_data = b"id,c,y\n1,a,1\n2,b,2\n3,,3\n4,NA,4\n"
    base_cfg = O.Cfg(cats=["c"], na=[b"NA"], target="y")
    for label, pred in (("invalid", b"id,c\n1,a\n2,\xff\n"), ("long", b"id,c\n1,a\n2," + b"z" * 300 + b"\n"), ("nul and quotes", b'id,c\n1,"a""b"\n2,a\x00\n3,b\n'),
                        ("markers", b"id,c\n1,NA\n2,\n3,na\n4,b\n"), ("unseen only", b"id,c\n1,x\n2,y\n3,x\n"), ("no rows", b"id,c\n"), ("column in another place", b"c,id\na,1\nb,2\n"),
                        ("target present and garbage", b"id,c,y\n1,a,zzz\n"), ("dropped column", b"id,c\n1,a\n")):
        compare(cnt, "predict " + label, fit_data, base_cfg, pred, skip2=True)
    print("C7: %d tables, a predict file each (another column order, unseen strings, markers, missing columns) and the file through its own vocabulary (%d agree), machine %s, load %.2f" % (made, selfok, os.uname().machine, load_info()))
    return cnt.report()


def mutate(rnd, base):
    b = bytearray(base)
    for _ in range(rnd.randint(1, 4)):
        if not b:
            b = bytearray(rnd.randbytes(rnd.randint(1, 20)))
        i = rnd.randrange(len(b))
        t = rnd.randint(0, 6)
        if t == 0:
            b[i] = rnd.randrange(256)
        elif t == 1:
            del b[i:i + rnd.randint(1, 8)]
        elif t == 2:
            b[i:i] = rnd.choice([b'"', b",", b"\n", b"\r", b"\x00", b"\xff", b"\xc3", b'""', "😀".encode(), b"\xed\xa0\x80", b"NA"])
        elif t == 3:
            b = b[:i]
        elif t == 4:
            j = rnd.randrange(len(b))
            b[i:i] = b[j:j + rnd.randint(1, 30)]
        elif t == 5:
            b[i:i] = rnd.randbytes(rnd.randint(1, 10))
        else:
            b[i:i + 1] = b'"' * rnd.randint(1, 5)
    return bytes(b)


def ml_ok(code, out):
    if code in TRAPS or code < 0:
        return False
    lines = out.decode("utf-8", "replace").splitlines()
    if len(lines) != 1:
        return False
    try:
        d = json.loads(lines[0])
    except ValueError:
        return False
    return d.get("ok") == (code in (0, 9)) and code in (0, 1, 2, 3, 4, 5, 8, 9)


def valid_reader(code, out):
    if code == 0:
        return "rows " in out
    if code == 8:
        return out.startswith("refused ") or out.startswith("refused-matrix ")
    return False


def gate_fuzz(runs, seed):
    """C8: nothing a file holds reaches a trap"""
    cnt = Count("C8 no trap")
    rnd = random.Random(seed)
    valid = []
    for _ in range(40):
        header, rows, cfg, data = gen_file(rnd)
        valid.append(data)
    bad = []
    t0 = time.time()
    n = 0
    flagsets = [["--cat", "c0"], ["--cat", "c0,c1", "--na-values", "NA,?"], ["--cat", "c0", "--max-categories", "2", "--max-string-bytes", "4", "--max-pool-bytes", "20"],
                ["--cat", "c0,c1,c2,c3", "--max-cat-columns", "2"]]
    (WORK / "f.csv").write_bytes(b"")
    for i in range(runs):
        t = rnd.random()
        if t < 0.15:
            data = rnd.randbytes(rnd.randint(0, 2000))
        elif t < 0.75:
            data = mutate(rnd, rnd.choice(valid))
        elif t < 0.85:
            data = b"id,c0,c1,c2,c3,x,y\n" + b"".join(b"1," + bytes(rnd.randrange(256) for _ in range(rnd.randint(0, 12))).replace(b"\n", b"").replace(b",", b"") + b",,,,1,2\n" for _ in range(rnd.randint(1, 20)))
        elif t < 0.93:
            data = b"id,c0,y\n" + b"".join(b"1," + bytes(rnd.choice(b'",\n\r a\xc3\xa9') for _ in range(rnd.randint(1, 30))) + b",2\n" for _ in range(rnd.randint(1, 10)))
        else:
            data = rnd.choice(valid)[:rnd.randint(0, 120)]
        (WORK / "f.csv").write_bytes(data)
        fl = rnd.choice(flagsets)
        try:
            p = subprocess.run([READER, "--mode", "cat", "--target", "y", "--max-line-bytes", "16777216"] + fl + [str(WORK / "f.csv")], capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            bad.append(("reader timeout", data[:50]))
            continue
        n += 1
        out = p.stdout.decode("latin-1")
        if p.returncode in TRAPS or p.returncode < 0 or not valid_reader(p.returncode, out):
            bad.append(("reader", p.returncode, out[:60], data[:60]))
        if i % 2 == 0:
            try:
                p2 = subprocess.run([ML, "--root", str(WORK), "fit", "--target", "y", "--output", "m.cml", "--overwrite"] + fl + ["f.csv"], capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                bad.append(("ml timeout", data[:50]))
                continue
            n += 1
            if not ml_ok(p2.returncode, p2.stdout):
                bad.append(("ml", p2.returncode, p2.stdout[:80], data[:60]))
    # the big ones
    big = [("4097 categorical columns", b",".join(b"c%d" % i for i in range(4097)) + b",y\n" + b",".join(b"v" for _ in range(4098)) + b"\n", ["--cat", ",".join("c%d" % i for i in range(4097))]),
           ("a cell of 70000 bytes", b"c0,y\n" + b"z" * 70000 + b",1\n", ["--cat", "c0"]),
           ("a cell of 65536 bytes", b"c0,y\n" + b"z" * 65536 + b",1\n", ["--cat", "c0", "--max-string-bytes", "65536"]),
           ("2^20 distinct values", b"c0,y\n" + b"".join(b"%d,1\n" % i for i in range(1 << 20)), ["--cat", "c0"]),
           ("invalid UTF-8 in every column", b"c0,c1,c2,y\n" + b"\xff,\xfe,\xc3,1\n", ["--cat", "c0,c1,c2"]),
           ("a million quotes", b"c0,y\n" + b'"' * 1000000 + b"\n", ["--cat", "c0"]),
           ("NULs", b"c0,y\n\x00\x00,1\n\x00,2\n", ["--cat", "c0"]),
           ("a quoted cell with a million doubled quotes", b'c0,y\n"' + b'""' * 500000 + b'",1\n', ["--cat", "c0", "--max-string-bytes", "65536"]),
           ("a million empty cells", b"c0,y\n" + b",1\n" * 1000000, ["--cat", "c0"])]
    for label, data, fl in big:
        (WORK / "f.csv").write_bytes(data)
        p = subprocess.run([READER, "--mode", "cat", "--target", "y", "--max-line-bytes", "16777216", "--max-cat-columns", "4096"] + fl + [str(WORK / "f.csv")], capture_output=True, timeout=300)
        n += 1
        out = p.stdout.decode("latin-1")
        if p.returncode in TRAPS or p.returncode < 0 or not valid_reader(p.returncode, out):
            bad.append((label, "reader", p.returncode, out[:60]))
        p2 = subprocess.run([ML, "--root", str(WORK), "fit", "--target", "y", "--output", "m.cml", "--overwrite", "--max-line-bytes", "16777216", "--max-cat-columns", "4096"] + fl + ["f.csv"], capture_output=True, timeout=300)
        n += 1
        if not ml_ok(p2.returncode, p2.stdout):
            bad.append((label, "ml", p2.returncode, p2.stdout[:80]))
    cnt.runs = n
    for b in bad[:15]:
        cnt.note("TRAP OR BAD ANSWER", b)
    print("C8: %d runs (reader-check --mode cat and ml fit --cat), machine %s, load %.2f, %.0f s" % (n, os.uname().machine, load_info(), time.time() - t0))
    return cnt.report()


def main():
    argv = sys.argv[1:]
    names = ("diff", "order", "threads", "limits", "utf8", "roundtrip", "fuzz", "all")
    if not argv or argv[0] not in names:
        print(__doc__)
        return 2

    def opt(name, default):
        return int(argv[argv.index(name) + 1]) if name in argv else default

    seed = opt("--seed", 1)
    which = argv[0]
    bad = 0
    if which in ("diff", "all"):
        bad += gate_diff(opt("--files", 1500), seed)
    if which in ("order", "all"):
        bad += gate_order(opt("--files", 150), seed)
    if which in ("threads", "all"):
        bad += gate_threads()
    if which in ("limits", "all"):
        bad += gate_limits()
    if which in ("utf8", "all"):
        bad += gate_utf8(seed)
    if which in ("roundtrip", "all"):
        bad += gate_roundtrip(opt("--files", 150), seed)
    if which in ("fuzz", "all"):
        bad += gate_fuzz(opt("--runs", 2400), seed)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

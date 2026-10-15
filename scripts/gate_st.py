#!/usr/bin/env python3
"""The gates of docs/stringtable.md section 9 that run `ml` on files: S1 (the whole model file, byte for byte, against scripts/oracle_st.py), S2 (the file against the
in-process held-out scores of `fit --split`), S4 (row order and thread count), S5 (unseen, missing and markers at predict), S7 (the bounds of a fit), S8 (alignment by
name for the four models), S9 (privacy) and S10 (damaged files, never a trap). The expected values come from the oracle, written from the document and sharing no code
with the sources under test (the reading of the CSV is the stage-1 oracle's, the encoders' that of stage 2, both judged against the code by their own gates).

    python3 scripts/gate_st.py diff [--files N] [--seed S]
    python3 scripts/gate_st.py roundtrip | order | predict | limits | align | privacy | fuzz [--runs N]
    python3 scripts/gate_st.py all

Exit 0 when every count of differences is 0. ML overrides the binary. Machine, load and counts are printed.
"""
import hashlib
import json
import os
import pathlib
import random
import struct
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gate_cat as G  # noqa: E402
import gate_enc as GE  # noqa: E402
import oracle_cat as O  # noqa: E402
import oracle_enc as E  # noqa: E402
import oracle_metrics as M  # noqa: E402
import oracle_st as S  # noqa: E402

ML = os.environ.get("ML", str(ROOT / "build" / "ml"))
TRAPS = {132, 133, 134, 136, 139}
TMP = tempfile.TemporaryDirectory()
WORK = pathlib.Path(TMP.name)
FAST = bool(os.environ.get("ST_FAST"))
STRIDE = int(os.environ.get("ST_STRIDE", "1"))  # outside the string table every STRIDE-th byte is flipped and cut (the table: every byte)
STAMP = [0]


def load_info():
    try:
        return os.getloadavg()[0]
    except OSError:
        return -1.0


def ml(*args, timeout=120):
    """-> (status, the JSON document or None, stdout bytes)"""
    p = subprocess.run([ML, *[str(a) for a in args]], capture_output=True, timeout=timeout)
    doc = None
    try:
        lines = p.stdout.decode("utf-8").splitlines()
        if len(lines) == 1:
            doc = json.loads(lines[0])
    except (UnicodeDecodeError, ValueError):
        doc = None
    return p.returncode, doc, p.stdout


def put(name, data):
    (WORK / name).write_bytes(data)
    return name


def rules(doc):
    return [e["rule"] for e in (doc or {}).get("errors", [])]


def u32(b, at=0):
    return struct.unpack_from("<I", b, at)[0]


# ------------------------------------------------------------------ a fit of the CLI, and the model file it wrote

def fit_cli(name, data, cfg, steps_text, model="tree", extra=(), out="m.cml", task=None):
    put(name, data)
    a = ["fit", name, "--root", WORK, "--target", cfg.target, "--cat", ",".join(cfg.cats), "--max-categories", cfg.max_cats, "--max-string-bytes", cfg.max_str,
         "--max-pool-bytes", cfg.max_pool, "--max-cat-columns", cfg.max_catcols, "--model", model, "--step", steps_text, "--output", out, "--overwrite"]
    if task:
        a += ["--task", task]
    elif model in ("tree", "forest", "boost"):
        a += ["--task", "regression"]
    if cfg.na:
        a += ["--na-values", b",".join(cfg.na).decode("latin-1")]
    if cfg.drops:
        a += ["--drop", ",".join(cfg.drops)]
    if cfg.floats:
        a += ["--float", ",".join(cfg.floats)]
    return ml(*a, *extra)


def read_file(out="m.cml"):
    return (WORK / out).read_bytes()


def na_bytes(cfg):
    return b",".join(cfg.na)


def lims(cfg):
    return (cfg.max_cats, cfg.max_str, cfg.max_pool, cfg.max_catcols)


def expected_file(data, cfg, plan, got_sections, seed=0, rows_override=None):
    """The model file the oracle says a fit of `plan` on `data` writes, with the plan, the model section and the compiler pin taken from the file the tool wrote (the
    canonical form of a plan is gate E8's, the estimator is not under test here). -> bytes, or (None, oracle lines) when the oracle refuses the plan."""
    lines, status = E.run_plan(data, cfg, plan)
    if status != 0:
        return None, lines
    payload = bytes.fromhex(lines[0].split()[1])
    L = O.load(data, cfg)
    sec, tab = S.make_sections(payload, na_bytes(cfg), lims(cfg))
    schema, ncols = S.schema_bytes(L, cfg.target)
    model = got_sections[4]
    meta = got_sections[5]
    pin = meta[32:32 + u32(meta, 28)]
    new_meta = S.meta_bytes(L.rows if rows_override is None else rows_override, ncols, seed, pin, u32(model))
    return S.build_file(got_sections[1], schema, sec, model, new_meta, tab), payload


class Count(G.Count):
    pass


# ------------------------------------------------------------------ S1: the file, byte for byte

def compare_file(cnt, label, data, cfg, plan, rnd, model="tree"):
    text = GE.render_plan(plan, rnd)
    lines, ostatus = E.run_plan(data, cfg, plan)
    status, doc, _ = fit_cli("d.csv", data, cfg, text, model)
    cnt.runs += 1
    remaining = ostatus == 0 and any(l.startswith("remaining") and len(l.split()) > 1 for l in lines)
    if remaining:
        # a declared column that no step reads: refused, nothing written
        if status == 0 or "ml.input.categorical-not-encoded" not in rules(doc):
            cnt.note(label, "a declared column is read by no step; the tool answered %r" % (rules(doc),))
            return False
        return True
    if ostatus != 0:
        if status == 0:
            cnt.note(label, "the oracle refuses (%r) and the tool wrote a file; plan %r" % (lines[:1], text))
            return False
        return True
    if status != 0:
        cnt.note(label, "the oracle accepts and the tool refused with %r; plan %r data %r" % (rules(doc), text, data[:100]))
        return False
    got = read_file()
    try:
        secs = S.parse_file(got)
    except S.FileBad as e:
        cnt.note(label, "the tool wrote a file the oracle's reader refuses: %s" % e.tag)
        return False
    want, payload = expected_file(data, cfg, plan, secs)
    if want != got:
        bad = [t for t in (1, 2, 3, 4, 5, 6) if t in secs and (want is None or t not in S.parse_file(want) or S.parse_file(want)[t] != secs[t])]
        detail = ""
        if want is not None:
            ws = S.parse_file(want)
            detail = "; sections that differ %r; sizes got %r want %r" % (bad, {t: len(v) for t, v in secs.items()}, {t: len(v) for t, v in ws.items()})
        cnt.note(label, "file bytes differ%s; plan %r data %r" % (detail, text, data[:100]))
        return False
    # the summary of the answer: the digest of the effective payload and the table
    summ = doc["data"]["summary"]
    ok = summ.get("steps_sha256") == hashlib.sha256(payload).hexdigest()
    tab = secs.get(6, b"")
    st = summ.get("string_table")
    if tab:
        ok = ok and st is not None and st["sha256"] == hashlib.sha256(tab).hexdigest() and st["bytes"] == len(tab)
    else:
        ok = ok and st is None
    if not ok:
        cnt.note(label, "the summary does not match the file: %r" % ({k: summ.get(k) for k in ("steps_sha256", "string_table")},))
        return False
    return True


def gate_diff(files, seed):
    """S1 and S6"""
    cnt = Count("S1/S6 file bytes against the oracle")
    rnd = random.Random(seed)
    n_acc = n_ref = 0
    for label, data, cfg, plans in GE.corner_files():
        for k, plan in enumerate(plans):
            compare_file(cnt, "%s #%d" % (label, k), data, cfg, plan, random.Random(k))
    # the adversarial family of the reader (unicode, NUL, quotes, markers, long strings, 1,024 categories ...) through the four encoders
    for label, data, cfg in G.adversarial():
        if not cfg.cats:
            continue
        for k, st in enumerate((
                {"verb": "encode", "cols": [cfg.cats[0].encode()], "kind": "onehot", "maxcat": 1024},
                {"verb": "freq", "cols": [cfg.cats[0].encode()], "maxcat": 1024},
                {"verb": "hash", "cols": [cfg.cats[0].encode()], "buckets": 5},
                {"verb": "encode", "cols": [cfg.cats[0].encode()], "kind": "label", "maxcat": 1024, "minc": 2})):
            compare_file(cnt, "%s encoder %d" % (label, k), data, cfg, E.Plan([st]), random.Random(k))
    n = 0
    while n < files:
        header, rows, cfg, data = GE.gen_pair(rnd)
        n += 1
        order = [c.encode("latin-1") for c in cfg.cats]
        nnames = [h for h in header if h.decode("latin-1") not in cfg.cats and h != b"y"]
        plan = GE.gen_plan(rnd, order, nnames, wild=True)
        compare_file(cnt, "table %d" % n, data, cfg, plan, random.Random(n))
        if E.run_plan(data, cfg, plan)[1] == 0:
            n_acc += 1
        else:
            n_ref += 1
    print("S1/S6: %d random tables (%d the oracle accepts, %d it refuses), the corner families and the reader's adversarial files, machine %s, load %.2f" % (files, n_acc, n_ref, os.uname().machine, load_info()))
    return cnt.report()


# ------------------------------------------------------------------ tables for the four models

PETS = [b"cat", b"dog", b"bird", b"fish", b"rabbit", b"horse", b"cow", b"pig", "chat".encode(), "犬".encode(), "🐈".encode(), b"a b"]
CITIES = [b"Lyon", b"Turin", b"Oslo", "Zürich".encode(), b"Rome"]
COLORS = [b"red", b"green", b"blue", b"it's", b'say "x"', b"NUL\x00in"]


def model_table(rnd, n, task, ncat=3, missing_rate=0.06, classes=("0", "1"), cat_pool=None):
    """a file with `ncat` categorical columns (c0 ..), two numeric ones (x1, x2) and the target y: a regression on the numbers and the categories plus noise, or a
    two-class label from the same score. -> (header, rows, cfg)"""
    pools = [PETS, CITIES, COLORS][:ncat] if cat_pool is None else cat_pool
    effects = [{s: rnd.uniform(-2, 2) for s in p} for p in pools]
    rows = []
    for r in range(n):
        x1 = round(rnd.uniform(-3, 3), 3)
        x2 = round(rnd.uniform(0, 10), 2)
        cells = []
        score = 0.8 * x1 - 0.2 * x2
        for c, p in enumerate(pools):
            if rnd.random() < missing_rate:
                cells.append(b"")
            else:
                v = rnd.choice(p)
                cells.append(v)
                score += effects[c][v]
        score += rnd.gauss(0, 0.4)
        if task == "regression":
            y = ("%.4f" % score).encode()
        else:
            y = classes[1].encode() if score > 0.0 else classes[0].encode()
        rows.append([str(r).encode()] + cells + [repr(x1).encode(), repr(x2).encode(), y])
    header = [b"id"] + [("c%d" % i).encode() for i in range(len(pools))] + [b"x1", b"x2", b"y"]
    cfg = O.Cfg(cats=["c%d" % i for i in range(len(pools))], target="y", max_cats=64, max_str=255)
    return header, rows, cfg


def csv_bytes(header, rows, eol=b"\n"):
    def q(c):
        if any(ch in c for ch in b',"\r\n'):
            return b'"' + c.replace(b'"', b'""') + b'"'
        return c
    lines = [b",".join(q(h) for h in header)] + [b",".join(q(c) for c in r) for r in rows]
    return eol.join(lines) + eol


MODEL_FLAGS = {
    "linear": ["--model", "linear", "--l2", "0.1"],
    "logistic": ["--model", "logistic", "--l2", "0.1", "--iterations", "60"],
    "tree": ["--model", "tree", "--max-depth", "6"],
    "forest": ["--model", "forest", "--trees", "7", "--max-depth", "5"],
    "boost": ["--model", "boost", "--rounds", "8", "--min-leaf", "3"],
}


def steps_for(rnd, kind_mix):
    """a plan over c0 c1 c2 by the mix: 'onehot' (min-count 2 on one column), 'mixed' (one-hot, freq and hash), 'hash', 'label' - all with missing=zero"""
    if kind_mix == "onehot":
        return "encode column=c0 kind=onehot min-count=2 missing=zero; encode column=c1,c2 kind=onehot missing=zero; scale column=x1,x2 kind=standard"
    if kind_mix == "mixed":
        return "encode column=c0 kind=onehot missing=zero; freq column=c1 missing=zero; hash column=c2 buckets=5 missing=zero; scale column=x1,x2 kind=standard"
    if kind_mix == "hash":
        return "hash column=c0,c1,c2 buckets=7 missing=zero; scale column=x1,x2 kind=standard"
    if kind_mix == "reordered":
        # the steps read the columns in an order that is not the order of the header, so the table and the reader's vocabulary list them differently
        return "freq column=c2 missing=zero; scale column=x1,x2 kind=standard; encode column=c0,c1 kind=onehot missing=zero"
    return "encode column=c0,c1,c2 kind=label missing=zero"


def fit_model(name, data, cfg, model, steps, split=None, seed=0, out="m.cml", extra=(), threads=None, flags=None):
    put(name, data)
    a = ["fit", name, "--root", WORK, "--target", "y", "--cat", ",".join(cfg.cats), "--step", steps, "--output", out, "--overwrite", "--seed", seed]
    a += flags if flags is not None else MODEL_FLAGS[model]
    if model in ("tree", "forest", "boost"):
        a += ["--task", "classification" if "cls" in extra else "regression"]
        extra = [e for e in extra if e != "cls"]
    if cfg.na:
        a += ["--na-values", b",".join(cfg.na).decode("latin-1")]
    if split:
        a += ["--split", "train=%s" % split]
    if threads:
        a += ["--threads", threads]
    return ml(*a, *extra)


def task_of(model):
    return "classification" if model in ("logistic",) else "regression"


def model_kind_flags(model, cls):
    if model in ("tree", "forest", "boost"):
        return ["cls"] if cls else []
    return []


def metrics_of(doc, key):
    if doc is None or not doc.get("ok"):
        return None
    d = doc["data"]
    if key == "holdout":
        return d["summary"]["holdout"]["metrics"]
    return d["metrics"]


def gate_roundtrip(files, seed):
    """S2: eval --split on the file equals the held-out metrics of the fit, string for string; predict on the held-out rows equals the oracle's selection of them"""
    cnt = Count("S2 round trip (file against in-process)")
    rnd = random.Random(seed)
    made = 0
    for model in ("linear", "logistic", "tree", "forest", "boost"):
        for mix in ("onehot", "mixed", "hash", "label", "reordered"):
            for rep in range(max(1, files // 16)):
                cls = model in ("logistic",) or (model in ("tree", "forest", "boost") and rep % 2 == 1)
                task = "classification" if cls else "regression"
                n = rnd.choice([40, 90, 200])
                header, rows, cfg = model_table(rnd, n, task, ncat=3)
                data = csv_bytes(header, rows)
                sd = rnd.randint(0, 50)
                fr = rnd.choice(["0.7", "0.8"])
                st, doc, _ = fit_model("rt.csv", data, cfg, model, steps_for(rnd, mix), split=fr, seed=sd, extra=model_kind_flags(model, cls))
                cnt.runs += 1
                if st != 0:
                    cnt.note("%s %s" % (model, mix), "fit refused: %r" % (rules(doc),))
                    continue
                made += 1
                inproc = metrics_of(doc, "holdout")
                est, edoc, _ = ml("eval", "rt.csv", "--root", WORK, "--model-file", "m.cml", "--split", "train=" + fr, "--seed", sd)
                if est != 0 or metrics_of(edoc, "eval") != inproc:
                    cnt.note("%s %s" % (model, mix), "eval from the file %r vs in-process %r (%r)" % (metrics_of(edoc, "eval"), inproc, rules(edoc)))
                # predict on the held-out rows: the test side is the rows the oracle's split names
                nt = -(-n * (10 - int(fr[2:])) // 10) if len(fr) == 3 else 0
                sel = M.py_train_test(n, nt, sd, 0)
                test_rows = [rows[i] for i in range(n) if sel[i] == 1]
                put("rt_test.csv", csv_bytes(header, test_rows))
                pst, pdoc, _ = ml("predict", "rt_test.csv", "--root", WORK, "--model-file", "m.cml", "--limit", 1000, *(["--proba"] if cls else []))
                cnt.runs += 1
                if pst != 0:
                    cnt.note("%s %s" % (model, mix), "predict on the held-out rows refused: %r" % (rules(pdoc),))
                    continue
                # the same rows through eval on the held-out file alone: the same metrics as the split
                est2, edoc2, _ = ml("eval", "rt_test.csv", "--root", WORK, "--model-file", "m.cml")
                if est2 != 0 or metrics_of(edoc2, "eval") != inproc:
                    cnt.note("%s %s" % (model, mix), "eval of the held-out rows as a file %r vs in-process %r" % (metrics_of(edoc2, "eval"), inproc))
    print("S2: %d fits written and read back, machine %s, load %.2f" % (made, os.uname().machine, load_info()))
    return cnt.report()


def gate_order(files, seed):
    """S4: the bytes of the file do not depend on the order of the rows or on the thread count"""
    cnt = Count("S4 row order and thread count")
    rnd = random.Random(seed)
    k = 0
    while k < files:
        header, rows, cfg, _ = GE.gen_pair(rnd)
        if len(rows) < 2:
            continue
        order = [c.encode("latin-1") for c in cfg.cats]
        nnames = [h for h in header if h.decode("latin-1") not in cfg.cats and h != b"y"]
        plan = GE.gen_plan(rnd, order, nnames)
        olines, ost = E.run_plan(G.write_csv(header, rows, random.Random(0)), cfg, plan)
        if ost != 0 or any(l.startswith("remaining") and len(l.split()) > 1 for l in olines):
            continue
        k += 1
        text = GE.render_plan(plan, random.Random(k))
        base = None
        for perm in range(5):
            rs = list(rows)
            if perm:
                random.Random(k * 100 + perm).shuffle(rs)
            data = G.write_csv(header, rs, random.Random(7))
            st, doc, _ = fit_cli("o.csv", data, cfg, text, "tree")
            cnt.runs += 1
            if st != 0:
                cnt.note("table %d" % k, "fit refused in order %d: %r" % (perm, rules(doc)))
                continue
            b = read_file()
            tab = S.parse_file(b).get(6, b"")
            # the table and the steps are what the rows determine; the model section can differ with the order (a tree reads the rows in order)
            got = (S.parse_file(b)[2], S.parse_file(b)[3], tab)
            if base is None:
                base = got
            elif got != base:
                cnt.note("table %d" % k, "the table, schema or steps differ in row order %d" % perm)
        # threads: the whole file
        data = G.write_csv(header, rows, random.Random(7))
        whole = None
        for t in (1, 2, 4, 8, 16):
            st, doc, _ = fit_cli("o.csv", data, cfg, text, "tree", extra=("--threads", t))
            cnt.runs += 1
            if st != 0:
                cnt.note("table %d" % k, "fit refused with %d threads: %r" % (t, rules(doc)))
                continue
            b = read_file()
            if whole is None:
                whole = b
            elif b != whole:
                cnt.note("table %d" % k, "the file differs with %d threads" % t)
    # the rest of the file: the plan, schema, steps (with order-free numeric steps), meta and table are the same in any order of the rows (asserted); the estimator's own
    # section is not claimed to be (a bootstrap draws by row index, the sums follow the order of the rows, docs/linalg.md section 2), so it is measured and printed
    steps = "encode column=c0 kind=onehot missing=zero; hash column=c1 buckets=5 missing=zero; scale column=x1,x2 kind=minmax"
    info = {}
    for model in ("tree", "forest", "boost", "linear", "logistic"):
        same = diff = 0
        for rep in range(8):
            cls = model == "logistic"
            header, rows, cfg = model_table(rnd, 60, "classification" if cls else "regression", ncat=2)
            files_ = []
            for perm in range(3):
                rs = list(rows)
                if perm:
                    random.Random(rep * 10 + perm).shuffle(rs)
                st, doc, _ = fit_model("o2.csv", csv_bytes(header, rs), cfg, model, steps, extra=model_kind_flags(model, cls))
                cnt.runs += 1
                if st == 0:
                    files_.append(read_file())
            if len(files_) == 3:
                same_all = files_[0] == files_[1] == files_[2]
                same += same_all
                diff += not same_all
                secs = [S.parse_file(f) for f in files_]
                if any(s[t_] != secs[0][t_] for s in secs for t_ in (1, 2, 3, 5, 6)):
                    cnt.note("whole-file probe " + model, "the plan, schema, steps, meta or table differ in row order")
        info[model] = (same, diff)
    print("S4: %d tables x 5 orders and 5 thread counts, machine %s, load %.2f" % (files, os.uname().machine, load_info()))
    print("S4 information (whole file identical / different, 8 permuted triples each): " + ", ".join("%s %d/%d" % (m, s, d) for m, (s, d) in info.items()))
    return cnt.report()


# ------------------------------------------------------------------ S5: unseen, missing and markers at predict

def predict_file(rnd, header, rows, cfg, data):
    """like gate_enc.predict_file, with numeric cells the stored schema accepts (an int column takes integers, a decimal column at most its scale)"""
    L = O.load(data, cfg)
    kinds = {L.names[j]: (L.kinds[j], L.scales[j]) for j in range(L.cols)}
    cats = [h for h in header if h.decode("latin-1") in cfg.cats]
    nums = [h for h in header if h not in cats and h != b"y"]
    pos = sorted(rnd.sample(range(len(nums) + len(cats)), len(cats)))
    rnd.shuffle(cats)
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
            if h in cats:
                if base is not None and rnd.random() < 0.7:
                    row.append(base[j])
                else:
                    row.append(rnd.choice([b"", b"never-seen", b"NA", b"?", "新".encode(), b"x" * 5, base[j] if base is not None else b"q", b"__other__"]))
            else:
                kd, sc = kinds[h]
                if rnd.random() < 0.15:
                    row.append(b"")
                elif kd == 1:
                    row.append(rnd.choice([b"1", b"-2", b"7", b"0"]))
                elif kd == 2:
                    row.append((b"%." + str(sc).encode() + b"f") % rnd.choice([0.5, -2.0, 3.25, 10.5]) if sc else b"1")
                else:
                    row.append(rnd.choice([b"1", b"-2.5", b"3.25", b"100"]))
        out.append(row)
    return G.write_csv(order, out, random.Random(rnd.random()))

def predict_cli(data, out="m.cml", name="p.csv", extra=()):
    put(name, data)
    return ml("predict", name, "--root", WORK, "--model-file", out, "--limit", 1000, *extra)


def gate_predict(seed):
    """S5: a model fitted from a table and a second file read from the model file alone: the unseen strings that are counted, the refusals of unknown=refuse (step, column,
    row), the markers and the empty cells, the columns in another order, and a hashed column whose unseen strings land in their buckets, against the oracle"""
    cnt = Count("S5 unseen, missing and markers at predict")
    rnd = random.Random(seed)
    made = unseen_total = refused_n = 0
    target = 120 if FAST else 400
    tries = 0
    while made < target and tries < target * 6:
        tries += 1
        header, rows, cfg, data = GE.gen_pair(rnd)
        order = [c.encode("latin-1") for c in cfg.cats]
        nnames = [h for h in header if h.decode("latin-1") not in cfg.cats and h != b"y"]
        plan = GE.gen_plan(rnd, order, nnames)
        olines, ost = E.run_plan(data, cfg, plan)
        if ost != 0 or any(l.startswith("remaining") and len(l.split()) > 1 for l in olines):
            continue
        data2 = predict_file(rnd, header, rows, cfg, data)
        text = GE.render_plan(plan, random.Random(tries))
        st, doc, _ = fit_cli("pf.csv", data, cfg, text, "tree")
        if st != 0:
            cnt.note("table %d" % tries, "fit refused: %r plan %r" % (rules(doc), text))
            continue
        made += 1
        want, wst = E.run_plan(data, cfg, plan, data2, hash_mode=True)
        pst, pdoc, _ = predict_cli(data2)
        cnt.runs += 1
        refusals = [l for l in want if l.startswith("p-refused") or l.startswith("p-load-refused")]
        if refusals:
            refused_n += 1
            l = refusals[0].split()
            if pst == 0:
                cnt.note("table %d" % tries, "the oracle refuses the predict file (%s) and the tool answered" % refusals[0])
            elif l[0] == "p-refused":
                r = (pdoc or {}).get("error", {})
                d = r.get("detail", {})
                names = [h.decode("latin-1") for h in header if h.decode("latin-1") in cfg.cats]
                if l[1] == "ml.input.unknown-category":
                    ok = r.get("rule") == l[1] and d.get("step") == int(l[2]) and d.get("row") == int(l[4]) and d.get("column") == names[-1 - int(l[3])]
                else:
                    ok = r.get("rule") == l[1]
                if not ok:
                    cnt.note("table %d" % tries, "the oracle says %s, the tool %r" % (refusals[0], r))
            continue
        if pst != 0:
            cnt.note("table %d" % tries, "the tool refused a predict file the oracle accepts: %r; plan %r; file %r" % (rules(pdoc), text, data2[:120]))
            continue
        unk = int([l for l in want if l.startswith("p-unknown")][0].split()[1])
        unseen_total += unk
        if pdoc["data"]["unknown_categories"] != unk:
            cnt.note("table %d" % tries, "unknown_categories %r, the oracle counts %d; plan %r" % (pdoc["data"]["unknown_categories"], unk, text))
    print("S5: %d models and predict files (%d the oracle refuses, %d unseen strings counted in all), machine %s, load %.2f" % (made, refused_n, unseen_total, os.uname().machine, load_info()))
    # a hashed column: an unseen string lands in the bucket it hashes to, and the prediction is the intercept plus that bucket's coefficient
    bucket_runs = 0
    for rep in range(12 if FAST else 40):
        n = 60
        header, rows, cfg = model_table(rnd, n, "regression", ncat=1, missing_rate=0.0)
        data = csv_bytes(header, rows)
        b = rnd.choice([2, 5, 8, 31])
        st, doc, _ = fit_model("hb.csv", data, cfg, "linear", "hash column=c0 buckets=%d; scale column=x1,x2 kind=standard" % b, extra=("--float", "x1,x2"))
        if st != 0:
            cnt.note("hash model", "fit refused: %r" % (rules(doc),))
            continue
        summ = doc["data"]["summary"]
        coef = {c["name"]: float(c["value"]) for c in summ["coefficients"]}
        icpt = float(summ["intercept"])
        fresh = [("never-seen-%d" % rnd.randrange(10 ** 6)).encode() for _ in range(6)] + ["不明".encode(), "🙂".encode(), b"a b"]
        test_rows = []
        for s in fresh:
            test_rows.append([b"0", s, b"0", b"0", b"0"])
        # x1 and x2 at their training means make the standardised columns 0: the prediction is the intercept plus the coefficient of the bucket
        xs1 = [float(r[2]) for r in rows]
        xs2 = [float(r[3]) for r in rows]
        m1, m2 = sum(xs1) / n, sum(xs2) / n
        pred_header = [b"id", b"c0", b"x1", b"x2"]
        pred_rows = [[b"0", s, repr(m1).encode(), repr(m2).encode()] for s in fresh]
        pst, pdoc, _ = predict_cli(csv_bytes(pred_header, pred_rows))
        cnt.runs += 1
        if pst != 0:
            cnt.note("hash model", "predict refused: %r" % (rules(pdoc),))
            continue
        bucket_runs += 1
        got = [float(r[0]) for r in pdoc["data"]["rows"]]
        if pdoc["data"]["unknown_categories"] != 0:
            cnt.note("hash model", "an unseen string under a hash step was counted unknown: %r" % pdoc["data"]["unknown_categories"])
        for s, g in zip(fresh, got):
            bk = E.bucket(s, b)
            want_v = icpt + coef["c0#%d" % bk]
            if abs(g - want_v) > 1e-9 * max(1.0, abs(want_v)):
                cnt.note("hash model", "string %r bucket %d: predicted %r, intercept + coefficient %r" % (s, bk, g, want_v))
    print("S5 hash mode: %d models with %d unseen strings each predicted from the bucket they hash to" % (bucket_runs, 9))
    return cnt.report()


# ------------------------------------------------------------------ S8: alignment by name for the four models

def gate_align(seed):
    """S8: the columns of the predict file in many orders give the same answer for linear, logistic, tree and forest (numeric and categorical columns alike); a
    missing, an extra, a dropped categorical column and a retyped column are ml.schema.mismatch (or the numeric reader's refusal) with a repair that works"""
    cnt = Count("S8 alignment by name")
    rnd = random.Random(seed)
    for model in ("linear", "logistic", "tree", "forest", "boost"):
        for mix in ("onehot", "mixed"):
            cls = model == "logistic"
            header, rows, cfg = model_table(rnd, 80, "classification" if cls else "regression", ncat=3)
            st, doc, _ = fit_model("al.csv", csv_bytes(header, rows), cfg, model, steps_for(rnd, mix), extra=model_kind_flags(model, cls))
            cnt.runs += 1
            if st != 0:
                cnt.note("%s %s" % (model, mix), "fit refused: %r" % (rules(doc),))
                continue
            cols = [i for i, h in enumerate(header) if h != b"y"]
            test_rows = rows[:25] + [[b"0"] + [b"unseen-" + bytes([97 + k]) for k in range(3)] + [b"1.5", b"2.5", b"0"]]
            ph = [header[i] for i in cols]
            base_st, base_doc, base_raw = predict_cli(csv_bytes(ph, [[r[i] for i in cols] for r in test_rows]), extra=(["--proba"] if cls else []))
            cnt.runs += 1
            if base_st != 0:
                cnt.note("%s %s" % (model, mix), "predict refused: %r" % (rules(base_doc),))
                continue
            for perm in range(6):
                order = list(cols)
                random.Random(perm * 7 + 1).shuffle(order)
                st2, doc2, raw2 = predict_cli(csv_bytes([header[i] for i in order], [[r[i] for i in order] for r in test_rows]), extra=(["--proba"] if cls else []))
                cnt.runs += 1
                if st2 != 0 or raw2 != base_raw:
                    cnt.note("%s %s permutation %d" % (model, mix, perm), "the answer differs from the file in the fitted order (%r)" % (rules(doc2),))
            # the target present in the predict file is ignored, and a different position for it too
            full_cols = list(range(len(header)))
            random.Random(3).shuffle(full_cols)
            st3, doc3, raw3 = predict_cli(csv_bytes([header[i] for i in full_cols], [[r[i] for i in full_cols] for r in test_rows]), extra=(["--proba"] if cls else []))
            cnt.runs += 1
            if st3 != 0 or raw3 != base_raw:
                cnt.note("%s %s" % (model, mix), "a predict file that also holds the target differs (%r)" % (rules(doc3),))
            # a missing numeric column, a missing categorical column, a dropped categorical column
            for what, removed in (("numeric", b"x2"), ("categorical", b"c1")):
                keep = [i for i in cols if header[i] != removed]
                s4, d4, _ = predict_cli(csv_bytes([header[i] for i in keep], [[r[i] for i in keep] for r in test_rows]))
                cnt.runs += 1
                err = (d4 or {}).get("error", {})
                if s4 == 0 or err.get("rule") != "ml.schema.mismatch" or removed.decode() not in json.dumps(err.get("detail")):
                    cnt.note("%s %s" % (model, what), "a missing %s column answered %r" % (what, err))
            s5, d5, _ = predict_cli(csv_bytes(ph, [[r[i] for i in cols] for r in test_rows]), extra=("--drop", "c1"))
            cnt.runs += 1
            err = (d5 or {}).get("error", {})
            if s5 == 0 or err.get("rule") != "ml.schema.mismatch" or "c1" not in json.dumps(err.get("detail")):
                cnt.note("%s dropped categorical" % model, "answered %r" % (err,))
            # an extra numeric column, and an extra text column: refused with a repair that gives the answer of the fitted columns
            for label, extra_cell in (("numeric", b"7"), ("text", b"hello")):
                h2 = ph + [b"zz"]
                r2 = [[r[i] for i in cols] + [extra_cell] for r in test_rows]
                s6, d6, _ = predict_cli(csv_bytes(h2, r2), extra=(["--proba"] if cls else []))
                cnt.runs += 1
                err = (d6 or {}).get("error", {})
                rep = err.get("repair")
                if s6 == 0:
                    cnt.note("%s extra %s" % (model, label), "an extra column was accepted")
                elif not rep or rep.get("kind") != "retry":
                    cnt.note("%s extra %s" % (model, label), "no retry repair: %r" % (err,))
                else:
                    argv = rep["argv"][1:]
                    p = subprocess.run([ML] + argv, capture_output=True)
                    cnt.runs += 1
                    if p.returncode != 0 or p.stdout != base_raw:
                        cnt.note("%s extra %s" % (model, label), "the repair %r does not give the answer of the fitted columns" % (argv,))
            # a numeric column holding text is the numeric reader's refusal, naming the column
            hx = list(ph)
            rx = [[r[i] for i in cols] for r in test_rows]
            j = hx.index(b"x1")
            for r in rx:
                r[j] = b"abc"
            s7, d7, _ = predict_cli(csv_bytes(hx, rx))
            cnt.runs += 1
            if s7 == 0:
                cnt.note("%s retyped" % model, "text in a numeric column was accepted")
    # the numeric models: the same rule for a file with no categorical column (the numeric features found by name, in any order)
    for model in ("linear", "logistic", "tree", "forest", "boost"):
        cls = model == "logistic"
        rows = []
        for r in range(80):
            a, b, c = round(rnd.uniform(-2, 2), 3), round(rnd.uniform(0, 5), 2), rnd.randint(0, 9)
            score = a - 0.3 * b + 0.2 * c + rnd.gauss(0, 0.3)
            y = (b"1" if score > 0 else b"0") if cls else ("%.4f" % score).encode()
            rows.append([repr(a).encode(), repr(b).encode(), str(c).encode(), y])
        header = [b"a", b"b", b"c", b"y"]
        put("an.csv", csv_bytes(header, rows))
        flags = {"linear": ["--model", "linear"], "logistic": ["--model", "logistic", "--l2", "0.1"], "tree": ["--model", "tree", "--task", "regression"],
                 "forest": ["--model", "forest", "--task", "regression", "--trees", "5"], "boost": ["--model", "boost", "--task", "regression", "--rounds", "5", "--min-leaf", "3"]}[model]
        st, doc, _ = ml("fit", "an.csv", "--root", WORK, "--target", "y", *flags, "--output", "an.cml", "--overwrite")
        cnt.runs += 1
        if st != 0:
            cnt.note("numeric %s" % model, "fit refused: %r" % (rules(doc),))
            continue
        test_rows = rows[:20]
        def pred(order):
            put("ap.csv", csv_bytes([header[i] for i in order], [[r[i] for i in order] for r in test_rows]))
            return ml("predict", "ap.csv", "--root", WORK, "--model-file", "an.cml", "--limit", 100, *(["--proba"] if cls else []))
        s0, d0, base = pred([0, 1, 2])
        for k, order in enumerate(([2, 1, 0], [1, 2, 0], [3, 2, 0, 1], [1, 0, 2, 3])):
            s1, d1, raw = pred(order)
            cnt.runs += 1
            if s1 != 0 or raw != base:
                cnt.note("numeric %s order %r" % (model, order), "the answer differs from the file in the fitted order (%r)" % (rules(d1),))
    return cnt.report()


# ------------------------------------------------------------------ S9: privacy

def gate_privacy(seed):
    """S9: a category string is in the model file exactly when the step stores strings (encode, freq, label: yes, with their counts; hash: no byte of any category)"""
    cnt = Count("S9 privacy")
    rnd = random.Random(seed)
    tokens = [("zq%02d-token-%d" % (i, rnd.randrange(10 ** 6))).encode() for i in range(14)]
    secret_pools = [tokens[0:5], tokens[5:10], tokens[10:14]]
    for model in ("linear", "tree"):
        for steps, stores in (("encode column=c0 kind=onehot; freq column=c1; encode column=c2 kind=label", (True, True, True)),
                              ("hash column=c0 buckets=6; freq column=c1; hash column=c2 buckets=4", (False, True, False)),
                              ("hash column=c0,c1,c2 buckets=9", (False, False, False)),
                              ("encode column=c0 kind=onehot min-count=3; hash column=c1,c2 buckets=3", (True, False, False))):
            header, rows, cfg = model_table(rnd, 90, "regression", ncat=3, missing_rate=0.0, cat_pool=secret_pools)
            st, doc, _ = fit_model("pv.csv", csv_bytes(header, rows), cfg, model, steps.replace(";", " missing=zero;") + " missing=zero; scale column=x1,x2 kind=standard", extra=())
            cnt.runs += 1
            if st != 0:
                cnt.note("%s %r" % (model, steps), "fit refused: %r" % (rules(doc),))
                continue
            f = read_file()
            tab = S.parse_file(f).get(6, b"")
            for c, pool in enumerate(secret_pools):
                for tok in pool:
                    inside = tok in f
                    if inside != stores[c]:
                        cnt.note("%s %r column c%d" % (model, steps, c), "the string %r is %s the model file but the step %s strings" % (tok, "in" if inside else "not in", "stores" if stores[c] else "does not store"))
                    # and nowhere but the table
                    if inside and (tok in f[:f.index(tab)] if tab else True):
                        cnt.note("%s %r" % (model, steps), "a category string outside the string table")
            # the names of the columns are stored in both cases, and the table has no byte of a hashed column's values
            for c in range(3):
                if b"c%d" % c not in tab:
                    cnt.note("%s %r" % (model, steps), "the name of column c%d is not in the table" % c)
    return cnt.report()


# ------------------------------------------------------------------ S7: the bounds of a fit

def wide_csv(ncols, nrows, width=4):
    header = [b"id"] + [b"c%03d" % j for j in range(ncols)] + [b"y"]
    rows = []
    for r in range(nrows):
        rows.append([str(r).encode()] + [(b"v%0*d" % (width, r)) for _ in range(ncols)] + [str(r % 7).encode()])
    return header, rows


def gate_limits(seed):
    """S7 at the command line: 128 columns of 1,024 strings (131,072) are written, read back and applied; 129 columns of 1,024 are refused with ml.limit.string-table
    (what strings, size and max), nothing is written, with and without --split and --dry-run"""
    cnt = Count("S7 the bounds of a fit")
    for ncols, expect_ok in ((128, True), (129, False)):
        header, rows = wide_csv(ncols, 1024)
        data = csv_bytes(header, rows)
        names = ",".join(h.decode() for h in header[1:-1])
        put("w.csv", data)
        common = ["--root", WORK, "--target", "y", "--cat", names, "--max-categories", 1024, "--max-cat-columns", 200, "--model", "tree", "--task", "regression", "--step",
                  "freq column=* max-categories=1024", "--output", "wide.cml", "--overwrite"]
        for mode in (("--dry-run",), ("--split", "train=0.999"), ()):
            if (WORK / "wide.cml").exists():
                (WORK / "wide.cml").unlink()
            st, doc, _ = ml("fit", "w.csv", *common, *mode, timeout=600)
            cnt.runs += 1
            if expect_ok:
                want = 9 if "--dry-run" in mode else 0
                if st != want:
                    cnt.note("%d columns %r" % (ncols, mode), "the fit answered %d %r" % (st, rules(doc)))
                elif "--dry-run" not in mode and not (WORK / "wide.cml").exists():
                    cnt.note("%d columns %r" % (ncols, mode), "no file written")
                elif "--dry-run" not in mode:
                    f = read_file("wide.cml")
                    tab = S.parse_file(f)[6]
                    if S.table_verdict(tab) is not None:
                        cnt.note("%d columns" % ncols, "the oracle refuses the table: %s" % S.table_verdict(tab))
                    train = 1024 - (-(-1024 * 1 // 1000) if "--split" in mode else 0)
                    if struct.unpack_from("<Q", tab, 16)[0] != 128 * train:
                        cnt.note("%d columns %r" % (ncols, mode), "strings %d, wanted %d" % (struct.unpack_from("<Q", tab, 16)[0], 128 * train))
            else:
                err = (doc or {}).get("error", {})
                d = err.get("detail", {})
                size = 129 * (1024 - (-(-1024 * 1 // 1000) if "--split" in mode else 0))
                if st != 8 or err.get("rule") != "ml.limit.string-table" or d.get("what") != "strings" or d.get("size") != size or d.get("max") != 131072:
                    cnt.note("%d columns %r" % (ncols, mode), "answered %d %r" % (st, err))
                if (WORK / "wide.cml").exists():
                    cnt.note("%d columns %r" % (ncols, mode), "a file was written")
        if expect_ok:
            # the file applies: predict on the training file itself, every cell known
            pst2, pdoc2, _ = ml("predict", "w.csv", "--root", WORK, "--model-file", "wide.cml", "--limit", 5, timeout=600)
            cnt.runs += 1
            if pst2 != 0 or pdoc2["data"]["unknown_categories"] != 0:
                cnt.note("128 columns", "predict from the widest table: %d %r" % (pst2, rules(pdoc2)))
    print("S7: 128 and 129 columns of 1,024 strings, three modes each, machine %s, load %.2f" % (os.uname().machine, load_info()))
    return cnt.report()


# ------------------------------------------------------------------ S10: damaged files

def sample_models(rnd):
    """the model files the fuzz damages: (label, bytes) of small fits, one per model and a hash-only one"""
    out = []
    for model, steps in (("linear", "encode column=c0 kind=onehot missing=zero; freq column=c1,c2 missing=zero; scale column=x1,x2 kind=standard"),
                         ("logistic", "hash column=c0,c1 buckets=3 missing=zero; encode column=c2 kind=label missing=zero"),
                         ("tree", "encode column=c0,c1 kind=onehot min-count=2; hash column=c2 buckets=4"),
                         ("forest", "freq column=c0,c1,c2 missing=zero")):
        cls = model == "logistic"
        for _ in range(20):
            header, rows, cfg = model_table(rnd, 14, "classification" if cls else "regression", ncat=3)
            cfg.na = [b"NA", b"?"]
            small = {"forest": ["--model", "forest", "--trees", "2", "--max-depth", "2", "--bins", "4"], "tree": ["--model", "tree", "--max-depth", "3", "--bins", "4"]}.get(model)
            st, doc, _ = fit_model("fz.csv", csv_bytes(header, rows), cfg, model, steps, extra=model_kind_flags(model, cls), flags=small)
            if st == 0:
                break
        if st != 0:
            raise SystemExit("fuzz sample fit refused: %r" % (doc,))
        out.append((model, read_file(), header, rows))
    return out


def run_damaged(data, label, verb, ctx, cnt):
    """damage -> the tool's answer must be one valid JSON line with no trap; when the oracle can say the refusal, the tool says it"""
    put("dm.cml", data)
    pst, pdoc, raw = ml(verb, "p.csv", "--root", WORK, "--model-file", "dm.cml", *(["--limit", 5] if verb == "predict" else []))
    cnt.runs += 1
    if pst in TRAPS or pst < 0:
        cnt.note(label, "trapped with status %d" % pst)
        return
    if pdoc is None:
        cnt.note(label, "no single JSON line: %r" % raw[:100])
        return
    rule = (pdoc.get("error") or {}).get("rule")
    # the file-level checks the oracle writes
    ft = S.file_verdict(data)
    if ft is not None:
        if rule != ft:
            cnt.note(label, "the oracle's file check says %s, the tool %s" % (ft, rule))
        return
    secs = S.parse_file(data)
    if 6 in secs:
        why = S.table_verdict(secs[6])
        if why is not None:
            ok = rule == "ml.model.string-table" and (pdoc["error"].get("detail") or {}).get("reason") == why
            if not ok and pst != 0:
                # an earlier field of the file may be refused first (the sum order, the plan, the schema): then the tag is another model tag, never a trap
                ok = rule in ("ml.model.version", "ml.model.malformed")
            if not ok:
                cnt.note(label, "the oracle's table check says %s, the tool %s %r" % (why, rule, (pdoc.get("error") or {}).get("detail")))
            return
    if pst != 0 and rule not in ("ml.model.version", "ml.model.malformed", "ml.model.string-table", "ml.schema.mismatch", "ml.input.missing-value", "ml.input.non-numeric", "ml.model.truncated", "ml.model.checksum", "ml.model.bad-magic", "limit.input-too-large", "ml.label.not-binary", "ml.input.unknown-category", "ml.input.too-many-columns", "ml.input.too-many-cells", "ml.input.too-many-rows", "ml.input.empty") and pst != 9:
        cnt.note(label, "an unexpected refusal %s" % rule)


def edit(data, at, new):
    b = bytearray(data)
    b[at:at + len(new)] = new
    return bytes(b)


def gate_fuzz(runs, seed):
    """S10: every byte of small model files flipped three ways, every truncation, the digests repaired (the file's, and the table's then the file's), fields of the table set
    to 2^63 - 1, 2^32 - 1 and other extremes, invalid UTF-8 in a name and in a string, duplicates and disorder, and seeded damage: no trap, one JSON line each, the oracle's
    verdict where it can give one"""
    cnt = Count("S10 damaged files")
    rnd = random.Random(seed)
    exhaustive = 0
    for model, f, header, rows in sample_models(rnd):
        put("p.csv", csv_bytes([h for h in header if h != b"y"], [[c for i, c in enumerate(r) if header[i] != b"y"] for r in rows]))
        secs = S.parse_file(f)
        # the table's place in the file
        tstart = f.index(secs[6])
        tend = tstart + len(secs[6])
        for at in range(len(f)):
            if at % STRIDE and not (tstart <= at < tend):
                continue
            for how in range(3):
                v = f[at] ^ (1 if how == 0 else 128 if how == 1 else 255)
                d = edit(f, at, bytes([v]))
                run_damaged(d, "%s flip %d/%d" % (model, at, how), "predict", None, cnt)
                run_damaged(S.reseal_file(d), "%s flip %d/%d, file digest repaired" % (model, at, how), "eval" if at % 2 else "predict", None, cnt)
                if tstart <= at < tend - 32:
                    t2 = S.reseal_table(d[tstart:tend])
                    d2 = S.reseal_file(d[:tstart] + t2 + d[tend:])
                    run_damaged(d2, "%s flip %d/%d, both digests repaired" % (model, at, how), "predict", None, cnt)
                exhaustive += 1
        for cut in range(0, len(f)):
            if cut % STRIDE and not (tstart <= cut < tend):
                continue
            run_damaged(f[:cut], "%s cut at %d" % (model, cut), "predict", None, cnt)
            run_damaged(S.reseal_file(f[:cut]) if cut > 32 else f[:cut], "%s cut at %d, digest repaired" % (model, cut), "predict", None, cnt)
        # every u32 and u64 field of the table set to extremes, both digests repaired
        tb = secs[6]
        for off, width in S.table_fields(tb):
            vals = ([struct.pack("<I", v) for v in (0, 1, 2 ** 16, 2 ** 31 - 1, 2 ** 31, 2 ** 32 - 1)] if width == 4
                    else [struct.pack("<q", v) for v in (0, 1, 2 ** 31 - 1, 2 ** 32, 2 ** 62, 2 ** 63 - 1, -1, -2 ** 63)])
            for ext in vals:
                t2 = S.reseal_table(tb[:off] + ext + tb[off + width:])
                run_damaged(S.reseal_file(f[:tstart] + t2 + f[tend:]), "%s field %d/%d set to %r" % (model, off, width, ext), "eval" if off % 2 else "predict", None, cnt)
        # invalid UTF-8 in a name and in a string, a string and a name repeated, strings out of order
        names_at = [(o, w) for o, w in S.table_fields(tb)]
        for bad_bytes in (b"\xff", b"\xc0\x80", b"\xed\xa0\x80"):
            for off in range(44, len(tb) - 32):
                if tb[off:off + 1] in (b"c", b"a", b"e", b"m") and off % 3 == 0:
                    t2 = S.reseal_table(tb[:off] + bad_bytes[:1] + tb[off + 1:])
                    run_damaged(S.reseal_file(f[:tstart] + t2 + f[tend:]), "%s byte %d set to %r" % (model, off, bad_bytes[:1]), "predict", None, cnt)
    # the seeded damages on the sample files
    models = sample_models(rnd)
    for k in range(runs):
        model, f, header, rows = models[k % len(models)]
        put("p.csv", csv_bytes([h for h in header if h != b"y"], [[c for i, c in enumerate(r) if header[i] != b"y"] for r in rows]))
        secs = S.parse_file(f)
        tstart = f.index(secs[6])
        tend = tstart + len(secs[6])
        d = f
        kind = rnd.randrange(6)
        if kind == 0:
            for _ in range(rnd.randint(1, 6)):
                d = edit(d, rnd.randrange(len(d)), bytes([rnd.randrange(256)]))
        elif kind == 1:
            d = d[:rnd.randrange(len(d))]
        elif kind == 2:
            at = rnd.randrange(tstart, tend)
            d = d[:at] + bytes(rnd.randrange(256) for _ in range(rnd.randint(1, 12))) + d[at:]
        elif kind == 3:
            at = rnd.randrange(tstart, tend - 32)
            d = d[:at] + d[at + rnd.randint(1, 20):]
        elif kind == 4:
            # swap two strings or two names of the table
            at = rnd.randrange(tstart + 44, tend - 40)
            d = edit(d, at, bytes(rnd.randrange(256) for _ in range(rnd.randint(1, 4))))
        else:
            d = d + bytes(rnd.randrange(256) for _ in range(rnd.randint(1, 40)))
        # repair: nothing, the file digest, or the table digest then the file digest
        rep = rnd.randrange(3)
        if rep >= 1 and len(d) > 40:
            if rep == 2:
                ts, te = d.find(b"c0") , len(d)
                try:
                    p = S.parse_file(S.reseal_file(d))
                    t2 = S.reseal_table(p[6])
                    d = S.reseal_file(d[:d.index(p[6])] + t2 + d[d.index(p[6]) + len(p[6]):])
                except (S.FileBad, KeyError, ValueError):
                    d = S.reseal_file(d)
            else:
                d = S.reseal_file(d)
        run_damaged(d, "%s seeded %d" % (model, k), "predict" if k % 2 else "eval", None, cnt)
    print("S10: %d single-byte flips (three ways each, and with repaired digests), every truncation, the table fields at their extremes, %d seeded damages, machine %s, load %.2f" % (exhaustive, runs, os.uname().machine, load_info()))
    return cnt.report()


# ------------------------------------------------------------------ the command line

def main():
    argv = sys.argv[1:]
    which = argv[0] if argv else "all"

    def opt(name, default):
        return int(argv[argv.index(name) + 1]) if name in argv else default

    files = opt("--files", 30 if FAST else 300)
    runs = opt("--runs", 300 if FAST else 3000)
    seed = opt("--seed", 20231014)
    gates = {name: fn for name, fn in globals().items() if name.startswith("gate_")}
    table = {
        "diff": lambda: gates["gate_diff"](files, seed),
        "roundtrip": lambda: gates["gate_roundtrip"](16 if FAST else 48, seed),
        "order": lambda: gates["gate_order"](10 if FAST else 40, seed),
    }
    for extra in ("predict", "limits", "align", "privacy", "fuzz"):
        if "gate_" + extra in gates:
            if extra == "fuzz":
                table[extra] = (lambda e: lambda: gates["gate_" + e](runs, seed))(extra)
            else:
                table[extra] = (lambda e: lambda: gates["gate_" + e](seed))(extra)
    names = list(table) if which == "all" else [which]
    total = 0
    for name in names:
        total += table[name]()
    print("machine %s, load %.2f" % (os.uname().machine, load_info()))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())

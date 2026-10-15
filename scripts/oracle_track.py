#!/usr/bin/env python3
"""The oracle of run tracking and skew checks (docs/tracking.md, gates T1 and T2): Python that writes the provenance section of a model file and the run record of a fit from the CSV and the
command line alone, and compares BYTES with what the real `ml` wrote.

What it does not borrow from the cancho code: the reading of the CSV (Python's csv module and float(), which rounds to nearest even as the reader does), the statistics (fractions.Fraction: the exact
sum, the exact sum of squares about the rounded mean, each rounded once by float(Fraction), then one IEEE division and one sqrt), the layout of section 7, the canonical JSON of the record, the
id and the check (hashlib). What it takes from the file it judges: the plan text (tested by the other oracles), and the metrics object of the record (tested by the eval oracles); everything
else of the record is rebuilt from the command line.

    python3 scripts/oracle_track.py --suite [N]     the cases of the table below, N repetitions of the seeded ones (default 1), against build/ml
    python3 scripts/oracle_track.py --self          the functions of this file on a hand-made example (no binary)
Exit 1 on any mismatch. ML_BIN names the binary.
"""

import hashlib
import json
import math
import os
import pathlib
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
BIN = os.environ.get("ML_BIN", str(ROOT / "build" / "ml"))
PIN = (ROOT / "cancho.toml").read_text().split('cancho = "')[1].split('"')[0]
FAILS = []
COUNT = {}


def check(ok, what):
    if not ok:
        FAILS.append(what)
        print("MISMATCH", what)
    return ok


def count(what, n=1):
    COUNT[what] = COUNT.get(what, 0) + n


# ---------------------------------------------------------------- the model file, parsed on its own


def sections(data):
    """{tag: payload} of a CML1 file, and the section count of its header. The checksum is verified."""
    assert data[:4] == b"CML1", "magic"
    assert struct.unpack("<I", data[4:8])[0] == 1
    n = struct.unpack("<I", data[8:12])[0]
    assert hashlib.sha256(data[:-32]).digest() == data[-32:], "checksum"
    pos, out, tags = 12, {}, []
    for _ in range(n):
        tag, length = struct.unpack("<IQ", data[pos:pos + 12])
        out[tag] = data[pos + 12:pos + 12 + length]
        tags.append(tag)
        pos += 12 + length
    assert pos == len(data) - 32
    assert tags == sorted(tags) and tags[:5] == [1, 2, 3, 4, 5]
    return out, n


def schema_of(payload):
    """(names, types, target index) of the schema section"""
    cols, target = struct.unpack("<II", payload[:8])
    pos, names, types = 8, [], []
    for _ in range(cols):
        (nl,) = struct.unpack("<I", payload[pos:pos + 4])
        pos += 4
        names.append(payload[pos:pos + nl].decode())
        pos += nl
        ty, _scale = struct.unpack("<II", payload[pos:pos + 8])
        pos += 8
        types.append(ty)
    assert pos == len(payload)
    return names, types, target


def meta_of(payload):
    rows, cols, seed, nh, cl = struct.unpack("<QQQII", payload[:32])
    pin = payload[32:32 + cl].decode()
    tail = payload[32 + cl:]
    return {"rows": rows, "cols": cols, "seed": seed, "pin": pin, "sum_order": struct.unpack("<I", tail)[0] if tail else None}


# ---------------------------------------------------------------- the CSV, as the reader reads it


def read_csv(data):
    import csv
    import io

    text = data.decode("utf-8")
    if text.startswith("﻿"):
        text = text[1:]
    rows = [r for r in csv.reader(io.StringIO(text, newline="")) if r]
    return rows[0], rows[1:]


def rounded(fr):
    """the double nearest the exact rational (ties to even), as Python's integer true division rounds"""
    return float(fr)


def column_stats(cells):
    """(count, min, max, mean, std) of the non-missing cells of a column, the rule of docs/tracking.md section 2.3; the four doubles as bits"""
    xs = [float(c) for c in cells if c != ""]
    n = len(xs)
    if n == 0:
        return n, 0, 0, 0, 0

    def bits(v):
        return 0 if v == 0 else struct.unpack("<q", struct.pack("<d", v))[0]

    mn, mx = min(xs), max(xs)
    mean = rounded(sum(Fraction(x) for x in xs) / n)
    sse = rounded(sum((Fraction(x) - Fraction(mean)) ** 2 for x in xs))
    std = math.sqrt(sse / n)
    return n, bits(mn), bits(mx), bits(mean), bits(std)


def provenance(data, features, train_mask=None):
    """the payload of section 7 for the file `data` (the bytes of the CSV): `features` the numeric feature column names in schema order, `train_mask` a list of booleans per data row (None: all)"""
    header, rows = read_csv(data)
    sel = [r for i, r in enumerate(rows) if train_mask is None or train_mask[i]]
    out = struct.pack("<I", 1) + struct.pack("<Q", len(data)) + hashlib.sha256(data).digest() + struct.pack("<QQ", len(rows), len(sel)) + struct.pack("<I", len(features))
    for name in features:
        j = header.index(name)
        n, mn, mx, me, sd = column_stats([r[j] for r in sel])
        out += struct.pack("<Qqqqq", n, mn, mx, me, sd)
    return out


def out_of_range(prov, header, rows, features, skip_target=None):
    """(cells, rows, {column: count}) the file's non-missing cells strictly outside [min, max] of the stored records"""
    nf = struct.unpack("<I", prov[60:64])[0]
    assert nf == len(features)
    lim = []
    for k in range(nf):
        n, mn, mx, me, sd = struct.unpack("<Qqqqq", prov[64 + 40 * k:64 + 40 * k + 40])
        f = lambda b: struct.unpack("<d", struct.pack("<q", b))[0]  # noqa: E731
        lim.append((n, f(mn), f(mx)))
    counts, cells, nrows = {}, 0, 0
    for r in rows:
        here = 0
        for k, name in enumerate(features):
            c = r[header.index(name)]
            if c == "" or lim[k][0] == 0:
                continue
            v = float(c)
            if v < lim[k][1] or v > lim[k][2]:
                counts[name] = counts.get(name, 0) + 1
                cells += 1
                here += 1
        nrows += here > 0
    return cells, nrows, counts


# ---------------------------------------------------------------- the run record


def jdump(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def split_reduced(text):
    """'train=0.8' -> {'num': 4, 'den': 5} (the fraction in lowest terms), None for no split"""
    if text is None:
        return None
    frac = text.split("=")[1]
    digits = len(frac.split(".")[1]) if "." in frac else 0
    den = 10 ** digits
    num = int(frac.replace(".", ""))
    g = math.gcd(num, den)
    return {"num": num // g, "den": den // g}


def inputs_obj(path, data, rows, target, task, plan, seed, positive, split, drop, flt, cat, na, reader, sum_order):
    return {
        "data": {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "rows": rows},
        "target": target, "task": task, "plan": plan, "seed": seed, "positive": positive, "split": split_reduced(split),
        "drop": drop, "float": flt, "cat": cat, "na_values": na, "reader": reader, "compiler": PIN, "sum_order": sum_order,
    }


def run_record(inputs, threads, model_path, model, counts, metrics):
    """(id, record bytes)"""
    inp = jdump(inputs)
    rid = hashlib.sha256(inp.encode()).hexdigest()
    body = '{"schema":"ml.run.v1","id":"%s","inputs":%s,"threads":%d,"model":%s,"counts":%s,"metrics":%s' % (
        rid, inp, threads, jdump({"path": model_path, "bytes": len(model), "sha256": hashlib.sha256(model).hexdigest()}), jdump(counts),
        "null" if metrics is None else jdump(metrics))
    check_ = hashlib.sha256(body.encode()).hexdigest()
    return rid, (body + ',"check":"%s"}\n' % check_).encode()


# ---------------------------------------------------------------- the real tool


def ml(root, *args):
    r = subprocess.run([BIN, "--root", str(root), *map(str, args)], capture_output=True)
    try:
        doc = json.loads(r.stdout.decode().splitlines()[0])
    except Exception:
        doc = None
    return r.returncode, doc


DATA = ROOT / "docs" / "examples" / "data"


def case_list():
    """(label, flags before the verb, csv, model kind word, task, extra)"""
    return [
        ("constant regression", ["--target", "progression"], "diabetes_train.csv"),
        ("constant classification", ["--task", "classification", "--target", "virginica"], "iris_train.csv"),
        ("linear", ["--model", "linear", "--target", "progression", "--l2", "0.5"], "diabetes_train.csv"),
        ("linear split", ["--model", "linear", "--target", "progression", "--split", "train=0.8", "--seed", "3", "--step", "scale column=* kind=standard"], "diabetes_train.csv"),
        ("logistic positive", ["--model", "logistic", "--target", "virginica", "--positive", "0"], "iris_train.csv"),
        ("logistic split 0.75", ["--model", "logistic", "--target", "virginica", "--split", "train=0.750", "--seed", "2"], "iris_train.csv"),
        ("tree split", ["--model", "tree", "--task", "regression", "--target", "progression", "--split", "train=0.7"], "diabetes_train.csv"),
        ("forest", ["--model", "forest", "--task", "classification", "--trees", "7", "--target", "virginica"], "iris_train.csv"),
        ("kmeans", ["--model", "kmeans", "--k", "3", "--drop", "virginica", "--seed", "5"], "iris_train.csv"),
        ("linear float drop", ["--model", "linear", "--target", "progression", "--float", "age", "--drop", "sex"], "diabetes_train.csv"),
        ("logistic categorical", ["--model", "logistic", "--l2", "1", "--target", "adopted", "--cat", "species,city", "--step", "encode column=species kind=onehot;encode column=city kind=label", "--drop", "id"], "pets.csv"),
        ("forest categorical split", ["--model", "forest", "--task", "classification", "--trees", "5", "--target", "adopted", "--cat", "species,city", "--step", "hash column=species buckets=8;freq column=city", "--drop", "id", "--na-values", "NA,?", "--split", "train=0.8"], "pets.csv"),
    ]


def flag_value(flags, name, default=None):
    return flags[flags.index(name) + 1] if name in flags else default


def oracle_case(label, flags, csvname, work):
    """fit with --track, then compare section 7 and the record with the oracle's bytes. Returns the run id."""
    for f in os.listdir(DATA):
        shutil.copy(DATA / f, work / f)
    (work / "runs").mkdir(exist_ok=True)
    rc, doc = ml(work, "--track", "runs", "--output", "m.ml", "--overwrite", *flags, "fit", csvname)
    if not check(rc == 0, "%s: fit failed %r" % (label, doc and doc.get("error"))):
        return None
    model = (work / "m.ml").read_bytes()
    data = (work / csvname).read_bytes()
    secs, n = sections(model)
    names, types, target = schema_of(secs[2])
    header, rows = read_csv(data)
    split = flag_value(flags, "--split")
    seed = int(flag_value(flags, "--seed", 0))
    mask = None
    if split:
        import oracle_metrics as om

        sp = split_reduced(split)
        n_rows = len(rows)
        n_test = -(-n_rows * (sp["den"] - sp["num"]) // sp["den"])
        test = om.py_train_test(n_rows, n_test, seed, 0)
        mask = [t == 0 for t in test]
    feats = [nm for j, nm in enumerate(names) if j != target and types[j] != 3]
    kind = struct.unpack("<I", secs[4][:4])[0]
    if kind in (1, 2, 3):
        feats = []
        mask = None
    want = provenance(data, feats, mask)
    check(7 in secs, "%s: no section 7" % label)
    if 7 in secs:
        check(secs[7] == want, "%s: section 7 differs from the oracle (%d bytes vs %d)" % (label, len(secs[7]), len(want)))
        count("section 7 compared byte for byte")
    # the record
    runs = sorted(p for p in (work / "runs").iterdir() if p.name.endswith(".json"))
    check(len(runs) == 1, "%s: %d records" % (label, len(runs)))
    rec = runs[0].read_bytes()
    meta = meta_of(secs[5])
    plan = secs[1].decode()
    task = {1: "regression", 2: "classification", 3: "regression"}.get(kind)
    if kind >= 4:
        task = "clustering" if kind == 6 else ("classification" if "task=classification" in plan or kind == 5 else "regression")
        if kind == 4:
            task = "regression"
    sum_order = meta["sum_order"]
    reader = None
    if "--cat" in flags:
        lim = lambda k, d: int(flag_value(flags, k, d))  # noqa: E731
        reader = {"max_categories": lim("--max-categories", 64), "max_string_bytes": lim("--max-string-bytes", 255), "max_pool_bytes": lim("--max-pool-bytes", 1048576), "max_cat_columns": lim("--max-cat-columns", 128)}
    nfile = len(rows)
    counts = {"file_rows": nfile, "train_rows": meta["rows"], "test_rows": nfile - meta["rows"] if mask else 0,
              "features": len(feats), "steps": struct.unpack("<I", secs[3][:4])[0] if kind >= 4 else 0}
    on_rec = json.loads(rec.decode().rstrip("\n").rsplit(',"check"', 1)[0] + "}")
    metrics = on_rec["metrics"]
    positive = int(flag_value(flags, "--positive")) if "--positive" in flags else None
    inputs = inputs_obj(csvname, data, nfile, None if kind == 6 else names[target], task, plan, seed, positive, split,
                        flag_value(flags, "--drop", ""), flag_value(flags, "--float", ""), flag_value(flags, "--cat", ""), flag_value(flags, "--na-values", ""), reader, sum_order)
    rid, want_rec = run_record(inputs, 1, "m.ml", model, counts, metrics)
    check(runs[0].name == rid + ".json", "%s: the record is named %s, the oracle's id is %s" % (label, runs[0].name, rid))
    check(rec == want_rec, "%s: the record differs from the oracle's bytes\n  real  %r\n  want  %r" % (label, rec[:400], want_rec[:400]))
    count("run record compared byte for byte")
    if kind >= 4:
        check(metrics is not None and metrics["on"] == ("held-out" if split else "train"), "%s: the metrics are labelled %r" % (label, metrics and metrics["on"]))
    else:
        check(metrics is None, "%s: a constant model has no metrics in its record" % label)
    return rid


def suite(reps):
    work = pathlib.Path(tempfile.mkdtemp(prefix="oracle_track_"))
    try:
        for r in range(reps):
            for label, flags, csvname in case_list():
                d = work / ("c%d_%s" % (r, label.replace(" ", "_")))
                d.mkdir()
                rid = oracle_case(label, flags, csvname, d)
                if rid is None:
                    continue
                # replay at 1 and 8 threads
                for th in (1, 8):
                    rc, doc = ml(d, "--threads", th, "runs", "verify", "runs", rid)
                    check(rc == 0 and doc and doc["data"].get("verified") is True, "%s: runs verify at %d threads: %r" % (label, th, doc and (doc.get("error") or doc.get("data"))))
                    count("replay equal at %d threads" % th)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    for k, v in sorted(COUNT.items()):
        print("%6d  %s" % (v, k))
    print("oracle_track: %d mismatches" % len(FAILS))
    return 1 if FAILS else 0


def selftest():
    data = b"a,b,t\n1,2.5,0\n3,,1\n-4,7.25,0\n"
    p = provenance(data, ["a", "b"], None)
    assert len(p) == 64 + 80
    n, mn, mx, me, sd = struct.unpack("<Qqqqq", p[64:104])
    f = lambda b: struct.unpack("<d", struct.pack("<q", b))[0]  # noqa: E731
    assert (n, f(mn), f(mx)) == (3, -4.0, 3.0) and abs(f(me) - 0.0) < 1e-12
    cells, rows, counts = out_of_range(p, *read_csv(b"a,b,t\n9,2.5,0\n-5,,1\n"), ["a", "b"])
    assert (cells, rows, counts) == (2, 2, {"a": 2}), (cells, rows, counts)
    assert split_reduced("train=0.80") == {"num": 4, "den": 5}
    print("oracle_track --self: ok")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--self"]:
        sys.exit(selftest())
    if a[:1] == ["--suite"]:
        sys.exit(suite(int(a[1]) if len(a) > 1 else 1))
    print(__doc__)
    sys.exit(2)

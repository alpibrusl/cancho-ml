#!/usr/bin/env python3
"""The oracle of `describe` (docs/serving.md sections 4 and 5): what a model file requires of its caller, read again in Python from the bytes of the file, with
none of the code of tools/ml, lib/pipeline or lib/transforms. The sections are walked by scripts/oracle_st.py's reader of the file level; the schema, the plan,
the meta, the model section and the string table are read here from the layout documents (docs/design.md 4.1, docs/stringtable.md section 2, docs/wiring.md,
docs/wiring-trees.md).

    from oracle_describe import describe, FileBad
    describe(file_bytes, limit=1000, max_bytes=16777216)  ->  the `data` object `ml describe` prints, as a dict

A file that is refused raises oracle_st.FileBad with the tag of the file-level check (the content checks are the library's and are not repeated here: this oracle
describes valid files).
"""

import hashlib
import re
import struct
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import oracle_st as S  # noqa: E402

FileBad = S.FileBad
TYPES = {0: "int", 1: "dec", 2: "float", 3: "categorical"}
ENCODERS = {7: "onehot", 8: "label", 9: "freq", 10: "hash"}


def tb(b):
    """text_or_bytes: a string when the bytes are UTF-8, else {"b64": ...}"""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        import base64
        return {"b64": base64.b64encode(b).decode()}


def read_schema(sc):
    cols, target = struct.unpack_from("<II", sc, 0)
    pos = 8
    out = []
    for _ in range(cols):
        nl = struct.unpack_from("<I", sc, pos)[0]
        pos += 4
        name = sc[pos:pos + nl]
        pos += nl
        ty, scale = struct.unpack_from("<II", sc, pos)
        pos += 8
        out.append((name, ty, scale))
    assert pos == len(sc)
    return out, target


def read_table(t):
    """-> (na markers, [four limits], [(step, payload dict)]) of a string table (docs/stringtable.md section 2)"""
    layout, entries, pool, strings, na_len = struct.unpack_from("<IIQQI", t, 0)
    na = t[28:28 + na_len]
    lims = list(struct.unpack_from("<IIII", t, 28 + na_len))
    pos = 44 + na_len
    out = []
    for _ in range(entries):
        step, ln = struct.unpack_from("<Iq", t, pos)
        out.append((step, S.parse_payload(t[pos + 12:pos + 12 + ln])))
        pos += 12 + ln
    return na, lims, out


def describe(data, limit=1000, max_bytes=16777216):
    sec = S.parse_file(data)
    plan = sec[1]
    cols, target = read_schema(sec[2])
    mdl = sec[4]
    meta = sec[5]
    kind = struct.unpack_from("<I", mdl, 0)[0]
    ptext = plan.decode("utf-8")
    m = re.search(r"^model kind=(\w+)(?: task=(\w+))?", ptext, re.M)
    model, task = m.group(1), m.group(2)
    if model == "linear":
        task = "regression"
    elif model == "logistic":
        task = "classification"
    rows = struct.unpack_from("<q", meta, 0)[0]
    cl = struct.unpack_from("<I", meta, 28)[0]
    pin = meta[32:32 + cl]
    sum_order = None
    if kind >= 4 and len(meta) == 36 + cl:
        sum_order = struct.unpack_from("<I", meta, 32 + cl)[0]
    classes = None
    positive = None
    if task == "classification":
        if kind == 2:
            # constant: tensor record u32 kind | u32 dtype | u32 rank | u64 shape | i64 n, classes, c0, c1, n0, n1
            c0, c1 = struct.unpack_from("<qq", mdl, 20 + 16)
        else:
            # logistic, tree, forest: the labels record closes the payload: i64 c0, c1
            c0, c1 = struct.unpack_from("<qq", mdl, len(mdl) - 16)
        classes = [c0, c1]
        positive = None if kind == 2 else c1
    columns = []
    for j, (name, ty, scale) in enumerate(cols):
        columns.append({"name": tb(name), "type": TYPES[ty], "scale": scale if ty == 1 else None, "role": "target" if j == target else "feature", "required": j != target})
    tname, tty, tscale = cols[target]
    if kind <= 3:
        missing = "refused"
    elif model in ("tree", "forest"):
        missing = "handled"
    else:
        missing = "needs_impute"
    categorical = []
    na_values = []
    reader = None
    if 6 in sec:
        na, lims, entries = read_table(sec[6])
        na_values = [tb(x) for x in na.split(b",")] if na else []
        reader = {"max_categories": lims[0], "max_string_bytes": lims[1], "max_pool_bytes": lims[2], "max_cat_columns": lims[3]}
        for _step, p in entries:
            for name, strs in p["cols"]:
                listed = min(limit, len(strs))
                categorical.append({
                    "column": tb(name), "encoder": ENCODERS[p["kind"]], "categories": len(strs), "buckets": p["par"] if p["kind"] == 10 else None,
                    "listed": listed, "truncated": listed < len(strs), "values": [{"value": tb(s), "count": c} for s, c in strs[:listed]],
                })
    if task == "regression":
        output = {"columns": ["prediction"], "with_proba": None, "label_threshold": None}
    elif kind <= 3:
        output = {"columns": ["label", "probability"], "with_proba": None, "label_threshold": None}
    else:
        output = {"columns": ["label"], "with_proba": ["label", "probability"], "label_threshold": "0.5"}
    return {
        "model": model, "task": task, "kind": kind, "plan": ptext, "rows": rows,
        "file": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "format": "CML1", "version": struct.unpack_from("<I", data, 4)[0],
                 "sections": struct.unpack_from("<I", data, 8)[0], "compiler": tb(pin), "sum_order": sum_order},
        "target": {"name": tb(tname), "type": TYPES[tty], "scale": tscale if tty == 1 else None},
        "columns": columns,
        "request": {"header": [c["name"] for c in columns if c["role"] == "feature"], "optional": [tb(tname)], "extra_columns": "refused", "missing_cells": missing},
        "categorical": categorical, "na_values": na_values, "reader": reader,
        "classes": classes, "positive": positive, "output": output,
        "limits": {"values_per_column": limit, "max_bytes": max_bytes},
    }


FIELD_ORDER = ["model", "task", "kind", "plan", "rows", "file", "target", "columns", "request", "categorical", "na_values", "reader", "classes", "positive", "output", "limits"]

if __name__ == "__main__":
    import json
    print(json.dumps(describe(open(sys.argv[1], "rb").read()), indent=1))

#!/usr/bin/env python3
"""The oracle of the categorical encoders (docs/categorical-encoders.md): sections 2 to 5 written again in Python, from the document, with none of the code of
lib/transforms/catenc.cho, lib/pipeline/mixed.cho or tools/enc-check. It works from the CELLS (the strings of the file), never from ids: its own counting, its own
bytewise sort (Python's `bytes` order), its own rare bucket, `count / n` as Python's integer division (correctly rounded), hashing with `hashlib.sha256`, and the
payload of section 5.2 written with `struct`. The reading of the CSV is the stage-1 oracle's (scripts/oracle_cat.py, judged against the reader by gate C1).

    from oracle_enc import Step, fit, apply, payload, run_plan, render
"""

import hashlib
import struct
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
import oracle_cat  # noqa: E402

OTHER = b"__other__"
ONEHOT, LABEL, FREQ, HASH = 7, 8, 9, 10


class Refusal(Exception):
    """tag, step (1-based), column name (bytes) or None, row (0-based) or None, and for column-kind the 1-based place and what the step needs"""

    def __init__(self, tag, step, column=None, row=None, place=None, needs=None):
        super().__init__(tag)
        self.tag, self.step, self.column, self.row, self.place, self.needs = tag, step, column, row, place, needs


class Step:
    pass


def bucket(s, b):
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big") % b


def effective(vocab, min_count):
    """vocab: [(string, count)] sorted bytewise. -> (list of effective strings sorted, {string: effective index}, effective counts) as section 3.1"""
    kept = [(s, c) for s, c in vocab if c >= min_count]
    merged = [(s, c) for s, c in vocab if c < min_count]
    names = [(s, c) for s, c in kept]
    if merged:
        names.append((OTHER, sum(c for _, c in merged)))
    names.sort(key=lambda t: t[0])
    index = {s: i for i, (s, _) in enumerate(names)}
    mapping = {s: index[s] for s, _ in kept}
    for s, _ in merged:
        mapping[s] = index[OTHER]
    return [s for s, _ in names], mapping, [c for _, c in names]


def fit(kind, step_no, colnames, train, unknown, missing, param, maxcat):
    """train: {column name (bytes): [cell or None]} for the TRAINING rows. -> Step; raises Refusal."""
    st = Step()
    st.kind, st.unknown, st.missing, st.param, st.cols = kind, unknown, missing, param, list(colnames)
    n = len(train[colnames[0]]) if colnames else 0
    if n < 1:
        raise Refusal("ml.input.empty", step_no)
    st.rows = n
    st.vocab = []
    counted = []
    for c in colnames:
        counts = {}
        for x in train[c]:
            if x is not None:
                counts[x] = counts.get(x, 0) + 1
        if kind != HASH and not counts:
            raise Refusal("ml.fit.all-missing", step_no, c)
        counted.append(sorted(counts.items(), key=lambda t: t[0]) if kind != HASH else [])
    for c, v in zip(colnames, counted):
        if kind != HASH:
            if param >= 2 and any(s == OTHER for s, _ in v):
                raise Refusal("ml.input.reserved-category", step_no, c)
            if len(effective(v, param)[0]) > maxcat:
                raise Refusal("ml.limit.categories", step_no, c)
        st.vocab.append(v)
    return st


def apply(st, step_no, cells_by_col, nrows, hash_known=None):
    """cells_by_col: {column name: [cell or None]}; -> (rows of lists of float|None, unknown count). A cell the step's vocabulary lacks is unseen."""
    out = [[] for _ in range(nrows)]
    unknown = 0
    first_refusal = None
    for j, c in enumerate(st.cols):
        if c not in cells_by_col:
            raise Refusal("ml.schema.mismatch", step_no, c)
        v = st.vocab[j]
        if st.kind != HASH:
            names, mapping, ecounts = effective(v, st.param)
            known = {s for s, _ in v}
        ne = len(names) if st.kind != HASH else 0
        width = ne if st.kind == ONEHOT else (st.param if st.kind == HASH else 1)
        for r, cell in enumerate(cells_by_col[c]):
            if cell is None:
                if st.missing == 0:
                    out[r] += [None] * width
                else:
                    out[r] += [-1.0] if st.kind == LABEL else [0.0] * width
                continue
            if st.kind == HASH:
                row = [0.0] * width
                if hash_known is not None and cell not in hash_known[c]:
                    # a string the stored vocabulary does not hold has no id and so no bytes to hash: an all-zero block, counted (section 3.5)
                    unknown += 1
                else:
                    row[bucket(cell, st.param)] = 1.0
                out[r] += row
                continue
            if cell not in known:
                if st.unknown == 1:
                    if first_refusal is None:
                        first_refusal = Refusal("ml.input.unknown-category", step_no, c, r)
                    out[r] += [0.0] * width
                    continue
                unknown += 1
                out[r] += [-1.0] if st.kind == LABEL else [0.0] * width
                continue
            e = mapping[cell]
            if st.kind == ONEHOT:
                row = [0.0] * width
                row[e] = 1.0
                out[r] += row
            elif st.kind == LABEL:
                out[r] += [float(e)]
            else:
                out[r] += [ecounts[e] / st.rows]
    if first_refusal is not None:
        raise first_refusal
    return out, unknown


def out_names(st):
    names = []
    for j, c in enumerate(st.cols):
        if st.kind in (ONEHOT, HASH):
            if st.kind == HASH:
                w = st.param
            else:
                w = len(effective(st.vocab[j], st.param)[0])
            names += [c + b"#" + str(i).encode() for i in range(w)]
        else:
            names.append(c)
    return names


def payload(st):
    out = struct.pack("<IIIIII", st.kind, 1, len(st.cols), st.unknown, st.missing, st.param) + struct.pack("<Q", st.rows)
    for j, c in enumerate(st.cols):
        out += struct.pack("<I", len(c)) + c + struct.pack("<I", len(st.vocab[j]))
        for s, n in st.vocab[j]:
            out += struct.pack("<I", len(s)) + s + struct.pack("<Q", n)
    return out


def bits(x):
    return struct.pack(">d", x).hex()


def fmt_matrix(rows, label, ncols):
    out = ["%s %d %d" % (label, len(rows), ncols)]
    for r in rows:
        out.append(" ".join("?" if x is None else bits(x) for x in r))
    return out



# ------------------------------------------------------------------ a whole plan on a table (what tools/enc-check prints)

NUMERIC_MINMAX = 2


class Plan:
    """steps: a list of dicts. verb 'encode'|'freq'|'hash'|'scale': cols (list of names (bytes) or '*'), and by verb: kind ('onehot'|'label'),
    maxcat (64), unknown (0|1), minc (1), missing (0|1), buckets, and for scale kind 'minmax' (the only numeric step the gates use)."""

    def __init__(self, steps):
        self.steps = steps


def cell_value(cell):
    kind, scale, bits = oracle_cat.numeric_value(cell)
    return struct.unpack("<d", struct.pack("<q", bits))[0]


def hex_matrix(rows, ncols, label):
    out = ["%s %d %d" % (label, len(rows), ncols)]
    for r in rows:
        out.append(" ".join("?" if x is None else "%016x" % (struct.unpack(">Q", struct.pack(">d", x))[0]) for x in r))
    return out


def minmax_payload(cols_in, cols, params):
    """transforms.md section 4: u32 kind | ints record | floats record, ints = [k, cols_in, c_0 .. c_{k-1}], floats rank 2 [k, 2]"""
    ints = [len(cols), cols_in] + list(cols)
    out = struct.pack("<I", NUMERIC_MINMAX) + struct.pack("<II", 1, 1) + struct.pack("<Q", len(ints)) + b"".join(struct.pack("<q", v) for v in ints)
    out += struct.pack("<II", 2, 2) + struct.pack("<QQ", len(cols), 2)
    for lo, d in params:
        out += struct.pack("<dd", lo, d)
    return out


def split_columns(L, target):
    """(numeric feature names, their per-row values (float or None), the categorical names in header order, their per-row cells (bytes or None))"""
    nnames, nvals = [], []
    for j in L.numeric:
        if L.names[j].decode("latin-1") == target:
            continue
        nnames.append(L.names[j])
        nvals.append([None if row[j][0] == "m" else struct.unpack("<d", struct.pack("<q", row[j][1]))[0] for row in L.cells])
    cnames = [L.names[j] for j in sorted(L.place, key=lambda j: L.place[j])]
    ccells = []
    for j in sorted(L.place, key=lambda j: L.place[j]):
        ccells.append([None if row[j][0] == "m" else row[j][1] for row in L.cells])
    return nnames, nvals, cnames, ccells


def run_plan(data, cfg, plan, data2=None, skip2=True):
    """The expected stdout lines of `enc-check` for these files and this plan, and its exit status."""
    try:
        L = oracle_cat.load(data, cfg)
    except oracle_cat.Refused as r:
        return ["load-refused %d %d %d %d" % (r.status, r.line, r.col, r.aux)], 8
    if L.rows == 0 or len(L.numeric) + L.ncat == 0:
        return ["load-refused-matrix ml.input.empty"], 8
    nnames, nvals, cnames, ccells = split_columns(L, cfg.target)
    rows = L.rows
    cat_index = {n: i for i, n in enumerate(cnames)}
    alive = {n: True for n in cnames}
    cur_names = list(nnames)
    cur = [list(c) for c in nvals]
    steps = []  # ("minmax", cols_in, cols, params) or ("cat", Step)
    try:
        for si, st in enumerate(plan.steps, 1):
            verb = st["verb"]
            fixed = {"scale": 1, "freq": 2, "hash": 2, "encode": 0}[verb]
            if st["cols"] == "*":
                want = fixed or 1
                if want == 1:
                    idx = list(range(len(cur_names)))
                    if not idx:
                        raise Refusal("ml.plan.column-kind", si, place=1, needs=1)
                    cls, sel = 1, idx
                else:
                    sel = [n for n in cnames if alive[n]]
                    if not sel:
                        raise Refusal("ml.plan.column-kind", si, place=1, needs=2)
                    cls = 2
            else:
                required, sel = fixed, []
                for place, name in enumerate(sorted(st["cols"]), 1):
                    if name in cur_names:
                        cl, at = 1, cur_names.index(name)
                    elif name in alive and alive[name]:
                        cl, at = 2, name
                    else:
                        raise Refusal("ml.plan.unknown-column", si)
                    if required == 0:
                        required = cl
                    if cl != required:
                        raise Refusal("ml.plan.column-kind", si, place=place, needs=required)
                    sel.append(at)
                cls = required
                sel = sorted(sel, key=(lambda a: a) if cls == 1 else (lambda n: cat_index[n]))
            if cls == 1 and verb == "encode":
                raise Refusal("ml.plan.column-kind", si, place=1, needs=2)  # the gates use encode only on categorical columns; a numeric encode is the old path
            if cls == 1 and verb == "scale" and st.get("minc", 1) != 1:
                raise Refusal("ml.plan.column-kind", si, place=1, needs=2)
            if cls == 1:
                params = []
                for c in sel:
                    vals = [v for v in cur[c] if v is not None]
                    if not vals:
                        raise Refusal("ml.fit.all-missing", si, place=c)
                    lo, hi = min(vals), max(vals)
                    d = hi - lo
                    if d == 0.0:
                        d = 1.0
                    params.append((lo, d))
                for c, (lo, d) in zip(sel, params):
                    cur[c] = [None if v is None else (v - lo) / d for v in cur[c]]
                steps.append(("minmax", len(cur_names), sel, params))
                continue
            if verb == "encode":
                kind = {"onehot": ONEHOT, "label": LABEL}[st["kind"]]
            else:
                kind = {"freq": FREQ, "hash": HASH}[verb]
            unknown = 0 if kind == HASH else st.get("unknown", 0)
            par = st["buckets"] if kind == HASH else st.get("minc", 1)
            maxcat = 64 if kind == HASH else st.get("maxcat", 64)
            train = {n: ccells[cat_index[n]] for n in sel}
            try:
                step = fit(kind, si, sel, train, unknown, st.get("missing", 0), par, maxcat)
            except Refusal as r:
                if r.column is not None:
                    r.column_index = cat_index[r.column]
                raise
            step.reader_index = [cat_index[n] for n in sel]
            block, unk = apply(step, si, train, rows)
            for j, n in enumerate(out_names(step)):
                cur_names.append(n)
                cur.append([block[r][j] for r in range(rows)])
            for n in sel:
                alive[n] = False
            steps.append(("cat", step))
    except Refusal as r:
        return refusal_lines(r, cnames, "")
    body = b""
    for s in steps:
        if s[0] == "minmax":
            p = minmax_payload(s[1], s[2], s[3])
        else:
            p = payload(s[1])
        body += struct.pack("<Q", len(p)) + p
    pay = struct.pack("<I", len(steps)) + body
    lines = ["payload " + pay.hex(), "names" + "".join(" " + n.decode("latin-1") for n in cur_names), "remaining" + "".join(" " + n.decode("latin-1") for n in cnames if alive[n])]
    lines += hex_matrix([[col[r] for col in cur] for r in range(rows)], len(cur), "train")
    if data2 is None:
        return lines, 0
    # predict: file2 read through the vocabulary of file1 (the reader's vocabulary: every string of the categorical columns of file1)
    vocab_in = L.vocab
    try:
        L2 = oracle_cat.load(data2, cfg, vocab_in=vocab_in, skip=skip2)
    except oracle_cat.Refused as r:
        return lines + ["p-load-refused %d %d %d %d" % (r.status, r.line, r.col, r.aux)], 8
    if L2.rows == 0:
        return lines + ["p-load-refused-matrix ml.input.empty"], 8
    n2names, n2vals, c2names, c2cells = split_columns(L2, cfg.target if not skip2 else "\0")
    rows2 = L2.rows
    cells2 = {n: c2cells[i] for i, n in enumerate(c2names)}
    # the numeric columns of file2 are the features in file order, as the tool takes them
    cur2 = [list(c) for c in n2vals]
    cur2_names = list(n2names)
    unknown_total = 0
    reader_sets = {n: set(vocab_in[i][1]) for i, n in enumerate(cnames)}
    try:
        for si, s in enumerate(steps, 1):
            if s[0] == "minmax":
                for c, (lo, d) in zip(s[2], s[3]):
                    cur2[c] = [None if v is None else (v - lo) / d for v in cur2[c]]
            else:
                step = s[1]
                block, unk = apply(step, si, cells2, rows2, reader_sets)
                unknown_total += unk
                for j in range(len(block[0]) if block else 0):
                    cur2.append([block[r][j] for r in range(rows2)])
    except Refusal as r:
        return lines + refusal_lines(r, cnames, "p-")[0], 8
    lines += hex_matrix([[col[r] for col in cur2] for r in range(rows2)], len(cur2), "p-matrix")
    lines.append("p-unknown %d" % unknown_total)
    return lines, 0


def refusal_lines(r, cnames, pre):
    a = r.step
    b, c = 0, 0
    if r.tag == "ml.plan.column-kind":
        b, c = r.place, r.needs
    elif r.tag == "ml.plan.unknown-column":
        b, c = 0, 0
    elif r.tag == "ml.fit.all-missing" and r.column is None:
        b = r.place
    elif r.tag in ("ml.limit.categories", "ml.fit.all-missing", "ml.input.reserved-category", "ml.input.unknown-category") and r.column is not None:
        b = -1 - cnames.index(r.column)
        c = r.row if r.row is not None else 0
    lines = ["%srefused %s %d %d %d" % (pre, r.tag, a, b, c)]
    if pre == "" and b < 0 and r.tag != "ml.plan.column-kind" and r.column is not None:
        lines.append("refused-column " + r.column.decode("latin-1"))
    return lines, 8


if __name__ == "__main__":
    # a small self-check against the documented known answers
    assert bucket(b"abc", 2 ** 64) == 0xBA7816BF8F01CFEA
    print("oracle_enc: ok")

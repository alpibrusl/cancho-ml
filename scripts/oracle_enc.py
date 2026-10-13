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
    for c in colnames:
        cells = train[c]
        counts = {}
        for x in cells:
            if x is not None:
                counts[x] = counts.get(x, 0) + 1
        if kind == HASH:
            st.vocab.append([])
            continue
        if not counts:
            raise Refusal("ml.fit.all-missing", step_no, c)
        v = sorted(counts.items(), key=lambda t: t[0])
        if param >= 2 and any(s == OTHER for s, _ in v):
            raise Refusal("ml.input.reserved-category", step_no, c)
        ne = len(effective(v, param)[0])
        if ne > maxcat:
            raise Refusal("ml.limit.categories", step_no, c)
        st.vocab.append(v)
    return st


def apply(st, step_no, cells_by_col, nrows):
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


if __name__ == "__main__":
    # a small self-check against the documented known answers
    assert bucket(b"abc", 2 ** 64 - 1 + 1) == 0xBA7816BF8F01CFEA
    print("oracle_enc: ok")

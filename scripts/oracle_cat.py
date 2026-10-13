#!/usr/bin/env python3
"""The oracle of the categorical columns (docs/categorical.md): the rules of sections 2 to 7 written again in Python, from the document, with none of the
code of tools/ml/csv.cho, lib/core/dict.cho or lib/core/catmat.cho. It reads the bytes of a CSV file by RFC 4180 as docs/reader.md section 2.2 states it (lines
split on LF, one CR dropped before each LF, a BOM dropped from the first line, blank lines outside quotes skipped, a quote opens a field only at its start,
`""` inside quotes is one quote), builds the vocabulary with Python's `bytes` order, applies the checks of section 5.1 and the limits of section 4, and prints
what `reader-check --mode cat` prints, line for line, so the two outputs are compared as text (scripts/gate_cat.py).

    from oracle_cat import Cfg, run, render
"""

import struct
from fractions import Fraction
import re

DEC = re.compile(rb"^[+-]?[0-9]+(\.[0-9]+)?$")


class Cfg:
    def __init__(self, cats=(), na=(), max_cats=64, max_str=255, max_pool=1048576, max_catcols=128, drops=(), floats=(), target="", max_rows=10000000):
        self.cats = list(cats)
        self.na = [m if isinstance(m, bytes) else m.encode() for m in na]
        self.max_cats = max_cats
        self.max_str = max_str
        self.max_pool = max_pool
        self.max_catcols = max_catcols
        self.drops = list(drops)
        self.floats = list(floats)
        self.target = target
        self.max_rows = max_rows


class Refused(Exception):
    """status, line, col, aux as the report of ml.csv.Report"""

    def __init__(self, status, line=0, col=0, aux=0):
        super().__init__((status, line, col, aux))
        self.status, self.line, self.col, self.aux = status, line, col, aux


def utf8_offset(cell):
    """-1 if well-formed UTF-8, else the offset of the first byte that is not (Python's decoder is strict in the same way: no overlong forms, no
    surrogates, nothing past U+10FFFF); the offset is where the decoder says the error starts"""
    try:
        cell.decode("utf-8")
        return -1
    except UnicodeDecodeError as e:
        return e.start


def scan_line(line, inside, seps):
    end = len(line)
    p = 0
    n = seps
    bad = False
    while p < end and not bad:
        found = line.find(b'"', p)
        if not inside:
            q = found if found >= 0 else end
            n += line[p:q].count(b",")
            if q == end:
                p = end
            elif q == 0 or line[q - 1] == 44:
                inside = True
                p = q + 1
            else:
                p = q + 1
        elif found < 0:
            p = end
        else:
            q = found
            if q + 1 < end and line[q + 1] == 34:
                p = q + 2
            else:
                inside = False
                if q + 1 == end:
                    p = end
                elif line[q + 1] == 44:
                    n += 1
                    p = q + 2
                else:
                    bad = True
    return inside, n, bad


def split_fields(rec):
    """[(first, last, quoted)] of a whole record"""
    end = len(rec)
    p = 0
    out = []
    while True:
        if p < end and rec[p] == 34:
            q = p + 1
            while True:
                at = rec.find(b'"', q)
                assert at >= 0
                if at + 1 < end and rec[at + 1] == 34:
                    q = at + 2
                    continue
                break
            first, last, quoted = p + 1, at, True
            nxt = end + 1 if at + 1 == end else at + 2
        else:
            q = rec.find(b",", p)
            if q >= 0:
                first, last, quoted, nxt = p, q, False, q + 1
            else:
                first, last, quoted, nxt = p, end, False, end + 1
        out.append((first, last, quoted))
        if nxt > end:
            break
        p = nxt
    return out


def records(data):
    """[(line number of the first line, the record's bytes)], or raises Refused (17 bad quote, 18 unterminated)"""
    raw = data.split(b"\n")
    if raw and raw[-1] == b"":
        raw.pop()
    recs = []
    have = False
    rec = b""
    rec_line = 0
    inside = False
    seps = 0
    for number, line in enumerate(raw, 1):
        if number == 1 and line[:3] == b"\xef\xbb\xbf":
            line = line[3:]
        if line.endswith(b"\r"):
            line = line[:-1]
        if len(line) == 0 and not have:
            continue
        if not have:
            rec_line = number
            seps = 0
            inside = False
            rec = line
        else:
            rec = rec + b"\n" + line
        have = True
        inside, seps, bad = scan_line(line, inside, seps) if (b'"' in line or inside) else (inside, line.count(b","), False)
        if bad:
            raise Refused(17, number)
        if not inside:
            recs.append((rec_line, rec, seps))
            have = False
    if have:
        raise Refused(18, rec_line)
    return recs


def cell_of(rec, f):
    first, last, quoted = f
    c = rec[first:last]
    if quoted:
        c = c.replace(b'""', b'"')
    return c


def dump(vocab):
    """vocab: [(name bytes, [category bytes])]"""
    out = struct.pack("<I", len(vocab))
    for name, cats in vocab:
        out += struct.pack("<I", len(name)) + name + struct.pack("<I", len(cats))
        for c in cats:
            out += struct.pack("<I", len(c)) + c
    return out


class Loaded:
    pass


def header_names(recs):
    if not recs:
        raise Refused(1)
    line, rec, seps = recs[0]
    names = [cell_of(rec, f) for f in split_fields(rec)]
    return line, names


def numeric_value(cell):
    """(kind, scale, double bits) of a numeric cell or raises Refused(8/9 ...) as (status, aux); the generators only write ints and short decimals"""
    low = cell.lower().lstrip(b"+-")
    if low in (b"nan", b"inf", b"infinity"):
        raise Refused(9, aux=0)
    if not DEC.match(cell):
        raise Refused(8, aux=2)
    if b"." in cell:
        scale = len(cell.split(b".")[1])
        kind = 2
    else:
        scale = 0
        kind = 1
    v = float(Fraction(cell.decode()))
    if v == 0.0:
        v = 0.0
    return kind, scale, struct.unpack("<q", struct.pack("<d", v))[0]


def is_missing(cell, na):
    return len(cell) == 0 or cell in na


def load(data, cfg, vocab_in=None, skip=False):
    """Pass 1 and pass 2 of a load, as docs/categorical.md describes. Returns a Loaded or raises Refused / RefusedMatrix.
    vocab_in: [(name, [categories])] (a predict) or None (a fit)."""
    recs = records(data)
    if not recs:
        raise Refused(1)
    hline, names = header_names(recs)
    cols = len(names)
    # check_names
    for i, n in enumerate(names):
        if n == b"" or n in names[:i]:
            raise Refused(5, hline, i, cols)
    idx = {n: i for i, n in enumerate(names)}
    drops = set()
    for k, d in enumerate(cfg.drops, 1):
        if d.encode() not in idx:
            raise Refused(12, hline, 0, k)
        drops.add(idx[d.encode()])
    floats = set()
    for k, d in enumerate(cfg.floats, 1):
        if d.encode() not in idx:
            raise Refused(19, hline, 0, k)
        floats.add(idx[d.encode()])
    tcol = idx.get(cfg.target.encode(), -1) if cfg.target else -1
    if not skip and tcol >= 0 and tcol in drops:
        raise Refused(14, hline, tcol)
    # which columns are categorical, in file order
    place = {}
    if vocab_in is None:
        want = set()
        for k, c in enumerate(cfg.cats, 1):
            if c.encode() not in idx:
                raise Refused(25, hline, 0, k)
            want.add(idx[c.encode()])
        for j in sorted(want):
            if j in drops:
                raise Refused(26, hline, j)
            if j == tcol:
                raise Refused(27, hline, j)
            if j in floats:
                raise Refused(28, hline, j)
        if len(want) > cfg.max_catcols:
            raise Refused(24, hline, 0, len(want))
        for n, j in enumerate(sorted(want)):
            place[j] = n
    else:
        for c, (name, _) in enumerate(vocab_in):
            j = idx.get(name, -1)
            if j < 0 or j in drops:
                raise Refused(29, hline, 0, c + 1)
            if j == tcol:
                raise Refused(27, hline, j)
            place[j] = c
    ncat = len(place)
    cat_names = [names[j] for j in sorted(place, key=lambda j: place[j])]
    # pass 1
    dicts = [set() for _ in range(ncat)]
    pool = 0
    kinds = [0] * cols
    scales = [0] * cols
    for j in place:
        kinds[j] = 4
    for j in floats:
        kinds[j] = 3
    rows = 0
    cells_by_row = []
    for line, rec, seps in recs[1:]:
        rows += 1
        if rows > cfg.max_rows:
            raise Refused(7, line, 0, rows)
        if seps + 1 != cols:
            raise Refused(3, line)
        fl = split_fields(rec)
        assert len(fl) == cols
        row = []
        for i in range(cols):
            if i in drops or (skip and i == tcol):
                row.append(None)
                continue
            cell = cell_of(rec, fl[i])
            if i in place:
                cj = place[i]
                if is_missing(cell, cfg.na):
                    row.append(("m",))
                    continue
                if len(cell) > cfg.max_str:
                    raise Refused(21, line, i, len(cell))
                off = utf8_offset(cell)
                if off >= 0:
                    raise Refused(20, line, i, off)
                if vocab_in is None:
                    if cell not in dicts[cj]:
                        if len(dicts[cj]) >= cfg.max_cats:
                            raise Refused(22, line, i, cfg.max_cats)
                        if pool + len(cell) > cfg.max_pool:
                            raise Refused(23, line, i, pool + len(cell))
                        dicts[cj].add(cell)
                        pool += len(cell)
                row.append(("c", cell))
            else:
                if len(cell) == 0:
                    row.append(("m",))
                    continue
                try:
                    kind, scale, bits = numeric_value(cell)
                except Refused as r:
                    raise Refused(r.status, line, i, r.aux)
                if i in floats:
                    kind = 3
                kinds[i] = max(kinds[i], kind)
                if kind < 3:
                    scales[i] = max(scales[i], scale)
                row.append(("n", bits))
        cells_by_row.append(row)
    if vocab_in is None:
        vocab = [(cat_names[c], sorted(dicts[c])) for c in range(ncat)]
    else:
        vocab = vocab_in
    L = Loaded()
    L.names, L.cols, L.rows, L.kinds, L.scales = names, cols, rows, kinds, scales
    L.drops, L.skip, L.tcol, L.place, L.vocab, L.cells = drops, skip, tcol, place, vocab, cells_by_row
    L.ncat = ncat
    L.numeric = [j for j in range(cols) if j not in drops and j not in place and not (skip and j == tcol)]
    L.na = cfg.na
    return L


class RefusedMatrix(Exception):
    pass


def render(L, mode="fit"):
    """The lines reader-check prints for a load, with the key prefix of `mode`: 'fit' prints the shape, the numeric cells, the ids and the vocabulary; 'predict' the
    ids (p-rows, p-id, p-unseen)."""
    out = []
    if L.rows == 0 or len(L.numeric) + L.ncat == 0:
        return ["refused-matrix ml.input.empty"] if mode == "fit" else ["p-refused-matrix ml.input.empty"]
    ids = []
    unseen = [0] * L.ncat
    lookup = {}
    for c, (_, cats) in enumerate(L.vocab):
        lookup[c] = {s: i for i, s in enumerate(cats)}
    for row in L.cells:
        r = []
        for j in sorted(L.place, key=lambda j: L.place[j]):
            cell = row[j]
            c = L.place[j]
            if cell[0] == "m":
                r.append(-1)
            else:
                i = lookup[c].get(cell[1])
                if i is None:
                    i = -2
                    unseen[c] += 1
                r.append(i)
        ids.append(r)
    pre = "" if mode == "fit" else "p-"
    if mode == "fit":
        kept = [j for j in range(L.cols) if j not in L.drops]
        out.append("rows %d" % L.rows)
        out.append("cols %d" % len(kept))
        for k, j in enumerate(kept):
            out.append("col %d %d %d %s" % (k, L.kinds[j], L.scales[j] if L.kinds[j] in (1, 2) else 0, L.names[j].decode("latin-1")))
        for r in range(L.rows):
            for c, j in enumerate(L.numeric):
                cell = L.cells[r][j]
                out.append("cell %d %d %s" % (r, c, "M" if cell[0] == "m" else "%016x" % (cell[1] & 0xFFFFFFFFFFFFFFFF)))
        out.append("rows %d" % L.rows)
    else:
        out.append("p-rows %d" % L.rows)
    for r in range(L.rows):
        for c in range(L.ncat):
            out.append("%sid %d %d %d %d" % (pre, r, c, ids[r][c], 1 if ids[r][c] == -1 else 0))
    for c in range(L.ncat):
        out.append("%sunseen %d %d" % (pre, c, unseen[c]))
    if mode == "fit":
        out.append("vocab %d" % L.ncat)
        for c, (name, cats) in enumerate(L.vocab):
            out.append("vcol %d %d %s" % (c, len(cats), name.decode("latin-1")))
            for i, s in enumerate(cats):
                out.append("cat %d %d %s" % (c, i, s.hex()))
        out.append("dump %s" % dump(L.vocab).hex())
    return out


def refusal_line(r, pre=""):
    return "%srefused %d %d %d %d" % (pre, r.status, r.line, r.col, r.aux)


def run(data, cfg, data2=None, skip2=False):
    """The expected stdout lines of `reader-check --mode cat` (with --predict when data2 is given) and the exit status."""
    try:
        L = load(data, cfg)
    except Refused as r:
        return [refusal_line(r)], 8
    lines = render(L, "fit")
    if lines and lines[0].startswith("refused-matrix"):
        return lines, 8
    if data2 is not None:
        try:
            L2 = load(data2, cfg, vocab_in=L.vocab, skip=skip2)
        except Refused as r:
            return lines + [refusal_line(r, "p-")], 8
        p = render(L2, "predict")
        lines += p
        if p and p[0].startswith("p-refused-matrix"):
            return lines, 8
    return lines, 0

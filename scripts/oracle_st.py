#!/usr/bin/env python3
"""The oracle of the string table (docs/stringtable.md): sections 2 to 4 written again in Python, from the document, with none of the code of
lib/transforms/strtab.cho, lib/pipeline/file.cho or tools/ml. It writes the table, the stubs of the `steps` section, the schema (with the type 3), the meta and the
whole `CML1` file for a fit (struct for the bytes, hashlib for the digests, its own counting from the cells through scripts/oracle_enc.py), and it READS a table
and a file independently: `table_verdict` answers the reason (a word of section 7) a table is refused with, or None, and `file_verdict` the first of the file level
checks (magic, version, structure, checksum) that refuses a file.

    from oracle_st import make_sections, schema_bytes, meta_bytes, build_file, table_verdict, file_verdict, parse_file
"""

import hashlib
import struct

MAX_TABLE = 12 * 1024 * 1024
MAX_POOL = 8 * 1024 * 1024
MAX_STRINGS = 131072
MAX_ENTRIES = 64
MAX_NA = 16383
OTHER = b"__other__"


def sha(b):
    return hashlib.sha256(b).digest()


class Limit(Exception):
    """the fit would write a table over a bound: what (1 strings, 2 pool bytes, 3 table bytes), size, max"""

    def __init__(self, what, size, maximum):
        super().__init__(what)
        self.what, self.size, self.max = what, size, maximum


class Bad(Exception):
    pass


def parse_payload(p):
    """A categorical step payload (categorical-encoders.md section 5.2) as catenc.decode reads it: a dict, or Bad. Written from the document."""
    n = len(p)
    if n < 32:
        raise Bad("short")
    kind, layout, k, unknown, missing, par = struct.unpack_from("<IIIIII", p, 0)
    rows = struct.unpack_from("<q", p, 24)[0]
    if not (7 <= kind <= 10) or layout != 1 or not (1 <= k <= 4096) or unknown > 1 or missing > 1 or not (1 <= rows <= 2 ** 31 - 1):
        raise Bad("head")
    if kind == 10:
        if not (2 <= par <= 1024) or unknown != 0:
            raise Bad("buckets")
    elif not (1 <= par <= 1000000):
        raise Bad("min-count")
    pos = 32
    cols = []
    names = set()
    for _ in range(k):
        if n - pos < 4:
            raise Bad("trunc")
        nl = struct.unpack_from("<I", p, pos)[0]
        pos += 4
        if nl < 1 or nl > 255 or n - pos < nl + 4:
            raise Bad("name")
        name = p[pos:pos + nl]
        if name in names:
            raise Bad("dup")
        names.add(name)
        pos += nl
        m = struct.unpack_from("<I", p, pos)[0]
        pos += 4
        if kind == 10:
            if m != 0:
                raise Bad("hash strings")
        elif m < 1 or m > 1024:
            raise Bad("ncat")
        strs = []
        total = 0
        prev = None
        for _ in range(m):
            if n - pos < 4:
                raise Bad("trunc")
            sl = struct.unpack_from("<I", p, pos)[0]
            pos += 4
            if sl < 1 or sl > 65536 or n - pos < sl + 8:
                raise Bad("string")
            s = p[pos:pos + sl]
            if prev is not None and not prev < s:
                raise Bad("order")
            if par >= 2 and kind != 10 and s == OTHER:
                raise Bad("reserved")
            c = struct.unpack_from("<q", p, pos + sl)[0]
            if c < 1 or c > rows:
                raise Bad("count")
            total += c
            if total > rows:
                raise Bad("counts")
            prev = s
            pos += sl + 8
            strs.append((s, c))
        cols.append((name, strs))
    if pos != n:
        raise Bad("trailing")
    return {"kind": kind, "k": k, "unknown": unknown, "missing": missing, "par": par, "rows": rows, "cols": cols}


def parse_steps(payload):
    """the steps payload of a fit: u32 n, then n times u64 length | step -> list of step byte strings"""
    n = struct.unpack_from("<I", payload, 0)[0]
    pos = 4
    out = []
    for _ in range(n):
        ln = struct.unpack_from("<Q", payload, pos)[0]
        pos += 8
        out.append(payload[pos:pos + ln])
        pos += ln
    assert pos == len(payload)
    return out


def make_sections(payload, na, lims):
    """The `steps` section of a file and the string table made from the payload of a fit. na: the --na-values text (bytes, maybe empty); lims: (max_categories,
    max_string_bytes, max_pool_bytes, max_cat_columns). -> (section, table); table is b'' when no step is categorical. Raises Limit."""
    steps = parse_steps(payload)
    section = struct.pack("<I", len(steps))
    ents = []
    for i, s in enumerate(steps, 1):
        kind = struct.unpack_from("<I", s, 0)[0]
        if kind >= 7:
            ents.append((i, s))
            section += struct.pack("<QII", 8, kind, 2)
        else:
            section += struct.pack("<Q", len(s)) + s
    if not ents:
        return section, b""
    strings = pool = 0
    for _, s in ents:
        for _, strs in [(n, st) for n, st in parse_payload(s)["cols"]]:
            strings += len(strs)
            pool += sum(len(x) for x, _ in strs)
    if strings > MAX_STRINGS:
        raise Limit(1, strings, MAX_STRINGS)
    if pool > MAX_POOL:
        raise Limit(2, pool, MAX_POOL)
    size = 28 + len(na) + 16 + sum(12 + len(s) for _, s in ents) + 32
    if len(na) > MAX_NA or size > MAX_TABLE or len(ents) > MAX_ENTRIES:
        raise Limit(3, size, MAX_TABLE)
    t = struct.pack("<IIQQI", 1, len(ents), pool, strings, len(na)) + na + struct.pack("<IIII", *lims)
    for i, s in ents:
        t += struct.pack("<IQ", i, len(s)) + s
    return section, t + sha(t)


def schema_bytes(L, target):
    """pipeline.schema's section for the kept columns of a loaded table (oracle_cat.Loaded) in file order: a categorical column has type 3, scale 0"""
    cols = [j for j in range(L.cols) if j not in L.drops]
    out = struct.pack("<II", len(cols), [L.names[j] for j in cols].index(target.encode()))
    for j in cols:
        if j in L.place:
            ty, sc = 3, 0
        elif L.kinds[j] == 1:
            ty, sc = 0, 0
        elif L.kinds[j] == 2:
            ty, sc = 1, L.scales[j]
        else:
            ty, sc = 2, 0
        out += struct.pack("<I", len(L.names[j])) + L.names[j] + struct.pack("<II", ty, sc)
    return out, len(cols)


def meta_bytes(rows, columns, seed, pin, model_kind):
    m = struct.pack("<QQQII", rows, columns, seed, 0, len(pin)) + pin
    if model_kind >= 4:
        m += struct.pack("<I", 3 if model_kind == 7 else 2)
    return m


def build_file(plan, schema, steps, model, meta, table=b""):
    secs = [(1, plan), (2, schema), (3, steps), (4, model), (5, meta)]
    if table:
        secs.append((6, table))
    body = b"CML1" + struct.pack("<II", 1, len(secs)) + b"".join(struct.pack("<IQ", t, len(p)) + p for t, p in secs)
    return body + sha(body)


class FileBad(Exception):
    """the first file-level check that refuses: the rule tag"""

    def __init__(self, tag):
        super().__init__(tag)
        self.tag = tag


def parse_file(data):
    """-> {tag: payload} for the sections, or raises FileBad (bad-magic, version, truncated, checksum, malformed) in the order of design.md 4.1"""
    n = len(data)
    if n < 4:
        raise FileBad("ml.model.truncated")
    if data[:4] != b"CML1":
        raise FileBad("ml.model.bad-magic")
    if n < 8:
        raise FileBad("ml.model.truncated")
    if struct.unpack_from("<I", data, 4)[0] != 1:
        raise FileBad("ml.model.version")
    body = n - 32
    if body < 12:
        raise FileBad("ml.model.truncated")
    count = struct.unpack_from("<I", data, 8)[0]
    if count not in (5, 6):
        raise FileBad("ml.model.malformed")
    pos = 12
    out = {}
    for s in range(count):
        if pos + 12 > body:
            raise FileBad("ml.model.truncated")
        tag, ln = struct.unpack_from("<Iq", data, pos)
        if ln < 0 or ln > body - pos - 12:
            raise FileBad("ml.model.truncated")
        if tag != s + 1:
            raise FileBad("ml.model.malformed")
        out[tag] = data[pos + 12:pos + 12 + ln]
        pos += 12 + ln
    if pos != body:
        raise FileBad("ml.model.malformed")
    if sha(data[:body]) != data[body:]:
        raise FileBad("ml.model.checksum")
    return out


def file_verdict(data):
    """None if the file-level checks pass, else the tag"""
    try:
        parse_file(data)
    except FileBad as e:
        return e.tag
    return None


def utf8_ok(b):
    try:
        b.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def table_verdict(t):
    """The reason (a word of docs/stringtable.md section 7) a string table is refused with, in the order of section 4, or None if it is accepted. Written from the
    document; nothing is shared with strtab.validate."""
    n = len(t)
    if n > MAX_TABLE or n < 28 + 16 + 32:
        return "header"
    if sha(t[:n - 32]) != t[n - 32:]:
        return "digest"
    body = n - 32
    if struct.unpack_from("<I", t, 0)[0] != 1:
        return "layout"
    ne = struct.unpack_from("<I", t, 4)[0]
    if ne < 1 or ne > MAX_ENTRIES:
        return "entries"
    pool, strs = struct.unpack_from("<qq", t, 8)
    nal = struct.unpack_from("<I", t, 24)[0]
    if pool < 0 or pool > MAX_POOL or strs < 0 or strs > MAX_STRINGS or nal > MAX_NA or nal > body - 44:
        return "header"
    na = t[28:28 + nal]
    if nal > 0:
        ms = na.split(b",")
        if len(ms) > 64 or any(len(m) < 1 or len(m) > 255 for m in ms):
            return "header"
    lc, ls, lp, lcol = struct.unpack_from("<IIII", t, 28 + nal)
    if not (1 <= lc <= 1024) or not (1 <= ls <= 65536) or not (1 <= lp <= 8388608) or not (1 <= lcol <= 4096):
        return "header"
    pos = 44 + nal
    prev = 0
    sum_pool = sum_strs = sum_cols = 0
    seen = []
    for _ in range(ne):
        if body - pos < 12:
            return "length"
        step = struct.unpack_from("<I", t, pos)[0]
        ln = struct.unpack_from("<q", t, pos + 4)[0]
        pos += 12
        if step <= prev or step > MAX_ENTRIES:
            return "step"
        if ln < 32 or ln > body - pos:
            return "length"
        try:
            pl = parse_payload(t[pos:pos + ln])
        except Bad:
            return "payload"
        if sum_cols + pl["k"] > lcol:
            return "limits"
        for name, st in pl["cols"]:
            if not utf8_ok(name):
                return "utf8"
            if name in seen:
                return "schema"
            if len(st) > lc:
                return "limits"
            for s, _ in st:
                if len(s) > ls:
                    return "limits"
                if not utf8_ok(s):
                    return "utf8"
                sum_pool += len(s)
            sum_strs += len(st)
        # the names of this step join the seen ones only after the whole step passed
        seen += [name for name, _ in pl["cols"]]
        sum_cols += pl["k"]
        prev = step
        pos += ln
    if pos != body:
        return "trailing"
    if sum_pool != pool or sum_strs != strs:
        return "totals"
    if sum_pool > lp or sum_cols > lcol:
        return "limits"
    return None


def reseal_table(t):
    return t[:-32] + sha(t[:-32])


def reseal_file(d):
    return d[:-32] + sha(d[:-32])


if __name__ == "__main__":
    # a small self-check against the layout of the document
    p = struct.pack("<IIIIII", 7, 1, 1, 0, 0, 1) + struct.pack("<q", 3) + struct.pack("<I", 1) + b"c" + struct.pack("<I", 2) + struct.pack("<I", 1) + b"a" + struct.pack("<q", 2) + struct.pack("<I", 1) + b"b" + struct.pack("<q", 1)
    steps = struct.pack("<II", 1, 0)[:4] + struct.pack("<Q", len(p)) + p
    sec, tab = make_sections(steps, b"NA", (64, 255, 1048576, 128))
    assert sec == struct.pack("<I", 1) + struct.pack("<QII", 8, 7, 2)
    assert table_verdict(tab) is None
    assert tab[:4] == struct.pack("<I", 1) and tab[-32:] == sha(tab[:-32])
    bad = bytearray(tab)
    bad[10] ^= 1
    assert table_verdict(bytes(bad)) == "digest"
    print("oracle_st: ok")

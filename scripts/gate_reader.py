#!/usr/bin/env python3
"""The gates of docs/reader.md section 5 that run the reader on files: RD2 (differential), RD3 (matrix and bitmap), RD4 (limits),
RD5 (CSV shape) and RD6 (no trap). The reader is `build/reader-check` (scripts/build_reader.sh); the expected values come from Python:
`float()` (correctly rounded), `fractions`, `csv`, and a model of the cell grammar written here from docs/reader.md section 2.1, not
from the Cancho source.

    python3 scripts/gate_reader.py diff [--cells N] [--seed S]
    python3 scripts/gate_reader.py matrix | limits | csv | fuzz [--runs N]
    python3 scripts/gate_reader.py all

Exit 0 when every difference count is 0. READER=path overrides the binary. Machine, load and counts are printed.
"""
import csv as pycsv
import io
import math
import os
import pathlib
import random
import re
import resource
import struct
import subprocess
import sys
import tempfile
import time
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parent.parent
READER = os.environ.get("READER", str(ROOT / "build" / "reader-check"))
TRAPS = {132, 133, 134, 136, 139}

# ------------------------------------------------------------------ the model of a cell (docs/reader.md section 2.1)

WORD = re.compile(rb"[+-]?(nan|inf|infinity)\Z", re.I)
DEC = re.compile(rb"([+-]?)([0-9]+)(?:\.([0-9]+))?\Z")
FLT = re.compile(rb"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
MANT = re.compile(rb"[+-]?([0-9]*\.?[0-9]*)")


def bits(x):
    return struct.unpack(">Q", struct.pack(">d", x))[0]


def spec(cell: bytes):
    """('missing',) | ('ok', kind, value, bits, scale) | ('refuse', status, aux). status/aux as ml.csv.Report."""
    if len(cell) == 0:
        return ("missing",)
    if WORD.match(cell):
        return ("refuse", 9, 0)
    m = DEC.match(cell)
    if m:
        ip, fp = m.group(2), m.group(3)
        idig = len(ip.lstrip(b"0"))
        frac = len(fp) if fp else 0
        if idig + frac <= 12:
            v = float(cell.decode())
            v = 0.0 if v == 0 else v
            return ("ok", 1 if fp is None else 2, v, bits(v), frac)
    if len(cell) > 1100:
        return ("refuse", 8, 4)
    if not FLT.match(cell):
        return ("refuse", 8, 2)
    v = float(cell.decode())
    if math.isinf(v):
        return ("refuse", 9, 1)
    mant = MANT.match(cell).group(1)
    if v == 0.0:
        if re.search(rb"[1-9]", mant):
            return ("refuse", 8, 3)
        return ("ok", 3, 0.0, 0, 0)
    return ("ok", 3, v, bits(v), 0)


# ------------------------------------------------------------------ generated cells

def exact_decimal(fr: Fraction) -> bytes:
    """A dyadic rational as the digits of an integer and a negative exponent (<= 767 significant digits for a halfway case)."""
    d = fr.denominator
    k = d.bit_length() - 1
    assert d == 1 << k
    num = fr.numerator * 5 ** k
    return b"%de-%d" % (num, k) if k else b"%d" % num


def halfway(x: float):
    """The exact midpoint of x and its successor, and the two decimals one unit of the last digit either side."""
    y = math.nextafter(x, math.inf)
    fy = Fraction(2) ** 1024 if math.isinf(y) else Fraction(y)
    mid = (Fraction(x) + fy) / 2
    d = mid.denominator
    k = d.bit_length() - 1
    num = mid.numerator * 5 ** k
    out = [b"%de-%d" % (num, k), b"%de-%d" % (num - 1, k), b"%de-%d" % (num + 1, k)]
    return out


def random_double(rnd):
    while True:
        b = rnd.getrandbits(64)
        x = struct.unpack(">d", struct.pack(">Q", b))[0]
        if math.isfinite(x):
            return x


def accepted_cells(rnd, n):
    """About n cells of every category the gate names; all of them are read by the reader or refused by the model; the model decides."""
    out = []
    # the fixed ones
    fixed = """0 -0 +0 0.0 -0.0 +0.0 0e5 -0e-5 -0.0e+10 0.000000000000000000000000000000000000 -0.000000000000000000000000000000000000
.5 5. +5 -.5 +.5e1 5.e3 1E5 1e+5 1e-5 1E+05 0001.5 1.5000 00000000000000000000001 1.000000000000 123456789012 1234567890123 12345678901234567890
9007199254740991 9007199254740992 9007199254740993 9007199254740994 9007199254740992.5 9007199254740993.5 4503599627370495.5
1e22 1e23 1e-22 1e-23 1e21 123456789012345678e5 0.1 0.2 0.3 0.7 1.1 2.675 1e15 1e16 1e17 123456789.123456789
4.9406564584124654e-324 5e-324 1e-323 2.2250738585072014e-308 2.2250738585072011e-308 2.225073858507201e-308
1.7976931348623157e308 1.7976931348623158e308 1.7976931348623159e308 1.797693134862315807e308 1.797693134862315708e308
2.4703282292062327e-324 2.4703282292062328e-324 2.4703282292062329e-324 1e-324 1e-400 1e400 1e999999999999999999999 1e-999999999999999999999
0e999999 0e-999999 0.0e0 00.00 1e0 1e00 1e-0 1.e1 -1.e-1 100000000000000000000000000000000e-30 0.00000000000000000000000000000000001e35
nan NaN NAN +nan -nan inf Inf INF +inf -inf infinity Infinity -INFINITY +Infinity""".split()
    for f in fixed:
        out.append(f.encode())
    out.append(b"1" + b"0" * 400 + b"e-400")
    out.append(b"1" + b"0" * 1090)
    out.append(b"1" + b"0" * 1099)
    out.append(b"0" * 1200 + b"1")
    out.append(b"0." + b"0" * 1090 + b"1")
    out.append(b"7" * 1100)
    out.append(b"7" * 1101)
    out.append(b"1." + b"0" * 1098)
    out.append(b"9" * 767 + b"e-300")
    # powers of ten and neighbours
    for e in range(-330, 311):
        for dg in (b"1", b"9", b"5", b"17976931348623157", b"22250738585072014", b"49406564584124654", b"12345678901234567"):
            out.append(dg + b"e%d" % e)
            out.append(b"0." + dg + b"e%d" % e)
    # halfway cases between adjacent doubles, normal, subnormal, near 2^53, near the maximum
    specials = [5e-324, 1e-323, 2.2250738585072009e-308, 2.2250738585072014e-308, 1.0, 2.0 ** 53, 2.0 ** 53 - 1, 2.0 ** 52, 0.1, 1e22, 1e23,
                1.7976931348623157e308, math.nextafter(1.7976931348623157e308, 0), math.ldexp(1.0, -1022), math.ldexp(1.0, -1074 + 52)]
    for x in specials:
        out.extend(halfway(x))
    out.extend(exact_decimal(Fraction(1, 2 ** 1075)).split())
    half = Fraction(1, 2 ** 1075)
    dd = exact_decimal(half)
    mant, ex = dd.split(b"e-")
    out.append(b"%de-%s" % (int(mant) - 1, ex))
    out.append(b"%de-%s" % (int(mant) + 1, ex))
    while len(out) < n:
        r = rnd.random()
        x = random_double(rnd) if r < 0.5 else rnd.choice([rnd.uniform(-1e3, 1e3), rnd.random(), math.ldexp(rnd.random(), rnd.randint(-1074, 1023)),
                                                               rnd.gauss(0, 1) * 10 ** rnd.randint(-30, 30)])
        k = rnd.random()
        if k < 0.25:
            out.append(repr(x).encode())
        elif k < 0.40:
            out.append(("%.17g" % x).encode())
        elif k < 0.50:
            out.append(("%.17e" % x).encode())
        elif k < 0.58:
            out.append(("%.15g" % x).encode())
        elif k < 0.64:
            out.append(("%.16e" % x).encode())
        elif k < 0.70:
            out.extend(halfway(abs(x) if x != 0 else 1.0)[:1] if rnd.random() < 0.5 else halfway(abs(x) if x != 0 else 1.0))
        elif k < 0.80:
            nd = rnd.randint(1, 40)
            digs = "".join(rnd.choice("0123456789") for _ in range(nd))
            e = rnd.randint(-345, 315)
            sign = rnd.choice(["", "-", "+"])
            out.append(("%s%s%s" % (sign, digs, "e%d" % e if rnd.random() < 0.8 else "")).encode())
        elif k < 0.88:
            # an int or a short decimal of up to 12 digits
            nd = rnd.randint(1, 12)
            digs = "".join(rnd.choice("0123456789") for _ in range(nd))
            if rnd.random() < 0.5:
                p = rnd.randint(0, nd)
                digs = digs[:p] + "." + digs[p:] if 0 < p < nd else digs
            out.append((rnd.choice(["", "-", "+"]) + digs).encode())
        elif k < 0.94:
            # a long mantissa with an exponent that brings it into range
            nd = rnd.randint(17, 60)
            digs = "".join(rnd.choice("0123456789") for _ in range(nd))
            out.append(("%s.%se%d" % (digs[0], digs[1:], rnd.randint(-330, 300))).encode())
        else:
            # exponent forms and zeros around the digits
            digs = "".join(rnd.choice("0123456789") for _ in range(rnd.randint(1, 20)))
            out.append((digs + "e" + rnd.choice(["", "+", "-"]) + str(rnd.randint(0, 400)).zfill(rnd.randint(1, 8))).encode())
    return out


def garbage_cells(rnd, n):
    alphabet = b"0123456789+-.eE xX_nNiIfFaAbB\x00\xff\xc3\xa9\t"
    out = []
    for _ in range(n):
        out.append(bytes(rnd.choice(alphabet) for _ in range(rnd.randint(1, 14))))
    out += [b" 1", b"1 ", b"1_000", b"0x10", b"0b1", b"1,5"[:1] + b"\xc2\xa0", b"\xef\xbc\x91", b"\xd9\xa1", b"1\x00", b"--1", b"+-1", b"1e", b"e5", b"1e+", b"1e5.5", b".", b"-", b"+", b".e5",
            b"1.2.3", b"1ee5", b"infinit", b"nan1", b"in", b"n", b"i", b"1f", b"1d5", b"1.5f", b"\xff" * 8, b"\x80" * 7, b"0x1p3", b"1e-", b"1e+-5", b"1 e5", b"1e 5", b"\xe2\x82\xac1"]
    return out


# ------------------------------------------------------------------ running the reader

def run(args, path, timeout=600):
    p = subprocess.run([READER] + args + [str(path)], capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.decode("latin-1"), p.stderr


def parse_dump(out):
    cells, cols, shape = {}, [], {}
    for line in out.splitlines():
        t = line.split(" ", 4)
        if t[0] == "cell":
            cells[(int(t[1]), int(t[2]))] = t[3]
        elif t[0] == "col":
            cols.append((int(t[2]), int(t[3]), t[4] if len(t) > 4 else ""))
        elif t[0] in ("rows", "cols", "missing"):
            shape[t[0]] = int(t[1])
        elif t[0] == "hash":
            shape["hash"] = t[1]
    return cells, cols, shape


def load_info():
    try:
        return os.getloadavg()[0]
    except OSError:
        return -1.0


def write_rows(path, cellrows, header):
    with open(path, "wb") as f:
        f.write(b",".join(header) + b"\n")
        for r in cellrows:
            f.write(b",".join(r) + b"\n")


def gate_diff(n_cells, seed):
    rnd = random.Random(seed)
    t0 = time.time()
    load0 = load_info()
    cells = accepted_cells(rnd, n_cells)
    ok_cells = []
    refused = []
    for c in cells:
        # a cell with a comma, quote or line break would change the file's structure: the csv gate has those
        if b"," in c or b'"' in c or b"\n" in c or b"\r" in c:
            continue
        s = spec(c)
        (ok_cells if s[0] in ("ok", "missing") else refused).append((c, s))
    garbage = [(c, spec(c)) for c in garbage_cells(rnd, 600) if b"," not in c and b'"' not in c and b"\n" not in c and b"\r" not in c and b"\x00" not in c]
    for c, s in garbage:
        (ok_cells if s[0] in ("ok", "missing") else refused).append((c, s))
    differences = []
    checked = 0
    ncols = 8
    with tempfile.TemporaryDirectory() as td:
        # accepted cells: big files, every cell's 64 bits, and the kind of each column
        chunk = 250000
        for start in range(0, len(ok_cells), chunk):
            part = ok_cells[start:start + chunk]
            rows = []
            for i in range(0, len(part), ncols):
                row = [c for c, _ in part[i:i + ncols]]
                row += [b"0"] * (ncols - len(row))
                rows.append(row)
            path = pathlib.Path(td) / ("ok%d.csv" % start)
            write_rows(path, rows, [b"c%d" % j for j in range(ncols)])
            code, out, err = run(["--mode", "dump"], path)
            if code != 0:
                differences.append(("file %d refused: %s" % (start, out[:200]), None, None))
                continue
            got, cols, shape = parse_dump(out)
            kinds = [0] * ncols
            for i, (c, s) in enumerate(part + [(b"0", spec(b"0"))] * (len(rows) * ncols - len(part))):
                r, j = divmod(i, ncols)
                g = got.get((r, j))
                if s[0] == "missing":
                    if g != "M":
                        differences.append((c, "M", g))
                    continue
                kinds[j] = max(kinds[j], s[1])
                want = "%016x" % s[3]
                if g != want:
                    differences.append((c, want, g))
                checked += 1
            for j in range(ncols):
                if cols[j][0] != kinds[j]:
                    differences.append(("kind of column %d" % j, kinds[j], cols[j][0]))
        # refused cells: one file each, the status and the aux the model says
        rcount = 0
        for c, s in refused:
            path = pathlib.Path(td) / "r.csv"
            with open(path, "wb") as f:
                f.write(b"a\n" + c + b"\n")
            code, out, err = run(["--mode", "dump"], path)
            want = "refused %d 2 0 %d" % (s[1], s[2])
            first = out.splitlines()[0] if out else ""
            if code in TRAPS or first != want:
                differences.append((c, want, first or "exit %d" % code))
            rcount += 1
    load1 = load_info()
    print("RD2 diff: %d accepted cells compared bit for bit (%d missing), %d refusals compared (status, aux), seed %d" %
          (checked, sum(1 for _, s in ok_cells if s[0] == "missing"), rcount, seed))
    print("          machine %s, load %.2f at start, %.2f at end, %.1f s" % (os.uname().machine, load0, load1, time.time() - t0))
    for d in differences[:20]:
        print("   DIFFERENCE", d)
    print("RD2 diff: %d differences" % len(differences))
    return len(differences)



M64 = (1 << 64) - 1


def mix(z):
    z &= M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def expected_hash(values, cols):
    """values: row-major list of 64-bit patterns, None for a missing cell (all ones): ml.rng.mix over bits ^ mix(index)."""
    h = 0
    for i, b in enumerate(values):
        b = M64 if b is None else b
        h = (h + mix(b ^ mix(i))) & M64
    return h


def gate_matrix(seed):
    """RD3: a matrix with missing cells, dropped columns and forced floats; types, missing set, values and the hash equal the model's."""
    rnd = random.Random(seed)
    pool = [c for c in accepted_cells(random.Random(seed + 1), 3000) if spec(c)[0] == "ok" and b"," not in c]
    differences = []
    files = 0
    cellsn = 0
    with tempfile.TemporaryDirectory() as td:
        for rows in (1, 2, 31, 32, 33, 62, 63, 64, 65, 95, 96, 97, 127, 128, 129, 200):
            for trial in range(12):
                cols = rnd.randint(1, 5)
                pm = rnd.choice([0.0, 0.1, 0.5, 0.9, 1.0])
                grid = [[(b"" if rnd.random() < pm else rnd.choice(pool)) for _ in range(cols)] for _ in range(rows)]
                # a one-column row with an empty cell would be a blank line, which is not a record: it is written quoted
                shown = [[(b'""' if (c == b"" and cols == 1) else c) for c in row] for row in grid]
                names = [b"n%d" % j for j in range(cols)]
                drop = [j for j in range(cols) if rnd.random() < 0.3]
                forced = [j for j in range(cols) if rnd.random() < 0.3]
                path = pathlib.Path(td) / "m.csv"
                write_rows(path, shown, names)
                args = ["--mode", "dump"]
                if drop:
                    args += ["--drop", ",".join(names[j].decode() for j in drop)]
                if forced:
                    args += ["--float", ",".join(names[j].decode() for j in forced)]
                code, out, err = run(args, path)
                kept = [j for j in range(cols) if j not in drop]
                files += 1
                if not kept:
                    # every column dropped: the matrix has no columns; the reader says so with a refusal tag
                    if not out.startswith("refused-matrix") and "cols 0" not in out:
                        differences.append(("all dropped", out[:80]))
                    continue
                if code != 0:
                    differences.append(("refused", rows, cols, out[:100]))
                    continue
                got, kinds, shape = parse_dump(out)
                if shape.get("rows") != rows or shape.get("cols") != len(kept):
                    differences.append(("shape", rows, cols, shape))
                    continue
                miss_total = 0
                vals = []
                for r in range(rows):
                    for jj, j in enumerate(kept):
                        s = spec(grid[r][j])
                        g = got[(r, jj)]
                        cellsn += 1
                        if s[0] == "missing":
                            miss_total += 1
                            vals.append(None)
                            if g != "M":
                                differences.append(("missing", rows, r, j, g))
                        else:
                            vals.append(s[3])
                            if g != "%016x" % s[3]:
                                differences.append(("value", rows, r, j, g, s))
                for jj, j in enumerate(kept):
                    ks = [spec(grid[r][j])[1] for r in range(rows) if spec(grid[r][j])[0] == "ok"]
                    want = max(ks) if ks else 0
                    if j in forced:
                        want = 3
                    if kinds[jj][0] != want:
                        differences.append(("kind", rows, j, kinds[jj], want))
                if shape.get("missing") != miss_total:
                    differences.append(("missing total", shape.get("missing"), miss_total))
                code, out, err = run(["--mode", "hash"] + args[2:], path)
                got2, k2, sh2 = parse_dump(out)
                if sh2.get("hash") != "%016x" % expected_hash(vals, len(kept)):
                    differences.append(("hash", rows, cols, sh2.get("hash")))
    print("RD3 matrix: %d files, %d cells compared (values, missing bits, kinds, hash)" % (files, cellsn))
    for d in differences[:20]:
        print("   DIFFERENCE", d)
    print("RD3 matrix: %d differences" % len(differences))
    return len(differences)


def peak_rss_mb(args):
    """The peak resident size of one run of the reader, in MB, measured in a child that runs nothing else."""
    code = ("import resource, subprocess, sys; p = subprocess.run(sys.argv[1:], capture_output=True); "
            "r = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss; print(p.returncode, r, p.stdout.decode('latin-1').splitlines()[:1])")
    out = subprocess.run([sys.executable, "-c", code] + [READER] + args, capture_output=True).stdout.decode().strip()
    rc, rss, first = out.split(" ", 2)
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return int(rc), int(rss) * scale / 1e6, first


def gate_limits():
    """RD4: each cap refuses the file one over it, with its status, and nothing sized by the file is built before the refusal."""
    differences = []
    checks = 0

    def expect(path, args, want, label):
        nonlocal checks
        checks += 1
        code, out, err = run(args, path)
        first = out.splitlines()[0] if out else "exit %d" % code
        if code in TRAPS or not first.startswith(want):
            differences.append((label, want, first))

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        p = td / "f.csv"
        for n in (1, 2, 10, 1000):
            write_rows(p, [[b"1.5", b"2"]] * (n + 1), [b"a", b"b"])
            expect(p, ["--mode", "dump", "--max-rows", str(n)], "refused 7 ", "rows over by one (%d)" % n)
            expect(p, ["--mode", "dump", "--max-rows", str(n + 1)], "rows %d" % (n + 1), "rows at the cap (%d)" % n)
            expect(p, ["--mode", "shape", "--max-rows", str(n)], "refused 7 ", "rows over by one, shape (%d)" % n)
        for k in (1, 2, 5, 4096):
            write_rows(p, [[b"1"] * (k + 1)], [b"c%d" % j for j in range(k + 1)])
            expect(p, ["--mode", "dump", "--max-columns", str(k)], "refused 6 ", "columns over by one (%d)" % k)
            expect(p, ["--mode", "dump", "--max-columns", str(k + 1)], "rows 1", "columns at the cap (%d)" % k)
        write_rows(p, [[b"1"] * 200000], [b"c%d" % j for j in range(200000)])
        expect(p, ["--mode", "dump", "--max-line-bytes", "16777216"], "refused 6 1 0 200000", "200,000 columns against the default cap")
        for r, c, m in ((10, 3, 29), (10, 3, 30), (1, 1, 1), (7, 7, 48), (7, 7, 49), (1000, 2, 1999), (1000, 2, 2000)):
            write_rows(p, [[b"1"] * c] * r, [b"c%d" % j for j in range(c)])
            if r * c > m:
                expect(p, ["--mode", "dump", "--max-cells", str(m)], "refused 13 ", "cells %dx%d over %d" % (r, c, m))
            else:
                expect(p, ["--mode", "dump", "--max-cells", str(m)], "rows %d" % r, "cells %dx%d at %d" % (r, c, m))
        write_rows(p, [[b"1"]], [b"a"])
        expect(p, ["--mode", "dump", "--max-cells", "999999999999999999"], "refused-matrix ml.param.out-of-range", "a cell cap over the ceiling")
        # a line, and a record that spans lines
        with open(p, "wb") as f:
            f.write(b"a,b\n" + b"1," + b"2" * 99 + b"\n")
        expect(p, ["--mode", "dump", "--max-line-bytes", "100"], "refused 2 2 0 101", "a line of 101 bytes, cap 100")
        with open(p, "wb") as f:
            f.write(b"a,b\n" + b"1," + b"2" * 98 + b"\n")
        expect(p, ["--mode", "dump", "--max-line-bytes", "100"], "rows 1", "a line of 100 bytes, cap 100")
        with open(p, "wb") as f:
            f.write(b'a,b\n1,"' + b"x" * 40 + b'\n' + b"y" * 40 + b'\n' + b"z" * 40 + b'"\n')
        expect(p, ["--mode", "dump", "--drop", "b", "--max-line-bytes", "100"], "refused 2 2 0 ", "a record of 3 lines over the cap")
        expect(p, ["--mode", "dump", "--drop", "b", "--max-line-bytes", "200"], "rows 1", "the same record under the cap")
        # memory: a file of 200 MB refused at its eleventh row does not hold it
        big = td / "big.csv"
        row = b"1.5,2.5,3.5,4.5\n"
        with open(big, "wb") as f:
            f.write(b"a,b,c,d\n")
            for _ in range(25):
                f.write(row * 500000)
        size = os.path.getsize(big)
        rc, rss_refused, first = peak_rss_mb(["--mode", "dump", "--max-rows", "10", str(big)])
        if rc != 8 or not first.startswith("[") or "refused 7" not in first:
            differences.append(("big file", rc, first))
        checks += 1
        if rss_refused > 16:
            differences.append(("peak memory of the refusal", rss_refused))
        rc, rss_cells, first = peak_rss_mb(["--mode", "dump", "--max-cells", "40", str(big)])
        checks += 1
        if rc != 8 or "refused 13" not in first or rss_cells > 16:
            differences.append(("big file, cell cap", rc, rss_cells, first))
        rc, rss_shape, first = peak_rss_mb(["--mode", "shape", str(big)])
        rc, rss_hash, first = peak_rss_mb(["--mode", "hash", "--max-cells", "400000000", str(big)])
        print("          context: the same file loaded whole into the matrix (hash mode, caps raised): peak %.0f MB, so a refusal that came after the allocation would show" % rss_hash)
        print("          context: the same %d MB file read whole (shape, pass 1 only): peak %.1f MB; refused at row 11: peak %.1f MB; at the cell cap: %.1f MB" %
              (size // 1000000, rss_shape, rss_refused, rss_cells))
    print("RD4 limits: %d checks" % checks)
    for d in differences[:20]:
        print("   DIFFERENCE", d)
    print("RD4 limits: %d differences" % len(differences))
    return len(differences)



def text_field(rnd):
    parts = ["", "abc", "a,b", 'say "hi"', "line\nbreak", "cr\r\nlf", ",", '"', '""', "tail ", " lead", "\u00e9\u4e2d", "1.5", "nan", "e5", "\n", "\n\n", "a\n,b\n\"c\""]
    return rnd.choice(parts) + rnd.choice(["", "x", "7"])


def csv_text(rows, quote_all, eol):
    buf = io.StringIO(newline="")
    w = pycsv.writer(buf, quoting=pycsv.QUOTE_ALL if quote_all else pycsv.QUOTE_MINIMAL, lineterminator=eol)
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


def gate_csv(seed):
    """RD5: quoting, CRLF, BOM, blank lines, records that span lines, at every alignment against the 64 KiB read of toolbox.lines."""
    rnd = random.Random(seed)
    pool = [c.decode() for c in accepted_cells(random.Random(seed + 7), 2000) if spec(c)[0] == "ok" and b"," not in c and b'"' not in c and b"\n" not in c and b"\r" not in c]
    differences = []
    files = 0
    cellsn = 0
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "q.csv"
        for trial in range(260):
            nrows = rnd.randint(1, 40)
            header = ["x0", "t1", "x2", "t3", "x4"]
            if rnd.random() < 0.3:
                header[1] = "t,1"
                header[3] = 't"3'
            rows = []
            for _ in range(nrows):
                rows.append([rnd.choice(pool), text_field(rnd), rnd.choice(pool), text_field(rnd), rnd.choice(pool)])
            eol = rnd.choice(["\n", "\r\n"])
            data = csv_text([header] + rows, rnd.random() < 0.4, eol)
            # the writer is checked first: Python's own reader gives the records back
            back = list(pycsv.reader(io.StringIO(data.decode("utf-8"), newline="")))
            if back != [header] + rows:
                differences.append(("generator", trial))
                continue
            if rnd.random() < 0.3:
                data = b"\xef\xbb\xbf" + data
            if rnd.random() < 0.3:
                # blank lines between records (not inside a quote): only after a record that ends a line, so rebuild with them
                lines_ = []
                for r in [header] + rows:
                    lines_.append(csv_text([r], False, eol))
                    if rnd.random() < 0.3:
                        lines_.append(eol.encode())
                data = (b"\xef\xbb\xbf" if data.startswith(b"\xef\xbb\xbf") else b"") + b"".join(lines_)
            # the alignment: a record of padding before the data moves the interesting bytes across the chunk edge of 65536
            if trial % 2 == 0:
                edge = rnd.choice([65536, 131072, 196608])
                pad_to = edge - rnd.randint(0, 60)
                body = data
                head_end = body.index(b"\n") + 1 if not body.startswith(b"\xef") else body.index(b"\n") + 1
                fill = max(0, pad_to - head_end - 12)
                padrow = ("9," + "p" * fill + ",9,q,9").encode() + (b"\r\n" if eol == "\r\n" else b"\n")
                data = body[:head_end] + padrow + body[head_end:]
                rows = [[ "9", "p" * fill, "9", "q", "9" ]] + rows
            with open(path, "wb") as f:
                f.write(data)
            code, out, err = run(["--mode", "dump", "--drop", header[1] if "," not in header[1] else header[1], "--max-line-bytes", "16777216"], path) if False else (None, None, None)
            args = ["--mode", "dump", "--max-line-bytes", "16777216", "--drop", ",".join(header[1:2] + header[3:4]).replace(",", ",")]
            # names that hold a comma cannot be written in a --drop list; drop by position is not offered: use the plain names
            if "," in header[1] or '"' in header[3]:
                args = ["--mode", "dump", "--max-line-bytes", "16777216"]
                code, out, err = run(args, path)
                # the text columns are not numbers: the reader refuses at the first row's text, which is the point (no trap, a clean refusal)
                files += 1
                if code in TRAPS or not (out.startswith("refused 8") or out.startswith("rows")):
                    differences.append(("refusal of a text column", trial, out[:80]))
                continue
            code, out, err = run(args, path)
            files += 1
            if code != 0:
                differences.append(("refused", trial, out[:100]))
                continue
            got, kinds, shape = parse_dump(out)
            if shape.get("rows") != len(rows) or shape.get("cols") != 3:
                differences.append(("shape", trial, shape, len(rows)))
                continue
            if [c[2] for c in kinds] != [header[0], header[2], header[4]]:
                differences.append(("names", trial, [c[2] for c in kinds]))
            for r, row in enumerate(rows):
                for jj, j in enumerate((0, 2, 4)):
                    s_ = spec(row[j].encode())
                    cellsn += 1
                    if got[(r, jj)] != "%016x" % s_[3]:
                        differences.append(("value", trial, r, j, got[(r, jj)], s_))
        # the refusals and their lines
        D = ["--drop", "b"]
        cases = [
            (b'a,b\n1,"x"y\n', "refused 17 2 0 0", "a quote closed and followed by a letter", D),
            (b'a,b\n1,2\n3,"unterminated\n4,5\n', "refused 18 3 0 0", "a quote open at the end of the file", []),
            (b'a,b\n1,2\n3,"x\ny"\n7\n', "refused 3 5 0 0", "a ragged row after a record that spans lines", D),
            (b'a,b\n1,"x\ny"\nz,6\n', "refused 8 4 0 2", "a bad number after a record that spans lines", D),
            (b'a,b\n1,"x\ny"\n2,"p\nq\nr"x\n', "refused 17 6 0 0", "a bad quote on the last line of a record that spans lines", D),
            (b'a,b\n1,2\n3,"x\ny"\n"q\nr",5\n6,7\n', "refused 8 5 0 2", "the line of a bad cell is where its record starts", D),
            (b'a,b\n"1","2"\n', "rows 1", "quoted numbers", []),
            (b'a,b\n"",2\n', "rows 1", "a quoted empty cell is missing", []),
            (b'a,b\n"1""",2\n', "refused 8 2 0 2", "a doubled quote in a number", []),
            (b'\xef\xbb\xbfa,b\r\n1,2\r\n\r\n3,4\r\n', "rows 2", "BOM, CRLF and a blank line", []),
            (b'"a,b",c\n1,2\n', "rows 1", "a quoted header name with a comma", []),
            (b'a,a\n1,2\n', "refused 5 1 1 2", "a duplicate name", []),
            (b',a\n1,2\n', "refused 5 1 0 2", "an empty name", []),
            (b'', "refused 1 0 0 0", "an empty file", []),
            (b'\n\n\n', "refused 1 0 0 0", "only blank lines", []),
            (b'a,b\n', "refused-matrix ml.input.empty", "a header and no rows", []),
            (b'a\n"', "refused 18 2 0 0", "a lone quote", []),
        ]
        for data, want, label, extra in cases:
            with open(path, "wb") as f:
                f.write(data)
            code, out, err = run(["--mode", "dump"] + extra, path)
            files += 1
            first = out.splitlines()[0] if out else "exit %d" % code
            if code in TRAPS or not first.startswith(want):
                differences.append((label, want, first))
    print("RD5 csv: %d files, %d numeric cells compared with Python's csv module and the cell model, %d refusal cases" % (files, cellsn, len(cases)))
    for d in differences[:20]:
        print("   DIFFERENCE", d)
    print("RD5 csv: %d differences" % len(differences))
    return len(differences)



def valid_output(code, out, mode="dump"):
    if code == 0:
        return "rows " in out and ("missing " in out or mode == "shape")
    if code == 8:
        return out.startswith("refused ") or out.startswith("refused-matrix ")
    if code == 3:
        return out.startswith("refused-open")
    return False


def mutate(rnd, base: bytes) -> bytes:
    b = bytearray(base)
    k = rnd.randint(1, 4)
    for _ in range(k):
        t = rnd.randint(0, 6)
        if not b:
            b = bytearray(rnd.randbytes(rnd.randint(1, 20)))
        i = rnd.randrange(len(b))
        if t == 0:
            b[i] = rnd.randrange(256)
        elif t == 1:
            del b[i:i + rnd.randint(1, 8)]
        elif t == 2:
            b[i:i] = rnd.choice([b'"', b",", b"\n", b"\r", b"e", b"-", b".", b"\x00", b"\xff", b'""', b"1e999999", b"0x", b"nan"])
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


def fuzz_inputs(rnd, n):
    valid = []
    for _ in range(40):
        rows = [[rnd.choice(["1", "2.5", "-3e4", "", "0.1", "nan", "abc", '"7"', '"a,b"', "1e400", "123456789012345678901"]) for _ in range(3)] for _ in range(rnd.randint(1, 6))]
        valid.append(b"a,b,c" + rnd.choice([b"\n", b"\r\n"]) + b"".join(b",".join(c.encode() for c in r) + b"\n" for r in rows))
    for i in range(n):
        t = rnd.random()
        if t < 0.15:
            yield "random bytes", rnd.randbytes(rnd.randint(0, 3000))
        elif t < 0.55:
            yield "mutated", mutate(rnd, rnd.choice(valid))
        elif t < 0.65:
            ln = rnd.randint(0, 1300)
            yield "cell of %d bytes" % ln, b"a\n" + bytes(rnd.choice(b"0123456789.eE+-") for _ in range(ln)) + b"\n"
        elif t < 0.80:
            yield "quotes", b"a,b\n" + bytes(rnd.choice(b'",\n\r1 ') for _ in range(rnd.randint(1, 200)))
        elif t < 0.90:
            digs = bytes(rnd.choice(b"0123456789") for _ in range(rnd.randint(1, 60)))
            yield "exponent", b"a\n" + digs + b"e" + rnd.choice([b"", b"-", b"+"]) + bytes(rnd.choice(b"0123456789") for _ in range(rnd.randint(1, 30))) + b"\n"
        else:
            yield "truncated", rnd.choice(valid)[:rnd.randint(0, 60)]


def gate_fuzz(runs, seed, ml=None):
    """RD6: nothing a file holds reaches a trap; each run ends 0, 8 or 3 and says so in the form the driver promises."""
    rnd = random.Random(seed)
    bad = []
    n = 0
    t0 = time.time()
    load0 = load_info()
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "f.csv"
        for label, data in fuzz_inputs(rnd, runs):
            with open(path, "wb") as f:
                f.write(data)
            for mode in ("dump",) if n % 4 else ("dump", "shape"):
                try:
                    code, out, err = run(["--mode", mode, "--max-line-bytes", "16777216"], path, timeout=60)
                except subprocess.TimeoutExpired:
                    bad.append((label, "timeout"))
                    continue
                if code in TRAPS or code < 0 or not valid_output(code, out, mode):
                    bad.append((label, code, out[:60], data[:60]))
            n += 1
        # the large ones
        big = [
            ("100,000-digit cell", b"a\n" + b"7" * 100000 + b"\n"),
            ("100,000-digit mantissa with 10^6 exponent", b"a\n" + b"1." + b"3" * 99990 + b"e1000000\n"),
            ("exponent of 100,000 digits", b"a\n1e" + b"9" * 100000 + b"\n"),
            ("negative exponent of 100,000 digits", b"a\n1e-" + b"9" * 100000 + b"\n"),
            ("5 MB line of digits", b"a\n" + b"1" * 5000000 + b"\n"),
            ("5 MB header", b"a," * 2500000 + b"b\n1\n"),
            ("5 MB of quotes", b"a\n" + b'"' * 5000000 + b"\n"),
            ("5 MB of commas", b"a\n" + b"," * 5000000 + b"\n"),
            ("a million blank lines", b"a\n" + b"\n" * 1000000),
            ("a million CRs", b"a\n" + b"\r" * 1000000),
            ("NULs", b"a\n\x00\x00\x00\n\x00"),
            ("only a BOM", b"\xef\xbb\xbf"),
            ("a BOM then a quote", b'\xef\xbb\xbf"'),
            ("deep spanning record", b'a\n"' + b"\n" * 200000 + b'"\n'),
        ]
        for label, data in big:
            with open(path, "wb") as f:
                f.write(data)
            for extra in ([], ["--max-line-bytes", "16777216"]):
                try:
                    code, out, err = run(["--mode", "dump"] + extra, path, timeout=120)
                except subprocess.TimeoutExpired:
                    bad.append((label, "timeout"))
                    continue
                if code in TRAPS or code < 0 or not valid_output(code, out):
                    bad.append((label, code, out[:60]))
            n += 1
    print("RD6 fuzz: %d inputs through reader-check (%s), machine %s, load %.2f -> %.2f, %.0f s" % (n, "dump and shape", os.uname().machine, load0, load_info(), time.time() - t0))
    for d in bad[:20]:
        print("   TRAP OR BAD OUTPUT", d)
    print("RD6 fuzz: %d traps or malformed answers" % len(bad))
    return len(bad)


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("diff", "matrix", "limits", "csv", "fuzz", "all"):
        print(__doc__)
        return 2
    what = argv[0]

    def opt(name, default):
        return int(argv[argv.index(name) + 1]) if name in argv else default

    bad = 0
    if what in ("diff", "all"):
        bad += gate_diff(opt("--cells", 2000000), opt("--seed", 1))
    if what in ("matrix", "all"):
        bad += gate_matrix(opt("--seed", 1))
    if what in ("limits", "all"):
        bad += gate_limits()
    if what in ("csv", "all"):
        bad += gate_csv(opt("--seed", 1))
    if what in ("fuzz", "all"):
        bad += gate_fuzz(opt("--runs", 5000), opt("--seed", 1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

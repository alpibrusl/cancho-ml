"""S3 (docs/serving-http.md sections 4 and 9): JSON numbers become CSV cells as Python's exact arithmetic says they must.

`build/serve-check` (tools/serve-check, a driver over tools/ml-serve/num.cho) is given lines `TYPE SCALE TEXT` and answers the cell the server would write, or the refusal. Python decides
what the answer has to be from the text alone:

* int (TYPE 0): the exact value (`decimal.Decimal`) must be a whole number below 10**12 in magnitude; the cell is its digits (`-0` is `0`); anything else is refused (1);
* dec (TYPE 1), scale s: the exact value times 10**s must be a whole number below 10**12; the cell is the value with exactly s digits after the point; anything else is refused (1);
* float (TYPE 2): the nearest double (`float(text)`, correctly rounded, ties to even) must be finite, and non-zero text must not read as zero, and the text must be at most 1,100 bytes; the cell is
  a shortest decimal that reads back to that double (as many digits as `repr`'s), with any zero written `0.0`; anything else is refused (2).

Generated and adversarial numbers: 17-digit and longer mantissas, subnormals, halfway cases between neighbouring doubles (exact midpoints and one digit either side), exponents, `-0`, `1e999`,
the edges of 12 digits, texts longer than 1,100 bytes. Then the same cases over HTTP, on a model with an int, a dec and a float column: the answer of the server is the answer of `ml predict` on a CSV
that Python wrote from the same rules.
"""

import decimal
import json
import math
import os
import random
import re
import struct
import subprocess
import unittest
from decimal import Decimal

from harness import ROOT
from serve_lib import Server, corpus, loads_tokens

CHECK = os.environ.get("SERVE_CHECK_BIN") or str(ROOT / "build" / "serve-check")
CTX = decimal.Context(prec=100000, Emin=decimal.MIN_EMIN, Emax=decimal.MAX_EMAX, traps=[decimal.InvalidOperation, decimal.DivisionByZero])
JSON_NUMBER = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")
LIMIT = 10**12


def expect_exact(text, scale):
    d = CTX.create_decimal(text)
    scaled = d.scaleb(scale, context=CTX)
    if scaled != scaled.to_integral_value() or scaled.copy_abs() >= LIMIT:
        return ("refused", 1)
    n = int(scaled)
    if scale == 0:
        return ("ok", str(n))
    sign = "-" if n < 0 else ""
    digits = str(abs(n)).rjust(scale + 1, "0")
    return ("ok", sign + digits[:-scale] + "." + digits[-scale:])


def expect_float(text):
    if len(text.encode()) > 1100:
        return ("refused", 2)
    x = float(text)
    if math.isinf(x):
        return ("refused", 2)
    if x == 0.0:
        if CTX.create_decimal(text) != 0:
            return ("refused", 2)
        return ("ok", "0.0")
    return ("ok", repr(x))


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def from_bits(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def midpoint_texts(x):
    """Decimal texts at the exact midpoint between x and the next double, and a digit either side of it: the halfway cases of correct rounding."""
    y = from_bits(bits(x) + 1)
    if math.isinf(y) or math.isnan(y):
        return []
    ctx = decimal.Context(prec=1200)
    m = ctx.divide(ctx.add(Decimal(x), Decimal(y)), 2)
    out = [str(m)]
    digits = m.as_tuple().digits
    for d in (-1, 1):
        n = ctx.add(m, Decimal((0, (1,), m.as_tuple().exponent - 5 if len(digits) < 1100 else m.as_tuple().exponent)) * d)
        out.append(str(n))
    return out


def numbers(rng):
    cases = []
    zeros = ["0", "-0", "0.0", "-0.0", "0e0", "0e999999", "-0e-999999", "0.000e5"]
    cases += zeros
    # integers and decimals
    for _ in range(4000):
        digits = rng.randint(1, 30)
        s = "".join(rng.choice("0123456789") for _ in range(digits)).lstrip("0") or "0"
        if rng.random() < 0.3:
            s += "." + "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 12)))
        if rng.random() < 0.2:
            s += rng.choice("eE") + rng.choice(["", "+", "-"]) + str(rng.randint(0, 40))
        if rng.random() < 0.4:
            s = "-" + s
        cases.append(s)
    for k in range(1, 25):
        for base in (10**k, 10**k - 1, 10**k + 1):
            cases += [str(base), str(-base), "%s.0" % base, "%de-%d" % (base, 3), "%s.5" % base, "%s.50" % base]
    cases += ["999999999999", "-999999999999", "1000000000000", "99999999999.99", "999999999999.99", "0.000000000005", "0.0000000000005", "123456789012", "12345678901.2", "1e11", "1e12",
              "1.5e2", "150.0", "15E1", "1500e-1", "0.0150e4", "3.125", "3.1250", "0.05", "5e-2", "500e-4", "1e-3", "1e2", "100e-2", "100000000000e-11", "1" + "0" * 11, "1" + "0" * 12]
    # floats: random doubles by their bits, as 17 digits and as repr
    for _ in range(4000):
        x = from_bits(rng.getrandbits(64))
        if math.isnan(x) or math.isinf(x):
            continue
        cases.append(repr(x))
        cases.append("%.17g" % x)
        cases.append("%.16e" % x)
        cases.append("%.20e" % x)
    for _ in range(1500):
        e = rng.randint(-330, 310)
        cases.append("%d.%de%d" % (rng.randint(1, 9), rng.getrandbits(60), e))
        cases.append("%de%d" % (rng.getrandbits(rng.randint(1, 80)), e))
    # subnormals, the extremes, the powers of two and of ten
    for b in list(range(1, 40)) + [0x000FFFFFFFFFFFFF, 0x0010000000000000, 0x0010000000000001, 0x7FEFFFFFFFFFFFFF, 0x7FEFFFFFFFFFFFFE, 0x3FF0000000000000, 0x3FF0000000000001, 0x433FFFFFFFFFFFFF, 0x4340000000000000]:
        x = from_bits(b)
        cases += [repr(x), "%.17g" % x, "%.30e" % x, repr(-x)]
        cases += midpoint_texts(x)
    cases += ["5e-324", "4.9406564584124654e-324", "2.4703282292062327e-324", "2.4703282292062328e-324", "2.4703282292062327208e-324", "2.2250738585072014e-308", "2.2250738585072011e-308",
              "1.7976931348623157e308", "1.7976931348623158e308", "1.7976931348623159e308", "1.797693134862315807e308", "1e308", "1e309", "1e999", "-1e999", "1e-999", "-1e-999", "1e400", "1e-400",
              "9007199254740992", "9007199254740993", "9007199254740994", "9007199254740993.0", "0.1", "0.2", "0.30000000000000004", "1e22", "1e23", "1e21", "1e-7", "1.5e-7", "123456789e-30",
              "1" + "0" * 400, "1." + "0" * 400 + "1", "0." + "0" * 400 + "1", "1e-" + "0" * 50 + "5", "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899",
              "8.9884656743115795e307", "4.35e-322", "-0.0000000000000000000000000000001", "1e100000", "1e-100000", "0e100000", "1e99999999999", "1e-99999999999"]
    # long texts around the limit of 1,100 bytes
    for n in (1090, 1099, 1100, 1101, 1200, 5000):
        cases.append("1." + "0" * (n - 2))
        cases.append("1." + "0" * (n - 3) + "1")
        cases.append("0." + "1" * (n - 2))
    return [c for c in dict.fromkeys(cases) if JSON_NUMBER.fullmatch(c)]


def run_check(lines):
    p = subprocess.run([CHECK], input=("\n".join(lines) + "\n").encode(), capture_output=True, timeout=300)
    assert p.returncode == 0, p
    out = p.stdout.decode().split("\n")
    assert out[-1] == ""
    return out[:-1]


class Cells(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = numbers(random.Random(20260929))

    def test_the_driver_answers_every_line(self):
        out = run_check(["0 0 1", "1 2 1.5", "2 0 1.5"])
        self.assertEqual(out, ["ok 1", "ok 1.50", "ok 1.5"])

    def test_int_cells(self):
        lines = ["0 0 " + c for c in self.cases]
        out = run_check(lines)
        self.assertEqual(len(out), len(lines))
        ok = bad = 0
        for c, got in zip(self.cases, out):
            kind, val = expect_exact(c, 0)
            want = "ok " + val if kind == "ok" else "refused 1"
            self.assertEqual(got, want, c[:80])
            ok += kind == "ok"
            bad += kind != "ok"
        self.assertGreater(ok, 100)
        self.assertGreater(bad, 100)
        print("serve json int: %d numbers, %d rendered, %d refused, all as Decimal says" % (len(out), ok, bad))

    def test_dec_cells(self):
        for scale in (1, 2, 3, 6, 11, 12):
            lines = ["1 %d %s" % (scale, c) for c in self.cases]
            out = run_check(lines)
            ok = 0
            for c, got in zip(self.cases, out):
                kind, val = expect_exact(c, scale)
                want = "ok " + val if kind == "ok" else "refused 1"
                self.assertEqual(got, want, (scale, c[:80]))
                ok += kind == "ok"
            self.assertGreater(ok, 20, scale)
        print("serve json dec: %d numbers x 6 scales, all as Decimal says" % len(self.cases))

    def test_float_cells(self):
        lines = ["2 0 " + c for c in self.cases]
        out = run_check(lines)
        ok = bad = 0
        for c, got in zip(self.cases, out):
            kind, val = expect_float(c)
            if kind == "refused":
                self.assertEqual(got, "refused 2", c[:80])
                bad += 1
                continue
            self.assertTrue(got.startswith("ok "), (c[:80], got))
            cell = got[3:]
            # the same double, written as the shortest decimal that reads back to it
            self.assertEqual(float(cell), float(c), (c[:80], cell))
            # shortest: as many significant digits as Python's repr. When two shortest decimals are equally near the double, cancho-table's printer and Python's repr may pick
            # different ones (-902899553826474.3 and -902899553826474.2 are the same double): the digits differ, the length does not
            self.assertEqual(len(CTX.create_decimal(cell).normalize().as_tuple().digits), len(CTX.create_decimal(val).normalize().as_tuple().digits), (c[:80], cell, val))
            if float(c) != 0:
                self.assertEqual(float(cell).hex(), float(c).hex())
            ok += 1
        self.assertGreater(ok, 5000)
        self.assertGreater(bad, 20)
        print("serve json float: %d numbers, %d rendered (same double, shortest decimal), %d refused" % (len(out), ok, bad))

    def test_the_grammar_of_an_answer_number(self):
        texts = ["0", "-0", "1.5", "1e21", "1.5e-7", "100.0", "-3", "0.450000", "01", "1.", ".5", "+1", "1e", "1e+", "--1", "1.5.5", "0x10", "nan", "inf", "", "-", "1e5", "1E5", "1E+5", "0e0", "00", "-01"]
        lines = ["3 0 " + t for t in texts if t != ""]
        out = run_check(lines)
        for t, got in zip([t for t in texts if t != ""], out):
            self.assertEqual(got, "json" if JSON_NUMBER.fullmatch(t) else "notjson", t)


NUM = "num.csv"


class Http(unittest.TestCase):
    """The same rules over HTTP: the linear model has `a` int, `b` dec(2) and `c` float."""

    @classmethod
    def setUpClass(cls):
        cls.sb, cls.models, _ = corpus()
        cls.srv = Server(cls.sb.dir, "linear.cml").start()
        cls.conn = cls.srv.conn()

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()
        cls.sb.cleanup()

    def cli(self, a, b, c):
        self.sb.write("one.csv", "a,b,c\n%s,%s,%s\n" % (a, b, c))
        r = self.sb.run("predict", "--model-file", "linear.cml", "one.csv")
        assert r.status == 0, r
        return r.data()["rows"]

    def predict(self, a, b, c):
        body = ('{"a":%s,"b":%s,"c":%s}' % (a, b, c)).encode()
        return self.conn.request("POST", "/predict", body=body, headers={"Content-Type": "application/json"})

    def test_the_server_renders_what_python_says(self):
        rng = random.Random(7)
        pool = numbers(rng)
        rng.shuffle(pool)
        pool = pool[:1500]
        accepted = refused = nonfinite = 0
        for i, text in enumerate(pool):
            # the number goes to each of the three columns in turn
            for col in "abc":
                args = {"a": "3", "b": "2.50", "c": "0.5"}
                args[col] = text
                r = self.predict(**args)
                if col == "a":
                    kind, val = expect_exact(text, 0)
                    # the schema judges an int first: a number that is not a whole number below 10**12 is its 422 whatever the renderer would say; the integer 10**12 and up too
                    is_int_text = kind == "ok"
                elif col == "b":
                    kind, val = expect_exact(text, 2)
                else:
                    kind, val = expect_float(text)
                if kind == "ok":
                    if r.status == 422 and r.json()["rule"] == "serve.non-finite-prediction":
                        # the model's answer is infinite: the command line prints it as text that is not a number, the server refuses
                        nonfinite += 1
                        continue
                    self.assertEqual(r.status, 200, (col, text[:60], r))
                    cell = val if col != "c" else repr(float(text))
                    cells = dict(args)
                    cells[col] = cell
                    want = self.cli(**cells)
                    self.assertEqual(loads_tokens(r.text())["rows"], want, (col, text[:60]))
                    accepted += 1
                else:
                    self.assertEqual(r.status, 422, (col, text[:60], r))
                    self.assertIn(r.json()["rule"], ("serve.number-not-exact", "ml.schema.mismatch"))
                    refused += 1
        print("serve json http: %d requests, %d answered as the command line answers on Python's rendering, %d refused with 422, %d refused because the model's answer is not finite" % (accepted + refused + nonfinite, accepted, refused, nonfinite))
        self.assertGreater(accepted, 1000)
        self.assertGreater(refused, 500)


if __name__ == "__main__":
    unittest.main()

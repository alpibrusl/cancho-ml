"""RD7 -- the constant model on float columns is real and right (docs/reader.md sections 4 and 5).

`fit`, `predict` and `eval` with a float target and float features are compared, bit for bit, with an oracle written here with exact
rational arithmetic (fractions.Fraction): the mean is the exact sum over n, rounded once (`float(Fraction)` is correctly rounded);
`sse` is the exact sum of (y - P)^2 rounded once; `rmse` is math.sqrt(sse / n) (two IEEE operations, both correctly rounded in Python
too). The data are seeded and include sums that leave 64 bits, values that cancel (1e308, -1e308), subnormals, and every row order.

The numbers in the answers are strings holding the shortest decimal that reads back to the double; they are compared as doubles
(float(text) == the oracle), which is also what "shortest round-trip" promises."""

import math
import random
import unittest
from fractions import Fraction

from harness import Sandbox, without_provenance


def finite(rnd):
    while True:
        x = __import__("struct").unpack(">d", rnd.getrandbits(64).to_bytes(8, "big"))[0]
        if math.isfinite(x):
            return x


def value_pool(rnd, kind):
    if kind == 0:
        return [rnd.uniform(-1e3, 1e3) for _ in range(rnd.randint(1, 30))]
    if kind == 1:
        return [finite(rnd) for _ in range(rnd.randint(1, 30))]
    if kind == 2:
        # cancellation: huge values that cancel, small values that survive
        base = [1e308, -1e308, 1.7976931348623157e308, -1.7976931348623157e308, 3.0, 0.1, 5e-324, -5e-324, 1e-320]
        return [rnd.choice(base) for _ in range(rnd.randint(2, 30))]
    if kind == 3:
        return [math.ldexp(rnd.random(), rnd.randint(-1074, -1000)) for _ in range(rnd.randint(1, 30))]
    if kind == 4:
        return [float(rnd.randint(-10 ** 6, 10 ** 6)) / 10 ** rnd.randint(0, 6) for _ in range(rnd.randint(1, 30))]
    return [rnd.choice([0.1, 0.2, 0.3, 1e16, 1.0, -1e16, 2.0 ** 53, 1.0 + 2.0 ** -52]) for _ in range(rnd.randint(1, 30))]


def csv_cell(rnd, x):
    k = rnd.random()
    if k < 0.4:
        return repr(x)
    if k < 0.6:
        return "%.17g" % x
    if k < 0.8:
        return "%.17e" % x
    return "%.20g" % x


def oracle_mean(ys):
    return float(sum(Fraction(y) for y in ys) / len(ys))


def oracle_sse(ys, p):
    fp = Fraction(p)
    return sum((Fraction(y) - fp) ** 2 for y in ys)


class Floats(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()

    def tearDown(self):
        self.sb.cleanup()

    def table(self, ys, rnd):
        rows = ["id,f,y"]
        for i, y in enumerate(ys):
            rows.append("%d,%s,%s" % (i, csv_cell(rnd, rnd.uniform(-5, 5)), csv_cell(rnd, y)))
        return "\n".join(rows) + "\n"

    def check(self, ys, rnd, label):
        # the cells are what the CSV holds, so the oracle reads them the way Python does
        text = self.table(ys, rnd)
        cells = [float(line.split(",")[2]) for line in text.splitlines()[1:]]
        self.sb.write("d.csv", text)
        r = self.sb.run("fit", "--overwrite", "--target", "y", "--output", "m.cml", "d.csv")
        self.assertEqual(r.status, 0, (label, r))
        mean = oracle_mean(cells)
        self.assertEqual(float(r.data()["summary"]["mean"]), mean, (label, r.data()["summary"], mean, ys))
        r = self.sb.run("predict", "--model-file", "m.cml", "d.csv")
        self.assertEqual({float(row[0]) for row in r.data()["rows"]}, {mean}, label)
        r = self.sb.run("eval", "--model-file", "m.cml", "d.csv")
        sse = oracle_sse(cells, mean)
        try:
            want_sse = float(sse)
        except OverflowError:
            self.assertEqual(r.first_rule(), "ml.metric.undefined", (label, ys))
            self.assertEqual(r.status, 8)
            return
        m = r.data()["metrics"]
        self.assertEqual(float(m["sse"]), want_sse, (label, m, ys))
        self.assertEqual(float(m["rmse"]), math.sqrt(want_sse / len(cells)), (label, m, ys))
        self.assertEqual(float(m["prediction"]), mean)

    def test_the_mean_sse_and_rmse_are_the_exact_ones_rounded_once(self):
        rnd = random.Random(20261012)
        for trial in range(240):
            self.check(value_pool(rnd, trial % 6), rnd, "trial %d" % trial)

    def test_the_answer_does_not_depend_on_the_order_of_the_rows(self):
        rnd = random.Random(7)
        ys = value_pool(rnd, 2) + value_pool(rnd, 1) + value_pool(rnd, 5)
        first = None
        for _ in range(6):
            rnd.shuffle(ys)
            self.sb.write("d.csv", "y\n" + "\n".join(repr(y) for y in ys) + "\n")
            self.sb.run("fit", "--overwrite", "--target", "y", "--output", "m.cml", "d.csv")
            r = self.sb.run("eval", "--model-file", "m.cml", "d.csv")
            digest = (without_provenance(self.sb.read("m.cml")), r.stdout)
            if first is None:
                first = digest
            self.assertEqual(digest, first)

    def test_a_sum_that_leaves_64_bits_and_a_double_is_exact(self):
        # 20,000 rows of 1e100 and one of -2e104: a double sum keeps the rounding error of every step; the exact one has none
        ys = [1e100] * 20000 + [-2e104]
        self.sb.write("d.csv", "y\n" + "\n".join(repr(y) for y in ys) + "\n")
        r = self.sb.run("fit", "--target", "y", "--output", "m.cml", "d.csv")
        self.assertEqual(float(r.data()["summary"]["mean"]), oracle_mean(ys))
        r = self.sb.run("eval", "--model-file", "m.cml", "d.csv")
        want = float(oracle_sse(ys, oracle_mean(ys)))
        self.assertEqual(float(r.data()["metrics"]["sse"]), want)

    def test_the_largest_doubles_have_a_mean(self):
        ys = [1.7976931348623157e308] * 5
        self.sb.write("d.csv", "y\n" + "\n".join(repr(y) for y in ys) + "\n")
        r = self.sb.run("fit", "--target", "y", "--output", "m.cml", "d.csv")
        self.assertEqual(float(r.data()["summary"]["mean"]), 1.7976931348623157e308)
        r = self.sb.run("eval", "--model-file", "m.cml", "d.csv")
        self.assertEqual(float(r.data()["metrics"]["sse"]), 0.0)

    def test_exact_ties_in_the_sse_go_to_even(self):
        # y = +-k * 2^-30 has the mean 0 exactly and sse = 2 k^2 2^-60; k odd with k^2 of 54 bits is an exact tie between two doubles
        for k in range(94906267, 94906267 + 160, 2):
            ys = [math.ldexp(k, -30), -math.ldexp(k, -30)]
            self.sb.write("d.csv", "y\n" + "\n".join(repr(y) for y in ys) + "\n")
            self.sb.run("fit", "--overwrite", "--target", "y", "--output", "m.cml", "d.csv")
            r = self.sb.run("eval", "--model-file", "m.cml", "d.csv")
            want = float(oracle_sse(ys, 0.0))
            self.assertEqual(float(r.data()["metrics"]["sse"]), want, k)

    def test_exact_ties_in_the_mean_go_to_even(self):
        e = 2.0 ** -52
        for ys in ([1.0, 1.0 + e], [1.0 + e, 1.0 + 2 * e], [1.0 + 2 * e, 1.0 + 3 * e], [-1.0, -(1.0 + e)], [1.0, 1.0, 1.0, 1.0 + e, 1.0 + e, 1.0 + e, 1.0 + e, 1.0 + e]):
            self.sb.write("d.csv", "y\n" + "\n".join(repr(y) for y in ys) + "\n")
            r = self.sb.run("fit", "--overwrite", "--target", "y", "--output", "m.cml", "d.csv")
            self.assertEqual(float(r.data()["summary"]["mean"]), oracle_mean(ys), ys)

    def test_signed_zero_is_zero(self):
        self.sb.write("d.csv", "y\n-0.0\n0.0\n-0e5\n")
        r = self.sb.run("fit", "--target", "y", "--output", "m.cml", "d.csv")
        self.assertEqual(r.data()["summary"]["mean"], "0.0")

    def test_a_float_feature_and_an_int_model_column(self):
        self.sb.write("a.csv", "x,y\n1,10\n2,20\n")
        self.sb.run("fit", "--target", "y", "--output", "i.cml", "a.csv")
        self.sb.write("b.csv", "x,y\n1.5,10\n2,20\n")
        r = self.sb.run("predict", "--model-file", "i.cml", "b.csv")
        self.assertEqual(r.first_rule(), "ml.schema.mismatch")
        # the other way round: a float column takes int and dec data
        self.sb.run("fit", "--target", "y", "--float", "x", "--output", "f.cml", "a.csv")
        r = self.sb.run("predict", "--model-file", "f.cml", "b.csv")
        self.assertEqual(r.status, 0, r)
        self.sb.write("c.csv", "x,y\n1e2,10\n")
        self.assertEqual(self.sb.run("predict", "--model-file", "f.cml", "c.csv").status, 0)
        self.assertEqual(self.sb.run("predict", "--model-file", "i.cml", "c.csv").first_rule(), "ml.schema.mismatch")

    def test_a_float_target_is_not_a_class_label(self):
        self.sb.write("d.csv", "a,lab\n1,1e0\n2,0\n")
        r = self.sb.run("fit", "--task", "classification", "--target", "lab", "--output", "m.cml", "d.csv")
        self.assertEqual(r.first_rule(), "ml.label.not-integer")

    def test_a_dec_column_that_needs_more_than_12_digits_is_a_float(self):
        self.sb.write("d.csv", "y\n1234567890.1\n0.123456789012\n")
        r = self.sb.run("fit", "--target", "y", "--output", "m.cml", "d.csv")
        self.assertEqual(r.status, 0, r)
        want = oracle_mean([1234567890.1, 0.123456789012])
        self.assertEqual(float(r.data()["summary"]["mean"]), want)

    def test_the_wide_dec_that_still_fits_keeps_its_exact_arithmetic(self):
        # 12 digits in all: an int/dec path, six more digits than the target, as the CLI slice wrote
        self.sb.write("d.csv", "y\n123456789.123\n0.1\n")
        r = self.sb.run("fit", "--target", "y", "--output", "m.cml", "d.csv")
        self.assertEqual(r.data()["summary"]["mean"], "61728394.611500000")


if __name__ == "__main__":
    unittest.main()

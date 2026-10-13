"""O1 -- the model is real and right.

`fit`, `predict` and `eval` are compared, value for value, with an oracle written here with exact rational arithmetic
(fractions.Fraction, Python integers): the training mean rounded half away from zero to 6 more digits than the target's scale,
the majority class and its probability, the accuracy, the sum of squared errors as an exact decimal, and the RMSE by the three
IEEE steps docs/cli.md section 3.3 names. The data are seeded and include values whose sums leave 64 bits."""

import math
import random
import unittest
from fractions import Fraction

from harness import Sandbox


def dec_text(m, scale):
    """Mantissa m at `scale` as the text a CSV would hold."""
    sign = "-" if m < 0 else ""
    d = str(abs(m)).rjust(scale + 1, "0")
    return sign + (d if scale == 0 else d[:-scale] + "." + d[-scale:])


def put_scaled(v, scale):
    return dec_text(v, scale)


def round_half_away(x):
    s = -1 if x < 0 else 1
    a = abs(x)
    f = math.floor(a)
    if a - f >= Fraction(1, 2):
        f += 1
    return s * f


def oracle_mean(mants, scale):
    """The prediction as a scaled integer at scale+6."""
    return round_half_away(Fraction(sum(mants) * 10**6, len(mants)))


def oracle_eval(mants_eval, scale_eval, p, f):
    """(sse_int, n): errors in units of 10^-f, against the prediction p (a scaled integer at f)."""
    sse = 0
    for m in mants_eval:
        y = m * 10 ** (f - scale_eval)
        e = y - p
        sse += e * e
    return sse


def float_rmse(sse, n, f):
    limbs = []
    t = sse
    for _ in range(6):
        limbs.append(t % 10**9)
        t //= 10**9
    assert t == 0
    x = 0.0
    for limb in reversed(limbs):
        x = x * 1e9 + float(limb)
    r = math.sqrt(x / float(n)) + 0.5
    return int(r)


class Oracle(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()

    def tearDown(self):
        self.sb.cleanup()

    def regression_case(self, rng, n, scale, digits, negative, n_eval, scale_eval, digits_eval):
        def mant(d):
            m = rng.randrange(10**d) if d > 0 else 0
            if rng.random() < 0.15:
                m = 10**d - 1 if d > 0 else 0
            return -m if negative and rng.random() < 0.5 else m

        train = [mant(digits) for _ in range(n)]
        # one value at the scale: the file's scale is the most fractional digits seen
        train[0] = int("1" * min(digits, 3)) if digits else 1
        extra = [(rng.randrange(1000), rng.randrange(10**3)) for _ in range(n)]
        text = "id,f,y\n" + "".join("%d,%s,%s\n" % (i, dec_text(extra[i][1], 3), dec_text(train[i], scale)) for i in range(n))
        self.sb.write("train.csv", text)
        r = self.sb.run("fit", "--overwrite", "--target", "y", "--output", "m.cml", "train.csv", "--max-rows", "100000000")
        self.assertEqual(r.status, 0, r)
        p = oracle_mean(train, scale)
        f = scale + 6
        self.assertEqual(r.data()["summary"]["mean"], put_scaled(p, f), "mean of %d values at scale %d" % (n, scale))
        # predict
        r = self.sb.run("predict", "--model-file", "m.cml", "--limit", "5", "train.csv")
        self.assertEqual(r.status, 0, r)
        self.assertEqual(r.data()["rows"], [[put_scaled(p, f)]] * min(5, n))
        self.assertEqual(r.data()["total_rows"], n)
        # eval on other values, with fewer fractional digits
        ev = [mant(digits_eval) for _ in range(n_eval)]
        text = "id,f,y\n" + "".join("%d,0.5,%s\n" % (i, dec_text(ev[i], scale_eval)) for i in range(n_eval))
        self.sb.write("ev.csv", text)
        r = self.sb.run("eval", "--model-file", "m.cml", "ev.csv")
        self.assertEqual(r.status, 0, r)
        sse = oracle_eval(ev, scale_eval, p, f)
        want_sse = put_scaled(sse, 2 * f)
        self.assertEqual(r.data()["metrics"]["sse"], want_sse)
        self.assertEqual(r.data()["metrics"]["rmse"], put_scaled(float_rmse(sse, n_eval, f), f))
        self.assertEqual(r.data()["metrics"]["prediction"], put_scaled(p, f))
        # the float steps are a good rmse: within 1e-12 relative of the exact square root
        true = math.sqrt(Fraction(sse, n_eval))
        got = float_rmse(sse, n_eval, f)
        self.assertLessEqual(abs(got - true), 1 + 1e-12 * true)

    def test_regression_seeded(self):
        rng = random.Random(20261012)
        for i in range(40):
            scale = rng.choice((0, 0, 1, 2, 3, 6, 9))
            digits = rng.randint(1, 12 - scale) if scale < 12 else 1
            digits = max(1, min(digits, 12 - scale))
            scale_eval = rng.randint(0, scale)
            digits_eval = rng.randint(1, max(1, 12 - scale))
            digits_eval = min(digits_eval, 12 - scale)
            n = rng.choice((1, 2, 3, 7, 50, 300))
            n_eval = rng.choice((1, 4, 100))
            self.regression_case(rng, n, scale, digits, rng.random() < 0.5, n_eval, scale_eval, digits_eval)

    def test_regression_extremes(self):
        # mantissas at the bound: 999999999999 (12 digits), negatives, one row, zero
        for values, scale in (([999999999999] * 3, 0), ([-999999999999, 999999999999], 0), ([0], 0), ([5], 3),
                              ([999999999999, 0, 1], 6 - 6), ([-7, -7, -8], 2)):
            text = "y\n" + "".join(dec_text(v, scale) + "\n" for v in values)
            self.sb.write("e.csv", text)
            r = self.sb.run("fit", "--overwrite", "--target", "y", "--output", "e.cml", "e.csv")
            self.assertEqual(r.status, 0, r)
            self.assertEqual(r.data()["summary"]["mean"], put_scaled(oracle_mean(values, scale), scale + 6), values)
            r = self.sb.run("eval", "--model-file", "e.cml", "e.csv")
            self.assertEqual(r.status, 0, r)
            f = scale + 6
            sse = oracle_eval(values, scale, oracle_mean(values, scale), f)
            self.assertEqual(r.data()["metrics"]["sse"], put_scaled(sse, 2 * f), values)

    def test_the_mean_rounds_half_away_from_zero(self):
        for values, want in (([1, 2], "1.500000"), ([-1, -2], "-1.500000"), ([0, 0, 1], "0.333333"), ([0, 0, 2], "0.666667"),
                             ([1, 1, 0, 0, 0, 0, 0, 0], "0.250000"), ([-1, 0, 0], "-0.333333"),
                             ([1] + [0] * 127, "0.007813"), ([-1] + [0] * 127, "-0.007813"), ([3] + [0] * 127, "0.023438"),
                             ([5] + [0] * 31, "0.156250"), ([1] + [0] * 639, "0.001563")):
            self.sb.write("e.csv", "y\n" + "".join("%d\n" % v for v in values))
            r = self.sb.run("fit", "--overwrite", "--target", "y", "--output", "e.cml", "e.csv")
            self.assertEqual(r.data()["summary"]["mean"], want, values)

    def classification_case(self, rng, n, labels):
        ys = [rng.choice(labels) for _ in range(n)]
        if len(set(ys)) < 2:
            ys[0], ys[1] = labels[0], labels[1]
        text = "f,lab\n" + "".join("%d,%d\n" % (rng.randrange(100), y) for y in ys)
        self.sb.write("cl.csv", text)
        r = self.sb.run("fit", "--overwrite", "--task", "classification", "--target", "lab", "--output", "c.cml", "cl.csv")
        self.assertEqual(r.status, 0, r)
        c0, c1 = sorted(set(ys))
        n0, n1 = ys.count(c0), ys.count(c1)
        majority = c1 if n1 > n0 else c0
        proba = (2 * n1 * 10**6 + n) // (2 * n)
        s = r.data()["summary"]
        self.assertEqual((s["classes"], s["counts"], s["majority"], s["probability"]), ([c0, c1], [n0, n1], majority, put_scaled(proba, 6)))
        # eval on another sample, with labels the model has not seen
        ev = [rng.choice(labels + [424242]) for _ in range(40)]
        self.sb.write("ce.csv", "f,lab\n" + "".join("1,%d\n" % y for y in ev))
        r = self.sb.run("eval", "--model-file", "c.cml", "ce.csv")
        self.assertEqual(r.status, 0, r)
        correct = sum(1 for y in ev if y == majority)
        m = r.data()["metrics"]
        self.assertEqual((m["correct"], m["total"], m["prediction"]), (correct, 40, majority))
        self.assertEqual(m["accuracy"], put_scaled((2 * correct * 10**6 + 40) // 80, 6))
        r = self.sb.run("predict", "--model-file", "c.cml", "--limit", "3", "ce.csv")
        self.assertEqual(r.data()["rows"], [[str(majority), put_scaled(proba, 6)]] * 3)

    def test_classification_seeded(self):
        rng = random.Random(99)
        for labels in ([0, 1], [-5, 7], [1, 0], [999999999999, -999999999999], [3, 4]):
            for n in (2, 3, 10, 101):
                self.classification_case(rng, n, labels)

    def test_a_tie_goes_to_the_smaller_label(self):
        self.sb.write("t.csv", "lab\n5\n3\n5\n3\n")
        r = self.sb.run("fit", "--task", "classification", "--target", "lab", "--output", "t.cml", "t.csv")
        self.assertEqual(r.data()["summary"], {"classes": [3, 5], "counts": [2, 2], "majority": 3, "probability": "0.500000"})

    def test_sums_that_leave_64_bits(self):
        # 10,000,000 rows of 12-digit values: the sum, about 1e19, is past 2^63. Mean and squared errors stay exact.
        n = 10_000_000
        base = 999_999_999_000
        vals_mod = [(i * 7919) % 1000 for i in range(1000)]
        total = sum(base + vals_mod[i % 1000] for i in range(n))
        self.assertGreater(total, 2**63)
        chunk = "".join("%d\n" % (base + v) for v in vals_mod)
        with open(self.sb.dir / "big.csv", "w") as fh:
            fh.write("y\n")
            for _ in range(n // 1000):
                fh.write(chunk)
        r = self.sb.run("fit", "--max-rows", "20000000", "--target", "y", "--output", "big.cml", "big.csv")
        self.assertEqual(r.status, 0, r)
        p = round_half_away(Fraction(total * 10**6, n))
        self.assertEqual(r.data()["summary"]["mean"], put_scaled(p, 6))
        r = self.sb.run("eval", "--max-rows", "20000000", "--model-file", "big.cml", "big.csv")
        self.assertEqual(r.status, 0, r)
        # sse by blocks of 1000 rows (every block is the same list of values)
        block = sum((((base + v) * 10**6 - p) ** 2) for v in vals_mod)
        sse = block * (n // 1000)
        self.assertEqual(r.data()["metrics"]["sse"], put_scaled(sse, 12))
        self.assertEqual(r.data()["metrics"]["rmse"], put_scaled(float_rmse(sse, n, 6), 6))


if __name__ == "__main__":
    unittest.main()

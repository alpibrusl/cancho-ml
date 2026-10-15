"""N1 to N7 of docs/nonfinite.md: a prediction, a probability or a squared distance that is not a finite number is the refusal `ml.predict.non-finite`, from `ml`, `ml-predict` and `ml-serve`.

What is judged from outside: the reproducer of issue 33 (a linear model and a feature of 1e308) is exit 8 with the row and the column in `detail` and no cell on standard output; the boundary of the
largest feature a model answers; a NaN that no infinity alone explains (a logistic score of inf minus inf); the squared distance of k-means with and without `--distance`, and in `eval`, where a sum
of finite squared distances that overflows is still `null`; tree and forest regression, which cannot be made infinite; paging, where the check covers every row scored and not the page; the three
binaries giving the same rule and the same row; and a fuzz of huge, subnormal and cancelling inputs through every kind of model on the three binaries, where no answer holds `i.nf`, `inf` or `nan`
and nothing traps."""

import json
import math
import subprocess
import unittest

from harness import BIN, PREDICT_BIN, TRAPS, Sandbox, run, validate
from serve_lib import Server

RULE = "ml.predict.non-finite"
SB = None


def ml(*args, **kw):
    return SB.run(*args, **kw)


def pr(*args, **kw):
    return SB.run(*args, binary_path=PREDICT_BIN, **kw)


def csv_of(header, rows):
    return ",".join(header) + "\n" + "".join(",".join(r) + "\n" for r in rows)


def detail_of(r):
    return r.error()["detail"]


LIN_TRAIN = "x,y\n1.5,15.5\n2.5,25.5\n3.5,35.5\n4.5,45.5\n5.5,55.5\n"


def lin2_train():
    rows = []
    for i in range(20):
        a = (i * 37 % 19) / 19.0
        b = (i * 11 % 17) / 17.0
        rows.append(("%r" % a, "%r" % b, "%r" % (3 * a - 3 * b + 0.1)))
    return csv_of(["a", "b", "y"], rows)


def lg_train():
    # small, nearly separable: with l2 = 0 the coefficients are in the thousands (a, b near 1e-3 apart)
    import random

    rng = random.Random(5)
    rows = []
    for i in range(60):
        a = rng.gauss(0, 1e-3)
        b = rng.gauss(0, 1e-3)
        rows.append(("%r" % a, "%r" % b, "1" if a - b + rng.gauss(0, 1e-3) > 0 else "0"))
    return csv_of(["a", "b", "lab"], rows)


def setUpModule():
    global SB
    SB = Sandbox()
    SB.write("lin.csv", LIN_TRAIN)
    assert ml("fit", "--model", "linear", "--target", "y", "--float", "x,y", "--output", "lin.ml", "lin.csv").status == 0
    SB.write("lin2.csv", lin2_train())
    assert ml("fit", "--model", "linear", "--target", "y", "--float", "a,b,y", "--output", "lin2.ml", "lin2.csv").status == 0
    SB.write("lg.csv", lg_train())
    r = ml("fit", "--model", "logistic", "--target", "lab", "--float", "a,b", "--iterations", "100", "--output", "lgbig.ml", "lg.csv")
    assert r.status == 0, r
    r = ml("fit", "--model", "logistic", "--target", "lab", "--float", "a,b", "--iterations", "100", "--l2", "0.001", "--output", "lgsmall.ml", "lg.csv")
    assert r.status == 0, r
    SB.write("km.csv", "a,b\n1,1\n1.1,0.9\n5,5\n5.1,4.9\n")
    assert ml("fit", "--model", "kmeans", "--k", "2", "--float", "a,b", "--output", "km.ml", "km.csv").status == 0
    SB.write("km1.csv", "x\n-1\n1\n-1\n1\n")
    assert ml("fit", "--model", "kmeans", "--k", "1", "--float", "x", "--output", "km1.ml", "km1.csv").status == 0
    SB.write("tr.csv", csv_of(["x", "y"], [("%d.5" % i, "1e100" if i % 2 else "-1e100") for i in range(10)]))
    assert ml("fit", "--model", "tree", "--task", "regression", "--target", "y", "--float", "x,y", "--output", "tree.ml", "tr.csv").status == 0
    assert ml("fit", "--model", "forest", "--task", "regression", "--trees", "8", "--target", "y", "--float", "x,y", "--output", "forest.ml", "tr.csv").status == 0
    SB.write("trc.csv", csv_of(["x", "lab"], [("%d.5" % i, str(i % 2)) for i in range(10)]))
    assert ml("fit", "--model", "tree", "--task", "classification", "--target", "lab", "--float", "x", "--output", "treec.ml", "trc.csv").status == 0
    SB.write("const.csv", "x,y\n1,1.7e308\n2,1.7e308\n3,1.6e308\n")
    assert ml("fit", "--model", "constant", "--task", "regression", "--float", "y", "--target", "y", "--drop", "x", "--output", "const.ml", "const.csv").status == 0


def tearDownModule():
    SB.cleanup()


def record(header, row):
    return "{" + ",".join(json.dumps(h) + ":" + c for h, c in zip(header, row)) + "}"


def served(model, header, rows, batch=True):
    """(status, the problem or document) of the rows sent to ml-serve as one batch, or as one record when batch is false"""
    with Server(SB.dir, model, "--max-batch-rows", "1000") as srv:
        if batch:
            body = ('{"rows":[' + ",".join(record(header, r) for r in rows) + "]}").encode()
            resp = srv.request("POST", "/predict/batch", body=body, headers={"Content-Type": "application/json"})
        else:
            resp = srv.request("POST", "/predict", body=record(header, rows[0]).encode(), headers={"Content-Type": "application/json"})
        return resp.status, resp.json()


class Reproducer(unittest.TestCase):
    """N1"""

    def test_the_issue_is_exit_8_with_the_row_and_the_column(self):
        SB.write("big.csv", "x\n1e308\n2.5\n")
        r = ml("predict", "--model-file", "lin.ml", "big.csv")
        self.assertEqual((r.status, r.first_rule()), (8, RULE), r)
        self.assertEqual(detail_of(r), {"row": 1, "column": "prediction"})
        self.assertEqual(validate(r), [])
        self.assertNotIn(b"i.nf", r.stdout)
        self.assertNotIn("rows", r.doc().get("data") or {})

    def test_the_row_is_the_first_one_and_one_based(self):
        SB.write("big3.csv", "x\n2.5\n3.5\n-1e308\n1e308\n")
        r = ml("predict", "--model-file", "lin.ml", "big3.csv")
        self.assertEqual((r.status, r.first_rule(), detail_of(r)["row"]), (8, RULE, 3), r)

    def test_text_format_writes_nothing_to_standard_output(self):
        SB.write("big.csv", "x\n1e308\n2.5\n")
        r = ml("predict", "--model-file", "lin.ml", "--format", "text", "big.csv")
        self.assertEqual(r.status, 8, r)
        self.assertEqual(r.stdout, b"")
        self.assertIn(RULE.encode(), r.stderr)

    def test_the_repair_is_none_and_the_exit_is_the_table_s(self):
        SB.write("big.csv", "x\n1e308\n2.5\n")
        r = ml("predict", "--model-file", "lin.ml", "big.csv")
        self.assertIsNone(r.error()["repair"])
        introspect = json.loads(subprocess.run([BIN_PATH(), "introspect"], capture_output=True, stdin=subprocess.DEVNULL).stdout)
        rules = {x["rule"] if isinstance(x, dict) else x: x for x in introspect["rules"]}
        self.assertIn(RULE, rules)

    def test_the_boundary_of_the_largest_feature(self):
        # the coefficient is 9.999999999999984: 1.79e307 times it is 1.79e308, below the largest double 1.7976931348623157e308; 1.8e307 times it is above
        for x, ok in (("1.79e307", True), ("1.8e307", False), ("-1.79e307", True), ("-1.8e307", False)):
            SB.write("edge.csv", "x\n%s\n" % x)
            r = ml("predict", "--model-file", "lin.ml", "edge.csv")
            if ok:
                self.assertEqual(r.status, 0, (x, r))
                self.assertTrue(math.isfinite(float(r.data()["rows"][0][0])), (x, r))
            else:
                self.assertEqual((r.status, r.first_rule()), (8, RULE), (x, r))

    def test_a_finite_file_is_answered_as_before(self):
        SB.write("fine.csv", "x\n1.5\n2.5\n")
        r = ml("predict", "--model-file", "lin.ml", "fine.csv")
        self.assertEqual(r.status, 0, r)
        self.assertEqual(r.data()["row_count"], 2)
        self.assertEqual([float(c[0]) for c in r.data()["rows"]], [15.500000000000028, 25.500000000000014])


def BIN_PATH():
    from harness import BIN

    return BIN


class Nan(unittest.TestCase):
    """N4: the NaN of an infinity minus an infinity, and the probability of a logistic model"""

    def test_linear_inf_minus_inf_is_a_nan_and_is_refused(self):
        SB.write("cancel.csv", "a,b\n0.5,0.5\n1e308,1e308\n")
        r = ml("predict", "--model-file", "lin2.ml", "cancel.csv")
        self.assertEqual((r.status, r.first_rule(), detail_of(r)), (8, RULE, {"row": 2, "column": "prediction"}), r)

    def test_logistic_probability_is_nan_when_two_infinite_terms_cancel(self):
        SB.write("lgh.csv", "a,b\n0.0005,0.0005\n1e308,1e308\n")
        r = ml("predict", "--model-file", "lgbig.ml", "--proba", "lgh.csv")
        self.assertEqual((r.status, r.first_rule()), (8, RULE), r)
        self.assertEqual(detail_of(r)["row"], 2)
        self.assertIn(detail_of(r)["column"], ("label", "probability"))
        # without --proba the label is chosen from the same value, so it is refused too
        r = ml("predict", "--model-file", "lgbig.ml", "lgh.csv")
        self.assertEqual((r.status, r.first_rule()), (8, RULE), r)

    def test_logistic_probability_is_finite_for_a_huge_input_with_a_small_coefficient(self):
        SB.write("lgs.csv", "a,b\n1e308,1e308\n1e308,-1e308\n-1e308,1e308\n1e-320,5e-324\n1,1\n1e300,-1e300\n")
        r = ml("predict", "--model-file", "lgsmall.ml", "--proba", "lgs.csv")
        self.assertEqual(r.status, 0, r)
        for label, p in r.data()["rows"]:
            self.assertIn(label, ("0", "1"))
            self.assertTrue(0.0 <= float(p) <= 1.0)

    def test_eval_refuses_the_same_nan_with_the_row(self):
        SB.write("lge.csv", "a,b,lab\n0.0005,0.0005,1\n1e308,1e308,0\n")
        r = ml("eval", "--model-file", "lgbig.ml", "lge.csv")
        self.assertEqual((r.status, r.first_rule(), detail_of(r)["row"]), (8, RULE, 2), r)


class Clustering(unittest.TestCase):
    """N2"""

    def setUp(self):
        SB.write("kb.csv", "a,b\n1,1\n1e200,1\n")

    def test_the_distance_that_overflows_is_refused_with_its_row(self):
        r = ml("predict", "--model-file", "km.ml", "--distance", "kb.csv")
        self.assertEqual((r.status, r.first_rule(), detail_of(r)), (8, RULE, {"row": 2, "column": "distance"}), r)

    def test_without_distance_the_answer_is_the_labels_as_before(self):
        r = ml("predict", "--model-file", "km.ml", "kb.csv")
        self.assertEqual(r.status, 0, r)
        self.assertEqual(r.data()["columns"], ["cluster"])
        self.assertEqual(len(r.data()["rows"]), 2)

    def test_eval_refuses_a_squared_distance_that_is_not_finite(self):
        SB.write("kbt.csv", "a,b\n1,1\n1e200,1\n5,5\n")
        r = ml("eval", "--model-file", "km.ml", "kbt.csv")
        self.assertEqual((r.status, r.first_rule(), detail_of(r)), (8, RULE, {"row": 2, "column": "distance"}), r)

    def test_a_sum_of_finite_squared_distances_that_overflows_is_still_null(self):
        # one centre at 0; each squared distance is 1.44e308, below the largest double; the sum of four is not
        SB.write("km1h.csv", "x\n1.2e154\n-1.2e154\n1.2e154\n-1.2e154\n")
        r = ml("eval", "--model-file", "km1.ml", "km1h.csv")
        self.assertEqual(r.status, 0, r)
        m = r.data()["metrics"]
        self.assertIsNone(m["inertia"])
        self.assertEqual(m["sizes"], [4])
        self.assertEqual(float(m["mean_distance"][0]), 1.2e154)
        # a squared distance of 1.9e308 is not finite itself: refused
        SB.write("km1i.csv", "x\n1.2e154\n1.4e154\n")
        r = ml("eval", "--model-file", "km1.ml", "km1i.csv")
        self.assertEqual((r.status, r.first_rule(), detail_of(r)["row"]), (8, RULE, 2), r)

    def test_the_largest_distance_that_is_accepted(self):
        for x, ok in (("1.3e154", True), ("1.4e154", False)):
            SB.write("e.csv", "x\n%s\n" % x)
            r = ml("predict", "--model-file", "km1.ml", "--distance", "e.csv")
            self.assertEqual(r.status == 0, ok, (x, r))


class TreesAndConstants(unittest.TestCase):
    """N3: no fixture makes a tree or a forest answer an infinity"""

    def test_regression_at_the_boundary_of_the_target_bound_is_finite(self):
        SB.write("th.csv", "x\n1e308\n-1e308\n3.5\n0\n-0.0\n5e-324\n")
        for model in ("tree.ml", "forest.ml"):
            r = ml("predict", "--model-file", model, "th.csv")
            self.assertEqual(r.status, 0, (model, r))
            for (v,) in r.data()["rows"]:
                self.assertTrue(math.isfinite(float(v)) and abs(float(v)) <= 1e100, (model, v))

    def test_a_target_past_the_bound_is_refused_at_the_fit(self):
        SB.write("t2.csv", csv_of(["x", "y"], [("%d.5" % i, "1.1e100") for i in range(10)]))
        r = ml("fit", "--model", "tree", "--task", "regression", "--target", "y", "--float", "x,y", "--output", "t2.ml", "--overwrite", "t2.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.input.non-finite"), r)

    def test_classification_is_a_label_and_a_ratio(self):
        SB.write("tc.csv", "x\n1e308\n-1e308\n0\n")
        r = ml("predict", "--model-file", "treec.ml", "tc.csv")
        self.assertEqual(r.status, 0, r)

    def test_a_constant_float_model_is_finite_by_construction(self):
        SB.write("c.csv", "x\n1\n2\n")
        r = ml("predict", "--model-file", "const.ml", "--drop", "x", "c.csv")
        self.assertEqual(r.status, 0, r)
        self.assertEqual(r.data()["rows"][0], ["1.6666666666666666e308"])


class Paging(unittest.TestCase):
    """N5: the check covers every row scored, not the page"""

    def setUp(self):
        rows = ["2.5"] * 6
        self.good = rows
        for at in (0, 2, 5):
            r = list(rows)
            r[at] = "1e308"
            SB.write("page%d.csv" % at, "x\n" + "\n".join(r) + "\n")

    def test_a_bad_row_outside_the_window_refuses_the_request(self):
        for at in (0, 2, 5):
            for flags in ([], ["--limit", "2"], ["--limit", "1", "--from", "3"], ["--from", "1", "--limit", "1"], ["--from", "100"]):
                r = ml("predict", "--model-file", "lin.ml", "page%d.csv" % at, *flags)
                self.assertEqual((r.status, r.first_rule(), detail_of(r)["row"]), (8, RULE, at + 1), (at, flags, r))
                self.assertNotIn(b'"rows"', r.stdout)

    def test_text_format_and_ml_predict_agree(self):
        for at in (0, 5):
            for binary in (ml, pr):
                r = binary("predict", "--model-file", "lin.ml", "--limit", "1", "--format", "text", "page%d.csv" % at)
                self.assertEqual((r.status, r.stdout), (8, b""), (at, r))

    def test_a_clean_file_pages_as_before(self):
        SB.write("clean.csv", "x\n" + "\n".join(self.good) + "\n")
        r = ml("predict", "--model-file", "lin.ml", "--limit", "2", "--from", "1", "clean.csv")
        self.assertEqual(r.status, 0, r)
        self.assertEqual((r.data()["row_count"], r.data()["total_rows"], r.data()["truncated"], r.data()["next"]), (2, 6, True, {"from": 3}))

    def test_eval_with_a_split_names_the_row_of_the_file(self):
        rows = ["%d.5,%r" % (i, 10 * (i + 0.5) + 0.5) for i in range(20)]
        rows[6] = "1e308,1"
        SB.write("sp.csv", "x,y\n" + "\n".join(rows) + "\n")
        refused = fine = 0
        for seed in range(1, 21):
            r = ml("eval", "--model-file", "lin.ml", "--split", "train=0.5", "--seed", str(seed), "sp.csv")
            if r.status == 0:
                fine += 1
                continue
            refused += 1
            self.assertEqual((r.status, r.first_rule(), detail_of(r)["row"]), (8, RULE, 7), (seed, r))
        self.assertGreaterEqual(refused, 3)
        self.assertGreaterEqual(fine, 3)


class FitIsUnchanged(unittest.TestCase):
    """`fit` keeps what it did: a coefficient that is not finite is `ml.fit.non-finite`, and a held-out row whose prediction is not finite is not a refusal of the fit (its metrics are `null`)"""

    def test_a_fit_whose_coefficients_overflow_is_still_ml_fit_non_finite(self):
        SB.write("ovf.csv", "x,y\n1e200,1\n2e200,0\n3e200,1\n-1e200,0\n")
        r = ml("fit", "--model", "linear", "--target", "y", "--output", "ovf.ml", "--overwrite", "ovf.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.fit.non-finite"), r)

    def test_a_held_out_row_that_overflows_is_refused_by_the_fit_as_before(self):
        # the held-out rows are scored by the metrics library, which refuses a prediction that is not finite as `ml.input.non-finite` (detail.metric): the fit does not use the new rule
        rows = ["%d.5,%r" % (i, 10 * (i + 0.5) + 0.5) for i in range(20)]
        rows[6] = "1e308,1"
        SB.write("fsp.csv", "x,y\n" + "\n".join(rows) + "\n")
        seen = set()
        for seed in range(1, 21):
            r = ml("fit", "--model", "linear", "--target", "y", "--float", "x,y", "--split", "train=0.5", "--seed", str(seed), "--output", "fsp.ml", "--overwrite", "fsp.csv")
            self.assertEqual(r.status, 8, (seed, r))
            self.assertIn(r.first_rule(), ("ml.fit.non-finite", "ml.input.non-finite"), (seed, r))
            seen.add(r.first_rule())
        self.assertEqual(seen, {"ml.fit.non-finite", "ml.input.non-finite"})


def cases():
    """(name, model, header, rows, flags, the row that is refused or None, the column)"""
    return [
        ("linear first", "lin.ml", ["x"], [["1e308"], ["2.5"]], [], 1, "prediction"),
        ("linear third", "lin.ml", ["x"], [["2.5"], ["3.5"], ["-1e308"], ["1e308"]], [], 3, "prediction"),
        ("linear nan", "lin2.ml", ["a", "b"], [["0.5", "0.5"], ["1e308", "1e308"]], [], 2, "prediction"),
        ("logistic nan", "lgbig.ml", ["a", "b"], [["0.0005", "0.0005"], ["1e308", "1e308"]], ["--proba"], 2, None),
        ("kmeans distance", "km.ml", ["a", "b"], [["1", "1"], ["1e200", "1"]], ["--distance"], 2, "distance"),
        ("linear clean", "lin.ml", ["x"], [["1.5"], ["2.5"]], [], None, None),
        ("tree clean", "tree.ml", ["x"], [["1e308"], ["2.5"]], [], None, None),
    ]


class Agreement(unittest.TestCase):
    """N6: `ml predict`, `ml-predict predict` and `ml-serve` give one rule and one row"""

    def test_three_binaries(self):
        for name, model, header, rows, flags, bad, column in cases():
            with self.subTest(case=name):
                SB.write("agree.csv", csv_of(header, rows))
                a = ml("predict", "--model-file", model, "agree.csv", *flags)
                b = pr("predict", "--model-file", model, "agree.csv", *flags)
                self.assertEqual(a.stdout.replace(BIN_PATH().encode(), PREDICT_BIN.encode()), b.stdout)
                self.assertEqual(a.status, b.status)
                status, doc = served(model, header, rows)
                if bad is None:
                    self.assertEqual((a.status, status), (0, 200), (a, doc))
                    self.assertEqual(a.data()["rows"], doc["rows"], name)
                    continue
                self.assertEqual((a.status, a.first_rule(), detail_of(a)["row"]), (8, RULE, bad), a)
                if column:
                    self.assertEqual(detail_of(a)["column"], column)
                self.assertEqual(status, 422, doc)
                self.assertEqual(doc["rule"], RULE)
                self.assertEqual(doc["error"]["rule"], RULE)
                self.assertEqual(doc["error"]["detail"]["row"], str(bad))
                self.assertEqual(doc["error"]["detail"]["column"], detail_of(a)["column"])
                self.assertEqual(doc["error"]["code"], a.error()["code"])
                self.assertEqual(a.stdout.count(b"i.nf"), 0)

    def test_one_record_is_row_one(self):
        status, doc = served("lin.ml", ["x"], [["1e308"]], batch=False)
        self.assertEqual((status, doc["rule"], doc["error"]["detail"]["row"]), (422, RULE, "1"), doc)

    def test_the_old_tag_is_gone(self):
        status, doc = served("lin.ml", ["x"], [["1e308"]], batch=False)
        self.assertNotIn("serve.non-finite-prediction", json.dumps(doc))
        for binary in (BIN_PATH(), PREDICT_BIN):
            out = subprocess.run([binary, "introspect"], capture_output=True, stdin=subprocess.DEVNULL).stdout.decode()
            self.assertIn(RULE, out)
        out = subprocess.run([__import__("serve_lib").SERVE_BIN, "introspect"], capture_output=True, stdin=subprocess.DEVNULL).stdout.decode()
        self.assertIn(RULE, out)
        self.assertNotIn("serve.non-finite-prediction", out)


VALUES = ["0", "-0.0", "1", "-1", "0.5", "1e-320", "5e-324", "-5e-324", "2.2250738585072014e-308", "1e154", "1.3e154", "1.4e154", "1e155", "1e200", "1.79e307", "1.8e307", "1e308",
          "-1e308", "1.7976931348623157e308", "-1.7976931348623157e308"]
FUZZ_MODELS = [("lin.ml", ["x"], []), ("lin2.ml", ["a", "b"], []), ("lgbig.ml", ["a", "b"], ["--proba"]), ("lgsmall.ml", ["a", "b"], ["--proba"]), ("km.ml", ["a", "b"], ["--distance"]),
               ("km.ml", ["a", "b"], []), ("tree.ml", ["x"], []), ("forest.ml", ["x"], []), ("treec.ml", ["x"], []), ("const.ml", ["x"], ["--drop", "x"])]


def combos(width):
    if width == 1:
        return [[v] for v in VALUES]
    return [[a, b] for a in VALUES for b in VALUES]


def cells_finite(doc):
    for row in doc["rows"]:
        for c in row:
            try:
                v = float(c)
            except ValueError:
                return False
            if not math.isfinite(v):
                return False
            if c.lower() in ("inf", "-inf", "nan", "i.nf") or "i.nf" in c:
                return False
    return True


class Fuzz(unittest.TestCase):
    """N7"""

    def test_no_answer_holds_an_infinity_and_nothing_traps(self):
        total = refused = answered = 0
        for model, header, flags in FUZZ_MODELS:
            rows = combos(len(header))
            SB.write("fz.csv", csv_of(header, rows))
            # each row alone: the verdict of a row, on `ml` and on `ml-predict`
            verdict = []
            for i, row in enumerate(rows):
                SB.write("fz1.csv", csv_of(header, [row]))
                a = ml("predict", "--model-file", model, "fz1.csv", *flags)
                self.assertNotIn(a.status, TRAPS, (model, row, a))
                self.assertIn(a.status, (0, 8), (model, row, a))
                if a.status == 0:
                    self.assertTrue(cells_finite(a.data()), (model, row, a))
                    verdict.append(None)
                    answered += 1
                else:
                    self.assertEqual(a.first_rule(), RULE, (model, row, a))
                    self.assertEqual(detail_of(a)["row"], 1)
                    verdict.append(detail_of(a)["column"])
                    refused += 1
                total += 1
                if i % 7 == 0:
                    b = pr("predict", "--model-file", model, "fz1.csv", *flags)
                    self.assertEqual((a.status, a.first_rule()), (b.status, b.first_rule()))
            # the whole file: refused at the first row that is refused alone, else answered
            a = ml("predict", "--model-file", model, "fz.csv", "--limit", "3", *flags)
            first = next((i for i, v in enumerate(verdict) if v is not None), None)
            if first is None:
                self.assertEqual(a.status, 0, (model, a))
            else:
                self.assertEqual((a.status, a.first_rule(), detail_of(a)["row"], detail_of(a)["column"]), (8, RULE, first + 1, verdict[first]), (model, a))
        self.assertGreater(refused, 40)
        self.assertGreater(answered, 100)
        print("non-finite fuzz: %d rows through %d model/flag pairs, %d answered with finite cells, %d refused with %s, no trap" % (total, len(FUZZ_MODELS), answered, refused, RULE))

    def test_the_server_agrees_on_a_whole_file(self):
        for model, header, flags in (("lin.ml", ["x"], []), ("lin2.ml", ["a", "b"], []), ("km.ml", ["a", "b"], ["--distance"])):
            rows = combos(len(header))[:250]
            SB.write("fz.csv", csv_of(header, rows))
            a = ml("predict", "--model-file", model, "fz.csv", "--limit", "1", *flags)
            status, doc = served(model, header, rows)
            if a.status == 0:
                self.assertEqual(status, 200)
            else:
                self.assertEqual((status, doc["rule"], doc["error"]["detail"]["row"]), (422, RULE, str(detail_of(a)["row"])))


if __name__ == "__main__":
    unittest.main()

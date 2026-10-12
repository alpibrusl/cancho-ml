"""H1 -- no fake models. `introspect` lists what is built; asking for another is a refusal; a float column is refused, never
read approximately; the decimals that are accepted are read exactly."""

import unittest

from harness import Sandbox, introspect


class Honesty(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()

    def tearDown(self):
        self.sb.cleanup()

    def test_the_models_are_exactly_the_built_ones(self):
        flags = {f["name"]: f for f in introspect()["flags"]}
        self.assertEqual(flags["--model"]["kind"], "choice:constant")
        self.assertEqual(flags["--model"]["default"], "constant")

    def test_another_model_name_is_refused(self):
        self.sb.write("reg.csv", "y\n1\n2\n")
        for name in ("logreg", "linear", "tree", "forest", "kmeans", "Constant", "constant ", ""):
            r = self.sb.run("fit", "--model", name, "--target", "y", "--output", "m.cml", "reg.csv")
            self.assertEqual(r.first_rule(), "args.bad-value", (name, r))
            self.assertFalse((self.sb.dir / "m.cml").exists())

    def test_the_stored_plan_names_the_model_it_is(self):
        self.sb.write("reg.csv", "y\n1\n2\n")
        self.sb.run("fit", "--target", "y", "--output", "m.cml", "reg.csv")
        self.assertIn(b"model kind=constant task=regression\n", self.sb.read("m.cml"))

    def test_float_text_is_refused_not_approximated(self):
        for cell in ("1e5", "1E-3", "2.5e0", "1e", "0.1234567890123", "123456789012.5", "1234567890123", "-0.00000000000001"):
            self.sb.write("f.csv", "y\n%s\n" % cell)
            r = self.sb.run("fit", "--overwrite", "--target", "y", "--output", "m.cml", "f.csv")
            self.assertIn(r.first_rule(), ("ml.input.unsupported-type", "ml.input.non-numeric"), (cell, r))
            self.assertEqual(r.status, 8)
        for cell in ("1e5", "0.1234567890123"):
            self.sb.write("f.csv", "y\n%s\n" % cell)
            r = self.sb.run("fit", "--overwrite", "--target", "y", "--output", "m.cml", "f.csv")
            self.assertEqual(r.first_rule(), "ml.input.unsupported-type")

    def test_the_hint_names_the_missing_dependency(self):
        self.sb.write("f.csv", "y\n1e5\n")
        r = self.sb.run("fit", "--target", "y", "--output", "m.cml", "f.csv")
        self.assertIn("float reader", r.error()["hint"])

    def test_decimals_are_exact(self):
        for cell, mean in (("0.1", "0.100000"), ("0.30000000", "0.300000"), ("-0.000001", "-0.000001"), ("007", "7.000000"),
                           ("+5", "5.000000"), ("123456789.123", "123456789.123000")):
            self.sb.write("f.csv", "y\n%s\n" % cell)
            r = self.sb.run("fit", "--overwrite", "--target", "y", "--output", "m.cml", "f.csv")
            self.assertEqual(r.status, 0, (cell, r))
            self.assertEqual(r.data()["summary"]["mean"][: len(mean)], mean[: len(mean)], cell)

    def test_the_summary_says_what_a_constant_is(self):
        self.assertIn("constant", introspect()["summary"])
        self.assertIn("training mean", introspect()["summary"])
        self.assertIn("majority class", introspect()["summary"])


if __name__ == "__main__":
    unittest.main()

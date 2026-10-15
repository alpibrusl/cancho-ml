"""H1 -- no fake models. `introspect` lists what is built; asking for another is a refusal; a float column is read to the nearest
double, never approximately; the decimals that are accepted are read exactly."""

import unittest

import hashlib

from harness import Sandbox, introspect, without_provenance


class Honesty(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()

    def tearDown(self):
        self.sb.cleanup()

    def test_the_models_are_exactly_the_built_ones(self):
        flags = {f["name"]: f for f in introspect()["flags"]}
        self.assertEqual(flags["--model"]["kind"], "choice:constant/linear/logistic/kmeans/tree/forest")
        self.assertEqual(flags["--model"]["default"], "constant")

    def test_another_model_name_is_refused(self):
        self.sb.write("reg.csv", "y\n1\n2\n")
        for name in ("logreg", "KMeans", "kmeans ", "Tree", "Forest", "Linear", "Constant", "constant ", ""):
            r = self.sb.run("fit", "--model", name, "--target", "y", "--output", "m.cml", "reg.csv")
            self.assertEqual(r.first_rule(), "args.bad-value", (name, r))
            self.assertFalse((self.sb.dir / "m.cml").exists())

    def test_a_plan_naming_constant_is_refused(self):
        # the plan grammar knows constant, which the verbs build without a plan (--model constant, docs/cli.md): a plan that names it is refused
        self.sb.write("reg.csv", "a,y\n1,2\n2,3\n3,5\n")
        for line in ("model kind=constant task=regression\n", "model kind=constant task=classification\n"):
            self.sb.write("p.txt", line)
            r = self.sb.run("fit", "--plan", "p.txt", "--target", "y", "--output", "m.cml", "reg.csv")
            self.assertEqual(r.first_rule(), "ml.plan.model-not-built", (line, r))
            self.assertFalse((self.sb.dir / "m.cml").exists())

    def test_the_stored_plan_names_the_model_it_is(self):
        self.sb.write("reg.csv", "y\n1\n2\n")
        self.sb.run("fit", "--target", "y", "--output", "m.cml", "reg.csv")
        self.assertIn(b"model kind=constant task=regression\n", self.sb.read("m.cml"))

    def test_float_text_is_read_exactly_not_approximated(self):
        # the nearest double, ties to even, shortest decimal back (cancho-table's reader, docs/reader.md): never a prefix, never a refusal
        for cell, mean in (("1e5", "100000.0"), ("1E-3", "0.001"), ("2.5e0", "2.5"), ("0.1234567890123", "0.1234567890123"),
                           ("123456789012.5", "123456789012.5"), ("1234567890123", "1234567890123.0"), ("-0.00000000000001", "-1e-14"),
                           ("9007199254740993", "9007199254740992.0"), ("0.1", "0.1")):
            self.sb.write("f.csv", "y\n%s\n" % cell)
            r = self.sb.run("fit", "--overwrite", "--float", "y", "--target", "y", "--output", "m.cml", "f.csv")
            self.assertEqual(r.status, 0, (cell, r))
            self.assertEqual(r.data()["summary"]["mean"], mean, cell)
        for cell in ("1e", "0x10", "1_0", " 1", "1e5.5", "e5"):
            self.sb.write("f.csv", "y\n%s\n" % cell)
            r = self.sb.run("fit", "--overwrite", "--target", "y", "--output", "m.cml", "f.csv")
            self.assertEqual(r.first_rule(), "ml.input.non-numeric", (cell, r))
            self.assertEqual(r.status, 8)

    def test_a_model_of_ints_and_decimals_is_byte_for_byte_what_the_cli_slice_wrote(self):
        # docs/cli.md section 2.4: this exact file, this exact command, written by the commit before the float reader
        self.sb.write("sales.csv", "id,x,y\n1,2.5,10\n2,3.5,20\n3,1.0,30\n4,0.25,41\n")
        r = self.sb.run("fit", "--target", "y", "--output", "sales.cml", "sales.csv")
        # the file now ends with the provenance section; without it the bytes are those of the commit before the float reader
        old = without_provenance(self.sb.read("sales.cml"))
        self.assertEqual(len(old), 316)
        self.assertEqual(hashlib.sha256(old).hexdigest(), "9b6a34144d1351f141f1ed2f1238f053afef57d815f29d047892239b63afbd05")
        self.assertEqual(r.data()["bytes"], 316 + 76)

    def test_the_unsupported_type_rule_is_gone(self):
        self.assertNotIn("ml.input.unsupported-type", [r["rule"] for r in introspect()["rules"]])
        self.assertNotIn("ml.input.unsupported-csv", [r["rule"] for r in introspect()["rules"]])

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

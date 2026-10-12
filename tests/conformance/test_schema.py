"""S1 -- a prediction input whose columns differ from the fitted schema is ml.schema.mismatch, with the missing, extra and
mistyped columns named in `detail` and a repair that names them; column order may differ; names are the identity."""

import unittest

from harness import Sandbox, run

REG = "id,x,y\n1,2.5,10\n2,3.5,20\n3,1.0,30\n4,0.25,41\n"


class Schema(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        self.sb.write("reg.csv", REG)
        self.assertEqual(self.sb.run("fit", "--target", "y", "--output", "m.cml", "reg.csv").status, 0)

    def tearDown(self):
        self.sb.cleanup()

    def predict(self, name, text, *extra, verb="predict"):
        self.sb.write(name, text)
        return self.sb.run(verb, "--model-file", "m.cml", *extra, name)

    def test_column_order_may_differ(self):
        r = self.predict("o.csv", "y,x,id\n10,2.5,1\n20,3.5,2\n")
        self.assertEqual(r.status, 0, r)
        self.assertEqual(r.data()["row_count"], 2)
        r = self.predict("o2.csv", "x,y,id\n2.5,10,1\n", verb="eval")
        self.assertEqual(r.status, 0, r)

    def test_missing_extra_and_mistyped_are_named(self):
        r = self.predict("a.csv", "id,q,z\n1,2,3\n")
        self.assertEqual(r.first_rule(), "ml.schema.mismatch")
        d = r.error()["detail"]
        self.assertEqual(d["missing"], ["x"])
        self.assertEqual(d["extra"], ["q", "z"])
        self.assertEqual(d["mistyped"], [])
        reason = r.error()["repair"]["reason"]
        self.assertIn("x", reason)
        self.assertIn("q", reason)
        self.assertIn("z", reason)
        self.assertEqual(r.error()["repair"]["kind"], "none")
        r = self.predict("b.csv", "id,x\n1.5,3.125\n")
        d = r.error()["detail"]
        self.assertEqual([m["name"] for m in d["mistyped"]], ["id", "x"])
        self.assertEqual([(m["expected"], m["found"]) for m in d["mistyped"]], [("int", "dec(1)"), ("dec(2)", "dec(3)")])
        self.assertIn("id", r.error()["repair"]["reason"])
        self.assertIn("x", r.error()["repair"]["reason"])

    def test_extras_alone_have_a_retry_that_works(self):
        r = self.predict("e.csv", "id,x,y,junk,more\n1,2.5,10,a,b\n".replace(",a,b", ",1,2"))
        self.assertEqual(r.first_rule(), "ml.schema.mismatch")
        rep = r.error()["repair"]
        self.assertEqual(rep["kind"], "retry")
        self.assertEqual(rep["argv"][-2:], ["--drop", "junk,more"])
        again = run(*rep["argv"][1:], cwd=self.sb.dir)
        self.assertEqual(again.status, 0, again)

    def test_the_target_is_needed_by_eval_not_by_predict(self):
        self.assertEqual(self.predict("n.csv", "id,x\n1,2.5\n").status, 0)
        r = self.predict("n2.csv", "id,x\n1,2.5\n", verb="eval")
        self.assertEqual(r.first_rule(), "ml.schema.mismatch")
        self.assertEqual(r.error()["detail"]["missing"], ["y"])

    def test_the_target_column_of_predict_is_not_read(self):
        r = self.predict("t.csv", "id,x,y\n1,2.5,not a number\n2,3,\n")
        self.assertEqual(r.status, 0, r)

    def test_dropped_columns_are_not_part_of_the_schema(self):
        self.assertEqual(self.sb.run("fit", "--drop", "id", "--target", "y", "--output", "d.cml", "reg.csv").status, 0)
        r = self.sb.run("predict", "--model-file", "d.cml", "reg.csv")
        self.assertEqual(r.first_rule(), "ml.schema.mismatch")
        self.assertEqual(r.error()["detail"]["extra"], ["id"])
        self.assertEqual(r.error()["repair"]["argv"][-2:], ["--drop", "id"])
        r = self.sb.run("predict", "--model-file", "d.cml", "--drop", "id", "reg.csv")
        self.assertEqual(r.status, 0, r)

    def test_dropping_a_needed_column_is_a_mismatch(self):
        r = self.sb.run("predict", "--model-file", "m.cml", "--drop", "x", "reg.csv")
        self.assertEqual(r.first_rule(), "ml.schema.mismatch")
        self.assertEqual(r.error()["detail"]["missing"], ["x"])

    def test_names_are_exact(self):
        r = self.predict("c.csv", "ID,x\n1,2.5\n")
        self.assertEqual(r.first_rule(), "ml.schema.mismatch")
        self.assertEqual(r.error()["detail"]["missing"], ["id"])
        self.assertEqual(r.error()["detail"]["extra"], ["ID"])

    def test_a_smaller_type_fits_a_larger_one(self):
        # the model's x is dec(2); integers and shorter decimals are read as dec(2)
        self.assertEqual(self.predict("s.csv", "id,x\n1,2\n2,3.5\n").status, 0)

    def test_non_utf8_names_are_still_a_datum(self):
        self.sb.write("u.csv", b"id,x\xff\n1,2\n")
        r = self.sb.run("predict", "--model-file", "m.cml", "u.csv")
        self.assertEqual(r.first_rule(), "ml.schema.mismatch")
        self.assertEqual(r.error()["detail"]["extra"], [{"b64": "eP8="}])
        self.assertEqual(r.error()["repair"]["kind"], "none")


if __name__ == "__main__":
    unittest.main()

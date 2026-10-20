"""I9 -- `GET /model` of `ml-serve` carries the `importance` member of `describe` (docs/interpret.md section 6, docs/serving-http.md): the bytes of the object the server returns are
the bytes `ml describe` prints for the same file, for a linear model, a forest and a k-means (null)."""

import json
import os
import sys
import unittest

from harness import ROOT, Sandbox

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_interp as O  # noqa: E402
import serve_lib  # noqa: E402


class ServedImportance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        sb = cls.sb
        sb.write("reg.csv", O.to_csv(*O.gen_dataset("reg", 200, 1)))
        steps = "scale column=* kind=standard"
        assert sb.run("fit", "--model", "linear", "--target", "y", "--step", steps, "--output", "lin.ml", "reg.csv").status == 0
        assert sb.run("fit", "--model", "forest", "--task", "regression", "--trees", "5", "--max-depth", "4", "--seed", "2", "--target", "y", "--output", "rf.ml", "reg.csv").status == 0
        assert sb.run("fit", "--model", "kmeans", "--k", "2", "--drop", "y", "--output", "km.ml", "reg.csv").status == 0

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def served(self, model):
        with serve_lib.Server(self.sb.dir, model) as s:
            r = s.request("GET", "/model")
            self.assertEqual(r.status, 200, r.text())
            return json.loads(r.text())

    def test_the_served_description_has_the_table_of_describe(self):
        for model in ("lin.ml", "rf.ml", "km.ml"):
            with self.subTest(model=model):
                d = self.sb.run("describe", "--model-file", model).data()
                got = self.served(model)
                self.assertIn("importance", got, got)
                self.assertEqual(got["importance"], d["importance"])
        self.assertIsNone(self.served("km.ml")["importance"])
        self.assertTrue(self.served("rf.ml")["importance"]["available"])

    def test_the_openapi_document_names_the_member(self):
        with serve_lib.Server(self.sb.dir, "lin.ml") as s:
            r = s.request("GET", "/openapi.json")
            self.assertEqual(r.status, 200)
            self.assertIn('"importance"', r.text())


if __name__ == "__main__":
    unittest.main()

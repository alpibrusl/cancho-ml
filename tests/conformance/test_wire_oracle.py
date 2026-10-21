"""W3, W6, W7 -- the oracle (scripts/oracle_wire.py) fits the same pipeline and model in Python, in the same summation order, and its model file is compared with
the one the real binary wrote BYTE FOR BYTE, not within a tolerance: linear and logistic, on seeded data with missing cells, constant columns and imputed, min-max and
standardised columns, a seeded split, a positive label and, in about two cases in five, a weight column (`--weight`, docs/weights.md: not a feature, not in the schema of the file). `predict` (on other rows, the columns in another order) and `eval` equal the oracle's doubles."""

import pathlib
import sys
import tempfile
import unittest

from harness import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_wire as ow  # noqa: E402

CASES = 60


class Oracle(unittest.TestCase):
    def suite(self, model, offset):
        bad = []
        judged = 0
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            for seed in range(1, CASES + 1):
                case = ow.gen_case(seed * 7919 + offset, model, weighted=True)
                before = len(ow.UNJUDGED)
                pr = ow.run_case(case, tmp, threads=1 + seed % 4)
                judged += len(ow.UNJUDGED) == before
                if pr:
                    bad.append((model, seed, pr))
        self.assertEqual(bad, [])
        self.assertGreaterEqual(judged, CASES * 3 // 4, "too many cases the oracle could not judge")

    def test_linear_bytes_predictions_and_metrics(self):
        self.suite("linear", 0)

    def test_logistic_bytes_predictions_and_metrics(self):
        self.suite("logistic", 101)

    def test_the_cases_reach_what_the_task_asks_for(self):
        # missing cells, a constant column, imputation, min-max and standard scaling, a bucket step, a one-hot encoding, a split, a positive label: all appear in the generated cases
        seen = set()
        for model, off in (("linear", 0), ("logistic", 101)):
            for seed in range(1, CASES + 1):
                c = ow.gen_case(seed * 7919 + off, model, weighted=True)
                if any("" in r for r in c.rows):
                    seen.add("missing")
                for st in c.steps:
                    seen.add({1: "impute", 2: "minmax", 3: "standard", 4: "onehot", 6: "bucket"}[st["kind"]])
                if c.split:
                    seen.add("split")
                if c.positive is not None:
                    seen.add("positive")
                if c.weight:
                    seen.add("weight")
                cols = list(zip(*[r[:-1] for r in c.rows]))
                if any(len(set(col) - {""}) == 1 for col in cols):
                    seen.add("constant column")
        self.assertEqual(seen, {"missing", "impute", "minmax", "standard", "onehot", "bucket", "split", "positive", "constant column", "weight"})


if __name__ == "__main__":
    unittest.main()

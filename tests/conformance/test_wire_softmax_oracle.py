"""M3 (the wiring half), M5 -- the oracle (scripts/oracle_wire.py) fits the same pipeline and the softmax model in Python, in the same summation order as the library, and its model file
is compared with the one the real binary wrote BYTE FOR BYTE, not within a tolerance: seeded data of 3, 4, 5 and 7 classes with missing cells, constant columns and imputed, min-max and
standardised columns, a seeded split, bucket and one-hot steps. `predict` (labels and probabilities, on other rows with the columns in another order) equals the oracle's doubles, and
every number of `eval` (the confusion matrix, the per-class lines, the averages, the one-versus-rest AUC and the log-loss) equals the oracle's, or is within a bound fixed in advance
for the log-loss. A case the oracle expects to be refused (a class with no training row) is refused with the oracle's tag."""

import pathlib
import sys
import tempfile
import unittest

from harness import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_wire as ow  # noqa: E402

CASES = 40
OFFSET = 202


class SoftmaxOracle(unittest.TestCase):
    def test_file_predictions_and_metrics_equal_the_oracles(self):
        bad = []
        judged = 0
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            for seed in range(1, CASES + 1):
                case = ow.gen_case(seed * 7919 + OFFSET, "softmax")
                before = len(ow.UNJUDGED)
                pr = ow.run_case(case, tmp, threads=1 + seed % 4)
                judged += len(ow.UNJUDGED) == before
                if pr:
                    bad.append((seed, pr))
        self.assertEqual(bad, [])
        self.assertGreaterEqual(judged, CASES * 3 // 4, "too many cases the oracle could not judge")

    def test_the_cases_reach_what_the_task_asks_for(self):
        seen = set()
        classes = set()
        outcomes = {"fit": 0, "refused": 0}
        for seed in range(1, CASES + 1):
            c = ow.gen_case(seed * 7919 + OFFSET, "softmax")
            if any("" in r for r in c.rows):
                seen.add("missing")
            for st in c.steps:
                seen.add({1: "impute", 2: "minmax", 3: "standard", 4: "onehot", 6: "bucket"}[st["kind"]])
            if c.split:
                seen.add("split")
            if float(c.l2) == 0.0:
                seen.add("l2 zero")
            else:
                seen.add("l2 positive")
            classes.add(len({r[-1] for r in c.rows}))
            try:
                expect, ctx = ow.fit_oracle(c)
            except AssertionError:
                continue
            outcomes["fit" if ctx is not None else "refused"] += 1
        self.assertTrue({"missing", "impute", "split", "l2 zero", "l2 positive", "minmax", "standard"} <= seen, seen)
        self.assertGreaterEqual(len(classes), 3, classes)
        self.assertGreaterEqual(outcomes["fit"], CASES // 2, outcomes)


if __name__ == "__main__":
    unittest.main()

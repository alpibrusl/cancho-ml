"""WT12 -- the model files of linear and logistic do not change (docs/wiring-trees.md section 1).

`golden_planned.json` was written by the binary of the parent commit `d481593` (before the tree and the forest were wired) with scripts/golden_planned.py: for 60
seeded cases of the wiring oracle, the exit status, the SHA-256 of the answer of `fit` and the SHA-256 of the model file. This binary reproduces all of it. The
record is never rewritten to make a test pass.
"""

import json
import pathlib
import sys
import tempfile
import unittest

from harness import BIN, ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import golden_planned  # noqa: E402

GOLDEN = json.loads((pathlib.Path(__file__).parent / "golden_planned.json").read_text())

# Issue 48 (docs/refusals-fit.md): the two logistic cases whose classes a hyperplane separates (the label is the sign of f0), fitted with l2 = 0. The record is not rewritten; these
# two answers are pinned here, with the SHA-256 of the new standard output (the binary and the sandbox replaced by @ML and @ROOT).
#   logistic-5  : 25 iterations. Before: exit 0 and a model with coefficients that grow with every iteration. Now: exit 8, ml.fit.separable (the loss of pass 2 is under ln 2), no file.
#   logistic-14 : 100 iterations. Before: exit 8, ml.linalg.not-positive-definite with an empty detail, late. Now: exit 8, ml.fit.separable, with the pass and the repair.
CHANGED = {
    "logistic-5": {"status": 8, "stdout": "a8b782215d8e04a802593881477701759227022d94ac2118ddebd7b6ddfd8eb5", "file": None},
    "logistic-14": {"status": 8, "stdout": "7bcc58909245c6c46b6f4def2e298dda8fb2145c8993c52d491a64f8f3110a65", "file": None},
}


class PlannedBaseline(unittest.TestCase):
    def test_the_record_covers_the_cases(self):
        self.assertEqual(sorted(GOLDEN), sorted(n for n, _ in golden_planned.cases()))
        self.assertEqual(len(GOLDEN), 60)
        self.assertGreaterEqual(sum(1 for g in GOLDEN.values() if g["file"]), 50)

    def test_the_two_changes_of_issue_48_differ_from_the_record_and_nothing_else_does(self):
        self.assertEqual(sorted(CHANGED), ["logistic-14", "logistic-5"])
        self.assertEqual(GOLDEN["logistic-5"]["status"], 0)
        self.assertIsNotNone(GOLDEN["logistic-5"]["file"])
        for k, v in CHANGED.items():
            self.assertNotEqual(GOLDEN[k], v)

    def test_every_model_file_of_linear_and_logistic_is_byte_for_byte_the_old_one(self):
        golden_planned.ow.BIN = BIN
        failures = []
        with tempfile.TemporaryDirectory() as td:
            for name, case in golden_planned.cases():
                got = golden_planned.record(case, pathlib.Path(td))
                if got != CHANGED.get(name, GOLDEN[name]):
                    failures.append((name, GOLDEN[name], got))
        self.assertEqual(failures, [], "a linear or logistic file or answer changed")


if __name__ == "__main__":
    unittest.main()

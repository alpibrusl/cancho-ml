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


class PlannedBaseline(unittest.TestCase):
    def test_the_record_covers_the_cases(self):
        self.assertEqual(sorted(GOLDEN), sorted(n for n, _ in golden_planned.cases()))
        self.assertEqual(len(GOLDEN), 60)
        self.assertGreaterEqual(sum(1 for g in GOLDEN.values() if g["file"]), 50)

    def test_every_model_file_of_linear_and_logistic_is_byte_for_byte_the_old_one(self):
        golden_planned.ow.BIN = BIN
        failures = []
        with tempfile.TemporaryDirectory() as td:
            for name, case in golden_planned.cases():
                got = golden_planned.record(case, pathlib.Path(td))
                if got != GOLDEN[name]:
                    failures.append((name, GOLDEN[name], got))
        self.assertEqual(failures, [], "a linear or logistic file or answer changed")


if __name__ == "__main__":
    unittest.main()

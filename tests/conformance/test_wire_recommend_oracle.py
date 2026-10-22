"""WR2, WR3, WR5, WR6, WR11 -- the oracle (scripts/oracle_wire_recommend.py) reads the same log and fits the same ALS in Python and writes the whole model file: header, plan, schema (no
target), the empty steps, the payload of docs/wiring-recommend.md section 5, meta with the sum order 3 and the SHA-256 trailer. The file `ml fit --model recommend` wrote is compared with
it BYTE FOR BYTE on seeded logs: explicit and implicit, with and without a rating column, repeated pairs of the implicit form, ids that look like numbers and ids that need quoting, factors
from 1 to 6, the held-out split. `describe`, `predict --for` (with and without the history), `predict --like` and `eval` equal the oracle's lists and doubles."""

import unittest

from harness import ROOT

import sys

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_wire_recommend as owr  # noqa: E402

CASES = 40


class RecommendOracle(unittest.TestCase):
    def test_the_oracle_fails_where_it_should(self):
        self.assertEqual(owr.controls(), [])

    def test_the_files_and_the_answers_equal_the_oracle(self):
        bad = []
        for seed in range(1, CASES + 1):
            bad += owr.check_case(seed)
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()

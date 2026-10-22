"""RK4 -- the oracle (scripts/oracle_wire_rank.py, over scripts/oracle_boost.py and oracle_rank.py) fits the same boosted ranking model in Python and writes the whole model file; the file
`ml fit --task ranking --group qid` wrote is compared with it BYTE FOR BYTE, on seeded files with groups whose rows are not adjacent, missing cells, steps, a row sample and weights;
`predict` equals the oracle's scores, `predict --rank` its ranks, and `eval --group` every metric of docs/ranking.md section 3 for four cut-offs and both gains."""

import pathlib
import tempfile
import unittest

from harness import ROOT
import sys

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_wire_rank as owr  # noqa: E402
import oracle_wire_boost as owb  # noqa: E402

CASES = 24


class RankOracle(unittest.TestCase):
    def suite(self, weighted, lam=False):
        bad = []
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            for seed in range(1, CASES + 1):
                case = owr.gen_case(seed * 7919 + 211, weighted=weighted)
                if lam:
                    case.task = "lambdarank"
                pr = owb.run_case(case, tmp, threads=1 + seed % 4)
                if not pr and (tmp / "m.cml").exists():
                    for k, gain in ((10, "exp"), (3, "linear"), (0, "exp"), (1, "exp")):
                        pr += owr.check_eval(case, tmp, k, gain)
                    pr += owr.check_rank_column(case, tmp)
                (tmp / "m.cml").unlink(missing_ok=True)
                if pr:
                    bad.append((seed, pr))
        self.assertEqual(bad, [])

    def test_ranking(self):
        self.suite(False)

    def test_weighted_ranking(self):
        self.suite(True)

    def test_lambdarank(self):
        self.suite(False, True)

    def test_weighted_lambdarank(self):
        self.suite(True, True)

    def test_the_weights_reach_the_file(self):
        import copy
        differ = 0
        # about one case in five has no relevance above 0 (a ceiling of 0): no pair, no gradient, and the weights cannot matter there
        for seed in range(1, 17):
            case = owr.gen_case(seed * 7919 + 211, weighted=True)
            plain = copy.deepcopy(case)
            plain.weights = None
            a, _ = owb.fit_oracle(case)
            b, _ = owb.fit_oracle(plain)
            differ += 1 if a != b else 0
        self.assertGreaterEqual(differ, 10)


if __name__ == "__main__":
    unittest.main()

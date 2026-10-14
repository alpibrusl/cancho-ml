"""WK4, WK5, WK6 -- the oracle (scripts/oracle_wire_kmeans.py) fits the same pipeline and the same k-means in Python and writes the whole model file: header, schema (no
target), stored steps, the payload of docs/models-linear.md section 3.4, meta with the sum order 2 and the SHA-256 trailer. The file `ml` wrote is compared with it BYTE FOR
BYTE, for both initialisations, k from 1 to 12, on seeded data with missing cells imputed, constant columns, scaled and bucketed columns, a label column that the fit drops and
a seeded split. `fit` (the metrics of the training rows and of the held-out rows), `predict` (clusters and squared distances, on other rows with the columns in another order)
and `eval` (the inertia, the sizes and the mean distances, with and without a split) equal the oracle's doubles. The cases with a missing cell the plan leaves are refusals the
oracle predicts. The 100,000-row cases are scripts/oracle_wire_kmeans.py --large (admit.py Q9 and the results of docs/wiring-kmeans.md section 12)."""

import hashlib
import pathlib
import sys
import tempfile
import unittest

from harness import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_wire_kmeans as owk  # noqa: E402

CASES = 40


class KMeansOracle(unittest.TestCase):
    def suite(self, init):
        bad = []
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            for seed in range(1, CASES + 1):
                case = owk.gen_case(seed * 7919 + (101 if init == "random" else 0))
                case.init = init
                case.init_flag = True
                pr = owk.run_case(case, tmp, threads=1 + seed % 4)
                if pr:
                    bad.append((init, seed, pr))
        self.assertEqual(bad, [])

    def test_kmeans_plus_plus(self):
        self.suite("kmeans++")

    def test_random_initialisation(self):
        self.suite("random")

    def test_the_cases_reach_what_the_gate_asks_for(self):
        seen = set()
        ks = set()
        for init in owk.INITS:
            for seed in range(1, CASES + 1):
                c = owk.gen_case(seed * 7919 + (101 if init == "random" else 0))
                ks.add(c.k)
                kinds = {st["kind"] for st in c.steps}
                cols = list(zip(*c.rows))
                if any("" in col for col in cols) and 1 in kinds:
                    seen.add("missing imputed")
                if any("" in col for col in cols) and 1 not in kinds:
                    seen.add("missing left (a refusal)")
                if any(len(set(col) - {""}) == 1 for col in cols):
                    seen.add("constant column")
                seen |= {{1: "impute", 2: "minmax", 3: "standard", 6: "bucket"}[k] for k in kinds}
                if c.split:
                    seen.add("split")
                if c.extra is not None:
                    seen.add("label dropped")
                if c.iters != 10:
                    seen.add("iterations")
        self.assertEqual(seen, {"missing imputed", "missing left (a refusal)", "constant column", "impute", "minmax", "standard", "bucket", "split", "label dropped", "iterations"})
        self.assertGreaterEqual(len(ks), 10)

    def test_the_oracle_is_sensitive_to_the_sum_order_field_and_to_the_target_index(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            case = owk.gen_case(7919)
            (tmp / "d.csv").write_text(case.csv())
            expect, ctx = owk.fit_oracle(case)
            self.assertIsNotNone(ctx)
            rc, out = owk.ow.ml("--root", tmp, *owk.fit_args(case, "m.cml"), "d.csv")
            self.assertEqual(rc, 0, out)
            self.assertEqual((tmp / "m.cml").read_bytes(), expect)
            body = bytearray(expect[:-32])
            body[-4:] = (3).to_bytes(4, "little")
            self.assertNotEqual(bytes(body) + hashlib.sha256(bytes(body)).digest(), (tmp / "m.cml").read_bytes())


if __name__ == "__main__":
    unittest.main()

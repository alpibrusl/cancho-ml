"""WT4, WT5 -- the oracle (scripts/oracle_wire_tree.py) fits the same pipeline and the same tree or forest in Python and writes the whole model file: header, schema,
stored steps, the payload of docs/models-tree.md section 7, the labels record, meta with the sum order 3 and the SHA-256 trailer. The file `ml` wrote is compared with it
BYTE FOR BYTE, for both models and both tasks, on seeded data with missing cells (kept or imputed), columns without a single cell, constant columns, scaled, bucketed
and one-hot columns, a seeded split and a positive label. `predict`, `predict --proba` (on other rows, the columns in another order) and every `eval` metric equal the
oracle's doubles. The 100,000-row cases are scripts/oracle_wire_tree.py --large (admit.py Q9 and the results of docs/wiring-trees.md section 11)."""

import pathlib
import sys
import tempfile
import unittest

from harness import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_wire_tree as owt  # noqa: E402

CASES = 30


class TreeOracle(unittest.TestCase):
    def suite(self, model, task, offset):
        bad = []
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            for seed in range(1, CASES + 1):
                case = owt.gen_case(seed * 7919 + offset, model, task)
                pr = owt.run_case(case, tmp, threads=1 + seed % 4)
                if pr:
                    bad.append((model, task, seed, pr))
        self.assertEqual(bad, [])

    def test_tree_classification(self):
        self.suite("tree", "classification", 0)

    def test_tree_regression(self):
        self.suite("tree", "regression", 101)

    def test_forest_classification(self):
        self.suite("forest", "classification", 31)

    def test_forest_regression(self):
        self.suite("forest", "regression", 132)

    def test_the_cases_reach_what_the_task_asks_for(self):
        # missing cells kept and imputed, a column with no cell, a constant column, impute, min-max, standard, bucket and one-hot steps, a split, a positive label, bootstrap off,
        # mtry given, a depth and a bin count given: all appear in the generated cases
        seen = set()
        for model in ("tree", "forest"):
            for task in ("classification", "regression"):
                for seed in range(1, CASES + 1):
                    c = owt.gen_case(seed * 7919 + (101 if task == "regression" else 0) + (31 if model == "forest" else 0), model, task)
                    cols = list(zip(*[r[:-1] for r in c.rows]))
                    kinds = {st["kind"] for st in c.steps}
                    if any("" in col for col in cols) and 1 not in kinds:
                        seen.add("missing kept")
                    if any("" in col for col in cols) and 1 in kinds:
                        seen.add("missing imputed")
                    if any(set(col) == {""} for col in cols):
                        seen.add("empty column")
                    if any(len(set(col) - {""}) == 1 for col in cols):
                        seen.add("constant column")
                    seen |= {{1: "impute", 2: "minmax", 3: "standard", 4: "onehot", 6: "bucket"}[k] for k in kinds}
                    if c.split:
                        seen.add("split")
                    if c.positive is not None:
                        seen.add("positive")
                    if c.hp.get("bootstrap") == 0:
                        seen.add("no bootstrap")
                    if "mtry" in c.hp:
                        seen.add("mtry")
                    if "max-depth" in c.hp and "bins" in c.hp and "min-leaf" in c.hp:
                        seen.add("depth bins leaf")
        self.assertEqual(seen, {"missing kept", "missing imputed", "empty column", "constant column", "impute", "minmax", "standard", "onehot", "bucket", "split", "positive",
                                "no bootstrap", "mtry", "depth bins leaf"})

    def test_the_oracle_is_sensitive_to_the_sum_order_field(self):
        # the file of ml equals the oracle's; a file that says 2 where the oracle says 3 does not: the comparison can fail
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            case = owt.gen_case(7919, "forest", "regression")
            (tmp / "d.csv").write_text(case.csv())
            expect, ctx = owt.fit_oracle(case)
            self.assertIsNotNone(ctx)
            rc, out = owt.ow.ml("--root", tmp, *owt.fit_args(case, "m.cml"), "d.csv")
            self.assertEqual(rc, 0, out)
            self.assertEqual((tmp / "m.cml").read_bytes(), expect)
            body = bytearray(expect[:-32])
            body[-4:] = (2).to_bytes(4, "little")
            changed = bytes(body) + hashlib.sha256(bytes(body)).digest()
            self.assertNotEqual(changed, (tmp / "m.cml").read_bytes())


if __name__ == "__main__":
    unittest.main()

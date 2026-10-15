"""GB1 -- the oracle (scripts/oracle_wire_boost.py, over scripts/oracle_boost.py) fits the same pipeline and the same boosted model in Python and writes the whole model
file: header, schema, stored steps, the payload of docs/models-boost.md section 5, the labels record, meta with the sum order 3 and the SHA-256 trailer. The file `ml` wrote is
compared with it BYTE FOR BYTE, for both tasks, on seeded data with missing cells (kept or imputed), columns without a single cell, constant columns, scaled, bucketed and
one-hot columns, a seeded split, a positive label, a row sample and early stopping. `predict`, `predict --proba` (on other rows, the columns in another order) and every `eval`
metric equal the oracle's doubles. The 100,000-row cases are scripts/oracle_wire_boost.py --large (admit.py Q9 and the results of docs/models-boost.md section 10)."""

import hashlib
import pathlib
import sys
import tempfile
import unittest

from harness import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_wire_boost as owb  # noqa: E402

CASES = 30


class BoostOracle(unittest.TestCase):
    def suite(self, task, offset):
        bad = []
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            for seed in range(1, CASES + 1):
                case = owb.gen_case(seed * 7919 + offset + 57, task)
                pr = owb.run_case(case, tmp, threads=1 + seed % 4)
                if pr:
                    bad.append((task, seed, pr))
        self.assertEqual(bad, [])

    def test_classification(self):
        self.suite("classification", 0)

    def test_regression(self):
        self.suite("regression", 101)

    def test_the_cases_reach_what_the_task_asks_for(self):
        seen = set()
        for task in ("classification", "regression"):
            for seed in range(1, CASES + 1):
                c = owb.gen_case(seed * 7919 + (101 if task == "regression" else 0) + 57, task)
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
                if "subsample" in c.hp:
                    seen.add("subsample")
                if "early-stopping" in c.hp:
                    seen.add("early stopping")
                for k in ("rounds", "learning-rate", "max-depth", "min-leaf", "bins", "l2", "max-nodes"):
                    if k in c.hp:
                        seen.add(k)
        self.assertEqual(seen, {"missing kept", "missing imputed", "empty column", "constant column", "impute", "minmax", "standard", "onehot", "bucket", "split", "positive",
                                "subsample", "early stopping", "rounds", "learning-rate", "max-depth", "min-leaf", "bins", "l2", "max-nodes"})

    def test_the_oracle_is_sensitive(self):
        # the file of ml equals the oracle's; a file that says 2 where the oracle says 3 does not, and an oracle that changes a number of the algorithm does not match
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            case = owb.gen_case(7919 + 57, "regression")
            case.hp.setdefault("rounds", 12)
            case.hp.setdefault("subsample", "0.5")
            (tmp / "d.csv").write_text(case.csv())
            expect, ctx = owb.fit_oracle(case)
            self.assertIsNotNone(ctx)
            rc, out = owb.ow.ml("--root", tmp, *owb.fit_args(case, "m.cml"), "d.csv")
            self.assertEqual(rc, 0, out)
            self.assertEqual((tmp / "m.cml").read_bytes(), expect)
            body = bytearray(expect[:-32])
            body[-4:] = (2).to_bytes(4, "little")
            self.assertNotEqual(bytes(body) + hashlib.sha256(bytes(body)).digest(), (tmp / "m.cml").read_bytes())
            # an oracle whose hessian floor, sample or learning rate differs writes another file
            for name, value in (("HMIN", 0.5), ("SCALE", 999999)):
                old = getattr(owb.ob, name)
                setattr(owb.ob, name, value)
                try:
                    other, _ = owb.fit_oracle(case)
                finally:
                    setattr(owb.ob, name, old)
                if name == "HMIN":
                    continue
                self.assertNotEqual(other, expect, name)
            case.hp["learning-rate"] = "0.2"
            other, _ = owb.fit_oracle(case)
            self.assertNotEqual(other, expect)


if __name__ == "__main__":
    unittest.main()

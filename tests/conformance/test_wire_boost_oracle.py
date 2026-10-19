"""GB1 -- the oracle (scripts/oracle_wire_boost.py, over scripts/oracle_boost.py) fits the same pipeline and the same boosted model in Python and writes the whole model
file: header, schema, stored steps, the payload of docs/models-boost.md section 5, the labels record, meta with the sum order 3 and the SHA-256 trailer. The file `ml` wrote is
compared with it BYTE FOR BYTE, for both tasks, on seeded data with missing cells (kept or imputed), columns without a single cell, constant columns, scaled, bucketed and
one-hot columns, a seeded split, a positive label, a row sample and early stopping. `predict`, `predict --proba` (on other rows, the columns in another order) and every `eval`
metric equal the oracle's doubles. The 100,000-row cases are scripts/oracle_wire_boost.py --large (admit.py Q9 and the results of docs/models-boost.md section 10)."""

import hashlib
import pathlib
import random
import sys
import tempfile
import unittest

import numpy as np

from harness import ROOT, Sandbox
from test_wire_boost import model_section, without_provenance

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

    def test_the_sample_boundary_and_the_rounding_of_K(self):
        # a draw equal to K is not in the sample (the comparison is strict), and K is the rounded product, not the truncated one: cases built so that both decide a byte
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            case = owb.gen_case(7919 + 57, "regression")
            case.hp.update({"rounds": 6, "min-leaf": 2})
            case.split, case.seed = None, 5
            case.hp.pop("early-stopping", None)
            n = len(case.rows)
            s0 = owb.ob.derive(owb.ob.derive(5, 9), 0)
            z = owb.ob.mix(s0 + owb.ob.GOLD)                                           # the first draw of round 0, for the first fit row
            v = (z >> 1) % 1000000
            for sub_ in ("%.6f" % (max(v, 1) / 1000000.0), "0.7000007"):
                case.hp["subsample"] = sub_
                expect, ctx = owb.fit_oracle(case)
                self.assertIsNotNone(ctx)
                (tmp / "d.csv").write_text(case.csv())
                (tmp / "m.cml").unlink(missing_ok=True)
                rc, out = owb.ow.ml("--root", tmp, *owb.fit_args(case, "m.cml"), "d.csv")
                self.assertEqual(rc, 0, out)
                self.assertEqual((tmp / "m.cml").read_bytes(), expect, sub_)
            self.assertEqual(ctx["res"]["subk"], 700001)
            self.assertGreater(n, 0)

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


class BoostSplitLeakage(unittest.TestCase):
    """docs/models-boost.md section 2.8: the validation rows of early stopping come from the training rows of the split, never from its test rows"""

    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def fit(self, target, csv, out, *extra):
        r = self.sb.run("fit", "--model", "boost", "--target", target, "--output", out, "--overwrite", *extra, csv)
        self.assertEqual(r.status, 0, r)
        return r

    def test_early_stopping_never_reads_the_held_out_rows(self):
        # the rows --split train=0.8 holds out are the test rows; changing their target must not change the model file (the validation rows come from the train side)
        import sys
        from harness import ROOT
        sys.path.insert(0, str(ROOT / "scripts"))
        import oracle_wire as ow
        rng = random.Random(6)
        rows = [["%.3f" % rng.uniform(0, 5), "%.3f" % rng.uniform(0, 5)] for _ in range(200)]
        ys = ["%.3f" % (float(a) + rng.gauss(0, 3)) for a, b in rows]
        mask = ow.split_masks(200, "train=0.8", 7)
        self.assertEqual(sum(mask), 40)
        ys2 = [("%.3f" % rng.uniform(-100, 100)) if mask[i] == 1 else ys[i] for i in range(200)]
        for name, yv in (("h1.csv", ys), ("h2.csv", ys2)):
            self.sb.write(name, "a,b,y\n" + "".join("%s,%s,%s\n" % (r_[0], r_[1], y) for r_, y in zip(rows, yv)))
        for name, out in (("h1.csv", "h1.cml"), ("h2.csv", "h2.cml")):
            self.fit("y", name, out, "--rounds", "100", "--early-stopping", "4", "--split", "train=0.8", "--seed", "7", "--min-leaf", "4")
        # the file also holds the provenance (the SHA-256 of the training file, which differs): the model, the steps and the plan are what must not
        self.assertEqual(without_provenance(self.sb.read("h1.cml")), without_provenance(self.sb.read("h2.cml")))
        # and the held-out metrics are the ones eval --split gives on the file
        a = self.sb.run("eval", "--model-file", "h2.cml", "--split", "train=0.8", "--seed", "7", "h2.csv").data()["metrics"]
        r = self.fit("y", "h2.csv", "h3.cml", "--rounds", "100", "--early-stopping", "4", "--split", "train=0.8", "--seed", "7", "--min-leaf", "4")
        self.assertEqual(r.data()["summary"]["holdout"]["metrics"], a)

    def test_the_validation_rows_of_a_stratified_split_never_come_from_the_test_rows(self):
        # the oracle fits ONLY the rows the split kept (the assignment `--emit-folds` prints): its validation tenth is drawn from them alone, and the file of ml holds the same
        # model section byte for byte, so no validation row was taken from the test rows; a plain holdout is the same, with the masks of the oracle
        ob, ow, owt = owb.ob, owb.ow, owb.owt
        rng = random.Random(8)
        n = 300
        rows = [(round(rng.uniform(0, 5), 3), round(rng.uniform(0, 5), 3)) for _ in range(n)]
        ys = [1 if a + rng.gauss(0, 1.2) > 2.5 else 0 for a, b in rows]
        self.sb.write("st.csv", "a,b,lab\n" + "".join("%s,%s,%d\n" % (a, b, y) for (a, b), y in zip(rows, ys)))
        args = ["--rounds", "100", "--early-stopping", "4", "--seed", "7", "--min-leaf", "4", "--task", "classification"]
        for label, extra, mask_of in (
                ("stratified", ["--split", "train=0.8", "--stratify", "--emit-folds"], None),
                ("plain", ["--split", "train=0.8"], lambda: ow.split_masks(n, "train=0.8", 7))):
            r = self.fit("lab", "st.csv", "st.cml", *args, *extra)
            if mask_of is None:
                mask = r.data()["summary"]["split"]["assignment"]
            else:
                mask = mask_of()
            tr = [i for i in range(n) if mask[i] == 0]
            self.assertGreater(len(tr), 200, label)
            X = np.array([rows[i] for i in tr], dtype=np.float64)
            res = ob.fit(X, np.zeros(X.shape, dtype=bool), np.array([float(ys[i]) for i in tr]), 0, 100, 0.1, 4, 4, 64, 1.0, 1.0, 4, 4194304, 7)
            self.assertLess(res["kept"], 100, label)                       # early stopping ended the fit: the validation rows decided
            expect = ob.payload(res, 2) + owt.labels_record(0, 1)
            self.assertEqual(model_section(self.sb.read("st.cml")), expect, label)


if __name__ == "__main__":
    unittest.main()
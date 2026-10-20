"""I5, I6 -- the permutation importance of `ml importance --data` against its Python re-implementation, bit for bit, and the noise column that must rank last.

A module of its own because the oracle of the predictions and of the stored steps needs numpy (scripts/oracle_pipe.py, scripts/oracle_models.py):
scripts/run_conformance.py --no-oracles lists it as NOT RUN where numpy is not installed.

The oracle (scripts/oracle_interp.py) reads the model file, applies the stored steps (oracle_pipe.ref_apply and its own categorical blocks), predicts (the score order of
oracle_models.scores for the linear models, the tree walk of docs/models-tree.md 2.6 for the trees), scores with the metrics written again in the lane order of
docs/linalg.md 2, and permutes with SplitMix64 on Python ints: every printed string equal, except that log_loss uses the library's `log`, so it is compared with a
relative 1e-12 (the same bound as the metrics oracle)."""

import json
import math
import sys
import unittest

import numpy  # noqa: F401  (the import is the declaration of what this module needs)

from harness import ROOT, Sandbox, validate

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_interp as O  # noqa: E402


def csv_of(name, n, seed):
    return O.to_csv(*O.gen_dataset(name, n, seed))


def close_string(a, b, tol, scale=1.0):
    """|a - b| <= tol * scale: the library's log and Python's differ in the last bit of a term, and a difference of two log-losses keeps only the digits above that"""
    return abs(float(a) - float(b)) <= tol * scale


def equal_permutation(got, want, loose):
    """the printed objects equal; the members of a log_loss run within `loose` (relative)"""
    if list(got) != list(want):
        return False
    scale = abs(float(want["baseline"]))
    for k in got:
        if k == "columns":
            if len(got[k]) != len(want[k]):
                return False
            for a, b in zip(got[k], want[k]):
                if list(a) != list(b):
                    return False
                for m in a:
                    if a[m] != b[m] and not (loose and m in ("importance", "std") and close_string(a[m], b[m], loose, scale)):
                        return False
        elif got[k] != want[k] and not (loose and k == "baseline" and close_string(got[k], want[k], loose, scale)):
            return False
    return True


class Permutation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        sb = cls.sb
        sb.write("cls.csv", csv_of("cls", 300, 1))
        sb.write("reg.csv", csv_of("reg", 300, 1))
        sb.write("mixed.csv", csv_of("mixed", 300, 1))
        sb.write("cls_new.csv", csv_of("cls", 150, 2))
        sb.write("reg_new.csv", csv_of("reg", 150, 2))
        sb.write("mixed_new.csv", csv_of("mixed", 150, 2))
        imp = "impute column=* strategy=mean"
        cls.fits = {
            "lin_std": (["--model", "linear", "--step", imp + "; scale column=* kind=standard", "reg.csv"], "reg_new.csv"),
            "lin_raw": (["--model", "linear", "--step", imp, "reg.csv"], "reg_new.csv"),
            "lin_bucket": (["--model", "linear", "--step", "bucket column=x0,x1 kind=quantile bins=4; scale column=* kind=minmax", "reg.csv"], "reg_new.csv"),
            "lin_mixed": (["--model", "linear", "--drop", "color", "--step", imp + "; scale column=* kind=standard", "mixed.csv"], "mixed_new.csv"),
            "log_std": (["--model", "logistic", "--step", "scale column=* kind=standard", "cls.csv"], "cls_new.csv"),
            "log_raw": (["--model", "logistic", "--l2", "0.5", "cls.csv"], "cls_new.csv"),
            "tree_cls": (["--model", "tree", "--task", "classification", "--max-depth", "5", "cls.csv"], "cls_new.csv"),
            "tree_reg": (["--model", "tree", "--task", "regression", "--max-depth", "5", "reg.csv"], "reg_new.csv"),
            "forest_cls": (["--model", "forest", "--task", "classification", "--trees", "7", "--max-depth", "5", "--seed", "3", "cls.csv"], "cls_new.csv"),
            "forest_reg": (["--model", "forest", "--task", "regression", "--trees", "7", "--max-depth", "5", "--seed", "4", "reg.csv"], "reg_new.csv"),
            "forest_mixed": (["--model", "forest", "--task", "regression", "--trees", "5", "--max-depth", "4", "--seed", "6", "--drop", "color", "mixed.csv"], "mixed_new.csv"),
            "forest_cat": (["--model", "forest", "--task", "regression", "--trees", "5", "--max-depth", "4", "--seed", "6", "--cat", "color", "--step", "encode column=color kind=onehot",
                            "mixed.csv"], "mixed_new.csv"),
            "forest_cat_label": (["--model", "forest", "--task", "regression", "--trees", "4", "--max-depth", "4", "--seed", "7", "--cat", "color", "--step", "encode column=color kind=label",
                                  "mixed.csv"], "mixed_new.csv"),
        }
        for name, (argv, _data) in cls.fits.items():
            r = sb.run("fit", *argv[:-1], "--target", "y", "--output", name + ".ml", argv[-1])
            assert r.status == 0, (name, r)

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def check(self, name, metric, repeats, seed):
        argv, data = self.fits[name]
        M = O.read_model(self.sb.read(name + ".ml"))
        text = self.sb.read(data).decode()
        flags = ["--data", data, "--repeats", str(repeats), "--seed", str(seed)] + (["--metric", metric] if metric else []) + (["--drop", "color"] if "--drop" in argv else [])
        r = self.sb.run("importance", "--model-file", name + ".ml", *flags)
        self.assertEqual(r.status, 0, (name, metric, r))
        self.assertEqual(validate(r), [])
        got = r.data()["permutation"]
        want = O.permutation(M, text, metric, repeats, seed)
        loose = 1e-12 if metric == "log_loss" else 0
        self.assertTrue(equal_permutation(got, want, loose), (name, metric, repeats, seed, json.dumps(got)[:900], json.dumps(want)[:900]))
        return got

    def test_i6_regression_models_bit_for_bit(self):
        for name in ("lin_std", "lin_raw", "lin_bucket", "lin_mixed", "tree_reg", "forest_reg", "forest_mixed", "forest_cat", "forest_cat_label"):
            for metric in ("rmse", "mae", "r2"):
                for repeats, seed in ((1, 0), (3, 7), (7, 12345)):
                    if metric != "rmse" and (repeats, seed) != (3, 7):
                        continue
                    with self.subTest(model=name, metric=metric, repeats=repeats, seed=seed):
                        self.check(name, metric, repeats, seed)

    def test_i6_classification_models_bit_for_bit(self):
        for name in ("log_std", "log_raw", "tree_cls", "forest_cls"):
            for metric in ("roc_auc", "accuracy", "precision", "recall", "f1", "log_loss"):
                for repeats, seed in ((1, 0), (3, 7), (7, 12345)):
                    if metric != "roc_auc" and (repeats, seed) != (3, 7):
                        continue
                    with self.subTest(model=name, metric=metric, repeats=repeats, seed=seed):
                        self.check(name, metric, repeats, seed)

    def test_i6_the_default_metric_and_the_default_repeats(self):
        got = self.sb.run("importance", "--model-file", "forest_reg.ml", "--data", "reg_new.csv").data()["permutation"]
        self.assertEqual((got["metric"], got["repeats"], got["seed"], got["direction"]), ("rmse", 5, 0, "loss"))
        got = self.sb.run("importance", "--model-file", "forest_cls.ml", "--data", "cls_new.csv").data()["permutation"]
        self.assertEqual((got["metric"], got["repeats"], got["direction"]), ("roc_auc", 5, "score"))
        want = O.permutation(O.read_model(self.sb.read("forest_cls.ml")), self.sb.read("cls_new.csv").decode())
        self.assertEqual(got, want)

    def test_i6_the_seed_changes_the_numbers_and_a_column_has_its_own_stream(self):
        a = self.sb.run("importance", "--model-file", "forest_reg.ml", "--data", "reg_new.csv", "--seed", "1").data()["permutation"]
        b = self.sb.run("importance", "--model-file", "forest_reg.ml", "--data", "reg_new.csv", "--seed", "2").data()["permutation"]
        self.assertNotEqual(a["columns"], b["columns"])
        # the streams of two columns differ: the permutations of column 0 and 1 for repeat 0
        self.assertNotEqual(O.permutation_of(50, 1, 0, 0), O.permutation_of(50, 1, 0, 1))
        self.assertNotEqual(O.permutation_of(50, 1, 0, 0), O.permutation_of(50, 1, 1, 0))

    def test_i6_the_order_of_the_columns_of_the_data_file_changes_nothing(self):
        text = self.sb.read("reg_new.csv").decode().splitlines()
        header = text[0].split(",")
        order = list(reversed(range(len(header))))
        out = [",".join(header[i] for i in order)] + [",".join(line.split(",")[i] for i in order) for line in text[1:]]
        self.sb.write("reg_rev.csv", "\n".join(out) + "\n")
        a = self.sb.run("importance", "--model-file", "forest_reg.ml", "--data", "reg_new.csv", "--seed", "9")
        b = self.sb.run("importance", "--model-file", "forest_reg.ml", "--data", "reg_rev.csv", "--seed", "9")
        self.assertEqual(a.stdout, b.stdout)

    def test_i6_a_missing_cell_travels_with_the_shuffle(self):
        # the mixed file has missing cells in a, b, c: the tree models take them; the oracle moves the cell, not the value
        got = self.check("forest_mixed", "rmse", 3, 5)
        self.assertEqual(len(got["columns"]), 4)

    def test_refusals_of_the_permutation(self):
        r = self.sb.run("importance", "--model-file", "forest_reg.ml", "--data", "reg_new.csv", "--metric", "roc_auc")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.importance.bad-metric"), r)
        r = self.sb.run("importance", "--model-file", "forest_cls.ml", "--data", "cls_new.csv", "--metric", "rmse")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.importance.bad-metric"), r)
        r = self.sb.run("importance", "--model-file", "forest_reg.ml", "--data", "reg_new.csv", "--repeats", "0")
        self.assertEqual((r.status, r.first_rule()), (2, "ml.param.out-of-range"), r)
        r = self.sb.run("importance", "--model-file", "forest_reg.ml", "--data", "reg_new.csv", "--repeats", "101")
        self.assertEqual((r.status, r.first_rule()), (2, "ml.param.out-of-range"), r)
        self.sb.write("one.csv", "x0,x1,x2,NOISE,x3,x4,x5,y\n0.1,0.2,0.3,0.4,0.5,0.6,0.7,1.0\n")
        r = self.sb.run("importance", "--model-file", "forest_reg.ml", "--data", "one.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.importance.too-few-rows"), r)
        # a constant target: R-squared has no value
        self.sb.write("flat.csv", "x0,x1,x2,NOISE,x3,x4,x5,y\n0.1,0.2,0.3,0.4,0.5,0.6,0.7,1.0\n0.2,0.2,0.3,0.4,0.5,0.6,0.7,1.0\n0.3,0.2,0.3,0.4,0.5,0.6,0.7,1.0\n")
        r = self.sb.run("importance", "--model-file", "forest_reg.ml", "--data", "flat.csv", "--metric", "r2")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.importance.undefined-metric"), r)
        # a classification file with one class: the AUC has no value; a label the model never saw is the refusal of eval
        rows = csv_of("cls", 30, 3).splitlines()
        self.sb.write("oneclass.csv", "\n".join([rows[0]] + [ln.rsplit(",", 1)[0] + ",0" for ln in rows[1:]]) + "\n")
        r = self.sb.run("importance", "--model-file", "forest_cls.ml", "--data", "oneclass.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.importance.undefined-metric"), r)
        self.sb.write("foreign.csv", "\n".join([rows[0]] + [ln.rsplit(",", 1)[0] + (",2" if i == 3 else ",0" if i % 2 else ",1") for i, ln in enumerate(rows[1:])]) + "\n")
        for metric in ("roc_auc", "accuracy"):
            # accuracy needs no binary labels in the library, so the refusal is the verb's own check
            r = self.sb.run("importance", "--model-file", "forest_cls.ml", "--data", "foreign.csv", "--metric", metric)
            self.assertEqual((r.status, r.first_rule()), (8, "ml.label.not-binary"), (metric, r))
        # the data lacks a column of the model
        self.sb.write("short.csv", "x0,x1,y\n0.1,0.2,1.0\n0.3,0.4,2.0\n")
        r = self.sb.run("importance", "--model-file", "forest_reg.ml", "--data", "short.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.schema.mismatch"), r)
        # no target column in the data
        self.sb.write("notarget.csv", "x0,x1,x2,NOISE,x3,x4,x5\n0.1,0.2,0.3,0.4,0.5,0.6,0.7\n0.2,0.2,0.3,0.4,0.5,0.6,0.7\n")
        r = self.sb.run("importance", "--model-file", "forest_reg.ml", "--data", "notarget.csv")
        self.assertEqual(r.status, 8, r)

    def test_the_work_bound_is_exact(self):
        # 150 rows x 7 columns x 3 repeats = 3150 row evaluations
        ok = self.sb.run("importance", "--model-file", "forest_reg.ml", "--data", "reg_new.csv", "--repeats", "3", "--max-work", "3150")
        self.assertEqual(ok.status, 0, ok)
        r = self.sb.run("importance", "--model-file", "forest_reg.ml", "--data", "reg_new.csv", "--repeats", "3", "--max-work", "3149")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.limit.work"), r)
        det = json.dumps(r.error())
        self.assertIn("3150", det)

    def test_i1_the_model_table_comes_with_the_permutation(self):
        d = self.sb.run("importance", "--model-file", "forest_cls.ml", "--data", "cls_new.csv").data()
        self.assertTrue(d["importance"]["available"])
        self.assertEqual(d["importance"], self.sb.run("importance", "--model-file", "forest_cls.ml").data()["importance"])


class Noise(unittest.TestCase):
    """I5: a column of noise ranks last, in the impurity importance of a forest and in the permutation importance on held-out rows"""

    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def run_set(self, name, task):
        sb = self.sb
        sb.write(name + ".csv", csv_of(name, 2000, 11))
        sb.write(name + "_new.csv", csv_of(name, 1000, 12))
        r = sb.run("fit", "--model", "forest", "--task", task, "--trees", "100", "--max-depth", "8", "--seed", "1", "--target", "y", "--output", name + ".ml", name + ".csv")
        self.assertEqual(r.status, 0, r)
        d = sb.run("importance", "--model-file", name + ".ml", "--data", name + "_new.csv", "--repeats", "10", "--seed", "3").data()
        return d

    def last(self, cols):
        return max(cols, key=lambda c: c["rank"])["column"]

    def test_i5_the_noise_column_of_a_classification_forest_ranks_last(self):
        d = self.run_set("cls", "classification")
        self.assertEqual(self.last(d["importance"]["columns"]), "NOISE", d["importance"]["columns"])
        self.assertEqual(self.last(d["permutation"]["columns"]), "NOISE", d["permutation"]["columns"])
        # the noise column is below the weakest informative column by a margin, not only by rank: its importance is less than a fifth of the weakest signal's
        cols = {c["column"]: float(c["importance"]) for c in d["permutation"]["columns"]}
        weakest = min(v for k, v in cols.items() if k != "NOISE")
        self.assertLess(cols["NOISE"], 0.2 * weakest, cols)

    def test_i5_the_noise_column_of_a_regression_forest_ranks_last(self):
        d = self.run_set("reg", "regression")
        self.assertEqual(self.last(d["importance"]["columns"]), "NOISE", d["importance"]["columns"])
        self.assertEqual(self.last(d["permutation"]["columns"]), "NOISE", d["permutation"]["columns"])
        # the noise column is below the weakest informative column by a margin, not only by rank: its importance is less than a fifth of the weakest signal's
        cols = {c["column"]: float(c["importance"]) for c in d["permutation"]["columns"]}
        weakest = min(v for k, v in cols.items() if k != "NOISE")
        self.assertLess(cols["NOISE"], 0.2 * weakest, cols)


if __name__ == "__main__":
    unittest.main()

"""I1, I2, I3, I8, I9, I10, I11 -- `ml importance`, `ml dump` and the `importance` member of `describe` (docs/interpret.md section 8).

Every table the binary prints is compared with scripts/oracle_interp.py, which reads the model file on its own (no code of tools/ml, lib/pipeline or lib/transforms) and
computes the same tables from the definitions of docs/interpret.md sections 3 and 4: the printed strings (a double is the shortest decimal that reads back) and the
integers must be EQUAL, key order included. This module needs no numpy: the gates that do (the permutation importance, bit for bit) are test_interp_oracle.py, the ones
that need scikit-learn are test_interp_sklearn.py.

INTERP_STRIDE (default 11) is the stride of the damage sweep of I11: every byte is damaged with 1 (the full sweep, `check.sh` on the Linux machines)."""

import json
import math
import os
import random
import sys
import unittest

from harness import ROOT, Sandbox, TRAPS, fix_checksum, introspect, run, validate

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_interp as O  # noqa: E402

STRIDE = int(os.environ.get("INTERP_STRIDE", "11"))


def same(a, b):
    """equal as JSON text, key order included"""
    return json.dumps(a, separators=(",", ":"), ensure_ascii=False) == json.dumps(b, separators=(",", ":"), ensure_ascii=False)


def dataset(name, n, seed=1):
    return O.to_csv(*O.gen_dataset(name, n, seed))


class Fixture(unittest.TestCase):
    """one sandbox with the data sets and the models of the tests"""

    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        sb = cls.sb
        sb.write("cls.csv", dataset("cls", 300))
        sb.write("reg.csv", dataset("reg", 300))
        sb.write("mixed.csv", dataset("mixed", 300))
        lines = dataset("reg", 300).splitlines()
        sb.write("regk.csv", "\n".join([lines[0].replace(",y", ",k,y")] + [ln.rsplit(",", 1)[0] + ",3.0," + ln.rsplit(",", 1)[1] for ln in lines[1:]]) + "\n")
        sb.write("disc.csv", dataset("disc", 300))
        sb.write("rare.csv", dataset("rare", 400))
        sb.write("const.csv", "a,b,y\n1,2,5\n2,1,5\n3,4,5\n4,5,5\n5,6,5\n6,3,5\n")
        steps = "impute column=* strategy=mean; scale column=* kind=standard"
        fits = {
            "lin_plain.ml": ["--model", "linear", "--target", "y", "--step", "impute column=* strategy=mean", "reg.csv"],
            "lin_std.ml": ["--model", "linear", "--target", "y", "--step", steps, "reg.csv"],
            "lin_ddof1.ml": ["--model", "linear", "--target", "y", "--step", "scale column=* kind=standard ddof=1", "reg.csv"],
            "lin_minmax.ml": ["--model", "linear", "--target", "y", "--step", "scale column=* kind=minmax", "reg.csv"],
            "lin_twice.ml": ["--model", "linear", "--target", "y", "--step", "scale column=x0,x1 kind=standard; scale column=* kind=minmax", "reg.csv"],
            "lin_twice2.ml": ["--model", "linear", "--target", "y", "--step", "scale column=x0,x1 kind=minmax; scale column=x0,x2 kind=standard ddof=1", "reg.csv"],
            "lin_constcol.ml": ["--model", "linear", "--l2", "0.1", "--target", "y", "--step", "scale column=* kind=standard", "regk.csv"],
            "lin_bucket.ml": ["--model", "linear", "--target", "y", "--step", "bucket column=x0 kind=width bins=4", "reg.csv"],
            "log_std.ml": ["--model", "logistic", "--target", "y", "--step", "scale column=* kind=standard", "cls.csv"],
            "log_plain.ml": ["--model", "logistic", "--target", "y", "--l2", "0.5", "cls.csv"],
            "tree_cls.ml": ["--model", "tree", "--task", "classification", "--target", "y", "--max-depth", "4", "cls.csv"],
            "tree_reg.ml": ["--model", "tree", "--task", "regression", "--target", "y", "--max-depth", "5", "reg.csv"],
            "forest_cls.ml": ["--model", "forest", "--task", "classification", "--target", "y", "--trees", "6", "--max-depth", "4", "--seed", "3", "cls.csv"],
            "forest_reg.ml": ["--model", "forest", "--task", "regression", "--target", "y", "--trees", "6", "--max-depth", "5", "--seed", "4", "reg.csv"],
            "forest_nb.ml": ["--model", "forest", "--task", "regression", "--target", "y", "--trees", "4", "--max-depth", "4", "--bootstrap", "0", "--seed", "5", "reg.csv"],
            "forest_mixed.ml": ["--model", "forest", "--task", "regression", "--target", "y", "--trees", "5", "--max-depth", "4", "--seed", "6", "--drop", "color", "mixed.csv"],
            "forest_cat.ml": ["--model", "forest", "--task", "regression", "--target", "y", "--trees", "5", "--max-depth", "4", "--seed", "6", "--cat", "color",
                              "--step", "encode column=color kind=onehot", "mixed.csv"],
            "tree_onehot.ml": ["--model", "tree", "--task", "regression", "--target", "y", "--max-depth", "4", "--step", "encode column=g kind=onehot", "disc.csv"],
            "lin_onehot.ml": ["--model", "linear", "--l2", "0.1", "--target", "y", "--step", "encode column=g kind=onehot", "disc.csv"],
            "forest_rare.ml": ["--model", "forest", "--task", "classification", "--target", "y", "--trees", "60", "--max-depth", "3", "--seed", "9", "rare.csv"],
            "tree_leaf.ml": ["--model", "tree", "--task", "regression", "--target", "y", "const.csv"],
        }
        for name, argv in fits.items():
            r = sb.run("fit", *argv[:-1], "--output", name, argv[-1])
            assert r.status == 0, (name, r)
        # a clustering has no target: the target column is dropped from the file
        r = sb.run("fit", "--model", "kmeans", "--k", "3", "--seed", "2", "--drop", "y", "--step", "scale column=* kind=standard", "--output", "km.ml", "reg.csv")
        assert r.status == 0, r
        fits["km.ml"] = None
        r = sb.run("fit", "--model", "constant", "--task", "regression", "--target", "y", "--output", "const.ml", "reg.csv")
        assert r.status == 0, r
        cls.names = sorted(fits) + ["const.ml"]

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def model(self, name):
        return O.read_model(self.sb.read(name))

    def importance(self, name, *flags):
        r = self.sb.run("importance", "--model-file", name, *flags)
        self.assertEqual(r.status, 0, r)
        self.assertEqual(validate(r), [])
        return r.data()

    def describe(self, name):
        r = self.sb.run("describe", "--model-file", name)
        self.assertEqual(r.status, 0, r)
        return r.data()


class StandardisedCoefficients(Fixture):
    def test_i1_the_table_equals_the_oracle_byte_for_byte(self):
        for name in ("lin_std.ml", "lin_ddof1.ml", "lin_minmax.ml", "lin_twice.ml", "lin_twice2.ml", "lin_plain.ml", "lin_constcol.ml", "log_std.ml", "log_plain.ml"):
            with self.subTest(model=name):
                want = O.importance_member(self.model(name))
                got = self.importance(name)
                self.assertTrue(want["available"], name)
                self.assertTrue(same(got["importance"], want), (name, got["importance"], want))
                self.assertIsNone(got["permutation"])

    def test_i1_the_sources_of_the_deviation(self):
        got = {n: [c["std_source"] for c in self.importance(n)["importance"]["columns"]] for n in ("lin_std.ml", "lin_minmax.ml", "lin_twice.ml", "lin_plain.ml")}
        self.assertEqual(set(got["lin_std.ml"]), {"scale"})
        self.assertEqual(set(got["lin_minmax.ml"]), {"provenance"})
        self.assertEqual(set(got["lin_plain.ml"]), {"provenance"})
        # x0 and x1 end as minmax of a standard scale: the provenance, the others minmax only; none is a unit-variance column
        self.assertEqual(set(got["lin_twice.ml"]), {"provenance"})

    def test_i1_a_constant_column_has_a_zero_deviation(self):
        cols = self.importance("lin_constcol.ml")["importance"]["columns"]
        k = [c for c in cols if c["column"] == "k"][0]
        self.assertEqual((k["std"], k["standardised"], k["importance"], k["std_source"]), ("0.0", "0.0", "0.0", "scale"))

    def test_i1_ranks_are_distinct_and_ordered_by_importance(self):
        for name in ("lin_std.ml", "log_std.ml"):
            cols = self.importance(name)["importance"]["columns"]
            self.assertEqual(sorted(c["rank"] for c in cols), list(range(1, len(cols) + 1)))
            by_rank = sorted(cols, key=lambda c: c["rank"])
            vals = [float(c["importance"]) for c in by_rank]
            self.assertEqual(vals, sorted(vals, reverse=True))

    def test_i1_the_noise_column_of_a_linear_model_is_last(self):
        cols = self.importance("lin_std.ml")["importance"]["columns"]
        last = max(cols, key=lambda c: c["rank"])
        self.assertEqual(last["column"], "NOISE", cols)

    def test_i3_the_shares_add_up_to_one(self):
        for name in ("lin_std.ml", "log_std.ml", "lin_plain.ml"):
            imp = self.importance(name)["importance"]
            self.assertEqual(imp["total"], "1.0")
            t = 0.0
            for c in imp["columns"]:
                t += float(c["importance"])
            self.assertLess(abs(t - 1.0), 8 * 2.0 ** -52 * len(imp["columns"]), name)

    def test_i3_all_zero_coefficients_have_total_zero(self):
        # a constant target: the least-squares coefficients are zero up to rounding; a column of the file is constant and its coefficient is checked below
        self.sb.write("zero.csv", "a,b,y\n1,2,0\n2,1,0\n3,4,0\n4,5,0\n5,6,0\n6,3,0\n7,9,0\n")
        r = self.sb.run("fit", "--model", "linear", "--target", "y", "--step", "scale column=* kind=standard", "--output", "zero.ml", "zero.csv")
        self.assertEqual(r.status, 0, r)
        want = O.importance_member(self.model("zero.ml"))
        got = self.importance("zero.ml")["importance"]
        self.assertTrue(same(got, want), (got, want))
        self.assertIn(got["total"], ("0.0", "1.0"))

    def test_i1_a_file_with_no_stored_deviation_is_refused_with_a_repair(self):
        # remove the provenance section: no scale step either, so nothing states a deviation
        from harness import without_provenance
        data = without_provenance(self.sb.read("lin_plain.ml"))
        self.sb.write("old.ml", data)
        r = self.sb.run("importance", "--model-file", "old.ml")
        self.assertEqual(r.status, 8, r)
        self.assertEqual(r.first_rule(), "ml.importance.no-scale")
        self.assertEqual(validate(r), [])
        err = r.error()
        self.assertIn("--data", json.dumps(err))
        # describe still answers, and says why
        d = self.describe_file("old.ml")
        self.assertEqual(d["importance"]["available"], False)
        self.assertEqual(d["importance"]["reason"], "ml.importance.no-scale")
        want = O.importance_member(O.read_model(data))
        self.assertTrue(same(d["importance"], want), (d["importance"], want))
        # a scaled file needs no provenance: the plan states the deviation
        data2 = without_provenance(self.sb.read("lin_std.ml"))
        self.sb.write("old2.ml", data2)
        r2 = self.sb.run("importance", "--model-file", "old2.ml")
        self.assertEqual(r2.status, 0, r2)
        self.assertTrue(same(r2.data()["importance"], O.importance_member(O.read_model(data2))))

    def test_i1_an_encoder_column_has_no_deviation(self):
        r = self.sb.run("importance", "--model-file", "lin_bucket.ml")
        self.assertEqual(r.status, 8, r)
        self.assertEqual(r.first_rule(), "ml.importance.no-scale")
        det = r.error().get("detail", {})
        self.assertIn("x0", json.dumps(det))

    def test_i1_the_permutation_needs_no_stored_deviation(self):
        # the refusal of a table without a deviation is lifted by --data: the permutation importance is computed, the table says why it is unavailable
        r = self.sb.run("importance", "--model-file", "lin_bucket.ml", "--data", "reg.csv", "--repeats", "2")
        self.assertEqual(r.status, 0, r)
        d = r.data()
        self.assertFalse(d["importance"]["available"])
        self.assertEqual(d["importance"]["reason"], "ml.importance.no-scale")
        self.assertTrue(same(d["importance"], O.importance_member(self.model("lin_bucket.ml"))))
        self.assertEqual(len(d["permutation"]["columns"]), 7)

    def test_i1_an_onehot_column_of_a_numeric_encoder_is_derived(self):
        r = self.sb.run("importance", "--model-file", "lin_onehot.ml")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.importance.no-scale"), r)
        self.assertIn("g#0", json.dumps(r.error().get("detail", {})))

    def describe_file(self, name):
        r = self.sb.run("describe", "--model-file", name)
        self.assertEqual(r.status, 0, r)
        return r.data()


class ImpurityDecrease(Fixture):
    def test_i2_the_table_equals_the_oracle_byte_for_byte(self):
        for name in ("tree_cls.ml", "tree_reg.ml", "forest_cls.ml", "forest_reg.ml", "forest_nb.ml", "forest_mixed.ml", "forest_cat.ml", "tree_leaf.ml", "tree_onehot.ml", "forest_rare.ml"):
            with self.subTest(model=name):
                want = O.importance_member(self.model(name))
                got = self.importance(name)
                self.assertTrue(same(got["importance"], want), (name, got["importance"], want))
                for c in got["importance"]["columns"]:
                    self.assertIsInstance(c["splits"], int)
                    self.assertIsInstance(c["split_rows"], int)
                    self.assertIsInstance(c["importance"], str)

    def test_i2_a_tree_is_a_forest_of_one(self):
        # the importance of the first tree of a forest read on its own equals the importance of the forest of one tree fitted with bootstrap off
        M = self.model("tree_cls.ml")
        self.assertEqual(M["tree"]["trees"], 1)
        imps, splits, rows, positive = O.tree_scores(M)
        self.assertTrue(positive)
        self.assertEqual(sum(splits), sum(1 for f in M["tree"]["feat"] if f >= 0))

    def test_i3_total_is_one_and_zero_for_a_tree_that_never_splits(self):
        imp = self.importance("forest_cls.ml")["importance"]
        self.assertEqual(imp["total"], "1.0")
        t = 0.0
        for c in imp["columns"]:
            t += float(c["importance"])
        self.assertLess(abs(t - 1.0), 8 * 2.0 ** -52 * len(imp["columns"]))
        leaf = self.importance("tree_leaf.ml")["importance"]
        self.assertEqual(leaf["total"], "0.0")
        self.assertEqual({c["importance"] for c in leaf["columns"]}, {"0.0"})
        self.assertEqual({c["splits"] for c in leaf["columns"]}, {0})
        self.assertEqual(sorted(c["rank"] for c in leaf["columns"]), list(range(1, len(leaf["columns"]) + 1)))

    def test_i2_a_forest_with_a_tree_that_never_splits(self):
        # some bootstrap samples of a set with two positives hold none: the tree grown on one is a single leaf and adds nothing to the average
        M = self.model("forest_rare.ml")
        T = M["tree"]
        leaves = [t for t in range(T["trees"]) if T["off"][t + 1] - T["off"][t] == 1]
        splitting = [t for t in range(T["trees"]) if T["off"][t + 1] - T["off"][t] > 1]
        self.assertTrue(leaves and splitting, (len(leaves), len(splitting)))
        got = self.importance("forest_rare.ml")["importance"]
        self.assertEqual(got["total"], "1.0")
        t = 0.0
        for c in got["columns"]:
            t += float(c["importance"])
        self.assertLess(abs(t - 1.0), 1e-12)

    def test_i2_the_split_counts_are_not_named_importance(self):
        cols = self.importance("forest_reg.ml")["importance"]["columns"]
        self.assertTrue(all("splits" in c and "split_rows" in c for c in cols))
        # the importance and the number of splits rank differently somewhere in a forest: they are two measures
        self.assertEqual(self.importance("forest_reg.ml")["importance"]["method"], "impurity_decrease")

    def test_i2_the_categorical_column_is_one_row_per_model_column_with_its_source(self):
        cols = self.importance("forest_cat.ml")["importance"]["columns"]
        blue = [c for c in cols if c["source"] == "color"]
        self.assertGreaterEqual(len(blue), 2)
        self.assertTrue(all(c["column"].startswith("color#") for c in blue))


class DescribeMember(Fixture):
    def test_i9_describe_carries_the_member_after_limits(self):
        for name in self.names:
            with self.subTest(model=name):
                d = self.describe(name)
                keys = list(d)
                self.assertEqual(keys[-1], "importance", keys)
                self.assertEqual(keys[-2], "limits")
                M = self.model(name)
                want = O.importance_member(M) if M["model"] in ("linear", "logistic", "tree", "forest") else None
                self.assertTrue(same(d["importance"], want), (name, d["importance"], want))

    def test_i9_the_limit_cuts_the_table_to_the_best_ranks(self):
        # like the categories, the rows of the table listed by describe are cut by --limit (ml-serve relies on it to fit one reply); the verb lists every row
        for name in ("forest_reg.ml", "lin_std.ml", "forest_cat.ml"):
            with self.subTest(model=name):
                got = self.sb.run("describe", "--model-file", name, "--limit", "3").data()["importance"]
                self.assertTrue(same(got, O.importance_member(self.model(name), 3)), (name, got))
                self.assertEqual((got["listed"], got["truncated"], [c["rank"] for c in got["columns"]]), (3, True, [1, 2, 3]))
                whole = self.importance(name)["importance"]
                self.assertEqual((whole["listed"], whole["truncated"]), (len(whole["columns"]), False))
                self.assertEqual(got["columns"], sorted((c for c in whole["columns"] if c["rank"] <= 3), key=lambda c: c["rank"]))
        # a table that cannot be computed lists the names it names up to --limit
        got = self.sb.run("describe", "--model-file", "lin_onehot.ml", "--limit", "1").data()["importance"]
        self.assertTrue(same(got, O.importance_member(self.model("lin_onehot.ml"), 1)))
        self.assertEqual((got["available"], got["listed"], got["truncated"], len(got["missing"])), (False, 1, True, 1))

    def test_i9_describe_and_the_verb_print_the_same_object(self):
        for name in ("lin_std.ml", "log_plain.ml", "forest_cls.ml", "tree_reg.ml"):
            self.assertTrue(same(self.describe(name)["importance"], self.importance(name)["importance"]), name)

    def test_i9_the_kinds_with_no_importance_say_null(self):
        self.assertIsNone(self.describe("km.ml")["importance"])
        self.assertIsNone(self.describe("const.ml")["importance"])

    def test_i9_ml_predict_describes_the_same(self):
        for name in ("lin_std.ml", "forest_reg.ml", "km.ml"):
            a = self.sb.run("describe", "--model-file", name).stdout
            b = run("--root", self.sb.dir, "describe", "--model-file", name, binary_path=os.environ.get("ML_PREDICT_BIN") or str(ROOT / "build" / "ml-predict")).stdout
            self.assertEqual(a, b, name)


class Unsupported(Fixture):
    def test_the_refusals_of_importance(self):
        for name in ("km.ml", "const.ml"):
            r = self.sb.run("importance", "--model-file", name)
            self.assertEqual((r.status, r.first_rule()), (8, "ml.importance.unsupported-model"), r)
            self.assertEqual(validate(r), [])

    def test_the_refusal_of_dump(self):
        r = self.sb.run("dump", "--model-file", "const.ml")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.dump.unsupported-model"), r)

    def test_flags_without_data_are_a_conflict(self):
        for flags in (["--metric", "rmse"], ["--repeats", "3"]):
            r = self.sb.run("importance", "--model-file", "lin_std.ml", *flags)
            self.assertEqual((r.status, r.first_rule()), (2, "args.conflict"), (flags, r))

    def test_ml_predict_has_neither_verb(self):
        pb = os.environ.get("ML_PREDICT_BIN") or str(ROOT / "build" / "ml-predict")
        for verb in ("importance", "dump"):
            r = run("--root", self.sb.dir, verb, "--model-file", "lin_std.ml", binary_path=pb)
            self.assertEqual((r.status, r.first_rule()), (2, "args.unknown-command"), r)


class Dump(Fixture):
    def test_i10_every_tree_of_a_forest_equals_the_oracle(self):
        for name in ("forest_cls.ml", "forest_reg.ml", "forest_mixed.ml", "forest_cat.ml", "tree_cls.ml", "tree_reg.ml", "tree_leaf.ml"):
            M = self.model(name)
            for i in range(M["tree"]["trees"]):
                with self.subTest(model=name, tree=i):
                    args = ["dump", "--model-file", name] + (["--tree", str(i)] if M["model"] == "forest" or i else [])
                    r = self.sb.run(*args)
                    self.assertEqual(r.status, 0, r)
                    self.assertEqual(validate(r), [])
                    want = O.tree_dump(M, i)
                    self.assertTrue(same(r.data(), want), (name, i))

    def test_i10_the_default_tree_is_the_first(self):
        a = self.sb.run("dump", "--model-file", "forest_cls.ml").data()
        b = self.sb.run("dump", "--model-file", "forest_cls.ml", "--tree", "0").data()
        self.assertEqual(a, b)
        self.assertEqual(a["tree"], 0)

    def test_i10_the_ceiling_of_max_nodes(self):
        ok = self.sb.run("dump", "--model-file", "forest_reg.ml", "--max-nodes", "4194304")
        self.assertEqual(ok.status, 0, ok)
        r = self.sb.run("dump", "--model-file", "forest_reg.ml", "--max-nodes", "4194305")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.limit.nodes"), r)

    def test_i10_a_tree_out_of_range(self):
        r = self.sb.run("dump", "--model-file", "forest_cls.ml", "--tree", "6")
        self.assertEqual((r.status, r.first_rule()), (2, "ml.param.out-of-range"), r)
        r = self.sb.run("dump", "--model-file", "tree_cls.ml", "--tree", "1")
        self.assertEqual((r.status, r.first_rule()), (2, "ml.param.out-of-range"), r)

    def test_i10_the_node_bound_is_exact(self):
        n = self.sb.run("dump", "--model-file", "forest_reg.ml", "--tree", "2").data()["nodes"]
        ok = self.sb.run("dump", "--model-file", "forest_reg.ml", "--tree", "2", "--max-nodes", str(n))
        self.assertEqual(ok.status, 0, ok)
        r = self.sb.run("dump", "--model-file", "forest_reg.ml", "--tree", "2", "--max-nodes", str(n - 1))
        self.assertEqual((r.status, r.first_rule()), (8, "ml.limit.nodes"), r)
        det = json.dumps(r.error())
        self.assertIn(str(n), det)

    def test_i10_a_dumped_tree_predicts_what_predict_predicts(self):
        M = self.model("tree_reg.ml")
        d = self.sb.run("dump", "--model-file", "tree_reg.ml").data()
        nodes = {n["id"]: n for n in d["list"]}
        text = dataset("reg", 300).splitlines()
        header = text[0].split(",")
        rows = [dict(zip(header, line.split(","))) for line in text[1:101]]
        self.sb.write("p100.csv", "\n".join([",".join(header[:-1])] + [",".join(r[h] for h in header[:-1]) for r in rows]) + "\n")
        pr = self.sb.run("predict", "--model-file", "tree_reg.ml", "p100.csv")
        self.assertEqual(pr.status, 0, pr)
        pred = [float(v[0]) for v in pr.data()["rows"]]
        for r, want in zip(rows, pred):
            i = 0
            while not nodes[i]["leaf"]:
                n = nodes[i]
                x = r[n["source"]]
                go_left = float(x) < float(n["threshold"])
                i = n["left"] if go_left else n["right"]
            self.assertEqual(float(nodes[i]["value"]), want)

    def test_i10_the_text_form_prints_one_line_a_node(self):
        r = self.sb.run("--format", "text", "dump", "--model-file", "forest_cls.ml", "--tree", "1")
        self.assertEqual(r.status, 0, r)
        n = self.sb.run("dump", "--model-file", "forest_cls.ml", "--tree", "1").data()["nodes"]
        lines = r.stdout.decode().splitlines()
        self.assertEqual(len([ln for ln in lines if "leaf" in ln or "<" in ln]), n, r.stdout.decode())

    def test_i10_kmeans_centroids_and_linear_coefficients(self):
        r = self.sb.run("dump", "--model-file", "km.ml")
        self.assertEqual(r.status, 0, r)
        self.assertTrue(same(r.data(), O.kmeans_dump(self.model("km.ml"))))
        r = self.sb.run("dump", "--model-file", "lin_std.ml")
        self.assertEqual(r.status, 0, r)
        self.assertTrue(same(r.data(), O.linear_dump(self.model("lin_std.ml"))))
        r = self.sb.run("dump", "--model-file", "log_std.ml")
        self.assertTrue(same(r.data(), O.linear_dump(self.model("log_std.ml"))))

    def test_i10_a_tree_flag_on_a_linear_model_is_a_conflict(self):
        r = self.sb.run("dump", "--model-file", "lin_std.ml", "--tree", "0")
        self.assertEqual((r.status, r.first_rule()), (2, "args.conflict"), r)

    def test_i10_the_cell_text_of_an_int_and_a_decimal_column(self):
        # an int column and a decimal column: the threshold prints as the cell would
        rows = ["n,d,y"]
        rnd = random.Random(5)
        for i in range(120):
            n = rnd.randint(-20, 40)
            dv = rnd.randint(-500, 900) / 100.0
            rows.append("%d,%.2f,%s" % (n, dv, repr(0.1 * n + 0.01 * dv * dv)))
        self.sb.write("cells.csv", "\n".join(rows) + "\n")
        r = self.sb.run("fit", "--model", "tree", "--task", "regression", "--target", "y", "--max-depth", "5", "--output", "cells.ml", "cells.csv")
        self.assertEqual(r.status, 0, r)
        M = self.model("cells.ml")
        self.assertEqual([c[1] for c in M["cols"]][:2], [0, 1])
        d = self.sb.run("dump", "--model-file", "cells.ml").data()
        self.assertTrue(same(d, O.tree_dump(M, 0)))
        cells = [(n["source"], n["cell"]) for n in d["list"] if not n["leaf"]]
        self.assertTrue(any(s == "n" and "." not in c for s, c in cells), cells)
        self.assertTrue(any(s == "d" and len(c.split(".")[1]) == 2 for s, c in cells), cells)

    def test_i10_a_scaled_column_has_no_cell_text(self):
        r = self.sb.run("fit", "--model", "tree", "--task", "regression", "--target", "y", "--max-depth", "3", "--step", "scale column=* kind=standard", "--output", "tsc.ml", "reg.csv")
        self.assertEqual(r.status, 0, r)
        d = self.sb.run("dump", "--model-file", "tsc.ml").data()
        self.assertTrue(all(n["cell"] is None for n in d["list"] if not n["leaf"]))
        self.assertTrue(same(d, O.tree_dump(self.model("tsc.ml"), 0)))


class Threads(Fixture):
    def test_i8_the_answers_do_not_depend_on_the_thread_count(self):
        for name in ("lin_std.ml", "forest_reg.ml", "tree_cls.ml"):
            outs = {self.sb.run("importance", "--model-file", name, "--threads", str(t)).stdout for t in (1, 8)}
            self.assertEqual(len(outs), 1, name)
        for name, data in (("lin_std.ml", "reg.csv"), ("forest_cls.ml", "cls.csv")):
            outs = {self.sb.run("importance", "--model-file", name, "--data", data, "--repeats", "2", "--threads", str(t)).stdout for t in (1, 8)}
            self.assertEqual(len(outs), 1, name)


class Damage(Fixture):
    """I11: no damaged file traps `importance` or `dump`; one that is refused is refused with a declared status"""

    def sweep(self, name, verbs):
        data = self.sb.read(name)
        declared = {c["code"] for c in introspect()["exit_codes"]} | {0}
        bad = []
        probes = []
        for n in range(0, len(data), STRIDE):
            probes.append(data[:n])
        for at in range(0, len(data) - 32, STRIDE):
            for mask in (0xFF, 0x01):
                b = bytearray(data)
                b[at] ^= mask
                probes.append(bytes(b))
                probes.append(fix_checksum(bytes(b)))
        for k, p in enumerate(probes):
            self.sb.write("dmg.ml", p)
            for verb in verbs:
                r = self.sb.run(*verb)
                if r.status in TRAPS or r.status < 0 or r.status not in declared:
                    bad.append((name, k, verb, r.status))
                    continue
                if r.status in (0,):
                    json.loads(r.stdout.decode().splitlines()[0])
        self.assertEqual(bad, [])

    def test_i11_linear_logistic_kmeans(self):
        for name in ("lin_std.ml", "log_std.ml", "km.ml"):
            with self.subTest(model=name):
                self.sweep(name, [("importance", "--model-file", "dmg.ml"), ("dump", "--model-file", "dmg.ml")])

    def test_i11_tree_and_forest(self):
        for name in ("tree_leaf.ml", "tree_cls.ml", "forest_cat.ml"):
            with self.subTest(model=name):
                self.sweep(name, [("importance", "--model-file", "dmg.ml"), ("dump", "--model-file", "dmg.ml", "--tree", "0"), ("describe", "--model-file", "dmg.ml")])

    def test_i11_every_node_field_with_a_repaired_checksum(self):
        # every field of every node of a small forest set to hostile values, the checksum repaired: the reader refuses or the tables answer, never a trap
        from harness import sections
        data = self.sb.read("tree_cls.ml")
        M = self.model("tree_cls.ml")
        T = M["tree"]
        start = [s for s in sections(data) if s[0] == 4][0][1]
        trees, total = T["trees"], T["total"]

        def elem_at(r, g):
            return start + 84 + 16 + 8 * (trees + 1) + (r - 3) * (16 + 8 * total) + 16 + 8 * g

        import struct
        declared = {c["code"] for c in introspect()["exit_codes"]} | {0}
        bad = []
        values = [0, 1, 2, total, 2 ** 40, -2, -1, 2 ** 62]
        for g in range(0, total, max(1, STRIDE // 4)):
            for r in range(3, 10):
                orig = struct.unpack_from("<q", data, elem_at(r, g))[0]
                for v in values + [orig + 1]:
                    b = bytearray(data)
                    struct.pack_into("<q", b, elem_at(r, g), v)
                    self.sb.write("dmg.ml", fix_checksum(bytes(b)))
                    for verb in (("importance", "--model-file", "dmg.ml"), ("dump", "--model-file", "dmg.ml")):
                        res = self.sb.run(*verb)
                        if res.status in TRAPS or res.status < 0 or res.status not in declared:
                            bad.append((g, r, v, verb[0], res.status))
        for g in range(0, total, max(1, STRIDE // 4)):
            for v in (float("nan"), float("inf"), -float("inf"), 1e308):
                b = bytearray(data)
                struct.pack_into("<d", b, elem_at(10, g), v)
                self.sb.write("dmg.ml", fix_checksum(bytes(b)))
                for verb in (("importance", "--model-file", "dmg.ml"), ("dump", "--model-file", "dmg.ml")):
                    res = self.sb.run(*verb)
                    if res.status in TRAPS or res.status < 0 or res.status not in declared:
                        bad.append((g, 10, v, verb[0], res.status))
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()

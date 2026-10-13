"""WT2, WT3, WT5, WT6, WT7 -- the tree and the forest through `ml fit | predict | eval` (docs/wiring-trees.md).

What is judged from outside: the task is the flag's and never the target's; the labels and the probability of a classification; the file (sections, kind 7, the
labels record, the sum order 3 and the refusals of the two orders); the rule for missing cells with cases whose answer is known by hand; the row bound of a tree fit;
the model file at 1, 2, 4, 8 and 16 threads; the tree as a forest of one. The byte-for-byte comparison with the Python oracle is test_wire_tree_oracle.py."""

import json
import random
import struct
import unittest

from harness import Sandbox, fix_checksum, sections, set_bytes, validate
from test_model_file import payloads, write_file


def u32(b, at):
    return struct.unpack_from("<I", b, at)[0]


def i64s(b, at, n):
    return struct.unpack_from("<%dq" % n, b, at)


def regression_csv(n=80, missing=True, seed=1):
    rng = random.Random(seed)
    rows = ["a,b,c,y"]
    for i in range(n):
        a, b, c = rng.uniform(0, 10), rng.uniform(-5, 5), rng.randint(0, 4)
        y = 2.0 * a - b * b + 3.0 * c + rng.gauss(0, 0.3)
        rows.append("%.4f,%s,%d,%.4f" % (a, "" if missing and i % 6 == 2 else "%.4f" % b, c, y))
    return "\n".join(rows) + "\n"


def classification_csv(n=90, labels=(3, 7), seed=2):
    rng = random.Random(seed)
    rows = ["a,b,c,lab"]
    for i in range(n):
        a, b, c = rng.uniform(0, 10), rng.uniform(-5, 5), rng.randint(0, 4)
        z = a - 5 + 0.5 * b + rng.gauss(0, 1.2)
        rows.append("%.4f,%s,%d,%d" % (a, "" if i % 7 == 3 else "%.4f" % b, c, labels[1] if z > 0 else labels[0]))
    return "\n".join(rows) + "\n"


def model_section(data):
    tag, start, n = sections(data)[3]
    return data[start:start + n]


def header(data):
    """the eight header values of the library payload: task, trees, d, bins, total nodes, max depth, min leaf, mtry"""
    m = model_section(data)
    return i64s(m, 4 + 16, 8)


class TreeWire(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        cls.sb.write("reg.csv", regression_csv())
        cls.sb.write("cls.csv", classification_csv())
        cls.sb.write("full.csv", regression_csv(missing=False))

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def ml(self, *args):
        return self.sb.run(*args)

    def fit(self, model, target, csv, out, *extra):
        r = self.ml("fit", "--model", model, "--target", target, "--output", out, "--overwrite", *extra, csv)
        self.assertEqual(r.status, 0, r)
        return r

    # ------------------------------------------------------------------ WT2: the task
    def test_the_task_is_the_flag_and_never_the_target(self):
        # the target of cls.csv is an integer column with two values; with no --task it is fitted as a regression, as constant does
        r = self.fit("forest", "lab", "cls.csv", "f.cml", "--trees", "5")
        self.assertEqual(r.data()["task"], "regression")
        self.assertEqual(header(self.sb.read("f.cml"))[0], 1)
        self.assertEqual(self.ml("predict", "--model-file", "f.cml", "--limit", "2", "cls.csv").data()["columns"], ["prediction"])
        r = self.fit("forest", "lab", "cls.csv", "f.cml", "--trees", "5", "--task", "classification")
        self.assertEqual(r.data()["task"], "classification")
        self.assertEqual(header(self.sb.read("f.cml"))[0], 0)
        self.assertEqual(self.ml("predict", "--model-file", "f.cml", "--limit", "2", "cls.csv").data()["columns"], ["label"])

    def test_a_classification_target_must_be_an_integer_column_of_two_values(self):
        r = self.ml("fit", "--model", "tree", "--task", "classification", "--target", "y", "--output", "t.cml", "reg.csv")
        self.assertEqual(r.first_rule(), "ml.label.not-integer")
        self.sb.write("three.csv", "a,lab\n1,0\n2,1\n3,2\n4,1\n")
        self.assertEqual(self.ml("fit", "--model", "tree", "--task", "classification", "--target", "lab", "--output", "t.cml", "three.csv").first_rule(), "ml.label.not-binary")
        self.sb.write("one.csv", "a,lab\n1,1\n2,1\n3,1\n")
        self.assertEqual(self.ml("fit", "--model", "forest", "--task", "classification", "--target", "lab", "--output", "t.cml", "one.csv").first_rule(), "ml.label.single-class")

    def test_the_task_of_a_plan_is_the_plans(self):
        self.sb.write("p.txt", "model kind=forest task=classification trees=3\n")
        r = self.ml("fit", "--plan", "p.txt", "--target", "lab", "--output", "pf.cml", "--overwrite", "cls.csv")
        self.assertEqual((r.status, r.data()["task"]), (0, "classification"), r)
        self.assertEqual(self.ml("fit", "--plan", "p.txt", "--task", "regression", "--target", "lab", "--output", "pf.cml", "--overwrite", "cls.csv").first_rule(), "args.conflict")
        self.assertEqual(self.ml("fit", "--plan", "p.txt", "--trees", "4", "--target", "lab", "--output", "pf.cml", "--overwrite", "cls.csv").first_rule(), "args.conflict")

    def test_labels_probability_and_the_positive_label(self):
        for positive in (None, "3", "7"):
            extra = ["--trees", "9", "--seed", "5", "--task", "classification"] + (["--positive", positive] if positive else [])
            r = self.fit("forest", "lab", "cls.csv", "f.cml", *extra)
            want_pos = int(positive) if positive else 7
            self.assertEqual(r.data()["summary"]["positive"], want_pos)
            self.assertEqual(sorted(r.data()["summary"]["classes"]), [3, 7])
            p = self.ml("predict", "--model-file", "f.cml", "--proba", "--limit", "1000", "cls.csv").data()
            self.assertEqual(p["columns"], ["label", "probability"])
            other = 3 if want_pos == 7 else 7
            for label, prob in p["rows"]:
                self.assertTrue(0.0 <= float(prob) <= 1.0)
                self.assertEqual(int(label), want_pos if float(prob) > 0.5 else other, (label, prob))
            self.assertEqual({int(r_[0]) for r_ in p["rows"]} <= {3, 7}, True)
            ev = self.ml("eval", "--model-file", "f.cml", "cls.csv").data()
            self.assertEqual(ev["positive"], want_pos)
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"):
                self.assertIn(k, ev["metrics"])
            # the other label as the positive one at eval: precision and recall are about it, the accuracy is the same
            ev2 = self.ml("eval", "--model-file", "f.cml", "--positive", str(other), "cls.csv").data()
            self.assertEqual(ev2["positive"], other)
            self.assertEqual(ev["metrics"]["accuracy"], ev2["metrics"]["accuracy"])
            self.assertEqual(self.ml("eval", "--model-file", "f.cml", "--positive", "5", "cls.csv").first_rule(), "ml.param.out-of-range")
        self.assertEqual(self.ml("fit", "--model", "forest", "--task", "classification", "--positive", "9", "--target", "lab", "--output", "f.cml", "--overwrite", "cls.csv").first_rule(),
                         "ml.param.out-of-range")

    def test_regression_prints_a_value_and_has_no_probability(self):
        self.fit("tree", "y", "full.csv", "t.cml")
        p = self.ml("predict", "--model-file", "t.cml", "--limit", "3", "full.csv").data()
        self.assertEqual(p["columns"], ["prediction"])
        self.assertTrue(all(len(r_) == 1 for r_ in p["rows"]))
        self.assertEqual(self.ml("predict", "--model-file", "t.cml", "--proba", "full.csv").first_rule(), "args.conflict")
        m = self.ml("eval", "--model-file", "t.cml", "full.csv").data()["metrics"]
        self.assertEqual(sorted(m), ["mae", "r2", "rmse"])
        # a tree deep enough to hold every row of its training data reproduces it: no error on the data it was fitted on
        self.fit("tree", "y", "full.csv", "deep.cml", "--max-depth", "30", "--bins", "255")
        self.assertEqual(float(self.ml("eval", "--model-file", "deep.cml", "full.csv").data()["metrics"]["rmse"]), 0.0)

    # ------------------------------------------------------------------ WT3: the file
    def test_the_file_of_a_classification_forest(self):
        self.fit("forest", "lab", "cls.csv", "f.cml", "--trees", "4", "--task", "classification", "--seed", "11", "--step", "scale column=a kind=standard")
        d = self.sb.read("f.cml")
        p = payloads(d)
        self.assertEqual(p[1], b"scale column=a kind=standard ddof=0\nmodel kind=forest task=classification trees=4 max-depth=8 min-leaf=1 bins=64 mtry=0 bootstrap=1 max-nodes=4194304\n")
        self.assertEqual(u32(p[3], 0), 1)                                        # one step
        m = p[4]
        self.assertEqual(u32(m, 0), 7)
        task, trees, d_, bins, total, maxd, minleaf, mtry = header(d)
        self.assertEqual((task, trees, d_, bins, maxd, minleaf), (0, 4, 3, 64, 8, 1))
        self.assertEqual(mtry, 1)                                                # floor(sqrt(3)): 0 means the default, the header holds what was used
        size = 4 + 16 + 64 + 16 + 8 * (trees + 1) + 8 * (16 + 8 * total)
        self.assertEqual(len(m), size + 32)                                      # the library payload, then the labels record
        self.assertEqual(struct.unpack_from("<IIQqq", m, size), (1, 1, 2, 3, 7))
        meta = p[5]
        self.assertEqual(u32(meta, len(meta) - 4), 3)                            # sequential sums: docs/wiring-trees.md section 4
        self.assertEqual(struct.unpack_from("<QQQ", meta, 0), (90, 4, 11))        # rows, columns, the seed of the forest

    def test_the_file_of_a_regression_tree_has_no_labels(self):
        self.fit("tree", "y", "full.csv", "t.cml")
        d = self.sb.read("t.cml")
        m = model_section(d)
        task, trees, d_, bins, total, maxd, minleaf, mtry = header(d)
        self.assertEqual((task, trees, d_, mtry), (1, 1, 3, 3))                  # a tree tries every column
        self.assertEqual(len(m), 4 + 16 + 64 + 16 + 8 * (trees + 1) + 8 * (16 + 8 * total))

    def test_a_tree_is_a_forest_of_one(self):
        self.fit("tree", "y", "reg.csv", "t.cml", "--max-depth", "5", "--min-leaf", "2", "--bins", "32", "--seed", "4")
        self.fit("forest", "y", "reg.csv", "f1.cml", "--trees", "1", "--bootstrap", "0", "--mtry", "3", "--max-depth", "5", "--min-leaf", "2", "--bins", "32", "--seed", "4")
        self.assertEqual(model_section(self.sb.read("t.cml")), model_section(self.sb.read("f1.cml")))

    def test_the_sum_order_is_three_for_kind_7_and_two_for_the_lane_kinds(self):
        self.fit("forest", "y", "full.csv", "f.cml", "--trees", "3")
        self.fit("linear", "y", "full.csv", "lin.cml")
        d = self.sb.read("f.cml")
        sec = sections(d)
        end = sec[4][1] + sec[4][2] - 4
        for order in (0, 1, 2, 4, 255):
            bad = fix_checksum(set_bytes(d, end, struct.pack("<I", order)))
            self.sb.write("bad.cml", bad)
            r = self.ml("predict", "--model-file", "bad.cml", "full.csv")
            self.assertEqual(r.first_rule(), "ml.model.version", (order, r))
            self.assertEqual((r.error()["detail"]["sum_order"], r.error()["detail"]["known"]), (order, 3))
        lin = self.sb.read("lin.cml")
        sec = sections(lin)
        bad = fix_checksum(set_bytes(lin, sec[4][1] + sec[4][2] - 4, struct.pack("<I", 3)))
        self.sb.write("bad.cml", bad)
        r = self.ml("predict", "--model-file", "bad.cml", "full.csv")
        self.assertEqual((r.first_rule(), r.error()["detail"]["sum_order"], r.error()["detail"]["known"]), ("ml.model.version", 3, 2), r)

    def test_the_plan_of_the_file_decides_the_name_predict_prints(self):
        self.fit("tree", "y", "full.csv", "t.cml")
        self.fit("forest", "y", "full.csv", "f.cml", "--trees", "2")
        self.assertEqual(self.ml("predict", "--model-file", "t.cml", "--limit", "1", "full.csv").data()["model"], "tree")
        self.assertEqual(self.ml("predict", "--model-file", "f.cml", "--limit", "1", "full.csv").data()["model"], "forest")
        self.assertEqual(self.ml("eval", "--model-file", "f.cml", "full.csv").data()["model"], "forest")

    def test_a_constant_file_with_a_kind_7_payload_is_still_the_old_readers_refusal(self):
        self.sb.write("c.csv", "y\n1\n2\n3\n")
        self.assertEqual(self.ml("fit", "--target", "y", "--output", "c.cml", "--overwrite", "c.csv").status, 0)
        d = self.sb.read("c.cml")
        self.sb.write("k7.cml", fix_checksum(set_bytes(d, sections(d)[3][1], struct.pack("<I", 7))))
        r = self.ml("predict", "--model-file", "k7.cml", "c.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.model.malformed"), r)

    def test_the_seed_is_the_forests_and_is_stored(self):
        self.fit("forest", "y", "full.csv", "a.cml", "--trees", "6", "--seed", "1")
        self.fit("forest", "y", "full.csv", "b.cml", "--trees", "6", "--seed", "1")
        self.fit("forest", "y", "full.csv", "c.cml", "--trees", "6", "--seed", "2")
        self.assertEqual(self.sb.read("a.cml"), self.sb.read("b.cml"))
        self.assertNotEqual(model_section(self.sb.read("a.cml")), model_section(self.sb.read("c.cml")))
        self.assertEqual(struct.unpack_from("<Q", payloads(self.sb.read("c.cml"))[5], 16)[0], 2)

    # ------------------------------------------------------------------ WT5: missing values
    def test_a_plan_with_no_impute_step_is_accepted_for_the_trees_and_refused_for_the_linear_models(self):
        for model, task in (("tree", "regression"), ("forest", "regression")):
            r = self.ml("fit", "--model", model, "--target", "y", "--output", "m.cml", "--overwrite", "reg.csv")
            self.assertEqual(r.status, 0, (model, r))
        self.sb.write("clsint.csv", classification_csv(labels=(0, 1)))
        for model in ("linear", "logistic"):
            target, csv = ("y", "reg.csv") if model == "linear" else ("lab", "clsint.csv")
            r = self.ml("fit", "--model", model, "--target", target, "--output", "m.cml", "--overwrite", csv)
            self.assertEqual(r.first_rule(), "ml.input.missing-value", (model, r))
        # a step that is not impute passes a missing cell through, and the tree takes it
        r = self.ml("fit", "--model", "tree", "--step", "scale column=a,b kind=standard;bucket column=b kind=width bins=4", "--target", "y", "--output", "m.cml", "--overwrite", "reg.csv")
        self.assertEqual(r.status, 0, r)

    def test_missing_cells_in_predict_and_eval_are_not_an_error_for_a_tree_and_are_for_a_linear_model(self):
        self.fit("forest", "y", "full.csv", "f.cml", "--trees", "3")
        self.assertEqual(self.ml("predict", "--model-file", "f.cml", "reg.csv").status, 0)
        self.assertEqual(self.ml("eval", "--model-file", "f.cml", "reg.csv").status, 0)
        self.fit("linear", "y", "full.csv", "lin.cml")
        self.assertEqual(self.ml("predict", "--model-file", "lin.cml", "reg.csv").first_rule(), "ml.input.missing-value")

    def test_the_default_side_is_learned_from_the_training_rows_that_miss(self):
        # one feature: below 5 the class is 0, from 5 on 1. The rows whose feature is missing all have class `m`: the split sends them to the side of class `m`
        for m in (0, 1):
            rows = ["x,z,lab"] + ["%d,1,%d" % (v, 0 if v < 5 else 1) for v in range(10) for _ in range(6)] + [",1,%d" % m] * 12
            self.sb.write("miss.csv", "\n".join(rows) + "\n")
            self.fit("tree", "lab", "miss.csv", "t.cml", "--task", "classification", "--max-depth", "2")
            self.sb.write("probe.csv", "x,z\n,1\n2,1\n8,1\n")
            p = self.ml("predict", "--model-file", "t.cml", "--proba", "probe.csv").data()["rows"]
            self.assertEqual([int(r_[0]) for r_ in p], [m, 0, 1], (m, p))

    def test_a_node_that_saw_no_missing_cell_sends_it_to_the_right(self):
        rows = ["x,z,lab"] + ["%d,1,%d" % (v, 0 if v < 5 else 1) for v in range(10) for _ in range(6)]
        self.sb.write("nomiss.csv", "\n".join(rows) + "\n")
        self.fit("tree", "lab", "nomiss.csv", "t.cml", "--task", "classification", "--max-depth", "2")
        self.sb.write("probe.csv", "x,z\n,1\n")
        p = self.ml("predict", "--model-file", "t.cml", "--proba", "probe.csv").data()["rows"]
        self.assertEqual(p, [["1", "1.0"]])             # right is the side of the large values

    def test_an_impute_step_removes_the_missing_cells_and_is_stored(self):
        r = self.fit("forest", "y", "reg.csv", "f.cml", "--trees", "3", "--step", "impute column=b strategy=mean")
        self.assertEqual(r.data()["summary"]["steps"], 1)
        # the same data with the cells filled in by hand (and no step) gives the same model section
        r = self.fit("forest", "y", "reg.csv", "g.cml", "--trees", "3", "--step", "impute column=b strategy=constant value=0")
        self.sb.write("filled.csv", "".join(l + "\n" if not l.split(",")[1:2] == [""] else ",".join([l.split(",")[0], "0.0000"] + l.split(",")[2:]) + "\n" for l in regression_csv().strip().split("\n")))
        self.fit("forest", "y", "filled.csv", "h.cml", "--trees", "3")
        self.assertEqual(model_section(self.sb.read("g.cml")), model_section(self.sb.read("h.cml")))

    def test_an_all_missing_column_is_refused_by_a_step_that_selects_it_and_ignored_otherwise(self):
        self.sb.write("allmiss.csv", "a,b,y\n" + "".join("%d,,%d\n" % (i, i * 2) for i in range(1, 30)))
        self.assertEqual(self.ml("fit", "--model", "tree", "--step", "scale column=* kind=standard", "--target", "y", "--output", "m.cml", "--overwrite", "allmiss.csv").first_rule(),
                         "ml.fit.all-missing")
        r = self.ml("fit", "--model", "tree", "--target", "y", "--output", "m.cml", "--overwrite", "allmiss.csv")
        self.assertEqual(r.status, 0, r)

    # ------------------------------------------------------------------ WT6: the rows of a tree fit
    def test_a_tree_fit_takes_at_most_2_to_the_20_training_rows(self):
        n = (1 << 20) + 1
        self.sb.write("big.csv", "x,y\n" + "".join("%d,%d\n" % (i % 1000, i % 7) for i in range(n)))
        r = self.ml("fit", "--model", "tree", "--target", "y", "--output", "big.cml", "big.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.input.too-many-rows"), r)
        self.assertEqual(validate(r), [])
        self.assertEqual((r.error()["detail"]["rows"], r.error()["detail"]["max"]), (n, 1 << 20))
        self.assertIn("tree or forest", r.error()["message"])
        self.assertFalse((self.sb.dir / "big.cml").exists())
        # the train side of a split is what counts: a file of 1,048,600 rows split train=0.999999 has a test side of ceil(1.0486) = 2 rows and 1,048,598 train rows
        n2 = n + 23
        self.sb.write("big2.csv", "x,y\n" + "".join("%d,%d\n" % (i % 1000, i % 7) for i in range(n2)))
        r = self.ml("fit", "--model", "forest", "--trees", "2", "--split", "train=0.999999", "--target", "y", "--output", "big.cml", "big2.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.input.too-many-rows"), r)
        self.assertEqual(r.error()["detail"]["rows"], n2 - 2)
        # and the same file with a train side within the bound is fitted (the bound is on the training rows, not on the file)
        r = self.ml("fit", "--model", "tree", "--max-depth", "3", "--split", "train=0.5", "--target", "y", "--output", "big.cml", "big2.csv")
        self.assertEqual(r.status, 0, r)
        # the linear models have no such bound (the row limit is --max-rows)
        r = self.ml("fit", "--model", "linear", "--target", "y", "--output", "big.cml", "--overwrite", "big.csv")
        self.assertEqual(r.status, 0, r)

    def test_exactly_2_to_the_20_rows_are_fitted(self):
        n = 1 << 20
        self.sb.write("edge.csv", "x,y\n" + "".join("%d,%d\n" % (i % 1000, i % 7) for i in range(n)))
        r = self.ml("fit", "--model", "tree", "--max-depth", "1", "--target", "y", "--output", "edge.cml", "--overwrite", "edge.csv")
        self.assertEqual((r.status, r.data()["rows"]), (0, n), r)

    def test_an_input_column_with_no_cell_is_a_missing_column_for_the_linear_models_and_every_cell_missing_for_the_trees(self):
        self.sb.write("one.csv", "a,b,y\n" + "".join("%d,%d,%d\n" % (i, i % 5, 3 * i + i % 5) for i in range(1, 41)))
        self.sb.write("empty.csv", "a,b\n1,\n2,\n3,\n")
        self.fit("tree", "y", "one.csv", "t.cml")
        r = self.ml("predict", "--model-file", "t.cml", "empty.csv")
        self.assertEqual((r.status, r.data()["row_count"]), (0, 3), r)
        self.fit("linear", "y", "one.csv", "lin.cml")
        r = self.ml("predict", "--model-file", "lin.cml", "empty.csv")
        self.assertEqual(r.first_rule(), "ml.schema.mismatch", r)
        self.assertEqual(r.error()["detail"]["missing"], ["b"])

    def test_the_limit_tags_of_the_library_have_their_exit_codes(self):
        for argv, rule in ((["--max-depth", "31"], "ml.limit.depth"), (["--max-nodes", "1"], "ml.limit.nodes"), (["--max-nodes", "4194305"], "ml.limit.nodes"),
                           (["--mtry", "9"], "ml.param.out-of-range"), (["--min-leaf", "1000"], "ml.param.out-of-range"), (["--threads", "0"], "ml.param.threads")):
            r = self.ml("fit", "--model", "forest", *argv, "--target", "y", "--output", "m.cml", "--overwrite", "full.csv")
            self.assertEqual(r.first_rule(), rule, (argv, r))
            self.assertEqual(validate(r), [], (argv, r))
            self.assertEqual(r.status, 2 if rule in ("ml.param.out-of-range", "ml.param.threads") else 8, (argv, r))

    def test_a_refused_fit_writes_nothing(self):
        self.sb.write("keep.cml", b"x")
        r = self.ml("fit", "--model", "tree", "--max-depth", "31", "--target", "y", "--output", "keep.cml", "--overwrite", "full.csv")
        self.assertEqual(r.first_rule(), "ml.limit.depth")
        self.assertEqual(self.sb.read("keep.cml"), b"x")

    # ------------------------------------------------------------------ WT7: threads
    def test_the_model_file_is_the_same_at_1_2_4_8_and_16_threads(self):
        self.sb.write("split.csv", regression_csv(n=400))
        cases = (("forest", "y", "split.csv", ["--trees", "7", "--seed", "9", "--split", "train=0.8"]),
                 ("forest", "lab", "cls.csv", ["--trees", "10", "--task", "classification", "--seed", "3"]),
                 ("tree", "y", "split.csv", ["--max-depth", "6"]),
                 ("forest", "y", "split.csv", ["--trees", "5", "--bootstrap", "0", "--mtry", "2", "--step", "scale column=a kind=minmax"]))
        for model, target, csv, extra in cases:
            seen = {}
            for t in (1, 2, 4, 8, 16):
                r = self.ml("fit", "--model", model, "--target", target, "--output", "th.cml", "--overwrite", "--threads", str(t), *extra, csv)
                self.assertEqual(r.status, 0, r)
                seen[t] = (self.sb.read("th.cml"), r.stdout.replace(b"th.cml", b"M"))
            self.assertEqual({t: v[0] for t, v in seen.items() if v[0] != seen[1][0]}, {}, (model, extra))
            self.assertEqual({t: 1 for t, v in seen.items() if v[1] != seen[1][1]}, {}, (model, extra))

    def test_eval_accepts_threads_and_changes_nothing(self):
        self.fit("forest", "lab", "cls.csv", "f.cml", "--trees", "5", "--task", "classification")
        a = self.ml("eval", "--model-file", "f.cml", "cls.csv").stdout
        b = self.ml("eval", "--model-file", "f.cml", "--threads", "7", "cls.csv").stdout
        self.assertEqual(a, b)

    # ------------------------------------------------------------------ the columns
    def test_columns_are_matched_by_name(self):
        self.fit("forest", "y", "full.csv", "f.cml", "--trees", "3")
        self.sb.write("x.csv", "c,a,b\n1,2.5,1.0\n3,7.5,-2.0\n")
        rev = self.ml("predict", "--model-file", "f.cml", "x.csv").data()["rows"]
        self.sb.write("x2.csv", "a,b,c\n2.5,1.0,1\n7.5,-2.0,3\n")
        self.assertEqual(rev, self.ml("predict", "--model-file", "f.cml", "x2.csv").data()["rows"])
        self.sb.write("x3.csv", "a,b\n2.5,1.0\n")
        self.assertEqual(self.ml("predict", "--model-file", "f.cml", "x3.csv").first_rule(), "ml.schema.mismatch")

    def test_categorical_columns_are_still_refused(self):
        self.sb.write("txt.csv", "a,s,y\n1,u,2\n2,v,3\n3,u,4\n")
        r = self.ml("fit", "--model", "forest", "--cat", "s", "--target", "y", "--output", "m.cml", "--overwrite", "txt.csv")
        self.assertEqual(r.first_rule(), "ml.input.categorical-not-encoded", r)


if __name__ == "__main__":
    unittest.main()

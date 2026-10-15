"""GB2, GB3, GB6 (the behaviour) -- boosted trees through `ml fit | predict | eval` (docs/models-boost.md).

What is judged from outside: the task is the flag's; the labels, the probability and the positive label; the file (kind 8, the header, the labels record, the sum order 3
and the refusals of the other orders); the scores by hand (a round of a known dataset); the rule for missing cells; early stopping (refused without --split, stopping on a
noisy target, never reading the held-out rows); the row sample (seeded, different with the seed); the limits and the range of every number; the model file at 1, 2, 4, 8 and
16 threads; the refusal to write a file `predict` cannot read. The byte-for-byte comparison with the Python oracle is test_wire_boost_oracle.py."""

import random
import struct
import unittest

from harness import Sandbox, fix_checksum, sections, set_bytes, validate
from test_model_file import payloads
from test_wire_trees import classification_csv, regression_csv


def i64s(b, at, n):
    return struct.unpack_from("<%dq" % n, b, at)


def f64s(b, at, n):
    return struct.unpack_from("<%dd" % n, b, at)


def u32(b, at):
    return struct.unpack_from("<I", b, at)[0]


def model_section(data):
    tag, start, n = sections(data)[3]
    return data[start:start + n]


def header(data):
    """the ten header values: task, rounds, d, bins, total nodes, max depth, min leaf, early stopping, rounds requested, subsample K"""
    return i64s(model_section(data), 20, 10)


def trees_of(data):
    """(base, learning rate, l2, [[(feat, bin, dleft, left, right, val) of each node] of each tree])"""
    m = model_section(data)
    task, rounds, d, bins, total = i64s(m, 20, 5)
    base, lr, l2 = f64s(m, 116, 3)
    offs = i64s(m, 156, rounds + 1)
    rec = lambda r: 156 + 8 * (rounds + 1) + (r - 4) * (16 + 8 * total) + 16  # noqa: E731
    cols = [i64s(m, rec(r), total) for r in range(4, 9)]
    vals = f64s(m, rec(9), total)
    out = []
    for t in range(rounds):
        out.append([tuple(c[g] for c in cols) + (vals[g],) for g in range(offs[t], offs[t + 1])])
    return base, lr, l2, out


class BoostWire(unittest.TestCase):
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

    def fit(self, target, csv, out, *extra):
        r = self.ml("fit", "--model", "boost", "--target", target, "--output", out, "--overwrite", *extra, csv)
        self.assertEqual(r.status, 0, r)
        return r

    # ------------------------------------------------------------------ the task
    def test_the_task_is_the_flag_and_never_the_target(self):
        r = self.fit("lab", "cls.csv", "b.cml", "--rounds", "5")
        self.assertEqual(r.data()["task"], "regression")
        self.assertEqual(header(self.sb.read("b.cml"))[0], 1)
        self.assertEqual(self.ml("predict", "--model-file", "b.cml", "--limit", "2", "cls.csv").data()["columns"], ["prediction"])
        r = self.fit("lab", "cls.csv", "b.cml", "--rounds", "5", "--task", "classification")
        self.assertEqual(r.data()["task"], "classification")
        self.assertEqual(header(self.sb.read("b.cml"))[0], 0)
        self.assertEqual(self.ml("predict", "--model-file", "b.cml", "--limit", "2", "cls.csv").data()["columns"], ["label"])

    def test_a_classification_target_must_be_an_integer_column_of_two_values(self):
        r = self.ml("fit", "--model", "boost", "--task", "classification", "--target", "y", "--output", "t.cml", "reg.csv")
        self.assertEqual(r.first_rule(), "ml.label.not-integer")
        self.sb.write("three.csv", "a,lab\n1,0\n2,1\n3,2\n4,1\n")
        self.assertEqual(self.ml("fit", "--model", "boost", "--task", "classification", "--target", "lab", "--output", "t.cml", "three.csv").first_rule(), "ml.label.not-binary")
        self.sb.write("one.csv", "a,lab\n1,1\n2,1\n3,1\n")
        self.assertEqual(self.ml("fit", "--model", "boost", "--task", "classification", "--target", "lab", "--output", "t.cml", "one.csv").first_rule(), "ml.label.single-class")

    def test_labels_probability_and_the_positive_label(self):
        for positive in (None, "3", "7"):
            extra = ["--rounds", "20", "--seed", "5", "--task", "classification", "--min-leaf", "2"] + (["--positive", positive] if positive else [])
            r = self.fit("lab", "cls.csv", "b.cml", *extra)
            want_pos = int(positive) if positive else 7
            self.assertEqual(r.data()["summary"]["positive"], want_pos)
            p = self.ml("predict", "--model-file", "b.cml", "--proba", "--limit", "1000", "cls.csv").data()
            self.assertEqual(p["columns"], ["label", "probability"])
            other = 3 if want_pos == 7 else 7
            for label, prob in p["rows"]:
                self.assertTrue(0.0 <= float(prob) <= 1.0)
                self.assertEqual(int(label), want_pos if float(prob) > 0.5 else other, (label, prob))
            ev = self.ml("eval", "--model-file", "b.cml", "cls.csv").data()
            self.assertEqual(ev["positive"], want_pos)
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"):
                self.assertIn(k, ev["metrics"])
            ev2 = self.ml("eval", "--model-file", "b.cml", "--positive", str(other), "cls.csv").data()
            self.assertEqual(ev["metrics"]["accuracy"], ev2["metrics"]["accuracy"])

    def test_regression_prints_a_value_and_has_no_probability(self):
        self.fit("y", "full.csv", "t.cml")
        p = self.ml("predict", "--model-file", "t.cml", "--limit", "3", "full.csv").data()
        self.assertEqual(p["columns"], ["prediction"])
        self.assertEqual(self.ml("predict", "--model-file", "t.cml", "--proba", "full.csv").first_rule(), "args.conflict")
        self.assertEqual(sorted(self.ml("eval", "--model-file", "t.cml", "full.csv").data()["metrics"]), ["mae", "r2", "rmse"])

    def test_boosting_learns_more_than_the_mean(self):
        # the target is a function of the features: the error falls as rounds are added (the learning-rate 0.5 makes it fast)
        rmse = []
        for rounds in (1, 5, 40):
            self.fit("y", "full.csv", "t.cml", "--rounds", str(rounds), "--learning-rate", "0.5", "--min-leaf", "2")
            rmse.append(float(self.ml("eval", "--model-file", "t.cml", "full.csv").data()["metrics"]["rmse"]))
        self.assertTrue(rmse[0] > rmse[1] > rmse[2], rmse)

    # ------------------------------------------------------------------ the file
    def test_the_file_of_a_classification(self):
        self.fit("lab", "cls.csv", "b.cml", "--rounds", "4", "--task", "classification", "--seed", "11", "--min-leaf", "1", "--step", "scale column=a kind=standard")
        d = self.sb.read("b.cml")
        p = payloads(d)
        self.assertEqual(p[1], b"scale column=a kind=standard ddof=0\nmodel kind=boost task=classification rounds=4 learning-rate=0.1 max-depth=4 min-leaf=1 bins=64 l2=1.0 subsample=1.0 "
                               b"early-stopping=0 max-nodes=4194304\n")
        self.assertEqual(u32(p[3], 0), 1)
        m = p[4]
        self.assertEqual(u32(m, 0), 8)
        task, rounds, d_, bins, total, maxd, minleaf, early, requested, subk = header(d)
        self.assertEqual((task, rounds, d_, bins, maxd, minleaf, early, requested, subk), (0, 4, 3, 64, 4, 1, 0, 4, 1000000))
        size = 156 + 8 * (rounds + 1) + 6 * (16 + 8 * total)
        self.assertEqual(len(m), size + 32)
        self.assertEqual(struct.unpack_from("<IIQqq", m, size), (1, 1, 2, 3, 7))
        base, lr, l2, trees = trees_of(d)
        self.assertEqual((lr, l2), (0.1, 1.0))
        self.assertEqual(len(trees), 4)
        n7 = sum(1 for line in classification_csv().splitlines()[1:] if line.endswith(",7"))
        import math
        self.assertEqual(base, math.log(n7 / (90 - n7)))
        meta = p[5]
        self.assertEqual(u32(meta, len(meta) - 4), 3)
        self.assertEqual(struct.unpack_from("<QQQ", meta, 0), (90, 4, 11))

    def test_the_file_of_a_regression_has_no_labels(self):
        self.fit("y", "full.csv", "t.cml", "--rounds", "3")
        d = self.sb.read("t.cml")
        task, rounds, d_, bins, total = header(d)[:5]
        self.assertEqual((task, rounds, d_), (1, 3, 3))
        self.assertEqual(len(model_section(d)), 156 + 8 * (rounds + 1) + 6 * (16 + 8 * total))
        r = self.fit("y", "full.csv", "t2.cml", "--rounds", "3")
        h = header(self.sb.read("t2.cml"))
        self.assertEqual((r.data()["summary"]["boost"]["rounds"], r.data()["summary"]["boost"]["nodes"], r.data()["summary"]["boost"]["bins"]), (h[1], h[4], h[3]))

    def test_the_sum_order_is_three_for_kind_8(self):
        self.fit("y", "full.csv", "b.cml", "--rounds", "3")
        d = self.sb.read("b.cml")
        sec = sections(d)
        end = sec[4][1] + sec[4][2] - 4
        for order in (0, 1, 2, 4, 255):
            bad = fix_checksum(set_bytes(d, end, struct.pack("<I", order)))
            self.sb.write("bad.cml", bad)
            r = self.ml("predict", "--model-file", "bad.cml", "full.csv")
            self.assertEqual(r.first_rule(), "ml.model.version", (order, r))
            self.assertEqual((r.error()["detail"]["sum_order"], r.error()["detail"]["known"]), (order, 3))

    def test_the_plan_of_the_file_decides_the_name_predict_prints(self):
        self.fit("y", "full.csv", "b.cml", "--rounds", "2")
        self.assertEqual(self.ml("predict", "--model-file", "b.cml", "--limit", "1", "full.csv").data()["model"], "boost")
        self.assertEqual(self.ml("eval", "--model-file", "b.cml", "full.csv").data()["model"], "boost")

    def test_a_constant_file_with_a_kind_8_payload_is_refused(self):
        self.sb.write("c.csv", "y\n1\n2\n3\n")
        self.assertEqual(self.ml("fit", "--target", "y", "--output", "c.cml", "--overwrite", "c.csv").status, 0)
        d = self.sb.read("c.cml")
        self.sb.write("k8.cml", fix_checksum(set_bytes(d, sections(d)[3][1], struct.pack("<I", 8))))
        r = self.ml("predict", "--model-file", "k8.cml", "c.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.model.malformed"), r)

    def test_describe_says_the_number_of_trees(self):
        rng = random.Random(8)
        rows = ["a,b,y"] + ["%.3f,%.3f,%.3f" % (a, b, a + rng.gauss(0, 6)) for a, b in ((rng.uniform(0, 5), rng.uniform(0, 5)) for _ in range(300))]
        self.sb.write("noisy2.csv", "\n".join(rows) + "\n")
        self.fit("y", "noisy2.csv", "b.cml", "--rounds", "300", "--learning-rate", "0.4", "--early-stopping", "3", "--split", "train=0.8", "--seed", "2", "--min-leaf", "5")
        h = header(self.sb.read("b.cml"))
        doc = self.ml("describe", "--model-file", "b.cml").data()
        self.assertEqual((doc["model"], doc["kind"], doc["file"]["sum_order"]), ("boost", 8, 3))
        self.assertLess(h[1], 300)
        self.assertEqual(doc["ensemble"], {"trees": h[1], "nodes": h[4], "rounds_requested": 300})

    # ------------------------------------------------------------------ the scores, by hand (GB3)
    def test_one_round_is_the_base_plus_the_shrunk_newton_step(self):
        # y = 0 for x < 5 and 10 for x >= 5, six rows of each: one depth-1 tree with no penalty separates them; with a learning rate of 0.5 the prediction is the mean
        # (5) plus half the step to each side's mean: 5 - 0.5 * 5 = 2.5 and 5 + 0.5 * 5 = 7.5
        rows = ["x,y"] + ["%d,%d" % (v, 0 if v < 5 else 10) for v in range(10) for _ in range(6)]
        self.sb.write("step.csv", "\n".join(rows) + "\n")
        self.fit("y", "step.csv", "b.cml", "--rounds", "1", "--learning-rate", "0.5", "--max-depth", "1", "--l2", "0", "--min-leaf", "1")
        base, lr, l2, trees = trees_of(self.sb.read("b.cml"))
        self.assertEqual((base, lr, l2), (5.0, 0.5, 0.0))
        self.assertEqual(len(trees), 1)
        self.assertEqual(len(trees[0]), 3)
        self.sb.write("probe.csv", "x\n0\n4\n5\n9\n")
        p = self.ml("predict", "--model-file", "b.cml", "probe.csv").data()["rows"]
        self.assertEqual([float(r_[0]) for r_ in p], [2.5, 2.5, 7.5, 7.5])
        # the penalty shrinks the step: 30 rows on a side, sum of gradients -150, hessian 30: -(-150 * -1)... the leaf is -G / (H + l2) with G = -150 (below) so 150 / 31
        self.fit("y", "step.csv", "b.cml", "--rounds", "1", "--learning-rate", "1.0", "--max-depth", "1", "--l2", "1", "--min-leaf", "1")
        p = self.ml("predict", "--model-file", "b.cml", "probe.csv").data()["rows"]
        self.assertEqual([float(r_[0]) for r_ in p], [5.0 + (-150.0) / 31.0 * -1.0 * -1.0 if False else 5.0 - 150.0 / 31.0] * 2 + [5.0 + 150.0 / 31.0] * 2)

    def test_a_constant_feature_leaves_the_mean(self):
        self.sb.write("const.csv", "x,y\n" + "".join("1,%d\n" % v for v in (2, 4, 6, 8, 10, 12)))
        self.fit("y", "const.csv", "b.cml", "--rounds", "5", "--min-leaf", "1")
        base, lr, l2, trees = trees_of(self.sb.read("b.cml"))
        self.assertEqual(base, 7.0)
        for t in trees:
            self.assertEqual(len(t), 1)
            self.assertEqual(t[0][5], 0.0)
        self.sb.write("p1.csv", "x\n1\n3\n")
        self.assertEqual([r_[0] for r_ in self.ml("predict", "--model-file", "b.cml", "p1.csv").data()["rows"]], ["7.0", "7.0"])

    def test_a_perfectly_separable_classification_is_not_refused_at_any_penalty(self):
        rows = ["x,lab"] + ["%d,%d" % (v, 0 if v < 5 else 1) for v in range(10) for _ in range(6)]
        self.sb.write("sep.csv", "\n".join(rows) + "\n")
        for l2 in ("0", "1"):
            self.fit("lab", "sep.csv", "b.cml", "--task", "classification", "--rounds", "60", "--learning-rate", "1.0", "--l2", l2, "--min-leaf", "1", "--max-depth", "1")
            p = self.ml("predict", "--model-file", "b.cml", "--proba", "--limit", "100", "sep.csv").data()["rows"]
            for (lab, prob), v in zip(p, [v for v in range(10) for _ in range(6)]):
                self.assertEqual(int(lab), 0 if v < 5 else 1)
                self.assertTrue(0.0 <= float(prob) <= 1.0)

    # ------------------------------------------------------------------ missing values
    def test_a_missing_cell_is_accepted_in_fit_predict_and_eval(self):
        self.fit("y", "reg.csv", "b.cml", "--rounds", "10")
        self.assertEqual(self.ml("predict", "--model-file", "b.cml", "reg.csv").status, 0)
        self.assertEqual(self.ml("eval", "--model-file", "b.cml", "reg.csv").status, 0)
        # a plan that imputes gives the model of the filled data
        self.fit("y", "reg.csv", "g.cml", "--rounds", "3", "--step", "impute column=b strategy=constant value=0")
        self.sb.write("filled.csv", "".join(l + "\n" if not l.split(",")[1:2] == [""] else ",".join([l.split(",")[0], "0.0000"] + l.split(",")[2:]) + "\n" for l in regression_csv().strip().split("\n")))
        self.fit("y", "filled.csv", "h.cml", "--rounds", "3")
        self.assertEqual(model_section(self.sb.read("g.cml")), model_section(self.sb.read("h.cml")))

    def test_the_default_side_is_learned_from_the_training_rows_that_miss(self):
        for m in (0, 1):
            rows = ["x,z,lab"] + ["%d,1,%d" % (v, 0 if v < 5 else 1) for v in range(10) for _ in range(6)] + [",1,%d" % m] * 12
            self.sb.write("miss.csv", "\n".join(rows) + "\n")
            self.fit("lab", "miss.csv", "b.cml", "--task", "classification", "--max-depth", "2", "--rounds", "30", "--min-leaf", "1", "--learning-rate", "0.5")
            self.sb.write("probe.csv", "x,z\n,1\n2,1\n8,1\n")
            p = self.ml("predict", "--model-file", "b.cml", "--proba", "probe.csv").data()["rows"]
            self.assertEqual([int(r_[0]) for r_ in p], [m, 0, 1], (m, p))

    # ------------------------------------------------------------------ the row sample and the seed
    def test_the_subsample_is_seeded(self):
        self.sb.write("s.csv", regression_csv(n=300, missing=False))
        base = ["--rounds", "8", "--min-leaf", "3"]
        self.fit("y", "s.csv", "a.cml", *base, "--subsample", "0.5", "--seed", "1")
        self.fit("y", "s.csv", "b.cml", *base, "--subsample", "0.5", "--seed", "1")
        self.fit("y", "s.csv", "c.cml", *base, "--subsample", "0.5", "--seed", "2")
        self.fit("y", "s.csv", "d.cml", *base, "--subsample", "1.0", "--seed", "1")
        self.fit("y", "s.csv", "e.cml", *base, "--seed", "1")
        self.assertEqual(self.sb.read("a.cml"), self.sb.read("b.cml"))
        self.assertNotEqual(model_section(self.sb.read("a.cml")), model_section(self.sb.read("c.cml")))
        self.assertNotEqual(model_section(self.sb.read("a.cml")), model_section(self.sb.read("d.cml")))
        # subsample 1.0 is no subsample, and without a subsample the seed changes nothing
        self.assertEqual(model_section(self.sb.read("d.cml")), model_section(self.sb.read("e.cml")))
        self.fit("y", "s.csv", "f.cml", *base, "--seed", "9")
        self.assertEqual(model_section(self.sb.read("e.cml")), model_section(self.sb.read("f.cml")))
        self.assertEqual(header(self.sb.read("a.cml"))[9], 500000)

    # ------------------------------------------------------------------ early stopping
    def test_early_stopping_is_refused_without_a_split(self):
        r = self.ml("fit", "--model", "boost", "--early-stopping", "3", "--target", "y", "--output", "b.cml", "full.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "args.conflict"), r)
        self.sb.write("p.txt", "model kind=boost task=regression early-stopping=3\n")
        r = self.ml("fit", "--plan", "p.txt", "--target", "y", "--output", "b.cml", "full.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "args.conflict"), r)
        self.assertEqual(validate(r), [])
        r = self.ml("fit", "--plan", "p.txt", "--split", "train=0.8", "--target", "y", "--output", "b.cml", "--overwrite", "full.csv")
        self.assertEqual(r.status, 0, r)

    def test_early_stopping_needs_ten_training_rows(self):
        rows = "a,y\n" + "".join("%d,%d\n" % (i, i * 2 % 7) for i in range(11))
        self.sb.write("e11.csv", rows)
        r = self.ml("fit", "--model", "boost", "--early-stopping", "2", "--min-leaf", "1", "--split", "train=0.9", "--target", "y", "--output", "e.cml", "--overwrite", "e11.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "ml.param.out-of-range"), r)    # 11 rows: 2 held out, 9 train rows
        self.sb.write("e12.csv", rows + "11,3\n")
        r = self.ml("fit", "--model", "boost", "--early-stopping", "2", "--min-leaf", "1", "--split", "train=0.9", "--target", "y", "--output", "e.cml", "--overwrite", "e12.csv")
        self.assertEqual(r.status, 0, r)                                                  # 12 rows: 2 held out, 10 train rows, 1 validation row

    def test_early_stopping_stops_on_a_noisy_target_and_keeps_the_best_round(self):
        rng = random.Random(4)
        rows = ["a,b,y"] + ["%.3f,%.3f,%.3f" % (a, b, a + rng.gauss(0, 6)) for a, b in ((rng.uniform(0, 5), rng.uniform(0, 5)) for _ in range(400))]
        self.sb.write("noisy.csv", "\n".join(rows) + "\n")
        r = self.fit("y", "noisy.csv", "b.cml", "--rounds", "400", "--learning-rate", "0.3", "--early-stopping", "5", "--split", "train=0.8", "--seed", "3", "--min-leaf", "5")
        boost = r.data()["summary"]["boost"]
        self.assertEqual(boost["rounds_requested"], 400)
        self.assertLess(boost["rounds"], 400)
        self.assertEqual(boost["best_round"], boost["rounds"])
        h = header(self.sb.read("b.cml"))
        self.assertEqual((h[1], h[7], h[8]), (boost["rounds"], 5, 400))
        # without early stopping all the rounds are fitted and kept
        r = self.fit("y", "noisy.csv", "all.cml", "--rounds", "40", "--learning-rate", "0.3", "--split", "train=0.8", "--seed", "3", "--min-leaf", "5")
        self.assertEqual(r.data()["summary"]["boost"]["rounds"], 40)
        self.assertNotIn("best_round", r.data()["summary"]["boost"])

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
        self.assertEqual(self.sb.read("h1.cml"), self.sb.read("h2.cml"))
        # and the held-out metrics are the ones eval --split gives on the file
        a = self.ml("eval", "--model-file", "h2.cml", "--split", "train=0.8", "--seed", "7", "h2.csv").data()["metrics"]
        r = self.fit("y", "h2.csv", "h3.cml", "--rounds", "100", "--early-stopping", "4", "--split", "train=0.8", "--seed", "7", "--min-leaf", "4")
        self.assertEqual(r.data()["summary"]["holdout"]["metrics"], a)

    # ------------------------------------------------------------------ limits and ranges
    def test_ranges_of_the_numbers(self):
        for argv, rule, status in ((["--learning-rate", "0"], "ml.param.out-of-range", 2), (["--learning-rate", "1.0001"], "ml.param.out-of-range", 2),
                                   (["--subsample", "0"], "ml.param.out-of-range", 2), (["--subsample", "1.0000001"], "ml.param.out-of-range", 2),
                                   (["--subsample", "0.0000001"], "ml.param.out-of-range", 2), (["--l2", "1e101"], "ml.param.out-of-range", 2),
                                   (["--rounds", "10001"], "ml.param.out-of-range", 2), (["--rounds", "0"], "ml.param.out-of-range", 2),
                                   (["--min-leaf", "1000"], "ml.param.out-of-range", 2), (["--bins", "1"], "ml.param.out-of-range", 2),
                                   (["--max-depth", "31"], "ml.limit.depth", 8), (["--max-nodes", "1"], "ml.limit.nodes", 8), (["--max-nodes", "4194305"], "ml.limit.nodes", 8),
                                   (["--threads", "0"], "ml.param.threads", 2), (["--trees", "3"], "args.conflict", 2), (["--mtry", "1"], "args.conflict", 2)):
            r = self.ml("fit", "--model", "boost", *argv, "--target", "y", "--output", "m.cml", "--overwrite", "full.csv")
            self.assertEqual((r.first_rule(), r.status), (rule, status), (argv, r))
            self.assertEqual(validate(r), [], (argv, r))

    def test_the_most_nodes_of_all_rounds(self):
        self.sb.write("n300.csv", regression_csv(n=300, missing=False))
        r = self.ml("fit", "--model", "boost", "--rounds", "10000", "--max-depth", "30", "--min-leaf", "1", "--target", "y", "--output", "m.cml", "--overwrite", "n300.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.limit.nodes"), r)
        # the product at the ceiling: 4,194,304 // 599 = 7002 rounds of a tree that can have 599 nodes
        r = self.ml("fit", "--model", "boost", "--rounds", "7003", "--max-depth", "30", "--min-leaf", "1", "--target", "y", "--output", "m.cml", "--overwrite", "n300.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.limit.nodes"), r)

    def test_an_explicit_zero_is_the_default_early_stopping(self):
        self.fit("y", "full.csv", "a.cml", "--rounds", "5", "--early-stopping", "0")
        self.fit("y", "full.csv", "b.cml", "--rounds", "5")
        self.assertEqual(self.sb.read("a.cml"), self.sb.read("b.cml").replace(b"early-stopping=0", b"early-stopping=0"))
        self.assertEqual(header(self.sb.read("a.cml"))[7], 0)

    def test_a_boosted_fit_takes_at_most_2_to_the_20_training_rows(self):
        n = (1 << 20) + 1
        self.sb.write("rows.csv", "x,y\n" + "".join("%d,%d\n" % (i % 1000, i % 7) for i in range(n)))
        r = self.ml("fit", "--model", "boost", "--target", "y", "--output", "rows.cml", "rows.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.input.too-many-rows"), r)
        self.assertEqual((r.error()["detail"]["rows"], r.error()["detail"]["max"]), (n, 1 << 20))
        self.assertFalse((self.sb.dir / "rows.cml").exists())
        r = self.ml("fit", "--model", "boost", "--rounds", "2", "--max-depth", "1", "--split", "train=0.5", "--target", "y", "--output", "rows.cml", "rows.csv")
        self.assertEqual(r.status, 0, r)

    def test_more_than_4096_columns_are_refused_at_fit_not_at_predict(self):
        d = 4097
        self.sb.write("wide.csv", ",".join("c%d" % j for j in range(d)) + ",y\n" + "".join(",".join(str((i * j) % 7) for j in range(d)) + ",%d\n" % i for i in range(30)))
        r = self.ml("fit", "--model", "boost", "--max-columns", "8192", "--min-leaf", "1", "--rounds", "2", "--target", "y", "--output", "wide.cml", "wide.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.input.too-many-columns"), r)
        self.assertFalse((self.sb.dir / "wide.cml").exists())
        r = self.ml("fit", "--model", "boost", "--max-columns", "8192", "--min-leaf", "1", "--rounds", "2", "--drop", "c0", "--target", "y", "--output", "wide.cml", "wide.csv")
        self.assertEqual(r.status, 0, r)
        self.assertEqual(self.ml("predict", "--model-file", "wide.cml", "--max-columns", "8192", "--limit", "2", "--drop", "y,c0", "wide.csv").status, 0)

    def test_the_total_of_nodes_is_bounded_at_exactly_2_to_the_22(self):
        self.sb.write("n300b.csv", regression_csv(n=300, missing=False))
        base = ["--model", "boost", "--max-depth", "9", "--max-nodes", "512", "--min-leaf", "20", "--max-bytes", "268435456", "--target", "y", "--output", "edge.cml", "--overwrite"]
        r = self.ml("fit", *base, "--rounds", "8193", "n300b.csv")                       # 8193 rounds of up to 512 nodes: over 4,194,304
        self.assertEqual((r.status, r.first_rule()), (8, "ml.limit.nodes"), r)
        r = self.ml("fit", *base, "--rounds", "8192", "--learning-rate", "0.01", "n300b.csv")  # exactly 4,194,304: allowed
        self.assertEqual(r.status, 0, r)

    def test_a_refused_fit_writes_nothing(self):
        self.sb.write("keep.cml", b"x")
        r = self.ml("fit", "--model", "boost", "--learning-rate", "2", "--target", "y", "--output", "keep.cml", "--overwrite", "full.csv")
        self.assertEqual(r.first_rule(), "ml.param.out-of-range")
        self.assertEqual(self.sb.read("keep.cml"), b"x")

    def test_a_file_predict_cannot_read_is_not_written(self):
        r = self.ml("fit", "--model", "boost", "--max-bytes", "2000", "--target", "y", "--output", "big.cml", "--overwrite", "full.csv")
        self.assertEqual(r.first_rule(), "limit.input-too-large")
        self.assertFalse((self.sb.dir / "big.cml").exists())
        r = self.ml("fit", "--model", "boost", "--target", "y", "--output", "ok.cml", "--overwrite", "full.csv")
        size = len(self.sb.read("ok.cml"))
        self.assertEqual(self.ml("fit", "--model", "boost", "--max-bytes", str(size), "--target", "y", "--output", "ok2.cml", "--overwrite", "full.csv").status, 0)
        self.assertEqual(self.ml("fit", "--model", "boost", "--max-bytes", str(size - 1), "--target", "y", "--output", "ok3.cml", "--overwrite", "full.csv").first_rule(), "limit.input-too-large")
        self.assertEqual(self.ml("predict", "--model-file", "ok2.cml", "--max-bytes", str(size), "--limit", "1", "full.csv").status, 0)

    def test_hostile_data(self):
        # one row, huge magnitudes, a constant target, duplicated rows: a model or a tagged refusal, never a trap
        cases = {"one": "x,y\n1,2\n", "huge": "x,y\n1e300,1e100\n-1e300,-1e100\n0,0\n5,3\n", "const": "x,y\n" + "".join("%d,3\n" % i for i in range(20)),
                 "dup": "x,y\n" + "1,2\n" * 30, "toobig": "x,y\n1,1e101\n2,3\n3,4\n"}
        for name, text in cases.items():
            self.sb.write(name + ".csv", text)
            r = self.ml("fit", "--model", "boost", "--min-leaf", "1", "--target", "y", "--output", name + ".cml", "--overwrite", name + ".csv")
            self.assertIn(r.status, (0, 2, 8), (name, r))
            self.assertEqual(validate(r), [], (name, r))
            if r.status == 0:
                self.assertEqual(self.ml("predict", "--model-file", name + ".cml", "--limit", "5", name + ".csv").status, 0, name)
        self.assertEqual(self.ml("fit", "--model", "boost", "--min-leaf", "1", "--target", "y", "--output", "t.cml", "--overwrite", "toobig.csv").first_rule(), "ml.input.non-finite")

    # ------------------------------------------------------------------ threads (GB2)
    def test_the_model_file_is_the_same_at_1_2_4_8_and_16_threads(self):
        self.sb.write("split.csv", regression_csv(n=400))
        cases = (("y", "split.csv", ["--rounds", "12", "--seed", "9", "--split", "train=0.8", "--subsample", "0.7"]),
                 ("lab", "cls.csv", ["--rounds", "10", "--task", "classification", "--seed", "3", "--min-leaf", "2"]),
                 ("y", "split.csv", ["--rounds", "300", "--early-stopping", "3", "--split", "train=0.8", "--seed", "1", "--learning-rate", "0.5"]),
                 ("y", "split.csv", ["--rounds", "6", "--bins", "255", "--step", "scale column=a kind=minmax"]))
        for target, csv, extra in cases:
            seen = {}
            for t in (1, 2, 4, 8, 16):
                r = self.ml("fit", "--model", "boost", "--target", target, "--output", "th.cml", "--overwrite", "--threads", str(t), *extra, csv)
                self.assertEqual(r.status, 0, r)
                seen[t] = (self.sb.read("th.cml"), r.stdout.replace(b"th.cml", b"M"))
            self.assertEqual({t: v[0] for t, v in seen.items() if v[0] != seen[1][0]}, {}, extra)
            self.assertEqual({t: 1 for t, v in seen.items() if v[1] != seen[1][1]}, {}, extra)

    def test_columns_are_matched_by_name(self):
        self.fit("y", "full.csv", "f.cml", "--rounds", "3")
        self.sb.write("x.csv", "c,a,b\n1,2.5,1.0\n3,7.5,-2.0\n")
        rev = self.ml("predict", "--model-file", "f.cml", "x.csv").data()["rows"]
        self.sb.write("x2.csv", "a,b,c\n2.5,1.0,1\n7.5,-2.0,3\n")
        self.assertEqual(rev, self.ml("predict", "--model-file", "f.cml", "x2.csv").data()["rows"])
        self.sb.write("x3.csv", "a,b\n2.5,1.0\n")
        self.assertEqual(self.ml("predict", "--model-file", "f.cml", "x3.csv").first_rule(), "ml.schema.mismatch")

    def test_categorical_columns_are_encoded_by_a_step(self):
        self.sb.write("txt.csv", "a,s,y\n" + "".join("%d,%s,%d\n" % (i, "u" if i % 3 else "v", (i % 3 == 0) * 5 + i) for i in range(1, 40)))
        r = self.ml("fit", "--model", "boost", "--cat", "s", "--target", "y", "--output", "m.cml", "--overwrite", "txt.csv")
        self.assertEqual(r.first_rule(), "ml.input.categorical-not-encoded", r)
        r = self.ml("fit", "--model", "boost", "--cat", "s", "--step", "encode column=s kind=onehot", "--rounds", "20", "--min-leaf", "2", "--target", "y", "--output", "m.cml", "--overwrite", "txt.csv")
        self.assertEqual(r.status, 0, r)
        self.assertIn("string_table", r.data()["summary"])
        p = self.ml("predict", "--model-file", "m.cml", "--drop", "y", "--limit", "3", "txt.csv")
        self.assertEqual(p.status, 0, p)


if __name__ == "__main__":
    unittest.main()

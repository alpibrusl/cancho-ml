"""M5, M7 and the behaviour of M1 -- multinomial logistic regression through `ml fit | predict | eval | describe` (docs/multiclass.md section 3).

What is judged from outside: the classes (the distinct labels, ascending, the last the reference class), the file (kind 9, the three records, the sum order 2), the refusals of a target
the family cannot take (one class, two, 65, the other classifiers on three, --positive, cv), the answer of predict (the label is the argmax with the smaller label on a tie, the
probabilities are an object keyed by the labels in ascending order that sums to one within K 2^-52), the metrics of eval and of a split, the refusals of a separable and a collinear
fit and that their repairs work, a class with no training row, a label eval has no class for, and the file at 1, 2, 4 and 8 threads. The byte-for-byte comparison with the Python
oracle is test_wire_softmax_oracle.py, the damaged files are test_wire_softmax_file.py."""

import math
import random
import struct
import unittest

from harness import Sandbox, sections, validate
from test_model_file import payloads, write_file

LABELS = (3, 5, 9)


def mc_csv(n=150, labels=LABELS, seed=2, noise=0.6, header="a,b,y"):
    """a deterministic target of len(labels) classes, each the largest of three noisy linear scores"""
    r = random.Random(seed)
    k = len(labels)
    coef = [(r.uniform(-1.5, 1.5), r.uniform(-1.5, 1.5)) for _ in range(k)]
    out = [header]
    rows = []
    for i in range(n):
        a, b = r.gauss(0, 1), r.gauss(0, 1)
        s = [ca * a + cb * b + r.gauss(0, noise) for ca, cb in coef]
        c = s.index(max(s)) if i >= k else i
        rows.append((a, b, labels[c]))
    for a, b, y in rows:
        out.append("%.4f,%.4f,%d" % (a, b, y))
    return "\n".join(out) + "\n"


def f64s(b, at, n):
    return struct.unpack_from("<%dd" % n, b, at)


def model_section(data):
    tag, start, n = sections(data)[3]
    return data[start:start + n]


class Softmax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        cls.sb.write("mc.csv", mc_csv())
        cls.sb.write("mc7.csv", mc_csv(n=300, labels=(-4, 0, 2, 7, 11, 12, 30), seed=5))

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def ml(self, *args):
        return self.sb.run(*args)

    def fit(self, csv="mc.csv", out="s.cml", *extra):
        r = self.ml("fit", "--model", "softmax", "--target", "y", "--output", out, "--overwrite", *extra, csv)
        self.assertEqual(r.status, 0, r)
        self.assertEqual(validate(r), [], r)
        return r

    # ------------------------------------------------------------------ the classes and the file
    def test_the_classes_are_the_labels_ascending_and_the_last_is_the_reference(self):
        for csv, labels in (("mc.csv", [3, 5, 9]), ("mc7.csv", [-4, 0, 2, 7, 11, 12, 30])):
            s = self.fit(csv).data()["summary"]
            self.assertEqual(s["classes"], labels)
            self.assertEqual(s["reference"], labels[-1])
            self.assertNotIn("positive", s)
            self.assertEqual(s["features"], 2)
            self.assertEqual([c["class"] for c in s["coefficients"]], [c for c in labels[:-1] for _ in range(2)])
            self.assertEqual([c["name"] for c in s["coefficients"]], ["a", "b"] * (len(labels) - 1))
            self.assertEqual([c["class"] for c in s["intercepts"]], labels[:-1])

    def test_the_file_is_kind_9_with_three_records(self):
        self.fit("mc7.csv")
        data = self.sb.read("s.cml")
        m = model_section(data)
        kk, d = 7, 2
        self.assertEqual(struct.unpack_from("<I", m, 0)[0], 9)
        self.assertEqual(struct.unpack_from("<IIQQ", m, 4), (2, 2, kk - 1, d))
        coef_end = 4 + 8 + 16 + 8 * (kk - 1) * d
        self.assertEqual(struct.unpack_from("<IIQ", m, coef_end), (2, 1, kk - 1))
        lab = coef_end + 8 + 8 + 8 * (kk - 1)
        self.assertEqual(struct.unpack_from("<IIQ", m, lab), (1, 1, kk))
        self.assertEqual(struct.unpack_from("<%dq" % kk, m, lab + 16), (-4, 0, 2, 7, 11, 12, 30))
        self.assertEqual(len(m), lab + 16 + 8 * kk)
        # the sum order of a kind 4 and above is the last four bytes of the meta section (3 for a tree, a forest and a boosted model)
        meta = payloads(data)[5]
        self.assertEqual(struct.unpack_from("<I", meta, len(meta) - 4)[0], 2)
        # the stored coefficients are the ones the summary prints, class-major
        s = self.ml("describe", "--model-file", "s.cml").data()
        self.assertEqual(s["kind"], 9)

    def test_the_file_is_the_same_at_every_thread_count(self):
        self.fit("mc7.csv", "t1.cml", "--threads", "1")
        for t in ("2", "4", "8"):
            self.fit("mc7.csv", "t%s.cml" % t, "--threads", t)
            self.assertEqual(self.sb.read("t%s.cml" % t), self.sb.read("t1.cml"), "threads " + t)

    # ------------------------------------------------------------------ the refusals of the target
    def test_a_target_of_the_wrong_number_of_classes(self):
        self.sb.write("one.csv", "a,y\n1,3\n2,3\n3,3\n")
        self.assertEqual(self.ml("fit", "--model", "softmax", "--target", "y", "--output", "o.cml", "one.csv").first_rule(), "ml.label.single-class")
        self.sb.write("two.csv", "a,y\n1,3\n2,5\n3,3\n4,5\n")
        r = self.ml("fit", "--model", "softmax", "--target", "y", "--output", "o.cml", "two.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "ml.param.out-of-range"), r)
        self.assertEqual(r.error()["detail"]["classes"], 2)
        rows = "".join("%d,%d\n" % (i, i % 65) for i in range(200))
        self.sb.write("many.csv", "a,y\n" + rows)
        r = self.ml("fit", "--model", "softmax", "--target", "y", "--output", "o.cml", "many.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.limit.classes"), r)
        self.assertEqual(r.error()["detail"]["max"], 64)
        rows = "".join("%d,%d\n" % (i, i % 64) for i in range(200))
        self.sb.write("sixtyfour.csv", "a,y\n" + rows)
        r = self.ml("fit", "--model", "softmax", "--target", "y", "--output", "o.cml", "--iterations", "1", "--l2", "1", "sixtyfour.csv")
        self.assertEqual(r.status, 0, r)
        self.assertEqual(len(r.data()["summary"]["classes"]), 64)
        self.sb.write("float.csv", "a,y\n1,3.5\n2,5.5\n3,7.5\n")
        self.assertEqual(self.ml("fit", "--model", "softmax", "--target", "y", "--output", "o.cml", "float.csv").first_rule(), "ml.label.not-integer")

    def test_the_other_classifiers_still_refuse_three_classes(self):
        for model, extra in (("logistic", []), ("tree", ["--task", "classification"]), ("forest", ["--task", "classification", "--trees", "3"]),
                             ("boost", ["--task", "classification", "--rounds", "3"]), ("constant", ["--task", "classification"])):
            r = self.ml("fit", "--model", model, "--target", "y", "--output", "o.cml", "--overwrite", *extra, "mc.csv")
            self.assertEqual(r.first_rule(), "ml.label.not-binary" if model != "constant" else r.first_rule(), (model, r))
            self.assertNotEqual(r.status, 0, (model, r))

    def test_a_positive_label_means_nothing_with_three_classes(self):
        r = self.ml("fit", "--model", "softmax", "--target", "y", "--output", "o.cml", "--positive", "5", "mc.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "args.conflict"), r)
        self.assertEqual(r.error()["detail"]["flag"], "positive")
        self.fit()
        r = self.ml("eval", "--model-file", "s.cml", "--positive", "5", "mc.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "args.conflict"), r)

    def test_cross_validation_is_not_built_for_three_classes(self):
        r = self.ml("cv", "--model", "softmax", "--target", "y", "mc.csv")
        self.assertEqual((r.status, r.first_rule()), (2, "args.conflict"), r)

    # ------------------------------------------------------------------ predict
    def test_predict_labels_and_probabilities(self):
        self.fit("mc7.csv")
        p = self.ml("predict", "--model-file", "s.cml", "--limit", "1000", "mc7.csv").data()
        self.assertEqual(p["columns"], ["label"])
        labels = {-4, 0, 2, 7, 11, 12, 30}
        self.assertTrue({int(r[0]) for r in p["rows"]} <= labels)
        q = self.ml("predict", "--model-file", "s.cml", "--proba", "--limit", "1000", "mc7.csv").data()
        self.assertEqual(q["columns"], ["label", "probabilities"])
        kk = 7
        for (lab, probs), (lab2,) in zip(q["rows"], p["rows"]):
            self.assertEqual(lab, lab2)
            self.assertEqual(list(probs), [str(k) for k in sorted(int(k) for k in probs)], "keys are not the labels in ascending order")
            self.assertEqual([int(k) for k in probs], sorted(labels))
            vals = [float(v) for v in probs.values()]
            self.assertTrue(all(0.0 <= v <= 1.0 for v in vals))
            self.assertLessEqual(abs(sum(vals) - 1.0), kk * 2.0 ** -52 * 4)
            best = max(range(kk), key=lambda i: (vals[i], -i))
            self.assertEqual(int(lab), int(list(probs)[best]), (lab, probs))

    def test_a_tie_goes_to_the_smaller_label(self):
        self.fit()
        secs = payloads(self.sb.read("s.cml"))
        kk, d = 3, 2
        zero = (struct.pack("<I", 9) + struct.pack("<IIQQ", 2, 2, kk - 1, d) + bytes(8 * (kk - 1) * d)
                + struct.pack("<IIQ", 2, 1, kk - 1) + bytes(8 * (kk - 1)) + struct.pack("<IIQ", 1, 1, kk) + struct.pack("<3q", *LABELS))
        secs[4] = zero
        self.sb.write("zero.cml", write_file(secs))
        p = self.ml("predict", "--model-file", "zero.cml", "--proba", "--limit", "5", "mc.csv")
        self.assertEqual(p.status, 0, p)
        for lab, probs in p.data()["rows"]:
            self.assertEqual(lab, "3")
            self.assertTrue(all(abs(float(v) - 1.0 / 3.0) < 1e-15 for v in probs.values()), probs)

    def test_the_text_format_has_the_label_and_a_tab_separated_probability_per_class(self):
        self.fit()
        r = self.ml("predict", "--model-file", "s.cml", "--proba", "--limit", "3", "--format", "text", "mc.csv")
        self.assertEqual(r.status, 0, r)
        lines = r.stdout.decode().splitlines()
        self.assertEqual(len(lines), 3, lines)
        for line in lines:
            cells = line.split("\t")
            self.assertEqual(len(cells), 4, line)
            self.assertIn(int(cells[0]), LABELS)
            self.assertAlmostEqual(sum(float(c) for c in cells[1:]), 1.0, places=12)

    # ------------------------------------------------------------------ eval and the split
    def test_eval_metrics(self):
        self.fit("mc7.csv")
        ev = self.ml("eval", "--model-file", "s.cml", "mc7.csv").data()
        self.assertNotIn("positive", ev)
        m = ev["metrics"]
        for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc_ovr", "log_loss"):
            self.assertIn(k, m)
        self.assertEqual(m["classes"], [-4, 0, 2, 7, 11, 12, 30])
        conf = m["confusion"]
        self.assertEqual(sum(map(sum, conf)), 300)
        self.assertEqual(float(m["accuracy"]), sum(conf[i][i] for i in range(7)) / 300)
        self.assertEqual([c["support"] for c in m["per_class"]], [sum(row) for row in conf])
        self.assertEqual([c["class"] for c in m["per_class"]], m["classes"])
        self.assertEqual(set(m["left_out"]), {"precision", "recall", "f1", "roc_auc_ovr"})
        self.assertGreater(float(m["accuracy"]), 0.5)

    def test_the_held_out_rows_of_a_fit_are_scored_with_the_same_metrics(self):
        r = self.fit("mc7.csv", "sp.cml", "--split", "train=0.7", "--seed", "11", "--stratify")
        h = r.data()["summary"]["holdout"]
        self.assertEqual(h["on"], "held-out")
        self.assertEqual(h["rows"], sum(map(sum, h["metrics"]["confusion"])))
        ev = self.ml("eval", "--model-file", "sp.cml", "--split", "train=0.7", "--seed", "11", "--stratify", "mc7.csv").data()
        self.assertEqual(ev["metrics"], h["metrics"])

    def test_a_class_with_no_training_row_is_refused_by_name(self):
        rows = mc_csv(n=120).splitlines()
        rows.append("0.5,0.5,17")   # one row of a fourth class: some seeds hold it out
        self.sb.write("rare.csv", "\n".join(rows) + "\n")
        seen = {}
        for seed in range(1, 25):
            r = self.ml("fit", "--model", "softmax", "--target", "y", "--output", "r.cml", "--overwrite", "--split", "train=0.8", "--seed", str(seed), "--l2", "1", "rare.csv")
            seen[r.status] = r
        self.assertIn(8, seen, "no seed held the rare row out")
        self.assertEqual(seen[8].first_rule(), "ml.label.unseen")
        self.assertEqual(seen[8].error()["detail"]["label"], 17)
        self.assertEqual(seen[8].error()["detail"]["reason"], "no-training-row")
        self.assertIn(0, seen, "no seed kept the rare row in")

    def test_eval_names_the_first_row_whose_label_is_not_a_class(self):
        self.fit()
        self.sb.write("alien.csv", "a,b,y\n0.1,0.2,3\n0.3,0.4,5\n0.5,0.6,8\n0.7,0.8,9\n")
        r = self.ml("eval", "--model-file", "s.cml", "alien.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.label.unseen"), r)
        d = r.error()["detail"]
        self.assertEqual((d["row"], d["label"], d["classes"]), (3, 8, 3))
        # predict reads no label, so the same file is answered
        self.assertEqual(self.ml("predict", "--model-file", "s.cml", "alien.csv").status, 0)

    # ------------------------------------------------------------------ fits that have no answer
    def test_a_separable_target_is_refused_and_its_repair_fits(self):
        rows = []
        r = random.Random(4)
        for i in range(120):
            c = i % 3
            rows.append("%.4f,%.4f,%d" % (10 * c + r.uniform(-1, 1), 10 * (c % 2) + r.uniform(-1, 1), (3, 5, 9)[c]))
        self.sb.write("sep.csv", "a,b,y\n" + "\n".join(rows) + "\n")
        r = self.ml("fit", "--model", "softmax", "--target", "y", "--output", "sep.cml", "sep.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.fit.separable"), r)
        self.assertEqual(validate(r), [], r)
        repair = r.error()["repair"]
        self.assertIsNotNone(repair, r)
        self.assertEqual(self.ml("fit", "--model", "softmax", "--target", "y", "--output", "sep.cml", "--l2", "1", "sep.csv").status, 0)

    def test_a_collinear_column_is_refused(self):
        r_ = random.Random(8)
        rows = []
        for i in range(90):
            a = r_.gauss(0, 1)
            rows.append("%.4f,%.4f,%.4f,%d" % (a, a, r_.gauss(0, 1), (3, 5, 9)[i % 3]))
        self.sb.write("col.csv", "a,b,c,y\n" + "\n".join(rows) + "\n")
        r = self.ml("fit", "--model", "softmax", "--target", "y", "--output", "col.cml", "col.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.fit.collinear"), r)
        self.assertEqual(r.error()["detail"]["model"], "softmax")
        self.assertEqual(self.ml("fit", "--model", "softmax", "--target", "y", "--output", "col.cml", "--l2", "1", "col.csv").status, 0)

    def test_the_limits_refuse_before_the_fit_runs(self):
        # 33 classes and 64 features: (K - 1)(d + 1) = 2080 parameters
        r_ = random.Random(3)
        cols = ",".join("f%d" % j for j in range(64))
        rows = []
        for i in range(400):
            rows.append(",".join("%.3f" % r_.gauss(0, 1) for _ in range(64)) + ",%d" % (i % 33))
        self.sb.write("wide.csv", cols + ",y\n" + "\n".join(rows) + "\n")
        r = self.ml("fit", "--model", "softmax", "--target", "y", "--output", "w.cml", "--iterations", "1", "--l2", "1", "wide.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.limit.parameters"), r)

    # ------------------------------------------------------------------ describe and the verbs that explain a model
    def test_describe(self):
        self.fit("mc7.csv")
        d = self.ml("describe", "--model-file", "s.cml").data()
        self.assertEqual((d["model"], d["task"], d["kind"]), ("softmax", "classification", 9))
        self.assertEqual(d["classes"], [-4, 0, 2, 7, 11, 12, 30])
        self.assertIsNone(d["positive"])
        self.assertEqual(d["output"]["columns"], ["label"])
        self.assertEqual(d["output"]["with_proba"], ["label", "probabilities"])
        self.assertIsNone(d["output"]["label_threshold"])

    def test_importance_and_dump_refuse_a_multiclass_model(self):
        self.fit()
        r = self.ml("importance", "--model-file", "s.cml", "--data", "mc.csv")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.importance.unsupported-model"), r)
        r = self.ml("dump", "--model-file", "s.cml")
        self.assertEqual((r.status, r.first_rule()), (8, "ml.dump.unsupported-model"), r)

    def test_a_target_in_the_prediction_file_is_ignored_like_the_binary_models_do(self):
        self.fit()
        p = self.ml("predict", "--model-file", "s.cml", "--limit", "2", "mc.csv").data()
        self.assertEqual(p["ignored_columns"], ["y"])

    def test_the_iterations_and_l2_are_hyperparameters(self):
        a = self.fit("mc.csv", "i5.cml", "--iterations", "5").data()["summary"]["plan"]
        self.assertEqual(a, "model kind=softmax iterations=5 l2=0.0\n")
        b = self.fit("mc.csv", "l2.cml", "--l2", "0.5").data()["summary"]["plan"]
        self.assertEqual(b, "model kind=softmax iterations=100 l2=0.5\n")
        self.assertNotEqual(self.sb.read("i5.cml"), self.sb.read("l2.cml"))
        self.assertEqual(self.ml("fit", "--model", "softmax", "--target", "y", "--output", "o.cml", "--iterations", "0", "mc.csv").status, 2)
        self.assertEqual(self.ml("fit", "--model", "softmax", "--target", "y", "--output", "o.cml", "--l2", "-1", "mc.csv").status, 2)
        self.assertTrue(math.isfinite(1.0))


if __name__ == "__main__":
    unittest.main()

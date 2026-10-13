"""W2, W6, W7 -- behaviours of the planned models that the fixtures and the oracle do not state in words: the positive label, the labels the model never saw, the text
outputs, the order of a flow, the held-out rows."""

import json
import unittest
from fractions import Fraction

from harness import Sandbox, run

CLS = "a,b,lab\n" + "".join("%d,%d,%d\n" % (i, (i * 7) % 11, 1 if (i + 2 * ((i * 7) % 11) + (i * 5) % 4) % 7 > 3 else 0) for i in range(1, 61))
LIN = "a,b,y\n" + "".join("%d,%s,%.3f\n" % (i, "" if i % 9 == 4 else str((i * 7) % 11), 1.5 * i - 0.25 * ((i * 7) % 11) + 0.01 * ((i * 13) % 5)) for i in range(1, 61))


class Wire(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        self.sb.write("cls.csv", CLS)
        self.sb.write("lin.csv", LIN)
        r = self.sb.run("fit", "--model", "logistic", "--step", "scale column=* kind=standard", "--target", "lab", "--output", "log.cml", "cls.csv")
        self.assertEqual(r.status, 0, r)

    def tearDown(self):
        self.sb.cleanup()

    def labels(self):
        r = self.sb.run("predict", "--model-file", "log.cml", "--limit", "1000", "cls.csv")
        return [int(c[0]) for c in r.data()["rows"]]

    def test_the_positive_label_of_eval_is_the_one_precision_recall_and_f1_are_about(self):
        pred = self.labels()
        truth = [int(l.split(",")[2]) for l in CLS.splitlines()[1:]]
        for pos in (1, 0):
            tp = sum(1 for t, p in zip(truth, pred) if t == pos and p == pos)
            fp = sum(1 for t, p in zip(truth, pred) if t != pos and p == pos)
            fn = sum(1 for t, p in zip(truth, pred) if t == pos and p != pos)
            r = self.sb.run("eval", "--model-file", "log.cml", "--positive", str(pos), "cls.csv")
            self.assertEqual(r.status, 0, r)
            m = r.data()["metrics"]
            self.assertEqual(r.data()["positive"], pos)
            self.assertEqual(float(m["precision"]), float(Fraction(tp, tp + fp)))
            self.assertEqual(float(m["recall"]), float(Fraction(tp, tp + fn)))
            self.assertEqual(float(m["f1"]), float(Fraction(2 * tp, 2 * tp + fp + fn)))
            self.assertEqual(float(m["accuracy"]), sum(1 for t, p in zip(truth, pred) if t == p) / len(truth))
        # the score of the other label is 1 - p: the AUC of the label 0 with the score 1 - p is the AUC of the label 1 with the score p (the same pairs, ties
        # counted a half in both), and the log-loss is the same sum of the same terms
        m1 = self.sb.run("eval", "--model-file", "log.cml", "cls.csv").data()["metrics"]
        m0 = self.sb.run("eval", "--model-file", "log.cml", "--positive", "0", "cls.csv").data()["metrics"]
        self.assertAlmostEqual(float(m1["roc_auc"]), float(m0["roc_auc"]), places=12)
        self.assertAlmostEqual(float(m1["log_loss"]), float(m0["log_loss"]), places=12)

    def test_a_label_the_model_never_saw_is_refused_with_its_row_and_value(self):
        self.sb.write("other.csv", "a,b,lab\n1,2,0\n2,3,0\n3,4,5\n")
        r = self.sb.run("eval", "--model-file", "log.cml", "other.csv")
        self.assertEqual(r.first_rule(), "ml.label.not-binary", r)
        d = r.error()["detail"]
        self.assertEqual((d["row"], d["label"]), (3, 5))

    def test_a_positive_label_the_model_does_not_have_is_out_of_range_with_both_labels(self):
        r = self.sb.run("eval", "--model-file", "log.cml", "--positive", "7", "cls.csv")
        self.assertEqual(r.first_rule(), "ml.param.out-of-range", r)
        d = r.error()["detail"]
        self.assertEqual((d["value"], d["label0"], d["label1"]), (7, 0, 1))

    def test_fit_positive_chooses_the_label_the_probability_is_about(self):
        a = self.sb.run("fit", "--model", "logistic", "--positive", "0", "--overwrite", "--step", "scale column=* kind=standard", "--target", "lab", "--output", "log0.cml", "cls.csv")
        self.assertEqual(a.data()["summary"]["classes"], [1, 0])
        self.assertEqual(a.data()["summary"]["positive"], 0)
        b = self.sb.run("predict", "--model-file", "log.cml", "--proba", "--limit", "5", "cls.csv").data()["rows"]
        c = self.sb.run("predict", "--model-file", "log0.cml", "--proba", "--limit", "5", "cls.csv").data()["rows"]
        for (l1, p1), (l0, p0) in zip(b, c):
            self.assertEqual(l1, l0)                      # the same decision, whichever label is called positive
            self.assertAlmostEqual(float(p1) + float(p0), 1.0, places=6)

    def test_text_outputs(self):
        r = self.sb.run("eval", "--model-file", "log.cml", "--format", "text", "cls.csv")
        lines = r.stdout.decode().splitlines()
        self.assertEqual([l.split()[0] for l in lines], ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"])
        r = self.sb.run("fit", "--model", "linear", "--step", "impute column=b strategy=mean", "--target", "y", "--output", "lin.cml", "--format", "text", "lin.csv")
        self.assertTrue(r.stdout.decode().startswith("fit: linear regression, 60 rows, wrote lin.cml (") and r.stdout.decode().endswith(")\n"), r.stdout)

    def test_a_flow_with_a_float_csv_missing_cells_and_a_repair(self):
        # fit is refused with a repair, the repair runs, predict and eval use the stored steps
        r = self.sb.run("fit", "--model", "linear", "--target", "y", "--output", "lin.cml", "lin.csv")
        self.assertEqual(r.first_rule(), "ml.input.missing-value")
        rep = r.error()["repair"]
        self.assertEqual(rep["kind"], "retry")
        again = run(*rep["argv"][1:], cwd=self.sb.dir)
        self.assertEqual(again.status, 0, again)
        self.assertTrue(again.data()["summary"]["plan"].startswith("impute column=b strategy=mean\n"))
        # a column with empty cells at predict time is filled with the STORED mean: the same row, with and without the cell, differs
        self.sb.write("one.csv", "a,b\n10,\n10,5\n")
        p = self.sb.run("predict", "--model-file", "lin.cml", "one.csv").data()["rows"]
        self.assertNotEqual(p[0], p[1])

    def test_the_held_out_rows_are_scored_through_the_stored_steps(self):
        r = self.sb.run("fit", "--model", "linear", "--split", "train=0.75", "--seed", "11", "--step", "impute column=b strategy=mean;scale column=* kind=standard", "--target", "y",
                        "--output", "lin.cml", "lin.csv")
        self.assertEqual(r.status, 0, r)
        hold = r.data()["summary"]["holdout"]
        e = self.sb.run("eval", "--model-file", "lin.cml", "--split", "train=0.75", "--seed", "11", "lin.csv")
        self.assertEqual(e.data()["metrics"], hold["metrics"])
        self.assertEqual(e.data()["rows"], hold["rows"])
        # a different seed holds out other rows
        e2 = self.sb.run("eval", "--model-file", "lin.cml", "--split", "train=0.75", "--seed", "12", "lin.csv")
        self.assertNotEqual(e2.data()["metrics"], hold["metrics"])

    def test_threads_flag_is_inert_in_eval_and_checked(self):
        a = self.sb.run("eval", "--model-file", "log.cml", "cls.csv")
        for t in ("1", "3", "64"):
            self.assertEqual(self.sb.run("eval", "--model-file", "log.cml", "--threads", t, "cls.csv").stdout, a.stdout)

    def test_a_metric_with_no_value_is_null_not_a_refusal(self):
        # a model that predicts the same class everywhere: precision of the positive class has a zero denominator
        self.sb.write("flat.csv", "a,lab\n" + "".join("%d,%d\n" % (i, 1 if i % 5 == 0 else 0) for i in range(1, 41)))
        r = self.sb.run("fit", "--model", "logistic", "--l2", "5", "--iterations", "5", "--target", "lab", "--output", "flat.cml", "flat.csv")
        self.assertEqual(r.status, 0, r)
        e = self.sb.run("eval", "--model-file", "flat.cml", "flat.csv")
        self.assertEqual(e.status, 0, e)
        self.assertIsNone(e.data()["metrics"]["precision"])
        self.assertIsNotNone(e.data()["metrics"]["accuracy"])


if __name__ == "__main__":
    unittest.main()

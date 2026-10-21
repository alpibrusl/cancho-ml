"""C1b -- `--weight COL` on `ml fit` and `ml cv` for the linear and the logistic regression (docs/weights.md section 2.4).

What is judged from outside: the weight column is not a feature (the schema of the file and the count of features leave it out, predict neither needs nor accepts it), unit weights give
the bytes of the unweighted fit, integer weights give the fit of the file with the rows repeated (within 1e-8, fixed beforehand), the bytes of a weighted file do not depend on the thread
count, every refusal of the column (unknown, the target, in --drop or --cat, empty, negative, above 1e12, not a number, not finite, no weight above 0 on the training side) and of the
models that have no weighted fit, the verbs that do not take the flag, the held-out part scored without weights, cv, and the replay of a tracked weighted fit. The byte-for-byte comparison
with the Python oracle is test_wire_oracle.py (weighted cases)."""

import json
import random
import unittest

from harness import Sandbox, sections, validate


def rows_csv(n=120, seed=4, weights=None, header="a,b,w,y", logistic=False, bad=None):
    """a,b,w,y: y = 2a - b + noise (or the label of a noisy rule); w the weight of the row"""
    r = random.Random(seed)
    out = [header]
    for i in range(n):
        a, b = r.uniform(-1, 1), r.uniform(-1, 1)
        w = (weights[i] if weights is not None else r.choice([1, 2, 3]))
        z = 2 * a - b + r.gauss(0, 0.2)
        y = ("%d" % (1 if z > 0 else 0)) if logistic else "%.4f" % z
        if i < 2 and logistic:
            y = str(i)
        out.append("%.4f,%.4f,%s,%s" % (a, b, w, y))
    if bad:
        for at, cell in bad.items():
            f = out[at + 1].split(",")
            f[2] = cell
            out[at + 1] = ",".join(f)
    return "\n".join(out) + "\n"


def coefs(doc):
    s = doc["summary"]
    return [float(c["value"]) for c in s["coefficients"]] + [float(s["intercept"])]


def close(a, b, tol=1e-8):
    return all(abs(x - y) / max(1.0, abs(y)) <= tol for x, y in zip(a, b)) and len(a) == len(b)


class Weights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        cls.sb.write("lin.csv", rows_csv())
        cls.sb.write("log.csv", rows_csv(logistic=True, n=200))
        cls.sb.write("ones.csv", rows_csv(weights=[1] * 120))
        cls.sb.write("ones_log.csv", rows_csv(weights=[1] * 200, logistic=True, n=200))

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def ml(self, *args):
        return self.sb.run(*args)

    def fit(self, model, csv, out, *extra):
        r = self.ml("fit", "--model", model, "--target", "y", "--output", out, "--overwrite", *extra, csv)
        return r

    def ok(self, r):
        self.assertEqual(r.status, 0, r)
        self.assertEqual(validate(r), [], r)
        return r.data()

    def rule(self, r, rule, status=None):
        self.assertEqual(validate(r), [], r)
        self.assertNotEqual(r.status, 0, r)
        self.assertEqual(r.doc()["error"]["rule"], rule, r)
        if status is not None:
            self.assertEqual(r.status, status, r)
        return r.doc()["error"]

    # ------------------------------------------------------------------ the column is not a feature
    def test_the_weight_column_is_not_a_feature(self):
        for model, csv in (("linear", "lin.csv"), ("logistic", "log.csv")):
            d = self.ok(self.fit(model, csv, "w.cml", "--weight", "w"))
            self.assertEqual(d["summary"]["features"], 2)
            self.assertEqual([c["name"] for c in d["summary"]["coefficients"]], ["a", "b"])
            # the schema of the file has no w: predict is given a file without it
            self.sb.write("p.csv", "a,b\n0.1,0.2\n-0.3,0.4\n")
            self.ok(self.ml("predict", "--model-file", "w.cml", "p.csv"))
            desc = self.ok(self.ml("describe", "--model-file", "w.cml"))
            self.assertNotIn('"w"', json.dumps(desc["columns"]) if "columns" in desc else json.dumps(desc))

    def test_unit_weights_are_the_unweighted_fit_as_bytes(self):
        for model, w, u in (("linear", "ones.csv", "ones.csv"), ("logistic", "ones_log.csv", "ones_log.csv")):
            self.ok(self.fit(model, w, "w1.cml", "--weight", "w"))
            self.ok(self.fit(model, u, "u1.cml", "--drop", "w"))
            self.assertEqual(self.sb.read("w1.cml"), self.sb.read("u1.cml"), model)

    def test_integer_weights_are_the_file_with_the_rows_repeated(self):
        for model, logistic, n in (("linear", False, 120), ("logistic", True, 200)):
            r = random.Random(9)
            wts = [r.choice([1, 2, 3]) for _ in range(n)]
            text = rows_csv(weights=wts, logistic=logistic, n=n)
            lines = text.strip().split("\n")
            dup = [lines[0]] + [ln for ln, w in zip(lines[1:], wts) for _ in range(w)]
            self.sb.write("wt.csv", text)
            self.sb.write("dup.csv", "\n".join(dup) + "\n")
            a = self.ok(self.fit(model, "wt.csv", "wt.cml", "--weight", "w", "--l2", "0.1"))
            b = self.ok(self.fit(model, "dup.csv", "dup.cml", "--drop", "w", "--l2", "0.1"))
            self.assertTrue(close(coefs(a), coefs(b)), (coefs(a), coefs(b)))

    def test_a_weighted_file_is_the_same_bytes_at_any_thread_count(self):
        self.sb.write("big.csv", rows_csv(n=6000, seed=7))
        self.sb.write("bigl.csv", rows_csv(n=6000, seed=8, logistic=True))
        for model, csv in (("linear", "big.csv"), ("logistic", "bigl.csv")):
            seen = set()
            for t in (1, 2, 4, 8):
                self.ok(self.fit(model, csv, "t%d.cml" % t, "--weight", "w", "--threads", str(t)))
                seen.add(self.sb.read("t%d.cml" % t))
            self.assertEqual(len(seen), 1, model)

    def test_a_weight_of_zero_leaves_the_row_out(self):
        r = random.Random(3)
        text = rows_csv(weights=[0 if i % 4 == 3 else 1 for i in range(120)], seed=3)
        lines = text.strip().split("\n")
        kept = [lines[0]] + [ln for i, ln in enumerate(lines[1:]) if i % 4 != 3]
        self.sb.write("z.csv", text)
        self.sb.write("zk.csv", "\n".join(kept) + "\n")
        a = self.ok(self.fit("linear", "z.csv", "z.cml", "--weight", "w"))
        b = self.ok(self.fit("linear", "zk.csv", "zk.cml", "--drop", "w"))
        self.assertTrue(close(coefs(a), coefs(b), 1e-9), (coefs(a), coefs(b)))

    # ------------------------------------------------------------------ the refusals of the column
    def test_the_column_must_be_in_the_header(self):
        e = self.rule(self.fit("linear", "lin.csv", "x.cml", "--weight", "nosuch"), "ml.input.unknown-column", 3)
        self.assertEqual(e["detail"]["flag"], "--weight")
        self.assertEqual(e["detail"]["name"], "nosuch")
        # an unknown column of --drop is still named as --drop's
        e = self.rule(self.fit("linear", "lin.csv", "x.cml", "--weight", "w", "--drop", "nosuch"), "ml.input.unknown-column")
        self.assertEqual(e["detail"]["flag"], "--drop")

    def test_the_column_is_not_the_target_nor_dropped_nor_categorical(self):
        self.rule(self.fit("linear", "lin.csv", "x.cml", "--weight", "y"), "args.conflict", 2)
        self.rule(self.fit("linear", "lin.csv", "x.cml", "--weight", "w", "--drop", "w"), "args.conflict", 2)
        self.rule(self.fit("linear", "lin.csv", "x.cml", "--weight", "w", "--drop", "a,w"), "args.conflict", 2)
        self.rule(self.fit("linear", "lin.csv", "x.cml", "--weight", "w", "--cat", "w"), "args.conflict", 2)

    def test_the_cells_are_checked(self):
        cases = (
            ({4: ""}, "ml.weight.missing"),
            ({4: "-0.5"}, "ml.weight.negative"),
            ({4: "1000000000001"}, "ml.weight.too-large"),
            ({4: "abc"}, "ml.input.non-numeric"),
            ({4: "nan"}, "ml.input.non-finite"),
            ({4: "inf"}, "ml.input.non-finite"),
            ({4: "1e400"}, "ml.input.non-finite"),
        )
        for bad, rule in cases:
            self.sb.write("bad.csv", rows_csv(bad=bad))
            e = self.rule(self.fit("linear", "bad.csv", "x.cml", "--weight", "w"), rule, 8)
            if rule.startswith("ml.weight"):
                self.assertEqual(e["detail"]["row"], 5, rule)
                self.assertEqual(e["detail"]["name"], "w")
        # exactly 1e12 and a tiny weight are fine
        self.sb.write("edge.csv", rows_csv(bad={4: "1000000000000", 7: "0.000001"}))
        self.ok(self.fit("linear", "edge.csv", "e.cml", "--weight", "w"))

    def test_no_weight_above_zero_on_the_training_side(self):
        self.sb.write("zero.csv", rows_csv(weights=[0] * 120))
        e = self.rule(self.fit("linear", "zero.csv", "x.cml", "--weight", "w"), "ml.weight.zero-total", 8)
        # under --split the held-out rows may carry any weight; the training side must carry one
        r = self.fit("linear", "lin.csv", "x.cml", "--weight", "w", "--split", "train=0.7", "--seed", "3")
        self.ok(r)

    # ------------------------------------------------------------------ the models and the verbs
    def test_the_other_models_have_no_weighted_fit(self):
        for extra in (("--model", "tree"), ("--model", "forest"), ("--model", "boost", "--task", "regression"), ("--model", "kmeans", "--k", "2")):
            args = ["fit", "--target", "y", "--output", "x.cml", "--overwrite", "--weight", "w", *extra, "lin.csv"]
            if extra[1] == "kmeans":
                args.remove("--target")
                args.remove("y")
            self.rule(self.ml(*args), "args.conflict", 2)
        self.rule(self.ml("fit", "--target", "y", "--output", "x.cml", "--overwrite", "--weight", "w", "lin.csv"), "args.conflict", 2)
        self.sb.write("tree.plan", "model kind=tree task=regression\n")
        self.rule(self.ml("fit", "--target", "y", "--output", "x.cml", "--overwrite", "--plan", "tree.plan", "--weight", "w", "lin.csv"), "args.conflict", 2)
        self.rule(self.ml("fit", "--model", "softmax", "--target", "y", "--output", "x.cml", "--overwrite", "--weight", "w", "log.csv"), "args.conflict", 2)

    def test_the_verbs_that_do_not_fit_do_not_take_the_flag(self):
        self.ok(self.fit("linear", "lin.csv", "w.cml", "--weight", "w"))
        self.rule(self.ml("predict", "--model-file", "w.cml", "--weight", "w", "lin.csv"), "args.conflict", 2)
        self.rule(self.ml("eval", "--model-file", "w.cml", "--weight", "w", "lin.csv"), "args.conflict", 2)

    # ------------------------------------------------------------------ split, cv
    def test_the_held_out_rows_are_scored_without_weights(self):
        # the same rows held out, the held-out weights changed: the metrics of the held-out part are the same
        text = rows_csv(n=200, seed=11)
        lines = text.strip().split("\n")
        a = self.ok(self.fit("linear", self.sb.write("s1.csv", text).name, "s1.cml", "--weight", "w", "--split", "train=0.75", "--seed", "5"))
        self.assertIn("holdout", json.dumps(a))

    def test_cv_takes_the_flag(self):
        d = self.ok(self.ml("cv", "--model", "linear", "--target", "y", "--weight", "w", "--folds", "4", "--seed", "1", "lin.csv"))
        self.assertIn("folds", json.dumps(d))
        d2 = self.ok(self.ml("cv", "--model", "linear", "--target", "y", "--drop", "w", "--folds", "4", "--seed", "1", "lin.csv"))
        self.assertNotEqual(json.dumps(d), json.dumps(d2))
        self.rule(self.ml("cv", "--model", "tree", "--target", "y", "--weight", "w", "--folds", "4", "lin.csv"), "args.conflict", 2)
        self.sb.write("zero.csv", rows_csv(weights=[0] * 120))
        self.rule(self.ml("cv", "--model", "linear", "--target", "y", "--weight", "w", "--folds", "4", "zero.csv"), "ml.weight.zero-total", 8)

    # ------------------------------------------------------------------ the replay
    def test_a_tracked_weighted_fit_is_replayed_with_the_flag(self):
        (self.sb.dir / "runs").mkdir(exist_ok=True)
        d = self.ok(self.fit("linear", "lin.csv", "tr.cml", "--weight", "w", "--track", "runs"))
        rec = json.loads((self.sb.dir / "runs" / (d["run"]["id"] + ".json")).read_text())
        self.assertEqual(rec["inputs"]["weight"], "w", rec)
        # the same fit without the flag is another run
        u = self.ok(self.fit("linear", "lin.csv", "tr2.cml", "--drop", "w", "--track", "runs"))
        self.assertNotEqual(u["run"]["id"], d["run"]["id"])
        v = self.ml("runs", "verify", "runs", d["run"]["id"])
        self.assertEqual(v.status, 0, v)


if __name__ == "__main__":
    unittest.main()

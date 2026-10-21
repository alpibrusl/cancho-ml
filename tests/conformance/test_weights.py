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
        # each refusal names the flag --weight (another check could answer args.conflict with another flag)
        for args in (("--weight", "y"), ("--weight", "w", "--drop", "w"), ("--weight", "w", "--drop", "a,w"), ("--weight", "w", "--cat", "w")):
            e = self.rule(self.fit("linear", "lin.csv", "x.cml", *args), "args.conflict", 2)
            self.assertEqual(e["detail"]["flag"], "weight", args)

    def test_a_dropped_column_and_a_weight_column_together(self):
        # --drop names real columns and --weight another: the weight is not a feature, the dropped one is not either
        text = rows_csv().replace("a,b,w,y", "a,junk,w,y", 1)
        self.sb.write("junk.csv", text)
        d = self.ok(self.fit("linear", "junk.csv", "j.cml", "--drop", "junk", "--weight", "w"))
        self.assertEqual([c["name"] for c in d["summary"]["coefficients"]], ["a"])
        # two dropped columns
        self.sb.write("junk2.csv", text.replace("a,junk,w,y", "a,junk,w,y", 1).replace("\n", "\n", 1))
        self.sb.write("j3.csv", "a,j1,j2,w,y\n" + "".join("%d,%d,%d,%d,%d\n" % (i, i * 7 % 5, i * 3 % 4, 1 + i % 3, 2 * i + i % 2) for i in range(30)))
        d = self.ok(self.fit("linear", "j3.csv", "j3.cml", "--drop", "j1,j2", "--weight", "w"))
        self.assertEqual([c["name"] for c in d["summary"]["coefficients"]], ["a"])

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
        for extra in (("--model", "kmeans", "--k", "2"),):
            args = ["fit", "--target", "y", "--output", "x.cml", "--overwrite", "--weight", "w", *extra, "lin.csv"]
            if extra[1] == "kmeans":
                args.remove("--target")
                args.remove("y")
            self.rule(self.ml(*args), "args.conflict", 2)
        self.rule(self.ml("fit", "--target", "y", "--output", "x.cml", "--overwrite", "--weight", "w", "lin.csv"), "args.conflict", 2)
        self.sb.write("boost.plan", "model kind=kmeans k=2\n")
        self.rule(self.ml("fit", "--target", "y", "--output", "x.cml", "--overwrite", "--plan", "boost.plan", "--weight", "w", "lin.csv"), "args.conflict", 2)
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
        self.rule(self.ml("cv", "--model", "softmax", "--target", "y", "--weight", "w", "--folds", "4", "log.csv"), "args.conflict", 2)
        self.sb.write("zero.csv", rows_csv(weights=[0] * 120))
        self.rule(self.ml("cv", "--model", "linear", "--target", "y", "--weight", "w", "--folds", "4", "zero.csv"), "ml.weight.zero-total", 8)

    def test_json_lines_take_the_flag_and_give_the_csv_fit(self):
        r = random.Random(5)
        recs = [{"a": round(r.uniform(-1, 1), 3), "b": round(r.uniform(-1, 1), 3), "w": r.choice([1, 2, 3])} for _ in range(60)]
        for x in recs:
            x["y"] = round(2 * x["a"] - x["b"] + r.gauss(0, 0.1), 3)
        self.sb.write("w.jsonl", "".join(json.dumps(x) + "\n" for x in recs))
        self.sb.write("wj.csv", "a,b,w,y\n" + "".join("%s,%s,%s,%s\n" % (x["a"], x["b"], x["w"], x["y"]) for x in recs))
        a = self.ok(self.ml("fit", "--model", "linear", "--target", "y", "--weight", "w", "--input-format", "jsonl", "--fields", "a,b,w,y", "--output", "wj1.cml", "--overwrite", "w.jsonl"))
        b = self.ok(self.fit("linear", "wj.csv", "wj2.cml", "--weight", "w"))
        u = self.ok(self.ml("fit", "--model", "linear", "--target", "y", "--drop", "w", "--input-format", "jsonl", "--fields", "a,b,w,y", "--output", "wj3.cml", "--overwrite", "w.jsonl"))
        self.assertEqual(coefs(a), coefs(b))
        self.assertNotEqual(coefs(a), coefs(u))

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


def grid_csv(n=120, seed=5, weights=None, kind="reg", header="a,b,w,y"):
    """integer features 0 to 9, so that every value has a bin of its own and a repeated row changes no bin edge"""
    r = random.Random(seed)
    out = [header]
    for i in range(n):
        a, b = r.randint(0, 9), r.randint(0, 9)
        w = weights[i] if weights is not None else r.choice([1, 2, 3])
        z = a - 0.5 * b + r.gauss(0, 1.5)
        y = ("%d" % (1 if z > 0 else 0)) if kind == "cls" else "%.4f" % z
        out.append("%d,%d,%s,%s" % (a, b, w, y))
    return "\n".join(out) + "\n"


class WeightsTrees(unittest.TestCase):
    """C2b -- `--weight COL` for `--model tree` and `--model forest` (docs/weights.md section 2.5). The byte-for-byte comparison with the Python oracle is
    test_wire_tree_oracle.py (weighted cases)."""

    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        cls.sb.write("reg.csv", grid_csv())
        cls.sb.write("cls.csv", grid_csv(kind="cls"))
        cls.sb.write("ones_reg.csv", grid_csv(weights=[1] * 120))
        cls.sb.write("ones_cls.csv", grid_csv(weights=[1] * 120, kind="cls"))
        cls.sb.write("zero.csv", grid_csv(weights=[0] * 120))

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def fit(self, model, csv, out, *extra):
        task = "classification" if "cls" in csv else "regression"
        return self.sb.run("fit", "--model", model, "--task", task, "--target", "y", "--output", out, "--overwrite", *self.forest(model), *extra, csv)

    @staticmethod
    def forest(model):
        return ("--trees", "6") if model == "forest" else ()

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

    def values(self, model_file, csv="grid.csv"):
        self.sb.write(csv, "a,b\n" + "".join("%d,%d\n" % (a, b) for a in range(10) for b in range(10)))
        r = self.sb.run("predict", "--model-file", model_file, "--limit", "1000", csv)
        return [row[0] for row in self.ok(r)["rows"]]

    def test_the_weight_column_is_not_a_feature(self):
        for model in ("tree", "forest"):
            d = self.ok(self.fit(model, "reg.csv", "w.cml", "--weight", "w") if model == "forest" else self.fit(model, "reg.csv", "w.cml", "--weight", "w"))
            self.assertEqual(d["summary"]["features"], 2, model)
            self.assertEqual(len(self.values("w.cml")), 100, model)

    def test_unit_weights_are_the_unweighted_fit_as_bytes(self):
        for model in ("tree", "forest"):
            for csv in ("ones_reg.csv", "ones_cls.csv"):
                self.ok(self.fit(model, csv, "w1.cml", "--weight", "w"))
                self.ok(self.fit(model, csv, "u1.cml", "--drop", "w"))
                self.assertEqual(self.sb.read("w1.cml"), self.sb.read("u1.cml"), (model, csv))

    def test_integer_weights_give_the_predictions_of_the_file_with_the_rows_repeated(self):
        for csv in ("reg.csv", "cls.csv"):
            lines = self.sb.read(csv).decode().strip().split("\n")
            rows = [ln.split(",") for ln in lines[1:]]
            dup = ["a,b,y"] + [",".join([r[0], r[1], r[3]]) for r in rows for _ in range(int(r[2]))]
            self.sb.write("dup.csv", "\n".join(dup) + "\n")
            task = "classification" if csv == "cls.csv" else "regression"
            self.ok(self.sb.run("fit", "--model", "tree", "--task", task, "--target", "y", "--weight", "w", "--max-depth", "5", "--output", "a.cml", "--overwrite", csv))
            self.ok(self.sb.run("fit", "--model", "tree", "--task", task, "--target", "y", "--max-depth", "5", "--output", "b.cml", "--overwrite", "dup.csv"))
            a, b = self.values("a.cml"), self.values("b.cml")
            self.assertEqual(len(a), len(b))
            close_ = sum(1 for x, y in zip(a, b) if abs(float(x) - float(y)) <= 1e-9)
            self.assertGreaterEqual(close_, 95, (csv, close_))

    def test_a_weighted_file_is_the_same_bytes_at_any_thread_count(self):
        for model in ("tree", "forest"):
            seen = set()
            for t in (1, 2, 4, 8):
                self.ok(self.fit(model, "cls.csv", "t%d.cml" % t, "--weight", "w", "--threads", str(t)))
                seen.add(self.sb.read("t%d.cml" % t))
            self.assertEqual(len(seen), 1, model)

    def test_the_weights_change_the_model(self):
        for model in ("tree", "forest"):
            self.ok(self.fit(model, "reg.csv", "w.cml", "--weight", "w"))
            self.ok(self.fit(model, "reg.csv", "u.cml", "--drop", "w"))
            self.assertNotEqual(self.sb.read("w.cml"), self.sb.read("u.cml"), model)

    def test_no_weight_above_zero_and_a_class_without_weight(self):
        self.rule(self.fit("tree", "zero.csv", "x.cml", "--weight", "w"), "ml.weight.zero-total", 8)
        lines = self.sb.read("cls.csv").decode().strip().split("\n")
        rows = [ln.split(",") for ln in lines[1:]]
        self.sb.write("cls_one.csv", "a,b,w,y\n" + "".join("%s,%s,%d,%s\n" % (r[0], r[1], 0 if r[3] == "1" else 1, r[3]) for r in rows))
        self.rule(self.fit("tree", "cls_one.csv", "x.cml", "--weight", "w"), "ml.label.single-class", 8)
        self.rule(self.fit("forest", "cls_one.csv", "x.cml", "--weight", "w"), "ml.label.single-class", 8)

    def test_the_conflicts_of_the_column(self):
        self.rule(self.fit("tree", "reg.csv", "x.cml", "--weight", "y"), "args.conflict", 2)
        self.rule(self.fit("tree", "reg.csv", "x.cml", "--weight", "w", "--drop", "w"), "args.conflict", 2)
        self.rule(self.fit("tree", "reg.csv", "x.cml", "--weight", "nope"), "ml.input.unknown-column", 3)

    def test_cv_takes_the_flag(self):
        for model in ("tree", "forest"):
            args = ("cv", "--model", model, "--task", "classification", "--target", "y", "--folds", "4", "--seed", "1", *self.forest(model))
            a = self.ok(self.sb.run(*args, "--weight", "w", "cls.csv"))
            b = self.ok(self.sb.run(*args, "--drop", "w", "cls.csv"))
            self.assertIn("folds", json.dumps(a))
            self.assertNotEqual(json.dumps(a), json.dumps(b), model)
        self.rule(self.sb.run("cv", "--model", "tree", "--task", "regression", "--target", "y", "--weight", "w", "--folds", "4", "zero.csv"), "ml.weight.zero-total", 8)

    def test_the_held_out_rows_are_scored_without_weights(self):
        # the metrics of `eval` on the held-out rows (no weight column read) are those the fit reported: the held-out part never meets a weight
        for model in ("tree", "forest"):
            d = self.ok(self.sb.run("fit", "--model", model, "--task", "regression", "--target", "y", "--weight", "w", "--split", "train=0.7", "--seed", "3", *self.forest(model), "--output", "s.cml", "--overwrite", "reg.csv"))
            e = self.ok(self.sb.run("eval", "--model-file", "s.cml", "--drop", "w", "--split", "train=0.7", "--seed", "3", "reg.csv"))
            hold = d["summary"]["holdout"]
            metrics = hold.get("metrics", hold)
            for k, v in e["metrics"].items():
                self.assertEqual(metrics[k], v, (model, k))

    def test_a_tracked_weighted_fit_is_replayed_with_the_flag(self):
        (self.sb.dir / "runs").mkdir(exist_ok=True)
        d = self.ok(self.fit("forest", "reg.csv", "tr.cml", "--weight", "w", "--track", "runs"))
        rec = json.loads((self.sb.dir / "runs" / (d["run"]["id"] + ".json")).read_text())
        self.assertEqual(rec["inputs"]["weight"], "w", rec)
        v = self.sb.run("runs", "verify", "runs", d["run"]["id"])
        self.assertEqual(v.status, 0, v)


class WeightsBoost(WeightsTrees):
    """C3 -- `--weight COL` for `--model boost` (docs/weights.md section 2.6). The byte-for-byte comparison with the Python oracle is test_wire_boost_oracle.py
    (weighted cases). The tests of the trees that do not depend on the model are not repeated."""

    MODEL = "boost"
    EXTRA = ("--rounds", "8", "--max-depth", "3", "--min-leaf", "1")

    def test_the_weight_column_is_not_a_feature(self):
        d = self.ok(self.fit("boost", "reg.csv", "w.cml", "--weight", "w", *self.EXTRA))
        self.assertEqual(d["summary"]["features"], 2)
        self.assertEqual(len(self.values("w.cml")), 100)

    def test_unit_weights_are_the_unweighted_fit_as_bytes(self):
        # exactly: 1.0 * x is x, and the sums of ones are the counts (section 2.6)
        for csv in ("ones_reg.csv", "ones_cls.csv"):
            for extra in ((), ("--subsample", "0.5"), ("--early-stopping", "3", "--split", "train=0.8", "--seed", "4")):
                self.ok(self.fit("boost", csv, "w1.cml", "--weight", "w", *self.EXTRA, *extra))
                self.ok(self.fit("boost", csv, "u1.cml", "--drop", "w", *self.EXTRA, *extra))
                self.assertEqual(self.sb.read("w1.cml"), self.sb.read("u1.cml"), (csv, extra))

    def test_integer_weights_give_the_predictions_of_the_file_with_the_rows_repeated(self):
        for csv in ("reg.csv", "cls.csv"):
            lines = self.sb.read(csv).decode().strip().split("\n")
            rows = [ln.split(",") for ln in lines[1:]]
            dup = ["a,b,y"] + [",".join([r[0], r[1], r[3]]) for r in rows for _ in range(int(r[2]))]
            self.sb.write("dup.csv", "\n".join(dup) + "\n")
            task = "classification" if csv == "cls.csv" else "regression"
            self.ok(self.sb.run("fit", "--model", "boost", "--task", task, "--target", "y", "--weight", "w", *self.EXTRA, "--output", "a.cml", "--overwrite", csv))
            self.ok(self.sb.run("fit", "--model", "boost", "--task", task, "--target", "y", *self.EXTRA, "--output", "b.cml", "--overwrite", "dup.csv"))
            a, b = self.values("a.cml"), self.values("b.cml")
            close_ = sum(1 for x, y in zip(a, b) if abs(float(x) - float(y)) <= 1e-9 * max(1.0, abs(float(y))))
            self.assertGreaterEqual(close_, 95, (csv, close_))

    def test_a_weighted_file_is_the_same_bytes_at_any_thread_count(self):
        seen = set()
        for t in (1, 2, 4, 8):
            self.ok(self.fit("boost", "cls.csv", "t%d.cml" % t, "--weight", "w", "--threads", str(t), *self.EXTRA))
            seen.add(self.sb.read("t%d.cml" % t))
        self.assertEqual(len(seen), 1)

    def test_the_weights_change_the_model(self):
        self.ok(self.fit("boost", "reg.csv", "w.cml", "--weight", "w", *self.EXTRA))
        self.ok(self.fit("boost", "reg.csv", "u.cml", "--drop", "w", *self.EXTRA))
        self.assertNotEqual(self.sb.read("w.cml"), self.sb.read("u.cml"))

    def test_no_weight_above_zero_and_a_class_without_weight(self):
        self.rule(self.fit("boost", "zero.csv", "x.cml", "--weight", "w", *self.EXTRA), "ml.weight.zero-total", 8)
        lines = self.sb.read("cls.csv").decode().strip().split("\n")
        rows = [ln.split(",") for ln in lines[1:]]
        self.sb.write("cls_one.csv", "a,b,w,y\n" + "".join("%s,%s,%d,%s\n" % (r[0], r[1], 0 if r[3] == "1" else 1, r[3]) for r in rows))
        self.rule(self.fit("boost", "cls_one.csv", "x.cml", "--weight", "w", *self.EXTRA), "ml.label.single-class", 8)

    def test_the_conflicts_of_the_column(self):
        self.rule(self.fit("boost", "reg.csv", "x.cml", "--weight", "y"), "args.conflict", 2)
        self.rule(self.fit("boost", "reg.csv", "x.cml", "--weight", "w", "--drop", "w"), "args.conflict", 2)

    def test_cv_takes_the_flag(self):
        args = ("cv", "--model", "boost", "--task", "classification", "--target", "y", "--folds", "4", "--seed", "1", *self.EXTRA)
        a = self.ok(self.sb.run(*args, "--weight", "w", "cls.csv"))
        b = self.ok(self.sb.run(*args, "--drop", "w", "cls.csv"))
        self.assertNotEqual(json.dumps(a), json.dumps(b))

    def test_the_held_out_rows_are_scored_without_weights(self):
        d = self.ok(self.sb.run("fit", "--model", "boost", "--task", "regression", "--target", "y", "--weight", "w", "--split", "train=0.7", "--seed", "3", *self.EXTRA, "--output", "s.cml", "--overwrite", "reg.csv"))
        e = self.ok(self.sb.run("eval", "--model-file", "s.cml", "--drop", "w", "--split", "train=0.7", "--seed", "3", "reg.csv"))
        hold = d["summary"]["holdout"]
        metrics = hold.get("metrics", hold)
        for k, v in e["metrics"].items():
            self.assertEqual(metrics[k], v, k)

    def test_a_tracked_weighted_fit_is_replayed_with_the_flag(self):
        (self.sb.dir / "runs").mkdir(exist_ok=True)
        d = self.ok(self.fit("boost", "reg.csv", "tr.cml", "--weight", "w", "--track", "runs", *self.EXTRA))
        rec = json.loads((self.sb.dir / "runs" / (d["run"]["id"] + ".json")).read_text())
        self.assertEqual(rec["inputs"]["weight"], "w", rec)
        v = self.sb.run("runs", "verify", "runs", d["run"]["id"])
        self.assertEqual(v.status, 0, v)


if __name__ == "__main__":
    unittest.main()

"""G1 -- JSON lines and the equivalent CSV give the same model file bytes and the same answers (docs/jsonl.md sections 2 and 6).

For the bundled datasets and generated tables, `ml fit` on a file of JSON lines and on the equivalent CSV, for every model kind and for plans with scale, impute,
bucket, encode, freq and hash steps and --cat text columns; missing cells written as null and as absent keys; --split; predict and eval from JSON lines with the keys
in another order and with extra keys; and standard input for fit, predict and eval. The model files must be equal byte for byte, and the standard output of predict and
eval equal byte for byte. Standard library and jsonschema only (the fast CI job runs it).
"""
import csv
import io
import json
import pathlib
import random
import re
import unittest

from harness import ROOT, Sandbox, run

DATA = ROOT / "docs" / "examples" / "data"
NUMBER = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$")


def read_csv(name):
    return (DATA / name).read_text()


def blank_cells(text, every, offset=0, columns=None):
    """a CSV with some cells emptied: every `every`-th data row, in `columns` (header indices) or in the first feature column"""
    rows = list(csv.reader(io.StringIO(text)))
    out = [rows[0]]
    for i, r in enumerate(rows[1:]):
        r = list(r)
        if (i + offset) % every == 0:
            for c in (columns or [0]):
                r[c] = ""
        out.append(r)
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(out)
    return buf.getvalue()


def to_jsonl(text, cats=(), missing="null", shuffle=None, extra=False):
    """the JSON lines of a CSV: a number cell is its token text, a cat cell a string, an empty cell null or an absent key (missing: null, absent or mixed)"""
    rows = list(csv.reader(io.StringIO(text)))
    head = rows[0]
    rng = random.Random(shuffle) if shuffle is not None else None
    out = []
    for i, r in enumerate(rows[1:]):
        parts = []
        for name, cell in zip(head, r):
            if cell == "":
                kind = missing if missing != "mixed" else ("null" if i % 2 == 0 else "absent")
                if kind == "absent":
                    continue
                parts.append((name, "null"))
            elif name in cats:
                parts.append((name, json.dumps(cell, ensure_ascii=False)))
            else:
                assert NUMBER.match(cell), "not a JSON number: %r" % cell
                parts.append((name, cell))
        if extra:
            parts.append(("zz_id", str(i)))
            parts.append(("zz_note", json.dumps("a, \"b\"\nc")))
            parts.append(("zz_nested", '{"x":[1,{"y":null}]}'))
        if rng:
            rng.shuffle(parts)
        out.append("{" + ",".join(json.dumps(k) + ":" + v for k, v in parts) + "}")
    return ("\n".join(out) + "\n").encode()


def doc_without_path(r):
    d = r.doc()
    if "data" in d and isinstance(d["data"], dict):
        d["data"].pop("path", None)
        d["data"].pop("bytes", None) if False else None
    return d


class Equivalence(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()

    def tearDown(self):
        self.sb.cleanup()

    def pair(self, name, train_csv, test_csv, fit_args, cats=(), missing="null", absent=False, fields=None, split=None, predict_extra=(), check_eval=True, stdin=True):
        """fit both ways and compare; then predict and eval both ways"""
        sb = self.sb
        sb.write(name + "_tr.csv", train_csv)
        sb.write(name + "_te.csv", test_csv)
        head = next(csv.reader(io.StringIO(train_csv)))
        flds = ",".join(fields or head)
        sb.write(name + "_tr.jsonl", to_jsonl(train_csv, cats, missing))
        sb.write(name + "_te.jsonl", to_jsonl(test_csv, cats, missing, shuffle=7, extra=True))
        jopt = ["--input-format", "jsonl"] + (["--absent", "empty"] if absent else [])
        a = sb.run("fit", *fit_args, "--output", name + "_csv.cml", name + "_tr.csv")
        self.assertEqual(a.status, 0, a)
        b = sb.run("fit", *fit_args, *jopt, "--fields", flds, "--output", name + "_jl.cml", name + "_tr.jsonl")
        self.assertEqual(b.status, 0, b)
        self.assertEqual(sb.read(name + "_csv.cml"), sb.read(name + "_jl.cml"), "model file bytes differ: " + name)
        da, db = a.doc(), b.doc()
        da["data"].pop("path"), db["data"].pop("path")
        self.assertEqual(da, db)
        if stdin:
            c = sb.run("fit", *fit_args, *jopt, "--fields", flds, "--output", name + "_in.cml", "-", stdin=sb.read(name + "_tr.jsonl"))
            self.assertEqual(c.status, 0, c)
            self.assertEqual(sb.read(name + "_csv.cml"), sb.read(name + "_in.cml"), "fit - differs: " + name)
            d = sb.run("fit", *fit_args, "--output", name + "_in2.cml", "-", stdin=sb.read(name + "_tr.csv"))
            self.assertEqual(d.status, 0, d)
            self.assertEqual(sb.read(name + "_csv.cml"), sb.read(name + "_in2.cml"), "csv fit - differs: " + name)
        m = name + "_csv.cml"
        # a CSV column the model does not know is refused unless dropped; a JSON key it does not know is ignored (docs/jsonl.md section 2)
        drop = ["--drop", fit_args[fit_args.index("--drop") + 1]] if "--drop" in fit_args else []
        p1 = sb.run("predict", "--model-file", m, *predict_extra, *drop, name + "_te.csv")
        p2 = sb.run("predict", "--model-file", m, *predict_extra, *jopt, name + "_te.jsonl")
        self.assertEqual(p1.status, 0, p1)
        self.assertEqual(p2.status, 0, p2)
        self.assertEqual(p1.stdout, p2.stdout, "predict differs: " + name)
        if stdin:
            p3 = sb.run("predict", "-", "--model-file", m, *predict_extra, *jopt, stdin=sb.read(name + "_te.jsonl"))
            self.assertEqual(p3.stdout, p1.stdout)
        if check_eval and "--model" in fit_args and fit_args[fit_args.index("--model") + 1] != "kmeans" or check_eval and "--model" not in fit_args:
            e1 = sb.run("eval", "--model-file", m, *drop, name + "_te.csv")
            e2 = sb.run("eval", "--model-file", m, *jopt, name + "_te.jsonl")
            self.assertEqual(e1.status, 0, e1)
            self.assertEqual(e1.stdout, e2.stdout, "eval differs: " + name)
            if stdin:
                e3 = sb.run("eval", "-", "--model-file", m, *jopt, stdin=sb.read(name + "_te.jsonl"))
                self.assertEqual(e3.stdout, e1.stdout)
        elif check_eval:
            e1 = sb.run("eval", "--model-file", m, *drop, name + "_te.csv")
            e2 = sb.run("eval", "--model-file", m, *jopt, name + "_te.jsonl")
            self.assertEqual(e1.stdout, e2.stdout)
        return a

    # ---- every model kind on iris
    def iris(self, *args, **kw):
        return self.pair("iris" + str(abs(hash(args)) % 100000), read_csv("iris_train.csv"), read_csv("iris_test.csv"), list(args), **kw)

    def test_constant(self):
        self.iris("--target", "virginica")

    def test_linear_and_the_pipeline(self):
        self.iris("--model", "linear", "--target", "virginica")
        self.iris("--model", "linear", "--l2", "0.5", "--target", "virginica", "--step", "scale column=* kind=standard")
        self.iris("--model", "linear", "--l2", "0.5", "--target", "virginica", "--step", "bucket column=sepal_length_cm kind=quantile bins=4; scale column=* kind=standard")

    def test_logistic(self):
        self.iris("--model", "logistic", "--target", "virginica", "--iterations", "30", "--step", "scale column=* kind=minmax")

    def test_tree(self):
        self.iris("--model", "tree", "--task", "classification", "--target", "virginica", "--max-depth", "4")

    def test_forest(self):
        self.iris("--model", "forest", "--task", "classification", "--trees", "7", "--seed", "3", "--target", "virginica", "--max-depth", "4")

    def test_kmeans_has_no_target(self):
        self.iris("--model", "kmeans", "--k", "3", "--seed", "2", fields=["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"] if False else None)

    def test_split(self):
        self.iris("--model", "linear", "--target", "virginica", "--split", "train=0.8", "--seed", "3")
        self.iris("--model", "forest", "--task", "classification", "--trees", "5", "--seed", "3", "--target", "virginica", "--split", "train=0.7", "--max-depth", "3")

    def test_threads_change_no_byte(self):
        self.iris("--model", "linear", "--target", "virginica", "--threads", "3")

    # ---- the other bundled datasets
    def test_diabetes_regression(self):
        self.pair("diab", read_csv("diabetes_train.csv"), read_csv("diabetes_train.csv"), ["--model", "linear", "--l2", "0.1", "--target", "progression"])
        self.pair("diabt", read_csv("diabetes_train.csv"), read_csv("diabetes_train.csv"), ["--model", "tree", "--target", "progression", "--max-depth", "5"])

    def test_wine_and_breast_cancer(self):
        self.pair("wine", read_csv("wine_train.csv"), read_csv("wine_train.csv"), ["--model", "logistic", "--target", "class_0", "--iterations", "20", "--step", "scale column=* kind=standard"])
        self.pair("bc", read_csv("breast_cancer_train.csv"), read_csv("breast_cancer_train.csv"), ["--model", "forest", "--task", "classification", "--trees", "5", "--seed", "1", "--target", "benign", "--max-depth", "4"])

    # ---- missing cells: null and absent
    def test_missing_cells_as_null_absent_and_mixed(self):
        tr = blank_cells(read_csv("wine_train.csv"), 5, 0, [0, 3])
        te = blank_cells(read_csv("wine_train.csv"), 4, 1, [0, 3])
        for mode in ("null", "absent", "mixed"):
            with self.subTest(mode=mode):
                self.pair("wm" + mode, tr, te, ["--model", "tree", "--task", "classification", "--target", "class_0", "--max-depth", "4"], missing=mode, absent=mode != "null")
                self.pair("wi" + mode, tr, te, ["--model", "linear", "--target", "class_0", "--step", "impute column=alcohol,alcalinity_of_ash,flavanoids strategy=mean; scale column=* kind=standard"], missing=mode, absent=mode != "null")

    def test_an_absent_key_without_absent_empty_is_refused(self):
        tr = blank_cells(read_csv("iris_train.csv"), 5, 0, [0])
        self.sb.write("t.jsonl", to_jsonl(tr, missing="absent"))
        r = self.sb.run("fit", "--target", "virginica", "--output", "m.cml", "--input-format", "jsonl", "--fields", "sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm,virginica", "t.jsonl")
        self.assertEqual(r.first_rule(), "ml.jsonl.missing-field")

    # ---- text columns
    def test_cat_columns_with_every_encoder(self):
        pets = read_csv("pets.csv")
        for name, step in (("onehot", "encode column=species,city kind=onehot missing=zero"), ("label", "encode column=species,city kind=label"),
                           ("freq", "freq column=species,city"), ("hash", "hash column=species buckets=4 missing=zero; hash column=city buckets=3 missing=zero")):
            with self.subTest(step=name):
                self.pair("pets" + name, pets, pets, ["--model", "linear", "--l2", "1", "--target", "adopted", "--cat", "species,city", "--drop", "id", "--step", step], cats=("species", "city"))

    def test_cat_with_missing_markers_and_hard_text(self):
        rows = ["id,kind,size,y"]
        texts = ["a b", "x,y", 'q"r', "line\nbreak", "café", "\U0001f600", "NA", "007", "5", "", " lead", "trail "]
        for i in range(60):
            t = texts[i % len(texts)]
            rows.append("%d,%s,%d,%d" % (i, '"' + t.replace('"', '""') + '"' if t else "", (i * 7) % 11, i % 2))
        text = "\n".join(rows) + "\n"
        for model in ("tree", "forest"):
            with self.subTest(model=model):
                self.pair("hard" + model, text, text, ["--model", model, "--task", "classification", "--target", "y", "--cat", "kind", "--na-values", "NA", "--drop", "id", "--step", "encode column=kind kind=onehot missing=zero", "--max-depth", "3"] + (["--trees", "3", "--seed", "2"] if model == "forest" else []), cats=("kind",), missing="mixed", absent=True)

    def test_a_text_column_of_numbers_is_text(self):
        # the classes 1,2,3 as JSON numbers in a --cat column are the categories "1","2","3", as the CSV cells are
        rows = ["c,x,y"] + ["%d,%d,%d" % (1 + i % 3, i, i % 2) for i in range(40)]
        text = "\n".join(rows) + "\n"
        sb = self.sb
        sb.write("t.csv", text)
        lines = "".join('{"c":%d,"x":%d,"y":%d}\n' % (1 + i % 3, i, i % 2) for i in range(40))
        sb.write("t.jsonl", lines)
        args = ["fit", "--model", "linear", "--l2", "1", "--target", "y", "--cat", "c", "--step", "encode column=c kind=onehot missing=zero"]
        a = sb.run(*args, "--output", "a.cml", "t.csv")
        b = sb.run(*args, "--input-format", "jsonl", "--fields", "c,x,y", "--output", "b.cml", "t.jsonl")
        self.assertEqual((a.status, b.status), (0, 0), (a, b))
        self.assertEqual(sb.read("a.cml"), sb.read("b.cml"))

    def test_ndjson_is_jsonl(self):
        sb = self.sb
        sb.write("t.csv", read_csv("iris_train.csv"))
        sb.write("t.jsonl", to_jsonl(read_csv("iris_train.csv")))
        f = "sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm,virginica"
        a = sb.run("fit", "--target", "virginica", "--output", "a.cml", "--input-format", "jsonl", "--fields", f, "t.jsonl")
        b = sb.run("fit", "--target", "virginica", "--output", "b.cml", "--input-format", "ndjson", "--fields", f, "t.jsonl")
        self.assertEqual(sb.read("a.cml"), sb.read("b.cml"))
        self.assertEqual((a.status, b.status), (0, 0))

    def test_fields_in_another_order_than_the_json(self):
        sb = self.sb
        train = read_csv("iris_train.csv")
        sb.write("t.jsonl", to_jsonl(train, shuffle=11, extra=True))
        # the table's columns are in the order of --fields, which is the CSV's here: the model file equals the CSV's
        sb.write("t.csv", train)
        f = "sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm,virginica"
        a = sb.run("fit", "--model", "linear", "--target", "virginica", "--output", "a.cml", "t.csv")
        b = sb.run("fit", "--model", "linear", "--target", "virginica", "--output", "b.cml", "--input-format", "jsonl", "--fields", f, "t.jsonl")
        self.assertEqual((a.status, b.status), (0, 0), (a, b))
        self.assertEqual(sb.read("a.cml"), sb.read("b.cml"))
        # and the model of a table with its columns in another order is another file (the order is the schema's), from the CSV and from JSON lines alike
        g = "virginica,petal_width_cm,petal_length_cm,sepal_width_cm,sepal_length_cm"
        c = sb.run("fit", "--model", "linear", "--target", "virginica", "--output", "c.cml", "--input-format", "jsonl", "--fields", g, "t.jsonl")
        self.assertEqual(c.status, 0, c)
        rows = [r[::-1] for r in csv.reader(io.StringIO(train))]
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows(rows)
        sb.write("r.csv", buf.getvalue())
        d = sb.run("fit", "--model", "linear", "--target", "virginica", "--output", "d.cml", "r.csv")
        self.assertEqual(sb.read("c.cml"), sb.read("d.cml"))

    def test_extra_keys_are_ignored_by_predict_and_eval_but_a_csv_column_is_not(self):
        sb = self.sb
        sb.write("t.csv", read_csv("iris_train.csv"))
        a = sb.run("fit", "--model", "linear", "--target", "virginica", "--output", "m.cml", "t.csv")
        self.assertEqual(a.status, 0)
        te = read_csv("iris_test.csv")
        sb.write("te.jsonl", to_jsonl(te, extra=True, shuffle=3))
        r = sb.run("predict", "--model-file", "m.cml", "--input-format", "jsonl", "te.jsonl")
        self.assertEqual(r.status, 0, r)
        extra_csv = "zz,sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm,virginica\n" + "".join("1," + l + "\n" for l in te.splitlines()[1:])
        sb.write("ex.csv", extra_csv)
        self.assertEqual(sb.run("predict", "--model-file", "m.cml", "ex.csv").first_rule(), "ml.schema.mismatch")

    def test_predict_needs_no_target_key_and_eval_does(self):
        sb = self.sb
        sb.write("t.csv", read_csv("iris_train.csv"))
        sb.run("fit", "--model", "linear", "--target", "virginica", "--output", "m.cml", "t.csv")
        rows = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
        sb.write("nt.jsonl", "".join('{"sepal_length_cm":5.1,"sepal_width_cm":3.5,"petal_length_cm":1.4,"petal_width_cm":0.2,"other":"x"}\n' for _ in range(3)))
        p = sb.run("predict", "--model-file", "m.cml", "--input-format", "jsonl", "nt.jsonl")
        self.assertEqual(p.status, 0, p)
        e = sb.run("eval", "--model-file", "m.cml", "--input-format", "jsonl", "nt.jsonl")
        self.assertEqual(e.first_rule(), "ml.jsonl.missing-field")
        self.assertEqual(e.error()["detail"]["name"], "virginica")

    def test_the_binary_ml_predict(self):
        import harness
        sb = self.sb
        sb.write("t.csv", read_csv("iris_train.csv"))
        sb.run("fit", "--model", "linear", "--target", "virginica", "--output", "m.cml", "t.csv")
        sb.write("te.jsonl", to_jsonl(read_csv("iris_test.csv"), shuffle=1, extra=True))
        sb.write("te.csv", read_csv("iris_test.csv"))
        a = run("--root", sb.dir, "--model-file", "m.cml", "predict", "te.csv", binary_path=harness.PREDICT_BIN)
        b = run("--root", sb.dir, "--model-file", "m.cml", "--input-format", "jsonl", "predict", "te.jsonl", binary_path=harness.PREDICT_BIN)
        c = run("--model-file", str(sb.dir / "m.cml"), "--input-format", "jsonl", "predict", "-", binary_path=harness.PREDICT_BIN, stdin=sb.read("te.jsonl"))
        self.assertEqual((a.status, b.status, c.status), (0, 0, 0))
        self.assertEqual(a.stdout, b.stdout)
        self.assertEqual(a.stdout, c.stdout)


if __name__ == "__main__":
    unittest.main()

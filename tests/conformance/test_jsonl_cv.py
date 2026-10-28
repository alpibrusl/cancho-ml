"""JSON lines meet the splits and the run records of the merge (docs/jsonl.md sections 2 and 4): `ml cv` on JSON lines gives the bytes of `ml cv` on the equivalent CSV for every
strategy (the group and the time columns come from JSON strings and numbers), `fit -` and `eval -` identify the bytes they read, a run record made from JSON lines holds the input format
and `ml runs verify` replays it at 1 and 8 threads, a record of standard input is not replayable, and a CSV record is what it was. Standard library only (the fast CI job runs it).
"""
import hashlib
import json
import unittest

from harness import Sandbox
from test_jsonl_equiv import read_csv, to_jsonl

VISITS = read_csv("visits.csv")
FIELDS = "customer,visit_date,basket,items,returning"
CV = ["cv", "--model", "tree", "--task", "classification", "--max-depth", "3", "--target", "returning", "--folds", "4", "--drop", "customer,visit_date"]
JL = ["--input-format", "jsonl", "--fields", FIELDS]
# the iris fit of the README (the run record of docs/tracking.md): its run id is the one the CSV path has always written
IRIS = ["fit", "--model", "logistic", "--target", "virginica", "--iterations", "25", "--l2", "0", "--output", "iris-lr.ml", "--track", "runs", "iris_train.csv"]
IRIS_RUN_ID = "34451ac49753dd960bde18cd6a2e0cb128d5a9f92ae4691a4f8c2c4bc765bc63"
LIN_FIELDS = "basket,items,returning"
LIN = ["fit", "--model", "linear", "--target", "returning", "--output", "m.cml", "--track", "runs"]
LIN_JL = ["--input-format", "jsonl", "--fields", LIN_FIELDS]


def numeric_dates(text):
    """the visits with 2024-02-12 written 20240212: a date that is a number"""
    rows = text.splitlines()
    out = [rows[0]]
    for r in rows[1:]:
        c = r.split(",")
        c[1] = c[1].replace("-", "")
        out.append(",".join(c))
    return "\n".join(out) + "\n"


class CrossValidation(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        self.sb.write("v.csv", VISITS)
        self.sb.write("v.jsonl", to_jsonl(VISITS, cats=("customer", "visit_date")))

    def tearDown(self):
        self.sb.cleanup()

    def same(self, *extra, csv_name="v.csv", jsonl_name="v.jsonl", fields=FIELDS):
        a = self.sb.run(*CV, *extra, csv_name)
        b = self.sb.run(*CV, "--input-format", "jsonl", "--fields", fields, *extra, jsonl_name)
        self.assertEqual(a.status, 0, a)
        self.assertEqual(b.status, 0, b)
        self.assertEqual(a.stdout, b.stdout, extra)
        return a

    def test_every_strategy_gives_the_bytes_of_the_csv(self):
        for extra in ((), ("--stratify",), ("--stratify", "--repeats", "2", "--seed", "3"), ("--split-group", "customer"), ("--split-group", "customer", "--emit-folds", "--seed", "5"),
                      ("--split-by", "time:visit_date"), ("--split-by", "time:visit_date", "--emit-folds")):
            with self.subTest(extra=extra):
                self.same(*extra)

    def test_a_date_that_is_a_json_string_is_text_and_a_date_that_is_a_json_number_is_a_number(self):
        # a string date: text in the mapping, parsed by --split-by as the CSV cell is (an ISO date); the ISO date and the same dates as numbers cut the rows the same way
        a = self.same("--split-by", "time:visit_date")
        self.assertEqual(a.doc()["data"]["strategy"], "time")
        nd = numeric_dates(VISITS)
        self.sb.write("n.csv", nd)
        self.sb.write("n.jsonl", to_jsonl(nd, cats=("customer",)))
        n = self.same("--split-by", "time:visit_date", csv_name="n.csv", jsonl_name="n.jsonl")
        fold = lambda r: [(x["train"], x["test"]) for x in r.doc()["data"]["results"]]
        self.assertEqual(fold(a), fold(n))
        # a column that is both a string in some lines and a number in others is refused, not guessed
        lines = self.sb.read("v.jsonl").decode().splitlines()
        lines[1] = lines[1].replace('"visit_date":"2024-03-20"', '"visit_date":20240320')
        self.sb.write("m.jsonl", "\n".join(lines) + "\n")
        r = self.sb.run(*CV, *JL, "--split-by", "time:visit_date", "m.jsonl")
        self.assertEqual(r.first_rule(), "ml.jsonl.mixed-kinds")

    def test_the_time_column_of_a_string_that_is_not_a_date_is_the_splits_refusal(self):
        lines = self.sb.read("v.jsonl").decode().splitlines()
        lines[2] = lines[2].replace('"visit_date":"2024-03-08"', '"visit_date":"8 March"')
        self.sb.write("bad.jsonl", "\n".join(lines) + "\n")
        r = self.sb.run(*CV, *JL, "--split-by", "time:visit_date", "bad.jsonl")
        self.assertEqual(r.first_rule(), "ml.split.time-format")
        c = read_csv("visits.csv").splitlines()
        c[3] = c[3].replace("2024-03-08", "8 March")
        self.sb.write("bad.csv", "\n".join(c) + "\n")
        self.assertEqual(self.sb.run(*CV, "--split-by", "time:visit_date", "bad.csv").first_rule(), "ml.split.time-format")

    def test_cv_needs_fields_and_refuses_them_with_csv(self):
        r = self.sb.run(*CV, "--input-format", "jsonl", "v.jsonl")
        self.assertEqual(r.first_rule(), "args.required-flag")
        self.assertEqual(r.error()["detail"]["flag"], "--fields")
        r = self.sb.run(*CV, "--fields", FIELDS, "v.csv")
        self.assertEqual(r.first_rule(), "args.conflict")

    def test_a_refusal_of_a_line_names_the_line(self):
        self.sb.write("bad.jsonl", b'{"basket":1,"items":2,"returning":0}\n{"basket":1,"items":\n')
        r = self.sb.run("cv", "--model", "linear", "--target", "returning", "--input-format", "jsonl", "--fields", LIN_FIELDS, "bad.jsonl")
        self.assertEqual(r.first_rule(), "ml.jsonl.syntax")
        self.assertEqual(r.error()["detail"]["line"], 2)


class Tracking(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        self.sb.write("v.csv", VISITS)
        self.sb.write("v.jsonl", to_jsonl(VISITS, cats=("customer", "visit_date")))
        (self.sb.dir / "runs").mkdir()

    def tearDown(self):
        self.sb.cleanup()

    def record(self, rid):
        return json.loads((self.sb.dir / "runs" / (rid + ".json")).read_text())

    def test_a_csv_run_id_is_what_it_was(self):
        self.sb.write("iris_train.csv", read_csv("iris_train.csv"))
        r = self.sb.run(*IRIS)
        self.assertEqual(r.status, 0, r)
        self.assertEqual(r.doc()["data"]["run"]["id"], IRIS_RUN_ID)
        data = self.record(IRIS_RUN_ID)["inputs"]["data"]
        self.assertEqual(sorted(data), ["bytes", "path", "rows", "sha256"])

    def test_a_json_lines_record_holds_the_format_and_verify_replays_it_at_1_and_8_threads(self):
        r = self.sb.run(*LIN, *LIN_JL, "v.jsonl")
        self.assertEqual(r.status, 0, r)
        rid = r.doc()["data"]["run"]["id"]
        data = self.record(rid)["inputs"]["data"]
        self.assertEqual((data["path"], data["format"], data["fields"], data["absent"]), ("v.jsonl", "jsonl", LIN_FIELDS, "refuse"))
        for t in ("1", "8"):
            v = self.sb.run("runs", "verify", "--threads", t, "runs", rid)
            self.assertEqual(v.status, 0, v)
            self.assertTrue(v.doc()["data"]["verified"])
            self.assertEqual(v.doc()["data"]["threads"], int(t))
        # the run id is not the CSV's: the record names another format
        self.sb.write("c.csv", "basket,items,returning\n" + "".join("%s,%s,%s\n" % tuple(line.split(",")[2:]) for line in VISITS.splitlines()[1:]))
        c = self.sb.run(*LIN, "--overwrite", "c.csv")
        self.assertNotEqual(c.doc()["data"]["run"]["id"], rid)

    def test_the_replay_of_text_columns_and_of_an_absent_key(self):
        lines = self.sb.read("v.jsonl").decode().splitlines()
        lines[2] = lines[2].replace(',"basket":20.23', "")
        self.sb.write("g.jsonl", "\n".join(lines) + "\n")
        args = ["fit", "--model", "tree", "--task", "classification", "--target", "returning", "--drop", "customer,visit_date", "--input-format", "jsonl", "--fields", FIELDS,
                "--absent", "empty", "--output", "t.cml", "--track", "runs", "g.jsonl"]
        r = self.sb.run(*args)
        self.assertEqual(r.status, 0, r)
        rid = r.doc()["data"]["run"]["id"]
        self.assertEqual(self.record(rid)["inputs"]["data"]["absent"], "empty")
        for t in ("1", "8"):
            v = self.sb.run("runs", "verify", "--threads", t, "runs", rid)
            self.assertEqual(v.status, 0, v)
        # the file changed: the hash says so before anything is refitted
        self.sb.write("g.jsonl", self.sb.read("g.jsonl") + b'{"customer":"c01","visit_date":"2024-05-01","basket":1,"items":1,"returning":0}\n')
        v = self.sb.run("runs", "verify", "runs", rid)
        self.assertEqual(v.first_rule(), "ml.runs.data-changed")

    def test_standard_input_is_hashed_and_its_record_is_not_replayable(self):
        data = self.sb.read("v.jsonl")
        r = self.sb.run(*LIN, *LIN_JL, "-", stdin=data)
        self.assertEqual(r.status, 0, r)
        rid = r.doc()["data"]["run"]["id"]
        d = self.record(rid)["inputs"]["data"]
        self.assertEqual((d["path"], d["bytes"], d["sha256"], d["format"]), ("-", len(data), hashlib.sha256(data).hexdigest(), "jsonl"))
        v = self.sb.run("runs", "verify", "runs", rid)
        self.assertEqual(v.first_rule(), "ml.runs.not-replayable")
        self.assertEqual(v.error()["detail"], {"flag": "stdin", "column": "-"})
        # a CSV on standard input is hashed too, and is not replayable either
        csv_in = b"basket,items,returning\n1,2,0\n3,4,1\n5,7,1\n"
        c = self.sb.run("fit", "--model", "linear", "--target", "returning", "--output", "s.cml", "--track", "runs", "-", stdin=csv_in)
        self.assertEqual(c.status, 0, c)
        cid = c.doc()["data"]["run"]["id"]
        self.assertEqual(self.record(cid)["inputs"]["data"]["sha256"], hashlib.sha256(csv_in).hexdigest())
        self.assertEqual(self.sb.run("runs", "verify", "runs", cid).first_rule(), "ml.runs.not-replayable")

    def test_fit_from_a_file_and_from_standard_input_are_one_model_file(self):
        data = self.sb.read("v.jsonl")
        a = self.sb.run(*LIN, *LIN_JL, "v.jsonl")
        first = self.sb.read("m.cml")
        b = self.sb.run(*LIN, *LIN_JL, "--overwrite", "-", stdin=data)
        self.assertEqual((a.status, b.status), (0, 0), (a, b))
        self.assertEqual(first, self.sb.read("m.cml"))
        # eval says whether it scored the file the model was fitted on, by the bytes of the file or of the standard input
        e1 = self.sb.run("eval", "--model-file", "m.cml", "--input-format", "jsonl", "v.jsonl")
        e2 = self.sb.run("eval", "--model-file", "m.cml", "--input-format", "jsonl", "-", stdin=data)
        e3 = self.sb.run("eval", "--model-file", "m.cml", "--input-format", "jsonl", "-", stdin=data + b'{"basket":1,"items":1,"returning":0}\n')
        self.assertEqual([e.doc()["data"]["on"] for e in (e1, e2, e3)], ["train", "train", "other"])


if __name__ == "__main__":
    unittest.main()

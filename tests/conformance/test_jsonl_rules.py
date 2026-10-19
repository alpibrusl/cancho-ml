"""G3 -- the refusals of JSON lines input: one fixture per new rule tag (fixtures.py, `_jsonl_cases`), run here without the rest of the fixtures, with the facts each
refusal carries and the repair run (docs/jsonl.md sections 3 and 6)."""
import json
import unittest

from fixtures import CASES
from harness import Sandbox, run
from runner import judge, run_case

JSONL = [c for c in CASES if c["id"].startswith(("jsonl-", "ml-jsonl-"))]


class Fixtures(unittest.TestCase):
    def test_every_jsonl_fixture(self):
        self.assertGreaterEqual(len(JSONL), 24)
        failures = []
        for c in JSONL:
            sb, r = run_case(c)
            try:
                problems = judge(c, sb, r)
                if problems:
                    failures.append((c["id"], problems))
            finally:
                sb.cleanup()
        self.assertEqual(failures, [], "\n".join("%s: %s" % f for f in failures))

    def test_every_new_tag_has_a_fixture(self):
        tags = {"ml.jsonl.syntax", "ml.jsonl.not-object", "ml.jsonl.missing-field", "ml.jsonl.not-scalar", "ml.jsonl.duplicate-key", "ml.jsonl.type-mismatch", "ml.jsonl.mixed-kinds", "ml.jsonl.too-many-nodes"}
        self.assertEqual(tags - {c["expect"]["rule"] for c in JSONL}, set())


class Facts(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()

    def tearDown(self):
        self.sb.cleanup()

    def fit(self, text, *extra, fields="a,y", file="d.jsonl"):
        self.sb.write(file, text)
        return self.sb.run("fit", "--input-format", "jsonl", "--fields", fields, "--target", "y", "--output", "m.cml", *extra, file)

    def detail(self, r):
        return r.error()["detail"]

    def test_syntax_names_the_line_the_row_and_the_offset(self):
        r = self.fit('{"a":1,"y":1.5}\n\n{"a":2,"y":2.5}\n{"a":3 "y":4}\n')
        self.assertEqual(r.first_rule(), "ml.jsonl.syntax")
        d = self.detail(r)
        self.assertEqual((d["path"], d["line"], d["row"]), ("d.jsonl", 4, 3))
        self.assertEqual(d["offset"], 7)
        self.assertIn(d["code"], ("syntax", "trailing"))
        self.assertTrue(d["reason"])

    def test_not_object_names_the_kind(self):
        for text, kind in (("[1]", "array"), ('"x"', "string"), ("5", "number"), ("true", "boolean"), ("null", "null")):
            d = self.detail(self.fit('{"a":1,"y":1}\n' + text + "\n"))
            self.assertEqual((d["line"], d["kind"]), (2, kind))

    def test_missing_field_repair_runs(self):
        r = self.fit('{"a":1,"y":1.5}\n{"y":2.5}\n{"a":3,"y":3.5}\n', "--model", "tree")
        self.assertEqual(r.first_rule(), "ml.jsonl.missing-field")
        d = self.detail(r)
        self.assertEqual((d["line"], d["row"], d["column"], d["name"]), (2, 2, 1, "a"))
        rep = r.error()["repair"]
        self.assertEqual(rep["kind"], "retry")
        self.assertEqual(rep["argv"][-2:], ["--absent", "empty"])
        again = run(*rep["argv"][1:], cwd=self.sb.dir)
        self.assertEqual(again.status, 0, again)
        # the repair is not offered when --absent was given
        r2 = self.fit('{"a":1,"y":1.5}\n{"y":2.5}\n', "--model", "tree", "--absent", "refuse")
        self.assertIsNone(r2.error()["repair"])

    def test_null_is_a_missing_cell_even_when_absent_is_refused(self):
        r = self.fit('{"a":1,"y":1.5}\n{"a":null,"y":2.5}\n{"a":3,"y":3.5}\n', "--model", "tree")
        self.assertEqual(r.status, 0, r)

    def test_not_scalar_duplicate_and_type_mismatch_name_the_column(self):
        d = self.detail(self.fit('{"a":1,"y":{"q":1}}\n'))
        self.assertEqual((d["column"], d["name"], d["kind"]), (2, "y", "object"))
        d = self.detail(self.fit('{"a":1,"y":[1]}\n'))
        self.assertEqual(d["kind"], "array")
        d = self.detail(self.fit('{"y":1,"a":1,"y":2}\n'))
        self.assertEqual((d["name"], d["line"]), ("y", 1))
        d = self.detail(self.fit('{"a":1,"y":1}\n{"a":"2","y":1}\n'))
        self.assertEqual((d["name"], d["found"], d["line"]), ("a", "string", 2))
        d = self.detail(self.fit('{"a":false,"y":1}\n'))
        self.assertEqual(d["found"], "boolean")

    def test_mixed_kinds_name_both_lines(self):
        r = self.fit('{"c":"x","y":1}\n{"c":"z","y":2}\n{"c":7,"y":3}\n', "--cat", "c", "--model", "tree", "--step", "encode column=c kind=onehot", fields="c,y")
        d = self.detail(r)
        self.assertEqual((r.first_rule(), d["name"], d["first_kind"], d["first_line"], d["kind"], d["line"]), ("ml.jsonl.mixed-kinds", "c", "string", 1, "number", 3))

    def test_a_string_that_looks_like_a_number_is_text_never_a_number(self):
        # in a numeric column: refused. In a --cat column: a category.
        self.assertEqual(self.fit('{"a":"5","y":1}\n{"a":"6","y":2}\n').first_rule(), "ml.jsonl.type-mismatch")
        r = self.fit('{"a":"5","y":1}\n{"a":"6","y":2}\n{"a":"007","y":3}\n', "--cat", "a", "--model", "tree", "--step", "encode column=a kind=onehot")
        self.assertEqual(r.status, 0, r)

    def test_a_number_the_column_type_cannot_hold_is_the_csv_paths_refusal(self):
        r = self.fit('{"a":1,"y":1}\n{"a":2,"y":2}\n{"a":1e400,"y":3}\n')
        self.assertEqual(r.first_rule(), "ml.input.non-finite")
        r = self.fit('{"a":1,"y":1}\n{"a":123456789012345678901234567890,"y":2}\n', "--float", "a")
        self.assertEqual(r.status, 0, r)

    def test_line_numbers_of_the_csv_reader_are_the_generated_csv(self):
        # a text where a number is read: the CSV reader behind the converter refuses at the generated line (docs/jsonl.md section 4)
        r = self.fit('{"a":1,"y":1}\n{"a":2,"y":2}\n', "--cat", "a")
        self.assertIn(r.first_rule(), ("ml.input.categorical-not-encoded", "ml.plan.column-kind"))
        r = self.fit('{"a":1,"y":1}\n{"a":2,"y":2}\n{"a":1.5.5,"y":3}\n')
        self.assertEqual(r.first_rule(), "ml.jsonl.syntax")

    def test_the_input_bound_for_a_file_and_the_csv_bound(self):
        r = self.fit('{"a":1,"y":1.5}\n' * 10, "--max-bytes", "100")
        self.assertEqual(r.first_rule(), "limit.input-too-large")
        self.assertEqual(self.detail(r)["what"], "input")
        r = self.fit("{}\n" * 100, "--absent", "empty", "--max-bytes", "400", "--model", "tree", fields="a,y,b,c,d,e,f,g")
        self.assertEqual(r.first_rule(), "limit.input-too-large")
        self.assertEqual(self.detail(r)["what"], "csv")


if __name__ == "__main__":
    unittest.main()

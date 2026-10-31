"""`ml predict --output FILE` (cancho-ml#101): the rows go to a file, one per line as `--format text` prints them, and the answer is a summary of the file, so an agent
that wants a predictions file never has the rows in its context. The file is written atomically and an existing one is refused without --overwrite; a refused prediction
writes nothing."""

import hashlib
import json
import unittest

from harness import Sandbox


class PredictOutput(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        self.sb.write("train.csv", "a,b,y\n" + "".join("%d,%d,%d\n" % (i, i * 3 % 7, 2 * i + i * 3 % 7) for i in range(1, 21)))
        self.sb.write("new.csv", "a,b\n" + "".join("%d,%d\n" % (i, i % 5) for i in range(1, 31)))
        r = self.sb.run("fit", "--model", "linear", "--target", "y", "--output", "m.cml", "train.csv")
        self.assertEqual(r.status, 0, r)

    def tearDown(self):
        self.sb.cleanup()

    def test_the_file_holds_every_row_as_text_prints_them_and_the_answer_describes_it(self):
        text = self.sb.run("predict", "--model-file", "m.cml", "--format", "text", "new.csv")
        self.assertEqual(text.status, 0, text)
        r = self.sb.run("predict", "--model-file", "m.cml", "--output", "p.tsv", "new.csv")
        self.assertEqual(r.status, 0, r)
        d = r.data()
        data = self.sb.read("p.tsv")
        self.assertEqual(data, text.stdout)
        self.assertEqual(d, {"path": "p.tsv", "rows": 30, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "written": True})
        self.assertNotIn("rows", [k for k in d if k == "columns"])

    def test_limit_and_from_choose_the_rows_and_without_them_it_is_all_of_them(self):
        r = self.sb.run("predict", "--model-file", "m.cml", "--output", "p.tsv", "--limit", "5", "--from", "2", "new.csv")
        self.assertEqual(r.status, 0, r)
        self.assertEqual((r.data()["rows"], len(self.sb.read("p.tsv").splitlines())), (5, 5))
        r = self.sb.run("predict", "--model-file", "m.cml", "--output", "q.tsv", "--limit", "1000", "new.csv")
        self.assertEqual(r.data()["rows"], 30)

    def test_an_existing_file_is_refused_unless_overwrite_and_is_left_alone(self):
        self.sb.write("p.tsv", "keep\n")
        r = self.sb.run("predict", "--model-file", "m.cml", "--output", "p.tsv", "new.csv")
        self.assertEqual((r.status, r.first_rule()), (5, "conflict.exists"))
        self.assertNotIn("data", r.doc())
        self.assertEqual(self.sb.read("p.tsv"), b"keep\n")
        r = self.sb.run("predict", "--model-file", "m.cml", "--output", "p.tsv", "--overwrite", "new.csv")
        self.assertEqual(r.status, 0, r)
        self.assertEqual(r.data()["rows"], 30)

    def test_a_missing_directory_is_a_refusal_with_no_rows_in_it(self):
        r = self.sb.run("predict", "--model-file", "m.cml", "--output", "nodir/p.tsv", "new.csv")
        self.assertEqual((r.status, r.first_rule()), (3, "io.not-found"))
        self.assertNotIn("data", r.doc())

    def test_a_refused_prediction_writes_nothing(self):
        self.sb.write("bad.csv", "a\n1\n2\n")
        r = self.sb.run("predict", "--model-file", "m.cml", "--output", "p.tsv", "bad.csv")
        self.assertNotEqual(r.status, 0, r)
        self.assertFalse((self.sb.dir / "p.tsv").exists())

    def test_format_text_says_what_was_written(self):
        r = self.sb.run("predict", "--model-file", "m.cml", "--output", "p.tsv", "--format", "text", "new.csv")
        self.assertEqual((r.status, r.stdout), (0, b"wrote p.tsv\n"))

    def test_without_output_nothing_changes(self):
        r = self.sb.run("predict", "--model-file", "m.cml", "new.csv")
        self.assertEqual(r.status, 0, r)
        self.assertEqual(r.data()["row_count"], 30)
        self.assertFalse((self.sb.dir / "p.tsv").exists())


if __name__ == "__main__":
    unittest.main()

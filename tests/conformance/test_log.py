"""L1 -- no payload is logged, and the log is bounded.

Cells, column names and path components carry canaries; whatever the run (a fit, a predict, an eval, a refusal), the log holds
only counts, flags, rule tags and a hash: every line is a closed object of log.v1, and no canary is in the file."""

import json
import os
import unittest

import jsonschema

from harness import Sandbox, log_schema, spec

CANARIES = ["zzcanaryname", "zzcanarycell", "424242", "zzcanarydir", "zzcanaryfile", "zzcanaryout"]
T = spec.tables()


def good_csv():
    return "id,zzcanaryname,y\n1,424242,10\n2,424243,20\n3,424244,30\n"


class Log(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        (self.sb.dir / "zzcanarydir").mkdir()
        self.sb.write("zzcanarydir/zzcanaryfile.csv", good_csv())

    def tearDown(self):
        self.sb.cleanup()

    def log_lines(self):
        data = self.sb.read("zzcanarydir/lg.ndjson")
        text = data.decode()
        for c in CANARIES:
            self.assertNotIn(c, text, "the log holds %r" % c)
        recs = [json.loads(l) for l in text.splitlines()]
        v = jsonschema.Draft202012Validator(log_schema())
        seq = 0
        for r in recs:
            v.validate(r)
            if r["event"] == "start":
                seq = 0
            self.assertEqual(r["seq"], seq)
            seq += 1
            self.assertEqual(list(r), T["log"][r["event"]], "the fields of %s are the table's, in order" % r["event"])
        return recs

    def go(self, *args, **kw):
        r = self.sb.run(*args, "--log", "zzcanarydir/lg.ndjson", **kw)
        for c in CANARIES:
            self.assertNotIn(c.encode(), r.stderr)
        return r

    def test_fit_predict_eval_log_counts_and_hashes_only(self):
        r = self.go("fit", "--target", "y", "--output", "zzcanarydir/zzcanaryout.cml", "zzcanarydir/zzcanaryfile.csv")
        self.assertEqual(r.status, 0, r)
        r = self.go("predict", "--model-file", "zzcanarydir/zzcanaryout.cml", "zzcanarydir/zzcanaryfile.csv")
        self.assertEqual(r.status, 0, r)
        r = self.go("eval", "--model-file", "zzcanarydir/zzcanaryout.cml", "zzcanarydir/zzcanaryfile.csv")
        self.assertEqual(r.status, 0, r)
        recs = self.log_lines()
        self.assertEqual([x["event"] for x in recs],
                         ["start", "read", "write", "end", "start", "read", "end", "start", "read", "end"])
        self.assertEqual(recs[2]["bytes"], len(self.sb.read("zzcanarydir/zzcanaryout.cml")))
        self.assertTrue(all(x["ok"] for x in recs if x["event"] == "end"))
        # seq restarts with each run
        self.assertEqual([x["seq"] for x in recs], [0, 1, 2, 3, 0, 1, 2, 0, 1, 2])

    def test_planned_models_log_counts_and_hashes_only(self):
        # the plan, the step text, the column names of the steps and the coefficients are in the answer, never in the log
        self.sb.write("zzcanarydir/mis.csv", "id,zzcanaryname,y\n1,424242,10\n2,,20\n3,424244,31\n4,424245,38\n5,424246,52\n")
        r = self.go("fit", "--model", "linear", "--step", "impute column=zzcanaryname strategy=mean", "--target", "y", "--output", "zzcanarydir/zzcanaryout.cml",
                    "zzcanarydir/mis.csv")
        self.assertEqual(r.status, 0, r)
        self.assertIn(b"zzcanaryname", r.stdout)                                  # the answer names the column
        r = self.go("predict", "--model-file", "zzcanarydir/zzcanaryout.cml", "zzcanarydir/mis.csv")
        self.assertEqual(r.status, 0, r)
        r = self.go("eval", "--model-file", "zzcanarydir/zzcanaryout.cml", "--split", "train=0.6", "--seed", "3", "zzcanarydir/mis.csv")
        self.assertEqual(r.status, 0, r)
        r = self.go("fit", "--model", "linear", "--target", "y", "--output", "zzcanarydir/zzcanaryout.cml", "--overwrite", "zzcanarydir/mis.csv")
        self.assertEqual(r.first_rule(), "ml.input.missing-value", r)
        r = self.go("fit", "--model", "linear", "--step", "scale column=zzcanarycell", "--target", "y", "--output", "zzcanarydir/zzcanaryout.cml", "zzcanarydir/mis.csv")
        self.assertEqual(r.first_rule(), "ml.plan.missing-key", r)
        recs = self.log_lines()
        self.assertEqual([x["event"] for x in recs],
                         ["start", "read", "write", "end", "start", "read", "end", "start", "read", "end", "start", "read", "end", "start", "end"])
        self.assertEqual([x["rule"] for x in recs if x["event"] == "end"], [None, None, None, "ml.input.missing-value", "ml.plan.missing-key"])

    def test_refusals_log_the_tag_and_nothing_else(self):
        self.sb.write("zzcanarydir/bad.csv", "id,zzcanaryname,y\n1,zzcanarycell,10\n")
        r = self.go("fit", "--target", "y", "--output", "zzcanarydir/zzcanaryout.cml", "zzcanarydir/bad.csv")
        self.assertEqual(r.first_rule(), "ml.input.non-numeric", r)
        r = self.go("fit", "--target", "zzcanaryname_nope", "--output", "zzcanarydir/zzcanaryout.cml", "zzcanarydir/zzcanaryfile.csv")
        self.assertEqual(r.first_rule(), "ml.input.unknown-column", r)
        r = self.go("fit", "--target", "y", "--output", "zzcanarydir/zzcanaryout.cml", "zzcanarydir/missing-zzcanaryfile.csv")
        self.assertEqual(r.first_rule(), "io.not-found", r)
        recs = self.log_lines()
        ends = [x for x in recs if x["event"] == "end"]
        self.assertEqual([x["rule"] for x in ends], ["ml.input.non-numeric", "ml.input.unknown-column", "io.not-found"])
        self.assertTrue(all(x["ok"] is False for x in ends))
        self.assertEqual([x["exit"] for x in ends], [8, 3, 3])

    def test_a_dry_run_writes_no_write_record(self):
        r = self.go("fit", "--dry-run", "--target", "y", "--output", "zzcanarydir/zzcanaryout.cml", "zzcanarydir/zzcanaryfile.csv")
        self.assertEqual(r.status, 9, r)
        self.assertEqual([x["event"] for x in self.log_lines()], ["start", "read", "end"])
        self.assertEqual(self.log_lines()[-1]["exit"], 9)

    def test_the_log_of_one_run_is_small_and_the_cap_refuses(self):
        sizes = []
        for i in range(3):
            self.go("fit", "--overwrite", "--target", "y", "--output", "zzcanarydir/zzcanaryout.cml", "zzcanarydir/zzcanaryfile.csv")
            sizes.append(len(self.sb.read("zzcanarydir/lg.ndjson")))
        per_run = sizes[1] - sizes[0]
        self.assertLessEqual(per_run, 2048)
        self.assertEqual(sizes[2] - sizes[1], per_run)
        # the cap: a log at least this long is refused before any work is done
        cap = sizes[2]
        r = self.sb.run("fit", "--overwrite", "--log", "zzcanarydir/lg.ndjson", "--log-max-bytes", str(cap), "--target", "y",
                        "--output", "zzcanarydir/never.cml", "zzcanarydir/zzcanaryfile.csv")
        self.assertEqual(r.first_rule(), "limit.log-full", r)
        self.assertFalse((self.sb.dir / "zzcanarydir" / "never.cml").exists())
        self.assertEqual(len(self.sb.read("zzcanarydir/lg.ndjson")), sizes[2], "a refused run appends nothing")
        # one byte below the cap: allowed
        r = self.sb.run("fit", "--overwrite", "--log", "zzcanarydir/lg.ndjson", "--log-max-bytes", str(cap + 1), "--target", "y",
                        "--output", "zzcanarydir/zzcanaryout.cml", "zzcanarydir/zzcanaryfile.csv")
        self.assertEqual(r.status, 0, r)
        self.assertLessEqual(len(self.sb.read("zzcanarydir/lg.ndjson")), cap + 1 + 2048)

    def test_no_log_flag_no_log_file(self):
        r = self.sb.run("fit", "--target", "y", "--output", "zzcanarydir/zzcanaryout.cml", "zzcanarydir/zzcanaryfile.csv")
        self.assertEqual(r.status, 0, r)
        self.assertEqual(sorted(p.name for p in (self.sb.dir / "zzcanarydir").iterdir()), ["zzcanaryfile.csv", "zzcanaryout.cml"])


if __name__ == "__main__":
    unittest.main()

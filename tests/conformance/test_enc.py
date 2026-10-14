"""E8, E9, E10 -- the encoders of categorical columns on the command line (docs/categorical-encoders.md sections 2, 4, 6 and 7, docs/stringtable.md section 6).

The matrices and the payload are judged bit for bit by scripts/gate_enc.py against scripts/oracle_enc.py; these tests judge what only the command line has: the canonical
plan, the resolution of columns and its refusals with their repairs, the file a categorical fit writes, the vocabulary of the training rows only, and that a
numeric plan keeps its text.
"""

import hashlib
import json
import sys
import unittest

from harness import ROOT, Sandbox, sections

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_metrics as OM  # noqa: E402

TXT = (b"id,sex,port,age,y\n1,male,S,20,1\n2,female,C,30,0\n3,male,S,40,1\n4,female,Q,50,0\n5,male,S,31,1\n6,female,C,22,0\n"
       b"7,male,S,35,1\n8,male,C,44,0\n9,female,S,29,0\n10,male,Q,51,1\n")
BASE = ["fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", "sex,port", "--output", "m.cml"]
ENC = "encode column=sex,port kind=onehot"


def fit(sb, *extra, step=ENC, file="t.csv"):
    return sb.run(*(BASE + ["--step", step] + list(extra) + [file]))


class Files(unittest.TestCase):
    """E9: the file of a categorical fit (docs/stringtable.md section 6)"""

    def setUp(self):
        self.sb = Sandbox()
        self.sb.write("t.csv", TXT)

    def tearDown(self):
        self.sb.cleanup()

    def test_a_categorical_fit_writes_the_file_and_says_what_it_holds(self):
        r = fit(self.sb)
        self.assertEqual(r.status, 0, r)
        d = r.data()
        self.assertIs(d["written"], True)
        self.assertEqual(d["path"], "m.cml")
        data = self.sb.read("m.cml")
        self.assertEqual(d["bytes"], len(data))
        self.assertEqual(d["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(int.from_bytes(data[8:12], "little"), 6)
        st = d["summary"]["string_table"]
        self.assertEqual((st["entries"], st["strings"], st["pool_bytes"]), (1, 5, 13))
        table = sections(data)[5]
        raw = data[table[1]:table[1] + table[2]]
        self.assertEqual((st["bytes"], st["sha256"]), (len(raw), hashlib.sha256(raw).hexdigest()))

    def test_an_existing_file_is_kept_without_overwrite_and_replaced_with_it(self):
        self.sb.write("m.cml", b"keep")
        r = fit(self.sb)
        self.assertEqual((r.first_rule(), self.sb.read("m.cml")), ("conflict.exists", b"keep"))
        r = fit(self.sb, "--overwrite")
        self.assertEqual(r.status, 0, r)
        self.assertNotEqual(self.sb.read("m.cml"), b"keep")
        self.sb.write("m.cml", b"keep")
        fit(self.sb, "--dry-run", "--overwrite")
        self.assertEqual(self.sb.read("m.cml"), b"keep")

    def test_a_split_fits_scores_and_writes_the_file(self):
        r = fit(self.sb, "--split", "train=0.7", "--seed", "3")
        self.assertEqual(r.status, 0, r)
        d = r.data()
        self.assertIs(d["written"], True)
        self.assertEqual(d["path"], "m.cml")
        s = d["summary"]
        self.assertEqual(s["split"], {"train": 7, "test": 3, "seed": 3})
        self.assertEqual(s["holdout"]["rows"], 3)
        self.assertEqual(len(s["steps_sha256"]), 64)
        self.assertEqual([(c["column"], c["categories"], c["output_columns"]) for c in s["categorical"]], [("sex", 2, 2), ("port", 3, 3)])
        self.assertTrue((self.sb.dir / "m.cml").exists())

    def test_a_dry_run_fits_reports_the_file_it_would_write_and_writes_nothing(self):
        r = fit(self.sb, "--dry-run")
        self.assertEqual(r.status, 9, r)
        self.assertIs(r.data()["written"], False)
        acts = r.doc()["planned_actions"]
        self.assertEqual([(a["op"], a["path"]) for a in acts], [("create", "m.cml")])
        self.assertEqual(len(acts[0]["sha256"]), 64)
        self.assertFalse((self.sb.dir / "m.cml").exists())
        real = fit(self.sb)
        self.assertEqual(real.data()["sha256"], acts[0]["sha256"])
        self.assertEqual(real.data()["bytes"], acts[0]["bytes"])

    def test_the_vocabulary_is_the_training_rows_only(self):
        """a category only the held-out rows hold is unseen for the step: it is not in the fitted vocabulary and unknown=refuse names its row"""
        n = 10
        seed = 5
        mask = OM.py_train_test(n, 3, seed, 0)
        rows = [r for r in TXT.split(b"\n")[1:] if r]
        test_rows = [i for i in range(n) if mask[i] == 1]
        train_rows = [i for i in range(n) if mask[i] == 0]
        # a port only the first held-out row has
        cells = [r.split(b",") for r in rows]
        cells[test_rows[0]][2] = b"Z"
        self.sb.write("z.csv", b"id,sex,port,age,y\n" + b"\n".join(b",".join(c) for c in cells) + b"\n")
        ok = fit(self.sb, "--split", "train=0.7", "--seed", str(seed), file="z.csv")
        self.assertEqual(ok.status, 0, ok)
        by = {c["column"]: c for c in ok.data()["summary"]["categorical"]}
        want = len({cells[i][2] for i in train_rows})
        self.assertEqual(by["port"]["categories"], want)
        self.assertNotIn(b"Z", {cells[i][2] for i in train_rows})
        refused = fit(self.sb, "--split", "train=0.7", "--seed", str(seed), step="encode column=sex kind=onehot; encode column=port kind=onehot unknown=refuse", file="z.csv")
        self.assertEqual(refused.first_rule(), "ml.input.unknown-category", refused)
        d = refused.error()["detail"]
        self.assertEqual((d["step"], d["column"]), (2, "port"))
        self.assertEqual(d["row"], 0)

    def test_models_without_a_categorical_step_are_unchanged(self):
        self.sb.write("n.csv", b"a,b,y\n1,2,5.1\n2,1,7.9\n3,4,9.3\n4,5,12.2\n5,6,13.8\n6,3,15.1\n7,9,19.7\n8,2,17.1\n")
        r = self.sb.run("fit", "--model", "linear", "--target", "y", "--output", "lin.cml", "n.csv")
        self.assertEqual(r.status, 0)
        self.assertTrue((self.sb.dir / "lin.cml").exists())
        p = self.sb.run("predict", "--model-file", "lin.cml", "n.csv")
        self.assertEqual(p.status, 0)


class Plan(unittest.TestCase):
    """E8: the plan and the resolution of columns"""

    def setUp(self):
        self.sb = Sandbox()
        self.sb.write("t.csv", TXT)

    def tearDown(self):
        self.sb.cleanup()

    def plan_of(self, step, **kw):
        r = fit(self.sb, "--dry-run", step=step, **kw)
        self.assertEqual(r.status, 9, (step, r))
        return r.data()["summary"]["plan"]

    def test_the_canonical_text_writes_min_count_and_missing_only_when_they_are_not_their_defaults(self):
        self.assertEqual(self.plan_of("encode column=sex,port kind=onehot").splitlines()[0], "encode column=port,sex kind=onehot max-categories=64 unknown=ignore")
        for spelled in ("encode kind=onehot column=port,sex min-count=1 missing=pass", "encode  column=sex,port\tkind=onehot min-count=01", "encode missing=pass column=sex,port kind=onehot unknown=ignore max-categories=064"):
            self.assertEqual(self.plan_of(spelled).splitlines()[0], "encode column=port,sex kind=onehot max-categories=64 unknown=ignore", spelled)
        self.assertEqual(self.plan_of("encode column=sex,port kind=label min-count=3 missing=zero").splitlines()[0],
                         "encode column=port,sex kind=label max-categories=64 unknown=ignore min-count=3 missing=zero")
        self.assertEqual(self.plan_of("encode column=sex kind=onehot; freq column=port").splitlines()[1], "freq column=port max-categories=64 unknown=ignore min-count=1 missing=pass")
        self.assertEqual(self.plan_of("encode column=sex kind=onehot; hash column=port buckets=08").splitlines()[1], "hash column=port buckets=8 missing=pass")

    def test_a_numeric_encode_line_is_the_text_it_always_was(self):
        self.sb.write("n.csv", b"a,b,y\n1,2,5.1\n2,1,7.9\n3,2,9.3\n4,1,12.2\n5,2,13.8\n6,1,15.1\n")
        r = self.sb.run("fit", "--model", "linear", "--l2", "1", "--target", "y", "--output", "n.cml", "--step", "encode column=b kind=onehot", "n.csv")
        self.assertEqual(r.status, 0, r)
        self.assertEqual(r.data()["summary"]["plan"].splitlines()[0], "encode column=b kind=onehot max-categories=64 unknown=ignore")

    def test_a_numeric_step_on_a_categorical_column_is_refused_with_an_encode_step_as_the_repair(self):
        r = fit(self.sb, step="scale column=age,sex kind=standard")
        self.assertEqual(r.first_rule(), "ml.plan.column-kind", r)
        e = r.error()
        self.assertEqual(e["detail"], {"step": 1, "column": "sex", "needs": "numeric"})
        self.assertEqual(e["repair"]["kind"], "retry")
        argv = e["repair"]["argv"]
        self.assertEqual(argv[argv.index("--step") + 1], "encode column=sex kind=label; scale column=age,sex kind=standard")

    def test_the_repair_runs(self):
        """the retry the refusal proposes gets past the plan: the label encoding keeps the name, so the scale still finds the column"""
        r = fit(self.sb, step="scale column=sex kind=standard", file="t.csv")
        argv = r.error()["repair"]["argv"][1:]
        k = argv.index("--root")
        again = self.sb.run(*(argv[:k] + argv[k + 2:]))
        # the plan is now fine; the port is read by no step
        self.assertEqual(again.first_rule(), "ml.input.categorical-not-encoded", again)
        self.assertEqual(again.error()["detail"]["columns"], "port")

    def test_freq_and_hash_on_a_numeric_column_and_a_mixed_list(self):
        for step, place in (("freq column=age", 1), ("hash column=age buckets=4", 1), ("freq column=age,sex", 1), ("encode column=age,sex kind=onehot", 2)):
            r = fit(self.sb, step=step)
            self.assertEqual(r.first_rule(), "ml.plan.column-kind", (step, r))
        r = fit(self.sb, step="hash column=age buckets=4")
        self.assertEqual(r.error()["detail"], {"step": 1, "column": "age", "needs": "categorical"})
        self.assertEqual(r.error()["repair"]["kind"], "none")

    def test_an_encoded_column_cannot_be_encoded_again_and_unknown_columns_are_unknown(self):
        r = fit(self.sb, step="encode column=sex,port kind=label; freq column=sex")
        self.assertEqual(r.first_rule(), "ml.plan.column-kind", r)
        r = fit(self.sb, step="encode column=sex,nope kind=onehot")
        self.assertEqual(r.first_rule(), "ml.plan.unknown-column", r)

    def test_a_star_means_the_numeric_columns_for_encode_and_the_categorical_ones_for_freq_and_hash(self):
        ok = fit(self.sb, "--dry-run", step="hash column=* buckets=3")
        self.assertEqual(ok.status, 9, ok)
        self.assertEqual([c["column"] for c in ok.data()["summary"]["categorical"]], ["sex", "port"])
        # encode * is the numeric columns: the categorical ones are then read by no step
        r = fit(self.sb, "--dry-run", step="encode column=* kind=onehot")
        self.assertEqual(r.first_rule(), "ml.input.categorical-not-encoded", r)
        self.assertEqual(r.error()["detail"]["columns"], "sex,port")

    def test_declared_columns_that_no_step_reads_are_named(self):
        r = fit(self.sb, step="encode column=sex kind=onehot")
        self.assertEqual(r.first_rule(), "ml.input.categorical-not-encoded", r)
        d = r.error()["detail"]
        self.assertEqual((d["columns"], d["categories"]), ("port", "3"))
        self.assertEqual(len(d["vocabulary_sha256"]), 64)
        r = fit(self.sb, step="scale column=age kind=minmax")
        self.assertEqual(r.error()["detail"]["columns"], "sex,port")

    def test_rare_bucket_collision_and_bounds(self):
        self.sb.write("o.csv", b"id,sex,port,age,y\n1,__other__,S,20,1\n2,a,C,30,0\n3,a,S,40,1\n4,b,Q,50,0\n5,a,S,31,1\n6,b,C,22,0\n")
        r = fit(self.sb, "--dry-run", step="encode column=sex kind=onehot min-count=2; encode column=port kind=onehot", file="o.csv")
        self.assertEqual(r.first_rule(), "ml.input.reserved-category", r)
        self.assertEqual(r.error()["detail"], {"step": 1, "column": "sex"})
        ok = fit(self.sb, "--dry-run", step="encode column=sex kind=onehot min-count=1; encode column=port kind=onehot", file="o.csv")
        self.assertEqual(ok.status, 9, ok)
        r = fit(self.sb, "--dry-run", step="encode column=sex,port kind=onehot max-categories=2")
        self.assertEqual(r.first_rule(), "ml.limit.categories", r)
        r2 = fit(self.sb, "--dry-run", step="encode column=sex,port kind=onehot max-categories=2 min-count=3")
        self.assertEqual(r2.first_rule(), "ml.limit.categories", r2)
        self.assertEqual(r.error()["detail"], {"step": 1, "column": "port"})
        # min-count brings the effective categories under the bound
        ok = fit(self.sb, "--dry-run", step="encode column=sex,port kind=onehot max-categories=3 min-count=3")
        self.assertEqual(ok.status, 9, ok)

    def test_min_count_and_missing_on_numeric_columns_are_refused(self):
        self.sb.write("n.csv", b"a,b,y\n1,2,5.1\n2,1,7.9\n3,2,9.3\n4,1,12.2\n")
        r = self.sb.run("fit", "--model", "linear", "--l2", "1", "--target", "y", "--output", "n.cml", "--step", "encode column=b kind=onehot min-count=2", "n.csv")
        self.assertEqual(r.first_rule(), "ml.plan.column-kind", r)

    def test_the_new_verbs_and_keys_are_bounded(self):
        for step, rule in (("hash column=sex", "ml.plan.missing-key"), ("hash column=sex buckets=1", "ml.param.out-of-range"), ("hash column=sex buckets=1025", "ml.param.out-of-range"),
                           ("freq column=sex min-count=0", "ml.param.out-of-range"), ("freq column=sex min-count=1000001", "ml.param.out-of-range"),
                           ("freq column=sex missing=maybe", "ml.plan.bad-value"), ("hash column=sex buckets=4 unknown=refuse", "ml.plan.unknown-key"), ("encode column=sex kind=onehot kind=label", "ml.plan.duplicate-key")):
            r = fit(self.sb, step=step)
            self.assertEqual(r.first_rule(), rule, (step, r))


class Rules(unittest.TestCase):
    """the missing and unseen rules, through the whole command line (the matrices are judged bit for bit by gate_enc.py)"""

    def setUp(self):
        self.sb = Sandbox()

    def tearDown(self):
        self.sb.cleanup()

    def test_the_hash_step_stores_no_category_and_counts_no_unknown(self):
        self.sb.write("t.csv", TXT)
        r = fit(self.sb, "--dry-run", step="hash column=sex,port buckets=5")
        self.assertEqual(r.status, 9, r)
        self.assertEqual([(c["kind"], c["categories"], c["output_columns"]) for c in r.data()["summary"]["categorical"]], [("hash", 0, 5), ("hash", 0, 5)])

    def test_missing_cells_pass_through_to_the_estimator_unless_missing_zero(self):
        t = TXT.replace(b"3,male,S,40,1", b"3,male,,40,1")
        self.sb.write("t.csv", t)
        r = fit(self.sb, step="encode column=sex,port kind=onehot", file="t.csv")
        self.assertEqual(r.first_rule(), "ml.input.missing-value", r)
        self.assertEqual(r.error()["detail"]["names"], "port#0,port#1,port#2")
        r = fit(self.sb, "--dry-run", step="encode column=sex,port kind=onehot missing=zero", file="t.csv")
        self.assertEqual(r.status, 9, r)

    def test_na_values_make_a_marker_missing(self):
        t = TXT.replace(b"3,male,S,40,1", b"3,male,NA,40,1")
        self.sb.write("t.csv", t)
        r = fit(self.sb, "--dry-run", "--na-values", "NA", step="encode column=sex,port kind=onehot missing=zero", file="t.csv")
        self.assertEqual(r.status, 9, r)
        by = {c["column"]: c for c in r.data()["summary"]["categorical"]}
        self.assertEqual(by["port"]["categories"], 3)
        r = fit(self.sb, "--dry-run", step="encode column=sex,port kind=onehot", file="t.csv")
        self.assertEqual({c["column"]: c for c in r.data()["summary"]["categorical"]}["port"]["categories"], 4)


if __name__ == "__main__":
    unittest.main()

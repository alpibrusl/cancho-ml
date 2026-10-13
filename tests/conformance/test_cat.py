"""C11 -- a categorical column is declared, never inferred, and stage 1 reads it and then refuses to fit it (docs/categorical.md sections 2, 5 and 8).

The vocabulary itself is judged by scripts/gate_cat.py against the oracle; these tests judge the command line: what is refused with which rule and which details,
that nothing is written, and that the digest the refusal carries is the one the oracle computes from the file.
"""

import hashlib
import sys
import unittest

from harness import ROOT, Sandbox

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_cat  # noqa: E402

TXT = b'id,sex,port,y\n1,male,S,2.5\n2,female,C,3\n3,male,,4\n4,"fe""male",Q,5\n5,NA,S,6\n'
FIT = ["fit", "--target", "y", "--output", "m.cml"]


def fit(sb, *extra, file="t.csv"):
    return sb.run(*(FIT + list(extra) + [file]))


class Declared(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        self.sb.write("t.csv", TXT)

    def tearDown(self):
        self.sb.cleanup()

    def test_an_undeclared_text_column_is_refused_with_the_drop_repair(self):
        r = fit(self.sb)
        self.assertEqual(r.first_rule(), "ml.input.non-numeric")
        err = r.error()
        self.assertEqual(err["repair"]["kind"], "retry")
        self.assertEqual(err["repair"]["argv"][-2:], ["--drop", "sex"])
        self.assertEqual(err["detail"]["name"], "sex")
        # the summary of the rule in introspect names --cat as the other way out
        import harness
        rules = {x["rule"]: x for x in harness.introspect()["rules"]}
        self.assertIn("--cat", rules["ml.input.non-numeric"]["summary"])

    def test_a_declared_column_is_read_and_then_refused_with_its_digest(self):
        r = fit(self.sb, "--cat", "sex,port", "--na-values", "NA")
        self.assertEqual(r.first_rule(), "ml.input.categorical-not-encoded")
        d = r.error()["detail"]
        self.assertEqual(d["columns"], "sex,port")
        self.assertEqual(d["categories"], "3,3")
        cfg = oracle_cat.Cfg(cats=["sex", "port"], na=["NA"], target="y")
        want = hashlib.sha256(oracle_cat.dump(oracle_cat.load(TXT, cfg).vocab)).hexdigest()
        self.assertEqual(d["vocabulary_sha256"], want)
        self.assertEqual(r.error()["repair"], None)
        self.assertFalse((self.sb.dir / "m.cml").exists())

    def test_the_markers_change_the_vocabulary_and_so_the_digest(self):
        a = fit(self.sb, "--cat", "sex,port").error()["detail"]
        b = fit(self.sb, "--cat", "sex,port", "--na-values", "NA").error()["detail"]
        self.assertEqual(a["categories"], "4,3")
        self.assertEqual(b["categories"], "3,3")
        self.assertNotEqual(a["vocabulary_sha256"], b["vocabulary_sha256"])

    def test_the_order_of_the_names_is_not_the_order_of_the_columns(self):
        a = fit(self.sb, "--cat", "sex,port").error()["detail"]
        b = fit(self.sb, "--cat", "port,sex").error()["detail"]
        self.assertEqual(a, b)
        self.assertEqual(a["columns"], "sex,port")

    def test_a_name_twice_is_one_column(self):
        a = fit(self.sb, "--cat", "sex,port,sex").error()["detail"]
        self.assertEqual(a["columns"], "sex,port")

    def test_dry_run_and_overwrite_change_nothing(self):
        self.sb.write("m.cml", b"keep")
        r = fit(self.sb, "--cat", "sex,port", "--dry-run", "--overwrite")
        self.assertEqual(r.first_rule(), "ml.input.categorical-not-encoded")
        self.assertEqual(self.sb.read("m.cml"), b"keep")

    def test_the_target_a_dropped_column_and_a_float_column_cannot_be_declared(self):
        for extra, flags in ((["--cat", "y"], "--cat, --target"), (["--cat", "sex", "--drop", "sex"], "--cat, --drop"), (["--cat", "id", "--float", "id"], "--cat, --float")):
            r = fit(self.sb, *extra)
            self.assertEqual(r.first_rule(), "args.conflict", extra)
            self.assertEqual(r.error()["detail"]["flags"], flags)

    def test_an_unknown_column_is_exit_3(self):
        r = fit(self.sb, "--cat", "sex,nope")
        self.assertEqual((r.first_rule(), r.status), ("ml.input.unknown-column", 3))
        self.assertEqual(r.error()["detail"]["flag"], "--cat")
        self.assertEqual(r.error()["detail"]["name"], "nope")

    def test_the_limits_are_in_their_ranges(self):
        for flag, bad in (("--max-categories", "0"), ("--max-categories", "1025"), ("--max-string-bytes", "0"), ("--max-string-bytes", "65537"),
                          ("--max-pool-bytes", "0"), ("--max-pool-bytes", "8388609"), ("--max-cat-columns", "0"), ("--max-cat-columns", "4097")):
            r = fit(self.sb, "--cat", "sex,port", flag, bad)
            self.assertEqual((r.first_rule(), r.status), ("ml.param.out-of-range", 2), (flag, bad))
        for flag, ok in (("--max-categories", "1024"), ("--max-string-bytes", "65536"), ("--max-pool-bytes", "8388608"), ("--max-cat-columns", "4096")):
            r = fit(self.sb, "--cat", "sex,port", flag, ok)
            self.assertEqual(r.first_rule(), "ml.input.categorical-not-encoded", (flag, ok))

    def test_a_limit_or_markers_without_cat_is_a_conflict_and_a_bad_list_is_a_bad_value(self):
        for flag, v in (("--na-values", "NA"), ("--max-categories", "9"), ("--max-string-bytes", "9"), ("--max-pool-bytes", "9"), ("--max-cat-columns", "9")):
            self.assertEqual(fit(self.sb, flag, v).first_rule(), "args.conflict", flag)
        for bad in ("", "NA,", ",NA", "NA,,?", ",".join("m%d" % i for i in range(65)), "x" * 256):
            self.assertEqual(fit(self.sb, "--cat", "sex", "--na-values", bad).first_rule(), "args.bad-value", bad[:20])
        self.assertEqual(fit(self.sb, "--cat", "").first_rule(), "args.bad-value")
        # 64 markers of 255 bytes are accepted
        ok = ",".join(("m%d" % i).ljust(255, "z") for i in range(64))
        self.assertEqual(fit(self.sb, "--cat", "sex,port", "--na-values", ok).first_rule(), "ml.input.categorical-not-encoded")

    def test_predict_and_eval_take_none_of_the_flags(self):
        self.sb.write("n.csv", b"id,x,y\n1,2,3\n2,3,5\n3,5,9\n")
        self.assertEqual(self.sb.run(*(FIT + ["n.csv"])).status, 0)
        for flag, v in (("--cat", "x"), ("--na-values", "NA"), ("--max-categories", "9")):
            for verb in ("predict", "eval"):
                r = self.sb.run(verb, "--model-file", "m.cml", flag, v, "n.csv")
                self.assertEqual(r.first_rule(), "args.conflict", (verb, flag))

    def test_each_limit_names_the_cell(self):
        r = fit(self.sb, "--cat", "sex,port", "--max-string-bytes", "5")
        self.assertEqual(r.first_rule(), "ml.limit.string-bytes")
        self.assertEqual(r.error()["detail"], {"line": 3, "column": 2, "name": "sex", "length": 6, "max": 5})
        r = fit(self.sb, "--cat", "sex,port", "--max-categories", "2")
        self.assertEqual(r.error()["detail"], {"line": 5, "column": 2, "name": "sex", "max": 2})
        r = fit(self.sb, "--cat", "sex,port", "--max-pool-bytes", "12")
        self.assertEqual(r.first_rule(), "ml.limit.string-pool")
        self.assertEqual(r.error()["detail"]["max"], 12)
        r = fit(self.sb, "--cat", "sex,port", "--max-cat-columns", "1")
        self.assertEqual(r.error()["detail"], {"columns": 2, "max": 1})

    def test_invalid_utf8_names_the_offset_and_has_no_drop_retry(self):
        self.sb.write("u.csv", b"id,sex,y\n1,male,2\n2,caf\xe9,3\n")
        r = fit(self.sb, "--cat", "sex", file="u.csv")
        self.assertEqual(r.first_rule(), "ml.input.invalid-utf8")
        self.assertEqual(r.error()["detail"], {"line": 3, "column": 2, "name": "sex", "offset": 3})
        self.assertEqual(r.error()["repair"], None)
        self.assertIn("re-encode", r.error()["hint"])

    def test_a_crlf_file_with_a_bom_reads_the_same(self):
        crlf = b"\xef\xbb\xbf" + TXT.replace(b"\n", b"\r\n")
        self.sb.write("c.csv", crlf)
        a = fit(self.sb, "--cat", "sex,port", file="c.csv").error()["detail"]
        b = fit(self.sb, "--cat", "sex,port").error()["detail"]
        self.assertEqual(a, b)

    def test_no_thread_count_changes_a_byte(self):
        outs = set()
        for t in ("1", "2", "4", "8", "16"):
            r = self.sb.run(*(["fit", "--model", "linear", "--target", "y", "--output", "m.cml", "--cat", "sex,port", "--threads", t, "t.csv"]))
            outs.add(r.stdout)
        self.assertEqual(len(outs), 1)


if __name__ == "__main__":
    unittest.main()

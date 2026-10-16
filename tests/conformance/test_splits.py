"""SP -- the split strategies and cross-validation (docs/splits.md section 9). The gates A1, B1, C1, D1 and S1 are the oracle's
(scripts/oracle_split.py: every rule written again in Python, the real binary fitted and evaluated on the files the oracle cuts; S1, the
scikit-learn comparison, is in test_splits_sklearn.py because its import needs numpy and scikit-learn, which the fast job of the workflow does not install);
E1 (one fixture per tag, with its repair) runs in test_rules through tests/conformance/fixtures.py; this file adds the checks that
need a file of their own: the hostile data and the arguments of F1, the canonical text, and the flags of the bins.

SPLIT_QUICK=1 runs the oracle gates on their small sets (the mutants of scripts/mutants_split.py do)."""

import itertools
import json
import os
import random
import sys
import tempfile
import unittest

from harness import TRAPS, Sandbox, introspect, run, spec, validate
from fixtures import SPL

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import oracle_split as O  # noqa: E402

QUICK = os.environ.get("SPLIT_QUICK") == "1"


class Oracle(unittest.TestCase):
    def gate(self, fn, **kw):
        O.FAILS.clear()
        with tempfile.TemporaryDirectory() as tmp:
            O.WORK[0] = tmp
            fn(tmp, QUICK, **kw)
        self.assertEqual(O.FAILS, [], "\n".join(O.FAILS[:12]))

    def test_a1_the_assignment_bytes(self):
        self.gate(O.gate_assign)

    def test_b1_invariants(self):
        self.gate(O.gate_invariants)

    def test_c1_end_to_end_no_leak(self):
        self.gate(O.gate_e2e)

    def test_d1_thread_invariance(self):
        self.gate(O.gate_threads)


class Surface(unittest.TestCase):
    def test_flags_are_in_the_table_and_the_verb_is_described(self):
        names = {f["name"] for f in introspect()["flags"]}
        for f in ("stratify", "split-by", "split-group", "folds", "repeats", "emit-folds"):
            self.assertIn("--" + f, names)
        from harness import skill
        self.assertIn("cv FILE", skill())

    def test_predict_and_serve_binaries_have_none_of_the_flags(self):
        import subprocess
        from harness import PREDICT_BIN
        for b in (PREDICT_BIN,):
            if not os.path.exists(b):
                self.skipTest("no ml-predict beside this ml (a mutant copy)")
            doc = json.loads(subprocess.run([b, "introspect"], capture_output=True, check=True).stdout)
            have = {f["name"] for f in doc["flags"]}
            for f in ("stratify", "split-by", "split-group", "folds", "repeats", "emit-folds"):
                self.assertNotIn("--" + f, have, "%s has --%s" % (b, f))

    def test_canonical_text_is_one_line_without_spaces_in_a_token(self):
        sb = Sandbox()
        try:
            sb.write("s.csv", SPL.replace("grp", "my group=1").replace("day", "d%y"))
            r = sb.run("cv", "--model", "tree", "--task", "classification", "--target", "y", "--drop", "my group=1,d%y,r", "--folds", "2", "--split-group", "my group=1", "s.csv")
            self.assertEqual(r.status, 0, r)
            self.assertEqual(r.data()["split"], "cv folds=2 repeats=1 group=my%20group%3D1 seed=0")
            self.assertEqual(r.data()["split"], O.spec_text("cv", "group", 0, "my group=1", K=2, R=1))
        finally:
            sb.cleanup()


class Hostile(unittest.TestCase):
    """F1 -- hostile data: an answer of the contract, never a trap, and the same assignment as the oracle when the tool answers."""

    def check(self, sb, argv, header, rows, strategy, K, expect=None):
        r = sb.run(*argv)
        self.assertEqual(validate(r), [], r)
        self.assertNotIn(r.status, TRAPS, r)
        if expect is not None:
            self.assertEqual(r.first_rule(), expect, r)
        return r

    def run_cv(self, header, rows, strategy, K, extra=(), crlf=False, bom=False, expect=None, seed=0):
        sb = Sandbox()
        try:
            O.write_csv(sb.dir / "h.csv", header, rows, crlf=crlf, bom=bom)
            argv = ["cv", "h.csv", "--target", "y", "--drop", "grp,day,t,cat", "--model", "tree", "--task", "classification", "--folds", str(K), "--seed", str(seed), "--emit-folds"] + O.strategy_flags(strategy) + list(extra)
            r = self.check(sb, argv, header, rows, strategy, K, expect)
            if r.status == 0:
                n, labels, gids, G, tkeys = O.oracle_inputs(header, rows, strategy)
                want = O.cv(strategy, n, K, 1, seed, labels, gids, G, tkeys)
                self.assertEqual(r.data()["assignment"]["folds"], want, strategy)
            return r
        finally:
            sb.cleanup()

    def test_all_one_class(self):
        rnd = random.Random(1)
        header, rows = O.gen_table(rnd, 40, "cls", groups=6)
        for r in rows:
            r[7] = "1"
        for strat in ("plain", "stratify", "group", "time"):
            self.run_cv(header, rows, strat, 3)

    def test_one_row_per_group_and_a_group_per_row(self):
        rnd = random.Random(2)
        header, rows = O.gen_table(rnd, 40, "cls", groups=40)
        for i, r in enumerate(rows):
            r[4] = "u%03d" % i
        r = self.run_cv(header, rows, "group", 5)
        self.assertEqual(r.status, 0, r)
        r = self.run_cv(header, rows, "group", 20)
        # two rows per fold: a fold of one class has no ROC-AUC, the refusal of `ml eval` on that side
        self.assertIn(r.first_rule(), (None, "ml.label.single-class"), r)

    def test_ties_in_the_time_column(self):
        rnd = random.Random(3)
        header, rows = O.gen_table(rnd, 60, "cls", groups=5, tie_dates=True)
        for r in rows:
            r[5] = "2024-01-01"
        r = self.run_cv(header, rows, "time", 4)
        self.assertEqual(r.status, 0, r)

    def test_many_groups(self):
        rnd = random.Random(4)
        header, rows = O.gen_table(rnd, 6000, "cls", groups=5000)
        r = self.run_cv(header, rows, "group", 5, extra=["--max-rows", "10000"])
        self.assertEqual(r.status, 0, r)

    def test_quoted_cells_crlf_and_bom(self):
        rnd = random.Random(5)
        header, rows = O.gen_table(rnd, 50, "cls", groups=6)
        for i, r in enumerate(rows):
            r[4] = ["a,b", 'say "hi"', "line\nbreak", "plain", "é", "z z"][i % 6]
        for strat in ("group", "time", "plain"):
            r = self.run_cv(header, rows, strat, 3, crlf=True, bom=True)
            self.assertEqual(r.status, 0, r)

    def test_missing_cells_in_the_split_columns(self):
        rnd = random.Random(6)
        header, rows = O.gen_table(rnd, 30, "cls", groups=6)
        rows[7][4] = ""
        self.run_cv(header, rows, "group", 3, expect="ml.split.missing-cell")
        rows[7][4] = "g01"
        rows[9][5] = ""
        self.run_cv(header, rows, "time", 3, expect="ml.split.missing-cell")

    def test_time_forms(self):
        rnd = random.Random(7)
        header, rows = O.gen_table(rnd, 24, "cls", groups=4)
        forms = ["2024-01-05", "2024-01-05T10:30", "2024-01-05 10:30:15", "2024-01-05T10:30:15.25", "2024-01-05T10:30:15.5Z", "2024-02-29T23:59:59Z"]
        for i, r in enumerate(rows):
            r[5] = forms[i % len(forms)].replace("01-05", "%02d-%02d" % (1 + i % 3, 1 + i % 28) if 1 + i % 28 < 29 else "01-05")
        self.run_cv(header, rows, "time", 3)
        for bad, reason in (("2023-02-29", "range"), ("2024-13-01", "range"), ("2024-01-01T24:00", "range"), ("2024-01-01T10:00:60", "range"),
                            ("2024-01-01T10:00+01:00", "offset"), ("2024-01-01Z", "form"), ("2024-W01-1", "form"), ("yesterday", "not-a-time")):
            rows2 = [list(r) for r in rows]
            rows2[3][5] = bad
            r = self.run_cv(header, rows2, "time", 3, expect="ml.split.time-format")
            self.assertEqual(r.error()["detail"]["reason"], reason, bad)

    def test_numeric_time_column(self):
        rnd = random.Random(8)
        header, rows = O.gen_table(rnd, 50, "cls", groups=4)
        sb = Sandbox()
        try:
            O.write_csv(sb.dir / "h.csv", header, rows)
            r = sb.run("cv", "h.csv", "--target", "y", "--drop", "grp,day,cat", "--model", "tree", "--task", "classification", "--folds", "3", "--emit-folds", "--split-by", "time:t")
            self.assertEqual(r.status, 0, r)
            keys = O.time_keys([r_[6] for r_ in rows])
            want = O.cv("time", 50, 3, 1, 0, None, None, 0, keys)
            self.assertEqual(r.data()["assignment"]["folds"], want)
        finally:
            sb.cleanup()


class Edges(unittest.TestCase):
    """F1 -- the edges that one input decides: a century that is not a leap year, the fraction of a second, a quote or a CR inside a group cell, the digit 0 and
    the trailing zeros of the canonical text, a class that would be held out whole, and the fold and repeat that a training fold without a class names."""

    def table(self, n=24, groups=4, seed=21):
        return O.gen_table(random.Random(seed), n, "cls", groups=groups)

    def cv(self, header, rows, strategy, K, crlf=False, trim_last_newline=False, expect=None):
        sb = Sandbox()
        try:
            O.write_csv(sb.dir / "h.csv", header, rows, crlf=crlf)
            if trim_last_newline:
                data = (sb.dir / "h.csv").read_bytes()
                (sb.dir / "h.csv").write_bytes(data.rstrip(b"\r\n"))
            col = {"group": ["--split-group", "grp"], "time": ["--split-by", "time:day"], "plain": []}[strategy]
            r = sb.run("cv", "h.csv", "--target", "y", "--drop", "grp,day,t,cat", "--model", "tree", "--task", "classification", "--folds", str(K), "--emit-folds", *col)
            if expect is None:
                self.assertEqual(r.status, 0, r)
                n, labels, gids, G, tkeys = O.oracle_inputs(header, rows, strategy)
                self.assertEqual(r.data()["assignment"]["folds"], O.cv(strategy, n, K, 1, 0, labels, gids, G, tkeys), strategy)
            else:
                self.assertEqual(r.first_rule(), expect, r)
            return r
        finally:
            sb.cleanup()

    def test_century_years_and_leap_days(self):
        header, rows = self.table()
        for i, r in enumerate(rows):
            r[5] = "2024-%02d-%02d" % (1 + i // 28, 1 + i % 28)
            r[7] = str(i % 2)
        for good in ("2000-02-29", "2024-02-29", "1600-02-29"):
            rows2 = [list(r) for r in rows]
            rows2[3][5] = good
            self.cv(header, rows2, "time", 3)
        for bad in ("1900-02-29", "2100-02-29", "2023-02-29"):
            rows2 = [list(r) for r in rows]
            rows2[3][5] = bad
            r = self.cv(header, rows2, "time", 3, expect="ml.split.time-format")
            self.assertEqual(r.error()["detail"]["reason"], "range", bad)

    def test_the_fraction_of_a_second_orders_rows(self):
        header, rows = self.table(n=30, seed=22)
        frac = random.Random(23).sample(range(100, 1000), 30)
        for i, r in enumerate(rows):
            r[5] = "2024-05-06T10:30:15.%d" % frac[i]
        self.cv(header, rows, "time", 3)

    def test_group_cells_with_quotes_and_a_cr_at_the_end_of_the_file(self):
        header, rows = self.table(n=32, groups=4, seed=24)
        names = ['g1', 'g"1', 'g2', 'g2"', 'g3', '"g3', 'g4', 'g4""']
        for i, r in enumerate(rows):
            r[4] = names[i % len(names)]
        self.cv(header, rows, "group", 4)
        # the group is the last cell of every line, the file is CRLF, and the last line has no line ending: its cell must equal the others'
        header2 = [h for h in header if h != "grp"] + ["grp"]
        rows2 = [[c for c, h in zip(r, header) if h != "grp"] + [r[header.index("grp")]] for r in rows]
        for i, r in enumerate(rows2):
            r[-1] = "g%d" % (i % 4)
        self.cv(header2, rows2, "group", 4, crlf=True, trim_last_newline=True)

    def test_the_canonical_text_of_a_fraction_and_of_a_name_with_a_zero(self):
        header, rows = self.table(n=40, groups=5, seed=25)
        sb = Sandbox()
        try:
            O.write_csv(sb.dir / "h.csv", header, rows)
            for text in ("train=0.80", "train=0.500", "train=0.75", "train=0.8000"):
                r = sb.run("fit", "h.csv", "--target", "y", "--drop", "grp,day,t,cat", "--model", "tree", "--task", "classification", "--split", text, "--output", "m-%s.ml" % text[6:])
                self.assertEqual(r.status, 0, r)
                self.assertEqual(r.data()["summary"]["split"]["spec"], O.spec_text("holdout", "plain", 0, train=text), text)
        finally:
            sb.cleanup()
        for name in ("grp0", "g0", "0", "a-0.b_0", "g0\u00e9"):
            h2 = [name if h == "grp" else h for h in header]
            sb = Sandbox()
            try:
                O.write_csv(sb.dir / "h.csv", h2, rows)
                r = sb.run("cv", "h.csv", "--target", "y", "--drop", name + ",day,t,cat", "--model", "tree", "--task", "classification", "--folds", "2", "--split-group", name)
                self.assertEqual(r.status, 0, r)
                self.assertEqual(r.data()["split"], O.spec_text("cv", "group", 0, name, K=2, R=1), name)
            finally:
                sb.cleanup()

    def test_a_class_held_out_whole_and_a_holdout_that_keeps_a_group(self):
        header, rows = self.table(n=12, groups=2, seed=26)
        for i, r in enumerate(rows):
            r[7] = "1" if i < 2 else "0"
        sb = Sandbox()
        try:
            O.write_csv(sb.dir / "h.csv", header, rows)
            r = sb.run("fit", "h.csv", "--target", "y", "--drop", "grp,day,t,cat", "--model", "tree", "--task", "classification", "--output", "m.ml", "--split", "train=0.1", "--stratify")
            self.assertEqual(r.first_rule(), "ml.split.stratum-too-small", r)
        finally:
            sb.cleanup()
        header, rows = self.table(n=10, groups=2, seed=27)
        for i, r in enumerate(rows):
            r[4] = "ga" if i < 5 else "gb"
            r[7] = str(i % 2)
        sb = Sandbox()
        try:
            O.write_csv(sb.dir / "h.csv", header, rows)
            r = sb.run("fit", "h.csv", "--target", "y", "--drop", "grp,day,t,cat", "--model", "tree", "--task", "classification", "--split", "train=0.2", "--split-group", "grp", "--output", "m.ml")
            self.assertEqual(r.status, 0, r)
            self.assertEqual((r.data()["summary"]["split"]["train"], r.data()["summary"]["split"]["test"]), (5, 5), r)
        finally:
            sb.cleanup()

    def test_a_training_fold_without_a_class_names_its_repeat_and_fold(self):
        sb = Sandbox()
        try:
            sb.write("spl.csv", "grp,day,a,b,y,r\n" + "".join("g%d,2024-01-%02d,%d,%d,%d,1.5\n" % (i // 3, i + 1, i, i % 4, 1 if i < 3 else 0) for i in range(12)))
            r = sb.run("cv", "--model", "tree", "--task", "classification", "--target", "y", "--drop", "grp,day,r", "--folds", "2", "--split-group", "grp", "spl.csv")
            self.assertEqual(r.first_rule(), "ml.label.single-class", r)
            self.assertEqual(sorted(k for k in r.error()["detail"] if k in ("fold", "repeat")), ["fold", "repeat"], r)
        finally:
            sb.cleanup()


class Arguments(unittest.TestCase):
    """F1 -- a seeded fuzz of argument vectors over the new flags and verb: every answer is the contract's, no trap."""

    def test_fuzz(self):
        sb = Sandbox()
        try:
            sb.write("spl.csv", SPL)
            sb.write("empty.csv", "grp,day,a,b,y,r\n")
            vocab = ["--folds", "--repeats", "--stratify", "--split-by", "--split-group", "--emit-folds", "--split", "--seed", "--threads", "--model", "--task", "--target", "--drop", "--output",
                     "--step", "--max-rows", "--positive", "--overwrite", "--dry-run"]
            values = ["1", "2", "3", "5", "20", "21", "0", "-1", "99999999999999999999", "", "time:day", "time:", "time", "time:grp", "time:nosuch", "date:day", "grp", "y", "day", "nosuch",
                      "train=0.8", "train=2", "tree", "linear", "logistic", "kmeans", "constant", "classification", "regression", "r", "grp,day,r", "grp,day,y", "text", "json", "m.cml", "..", "a,b"]
            rnd = random.Random(int(os.environ.get("FUZZ_SEED", "11")))
            n = int(os.environ.get("FUZZ_N", "120" if not QUICK else "40"))
            for i in range(n):
                argv = [rnd.choice(["cv", "cv", "cv", "fit", "eval"]), rnd.choice(["spl.csv", "spl.csv", "empty.csv", "nosuch.csv"])]
                if argv[0] == "eval":
                    argv += ["--model-file", "nosuch.cml"]
                for _ in range(rnd.randrange(1, 9)):
                    f = rnd.choice(vocab)
                    argv.append(f)
                    if f not in ("--stratify", "--emit-folds", "--overwrite", "--dry-run"):
                        argv.append(rnd.choice(values))
                r = sb.run(*argv)
                self.assertNotIn(r.status, TRAPS, argv)
                self.assertEqual(validate(r), [], (argv, r))
        finally:
            sb.cleanup()


if __name__ == "__main__":
    unittest.main()

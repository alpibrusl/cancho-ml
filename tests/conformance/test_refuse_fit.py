"""R48 -- collinear columns and separable data are refused by name (docs/refusals-fit.md, issue 48).

The fixtures of `_refuse_cases()` (the tag, the detail, every argv of the repair run in the same sandbox and gives a model), the controls that must keep fitting, the
exact-fraction oracle for the rank decision and the dependency, the exact separability oracle with an independent Newton iteration, and the fuzz with thread invariance.
`scripts/mutants_refusals.py` runs this module against each mutant of the checks.
"""

import json
import os
import pathlib
import subprocess
import sys
import unittest

from fixtures import CASES, RBASE
from harness import BIN, ROOT, Sandbox, run
from runner import apply_setup, judge, run_case

IDS = [c for c in CASES if c["id"].startswith(("ml-fit-collinear", "ml-fit-separable", "ml-fit-nearly", "ml-fit-overlapping", "ml-fit-near-collinear", "ml-fit-full-one-hot",
                                               "ml-linalg-not-positive-definite"))]


def oracle(*args):
    env = dict(os.environ, ML_BIN=BIN)
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "oracle_refuse.py"), *args, "--ml", BIN], capture_output=True, text=True, env=env)


class RefuseFit(unittest.TestCase):
    def test_there_are_fixtures_for_both_rules_and_the_controls(self):
        self.assertGreaterEqual(len(IDS), 20)
        rules = {c["expect"]["rule"] for c in IDS}
        self.assertTrue({"ml.fit.collinear", "ml.fit.separable", "ml.linalg.not-positive-definite", None} <= rules)

    def test_every_fixture(self):
        failures = []
        for c in IDS:
            sb, r = run_case(c)
            try:
                p = judge(c, sb, r)
                if p:
                    failures.append((c["id"], p))
            finally:
                sb.cleanup()
        self.assertEqual(failures, [], "\n".join("%s: %s" % f for f in failures))

    def test_every_argv_of_every_repair_runs_and_gives_a_model(self):
        n = 0
        for c in IDS:
            if not c["expect"].get("repair_runs"):
                continue
            sb, r = run_case(c)
            try:
                rep = r.doc()["error"]["repair"]
                argvs = [rep["argv"]] if rep["kind"] == "retry" else [o["argv"] for o in rep["options"]]
                for argv in argvs:
                    for f in sb.dir.glob("*.cml"):
                        f.unlink()
                    again = run(*argv[1:], cwd=sb.dir)
                    self.assertEqual(again.status, 0, "%s: %r" % (c["id"], again))
                    self.assertTrue((sb.dir / "m.cml").exists(), c["id"])
                    n += 1
            finally:
                sb.cleanup()
        self.assertGreaterEqual(n, 12)

    def test_a_repair_that_widens_nothing(self):
        for c in IDS:
            if not c["expect"].get("repair_runs"):
                continue
            sb, r = run_case(c)
            try:
                rep = r.doc()["error"]["repair"]
                argvs = [rep["argv"]] if rep["kind"] == "retry" else [o["argv"] for o in rep["options"]]
                for argv in argvs:
                    added = [a for a in argv[1:] if a not in r.argv[1:]]
                    self.assertFalse({"--root", "--overwrite", "--output", "--log"} & set(added), c["id"])
            finally:
                sb.cleanup()

    def test_the_answer_does_not_depend_on_the_thread_count(self):
        for c in IDS:
            if c["expect"]["rule"] not in ("ml.fit.collinear", "ml.fit.separable"):
                continue
            outs = set()
            for t in ("1", "2", "4", "8", "16"):
                c2 = dict(c, argv=list(c["argv"]) + ["--threads", t])
                sb, r = run_case(c2)
                try:
                    d = r.doc()
                    for e in [d["error"]] + d["errors"]:
                        e.pop("repair", None)
                    outs.add(json.dumps(d, sort_keys=True))
                finally:
                    sb.cleanup()
            self.assertEqual(len(outs), 1, c["id"])

    def _repairs(self, r):
        rep = r.doc()["error"]["repair"]
        return [rep["argv"]] if rep["kind"] == "retry" else [o["argv"] for o in rep["options"]]

    def test_cv_refuses_the_fold_that_has_no_answer_and_names_it_and_the_repair_runs(self):
        """a training fold can be separable or collinear when the whole data is not: the refusal of `ml cv` names the repeat and the fold, and its repairs give a cross-validation"""
        for cid, fold in (("ml-cv-fold-separable-names-the-fold", 1), ("ml-cv-fold-collinear-names-the-fold", 0)):
            c = next(c for c in CASES if c["id"] == cid)
            sb, r = run_case(c)
            try:
                self.assertEqual(r.status, 8, r)
                det = r.doc()["error"]["detail"]
                self.assertEqual((det["repeat"], det["fold"]), (0, fold), cid)
                for argv in self._repairs(r):
                    again = run(*argv[1:], cwd=sb.dir)
                    self.assertEqual(again.status, 0, "%s: %r" % (cid, again))
                    self.assertTrue(again.doc()["data"]["split"].startswith("cv folds="), cid)
            finally:
                sb.cleanup()

    def test_the_whole_data_of_the_cv_fixtures_is_not_refused(self):
        for cid, extra in (("ml-cv-fold-separable-names-the-fold", ["--model", "logistic", "--target", "y", "cvsep.csv"]),
                           ("ml-cv-fold-collinear-names-the-fold", ["--model", "linear", "--target", "y", "--drop", "day", "cvcol.csv"])):
            c = next(c for c in CASES if c["id"] == cid)
            sb = Sandbox()
            try:
                apply_setup(sb, c["setup"])
                r = sb.run("fit", *extra, "--output", "w.cml")
                self.assertEqual(r.status, 0, "%s: %r" % (cid, r))
            finally:
                sb.cleanup()

    def test_a_refused_fit_writes_no_run_record_and_a_record_made_before_still_verifies(self):
        for tag, argv, data in (("ml.fit.separable", ["--model", "logistic", "--target", "y", "sep.csv"], None),
                                ("ml.fit.collinear", ["--model", "linear", "--target", "y", "dup.csv"], None)):
            sb = Sandbox()
            try:
                apply_setup(sb, RBASE + [("mkdir", "runs")])
                # a run that is made first (a linear fit of an ordinary file): its record is the one that must stay as it is
                sb.write("ok.csv", "a,y\n1,2.1\n2,3.9\n3,6.2\n4,7.8\n5,10.1\n")
                good = sb.run("fit", "--model", "linear", "--target", "y", "--output", "g.cml", "--track", "runs", "ok.csv")
                self.assertEqual(good.status, 0, good)
                before = sorted((p.name, p.read_bytes()) for p in (sb.dir / "runs").iterdir())
                self.assertEqual(len(before), 1)
                r = sb.run("fit", *argv, "--output", "m.cml", "--track", "runs")
                self.assertEqual(r.status, 8, r)
                self.assertEqual(r.doc()["error"]["rule"], tag)
                self.assertFalse((sb.dir / "m.cml").exists())
                after = sorted((p.name, p.read_bytes()) for p in (sb.dir / "runs").iterdir())
                self.assertEqual(after, before, "a refused fit wrote or changed a run record")
                # the repair of the refusal, with --track kept, writes the second record
                for av in self._repairs(r):
                    for f in sb.dir.glob("m.cml"):
                        f.unlink()
                    fixed = run(*av[1:], cwd=sb.dir)
                    self.assertEqual(fixed.status, 0, fixed)
                    self.assertEqual(len([p for p in (sb.dir / "runs").iterdir() if len(p.name) == 69]), 2)
                    for p in (sb.dir / "runs").iterdir():
                        if len(p.name) == 69 and (p.name, p.read_bytes()) not in before:
                            p.unlink()
                rid = before[0][0][:64]
                v = sb.run("runs", "verify", "runs", rid)
                self.assertEqual(v.status, 0, v)
                self.assertTrue(v.doc()["data"]["verified"])
            finally:
                sb.cleanup()

    def test_the_l2_repair_gives_a_fit_under_split_stratify_and_cat(self):
        cases = [
            ("ml.fit.separable", ["--model", "logistic", "--target", "y", "--split", "train=0.67", "--stratify", "sep.csv"]),
            ("ml.fit.separable", ["--model", "logistic", "--target", "y", "--split", "train=0.7", "sep20.csv"]),
            ("ml.fit.collinear", ["--model", "linear", "--cat", "sp", "--step", "encode column=sp kind=onehot", "--drop", "id", "--target", "y", "--split", "train=0.75", "sp.csv"]),
            ("ml.fit.collinear", ["--model", "logistic", "--cat", "sp", "--step", "encode column=sp kind=onehot", "--drop", "id", "--target", "y", "--split", "train=0.75", "--stratify", "sp.csv"]),
            ("ml.fit.collinear", ["--model", "linear", "--cat", "sp", "--step", "encode column=sp kind=onehot", "--drop", "id", "--target", "y", "--split-by", "time:id", "--split", "train=0.75", "sp.csv"]),
        ]
        for tag, argv in cases:
            sb = Sandbox()
            try:
                apply_setup(sb, RBASE)
                sb.write("sep20.csv", "x1,x2,y\n" + "".join("%s,%d,%d\n" % (i - 9.5, (i * 7) % 5 - 2, int(i >= 10)) for i in range(20)))
                r = sb.run("fit", *argv, "--output", "m.cml")
                self.assertEqual(r.status, 8, "%s: %r" % (argv, r))
                self.assertEqual(r.doc()["error"]["rule"], tag, argv)
                for av in self._repairs(r):
                    fixed = run(*av[1:], cwd=sb.dir)
                    self.assertEqual(fixed.status, 0, "%s: %r" % (av, fixed))
                    self.assertTrue((sb.dir / "m.cml").exists())
                    self.assertIn("split", fixed.doc()["data"]["summary"], av)
            finally:
                sb.cleanup()

    def test_oracle_of_the_rank_decision(self):
        r = oracle("rank", "--n", "200")
        self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-500:])

    def test_fuzz_no_trap_and_thread_invariance(self):
        r = oracle("fuzz", "--n", "24")
        self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-500:])


if __name__ == "__main__":
    unittest.main()

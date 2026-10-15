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

from fixtures import CASES
from harness import BIN, ROOT, run
from runner import judge, run_case

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

    def test_oracle_of_the_rank_decision(self):
        r = oracle("rank", "--n", "200")
        self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-500:])

    def test_oracle_of_separation(self):
        r = oracle("sep", "--n", "90")
        self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-500:])

    def test_fuzz_no_trap_and_thread_invariance(self):
        r = oracle("fuzz", "--n", "24")
        self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-500:])


if __name__ == "__main__":
    unittest.main()

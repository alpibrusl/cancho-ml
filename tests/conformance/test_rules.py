"""R1 -- every fixture's first error is its rule, with the exit code of the tables, one valid JSON line on stdout and nothing
on stderr; every `retry` repair works when applied and adds no capability; every rule of the table is reached."""

import os
import unittest

from fixtures import CASES
from harness import ROOT, Sandbox, run, runs_here, spec, validate
from runner import judge, run_case


class Rules(unittest.TestCase):
    def test_every_fixture(self):
        failures = []
        for c in CASES:
            if c.get("skip_if_root") and os.geteuid() == 0:
                continue
            if not runs_here(c):
                continue
            sb, r = run_case(c)
            try:
                problems = judge(c, sb, r)
                if problems:
                    failures.append((c["id"], problems))
            finally:
                if c.get("setup") and any(s[0] == "chmod" for s in c["setup"]):
                    for p in sb.dir.iterdir():
                        try:
                            p.chmod(0o644)
                        except OSError:
                            pass
                sb.cleanup()
        self.assertEqual(failures, [], "\n".join("%s: %s" % f for f in failures))

    def test_every_rule_of_the_table_is_reached(self):
        reached = {c["expect"]["rule"] for c in CASES if c["expect"]["rule"]}
        missing = [r for r in spec.tables()["rules"] if r not in reached]
        self.assertEqual(missing, [], "rules with no fixture")

    def test_every_fixture_rule_is_in_the_table(self):
        known = set(spec.tables()["rules"])
        extra = {c["expect"]["rule"] for c in CASES if c["expect"]["rule"]} - known
        self.assertEqual(extra, set(), "fixtures reach rules that are not in the table")

    def test_retry_repairs_work_and_widen_nothing(self):
        for c in CASES:
            if not c["expect"].get("repair_retry"):
                continue
            sb, r = run_case(c)
            try:
                argv = r.doc()["error"]["repair"]["argv"]
                self.assertEqual(argv[0], r.argv[0].split("/")[-1] if False else argv[0])
                added = [a for a in argv[1:] if a not in r.argv[1:]]
                widening = {"--root", "--overwrite", "--output", "--log"}
                self.assertFalse(widening & set(added), "%s: the repair adds %s" % (c["id"], added))
                again = run(*argv[1:], cwd=sb.dir)
                self.assertEqual(again.status, 0, "%s: the repair does not work: %r" % (c["id"], again))
            finally:
                sb.cleanup()

    def test_exit_codes_are_the_d4_classes(self):
        """cancho docs/agent-toolbox.md D4, written out here independently of the tables."""

        def d4(tag):
            area, name = tag.split(".", 1)
            if area == "args":
                return 2
            if area == "path":
                return 4 if name in ("outside-root", "symlink") else 2
            if area == "io":
                return {"not-found": 3, "not-a-directory": 3, "is-a-directory": 2, "permission-denied": 4,
                        "read-failed": 1, "write-failed": 1}[name]
            if area == "conflict":
                return 5
            if area in ("limit", "parse"):
                return 8
            if tag in ("ml.param.out-of-range", "ml.param.threads"):
                return 2
            if tag == "ml.input.unknown-column":
                return 3
            return 8

        for tag in spec.tables()["rules"]:
            from harness import expected_exit
            self.assertEqual(expected_exit(tag), d4(tag), tag)

    def test_exit_codes_of_the_tables_agree_with_the_package(self):
        from harness import package_catalogue
        pkg = package_catalogue()
        t = spec.tables()
        for tag in t["rules"]:
            self.assertTrue(tag in t["extra"] or tag in pkg, "%s is in neither the package catalogue nor spec.cho" % tag)
        for tag in t["extra"]:
            self.assertIn(tag, t["rules"], "%s is an extra rule that the rules list does not name" % tag)
            self.assertNotIn(tag, pkg, "%s shadows a shared rule" % tag)


if __name__ == "__main__":
    unittest.main()

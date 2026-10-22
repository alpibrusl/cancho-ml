"""T1 -- one table drives the parser, `introspect` and the `skill` text, and the schemas and the fixtures agree with it.

The tables are read from tools/ml/spec.cho (scripts/spec.py). For every flag, every rule and every log record the test asks:
is it in introspect, in the text of skill, in the schema, and in a fixture? A thing in one place and not in the others fails.
"""

import json
import os
import re
import unittest

import jsonschema

from harness import ROOT, Sandbox, introspect, log_schema, run, schema, skill, spec
import schemas  # scripts/schemas.py
from fixtures import CASES, log_cases
from runner import run_case

T = spec.tables()
FLAGS = ["--" + f["name"] for f in T["flags"]]


def argv_flags(argv):
    """The long flags a command line uses, whether spelled `--f v` or `--f=v`."""
    return {a.split("=", 1)[0] for a in argv if isinstance(a, str) and a.startswith("--") and len(a) > 2}


class Tables(unittest.TestCase):
    def test_flags_in_introspect(self):
        got = [f["name"] for f in introspect()["flags"]]
        self.assertEqual(sorted(got), sorted(FLAGS))

    def test_flag_kinds_roles_defaults_in_introspect(self):
        by = {f["name"]: f for f in introspect()["flags"]}
        for f in T["flags"]:
            g = by["--" + f["name"]]
            self.assertEqual(g["kind"], f["kind"], f["name"])
            self.assertEqual(g["role"], f["role"], f["name"])
            self.assertEqual(g["default"], f["default"] or None, f["name"])

    def test_flags_in_skill(self):
        text = skill()
        missing = [f for f in FLAGS if "`%s" % f not in text]
        self.assertEqual(missing, [], "flags missing from the skill text")

    def test_flags_in_schema(self):
        self.assertEqual(sorted(schema()["$defs"]["flag"]["enum"]), sorted(FLAGS))
        self.assertEqual(sorted(introspect()["schemas"]["ml.v1"]["$defs"]["flag"]["enum"]), sorted(FLAGS))

    def test_flags_in_a_fixture(self):
        used = set()
        for c in CASES:
            used |= argv_flags(c["argv"])
        self.assertEqual([f for f in FLAGS if f not in used], [], "flags no fixture uses")
        self.assertEqual(sorted(used - set(FLAGS) - {"--bogus"}), [], "fixtures use flags that are not in the table")

    def test_every_flag_is_taken_by_some_verb_and_every_listed_flag_exists(self):
        names = {f["name"] for f in T["flags"]}
        union = set()
        for verb, listed in T["allowed"].items():
            self.assertEqual(sorted(set(listed) - names), [], "%s lists flags that are not in the table" % verb)
            union |= set(listed)
        self.assertEqual(sorted(names - union), [], "flags no verb takes")

    def test_rules_in_introspect(self):
        got = [r["rule"] for r in introspect()["rules"]]
        self.assertEqual(sorted(got), sorted(T["rules"]))

    def test_rules_in_skill(self):
        text = skill()
        self.assertEqual([r for r in T["rules"] if "`%s`" % r not in text], [])

    def test_rules_in_schema(self):
        self.assertEqual(sorted(schema()["$defs"]["rule"]["enum"]), sorted(T["rules"]))

    def test_rules_in_a_fixture(self):
        reached = {c["expect"]["rule"] for c in CASES if c["expect"]["rule"]}
        self.assertEqual([r for r in T["rules"] if r not in reached], [])

    def test_limits_in_introspect(self):
        got = {l["name"]: (l["default"], l["ceiling"]) for l in introspect()["limits"]}
        self.assertEqual(got, T["limits"])

    def test_limit_defaults_are_the_flag_defaults(self):
        by = {f["name"]: f for f in T["flags"]}
        for name, (default, ceiling) in T["limits"].items():
            self.assertIn(name, by, "limit %s is not a flag" % name)
            # a flag without one default of its own (k is required; iterations has one for each model it serves) says so with an empty default, and the limit then
            # names the least value the flag accepts or the smallest default among its models
            if by[name]["default"] != "":
                self.assertEqual(int(by[name]["default"]), default, name)
            self.assertLessEqual(default, ceiling)

    def test_schemas_are_the_regeneration_of_the_tables(self):
        for name, sch in schemas.all_schemas().items():
            self.assertEqual((ROOT / "schemas" / name).read_text(), schemas.render(sch), "%s is stale: run scripts/schemas.py" % name)

    def test_embedded_schema_is_the_committed_one(self):
        self.assertEqual(introspect()["schemas"]["ml.v1"], schema())

    def test_log_records_in_the_log_schema(self):
        events = []
        for variant in log_schema()["oneOf"]:
            events.append(variant["properties"]["event"]["const"])
            self.assertEqual(list(variant["properties"]), T["log"][events[-1]])
            self.assertEqual(variant["additionalProperties"], False)
        self.assertEqual(sorted(events), sorted(T["log"]))

    def test_log_records_in_a_fixture(self):
        seen = set()
        for c in log_cases():
            sb, r = run_case(c)
            try:
                for line in sb.read("lg.ndjson").decode().splitlines():
                    rec = json.loads(line)
                    jsonschema.Draft202012Validator(log_schema()).validate(rec)
                    seen.add(rec["event"])
            finally:
                sb.cleanup()
        self.assertEqual(sorted(seen), sorted(T["log"]), "log records no fixture reaches")

    def test_every_case_is_unique(self):
        ids = [c["id"] for c in CASES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_the_models_in_the_flag_table_are_the_built_ones(self):
        # honesty (H1): the list of models is one choice flag, and this is the list of what is built
        by = {f["name"]: f for f in T["flags"]}
        import gen_models
        built = [m["name"] for m in gen_models.load_manifests(ROOT) if m["wired"]]
        self.assertEqual(by["model"]["kind"], "choice:" + "/".join(built))
        self.assertEqual(built, ["constant", "linear", "logistic", "kmeans", "tree", "forest", "boost", "softmax", "recommend"])


if __name__ == "__main__":
    unittest.main()

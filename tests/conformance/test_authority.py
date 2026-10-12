"""A1 -- the authority manifest is derived by the compiler, committed, embedded in the binary and gated against a ceiling.

`scripts/manifest.py --check` fails when the committed manifest, the generated module or the binary's `introspect` differ from a
fresh `cancho authority`, or when a derived label is outside the ceiling of tools.toml. The ceiling here is exact (no slack):
what the tool is allowed is what it does."""

import json
import subprocess
import sys
import tomllib
import unittest

from harness import ROOT, introspect

EXPECTED = {"args", "dir_read", "dir_write", "err_write", "file_read", "file_write", 'fs_read("")', "heap", "io_write"}
FORBIDDEN = {"ffi", "net_out", "net_in", "clock", "fs_write", "io_read", "conc", "exec", "env"}


def label_text(l):
    return l["name"] if l["argument"] is None else '%s("%s")' % (l["name"], l["argument"])


class Authority(unittest.TestCase):
    def test_manifest_check_passes(self):
        p = subprocess.run([sys.executable, str(ROOT / "scripts" / "manifest.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_the_binary_prints_the_committed_authority(self):
        committed = json.loads((ROOT / "manifests" / "ml.authority.json").read_text())
        self.assertEqual(introspect()["authority"], committed)

    def test_the_authority_is_bounded_and_exactly_these_labels(self):
        a = introspect()["authority"]
        self.assertTrue(a["bounded"])
        self.assertEqual(a["unbounded_by"], [])
        self.assertEqual(a["foreign_symbols"], [])
        self.assertEqual({label_text(l) for l in a["labels"]}, EXPECTED)
        self.assertEqual({label_text(l) for l in a["labels"]} & FORBIDDEN, set())

    def test_the_ceiling_is_exact(self):
        ceilings = tomllib.loads((ROOT / "tools.toml").read_text())
        self.assertEqual(set(ceilings["ml"]["allow"]), EXPECTED)
        self.assertEqual({label_text(l) for l in introspect()["authority"]["labels"]}, set(ceilings["ml"]["allow"]))

    def test_no_path_write_no_clock_no_environment_in_the_introspect_claims(self):
        d = introspect()
        self.assertFalse(d["reads_environment"])
        self.assertFalse(d["reads_clock"])
        self.assertTrue(d["integers_only"])

    def test_the_binary_is_the_tool_in_the_manifest(self):
        # one binary per tool; the ceiling file knows only `ml` as an agent tool
        ceilings = tomllib.loads((ROOT / "tools.toml").read_text())
        self.assertEqual([k for k in ceilings if isinstance(ceilings[k], dict)], ["ml"])


if __name__ == "__main__":
    unittest.main()

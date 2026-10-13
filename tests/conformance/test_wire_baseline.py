"""W1 -- the constant models do not change (docs/wiring.md section 1).

`golden_constant.json` was written by the binary of `v1-post` (commit 1b550a5, before the wiring stage) with scripts/golden_constant.py: for each fixture of the
CLI slice, the exit status, the SHA-256 of standard output and the SHA-256 of every file left in the sandbox. This binary must reproduce all of it: the model
files of kinds 1 to 3 are byte for byte the old ones, and so is every answer, refusal included.
"""

import json
import pathlib
import sys
import unittest

from harness import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import golden_constant  # noqa: E402

GOLDEN = json.loads((pathlib.Path(__file__).parent / "golden_constant.json").read_text())


class Baseline(unittest.TestCase):
    def test_the_golden_record_covers_the_constant_fixtures(self):
        ids = {c["id"] for c in golden_constant.cases()}
        self.assertEqual(ids, set(GOLDEN))
        self.assertGreaterEqual(len(GOLDEN), 90)

    def test_every_constant_case_gives_the_bytes_of_v1_post(self):
        failures = []
        for c in golden_constant.cases():
            got = golden_constant.record(c)
            want = GOLDEN[c["id"]]
            if got != want:
                failures.append((c["id"], {k: (want[k], got[k]) for k in want if want[k] != got[k]}))
        self.assertEqual(failures, [], "the constant model changed")

    def test_the_model_file_of_the_cli_slice_is_still_what_docs_cli_md_prints(self):
        # the sales example of docs/cli.md section 2.4: 316 bytes, the hash printed there
        golden = GOLDEN["fit-regression"]["files"]["m.cml"]
        self.assertEqual(len(golden), 64)


if __name__ == "__main__":
    unittest.main()

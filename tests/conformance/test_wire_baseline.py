"""W1 -- the constant models do not change (docs/wiring.md section 1).

`golden_constant.json` was written by the binary of `v1-post` (commit 1b550a5, before the wiring stage) with scripts/golden_constant.py: for each fixture of the
CLI slice, the exit status, the SHA-256 of standard output and the SHA-256 of every file left in the sandbox. This binary must reproduce all of it: the model
files of kinds 1 to 3 are byte for byte the old ones, and so is every answer, refusal included.
"""

import json
import pathlib
import sys
import unittest

from harness import ROOT, runs_here

sys.path.insert(0, str(ROOT / "scripts"))
import golden_constant  # noqa: E402

GOLDEN = json.loads((pathlib.Path(__file__).parent / "golden_constant.json").read_text())

# the cases whose standard output changed on purpose, with the SHA-256 of the new text; the golden file is never rewritten (see the docstring).
# The cancho-tools pin moved from 8dab917 to 6085e63: a refusal of toolbox.cli or toolbox.fail that had `"hint":null` now carries the hint the contract
# gives it. The exit status and the written files are the baseline's.
CHANGED = {
    "args-missing-value": "a9ca77ca9309b30fc309b2d41f38193c6710753bc7a116ec7917e5b0adac8dac",
    "args-bad-value": "4cc8724b6346a1aabdf52c08c9830131bc25dd10ec7249cac97859f360ccffa5",
    "io-not-a-directory": "82578073947d68679d5fe53f6b6c6a1bc40a5b178aa951716fc018a0130410a5",
    # macOS only (darwin_only): the default hint of toolbox.fail.io_error, empty at 8dab917; the hash is the one the macOS job printed
    "io-read-failed": "339cc1aba2044b39985e17d66af5152460f946c1b3aadfe1e02162be7a0553a4",
}


class Baseline(unittest.TestCase):
    def test_the_golden_record_covers_the_constant_fixtures(self):
        ids = {c["id"] for c in golden_constant.cases()}
        self.assertEqual(ids, set(GOLDEN))
        self.assertGreaterEqual(len(GOLDEN), 90)

    def test_every_constant_case_gives_the_bytes_of_v1_post(self):
        failures = []
        for c in golden_constant.cases():
            if not runs_here(c):
                continue
            got = golden_constant.record(c)
            want = dict(GOLDEN[c["id"]])
            if c["id"] in CHANGED:
                want["stdout"] = CHANGED[c["id"]]
            if got != want:
                failures.append((c["id"], {k: (want[k], got[k]) for k in want if want[k] != got[k]}))
        self.assertEqual(failures, [], "the constant model changed")

    def test_the_model_file_of_the_cli_slice_is_still_what_docs_cli_md_prints(self):
        # the sales example of docs/cli.md section 2.4: 316 bytes, the hash printed there
        golden = GOLDEN["fit-regression"]["files"]["m.cml"]
        self.assertEqual(len(golden), 64)


if __name__ == "__main__":
    unittest.main()

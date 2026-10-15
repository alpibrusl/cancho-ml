"""C9 -- numeric inputs do not change (docs/categorical.md section 1).

`golden_numeric.json` was written by the binary of `v1-next` before stage 1 of issue 21 (commit 7d6183d, scripts/golden_numeric.py): for every fixture that
existed then, the exit status, the SHA-256 of standard output and the SHA-256 of every file left in the sandbox. The binary under test reproduces all of it: the
model files of the constant, linear and logistic models with their steps, and every refusal of the reader, byte for byte. The record is never rewritten to
make a test pass. A fixture added since (the categorical ones) is not in it.

THREE exceptions, pinned here and not in the record. Two of them are the texts that name the wired models (docs/wiring-kmeans.md section 2): the refusal of a flag that needs a
model other than constant lists the models, and the plan of `ml-plan-model-not-built` names constant, since kmeans is built. The third (docs/categorical-encoders.md section 9): stage 2 of issue 21 added the plan verbs `freq` and `hash`, and the
text of `ml.plan.unknown-step` lists the verbs ("not impute, scale, encode, bucket, freq, hash or model", "one of the seven verbs"). Leaving the old text would
make `introspect` and the refusal say something false. The standard output of that one case is therefore compared with the hash of the new text, and everything
else about it (status, files) with the record.
"""

import hashlib
import json
import pathlib
import sys
import unittest

from harness import ROOT, runs_here

sys.path.insert(0, str(ROOT / "scripts"))
import golden_constant  # noqa: E402
import golden_numeric  # noqa: E402

GOLDEN = json.loads((pathlib.Path(__file__).parent / "golden_numeric.json").read_text())

# the numeric fixtures whose standard output changed on purpose, with the SHA-256 of the new text (see the docstring)
CHANGED = {
    "ml-plan-unknown-step": "9aac803db4b8420d95a03f05f56028c5f5d260a811033d76745e996edffad86e",
    # the plan of this fixture names constant, which takes no plan (it named kmeans before k-means was wired: docs/wiring-kmeans.md section 2); the exit status is the same
    "ml-plan-model-not-built": "1d966eb36f993ca466d7c2856019eedc074988a9803d05363d4d53da0ef662ed",
    # the message of the refusal lists the models that can take the flag: kmeans is one of them
    "args-conflict-step-needs-a-model": "c04310948bae0754f026ed3b753c7ad1b0039b2e3b4beb96847dd65ef5d3ea23",
    # the answer of a fit or an eval with --split gained `split.spec`, the canonical text of the split (docs/splits.md section 2); status and files are the record's
    "fit-linear-split-seed-threads": "084103d88c9e88343acc673eabc4d17b661a77e652a1b2057492a82dcc763730",
    "fit-logistic-split-holdout": "a2496e4b0e478267fbd4e33c1e9a75076177950551ac1fb7d69918f848bfc971",
    "eval-linear-split": "f77f8244fa21bfadecf460ef363642ecebcc99f15233956594f68d403d55ef46",
}
# the file of that case that changed with it: the plan the fixture writes
CHANGED_FILES = {"ml-plan-model-not-built": {"p.txt": hashlib.sha256(b"model kind=constant task=regression\n").hexdigest()}}


class NumericBaseline(unittest.TestCase):
    def test_every_old_fixture_is_still_in_the_suite(self):
        ids = {c["id"] for c in golden_numeric.cases()}
        self.assertEqual(sorted(set(GOLDEN) - ids), [], "a numeric fixture of the baseline was removed")
        self.assertGreaterEqual(len(GOLDEN), 150)

    def test_every_old_fixture_gives_the_bytes_of_the_baseline(self):
        failures = []
        for c in golden_numeric.cases():
            if c["id"] not in GOLDEN or not runs_here(c):
                continue
            got = golden_constant.record(c)
            want = dict(GOLDEN[c["id"]])
            if c["id"] in CHANGED:
                want["stdout"] = CHANGED[c["id"]]
            if c["id"] in CHANGED_FILES:
                want["files"] = {**want["files"], **CHANGED_FILES[c["id"]]}
            if got != want:
                failures.append((c["id"], {k: (want[k], got[k]) for k in want if want[k] != got[k]}))
        self.assertEqual(failures, [], "a numeric case changed")

    def test_the_one_exception_is_the_old_hash_and_nothing_else(self):
        """the record still holds the OLD hash of that case (it is not rewritten), and the exception differs from it"""
        for k, new in CHANGED.items():
            self.assertNotEqual(GOLDEN[k]["stdout"], new)
        self.assertEqual(sorted(CHANGED), ["args-conflict-step-needs-a-model", "eval-linear-split", "fit-linear-split-seed-threads", "fit-logistic-split-holdout", "ml-plan-model-not-built", "ml-plan-unknown-step"])

    def test_the_record_holds_the_wired_models(self):
        wired = [k for k in GOLDEN if "linear" in k or "logistic" in k or "pipeline" in k or "plan" in k or "step" in k]
        self.assertGreaterEqual(len(wired), 30)


if __name__ == "__main__":
    unittest.main()

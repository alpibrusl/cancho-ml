"""C9 -- numeric inputs do not change (docs/categorical.md section 1).

`golden_numeric.json` was written by the binary of `v1-next` before stage 1 of issue 21 (commit 7d6183d, scripts/golden_numeric.py): for every fixture that
existed then, the exit status, the SHA-256 of standard output and the SHA-256 of every file left in the sandbox. The binary under test reproduces all of it: the
model files of the constant, linear and logistic models with their steps, and every refusal of the reader, byte for byte. The record is never rewritten to
make a test pass. A fixture added since (the categorical ones) is not in it.

THREE exceptions, pinned here and in the record. Two of them are the texts that name the wired models (docs/wiring-kmeans.md section 2): the refusal of a flag that needs a
model other than constant lists the models, and the plan of `ml-plan-model-not-built` names constant, since kmeans is built. The third (docs/categorical-encoders.md section 9): stage 2 of issue 21 added the plan verbs `freq` and `hash`, and the
text of `ml.plan.unknown-step` lists the verbs ("not impute, scale, encode, bucket, freq, hash or model", "one of the seven verbs"). Leaving the old text would
make `introspect` and the refusal say something false. The standard output of that one case is therefore compared with the hash of the new text, and everything
else about it (status, files) with the record.

SIX MORE, when the cancho-tools pin moved from 8dab917 to 6085e63: `toolbox.cli` and `toolbox.fail` now give `args.missing-value`, `args.bad-value` and
`io.not-a-directory` a hint where they gave `"hint":null` (`args-missing-value`, `args-bad-value`, `args-bad-value-cat-empty`, `args-bad-value-init`,
`io-not-a-directory`, and on macOS `io-read-failed`, which takes the default hint of `toolbox.fail.io_error`). The hint is the only change in each text; the exit status and the files are the record's.
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
    # the messages of ml.limit.depth and ml.limit.nodes name the models that can reach them: boost is one of them (docs/models-boost.md section 6); the exit status and the written files are the baseline's
    "ml-limit-depth": "57fbe4d123801a8801e95bc5ea9011434e4b091c3f7978c91a2785d749d293b5",
    "ml-limit-nodes-a-tree-that-needs-more": "d25e6fd1726e375c4f1da9a30994f3a710a059bdc704876a235d49ab9d12a1bf",
    "ml-limit-nodes-over-the-ceiling": "d25e6fd1726e375c4f1da9a30994f3a710a059bdc704876a235d49ab9d12a1bf",
    "ml-limit-nodes-of-a-forest": "d25e6fd1726e375c4f1da9a30994f3a710a059bdc704876a235d49ab9d12a1bf",
    # the cancho-tools pin moved from 8dab917 to 6085e63: a refusal of toolbox.cli or toolbox.fail that had `"hint":null` now carries the hint the contract gives it
    # (`args.missing-value`, `args.bad-value`, `io.not-a-directory`); the exit status and the written files are the baseline's
    "args-missing-value": "a9ca77ca9309b30fc309b2d41f38193c6710753bc7a116ec7917e5b0adac8dac",
    "args-bad-value": "4cc8724b6346a1aabdf52c08c9830131bc25dd10ec7249cac97859f360ccffa5",
    "io-not-a-directory": "82578073947d68679d5fe53f6b6c6a1bc40a5b178aa951716fc018a0130410a5",
    "args-bad-value-cat-empty": "e2755d744dd8b486394d6496b700e6e0dcb48d457b4900772b99eeb2b9cf9b3c",
    "args-bad-value-init": "383164819cef1752d14f3d73db8db67c12a221f831906486f47561cbb442be25",
    # macOS only (darwin_only): io.read-failed takes the default hint of toolbox.fail.io_error, which was empty at 8dab917; the hash is the one the macOS job printed
    "io-read-failed": "339cc1aba2044b39985e17d66af5152460f946c1b3aadfe1e02162be7a0553a4",
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
        """the record of the deliberate re-record of M22 (docs/tracking.md section 8) holds the hash of the text these cases print now: the exceptions equal it"""
        for k, new in CHANGED.items():
            self.assertEqual(GOLDEN[k]["stdout"], new)
        self.assertEqual(sorted(CHANGED), ["args-bad-value", "args-bad-value-cat-empty", "args-bad-value-init", "args-conflict-step-needs-a-model", "args-missing-value", "io-not-a-directory", "io-read-failed", "ml-limit-depth", "ml-limit-nodes-a-tree-that-needs-more", "ml-limit-nodes-of-a-forest", "ml-limit-nodes-over-the-ceiling", "ml-plan-model-not-built", "ml-plan-unknown-step"])

    def test_the_record_holds_the_wired_models(self):
        wired = [k for k in GOLDEN if "linear" in k or "logistic" in k or "pipeline" in k or "plan" in k or "step" in k]
        self.assertGreaterEqual(len(wired), 30)


if __name__ == "__main__":
    unittest.main()

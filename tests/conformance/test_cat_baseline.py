"""C9 -- numeric inputs do not change (docs/categorical.md section 1).

`golden_numeric.json` was written by the binary of `v1-next` before stage 1 of issue 21 (commit 7d6183d, scripts/golden_numeric.py): for every fixture that
existed then, the exit status, the SHA-256 of standard output and the SHA-256 of every file left in the sandbox. The binary under test reproduces all of it: the
model files of the constant, linear and logistic models with their steps, and every refusal of the reader, byte for byte. The record is never rewritten to
make a test pass. A fixture added since (the categorical ones) is not in it.

THREE exceptions, pinned here and in the record (the re-record of M22, docs/tracking.md section 8), and eight answers that gained `split.spec` after it, pinned here and not in the record. Two of them are the texts that name the wired models (docs/wiring-kmeans.md section 2): the refusal of a flag that needs a
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
    # the answer of a fit or an eval with --split gained `split.spec`, the canonical text of the split (docs/splits.md section 2), after the re-record of M22 wrote the record: the pin is the hash of the answer of the binary that has
    # the provenance section and the spec (8 cases); status and files are the record's
    "fit-linear-split-seed-threads": "bd82ff599d596d10fa17a619ada5b064542f498c0d389a0afd05141deaf10a8d",
    "fit-logistic-split-holdout": "0f378106ae9e74222cb43ad9835df812d1bb0500c48fe91200b976904e1d0ab2",
    "eval-linear-split": "ff98ae1c29c7d9571ecc3c183c5f634b1e900de17192ebc72dd3002885e321a1",
    "fit-forest-split-holdout": "35700d8b136bd9f6128b0f76ed1f1e048d99c8831f9673ef4432d537502ec805",
    "fit-forest-categorical-split-scores-held-out-rows": "0ec886e8ca7c9784f1f53b2e3b70dd6aec18f2e4a105bc07d7ca5443d3984e11",
    "fit-tree-categorical-split-takes-a-missing-cell": "f57d8c601ff0674a241f9eb990e6808b95aca7f52959b678d510f73cd27063d0",
    "fit-kmeans-split-holdout": "2e1152c2a030ce43314cf9d6883657c8dee2f52262fd2acc4f251ae99724c96a",
    "eval-kmeans-split": "7c061d43d0fd3d37b581c65f023762e05a42eeaa9bca10230181ada5e543fddb",
}
# the cases whose answer gained `split.spec` after the re-record of M22: the pin is the hash of the answer with it, the record keeps the hash of the answer without it
SPLIT_ANSWERS = {"eval-kmeans-split", "eval-linear-split", "fit-forest-categorical-split-scores-held-out-rows", "fit-forest-split-holdout", "fit-kmeans-split-holdout", "fit-linear-split-seed-threads", "fit-logistic-split-holdout", "fit-tree-categorical-split-takes-a-missing-cell"}
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
            if k in SPLIT_ANSWERS:
                # the record is the re-record of M22 (section 7 of the model file), written before the answer gained `split.spec`: it keeps the hash without it
                self.assertNotEqual(GOLDEN[k]["stdout"], new)
            else:
                self.assertEqual(GOLDEN[k]["stdout"], new)
        self.assertEqual(sorted(CHANGED), ["args-conflict-step-needs-a-model", "eval-kmeans-split", "eval-linear-split", "fit-forest-categorical-split-scores-held-out-rows", "fit-forest-split-holdout", "fit-kmeans-split-holdout", "fit-linear-split-seed-threads", "fit-logistic-split-holdout", "fit-tree-categorical-split-takes-a-missing-cell", "ml-plan-model-not-built", "ml-plan-unknown-step"])

    def test_the_record_holds_the_wired_models(self):
        wired = [k for k in GOLDEN if "linear" in k or "logistic" in k or "pipeline" in k or "plan" in k or "step" in k]
        self.assertGreaterEqual(len(wired), 30)


if __name__ == "__main__":
    unittest.main()

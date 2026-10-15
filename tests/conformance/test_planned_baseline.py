"""WT12 -- the model files of linear and logistic do not change (docs/wiring-trees.md section 1).

`golden_planned.json` was written by the binary of the parent commit `d481593` (before the tree and the forest were wired) with scripts/golden_planned.py: for 60
seeded cases of the wiring oracle, the exit status, the SHA-256 of the answer of `fit` and the SHA-256 of the model file. This binary reproduces all of it. The
record is never rewritten to make a test pass.
"""

import json
import pathlib
import sys
import tempfile
import unittest

from harness import BIN, ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import golden_planned  # noqa: E402

GOLDEN = json.loads((pathlib.Path(__file__).parent / "golden_planned.json").read_text())

# The answer of a fit with --split gained `split.spec`, the canonical text of the split (docs/splits.md section 2), after the re-record of M22 (docs/tracking.md section 8) wrote the record: the SHA-256 of that
# answer is pinned here, from the answer of the binary that has both the provenance section and the spec; the status and the model file of every case stay the record's. The record keeps the hashes
# of the answers without the spec.
CHANGED = {
    "linear-11": "0af79b62ff29368b00aff15ba748b7a89c740062d8f355b13763c709b17f9209",
    "linear-12": "9eb4a219bd81fe5c13513605bb9b439fb2400cebc4d8d1d5bb91c68819b7d1a3",
    "linear-13": "a1afa791ac713dc450b94cfff59649dded7af4f06cc9dbe2bfb9b4f9caf8b63b",
    "linear-14": "6f17a6adf7cc9ae6d5b909a73c9b1e4b5df4154f4899a1d84dcdd8d3a277a0e5",
    "linear-19": "2914d2565f14a847b542f9dace782157076cef4bb30937d3c050cf612f91af96",
    "linear-20": "bb2528f559f2ffdca3c5827f6029183fb94444f092f7f54a8ca9a0fcfcdd071c",
    "linear-22": "5ea31c472e892c2654649431cfe3bcf259d4c13aa33250be1197d878408f69dd",
    "linear-25": "fdec857f10c1efc99d43040c13c2011084aaaa1b57cb522b739e9f6623955b45",
    "linear-26": "c0dc9b108de8d176dde83179d15ab7e44ac0de284f71ecf6e9a0b9fdc20b215e",
    "linear-3": "3ac222742f050369ac89c4ce8aebdad73440d73e99122c33d61eb5c31476d184",
    "linear-30": "2b3fcef14c7b58c1cb32a36ef1c7819b371ba002bc120ae63abe47c4e5492f5f",
    "linear-5": "bfc8ed72230c282351d5210a8c6b6ab1a118364d263827db8f631bc2028e26f6",
    "linear-6": "2784be93167ba8b00cd9f80e776f0e908f19f76f75b7a6b346c165a8294215e0",
    "logistic-10": "0a9e7f31ca151ee7a6d6d9cdc66d0682824e52617486b264808c404e3c2867d8",
    "logistic-12": "45ab7c44c7efcf851180c49cb1afba8e792645225c2d10e39d39902922d4695c",
    "logistic-17": "809506914233d519ea1109171568c1b956cf15c7f8d042357fc8fb2dcee8fef7",
    "logistic-19": "66c61a8941a80904845579a2faf0640e5d79b2ba7996fe0daf17d96d4a347ead",
    "logistic-24": "16dbdd683d06acbdadf9690e6bee0a349edd9beabdc1cf6490580a0237743e8d",
    "logistic-26": "66c0db6213b53e58e771d52b3972933803180b8f23e5dbf045e60f40e69458d4",
    "logistic-5": "fc7b4c77cd13c4c96d001d737ac1c72e61334da023be5c5e1ad77c303f810691",
    "logistic-8": "037e9e8f207e12eb4217144c89d9cd1c33185314c95a6e82e34bc6c200014b3c",
}


class PlannedBaseline(unittest.TestCase):
    def test_the_record_covers_the_cases(self):
        self.assertEqual(sorted(GOLDEN), sorted(n for n, _ in golden_planned.cases()))
        self.assertEqual(len(GOLDEN), 60)
        self.assertGreaterEqual(sum(1 for g in GOLDEN.values() if g["file"]), 50)

    def test_every_model_file_of_linear_and_logistic_is_byte_for_byte_the_old_one(self):
        golden_planned.ow.BIN = BIN
        failures = []
        with tempfile.TemporaryDirectory() as td:
            for name, case in golden_planned.cases():
                got = golden_planned.record(case, pathlib.Path(td))
                want = dict(GOLDEN[name])
                if name in CHANGED:
                    want["stdout"] = CHANGED[name]
                if got != want:
                    failures.append((name, GOLDEN[name], got))
        self.assertEqual(failures, [], "a linear or logistic file or answer changed")

    def test_the_exceptions_are_the_old_hash_and_nothing_else(self):
        for k, new in CHANGED.items():
            self.assertNotEqual(GOLDEN[k]["stdout"], new)
        self.assertEqual(len(CHANGED), 21)


if __name__ == "__main__":
    unittest.main()

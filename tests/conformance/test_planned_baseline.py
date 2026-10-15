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

# The answer of a fit with --split gained `split.spec`, the canonical text of the split (docs/splits.md section 2): the SHA-256 of that answer changes, with the new hash pinned
# here; the status and the model file of every case stay the record's. The record keeps the old hashes.
CHANGED = {
    "linear-11": "0a240f8abb9e62fbc437b16d9d04529290ef91a908f751e5dc9398d35bbe8382",
    "linear-12": "2b5124f6df393972f61fb08ed3961e59589a2dbdbbb332735a466b129e72b2af",
    "linear-13": "db322bd299b1f77c9397e1ac3956ae346ba559e44f8abfc0e4bd323c780af809",
    "linear-14": "36a1c2261091c26fe33333817f4e7577ab2e63ccb054b14ba63a4292f880d79e",
    "linear-19": "ac1ea0f469603e6df24e5fd02585b5ab8c2b7f62740302916c154aac4f9ed9c1",
    "linear-20": "3b26625ddb46e08e8b4d32d9cc45a2810e28836eee0984669425690cfc414f32",
    "linear-22": "b67c09479dcde077e3497113f7eba571cb60c67da946516b6f32c4c8f2964ab3",
    "linear-25": "db6f7435a1b1c27620d88d2b2be84deaa037122ace92b08ec5c5c1753e696880",
    "linear-26": "03f94487114c1cb060a90e499fa29f8d6ab61cd103887871ade2145c63332a97",
    "linear-3": "20dc819420b6dbbdad6c4e09f0072a729ba853120f47ce7547cdd05e8ec27529",
    "linear-30": "fa5b415925b0ec38cc839cca057ee5230a2b473c7fe0248fc9badea34f16ff35",
    "linear-5": "08ad9de7dfdececcdd1a2ba81f051cb3e772115b74bd787e3c3fd664dc65dedb",
    "linear-6": "67d154508cd675cbcbcc8953db9973ce043e5335181b884932bc22d01bef8ea0",
    "logistic-10": "ff7714f0b7371f628c411f63b0860ec598c3e60f7bf33e5a8041f807164de5ed",
    "logistic-12": "16bb32a82fe1301d9f3e8be8dd94ebff5dda94b203ed75fb974294d51946cbf7",
    "logistic-17": "a45bfb65a8a645e14a7f5ff73710eefb5076718d53bc19d42ff8ac4771fb9743",
    "logistic-19": "bc247e8f7d3098f90e0fe71591ed6543924b54e7881a73b6041b27b2fbea2878",
    "logistic-24": "b70266c3b59af04b672cf8fd0b7495f2b0c2238b0a7e8c9b162e8f982d244506",
    "logistic-26": "9ba5b33ee5ab982824fbf64b692c642a47cda58c836eabef4585190a7f4b1a0a",
    "logistic-5": "d7f8608de101e0e7469b73220c432bc975d76f78db0bd88e0ed4f282dc705c47",
    "logistic-8": "78f0be67e0cb142b52d20cb9957c4c60072e6559331d7f030b26aed44996ec34",
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

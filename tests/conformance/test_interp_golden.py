"""I9 -- `describe` gains one member and changes nothing else (docs/interpret.md section 6).

tests/conformance/golden_describe_before_importance.json holds the answers of `ml describe` of the 19 models of the serving corpus as they were before the member `importance` existed.
The answers of the built binary, with the member `importance` (the last one) taken out of the text, are those bytes: no other member, no order of members, no number moved."""

import json
import pathlib
import sys
import unittest

from harness import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import golden_describe  # noqa: E402

BEFORE = pathlib.Path(__file__).resolve().parent / "golden_describe_before_importance.json"
KEY = ',"importance":'


def without_member(text):
    """the answer's text with the member `importance` (the one after `limits`) removed; the value is decoded to find where it ends"""
    cut = text.index(KEY, text.index(',"limits":'))
    value, end = json.JSONDecoder().raw_decode(text[cut + len(KEY):])
    return text[:cut] + text[cut + len(KEY) + end:], value


class DescribeGolden(unittest.TestCase):
    def test_the_answers_without_the_member_are_the_answers_before(self):
        old = json.loads(BEFORE.read_text())
        new = json.loads(golden_describe.record())
        self.assertEqual(sorted(new), sorted(old))
        for model, text in new.items():
            stripped, value = without_member(text)
            self.assertEqual(stripped, old[model], model)
            doc = json.loads(text)
            self.assertEqual(list(doc["data"])[-1], "importance", model)
            self.assertEqual(doc["data"]["importance"], value)

    def test_the_check_can_fail(self):
        # an answer with another member moved fails the comparison
        old = json.loads(BEFORE.read_text())
        new = json.loads(golden_describe.record())
        model = sorted(new)[0]
        stripped, _ = without_member(new[model].replace('"rows":', '"rows ":', 1))
        self.assertNotEqual(stripped, old[model])


if __name__ == "__main__":
    unittest.main()

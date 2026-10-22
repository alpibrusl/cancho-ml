"""WR9 -- a damaged file of kind 10 never traps, and the payload is read as strictly as docs/wiring-recommend.md section 5 says.

Every prefix of a file (at a stride) and every flipped byte (at a stride) of an explicit and an implicit file is refused with a model tag or, for a flip the checksum would not catch because
it is repaired, answered or refused with a tag; seeded multi-byte damage of the payload with the checksum repaired; and the fields that one edit can break one rule at a time (the kind, the
method, the factors, the counts, the lambda and the alpha, the id tables, the history) are each ml.model.malformed or ml.model.truncated. The payload itself is judged field by field by
`oracle_recommend.py --payload`, against the library; this test judges the file around it and the verbs that read it."""

import random
import struct
import unittest

from harness import TRAPS, Sandbox, fix_checksum, run, sections, set_bytes, validate

MODEL_RULES = {"ml.model.bad-magic", "ml.model.version", "ml.model.truncated", "ml.model.checksum", "ml.model.malformed"}
LOG = "user,item,rating\n" + "".join("%s,%s,%d\n" % (u, i, 1 + (k * 5 + j * 3) % 5) for k, u in enumerate(["ann", "bob", "cy", "di", "ed"]) for j, i in enumerate(["p", "q", "r", "s"][: 2 + k % 3]))
STRIDE = 7


class RecommendFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        cls.sb.write("log.csv", LOG)
        cls.sb.write("who.csv", "who\nann\nbob\n")
        cls.files = {}
        for name, extra in (("e", ["--rating", "rating"]), ("i", ["--feedback", "implicit"])):
            r = cls.sb.run("fit", "--model", "recommend", "--user", "user", "--item", "item", *extra, "--factors", "2", "--iterations", "2", "--output", name + ".cml", "log.csv")
            assert r.status == 0, r
            cls.files[name] = cls.sb.read(name + ".cml")

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def answer(self, data, verb="predict"):
        self.sb.write("x.cml", data)
        if verb == "describe":
            return self.sb.run("describe", "--model-file", "x.cml")
        if verb == "eval":
            return self.sb.run("eval", "--model-file", "x.cml", "--user", "user", "--item", "item", "--top", "1", "log.csv")
        return self.sb.run("predict", "--model-file", "x.cml", "--for", "who", "--top", "1", "who.csv")

    def refused(self, data, label, rules=MODEL_RULES, verb="predict"):
        r = self.answer(data, verb)
        self.assertNotIn(r.status, TRAPS, (label, r))
        self.assertIn(r.first_rule(), rules, (label, r))
        self.assertEqual(validate(r), [], (label, r))

    def either(self, data, label, verb="predict"):
        """answered, or refused with a model tag: never a trap, and an answer is valid"""
        r = self.answer(data, verb)
        self.assertNotIn(r.status, TRAPS, (label, r))
        self.assertEqual(validate(r), [], (label, r))
        if r.status != 0:
            self.assertIn(r.first_rule(), MODEL_RULES | {"ml.recommend.unknown-user", "ml.param.out-of-range"}, (label, r))

    def test_the_undamaged_files_are_answered(self):
        for key, data in self.files.items():
            for verb in ("predict", "describe", "eval"):
                self.assertEqual(self.answer(data, verb).status, 0, (key, verb))

    def test_every_truncation_is_refused(self):
        for key, data in self.files.items():
            for n in list(range(0, 80)) + list(range(80, len(data), STRIDE)):
                self.refused(data[:n], "%s prefix %d" % (key, n))

    def test_every_flipped_byte_is_refused_by_the_checksum(self):
        for key, data in self.files.items():
            for at in range(0, len(data), 3):
                b = bytearray(data)
                b[at] ^= 0x5A
                self.refused(bytes(b), "%s flip %d" % (key, at))

    def test_every_flipped_byte_with_the_checksum_repaired_is_answered_or_refused(self):
        for key, data in self.files.items():
            for at in range(0, len(data) - 32, 2):
                b = bytearray(data)
                b[at] ^= 0x5A
                self.either(fix_checksum(bytes(b)), "%s repaired flip %d" % (key, at), "describe")

    def test_seeded_damage_of_the_payload(self):
        rng = random.Random(10)
        for key, data in self.files.items():
            tag, start, n = sections(data)[3]
            for _ in range(150):
                b = bytearray(data)
                for _ in range(rng.randint(1, 6)):
                    b[start + rng.randrange(n)] = rng.randrange(256)
                self.either(fix_checksum(bytes(b)), "%s seeded" % key)

    def edit(self, key, at, new):
        data = self.files[key]
        return fix_checksum(set_bytes(data, sections(data)[3][1] + at, new))

    def test_the_header_fields(self):
        for key in self.files:
            for at, values in ((0, (0, 1, 9, 11, 1 << 40)), (8, (2, 3, 1 << 63, (1 << 64) - 1)), (16, (0, 257, 1 << 62)), (24, (0, 1001, 1 << 62)), (32, (0, 1 << 20 | 1, 1 << 62)), (40, (0, 1 << 20 | 1, 1 << 62)),
                               (48, (0, 1 << 25 | 1, 1 << 62)), (56, (0, (1 << 64) - 1, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000)), (64, (0x7FF0000000000000, 0x7FF8000000000000, 0xBFF0000000000000))):
                for v in values:
                    self.either(self.edit(key, at, struct.pack("<Q", v)), "%s header %d = %#x" % (key, at, v), "describe")

    def test_the_length_of_the_payload(self):
        for key in self.files:
            data = self.files[key]
            tag, start, n = sections(data)[3]
            for cut in (1, 7, 8, 9, 16, 80, n - 80):
                body = data[start:start + n - cut]
                head = data[:start - 8] + struct.pack("<Q", len(body))
                self.refused(fix_checksum(head + body + data[start + n:]), "%s cut %d" % (key, cut))
            for add in (b"\0", b"\0" * 8, bytes(range(40))):
                body = data[start:start + n] + add
                head = data[:start - 8] + struct.pack("<Q", len(body))
                self.refused(fix_checksum(head + body + data[start + n:]), "%s add %r" % (key, add[:3]), rules={"ml.model.malformed"})

    def test_a_plan_that_disagrees_with_the_payload_is_malformed(self):
        data = self.files["e"]
        tag, start, n = sections(data)[0]
        plan = data[start:start + n]
        for old, new in ((b"factors=2", b"factors=3"), (b"feedback=explicit", b"feedback=implicit"), (b"iterations=2", b"iterations=3")):
            self.assertIn(old, plan)
            body = plan.replace(old, new)
            head = data[:start - 8] + struct.pack("<Q", len(body))
            self.refused(fix_checksum(head + body + data[start + n:]), "plan %r" % new, rules={"ml.model.malformed"}, verb="describe")

    def test_the_meta_rows_must_be_the_interactions(self):
        data = self.files["e"]
        tag, start, n = sections(data)[4]
        rows = struct.unpack_from("<Q", data, start)[0]
        for new in (rows + 1, rows - 1, 0, 1 << 40):
            self.refused(fix_checksum(set_bytes(data, start, struct.pack("<Q", new))), "meta rows %d" % new, rules={"ml.model.malformed"}, verb="describe")

    def test_the_schema_must_have_no_target_and_two_or_three_columns(self):
        data = self.files["e"]
        tag, start, n = sections(data)[1]
        for at, new in ((0, struct.pack("<I", 2)), (0, struct.pack("<I", 4)), (4, struct.pack("<I", 0)), (4, struct.pack("<I", 2))):
            self.refused(fix_checksum(set_bytes(data, start + at, new)), "schema %d" % at, rules=MODEL_RULES, verb="describe")

    def with_section(self, data, index, body):
        tag, start, n = sections(data)[index]
        head = data[:start - 8] + struct.pack("<Q", len(body))
        return fix_checksum(head + body + data[start + n:])

    def test_an_explicit_file_with_the_two_column_schema_of_an_implicit_one_is_malformed(self):
        # the plan and the payload say explicit; the schema (a whole, valid one) names two columns: only the check of the column count can refuse it
        data = self.files["e"]
        schema = self.files["i"][sections(self.files["i"])[1][1]:][:sections(self.files["i"])[1][2]]
        # the column count of the meta section is set to two as well, or the check of the meta (mc != scols) refuses the file before the check of the shape is reached
        swapped = self.with_section(data, 1, schema)
        two = fix_checksum(set_bytes(swapped, sections(swapped)[4][1] + 8, struct.pack("<Q", 2)))
        self.refused(two, "explicit with two columns", rules={"ml.model.malformed"}, verb="describe")
        self.refused(two, "explicit with two columns", rules={"ml.model.malformed"})

    def test_a_plan_of_the_recommender_over_the_payload_of_another_kind_is_malformed(self):
        data = self.files["e"]
        tag, start, n = sections(data)[3]
        for kind in (1, 2, 3, 8, 11):
            self.refused(fix_checksum(set_bytes(data, start, kind.to_bytes(4, "little"))), "model kind %d" % kind, rules={"ml.model.malformed"}, verb="describe")


if __name__ == "__main__":
    unittest.main()

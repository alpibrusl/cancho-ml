"""GB6 -- a damaged file of kind 8 never traps, and the payload is read as strictly as docs/models-boost.md section 5 says.

Every truncation and every flipped byte of a classification and two regression files (with missing cells, stored steps, a row sample and early stopping) is refused with a tag,
with the checksum wrong; seeded damage of every section with the checksum repaired is refused with a tag or answered; the huge counts a hostile file would put in a header or a
record header (2^63 - 1 and its neighbours) are refused before any allocation; and the fields of the payload that a single edit can break one rule at a time (the header, the
floats, the record headers, the offsets, the nodes, the labels record, the length) are each ml.model.malformed."""

import random
import struct
import unittest
from concurrent.futures import ThreadPoolExecutor

from harness import TRAPS, Sandbox, fix_checksum, run, sections, set_bytes, validate
from test_model_file import MODEL_RULES, payloads, write_file
from test_wire_boost import header
from test_wire_trees import classification_csv, regression_csv

OTHER_RULES = {"ml.schema.mismatch", "ml.input.unknown-category", "ml.input.non-finite"}


def i64(b, at):
    return struct.unpack_from("<q", b, at)[0]


class BoostFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        cls.sb.write("reg.csv", regression_csv(n=60))
        cls.sb.write("cls.csv", classification_csv(n=50))
        steps = "scale column=a kind=standard;bucket column=c kind=width bins=3"
        for out, args in (("cf.cml", ["--task", "classification", "--rounds", "5", "--max-depth", "3", "--min-leaf", "2", "--seed", "4", "--step", steps, "--target", "lab", "cls.csv"]),
                          ("rf.cml", ["--rounds", "4", "--max-depth", "3", "--min-leaf", "2", "--bins", "8", "--subsample", "0.7", "--target", "y", "reg.csv"]),
                          ("es.cml", ["--rounds", "50", "--max-depth", "2", "--min-leaf", "2", "--bins", "8", "--early-stopping", "3", "--split", "train=0.8", "--learning-rate", "0.9", "--target", "y", "reg.csv"])):
            r = cls.sb.run("fit", "--model", "boost", "--output", out, *args)
            assert r.status == 0, r
        cls.files = {"cf": cls.sb.read("cf.cml"), "rf": cls.sb.read("rf.cml"), "es": cls.sb.read("es.cml")}
        cls.csv = {"cf": "cls.csv", "rf": "reg.csv", "es": "reg.csv"}

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def edit_model(self, key, at, new):
        data = self.files[key]
        start = sections(data)[3][1]
        return fix_checksum(set_bytes(data, start + at, new))

    def resize_model(self, key, cut=0, add=b""):
        data = self.files[key]
        tag, start, n = sections(data)[3]
        body = data[start:start + n]
        body = body[:len(body) - cut] + add
        head = data[:start - 8] + struct.pack("<Q", len(body))
        return fix_checksum(head + body + data[start + n:])

    def predict(self, key, data):
        self.sb.write("e.cml", data)
        return self.sb.run("predict", "--model-file", "e.cml", "--limit", "5", self.csv[key])

    def refused(self, key, data, label, rule="ml.model.malformed"):
        r = self.predict(key, data)
        self.assertNotIn(r.status, TRAPS, (label, r))
        self.assertIn(r.first_rule(), rule if isinstance(rule, tuple) else (rule,), (label, r))
        self.assertEqual(validate(r), [], (label, r))

    # ------------------------------------------------------------------ the payload, one rule at a time
    def test_the_undamaged_files_are_answered(self):
        for key, data in self.files.items():
            r = self.predict(key, data)
            self.assertEqual(r.status, 0, (key, r))

    def test_the_early_stopped_file_holds_fewer_rounds_than_it_asked(self):
        h = header(self.files["es"])
        self.assertEqual((h[7], h[8]), (3, 50))
        self.assertLess(h[1], 50)

    def test_the_header_of_the_payload(self):
        # the header record starts at 4 (record header 16 bytes), its ten values at 20: task, rounds, d, bins, total, max_depth, min_leaf, early, requested, subsample K
        for key in self.files:
            head = list(header(self.files[key]))
            task, rounds, d, bins, total, maxd, minleaf, early, requested, subk = head
            for k, vals in ((0, [2, -1]), (1, [0, 10001, rounds + 1, rounds - 1, (1 << 63) - 1]), (2, [0, d + 1, d - 1, 4097, (1 << 63) - 1]), (3, [1, 256]),
                            (4, [0, total + 1, total - 1, (1 << 22) + 1, 1 << 40, (1 << 63) - 1, -1]), (5, [0, 31, (1 << 63) - 1]), (6, [0, -1]),
                            (7, [-1, 1001, (1 << 63) - 1]), (8, [0, 10001, rounds - 1, (1 << 63) - 1]), (9, [0, 1000001, -1, (1 << 63) - 1])):
                for v in vals:
                    if v == head[k]:
                        continue
                    self.refused(key, self.edit_model(key, 20 + 8 * k, struct.pack("<q", v)), "%s header %d = %d" % (key, k, v))
            # early stopping off but fewer rounds than requested, and requested below the rounds
            if early == 0:
                self.refused(key, self.edit_model(key, 20 + 8 * 8, struct.pack("<q", rounds + 1)), "%s requested above the rounds with no early stopping" % key)

    def test_the_floats_of_the_payload(self):
        # base score, learning rate, l2 at 116, 124, 132
        nan, inf = 0x7FF8000000000000, 0x7FF0000000000000
        for key in self.files:
            for at, bits_list in ((116, [nan, inf, 0xFFF0000000000000]), (124, [nan, inf, 0, 0x8000000000000000, 0x3FF0000000000001, 0xBFF0000000000000]),
                                  (132, [nan, inf, 0x8000000000000001, 0xBFF0000000000000, 0x7FEFFFFFFFFFFFFF]) ):
                for bits in bits_list:
                    self.refused(key, self.edit_model(key, at, struct.pack("<Q", bits)), "%s float at %d = %x" % (key, at, bits))

    def test_the_kind_and_the_record_headers(self):
        hdrs = [4, 100, 140]
        for key in self.files:
            m = payloads(self.files[key])[4]
            trees = i64(m, 28)
            total = i64(m, 52)
            starts = hdrs + [156 + 8 * (trees + 1) + (r - 4) * (16 + 8 * total) for r in range(4, 10)]
            edits = [(0, struct.pack("<I", k)) for k in (4, 5, 6, 7, 0, 9)]
            for s in starts:
                edits += [(s, struct.pack("<I", 7)), (s + 4, struct.pack("<I", 2)), (s + 8, struct.pack("<Q", 7)), (s + 8, struct.pack("<Q", (1 << 63) - 1)), (s + 8, struct.pack("<Q", 1 << 62))]
            for at, new in edits:
                self.refused(key, self.edit_model(key, at, new), "%s record header at %d" % (key, at), ("ml.model.malformed", "ml.model.version", "ml.model.truncated"))

    def test_the_length_of_the_payload(self):
        for key in self.files:
            for cut in (1, 8, 31, 32, 33, 100):
                self.refused(key, self.resize_model(key, cut=cut), "%s cut %d" % (key, cut), ("ml.model.malformed", "ml.model.truncated"))
            for add in (b"\0", b"\0" * 8, b"\0" * 31, b"\0" * 33, bytes(range(40))):
                self.refused(key, self.resize_model(key, add=add), "%s add %d" % (key, len(add)))
        self.refused("cf", self.resize_model("cf", cut=len(payloads(self.files["cf"])[4])), "empty model section", ("ml.model.malformed", "ml.model.truncated"))
        for keep in (3, 4, 20, 100, 139, 140, 155):
            m = payloads(self.files["rf"])[4]
            self.refused("rf", self.resize_model("rf", cut=len(m) - keep), "kept %d bytes" % keep, ("ml.model.malformed", "ml.model.truncated"))

    def test_the_labels_record_of_a_classification(self):
        data = self.files["cf"]
        m = payloads(data)[4]
        lab = len(m) - 32
        self.assertEqual(struct.unpack_from("<IIQqq", m, lab), (1, 1, 2, 3, 7))
        for at, new in ((0, struct.pack("<I", 2)), (0, struct.pack("<I", 0)), (4, struct.pack("<I", 2)), (8, struct.pack("<Q", 1)), (8, struct.pack("<Q", 3)), (8, struct.pack("<Q", 1 << 60)),
                        (16, struct.pack("<q", 7)), (24, struct.pack("<q", 3))):
            self.refused("cf", self.edit_model("cf", lab + at, new), "labels at %d" % at)
        r = self.predict("cf", self.edit_model("cf", lab + 16, struct.pack("<qq", -5, 100)))
        self.assertEqual(r.status, 0, r)
        self.assertTrue({int(row[0]) for row in r.data()["rows"]} <= {-5, 100})

    def test_a_regression_payload_has_no_labels(self):
        self.refused("rf", self.resize_model("rf", add=struct.pack("<IIQqq", 1, 1, 2, 0, 1)), "labels on a regression")

    def test_the_task_of_the_plan_and_the_task_of_the_payload_agree(self):
        for key, old, new in (("rf", b"task=regression", b"task=classification"), ("cf", b"task=classification", b"task=regression")):
            secs = payloads(self.files[key])
            self.assertIn(old, secs[1])
            secs[1] = secs[1].replace(old, new)
            self.refused(key, write_file(secs), "%s plan %s over the payload" % (key, new))

    def test_the_tree_offsets_and_the_nodes(self):
        for key in self.files:
            m = payloads(self.files[key])[4]
            trees, total = i64(m, 28), i64(m, 52)
            first = 156
            for t in range(trees + 1):
                good = i64(m, first + 8 * t)
                for v in (good + 1, good - 1, 0, total + 1, -1, (1 << 63) - 1):
                    if v == good:
                        continue
                    self.refused(key, self.edit_model(key, first + 8 * t, struct.pack("<q", v)), "%s offset %d = %d" % (key, t, v))
            d, bins = i64(m, 36), i64(m, 44)
            rec = lambda r: 156 + 8 * (trees + 1) + (r - 4) * (16 + 8 * total) + 16  # noqa: E731
            bad_by_record = {4: [d, -2, 1 << 40, (1 << 63) - 1], 5: [bins, -2, 1 << 40], 6: [2, -1, 1 << 40], 7: [-2, total, 1 << 40, (1 << 63) - 1], 8: [-2, total, 1 << 40, (1 << 63) - 1]}
            rng = random.Random(5)
            for r_, vals in bad_by_record.items():
                for g in rng.sample(range(total), min(total, 8)):
                    for v in vals:
                        at = rec(r_) + 8 * g
                        if i64(m, at) == v:
                            continue
                        r = self.predict(key, self.edit_model(key, at, struct.pack("<q", v)))
                        self.assertNotIn(r.status, TRAPS, (key, r_, g, v))
                        self.assertEqual(validate(r), [], (key, r_, g, v))
                        self.assertEqual(r.first_rule(), "ml.model.malformed", (key, r_, g, v, r))

    def test_a_value_that_is_not_a_number_is_refused(self):
        for key in self.files:
            m = payloads(self.files[key])[4]
            trees, total = i64(m, 28), i64(m, 52)
            val0 = 156 + 8 * (trees + 1) + 5 * (16 + 8 * total) + 16
            for bits in (0x7FF8000000000000, 0x7FF0000000000000, 0xFFF0000000000000):
                self.refused(key, self.edit_model(key, val0, struct.pack("<Q", bits)), "%s val NaN or inf" % key)

    def test_a_leaf_value_may_be_any_finite_number(self):
        # the values are the model: a file whose leaf says another finite number is a valid model with another prediction
        key = "rf"
        m = payloads(self.files[key])[4]
        trees, total = i64(m, 28), i64(m, 52)
        val0 = 156 + 8 * (trees + 1) + 5 * (16 + 8 * total) + 16
        before = self.predict(key, self.files[key]).data()["rows"]
        # the last node of the first tree is a leaf (preorder)
        off = i64(m, 156 + 8)
        at = val0 + 8 * (off - 1)
        r = self.predict(key, self.edit_model(key, at, struct.pack("<d", 1e30)))
        self.assertEqual(r.status, 0, r)
        self.assertNotEqual(r.data()["rows"], before)

    # ------------------------------------------------------------------ nothing traps
    def check_many(self, items, accept_ok):
        sb = self.sb

        def one(item):
            label, key, data, repaired = item
            name = "w-%d.cml" % (abs(hash(label)) % 10 ** 9)
            sb.write(name, data)
            return label, repaired, run("--root", sb.dir, "predict", "--model-file", name, "--limit", "3", self.csv[key])

        bad = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for label, repaired, r in ex.map(one, items):
                if r.status in TRAPS or r.status < 0:
                    bad.append((label, "TRAP %d" % r.status))
                    continue
                pr = validate(r)
                if pr:
                    bad.append((label, pr))
                elif r.status == 0:
                    if not repaired and not accept_ok:
                        bad.append((label, "a damaged file with a wrong checksum was accepted"))
                elif r.first_rule() not in MODEL_RULES | OTHER_RULES:
                    bad.append((label, r.first_rule()))
        self.assertEqual(bad, [])

    def test_every_truncation_is_refused(self):
        items = []
        for key, data in self.files.items():
            for cut in range(len(data)):
                items.append(("%s-cut-%d" % (key, cut), key, data[:cut], False))
        self.check_many(items, False)

    def test_every_flipped_byte_is_refused(self):
        items = []
        for key, data in self.files.items():
            for at in range(len(data)):
                for mask in (0x01, 0x80):
                    b = bytearray(data)
                    b[at] ^= mask
                    items.append(("%s-flip-%d-%d" % (key, at, mask), key, bytes(b), False))
        self.check_many(items, False)

    def test_seeded_damage_of_each_section_with_a_repaired_checksum(self):
        rng = random.Random(20261017)
        items = []
        for i in range(1500):
            key = rng.choice(list(self.files))
            data = self.files[key]
            b = bytearray(data)
            tag, start, n = rng.choice([s for s in sections(data) if s[2] > 0] + [sections(data)[3]] * 3)
            for _ in range(rng.randint(1, 4)):
                at = start + rng.randrange(n)
                b[at] = rng.choice((0, 1, 2, 0x7F, 0x80, 0xFF, rng.randrange(256)))
            items.append(("%s-damage-%d" % (key, i), key, fix_checksum(bytes(b)), True))
        self.check_many(items, True)

    def test_seeded_damage_of_the_model_section_by_whole_words(self):
        rng = random.Random(20261018)
        words = [0, 1, 2, 3, -1, -2, 1 << 20, (1 << 20) + 1, 1 << 22, (1 << 22) + 1, 1 << 31, 1 << 32, 1 << 62, (1 << 63) - 1, -(1 << 63)]
        items = []
        for i in range(1500):
            key = rng.choice(list(self.files))
            data = self.files[key]
            tag, start, n = sections(data)[3]
            b = bytearray(data)
            for _ in range(rng.randint(1, 3)):
                at = start + 4 + 8 * rng.randrange((n - 4) // 8)
                b[at:at + 8] = struct.pack("<q", rng.choice(words))
            items.append(("%s-word-%d" % (key, i), key, fix_checksum(bytes(b)), True))
        self.check_many(items, True)


if __name__ == "__main__":
    unittest.main()

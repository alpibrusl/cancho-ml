"""WT3, WT11 -- a damaged file of kind 7 never traps, and the payload is read as strictly as docs/wiring-trees.md section 4 says.

Every truncation and every flipped byte of a classification forest, a regression forest and a tree (with missing cells and stored steps) is refused with a tag, with
the checksum wrong; seeded damage of every section with the checksum repaired is refused with a tag or answered; and the fields of the payload that a single edit can
break one rule at a time (the header, the record headers, the offsets, the nodes, the labels record, the length) are each ml.model.malformed."""

import random
import struct
import unittest
from concurrent.futures import ThreadPoolExecutor

from harness import TRAPS, Sandbox, fix_checksum, run, sections, set_bytes, validate
from test_model_file import MODEL_RULES, payloads
from test_wire_trees import classification_csv, regression_csv

OTHER_RULES = {"ml.schema.mismatch", "ml.input.unknown-category", "ml.input.non-finite"}


def i64(b, at):
    return struct.unpack_from("<q", b, at)[0]


class TreeFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        cls.sb.write("reg.csv", regression_csv(n=40))
        cls.sb.write("cls.csv", classification_csv(n=50))
        steps = "scale column=a kind=standard;bucket column=c kind=width bins=3"
        for out, args in (("cf.cml", ["--model", "forest", "--task", "classification", "--trees", "3", "--max-depth", "3", "--seed", "4", "--step", steps, "--target", "lab", "cls.csv"]),
                          ("rf.cml", ["--model", "forest", "--trees", "3", "--max-depth", "3", "--bins", "8", "--target", "y", "reg.csv"]),
                          ("t.cml", ["--model", "tree", "--max-depth", "4", "--bins", "8", "--target", "y", "reg.csv"])):
            r = cls.sb.run("fit", "--output", out, *args)
            assert r.status == 0, r
        cls.files = {"cf": cls.sb.read("cf.cml"), "rf": cls.sb.read("rf.cml"), "t": cls.sb.read("t.cml")}
        cls.csv = {"cf": "cls.csv", "rf": "reg.csv", "t": "reg.csv"}

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def edit_model(self, key, at, new):
        """the file with `new` written at offset `at` of the model section, the checksum repaired"""
        data = self.files[key]
        start = sections(data)[3][1]
        return fix_checksum(set_bytes(data, start + at, new))

    def resize_model(self, key, cut=0, add=b""):
        """the file whose model section lost its last `cut` bytes and gained `add`, the section length and the checksum repaired"""
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

    def test_the_header_of_the_library_payload(self):
        # the header record starts at 4 (record header 16 bytes), its eight values at 20: task, trees, d, bins, total, max_depth, min_leaf, mtry
        for key in self.files:
            head = [i64(self.files[key][sections(self.files[key])[3][1]:], 20 + 8 * k) for k in range(8)]
            task, trees, d, bins, total, maxd, minleaf, mtry = head
            for k, vals in ((0, [2, -1]), (1, [0, 1025, trees + 1, trees - 1 if trees > 1 else 99]), (2, [0, d + 1, d - 1]), (3, [1, 256]),
                            (4, [0, total + 1, total - 1, 1 << 24 + 1, 1 << 40]), (5, [0, 31]), (6, [0]), (7, [0, d + 1])):
                for v in vals:
                    if v == head[k]:
                        continue
                    self.refused(key, self.edit_model(key, 20 + 8 * k, struct.pack("<q", v)), "%s header %d = %d" % (key, k, v))

    def test_the_kind_and_the_record_headers(self):
        for key in self.files:
            for at, new in ((0, struct.pack("<I", 4)), (0, struct.pack("<I", 5)), (0, struct.pack("<I", 6)), (0, struct.pack("<I", 0)), (4, struct.pack("<I", 2)), (8, struct.pack("<I", 2)),
                            (12, struct.pack("<Q", 7)), (12, struct.pack("<Q", 1 << 62))):
                # a kind of 4 or 5 in the payload of a file whose plan names a forest is a payload of another family: malformed, or, the sum order of those kinds being 2, a version
                self.refused(key, self.edit_model(key, at, new), "%s record header at %d" % (key, at), ("ml.model.malformed", "ml.model.version"))

    def test_the_length_of_the_payload(self):
        for key in self.files:
            for cut in (1, 8, 31, 32, 33, 100):
                self.refused(key, self.resize_model(key, cut=cut), "%s cut %d" % (key, cut))
            for add in (b"\0", b"\0" * 8, b"\0" * 31, b"\0" * 33, bytes(range(40))):
                self.refused(key, self.resize_model(key, add=add), "%s add %d" % (key, len(add)))
        self.refused("cf", self.resize_model("cf", cut=len(payloads(self.files["cf"])[4])), "empty model section")

    def test_the_labels_record_of_a_classification(self):
        data = self.files["cf"]
        m = payloads(data)[4]
        lab = len(m) - 32
        self.assertEqual(struct.unpack_from("<IIQqq", m, lab), (1, 1, 2, 3, 7))
        for at, new in ((0, struct.pack("<I", 2)), (0, struct.pack("<I", 0)), (4, struct.pack("<I", 2)), (8, struct.pack("<Q", 1)), (8, struct.pack("<Q", 3)), (8, struct.pack("<Q", 1 << 60)),
                        (16, struct.pack("<q", 7)), (24, struct.pack("<q", 3))):
            self.refused("cf", self.edit_model("cf", lab + at, new), "labels at %d" % at)
        # other labels are fine: they are only what predict prints
        r = self.predict("cf", self.edit_model("cf", lab + 16, struct.pack("<qq", -5, 100)))
        self.assertEqual(r.status, 0, r)
        self.assertTrue({int(row[0]) for row in r.data()["rows"]} <= {-5, 100})

    def test_a_regression_payload_ends_with_its_values_and_has_no_labels(self):
        data = self.files["rf"]
        self.assertEqual(len(payloads(data)[4]) % 8, 4)
        self.refused("rf", self.resize_model("rf", add=struct.pack("<IIQqq", 1, 1, 2, 0, 1)), "labels on a regression")

    def test_the_task_of_the_plan_and_the_task_of_the_payload_agree(self):
        # the plan says classification: turn the payload's task into regression, and a regression plan's into classification
        self.refused("cf", self.edit_model("cf", 20, struct.pack("<q", 1)), "payload regression, plan classification")
        data = self.files["rf"]
        self.refused("rf", self.edit_model("rf", 20, struct.pack("<q", 0)), "payload classification, plan regression")

    def test_the_tree_offsets_and_the_nodes(self):
        for key in self.files:
            data = self.files[key]
            m = payloads(data)[4]
            task, trees, d, bins, total = (i64(m, 20 + 8 * k) for k in range(5))
            offs = 4 + 16 + 64                                   # the offsets record header, then trees + 1 values
            first = offs + 16
            for t in range(trees + 1):
                good = i64(m, first + 8 * t)
                for v in (good + 1, good - 1, 0, total + 1, -1):
                    if v == good:
                        continue
                    self.refused(key, self.edit_model(key, first + 8 * t, struct.pack("<q", v)), "%s offset %d = %d" % (key, t, v))
            # the fields of every node, one record each: feat, bin, dleft, left, right, na, nb; the position of node g of record r (3 to 9)
            rec = lambda r: 4 + 16 + 64 + 16 + 8 * (trees + 1) + (r - 3) * (16 + 8 * total) + 16
            bad_by_record = {3: [d, -2, 1 << 40], 4: [bins, -2, 1 << 40], 5: [2, -1, 1 << 40], 6: [-2, total, 1 << 40], 7: [-2, total, 1 << 40], 8: [-1, 1 << 40, (1 << 63) - 1], 9: [-1, 1 << 40, (1 << 63) - 1]}
            rng = random.Random(5)
            for r_, vals in bad_by_record.items():
                for g in rng.sample(range(total), min(total, 6)):
                    for v in vals:
                        at = rec(r_) + 8 * g
                        if i64(m, at) == v:
                            continue
                        r = self.predict(key, self.edit_model(key, at, struct.pack("<q", v)))
                        self.assertNotIn(r.status, TRAPS, (key, r_, g, v))
                        self.assertEqual(validate(r), [], (key, r_, g, v))
                        if r_ <= 7:
                            # feat, bin, dleft, left, right: out of range or not the preorder of the tree
                            self.assertEqual(r.first_rule(), "ml.model.malformed", (key, r_, g, v, r))
                        elif r.status != 0:
                            # the counts of an internal node are not read by a walk, a leaf's must make its value
                            self.assertEqual(r.first_rule(), "ml.model.malformed", (key, r_, g, v, r))

    def test_a_threshold_that_is_not_a_number_is_refused(self):
        for key in self.files:
            m = payloads(self.files[key])[4]
            trees, total = i64(m, 28), i64(m, 52)
            val0 = 4 + 16 + 64 + 16 + 8 * (trees + 1) + 7 * (16 + 8 * total) + 16
            for bits in (0x7FF8000000000000, 0x7FF0000000000000, 0xFFF0000000000000):
                self.refused(key, self.edit_model(key, val0, struct.pack("<Q", bits)), "%s val NaN or inf" % key)

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
        rng = random.Random(20261015)
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
        # a node field or an offset or a header value replaced by a word that is likely to be at a boundary
        rng = random.Random(20261016)
        words = [0, 1, 2, 3, -1, -2, 1 << 20, (1 << 20) + 1, 1 << 31, 1 << 32, 1 << 62, (1 << 63) - 1, -(1 << 63)]
        items = []
        for i in range(1200):
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

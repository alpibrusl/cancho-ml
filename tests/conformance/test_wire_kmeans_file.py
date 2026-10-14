"""WK4, WK12 -- a damaged file of kind 6 never traps, and the payload is read as strictly as docs/wiring-kmeans.md section 6 says.

Every truncation and every flipped byte of two files (one with stored steps, one with a string table) is refused with a tag, with the checksum wrong; seeded damage of
every section with the checksum repaired is refused with a tag or answered; and the fields that a single edit can break one rule at a time (the kind, the record
header, k, the width, every centroid value, the length, the target index of the schema, the k of the plan) are each ml.model.malformed."""

import random
import struct
import unittest
from concurrent.futures import ThreadPoolExecutor

from harness import TRAPS, Sandbox, fix_checksum, run, sections, set_bytes, validate
from test_model_file import MODEL_RULES, payloads, write_file
from test_wire_kmeans import blobs_csv

OTHER_RULES = {"ml.schema.mismatch", "ml.input.unknown-category", "ml.input.non-finite", "ml.input.missing-value", "ml.model.string-table", "ml.limit.string-table"}


class KMeansFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        cls.sb.write("a.csv", blobs_csv(n=45, d=3, label=False))
        rows = ["x,city"] + ["%.2f,%s" % (i * 0.37 % 5, "abc"[i % 3]) for i in range(30)]
        cls.sb.write("c.csv", "\n".join(rows) + "\n")
        for out, args in (("s.cml", ["--k", "3", "--seed", "4", "--step", "scale column=f0 kind=standard;bucket column=f2 kind=width bins=3", "a.csv"]),
                          ("c.cml", ["--k", "2", "--cat", "city", "--step", "encode column=city kind=onehot", "c.csv"])):
            r = cls.sb.run("fit", "--model", "kmeans", "--output", out, *args)
            assert r.status == 0, r
        cls.files = {"s": cls.sb.read("s.cml"), "c": cls.sb.read("c.cml")}
        cls.csv = {"s": "a.csv", "c": "c.csv"}

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

    def test_the_undamaged_files_are_answered(self):
        for key, data in self.files.items():
            r = self.predict(key, data)
            self.assertEqual(r.status, 0, (key, r))

    def test_the_record_header_k_and_the_width(self):
        for key in self.files:
            m = payloads(self.files[key])[4]
            k, d = struct.unpack_from("<QQ", m, 12)
            for at, new in ((0, struct.pack("<I", 4)), (0, struct.pack("<I", 5)), (0, struct.pack("<I", 7)), (0, struct.pack("<I", 0)), (4, struct.pack("<I", 1)), (4, struct.pack("<I", 3)),
                            (8, struct.pack("<I", 1)), (8, struct.pack("<I", 3)), (8, struct.pack("<I", 0)),
                            (12, struct.pack("<Q", 0)), (12, struct.pack("<Q", k + 1)), (12, struct.pack("<Q", k - 1)), (12, struct.pack("<Q", 1025)), (12, struct.pack("<Q", 1 << 62)),
                            (12, struct.pack("<Q", (1 << 63) - 1)), (12, struct.pack("<Q", (1 << 64) - 1)),
                            (20, struct.pack("<Q", 0)), (20, struct.pack("<Q", d + 1)), (20, struct.pack("<Q", d - 1)), (20, struct.pack("<Q", 1 << 62)), (20, struct.pack("<Q", (1 << 64) - 1))):
                self.refused(key, self.edit_model(key, at, new), "%s field at %d = %r" % (key, at, new), ("ml.model.malformed", "ml.model.version"))

    def test_the_length_of_the_payload(self):
        for key in self.files:
            for cut in (1, 7, 8, 9, 16, 28, 32):
                self.refused(key, self.resize_model(key, cut=cut), "%s cut %d" % (key, cut))
            for add in (b"\0", b"\0" * 8, bytes(range(40))):
                self.refused(key, self.resize_model(key, add=add), "%s add %d" % (key, len(add)))
            self.refused(key, self.resize_model(key, cut=len(payloads(self.files[key])[4])), "%s empty model section" % key)

    def test_every_centroid_value_must_be_a_number(self):
        for key in self.files:
            m = payloads(self.files[key])[4]
            k, d = struct.unpack_from("<QQ", m, 12)
            for i in range(k * d):
                for bits in (0x7FF8000000000000, 0x7FF0000000000000, 0xFFF0000000000000, 0xFFFFFFFFFFFFFFFF):
                    self.refused(key, self.edit_model(key, 28 + 8 * i, struct.pack("<Q", bits)), "%s centroid %d = %x" % (key, i, bits))
            # a finite value, however odd, is only a centroid
            r = self.predict(key, self.edit_model(key, 28, struct.pack("<d", 1e300)))
            self.assertEqual(r.status, 0, r)

    def test_the_schema_target_index_and_the_kind_agree(self):
        for key in self.files:
            data = self.files[key]
            sc = sections(data)[1][1]
            cols = struct.unpack_from("<I", data, sc)[0]
            self.assertEqual(struct.unpack_from("<I", data, sc + 4)[0], cols)
            for target in (0, cols - 1, cols + 1, 1 << 31, (1 << 32) - 1):
                self.refused(key, fix_checksum(set_bytes(data, sc + 4, struct.pack("<I", target))), "%s target %d" % (key, target))

    def test_the_plan_and_the_payload_agree_on_k(self):
        for key in self.files:
            secs = payloads(self.files[key])
            plan = secs[1]
            k = struct.unpack_from("<Q", secs[4], 12)[0]
            old = ("k=%d " % k).encode()
            self.assertIn(old, plan)
            for new_k in (1, k + 1, 1024):
                if new_k == k:
                    continue
                s2 = dict(secs)
                s2[1] = plan.replace(old, ("k=%d " % new_k).encode())
                from test_model_file import write_file
                self.refused(key, write_file(s2), "%s plan k=%d over %d centroids" % (key, new_k, k))

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
            tag, start, n = rng.choice([s for s in sections(data) if s[2] > 0] + [sections(data)[3]] * 3 + [sections(data)[1]] * 2)
            for _ in range(rng.randint(1, 4)):
                at = start + rng.randrange(n)
                b[at] = rng.choice((0, 1, 2, 0x7F, 0x80, 0xFF, rng.randrange(256)))
            items.append(("%s-damage-%d" % (key, i), key, fix_checksum(bytes(b)), True))
        self.check_many(items, True)

    def test_seeded_damage_of_the_model_section_by_whole_words(self):
        rng = random.Random(20261018)
        words = [0, 1, 2, 3, -1, -2, 1 << 10, 1 << 11, 1 << 17, 1 << 31, 1 << 32, 1 << 62, (1 << 63) - 1, -(1 << 63)]
        items = []
        for i in range(1000):
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

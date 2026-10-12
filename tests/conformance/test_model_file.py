"""M1 -- the model file (docs/design.md section 4.1, docs/cli.md section 4).

The layout is parsed here by an independent reader written from the document; the reader of the tool is judged by what it does
to damaged files: every truncation, every flipped byte, seeded multi-byte damage, and structural damage whose checksum is
repaired. None may trap; none with a wrong checksum may succeed; each is refused with one of the five model tags."""

import hashlib
import os
import random
import struct
import unittest

from harness import TRAPS, Sandbox, fix_checksum, sections, set_bytes, validate

MODEL_RULES = {"ml.model.bad-magic", "ml.model.version", "ml.model.truncated", "ml.model.checksum", "ml.model.malformed"}
REG = "id,x,y\n1,2.5,10\n2,3.5,20\n3,1.0,30\n4,0.25,41\n"
CLS = "a,b,lab\n1,2,0\n2,3,1\n3,4,1\n4,5,1\n5,6,0\n"


def u32(b, at):
    return struct.unpack_from("<I", b, at)[0]


def i64(b, at):
    return struct.unpack_from("<q", b, at)[0]


def parse(data):
    """An independent reader of CML1, from design.md 4.1: the fields of a model file, checked as it goes."""
    assert data[:4] == b"CML1"
    assert u32(data, 4) == 1
    assert u32(data, 8) == 5
    secs = {}
    pos = 12
    for want in range(1, 6):
        tag = u32(data, pos)
        n = struct.unpack_from("<Q", data, pos + 4)[0]
        assert tag == want
        secs[tag] = data[pos + 12:pos + 12 + n]
        pos += 12 + n
    assert pos == len(data) - 32
    assert hashlib.sha256(data[:-32]).digest() == data[-32:]
    out = {"plan": secs[1].decode()}
    sc = secs[2]
    ncols, target = u32(sc, 0), u32(sc, 4)
    cols = []
    p = 8
    for _ in range(ncols):
        n = u32(sc, p)
        name = sc[p + 4:p + 4 + n].decode()
        ty, scale = u32(sc, p + 4 + n), u32(sc, p + 8 + n)
        cols.append((name, "dec" if ty else "int", scale))
        p += 12 + n
    assert p == len(sc)
    out["columns"], out["target"] = cols, target
    assert secs[3] == b"\0\0\0\0"
    md = secs[4]
    kind, dtype, rank, shape = u32(md, 0), u32(md, 4), u32(md, 8), struct.unpack_from("<Q", md, 12)[0]
    assert (dtype, rank) == (1, 1)
    out["kind"] = kind
    out["stats"] = [i64(md, 20 + 8 * k) for k in range(shape)]
    assert len(md) == 20 + 8 * shape
    me = secs[5]
    out["rows"], out["cols"], out["seed"] = struct.unpack_from("<QQQ", me, 0)
    assert u32(me, 24) == 0
    cl = u32(me, 28)
    assert len(me) == 32 + cl
    out["compiler"] = me[32:].decode()
    return out


class ModelFile(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        self.sb.write("reg.csv", REG)
        self.sb.write("cls.csv", CLS)
        self.assertEqual(self.sb.run("fit", "--target", "y", "--output", "m.cml", "reg.csv").status, 0)
        self.assertEqual(self.sb.run("fit", "--task", "classification", "--target", "lab", "--output", "c.cml", "cls.csv").status, 0)
        self.reg = self.sb.read("m.cml")
        self.cls = self.sb.read("c.cml")

    def tearDown(self):
        self.sb.cleanup()

    def try_model(self, data, verb="predict"):
        self.sb.write("bad.cml", data)
        csv = "reg.csv"
        return self.sb.run(verb, "--model-file", "bad.cml", csv)

    def check_refused(self, data, label, allowed=MODEL_RULES):
        r = self.try_model(data)
        self.assertNotIn(r.status, TRAPS, "%s trapped: %r" % (label, r))
        self.assertEqual(validate(r), [], "%s: %r" % (label, r))
        self.assertEqual(r.status, 8, "%s: %r" % (label, r))
        self.assertIn(r.first_rule(), allowed, "%s: %r" % (label, r))
        return r.first_rule()

    # ------------------------------------------------------------------ the format
    def test_layout_regression(self):
        m = parse(self.reg)
        self.assertEqual(m["plan"], "model kind=constant task=regression\n")
        self.assertEqual(m["columns"], [("id", "int", 0), ("x", "dec", 2), ("y", "int", 0)])
        self.assertEqual(m["target"], 2)
        self.assertEqual(m["kind"], 1)
        n, scale, sq, sr = m["stats"]
        self.assertEqual((n, scale), (4, 0))
        self.assertEqual(sq * 10**9 + sr, 101)
        self.assertEqual((m["rows"], m["cols"], m["seed"]), (4, 3, 0))
        self.assertTrue(m["compiler"].startswith("a4572ea"))

    def test_layout_classification(self):
        m = parse(self.cls)
        self.assertEqual(m["plan"], "model kind=constant task=classification\n")
        self.assertEqual(m["kind"], 2)
        self.assertEqual(m["stats"], [5, 2, 0, 1, 2, 3])

    def test_the_file_has_no_timestamp_host_or_path(self):
        data = self.reg
        for needle in (str(self.sb.dir).encode(), b"/tmp", b"/var", os.uname().nodename.encode()):
            self.assertNotIn(needle, data)

    def test_sha256_trailer_is_the_hash_of_everything_before(self):
        self.assertEqual(hashlib.sha256(self.reg[:-32]).digest(), self.reg[-32:])

    def test_the_reported_hash_is_the_files(self):
        r = self.sb.run("fit", "--overwrite", "--target", "y", "--output", "m.cml", "reg.csv")
        self.assertEqual(r.data()["sha256"], hashlib.sha256(self.sb.read("m.cml")).hexdigest())
        self.assertEqual(r.data()["bytes"], len(self.sb.read("m.cml")))

    # ------------------------------------------------------------------ refusals by name
    def test_bad_magic(self):
        self.assertEqual(self.check_refused(b"XXXX" + self.reg[4:], "magic"), "ml.model.bad-magic")
        self.assertEqual(self.check_refused(b"CML2" + self.reg[4:], "magic2"), "ml.model.bad-magic")
        self.assertEqual(self.check_refused(b"cml1" + self.reg[4:], "lower"), "ml.model.bad-magic")

    def test_version(self):
        for v in (0, 2, 0xFFFFFFFF):
            self.assertEqual(self.check_refused(set_bytes(self.reg, 4, v.to_bytes(4, "little")), "version %d" % v), "ml.model.version")

    def test_truncated_at_every_length(self):
        for n in range(len(self.reg)):
            got = self.check_refused(self.reg[:n], "length %d" % n)
            self.assertEqual(got, "ml.model.truncated", "a %d-byte prefix of a %d-byte file" % (n, len(self.reg)))

    def test_appended_bytes_are_refused(self):
        for extra in (b"\0", b"x" * 31, b"y" * 32, b"z" * 1000):
            self.check_refused(self.reg + extra, "append %d" % len(extra))

    def test_checksum_wrong(self):
        for s_tag, start, n in sections(self.reg):
            if n == 0:
                continue
            for at in (start, start + n - 1):
                b = bytearray(self.reg)
                b[at] ^= 0x01
                got = self.check_refused(bytes(b), "section %d byte %d" % (s_tag, at))
                self.assertEqual(got, "ml.model.checksum")
        b = bytearray(self.reg)
        b[-1] ^= 0xFF
        self.assertEqual(self.check_refused(bytes(b), "last byte"), "ml.model.checksum")

    # ------------------------------------------------------------------ the fuzz
    def test_every_single_byte_flip_is_refused_never_accepted(self):
        for model in (self.reg, self.cls):
            for at in range(len(model)):
                for mask in (0x01, 0x80, 0xFF):
                    b = bytearray(model)
                    b[at] ^= mask
                    self.check_refused(bytes(b), "flip %#x at %d" % (mask, at))

    def test_seeded_multi_byte_damage(self):
        rng = random.Random(20261012)
        for i in range(400):
            b = bytearray(rng.choice((self.reg, self.cls)))
            for _ in range(rng.randint(1, 8)):
                kind = rng.randint(0, 3)
                at = rng.randrange(len(b))
                if kind == 0:
                    b[at] = rng.randrange(256)
                elif kind == 1:
                    del b[at:at + rng.randint(1, 20)]
                elif kind == 2:
                    b[at:at] = bytes(rng.randrange(256) for _ in range(rng.randint(1, 20)))
                else:
                    b[at:at + 8] = b"\xff" * 8
                if not b:
                    b = bytearray(b"C")
            self.check_refused(bytes(b), "damage %d" % i)

    def test_structural_damage_with_a_repaired_checksum_is_refused_or_valid_never_a_trap(self):
        rng = random.Random(7)
        outcomes = {"ok": 0, "refused": 0}
        for i in range(600):
            model = rng.choice((self.reg, self.cls))
            b = bytearray(model)
            for _ in range(rng.randint(1, 3)):
                tag, start, n = rng.choice([s for s in sections(model) if s[2] > 0])
                at = start + rng.randrange(n)
                b[at] = rng.choice((0, 1, 2, 0x7F, 0x80, 0xFF, rng.randrange(256)))
            data = fix_checksum(bytes(b))
            r = self.try_model(data)
            self.assertNotIn(r.status, TRAPS, "damage %d trapped: %r" % (i, r))
            self.assertEqual(validate(r), [], "damage %d: %r" % (i, r))
            if r.status == 0:
                outcomes["ok"] += 1
            else:
                self.assertEqual(r.status, 8)
                # a repaired file can be a different, valid model (a renamed column): then the input no longer fits it
                self.assertIn(r.first_rule(), MODEL_RULES | {"ml.schema.mismatch"}, repr(r))
                outcomes["refused"] += 1
        self.assertGreater(outcomes["refused"], 100)

    def test_every_field_check_has_a_file_that_reaches_it(self):
        reg_secs = sections(self.reg)
        cls_secs = sections(self.cls)

        def edit(model, secs, tag, offset, value, size=4):
            start = secs[tag - 1][1]
            return fix_checksum(set_bytes(model, start + offset, value.to_bytes(size, "little")))

        cases = [
            ("plan text", fix_checksum(set_bytes(self.reg, reg_secs[0][1], b"nodel"))),
            ("zero columns", edit(self.reg, reg_secs, 2, 0, 0)),
            ("too many columns", edit(self.reg, reg_secs, 2, 0, 9000)),
            ("more columns than the section holds", edit(self.reg, reg_secs, 2, 0, 50)),
            ("target past the columns", edit(self.reg, reg_secs, 2, 4, 3)),
            ("name length past the section", edit(self.reg, reg_secs, 2, 8, 1000)),
            ("name length makes fields overlap", edit(self.reg, reg_secs, 2, 8, 0xFFFFFFFF)),
            ("type 5", edit(self.reg, reg_secs, 2, 8 + 4 + 2, 5)),
            ("scale 13", edit(self.reg, reg_secs, 2, 8 + 4 + 2 + 4, 13)),
            ("int with a scale", edit(self.reg, reg_secs, 2, 8 + 4 + 2 + 4, 1)),
            ("steps not empty", edit(self.reg, reg_secs, 3, 0, 1)),
            ("kind 7", edit(self.reg, reg_secs, 4, 0, 7)),
            ("kind disagrees with the plan", edit(self.reg, reg_secs, 4, 0, 2)),
            ("dtype 2", edit(self.reg, reg_secs, 4, 4, 2)),
            ("rank 2", edit(self.reg, reg_secs, 4, 8, 2)),
            ("shape 5", edit(self.reg, reg_secs, 4, 12, 5, 8)),
            ("n = 0", edit(self.reg, reg_secs, 4, 20, 0, 8)),
            ("n negative", edit(self.reg, reg_secs, 4, 20, (1 << 64) - 5, 8)),
            ("n over the ceiling", edit(self.reg, reg_secs, 4, 20, 2_000_000_000, 8)),
            ("scale disagrees with the schema", edit(self.reg, reg_secs, 4, 28, 3, 8)),
            ("sum of quotients out of bounds", edit(self.reg, reg_secs, 4, 36, 10**15, 8)),
            ("sum of remainders out of bounds", edit(self.reg, reg_secs, 4, 44, (1 << 64) - 10**15, 8)),
            ("classes not ordered", edit(self.cls, cls_secs, 4, 20 + 8 * 2, 5, 8)),
            ("three classes", edit(self.cls, cls_secs, 4, 20 + 8, 3, 8)),
            ("class count 0", edit(self.cls, cls_secs, 4, 20 + 16 + 24, 0, 8)),
            ("counts do not add up", edit(self.cls, cls_secs, 4, 20 + 8 * 4, 9, 8)),
            ("rows disagree with n", edit(self.reg, reg_secs, 5, 0, 5, 8)),
            ("columns disagree with the schema", edit(self.reg, reg_secs, 5, 8, 4, 8)),
            ("hyperparameters", edit(self.reg, reg_secs, 5, 24, 1)),
            ("compiler length past the section", edit(self.reg, reg_secs, 5, 28, 300)),
            ("compiler length off by one", edit(self.reg, reg_secs, 5, 28, 5)),
        ]
        for label, data in cases:
            got = self.check_refused(data, label)
            self.assertEqual(got, "ml.model.malformed", label)

    def test_section_tags_out_of_order_and_wrong_count(self):
        secs = sections(self.reg)
        swapped = set_bytes(self.reg, 12, (2).to_bytes(4, "little"))
        self.assertEqual(self.check_refused(fix_checksum(swapped), "tag 2 first"), "ml.model.malformed")
        four = set_bytes(self.reg, 8, (4).to_bytes(4, "little"))
        self.assertEqual(self.check_refused(fix_checksum(four), "four sections"), "ml.model.malformed")
        many = set_bytes(self.reg, 8, (0xFFFFFFFF).to_bytes(4, "little"))
        self.assertEqual(self.check_refused(fix_checksum(many), "4e9 sections"), "ml.model.malformed")

    def test_bytes_between_the_last_section_and_the_trailer_are_refused(self):
        # a crafted file: the sections, junk, and the SHA-256 of all of it. Appended bytes after the trailer move the trailer and
        # fail the checksum; these pass it, and only the structure check refuses them (mutant m1-trailing-bytes-accepted)
        for junk in (b"x", b"junk" * 10):
            body = self.reg[:-32] + junk
            data = body + hashlib.sha256(body).digest()
            self.assertEqual(self.check_refused(data, "junk %d" % len(junk)), "ml.model.malformed")

    def test_a_huge_declared_length_does_not_allocate(self):
        # a section length of 2^62 is refused as truncated before anything is sized by it
        big = set_bytes(self.reg, 12 + 4, (1 << 62).to_bytes(8, "little"))
        self.assertEqual(self.check_refused(fix_checksum(big), "2^62"), "ml.model.truncated")
        neg = set_bytes(self.reg, 12 + 4, (1 << 63).to_bytes(8, "little"))
        self.assertEqual(self.check_refused(fix_checksum(neg), "2^63"), "ml.model.truncated")

    def test_eval_reads_the_same_model(self):
        for data in (self.reg[:100], b"junk", self.reg[:-1]):
            self.sb.write("bad.cml", data)
            r = self.sb.run("eval", "--model-file", "bad.cml", "reg.csv")
            self.assertIn(r.first_rule(), MODEL_RULES, r)

    def test_a_model_for_the_other_task_is_not_a_corruption(self):
        r = self.sb.run("predict", "--model-file", "c.cml", "cls.csv")
        self.assertEqual(r.status, 0, r)


if __name__ == "__main__":
    unittest.main()

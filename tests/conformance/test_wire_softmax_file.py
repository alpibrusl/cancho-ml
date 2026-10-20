"""M8 -- a damaged file of kind 9 never traps, and the payload is read as strictly as docs/multiclass.md section 3.3 says.

Every truncation and every flipped byte of two files (K = 3 with a stored scaling step, K = 7 with a bucketed column) is refused with a tag, with the checksum wrong; seeded damage
of the model section with the checksum repaired is refused with a tag or answered; the huge counts a hostile file would put in a record header (2^63 - 1, and the pair that wraps the
size of the payload) are refused before any allocation; and the fields of the payload that a single edit can break one rule at a time (the kind, the record headers, the extents, the
coefficients, the labels, the length, the plan that names another model) are each ml.model.malformed."""

import random
import struct
import unittest
from concurrent.futures import ThreadPoolExecutor

from harness import TRAPS, Sandbox, fix_checksum, run, sections, set_bytes, validate
from test_model_file import MODEL_RULES, payloads, write_file
from test_wire_softmax import mc_csv

# a coefficient of 1e300 is a valid number whose scores overflow: ml.predict.non-finite is the answer, not a fault of the reader
OTHER_RULES = {"ml.schema.mismatch", "ml.input.unknown-category", "ml.input.non-finite", "ml.predict.non-finite"}
NAN, INF = 0x7FF8000000000000, 0x7FF0000000000000


def i64(b, at):
    return struct.unpack_from("<q", b, at)[0]


class Layout:
    """the offsets of the payload of K classes and d features: kind | coef record (header 24, then K - 1 d floats) | intercept record (header 16, K - 1 floats) | labels record (header 16, K ints)"""

    def __init__(self, kk, d):
        self.kk, self.d, self.kc = kk, d, kk - 1
        self.coef_head = 4
        self.coef = 28
        self.icp_head = self.coef + 8 * self.kc * d
        self.icp = self.icp_head + 16
        self.lab_head = self.icp + 8 * self.kc
        self.lab = self.lab_head + 16
        self.end = self.lab + 8 * kk


class SoftmaxFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        cls.sb.write("c3.csv", mc_csv(n=60))
        r = random.Random(9)
        rows = ["a,b,c,y"]
        for i in range(90):
            rows.append("%.4f,%.4f,%d,%d" % (r.gauss(0, 1), r.gauss(0, 1), r.randint(1, 3), (-2, 4, 5, 8, 20, 21, 40)[i % 7]))
        cls.sb.write("c7.csv", "\n".join(rows) + "\n")
        cls.cases = {}
        for key, csv, steps in (("k3", "c3.csv", "scale column=a kind=standard"), ("k7", "c7.csv", "bucket column=c kind=width bins=3")):
            args = ["fit", "--model", "softmax", "--target", "y", "--output", key + ".cml", "--l2", "0.5", "--iterations", "6", "--step", steps]
            r_ = cls.sb.run(*args, csv)
            assert r_.status == 0, r_
            cls.cases[key] = (csv, cls.sb.read(key + ".cml"))
        cls.files = {k: v[1] for k, v in cls.cases.items()}
        cls.csv = {k: v[0] for k, v in cls.cases.items()}
        cls.lay = {}
        for key, data in cls.files.items():
            m = payloads(data)[4]
            kc, d = i64(m, 12), i64(m, 20)
            cls.lay[key] = Layout(kc + 1, d)
            assert len(m) == cls.lay[key].end, (key, len(m), cls.lay[key].end)

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

    def test_the_layout_is_the_one_the_tests_assume(self):
        for key, data in self.files.items():
            m = payloads(data)[4]
            L = self.lay[key]
            self.assertEqual(struct.unpack_from("<I", m, 0)[0], 9)
            self.assertEqual(struct.unpack_from("<IIQQ", m, L.coef_head), (2, 2, L.kc, L.d))
            self.assertEqual(struct.unpack_from("<IIQ", m, L.icp_head), (2, 1, L.kc))
            self.assertEqual(struct.unpack_from("<IIQ", m, L.lab_head), (1, 1, L.kk))

    def test_the_kind(self):
        for key in self.files:
            for k in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 1 << 31, 0xFFFFFFFF):
                self.refused(key, self.edit_model(key, 0, struct.pack("<I", k)), "%s kind %d" % (key, k), ("ml.model.malformed", "ml.model.version"))

    def test_the_record_headers(self):
        for key in self.files:
            L = self.lay[key]
            for at, vals in ((L.coef_head, (0, 1, 3, 9, 1 << 31)), (L.coef_head + 4, (0, 1, 3, 1 << 31)), (L.icp_head, (0, 1, 3, 1 << 31)), (L.icp_head + 4, (0, 2, 3)),
                             (L.lab_head, (0, 2, 3, 1 << 31)), (L.lab_head + 4, (0, 2, 3))):
                for v in vals:
                    cur = struct.unpack_from("<I", payloads(self.files[key])[4], at)[0]
                    if v != cur:
                        self.refused(key, self.edit_model(key, at, struct.pack("<I", v)), "%s record header at %d = %d" % (key, at, v))

    def test_the_extents(self):
        for key in self.files:
            L = self.lay[key]
            kc, d, kk = L.kc, L.d, L.kk
            for at, cur, vals in ((L.coef_head + 8, kc, (0, 1, kc - 1, kc + 1, 63, 64, 1 << 40, (1 << 63) - 1, -1)),
                                  (L.coef_head + 16, d, (0, d - 1, d + 1, 257, 1 << 40, (1 << 63) - 1, -1)),
                                  (L.icp_head + 8, kc, (0, 1, kc - 1, kc + 1, 1 << 40, (1 << 63) - 1, -1)),
                                  (L.lab_head + 8, kk, (0, 1, 2, kk - 1, kk + 1, 65, 1 << 40, (1 << 63) - 1, -1))):
                for v in vals:
                    if v != cur:
                        self.refused(key, self.edit_model(key, at, struct.pack("<q", v)), "%s extent at %d = %d" % (key, at, v))

    def test_counts_that_wrap_the_size_are_refused_before_any_allocation(self):
        # kc * d is multiplied: kc = 2^62 and d = 4 wrap to 0, so only the checks before the multiplication stand between such a header and an allocation
        for key in self.files:
            L = self.lay[key]
            for kc, d in ((1 << 62, 4), (1 << 61, 8), (3, 1 << 62), (1 << 32, 1 << 32), ((1 << 63) - 1, 2)):
                data = self.edit_model(key, L.coef_head + 8, struct.pack("<qq", kc, d))
                self.refused(key, data, "%s wrapped extents %d x %d" % (key, kc, d))

    def test_the_coefficients(self):
        for key in self.files:
            L = self.lay[key]
            for at in (L.coef, L.coef + 8 * (L.kc * L.d - 1), L.icp, L.icp + 8 * (L.kc - 1)):
                for bits in (NAN, INF, 0xFFF0000000000000, 0xFFF8000000000001):
                    self.refused(key, self.edit_model(key, at, struct.pack("<Q", bits)), "%s coefficient at %d = %x" % (key, at, bits))
            # a finite number of any size is a coefficient: the file is a valid model with other answers
            r = self.predict(key, self.edit_model(key, L.coef, struct.pack("<d", 1e30)))
            self.assertEqual(r.status, 0, (key, r))

    def test_the_labels(self):
        for key in self.files:
            L = self.lay[key]
            labels = list(struct.unpack_from("<%dq" % L.kk, payloads(self.files[key])[4], L.lab))
            for i in range(L.kk - 1):
                # equal to the next, above the next, equal to the one before
                self.refused(key, self.edit_model(key, L.lab + 8 * i, struct.pack("<q", labels[i + 1])), "%s label %d equal to the next" % (key, i))
                self.refused(key, self.edit_model(key, L.lab + 8 * i, struct.pack("<q", labels[i + 1] + 1)), "%s label %d above the next" % (key, i))
            self.refused(key, self.edit_model(key, L.lab + 8 * (L.kk - 1), struct.pack("<q", labels[-2])), "%s the last label equal to the one before" % key)
            self.refused(key, self.edit_model(key, L.lab, struct.pack("<q", -(10 ** 12) - 1)), "%s a label of 13 digits" % key)
            self.refused(key, self.edit_model(key, L.lab + 8 * (L.kk - 1), struct.pack("<q", 10 ** 12 + 1)), "%s a label of 13 digits" % key)
            self.refused(key, self.edit_model(key, L.lab + 8 * (L.kk - 1), struct.pack("<q", (1 << 63) - 1)), "%s the largest int" % key)
            self.refused(key, self.edit_model(key, L.lab, struct.pack("<q", -(1 << 63))), "%s the smallest int" % key)
            # other labels in ascending order are another valid model
            new = [-50 + 3 * i for i in range(L.kk)]
            r = self.predict(key, self.edit_model(key, L.lab, struct.pack("<%dq" % L.kk, *new)))
            self.assertEqual(r.status, 0, (key, r))
            self.assertTrue({int(row[0]) for row in r.data()["rows"]} <= set(new))

    def test_the_length_of_the_payload(self):
        for key in self.files:
            for cut in (1, 8, 16, 31, 32, 33, 100):
                self.refused(key, self.resize_model(key, cut=cut), "%s cut %d" % (key, cut), ("ml.model.malformed", "ml.model.truncated"))
            for add in (b"\0", b"\0" * 8, b"\0" * 31, b"\0" * 33, bytes(range(40))):
                self.refused(key, self.resize_model(key, add=add), "%s add %d" % (key, len(add)))
            m = payloads(self.files[key])[4]
            for keep in (0, 3, 4, 20, 27, 28, 100):
                if keep < len(m):
                    self.refused(key, self.resize_model(key, cut=len(m) - keep), "%s kept %d bytes" % (key, keep), ("ml.model.malformed", "ml.model.truncated"))

    def test_the_plan_and_the_payload_agree(self):
        for key in self.files:
            secs = payloads(self.files[key])
            self.assertIn(b"kind=softmax", secs[1])
            for new in (b"kind=logistic", b"kind=linear", b"kind=tree", b"kind=boost", b"kind=forest", b"kind=kmeans"):
                s2 = dict(secs)
                kw = b" iterations=6 l2=0.5" if new in (b"kind=logistic",) else b" task=classification" if new in (b"kind=tree", b"kind=forest", b"kind=boost") else b""
                if new == b"kind=kmeans":
                    kw = b" k=2"
                head = secs[1][:secs[1].index(b"kind=softmax")]
                s2[1] = head + new + kw + b"\n"
                self.refused(key, write_file(s2), "%s plan %s over the payload" % (key, new.decode()), ("ml.model.malformed", "ml.model.version", "ml.model.truncated"))
        # and the other way: a plan that says softmax over the payload of a logistic model (kind 5) or of a tree (kind 7)
        self.sb.write("lg.csv", "a,y\n1,0\n2,1\n3,1\n4,0\n5,1\n6,0\n7,1\n8,0\n")
        for model, extra, tag in (("logistic", ["--l2", "1"], b"kind=logistic"), ("tree", ["--task", "classification"], b"kind=tree")):
            r = self.sb.run("fit", "--model", model, "--target", "y", "--output", "lg.cml", "--overwrite", *extra, "lg.csv")
            self.assertEqual(r.status, 0, r)
            secs = payloads(self.sb.read("lg.cml"))
            line = secs[1].index(tag)
            secs[1] = secs[1][:line] + b"kind=softmax iterations=6 l2=0.5\n"
            self.sb.write("e.cml", write_file(secs))
            r = self.sb.run("predict", "--model-file", "e.cml", "lg.csv")
            self.assertIn(r.first_rule(), ("ml.model.malformed", "ml.model.version"), (model, r))

    def test_the_sum_order_of_the_meta(self):
        for key in self.files:
            secs = payloads(self.files[key])
            meta = bytearray(secs[5])
            for order in (0, 1, 3, 4, 1 << 20):
                meta[-4:] = struct.pack("<I", order)
                s2 = dict(secs)
                s2[5] = bytes(meta)
                self.refused(key, write_file(s2), "%s sum order %d" % (key, order), ("ml.model.version", "ml.model.malformed"))

    def test_a_score_that_overflows_is_refused_not_printed(self):
        # coefficients of 1e308 and a feature of 10: the scores are infinite and the probabilities are not numbers (docs/nonfinite.md)
        key = "k3"
        L = self.lay[key]
        data = self.edit_model(key, L.coef, struct.pack("<d", 1.0e308))
        self.sb.write("big.csv", "a,b,y\n1000.0,1000.0,3\n2.0,2.0,5\n")
        self.sb.write("o.cml", data)
        for verb in (["predict", "--proba"], ["predict"], ["eval"]):
            r = self.sb.run(*verb, "--model-file", "o.cml", "big.csv")
            self.assertNotIn(r.status, TRAPS, r)
            self.assertEqual(validate(r), [], r)
            self.assertIn(r.first_rule(), (None, "ml.predict.non-finite", "ml.metric.undefined", "ml.input.non-finite"), r)
            if r.status != 0:
                self.assertEqual(r.status, 8, r)

    def test_a_file_over_the_parameter_bound_is_refused_by_the_reader(self):
        # ml fit refuses K = 33 with 64 features (2080 parameters), so only a hand-made file reaches the reader with them: written here by the layout of design.md 4.1 (the oracle of
        # scripts/oracle_wire.py writes the same bytes but needs numpy, which the fast jobs of the workflow do not install)
        import hashlib

        def put_str(t):
            b = t.encode()
            return struct.pack("<I", len(b)) + b

        def section(tag, payload):
            return struct.pack("<IQ", tag, len(payload)) + payload

        for kk, d, ok in ((33, 64, False), (32, 64, True), (3, 256, True), (3, 257, False)):
            names = ["f%d" % j for j in range(d)] + ["y"]
            schema = struct.pack("<II", len(names), d) + b"".join(put_str(n) + struct.pack("<II", t, sc) for n, (t, sc) in zip(names, [(1, 1)] * d + [(0, 0)]))
            kc = kk - 1
            payload = (struct.pack("<I", 9) + struct.pack("<IIQQ", 2, 2, kc, d) + bytes(8 * kc * d) + struct.pack("<IIQ", 2, 1, kc) + bytes(8 * kc)
                       + struct.pack("<IIQ", 1, 1, kk) + struct.pack("<%dq" % kk, *range(kk)))
            old = payloads(self.files["k3"])[5]
            cl = struct.unpack_from("<I", old, 28)[0]
            pin = struct.pack("<QQQII", 10, d + 1, 0, 0, cl) + old[32:32 + cl] + struct.pack("<I", 2)
            body = (b"CML1" + struct.pack("<II", 1, 5) + section(1, b"model kind=softmax iterations=1 l2=1.0\n") + section(2, schema) + section(3, struct.pack("<I", 0))
                    + section(4, payload) + section(5, pin))
            self.sb.write("big.cml", body + hashlib.sha256(body).digest())
            self.sb.write("big.csv", ",".join(names[:-1]) + "\n" + ",".join("1.0" for _ in range(d)) + "\n")
            r = self.sb.run("predict", "--model-file", "big.cml", "big.csv")
            self.assertNotIn(r.status, TRAPS, (kk, d, r))
            if ok:
                self.assertEqual(r.status, 0, (kk, d, r))
            else:
                self.assertIn(r.first_rule(), ("ml.model.malformed",), (kk, d, r))

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

    def test_seeded_damage_of_the_model_section_with_the_checksum_repaired(self):
        rng = random.Random(20261020)
        items = []
        for key, data in self.files.items():
            start, n = sections(data)[3][1], sections(data)[3][2]
            for i in range(400):
                b = bytearray(data)
                for _ in range(rng.randint(1, 4)):
                    at = start + rng.randrange(n)
                    b[at] = rng.choice([0, 0xFF, rng.randrange(256), b[at] ^ (1 << rng.randrange(8))])
                items.append(("%s-seed-%d" % (key, i), key, fix_checksum(bytes(b)), True))
        self.check_many(items, True)


if __name__ == "__main__":
    unittest.main()

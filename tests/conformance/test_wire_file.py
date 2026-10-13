"""W2, W9 -- one file, stored steps, nothing to skew; and a damaged file of a planned model never traps.

The file of a planned model holds the plan, the schema, the fitted steps and the estimator. `predict` and `eval` take no plan flag; the stored parameters are what
they apply (editing one, with the checksum repaired, changes the answer); the text of the plan is a description, and a plan that disagrees with the steps is
malformed. Damaged files (every truncation, every flipped byte, seeded damage of each section with and without a repaired checksum) are refused with a tag."""

import random
import struct
import unittest
from concurrent.futures import ThreadPoolExecutor

from harness import TRAPS, Sandbox, fix_checksum, run, sections, set_bytes, validate
from test_model_file import MODEL_RULES, payloads, write_file

LIN = "a,b,y\n" + "".join("%d,%s,%.3f\n" % (i, "" if i % 5 == 2 else str((i * 7) % 11), 1.5 * i - 0.25 * ((i * 7) % 11) + 0.01 * ((i * 13) % 5)) for i in range(1, 41))
CLS = "a,b,lab\n" + "".join("%d,%d,%d\n" % (i, (i * 7) % 11, 1 if (i + 3 * ((i * 7) % 11) + (i * 5) % 4) % 9 > 4 else 0) for i in range(1, 41))
FIT_LIN = ["fit", "--model", "linear", "--l2", "0.5", "--step", "impute column=b strategy=mean;scale column=a,b kind=standard", "--target", "y", "--output", "m.cml", "lin.csv"]
FIT_LOG = ["fit", "--model", "logistic", "--iterations", "30", "--l2", "0.5", "--step", "scale column=* kind=minmax", "--target", "lab", "--output", "c.cml", "cls.csv"]


def predict_rows(sb, model, csv="lin.csv", *extra):
    r = sb.run("predict", "--model-file", model, "--limit", "1000", *extra, csv)
    return r


class WireFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        cls.sb.write("lin.csv", LIN)
        cls.sb.write("cls.csv", CLS)
        assert cls.sb.run(*FIT_LIN).status == 0
        assert cls.sb.run(*FIT_LOG).status == 0
        cls.lin = cls.sb.read("m.cml")
        cls.log = cls.sb.read("c.cml")

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    # ------------------------------------------------------------------ one file
    def test_the_file_holds_the_plan_the_schema_the_steps_and_the_estimator(self):
        p = payloads(self.lin)
        self.assertEqual(p[1], b"impute column=b strategy=mean\nscale column=a,b kind=standard ddof=0\nmodel kind=linear l2=0.5\n")
        self.assertEqual(struct.unpack_from("<I", p[2], 0)[0], 3)                     # a, b, y
        self.assertEqual(struct.unpack_from("<I", p[3], 0)[0], 2)                     # two steps
        self.assertEqual(struct.unpack_from("<I", p[4], 0)[0], 4)                     # kind 4
        meta = p[5]
        self.assertEqual(struct.unpack_from("<I", meta, len(meta) - 4)[0], 2)         # the sum order
        pl = payloads(self.log)
        self.assertEqual(struct.unpack_from("<I", pl[4], 0)[0], 5)
        # kind, coef (record of 2 features), intercept, labels: 4 + (8 + 8 + 2 * 8) + (8 + 8 + 8) + 32
        self.assertEqual(len(pl[4]), 4 + 32 + 24 + 32)

    def test_a_file_without_steps_is_still_one_file(self):
        sb = Sandbox()
        try:
            sb.write("lin.csv", LIN.replace(",,", ",1,"))
            r = sb.run("fit", "--model", "linear", "--target", "y", "--output", "m.cml", "lin.csv")
            self.assertEqual(r.status, 0, r)
            self.assertEqual(struct.unpack_from("<I", payloads(sb.read("m.cml"))[3], 0)[0], 0)
        finally:
            sb.cleanup()

    # ------------------------------------------------------------------ no plan at predict or eval
    def test_predict_and_eval_take_no_plan_step_or_model_flag(self):
        for flag, value in (("--plan", "p.txt"), ("--step", "scale column=a kind=minmax"), ("--model", "linear"), ("--l2", "1"), ("--iterations", "3"), ("--task", "regression")):
            self.sb.write("p.txt", "model kind=linear\n")
            for verb in ("predict", "eval"):
                r = self.sb.run(verb, "--model-file", "m.cml", flag, value, "lin.csv")
                self.assertEqual(r.first_rule(), "args.conflict", (verb, flag, r))

    def test_the_flags_of_predict_and_eval_in_the_tool_do_not_include_a_plan(self):
        from harness import spec
        allowed = spec.tables()["allowed"]
        for verb in ("predict", "eval"):
            for flag in ("plan", "step", "model", "l2", "iterations", "task", "target", "output"):
                self.assertNotIn(flag, allowed[verb], (verb, flag))

    # ------------------------------------------------------------------ the stored parameters are what is applied
    def sealed_with(self, data, tag, at, new):
        """the file with `new` written at offset `at` of the payload of section `tag`, the checksum repaired"""
        start = [s for s in sections(data) if s[0] == tag][0][1]
        return fix_checksum(set_bytes(data, start + at, new))

    def test_editing_a_stored_step_parameter_with_a_repaired_checksum_changes_the_answer(self):
        base = predict_rows(self.sb, "m.cml")
        self.assertEqual(base.status, 0, base)
        # the first step is the impute (kind 1): u32 kind, ints record (u32 dtype, u32 rank, u64 shape, ints), floats record. Its fill is the last 8 bytes of the step
        steps = payloads(self.lin)[3]
        first_len = struct.unpack_from("<Q", steps, 4)[0]
        fill_at = 4 + 8 + first_len - 8
        edited = self.sealed_with(self.lin, 3, fill_at, struct.pack("<d", 1000.0))
        self.sb.write("e.cml", edited)
        r = predict_rows(self.sb, "e.cml")
        self.assertEqual(r.status, 0, r)
        self.assertNotEqual(r.data()["rows"], base.data()["rows"], "the stored fill was not used")
        # without the repair, the checksum catches it
        broken = set_bytes(self.lin, [s for s in sections(self.lin) if s[0] == 3][0][1] + fill_at, struct.pack("<d", 1000.0))
        self.sb.write("e2.cml", broken)
        self.assertEqual(predict_rows(self.sb, "e2.cml").first_rule(), "ml.model.checksum")

    def test_editing_a_stored_divisor_changes_the_answer_and_a_bad_one_is_malformed(self):
        steps = payloads(self.lin)[3]
        first_len = struct.unpack_from("<Q", steps, 4)[0]
        second = 4 + 8 + first_len
        second_len = struct.unpack_from("<Q", steps, second)[0]
        # the last 8 bytes of the standard scaler are the divisor of its last column
        d_at = second + 8 + second_len - 8
        base = predict_rows(self.sb, "m.cml").data()["rows"]
        self.sb.write("e.cml", self.sealed_with(self.lin, 3, d_at, struct.pack("<d", 2.0 * struct.unpack_from("<d", steps, d_at)[0])))
        self.assertNotEqual(predict_rows(self.sb, "e.cml").data()["rows"], base)
        for bad in (0.0, -1.0):
            self.sb.write("e.cml", self.sealed_with(self.lin, 3, d_at, struct.pack("<d", bad)))
            self.assertEqual(predict_rows(self.sb, "e.cml").first_rule(), "ml.model.malformed", bad)

    def test_the_plan_text_is_a_description_and_must_agree_with_the_steps(self):
        base = predict_rows(self.sb, "m.cml").data()["rows"]
        plan_at = 0
        # l2=0.5 -> l2=0.7 (same length): a description of how the model was fitted, no step disagrees; the answers are those of the stored parameters
        text = payloads(self.lin)[1]
        i = text.index(b"l2=0.5") + 3
        self.sb.write("e.cml", self.sealed_with(self.lin, 1, i, b"7"))
        r = predict_rows(self.sb, "e.cml")
        self.assertEqual(r.status, 0, r)
        self.assertEqual(r.data()["rows"], base)
        # strategy=mean -> strategy=constant would change the length; median is 6 letters, mean is 4: swap the two steps' kinds instead, which disagrees with the steps
        j = text.index(b"mean")
        for new in (b"mode",):
            self.sb.write("e.cml", self.sealed_with(self.lin, 1, j, new))
            self.assertEqual(predict_rows(self.sb, "e.cml").first_rule(), "ml.model.malformed", new)
        # a plan whose canonical form is not itself (a blank doubled) is malformed too
        self.sb.write("e.cml", self.sealed_with(self.lin, 1, text.index(b" column=b"), b"  "))
        self.assertEqual(predict_rows(self.sb, "e.cml").first_rule(), "ml.model.malformed")

    def test_the_estimator_payload_is_checked(self):
        pm = payloads(self.lin)[4]
        for at, new in ((0, struct.pack("<I", 5)), (4, struct.pack("<I", 1)), (8, struct.pack("<I", 2)), (12, struct.pack("<Q", 99)), (len(pm) - 8, struct.pack("<d", float("inf")))):
            self.sb.write("e.cml", self.sealed_with(self.lin, 4, at, new))
            r = predict_rows(self.sb, "e.cml")
            self.assertIn(r.status, (8,), (at, r))
            self.assertIn(r.first_rule(), ("ml.model.malformed",), (at, r))
        # the labels record of a logistic model: equal labels, a wrong dtype, a wrong length
        pl = payloads(self.log)[4]
        lab = len(pl) - 32
        for at, new in ((lab, struct.pack("<I", 2)), (lab + 8, struct.pack("<Q", 3)), (lab + 24, pl[lab + 16:lab + 24])):
            self.sb.write("e.cml", self.sealed_with(self.log, 4, at, new))
            r = self.sb.run("predict", "--model-file", "e.cml", "cls.csv")
            self.assertEqual(r.first_rule(), "ml.model.malformed", (at, r))

    def test_the_schema_section_is_checked(self):
        sc = payloads(self.lin)[2]
        for at, new in ((0, struct.pack("<I", 0)), (0, struct.pack("<I", 99999)), (4, struct.pack("<I", 9)), (8, struct.pack("<I", 1 << 30))):
            self.sb.write("e.cml", self.sealed_with(self.lin, 2, at, new))
            r = predict_rows(self.sb, "e.cml")
            self.assertEqual(r.first_rule(), "ml.model.malformed", (at, r))

    # ------------------------------------------------------------------ damaged files never trap (W9)
    def check_many(self, items, accept_ok):
        """items: [(label, bytes, repaired)]; every one is answered with a rule tag or (repaired only) a valid answer; none traps"""
        sb = self.sb

        def one(item):
            label, data, repaired = item
            name = "w-%d.cml" % (abs(hash(label)) % 10 ** 9)
            sb.write(name, data)
            r = run("--root", sb.dir, "predict", "--model-file", name, "--limit", "3", "lin.csv" if label.startswith("lin") else "cls.csv")
            return label, repaired, r

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
                else:
                    if r.first_rule() not in MODEL_RULES | {"ml.schema.mismatch", "ml.input.unknown-category", "ml.input.missing-value", "ml.fit.non-finite"}:
                        bad.append((label, r.first_rule()))
        self.assertEqual(bad, [])

    def test_every_truncation_is_refused(self):
        items = []
        for name, data in (("lin", self.lin), ("log", self.log)):
            for cut in range(len(data)):
                items.append(("%s-cut-%d" % (name, cut), data[:cut], False))
        self.check_many(items, False)

    def test_every_flipped_byte_is_refused(self):
        items = []
        for name, data in (("lin", self.lin), ("log", self.log)):
            for at in range(len(data)):
                for mask in (0x01, 0x80):
                    b = bytearray(data)
                    b[at] ^= mask
                    items.append(("%s-flip-%d-%d" % (name, at, mask), bytes(b), False))
        self.check_many(items, False)

    def test_seeded_damage_of_each_section_with_a_repaired_checksum(self):
        rng = random.Random(20261014)
        items = []
        for i in range(900):
            name, data = rng.choice((("lin", self.lin), ("log", self.log)))
            b = bytearray(data)
            tag, start, n = rng.choice([s for s in sections(data) if s[2] > 0])
            for _ in range(rng.randint(1, 3)):
                at = start + rng.randrange(n)
                b[at] = rng.choice((0, 1, 2, 0x7F, 0x80, 0xFF, rng.randrange(256)))
            items.append(("%s-damage-%d" % (name, i), fix_checksum(bytes(b)), True))
        self.check_many(items, True)

    def test_a_declared_length_of_a_step_or_a_record_that_runs_past_the_end_is_refused(self):
        steps = payloads(self.lin)[3]
        for new in (struct.pack("<Q", 1 << 62), struct.pack("<Q", len(steps)), struct.pack("<Q", 0)):
            self.sb.write("e.cml", self.sealed_with(self.lin, 3, 4, new))
            self.assertEqual(predict_rows(self.sb, "e.cml").first_rule(), "ml.model.malformed")
        self.sb.write("e.cml", self.sealed_with(self.lin, 3, 0, struct.pack("<I", 1 << 31)))
        self.assertEqual(predict_rows(self.sb, "e.cml").first_rule(), "ml.model.malformed")


if __name__ == "__main__":
    unittest.main()

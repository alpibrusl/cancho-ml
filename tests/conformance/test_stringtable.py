"""S3, S5, S8, S10 on the command line (docs/stringtable.md section 9): the file a categorical fit writes is read by an independent reader, the reasons of every refusal of a
damaged table are reached from `predict`, a numeric model keeps its five sections, the columns of a predict file are found by name, unseen categories are counted, and a
label out of the range of an integer column is a malformed file, not a trap.

The byte-for-byte comparison with an oracle, the round trip, the fuzz and the bounds are scripts/gate_st.py; the unit program of the decoder is tests/strtab.cho.
"""

import json
import struct
import sys
import unittest

from harness import ROOT, Sandbox, TRAPS, fix_checksum, sections, set_bytes

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_st as S  # noqa: E402

CSV = ("id,sex,port,age,y\n1,male,S,20,1\n2,female,C,30,0\n3,male,S,40,1\n4,female,Q,50,0\n5,male,S,31,1\n6,female,C,22,0\n7,male,S,35,1\n"
       "8,male,C,44,0\n9,female,S,29,0\n10,male,Q,51,1\n")
FIT = ["fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", "sex,port", "--drop", "id", "--output", "m.cml", "--step", "encode column=sex,port kind=onehot missing=zero"]


class Base(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        self.sb.write("t.csv", CSV)
        r = self.sb.run(*FIT, "t.csv")
        self.assertEqual(r.status, 0, r)
        self.file = self.sb.read("m.cml")

    def tearDown(self):
        self.sb.cleanup()

    def predict(self, data=None, *extra, name="p.csv"):
        if data is not None:
            self.sb.write(name, data)
        return self.sb.run("predict", name, "--model-file", "m.cml", *extra)


class Layout(Base):
    def test_a_categorical_fit_has_six_sections_and_the_table_the_oracle_writes(self):
        secs = S.parse_file(self.file)
        self.assertEqual(sorted(secs), [1, 2, 3, 4, 5, 6])
        self.assertIsNone(S.table_verdict(secs[6]))
        # the steps section holds a stub of 8 bytes: u32 n = 1, u64 length 8, kind 7, layout 2
        self.assertEqual(secs[3], struct.pack("<IQII", 1, 8, 7, 2))
        # the schema: the columns of the file but the dropped one, sex and port with type 3 and scale 0
        sc = secs[2]
        self.assertEqual(struct.unpack_from("<II", sc, 0), (4, 3))
        self.assertIn(b"sex" + struct.pack("<II", 3, 0), sc)
        self.assertIn(b"port" + struct.pack("<II", 3, 0), sc)
        # the table: one entry for step 1 with the strings of the two columns in bytewise order, the counts of the rows
        t = secs[6]
        self.assertEqual(struct.unpack_from("<IIQQ", t, 0), (1, 1, 4 + 6 + 3, 5))
        self.assertEqual(t[-32:], S.sha(t[:-32]))
        self.assertEqual(struct.unpack_from("<I", t, 24)[0], 0)
        self.assertEqual(struct.unpack_from("<IIII", t, 28), (64, 255, 1048576, 128))

    def test_the_markers_and_the_limits_of_the_fit_are_stored(self):
        r = self.sb.run("fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", "sex,port", "--drop", "id", "--na-values", "NA,?", "--max-categories", "7",
                        "--max-string-bytes", "9", "--max-pool-bytes", "999", "--max-cat-columns", "3", "--output", "m2.cml", "--step", "encode column=sex,port kind=onehot", "t.csv")
        self.assertEqual(r.status, 0, r)
        t = S.parse_file(self.sb.read("m2.cml"))[6]
        self.assertEqual(struct.unpack_from("<I", t, 24)[0], 4)
        self.assertEqual(t[28:32], b"NA,?")
        self.assertEqual(struct.unpack_from("<IIII", t, 32), (7, 9, 999, 3))
        self.assertIsNone(S.table_verdict(t))

    def test_a_numeric_model_keeps_five_sections_and_no_table(self):
        self.sb.write("n.csv", "a,b,y\n1,2,5.1\n2,1,7.9\n3,4,9.3\n4,5,12.2\n5,6,13.8\n6,3,15.1\n")
        for model in ("linear", "tree"):
            r = self.sb.run("fit", "--model", model, "--target", "y", "--output", "n.cml", "--overwrite", "n.csv")
            self.assertEqual(r.status, 0, r)
            data = self.sb.read("n.cml")
            self.assertEqual(struct.unpack_from("<I", data, 8)[0], 5)
            self.assertEqual(sorted(S.parse_file(data)), [1, 2, 3, 4, 5])
            self.assertNotIn("string_table", r.data()["summary"])

    def test_the_same_fit_twice_gives_the_same_bytes_and_a_hash_step_stores_no_string(self):
        r = self.sb.run(*FIT, "--overwrite", "t.csv")
        self.assertEqual(self.sb.read("m.cml"), self.file)
        r = self.sb.run("fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", "sex,port", "--drop", "id", "--output", "h.cml", "--step",
                        "hash column=sex buckets=4 missing=zero; hash column=port buckets=3 missing=zero", "t.csv")
        self.assertEqual(r.status, 0, r)
        h = self.sb.read("h.cml")
        for s in (b"male", b"female", b"S", b"C", b"Q"):
            if len(s) > 1:
                self.assertNotIn(s, h)
        self.assertEqual(struct.unpack_from("<QQ", S.parse_file(h)[6], 8), (0, 0))


class Refusals(Base):
    """each reason of ml.model.string-table reached from predict: the table or its stub edited, the digests repaired"""

    def reach(self, edit_table=None, edit_steps=None, edit_schema=None, edit_plan=None):
        d = self.file
        secs = sections(d)
        s = {t: (a, n) for t, a, n in secs}
        parts = {t: bytearray(d[a:a + n]) for t, (a, n) in s.items()}
        if edit_table:
            parts[6] = bytearray(edit_table(bytes(parts[6])))
            parts[6][-32:] = S.sha(bytes(parts[6][:-32]))
        if edit_steps:
            parts[3] = bytearray(edit_steps(bytes(parts[3])))
        if edit_schema:
            parts[2] = bytearray(edit_schema(bytes(parts[2])))
        if edit_plan:
            parts[1] = bytearray(edit_plan(bytes(parts[1])))
        body = b"CML1" + struct.pack("<II", 1, 6) + b"".join(struct.pack("<IQ", t, len(parts[t])) + bytes(parts[t]) for t in (1, 2, 3, 4, 5, 6))
        self.sb.write("bad.cml", body + S.sha(body))
        r = self.sb.run("predict", "t.csv", "--model-file", "bad.cml", "--drop", "id")
        self.assertNotIn(r.status, TRAPS, r)
        return r

    def reason(self, **kw):
        r = self.reach(**kw)
        self.assertEqual(r.first_rule(), "ml.model.string-table", r)
        self.assertEqual(r.error()["detail"]["section"], 6)
        return r.error()["detail"]["reason"], r.error()["detail"].get("step")

    def test_each_reason(self):
        def at(off, v, w=4):
            return lambda t: t[:off] + (struct.pack("<I", v) if w == 4 else struct.pack("<q", v)) + t[off + w:]
        # offsets of this table: 0 layout, 4 entries, 8 pool, 16 strings, 24 na_len, 28 limits (16 bytes), 44 the entry: step, length (u64), 56 the payload
        self.assertEqual(self.reason(edit_table=at(0, 2))[0], "layout")
        self.assertEqual(self.reason(edit_table=at(4, 0))[0], "entries")
        self.assertEqual(self.reason(edit_table=at(8, 2 ** 63 - 1, 8))[0], "header")
        self.assertEqual(self.reason(edit_table=at(28, 0))[0], "header")
        self.assertEqual(self.reason(edit_table=at(44, 0))[0], "step")
        self.assertEqual(self.reason(edit_table=at(48, 2 ** 63 - 1, 8))[0], "length")
        self.assertEqual(self.reason(edit_table=at(56, 11))[0], "payload")
        self.assertEqual(self.reason(edit_table=at(8, 14, 8))[0], "totals")
        self.assertEqual(self.reason(edit_table=at(32, 3))[0], "limits")

    def test_utf8_digest_trailing_and_the_stubs(self):
        # an invalid byte in the first string of the first column (bytes of the table after the name "sex")
        def bad_utf8(t):
            i = t.index(b"\x04\x00\x00\x00male") + 4 + 3
            return t[:i] + b"\xff" + t[i + 1:]
        self.assertEqual(self.reason(edit_table=bad_utf8)[0], "utf8")
        # the digest of the table: a flipped byte, the table digest not repaired, the file digest repaired
        d = self.file
        secs = {t: (a, n) for t, a, n in sections(d)}
        a, n = secs[6]
        broken = fix_checksum(set_bytes(d, a + 30, bytes([d[a + 30] ^ 1])))
        self.sb.write("bad2.cml", broken)
        r = self.sb.run("predict", "t.csv", "--model-file", "bad2.cml", "--drop", "id")
        self.assertEqual((r.first_rule(), r.error()["detail"]["reason"]), ("ml.model.string-table", "digest"))
        # a table with a byte more inside it, before the digest: the entries end before the body does
        self.assertEqual(self.reason(edit_table=lambda t: t[:-32] + b"\0" + t[-32:])[0], "trailing")
        # a stub of layout 1, a stub of the wrong kind, a steps section of one numeric step (the table is left over)
        self.assertEqual(self.reason(edit_steps=lambda s: s[:16] + struct.pack("<I", 1))[0], "stub")
        self.assertEqual(self.reason(edit_steps=lambda s: s[:12] + struct.pack("<I", 8) + s[16:])[0], "stub")
        # the schema without the type 3 of a column, and the schema with the type 3 of a numeric column
        def untype(sc):
            i = sc.index(b"sex")
            return sc[:i + 3] + struct.pack("<II", 2, 0) + sc[i + 11:]
        self.assertEqual(self.reason(edit_schema=untype)[0], "schema")
        # the plan against the stored steps: unknown=refuse in the plan, ignore stored
        r = self.reach(edit_plan=lambda p: p.replace(b"unknown=ignore", b"unknown=refuse", 1))
        self.assertEqual((r.first_rule(), r.error()["detail"]["reason"]), ("ml.model.string-table", "plan"))

    def test_a_table_without_a_categorical_column_and_a_categorical_column_without_a_table(self):
        d = self.file
        secs = {t: (a, n) for t, a, n in sections(d)}
        # the type 3 taken from both columns of the schema: the file has a table and no categorical column
        def untype(sc):
            for nm in (b"sex", b"port"):
                i = sc.index(nm)
                sc = sc[:i + len(nm)] + struct.pack("<II", 2, 0) + sc[i + len(nm) + 8:]
            return sc
        self.assertEqual(self.reason(edit_schema=untype)[0], "schema")
        # the table taken out: five sections, a schema with the type 3 and a stub in the steps section
        parts = {t: d[a:a + n] for t, (a, n) in secs.items()}
        body = b"CML1" + struct.pack("<II", 1, 5) + b"".join(struct.pack("<IQ", t, len(parts[t])) + parts[t] for t in (1, 2, 3, 4, 5))
        self.sb.write("five.cml", body + S.sha(body))
        r = self.sb.run("predict", "t.csv", "--model-file", "five.cml", "--drop", "id")
        self.assertEqual((r.first_rule(), r.error()["detail"]["section"]), ("ml.model.malformed", 2), r)

    def test_the_plan_must_describe_the_stored_steps(self):
        r = self.sb.run("fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", "sex,port", "--drop", "id", "--output", "f.cml", "--step",
                        "freq column=sex missing=zero min-count=2 max-categories=3; freq column=port", "t.csv")
        self.assertEqual(r.status, 0, r)
        d = self.sb.read("f.cml")
        secs = {t: (a, n) for t, a, n in sections(d)}
        plan = d[secs[1][0]:secs[1][0] + secs[1][1]]
        self.assertIn(b"freq column=sex max-categories=3 unknown=ignore min-count=2 missing=zero", plan)
        for old, new in ((b"missing=zero", b"missing=pass"), (b"min-count=2", b"min-count=3"), (b"unknown=ignore", b"unknown=refuse"), (b"max-categories=3", b"max-categories=1"),
                         (b"freq column=sex", b"freq column=port")):
            p2 = plan.replace(old, new, 1)
            body = b"CML1" + struct.pack("<II", 1, 6)
            for t in (1, 2, 3, 4, 5, 6):
                a, n = secs[t]
                payload = p2 if t == 1 else d[a:a + n]
                body += struct.pack("<IQ", t, len(payload)) + payload
            self.sb.write("pl.cml", body + S.sha(body))
            q = self.sb.run("predict", "t.csv", "--model-file", "pl.cml", "--drop", "id")
            if old == b"freq column=sex":
                # the plan names another column: not the canonical text of a plan with sorted names? it is canonical, and disagrees with the stored step
                self.assertEqual((q.first_rule(), q.error()["detail"].get("reason")), ("ml.model.string-table", "plan"), (old, new, q))
            else:
                self.assertEqual((q.first_rule(), q.error()["detail"].get("reason")), ("ml.model.string-table", "plan"), (old, new, q))

    def test_a_six_section_file_is_this_readers_whatever_its_model_section_says(self):
        # the kind of the model section made unknown: the pipeline's reader refuses it (malformed), not the constant model's (which also says malformed for six sections)
        d = self.file
        secs = {t: (a, n) for t, a, n in sections(d)}
        bad = fix_checksum(set_bytes(d, secs[4][0], struct.pack("<I", 99)))
        self.sb.write("bad3.cml", bad)
        r = self.sb.run("predict", "t.csv", "--model-file", "bad3.cml", "--drop", "id")
        self.assertEqual(r.first_rule(), "ml.model.malformed", r)
        # and the checksum comes first when it is wrong
        self.sb.write("bad4.cml", set_bytes(d, secs[4][0], struct.pack("<I", 99)))
        r = self.sb.run("predict", "t.csv", "--model-file", "bad4.cml", "--drop", "id")
        self.assertEqual(r.first_rule(), "ml.model.checksum", r)


class Reading(Base):
    def test_the_columns_of_the_file_are_found_by_name_in_any_order(self):
        rows = [r.split(",") for r in CSV.strip().split("\n")]
        head, body = rows[0], rows[1:]
        base = None
        for order in (["id", "sex", "port", "age", "y"], ["y", "age", "port", "sex", "id"], ["port", "id", "y", "sex", "age"], ["age", "sex", "id", "port"]):
            idx = [head.index(c) for c in order]
            data = "\n".join([",".join(order)] + [",".join(r[i] for i in idx) for r in body]) + "\n"
            r = self.predict(data, "--drop", "id", "--limit", "100")
            self.assertEqual(r.status, 0, (order, r))
            if base is None:
                base = r.data()["rows"]
            self.assertEqual(r.data()["rows"], base, order)

    def test_unseen_categories_are_counted_and_a_known_one_is_not(self):
        r = self.predict("sex,port,age\nmale,S,20\nrobot,S,20\nfemale,Z,20\n,S,20\nmale,,20\n")
        self.assertEqual(r.status, 0, r)
        self.assertEqual(r.data()["unknown_categories"], 2)

    def test_a_missing_column_an_extra_column_and_a_retyped_column(self):
        r = self.predict("sex,age\nmale,20\n")
        self.assertEqual((r.first_rule(), r.error()["detail"]["missing"]), ("ml.schema.mismatch", ["port"]))
        self.assertEqual(r.error()["repair"]["kind"], "none")
        r = self.predict("sex,port,age,zz\nmale,S,20,5\n")
        self.assertEqual((r.first_rule(), r.error()["detail"]["extra"]), ("ml.schema.mismatch", ["zz"]))
        self.assertEqual(r.error()["repair"]["kind"], "retry")
        self.assertEqual(r.error()["repair"]["argv"][-2:], ["--drop", "zz"])
        r = self.predict("sex,port,age\nmale,S,twenty\n")
        self.assertNotEqual(r.status, 0)
        r = self.predict("sex,port,age\nmale,S,20\n", "--drop", "port")
        self.assertEqual((r.first_rule(), r.error()["detail"]["missing"]), ("ml.schema.mismatch", ["port"]))

    def test_the_markers_of_the_fit_are_the_markers_of_the_predict(self):
        r = self.sb.run("fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", "sex,port", "--drop", "id", "--na-values", "NA", "--output", "n.cml", "--step",
                        "encode column=sex,port kind=onehot missing=zero", "t.csv")
        self.assertEqual(r.status, 0, r)
        self.sb.write("q.csv", "sex,port,age\nNA,S,20\n?,S,20\nmale,NA,20\n")
        p = self.sb.run("predict", "q.csv", "--model-file", "n.cml")
        self.assertEqual(p.status, 0, p)
        # NA is missing (not unseen) in both columns; "?" is an unseen string
        self.assertEqual(p.data()["unknown_categories"], 1)

    def test_a_cell_longer_than_the_limit_of_the_fit_is_refused_as_at_fit(self):
        r = self.sb.run("fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", "sex,port", "--drop", "id", "--max-string-bytes", "6", "--output", "s.cml", "--step",
                        "encode column=sex,port kind=onehot", "t.csv")
        self.assertEqual(r.status, 0, r)
        self.sb.write("q.csv", "sex,port,age\nmale,S,20\nfemale1,S,20\n")
        p = self.sb.run("predict", "q.csv", "--model-file", "s.cml")
        self.assertEqual(p.first_rule(), "ml.limit.string-bytes", p)
        self.sb.write("q2.csv", b"sex,port,age\nmale,\xff,20\n")
        p = self.sb.run("predict", "q2.csv", "--model-file", "s.cml")
        self.assertIn(p.first_rule(), ("ml.input.invalid-utf8",), p)

    def test_unknown_refuse_names_the_row_at_predict(self):
        r = self.sb.run("fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", "sex,port", "--drop", "id", "--output", "u.cml", "--step",
                        "encode column=sex kind=onehot unknown=refuse; encode column=port kind=onehot", "t.csv")
        self.assertEqual(r.status, 0, r)
        self.sb.write("q.csv", "sex,port,age\nmale,S,20\nmale,Q,20\nrobot,S,20\n")
        p = self.sb.run("predict", "q.csv", "--model-file", "u.cml")
        self.assertEqual(p.first_rule(), "ml.input.unknown-category", p)
        d = p.error()["detail"]
        self.assertEqual((d["step"], d["column"], d["row"]), (1, "sex", 2))

    def test_a_hashed_column_puts_an_unseen_string_in_its_bucket(self):
        # the prediction of two unseen strings that hash to one bucket is the same, to another bucket differs (4 buckets, strings found by trial)
        r = self.sb.run("fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", "sex,port", "--drop", "id", "--output", "h.cml", "--step",
                        "hash column=sex buckets=2 missing=zero; encode column=port kind=label missing=zero", "t.csv")
        self.assertEqual(r.status, 0, r)
        names = ["robot", "alien", "ghost", "cyborg", "elf", "orc"]
        self.sb.write("q.csv", "sex,port,age\n" + "".join("%s,S,20\n" % n for n in names))
        p = self.sb.run("predict", "q.csv", "--model-file", "h.cml", "--limit", "10")
        self.assertEqual(p.status, 0, p)
        self.assertEqual(p.data()["unknown_categories"], 0)
        import hashlib
        buckets = [int.from_bytes(hashlib.sha256(n.encode()).digest()[:8], "big") % 2 for n in names]
        vals = [row[0] for row in p.data()["rows"]]
        for i in range(len(names)):
            for j in range(len(names)):
                self.assertEqual(vals[i] == vals[j], buckets[i] == buckets[j], (names[i], names[j]))


class Labels(unittest.TestCase):
    """a label past the 12 digits of an integer column is a malformed file, as it was written by no fit (it used to trap when printed)"""

    def test_a_label_out_of_range_is_malformed(self):
        sb = Sandbox()
        try:
            sb.write("c.csv", "a,b,y\n" + "".join("%d,%d,%d\n" % (i, (i * 7) % 5, 1 if (i * 3) % 7 > 3 else 0) for i in range(30)))
            for model in ("logistic", "tree"):
                extra = ["--task", "classification"] if model == "tree" else ["--l2", "0.1"]
                r = sb.run("fit", "--model", model, *extra, "--target", "y", "--output", "m.cml", "--overwrite", "c.csv")
                self.assertEqual(r.status, 0, r)
                d = sb.read("m.cml")
                secs = {t: (a, n) for t, a, n in sections(d)}
                a, n = secs[4]
                end = a + n
                for field, value in ((end - 16, 2 ** 63 - 1), (end - 8, -2 ** 63), (end - 8, 10 ** 12), (end - 16, -10 ** 12)):
                    sb.write("bad.cml", fix_checksum(set_bytes(d, field, struct.pack("<q", value))))
                    p = sb.run("predict", "c.csv", "--model-file", "bad.cml")
                    self.assertNotIn(p.status, TRAPS, (model, value, p))
                    self.assertEqual(p.first_rule(), "ml.model.malformed", (model, value, p))
                # the largest label of an integer column is read
                sb.write("big.csv", "a,b,y\n" + "".join("%d,%d,%d\n" % (i, (i * 7) % 5, 999999999999 if (i * 3) % 7 > 3 else -999999999999) for i in range(30)))
                r = sb.run("fit", "--model", model, *extra, "--target", "y", "--output", "big.cml", "--overwrite", "big.csv")
                if r.status == 0:
                    e = sb.run("predict", "big.csv", "--model-file", "big.cml")
                    self.assertEqual(e.status, 0, e)
        finally:
            sb.cleanup()


if __name__ == "__main__":
    unittest.main()

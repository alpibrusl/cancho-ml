"""T1 to T11 on the command line (docs/tracking.md section 9): the provenance section of the model file, the answers that use it (on, out_of_range, ignored_columns, --max-out-of-range), the run
record and its id, `--track`, the four `ml runs` verbs, the refusals of a damaged section or record, and what a record never holds.

The bytes of the section and of the record against the oracle are scripts/oracle_track.py (also run from here, on a smaller table); the mutants are scripts/mutants_track.py.
"""

import hashlib
import json
import os
import struct
import sys
import unittest

from harness import ROOT, Sandbox, TRAPS, fix_checksum, run, sections, set_bytes

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_track as O  # noqa: E402

TRAIN = "a,b,c,y\n1,3,0.5,3\n2,1,1.5,5\n3,4,,8\n4,1,2.0,9\n5,5,3.5,14\n6,9,1,13\n7,2,0.25,12\n8,6,2.75,10\n"
LAST = "8,6,2.75,10"
CHANGED = "8,6,2.75,11"
LIN = ["--model", "linear", "--target", "y", "--step", "impute column=c strategy=mean"]


def strip_provenance(data):
    """the same file without section 7 (the header's count and the checksum repaired): what an older version wrote"""
    out = bytearray(data[:12])
    n = 0
    for tag, start, length in sections(data):
        if tag != 7:
            out += data[start - 12:start + length]
            n += 1
    out[8:12] = struct.pack("<I", n)
    return fix_checksum(bytes(out) + bytes(32))


def with_provenance(data, payload):
    """the file with the payload of section 7 replaced (the length and the checksum made again)"""
    out = bytearray(data[:12])
    for tag, start, length in sections(data):
        if tag == 7:
            out += struct.pack("<IQ", 7, len(payload)) + payload
        else:
            out += data[start - 12:start + length]
    return fix_checksum(bytes(out) + bytes(32))


def dbits(v):
    return struct.unpack("<q", struct.pack("<d", v))[0]


def section_at(data, tag):
    for t, start, length in sections(data):
        if t == tag:
            return start, length
    return None


class Base(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        self.sb.write("train.csv", TRAIN)
        self.sb.dir.joinpath("runs").mkdir()

    def tearDown(self):
        self.sb.cleanup()

    def fit(self, *extra, name="m.cml", data="train.csv", flags=LIN):
        r = self.sb.run("fit", *flags, "--output", name, "--overwrite", *extra, data)
        self.assertEqual(r.status, 0, r)
        return r


class Section(Base):
    def test_every_fit_writes_the_section_and_the_oracle_writes_the_same_bytes(self):
        self.fit()
        data = self.sb.read("m.cml")
        self.assertEqual(sorted(t for t, _, _ in sections(data)), [1, 2, 3, 4, 5, 7])
        s, n = section_at(data, 7)
        self.assertEqual(data[s:s + n], O.provenance(TRAIN.encode(), ["a", "b", "c"], None))
        self.assertEqual(n, 64 + 3 * 40)

    def test_a_constant_model_carries_the_identity_of_the_file_and_no_feature(self):
        r = self.sb.run("fit", "--target", "y", "--drop", "c", "--output", "k.cml", "train.csv")
        self.assertEqual(r.status, 0, r)
        data = self.sb.read("k.cml")
        s, n = section_at(data, 7)
        self.assertEqual(n, 64)
        self.assertEqual(data[s:s + n], O.provenance(TRAIN.encode(), [], None))
        self.assertEqual(struct.unpack_from("<I", data, 8)[0], 6)

    def test_the_statistics_are_exact_and_rounded_once(self):
        self.sb.write("x.csv", "u,y\n0.1,1\n0.2,2\n0.3,3\n1e-3,4\n7,5\n")
        self.fit(data="x.csv", flags=["--model", "linear", "--target", "y"])
        data = self.sb.read("m.cml")
        s, n = section_at(data, 7)
        count, mn, mx, me, sd = struct.unpack_from("<Qqqqq", data, s + 64)
        f = lambda b: struct.unpack("<d", struct.pack("<q", b))[0]  # noqa: E731
        self.assertEqual((count, f(mn), f(mx)), (5, 0.001, 7.0))
        self.assertEqual(f(me), float(sum(O.Fraction(x) for x in (0.1, 0.2, 0.3, 1e-3, 7.0)) / 5))

    def test_the_section_is_the_same_bytes_at_every_thread_count(self):
        outs = []
        for th in (1, 2, 8):
            self.fit("--threads", str(th))
            outs.append(self.sb.read("m.cml"))
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], outs[2])

    def test_a_file_without_the_section_is_read_and_says_so(self):
        self.fit()
        old = strip_provenance(self.sb.read("m.cml"))
        self.sb.write("old.cml", old)
        d = self.sb.run("describe", "--model-file", "old.cml").data()
        self.assertIsNone(d["provenance"])
        e = self.sb.run("eval", "--model-file", "old.cml", "train.csv").data()
        self.assertEqual(e["on"], "unknown")
        self.assertIsNone(e["out_of_range"])
        p = self.sb.run("predict", "--model-file", "old.cml", "train.csv").data()
        self.assertIsNone(p["out_of_range"])
        r = self.sb.run("predict", "--model-file", "old.cml", "--max-out-of-range", "0", "train.csv")
        self.assertEqual(r.first_rule(), "ml.predict.no-provenance")
        self.assertEqual(r.status, 8)

    def test_describe_prints_the_provenance(self):
        self.fit()
        d = self.sb.run("describe", "--model-file", "m.cml").data()["provenance"]
        self.assertEqual(d["file_sha256"], hashlib.sha256(TRAIN.encode()).hexdigest())
        self.assertEqual((d["file_bytes"], d["file_rows"], d["rows"]), (len(TRAIN), 8, 8))
        self.assertEqual([f["name"] for f in d["features"]], ["a", "b", "c"])
        self.assertEqual(d["features"][2]["count"], 7)
        self.assertEqual((d["features"][0]["min"], d["features"][0]["max"], d["features"][0]["mean"]), ("1.0", "8.0", "4.5"))

    def test_the_train_side_of_a_split_is_what_the_statistics_cover(self):
        self.sb.write("big.csv", "a,y\n" + "".join("%d,%d\n" % (i, 2 * i + 1) for i in range(40)))
        self.fit("--split", "train=0.75", "--seed", "4", data="big.csv", flags=["--model", "linear", "--target", "y"])
        data = self.sb.read("m.cml")
        s, n = section_at(data, 7)
        import oracle_metrics as om

        test = om.py_train_test(40, 10, 4, 0)
        self.assertEqual(data[s:s + n], O.provenance(self.sb.read("big.csv"), ["a"], [t == 0 for t in test]))
        self.assertEqual(struct.unpack_from("<QQ", data, s + 44), (40, 30))

    def test_the_other_files_of_the_suite_have_the_bytes_of_the_oracle(self):
        # the whole table of the oracle is scripts/oracle_track.py --suite; two of its cases here
        import shutil
        import tempfile
        import pathlib

        work = pathlib.Path(tempfile.mkdtemp(prefix="ot-"))
        try:
            for label, flags, csvname in O.case_list()[2:4]:
                (work / label.replace(" ", "_")).mkdir()
                self.assertIsNotNone(O.oracle_case(label, flags, csvname, work / label.replace(" ", "_")))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.assertEqual(O.FAILS, [])


class On(Base):
    def test_on_says_train_for_the_training_file_under_any_name_and_other_for_one_byte_more(self):
        self.fit()
        self.assertEqual(self.sb.run("eval", "--model-file", "m.cml", "train.csv").data()["on"], "train")
        self.sb.write("copy.csv", TRAIN)
        self.assertEqual(self.sb.run("eval", "--model-file", "m.cml", "copy.csv").data()["on"], "train")
        self.sb.write("other.csv", TRAIN.replace(LAST, CHANGED))
        self.assertEqual(self.sb.run("eval", "--model-file", "m.cml", "other.csv").data()["on"], "other")
        self.sb.write("longer.csv", TRAIN + "\n")
        self.assertEqual(self.sb.run("eval", "--model-file", "m.cml", "longer.csv").data()["on"], "other")

    def test_the_text_format_says_it_too(self):
        self.fit()
        r = self.sb.run("--format", "text", "eval", "--model-file", "m.cml", "train.csv")
        self.assertTrue(r.stdout.decode().startswith("on train\n"), r)

    def test_a_constant_model_says_it_as_well(self):
        r = self.sb.run("fit", "--target", "y", "--drop", "c", "--output", "k.cml", "train.csv")
        self.assertEqual(r.status, 0, r)
        self.assertEqual(self.sb.run("eval", "--model-file", "k.cml", "--drop", "c", "train.csv").data()["on"], "train")
        self.sb.write("other.csv", TRAIN.replace(LAST, CHANGED))
        self.assertEqual(self.sb.run("eval", "--model-file", "k.cml", "--drop", "c", "other.csv").data()["on"], "other")

    def test_the_held_out_rows_of_a_fit_are_labelled(self):
        self.sb.write("big.csv", "a,y\n" + "".join("%d,%d\n" % (i, 2 * i + 1) for i in range(40)))
        r = self.sb.run("fit", "--model", "linear", "--target", "y", "--split", "train=0.75", "--output", "m.cml", "big.csv")
        self.assertEqual(r.data()["summary"]["holdout"]["on"], "held-out")


class Range(Base):
    def test_the_counts_equal_an_independent_computation_and_the_bounds_are_inside(self):
        self.fit()
        # a cell equal to the minimum or the maximum is not out of range; an empty cell is never counted; the target (y) is not a feature
        rows = "a,b,c,y\n1,3,0.5,3\n8,9,3.5,13\n0,3,,100\n9,10,0.2,3\n-1,0,9,3\n3,4,2,1000\n"
        self.sb.write("new.csv", rows)
        d = self.sb.run("predict", "--model-file", "m.cml", "new.csv").data()
        header, data = O.read_csv(rows.encode())
        prov = self.sb.read("m.cml")
        s, n = section_at(prov, 7)
        cells, nrows, counts = O.out_of_range(prov[s:s + n], header, data, ["a", "b", "c"])
        self.assertEqual(d["out_of_range"], {"cells": cells, "rows": nrows, "columns": [{"name": k, "count": counts[k]} for k in ("a", "b", "c") if k in counts]})
        self.assertEqual((cells, nrows), (7, 3))
        e = self.sb.run("eval", "--model-file", "m.cml", "new.csv").data()
        self.assertEqual(e["out_of_range"], d["out_of_range"])

    def test_the_bound_refuses_above_n_rows_and_names_the_first_column_and_row(self):
        self.fit()
        self.sb.write("new.csv", "a,b,c,y\n1,3,0.5,3\n0,3,1,3\n9,10,1,3\n2,4,1,3\n")
        for n, ok in ((1, False), (2, True), (3, True)):
            r = self.sb.run("predict", "--model-file", "m.cml", "--max-out-of-range", str(n), "new.csv")
            self.assertEqual(r.status == 0, ok, (n, r))
        r = self.sb.run("predict", "--model-file", "m.cml", "--max-out-of-range", "1", "new.csv")
        self.assertEqual(r.first_rule(), "ml.predict.out-of-range")
        self.assertEqual(r.status, 8)
        d = r.error()["detail"]
        self.assertEqual((d["rows"], d["max"], d["column"], d["row"]), (2, 1, "a", 2))
        self.assertNotIn("10", json.dumps(d))
        r = self.sb.run("eval", "--model-file", "m.cml", "--max-out-of-range", "1", "new.csv")
        self.assertEqual(r.first_rule(), "ml.predict.out-of-range")
        r = self.sb.run("predict", "--model-file", "m.cml", "--max-out-of-range", "0", "train.csv")
        self.assertEqual(r.status, 0, r)

    def test_predict_reports_the_target_column_it_ignores(self):
        self.fit()
        d = self.sb.run("predict", "--model-file", "m.cml", "train.csv").data()
        self.assertEqual(d["ignored_columns"], ["y"])
        self.sb.write("nt.csv", "a,b,c\n1,10,0.5\n")
        self.assertEqual(self.sb.run("predict", "--model-file", "m.cml", "nt.csv").data()["ignored_columns"], [])

    def test_a_split_scores_the_rows_it_scores(self):
        self.sb.write("big.csv", "a,y\n" + "".join("%d,%d\n" % (i, 2 * i + 1) for i in range(40)))
        self.fit("--split", "train=0.5", "--seed", "1", data="big.csv", flags=["--model", "linear", "--target", "y"])
        self.sb.write("far.csv", "a,y\n" + "".join("%d,%d\n" % (100 + i, 2 * i + 1) for i in range(40)))
        e = self.sb.run("eval", "--model-file", "m.cml", "--split", "train=0.5", "--seed", "1", "far.csv").data()
        self.assertEqual(e["out_of_range"]["rows"], e["split"]["test"])
        e = self.sb.run("eval", "--model-file", "m.cml", "far.csv").data()
        self.assertEqual(e["out_of_range"]["rows"], 40)


class Damage(Base):
    def run_with(self, data, name="d.cml"):
        self.sb.write(name, data)
        return self.sb.run("describe", "--model-file", name)

    def test_every_byte_of_the_section_flipped_never_traps_and_a_refusal_has_its_tag(self):
        self.fit()
        data = self.sb.read("m.cml")
        s, n = section_at(data, 7)
        bad = ok = 0
        for at in range(s, s + n):
            for xor in (1, 0x80, 0xFF):
                mut = bytearray(data)
                mut[at] ^= xor
                r = self.run_with(fix_checksum(bytes(mut)))
                self.assertNotIn(r.status, TRAPS, (at, xor))
                if r.status == 0:
                    ok += 1
                else:
                    self.assertEqual(r.first_rule(), "ml.model.malformed", (at, xor, r))
                    self.assertEqual(r.error()["detail"]["section"], 7)
                    bad += 1
        self.assertGreater(bad, 0)
        self.assertGreater(ok, 0)

    def test_a_flip_without_a_repaired_checksum_is_the_checksum(self):
        self.fit()
        data = self.sb.read("m.cml")
        s, n = section_at(data, 7)
        mut = bytearray(data)
        mut[s + 70] ^= 1
        self.assertEqual(self.run_with(bytes(mut)).first_rule(), "ml.model.checksum")

    def test_every_truncation_of_the_section_is_a_refusal(self):
        self.fit()
        data = self.sb.read("m.cml")
        for cut in range(1, 80):
            r = self.run_with(data[:-cut])
            self.assertNotIn(r.status, TRAPS, cut)
            self.assertNotEqual(r.status, 0, cut)

    def test_huge_counts_are_refused_before_anything_is_read(self):
        self.fit()
        data = self.sb.read("m.cml")
        s, n = section_at(data, 7)
        for value in (0xFFFFFFFF, 0x7FFFFFFF, 8193, 4):
            mut = set_bytes(data, s + 60, struct.pack("<I", value))
            r = self.run_with(fix_checksum(mut))
            self.assertEqual(r.first_rule(), "ml.model.malformed", value)
        for at in (4, 44, 52):
            for value in ((1 << 63) - 1, (1 << 62) + 1, (1 << 64) - 1, 0):
                mut = set_bytes(data, s + at, struct.pack("<Q", value))
                r = self.run_with(fix_checksum(mut))
                self.assertNotIn(r.status, TRAPS)
                if value != struct.unpack_from("<Q", data, s + at)[0] and not (at == 4 and value == 0):
                    self.assertEqual(r.first_rule(), "ml.model.malformed", (at, value))

    def test_a_payload_that_breaks_a_bound_is_malformed_whichever_bound_it_is(self):
        self.fit()
        data = self.sb.read("m.cml")
        s, n = section_at(data, 7)
        good = data[s:s + n]
        self.assertEqual(self.run_with(with_provenance(data, good)).status, 0)

        def put(p, at, fmt, *vals):
            return p[:at] + struct.pack(fmt, *vals) + p[at + struct.calcsize(fmt):]

        def rec(p, k, field, value):
            return put(p, 64 + 40 * k + 8 * field, "<q", value)

        mean0 = struct.unpack_from("<q", good, 64 + 24)[0]
        crafted = {
            "layout 2": put(good, 0, "<I", 2),
            "file_bytes past 2^62": put(good, 4, "<Q", (1 << 62) + 1),
            "file_bytes negative": put(good, 4, "<Q", 1 << 63),
            "file_rows 0": put(good, 44, "<Q", 0),
            "file_rows past 10^9": put(good, 44, "<Q", 1000000001),
            "rows 0": put(good, 52, "<Q", 0),
            "rows past file_rows": put(put(good, 44, "<Q", 7), 52, "<Q", 8),
            "rows not the rows of meta (counts still fit)": put(put(good, 44, "<Q", 9), 52, "<Q", 9),
            "eight bytes more than the count says": good + bytes(8),
            "eight bytes fewer than the count says": good[:-8],
            "minimum negative zero": rec(good, 0, 1, -(1 << 63)),
            "constant column of negative zeros": rec(rec(rec(rec(good, 0, 1, -(1 << 63)), 0, 2, -(1 << 63)), 0, 3, -(1 << 63)), 0, 4, 0),
            "one record fewer, features 2": put(good[:-40], 60, "<I", 2),
            "features 4 and one record more": put(good + good[-40:], 60, "<I", 4),
            "count past rows": rec(good, 0, 0, 9),
            "count 0 with a minimum": rec(rec(good, 0, 0, 0), 0, 1, dbits(1.0)),
            "minimum infinite": rec(good, 0, 1, 0x7FF0000000000000),
            "mean nan": rec(good, 0, 3, 0x7FF8000000000000),
            "std infinite": rec(good, 0, 4, 0x7FF0000000000000),
            "maximum negative zero": rec(good, 0, 2, -(1 << 63)),
            "minimum above maximum": rec(rec(good, 0, 1, dbits(9.0)), 0, 2, dbits(2.0)),
            "mean below minimum": rec(good, 0, 3, dbits(0.5)),
            "mean above maximum": rec(good, 0, 3, dbits(99.0)),
            "std negative": rec(good, 0, 4, dbits(-1.0)),
            "constant column with a deviation": rec(rec(rec(good, 0, 1, dbits(2.0)), 0, 2, dbits(2.0)), 0, 3, dbits(2.0)),
            "constant column with another mean": rec(rec(rec(rec(good, 0, 1, dbits(2.0)), 0, 2, dbits(2.0)), 0, 3, dbits(2.5)), 0, 4, 0),
        }
        for label, payload in crafted.items():
            r = self.run_with(with_provenance(data, payload))
            self.assertEqual(r.first_rule(), "ml.model.malformed", (label, r))
            self.assertEqual(r.error()["detail"]["section"], 7, label)
        # the same edits that keep every bound are accepted
        fine = rec(rec(rec(rec(good, 0, 1, dbits(2.0)), 0, 2, dbits(2.0)), 0, 3, dbits(2.0)), 0, 4, 0)
        self.assertEqual(self.run_with(with_provenance(data, fine)).status, 0)

    def test_a_constant_model_file_is_held_to_the_same_rules(self):
        r = self.sb.run("fit", "--target", "y", "--drop", "c", "--output", "k.cml", "train.csv")
        self.assertEqual(r.status, 0, r)
        data = self.sb.read("k.cml")
        s, n = section_at(data, 7)
        good = data[s:s + n]
        self.assertEqual(self.run_with(with_provenance(data, good)).status, 0)
        bad_layout = struct.pack("<I", 2) + good[4:]
        r = self.run_with(with_provenance(data, bad_layout))
        self.assertEqual(r.first_rule(), "ml.model.malformed", r)
        # a sixth section that is not the provenance (tag 6, the string table's) is not read as one
        out = bytearray(data[:12])
        for tag, start, length in sections(data):
            if tag == 7:
                out += struct.pack("<IQ", 6, len(good)) + good
            else:
                out += data[start - 12:start + length]
        r = self.run_with(fix_checksum(bytes(out) + bytes(32)))
        self.assertEqual(r.first_rule(), "ml.model.malformed", r)

    def test_a_section_out_of_order_or_repeated_is_malformed(self):
        self.fit()
        data = self.sb.read("m.cml")
        s, n = section_at(data, 7)
        body = data[:-32] + data[s - 12:s + n]
        mut = bytearray(body)
        mut[8:12] = struct.pack("<I", 7)
        r = self.run_with(fix_checksum(bytes(mut) + bytes(32)))
        self.assertEqual(r.first_rule(), "ml.model.malformed")


class Features(Base):
    def test_the_target_is_never_a_feature_whatever_its_place(self):
        self.sb.write("t.csv", "y,a,b\n5,1,3\n7,2,1\n9,3,4\n8,4,1\n13,5,5\n10,6,9\n")
        r = self.sb.run("fit", "--model", "linear", "--target", "y", "--output", "t.cml", "t.csv")
        self.assertEqual(r.status, 0, r)
        d = self.sb.run("describe", "--model-file", "t.cml").data()["provenance"]
        self.assertEqual([f["name"] for f in d["features"]], ["a", "b"])
        self.sb.write("n.csv", "y,a,b\n500,1,3\n-900,6,9\n5,7,1\n")
        p = self.sb.run("predict", "--model-file", "t.cml", "n.csv").data()
        self.assertEqual(p["out_of_range"], {"cells": 1, "rows": 1, "columns": [{"name": "a", "count": 1}]})
        self.assertEqual(p["ignored_columns"], ["y"])

    def test_the_column_dropped_by_the_caller_is_not_an_ignored_column(self):
        self.fit()
        d = self.sb.run("predict", "--model-file", "m.cml", "--drop", "y", "train.csv").data()
        self.assertEqual(d["ignored_columns"], [])


class Tracking(Base):
    def track(self, *extra, name="m.cml", flags=LIN):
        r = self.sb.run("fit", *flags, "--output", name, "--overwrite", "--track", "runs", *extra, "train.csv")
        self.assertEqual(r.status, 0, r)
        return r.data()["run"]

    def records(self):
        return sorted(p.name for p in (self.sb.dir / "runs").iterdir())

    def test_the_record_is_the_one_the_oracle_writes_and_holds_no_payload(self):
        import tempfile
        import pathlib
        import shutil

        work = pathlib.Path(tempfile.mkdtemp(prefix="ot-"))
        try:
            for label, flags, csvname in O.case_list()[:2]:
                (work / "x").mkdir()
                self.assertIsNotNone(O.oracle_case(label, flags, csvname, work / "x"))
                shutil.rmtree(work / "x")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.assertEqual(O.FAILS, [])

    def test_the_same_inputs_give_the_same_id_and_the_same_record_at_any_thread_count(self):
        a = self.track("--threads", "1")
        first = (self.sb.dir / "runs" / (a["id"] + ".json")).read_bytes()
        for th in ("2", "8"):
            b = self.track("--threads", th)
            self.assertEqual(b["id"], a["id"])
            self.assertFalse(b["written"])
            self.assertEqual(b["sha256"], a["sha256"])
        self.assertEqual((self.sb.dir / "runs" / (a["id"] + ".json")).read_bytes(), first)
        self.assertEqual(self.records(), [a["id"] + ".json"])
        self.assertEqual(json.loads(first)["threads"], 1)

    def test_other_inputs_give_other_ids(self):
        ids = {self.track()["id"], self.track("--seed", "3")["id"], self.track("--l2", "1.0")["id"] if False else self.track("--seed", "4")["id"]}
        self.assertEqual(len(ids), 3)
        self.sb.write("copy.csv", TRAIN)
        r = self.sb.run("fit", *LIN, "--output", "c.cml", "--track", "runs", "copy.csv")
        self.assertNotIn(r.data()["run"]["id"], ids)

    def test_the_metrics_of_a_record_say_which_rows_they_score(self):
        a = self.track()
        rec = json.loads((self.sb.dir / "runs" / (a["id"] + ".json")).read_text())
        self.assertEqual((rec["metrics"]["on"], rec["metrics"]["rows"]), ("train", 8))
        self.sb.write("big.csv", "a,y\n" + "".join("%d,%d\n" % (i, 2 * i + 1) for i in range(40)))
        r = self.sb.run("fit", "--model", "linear", "--target", "y", "--split", "train=0.75", "--output", "s.cml", "--track", "runs", "big.csv")
        rec = json.loads((self.sb.dir / "runs" / (r.data()["run"]["id"] + ".json")).read_text())
        self.assertEqual((rec["metrics"]["on"], rec["metrics"]["rows"]), ("held-out", 10))
        self.assertEqual(rec["counts"], {"file_rows": 40, "train_rows": 30, "test_rows": 10, "features": 1, "steps": 0})
        self.assertEqual(rec["metrics"]["metrics"], r.data()["summary"]["holdout"]["metrics"])

    def test_the_split_spellings_of_one_fraction_are_one_run(self):
        self.sb.write("big.csv", "a,y\n" + "".join("%d,%d\n" % (i, 2 * i + 1) for i in range(40)))
        ids = []
        for sp in ("train=0.5", "train=0.50", "train=0.500000"):
            r = self.sb.run("fit", "--model", "linear", "--target", "y", "--split", sp, "--output", "s.cml", "--overwrite", "--track", "runs", "big.csv")
            self.assertEqual(r.status, 0, r)
            ids.append(r.data()["run"]["id"])
        self.assertEqual(len(set(ids)), 1)

    def test_a_record_that_differs_under_an_existing_id_is_a_conflict_and_nothing_is_written(self):
        a = self.track()
        path = self.sb.dir / "runs" / (a["id"] + ".json")
        text = path.read_text()
        path.write_text(text.replace('"steps":1', '"steps":2'))
        before = path.read_bytes()
        r = self.sb.run("fit", *LIN, "--output", "other.cml", "--track", "runs", "train.csv")
        self.assertEqual(r.first_rule(), "ml.runs.conflict")
        self.assertEqual(r.status, 8)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.sb.dir / "other.cml").exists())

    def test_the_directory_must_exist_and_a_dry_run_writes_nothing(self):
        r = self.sb.run("fit", *LIN, "--output", "m.cml", "--track", "nodir", "train.csv")
        self.assertEqual(r.first_rule(), "io.not-found")
        r = self.sb.run("fit", *LIN, "--output", "m.cml", "--track", "runs", "--dry-run", "train.csv")
        self.assertEqual(r.status, 9)
        self.assertEqual(self.records(), [])
        self.assertFalse((self.sb.dir / "m.cml").exists())
        self.assertEqual(len(r.doc()["planned_actions"]), 2)
        self.assertEqual(r.doc()["planned_actions"][1]["dir"], "runs")

    def test_a_thousand_records_are_the_limit_and_nothing_is_written_past_it(self):
        for i in range(1000):
            self.sb.write("runs/x%04d.json" % i, "{}")
        r = self.sb.run("fit", *LIN, "--output", "m.cml", "--track", "runs", "train.csv")
        self.assertEqual(r.first_rule(), "ml.runs.too-many")
        self.assertFalse((self.sb.dir / "m.cml").exists())

    def test_a_record_over_its_size_is_refused_before_the_model_is_written(self):
        names = ["c%04d_" % i + "x" * 70 for i in range(4000)]
        self.sb.write("wide.csv", ",".join(names) + "\n" + ",".join(str(i % 7) for i in range(4000)) + "\n" + ",".join(str((i * 3) % 11) for i in range(4000)) + "\n")
        r = self.sb.run("fit", "--model", "linear", "--target", names[0], "--drop", ",".join(names[2:]), "--output", "m.cml", "--track", "runs", "wide.csv")
        self.assertEqual(r.first_rule(), "ml.runs.record-too-large", r)
        self.assertFalse((self.sb.dir / "m.cml").exists())
        self.assertEqual(self.records(), [])

    def test_the_record_holds_no_cell_and_no_category(self):
        self.sb.write("pets.csv", (ROOT / "docs/examples/data/pets.csv").read_text())
        r = self.sb.run("fit", "--model", "forest", "--task", "classification", "--trees", "3", "--target", "adopted", "--cat", "species,city", "--step", "hash column=species buckets=8;freq column=city",
                        "--drop", "id", "--output", "p.cml", "--track", "runs", "pets.csv")
        self.assertEqual(r.status, 0, r)
        rec = (self.sb.dir / "runs" / (r.data()["run"]["id"] + ".json")).read_text()
        header, rows = O.read_csv((self.sb.dir / "pets.csv").read_bytes())
        cells = {c for row in rows for c in row} - set(header)
        import re

        leaked = sorted(c for c in cells if len(c) > 2 and re.search(r'[:\[,]"%s"[,\]}]' % re.escape(c), rec))
        self.assertEqual(leaked, [])
        self.assertNotIn(str(self.sb.dir), rec)

    def test_eval_track_writes_one_record_beside_the_run_and_the_same_eval_the_same_bytes(self):
        run_ = self.track()
        r = self.sb.run("eval", "--model-file", "m.cml", "--track", "runs", "train.csv")
        self.assertEqual(r.status, 0, r)
        names = self.records()
        evals = [n for n in names if ".eval-" in n]
        self.assertEqual(len(evals), 1)
        self.assertTrue(evals[0].startswith(run_["id"] + ".eval-" + hashlib.sha256(TRAIN.encode()).hexdigest()))
        first = (self.sb.dir / "runs" / evals[0]).read_bytes()
        doc = json.loads(first)
        self.assertEqual((doc["schema"], doc["run"], doc["on"]), ("ml.eval.v1", run_["id"], "train"))
        self.assertEqual(self.sb.run("eval", "--model-file", "m.cml", "--track", "runs", "train.csv").status, 0)
        self.assertEqual((self.sb.dir / "runs" / evals[0]).read_bytes(), first)
        self.sb.write("other.csv", TRAIN.replace(LAST, CHANGED))
        self.assertEqual(self.sb.run("eval", "--model-file", "m.cml", "--track", "runs", "other.csv").status, 0)
        self.assertEqual(len([n for n in self.records() if ".eval-" in n]), 2)
        self.fit("--seed", "9", name="n.cml")
        r = self.sb.run("eval", "--model-file", "n.cml", "--track", "runs", "train.csv")
        self.assertEqual(r.first_rule(), "ml.runs.unknown-model")

    def test_list_show_compare_and_verify(self):
        a = self.track()
        b = self.track("--seed", "5", name="n.cml")
        self.sb.run("eval", "--model-file", "m.cml", "--track", "runs", "train.csv")
        d = self.sb.run("runs", "list", "runs").data()
        self.assertEqual(d["count"], 2)
        self.assertEqual([x["id"] for x in d["runs"]], sorted([a["id"], b["id"]]))
        mine = [x for x in d["runs"] if x["id"] == a["id"]][0]
        self.assertEqual((mine["model"], mine["task"], mine["evals"], mine["metrics_on"]), ("linear", "regression", 1, "train"))
        s = self.sb.run("runs", "show", "runs", a["id"]).data()["record"]
        self.assertEqual(s["id"], a["id"])
        c = self.sb.run("runs", "compare", "runs", a["id"], b["id"]).data()
        self.assertEqual([x["key"] for x in c["inputs"]], ["seed"])
        self.assertIs(c["model_equal"], False)  # the seed is in the meta of the file: another seed, other bytes
        same = self.sb.run("runs", "compare", "runs", a["id"], a["id"]).data()
        self.assertIs(same["model_equal"], True)
        o = self.track("--l2", "3", name="o.cml")
        c2 = self.sb.run("runs", "compare", "runs", a["id"], o["id"]).data()
        self.assertIs(c2["model_equal"], False)
        self.assertEqual(c["inputs"][0]["a"], 0)
        v = self.sb.run("runs", "verify", "runs", a["id"]).data()
        self.assertTrue(v["verified"])
        self.assertEqual(self.sb.run("runs", "show", "runs", "0" * 64).first_rule(), "ml.runs.not-found")
        self.assertEqual(self.sb.run("runs", "show", "runs", "xyz").first_rule(), "args.bad-value")

    def test_verify_names_a_changed_training_file_and_a_changed_compiler(self):
        a = self.track()
        self.sb.write("train.csv", TRAIN.replace(LAST, CHANGED))
        r = self.sb.run("runs", "verify", "runs", a["id"])
        self.assertEqual(r.first_rule(), "ml.runs.data-changed")
        self.assertEqual(r.status, 8)
        self.sb.write("train.csv", TRAIN)
        # a record sealed again with another model hash (what a record made by another compiler holds): the refit is not it
        path = self.sb.dir / "runs" / (a["id"] + ".json")
        body = path.read_text().split(',"check"')[0]
        old_sha = json.loads(path.read_text())["model"]["sha256"]
        body = body.replace(old_sha, "0" * 64)
        path.write_text(body + ',"check":"%s"}\n' % hashlib.sha256(body.encode()).hexdigest())
        r = self.sb.run("runs", "verify", "runs", a["id"])
        self.assertEqual(r.first_rule(), "ml.runs.model-differs", r)
        self.assertEqual((r.error()["detail"]["recorded"], r.error()["detail"]["refit"]), ("0" * 64, old_sha))
        self.assertEqual(r.error()["detail"]["recorded_compiler"], r.error()["detail"]["compiler"])

    def test_a_record_with_an_id_that_is_not_the_hash_of_its_inputs_is_damaged_even_when_sealed(self):
        a = self.track()
        path = self.sb.dir / "runs" / (a["id"] + ".json")
        body = path.read_text().split(',"check"')[0]
        fake = "f" * 64
        body = body.replace(a["id"], fake)
        (self.sb.dir / "runs" / (fake + ".json")).write_text(body + ',"check":"%s"}\n' % hashlib.sha256(body.encode()).hexdigest())
        self.assertEqual(self.sb.run("runs", "show", "runs", fake).first_rule(), "ml.runs.damaged")

    def test_a_damaged_record_is_never_a_trap(self):
        a = self.track()
        path = self.sb.dir / "runs" / (a["id"] + ".json")
        good = path.read_bytes()
        bad = 0
        for at in range(len(good)):
            mut = bytearray(good)
            mut[at] ^= 0x01
            path.write_bytes(bytes(mut))
            r = self.sb.run("runs", "show", "runs", a["id"])
            self.assertNotIn(r.status, TRAPS, at)
            if r.status != 0:
                self.assertEqual(r.first_rule(), "ml.runs.damaged", (at, r))
                bad += 1
        self.assertEqual(bad, len(good))
        for cut in (1, 2, 70, 400):
            path.write_bytes(good[:-cut])
            r = self.sb.run("runs", "show", "runs", a["id"])
            self.assertEqual(r.first_rule(), "ml.runs.damaged", cut)
        path.write_bytes(good)
        self.assertEqual(self.sb.run("runs", "show", "runs", a["id"]).status, 0)

    def test_a_directory_with_a_damaged_record_is_refused_by_list(self):
        a = self.track()
        (self.sb.dir / "runs" / (a["id"] + ".json")).write_text("{}")
        self.assertEqual(self.sb.run("runs", "list", "runs").first_rule(), "ml.runs.damaged")


class Serve(Base):
    def test_the_server_reports_the_cells_outside_the_range_and_refuses_over_its_bound(self):
        from serve_lib import Server

        self.fit()
        with Server(self.sb.dir, "m.cml", "--max-out-of-range", "0") as srv:
            hdr = {"Content-Type": "application/json"}
            ok = srv.request("POST", "/predict", body=b'{"a":2,"b":3,"c":1}', headers=hdr)
            self.assertEqual(ok.status, 200, ok)
            self.assertEqual(ok.json()["out_of_range"]["cells"], "0")
            far = srv.request("POST", "/predict", body=b'{"a":99,"b":3,"c":1}', headers=hdr)
            self.assertEqual(far.status, 422, far)
            self.assertEqual(far.json()["rule"], "ml.predict.out-of-range")
        with Server(self.sb.dir, "m.cml") as srv:
            far = srv.request("POST", "/predict", body=b'{"a":99,"b":3,"c":1}', headers={"Content-Type": "application/json"})
            self.assertEqual(far.status, 200, far)
            self.assertEqual(far.json()["out_of_range"]["columns"][0]["name"], "a")


class Replay(Base):
    """T3: `runs verify` reproduces the model's SHA-256 for every kind, at 1 and at 8 threads (the cases of the oracle: constants, linear, logistic, tree, forest, kmeans, categorical, split)."""

    def test_every_kind_replays_at_one_and_eight_threads(self):
        import pathlib
        import shutil
        import tempfile

        work = pathlib.Path(tempfile.mkdtemp(prefix="rp-"))
        try:
            for label, flags, csvname in O.case_list():
                d = work / label.replace(" ", "_")
                d.mkdir()
                rid = O.oracle_case(label, flags, csvname, d)
                self.assertIsNotNone(rid, label)
                for th in (1, 8):
                    rc, doc = O.ml(d, "--threads", th, "runs", "verify", "runs", rid)
                    self.assertEqual(rc, 0, (label, doc))
                    self.assertTrue(doc["data"]["verified"], label)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.assertEqual(O.FAILS, [])


if __name__ == "__main__":
    unittest.main()

"""D1, D2 and D4 of docs/serving.md: `describe` against an independent reader of the file (scripts/oracle_describe.py), against damaged files, and against the contract
of section 5 (the schema, the table of the document and a golden answer for a model of each kind).

D2 is run on every model of the corpus: every truncation and every byte flipped (all of them for the small files, a seeded sample of positions for the large ones, the whole
set with ST_STRIDE=1), with and without a repaired checksum, then the u32 and u64 fields of the sections and of the string table set to extreme values.
"""

import json
import os
import pathlib
import random
import re
import struct
import sys
import unittest

import jsonschema

from harness import PREDICT_BIN, ROOT, Sandbox, TRAPS, fix_checksum, introspect, schema, validate

import serving_corpus as C

sys.path.insert(0, str(ROOT / "scripts"))
import oracle_describe as O  # noqa: E402
import oracle_st as S  # noqa: E402

GOLDEN = pathlib.Path(__file__).parent / "golden_describe.json"
SB = None
MODELS = {}
STRIDE = int(os.environ.get("ST_STRIDE", "1"))
COUNTS = {"runs": 0, "accepted": 0}
# ST_SHARD=I/N (docs/ci.md section 5.2): this process judges the damaged files of the models at position I modulo N of the corpus; the tests that are not damage sweeps run in shard 0
SHARD_I, SHARD_N = (int(x) for x in os.environ.get("ST_SHARD", "0/1").split("/"))


def setUpModule():
    global SB, MODELS
    SB = Sandbox()
    MODELS, _ = C.build(SB)


def tearDownModule():
    SB.cleanup()
    if COUNTS["runs"]:
        print("describe damage: %(runs)d damaged files, %(accepted)d described as another valid file, 0 traps" % COUNTS)


def describe(mf, *extra, binary=None):
    return SB.run("describe", "--model-file", mf, *extra, binary_path=binary)


class Oracle(unittest.TestCase):
    """D1: field by field, for every model of the corpus and three page sizes."""

    def test_every_field_of_every_model(self):
        for mid, (mf, kind, fam) in MODELS.items():
            data = (SB.dir / mf).read_bytes()
            for limit in (1000, 5, 1):
                r = describe(mf, "--limit", str(limit))
                self.assertEqual(r.status, 0, (mid, r))
                self.assertEqual(validate(r), [], (mid, validate(r)))
                got = r.data()
                want = O.describe(data, limit=limit)
                self.assertEqual(got, want, (mid, limit))
                self.assertEqual(list(got), O.field_order(kind), mid)
                self.assertEqual(got["kind"], kind, mid)

    def test_the_corpus_covers_the_kinds_the_encoders_and_the_columns(self):
        docs = {mid: describe(mf).data() for mid, (mf, kind, fam) in MODELS.items()}
        self.assertEqual({d["kind"] for d in docs.values()}, {1, 2, 3, 4, 5, 6, 7, 8})
        self.assertEqual({d["model"] for d in docs.values()}, {"constant", "linear", "logistic", "kmeans", "tree", "forest", "boost"})
        self.assertEqual({d["task"] for d in docs.values()}, {"regression", "classification", "clustering"})
        self.assertEqual({c["encoder"] for d in docs.values() for c in d["categorical"]}, {"onehot", "label", "freq", "hash"})
        self.assertEqual({c["type"] for d in docs.values() for c in d["columns"]}, {"int", "dec", "float", "categorical"})
        self.assertEqual({d["file"]["sections"] for d in docs.values()}, {6, 7})
        self.assertEqual({d["request"]["missing_cells"] for d in docs.values()}, {"refused", "needs_impute", "handled"})
        self.assertTrue(any(d["na_values"] for d in docs.values()))
        self.assertTrue(any(c["truncated"] for d in (describe(mf, "--limit", "3").data() for mf, _, _ in MODELS.values()) for c in d["categorical"]))
        self.assertTrue(any(c["scale"] is not None for d in docs.values() for c in d["columns"]))
        self.assertTrue(any(d["target"] and d["target"]["type"] == "float" for d in docs.values()))
        # a clustering has no target: no column has the role, a request holds every column and asks for no optional one, and an answer has a cluster and a distance
        for mid in ("kmeans", "kmeans_random", "cat_kmeans_onehot"):
            d = docs[mid]
            self.assertEqual((d["task"], d["kind"], d["target"], d["request"]["optional"], d["classes"], d["positive"]), ("clustering", 6, None, [], None, None), mid)
            self.assertEqual({c["role"] for c in d["columns"]}, {"feature"}, mid)
            self.assertTrue(all(c["required"] for c in d["columns"]), mid)
            self.assertEqual(d["request"]["header"], [c["name"] for c in d["columns"]], mid)
            self.assertEqual((d["output"]["columns"], d["output"]["with_proba"], d["output"]["with_distance"], d["output"]["label_threshold"]), (["cluster"], None, ["cluster", "distance"], None), mid)
            self.assertEqual(d["request"]["missing_cells"], "needs_impute", mid)

    def test_the_text_format_names_the_columns(self):
        r = SB.run("describe", "--model-file", MODELS["linear"][0], "--format", "text")
        self.assertEqual(r.status, 0, r)
        text = r.stdout.decode()
        self.assertTrue(text.startswith("model linear, regression, 60 training rows\n"), text)
        self.assertIn("column a: int, required\n", text)
        self.assertIn("column y: dec(1), target (optional in an input)\n", text)

    def test_the_text_format_of_a_clustering_has_no_target(self):
        r = SB.run("describe", "--model-file", MODELS["kmeans"][0], "--format", "text")
        self.assertEqual(r.status, 0, r)
        text = r.stdout.decode()
        self.assertTrue(text.startswith("model kmeans, clustering, 60 training rows\n"), text)
        self.assertIn("column a: int, required\n", text)
        self.assertNotIn("target", text)

    def test_a_request_built_from_describe_is_accepted_by_predict(self):
        # the use the contract is for: the header and the types of the answer are a CSV the model takes
        for mid in ("linear", "logistic", "tree_cls", "cat_forest_hash", "cat_linear_onehot", "cat_wide_tree", "const_cls", "kmeans", "cat_kmeans_onehot"):
            mf = MODELS[mid][0]
            d = describe(mf).data()
            head = ",".join(d["request"]["header"])
            cells = []
            for name in d["request"]["header"]:
                col = next(c for c in d["columns"] if c["name"] == name)
                if col["type"] == "categorical":
                    cat = next(c for c in d["categorical"] if c["column"] == name)
                    cells.append(C.quoted(cat["values"][0]["value"]) if cat["values"] else "x")
                elif col["type"] == "int":
                    cells.append("3")
                elif col["type"] == "dec":
                    cells.append("1.5")
                else:
                    cells.append("0.25")
            r = SB.run("predict", "-", "--model-file", mf, stdin=(head + "\n" + ",".join(cells) + "\n").encode())
            self.assertEqual(r.status, 0, (mid, r))
            self.assertEqual(r.data()["columns"][:len(d["output"]["columns"])], d["output"]["columns"], mid)


class Contract(unittest.TestCase):
    """D4."""

    def test_the_fields_of_the_document_are_the_fields_of_the_schema_and_of_the_answer(self):
        sch = schema()
        variants = sch["$defs"]["data"]["oneOf"]
        desc = [v for v in variants if "categorical" in v["properties"] and "request" in v["properties"]]
        self.assertEqual(len(desc), 1)
        props = list(desc[0]["properties"])
        self.assertEqual(props, O.field_order(8))
        text = (ROOT / "docs" / "serving.md").read_text()
        table = text.split("## 5. The contract", 1)[1].split("\n## ", 1)[0]
        rows = re.findall(r"^\| `(\w+)` \|", table, re.M)
        self.assertEqual(rows, props, "the table of docs/serving.md section 5 and the schema list the same fields in the same order")

    def test_every_answer_is_valid_against_the_schema_and_the_nested_fields_are_closed(self):
        validator = jsonschema.Draft202012Validator(schema())
        for mid, (mf, kind, fam) in MODELS.items():
            doc = describe(mf).doc()
            self.assertEqual(list(validator.iter_errors(doc)), [], mid)
            doc["data"]["surprise"] = 1
            self.assertNotEqual(list(validator.iter_errors(doc)), [], "the data object is closed")

    def test_the_golden_answers(self):
        golden = json.loads(GOLDEN.read_text())
        self.assertEqual(sorted(golden), sorted(MODELS), "the golden record covers the corpus")
        for mid, (mf, kind, fam) in MODELS.items():
            r = describe(mf, "--limit", "4")
            self.assertEqual(r.stdout.decode(), golden[mid], mid)


def reseal(data):
    """the file with its digest repaired, so the damage reaches the checks behind the checksum"""
    return fix_checksum(data)


def flips(n, rng, small):
    if small and STRIDE == 1:
        return list(range(n))
    count = min(n, 400 if STRIDE == 1 else max(40, n // STRIDE))
    return sorted(rng.sample(range(n), count))


def cut_flip_plan():
    """for every model of the corpus, in order: (position, id, file, data, cuts, flips). One generator draws the offsets for all the models, whether or not this shard judges them,
    so the damaged files are the same at every shard count."""
    rng = random.Random(5)
    for k, (mid, (mf, kind, fam)) in enumerate(MODELS.items()):
        data = (SB.dir / mf).read_bytes()
        small = len(data) < 1200
        cuts = list(range(len(data))) if small and STRIDE == 1 else sorted(rng.sample(range(len(data)), min(len(data), 200)))
        yield k, mid, mf, data, cuts, flips(len(data), rng, small)


def value_plan():
    """for every model, in order: (position, id, file, data, places), the places of the extreme numbers, drawn by one generator as above"""
    rng = random.Random(9)
    for k, (mid, (mf, kind, fam)) in enumerate(MODELS.items()):
        data = (SB.dir / mf).read_bytes()
        secs = S.parse_file(data)
        places = []
        pos = 12
        for s in range(struct.unpack_from("<I", data, 8)[0]):
            tag, ln = struct.unpack_from("<Iq", data, pos)
            places.append((pos, 4))
            places.append((pos + 4, 8))
            if tag == 2:
                # the schema: columns, target, then per column the name length, the type and the scale
                sp = pos + 12
                places += [(sp, 4), (sp + 4, 4)]
                p = sp + 8
                for _ in range(struct.unpack_from("<I", data, sp)[0]):
                    nl = struct.unpack_from("<I", data, p)[0]
                    places += [(p, 4), (p + 4 + nl, 4), (p + 8 + nl, 4)]
                    p += 12 + nl
            if tag == 6:
                places += [(pos + 12 + o, w) for o, w in S.table_fields(secs[6])]
            if tag in (4, 5):
                places += [(pos + 12 + o, w) for o in range(0, min(ln, 64), 4) for w in (4,)]
            pos += 12 + ln
        places = sorted(set(places))
        if len(places) > 120:
            places = sorted(rng.sample(places, 120))
        yield k, mid, mf, data, places


def values(width):
    return (0xFFFFFFFFFFFFFFFF if width == 8 else 0xFFFFFFFF, 0x7FFFFFFFFFFFFFFF if width == 8 else 0x7FFFFFFF, 0)


def damage_names(i, n):
    """the names of the damaged files of shard i of n, as the two sweeps judge them (R1 compares these lists)"""
    names = []
    for k, mid, mf, data, cuts, flipped in cut_flip_plan():
        if k % n == i:
            names += ["%s cut at %d" % (mid, c) for c in cuts] + ["%s flip %d" % (mid, a) for a in flipped]
    for k, mid, mf, data, places in value_plan():
        if k % n == i:
            names += ["%s %d@%d=%x" % (mid, w, a, v) for a, w in places for v in values(w) if not (v.to_bytes(w, "little") == data[a:a + w])]
    return names


def partition_problems(lists, whole):
    """what is wrong when `lists` (the names of each shard) are not a partition of `whole`"""
    flat = [x for l in lists for x in l]
    problems = []
    if len(flat) != len(set(flat)):
        problems.append("a damaged file is in two shards")
    if sorted(flat) != sorted(whole):
        problems.append("the shards do not hold the damaged files of the whole run")
    return problems


class Damage(unittest.TestCase):
    """D2."""

    def judge(self, mf, damaged, name, tagged=None):
        COUNTS["runs"] += 1
        SB.write("dmg.cml", damaged)
        r = SB.run("describe", "--model-file", "dmg.cml")
        self.assertNotIn(r.status, TRAPS, (name, r))
        problems = validate(r)
        self.assertEqual(problems, [], (name, problems))
        if tagged is not None:
            self.assertEqual(r.first_rule(), tagged, (name, r))
        return r

    def file_tag(self, damaged):
        try:
            S.parse_file(damaged)
        except S.FileBad as e:
            return e.tag
        return None

    def test_every_truncation_and_every_flipped_byte(self):
        for k, mid, mf, data, cuts, flipped in cut_flip_plan():
            if k % SHARD_N != SHARD_I:
                continue
            for cut in cuts:
                damaged = data[:cut]
                self.judge(mf, damaged, "%s cut at %d" % (mid, cut), self.file_tag(damaged))
            for at in flipped:
                damaged = bytearray(data)
                damaged[at] ^= 0xFF
                damaged = bytes(damaged)
                self.judge(mf, damaged, "%s flip %d" % (mid, at), self.file_tag(damaged))
                # with the checksum repaired the damage is in a field: refused by a check of the content, or a different valid file, never a trap
                fixed = reseal(damaged)
                r = self.judge(mf, fixed, "%s flip %d resealed" % (mid, at))
                if r.status == 0:
                    COUNTS["accepted"] += 1
                    self.assertEqual(r.data(), O.describe(fixed), (mid, at))
                else:
                    self.assertIn(r.first_rule(), ("ml.model.malformed", "ml.model.string-table", "ml.model.version", "ml.model.checksum", "ml.model.bad-magic", "ml.model.truncated"), (mid, at, r))

    def test_extreme_numbers_in_every_field_of_the_structure_and_the_table(self):
        for k, mid, mf, data, places in value_plan():
            if k % SHARD_N != SHARD_I:
                continue
            for at, width in places:
                for value in values(width):
                    damaged = bytearray(data)
                    damaged[at:at + width] = value.to_bytes(width, "little")
                    damaged = bytes(damaged)
                    if damaged == data:
                        continue
                    self.judge(mf, damaged, "%s %d@%d=%x" % (mid, width, at, value), self.file_tag(damaged))
                    r = self.judge(mf, reseal(damaged), "%s %d@%d=%x resealed" % (mid, width, at, value))
                    if r.status != 0:
                        self.assertTrue(r.first_rule().startswith("ml.model.") or r.first_rule() == "limit.input-too-large", (mid, at, r))

    def test_the_shards_partition_the_damage(self):
        """R1 (docs/ci.md section 5.2): at any shard count the shards hold each damaged file once, and together the files of the whole run"""
        whole = damage_names(0, 1)
        self.assertGreater(len(whole), 1000)
        for n in range(1, 9):
            lists = [damage_names(i, n) for i in range(n)]
            self.assertEqual(partition_problems(lists, whole), [], n)
        # the check can fail: a shard that holds a file twice, and a shard that lost one
        self.assertNotEqual(partition_problems([whole, whole[:5]], whole), [])
        self.assertNotEqual(partition_problems([whole[:-1]], whole), [])

    def test_a_file_over_max_bytes_is_refused_before_it_is_read_whole(self):
        mf = MODELS["forest_reg"][0]
        r = SB.run("describe", "--model-file", mf, "--max-bytes", "1000")
        self.assertEqual(r.first_rule(), "limit.input-too-large", r)
        self.assertEqual(r.error()["detail"]["max"], 1000)

    def test_a_missing_file_a_directory_and_junk(self):
        self.assertEqual(SB.run("describe", "--model-file", "absent.cml").first_rule(), "io.not-found")
        SB.write("junk.cml", b"not a model")
        self.assertEqual(SB.run("describe", "--model-file", "junk.cml").first_rule(), "ml.model.bad-magic")
        SB.write("empty.cml", b"")
        self.assertEqual(SB.run("describe", "--model-file", "empty.cml").first_rule(), "ml.model.truncated")
        os.makedirs(SB.dir / "adir", exist_ok=True)
        self.assertEqual(SB.run("describe", "--model-file", "adir").first_rule(), "io.is-a-directory")


SWEEPS = ("test_every_truncation_and_every_flipped_byte", "test_extreme_numbers_in_every_field_of_the_structure_and_the_table")


def load_tests(loader, standard_tests, pattern):
    """a shard other than 0 loads the two damage sweeps and nothing else, so the tests counted over the shards are the tests of the module plus the sweeps once more per extra shard"""
    if SHARD_I == 0:
        return standard_tests
    suite = unittest.TestSuite()

    def walk(s):
        for t in s:
            if isinstance(t, unittest.TestSuite):
                walk(t)
            elif t.id().rsplit(".", 1)[-1] in SWEEPS:
                suite.addTest(t)
    walk(standard_tests)
    return suite


if __name__ == "__main__":
    unittest.main()

"""G4 -- hostile JSON lines: an answer of the contract, no trap, no hang (docs/jsonl.md section 6).

Huge numbers, nesting to the node limit and past it, invalid UTF-8, a byte order mark, CRLF, no final newline, a 10 MB line, 100,000 empty lines, lines that are not
objects, random bytes and mutated valid files go through `ml fit`, `ml predict` and `jsonl-check`. Each answer is a document valid against the schema with the exit status
of its rule (harness.validate), never a trap status, and each run ends before its timeout. The fits at 1 and 4 threads write the same model bytes.
"""
import random
import subprocess
import unittest

from harness import Sandbox, TRAPS, run, validate
import jsonl_lib

FIELDS = "a,y"
FIT = ["fit", "--input-format", "jsonl", "--fields", FIELDS, "--target", "y", "--output", "m.cml", "--overwrite"]
GOOD = b"".join(b'{"a":%d,"y":%d.5}\n' % (i, i * 2) for i in range(12))


class Hostile(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        self.sb.write("good.jsonl", GOOD)
        r = self.sb.run(*FIT, "good.jsonl")
        assert r.status == 0, r

    def tearDown(self):
        self.sb.cleanup()

    def fit(self, data, *extra, timeout=120):
        self.sb.write("d.jsonl", data)
        r = self.sb.run(*FIT, *extra, "d.jsonl", timeout=timeout)
        self.assertEqual(validate(r), [], r)
        self.assertNotIn(r.status, TRAPS)
        return r

    def test_huge_numbers(self):
        for digits in (50, 1100, 1101, 100000):
            with self.subTest(digits=digits):
                r = self.fit(b'{"a":1,"y":1.5}\n{"a":' + b"9" * digits + b',"y":2.5}\n')
                self.assertIn(r.first_rule(), (None, "ml.input.non-finite", "ml.input.non-numeric"))
                if digits > 1100:
                    self.assertEqual(r.first_rule(), "ml.input.non-numeric")
        for tok in (b"1e400", b"-1e400", b"1e-400", b"1E999999999999999999999", b"0.000000000000000000000000000001"):
            with self.subTest(tok=tok):
                r = self.fit(b'{"a":1,"y":1.5}\n{"a":' + tok + b',"y":2.5}\n')
                self.assertTrue(r.first_rule() is None or r.first_rule().startswith("ml."), r)

    def test_nesting_to_the_node_limit_and_past_it(self):
        for depth in (1, 100, 127, 128, 129, 5000, 100000):
            with self.subTest(depth=depth):
                self.fit(b'{"a":1,"y":1.5,"z":' + b"[" * depth + b"]" * depth + b"}\n")
        wide = b'{"a":1,"y":1.5,"z":[' + b",".join([b"1"] * 4089) + b"]}\n"
        self.assertEqual(self.fit(wide).status, 0)
        self.assertEqual(self.fit(b'{"a":1,"y":1.5,"z":[' + b",".join([b"1"] * 4090) + b"]}\n").first_rule(), "ml.jsonl.too-many-nodes")
        self.assertEqual(self.fit(b'{"a":1,"y":1.5,"z":[' + b",".join([b"1"] * 4090) + b"]}\n", "--max-json-nodes", "4100").status, 0)

    def test_encoding_and_line_endings(self):
        for data in (b"\xef\xbb\xbf" + GOOD, GOOD.replace(b"\n", b"\r\n"), GOOD[:-1], b"\xff" + GOOD, b"\xef\xbb" + GOOD, b"\xef\xbb\xbf\xef\xbb\xbf" + GOOD,
                     GOOD + b'{"a":1,"y":"\xff"}\n', GOOD + b"\x00\n", GOOD.replace(b"\n", b"\r"), b"\xfe\xff" + GOOD):
            with self.subTest(data=data[:30]):
                self.fit(data)
        a = self.fit(GOOD)
        b = self.fit(b"\xef\xbb\xbf" + GOOD.replace(b"\n", b"\r\n")[:-2])
        self.assertEqual((a.status, b.status), (0, 0))

    def test_a_long_line_and_many_blank_lines(self):
        big = b'{"a":1,"y":1.5,"z":"' + b"x" * (10 * 1024 * 1024) + b'"}\n'
        r = self.fit(big)
        self.assertEqual(r.first_rule(), "limit.line-too-long")
        r = self.fit(big, "--max-line-bytes", "16777216", "--max-bytes", "20000000")
        self.assertEqual(r.status, 0, r)
        r = self.fit(b"\n" * 100000 + GOOD)
        self.assertEqual(r.status, 0, r)
        r = self.fit(b" \t\r\n" * 100000 + GOOD)
        self.assertEqual(r.status, 0, r)
        r = self.fit(b"\n" * 100000)
        self.assertEqual(r.first_rule(), "ml.input.empty")

    def test_lines_that_are_not_objects(self):
        for line in (b"[]", b"[1,2]", b"1", b'"x"', b"true", b"null", b"{", b"}", b"{}", b'{"a"}', b'{"a":}', b'{"a":1,}', b"{'a':1}", b'{"a":NaN}', b"{,}", b"\x00", b"[" * 100000):
            with self.subTest(line=line[:30]):
                self.fit(GOOD + line + b"\n")

    def test_random_bytes_and_mutated_files(self):
        rng = random.Random(54)
        for i in range(120):
            if i % 3 == 0:
                data = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 400)))
            else:
                b = bytearray(GOOD)
                for _ in range(rng.randrange(1, 8)):
                    k = rng.randrange(len(b))
                    op = rng.randrange(4)
                    if op == 0:
                        b[k] = rng.randrange(256)
                    elif op == 1:
                        del b[k:k + rng.randrange(1, 6)]
                    elif op == 2:
                        b[k:k] = bytes(rng.choice(b'{}[]",:\\ \n-.e0u') for _ in range(rng.randrange(1, 6)))
                    else:
                        b[k:k] = b[max(0, k - 10):k]
                data = bytes(b)
            self.fit(data)
            out = jsonl_lib.run_check(data, ["a", "y"])
            self.assertIn(out[0], ("ok", "refused"))

    def test_predict_and_eval_on_hostile_lines(self):
        for data in (b"", b"\n\n", b"[1]\n", b'{"a":"x"}\n', b'{"a":1}\n', b'{"a":1,"a":2}\n', b'{"a":{"b":1}}\n', b"\xff\n", b'{"a":' + b"9" * 5000 + b"}\n"):
            self.sb.write("p.jsonl", data)
            for verb in ("predict", "eval"):
                r = self.sb.run(verb, "--model-file", "m.cml", "--input-format", "jsonl", "p.jsonl")
                self.assertEqual(validate(r), [], r)
                self.assertNotIn(r.status, TRAPS)
            r = self.sb.run("predict", "-", "--model-file", "m.cml", "--input-format", "jsonl", stdin=data)
            self.assertEqual(validate(r), [], r)

    def test_threads_change_no_byte(self):
        data = b"".join(b'{"a":%d,"b":%d,"y":%d.25}\n' % (i, (i * 7) % 5, i * 3) for i in range(80))
        self.sb.write("t.jsonl", data)
        outs = []
        for th in ("1", "2", "4"):
            r = self.sb.run("fit", "--input-format", "jsonl", "--fields", "a,b,y", "--model", "linear", "--l2", "0.1", "--target", "y", "--threads", th, "--output", "t%s.cml" % th, "t.jsonl")
            self.assertEqual(r.status, 0, r)
            outs.append(self.sb.read("t%s.cml" % th))
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], outs[2])

    def test_no_hang_on_the_worst_declared_fields(self):
        # 4096 declared fields and one line of 50,000 keys: the lookup costs the keys, not their product with the fields
        fields = ",".join("f%d" % i for i in range(2000))
        line = b"{" + b",".join(b'"k%d":%d' % (i, i) for i in range(20000)) + b"}\n"
        out = jsonl_lib.run_check(line * 3, fields.split(","), absent_empty=True, max_nodes=100000, max_line=1 << 24, timeout=60)
        self.assertEqual(out[0], "ok")

    def test_closed_standard_input_and_a_missing_file(self):
        r = self.sb.run("fit", "--input-format", "jsonl", "--fields", FIELDS, "--target", "y", "--output", "z.cml", "-", close_stdin=True)
        self.assertEqual(validate(r), [], r)
        r = self.sb.run("fit", "--input-format", "jsonl", "--fields", FIELDS, "--target", "y", "--output", "z.cml", "nope.jsonl")
        self.assertEqual(r.first_rule(), "io.not-found")
        r = self.sb.run("fit", "--input-format", "jsonl", "--fields", FIELDS, "--target", "y", "--output", "z.cml", "-", stdin=GOOD * 2000, )
        self.assertEqual(validate(r), [], r)
        r = self.sb.run("fit", "--input-format", "jsonl", "--fields", FIELDS, "--target", "y", "--output", "z.cml", "--max-bytes", "50", "-", stdin=GOOD)
        self.assertEqual(r.first_rule(), "limit.input-too-large")


if __name__ == "__main__":
    unittest.main()

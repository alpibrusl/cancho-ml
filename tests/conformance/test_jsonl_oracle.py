"""G2 -- the cell text of every JSON value kind against Python's `json` (docs/jsonl.md sections 3 and 6).

`jsonl-check` prints the CSV that tools/ml/jsonl.cho makes; tests/conformance/jsonl_ref.py writes it again from the specification with Python's `json`. They are
compared byte for byte, on tables of cases for each kind of value and on seeded random lines. Standard library only.
"""
import json
import random
import unittest

import jsonl_lib
from jsonl_lib import check, run_check


def lines(*objs):
    return b"".join((o if isinstance(o, bytes) else o.encode("utf-8")) + b"\n" for o in objs)


class Numbers(unittest.TestCase):
    TOKENS = ["0", "-0", "1", "1.0", "-1", "12345678901234567", "0.1", "1e3", "1E3", "1E-7", "1e+7", "-0.0", "123456789012345678901234567890", "0.30000000000000004",
              "4.9e-324", "2.2250738585072014e-308", "1.7976931348623157e308", "1e400", "-1e400", "1e-400", "100000000000000000000", "5e0", "0.5E+1"]

    def test_the_exact_token_text(self):
        for tok in self.TOKENS:
            with self.subTest(tok=tok):
                got = check(self, lines('{"a":%s}' % tok), ["a"])
                self.assertEqual(got, ("ok", ("a\n" + tok + "\n").encode()))

    def test_a_number_is_never_rewritten(self):
        got = check(self, lines('{"a":1.50,"b":1E3}'), ["a", "b"])
        self.assertEqual(got[1], b"a,b\n1.50,1E3\n")

    def test_not_numbers_are_syntax_errors(self):
        for tok in ["01", "1.", ".5", "+1", "NaN", "Infinity", "-Infinity", "0x10", "1e", "--1", "1_000", "00", "-", "1.e3"]:
            with self.subTest(tok=tok):
                got = check(self, lines('{"a":%s}' % tok), ["a"])
                self.assertEqual(got[0], "refused")
                self.assertEqual(got[1][0], 1)


class Strings(unittest.TestCase):
    TEXTS = ["", "x", "a b", "a,b", 'a"b', 'a""b', "a\nb", "a\rb", "a\r\nb", "tab\there", "back\\slash", "café", "é", "中文", "\U0001f600", "\u0000", "a\u0000b",
             "  ", "﻿", " lead", "trail ", "007", "5", "true", "null", "NA", "\x7f", "\x1f"]

    def test_decoded_text_both_spellings(self):
        for text in self.TEXTS:
            for ascii_only in (True, False):
                with self.subTest(text=text, ascii=ascii_only):
                    line = json.dumps({"s": text}, ensure_ascii=ascii_only)
                    got = check(self, lines(line), ["s"], cats=["s"])
                    self.assertEqual(got[0], "ok")

    def test_surrogate_pairs_and_escapes_spelled_by_hand(self):
        for body, want in [(r"😀", "\U0001f600"), (r"é", "é"), (r"\/", "/"), (r"\b\f\n\r\t", "\b\f\n\r\t"), (r"AB", "AB"), (r"😀x", "\U0001f600x"),
                           (r"\u0000", "\u0000"), (r'\"', '"'), (r"\\", "\\")]:
            with self.subTest(body=body):
                got = check(self, lines('{"s":"%s"}' % body), ["s"], cats=["s"])
                self.assertEqual(got[0], "ok")
                self.assertIn(jsonl_lib.jsonl_ref.csv_cell(want).encode(), got[1])

    def test_lone_surrogates_and_bad_escapes_are_refused(self):
        for body in [r"\ud800", r"\udc00", r"\ud83dx", r"\ud83dA", r"\x41", r"\u12", r"\u12G4", "\\"]:
            with self.subTest(body=body):
                got = check(self, lines('{"s":"%s"}' % body), ["s"], cats=["s"])
                self.assertEqual((got[0], got[1][0]), ("refused", 1))

    def test_raw_control_and_invalid_utf8(self):
        for raw in [b'{"s":"a\x01b"}', b'{"s":"a\nb"}'.replace(b"\n", b"\x0b"), b'{"s":"\xff"}', b'{"s":"\xc0\x80"}', b'{"s":"\xed\xa0\x80"}', b'{"s":"\xf4\x90\x80\x80"}', b'{"s":"a\xe2\x82"}']:
            with self.subTest(raw=raw):
                got = check(self, raw + b"\n", ["s"], cats=["s"])
                self.assertEqual((got[0], got[1][0]), ("refused", 1))

    def test_a_string_in_a_numeric_column_is_text_not_a_number(self):
        for text in ["5", "5.0", "", "NA", "true"]:
            with self.subTest(text=text):
                got = check(self, lines(json.dumps({"a": text})), ["a"])
                self.assertEqual(got, ("refused", (7, 1, 1, 0, 1, 0, 0)))

    def test_keys_decoded(self):
        got = check(self, lines(r'{"a":1,"b":2}'), ["a", "b"])
        self.assertEqual(got[1], b"a,b\n1,2\n")
        got = check(self, lines(r'{"a":1,"a":2}'), ["a"])
        self.assertEqual(got, ("refused", (6, 1, 1, 0, 0, 0, 0)))


class Kinds(unittest.TestCase):
    def test_booleans_and_null_and_absent(self):
        got = check(self, lines('{"a":1,"b":true,"c":false,"d":null}'), ["a", "b", "c", "d"], cats=["b", "c", "d"])
        self.assertEqual(got[1], b"a,b,c,d\n1,true,false,\n")
        got = check(self, lines('{"a":1}'), ["a", "b"], absent_empty=True, cats=["b"])
        self.assertEqual(got[1], b"a,b\n1,\n")
        got = check(self, lines('{"a":1}'), ["a", "b"])
        self.assertEqual(got, ("refused", (4, 1, 1, 1, 0, 0, 0)))

    def test_a_boolean_in_a_numeric_column_is_refused_never_guessed(self):
        self.assertEqual(check(self, lines('{"a":true}'), ["a"]), ("refused", (7, 1, 1, 0, 2, 0, 0)))

    def test_nested_values_name_the_column(self):
        self.assertEqual(check(self, lines('{"a":{"x":1}}'), ["a"]), ("refused", (5, 1, 1, 0, 0, 0, 0)))
        self.assertEqual(check(self, lines('{"a":1,"b":[1]}'), ["a", "b"]), ("refused", (5, 1, 1, 1, 1, 0, 0)))
        self.assertEqual(check(self, lines('{"a":{},"b":[]}'), ["b"], absent_empty=True)[0], "refused")
        # a nested value under a key that is not declared is not looked at
        self.assertEqual(check(self, lines('{"a":1,"z":{"y":[1,2,{"x":null}]}}'), ["a"])[0], "ok")

    def test_mixed_kinds_in_a_text_column(self):
        data = lines('{"a":"x"}', '{"a":"y"}', '{"a":5}')
        self.assertEqual(check(self, data, ["a"], cats=["a"]), ("refused", (8, 3, 3, 0, 1, 1, 0)))
        data = lines('{"a":5}', "", '{"a":"5"}')
        self.assertEqual(check(self, data, ["a"], cats=["a"]), ("refused", (8, 3, 2, 0, 0, 1, 1)))
        data = lines('{"a":true}', '{"a":"true"}')
        self.assertEqual(check(self, data, ["a"], cats=["a"]), ("refused", (8, 2, 2, 0, 2, 1, 1)))
        # nulls and absent keys have no kind
        data = lines('{"a":null}', '{"a":"x"}', '{}', '{"a":"y"}')
        self.assertEqual(check(self, data, ["a"], cats=["a"], absent_empty=True)[0], "ok")

    def test_duplicates(self):
        self.assertEqual(check(self, lines('{"a":1,"a":2}'), ["a"]), ("refused", (6, 1, 1, 0, 0, 0, 0)))
        self.assertEqual(check(self, lines('{"a":1,"b":1,"a":null}'), ["a", "b"]), ("refused", (6, 1, 1, 0, 0, 0, 0)))
        # a repeated key that is not declared is allowed, wherever it is
        self.assertEqual(check(self, lines('{"a":1,"z":1,"z":{"q":1}}'), ["a"])[0], "ok")

    def test_roots_that_are_not_objects(self):
        for text, kind in [("[1]", 0), ('"s"', 1), ("5", 2), ("true", 3), ("false", 3), ("null", 4), ("[]", 0), ("[[]]", 0), ("1.5", 2)]:
            with self.subTest(text=text):
                self.assertEqual(check(self, lines(text), ["a"]), ("refused", (2, 1, 1, 0, kind, 0, 0)))

    def test_blank_lines_bom_crlf_and_the_last_line(self):
        data = b"\xef\xbb\xbf" + b'{"a":1}\r\n\r\n  \t \n{"a":2}\r\n{"a":3}'
        got = check(self, data, ["a"])
        self.assertEqual(got[1], b"a\n1\n\n\n2\n3\n")
        self.assertEqual(check(self, b"", ["a"])[1], b"a\n")
        self.assertEqual(check(self, b"\n\n\n", ["a"])[1], b"a\n\n\n\n")

    def test_one_column_with_an_empty_cell_is_not_a_blank_line(self):
        got = check(self, lines('{"a":null}', '{"a":""}', '{}'), ["a"], cats=["a"], absent_empty=True)
        self.assertEqual(got[1], b'a\n""\n""\n""\n')

    def test_order_of_the_fields_is_the_declaration(self):
        got = check(self, lines('{"b":2,"a":1,"c":3}'), ["c", "a", "b"])
        self.assertEqual(got[1], b"c,a,b\n3,1,2\n")

    def test_header_names_are_quoted_like_cells(self):
        got = check(self, lines('{"a b":1,"c\\"d":2}'), ["a b", 'c"d'])
        self.assertEqual(got[1], b'a b,"c""d"\n1,2\n')


class Limits(unittest.TestCase):
    def test_the_line_bound(self):
        line = b'{"a":"' + b"x" * 100 + b'"}'
        self.assertEqual(check(self, line + b"\n", ["a"], cats=["a"], max_line=len(line))[0], "ok")
        self.assertEqual(check(self, line + b"\n", ["a"], cats=["a"], max_line=len(line) - 1), ("refused", (9, 1, 0, 0, len(line), 0, 0)))
        # the bound counts the CR
        self.assertEqual(check(self, line + b"\r\n", ["a"], cats=["a"], max_line=len(line))[0], "refused")

    def test_the_node_bound_is_exact(self):
        # nodes: the object, 2 keys, 2 values = 5
        data = lines('{"a":1,"b":2}')
        self.assertEqual(check(self, data, ["a"], max_nodes=5)[0], "ok")
        self.assertEqual(check(self, data, ["a"], max_nodes=4), ("refused", (3, 1, 1, 0, 4, 0, 0)))
        data = lines('{"a":1,"z":[1,2,3]}')
        self.assertEqual(check(self, data, ["a"], max_nodes=8)[0], "ok")
        self.assertEqual(check(self, data, ["a"], max_nodes=7)[0], "refused")

    def test_depth(self):
        for depth, ok in [(127, True), (128, True), (129, False)]:
            inner = "[" * (depth - 1) + "]" * (depth - 1)
            data = lines('{"a":1,"z":%s}' % inner)  # the object is depth 1
            got = check(self, data, ["a"], max_nodes=4096)
            self.assertEqual(got[0], "ok" if ok else "refused", depth)

    def test_the_input_bound_and_the_csv_bound(self):
        data = lines('{"a":1}') * 10
        self.assertEqual(check(self, data, ["a"], max_bytes=len(data))[0], "ok")
        self.assertEqual(check(self, data, ["a"], max_bytes=len(data) - 1), ("refused", (10, 0, 0, 0, len(data) - 1, 0, 0)))
        fields = ["f%d" % i for i in range(200)]
        data = b"{}\n" * 100
        got = check(self, data, fields, absent_empty=True, max_bytes=len(data) + 10)
        self.assertEqual(got[0], "refused")
        self.assertEqual(got[1][0], 10)
        self.assertEqual(got[1][5], 1)


class Random(unittest.TestCase):
    """seeded random lines: random keys, kinds and spellings; the driver and the reference agree on every one, accepted or refused"""

    WORDS = ["", "a", "b c", "x,y", 'q"r', "line\nbreak", "é中", "\U0001f600", "007", "5", "null", "NA", "\u0000", "tab\t"]
    NUMS = ["0", "-0", "7", "7.0", "1e3", "-12.5", "12345678901234567890", "0.1", "1E-7", "3.14159"]

    def value(self, rng, depth=0):
        k = rng.randrange(12)
        if k < 4:
            return rng.choice(self.NUMS)
        if k < 7:
            return json.dumps(rng.choice(self.WORDS), ensure_ascii=rng.random() < 0.5)
        if k == 7:
            return rng.choice(["true", "false"])
        if k == 8:
            return "null"
        if depth < 3 and k == 9:
            return "[" + ",".join(self.value(rng, depth + 1) for _ in range(rng.randrange(3))) + "]"
        if depth < 3 and k == 10:
            return "{" + ",".join('"%s":%s' % (rng.choice("pqr"), self.value(rng, depth + 1)) for _ in range(rng.randrange(3))) + "}"
        return rng.choice(self.NUMS)

    def test_seeded(self):
        rng = random.Random(54)
        names = ["a", "b", "c", "d e", "f;g"]
        for case in range(250):
            fields = rng.sample(names, rng.randrange(1, 4))
            cats = [f for f in fields if rng.random() < 0.5]
            # a line both past the node bound and malformed further on is the node bound's (the parser fills the tape as it goes, docs/jsonl.md): the reference cannot tell, so
            # the small bounds are used on well-formed lines only
            max_nodes = rng.choice([4096, 12, 6])
            out = []
            for _ in range(rng.randrange(1, 6)):
                keys = [k for k in names + ["z", "y"] if rng.random() < 0.8]
                if rng.random() < 0.05 and keys:
                    keys.append(rng.choice(keys))
                rng.shuffle(keys)
                kind = rng.random()
                # mostly clean columns so that long accepted runs happen: a numeric field gets a number, a text field a string
                parts = []
                for k in keys:
                    if k in fields and rng.random() < 0.9:
                        v = json.dumps(rng.choice(self.WORDS), ensure_ascii=False) if k in cats else rng.choice(self.NUMS)
                    else:
                        v = self.value(rng)
                    parts.append(json.dumps(k) + ":" + v)
                line = "{" + ",".join(parts) + "}"
                if kind < 0.03 and max_nodes == 4096:
                    line = line[:-1]
                elif kind < 0.05:
                    line = "  " + line + "  "
                out.append(line)
                if rng.random() < 0.1:
                    out.append("")
            data = ("\r\n" if rng.random() < 0.2 else "\n").join(out).encode("utf-8") + (b"\n" if rng.random() < 0.7 else b"")
            check(self, data, fields, cats=cats, absent_empty=rng.random() < 0.5, max_nodes=max_nodes)


if __name__ == "__main__":
    unittest.main()

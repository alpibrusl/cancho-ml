"""S1 (docs/serving-http.md sections 4 and 4.1): the answer over HTTP is the answer of `ml predict`, value by value, as text.

For each of the 19 models of the serving corpus (kinds 1 to 7, k-means included, categorical, hash and missing-cell cases) and each input CSV of its family, the rows are sent to
`ml-serve` as records, one request each (`POST /predict`) and all at once (`POST /predict/batch`), and `ml predict` runs on the same rows as a CSV. The comparison is exact: the
CLI's cells are strings, the server's values are the number tokens of the response body (read with `parse_float=str, parse_int=str`), and `str == str` is the test. A row the server
refuses is checked against the command line too: a refusal of the predictor is the command line's rule on that row, and a string outside a listed vocabulary (`enum`, 422 `choice`) is a row the
command line accepts and counts as an unseen category.
"""

import csv
import io
import json
import re
import subprocess
import unittest

from harness import BIN, ROOT, run
from serve_lib import Server, corpus

NUMBER = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")

SB = None
MODELS = {}
INPUTS = {}


def setUpModule():
    global SB, MODELS, INPUTS
    SB, MODELS, INPUTS = corpus()


def tearDownModule():
    SB.cleanup()


def ml(*args):
    return SB.run(*args)


def describe(model_file):
    r = ml("describe", "--model-file", model_file, "--limit", "1000000")
    assert r.status == 0, r
    return r.data()


def read_rows(name):
    text = SB.read(name).decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    rows = [r for r in rows if r]
    return rows[0], rows[1:]


def record_text(desc, header, row):
    """The JSON text of the record for a CSV row, or None when the row cannot be sent (a cell that is empty for a model that takes no null, a missing column, a bad number)."""
    nullable = desc["request"]["missing_cells"] == "handled"
    cells = dict(zip(header, row))
    parts = []
    for c in desc["columns"]:
        if c["role"] == "target":
            continue
        if c["name"] not in cells:
            return None
        cell = cells[c["name"]]
        if cell == "":
            if not nullable:
                return None
            tok = "null"
        elif c["type"] == "categorical":
            tok = json.dumps(cell)
        elif NUMBER.fullmatch(cell):
            tok = cell
        else:
            return None
        parts.append(json.dumps(c["name"]) + ":" + tok)
    return "{" + ",".join(parts) + "}"


def cli_flags(desc):
    out = desc["output"]
    if out["with_proba"] is not None:
        return ["--proba"]
    if out["with_distance"] is not None:
        return ["--distance"]
    return []


def write_csv(name, desc, header, rows):
    names = desc["request"]["header"]
    names = [n if isinstance(n, str) else None for n in names]
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(names)
    idx = [header.index(n) for n in names]
    for r in rows:
        w.writerow([r[i] for i in idx])
    SB.write(name, out.getvalue().encode("utf-8"))


def tokens(resp):
    d = resp.json()
    return d["columns"], d["rows"], d["unknown_categories"], d["model_sha256"]


class Bytes(unittest.TestCase):
    def test_every_model_every_input(self):
        total = {"rows": 0, "values": 0, "enum_refused": 0, "predictor_refused": 0, "skipped_rows": 0, "pairs": 0}
        for mid, (mfile, kind, family) in MODELS.items():
            desc = describe(mfile)
            sha = desc["file"]["sha256"]
            with Server(SB.dir, mfile, "--max-batch-rows", "1000") as srv:
                keep = srv.conn()
                for iid, (iname, ifamily) in INPUTS.items():
                    if ifamily != family:
                        continue
                    with self.subTest(model=mid, input=iid):
                        header, rows = read_rows(iname)
                        kept, texts = [], []
                        for r in rows:
                            if len(r) != len(header):
                                continue
                            t = record_text(desc, header, r)
                            if t is None:
                                total["skipped_rows"] += 1
                                continue
                            kept.append(r)
                            texts.append(t)
                        accepted, one_by_one = [], []
                        for r, t in zip(kept, texts):
                            resp = keep.request("POST", "/predict", body=t.encode(), headers={"Content-Type": "application/json"})
                            self.assertEqual(resp.headers.get("x-model-sha256"), sha)
                            if resp.status == 200:
                                cols, vals, unk, got_sha = tokens(resp)
                                self.assertEqual(got_sha, sha)
                                self.assertEqual(len(vals), 1)
                                accepted.append((r, t))
                                one_by_one.append((vals[0], unk))
                                continue
                            # a refusal: it must be one the command line agrees with
                            problem = resp.json()
                            self.assertEqual(resp.ctype().split(";")[0], "application/problem+json", resp)
                            write_csv("one.csv", desc, header, [r])
                            cli = ml("predict", "--model-file", mfile, "one.csv", *cli_flags(desc))
                            if problem["rule"] == "ml.schema.mismatch" and any(e["code"] == "choice" for e in problem["errors"]):
                                # the command line accepts an unseen category, unless the plan says unknown=refuse: then it refuses with that rule too
                                self.assertTrue(cli.status == 0 or cli.first_rule() == "ml.input.unknown-category", ("the server refused a category the command line accepts", cli))
                                total["enum_refused"] += 1
                            else:
                                self.assertNotEqual(cli.status, 0, ("the server refused a row the command line predicts", problem, cli))
                                self.assertEqual(cli.first_rule(), problem["rule"], (problem, cli))
                                total["predictor_refused"] += 1
                        if not accepted:
                            continue
                        rs = [r for r, t in accepted]
                        write_csv("sub.csv", desc, header, rs)
                        cli = ml("predict", "--model-file", mfile, "sub.csv", "--limit", "1000", *cli_flags(desc))
                        self.assertEqual(cli.status, 0, cli)
                        want = cli.data()
                        self.assertEqual(want["row_count"], len(rs))
                        # one request per row
                        for i, (vals, unk) in enumerate(one_by_one):
                            self.assertEqual(vals, want["rows"][i], ("single", mid, iid, i))
                        # all at once
                        body = ('{"rows":[' + ",".join(t for r, t in accepted) + "]}").encode()
                        resp = keep.request("POST", "/predict/batch", body=body, headers={"Content-Type": "application/json"})
                        self.assertEqual(resp.status, 200, resp)
                        cols, vals, unk, got_sha = tokens(resp)
                        self.assertEqual(cols, want["columns"])
                        self.assertEqual(vals, want["rows"], ("batch", mid, iid))
                        self.assertEqual(int(unk), want.get("unknown_categories", 0))
                        self.assertEqual(want["task"], resp.json()["task"])
                        self.assertEqual(want["model"], resp.json()["model"])
                        total["rows"] += len(rs)
                        total["values"] += sum(len(v) for v in vals)
                        total["pairs"] += 1
        print("serve bytes: %(pairs)d model/input pairs, %(rows)d rows, %(values)d values compared as text; %(enum_refused)d rows refused for their category, "
              "%(predictor_refused)d rows refused by the predictor (rule equal to the command line's), %(skipped_rows)d rows not sendable" % total)
        self.assertGreater(total["values"], 1000)


if __name__ == "__main__":
    unittest.main()

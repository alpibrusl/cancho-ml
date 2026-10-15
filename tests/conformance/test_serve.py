"""S2 (docs/serving-http.md section 9): `ml-serve` end to end over real sockets.

A Python client starts the binary on a free port and checks readiness, every endpoint, every refusal with its status and body, keep-alive, pipelining, a request split across
many one-byte writes, a body larger than the limit, the wrong content type, invalid JSON (duplicate keys, NaN, Infinity, huge numbers, deep nesting, a byte order mark, invalid
UTF-8), unknown, missing and extra columns, unseen categories, null, one record and a batch at the limit and over it, the token, the refusals at start, and the log. Every
documented response is validated against the document the server serves.
"""

import json
import os
import socket
import subprocess
import time
import unittest

import jsonschema

from harness import BIN, ROOT, run
from serve_lib import SERVE_BIN, Conn, Server, corpus, free_port, loads_tokens

SB = None
MODELS = {}
JSON = {"Content-Type": "application/json"}


def setUpModule():
    global SB, MODELS
    SB, MODELS, _ = corpus()


def tearDownModule():
    SB.cleanup()


def describe(mfile, limit="1000"):
    r = SB.run("describe", "--model-file", mfile, "--limit", limit)
    assert r.status == 0, r
    return r.data()


class Doc:
    """The document a server serves, and the check of a response against it."""

    def __init__(self, doc):
        self.doc = doc

    def check(self, path, method, resp):
        ops = self.doc["paths"][path][method.lower()]
        r = ops["responses"].get(str(resp.status))
        if r is not None and "$ref" in r:
            r = self.doc["components"]["responses"][r["$ref"].rsplit("/", 1)[1]]
        if r is None:
            return "status %d is not documented for %s %s" % (resp.status, method, path)
        ctype = resp.ctype().split(";")[0]
        content = r.get("content", {})
        if ctype not in content:
            return "content type %r is not documented for %d %s %s" % (ctype, resp.status, method, path)
        schema = dict(content[ctype].get("schema", {}))
        if not schema:
            return None
        schema["components"] = self.doc["components"]
        try:
            jsonschema.validate(json.loads(resp.text()), schema, cls=jsonschema.Draft202012Validator)
        except jsonschema.ValidationError as e:
            return "the body breaks the document: %s" % e.message[:200]
        return None


class Common(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "linear.cml"
        cls.srv = Server(SB.dir, cls.model, "--max-batch-rows", "5", "--max-body-bytes", "2000").start()
        cls.doc = Doc(json.loads(cls.srv.request("GET", "/openapi.json").text()))
        cls.desc = describe(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def ok(self, resp, path, method="POST"):
        why = self.doc.check(path, method, resp)
        self.assertIsNone(why, (why, resp))

    def post(self, path, body, headers=JSON, conn=None):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        c = conn or self.srv
        return c.request("POST", path, body=b, headers=headers)

    def refused(self, resp, status, rule=None, path="/predict"):
        self.assertEqual(resp.status, status, resp)
        self.assertEqual(resp.ctype().split(";")[0], "application/problem+json", resp)
        d = loads_tokens(resp.text())
        self.assertEqual(int(d["status"]), status)
        if rule is not None:
            self.assertEqual(d["rule"], rule, d)
        self.assertEqual(resp.headers["x-model-sha256"], self.desc["file"]["sha256"])
        if path in self.doc.doc["paths"]:
            self.ok(resp, path)
        return d


RECORD = {"a": 3, "b": 2.50, "c": 0.5}


class Endpoints(Common):
    def test_health_and_the_digest(self):
        r = self.srv.request("GET", "/health")
        self.assertEqual(r.status, 200)
        self.assertEqual(r.json(), {"ok": "true"} if False else json.loads(r.text()))
        self.assertEqual(json.loads(r.text()), {"ok": True, "model_sha256": self.desc["file"]["sha256"]})
        self.assertEqual(r.headers["x-model-sha256"], self.desc["file"]["sha256"])
        self.ok(r, "/health", "GET")
        # the digest is the SHA-256 of the file's bytes
        import hashlib
        self.assertEqual(hashlib.sha256(SB.read(self.model)).hexdigest(), self.desc["file"]["sha256"])

    def test_model_is_describe(self):
        r = self.srv.request("GET", "/model")
        self.assertEqual(r.status, 200)
        self.assertEqual(json.loads(r.text()), self.desc)
        self.ok(r, "/model", "GET")

    def test_openapi_is_generated_from_the_model(self):
        r = self.srv.request("GET", "/openapi.json")
        self.assertEqual(r.status, 200)
        doc = json.loads(r.text())
        self.assertEqual(doc["openapi"], "3.1.0")
        self.assertEqual(sorted(doc["paths"]), ["/health", "/model", "/openapi.json", "/predict", "/predict/batch"])
        rec = doc["components"]["schemas"]["Record"]
        self.assertEqual(rec["additionalProperties"], False)
        feats = [c for c in self.desc["columns"] if c["role"] == "feature"]
        self.assertEqual(list(rec["properties"])[: len(feats)], [c["name"] for c in feats])
        for c in feats:
            p = rec["properties"][c["name"]]
            self.assertEqual(p["type"], {"int": "integer", "dec": "number", "float": "number"}[c["type"]])
            self.assertIn(c["name"], rec["required"])
        self.assertNotIn("y", rec["required"])
        batch = doc["components"]["schemas"]["Batch"]
        self.assertEqual(batch["properties"]["rows"]["maxItems"], 5)
        self.assertEqual(batch["properties"]["rows"]["minItems"], 1)
        self.assertIn(self.desc["file"]["sha256"], doc["info"]["description"])

    def test_predict_one(self):
        r = self.post("/predict", RECORD)
        self.assertEqual(r.status, 200, r)
        d = loads_tokens(r.text())
        self.assertEqual(d["columns"], ["prediction"])
        self.assertEqual(len(d["rows"]), 1)
        self.assertEqual(d["model_sha256"], self.desc["file"]["sha256"])
        self.assertEqual(r.headers["x-model-sha256"], d["model_sha256"])
        self.assertEqual(r.ctype(), "application/json")
        self.ok(r, "/predict")
        # the command line says the same
        SB.write("rec.csv", "a,b,c\n3,2.50,0.5\n")
        cli = SB.run("predict", "--model-file", self.model, "rec.csv")
        self.assertEqual(cli.data()["rows"], d["rows"])

    def test_predict_batch_at_the_limit_and_over(self):
        rows = [dict(RECORD, a=i) for i in range(5)]
        r = self.post("/predict/batch", {"rows": rows})
        self.assertEqual(r.status, 200, r)
        self.assertEqual(len(loads_tokens(r.text())["rows"]), 5)
        self.ok(r, "/predict/batch")
        r = self.post("/predict/batch", {"rows": rows + [RECORD]})
        d = self.refused(r, 413, "serve.batch-too-large", "/predict/batch")
        r = self.post("/predict/batch", {"rows": []})
        d = self.refused(r, 422, "ml.schema.mismatch", "/predict/batch")
        self.assertEqual(d["errors"][0]["code"], "min_items")
        r = self.post("/predict/batch", {"rows": [RECORD, dict(RECORD, a="x")]})
        d = self.refused(r, 422, "ml.schema.mismatch", "/predict/batch")
        self.assertEqual(d["errors"][0]["pointer"], "/rows/1/a")

    def test_every_error_at_once_with_its_pointer(self):
        r = self.post("/predict", {"a": "x", "zz": 1, "d": 2})
        d = self.refused(r, 422, "ml.schema.mismatch")
        pairs = {(e["pointer"], e["code"]) for e in d["errors"]}
        self.assertEqual(pairs, {("/a", "type"), ("/b", "required"), ("/c", "required"), ("/zz", "unknown"), ("/d", "unknown")})
        self.assertEqual(int(d["count"]), 5)

    def test_null_where_the_model_does_not_take_it(self):
        r = self.post("/predict", dict(RECORD, c=None))
        d = self.refused(r, 422, "ml.schema.mismatch")
        self.assertEqual(d["errors"][0]["pointer"], "/c")

    def test_the_target_may_be_sent_and_is_ignored(self):
        a = self.post("/predict", dict(RECORD, y=123.5))
        b = self.post("/predict", RECORD)
        self.assertEqual(a.status, 200)
        self.assertEqual(loads_tokens(a.text())["rows"], loads_tokens(b.text())["rows"])

    def test_numbers_that_cannot_be_rendered_exactly(self):
        for body, ptr in [('{"a":3.0000000000000000001,"b":2.5,"c":0.5}', "/a"), ('{"a":3,"b":2.505,"c":0.5}', "/b"), ('{"a":3,"b":2.5,"c":1e999}', "/c"),
                          ('{"a":3,"b":2.5,"c":-1e999}', "/c"), ('{"a":3,"b":1234567890123.5,"c":0.5}', "/b"), ('{"a":3,"b":2.5,"c":1e-999}', "/c")]:
            with self.subTest(body=body):
                r = self.post("/predict", body.encode())
                d = self.refused(r, 422, "serve.number-not-exact")
                self.assertEqual(d["errors"][0]["pointer"], ptr)
        # an int column takes integers: 1.5 and 1e13 are the schema's (type and maximum), before any rendering
        for body in ('{"a":1.5,"b":2.5,"c":0.5}', '{"a":1e13,"b":2.5,"c":0.5}', '{"a":1000000000000,"b":2.5,"c":0.5}'):
            d = self.refused(self.post("/predict", body.encode()), 422, "ml.schema.mismatch")
            self.assertEqual(d["errors"][0]["pointer"], "/a")
        # whole numbers written as floats are integers, as the document says
        for a in ("3.0", "3e0", "0.3e1", "30e-1"):
            r = self.post("/predict", ('{"a":%s,"b":2.5,"c":0.5}' % a).encode())
            self.assertEqual(r.status, 200, (a, r))
        # a dec column takes the digits of its scale and any spelling of the same value
        want = loads_tokens(self.post("/predict", RECORD).text())["rows"]
        for b in ("2.5", "2.50", "250e-2", "0.25e1", "2.500"):
            r = self.post("/predict", ('{"a":3,"b":%s,"c":0.5}' % b).encode())
            self.assertEqual(loads_tokens(r.text())["rows"], want, b)
        # -0 is 0
        r1 = self.post("/predict", b'{"a":-0,"b":-0.00,"c":-0.0}')
        r2 = self.post("/predict", b'{"a":0,"b":0,"c":0}')
        self.assertEqual(loads_tokens(r1.text())["rows"], loads_tokens(r2.text())["rows"])


class Refusals(Common):
    def test_not_found_and_method(self):
        r = self.srv.request("GET", "/nope")
        d = self.refused(r, 404, None, "/nope")
        r = self.srv.request("DELETE", "/predict")
        self.assertEqual(r.status, 405)
        self.assertEqual(r.headers["allow"], "POST")
        r = self.srv.request("GET", "/predict")
        self.assertEqual(r.status, 405)
        r = self.srv.request("POST", "/health", body=b"{}", headers=JSON)
        self.assertEqual(r.status, 405)
        r = self.srv.request("POST", "/admin/reload", body=b"{}", headers=JSON)
        self.assertEqual(r.status, 404)

    def test_content_type(self):
        for h in ({"Content-Type": "text/plain"}, {"Content-Type": "application/jsonx"}, {"Content-Type": "application/x-www-form-urlencoded"}, {}):
            r = self.post("/predict", RECORD, headers=h)
            self.refused(r, 415, "serve.media-type")
        for h in ({"Content-Type": "application/json; charset=utf-8"}, {"Content-Type": "Application/JSON"}):
            self.assertEqual(self.post("/predict", RECORD, headers=h).status, 200)

    def test_invalid_json(self):
        bad = [b"", b"{", b"nope", b'{"a":1,}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":-Infinity}', b'[1,2', b"\xef\xbb\xbf" + json.dumps(RECORD).encode(), b'{"a":"\xff"}', b'{"a":"\x01"}',
               json.dumps(RECORD).encode() + b" x", b"{'a':1}", b'{"a":01,"b":2,"c":3}', b'{"a":+1}', b'{"a":.5}', b'{"a":1.}', b"[" * 1500, b'{"a":' * 300, b'"\\ud800"', b"\x00"]
        for body in bad:
            with self.subTest(body=body[:40]):
                r = self.post("/predict", body)
                d = self.refused(r, 400, "serve.body-not-json")
                self.assertIn("byte", d["detail"])

    def test_json_that_is_not_an_object(self):
        for body in (b"[]", b"3", b'"x"', b"null", b"true"):
            r = self.post("/predict", body)
            d = self.refused(r, 422, "ml.schema.mismatch")
        r = self.post("/predict/batch", b'{"rows":{"a":1}}')
        self.refused(r, 422, "ml.schema.mismatch", "/predict/batch")
        r = self.post("/predict/batch", json.dumps(RECORD).encode())
        d = self.refused(r, 422, "ml.schema.mismatch", "/predict/batch")
        self.assertIn("/rows", [e["pointer"] for e in d["errors"]])

    def test_duplicate_keys(self):
        r = self.post("/predict", b'{"a":1,"a":2,"b":2.5,"c":0.5}')
        self.refused(r, 400, "serve.duplicate-key")
        r = self.post("/predict", b'{"a":1,"b":2.5,"c":0.5,"y":1,"y":2}')
        self.refused(r, 400, "serve.duplicate-key")
        r = self.post("/predict/batch", b'{"rows":[{"a":1,"b":2.5,"c":0.5},{"a":1,"b":2.5,"c":0.5,"c":0.5}]}')
        d = self.refused(r, 400, "serve.duplicate-key", "/predict/batch")
        self.assertEqual(d["errors"][0]["pointer"], "/rows/1")
        r = self.post("/predict/batch", b'{"rows":[{"a":1,"b":2.5,"c":0.5}],"rows":[]}')
        self.refused(r, 400, "serve.duplicate-key", "/predict/batch")

    def test_deep_nesting_does_not_trap(self):
        r = self.post("/predict", b'{"a":' + b"[" * 70 + b"]" * 70 + b',"b":1,"c":1}')
        self.assertIn(r.status, (400, 422))
        self.assertTrue(self.srv.alive())

    def test_a_body_over_the_limit(self):
        # --max-body-bytes is 2000: a body past it that the loop's buffer still holds is the handler's refusal, problem+json
        body = json.dumps(dict(RECORD, y="x" * 3000)).encode()
        r = self.srv.conn().request("POST", "/predict", body=body, headers=JSON)
        self.refused(r, 413, "serve.body-too-large")
        # a body past what the buffer holds (the limit and a head) is the loop's: 413 with its own body, and the connection is closed
        huge = json.dumps(dict(RECORD, y="x" * 30000)).encode()
        c = self.srv.conn()
        r = c.request("POST", "/predict", body=huge, headers=JSON)
        self.assertEqual(r.status, 413)
        self.assertEqual(r.ctype().split(";")[0], "application/json")
        self.assertIn("error", json.loads(r.text()))
        # the server serves the next connection, and a body at the limit
        self.assertEqual(self.srv.request("GET", "/health").status, 200)
        ok = json.dumps(dict(RECORD, y="x" * 1900)).encode()
        self.assertLessEqual(len(ok), 2000)
        self.assertEqual(self.post("/predict", ok).status, 200)
        # the limit is exact: 2000 bytes are served, 2001 are not
        for n in range(1900, 2100):
            b = json.dumps(dict(RECORD, y="x" * n)).encode()
            if len(b) == 2000:
                self.assertEqual(self.post("/predict", b).status, 200)
            if len(b) == 2001:
                self.refused(self.post("/predict", b), 413, "serve.body-too-large")

    def test_a_head_over_the_limit(self):
        c = self.srv.conn()
        c.send(b"GET /health HTTP/1.1\r\nHost: x\r\nX-Pad: " + b"a" * 20000 + b"\r\n\r\n")
        r = c.read_response()
        self.assertIn(r.status, (400, 413, 431))
        self.assertEqual(self.srv.request("GET", "/health").status, 200)

    def test_malformed_http(self):
        for raw in (b"NOT HTTP\r\n\r\n", b"GET\r\n\r\n", b"GET /health HTTP/9.9\r\n\r\n", b"POST /predict HTTP/1.1\r\nContent-Length: abc\r\n\r\n", b"GET /health HTTP/1.1\r\nBad Header\r\n\r\n",
                    b"POST /predict HTTP/1.1\r\nContent-Length: -5\r\n\r\n", b"\x00\x01\x02\r\n\r\n"):
            with self.subTest(raw=raw[:30]):
                c = self.srv.conn()
                c.send(raw)
                try:
                    r = c.read_response()
                    self.assertIn(r.status, (400, 404, 405, 413, 431, 505, 411))
                except (EOFError, ConnectionError, socket.timeout):
                    pass
        self.assertEqual(self.srv.request("GET", "/health").status, 200)


class Transport(Common):
    def test_keep_alive(self):
        c = self.srv.conn()
        for i in range(5):
            r = c.request("POST", "/predict", body=json.dumps(dict(RECORD, a=i)).encode(), headers=JSON)
            self.assertEqual(r.status, 200)
            self.assertEqual(r.headers.get("connection", "").lower(), "keep-alive")
        r = c.request("GET", "/health", close=True)
        self.assertEqual(r.status, 200)
        self.assertEqual(c.sock.recv(10), b"")  # closed by the server

    def test_pipelined_requests(self):
        c = self.srv.conn()
        reqs = [Conn.build("POST", "/predict", json_body=dict(RECORD, a=i)) for i in range(5)] + [Conn.build("GET", "/health"), Conn.build("GET", "/nope")]
        c.send(b"".join(reqs))
        got = [c.read_response() for _ in reqs]
        self.assertEqual([r.status for r in got], [200] * 6 + [404])
        alone = [loads_tokens(self.post("/predict", dict(RECORD, a=i)).text())["rows"] for i in range(5)]
        self.assertEqual([loads_tokens(r.text())["rows"] for r in got[:5]], alone)
        self.assertEqual(len({str(a) for a in alone}), 5)

    def test_a_request_split_across_many_writes(self):
        req = Conn.build("POST", "/predict", json_body=RECORD)
        c = self.srv.conn()
        c.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for i in range(len(req)):
            c.send(req[i:i + 1])
            if i % 16 == 0:
                time.sleep(0.001)
        r = c.read_response()
        self.assertEqual(r.status, 200)
        # and a request whose body arrives long after its head
        c2 = self.srv.conn()
        body = json.dumps(RECORD).encode()
        c2.send(Conn.build("POST", "/predict", body=body, headers=JSON)[: -len(body)])
        time.sleep(0.2)
        c2.send(body[:5])
        time.sleep(0.2)
        c2.send(body[5:])
        self.assertEqual(c2.read_response().status, 200)

    def test_two_connections_interleave(self):
        a, b = self.srv.conn(), self.srv.conn()
        req = Conn.build("POST", "/predict", json_body=RECORD)
        a.send(req[:20])
        r = b.request("POST", "/predict", json_body=RECORD)
        self.assertEqual(r.status, 200)
        a.send(req[20:])
        self.assertEqual(a.read_response().status, 200)

    def test_chunked_request_body(self):
        body = json.dumps(RECORD).encode()
        raw = b"POST /predict HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n"
        raw += b"%x\r\n" % 7 + body[:7] + b"\r\n" + b"%x\r\n" % (len(body) - 7) + body[7:] + b"\r\n0\r\n\r\n"
        c = self.srv.conn()
        c.send(raw)
        self.assertEqual(c.read_response().status, 200)

    def test_the_log_has_no_data(self):
        marker = "zq7marker"
        self.post("/predict", dict(RECORD, **{marker: 1}))
        self.srv.request("GET", "/health?token=" + marker)
        log = self.srv.stderr()
        self.assertIn("listening on %d" % self.srv.port, log)
        self.assertIn("POST /predict 422", log)
        self.assertNotIn(marker, log)
        self.assertNotIn("0.5", log)
        self.assertNotIn(self.desc["file"]["sha256"], log)
        for line in log.splitlines():
            self.assertRegex(line, r"^(listening on \d+|[A-Z]+ /\S* \d{3} \d+ms)$")


class Kinds(unittest.TestCase):
    """The other kinds of answer: a classification with its probability, a clustering with its distance, categories with a listed vocabulary, null where the model takes it."""

    def serve(self, mfile, *flags, **kw):
        s = Server(SB.dir, mfile, *flags, **kw).start()
        self.addCleanup(s.stop)
        return s

    def test_classification_has_label_and_probability(self):
        for mid in ("logistic", "tree_cls", "forest_cls", "const_cls"):
            s = self.serve(MODELS[mid][0])
            d = loads_tokens(s.request("POST", "/predict", body=b'{"a":3,"b":2.5,"c":0.5}', headers=JSON).text())
            self.assertEqual(d["columns"], ["label", "probability"], mid)
            self.assertEqual(d["task"], "classification")
            self.assertEqual(len(d["rows"][0]), 2)

    def test_clustering_has_cluster_and_distance(self):
        s = self.serve(MODELS["kmeans"][0])
        d = loads_tokens(s.request("POST", "/predict", body=b'{"a":3,"b":2.5,"c":0.5}', headers=JSON).text())
        self.assertEqual(d["columns"], ["cluster", "distance"])
        self.assertEqual(d["task"], "clustering")

    def test_regression_has_prediction(self):
        for mid in ("const_reg_dec", "const_reg_float", "tree_reg", "forest_reg"):
            s = self.serve(MODELS[mid][0])
            d = loads_tokens(s.request("POST", "/predict", body=b'{"a":3,"b":2.5,"c":0.5}', headers=JSON).text())
            self.assertEqual(d["columns"], ["prediction"], mid)

    def test_categories_are_a_listed_vocabulary(self):
        mfile = MODELS["cat_linear_onehot"][0]
        desc = describe(mfile)
        s = self.serve(mfile)
        doc = json.loads(s.request("GET", "/openapi.json").text())
        rec = doc["components"]["schemas"]["Record"]["properties"]
        values = sorted(v["value"] for v in desc["categorical"][0]["values"])
        self.assertEqual(sorted(rec["c0"]["enum"]), values)
        ok = {"c0": values[0], "c1": "x", "n": 10}
        self.assertEqual(s.request("POST", "/predict", body=json.dumps(ok).encode(), headers=JSON).status, 200)
        r = s.request("POST", "/predict", body=json.dumps(dict(ok, c0="never-seen")).encode(), headers=JSON)
        self.assertEqual(r.status, 422)
        d = r.json()
        self.assertEqual(d["rule"], "ml.schema.mismatch")
        self.assertEqual((d["errors"][0]["pointer"], d["errors"][0]["code"]), ("/c0", "choice"))
        # a string with a comma, a quote, a newline is a category like another
        ok2 = dict(ok, c0="a,b")
        self.assertEqual(s.request("POST", "/predict", body=json.dumps(ok2).encode(), headers=JSON).status, 422 if "a,b" not in values else 200)

    def test_categories_that_need_the_csv_quoting(self):
        mfile = MODELS["cat_tree_freq"][0]
        s = self.serve(mfile)
        desc = describe(mfile)
        vals = [v["value"] for v in desc["categorical"][0]["values"]]
        for v in vals:
            r = s.request("POST", "/predict", body=json.dumps({"c0": v, "c1": "x", "n": 3}).encode(), headers=JSON)
            self.assertIn(r.status, (200, 422), (v, r))
        for v in ("a,b", 'q"uote', "multi\nline", "café"):
            if v in vals:
                r = s.request("POST", "/predict", body=json.dumps({"c0": v, "c1": "x", "n": 3}).encode(), headers=JSON)
                self.assertEqual(r.status, 200, (v, r))
        # a CR LF inside a category would be read as an LF: refused with its pointer, never changed
        r = s.request("POST", "/predict", body=json.dumps({"c0": "a\r\nb", "c1": "x", "n": 3}).encode(), headers=JSON)
        self.assertEqual(r.status, 422)
        self.assertIn(r.json()["rule"], ("serve.string-not-renderable", "ml.schema.mismatch"))

    def test_strings_the_csv_carries_as_sent(self):
        # the hash step has no vocabulary to list, so any string is a category: a lone CR, U+0000, a comma, quotes and a line feed are written as sent and read back as sent
        mfile = MODELS["cat_forest_hash"][0]
        s = self.serve(mfile)
        base = None
        seen = set()
        for v in ("ab", "a\rb", "a\u0000b", "a,b", 'q"uote', "multi\nline", "caf\u00e9", "a\r", "\r", "\u0000"):
            r = s.request("POST", "/predict", body=json.dumps({"c0": v, "c1": "x", "n": 3}).encode(), headers=JSON)
            self.assertEqual(r.status, 200, (v, r))
            # the command line reads the same text from a CSV and predicts the same
            import csv as _csv, io as _io
            out = _io.StringIO()
            _csv.writer(out, lineterminator="\n").writerow(["c0", "c1", "n"])
            _csv.writer(out, lineterminator="\n").writerow([v, "x", "3"])
            SB.write("str.csv", out.getvalue().encode())
            cli = SB.run("predict", "--model-file", mfile, "str.csv")
            self.assertEqual(cli.status, 0, (v, cli))
            self.assertEqual(r.json()["rows"], cli.data()["rows"], v)
            seen.add(tuple(r.json()["rows"][0]))
        self.assertGreater(len(seen), 2)

    def test_null_where_the_model_takes_it(self):
        mfile = MODELS["cat_forest_hash"][0]
        desc = describe(mfile)
        self.assertEqual(desc["request"]["missing_cells"], "handled")
        s = self.serve(mfile)
        doc = json.loads(s.request("GET", "/openapi.json").text())
        rec = doc["components"]["schemas"]["Record"]["properties"]
        self.assertIn("null", rec["n"]["type"])
        r = s.request("POST", "/predict", body=b'{"c0":"red","c1":"x","n":null}', headers=JSON)
        self.assertEqual(r.status, 200, r)
        SB.write("nul.csv", "c0,c1,n\nred,x,\n")
        cli = SB.run("predict", "--model-file", mfile, "nul.csv")
        self.assertEqual(r.json()["rows"], cli.data()["rows"])

    def test_a_model_with_a_column_named_rows(self):
        SB.write("rows.csv", "rows,x,y\n1,2,3\n2,3,5\n3,5,8\n4,7,11\n5,11,16\n")
        fit = SB.run("fit", "--target", "y", "--output", "rowsmodel.cml", "--model", "linear", "--step", "impute column=* strategy=mean", "rows.csv")
        self.assertEqual(fit.status, 0, fit)
        s = self.serve("rowsmodel.cml")
        r = s.request("POST", "/predict", body=b'{"rows":2,"x":3}', headers=JSON)
        self.assertEqual(r.status, 200, r)
        r = s.request("POST", "/predict/batch", body=b'{"rows":[{"rows":2,"x":3}]}', headers=JSON)
        self.assertEqual(r.status, 200, r)


class Token(unittest.TestCase):
    TOKEN = "s3cret-Tok_en.value~1"

    @classmethod
    def setUpClass(cls):
        cls.srv = Server(SB.dir, "linear.cml", token=cls.TOKEN).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def test_every_operation_but_health_needs_the_token(self):
        auth = {"Authorization": "Bearer " + self.TOKEN}
        self.assertEqual(self.srv.request("GET", "/health").status, 200)
        for method, path in (("GET", "/model"), ("GET", "/openapi.json"), ("POST", "/predict"), ("POST", "/predict/batch")):
            body = json.dumps(RECORD).encode() if method == "POST" else None
            if path.endswith("batch"):
                body = json.dumps({"rows": [RECORD]}).encode()
            r = self.srv.request(method, path, body=body, headers=JSON)
            self.assertEqual(r.status, 401, (path, r))
            self.assertEqual(r.headers["x-model-sha256"], describe("linear.cml")["file"]["sha256"])
            r = self.srv.request(method, path, body=body, headers=dict(JSON, Authorization="Bearer wrong"))
            self.assertEqual(r.status, 401, (path, r))
            r = self.srv.request(method, path, body=body, headers=dict(JSON, Authorization=self.TOKEN))
            self.assertEqual(r.status, 401, (path, r))
            r = self.srv.request(method, path, body=body, headers=dict(JSON, **auth))
            self.assertEqual(r.status, 200, (path, r))

    def test_the_token_is_nowhere_in_the_server(self):
        auth = {"Authorization": "Bearer " + self.TOKEN}
        a = self.srv.request("GET", "/openapi.json", headers=auth).text()
        b = self.srv.request("GET", "/model", headers=auth).text()
        self.assertNotIn(self.TOKEN, a + b + self.srv.stderr())
        self.srv.request("GET", "/model", headers={"Authorization": "Bearer " + self.TOKEN + "x"})
        self.assertNotIn(self.TOKEN, self.srv.stderr())
        doc = json.loads(a)
        self.assertIn("securitySchemes", doc["components"])

    def test_the_document_says_who_may_call(self):
        doc = json.loads(self.srv.request("GET", "/openapi.json", headers={"Authorization": "Bearer " + self.TOKEN}).text())
        self.assertEqual(doc["paths"]["/health"]["get"].get("security"), [])
        self.assertTrue(doc["paths"]["/predict"]["post"]["security"])
        self.assertIn("401", doc["paths"]["/predict"]["post"]["responses"])


class Start(unittest.TestCase):
    """The refusals at start: the rule, the exit code, no listening."""

    def start(self, *args, root=None):
        p = subprocess.run([SERVE_BIN, *args], capture_output=True, timeout=20, stdin=subprocess.DEVNULL)
        out = p.stdout.decode()
        return p.returncode, (json.loads(out.splitlines()[0]) if out.strip() else None), p.stderr.decode()

    def test_no_token_flag_is_refused(self):
        code, d, _ = self.start("--root", str(SB.dir), "--model-file", "linear.cml")
        self.assertEqual((code, d["error"]["rule"]), (2, "serve.no-token"))
        code, d, _ = self.start("--root", str(SB.dir), "--model-file", "linear.cml", "--no-token", "--token-file", "linear.cml")
        self.assertEqual((code, d["error"]["rule"]), (2, "args.conflict"))

    def test_the_token_file(self):
        for text, rule in ((b"", "serve.token"), (b"\n", "serve.token"), (b"has space\n", "serve.token"), (b"a" * 5000, "serve.token"), (b"caf\xc3\xa9\n", "serve.token"), (b"tab\there\n", "serve.token")):
            SB.write("tok.txt", text)
            code, d, _ = self.start("--root", str(SB.dir), "--model-file", "linear.cml", "--token-file", "tok.txt")
            self.assertEqual(d["error"]["rule"], rule, text[:20])
            self.assertNotEqual(code, 0)
        code, d, _ = self.start("--root", str(SB.dir), "--model-file", "linear.cml", "--token-file", "missing.txt")
        self.assertEqual((code, d["error"]["rule"]), (3, "io.not-found"))

    def test_the_model_file_is_refused_with_the_describe_refusal(self):
        data = bytearray(SB.read("linear.cml"))
        data[len(data) // 2] ^= 0xFF
        SB.write("damaged.cml", bytes(data))
        want = SB.run("describe", "--model-file", "damaged.cml")
        code, d, _ = self.start("--root", str(SB.dir), "--model-file", "damaged.cml", "--no-token")
        self.assertNotEqual(want.status, 0)
        self.assertEqual(code, want.status)
        self.assertEqual(d["error"]["rule"], want.first_rule())
        SB.write("empty.cml", b"")
        code, d, _ = self.start("--root", str(SB.dir), "--model-file", "empty.cml", "--no-token")
        self.assertNotEqual(code, 0)
        code, d, _ = self.start("--root", str(SB.dir), "--model-file", "nothing.cml", "--no-token")
        self.assertEqual((code, d["error"]["rule"]), (3, "io.not-found"))
        code, d, _ = self.start("--root", str(SB.dir), "--model-file", "../x", "--no-token")
        self.assertNotEqual(code, 0)
        code, d, _ = self.start("--root", str(SB.dir), "--model-file", "linear.cml", "--no-token", "--max-bytes", "100")
        self.assertEqual((code, d["error"]["rule"]), (8, "limit.input-too-large"))

    def test_flags(self):
        for flags, rule in ((["--port", "0"], "ml.param.out-of-range"), (["--port", "70000"], "ml.param.out-of-range"), (["--max-batch-rows", "1001"], "ml.param.out-of-range"),
                            (["--max-body-bytes", "99999999"], "ml.param.out-of-range"), (["--idle-seconds", "0"], "ml.param.out-of-range"), (["--bogus"], "args.unknown-flag"),
                            (["extra"], "args.too-many-operands")):
            code, d, _ = self.start("--root", str(SB.dir), "--model-file", "linear.cml", "--no-token", *flags)
            self.assertEqual(code, 2, flags)
            self.assertEqual(d["error"]["rule"], rule, flags)
        code, d, _ = self.start("--no-token")
        self.assertEqual((code, d["error"]["rule"]), (2, "args.required-flag"))

    def test_a_port_in_use(self):
        with Server(SB.dir, "linear.cml") as first:
            code, d, _ = self.start("--root", str(SB.dir), "--model-file", "linear.cml", "--no-token", "--port", str(first.port))
            self.assertEqual((code, d["error"]["rule"]), (5, "serve.listen"))

    def test_introspect_and_skill(self):
        p = subprocess.run([SERVE_BIN, "introspect"], capture_output=True)
        d = json.loads(p.stdout)
        self.assertEqual(d["tool"], "ml-serve")
        self.assertEqual([f["name"] for f in d["flags"]], ["--root", "--model-file", "--port", "--token-file", "--no-token", "--max-body-bytes", "--max-batch-rows", "--idle-seconds", "--max-bytes"])
        self.assertIn("serve.no-token", [r["rule"] for r in d["rules"]])
        p = subprocess.run([SERVE_BIN, "skill"], capture_output=True)
        self.assertEqual(p.returncode, 0)
        self.assertIn(b"ml-serve", p.stdout)

    def test_startup_reads_the_model_once(self):
        # the model file may be removed after start: the server holds its bytes
        import shutil
        shutil.copy(SB.dir / "linear.cml", SB.dir / "gone.cml")
        s = Server(SB.dir, "gone.cml").start()
        try:
            os.remove(SB.dir / "gone.cml")
            r = s.request("POST", "/predict", body=json.dumps(RECORD).encode(), headers=JSON)
            self.assertEqual(r.status, 200, r)
        finally:
            s.stop()

    def test_listens_on_every_address(self):
        """docs/serving-http.md 5.4: the pinned compiler binds every address, so the token rule applies to every start (this connects through the machine's own non-loopback address)."""
        host = socket.gethostbyname(socket.gethostname())
        if host.startswith("127."):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("10.255.255.255", 1))
                host = s.getsockname()[0]
                s.close()
            except OSError:
                self.skipTest("no non-loopback address")
        if host.startswith("127."):
            self.skipTest("no non-loopback address")
        with Server(SB.dir, "linear.cml") as srv:
            c = socket.create_connection((host, srv.port), timeout=5)
            c.sendall(Conn.build("GET", "/health"))
            self.assertIn(b"200", c.recv(100))
            c.close()


if __name__ == "__main__":
    unittest.main()

"""Gate D of docs/mcp.md: malformed and hostile input. Every case gets an answer of the contract (a JSON-RPC response, or silence for a notification), the process neither
traps nor hangs, and the next request is served."""

import json
import random
import unittest

import mcp_client as mc
from harness import Sandbox


def ping(c, id=9001):
    r = c.request("ping", id=id)
    assert r == {"jsonrpc": "2.0", "id": id, "result": {}}, r


class Hostile(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        self.addCleanup(self.sb.cleanup)
        self.sb.write("reg.csv", "id,x,y\n1,2.5,10\n2,3.5,20\n3,1.0,30\n4,0.25,41\n")

    def client(self, *flags, init=True):
        c = mc.Client("--root", self.sb.dir, *flags, cwd=self.sb.dir)
        self.addCleanup(c.kill)
        if init:
            c.initialize()
        return c

    def err(self, c, line, code, id="null"):
        """Send `line` (bytes or text), expect a JSON-RPC error with `code` and the id token `id`, and that the server still answers."""
        c.send_line(line)
        r = c.read()
        self.assertEqual(r["error"]["code"], code, r)
        self.assertEqual(json.dumps(r["id"]), id)
        self.assertEqual(r["jsonrpc"], "2.0")
        self.assertIn("message", r["error"])
        ping(c)
        return r

    def test_invalid_json(self):
        c = self.client()
        for line in ['{', '{"jsonrpc":', '}{', 'nul', '{"a":1,}', "{'a':1}", '{"a":01}', '{"a":NaN}', '{"a":"\\x"}', '[1,2', '"just a string"x', '{"jsonrpc":"2.0","id":1,"method":"ping"} trailing']:
            self.err(c, line, -32700)

    def test_not_a_message(self):
        c = self.client()
        for line in ['1', '"x"', 'true', 'null', '[]', '[1]', '[{"jsonrpc":"2.0","id":1,"method":"ping"}]']:
            self.err(c, line, -32600)

    def test_empty_lines_are_not_messages(self):
        c = self.client()
        c.write(b"\n\n   \n\r\n")
        self.assertTrue(c.expect_silence(0.3))
        ping(c)

    def test_a_batch_is_refused_whole(self):
        c = self.client()
        r = self.err(c, '[{"jsonrpc":"2.0","id":1,"method":"ping"},{"jsonrpc":"2.0","id":2,"method":"ping"}]', -32600)
        self.assertIn("batch", r["error"]["message"])
        self.assertTrue(c.expect_silence(0.2))

    def test_unknown_method(self):
        c = self.client()
        r = c.request("no/such/method", {}, id=7)
        self.assertEqual((r["id"], r["error"]["code"]), (7, -32601))
        r = c.request("server/discover", {}, id="x")
        self.assertEqual((r["id"], r["error"]["code"]), ("x", -32601))
        r = c.request("resources/list", {}, id=8)
        self.assertEqual(r["error"]["code"], -32601)
        r = c.request("tools/list", {"cursor": "abc"}, id=9)
        self.assertEqual(r["error"]["code"], -32602)
        r = c.request("tools/list", {"cursor": ""}, id=10)
        self.assertIn("tools", r["result"])

    def test_a_call_before_initialize(self):
        c = self.client(init=False)
        for method, params in (("tools/list", None), ("tools/call", {"name": "describe", "arguments": {}}), ("resources/list", None)):
            r = c.request(method, params, id=3)
            self.assertEqual(r["error"]["code"], -32002, r)
        ping(c)
        r = c.initialize()
        self.assertIn("result", r)
        self.assertIn("tools", c.request("tools/list")["result"])

    def test_initialize_is_checked_and_negotiates(self):
        c = self.client(init=False)
        r = c.request("initialize", {"protocolVersion": "1900-01-01", "capabilities": {}, "clientInfo": {}}, id=1)
        self.assertEqual(r["result"]["protocolVersion"], mc.REVISION)
        for params in (None, [], {}, {"protocolVersion": 5}, "x"):
            r = c.request("initialize", params, id=2)
            self.assertEqual(r["error"]["code"], -32602, params)

    def test_wrong_arguments(self):
        self.sb.write("reg.csv", "id,x,y\n1,2.5,10\n2,3.5,20\n")
        self.assertEqual(self.sb.run("fit", "--target", "y", "--output", "m.cml", "reg.csv").status, 0)
        c = self.client("--allow-write")
        bad = [
            ("describe", {"model-file": 5}),
            ("describe", {"model-file": "m.cml", "limit": "10"}),
            ("describe", {"model-file": "m.cml", "limit": 1.5}),
            ("describe", {"model-file": "m.cml", "limit": -1}),
            ("describe", {"model-file": "m.cml", "limit": 1e3}),
            ("describe", {"model-file": ["m.cml"]}),
            ("describe", {"model-file": None}),
            ("describe", {"model-file": "m\u0000.cml"}),
            ("describe", {"nope": 1}),
            ("describe", {"model-file": "m.cml", "format": "text"}),
            ("describe", {"model-file": "m.cml", "root": "/"}),
            ("fit", {"file": "reg.csv", "output": "o.cml", "target": "y", "overwrite": "yes"}),
            ("fit", {"file": "reg.csv", "output": "o.cml", "target": "y", "model": 3}),
            ("runs_show", {"dir": 1, "id": 2}),
            ("predict", {"model-file": "m.cml", "records": "rows"}),
            ("predict", {"model-file": "m.cml", "records": [1, 2]}),
            ("no_such_tool", {}),
        ]
        for name, args in bad:
            r = c.call(name, args)
            self.assertEqual(r["error"]["code"], -32602, (name, args, r))
            self.assertEqual(r["jsonrpc"], "2.0")
        for params in (None, {}, {"name": 3}, {"name": "describe", "arguments": [1]}, {"name": "describe", "arguments": "x"}, {"name": "describe", "arguments": 5}, [], "x"):
            r = c.request("tools/call", params)
            self.assertEqual(r["error"]["code"], -32602, params)
        ping(c)

    def test_a_key_twice(self):
        c = self.client()
        c.send_line('{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"describe","arguments":{"model-file":"a","model-file":"b"}}}')
        self.assertEqual(c.read()["error"]["code"], -32602)
        ping(c)

    def test_the_same_id_twice(self):
        c = self.client()
        for _ in range(3):
            self.assertEqual(c.request("ping", id=5), {"jsonrpc": "2.0", "id": 5, "result": {}})
        self.assertEqual(c.request("ping", id="5")["id"], "5")

    def test_ids_of_every_json_type(self):
        c = self.client()
        for token in ("0", "1", "-7", "123456789012345678", "\"a\"", "\"\"", "\"\\u00e9\\n\""):
            c.send_line('{"jsonrpc":"2.0","id":%s,"method":"ping"}' % token)
            line = c.read_line()
            # the token is echoed as written
            self.assertEqual(line, ('{"jsonrpc":"2.0","id":%s,"result":{}}' % token).encode(), token)
        for token in ("1.5", "1e3", "true", "false", "null", "{}", "[]", '{"a":1}', "[1]"):
            c.send_line('{"jsonrpc":"2.0","id":%s,"method":"ping"}' % token)
            r = c.read()
            self.assertEqual((r["id"], r["error"]["code"]), (None, -32600), token)
        ping(c)

    def test_jsonrpc_member(self):
        c = self.client()
        for line, ident in (('{"id":1,"method":"ping"}', "1"), ('{"jsonrpc":"1.0","id":2,"method":"ping"}', "2"), ('{"jsonrpc":2,"id":3,"method":"ping"}', "3"), ('{"jsonrpc":"2.0","id":4}', "null"), ('{"jsonrpc":"2.0","id":5,"method":7}', "null")):
            self.err(c, line, -32600, ident)

    def test_notifications_get_no_answer(self):
        c = self.client()
        c.notify("notifications/initialized")
        c.notify("notifications/cancelled", {"requestId": 1, "reason": "x"})
        c.notify("notifications/cancelled")
        c.notify("notifications/progress", {"progressToken": "t", "progress": 1})
        c.notify("no/such/notification", {"a": 1})
        c.send_line('{"jsonrpc":"2.0","id":5,"result":{}}')
        c.send_line('{"jsonrpc":"2.0","id":6,"error":{"code":1,"message":"m"}}')
        self.assertTrue(c.expect_silence(0.4))
        ping(c)

    def test_cancellation_of_a_call_is_read_after_the_answer(self):
        with open(self.sb.dir / "big.csv", "w") as f:
            f.write("a,b,y\n")
            for i in range(5000):
                f.write("%d,%d,%d\n" % (i, i * 7 % 13, i % 11))
        c = self.client("--allow-write")
        req = {"jsonrpc": "2.0", "id": 77, "method": "tools/call", "params": {"name": "fit", "arguments": {"file": "big.csv", "target": "y", "output": "big.cml", "model": "linear"}}}
        # the request and its cancellation in one write: the call runs to its end, the answer arrives, the cancellation names a request that is over
        c.write((json.dumps(req) + "\n" + json.dumps({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 77}}) + "\n").encode())
        r = c.read(120)
        self.assertEqual(r["id"], 77)
        self.assertFalse(r["result"]["isError"], r)
        self.assertTrue(c.expect_silence(0.3))
        self.assertTrue((self.sb.dir / "big.cml").exists())
        ping(c)

    def test_message_size_limit(self):
        c = self.client("--max-message-bytes", "300")
        pad = "x" * 100
        small = '{"jsonrpc":"2.0","id":1,"method":"ping","params":{"p":"%s"}}' % pad
        self.assertLess(len(small), 300)
        c.send_line(small)
        self.assertIn("result", c.read())
        # exactly at the limit is read; one byte more is refused before it is kept
        at = '{"jsonrpc":"2.0","id":2,"method":"ping","params":{"p":"%s"}}' % ("y" * (300 - len('{"jsonrpc":"2.0","id":2,"method":"ping","params":{"p":""}}')))
        self.assertEqual(len(at), 300)
        c.send_line(at)
        self.assertIn("result", c.read())
        over = at[:-2] + "z" + at[-2:]
        self.assertEqual(len(over), 301)
        self.err(c, over, -32600)
        c.send_line(b"q" * 1000000)
        r = c.read()
        self.assertEqual(r["error"]["code"], -32600)
        self.assertIn("--max-message-bytes", r["error"]["message"])
        ping(c)

    def test_the_default_limit_is_a_mebibyte(self):
        c = self.client()
        ok = '{"jsonrpc":"2.0","id":1,"method":"ping","params":{"p":"%s"}}' % ("a" * (1048576 - 60))
        self.assertLessEqual(len(ok), 1048576)
        c.send_line(ok)
        self.assertIn("result", c.read())
        c.send_line(b'{"jsonrpc":"2.0","id":1,"method":"ping","params":{"p":"' + b"a" * 1048576 + b'"}}')
        self.assertEqual(c.read()["error"]["code"], -32600)
        ping(c)

    def test_the_limit_flag_is_checked(self):
        for v in ("0", "16777217", "x", "-1"):
            c = mc.Client("--max-message-bytes", v)
            status, err = c.finish()
            self.assertEqual(status, 2, v)
            self.assertTrue(err, v)
        c = mc.Client("--max-message-bytes", "16777216")
        c.initialize()
        ping(c)
        self.assertEqual(c.finish()[0], 0)

    def test_nesting(self):
        c = self.client()
        for depth in (100, 1000, 100000):
            self.err(c, "[" * depth + "]" * depth, -32600 if depth <= 400 else -32700, "null")
            line = '{"jsonrpc":"2.0","id":1,"method":"ping","params":' + "[" * depth + "]" * depth + "}"
            c.send_line(line)
            r = c.read()
            self.assertIn("jsonrpc", r)
        ping(c)

    def test_a_million_elements_within_the_limit(self):
        c = self.client("--max-message-bytes", "4000000")
        c.send_line('{"jsonrpc":"2.0","id":1,"method":"ping","params":{"a":[' + ",".join(["1"] * 1000000) + "]}}")
        self.assertIn("result", c.read(120))
        ping(c)

    def test_invalid_utf8(self):
        c = self.client()
        for raw in (b'{"jsonrpc":"2.0","id":1,"method":"ping","x":"\xff"}', b'{"jsonrpc":"2.0","id":1,"method":"ping","x":"\xc3("}', b'\xef\xbb\xbf{"jsonrpc":"2.0","id":1,"method":"ping"}',
                    b'{"jsonrpc":"2.0","id":1,"method":"ping","x":"\xed\xa0\x80"}', b'{"jsonrpc":"2.0","id":1,"method":"ping","x":"\\ud800"}', b'{"jsonrpc":"2.0","id":1,"method":"ping","x":"a\x01b"}'):
            self.err(c, raw, -32700)

    def test_nul_and_control_bytes(self):
        c = self.client()
        for raw in (b"\x00", b"\x00\x00\x00", b'{"jsonrpc":"2.0",\x00"id":1,"method":"ping"}', b"\x1b[31m", b"\x7f"):
            c.send_line(raw)
            line = c.read_line()
            if line is not None:
                self.assertEqual(json.loads(line)["error"]["code"], -32700, raw)
        ping(c)

    def test_a_carriage_return_ends_nothing_but_the_line(self):
        c = self.client()
        c.write(b'{"jsonrpc":"2.0","id":1,"method":"ping"}\r\n')
        self.assertEqual(c.read()["id"], 1)
        # a carriage return inside a line is JSON whitespace
        c.write(b'{"jsonrpc":"2.0","id":2,\r"method":"ping"}\n')
        self.assertEqual(c.read()["id"], 2)
        c.write(b'{"jsonrpc":"2.0","id":3,"method":"ping"}\r\r\n')
        self.assertEqual(c.read()["id"], 3)
        ping(c)

    def test_one_byte_chunks(self):
        c = self.client(init=False)
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": mc.REVISION, "capabilities": {}, "clientInfo": {"name": "t"}}}
        c.write_slowly((json.dumps(init) + "\n").encode())
        self.assertIn("result", c.read())
        c.write_slowly(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        c.write_slowly((json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n").encode())
        self.assertEqual(len(c.read()["result"]["tools"]), 9)
        self.sb.write("m.csv", "id,x,y\n1,2.5,10\n2,3.5,20\n")
        c.write_slowly((json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "describe", "arguments": {"model-file": "absent.cml"}}}) + "\n").encode(), 3)
        r = c.read()
        self.assertTrue(r["result"]["isError"])
        self.assertEqual(r["result"]["structuredContent"]["error"]["rule"], "io.not-found")

    def test_eof_in_the_middle_of_a_message(self):
        c = self.client()
        c.write(b'{"jsonrpc":"2.0","id":1,"meth')
        c.close_input()
        r = c.read()
        self.assertEqual(r["error"]["code"], -32700)
        status, err = c.finish()
        self.assertEqual(status, 0)
        self.assertEqual(err, b"")

    def test_eof_after_a_complete_message_without_a_line_feed(self):
        c = self.client()
        c.write(b'{"jsonrpc":"2.0","id":1,"method":"ping"}')
        c.close_input()
        self.assertEqual(c.read()["result"], {})
        self.assertEqual(c.finish()[0], 0)

    def test_eof_at_once(self):
        c = mc.Client()
        status, err = c.finish()
        self.assertEqual((status, err), (0, b""))

    def test_every_exit_code_of_the_command_line_is_a_tool_result(self):
        self.sb.write("reg.csv", "id,x,y\n1,2.5,10\n2,3.5,20\n3,1.0,30\n4,0.25,41\n")
        self.sb.write("txt.csv", "a,y\nhello,1\nworld,2\n")
        c = self.client("--allow-write")
        ok = c.call("fit", {"file": "reg.csv", "target": "y", "output": "m.cml"})["result"]
        self.assertEqual((ok["isError"], ok["_meta"]["ml/exit"]), (False, 0))
        dry = c.call("fit", {"file": "reg.csv", "target": "y", "output": "m.cml", "overwrite": True, "dry-run": True})["result"]
        self.assertEqual((dry["isError"], dry["_meta"]["ml/exit"], dry["structuredContent"]["dry_run"]), (False, 9, True))
        cases = [
            ({"file": "reg.csv", "output": "x.cml"}, 2, "args.required-flag"),
            ({"file": "gone.csv", "target": "y", "output": "x.cml"}, 3, "io.not-found"),
            ({"file": "../reg.csv", "target": "y", "output": "x.cml"}, 2, "path.dotdot"),
            ({"file": "/etc/hosts", "target": "y", "output": "x.cml"}, 4, "path.outside-root"),
            ({"file": "reg.csv", "target": "y", "output": "m.cml"}, 5, "conflict.exists"),
            ({"file": "txt.csv", "target": "y", "output": "x.cml"}, 8, "ml.input.non-numeric"),
        ]
        for args, exit_code, rule in cases:
            r = c.call("fit", args)["result"]
            self.assertTrue(r["isError"], args)
            self.assertEqual((r["_meta"]["ml/exit"], r["structuredContent"]["error"]["rule"]), (exit_code, rule), args)
        ping(c)

    def test_random_lines(self):
        rng = random.Random(45)
        c = self.client()
        pieces = [b'{"jsonrpc":"2.0",', b'"id":', b'1', b'"s"', b',"method":', b'"ping"', b'"tools/call"', b'"tools/list"', b',"params":{', b'"name":"describe"', b'"arguments":{}', b'}', b'{', b'[', b']', b'"', b'\\', b'\xff', b'\x00', b'null', b'true', b'1e999', b'-', b'0.5', b',']
        answered = 0
        for n in range(400):
            line = b"".join(rng.choice(pieces) for _ in range(rng.randint(1, 14))).replace(b"\n", b" ")
            c.send_line(line)
            # a marker request after the line: whatever the line earned (a response, or nothing) arrives before the marker's answer
            marker = 10_000_000 + n
            c.send_line('{"jsonrpc":"2.0","id":%d,"method":"ping"}' % marker)
            while True:
                got = c.read_line(30)
                self.assertIsNotNone(got, "the server ended")
                msg = json.loads(got)
                self.assertEqual(msg["jsonrpc"], "2.0")
                if msg.get("id") == marker:
                    break
                answered += 1
                self.assertTrue("result" in msg or "error" in msg, msg)
                if "error" in msg:
                    self.assertIn(msg["error"]["code"], (-32700, -32600, -32601, -32602, -32002))
        ping(c, 424242)
        self.assertTrue(c.alive())
        self.assertGreater(answered, 20)
        status, err = c.finish()
        self.assertEqual((status, err), (0, b""))


if __name__ == "__main__":
    unittest.main()

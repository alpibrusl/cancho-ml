"""Gates A and C of docs/mcp.md: the protocol client calls the tools of `ml-mcp` and the answers are the answers of `ml`, byte for byte.

* A: every fixture of the command line (fixtures.py) that a tool call can express is run three times in the same scratch directory: by the command line with the
  arguments in the order the server builds them (argv[0] `ml`), by the server (`tools/call`), and by the command line in `--format text`. The `structuredContent` of the call is the
  line the command line printed (the same bytes), the exit code of the equivalent command is `_meta` `ml/exit`, `isError` is whether the command refused, and `content[0].text` is the
  text form of a success.
* C: each class of refusal reaches the client as data (the rule, the exit code and the repair), `runs_verify` of a record runs, the read-only server refuses what writes.
"""

import json
import os
import pathlib
import shutil
import subprocess
import unittest

import mcp_client as mc
from fixtures import CASES
from harness import BIN, ROOT, Sandbox, introspect, runs_here
from runner import apply_setup, first_run_id, substitute

TOOLS = json.loads((ROOT / "docs" / "mcp" / "tools.json").read_text())
CALLS = TOOLS["calls"]
FLAGS = introspect()["flags"]
FLAG = {f["name"][2:]: f for f in FLAGS}


def translate(argv):
    """The (tool, arguments) a command line of the fixtures says, or None when no tool call says it (a flag the server owns, a form the JSON cannot hold)."""
    flags, operands, i = {}, [], 0
    while i < len(argv):
        t = argv[i]
        if t.startswith("--"):
            if "=" in t:
                return None
            f = FLAG.get(t[2:])
            if f is None or t[2:] in flags:
                return None
            if f["kind"] == "bool":
                flags[t[2:]] = True
                i += 1
            else:
                if i + 1 >= len(argv):
                    return None
                flags[t[2:]] = argv[i + 1]
                i += 2
        elif t.startswith("-") and t != "-":
            return None
        else:
            operands.append(t)
            i += 1
    if not operands or operands[0] in ("introspect", "skill"):
        return None
    if operands[0] == "runs":
        if len(operands) < 2:
            return None
        tool, rest = "runs_" + operands[1], operands[2:]
    else:
        tool, rest = operands[0], operands[1:]
    if tool not in CALLS:
        return None
    c = CALLS[tool]
    props = {p["name"]: p for p in c["props"]}
    args = {}
    for name, v in flags.items():
        if name not in props:
            return None
        kind = props[name]["kind"]
        if kind == "bool":
            args[name] = True
        elif kind == "nat":
            if not (v.isdigit() and len(v) <= 18 and str(int(v)) == v):
                return None
            args[name] = int(v)
        else:
            if v == "" or "\x00" in v:
                return None
            args[name] = v
    if len(rest) > len(c["operands"]):
        return None
    for name, v in zip(c["operands"], rest):
        if v == "-" or v == "" or "\x00" in v:
            return None
        args[name] = v
    return tool, args


def canonical(tool, args, root):
    """The command line the server builds for a call: flags in the order of the table, the verb, the subcommand, the operands."""
    c = CALLS[tool]
    out = ["ml"]
    for f in FLAGS:
        n = f["name"][2:]
        if n == "root":
            out += ["--root", str(root)]
        elif n in args:
            if f["kind"] == "bool":
                if args[n]:
                    out.append("--" + n)
            else:
                out += ["--" + n, str(args[n])]
    out.append(c["verb"])
    if c["sub"]:
        out.append(c["sub"])
    out += [str(args[o]) for o in c["operands"] if o in args]
    return out


def cli(argv, cwd):
    p = subprocess.run(argv, executable=BIN, capture_output=True, cwd=cwd, input=b"", timeout=300)
    return p.returncode, p.stdout, p.stderr


def wipe(sb):
    for p in list(sb.dir.iterdir()):
        try:
            os.chmod(p, 0o755)
        except OSError:
            pass
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink()


class CaseGate(unittest.TestCase):
    """A: the fixtures, through the tools."""

    def test_every_fixture_a_call_can_say(self):
        failures, translated, texts = [], 0, 0
        for c in CASES:
            if c.get("skip_if_root") and os.geteuid() == 0:
                continue
            if not runs_here(c) or c.get("close_stdin") or c.get("close_stdout") or c.get("noroot") or c.get("no_json"):
                continue
            sb = Sandbox()
            try:
                def fresh():
                    wipe(sb)
                    apply_setup(sb, c["setup"])

                fresh()
                said = translate(substitute(c["argv"], sb))
                if said is None:
                    continue
                tool, args = said
                translated += 1
                fresh()
                status, out, err = cli(canonical(tool, args, sb.dir), sb.dir)
                fresh()
                with mc.Client("--root", sb.dir, "--allow-write", cwd=sb.dir) as m:
                    m.initialize()
                    resp, raw = m.call_raw(tool, args)
                    res = resp.get("result") if resp else None
                    if res is None:
                        failures.append((c["id"], "no result: %r" % (resp,)))
                        continue
                    want = out.rstrip(b"\n")
                    got = mc.raw_structured(raw)
                    if got != want:
                        failures.append((c["id"], "structuredContent differs:\n  cli %r\n  mcp %r" % (want[:300], got[:300])))
                    if res["_meta"]["ml/exit"] != status:
                        failures.append((c["id"], "exit %r, the command exited %d" % (res["_meta"]["ml/exit"], status)))
                    if res["isError"] != (status not in (0, 9)):
                        failures.append((c["id"], "isError %r with exit %d" % (res["isError"], status)))
                    if status == 0:
                        fresh()
                        s2, text, _ = cli(canonical(tool, args, sb.dir) + ["--format", "text"], sb.dir)
                        if s2 == 0:
                            texts += 1
                            expect = text.decode() if text else want.decode()
                            if res["content"][0]["text"] != expect:
                                failures.append((c["id"], "text %r, the command prints %r" % (res["content"][0]["text"][:120], expect[:120])))
            finally:
                sb.cleanup()
        self.assertEqual(failures, [], "\n".join("%s: %s" % f for f in failures[:20]))
        # the gate cannot pass by translating nothing
        self.assertGreater(translated, 150, "only %d fixtures were expressed as calls" % translated)
        self.assertGreater(texts, 20)


def with_model(sb):
    sb.write("reg.csv", "id,x,y\n1,2.5,10\n2,3.5,20\n3,1.0,30\n4,0.25,41\n")
    r = sb.run("fit", "--target", "y", "--output", "m.cml", "reg.csv")
    assert r.status == 0


class Tools(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        self.addCleanup(self.sb.cleanup)

    def server(self, *flags):
        c = mc.Client("--root", self.sb.dir, *flags, cwd=self.sb.dir)
        self.addCleanup(c.kill)
        c.initialize()
        return c

    def test_handshake_and_list(self):
        c = mc.Client("--root", self.sb.dir, cwd=self.sb.dir)
        self.addCleanup(c.kill)
        r = c.initialize()
        self.assertEqual(r["result"]["protocolVersion"], mc.REVISION)
        self.assertEqual(r["result"]["capabilities"], {"tools": {"listChanged": False}})
        self.assertEqual(r["result"]["serverInfo"]["name"], "ml-mcp")
        names = [t["name"] for t in c.list_tools()]
        self.assertEqual(names, [t["name"] for t in TOOLS["readonly_tools"]])
        self.assertNotIn("fit", names)
        self.assertEqual(c.request("ping")["result"], {})

    def test_the_list_with_write_is_the_generated_one(self):
        c = self.server("--allow-write")
        self.assertEqual(c.list_tools(), TOOLS["tools"])

    def test_the_list_without_write_is_the_generated_one(self):
        c = self.server()
        self.assertEqual(c.list_tools(), TOOLS["readonly_tools"])

    def test_read_only_refuses_what_writes(self):
        with_model(self.sb)
        c = self.server()
        r = c.call("fit", {"file": "reg.csv", "target": "y", "output": "n.cml"})["result"]
        self.assertTrue(r["isError"])
        self.assertEqual(r["structuredContent"]["error"]["rule"], "mcp.read-only")
        self.assertEqual(r["_meta"]["ml/exit"], 4)
        self.assertFalse((self.sb.dir / "n.cml").exists())
        r = c.call("eval", {"model-file": "m.cml", "file": "reg.csv", "track": "runs"})["result"]
        self.assertEqual(r["structuredContent"]["error"]["rule"], "mcp.read-only")
        self.assertEqual(r["structuredContent"]["error"]["detail"], {"flag": "track"})
        # a read goes through
        r = c.call("describe", {"model-file": "m.cml"})["result"]
        self.assertFalse(r["isError"])

    def test_write_needs_a_root(self):
        c = mc.Client("--allow-write")
        status, err = c.finish()
        self.assertEqual(status, 2)
        self.assertIn(b"--root", err)

    def test_introspect_is_the_table(self):
        c = self.server()
        r = c.call("introspect", {})["result"]
        doc = r["structuredContent"]
        ref = introspect()
        for k in ("tool", "flags", "rules", "limits", "exit_codes", "operands", "usage"):
            self.assertEqual(doc[k], ref[k], k)
        self.assertEqual(doc["authority"]["effects"], ref["authority"]["effects"])

    def test_records_are_the_file(self):
        with_model(self.sb)
        self.sb.write("new.csv", "id,x\n5,2.0\n6,4.5\n")
        c = self.server()
        a = c.call("predict", {"model-file": "m.cml", "file": "new.csv"})["result"]["structuredContent"]
        b = c.call("predict", {"model-file": "m.cml", "records": [{"id": 5, "x": 2.0}, {"id": 6, "x": 4.5}]})["result"]["structuredContent"]
        self.assertEqual(a, b)
        # a number that cannot be written exactly
        r = c.call("predict", {"model-file": "m.cml", "records": [{"id": 5, "x": 2.0}, {"id": 6, "x": 4.123456789012345678}]})["result"]
        self.assertTrue(r["isError"])
        self.assertEqual(r["structuredContent"]["error"]["rule"], "serve.number-not-exact")
        self.assertEqual(r["structuredContent"]["error"]["detail"]["pointer"], "/arguments/records/1/x")
        r = c.call("predict", {"model-file": "m.cml", "records": [{"id": 5, "x": 2.0, "nope": 1}]})["result"]
        self.assertEqual(r["structuredContent"]["error"]["rule"], "mcp.record-column")
        # a file and records, or neither
        e = c.call("predict", {"model-file": "m.cml", "file": "new.csv", "records": [{"id": 5, "x": 2.0}]})
        self.assertEqual(e["error"]["code"], -32602)
        e = c.call("predict", {"model-file": "m.cml"})
        self.assertEqual(e["error"]["code"], -32602)

    def test_runs_and_verify(self):
        self.sb.write("reg.csv", "id,x,y\n1,2.5,10\n2,3.5,20\n3,1.0,30\n4,0.25,41\n")
        (self.sb.dir / "runs").mkdir()
        r = self.sb.run("fit", "--target", "y", "--output", "m.cml", "--track", "runs", "reg.csv")
        self.assertEqual(r.status, 0, r)
        c = self.server("--allow-write")
        lst = c.call("runs_list", {"dir": "runs"})["result"]["structuredContent"]["data"]
        self.assertEqual(lst["count"], 1)
        rid = lst["runs"][0]["id"]
        shown = c.call("runs_show", {"dir": "runs", "id": rid})["result"]["structuredContent"]["data"]
        self.assertEqual(shown["record"]["id"], rid)
        cmp_ = c.call("runs_compare", {"dir": "runs", "id_a": rid, "id_b": rid})["result"]["structuredContent"]["data"]
        self.assertTrue(cmp_["model_equal"])
        ver = c.call("runs_verify", {"dir": "runs", "id": rid})["result"]["structuredContent"]["data"]
        self.assertTrue(ver["verified"])
        self.assertEqual(ver["id"], rid)
        bad = c.call("runs_show", {"dir": "runs", "id": "0" * 64})["result"]
        self.assertTrue(bad["isError"])
        self.assertEqual(bad["structuredContent"]["error"]["rule"], "ml.runs.not-found")

    def test_refusals_are_data(self):
        with_model(self.sb)
        c = self.server("--allow-write")
        cases = [
            ("fit", {"file": "reg.csv", "output": "m.cml", "target": "y"}, "conflict.exists", 5),
            ("fit", {"file": "nope.csv", "output": "n.cml", "target": "y"}, "io.not-found", 3),
            ("fit", {"file": "../reg.csv", "output": "n.cml", "target": "y"}, "path.dotdot", 2),
            ("fit", {"file": "/etc/passwd", "output": "n.cml", "target": "y"}, "path.outside-root", 4),
            ("fit", {"file": str(self.sb.dir / "reg.csv"), "output": "n.cml", "target": "y"}, "path.absolute", 2),
            ("fit", {"file": "reg.csv", "output": "n.cml"}, "args.required-flag", 2),
            ("fit", {"file": "reg.csv", "output": "n.cml", "target": "y", "trees": 5}, "args.conflict", 2),
            ("fit", {"file": "reg.csv", "output": "n.cml", "target": "y", "model": "forest", "trees": 99999}, "ml.param.out-of-range", 2),
            ("fit", {"file": "reg.csv", "output": "n.cml", "target": "nope"}, "ml.input.unknown-column", 3),
            ("eval", {"model-file": "reg.csv", "file": "reg.csv"}, "ml.model.bad-magic", 8),
            ("describe", {"model-file": "m.cml", "limit": 99999999999999}, "ml.param.out-of-range", 2),
            ("predict", {"model-file": "m.cml", "file": "reg.csv"}, "ml.schema.mismatch", 8),
        ]
        for tool, args, rule, exit_code in cases:
            r = c.call(tool, args)["result"]
            self.assertTrue(r["isError"], (tool, args))
            e = r["structuredContent"]["error"]
            self.assertEqual((e["rule"], r["_meta"]["ml/exit"]), (rule, exit_code), (tool, args, e))
            self.assertIn("repair", e)
            self.assertIn("hint", e)

    def test_a_repair_is_a_command_the_cli_runs(self):
        self.sb.write("reg.csv", "id,x,y\n1,2.5,10\n2,3.5,20\n3,1.0,30\n4,0.25,41\n")
        c = self.server("--allow-write")
        r = c.call("fit", {"file": "a/../reg.csv", "output": "n.cml", "target": "y"})["result"]
        rep = r["structuredContent"]["error"]["repair"]
        self.assertEqual(rep["kind"], "retry")
        self.assertEqual(rep["argv"][0], "ml")
        p = subprocess.run(rep["argv"], executable=BIN, cwd=self.sb.dir, capture_output=True)
        self.assertEqual(p.returncode, 0, p.stdout)


if __name__ == "__main__":
    unittest.main()

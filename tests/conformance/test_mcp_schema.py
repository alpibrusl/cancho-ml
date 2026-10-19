"""Gate B of docs/mcp.md, the half that needs the server: the tools the server lists are the generated ones, a valid argument set per tool is accepted and an invalid one is refused with
the refusal of the command line, and the real answer of every tool validates against its `outputSchema`."""

import json
import pathlib
import sys
import unittest

import jsonschema

import mcp_client as mc
from harness import ROOT, Sandbox, introspect
from test_mcp import CALLS, TOOLS, canonical, cli

sys.path.insert(0, str(ROOT / "scripts"))
import check_mcp_tools  # noqa: E402

V = jsonschema.Draft202012Validator

REG = "id,x,y\n1,2.5,10\n2,3.5,20\n3,1.0,30\n4,0.25,41\n"
CLS = "a,b,lab\n1,2,0\n2,1,0\n3,4,1\n4,1,1\n5,6,0\n6,3,1\n7,9,1\n8,2,0\n9,7,1\n10,1,0\n3,3,0\n6,6,1\n"


class Schema(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()
        self.addCleanup(self.sb.cleanup)
        self.sb.write("reg.csv", REG)
        self.sb.write("cls.csv", CLS)
        self.sb.write("lin.csv", "a,b,y\n1,2,5.1\n2,1,7.9\n3,4,9.3\n4,5,12.2\n5,6,13.8\n6,3,15.1\n7,9,19.7\n8,2,17.1\n")
        (self.sb.dir / "runs").mkdir()
        self.c = mc.Client("--root", self.sb.dir, "--allow-write", cwd=self.sb.dir)
        self.addCleanup(self.c.kill)
        self.c.initialize()

    def tool(self, name):
        return next(t for t in TOOLS["tools"] if t["name"] == name)

    def test_the_listed_tools_validate_as_schemas(self):
        for t in self.c.list_tools():
            V.check_schema(t["inputSchema"])
            V.check_schema(t["outputSchema"])

    def test_a_valid_argument_set_is_accepted_by_the_server(self):
        # the schema says what a client may send; the server answers it with a tool result (never a protocol error), whatever the library says of the values
        for t in self.c.list_tools():
            s = t["inputSchema"]
            v = V(s)
            for boundary in (False, True):
                args = check_mcp_tools.base_args(s)
                for p, ps in s["properties"].items():
                    if p != "records":
                        args[p] = check_mcp_tools.sample(ps, boundary)
                args.pop("records", None)
                args["file"] = args.get("file", "x.csv") if "file" in s["properties"] else None
                args = {k: v_ for k, v_ in args.items() if v_ is not None}
                self.assertEqual(list(v.iter_errors(args)), [], (t["name"], args))
                r = self.c.call(t["name"], args)
                self.assertIn("result", r, (t["name"], args, r))
                self.assertIn("structuredContent", r["result"])

    def test_an_invalid_argument_is_refused_by_the_schema_and_by_the_library(self):
        self.assertEqual(self.sb.run("fit", "--target", "y", "--output", "m.cml", "reg.csv").status, 0)
        checked = 0
        for name, c in CALLS.items():
            t = self.tool(name)
            s = t["inputSchema"]
            for p in c["props"]:
                ps = s["properties"].get(p["name"])
                if ps is None or "maximum" not in ps:
                    continue
                args = {"file": "reg.csv", "model-file": "m.cml", "dir": "runs", "id": "0" * 64, "id_a": "0" * 64, "id_b": "0" * 64, "target": "y", "output": "z.cml"}
                args = {k: v for k, v in args.items() if k in s["properties"]}
                args[p["name"]] = ps["maximum"] + 1
                self.assertNotEqual(list(V(s).iter_errors(args)), [], (name, p["name"]))
                r = self.c.call(name, args)["result"]
                self.assertTrue(r["isError"], (name, args))
                status, out, _ = cli(canonical(name, args, self.sb.dir), self.sb.dir)
                self.assertEqual(mc.raw_structured(json.dumps(r).encode()) if False else json.dumps(r["structuredContent"], separators=(",", ":"), ensure_ascii=False).encode(), out.rstrip(b"\n"), (name, p["name"]))
                checked += 1
        self.assertGreater(checked, 25)

    def test_the_answer_of_every_tool_validates_against_its_output_schema(self):
        calls = [
            ("fit", {"file": "reg.csv", "target": "y", "output": "m.cml"}),
            ("fit", {"file": "reg.csv", "target": "y", "output": "m.cml", "overwrite": True, "dry-run": True}),
            ("fit", {"file": "reg.csv", "target": "y", "output": "lin.cml", "model": "linear", "track": "runs"}),
            ("fit", {"file": "cls.csv", "target": "lab", "output": "log.cml", "model": "logistic", "task": "classification"}),
            ("fit", {"file": "cls.csv", "target": "lab", "output": "tree.cml", "model": "tree", "task": "classification", "min-leaf": 1}),
            ("fit", {"file": "reg.csv", "output": "km.cml", "model": "kmeans", "k": 2}),
            ("predict", {"model-file": "m.cml", "file": "reg.csv"}),
            ("predict", {"model-file": "log.cml", "file": "cls.csv", "proba": True}),
            ("predict", {"model-file": "km.cml", "file": "reg.csv", "distance": True}),
            ("predict", {"model-file": "m.cml", "records": [{"id": 9, "x": 1.5}]}),
            ("eval", {"model-file": "m.cml", "file": "reg.csv"}),
            ("eval", {"model-file": "log.cml", "file": "cls.csv"}),
            ("eval", {"model-file": "lin.cml", "file": "reg.csv", "track": "runs"}),
            ("cv", {"file": "lin.csv", "target": "y", "model": "linear", "folds": 2}),
            ("describe", {"model-file": "m.cml"}),
            ("describe", {"model-file": "km.cml"}),
            ("introspect", {}),
        ]
        for name, args in calls:
            r = self.c.call(name, args)["result"]
            self.assertFalse(r["isError"], (name, args, r["structuredContent"]))
            errs = list(V(self.tool(name)["outputSchema"]).iter_errors(r["structuredContent"]))
            self.assertEqual([e.message[:200] for e in errs], [], (name, args))
        rid = self.c.call("runs_list", {"dir": "runs"})["result"]["structuredContent"]["data"]["runs"][0]["id"]
        for name, args in (("runs_list", {"dir": "runs"}), ("runs_show", {"dir": "runs", "id": rid}), ("runs_compare", {"dir": "runs", "id_a": rid, "id_b": rid}), ("runs_verify", {"dir": "runs", "id": rid})):
            r = self.c.call(name, args)["result"]
            self.assertFalse(r["isError"], (name, r["structuredContent"]))
            self.assertEqual([e.message[:200] for e in V(self.tool(name)["outputSchema"]).iter_errors(r["structuredContent"])], [], name)

    def test_a_refusal_validates_against_the_output_schema(self):
        for name, args in (("fit", {"file": "nope.csv", "target": "y", "output": "z.cml"}), ("describe", {"model-file": "reg.csv"}), ("fit", {"file": "reg.csv", "target": "y", "output": "m.cml", "trees": 3}),
                           ("predict", {"model-file": "absent.cml", "records": [{"x": 1}]}), ("runs_show", {"dir": "runs", "id": "1" * 64})):
            r = self.c.call(name, args)["result"]
            self.assertTrue(r["isError"], name)
            self.assertEqual([e.message[:200] for e in V(self.tool(name)["outputSchema"]).iter_errors(r["structuredContent"])], [], (name, r["structuredContent"]["error"]["rule"]))
        # the rules of the server itself are in the enum too
        ro = mc.Client("--root", self.sb.dir, cwd=self.sb.dir)
        self.addCleanup(ro.kill)
        ro.initialize()
        r = ro.call("fit", {"file": "reg.csv", "target": "y", "output": "q.cml"})["result"]
        self.assertEqual(r["structuredContent"]["error"]["rule"], "mcp.read-only")
        self.assertEqual(list(V(self.tool("fit")["outputSchema"]).iter_errors(r["structuredContent"])), [])


if __name__ == "__main__":
    unittest.main()

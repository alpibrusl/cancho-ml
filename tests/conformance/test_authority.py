"""A1 -- the authority manifest is derived by the compiler, committed, embedded in the binary and gated against a ceiling.

`scripts/manifest.py --check` fails when the committed manifest, the generated module or the binary's `introspect` differ from a
fresh `cancho authority`, or when a derived label is outside the ceiling of tools.toml. The ceiling here is exact (no slack):
what the tool is allowed is what it does."""

import json
import pathlib
import subprocess
import sys
import tomllib
import unittest

from harness import PREDICT_BIN, ROOT, introspect
from mcp_client import BIN as MCP_BIN
from serve_lib import SERVE_BIN

EXPECTED = {"args", "conc", "dir_read", "dir_write", "err_write", "file_read", "file_write", 'fs_read("")', "heap", "io_read", "io_write"}
# `ml-predict` (docs/serving.md section 3.3): the row of `ml` without the writes and the thread
EXPECTED_PREDICT = {"args", "dir_read", "err_write", "file_read", 'fs_read("")', "heap", "io_read", "io_write"}
# `ml-serve` (docs/serving-http.md section 6): a server of cancho-web. No write of any file, no thread, no exec, no foreign code, no net_out, no standard input
EXPECTED_SERVE = {"args", "clock", "conn_accept", "conn_read", "conn_write", "dir_read", "err_write", "file_read", 'fs_read("")', "heap", "io_write", 'net_in("0.0.0.0:")', "poll"}
# `ml-mcp` (docs/mcp.md section 7): the row of `ml`, nothing more: no network, no listener, no clock, no exec, no foreign code
EXPECTED_MCP = EXPECTED
FORBIDDEN = {"ffi", "net_out", "net_in", "clock", "fs_write", "exec", "env"}


def label_text(l):
    return l["name"] if l["argument"] is None else '%s("%s")' % (l["name"], l["argument"])


class Authority(unittest.TestCase):
    def test_manifest_check_passes(self):
        p = subprocess.run([sys.executable, str(ROOT / "scripts" / "manifest.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_the_binary_prints_the_committed_authority(self):
        committed = json.loads((ROOT / "manifests" / "ml.authority.json").read_text())
        self.assertEqual(introspect()["authority"], committed)

    def test_the_authority_is_bounded_and_exactly_these_labels(self):
        a = introspect()["authority"]
        self.assertTrue(a["bounded"])
        self.assertEqual(a["unbounded_by"], [])
        self.assertEqual(a["foreign_symbols"], [])
        self.assertEqual({label_text(l) for l in a["labels"]}, EXPECTED)
        self.assertEqual({label_text(l) for l in a["labels"]} & FORBIDDEN, set())

    def test_the_ceiling_is_exact(self):
        ceilings = tomllib.loads((ROOT / "tools.toml").read_text())
        self.assertEqual(set(ceilings["ml"]["allow"]), EXPECTED)
        self.assertEqual({label_text(l) for l in introspect()["authority"]["labels"]}, set(ceilings["ml"]["allow"]))

    def test_every_label_beyond_the_base_row_has_a_reason(self):
        # docs/wiring.md section 6: conc is allowed because the fits spawn threads, and the ceiling file says why
        ceilings = tomllib.loads((ROOT / "tools.toml").read_text())
        base = set(tomllib.loads((ROOT / "lib" / "models" / "contract.toml").read_text())["authority"]["base"])
        beyond = set(ceilings["ml"]["allow"]) - base
        self.assertEqual(beyond, {"conc", "io_read"})
        self.assertTrue(ceilings["ml"]["reasons"]["conc"])
        self.assertTrue(ceilings["ml"]["reasons"]["io_read"])
        # a model that adds a label declares it; standard input (io_read) is the tool's, not a model's
        import gen_models
        declared = set()
        for m in gen_models.load_manifests(ROOT):
            declared |= set(m["authority"]["adds"])
        self.assertEqual(declared, beyond - {"io_read"})

    def test_a_label_allowed_without_a_reason_fails_the_manifest_check(self):
        import shutil, tempfile
        d = pathlib.Path(tempfile.mkdtemp())
        try:
            text = (ROOT / "tools.toml").read_text()
            text = text.replace('conc = "', 'concX = "')
            (d / "tools.toml").write_text(text)
            sys.path.insert(0, str(ROOT / "scripts"))
            import manifest
            ceilings = tomllib.loads(text)
            auth = json.loads((ROOT / "manifests" / "ml.authority.json").read_text())
            problems = manifest.gate("ml", auth, ceilings)
            self.assertTrue(any("no reason" in p for p in problems), problems)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_no_path_write_no_clock_no_environment_in_the_introspect_claims(self):
        d = introspect()
        self.assertFalse(d["reads_environment"])
        self.assertFalse(d["reads_clock"])
        self.assertTrue(d["integers_only"])

    def test_the_binary_is_the_tool_in_the_manifest(self):
        # one binary per tool; the ceiling file knows only `ml` as an agent tool
        ceilings = tomllib.loads((ROOT / "tools.toml").read_text())
        self.assertEqual([k for k in ceilings if isinstance(ceilings[k], dict)], ["ml", "ml-predict", "ml-serve", "ml-mcp"])


def predict_introspect():
    return json.loads(subprocess.run([PREDICT_BIN, "introspect"], capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout)


class AuthorityPredict(unittest.TestCase):
    """P3: the row of `ml-predict`, derived by the compiler, committed, embedded, within its ceiling."""

    def test_the_binary_prints_the_committed_authority(self):
        committed = json.loads((ROOT / "manifests" / "ml-predict.authority.json").read_text())
        self.assertEqual(predict_introspect()["authority"], committed)

    def test_the_row_is_exactly_these_labels_and_bounded(self):
        a = predict_introspect()["authority"]
        self.assertTrue(a["bounded"])
        self.assertEqual((a["unbounded_by"], a["foreign_symbols"]), ([], []))
        labels = {label_text(l) for l in a["labels"]}
        self.assertEqual(labels, EXPECTED_PREDICT)
        self.assertEqual(labels & FORBIDDEN, set())
        self.assertEqual(labels & {"conc", "file_write", "dir_write"}, set())

    def test_the_ceiling_is_exact_and_every_label_beyond_the_base_row_has_a_reason(self):
        ceilings = tomllib.loads((ROOT / "tools.toml").read_text())
        self.assertEqual(set(ceilings["ml-predict"]["allow"]), EXPECTED_PREDICT)
        base = set(tomllib.loads((ROOT / "lib" / "models" / "contract.toml").read_text())["authority"]["base"])
        self.assertEqual(set(ceilings["ml-predict"]["allow"]) - base, {"io_read"})
        self.assertTrue(ceilings["ml-predict"]["reasons"]["io_read"])
        # ml-predict has no label ml does not have
        self.assertLessEqual(EXPECTED_PREDICT, EXPECTED)

    def test_the_ceiling_check_fails_for_a_capability_added_to_the_derived_row(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import manifest
        ceilings = tomllib.loads((ROOT / "tools.toml").read_text())
        auth = json.loads((ROOT / "manifests" / "ml-predict.authority.json").read_text())
        self.assertEqual(manifest.gate("ml-predict", auth, ceilings), [])
        for extra in ("file_write", "dir_write", "conc", "net_out", "clock", "ffi"):
            mutated = json.loads(json.dumps(auth))
            mutated["labels"].append({"name": extra, "argument": None, "bounded": True})
            self.assertNotEqual(manifest.gate("ml-predict", mutated, ceilings), [], extra)
        mutated = json.loads(json.dumps(auth))
        mutated["labels"].append({"name": "fs_write", "argument": "", "bounded": True})
        self.assertNotEqual(manifest.gate("ml-predict", mutated, ceilings), [])


def serve_introspect():
    return json.loads(subprocess.run([SERVE_BIN, "introspect"], capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout)


class AuthorityServe(unittest.TestCase):
    """A1 (docs/serving-http.md section 9): the row of `ml-serve`, derived by the compiler, committed, embedded, within its ceiling; and what it does not hold."""

    def test_the_binary_prints_the_committed_authority(self):
        committed = json.loads((ROOT / "manifests" / "ml-serve.authority.json").read_text())
        self.assertEqual(serve_introspect()["authority"], committed)

    def test_the_row_is_exactly_these_labels_and_bounded(self):
        a = serve_introspect()["authority"]
        self.assertTrue(a["bounded"])
        self.assertEqual((a["unbounded_by"], a["foreign_symbols"]), ([], []))
        labels = {label_text(l) for l in a["labels"]}
        self.assertEqual(labels, EXPECTED_SERVE)
        # no write of any file, no directory write, no thread, no exec, no foreign code, no outbound network, no standard input, no environment
        self.assertEqual(labels & {"conc", "file_write", "dir_write", "exec", "ffi", "net_out", "io_read", "env", 'fs_write("")', "fs_write"}, set())
        self.assertFalse(any(l["name"] in ("ffi", "exec", "net_out", "fs_write", "file_write", "dir_write", "conc", "env") for l in a["labels"]))

    def test_the_ceiling_is_exact_and_every_label_beyond_the_base_row_has_a_reason(self):
        ceilings = tomllib.loads((ROOT / "tools.toml").read_text())
        self.assertEqual(set(ceilings["ml-serve"]["allow"]), EXPECTED_SERVE)
        base = set(tomllib.loads((ROOT / "lib" / "models" / "contract.toml").read_text())["authority"]["base"])
        for label in set(ceilings["ml-serve"]["allow"]) - base:
            self.assertTrue(ceilings["ml-serve"]["reasons"].get(label.split("(")[0]), label)

    def test_net_in_and_clock_are_the_server_only(self):
        ceilings = tomllib.loads((ROOT / "tools.toml").read_text())
        for tool in ("ml", "ml-predict"):
            self.assertFalse({a.split("(")[0] for a in ceilings[tool]["allow"]} & {"net_in", "clock", "conn_accept", "conn_read", "conn_write", "poll"}, tool)

    def test_the_ceiling_check_fails_for_a_capability_added_to_the_derived_row(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import manifest
        ceilings = tomllib.loads((ROOT / "tools.toml").read_text())
        auth = json.loads((ROOT / "manifests" / "ml-serve.authority.json").read_text())
        self.assertEqual(manifest.gate("ml-serve", auth, ceilings), [])
        for extra in ("file_write", "dir_write", "conc", "net_out", "ffi", "exec", "env", "io_read"):
            mutated = json.loads(json.dumps(auth))
            mutated["labels"].append({"name": extra, "argument": None, "bounded": True})
            self.assertNotEqual(manifest.gate("ml-serve", mutated, ceilings), [], extra)
        mutated = json.loads(json.dumps(auth))
        mutated["labels"].append({"name": "fs_write", "argument": "", "bounded": True})
        self.assertNotEqual(manifest.gate("ml-serve", mutated, ceilings), [])
        # net_in and clock are the server's: another tool's ceiling refuses them
        for tool in ("ml", "ml-predict"):
            a = json.loads((ROOT / "manifests" / (tool + ".authority.json")).read_text())
            for extra in ("net_in", "clock"):
                mutated = json.loads(json.dumps(a))
                mutated["labels"].append({"name": extra, "argument": None, "bounded": True})
                self.assertNotEqual(manifest.gate(tool, mutated, ceilings), [], (tool, extra))

    def test_a_label_of_the_server_without_a_reason_fails_the_check(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import manifest
        text = (ROOT / "tools.toml").read_text().replace('poll = "', 'pollX = "')
        ceilings = tomllib.loads(text)
        auth = json.loads((ROOT / "manifests" / "ml-serve.authority.json").read_text())
        self.assertTrue(any("no reason" in p for p in manifest.gate("ml-serve", auth, ceilings)))
        # and net_in or clock named without their reason is refused as forbidden
        text = (ROOT / "tools.toml").read_text().replace('net_in = "', 'net_inX = "')
        ceilings = tomllib.loads(text)
        self.assertTrue(manifest.gate("ml-serve", auth, ceilings))


if __name__ == "__main__":
    unittest.main()


def mcp_introspect():
    return json.loads(subprocess.run([MCP_BIN, "introspect"], capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout)


class AuthorityMcp(unittest.TestCase):
    """F (docs/mcp.md section 7): the row of `ml-mcp`, derived by the compiler, committed, embedded, within its ceiling; and what it does not hold."""

    def test_the_binary_prints_the_committed_authority(self):
        committed = json.loads((ROOT / "manifests" / "ml-mcp.authority.json").read_text())
        self.assertEqual(mcp_introspect()["authority"], committed)

    def test_the_row_is_the_row_of_ml_and_holds_no_network_clock_exec_or_foreign_code(self):
        a = mcp_introspect()["authority"]
        self.assertTrue(a["bounded"])
        self.assertEqual((a["unbounded_by"], a["foreign_symbols"]), ([], []))
        labels = {label_text(l) for l in a["labels"]}
        self.assertEqual(labels, EXPECTED_MCP)
        self.assertEqual(labels & FORBIDDEN, set())
        self.assertEqual(labels & {"net_in", "net_out", "conn_accept", "conn_read", "conn_write", "poll", "clock", "exec", "ffi", "env"}, set())

    def test_the_ceiling_is_exact_and_every_label_beyond_the_base_row_has_a_reason(self):
        ceilings = tomllib.loads((ROOT / "tools.toml").read_text())
        self.assertEqual(set(ceilings["ml-mcp"]["allow"]), EXPECTED_MCP)
        base = set(tomllib.loads((ROOT / "lib" / "models" / "contract.toml").read_text())["authority"]["base"])
        for label in set(ceilings["ml-mcp"]["allow"]) - base:
            self.assertTrue(ceilings["ml-mcp"]["reasons"].get(label.split("(")[0]), label)

    def test_the_ceiling_check_fails_for_a_capability_added_to_the_derived_row(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import manifest
        ceilings = tomllib.loads((ROOT / "tools.toml").read_text())
        auth = json.loads((ROOT / "manifests" / "ml-mcp.authority.json").read_text())
        self.assertEqual(manifest.gate("ml-mcp", auth, ceilings), [])
        for extra in ("net_in", "net_out", "clock", "ffi", "exec", "env", "poll", "conn_accept"):
            mutated = json.loads(json.dumps(auth))
            mutated["labels"].append({"name": extra, "argument": None, "bounded": True})
            self.assertNotEqual(manifest.gate("ml-mcp", mutated, ceilings), [], extra)
        mutated = json.loads(json.dumps(auth))
        mutated["labels"].append({"name": "fs_write", "argument": "", "bounded": True})
        self.assertNotEqual(manifest.gate("ml-mcp", mutated, ceilings), [])

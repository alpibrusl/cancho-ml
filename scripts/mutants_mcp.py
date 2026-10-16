#!/usr/bin/env python3
"""Mutants for the gates of docs/mcp.md (gate E): a gate that cannot fail judges nothing.

As scripts/mutants_serve.py: each mutant is a copy of the tree with one change, `ml-mcp` rebuilt from the copy (the other binaries are carried as they are), and the tests of the
gate it targets run against it. KILLED means a test failed (or the authority check did); SURVIVED means every test passed, and the mutant must then carry a reason (`why`).
The copy carries every built executable (`build/ml`, `build/ml-predict`, `build/ml-serve`, the check programs), because the tests compare the server with the command line.

    python3 scripts/mutants_mcp.py              # all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_mcp.py --only ID    # one
    python3 scripts/mutants_mcp.py --list
    python3 scripts/mutants_mcp.py --check      # every `old` text is found exactly where it is expected (no build, no test)
    python3 scripts/mutants_mcp.py --jobs N     # N mutants at a time (default 1)

Run it on a frozen copy of the tree when `cancho install` may run in the original (it rewrites build/deps).
"""

import concurrent.futures
import glob
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_cli as base  # noqa: E402

ROOT = base.ROOT
M = base.M

MAIN = "tools/ml-mcp/main.cho"
CALL = "tools/ml-mcp/call.cho"
RPC = "tools/ml-mcp/rpc.cho"
RECS = "tools/ml-mcp/recs.cho"
MSPEC = "tools/ml-mcp/mspec.cho"
TOOLS = "tools/ml-mcp/tools.cho"
GEN = "scripts/mcp_tools.py"

T = ["test_mcp.Tools"]
H = ["test_mcp_hostile"]
S = ["test_mcp_schema"]
CASES = ["test_mcp.CaseGate"]

MUTANTS = [
    # ------------------------------------------------------------------ the framing
    M("f-limit-is-one-byte-more", "D", "a line may hold cap + 1 bytes", [(MAIN, "            } else if n >= cap {\n                over = true;\n            } else {\n                data = buffer.push(heap, data, byte_of(c));",
                                                                         "            } else if n > cap {\n                over = true;\n            } else {\n                data = buffer.push(heap, data, byte_of(c));", 1)], H),
    M("f-limit-is-one-byte-less", "D", "a line may hold cap - 1 bytes", [(MAIN, "            } else if n >= cap {\n                over = true;\n            } else {\n                data = buffer.push(heap, data, byte_of(c));",
                                                                           "            } else if n >= cap - 1 {\n                over = true;\n            } else {\n                data = buffer.push(heap, data, byte_of(c));", 1)], H),
    M("f-long-line-is-kept", "D", "a line over the limit is read and answered like any other", [(MAIN, "        } else if status == 1 {\n            buffer.drop(heap, response);", "        } else if status == 9 {\n            buffer.drop(heap, response);", 1)], H),
    M("f-limit-ceiling-raised", "D", "--max-message-bytes may be 16777217", [(MAIN, "if cap < 1 || cap > 16777216 {", "if cap < 1 || cap > 16777217 {", 1)], H),
    M("f-limit-zero-allowed", "D", "--max-message-bytes may be 0", [(MAIN, "if cap < 1 || cap > 16777216 {", "if cap < 0 || cap > 16777216 {", 1)], H),
    M("f-blank-line-answered", "D", "a line of white space is a malformed message", [(MAIN, "                if !blank(buffer.bytes(lr)) {", "                if true {", 1)], H),
    M("f-carriage-return-kept-at-the-end", "D", "the carriage return before the line feed is part of the line (a strict parse would refuse it only in a string)", [(MAIN, "            if c == 13 {\n                held_cr = true;", "            if c == 7000 {\n                held_cr = true;", 1)], H),
    M("f-eof-in-a-line-is-dropped", "D", "the end of the input in the middle of a line leaves the line unanswered", [(MAIN, "            } else {\n                status = 3;\n            }", "            } else {\n                status = 2;\n            }", 1)], H),
    M("f-answer-not-flushed", "D", "the answer is written and not flushed", [(MAIN, "                out.flushed(io);", "                0;", 1)], H),
    # ------------------------------------------------------------------ the id and the envelope
    M("i-integer-id-rewritten", "A", "the id of a response is not the token of the request", [(RPC, "        return buffer.append(heap, b, line[tape[3 * node + 1]..tape[3 * node + 2]]);\n    }\n    return buffer.append(heap, b, \"null\");",
                                                       "        return buffer.append(heap, b, \"0\");\n    }\n    return buffer.append(heap, b, \"null\");", 1)], H),
    M("i-string-id-unquoted", "A", "a string id is echoed without its quotes", [(RPC, "        b = buffer.push(heap, b, byte_of('\"'));\n        b = buffer.append(heap, b, line[tape[3 * node + 1]..tape[3 * node + 2]]);\n        return buffer.push(heap, b, byte_of('\"'));",
                                                                            "        b = buffer.append(heap, b, line[tape[3 * node + 1]..tape[3 * node + 2]]);\n        return b;", 1)], H),
    M("i-float-id-accepted", "D", "a float is an id", [(RPC, "json.kind(tape, node) == json.kind_int()", "json.is_number(tape, node)", 1)], H),
    M("i-notification-answered", "D", "a request without an id is answered", [(MAIN, "    } else if idn < 0 {\n        // a notification", "    } else if false {\n        // a notification", 1)], H),
    M("i-null-id-accepted", "D", "an id that is null is an id", [(MAIN, "    } else if null_id {", "    } else if false {", 1)], H),
    M("i-jsonrpc-version-unchecked", "D", "jsonrpc is not 2.0 and it goes on", [(MAIN, "    if jv < 0 || !json.is_string(tape, jv) || !json.string_equals(line, tape, jv, \"2.0\") {", "    if false {", 1)], H),
    M("e-isError-always-false", "A", "a refusal is not an error", [(CALL, "response = answered(heap, id, buffer.bytes(dr), buffer.bytes(dr), !ok, exit);", "response = answered(heap, id, buffer.bytes(dr), buffer.bytes(dr), false, exit);", 1)], T),
    M("e-exit-code-zero", "A", "_meta ml/exit is 0 for a refusal", [(CALL, "response = answered(heap, id, buffer.bytes(dr), buffer.bytes(dr), !ok, exit);", "response = answered(heap, id, buffer.bytes(dr), buffer.bytes(dr), !ok, 0);", 1)], T),
    M("e-text-is-the-document", "A", "content[0].text is the document and not the text form of a success", [(CALL, "                        if ok && len(buffer.bytes(t)) > 0 {", "                        if false {", 1)], CASES),
    M("e-structured-content-is-the-data", "A", "structuredContent is the data member and not the document", [(CALL, "let doc = out.document(heap, \"ml\", \"ml.v1\", app.version(), buffer.bytes(d), buffer.bytes(x), er);", "let doc = buffer.append(heap, buffer.empty(heap, 16), buffer.bytes(d));", 1)], CASES),
    # ------------------------------------------------------------------ the lifecycle and the methods
    M("l-no-initialize-needed", "D", "a call before initialize is served", [(MAIN, "            } else if !init {", "            } else if false {", 1)], H),
    M("l-unknown-method-is-empty", "D", "an unknown method gets an empty result", [(MAIN, "                response = rpc.error_response(heap, idb, rpc.method_not_found(), \"the server has tools/list, tools/call, ping and initialize\", \"/method\");",
                                                    "                response = rpc.result_response(heap, idb, \"{}\");", 1)], H),
    M("l-batch-is-served-as-an-object", "D", "a batch is not refused as a batch", [(MAIN, "        } else if json.is_array(contents(tp), 0) {", "        } else if false {", 1)], H),
    M("l-other-version-echoed", "D", "initialize answers the revision the client asked for", [(MAIN, "    w = json.put_string(heap, w, tools.revision());", "    w = json.put_string(heap, w, \"1900-01-01\");", 1)], T + H),
    M("l-cursor-ignored", "D", "tools/list takes any cursor", [(MAIN, "                    if !json.is_object(tape, params) || (c >= 0 && !(json.is_string(tape, c) && json.string_length(line, tape, c) == 0)) {", "                    if false {", 1)], H),
    # ------------------------------------------------------------------ the arguments
    M("a-duplicate-key-accepted", "D", "a key given twice is accepted", [(CALL, "            } else if json.get(line, tape, args, name) != v {\n                status = 3;", "            } else if false {\n                status = 3;", 1)], H),
    M("a-nul-accepted", "D", "a NUL in a string argument is accepted", [(CALL, "                        if has_nul(buffer.bytes(tr)) {", "                        if false {", 1)], H),
    M("a-nat-negative-accepted", "D", "a negative nat is a nat", [(CALL, "|| int_of(line[tape[3 * v + 1]]) == '-' {", "|| false {", 1)], H),
    M("a-float-nat-accepted", "D", "1.5 is a nat", [(CALL, "if json.kind(tape, v) != json.kind_int() || int_of(line[tape[3 * v + 1]]) == '-' {", "if int_of(line[tape[3 * v + 1]]) == '-' {", 1)], H),
    M("a-unknown-tool-is-a-tool-result", "D", "an unknown tool name is not refused", [(CALL, "    if len(row) == 0 {\n        var response = buffer.empty(heap, 1);", "    if len(row) == 1000 {\n        var response = buffer.empty(heap, 1);", 1)], H),
    M("a-both-file-and-records", "D", "predict takes a file and records together", [(CALL, "        if has_file == (recs_node >= 0) {", "        if false {", 1)], T),
    # ------------------------------------------------------------------ read-only
    M("r-read-only-switch-removed", "C", "a write tool is called without --allow-write", [(CALL, "    if !allow_write {\n        if bytes.equal(bytes.field(row, '|', 7), \"1\") {", "    if false {\n        if bytes.equal(bytes.field(row, '|', 7), \"1\") {", 1)], T),
    M("r-write-flags-not-checked", "C", "a write flag of eval is accepted without --allow-write", [(CALL, "                if v >= 0 && json.kind(tape, v) != json.kind_false() {\n                    return read_only(heap, id, \"flag\", bytes.field(ws, ',', k));",
                                                                    "                if false {\n                    return read_only(heap, id, \"flag\", bytes.field(ws, ',', k));", 1)], T),
    M("r-list-is-always-the-full-one", "C", "the read-only list offers fit", [(MAIN, "                } else if allow_write {\n                    buffer.drop(heap, response);\n                    response = rpc.result_response(heap, idb, tools.list_write());", "                } else if true {\n                    buffer.drop(heap, response);\n                    response = rpc.result_response(heap, idb, tools.list_write());", 1)], T),
    M("r-write-needs-no-root", "C", "--allow-write without --root starts", [(MAIN, "if argv.has(src, \"allow-write\") && !argv.has(src, \"root\") {", "if false {", 1)], T),
    M("r-refusal-is-not-exit-4", "C", "mcp.read-only has the exit code 8", [(MSPEC, "mcp.read-only|4|never|", "mcp.read-only|8|never|", 1)], T + S),
    # ------------------------------------------------------------------ the root
    M("t-root-not-passed", "C", "the root of the server is not part of the command line of a call", [(CALL, "        if bytes.equal(name, \"root\") {\n            if len(root) > 0 {", "        if bytes.equal(name, \"root\") {\n            if len(root) > 1000000 {", 1)], T),
    M("t-root-checked-twice-skipped", "C", "the double-dot check of the root is skipped in a call", [(CALL, "        if paths.has_double_dot(buffer.bytes(rb)) {\n            e1 = paths.refuse_double_dot(heap, e1, buffer.bytes(rb), argv.value_index(src, \"root\"));\n        }\n    }\n    var response",
                                                                  "        if false {\n            e1 = paths.refuse_double_dot(heap, e1, buffer.bytes(rb), argv.value_index(src, \"root\"));\n        }\n    }\n    var response", 1)], T),
    # ------------------------------------------------------------------ the records
    M("c-header-dropped", "A", "the CSV of the records has no header", [(RECS, "csv = buffer.append(heap, csv, cols.header(csr));", "csv = csv;", 1)], T),
    M("c-scale-forgotten", "A", "a dec column is written with scale 0", [(RECS, "type_code(desc, tape, c), scale);", "type_code(desc, tape, c), 0);", 1)], T),
    M("c-any-key-is-a-column", "A", "a key that is not a column is accepted", [(RECS, "                if !known_key(cs, line, tape, k) {", "                if false {", 1)], T),
    M("c-key-twice-accepted", "A", "a column twice in a record is accepted", [(RECS, "            if status == done() && json.count(tape, j) != present_keys(cs, line, tape, j) {", "            if false {", 1)], T),
    M("c-inexact-number-accepted", "A", "an inexact number is written anyway", [(RECS, "                if st == cols.inexact() {\n                    status = 2;", "                if false {\n                    status = 2;", 1)], T),
    M("c-pointer-without-the-row", "A", "the pointer of a fault names no row", [(RECS, "        pointer = buffer.push_nat(heap, pointer, bad_row);", "        pointer = buffer.push_nat(heap, pointer, 0);", 1)], T),
    # ------------------------------------------------------------------ the tools, generated
    M("g-a-tool-is-missing-from-the-list", "B", "tools/list lacks runs_verify", [(TOOLS, '{\\"name\\":\\"runs_verify\\"', '{\\"name\\":\\"runs_verifx\\"', 1)], T),
    M("g-the-call-table-loses-a-flag", "B", "the call table of describe lacks limit", [(TOOLS, "describe|describe||||model-file~path~path-read;limit~nat~none;max-bytes~nat~none|0", "describe|describe||||model-file~path~path-read;max-bytes~nat~none|0", 1)], T + S),
    M("g-a-ceiling-is-wrong-in-the-generator", "B", "the generator writes the ceiling plus one", [(GEN, '            s["maximum"] = limits[name][1]', '            s["maximum"] = limits[name][1] + 1', 1)], [], script="check_mcp_tools.py --introspect"),
    M("g-a-kind-is-wrong-in-the-generator", "B", "the generator writes nat as number", [(GEN, '        s["type"] = "integer"', '        s["type"] = "number"', 1)], [], script="check_mcp_tools.py"),
    M("g-the-table-is-not-the-source", "B", "the generator reads the usage of another verb for required flags", [(GEN, "            req = required_flags(line)", "            req = []", 1)], [], script="check_mcp_tools.py"),
    # ------------------------------------------------------------------ authority
    M("x-the-clock-is-read", "F", "ml-mcp reads the clock", [(MAIN, "    release(clock);\n    var status = 0;", "    var tick = 0;\n    borrow clock as &kk in {\n        tick = clock_ms(kk);\n    }\n    release(clock);\n    var status = 0;", 1)], ["test_authority"], kind="authority"),
    M("x-the-ceiling-is-wider-than-the-row", "F", "the ceiling of ml-mcp lacks conc", [("tools.toml", '[ml-mcp]\nschema = "ml"\nallow = ["args", "conc", ', '[ml-mcp]\nschema = "ml"\nallow = ["args", ', 1)], ["test_authority"], kind="authority"),
    M("x-a-network-label-in-the-ceiling", "F", "the ceiling of ml-mcp names net_in", [("tools.toml", '[ml-mcp]\nschema = "ml"\nallow = ["args", "conc", ', '[ml-mcp]\nschema = "ml"\nallow = ["args", "net_in(\\"\\")", "conc", ', 1)], ["test_authority"], kind="authority"),
]

# an edit is (file, old, new) or (file, old, new, nth): the occurrence is the first by default
for _m in MUTANTS:
    _m["edits"] = [e if len(e) == 4 else (*e, 1) for e in _m["edits"]]


def libs(g):
    return (g("lib/core/*.cho") + g("lib/linalg/*.cho") + g("lib/metrics/*.cho") + g("lib/models/*.cho") + g("lib/models/kmeans/*.cho") + g("lib/models/linear/*.cho") + g("lib/models/logistic/*.cho") +
            g("lib/models/tree/*.cho") + g("lib/models/treewire/*.cho") + g("lib/transforms/*.cho") + g("lib/pipeline/*.cho") + g("build/deps/*.cho"))


def build_mcp(tree):
    g = lambda pat: sorted(glob.glob(str(tree / pat)))
    shared = [f for f in g("tools/ml/*.cho") if not f.endswith("/main.cho")]
    files = libs(g) + [str(tree / "generated/ml-mcp/built.cho")] + shared + [str(tree / "tools/ml-serve/cols.cho"), str(tree / "tools/ml-serve/num.cho")] + g("tools/ml-mcp/*.cho")
    target = tree / "build" / "ml-mcp"
    p = subprocess.run([base.compiler(), "build", "--std", *files, "-o", str(target)], capture_output=True, text=True, cwd=tree)
    return (target if p.returncode == 0 else None), p.stdout + p.stderr


def copy_tree(dest):
    base.copy_tree(dest)
    # the command line, which the tests compare the server with
    shutil.copy2(ROOT / "build" / "ml", dest / "build" / "ml")
    # the copy carries the venv of the original through the PATH of the caller; nothing to copy


def env_for(tree):
    return dict(os.environ, ML_BIN=str(tree / "build" / "ml"), ML_PREDICT_BIN=str(tree / "build" / "ml-predict"), ML_SERVE_BIN=str(tree / "build" / "ml-serve"), ML_MCP_BIN=str(tree / "build" / "ml-mcp"),
                PYTHONDONTWRITEBYTECODE="1", CANCHO=base.compiler(), ST_STRIDE="25")


def run_tests(tree, targets):
    p = subprocess.run([sys.executable, "-W", "ignore", "-m", "unittest", *targets], capture_output=True, text=True, cwd=tree / "tests" / "conformance", env=env_for(tree))
    return p.returncode, p.stdout + p.stderr


def run_script(tree, script):
    p = subprocess.run([sys.executable, str(tree / "scripts" / script.split()[0]), *script.split()[1:]], capture_output=True, text=True, cwd=tree, env=env_for(tree))
    return p.returncode, p.stdout + p.stderr


def run_one(m):
    with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        copy_tree(tree)
        problems = base.apply_edits(tree, m)
        if problems:
            return "INVALID", "; ".join(problems)
        kind = m["opt"].get("kind", "")
        if kind == "authority":
            r = subprocess.run([sys.executable, str(tree / "scripts" / "manifest.py"), "--tree", str(tree)], capture_output=True, text=True, cwd=tree)
            if r.returncode != 0:
                return "KILLED", "manifest.py --tree: " + " | ".join(l for l in r.stdout.splitlines() if l.startswith("FAIL"))[:300]
            return "SURVIVED", ""
        script = m["opt"].get("script")
        if script:
            code, out = run_script(tree, script)
            return ("KILLED", script + ": " + (out.strip().splitlines() or [""])[0][:200]) if code != 0 else ("SURVIVED", "")
        built, log = build_mcp(tree)
        if built is None:
            return "INVALID", "build failed: " + log[:600]
        code, out = run_tests(tree, m["tests"])
        if code != 0:
            names = base.failing_tests(out)
            return "KILLED", "tests: " + ", ".join(names[:4]) + ("" if names else out.strip().splitlines()[-1][:200])
        return "SURVIVED", ""


def baseline():
    with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        copy_tree(tree)
        built, log = build_mcp(tree)
        if built is None:
            return "INVALID", "build failed: " + log[:600]
        code, out = run_tests(tree, ["test_mcp", "test_mcp_schema", "test_mcp_hostile", "test_authority"])
        return ("PASSED", "") if code == 0 else ("FAILED", out[-800:])


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        for m in MUTANTS:
            print("%-44s %s  %s" % (m["id"], m["gate"], m["what"]))
        return 0
    if "--check" in argv:
        bad = 0
        with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
            tree = pathlib.Path(td) / "tree"
            shutil.copytree(ROOT, tree, ignore=shutil.ignore_patterns(".git", "build", "__pycache__", "*.pyc"), symlinks=True)
            for m in MUTANTS:
                problems = base.apply_edits(tree, m)
                if problems:
                    print("INVALID %s: %s" % (m["id"], "; ".join(problems)))
                    bad += 1
                for item in m["edits"]:
                    shutil.copy(ROOT / item[0], tree / item[0])
        print("%d mutants, %d with an edit that does not apply" % (len(MUTANTS), bad))
        return 1 if bad else 0
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 1
    todo = [m for m in MUTANTS if not only or m["id"] == only]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run_one, m): m for m in todo}
        for f in concurrent.futures.as_completed(futures):
            m = futures[f]
            verdict, detail = f.result()
            results.append((m, verdict, detail))
            print("%-9s %-44s %s  %s" % (verdict, m["id"], m["gate"], detail), flush=True)
    if not only:
        v, d = baseline()
        print("%-9s %-44s %s  %s" % (v, "baseline (the unchanged tree)", "-", d), flush=True)
    survivors = [(m, d) for m, v, d in results if v == "SURVIVED"]
    invalid = [m["id"] for m, v, d in results if v == "INVALID"]
    killed = sum(1 for m, v, d in results if v == "KILLED")
    print("\n%d mutants: %d killed, %d survived, %d invalid" % (len(results), killed, len(survivors), len(invalid)))
    for m, d in survivors:
        print("SURVIVOR %s: %s" % (m["id"], m["opt"].get("why", "NO REASON GIVEN")))
    unexplained = [m["id"] for m, d in survivors if not m["opt"].get("why")]
    return 1 if (unexplained or invalid) else 0


if __name__ == "__main__":
    sys.exit(main())

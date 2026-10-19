#!/usr/bin/env python3
"""Mutants for the gates of docs/jsonl.md section 6 (G1 to G4): a gate that cannot fail judges nothing.

Each mutant is a copy of the tree (scripts/mutants_cli.py's copy, which carries every built executable) with one edit to tools/ml/jsonl.cho, tools/ml/jsonlcli.cho or
tools/ml/app.cho. The gate named in the mutant runs against a binary rebuilt from the copy: "oracle" rebuilds jsonl-check and runs tests/conformance/test_jsonl_oracle.py (the
cell text against Python's json); "cli" rebuilds ml and runs test_jsonl_equiv.py and test_jsonl_rules.py (the model bytes against the CSV's, the refusals); "cv" rebuilds ml and runs test_jsonl_cv.py (cv, the identity of the
input, the run record and its replay). KILLED: a test
failed; SURVIVED: every test passed, and the mutant then carries a reason.

    python3 scripts/mutants_jsonl.py              # all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_jsonl.py --only ID
    python3 scripts/mutants_jsonl.py --list
    python3 scripts/mutants_jsonl.py --check      # every edit applies where it is expected (no build, no test)
"""
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mutants_cli as mc  # noqa: E402

ROOT = mc.ROOT
J = "tools/ml/jsonl.cho"
C = "tools/ml/jsonlcli.cho"


def M(id, gate, what, edits, **opt):
    return dict(id=id, gate=gate, what=what, edits=edits, opt=opt)


MUTANTS = [
    # ------------------------------------------------------------------ the mapping of values to cells
    M("map-null-is-a-word", "oracle", "null is written as the word true", [(J, "            if jk == 0 {\n                empty = true;", "            if jk == 0 {\n                which = 3;", 1)]),
    M("map-true-is-false", "oracle", "true is written as false", [(J, "            } else if jk == 2 {\n                kind = 2;\n                which = 3;", "            } else if jk == 2 {\n                kind = 2;\n                which = 2;", 1)]),
    M("map-number-loses-a-digit", "oracle", "the text of a number loses its last byte", [(J, "                a = tape[3 * nd + 1];\n                b = tape[3 * nd + 2];\n            } else {", "                a = tape[3 * nd + 1];\n                b = tape[3 * nd + 2] - 1;\n            } else {", 1)]),
    M("map-string-not-decoded", "oracle", "a string with an escape is written undecoded", [(J, "                if json.string_plain(tape, nd) {", "                if true {", 1)]),
    M("map-key-not-decoded", "oracle", "a key with an escape is looked up undecoded", [(J, "        if json.string_plain(tape, k) {", "        if true {", 1)]),
    M("map-object-array-swapped", "oracle", "the kind of a nested value is swapped in the refusal", [(J, "7 - jk", "jk - 6", 1)]),
    M("map-root-kind", "oracle", "a root string is reported as an array", [(J, "    if k == 5 {\n        return 1;", "    if k == 5 {\n        return 0;", 1)]),
    M("map-field-order", "oracle", "the cells are written in the reverse of the declaration", [(J, "        let nd = nodes[j];", "        let nd = nodes[count - 1 - j];", 1)]),
    M("map-header-without-quote", "oracle", "a header name is never quoted", [(J, "        out = put_cell(heap, out, name_at(names, ends, h), false);", "        out = buffer.append(heap, out, name_at(names, ends, h));", 1)]),
    M("map-comma-not-quoted", "oracle", "a cell with a comma is not quoted", [(J, "        if c == 44 || c == 34 || c == 13 || c == 10 {", "        if c == 34 || c == 13 || c == 10 {", 1)]),
    M("map-lf-not-quoted", "oracle", "a cell with an LF is not quoted", [(J, "        if c == 44 || c == 34 || c == 13 || c == 10 {", "        if c == 44 || c == 34 || c == 13 {", 1)]),
    M("map-quote-not-doubled", "oracle", "a quote inside a quoted cell is not doubled", [(J, "        if int_of(cell[i]) == 34 {\n            o = buffer.push(heap, o, byte_of(34));\n        }", "", 1)]),
    M("map-single-column-blank", "oracle", "an empty cell of a table of one column is a blank line", [(J, "            if count == 1 {\n                o = buffer.append(heap, o, \"\\\"\\\"\");", "            if false {\n                o = buffer.append(heap, o, \"\\\"\\\"\");", 1)]),
    # ------------------------------------------------------------------ absent and null
    M("absent-always-refused", "oracle", "--absent empty is ignored", [(J, "            if !absent_empty {\n                return (o, fail_at(4", "            if true {\n                return (o, fail_at(4", 1)]),
    M("absent-never-refused", "oracle", "a missing key is always an empty cell", [(J, "            if !absent_empty {\n                return (o, fail_at(4", "            if false {\n                return (o, fail_at(4", 1)]),
    M("null-refused", "oracle", "null is refused as a missing key", [(J, "            if jk == 0 {\n                empty = true;", "            if jk == 0 {\n                return (o, fail_at(4, lineno, row, j, 0, 0, 0));", 1)]),
    # ------------------------------------------------------------------ the protection of numbers and text
    M("protect-string-in-number-column", "oracle", "a string is accepted in a numeric column", [(J, "                if mask[j] == 0 && kind != 0 {", "                if false && mask[j] == 0 && kind != 0 {", 1)]),
    M("protect-number-in-text-column", "oracle", "a number is refused in a text column", [(J, "                if mask[j] == 0 && kind != 0 {", "                if mask[j] == 0 || kind != 0 {", 1)]),
    M("protect-mixed-kinds-allowed", "oracle", "a text column may mix numbers and strings", [(J, "                if kinds[2 * j] >= 0 && kinds[2 * j] != kind {", "                if false && kinds[2 * j] >= 0 && kinds[2 * j] != kind {", 1)]),
    M("protect-first-kind-not-kept", "oracle", "the kind of the first value is not remembered", [(J, "                if kinds[2 * j] < 0 {\n                    kinds[2 * j] = kind;", "                if false {\n                    kinds[2 * j] = kind;", 1)]),
    M("protect-first-line-not-kept", "oracle", "the line of the first value is not remembered", [(J, "                    kinds[2 * j + 1] = lineno;", "                    kinds[2 * j + 1] = 0;", 1)]),
    M("dup-allowed", "oracle", "a repeated declared key is not refused (the last wins)", [(J, "            if nodes[f] >= 0 {\n                return", "            if false && nodes[f] >= 0 {\n                return", 1)]),
    # ------------------------------------------------------------------ the limits
    M("limit-line-off-by-one", "oracle", "a line of exactly --max-line-bytes is refused", [(J, "if len(line) > line_cap {", "if len(line) >= line_cap {", 1)]),
    M("limit-line-removed", "oracle", "the line bound is not applied", [(J, "if len(line) > line_cap {", "if false && len(line) > line_cap {", 1)]),
    M("limit-nodes-one-more", "oracle", "the tape holds one node more than --max-json-nodes", [(J, "var tape = box_slice(heap, 3 * max_nodes, 0);", "var tape = box_slice(heap, 3 * max_nodes + 3, 0);", 1)]),
    M("limit-csv-off-by-one", "oracle", "a CSV of exactly --max-bytes is refused", [(J, "} else if used > out_cap {", "} else if used >= out_cap {", 1)]),
    M("limit-csv-removed", "oracle", "the generated CSV is not bounded", [(J, "} else if used > out_cap {", "} else if false {", 1)]),
    M("limit-depth-mapped-to-nodes", "oracle", "every parser error is the node limit", [(J, "        return (out, fail_at(1, lineno, row, 0, json.error_position(parsed), json.error_code(parsed), 0));", "        return (out, fail_at(3, lineno, row, 0, len(tape) / 3, 0, 0));", 1)]),
    # ------------------------------------------------------------------ lines, blanks, the byte order mark
    M("input-bom-kept", "oracle", "a byte order mark is not dropped", [(J, "        start = 3;", "        start = 0;", 1)]),
    M("input-blank-line-dropped", "oracle", "a blank line is not written", [(J, "                            } else if is_blank(line) {\n                                out = buffer.push(heap, out, byte_of(10));", "                            } else if is_blank(line) {\n                                lineno = lineno + 0;", 1)]),
    M("input-cr-not-blank", "oracle", "a line of only a CR is a record", [(J, "        if c != 32 && c != 9 && c != 13 {", "        if c != 32 && c != 9 {", 1)]),
    M("input-row-counts-blank", "oracle", "the row number counts blank lines", [(J, "                            } else if is_blank(line) {\n", "                            } else if is_blank(line) {\n                                row = row + 1;\n", 1)]),
    # ------------------------------------------------------------------ the command line
    M("cli-stdin-ignored", "cli", "standard input is not the input of a JSON lines run", [(C, "            if paths.is_stdin(ob.file) {", "            if false && paths.is_stdin(ob.file) {", 1)]),
    M("cli-fit-stdin-not-wanted", "cli", "fit - does not read standard input", [("tools/ml/app.cho", "    if verb == 1 || verb == 3 {\n        return argv.operand_count", "    if verb == 3 {\n        return argv.operand_count", 1)]),
    M("cli-eval-stdin-not-wanted", "cli", "eval - does not read standard input", [("tools/ml/app.cho", "    if verb == 1 || verb == 3 {\n        return argv.operand_count", "    if verb == 1 {\n        return argv.operand_count", 1)]),
    M("cli-ndjson-not-an-alias", "cli", "ndjson is not jsonl", [(C, "    return bytes.equal(f, \"jsonl\") || bytes.equal(f, \"ndjson\");", "    return bytes.equal(f, \"jsonl\");", 1)]),
    M("cli-cat-not-text", "cli", "--cat names no text column of the JSON lines", [(C, "texts = buffer.append(heap, texts, argv.text(src, \"cat\"));", "texts = buffer.append(heap, texts, \"\");", 1)]),
    M("cli-predict-reads-the-target", "cli", "predict requires the target key as eval does", [(C, "model_fields(heap, fs, root, ob.model, max_bytes, verb == 3)", "model_fields(heap, fs, root, ob.model, max_bytes, true)", 1)]),
    M("cli-eval-skips-the-target", "cli", "eval does not read the target key", [(C, "model_fields(heap, fs, root, ob.model, max_bytes, verb == 3)", "model_fields(heap, fs, root, ob.model, max_bytes, false)", 1)]),
    M("cli-fields-not-required", "cli", "fit of JSON lines without --fields is not refused", [(C, "        if !argv.has(src, \"fields\") {\n            if verb == 5 {", "        if false {\n            if verb == 5 {", 1)]),
    M("cli-conflicts-not-refused", "cli", "--fields with CSV is not refused", [(C, "        if argv.has(src, \"fields\") {\n            e = app.flag_problem(heap, e, \"args.conflict\"", "        if false {\n            e = app.flag_problem(heap, e, \"args.conflict\"", 1)]),
    M("cli-retry-never-offered", "cli", "the missing-field refusal offers no repair", [(C, "    if f.code == 4 && retry_ok {", "    if false && f.code == 4 && retry_ok {", 1)]),
    M("cli-absent-ignored", "cli", "--absent empty is not passed to the reader", [(C, "    let absent_empty = bytes.equal(argv.text(src, \"absent\"), \"empty\");", "    let absent_empty = false;", 1)]),
    M("cli-input-bound-ignored", "cli", "a file of JSON lines is read without the --max-bytes bound", [(C, "paths.full(ob.file), max_bytes);", "paths.full(ob.file), 268435456);", 1)]),
    M("cli-line-cap-ignored", "cli", "--max-line-bytes is not passed to the reader", [(C, "absent_empty, max_nodes, line_cap, max_bytes);", "absent_empty, max_nodes, 16777216, max_bytes);", 1)]),
    M("cli-nodes-ignored", "cli", "--max-json-nodes is not passed to the reader", [(C, "absent_empty, max_nodes, line_cap, max_bytes);", "absent_empty, 4096, line_cap, max_bytes);", 1)]),
    M("cli-shown-name-lost", "cli", "the converted input is shown under another name", [(C, "nf = paths.of_mem(heap, buffer.bytes(cr), buffer.bytes(sr2));", "nf = paths.of_mem(heap, buffer.bytes(cr), \"x\");", 1)],
      why="equivalent: the name of the converted input is read by no answer. The refusals of the CSV reader carry a line and a column and no path, and a read of bytes in memory cannot fail with an errno; the refusals of the JSON lines reader take the name from the original Rp"),
    # ------------------------------------------------------------------ ml cv, the identity of the input, the run record and its replay (tests/conformance/test_jsonl_cv.py)
    M("cv-fields-not-read", "cv", "cv does not declare the columns from --fields", [(C, "    if verb == 1 || verb == 5 {\n        let list", "    if verb == 1 {\n        let list", 1)]),
    M("cv-fields-not-required", "cv", "cv of JSON lines without --fields is not refused", [(C, "    if verb == 1 || verb == 5 {\n        if !argv.has(src, \"fields\") {", "    if verb == 1 {\n        if !argv.has(src, \"fields\") {", 1)]),
    M("cv-drop-is-not-text", "cv", "a column named by --drop is read as a number", [(C, "texts = buffer.append(heap, texts, argv.text(src, \"drop\"));", "texts = buffer.append(heap, texts, \"\");", 1)]),
    M("cv-split-column-read-by-name", "cv", "--split-by and --split-group read the column from the file by its name", [("tools/ml/splitwire.cho", "fsio.read_input(heap, fs, root, file, o.max_bytes)", "fsio.read_whole(heap, fs, root, paths.shown(file), paths.full(file), o.max_bytes)", 1)]),
    M("id-raw-digest-lost", "cv", "the digest of the bytes the user gave is not kept: the CSV made of them is hashed", [(C, "nf = paths.with_raw(heap, nf, buffer.bytes(gr), buffer.size(dr2));", "nf = nf;", 1)]),
    M("id-stdin-not-hashed", "cv", "standard input is looked up as a file named - to be hashed", [("tools/ml/fsio.cho", "    if paths.is_stdin(file) {\n        return (sha_bytes", "    if false && paths.is_stdin(file) {\n        return (sha_bytes", 1)]),
    M("record-without-format", "cv", "a run record of JSON lines holds no format", [("tools/ml/track.cho", "    if len(jsonl_fields) > 0 {", "    if false && len(jsonl_fields) > 0 {", 1)]),
    M("record-csv-gains-format", "cv", "a run record of a CSV holds the format members", [("tools/ml/track.cho", "    if len(jsonl_fields) > 0 {", "    if len(jsonl_fields) >= 0 {", 1)]),
    M("replay-reads-csv", "cv", "runs verify reads the recorded JSON lines file as a CSV", [("tools/ml/runs.cho", "if go && is_jsonl {", "if go && false {", 1)]),
    M("replay-absent-rule-lost", "cv", "runs verify forgets the --absent rule of the record", [("tools/ml/runs.cho", "abs_empty = bytes.equal(buffer.bytes(dar0), \"empty\");", "abs_empty = false;", 1)]),
    M("replay-drop-is-not-text", "cv", "runs verify reads the dropped columns of the record as numbers", [("tools/ml/runs.cho", "texts = buffer.append(heap, texts, buffer.bytes(drr));", "texts = buffer.append(heap, texts, \"\");", 1)]),
    M("replay-stdin-record-refitted", "cv", "runs verify tries to open the file named - of a record of standard input", [("tools/ml/runs.cho", "if go && bytes.equal(buffer.bytes(dpr), \"-\") {", "if go && false && bytes.equal(buffer.bytes(dpr), \"-\") {", 1)]),
]


def build_check(tree):
    env = dict(os.environ, CANCHO=mc.compiler(), ML_DEPS=str(tree / "build" / "deps"))
    p = subprocess.run(["sh", str(tree / "scripts" / "build_jsonl.sh"), str(tree / "build" / "jsonl-check")], capture_output=True, text=True, cwd=tree, env=env)
    return p.returncode == 0, p.stdout + p.stderr


def run_tests(tree, gate):
    env = dict(os.environ, ML_BIN=str(tree / "build" / "ml"), JSONL_CHECK_BIN=str(tree / "build" / "jsonl-check"), PYTHONDONTWRITEBYTECODE="1", CANCHO=mc.compiler())
    targets = {"oracle": ["test_jsonl_oracle"], "cli": ["test_jsonl_equiv", "test_jsonl_rules"], "cv": ["test_jsonl_cv"]}[gate]
    p = subprocess.run([sys.executable, "-m", "unittest", *targets], capture_output=True, text=True, cwd=tree / "tests" / "conformance", env=env)
    return p.returncode, p.stdout + p.stderr


def run_one(m):
    with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        mc.copy_tree(tree)
        problems = mc.apply_edits(tree, m)
        if problems:
            return "INVALID", "; ".join(problems)
        if m["gate"] == "oracle":
            ok, log = build_check(tree)
        else:
            binary, log = mc.build(tree)
            ok = binary is not None
        if not ok:
            return "INVALID", "build failed: " + log[:400]
        code, out = run_tests(tree, m["gate"])
        if code != 0:
            failing = mc.failing_tests(out)
            return "KILLED", "tests: " + ", ".join(failing[:3]) + ("" if failing else out.strip().splitlines()[-1][:200])
        return "SURVIVED", ""


def baseline():
    with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        mc.copy_tree(tree)
        ok, log = build_check(tree)
        binary, log2 = mc.build(tree)
        if not ok or binary is None:
            return "INVALID", "build failed: " + (log + log2)[:400]
        for gate in ("oracle", "cli", "cv"):
            code, out = run_tests(tree, gate)
            if code != 0:
                return "FAILED", out[-600:]
        return "PASSED", ""


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        for m in MUTANTS:
            print("%-34s %-6s %s" % (m["id"], m["gate"], m["what"]))
        return 0
    if "--check" in argv:
        bad = 0
        with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
            tree = pathlib.Path(td) / "tree"
            mc.copy_tree(tree)
            for m in MUTANTS:
                problems = mc.apply_edits(tree, m)
                if problems:
                    print("INVALID %s: %s" % (m["id"], "; ".join(problems)))
                    bad += 1
                for item in m["edits"]:
                    (tree / item[0]).write_text((ROOT / item[0]).read_text())
        ids = [m["id"] for m in MUTANTS]
        if len(set(ids)) != len(ids):
            print("a mutant id is used twice")
            bad += 1
        print("%d mutants, %d with an edit that does not apply" % (len(MUTANTS), bad))
        return 1 if bad else 0
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    results = []
    for m in MUTANTS:
        if only and m["id"] != only:
            continue
        verdict, detail = run_one(m)
        results.append((m, verdict, detail))
        print("%-9s %-34s %-6s %s" % (verdict, m["id"], m["gate"], detail), flush=True)
    if not only:
        v, d = baseline()
        print("%-9s %-34s %-6s %s" % (v, "baseline (the unchanged tree)", "-", d), flush=True)
    survivors = [(m, d) for m, v, d in results if v == "SURVIVED"]
    invalid = [m["id"] for m, v, d in results if v == "INVALID"]
    killed = sum(1 for m, v, d in results if v == "KILLED")
    print("\n%d mutants: %d killed, %d survived, %d invalid" % (len(results), killed, len(survivors), len(invalid)))
    for m, d in survivors:
        print("SURVIVOR %s: %s" % (m["id"], m["opt"].get("why", "NO REASON GIVEN")))
    unexplained = [m["id"] for m, d in survivors if not m["opt"].get("why")]
    return 1 if (unexplained or invalid) else 0


sys.exit(main())

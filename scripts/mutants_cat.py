#!/usr/bin/env python3
"""Mutants for the gates of docs/categorical.md section 1: a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change (a few have two or three edits) to a source of this repository, the reader-check (and, for the command-line mutants,
`ml`) rebuilt from the copy, and the gates named in the mutant run against it. KILLED: a gate reported differences, a trap or a failed test; SURVIVED: none did, and the
mutant then carries a reason (an equivalent mutant) or is a gap that docs/categorical.md section 9 names.

    python3 scripts/mutants_cat.py              # control, then all mutants
    python3 scripts/mutants_cat.py --only ID    # one
    python3 scripts/mutants_cat.py --list
    python3 scripts/mutants_cat.py --check      # every `old` text is found exactly once (no build, no run)

The gates run with reduced sizes so that thirty rebuilds fit; a survivor is re-run at full size by hand. The mutant `c2-first-appearance` is judged by the row-order
gate ALONE: the issue requires that a dictionary ordered by first appearance fail it.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable


def compiler():
    here = ROOT.parent / "cancho-compiler-a4572ea" / "target" / "release" / "cancho"
    return os.environ.get("CANCHO") or (str(here) if here.exists() else "cancho")


def M(id, gates, what, edits, why=None):
    return dict(id=id, gates=gates, what=what, edits=edits, why=why)


# gate name -> (needs ml, argv run in the tree)
GATES = {
    "unit": (False, "unit"),
    "diff": (False, ["scripts/gate_cat.py", "diff", "--files", "400"]),
    "order": (False, ["scripts/gate_cat.py", "order", "--files", "80"]),
    "utf8": (False, ["scripts/gate_cat.py", "utf8"]),
    "limits": (True, ["scripts/gate_cat.py", "limits"]),
    "roundtrip": (False, ["scripts/gate_cat.py", "roundtrip", "--files", "60"]),
    "threads": (True, ["scripts/gate_cat.py", "threads"]),
    "fuzz": (True, ["scripts/gate_cat.py", "fuzz", "--runs", "300"]),
    "cli": (True, "cli"),
}

LOCATE = """        let mid = (lo + hi) / 2;
        let r = bytes.compare(text_of(d, contents(d.ent)[base + mid]), cell);
        if r == 0 {
            return mid * 2 + 1;
        }
        if r < 0 {
            lo = mid + 1;
        } else {
            hi = mid;
        }"""

FIND = '        let mid = (lo + hi) / 2;\n        let r = bytes.compare(string(v, c, mid), cell);\n        if r == 0 {\n            return mid;\n        }\n        if r < 0 {\n            lo = mid + 1;\n        } else {\n            hi = mid;\n        }'
FIND_LINEAR = '        if bytes.compare(string(v, c, lo), cell) == 0 {\n            return lo;\n        }\n        lo = lo + 1;'

LOCATE_LINEAR = """        if bytes.compare(text_of(d, contents(d.ent)[base + lo]), cell) == 0 {
            return lo * 2 + 1;
        }
        lo = lo + 1;"""

MUTANTS = [
    # ------------------------------------------------------------------ C1, C2: the order of the vocabulary
    M("c2-first-appearance", ["order"], "the vocabulary keeps the strings in the order they are met (linear search in the dictionary and in the frozen vocabulary, insertion at the end): ids are consistent, only the order of the rows decides them",
      [("lib/core/dict.cho", LOCATE, LOCATE_LINEAR), ("lib/core/dict.cho", FIND, FIND_LINEAR)]),
    M("c1-first-appearance-diff", ["diff"], "the same mutant, judged by the vocabulary gate (it is killed there too)",
      [("lib/core/dict.cho", LOCATE, LOCATE_LINEAR), ("lib/core/dict.cho", FIND, FIND_LINEAR)]),
    M("c2-first-appearance-dictionary-only", ["order"], "the dictionary appends in the order met but the frozen vocabulary is searched by binary search (lookups fail, so the file is refused: still killed)",
      [("lib/core/dict.cho", LOCATE, LOCATE_LINEAR)]),
    M("c1-descending", ["diff", "unit"], "the dictionary is sorted descending",
      [("lib/core/dict.cho", "            return mid * 2 + 1;\n        }\n        if r < 0 {", "            return mid * 2 + 1;\n        }\n        if r > 0 {")]),
    M("c1-freeze-reversed", ["diff", "unit"], "the frozen vocabulary is written in reverse",
      [("lib/core/dict.cho", "text_of(dr, contents(dr.ent)[c * dr.maxc + k])", "text_of(dr, contents(dr.ent)[c * dr.maxc + n - 1 - k])")]),
    M("c1-ids-from-the-row", ["diff", "roundtrip"], "the id of a cell is its row",
      [("tools/ml/csv.cho", "        id = dict.find(cs.voc, cj, cell);", "        id = row;")]),
    M("c1-trim", ["diff"], "a cell is trimmed before it is looked up or interned",
      [("tools/ml/csv.cho", "let (s2, a2) = cat_cell(cs, cc, raw, place - 1, row, writes);", "let (s2, a2) = cat_cell(cs, cc, bytes.trim(raw), place - 1, row, writes);")]),
    M("c1-doubled-quotes-kept", ["diff"], "a doubled quote in a quoted field is not undone",
      [("tools/ml/csv.cho", "            if quoted && fb[3 * i + 2] == 1 && index_of_byte(raw, byte_of(34)) >= 0 {", "            if false {")]),
    M("c1-category-bitmap-missing", ["diff"], "a missing cell gets the id -1 and no bitmap bit",
      [("tools/ml/csv.cho", "            catmat.mark_missing(contents(cs.cmiss), cs.cwpc, row, cj);", "            0;")]),
    M("c1-column-places-reversed", ["diff"], "the categorical columns are numbered from the right",
      [("tools/ml/csv.cho", "                count = count + 1;\n                flags[4 * cols + j] = count;", "                count = count + 1;\n                flags[4 * cols + j] = cols - j;")],
      ),
    # ------------------------------------------------------------------ C6: missing
    M("c6-empty-is-a-category", ["diff"], "an empty cell is interned (the dictionary refuses it)",
      [("tools/ml/csv.cho", "    if len(cell) > 0 && !is_marker(cell, cc.na) && !is_marker(cell, contents(cs.nab)[0..len(contents(cs.nab)) - 1]) {", "    if !is_marker(cell, cc.na) && !is_marker(cell, contents(cs.nab)[0..len(contents(cs.nab)) - 1]) {")]),
    M("c6-markers-ignored", ["diff"], "the markers of --na-values are not applied",
      [("tools/ml/csv.cho", "    if len(cell) > 0 && !is_marker(cell, cc.na) && !is_marker(cell, contents(cs.nab)[0..len(contents(cs.nab)) - 1]) {", "    if len(cell) > 0 && !is_marker(cell, contents(cs.nab)[0..len(contents(cs.nab)) - 1]) {")]),
    M("c6-marker-by-prefix", ["diff"], "a marker matches the start of a cell",
      [("tools/ml/csv.cho", "if bytes.equal(cell, bytes.field(list, ',', k)) {", "if bytes.starts_with(cell, bytes.field(list, ',', k)) {")]),
    M("c6-marker-ignoring-case", ["diff"], "a marker matches without regard to ASCII case",
      [("tools/ml/csv.cho", "        if bytes.equal(cell, bytes.field(list, ',', k)) {\n            return true;\n        }", """        let m = bytes.field(list, ',', k);
        if len(m) == len(cell) {
            var same = true;
            var q = 0;
            while q < len(m) {
                if bytes.to_lower(int_of(m[q])) != bytes.to_lower(int_of(cell[q])) {
                    same = false;
                }
                q = q + 1;
            }
            if same {
                return true;
            }
        }""")]),
    M("c6-marker-last-ignored", ["diff"], "the last marker of the list is not looked at",
      [("tools/ml/csv.cho", "    let count = bytes.count_byte(list, ',') + 1;\n    var k = 1;\n    while k <= count {\n        if bytes.equal(cell,", "    let count = bytes.count_byte(list, ',');\n    var k = 1;\n    while k <= count {\n        if bytes.equal(cell,")]),
    # ------------------------------------------------------------------ C5: UTF-8
    M("c5-intern-no-utf8-check", ["unit"], "the dictionary does not check UTF-8 (the reader still does)",
      [("lib/core/dict.cho", "    if utf8_offset(cell) >= 0 {\n        return refuse.input_invalid_utf8();", "    if false {\n        return refuse.input_invalid_utf8();")]),
    M("c5-reader-no-utf8-check", ["diff", "roundtrip"], "the reader does not check UTF-8 (the dictionary still does at fit; a lookup does not)",
      [("tools/ml/csv.cho", "        if off >= 0 {\n            return (20, off);", "        if false {\n            return (20, off);")]),
    M("c5-utf8-errors-skipped", ["utf8", "unit"], "an invalid byte is stepped over",
      [("lib/core/dict.cho", "                utf8.Step::Invalid(_) => {\n                    return at;", "                utf8.Step::Invalid(_) => {\n                    at = at + 1;")]),
    M("c5-utf8-first-byte-width", ["utf8", "unit"], "a multibyte sequence is judged by its first byte only",
      [("lib/core/dict.cho", "                utf8.Step::Code(_, w) => {\n                    at = at + w;", "                utf8.Step::Code(_, w) => {\n                    at = at + 1;")]),
    M("c5-offset-off-by-one", ["utf8", "diff"], "the offset reported is one past the bad byte",
      [("lib/core/dict.cho", "                utf8.Step::Invalid(_) => {\n                    return at;", "                utf8.Step::Invalid(_) => {\n                    return at + 1;")]),
    # ------------------------------------------------------------------ C4: the limits
    M("c4-categories-off-by-one", ["unit", "diff"], "a column may hold one string more than the limit",
      [("lib/core/dict.cho", "    if n >= d.maxc {", "    if n > d.maxc {")]),
    M("c4-categories-one-short", ["unit", "diff"], "a column may hold one string fewer than the limit",
      [("lib/core/dict.cho", "    if n >= d.maxc {", "    if n >= d.maxc - 1 {")]),
    M("c4-string-bytes-off-by-one-dict", ["unit"], "the dictionary refuses a string of exactly the limit",
      [("lib/core/dict.cho", "    if len(cell) > d.maxs {", "    if len(cell) >= d.maxs {")]),
    M("c4-string-bytes-off-by-one-reader", ["diff", "limits"], "the reader refuses a cell of exactly the limit",
      [("tools/ml/csv.cho", "        if len(cell) > cc.max_str {", "        if len(cell) >= cc.max_str {")]),
    M("c4-string-bytes-after-utf8", ["diff"], "the UTF-8 check comes before the length check",
      [("tools/ml/csv.cho", """        if len(cell) > cc.max_str {
            return (21, len(cell));
        }
        let off = dict.utf8_offset(cell);
        if off >= 0 {
            return (20, off);
        }""", """        let off = dict.utf8_offset(cell);
        if off >= 0 {
            return (20, off);
        }
        if len(cell) > cc.max_str {
            return (21, len(cell));
        }""")]),
    M("c4-pool-off-by-one", ["unit", "diff"], "the pool may be one byte over its limit",
      [("lib/core/dict.cho", "    if len(cell) > d.maxp - used {", "    if len(cell) > d.maxp - used + 1 {")]),
    M("c4-pool-counts-only-new-strings", ["unit"], "a repeated string is charged to the pool again",
      [("lib/core/dict.cho", "    let at = locate(d, col, cell);\n    if at % 2 == 1 {\n        return 0;\n    }", "    let at = locate(d, col, cell);\n    if at % 2 == 1 {\n        if len(cell) > d.maxp - contents(d.used)[0] {\n            return refuse.limit_string_pool();\n        }\n        return 0;\n    }")]),
    M("c4-columns-off-by-one-reader", ["diff", "limits"], "the reader refuses as many categorical columns as the limit",
      [("tools/ml/csv.cho", "        if count > cc.max_catcols {", "        if count >= cc.max_catcols {")]),
    M("c4-columns-off-by-one-dict", ["unit"], "the dictionary refuses as many columns as the limit",
      [("lib/core/dict.cho", "    if ncols > max_cols {", "    if ncols >= max_cols {")]),
    M("c4-dict-columns-limit-unchecked", ["unit"], "the dictionary does not compare the column count with its limit",
      [("lib/core/dict.cho", "    if ncols > max_cols {\n        return (empty_dict(heap), refuse.limit_categorical_columns());\n    }", "    if ncols > max_cols + 1000000 {\n        return (empty_dict(heap), refuse.limit_categorical_columns());\n    }")]),
    M("c4-ids-not-counted-in-cells", ["limits"], "the id cells are not added to the numeric ones in the cell cap of the loader",
      [("tools/ml/csvload.cho", "limits.check_matrix(rows, k + ncat, cfg.max_rows, cfg.max_cols, cfg.max_cells)", "limits.check_matrix(rows, k + 1, cfg.max_rows, cfg.max_cols, cfg.max_cells)")],
      "the scan's own row check divides the cell cap by every column of the header, dropped ones too, so no file reaches this second check with more cells than the cap: equivalent"),
    # ------------------------------------------------------------------ C7: the stored vocabulary
    M("c7-unseen-mapped-to-an-id", ["roundtrip"], "a string the vocabulary does not hold gets id 0",
      [("tools/ml/csv.cho", "            id = catmat.unseen_id();", "            id = 0;")]),
    M("c7-unseen-not-counted", ["roundtrip"], "unseen strings are not counted",
      [("tools/ml/csv.cho", "contents(cs.unseen)[cj] = contents(cs.unseen)[cj] + 1;", "contents(cs.unseen)[cj] = contents(cs.unseen)[cj] + 0;")]),
    M("c7-fit-lookup-miss-ignored", ["diff"], "a fit whose second pass meets a string the first did not is accepted with id -2",
      [("tools/ml/csv.cho", "                if cs.kind == 2 {\n                    return (15, 0);\n                }\n", "")],
      "pass 2 of a fit reads the file the first pass read; the gates read a file that does not change between the passes, so the branch (a file changed under the reader) is not reachable from them. A gap, named in docs/categorical.md section 9"),
    M("c7-predict-columns-from-the-flag", ["roundtrip"], "a predict takes its categorical columns from the names given, not from the vocabulary",
      [("tools/ml/csv.cho", "    if kind == 1 {\n        apply_mask(flags, cols, cc.cat_mask, 4, 1);\n        let (st, k) = apply_list(", "    if kind != 0 {\n        apply_mask(flags, cols, cc.cat_mask, 4, 1);\n        let (st, k) = apply_list(")]),
    # ------------------------------------------------------------------ C3, C11: the command line
    M("c3-threads-in-the-detail", ["threads"], "the thread count reaches the answer of the refusal",
      [("tools/ml/catwire.cho", "pub fn not_encoded[&h, &v, &n](heap: &!h Heap, errs: fail.Errors, voc: &v dict.Vocab, rem: &n names.Names)", "pub fn not_encoded[&h, &v, &n](heap: &!h Heap, errs: fail.Errors, voc: &v dict.Vocab, rem: &n names.Names, extra: int)"),
       ("tools/ml/catwire.cho", "counts = buffer.push_nat(heap, counts, dict.count(voc, at));", "counts = buffer.push_nat(heap, counts, dict.count(voc, at) + extra - 1);"),
       ("tools/ml/catcli.cho", "fn not_encoded[&h, &v](heap: &!h Heap, errs: fail.Errors, voc: &v dict.Vocab)", "fn not_encoded[&h, &v](heap: &!h Heap, errs: fail.Errors, voc: &v dict.Vocab, extra: int)"),
       ("tools/ml/catcli.cho", "out = catwire.not_encoded(heap, out, voc, nr);", "out = catwire.not_encoded(heap, out, voc, nr, extra);"),
       ("tools/ml/catcli.cho", "e = not_encoded(heap, e, vr);", "e = not_encoded(heap, e, vr, o.threads);"),
       ("tools/ml/wire.cho", "e = catwire.not_encoded(heap, e, cvoc, remr);", "e = catwire.not_encoded(heap, e, cvoc, remr, o.threads);")]),
    M("c11-invalid-utf8-reported-as-a-length", ["cli"], "the refusal of invalid UTF-8 carries the rule of the byte limit",
      [("tools/ml/verbs.cho", '        rule = "ml.input.invalid-utf8";\n        message', '        rule = "ml.limit.string-bytes";\n        message')]),
    M("c11-target-may-be-categorical", ["cli", "diff"], "--cat on the target is accepted",
      [("tools/ml/csv.cho", "                if j == tcol {\n                    return (cs, 27, j, 0);\n                }\n                if flags[j] == 3 {", "                if flags[j] == 3 {")]),
    M("c11-limit-flag-without-cat-accepted", ["cli"], "a limit flag without --cat is accepted",
      [("tools/ml/app.cho", """    if cli.has(parsed, table, "max-categories") {
        e = flag_problem(heap, e, "args.conflict", "this flag needs --cat", "name the categorical columns with --cat", "flag", "--max-categories");
    }""", "")]),
    M("c11-na-values-list-not-checked", ["cli"], "an empty marker in --na-values is accepted",
      [("tools/ml/app.cho", "                if len(m) == 0 || len(m) > 255 {", "                if len(m) > 255 {")]),
    M("c11-cat-fit-writes-a-model", ["cli"], "the refusal is not reached: --cat with a read that succeeds falls through to the numeric fit",
      [("tools/ml/main.cho", "        if o.cat_given && !wire.planned(o.plan_given, o.model) {\n            return (catcli.fit(heap, args, fs, root, ps.file, o, e3, lg), log_ok);\n        }", "")]),
    M("c11-unknown-cat-name-ignored", ["cli"], "--cat naming a column that is not in the header is not an error",
      [("tools/ml/csv.cho", "        let (st, k) = apply_list(flags, names, ends, cols, cc.cols, 4, 1, 25);\n        if st != 0 {", "        let (st, k) = apply_list(flags, names, ends, cols, cc.cols, 4, 1, 25);\n        if st != 0 && false {")]),
]


def check_pattern(m):
    problems = []
    for file, old, new in m["edits"]:
        p = ROOT / file
        if not p.exists():
            problems.append("%s: no such file %s" % (m["id"], file))
            continue
        n = p.read_text().count(old)
        if n != 1:
            problems.append("%s: pattern occurs %d times in %s (must be exactly 1)" % (m["id"], n, file))
        if old == new:
            problems.append("%s: replacement equals pattern" % m["id"])
    return problems


def tree_copy(dest):
    ignore = shutil.ignore_patterns(".git", "build", "__pycache__", "*.pyc", "docs")
    shutil.copytree(ROOT, dest, ignore=ignore, symlinks=True)
    (dest / "build").mkdir(exist_ok=True)
    shutil.copytree(ROOT / "build" / "deps", dest / "build" / "deps")


def run_gates(tree, env, gates):
    failures = []
    for g in gates:
        needs_ml, what = GATES[g]
        if what == "unit":
            r = subprocess.run([compiler(), "test", "--std"] + sorted(str(p.relative_to(tree)) for p in (tree / "lib" / "core").glob("*.cho")) + ["lib/linalg/tensor.cho", "tests/cat.cho"],
                               capture_output=True, text=True, env=env, cwd=tree, timeout=900)
            if r.returncode != 0:
                failures.append("unit: " + (r.stdout.splitlines() or ["?"])[-1][:120])
            continue
        if what == "cli":
            r = subprocess.run([PY, "-m", "unittest", "test_cat", "test_rules"], capture_output=True, text=True, env=env, cwd=tree / "tests" / "conformance", timeout=900)
            if r.returncode != 0:
                failures.append("cli: " + (r.stderr.strip().splitlines() or ["?"])[-1][:140])
            continue
        r = subprocess.run([PY, str(tree / what[0])] + what[1:], capture_output=True, text=True, env=env, cwd=tree, timeout=1800)
        if r.returncode != 0:
            lines = [l for l in r.stdout.splitlines() if "DIFF" in l or "differences" in l]
            failures.append("%s: %s" % (g, (lines[-1] if lines else r.stdout[-200:] + r.stderr[-200:]).strip()[:160]))
    return failures


def build_and_gate(m, gates):
    with tempfile.TemporaryDirectory(prefix="ml-cat-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        tree_copy(tree)
        if m:
            for file, old, new in m["edits"]:
                path = tree / file
                text = path.read_text()
                if text.count(old) != 1:
                    return "INVALID", "pattern does not apply to the copy"
                path.write_text(text.replace(old, new, 1))
        env = dict(os.environ, CANCHO=compiler())
        b = subprocess.run(["sh", str(tree / "scripts" / "build_reader.sh"), str(tree / "build" / "reader-check")], capture_output=True, text=True, env=env, cwd=tree)
        if b.returncode != 0:
            return "INVALID", "reader-check build failed: " + (b.stdout + b.stderr)[:300]
        env["READER"] = str(tree / "build" / "reader-check")
        if any(GATES[g][0] for g in gates):
            b = subprocess.run([compiler(), "build", "--bin", "ml"], capture_output=True, text=True, env=env, cwd=tree)
            if b.returncode != 0:
                return "INVALID", "ml build failed: " + (b.stdout + b.stderr)[:300]
            env["ML"] = str(tree / "build" / "ml")
            env["ML_BIN"] = str(tree / "build" / "ml")
        failures = run_gates(tree, env, gates)
        return ("KILLED", "; ".join(failures)) if failures else ("SURVIVED", "")


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        for m in MUTANTS:
            print("%-44s %-22s %s" % (m["id"], ",".join(m["gates"]), m["what"]))
        return 0
    bad = []
    for m in MUTANTS:
        bad += check_pattern(m)
    if bad:
        print("\n".join(bad))
        return 1
    if "--check" in argv:
        print("%d categorical mutants, every pattern applies once" % len(MUTANTS))
        return 0
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    if not only:
        allg = sorted({g for m in MUTANTS for g in m["gates"]})
        verdict, detail = build_and_gate(None, allg)
        print("control (unmutated): %s" % ("passes every gate" if verdict == "SURVIVED" else "FAILS: " + detail), flush=True)
        if verdict != "SURVIVED":
            return 1
    killed, survived, explained = 0, [], 0
    for m in MUTANTS:
        if only and m["id"] != only:
            continue
        verdict, detail = build_and_gate(m, m["gates"])
        if verdict == "KILLED":
            killed += 1
            print("%-44s KILLED   (%s) %s" % (m["id"], ",".join(m["gates"]), detail[:120]), flush=True)
        elif verdict == "SURVIVED" and m["why"]:
            explained += 1
            print("%-44s SURVIVED, explained: %s" % (m["id"], m["why"][:140]), flush=True)
        else:
            survived.append(m["id"])
            print("%-44s %s %s" % (m["id"], verdict, detail), flush=True)
    print("%d killed, %d survived and explained, %d unexplained or invalid" % (killed, explained, len(survived)))
    if survived:
        print("UNEXPLAINED: " + ", ".join(survived))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

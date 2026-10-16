#!/usr/bin/env python3
"""Mutants for the reader gates of docs/reader.md section 5: a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change (several edits for the few that need two) to a source of this repository, or to
the installed text of one of the five cancho-table modules, in the copy's own build/deps (the tree's build/deps is never touched), the reader-check binary rebuilt from the
copy, and the gate named in the mutant run against it. KILLED: the gate reported differences, a trap or a failed build of a check; SURVIVED:
it reported none, and the mutant then carries a reason (an equivalent mutant) or is a gap that docs/reader.md section 6 names.

    python3 scripts/mutants_reader.py              # control, then all mutants
    python3 scripts/mutants_reader.py --only ID    # one
    python3 scripts/mutants_reader.py --list
    python3 scripts/mutants_reader.py --check      # every `old` text is found exactly once where it is expected (no build, no run)

The gates run with reduced sizes (diff 300,000 cells, fuzz 1,500 runs) so that thirty rebuilds fit; a survivor is re-run at full size by hand.
"""
import glob
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def dep_file(deps, name):
    """The file of `deps` (a build/deps directory) that declares cancho-table's module `name`: the installed text of the pinned store."""
    hits = [f for f in sorted(pathlib.Path(deps).glob("*.cho")) if re.search(r"^module %s;" % name, f.read_text(), re.M)]
    assert len(hits) == 1, "module %s is in %d files of %s" % (name, len(hits), deps)
    return hits[0]
PY = sys.executable


def compiler():
    here = ROOT.parent / "cancho-compiler-0207556" / "target" / "release" / "cancho"
    return os.environ.get("CANCHO") or (str(here) if here.exists() else "cancho")


# id, gate, what, edits [(file, old, new)], reason (for an expected survivor)
# `file` is a path in this repository, or table:NAME.cho for the installed text of a cancho-table module (in the copy's build/deps)
def M(id, gate, what, edits, why=None):
    return dict(id=id, gate=gate, what=what, edits=edits, why=why)


GATES = {
    "diff": ["diff", "--cells", "300000"],
    "matrix": ["matrix"],
    "limits": ["limits"],
    "csv": ["csv"],
    "fuzz": ["fuzz", "--runs", "1500"],
}

MUTANTS = [
    # ------------------------------------------------------------------ RD2: the reader against Python
    M("rd2-clinger-limit-23", "diff", "Clinger's fast path allows 10^23 (not exactly representable)",
      [("table:flt.cho", "e10 >= 0 - 22 && e10 <= 22 {", "e10 >= 0 - 22 && e10 <= 23 {")]),
    M("rd2-clinger-mantissa-2p53-plus-1", "diff", "Clinger's fast path allows a mantissa of 2^53 + 1",
      [("table:flt.cho", "m <= 9007199254740992 && e10", "m <= 9007199254740993 && e10")]),
    M("rd2-sticky-retry-removed", "diff", "the retry with the 19-digit window plus one is skipped (digits dropped can change the rounding)",
      [("table:flt.cho", "if dropped && eisel_lemire(e10, wrapping_add(m, 1)) != bits {", "if false && eisel_lemire(e10, wrapping_add(m, 1)) != bits {")]),
    M("rd2-halfway-test-removed", "diff", "Eisel-Lemire's exact-halfway correction is skipped",
      [("table:flt.cho", "if lo >= 0 && lo <= 1 && q >= 0 - 4", "if false && lo >= 0 && lo <= 1 && q >= 0 - 4")]),
    M("rd2-truncate-not-round", "diff", "Eisel-Lemire truncates instead of rounding to nearest",
      [("table:flt.cho", "    mant = mant + (mant & 1);\n    mant = mant >> 1;\n    if mant >= 9007199254740992 {", "    mant = mant + 0;\n    mant = mant >> 1;\n    if mant >= 9007199254740992 {")]),
    M("rd2-negative-sign-dropped", "diff", "a float cell's sign is dropped (the key is the positive one)",
      [("table:flt.cho", "    if negative {\n        return (0 - 1 - bits, 0);\n    }", "    if false {\n        return (0 - 1 - bits, 0);\n    }")]),
    M("rd2-signed-zero-refused", "diff", "-0e5 is refused as underflow instead of read as 0",
      [("table:flt.cho", "    if !nonzero {\n        return (0, 0);", "    if !nonzero && !negative {\n        return (0, 0);")]),
    M("rd2-subnormal-flush", "diff", "a result that rounds to a subnormal is read as zero",
      [("table:flt.cho", "        if 1 - power2 >= 64 {\n            return 0;\n        }", "        if 1 - power2 >= 0 {\n            return 0;\n        }")]),
    M("rd2-sign-dropped-int", "diff", "the sign of an int or a short decimal is dropped",
      [("tools/ml/csvcell.cho", "        neg = first == '-';", "        neg = false;")]),
    M("rd2-reciprocal", "diff", "the value of a short decimal is m * (1 / 10^S) instead of m / 10^S",
      [("tools/ml/csvcell.cho", "    return float_of(m) / pow10f[frac];\n}\n\n// the order-preserving", "    return float_of(m) * (1.0 / pow10f[frac]);\n}\n\n// the order-preserving")]),
    M("rd2-digits-11", "matrix", "the exact path takes 11 digits, not 12 (a 12-digit decimal becomes a float column)",
      [("tools/ml/csvcell.cho", "    if idig + frac > 12 {\n        return (5, 0, 0, 0);", "    if idig + frac > 11 {\n        return (5, 0, 0, 0);")]),
    M("rd2-point-forms-refused", "diff", ".5 and 5. are refused (they are not tried on the float reader)",
      [("tools/ml/csvcell.cho", "    if st == 1 || st == 3 {\n        return (st, 0, 0, 0, 0, 0);", "    if st == 1 || st == 2 || st == 3 {\n        return (st, 0, 0, 0, 0, 0);")]),
    M("rd2-nan-stored", "diff", "nan and inf are read as the number 0 (both layers that refuse them are changed)",
      [("tools/ml/csvcell.cho", 'same_ignoring_case(rest, "nan") || same_ignoring_case(rest, "inf") || same_ignoring_case(rest, "infinity")', "false"),
       ("tools/ml/csvcell.cho", "    if fs == 10 {\n        return (3, 0, 0, 0, 0, 0);\n    }", "    if fs == 10 {\n        return (0, 3, 0, 0, 0, 0);\n    }")]),
    M("rd2-nan-one-layer", "diff", "only csvcell's own nan/inf check is removed (flt refuses them as well)",
      [("tools/ml/csvcell.cho", 'same_ignoring_case(rest, "nan") || same_ignoring_case(rest, "inf") || same_ignoring_case(rest, "infinity")', "false")],
      why="equivalent: nan and inf are refused twice, by ml.csvcell.decimal and by flt.parse_float (its status 10), and the answer is the same"),
    M("rd2-overflow-accepted", "diff", "a cell past the range of a double is read as the largest double instead of refused",
      [("tools/ml/csvcell.cho", "    if fs == 11 {\n        return (4, 0, 0, 0, 0, 0);\n    }", "    if fs == 11 {\n        return (0, 3, 0, 0, 0, 9218868437227405311);\n    }")]),
    M("rd2-underflow-accepted", "diff", "a non-zero cell that reads as zero is accepted as zero",
      [("tools/ml/csvcell.cho", "    if fs == 12 {\n        return (5, 0, 0, 0, 0, 0);\n    }", "    if fs == 12 {\n        return (0, 3, 0, 0, 0, 0);\n    }")]),
    M("rd2-long-cell-accepted-as-garbage", "diff", "a cell longer than 1,100 bytes is reported as 'not a number'",
      [("tools/ml/csvcell.cho", "    if fs == 13 {\n        return (6, 0, 0, 0, 0, 0);\n    }", "    if fs == 13 {\n        return (2, 0, 0, 0, 0, 0);\n    }")]),
    M("rd2-empty-cell-check-removed", "diff", "the empty-cell check is removed (cell[0] is read of an empty cell)",
      [("tools/ml/csvcell.cho", "    let n = len(cell);\n    if n == 0 {\n        return (1, 0, 0, 0);\n    }", "    let n = len(cell);")]),
    # ------------------------------------------------------------------ RD3: the matrix and the bitmap
    M("rd3-missing-as-zero", "matrix", "an empty cell is stored as 0.0 with no bit",
      [("tools/ml/csv.cho", "matrix.mark_missing(miss, cfg.wpc, row, mj);", "data[row * cfg.mcols + mj] = 0.0;")]),
    M("rd3-bit-off-by-one", "matrix", "the bit of row r is bit r + 1",
      [("lib/core/matrix.cho", "miss[w] = miss[w] | 1 << (r & 31);", "miss[w] = miss[w] | 1 << ((r + 1) & 31);")]),
    M("rd3-word-of-64", "matrix", "the word of row r is r / 64 (rows 32 to 63 collide with 0 to 31)",
      [("lib/core/matrix.cho", "    let w = c * wpc + (r >> 5);\n    miss[w] = miss[w]", "    let w = c * wpc + (r >> 6);\n    miss[w] = miss[w]")]),
    M("rd3-next-column", "matrix", "the bit is written to the next column's words",
      [("lib/core/matrix.cho", "    let w = c * wpc + (r >> 5);\n    miss[w] = miss[w]", "    let w = (c + 1) * wpc + (r >> 5);\n    miss[w] = miss[w]")]),
    M("rd3-types-do-not-widen", "matrix", "a column's kind is the kind of its last cell, not the widest",
      [("tools/ml/csv.cho", "                if flags[i] < kind {\n                    flags[i] = kind;\n                }", "                if flags[i] != kind {\n                    flags[i] = kind;\n                }")]),
    M("rd3-drop-ignored", "matrix", "--drop marks nothing",
      [("tools/ml/csv.cho", "cfg.drops, 3, 1, 12", "cfg.drops, 3, 0, 12")]),
    M("rd3-float-flag-ignored", "matrix", "--float forces nothing",
      [("tools/ml/csv.cho", "cfg.floats, 0, 3, 19", "cfg.floats, 0, 0, 19")]),
    M("rd3-column-index-counts-dropped", "matrix", "the matrix column of a cell is its file column, dropped ones included",
      [("tools/ml/csv.cho", "        if !dropped && !skipped && place == 0 {\n            mj = mj + 1;\n        }", "        mj = mj + 1;")]),
    # ------------------------------------------------------------------ RD4: the limits
    M("rd4-row-cap-off-by-one", "limits", "one more row than --max-rows is read",
      [("tools/ml/csv.cho", "if rows > cfg.max_rows {", "if rows > cfg.max_rows + 1 {")]),
    M("rd4-column-cap-off-by-one", "limits", "one more column than --max-columns is read",
      [("tools/ml/csv.cho", "if c > cfg.max_cols {", "if c > cfg.max_cols + 1 {")]),
    M("rd4-cell-cap-off-by-one", "limits", "one more cell than --max-cells is read",
      [("tools/ml/csv.cho", "} else if rows > cfg.max_cells / cols {", "} else if rows > cfg.max_cells / cols + 1 {")]),
    M("rd4-record-cap-removed", "limits", "the line cap is not applied to a record that spans lines",
      [("tools/ml/csv.cho", "if have_rec && held + 1 + len(line) > cfg.cap {", "if false && held + 1 + len(line) > cfg.cap {")]),
    M("rd4-matrix-before-check", "limits", "the matrix is allocated before the caps are compared (a 200 MB refusal would hold the data)",
      [("lib/core/matrix.cho", "    let code = limits.check_matrix(rows, cols, rows_cap, cols_cap, cells_cap);\n    if code != 0 {\n        return (empty(heap), code);\n    }",
        "    var big = box_slice(heap, rows * cols + 1, 1.0);\n    var probe = 0.0;\n    borrow mut big as &!bw in {\n        contents(bw)[rows * cols / 2] = 3.0;\n        probe = contents(bw)[rows * cols / 3] + contents(bw)[rows * cols / 2];\n    }\n    unbox_slice(heap, big);\n    if probe > 100.0 {\n        return (empty(heap), 1);\n    }\n    let code = limits.check_matrix(rows, cols, rows_cap, cols_cap, cells_cap);\n    if code != 0 {\n        return (empty(heap), code);\n    }")]),
    M("rd4-cell-cap-by-product", "limits", "the cell cap is compared as rows * cols > cap",
      [("tools/ml/csv.cho", "} else if rows > cfg.max_cells / cols {", "} else if rows * cols > cfg.max_cells {")],
      why="equivalent: every cap is at most 2^31 - 1 (ml.limits.ceiling_dim) so rows * cols stays below 2^62; the division is defence in depth, not a different answer"),
    # ------------------------------------------------------------------ RD5: the CSV shape
    M("rd5-cr-kept", "csv", "a CR before the LF is part of the last cell",
      [("tools/ml/csv.cho", "if len(line) > 0 && int_of(line[len(line) - 1]) == 13 {", "if len(line) > 0 && int_of(line[len(line) - 1]) == 14 {")]),
    M("rd5-bom-kept", "csv", "the byte order mark is part of the first column's name",
      [("tools/ml/csv.cho", "int_of(line[0]) == 239 && int_of(line[1]) == 187", "int_of(line[0]) == 238 && int_of(line[1]) == 187")]),
    M("rd5-bad-quote-accepted", "csv", "a closing quote followed by a letter is not a bad quote",
      [("table:reader.cho", "                } else {\n                    bad = true;\n                }\n            }\n        }\n    }\n    return (inside, n, bad);", "                } else {\n                    bad = false;\n                }\n            }\n        }\n    }\n    return (inside, n, bad);")]),
    M("rd5-spanning-record-line", "csv", "the line of a record that spans lines is the line it ends at",
      [("tools/ml/csv.cho", "            rec = buffer.push(heap, rec, byte_of(10));", "            rec = buffer.push(heap, rec, byte_of(10));\n            rec_line = number;")]),
    M("rd5-unterminated-quote-accepted", "csv", "a quote open at the end of the file is dropped silently",
      [("tools/ml/csv.cho", "            if have_rec && rep.status == 0 {\n                rep = report(18, rec_line, 0, 0);\n            }", "")]),
    M("rd5-blank-line-in-quote-skipped", "csv", "a blank line inside a quoted field is skipped",
      [("tools/ml/csv.cho", "if len(line) == 0 && !have_rec {", "if len(line) == 0 {")],
      why="equivalent for everything a numeric reader can observe: a blank line inside a quoted field changes the field's text, which is never read as a number (it is dropped or refused), not the quote state, the field count or any line number; only the record's byte length for the cap differs"),
    # ------------------------------------------------------------------ RD6: no trap
    M("rd6-field-count-not-checked", "fuzz", "the number of fields of a row is not compared with the header's",
      [("tools/ml/csv.cho", "} else if seps + 1 != cols {", "} else if false {")]),
    M("rd6-fields-count-ignored", "fuzz", "the field count of reader.fields is ignored for a quoted row",
      [("tools/ml/csv.cho", "        if inside || bad || n != cols {\n            return (3, 0, 0);\n        }", "")],
      why="equivalent: reader.scan has already counted the same fields and flagged the same quotes of the same record (seps + 1 != cols is checked first), and the two functions are kept in step by cancho-table's own differential test"),
]


def check_pattern(m):
    problems = []
    for file, old, new in m["edits"]:
        if file.startswith("table:"):
            p = dep_file(ROOT / "build" / "deps", file[6:-4])
        else:
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
    deps = ROOT / "build" / "deps"
    (dest / "build").mkdir(exist_ok=True)
    shutil.copytree(deps, dest / "build" / "deps")


def build_and_gate(m, gates):
    with tempfile.TemporaryDirectory(prefix="ml-rd-mutant-") as td:
        td = pathlib.Path(td)
        tree = td / "tree"
        tree_copy(tree)
        if m:
            for file, old, new in m["edits"]:
                path = dep_file(tree / "build" / "deps", file[6:-4]) if file.startswith("table:") else tree / file
                text = path.read_text()
                if text.count(old) != 1:
                    return "INVALID", "pattern does not apply to the copy"
                path.write_text(text.replace(old, new, 1))
        env = dict(os.environ, CANCHO=compiler())
        b = subprocess.run(["sh", str(tree / "scripts" / "build_reader.sh"), str(tree / "build" / "reader-check")], capture_output=True, text=True, env=env, cwd=tree)
        if b.returncode != 0:
            return "INVALID", "build failed: " + (b.stdout + b.stderr)[:300]
        env["READER"] = str(tree / "build" / "reader-check")
        failures = []
        for g in gates:
            r = subprocess.run([PY, str(ROOT / "scripts" / "gate_reader.py")] + GATES[g], capture_output=True, text=True, env=env, timeout=1800)
            if r.returncode != 0:
                lines = [l for l in r.stdout.splitlines() if "DIFFERENCE" in l or "TRAP" in l or "differences" in l or "traps" in l]
                failures.append("%s: %s" % (g, (lines[0] if lines else r.stdout[-200:] + r.stderr[-200:]).strip()[:160]))
        return ("KILLED", "; ".join(failures)) if failures else ("SURVIVED", "")


def pin_mutants():
    """RD1: the pin check must fail on a rev that differs between modules, a rev the docs do not document, a short rev, a wrong path, a
    missing dependency, a dependency dropped, a sixth cancho-table store added (not in the documented list), a document row that names a
    sixth or leaves one out, a local sources entry, and a module declared under lib/, tools/ or tests/ (a copy)."""
    results = []
    script = ROOT / "scripts" / "check_table_pin.sh"
    toml = (ROOT / "cancho.toml").read_text()
    doc = (ROOT / "docs" / "reader.md").read_text()
    control = subprocess.run(["sh", str(script)], capture_output=True, text=True, cwd=ROOT)
    results.append(("rd1-control", control.returncode == 0, control.stdout.strip() or control.stderr.strip()))
    m = re.search(r'\[dependencies\.flt\]\ngit = "[^"]*"\nrev = "([0-9a-f]{40})"', toml)
    assert m, "no [dependencies.flt] in cancho.toml"
    good = m.group(1)
    other = ("0" if good[0] != "0" else "1") + good[1:]

    def run(name, toml_text=None, doc_text=None, files=None):
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            env = dict(os.environ)
            cwd = ROOT
            if toml_text is not None:
                (td / "cancho.toml").write_text(toml_text)
                env["TOML"] = str(td / "cancho.toml")
            if doc_text is not None:
                (td / "reader.md").write_text(doc_text)
                env["DOC"] = str(td / "reader.md")
            if files:
                tree = td / "tree"
                shutil.copytree(ROOT, tree, ignore=shutil.ignore_patterns(".git", "build", "__pycache__"))
                for f, text in files.items():
                    (tree / f).write_text(text)
                cwd = tree
            r = subprocess.run(["sh", str(cwd / "scripts" / "check_table_pin.sh")], capture_output=True, text=True, cwd=cwd, env=env)
            results.append((name, r.returncode != 0, r.stderr.strip()))

    flt_blk = 'git = "https://github.com/alpibrusl/cancho-table"\nrev = "%s"\npath = ".cancho-vcs/flt"' % good
    run("rd1-rev-of-one-module-differs", toml.replace(flt_blk, flt_blk.replace(good, other)))
    run("rd1-rev-not-the-documented-one", None, doc.replace("cancho-table rev: `%s`" % good, "cancho-table rev: `%s`" % other))
    run("rd1-rev-short", toml.replace('rev = "%s"\npath = ".cancho-vcs/flt"' % good, 'rev = "%s"\npath = ".cancho-vcs/flt"' % good[:7]))
    run("rd1-path-of-another-module", toml.replace('path = ".cancho-vcs/flt"', 'path = ".cancho-vcs/dec"'))
    run("rd1-dependency-missing", toml.replace("[dependencies.reader]", "[dependencies.readerx]"))
    pow5_blk = '[dependencies.pow5]\ngit = "https://github.com/alpibrusl/cancho-table"\nrev = "%s"\npath = ".cancho-vcs/pow5"\n\n' % good
    assert pow5_blk in toml, "no [dependencies.pow5] block of the expected form"
    run("rd1-dependency-dropped", toml.replace(pow5_blk, ""))
    run("rd1-sixth-store-added", toml.replace(pow5_blk, pow5_blk + pow5_blk.replace("pow5", "query")))
    row = [l for l in doc.splitlines() if l.startswith("| RD1 ")][0]
    run("rd1-doc-row-names-a-sixth", None, doc.replace(row, row + " query"))
    run("rd1-doc-row-leaves-one-out", None, doc.replace(row, re.sub(r"`facc`", "`fac`", row)))
    run("rd1-local-sources-entry", toml.replace('"tools/ml", "generated/ml"', '"tools/ml", "generated/ml", "../cancho-table-main/tools/table/flt.cho"'))
    run("rd1-vendored-copy", files={"tools/ml/flt.cho": "module flt;\n"})
    run("rd1-vendored-copy-under-lib", files={"lib/core/reader.cho": "// edited\nmodule reader;\n"})
    return results


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        for m in MUTANTS:
            print("%-40s %-7s %s" % (m["id"], m["gate"], m["what"]))
        return 0
    bad = []
    for m in MUTANTS:
        bad += check_pattern(m)
    if bad:
        print("\n".join(bad))
        return 1
    if "--check" in argv:
        print("%d reader mutants, every pattern applies once" % len(MUTANTS))
        return 0
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    log = []
    if not only:
        verdict, detail = build_and_gate(None, list(GATES))
        print("control (unmutated): %s" % ("passes every gate" if verdict == "SURVIVED" else "FAILS: " + detail), flush=True)
        if verdict != "SURVIVED":
            return 1
        for name, ok, text in pin_mutants():
            print("%-44s %s  %s" % (name, "KILLED" if ok and name != "rd1-control" else ("passes" if ok else "FAILS"), text[:90]), flush=True)
            log.append((name, "KILLED" if ok else "SURVIVED"))
    killed, survived, explained = 0, [], 0
    for m in MUTANTS:
        if only and m["id"] != only:
            continue
        gate = m["gate"]
        verdict, detail = build_and_gate(m, [gate])
        if verdict == "KILLED":
            killed += 1
            print("%-40s KILLED   (%s) %s" % (m["id"], gate, detail[:110]), flush=True)
        elif verdict == "SURVIVED" and m["why"]:
            explained += 1
            print("%-40s SURVIVED, explained: %s" % (m["id"], m["why"][:120]), flush=True)
        else:
            survived.append(m["id"])
            print("%-40s %s %s" % (m["id"], verdict, detail), flush=True)
    print("%d killed, %d survived and explained, %d unexplained or invalid" % (killed, explained, len(survived)))
    if survived:
        print("UNEXPLAINED: " + ", ".join(survived))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

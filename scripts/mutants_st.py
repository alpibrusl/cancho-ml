#!/usr/bin/env python3
"""Mutants for the gates of docs/stringtable.md section 9: a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change (a few have two) to a source of this repository; what the mutant touches is rebuilt in the copy (the unit programs,
`ml`) and the gates named for the mutant run against it. KILLED: a gate reported a difference, a trap or a failed test; SURVIVED: none did, and the mutant then carries a
reason (an equivalent mutant) or is a gap that docs/stringtable.md section 10 names.

    python3 scripts/mutants_st.py              # control, then all mutants
    python3 scripts/mutants_st.py --only ID    # one
    python3 scripts/mutants_st.py --list
    python3 scripts/mutants_st.py --check      # every `old` text is found exactly once (no build, no run)

The gates run with reduced sizes (ST_FAST) so that sixty rebuilds fit; a survivor is re-run at full size by hand. Run it on a frozen copy of the tree: a concurrent
`cancho install` rewrites build/deps.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable
STR = "lib/transforms/strtab.cho"
CAT = "lib/transforms/catenc.cho"
FILE = "lib/pipeline/file.cho"
MIX = "lib/pipeline/mixed.cho"
CSV = "tools/ml/csv.cho"
WIO = "tools/ml/wireio.cho"
WIRE = "tools/ml/wire.cho"
CW = "tools/ml/catwire.cho"
KIND = "lib/core/kinds.cho"


def compiler():
    here = ROOT.parent / "cancho-compiler-a4572ea" / "target" / "release" / "cancho"
    return os.environ.get("CANCHO") or (str(here) if here.exists() else "cancho")


def M(id, gates, what, edits, why=None):
    return dict(id=id, gates=gates, what=what, edits=edits, why=why)


def fn_returns(path, name, old, new):
    return (path, "pub fn %s() -> [] int {\n    return %s;" % (name, old), "pub fn %s() -> [] int {\n    return %s;" % (name, new))


GATES = {
    "unit": "unit",
    "diff": ["scripts/gate_st.py", "diff", "--files", "60"],
    "order": ["scripts/gate_st.py", "order"],
    "roundtrip": ["scripts/gate_st.py", "roundtrip"],
    "predict": ["scripts/gate_st.py", "predict"],
    "align": ["scripts/gate_st.py", "align"],
    "privacy": ["scripts/gate_st.py", "privacy"],
    "limits": ["scripts/gate_st.py", "limits"],
    "fuzz": ["scripts/gate_st.py", "fuzz", "--runs", "150"],
    "cli": "cli",
    "registry": "registry",
}

MUTANTS = [
    # ------------------------------------------------------------------ the layout of the table (S1)
    M("l-table-layout-number", ["diff", "unit"], "the layout number written is 2", [fn_returns(KIND, "table_layout", "1", "2")]),
    M("l-entries-count", ["diff", "unit"], "the header says one entry more than the table holds", [(STR, "    table = put32(heap, table, cats);", "    table = put32(heap, table, cats + 1);")]),
    M("l-pool-field", ["diff", "unit"], "the pool bytes of the header are one too many", [(STR, "    table = put64(heap, table, pool);", "    table = put64(heap, table, pool + 1);")]),
    M("l-strings-field", ["diff", "unit"], "the number of strings of the header is one too many", [(STR, "    table = put64(heap, table, strings);", "    table = put64(heap, table, strings + 1);")]),
    M("l-na-not-stored", ["diff", "cli"], "the markers are not stored (the length is 0)", [(STR, "    table = put32(heap, table, len(na));\n    table = buffer.append(heap, table, na);", "    table = put32(heap, table, 0);")]),
    M("l-na-not-passed", ["diff", "cli"], "the fit passes no markers to the table", [(WIRE, "strtab.pack(heap, buffer.bytes(tr.fitted.payload), o.na, o.max_cats", "strtab.pack(heap, buffer.bytes(tr.fitted.payload), \"\", o.max_cats")]),
    M("l-limits-swapped", ["diff", "unit"], "max_categories and max_string_bytes are written in the other order", [(STR, "    table = put32(heap, table, max_cats);\n    table = put32(heap, table, max_str);", "    table = put32(heap, table, max_str);\n    table = put32(heap, table, max_cats);")]),
    M("l-string-limit-constant", ["diff", "cli"], "the string limit stored is always 255", [(WIRE, "o.max_cats, o.max_str, o.max_pool, o.max_catcols);", "o.max_cats, 255, o.max_pool, o.max_catcols);")]),
    M("l-digest-span", ["diff", "unit"], "the digest covers the table but its first byte", [(STR, "        dg = digest_into(heap, dg, buffer.bytes(tr));", "        dg = digest_into(heap, dg, buffer.bytes(tr)[1..len(buffer.bytes(tr))]);")]),
    M("l-step-number", ["diff", "unit"], "an entry carries the 0-based place of its step", [(STR, "            table = put32(heap, table, i + 1);", "            table = put32(heap, table, i);")]),
    M("l-entry-length", ["diff", "unit"], "an entry's length is one too many", [(STR, "            table = put64(heap, table, l);", "            table = put64(heap, table, l + 1);")]),
    M("l-stub-layout", ["diff", "unit"], "a stub carries the layout number of the table", [(STR, "            section = put32(heap, section, kinds.stub_layout());", "            section = put32(heap, section, kinds.table_layout());")]),
    M("l-schema-type", ["diff", "cli"], "a categorical column is typed as an integer in the schema", [(CW, "contents(tw)[2 * at] = kinds.schema_type_categorical();", "contents(tw)[2 * at] = 0;")]),
    M("l-meta-columns", ["diff", "roundtrip"], "the meta counts one column less than the schema", [(WIRE, "contents(mr)[0..end], tr.rows, fcols, o.seed", "contents(mr)[0..end], tr.rows, fcols - 1, o.seed")]),
    M("l-table-always", ["cli"], "a file always has the sixth section, empty when no step needs it", [(FILE, "    if len(table) > 0 {\n        out = put32(heap, out, kinds.section_count());", "    if true {\n        out = put32(heap, out, kinds.section_count());")]),
    # ------------------------------------------------------------------ the bounds (S7)
    M("b-strings-bound", ["unit"], "the strings bound is 131,073", [fn_returns(STR, "max_strings", "131072", "131073")]),
    M("b-pool-bound", ["unit"], "the pool bound is 8 MiB + 1", [fn_returns(STR, "max_pool", "8388608", "8388609")]),
    M("b-table-bound", ["unit"], "the table bound is 12 MiB + 1", [fn_returns(STR, "max_bytes", "12582912", "12582913")]),
    M("b-entries-bound", ["unit"], "the entries bound is 65", [fn_returns(STR, "max_entries", "64", "65")]),
    M("b-na-bound", ["unit"], "the markers text may have 16,384 bytes", [fn_returns(STR, "max_na", "16383", "16384")]),
    M("b-markers-bound", ["unit"], "65 markers are read", [fn_returns(STR, "max_markers", "64", "65")]),
    M("b-marker-length", ["unit"], "a marker of 256 bytes is read", [fn_returns(STR, "max_marker", "255", "256")]),
    M("b-size-compared-with-less-or-equal", ["unit"], "a table of exactly 12 MiB is over the bound", [(STR, "    if n > max_bytes() || n < 28 + 16 + 32 {", "    if n >= max_bytes() || n < 28 + 16 + 32 {")]),
    M("b-digest-unchecked", ["unit", "fuzz"], "the digest of the table is not compared", [(STR, "    if !digest_ok(heap, table) {", "    if false {")]),
    M("b-step-order", ["unit", "cli"], "two entries may have the same step", [(STR, "            if step <= prev || step > max_entries() {", "            if step < prev || step > max_entries() {")]),
    M("b-utf8-names", ["unit", "fuzz"], "a name that is not UTF-8 is read", [(STR, "                            if dict.utf8_offset(nm) >= 0 {", "                            if false {")]),
    M("b-utf8-strings", ["unit", "fuzz"], "a string that is not UTF-8 is read", [(STR, "                                } else if dict.utf8_offset(s) >= 0 {", "                                } else if false {")]),
    M("b-duplicate-names", ["unit", "fuzz"], "one column read by two steps is read", [(STR, "                                    if bytes.equal(names_of(table, contents(nar), q), nm) {", "                                    if false {")]),
    M("b-limit-categories", ["unit"], "a column over the stored max_categories is read", [(STR, "                            if fault.code == 0 && catenc.ncat(sr, c) > lim_cats {", "                            if fault.code == 0 && catenc.ncat(sr, c) > lim_cats + 1000 {")]),
    M("b-limit-string-bytes", ["unit"], "a string over the stored max_string_bytes is read", [(STR, "                                if len(s) > lim_str {", "                                if len(s) > lim_str + 1 {")]),
    M("b-totals", ["unit"], "the totals of the header may disagree with the entries", [(STR, "    if sum_pool != pool || sum_strings != strs {", "    if sum_pool != pool && sum_strings != strs {")]),
    M("b-limit-pool", ["unit"], "a pool over the stored max_pool_bytes is read", [(STR, "    if sum_pool > lim_pool || sum_cols > lim_cols {", "    if sum_cols > lim_cols {")]),
    M("b-limit-columns", ["unit"], "more columns than the stored max_cat_columns are read", [(STR, "    if sum_pool > lim_pool || sum_cols > lim_cols {", "    if sum_pool > lim_pool {"), (STR, "                        if sum_cols + catenc.ncols(sr) > lim_cols {", "                        if false {")]),
    M("b-limit-columns-in-the-loop", ["unit"], "the per-entry check of the columns is left to the final one", [(STR, "                        if sum_cols + catenc.ncols(sr) > lim_cols {", "                        if false {")],
      why="equivalent in verdict: the final comparison gives the same reason; the check in the loop exists to stop the quadratic comparison of names before it grows past 4,096 names, and no verdict can tell"),
    M("b-join-step-of-the-entry", ["unit"], "a stub takes the next entry whatever its step", [(STR, "entry_step(table, used) != i + 1 {", "false {")]),
    M("b-join-kind", ["unit"], "a stub may take an entry of another kind", [(STR, "                        if get32(pl, 0) != kd {", "                        if false {")]),
    M("b-join-left-over-entries", ["unit"], "entries without a stub are accepted", [(STR, "(pos != len(section) || used != ne)", "(pos != len(section))")]),
    # ------------------------------------------------------------------ the file (S1, S3, S10)
    M("f-six-sections-routing", ["cli"], "a six-section file is read by the constant model's reader when its model kind is unknown", [(WIO, "est.wired(peek_kind(buffer.bytes(dr))) || six_sections(buffer.bytes(dr));", "est.wired(peek_kind(buffer.bytes(dr)));")]),
    M("f-type3-without-table", ["cli"], "a categorical column of the schema may come without a table", [(FILE, "    if target_cat || ncat > 0 && !has_table {", "    if target_cat {")]),
    M("f-table-without-type3", ["cli"], "a table may come without a categorical column in the schema", [(FILE, "    } else if has_table && ncat == 0 {", "    } else if false {")]),
    M("f-schema-names-against-table", ["cli", "fuzz"], "the names of the table are not compared with the type 3 columns", [(FILE, "                        same = mixed.columns_agree(heap, buffer.bytes(fr), cr);", "                        same = true;")]),
    M("f-plan-against-steps-skipped", ["cli"], "the plan is not compared with the stored steps of a model with a table", [(MIX, "pub fn agrees[&h, &p, &n, &c, &s](heap: &!h Heap, pl: &p plan.Plan, nm: &n names.Names, cats: &c names.Names, payload: &s [byte]) -> [heap] (bool, int) {\n    let ns = plan.steps(pl);", "pub fn agrees[&h, &p, &n, &c, &s](heap: &!h Heap, pl: &p plan.Plan, nm: &n names.Names, cats: &c names.Names, payload: &s [byte]) -> [heap] (bool, int) {\n    if true {\n        return (true, 0);\n    }\n    let ns = plan.steps(pl);")]),
    M("f-plan-missing-mode", ["cli"], "the missing mode of a stored step is not compared with the plan", [(MIX, "    if catenc.missing_mode(st) != missing {\n        return false;\n    }", "    if false {\n        return false;\n    }")]),
    M("f-plan-min-count", ["cli"], "the min-count (or the unknown mode) of a stored step is not compared with the plan", [(MIX, "    } else if catenc.param(st) != plan.min_count_of(pl, i) || catenc.unknown_mode(st) != sp.a {", "    } else if catenc.param(st) != plan.min_count_of(pl, i) && catenc.unknown_mode(st) != sp.a {")]),
    M("f-plan-max-categories", ["cli"], "the category bound of the plan is not compared with the effective categories", [(MIX, "            if kd != kinds.step_hash() && catenc.effective_count(heap, st, k) > sp.b {", "            if false {")]),
    M("f-plan-listed-columns", ["cli"], "a column the plan lists that the step did not read is accepted", [(MIX, "    if !plan.all_columns(pl, i) && k != plan.listed(pl, i) {", "    if false {")]),
    M("f-label-range", ["cli"], "a class label may have any number of digits", [(("lib/models/record.cho"), "    return 999999999999;", "    return 9999999999999;")]),
    # ------------------------------------------------------------------ reading a file with a table (S2, S5, S8)
    M("r-hash-mode-off", ["predict", "cli"], "the reader of a model file looks a hashed column's cells up as strings", [(CSV, "        if cs.kind == 3 && dict.hash_buckets(cs.voc, cj) > 0 {", "        if false {")]),
    M("r-hash-bucket-wrong", ["predict", "unit"], "the bucket adds the halves of the digest without the weight of the high one", [("lib/core/bucket.cho", "    return (hi % b * (4294967296 % b) + lo % b) % b;", "    return (hi % b + lo % b) % b;")]),
    M("r-hash-id-range", ["predict"], "a bucket equal to the number of buckets is a bucket", [(CAT, "                                                    if id >= 0 && id < direct {", "                                                    if id >= 0 && id <= direct {")]),
    M("r-hash-buckets-check", ["predict", "cli"], "a hashed column is accepted by a step of another number of buckets, and refused by the right one", [(CAT, "dict.hash_buckets(rv, rc) != st.param)", "dict.hash_buckets(rv, rc) == st.param)")]),
    M("r-markers-ignored", ["predict", "cli"], "the markers stored in the file are not applied", [(CSV, " && !is_marker(cell, contents(cs.nab)[0..len(contents(cs.nab)) - 1])", "")]),
    M("r-string-limit-ignored", ["cli"], "the string limit of the fit is not applied at predict", [(WIO, "max_str: strtab.limit_of(tbl, 1)", "max_str: 255")]),
    M("r-by-position", ["roundtrip", "cli"], "a column of a step is looked up in the reader's vocabulary by its place in the step", [(CAT, "                let rc = dict.find_column(rv, name(st, j));", "                let rc = j;")]),
    M("r-numeric-by-position", ["align"], "the numeric features are taken from the file by their place", [(WIO, "contents(cw)[f2] = names.find(gq2, schema.name(st.sc, j));", "contents(cw)[f2] = f2;")]),
    M("r-missing-column-is-a-reader-refusal", ["align", "cli"], "a vocabulary column the file lacks is the reader's refusal, not the schema check's", [(CSV, "        if absent && !cc.lenient {", "        if absent {")]),
    M("r-unseen-hash-counted", ["predict"], "the reader of a model file looks up a hashed column through an empty vocabulary", [(CAT, "                                                    if id >= 0 && id < direct {\n                                                        e = id;\n                                                    }", "                                                    if false {\n                                                        e = id;\n                                                    }")]),
    M("r-set-hash-minimum", ["cli", "predict"], "a hashed column of 2 buckets is not marked as hashed", [("lib/core/dict.cho", "if c < 0 || c >= v.ncols || contents(v.counts)[c] != 0 || b < 2 {", "if c < 0 || c >= v.ncols || contents(v.counts)[c] != 0 || b < 3 {")]),
    M("r-decode-duplicate-names", ["unit"], "catenc.decode accepts two columns of one name", [(CAT, "                    dup = true;", "                    dup = false;")]),
    # ------------------------------------------------------------------ privacy (S9)
    M("p-hash-stores-strings", ["privacy"], "a hash step's entry stores the strings the reader met", [(CAT, "                                            while id < vc && kd != kinds.step_hash() {\n                                                let q = contents(cr)[base + id];", "                                            while id < vc {\n                                                let q = contents(cr)[base + id];")]),
    # ------------------------------------------------------------------ the registry (S12)
    M("t-code-reused", ["registry"], "the code of ml.model.string-table is the one of ml.input.reserved-category", [fn_returns("lib/transforms/codes.cho", "string_table", "328", "327")]),
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
    ignore = shutil.ignore_patterns(".git", "build", "__pycache__", "*.pyc", "bench", "*.html", "*.png", "*.svg", "*.jsonl")
    shutil.copytree(ROOT, dest, ignore=ignore, symlinks=True)
    (dest / "build").mkdir(exist_ok=True)
    shutil.copytree(ROOT / "build" / "deps", dest / "build" / "deps")


def lib_files(tree):
    deps = sorted(str(p) for p in (tree / "build" / "deps").glob("*.cho"))
    return ["lib/core/*.cho", "lib/linalg/tensor.cho", "lib/linalg/order.cho", "lib/metrics/sums.cho", "lib/transforms/*.cho", "lib/pipeline/*.cho", "tools/ml/csvcell.cho"] + deps + ["tests/pt.cho"]


def expand(tree, pats):
    out = []
    for p in pats:
        if "*" in p:
            out += sorted(str(q.relative_to(tree)) if not os.path.isabs(p) else str(q) for q in tree.glob(p))
        else:
            out.append(p)
    return out


CLI_MODULES = ["test_stringtable", "test_enc", "test_tables", "test_rules", "test_cat_baseline", "test_model_file", "test_wire_baseline", "test_wire_trees"]


def run_gates(tree, env, gates):
    failures = []
    for g in gates:
        what = GATES[g]
        if what == "unit":
            for prog in ("tests/strtab.cho", "tests/enc.cho"):
                r = subprocess.run([compiler(), "test", "--std"] + expand(tree, lib_files(tree)) + [prog], capture_output=True, text=True, env=env, cwd=tree, timeout=900)
                if r.returncode != 0 or "FAILED" in r.stdout or "error:" in r.stdout + r.stderr:
                    last = [l for l in (r.stdout + r.stderr).splitlines() if "error:" in l or "FAILED" in l or "trapped" in l]
                    failures.append("unit %s: %s" % (prog, (last[0] if last else "?")[:140]))
                    break
            continue
        if what == "cli":
            r = subprocess.run([PY, "-m", "unittest"] + CLI_MODULES, capture_output=True, text=True, env=env, cwd=tree / "tests" / "conformance", timeout=1500)
            if r.returncode != 0:
                failures.append("cli: " + (r.stderr.strip().splitlines() or ["?"])[-1][:140])
            continue
        if what == "registry":
            r = subprocess.run([PY, "scripts/check_registry.py"], capture_output=True, text=True, env=env, cwd=tree)
            if r.returncode != 0:
                failures.append("registry: " + (r.stdout + r.stderr).strip()[:140])
            continue
        r = subprocess.run([PY, str(tree / what[0])] + what[1:], capture_output=True, text=True, env=env, cwd=tree, timeout=2400)
        if r.returncode != 0:
            lines = [l for l in r.stdout.splitlines() if "DIFF" in l or "differences" in l]
            failures.append("%s: %s" % (g, (lines[-1] if lines else r.stdout[-200:] + r.stderr[-200:]).strip()[:160]))
    return failures


def build_and_gate(m, gates):
    with tempfile.TemporaryDirectory(prefix="ml-st-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        tree_copy(tree)
        if m:
            for file, old, new in m["edits"]:
                path = tree / file
                text = path.read_text()
                if text.count(old) != 1:
                    return "INVALID", "pattern does not apply to the copy"
                path.write_text(text.replace(old, new, 1))
        env = dict(os.environ, CANCHO=compiler(), ST_FAST="1", ST_STRIDE="11")
        need_ml = any(g in ("diff", "order", "roundtrip", "predict", "align", "privacy", "limits", "fuzz", "cli") for g in gates)
        if need_ml:
            b = subprocess.run([compiler(), "build", "--bin", "ml"], capture_output=True, text=True, env=env, cwd=tree)
            if b.returncode != 0:
                return "INVALID", "ml build failed: " + (b.stdout + b.stderr)[-300:]
            env["ML"] = str(tree / "build" / "ml")
            env["ML_BIN"] = str(tree / "build" / "ml")
        failures = run_gates(tree, env, gates)
        return ("KILLED", "; ".join(failures)) if failures else ("SURVIVED", "")


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        for m in MUTANTS:
            print("%-40s %-22s %s" % (m["id"], ",".join(m["gates"]), m["what"]))
        return 0
    bad = []
    for m in MUTANTS:
        bad += check_pattern(m)
    ids = [m["id"] for m in MUTANTS]
    if len(ids) != len(set(ids)):
        bad.append("an id twice")
    if bad:
        print("\n".join(bad))
        return 1
    if "--check" in argv:
        print("%d string-table mutants, every pattern applies once" % len(MUTANTS))
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
            print("%-40s KILLED   (%s) %s" % (m["id"], ",".join(m["gates"]), detail[:120]), flush=True)
        elif verdict == "SURVIVED" and m["why"]:
            explained += 1
            print("%-40s SURVIVED, explained: %s" % (m["id"], m["why"][:140]), flush=True)
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

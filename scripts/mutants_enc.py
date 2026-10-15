#!/usr/bin/env python3
"""Mutants for the gates of docs/categorical-encoders.md section 1: a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change (a few have two or three edits) to a source of this repository; what the mutant touches is rebuilt in the copy
(`enc-check`, the unit program, `pipe-check`, `ml`) and the gates named for the mutant run against it. KILLED: a gate reported differences, a trap or a failed test;
SURVIVED: none did, and the mutant then carries a reason (an equivalent mutant) or is a gap that docs/categorical-encoders.md section 9 names.

    python3 scripts/mutants_enc.py              # control, then all mutants
    python3 scripts/mutants_enc.py --only ID    # one
    python3 scripts/mutants_enc.py --list
    python3 scripts/mutants_enc.py --check      # every `old` text is found exactly once (no build, no run)

The gates run with reduced sizes (ENC_FAST) so that forty rebuilds fit; a survivor is re-run at full size by hand. The mutant `e2-first-appearance` is judged by the
row-order gate ALONE: the issue requires that a vocabulary ordered by first appearance fail it.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable
CAT = "lib/transforms/catenc.cho"


def compiler():
    here = ROOT.parent / "cancho-compiler-0207556" / "target" / "release" / "cancho"
    return os.environ.get("CANCHO") or (str(here) if here.exists() else "cancho")


def M(id, gates, what, edits, why=None):
    return dict(id=id, gates=gates, what=what, edits=edits, why=why)


# gate name -> what it needs and runs
GATES = {
    "unit": "unit",
    "diff": ["scripts/gate_enc.py", "diff", "--files", "150"],
    "order": ["scripts/gate_enc.py", "order", "--files", "40"],
    "hash": ["scripts/gate_enc.py", "hash"],
    "threads": ["scripts/gate_enc.py", "threads"],
    "fuzz": ["scripts/gate_enc.py", "fuzz", "--runs", "150"],
    "plan": "plan",
    "cli": "cli",
    "registry": "registry",
}

FIT_COUNT = """                            if contents(cw)[base + id] == 0 {
                                seen = seen + 1;
                                pool_bytes = pool_bytes + len(dict.string(rv, c, id));
                            }"""
FIT_COUNT_STAMP = """                            if contents(cw)[base + id] == 0 {
                                seen = seen + 1;
                                pool_bytes = pool_bytes + len(dict.string(rv, c, id));
                                contents(cw)[base + id] = seen * 4294967296;
                            }"""
FIT_BUILD = """                                            var id = 0;
                                            while id < vc && kd != kinds.step_hash() {
                                                let q = contents(cr)[base + id];
                                                if q > 0 {"""
FIT_BUILD_FIRST = """                                            var id = 0;
                                            var kk = 0;
                                            while kk < vc && kd != kinds.step_hash() {
                                                id = 0;
                                                while id < vc && contents(cr)[base + id] / 4294967296 != kk + 1 {
                                                    id = id + 1;
                                                }
                                                var q = 0;
                                                if id < vc {
                                                    q = contents(cr)[base + id] % 4294967296;
                                                }
                                                if q > 0 {"""
FIT_BUILD_TAIL = """                                                id = id + 1;
                                            }
                                            base = base + vc;"""
FIT_BUILD_TAIL_FIRST = """                                                kk = kk + 1;
                                            }
                                            base = base + vc;"""
APPLY_MID = """                                                    let mid = (lo + hi) / 2;
                                                    let cmp = bytes.compare(string(st, j, mid), sv);"""
APPLY_MID_LINEAR = """                                                    let mid = lo;
                                                    let cmp = bytes.compare(string(st, j, mid), sv);"""
APPLY_BRANCH = """                                                    } else if cmp < 0 {
                                                        lo = mid + 1;
                                                    } else {
                                                        hi = mid;
                                                    }"""
APPLY_BRANCH_LINEAR = """                                                    } else {
                                                        lo = mid + 1;
                                                    }"""
FIRST_APPEARANCE = [(CAT, FIT_COUNT, FIT_COUNT_STAMP), (CAT, FIT_BUILD, FIT_BUILD_FIRST), (CAT, FIT_BUILD_TAIL, FIT_BUILD_TAIL_FIRST), (CAT, APPLY_MID, APPLY_MID_LINEAR), (CAT, APPLY_BRANCH, APPLY_BRANCH_LINEAR)]

MUTANTS = [
    # ------------------------------------------------------------------ E2: the order of the vocabulary and of the rows
    M("e2-first-appearance", ["order"], "the fitted vocabulary keeps the strings in the order the rows meet them (the lookup is a linear search, so every cell is still found): the payload and the label and one-hot places follow the row order",
      FIRST_APPEARANCE),
    M("e1-first-appearance-diff", ["diff"], "the same mutant, judged by the oracle", FIRST_APPEARANCE),
    M("e2-descending-vocabulary", ["diff", "unit"], "the fitted vocabulary is written from the largest string to the smallest",
      [(CAT, "                                            var id = 0;\n                                            while id < vc && kd != kinds.step_hash() {\n                                                let q = contents(cr)[base + id];",
        "                                            var id = 0;\n                                            while id < vc && kd != kinds.step_hash() {\n                                                let q = contents(cr)[base + vc - 1 - id];")]),
    M("e2-label-from-the-row", ["order", "diff"], "a label is the effective index plus the parity of its row",
      [(CAT, "matrix.put(contents(dw), ocols, r, o, float_of(e));", "matrix.put(contents(dw), ocols, r, o, float_of(e + r % 2));")]),
    # ------------------------------------------------------------------ E5: the rare bucket
    M("e5-threshold-le", ["diff", "unit"], "a category with a count AT the threshold is merged (<= for <)",
      [(CAT, "        if n < min_count {\n            merged = merged + n;", "        if n <= min_count {\n            merged = merged + n;"),
       (CAT, "                if n < min_count {\n                    contents(iw)[i] = below;", "                if n <= min_count {\n                    contents(iw)[i] = below;")]),
    M("e5-threshold-plus-one", ["diff", "unit"], "the threshold is min-count + 1",
      [(CAT, "        if n < min_count {\n            merged = merged + n;", "        if n < min_count + 1 {\n            merged = merged + n;"),
       (CAT, "                if n < min_count {\n                    contents(iw)[i] = below;", "                if n < min_count + 1 {\n                    contents(iw)[i] = below;")]),
    M("e5-threshold-minus-one", ["diff", "unit"], "the threshold is min-count - 1",
      [(CAT, "        if n < min_count {\n            merged = merged + n;", "        if n < min_count - 1 {\n            merged = merged + n;"),
       (CAT, "                if n < min_count {\n                    contents(iw)[i] = below;", "                if n < min_count - 1 {\n                    contents(iw)[i] = below;")]),
    M("e5-bucket-last", ["diff", "unit"], "the bucket is placed after every kept string, not in its bytewise place",
      [(CAT, "            if bytes.compare(string(st, c, i), other()) < 0 {\n                below = below + 1;", "            if true {\n                below = below + 1;")]),
    M("e5-bucket-first", ["diff", "unit"], "the bucket is placed before every kept string",
      [(CAT, "            if bytes.compare(string(st, c, i), other()) < 0 {\n                below = below + 1;", "            if false {\n                below = below + 1;")]),
    M("e5-collision-unchecked", ["diff", "unit"], "a real category named __other__ is not refused at fit",
      [(CAT, "                if par >= 2 {\n                    var i2 = 0;", "                if par >= 2 && false {\n                    var i2 = 0;")]),
    M("e5-merged-counts-not-added", ["diff", "unit"], "the bucket's count is the number of merged categories, not of rows",
      [(CAT, "            merged = merged + n;", "            merged = merged + 1;")]),
    M("e5-bucket-counted-as-unseen", ["diff", "unit"], "a merged category is mapped to no category at all (unseen)",
      [(CAT, "                if n < min_count {\n                    contents(iw)[i] = below;", "                if n < min_count {\n                    contents(iw)[i] = 0 - 1;")]),
    M("e5-bound-before-the-bucket", ["diff", "unit"], "max-categories is compared with the number of strings, not of effective categories",
      [(CAT, "                    if ne > maxcat {", "                    if ncat(sr, j) > maxcat {")]),
    # ------------------------------------------------------------------ E1: one-hot, label, frequency
    M("e1-label-from-one", ["diff", "unit"], "labels start at 1",
      [(CAT, "matrix.put(contents(dw), ocols, r, o, float_of(e));", "matrix.put(contents(dw), ocols, r, o, float_of(e + 1));")]),
    M("e1-onehot-reversed", ["diff", "unit"], "the one-hot block is written from the right",
      [(CAT, "matrix.put(contents(dw), ocols, r, o + e, 1.0);", "matrix.put(contents(dw), ocols, r, o + width - 1 - e, 1.0);")]),
    M("e1-freq-reciprocal", ["diff", "unit"], "a frequency is the count times the rounded reciprocal of the rows (two roundings)",
      [(CAT, "float_of(contents(cr)[e]) / float_of(st.rows)", "float_of(contents(cr)[e]) * (1.0 / float_of(st.rows))")]),
    M("e1-freq-denominator", ["diff", "unit"], "the denominator of a frequency is the rows plus one",
      [(CAT, "float_of(contents(cr)[e]) / float_of(st.rows)", "float_of(contents(cr)[e]) / float_of(st.rows + 1)")]),
    M("e1-freq-of-the-merged-category", ["diff", "unit"], "a merged category keeps its own count instead of the bucket's",
      [(CAT, "            if has_other {\n                contents(ew)[below] = merged;\n            }", "            if has_other {\n                contents(ew)[below] = 1;\n            }")]),
    M("e1-column-order", ["diff", "unit"], "the columns of one step are written from the last to the first",
      [(CAT, "                                let rc = contents(rr)[j];\n                                let vc = dict.count(rv, rc);", "                                let rc = contents(rr)[k - 1 - j];\n                                let vc = dict.count(rv, rc);")]),
    M("e1-names-not-sorted-into-the-steps", ["diff"], "the appended columns go before the columns the step receives",
      [("lib/pipeline/mixed.cho", "let (z, zc) = hstack(heap, curr, yr, rows_cap, cols_cap, cells_cap);\n                                    if zc != 0 {\n                                        fault = codes.fault(codes.from_refuse(zc), s + 1, 0, 0);",
        "let (z, zc) = hstack(heap, yr, curr, rows_cap, cols_cap, cells_cap);\n                                    if zc != 0 {\n                                        fault = codes.fault(codes.from_refuse(zc), s + 1, 0, 0);")]),
    # ------------------------------------------------------------------ E4: unknown and missing
    M("e4-unknown-never-refused", ["diff", "unit"], "unknown=refuse is ignored",
      [(CAT, "if st.unknown == 1 {", "if false {")]),
    M("e4-unknown-always-refused", ["diff", "unit"], "unknown=ignore is a refusal",
      [(CAT, "if st.unknown == 1 {", "if true {")]),
    M("e4-unseen-not-counted", ["diff", "unit"], "an unseen cell is not counted",
      [(CAT, "unknown = unknown + 1;", "unknown = unknown;")]),
    M("e4-missing-is-unseen", ["diff", "unit"], "a missing cell is taken for an unseen string",
      [(CAT, "                                            if id == 0 - 1 {", "                                            if id == 0 - 99 {")]),
    M("e4-missing-zero-ignored", ["diff", "unit"], "missing=zero leaves the cells missing",
      [(CAT, "                                                if st.missing == 0 {", "                                                if true {")]),
    M("e4-label-unseen-is-zero", ["diff", "unit"], "an unseen label is 0.0",
      [(CAT, "                                                        if kd == kinds.step_cat_label() {\n                                                            matrix.put(contents(dw), ocols, r, o, 0.0 - 1.0);",
        "                                                        if kd == kinds.step_cat_label() {\n                                                            matrix.put(contents(dw), ocols, r, o, 0.0);")]),
    M("e4-unseen-mapped-to-the-nearest", ["diff", "unit"], "an unseen string takes the last category before it",
      [(CAT, "                                                if found >= 0 {\n                                                    contents(tv)[id] = contents(er)[found];", "                                                if found < 0 && lo > 0 {\n                                                    contents(tv)[id] = contents(er)[lo - 1];\n                                                }\n                                                if found >= 0 {\n                                                    contents(tv)[id] = contents(er)[found];")]),
    M("e4-reader-unseen-id-is-a-category", ["diff"], "a string the reader could not find (id -2) is looked up in the last place of the table",
      [(CAT, "                                                } else if id >= 0 && id < vc {\n                                                    e = contents(tr)[id];", "                                                } else if id >= 0 - 2 && id < vc && id != 0 - 1 {\n                                                    e = contents(tr)[(id + vc) % vc];")]),
    # ------------------------------------------------------------------ E6: the hash
    M("e6-hash-little-endian", ["hash", "diff", "unit"], "the first 8 bytes of the digest are read little-endian",
      [("lib/core/bucket.cho", "    let hi = int_of(dg[0]) << 24 | int_of(dg[1]) << 16 | int_of(dg[2]) << 8 | int_of(dg[3]);\n    let lo = int_of(dg[4]) << 24 | int_of(dg[5]) << 16 | int_of(dg[6]) << 8 | int_of(dg[7]);",
        "    let hi = int_of(dg[7]) << 24 | int_of(dg[6]) << 16 | int_of(dg[5]) << 8 | int_of(dg[4]);\n    let lo = int_of(dg[3]) << 24 | int_of(dg[2]) << 16 | int_of(dg[1]) << 8 | int_of(dg[0]);")]),
    M("e6-hash-top-bit-dropped", ["hash", "diff", "unit"], "the sign bit of the 64-bit value is dropped (a signed reading)",
      [("lib/core/bucket.cho", "    return (hi % b * (4294967296 % b) + lo % b) % b;", "    return ((hi & 2147483647) % b * (4294967296 % b) + lo % b) % b;")]),
    M("e6-hash-first-four-bytes", ["hash", "diff", "unit"], "only the first 4 bytes of the digest are used",
      [("lib/core/bucket.cho", "    return (hi % b * (4294967296 % b) + lo % b) % b;", "    return hi % b;")]),
    M("e6-hash-modulus-off-by-one", ["hash", "diff", "unit"], "the modulus is B - 1",
      [("lib/core/bucket.cho", "    return (hi % b * (4294967296 % b) + lo % b) % b;", "    return (hi % b * (4294967296 % b) + lo % b) % (b - 1);")]),
    M("e6-hash-last-eight-bytes", ["hash", "diff", "unit"], "the last 8 bytes of the digest",
      [("lib/core/bucket.cho", "    let hi = int_of(dg[0]) << 24 | int_of(dg[1]) << 16 | int_of(dg[2]) << 8 | int_of(dg[3]);\n    let lo = int_of(dg[4]) << 24 | int_of(dg[5]) << 16 | int_of(dg[6]) << 8 | int_of(dg[7]);",
        "    let hi = int_of(dg[24]) << 24 | int_of(dg[25]) << 16 | int_of(dg[26]) << 8 | int_of(dg[27]);\n    let lo = int_of(dg[28]) << 24 | int_of(dg[29]) << 16 | int_of(dg[30]) << 8 | int_of(dg[31]);")]),
    M("e6-hash-of-the-id", ["hash", "diff"], "the bucket is taken from the reader's id, not from the bytes",
      [(CAT, "contents(tv)[id] = bucket_of(contents(sx), contents(dx), dict.string(rv, rc, id), st.param);", "contents(tv)[id] = id % st.param;")]),
    # ------------------------------------------------------------------ E7: the payload
    M("e7-decode-accepts-unsorted", ["unit"], "decode does not check that the strings ascend",
      [(CAT, "            if prev_len >= 0 && bytes.compare(data[prev_from..prev_from + prev_len], data[pos..pos + sl]) >= 0 {", "            if prev_len >= 0 && false {")]),
    M("e7-decode-accepts-counts-over-rows", ["unit"], "decode does not compare the counts with the rows",
      [(CAT, "            if total > rows {\n                return (empty(heap), bad);\n            }", "")]),
    M("e7-decode-accepts-zero-count", ["unit"], "decode accepts a count of 0",
      [(CAT, "            if cn < 1 || cn > rows {", "            if cn < 0 || cn > rows {")]),
    M("e7-decode-trailing-bytes", ["unit"], "decode accepts bytes after the last category",
      [(CAT, "    if pos != n {\n        return (empty(heap), bad);\n    }\n    // the names are distinct", "    // the names are distinct")]),
    M("e7-payload-field-order", ["diff", "unit"], "the unknown and missing words are written the other way round",
      [(CAT, "    o = put32(heap, o, st.unknown);\n    o = put32(heap, o, st.missing);", "    o = put32(heap, o, st.missing);\n    o = put32(heap, o, st.unknown);")]),
    M("e7-decode-name-length", ["unit"], "decode accepts a name of 256 bytes",
      [(CAT, "        if nl < 1 || nl > max_name() || n - pos < nl + 4 {", "        if nl < 1 || nl > 100000 || n - pos < nl + 4 {")]),
    # ------------------------------------------------------------------ E8: the plan and the columns
    M("e8-default-written", ["plan", "cli"], "min-count=1 is written into the canonical text of a numeric encode line",
      [("lib/pipeline/plan.cho", "min-count:int=1,1000000:~1 missing:enum=pass,zero:~pass", "min-count:int=1,1000000:=1 missing:enum=pass,zero:~pass")]),
    M("e8-explicit-default-kept", ["plan", "cli"], "an explicit default of an omitted-when-default key is written back",
      [("lib/pipeline/plan.cho", "                var differs = true;\n                borrow s2 as &s2r in {\n                    differs = !bytes.equal(buffer.bytes(s2r), rule[1..len(rule)]);", "                var differs = true;\n                borrow s2 as &s2r in {\n                    differs = true;")]),
    M("e8-star-encode-reaches-categorical", ["cli", "diff"], "encode column=* is the categorical columns",
      [("lib/pipeline/mixed.cho", "        var want = fixed;\n        if want == 0 {\n            want = 1;\n        }", "        var want = fixed;\n        if want == 0 {\n            want = 2;\n        }")]),
    M("e8-kind-check-removed", ["cli", "diff"], "a numeric step on a categorical column is not refused",
      [("lib/pipeline/mixed.cho", "                    if cl != required {\n                        fault = column_kind(step, t + 1, required);", "                    if cl != cl + 1 {\n                        fault = column_kind(step, t + 1, required);")]),
    M("e8-numeric-first-lookup-reversed", ["diff"], "a name is looked up among the categorical columns first",
      [("lib/pipeline/mixed.cho", "                var cl = 1;\n                var at = names.find(cn, nm);\n                if at < 0 {\n                    cl = 2;\n                    at = find_cat(rv, alive, nm);\n                }",
        "                var cl = 2;\n                var at = find_cat(rv, alive, nm);\n                if at < 0 {\n                    cl = 1;\n                    at = names.find(cn, nm);\n                }")]),
    M("e8-column-stays-alive", ["cli", "diff"], "an encoded categorical column can be encoded again",
      [("lib/pipeline/mixed.cho", "                        contents(aw)[contents(cr)[j]] = 0;", "                        contents(aw)[contents(cr)[j]] = 1;")]),
    M("e8-unread-columns-silent", ["cli"], "a declared column no step reads is dropped without a word",
      [("tools/ml/wire.cho", "                                    if names.count(remr) > 0 {", "                                    if names.count(remr) > 100000 {")]),
    M("e8-min-count-on-numeric-accepted", ["cli"], "min-count on a numeric encode is accepted",
      [("lib/pipeline/plan.cho", "contents(p.specs)[8 * i] >= 9 || contents(p.specs)[8 * i + 6] > 1", "contents(p.specs)[8 * i] >= 9")]),
    # ------------------------------------------------------------------ E9: no file before stage 3
    M("e9-file-not-made", ["cli"], "a categorical fit makes no file bytes (the answer says it wrote an empty file)",
      [("tools/ml/wire.cho", "    if ds.cat {\n        // the schema of every column the model reads", "    if false {\n        // the schema of every column the model reads")]),
    M("e9-dry-run-writes", ["cli"], "a dry run of a categorical fit is told to write",
      [("tools/ml/wire.cho", "        let (ws2, wa2) = fsio.write_atomic(heap, fs, root, paths.shown(out), paths.full(out), data, o.overwrite, o.dry);", "        let (ws2, wa2) = fsio.write_atomic(heap, fs, root, paths.shown(out), paths.full(out), data, o.overwrite, false);")]),
    M("e9-split-trained-on-all-rows", ["cli"], "the vocabulary of a split fit is built from all the rows",
      [("tools/ml/wire.cho", "catwire.take_ids(heap, cids, sel, want, o.max_rows, o.max_cols, o.max_cells);\n            if icode != 0 {", "catwire.take_ids(heap, cids, sel, 0 - 1, o.max_rows, o.max_cols, o.max_cells);\n            if icode != 0 {")]),
    M("e9-holdout-uses-train-ids", ["cli"], "the held-out rows are scored on the ids of the training rows",
      [("tools/ml/wire.cho", "let (idt, icode) = catwire.take_ids(heap, cids, sel, 1, o.max_rows, o.max_cols, o.max_cells);", "let (idt, icode) = catwire.take_ids(heap, cids, sel, 0, o.max_rows, o.max_cols, o.max_cells);")]),
    # ------------------------------------------------------------------ E3: the thread count
    M("e3-threads-in-the-answer", ["threads"], "the thread count reaches the summary",
      [("tools/ml/wire.cho", "        w = catwire.put_categorical(heap, w, buffer.bytes(tr.fitted.payload));", "        w = catwire.put_categorical(heap, w, buffer.bytes(tr.fitted.payload));\n        w = json.put_key(heap, w, \"threads\");\n        w = json.put_int(heap, w, o.threads);")]),
    # ------------------------------------------------------------------ E10, E12: numeric inputs and tables
    M("e10-step-kind-moved", ["registry"], "step kind 7 is claimed twice",
      [("lib/core/kinds.cho", "pub fn step_cat_label() -> [] int {\n    return 8;", "pub fn step_cat_label() -> [] int {\n    return 7;")]),
    M("e12-tag-without-fixture", ["cli"], "a rule tag is in the table and no fixture reaches it",
      [("tests/conformance/fixtures.py", 'bad("ml.input.reserved-category")', 'bad("ml.input.unknown-category")')]),
    # ------------------------------------------------------------------ E11: the bounds
    M("e11-categories-bound-removed-in-decode", ["unit"], "decode accepts more categories than the ceiling",
      [(CAT, "(m < 1 || m > max_categories())", "(m < 1 || m > 100000000)")]),
    M("e11-name-lookup-unchecked", ["unit"], "a reader column index past the ids is not checked at fit",
      [(CAT, "        if sel[j] <= prev || sel[j] >= dict.ncols(rv) || sel[j] >= catmat.cols(ids) {", "        if sel[j] <= prev {")]),
]

# Equivalent or not detectable by these gates (each with its reason); filled in after the first full run
WHY = {
    "e8-numeric-first-lookup-reversed": "equivalent: a name is both a numeric column and a not-yet-encoded categorical one only if a step wrote a numeric column with the name of another declared column, and the "
                                        "encoders write NAME (a column that is then no longer categorical) or NAME#i, and a plan cannot name a column that holds `#` (docs/pipeline.md section 2), so the order of the two "
                                        "lookups is unobservable",
}
for m in MUTANTS:
    m["why"] = m["why"] or WHY.get(m["id"])


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
    # the documents the registry gate reads are copied (docs/*.md); the logs and the pages are not
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


def run_gates(tree, env, gates):
    failures = []
    for g in gates:
        what = GATES[g]
        if what == "unit":
            r = subprocess.run([compiler(), "test", "--std"] + expand(tree, lib_files(tree)) + ["tests/enc.cho"], capture_output=True, text=True, env=env, cwd=tree, timeout=900)
            if r.returncode != 0 or "FAILED" in r.stdout or "error:" in r.stdout + r.stderr:
                last = [l for l in (r.stdout + r.stderr).splitlines() if "error:" in l or "FAILED" in l or "trapped" in l]
                failures.append("unit: " + (last[0] if last else "?")[:140])
            continue
        if what == "plan":
            b = subprocess.run([compiler(), "build", "--std"] + expand(tree, lib_files(tree)) + ["tools/pipe-check/main.cho", "-o", str(tree / "build" / "pipe-check")], capture_output=True, text=True, env=env, cwd=tree)
            if b.returncode != 0:
                return ["plan: pipe-check does not build"]
            e2 = dict(env, PIPE_CHECK=str(tree / "build" / "pipe-check"), PIPE_PLANS="250")
            r = subprocess.run([PY, "scripts/oracle_pipe.py", "--plans"], capture_output=True, text=True, env=e2, cwd=tree, timeout=900)
            if r.returncode != 0:
                failures.append("plan: " + ((r.stdout + r.stderr).strip().splitlines() or ["?"])[-1][:140])
            continue
        if what == "cli":
            r = subprocess.run([PY, "-m", "unittest", "test_enc", "test_tables"], capture_output=True, text=True, env=env, cwd=tree / "tests" / "conformance", timeout=900)
            if r.returncode != 0:
                failures.append("cli: " + (r.stderr.strip().splitlines() or ["?"])[-1][:140])
            continue
        if what == "registry":
            r = subprocess.run([PY, "scripts/check_registry.py"], capture_output=True, text=True, env=env, cwd=tree)
            if r.returncode != 0:
                failures.append("registry: " + (r.stdout + r.stderr).strip()[:140])
            continue
        r = subprocess.run([PY, str(tree / what[0])] + what[1:], capture_output=True, text=True, env=env, cwd=tree, timeout=1800)
        if r.returncode != 0:
            lines = [l for l in r.stdout.splitlines() if "DIFF" in l or "differences" in l]
            failures.append("%s: %s" % (g, (lines[-1] if lines else r.stdout[-200:] + r.stderr[-200:]).strip()[:160]))
    return failures


def build_and_gate(m, gates):
    with tempfile.TemporaryDirectory(prefix="ml-enc-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        tree_copy(tree)
        if m:
            for file, old, new in m["edits"]:
                path = tree / file
                text = path.read_text()
                if text.count(old) != 1:
                    return "INVALID", "pattern does not apply to the copy"
                path.write_text(text.replace(old, new, 1))
        env = dict(os.environ, CANCHO=compiler(), ENC_FAST="1")
        need_enc = any(g in ("diff", "order", "hash", "threads", "fuzz") for g in gates)
        need_ml = any(g in ("threads", "fuzz", "cli") for g in gates)
        if need_enc:
            b = subprocess.run(["sh", str(tree / "scripts" / "build_enc.sh"), str(tree / "build" / "enc-check")], capture_output=True, text=True, env=env, cwd=tree)
            if b.returncode != 0:
                return "INVALID", "enc-check build failed: " + (b.stdout + b.stderr)[:300]
            env["ENC"] = str(tree / "build" / "enc-check")
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
            print("%-44s %-22s %s" % (m["id"], ",".join(m["gates"]), m["what"]))
        return 0
    bad = []
    for m in MUTANTS:
        bad += check_pattern(m)
    if bad:
        print("\n".join(bad))
        return 1
    if "--check" in argv:
        print("%d categorical-encoder mutants, every pattern applies once" % len(MUTANTS))
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

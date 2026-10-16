#!/usr/bin/env python3
"""Mutants for the gates of docs/tracking.md (T12): run tracking and the skew checks. A gate that cannot fail judges nothing.

As scripts/mutants_serve.py: each mutant is a copy of the tree with one change, `ml` rebuilt from the copy, and the tests of the gate it targets run against it. KILLED means a test failed;
SURVIVED means every test passed, and the mutant must then carry a reason (`why`).

    python3 scripts/mutants_track.py              # all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_track.py --only ID    # one
    python3 scripts/mutants_track.py --list
    python3 scripts/mutants_track.py --check      # every `old` text is found exactly where it is expected (no build, no test)

Run it on a frozen copy of the tree when `cancho install` may run in the original (it rewrites build/deps).
"""

import glob
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_cli as base  # noqa: E402

ROOT = base.ROOT
M = base.M

PROV = "tools/ml/prov.cho"
TRK = "tools/ml/track.cho"
PS = "lib/pipeline/provsec.cho"
FILE = "lib/pipeline/file.cho"
MFILE = "tools/ml/mfile.cho"
WIO = "tools/ml/wireio.cho"
WIRE = "tools/ml/wire.cho"
RUNS = "tools/ml/runs.cho"
SPLS = "tools/ml/splitspec.cho"
FSIO = "tools/ml/fsio.cho"
MDESC = "tools/ml/mdesc.cho"
SECTION = ["test_track.Section", "test_track.Damage"]
RANGE = ["test_track.Range", "test_track.Features"]
TRACK = ["test_track.Tracking"]

NAIVE_MEAN = """var nsum = 0.0;
                        var nq = 0;
                        while nq < n {
                            if (want < 0 || sel[nq] == want) && !matrix.is_missing(x, nq, j) {
                                nsum = nsum + matrix.value(x, nq, j);
                            }
                            nq = nq + 1;
                        }
                        let mean_bits = norm_bits(nsum / float_of(count));"""

MUTANTS = [
    # ------------------------------------------------------------------ the statistics and the section (T1)
    M("t-naive-sum", "T1", "the mean is a float sum in row order, not the exact sum rounded once", [(PROV, "let mean_bits = csvcell.bits_of_key(facc.mean_key(av, 0, count));", NAIVE_MEAN, 1)], SECTION + ["test_track.Replay"]),
    M("t-std-about-zero", "T1", "the deviation is taken about zero (the squares of the cells)", [(PROV, "fsq.add_square(contents(qw), bits_of(matrix.value(x, r, j)), mean_bits);", "fsq.add_square(contents(qw), bits_of(matrix.value(x, r, j)), 0);", 1)], SECTION),
    M("t-std-sample", "T1", "the deviation divides by n - 1", [(PROV, "sqrt(sse / float_of(count))", "sqrt(sse / float_of(count - 1))", 1)], SECTION),
    M("t-min-of-the-first-cell-only", "T1", "the minimum is the first cell", [(PROV, "if count == 0 || v < mn {\n                                mn = v;", "if count == 0 {\n                                mn = v;", 1)], SECTION),
    M("t-missing-cell-in-the-statistics", "T1", "a missing cell counts as 0.0 in the statistics", [(PROV, "if (want < 0 || sel[r] == want) && !matrix.is_missing(x, r, j) {\n                            let v = matrix.value(x, r, j);", "if (want < 0 || sel[r] == want) {\n                            let v = matrix.value(x, r, j);", 1)], SECTION),
    M("t-all-rows-in-the-statistics", "T1", "the statistics cover the held-out rows too", [(WIRE, "let (st, bad) = prov.stats(heap, dr.x, contents(sr), want, contents(fr), nf);", "let (st, bad) = prov.stats(heap, dr.x, contents(sr), 0 - 1, contents(fr), nf);", 1)], SECTION),
    M("t-hash-of-a-prefix", "T1", "the SHA-256 of the training file leaves out the last byte of the last chunk", [(FSIO, "hasher = sha.update(hasher, contents(cw)[0..n]);", "hasher = sha.update(hasher, contents(cw)[0..n - 1]);", 1)], SECTION),
    M("t-size-of-the-last-chunk", "T1", "the size of the training file is that of its last read", [(FSIO, "size = size + n;", "size = n;", 1)], SECTION),
    M("t-rows-of-the-training-side", "T1", "the section's rows is the rows of the file", [(WIRE, "pv = provsec.pack(heap, size1, buffer.bytes(hr), dr.nrows, tq.rows, nf, contents(str));", "pv = provsec.pack(heap, size1, buffer.bytes(hr), dr.nrows, dr.nrows, nf, contents(str));", 1)], SECTION),
    # ------------------------------------------------------------------ the decoder (T9)
    M("t-decoder-skips-the-length", "T9", "the length of the payload is not compared with the count it states", [(PS, " || len(p) != header_len() + record_len() * nf {", " {", 1)], ["test_track.Damage"]),
    M("t-decoder-skips-the-count-cap", "T9", "the count of features is not capped at 8192", [(PS, "if nf < 0 || nf > max_features() ||", "if nf < 0 ||", 1)], ["test_track.Damage"],
      why="redundant with the equal length and the equality with the schema's feature count (at most 8192 columns): a payload past the cap fails one of them whatever the cap says"),
    M("t-decoder-skips-the-layout", "T9", "the layout number is not checked", [(PS, "    if get32(p, 0) != layout() {\n        return 1;\n    }\n", "", 1)], ["test_track.Damage"]),
    M("t-decoder-skips-file-bytes", "T9", "file_bytes is not bounded", [(PS, "if fb < 0 || fb > max_file_bytes() ||", "if", 1)], ["test_track.Damage"]),
    M("t-decoder-skips-file-rows", "T9", "file_rows is not bounded", [(PS, "fr < 1 || fr > max_rows() ||", "", 1)], ["test_track.Damage"]),
    M("t-decoder-skips-rows-below-file-rows", "T9", "rows may be above file_rows", [(PS, "r < 1 || r > fr || r != rows_meta", "r < 1 || r != rows_meta", 1)], ["test_track.Damage"]),
    M("t-decoder-skips-rows-of-meta", "T9", "rows is not compared with the rows of meta", [(PS, "r < 1 || r > fr || r != rows_meta ||", "r < 1 || r > fr ||", 1)], ["test_track.Damage"]),
    M("t-decoder-skips-the-schema-features", "T9", "the count of features is not compared with the schema's", [(PS, "|| nf != numeric_features {", "{", 1)], ["test_track.Damage"]),
    M("t-decoder-skips-count-below-rows", "T9", "a count may be above the rows", [(PS, "if count < 0 || count > r {", "if count < 0 {", 1)], ["test_track.Damage"]),
    M("t-decoder-skips-the-empty-column-rule", "T9", "a column with no cell may hold a range", [(PS, "if mn != 0 || mx != 0 || me != 0 || sd != 0 {\n                return 1;\n            }", "", 1)], ["test_track.Damage"]),
    M("t-decoder-skips-finite", "T9", "an infinite or NaN statistic is accepted", [(PS, "if !finite(mn) || !finite(mx) || !finite(me) || !finite(sd) || negative_zero(mn)", "if negative_zero(mn)", 1)], ["test_track.Damage"]),
    M("t-decoder-skips-negative-zero", "T9", "a negative zero is accepted", [(PS, "|| negative_zero(mn) || negative_zero(mx) || negative_zero(me) || sd < 0 {", "|| sd < 0 {", 1)], ["test_track.Damage"]),
    M("t-decoder-skips-the-std-sign", "T9", "a negative deviation is accepted", [(PS, "|| negative_zero(me) || sd < 0 {", "|| negative_zero(me) {", 1)], ["test_track.Damage"]),
    M("t-decoder-skips-min-below-max", "T9", "a minimum above the maximum is accepted", [(PS, "if fmn > fmx || fme < fmn || fme > fmx {", "if fme < fmn || fme > fmx {", 1)], ["test_track.Damage"],
      why="equivalent: a mean not below the minimum and not above the maximum implies minimum <= maximum, so the first term never decides"),
    M("t-decoder-skips-mean-in-range", "T9", "a mean outside [min, max] is accepted", [(PS, "if fmn > fmx || fme < fmn || fme > fmx {", "if fmn > fmx {", 1)], ["test_track.Damage"]),
    M("t-decoder-skips-the-constant-rule", "T9", "a constant column may have a deviation or another mean", [(PS, "if fmn == fmx && (me != mn || sd != 0) {\n                return 1;\n            }", "", 1)], ["test_track.Damage"]),
    M("t-structure-skips-the-order", "T9", "a section may repeat or come out of order", [(FILE, "if tag <= prev || tag != kinds.section_strings() && tag != kinds.section_provenance() {", "if tag != kinds.section_strings() && tag != kinds.section_provenance() {", 1)], ["test_track.Damage"]),
    M("t-structure-count-of-seven", "T9", "the header may promise more sections than the layout has", [(FILE, "if count < kinds.core_sections() || count > kinds.section_count() {", "if count < kinds.core_sections() || count > kinds.section_count() + 1 {", 1)], ["test_track.Damage"],
      why="a count of 8 reads an eighth section whose tag must be 6 or 7 and above the one before: the walk refuses it as malformed by the order rule (t-structure-skips-the-order), so the bound on the count is not what decides"),
    M("t-constant-reader-accepts-any-tag", "T9", "the constant model's reader takes any tag for a sixth section", [(MFILE, "|| s == 5 && tag != 7 {", "{", 1)], ["test_track.Damage"]),
    M("t-section-not-validated-in-constants", "T9", "the constant model's reader does not look at section 7", [(MFILE, "if places[10] > 0 && provsec.validate(data[places[10]..places[10] + places[11]], rows, 0) != 0 {", "if false {", 1)], ["test_track.Damage"]),
    # ------------------------------------------------------------------ the answers (T5, T6, T7)
    M("t-on-always-train", "T5", "eval says train whatever the file", [(PROV, "    if same {\n        return (\"train\", 0, 0);\n    }", "    if true {\n        return (\"train\", 0, 0);\n    }", 1)], ["test_track.On"]),
    M("t-on-never-hashes", "T5", "eval says unknown for a model with provenance", [(PROV, "    if len(pv) == 0 {\n        return (\"unknown\", 0, 0);\n    }", "    if true {\n        return (\"unknown\", 0, 0);\n    }", 1)], ["test_track.On"]),
    M("t-range-min-inclusive", "T6", "a cell equal to the minimum is out of range", [(PROV, "if v < contents(lr)[k] || v > contents(hr)[k] {", "if v <= contents(lr)[k] || v > contents(hr)[k] {", 1)], RANGE),
    M("t-range-max-inclusive", "T6", "a cell equal to the maximum is out of range", [(PROV, "if v < contents(lr)[k] || v > contents(hr)[k] {", "if v < contents(lr)[k] || v >= contents(hr)[k] {", 1)], RANGE),
    M("t-range-counts-missing", "T6", "a missing cell is compared (it holds 0.0)", [(PROV, "if contents(vr)[k] == 1 && !matrix.is_missing(x, r, cols[k]) {", "if contents(vr)[k] == 1 {", 1)], RANGE),
    M("t-range-counts-the-target", "T6", "the target takes the place of a feature in the check", [(WIO, "if j != schema.target(st.sc) && schema.type_of(st.sc, j) != kinds.schema_type_categorical() {\n                    borrow gn as &gq2 in {", "if schema.type_of(st.sc, j) != kinds.schema_type_categorical() {\n                    borrow gn as &gq2 in {", 1)], RANGE),
    M("t-range-all-columns-one-count", "T6", "every column's count is the total", [(WIO, "w = json.put_int(heap, w, contents(rg.counts)[k]);", "w = json.put_int(heap, w, rg.cells);", 1)], RANGE),
    M("t-range-rows-are-cells", "T6", "rows is the number of cells", [(WIO, "    w = json.put_key(heap, w, \"rows\");\n    w = json.put_int(heap, w, rg.rows);", "    w = json.put_key(heap, w, \"rows\");\n    w = json.put_int(heap, w, rg.cells);", 1)], RANGE),
    M("t-range-no-provenance-column-skip", "T6", "a column with count 0 is compared with its zeros", [(PROV, "if provsec.field(prov, k, 0) > 0 {\n                        contents(vw)[k] = 1;", "if true {\n                        contents(vw)[k] = 1;", 1)], RANGE, why="the fixtures have no column that no training cell filled; the rule is in the decoder (a column with count 0 has bits 0) and in docs/tracking.md section 3.2"),
    M("t-bound-one-late", "T7", "--max-out-of-range N accepts N + 1 rows", [(WIO, "if o.max_oor_given && rr.rows > o.max_oor {", "if o.max_oor_given && rr.rows > o.max_oor + 1 {", 1)], RANGE),
    M("t-bound-one-early", "T7", "--max-out-of-range N refuses N rows", [(WIO, "if o.max_oor_given && rr.rows > o.max_oor {", "if o.max_oor_given && rr.rows >= o.max_oor {", 1)], RANGE),
    M("t-bound-names-the-row-before", "T7", "the refusal names the row before the first out-of-range one", [(PROV, "first_row = r + 1;", "first_row = r;", 1)], RANGE),
    M("t-bound-ignored-without-provenance", "T7", "--max-out-of-range is accepted by a model with no provenance", [(WIO, "if go && o.max_oor_given && !file.has_provenance(stored.lo) {", "if false {", 1)], SECTION),
    M("t-ignored-columns-never", "T6", "predict lists no ignored column", [(WIO, "if j >= 0 && !csv.dropped(sh, j) {\n            w = text.put(heap, w, tname);", "if false {\n            w = text.put(heap, w, tname);", 1)], RANGE),
    M("t-ignored-columns-with-dropped", "T6", "a column dropped by the caller is listed as ignored", [(WIO, "if j >= 0 && !csv.dropped(sh, j) {\n            w = text.put(heap, w, tname);", "if j >= 0 {\n            w = text.put(heap, w, tname);", 1)], RANGE),
    M("t-describe-drops-the-provenance", "T1", "describe prints null for the provenance", [(MDESC, "    if vlen < provsec.header_len() {\n        w = json.put_null(heap, w);", "    if true {\n        w = json.put_null(heap, w);", 1)], SECTION),
    M("t-holdout-label", "T5", "the held-out rows of a fit are labelled train", [(WIRE, 'holdout(heap, args, ds, cids, cvoc, pls, tr, sel, 1, ntest, "held-out", o, e);', 'holdout(heap, args, ds, cids, cvoc, pls, tr, sel, 1, ntest, "train", o, e);', 1)], ["test_track.On", "test_track.Tracking"]),
    # ------------------------------------------------------------------ the record and the id (T2, T10)
    M("t-id-includes-the-thread-count", "T2", "the id is the hash of the inputs cut by the thread count", [(TRK, "        id = run_id(heap, buffer.bytes(ir));", "        id = run_id(heap, buffer.bytes(ir)[0..len(buffer.bytes(ir)) - o.threads]);", 1)], TRACK),
    M("t-record-compare-includes-threads", "T10", "the same run at another thread count is a conflict", [(TRK, "same = true;\n", "same = true;\n", 0)] if False else [(TRK, "let cand = record(heap, buffer.bytes(idr), buffer.bytes(i2), pthreads,", "let cand = record(heap, buffer.bytes(idr), buffer.bytes(i2), o.threads,", 1)], TRACK),
    M("t-conflict-not-detected", "T10", "a record that differs under an existing id is not refused", [(TRK, "            if !same {\n                                var w = open_error(heap, \"ml.runs.conflict\");", "            if false {\n                                var w = open_error(heap, \"ml.runs.conflict\");", 1)], TRACK),
    M("t-record-overwrites", "T10", "an existing record is replaced", [(TRK, "write_atomic(heap, fs, root, paths.shown(rr), paths.full(rr), data, false, dry);", "write_atomic(heap, fs, root, paths.shown(rr), paths.full(rr), data, true, dry);", 1),
                                                                      (TRK, "                        write = false;", "                        write = true;", 1),
                                                                      (TRK, "            if !same {\n                                var w = open_error(heap, \"ml.runs.conflict\");", "            if false {\n                                var w = open_error(heap, \"ml.runs.conflict\");", 1)], TRACK),
    M("t-record-limit-off", "T10", "the 1001st record is written", [(TRK, "if go && !present && count >= max_records() {", "if go && !present && count >= max_records() + 1000 {", 1)], TRACK),
    M("t-record-size-off", "T10", "a record of any size is written", [(TRK, "if recsize > max_record_bytes() {", "if recsize > max_record_bytes() * 100 {", 1)], TRACK),
    M("t-split-record-drops-the-strategy", "T2", "the split text of a record is written without its strategy", [(WIRE, "spec_buf = splitspec.text_nd(heap, false, o.split_num, o.split_den, 0, 0, splitspec.strategy(sp), spcol, o.seed);", "spec_buf = splitspec.text_nd(heap, false, o.split_num, o.split_den, 0, 0, 0, spcol, o.seed);", 1)], TRACK + ["test_track.Splits"]),
    M("t-eval-key-ignores-the-split", "T10", "one file scored under two splits is one eval record", [(TRK, 'if !bytes.equal(buffer.bytes(splr), "none") {', "if false {", 1)], ["test_track.Splits"]),
    M("t-metrics-label-swapped", "T2", "the metrics of a record without a split are labelled held-out", [(WIRE, 'holdout(heap, args, ds, cids, cvoc, pls, tr, sel, 0 - 1, ds.nrows, "train", o, e);', 'holdout(heap, args, ds, cids, cvoc, pls, tr, sel, 0 - 1, ds.nrows, "held-out", o, e);', 1)], TRACK + ["test_track.On"]),
    M("t-id-is-not-the-hash-of-the-inputs", "T2", "the id is the hash of the inputs cut by one byte", [(TRK, "        id = run_id(heap, buffer.bytes(ir));", "        id = run_id(heap, buffer.bytes(ir)[0..len(buffer.bytes(ir)) - 1]);", 1)], TRACK),
    M("t-check-not-checked", "T9", "the check at the end of a record is not compared", [(TRK, "        same = bytes.equal(buffer.bytes(hr), src[n - 67..n - 3]);", "        same = true;", 1)], TRACK),
    M("t-id-not-checked", "T9", "the id of a record is not compared with its inputs", [(TRK, "        same = bytes.equal(buffer.bytes(hr), src[id_at..id_at + 64]);", "        same = true;", 1)], TRACK),
    M("t-eval-record-wrong-run", "T10", "an eval record is written beside any run", [(TRK, "if sh >= 0 && json.string_equals(text, contents(tp), sh, model_hex) {", "if sh >= 0 {", 1)], TRACK),
    # ------------------------------------------------------------------ the verbs (T3)
    M("t-verify-always-true", "T3", "verify says verified whatever the refit gave", [(RUNS, "if bytes.equal(buffer.bytes(fhr), buffer.bytes(msr)) {", "if true {", 1)], TRACK),
    M("t-verify-compares-the-data", "T3", "verify compares the refit with the hash of the data", [(RUNS, "if bytes.equal(buffer.bytes(fhr), buffer.bytes(msr)) {", "if bytes.equal(buffer.bytes(fhr), buffer.bytes(dsr)) {", 1)], TRACK + ["test_track.Replay"]),
    M("t-verify-skips-the-data-hash", "T3", "verify does not compare the training file with the record", [(RUNS, "if !bytes.equal(buffer.bytes(fdr), buffer.bytes(dsr)) {", "if false {", 1)], TRACK),
    M("t-verify-forgets-the-split", "T3", "the refit does not hold the rows out", [(RUNS, "sden = sd;", "sden = 0;", 1)], TRACK + ["test_track.Replay", "test_track.Splits"]),
    M("t-verify-forgets-the-strategy", "T3", "the refit holds the rows out at random whatever the strategy", [(RUNS, "sstrat = sst;", "sstrat = 0;", 1)], ["test_track.Replay", "test_track.Splits"]),
    M("t-verify-forgets-the-split-column", "T3", "the refit has no group or time column", [(RUNS, "ro, spo, buffer.bytes(scr), false, e, lg);", 'ro, spo, "", false, e, lg);', 1)], ["test_track.Replay", "test_track.Splits"]),
    M("t-split-text-not-canonical-is-read", "T9", "a split text that splitspec.text would not write is replayed", [(SPLS, "ok = bytes.equal(buffer.bytes(ar), text);", "ok = true;", 1)], ["test_track.Splits"]),
    M("t-verify-forgets-the-drop-list", "T3", "the refit keeps the dropped columns", [(RUNS, "dmask = m1;", "dmask = 0;", 1)], ["test_track.Replay"]),
    M("t-verify-forgets-the-float-list", "T3", "the refit infers the float columns", [(RUNS, "fmask = m2;", "fmask = 0;", 1)], ["test_track.Replay"]),
    M("t-verify-forgets-the-cat-list", "T3", "the refit declares no categorical column", [(RUNS, "cmask = m3;", "cmask = 0;", 1)], ["test_track.Replay"]),
    M("t-verify-forgets-the-markers", "T3", "the refit has no missing-value marker", [(RUNS, "buffer.bytes(tgr), buffer.bytes(nar), ro, spo, buffer.bytes(scr), false, e, lg);", "buffer.bytes(tgr), \"\", ro, spo, buffer.bytes(scr), false, e, lg);", 1)], ["test_track.Replay"]),
    M("t-verify-forgets-the-seed", "T3", "the refit uses seed 0", [(RUNS, "let seed = field_int(text, contents(tp), inp, \"seed\", 0);", "let seed = 0;", 1)], ["test_track.Replay"]),
    M("t-verify-forgets-the-positive-label", "T3", "the refit takes the larger label as positive", [(RUNS, "let has_pos = pn >= 0 && json.is_int(contents(tp), pn);", "let has_pos = false;", 1)], ["test_track.Replay"]),
    M("t-list-skips-the-damaged", "T9", "list passes over a record that does not validate", [(RUNS, "                        } else {\n                            good = false;\n                        }", "                        } else {\n                            good = true;\n                        }", 1)], TRACK),
    M("t-compare-equal-always", "T3", "compare says the models are equal", [(RUNS, "w = json.put_bool(heap, w, bytes.equal(json.string_view(texta, contents(tpa), ma), json.string_view(textb, contents(tpb), mb)));", "w = json.put_bool(heap, w, true);", 1)], TRACK),
    M("t-sha-not-hashed-twice", "T1", "the training file is not hashed again after the fit", [(WIRE, "        if !same {\n            e = verbs.problem(heap, e, \"io.read-failed\"", "        if false {\n            e = verbs.problem(heap, e, \"io.read-failed\"", 1)], SECTION,
      why="a file that changes between the hashes of one fit cannot be arranged from a test without a second process writing it during the fit; the check is read in docs/tracking.md section 6"),
]

base.MUTANTS = MUTANTS


def build_ml(tree):
    g = lambda pat: sorted(glob.glob(str(tree / pat)))
    libs = (g("lib/core/*.cho") + g("lib/linalg/*.cho") + g("lib/metrics/*.cho") + g("lib/models/*.cho") + g("lib/models/kmeans/*.cho") + g("lib/models/linear/*.cho") +
            g("lib/models/logistic/*.cho") + g("lib/models/tree/*.cho") + g("lib/models/treewire/*.cho") + g("lib/transforms/*.cho") + g("lib/pipeline/*.cho") + g("build/deps/*.cho"))
    target = tree / "build" / "ml"
    p = subprocess.run([base.compiler(), "build", "--std", *(libs + [str(tree / "generated/ml/built.cho")] + g("tools/ml/*.cho")), "-o", str(target)], capture_output=True, text=True, cwd=tree)
    return (target if p.returncode == 0 else None), p.stdout + p.stderr


def run_tests(tree, targets):
    env = dict(os.environ, ML_BIN=str(tree / "build" / "ml"), PYTHONDONTWRITEBYTECODE="1", CANCHO=base.compiler())
    p = subprocess.run([sys.executable, "-m", "unittest", *targets], capture_output=True, text=True, cwd=tree / "tests" / "conformance", env=env)
    return p.returncode, p.stdout + p.stderr


def run_one(m):
    with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        base.copy_tree(tree)
        problems = base.apply_edits(tree, m)
        if problems:
            return "INVALID", "; ".join(problems)
        built, log = build_ml(tree)
        if built is None:
            return "INVALID", "build failed: " + log[:600]
        code, out = run_tests(tree, m["tests"])
        if code != 0:
            return "KILLED", "tests: " + ", ".join(base.failing_tests(out)[:4]) + ("" if base.failing_tests(out) else out.strip().splitlines()[-1][:200])
        return "SURVIVED", ""


def baseline():
    with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        base.copy_tree(tree)
        built, log = build_ml(tree)
        if built is None:
            return "INVALID", "build failed: " + log[:600]
        code, out = run_tests(tree, ["test_track." + c for c in ("Section", "On", "Range", "Damage", "Features", "Tracking", "Replay")])  # Serve needs ml-serve, which this script does not build
        return ("PASSED", "") if code == 0 else ("FAILED", out[-800:])


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        for m in MUTANTS:
            print("%-48s %s  %s" % (m["id"], m["gate"], m["what"]))
        return 0
    if "--check" in argv:
        bad = 0
        import shutil
        with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
            tree = pathlib.Path(td) / "tree"
            base.copy_tree(tree)
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
    results = []
    for m in MUTANTS:
        if only and m["id"] != only:
            continue
        verdict, detail = run_one(m)
        results.append((m, verdict, detail))
        print("%-9s %-48s %s  %s" % (verdict, m["id"], m["gate"], detail), flush=True)
    if not only:
        v, d = baseline()
        print("%-9s %-48s %s  %s" % (v, "baseline (the unchanged tree)", "-", d), flush=True)
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

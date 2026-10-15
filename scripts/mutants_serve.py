#!/usr/bin/env python3
"""Mutants for the gates of docs/serving.md: predict from standard input, `ml-predict` and `describe`. A gate that cannot fail judges nothing.

As scripts/mutants_cli.py: each mutant is a copy of the tree with one change, both binaries rebuilt from the copy, and the tests of the gate it targets run against them.
KILLED means a test failed (or the authority check did); SURVIVED means every test passed, and the mutant must then carry a reason (`why`).

    python3 scripts/mutants_serve.py              # all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_serve.py --only ID    # one
    python3 scripts/mutants_serve.py --list
    python3 scripts/mutants_serve.py --check      # every `old` text is found exactly where it is expected (no build, no test)

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

PRED = "tools/ml-predict/main.cho"
MDESC = "tools/ml/mdesc.cho"

MUTANTS = [
    # ------------------------------------------------------------------ V: standard input
    M("v-bound-allows-one-more", "V3", "standard input may hold cap + 1 bytes", [("tools/ml/paths.cho", "} else if n >= cap {", "} else if n > cap {", 1)], ["test_serving.StdinBounds"]),
    M("v-bound-one-short", "V3", "standard input is refused at cap - 1 bytes", [("tools/ml/paths.cho", "} else if n >= cap {", "} else if n >= cap - 1 {", 1)], ["test_serving.StdinBounds"]),
    M("v-size-in-the-refusal", "V3", "the refusal says size cap, not cap + 1", [("tools/ml/app.cho", 'w = fail.detail_int(heap, w, "size", cap + 1);', 'w = fail.detail_int(heap, w, "size", cap);', 1)],
      ["test_serving.StdinBounds"]),
    M("v-nul-ends-the-input", "V1", "a NUL byte ends standard input", [("tools/ml/paths.cho", "if c < 0 {\n            going = false;", "if c <= 0 {\n            going = false;", 1)], ["test_serving.StdinBounds", "test_serving.StdinEqualsFile"]),
    M("v-chunk-repeats-a-byte", "V1", "the next chunk of memory starts one byte early", [("tools/ml/csv.cho", "Source::Mem(at + n));", "Source::Mem(at + n - 1));", 1)], ["test_serving.StdinEqualsFile"]),
    M("v-chunk-drops-a-byte", "V1", "the next chunk of memory starts one byte late", [("tools/ml/csv.cho", "Source::Mem(at + n));", "Source::Mem(at + n + 1));", 1)], ["test_serving.StdinEqualsFile"]),
    M("v-eval-reads-standard-input", "V4", "eval reads standard input for the operand -", [("tools/ml/app.cho", "    if verb != 2 {\n        return false;\n    }", "    if verb != 2 && verb != 3 {\n        return false;\n    }", 1)],
      ["test_serving.Operand"]),
    M("v-dash-is-not-standard-input", "V4", "the operand - is read as a file", [("tools/ml/app.cho", 'bytes.equal(cli.operand(args, parsed, 1), "-")', 'bytes.equal(cli.operand(args, parsed, 1), "--")', 1)],
      ["test_serving.Operand"]),
    M("v-dot-slash-dash-is-standard-input", "V4", "any operand with a dash is standard input", [("tools/ml/app.cho", 'bytes.equal(cli.operand(args, parsed, 1), "-")', 'bytes.find(cli.operand(args, parsed, 1), "-") >= 0', 1)],
      ["test_serving.Operand", "test_serving.StdinEqualsFile"]),
    M("v-no-operand-is-an-error", "V4", "no operand is not standard input", [("tools/ml/app.cho", "return cli.operand_count(parsed) == 1 || cli.operand_count(parsed) == 2", "return cli.operand_count(parsed) == 2", 1)],
      ["test_serving.StdinEqualsFile", "test_serving.StdinBounds"]),
    M("v-stdin-read-before-the-command-line-is-checked", "V3", "standard input is read even when the command line is refused", [("tools/ml/app.cho", "if !none || !stdin_wanted(args, parsed, verb) {", "if !stdin_wanted(args, parsed, verb) {", 1)],
      ["test_serving.StdinBounds"]),
    M("v-limit-not-applied-to-stdin", "V2", "the line cap is not passed to the scan of memory", [("tools/ml/csvload.cho", "csv.scan_cat(heap, csv.Source::Mem(0), paths.mem(file), cfg, cc,", "csv.scan_cat(heap, csv.Source::Mem(0), paths.mem(file), csv.Cfg { cap: 1048576, max_rows: cfg.max_rows, max_cols: cfg.max_cols, max_cells: cfg.max_cells, mode: cfg.mode, tscale: cfg.tscale, pred: cfg.pred, drops: cfg.drops, floats: cfg.floats, missing: cfg.missing, mrows: cfg.mrows, mcols: cfg.mcols, wpc: cfg.wpc }, cc,", 1)],
      ["test_serving.StdinEqualsFile"]),
    M("v-constant-model-reads-the-file-name", "V1", "verbs.scan_file opens the file even for standard input", [("tools/ml/verbs.cho", "    if paths.is_stdin(file) {\n        let (src, sh, rep, k2) = csv.scan(", "    if paths.is_stdin(file) && false {\n        let (src, sh, rep, k2) = csv.scan(", 1)],
      ["test_serving.StdinEqualsFile"]),
    M("v-planned-model-reads-the-file-name", "V1", "csvload opens the file even for standard input", [("tools/ml/csvload.cho", "    if paths.is_stdin(file) {\n        // standard input", "    if paths.is_stdin(file) && false {\n        // standard input", 1)],
      ["test_serving.StdinEqualsFile"]),
    # ------------------------------------------------------------------ P: ml-predict
    M("p-fit-is-a-verb", "P2", "ml-predict knows the verb fit", [("tools/ml/app.cho", 'if tool == 1 && bytes.equal(text, "fit") {', 'if bytes.equal(text, "fit") {', 1)], ["test_serving.MlPredictSurface"]),
    M("p-eval-is-a-verb", "P2", "ml-predict knows the verb eval", [("tools/ml/app.cho", 'if tool == 1 && bytes.equal(text, "eval") {', 'if bytes.equal(text, "eval") {', 1)], ["test_serving.MlPredictSurface"]),
    M("p-a-fit-flag-in-the-table", "P2", "ml-predict's table has --seed", [("tools/ml/spec.cho", 'pub fn predict_flag_table() -> [] &static [byte] {\n    return "', 'pub fn predict_flag_table() -> [] &static [byte] {\n    return "seed||nat|none|0|a flag of fit;', 1)],
      ["test_serving.MlPredictSurface", "test_serving.FlagTable"]),
    M("p-log-flag-in-the-table", "P2", "ml-predict's table has --log", [("tools/ml/spec.cho", 'pub fn predict_flag_table() -> [] &static [byte] {\n    return "', 'pub fn predict_flag_table() -> [] &static [byte] {\n    return "log||path|path-write||the log;', 1)],
      ["test_serving.MlPredictSurface", "test_serving.FlagTable"]),
    M("p-predict-flag-with-another-default", "T1", "the limit has another default in ml-predict", [("tools/ml/spec.cho", ";limit||nat|none|1000|predict: the most rows in one page of predictions, at least 1. describe: the most category strings listed for each categorical column, at least 1. runs list: the most run records listed, at least 1;from", ";limit||nat|none|500|predict: the most rows in one page of predictions, at least 1. describe: the most category strings listed for each categorical column, at least 1. runs list: the most run records listed, at least 1;from", 2)],
      ["test_serving.FlagTable"]),
    M("p-describe-takes-proba", "T1", "describe takes --proba", [("tools/ml/spec.cho", 'if verb == 4 {\n        return "root,model-file,limit,max-bytes,format";', 'if verb == 4 {\n        return "root,model-file,limit,max-bytes,format,proba";', 1)],
      ["test_serving.FlagTable"]),
    M("p-ml-predict-describes-another-model", "D3", "ml-predict's describe reads the model flag of predict, ignoring --limit", [(PRED, "return mdesc.describe(heap, fs, root, ps.model, o, e1, lg);", "return wireio.predict(heap, args, fs, root, ps.file, ps.model, o, e1, lg);", 1)],
      ["test_serving.MlPredict"]),
    M("p-ml-predict-drops-the-page", "P1", "ml-predict reads no --limit", [(PRED, "let o = app.options(args, parsed, table);", "let o = app.options(args, parsed, spec.flag_table());", 1)], ["test_serving.MlPredict"],
      why="options() reads the flags by name from the table it is given: the full table gives the same values for the flags both have; an equivalent mutant"),
    M("p-ml-predict-calls-the-log", "P3", "ml-predict appends to a log file", [(PRED, "import ml.app;", "import ml.app;\nimport ml.fsio;", 1),
                                                                                (PRED, "-> [heap, args, fs_read(\"\"), dir_read, file_read] verbs.Out {\n    let table = spec.predict_flag_table();\n    let lg = buffer.empty(heap, 1);",
                                                                                 "-> [heap, args, fs_read(\"\"), dir_read, dir_write, file_read, file_write] verbs.Out {\n    let table = spec.predict_flag_table();\n    let (ws, wa) = fsio.append_log(fs, root, \"x\", \"x\", \"x\", 10);\n    let lg = buffer.empty(heap, 1);", 1),
                                                                                (PRED, "fs: &f Fs(\"\"), io: &!i Io, errs: fail.Errors) -> [heap, args, fs_read(\"\"), dir_read, file_read, io_read, io_write, err_write] int {\n    let table = spec.predict_flag_table();\n    let (verb, e0)",
                                                                                 "fs: &f Fs(\"\"), io: &!i Io, errs: fail.Errors) -> [heap, args, fs_read(\"\"), dir_read, dir_write, file_read, file_write, io_read, io_write, err_write] int {\n    let table = spec.predict_flag_table();\n    let (verb, e0)", 1),
                                                                                (PRED, "fn run[&h, &g, &f, &i](heap: &!h Heap, args: &g Args, fs: &f Fs(\"\"), io: &!i Io) -> [heap, args, fs_read(\"\"), dir_read, file_read, io_read, io_write, err_write] int {",
                                                                                 "fn run[&h, &g, &f, &i](heap: &!h Heap, args: &g Args, fs: &f Fs(\"\"), io: &!i Io) -> [heap, args, fs_read(\"\"), dir_read, dir_write, file_read, file_write, io_read, io_write, err_write] int {", 1)],
      ["test_authority"], kind="authority"),
    M("p-ceiling-without-io-read", "V5", "the ceiling of ml-predict lacks io_read", [("tools.toml", '[ml-predict]\nschema = "ml"\nallow = ["args", "dir_read", "err_write", "file_read", "fs_read(\\"\\")", "heap", "io_read", "io_write"]',
                                                                                  '[ml-predict]\nschema = "ml"\nallow = ["args", "dir_read", "err_write", "file_read", "fs_read(\\"\\")", "heap", "io_write"]', 1)], ["test_authority"], kind="authority"),
    M("p-ceiling-too-wide", "P3", "the ceiling of ml-predict allows file_write", [("tools.toml", '[ml-predict]\nschema = "ml"\nallow = ["args", "dir_read", "err_write", "file_read", "fs_read(\\"\\")", "heap", "io_read", "io_write"]',
                                                                                  '[ml-predict]\nschema = "ml"\nallow = ["args", "dir_read", "err_write", "file_read", "file_write", "fs_read(\\"\\")", "heap", "io_read", "io_write"]', 1)], ["test_authority"], kind="ceiling"),
    M("p-no-reason-for-io-read", "V5", "ml has no reason for io_read", [("tools.toml", 'io_read = "predict reads the CSV from standard input for the operand - or no operand: getchar', 'io_read_ = "predict reads the CSV from standard input for the operand - or no operand: getchar', 1)],
      ["test_authority"], kind="authority"),
    M("p-ml-without-io-read-in-the-ceiling", "V5", "the ceiling of ml lacks io_read", [("tools.toml", '"fs_read(\\"\\")", "heap", "io_read", "io_write"]\n\n[ml.reasons]', '"fs_read(\\"\\")", "heap", "io_write"]\n\n[ml.reasons]', 1)], ["test_authority"], kind="authority"),
    # ------------------------------------------------------------------ D: describe
    M("d-encoder-name", "D1", "freq is spelled label", [(MDESC, 'if kd == 9 {\n        return "freq";', 'if kd == 9 {\n        return "label";', 1)], ["test_describe.Oracle"]),
    M("d-required-target", "D1", "the target is required", [(MDESC, "w = json.put_bool(heap, w, j != tcol);", "w = json.put_bool(heap, w, true);", 1)], ["test_describe.Oracle"]),
    M("d-scale-of-a-float", "D1", "the scale is printed for float columns, not decimals", [(MDESC, "fn put_scale[&h](heap: &!h Heap, w0: json.Writer, ty: int, scale: int) -> [heap] json.Writer {\n    var w = json.put_key(heap, w0, \"scale\");\n    if ty == 1 {",
                                                                                              "fn put_scale[&h](heap: &!h Heap, w0: json.Writer, ty: int, scale: int) -> [heap] json.Writer {\n    var w = json.put_key(heap, w0, \"scale\");\n    if ty == 2 {", 1)], ["test_describe.Oracle"]),
    M("d-truncated-at-equal", "D1", "truncated is true when everything is listed", [(MDESC, "w = json.put_bool(heap, w, listed < ncat);", "w = json.put_bool(heap, w, listed <= ncat);", 1)], ["test_describe.Oracle"]),
    M("d-listed-one-short", "D1", "one string fewer than --limit is listed", [(MDESC, "listed = limit;", "listed = limit - 1;", 1)], ["test_describe.Oracle"]),
    M("d-count-of-the-first-string", "D1", "every count is the first string's", [(MDESC, "catenc.count(sr, c, k)", "catenc.count(sr, c, 0)", 1)], ["test_describe.Oracle"]),
    M("d-positive-is-the-negative-label", "D1", "positive is c0", [(MDESC, "if task == 2 && kind > 3 {\n        w = json.put_int(heap, w, c1);", "if task == 2 && kind > 3 {\n        w = json.put_int(heap, w, c0);", 1)], ["test_describe.Oracle"]),
    M("d-classes-swapped", "D1", "the classes are printed c1, c0", [(MDESC, "        w = json.put_int(heap, w, c0);\n        w = json.put_int(heap, w, c1);\n        w = json.end_array(heap, w);", "        w = json.put_int(heap, w, c1);\n        w = json.put_int(heap, w, c0);\n        w = json.end_array(heap, w);", 1)],
      ["test_describe.Oracle"]),
    M("d-missing-cells-of-linear", "D1", "linear is said to refuse missing cells", [(MDESC, "} else if kind <= 3 {\n        w = json.put_string(heap, w, \"refused\");", "} else if kind <= 4 {\n        w = json.put_string(heap, w, \"refused\");", 1)], ["test_describe.Oracle"]),
    M("d-sum-order-of-linear", "D1", "linear has no sum order", [(MDESC, "if mlen == 36 + cl && kind >= 4 {", "if mlen == 36 + cl && kind >= 5 {", 1)], ["test_describe.Oracle"]),
    M("d-digest-of-a-prefix", "D1", "the digest leaves out the last byte", [(MDESC, "let hex = mfile.sha256_hex(heap, data);", "let hex = mfile.sha256_hex(heap, data[0..len(data) - 1]);", 1)], ["test_describe.Oracle"]),
    M("d-markers", "D1", "a one-marker list is not listed", [(MDESC, "    if len(na) > 0 {\n        let count", "    if len(na) > 4 {\n        let count", 1)], ["test_describe.Oracle"]),
    M("d-reader-limits-swapped", "D1", "max_string_bytes is the pool limit", [(MDESC, "w = json.put_int(heap, w, strtab.limit_of(tbl, 1));", "w = json.put_int(heap, w, strtab.limit_of(tbl, 2));", 1)], ["test_describe.Oracle"]),
    M("d-proba-columns-for-all", "D1", "a constant classification takes --proba", [(MDESC, "    w = json.put_key(heap, w, \"with_proba\");\n    if task == 2 && kind > 3 {", "    w = json.put_key(heap, w, \"with_proba\");\n    if task == 2 {", 1)], ["test_describe.Oracle"]),
    M("d-target-type-of-the-first-column", "D1", "the target is typed as column 0", [(MDESC, "w = json.put_string(heap, w, type_word(schema.type_of(sc, tcol)));\n        w = put_scale(heap, w, schema.type_of(sc, tcol), schema.scale_of(sc, tcol));", "w = json.put_string(heap, w, type_word(schema.type_of(sc, 0)));\n        w = put_scale(heap, w, schema.type_of(sc, 0), schema.scale_of(sc, 0));", 1)],
      ["test_describe.Oracle"]),
    M("d-clustering-has-a-target", "D1", "a clustering is described with a target object", [(MDESC, "    w = json.put_key(heap, w, \"target\");\n    if has_t {", "    w = json.put_key(heap, w, \"target\");\n    if true {", 1)], ["test_describe.Oracle"]),
    M("d-clustering-output-label", "D1", "a clustering answers a label, not a cluster", [(MDESC, "    } else if task == 4 {\n        w = json.put_string(heap, w, \"cluster\");", "    } else if task == 40 {\n        w = json.put_string(heap, w, \"cluster\");", 1)], ["test_describe.Oracle"]),
    M("d-clustering-without-distance", "D1", "a clustering has no --distance columns", [(MDESC, "    w = json.put_key(heap, w, \"with_distance\");\n    if task == 4 {", "    w = json.put_key(heap, w, \"with_distance\");\n    if task == 40 {", 1)], ["test_describe.Oracle"]),
    M("d-clustering-takes-missing", "D1", "a clustering is described as taking a missing cell", [(MDESC, "est.takes_missing(wireio.stored_kind(sr))", "est.is_blob(wireio.stored_kind(sr))", 1)], ["test_describe.Oracle"]),
    M("d-header-with-the-target", "D1", "the request header holds the target", [(MDESC, "        if j != tcol {\n            w = text.put(heap, w, schema.name(sc, j));", "        if true {\n            w = text.put(heap, w, schema.name(sc, j));", 1)], ["test_describe.Oracle"]),
    M("d-plan-empty", "D1", "the plan is not printed", [(MDESC, 'w = put_key_text(heap, w, "plan", plan_bytes);', 'w = put_key_text(heap, w, "plan", data[0..0]);', 1)], ["test_describe.Oracle"]),
    M("d-pin-one-late", "D1", "the compiler pin starts one byte late", [(MDESC, "pin = data[mat + 32..mat + 32 + cl];", "pin = data[mat + 33..mat + 32 + cl];", 1)], ["test_describe.Oracle"]),
    M("d-buckets-plus-one", "D1", "hash buckets are one more", [(MDESC, "w = json.put_int(heap, w, catenc.param(sr));", "w = json.put_int(heap, w, catenc.param(sr) + 1);", 1)], ["test_describe.Oracle"]),
    M("d-kind-is-the-dispatch-key", "D1", "kind is the key of the forest (107), not the file's 7", [(MDESC, "est.kind_of(wireio.stored_kind(sr)), wireio.stored_task", "wireio.stored_kind(sr), wireio.stored_task", 1)], ["test_describe.Oracle"]),
    M("d-sections", "D1", "sections is the version", [(MDESC, "w = json.put_int(heap, w, le32(data, 8));", "w = json.put_int(heap, w, le32(data, 4));", 1)], ["test_describe.Oracle"]),
    M("d-max-bytes-is-the-limit", "D1", "limits.max_bytes is --limit", [(MDESC, "w = json.put_int(heap, w, o.max_bytes);\n    w = json.end_object(heap, w);\n    w = json.end_object(heap, w);", "w = json.put_int(heap, w, o.limit);\n    w = json.end_object(heap, w);\n    w = json.end_object(heap, w);", 1)],
      ["test_describe.Oracle"]),
    M("d-field-renamed", "D4", "a field is renamed in the code only", [(MDESC, 'w = json.put_key(heap, w, "extra_columns");', 'w = json.put_key(heap, w, "extra_column");', 1)], ["test_describe.Oracle", "test_describe.Contract"]),
    M("d-field-renamed-in-the-schema", "D4", "a field is renamed in the schema only", [("scripts/schemas.py", '"extra_columns": {"const": "refused"}', '"extra_cols": {"const": "refused"}', 1)], ["test_describe.Contract"], regen=True),
    M("d-field-order", "D4", "two fields swap places", [(MDESC, '    w = json.put_key(heap, w, "rows");\n    w = json.put_int(heap, w, rows);\n    // provenance', '    // provenance', 1),
                                                          (MDESC, '    // target\n    w = json.put_key(heap, w, "target");', '    w = json.put_key(heap, w, "rows");\n    w = json.put_int(heap, w, rows);\n    // target\n    w = json.put_key(heap, w, "target");', 1)],
      ["test_describe.Oracle", "test_describe.Contract"]),
    M("d-bound-read-after-allocation", "D2", "the file is read whole before --max-bytes is looked at (a size from the file is used first)", [(MDESC, "    if st != 0 {\n        e = read_problem(heap, e, st, aux, o.max_bytes, paths.shown(model));", "    if st != 0 && st != 3 {\n        e = read_problem(heap, e, st, aux, o.max_bytes, paths.shown(model));", 1)],
      ["test_describe.Damage"]),
    M("d-damaged-file-described-anyway", "D2", "a refused file falls through to the description", [(MDESC, "            if ds != 0 {\n                e = verbs.model_error(heap, e, ds, da, paths.shown(model));\n            } else {", "            if ds != 0 && ds != 4 {\n                e = verbs.model_error(heap, e, ds, da, paths.shown(model));\n            } else {", 1)],
      ["test_describe.Damage"]),
]

# the golden answer of one model has another digest (the text is data, not code)
MUTANTS.append(M("d-golden-changed", "D4", "the golden answer of one model has another sha256", [("tests/conformance/golden_describe.json", "sha256\\\":\\\"", "sha256\\\":\\\"0", 1)], ["test_describe.Contract"]))
base.MUTANTS = MUTANTS


def build_both(tree):
    g = lambda pat: sorted(glob.glob(str(tree / pat)))
    libs = (g("lib/core/*.cho") + g("lib/linalg/*.cho") + g("lib/metrics/*.cho") + g("lib/models/*.cho") + g("lib/models/kmeans/*.cho") + g("lib/models/linear/*.cho") +
            g("lib/models/logistic/*.cho") + g("lib/models/tree/*.cho") + g("lib/models/treewire/*.cho") + g("lib/transforms/*.cho") + g("lib/pipeline/*.cho") + g("build/deps/*.cho"))
    shared = [f for f in g("tools/ml/*.cho") if not f.endswith("/main.cho")]
    out = {}
    log = ""
    for name, files in (("ml", libs + [str(tree / "generated/ml/built.cho")] + g("tools/ml/*.cho")),
                        ("ml-predict", libs + [str(tree / "generated/ml-predict/built.cho")] + shared + [str(tree / "tools/ml-predict/main.cho")])):
        target = tree / "build" / name
        p = subprocess.run([base.compiler(), "build", "--std", *files, "-o", str(target)], capture_output=True, text=True, cwd=tree)
        log += p.stdout + p.stderr
        if p.returncode != 0:
            return None, log
        out[name] = target
    return out, log


def run_tests(tree, targets):
    env = dict(os.environ, ML_BIN=str(tree / "build" / "ml"), ML_PREDICT_BIN=str(tree / "build" / "ml-predict"), PYTHONDONTWRITEBYTECODE="1", CANCHO=base.compiler(), ST_STRIDE="25")
    p = subprocess.run([sys.executable, "-m", "unittest", *targets], capture_output=True, text=True, cwd=tree / "tests" / "conformance", env=env)
    return p.returncode, p.stdout + p.stderr


def run_one(m):
    with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        base.copy_tree(tree)
        problems = base.apply_edits(tree, m)
        if problems:
            return "INVALID", "; ".join(problems)
        if m["opt"].get("regen"):
            r = subprocess.run([sys.executable, str(tree / "scripts" / "schemas.py")], capture_output=True, text=True, cwd=tree)
            if r.returncode != 0:
                return "INVALID", "schemas.py: " + r.stdout + r.stderr
        kind = m["opt"].get("kind", "")
        if kind == "authority":
            # the manifest is derived from the mutated sources and compared with the committed manifests and the ceilings
            r = subprocess.run([sys.executable, str(tree / "scripts" / "manifest.py"), "--tree", str(tree)], capture_output=True, text=True, cwd=tree)
            if r.returncode != 0:
                return "KILLED", "manifest.py --tree: " + " | ".join(l for l in r.stdout.splitlines() if l.startswith("FAIL"))[:300]
            return "SURVIVED", ""
        built, log = build_both(tree)
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
        built, log = build_both(tree)
        if built is None:
            return "INVALID", "build failed: " + log[:600]
        code, out = run_tests(tree, ["test_serving", "test_describe", "test_authority"])
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

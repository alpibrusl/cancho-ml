#!/usr/bin/env python3
"""Mutants for the wiring of --weight (docs/weights.md section 2.4, gates WG1 to WG9 at the command line): a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change to the new code (the column and the drop list, the reading and the checks of the cells, the refusals of the flag, the gather of the
training weights, the weighted fit, the run record and its replay), the binary rebuilt from the copy, and the tests of the gates it targets run against it. KILLED: a test failed.
SURVIVED: every test passed, and the mutant must carry a reason (`why`). The machinery is that of scripts/mutants_cli.py and scripts/mutants_wire.py.

    python3 scripts/mutants_wire_weights.py              all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_wire_weights.py --only ID
    python3 scripts/mutants_wire_weights.py --list
    python3 scripts/mutants_wire_weights.py --check      every `old` text is found exactly once where it is expected (no build, no test)
"""

import pathlib
import shutil
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_cli as mc  # noqa: E402
import mutants_wire as mw  # noqa: E402

ROOT = mc.ROOT
W = "tools/ml/wire.cho"
WC = "tools/ml/weightcol.cho"
AV = "tools/ml/argv.cho"
CV = "tools/ml/cv.cho"
RU = "tools/ml/runs.cho"
TK = "tools/ml/track.cho"
VB = "tools/ml/verbs.cho"
LN = "lib/models/linear/adapter.cho"
LG = "lib/models/logistic/adapter.cho"
ES = "tools/ml/est.cho"
AT = "lib/models/treewire/adapt.cho"

WT = ["test_weights"]
ORACLE = ["test_weights", "test_wire_oracle"]
ORACLE_T = ["test_weights", "test_wire_tree_oracle"]


def M(id, gate, what, edits, tests, **opt):
    return dict(id=id, gate=gate, what=what, edits=[e if len(e) == 4 else (*e, 1) for e in edits], tests=tests, opt=opt)


MUTANTS = [
    # ------------------------------------------------------------------ the column
    M("ww-weight-is-a-feature", "WG1", "the reader keeps the weight column, so it is a feature", [(AV, "    if has(src, \"weight\") {\n        if len(text(src, \"drop\")) > 0 {", "    if false {\n        if len(text(src, \"drop\")) > 0 {")], WT),
    M("ww-drop-list-separator", "WG1", "the weight name is glued to the last --drop name", [(AV, "            b = buffer.push(heap, b, byte_of(44));", "            b = b;")], WT),
    M("ww-unknown-column-names-drop", "WG6", "an unknown weight column is reported as an unknown --drop column", [(VB, "    if st == 12 && argv.has(src, \"weight\") && rep.aux > argv.names_in(argv.text(src, \"drop\")) {", "    if st == 12 && argv.has(src, \"weight\") && rep.aux > 99 {")], WT),
    M("ww-conflict-target", "WG6", "the weight column may be the target", [(WC, "    if argv.has(src, \"target\") && bytes.equal(w, argv.text(src, \"target\")) {", "    if false {")], WT),
    M("ww-conflict-drop", "WG6", "the weight column may be in --drop", [(WC, "    if argv.has(src, \"drop\") && list_has(argv.text(src, \"drop\"), w) {", "    if false {")], WT),
    M("ww-conflict-cat", "WG6", "the weight column may be in --cat", [(WC, "    if argv.has(src, \"cat\") && list_has(argv.text(src, \"cat\"), w) {", "    if false {")], WT),
    # ------------------------------------------------------------------ the cells
    M("ww-missing-is-zero", "WG6", "an empty cell is a weight of 0", [(WC, "    if st == 1 {\n        return (1, 0.0);", "    if st == 1 {\n        return (0, 0.0);")], WT),
    M("ww-negative-allowed", "WG6", "a negative weight is accepted", [(WC, "    if v < 0.0 {\n        return (4, v);", "    if v < 0.0 - 1.0e300 {\n        return (4, v);")], WT),
    M("ww-too-large-bound", "WG6", "a weight of exactly 1e12 is refused", [(WC, "    if v > 1000000000000.0 {", "    if v >= 1000000000000.0 {")], WT),
    M("ww-not-a-number-ok", "WG6", "a cell that is not a number is a weight of 0", [(WC, "    if st == 2 || st == 5 || st == 6 {\n        return (2, 0.0);", "    if st == 2 || st == 5 || st == 6 {\n        return (0, 0.0);")], WT),
    M("ww-out-of-range-ok", "WG6", "a cell past the range of a double is a weight of 0", [(WC, "    if st == 3 || st == 4 {\n        return (3, 0.0);", "    if st == 3 {\n        return (3, 0.0);")], WT),
    M("ww-row-is-zero-based", "WG6", "the row of a bad weight is counted from 0", [(WC, "e = weight_error(heap, e, \"ml.weight.negative\", i + 1, name, \"\");", "e = weight_error(heap, e, \"ml.weight.negative\", i, name, \"\");")], WT),
    M("ww-zero-total-ignored", "WG6", "a training side of weight 0 goes on to the fit", [(W, "                if !any {\n                    var side", "                if false {\n                    var side")], WT),
    # ------------------------------------------------------------------ the fit
    M("ww-weights-not-used", "WG3", "the fit ignores the weights", [(W, "            if len(contents(ds.wall)) > 1 {", "            if false {")], WT),
    M("ww-gather-all-rows", "WG3", "the weights of every row go to the fit, not those of the training side", [(WC, "    var out = box_slice(heap, cnt + 1, 0.0);\n    var any = false;\n    borrow mut out as &!ow in {\n        var at = 0;\n        r = 0;\n        while r < n {\n            if want < 0 || sel[r] == want {", "    var out = box_slice(heap, cnt + 1, 0.0);\n    var any = false;\n    borrow mut out as &!ow in {\n        var at = 0;\n        r = 0;\n        while r < n {\n            if true {")], ORACLE),
    M("ww-weights-of-the-wrong-rows", "WG3", "the weights of the training side are those of the first rows", [(W, "contents(yfr), contents(wtr2)[0..rows], contents(pls.ip)", "contents(yfr), contents(ds.wall)[0..rows], contents(pls.ip)")], ORACLE),
    M("ww-linear-weighted-dispatch", "WG3", "the linear adapter fits unweighted", [(LN, "    return linreg.fit_weighted(heap, threads, xt, xd, yd, wd, fp[0], out, diag);", "    return linreg.fit(heap, threads, xt, xd, yd, fp[0], out, diag);")], WT),
    M("ww-logistic-weighted-dispatch", "WG3", "the logistic adapter fits unweighted", [(LG, "    return logreg.fit_weighted(heap, threads, xt, xd, yd, wd, ip[0], fp[0], out, diag);", "    return logreg.fit(heap, threads, xt, xd, yd, ip[0], fp[0], out, diag);")], WT),
    # ------------------------------------------------------------------ the models and the verbs
    M("ww-unsupported-fit", "WG6", "a model without a weighted fit takes --weight in fit", [(W, "    if go && len(wn) > 0 && !est.weighted(kind) {", "    if go && len(wn) > 0 && false {")], WT),
    M("ww-unsupported-cv", "WG6", "a model without a weighted fit takes --weight in cv", [(CV, "    if go && argv.has(src, \"weight\") && !est.weighted(kind) {", "    if go && argv.has(src, \"weight\") && false {")], WT),
    M("ww-unsupported-constant", "WG6", "the constant model takes --weight", [(W, "            if argv.has(src, \"weight\") {\n                e = weightcol.unsupported(heap, e);", "            if false {\n                e = weightcol.unsupported(heap, e);")], WT),
    # ------------------------------------------------------------------ the record and the replay
    M("ww-record-without-weight", "WG8", "the run record does not keep the weight column", [(TK, "    if len(weight) > 0 {\n        // the weight column of the fit", "    if false {\n        // the weight column of the fit")], WT),
    M("ww-replay-weight-not-dropped", "WG8", "the replay reads the weight column as a feature", [(RU, "                            drops = buffer.append(heap, drops, buffer.bytes(wn0));", "                            drops = drops;")], WT),
    M("ww-replay-weight-not-passed", "WG8", "the replay refits without the weights", [(RU, "buffer.bytes(nar), buffer.bytes(wnr), ro, spo", "buffer.bytes(nar), \"\", ro, spo")], WT),
    # ------------------------------------------------------------------ the trees and the forest (docs/weights.md section 2.5)
    M("wt-blob-weights-not-used", "WG3", "the fit of a tree or a forest ignores the weights", [(W, "            if go && len(contents(ds.wall)) > 1 {", "            if false && len(contents(ds.wall)) > 1 {")], ORACLE_T),
    M("wt-blob-weights-of-the-wrong-rows", "WG3", "the weights of a tree's training side are those of the first rows", [(W, "contents(yfr)[0..rows], contents(wtr2)[0..rows]", "contents(yfr)[0..rows], contents(ds.wall)[0..rows]")], ORACLE_T),
    M("wt-blob-zero-total-continues", "WG6", "a tree is fitted on a training side of weight 0", [(W, "                    e = weightcol.zero_total(heap, e, side);\n                    go = false;\n                } else {\n                    borrow yfit as &yfr in {", "                    e = weightcol.zero_total(heap, e, side);\n                } else {\n                    borrow yfit as &yfr in {")], WT),
    M("wt-library-fit-unweighted", "WG3", "the adapters of the trees fit unweighted", [(AT, "    if wtd {\n        return forest.fit_weighted(", "    if false {\n        return forest.fit_weighted(")], ORACLE_T),
    M("wt-tree-weighted-flag", "WG6", "the tree takes no --weight", [(ES, "    if kind == 7 {\n        return true;\n    }\n    if kind == 107 {\n        return true;\n    }", "    if kind == 107 {\n        return true;\n    }")], WT),
    M("wt-forest-weighted-flag", "WG6", "the forest takes no --weight", [(ES, "    if kind == 7 {\n        return true;\n    }\n    if kind == 107 {\n        return true;\n    }", "    if kind == 7 {\n        return true;\n    }")], WT),
    M("wt-tree-dispatch-unweighted", "WG3", "the dispatch of a tree to its weighted fit is the unweighted one", [(ES, "        return adapter_tree.fit_blob_weighted(heap, threads, xt, xd, miss, wpc, yd, wd, ip, fp, seed, c0, c1, rows_cap, cols_cap, cells_cap);", "        return adapter_tree.fit_blob(heap, threads, xt, xd, miss, wpc, yd, ip, fp, seed, c0, c1, rows_cap, cols_cap, cells_cap);")], ORACLE_T),
    M("wt-forest-dispatch-unweighted", "WG3", "the dispatch of a forest to its weighted fit is the unweighted one", [(ES, "        return adapter_forest.fit_blob_weighted(heap, threads, xt, xd, miss, wpc, yd, wd, ip, fp, seed, c0, c1, rows_cap, cols_cap, cells_cap);", "        return adapter_forest.fit_blob(heap, threads, xt, xd, miss, wpc, yd, ip, fp, seed, c0, c1, rows_cap, cols_cap, cells_cap);")], ORACLE_T),
]


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        for m in MUTANTS:
            print("%-48s %s  %s" % (m["id"], m["gate"], m["what"]))
        return 0
    if "--check" in argv:
        bad = 0
        with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
            tree = pathlib.Path(td) / "tree"
            mc.copy_tree(tree)
            ids = [m["id"] for m in MUTANTS]
            if len(set(ids)) != len(ids):
                print("two mutants with one id")
                bad += 1
            for m in MUTANTS:
                problems = mc.apply_edits(tree, m)
                if problems:
                    print("INVALID %s: %s" % (m["id"], "; ".join(problems)))
                    bad += 1
                for item in m["edits"]:
                    shutil.copy(ROOT / item[0], tree / item[0])
        print("%d weight mutants, %d with an edit that does not apply" % (len(MUTANTS), bad))
        return 1 if bad else 0
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    results = []
    for m in MUTANTS:
        if only and m["id"] != only:
            continue
        verdict, detail = mw.run_one(m)
        results.append((m, verdict, detail))
        print("%-9s %-48s %s  %s" % (verdict, m["id"], m["gate"], detail), flush=True)
    if not only:
        v, d = mc.baseline()
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

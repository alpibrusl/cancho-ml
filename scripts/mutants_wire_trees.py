#!/usr/bin/env python3
"""Mutants for the wiring of the tree and the forest (docs/wiring-trees.md gate WT10): a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change to the new code (the adapters and their shared glue in lib/models/treewire, the manifests' mapping of flags to
hyperparameters in the generator, the task, the probability, the reader of the kind 7 payload, the row bound, the acceptance of missing cells, the thread count, the sum
order), the binary rebuilt from the copy, and the tests of the gates it targets run against it. KILLED: a test failed. SURVIVED: every test passed, and the mutant must
carry a reason (`why`). The machinery is that of scripts/mutants_cli.py and scripts/mutants_wire.py.

    python3 scripts/mutants_wire_trees.py              all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_wire_trees.py --only ID
    python3 scripts/mutants_wire_trees.py --list
    python3 scripts/mutants_wire_trees.py --check      every `old` text is found exactly once where it is expected (no build, no test)
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_cli as mc  # noqa: E402
import mutants_wire as mw  # noqa: E402

ROOT = mc.ROOT
W = "tools/ml/wire.cho"
IO = "tools/ml/wireio.cho"
D = "tools/ml/wiredata.cho"
G = "scripts/gen_models.py"
A = "lib/models/treewire/adapt.cho"
AF = "lib/models/treewire/adapter_forest.cho"
AT = "lib/models/treewire/adapter_tree.cho"
K = "lib/core/kinds.cho"
FI = "lib/pipeline/file.cho"

ORACLE = ["test_wire_tree_oracle"]
TREES = ["test_wire_trees"]
FILES = ["test_wire_trees_file"]
RULES = ["test_rules"]
BASE = ["test_wire_baseline", "test_cat_baseline", "test_planned_baseline"]


def M(id, gate, what, edits, tests, **opt):
    return dict(id=id, gate=gate, what=what, edits=[e if len(e) == 4 else (*e, 1) for e in edits], tests=tests, opt=opt)


MUTANTS = [
    # ------------------------------------------------------------------ the flags to the hyperparameters
    M("wt-forest-min-leaf-from-the-bins", "WT4", "the forest's min-leaf is read from the place of bins", [(AF, "min_leaf: ip[3]", "min_leaf: ip[4]")], ORACLE + TREES),
    M("wt-forest-bins-from-the-mtry", "WT4", "the forest's bins are read from the place of mtry", [(AF, "bins: ip[4]", "bins: ip[5]")], ORACLE + TREES),
    M("wt-forest-mtry-from-the-bootstrap", "WT4", "the forest's mtry is read from the place of bootstrap", [(AF, "mtry: ip[5]", "mtry: ip[6]")], ORACLE + TREES),
    M("wt-forest-bootstrap-always-on", "WT4", "the forest bootstraps whatever --bootstrap says", [(AF, "bootstrap: ip[6]", "bootstrap: 1")], ORACLE + TREES),
    M("wt-forest-max-nodes-ignored", "WT6", "the forest's --max-nodes is not read", [(AF, "max_nodes: ip[7]", "max_nodes: 4194304")], TREES + RULES),
    M("wt-forest-max-depth-from-the-trees", "WT4", "the forest's max-depth is read from the place of trees", [(AF, "max_depth: ip[2]", "max_depth: ip[1]")], ORACLE + TREES),
    M("wt-tree-bootstraps", "WT4", "the tree draws its rows with replacement", [(AT, "bootstrap: 0", "bootstrap: 1")], ORACLE + TREES),
    M("wt-tree-tries-one-column", "WT4", "the tree tries one column at each node", [(AT, "mtry: d,", "mtry: 1,")], ORACLE + TREES),
    M("wt-tree-is-two-trees", "WT4", "the tree is a forest of two", [(AT, "trees: 1,", "trees: 2,")], ORACLE + TREES),
    M("wt-tree-min-leaf-from-the-depth", "WT4", "the tree's min-leaf is read from the place of max-depth", [(AT, "min_leaf: ip[2]", "min_leaf: ip[1]")], ORACLE + TREES),
    M("wt-enum-index-reversed", "WT2", "the plan's task=classification is stored as the index of regression", [(G, "\"            ip[%d] = %d;\" % (ni, vi)]", "\"            ip[%d] = %d;\" % (ni, 1 - vi)]")], TREES + ORACLE, gen=True),
    M("wt-task-of-the-plan-reversed", "WT2", "task_of answers classification for the index of regression", [(G, "\"        if ip[%d] == %d {\" % (idx, vals.index(\"classification\"))", "\"        if ip[%d] == %d {\" % (idx, vals.index(\"regression\"))")], TREES + ORACLE, gen=True),
    M("wt-zero-is-not-a-flag-value", "WT1", "the command line accepts no 0 for --mtry and --bootstrap", [(G, "        if mins[n] != 1:", "        if False:")], TREES + ORACLE, gen=True),
    # ------------------------------------------------------------------ the task
    M("wt-task-token-always-regression", "WT2", "the model line of a fit from flags always says task=regression", [(W, "b = buffer.append(heap, b, cli.text(args, parsed, table, \"task\"));", "b = buffer.append(heap, b, \"regression\");")], TREES + ORACLE + RULES),
    M("wt-positive-with-a-regression-forest-accepted", "WT2", "--positive with a regression forest goes on", [(W, "            if ptask == 1 && cli.has(parsed, table, \"positive\") {", "            if false && cli.has(parsed, table, \"positive\") {")], RULES),
    M("wt-task-flag-conflicts-with-a-tree", "WT2", "--task is a conflict for a model of both tasks", [(W, "if cli.has(parsed, table, \"task\") && est.task(kind) != 3 {", "if cli.has(parsed, table, \"task\") {")], TREES + ORACLE + RULES),
    M("wt-classification-trains-the-other-label", "WT2", "a classification tree is trained with 1.0 on the negative label", [(W, "                            if truncate(contents(yr)[i]) == ds.c1 {", "                            if truncate(contents(yr)[i]) == ds.c0 {")], ORACLE),
    M("wt-holdout-task-is-classification", "WT2", "the held-out metrics of a fit are those of a classification", [(W, "wiredata.metrics_block(heap, pls.kind, est.task_of(pls.kind, contents(pls.ip)), xq2,", "wiredata.metrics_block(heap, pls.kind, 2, xq2,")], ORACLE + TREES),
    M("wt-task-of-the-file-from-the-plan-kind", "WT2", "predict reads the task of a forest from est.task, which is 3", [(IO, "    var task = est.task(kind);\n    borrow lo as &lr in {\n        if est.is_blob(kind) {", "    var task = est.task(kind);\n    borrow lo as &lr in {\n        if false {")], TREES + ORACLE + RULES),
    # ------------------------------------------------------------------ the probability and the label
    M("wt-proba-is-the-class", "WT2", "--proba prints the class where the probability goes", [(D, "return est.proba_blob(heap, kind, blob,", "return est.predict_blob(heap, kind, blob,")], ORACLE + TREES),
    M("wt-forest-proba-mode", "WT2", "the forest's probability is its class", [(AF, "return adapt_trees.predict_rows(heap, blob, xt, xd, miss, wpc, out, 1);", "return adapt_trees.predict_rows(heap, blob, xt, xd, miss, wpc, out, 0);")], ORACLE + TREES),
    M("wt-class-of-a-tie", "WT2", "a probability of exactly one half is the positive class", [(A, "            if out[i] > 0.5 {", "            if out[i] >= 0.5 {")], ORACLE + TREES,
      why="a tie needs a forest whose positive fraction is exactly 1/2 for some row; the generated forests have 1, 3, 7 or 10 trees and a leaf is a fraction of its rows, so 1/2 is reachable only by luck; docs/models-tree.md unit test `aggregation` pins the rule for the library, and eval's predictions are the class of the same function"),
    M("wt-regression-proba-allowed", "WT2", "the adapter answers a probability for a regression", [(A, "    if mode == 1 && task != 0 {", "    if false {")], TREES + RULES,
      why="a second line of defence: predict --proba on a regression is args.conflict before the adapter is called and eval calls proba for a classification only; no command line reaches the adapter's refusal"),
    # ------------------------------------------------------------------ the kind 7 reader
    M("wt-labels-record-not-checked", "WT11", "the labels record of a classification is read without its checks", [(A, "    if lab.code != 0 || lab.end != len(src) {", "    if false {")], FILES),
    M("wt-longer-payload-accepted", "WT11", "bytes after the payload are accepted", [(A, "    if len(src) != want {", "    if len(src) < want {")], FILES),
    M("wt-width-of-the-payload-not-compared", "WT11", "the payload's number of columns is not compared with the schema's", [(A, "    if code != 0 || fd != d {", "    if code != 0 {")], FILES),
    M("wt-task-of-plan-and-payload-not-compared", "WT11", "a plan that says classification over a regression payload is accepted", [(IO, "    if dcode == 0 && plan_task != 0 && plan_task != task {", "    if false {")], FILES),
    M("wt-library-reader-skipped-for-the-header", "WT11", "the header size of the payload is not checked against the ceilings", [(A, "total < trees || total > params.nodes_total_max()", "total < trees")], FILES,
      why="a total above the ceiling makes lib_size compute a size the file cannot have (len(src) != want is malformed first) or, for a total the size equation can hold, the library reader's own ceiling refuses it: two checks of one bound"),
    M("wt-plan-key-not-checked-against-the-kind", "WT11", "the dispatch key named by the plan is not compared with the kind of the payload", [(IO, "    if est.kind_of(key) != kind {", "    if false {")], FILES,
      why="pipeline.file.decode has already refused a plan whose model family is not the payload's (model_family_ok): this comparison is a second line of defence that no file reaches"),
    M("wt-sum-order-of-kind-7-is-2", "WT3", "a tree or forest records the lane order", [(K, "pub fn sum_order_sequential() -> [] int {\n    return 3;", "pub fn sum_order_sequential() -> [] int {\n    return 2;")], ORACLE + TREES),
    M("wt-sum-order-3-for-every-ordered-kind", "WT12", "the lane kinds record the sequential order too", [(K, "    if kind == model_tree() {\n        return sum_order_sequential();", "    if kind >= first_ordered_model() {\n        return sum_order_sequential();")], BASE),
    M("wt-sum-order-checked-as-2", "WT3", "the reader requires the lane order of a kind 7 file", [(FI, "if so != kinds.sum_order_for(get32(md0, 0)) {", "if so != kinds.sum_order() {")], TREES + ORACLE),
    M("wt-kind-7-of-a-constant-plan-is-ours", "WT12", "a constant file with a kind 7 payload goes to the tree reader", [(IO, "kind == 7 && !names_tree(data[plan_at..plan_at + plan_len])", "false")], BASE),
    # ------------------------------------------------------------------ the row bound
    M("wt-row-bound-removed", "WT6", "the training rows of a tree fit are not counted", [(W, "        if ntrain > params.rows_max() {", "        if false {")], TREES),
    M("wt-row-bound-off-by-one", "WT6", "a tree fit takes 2^20 - 1 rows at most", [(W, "        if ntrain > params.rows_max() {", "        if ntrain >= params.rows_max() {")], TREES),
    M("wt-row-bound-counts-the-file", "WT6", "the bound counts the rows of the file, not the training rows", [(W, "ntrain = wiredata.count_rows(contents(sq0), dn, want);", "ntrain = dn;")], TREES),
    M("wt-row-bound-message", "WT6", "the refusal says the generic thing about --max-rows", [(D, "a tree or forest fit takes at most 1048576 training rows (the bound of its exact Gini comparison)", "the file has more data rows than --max-rows")], TREES),
    # ------------------------------------------------------------------ missing cells
    M("wt-fit-refuses-missing-for-the-trees", "WT5", "a tree fit with a missing cell after the steps is refused", [(W, "                if mc > 0 && !est.is_blob(pls.kind) {\n                    borrow mlist as &lr in {\n                        e = wiredata.missing_error(heap, args, e, mc, buffer.bytes(lr), retry_ok);", "                if mc > 0 {\n                    borrow mlist as &lr in {\n                        e = wiredata.missing_error(heap, args, e, mc, buffer.bytes(lr), retry_ok);")], TREES + ORACLE + RULES),
    M("wt-predict-refuses-missing-for-the-trees", "WT5", "predict with a missing cell is refused for a tree", [(IO, "        if mc > 0 && !est.is_blob(st.kind) {\n            borrow mlist as &lr in {\n                e = wiredata.missing_error(heap, args, e, mc, buffer.bytes(lr), false);", "        if mc > 0 {\n            borrow mlist as &lr in {\n                e = wiredata.missing_error(heap, args, e, mc, buffer.bytes(lr), false);")], TREES + ORACLE + RULES),
    M("wt-holdout-refuses-missing-for-the-trees", "WT5", "the held-out rows of a split fit with a missing cell are refused for a tree", [(W, "                        if mc > 0 && !est.is_blob(pls.kind) {\n                            borrow mlist as &lr in {\n                                e = wiredata.missing_error(heap, args, e, mc, buffer.bytes(lr), false);", "                        if mc > 0 {\n                            borrow mlist as &lr in {\n                                e = wiredata.missing_error(heap, args, e, mc, buffer.bytes(lr), false);")], ORACLE),
    M("wt-empty-column-is-missing-for-all", "WT5", "the schema check treats a column with no cell as absent for the trees too", [(IO, "o, need_target, est.is_blob(st.kind));", "o, need_target, false);")], TREES + ORACLE),
    M("wt-empty-column-is-fine-for-all", "WT5", "the schema check accepts a column with no cell for every planned model", [(IO, "o, need_target, est.is_blob(st.kind));", "o, need_target, true);")], TREES),
    M("wt-bitmap-bit-order", "WT5", "the bytes of the library's bitmap are taken from the other end of the word", [(A, "let v = miss[c * wpc + (b >> 2)] >> 8 * (b & 3) & 255;", "let v = miss[c * wpc + (b >> 2)] >> 8 * (3 - (b & 3)) & 255;")], ORACLE),
    M("wt-bitmap-word-of-a-byte", "WT5", "a byte of the bitmap comes from the wrong word", [(A, "let v = miss[c * wpc + (b >> 2)] >> 8 * (b & 3) & 255;", "let v = miss[c * wpc + (b >> 3)] >> 8 * (b & 3) & 255;")], ORACLE),
    M("wt-bitmap-column-start", "WT5", "the bitmap of column c starts at the wrong word", [(A, "let v = miss[c * wpc + (b >> 2)] >> 8 * (b & 3) & 255;", "let v = miss[c + wpc * 0 + (b >> 2)] >> 8 * (b & 3) & 255;")], ORACLE),
    M("wt-bitmap-not-passed-when-any", "WT5", "the fit hands the library an empty bitmap although cells are missing", [(A, "        if any {\n            let (f2, c2) = forest.fit(", "        if false {\n            let (f2, c2) = forest.fit(")], ORACLE),
    M("wt-bitmap-not-passed-at-predict", "WT5", "the prediction hands the library an empty bitmap although cells are missing", [(A, "            if any {\n                rc = walk(", "            if false {\n                rc = walk(")], ORACLE),
    # ------------------------------------------------------------------ the thread count
    M("wt-threads-minus-one", "WT7", "the library gets one thread fewer than --threads", [(AF, "seed: seed, threads: threads }", "seed: seed, threads: threads - 1 }")], TREES + ORACLE),
    M("wt-threads-reach-the-seed", "WT7", "the seed of the forest depends on the thread count", [(AF, "seed: seed, threads: threads }", "seed: seed + threads - 1, threads: threads }")], TREES + ORACLE),
    M("wt-tree-threads-reach-the-seed", "WT7", "the seed of the tree depends on the thread count", [(AT, "seed: seed, threads: threads }", "seed: seed + threads - 1, threads: threads }")], TREES + ORACLE),
    M("wt-threads-not-passed-to-the-fit", "WT7", "the forest is fitted with one thread whatever --threads says", [(AF, "seed: seed, threads: threads }", "seed: seed, threads: 1 }")], TREES + ORACLE,
      why="an equivalent mutant by design: no thread count changes a byte of the model (the property this gate asserts), so a fit that ignores the flag writes the same file; nothing a test reads from the file or the answer tells the two apart"),
    M("wt-fit-threads-from-the-seed", "WT7", "ml hands the adapter the seed where the threads go", [(W, "est.fit_blob(heap, pls.kind, o.threads,", "est.fit_blob(heap, pls.kind, o.seed + 1,")], TREES + ORACLE),
    # ------------------------------------------------------------------ the answers
    M("wt-summary-trees-from-the-wrong-field", "WT2", "the summary of a fit reads the number of trees from the number of nodes", [(W, "w = json.put_int(heap, w, fbits.get64(blob, 28));", "w = json.put_int(heap, w, fbits.get64(blob, 52));")], TREES + RULES),
    M("wt-name-of-a-forest-is-the-trees", "WT2", "a forest file is read under the tree's dispatch key", [(IO, "    kind = key;", "    kind = est.kind_of(key);")], TREES),
    # ------------------------------------------------------------------ the tags
    M("wt-limit-depth-tag-missing", "WT6", "the code of the library's depth refusal has no tag in ml", [(D, "    if code >= 101 && code <= deny.last() {", "    if code >= 101 && code <= 100 {")], RULES + TREES),
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
            for m in MUTANTS:
                problems = mc.apply_edits(tree, m)
                if problems:
                    print("INVALID %s: %s" % (m["id"], "; ".join(problems)))
                    bad += 1
                for item in m["edits"]:
                    shutil.copy(ROOT / item[0], tree / item[0])
        print("%d treewire mutants, %d with an edit that does not apply" % (len(MUTANTS), bad))
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

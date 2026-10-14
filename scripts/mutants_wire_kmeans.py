#!/usr/bin/env python3
"""Mutants for the wiring of k-means (docs/wiring-kmeans.md gate WK11): a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change to the new code (the adapter, the manifest's mapping of flags to hyperparameters in the generator, the optional target, the
task value, the predict column, the distances, the inertia, the reader of the kind 6 payload and of the schema without a target, the bounds of k, the choice of the
initialisation, the scoring of held-out rows, the thread count), the binary rebuilt from the copy, and the tests of the gates it targets run against it. KILLED: a test failed.
SURVIVED: every test passed, and the mutant must carry a reason (`why`). The machinery is that of scripts/mutants_cli.py and scripts/mutants_wire.py.

    python3 scripts/mutants_wire_kmeans.py              all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_wire_kmeans.py --only ID
    python3 scripts/mutants_wire_kmeans.py --list
    python3 scripts/mutants_wire_kmeans.py --check      every `old` text is found exactly once where it is expected (no build, no test)
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
IO = "tools/ml/wireio.cho"
D = "tools/ml/wiredata.cho"
MAIN = "tools/ml/main.cho"
G = "scripts/gen_models.py"
A = "lib/models/kmeans/adapter.cho"
MT = "lib/models/kmeans/model.toml"
K = "lib/core/kinds.cho"
FI = "lib/pipeline/file.cho"
SC = "lib/pipeline/schema.cho"
CW = "tools/ml/catwire.cho"

KM = ["test_wire_kmeans"]
ORACLE = ["test_wire_kmeans_oracle"]
FILES = ["test_wire_kmeans_file"]
RULES = ["test_rules"]
TABLES = ["test_tables"]
BASE = ["test_wire_baseline", "test_cat_baseline", "test_planned_baseline"]


def M(id, gate, what, edits, tests, **opt):
    return dict(id=id, gate=gate, what=what, edits=[e if len(e) == 4 else (*e, 1) for e in edits], tests=tests, opt=opt)


MUTANTS = [
    # ------------------------------------------------------------------ the flags to the hyperparameters, the initialisation
    M("wk-k-from-the-iterations", "WK5", "the adapter reads k from the place of the iterations", [(A, "    let k = ip[0];", "    let k = ip[1];")], KM + ORACLE),
    M("wk-iterations-from-k", "WK5", "the adapter reads the iterations from the place of k", [(A, "    let iters = ip[1];", "    let iters = ip[0];")], KM + ORACLE),
    M("wk-init-inverted", "WK5", "random is kmeans++ and kmeans++ is random", [(A, "    let init = ip[2];", "    let init = 1 - ip[2];")], KM + ORACLE),
    M("wk-init-always-random", "WK5", "the initialisation is random whatever --init says", [(A, "    let init = ip[2];", "    let init = 0;")], KM + ORACLE),
    M("wk-init-index-of-the-enum", "WK1", "the plan's init=kmeans++ is stored as the index of random", [(G, "\"            ip[%d] = %d;\" % (ni, vi)]", "\"            ip[%d] = %d;\" % (ni, 1 - vi)]")], KM + ORACLE + TABLES, gen=True),
    M("wk-init-word-reversed", "WK3", "the fit says kmeans++ where the initialisation is random", [(D, "    if i == 0 {\n        return \"random\";", "    if i == 1 {\n        return \"random\";")], KM),
    M("wk-init-has-no-flag", "WK1", "an enum hyperparameter other than the task has no flag", [(G, "    return [h for h in m[\"hyperparameter\"] if not (h[\"type\"] == \"enum\" and h[\"name\"] == \"task\")]", "    return [h for h in m[\"hyperparameter\"] if h[\"type\"] != \"enum\"]")], KM + TABLES, gen=True),
    M("wk-k-optional", "WK7", "k has a default of 3 in the manifest", [(MT, "max = 1024\nrequired = true", "max = 1024\ndefault = \"3\"")], KM + RULES, gen=True),
    M("wk-k-ceiling-2048", "WK7", "the ceiling of k is 2048 in the manifest", [(MT, "min = 1\nmax = 1024\nrequired = true", "min = 1\nmax = 2048\nrequired = true")], KM + RULES + TABLES, gen=True),
    M("wk-task-number-of-a-clustering", "WK3", "a clustering is a regression for the dispatch", [(G, "\"both\": \"3\", \"clustering\": \"4\"", "\"both\": \"3\", \"clustering\": \"1\"")], KM + ORACLE + RULES, gen=True),
    M("wk-kmeans-takes-missing-cells", "WK6", "a blob model of any task takes missing cells", [(G, "for m in blob if m[\"task\"] == \"both\")", "for m in blob)")], KM + RULES, gen=True),
    # ------------------------------------------------------------------ no target
    M("wk-target-required-for-kmeans", "WK2", "fit --model kmeans needs --target too", [(MAIN, " && est.task(est.kind_named(cli.text(args, parsed, table, \"model\"))) != 4 {", " {")], KM + RULES),
    M("wk-target-never-required", "WK2", "no model needs --target", [(MAIN, "if !cli.has(parsed, table, \"target\") && !cli.has(parsed, table, \"plan\")", "if false && !cli.has(parsed, table, \"plan\")")], KM + RULES),
    M("wk-target-allowed-with-kmeans", "WK2", "--target is accepted with --model kmeans", [(W, "                if cli.has(parsed, table, \"target\") {\n                    e = wiredata.conflict(heap, e, \"target\"", "                if false {\n                    e = wiredata.conflict(heap, e, \"target\"")], KM + RULES),
    M("wk-task-allowed-with-kmeans", "WK2", "--task is accepted with --model kmeans", [(W, "                if cli.has(parsed, table, \"task\") {\n                    e = wiredata.conflict(heap, e, \"task\", \"kmeans is", "                if false {\n                    e = wiredata.conflict(heap, e, \"task\", \"kmeans is")], KM + RULES),
    M("wk-positive-allowed-with-kmeans", "WK2", "--positive is accepted with --model kmeans", [(W, "            if ptask == 4 && cli.has(parsed, table, \"positive\") {", "            if false {")], KM + RULES),
    M("wk-plan-target-conflict-dropped", "WK2", "a target given with a plan of k-means is accepted", [(W, "        if task == 4 && o.target_given {", "        if false {")], KM + RULES),
    M("wk-plan-target-required-dropped", "WK2", "a plan of another model with no --target is accepted", [(W, "        } else if task != 4 && !o.target_given {", "        } else if false {")], KM + RULES),
    M("wk-load-demands-the-target", "WK2", "the reader refuses a file with no target column for k-means", [(W, "        } else if tcol < 0 && task != 4 {", "        } else if tcol < 0 {")], KM + RULES),
    M("wk-features-lose-a-column", "WK5", "a table with no target has its last column dropped from the features", [(W, "    return ds.ncols;\n}", "    return ds.ncols - 1;\n}")], KM + ORACLE),
    M("wk-schema-target-is-column-zero", "WK4", "the schema of a fit without a target marks the first column", [(D, "    var target = kept;", "    var target = 0;")], KM + ORACLE + FILES),
    M("wk-numeric-schema-target-is-column-zero", "WK4", "the numeric schema of a categorical fit marks the first column", [(CW, "    var target = kept;", "    var target = 0;", 1)], KM),
    M("wk-full-schema-target-is-column-zero", "WK4", "the full schema of a categorical fit marks the first column", [(CW, "    var target = kept;", "    var target = 0;", 2)], KM),
    M("wk-empty-file-lower-bound", "WK7", "a file with no column left is not empty for k-means", [(W, "    } else if task == 4 && ncols + ncat < 1 {", "    } else if task == 4 && ncols + ncat < 0 {")], KM,
      why="a second line of defence: the reader refuses a file whose every column is dropped (ml.input.empty, 'a header and no data rows') before load_data counts the columns, so the check is reached by no input; it stays for the day the reader keeps a zero-column table"),
    M("wk-schema-and-model-kind-not-compared", "WK4", "a k-means file with a target and another file with none are accepted", [(FI, "    if mk == kinds.model_kmeans() == has_target {", "    if false {")], KM + FILES + RULES),
    M("wk-schema-target-above-columns", "WK4", "the schema reader accepts a target index of the number of columns plus one", [(SC, "target > cols {", "target > cols + 1 {")], KM + FILES),
    M("wk-meta-columns", "WK4", "meta counts the features plus one columns for every kind", [(FI, "mc != scols ||", "mc != scols + 1 ||")], KM + FILES + BASE),
    # ------------------------------------------------------------------ the verbs
    M("wk-task-word", "WK3", "the answers call a clustering a regression", [(D, "    if task == 4 {\n        return \"clustering\";\n    }\n", "")], KM + ORACLE),
    M("wk-predict-column", "WK3", "predict names its first column label", [(IO, "w = json.put_string(heap, w, \"cluster\");", "w = json.put_string(heap, w, \"label\");")], KM + ORACLE + RULES),
    M("wk-distance-column", "WK3", "predict names its distance column probability", [(IO, "w = json.put_string(heap, w, \"distance\");", "w = json.put_string(heap, w, \"probability\");")], KM + ORACLE + RULES),
    M("wk-distance-not-computed", "WK3", "predict computes no distance", [(IO, "(task == 2 || task == 4 && with_distance)", "(task == 2)")], KM + ORACLE),
    M("wk-cluster-index-plus-one", "WK3", "predict prints the cluster counted from 1", [(IO, "put_label(heap, w, plain, truncate(contents(pq)[k]));", "put_label(heap, w, plain, truncate(contents(pq)[k]) + 1);")], KM + ORACLE),
    M("wk-distance-accepted-by-linear", "WK3", "--distance is accepted by a model that has no centres", [(IO, "    if go && o.distance {", "    if false && o.distance {")], KM + RULES),
    M("wk-distance-accepted-by-constant", "WK3", "--distance is accepted by a constant model", [(IO, "    if verb == 2 && o.distance {", "    if false && o.distance {")], KM + RULES),
    M("wk-proba-accepted-by-kmeans", "WK3", "--proba is accepted by k-means", [(IO, "        } else if is_clu {", "        } else if false {")], KM + RULES),
    M("wk-eval-positive-accepted", "WK3", "--positive is accepted by eval of k-means", [(IO, "        if task == 4 && o.positive_given {", "        if false {")], KM + RULES),
    M("wk-eval-reads-a-target", "WK3", "eval of a clustering reads a target column", [(IO, "                    if ir.tj >= 0 {", "                    if true {")], KM + ORACLE),
    M("wk-eval-clusters-field", "WK3", "eval does not say the number of clusters", [(IO, "w = json.put_key(heap, w, \"clusters\");", "w = json.put_key(heap, w, \"cluster_count\");")], KM + ORACLE + RULES),
    M("wk-distance-to-the-first-centre", "WK3", "the distance of a row is to the first centre, not to its own", [(A, "contents(cr)[c * d..c * d + d], d);", "contents(cr)[0..d], d);")], KM + ORACLE),
    M("wk-predict-and-distance-swapped", "WK3", "mode 0 of the adapter answers the distance and mode 1 the cluster", [(A, "                    if mode == 0 {", "                    if mode == 1 {")], KM + ORACLE),
    M("wk-eval-metrics-of-the-fit-rows", "WK3", "the held-out rows of a split fit are the training rows", [(W, "wiredata.take(heap, ds.x, sel, 1, contents(fr), nfeat, o.max_rows, o.max_cols, o.max_cells);\n            matrix.drop(heap, xte);", "wiredata.take(heap, ds.x, sel, 0, contents(fr), nfeat, o.max_rows, o.max_cols, o.max_cells);\n            matrix.drop(heap, xte);")], KM + ORACLE),
    M("wk-holdout-k", "WK3", "the held-out scoring is told k = 1", [(W, "hc0 = contents(pls.ip)[0];", "hc0 = 1;")], KM + ORACLE),
    M("wk-summary-k", "WK3", "the summary of a fit says the iterations where it says k", [(W, "    w = json.put_int(heap, w, ip[0]);", "    w = json.put_int(heap, w, ip[1]);")], KM + ORACLE),
    # ------------------------------------------------------------------ the metrics
    M("wk-inertia-of-absolute-differences", "WK5", "the inertia sums |a - b| of a term with itself", [(D, "inertia = sums.reduce(sums.kind_val(),", "inertia = sums.reduce(sums.kind_abs(),")], KM + ORACLE),
    M("wk-inertia-drops-the-last-row", "WK5", "the inertia leaves out the last row", [(D, "contents(dr)[0..n], contents(dr)[0..n], 0.0, n, contents(bw)", "contents(dr)[0..n], contents(dr)[0..n], 0.0, n - 1, contents(bw)")], KM + ORACLE),
    M("wk-mean-distance-divides-by-the-rows", "WK5", "the mean distance of a cluster is divided by all the rows", [(D, "contents(mw)[c] = total / float_of(cnt);", "contents(mw)[c] = total / float_of(n);")], KM + ORACLE),
    M("wk-mean-of-squared-distances", "WK5", "the mean distance of a cluster is the mean squared distance", [(D, "= sqrt(contents(dr)[i]);", "= contents(dr)[i];")], KM + ORACLE),
    M("wk-sizes-count-twice", "WK5", "a row counts twice in the size of its cluster", [(D, "contents(sw)[c] = contents(sw)[c] + 1;", "contents(sw)[c] = contents(sw)[c] + 2;")], KM + ORACLE),
    M("wk-grouping-offsets", "WK5", "the rows of a cluster are grouped from the start of the previous one", [(D, "contents(tw)[c + 1] = contents(tw)[c] + contents(sr)[c];", "contents(tw)[c + 1] = contents(sr)[c];")], KM + ORACLE),
    M("wk-empty-cluster-has-a-mean", "WK5", "an empty cluster has the mean distance 0", [(D, "let ok = contents(sr)[c] > 0 && common.is_finite(contents(mr)[c]);", "let ok = common.is_finite(contents(mr)[c]);")], KM),
    # ------------------------------------------------------------------ the bounds
    M("wk-k-above-the-rows-by-one", "WK7", "k may be one more than the training rows in the wiring", [(D, "    if k > rows {", "    if k > rows + 1 {")], KM + RULES),
    M("wk-features-bound", "WK7", "the bound on k * features is not compared", [(D, "    if d < 1 || d > common.max_slot() || k > common.max_slot() / d || k * d + k + 1 > common.max_slot() {", "    if d < 1 {")], KM),
    M("wk-features-bound-message", "WK7", "the wiring's bound on k * features is shifted by one", [(D, "k * d + k + 1 > common.max_slot() {\n        return 2;", "k * d + k + 2 > common.max_slot() {\n        return 2;")], KM,
      why="the boundary k * d + k + 1 = 131072 needs k * (d + 1) = 131071, a prime larger than the 8,192 columns and the 1,024 clusters the flags allow: no input reaches the exact boundary; p_models.cho tests the library at it"),
    M("wk-diagnostics-bound", "WK7", "the diagnostics of the iterations are not bounded in the wiring", [(D, "    if iterations + 1 > 16777216 / (k * d + k + 1) {", "    if false {")], KM),
    M("wk-diagnostics-bound-by-one", "WK7", "the diagnostics bound counts the iterations without the final pass", [(D, "    if iterations + 1 > 16777216 / (k * d + k + 1) {", "    if iterations > 16777216 / (k * d + k + 1) {")], KM,
      why="the boundary is at 129 iterations of 1,024 clusters on 126 features, and the first fit that passes it (128 iterations on 1,030 rows) is about 17 gigaflops: a test of it would take minutes; the pair of fits at 129 and 1 pins the bound within a factor 129, and p_models.cho tests the adapter's own copy of the arithmetic"),
    M("wk-adapter-k-above-the-rows", "WK7", "the adapter does not compare k with the rows", [(A, "iters < 1 || iters > common.max_iters() || init < 0 || init > 1 {", "iters < 1 || iters > common.max_iters() || init < 0 || init > 1 || false {"), (A, " || k > n ||", " ||")], KM,
      why="a second line of defence: the wiring compares k with the training rows first (wk-k-above-the-rows-by-one), and the library compares it again in kmeans.fit with the same tag"),
    # ------------------------------------------------------------------ the kind 6 reader
    M("wk-payload-width-not-compared", "WK12", "the payload's width is not compared with the schema's", [(A, "|| w != d ||", "||")], KM + FILES,
      why="a second line of defence: with the width unchecked the size comparison uses the caller's width, and the record reader then compares the width it read (s1) with it and refuses the file with the same tag"),
    M("wk-payload-k-ceiling", "WK12", "the payload's k is not bounded by 1,024", [(A, "k > common.max_k() || w != d", "w != d")], FILES,
      why="a k above 1,024 in a payload cannot have the size the header announces for any width below 131072 / k, and the size comparison that follows refuses the file with the same tag: two checks of one bound"),
    M("wk-payload-size-not-compared", "WK12", "the payload's length is not compared with the size its header announces", [(A, "    if len(blob) != record.kmeans_size(k, d) {", "    if false {")], FILES,
      why="the record reader (decode_f64) refuses a payload shorter than the record, and the comparison of its end with the length of the payload refuses a longer one: the size comparison before the allocation is a second line of defence that keeps the allocation bounded by the file, and no input reaches a case where only it refuses"),
    M("wk-payload-end-not-compared", "WK12", "bytes after the record are accepted", [(A, "|| r.end != len(blob) {", "{")], FILES,
      why="the size comparison before the allocation refuses a payload of another length first (see wk-payload-size-not-compared, whose reason is the same pair)"),
    M("wk-payload-values-not-finite-checked", "WK12", "a centroid that is not a number is accepted", [(A, "        if r.code != 0 || r.rank != 2", "        if r.rank != 2")], FILES,
      why="record.decode_f64 answers a bad value with the end offset of the record's start, so the comparison of the end with the length refuses the file all the same: the code check is a second line of defence"),
    M("wk-decode-says-regression", "WK12", "the decoder says the payload is a regression", [(A, "    return record.decoded(0, k, 0, 4);", "    return record.decoded(0, k, 0, 1);")], KM + ORACLE),
    M("wk-decode-forgets-k", "WK12", "the decoder says the payload has no clusters", [(A, "    return record.decoded(0, k, 0, 4);", "    return record.decoded(0, 0, 0, 4);")], KM + FILES),
    M("wk-plan-k-and-payload-not-compared", "WK12", "a plan that says k = 4 over three centroids is accepted", [(IO, "    if dcode == 0 && plan_k != 0 && plan_k != c0 {", "    if false {")], FILES + KM),
    M("wk-sum-order-of-kind-6-is-3", "WK4", "k-means records the sequential order", [(K, "    if kind == model_tree() {", "    if kind == model_tree() || kind == model_kmeans() {")], KM + ORACLE + FILES),
    # ------------------------------------------------------------------ threads and the seed
    M("wk-threads-minus-one", "WK8", "the fit gets one thread fewer than --threads", [(W, "est.fit_blob(heap, pls.kind, o.threads,", "est.fit_blob(heap, pls.kind, o.threads - 1,")], KM + ORACLE),
    M("wk-threads-reach-the-seed", "WK8", "the seed of the initialisation depends on the thread count", [(A, "threads, xt, xd, k, iters, seed, init,", "threads, xt, xd, k, iters, seed + threads - 1, init,")], KM + ORACLE),
    M("wk-seed-ignored", "WK5", "the seed of the command line is not given to the fit", [(W, "contents(pls.fp), o.seed, ds.c0, ds.c1, o.max_rows, o.max_cols, o.max_cells);\n                    unbox_slice(heap, blob);", "contents(pls.fp), 0, ds.c0, ds.c1, o.max_rows, o.max_cols, o.max_cells);\n                    unbox_slice(heap, blob);")], KM + ORACLE),
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
        print("%d kmeans mutants, %d with an edit that does not apply" % (len(MUTANTS), bad))
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

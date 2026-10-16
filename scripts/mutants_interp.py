#!/usr/bin/env python3
"""Mutants for the gates of docs/interpret.md section 8: a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change (an edit to a source), `ml` rebuilt from the copy with the explicit file list that package mode would pass, and the tests of
the gate it targets run against it (scripts/mutants_cli.py carries the copy, the build and the test runner, and the binaries other than `ml` are carried as they are). KILLED
means a test failed; SURVIVED means every test passed, and the mutant must then carry a reason: an equivalent mutant, shown here, or the gate is strengthened.

    build/venv/bin/python scripts/mutants_interp.py            # all mutants, then the baseline (the unchanged copy must pass)
    build/venv/bin/python scripts/mutants_interp.py --only ID
    python3 scripts/mutants_interp.py --list
    python3 scripts/mutants_interp.py --check                  # every `old` text is found where it is expected (no build, no test)

Run it with the interpreter of build/venv: the permutation gates need numpy.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mutants_cli as C  # noqa: E402

ROOT = C.ROOT
I = "tools/ml/interp.cho"
D = "tools/ml/idump.cho"
P = "tools/ml/perm.cho"
A = "tools/ml/app.cho"

STD = ["test_interp.StandardisedCoefficients"]
MDI = ["test_interp.ImpurityDecrease"]
DUMP = ["test_interp.Dump"]
DESC = ["test_interp.DescribeMember"]
PERM = ["test_interp_oracle.Permutation"]
NOISE = ["test_interp_oracle.Noise"]
REFUSE = ["test_interp.Unsupported", "test_interp_oracle.Permutation"]


def M(id, gate, what, edits, tests, **opt):
    return C.M(id, gate, what, [e if len(e) == 4 else e + (1,) for e in edits], tests, **opt)


MUTANTS = [
    # ------------------------------------------------------------------ I1: the standardised coefficients
    M("i1-coefficient-divided-by-the-deviation", "I1", "standardised = coefficient / deviation", [(I, "var v = wv[j] * sd;", "var v = wv[j] / sd;")], STD),
    M("i1-ddof-one-deviation-is-one", "I1", "a standard scaler with ddof 1 states a unit deviation", [(I, "return (math.sqrt(float_of(n - 1) / float_of(n)), 1);", "return (1.0, 1);")], STD),
    M("i1-provenance-deviation-times-divisor", "I1", "the raw deviation is multiplied by the divisor instead of divided", [(I, "let sd = std / l_div(lin, j);", "let sd = std * l_div(lin, j);")], STD),
    M("i1-provenance-mean-for-deviation", "I1", "the mean field of the provenance record is read for the deviation", [(I, "std = fbits.decode(provsec.field(pv, k, 4));", "std = fbits.decode(provsec.field(pv, k, 3));")], STD),
    M("i1-divisor-not-composed", "I1", "a second scale step replaces the divisor instead of multiplying it", [(I, "div = div * d;", "div = d;")], STD),
    M("i1-unit-lost", "I1", "a standard step does not remember its ddof", [(I, "unit = steps.int_at(st, 2 + k);", "unit = 0;")], STD),
    M("i1-constant-column-unit", "I1", "the deviation of a column the provenance says is constant is not 0", [(I, "if has && std == 0.0 {\n            return (0.0, 1);", "if has && std == 1.0e300 {\n            return (0.0, 1);")], STD),
    M("i1-share-not-normalised", "I1", "the importance is the absolute standardised coefficient, not its share", [(I, "contents(iw)[j] = fabs(contents(zr)[j]) / total;", "contents(iw)[j] = fabs(contents(zr)[j]);")], STD),
    M("i1-rank-ties-to-the-higher-index", "I1", "equal importances rank the higher index first", [(I, "vals[i] == vals[j] && i < j", "vals[i] == vals[j] && i > j")], MDI + STD),
    M("i1-derived-column-gets-a-deviation", "I1", "an encoder's column is given the provenance deviation of its raw column", [(I, "if l_derived(lin, j) {\n        return (0.0, 0);\n    }", "if false {\n        return (0.0, 0);\n    }")],
      STD),
    # ------------------------------------------------------------------ I2: the impurity decrease
    M("i2-gini-without-the-factor-two", "I2", "the Gini identity loses its factor 2", [(I, "let v = (2.0 * (df * df)) /", "let v = (df * df) /")], MDI,
      why="equivalent: the printed importance divides every decrease of a tree by the total of the tree, which divides the constant factor out (the factor 2 is the Gini definition, n G - nL GL - nR GR, and shows only in a raw decrease, which is not printed)"),
    M("i2-gini-class-zero-counts", "I2", "the identity reads the wrong counts (lb * nl - rb * nr)", [(I, "let dd = lb * nr - rb * nl;", "let dd = lb * nl - rb * nr;")], MDI),
    M("i2-gini-symmetric-counts", "I2", "the identity reads the class-0 counts of the children (equivalent: D changes sign and is squared)", [(I, "let dd = lb * nr - rb * nl;", "let dd = (nl - lb) * nr - (nr - rb) * nl;")], MDI,
      why="equivalent: with the class-0 counts D becomes -D and only D squared is used"),
    M("i2-variance-weight", "I2", "the variance reduction weight nL nR / n becomes nL nR / nL", [(I, "((float_of(nl) * float_of(nr)) / float_of(n)) * (d * d)", "((float_of(nl) * float_of(nr)) / float_of(nl)) * (d * d)")], MDI),
    M("i2-leaf-sum-without-count", "I2", "a leaf's sum of the target is its value, not count times value", [(I, "contents(uw)[i] = float_of(node_i(m, 8, trees, total, i)) * node_f(m, 10, trees, total, i);", "contents(uw)[i] = node_f(m, 10, trees, total, i);")], MDI),
    M("i2-node-sum-difference", "I2", "an internal node's sum is the difference of its children's", [(I, "contents(uw)[i] = contents(uw)[lo + node_i(m, 6, trees, total, i)] + contents(uw)[lo + node_i(m, 7, trees, total, i)];",
                                                                       "contents(uw)[i] = contents(uw)[lo + node_i(m, 6, trees, total, i)] - contents(uw)[lo + node_i(m, 7, trees, total, i)];")], MDI),
    M("i2-tree-not-normalised", "I2", "a tree's vector is added to the forest's without dividing by its own total", [(I, "contents(aw)[f] = contents(aw)[f] + contents(tw)[f] / tt;", "contents(aw)[f] = contents(aw)[f] + contents(tw)[f];")], MDI),
    M("i2-forest-divided-by-trees", "I2", "the forest's vector is divided by the number of trees, not renormalised", [(I, "contents(iw)[f] = contents(ar)[f] / s;", "contents(iw)[f] = contents(ar)[f] / float_of(trees);")], MDI),
    M("i2-task-swapped", "I2", "a classification is read as a regression", [(I, "if task != 0 {", "if task == 0 {")], MDI),
    M("i2-split-count-by-two", "I2", "the split counter adds 2", [(I, "contents(sw)[feat] = contents(sw)[feat] + 1;", "contents(sw)[feat] = contents(sw)[feat] + 2;")], MDI),
    M("i2-split-rows-class-zero-only", "I2", "the rows of a classification node are its class-0 count", [(I, "rows = na + nb;", "rows = na;")], MDI),
    M("i2-a-leaf-tree-adds-nan", "I2", "a tree that never splits is divided by its zero total", [(I, "if finite(tt) && tt > 0.0 {", "if tt >= 0.0 {")], MDI),
    M("i2-total-of-an-unsplit-forest", "I2", "the total says 1.0 when no tree splits", [(I, "    if positive {\n        plain = buffer.append(heap, plain, \"1.0\");\n    } else {\n        plain = buffer.append(heap, plain, \"0.0\");\n    }\n    plain = buffer.append(heap, plain, \" trees \");",
                                                                            "    if true {\n        plain = buffer.append(heap, plain, \"1.0\");\n    } else {\n        plain = buffer.append(heap, plain, \"0.0\");\n    }\n    plain = buffer.append(heap, plain, \" trees \");")],
      MDI),
    # ------------------------------------------------------------------ I9: describe
    M("i9-member-renamed", "I9", "describe names the member importancex", [("tools/ml/mdesc.cho", 'w = json.put_key(heap, w, "importance");', 'w = json.put_key(heap, w, "importancex");')], DESC),
    M("i9-member-null-for-trees", "I9", "describe's table is the null of a model with none", [("tools/ml/mdesc.cho", "buffer.bytes(ibr), o);\n                        buffer.drop(heap, payload);", "\"null\", o);\n                        buffer.drop(heap, payload);")], DESC),
    # ------------------------------------------------------------------ I10: the dump
    M("i10-depth-not-increased", "I10", "a child has the depth of its parent", [(D, "contents(dw)[l] = d + 1;", "contents(dw)[l] = d;")], DUMP),
    M("i10-default-side-flipped", "I10", "the side of missing cells is exchanged", [(D, 'if dleft == 1 {\n                    w = json.put_string(heap, w, "left");', 'if dleft == 0 {\n                    w = json.put_string(heap, w, "left");')], DUMP),
    M("i10-cell-of-a-scaled-column", "I10", "a scaled column is given a cell text", [(D, "if interp.l_derived(lin, feat) || interp.l_scaled(lin, feat) {", "if interp.l_derived(lin, feat) {")], DUMP),
    M("i10-decimal-truncated", "I10", "a decimal cell is truncated instead of rounded", [(D, "k = truncate(t + 0.5);", "k = truncate(t);")], DUMP),
    M("i10-tree-bound-off-by-one", "I10", "a tree equal to the bound is refused", [(D, "if n > limit {", "if n >= limit {")], DUMP),
    M("i10-tree-index-off-by-one", "I10", "the tree at the number of trees is read", [(D, "if ti >= trees {", "if ti > trees {")], DUMP),
    M("i10-default-bound", "I10", "the default bound of the dump is 10", [(D, "fn default_nodes() -> [] int {\n    return 10000;", "fn default_nodes() -> [] int {\n    return 10;")], DUMP),
    M("i10-ceiling", "I10", "the ceiling of --max-nodes is one more", [(D, "fn node_ceiling() -> [] int {\n    return 4194304;", "fn node_ceiling() -> [] int {\n    return 4194305;")], DUMP),
    M("i10-raw-centroid-without-offset", "I10", "the raw value of a centroid forgets the offset", [(D, "z * interp.l_div(lin, j) + interp.l_off(lin, j)", "z * interp.l_div(lin, j)")], DUMP),
    M("i10-centroids-transposed", "I10", "the centroid table is read column-major", [(D, "28 + 8 * (c * d + j)", "28 + 8 * (j * k + c)")], DUMP),
    M("i10-intercept-index", "I10", "the intercept is the first coefficient", [(D, "o = verbs.put_double(heap, o, interp.bits(wv[w]));", "o = verbs.put_double(heap, o, interp.bits(wv[0]));")], DUMP),
    M("i10-source-of-a-categorical-column", "I10", "a categorical block has no source column", [(I, "let src = schema_index(sc, catenc.name(st, c));", "let src = 0;")], DUMP + MDI),
    M("i10-cell-int-integral-check", "I10", "an int column's threshold is printed without checking that it is whole (equivalent for the thresholds a fit stores)", [(D, "if float_of(iv) == v {", "if true {")], DUMP,
      why="equivalent: the thresholds of an int column are cells of the training file, whole numbers"),
    # ------------------------------------------------------------------ I6: the permutation
    M("i6-stream-kind", "I6", "the shuffle draws from the stream of the group split", [(P, "kinds.stream_permutation(), rep)", "kinds.stream_group(), rep)")], PERM),
    M("i6-repeat-index", "I6", "every repeat uses the stream of repeat 0", [(P, "kinds.stream_permutation(), rep), j);", "kinds.stream_permutation(), 0), j);")], PERM),
    M("i6-column-index", "I6", "every column uses the stream of column 0", [(P, "kinds.stream_permutation(), rep), j);", "kinds.stream_permutation(), rep), 0);")], PERM),
    M("i6-seed-ignored", "I6", "the seed does not reach the stream", [(P, "splits.stream(o.seed, kinds.stream_permutation(), rep)", "splits.stream(0, kinds.stream_permutation(), rep)")], PERM),
    M("i6-shuffle-bound", "I6", "the shuffle leaves the last row in place", [(P, "splits.shuffle(contents(pw), 0, n, s0)", "splits.shuffle(contents(pw), 0, n - 1, s0)")], PERM),
    M("i6-score-direction", "I6", "a score is reported as permuted minus baseline", [(P, "var dv = base - v;", "var dv = v - base;")], PERM),
    M("i6-log-loss-is-a-score", "I6", "log-loss is read as a score", [(P, "return m == 1 || m == 2 || m == 9;", "return m == 1 || m == 2;")], PERM),
    M("i6-mean-divisor", "I6", "the mean is divided by one repeat too many", [(P, "mean = mean / float_of(reps);", "mean = mean / float_of(reps + 1);")], PERM),
    M("i6-std-divisor", "I6", "the standard deviation is the sample one", [(P, "contents(dw)[s] = math.sqrt(vr / float_of(reps));", "contents(dw)[s] = math.sqrt(vr / float_of(reps - 1));")], PERM),
    M("i6-nothing-permuted", "I6", "the numeric and the categorical columns are not shuffled", [(P, "nfeat, pj, contents(prr)", "nfeat, 0 - 1, contents(prr)"), (P, "ids, pc, contents(prr)", "ids, 0 - 1, contents(prr)")], PERM + NOISE),
    M("i6-numeric-column-not-shuffled", "I6", "a numeric column keeps its order", [(P, "nfeat, pj, contents(prr)", "nfeat, 0 - 1, contents(prr)")], PERM),
    M("i6-categorical-column-not-shuffled", "I6", "a categorical column keeps its order", [(P, "ids, pc, contents(prr)", "ids, 0 - 1, contents(prr)")], PERM),
    M("i6-missing-flag-stays", "I6", "a missing cell stays in its row while the values move", [(P, "if matrix.is_missing(x, src, cols[j]) {", "if matrix.is_missing(x, r, cols[j]) {")], PERM),
    M("i6-target-rotated", "I6", "the metric compares the predictions with the target of the next row (a permuted target)", [(P, "sc = regress.rmse(heap, yv[0..n], pred[0..n], cap);", "sc = regress.rmse(heap, yv[1..n], pred[0..n - 1], cap);")], PERM + NOISE),
    M("i6-baseline-on-the-permuted-order", "I6", "the baseline is scored with the ids permuted", [(P, "if code == 0 && is_cat {", "if code == 0 && is_cat && pj >= 0 {")], PERM),
    # ------------------------------------------------------------------ I12: the command line
    M("i12-repeats-ceiling", "I12", "101 repeats are accepted", [(A, "if v < 1 || v > 100 {", "if v < 1 || v > 101 {")], REFUSE),
    M("i12-flags-without-data", "I12", "--metric and its kind are accepted without --data", [(A, 'if !cli.has(parsed, table, "data") {', "if false {")], REFUSE),
    M("i12-too-few-rows", "I12", "a one-row file is accepted", [(P, "if n < 2 {", "if n < 1 {")], REFUSE),
    M("i12-work-bound-inclusive", "I12", "a run exactly at --max-work is refused", [(P, "n * nf * reps > max_work", "n * nf * reps >= max_work")], PERM),
    M("i12-metric-of-the-other-task", "I12", "a classification metric is accepted for a regression", [(P, "if m == 0 || (task == 1 && m > 3) || (task == 2 && m <= 3) {", "if m == 0 || (task == 2 && m <= 3) {")], PERM),
    M("i12-default-classification-metric", "I12", "the default metric of a classification is accuracy", [(P, "    } else {\n        m = 4;\n    }", "    } else {\n        m = 5;\n    }")], PERM),
    M("i12-default-repeats", "I12", "the default is 4 repeats", [(P, "pub fn default_repeats() -> [] int {\n    return 5;", "pub fn default_repeats() -> [] int {\n    return 4;")], PERM),
    M("i12-no-scale-with-data", "I12", "a table without a deviation refuses the run even with --data", [(P, "if ist == 1 && !rq.data_given {", "if ist == 1 {")], REFUSE + ["test_interp.StandardisedCoefficients"]),
    M("i12-single-class-metric", "I12", "an AUC with one class is a refusal of the metric, not an undefined metric", [(P, " || c5 == rules.single_class()", "")], PERM),
    M("i12-foreign-label", "I12", "a label the model never saw is scored", [(P, "if lab != c0 && lab != c1 && foreign < 0 {", "if false {")], PERM),
    M("i12-stream-kind-registry", "I12", "the registry lists the permutation stream as 12", [("lib/core/kinds.cho", "pub fn stream_permutation() -> [] int {\n    return 11;", "pub fn stream_permutation() -> [] int {\n    return 12;")], PERM),
]


def check():
    """every `old` text occurs exactly once in the file it is aimed at, and differs from its replacement"""
    bad = 0
    ids = set()
    for m in MUTANTS:
        if m["id"] in ids:
            print("INVALID %s: named twice" % m["id"])
            bad += 1
        ids.add(m["id"])
        for f, old, new, nth in m["edits"]:
            text = (ROOT / f).read_text()
            n = text.count(old)
            if n != 1:
                print("INVALID %s: %r occurs %d times in %s (it must occur once)" % (m["id"], old[:70], n, f))
                bad += 1
            if old == new:
                print("INVALID %s: the replacement equals the pattern" % m["id"])
                bad += 1
    print("%d mutants, %d edits that do not apply" % (len(MUTANTS), bad))
    return 1 if bad else 0


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        for m in MUTANTS:
            print("%-44s %s  %s" % (m["id"], m["gate"], m["what"]))
        return 0
    if "--check" in argv:
        return check()
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    results = []
    for m in MUTANTS:
        if only and m["id"] != only:
            continue
        verdict, detail = C.run_one(m)
        results.append((m, verdict, detail))
        print("%-9s %-44s %s  %s" % (verdict, m["id"], m["gate"], detail), flush=True)
    if not only:
        v, d = baseline()
        print("%-9s %-44s %s  %s" % (v, "baseline (the unchanged tree)", "-", d), flush=True)
    survivors = [(m, d) for m, v, d in results if v == "SURVIVED"]
    invalid = [m["id"] for m, v, d in results if v == "INVALID"]
    killed = sum(1 for m, v, d in results if v == "KILLED")
    print("\n%d mutants: %d killed, %d survived, %d invalid" % (len(results), killed, len(survivors), len(invalid)))
    for m, d in survivors:
        print("SURVIVOR %s: %s" % (m["id"], m["opt"].get("why", "NO REASON GIVEN")))
    unexplained = [m["id"] for m, d in survivors if not m["opt"].get("why")]
    return 1 if (unexplained or invalid) else 0


def baseline():
    """the unchanged copy passes the tests of the gates this script judges"""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        C.copy_tree(tree)
        binary, log = C.build(tree)
        if binary is None:
            return "INVALID", "build failed: " + log[:400]
        code, out = C.run_tests(tree, sorted(set(t for m in MUTANTS for t in m["tests"])))
        return ("PASSED", "") if code == 0 else ("FAILED", out[-600:])


if __name__ == "__main__":
    sys.exit(main())

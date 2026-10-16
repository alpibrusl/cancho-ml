#!/usr/bin/env python3
"""Mutants for the gates of docs/splits.md section 9: a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one edit to a source, `ml` rebuilt from the copy, and the tests it targets run against it:
`test_splits` (the oracle gates A1, B1, C1, D1 on their quick sets, the hostile data and the arguments of F1) and, for a refusal,
`test_rules` (the fixture of every tag, with its repair). KILLED means a test failed; SURVIVED means every test passed, and the mutant
then carries a reason. The planted leak of gate C2 is the mutant `leak-steps-fitted-on-all-rows`.

    python3 scripts/mutants_split.py              # all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_split.py --only ID    # one (a substring)
    python3 scripts/mutants_split.py --list
    python3 scripts/mutants_split.py --check      # every `old` text is found exactly once (no build, no test)
"""

import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mutants_cli as M  # noqa: E402

ROOT = M.ROOT
S, R = ["test_splits"], ["test_rules.Rules.test_every_fixture", "test_rules.Rules.test_retry_repairs_work_and_widen_nothing"]
SR = S + R


def mut(id, what, file, old, new, tests=S, why=None):
    return dict(id=id, what=what, file=file, old=old, new=new, tests=tests, why=why)


PLAN = "tools/ml/splitplan.cho"
COL = "tools/ml/splitcol.cho"
SPEC = "tools/ml/splitspec.cho"
WIRE = "tools/ml/splitwire.cho"
CV = "tools/ml/cv.cho"

MUTANTS = [
    # ---------------------------------------------------------------- the assignment
    mut("strat-slot-phase", "the stratified holdout deals its test slots one position late", PLAN, "if (t + 1) * n_test / n - t * n_test / n == 1 {", "if (t + 2) * n_test / n - (t + 1) * n_test / n == 1 {"),
    mut("strat-holdout-stream-index", "both classes of the stratified holdout shuffle with the same stream", PLAN, "splits.shuffle(a, n0, n1, splits.stream(seed, splits.kind_strat_holdout(), 1));", "splits.shuffle(a, n0, n1, splits.stream(seed, splits.kind_strat_holdout(), 0));"),
    mut("strat-holdout-kind", "the stratified holdout draws from the stream of the k-fold", PLAN, "splits.shuffle(a, 0, n0, splits.stream(seed, splits.kind_strat_holdout(), 0));", "splits.shuffle(a, 0, n0, splits.stream(seed, splits.kind_kfold(), 0));"),
    mut("strat-holdout-last-class-whole", "a class that would have no training row is not refused", PLAN, "        if c1test == n1 {\n            return (2, hi, n1);\n        }", "        if c1test == n1 + 1 {\n            return (2, hi, n1);\n        }", tests=SR),
    mut("group-shuffle-kind", "the groups are shuffled with the stream of the k-fold", PLAN, "splits.shuffle(a, 0, g, splits.stream(seed, splits.kind_group(), rep));", "splits.shuffle(a, 0, g, splits.stream(seed, splits.kind_kfold(), rep));"),
    mut("group-holdout-stop-rule", "the test side takes a group while it has at most n_test rows", PLAN, "if rows < n_test && taken < groups - 1 {", "if rows <= n_test && taken < groups - 1 {"),
    mut("group-holdout-keeps-no-train", "the test side may take every group", PLAN, "if rows < n_test && taken < groups - 1 {", "if rows < n_test {"),
    mut("group-cv-tie-to-last-fold", "an equal load goes to the last fold, not the first", PLAN, "if contents(lw)[f] < contents(lw)[best] {", "if contents(lw)[f] <= contents(lw)[best] {"),
    mut("group-cv-repeat-ignored", "every repeat of a group cv shuffles with the stream of repeat 0", PLAN, "let perm = shuffled_groups(heap, groups, seed, rep);", "let perm = shuffled_groups(heap, groups, seed, 0);"),
    mut("group-cv-first-fold-always", "a group goes to the first fold", PLAN, "                            contents(ww)[grp] = best;", "                            contents(ww)[grp] = 0;"),
    mut("time-block-label", "a time block is labelled with its own number, not the number minus 1", PLAN, "out[order[p]] = b - 1;", "out[order[p]] = b;"),
    mut("time-holdout-newest-trains", "the newest rows train and the oldest test (by flipping the cut)", PLAN, "var p = n - n_test;", "var p = 0;"),
    mut("time-ties-not-stable", "equal keys are not kept in the order of the file", COL, "    return k2[a] <= k2[b];", "    return k2[a] < k2[b];"),
    mut("time-nanoseconds-ignored", "the fraction of a second is not compared", COL, "    return k2[a] <= k2[b];", "    return true;"),
    mut("time-month-day-swapped", "the date key puts the day before the month", COL, "(((y * 100 + mo) * 100 + d) * 100 + hh) * 10000", "(((y * 100 + d) * 100 + mo) * 100 + hh) * 10000"),
    mut("time-leap-year", "a year divisible by 100 is a leap year", COL, "if y % 4 == 0 && y % 100 != 0 || y % 400 == 0 {", "if y % 4 == 0 {", tests=S),
    mut("group-rank-order-reversed", "the groups are ranked in descending byte order", COL, "return bytes.compare(text[off[a]..off[a + 1]], text[off[b]..off[b + 1]]) <= 0;", "return bytes.compare(text[off[a]..off[a + 1]], text[off[b]..off[b + 1]]) >= 0;"),
    mut("group-cells-trimmed-of-quotes", "a quoted doubled quote is dropped, not unescaped", COL, "                    if keep {\n                        out = buffer.push(heap, out, byte_of('\"'));\n                    }\n                    pos = pos + 2;", "                    pos = pos + 2;"),
    mut("column-cr-kept", "a CR before the LF stays in the last cell of a line", COL, "        if end > start && pos < n && int_of(data[pos]) == 10 && int_of(data[pos - 1]) == 13 {\n            end = end - 1;\n        }", ""),
    mut("canonical-seed-token", "the canonical text spells the seed another way", SPEC, "b = buffer.append(heap, b, \" seed=\");", "b = buffer.append(heap, b, \" seed:\");"),
    mut("canonical-trailing-zeros", "the canonical train fraction keeps its trailing zeros", SPEC, "while last > 1 && contents(cw)[last - 1] == 0 {", "while false {"),
    mut("canonical-name-digit-escaped", "the digit 0 of a column name is written as %30", SPEC, "c >= '0' && c <= '9' ||", "c >= '1' && c <= '9' ||"),
    mut("emit-negative-vector", "a negative fold (the first block of a time split) is written without its sign", WIRE, "        if vals[i] < 0 {\n            b = buffer.push(heap, b, byte_of('-'));", "        if false {\n            b = buffer.push(heap, b, byte_of('-'));"),
    # ---------------------------------------------------------------- the refusals (fixtures of test_rules)
    mut("refuse-combination", "two strategies are accepted", SPEC, "    if n > 1 {\n        var a: &static [byte] = \"--stratify\";", "    if n > 7 {\n        var a: &static [byte] = \"--stratify\";", tests=R),
    mut("refuse-folds-range", "fifty folds are accepted", SPEC, "if sp.folds < 2 || sp.folds > folds_max() {", "if sp.folds < 2 || sp.folds > 99 {", tests=R),
    mut("refuse-repeats-with-time", "repeats above 1 are accepted with a time split", SPEC, "} else if sp.repeats > 1 && sp.by_given && by_ok(argv.text(src, \"split-by\")) {", "} else if false {", tests=R),
    mut("refuse-split-with-cv", "cv accepts --split", SPEC, "        if argv.has(src, \"split\") {\n            e = combination(heap, e, \"--split\", \"cv\");\n        }", "", tests=R),
    mut("refuse-stratify-target", "a regression is stratified", WIRE, "    if strategy == 1 && task != 2 {\n        return (assign, false,", "    if false {\n        return (assign, false,", tests=R),
    mut("refuse-stratify-target-holdout", "a regression is stratified in a holdout", WIRE, "    if strategy == 1 && task != 2 {\n        return (sel, 0, false,", "    if false {\n        return (sel, 0, false,", tests=R),
    mut("refuse-column-target", "the target may be the group column", WIRE, "        if bytes.equal(col, tg) {", "        if false {", tests=R),
    mut("refuse-emit-bound", "no bound on --emit-folds", SPEC, "return 1048576;", "return 1073741824;", tests=R),
    mut("refuse-missing-cell", "an empty group cell is a group of its own", COL, "        if off[i + 1] == off[i] {\n            return i + 1;", "        if false {\n            return i + 1;", tests=SR),
    mut("refuse-offset-reason", "an offset is reported as a form", COL, "                        return (3, 0, 0, 0);", "                        return (2, 0, 0, 0);"),
    mut("refuse-stratum-cv", "a class smaller than the folds goes to the library", PLAN, "        if n0 < k {\n            return (2, lo, n0);", "        if n0 < 0 {\n            return (2, lo, n0);", tests=R),
    mut("refuse-stratum-cv-high-class", "the larger class smaller than the folds goes to the library", PLAN, "        if n1 < k {\n            return (2, hi, n1);", "        if n1 < 0 {\n            return (2, hi, n1);", tests=R),
    mut("refuse-too-few-groups-cv", "fewer groups than folds are accepted", PLAN, "        if groups < k {", "        if groups < 0 {", tests=SR),
    mut("repair-stratify-removed", "the repair of ml.split.stratify-target keeps a malformed --stratify= argument", SPEC, "    w = retry_edit(heap, w, src, \"--stratify\", false, false, \"\");", "    w = retry_edit(heap, w, src, \"--stratify\", false, true, \"\");", tests=R),
    mut("repair-folds-not-lowered", "the repair of a small class keeps --folds", SPEC, "    if cv && rows >= 2 {\n        var v = buffer.push_nat(heap, buffer.empty(heap, 8), rows);", "    if false {\n        var v = buffer.push_nat(heap, buffer.empty(heap, 8), rows);", tests=R),
    mut("repair-emit-kept", "the repair of ml.split.emit-too-large keeps --emit-folds", SPEC, "    w = retry_edit(heap, w, src, \"--emit-folds\", false, false, \"\");", "    w = retry_edit(heap, w, src, \"--emit-folds\", false, true, \"\");", tests=R),
    mut("refuse-single-class-fold", "a training fold with one class is fitted", CV, "                if lacking {", "                if false {", tests=SR),
    # ---------------------------------------------------------------- the aggregate
    mut("agg-population-std", "the standard deviation divides by m, not m - 1", CV, "return (mean, sqrt(ss / float_of(m - 1)));", "return (mean, sqrt(ss / float_of(m)));"),
    mut("agg-mean-divisor", "the mean divides by m + 1", CV, "mean = total / float_of(m);", "mean = total / float_of(m + 1);"),
    mut("agg-deviation-not-centred", "the squares are summed about 0, not about the mean", CV, "ss = sums.reduce(sums.kind_dev(), vals, vals, mean,", "ss = sums.reduce(sums.kind_dev(), vals, vals, 0.0,"),
    mut("agg-fold-order", "the folds are aggregated in reverse", CV, "let frag = frags[ends[k]..ends[k + 1]];", "let frag = frags[ends[count - 1 - k]..ends[count - k]];"),
    # ---------------------------------------------------------------- the leak
    mut("leak-steps-fitted-on-all-rows", "the steps of the plan are fitted on all rows, the estimator on the training rows (C2)", "tools/ml/wire.cho",
        "                    run.drop_fitted(heap, fitted);\n                    matrix.drop(heap, xt);\n                    fitted = f2;\n                    xt = xt2;\n                    if flt.code != 0 {\n                        e = wiredata.code_error(heap, e, flt.code, flt.a, flt.b, flt.c);",
        "                    var xall = matrix.empty(heap);\n                    borrow featcols as &fcr in {\n                        let (xm, callcode) = wiredata.take(heap, ds.x, sel, 0 - 1, contents(fcr), nfeat, o.max_rows, o.max_cols, o.max_cells);\n                        matrix.drop(heap, xall);\n                        xall = xm;\n                    }\n"
        "                    var leaked = run.empty_fitted(heap);\n                    var xtl = matrix.empty(heap);\n"
        "                    borrow xall as &xar in {\n"
        "                        let (fl, xl, fault) = run.fit(heap, pls.pl, nr, xar, o.max_rows, o.max_cols, o.max_cells);\n"
        "                        borrow fl as &flr in {\n"
        "                            let (xt3, unk3, f3) = run.apply_steps(heap, buffer.bytes(flr.payload), xr, o.max_rows, o.max_cols, o.max_cells);\n"
        "                            matrix.drop(heap, xtl);\n                            xtl = xt3;\n                        }\n"
        "                        run.drop_fitted(heap, leaked);\n                        leaked = fl;\n                        matrix.drop(heap, xl);\n                    }\n"
        "                    matrix.drop(heap, xall);\n                    run.drop_fitted(heap, f2);\n                    matrix.drop(heap, xt2);\n"
        "                    run.drop_fitted(heap, fitted);\n                    matrix.drop(heap, xt);\n                    fitted = leaked;\n                    xt = xtl;\n                    if flt.code != 0 {\n                        e = wiredata.code_error(heap, e, flt.code, flt.a, flt.b, flt.c);", tests=S),
    mut("leak-training-on-all-rows", "every fold is trained on all rows, the held-out ones included", CV, "wire.train(heap, src, dr, cir, cvr, pr, contents(sr), 0, o, retry_ok, e);", "wire.train(heap, src, dr, cir, cvr, pr, contents(sr), 0 - 1, o, retry_ok, e);"),
    mut("leak-time-future-in-training", "a time fold trains on the blocks after the one it tests", CV, "} else if strategy == 3 && a > f {", "} else if strategy == 3 && a > f + 1 {"),
]


def check():
    bad = []
    seen = set()
    for m in MUTANTS:
        if m["id"] in seen:
            bad.append("%s: named twice" % m["id"])
        seen.add(m["id"])
        p = ROOT / m["file"]
        if not p.exists():
            bad.append("%s: no file %s" % (m["id"], m["file"]))
            continue
        n = p.read_text().count(m["old"])
        if n != 1:
            bad.append("%s: the pattern occurs %d times in %s (must be 1)" % (m["id"], n, m["file"]))
        if m["old"] == m["new"]:
            bad.append("%s: replacement equals pattern" % m["id"])
    return bad


def run_tests(tree, targets):
    env = dict(os.environ, ML_BIN=str(tree / "build" / "ml"), PYTHONDONTWRITEBYTECODE="1", CANCHO=M.compiler(), SPLIT_QUICK="1")
    p = subprocess.run([sys.executable, "-m", "unittest", *targets], capture_output=True, text=True, cwd=tree / "tests" / "conformance", env=env, timeout=1800)
    return p.returncode, p.stdout + p.stderr


def run_one(m):
    with tempfile.TemporaryDirectory() as td:
        tree = pathlib.Path(td) / "t"
        M.copy_tree(tree)
        if m is not None:
            f = tree / m["file"]
            f.write_text(f.read_text().replace(m["old"], m["new"], 1))
        binary, log = M.build(tree)
        if binary is None:
            return "INVALID", "build failed: " + log[:600]
        code, out = run_tests(tree, m["tests"] if m else SR)
        return ("SURVIVED" if code == 0 else "KILLED"), out[-500:]


def main():
    if "--list" in sys.argv:
        for m in MUTANTS:
            print("%-38s %s" % (m["id"], m["what"]))
        return 0
    bad = check()
    if bad:
        print("\n".join(bad))
        return 1
    if "--check" in sys.argv:
        print("mutants_split: %d mutants, every pattern applies exactly once" % len(MUTANTS))
        return 0
    if "--baseline" in sys.argv:
        verdict, tail = run_one(None)
        print("baseline (the unchanged copy): %s" % ("PASS" if verdict == "SURVIVED" else "FAILS\n" + tail))
        return 0 if verdict == "SURVIVED" else 1
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    killed, survived, invalid = 0, [], []
    for m in MUTANTS:
        if only and only not in m["id"]:
            continue
        verdict, tail = run_one(m)
        if verdict == "KILLED":
            killed += 1
            print("%-38s killed" % m["id"], flush=True)
        elif verdict == "INVALID":
            invalid.append(m["id"])
            print("%-38s INVALID\n%s" % (m["id"], tail), flush=True)
        else:
            survived.append(m["id"])
            print("%-38s SURVIVED%s" % (m["id"], (" (explained: %s)" % m["why"]) if m["why"] else ""), flush=True)
    if not only:
        verdict, tail = run_one(None)
        print("baseline (the unchanged copy): %s" % ("PASS" if verdict == "SURVIVED" else "FAILS\n" + tail), flush=True)
    unexplained = [i for i in survived if not next(m for m in MUTANTS if m["id"] == i)["why"]]
    print("mutants_split: %d killed, %d survived (%d explained), %d invalid" % (killed, len(survived), len(survived) - len(unexplained), len(invalid)))
    return 1 if (unexplained or invalid) else 0


if __name__ == "__main__":
    sys.exit(main())

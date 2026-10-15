#!/usr/bin/env python3
"""Mutants for the refusal of a prediction that is not a finite number (docs/nonfinite.md section 6): a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change (the check removed, the check over the printed window only, a wrong row, a wrong column, a wrong exit code, the NaN or the infinity not
detected, the distance not checked, the checks of `eval` removed, the server's status for the rule, a registry entry), `ml`, `ml-predict` and `ml-serve` rebuilt from the copy, and the tests
or scripts of the gate it targets run against them. KILLED: a test or a script failed. SURVIVED: every one passed, and the mutant must carry a reason (`why`).

    python3 scripts/mutants_nonfinite.py              # all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_nonfinite.py --only ID
    python3 scripts/mutants_nonfinite.py --list
    python3 scripts/mutants_nonfinite.py --check      # every `old` text is found where it is expected (no build, no test)

Run it on a frozen copy of the tree when `cancho install` may run in the original (it rewrites build/deps).
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_cli as base  # noqa: E402

ROOT = base.ROOT
IO = "tools/ml/wireio.cho"
D = "tools/ml/wiredata.cho"
V = "tools/ml/verbs.cho"
SRV = "tools/ml-serve/srv.cho"
SS = "tools/ml-serve/sspec.cho"
TMPL = "tools/ml/spec.tmpl"
REF = "lib/core/refuse.cho"
ALL = ["test_nonfinite"]


def M(id, gate, what, edits, tests=(), cmds=(), **opt):
    return dict(id=id, gate=gate, what=what, edits=[e if len(e) == 4 else (*e, 1) for e in edits], tests=list(tests), cmds=[list(c) for c in cmds], opt=opt)


MUTANTS = [
    # ------------------------------------------------------------------ N1, N5: the check of predict
    M("nf-check-removed", "N1", "predict never refuses a row that is not finite (the door still stops the cell)", [(IO, "        if badrow >= 0 {\n            code = refuse.predict_non_finite();", "        if false {\n            code = refuse.predict_non_finite();")], ["test_nonfinite.Reproducer", "test_nonfinite.Agreement"]),
    M("nf-check-and-door-removed", "N1", "no check and a door that lets an infinity through: the text of the issue again",
      [(IO, "        if badrow >= 0 {\n            code = refuse.predict_non_finite();", "        if false {\n            code = refuse.predict_non_finite();"), (V, "    if !bits_finite(bits) {\n        return (into, false);\n    }\n    return (flt.put_float", "    return (flt.put_float")],
      ["test_nonfinite.Reproducer", "test_nonfinite.Fuzz"]),
    M("nf-prefix-of-the-window-only", "N5", "the scan stops at the end of the page", [(IO, "wiredata.first_bad_score(task, contents(pq0), contents(qq0), total, with_distance);", "wiredata.first_bad_score(task, contents(pq0), contents(qq0), end, with_distance);")], ["test_nonfinite.Paging"]),
    M("nf-the-window-only", "N5", "the scan reads the rows of the page and no others", [(IO, "wiredata.first_bad_score(task, contents(pq0), contents(qq0), total, with_distance);", "wiredata.first_bad_score(task, contents(pq0)[start..end], contents(qq0)[start..end], end - start, with_distance);")], ["test_nonfinite.Paging"]),
    M("nf-row-is-zero-based", "N1", "the row in detail is the index, not the number", [(IO, "wiredata.non_finite_prediction(heap, e, brow + 1,", "wiredata.non_finite_prediction(heap, e, brow,")], ALL),
    M("nf-row-of-the-page", "N5", "the row in detail is relative to the page", [(IO, "wiredata.non_finite_prediction(heap, e, brow + 1,", "wiredata.non_finite_prediction(heap, e, brow + 1 - o.from,")], ["test_nonfinite.Paging"]),
    M("nf-column-always-prediction", "N2", "the column in detail is prediction whatever the task", [(D, "pub fn column_word(c: int) -> [] &static [byte] {\n    if c == 1 {", "pub fn column_word(c: int) -> [] &static [byte] {\n    if c == 9 {")], ["test_nonfinite.Clustering", "test_nonfinite.Agreement"]),
    M("nf-wrong-exit-code", "N1", "the table says exit 2 for the rule", [(TMPL, "ml.predict.non-finite|8|never|", "ml.predict.non-finite|2|never|")], ["test_nonfinite.Reproducer", "test_tables"], regen=True),
    M("nf-no-detail-row", "N1", "the detail names no row", [(D, "    w = fail.detail_int(heap, w, \"row\", row);\n    w = fail.detail_str(heap, w, \"column\", column);\n    return fail.add(heap, errs, w);\n}\n\n// the row of the file that is the k-th", "    w = fail.detail_str(heap, w, \"column\", column);\n    return fail.add(heap, errs, w);\n}\n\n// the row of the file that is the k-th")], ["test_nonfinite.Reproducer"]),
    # ------------------------------------------------------------------ N4: the NaN and the infinity
    M("nf-nan-passes", "N4", "a NaN is taken for a finite value", [(D, "        if !common.is_finite(v[i]) {\n            return i;", "        if v[i] - v[i] != 0.0 && v[i] == v[i] {\n            return i;")], ["test_nonfinite.Nan", "test_nonfinite.Fuzz"]),
    M("nf-infinity-passes", "N1", "only a NaN is refused", [(D, "        if !common.is_finite(v[i]) {\n            return i;", "        if v[i] != v[i] {\n            return i;")], ["test_nonfinite.Reproducer", "test_nonfinite.Fuzz"]),
    M("nf-probability-not-read", "N4", "a classification reads the label's value and not the probability", [(D, "    let a = first_non_finite(pred, n);\n    let b = first_non_finite(prob, n);", "    let a = first_non_finite(pred, n);\n    let b = 0 - 1;")], ["test_nonfinite.Nan"], why="the logistic NaN reaches both vectors (the label comes from the same score), so reading the probability adds nothing the label does not already show for the fixture; the row of a probability alone is not constructible from a logistic model"),
    M("nf-label-not-read", "N4", "a classification reads the probability and not the value the label is chosen from", [(D, "    let a = first_non_finite(pred, n);\n    let b = first_non_finite(prob, n);", "    let a = 0 - 1;\n    let b = first_non_finite(prob, n);")], ["test_nonfinite.Nan"], why="same reason: the two vectors are non-finite together for every input the fixtures reach"),
    # ------------------------------------------------------------------ N2: distances
    M("nf-distance-not-checked", "N2", "predict --distance prints the squared distance without a check", [(D, "        if dist {\n            row = first_non_finite(prob, n);", "        if false {\n            row = first_non_finite(prob, n);")], ["test_nonfinite.Clustering", "test_nonfinite.Agreement"]),
    M("nf-distance-always-checked", "N2", "predict without --distance computes nothing but checks the labels' vector as a distance", [(D, "        if dist {\n            row = first_non_finite(prob, n);", "        if true {\n            row = first_non_finite(prob, n);")], ["test_nonfinite.Clustering"], why="the vector of distances is zero when it is not computed, so checking it always cannot refuse: the check has no effect without --distance"),
    M("nf-eval-kmeans-not-checked", "N2", "eval of k-means refuses nothing", [(D, "    if code == 0 && strict && n >= 1 {", "    if code == 0 && false && n >= 1 {")], ["test_nonfinite.Clustering"]),
    M("nf-eval-regression-not-checked", "N1", "eval of a regression has no check (the metrics library refuses it as ml.input.non-finite)", [(D, "            if strict {\n                bad = first_non_finite(contents(pr), n);", "            if false {\n                bad = first_non_finite(contents(pr), n);")], ["test_nonfinite.Paging"]),
    M("nf-eval-classification-not-checked", "N4", "eval of a classification has no check", [(D, "        if code == 0 && strict {\n            borrow pred as &pr0 in {", "        if code == 0 && false {\n            borrow pred as &pr0 in {")], ["test_nonfinite.Nan"]),
    M("nf-eval-row-of-the-held-out-rows", "N5", "eval with --split names the row among the held-out rows", [(D, "    if want < 0 {\n        return k;\n    }\n    var i = 0;", "    if want < 100 {\n        return k;\n    }\n    var i = 0;")], ["test_nonfinite.Paging"]),
    M("nf-eval-passes-the-wrong-split", "N5", "the row of the file counts the held-out rows from the wrong end", [(D, "            if seen == k {\n                return i;", "            if seen == k + 1 {\n                return i;")], ["test_nonfinite.Paging"]),
    # ------------------------------------------------------------------ N6: the server
    M("nf-server-status-missing", "N6", "status_of does not list the rule: the server answers 500", [(SRV, 'if app.listed("ml.predict.non-finite,ml.schema.mismatch,', 'if app.listed("ml.schema.mismatch,')], ["test_nonfinite.Agreement", "test_serve.Statuses"]),
    M("nf-server-guard-old-tag", "N6", "the server's guard answers with the old tag", [(SRV, '422, "ml.predict.non-finite", "the model\'s answer', '422, "serve.non-finite-prediction", "the model\'s answer')], ["test_nonfinite.Agreement"], why="the guard is not reachable: ml predict refuses a row that is not finite before the server sees a document with a cell that is not a number (docs/nonfinite.md section 4)"),
    M("nf-server-rules-lack-the-tag", "N8", "ml-serve introspect does not list the rule", [(SS, "serve.listen;ml.predict.non-finite;serve.internal", "serve.listen;serve.internal")], ["test_nonfinite.Agreement"]),
    # ------------------------------------------------------------------ N8, N9: the registry and the door
    M("nf-tag-renamed-in-the-core", "N8", "refuse.cho names the tag otherwise", [(REF, 'return "ml.predict.non-finite";', 'return "ml.predict.nonfinite";')], (), cmds=[["python3", "scripts/check_registry.py"]]),
    M("nf-tag-missing-in-design", "N8", "design.md 8.1 lacks code 21", [("docs/design.md", "| 21 | `ml.predict.non-finite` | 8 | `ml.refuse` |\n", "")], (), cmds=[["python3", "scripts/check_registry.py"]]),
    M("nf-direct-put-float", "N9", "a call of flt.put_float outside the door", [(IO, "    let (b, fin) = verbs.put_float_text(heap, buffer.empty(heap, 32), bits_of(v));", "    var b = flt.put_float(heap, buffer.empty(heap, 32), csvcell.bits_of_key(bits_of(v)));\n    let fin = true;")], (), cmds=[["python3", "scripts/check_nonfinite.py"]]),
    M("nf-door-without-the-test", "N9", "the door calls flt.put_float without testing the bits", [(V, "    if !bits_finite(bits) {\n        return (into, false);\n    }\n    return (flt.put_float", "    return (flt.put_float")], (), cmds=[["python3", "scripts/check_nonfinite.py"]]),
    M("nf-door-lets-bits-through", "N9", "bits_finite says yes to an exponent field of all ones", [(V, "    return (bits >> 52 & 2047) != 2047;", "    return (bits >> 52 & 2047) != 2046;")], ["test_nonfinite.Fuzz"], why="the door is behind the scan: no path hands it a non-finite double (docs/nonfinite.md section 5), so a door that does not stop one is not seen by an answer; check_nonfinite.py gates its text (the mutant above) and the pair `nf-check-and-door-removed` shows the door alone"),
]


def regen(tree):
    for s in ("gen_models.py", "schemas.py", "manifest.py"):
        r = subprocess.run([sys.executable, str(tree / "scripts" / s)], capture_output=True, text=True, cwd=tree, env=dict(os.environ, CANCHO=base.compiler()))
        if r.returncode != 0 and s != "manifest.py":
            return s + ": " + r.stdout[-300:] + r.stderr[-300:]
    return None


def build_all(tree):
    log = ""
    for b in ("ml", "ml-predict", "ml-serve"):
        p = subprocess.run([base.compiler(), "build", "--bin", b], capture_output=True, text=True, cwd=tree)
        log += p.stdout[-300:] + p.stderr[-300:]
        if p.returncode != 0:
            return False, "build %s failed: %s" % (b, log)
    return True, log


def env_for(tree):
    return dict(os.environ, ML_BIN=str(tree / "build" / "ml"), ML_PREDICT_BIN=str(tree / "build" / "ml-predict"), ML_SERVE_BIN=str(tree / "build" / "ml-serve"), PYTHONDONTWRITEBYTECODE="1", CANCHO=base.compiler())


def run_tests(tree, targets):
    p = subprocess.run([sys.executable, "-m", "unittest", *targets], capture_output=True, text=True, cwd=tree / "tests" / "conformance", env=env_for(tree))
    return p.returncode, p.stdout + p.stderr


def run_one(m):
    with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        base.copy_tree(tree)
        problems = base.apply_edits(tree, m)
        if problems:
            return "INVALID", "; ".join(problems)
        if m["opt"].get("regen"):
            bad = regen(tree)
            if bad:
                return "INVALID", bad
        # scripts that read the sources only
        for cmd in m["cmds"]:
            argv = [sys.executable if a == "python3" else a for a in cmd]
            r = subprocess.run(argv, capture_output=True, text=True, cwd=tree, env=env_for(tree))
            if r.returncode != 0:
                out = (r.stdout + r.stderr).strip()
                return "KILLED", "%s: %s" % (cmd[-1], out.splitlines()[0][:200] if out else "exit %d" % r.returncode)
        if not m["tests"]:
            return "SURVIVED", ""
        ok, log = build_all(tree)
        if not ok:
            return "INVALID", log[:500]
        code, out = run_tests(tree, m["tests"])
        if code != 0:
            f = base.failing_tests(out)
            return "KILLED", "tests: " + ", ".join(f[:4]) + ("" if f else out.strip().splitlines()[-1][:200])
        return "SURVIVED", ""


def baseline():
    with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        base.copy_tree(tree)
        ok, log = build_all(tree)
        if not ok:
            return "INVALID", log[:500]
        code, out = run_tests(tree, ALL + ["test_serve.Statuses"])
        for cmd in (["check_nonfinite.py"], ["check_registry.py"]):
            r = subprocess.run([sys.executable, str(tree / "scripts" / cmd[0])], capture_output=True, text=True, cwd=tree)
            if r.returncode != 0:
                return "FAILED", cmd[0] + ": " + (r.stdout + r.stderr)[-300:]
        return ("PASSED", "") if code == 0 else ("FAILED", out[-600:])


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        for m in MUTANTS:
            print("%-40s %s  %s" % (m["id"], m["gate"], m["what"]))
        return 0
    if "--check" in argv:
        bad = 0
        with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
            tree = pathlib.Path(td) / "tree"
            base.copy_tree(tree)
            ids = [m["id"] for m in MUTANTS]
            if len(set(ids)) != len(ids):
                print("two mutants with one id")
                bad += 1
            for m in MUTANTS:
                problems = base.apply_edits(tree, m)
                if problems:
                    print("INVALID %s: %s" % (m["id"], "; ".join(problems)))
                    bad += 1
                for item in m["edits"]:
                    shutil.copy(ROOT / item[0], tree / item[0])
        print("%d non-finite mutants, %d with an edit that does not apply" % (len(MUTANTS), bad))
        return 1 if bad else 0
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    results = []
    for m in MUTANTS:
        if only and m["id"] != only:
            continue
        verdict, detail = run_one(m)
        results.append((m, verdict, detail))
        print("%-9s %-40s %s  %s" % (verdict, m["id"], m["gate"], detail), flush=True)
    if not only:
        v, d = baseline()
        print("%-9s %-40s %s  %s" % (v, "baseline (the unchanged tree)", "-", d), flush=True)
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

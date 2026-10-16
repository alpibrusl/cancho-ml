#!/usr/bin/env python3
"""Mutants for the wiring stage (docs/wiring.md section 1): a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change to the new code (tools/ml/wire*.cho, the reader's mode 9, the generator, the template of the tables), the binary
rebuilt from the copy, and the tests of the gates it targets run against it. KILLED: a test failed. SURVIVED: every test passed, and the mutant must carry a
reason (`why`). The machinery is that of scripts/mutants_cli.py.

    python3 scripts/mutants_wire.py              all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_wire.py --only ID
    python3 scripts/mutants_wire.py --list
    python3 scripts/mutants_wire.py --check      every `old` text is found exactly once where it is expected (no build, no test)
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

ROOT = mc.ROOT
W = "tools/ml/wire.cho"
IO = "tools/ml/wireio.cho"
D = "tools/ml/wiredata.cho"
G = "scripts/gen_models.py"
T = "tools/ml/spec.tmpl"
C = "tools/ml/csv.cho"

# id, gate, what, edits [(file, old, new, nth)], tests; options: gen (run gen_models.py and schemas.py after the edits), why
ORACLE = ["test_wire_oracle"]
FILE = ["test_wire_file"]
RULES = ["test_rules"]
DET = ["test_determinism"]


def M(id, gate, what, edits, tests, **opt):
    return dict(id=id, gate=gate, what=what, edits=[e if len(e) == 4 else (*e, 1) for e in edits], tests=tests, opt=opt)


MUTANTS = [
    # ------------------------------------------------------------------ the plan: flags to text
    M("w-step-separator-is-a-comma", "W3", "the steps of --step are separated by commas, not semicolons", [(W, "if int_of(o.step[i]) == ';' {", "if int_of(o.step[i]) == ',' {")], RULES + ORACLE),
    M("w-last-step-not-ended", "W3", "the line of the last step is not ended: the model line is glued to it", [(W, "    if len(o.step) > 0 {\n        b = buffer.push(heap, b, byte_of(10));", "    if false {\n        b = buffer.push(heap, b, byte_of(10));")], RULES + ORACLE),
    M("w-hyperparameter-flags-dropped", "W3", "the model line is built without the hyperparameter flags", [(W, "    b = buffer.append(heap, b, model_tokens);\n", "")], RULES + ORACLE),
    M("w-model-not-built-accepted", "W8", "a plan naming a model that is not wired goes on", [(W, "            if kind == 0 {\n                e = wiredata.not_built(heap, e, nm);", "            if false {\n                e = wiredata.not_built(heap, e, nm);")], RULES),
    M("w-bad-hyperparameter-token-ignored", "W8", "a model-line token that is not a value of its kind goes on", [(W, "                if bad != 0 {\n                    e = wiredata.code_error(heap, e, codes.plan_bad_value(), 0, 0, 0);\n                    go = false;", "                if false {\n                    e = wiredata.code_error(heap, e, codes.plan_bad_value(), 0, 0, 0);\n                    go = false;")],
      RULES, why="the plan parser has already refused every token that hparams would (its grammar is the same table): this check is a second line of defence that no input reaches"),
    # ------------------------------------------------------------------ the data: split, target, labels
    M("w-train-and-test-swapped", "W7", "the model is fitted on the test rows", [(W, "            want = 0;\n            ntest = nt;", "            want = 1;\n            ntest = nt;")], ORACLE),
    M("w-steps-fitted-on-all-rows", "W2", "the steps and the model are fitted on every row, the split only scores", [(W, "let (t3, ok4, e4) = train(heap, args, dr, cir, cvr, pr, contents(sr), want, o, retry_ok, 0, 0 - 1, e);", "let (t3, ok4, e4) = train(heap, args, dr, cir, cvr, pr, contents(sr), 0 - 1, o, retry_ok, 0, 0 - 1, e);")], ORACLE),
    M("w-split-test-side-is-the-train-fraction", "W7", "the test side has the train fraction's size", [("tools/ml/splitwire.cho", "let nt = splits.n_test_for(n, num, den);", "let nt = splits.n_test_for(n, den - num, den);")], ORACLE),
    M("w-split-seed-ignored", "W7", "the split is drawn with seed 0", [("tools/ml/splitplan.cho", "let code = splits.train_test(heap, n, n_test, seed, 0, cap, out);", "let code = splits.train_test(heap, n, n_test, 0, 0, cap, out);")], ORACLE),
    M("w-seed-not-stored", "W3", "the seed is not written to the file", [(W, "tr.rows, schema.columns(ds.sc), o.seed, built.compiler());", "tr.rows, schema.columns(ds.sc), 0, built.compiler());")], ORACLE),
    M("w-eval-split-scores-the-train-side", "W7", "eval --split scores the rows the fit trained on", [(IO, "                want = 1;\n                ntest = nt;", "                want = 0;\n                ntest = nt;")], ORACLE),
    M("w-holdout-scores-the-train-side", "W7", "the held-out rows of fit are the training rows", [(W, "wiredata.take(heap, ds.x, sel, which, contents(fr), nfeat, o.max_rows, o.max_cols, o.max_cells);\n            matrix.drop(heap, xte);", "wiredata.take(heap, ds.x, sel, 0, contents(fr), nfeat, o.max_rows, o.max_cols, o.max_cells);\n            matrix.drop(heap, xte);")], RULES),
    M("w-empty-target-accepted", "W8", "an empty target cell is read as 0.0", [(W, "        if bad > 0 {\n            e = wiredata.empty_target(heap, e, tg, bad);", "        if false {\n            e = wiredata.empty_target(heap, e, tg, bad);")], RULES),
    M("w-positive-flag-ignored", "W6", "fit --positive does not choose the label", [(W, "                if o.positive_given {\n                    let pv = o.positive_v;", "                if false {\n                    let pv = o.positive_v;")], RULES + ORACLE),
    M("w-labels-stored-as-0-and-1", "W3", "the labels record always says 0 and 1", [(W, "end = est.encode(kind, contents(mw), 0, tr.d, contents(tr.wv), ds.c0, ds.c1);", "end = est.encode(kind, contents(mw), 0, tr.d, contents(tr.wv), 0, 1);")], ORACLE),
    M("w-class-1-is-the-negative-label", "W3", "the fit trains y = 1 on the negative label", [(W, "if truncate(contents(yr)[i]) == ds.c1 {", "if truncate(contents(yr)[i]) == ds.c0 {", 2)], ORACLE),
    # ------------------------------------------------------------------ missing cells
    M("w-missing-cells-reach-the-estimator", "W8", "a fit with an empty cell after the steps goes on", [(W, "                if mc > 0 && !est.takes_missing(pls.kind) {\n                    borrow mlist as &lr in {\n                        e = wiredata.missing_error(heap, args, e, mc, buffer.bytes(lr), retry_ok);", "                if false {\n                    borrow mlist as &lr in {\n                        e = wiredata.missing_error(heap, args, e, mc, buffer.bytes(lr), retry_ok);")], RULES),
    M("w-missing-cells-in-predict-not-checked", "W8", "predict applies the model to a matrix that still has empty cells", [(IO, "        if mc > 0 && !est.takes_missing(st.kind) {\n            borrow mlist as &lr in {\n                e = wiredata.missing_error(heap, args, e, mc, buffer.bytes(lr), false);", "        if false {\n            borrow mlist as &lr in {\n                e = wiredata.missing_error(heap, args, e, mc, buffer.bytes(lr), false);")], RULES),
    M("w-repair-adds-a-flag-that-is-not-one", "W8", "the repair of a missing value appends --steps", [("tools/ml/wiredata.cho", 'rw = json.put_string(heap, rw, "--step");', 'rw = json.put_string(heap, rw, "--steps");')], RULES),
    # ------------------------------------------------------------------ predict and eval
    M("w-columns-matched-by-position", "W2", "the fitted features are taken from the input's columns by position", [(IO, "                        contents(cw)[f2] = names.find(gq2, schema.name(st.sc, j));", "                        contents(cw)[f2] = f2;")], ORACLE),
    M("w-schema-check-skipped", "W2", "the schema of a planned model's input is not checked", [(IO, "let (e2, same) = verbs.schema_check(heap, args, e, st.view, sr, o, need_target, est.takes_missing(st.kind));", "let (e2, same) = (e, true);")], RULES),
    M("w-predict-wants-the-target", "W2", "predict asks for the target column", [(IO, "read_input(heap, args, fs, root, file, stored, o, false, e, lg)", "read_input(heap, args, fs, root, file, stored, o, true, e, lg)")], RULES),
    M("w-predict-reads-the-target-cell", "W2", "predict of a planned model reads the target column as numbers", [(C, "let skipped = (cfg.mode == 5 || cfg.mode == 9) && i == tcol;", "let skipped = cfg.mode == 5 && i == tcol;")], RULES),
    M("w-proba-column-is-the-label", "W6", "--proba prints the label where the probability goes", [(IO, "code = wiredata.proba_into(heap, kind, xs, wv, blob, contents(qw));", "code = wiredata.predict_into(heap, kind, xs, wv, blob, contents(qw));")], ORACLE),
    M("w-proba-flag-ignored", "W6", "--proba adds no column", [(IO, "                    if with_proba {\n                        w = json.put_string(heap, w, \"probability\");", "                    if false {\n                        w = json.put_string(heap, w, \"probability\");")], RULES + ORACLE),
    M("w-proba-is-the-other-class", "W6", "the probability printed is that of the other class", [(IO, "                            let (w3, p3, f3) = put_cell(heap, w, plain, contents(qq)[k]);", "                            let (w3, p3, f3) = put_cell(heap, w, plain, 1.0 - contents(qq)[k]);", 2)], ORACLE),
    M("w-labels-swapped-in-predict", "W6", "the predicted label is the other label", [(IO, "                        var label = c0;\n                        if contents(pq)[k] > 0.5 {\n                            label = c1;", "                        var label = c1;\n                        if contents(pq)[k] > 0.5 {\n                            label = c0;")], ORACLE),
    M("w-eval-score-not-flipped-for-the-other-label", "W6", "eval --positive with the other label uses the probability of the first", [("tools/ml/wiredata.cho", "                                        contents(sw)[i] = 1.0 - contents(qr)[i];", "                                        contents(sw)[i] = contents(qr)[i];")], ORACLE + ["test_wire"]),
    M("w-precision-and-recall-swapped", "W6", "eval computes recall where precision goes", [("tools/ml/wiredata.cho", "let pr = classify.precision(yt, yp, pos, cap);", "let pr = classify.recall(yt, yp, pos, cap);")], ORACLE),
    M("w-eval-positive-ignored", "W6", "eval --positive is not applied", [(IO, "            if pv == c0 || pv == c1 {\n                pos = pv;", "            if pv == c0 || pv == c1 {\n                pos = c1;")], ORACLE + ["test_wire"]),
    M("w-unknown-label-in-eval-accepted", "W8", "eval takes a label the model never saw", [(IO, "                if foreign >= 0 {", "                if false {")], RULES + ["test_wire"]),
    # ------------------------------------------------------------------ a model file of kind 4 or 5 made with another summation order
    M("w-sum-order-refusal-names-the-version", "M1", "the detail of the refusal calls the sum order a version", [(D, 'w = fail.detail_int(heap, w, "sum_order", f.a);', 'w = fail.detail_int(heap, w, "version", f.a);')], ["test_model_file"]),
    M("w-sum-order-known-is-the-wrong-number", "M1", "the detail gives the value found as the value this reader knows", [(D, 'w = fail.detail_int(heap, w, "known", f.c);', 'w = fail.detail_int(heap, w, "known", f.a);')], ["test_model_file"]),
    # ------------------------------------------------------------------ threads
    M("w-thread-count-modulo-4", "W5", "the thread count reaches the library as t mod 4", [(W, "code = est.fit(heap, kind, o.threads, matrix.shape(xr),", "code = est.fit(heap, kind, o.threads % 4, matrix.shape(xr),")], DET),
    M("w-thread-ceiling-removed", "W5", "a thread count over 64 is accepted by eval", [(W, "if t < 1 || t > 64 {", "if t < 1 || t > 640 {")], RULES),
    M("w-thread-floor-removed", "W5", "a thread count of 0 is accepted by eval", [(W, "if t < 1 || t > 64 {", "if t < 0 || t > 64 {")], RULES),
    # ------------------------------------------------------------------ the generator and the tables
    M("w-gen-hyperparameter-token-off-by-one", "W3", "est.hparams reads the token before the hyperparameter", [(G, "tok = \"bytes.field(line, ' ', %d)\" % (3 + k)", "tok = \"bytes.field(line, ' ', %d)\" % (2 + k)")], ORACLE + RULES, gen=True),
    M("w-gen-takes-says-yes-to-every-flag", "W8", "a model takes every hyperparameter flag", [(G, "        o.append(\"    if kind == %d {\" % keys[m[\"name\"]])\n        for h in flag_hps(m):\n            o += [\"        if bytes.equal(flag, %s) {\" % lit(h[\"name\"]), \"            return true;\", \"        }\"]", "        o.append(\"    if kind == %d {\" % keys[m[\"name\"]])\n        for h in flag_hps(m):\n            o += [\"        if true {\", \"            return true;\", \"        }\"]")], RULES, gen=True),
    M("w-table-lets-predict-take-a-plan", "W2", "the flag table lists --plan among the flags of predict", [(T, 'return "root,model-file,drop,limit,from,proba,', 'return "root,model-file,drop,limit,from,proba,plan,')], FILE + ["test_tables"], gen=True),
    M("w-table-model-choice-lists-a-library", "W8", "logreg is a choice of --model", [(G, "return \"model||choice:%s|none|constant|%s\" % (\"/\".join(names), help_)", "return \"model||choice:%s|none|constant|%s\" % (\"/\".join(names + [\"logreg\"]), help_)")], ["test_tables", "test_honesty"], gen=True),
]


def regen(tree):
    for s in ("gen_models.py", "schemas.py", "manifest.py"):
        r = subprocess.run([sys.executable, str(tree / "scripts" / s)], capture_output=True, text=True, cwd=tree, env=dict(os.environ, CANCHO=mc.compiler()))
        if r.returncode != 0 and s != "manifest.py":
            return s + ": " + r.stdout[-300:] + r.stderr[-300:]
    return None


def run_one(m):
    with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        mc.copy_tree(tree)
        problems = mc.apply_edits(tree, m)
        if problems:
            return "INVALID", "; ".join(problems)
        if m["opt"].get("gen"):
            bad = regen(tree)
            if bad:
                return "INVALID", bad
        else:
            # the tables did not change, but the embedded manifest must match the sources that did
            subprocess.run([sys.executable, str(tree / "scripts" / "manifest.py")], capture_output=True, text=True, cwd=tree, env=dict(os.environ, CANCHO=mc.compiler()))
        binary, log = mc.build(tree)
        if binary is None:
            return "INVALID", "build failed: " + log[:400]
        code, out = mc.run_tests(tree, m["tests"])
        if code != 0:
            f = mc.failing_tests(out)
            return "KILLED", "tests: " + ", ".join(f[:4]) + ("" if f else out.strip().splitlines()[-1][:200])
        return "SURVIVED", ""


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
        print("%d wiring mutants, %d with an edit that does not apply" % (len(MUTANTS), bad))
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

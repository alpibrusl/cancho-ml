#!/usr/bin/env python3
"""Mutants for the wiring of the softmax regression (docs/multiclass.md section 8, gates M1 to M9): a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change to the new code (the classes of the target, the fit, the payload reader, the answers of predict and eval, the multiclass
metrics, describe, the refusals of cv and dump, the plan against the payload), the binary rebuilt from the copy, and the tests of the gates it targets run against it. KILLED: a
test failed. SURVIVED: every test passed, and the mutant must carry a reason (`why`). The machinery is that of scripts/mutants_cli.py and scripts/mutants_wire.py.

    python3 scripts/mutants_wire_softmax.py              all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_wire_softmax.py --only ID
    python3 scripts/mutants_wire_softmax.py --list
    python3 scripts/mutants_wire_softmax.py --check      every `old` text is found exactly once where it is expected (no build, no test)
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
CV = "tools/ml/cv.cho"
MD = "tools/ml/mdesc.cho"
ID = "tools/ml/idump.cho"
MW = "tools/ml/mcwire.cho"
MC = "tools/ml/mcdata.cho"
MM = "tools/ml/mcmetrics.cho"
A = "lib/models/softmax/adapter.cho"
FI = "lib/pipeline/file.cho"

SM = ["test_wire_softmax"]
ORACLE = ["test_wire_softmax_oracle"]
FILES = ["test_wire_softmax_file"]
RULES = ["test_rules"]
TABLES = ["test_tables"]


def M(id, gate, what, edits, tests, **opt):
    return dict(id=id, gate=gate, what=what, edits=[e if len(e) == 4 else (*e, 1) for e in edits], tests=tests, opt=opt)


MUTANTS = [
    # ------------------------------------------------------------------ the classes of the target
    M("sw-two-classes-accepted", "M7", "softmax takes a target of two classes", [(W, "            } else if kk == 2 {", "            } else if kk == 99 {")], SM),
    M("sw-class-ceiling-65", "M2", "65 classes pass the wiring's ceiling", [(W, "            } else if kk > mcdata.max_classes() {", "            } else if kk > mcdata.max_classes() + 1 {")], SM),
    M("sw-single-class-accepted", "M7", "a target of one class goes on to the fit", [(W, "            if kk < 2 {", "            if kk < 1 {")], SM),
    M("sw-positive-allowed", "M1", "--positive is accepted by a fit of three classes", [(W, "            } else if o.positive_given {\n                e = wiredata.conflict(heap, e, \"positive\", \"a model of more than two classes", "            } else if false {\n                e = wiredata.conflict(heap, e, \"positive\", \"a model of more than two classes")], SM),
    M("sw-distinct-descending", "M8", "the classes are collected in descending order", [(MC, "while at > 0 && l[at - 1] > v {", "while at > 0 && l[at - 1] < v {")], SM + ORACLE),
    M("sw-distinct-stops-at-64", "M2", "the count stops at 64, so 65 classes look like 64", [(MC, "while i < n && count <= max_classes() {", "while i < n && count < max_classes() {")], SM),
    M("sw-class-index-plus-one", "M3", "the class index of a row is one too large", [(MC, "            out[i] = float_of(c);", "            out[i] = float_of(c + 1);")], SM + ORACLE),
    M("sw-first-empty-class-ignored", "M7", "a class with no training row goes on to the fit", [(MW, "    if foreign >= 0 || empty >= 0 {", "    if foreign >= 0 {")], SM),
    M("sw-class-absent-names-the-first-label", "M7", "the class with no training row is named by the first label", [(W, "e = mcwire.class_absent(heap, e, contents(lbr2)[fempty]);", "e = mcwire.class_absent(heap, e, contents(lbr2)[0]);")], SM),
    M("sw-training-rows-capped-like-a-tree", "M2", "the fit takes at most the rows a tree takes", [(W, "    if go && est.is_blob(kind) && task != 4 && kind != 9 {", "    if go && est.is_blob(kind) && task != 4 {")], SM),
    # ------------------------------------------------------------------ the fit
    M("sw-fit-not-routed", "M3", "the fit of kind 9 goes through the generic blob fit", [(W, "    if go && pls.kind == 9 {", "    if go && pls.kind == 99 {")], SM + ORACLE),
    M("sw-iterations-ignored", "M3", "the fit runs 100 iterations whatever the plan says", [(W, "contents(lbr)[0..kk], kk, contents(pls.ip)[0], contents(pls.fp)[0]);", "contents(lbr)[0..kk], kk, 100, contents(pls.fp)[0]);")], SM + ORACLE),
    M("sw-l2-ignored", "M3", "the fit has no penalty whatever the plan says", [(W, "contents(lbr)[0..kk], kk, contents(pls.ip)[0], contents(pls.fp)[0]);", "contents(lbr)[0..kk], kk, contents(pls.ip)[0], 0.0);")], SM + ORACLE),
    M("sw-collinear-says-logistic", "M7", "the collinearity refusal names the binary model", [(W, "e = fitfail.collinear(heap, src, e, pls.kind, contents(dgr)", "e = fitfail.collinear(heap, src, e, 5, contents(dgr)")], SM),
    M("sw-fit-summary-dropped", "M5", "the summary of a fit of kind 9 is the coefficient vector of a binary one", [(W, "    } else if kind == 9 {\n        w = mcwire.put_summary(", "    } else if kind == 99 {\n        w = mcwire.put_summary(")], SM),
    M("sw-summary-binary-classes", "M5", "the summary of a fit of kind 9 also says the two classes and the positive label", [(W, "    if task == 2 && kind != 9 {\n        w = json.put_key(heap, w, \"classes\");", "    if task == 2 {\n        w = json.put_key(heap, w, \"classes\");")], SM),
    M("sw-summary-reference-first", "M5", "the reference class of the summary is the first", [(MW, "                w = json.put_int(heap, w, ls[kk - 1]);", "                w = json.put_int(heap, w, ls[0]);")], SM),
    M("sw-summary-intercept-class-shift", "M5", "the intercepts are labelled with the next class", [(MW, "                    w = json.put_int(heap, w, ls[k]);\n                    w = json.put_key(heap, w, \"value\");\n                    w = verbs.put_double(heap, w, bits_of(ws[k * (d + 1) + d]));", "                    w = json.put_int(heap, w, ls[k + 1]);\n                    w = json.put_key(heap, w, \"value\");\n                    w = verbs.put_double(heap, w, bits_of(ws[k * (d + 1) + d]));")], SM),
    # ------------------------------------------------------------------ the payload
    M("sw-labels-may-repeat", "M8", "two equal labels are accepted by the reader", [(A, "|| k > 0 && v <= contents(lw)[k - 1] {", "|| k > 0 && v < contents(lw)[k - 1] {")], FILES),
    M("sw-label-range-upper", "M8", "a label of 13 digits is accepted by the reader", [(A, "v < 0 - record.label_max() || v > record.label_max()", "v < 0 - record.label_max()")], FILES),
    M("sw-label-range-lower", "M8", "a label of minus 13 digits is accepted by the reader", [(A, "v < 0 - record.label_max() || v > record.label_max()", "v > record.label_max()")], FILES),
    M("sw-reader-width-not-compared", "M8", "the reader does not compare the payload's width with the schema's", [(A, "w != d || spass.params(d, kc + 1)", "spass.params(d, kc + 1)")], FILES,
      why="a second line of defence: with the width unchecked the length comparison that follows uses the caller's width, and a payload of another width has another length"),
    M("sw-reader-kc-minimum", "M8", "the reader accepts a header of one coefficient row (two classes)", [(A, "if kc < 2 || kc > softmax.max_classes() - 1", "if kc < 1 || kc > softmax.max_classes() - 1")], FILES,
      why="a second line of defence: a payload of two classes has the length of two classes, which the length comparison does not take for the K the header of the labels record says (and the labels record's extent is compared with K)"),
    M("sw-reader-parameter-bound", "M8", "the reader accepts 2049 parameters", [(A, "spass.params(d, kc + 1) > softmax.max_params()", "spass.params(d, kc + 1) > softmax.max_params() + 1000")], FILES),
    M("sw-reader-class-ceiling", "M8", "the reader accepts 65 classes", [(A, "kc > softmax.max_classes() - 1", "kc > softmax.max_classes()")], FILES),
    M("sw-reader-length-not-compared", "M8", "the reader does not compare the payload's length with the size it announces", [(A, "    if len(blob) != payload_size(kk, d) {", "    if false {")], FILES),
    M("sw-reader-labels-end", "M8", "bytes after the labels are accepted", [(A, "|| r2.end + 16 + 8 * kk != len(blob) {", "{")], FILES,
      why="the length comparison before the allocation refuses a payload of another length first (see sw-reader-length-not-compared)"),
    M("sw-plan-payload-family", "M8", "a plan of softmax is accepted over any payload", [(FI, "        return mk == kinds.model_softmax();", "        return true;")], FILES),
    # ------------------------------------------------------------------ predict
    M("sw-predict-columns-binary", "M5", "predict --proba names the second column probability", [(IO, "if with_proba && kind == 9 {", "if with_proba && kind == 99 {")], SM + ORACLE),
    M("sw-predict-label-by-threshold", "M5", "the label is chosen by the binary threshold", [(IO, "                    } else if kind == 9 {\n                        borrow labs as &lbr in {", "                    } else if kind == 99 {\n                        borrow labs as &lbr in {")], SM + ORACLE),
    M("sw-predict-no-probabilities", "M5", "the probabilities cell is not written", [(IO, "                            if with_proba {\n                                let (w3, p3, f3) = put_probs(", "                            if false {\n                                let (w3, p3, f3) = put_probs(")], SM + ORACLE),
    M("sw-predict-keys-reversed", "M5", "the keys of the probabilities are the labels in descending order", [(IO, "labs[j], 0);\n        borrow key as &kr in {", "labs[kk - 1 - j], 0);\n        borrow key as &kr in {")], SM + ORACLE),
    M("sw-predict-label-of-the-last-class", "M5", "every row gets the label of the reference class", [(IO, "put_label(heap, w, plain, contents(lbr)[truncate(contents(pq)[k])]);", "put_label(heap, w, plain, contents(lbr)[kk - 1]);")], SM + ORACLE),
    # ------------------------------------------------------------------ eval, the metrics
    M("sw-eval-positive-allowed", "M1", "eval --positive is accepted by a model of three classes", [(IO, "        if go && kind == 9 && o.positive_given {", "        if go && kind == 99 && o.positive_given {")], SM + RULES),
    M("sw-eval-unseen-dropped", "M7", "a label the model has no class for is not named", [(IO, "                        if at >= 0 {\n                            e = mcwire.unseen_row(", "                        if false {\n                            e = mcwire.unseen_row(")], SM + ORACLE),
    M("sw-eval-binary-label-check", "M7", "eval checks the labels against the smallest and the largest class only", [(IO, "            if go && task == 2 && kind != 9 {\n                var i = 0;", "            if go && task == 2 {\n                var i = 0;")], SM + ORACLE),
    M("sw-eval-says-positive", "M5", "the answer of eval of three classes names a positive label", [(IO, "                                                if task == 2 && kind != 9 {\n                                                    w = json.put_key(heap, w, \"positive\");", "                                                if task == 2 {\n                                                    w = json.put_key(heap, w, \"positive\");")], SM + ORACLE),
    M("sw-metrics-not-routed", "M4", "the metrics of kind 9 are those of a binary model", [(D, "    if kind == 9 {\n        // a multiclass model: the metrics", "    if kind == 99 {\n        // a multiclass model: the metrics")], SM + ORACLE),
    M("sw-metrics-weighted-as-macro", "M4", "the weighted averages are macro averages", [(MM, "multiclass.average(cm, kk, which, weighted == 1)", "multiclass.average(cm, kk, which, false)")], SM + ORACLE),
    M("sw-metrics-recall-weighted-key", "M4", "the weighted recall has the key of the macro recall", [(MM, "key = \"recall_weighted\";", "key = \"recall_macro\";")], SM + ORACLE),
    M("sw-metrics-predicted-label-zero", "M4", "every row is predicted as the first class", [(MM, "contents(ypw)[i] = contents(lr)[truncate(contents(pr)[i])];", "contents(ypw)[i] = contents(lr)[0];")], SM + ORACLE),
    M("sw-metrics-true-label-shift", "M4", "the true label of a row is its neighbour's", [(MM, "contents(tw)[i] = truncate(yv[i]);", "contents(tw)[i] = truncate(yv[(i + 1) % n]);")], SM + ORACLE),
    M("sw-metrics-left-out-zero", "M4", "the classes the macro averages left out are not counted", [(MM, "contents(lw)[which] = a.left_out;", "contents(lw)[which] = 0;")], SM + ORACLE),
    M("sw-metrics-nonfinite-not-scanned", "M5", "a probability that is not finite is scored", [(MM, "            if strict {\n                borrow prob as &qr0 in {", "            if false {\n                borrow prob as &qr0 in {")], SM),
    # ------------------------------------------------------------------ describe, cv, dump
    M("sw-describe-two-classes", "M5", "describe lists the smallest and the largest class only", [(MD, "    if task == 2 && kind == kinds.model_softmax() {", "    if task == 2 && kind == 99 {")], SM + ["test_describe"]),
    M("sw-describe-positive", "M5", "describe names a positive label", [(MD, "    w = json.put_key(heap, w, \"positive\");\n    if task == 2 && kind > 3 && kind != kinds.model_softmax() {", "    w = json.put_key(heap, w, \"positive\");\n    if task == 2 && kind > 3 {")], SM),
    M("sw-describe-threshold", "M5", "describe says the label threshold is 0.5", [(MD, "    w = json.put_key(heap, w, \"label_threshold\");\n    if task == 2 && kind > 3 && kind != kinds.model_softmax() {", "    w = json.put_key(heap, w, \"label_threshold\");\n    if task == 2 && kind > 3 {")], SM),
    M("sw-cv-accepted", "M1", "cv takes a model of three classes", [(CV, "    if go && kind == 9 {\n        e = wiredata.conflict(heap, e, \"model\"", "    if false {\n        e = wiredata.conflict(heap, e, \"model\"")], SM),
    M("sw-dump-linear", "M7", "dump prints a multiclass model as a coefficient vector", [(ID, "    if kind == 8 || kind == 9 {", "    if kind == 8 {")], SM),
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
        print("%d softmax mutants, %d with an edit that does not apply" % (len(MUTANTS), bad))
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

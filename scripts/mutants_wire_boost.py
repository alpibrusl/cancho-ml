#!/usr/bin/env python3
"""Mutants for the wiring of the boosted trees (docs/models-boost.md gate GB7, the half that is not the library): a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change to the wiring code (the rule that early stopping needs --split, the summary of a fit, describe's ensemble, the reader of a
file whose plan names a boosted model, the generator's mapping of a flag and of the early-stopping hyperparameter, the schema of the answers, the kind number in the plan
family), the binary rebuilt from the copy, and the tests of the gates it targets run against it. KILLED: a test failed. SURVIVED: every test passed, and the mutant must carry a
reason (`why`). The library's mutants are scripts/mutants_boost.py; the machinery is that of scripts/mutants_wire_trees.py.

    python3 scripts/mutants_wire_boost.py              all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_wire_boost.py --only ID
    python3 scripts/mutants_wire_boost.py --slice 0:12
    python3 scripts/mutants_wire_boost.py --list
    python3 scripts/mutants_wire_boost.py --check      every `old` text is found exactly once where it is expected (no build, no test)
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
MD = "tools/ml/mdesc.cho"
G = "scripts/gen_models.py"
S = "scripts/schemas.py"
FI = "lib/pipeline/file.cho"
OD = "scripts/oracle_describe.py"

BOOST = ["test_wire_boost"]
ORACLE = ["test_wire_boost_oracle"]
FILES = ["test_wire_boost_file"]
RULES = ["test_rules"]
DESC = ["test_describe.Oracle", "test_describe.Contract"]
BASE = ["test_wire_baseline", "test_cat_baseline", "test_planned_baseline"]


def M(id, gate, what, edits, tests, **opt):
    return dict(id=id, gate=gate, what=what, edits=[e if len(e) == 4 else (*e, 1) for e in edits], tests=tests, opt=opt)


MUTANTS = [
    # ------------------------------------------------------------------ early stopping needs --split
    M("wb-early-stopping-without-a-split-accepted", "GB6", "early stopping without --split goes on", [(W, "    if go && !o.split_given {\n        borrow pls as &npq in {", "    if false && !o.split_given {\n        borrow pls as &npq in {")], BOOST + RULES),
    M("wb-early-stopping-needs-a-split-always", "GB6", "early stopping is refused even with --split", [(W, "    if go && !o.split_given {\n        borrow pls as &npq in {", "    if go {\n        borrow pls as &npq in {")], BOOST + RULES),
    M("wb-needs-split-at-zero", "GB6", "an early-stopping of 0 also needs --split", [(G, "\"        return ip[%d] > 0;\" % k", "\"        return ip[%d] >= 0;\" % k")], BOOST + RULES, gen=True),
    M("wb-needs-split-index", "GB6", "the early-stopping value is read from the neighbour hyperparameter", [(G, "\"        return ip[%d] > 0;\" % k", "\"        return ip[%d] > 0;\" % (k - 1)")], BOOST + RULES, gen=True),
    M("wb-zero-is-not-a-flag-value", "GB6", "the command line accepts no 0 for --early-stopping", [(G, "        if mins[n] != 1:", "        if False:")], BOOST, gen=True),
    # ------------------------------------------------------------------ early stopping under the split strategies (docs/models-boost.md section 2.8)
    M("wb-es-group-and-time-split-accepted-in-fit", "GB6", "early stopping with --split-group or --split-by goes on in fit", [(W, "if go && o.split_given && splitspec.strategy(sp) >= 2 {", "if go && o.split_given && splitspec.strategy(sp) >= 4 {")], BOOST),
    M("wb-es-time-split-accepted-in-fit", "GB6", "only the group split refuses early stopping in fit", [(W, "if go && o.split_given && splitspec.strategy(sp) >= 2 {", "if go && o.split_given && splitspec.strategy(sp) == 2 {")], BOOST),
    M("wb-es-group-split-accepted-in-fit", "GB6", "only the time split refuses early stopping in fit", [(W, "if go && o.split_given && splitspec.strategy(sp) >= 2 {", "if go && o.split_given && splitspec.strategy(sp) >= 3 {")], BOOST),
    M("wb-es-stratified-split-refused-in-fit", "GB6", "the stratified split refuses early stopping too", [(W, "if go && o.split_given && splitspec.strategy(sp) >= 2 {", "if go && o.split_given && splitspec.strategy(sp) >= 1 {")], BOOST),
    M("wb-es-group-and-time-split-accepted-in-cv", "GB6", "early stopping with a group or a time strategy goes on in cv", [("tools/ml/cv.cho", "    if go && splitspec.strategy(sp) >= 2 {", "    if go && splitspec.strategy(sp) >= 4 {")], BOOST),
    M("wb-es-stratified-refused-in-cv", "GB6", "cv with stratified folds refuses early stopping", [("tools/ml/cv.cho", "    if go && splitspec.strategy(sp) >= 2 {", "    if go && splitspec.strategy(sp) >= 1 {")], BOOST),
    M("wb-es-repair-keeps-the-flag", "GB6", "the repair of the refusal keeps --early-stopping", [("tools/ml/splitspec.cho", 'w = retry_edit(heap, w, args, "--early-stopping", true, false, "");', 'w = retry_edit(heap, w, args, "--early-stopping", true, true, "5");')], BOOST),
    M("wb-es-repair-leaves-the-value", "GB6", "the repair drops --early-stopping and leaves its value", [("tools/ml/splitspec.cho", 'w = retry_edit(heap, w, args, "--early-stopping", true, false, "");', 'w = retry_edit(heap, w, args, "--early-stopping", false, false, "");')], BOOST),
    M("wb-es-detail-names-the-wrong-flag", "GB6", "the detail names --split-by for a group split", [("tools/ml/splitspec.cho", '    if strategy == 2 {\n        w = fail.detail_str(heap, w, "other", "--split-group");', '    if strategy == 3 {\n        w = fail.detail_str(heap, w, "other", "--split-group");')], BOOST),
    # ------------------------------------------------------------------ the summary of a fit
    M("wb-summary-of-a-forest", "GB2", "the summary of a boosted fit is a forest's", [(W, "record.get32(contents(tr.blob), 0) == kinds.model_boost()", "false")], BOOST + RULES),
    M("wb-summary-rounds-field", "GB2", "the rounds of the summary are read from the node count", [(W, "    w = json.put_key(heap, w, \"rounds\");\n    w = json.put_int(heap, w, fbits.get64(blob, 28));", "    w = json.put_key(heap, w, \"rounds\");\n    w = json.put_int(heap, w, fbits.get64(blob, 52));")], BOOST + RULES),
    M("wb-summary-requested-field", "GB2", "the rounds requested of the summary are the rounds kept", [(W, "    w = json.put_key(heap, w, \"rounds_requested\");\n    w = json.put_int(heap, w, fbits.get64(blob, 84));", "    w = json.put_key(heap, w, \"rounds_requested\");\n    w = json.put_int(heap, w, fbits.get64(blob, 28));")], BOOST),
    M("wb-summary-learning-rate-field", "GB2", "the learning rate of the summary is the base score", [(W, "    w = verbs.put_double(heap, w, fbits.get64(blob, 124));", "    w = verbs.put_double(heap, w, fbits.get64(blob, 116));")], BOOST + RULES),
    M("wb-summary-best-round-always", "GB2", "the summary names a best round also without early stopping", [(W, "    if fbits.get64(blob, 76) > 0 {\n        w = json.put_key(heap, w, \"best_round\");", "    if true {\n        w = json.put_key(heap, w, \"best_round\");")], BOOST),
    M("wb-summary-best-round-never", "GB2", "the summary never names a best round", [(W, "    if fbits.get64(blob, 76) > 0 {\n        w = json.put_key(heap, w, \"best_round\");", "    if false {\n        w = json.put_key(heap, w, \"best_round\");")], BOOST),
    M("wb-summary-nodes-field", "GB2", "the nodes of the summary are the rounds", [(W, "    w = json.put_key(heap, w, \"nodes\");\n    w = json.put_int(heap, w, fbits.get64(blob, 52));\n    w = json.put_key(heap, w, \"bins\");\n    w = json.put_int(heap, w, fbits.get64(blob, 44));\n    w = json.put_key(heap, w, \"max_depth\");\n    w = json.put_int(heap, w, fbits.get64(blob, 60));\n    w = json.put_key(heap, w, \"min_leaf\");\n    w = json.put_int(heap, w, fbits.get64(blob, 68));\n    w = json.put_key(heap, w, \"learning_rate\");", "    w = json.put_key(heap, w, \"nodes\");\n    w = json.put_int(heap, w, fbits.get64(blob, 28));\n    w = json.put_key(heap, w, \"bins\");\n    w = json.put_int(heap, w, fbits.get64(blob, 44));\n    w = json.put_key(heap, w, \"max_depth\");\n    w = json.put_int(heap, w, fbits.get64(blob, 60));\n    w = json.put_key(heap, w, \"min_leaf\");\n    w = json.put_int(heap, w, fbits.get64(blob, 68));\n    w = json.put_key(heap, w, \"learning_rate\");")], BOOST + RULES),
    # ------------------------------------------------------------------ describe
    M("wb-describe-trees-field", "GB8", "describe's trees are read from the node count", [(MD, "            w = json.put_key(heap, w, \"trees\");\n            w = json.put_int(heap, w, fbits.get64(data, mdat + 28));", "            w = json.put_key(heap, w, \"trees\");\n            w = json.put_int(heap, w, fbits.get64(data, mdat + 52));")], BOOST + DESC),
    M("wb-describe-nodes-field", "GB8", "describe's nodes are read from the rounds", [(MD, "            w = json.put_key(heap, w, \"nodes\");\n            w = json.put_int(heap, w, fbits.get64(data, mdat + 52));", "            w = json.put_key(heap, w, \"nodes\");\n            w = json.put_int(heap, w, fbits.get64(data, mdat + 28));")], BOOST + DESC),
    M("wb-describe-requested-field", "GB8", "describe's rounds requested are the rounds kept", [(MD, "            w = json.put_int(heap, w, fbits.get64(data, mdat + 84));", "            w = json.put_int(heap, w, fbits.get64(data, mdat + 28));")], BOOST + DESC),
    M("wb-describe-ensemble-for-every-kind", "GB8", "describe adds the ensemble to the other kinds too", [(MD, "    if kind == kinds.model_boost() {\n        let (mdat, mdlen)", "    if kind >= 4 {\n        let (mdat, mdlen)")], BOOST + DESC),
    # ------------------------------------------------------------------ the file whose plan names a boosted model
    M("wb-plan-family-of-boost", "GB6", "a boost plan accepts the payload of a tree", [(FI, "    if pk == 6 {\n        return mk == kinds.model_boost();", "    if pk == 6 {\n        return mk == kinds.model_tree();")], BOOST + FILES),
    M("wb-kind-7-under-a-boost-plan", "GB6", "a kind 7 payload under a boost plan goes to the old reader", [(IO, " || bytes.find(plan, \"model kind=boost\") >= 0;", ";")], RULES,
      why="the old reader refuses the same file as ml.model.version (sum order 3 against its 2) where this one refuses it as malformed: both are a refusal of a damaged file, and the fixture accepts the tag it gets today only through the second; if the test cannot tell them apart this survives"),
    M("wb-sum-order-version-detail", "GB6", "the sum order of a kind 8 file is not compared", [(FI, "if so != kinds.sum_order_for(get32(md0, 0)) {", "if false {")], BOOST + FILES),
    # ------------------------------------------------------------------ the generator and the schemas
    M("wb-takes-missing-excludes-boost", "GB1", "a boosted model refuses a missing cell like a linear one", [(G, "(\" || \".join(\"kind == %d\" % keys[m[\"name\"]] for m in blob if m[\"task\"] == \"both\") or \"false\")", "(\" || \".join(\"kind == %d\" % keys[m[\"name\"]] for m in blob if m[\"task\"] == \"both\" and m[\"name\"] != \"boost\") or \"false\")")], BOOST + ORACLE, gen=True),
    M("wb-schema-boost-learning-rate", "GB6", "the schema of the summary types the learning rate as a count", [(S, "\"learning_rate\": FLT, \"base_score\": FLT,", "\"learning_rate\": NAT, \"base_score\": FLT,")], RULES, gen=True),
    M("wb-schema-ensemble-required", "GB8", "describe's ensemble is required of every model", [(S, "}, [\"model\", \"task\", \"kind\", \"plan\", \"rows\", \"file\", \"target\",", "}, [\"model\", \"task\", \"kind\", \"plan\", \"rows\", \"file\", \"ensemble\", \"target\",")], DESC + RULES, gen=True),
    M("wb-model-not-in-the-wired-list", "GB11", "boost is left out of the --model choice list", [(G, "names = [m[\"name\"] for m in models if m[\"wired\"]]\n    tiers", "names = [m[\"name\"] for m in models if m[\"wired\"] and m[\"name\"] != \"boost\"]\n    tiers")], RULES + BOOST, gen=True),
    M("wb-hparam-ok-for-floats", "GB6", "a float flag of a boosted model is not checked as a number", [(G, "        return float_value(value) >= 0.0;", "        return true;")], BOOST + RULES, gen=True,
      ),
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
        print("%d wiring mutants, %d with an edit that does not apply" % (len(MUTANTS), bad))
        return 1 if bad else 0
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    # --slice A:B runs the mutants A to B - 1 of the list (the suites are long: slices run side by side; the baseline is run by the slice that starts at 0)
    lo, hi = 0, len(MUTANTS)
    if "--slice" in argv:
        a, b = argv[argv.index("--slice") + 1].split(":")
        lo, hi = int(a or 0), int(b or len(MUTANTS))
    results = []
    for m in MUTANTS[lo:hi]:
        if only and m["id"] != only:
            continue
        verdict, detail = mw.run_one(m)
        results.append((m, verdict, detail))
        print("%-9s %-48s %s  %s" % (verdict, m["id"], m["gate"], detail), flush=True)
    if not only and lo == 0:
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

#!/usr/bin/env python3
"""Mutants of the wiring of the recommender (docs/wiring-recommend.md gate WR14): a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change to the code of the stage (the log reader and the numbering of the ids, the split and the renumbering of the items, the plan, the file, the
answers of fit, predict, eval and describe, the reader of the file around the payload, the checks of the command line), the binary rebuilt from the copy, and the tests of the gates it targets run
against it. KILLED: a test failed. SURVIVED: every test passed, and the mutant must carry a reason (`why`). The machinery is that of scripts/mutants_cli.py and scripts/mutants_wire.py.

    python3 scripts/mutants_recommend_wire.py              all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_recommend_wire.py --only ID
    python3 scripts/mutants_recommend_wire.py --list
    python3 scripts/mutants_recommend_wire.py --check      every `old` text is found exactly once where it is expected (no build, no test)
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
LOAD = "tools/ml/recload.cho"
SPLIT = "tools/ml/recsplit.cho"
WIRE = "tools/ml/recwire.cho"
PRED = "tools/ml/recpred.cho"
EVAL = "tools/ml/receval.cho"
IO = "tools/ml/recio.cho"
FLAGS = "tools/ml/recflags.cho"
APP = "tools/ml/app.cho"
FI = "lib/pipeline/file.cho"

ORACLE = ["test_wire_recommend_oracle"]
LOG = ["test_wire_recommend"]
FILE = ["test_wire_recommend_file"]
RULES = ["test_rules"]
TABLES = ["test_tables"]


def M(id, gate, what, edits, tests, **opt):
    return dict(id=id, gate=gate, what=what, edits=[e if len(e) == 4 else (*e, 1) for e in edits], tests=tests, opt=opt)


MUTANTS = [
    # ------------------------------------------------------------------ the log: columns, ids, ratings
    M("wr-rating-read-off-by-one", "WR3", "a rating is read as the number plus one", [(LOAD, "contents(vw)[r] = csvcell.double(kind, m, frac, key);", "contents(vw)[r] = csvcell.double(kind, m, frac, key) + 1.0;")], ORACLE),
    M("wr-empty-id-accepted", "WR2", "an empty id cell is an id", [(LOAD, "    if miss > 0 {", "    if false {")], RULES),
    M("wr-long-id-limit-256", "WR7", "an id of 256 bytes is accepted", [(LOAD, "if off[r + 1] - off[r] > 255 {", "if off[r + 1] - off[r] > 256 {")], RULES),
    M("wr-users-ceiling-plus-one", "WR7", "one user over the ceiling is accepted", [(LOAD, "    if count > ceiling {", "    if count > ceiling + 1 {")], RULES),
    M("wr-row-cap-not-the-interactions-limit", "WR7", "the rows are capped by --max-rows only", [(LOAD, "let cap = min_of(o.max_rows, log.max_interactions());", "let cap = o.max_rows;")], RULES),
    # ------------------------------------------------------------------ the held-out part
    M("wr-items-of-heldout-rows-kept", "WR11", "an item that has only held-out rows keeps a number", [(SPLIT, "            if mask[r] == 0 {\n                contents(mw)[iid[r]] = 0;", "            if true {\n                contents(mw)[iid[r]] = 0;")], ORACLE),
    M("wr-train-rows-are-the-heldout-rows", "WR11", "the training log is made of the held-out rows", [(SPLIT, "                    if mask[k] == 0 {\n                        contents(uw)[at] = contents(rows.uid)[k];", "                    if mask[k] == 1 {\n                        contents(uw)[at] = contents(rows.uid)[k];")], ORACLE),
    M("wr-history-pair-is-a-test-item", "WR11", "a held-out pair that is in the history is a test item", [(SPLIT, "} else if has_item(hitem, hoff[u], hoff[u + 1], m2) {", "} else if false {")], ORACLE),
    M("wr-repeated-test-item-twice", "WR11", "a test item repeated in the held-out rows is listed twice", [(SPLIT, "                                if !dup {", "                                if true {")], ORACLE),
    # ------------------------------------------------------------------ the plan, the file
    M("wr-no-rating-is-explicit", "WR2", "a log with no rating column is fitted as explicit when --feedback is not given", [(WIRE, "        line = buffer.append(heap, line, \" feedback=implicit\");", "        line = buffer.append(heap, line, \"\");")], ORACLE + LOG),
    M("wr-explicit-payload-keeps-alpha", "WR3", "an explicit payload holds the alpha of the plan", [(WIRE, "    var alpha = 0.0;\n    if pa.implicit {\n        alpha = pa.alpha;\n    }", "    var alpha = pa.alpha;")], ORACLE),
    M("wr-meta-rows-are-the-rows-read", "WR3", "the meta of the file counts the rows read, not the interactions kept", [(WIRE, "buffer.bytes(pr), lx.nnz, cols, o.seed, built.compiler());", "buffer.bytes(pr), nrows, cols, o.seed, built.compiler());")], ORACLE),
    M("wr-seed-not-stored", "WR3", "the seed is not written to the file", [(WIRE, "buffer.bytes(pr), lx.nnz, cols, o.seed, built.compiler());", "buffer.bytes(pr), lx.nnz, cols, 0, built.compiler());")], ORACLE),
    M("wr-l2-zero-accepted", "WR7", "the plan checks lambda against 0 or more", [(WIRE, "    if pa.ok && !(pa.lambda > 0.0 && pa.lambda - pa.lambda == 0.0) {", "    if pa.ok && !(pa.lambda >= 0.0 && pa.lambda - pa.lambda == 0.0) {")], RULES,
      why="the solver refuses the same value with the same rule (ml.param.out-of-range, exit 2): the check of the plan only names the flag in its detail, which a fixture of the rule does not look at"),
    M("wr-split-default-top-not-clamped", "WR5", "the cut-off of the held-out metrics is 10 whatever the number of items", [(WIRE, "    if !argv.has(src, \"top\") && k > tr.ni {", "    if false {")], RULES),
    M("wr-eval-work-doubled", "WR7", "the ceiling of the table of scores is twice the one the document says", [(WIRE, "    return nu * ni <= max_score_cells() && nu * ni * f <= max_eval_work();", "    return nu * ni <= 2 * max_score_cells() && nu * ni * f <= max_eval_work();")], RULES),
    M("wr-scores-negated", "WR11", "the held-out metrics are those of the reversed ranking", [(WIRE, "                contents(sw)[u * ni + i] = s;", "                contents(sw)[u * ni + i] = 0.0 - s;")], ORACLE),
    # ------------------------------------------------------------------ predict
    M("wr-keep-seen-ignored", "WR5", "the history is always left out", [(PRED, "if by_user && !keep {", "if by_user {")], ORACLE),
    M("wr-rank-from-zero", "WR5", "the rank counts from 0", [(PRED, "w = json.put_int(heap, w, r + 1);", "w = json.put_int(heap, w, r);")], ORACLE),
    M("wr-page-end-inclusive", "WR5", "a page holds one row too many", [(PRED, "if at + r >= start && at + r < end {", "if at + r >= start && at + r <= end {")], RULES + ORACLE),
    M("wr-like-keeps-the-item-itself", "WR5", "the neighbours of an item include the item", [(PRED, "contents(exw)[0] = who_q;", "contents(exw)[0] = 0;")], ORACLE),
    M("wr-cosine-without-the-root", "WR5", "the norm of an item is the sum of its squares, not the root", [(PRED, "contents(nw)[it] = math_sqrt(dot(qf, qf, it, it, f));", "contents(nw)[it] = dot(qf, qf, it, it, f);")], ORACLE),
    M("wr-unknown-user-accepted", "WR6", "an id the model has not seen is the user before the first", [(PRED, "                                if found < 0 {", "                                if found < 0 - 1 {")], RULES),
    # ------------------------------------------------------------------ eval
    M("wr-cold-users-counted-warm", "WR5", "a user the model has never seen is not counted as cold", [(EVAL, "        if umap[d] < 0 {", "        if umap[d] < 0 - 1 {")], ORACLE + RULES),
    # ------------------------------------------------------------------ the file around the payload
    M("wr-plan-and-payload-not-compared", "WR9", "a plan that disagrees with the payload is accepted", [(IO, "if bad != 0 || contents(ir)[0] != mr.f || contents(ir)[1] != mr.iterations || contents(ir)[2] != method {", "if bad != 0 {")], FILE),
    M("wr-rows-and-interactions-not-compared", "WR9", "a file whose meta counts other rows than its history is accepted", [(IO, "    if rows != nnz {", "    if false {")], FILE),
    M("wr-explicit-file-without-rating-accepted", "WR9", "an explicit file with two columns is accepted", [(IO, "|| !mr.implicit && cols != 3 {", " {")], FILE),
    M("wr-kind-10-family-not-checked", "WR9", "a plan of the recommender over a payload of another kind is accepted", [(FI, "    if pk == 8 {\n        return mk == kinds.model_recommend();\n    }\n", "    if pk == 8 {\n        return true;\n    }\n")], FILE),
    # ------------------------------------------------------------------ the command line
    M("wr-user-not-required", "WR1", "fit --model recommend runs without --user", [(FLAGS, "    if !argv.has(src, \"user\") {", "    if false {")], RULES + LOG),
    M("wr-time-without-split-accepted", "WR10", "--time without --split-user-last is accepted", [(FLAGS, "    } else if argv.has(src, \"time\") && !argv.has(src, \"split-user-last\") {", "    } else if false {")], RULES),
    M("wr-top-without-split-accepted", "WR10", "--top is accepted by a fit with no split", [(FLAGS, "    if argv.has(src, \"top\") && !argv.has(src, \"split-user-last\") {", "    if false {")], RULES),
    M("wr-explicit-without-rating-accepted", "WR10", "--feedback explicit is accepted with no rating column", [(FLAGS, "    if argv.has(src, \"feedback\") && !implicit && !argv.has(src, \"rating\") {", "    if false {")], RULES),
    M("wr-alpha-with-explicit-accepted", "WR10", "--alpha is accepted by an explicit fit", [(FLAGS, "    if argv.has(src, \"alpha\") && !implicit {", "    if false {")], RULES),
    M("wr-target-required-again", "WR10", "fit --model recommend needs --target", [(APP, " && est.task(est.kind_named(argv.text(src, \"model\"))) != 5 {", " {")], RULES + LOG + ORACLE),
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
        print("%d recommender wiring mutants, %d with an edit that does not apply" % (len(MUTANTS), bad))
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

#!/usr/bin/env python3
"""Mutants for the refusals of docs/refusals-fit.md (issue 48): ml.fit.collinear and ml.fit.separable. Each is a copy of the tree with one change, `ml` rebuilt from the copy
(the copy carries every other built executable, as scripts/mutants_cli.py's does) and tests/conformance/test_refuse_fit.py run against it: the fixtures, every argv of
every repair, the exact-fraction oracle of the rank decision, the exact separability oracle with an independent Newton iteration, the fuzz and the thread invariance.
KILLED means a test failed; SURVIVED must carry a reason.

    python3 scripts/mutants_refusals.py             # all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_refusals.py --only ID
    python3 scripts/mutants_refusals.py --list
    python3 scripts/mutants_refusals.py --check     # every pattern applies once (no build, no test)
"""

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mutants_cli as mc  # noqa: E402

RANK = "lib/models/rank.cho"
LOG = "lib/models/linear/logreg.cho"
LIN = "lib/models/linear/linreg.cho"
FF = "tools/ml/fitfail.cho"
CHOL = "lib/linalg/chol.cho"
T = ["test_refuse_fit"]


def M(id, what, edits, **opt):
    return mc.M(id, "R48", what, [e if len(e) == 4 else e + (1,) for e in edits], T, **opt)


MUTANTS = [
    # ---- the rank test and its tolerance
    M("tolerance-2^20-larger", "the pivot tolerance is 2^-20 (legitimate near-collinear data is refused)", [(CHOL, "return 1.0 / 1099511627776.0;", "return 1.0 / 1048576.0;")]),
    M("tolerance-2^20-smaller", "the pivot tolerance is 2^-60 (rounding noise is accepted: dependent columns fit)", [(CHOL, "return 1.0 / 1099511627776.0;", "return 1.0 / 1152921504606846976.0;")]),
    M("diagnosis-test-inverted", "the diagnosis finds the first pivot that PASSES", [(RANK, "if !(s > chol.pivot_tol() * ajj) {", "if s > chol.pivot_tol() * ajj {")]),
    M("diagnosis-test-absolute", "the diagnosis tests the pivot against the tolerance alone, not against its diagonal element", [(RANK, "if !(s > chol.pivot_tol() * ajj) {", "if !(s > chol.pivot_tol()) {")]),
    M("diagnosis-tolerance-differs", "the diagnosis uses a tolerance 2^10 larger than linalg.chol (the two disagree: the column named is not the one that failed)",
      [(RANK, "if !(s > chol.pivot_tol() * ajj) {", "if !(s > chol.pivot_tol() * 1024.0 * ajj) {")]),
    M("pivot-index-off-by-one", "the pivot index is written one too high", [(RANK, "diag[1] = float_of(bad);", "diag[1] = float_of(bad + 1);")]),
    M("relative-pivot-is-the-sum", "the relative pivot is written as the sum itself", [(RANK, "ratio = s / ajj;", "ratio = s;")]),
    M("dependency-forward-sign", "the forward solve of the dependency subtracts with the wrong sign", [(RANK, "s = s - l[q * p + k] * diag[4 + k];", "s = s + l[q * p + k] * diag[4 + k];")]),
    M("dependency-backward-index", "the backward solve of the dependency reads L transposed", [(RANK, "s = s - l[k * p + q] * diag[4 + k];", "s = s - l[q * p + k] * diag[4 + k];")]),
    M("zero-first-column-not-converted", "a failure at the first column is not turned into ml.fit.collinear", [(LIN, "if rank.record(heap, p, contents(grr), 0, diag) >= 0 {", "if rank.record(heap, p, contents(grr), 0, diag) > 0 {")]),
    M("collinear-not-converted", "the linear fit keeps ml.linalg.not-positive-definite", [(LIN, "if rank.record(heap, p, contents(grr), 0, diag) >= 0 {", "if rank.record(heap, p, contents(grr), 0, diag) >= 1000 {")]),
    M("logistic-pass0-not-collinear", "the first Newton pass is not told apart: a collinear design is called separable", [(LOG, "                if it == 0 {", "                if it == 1 {")]),
    # ---- the naming
    M("column-named-is-the-first", "the column in the detail is always column 0", [(FF, 'w = verbs.put_name(heap, w, "column", names.name(nm, j));', 'w = verbs.put_name(heap, w, "column", names.name(nm, 0));')]),
    M("columns-cut-too-high", "the dependency lists nothing (the cut is 2^20 times the largest coefficient)", [(FF, "let cut = cmax / 1048576.0;", "let cut = cmax * 1048576.0;")]),
    M("columns-cut-too-low", "the dependency lists every column, noise included", [(FF, "let cut = cmax / 1048576.0;", "let cut = 0.0;")]),
    M("intercept-not-named", "the intercept is called by the name of column 0", [(FF, 'w = fail.detail_str(heap, w, "column", "(intercept)");', 'w = verbs.put_name(heap, w, "column", names.name(nm, 0));')]),
    # ---- the repair
    M("repair-l2-too-small-by-2^30", "the ridge of the repair is chosen from the diagonal times zero (always 1)", [(FF, "decade_text(heap, maxdiag / 1073741824.0)", "decade_text(heap, maxdiag * 0.0)")]),
    M("repair-l2-of-the-separable-fit", "the ridge of the separable repair is always 1", [(FF, "decade_text(heap, diag[4] / 1024.0)", "decade_text(heap, diag[4] * 0.0)")]),
    M("repair-explicit-l2-appended", "an explicit --l2 0 is not replaced: the repair carries --l2 twice", [(FF, 'if i > 0 && bytes.equal(arg(args, i - 1), "--l2") {', "if false {")]),
    M("repair-drop-names-column-0", "the --drop repair names the first column", [(FF, "w = put_argv_drop(heap, w, args, names.name(nm, dropk));", "w = put_argv_drop(heap, w, args, names.name(nm, 0));")]),
    M("repair-drop-with-steps", "--drop is offered although a step names the column", [(FF, "!o.plan_given && !o.step_given && !o.cat_given && !o.drop_given", "!o.plan_given && !o.drop_given")]),
    M("repair-plan-gets-an-argv-collinear", "a plan form gets an argv (--l2 cannot be given beside --plan)", [(FF, "        if o.plan_given {\n            w = plan_repair(heap, w);\n        } else if can_drop {", "        if false {\n            w = plan_repair(heap, w);\n        } else if can_drop {")]),
    M("repair-plan-gets-an-argv-separable", "a plan form gets an argv (--l2 cannot be given beside --plan)", [(FF, "        if o.plan_given {\n            w = plan_repair(heap, w);\n        } else {", "        if false {\n            w = plan_repair(heap, w);\n        } else {")]),
    # ---- separation
    M("loss-bound-too-high", "the loss bound is 2 (an overlapping set is refused)", [(LOG, "return 0.6931;", "return 2.0;")]),
    M("loss-bound-too-low", "the loss bound is 0.3 (the refusal comes later than the exact pass)", [(LOG, "return 0.6931;", "return 0.3;")]),
    M("loss-test-removed", "the loss test never fires", [(LOG, "if loss < separable_bound() {", "if false {")]),
    M("loss-test-with-a-ridge", "the loss test also runs with l2 above 0", [(LOG, "        if code == 0 && l2 == 0.0 {", "        if code == 0 {")]),
    M("scale-test-removed", "the predictor-scale test never fires (quasi-complete separation is not seen)", [(LOG, "predictor_scale(p, nf, contents(wr), diag) > scale_bound()", "false")]),
    M("scale-bound-too-low", "the predictor-scale bound is 1 (an ordinary fit is refused)", [(LOG, "return 50.0;", "return 1.0;")]),
    M("scale-without-the-factor-4", "the predictor scale forgets that the Hessian of pass 0 is the Gram matrix over 4n", [(LOG, "return sqrt(4.0 * q);", "return sqrt(q);")]),
    M("late-pivot-form-removed", "a pivot failure after the first pass keeps ml.linalg.not-positive-definite with l2 0", [(LOG, "                } else if l2 == 0.0 {", "                } else if false {")]),
    M("late-pivot-form-with-a-ridge", "the late pivot form also refuses with l2 above 0", [(LOG, "                } else if l2 == 0.0 {", "                } else if true {")]),
    M("pass-number-off-by-one", "the pass in the detail is one too high", [(LOG, "diag[0] = float_of(it);", "diag[0] = float_of(it + 1);")]),
    M("norm-is-not-rooted", "the norm of the coefficients is the sum of squares", [(LOG, "diag[3] = sqrt(q);", "diag[3] = q;")]),
]

mc.MUTANTS = MUTANTS

if __name__ == "__main__":
    sys.exit(mc.main())

#!/usr/bin/env python3
"""Mutants for the boosting library and its adapter (docs/models-boost.md gate GB7): a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change to lib/models/boost (the gradient and hessian formulas, the shrinkage, the leaf value, the split gain, the default side, the
subsample stream, the validation split, the early-stopping rule, the sum order, the bounds, the reader of the payload, the score kernel) or to the registry numbers, the binary
rebuilt from the copy, and the tests of the gates it targets run against it (the Python oracle first: it writes the same bytes or the mutant is killed). KILLED: a test failed.
SURVIVED: every test passed, and the mutant must carry a reason (`why`). The machinery is that of scripts/mutants_cli.py and scripts/mutants_wire.py; the mutants of the wiring
(flags, the summary, early stopping's rule, describe) are scripts/mutants_wire_boost.py.

    python3 scripts/mutants_boost.py              all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_boost.py --only ID
    python3 scripts/mutants_boost.py --slice 0:40
    python3 scripts/mutants_boost.py --list
    python3 scripts/mutants_boost.py --check      every `old` text is found exactly once where it is expected (no build, no test)
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
GR = "lib/models/boost/grow.cho"
FI = "lib/models/boost/fit.cho"
MO = "lib/models/boost/model.cho"
PR = "lib/models/boost/predict.cho"
BP = "lib/models/boost/bparams.cho"
AD = "lib/models/boost/adapter.cho"
K = "lib/core/kinds.cho"

ORACLE = ["test_wire_boost_oracle"]
BOOST = ["test_wire_boost"]
F = "test_wire_boost_file.BoostFile."
RULES = ["test_rules"]


def M(id, gate, what, edits, tests, **opt):
    return dict(id=id, gate=gate, what=what, edits=[e if len(e) == 4 else (*e, 1) for e in edits], tests=tests, opt=opt)


MUTANTS = [
    # ------------------------------------------------------------------ the gradient and the hessian
    M("b-regression-gradient-sign", "GB1", "the gradient of the squared error is y - F", [(FI, "gv[r2] = fv[r2] - yv[r2];", "gv[r2] = yv[r2] - fv[r2];")], ORACLE),
    M("b-regression-hessian-two", "GB1", "the hessian of the squared error is 2", [(FI, "hv[r2] = 1.0;", "hv[r2] = 2.0;")], ORACLE),
    M("b-logistic-gradient-sign", "GB1", "the gradient of the logistic loss is y - p", [(FI, "gv[r2] = pp - yv[r2];", "gv[r2] = yv[r2] - pp;")], ORACLE),
    M("b-logistic-hessian-is-p", "GB1", "the hessian of the logistic loss is p", [(FI, "hv[r2] = pp * (1.0 - pp);", "hv[r2] = pp;")], ORACLE),
    M("b-logistic-probability-of-the-wrong-sign", "GB1", "the probability is taken of -F", [(FI, "let pp = logreg.proba_of(fv[r2]);", "let pp = logreg.proba_of(0.0 - fv[r2]);")], ORACLE),
    # ------------------------------------------------------------------ the base score
    M("b-base-score-regression-divides-by-n", "GB1", "the mean divides by all the rows, not the fit rows", [(FI, "f0 = sy / float_of(m);", "f0 = sy / float_of(n);")], ORACLE,
      why="with no early stopping m = n, so the fit rows are all the rows; the early-stopping cases of the oracle suite have a regression target, but the mean over n rows of the validation rows too differs only if the base is used before the first round: it IS used (F starts at it), so a case with early stopping kills this mutant if one is generated; if it survives the suite has no regression early-stopping case"),
    M("b-base-score-logit-inverted", "GB1", "the log-odds of the classification is inverted", [(FI, "f0 = math.log(float_of(n1) / float_of(m - n1));", "f0 = math.log(float_of(m - n1) / float_of(n1));")], ORACLE),
    M("b-base-score-is-the-rate", "GB1", "the base score of a classification is the rate, not its log-odds", [(FI, "f0 = math.log(float_of(n1) / float_of(m - n1));", "f0 = float_of(n1) / float_of(m);")], ORACLE),
    # ------------------------------------------------------------------ the leaf value and the shrinkage
    M("b-leaf-value-sign", "GB1", "the Newton step has the sign of the gradient", [(GR, "lv = (0.0 - gt) / (ht + l2);", "lv = gt / (ht + l2);")], ORACLE),
    M("b-leaf-value-without-the-penalty", "GB1", "the leaf value divides by the hessian alone", [(GR, "lv = (0.0 - gt) / (ht + l2);", "lv = (0.0 - gt) / ht;")], ORACLE),
    M("b-no-shrinkage", "GB1", "the leaf value is not multiplied by the learning rate", [(GR, "return eta * lv;", "return lv;")], ORACLE),
    M("b-shrinkage-twice", "GB1", "the learning rate is applied again when a row takes its leaf value", [(FI, "fv[i] = fv[i] + ov[g];", "fv[i] = fv[i] + p.lr * ov[g];")], ORACLE),
    M("b-leaf-floor-inverted", "GB1", "a leaf with a hessian below the floor takes the Newton step", [(GR, "if ht + l2 >= bparams.hmin() {\n        lv =", "if ht + l2 < bparams.hmin() {\n        lv =")], ORACLE),
    M("b-leaf-floor-removed", "GB1", "the Newton step is taken whatever the hessian", [(GR, "if ht + l2 >= bparams.hmin() {\n        lv =", "if true {\n        lv =")], ORACLE,
      why="the floor matters only for a leaf whose hessian plus l2 is below 1e-3: a classification with a saturated probability and l2 = 0. A division of 0 by 0 gives NaN and the fit is refused by the non-finite check, which the oracle reports as not judged; the generated cases never saturate a probability to that degree"),
    # ------------------------------------------------------------------ the split
    M("b-gain-test-accepts-zero", "GB1", "a split of zero gain is taken", [(GR, "found = wf[osc] - ps > 0.0;", "found = wf[osc] - ps >= 0.0;")], ORACLE + BOOST),
    M("b-gain-test-removed", "GB1", "the best candidate is always taken", [(GR, "found = wf[osc] - ps > 0.0;", "found = true;")], ORACLE + BOOST),
    M("b-parent-score-without-the-penalty", "GB1", "the parent's score forgets l2", [(GR, "let ps = gt * gt / (ht + p.l2);", "let ps = gt * gt / ht;")], ORACLE),
    M("b-left-score-without-the-penalty", "GB1", "the left side's score forgets l2", [(GR, "let sc = gl * gl / (hl + l2) + gr * gr / (hr + l2);", "let sc = gl * gl / hl + gr * gr / (hr + l2);")], ORACLE),
    M("b-right-score-without-the-penalty", "GB1", "the right side's score forgets l2", [(GR, "let sc = gl * gl / (hl + l2) + gr * gr / (hr + l2);", "let sc = gl * gl / (hl + l2) + gr * gr / hr;")], ORACLE),
    M("b-right-gradient-wrong", "GB1", "the right side's gradient sum is the node's minus the left's minus one", [(GR, "let gr = gt - gl;", "let gr = gt - gl - 1.0;")], ORACLE),
    M("b-tie-replaces", "GB1", "an equal score replaces the best", [(GR, "if wk[ob] == 0 || sc > wf[osc] {", "if wk[ob] == 0 || sc >= wf[osc] {")], ORACLE,
      why="a tie of two float scores needs two different candidates with the same score to the last bit: the generated data have a few hundred rows of real-valued columns, so the only ties are the duplicates one feature makes with itself (variant 0 and 1 of a node with no missing row are not both evaluated); the tree library's gate has a dedicated edge set for it and this one does not"),
    M("b-valid-left-min-leaf", "GB1", "the left side may have fewer than min-leaf rows", [(GR, "if nl >= ml && nr >= ml {", "if nr >= ml {")], ORACLE),
    M("b-valid-right-min-leaf", "GB1", "the right side may have fewer than min-leaf rows", [(GR, "if nl >= ml && nr >= ml {", "if nl >= ml {")], ORACLE),
    M("b-valid-min-leaf-strict", "GB1", "a side needs more than min-leaf rows", [(GR, "if nl >= ml && nr >= ml {", "if nl > ml && nr >= ml {")], ORACLE),
    M("b-hessian-floor-left-dropped", "GB1", "the left side's hessian floor is not tested", [(GR, "if hl >= bparams.hmin() && hr >= bparams.hmin() {", "if hr >= bparams.hmin() {")], ORACLE,
      why="a side with a hessian below 1e-3 and at least min-leaf rows happens only in a classification whose probabilities are near 0 or 1 on that side (p (1 - p) < 1e-3 / rows): after many rounds on separable data with a large learning rate; the generated cases are noisy, and the by-hand test of a separable set (test_a_perfectly_separable_classification_is_not_refused_at_any_penalty) has no missing cell and one split, so the floor is not what decides"),
    M("b-hessian-floor-right-dropped", "GB1", "the right side's hessian floor is not tested", [(GR, "if hl >= bparams.hmin() && hr >= bparams.hmin() {", "if hl >= bparams.hmin() {")], ORACLE,
      why="the same as the left floor"),
    M("b-variant-one-never-tried", "GB1", "the missing rows are never sent left", [(GR, "        if mc > 0 {\n            cand(wk, wf, ob, osc, nl + mc, gl + mg, hl + mh, 1,", "        if false {\n            cand(wk, wf, ob, osc, nl + mc, gl + mg, hl + mh, 1,")], ORACLE),
    M("b-variant-one-without-the-count", "GB1", "variant 1 forgets the missing rows in the left count", [(GR, "cand(wk, wf, ob, osc, nl + mc, gl + mg, hl + mh, 1,", "cand(wk, wf, ob, osc, nl, gl + mg, hl + mh, 1,")], ORACLE),
    M("b-variant-one-without-the-gradient", "GB1", "variant 1 forgets the missing rows' gradients", [(GR, "cand(wk, wf, ob, osc, nl + mc, gl + mg, hl + mh, 1,", "cand(wk, wf, ob, osc, nl + mc, gl, hl + mh, 1,")], ORACLE),
    M("b-variant-one-without-the-hessian", "GB1", "variant 1 forgets the missing rows' hessians", [(GR, "cand(wk, wf, ob, osc, nl + mc, gl + mg, hl + mh, 1,", "cand(wk, wf, ob, osc, nl + mc, gl + mg, hl, 1,")], ORACLE),
    M("b-missing-gradient-from-the-last-bin", "GB1", "the missing bin's gradient sum is read one place off", [(GR, "let mg = wf[nbins];", "let mg = wf[nbins - 1];")], ORACLE),
    M("b-missing-hessian-from-the-gradients", "GB1", "the missing bin's hessian sum is read from the gradients", [(GR, "let mh = wf[w1 + nbins];", "let mh = wf[nbins];")], ORACLE),
    M("b-default-side-flipped", "GB1", "a missing row follows the other side than the split chose", [(GR, "                if v == nbins {\n                    lf = bdl;", "                if v == nbins {\n                    lf = 1 - bdl;")], ORACLE),
    M("b-default-side-not-stored", "GB1", "the default side of a node is stored as 0", [(GR, "oi[2 * cap + ni] = bdl;", "oi[2 * cap + ni] = 0;")], ORACLE),
    M("b-partition-bin-boundary", "GB1", "a present row goes left at a strict comparison of its bin", [(GR, "                if v <= bb {\n                    lf = 1;", "                if v < bb {\n                    lf = 1;")], ORACLE),
    M("b-threshold-from-the-wrong-feature", "GB1", "the stored threshold is the edge of feature 0", [(GR, "ov[ni] = ed[bf * ne + bb];", "ov[ni] = ed[bb];")], ORACLE),
    M("b-depth-limit-off-by-one", "GB1", "a tree grows one level deeper than max-depth", [(GR, "if dep < p.max_depth && m >= 2 * p.min_leaf {", "if dep <= p.max_depth && m >= 2 * p.min_leaf {")], ORACLE),
    M("b-min-leaf-pretest", "GB1", "a node may split with fewer than 2 min-leaf rows", [(GR, "if dep < p.max_depth && m >= 2 * p.min_leaf {", "if dep < p.max_depth && m >= p.min_leaf {")], ORACLE,
      why="a node with fewer than 2 min-leaf rows has no valid candidate (each side needs min-leaf rows), so the pretest is an optimisation: the leaf is the same"),
    M("b-node-budget-off-by-one", "GB1", "a tree needs one node fewer than the budget to be refused", [(GR, "        if cnt > cap {", "        if cnt >= cap {")], BOOST + RULES),
    M("b-histogram-hessian-from-the-gradient", "GB1", "the hessian histogram sums gradients", [(GR, "wf[w1 + b] = wf[w1 + b] + hv[r];", "wf[w1 + b] = wf[w1 + b] + gv[r];")], ORACLE),
    M("b-node-hessian-total-from-the-gradient", "GB1", "the node's hessian total sums gradients", [(GR, "ht = ht + hv[r];", "ht = ht + gv[r];")], ORACLE),
    M("b-histogram-sums-from-the-top", "GB1", "the cumulative sum over bins runs from the last bin", [(GR, "        gl = gl + wf[b];\n        hl = hl + wf[w1 + b];", "        gl = gl + wf[nbins - 1 - b];\n        hl = hl + wf[w1 + nbins - 1 - b];")], ORACLE),
    # ------------------------------------------------------------------ the sample
    M("b-sample-compares-with-less-or-equal", "GB1", "a row is in the sample when its draw is at most K", [(FI, "if rng.uniform(rng.output(s), bparams.subsample_scale()) < subk {", "if rng.uniform(rng.output(s), bparams.subsample_scale()) <= subk {")], ORACLE),
    M("b-sample-stream-kind", "GB1", "the sample is drawn from the validation stream's kind", [(FI, "rng.derive(rng.derive(p.seed, kinds.stream_boost_sample()), round)", "rng.derive(rng.derive(p.seed, kinds.stream_boost_validation()), round)")], ORACLE),
    M("b-sample-stream-round", "GB1", "the sample stream of round r is that of round r + 1", [(FI, "rng.derive(rng.derive(p.seed, kinds.stream_boost_sample()), round)", "rng.derive(rng.derive(p.seed, kinds.stream_boost_sample()), round + 1)")], ORACLE),
    M("b-sample-ignores-the-seed", "GB1", "the sample stream does not depend on the seed", [(FI, "rng.derive(rng.derive(p.seed, kinds.stream_boost_sample()), round)", "rng.derive(rng.derive(0, kinds.stream_boost_sample()), round)")], ORACLE),
    M("b-sample-draw-per-row-of-n", "GB1", "the draws are made for every row, not only the fit rows' sequence", [(FI, "            var s = rng.derive(rng.derive(p.seed, kinds.stream_boost_sample()), round);\n            k = 0;\n            while k < m {\n                s = rng.advance(s);", "            var s = rng.derive(rng.derive(p.seed, kinds.stream_boost_sample()), round);\n            k = 0;\n            while k < m {\n                s = rng.advance(rng.advance(s));")], ORACLE),
    M("b-sample-full-when-k-max", "GB1", "a subsample of 1.0 draws like any other", [(FI, "        if subk == bparams.subsample_scale() {", "        if false {")], ORACLE,
      why="a draw below K = 1,000,000 is always true (uniform(1e6) is at most 999,999), so the sample is every fit row either way, in the same order: the shortcut only saves the draws"),
    M("b-sample-rounding", "GB1", "K is truncated, not rounded", [(BP, "return truncate(s * 1000000.0 + 0.5);", "return truncate(s * 1000000.0);")], ORACLE + BOOST,
      why="K differs only when subsample * 1e6 has a fractional part of 0.5 or more; the values of the suite (0.5, 0.8, 0.25, 0.999999) have none, and IEEE products like 0.8 * 1e6 = 800000.0000000001 truncate to the same K"),
    # ------------------------------------------------------------------ the scores
    M("b-scores-start-at-zero", "GB1", "the scores start at 0 whatever the base score", [(FI, "        fv[i] = f0;", "        fv[i] = 0.0;")], ORACLE),
    M("b-only-fit-rows-take-the-leaf", "GB1", "the rows outside the fit rows do not take the tree's value", [(FI, "        i = 0;\n        while i < n {\n            var g = 0;", "        i = 0;\n        while i < m {\n            var g = 0;")], ORACLE,
      why="with no early stopping m = n; the validation rows of early stopping are the ones that matter (their scores decide when to stop), and the suite's early-stopping cases stop at a round that does not depend on them only by luck: if this survives the cases of the suite never separate it"),
    M("b-walk-missing-by-bin-goes-left", "GB1", "a missing bin follows the opposite default side while the scores are updated", [(FI, "                    left = oi[2 * cap + g] == 1;", "                    left = oi[2 * cap + g] == 0;")], ORACLE),
    M("b-walk-bin-compare", "GB1", "a row walks left at a strict comparison of its bin", [(FI, "                    left = b <= oi[cap + g];", "                    left = b < oi[cap + g];")], ORACLE),
    # ------------------------------------------------------------------ early stopping and the validation split
    M("b-validation-rows-n-over-9", "GB1", "the validation part is a ninth of the rows", [(BP, "    return n / 10;", "    return n / 9;")], ORACLE),
    M("b-validation-stream-ignores-the-seed", "GB1", "the validation split does not depend on the seed", [(FI, "var s = rng.derive(p.seed, kinds.stream_boost_validation());", "var s = rng.derive(0, kinds.stream_boost_validation());")], ORACLE),
    M("b-validation-draw-range", "GB1", "the shuffle draws over all n rows at each step", [(FI, "let j = i + rng.uniform(rng.output(s), n - i);", "let j = rng.uniform(rng.output(s), n);")], ORACLE),
    M("b-validation-stream-kind", "GB1", "the validation split is drawn from the sample stream's kind", [(FI, "var s = rng.derive(p.seed, kinds.stream_boost_validation());", "var s = rng.derive(p.seed, kinds.stream_boost_sample());")], ORACLE),
    M("b-early-stop-equal-improves", "GB1", "an equal loss counts as an improvement", [(FI, "if best_round == 0 || vl < best {", "if best_round == 0 || vl <= best {")], ORACLE,
      why="two validation losses equal to the last bit in consecutive rounds need a round whose trees change no validation score (all leaves 0 for those rows): not reached by the generated data"),
    M("b-early-stop-patience-off-by-one", "GB1", "the fit stops one round later than the patience", [(FI, "                if since >= p.early {", "                if since > p.early {")], ORACLE),
    M("b-early-stop-keeps-the-last-round", "GB1", "the model keeps every round that ran, not the best", [(FI, "        pr[1] = best_round;", "        pr[1] = done;")], ORACLE),
    M("b-validation-loss-mean-over-m", "GB1", "the validation loss divides by the fit rows", [(FI, "let vl = sum / float_of(nv);", "let vl = sum / float_of(m);")], ORACLE,
      why="a constant factor of every round's loss does not change which round is best, nor the rule 'no improvement': the stopping round is the same (the loss itself is not stored)"),
    M("b-validation-loss-regression-not-squared", "GB1", "the regression validation loss is the absolute-looking (F - y)", [(FI, "return (f - y) * (f - y);", "return f - y;")], ORACLE),
    M("b-validation-loss-classification-without-log", "GB1", "the logistic loss forgets its log1p term", [(FI, "return (mx - y * f) + mexp.log1p(mexp.exp(0.0 - af));", "return mx - y * f;")], ORACLE),
    M("b-validation-loss-classification-sign", "GB1", "the logistic loss has y F with the wrong sign", [(FI, "return (mx - y * f) + mexp.log1p(mexp.exp(0.0 - af));", "return (mx + y * f) + mexp.log1p(mexp.exp(0.0 - af));")], ORACLE),
    M("b-early-off-reads-the-best-round", "GB1", "without early stopping the kept rounds are the best round (0)", [(FI, "    if p.early > 0 {\n        pr[1] = best_round;", "    if true {\n        pr[1] = best_round;")], ORACLE),
    # ------------------------------------------------------------------ the checks of the fit
    M("b-target-bound-removed", "GB6", "a regression target above 1e100 is accepted", [(FI, "if !(v == v && v - v == 0.0 && v <= big && v >= 0.0 - big) {", "if !(v == v && v - v == 0.0) {")], BOOST),
    M("b-fit-non-finite-check-removed", "GB6", "a non-finite score after a round is not refused", [(FI, "            if !(v - v == 0.0) {\n                return refuse.fit_non_finite();", "            if false {\n                return refuse.fit_non_finite();")], BOOST,
      why="not reachable: a regression target is bounded by 1e100 and n by 2^20, so no score or sum overflows (docs/models-boost.md section 3.3); the check defends a bound that someone might widen"),
    M("b-single-class-after-the-validation-split", "GB6", "a classification whose fit rows hold one class goes on", [(FI, "        if n1 == 0 || n1 == m {", "        if false {")], BOOST,
      why="the fit rows hold one class only when the validation part took the whole of the other class: the check of check_target on all rows has refused every file with one class, and a file with two classes of at least 10 rows each is not reduced to one by a tenth of its rows"),
    M("b-label-check-removed", "GB6", "a label other than 0 and 1 is read as 0", [(FI, "            } else if v != 0.0 {\n                return deny.not_binary();", "            } else if false {\n                return deny.not_binary();")], BOOST + RULES,
      why="the wiring maps every label to 0.0 or 1.0 before the library sees it (ml.label.* refusals are the wiring's), so the library's own check is a second line of defence"),
    # ------------------------------------------------------------------ the bounds
    M("b-bound-rounds-inclusive", "GB6", "10,000 rounds are refused", [(BP, "if p.rounds < 1 || p.rounds > rounds_max() {", "if p.rounds < 1 || p.rounds >= rounds_max() {")], BOOST),
    M("b-bound-learning-rate-one", "GB6", "a learning rate of 1 is refused", [(BP, "if !(p.lr > 0.0 && p.lr <= 1.0) {", "if !(p.lr > 0.0 && p.lr < 1.0) {")], BOOST),
    M("b-bound-learning-rate-zero", "GB6", "a learning rate of 0 is accepted", [(BP, "if !(p.lr > 0.0 && p.lr <= 1.0) {", "if !(p.lr >= 0.0 && p.lr <= 1.0) {")], BOOST),
    M("b-bound-l2-zero", "GB6", "an l2 of 0 is refused", [(BP, "if !(p.l2 >= 0.0 && p.l2 <= l2_max()) {", "if !(p.l2 > 0.0 && p.l2 <= l2_max()) {")], ORACLE + BOOST),
    M("b-bound-l2-max", "GB6", "an l2 above 1e100 is accepted", [(BP, "if !(p.l2 >= 0.0 && p.l2 <= l2_max()) {", "if !(p.l2 >= 0.0) {")], BOOST),
    M("b-bound-subsample-one", "GB6", "a subsample of 1 is refused", [(BP, "if !(p.subsample > 0.0 && p.subsample <= 1.0) || subsample_k(p.subsample) < 1 {", "if !(p.subsample > 0.0 && p.subsample < 1.0) || subsample_k(p.subsample) < 1 {")], BOOST),
    M("b-bound-subsample-above-one", "GB6", "a subsample above 1 is accepted", [(BP, "if !(p.subsample > 0.0 && p.subsample <= 1.0) || subsample_k(p.subsample) < 1 {", "if !(p.subsample > 0.0) || subsample_k(p.subsample) < 1 {")], BOOST),
    M("b-bound-subsample-k-zero", "GB6", "a subsample that rounds to K = 0 is accepted", [(BP, "|| subsample_k(p.subsample) < 1 {", "|| subsample_k(p.subsample) < 0 {")], BOOST),
    M("b-bound-early-needs-ten-rows", "GB6", "early stopping on 9 rows is accepted", [(BP, "if p.early > 0 && n < 10 {", "if p.early > 0 && n < 9 {")], BOOST + RULES),
    M("b-bound-min-leaf-above-n", "GB6", "a min-leaf above the rows is accepted", [(BP, "if p.min_leaf < 1 || p.min_leaf > n {", "if p.min_leaf < 1 {")], BOOST + RULES),
    M("b-bound-columns-4096", "GB6", "a fit on more than 4096 columns goes on and writes a file predict refuses", [(BP, "    if d > 4096 {\n        return refuse.too_many_columns();\n    }\n", "")], BOOST),
    M("b-bound-depth-ceiling", "GB6", "a depth of 30 is refused", [(BP, "if p.max_depth > params.depth_max() {", "if p.max_depth >= params.depth_max() {")], BOOST,
      why="a depth of exactly 30 needs a fit that the suite does not make (the node arrays of a tree of depth 30 on a few hundred rows are small, but the case is not in the suite); the boundary of 31 is tested"),
    M("b-bound-total-nodes", "GB6", "rounds times the nodes of a tree is compared with a strict ceiling", [(BP, "if p.rounds > nodes_total_max() / node_cap(p, fit_rows) {", "if p.rounds >= nodes_total_max() / node_cap(p, fit_rows) {")], BOOST),
    M("b-node-cap-depth", "GB6", "the cap of a tree's nodes forgets the +1 of the depth", [(BP, "c = min2(c, (1 << p.max_depth + 1) - 1);", "c = min2(c, (1 << p.max_depth) - 1);")], BOOST + ORACLE,
      why="the cap only decides a refusal when a tree needs more nodes than min(max-nodes, 2m - 1, 2^(D+1) - 1): at most 2^(D+1) - 1 nodes ever exist, so cap - 1 would refuse a full tree; a full tree of depth 1..8 on the suite's rows needs min-leaf 1 and a gain at every node, which the generated data rarely give for the larger depths"),
    M("b-copies-ceiling", "GB6", "the memory ceiling of the binning copies is not applied", [(BP, "    if n * d * 9 + p.threads * (n * d + 24 * n) > params.copies_max() {\n        return refuse.threads();\n    }\n", "")], BOOST,
      why="the smallest input that exceeds it is about 2.3e8 cells (a CSV of more than 400 MB): the tree's own gate lists it without a fixture (docs/wiring-trees.md section 6)"),
    # ------------------------------------------------------------------ the reader of the payload
    M("b-read-rounds-ceiling", "GB6", "the reader accepts more rounds than the ceiling", [(MO, "rounds < 1 || rounds > bparams.rounds_max() ||", "rounds < 1 ||")], [F + "test_the_header_of_the_payload", F + "test_counts_that_wrap_the_size_are_refused_before_any_allocation"],
      why="adapter.lib_size applies the same ceiling on the rounds (and on the nodes) to the header before model.read is called, so a file with a wrapped or huge count never reaches the reader's own check; the two are one bound written twice, each the other's second line of defence (b-adapter-size-ceiling removes the adapter's)"),
    M("b-read-columns-ceiling", "GB6", "the reader accepts more than 4096 columns", [(MO, "|| d < 1 || d > 4096 ||", "|| d < 1 ||")], [F + "test_the_header_of_the_payload"],
      why="the number of columns of the payload is compared with the width of the schema by the adapter (fd != d, mutated separately as b-adapter-width-not-compared), and the schema is bounded by --max-columns; the reader's own ceiling of 4096 is a second line of defence that a one-field edit of a real file cannot isolate"),
    M("b-read-bins-range", "GB6", "the reader accepts 256 bins", [(MO, "nbins < 2 || nbins > 255", "nbins < 2")], [F + "test_the_header_of_the_payload"]),
    M("b-read-depth-ceiling", "GB6", "the reader accepts a depth above 30", [(MO, "max_depth < 1 || max_depth > params.depth_max() ||", "max_depth < 1 ||")], [F + "test_the_header_of_the_payload"]),
    M("b-read-min-leaf", "GB6", "the reader accepts a min-leaf of 0", [(MO, "|| min_leaf < 1 {", "{")], [F + "test_the_header_of_the_payload"]),
    M("b-read-early-range", "GB6", "the reader accepts an early-stopping above its ceiling", [(MO, "if early < 0 || early > bparams.early_max() ||", "if early < 0 ||")], [F + "test_the_header_of_the_payload"]),
    M("b-read-requested-below-rounds", "GB6", "the reader accepts fewer rounds requested than kept", [(MO, "requested < rounds || requested > bparams.rounds_max() ||", "requested > bparams.rounds_max() ||")], [F + "test_the_header_of_the_payload"]),
    M("b-read-requested-ceiling", "GB6", "the reader accepts more rounds requested than the ceiling", [(MO, "requested < rounds || requested > bparams.rounds_max() ||", "requested < rounds ||")], [F + "test_the_header_of_the_payload"]),
    M("b-read-early-off-fewer-rounds", "GB6", "a file without early stopping may hold fewer rounds than requested", [(MO, "early == 0 && requested != rounds {", "false {")], [F + "test_the_header_of_the_payload"]),
    M("b-read-subsample-k-zero", "GB6", "the reader accepts K = 0", [(MO, "if subk < 1 || subk > bparams.subsample_scale() ||", "if subk > bparams.subsample_scale() ||")], [F + "test_the_header_of_the_payload"]),
    M("b-read-subsample-k-ceiling", "GB6", "the reader accepts K above 1,000,000", [(MO, "if subk < 1 || subk > bparams.subsample_scale() ||", "if subk < 1 ||")], [F + "test_the_header_of_the_payload"]),
    M("b-read-nodes-below-rounds", "GB6", "the reader accepts fewer nodes than rounds", [(MO, "|| total < rounds || total > bparams.nodes_total_max() {", "|| total > bparams.nodes_total_max() {")], [F + "test_the_header_of_the_payload"],
      why='a payload with fewer nodes than rounds fails the offsets (strictly increasing from 0 to the node count, one per round), so the header comparison is a second line of defence that is never the only check'),
    M("b-read-nodes-ceiling", "GB6", "the reader accepts more nodes than the ceiling", [(MO, "|| total < rounds || total > bparams.nodes_total_max() {", "|| total < rounds {")], [F + "test_the_header_of_the_payload", F + "test_counts_that_wrap_the_size_are_refused_before_any_allocation"],
      why='the same as b-read-rounds-ceiling: adapter.lib_size refuses a node count above the ceiling before model.read runs'),
    M("b-read-truncated-is-malformed", "GB6", "a payload shorter than its header asks is not reported as truncated", [(MO, "    if n < size {\n        return bad(heap, deny.model_truncated());", "    if n < size {\n        return bad(heap, deny.model_malformed());")], [F + "test_the_length_of_the_payload"],
      why="every caller maps both tags to ml.model.malformed (the wiring's decoder answers one code), so the distinction is not visible from outside; the file-level truncation is the reader of the sections"),
    M("b-read-longer-payload", "GB6", "a payload longer than its header asks is accepted", [(MO, "    if n > size {\n        return bad(heap, deny.model_malformed());", "    if false {\n        return bad(heap, deny.model_malformed());")], [F + "test_the_length_of_the_payload"],
      why="the adapter compares the length with the size it computed before calling the reader (a second line of defence, mutated separately): the reader's own check is never reached by a file"),
    M("b-read-floats-record-header", "GB6", "the floats record header is not compared", [(MO, "if !rec_ok(data, 100, 2, 3) || !rec_ok(data, 140, 1, rounds + 1) {", "if !rec_ok(data, 140, 1, rounds + 1) {")], [F + "test_the_kind_and_the_record_headers"]),
    M("b-read-offsets-record-header", "GB6", "the offsets record header is not compared", [(MO, "if !rec_ok(data, 100, 2, 3) || !rec_ok(data, 140, 1, rounds + 1) {", "if !rec_ok(data, 100, 2, 3) {")], [F + "test_the_kind_and_the_record_headers"]),
    M("b-read-node-record-headers", "GB6", "the node records' headers are not compared", [(MO, "        if !rec_ok(data, record_at(r, rounds, total), dt, total) {\n            return bad(heap, deny.model_malformed());", "        if false {\n            return bad(heap, deny.model_malformed());")], [F + "test_the_kind_and_the_record_headers"]),
    M("b-read-kind", "GB6", "the reader does not compare the kind", [(MO, "if get32(data, 0) != kind() || !rec_ok(data, 4, 1, 10) {", "if !rec_ok(data, 4, 1, 10) {")], [F + "test_the_kind_and_the_record_headers"],
      why="the file reader (pipeline.file.decode) has already compared the kind with the plan's family before the adapter runs, and the adapter's own size check reads the header at fixed offsets: a payload of another kind with a valid layout cannot be built by a one-field edit"),
    M("b-read-header-record", "GB6", "the header record's own header is not compared", [(MO, "if get32(data, 0) != kind() || !rec_ok(data, 4, 1, 10) {", "if get32(data, 0) != kind() {")], [F + "test_the_kind_and_the_record_headers"]),
    M("b-read-learning-rate-range", "GB6", "the reader accepts a learning rate of 0 or above 1", [(MO, "lr > 0.0 && lr <= 1.0 && l2", "lr == lr && l2")], [F + "test_the_floats_of_the_payload"]),
    M("b-read-l2-range", "GB6", "the reader accepts a negative l2", [(MO, "l2 >= 0.0 && l2 <= bparams.l2_max()", "l2 == l2")], [F + "test_the_floats_of_the_payload"]),
    M("b-read-base-finite", "GB6", "the reader accepts an infinite base score", [(MO, "if !(base - base == 0.0 && lr", "if !(lr")], [F + "test_the_floats_of_the_payload"]),
    M("b-read-float-bits", "GB6", "the reader does not require the bits of the floats to be reproduced", [(MO, "    if bits_of(base) != fbits.get64(data, 116) || bits_of(lr) != fbits.get64(data, 124) || bits_of(l2) != fbits.get64(data, 132) {", "    if false {")], [F + "test_the_floats_of_the_payload"],
      why="a float whose bits are not reproduced by fbits.decode is an infinity or a NaN (decode answers NaN for them) or a subnormal; the range checks that follow refuse every NaN (a comparison is false) and the subnormal l2 or base is accepted either way: the bit check is a second line of defence for the values the comparisons let through"),
    M("b-read-offset-start", "GB6", "the first tree may start after node 0", [(MO, "if tof[0] != 0 || tof[rounds] != total {", "if tof[rounds] != total {")], [F + "test_the_tree_offsets_and_the_nodes"]),
    M("b-read-offset-end", "GB6", "the last offset may differ from the node count", [(MO, "if tof[0] != 0 || tof[rounds] != total {", "if tof[0] != 0 {")], [F + "test_the_tree_offsets_and_the_nodes"],
      why="a last offset other than the node count either ends the last tree early (a right child still pending) or runs the walk past the arrays' own end of that tree, and check_tree refuses both: the comparison with the node count is a second line of defence"),
    M("b-read-offsets-increase", "GB6", "an empty tree is accepted", [(MO, "if tof[t + 1] <= tof[t] {", "if tof[t + 1] < tof[t] {")], [F + "test_the_tree_offsets_and_the_nodes"],
      why="an empty tree would make the next tree start where this one does, and the next tree's walk then meets a root after a complete tree with nothing pending, which check_tree refuses: the strict increase is a second line of defence"),
    M("b-read-feature-range", "GB6", "a feature number past the columns is accepted", [(MO, "if feat < 0 || feat >= d || bin < 0", "if feat < 0 || bin < 0")], [F + "test_the_tree_offsets_and_the_nodes"]),
    M("b-read-bin-range", "GB6", "a bin past the last split bin is accepted", [(MO, "|| bin > nbins - 2 ||", "||")], [F + "test_the_tree_offsets_and_the_nodes"]),
    M("b-read-dleft-range", "GB6", "a default side of 2 is accepted", [(MO, "|| dl != 0 && dl != 1 {", "{")], [F + "test_the_tree_offsets_and_the_nodes"]),
    M("b-read-left-is-next", "GB6", "a left child that is not the next node is accepted", [(MO, "if lc != i + 1 || cur >= max_depth {", "if cur >= max_depth {")], [F + "test_the_tree_offsets_and_the_nodes"]),
    M("b-read-depth-of-a-tree", "GB6", "a tree deeper than the header's max-depth is accepted", [(MO, "if lc != i + 1 || cur >= max_depth {", "if lc != i + 1 {")], [F + "test_the_tree_offsets_and_the_nodes", F + "test_the_header_of_the_payload"]),
    M("b-read-leaf-fields", "GB6", "a leaf with a child or a bin is accepted", [(MO, "if bin != 0 - 1 || dl != 0 || lc != 0 - 1 || rc != 0 - 1 {", "if dl != 0 {")], [F + "test_the_tree_offsets_and_the_nodes"]),
    M("b-read-pending-children", "GB6", "a tree that ends with a right child still to come is accepted", [(MO, "    if stack != 0 {\n        return deny.model_malformed();\n    }\n    return 0;\n}\n\n// Read a payload", "    return 0;\n}\n\n// Read a payload")], [F + "test_the_tree_offsets_and_the_nodes"],
      why="the same argument as the tree library's reader (docs/models-tree.md section 7): a pending right child can never be popped by a node of the tree, and a pushed right child at the end belongs to a last node that is internal, which has its left child missing: every one-field edit that leaves a child pending also breaks the offsets or the left-is-next rule first"),
    M("b-read-pop-index", "GB6", "a node that follows a leaf may be any index", [(MO, "                if st[stack] != i {", "                if false {")], [F + "test_the_tree_offsets_and_the_nodes"]),
    # ------------------------------------------------------------------ the score kernel
    M("b-predict-starts-at-zero", "GB1", "the score of a row starts at 0", [(PR, "var acc = base;", "var acc = 0.0;")], ORACLE),
    M("b-predict-threshold-equal-goes-left", "GB1", "a value equal to the threshold goes left", [(PR, "left = row[f] < nv[g];", "left = row[f] <= nv[g];")], ORACLE),
    M("b-predict-default-side-flipped", "GB1", "a missing cell follows the other side", [(PR, "left = dl[g] == 1;", "left = dl[g] == 0;")], ORACLE),
    M("b-predict-skips-the-last-tree", "GB1", "the last round's tree is not added", [(PR, "while t < rounds {", "while t < rounds - 1 {")], ORACLE),
    M("b-predict-adds-the-wrong-node", "GB1", "the value of the parent of the leaf is added", [(PR, "acc = acc + nv[g];", "acc = acc + nv[b0];")], ORACLE),
    M("b-predict-descends-right-first", "GB1", "left and right swapped in the walk", [(PR, "                if left {\n                    g = b0 + lc[g];\n                } else {\n                    g = b0 + rc[g];", "                if left {\n                    g = b0 + rc[g];\n                } else {\n                    g = b0 + lc[g];")], ORACLE),
    M("b-predict-ignores-the-bitmap", "GB1", "the bitmap of missing cells is not read", [(PR, "if len(bm) != 0 && bins.is_missing(bm, st, f, i) {", "if false {")], ORACLE),
    # ------------------------------------------------------------------ the adapter
    M("b-adapter-rounds-from-the-depth", "GB1", "the rounds are read from the place of max-depth", [(AD, "rounds: ip[1],", "rounds: ip[2],")], ORACLE + BOOST),
    M("b-adapter-depth-from-the-leaf", "GB1", "max-depth is read from the place of min-leaf", [(AD, "max_depth: ip[2],", "max_depth: ip[3],")], ORACLE + BOOST),
    M("b-adapter-min-leaf-from-the-bins", "GB1", "min-leaf is read from the place of bins", [(AD, "min_leaf: ip[3],", "min_leaf: ip[4],")], ORACLE + BOOST),
    M("b-adapter-bins-from-the-early", "GB1", "bins are read from the place of early-stopping", [(AD, "bins: ip[4],", "bins: ip[5],")], ORACLE + BOOST),
    M("b-adapter-early-from-the-nodes", "GB1", "early-stopping is read from the place of max-nodes", [(AD, "early: ip[5],", "early: ip[6],")], ORACLE + BOOST + RULES),
    M("b-adapter-max-nodes-ignored", "GB1", "max-nodes is not read", [(AD, "max_nodes: ip[6],", "max_nodes: 4194304,")], BOOST + RULES),
    M("b-adapter-learning-rate-from-l2", "GB1", "the learning rate is read from the place of l2", [(AD, "lr: fp[0],", "lr: fp[1],")], ORACLE + BOOST),
    M("b-adapter-l2-from-the-subsample", "GB1", "l2 is read from the place of subsample", [(AD, "l2: fp[1],", "l2: fp[2],")], ORACLE + BOOST),
    M("b-adapter-subsample-from-the-rate", "GB1", "the subsample is read from the place of the learning rate", [(AD, "subsample: fp[2],", "subsample: fp[0],")], ORACLE + BOOST),
    M("b-adapter-task-ignored", "GB1", "the task is always a regression", [(AD, "let p = bparams.Params { task: ip[0],", "let p = bparams.Params { task: 1,")], ORACLE + BOOST),
    M("b-adapter-seed-is-the-threads", "GB2", "the seed handed to the library is the thread count", [(AD, "seed: seed, threads: threads };", "seed: threads, threads: threads };")], ORACLE + BOOST),
    M("b-adapter-threads-minus-one", "GB2", "the library gets one thread fewer than --threads", [(AD, "seed: seed, threads: threads };", "seed: seed, threads: threads - 1 };")], BOOST + ORACLE),
    M("b-adapter-threads-not-passed", "GB2", "the library is given one thread whatever --threads says", [(AD, "seed: seed, threads: threads };", "seed: seed, threads: 1 };")], BOOST + ORACLE,
      why="an equivalent mutant by design: no thread count changes a byte of the model (the property GB2 asserts, and the binning is the only threaded part), so a fit that ignores the flag writes the same file"),
    M("b-adapter-class-of-one-half", "GB1", "a probability of exactly one half is the positive class", [(AD, "                if pr > 0.5 {", "                if pr >= 0.5 {")], ORACLE + BOOST,
      why="a probability of exactly 0.5 needs a score whose sigmoid rounds to one half: |score| below 1e-16, a sum of leaf values that cancel to the last bit: the generated rows never reach it (the same tie as the tree's `wt-class-of-a-tie`)"),
    M("b-adapter-proba-mode-is-the-class", "GB1", "the probability of a classification is its class", [(AD, "    return predict_rows(heap, blob, xt, xd, miss, wpc, out, 1);", "    return predict_rows(heap, blob, xt, xd, miss, wpc, out, 0);")], ORACLE + BOOST),
    M("b-adapter-proba-of-a-regression", "GB1", "a regression answers a probability", [(AD, "    if mode == 1 && task != 0 {", "    if false {")], BOOST + RULES,
      why="a second line of defence: predict --proba on a regression is args.conflict before the adapter is called and eval asks for a probability only of a classification"),
    M("b-adapter-labels-not-written", "GB1", "the labels record of a classification is not written", [(AD, "            if p.task == 0 {\n                record.encode_labels(contents(ow), size, c0, c1);", "            if false {\n                record.encode_labels(contents(ow), size, c0, c1);")], ORACLE + BOOST),
    M("b-adapter-labels-length", "GB6", "the decoder accepts bytes after the payload", [(AD, "    if len(src) != want {", "    if len(src) < want {")], [F + "test_the_length_of_the_payload"]),
    M("b-adapter-labels-record-unchecked", "GB6", "the labels record of a classification is read without its checks", [(AD, "    if lab.code != 0 || lab.end != len(src) {", "    if false {")], [F + "test_the_labels_record_of_a_classification"]),
    M("b-adapter-width-not-compared", "GB6", "the payload's number of columns is not compared with the schema's", [(AD, "    if code != 0 || fd != d {", "    if code != 0 {")], BOOST + [F + "test_the_header_of_the_payload"]),
    M("b-adapter-size-ceiling", "GB6", "the size computed from the header is used with a node count above the ceiling", [(AD, "task < 0 || task > 1 || rounds < 1 || rounds > bparams.rounds_max() || total < rounds || total > bparams.nodes_total_max() {", "task < 0 || task > 1 || rounds < 1 || rounds > bparams.rounds_max() || total < rounds {")], [F + "test_the_header_of_the_payload"],
      why="a total above the ceiling gives a size the file cannot have (the length comparison that follows is malformed first) or, when the size equation holds, the library reader refuses the total again: two checks of one bound"),
    # ------------------------------------------------------------------ the registry
    M("b-kind-number", "GB11", "the kind of a boosted model is 9", [(K, "pub fn model_boost() -> [] int {\n    return 8;", "pub fn model_boost() -> [] int {\n    return 9;")], BOOST + ORACLE),
    M("b-sum-order-of-kind-8-is-2", "GB1", "a boosted model records the lane order", [(K, "    if kind == model_boost() {\n        return sum_order_sequential();", "    if false {\n        return sum_order_sequential();")], BOOST + ORACLE),
    M("b-sample-stream-number", "GB1", "the sample stream is kind 6", [(K, "pub fn stream_boost_sample() -> [] int {\n    return 7;", "pub fn stream_boost_sample() -> [] int {\n    return 6;")], ORACLE),
    M("b-validation-stream-number", "GB1", "the validation stream is kind 9", [(K, "pub fn stream_boost_validation() -> [] int {\n    return 8;", "pub fn stream_boost_validation() -> [] int {\n    return 9;")], ORACLE),
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
        print("%d boost mutants, %d with an edit that does not apply" % (len(MUTANTS), bad))
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

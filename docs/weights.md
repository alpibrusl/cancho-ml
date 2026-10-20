# cancho-ml sample weights (issue 59 item 1, stage C of `multiclass.md`)

**Status: a specification, written BEFORE any code of this stage. Every number that a script of this repository has not printed is marked [choice], [open] or [read]; section 7 is filled in after the gates have run.** Tags as in `design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from a document or the code, **[open]** not decided or not yet checked. A claim found false is corrected in place and marked.

Nothing weights a row today: no flag of the tool takes a weight, and the tree, forest and boost lists name sample weights as not built (**[read]** `models-tree.md` section 1, `models-boost.md` section 1). Weights are needed for imbalanced data and for importance weights, and they are the first piece of the logged-bandit line of issue 59 (off-policy evaluation and policy learning need the inverse of a propensity as a weight).

## 1. The flag and its meaning

`--weight COL` on `fit` and `cv`: the column that holds the weight of each row. **[choice]** `eval` takes it in a later stage (the weighted metrics are a section of their own: their exact-integer forms become float sums); `predict` and `ml-serve` take no weight, so asking for one is the refusal of an unknown flag as today.

* The weight column is a numeric column of the file; it is **not a feature** (the model's schema does not list it) and it is not the target (`args.conflict`, exit 2, when `--weight` names `--target`). It is read as a float, and a column named by `--drop` or `--cat` is not usable as the weight (`args.conflict`).
* A weight is finite and `0 <= w <= W_max`, `W_max` = 1e12 **[choice]**: a bound under which the sums of up to `2^20` rows stay far from overflow and from the point where adding a small weight to a large sum loses all of its digits for the sizes the tool takes (**[open]** the second claim is checked by gate WG6, not assumed). A negative weight is `ml.weight.negative`, a non-finite one `ml.input.non-finite` as any cell, a missing cell `ml.weight.missing`, one above `W_max` `ml.weight.too-large` (all exit 8, the line and column in `detail`). The sum of all weights being 0 is `ml.weight.zero-total` (exit 8).
* **The meaning of a weight `w` is `w` copies of the row** in every loss, gradient, hessian, impurity and mean the fit computes. Counts that are limits or guards still count rows, not weights **[choice]**: `min-leaf`, the row ceilings, the validation split of early stopping and the stratified splits of `cv` are about rows. A row of weight 0 is a row that contributes nothing to any sum; the bins of the trees are computed from all the rows given to the fit, as today (**[read]** `models-boost.md` 2.1), so removing a zero-weight row from the file changes the bin edges of a tree and the equivalence of gate WG3 does not claim it for the models that bin.
* The weights are not normalised, and the regularisation keeps the form each model has (**[read]** `models-linear.md`): the ridge objective is the **sum** of squares plus `l2 ||w||^2`, so weights `c w` with `c l2` give the model of weights `w` with `l2`; the logistic objective is the **mean** negative log-likelihood plus `l2 / 2 ||w||^2` (the form that makes the oracle's scikit-learn `C = 1 / (l2 n)` correspond, `models-linear.md`), and with weights the mean is taken over the **sum of the weights**, so that a weight `w` is `w` copies of the row and the weights `c w` with the same `l2` give the same model. These are gate WG4, stated for those two models only.

## 2. Model by model

| model | what the weight multiplies | arithmetic |
|---|---|---|
| `linear` | each row's term in the normal equations, `X' W X` and `X' W y` | the decisive sums of `linalg.order`, now with the weight in each term, in the same block order, so the result is the same at any thread count **[open]** the weighted Gram matrix goes through the same Cholesky and the same relative pivot test; the test is a ratio, so it is unchanged in kind |
| `logistic` | each row's gradient and hessian term, and its term of the summed loss | the same sums; the separation test compares the **weighted** summed loss with its bound, which assumes weights of at least 1 to be meaningful for a count and is otherwise a different quantity: **[open]** gate WG7 decides whether the test is applied to the weighted loss divided by the sum of weights, and the answer is written in section 7 |
| `tree`, `forest` regression | the node sums `Sm`, `s_b` and the node size used in the gain | the sequential float sums in the node's row order of `models-tree.md` 3.2, with `w_i y_i` and `w_i` accumulated |
| `tree`, `forest` classification | the class counts of a node | **a new path**: the exact integer Gini of `models-tree.md` 3.1 needs integer counts and cannot take a float weight. A weighted classification tree computes the Gini decrease in floats from the weighted class sums, in the node's row order **[choice]**. The unweighted path and its exact comparison are unchanged and are the only path used when no `--weight` is given |
| `forest` | the bootstrap draw stays a draw of rows (uniform, stream kind of the forest); the weight enters each tree's sums | |
| `boost` | the gradient and the hessian of each row, so the Newton leaf `-G / (H + lambda)` and the split gain | the gradient and hessian are multiplied by `w_i` before they enter a histogram; the row subsample draws rows, not weights |
| `kmeans` | **not supported**: `--weight` with `--model kmeans` is `args.conflict` with the repair "drop --weight: k-means here has no weighted centroid" **[choice]** | |
| `constant` | the weighted mean or the weighted majority | **[open]** the exact-integer form of the constant model's file (`mfile.cho`: counts as i64) cannot hold a float weighted mean; `--weight` with `--model constant` is `args.conflict` until a weighted form is chosen |

## 3. The file

A model fitted with weights is the same kind with the same payload shape: nothing of the weights is needed to predict. **[choice]** The provenance section (**[read]** `docs/tracking.md` section 2) records that a weight column was used and its name, so `describe` can say so and `ml runs verify` replays the fit with the same flag (the run record keeps the command line; **[open]** whether the record needs a field of its own or the flag in the command is enough is checked by gate WG8). Whether this is a new member of the provenance section, and so a change of the optional section's bytes only for a weighted fit, is decided at the review of this document; a file with no weights keeps its bytes.

## 4. Where the work is

`tools/ml` (the flag in `spec`, the reading of the column in `wireio` and `csvload`, `wire` and `verbs` to hand the weights to the fits), `lib/models/linear`, `lib/models/tree`, `lib/models/boost` (the sums), `lib/core` for the refusal codes, `mdesc` and the schemas for the `describe` member. The generated tables change where a flag or a tag appears.

## 5. Gates (written first; each can fail)

| gate | what | must be able to fail by |
|---|---|---|
| WG1 | **binary bytes unchanged**: with no `--weight`, every file and answer is equal to the one before the stage, by the fingerprint golden of `docs/bench/xplat_fp.txt` on every platform | a weighted code path taken without the flag |
| WG2 | **all weights 1.0 equal no weights**, as bytes, for the models whose sums are floats in a fixed order (linear, logistic, regression tree and forest, boost) **[open]**: this should hold because `w * x` with `w = 1.0` is `x` exactly; where it does not (a sum that is reassociated for the weights), the difference is explained in section 7 and the gate is a bound | a multiplication that is applied to one sum and not another |
| WG3 | **duplication**: integer weights `w_i` give the model of the file in which row `i` appears `w_i` times, for the linear and logistic regression within a bound fixed in advance on the coefficients (the sums are in a different order, so bytes are not claimed), and for the trees and boost as a property of their splits on data without ties in the bins | a weight applied once too often |
| WG4 | **scale**: weights `c w` with `c l2` equal weights `w` with `l2` for the linear regression, and weights `c w` with the same `l2` equal weights `w` for the logistic regression, within a bound fixed in advance (a power-of-two `c` is exact in the products and is the first control) | a penalty that is not scaled, a mean taken over the rows instead of the weights |
| WG5 | **against an independent oracle**: a Python implementation of each weighted sum in the stated order matches the library as bytes for the linear, logistic, tree, forest and boost, and `scikit-learn`'s `sample_weight` is compared as a **bound fixed in advance**, with its outcome written in section 7 | a weight dropped from the hessian |
| WG6 | **numeric bounds**: weights at `W_max` and at `1e-300`, `2^20` rows, no overflow, no NaN, and the refusals of section 1 each reached by a fixture | a bound moved by one |
| WG7 | **separation** with weights: a fixture of separable classes with `l2 = 0` is refused and one with a few rows of weight `0` is judged on the rows that have weight | a test that reads the unweighted loss |
| WG8 | **replay and provenance**: `ml runs verify` of a weighted fit gives the same bytes | a record that loses the flag |
| WG9 | **thread invariance** (1, 2, 4, 8) of every weighted fit; **fuzz** of hostile weights (zero rows, all zero, negative, `nan`, `inf`, `1e308`, an empty cell, the target or a dropped column named as the weight) and of damaged files; **mutants** of the new code, all killed or each survivor explained; **admission**; no source file over 2,000 lines | — |

## 6. What this does not do

No weighted `eval` yet (**[choice]**, section 1), no per-class weights flag (`--class-weight balanced` is a function of the labels and is a later, smaller addition once `--weight` exists), no weights for `kmeans` or the constant model, no change to the splits of `cv`.

## 7. Results

Empty until the gates have run. A number is written here only if a script of this repository printed it, with the machine, the threads and the minimum of N.

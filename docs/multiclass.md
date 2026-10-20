# cancho-ml multiclass classification (issue 37, stage M16; boosting follows as the second half of M15)

**Status: a specification, written BEFORE any code of this stage (the first commit of the branch holds only this file and `weights.md`). Every number that a script of this repository has not printed is marked [choice], [open] or [read]; section 9 is filled in after the gates have run.** Tags as in `design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from a document or the code, **[open]** not decided or not yet checked. A claim found false is corrected in place and marked.

Today a third label is `ml.label.not-binary` (exit 8) in every classifier: the constant model, the logistic regression, the tree, the forest and the boost (**[read]** `lib/core/refuse.cho`, `lib/models/tree/deny.cho`, `tools/ml/wire.cho`). The model file holds `labels` as an i64 tensor of shape [2] for the logistic model and a class count of 2 for the constant one (**[read]** `docs/wiring.md` section 3, `tools/ml/mfile.cho`). This stage lets a classification target have `K` integer labels, `3 <= K <= K_max`, in every classifier that can carry it, one model family per sub-stage, and changes no byte of any binary file.

## 1. Scope and order

The order is a **[choice]**; each stage is a pull request of its own, ends with its gates passing, and can stay the last one built.

| stage | what | why here |
|---|---|---|
| A | the multiclass **metrics** in `lib/metrics`: confusion matrix, accuracy, per-class precision, recall and F1, their macro, micro and weighted averages, log-loss, one-versus-rest ROC-AUC | they depend on no model, and every later stage is judged by them |
| B | **multinomial logistic regression** (`--model logistic`, a new model kind), the label coding, `fit` / `predict` / `eval` / `describe` / `ml-serve` for it | the smallest model that exercises the whole path: file, wiring, answers, serving |
| C | **sample weights** (`weights.md`), all fits | the tree and boosting loops are rewritten once, not twice |
| D | **trees and forests** with class histograms | needs the label path of B and the weights of C |
| E | **boosting** with `K` trees per round | needs D's histograms in the leaf and C's weights in the gradient |
| F | the interpretation verbs (`importance`, `dump`), `ml-mcp` and JSON lines for the new answers | they already refuse a model they cannot explain (`ml.importance.unsupported-model`), so F can follow each stage |

**Not in this stage** (**[choice]**): multilabel targets (several labels per row), an ordinal target (the order of the labels carries meaning), a class prior or a cost matrix, calibration, a hierarchy of classes, k-nearest neighbours and naive Bayes (M8, issue 9).

## 2. Labels

* A classification target is an integer column. With `K` distinct values in the training rows, `K = 1` is `ml.label.single-class` as today, `K = 2` is the binary path as today (byte for byte), `3 <= K <= K_max` is the multiclass path, `K > K_max` is a new refusal, `ml.limit.classes` (exit 8) with `K` and `K_max` in `detail`.
* `K_max` = 64 **[choice]**. The reason is the memory of the Newton step of section 3 and the width of the answer, not a property of the models; the number is in the limits table (`introspect`) and is raised by a measured change, not by a flag. **[open]** the cost of `K = 64` with 100 features is computed in section 3.3 and measured by gate M2.
* The labels are stored **sorted ascending** in the model file, so class `k` is the `k`-th smallest label. A label the model never saw, at `eval`, is `ml.label.not-binary` for a binary model today and `ml.label.unseen` (exit 8, the label and the classes in `detail`) for a multiclass one **[choice]**: the old tag names a property (not two classes) that is no longer the issue.
* `--positive` is meaningful for a binary model only. With a multiclass model it is `args.conflict` (exit 2) with the repair "drop --positive: the metrics are those of every class" **[choice]**. A one-versus-rest report of one class is the per-class line of section 4.

## 3. Multinomial logistic regression (stage B)

### 3.1 The model

`K` classes, `d` features after the steps, class scores `z_k = w_k . x + b_k`, probabilities `p_k = exp(z_k) / sum_j exp(z_j)`. The softmax is invariant to adding one vector to every `(w_k, b_k)`, so the problem has a direction along which nothing changes. **[choice]** The last class is the **reference class**: its `(w, b)` is the zero vector and is not fitted, which removes the direction and leaves `P = (K - 1)(d + 1)` parameters. `l2` penalises the weights and never the biases, as in the binary model (**[read]** `docs/models-linear.md` section 3.2); with the reference class fixed it is no longer needed for identifiability.

The loss is the mean negative log-likelihood plus `l2 / 2` times the sum of squares of the weights, as the binary model's, so that for `K = 2` the multiclass fit with the second class as reference would be the binary fit with the second label as the negative one **[open]**: gate M3 checks whether the two agree as doubles, and the answer is written in section 9 whatever it is. The binary model is not rerouted through this code in any case: the binary files and their path do not change.

### 3.2 The solver (normative; `scripts/oracle_models.py softmax` implements exactly this text)

Newton's method with the full block Hessian, a **fixed number of passes** (`--iterations`, 1 to 1000, default 100, the binary model's flag), no line search, no stopping rule, from `w = 0` **[choice: the binary solver's design, `models-linear.md` 3.2, applied to the block system]**. `Kc = K - 1` modelled classes (class indices `0 .. Kc - 1`, the reference class has index `Kc`), `d` features, `D = d + 1` columns with the bias last (`x~ = (x_1, .., x_d, 1.0)`), `P = Kc * D` parameters, parameter `A = k * D + a` is column `a` of class `k` (class-major). `y_i` is the class index of row `i`, an integer in `0 .. K - 1`, given to the library as a float.

**Row terms.** For row `i` with the current parameters `w`:

1. scores `z_k = b_k`, then `z_k = z_k + x_j * w_{k,j}` for `j` ascending (the binary score order, `models/score.cho`), for `k < Kc`; the reference score is `0.0`;
2. `m = max(0.0, z_0, .., z_{Kc-1})`; `e_k = exp(z_k - m)` for `k < Kc` and `e_ref = exp(0.0 - m)` with `mexp.exp` (so one `e` is exactly `1.0`); `s = ((e_0 + e_1) + ..) + e_{Kc-1}) + e_ref` in that order; `p_k = e_k / s`;
3. `r_k = p_k - t_k` with `t_k = 1.0` when `y_i = k` and `0.0` otherwise (the reference class has `t_k = 0.0` for every modelled `k`);
4. `S[k][j] = p_k * (1.0 - p_k)` for `k = j` and `0.0 - (p_k * p_j)` for `k != j` (the binary `s = p (1 - p)` for `Kc = 1`);
5. loss term `l_i = (m + log(s)) - z_y` with `std.math.log` and `z_y = 0.0` for the reference class;
6. `q_i = (((z_0 * z_0) + (z_1 * z_1)) + ..)` in class order.

**The `W = P + P (P + 1) / 2 + 2` sums of a pass**, in this order: the gradient `G[A] = sum_i fl(r_k * x~_a)`; the Hessian lower triangle by rows, `H[A][B]` for `B <= A` (rows `A` ascending, `B` ascending in a row) `= sum_i fl(fl(S[k][j] * x~_a) * x~_b)` with `A = (k, a)`, `B = (j, b)`; `Q = sum_i q_i`; `L = sum_i l_i`. Every sum is a **decisive sum of `docs/linalg.md` section 2**: blocks of 256 rows, row `i` of a block to lane `i mod 4`, each lane ascending from `+0.0`, the partial of a block `(l0 + l1) + (l2 + l3)`, the partials combined by the level-wise tree of `linalg.order`, and with threads the chunks of 2048 rows of `models.par`; so the sums, and everything computed from them, have the same bits at any thread count.

**The step.** `g = sums / n` and `Hm = sums / n` (lower triangle), `l2 * w` added to the entries of `g` that are weights (not biases) and `l2` to the diagonal entries of `Hm` that are weights; the objective printed in `diag` is `L / n + 0.5 * l2 * sum(w_weight^2)`; Cholesky of `Hm`, two triangular solves, `w = w - delta`; a non-finite value anywhere is `ml.fit.non-finite`. The result for `K = 2` is the binary fit up to the rounding of the probabilities (`p` here is `e / s`, there `1 / (1 + e)` or `e / (1 + e)`): gate M3 prints whether the two agree as doubles.

**Limits, all before any allocation.** `K` is 3 to 64 (`ml.limit.classes`); `D <= 257` (the linear models' 256 features); `P <= 2048`; `ww = W` and `ww * (nchunks + depth + 2 + 4 * min(64, nchunks)) + 2 * P * P <= 2^26` floats (512 MiB), with `nchunks = ceil(n / 2048)` and `depth = common.depth_for(blocks(n))`: the slot of the pass is `ww` floats and the pass holds one per chunk, one per tree level and a scratch of four per thread **[choice, arithmetic of `models/par.cho`]**. The bound does not depend on `--threads` (a refusal that changed with the thread count would break determinism), so it is the worst case of the pass. It is `ml.limit.parameters` (exit 8, `P`, `n`, the floats needed and the budget in `detail`). *Corrected at stage B1: this section first fixed `P_max` = 2048 from the memory of the Hessian alone; the pass holds `ww` floats per chunk, so the bound depends on `n` too, and `D <= 257` is the existing feature ceiling.*

**Refusals**, each with a rule tag, none a trap, the first three decided as the binary model decides them (`docs/refusals-fit.md`):

| tag | exit | when |
|---|---|---|
| `ml.fit.separable` | 8 | `l2 = 0` and (reason 0) the summed loss `L` of a pass is under 0.6931, the binary bound (**[read]** `logreg.cho`: the sum over the rows, so each row's term is under `ln 2` and its true class has probability above 1/2, hence is the argmax; a bound of `ln K` would not give that for `K > 2`), or (reason 2) the root mean square of the scores, `sqrt(Q / (n * Kc))`, is over 50, or (reason 1) a pivot fails after the first pass; the record in `diag` is the binary one |
| `ml.fit.collinear` | 8 | the relative pivot test fails on the first pass (the columns of the design are dependent); the record is `models/rank.cho`'s |
| `ml.linalg.not-positive-definite` | 8 | a pivot fails after the first pass with `l2 > 0` |
| `ml.limit.classes` | 8 | `K > 64` |
| `ml.limit.parameters` | 8 | the bounds above |
| `ml.label.single-class` | 8 | fewer than two classes in the training rows |

The comparison with the binary solver of the cost of the multiclass pass is gate M2.

### 3.3 The file

A new model kind, **9**, `softmax` **[choice]** (a new kind rather than a wider kind 5, because the coefficient tensor changes shape and a reader that knows kind 5 must keep refusing a file it cannot read as `ml.model.version`, never mis-read it). The payload: `coef` f64 shape `[K - 1, d]`, `intercept` f64 shape `[K - 1]`, `labels` i64 shape `[K]` ascending. The reader checks, before allocating, `3 <= K <= K_max`, the labels strictly ascending, `(K - 1)(d + 1) <= 2048`, the tensor lengths against the bytes that are there, every coefficient finite, and the sum order of the section 5 meta as for kinds 4 and above. The file version stays 1: a new kind is not a new format. `describe` gains no member for a binary model; for kind 9 it lists the classes under the existing `classes` member (**[read]** `describe` already has `classes`).

### 3.4 Answers

* **`predict`**: `prediction` is the label of the largest probability; ties go to the **smaller label** (stated; with the same doubles on every platform a tie is the rare case of equal logits). The binary answer is unchanged.
* **`predict --proba`**: for a multiclass model the answer gains a `probabilities` member, a JSON object whose keys are the labels as decimal strings in ascending order and whose values are the probabilities, in the shortest form that reads back to the same double, as every float of the answer; the binary `probability` column is unchanged **[choice]**. The probabilities are computed with the max-subtracted softmax in a fixed order, and a row's probabilities are renormalised by none: their sum is within `K * 2^-52` of 1 and not exactly 1, which gate M5 states and checks.
* **`ml-serve`**: `label` and `probabilities` in the order above; `columns` is `["label","probabilities"]` for a multiclass model (the binary `["label","probability"]` is unchanged) **[choice]**.
* **`eval`**: the metrics of section 4.

## 4. Metrics (stage A)

All are functions of integer counts or of the probabilities, with every sum in the order of the classes then the rows, and each is `null` with its reason when it has no value, as the binary metrics (**[read]** `docs/metrics.md`). `C[a][b]` is the number of rows whose true class is `a` and whose predicted class is `b`.

| metric | definition |
|---|---|
| `confusion` | the matrix `C`, rows the true class, in the label order of the model |
| `accuracy` | `sum_k C[k][k] / n` |
| per class `k`: `precision`, `recall`, `f1`, `support` | `C[k][k] / sum_a C[a][k]`, `C[k][k] / sum_b C[k][b]`, the harmonic mean of the two as the binary F1 (**[read]** `lib/metrics/classify.cho`), `sum_b C[k][b]`; `null` for a class with a zero denominator |
| `macro` of precision, recall, F1 | the mean of the per-class values over the classes that have one **[choice]**; the number of classes left out is reported |
| `micro` | for single-label multiclass, the micro precision, recall and F1 all equal the accuracy, and are reported as that one value, not three **[choice]** |
| `weighted` | `sum_k support_k * v_k / sum_k support_k` over the classes that have a value, the sum in the order of the classes; a class that has a value and no row weighs nothing; no row among the classes with a value is `undefined`; the classes left out are counted **[choice, written at stage A]** |
| `log_loss` | the mean of `-ln(clip(p_true))` with the binary metric's clip to `[eps, 1 - eps]`, `eps = 2^-52` (**[read]** `lib/metrics/logloss.cho`; corrected at stage A: this row first said `max(p_true, eps)` and the code and the oracle use the binary clip on both sides); the sum is the decisive sum of `docs/linalg.md` section 2. The probability table is `n * K` finite values in `[0, 1]`; its rows are neither renormalised nor checked to sum to 1 |
| `roc_auc_ovr` | the mean over the classes of the binary AUC of `class k` against the rest, using the probability of `k` as the score and the binary AUC's rule for ties (midranks, **[read]** `docs/metrics.md` M2); the per-class values are reported; a class absent from the rows, or the only one present, is left out and counted **[choice]** |

**[open]** `roc_auc_ovo` (all pairs) is not built; if the maintainer asks for it, it is a section of its own.

## 5. Trees and forests (stage D)

A node's counts become a histogram of `K` counts; the criterion is the Gini impurity `1 - sum_k (c_k / n)^2` (**[read]** the binary tree's criterion is the Gini impurity, `models-tree.md`), the leaf holds the `K` counts, and the prediction is the class of the largest count (ties to the smaller label). The forest averages the leaves' class frequencies over its trees and takes the largest, with the same tie rule **[choice]**. The payload variant carries `K` and `K` counts per node; the node limit and the memory bound count `K` per node and are checked before allocation. Binary files are unchanged and the binary code path is not the multiclass one run with `K = 2`: the fingerprint golden decides (section 8). The interpretation verbs stay `ml.importance.unsupported-model` for the multiclass tree and forest until F.

## 6. Boosting (stage E)

Each round fits `K` trees, one per class, on the gradient `g_k = p_k - y_k` and the hessian `h_k = p_k (1 - p_k)` of the softmax loss at the current scores, with the Newton leaf `-G / (H + lambda)` times the learning rate, as the binary boost (**[read]** `models-boost.md` section 2). The row sample of a round is one draw shared by the `K` trees of the round **[choice]** (stream kind 9, as today; no new stream). The trees of a round are grown in class order, so the file and the sums are the same at any thread count. The file carries `K` and `rounds * K` trees; the node ceiling of the binary boost applies to `rounds * K * (nodes per tree)`; the early-stopping metric is the multiclass log-loss of the validation rows. **[open]** whether `h_k = p_k (1 - p_k)` or a larger bound is used: the choice is judged by the oracle of gate E1 against a direct finite-difference check of the loss decrease, and the one that is kept is written here with the reason.

## 7. Where the work is

`lib/metrics` (stage A), `lib/models/softmax` (B), `lib/models/tree` and `lib/models/boost` (D, E), `tools/ml` for the label path (`wire`, `wireio`, `verbs`, `mdesc`), `tools/ml-serve` for the answers, and the generated tables (`spec`, schemas, the `describe` golden), which change only where a new member or tag appears.

## 8. Gates (written first; each can fail)

| gate | what | must be able to fail by |
|---|---|---|
| M1 | **binary bytes unchanged**: the fingerprint golden of `docs/bench/xplat_fp.txt` (every fit, 122 lines) is equal after every stage, on every platform, and the `describe` golden of a binary model is equal | any change to a binary path; the check is run with a mutant that routes a binary fit through the multiclass code |
| M2 | **memory and time bounds**: `ml.limit.classes`, `ml.limit.parameters` refuse before any allocation (a fit with `K = 65`, and with `P = 2049`, allocates nothing: the bound is checked in the arguments' order); the time and peak memory of one pass at `K = 64`, `d = 31` (`P = 2016`) with 2,000 rows (the largest `n` the bound allows there is printed by the gate) are measured and reported with the machine, threads and minimum of 5 | a ceiling moved by one |
| M3 | **solver against an independent oracle**: a Python implementation of the text of 3.2 (the same pass count, the same sums in the same order) matches the coefficients as bytes on generated problems with `K` in 3, 4, 7 and 64; the scikit-learn multinomial `LogisticRegression` is compared as a **bound fixed in advance** on the probabilities, and its outcome (including a failure to meet the bound) is written in section 9; a control shows the bound fails for a wrong reference class or a wrong penalty | a wrong penalty on the bias, a different reference class |
| M4 | **metrics against an independent computation** (`scripts/oracle_metrics.py` style): every metric of section 4 on generated confusion matrices, with the empty-class, one-class and tie controls | a swapped macro/weighted, a wrong tie rule |
| M5 | **answers**: the probabilities of a row sum to 1 within `K * 2^-52`; the label is the argmax with the stated tie rule on a constructed tie; `predict`, `ml-serve` and `ml-mcp` give the same bytes for the same rows | an argmax that takes the last tie |
| M6 | **thread invariance**: the model file and the answers are equal at 1, 2, 4 and 8 threads | an unordered sum |
| M7 | **separation and collinearity** refuse with data and a repair, and the repair makes the fit pass; no input reaches a trap (the argument fuzz includes every new flag and verb, in the module the fuzz is sharded into) | a removed refusal |
| M8 | **damaged files**: every truncation and every flipped byte of a kind-9 file, with and without a repaired checksum, is refused or answered, never a trap | a reader that trusts `K` before the length check |
| M9 | **mutants** of the new code: all killed, or each survivor explained | — |
| M10 | **admission** (`scripts/admit.py`) of each new library, and a check that no source file is over 2,000 lines | — |

Gates for stages D and E repeat M1, M5 to M10 for the tree, the forest and the boost, add an oracle for the class histograms and the softmax boosting (E1), and compare with scikit-learn only as a fixed bound with the outcome written down. Nothing is stated against another library's speed.

## 9. Results

Stage A (the metrics of section 4): the gates MC1 to MC5 of `metrics.md` section 1 pass and their output is in `metrics.md` section 5.5, with the 51 mutants of the new code (50 killed, 1 explained) in section 6. The cost of a pass of the solver (gate M2), the comparison of the solver with scikit-learn (M3) and the rest of the stages are empty until they have run. A number is written here only if a script of this repository printed it, with the machine, the threads and the minimum of N.

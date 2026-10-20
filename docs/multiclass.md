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

### 3.2 The solver

Newton's method with the full block Hessian of size `P x P`, a **fixed number of passes** (`--iterations`, 1 to 1000, default 100, the binary model's flag), no line search, no stopping rule, from `w = 0` **[choice: the binary solver's design, `models-linear.md` 3.2, applied to the block system]**. The gradient and the Hessian are sums over the rows in the fixed row order, with the decisive-sum machinery of `linalg.order` the binary solver uses, so that the result is the same at any thread count. The Hessian block `(k, j)` is `sum_i x_i x_i' (delta_kj p_ik - p_ik p_ij) + l2 I`; its solve is the Cholesky of `lib/linalg`. **[open]** whether the block Hessian is positive definite enough to factor without the ridge when classes are well separated is judged by the refusals below, not assumed.

**Refusals**, each with a rule tag and a repair, none a trap:

| tag | exit | when |
|---|---|---|
| `ml.fit.separable` | 8 | extended: the **summed** loss of a pass is under 0.6931, the binary bound (**[read]** `logreg.cho`: the sum over the rows, so each row's term is under `ln 2` and its true class has probability above 1/2, hence is the argmax; a bound of `ln K` would not give that for `K > 2` and is not used), with `l2 = 0`; the scale test of the binary model; repair "set --l2" |
| `ml.fit.collinear` | 8 | as today, on the design columns (the relative pivot test of the first pass) |
| `ml.linalg.not-positive-definite` | 8 | as today, on a pivot of the block Hessian after the first pass |
| `ml.limit.classes` | 8 | `K > K_max` |
| `ml.limit.parameters` | 8 | `(K - 1)(d + 1)` above `P_max`, **before** any allocation |

`P_max` = 2048 **[choice]**; the Hessian is then `2048 x 2048` doubles, `2048^2 * 8 = 33,554,432` bytes, and the Cholesky factor the same again (arithmetic, not a measurement). **[open]** the time of one pass at `P_max` with `n` rows is measured by gate M2 on this machine, with `--threads 1` and `4`, the minimum of 5, before the ceiling is final.

### 3.3 The file

A new model kind, **9**, `softmax` **[choice]** (a new kind rather than a wider kind 5, because the coefficient tensor changes shape and a reader that knows kind 5 must keep refusing a file it cannot read as `ml.model.version`, never mis-read it). The payload: `coef` f64 shape `[K - 1, d]`, `intercept` f64 shape `[K - 1]`, `labels` i64 shape `[K]` ascending. The reader checks, before allocating, `3 <= K <= K_max`, the labels strictly ascending, `(K - 1)(d + 1) <= P_max`, the tensor lengths against the bytes that are there, every coefficient finite, and the sum order of the section 5 meta as for kinds 4 and above. The file version stays 1: a new kind is not a new format. `describe` gains no member for a binary model; for kind 9 it lists the classes under the existing `classes` member (**[read]** `describe` already has `classes`).

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
| M2 | **memory and time bounds**: `ml.limit.classes`, `ml.limit.parameters` refuse before any allocation (a fit with `K = 65`, and with `P = 2049`, allocates nothing: the bound is checked in the arguments' order); the time and memory of one pass at `K = 64`, `d = 31` (`P = 2016`) are measured and reported with the machine, threads and minimum of 5 | a ceiling moved by one |
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

# cancho-ml learning to rank (issue 69, stage M29)

**Status: a specification, written BEFORE any code of this stage.** Every number that a script of this repository has not printed is a **[choice]** or an **[estimate]**; nothing here is a measurement. Claims about other libraries are not made: where a convention differs between libraries it is stated as a choice of this tool, not as a comparison.

Today `ml` predicts one number per row. A search result list, a shortlist or a worklist is judged by its **order inside a query**, not by one number per row. This stage lets a boosted model be fitted on a file with a **group column**, and judges the order with ranking metrics. It is a batch function of the file and the seed, as every verb is; it needs the sample weights, which are built (`weights.md`).

## 1. Scope and order

Each stage is a pull request of its own and ends with its gates passing. The order is a **[choice]**.

| stage | what | why here |
|---|---|---|
| R1 | the **metrics** in `lib/metrics`: NDCG@k, MAP@k, MRR@k, precision@k, recall@k, per group and averaged, with the tie rule and the groups the metric is not defined for; the Python oracle | they define "better" for every later stage, and they are testable with no model |
| R2 | the **group column**: `--group COL` read like `--split-group`, the limits on a group, the refusals, `eval` of a score column against a relevance column | the model needs the groups; this stage needs no model |
| R3 | **pairwise ranking with `--model boost --task ranking`**: the pairwise logistic loss, the file, `fit`, `predict` (and `--rank`), `eval`, `cv` by group, `describe` | the first model that ranks |
| R4 | **LambdaRank**: the pairs weighted by the change in NDCG that swapping them would make; `ml-serve`, `ml-mcp`, JSON lines | the better objective, and the surfaces that already exist |

**Not in this stage:** position-bias correction and counterfactual ranking from click logs (the logged-feedback line of issue 59), neural rankers, a candidate generator (issue 70), early stopping with a ranking task (section 6), the group weights of section 5, a ranking tree or forest (`--model tree` and `forest` answer `args.conflict` for `--task ranking`).

## 2. The data and the group column

A row is `group, features..., relevance`. **Relevance** is the target column: an integer `0 <= r <= R_max`, `R_max = 30` **[choice]**. The bound keeps the exponential gain `2^r - 1` an integer below `2^30`, exact in a double, and no real graded label set is wider. A non-integer or out-of-range relevance is `ml.rank.bad-relevance` (exit 8, the row and the value in `detail`). A binary label is the case `r in {0, 1}`.

`--group COL` names the **group column** (a query id: any text, read as `--split-group` reads it, `splits.md` section 4). It is not a feature and not the target (`args.conflict`, as the weight column); a missing cell is `ml.rank.group-missing` (exit 8, row in `detail`), never a group of its own. The groups are the distinct cells sorted bytewise, numbered `0 .. G-1`; **the rows of a group are in file order**, and they need not be next to each other in the file. This order is the tie order of section 3 and the pair order of section 4, so every result is a function of the file.

Limits **[choice, to be measured at R3]**: a group has at most `G_max = 4096` rows (`ml.limit.group-size`), and the pairs of one round (section 4) at most `2^28` (`ml.limit.pairs`); both are refused **before** anything sized by them is allocated, with the number in the limits table of `introspect`. The number of pairs of a group with `c_r` rows of relevance `r` is `(g^2 - sum c_r^2) / 2`, computed from the counts alone. A group of one row, or whose rows all have one relevance, has no pair: it is allowed, and it contributes nothing to the gradient.

## 3. The metrics (stage R1)

All metrics are computed **per group**, then averaged. For a group of `g` rows, the **ranking** is the rows ordered by predicted score descending, **ties broken by file order** (the earlier row first) **[choice, stated by the issue]**. The tie rule is optimistic for a file whose relevant rows come first and pessimistic otherwise; a metric is therefore a function of the file and not only of the scores, and the answer says "ties: file order". A non-finite score is `ml.input.non-finite`.

`k` is a parameter `1 <= k <= G_max` (flag `--rank-k`, default 10 **[choice]**; `--rank-k 0` is the whole group). A group of fewer than `k` rows uses its `g` rows.

* **DCG@k** `= sum over r = 1..min(k,g) of gain(rel_r) * disc(r)`, summed **sequentially in rank order from +0.0**. `gain(rel) = 2^rel - 1` (default, exact integer as a double) or `rel` with `--gain linear` **[choice]**. `disc(r) = 1 / log2(r + 1)`, computed as `1.0 / (log(float(r + 1)) / log(2.0))` with `std.math.log` (the cancho port of fdlibm, not the platform's libm; the oracle ports the same function, as `oracle_models.py` does), one table for `r = 1 .. G_max`. Each operation is a separate rounding, no fused multiply-add.
* **NDCG@k** `= DCG@k / IDCG@k`, the `IDCG` being the DCG of the same group with its rows ordered by relevance descending (a function of the multiset of relevances alone, so the tie rule does not enter it). One division.
* **MAP@k.** A row is *relevant* iff `rel > 0` (the binary reading of a graded label **[choice]**). `AP@k = (sum over ranks r <= k with a relevant row of precision@r) / min(k, R)`, `R` the number of relevant rows of the group. Precision at `r` is `(relevant rows among the first r) / r`, one division; the sum is sequential in rank order.
* **MRR@k** `= 1 / r*`, `r*` the rank of the first relevant row, or `0.0` when no relevant row is in the first `k`.
* **Precision@k** `= (relevant rows among the first min(k,g)) / k` **[choice]**: a group smaller than `k` cannot reach 1, which is the usual convention and is stated, not hidden. **Recall@k** `= (relevant rows among the first min(k,g)) / R`.

**Groups the metric is not defined for.** A group with no relevant row has `IDCG = 0` (NDCG), `R = 0` (MAP, recall) and no first relevant row (MRR): **it is left out of the mean** **[choice]**, and the answer reports both numbers as data: `groups` (all) and `groups_scored` (the ones in the mean), so the reader sees how many were left out. If no group is scored, every metric is `null` and the fit or `eval` is not refused. (Counting such a group as 0 or as 1 are both used in practice; neither is picked silently.) Precision@k is defined for every group and is averaged over all of them.

**The mean** is the sum of the per-group values in group order (bytewise order of the group cells), sequential from +0.0, divided once by `groups_scored` (or `groups`). It is not a mean weighted by the group's size.

## 4. The pairwise objective (stage R3)

`--model boost --task ranking` fits the boosted trees of `models-boost.md` on the scores `F_i` of the rows, with a different gradient and hessian in each round. The tree growth, the bins, the leaf values, the shrinkage and the file layout are those of `models-boost.md`; **only the rows `g_i`, `h_i` of section 2.4 there change**. `--task` is the place where regression and classification already live, so a ranking task is a third value of it rather than a second flag that could contradict the first **[choice; the issue wrote `--objective rank`]**. The task code in the file is 2.

**Base score** `F0 = 0.0` **[choice]**: a score is compared only inside its group, so a shift changes nothing.

**The loss.** For every ordered pair `(i, j)` of rows of one group with `rel_i > rel_j`, the loss is `log(1 + exp(-(F_i - F_j)))`. With `rho = 1 / (1 + exp(F_i - F_j))` (computed as `models.logreg.proba_of(F_j - F_i)`, the sigmoid the logistic fit uses, so a large `|F_i - F_j|` neither overflows nor loses the result), the pair adds

* `g_i += -w_ij * rho`, `g_j += +w_ij * rho`,
* `h_i += w_ij * rho * (1 - rho)`, `h_j += w_ij * rho * (1 - rho)`.

These are the first and the second derivative of the pair's loss with respect to each score, so a gate can check them by finite differences (section 7). `w_ij` is the weight of the pair (section 5). The rows' sums are accumulated in the **fixed order** `i` ascending in file order within the group, then `j` ascending, the two updates of a pair made in that order; every sum is sequential from +0.0, as in `models-boost.md` 3.1. Groups are independent, so they may be computed on several threads only if each group's sums are made by one thread in that order (the result is then the same at every thread count); R3 starts single-threaded and measures before it does anything else.

**The cost.** The loop is over the pairs, `(g^2 - sum c_r^2) / 2` per group, no more than `2^28` per round. **[estimate]** it is the dominant cost of a ranking fit and R3 measures it before the limit above is fixed.

**LambdaRank (stage R4).** The pair's `w_ij` is multiplied by `|delta NDCG_ij|`, the absolute change of the group's NDCG@`k` when the rows `i` and `j` swap places in the **current ranking** (with the tie rule of section 3), divided by the group's `IDCG`; a group with `IDCG = 0` has no pair. The ranks come from one sort of the group per round (ties by file order). This is a second objective of the same task (`--rank-objective lambdarank`, the pairwise one the default until R4's gates say which is better **[open]**); it changes only `w_ij`, so R3 builds the loop to take it.

## 5. Weights

The row weights of `weights.md` apply. A weight is `w` copies of the row (`weights.md` section 1), so the copies of row `i` pair with the copies of row `j`: **`w_ij = w_i * w_j`** **[choice]**. A row of weight 0 is in no pair that counts. The consequence is stated, not hidden: giving every row of a group the weight `c` multiplies that group's gradient and hessian by `c^2`, not `c`. A weight per group (every row of the group `c`, the pair terms scaled by `c`) is therefore **not built** and is not what `--weight` does; whoever wants it gives the rows `sqrt(c)`. **[open]** the issue asked that the per-row and the per-group forms be fixed together, and this is the per-row one; a group weight can be added later as a second flag without changing a byte of a file fitted without it.

The metrics of section 3 are **not weighted** (held-out rows are scored without weights, as everywhere in this tool).

## 6. Splits, `eval`, `cv`, early stopping

A ranking task must split **by group**: a group split between the training and the test rows leaks the query. So `--group COL` implies the group split on the same column in `fit --split` and in `cv` (`splits.md` sections 2 and 3, `--split-group COL` with `COL` the group column); naming the other strategies with a ranking task (`--stratify`, `--split-by`, or `--split-group` on another column) is **`ml.split.group-required`** (exit 2, repair: "drop --stratify, or name the group column with --group") **[choice]**: the refusal states the leak and gives the way out. `ml cv` reports every ranking metric of section 3 per fold and their mean.

`--early-stopping` with a ranking task is **not built** in these stages **[choice]**: the validation rows of early stopping are a random tenth of the rows (`models-boost.md` 2.7), which splits groups; a tenth of the groups is a design of its own (a new stream and a new rule). It is `ml.split.combination` with a repair that names the flag to drop. (Early stopping is the one reason the stream kinds of `rng.md` may need a new entry; nothing in R1 to R4 draws a new random number, so none is registered.)

## 7. The surfaces

* **`fit`**: `--task ranking --group COL --target REL`. The group column and the target are named, the group column is not a feature and is **not stored in the schema** of the file; the file records its **name** in the provenance section (as the weight column's name is recorded in the run record), so `describe` can say "ranking, grouped by `COL`" and `ml runs verify` replays the fit with the same flags. The format version and the fingerprint rules of the boosting stage apply: a file of task 0 or 1 keeps its bytes.
* **`predict`** needs no group: it answers one score per row. With `--rank --group COL` it adds the rank of each row inside its group (`1` = best, ties by file order), the file being read a second time for the group column. Without `--group`, `--rank` is `args.conflict`.
* **`eval`**: `--group COL` (and `--rank-k`, `--gain`) on a file with the relevance column: the metrics of section 3 and `groups`, `groups_scored`, `ties: "file order"`. The other metrics of a regression (RMSE, ...) are not reported for a ranking model.
* **`ml-serve`** (R4): a request of rows of **one** group answers their scores and ranks; the bytes equal `ml predict --rank` on the same rows.
* **`describe`**: the task, the group column's name, the objective, `R_max`. **`importance`** and **`dump`** are the boosting ones; whether they accept a ranking model is checked at R3 and refused with `ml.importance.unsupported-model` if not.

New rule tags **[choice]**: `ml.rank.bad-relevance`, `ml.rank.group-missing`, `ml.limit.group-size`, `ml.limit.pairs`, `ml.split.group-required`. Each has an exit code, a repair where one is safe, and a fixture in the table of rules; `status.md` counts them when they exist.

## 8. Gates (written first; each can fail)

| id | gate | judged by | a mutant that must die |
|---|---|---|---|
| RK1 | **The metrics against an independent oracle.** A Python implementation from the definitions of section 3 (no cancho code; the `log` of `std/math.cho` ported as for the other oracles) on generated groups: sizes 1 to 200, relevance 0 to 30 and binary, all-equal relevance, ties in the score, groups with no relevant row, `k` below, at and above the group size. Every per-group value and every mean equals the oracle's **bits** | `scripts/oracle_rank.py`, a check driver, a test | the discount `log2(r)` for `log2(r+1)`; the gain `2^rel`; the tie broken by the later row; `IDCG` of the predicted order; the denominator `k` for `min(k, R)`; an empty group counted as 0 |
| RK2 | **The tie rule and the undefined groups are controls**: a file with ties whose NDCG differs between file order and reverse file order, a file where every group is empty (all metrics `null`), a file with one empty group among others (`groups_scored = groups - 1`) | a test | the tie rule ignored; the empty group in the mean |
| RK3 | **Gradient and hessian by finite differences** on small groups (the pair loss of section 4 differentiated numerically, step and bound fixed in the file before the run), including a pair whose scores are equal, far apart, and a group with one relevance | `scripts/gate_rank_grad.py` | the sign of a gradient; the hessian `rho` for `rho (1 - rho)`; a pair counted once |
| RK4 | **The model file against the oracle, byte for byte**: the Python oracle of `models-boost.md` with the ranking gradient writes the file `ml fit --task ranking` writes, on generated files with groups, missing cells and weights; `predict`, `predict --rank` and the `eval` metrics equal its doubles | `oracle_wire_boost.py --ranking`, `test_wire_boost_oracle.py` | the order of the pair sums; the base score; the pair weight `w_i * w_j` |
| RK5 | **Thread invariance**: the same bytes at 1, 2, 4, 8 threads | a test | a sum split across threads |
| RK6 | **Splits by group**: no group in two folds or in both sides; every strategy that would split a group is `ml.split.group-required`; the folds are those of `--split-group` on the same column | tests | the group split dropped |
| RK7 | **Hostile files** (fuzz): one-row groups, groups of one relevance, a group over `G_max`, a pair count over the ceiling (refused before allocation), a missing group cell, a relevance of `-1`, `31`, `1.5`, `NaN`, a group column that is also the target or in `--drop`: every one a rule tag, never a trap, never an allocation sized by the input before its check | `test_fuzz_*` | a check removed |
| RK8 | **Damaged files**: the file reader and the checks of `models-boost.md` 5 apply to task 2 (a task above 2 is refused; the base score of a ranking file is 0.0) | `test_wire_boost_file.py` | the new task code accepted wherever the old were |
| RK9 | **Weights**: unit weights give the bytes of the unweighted fit exactly (`1.0 * 1.0`); integer weights `w_i` equal the file with the rows repeated for the rows' **pairs** (`w_i * w_j` pairs of copies), within a bound fixed in the test | tests | the pair weight `w_i + w_j` |
| RK10 | **Another library's `rank:pairwise`** only if installed here, with the bound fixed in advance in the script; nothing is claimed against any library that was not measured | `scripts/gate_wire_sklearn_*`-style, on request | not applicable |
| RK11 | **Mutants** of the code of every stage: all killed, or each survivor explained | `scripts/mutants_rank.py` | itself |
| RK12 | **`ml-serve` byte equality**, `describe`, the tool schemas of `ml-mcp`, the tables regenerated and checked | existing checks | |

## 9. Open questions

* **[open]** The two limits of section 2 (4,096 rows and `2^28` pairs) are guesses; R3 measures the pair loop on one core (rows per second at several group sizes) and fixes them from the numbers, correcting this document.
* **[open]** Whether the pairwise or the LambdaRank objective is the default after R4 is decided by the gates, not assumed.
* **[open]** A weighted mean of the per-group metrics (by group size, or by a group weight) is not built; the mean is over groups.
* **[open]** `MAP` uses `rel > 0` as relevant; a threshold flag (`--relevant-from R`) is a small addition if a graded file needs it.

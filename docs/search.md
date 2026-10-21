# cancho-ml hyperparameter search (issue 12, M11)

**Status: a specification, written BEFORE any code of this stage. Every number that a script of this repository has not printed is marked [choice], [open] or [read]; section 10 is filled in after the gates have run.** Tags as in `design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from a document or the code, **[open]** not decided or not yet checked. A claim found false is corrected in place and marked.

Today `ml cv` scores one set of hyperparameters on K folds (`docs/splits.md`). Choosing the set is done by hand: run `cv` again with another `--l2` or `--max-depth`, read the numbers, compare. This stage adds the loop, as a verb, with the properties the rest of the tool has: the same file, seed and flags give the same bytes at any thread count; every size is bounded and refused before the work; every refusal has a rule tag; the answer is data that an agent can act on.

What this is **not**: fine-tuning of a pretrained model (the project has none, `design.md` epic), a warm start of a fitted model, or a search over the *structure* of a plan (which steps, in which order). It searches the hyperparameters of one model, with the plan's steps as given.

## 1. The surface **[choice]**

One new verb, `ml search`, in the flag table like every other verb (parser, `introspect`, `skill`, the schemas and `ml-mcp` come from the table). It **writes nothing** (as `ml cv`: it fits each fold in memory), so the `ml` authority gains nothing and `ml-predict` and `ml-serve` are unaffected. A search is followed by `ml fit` with the flags it prints; a verb that both searched and wrote a model would be two verbs (the argument of `splits.md` section 1 for `cv`).

`ml search` takes everything `ml cv` takes (`--model`, `--task`, `--target`, `--step`, `--drop`, `--float`, `--cat`, `--na-values`, `--seed`, `--threads`, `--stratify`, `--split-by`, `--split-group`, `--folds`, `--repeats`, `--weight`, the reader limits, `--input-format`) except `--plan` (see below), and these:

| flag | kind | default | meaning |
|---|---|---|---|
| `--space TEXT` | text, required | none | the hyperparameters to vary and their values (section 2) |
| `--method grid\|random` | enum | `grid` | every combination, or `--n-iter` seeded draws |
| `--n-iter N` | nat, 1 to 200 | 20 | draws of a random search; refused with `--method grid` |
| `--metric NAME` | text | `rmse` for a regression, `log_loss` for a classification | the metric that ranks the candidates (section 4) |
| `--max-fits N` | nat, 1 to 2000 | 200 | the ceiling on `candidates * folds * repeats`, checked before any fit |
| `--emit-folds` | bool | off | as for `cv`: the assignment of the rows, once, in the answer |

* **`--plan` is `args.conflict`.** A plan text describes the whole fit including the model line with its hyperparameters, and "no other flag may also describe it" is the rule `fit` and `cv` already apply (`wire.check_flags`); a space inside a plan would be a second grammar. Steps are given with `--step`, as in `cv`.
* **A name in `--space` that is also given as its own flag is `args.conflict`** (`--l2 0.5 --space l2=0.1,1`): one place says what a hyperparameter is. A hyperparameter given as its own flag and not in the space is **fixed** for every candidate.
* **`--space` with a model that has no hyperparameter by that name** is `ml.search.bad-space` (section 2), as a flag the model does not take is `args.conflict` in `fit`.
* The verbs that do not search (`fit`, `cv`, `predict`, `eval`, ...) do not list the new flags: `args.conflict` as for any flag a verb does not take (`tests/conformance/test_tables.py` checks that each flag is listed by some verb).

## 2. The space **[choice]**

`--space` is one text of entries `NAME=VALUES` separated by `;` **[choice]** (a flag may be given once: `args.duplicate-flag`). It is at most 4096 bytes and 16 entries (`ml.search.bad-space`, reason `too-long`).

`NAME` is a hyperparameter of the model (`introspect` lists them per model; the definitions are the `[[hyperparameter]]` tables of `lib/models/*/model.toml`, **[read]** type `int`, `float` or `enum`, `min`, `max`, `nonneg`, `default`). `VALUES` is one of:

* a **list** `v1,v2,...` (at least one, at most 64 values, no value twice): each value is validated as the flag would be (`est.hparam_ok`): an int in `[min, max]`, a float that is finite and, for `nonneg`, not negative, an enum member. Grid and random both take a list (random picks one).
* a **range**, for `--method random` only: `lo..hi` (uniform), `log:lo..hi` (log-uniform, `lo > 0`) for a float hyperparameter, `int:lo..hi` for an int one, `lo <= hi`, both inside the hyperparameter's own bounds. A range with `--method grid` is `ml.search.bad-space` (reason `range-in-grid`): a grid has no step here, and inventing one would be a rule of its own.

**The canonical text** of a space is the entries sorted by name (bytewise), each value written as the shortest text that reads back to the same number (`flt.put_float` for floats, decimal for ints, the enum member as given), joined by `;`. Two spellings of one space (`l2=0.1,1` and `l2=1,0.10`) have one canonical text *only after sorting the values of a list*; **the order of the values of a list is kept** **[choice]** because it is the order of the candidates (ties go to the first) and the user may mean it. The canonical text is in the answer.

## 3. The candidates **[choice]**

* **Grid.** The candidates are the combinations in the lexicographic order of the sorted names, the first name slowest and the last fastest; within a name the order of its list. The count is the product of the list lengths, computed with a guard against overflow (any product above `--max-fits` is already refused).
* **Random.** Candidate `j` (0-based) takes, for the hyperparameter at sorted position `p`, one draw of the stream `derive(derive(derive(seed, 12), j), p)` **[choice]** (stream kind 12, to be registered in `docs/rng.md` before use: "hyperparameter draw of a random search; index: the draw, then the position of the name"). A list picks the value at `floor(u * n)`; a range maps `u` in `[0, 1)` to `lo + u * (hi - lo)` (uniform), `exp(log lo + u * (log hi - log lo))` (log), or `lo + floor(u * (hi - lo + 1))` (int). `u` is the library's `unit(output(...))` of `rng.md` section 2, so numpy can reproduce it. A floating-point result is rounded to the shortest text that reads back and the *text* is what the fit receives, so a candidate printed in the answer is exactly a candidate that can be given to `ml fit`. The draws can repeat a candidate; **duplicates are removed, the first occurrence is kept**, and the answer reports `drawn` (`--n-iter`) and `candidates` (distinct). The cost bound uses `candidates`.
* **Fixed hyperparameters** (own flags) are copied into every candidate's plan.
* **Every candidate is checked as `ml fit` would check it** before the first fit: hyperparameters out of range are impossible by section 2, but a *combination* may be refused by the model (`ml.param.out-of-range` for iterations times features, the tree node ceiling, `--early-stopping` with a group or time split). Such a candidate is not skipped silently: the search stops with that refusal, naming the candidate index and its parameters in `detail` **[choice]** (a grid in which some cells silently vanish is a result nobody can reproduce by reading the space).

## 4. The score, the ranking and the selection **[choice]**

* **The folds are made once** (by `ml.splitwire.cv_assign`, the function `cv` uses, from `--seed`, `--folds`, `--repeats` and the strategy) and **every candidate is scored on the same folds**. A search that drew new folds per candidate would rank noise (gate S4 has a mutant for it).
* **A candidate's score** is the metric named by `--metric`, averaged over the `M = folds * repeats` fold results exactly as `cv` aggregates them (`splits.md` section 6: the package's one sum order, `mean = sum / M`, sample standard deviation with divisor `M - 1`). The answer carries `mean`, `std`, `n` and the per-fold values. The per-fold metrics are those `ml eval` prints for the held-out rows (same code).
* **Metrics and direction** **[choice]**: lower is better for `rmse`, `mae`, `log_loss`, `mse`; higher is better for `r2`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc` and the `_macro` and `_weighted` variants of the multiclass stage. A name that is not a scalar metric of the task of the model is `ml.search.bad-metric` (exit 2; `detail`: the names the task has). The default is a proper scoring rule defined for every model of its task: `rmse` and `log_loss`.
* **Ranking.** Candidates are ordered by `mean` in the direction of the metric; **ties** (equal strings after the formatting of section 6, i.e. equal doubles) keep candidate order, so the result is a function of the space. A candidate whose metric is `null` in any fold (undefined: a fold with one class has no AUC) has score `null`, is ranked after every candidate with a score, in candidate order, and is reported with `reason`. If **no** candidate has a score: `ml.search.no-score` (exit 8, `detail`: the metric and the count).
* **The best** is rank 1. The answer prints it as the **flags** that reproduce it (`"--l2 0.1 --max-depth 5"`) and as the **model line** of the plan, so an agent or a person can run `ml fit` with it. Selection rules other than the best mean (the one-standard-error rule, for the simplest model within one `std` of the best) are **[open]**: the data to apply them (`mean`, `std`, `n` of every candidate) is in the answer, and a rule is a later flag with its own gate.
* **The estimate is optimistic and the answer says so** **[choice]**: the same folds chose the winner and scored it, so `best.score` is biased upward as a prediction of the winner's error on new data. `estimate` in the answer is `"selection"` and the documentation of the flag says what that means; stage S2 (section 8) adds the honest number.

## 5. Leakage and what is fitted on what **[choice]**

Every fold of every candidate is fitted by `wire.train` on the fold's training rows, the function `fit` and `cv` use: the steps of the plan are fitted per fold, never on all rows, and the held-out rows go through the stored steps (`splits.md` section 5). The search adds exactly one new way to leak, and it is the selection itself (section 4); it adds no path from a held-out row to a fit. With `--weight`, a candidate's fold fits use the weights of the fold's training rows (`weights.md` section 2.4), and the held-out rows are scored without weights.

## 6. The answer **[choice]**

```
{"model":..,"task":..,"target":..,"rows":N,"columns":..,
 "method":"grid"|"random","space":"<canonical text>","drawn":N,"candidates":N,"fits":N,
 "metric":"rmse","direction":"min"|"max","split":"cv folds=5 repeats=1 seed=0","strategy":"kfold","estimate":"selection",
 "results":[{"rank":1,"index":0,"params":{"l2":"0.1","max-depth":"5"},"score":{"mean":"..","std":"..","n":M},"folds":["..",..]}, ...],
 "best":{"index":0,"params":{..},"flags":"--l2 0.1 --max-depth 5","plan":"model kind=... \n","score":{..}},
 "assignment":{...}}                                                             (with --emit-folds)
```

* `results` lists every candidate, best first, by the ranking of section 4; `params` has every searched hyperparameter as the text a flag takes; a candidate with a `null` score has `"score":null`, `"folds":[..]` with `null` where the metric was undefined, and `"reason"`.
* All numbers are strings written by the one float writer (`verbs.put_float_text`), as everywhere in the tool; `null` for an undefined metric. The bytes of the answer are a function of the file, the flags and the seed: no clock, no host, no path beyond `--root`-relative names, no payload (no row, no cell).
* `--format text` prints one line per candidate (rank, score, params) and the best flags.
* The answer is bounded: at most `2000` candidates (the cap of `--max-fits`) times at most `20 * 10` fold values each, i.e. 400,000 numbers in the worst case, and the writer's own ceiling (`ml.limit.output-too-large` **[open]**: if the existing output cap of `cv` is smaller than this worst case, the search uses it and refuses with the tag `cv` uses; checked at the gate S6).

## 7. Refusals **[choice]**

Tags are registered in `tools/ml/spec.tmpl`, `docs/design.md` sections 8 and 8.1, with a fixture each. New tags:

| tag | exit | when | repair |
|---|---|---|---|
| `ml.search.bad-space` | 2 | the text of `--space` is malformed, names a hyperparameter the model lacks, has a value out of the hyperparameter's range, a range with `--method grid`, a list with a repeated value, or is over 4096 bytes or 16 entries (`detail`: `position`, `name`, `reason`) | none (the reason says what to change) |
| `ml.search.too-many-fits` | 2 | `candidates * folds * repeats` is above `--max-fits` (`detail`: the three factors and the product, `max`) | `retry` with `--max-fits` = the product when it is within the ceiling 2000; `choose` between fewer values and fewer folds otherwise |
| `ml.search.bad-metric` | 2 | `--metric` is not a scalar metric of the task | `choose` among the metrics of the task |
| `ml.search.no-score` | 8 | no candidate has a defined score for the metric | none |

Existing tags reused: `args.conflict` (`--plan`, a hyperparameter both as a flag and in the space, `--n-iter` with `--method grid`, a model with no hyperparameters, `--model constant` as in `cv`), `args.required-flag` (`--space`), `ml.param.out-of-range` and the other refusals of `fit` for a candidate (section 3), `ml.split.*` for the folds, `ml.weight.*` for the weights. A search over a model with **no** hyperparameter is `args.conflict`: there is nothing to search.

## 8. Stages **[choice]**

| stage | content | gates |
|---|---|---|
| S1 | `ml search` for grid and random over `--folds`, the answer of section 6, refusals, `--weight`, `--format text`, `ml-mcp` tool (derived from the table) | S1 to S8 below |
| S2 | `--split train=F` (and its strategies): the search runs on the training side, the winner is refitted on the whole training side and scored on the held-out rows, the answer gains `holdout` (`estimate` becomes `"holdout"`). This is the unbiased number; it needs the plan-per-candidate refit and the holdout code of `fit` | S9 |
| S3 | `--track DIR`: a record `ml.search.v1` (the canonical inputs, the canonical space, the candidate list, the scores, the winner), `ml runs verify` replays a search and compares the bytes (the second record schema `tracking.md` section 5.7 deferred for `cv`) | S10 |
| later | successive halving (a budgeted search that drops the worse half at each rung), the one-standard-error selection, candidates in parallel; each is its own issue with its own gate | |

## 9. Determinism, bounds, and what is not claimed

* The same file, flags and seed give the same answer, byte for byte, at any thread count and on any machine: the folds come from `--seed` before any fit; the candidates come from the grid order or the stream of section 3; the candidates are fitted one after the other, each fit threaded as `cv` threads it (`--threads`); the aggregation is the package's one sum order. No clock, no environment, no iteration order that is not sorted.
* Every size is compared with its cap before the first fit: the space text and entries, the values per entry, the candidates, the fits, the answer. No input reaches a trap (gate S7).
* **Cost.** A search is `candidates * folds * repeats` fits; the time of one is that of the model on `(K - 1) / K` of the rows. The cap on fits bounds the *count*, not the time: a boosting fit of 10,000 rounds on 10 million rows is as slow inside a search as outside it. The number printed in the answer (`fits`) is what lets an agent estimate; **no time estimate is made by the tool and none is claimed here** (the measured cost of one fit per model is in `benchmarks.html` and `status.md` for the machine that measured it).
* **Not claimed:** that the winner is the best model for new data (section 4: the selection is optimistic); equality of the *winner* with scikit-learn's `GridSearchCV` (its folds are another assignment: what is compared is the scores on the same explicit folds and the argmax where the gap between the best two is larger than a margin fixed in advance, gate S5); anything about speed against another library.

## 10. The gates (written before the code; each can fail) **[choice]**

| gate | what | must be able to fail by |
|---|---|---|
| S1 | **a candidate is a cv.** For each of 12 generated spaces (grid and random, linear, logistic, tree, forest, boost, classification and regression, with `--weight` and with a group split in some), each candidate's `score`, `folds` and `params` equal, as strings, the `ml cv` answer of that candidate given as its own flags (`summary.metrics` and `results[].metrics`) | folds drawn per candidate; a candidate's parameter not applied; a different aggregation |
| S2 | **the candidates.** A Python port of the grid order and of the random draws (stream kind 12, the `unit` of `rng.md`) gives the same candidate list as the answer, as bytes of the `params` text, for 40 seeds, lists, ranges, log ranges and int ranges; duplicates removed in the stated order | a stream index off by one; the first name fastest; a range end excluded |
| S3 | **the ranking.** A Python ranking of the scores of S1 with the direction of each metric and the tie rule equals `results[].rank` and `best`; controls: a metric of each direction, ties (a space whose candidates are equal, e.g. a hyperparameter the data cannot see), and a `null` score ranked last | direction swapped; ties to the last; a null ranked first |
| S4 | **same folds, no leakage.** The assignment is made once (`--emit-folds` equals the one of `cv` with the same flags); the label of a row is changed to garbage: the fits of every fold in which that row is held out are unchanged (a row is held out in exactly one fold per repeat), checked by the equality of the model of a fold with `ml fit` on the fold's training rows (`cv` gate C1, extended to search) | folds per candidate; steps fitted on all rows |
| S5 | **against scikit-learn.** `GridSearchCV` with `PredefinedSplit` on the explicit folds of `--emit-folds`, `Ridge(alpha)` and `LogisticRegression(C)` on the same spaces: every mean score within a bound fixed in the script (relative 1e-6 for the ridge, 1e-4 for the logistic with `tol` 1e-12), and the same winner when the gap between the best two exceeds 1e-3 relative; a control with the unweighted fit on a weighted search must disagree | a score of another candidate; an inverted direction |
| S6 | **bounds and refusals.** A fixture for each tag of section 7 and each `args.conflict` of section 1; the product above `--max-fits` refused before any fit (a mutant that fits first is caught by a fit that would trap or take long); the answer of the largest allowed search is written or refused with the tag `cv` uses | a cap moved by one; refused after the work |
| S7 | **fuzz.** Hostile spaces (empty, `=`, `;;`, 17 entries, 65 values, `nan`, `inf`, `-0`, `1e999`, `log:0..1`, `int:5..1`, a name twice, non-UTF-8, a NUL) answer a refusal with a tag and never trap, 500 rounds at `FUZZ_N` | a trap |
| S8 | **thread invariance and mutants.** The answer is the same bytes at 1, 2, 4 and 8 threads for every model; mutants of the new code (the order of the grid, the stream kind, the dedupe, the ranking, the tie rule, the aggregation, the cap, the direction table, the candidate's plan) are all killed or each survivor explained; `admit.py`-style admission of the verb; no source file over 2,000 lines | an order that depends on the thread |
| S9 | **the holdout (stage S2).** The winner refitted on the training side and scored on the held-out rows equals `ml fit --split ... <best flags>` followed by its own held-out metrics, as bytes; the held-out labels do not influence the candidate scores (set to garbage: same `results`) | held-out rows in a fold |
| S10 | **the record (stage S3).** `ml runs verify` of a search record gives the same bytes; a record edited by one byte is `ml.runs.damaged` | a record that loses the space or the seed |

## 11. Results

Empty until the gates have run. A number is written here only if a script of this repository printed it, with the machine, the threads and the minimum of N.

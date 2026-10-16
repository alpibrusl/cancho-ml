# cancho-ml refusals at fit time: collinear columns and separable data (issue 48)

**Status: sections 1 to 5 (the defects, the decisions, the gates) were written before any code of this change. Section 6 holds the results; a number
is in it only if a script of this repository printed it, with its conditions.** Tags as in `design.md`: **[choice]** a decision the maintainer can
overturn, **[read]** read from the code or a document, **[measured]** printed by a script here, **[open]** not decided.
Scope is the text of alpibrusl/cancho-ml#48. Issue 37 (multiclass logistic) needs the same two checks per class block: section 5 says what it can reuse.

## 1. The defects, reproduced on `main` (6b4f926) with the built `ml`

Data of the fixtures `tests/conformance/fixtures.py` `_refuse_cases()` (all written before the fix; the ones marked FAILS today fail on `main`).

**(a) Separable data, `--l2 0` (the default).** Six rows, `x1 = -3 -2 -1 | 1 2 3`, `x2` mixed, the class is the sign of `x1`.
`ml fit --model logistic --l2 0 --iterations 25 --target y sep.csv` **succeeds** and writes `x1 = 15.162324848990629`, `x2 = -5.054108282996878`,
intercept `8.6e-10`: every training row right, a norm that grows by about 0.63 per iteration without limit (the optimum is at infinity), and no word about it.
With the default 100 iterations the same command fails *late* with `ml.linalg.not-positive-definite` and `"detail":{}`: the weights `p(1-p)` of the Newton
Hessian have vanished and the relative pivot test fails at some pass nobody is told. Which of the two a user sees depends on the number of iterations.

**(b) Collinear design, `--l2 0`.** Four designs, each fails with `ml.linalg.not-positive-definite`, `"detail":{}`, `"repair":null`, after the sums were done, and the
message says "dependent or nearly dependent columns" without saying which:
two equal columns `a, b`; a constant column `k` beside the intercept; the full one-hot block of a 3-level text column beside the intercept (linear and logistic);
and a ridge too small to matter (`--l2 1e-300` on equal columns). The README's categorical flow carries `--l2 0.1` for this reason.

## 2. Decisions

### D1. The rank test **[choice, measured in 6.1]**

**The test is the relative pivot test the Cholesky factorisation already runs** (`docs/models-linear.md` 3.1: column `j`'s pivot `s_jj` is accepted only if
`s_jj > 2^-40 * a_jj`, `a_jj` the diagonal element of the matrix being factored). It is a deterministic test on the pivots of the Gram matrix already computed: for
linear regression the matrix `[X | 1]' [X | 1] + l2` on the feature diagonal; for logistic regression the Newton Hessian of the pass (at pass 0 the weights are all
`1/4`, so the Hessian of pass 0 is the Gram matrix over `4n`, plus `l2`). It costs nothing on a fit that succeeds, and **no pass over the data is added**.

* **The tolerance is relative to each column's own diagonal** (`s_jj / a_jj` is `1 - R_j^2`, `R_j` the multiple correlation of column `j` with the ones before it):
  so it does not depend on the scale of a column, and it does not depend on `n`. Its dependence on the size `p` of the matrix is the rounding noise of the
  elimination, which grows with `p`: section 6.1 measures that noise at `p` up to 257 (the largest the estimators take) and `n` up to 20,000 and compares it with `2^-40`.
  **This change does not change the number `2^-40`**: changing it would change which fits succeed.
* **The decision is still made by the factorisation** (one decision, not two that could disagree). What is new is what happens when it says no: the failing
  matrix goes to a diagnosis that is pure data (`lib/models/rank.cho`): the index `j` of the first pivot at or below the tolerance, the relative pivot
  `s_jj / a_jj`, and the **dependency**: the least-squares coefficients `c` of column `j` on the columns before it, `c = G11^-1 g_j` (forward and backward
  solves with the factor of the leading block, which is accepted by definition of `j`). It reads "column `j` is, to within the tolerance, `sum_k c_k * column k`":
  the columns with a coefficient that matters are named, with the coefficients. For the full one-hot block beside the intercept this is `(intercept) = 1 * sp#0 + 1 * sp#1 + 1 * sp#2`.
* The intercept is the **last** column of the matrix (index `d`), so a dependency that involves it is reported at the intercept, and a pair of equal columns at the second of them.
* `ml.fit.collinear` replaces `ml.linalg.not-positive-definite` for every failure of the pivot test in linear regression and at pass 0 of logistic regression, **whatever
  `l2` is** (with `l2` positive but below `2^-40 * a_jj` the matrix is numerically singular too: the detail carries `l2`, and the repair is a larger one).
  A fit that succeeds today succeeds unchanged: the diagnosis runs only on a failure, the arithmetic of a success is not touched.
* **Where `ml.linalg.not-positive-definite` remains**: a pivot failure at a pass after the first of logistic regression with `l2` positive (the ridge makes the fit well
  posed in theory, the weights have vanished in practice: a ridge below `2^-40` of the diagonal), and any failure whose diagnosis finds no failing pivot (NaN). It is still a
  tag with a fixture.

### D2. Separation in logistic regression **[choice, measured in 6.2]**

`l2 = 0` only. With `l2` positive the objective is strictly convex and has a minimiser: **never refused for separation** (a fixture says so).

**The criterion is the summed loss of the pass.** The Newton pass already computes `ell = sum_i loss_i` at the current `w` (the last decisive sum, slot `W-1`). If
`ell < 0.6931` (`ln 2 = 0.693147...`; the bound is stated rounded down by 5e-5) then **every** row has `loss_i < ln 2`, that is, a positive margin: `w` classifies every
training row correctly. A `w` that does so makes the data separable (and the maximum-likelihood estimate nonexistent: the loss falls along `c * w` as `c` grows). So:

* **Sound, no false refusal for any data.** If the data are not separable, no `w` has every margin positive, some term is at least `ln 2` in exact arithmetic, and the
  computed `ell` (an ordered sum of terms with a relative error around `1e-15`) is far above the bound. The only way to be wrong is a rounding error of 7e-5 relative in `ell`.
* **Free**: no pass, no storage: one comparison on a number already there. Not "exact zero loss" (never reached) and not "coefficient norm past a bound" (the bound would
  be a function of the data's scale, which is what separation makes meaningless).
* **A second test, for separation with ties (quasi-complete separation), added after the first measurement of the oracle [choice, measured in 6.2].** A set where some rows of both classes
lie on the separating plane (`x = -2 -1 0 | 0 1 2`, the classes meeting at 0) has no maximum-likelihood estimate either, but the tied rows keep a loss of `ln 2` each, so the loss
test never fires, and Newton's coefficient grows by about one per pass for ever (99.8 after 100 passes). The test: the **predictor scale** `sqrt(w' G w / n)`, the root mean square of
the linear predictor `z` over the training rows, with `G` the Gram matrix of the columns with the bias. It needs no pass: the Hessian of pass 0 (slot 0 of the decisive sums) is `G / (4n)`
because every weight is 1/4 at `w = 0`. It does not depend on the units of the features. A maximum-likelihood fit saturates only as far as the data force it (a probability of 1e-9 is
`z = -20.7`); a root mean square of **50** is a probability of 2e-22 on average: the fit is running away along a direction that separates. `reason` is `scale`. The bound is a choice.
**[estimate, not measured]** A fit that has a maximum-likelihood estimate and a root mean square of 50 would need predictions that the data contradict by orders of magnitude; the oracle of 6.2 finds no such set among its 138 overlapping ones, and nothing larger was tried. The test does not look at single rows, so a runaway confined to one row (a column that is zero everywhere else) is seen by the loss test only.

* **What it misses, stated**: separable data that the fixed passes have not yet driven under the bound (`iterations` too small: with 1 or 2 passes the loss test never fires; the scale
  test needs about 50 passes at a scale growing by one per pass), and separation in a fit with `l2 > 0` (by decision not refused).
* **The late form** (`reason` `pivot`): a pivot failure at pass `k >= 1` with `l2 = 0` (the Hessian weights have vanished). The Hessian of pass 0 passed the
  test, so the columns are not collinear: what remains that kills the pivots is a weight collapse, which is the numerical face of separation. After the two tests above this fires for
  separation with ties where the weights vanish on every row.
* The detail: `reason` (`loss`, `scale` or `pivot`), `pass` (0-based, the pass at the start of which it was seen), `loss_sum` and `loss_bound`, `coefficient_norm`
  (the Euclidean norm of `w`, bias included, at that pass), `predictor_scale` and `scale_bound`, `l2`, `iterations`. The tolerances are data in the refusal, as the issue asks.
* Repair: `--l2 X` (see D5), `X` computed so that the repaired fit is well posed at every pass.

### D3. Which models, flags and verbs

Linear regression and logistic regression, both `--model` and `--plan` forms, **after the steps ran**: the check sees the encoded matrix and names its columns (`sp#0`, `age`,
`(intercept)`). Fits with `--split` check the training rows only (the matrix that is fitted). `tree`, `forest`, `kmeans` and `constant` have no normal equations and no Newton pass:
**unaffected** (a test says a duplicate column fits a tree). `predict`, `eval` and `ml-serve` read a stored model and are unaffected (the rule is fit-time only).
Multiclass (#37) will reuse `lib/models/rank.cho` per class block, and the loss test per block.

### D4. No existing answer changes **[checked in 6.4]**

* A fit that succeeds today produces the same bytes: the checks add no arithmetic to a success path **except one comparison per Newton pass for `l2 = 0`**, which refuses a fit
  whose training rows are all classified correctly with summed loss under `ln 2`. That fit did "succeed" before, with coefficients that grow with the iteration count: it is
  the behaviour the issue calls a defect. Section 6.4 lists every fixture, golden and gate data set with `l2 = 0` that is separable, by running them.
* `ml-linalg-not-positive-definite` (two equal columns, `l2 = 0`) now answers `ml.fit.collinear`: its stdout changes by design and the record `golden_numeric.json` holds the new answer (one commit re-records it with the two planned cases of section 6.4).
  The tag `ml.linalg.not-positive-definite` keeps a fixture of its own (D1).
* The unit tests of `lib/` that expect the code 9 for a collinear Gram keep passing if the library returns the new code only where the estimators call it; section 6.4 lists the ones that change.

### D5. Registration and repair **[read for the format, choice for the values]**

* Codes in `lib/core/refuse.cho`: **22 `ml.fit.collinear`, 23 `ml.fit.separable`**, exit 8, each with a repair text (the table of `tag()`/`repair()`); `count()` 23. Also: `design.md` 8.1 (both tables), `docs/integration.md`, `spec.tmpl` (`extra()` and `rules()`, hence `spec.cho`), the schemas (`scripts/schemas.py`), the manifests (`manifests/`, `scripts/manifest.py`), `lib/models/linear/model.toml` and `lib/models/logistic/model.toml` refusal lists and `docs/models.md`, `docs/cli.md`, `docs/status.md`, `README.md`, `docs/index.html`, `evidence.html`. `ml-serve` is unaffected.
* **The repair is an argv, in the repo's format** (`fail.retry_open`, `fail.choose_open`): a `retry` when one repair exists, a `choose` with the options when there are several.
  * `ml.fit.separable`: `--l2 X`. If the command line has `--l2 V` its value is *replaced*; otherwise `--l2 X` is appended. With `--plan` the hyperparameters are in the plan's model line
    and no flag may also give them (`args.conflict`): the repair is `none` with a reason that names the key (`l2=` on the model line).
  * `ml.fit.collinear`: `--l2 X` (same rules) and, when it is safe, `--drop NAME`: `NAME` is an input column the dependency involves (the column `j` itself when it is one; a column of a
    one-hot block resolves to its source column), the command line has no `--step`, `--plan` or `--cat` (dropping a column that a step names is `ml.plan.unknown-column`), and the target is not `NAME`. When `--drop`
    was already given, `NAME` is added to its list. `drop-first` on the encoder: **it does not exist** (`grep drop-first` finds nothing), so it is not offered.
  * `X` **[choice]**: for a collinear matrix the smallest power of ten that is at least `2^-30 * max_i a_ii` of the matrix as factored (the relative pivot of the ridge fit is then at least `2^-30`,
    a thousand times the tolerance: the ridge of the fitted dependent direction is `X`, its pivot at least `X`); for a separable fit the smallest power of ten at least `2^-10 * max_i H_ii` of the
    pass-0 Hessian (every later Hessian has diagonal at most that plus `X`, and pivot at least `X`, so the relative pivot is at least about `2^-10`). Both are one significant digit, written as a decimal the flag parser reads.
    The repaired fit is therefore well posed at every pass; gate A runs every repair.

### D6. With splits, cross-validation, run tracking and serving **[choice, tested in `test_refuse_fit.py`]**

* **`ml cv`.** Every training fold is fitted by the function `fit` uses, so a fold can be refused: the training rows of one fold can be separable or have a dependent column when the whole data has neither.
  The refusal is `ml.fit.collinear` or `ml.fit.separable` as for `fit`, and its `detail` also holds `repeat` and `fold` (0-based, as the entries of `results` of a cross-validation have them),
  the way `ml.label.single-class` names them. The repair is the argv of the `cv` invocation with `--l2 X` set (or the `--drop` option), and it runs: the first fold to be refused decides, and
  `X` is computed from that fold's matrix, so a second fold with a larger diagonal could be refused again; the repaired command is then refused with its own `X`. A `fit` and a `fit --split` carry no `repeat` or `fold`
  (a holdout is one fit of the training rows). Two fixtures: `ml-cv-fold-separable-names-the-fold` (fold 1 of 3 on twelve rows; the whole file is fitted with `l2` 0) and `ml-cv-fold-collinear-names-the-fold`
  (a duplicate column in the first block of a time split; the whole file is fitted).
* **`fit --track`.** A refused fit writes no run record and no model file: the check runs before the record is planned, so the `--track` directory is byte for byte what it was (a test lists it before and after),
  and the record of an earlier fit still verifies (`ml runs verify`). The repair keeps `--track` and writes the record of the repaired fit.
* **`ml-predict` and `ml-serve`.** Fit-time only. The three binaries are built from the same sources (`scripts/check_bins.py` checks the lists), so the check is in their code, but their verbs read a stored model and never
  call a fit. Their manifests (`manifests/ml-predict.authority.json`, `manifests/ml-serve.authority.json`) and their authority rows are those of `main`: no capability is new.

* **The `--l2` repair under `--split`, `--stratify` and `--cat`.** The check sees the training rows of the split and the encoded matrix, and the repair keeps every other flag, so the repaired command fits under
  a holdout, a stratified holdout, a time holdout and a one-hot encoding (five cases of `test_the_l2_repair_gives_a_fit_under_split_stratify_and_cat`).

## 3. Gates (before the code; each can fail)

| id | gate | judged by | what makes it fail |
|---|---|---|---|
| A | The fixtures of section 4: each defect gives the tag, with the detail and a repair; **every argv of every repair is run in the same sandbox and gives a model** | `tests/conformance/test_rules.py` (`test_every_fixture`, `test_repairs_of_fit_refusals_run`) | on `main`: 11 of the 17 cases fail (section 4) |
| B | **Oracle for the rank decision.** `scripts/oracle_refuse.py` builds `[X \| 1]` for generated families (duplicate columns, full one-hot, constants, scaled copies, sums of columns, near-collinear at `1 - rho` from 1e-3 to 1e-16 and scales 1e-8 to 1e8, well conditioned data of many shapes and sizes), takes the Gram matrix in **exact fractions** (`fractions.Fraction`), eliminates it exactly, and calls `ml fit` on the same CSV: `ml` must refuse exactly when the exact relative pivot is at most `2^-40` (a band `[2^-41, 2^-39]` is reported separately, not counted), must name as `column` the exact first failing pivot's column, and the named `columns`/`coefficients` must be the exact least-squares dependency to 1e-6 | `python3 scripts/oracle_refuse.py rank` | a mutant of the tolerance, the pivot test, the column naming; counts of both verdicts are printed (a family that never refuses or never accepts is itself a failure) |
| C | **Separation.** `scripts/oracle_refuse.py sep`: an exact separability test (integer data, rational arithmetic: the perceptron / LP feasibility of `y_i (w . x_i + b) >= 1` solved by exact simplex on small instances, and by a cutting argument for the quasi-complete case), on separable, nearly separable (one flipped row, one duplicated-with-other-label row) and non-separable families. `ml` with `l2 = 0` and 100 iterations must **never refuse a non-separable set**, must refuse separable sets with the margin the loss bound needs, and the sets that are separable but not refused are counted and each explained (iterations too few, quasi-complete); with `l2 > 0` nothing is refused | `python3 scripts/oracle_refuse.py sep` | a mutant that raises the bound above `ln 2` (false refusals on non-separable sets), or removes the test (separable sets pass) |
| D | The README's categorical flow still carries `--l2 0.1` and its output is unchanged; a generated block shows the refusal and the repair run | `scripts/site.py`, `scripts/check_docs.sh` | the block is generated from a real run: a change of the refusal changes the README and the check fails |
| E | **Mutants** `scripts/mutants_refusals.py` (a copy of the tree with every built executable, as `scripts/mutants_cli.py`): the tolerance, the pivot test, the column naming, the dependency coefficients, the separation bound, the loss comparison, the `pivot` late form, the repair (`--l2` value, the replace of an explicit `--l2`, the `--drop` name), `l2 > 0` not refused, the pass number. Baselines PASS first | `python3 scripts/mutants_refusals.py` | a survivor is explained in section 6 or the gate is wrong |
| F | **Fuzz and invariance.** Random designs (random rank, scales `1e-300` to `1e300`, no NaN, all rows equal, one row, `p` to 257): no trap, the verdict agrees with the exact oracle where it is not in the band; thread counts 1, 2, 4, 8, 16 give one answer byte for byte; the 22 fits of the cross-platform fingerprint give no refusal (`scripts/xplat_check.py` prints 88 lines equal to `docs/bench/xplat_fp.txt`); the example data give no new refusal | `scripts/oracle_refuse.py fuzz`, `scripts/xplat_check.py`, `scripts/gate_*` | as above |

## 4. The fixtures, written first (they fail on `main`)

`tests/conformance/fixtures.py` `_refuse_cases()`: 17 cases when written (20 now: the quasi-complete case, the ridge below the tolerance after the first pass, and a step that makes `--drop` unsafe). On `main` (run before the fix): the 7 collinear cases answer `ml.linalg.not-positive-definite`; `ml-fit-separable`
and `-explicit-zero-l2-is-replaced` and `-in-a-plan-has-no-argv` answer the same (100 iterations); `ml-fit-separable-25-iterations` **succeeds** (exit 0, the coefficients of section 1); that is 11 failing.
The 6 controls pass on `main` and must keep passing: a positive `l2` on separable data, two passes (too few to see it), overlapping classes, a nearly separable set with one
contradicting row, near-collinear columns (`1e-5` apart), the full one-hot block with `l2 = 0.1`.

## 5. What is not done

* No refusal for badly scaled data (docs/status.md "Newton has no damping" stays).
* Separation with `l2 > 0` is not refused (it is well posed), and a separable fit with too few passes is not seen (D2).
* Multiclass (#37) is not built; D3 says what it reuses.
* No `drop-first` encoder option (does not exist); no minimal-subset search for the columns to drop (the dependency is reported as data; the repair drops one named input column).

## 6. Results

Machine: Mac arm64, load average 90 to 100 from other jobs (so no timing is claimed), compiler a4572ea, seeds 20261015 (oracles) and 20261015 (`pivot_evidence_wide.py`).

### 6.1 The tolerance, measured (`python3 scripts/pivot_evidence_wide.py`)

No new number: `2^-40` stays (`models-linear.md` 3.1 has its measurements at p up to 21). What was not measured is its dependence on the size of the matrix. The relative pivot `s_jj / a_jj` is scale free by
construction; the rounding noise it carries is measured here at the sizes the estimators take (`oracle_linalg.chol_o`, the twin of `lib/linalg/chol.cho`, p = d + 1):

| data | d | n | smallest / largest relative pivot |
|---|---|---|---|
| well conditioned, standard normal | 8, 32, 64, 128, 256 | 500 and 5,000 | smallest 0.65 (d = 128, n = 500), 0.94 at d = 256, n = 5,000; 1.1e12 times the tolerance at n = 5,000 |
| last feature = a random combination of all the others (exactly dependent) | 3, 8, 32, 64, 128, 256 | 500 and 5,000 | noise between -4.1e-16 and +3.0e-16: **at most 4.5e-4 of the tolerance, 2,200 times below it**, with no growth in d |
| a constant column beside the intercept | 1, 8, 64, 256 | 2,000 | the intercept's relative pivot is +2.3e-16, -1.1e-16, 0, 0 (refused) |

The exact oracle (gate B, 3,000 designs) puts 1,128 designs at a relative pivot of 0 or below and the others across the decades from 1e-35 to 1e-1; **at 1e-13 and 1e-12 there are 108 and 72 designs, and the verdict
agrees with the exact one for every design**, so the decision at the boundary is the exact decision to the rounding of the sums. The dependency coefficients are the solution of a system whose matrix is the leading block
of the Gram matrix: their error is about the unit roundoff times its condition number (a near pair before the failing column loosens it; the oracle allows `max(1e-5, 1e-14 / smallest relative pivot before j)` times the largest
coefficient and the family "a near pair before a dependent column" exercises it). Not measured: p above 257 (the estimators refuse it), n above 20,000 for p above 3, other platforms (the fingerprint of `scripts/xplat_check.py` is equal, which says the bits agree, not that noise is the same).

### 6.2 Separation, measured (`python3 scripts/oracle_refuse.py sep --n 600`)

600 integer point sets (the line and the plane, coordinates -6 to 6, 6 to 14 points; the exact verdict by candidate normals in integer arithmetic): 282 complete separations, 292 overlaps, 26 separations with ties.
With `--l2 0` and 100 passes: **281 of the 282 complete ones are refused by the loss test, the last by the scale test; the 292 overlaps are all fitted; of the 26 ties, 21 are refused by the scale test and 5 at a vanishing pivot.**
No set of any kind is answered `ml.linalg.not-positive-definite` with `l2 = 0`; a positive `l2` (0.01 and 1e-9) is never refused for separation. An independent numpy Newton iteration (the same method, written apart) predicts
303 refusals and their pass, and `ml` gives the same reason at the same pass for all of them, with the summed loss and the norm equal to 1e-6; it predicts 292 quiet ends, and all 292 fit.

On the default data of the README (`shop.csv`, 6 rows): refused at pass 3 with a summed loss of 0.399, the norm of the coefficients 3.84. The first fixture (`sep.csv`): pass 2, loss 0.407, norm 1.447; with `--iterations 25`
the fit used to succeed with `x1 = 15.16` (section 1); with `--iterations 2` it still does (fixture): two passes do not reach the bound.

### 6.3 The gates

| gate | result |
|---|---|
| A fixtures and repairs | 20 cases of `_refuse_cases()` (one keeps `ml.linalg.not-positive-definite`: a ridge below the tolerance after the first pass) plus the old id `ml-linalg-not-positive-definite`, now `ml.fit.collinear`: all pass; every argv of every repair is run in the same sandbox and gives a model (`test_every_argv_of_every_repair_runs_and_gives_a_model`) |
| B rank oracle | 3,000 designs + 300 one-hot blocks, 0 disagreements (section 6.1) |
| C separation | 600 sets, 0 disagreements (section 6.2) |
| D README | the flow of pets.csv keeps `--l2 0.1` and its output is unchanged; the generated block `refuse` shows both refusals and runs the repair as offered (`scripts/site.py` raises if the repair does not run) |
| E mutants | `scripts/mutants_refusals.py`: see 6.5 |
| F fuzz, threads, fingerprint | `oracle_refuse.py fuzz --n 300`: 600 fits (both models) at 1, 2, 4, 8 and 16 threads, 0 traps, one answer each (verdicts: 42 fitted, 405 collinear, 88 non-finite, 6 separable, 59 single class); `scripts/xplat_check.py`: 88 lines for 22 fits equal `docs/bench/xplat_fp.txt` |

### 6.4 What changed and why

* `tests/conformance/fixtures.py` `ml-linalg-not-positive-definite` (two equal columns, `l2` 0): the tag is `ml.fit.collinear`. Status 8 and files are those of the baseline; the record `golden_numeric.json` holds the new answer.
* `tests/conformance/golden_planned.json`, 2 of 60: `logistic-5` (exit 0 with a model, now exit 8 and no file: separable data, 25 iterations, `l2` 0) and `logistic-14` (exit 8, tag changed). Both are re-recorded in `golden_planned.json` (`scripts/golden_track.py --write`) in the same commit as `golden_numeric.json`, which changes in one entry only; the other 58 planned cases and the other 273 numeric ones give the bytes `main` recorded (a diff of the records against `main` shows exactly these three entries). `golden_planned.record` also replaces the binary path in the output (no case had one).
* `tests/p_models.cho`: five expectations of the code 9 for a singular Gram matrix are the code 22; `named_refusals` is new.
* The worked examples: the penalty-free fits of breast cancer and wine (`--l2 0`) are refused with `ml.fit.separable`; `scripts/examples_common.py` has one such run each (it had two and three); the generated pages and `docs/examples.md` item 5 say so.
* Unchanged: the 22 fits of the fingerprint; the oracle suites of `lib/models` (bit for bit); every other conformance record. A fit that succeeded before and is not separable (completely, or with ties past the scale bound) succeeds with the same bytes.

### 6.5 Mutants (`python3 scripts/mutants_refusals.py`; log `docs/bench/mutants_refusals.log`)

34 mutants of the tolerance, the pivot test of the diagnosis, the pivot index, the relative pivot, the forward and backward solves of the dependency, the conversion of a failure at the first column, the first-pass test of
logistic regression, the column named (the pivot, the intercept), the cut that lists the columns, the ridge of both repairs, the replace of an explicit `--l2`, the `--drop` name and its guard, the plan form of both repairs, the loss bound
(too high, too low), the loss test removed, the loss test with a ridge, the scale test removed, its bound and its factor 4, the late pivot form removed and with a ridge, the pass number and the norm. **All 34 are killed.**
In the first run 3 survived and were not accepted as equivalent: the diagnosis with a tolerance 2^10 larger (the oracle had no design with a near but fine pair before a dependent column: the family `mixed` was added), the relative
pivot written as the sum (the oracle did not read it: it compares it with the exact one to 5e-15 now) and `--drop` offered beside a step (a fixture where the step names the column was added); each was killed on the re-run
(`docs/bench/mutants_refusals.log`). The unmutated baseline of that run failed one test, the wire oracle's expectation for separable logistic data (`scripts/oracle_wire.py` and `scripts/oracle_models.py` mirror the new refusals now); the
baseline is the full conformance suite of `scripts/check.sh`, which passes (section 6.8).

### 6.8 The full run

`sh scripts/check.sh` on the final tree, Mac arm64 under a load average near 100: 449 conformance tests OK (298 fixtures), every other step passes (53 minutes; no timing is claimed). `scripts/check_docs.sh`,
`check_prose.py`, `check_links.py`, `check_doc_refs.py`, `check_bins.py`, `check_workflow.py` pass. One native run on Linux x86-64 (gram, `taskset -c 0-6`): `xplat_check.py` 88 lines equal to the golden,
`gate_models.sh` all bits equal, `oracle_refuse.py all --n 400` every verdict agrees (the same verdicts as on the Mac), `test_refuse_fit` 8 tests OK.

### 6.6 The real session

`python3 scripts/site.py` runs, in a temporary directory with `ml` on the PATH, the one-hot refusal of `pets.csv` and the separable refusal of `shop.csv`, each followed by the repair it offered, run as offered. The blocks are in the README
(section "A fit that has no answer is refused by name") and `docs/index.html`.

### 6.7 Not done

Multiclass (#37); `drop-first` (does not exist); a minimal set of columns to drop (the dependency is data; the repair drops one input column); refusing separable data that the passes do not reach.

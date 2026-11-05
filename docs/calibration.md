# `ml calibrate`, `ml isotonic` and the `target` step: a calibration gate with an exit status, a monotone fit, and a target encoder

**Status: designed before any code. Nothing here is built.** Tags as in `design.md`: **[measured]** printed by a script, with its conditions; **[choice]** a decision the maintainer can overturn; **[read]** read from a file (named); **[open]** not decided and not measured.

## 0. Where this comes from

`cancho-consulting/metasearch-bidding` is moving from Python to cancho. Three pieces of its `service/bidding/model.py` have no counterpart in `ml` **[read]**, and `docs/multiclass.md` and `docs/status.md` already list two of the three as not done ("calibration"; "Target encoding"):

| Python | does | `ml` today |
|---|---|---|
| `calibration(p, y, bins, min_expected, z)` and `level_correction(p, y, days)` | quantile-bin table of predicted against observed, and a **pass/fail gate** on it; a recalibration of the level judged on the newer half of the days | no verb; `eval` gives metrics, not a gate **[read]** `docs/metrics.md` |
| `pav_increasing(y, w)` | weighted pool-adjacent-violators: the nearest non-decreasing sequence | no model or verb **[read]** `grep -i isotonic docs/*.md lib` finds nothing |
| `encodings(o, hotels, mature_mask)` | out-of-fold, shrunk **target encoding** of a high-cardinality category (and `log1p` of its count), with a full-data encoding for scoring | `encode`, `freq`, `hash` exist; **target is not in the plan verbs** **[read]** `docs/categorical-encoders.md`, `docs/status.md` line "Target encoding" |

Each is the standard form of a thing every probability model that gates itself on calibration, fits a monotone response, or has a category with thousands of levels needs. None is specific to hotels.

This document adds **two verbs and one plan step**: `ml calibrate`, `ml isotonic`, and `target` (the eighth plan verb: the plan grammar's "one of the seven verbs" **[read]** `docs/categorical-encoders.md` becomes eight). It adds no dependency.

### 0.1 A decision is an exit status

The client's design rule is "a decision is a verb with an exit status, not an expression in a plan" **[read]** `docs/pure-cancho.md` section 3 of the client's repository, which `cancho-flow` shares: a plan's only condition is `when:{step,exit}` **[read]** `cancho-flow/docs/design.md` section 6. So `calibrate` must be able to say *no* in its exit status and, in the same run, still print the table the operator needs to see why. Section 1.4 says how.

## 1. `ml calibrate`

```
ml calibrate (--model M.ml | --proba P) --label Y [--weight-by none]
             [--bins 10] [--min-expected 100] [--z 3] [--overall 0.95:1.05] [--band 0.8:1.25]
             [--recalibrate level --days D] [--gate] FILE
```

`FILE` is a table. The predictions are either scored by `--model` (a binary classifier; the probability of the positive class, as `predict --proba` writes it **[read]** `docs/cli.md`) or read from the column `--proba`. `--label` is a 0/1 column. Both defaults are the Python's.

### 1.1 The table, exactly

With `n` rows (rows with a missing or non-finite probability or label are counted in `skipped` and left out):

* Sort by probability, **stably** (ties keep file order; the Python's `argsort(kind="stable")`).
* Bin `i` (0-based, `i < bins`) holds the rows at positions `floor(i n / bins)` up to but not including `floor((i + 1) n / bins)`. A bin with no rows is left out of the table.
* Per bin: `n`, `mean_pred`, `observed` (mean label), `ratio = observed / mean_pred` (absent when `mean_pred = 0`), `expected_bookings = sum p`, `sd = sqrt(max(sum p (1 - p), 1e-12))`, `z = (sum y - sum p) / sd`, `judged = (expected_bookings >= min_expected)`.
* `overall_ratio = sum y / max(sum p, 1e-12)`.
* `bad_bins`: the judged bins whose `ratio` is outside `--band` **and** whose `abs(z) > z`.
* `ok = overall_ratio inside --overall and no bad bin`.

**The bin edges differ from the Python's in a case that matters.** The Python computes `np.linspace(0, n, bins + 1).astype(int)`, a floating-point product truncated; this document uses the exact integer `floor(i n / bins)`. **[measured]** (Apple M4 Max, Python 3.14, numpy 2.4.6, `n` from 1 to 30,000): for `bins = 5` and `bins = 10` the two are equal for every `n`; for `bins = 20` they differ in 136 values of `n`, the first at `n = 164` (edge 15: 122 against 123). The Python's default and the client's value are 10, so for them the verb equals the Python at every `n` tested; for other `bins` it is the exact one and the Python's is the approximate one. **[choice]** this is the correct side, and the client's gate gets it for free.

### 1.2 Arithmetic

Sums are exact (`acc.psum`-style accumulation rounded once, as `eval`'s metrics are **[read]** `docs/metrics.md`), so `mean_pred`, `observed`, `expected_bookings`, `sum p (1 - p)`, and `overall_ratio` are correctly rounded values of exact quantities; `sd` and `z` are the square root and quotient of rounded values and are documented as two roundings. Order of rows within a bin does not matter to any figure. **The sort is the only order-dependent step**, and it is stable on the file order.

### 1.3 `--recalibrate level --days D`

The Python's `level_correction`: `cut` is the median of the days `D`; the older half is `D <= cut`, the newer half `D > cut`; if either half is empty or the older half's `sum p` is not positive the factor is `1` and **no table is produced for the newer half** (the Python returns `None`; here `recalibrated: null` with `reason`). Otherwise `r = sum y / sum p` over the older half, and the table of section 1.1 is computed on the newer half with `p' = clip(p r, 0, 1)`. The factor and both halves' counts are in the answer. The point of the design, which the verb keeps: **the gate is judged on days the factor was not fitted on**, so it cannot be passed by fitting it **[read]** `model.py` docstring.

The median is the median of the doubles `D` when `D` is numeric; **[choice]** with an even number of rows it is the mean of the two middle values, as numpy's, and a day column that is not numeric is refused (`ml.calibrate.days-not-numeric`).

### 1.4 The exit status

Two behaviours, one flag **[choice]**:

* without `--gate`: the answer is printed and the exit status is 0 whether `ok` is true or false. This is `eval`'s behaviour: the verb reports.
* with `--gate`: the same answer is printed, then **exit 1 with the rule tag `ml.calibrate.gate-failed`** when `ok` is false. The refusal carries the table in `data` (a refusal that holds the evidence), the overall ratio, and the bad bins; `repair` is absent (nothing about the request is wrong).

A caller that puts `calibrate --gate` in a plan reads the exit status; a caller that wants the numbers reads the data. **[open]** whether a refusal that prints a full answer is acceptable to the contract (`cancho-tools` `contract/fail`): the alternative is a different exit code for "the gate said no" that is not a refusal, which the contract has no slot for.

### 1.5 Refusals **[choice]**

`ml.calibrate.labels-not-binary` (a value other than 0 and 1), `ml.calibrate.proba-out-of-range` (outside `[0, 1]`), `ml.calibrate.too-few-rows` (fewer than `bins`), `ml.calibrate.bins-range` (1 to 1000), `ml.calibrate.model-not-binary`, `ml.calibrate.days-not-numeric`, `ml.calibrate.band-order` (low above high), and the reader's and limits' existing tags. A probability exactly 0 or 1 is allowed.

## 2. `ml isotonic`

```
ml isotonic --x X --y Y [--weight W] [--decreasing] [--output CSV] FILE
```

The weighted least-squares monotone fit, by pool-adjacent-violators. Output: one row per distinct `x` in increasing order: `x`, `fitted`, `weight` (sum of the weights at that `x`), `block` (index of the pooled block that holds it), and per block `n_points`.

### 2.1 Definition

Rows are sorted by `x` (stable); rows with equal `x` are **first pooled into one point** with weight `sum w` and value `sum(w y) / sum w`, then PAV runs on the distinct points. (The Python is given points already distinct and in order; pooling ties is the standard definition and does not change its input.) A block's value is `sum(w y) / sum(w)` over its members, **correctly rounded once from exact sums**; two adjacent blocks are merged when the first's value is **strictly greater** than the second's, so equal blocks stay separate, which changes no fitted value **[read]** `pav_increasing`. `--decreasing` is the same with the comparison reversed.

**Weights must be positive.** The Python guards a block weight with `max(w, 1e-12)` and its caller passes `1e-6` for a value borrowed from a neighbour **[read]** `fit_response`; a weight of 0 is ambiguous (no information, or a division by zero), so the verb refuses it (`ml.isotonic.weight-not-positive`) and the caller states the pseudo-weight it means. Missing or non-finite cells are refused by default (`ml.isotonic.missing`); **[open]** `--skip-missing`.

### 2.2 What it is not **[choice]**

Not a model file in v1: no `fit | predict` pair, no interpolation between `x` values, no `--min-points`, no confidence band. The client reads the table. A model kind `isotonic` that predicts at new `x` by step or by linear interpolation is a later change, because its interpolation rule is a choice that the Python (which reads the fitted values at its own bin centres) does not make.

### 2.3 Where a use of it ends: the response curve

The client's `fit_response` (bin the bid over a reference value, share won and price paid per bin, each made monotone) is `table` for the binning and sums, and `ml isotonic` twice. Nothing else in it is new.

## 3. The `target` plan step

```
target column=hotel_id [k=50] [folds=5] [fold-by=click_id] [fit-rows=mature] [log-count=yes] [missing=pass]
```

A categorical column (any column the plan can declare categorical, **[read]** `docs/categorical-encoders.md` section 2) becomes two numeric columns: `hotel_id_te` and `hotel_id_logn` (the second omitted with `log-count=no`).

### 3.1 The encoding **[read from the Python; all choices of the client carried over]**

For a category `c` with `n_c` fit rows and `b_c` positives among them, and `p0 = sum_c b_c / sum_c n_c` over all fit rows:

* `te(c) = (b_c + k p0) / (n_c + k)`, and `logn(c) = log1p(n_c)`.
* **Out of fold.** Every row has a fold `f(row) = fold-by value mod folds`, for an integer column (when `fold-by` is absent, the row's 0-based position mod `folds`). For a *training* row in fold `f`, `n_c` and `b_c` are those of the fit rows **not in fold `f`**; `p0` is the full-data one. The row therefore never sees its own label through its encoding; this is what makes the feature safe to fit a model on.
* **Scoring** (`predict`, `eval`, `serve`) uses the full-data `n_c`, `b_c` stored in the model file: a row never in the training file gets the same encoding as any other row of its category.
* An unseen category gets `te = p0`, `logn = 0` (`n_c = 0`), and is counted in `unseen` as `encode` does.
* **`fit-rows`** names a 0/1 column: only rows with 1 contribute to `n_c`, `b_c` and `p0` (the Python's *mature* clicks: rows whose label is still unknown are encoded but never counted). Absent, all rows with a known label count.

### 3.2 The decision a reader should see

Out-of-fold encoding is **not free**: a training row's encoding is made from 80% of the data (with 5 folds) and a scoring row's from 100%, so train and score encodings differ in noise. The Python accepts that and so does the design **[choice]**; it is the standard remedy for the leakage of a naive target mean and the alternative (a naive mean) leaks. The model file stores the full-data table, not the folds. `describe` gives `k`, `folds`, `fold-by`, `fit-rows`, and the per-column `unseen` rate seen at the last eval.

### 3.3 Where it sits in the plan

`target` is a step like `encode`: it is declared in a plan before `model`, it needs the target column to be named (`--target` of `fit`), and it **refuses to run without one** (`ml.plan.target-needs-label`). It cannot be used in a pipeline of `cluster`/`kmeans` (no label). Step kind: **11** in the model-file layout (the next after `hash`'s 10 **[read]** `docs/categorical-encoders.md` section 5.2, 7 to 10), with the string table holding the categories and, per category, `n_c` and `b_c` as exact integers (`i64`), plus `p0` as a double and `k`, `folds`.

### 3.4 Refusals **[choice]**

`ml.plan.target-needs-label`, `ml.plan.target-column-not-categorical`, `ml.plan.target-k` (`k` not positive and finite), `ml.plan.target-folds` (below 2 or above the number of fit rows), `ml.plan.target-fold-by` (not an integer column), `ml.plan.target-fit-rows` (not 0/1), `ml.fit.target-label-not-binary` (**[open]**: v1 supports a 0/1 label and a regression target, for which `te` is the shrunk mean of the target and `p0` the overall mean; whether to ship the regression form in the first change), and the limits on the number of categories already in force for `encode`.

## 4. What is not in this change **[choice]**

A marginalising predict (the client's `Scorer` averages `P(book)` over a grid of cancellation policies, leads and pairs): that is a cross product in `table`, `predict --proba`, and a weighted mean by `table --group --agg`, with nothing for `ml` to add. A calibration *map* (Platt or isotonic recalibration applied at predict time): `calibrate` judges, `isotonic` fits a monotone table, and applying one is the client's. Reliability diagrams (an image). Multiclass calibration (`docs/multiclass.md` lists it as not done and still is).

## 5. Gates, written before the code; each must be able to fail

| id | judges | how | fails on |
|---|---|---|---|
| C1 | `calibrate` by hand | a fixture of 20 rows and 4 bins, every figure in this document worked on paper (section 8) | any definition change |
| C2 | `calibrate` against numpy | 1000 seeded cases (n from 1 to 20,000, bins 5, 10, 20, ties in `p`, `p` at 0 and 1, all-zero labels): table, `overall_ratio`, `bad_bins`, `ok` equal to the client's Python `calibration()` re-implemented in `scripts/oracle_calibrate.py` from section 1.1 to a **relative 1e-12** for the sums and **exactly** for the integers and `ok`; for `bins = 20` the oracle uses the exact edges (the Python's are an approximation, section 1.1) | a difference |
| C3 | edges | every `n` from 1 to 30,000 for `bins = 5, 10`: the verb's edges equal `linspace(...).astype(int)`; for 20, equal `i n // bins` | a difference |
| C4 | the level correction | the gate on the newer half only; a mutant that fits and judges on the same rows is killed by a table where it would pass and the honest one fails | the mutant surviving |
| C5 | exit status | with `--gate`, exit 1 and tag `ml.calibrate.gate-failed` exactly when `ok` is false; without it, exit 0 either way | a wrong status |
| C6 | order | the same table in 20 row orders: identical bytes up to the stable sort's tie order (so tied `p` are given distinct labels in the test, and a case with ties of `p` and different labels proves the tie order is the file's) | any other dependence |
| I1 | `isotonic` by hand | the fixture of `pav_increasing` with a violator at the start, the middle and the end | any change |
| I2 | `isotonic` against numpy and scikit-learn | 1000 seeded cases (ties in `x`, weights from 1e-6 to 1e6): fitted values equal to an independent PAV in `scripts/oracle_isotonic.py` exactly (the arithmetic is `sum(w y) / sum w`), and to `sklearn.isotonic.IsotonicRegression` within a **relative 1e-9** **[choice]** (scikit-learn is used by the oracle only; not a dependency) | a gap above the bound |
| I3 | the fit is monotone and is the minimum | the output is non-decreasing; no neighbouring-block merge or split lowers the weighted sum of squares (checked by exhaustive search on cases of up to 8 points) | a non-monotone output or a better fit exists |
| I4 | refusals | each tag reached | a tag that cannot be reached |
| T1 | `target` by hand | a 12-row fixture, 3 categories, 3 folds, every `te` worked on paper | any change |
| T2 | no leakage | a category with one row: the encoding of that row is `p0`'s prior (`n_c = 0` out of fold), and a model fitted on `te` alone cannot read its label back (a property test: the fitted `te` of a row is independent of its own label) | the encoding depending on the row's own label |
| T3 | against the Python | the client's `encodings()` re-implemented in `scripts/oracle_target.py`: `te_click`, `logn_click`, `te_hotel`, `logn_hotel`, `p0` equal to **relative 1e-12** on 200 seeded tables (including a category of one row and rows outside `fit-rows`) | a difference |
| T4 | the plan, the file, the serve | the step round-trips through the model file; `describe` reports it; `serve` and `predict` give the same encoding for a row; a model file with a changed `n_c` is refused by `decode` | any difference |
| T5 | train/serve skew | `ml runs`' skew check **[read]** `docs/tracking.md` notices an unseen-category rate | the check missing it |
| X1 | no panic | the fuzz of `tests/t_metrics_fuzz.cho` extended to the three: random lengths, non-finite values, extreme labels | a trap |
| X2 | determinism | thread counts 1 and 8 and the platforms of the CI matrix: identical bytes | any difference |

Each gate ships with its mutants (`scripts/mutants_calibrate.py`, `mutants_isotonic.py`, `mutants_target.py`): all killed, or each survivor explained. No file over 2,000 lines: `lib/metrics/calibrate.cho`, `lib/metrics/isotonic.cho`, `lib/transforms/target.cho`.

## 6. Order and size **[choice]**

1. `calibrate` (smallest, unblocks the client's refit gate and the level correction); C1 to C6, X1, X2.
2. `isotonic`; I1 to I4.
3. `target`, last, because it changes the model-file layout (a new step kind), the plan grammar and `describe`; T1 to T5.

## 7. What the consumer must accept

A. `calibrate`'s edges for `bins` other than 5 and 10 are the exact ones, not the Python's (section 1.1). The client uses 10.

B. `isotonic` refuses a zero weight; the client passes `1e-6` explicitly where it means "borrowed".

C. The out-of-fold encoding is made by a fold rule the client names (`click_id mod 5`). The client's value and `fit-rows` (mature clicks) are plan arguments, not defaults.

## 8. Hand-worked fixtures

*Written with the code of each gate, before it: C1 (20 rows, 4 bins), I1, T1.* **[open]**

## 9. Open questions for the maintainer

1. **A gate verb that fails with its evidence** (section 1.4): a refusal carrying `data`, or a different exit code? The contract's choice decides.
2. **The regression form of `target`** in the first change, or binary only?
3. **A model kind `isotonic`** now, or later (section 2.2)?
4. **`--skip-missing`** on `isotonic` and `target`, or refusal only?
5. **The eighth plan verb** changes the canonical form of every plan's text? (It must not: plans that do not use `target` keep their bytes. Gate T4 includes the check.)

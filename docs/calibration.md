# `ml calibrate`, `ml isotonic` and the `target` step: a calibration gate with an exit status, a monotone fit, and a target encoder

**Status: designed before any code. The calibration table and gate (section 1), the isotonic fit (section 2) and the out-of-fold target encoding (section 3) are built as libraries (sections 10 and 11 say what was built, what was measured, and where the build differs from this design); `ml calibrate` and `ml isotonic` are on the command line (section 12); the `target` step in the model file and the plan is not built.** Tags as in `design.md`: **[measured]** printed by a script, with its conditions; **[choice]** a decision the maintainer can overturn; **[read]** read from a file (named); **[open]** not decided and not measured.

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

*(Corrected by the build, section 10: the sums are not exact.)* The sums of the probabilities and of `p (1 - p)` are the package's decisive sums (`metrics.sums`: blocks of 256 terms, four lanes, the fixed tree **[read]** `docs/metrics.md` section 2.4, `lib/linalg/order.cho`), the observed rate is the correctly rounded quotient of two exact integers (`metrics.ratio`), and `sd` and `z` are the square root and the quotient of rounded values; the gate compares them with the exact values to a relative 1e-12 (and `z`, a difference of nearly equal sums, to 1e-12 of the size of its operands). Order of rows within a bin does not matter to any figure. **The sort is the only order-dependent step**, and it is stable on the file order.

### 1.3 `--recalibrate level --days D`

The Python's `level_correction`: `cut` is the median of the days `D`; the older half is `D <= cut`, the newer half `D > cut`; if either half is empty or the older half's `sum p` is not positive the factor is `1` and **no table is produced for the newer half** (the Python returns `None`; here `recalibrated: null` with `reason`). Otherwise `r = sum y / sum p` over the older half, and the table of section 1.1 is computed on the newer half with `p' = clip(p r, 0, 1)`. The factor and both halves' counts are in the answer. The point of the design, which the verb keeps: **the gate is judged on days the factor was not fitted on**, so it cannot be passed by fitting it **[read]** `model.py` docstring.

The median is the median of the doubles `D` when `D` is numeric; **[choice]** with an even number of rows it is the mean of the two middle values, as numpy's, and a day column that is not numeric is refused (`ml.calibrate.days-not-numeric`).

### 1.4 The exit status

Two behaviours, one flag **[choice]**:

* without `--gate`: the answer is printed and the exit status is 0 whether `ok` is true or false. This is `eval`'s behaviour: the verb reports.
* with `--gate`: *(corrected by the build, section 12: the answer is not printed with the refusal; the exit is 8, not 1)* the same answer is printed, then **exit 1 with the rule tag `ml.calibrate.gate-failed`** when `ok` is false. The refusal carries the table in `data` (a refusal that holds the evidence), the overall ratio, and the bad bins; `repair` is absent (nothing about the request is wrong).

A caller that puts `calibrate --gate` in a plan reads the exit status; a caller that wants the numbers reads the data. **[open]** whether a refusal that prints a full answer is acceptable to the contract (`cancho-tools` `contract/fail`): the alternative is a different exit code for "the gate said no" that is not a refusal, which the contract has no slot for.

### 1.5 Refusals **[choice]**

*(Corrected by the build: the library reuses the codes of `lib/metrics/rules.cho`, so no tag, registry entry or spec table changes; the `ml.calibrate.*` tags below are what the wiring stage may add.)* As built: `ml.metric.undefined` (no rows), `ml.metric.length-mismatch`, `ml.label.not-binary` (a label other than 0 and 1), `ml.metric.bad-probability` (outside `[0, 1]`), `ml.input.non-finite`, `ml.param.out-of-range` (bins below 1, above 1000 or above the rows; a cap outside its range), `ml.input.too-many-rows`. Designed for the wiring stage: `ml.calibrate.model-not-binary`, `ml.calibrate.days-not-numeric`, `ml.calibrate.band-order`. A probability exactly 0 or 1 is allowed.

## 2. `ml isotonic`

```
ml isotonic --x X --y Y [--weight W] [--decreasing] [--output CSV] FILE
```

The weighted least-squares monotone fit, by pool-adjacent-violators. Output: one row per distinct `x` in increasing order: `x`, `fitted`, `weight` (sum of the weights at that `x`), `block` (index of the pooled block that holds it), and per block `n_points`.

### 2.1 Definition

Rows are sorted by `x` (stable); rows with equal `x` are **first pooled into one point** with weight `sum w` and value `sum(w y) / sum w`, then PAV runs on the distinct points. (The Python is given points already distinct and in order; pooling ties is the standard definition and does not change its input.) A block's value is `sum(w y) / sum(w)` over its members *(corrected by the build, section 10: the two sums are plain double sums added in the order the stack merges the blocks, which is the arithmetic of the client's `pav_increasing` and gives its bits; they are not exact sums)*; two adjacent blocks are merged when the first's value is **strictly greater** than the second's, so equal blocks stay separate, which changes no fitted value **[read]** `pav_increasing`. `--decreasing` is the same with the comparison reversed.

**Weights must be positive.** The Python guards a block weight with `max(w, 1e-12)` and its caller passes `1e-6` for a value borrowed from a neighbour **[read]** `fit_response`; a weight of 0 is ambiguous (no information, or a division by zero), so the fit refuses it (as built: `ml.param.out-of-range`) and the caller states the pseudo-weight it means. A non-finite `x`, `y` or weight is `ml.input.non-finite`; **[open]** `--skip-missing` for the wiring stage.

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
* **Out of fold.** Every row has a fold `f(row) = fold-by value mod folds`, for an integer column (when `fold-by` is absent, the row's 0-based position mod `folds`). For a *training* row in fold `f`, `n_c` and `b_c` are those of the fit rows **not in fold `f`**; `p0` is the full-data one. *(Corrected by the build, section 11: the claim that followed here, that a row never sees its own label, is false as stated.)* The **counts** of a row's encoding never contain the row or any row of its fold, but `p0` is the base rate of ALL fit rows and contains the row's own label with weight `1 / (fit rows)`: flipping one label moves every encoding by `k (p0' - p0) / (n + k)` (a row of the same category in another fold moves by more, its counts hold the flipped row). The path is small and exact, it is the client's, and the gate T2 tests the identity; an encoding that removed it would take `p0` out of fold too, which is a change of the statistic and not a port.
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
| C5 | exit status | with `--gate`, exit 8 (corrected from 1, section 12) and tag `ml.calibrate.gate-failed` exactly when `ok` is false; without it, exit 0 either way | a wrong status |
| C6 | order | the same table in 20 row orders: identical bytes up to the stable sort's tie order (so tied `p` are given distinct labels in the test, and a case with ties of `p` and different labels proves the tie order is the file's) | any other dependence |
| I1 | `isotonic` by hand | the fixture of `pav_increasing` with a violator at the start, the middle and the end | any change |
| I2 | `isotonic` against numpy and scikit-learn | 1000 seeded cases (ties in `x`, weights from 1e-6 to 1e6): fitted values equal to an independent PAV in `scripts/oracle_isotonic.py` exactly (the arithmetic is `sum(w y) / sum w`), and to `sklearn.isotonic.IsotonicRegression` within a **relative 1e-9** **[choice]** (scikit-learn is used by the oracle only; not a dependency) | a gap above the bound |
| I3 | the fit is monotone and is the minimum | the output is non-decreasing; no neighbouring-block merge or split lowers the weighted sum of squares (checked by exhaustive search on cases of up to 8 points) | a non-monotone output or a better fit exists |
| I4 | refusals | each tag reached | a tag that cannot be reached |
| T1 | `target` by hand | a 12-row fixture, 3 categories, 3 folds, every `te` worked on paper | any change |
| T2 | the path of a label | *(rewritten by the build, section 11)* flipping the label of a fit row changes the encoding of each row of ITS OWN FOLD by exactly `k (p0' - p0) / (n + k)` and by nothing else, and moves some row of the same category in another fold; a category of one row is encoded from nothing (`n = 0`: te = p0, logn = 0); the naive full-data encoding fails the property | the encoding depending on the row's own label |
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

**C1** (`tests/t_calibrate.cho`): eight rows, probabilities 0.5, 0.125, 0.25, 0.75, 0.375, 0.625, 0.875, 0.0625 (all multiples of 1/16, so every sum is exact), labels 1, 0, 0, 1, 0, 1, 1, 0, four bins. Sorted: bins `{1/16, 1/8}`, `{1/4, 3/8}`, `{1/2, 5/8}`, `{3/4, 7/8}`. Per bin: `n = 2`; mean predicted 3/32, 5/16, 9/16, 13/16; positives 0, 0, 2, 2 so observed 0, 0, 1, 1; expected positives 3/16, 5/8, 9/8, 13/8; ratio 0, 0, 16/9, 16/13; `sum p (1 - p)` = 43/256, 27/64, 31/64, 19/64; `z` = (positives - expected) / sqrt of that = -0.457, -0.962, +1.571, +0.516. With `min_expected` 1, `z` limit 1, overall band 0.95 to 1.05, bin band 0.8 to 1.25: bins 2 and 3 are judged (expected at least 1); bin 2 is bad (ratio 16/9 outside the band, `|z|` 1.57 above 1), bin 3 is not (16/13 = 1.23 is inside); overall = 4 / (57/16) = 64/57 = 1.123, above 1.05, so `ok` is false with one bad bin.

**I1** (`tests/t_isotonic.cho`): `x = 1..5`, `y = 1, 3, 2, 4, 3.5`, unit weights: the pairs (3, 2) and (4, 3.5) pool to 2.5 and 3.75; fit 1, 2.5, 2.5, 3.75, 3.75 in blocks 0, 1, 1, 2, 2. `x = 1, 2, 3`, `y = 3, 1, 2`, weights 1, 1, 2: the first pair pools to 2, the third point's value is 2, equal to it, so the blocks stay apart: fit 2, 2, 2 in blocks 0, 0, 1. Ties: `x = 1, 1, 2`, `y = 2, 4, 1`, weights 1, 3, 2: the point `x = 1` has weight 4 and value 14/4, it violates `x = 2` (value 1, weight 2) and both pool to 16/6.

**T1** (`tests/t_targetenc.cho`): six rows, categories 0 0 0 1 1 1, labels 1 0 1 0 0 1, all fit, folds 0 1 0 1 0 1, `k = 2`. Category 0 has 3 rows and 2 positives, category 1 has 3 rows and 1: `p0 = 3/6 = 1/2`, `k p0 = 1`. Out of fold: row 0 (category 0, fold 0) sees 1 row and 0 positives: `te = (0 + 1) / (1 + 2) = 1/3`, `logn = log1p(1)`; row 1 (fold 1) sees 2 rows and 2 positives: `3/4`, `log1p(2)`; row 2 as row 0; row 3 (category 1, fold 1) sees 1 row, 0 positives: `1/3`; row 4 (fold 0) sees 2 rows, 1 positive: `(1 + 1) / (2 + 2) = 1/2`, `log1p(2)`; row 5 as row 3. Full data, for scoring: category 0 `(2 + 1) / (3 + 2) = 3/5`, category 1 `(1 + 1) / 5 = 2/5`, both `log1p(3)`.

## 9. Open questions for the maintainer

1. **A gate verb that fails with its evidence** (section 1.4): a refusal carrying `data`, or a different exit code? The contract's choice decides.
2. **The regression form of `target`** in the first change, or binary only?
3. **A model kind `isotonic`** now, or later (section 2.2)?
4. **`--skip-missing`** on `isotonic` and `target`, or refusal only?
5. **The eighth plan verb** changes the canonical form of every plan's text? (It must not: plans that do not use `target` keep their bytes. Gate T4 includes the check.)

## 10. What the build measured, and where it differs from sections 1 to 9

Machine: Apple M4 Max, macOS, Python 3.14, numpy 2.4.6, scikit-learn present, the compiler `0207556` of `cancho.toml`; one run each.

**Staging.** The repository's own pattern for a new group of numbers is a library first, a driver, an independent oracle and mutants, and the wiring into the command line at a later stage (`docs/metrics.md` did it for the metrics, `docs/linalg.md` for the kernels). `lib/metrics/calibrate.cho` and `lib/metrics/isotonic.cho` follow it: the verbs `ml calibrate` and `ml isotonic`, their flags, spec rows, schemas, fixtures, MCP tools and pages are **not built**, and so the client cannot call them yet. The driver `tools/metrics-check` has the subcommands `calibrate`, `level` and `isotonic`.

**Differences from the design** (each corrected in place above): the sums are the package's, not exact; the refusals reuse the existing codes; empty input is `ml.metric.undefined`; the isotonic arithmetic is plain doubles in the merge order of the client's `pav_increasing`.

**C2** (`scripts/gate_calibrate.py`, 300 random tables from 1 to 20,000 rows, 1 to 20 bins, ties, probabilities at 0 and 1, all-zero labels, a bin with positives at probability 0, a total probability below 1e-3): the integers, the observed rate and the flags equal an independent implementation in exact fractions, the floats within a relative 1e-12 (the `z` bound is relative to the size of its operands: the first run showed `z` values near 1e-3 that were differences of sums near 100, and a bound relative to `z` itself was the wrong bound). 125 per-bin comparisons of a flag were left out because the value sat within 1e-9 of its threshold. **C4**: 40 level-correction cases and one with an older half whose probabilities sum to 0. **C3**: the exact bin edges equal `linspace` truncated for 5 and 10 bins at every `n` to 30,000 and differ for 20 bins at `n = 164` (the gate has that case). **C5** is not built: it is the exit status of the verb. Controls changed the answer on 26 of the 65 cases with ties (an unstable order), 51 of 299 (the bad-bin rule without the `z` condition), 1 of 299 (edges from `linspace`, which is why the fixed case `n = 164` with 20 bins exists), and 17 of 47 level cases (fitting and judging on one half). Mutants (`scripts/mutants_calibrate.py`): **35 of 35 killed**; the first run killed 31, and the four survivors were gaps in the gate's cases (no bin with mean 0 and positives; no total below 1e-3; no table balanced overall but bad in a bin; no older half summing to 0), now covered.

**I2 to I4** (`scripts/gate_isotonic.py`, 400 cases of 1 to 300 points with ties, weights from 1e-6 to 1000, the increasing and the decreasing fit): the fit has the BITS of the same algorithm in doubles written separately in the gate, is within a relative 1e-12 of the exact fraction PAV, has the same blocks, is monotone, and for 274 cases of up to 8 points its weighted sum of squares equals the minimum over all 2^(n-1) partitions into consecutive blocks; 226 cases agree with `sklearn.isotonic.IsotonicRegression` within 1e-9. The driver gives the bits of the client's `pav_increasing` on 300 random inputs of distinct `x`. Unit tests: `tests/t_calibrate.cho` (section 8, C1) and `tests/t_isotonic.cho` (I1). Mutants (`scripts/mutants_isotonic.py`): **24 of 25 killed, 1 survived and explained** (`tie-test`: equivalent, the rows are sorted). The first run found that a mutant which never pops the stack makes the unit test loop forever, which the harness took for a crash rather than a kill; the gates now cut a hung unit test at 120 s and a hung driver at 30 s and count both as failures. The mutants of the calibration table were re-run on the final gate: 35 of 35 killed.

**Not built, not measured:** the `ml calibrate --gate` exit status and its refusal-with-evidence question (question 1); `ml isotonic` as a verb; the `target` step (T1 to T5); the fuzz of the driver for the two new subcommands (`scripts/gate_metrics.py --fuzz` covers the older ones); the cost of either function.

## 11. What the build of the target encoding measured, and what it corrected

Machine: Apple M4 Max, macOS, Python 3.14, numpy 2.4.6, the compiler `0207556`; one run each. **Library only**, as sections 1 and 2: `lib/metrics/targetenc.cho` is the function of integer codes (`encode`), with the driver subcommand `target`; the model-file step kind, the plan verb `target` with its grammar, `describe`, `serve` and the string table are **not built**.

**A claim of this document was false, and is corrected in place (section 3.1).** The text said that a row "never sees its own label through its encoding". The first run of the gate for T2 found 17,947 differences: `p0` is the base rate of ALL fit rows, so flipping one label moves `p0` by `1 / (fit rows)` and with it the encoding of every row, own fold included, by `k (p0' - p0) / (n + k)`. The counts of a row's own fold are excluded, so this is the only path; it is small, exact and the client's. T2 now tests exactly that identity for the rows of the flipped row's own fold (within 8e-16 of the encoding) and that some row of the same category in another fold moves.

**T3** (`scripts/gate_target.py`, 300 seeded tables of 1 to 1000 rows, 1 to 30 categories, 2 to 7 folds, `k` from 0.5 to 1000, rows outside the vocabulary, negative and large fold values, a share of rows not fit): counts, `p0` and every integer equal an independent implementation in exact fractions; the floats are within a relative 1e-12. Against the client's `encodings()` on 60 random tables (scratch check, not a gate: it needs the client's repository), the largest absolute difference is 8.9e-16. The identity of T2 was checked on 275 cases, and another fold moved on 208 of them. Controls (the full-data encoding, `p0` over every row, a fold of `fold mod (folds + 1)`) changed the answer on 199, 223 and 187 of 300 cases. Mutants (`scripts/mutants_target.py`): **30 of 30 killed**; the first run killed 29, and the survivor was a gap in the gate (a table with no category at all was never tried).

**Not built, not measured:** T4 (the step in the model file, `describe`, `serve`), T5 (the skew check), the regression form (question 2), the cost of `encode`.

## 12. What the wiring of `ml calibrate` and `ml isotonic` measured, and what it corrected

The two functions are on the command line (`tools/ml/calcli.cho`; the flags are in `docs/cli.md` section 12). The tables the repository keeps of its own surface were brought up to date and are the gates of this change: `tools/ml/spec.tmpl` (twelve flags, the per-verb lists of flags, the usage, the rules), `scripts/schemas.py` (the two answers), `scripts/mcp_tools.py` (two tools, both read-only: 12 tools in all), the authority manifests (no new capability: the two verbs read a file as `eval` does and write nothing), `tests/conformance/fixtures.py` (20 cases that every flag and every rule tag must be reached by).

**Corrected by the wiring.** (1) `--bins` already exists (the histogram bins of the boosted trees), so the number of bins is `--prob-bins`. (2) The gate's refusal does not carry the table: a refusal prints `error` and no `data`; its `detail` holds `bad_bins` and `rows`, and the table is the answer of the same call without `--gate`. This settles question 1 of section 9: **a gate that fails with its evidence is not possible in this contract; the table is one call without `--gate`, the decision is one call with it.** (3) The exit of `ml.calibrate.gate-failed` is **8**, not 1: the independent test of the D4 classes (`test_exit_codes_are_the_d4_classes`) places a verdict about the data in the class of the data (8) and keeps 1 for a failed read or write. A plan that must tell a failed gate from an unreadable file reads the rule tag in the output, not the exit status. (4) The model of the verbs is the data file only: `--model-file` scoring (`ml calibrate --model M FILE`, section 1) is not built. (5) The library's `ml.metric.length-mismatch` cannot be reached from the command line (the columns of one file have one length) and stays pending.

**Measured** (Apple M4 Max; one run each): the 20 fixture cases pass through the command line (`test_rules`), the tables, the schema and the MCP schema tests pass (`test_tables`, `test_schema`, `test_mcp_schema`), as do `test_fuzz_args`, `test_mcp`, `test_authority` and `test_log` (59 tests, the first full run of which found the two things above; after them `test_rules` passes). The gates of the libraries (`gate_calibrate.py`, `gate_isotonic.py`, `gate_target.py`) and their mutants are unchanged by the wiring.

**Not done:** the rest of the conformance suite was not run (it takes more than an hour); the pages and the README (`scripts/site.py`) were not regenerated; `scripts/mutants_cli.py` and the mutants of the MCP server were not run.

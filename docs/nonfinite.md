# A prediction that is not a finite number is a refusal (issue 33)

**Status: this document is written before the code it judges. Sections 1 to 6 fix the behaviour and the gates; section 7 is filled only with what a script of
this repository printed. Tags as in `design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from the code, **[measured]** printed by a
script here, **[open]** not decided.**

## 1. The problem

`ml predict` and `ml-predict` print the double of a prediction through cancho-table's `flt.put_float`. **[read]** That function takes the key of a finite double:
it calls `std.fmt.float_into` (which writes `inf` for an infinity), then splits the text into digits and an exponent, and with no `e` in `inf` the digit loop reads
the letters as digits. The cell is the text `i.nf`, inside a JSON string, with exit 0. **[measured at the issue, main 13d6c4a]** A linear model fitted on
`x = 1.5 .. 5.5` and asked for `x = 1e308` prints `[["i.nf"], ["25.500000000000014"]]`. `ml-serve` already refuses such a row (`serve.non-finite-prediction`, 422)
because its answer is built from the document `ml predict` makes and it finds a cell that is not a number. A client of the command line gets the text.

The command line must refuse as the server does, with one rule, for all three binaries.

## 2. The rule **[choice]**

| tag | exit | when | repair |
|---|---|---|---|
| `ml.predict.non-finite` | 8 | a value the model computes for a row of the input and the answer would carry or rest on is not a finite number: a prediction, a probability, a squared distance | none: the row is named in `detail`; scale or check the cell, or drop the row |

The name is `<area>.<name>` with the area `predict`, the verb whose answer is refused (`eval` scores through the same predictors and raises the same rule, so a
second tag for one rule would be the repurposing `design.md` section 8 forbids). Exit 8 is a precondition: the answer is no (`docs/cli.md`). The registry code is
21 of `lib/core/refuse.cho` (the band 1 to 99 of `ml.refuse`), `design.md` section 8.1 and `tools/ml/spec.tmpl` `extra()` and `rules()`.

`detail` is `{"row": N, "column": C}`: `N` is the 1-based number of the first row, in the order of the file read, whose value is not finite, and `C` is the name of the
column of the answer that holds it: `prediction` (a regression), `label` (the class a classification predicts: not reachable for the kinds built, which threshold a probability), `probability` (a classification), `distance` (a clustering). For `eval` with `--split`, `N` is the
row of the file, not the row among the held-out ones, because the file is what the caller can open.

**Not a finite number** is NaN or an infinity; the test is `v - v == 0.0`, the one `models.common.is_finite` has. NaN counts: `inf - inf` in a dot product is NaN
**[read]**, and a NaN prediction is not `> 0.5`, so a classification would silently choose the other label.

## 3. Where a non-finite value can reach a printed cell

Every call of `flt.put_float` in `tools/ml` and `tools/ml-serve` **[read]**, and what bounds its argument:

| call | what it prints | can the double be non-finite? | decision |
|---|---|---|---|
| `wireio.put_cell` | a prediction, a probability, a squared distance of `predict` | yes, for an input the model was not fitted for (a feature near 1e308 times a coefficient above 1, a sum of two such terms; a squared distance past 1.8e308) | **refused** by the check of section 4 |
| `wiredata.member` (`eval`) | a metric | a metric with no value is `undefined`, printed as `null` and named in `undefined` (`metrics.rules`); a finite prediction set can still make `rmse` overflow: that is `null` as before | unchanged; a **prediction** that is not finite is refused first |
| `wiredata.cluster_metrics` | the inertia and the mean distance of each cluster | the sum of finite squared distances can overflow: `null` as before. A **squared distance of one row** that is not finite | the row is refused (section 4); the sum that overflows from finite rows stays `null` |
| `verbs.predict_rows` (constant float regression) | the stored mean | no: the reader requires the bits of a finite double (`mfile`, `mean is the bits of a finite double`), and the exact mean of finite doubles is finite | unchanged; the door of section 5 covers a handcrafted file |
| `verbs.eval` (constant float regression) | `sse`, `rmse`, the mean | `sse` past the largest double is `ml.metric.undefined` already; `rmse = sqrt(sse / n)` is finite when `sse` is | unchanged |
| `wire.put_coefficients` (`fit`) | the coefficients and the intercept | a coefficient that is not finite is `ml.fit.non-finite` before the file is written, for the linear, logistic and k-means fits **[read]**; this stays as it is, and the gate S1 reaches it | verified, not changed |
| `tools/ml-serve/num.cho` | a double of a request | a number past the range of a double is `serve.number-not-exact` before it reaches the printer | unchanged |

By model kind, for `predict` (and the same predictors under `eval`):

* **linear regression:** `x . coef + intercept` is infinite when a feature is large; this is the reproducer. Refused.
* **logistic regression:** the probability is a squashing of the score into `[0, 1]`, and the label is `probability > 0.5` **[read]**. An infinite score alone gives 0 or 1, but
  a score that is the sum of two infinite terms of opposite sign is NaN, and NaN is neither: a model with coefficients in the thousands (`l2 = 0` on data that is nearly
  separated) and a row of two features of 1e308 reaches it. So the claim "a probability cannot be infinite" holds, and "a probability is always finite" does not; the check
  reads the probability, and the class the model predicts as well (that class is a threshold of the probability, so it is 0 or 1 even for a NaN, and only the probability is named in
  the refusal; gate N4 builds the model and the row).
* **k-means:** the label is an argmin and always an integer. The squared distance `--distance` prints is a sum of squares and overflows for a feature past 1.3e154.
  Refused for `--distance`. Without `--distance` the distances are not computed, the label is printed as before, and the check does not run **[choice]**: nothing printed is
  not a number, and the cost of computing the distances for every `predict` is not paid. `eval` computes them for its metrics, so `eval` refuses (section 4).
* **tree and forest, regression:** a leaf holds the mean of the training targets of its rows, and the reader bounds a target by 1e100 (`models-tree.md` section 4), so a
  prediction is a mean of such values and a mean of means: finite. A training target cannot be infinite (`ml.input.non-finite`). The check runs anyway (one rule for all
  kinds), and a fixture at the boundary of 1e100 shows a finite prediction. **A fixture that makes a tree print `i.nf` does not exist and is not claimed possible.**
* **tree and forest, classification:** a label and a probability that is a ratio of counts: finite.
* **constant models:** finite by construction (the table above).

## 4. Where the check is, and what it covers

**One function, at the one point where the doubles are known.** `wiredata.first_bad_score(task, predictions, probabilities, n, with_distance)`, built on `first_non_finite`,
returns the first row, in file order, whose value is not finite and the column that holds it, or -1. `predict_page` (`tools/ml/wireio.cho`) fills the vector of predictions
and, where the task has them, the vector of probabilities or squared distances for **all the rows of the input**, then calls it **before anything is written**;
`wiredata.metrics_block` (`eval`) calls it after `predict_into` and `proba_into` (and `cluster_metrics` for the squared distances). Both raise the error
`wiredata.non_finite_prediction(errs, row, column)`. A second, later guard is the door of section 5: a cell that the printer would be handed and cannot write makes
`predict_page` answer the same refusal, so a scan that was wrong could not print `i.nf` (the mutant `nf-check-removed` shows the guard alone sees the page and no more). `ml predict`, `ml-predict predict` and `ml-serve` all call `wireio.predict` or `wireio.predict_loaded`, so there is no
second implementation and nothing to keep equal by hand; the server maps the rule to 422 (`status_of`).

**The refusal is all-or-nothing [choice].** The answer of `predict` is a document built after the scan: no cell is written for a refused request, as for every other
refusal. `--format text` the same.

**Paging [choice].** `--limit` and `--from` select the rows that are printed. The check covers **all the rows scored**, not the page: a request for rows 1 to 10 of a file
whose row 500 is not finite is refused, with row 500 in `detail`. Reasons: a caller that pages through a file must not see page 1 succeed and page 50 refuse (the same
file, the same model, an answer that depends on the window); `total_rows` and `next` promise a complete file; and `eval` has no window, so one rule for both. **Cost:** the predictions of all the rows were already computed (the page is a slice of a vector of `total` values, **[read]** `predict_page`), so the check adds one pass over `total`
doubles. **[measured]** `predict --limit 1` of a linear model on a file of 1,000,000 rows of two float columns (`/private/tmp` scratch, Mac M4 Max, load average 19 to 25 from the
mutant suites running beside it, so a ceiling and not a speed): minimum of 5 runs 0.336 and 0.337 s before the change, 0.348 and 0.350 s after, 12 ms for 1,000,000 rows, 3.6 %.

**`eval`.** A prediction, probability or squared distance of any scored row that is not finite is refused with the same rule and the same detail, before any metric.
**[measured, the tree before the change]** `eval` of a regression or a classification with such a row was already a refusal, but the metrics library's (`ml.input.non-finite`,
`detail` `{"metric": "rmse"}` or `"roc_auc"`): the right exit and no row, under a tag that says a *cell* is not finite. `eval` of k-means was the silent case: a squared
distance past the largest double gave `"inertia": null` and `null` mean distances, exit 0. A metric that overflows from finite values (`sse` past the largest double, an
inertia that is a sum of finite squared distances) keeps its documented meaning: `null`, named in `undefined`, or `ml.metric.undefined` for a constant float regression. The
test of `tests/conformance/test_wire_kmeans.py` that fed a squared distance past the largest double to the inertia and expected `null` is re-recorded (section 7), and
`test_nonfinite.py` keeps a case whose rows are finite and whose sum overflows, which is still `null`. With `--split`, `detail.row` is the row of the file.

**`fit`.** A fit whose coefficients become non-finite is `ml.fit.non-finite` already, for the three fitted kinds. Unchanged; `test_nonfinite.FitIsUnchanged` pins it. A `fit --split`
scores the held-out rows through `metrics_block` too, with the new check **off** (`strict = false`): a held-out row whose prediction is not finite is refused by the metrics
library as `ml.input.non-finite` with `detail.metric`, as it was, and a k-means fit keeps its training metrics. The tag is the less exact of the two; changing it is not part of
this change, which leaves `fit` alone **[choice]**, and the finding is in section 7.

**`ml-serve`.** `serve.non-finite-prediction` is removed. A request whose row is not finite is `422` with the body field `rule` equal to `ml.predict.non-finite`, the
`error` object of the predictor (with the row of the request's batch, 1-based, in the order of `rows`), as every other refusal of the predictor with status 422. The
check of the server that looked for a cell that is not a number stays as a guard that cannot be reached, answered with the same rule.

## 5. The door of `flt.put_float` **[choice]**

cancho-table's `flt.put_float` has no finite-only guard and its contract does not say it needs one (the finding is for cancho-table). Here every call goes through
`verbs.put_float_text(heap, buffer, bits)`, which returns the text and whether the double was finite. A caller that has a refusal to raise raises it on `false`
(`wireio.put_cell` does: a cell the scan should have refused cannot be written). A caller printing a metric writes `null`, the convention of every metric with no value.
`scripts/check_nonfinite.py` fails if `flt.put_float` is called anywhere in `tools/ml` outside that function, and a mutant of it (section 6) calls it directly.

## 6. The gates (written before the code; each must be able to fail)

| id | gate | test or script | mutants that must be killed |
|---|---|---|---|
| N1 | the reproducer of the issue: a linear model and a feature of 1e308 through `--float`: `ml predict` exits 8 with `ml.predict.non-finite`, `detail.row` 1, `detail.column` `prediction`, and writes nothing to stdout but the error | `tests/conformance/test_nonfinite.py`, `fixtures.py` case `ml-predict-non-finite` | check removed; wrong exit code; row 0-based |
| N2 | k-means: `--distance` on a row 1e200 away refuses with `distance`; the same file without `--distance` is answered as before; `eval` refuses; an `eval` whose rows are finite and whose sum overflows still prints `null` | `test_nonfinite.py` | `--distance` check removed; eval check removed; the `null` path removed |
| N3 | tree and forest regression at the boundary of 1e100: finite, and a huge feature cannot make the prediction infinite; a target past 1e100 is `ml.input.non-finite` at the fit | `test_nonfinite.py` | none (the gate states that no fixture reaches the rule) |
| N4 | logistic: probabilities and labels over huge and tiny features are finite, 0 or 1, and the boundary (the largest feature that is not refused) is shown | `test_nonfinite.py` | check on the probability removed (if a NaN is reachable) |
| N5 | paging: a non-finite row beyond `--limit`, in the first row, in the last row and in the middle; the same rule and row for `--from`; and `--format text` | `test_nonfinite.py` | check only over the printed window; row of the page, not of the file |
| N6 | the three binaries agree: for every case of N1 to N5, `ml predict`, `ml-predict predict` and `ml-serve` (one request, and the same rows as a batch) give the same rule, the same row for the file (the batch's row is the 1-based position in `rows`), and 422 for the server; the byte-equality corpora of `test_serving.py` and `test_serve_bytes.py` gain the refusal rows | `test_serving.py`, `test_serve_bytes.py`, `test_serve.py` | the server's own check restored under the old tag; `status_of` without the rule |
| N7 | fuzz: huge, subnormal and NaN-producing inputs (`1e308`, `-1e308`, `1e-320`, `5e-324`, sums that cancel to NaN) through all three binaries and every model kind: every answer is a document or a refusal, never `i.nf`, never a trap | `test_nonfinite.py` | none |
| N8 | the registry: the rule is in `refuse.cho`, `design.md` section 8.1, `spec.tmpl`, the schemas, the introspect output of the three binaries, `sspec.cho`, and `serving-http.md` section 5.3, and `status_of` | `tests/core.cho`, `tests/t_registry.cho`, `check_registry.py`, `test_tables.py`, `test_schema.py` | a tag renamed in one place |
| N9 | no call of `flt.put_float` outside the door | `scripts/check_nonfinite.py` | a direct call added |
| G | every fixture and golden file unchanged except where `i.nf`, or the `null` inertia of an overflowing squared distance, was recorded | the existing suite | |

The mutants of the new code are in `scripts/mutants_nonfinite.py`, on a frozen copy of the tree; each is killed or a survivor is listed with its reason in section 7.

## 7. Results

Filled after the gates run.

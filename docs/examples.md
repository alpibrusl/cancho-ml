# Worked examples: classic datasets, `ml` against scikit-learn on the same files

**Status: this document was written before any example was generated or run (the commit that adds it has no CSV, no reference file and no page). Sections 1 to 5 are the gate: what is compared, with which parameters, and the tolerance accepted and why. Section 6 is filled in after the runs and holds only what a script printed.** Tags as in `design.md`: **[choice]** is a decision the maintainer can overturn.

## 1. What is compared, and what is not

For each example the **same two files** go to both programs: `docs/examples/data/<name>_train.csv` and `<name>_test.csv`.

1. `ml fit` on the train file (the command on the page, as run);
2. `ml eval` on the test file, and `ml predict --proba` for a few test rows;
3. scikit-learn fits on the train file and scores the test file (`scripts/examples_data.py`, once; the result is committed in `docs/examples/reference.json` with the scikit-learn, numpy, scipy and Python versions and the command line of the script).

Compared: the **coefficients and the intercept** that `ml fit` prints (in the space the model sees: after the `scale` step for the standardised examples, and the scikit-learn side is fitted on the features standardised by `StandardScaler` fitted on the train file, so it is the same space), and **every test metric `ml eval` prints** (regression: `rmse`, `mae`, `r2`; classification: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `log_loss`). The positive class is label 1 in every classification example. Not compared: speed (nothing here is timed, nothing about speed is claimed), memory, other scikit-learn solvers than the ones named.

Datasets are the copy **bundled with scikit-learn** (`sklearn.datasets.load_*`, no `fetch_*`, no network). The page names the version. They are classic teaching sets; no claim is made about medicine, botany or chemistry.

## 2. The objectives, and the mapping of `--l2` to scikit-learn's parameter

From `docs/models-linear.md` sections 3 and 4 (the model's own definition) and tested below, not assumed:

* **Linear, ridge.** `ml` minimises `||y - Xw - b||^2 + l2 ||w||^2`, the intercept `b` not penalised (not divided by `n`). That is scikit-learn's `Ridge(alpha=l2)`; with `l2 = 0` it is `LinearRegression()`. Mapping: `alpha = l2`.
* **Logistic.** `ml` minimises `(1/n) sum log-loss + (l2/2) ||w||^2`, the bias not penalised. scikit-learn's `LogisticRegression(C=C, penalty="l2")` minimises `C sum log-loss + (1/2) ||w||^2`. Dividing by `C n`: `C = 1 / (l2 * n_train)`. With `l2 = 0` the reference is `penalty=None`. Note that scikit-learn's default is `C = 1`, which is `l2 = 1 / n_train` and not `l2 = 1`; **every reference states `C` or `alpha` explicitly, none relies on a default.**
* **Test of the mapping, before any example** (`scripts/examples_data.py --selftest`, run by the same script that writes the data and refusing to write anything if it fails): on a toy file of 40 rows and 3 standard-normal features written by the script itself, `ml fit` and scikit-learn must agree on the coefficients to the tolerances of section 3, for linear with `l2` in {0, 2}, and for logistic with `l2` in {0.05} and `l2 = 0`. A wrong mapping (for example `C = 1/l2`) must fail this test; the script also runs that wrong mapping and requires that it **does** disagree, so the test can fail.

## 3. The tolerances, and why **[choice]**

| quantity | tolerance | why |
|---|---|---|
| linear / ridge coefficients and intercept | `|d| <= 1e-9 * max(1, |value|)` | both are direct solves (`ml`: normal equations and Cholesky; scikit-learn: SVD least squares, or Cholesky for ridge). The difference is rounding, which grows with the condition number; these designs have condition numbers far below `1e6`, so `1e-9` leaves a margin of several orders and still catches a wrong objective (an `l2` off by a factor of `n` moves the coefficients in the third digit or earlier) |
| linear / ridge test `rmse`, `mae`, `r2` | `|d| <= 1e-9 * max(1, |value|)` | the same solve, then the same arithmetic |
| logistic coefficients and intercept | `|d| <= 1e-6 * max(1, |value|)` | the figure of the existing G-sklearn gate (`docs/models-linear.md` 2 and 6.4). `ml`'s Newton converges quadratically to the optimum to rounding; scikit-learn's L-BFGS stops short of it even at `tol=1e-12`, and 6.4 measured that gap as 1e-9 to 1e-7. The script also fits scikit-learn with its Newton solver (`newton-cholesky`) and records both, so the noise of the reference itself is visible |
| logistic `log_loss`, `roc_auc` | `|d| <= 1e-6` absolute | a smooth function of the coefficients at the level of their tolerance |
| `accuracy`, `precision`, `recall`, `f1` | exactly equal (they are ratios of integer counts) **unless** a test row has a score within `1e-6` of the decision boundary, in which case the row is named and the difference is reported | a label flips only if a score crosses 0; below the coefficient tolerance that can happen only for a score that near 0 |

The iterations of `ml` (no stopping rule) are chosen so that more of them change nothing: each logistic example is fitted with the chosen count and with a larger one, and the largest coefficient change is printed (section 4).

## 4. Method details

* **Split [choice].** `sklearn.model_selection.train_test_split(test_size=0.25, random_state=0)`; **stratified** by the target for the classification examples, not stratified for the regression. The seed was fixed before any model was fitted and is not tuned. `ml`'s own `--split` is not used: both programs get the same two files.
* **Files.** Float columns are written with Python's `repr` (the shortest decimal that reads back to the same double); the target of a classification is the integer 0 or 1; the header is the feature names with spaces replaced by underscores.
* **`ml` flags.** The `--l2` and `--iterations` of each example are fixed here, before the run: breast cancer `--l2 0.01 --iterations 25`; iris (versicolor against virginica) `--l2 0 --iterations 25`; wine (class 0 against the rest) `--l2 0.01 --iterations 25`; diabetes linear `--l2 0`, ridge `--l2 0.05`. Standardisation: `--step 'scale column=* kind=standard'` where the page says so (population variance, `ddof=0`, which is `StandardScaler`'s).
* **A command that does not work is shown.** If the first, obvious command (for example `--l2 0` on data that a hyperplane separates) is refused, or gives a non-finite or a diverging fit, the page shows the refusal, its repair and the fix that worked. It does not start from the fix.
* **Regeneration.** `python3 scripts/examples_data.py` rewrites the CSVs and `reference.json` (needs scikit-learn). `python3 scripts/site.py` runs every `ml` command of the page on the committed CSVs with the built `ml` and writes the output; `scripts/check_docs.sh` fails if the page or the README differs from it, **and** if a committed CSV no longer has the SHA-256 recorded in `reference.json`. The scikit-learn column is read from `reference.json` and labelled with its version; **the check does not re-run scikit-learn.**

## 5. A disagreement is reported, not hidden

If any row of any table is outside its tolerance, the generator marks it **differs** in the table and refuses to pass the docs check until the cause is written next to the table (it is not allowed to be dropped, widened after the fact, or averaged away). If the cause is unknown, the page says "unexplained". The tolerances above are not changed after the results are seen; a different tolerance would be a new section with the old one kept.

## 6. Results

Written after the runs. The table is generated by `scripts/site.py` from the built `ml` and from `reference.json` (so it cannot differ from the page), and the prose is written by hand around it. Everything not in the table is a statement about it, not a new number.

<!-- gen:ex-results -->
| example, run | metric | ml | scikit-learn 1.8.0 | absolute difference | verdict |
|---|---|---:|---:|---:|---|
| Breast cancer, fit | accuracy | 0.951048951048951 | 0.951048951048951 | 0 | within |
| Breast cancer, fit | precision | 0.946236559139785 | 0.946236559139785 | 0 | within |
| Breast cancer, fit | recall | 0.9777777777777777 | 0.9777777777777777 | 0 | within |
| Breast cancer, fit | f1 | 0.9617486338797814 | 0.9617486338797814 | 0 | within |
| Breast cancer, fit | roc_auc | 0.9939203354297694 | 0.9939203354297694 | 0 | within |
| Breast cancer, fit | log_loss | 0.09862906835282033 | 0.09862905898568457 | 9.4e-09 | within |
| Breast cancer, fit | largest coefficient or intercept difference (at texture_error) |  |  | 4.2e-07 | within |
| Breast cancer, fit | first 5 test rows, largest prediction or probability difference | | | 3.1e-09 | within |
| Diabetes, ols | rmse | 56.39290423586858 | 56.39290423586856 | 1.4e-14 | within |
| Diabetes, ols | mae | 45.12056307439618 | 45.12056307439618 | 0 | within |
| Diabetes, ols | r2 | 0.35940880381777107 | 0.3594088038177712 | 1.1e-16 | within |
| Diabetes, ols | largest coefficient or intercept difference (at s1) |  |  | 5.2e-12 | within |
| Diabetes, ols | first 5 test rows, largest prediction or probability difference | | | 1.4e-13 | within |
| Diabetes, ridge | rmse | 56.27420487723617 | 56.27420487723619 | 1.4e-14 | within |
| Diabetes, ridge | mae | 44.9573682109911 | 44.9573682109911 | 0 | within |
| Diabetes, ridge | r2 | 0.3621026796026515 | 0.3621026796026512 | 3.3e-16 | within |
| Diabetes, ridge | largest coefficient or intercept difference (at s1) |  |  | 3.8e-12 | within |
| Diabetes, ridge | first 5 test rows, largest prediction or probability difference | | | 2.0e-13 | within |
| Iris, fit | accuracy | 1.0 | 1.0 | 0 | within |
| Iris, fit | precision | 1.0 | 1.0 | 0 | within |
| Iris, fit | recall | 1.0 | 1.0 | 0 | within |
| Iris, fit | f1 | 1.0 | 1.0 | 0 | within |
| Iris, fit | roc_auc | 1.0 | 1.0 | 0 | within |
| Iris, fit | log_loss | 0.024399970263418697 | 0.024399937949542223 | 3.2e-08 | within |
| Iris, fit | largest coefficient or intercept difference (at intercept) |  |  | 1.1e-04 | **differs** |
| Iris, fit | first 5 test rows, largest prediction or probability difference | | | 1.1e-09 | within |
| Wine, fit | accuracy | 1.0 | 1.0 | 0 | within |
| Wine, fit | precision | 1.0 | 1.0 | 0 | within |
| Wine, fit | recall | 1.0 | 1.0 | 0 | within |
| Wine, fit | f1 | 1.0 | 1.0 | 0 | within |
| Wine, fit | roc_auc | 1.0 | 1.0 | 0 | within |
| Wine, fit | log_loss | 0.05163738931014777 | 0.05163738956541201 | 2.6e-10 | within |
| Wine, fit | largest coefficient or intercept difference (at alcohol) |  |  | 4.7e-08 | within |
| Wine, fit | first 5 test rows, largest prediction or probability difference | | | 9.9e-09 | within |

* Iris, fit: Outside the tolerance of docs/examples.md section 3: largest coefficient or intercept difference (5 values). The cause is the reference solver and it is measured here: scikit-learn's own two solvers differ from each other by up to 2.3e-05 in a coefficient (L-BFGS stops short of the optimum), while ml differs from scikit-learn's Newton solver by at most 6.2e-10, inside the same tolerance. The tolerance was not moved.

Iteration checks (each run against the run it repeats with more iterations; the largest change in a coefficient or the intercept):

* Breast cancer, `--l2 0.01`, 25 iterations against 100: 2.2e-16
* Iris, `--l2 0`, 25 iterations against 100: 8.5e-14
* Wine, `--l2 0.01`, 25 iterations against 100: 2.2e-16
<!-- /gen:ex-results -->

What the runs showed, in the order of the sections above:

1. **The mapping held.** The self-test passed before any file was written (linear at `l2` 0 and 2, logistic at `l2` 0 and 0.05, to far below the tolerances), and the deliberately wrong mapping `C = 1/l2` was caught by it.
2. **Linear and ridge on the diabetes data agree** with `LinearRegression` and `Ridge` far inside the tolerance of section 3, on the coefficients, the three metrics and the first predictions.
3. **Breast cancer and wine (logistic with `--l2 0.01`)**, and the iris log-loss and ROC-AUC, are inside their tolerances; the ratios of integer counts (accuracy, precision, recall, F1) are equal.
4. **One row is outside its tolerance: the iris coefficients against scikit-learn's L-BFGS fit.** Reported as section 5 requires, not hidden and not widened: the generated note under the table gives the cause as measured. scikit-learn's own L-BFGS and Newton (`newton-cholesky`) fits differ from each other by more than the tolerance, and `ml` agrees with the Newton one inside it. The reference fixed in section 3 was the L-BFGS fit; the Newton fit is recorded beside it for exactly this purpose. The reading is that the difference is in the reference solver, supported by a third solver, not proved by it.
5. **Separable data, no penalty (breast cancer and wine, `--l2 0`)** did not fail the way this plan expected. At the time of the runs `ml` returned a fit with large coefficients and no complaint with 25 iterations (every training row right, and on the held-out rows a worse log-loss than the penalised fit's), and with more iterations the coefficients moved or the fit was refused late with `ml.linalg.not-positive-definite`. On separable data there is no optimum to agree about, so no comparison with scikit-learn was made for those runs. Issue 48 closed the gap (`docs/refusals-fit.md`): both runs are refused with `ml.fit.separable`, with the repair as an argv. The repair that works is a penalty (`--l2 0.01`), after which more iterations change nothing.
6. **A column typed per file** (wine): the fit worked, the scoring of the test file was refused with `ml.schema.mismatch`, and `--float` at fit time was the fix. This was found by running the example, not planned.

# Status: what is built, what is not, what was measured

`ml` is early and alpha. It is not for production. This page says what is built, what is not, what has been measured (each figure with its source document and conditions) and what is open. A figure that has no source in this repository is not here.

All the measurements are from one machine, a Mac (arm64, Apple M4 Max, 16 cores, macOS 26.2), with the compiler at `a4572ea`. The machine is shared and was never quiet: every timing is flagged `load >= 2`, and no quiet-machine run exists. Correctness gates do not depend on the load. Nothing has been run on a second machine.

## What is built, and reachable from the command line

The command line is `ml fit | predict | eval | introspect | skill`. It builds three models: `--model constant` (the training mean for a regression, the majority class and its probability for a classification), `--model linear` and `--model logistic`, the last two through a pipeline of fitted steps stored in the same file ([wiring.md](wiring.md), [modules.md](modules.md)).

| item | what it is | source |
|---|---|---|
| `fit`, `predict`, `eval` with `--model constant` | A regression of an integer, decimal or float target (the exact mean; for `eval`, the exact sum of squared errors and an `rmse`), or a binary classification (the majority class, the smaller label on a tie, the probability to six digits; `eval` gives the accuracy). | [cli.md](cli.md) sections 2, 3 |
| `fit`, `predict`, `eval` with `--model linear` and `--model logistic` | Least squares (ridge with `--l2`) and binary logistic regression (Newton, `--iterations`, `--l2`), fitted through the steps of a plan (`--step`, `--plan`: `impute`, `scale`, `encode`, `bucket`) on the training rows; the fitted steps and the estimator are one `CML1` file, and `predict` (with `--proba` for the probability) and `eval` apply the stored steps. `--split train=F --seed N` holds out rows (the metrics library's split); `--positive LABEL` names the positive class; `--threads N` threads `fit`. `eval` gives RMSE, MAE and R2, or accuracy, precision, recall, F1, ROC-AUC and log-loss. An empty cell is a missing value that the plan must impute, else `ml.input.missing-value` with a repair. | [wiring.md](wiring.md) |
| the model module contract | A manifest per model (`lib/models/*/model.toml`), a generator (`scripts/gen_models.py`) for the dispatch, the flag and plan-key tables and the docs table, an admission script (`scripts/admit.py`, checks Q0 to Q11) and mutant modules that it must refuse (`tests/modules`). | [modules.md](modules.md) |
| the CSV reader | Integer, decimal and float columns, RFC 4180 quoting, a byte order mark, CRLF. A float is the nearest double, by cancho-table's exact reader (a pinned package dependency, not a copy). `nan`, `inf` and a non-number are refused with line and column, and so is an empty cell by the constant model; the pipeline models read it as a missing value. | [reader.md](reader.md) |
| declared text columns | `fit --cat NAMES` reads text columns as bytes into a dictionary sorted bytewise (`lib/core/dict.cho`) and an `i64` id tensor with a missing bitmap beside the float matrix (`lib/core/catmat.cho`); `--na-values`; limits on categories, bytes, pool and columns, each refused before allocation; UTF-8 validated. A fit that declares one is refused with `ml.input.categorical-not-encoded`, because no step encodes text; the loader for `predict` with a stored vocabulary is a library API. | [categorical.md](categorical.md) |
| the model file `CML1` | Five sections, tensor records, a SHA-256 trailer. No timestamp, host or path. Written to a temporary name, synced and renamed. | [cli.md](cli.md) section 4, [design.md](design.md) section 4.1 |
| refusals | 75 rule tags in `introspect`, each with an exit code and whether a repair can exist; one JSON line against `ml.v1`. | [cli.md](cli.md), [design.md](design.md) section 8 |
| `--dry-run`, `--overwrite` | `fit --dry-run` answers what it would write and exits 9. Not run under `strace`. | [cli.md](cli.md) section 2 |
| limits | Thirteen flags have a default and a ceiling: `--limit`, `--from`, `--max-rows`, `--max-columns`, `--max-cells`, `--max-line-bytes`, `--max-bytes`, `--log-max-bytes`, the four limits of the categorical columns and `--iterations`. Each is checked before anything it sizes is allocated. | [cli.md](cli.md) section 2.2 |
| `introspect`, `skill` | Generated from the table the parser runs on. | [cli.md](cli.md) section 1 |
| the authority row | `args`, `conc`, `dir_read`, `dir_write`, `err_write`, `file_read`, `file_write`, `fs_read("")`, `heap`, `io_write`; bounded; no foreign function; no `net_out`, `net_in`, `ffi`, `clock` or standard input. `conc` is there for `fit --threads`, with its reason in `tools.toml`. | [`manifests/ml.authority.json`](../manifests/ml.authority.json), [cli.md](cli.md) section 6 |
| `--log` | NDJSON records of counts, flags, rule tags and a hash; no cell, column name or path; bounded. | [cli.md](cli.md) section 5 |
| the reader as a package dependency | Five `[dependencies.*]` (`pow5`, `dec`, `flt`, `facc`, `reader`) on the stores of [alpibrusl/cancho-table#44](https://github.com/alpibrusl/cancho-table/pull/44), pinned to `202b214`; `scripts/check_table_pin.sh` checks the pin. | [reader.md](reader.md) section 1 |

## What is built as a library

Each has gates, an oracle and mutants (see [the evidence page](https://alpibrusl.github.io/cancho-ml/evidence.html)). The transforms, the pipeline, the metrics and splits, linear regression and logistic regression are called by the verbs; the wiring has its own gates ([wiring.md](wiring.md)). k-means, decision trees and random forests are called by no verb: each has a manifest with `wired = false`, and wiring one takes an adapter and the admission script.

| item | what it is | source |
|---|---|---|
| tensor and linear algebra | Rank 1 to 4 descriptors; `dot`, `axpy`, `gemv`, `syrk`, `gemm`, Cholesky and triangular solves, with one written summation order (blocks of 256 terms, 4 lanes in a block, a fixed tree over blocks, Cholesky panels of 32) that does not depend on the thread count; the metrics and the transforms use the same function. | [linalg.md](linalg.md) section 2, [lanes.md](lanes.md) |
| random generator | SplitMix64, derived streams, six registered stream kinds. | [rng.md](rng.md) |
| metrics and splits | Accuracy, precision, recall, F1, ROC-AUC (integers, midranks), RMSE, MAE, R-squared, log-loss; k-fold, train/test, stratified. | [metrics.md](metrics.md) |
| linear and ridge regression | Normal equations, Cholesky. Dependent columns are refused by a relative pivot test (a pivot must exceed 2^-40 of its own diagonal element), not by the sign of a rounded number. | [models-linear.md](models-linear.md) section 3.1 |
| logistic regression, binary | Newton, a fixed number of iterations, optional l2. | [models-linear.md](models-linear.md) |
| k-means (not wired) | Seeded k-means++ or random initialisation, a fixed number of Lloyd iterations. | [models-linear.md](models-linear.md) |
| decision trees and random forests (not wired) | Histogram split search, exact integer Gini, a learned default side for missing values, threaded over trees. Binary classification and regression. | [models-tree.md](models-tree.md) |
| transforms | `impute` (mean, median, constant), `scale` (min-max, standard), `encode` (one-hot, label), `bucket` (width, quantile); fitted on the training rows only, with a record per step; their sums use the package's summation order. | [transforms.md](transforms.md) |
| pipeline | A plan text with one canonical form, fitting a chain of steps, matching input columns by name, and one `CML1` file that holds the plan, the schema, every fitted step and the estimator payload. | [pipeline.md](pipeline.md) |
| the path check of the test tools | `models-check`, `tree-check`, `reader-check`, `linalg-check` and `metrics-check` refuse a path with two dots in a row (`path.dotdot`, exit 2) and trap on no missing operand; they are test drivers, not libraries. | [lanes.md](lanes.md) section 1 |
| the sum order in a library model file | A file of model kind 4 to 7 ends its `meta` section with a `u32 sum_order` (2 is the current order); a reader refuses another value with `ml.model.version`. Kinds 1 to 3, the constant models, have no such field. `ml` writes kinds 4 and 5 and builds no verb for kinds 6 and 7. | [design.md](design.md) section 4.1 |

## What is not built

| item | state |
|---|---|
| k-means, trees and forests in the verbs | each needs an adapter, a manifest with `wired = true` and a passing `scripts/admit.py` ([modules.md](modules.md)) |
| text columns (categories), issue 21 | Declared `fit --cat` columns are read into a bytewise-sorted dictionary and an id tensor, with limits and refusals ([categorical.md](categorical.md)). The steps that read them work in `fit` of `linear` and `logistic`: `encode` (one-hot, label) with a rare-category bucket (`min-count`, the bucket `__other__`), `freq` (count over rows, one rounding), `hash` (SHA-256, first 8 bytes big-endian, modulo B), `unknown` and `missing` rules, and a fitted vocabulary carried by every step with a defined byte layout ([categorical-encoders.md](categorical-encoders.md)). **A fit with a categorical step writes no model file** (the file cannot hold the vocabulary): `--split` fits and scores in one invocation, `--dry-run` reports, otherwise `ml.model.vocabulary-not-stored`. Not built: the string table in the model file, `predict` and `eval` of categorical columns, categorical splits in trees, free text |
| `eval --threads` | accepted and checked; nothing in `eval` is threaded |
| the MCP server | not built; how it would be derived is [cli.md](cli.md) section 9 |
| `ml-serve` (issue #16) | not built |
| multiclass | not built; the format reserves a class count (`ml.label.not-binary` for a third label) |
| boosting, naive Bayes, k-NN | in the plan ([design.md](design.md) section 1), not built |
| imputation in the constant model | `--model constant` refuses an empty cell; the pipeline models take it as a missing value |
| neural networks, GPU | not planned in v1 |
| a second machine | the bytes are untested across machines |
| CI | no workflow in this repository; `scripts/check.sh` needs the network for `cancho install` and `../ml-data` for the tree gate |

## What was measured

Correctness, by script, on this tree. `scripts/check.sh` exits 0: 156 conformance tests pass, none skipped, in about 96 seconds (Mac, load 5 to 7).

| what | figure | source |
|---|---|---|
| mutants | 790 patterns across eleven scripts: 762 killed, 28 survive, each explained, 0 unexplained (core 50, paths 19, linear algebra 134, models 93, metrics 146, trees 100, reader 40, categorical 42, command line 53, wiring 40, pipeline 73); plus 11 pin mutants, 11 deliberately wrong oracles of the linear algebra, 6 mutant modules of the admission script and 27 mutants of the prose lint, all caught | `docs/bench/mutants_*.log`, [the evidence page](https://alpibrusl.github.io/cancho-ml/evidence.html#mutants) |
| command line conformance | 156 tests pass, none skipped (the determinism test of `--threads` runs); 183 fixtures reach all 35 flags, 72 rules and 4 log records; for 161 numeric fixtures the exit status and the SHA-256 of the output and of every written file equal `golden_numeric.json` | [cli.md](cli.md) section 7 |
| the wired models against an oracle | the model file of `linear` and `logistic` equal, byte for byte, to the file a Python implementation writes (60 seeded cases each, with missing cells, constant columns, impute/min-max/standard steps, a split and a positive label), and `predict` and `eval` equal to its doubles; against scikit-learn within bounds fixed first (worst observed relative coefficient error 8.8e-15 for linear against a bound of 1e-9, 1.9e-14 for logistic against 1e-5) | [wiring.md](wiring.md) sections 7, 8 |
| reader against Python `float()` | 1,966,143 accepted cells and 34,380 refusals compared: 0 differences | [reader.md](reader.md) section 6 |
| linear algebra against numpy (the 4-lane order) | 104 shapes, all bits equal; the same bytes at 1, 2, 3, 4, 8 and 16 threads on 15 shapes; the C kernels of `scripts/c_parity` give the same hash on 13 shapes | [lanes.md](lanes.md) sections 2, 5 |
| linear models and k-means | 70 oracle cases bit for bit (including nearly dependent and badly scaled designs); 65 runs at 1 to 16 threads equal; 39 scikit-learn comparisons within the tolerance written beforehand | [models-linear.md](models-linear.md) section 3.1, 6.2, 6.4 |
| worked examples | four classic datasets (breast cancer Wisconsin, diabetes, iris restricted to two species, wine, as bundled with scikit-learn 1.8.0) solved by `ml` and compared with scikit-learn on the same train and test files, with the mapping of `--l2` tested and the tolerances written before any run: every comparison is inside its tolerance except one, the iris coefficients against scikit-learn's L-BFGS fit, whose cause is measured (scikit-learn's own two solvers differ by more than the tolerance and `ml` agrees with its Newton solver). Two failures are shown: a penalty-free logistic fit on separable data (succeeds with large coefficients, then is refused with more iterations) and a column typed differently by two files. scikit-learn is run by `scripts/examples_data.py` and not by the docs check | [examples.md](examples.md), [the examples page](https://alpibrusl.github.io/cancho-ml/examples.html) |
| the relative pivot test | rounding noise of exactly dependent columns, at most 7.9e-16 of the diagonal in 960 fits; a standardised pair with correlation 1 - 1e-6 has 2.0e-06; the tolerance is 9.1e-13 | [models-linear.md](models-linear.md) section 3.1, `scripts/pivot_evidence.py` |
| transforms and pipeline | 29 + 2 unit and fuzz tests of the transforms, 14 + 5 of the pipeline; the Python oracle agrees bit for bit on 2,500 transform cases, 600 chains, 300 schema alignments, 150 files with steps and 186 sums; scikit-learn within stated bounds | [transforms.md](transforms.md) section 6, [pipeline.md](pipeline.md) section 6; `scripts/gate_pipe.sh` prints the counts |
| categorical columns | the vocabulary, ids, missing set and refusals of `reader-check --mode cat` equal to a Python oracle on 44 adversarial and 1,500 seeded files (0 differences); row order (750 runs), thread count (90 runs), limits (75 runs), UTF-8 (8,901 runs), round trip (459 runs) and 3,618 no-trap runs: 0 differences | [categorical.md](categorical.md) section 9, `docs/bench/gate_cat.log` |
| the path check | 69 fixtures, 595 sweep vectors and 2,000 fuzz vectors over the five test drivers: 0 failures | [lanes.md](lanes.md) section 1 |
| metrics and splits | 2,978 split cases bit equal; 3,066 ratios, 109 AUC and 1,033 sum cases bit equal; log-loss terms 13,407 of 13,896 bit equal, the rest within 1.00 ulp | [metrics.md](metrics.md) section 5.1 |
| trees and forests | 20 (configuration, size) pairs identical to the oracle; 11 rows by 7 thread counts equal; 21 unit tests | [models-tree.md](models-tree.md) section 10 |
| registries | 7 model kinds, 6 stream kinds, 64 refusal codes in 4 bands; tables, sources and documents agree | `scripts/check_registry.py` |

Speed, every cell at `load >= 2` and none a claim. **The linear-algebra and model figures in this table were measured with the summation order of one chain per block, not the 4-lane order the libraries use today; what the lane order costs or buys in scalar cancho is not measured** ([lanes.md](lanes.md) section 6). The tables with their conditions are [the benchmarks page](https://alpibrusl.github.io/cancho-ml/benchmarks.html) and the raw output is in `docs/bench/`:

| what | figure | source |
|---|---|---|
| models against numpy's same algorithm, 1,000,000 rows, 1 thread | logistic 236 ms against 170; linear 32.0 against 16.1; k-means 306 against 1,148 (cancho slower on the first two) | [models-linear.md](models-linear.md) section 6.5, `bench/models-bench.txt` |
| models against scikit-learn | logistic 126 ms for the default call (cancho 1.87 times slower) and 223 for `tol=1e-12` (1.06); k-means 330 ms (0.93) | the same |
| thread scaling, logistic regression, 1,000,000 rows | 240 ms at 1 thread, 59 ms at 16; linear regression 34 and 24 ms | [models-linear.md](models-linear.md) section 6.7 |
| dense linear algebra against numpy and Accelerate, 1 thread | 8 of 29 cells match, 21 do not (1.05 to 14.86 times slower); square gemm 13.8 to 16.5 times slower | [linalg.md](linalg.md) section 5.2, `bench/gate-1thread.jsonl` |
| AUC, 1,000,000 scores | 24.0 ms against 146.6 for `sklearn.roc_auc_score` and 86.2 for a thin numpy version | [metrics.md](metrics.md) section 5.4 |
| log-loss, 1,000,000 probabilities | 16.0 ms; 3.0 times slower than a thin numpy version, 3.2 times faster than `sklearn.log_loss` (51.9 ms) | the same |
| forest, 10 trees, 100,000 rows | fit 0.0516 s at 1 thread and 0.0100 s at 16. scikit-learn's exact-split forest: 1.06 s and 0.132 s. **Different algorithms; not a speed claim.** | [models-tree.md](models-tree.md) section 10.9 |
| the reader, float17 file, pass 1 | 17.1 M cells/s against cancho-table's own 19.6 M/s (13% slower) | [reader.md](reader.md) section 6, `bench/bench_reader.log` |
| the command line | not timed | |

## Open items

| item | state | source |
|---|---|---|
| **G-ref, regression forest R1** | **red.** RMSE 2.784408 against scikit-learn's 3.029622, 8.1% apart; the tolerance is 3%. Most of the gap is attributed to the algorithms (quantile bins against exact thresholds) and 3.1% (with missing cells) to nothing. The tolerance is kept. The maintainer decides: keep the rule, compare on bin indices, or drop the regression comparison. Reproduce: `python3 scripts/gate_sklearn.py --big; echo $?` prints the R1 row and 1. | [models-tree.md](models-tree.md) section 10.8 |
| `check.sh` needs `../ml-data` | and the network for `cancho install` | [README](../README.md) |
| `rmse` is not exact | three IEEE steps after the exact sum; an exact integer square root is about 60 lines and is not decided | [cli.md](cli.md) section 10 |
| the pivot test is a threshold, not a rank computation | a column explained by the preceding columns to within 2^-40 of its own sum of squares is refused; noise above that was not seen in 960 fits (n up to 20,000) and was not measured for larger n or d in the hundreds | [models-linear.md](models-linear.md) section 3.1 |
| the sum-order field has one writer | only `lib/pipeline/file.cho` writes it (and a Python writer in the conformance tests); `ml` writes kinds 4 and 5 through it, no verb writes kinds 6 and 7 | [design.md](design.md) section 4.1 |
| the contract package's text is wrong for `ml` in six places | `introspect.guarantees.concurrency` says "read-only: it never writes", and `integers_only` is true of the output and not of the code, among others; upstream issues are not filed | [cli.md](cli.md) section 8, [reader.md](reader.md) section 7 |
| `dry_run` and atomicity | "no mutating system call" under `--dry-run` was not run under `strace`; two writers racing on one output can fail one of them, and this was not tested | [cli.md](cli.md) sections 2, 8 |
| the tree library is limited to 2^20 rows | the exact Gini comparison; `ml`'s cap is 10,000,000 rows, so a verb needs a sampled-fit repair or a wider comparison | [models-tree.md](models-tree.md) section 10.11 |
| empty clusters keep their centroid | scikit-learn relocates the farthest point | [models-linear.md](models-linear.md) section 6.8 |
| Newton has no damping | nearly separable data can fail with `ml.fit.non-finite` or with a Hessian pivot below the relative tolerance (`ml.linalg.not-positive-definite`); whether the CLI should refuse unscaled input is not decided | [models-linear.md](models-linear.md) section 6.8 |
| zero division in a metric is a refusal | `ml.metric.undefined`, where scikit-learn returns 0.0 with a warning; a `--zero-division` flag may be wanted | [metrics.md](metrics.md) section 7 |
| `-0` reads as `0.0` | cancho-table's rule | [reader.md](reader.md) section 7 |
| two compiler pins | cancho-ml is on `a4572ea`, cancho-table's stores are published by `e47d647`; they install and build under `a4572ea` | [reader.md](reader.md) section 7 |
| not measured | the benchmarks through the command line; a model fit's memory; the 1,000,000-row identity and timing runs of the tree library beyond the defaults | [models-tree.md](models-tree.md) section 10 |

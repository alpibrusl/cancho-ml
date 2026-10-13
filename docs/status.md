# Status: what is built, what is not, what was measured

`ml` is early and alpha. It is not for production. This page says what is built, what is left, what has been measured (each figure with its source document and conditions) and what is open. Where a figure has no source in this repository it is not here.

All the measurements are from one machine, a Mac (arm64, Apple M4 Max, 16 cores, macOS 26.2), with the compiler at `a4572ea`. The machine was shared and never quiet: every timing is flagged `load >= 2`, and **a serial rerun on a quiet machine is pending**. Correctness gates do not depend on the load. Nothing has been run on a second machine.

## What is built, and reachable from the command line

The command line is `ml fit | predict | eval | introspect | skill`. The only model it can build is `--model constant`: the training mean for a regression, the majority class and its probability for a classification.

| item | what it is | source |
|---|---|---|
| `fit`, `predict`, `eval` with `--model constant` | A regression of an integer, decimal or float target (the exact mean; for `eval`, the exact sum of squared errors and an `rmse`), or a binary classification (the majority class, the smaller label on a tie, the probability to six digits; `eval` gives the accuracy). | [cli.md](cli.md) sections 2, 3 |
| the CSV reader | Integer, decimal and float columns, RFC 4180 quoting, a byte order mark, CRLF. A float is the nearest double, by cancho-table's exact reader (a pinned clone, not a copy). `nan`, `inf`, an empty cell and a non-number are refused with line and column. | [reader.md](reader.md) |
| the model file `CML1` | Five sections, tensor records, a SHA-256 trailer. No timestamp, host or path. Written to a temporary name, synced and renamed. | [cli.md](cli.md) section 4, [design.md](design.md) section 4.1 |
| refusals | 49 rule tags in `introspect`, each with an exit code and whether a repair can exist; one JSON line against `ml.v1`. | [cli.md](cli.md), [design.md](design.md) section 8 |
| `--dry-run`, `--overwrite` | `fit --dry-run` answers what it would write and exits 9. Not run under `strace`. | [cli.md](cli.md) section 2 |
| limits | Eight flags with a default and a ceiling, checked before allocation. | [cli.md](cli.md) section 2.2 |
| `introspect`, `skill` | Generated from the table the parser runs on. | [cli.md](cli.md) section 1 |
| the authority row | `args`, `dir_read`, `dir_write`, `err_write`, `file_read`, `file_write`, `fs_read("")`, `heap`, `io_write`; bounded; no foreign function; no `net_out`, `net_in`, `ffi`, `clock`, standard input or `conc`. | [`manifests/ml.authority.json`](../manifests/ml.authority.json), [cli.md](cli.md) section 6 |
| `--log` | NDJSON records of counts, flags, rule tags and a hash; no cell, column name or path; bounded. | [cli.md](cli.md) section 5 |

## What is built as a library and not reachable

Each has gates, an oracle and mutants (see [the evidence page](https://alpibrusl.github.io/cancho-ml/evidence.html)). No verb calls any of them. Wiring a verb means a model kind in `tools/ml/mfile.cho` with a parser in the conformance tests, the library's rule tags moved from `pending()` to the tables of `introspect`, schemas and fixtures regenerated, and `conc` in the authority row when `--threads` arrives ([integration.md](integration.md) section 6).

| item | what it is | source |
|---|---|---|
| tensor and linear algebra | Rank 1 to 4 descriptors; `dot`, `axpy`, `gemv`, `syrk`, `gemm`, Cholesky and triangular solves, with a written summation order that does not depend on the thread count. | [linalg.md](linalg.md) |
| random generator | SplitMix64, derived streams, six registered stream kinds. | [rng.md](rng.md) |
| metrics and splits | Accuracy, precision, recall, F1, ROC-AUC (integers, midranks), RMSE, MAE, R-squared, log-loss; k-fold, train/test, stratified. | [metrics.md](metrics.md) |
| linear and ridge regression | Normal equations, Cholesky. | [models-linear.md](models-linear.md) |
| logistic regression, binary | Newton, a fixed number of iterations, optional l2. | [models-linear.md](models-linear.md) |
| k-means | Seeded k-means++ or random initialisation, a fixed number of Lloyd iterations. | [models-linear.md](models-linear.md) |
| decision trees and random forests | Histogram split search, exact integer Gini, a learned default side for missing values, threaded over trees. Binary classification and regression. | [models-tree.md](models-tree.md) |

## What is left

| item | state |
|---|---|
| transforms and pipeline (`impute`, `scale`, `encode`) | being built; the plan is a design ([design.md](design.md) sections 4.2, 5) |
| model wiring (the verbs calling the models above) | next |
| `--threads` in `ml` | not built; the libraries are threaded |
| the MCP server | not started; how it would be derived is [cli.md](cli.md) section 9 |
| `ml-serve` (issue #16) | not started |
| multiclass | not built; the format reserves a class count |
| boosting, naive Bayes, k-NN | in the plan ([design.md](design.md) section 1), not built |
| imputation of a missing cell | not built; `fit` refuses an empty cell |
| neural networks, GPU | not planned in v1 |
| the reader as a package dependency | waits for [alpibrusl/cancho-table#44](https://github.com/alpibrusl/cancho-table/pull/44) (draft); until then a local clone at `../cancho-table-main` ([reader.md](reader.md) section 1) |
| a second machine | not done; the bytes are untested across machines |
| CI | there is no workflow in this repository; `scripts/check.sh` is what one would run and it needs the sibling clones `../cancho-table-main` and `../ml-data` |

## What was measured

Correctness, by script, on the merged tree ([integration.md](integration.md) section 3), and the conformance suite rerun on this commit (101 tests, OK, 1 skipped, 38 seconds, Mac):

| what | figure | source |
|---|---|---|
| mutants | 534 across seven scripts; 510 killed; 24 survive, each explained; 0 unexplained | [integration.md](integration.md) section 3 |
| command line conformance | 101 tests pass, 1 skipped (the `--threads` branch of the determinism test: there is no such flag) | [integration.md](integration.md) section 3 |
| reader against Python `float()` | 1,966,143 accepted cells and 34,380 refusals compared: 0 differences | [reader.md](reader.md) section 6 |
| linear algebra against numpy (same order) | 46 cases, all bits equal; the same bytes at 1, 2, 3, 4, 8 and 16 threads | [linalg.md](linalg.md) section 5.1, 5.4 |
| linear models and k-means | 55 oracle cases bit for bit; 65 runs at 1 to 16 threads equal; 39 scikit-learn comparisons within the tolerance written beforehand | [models-linear.md](models-linear.md) section 6.2, 6.4 |
| metrics and splits | 2,978 split cases bit equal; 3,066 ratios, 109 AUC and 1,033 sum cases bit equal; log-loss terms 13,407 of 13,896 bit equal, the rest within 1.00 ulp | [metrics.md](metrics.md) section 5.1 |
| trees and forests | 20 (configuration, size) pairs identical to the oracle; 11 rows by 7 thread counts equal; 21 unit tests | [models-tree.md](models-tree.md) section 10 |
| registries | 7 model kinds, 6 stream kinds, 34 refusal codes in 3 bands; tables, sources and documents agree | [integration.md](integration.md) section 4 |

Speed, every cell at `load >= 2` and none a claim. The tables with their conditions are [the benchmarks page](https://alpibrusl.github.io/cancho-ml/benchmarks.html) and the raw output is in `docs/bench/`:

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

From [integration.md](integration.md) section 7 and the stage documents:

| item | state | source |
|---|---|---|
| **G-ref, regression forest R1** | **red.** RMSE 2.784408 against scikit-learn's 3.029622, 8.1% apart; the tolerance was 3%. The stage attributed most of the gap to the algorithms (quantile bins against exact thresholds) and 3.1% (with missing cells) to nothing. The tolerance was not moved. The maintainer decides: keep the rule, compare on bin indices, or drop the regression comparison. Reproduce: `python3 scripts/gate_sklearn.py --big; echo $?` prints the R1 row and 1. | [models-tree.md](models-tree.md) section 10.8 |
| `models-check` traps without a message on a path containing `..` | a test driver, not a library; `ml` refuses such paths itself. Cause not investigated. Reproduce: [integration.md](integration.md) section 7 | [integration.md](integration.md) section 7 |
| `check.sh` needs `../ml-data` | in addition to `../cancho-table-main` | [integration.md](integration.md) section 7 |
| `rmse` is not exact | three IEEE steps after the exact sum; an exact integer square root is about 60 lines and is not decided | [cli.md](cli.md) section 10 |
| a different summation order in metrics and linalg | metrics use 4 lanes in a block of 256 terms, linalg one chain; `sums.lanes()` is the metrics' own constant. The lane count is the maintainer's decision, and it would change every dot-shaped reduction | [integration.md](integration.md) section 5, [linalg.md](linalg.md) section 5.7 |
| the contract package's text is wrong for `ml` in six places | `introspect.guarantees.concurrency` says "read-only: it never writes", and `integers_only` is true of the output and not of the code, among others; upstream issues not filed | [cli.md](cli.md) section 8, [reader.md](reader.md) section 7 |
| `dry_run` and atomicity | "no mutating system call" under `--dry-run` was not run under `strace`; two writers racing on one output can fail one of them, and was not tested | [cli.md](cli.md) sections 2, 8 |
| the tree library is limited to 2^20 rows | the exact Gini comparison; `ml`'s cap is 10,000,000 rows, so a verb needs a sampled-fit repair or a wider comparison | [models-tree.md](models-tree.md) section 10.11 |
| empty clusters keep their centroid | scikit-learn relocates the farthest point | [models-linear.md](models-linear.md) section 6.8 |
| Newton has no damping | nearly separable data can fail with `ml.fit.non-finite`; whether the CLI should refuse unscaled input is not decided | [models-linear.md](models-linear.md) section 6.8 |
| zero division in a metric is a refusal | `ml.metric.undefined`, where scikit-learn returns 0.0 with a warning; a `--zero-division` flag may be wanted | [metrics.md](metrics.md) section 7 |
| `-0` reads as `0.0` | cancho-table's rule | [reader.md](reader.md) section 7 |
| two compiler pins | cancho-ml is on `a4572ea`, cancho-table on `e47d647`; the numeric modules build under both | [reader.md](reader.md) section 7 |
| not run at integration | the benchmarks, a package-mode build of `models-check`, and the 1,000,000-row identity and timing runs of the tree stage | [integration.md](integration.md) section 3 |

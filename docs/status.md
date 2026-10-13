# Status: what is built, what is not, what was measured

`ml` is early and alpha. It is not for production. This page says what is built, what is left, what has been measured (each figure with its source document and conditions) and what is open. Where a figure has no source in this repository it is not here.

All the measurements are from one machine, a Mac (arm64, Apple M4 Max, 16 cores, macOS 26.2), with the compiler at `a4572ea`. The machine was shared and never quiet: every timing is flagged `load >= 2`, and **a serial rerun on a quiet machine is pending**. Correctness gates do not depend on the load. Nothing has been run on a second machine.

## What is built, and reachable from the command line

The command line is `ml fit | predict | eval | introspect | skill`. The only model it can build is `--model constant`: the training mean for a regression, the majority class and its probability for a classification.

| item | what it is | source |
|---|---|---|
| `fit`, `predict`, `eval` with `--model constant` | A regression of an integer, decimal or float target (the exact mean; for `eval`, the exact sum of squared errors and an `rmse`), or a binary classification (the majority class, the smaller label on a tie, the probability to six digits; `eval` gives the accuracy). | [cli.md](cli.md) sections 2, 3 |
| the CSV reader | Integer, decimal and float columns, RFC 4180 quoting, a byte order mark, CRLF. A float is the nearest double, by cancho-table's exact reader (a pinned package dependency, not a copy). `nan`, `inf`, an empty cell and a non-number are refused with line and column. | [reader.md](reader.md) |
| the model file `CML1` | Five sections, tensor records, a SHA-256 trailer. No timestamp, host or path. Written to a temporary name, synced and renamed. | [cli.md](cli.md) section 4, [design.md](design.md) section 4.1 |
| refusals | 49 rule tags in `introspect`, each with an exit code and whether a repair can exist; one JSON line against `ml.v1`. | [cli.md](cli.md), [design.md](design.md) section 8 |
| `--dry-run`, `--overwrite` | `fit --dry-run` answers what it would write and exits 9. Not run under `strace`. | [cli.md](cli.md) section 2 |
| limits | Eight flags with a default and a ceiling, checked before allocation. | [cli.md](cli.md) section 2.2 |
| `introspect`, `skill` | Generated from the table the parser runs on. | [cli.md](cli.md) section 1 |
| the authority row | `args`, `dir_read`, `dir_write`, `err_write`, `file_read`, `file_write`, `fs_read("")`, `heap`, `io_write`; bounded; no foreign function; no `net_out`, `net_in`, `ffi`, `clock`, standard input or `conc`. | [`manifests/ml.authority.json`](../manifests/ml.authority.json), [cli.md](cli.md) section 6 |
| `--log` | NDJSON records of counts, flags, rule tags and a hash; no cell, column name or path; bounded. | [cli.md](cli.md) section 5 |

## What is built as a library and not reachable

Each has gates, an oracle and mutants (see [the evidence page](https://alpibrusl.github.io/cancho-ml/evidence.html)). No verb calls any of them: the transforms and the pipeline are not wired into `fit`, `predict` or `eval` any more than the models are. Wiring a verb means a model kind in `tools/ml/mfile.cho` with a parser in the conformance tests, the library's rule tags moved from `pending()` to the tables of `introspect`, schemas and fixtures regenerated, and `conc` in the authority row when `--threads` arrives ([integration.md](integration.md) section 6).

| item | what it is | source |
|---|---|---|
| tensor and linear algebra | Rank 1 to 4 descriptors; `dot`, `axpy`, `gemv`, `syrk`, `gemm`, Cholesky and triangular solves, with one written summation order (blocks of 256 terms, 4 lanes in a block, a fixed tree over blocks, Cholesky panels of 32) that does not depend on the thread count; the metrics and the transforms use the same function. | [linalg.md](linalg.md) section 2, [lanes.md](lanes.md) |
| random generator | SplitMix64, derived streams, six registered stream kinds. | [rng.md](rng.md) |
| metrics and splits | Accuracy, precision, recall, F1, ROC-AUC (integers, midranks), RMSE, MAE, R-squared, log-loss; k-fold, train/test, stratified. | [metrics.md](metrics.md) |
| linear and ridge regression | Normal equations, Cholesky. Dependent columns are refused by a relative pivot test (a pivot must exceed 2^-40 of its own diagonal element), not by the sign of a rounded number. | [models-linear.md](models-linear.md) section 3.1 |
| logistic regression, binary | Newton, a fixed number of iterations, optional l2. | [models-linear.md](models-linear.md) |
| k-means | Seeded k-means++ or random initialisation, a fixed number of Lloyd iterations. | [models-linear.md](models-linear.md) |
| decision trees and random forests | Histogram split search, exact integer Gini, a learned default side for missing values, threaded over trees. Binary classification and regression. | [models-tree.md](models-tree.md) |
| transforms | `impute` (mean, median, constant), `scale` (min-max, standard), `encode` (one-hot, label), `bucket` (width, quantile); fitted on the training rows only, with a record per step; their sums use the package's summation order. | [transforms.md](transforms.md) |
| pipeline | A plan text with one canonical form, fitting a chain of steps, matching input columns by name, and one `CML1` file that holds the plan, the schema, every fitted step and the estimator payload. | [pipeline.md](pipeline.md) |
| the path check of the test tools | `models-check`, `tree-check`, `reader-check` and `metrics-check` refuse a path with two dots in a row (`path.dotdot`, exit 2) and no longer trap on a missing operand; they are test drivers, not libraries. | [lanes.md](lanes.md) section 1 |
| the sum order in a library model file | A file of model kind 4 to 7 ends its `meta` section with a `u32 sum_order` (2 is the current order); a reader refuses another value with `ml.model.version`. Kinds 1 to 3, the constant models `ml` writes, have no such field and keep their bytes. `ml` reads the field but does not build kinds 4 to 7. | [design.md](design.md) section 4.1, [integration.md](integration.md) section 8.3 |

## What is left

| item | state |
|---|---|
| wiring the transforms, the pipeline and the models into the verbs (a plan flag, model kinds in `mfile.cho`, the tags moved to `introspect`) | next; all are libraries today, see above |
| `--threads` in `ml` | not built; the libraries are threaded |
| the MCP server | not started; how it would be derived is [cli.md](cli.md) section 9 |
| `ml-serve` (issue #16) | not started |
| multiclass | not built; the format reserves a class count |
| boosting, naive Bayes, k-NN | in the plan ([design.md](design.md) section 1), not built |
| imputation of a missing cell in `fit` | the `impute` transform exists as a library; `fit` still refuses an empty cell |
| neural networks, GPU | not planned in v1 |
| the reader as a package dependency | done: six `[dependencies.*]` on the stores of [alpibrusl/cancho-table#44](https://github.com/alpibrusl/cancho-table/pull/44) (merged), pinned to `202b214` ([reader.md](reader.md) section 1) |
| a second machine | not done; the bytes are untested across machines |
| CI | there is no workflow in this repository; `scripts/check.sh` is what one would run and it needs the network for `cancho install` and the sibling clone `../ml-data` |

## What was measured

Correctness, by script, on the merged tree ([integration.md](integration.md) section 3), and the conformance suite rerun on this commit (105 tests, OK, 1 skipped, 49 seconds, Mac, load average about 8 to 10):

| what | figure | source |
|---|---|---|
| mutants | 702 patterns across nine scripts (670 at the merge); 677 killed; 25 survive, each explained; 0 unexplained; plus 11 deliberately wrong oracles of the linear algebra, all caught | [integration.md](integration.md) section 8.4 |
| command line conformance | 105 tests pass, 1 skipped (the `--threads` branch of the determinism test: there is no such flag) | [cli.md](cli.md) section 7 |
| reader against Python `float()` | 1,966,143 accepted cells and 34,380 refusals compared: 0 differences | [reader.md](reader.md) section 6 |
| linear algebra against numpy (the 4-lane order) | 104 shapes, all bits equal; the same bytes at 1, 2, 3, 4, 8 and 16 threads; three implementations of the order (Cancho, numpy, C) agree bit for bit | [lanes.md](lanes.md) sections 2, 5 |
| linear models and k-means | 70 oracle cases bit for bit (including nearly dependent and badly scaled designs); 65 runs at 1 to 16 threads equal; 39 scikit-learn comparisons within the tolerance written beforehand | [models-linear.md](models-linear.md) section 3.1, 6.2, 6.4 |
| the relative pivot test | rounding noise of exactly dependent columns, at most 7.9e-16 of the diagonal in 960 fits; a standardised pair with correlation 0.999999 has 2.0e-06; the tolerance is 9.1e-13 | [models-linear.md](models-linear.md) section 3.1, `scripts/pivot_evidence.py` |
| transforms and pipeline | 29 + 2 unit and fuzz tests of the transforms, 14 + 5 of the pipeline; the Python oracle agrees bit for bit on 2,500 transform cases, 600 chains, 300 schema alignments, 150 files with steps and 186 sums; scikit-learn within stated bounds | [transforms.md](transforms.md) section 6, [pipeline.md](pipeline.md) section 6; the test counts are printed by `scripts/gate_pipe.sh` |
| the path check | 69 fixtures, 595 sweep vectors and 2,000 fuzz vectors over five test tools: 0 failures | [lanes.md](lanes.md) section 1 |
| metrics and splits | 2,978 split cases bit equal; 3,066 ratios, 109 AUC and 1,033 sum cases bit equal; log-loss terms 13,407 of 13,896 bit equal, the rest within 1.00 ulp | [metrics.md](metrics.md) section 5.1 |
| trees and forests | 20 (configuration, size) pairs identical to the oracle; 11 rows by 7 thread counts equal; 21 unit tests | [models-tree.md](models-tree.md) section 10 |
| registries | 7 model kinds, 6 stream kinds, 59 refusal codes in 4 bands; tables, sources and documents agree | [integration.md](integration.md) section 4, `scripts/check_registry.py` |

Speed, every cell at `load >= 2` and none a claim. **Every figure in this table was measured before the 4-lane summation order replaced the previous one, and was not repeated** ([lanes.md](lanes.md) section 6). The tables with their conditions are [the benchmarks page](https://alpibrusl.github.io/cancho-ml/benchmarks.html) and the raw output is in `docs/bench/`:

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
| `check.sh` needs `../ml-data` | and the network for `cancho install` | [integration.md](integration.md) section 7 |
| `rmse` is not exact | three IEEE steps after the exact sum; an exact integer square root is about 60 lines and is not decided | [cli.md](cli.md) section 10 |
| **the speed tables predate the 4-lane order** | every timing in this page and in `docs/bench/` was taken with the previous summation order (one chain per block in linalg); what the lane order costs or buys in scalar cancho was not measured. The benchmarks were not rerun | [lanes.md](lanes.md) section 6 |
| the pivot test is a threshold, not a rank computation | a column explained by the earlier ones to within 2^-40 of its own sum of squares is refused; noise above that was not seen in 960 fits (n up to 20,000) and was not measured for larger n or d in the hundreds | [models-linear.md](models-linear.md) section 3.1 |
| the sum-order field has one writer | only `lib/pipeline/file.cho` writes it (and a Python writer in the conformance tests); no verb writes a model of kind 4 to 7 yet | [design.md](design.md) section 4.1 |
| the contract package's text is wrong for `ml` in six places | `introspect.guarantees.concurrency` says "read-only: it never writes", and `integers_only` is true of the output and not of the code, among others; upstream issues not filed | [cli.md](cli.md) section 8, [reader.md](reader.md) section 7 |
| `dry_run` and atomicity | "no mutating system call" under `--dry-run` was not run under `strace`; two writers racing on one output can fail one of them, and was not tested | [cli.md](cli.md) sections 2, 8 |
| the tree library is limited to 2^20 rows | the exact Gini comparison; `ml`'s cap is 10,000,000 rows, so a verb needs a sampled-fit repair or a wider comparison | [models-tree.md](models-tree.md) section 10.11 |
| empty clusters keep their centroid | scikit-learn relocates the farthest point | [models-linear.md](models-linear.md) section 6.8 |
| Newton has no damping | nearly separable data can fail with `ml.fit.non-finite` or, now, with a Hessian pivot below the relative tolerance (`ml.linalg.not-positive-definite`); whether the CLI should refuse unscaled input is not decided | [models-linear.md](models-linear.md) section 6.8 |
| zero division in a metric is a refusal | `ml.metric.undefined`, where scikit-learn returns 0.0 with a warning; a `--zero-division` flag may be wanted | [metrics.md](metrics.md) section 7 |
| `-0` reads as `0.0` | cancho-table's rule | [reader.md](reader.md) section 7 |
| two compiler pins | cancho-ml is on `a4572ea`, cancho-table on `e47d647`; the numeric modules build under both | [reader.md](reader.md) section 7 |
| not run at integration | the benchmarks, a package-mode build of `models-check`, and the 1,000,000-row identity and timing runs of the tree stage | [integration.md](integration.md) section 3 |

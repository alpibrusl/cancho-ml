# Integration of the six v1 stage branches (`v1-int`)

**Status: what was done when core + linalg, the CLI slice, the reader, metrics + splits, linear/logistic/k-means and trees + forests
were merged into one tree, and what every gate says about the merged tree.** No feature was added and no gate was loosened. Every number
below was printed by the script named next to it, in this session, on one machine: Mac arm64 (Darwin 25.2.0), cancho `a4572ea`, Python
3.14.5, numpy 2.4.6, scikit-learn 1.8.0; the machine was loaded by the mutant runs (up to a load average of 36) when the mutants ran, which
changes no verdict but is the reason no time is quoted. Benchmarks were not run.

## 1. What conflicted

**Text.** Git merged the six branches with one text conflict, in `docs/design.md` (the linear stage and the tree stage both appended a
section at the end: kept both, now sections 14 and 15). Nothing else conflicted in text.

**Meaning, found by reading the stages against each other.** These merged without a conflict marker and were wrong:

| what | the collision | resolved |
|---|---|---|
| model kind | 3 was `tools/ml/mfile.cho`'s constant regression of a float target, the linear stage's linear regression (`lib/models/record.cho`, `tests/p_models.cho`, `docs/models-linear.md`) and the tree stage's forest (`docs/models-tree.md`); `design.md` section 15 said the tree was 6 | one table, section 4: 1, 2, 3 as shipped; linear 4, logistic 5, k-means 6, tree 7 |
| refusal code | the metrics library numbered its tags 1 to 11, `ml.refuse` has 1 to 15 with other meanings (metrics 4 = `single-class`, refuse 4 = `input-empty`); the tree library has 101 to 108 | bands, one per library: 1-99 `ml.refuse`, 101-199 `tree.deny`, 201-299 `metrics.rules` (metrics now 201 to 211) |
| RNG stream kind | metrics used kinds 1, 2, 3 for its three splits; the tree uses 1 (bootstrap) and 2 (node features); k-means uses 3. The same seed in a split and in a forest or in k-means would give the two the same draws | tree 1, 2 and k-means 3 kept; the splits are 4, 5, 6 |
| build lists | `lib/core/matrix.cho` (reader stage) imports `linalg.tensor`; the metrics gate and the tree gate build `lib/core/*.cho` with explicit lists that had no `tensor.cho` | `lib/linalg/tensor.cho` added to the lists (`scripts/gate_metrics.sh`, `scripts/gate_tree.py` twice) |
| not in CI | `scripts/check.sh` ran neither the metrics gate nor the tree gate nor any registry check | added (section 3) |
| rule tags | the libraries' tags that `ml` cannot reach were in no table of `tools/ml/spec.cho`, and `design.md` section 8 had rows for few of them | `spec.cho` `pending()`, `design.md` section 8 and 8.1 |

**Looked at and found without a collision:** tensor dtypes (1 = i64, 2 = f64 in `mfile.cho`, `record.cho`, `tree/model.cho`; the schema column
types 0, 1, 2 are a different field); section tags of the file (1 to 5, only `mfile.cho` writes them); module names (all libraries, the three
stage test programs and `lib/core` compile in one program: `tests/t_registry.cho`); the import graph (no cycle, the compiler would refuse
one); `cancho.toml` (`linalg-check`, `models-check`, `ml`: distinct names, `lib/core` shared; the drivers `metrics-check`, `tree-check` and
`reader-check` have no `[[bin]]` entry and are built by their gate scripts from explicit lists, as `models-check` is); `tools.toml` (one tool, `ml`,
whose derived authority is unchanged: `manifest.py --check`); `schemas/` and `generated/ml` (not changed by any stage but the CLI's; `schemas.py
--check`); `table.pin` (`check_table_pin.sh`). The mutant harnesses of the stages each copy their own file lists and do not interact.

## 2. What was changed (one commit each)

1. `integration: lib/core/matrix.cho imports linalg.tensor...` : the three build lists above.
2. `Fix the cross-stage collisions...` : `lib/core/kinds.cho` (the registry in code), `tests/t_registry.cho` (5 tests: the model kinds, the stream
   kinds, the stages' own constants against the table, no two stages sharing a model kind, no two libraries sharing a refusal code and one
   exit code per tag), `scripts/check_registry.py` (the documents and the sources that still write a bare number), the renumbering below, and
   `scripts/check.sh`. The new tests can fail: with the tree's kind set back to 3, `test_stages_use_the_registry` and
   `test_no_two_stages_share_a_model_kind` fail (run, then restored).
3. `Register the tags...` : `tools/ml/spec.cho` `pending()` (13 tags, not in `rules()`, so not in `introspect`, the schemas or the fixtures),
   `design.md` section 8 (a table of the 13), `docs/metrics.md` (its four tags are no longer "new").
4. This document, `design.md` section 16 (the metrics stage had none), the README line.

**Files that exist keep their bytes.** `tools/ml/mfile.cho` was not touched, and every model file the CLI writes (kinds 1, 2, 3) is what it was:
the 101 conformance tests, among them `test_model_file.py` (an independent parser, 20 tests) and the oracle tests, pass unchanged but for the
one case below.

**Edits to tests, scripts and oracles that the renumbering forced** (each one keeps what the gate checks or makes it stricter; none was made to
pass a failing gate, and no gate failed because of the merge but for the file lists of the first commit):

* `tests/p_models.cho`: the hand-written layout bytes, kind 3 to 4 (linear) and 5 to 6 (k-means).
* `tests/conformance/test_model_file.py`: the refused case "kind 7" (7 was "unassigned") is now "kind 99", and kinds 4, 5, 6, 7 are cases of their
  own (`ml` does not build them: `ml.model.malformed`).
* metrics: `lib/metrics/rules.cho` (codes 201 to 211, `first()`, `last()`), `lib/metrics/splits.cho` (kinds 4, 5, 6),
  `tests/t_metrics.cho` (the registry test loops from `first()` to `last()`, and also asks that the code before the band and the one after it
  have no tag and that `last - first + 1 = count`), `tests/t_metrics_fuzz.cho` (a code must be 0 or inside the band; before: between 0 and 11),
  `tests/t_metrics_split.cho` (the stream vector is now `stream(42, 4, 0) = f94f852667e9fcae`, the kinds are 4, 5, 6), `scripts/gate_metrics.py`
  (the tag list must be 201, 202, ...), `scripts/oracle_metrics.py` (the three kinds; the three reference vectors recomputed with its own
  Python implementation: k-fold `0 0 2 2 0 2 1 1 0 1`, train/test `0 0 1 0 0 0 0 1 1 0`, stratified `1 2 0 2 1 0 0 2 1 0`; the 2,978 cases
  of the splits gate still agree bit for bit), `scripts/mutants_metrics.py` (five patterns re-aimed at the new numbers), `docs/metrics.md`.
  **A consequence to know:** the assignment vectors of the splits for a given seed changed, because the stream kinds changed. Nothing stored
  depends on them yet.
* `docs/models-linear.md`, `docs/models-tree.md`, `docs/design.md` sections 14 and 15: the numbers, each with a note saying it was renumbered.

**A gate that is wrong:** none found. The tree gate G-ref against scikit-learn fails for the regression forest R1 (section 3): that is the
stage's own recorded verdict (`docs/models-tree.md` 10.8, `docs/bench/gate_sklearn.txt`); the rerun here prints the same rows byte for byte
(`diff` empty), so it is not an integration breakage and its tolerance was not moved.

## 3. The gates on the merged tree

All commands from the repository root, `CANCHO=../cancho-compiler-a4572ea/target/release/cancho`.

| gate | command | result |
|---|---|---|
| everything CI runs | `sh scripts/check.sh` (exit 0) | green; contents below |
| format | `cancho fmt --check lib tests tools` | clean |
| core, exact, matrix, fsq unit tests | `cancho test` of `tests/core.cho`, `exact.cho`, `matrix.cho`, `fsq.cho` | 20, 7, 2 (matrix) and 3 passed, 0 failed (as printed) |
| linalg unit tests | `t_dot t_gemv t_syrk t_chol t_gemm t_fuzz` | 8, 4, 2, 5, 2, 8 passed |
| linalg against numpy | `oracle_linalg.py --tree`, `--suite`, `gate_par.sh` (thread suite 1,2,3,4,8,16) | tree for 1..4096 blocks ok; 46 cases all bits equal; thread suite no mismatch |
| linalg per kernel | `gate_linalg.sh` for dot, axpy, gemv, syrk, chol, fuzz, gemm (run one by one) | 7 of 7 exit 0, "all bits equal" |
| models | `sh scripts/gate_models.sh` | 23 unit tests ok; oracle suite 120 configuration lines (single thread and 2, 4, 8, 16 threads) all bits equal |
| models against scikit-learn | `sklearn_models.py` (data `ml-data`, absolute path) | all 39 comparisons PASS ("sklearn gate: all PASS") |
| metrics | `sh scripts/gate_metrics.sh` | 3 test programs ok; ratio 3,066 pairs, counts 68, AUC 109, sums 1,033, log-loss terms 13,407 of 13,896 bit equal (worst 1.00 ulp), splits 2,978 cases, sklearn table, tags: 11 registered, 11 reached by 45 fixtures, fuzz 400 driver runs (85 ok, 315 refused with a tag, none outside exits 0/2/8) |
| trees | `sh scripts/check_tree.sh` (`gate_tree.py` default, `mutants_tree.py --check`) | 21 unit tests; exact comparison 2,006 pairs 0 wrong; identity with the Python oracle IDENTICAL on 20 (configuration, size) pairs; thread gate EQUAL on 11 cases at 1, 2, 3, 4, 5, 8, 16 threads; fuzz of the model reader 0 unexpected; identical to spike F (C1 at 1,000 and 100,000 rows); "GATE passed" |
| trees against scikit-learn (G-ref) | `gate_sklearn.py --big` | **exit 1, as the stage recorded**: "G-ref (pre-registered) FAILED": R1 (regression, missing values, 100,000 rows) RMSE 2.784408 against 3.029622, 8.1% apart with the tolerance 3%; the other pre-registered rows are within (C1, C2, C3, R2, and C1 at 1,000,000 rows). The output equals `docs/bench/gate_sklearn.txt` byte for byte (`diff` empty); the cause is attributed in `docs/models-tree.md` 10.8 |
| registries | `tests/t_registry.cho`, `scripts/check_registry.py` | 5 passed; 7 model kinds, 6 stream kinds, 34 refusal codes in 3 bands: tables, sources and documents agree |
| reader | `build_reader.sh`, `gate_reader.py all`, `mutants_reader.py --check` | RD2 1,966,143 accepted cells and 34,380 refusals compared, 0 differences; RD3 192 files, 28,518 cells, 0; RD4 36 limit checks, 0; RD5 277 files, 11,331 cells, 17 refusal cases, 0; RD6 5,014 + 5,115 runs, 0 traps |
| CLI | `schemas.py --check`, `manifest.py --check`, `python3 -m unittest discover` in `tests/conformance`, `mutants_cli.py --check` | schemas and authority manifest agree; **101 tests OK (1 skipped: the determinism test across `--threads`, which `ml` does not have)**; 45 mutant edits apply |
| the stages' mutant lists apply | `mutants_{core,linalg,metrics,models,reader,tree,cli}.py --check` | 50, 86, 146, 67, 40, 100, 45 patterns apply once |

**The mutant runs (the long ones; all were run, each with its control passing first):**

| script | mutants | killed | survived, each explained in the script | unexplained |
|---|---|---|---|---|
| `mutants_core.py` | 50 | 48 | 2 | 0 |
| `mutants_linalg.py` | 86 (dot 13, axpy 7, gemv 16, syrk 12, chol 17, fuzz 2, par 9, gemm 10) | 84 | 2 (both in dot) | 0 |
| `mutants_models.py` | 67 | 65 | 2 | 0 |
| `mutants_metrics.py` | 146 | 138 | 8 | 0 |
| `mutants_tree.py` | 100 | 94 | 6 | 0 (and 0 that did not build) |
| `mutants_reader.py` | 40 | 36 | 4 | 0 |
| `mutants_cli.py` | 45 | 45 | 0 | 0 |

The counts equal the ones each stage recorded for itself (metrics 138 and 8, tree 94 and 6, models 65 and 2, reader 36 and 4, cli 45).

**Not run:** the benchmarks (`bench_*.py`, `blas_thread_variance.py`, `scripts/ablate`, `scripts/c_parity`), as instructed; a second machine
(D2, Mac against Linux x86-64: the bytes across platforms are untested here as in each stage's document); a package-mode build of
`models-check` (`cancho build --bin models-check`; its entry exists and was never built in package mode by its stage); the 1,000,000-row
timing and identity runs of the tree stage (`gate_tree.py --sizes` beyond the defaults).

## 4. The final registries

The tables themselves are in `design.md` (4.1 model kinds, 8 and 8.1 refusals) and `docs/rng.md`; `lib/core/kinds.cho`, `lib/core/refuse.cho`,
`lib/metrics/rules.cho` and `lib/models/tree/deny.cho` are the code; `tests/t_registry.cho` and `scripts/check_registry.py` fail on any
disagreement. Summary:

* **Model kinds** (`u32` opening the `model` section): 1 constant regression (int, dec), 2 constant classification, 3 constant regression of a
  float target, 4 linear, 5 logistic, 6 k-means, 7 tree or forest (the task is a field of its first record). The next is 8. `mfile.cho` writes
  and reads 1 to 3 only.
* **Tensor dtypes:** 1 = i64, 2 = f64. **Section tags:** 1 plan, 2 schema, 3 steps, 4 model, 5 meta.
* **Refusal codes:** `ml.refuse` 1 to 15 (tensor.rank, tensor.shape, tensor.bounds, input.empty, input.too-many-rows, input.too-many-columns,
  input.too-many-cells, param.out-of-range, linalg.not-positive-definite, linalg.singular, param.threads, input.non-finite,
  label.single-class, label.not-binary, fit.non-finite); `tree.deny` 101 to 108 (input.non-finite, label.not-binary, label.single-class,
  limit.depth, limit.nodes, model.truncated, model.malformed, schema.mismatch; for a code below 101 it answers as `ml.refuse`); `metrics.rules`
  201 to 211 (input.empty, input.too-many-rows, input.non-finite, label.single-class, label.not-binary, metric.undefined, param.out-of-range,
  metric.length-mismatch, metric.bad-probability, split.empty-fold, split.stratum-too-small). 301 and above are free. A tag in two bands is one
  rule with two library-local codes: the wiring translates by tag.
* **Exit codes:** 2 for `ml.param.out-of-range` and `ml.param.threads`, 8 for every other library tag, one exit code per tag in all three bands.
* **Rule tags reachable from `ml`:** `extra()` and `rules()` of `tools/ml/spec.cho`, unchanged *(at the merge; 18 were added at W1: `docs/design.md` section 19)*. **Tags of the libraries not reachable yet:**
  `pending()` (13): ml.tensor.rank, ml.tensor.shape, ml.tensor.bounds, ml.linalg.not-positive-definite, ml.linalg.singular, ml.param.threads,
  ml.fit.non-finite, ml.limit.depth, ml.limit.nodes, ml.metric.length-mismatch, ml.metric.bad-probability, ml.split.empty-fold,
  ml.split.stratum-too-small. `ml.limit.fits` of the first list in `design.md` section 8 is built by no stage.
* **RNG stream kinds** (`derive(derive(seed, kind), index)`): 1 tree bootstrap, 2 tree node features, 3 k-means centers, 4 train/test split,
  5 k-fold, 6 stratified k-fold. Test-input streams (`derive(seed, 1..3)` in `linalg-check`, `models-check` and the oracles) use a generator's
  own seed and never run in a library path. The next kind is 7.

## 5. Where the summation order's constants live (for the lane task; nothing here was changed)

* `lib/linalg/order.cho`: `block()` = 256 (the block length K), `group()`, `slots()`, and the fixed binary tree (`settle`, `settle_root`,
  `finish`). **No lane constant exists there**: inside a block every linalg kernel (`dot`, `gemv`, `syrk`, `gemm`, ...) is one ascending chain,
  which is the order of `docs/linalg.md` section 2. The models (`lib/models`) take their sums from these kernels.
* `lib/metrics/sums.cho`: `lanes()` = 4 and `block_partial`, which unrolls 4 accumulators (`s0..s3`, term `i` of the block to lane `i mod 4`,
  block partial `(l0 + l1) + (l2 + l3)`), then the same tree of `linalg.order`. It is **its own constant** and its own order, documented in
  `docs/metrics.md` section 2.4. So the two libraries do not share a summation order in a block today.
* The tree library's regression sums are sequential in a fixed row order (`docs/models-tree.md`), and its classification sums are exact
  integers: neither depends on a lane count.

## 6. What is not wired

> **Superseded at stage W1 (`docs/wiring.md`), kept as it was written at the merge.** `ml fit | predict | eval` now call the pipeline, the transforms, the metrics
> and the splits, and build linear (kind 4) and logistic (kind 5) models through `tools/ml/wire.cho`; `--threads` exists and `conc` is in the authority row; the tags the
> verbs reach moved out of `pending()` (nine stay, `docs/wiring.md` section 5). Still not wired: k-means, trees and forests, multiclass, `ml.limit.fits`, text columns.

* **The CLI verbs call none of the libraries.** `ml fit | predict | eval` build and read only kinds 1, 2, 3 (the constant model). `lib/linalg`,
  `lib/metrics`, `lib/models` (linear, logistic, k-means) and `lib/models/tree` are reached by their own drivers and tests only.
* `tools/ml/mfile.cho` does not read or write kinds 4 to 7 (it refuses them as `ml.model.malformed`; one conformance case each). The writers
  and readers of those payloads are `models.record` and `tree.model`.
* The pipeline (`impute`, `scale`, `encode`), `--threads`, `ml.limit.fits`, multiclass, the second machine, the shared reader as a package
  dependency of cancho-table (`docs/reader.md` section 1).
* Wiring a verb means: move its tags from `pending()` to `extra()` and `rules()` with a fixture each, regenerate `schemas/` and
  `generated/ml` (`scripts/schemas.py`, `scripts/manifest.py`), add `conc` to the authority row when `--threads` arrives, and add the kind to
  `mfile.cho` with a parser in `tests/conformance/test_model_file.py`.

## 7. Open items, with reproducers

1. **G-ref, regression forest R1** (above): a recorded, pre-registered gate that fails; the stage attributed it and did not move the tolerance.
   Reproduce: `python3 scripts/gate_sklearn.py --big; echo $?` prints 1 and the R1 row.
2. **`models-check` traps without a message on a path containing `..`** (found while running `sklearn_models.py` with `ML_DATA` set to a path
   with `..`; also in the linear stage's own tree, so not an integration effect). Reproduce, from the repository root after
   `sh scripts/gate_models.sh`: `head -c 8000 /dev/zero > y.bin; build/gate-models/models-check linreg ../ml-data/n1000/x.bin y.bin n=1000 d=20; echo $?`
   prints 132 (SIGILL) and nothing, where a missing file prints `error: bad or missing file` and exits 1, and an absolute path without `..`
   works. It is a test driver, not a library, and `ml` checks paths itself (`path.dotdot`); not fixed here because the cause (the runtime's
   `open_read` on such a path) was not investigated. **Fixed afterwards: `docs/lanes.md` section 1** (the cause is `docs/upstream/dotdot_path.cho`).
3. `check.sh` now needs `../ml-data` (the files of spike F) for the tree gate, in addition to `../cancho-table-main`.

## 8. Post-merge (`v1-post`: the tree after `v1-lanes`, `v1-pipe` and `v1-docs` were merged into `v1-int`)

Machine: Mac arm64 (Darwin 25.2.0), cancho `a4572ea`, Python 3.14.5, numpy 2.4.6. No benchmark was run. Each step below is one commit.

### 8.1 Green tree (step 1)

Run on `v1-int` (`819e027`): `scripts/gate_pipe.sh` passed unchanged (t_transforms 29, fuzz 2, t_pipeline 13, fuzz 4, registry 7; `oracle_pipe.py`
all cases agree, `sums` 186 vectors); `scripts/check.sh` stopped at the registry program and `check_docs.sh` was not part of it. What the merge broke, and the fix:

| finding | fix (code, not gate) |
|---|---|
| `scripts/check.sh` compiled `tests/t_registry.cho` from the old file list, but the merged `t_registry.cho` imports `transforms.codes` (error: no module) | the list is the one of `gate_pipe.sh` (transforms, pipeline, `tools/ml/csvcell.cho` and the three decimal modules of cancho-table) |
| `check.sh` ran neither `check_docs.sh` nor (it did) `gate_pipe.sh`/`mutants_pipe.py --check` | `gate_pipe.sh` and `mutants_pipe.py --check` were already in the merged block (verified); `check_docs.sh` added at its end |
| expected: the oracle's `ordered_sum` out of step with the package order | **did not happen.** `v1-pipe` already reached the decisive sums through `metrics.sums.reduce`, which `v1-lanes` moved onto `linalg.order` (`block_partial`); the oracle already had the lane rule, and `--sums` agreed bit for bit before and after. `dsum.cho` still calls `metrics.sums.reduce` (one door), not `order.cho` directly: calling it directly would be a second copy of the block loop |
| a real gap found while checking "one order": `metrics.sums.reduce` cut blocks at a literal `256`, not at `order.block()`, so a change of K in `order.cho` would not have reached the metrics or the transforms | `reduce` uses `order.block()`; two new mutants in `mutants_pipe.py` (`order-lane-combine-fold`, `order-block-length-128`; the second **survived** before the fix and is killed after); `mutants_metrics.py` `sum-block`, `sum-block-up` re-aimed at the new text (still killed) |
| stale prose: `transforms.md`, `pipeline.md`, the oracle docstring and `dsum.cho` said the lane change was "in progress" or named `metrics.sums` as the order | corrected in place |

`mutants_pipe.py`: 62 patterns before, 64 after. `mutants_metrics.py`: 146 and 146.

### 8.2 Rank deficiency (step 2)

The 4-lane order made equal columns with `l2 = 0` silently fit (`docs/lanes.md` section 3): the pivot test `s > 0.0` saw rounding noise. The pivot test is now
`s > 2^-40 * a_jj` in `lib/linalg/chol.cho` (`pivot_tol()`, `pivot_ok`), tag unchanged (`ml.linalg.not-positive-definite`, exit 8), with new repair texts in
`lib/core/refuse.cho` `repair(9)`, `tools/ml/spec.cho` `pending()` and `design.md` section 8. The rule, the choice of `tol` and the measurements (relative pivot of
well-conditioned data 0.96 to 0.99; of a standardised pair with correlation 0.999999, 2.0e-06; rounding noise of dependent columns at most 7.9e-16 in absolute value over 960
fits; the pair at `1 - rho = 1e-12`, 2.0e-12, accepted) are `models-linear.md` 3.1 and `scripts/pivot_evidence.py`. Gates that changed their expectation: `tests/p_models.cho`
`linreg_refusals` (the strong form is back: equal columns refused at 1 and 4 threads; plus a nearly dependent design fitted, a 1e-7 one refused, a badly scaled one fitted);
`tests/t_chol.cho` (new `test_chol_relative_pivot`); the oracle suite (`oracle_models.py`: 7 configurations added with `dup=4`, `5`, `6` in the driver `tools/models-check`
and the Python generator); `oracle_linalg.py` `chol_o` follows the rule. No tolerance of any comparison was changed. The `sklearn_models.py` comparison passes (all rows PASS,
numbers as before: the pivot test does not touch a well-conditioned fit). Mutants: `mutants_linalg.py` 127 -> 134 (the two `chol-pivot` patterns re-aimed, 7 new,
all killed), `mutants_models.py` 81 -> 87 (6 new, all killed; `lin-pivot-wrong-diag` first survived, until the badly scaled design was added to the gate).

### 8.3 The sum order in the model file (step 3)

Rule first (`design.md` 4.1, section 17), then code. **Encoding: presence by model kind, no flag, no version bump.** A model file of kind 4 to 7 has a `u32 sum_order` as the
last four bytes of `meta` (after the compiler pin), value 2 = the order of `docs/linalg.md` section 2; 1 is retired and never written. Kinds 1 to 3 have no such field and
their bytes are exactly those of `v1-int`: `tools/ml/mfile.cho` still writes only kinds 1 to 3, its encoder is untouched, and `tests/conformance/test_model_file.py` parses
the three constant kinds (`meta` is `32 + pin length` bytes) and compares the parsed sections with its own writer byte for byte. Readers (`mfile.decode`, `file.decode`) read the
field after the checksum and before the other fields, for kinds 4 to 7, when `meta` has the shape `36 + pin length`; a value other than 2 is `ml.model.version`
(`ml`: status 6 of `mfile.decode`, detail `sum_order` and `known`; `lib/pipeline`: a `Fault` with `a` the value, `b` = 1, `c` = 2). A `meta` of any other length is not read for the order and
is refused by the ordinary field checks as `ml.model.malformed`; so is a constant model with four extra bytes. `lib/pipeline/file.cho` `encode` writes the field when the leading `u32`
of the model payload is 4 or more. Numbers shared by `mfile.cho` and `lib/core/kinds.cho` (`sum_order`, the first ordered kind, the last kind) are compared by `scripts/check_registry.py`.

Gates: `tests/conformance/test_model_file.py` 24 tests (was 20): an independent writer and reader of the field; kinds 4 to 7 x orders 0, 1, 3, 4, 255, 65536, 4294967295 are `ml.model.version` with the
detail; order 2 is `ml.model.malformed` (kind not built); wrong `meta` lengths; a corrupted-file fuzz of a kind 4 file (every truncation, every byte flipped three ways with and without a repaired
checksum, 300 seeded damages). `tests/t_pipeline.cho` (14 tests, was 13) and `t_pipeline_fuzz.cho` (5, was 4) do the same for the library. Mutants: `mutants_cli.py` 45 -> 53 (8 new, all killed),
`mutants_pipe.py` 64 -> 73 (9 new, all killed; `order-check-for-constants` needed a test that sets the kind byte to 1 and the order to 3).
Registered tag text changed: `ml.model.version` in `tools/ml/spec.cho` and `lib/transforms/codes.cho` now also says "or a model of kind 4 or above made with a summation order this reader does not implement";
`manifests/ml.authority.json` and `generated/ml/built.cho` were regenerated by `scripts/manifest.py` (the function list of `ml` changed), not edited.

### 8.4 Stale documents (step 4)

Found by the documentation pass, corrected in place with numbers printed by the scripts named (Mac arm64, `a4572ea`, load average about 8 to 10 while the mutant suites ran):
`docs/cli.md` section 7 (105 conformance tests, 99 fixtures, 20 flags, 49 rules, 4 log records; 53 mutants, with the M1 row at 12 and a new RD7 row of 11; the old text said 88, 90, 19, 48 and 34),
`docs/models-linear.md` section 6.8 item 5 (the model kinds `mfile.cho` does not build are 4 to 6; kind 3 is the constant float regression and is known),
the comment of `scripts/check.sh` on the mutant counts (87 for `mutants_models.py`, 53 for `mutants_cli.py`), `docs/transforms.md` and `docs/pipeline.md` ("once the lane-order change has landed"),
`README.md` (the Status paragraph and the "What it cannot do yet" bullet on the transforms: they exist as libraries, are not wired) and `docs/status.md` (the transforms and the pipeline moved from "being built" to the library table; the 4-lane order, the path check, the relative pivot
test and the sum-order field added; the measured table refreshed: 105 conformance tests, 702 mutants, 104 linalg shapes, 70 model-oracle cases, 59 refusal codes in 4 bands; the old "different summation order" open item replaced by the real open item,
that the speed tables predate the lane order). `python3 scripts/site.py` regenerated `docs/index.html` (the rule table carries the new `ml.model.version` text) and `scripts/check_docs.sh` passes.
Not corrected because not re-measured: the speed tables (`docs/bench/`, `linalg.md` 5, `models-linear.md` 6), which are marked as taken before the lane order.

### 8.5 Everything on the final tree (step 5)

| suite | patterns at the merge (`819e027`) | patterns now | killed | survive, each explained | unexplained |
|---|---|---|---|---|---|
| `mutants_core.py` | 50 | 50 | 48 | 2 | 0 |
| `mutants_linalg.py` | 127 | 134 (chol 26 -> 33) | 132 | 2 | 0 |
| `mutants_linalg.py oracle` (wrong oracles) | 11 | 11 | 11 | 0 | 0 |
| `mutants_models.py` | 81 | 87 (+6 pivot) | 85 | 2 | 0 |
| `mutants_metrics.py` | 146 | 146 | 138 | 8 | 0 |
| `mutants_tree.py` | 100 | 100 | 94 | 6 | 0 |
| `mutants_reader.py` | 40 | 40 | 36 | 4 | 0 |
| `mutants_cli.py` | 45 | 53 (+8 sum order) | 53 | 0 | 0 |
| `mutants_paths.py` | 19 | 19 | 19 | 0 | 0 |
| `mutants_pipe.py` | 62 | 73 (+2 one-order, +9 sum order) | 72 | 1 | 0 |
| total (without the oracle mutants) | 670 | 702 | 677 | 25 | 0 |

Gates: `sh scripts/check.sh` exit 0 (it now ends with `check_docs.sh`); the stage gate scripts it calls (`gate_par.sh`, `gate_models.sh`, `gate_metrics.sh`, `check_tree.sh`, `gate_reader.py`, `gate_paths.py`, `gate_pipe.sh`) pass;
`sklearn_models.py` all 39 comparisons PASS. `gate_sklearn.py --big` was not rerun (the trees were not touched; its recorded red gate, G-ref R1, is unchanged). No benchmark was run.
Gates whose expectation changed in `v1-post`: only `tests/p_models.cho` `linreg_refusals` (back to the strong refusal, section 8.2), the cases added to the model oracle suite, and the new tests of the sum order; no tolerance was changed.

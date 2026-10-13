# Wiring the first models into `ml fit | predict | eval` (stage W1 of v1)

**Status: sections 1 to 6 (the gates, the command line, the flow, the file, the refusals, the authority) were written before any code of this stage. Section 8
holds the results; a number or a verdict is in it only if a script of this repository printed it, with its conditions.** Tags as in `design.md`: **[choice]** a decision the
maintainer can overturn, **[read]** read from a document or the code, **[open]** not decided. The contract that makes a model a module is `docs/modules.md`.

Before this stage `ml` built one model, `constant`, and no library was called by a verb (`docs/integration.md` section 6). Now `ml` also fits **`linear`**
(ordinary least squares and ridge, `models.linreg`) and **`logistic`** (binary, Newton, `models.logreg`), **through the pipeline** (`docs/pipeline.md`): a plan of
`impute`, `scale`, `encode` and `bucket` steps fitted on the training rows, and the estimator, stored together in **one** `CML1` file that `predict` and
`eval` read. `--model constant` is untouched, byte for byte. k-means and the trees are libraries with a manifest and are not wired. *(Corrected in place: the tree and the forest were wired afterwards, `docs/wiring-trees.md`; k-means is the one that is not.)*

## 1. The gates (fixed before the code; each can fail, section 8 shows it failing)

| id | gate | judged by | a mutant it must kill |
|---|---|---|---|
| W1 | **Constant models do not change.** For every constant case of the conformance fixtures, the exit status, standard output and every file left in the sandbox (the model files first) equal, byte for byte, what the `v1-post` binary (commit `1b550a5`, built before this stage) produced: `tests/conformance/golden_constant.json`, written by `scripts/golden_constant.py` from that binary and never rewritten | `test_wire_baseline.py` | a changed field in `mfile.encode`; a flag default that leaks into a constant plan |
| W2 | **One file, stored steps, nothing to skew.** `fit` with a plan writes steps and estimator in one file whose sections are read by the independent parser; `predict` and `eval` take no plan flag (`args.conflict`); editing a stored step parameter with the checksum repaired changes the predictions; a plan section edited so that it disagrees with the steps, or is not canonical, is `ml.model.malformed`, and one edited only in a hyperparameter (a description) changes no answer | `test_wire.py`, `test_wire_file.py` | predict reading a plan; the steps not applied; the plan not compared with the steps |
| W3 | **Oracle, bit for bit.** A Python implementation fits the same pipeline and model in the same summation order and its file equals the file the real `ml` writes, byte for byte, for linear and logistic on seeded data with missing cells, constant columns and imputed, min-max and standardised columns; `predict` values and `eval` metrics equal the oracle's doubles | `test_wire_oracle.py`, `scripts/oracle_wire.py` | a step applied in the other order; `ddof` 1 for 0; the imputation mean over all rows; a coefficient rounded |
| W4 | **scikit-learn, with tolerances stated before the run.** Coefficients, intercept, RMSE, ROC-AUC and log-loss of `ml` against `LinearRegression` and `LogisticRegression` on the same data, without a penalty on either side (section 7; a penalised fit is not the same objective in the two libraries and is not compared) | `scripts/gate_wire_sklearn.py` | a tolerance exceeded |
| W5 | **Threads change no byte.** `fit --threads t` for t in 1, 2, 4, 8, 16 (and 1 to 16 in `admit`) gives the same file and the same stdout (but for nothing: stdout has no thread count) for linear and logistic; the D1 test that was skipped runs and passes | `test_determinism.py`, `admit.py` Q6 | a thread count that reaches the sum order |
| W6 | **Metrics are the library's and agree with an independent computation.** `eval` accuracy, precision, recall, F1, ROC-AUC, log-loss, RMSE, MAE and R2 equal Python (`scripts/oracle_metrics.py` functions and numpy) as doubles; the positive label flag changes precision, recall, F1, AUC and log-loss as it should | `test_wire_oracle.py` | positive and negative swapped; log-loss clipped at another epsilon |
| W7 | **Seeded split.** `--split train=F --seed N` selects the rows the `lib/metrics` train/test split selects (the oracle recomputes the permutation from the stream registry, kind 4), the same on every run and at every thread count; the test side is `ceil(n * (1 - F))` rows | `test_wire_oracle.py` | the stream kind changed; train and test swapped |
| W8 | **Tables and tags.** Every flag, every rule tag and every log record is reached by a fixture (gate T1 of `docs/cli.md`, extended); each tag the libraries can answer is either reachable from `ml` with a fixture or listed in `pending()` with the reason it cannot be reached (section 5) | `test_tables.py` | a flag with no fixture; a tag moved with no fixture |
| W9 | **No trap.** The argument fuzz includes the new flags and verbs; corrupted model files of kinds 4 and 5 (every truncation, every flipped byte with and without a repaired checksum, seeded damage of the plan, steps and estimator sections) are refused with a tag; plans from a fuzzer reach `fit` | `test_fuzz_args.py`, `test_wire_fuzz.py` | a removed bound in the estimator decoder |
| W10 | **Authority.** The derived row is the old row plus `conc`, with the reason in `tools.toml`; a binary that also reads standard input, the clock or writes by path fails the check | `test_authority.py`, `scripts/manifest.py` | a1 mutants of `docs/cli.md`, plus a label added without a reason |
| W11 | **Honest documents.** The README and the page show real output of the binary (`gen:` blocks), and no capability appears there before a gate above passed | `scripts/site.py --check` | a stale block |

## 2. The command line

```
ml [FLAGS] fit FILE       fit a model (constant, or a pipeline and an estimator) and write it
ml [FLAGS] predict FILE   apply a stored model to a CSV
ml [FLAGS] eval FILE      score a stored model on a CSV
```

New flags (the table is `ml introspect | jq .flags`; this list is for reading, and gate T1 checks the table against its projections). The ones marked *model* are
generated from the manifests (`docs/modules.md` section 3).

| flag | verbs | meaning |
|---|---|---|
| `--model constant\|linear\|logistic` | fit | *model*: the choice list is the wired models. Default `constant`, whose files and answers are the ones of `docs/cli.md` |
| `--plan FILE` | fit | a plan text (`docs/pipeline.md` section 2) **with its `model` line**; it is the one description of the fit: `--model`, `--step`, `--task` and the hyperparameter flags are then `args.conflict` |
| `--step TEXT` | fit | one or several steps, `verb key=value ...`, separated by `;` (**[choice]**: the pinned argument parser refuses a flag given twice, so the steps of one `ml` call share one value). The model line is built from `--model` and the hyperparameter flags |
| `--l2 X` | fit | *model* (linear, logistic): the ridge penalty, a finite number, 0 or more. Default `0.0` |
| `--iterations N` | fit | *model* (logistic): Newton iterations, 1 to 1,000. Default 100 |
| `--positive LABEL` | fit, eval | a logistic classification's positive label: the label whose probability the model predicts. `fit`: default the larger of the two labels; `eval`: default the model's own, and the other label is allowed (precision, recall and F1 are then those of that label, and its score is `1 - p`) |
| `--split train=F` | fit, eval | `0 < F < 1`, at most 6 digits after the point. `fit` trains on the train rows and reports the held-out rows' metrics (`holdout`); `eval` scores the test rows only |
| `--seed N` | fit, eval | the seed of the split (the stream registry, kind 4); stored in the file's `meta`. Default 0 |
| `--threads N` | fit, eval | 1 to 64, else `ml.param.threads` (exit 2). Default 1. In `fit` it is the thread count of the library kernels; **in `eval` nothing is threaded**: the flag is accepted and checked so that one command line can drive both verbs, and the bytes are the same at every value (W5) |
| `--proba` | predict | add the `probability` column (the positive label's probability) to a logistic model's answer |

`predict` and `eval` take no plan, no step and no model flag: the stored file is the only description. A flag a verb does not take, and a flag a model does not take
(`--iterations` with `--model linear`), is `args.conflict` as before. `--task` is not needed for the new models (`linear` is a regression, `logistic` a classification)
and is `args.conflict` if it says otherwise.

## 3. The flow

**fit** (planned model). The order is: the command line and the paths; the plan (from `--plan` or built from `--step` and the model flags) is **parsed first, before any data is
read** (a bad plan costs nothing); the model must be wired (`ml.plan.model-not-built` otherwise); the CSV is read into a rank-2 matrix with the missing-cell bitmap
(`ml.csvload`, `docs/reader.md` section 3): an empty cell is *missing* here, not an error; the target column is taken out of the matrix (a missing target cell is
`ml.input.non-numeric`, reason `empty`); a logistic target must be an integer column of two values; `--split` selects the train rows; the pipeline fits its steps on the train rows
(`pipeline.run.fit`); **if a feature cell is still missing after the steps, the fit is refused** with `ml.input.missing-value` and a repair that names the step to add (section 5);
the estimator fits (`ml.est.fit`, `--threads`); the file is encoded (`pipeline.file.encode`: plan, schema, steps, model, meta with the seed and, for kinds 4 and above, the sum order) and written
atomically as before (`--overwrite`, `--dry-run`).

**predict / eval** (planned model). The file is read and checked in the order of `docs/cli.md` section 4 (magic, version, truncation, checksum, sum order, fields); a file whose model kind is
1 to 3 goes to the old code, unchanged. The input CSV is read into a matrix; the schema check is the old one (`ml.schema.mismatch` with the missing, extra and mistyped columns, `--drop` as the
repair for extras); the columns are aligned to the fitted features by name; `pipeline.run.apply_steps` applies the **stored** steps (never a plan); a still-missing cell is refused as in `fit`;
the estimator predicts. `eval` reads the target too, applies `--split` to the file if asked, and computes the metrics of `lib/metrics`.

**Matrices and column types.** The reader builds the matrix of all kept columns in file order; text and non-numeric cells are `ml.input.non-numeric` with the `--drop` repair, as before. *The hook for W2
(text categories) was used by stage 1 of issue 21 (`docs/categorical.md`):* the reader's shape has a kind per column (0 none, 1 int, 2 dec, 3 float, **4 categorical**; the schema type code 3 is reserved for it
until stage 3 writes it, both registered in `lib/core/kinds.cho`), a text column is **declared** with `fit --cat NAMES` and never inferred, it is not a column of the float matrix but
of an `i64` id tensor beside it (`lib/core/catmat.cho`, ids from the bytewise-sorted dictionary of `lib/core/dict.cho`, -1 missing, -2 unseen at predict), and the only place that reads a cell as a
string is `read_row` through `cat_cell` in `tools/ml/csv.cho`. **Nothing in the estimators looks at a cell, and they did not change:** the wired models still see only the float matrix.
*Stage 2 of issue 21 (`docs/categorical-encoders.md`) built what this paragraph awaited: a planned model with `--cat` reads the table with `csvload.load_cat`, `pipeline.mixed`
fits the plan over the float matrix and the ids (a step reads numeric columns or categorical ones, decided by its verb and by `ml.plan.column-kind`), the encoders append their numeric columns, and the estimator
sees only the float matrix, as before. `tools/ml/catcli.cho` still answers a fit of `--model constant` with a declared column. The fit writes NO file while the model file cannot hold a vocabulary
(`ml.model.vocabulary-not-stored`, or `--split` / `--dry-run` for a fit and a score in one invocation); `predict` and `eval` take no categorical column. A numeric plan does not enter `pipeline.mixed`.*

## 4. The model file

A planned model is a `CML1` file exactly as `docs/pipeline.md` section 5 defines: `plan` (the canonical text, `model kind=linear l2=0.5\n` last), `schema` (as `mfile.cho`: all kept columns with the
target marked), `steps` (`u32 n` and the records), `model`, `meta` (`rows` = train rows, `columns`, `seed`, the pin, then `u32 sum_order = 2`).

`model` payloads: `u32 kind` then tensor records of `models.record` (`f64`, rank 1):

| kind | records | payload version |
|---|---|---|
| 4 linear | `coef` shape [d], `intercept` shape [1] | 1 |
| 5 logistic | `coef` shape [d], `intercept` shape [1], **`labels` i64 shape [2]: the negative and the positive label** | 2 |

`d` is the width the steps leave. The labels record is new: the library's payload for kind 5 had no place for the two labels of a CSV target (`0.0` and `1.0` are the library's), and `predict` must print
the labels the file was fitted on. No file of kind 5 existed before (the verbs did not write one), so version 1 of the logistic payload is retired without a compatibility cost; `tests/p_models.cho` has the
library's own test of the payload and of the labels record (`logistic_payload_and_its_labels`: the hand-written layout, every truncation, the wrong kind, trailing bytes, equal labels, a wrong dtype), and `scripts/mutants_models.py` has six patterns for it.

Kinds 1 to 3 are `mfile.encode`'s, unchanged (W1).

## 5. Refusals

Every tag of the libraries that can be reached from a command line moves from `pending()` in `tools/ml/spec.cho` to `extra()` and `rules()` with a fixture; the plan tags carry their line and column in
`detail`. New tags (exit 8):

| tag | when | repair |
|---|---|---|
| `ml.input.missing-value` | a feature cell is empty and the plan has no `impute` step for its column; `detail`: the columns | `retry` with `--step "impute column=a,b strategy=mean"` appended when no `--step` or `--plan` was given (and the names are plain); otherwise `none` with the step to add as the reason |
| `ml.plan.model-not-built` | the plan's `model` line names a model `ml` does not build (`kmeans`, or `constant`: use `--model constant`; `tree` and `forest` are built since `docs/wiring-trees.md`, whose summary in the rule table still lists them, left as it was so that the fixture of this rule keeps its bytes) | none |

**Reachable now, with a fixture:** `ml.linalg.not-positive-definite` (equal columns, `l2 = 0`), `ml.fit.non-finite` (logistic, diverging data), `ml.param.threads` (`--threads 0`), `ml.split.empty-fold`
(`--split` on too few rows), `ml.fit.all-missing`, `ml.limit.categories`, `ml.input.unknown-category`, the eight `ml.plan.*` tags of the parser and the column resolution, `ml.limit.plan`, and the metric refusals that remain in `eval`
(`ml.label.single-class` for a ROC-AUC with one class, `ml.label.not-binary` for a label the model never saw). **A metric with no value** (a precision with no predicted positive, R2 of a constant target, a non-finite result) is **`null`** in `metrics`, not a refusal: `ml.metric.undefined` stays reachable from the constant model only. **Not reachable, and staying in `pending()` with the reason:** `ml.tensor.rank`, `ml.tensor.shape`, `ml.tensor.bounds` (descriptors are built
inside the wiring from a validated shape), `ml.linalg.singular` (a zero on the diagonal of a triangular solve; the relative pivot test refuses it earlier), `ml.metric.length-mismatch`,
`ml.metric.bad-probability` (the wiring hands the metrics arrays of one length and probabilities in [0, 1] by construction), `ml.limit.depth`, `ml.limit.nodes` *(reachable since the trees were wired: `docs/wiring-trees.md` section 6)*,
`ml.split.stratum-too-small` (only the stratified split raises it; the verbs use the train/test split). `docs/design.md` section 8 says which.

## 6. Authority

The authority row of `ml` gains **`conc`**, because `ml` now links `linalg.par`, which spawns threads for `fit --threads N`. The label is derived by the compiler, so it is in the row at `--threads 1` too.
The deliberate change: `tools.toml` allows `conc` for `ml` and states the reason in `[ml.reasons]`; `scripts/manifest.py` refuses a label outside the base row (the row of `v1-post`) that has no reason, so a
label cannot be allowed by editing the ceiling alone; `manifests/ml.authority.json` and `generated/ml/built.cho` are regenerated, not edited; `docs/cli.md` section 6 says it. The test that no thread count
changes a byte (W5) is in the same diff, as `docs/cli.md` section 6 required. A model that needs another label states it in its manifest (`docs/modules.md`).

## 7. The scikit-learn comparison [choice]: tolerances fixed here, before the run

Data: seeded, 2,000 rows, 6 features, the target generated by a known linear rule plus noise (regression) or a noisy threshold of it (classification), with 5 percent of the cells of two columns missing and imputed by
the mean in both libraries. Standardisation in both (`ddof` 0). A ridge or a penalised logistic regression is not the same objective in the two libraries (ours does not scale by `n`, and
scikit-learn's logistic regression leaves the intercept unpenalised and uses a different solver), so the comparison uses **no penalty on either side** (`l2 = 0`; `LinearRegression`; `LogisticRegression` with `C = inf`, Newton-CG to `1e-12`) and checks the end-to-end result:
coefficients and intercept of the regression: relative `1e-9` of the largest coefficient; RMSE: relative `1e-9`; logistic coefficients and intercept (100 Newton iterations against a different solver, which stops at a different place): relative `1e-5`; ROC-AUC: absolute `1e-6`; log-loss: relative `1e-5`. If a tolerance is exceeded the gate fails and is reported, not widened.

## 8. Results

Machine: Mac arm64 (Darwin 25.2.0), cancho `a4572ea`, Python 3.14.5, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.8.0. The runs shared the machine with the mutant suites (the 1-minute load average was between 3 and 6 when sampled), so none is a quiet-machine run and no time is a claim. **No benchmark was run.** Every figure was printed by the script or the test named.

| gate | evidence |
|---|---|
| W1 constant models unchanged | `test_wire_baseline.py`: 97 constant cases (those of the CLI slice that use none of the new flags, but the refusal that lists the models) reproduce the exit status, the SHA-256 of standard output and the SHA-256 of every file left in the sandbox of `golden_constant.json`, written by the `v1-post` binary. The sales example's hash of `docs/cli.md` section 2.4 is among them |
| W2 one file, stored steps | `test_wire_file.py` (13 tests): the sections of a linear and a logistic file; `predict` and `eval` refuse `--plan`, `--step`, `--model`, `--l2`, `--iterations` and `--task` (`args.conflict`) and none is in `allowed()`; editing a stored fill or divisor with the checksum repaired changes the answers, and without the repair it is `ml.model.checksum`; a divisor of 0 or -1, an estimator record with another kind, dtype, shape or a non-finite value, a schema with 0 columns or a target out of range, a step length past the end: `ml.model.malformed`; a plan edited to disagree with the steps or to be non-canonical: `ml.model.malformed`, edited only in `l2`: the same answers |
| W3 oracle, bytes | `test_wire_oracle.py` and `scripts/oracle_wire.py --suite 60`: for linear 60 cases, 59 model files equal byte for byte (26 with a split) and 1 refusal equal (`ml.input.unknown-category` of a held-out row met by an `unknown=refuse` one-hot step); for logistic 60 cases, 52 files equal (15 with a split), 8 not judged (the oracle's `exp` is the library's table form, defined for \|x\| <= 660; `ml` is only checked not to trap there). Each case also compares `predict` on the training rows and on 40 other rows with the columns reversed and the target left out, and `eval` (twice for logistic, once per positive label), all as doubles. The generated data have missing cells (53 of 60 cases have some, 5 or 15 percent of a column), a constant column (21 of 60 cases), int, decimal, large-magnitude and exponent columns, a five-category column in some, impute (mean, median, constant), min-max and standard (`ddof` 0 and 1) scaling, bucket (width and quantile) and one-hot steps. `--suite 300` (300 cases a model, 30 not judged) also agreed. The oracle's sensitivity was checked: a changed sum-order field makes every file differ |
| W4 scikit-learn | `scripts/gate_wire_sklearn.py`, three seeds each, bounds of section 7: linear coefficients and intercept worst 8.8e-15 (bound 1e-9), RMSE 0 (1e-9); logistic coefficients and intercept worst 1.9e-14 (1e-5), ROC-AUC 0 (1e-6), log-loss worst 1.5e-16, relative (1e-5): 15 comparisons, all PASS |
| W5 threads | `test_determinism.py`: linear (with a split) and logistic fitted at 1, 2, 4, 8 and 16 threads, `eval` at the same: one model file and one stdout per model. `scripts/admit.py` Q6: `--threads` 1 to 16, one file (616 and 667 bytes) |
| W6 metrics | inside W3, and `test_wire.py` (9 tests): precision, recall and F1 equal the integer ratios for both positive labels, the AUC of the other label with the score `1 - p` equals the first, a model that predicts one class gives `null` for precision and a number for accuracy |
| W7 split | inside W3 (41 cases with a split): the model equals the oracle's, which draws the mask from SplitMix64 on Python ints (stream kind 4); `test_wire.py`: the held-out metrics of `fit` equal those of `eval --split` with the same seed, another seed gives others |
| W8 tables | `test_tables.py` (T1) with 29 flags, 67 rules, 4 log records: all reached by 163 fixtures; `scripts/schemas.py`, `gen_models.py` and `manifest.py --check` clean |
| W9 no trap | `test_fuzz_args.py` (5 tests; the new flags in every mistake: 3,388 vectors of 11 flags, 7 bases and 21 values, plus 2,500 seeded vectors with the new vocabulary); `test_wire_file.py`: 1,101 truncations, 2,202 flipped bytes (2 masks at every offset of two files of 592 and 509 bytes), 900 seeded damages of single sections with a repaired checksum: none traps, none with a wrong checksum is accepted; `admit.py` Q7: 764 and 807 more |
| W10 authority | `manifest.py --check`: `args, conc, dir_read, dir_write, err_write, file_read, file_write, fs_read(""), heap, io_write`; `test_authority.py` (8 tests, 2 new: the reason of every label beyond the base row, and a label allowed without one fails `manifest.gate`); `a1-reads-standard-input` and `a1-reads-the-clock` are still killed, by the ceiling (`io_read is not within the ceiling`) |
| W11 documents | `scripts/site.py --check`, `scripts/check_links.py`, `scripts/seo_check.py`: clean; the README and the page carry the pipeline flow and the split with the program's own output |
| the whole | `sh scripts/check.sh`: exit 0 (run in a clean clone of the commit); 138 conformance tests, none skipped (85.9 s in that run) |

**Mutants of the new code** (`scripts/mutants_wire.py`, `docs/bench/mutants_wire.log`): **40 patterns, 39 killed, 1 survivor explained, 0 unexplained**, and the unchanged tree passes. They are on the plan from flags to text (the step separator, the last line, the hyperparameter tokens), the plan parsing path (a model that is not wired, a bad token), the order of the steps (fitted on all rows, train and test swapped, the split's size, seed and sides, a held-out side scored as the training side), proba (the column, the flag, the other class, swapped labels), the positive label (`fit`, `eval`, the flipped score), the metrics (precision and recall swapped), missing cells (in `fit` and in `predict`, the repair's flag), the schema (by position, not checked, the target asked for or read), the threads (`t mod 4`, the ceiling, the floor), the sum-order refusal's detail, the generator (the token offset, `takes`) and the tables (`--plan` listed for `predict`, a library in the `--model` choice). The survivor, `w-bad-hyperparameter-token-ignored`, is a second line of defence behind the plan parser, which has refused every token that `hparams` would. **The first run had four survivors**: three had no test (`--positive` of `eval` and its flipped score were never exercised with the other label; the check of an unseen label was shadowed by the library's own `ml.label.not-binary`, which has the same tag), and the tests that kill them are `scripts/oracle_wire.py`'s second `eval` with the other label and `test_wire.py`; the fourth is the survivor above.
**Mutants of the module code**: `scripts/mutants_models.py` 93 patterns (87 and six new for the logistic payload and its labels record: 91 killed, 2 survivors explained, both as before), `scripts/mutants_cli.py` 53 patterns, all killed after two were re-aimed (`t1-fixture-reaches-another-rule`: `ml.model.malformed` has two fixtures now; `m1-sum-order-from-kind-5` became `-from-kind-7`: kinds 4 and 5 are read by the pipeline's reader, so `mfile.cho`'s check is reached by kinds 6 and 7 only), `docs/bench/mutants_models.log`, `docs/bench/mutants_cli_w1.log`.

**What the gates do not claim.** The oracle is written by the same author from the same documents and reuses the transform and estimator oracles of the earlier stages (which are compared with the libraries by their own gates): it checks the wiring against the libraries, not the libraries against the world; scikit-learn is the outside check, within bounds, for unpenalised fits only. Nothing was run on a second machine. `eval --threads` has no threaded kernel, so W5 on `eval` shows only that the flag changes nothing. The eight cases of 120 that the oracle cannot judge are checked for a trap, not for a value.

## 9. Open questions

1. `--step` takes several steps in one value because the parser refuses a repeated flag. A repeatable flag kind would be a change to the pinned contract package.
2. `eval --threads` is inert. If a threaded predict is ever added the flag already exists.
3. Text categories: stage 1 of issue 21 reads and bounds them (`docs/categorical.md`); stage 2 encodes them (`docs/categorical-encoders.md`: a fit with them writes no file); the string table in the model file is stage 3.

# Wiring the first models into `ml fit | predict | eval` (stage W1 of v1)

**Status: sections 1 to 6 (the gates, the command line, the flow, the file, the refusals, the authority) were written before any code of this stage. Section 8
holds the results; a number or a verdict is in it only if a script of this repository printed it, with its conditions.** Tags as in `design.md`: **[choice]** a decision the
maintainer can overturn, **[read]** read from a document or the code, **[open]** not decided. The contract that makes a model a module is `docs/modules.md`.

Before this stage `ml` built one model, `constant`, and no library was called by a verb (`docs/integration.md` section 6). Now `ml` also fits **`linear`**
(ordinary least squares and ridge, `models.linreg`) and **`logistic`** (binary, Newton, `models.logreg`), **through the pipeline** (`docs/pipeline.md`): a plan of
`impute`, `scale`, `encode` and `bucket` steps fitted on the training rows, and the estimator, stored together in **one** `CML1` file that `predict` and
`eval` read. `--model constant` is untouched, byte for byte. k-means and the trees are libraries with a manifest and are not wired.

## 1. The gates (fixed before the code; each can fail, section 8 shows it failing)

| id | gate | judged by | a mutant it must kill |
|---|---|---|---|
| W1 | **Constant models do not change.** For every constant case of the conformance fixtures, the model file written by this `ml` equals, byte for byte, the file written by the `v1-post` binary (`ML_BASELINE`, the committed `1b550a5` build) and stdout equals too | `test_wire_baseline.py` | a changed field in `mfile.encode`; a flag default that leaks into a constant plan |
| W2 | **One file, stored steps, nothing to skew.** `fit` with a plan writes steps and estimator in one file whose sections are read by the independent parser; `predict` and `eval` take no plan flag (`args.conflict`); editing a stored step parameter with the checksum repaired changes the predictions; a plan section edited with the checksum repaired and the steps untouched is `ml.model.malformed` | `test_wire.py`, `test_wire_file.py` | predict reading a plan; the steps not applied; the plan not compared with the steps |
| W3 | **Oracle, bit for bit.** A Python implementation fits the same pipeline and model in the same summation order and its file equals the file the real `ml` writes, byte for byte, for linear and logistic on seeded data with missing cells, constant columns and imputed, min-max and standardised columns; `predict` values and `eval` metrics equal the oracle's doubles | `test_wire_oracle.py`, `scripts/oracle_wire.py` | a step applied in the other order; `ddof` 1 for 0; the imputation mean over all rows; a coefficient rounded |
| W4 | **scikit-learn, with tolerances stated before the run.** Coefficients, RMSE and ROC-AUC of `ml` against `LinearRegression`, `Ridge` and `LogisticRegression` on the same data (section 7) | `scripts/gate_wire_sklearn.py` | a tolerance exceeded |
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
(text categories):* the schema section already has a type code per column (0 int, 1 dec, 2 float; 3 is reserved), the reader's shape has a kind per column (0 none, 1 int, 2 dec, 3 float; 4 is reserved
for text), and the only place that decides a cell is not a number is `ml.csvcell.read` through `read_row` in `tools/ml/csv.cho`. W2 adds a column kind `text` there, a code column (an interned
category index) in the matrix, and the `encode` step reading it; nothing in `tools/ml/wire.cho` or the estimators looks at a cell, so nothing in them changes.

## 4. The model file

A planned model is a `CML1` file exactly as `docs/pipeline.md` section 5 defines: `plan` (the canonical text, `model kind=linear l2=0.5\n` last), `schema` (as `mfile.cho`: all kept columns with the
target marked), `steps` (`u32 n` and the records), `model`, `meta` (`rows` = train rows, `columns`, `seed`, the pin, then `u32 sum_order = 2`).

`model` payloads: `u32 kind` then tensor records of `models.record` (`f64`, rank 1):

| kind | records | payload version |
|---|---|---|
| 4 linear | `coef` shape [d], `intercept` shape [1] | 1 |
| 5 logistic | `coef` shape [d], `intercept` shape [1], **`labels` i64 shape [2]: the negative and the positive label** | 2 |

`d` is the width the steps leave. The labels record is new: the library's payload for kind 5 had no place for the two labels of a CSV target (`0.0` and `1.0` are the library's), and `predict` must print
the labels the file was fitted on. No file of kind 5 existed before (the verbs did not write one), so version 1 of the logistic payload is retired without a compatibility cost; `tests/p_models.cho` keeps the
library's own layout test, with the new record added.

Kinds 1 to 3 are `mfile.encode`'s, unchanged (W1).

## 5. Refusals

Every tag of the libraries that can be reached from a command line moves from `pending()` in `tools/ml/spec.cho` to `extra()` and `rules()` with a fixture; the plan tags carry their line and column in
`detail`. New tags (exit 8):

| tag | when | repair |
|---|---|---|
| `ml.input.missing-value` | a feature cell is empty and the plan has no `impute` step for its column; `detail`: the columns | `retry` with `--step "impute column=a,b strategy=mean"` appended when no `--step` or `--plan` was given (and the names are plain); otherwise `none` with the step to add as the reason |
| `ml.plan.model-not-built` | the plan's `model` line names a model `ml` does not build (`kmeans`, `tree`, `forest`, or `constant`: use `--model constant`) | none |

**Reachable now, with a fixture:** `ml.linalg.not-positive-definite` (equal columns, `l2 = 0`), `ml.fit.non-finite` (logistic, diverging data), `ml.param.threads` (`--threads 0`), `ml.split.empty-fold`
(`--split` on too few rows), `ml.fit.all-missing`, `ml.limit.categories`, `ml.input.unknown-category`, the nine `ml.plan.*`, `ml.limit.plan`, and the metric tags the metrics raise (`ml.metric.undefined`,
`ml.label.single-class`, `ml.label.not-binary` in `eval`). **Not reachable, and staying in `pending()` with the reason:** `ml.tensor.rank`, `ml.tensor.shape`, `ml.tensor.bounds` (descriptors are built
inside the wiring from a validated shape), `ml.linalg.singular` (a zero on the diagonal of a triangular solve; the relative pivot test refuses it earlier), `ml.metric.length-mismatch`,
`ml.metric.bad-probability` (the wiring hands the metrics arrays of one length and probabilities in [0, 1] by construction), `ml.limit.depth`, `ml.limit.nodes` (trees are not wired),
`ml.split.stratum-too-small` (only the stratified split raises it; the verbs use the train/test split). `docs/design.md` section 8 says which.

## 6. Authority

The authority row of `ml` gains **`conc`**, because `ml` now links `linalg.par`, which spawns threads for `fit --threads N`. The label is derived by the compiler, so it is in the row at `--threads 1` too.
The deliberate change: `tools.toml` allows `conc` for `ml` and states the reason in `[ml.reasons]`; `scripts/manifest.py` refuses a label outside the base row (the row of `v1-post`) that has no reason, so a
label cannot be allowed by editing the ceiling alone; `manifests/ml.authority.json` and `generated/ml/built.cho` are regenerated, not edited; `docs/cli.md` section 6 says it. The test that no thread count
changes a byte (W5) is in the same diff, as `docs/cli.md` section 6 required. A model that needs another label states it in its manifest (`docs/modules.md`).

## 7. The scikit-learn comparison [choice]: tolerances fixed here, before the run

Data: seeded, 2,000 rows, 6 features, the target generated by a known linear rule plus noise (regression) or its sign (classification), with 5 percent of the cells of two columns missing and imputed by
the mean in both libraries. Standardisation in both (`ddof` 0). `Ridge(alpha = l2)` and `LogisticRegression(penalty = l2, C = 1/l2)` are not the same objective as ours (ours does not scale by `n`, and
scikit-learn's logistic regression leaves the intercept unpenalised and uses a different solver), so the comparison uses `l2 = 0` for the regression coefficients and checks the end-to-end result for the rest:
coefficients of the regression: relative `1e-9` of the largest coefficient; RMSE: relative `1e-9`; logistic coefficients with `l2` tiny (`1e-8`) and 100 Newton iterations: relative `1e-5` (a different solver
stops at a different place); ROC-AUC: absolute `1e-6`; log-loss: relative `1e-5`. If a tolerance is exceeded the gate fails and is reported, not widened.

## 8. Results

*(Filled in when the code and its runs exist.)*

## 9. Open questions

1. `--step` takes several steps in one value because the parser refuses a repeated flag. A repeatable flag kind would be a change to the pinned contract package.
2. `eval --threads` is inert. If a threaded predict is ever added the flag already exists.
3. Text categories are W2.

# Wiring the tree and the forest into `ml fit | predict | eval` (stage W3a of v1)

**Status: sections 1 to 10 (the gates, the command line, the tasks, the file, the missing-value rule, the refusals, the threads, the oracle, the
scikit-learn tolerances, the admission) were written before any code of this stage; the first commit of the branch holds only this file. Section 11
holds the results; a number or a verdict is in it only if a script of this repository printed it, with its conditions.** Tags as in `design.md`:
**[choice]** a decision the maintainer can overturn, **[read]** read from a document or the code, **[open]** not decided. A claim found false is
corrected in place and marked.

Until this stage `--model` offered `constant`, `linear` and `logistic`. The decision tree and the random forest of `docs/models-tree.md` were
a library with a manifest and `wired = false`. This stage wires them **through the same door as linear and logistic** (`docs/modules.md` is the
contract, `docs/wiring.md` the flow): the manifests gain the flags, `scripts/gen_models.py` writes the dispatch, the plan keys, the flag table and the
schemas, and `fit` stores steps and model in one `CML1` file. Categorical (text) columns stay out of scope: `--cat` is refused as today
(`ml.input.categorical-not-encoded`), stage 2 and 4 of issue 21 own them.

## 1. The gates (fixed before the code; each can fail, section 11 shows it failing)

| id | gate | judged by | a mutant it must kill |
|---|---|---|---|
| WT1 | **Tables from the manifests.** The flags of tree and forest (section 2) are data in `lib/models/{tree,forest}/model.toml`; `gen_models.py --check`, `schemas.py --check` and the table gate T1 are clean; every new flag is reached by a fixture and every new rule by one with its exit code; the `--model` choice list is the wired models | `gen_models.py --check`, `test_tables.py` | a flag with no fixture; a range edited in the generated file only |
| WT2 | **Tasks.** `--task` chooses classification or regression for tree and forest (never inferred from the target, as for constant, section 3); a classification target must be an integer column of two values; `predict` prints the label (or the value), `predict --proba` the probability of the positive label, `eval` the metrics of `lib/metrics` that linear and logistic print | `test_wire_trees.py`, oracle | the task read from the wrong place; `--proba` printing the label; positive and negative swapped |
| WT3 | **The model file, kind 7.** `fit` writes one `CML1` file whose `model` section is the library payload of `docs/models-tree.md` section 7 and, for a classification, the `labels` record (section 4); the steps are stored and applied by `predict`/`eval`; `sum_order` is the value section 4 gives; a file is read by an independent Python reader | `test_wire_trees.py`, `oracle_wire_tree.py` | the labels record dropped; the sum order of the lane kernels written |
| WT4 | **Oracle, bit for bit.** The Python oracle (`scripts/oracle_tree.py` for the library, `scripts/oracle_wire_tree.py` for the file) writes the payload and the whole file, header, schema, stored steps, meta and trailer included, and they equal what `ml` wrote, on generated data with missing cells, constant columns, scaled, bucketed and imputed columns, regression and classification, tree and forest, at 1,000 and 100,000 rows where affordable; `predict`, `predict --proba` and every `eval` metric equal the oracle's doubles | `test_wire_tree_oracle.py`, `oracle_wire_tree.py --suite` | a step applied in the other order; the default side flipped; a leaf value rounded |
| WT5 | **Missing values.** A plan with no `impute` step is accepted for tree and forest and refused (`ml.input.missing-value`) for linear and logistic, as before; the rule of section 5 holds on cases that need it (a missing cell in the training rows, in the prediction rows, in a column the steps leave alone, in a bucketed column) against the oracle | `test_wire_trees.py`, oracle | the refusal kept for trees; the refusal dropped for linear |
| WT6 | **Refusals reachable from `ml`.** `ml.limit.depth`, `ml.limit.nodes`, `ml.input.too-many-rows` (the 2^20 rows of a tree fit, refused before the tree allocates anything and before the rows are copied), `ml.param.threads`, `ml.param.out-of-range` (mtry, min-leaf), `ml.label.*`, `ml.input.non-finite` have fixtures with their exit codes; `ml.limit.depth` and `ml.limit.nodes` leave `pending()` | `test_tables.py`, `test_wire_trees.py` | the row cap removed or moved after the copy; a tag moved with no fixture |
| WT7 | **Threads change no byte.** `fit --threads t` for t in 1, 2, 4, 8, 16 (1 to 16 in `admit`) writes one file and prints one answer, for tree and forest, both tasks, with missing cells and a split | `test_determinism.py`, `admit.py` Q6 | a thread count that reaches the streams (`mutant-threads-one-count`, `-eleven` still refused) |
| WT8 | **scikit-learn, tolerances fixed in section 9 before the run.** `RandomForest*` and `DecisionTree*` against `ml` on the shipped train/test files of `docs/examples/data`: accuracy, ROC-AUC, RMSE within the stated bounds; a deliberately broken control (depth 1) is outside them; the library's red gate R1 is referenced, not hidden and not moved | `scripts/gate_wire_sklearn.py tree|forest` | a tolerance exceeded |
| WT9 | **Admission.** `scripts/admit.py tree` and `forest`: Q0 to Q11 all PASS for `supported`, else the manifest says `experimental` and section 11 lists the failing checks; the six mutant modules are still refused | `admit.py`, `tests/modules/run.py` | the six mutant modules |
| WT10 | **Mutants of the wiring code.** Patterns on the flag-to-hyperparameter mapping, the task choice, proba, the kind 7 reader, the row cap, the missing-value acceptance and the thread count: all killed, or each survivor explained | `scripts/mutants_wire_trees.py` | itself |
| WT11 | **No trap.** The argument fuzz includes the new flags; damaged kind-7 files (every truncation, flipped bytes with and without a repaired checksum, seeded damage of the plan, steps and model sections, the labels record) are refused with a tag | `test_fuzz_args.py`, `test_wire_trees_file.py`, `admit.py` Q7 | a removed bound of the payload reader |
| WT12 | **Golden.** Every earlier fixture is unchanged (`golden_constant.json`, the 156 conformance tests that passed at `d481593`), and the model files of linear and logistic that the parent binary wrote are, byte for byte, those this binary writes (`tests/conformance/golden_planned.json`, written from the parent binary and never rewritten) | `test_wire_baseline.py` | a changed field in `file.encode` for kinds 4 and 5 |

## 2. The command line

New flags, generated from the manifests (the table is `ml introspect | jq .flags`; this is for reading). All are `fit` flags; `--model tree` takes the first
five rows, `--model forest` all of them. A flag a model does not take is `args.conflict`, as `--iterations` with `--model linear` is.

| flag | model | meaning | range, default |
|---|---|---|---|
| `--trees N` | forest | the number of trees | 1 to 1,024, 100 |
| `--max-depth N` | tree, forest | the deepest tree; above 30 is `ml.limit.depth` (the ceiling that keeps the heap number of a node in 31 bits) | 1 to 1,000, 8 |
| `--min-leaf N` | tree, forest | the fewest rows in a leaf; above the number of training rows is `ml.param.out-of-range` | 1 to 1,000,000, 1 |
| `--bins N` | tree, forest | the quantile bins of the split search | 2 to 255, 64 |
| `--mtry N` | forest | features tried per node; 0 is floor(sqrt(d)) for a classification and max(1, d/3) for a regression; above d is `ml.param.out-of-range`. A tree uses all d | 0 to 4,096, 0 |
| `--bootstrap N` | forest | 1 draws the rows of each tree with replacement, 0 uses every row once. A tree does not bootstrap | 0 to 1, 1 |
| `--max-nodes N` | tree, forest | the nodes one tree may have; above 4,194,304, or a forest whose slots would pass 2^24 nodes, is `ml.limit.nodes`; a tree that needs more is refused too, never cut | 1 to 999,999,999, 4,194,304 |

The other flags are the existing ones: `--task` (section 3), `--seed N` (the seed of the bootstrap and feature streams **and** of `--split`: one seed, stored
in `meta`; the library derives its streams of kinds 1 and 2 from it and the split uses kind 4, so the two never share a stream), `--threads N` (the threads
of the fit, 1 to 64; `predict` and `eval` are not threaded: the flag is accepted by `eval` and inert, as for the other models), `--positive`, `--split`,
`--step`, `--plan`. A plan names the model on its model line, `model kind=forest task=classification trees=50 max-depth=6`, and carries the same keys in the
order of the manifest; the canonical text always lists every key (defaults included), as it does for the other models.

## 3. The tasks [choice]

`tree` and `forest` serve both tasks, so the task is a hyperparameter of the model line (`task`, required, `classification` or `regression`). On the
command line it is the existing `--task` flag, default `regression` as for `constant` (`docs/cli.md` section 2): **the task is never inferred from the
target**, the same rule that `constant`, `linear` (always a regression) and `logistic` (always a classification) already follow, so one flag means one thing.
With `--plan` the task is the plan's.

* **classification**: binary, as logistic: the target must be an integer column with exactly two values (`ml.label.not-integer`, `ml.label.single-class`,
  `ml.label.not-binary`, as in `docs/wiring.md`); `--positive LABEL` names the label whose probability the model predicts (default the larger label); the fit
  trains with 1.0 for the positive label and 0.0 for the other; `predict` prints the label (the positive one when the probability is above 0.5, a tie goes to
  the other, the library's rule `docs/models-tree.md` 2.6); `predict --proba` adds the probability of the positive label (the mean over the trees of the leaf
  probabilities); `eval` prints accuracy, precision, recall, F1, ROC-AUC and log-loss for the positive label, with `--positive` choosing the label as for logistic.
* **regression**: the target is any numeric column; `predict` prints the mean over the trees; `predict --proba` is `args.conflict`; `eval` prints rmse, mae, r2.
  A regression target with a magnitude above 1e100 is `ml.input.non-finite` (the library's bound, `docs/models-tree.md` 4).

## 4. The model file

`model` payload of kind 7: the payload of the library, byte for byte (`docs/models-tree.md` section 7: `u32 kind = 7`, ten tensor records: header of eight
`i64` (task, trees, d, bins, total nodes, max depth, min leaf, mtry), tree offsets, feat, bin, dleft, left, right, na, nb as `i64`, val as `f64`), and for a
**classification** one more record after them, the one logistic has: `labels`, `i64` rank 1 shape [2], the negative and the positive label (`payload_version`
2 of the wired layout; the library's own payload is version 1, and a regression payload is exactly the library's). The CSV target's labels have no place in the
library payload, which knows only 0.0 and 1.0.

The reader of the file (`decode_blob` of the adapters, in `lib/models/treewire/`) reads the header, computes the size the library payload must have **before it allocates**,
hands the library reader (`tree.model.read`, which checks every record, every node, the preorder and the depth) exactly that many bytes, and then reads the labels
record (a classification: exactly 32 more bytes with two different labels; a regression: nothing more). **Every failure of the model section is `ml.model.malformed`** with
`section` 4, whatever tag the library gave (a payload shorter than its header asks, longer, a wrong dtype, rank or extent, equal labels, a node ceiling, a column count
that is not the schema's), as for linear and logistic: the file's own structure (a section that runs past the end) was refused as `ml.model.truncated` before the payload is
looked at. Two further rules, both about a file that disagrees with itself: the `task` of the plan's model line must be the payload's (a plan that says classification over
a regression payload is malformed; the other hyperparameters of the plan are a description, as `l2` is for linear), and the plan must name a model whose kind is the payload's
(already refused by `pipeline.file`). A constant plan with a kind 7 payload is not this module's: the old reader refuses it, byte for byte as before. Its fuzz is WT11.

**The summation order field [choice].** `lib/pipeline/file.cho` writes `u32 sum_order = 2` at the end of `meta` for every kind from 4 on, meaning the 4-lane order of
`docs/linalg.md` section 2. **A tree or a forest uses no lane sum.** `docs/models-tree.md` 3.2: the Gini decision is exact integer arithmetic; a regression sum is
a sequential IEEE sum from +0.0 in the node's row order (stable partition), the forest value a sequential sum over the trees in tree order; no sum is split
across threads, a tree is built by one thread, so the lane order cannot reach a byte of the payload. A file that said `2` would claim a
property it does not have (and would be refused by a future reader that changes the lanes though the payload does not depend on them). So **kind 7 records
`sum_order = 3`: "sequential, fixed row order"**, registered in `lib/core/kinds.cho` (`sum_order_sequential()`) and in `docs/design.md` 4.1 where 3 was
unassigned. A reader that knows only the order 2 refuses a kind 7 file with `ml.model.version` (detail `sum_order` 3, `known` 2) and a reader of this version
refuses a kind 7 file that says 2 (the same tag, `known` 3): the field is exact per kind. Kinds 4 and 5 are unchanged byte for byte (WT12).

## 5. Missing values [choice]

The matrix of the reader carries a bitmap of missing cells (`ml.matrix`, 32 rows to a word, one run of words per column) and the library takes a per-column
bitmap of bytes (8 rows to a byte, `docs/models-tree.md` section 6); the adapter converts one into the other, row `i` of column `c` being bit `i & 7` of byte
`i >> 3` of the column's run, and passes it with the value 0.0 stored in the cell, which the library never reads. The rule:

1. **Steps.** As `docs/transforms.md` 2.1 says, only `impute` removes a missing cell; `scale`, `bucket` and `encode` pass it through (a `one-hot` block is missing as a whole).
   The fit of a step looks at the non-missing cells; a column with none is `ml.fit.all-missing` for every step but `impute strategy=constant`. This is unchanged.
2. **Fit.** For **tree and forest** a missing cell left after the steps is **not** an error: the matrix and its bitmap go to the library, which learns, for each
   split, the side missing rows take (`dleft`, by the same criterion as the threshold; ties go right, `docs/models-tree.md` 2.4). A column missing in every training row
   (that no step selected) has no valid split and is never used.
   For **linear and logistic** the refusal `ml.input.missing-value` stays, with its repair, as in `docs/wiring.md` section 3.
3. **Predict and eval.** A missing cell follows the stored default side of each node it meets (a node that saw no missing row in training has `dleft = 0`: the
   missing row goes right); it is never an error and has no repair. Linear and logistic keep the refusal.
4. **The target** is never missing: an empty target cell is `ml.input.non-numeric` with reason `empty`, as for every model.
5. The stored steps are the same with and without missing cells; a plan that imputes gives, for a tree, the model the oracle gives for the imputed matrix.

The rule is tested against the library oracle (the oracle takes the matrix with its missing cells, `oracle_tree.py` already does) and against the refusal for the
linear models (WT5).

## 6. Refusals: which tags become reachable [choice]

| tag | exit | reached by | where it is checked |
|---|---|---|---|
| `ml.limit.depth` | 8 | `--max-depth 31` (or a plan key) | `tree.params.check`, before any allocation sized by the data |
| `ml.limit.nodes` | 8 | `--max-nodes 4194305`; `--trees 1024 --max-depth 30` on enough rows that `trees * cap` passes 2^24; a tree that needs more nodes than `--max-nodes` | `tree.params.check` (the first two), the growing tree (the third: the lowest tree's, at every thread count) |
| `ml.input.too-many-rows` | 8 | a tree or forest fit with more than 1,048,576 training rows (2^20: `docs/models-tree.md` 3.1, the exactness bound of the Gini comparison) | **after the file is read and the split chosen, before any copy of the rows** (the training rows are counted: the matrix of the reader exists, bounded by `--max-rows` as for every model, but the copies the steps and the library make, and everything the tree allocates, do not), with its own message and hint; the repair is none: fit on a sample or take `--split train=F` so that the train side is within the bound. The bound is on the training rows: 1,048,576 are fitted and a split of a larger file whose train side is within the bound is too |
| `ml.param.threads` | 2 | `--threads 0`, `--threads 65` (already), and the copies of a threaded fit above 2^34 bytes | the flag check; the copy ceiling is a formula of `n`, `d` and the threads: the smallest input that exceeds it is about 2.3e8 cells, so **it has no fixture** (a CSV of more than 400 MB); it stays covered by `tests/t_tree.cho` at the library level and is listed as such |
| `ml.param.out-of-range` | 2 | `--mtry` above d, `--min-leaf` above n | `tree.params.check` |
| `ml.label.single-class`, `ml.label.not-binary`, `ml.label.not-integer` | 8 | as for logistic | the wiring |
| `ml.input.non-finite` | 8 | a regression target above 1e100 | the library |
| `ml.model.truncated`, `ml.model.malformed`, `ml.model.version` | 8 | damaged files (WT11) | the file reader |

`ml.limit.depth` and `ml.limit.nodes` move from `pending()` to `extra()` and `rules()` in `tools/ml/spec.tmpl` (the tags of the libraries that `ml` does reach are in
`extra()`). `ml.metric.*`, `ml.tensor.*`, `ml.linalg.singular`, `ml.split.stratum-too-small` stay in `pending()` as before. `docs/design.md` section 8 is updated for the two.

## 7. Threads

`--threads N` is the library's `threads`: the forest is built by a static partition of the trees (`docs/models-tree.md` section 5), each tree wholly by one thread with
streams that depend on the seed and the tree number alone, so the packed forest is the same at every thread count; the authority row has `conc` already (both
tree manifests say `adds = []`: the linear models declared it first). `docs/models-tree.md` G-thread compares the library at 1, 2, 3, 4, 5, 8, 16 threads; WT7
compares **the model file** of `ml` at 1, 2, 4, 8, 16 (and `admit` at 1 to 16), for both tasks, with missing cells and a split. `predict` and `eval` run the
kernel with one thread (the answer does not depend on it).

## 8. The oracle

`scripts/oracle_tree.py` already reproduces the library's nodes and predictions line by line. This stage adds to it the **payload**: the records of
section 4 serialised in Python (`struct`), and a `__main__` guard so that it can be imported. `scripts/oracle_wire_tree.py` imports it together with the
helpers of `scripts/oracle_wire.py` (the CSV types, the schema, the step records, the meta section, the trailer, the transforms of `oracle_pipe.py`, the split
of `oracle_metrics.py`) and fits, in Python, the same pipeline and the same model as `ml fit`: it writes the file and the test compares it with the file of `ml`,
byte for byte; then `predict`, `predict --proba` and `eval` on the training rows and on other rows with the columns reversed, as doubles. The generated data have
missing cells (about half the cases), constant columns, int, decimal and float columns, impute, scale, bucket steps, a split and `--positive`, regression and
classification, tree and forest, with `mtry`, `bootstrap`, `bins`, `min-leaf` and `max-depth` varied. 1,000 rows and 100,000 rows are run (`--rows`, the latter
only in a dedicated test and in `admit`'s large case). **What this does not show**: the oracle is written by the same author from the same document, and the
library part is compared with `lib/models/tree` by `docs/models-tree.md` G-ident; the oracle checks the wiring (flags, steps, task, labels, file, missing cells),
not the algorithm against the world.

## 9. The scikit-learn comparison: tolerances fixed here, before the run [choice]

Data: the shipped train/test files of `docs/examples/data` (the same two files go to both programs): breast cancer (binary, 426 train / 143 test rows, 30 features),
iris versicolor against virginica (binary, 75 / 25 rows, 4 features), wine class 0 against the rest (binary, 133 / 45 rows, 13 features) and diabetes (regression,
331 / 111 rows, 10 features). Settings: `ml --model forest --trees 100 --max-depth 8 --min-leaf 1 --bins 64 --seed 0` against
`RandomForest{Classifier,Regressor}(n_estimators=100, max_depth=8, min_samples_leaf=1, bootstrap=True, random_state=0, n_jobs=1)` (`max_features` sqrt for
classification, 1/3 for regression, which are the library's defaults for `--mtry 0`), and `ml --model tree --max-depth 8` against
`DecisionTree{Classifier,Regressor}(max_depth=8, random_state=0)`. A **different algorithm** (exact thresholds on raw values against 64 quantile bins, its own random
generator, a float criterion, a different treatment of missing cells), so only the quality of the result is compared:

| quantity | forest | tree |
|---|---|---|
| test accuracy | within 0.03 absolute | within 0.06 |
| test ROC-AUC | within 0.02 absolute | within 0.06 |
| test RMSE | within 5% relative | within 12% relative |

The wider bounds for a single tree say how much variance one tree has on 100 to 400 rows; they are a sanity bound against a shared misreading, not evidence of
correctness (the oracle gate is). The control is a depth-1 forest and a depth-1 tree on the same files: it must fall outside the bound on at least one dataset
per task, or the gate cannot fail and is reported as such. **The library's own pre-registered gate G-ref (`docs/models-tree.md` 10.8) failed on its regression
configuration R1 (RMSE 8.1% away against a 3% rule, on a synthetic target with a product term); that result stands, the 3% rule is not moved, and the wiring's 5% is a
different rule on different data and does not replace it.** If a bound here is exceeded the gate fails and is reported as it is; the bounds are not widened after the run.

## 10. Admission and the module contract

* The adapter interface of `docs/modules.md` section 4 returns fixed-size float parameters and cannot carry a forest, whose size is known only after the fit.
  A manifest key `shape = "blob"` selects a second interface, written in `docs/modules.md` section 4b and generated by `scripts/gen_models.py` for tree and forest:
  `fit_blob`, `predict_blob`, `proba_blob`, `decode_blob` (the fitted model is a byte payload that the adapter builds and reads). The eight functions of section 4 stay
  the contract of the vector models and are untouched.
* `tree` and `forest` share kind 7 (a declared pair). Their dispatch key in the generated `est` is the kind, plus 100 for the second of the pair
  (`forest` = 107), because a fit must tell them apart (different hyperparameters) while a file only has the kind; `est.kind_of(key)` maps back, and the stored plan
  names the model that predict prints.
* The glue of the two adapters is `lib/models/treewire/` (not `lib/models/tree/`, which stays the library that `tree-check` builds alone): `adapt.cho` (module `models.adapt_trees`: the
  bitmap, the fit that returns the payload, the single-thread prediction, the reader), `adapter_tree.cho` and `adapter_forest.cho`. The manifests list them in `files`.
* The generic check of the integer flags of the command line is 1 to the ceiling of `spec.limits()`; `--mtry 0` and `--bootstrap 0` need a lower bound of 0, which the manifest says
  (`min = 0`) and the generated `est.hparam_min` gives to `check_nats` in `tools/ml/main.cho` (the one line of `main.cho` this stage changes).
* `admit.py` is adapted to the second interface: Q1 compares the four functions of the blob adapter with the template (the same way it compares the eight), Q6 and Q7
  fit with a task of the manifest (`tree` and `forest` serve both; the admission samples use a classification and a regression), Q9 runs the oracle with
  `scripts/oracle_wire_tree.py`. `tier = "supported"` only if all twelve pass; otherwise the manifest stays `experimental` and section 11 names the failing checks.
* The six mutant modules of `tests/modules/` must still be refused; the control (a twin of `linear`) must still be admitted.

## 11. Results

Machine: Mac arm64 (Darwin 25.2.0), cancho `a4572ea`, Python 3.14.5, numpy 2.4.6, scikit-learn 1.8.0. The machine was shared (load average 3 to 6) and the mutant suites ran beside other jobs: **no time is a claim, and no benchmark was run.** Every figure was printed by the script or test named.

| gate | evidence |
|---|---|
| WT1 tables | `gen_models.py --check`, `schemas.py --check`, `manifest.py --check` clean; `test_tables` with 42 flags, 20 limits, 74 rules, all reached by fixtures (`tests/conformance/fixtures.py` has 34 new cases: every new flag, `ml.limit.depth`, `ml.limit.nodes`, the out-of-range and conflict refusals) |
| WT2, WT3, WT5, WT6, WT7 | `test_wire_trees.py`: 27 tests (the task is the flag's; labels, probability, `--positive` and `eval` with both labels; the file's sections, header, labels record and `sum_order` 3; the value 2 and the others refused with `known` 3, a kind 4 file with 3 refused with `known` 2; a tree equals a forest of one; the default side learned from the training rows that miss, and right where none missed; an impute step; an all-missing column; the 2^20 rows: 1,048,577 refused with the tree's message, 1,048,576 fitted, a split whose train side is 1,048,598 refused and one within the bound fitted; the model file at 1, 2, 4, 8, 16 threads, four configurations) |
| WT4 oracle | `oracle_wire_tree.py --suite 40`: 160 cases (tree and forest, both tasks, 30 to 700 rows), **all byte for byte**: 73 with missing cells left in (13,129 cells), 19,084 nodes, 1,021 splits whose default side is left, 62 with a split, 129 with stored steps, 12 with a column that has no cell at all; `predict`, `predict --proba` on other rows with the columns reversed, and every `eval` metric (with both positive labels) equal the oracle's doubles. `--large`: 100,000 rows, tree and forest, both tasks, 8 cases, byte for byte, and the file at 1, 2, 4, 8, 16 threads equal to the oracle's. The oracle is sensitive: a file that says `sum_order` 2 differs (`test_the_oracle_is_sensitive_to_the_sum_order_field`) |
| WT8 scikit-learn | `scripts/gate_wire_sklearn.py forest tree` (the tolerances of section 9, unchanged): 14 comparisons, **all PASS**: forest accuracy distance 0 to 0.0222 (bound 0.03; wine, 45 test rows, one row), AUC 0 to 0.0005 (0.02), RMSE 2.85% on diabetes (5%); tree accuracy 0 to 0.04 (0.06; iris, 25 test rows, one row), AUC 0 to 0.0417 (0.06), RMSE 6.7% (12%). **The pre-registered control C1 (depth 1 against depth 8) stays inside the bounds on every set**: on these small, easy files a forest of 100 stumps is as good as one of depth 8 (diabetes RMSE 60.02 against 59.92), so C1 cannot make this gate fail and the section 9 sentence "reported as such" applies. C2 (a stump against the forest reference: outside on breast cancer, wine and diabetes) and C3 (the train target permuted: outside on all four, both models) were added after that first run, are labelled so in the script, and are what shows the gate can fail. No control separates the tree's RMSE bound alone (the stump is 6.5% away on diabetes). The library's red gate R1 (8.1% against 3%, `docs/models-tree.md` 10.8) stands; this 5% rule is a different rule on different data and does not replace it |
| WT9 admission | `scripts/admit.py tree` and `forest`, no `--fast`: **ADMITTED, 12 of 12 PASS each**; both manifests say `tier = "supported"`. Q6: `--threads` 1 to 16, both tasks, one file each (32,384 and 19,238 bytes for the tree, 2,141,047 and 1,377,693 for the forest of 100 trees on 1,500 rows); Q7: 4,692 and 10,635 damaged files; Q9: 60 oracle cases each; Q10: 156 patterns in two suites; Q11: 7 comparisons. `tests/modules/run.py`: the control admitted, the six mutant modules refused on the check each was written for (`mutant-kind-reuse` also fails Q1, as before). Q4 needed the `[sums] sequential` key (section 10 and `docs/modules.md`): without it the tree's two sequential float sums are rightly outside the lane order |
| WT10 mutants | `scripts/mutants_wire_trees.py` (`docs/bench/mutants_wire_trees.log`): **55 patterns, 51 killed, 4 survivors, all explained**, baseline passes: a second adapter-interface slot read from the wrong place (forest and tree), the task token, the proba mode, the labels record, the trailing bytes, the width, the plan/payload task, the sum order (three ways), the row bound (removed, off by one, counting the file, the message), the missing-cell rule in fit, predict and the held-out rows, the schema rule for an empty column, the bitmap (bit order, word, column, not passed at fit or predict), the thread count (minus one, reaching the seed). Survivors: `wt-regression-proba-allowed` and `wt-plan-key-not-checked-against-the-kind` (second lines of defence no input reaches), `wt-tree-threads-reach-the-seed` (a tree does not depend on the seed), `wt-threads-not-passed-to-the-fit` (no thread count changes a byte: the property itself). The first run had two unexplained survivors: one was a test that did not test (a payload with its labels record edited was refused for its length before the task check was reached; the test now rewrites the plan instead), one an equivalent mutant. `scripts/mutants_tree.py` (`docs/bench/mutants_tree_w3a.log`): 101 patterns, 95 killed, the same 6 survivors as before, the new reader mutant killed by the fuzz. Re-aimed patterns of earlier suites were re-run: `mutants_wire.py` all 40 (37 killed, 1 explained survivor; two patterns were re-aimed after that run and re-run: killed), `mutants_cli.py` the two re-aimed, `mutants_pipe.py order*`: killed. **`mutants_cli.py` (53) and `mutants_models.py` were not re-run in full** (their patterns all still apply: `--check`) |
| WT11 no trap | `test_fuzz_args.py`: 7 tests, the new flags in every mistake (the 8 bases, 12 flags, 27 values, in four spellings) and 1,250 seeded vectors; `test_wire_trees_file.py`: 13 tests: every truncation and flipped byte (2 masks) of three files, 1,500 seeded damages of every section and 1,200 of whole words of the model section with a repaired checksum, and the header, record headers, offsets, node fields and labels record one rule at a time: none traps, none with a wrong checksum is accepted. **It found a real trap in the library**: a class count of 2^63 - 1 made the payload reader add it before bounding it (exit -4); fixed in `lib/models/tree/model.cho`, corrected in place in `docs/models-tree.md` 10.6, mutant added |
| WT12 golden | `golden_numeric.json` and `golden_constant.json` unchanged and reproduced; `golden_planned.json` (60 cases of linear and logistic, written by the parent binary) reproduced byte for byte |
| the whole | `sh scripts/check.sh`: exit 0 (206 conformance tests in 259 s, none skipped) |

**What `ml` does now for a tree or forest, with a real session** (breast cancer and diabetes from `docs/examples/data`):

```
$ ml fit --model forest --task classification --trees 20 --seed 7 --threads 2 --target benign --output bc.cml breast_cancer_train.csv
  -> forest classification, 426 rows, 41,888 bytes; summary.forest {trees 20, nodes 630, bins 64, max_depth 8, min_leaf 1, mtry 5}, classes [0,1], positive 1
$ ml predict --model-file bc.cml --proba --limit 3 breast_cancer_test.csv      -> label, probability: (1, 1.0) (0, 0.0) (0, 0.0)
$ ml eval --model-file bc.cml breast_cancer_test.csv                           -> accuracy 0.937, precision 0.955, recall 0.944, f1 0.950, roc_auc 0.978, log_loss 0.366
$ ml fit --model tree --target progression --output d.cml diabetes_train.csv ; ml eval --model-file d.cml diabetes_test.csv   -> rmse 79.77, mae 62.49, r2 -0.28
```

(One run of `ml` on one machine, with `--trees 20` for the demonstration; the single deep tree overfits diabetes, as scikit-learn's does: 74.8.)

**The missing-value rule as built.** A cell still missing after the steps goes to the library with its bitmap, in fit, predict, eval and the held-out rows of a split fit, for `tree` and `forest`; the linear models refuse it as before. The default side of a split is learned from the training rows that miss at that node (ties right); a node that saw none sends it right. A column with no cell at all in a prediction file is the fitted column with every cell missing for a tree or forest (for the linear models it is still reported as a missing column, a behaviour of the reader the oracle found).

**What the gates do not claim.** The oracle is written by the same author from the same document, and its library part is compared with `lib/models/tree` by G-ident, not with another implementation; scikit-learn is the outside check, within bounds that the small data make loose, on a different algorithm. Only this Mac was used; the bytes at other machines are not tested. The forest of 1,024 trees on a large file can make a model file over the 16 MiB default of `--max-bytes`, which `predict` then refuses with `limit.input-too-large` (the flag raises it). The `ml.param.threads` memory ceiling has no fixture (section 6).

## 12. Open questions

1. **`ml.plan.model-not-built` still lists `tree` and `forest`** in its summary (introspect, skill, the rule table). Correcting it changes the stdout of the existing fixture `ml-plan-model-not-built`, which the golden record of `golden_numeric.json` pins byte for byte; the golden was kept, the text was not changed. The same for the message of `args.conflict` for a flag that needs a model other than constant ("--model linear or --model logistic"). The maintainer decides whether to re-record those two entries.
2. **README.md, docs/index.html and docs/status.md prose** still say "three models are reachable" and "a plan naming a tree is `ml.plan.model-not-built`"; only their generated blocks were regenerated (flags 42, limits 20, rules 74, and the `honest` demo, which now refuses `kmeans` because `forest` is built).
3. A column of a linear or logistic model's prediction file with no cell at all is reported as a missing column of the schema (`verbs.schema_check`); it is accepted for the tree and forest by a flag of that function. Whether the linear models should answer `ml.input.missing-value` instead is a change of an existing behaviour, left.
4. `--cat` is still refused for the tree and forest (`ml.input.categorical-not-encoded`): stages 2 and 4 of issue 21.
5. The seed of a forest and the seed of `--split` are one flag. A user who wants another bootstrap on the same split changes the split too.
6. The 2^20 row bound is the library's exactness bound; a three-limb comparison would raise it.

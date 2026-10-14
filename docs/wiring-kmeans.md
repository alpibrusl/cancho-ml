# Wiring k-means into `ml fit | predict | eval` (stage W3b of v1)

**Status: sections 1 to 11 (the gates, the command line, the verbs, the missing target, the plan, the model file, the refusals, the threads, the oracle, the
scikit-learn comparison, the admission) are written before any code of this stage; the first commit of the branch holds only this file. Section 12 holds the
results; a number or a verdict is in it only if a script of this repository printed it, with its conditions.** Tags as in `design.md`: **[choice]** a decision the
maintainer can overturn, **[read]** read from a document or the code, **[open]** not decided. A claim found false is corrected in place and marked.

The library of `docs/models-linear.md` section 3.3 (seeded initialisation, a fixed number of Lloyd iterations, the lowest index wins a tie, an empty cluster keeps its
centroid, the decisive sums in the package order) has a manifest and `wired = false`. This stage wires it **through the same door as linear, logistic, tree and
forest** (`docs/modules.md` is the contract, `docs/wiring.md` and `docs/wiring-trees.md` the flow): the manifest gains the flags, `scripts/gen_models.py`
writes the dispatch, the plan keys, the flag table and the schemas, and `fit` stores steps and model in one `CML1` file.

k-means differs from the four models wired before in one respect that shapes the stage: **it is unsupervised.** A fit has no target column, so `--target` is not
needed, the model file has a schema without a target, and `predict` and `eval` need no column but the features. Everything else (steps, categorical encoders, the
string table, `--split`, `--threads`, `--seed`, the file reader's limits) is the existing machinery.

## 1. The gates (fixed before the code; each can fail, section 12 shows it failing)

| id | gate | judged by | a mutant it must kill |
|---|---|---|---|
| WK1 | **Tables from the manifest.** The flags `--k`, `--iterations`, `--init` and `--distance` (section 2) are data in `lib/models/kmeans/model.toml` and the verb tables; `gen_models.py --check`, `schemas.py --check` and the table gate T1 are clean; every new flag is reached by a fixture and every new rule by one with its exit code; the `--model` choice list is the six wired models | `gen_models.py --check`, `test_tables.py` | a flag with no fixture; a range edited in the generated file only |
| WK2 | **No target.** `fit --model kmeans` needs no `--target`; `--target` with k-means is `args.conflict`; every other model still needs it (`args.required-flag`); a plan file whose model line is k-means is treated the same way after the plan is read | `test_wire_kmeans.py` | the requirement dropped for every model; the conflict dropped |
| WK3 | **The verbs.** `fit` prints k, the iterations, the initialisation, the inertia and the cluster sizes; `predict` prints the cluster of each row (column `cluster`), and with `--distance` the squared distance to its centre (column `distance`); `eval` prints the inertia, the cluster sizes and the mean distance of each cluster; the task is `clustering` in the answer, in `est` and in `introspect` | `test_wire_kmeans.py`, oracle | a predict column in the wrong place; the distance of the second nearest centre; the inertia of the training rows printed for a held-out file |
| WK4 | **The model file, kind 6.** `fit` writes one `CML1` file whose `model` section is the payload of `docs/models-linear.md` section 3.4 (`u32 kind = 6`, one `f64` record `centroids`, rank 2, `k x d`) and whose schema has no target; the steps are stored and applied by `predict`/`eval`; `sum_order` is 2 (section 6); a file is read by an independent Python reader | `test_wire_kmeans.py`, `oracle_wire_kmeans.py` | the target index written as 0; the sum order 3 written |
| WK5 | **Oracle, bit for bit.** The Python oracle (`scripts/oracle_models.py` for the library payload, `scripts/oracle_wire_kmeans.py` for the file) writes the payload and the whole file, header, schema, stored steps, meta and trailer included, and they equal what `ml` wrote, on generated data with missing cells imputed, constant columns, scaled, bucketed columns, `k` from 1 to 12, both initialisations, at 1,000 and 100,000 rows; `predict` (clusters and distances) and `eval` (inertia, sizes, mean distances) equal the oracle's doubles | `test_wire_kmeans_oracle.py`, `oracle_wire_kmeans.py --suite` | a step applied in the other order; the tie rule; the lanes of the inertia |
| WK6 | **Missing cells are refused.** A matrix cell left missing after the steps is `ml.input.missing-value` with the `impute` repair, in `fit`, `predict`, `eval` and the held-out rows of a split fit, as for linear and logistic | `test_wire_kmeans.py` | the refusal dropped; the tree's acceptance given to k-means |
| WK7 | **Refusals reachable from `ml`, before any allocation sized by the data.** `--k` outside 1 to 1,024 (`ml.param.out-of-range` from the flag check), `k` above the training rows, `k * features + k + 1` above 131,072, `(iterations + 1) * (k * features + k + 1)` above 2^24 (the diagnostics the library asks for), `--max-rows` (`ml.input.too-many-rows`), `--threads` (`ml.param.threads`), `ml.plan.missing-key` for a fit with no `--k` | `test_tables.py`, `test_wire_kmeans.py` | a bound removed or moved after the allocation |
| WK8 | **Threads change no byte.** `fit --threads t` for t in 1, 2, 4, 8, 16 (1 to 16 in `admit`) writes one file and prints one answer, both initialisations, with a split and with steps | `test_determinism.py`, `admit.py` Q6 | a thread count that reaches the sums (`mutant-threads-one-count`, `-eleven`) |
| WK9 | **scikit-learn, with the initial centres fixed** (section 10): centroids and inertia equal within the tolerances fixed there; the seeded initialisations of the two programs are not compared | `scripts/gate_wire_sklearn.py kmeans` | a tolerance exceeded; a deliberately wrong control |
| WK10 | **Admission.** `scripts/admit.py kmeans`: Q0 to Q11 all PASS for `supported`, else the manifest says `experimental` and section 12 lists the failing checks; the six mutant modules are still refused and the control is admitted | `admit.py`, `tests/modules/run.py` | the six mutant modules |
| WK11 | **Mutants of the wiring code.** Patterns on the flag mapping, the optional target, the task value, the predict column, the distances, the inertia, the kind 6 reader, the bounds of `k`, the choice of the initialisation and the held-out scoring: all killed, or each survivor explained | `scripts/mutants_wire_kmeans.py` | itself |
| WK12 | **No trap.** The argument fuzz includes the new flags; damaged kind 6 files (every truncation, flipped bytes with and without a repaired checksum, seeded damage of the plan, schema, steps and model sections) are refused with a tag | `test_fuzz_args.py`, `test_wire_kmeans_file.py`, `admit.py` Q7 | a removed bound of the payload reader |
| WK13 | **Golden.** Every earlier fixture and file is unchanged but where its text lists the models or the verbs; those are re-recorded in one commit with the reason; `golden_constant.json`, `golden_numeric.json` and `golden_planned.json` (the model files of linear and logistic) reproduce | `test_wire_baseline.py` and the conformance suite | a changed field in `file.encode` for kinds 4, 5 and 7 |
| WK14 | **Honest documents.** The README flow and the pages show real output (`gen:` blocks); the status page says six models are reachable | `scripts/site.py --check`, `check_prose.py`, `check_links.py` | a stale block |

## 2. The command line

New flags, generated from the manifest or the verb tables (`ml introspect | jq .flags` is the table; this is for reading). `--k`, `--iterations` and `--init` are `fit`
flags; `--distance` is a `predict` flag.

| flag | meaning | range, default |
|---|---|---|
| `--k N` | the number of clusters; required for a k-means fit (a fit without it is `ml.plan.missing-key`, as a plan with no `k=`) | 1 to 1,024 |
| `--iterations N` | the Lloyd iterations: a fixed number, with no stopping rule; the library runs `N` updates and one final assignment pass. The flag is shared with `logistic` (its Newton iterations) | 1 to 1,000, 10 for k-means |
| `--init NAME` | `kmeans++` (the first centre is a seeded row; each next one is drawn with probability proportional to the squared distance to the nearest centre chosen) or `random` (seeded rows without repetition); both are drawn from the stream of kind 3 of `--seed` | `kmeans++` or `random`, `kmeans++` |
| `--distance` | `predict`: add the `distance` column, the squared Euclidean distance of the row to the centre of its cluster; refused with any other model | |

The other flags are the existing ones: `--seed N` (the seed of the initialisation **and** of `--split`: one seed, stored in `meta`; the library derives its stream with kind 3
and the split uses kind 4, so the two never share a stream), `--threads N` (the threads of the fit; `predict` and `eval` are not threaded), `--split`, `--step`, `--plan`,
`--drop`, `--cat`. A plan names the model on its model line, `model kind=kmeans k=3 iterations=10 init=kmeans++`; the canonical text always lists every key in the order of
the manifest (`k`, `iterations`, `init`).

An enum hyperparameter other than `task` has a flag of its own, generated from the manifest (`--init`, kind `choice:random/kmeans++`; an unknown value is `args.bad-value` from the
flag parser). A flag that two models share (`iterations`) must agree on type and range and may differ in default; the flag table's default is documentation (an absent flag
leaves the key out of the model line and the plan parser writes the model's own default).

**Flags k-means refuses (`args.conflict`)** [choice]: `--target` (k-means has no target; the message says so), `--task` (the task is fixed), `--positive` (no label), `--proba`
(no probability; `--distance` is the second column), and `--l2`, `--trees` and the other hyperparameters of the other models (`est.takes`, as for every model).
`--distance` with any other model, and `--k`, `--init` with a model that has no such key, are `args.conflict` too. **`--target` is optional for k-means only**: every other
model still needs it (`args.required-flag`). With `--plan` the flag check cannot know the model, so it defers: after the plan is read, `--target` with a k-means plan is
`args.conflict` and a missing `--target` with any other plan is `args.required-flag`, the same rules at a later point.

The texts that name the reachable models (`ml.plan.model-not-built`, the `args.conflict` message of a flag that needs a model other than constant, the `--model` help,
`introspect`) name the six models.

## 3. The verbs [choice]

* **fit.** The summary has `clusters`: `{k, iterations, init, rows, metrics}`, where `metrics` is `{inertia, sizes, mean_distance}` of the training rows (after the steps), next to
  `plan`, `steps`, `features`. `inertia` is the decisive sum of the squared distances of the rows to their centres in the package order (section 6); `sizes` the rows per cluster;
  `mean_distance` the mean Euclidean distance of the rows of each cluster to its centre, `null` for an empty cluster. With `--split train=F` the answer adds `split` and `holdout`
  (`{rows, metrics}`): the same three numbers on the held-out rows, scored through the stored steps and the fitted centroids. `target` is `null`; `task` is `clustering`.
* **predict.** `columns` is `["cluster"]`, or `["cluster", "distance"]` with `--distance`; a row of `rows` is the cluster index (0 to k-1, the lowest index among equal
  distances) and the squared distance as the shortest decimal that reads back. The file needs the feature columns only; a column that is not a feature is `ml.schema.mismatch`
  with the `--drop` repair, as for every model. `unknown_categories` as for the others.
* **eval.** `metrics`: `inertia` (the sum of the squared distances of the rows to the centre of their cluster, a double), `sizes` (rows per cluster), `mean_distance` (per
  cluster, `null` for a cluster with no row). `rows` and `clusters` beside it. `--split train=F --seed N` scores the test rows only, as for the other models: the inertia of
  held-out rows. The file needs no target column. There is no label, so there is no accuracy: k-means is not scored against a target.
* The **decisive sums** are the lane order of `docs/linalg.md` section 2: the inertia is the sum of the squared distances of the rows in row order, and the mean distance of a
  cluster is the sum of the Euclidean distances of its rows, **in the order of those rows in the file**, divided by their number; both through `lib/metrics/sums.cho`
  (blocks of 256 terms, four lanes, the fixed tree). A squared distance is the left fold over the features of `(x_j - c_j)^2` from `+0.0`, as the library forms it. The
  inertia of the training rows equals the library's inertia of the final pass bit for bit (the same terms in the same order; section 9 compares it with the oracle).

## 4. A fit without a target [choice]

The reader takes the target column by name; with no name the target column is -1 and every kept column is a feature. The schema section is
`u32 columns | u32 target | per column ...` (`docs/pipeline.md` section 3.1) and **the target index of a model without a target is the number of columns**: no column has
that index, so every column is a feature, and `schema.decode` accepts a target index up to the number of columns where it accepted one below it. The file reader requires
the pair to agree with the model: a kind 6 file whose schema has a target, and a file of any other kind whose schema has none, is `ml.model.malformed` (section 2 of the file).
`meta` holds the number of columns in the place of `features + 1`. `--drop` of the label column is how a labelled file is used (the iris example of the README drops
`virginica`). The `unknown-column` refusal for a missing target does not apply; `--target` is `args.conflict`.

## 5. The plan, the steps and missing cells [choice]

Numeric steps (`impute`, `scale`, `bucket`) and the categorical encoders (`encode`, `freq`, `hash`; `--cat`) work as for the other models: fitted on the training rows,
stored, applied by `predict` and `eval`. A feature cell that is still missing after the steps is **refused**: `ml.input.missing-value`, with the `impute` repair of
`docs/wiring.md` section 3, in `fit`, `predict`, `eval` and the held-out rows of a split fit. The distance to a centroid is not defined for a missing coordinate and the
library has no rule for one; a rule would be a choice that moves clusters. k-means is sensitive to the scale of the features, which the README flow handles with a `scale` step; the
library does not scale.

## 6. The model file [choice]

`model` payload of kind 6: `u32 kind = 6`, then the record `centroids` (`u32` dtype 2, `u32` rank 2, `u64` shape `[k, d]`, then `k * d` little-endian doubles, row-major),
`docs/models-linear.md` section 3.4; `payload_version` 1. The reader is the blob adapter's `decode_blob` (`docs/modules.md` section 4b): it reads the header, computes the size
the payload must have **before it allocates**, requires dtype 2, rank 2, `1 <= k <= 1,024`, `d` equal to the width the steps leave, `k * d + k + 1 <= 131,072`, every
double finite, and no byte after the record. Every failure is `ml.model.malformed` with `section` 4. The plan's `k` must equal the payload's (a file that disagrees with
itself is malformed); the other hyperparameters of the plan (`iterations`, `init`) describe the fit and are not checked, as `l2` is not.

**The summation order field.** `lib/pipeline/file.cho` writes `u32 sum_order = 2` at the end of `meta` for every kind from 4 on, meaning the 4-lane order of `docs/linalg.md`
section 2. **A k-means file records 2**: its decisive sums, the sums of the Lloyd update (per cluster and feature, the counts, the inertia) and the metrics above, are the lane
order. One sum is not: the k-means++ initialisation forms the total of the nearest squared distances and the running comparison as a left fold in row order, on one thread,
by design (`docs/models-linear.md` 3.3); it picks the initial rows and is a sequential sum that no thread count and no lane layout changes. The field says what the decisive
sums of the fit are; it does not say there is no sequential sum. A reader that knows the order 3 only refuses a kind 6 file with `ml.model.version`, as for any kind 4 to 6 file.

## 7. Refusals [choice]

| tag | exit | reached by | where it is checked |
|---|---|---|---|
| `args.conflict` | 2 | `--target`, `--task`, `--positive`, `--proba` with k-means; `--distance` with another model | `wire.check_flags`, and after the plan is read |
| `args.required-flag` | 2 | `--target` missing for another model, with `--plan` too | `main.check_flags`, after the plan is read |
| `ml.param.out-of-range` | 2 | `--k` 0 or above 1,024 or `--iterations` 0 or above 1,000 (the nat flags); `k` above the training rows; `k * features + k + 1` above 131,072; `(iterations + 1) * (k * features + k + 1)` above 2^24 floats | the flag check; the adapter, before the allocation of the labels and the diagnostics |
| `ml.plan.missing-key` | 8 | no `--k` (a plan with no `k=`) | the plan parser |
| `ml.input.too-many-rows` | 8 | more data rows than `--max-rows` | the reader, before the matrix is built |
| `ml.input.missing-value` | 8 | a feature cell still missing after the steps | the wiring, with the repair |
| `ml.input.non-finite` | 8 | a cell that is not finite (the reader refuses it already), a sum that overflows is `ml.fit.non-finite` | the reader; the library |
| `ml.fit.non-finite` | 8 | sums that overflow a double (cells of magnitude near 1e308) | the library |
| `ml.param.threads` | 2 | `--threads 0`, `--threads 65` | the flag check |
| `ml.model.truncated`, `ml.model.malformed`, `ml.model.version` | 8 | damaged files (WK12), a schema whose target disagrees with the kind, a `k` the plan and the payload disagree on | the file reader |

`ml.plan.model-not-built` is left for `constant`, which takes no plan.

## 8. Threads

`--threads N` is the library's `threads`: the Lloyd passes are cut into aligned chunks of whole blocks and combined by the fixed tree of `docs/linalg.md` section 2, so the
model is the same at every thread count (`docs/models-linear.md` G-thread compares the library at 1, 2, 3, 4, 7, 8, 16). WK8 compares **the model file** of `ml` at 1, 2, 4, 8, 16
(and `admit` at 1 to 16), both initialisations, with a split and with steps. `predict` and `eval` run with one thread; the metrics of section 3 are single-threaded sums of a
fixed order. A threaded fit copies the rows each owns at every pass (`docs/models-linear.md` section 5): the memory is about twice the training matrix, and the threaded fit is
for determinism, not for speed on a memory-bound shape.

## 9. The oracle

`scripts/oracle_models.py` reproduces the library (`kmeans_fit`: the initialisation, the lane sums, the update rule, the labels) with numpy performing the same operations in the
same order. This stage adds to it the **payload** (the record of section 6 in Python with `struct`) and imports it from `scripts/oracle_wire_kmeans.py` together with the helpers of
`scripts/oracle_wire.py` (CSV types, schema, step records, meta, trailer, the transforms of `oracle_pipe.py`, the split of `oracle_metrics.py`). The oracle fits the same pipeline and
the same model as `ml fit`, writes the file, and the test compares it with the file of `ml` byte for byte; then `predict` (clusters and distances), `eval` (inertia, sizes, mean
distances, with and without a split) on the training rows and on other rows with the columns reversed, as doubles. The generated data have missing cells (imputed by a step: about
half the cases), constant columns, int, decimal and float columns, `impute`, `scale` and `bucket` steps, `k` from 1 to 12, both initialisations, an iteration count from 1 to 20
and a split; 1,000 rows and 100,000 rows (the latter in a dedicated test and in `admit`'s large case). **What this does not show**: the oracle is written by the same author from
the same document, and its library part is compared with `lib/models/kmeans` by `docs/models-linear.md` G-oracle; the oracle checks the wiring (flags, steps, absence of a target,
file, metrics), not the algorithm against the world.

## 10. The scikit-learn comparison: the initial centres are fixed [choice], tolerances fixed here, before the run

The two programs draw their initial centres differently (`ml` from its stream, scikit-learn from its generator), so **a comparison with each model's own seeded initialisation
compares two random choices, not two implementations, and is not made.** The gate gives scikit-learn the initial centres of `ml`:
`KMeans(n_clusters=k, init=<the initial centres as an array>, n_init=1, algorithm='lloyd', max_iter=<iterations>, tol=0)` on the same train file after the same steps (standard
scaling with `ddof` 0 on both sides; the initial centres are the rows `ml` chooses, taken from the oracle, which WK5 shows equal to the rows `ml` chooses). Data: the shipped
iris, wine and breast-cancer train files of `docs/examples/data` without their label, k = 3, 3 and 2, 10 iterations, both initialisations, and a seeded blob set of 2,000 rows. Fixed in advance:

| quantity | bound |
|---|---|
| inertia, relative | 1e-9 (the bound of the library's gate) |
| centroids, the largest absolute difference over the largest absolute centroid value | 1e-9 |
| the cluster of every row | equal on every row (a row within 1e-9 relative of equidistant is reported with its margin and is the one allowed exception) |

scikit-learn moves the farthest point to an empty cluster and `ml` keeps the centroid (`docs/models-linear.md` 3.3): a configuration where a cluster empties is outside the gate and is
reported as such; the configurations above have none and the script checks it. The control is a deliberately wrong run (one iteration fewer on the scikit-learn side) which must fall outside the bound on at least one data set, or the gate cannot fail and is reported as such. If a bound is exceeded the gate fails and is reported as it is; the bounds are not widened after the run.

## 11. Admission and the module contract

* k-means is a **blob** model (`docs/modules.md` section 4b): the size of the payload is `4 + record(k, d)` and `k` is a hyperparameter, while the vector interface asks for `out_len(d)`
  with no hyperparameter. `shape = "blob"` is allowed with `task = "clustering"` besides `both`; the generator's rule that a blob model has a `task` enum is for `both` only.
* The four functions of the blob interface keep their signatures; for a clustering model their results are: `predict_blob` the cluster of each row (the integer as a double), `proba_blob`
  **the squared distance to the centre of that cluster** (the second output of the model: the probability of the positive class for a classification, a refusal for a regression, the squared
  distance for a clustering; the name is the interface's), `decode_blob` the code, `k` in `c0`, 0 in `c1` and 4 in `end` (the task: 1 regression, 2 classification, 4 clustering; the
  tree and forest keep 1 and 2, and `est.task` answers 3 for "either"). `fit_blob` takes an empty target slice.
* The generated `est` gains `takes_missing(kind)` (a blob model of both tasks: the tree and the forest) beside `is_blob`, so that the missing-cell refusal is a fact of the manifest.
  `est.task` answers 4 for clustering. `hparam_flag`, `takes`, `hparams` and `hparam_min` include enum hyperparameters other than `task`.
* The adapter is `lib/models/kmeans/adapter.cho` (module `models.adapter_kmeans`). It calls `models.kmeans.fit` and `models.kmeans.predict`, forms the squared distance of a row to its
  centre with `models.kpass.dist2` (the left fold the library uses), and has no sum of its own; the files that sum in the package order are `kpass.cho` and `lib/metrics/sums.cho`.
* `admit.py` is adapted to a clustering model: Q6 and Q7 fit without a target, Q9 runs the oracle with `scripts/oracle_wire_kmeans.py`, Q11 runs the comparison of section 10.
  `tier = "supported"` only if all twelve pass; otherwise the manifest stays `experimental` and section 12 names the failing checks. The six mutant modules of `tests/modules/` must still
  be refused and the control (a twin of `linear`) admitted.

## 12. Results

(Filled when the gates have been run.)

## 13. Open questions

(Filled with the results.)

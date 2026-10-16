# cancho-ml design (M1) — DRAFT, sections 1 to 8 and 10; sections 9 and 11 wait for M0

**Status: draft for the maintainer's review. Not code.** Every claim is tagged:

* **[read]** read from a document of the compiler (`cancho` `a4572ea` when the design was written; the pin is `0207556`) or of cancho-table; not re-measured here.
* **[choice]** a decision, with the alternative named; the maintainer can overturn it.
* **[open]** not decided; the section says what decides it.
* **[M0]** will be settled by a number from `docs/spikes.md`; nothing is claimed until it is in that file.

## 1. Scope

Classical ML on **numeric tabular data held in memory**: transforms, a pipeline, training, prediction, evaluation. Not deep
learning, not sparse data, not streaming training, not out-of-core. Every model has a size limit and a refusal when the data
exceeds it (section 8).

**v1 models [choice]:** linear and logistic regression, k-means, decision trees, random forests. Boosting, naive Bayes and
k-NN stay in the plan (M8) and are not in v1 unless a measured need appears; the alternative, building them with M7, spends
effort before the pipeline and the format have been proved.

**Classification targets [open]:** binary only, or binary plus multiclass? Multiclass touches the label encoding, the model
bytes (one weight vector per class or a class-count in every leaf), the metrics (macro and micro averages, a multiclass
log-loss) and the refusal set. Recommendation: binary in M2 to M6 and the format reserves a class count, so that adding
multiclass in M7 changes no existing byte. Needs the maintainer's yes.

## 2. What is promised, in tiers

| tier | promise | where it can fail | status |
|---|---|---|---|
| D1 | same data, seed and plan give the same model and prediction bytes at **any thread count** | any parallel float sum added in thread order | **[M0]** which of exact accumulators or a fixed reduction tree |
| D2 | the same bytes on Mac aarch64 and Linux x86-64 | a platform `libm` call; `fma` contraction; a different `exp` | **[read]** `float` is IEEE binary64, no reassociation, no `fma` contraction (`floating-point.md`, `compile-time.md`); `exp`/`log` are cancho code in `std/math.cho`, not libm (`float-math.md` §9). **[M0]** G-det checks it on both machines |
| D3 | a stored model predicts the same bytes forever, across versions of the tool | a format change | section 4: every format has a version; a reader refuses a version it does not know (`ml.model.version`) |

If D1 cannot be kept for a feature, the feature is refused or the exception is written next to it with a test
(`CLAUDE.md`).

## 3. The numeric core

* **Floats are `float` (binary64).** No `f32` in the model (the compiler has `f32` since edition 6, `f32.md`; it is not used
  here: a second width is a second rounding story).
* **A matrix is row-major, `rows * cols` doubles in one array, with `rows` and `cols` limits** (section 8). Missing values are
  a separate per-column bitmap, not NaN **[choice]**: `fmin`/`fmax` treat NaN as missing but `==` and `<` do not, and a sort
  over NaN gives a garbage order (`floating-point.md` §5), so NaN is refused on input (`ml.input.non-finite`) and imputation
  works from the bitmap.
* **Reading.** Cells go through cancho-table's exact reader (`:int`, `:dec(S)`, `:float`, correctly rounded). *(Corrected in place: this
  bullet said the reader of 15 to 17 digit doubles, `tools/table/flt.cho`, was on a branch and **not merged**. It is merged: cancho-table PR #32,
  `main` at `7f36608`. What `ml` does with it and how it is pinned (five `[dependencies.*]` at cancho-table `202b214`) is `docs/reader.md`.)*
* **No `float_of_bits`.** cancho has `bits_of` and no inverse; it is "missing on purpose" (`floating-point.md` §4.1)
  **[read]**. A stored double is therefore rebuilt from integers: mantissa as `float_of`, then `ldexp` from `std/math.cho`,
  which is exact for normal numbers. The cost of that load is **[M0]** (it is outside the timed fit but inside `predict`
  start-up, so it matters for M10).
* **`exp`, `log`, `sqrt`.** `sqrt` is a builtin; `exp` and `log` are `std/math.cho`, 1 ulp from glibc on the ranges measured
  there **[read]**. The tool uses them as they are, so its bytes are the compiler's, not the platform's. A change to those
  functions changes model bytes, so the model records the compiler commit it was built with **[choice]** (a header field,
  informational: a different commit is not refused, because only a changed result matters, and that is tested by the
  corpus in M10).

### 3.1 Linear algebra and tensors: a cancho package, rank-N from the start **[choice]**

Linear algebra is a **cancho package** (working name `cancho-linalg`), not a language feature and not an external library
(BLAS, via FFI). Reasons: a BLAS picks kernels per CPU, may use `fma` and splits by thread count, so it cannot keep D1/D2; and
FFI is unbounded in the authority report (lexsys-gpu measured `ffi("liblexgpu")` as unbounded, cancho#251 **[read]**). Spikes
L and S point the same way: the 4.4x of spike L came from library techniques (a row read through a subslice, 8 rows in flight,
a faster `exp`), none of which needs array syntax; what remains is compiler-level (bounds checks, no SIMD). Language changes are
asked for upstream only with a measured reproducer (section 11).

**The type is a tensor descriptor, with matrices as the rank-2 case.** `offset`, `rank`, `shape[0..rank)`, `strides[0..rank)`,
all plain `int`s; the elements live in a separate `[float]` that every kernel takes beside the descriptor. It is a descriptor and
not a view that holds the data because cancho structs cannot hold reference fields, as reported by lexsys-gpu **[read,
cancho#251; not re-checked at `0207556`]**. Consequences:

* `transpose`, `reshape` (when contiguous), slicing, `row`/`column` and broadcasting (stride 0) are descriptor edits: no copy,
  no kernel. A matrix is `rank = 2`, row-major strides `(cols, 1)`.
* **Rank is bounded** at `R_MAX = 4` **[choice, estimate]**; a larger rank is a refusal (`ml.tensor.rank`). The descriptor's
  size is fixed by `R_MAX`, so adding rank 3 or 4 later changes no existing function, descriptor layout, or stored byte.
* v1 ships rank 1 and 2 kernels only (`dot`, `axpy`, `gemv`, blocked `gemm`, `syrk`, Cholesky, triangular solve). Rank 3 and 4
  arrive with a contraction `contract(a, b, axes)` whose order is the same normative rule as `gemm`'s, when a model needs it
  (small exported networks, M10 and after), not before.
* **Dtype.** `float` (binary64) only in v1. `f32` exists from edition 6; whether the package is generic over it is **[open]**
  and waits for a model that wants it, since a second width is a second set of rounding rules (section 3).

**Every reduction has a normative summation order, and it is part of the API.** A reduction over an axis adds in ascending index
order inside a block of a fixed length `K`; block partials are combined in a fixed binary tree by block index. `K` and the tree
shape are constants of the package, never functions of the thread count. This is the fixed-reduction arm C of `docs/spikes.md` (section 15),
or its exact-accumulator alternative; which one is **[M0]**. Each kernel's document says its order, and its test compares it to
a reference written in that order (as spikes L and S do with numpy). Threading only decides which thread runs which block.
**Decided at `v1-lanes` (section 17):** inside a block the order is 4 lanes (term `i` to lane `i mod 4`, lanes combined as
`(l0 + l1) + (l2 + l3)`), `K = 256`, then the tree; the definition is `docs/linalg.md` section 2.

**Where it lives.** A module in cancho-ml first; extracted to its own repository when a second consumer needs it. The
descriptor has no ML types in it, so tensors for other programs (and later a cancho-gpu backend, whose tile kernels would take
the same descriptor) do not require a redesign.

## 4. File formats

Both are **bytes, versioned, with no hidden state**: no timestamp, no host name, no path, no random identifier. Two fits of
the same input produce the same file.

### 4.1 Model file **[choice]**

```
magic        4 bytes   "CML1"
version      u32 LE    1
sections     u32 LE    count, then per section: tag u32 LE, length u64 LE, payload
```

* all integers little-endian; a double is its 8 bytes (the bits), little-endian; no padding;
* **every numeric parameter block is a tensor record**: dtype tag, rank, shape (rank `u64`s), then the elements row-major,
  contiguous. A coefficient vector is rank 1, a weight matrix rank 2, a future kernel rank 4: the format already has the
  field, so tensors add no format version (section 3.1);
* sections in a fixed order: `plan` (the canonical pipeline text, section 5), `schema` (column names, types, order),
  `steps` (each transform's fitted parameters), `model` (the estimator), `meta` (seed, row and column counts, hyperparameters,
  compiler commit), then a **32-byte SHA-256 trailer over every byte before it** *(corrected in place: the first draft put the
  digest in `meta`, "a SHA-256 of all sections before it", which cannot cover `meta` itself or the section headers; a trailer
  covers the whole file)*;
* a reader checks magic, version, section tags, every length against the file size and against the section limits, and the
  SHA-256, and refuses with a tag from section 8 before it allocates anything sized by the file. The order is bad-magic, version,
  truncated (the structure), checksum, then malformed (every field against its bound).

**Built in M9's slice (`tools/ml/mfile.cho`, byte layout in the header of that file and in `docs/cli.md` section 4):** five sections,
tags 1 to 5, `schema` = `u32` columns, `u32` target, per column `u32` name length, name, `u32` type (0 int, 1 dec), `u32` scale; `steps` =
`u32` 0; `model` = `u32` kind (1 constant regression, 2 constant classification) and a tensor record (`u32` dtype 1 = i64, `u32`
rank 1, `u64` shape, `i64` elements); `meta` = `u64` rows, `u64` columns, `u64` seed 0, `u32` hyperparameters 0, `u32` length and the
compiler pin. The layout is parsed by an independent reader in `tests/conformance/test_model_file.py`, and a corrupted-file fuzz
(every truncation, every byte flipped three ways, seeded damage with and without a repaired checksum) never traps.

**The sum order in `meta` (rule written at `v1-post`, before the code; the proposal of section 17).** The arithmetic of a model of kind 4 to 7
(linear, logistic, k-means, tree or forest) depends on the summation order of `docs/linalg.md` section 2, so the file says which order made it. The
rule is a function of the model kind, which a reader knows before it reads `meta` (the `model` section comes first):

* **kinds 1 to 3** (constants: exact integer and `facc` arithmetic, no float order): the `meta` payload is exactly what it was, `u64 rows | u64 columns | u64 seed
  | u32 hyperparameters | u32 length | compiler pin`. **Their bytes do not change**, and a kind 1 to 3 file whose `meta` is longer than `32 + length` is
  `ml.model.malformed`, as before.
* **kinds 4 and above** (every kind whose row of the table below says so; today 4 to 7): the payload is that, then **`u32 sum_order`**, the last four bytes of `meta`:
  `meta` is exactly `36 + length` bytes, else `ml.model.malformed` (a `meta` of any other length is not looked at for the order: the ordinary field checks refuse it). `sum_order` = **2** for the order of `docs/linalg.md` section 2
  (`K = 256`, `L = 4`, `(l0 + l1) + (l2 + l3)`, the fixed tree, panels of 32). **1** is the order before `v1-lanes` (one chain per block): retired,
  never written by a model file, refused if found. **3 is the order of kind 7** (a tree or a forest: every float sum is a sequential left fold in a fixed row or tree order and no lane sum is made, `docs/models-tree.md` 3.2; `docs/wiring-trees.md` section 4): a reader requires exactly the value of the kind, 2 for kinds 4 to 6 and 3 for kind 7. **Kind 6 (k-means) records 2**: its decisive sums, the sums of the Lloyd update and the metrics of `eval`, are the lane order; the one sum that is not is the k-means++ total of the initialisation, a left fold in row order on one thread that picks the initial rows (`docs/wiring-kmeans.md` section 6). 0 and 4 and above are unassigned. No flag and no version bump: the file `version` covers the layout, and the layout
  of a kind is fixed by the kind.
* **A reader refuses a file whose `sum_order` it does not implement with `ml.model.version`** (detail `sum_order` found, `known` the order of the kind: 2, or 3 for kind 7): it cannot reproduce the
  parameters' arithmetic, and the same refusal tag as an unknown format version says "refit with this version". The order of the checks is bad-magic, version,
  truncated, checksum, **sum order (version)**, then malformed: the field is inside a checksummed section, so it is read after the checksum, and before the other fields,
  so a file of another order is never reported as malformed because its parameters look odd. A reader that does not build a kind (today `ml`, which builds 1 to 3) still reads
  the sum order of a kind 4 to 7 file first: a wrong order is `ml.model.version`, a right one is `ml.model.malformed` (section 4: "not built").
* Writers: `lib/pipeline/file.cho` writes the field for the kinds it is given (4 and above); `tools/ml/mfile.cho` writes kinds 1 to 3 only and so never writes it. Tests:
  `tests/conformance/test_model_file.py` (an independent reader and writer of the field; a file of kind 4 to 7 with order 0, 1, 3 and `0xFFFFFFFF` is `ml.model.version`, a
  kind 1 to 3 file with the extra four bytes is malformed), `tests/t_pipeline.cho`, `t_pipeline_fuzz.cho`, `oracle_pipe.py --file`.

**Registry of the numbers in the file (added at integration; one table, a number is never reused).** `lib/core/kinds.cho` holds the
model kinds and `tests/t_registry.cho` and `scripts/check_registry.py` fail if two stages claim one number or a stage's own constant
differs from this table. Model kinds (the `u32` that opens the payload of the `model` section):

| kind | model | records after the kind | owner | in `mfile.cho`? |
|---|---|---|---|---|
| 1 | constant regression of an int or dec target | one i64 record, shape [4] | CLI slice | yes |
| 2 | constant classification | one i64 record, shape [6] | CLI slice | yes |
| 3 | constant regression of a float target | one i64 record, shape [2] | reader stage | yes |
| 4 | linear regression | `coef` f64 rank 1 `d`; `intercept` f64 rank 1 `1` | `lib/models/record.cho` | no (`pipeline.file`, since W1) |
| 5 | logistic regression | the same two records, then `labels` i64 rank 1 `2`: the negative and the positive label (payload version 2, W1) | `lib/models/record.cho` | no (`pipeline.file`, since W1) |
| 6 | k-means | `centroids` f64 rank 2 `k x d`; the schema section of the file has no target (its target index is the number of columns) | `lib/models/record.cho`, `lib/models/kmeans/adapter.cho` | no (`pipeline.file`, `docs/wiring-kmeans.md` section 6) |
| 7 | decision tree or random forest | ten records, i64 and f64, rank 1 (`docs/models-tree.md` section 7); the task (classification or regression) is a field of the first | `lib/models/tree/model.cho` | no |

**Step kinds** (the `u32` that opens one step in the `steps` section; `lib/core/kinds.cho` `step_*`, `docs/transforms.md` section 4; the `steps` payload is `u32 n` then `n` times `u64 length | step`, and `n = 0` is the four bytes `mfile.cho` writes):

| step kind | transform | plan line |
|---|---|---|
| 1 | impute (mean, median, constant) | `impute` |
| 2 | scale, min-max | `scale kind=minmax` |
| 3 | scale, standard (mean and std, `ddof` 0 or 1) | `scale kind=standard` |
| 4 | one-hot encoding | `encode kind=onehot` |
| 5 | label encoding | `encode kind=label` |
| 6 | bucket (width or quantile edges) | `bucket` |
| 7 | one-hot encoding of a categorical column | `encode kind=onehot` on a categorical column |
| 8 | label encoding of a categorical column | `encode kind=label` on a categorical column |
| 9 | frequency encoding of a categorical column | `freq` |
| 10 | hashing of a categorical column | `hash` |

**Section tags** (`lib/core/kinds.cho` `section_*`): 1 plan, 2 schema, 3 steps, 4 model, 5 meta, **6 strings (the string table, present only when a step of kind 7 to 10 needs it)** and **7 provenance (the SHA-256 and size of the training file, its rows and the range of each numeric feature: `docs/tracking.md` section 2; present in every file `fit` of this version writes, absent from the files of older versions, which are read and say `provenance: absent`)**. The header's `sections` is 5, 6 or 7 and does not say which of the two optional sections a count of 6 holds: a reader walks the tags (each optional tag is greater than the one before it, 6 then 7). A file with neither has the five-section bytes it always had; the constant models (kinds 1 to 3) carry section 7 with no feature record. The byte layout of the table, its bounds (pool at most 8 MiB, strings at most 2^17, the section at most 12 MiB), its SHA-256, the 8-byte stub that stands for a categorical step in the `steps` section (kind, layout 2), the schema type 3 (a categorical input column) and the refusals `ml.model.string-table` (code 328) and `ml.limit.string-table` (code 329) are in `docs/stringtable.md`. The next step kind is 11. Kinds 7 to 10 are the categorical steps of issue 21: their payload is `docs/categorical-encoders.md` section 5.2, `transforms.steps` does not read them, and in a model file each is an 8-byte stub whose payload is an entry of the string table (`docs/stringtable.md`).

Kinds 4 and 5 are built by `ml` since W1 (`tools/ml/wire.cho`, through `lib/pipeline/file.cho` and the adapters of `lib/models`; the logistic payload gained a labels record, `docs/wiring.md` section 4); kinds 6 and 7 are written by libraries only, and `mfile.cho` (the reader of the constant kinds)
refuses them as `ml.model.malformed` (`tests/conformance/test_model_file.py` has a case for each). Kind 0 and kinds above 7 are unassigned; the next model takes 8. Tensor
dtypes: 1 = i64, 2 = f64 (one table for every record, the same in `mfile.cho`, `record.cho` and `tree/model.cho`). Section tags of the
file: 1 plan, 2 schema, 3 steps, 4 model, 5 meta, in that order. The column types in `schema` (0 int, 1 dec, 2 float, 3 categorical) are not dtypes.
The stream kinds of the random generator have their own registry in `docs/rng.md`; the numeric refusal codes of the libraries are in
section 8.1.

Alternative considered: a text format (shortest round-trip decimals, as `std.fmt.float_into` prints). It is readable and avoids
the `float_of_bits` workaround, but a decimal parse is a correctly-rounded parse, which is the hard part cancho-table's float
reader spent a design on; the bit-pattern load is a few integer operations. Whether binary is also **fast** to load is **[M0]**.
`introspect` prints the model as JSON for agents either way.

### 4.2 Pipeline plan **[choice]**

A plan is a small declarative text, one step per line, canonicalised before it is stored (fixed key order, no comments, one
spelling per number), so the same plan has the same bytes however it was written:

```
impute column=* strategy=mean
scale column=age,income kind=standard
encode column=country kind=onehot max-categories=64
model kind=logreg iterations=200 rate=0.1 l2=0
```

A plan comes from flags or a file; `ml fit` stores the canonical form. There is no plan that can be applied to a model other
than the one fitted with it (section 5).

## 5. Pipeline semantics

* `fit` runs each step's fit on the training rows only, applies it, and feeds the next step. The model file holds every
  step's fitted parameters. `predict` applies the stored steps and the estimator, and there is no command that applies a plan
  without its stored parameters, which makes train/serve skew impossible by construction (M4's gate).
* **Schema.** The fit input's column names and types are stored. A prediction input with different columns is refused with
  `ml.schema.mismatch` and a `repair` that names the missing and extra columns. Column order may differ if names match
  **[choice]**; names are the identity, not positions.
* **Unknown categories** at predict time: the one-hot step emits all zeros for the block and the result row carries a count of
  such cells in the summary; it never guesses a nearest category **[choice]**. An explicit `--unknown refuse` turns it into
  `ml.input.unknown-category`.

## 6. Determinism rules

1. **The RNG** is SplitMix64 **[choice]** (one `int` of state; the output function uses wrapping multiplication, which
   cancho has as `wrapping_mul`, `defined-behaviour.md` **[read]**). Alternative: xoshiro256**, more state, no benefit here.
   Its stream is specified in `docs/rng.md` with reference vectors taken from the published reference implementation and
   checked by a test (M2).
2. **Streams are derived, never shared.** A tree, a fold, a bootstrap sample or a centroid initialisation gets
   `splitmix(seed, kind, index)`; no draw depends on which thread runs it.
3. **Threads.** cancho has `spawn`/`join` (real pthreads), no atomics, no channels, no shared mutable state; the number of
   threads is a call depth (64 at most); a result comes back through a fixed-size slot **[read]** (`cancho-table`
   `docs/parallel.md` §1). The partition of rows into blocks is therefore static, and **the block size is a function of the
   data and not of the thread count** **[choice]**, which is what lets a fixed reduction tree keep D1 at no extra cost to
   the structure of the code. The exact-accumulator alternative is in `facc.cho` in cancho-table. Which one the decisive
   sums use is **[M0]**.
4. **Ties** are broken by lowest index (feature, then bin, then row), written once and used by every tree builder.
5. **Decisive sums** are: every gradient, every split gain, every centroid update, every metric. A sum that only reports a
   count is not decisive.
6. **Where D1 cannot hold** (none known). An estimator found later that needs thread-order floats is documented here, not
   shipped silently.

## 7. Authority

[read] cancho reports a program's authority per binary, and cancho-table's `table` is checked against a committed manifest in
CI (`tools.toml`, `manifests/`). The expected rows **[choice]**:

| binary | rows | not allowed |
|---|---|---|
| `ml` (fit, predict, eval, introspect, skill) | read files named on the command line (`file_read`, `fs_read("")`, `dir_read`); write the model and the log it is told to, beneath a directory it opened (`dir_write`, `file_write`: no `fs_write`, no path write); heap; `conc` (`spawn`): in the row since W1, because `fit --threads N` calls `lib/linalg/par.cho`; its reason is in `tools.toml` `[ml.reasons]` and `docs/wiring.md` section 6 | network, exec, ffi, clock, environment beyond arguments, standard input |
| `ml-predict` (M10) | read a model file, read an input; heap | everything else, including writes other than standard output and `conc` if single-threaded |
| `ml-serve` (M12, `docs/serving-http.md`) | read a model file and a token file once; listen, accept, read and write connections, wait on them, read the clock; heap. The row is `net_in("")` for either `--bind`: one binary chooses between two literal `tcp_listen_on` calls, so the row does not say which address it listens on, and the loopback default is proved by a socket test (`docs/serving-http.md` section 5.4) | every write, `conc`, `exec`, `ffi`, `net_out`, standard input |

A change to a binary's authority fails CI until the manifest changes in the same diff (M9's gate, built: `scripts/manifest.py
--check`, `tools.toml`, `manifests/ml.authority.json`; the derived row of the CLI slice was the first, without `conc`; W1 added `conc` deliberately, `docs/cli.md`
section 6). Model files and data files are inputs and are never executed or `include`d.

## 8. Refusals

Rule tags in the toolbox style: `<area>.<name>`, an exit code from the toolbox table, a `repair` where one is safe, and a
fixture for every tag **[read]** (`agent-toolbox.md` D4, D5). Exit codes: arguments 2, not found 3, permission 4, a limit or
a parse failure 8, an I/O failure 1. The first set, to be completed in M2 to M9:

| tag | exit | when | repair |
|---|---|---|---|
| `ml.input.empty` | 8 | no rows after the header | — |
| `ml.input.too-many-rows` / `too-many-columns` | 8 | over the limit (flag, ceiling in section 9) | the limit flag, or a row sample |
| `ml.input.non-numeric` | 8 | a cell that is not of the column's type; line and column in `detail` | declare the column `:text` or drop it |
| `ml.input.non-finite` | 8 | `nan`, `inf` | impute or drop the row |
| `ml.input.unknown-category` | 8 | only with `--unknown refuse` | — |
| `ml.label.single-class` | 8 | training or AUC with one class | — |
| `ml.label.not-binary` | 8 | a label with more than two values while multiclass is not built | — |
| `ml.schema.mismatch` | 8 | prediction columns differ from the fitted schema | the difference, named |
| `ml.param.out-of-range` | 2 | a hyperparameter outside its range | the range |
| `ml.limit.depth` / `nodes` / `fits` | 8 | a tree or a search over its ceiling | the flag |
| `ml.fit.non-finite` | 8 | a coefficient or a gain became non-finite | a smaller rate, or scaling |
| `ml.metric.undefined` | 8 | a metric with no defined value (empty fold, one class) | — |
| `ml.model.bad-magic` / `version` / `truncated` / `checksum` | 8 | the model file fails a check in 4.1 | — |
| `ml.model.schema-mismatch` | 8 | see `ml.schema.mismatch` | — |

No refusal depends on running out of memory: every allocation sized by input is first compared to a limit (M2's gate).

*Added by the CLI slice (`tools/ml/spec.cho` is the table; `docs/cli.md`; one fixture per tag in `tests/conformance/fixtures.py`):*

| tag | exit | when | repair |
|---|---|---|---|
| `args.unknown-command` | 2 | the first operand is not `fit`, `predict` or `eval` | none |
| `ml.param.out-of-range` | 2 | a numeric flag outside its range (the range is in `detail`) | none |
| `ml.input.too-many-cells` | 8 | rows times columns over `--max-cells` (checked without an overflowing product) | the flag |
| ~~`ml.input.unsupported-type`~~ | | *removed by the reader stage: an exponent or more than 12 digits is a float column* | |
| ~~`ml.input.unsupported-csv`~~ | | *removed by the reader stage: quoted CSV is read* | |
| `parse.csv-bad-quote`, `parse.csv-unterminated-quote` | 8 | *(added by the reader stage)* a closing quote followed by something other than a comma, a quote open at the end of the file (cancho-table's tags) | none |
| `ml.metric.undefined` | 8 | *(reachable since the reader stage)* the sum of squared errors of a float regression is beyond the largest double | none |
| `ml.input.duplicate-column` | 8 | two columns, or an empty column, with the same header name | none |
| `ml.input.unknown-column` | 3 | `--target` or `--drop` names a column that is not in the header | none |
| `ml.label.not-integer` | 8 | a classification target that is not an integer column | none |
| `ml.model.malformed` | 8 | the checksum is right and a field is not (a repaired file) | none |
| `parse.csv-ragged-row` | 8 | a data row with another number of fields than the header (cancho-table's tag) | none |
| `limit.log-full` | 8 | `--log` names a file already `--log-max-bytes` long | none |

Reused from the toolbox catalogue: `args.*`, `path.*`, `io.*`, `conflict.exists`, `limit.line-too-long`, `limit.input-too-large`
(the model file over `--max-bytes`). `ml.input.non-numeric` gained a `retry` repair (`--drop NAME`, for a feature column) and
`ml.schema.mismatch` one when only extra columns differ (`--drop` them); otherwise a repair is `none` with a reason naming the
columns. **Not reachable from the CLI when the slice was written, so without a fixture and not in `introspect`:** `ml.tensor.*`, `ml.linalg.*`,
`ml.param.threads` (no kernel used linear algebra, no `--threads`), `ml.limit.depth/nodes/fits`, `ml.fit.non-finite`
(no tree, no iterative fit). *(At W1 the verbs call the libraries: all of these but `ml.tensor.*`, `ml.linalg.singular`, `ml.limit.depth`, `ml.limit.nodes` and `ml.limit.fits` became reachable, with a fixture each: section 19, `docs/wiring.md` section 5.)* *(`ml.metric.undefined` became reachable with float regression.)* `ml.model.schema-mismatch` of the first list is
dropped: it said "see `ml.schema.mismatch`", and a second tag for one rule is the repurposing D5 forbids. `ml.input.empty`,
`ml.input.too-many-rows`, `ml.input.too-many-columns`, `ml.input.non-numeric`, `ml.input.non-finite`, `ml.label.single-class`,
`ml.label.not-binary`, `ml.schema.mismatch` and the four model tags of the first list are built as listed there, at the exits listed.


*Added by stage 1 (`lib/core/refuse.cho`, one fixture per tag in `tests/core.cho`):* `ml.tensor.rank`, `ml.tensor.shape`,
`ml.tensor.bounds` (exit 8); `ml.input.too-many-cells` (exit 8: rows x columns over the cell cap, checked without a multiplication
that can overflow); `ml.linalg.not-positive-definite` (exit 8: a Cholesky pivot that is not > 2^-40 times its own diagonal element (so not > 0.0 either), NaN included; the relative test is `v1-post`, `models-linear.md` 3.1),
`ml.linalg.singular` (exit 8: a zero diagonal entry in a triangular solve); `ml.param.threads` (exit 2: a thread count outside
1..64). The default caps (10,000,000 rows, 4,096 columns, 400,000,000 cells) are provisional and wait for section 9.

*Registered at integration: the tags of the libraries that `ml` did not call yet (`tools/ml/spec.cho` `pending()`; not in `introspect`, the
schemas or the fixtures until a verb reaches them, when each moves to `extra()` and `rules()` with a fixture; the ones the wired verbs reach moved at W1, nine stay: `docs/wiring.md` section 5). The tags of the libraries
that `ml` reaches (`ml.input.*`, `ml.label.*`, `ml.schema.mismatch`, `ml.model.truncated` and `malformed`, `ml.metric.undefined`,
`ml.param.out-of-range`) are in the tables above. `ml.limit.fits` of the first list is built by no stage and stays unlisted.*

| tag | exit | when | repair |
|---|---|---|---|
| `ml.tensor.rank` | 8 | a kernel was given a tensor of rank 0 or above 4 | use a tensor of rank 1 to 4 |
| `ml.tensor.shape` | 8 | an extent below 1, or a reshape of a tensor that is not contiguous | extents of at least 1; reshape only a contiguous tensor to the same element count |
| `ml.tensor.bounds` | 8 | the offset, shape and strides of a descriptor reach outside the data beside it | keep them inside the data |
| `ml.linalg.not-positive-definite` | 8 | a Cholesky pivot that is not above 2^-40 times its own diagonal element (a non-positive pivot included) (dependent or nearly dependent columns), NaN included | set l2 above 2^-40 times the diagonal (n for standardised features), drop one of the dependent columns, or centre and scale the columns |
| `ml.linalg.singular` | 8 | a zero on the diagonal of a triangular solve | none |
| `ml.param.threads` | 2 | a thread count outside 1 to 64, or thread copies of a tree fit over the memory ceiling | a thread count of 1 to 64 |
| `ml.fit.non-finite` | 8 | a coefficient, a loss or a gain became non-finite during a fit | scale the features, add l2, or fewer iterations |
| `ml.limit.depth` | 8 | a tree over the depth ceiling (30) *(reachable since W3a: `extra()`, `docs/wiring-trees.md` section 6)* | a max-depth of at most 30 |
| `ml.limit.nodes` | 8 | a forest over the node ceiling *(reachable since W3a)* | raise --max-nodes, lower the depth or the trees, or raise min-samples-leaf |
| `ml.metric.length-mismatch` | 8 | two arrays of a metric have different lengths | one prediction per row |
| `ml.metric.bad-probability` | 8 | a probability outside 0 to 1 | the probability of the positive class |
| `ml.split.empty-fold` | 8 | k above the number of rows, or a train/test split that leaves a side empty | fewer folds, a smaller test size |
| `ml.split.stratum-too-small` | 8 | a class with fewer rows than k *(reachable from `ml cv --stratify` and `fit --split --stratify`: `extra()` since `docs/splits.md`)* | fewer folds |

*Registered by the transforms and the pipeline stage (library only: `lib/transforms/codes.cho`, `docs/transforms.md` section 1 X7, `docs/pipeline.md` section 2.1; `tools/ml/spec.cho` `pending()`; none is reachable from `ml` yet):*

| tag | exit | when | repair |
|---|---|---|---|
| `ml.fit.all-missing` | 8 | a selected column has no non-missing cell in the training rows (all transforms but `impute strategy=constant`); the step and the column are in `detail` | drop the column, or `impute strategy=constant` |
| `ml.limit.categories` | 8 | a column has more distinct values than `max-categories` | raise `max-categories` up to 1024, or `bucket` the column |
| `ml.input.unknown-category` | 8 | an `encode ... unknown=refuse` step meets a value no training row had; step, column and row in `detail` (this is the tag of the first list, now built) | refit with the value in the training rows, or `unknown=ignore` |
| `ml.plan.syntax` | 8 | a plan line is not `verb key=value ...`; line and column in `detail` | one step per line, `verb key=value` |
| `ml.plan.unknown-step` | 8 | the verb is not `impute`, `scale`, `encode`, `bucket`, `freq`, `hash` or `model` | one of the seven verbs |
| `ml.plan.unknown-key` | 8 | a key the verb does not take | the keys of the verb (`docs/pipeline.md` section 2) |
| `ml.plan.missing-key` | 8 | a required key is absent | add the key |
| `ml.plan.bad-value` | 8 | a value not in the key's list or not a number of its kind | the values of the key |
| `ml.plan.duplicate-key` | 8 | a key twice on one line | write each key once |
| `ml.plan.model-line` | 8 | no `model` line, two, or a step after it | end the plan with one `model` line |
| `ml.plan.unknown-column` | 8 | a column a step names is not a column of the matrix it receives | names are compared byte for byte |
| `ml.limit.plan` | 8 | a plan over 64 KiB, 64 lines or 4,096 listed names, or a name over 255 bytes | shorten the plan |
| `ml.plan.column-kind` | 8 | a step reads the other kind of column than the one it names: a numeric step on a categorical column, `freq` or `hash` on a numeric one, a mixed list (stage 2 of issue 21, `docs/categorical-encoders.md` section 4.2); step, column and what the step needs in `detail` | `retry` with an `encode` step put before, when the plan came from `--step` |
| `ml.input.reserved-category` | 8 | `min-count` is 2 or more and a categorical column holds the value `__other__`, the name of the rare bucket (step and column in `detail`) | none: rename the value or use `min-count=1` |

The library also answers, with its own codes in its band, tags that exist above: `ml.input.empty`, `ml.input.too-many-rows`, `ml.input.too-many-columns`, `ml.input.too-many-cells`, `ml.param.out-of-range`,
`ml.input.non-finite` (a cell of a matrix given to a transform, or a transformed value that overflowed), `ml.fit.non-finite`, `ml.schema.mismatch` and `ml.model.bad-magic`, `version`, `truncated`, `checksum`, `malformed`.

*Added at W1 (`tools/ml/spec.tmpl`, `docs/wiring.md` section 5):*

| tag | exit | when | repair |
|---|---|---|---|
| `ml.input.missing-value` | 8 | a feature cell is empty after the steps of the plan (no `impute` covers its column); the columns are in `detail` | `retry` with `--step "impute column=NAMES strategy=mean"` when the plan came from no flag, else `none` with the step to add |
| `ml.plan.model-not-built` | 8 | the `model` line of a plan names a model `ml` does not build (`constant`, which takes no plan) | none |

*Added by stage 1 of issue 21, categorical columns (`docs/categorical.md` section 5; `tools/ml/spec.tmpl`, `lib/core/refuse.cho` codes 16 to 20, one fixture per tag in `tests/conformance/fixtures.py`):*

| tag | exit | when | repair |
|---|---|---|---|
| `ml.limit.string-bytes` | 8 | a cell of a categorical column (`--cat`) is longer than `--max-string-bytes` (default 255, ceiling 65,536) | none (raise the flag, or drop the column) |
| `ml.limit.string-pool` | 8 | the distinct categories of all the categorical columns together would take more than `--max-pool-bytes` (default 1 MiB, ceiling 8 MiB) | none |
| `ml.limit.categorical-columns` | 8 | `--cat` names more columns than `--max-cat-columns` (default 128, ceiling 4,096), seen at the header | none |
| `ml.input.invalid-utf8` | 8 | a cell of a categorical column is not well-formed UTF-8; line, column, name and byte offset in `detail` | none: re-encode the file, or remove the column from `--cat` and drop it |
| `ml.input.categorical-not-encoded` | 8 | a column named by `--cat` is read by no step of the plan (since stage 2; at stage 1 no step could); names, category counts and the SHA-256 of the canonical vocabulary dump in `detail` | none: an `encode`, `freq` or `hash` step, or `--drop` |
| `ml.model.string-table` | 8 | the string table of a model file is damaged, passes a bound of `docs/stringtable.md` section 4, or disagrees with the steps, the plan or the schema of its own file (`reason`, `step` and `section` in `detail`) | none: refit |
| `ml.limit.string-table` | 8 | a fit would write a string table over 131,072 strings, 8 MiB of string bytes or 12 MiB (`what`, `size`, `max` in `detail`) | none: fewer or shorter categories, or a `hash` step |

*Where `ml-serve` listens, and the token (`docs/serving-http.md` sections 5.4 and 5.6; `tools/ml-serve/sspec.cho`, `tools/ml-serve/main.cho`; refused before the port is bound):*

| tag | exit | when | repair |
|---|---|---|---|
| `serve.no-token` | 2 | `--bind all` without `--token-file`: on every address of the host any host that can reach the port can call the model | `--token-file FILE`, or drop `--bind all` to listen on 127.0.0.1 only |
| `args.conflict` | 2 | `--bind all` with `--no-token`, or `--token-file` with `--no-token` | drop `--no-token` |
| `args.bad-value` | 2 | `--bind` is neither `loopback` nor `all` | `--bind loopback` or `--bind all` |
| `serve.token` | 8 | the token file is empty, over 4,096 bytes, or holds a byte outside `!` to `~` | write the token alone in the file |

*A prediction that is not a finite number (`docs/nonfinite.md`; `tools/ml/spec.tmpl`, `lib/core/refuse.cho` code 21, one fixture in `tests/conformance/fixtures.py`):*

| tag | exit | when | repair |
|---|---|---|---|
| `ml.predict.non-finite` | 8 | a prediction, a probability or a squared distance of a row scored by `predict`, `ml-predict`, `eval` or `ml-serve` is NaN or infinite; the 1-based row of the file and the column of the answer (`prediction`, `label`, `probability`, `distance`) are in `detail`; all the rows scored are checked, not only the page | none: scale the inputs, or check or drop the row |

`ml.limit.categories` (above, from the transforms) is also answered by the reader for a categorical column that passes `--max-categories` (default 64, ceiling 1,024): one rule, two places.

*Added by M22, run tracking and train/serve skew (`docs/tracking.md` section 5.5; `tools/ml/spec.tmpl`, one fixture per tag in `tests/conformance/test_track.py`, `tests/conformance/fixtures.py`):*

| tag | exit | when | repair |
|---|---|---|---|
| `ml.predict.out-of-range` | 8 | more rows of a `predict` or `eval` hold a numeric feature outside the range stored in the model file than `--max-out-of-range` allows; rows, bound, first column and first row in `detail` | none |
| `ml.predict.no-provenance` | 8 | `--max-out-of-range` was given and the model file has no provenance section | none: refit with this version |
| `ml.runs.conflict` | 8 | a run record with this id exists and differs from the one to write in more than the thread count and the name of the model file | none |
| `ml.runs.too-many` | 8 | the `--track` directory holds 1000 records already | none |
| `ml.runs.record-too-large` | 8 | a record would be over 262,144 bytes | none |
| `ml.runs.not-found` | 3 | no record has this id | none |
| `ml.runs.unknown-model` | 8 | no record of the directory has the SHA-256 of the model file `eval --track` was given | none |
| `ml.runs.damaged` | 8 | a record does not parse, has a member out of its type, an id that is not the hash of its inputs, or a check that is not the hash of what precedes it | none |
| `ml.runs.data-changed` | 8 | the training file a record names does not have the SHA-256 it holds | none |
| `ml.runs.model-differs` | 8 | the refit of `runs verify` gave a file whose SHA-256 is not the record's | none |
| `ml.runs.not-replayable` | 8 | a column carrying `--drop`, `--float` or `--cat` is past the 62nd position of the header | none |

`ml.model.malformed` also answers a damaged section 7 (`detail.section` 7); `io.not-found` a `--track` directory that does not exist; 

*Added by issue 42, split strategies and cross-validation (`docs/splits.md` section 7; `tools/ml/spec.tmpl`, one fixture per tag in `tests/conformance/fixtures.py`; `ml.split.empty-fold` and `ml.split.stratum-too-small` above are the library's, reached by the same verbs):*

| tag | exit | when | repair |
|---|---|---|---|
| `ml.split.stratify-target` | 2 | `--stratify` with a regression or a clustering | `retry` without `--stratify` |
| `ml.split.too-few-groups` | 8 | fewer groups than `--folds`, or fewer than 2 | `retry` with `--folds` set to the number of groups when it is at least 2, else none |
| `ml.split.time-format` | 8 | a cell of the `--split-by` column that is not a number or an accepted ISO-8601 date or datetime, or of the other kind than the first (`row`, `column`, `reason`) | none |
| `ml.split.missing-cell` | 8 | an empty cell in the `--split-by` or `--split-group` column | none |
| `ml.split.combination` | 2 | two of `--stratify`, `--split-group`, `--split-by`; `cv` with `--split`; `--repeats` above 1 with `--split-by` | `retry` with `--repeats 1` for the last, else none (which rule to keep is a judgement) |
| `ml.split.out-of-range` | 2 | `--folds` outside 2 to 20, `--repeats` outside 1 to 10 | none |
| `ml.split.column-target` | 2 | the group or time column is the target | none |
| `ml.split.emit-too-large` | 8 | `--emit-folds` over 1,048,576 entries | `retry` without `--emit-folds` |

## 8.1 The numeric codes of the libraries (added at integration)

The libraries answer a refusal as an `int` code (0 is none) and each has a function from code to tag, exit code and repair. Three
stages numbered theirs alone (the metrics stage 1 to 11, which were `ml.refuse`'s 1 to 11 with other meanings; the tree stage 101 to
108), so the numbers are now **bands, one per library, and no number is in two bands**. A code means a tag only inside its band; the same
tag may have a code in several bands (it is the same rule, and the wiring stage translates by tag, never by number). Bands: 1 to 99
`ml.refuse` (the core; the linear stage's codes 12 to 15 are here; linalg, models, reader and splits call it), 101 to 199 `tree.deny`,
201 to 299 `metrics.rules`,  `tests/t_registry.cho` fails if a number is claimed by two bands,
if a band has a gap or a duplicate tag, or if one tag has two exit codes; `scripts/check_registry.py` checks this table against the
sources and against the tables of section 8 and `tools/ml/spec.cho`.

| code | tag | exit | library |
|---|---|---|---|
| 1 | `ml.tensor.rank` | 8 | `ml.refuse` |
| 2 | `ml.tensor.shape` | 8 | `ml.refuse` |
| 3 | `ml.tensor.bounds` | 8 | `ml.refuse` |
| 4 | `ml.input.empty` | 8 | `ml.refuse` |
| 5 | `ml.input.too-many-rows` | 8 | `ml.refuse` |
| 6 | `ml.input.too-many-columns` | 8 | `ml.refuse` |
| 7 | `ml.input.too-many-cells` | 8 | `ml.refuse` |
| 8 | `ml.param.out-of-range` | 2 | `ml.refuse` |
| 9 | `ml.linalg.not-positive-definite` | 8 | `ml.refuse` |
| 10 | `ml.linalg.singular` | 8 | `ml.refuse` |
| 11 | `ml.param.threads` | 2 | `ml.refuse` |
| 12 | `ml.input.non-finite` | 8 | `ml.refuse` |
| 13 | `ml.label.single-class` | 8 | `ml.refuse` |
| 14 | `ml.label.not-binary` | 8 | `ml.refuse` |
| 15 | `ml.fit.non-finite` | 8 | `ml.refuse` |
| 16 | `ml.limit.categories` | 8 | `ml.refuse` |
| 17 | `ml.limit.string-bytes` | 8 | `ml.refuse` |
| 18 | `ml.limit.string-pool` | 8 | `ml.refuse` |
| 19 | `ml.limit.categorical-columns` | 8 | `ml.refuse` |
| 20 | `ml.input.invalid-utf8` | 8 | `ml.refuse` |
| 21 | `ml.predict.non-finite` | 8 | `ml.refuse` |
| 101 | `ml.input.non-finite` | 8 | `tree.deny` |
| 102 | `ml.label.not-binary` | 8 | `tree.deny` |
| 103 | `ml.label.single-class` | 8 | `tree.deny` |
| 104 | `ml.limit.depth` | 8 | `tree.deny` |
| 105 | `ml.limit.nodes` | 8 | `tree.deny` |
| 106 | `ml.model.truncated` | 8 | `tree.deny` |
| 107 | `ml.model.malformed` | 8 | `tree.deny` |
| 108 | `ml.schema.mismatch` | 8 | `tree.deny` |
| 201 | `ml.input.empty` | 8 | `metrics.rules` |
| 202 | `ml.input.too-many-rows` | 8 | `metrics.rules` |
| 203 | `ml.input.non-finite` | 8 | `metrics.rules` |
| 204 | `ml.label.single-class` | 8 | `metrics.rules` |
| 205 | `ml.label.not-binary` | 8 | `metrics.rules` |
| 206 | `ml.metric.undefined` | 8 | `metrics.rules` |
| 207 | `ml.param.out-of-range` | 2 | `metrics.rules` |
| 208 | `ml.metric.length-mismatch` | 8 | `metrics.rules` |
| 209 | `ml.metric.bad-probability` | 8 | `metrics.rules` |
| 210 | `ml.split.empty-fold` | 8 | `metrics.rules` |
| 211 | `ml.split.stratum-too-small` | 8 | `metrics.rules` |
| 301 | `ml.input.empty` | 8 | `transforms.codes` |
| 302 | `ml.input.too-many-columns` | 8 | `transforms.codes` |
| 303 | `ml.param.out-of-range` | 2 | `transforms.codes` |
| 304 | `ml.fit.non-finite` | 8 | `transforms.codes` |
| 305 | `ml.fit.all-missing` | 8 | `transforms.codes` |
| 306 | `ml.limit.categories` | 8 | `transforms.codes` |
| 307 | `ml.input.unknown-category` | 8 | `transforms.codes` |
| 308 | `ml.schema.mismatch` | 8 | `transforms.codes` |
| 309 | `ml.model.bad-magic` | 8 | `transforms.codes` |
| 310 | `ml.model.version` | 8 | `transforms.codes` |
| 311 | `ml.model.truncated` | 8 | `transforms.codes` |
| 312 | `ml.model.checksum` | 8 | `transforms.codes` |
| 313 | `ml.model.malformed` | 8 | `transforms.codes` |
| 314 | `ml.plan.syntax` | 8 | `transforms.codes` |
| 315 | `ml.plan.unknown-step` | 8 | `transforms.codes` |
| 316 | `ml.plan.unknown-key` | 8 | `transforms.codes` |
| 317 | `ml.plan.missing-key` | 8 | `transforms.codes` |
| 318 | `ml.plan.bad-value` | 8 | `transforms.codes` |
| 319 | `ml.plan.duplicate-key` | 8 | `transforms.codes` |
| 320 | `ml.plan.model-line` | 8 | `transforms.codes` |
| 321 | `ml.plan.unknown-column` | 8 | `transforms.codes` |
| 322 | `ml.limit.plan` | 8 | `transforms.codes` |
| 323 | `ml.input.too-many-rows` | 8 | `transforms.codes` |
| 324 | `ml.input.too-many-cells` | 8 | `transforms.codes` |
| 325 | `ml.input.non-finite` | 8 | `transforms.codes` |
| 326 | `ml.plan.column-kind` | 8 | `transforms.codes` |
| 327 | `ml.input.reserved-category` | 8 | `transforms.codes` |
| 328 | `ml.model.string-table` | 8 | `transforms.codes` |
| 329 | `ml.limit.string-table` | 8 | `transforms.codes` |

## 9. Limits and sizes — waits for M0

The ceilings for rows, columns, tree nodes, forest size and search fits are **[M0]**: the row ceiling depends on the speed
factor (decision rule in `docs/spikes.md` §5) and on the memory per row (8 bytes per cell, so the 1,000,000 by 20 spike
matrix is 160 MB; **[estimate]**, not measured here).

## 10. Sharing code with cancho-table **[open]**

What is shared: the reader (`:int`, `:dec`, `:float`), the typed keys and the exact accumulator (`facc`). What is shared by
**pin, not copy** (`CLAUDE.md`): the `contract/` package of `cancho-tools`. Options for the first three: (a) a package that
cancho-table exports and cancho-ml depends on at a pinned commit; (b) a small `cancho-numeric` package both depend on.
Recommendation: (a) now, (b) only if cancho-table's reader moves for cancho-ml's sake. *(Done, `docs/reader.md` section 1: option (a) needed
no change to any source of cancho-table, only that it publish `pow5`, `dec`, `flt`, `query`, `facc` and `reader` as `.cancho-vcs` stores
at a commit, with a CI check that they match (cancho-table PR #44, merged). `ml` pins five of them as `[dependencies.*]` (`pow5`, `dec`, `flt`, `facc`, `reader`); the proposal is kept in
`docs/upstream/` as the record. In the other direction, `lib/core`, `lib/linalg` and `lib/metrics` are published as 23 `.cancho-vcs` stores that cancho-stats pins: `docs/stores.md`.)*

## 11. What M0 must settle before section 6 and 9 are final

Upstream asks, each only with a measured reproducer (spikes L and S have the first three): bounds checks of a per-row subslice
discharged; a 2-D view or slice-of-slice type (or struct reference fields, which would let a view hold its data); a faster
`std.math.exp` (the tuned one is bit-identical on 114M points); `float_of_bits`; text-to-float in `std`.

* the determinism arm (exact accumulator against fixed reduction tree) and the threshold for choosing between them;
* the cost of `exp`/`log` per call, and of the bit-pattern load;
* whether the LLVM backend builds `float` code at `a4572ea`, and what it gains;
* the speed factor against scikit-learn, which sets the row ceiling.

## 12. Stage 1 of v1: the linear algebra and the core (added with `docs/linalg.md`)

`docs/linalg.md` holds the gate, the summation-order specification and the results of section 3.1's package; it was written
before the kernels. v1 scope decisions (maintainer): binary classification and regression only (the format reserves a class
count); cancho-table is shared by a pinned package dependency (*not possible yet, and what is: `docs/reader.md` section 1*); descriptors are plain ints with
data beside them; every reduction has the fixed-block, fixed-tree order of `docs/linalg.md` section 2, behind one function.

## 13. Stage 2 of v1, early: the command line (M9 as a vertical slice)

`docs/cli.md` holds the gates (written before the code), the reference, the arithmetic of the one model that is built
(`--model constant`), the log, the authority and the verdicts. What it changes in this document: the model file's digest is a
trailer (section 4.1, corrected in place), section 7's row for `ml` has the labels the derivation printed, section 8 has the tags the
slice added and the ones nothing reaches yet. What it does not settle: the speed factor (section 9), the second machine for D2, the
shared reader (section 10). *(Settled by the reader stage: `tools/ml/csv*.cho` now reads cells with cancho-table's own modules, pinned as
`[dependencies.*]` on cancho-table's `.cancho-vcs` stores; `docs/reader.md`.)*


## 14. Stage M6 of v1: linear regression, logistic regression, k-means (library; not wired into the command line)

`docs/models-linear.md` holds the gates (written before the code), the specification of the three estimators (every sum in the order
of `docs/linalg.md` section 2), the tensor records of the model file, the refusals, the mutants and the measurements. What it changes
in this document: section 8 gains four tags (`ml.input.non-finite`, `ml.label.single-class`, `ml.label.not-binary` and `ml.fit.non-finite`,
`lib/core/refuse.cho` codes 12 to 15, all exit 8, each with a fixture in `tests/p_models.cho`); section 4.1 gains `dtype 2 = f64` for a
tensor record and the model kinds 4 (linear), 5 (logistic) and 6 (k-means) *(as renumbered at integration; the stage wrote 3, 4, 5, and 3 was taken)*, not yet in `tools/ml/mfile.cho`; `docs/rng.md` gains the
stream kind 3 (k-means). Section 6 is kept: every decisive sum of the three estimators is the fixed block-and-tree order, so the
thread count changes no byte (the gate compares the raw sums and the loss and inertia lines, not only the coefficients).

## 15. Stage M7: decision trees and random forests (library only)

`docs/models-tree.md` holds the algorithm, the numeric decisions (exact integer Gini; sequential float regression sums in a
fixed row order), the limits and tags, the missing-value rule (a learned default side per split), the tensor records of the
`model` section (kind 7; see the kind table in section 4.1), the gates, the timing protocol and the verdicts (its section 10); it was written before the code.
It is a library under `lib/models/tree/`, driven by `tools/tree-check` for the gates; the command line does not call it yet.
What it changes in this document: model kind 7 for section 4.1 *(the stage wrote 6 here and 3 in its own document; integration fixed 7)*; the tags `ml.limit.depth`, `ml.limit.nodes`, `ml.param.threads`
(also for the memory the thread copies would take) become reachable from a library call (section 8; none is reachable from `ml`
until the verbs call the library); the row ceiling of a tree fit is 2^20 (the exact Gini comparison), a figure for section 9.

## 16. Stage M5: metrics and splits (library only; added at integration)

`docs/metrics.md` holds the gates (written before the code), the specification (a bounded draw by rejection, Fisher-Yates, the splits,
the order of every decisive sum, the metrics), the rule tags and the results. The stage did not edit this document; the integration
registered what it adds: its rule tags in section 8 and 8.1 (codes 201 to 211, `lib/metrics/rules.cho`), its stream kinds in `docs/rng.md`
(they are 4, 5 and 6 since the integration; the stage had 1, 2 and 3, which the tree and k-means streams also used), and the summation order
of its sums, which was **not** the one of `docs/linalg.md` section 2 at the time (4 lanes in a block of 256 terms against one chain): section 3.1 says every
reduction has *a* normative order, each kernel's document names its own, and `docs/integration.md` says where the constant lives.
*(Corrected in place at `v1-lanes`, section 17: the two orders are now one, the metrics' rule; `docs/lanes.md`.)* The library
is not called by the command line yet.

## 17. `v1-lanes`: one summation order with four lanes (and what it does to bytes)

**Decision [choice, from `docs/simd-ceiling.md` section 11].** `K = 256`, `L = 4`, tree across blocks, Cholesky panels of
`NB = 32`; one definition, `docs/linalg.md` section 2, one set of constants, `lib/linalg/order.cho` (`block`, `lanes`, `lane_combine`,
`block_partial`, `panel`, `group`). The two conventions that existed (`order.cho`: one chain per block; `sums.cho`: 4 lanes) are one.
`docs/lanes.md` has the gates, the mutants and the list of gates whose expected values had to change.

**What changed in bytes.** Any number produced by the previous order and a number produced by this one can differ in the last bits:
`dot`, `gemv_n`, `gemv_t`, `syrk`, `gemm`, `cholesky`; therefore every linear-regression coefficient, every logistic and k-means fit
(gradient, Hessian, loss, cluster sums, inertia) and everything derived from them. **Unchanged bytes:** `axpy`, the triangular solves,
the metrics (they already followed this rule), the tree and forest (their regression sums are sequential in a fixed row order), the
random generator and splits, the command line's constant models (exact integer and `facc` arithmetic: no float order), and the
model file of kinds 1 to 3. Which bytes were measured to be unchanged is in `docs/lanes.md`.

**The model file and the order.** There are no shipped models of kinds 4 to 7: `mfile.cho` refuses them and the libraries' `record.cho`
writes the `model` section payload only. The CML1 `meta` section records the compiler pin; its `u32 hyperparameters` is a count that is 0
and has no defined entries, and the file `version` covers the layout, not the arithmetic. **So there is no existing place for a "sum order
version", and none was added.** *Implemented at `v1-post`* (the maintainer's decision was given with the task): the `meta` section of a file of **kind 4 or above** gets a
`u32 sum order` after the compiler pin (1 = one chain per block, retired and never written by a model file; **2 = this order**), and a
reader refuses a file whose value it does not implement (`ml.model.version`); the rule, with the encoding (presence by kind, no flag) is section 4.1. The
extension costs nothing in compatibility because no file that needs it existed; kinds 1 to 3 do not depend on a float order and write the layout of section 4.1
unchanged, byte for byte (`tests/conformance/test_model_file.py` parses it).

## 18. Stages M3 and M4: transforms and the pipeline (libraries only)

`docs/transforms.md` and `docs/pipeline.md` hold the gates (written before the code), the rules for empty, constant, all-missing and missing columns, the
plan grammar, the records of the `steps` section and the verdicts. They are `lib/transforms` and `lib/pipeline`, driven by `tools/pipe-check`; the command
line does not call them yet. What they change here: section 4.1 gains the step kinds and the section tags as registries, section 8 and 8.1 the tags and
the band 301 to 325 (`lib/transforms/codes.cho`), `tools/ml/spec.cho` `pending()` twelve tags. Section 4.2 is made precise by `docs/pipeline.md` section 2
(the model line's keys are provisional); section 5 is built (`pipeline.run`, `pipeline.schema`, `pipeline.file`).

## 19. Stage W1: the module contract and the first models in the verbs

`docs/modules.md` is the contract of a model module (a manifest, a fixed adapter, a generator, an admission script, mutant modules that the script must
refuse) and `docs/wiring.md` is the wiring of `linear` and `logistic` into `ml fit | predict | eval` through the pipeline, with its gates, flow, refusals and results.
What it changes in this document: section 4.1 (kinds 4 and 5 are built by `ml`; the logistic payload has a labels record; `mfile.cho` still reads the constant kinds only),
section 7 (`conc` is in the row of `ml`, with its reason), section 8 (two tags, `ml.input.missing-value` and `ml.plan.model-not-built`; sixteen tags moved from
`pending()` to the tables of `introspect` with a fixture each, nine stay in `pending()` because no input can reach them), and section 4.2 (the keys of the `model` line are
generated from the manifests). The summation order, the registry of kinds and the stream kinds are unchanged.

## 20. Stage 1 of issue 21: categorical columns, the reader, the matrix and the limits (`docs/categorical.md`)

Text columns are **declared** (`--cat NAMES` of `fit`, never inferred), read into a dictionary sorted bytewise (`lib/core/dict.cho`: a sorted array with binary search, no hash) and an `i64` id
tensor beside the float matrix (`lib/core/catmat.cho`), with a missing bitmap, `--na-values`, and five limits each refused before the thing it limits exists (categories 64/1,024, bytes 255/65,536,
pool 1 MiB/8 MiB, columns 128/4,096, and the id cells in `--max-cells`). Stage 2 (`docs/categorical-encoders.md`) built the steps that read them: `encode` on a categorical column (one-hot or label, with the rare bucket `min-count`, `missing`, `unknown`), `freq` and `hash`, in `lib/transforms/catenc.cho` and `lib/pipeline/mixed.cho`; a declared column that no step reads is refused with `ml.input.categorical-not-encoded`.
The reader's column kind 4 and the schema type code 3 are registered in `lib/core/kinds.cho`. Stage 3 (`docs/stringtable.md`) stores the vocabulary of the categorical steps in a sixth section of the model file. Stage 4 of the issue (native categorical splits in trees) is not built.

## 21. Stage W3a: the tree and the forest in `ml fit | predict | eval` (`docs/wiring-trees.md`)

`--model tree` and `--model forest` are wired through the module contract (a second adapter interface, `docs/modules.md` section 4b). Registries: model kind 7 is written by both (the plan
names which); its `sum_order` is **3** (section 4.1: sequential sums, no lane order); the payload is the library's with the labels record of a classification after it. Section 8: `ml.limit.depth` and
`ml.limit.nodes` moved from `pending()` to `extra()` and `rules()` with a fixture each; `ml.input.too-many-rows` has a second cause (a tree fit takes at most 2^20 = 1,048,576 training rows); `ml.param.threads`
has one more cause that no fixture can reach (the memory ceiling of the thread copies, about 2.3e8 cells). A tree or forest takes missing cells itself (a learned default side per split); the linear models
keep refusing them. The libraries are otherwise unchanged except for the order of two comparisons in the reader of the payload, which a fuzz of the wiring found to trap on a class count of 2^63 - 1
(`docs/models-tree.md` 10.6).

## 22. Issue 42: split strategies and cross-validation from the command line (`docs/splits.md`)

`ml cv` is a verb of `ml` (not of `ml-predict` or `ml-serve`: a split is a training-time concept, and `scripts/check_bins.py` keeps the flags out of their tables). Six flags are new
(`--stratify`, `--split-by`, `--split-group`, `--folds`, `--repeats`, `--emit-folds`). The stream registry takes kinds 7 (stratified holdout) and 8 (the shuffle of the groups); the plain holdout, k-fold and
stratified k-fold keep kinds 4, 5 and 6, so every existing assignment is unchanged. Section 8 takes nine tags (above); `ml.split.stratum-too-small` moved from `pending()` to `extra()`. The canonical text of a
split (`holdout train=0.8 stratify seed=1`, `cv folds=5 repeats=1 group=customer seed=3`) is written by one function (`tools/ml/splitspec.cho`) for the tracking record of issue 43. The answer of `fit --split` and of
`eval --split` gains `split.spec` (and `split.assignment` with `--emit-folds`); `schemas/ml.v1.json` gains the answer of `cv`.

## 23. M22: run tracking and train/serve skew (`docs/tracking.md`)

Every model file `fit` writes carries a provenance section (tag 7): the training file's SHA-256 and size, the rows, and for each numeric feature the count, minimum, maximum, exact mean and population
standard deviation of the training cells as the reader read them. `eval` says whether it scored the training file (`on`); `predict`, `eval` and `ml-serve` count the cells outside the stored range
(`out_of_range`, bounded by `--max-out-of-range`); `fit --track DIR` writes a canonical run record whose id is the hash of the inputs that determine the model bytes; `ml runs list|show|compare|verify`
read the records and replay a fit in memory. The split of a record is the canonical split text of section 22 (`none`, `holdout train=0.8 stratify seed=1`, ...), and a replay holds the same rows out for every strategy. `ml cv` takes no `--track`
(`docs/tracking.md` section 5.7). The registries are unchanged except the section tag (7) and the rule tags of section 8; no model kind, step kind or stream kind is added, and no capability.

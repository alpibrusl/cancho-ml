# cancho-ml design (M1) — DRAFT, sections 1 to 8 and 10; sections 9 and 11 wait for M0

**Status: draft for the maintainer's review. Not code.** Every claim is tagged:

* **[read]** read from a document of the pinned compiler (`cancho` `a4572ea`) or of cancho-table; not re-measured here.
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
* **Reading.** Cells go through cancho-table's exact reader (`:int`, `:dec(S)`, `:float`, correctly rounded). The reader that
  handles 15 to 17 digit doubles is `tools/table/flt.cho` on a cancho-table branch, **not merged** at the time of writing
  **[read]**; M2 depends on it or on the merged reader, and which it is is recorded in M2.
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
cancho#251; not re-checked at `a4572ea`]**. Consequences:

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
shape are constants of the package, never functions of the thread count. This is the fixed-reduction arm C of `docs/spikes.md`,
or its exact-accumulator alternative; which one is **[M0]**. Each kernel's document says its order, and its test compares it to
a reference written in that order (as spikes L and S do with numpy). Threading only decides which thread runs which block.

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
| `ml` (fit, predict, eval, introspect, skill) | read files named on the command line (`file_read`, `fs_read("")`, `dir_read`); write the model and the log it is told to, beneath a directory it opened (`dir_write`, `file_write`: no `fs_write`, no path write); heap; `conc` (`spawn`) **once a threaded kernel is in the binary** | network, exec, ffi, clock, environment beyond arguments, standard input |
| `ml-predict` (M10) | read a model file, read an input; heap | everything else, including writes other than standard output and `conc` if single-threaded |

A change to a binary's authority fails CI until the manifest changes in the same diff (M9's gate, built: `scripts/manifest.py
--check`, `tools.toml`, `manifests/ml.authority.json`; the derived row of the slice is the first, without `conc`, in `docs/cli.md`
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
| `ml.input.unsupported-type` | 8 | a cell with an exponent, or a column that needs more than 12 digits; floating-point text needs cancho-table's float reader | none |
| `ml.input.unsupported-csv` | 8 | a double quote (quoted CSV is not read) | none |
| `ml.input.duplicate-column` | 8 | two columns, or an empty column, with the same header name | none |
| `ml.input.unknown-column` | 3 | `--target` or `--drop` names a column that is not in the header | none |
| `ml.label.not-integer` | 8 | a classification target that is not an integer column | none |
| `ml.model.malformed` | 8 | the checksum is right and a field is not (a repaired file) | none |
| `parse.csv-ragged-row` | 8 | a data row with another number of fields than the header (cancho-table's tag) | none |
| `limit.log-full` | 8 | `--log` names a file already `--log-max-bytes` long | none |

Reused from the toolbox catalogue: `args.*`, `path.*`, `io.*`, `conflict.exists`, `limit.line-too-long`, `limit.input-too-large`
(the model file over `--max-bytes`). `ml.input.non-numeric` gained a `retry` repair (`--drop NAME`, for a feature column) and
`ml.schema.mismatch` one when only extra columns differ (`--drop` them); otherwise a repair is `none` with a reason naming the
columns. **Not reachable from the CLI today, so without a fixture and not in `introspect`:** `ml.tensor.*`, `ml.linalg.*`,
`ml.param.threads` (no kernel uses linear algebra yet, no `--threads`), `ml.limit.depth/nodes/fits`, `ml.fit.non-finite`,
`ml.metric.undefined` (no tree, no iterative fit, no metric with an undefined value). `ml.model.schema-mismatch` of the first list is
dropped: it said "see `ml.schema.mismatch`", and a second tag for one rule is the repurposing D5 forbids. `ml.input.empty`,
`ml.input.too-many-rows`, `ml.input.too-many-columns`, `ml.input.non-numeric`, `ml.input.non-finite`, `ml.label.single-class`,
`ml.label.not-binary`, `ml.schema.mismatch` and the four model tags of the first list are built as listed there, at the exits listed.


*Added by stage 1 (`lib/core/refuse.cho`, one fixture per tag in `tests/core.cho`):* `ml.tensor.rank`, `ml.tensor.shape`,
`ml.tensor.bounds` (exit 8); `ml.input.too-many-cells` (exit 8: rows x columns over the cell cap, checked without a multiplication
that can overflow); `ml.linalg.not-positive-definite` (exit 8: a Cholesky pivot that is not > 0.0, NaN included),
`ml.linalg.singular` (exit 8: a zero diagonal entry in a triangular solve); `ml.param.threads` (exit 2: a thread count outside
1..64). The default caps (10,000,000 rows, 4,096 columns, 400,000,000 cells) are provisional and wait for section 9.

## 9. Limits and sizes — waits for M0

The ceilings for rows, columns, tree nodes, forest size and search fits are **[M0]**: the row ceiling depends on the speed
factor (decision rule in `docs/spikes.md` §5) and on the memory per row (8 bytes per cell, so the 1,000,000 by 20 spike
matrix is 160 MB; **[estimate]**, not measured here).

## 10. Sharing code with cancho-table **[open]**

What is shared: the reader (`:int`, `:dec`, `:float`), the typed keys and the exact accumulator (`facc`). What is shared by
**pin, not copy** (`CLAUDE.md`): the `contract/` package of `cancho-tools`. Options for the first three: (a) a package that
cancho-table exports and cancho-ml depends on at a pinned commit; (b) a small `cancho-numeric` package both depend on.
Recommendation: (a) now, (b) only if cancho-table's reader moves for cancho-ml's sake. Needs a look at how cancho-table is
packaged (`cancho.toml` shows pinned `[dependencies]` by git rev and path) before the choice is made.

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
count); cancho-table is shared by a pinned package dependency (not yet pinned: see `cancho.toml`); descriptors are plain ints with
data beside them; every reduction has the fixed-block, fixed-tree order of `docs/linalg.md` section 2, behind one function.

## 13. Stage 2 of v1, early: the command line (M9 as a vertical slice)

`docs/cli.md` holds the gates (written before the code), the reference, the arithmetic of the one model that is built
(`--model constant`), the log, the authority and the verdicts. What it changes in this document: the model file's digest is a
trailer (section 4.1, corrected in place), section 7's row for `ml` has the labels the derivation printed, section 8 has the tags the
slice added and the ones nothing reaches yet. What it does not settle: the speed factor (section 9), the second machine for D2, the
shared reader (section 10: cancho-table is not a package, so the slice has a reader of its own, `tools/ml/csv.cho`, until it is).


## 14. Stage M6 of v1: linear regression, logistic regression, k-means (library; not wired into the command line)

`docs/models-linear.md` holds the gates (written before the code), the specification of the three estimators (every sum in the order
of `docs/linalg.md` section 2), the tensor records of the model file, the refusals, the mutants and the measurements. What it changes
in this document: section 8 gains four tags (`ml.input.non-finite`, `ml.label.single-class`, `ml.label.not-binary` and `ml.fit.non-finite`,
`lib/core/refuse.cho` codes 12 to 15, all exit 8, each with a fixture in `tests/p_models.cho`); section 4.1 gains `dtype 2 = f64` for a
tensor record and the model kinds 3 (linear), 4 (logistic) and 5 (k-means), not yet in `tools/ml/mfile.cho`; `docs/rng.md` gains the
stream kind 3 (k-means). Section 6 is kept: every decisive sum of the three estimators is the fixed block-and-tree order, so the
thread count changes no byte (the gate compares the raw sums and the loss and inertia lines, not only the coefficients).

# cancho-ml encoders for categorical columns (stage 2 of issue 21)

**Status: sections 1 to 8 (the gates, the rules, the grammar, the payload, the command line, the refusals, the limits) were written before any code of this stage.
Section 9 holds the results; a number or a verdict is in it only if a script of this repository printed it, with the conditions.** Tags as in `design.md`:
**[choice]** a decision the maintainer can overturn, **[read]** read from a document or the code, **[measured]** printed by a script here, **[open]** not decided.
The decisions of the issue (alpibrusl/cancho-ml#21) are binding and are quoted where they apply. Stage 1 (`docs/categorical.md`) gave the reader, the
vocabulary, the ids and the limits; this stage gives the **steps that read them**: one-hot and label encoding of a text column, a rare-category bucket,
frequency encoding and hashing, with a plan grammar, a fitted state that carries the vocabulary, and a byte layout for that state.

**What this stage does not do.** It does not write a categorical column into a model file: stage 3 (the string table in `CML1`) is not here. Until stage 3
`ml fit` fits and scores categorical plans in one invocation and writes **nothing** (section 6), so that no file exists that `predict` could not reproduce.
Stage 3 is designed to be a mechanical addition (section 5.4): everything it has to store is already one explicit object, `FitVocab`, with an encoder and a decoder
tested here. Target encoding, native categorical splits in trees and free text are not in this stage.

## 1. The gates (fixed before the code; each can fail, section 9 shows it failing)

A gate is a script or a test run by `scripts/check.sh`. "Can fail" is shown by a **mutant** (`scripts/mutants_enc.py`: a one-line change to a source of this
repository, built and run in a copy of the tree). A survivor is listed with its reason in section 9.

| id | gate | judged by | a mutant it must kill |
|---|---|---|---|
| E1 | **Oracle, bit for bit.** `enc-check` (a driver, `tools/enc-check`) fits a plan on the categorical and numeric columns of a CSV file and prints the transformed training matrix as the bits of its doubles, the column names, the steps payload (hex) and, for a second file read through the fitted reader vocabulary, the transformed matrix and the unknown count. `scripts/oracle_enc.py` implements section 3 again in Python from the **cells** (strings), not from ids: its own counting, its own bytewise sort, its own rare bucket, `count / n` as Python's correctly rounded integer division, hashing with `hashlib.sha256`, and the payload of section 5.2 written with `struct`. The two agree on every file: matrix bits, names, payload bytes, unknown count, and for a refused plan the tag, step and column. Files: 400 seeded random tables (1 to 4 categorical columns, 0 to 300 rows, cardinality 1 to 70, missing cells and markers, 0 to 2 numeric columns) with random plans over the four encoders, and the adversarial family of E4 to E6 | `scripts/gate_enc.py diff` | one-hot position off by one; label from 1; first-appearance vocabulary; freq with the denominator of the non-missing rows; freq rounded by a multiplication by `1/n`; the unknown rule swapped; the missing rule swapped |
| E2 | **Row order.** For each file, five seeded permutations of its data rows give the same payload bytes, and the transformed row of a permuted row is the transformed row of the original (a row's output does not depend on its place); a fit that is refused is refused in every order | `scripts/gate_enc.py order` | **a vocabulary ordered by first appearance must fail** (`e2-first-appearance`); counts accumulated in an order-dependent float; the bucket named by the first merged category |
| E3 | **Thread count.** `ml fit --cat ... --split ... --threads T` for T in 1, 2, 4, 8, 16 gives the same standard output byte for byte (coefficients, `steps_sha256`, the held-out metrics) | `scripts/gate_enc.py threads` | the thread count mixed into the digest; a per-thread count of categories |
| E4 | **Unknown and missing.** A second file with unseen strings, markers, empty cells, other column order and a vocabulary column absent: unseen cells are counted and give an all-zero block (one-hot, hash never), `-1.0` (label), `0.0` (freq), or `ml.input.unknown-category` with `unknown=refuse` (step, column, row); a missing cell follows `missing=pass|zero` | `scripts/gate_enc.py diff` (adversarial family) | `unknown=refuse` ignored; a missing cell counted as unseen; an unseen string mapped to the nearest category |
| E5 | **The rare bucket.** `min-count=N` merges the categories whose training count is below N into `__other__`: counts N-1, N and N+1 (the tie at the threshold keeps the category), N = 1 merges nothing, N above every count merges all, the bucket sorts bytewise among the kept categories, a real category named `__other__` is refused with `ml.input.reserved-category` when N >= 2, and is an ordinary category when N = 1; the merged counts add up | `scripts/gate_enc.py diff`, `tests/enc.cho` | threshold `<` for `<=`; the bucket always last; the collision not checked; merged counts not added |
| E6 | **Hashing.** The bucket of a string is the first 8 bytes of SHA-256, read big-endian as an unsigned 64-bit integer, modulo B. Known answers (SHA-256 of `abc`, of the empty string and of a 1,000-byte string; the bucket of each at B = 2, 7, 1024); 5,000 random byte strings (including strings that differ in one byte, multi-byte UTF-8, NUL, 65,536 bytes) against `hashlib` at B in {2, 3, 7, 16, 1000, 1024}; strings whose top bit of the 8-byte value is set (the unsigned modulo); the output does not depend on the vocabulary, the row order or the training rows | `scripts/gate_enc.py hash`, `tests/enc.cho` | little-endian read; a signed modulo; the first 4 bytes; B-1 for B; the last 8 bytes |
| E7 | **The payload.** `encode` of a fitted step equals the bytes of section 5.2 written by hand in the test; `decode` of them returns the same vocabulary, counts and parameters, bit for bit; `apply` of the decoded step equals `apply` of the fitted one; every truncation and every field out of its bound (section 5.3) is refused with `ml.model.malformed`, before anything sized by the bytes is allocated; unsorted strings, a count of 0, counts over the rows, a duplicate name, a bad layout number are all refused | `tests/enc.cho` | a bound removed; a field order changed; the strict ascending check removed |
| E8 | **The plan.** The canonical form (section 4.1): equivalent spellings give one text (key order, blanks, CRLF, `min-count=01`, an explicit default `min-count=1` or `missing=pass` on `encode` is dropped), canonicalising twice changes nothing, every malformed line is refused with its tag, line and column. Column resolution (section 4.2): a numeric-only step on a categorical column, `freq` or `hash` on a numeric column and a mixed list are `ml.plan.column-kind` with a repair, an encoded column cannot be encoded again, the declared columns that no step reads are `ml.input.categorical-not-encoded` | `tests/t_pipeline.cho`, `scripts/gate_enc.py plan`, `tests/conformance/test_enc.py` | a default written; `*` of `encode` reaching a categorical column; the kind check removed |
| E9 | **No file before stage 3.** A fit with a categorical step and no `--split` and no `--dry-run` is refused with `ml.model.vocabulary-not-stored` and writes nothing; with `--split` it fits, scores the held-out rows through the fitted steps, reports `written: false` and writes nothing (the output path does not exist afterwards, an existing file is untouched); `--dry-run` fits, reports and writes nothing. `predict` and `eval` of a model that was fitted without a categorical step are unchanged | `tests/conformance/test_enc.py` | the file written; the refusal removed; `--overwrite` touching an existing file |
| E10 | **Numeric inputs do not change.** The 161 fixtures of `tests/conformance/golden_numeric.json` and the constant golden of W1 give the same status, standard output and files; the canonical text of a numeric `encode` line is the text of before; a numeric plan's model file has the same bytes | `test_cat_baseline.py`, `test_wire_baseline.py`, `tests/t_pipeline.cho` | a default written into a numeric `encode`; a step number moved |
| E11 | **No trap.** A seeded fuzz of `ml fit --cat` with random plans and files (at least 2,000 runs: random bytes in plans, hostile keys and values, `*` everywhere, a column in two steps, all cells missing, one category, 1,025 categories, 2^20 distinct values, invalid UTF-8) and of `decode` of damaged payloads (`tests/enc.cho`: every truncation, every flipped byte, 3,000 seeded damages): no trap status, one valid JSON answer each time | `scripts/gate_enc.py fuzz`, `tests/enc.cho` | a bound removed in the encoders or the decoder |
| E12 | **Tables.** The new rules, the new verbs and the step kinds are in `spec.cho`, `introspect`, the schemas, the registries and the fixtures once (T1 of `cli.md` and the registry gates stay green) | `test_tables.py`, `t_registry.cho`, `check_registry.py` | as T1 |
| E13 | **scikit-learn, end to end.** Generated data with 3 categorical and 2 numeric columns; the split is the library's seeded split; `ml fit --cat ... --split` against `StandardScaler` + `OneHotEncoder` + `LogisticRegression(C = 1 / (l2 n_train))` on the same train and test files: held-out accuracy equal, ROC-AUC within 1e-6 absolute, log-loss within 1e-6 relative, coefficients within 1e-5 absolute (the tolerances of `wiring.md` section 7, fixed here before the run, `l2` > 0 because a full one-hot block with an intercept is rank deficient); a difference is reported, not hidden | `scripts/gate_enc.py sklearn` | a tolerance exceeded |

Nothing in a gate is "expected to pass": the mutants are the rows of `scripts/mutants_enc.py`, and section 9 reports the verdict of each, with the survivors.

## 2. Columns in a plan

After stage 1 a fit has two kinds of columns: the **numeric** kept columns (the float matrix, with their names) and the **declared categorical** columns (`--cat`, the ids and the
vocabulary of the reader). A plan starts with both. A **step** reads one of the two kinds, never a mix, and its verb decides which:

| verb | reads | writes |
|---|---|---|
| `impute`, `scale`, `bucket` | numeric columns (`*` = every numeric column the step receives) | numeric columns, in place |
| `encode` | `kind=onehot\|label`: numeric columns (as before) **or** categorical columns | numeric: as before. Categorical: appended (below) |
| `freq` | categorical columns (`*` = every categorical column not yet encoded) | one numeric column each, appended |
| `hash` | categorical columns (`*` as `freq`) | `buckets` numeric columns each, appended |

* **`*` in `encode` means every numeric column**, as before (a plan of stage W1 keeps its meaning); a categorical column is named. This is the one asymmetry and it is
  deliberate: `*` reaching a column the plan's author did not know about would change the width of the model silently.
* **A categorical column is read by one step and then it is a set of numeric columns.** After `encode column=sex kind=onehot` the name `sex` is no longer a categorical
  column: `freq column=sex` is `ml.plan.column-kind` (the name is a numeric column now, or not a column at all: section 4.2). One column, one encoding [choice]; two
  encodings of one column (one-hot and frequency, say) would need a plan key to name the outputs and are left for a use that asks for them.
* **Where the output goes [choice].** The numeric columns of the matrix keep their places. The columns a categorical step writes are **appended after all the columns the step
  receives**, in the order of the steps; inside one step, in the order of the categorical columns' places in the file header, and inside one column in vocabulary (bytewise)
  order. Later steps see them as numeric columns (a `scale column=*` after an `encode` also scales the one-hot columns, which is what the plan says). Names: `NAME#i` for the
  `i`-th column of a block (one-hot, hash), `NAME` for a label or a frequency.
* A declared column that no step reads is `ml.input.categorical-not-encoded` (stage 1's tag; the message now says "is read by no step"). Fitting without it would drop it silently,
  which the rule "declared, not inferred" forbids.

## 3. The encoders (exact rules)

Notation. A categorical column `c` at fit has `N` training rows. Its **fitted vocabulary** is the set of strings that occur in the training rows, sorted bytewise (the
order of `std.bytes.compare`: unsigned byte by byte, a prefix before its extensions): `s_0 < ... < s_{m-1}`, with their **training counts** `n_0, ..., n_{m-1}` (integers,
each at least 1; their sum is `N` minus the missing cells). With `--split` the vocabulary is built from the **training rows only** (the reader's vocabulary also holds the held-out strings;
they are `unseen` for the step). The strings are the bytes of the cells: nothing is normalised, folded or trimmed (`docs/categorical.md` section 6).

### 3.1 The effective categories and the rare bucket (`min-count`)

`min-count=M` (an integer, 1 to 1,000,000, default 1). A category is **merged** when `n_i < M`, **kept** when `n_i >= M` (a category at exactly M is kept: the tie keeps).
If at least one category is merged, the **effective categories** are the kept strings **and the bucket `__other__`**, sorted together bytewise; otherwise they are the kept strings
(all of them). `ne` is their number. The effective count of a kept category is `n_i`; that of the bucket is the sum of the merged `n_i`.
Every training string maps to one effective category; a string the vocabulary does not hold is **unseen** (below), never the bucket.

* `M = 1` merges nothing, whatever the data; `M` above every count merges everything, and the effective categories are `__other__` alone.
* **`__other__` never collides with a real category [choice].** If `M >= 2` and the fitted vocabulary of the column holds the string `__other__`, the fit is refused with
  `ml.input.reserved-category` (step, column), whether or not that category would itself be merged. The repair is stated in the hint: rename that value in the file, or
  use `min-count=1`. With `M = 1` the string is an ordinary category (there is no bucket). There is no escape syntax: an escape would make two spellings of one category.
* The bucket is placed **in sorted position**, not last: `ne` strings in one bytewise order is one rule, the label ids and the one-hot columns follow it, and the position
  does not depend on the data except through which strings are kept.
* `ne` is bounded by `max-categories` (1 to 1,024, default 64): `ne > max-categories` is `ml.limit.categories` (step, column), as `encode` always did. The reader has its own
  `--max-categories` for the number of distinct strings it will intern; the two compose, and the smaller one of them that is exceeded is the one that refuses: a column of 500
  strings read with `--max-categories 1024` and encoded with `min-count=20 max-categories=32` is the use case of the bucket [answers question 2 of `categorical.md` section 10].

### 3.2 `encode kind=onehot` on a categorical column

A block of `ne` columns replaces the column (appended, section 2): cell `(r, j)` is `1.0` when the row's effective category is the `j`-th, else `0.0`.

* **Unseen at apply** (a string the fitted vocabulary does not hold, or, with a stored reader vocabulary, id -2): `unknown=ignore` (default): the block is all `0.0` and the step's
  unknown count grows by one; `unknown=refuse`: `ml.input.unknown-category` (step, column, row). Never guessed.
* **Missing** (id -1): `missing=pass` (default): every cell of the block is missing (the matrix bit set, the value `0.0`), the existing rule of `transforms.md` 2.1 for a one-hot block,
  so a linear model refuses the matrix with `ml.input.missing-value` until the cell is dealt with; `missing=zero`: the block is all `0.0` and the cells are not missing. A missing cell is **not**
  unseen and is not counted as unknown. **[choice]** The other candidate, a distinct indicator column, would change the width by one for every column whether or not the data
  has a missing cell; `zero` plus a separate `__missing__` category in the file (an `NA` that is not a marker) says the same with no new mechanism.

### 3.3 `encode kind=label`

One column `NAME` with `float(j)`, `j` the place of the row's effective category among the `ne` (so `0.0 .. ne-1`). Unseen: `unknown=ignore`: `-1.0`, counted; `refuse`: as above.
Missing: `pass`: the cell is missing; `zero`: `-1.0` (**the same value as an unseen string** [choice]: an estimator cannot tell them apart, a one-hot column can). A label is an ordering the
data does not have; the step exists because the issue lists it and tree models (later) can use it.

### 3.4 `freq`

One column `NAME` with `float(ec) / float(N)`: `ec` the effective count of the row's effective category, `N` the number of **training rows** (all of them, the missing ones included;
the denominator is a number the plan's reader can state without knowing the data's missing pattern). Both are integers below 2^53, converted exactly, then **one** IEEE division:
the result is the correctly rounded `ec / N`, independent of row order and thread count (integer counts, no float accumulation). Unseen: `unknown=ignore`: `0.0`, counted; `refuse`: as above.
Missing: `pass`: the cell is missing; `zero`: `0.0`. With `min-count`, all merged categories share the frequency of the bucket.

### 3.5 `hash column=... buckets=B`

For a category string `s`: `d = SHA-256(s)` (the bytes of the cell, FIPS 180-4), `h = d[0..8)` read **big-endian as an unsigned 64-bit integer**, the bucket is `h mod B`, with `B` an
integer from 2 to 1,024 (a key with no default). The output is a block of `B` columns `NAME#0 .. NAME#(B-1)`: `1.0` in the bucket's column, `0.0` elsewhere. **No sign trick**
(no second hash for a sign): collisions add up in the same column and are documented, not corrected: two strings with the same `h mod B` are the same category to the model, and
for `B` much smaller than the number of distinct strings most columns hold several. The expected number of strings that share a column with another, for `m` strings in `B` buckets,
is the birthday count; the step does not report it. No randomness anywhere: the result is a function of the bytes of the cell and of `B`, independent of the vocabulary, the training
rows, the row order, the thread count and the machine.
* **Missing**: `pass` (default): the whole block is missing; `zero`: all `0.0`. There is no unknown rule (every string has a bucket); a string whose bytes the reader could not give
  (id -2 of a stored vocabulary) cannot be hashed and is an all-zero block that **counts as unknown** (stage 2 never meets this in `ml`, which hashes through the reader's full vocabulary
  at fit; stage 3 must read hashed columns with a mode that hashes the cell, section 5.4).
* A hashed column still passes through the reader's dictionary at stage 2, so the reader's bounds (`--max-categories` up to 1,024, the pool) apply to it. The privacy point of the issue
  ("the hashing encoder is the option that does not store strings") holds for the **step**: its fitted state is `B` and the column's name and **holds no category** (section 5). The reader
  change that hashes without interning is stage 3's.

## 4. The plan

### 4.1 Grammar (`docs/pipeline.md` section 2 is extended; the keys of the other verbs are unchanged)

| verb | keys in canonical order | values |
|---|---|---|
| `encode` | `column` `kind` `max-categories` `unknown` then, **only when not the default**, `min-count` `missing` | as before; `min-count` 1 to 1,000,000 (default 1, omitted when 1); `missing` `pass` `zero` (default `pass`, omitted when `pass`) |
| `freq` | `column` `max-categories` `unknown` `min-count` `missing` | `column` a list of categorical names or `*`; `1 .. 1024` (default 64); `ignore` `refuse` (default `ignore`); `1 .. 1000000` (default 1); `pass` `zero` (default `pass`) |
| `hash` | `column` `buckets` `missing` | list or `*`; `2 .. 1024` (required); `pass` `zero` (default `pass`) |

**A numeric `encode` line has the canonical text it always had** (E10): `min-count` and `missing` are written only when they differ from their defaults, so the plan stored in a
model file of a numeric pipeline is unchanged byte for byte. The two other verbs are new and write every key (`every default written`). A rule of the table-driven grammar
(`~default`, "omitted when equal to the default") makes this a property of the table, not of a special case in the code. `min-count` and `missing` on a line whose columns are numeric,
with a value other than the default, are `ml.plan.column-kind`.

### 4.2 Resolution at fit (the order of the checks)

For each step, in order: (1) every named column is looked up **first among the numeric columns the step receives, then among the categorical columns not yet encoded**; a name in
neither is `ml.plan.unknown-column` (as before). (2) The class of the step's columns (all numeric or all categorical) is decided: a list that mixes the two classes, a
numeric-only verb (`impute`, `scale`, `bucket`) with a categorical column, `freq` or `hash` with a numeric column, and a `min-count` or `missing` on numeric columns, are
`ml.plan.column-kind` (step, the 1-based place of the offending name in the step's list, what the step needs: `numeric` or `categorical`). The repair is `retry` with a
`--step` that puts `encode column=NAME kind=onehot` before the step when the plan came from `--step` and the name is plain, and `none` with that sentence as the reason otherwise.
(3) The fit and the apply on the training rows. `*` is every numeric column for the numeric verbs and for `encode`, every not-yet-encoded categorical column for `freq`
and `hash`; a `*` that selects no column is `ml.plan.column-kind`.

## 5. The fitted state

### 5.1 The explicit object

Every categorical step carries a **`FitVocab`** (`lib/transforms/catenc.cho`): per column of the step, its **name**, and its training vocabulary: the sorted strings and their
integer training counts. With the step's parameters (`unknown`, `missing`, `param` = `min-count` or `buckets`, and `rows` = `N`) this is the **whole** fitted state: the effective
categories, the bucket, the frequencies are computed from it by the rules of section 3, never stored separately, so there is one source of truth and nothing to skew. A `hash`
step has columns and no strings. `FitVocab` has `encode` and `decode` to the bytes of section 5.2, tested here (E7). The fit result (`pipeline.run.Fitted`'s payload) holds the
steps in the order of the plan, each as its payload; the in-memory `decode` of that payload (`pipeline.mixed.step_at`) returns the `FitVocab` back.

### 5.2 The byte layout of a categorical step (new step kinds 7 to 10, little endian)

```
u32 kind          7 onehot, 8 label, 9 freq, 10 hash        (lib/core/kinds.cho)
u32 layout        1
u32 k             number of columns, 1 .. 4096
u32 unknown       0 ignore, 1 refuse                          (kind 10: 0)
u32 missing       0 pass, 1 zero
u32 param         kinds 7 to 9: min-count, 1 .. 1,000,000;  kind 10: buckets, 2 .. 1,024
u64 rows          N, 1 .. 2^31 - 1
FitVocab, k times:
  u32 name_len    1 .. 255          name bytes
  u32 ncat        kinds 7 to 9: 1 .. 1,024;  kind 10: 0
  ncat times:  u32 len (1 .. 65,536)  bytes  u64 count (1 .. rows)
```
The file section is not touched: a step of kind 7 to 10 is outside `transforms.steps.decode` (kinds 1 to 6 only), so `pipeline.file.decode` refuses a `steps` section that holds one with
`ml.model.malformed`, and nothing writes one. The bytes exist so that stage 3 stores them; they are the payload of `Fitted` and are what `enc-check` prints and `oracle_enc.py` writes again.

### 5.3 What `decode` checks (before anything sized by the bytes is allocated)

The layout number, `k`, every length against the bytes that remain, the parameter ranges, `unknown` and `missing` in {0, 1}, names non-empty and distinct within the step,
strings strictly ascending bytewise within a column, each count between 1 and `rows`, the sum of a column's counts at most `rows`, `ncat` zero exactly for kind 10, and no trailing byte. Anything
else is `ml.model.malformed`.

### 5.4 What stage 3 must serialise (and do)

1. The `FitVocab` of every categorical step, **as section 5.2 writes it** (or the same object moved to a string-table section: `FitVocab.encode` is the one function either way).
2. The schema type code 3 (`kinds.schema_type_categorical`) for each categorical input column, and the `--na-values` markers and the reader limits that built the vocabulary (a marker honoured at fit and not at predict skews silently).
3. At predict, the reader's vocabulary is built **from the steps' `FitVocab`s** (by name): the loader's `given` mode already does this, and `apply` already maps ids by string, so a stored vocabulary is
   used as it is.
4. For `hash` columns, a loader mode that hashes the cell bytes and interns nothing (so that an unseen string, which has no id, still has a bucket and the bounds of the dictionary do not apply).
5. The checksum, the bounds and the damaged-file fuzz of the string table (stage 3's gates), and `pipeline.file.decode`'s knowledge of kinds 7 to 10 (the chain of widths: a categorical step adds columns).

## 6. The command line, and what `fit` does about files

`--cat NAMES` and its flags are stage 1's. With `--cat`, a planned model (`--model linear|logistic`, `--plan`, `--step`) reads the file with the categorical loader, fits the plan on the **training rows** (all, or
the `--split` train side) with the categorical steps, and fits the estimator on the numeric matrix the steps leave. `--model constant` has no steps: it is refused as before
(`ml.input.categorical-not-encoded`).

**No model file with a categorical step before stage 3 [choice, the honest option]:** the file format cannot hold the vocabulary, so a file would not reproduce the fit.

| invocation | result |
|---|---|
| a categorical step, no `--split`, no `--dry-run` | `ml.model.vocabulary-not-stored` (exit 8), repair `none`, hint: add `--split train=F` to fit and score in one invocation without writing a model; nothing is written. The refusal comes **after** the plan and the data have been checked, so a plan error is reported first |
| a categorical step and `--split train=F [--seed N]` | the fit on the train rows, the held-out metrics from the **fitted steps** applied to the test rows (the same `apply` that stage 3's predict will use), `written: false` in the answer, no `path`, `bytes` or `sha256`, the summary with `steps_sha256` (SHA-256 of the steps payload, so the fitted vocabulary and counts can be compared) and the categorical columns' sizes. `--output` is still required by the command line (the flag table is unchanged) and is **not** touched: an existing file stays, a missing one is not created |
| a categorical step and `--dry-run` | as above without a split; exit 9 as every dry run |
| no categorical step (numeric plan, with or without `--cat` of a column some step reads: not possible) | unchanged |

The held-out metrics path is `fit`'s own (`wiring.md` W7): the test side is `ceil(n * (1 - F))` rows, the same split as `eval --split`. `predict` and `eval` take no categorical column
before stage 3 (`ml.input.non-numeric` as before).

## 7. Refusals

| tag | exit | when | `detail` | repair |
|---|---|---|---|---|
| `ml.plan.column-kind` (new) | 8 | section 4.2 | `step`, `column` (the name), `needs` (`numeric` or `categorical`) | `retry` with `--step "encode column=NAME kind=onehot; ..."` for a numeric-only verb on a categorical column from `--step`; else none |
| `ml.input.reserved-category` (new) | 8 | `min-count` >= 2 and the vocabulary holds `__other__` | `step`, `column` | none: rename the value or use `min-count=1` |
| `ml.model.vocabulary-not-stored` (new, wiring) | 8 | section 6 | `columns` (the encoded categorical columns) | none |
| `ml.input.categorical-not-encoded` (stage 1) | 8 | a declared column that no step reads | `columns`, `categories`, `vocabulary_sha256` (of the reader's whole vocabulary, as stage 1) | none |
| `ml.limit.categories` (reused) | 8 | effective categories > `max-categories` | `step`, `column` (the name) | none |
| `ml.input.unknown-category` (reused) | 8 | unseen string with `unknown=refuse` | `step`, `column` (the name), `row` | none |
| `ml.fit.all-missing` (reused) | 8 | every training cell of a column of a `encode` or `freq` step is missing (`hash` learns nothing and accepts it) | `step`, `column` | none |
| `ml.input.too-many-columns` (reused) | 8 | the matrix the steps leave is wider than `--max-columns` | as before | none |
| `ml.plan.*`, `ml.param.out-of-range` (reused) | | the grammar of section 4.1 | line, column | none |

## 8. Limits (all before allocation)

`k <= 4096` columns in a step, names to 255 bytes, strings to 65,536 bytes (the reader's ceiling), categories per column to 1,024, `min-count` to 1,000,000, buckets 2 to 1,024, the
payload bounded by the bytes that exist, the output width by `--max-columns` (`matrix.make` compares it before allocating), and the plan limits of `pipeline.md` section 2. A column of 1,025 distinct
strings is refused by the reader (`ml.limit.categories`, stage 1) and, if the reader's bound is raised to its ceiling, a step with `max-categories` below the effective count refuses it again.

## 9. Results

*(written after the gates ran)*

## 10. Open questions

1. A second encoding of one column (section 2) and a plan key to name the outputs.
2. A distinct missing indicator (section 3.2), if a use needs it.
3. The hashing reader mode and the string table (stage 3, section 5.4).
4. The speed of the per-cell remap tables is not measured; no speed is claimed.

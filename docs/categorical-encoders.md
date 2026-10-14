# cancho-ml encoders for categorical columns (stage 2 of issue 21)

**Status: sections 1 to 8 (the gates, the rules, the grammar, the payload, the command line, the refusals, the limits) were written before any code of this stage.
Section 9 holds the results; a number or a verdict is in it only if a script of this repository printed it, with the conditions.** Tags as in `design.md`:
**[choice]** a decision the maintainer can overturn, **[read]** read from a document or the code, **[measured]** printed by a script here, **[open]** not decided.
The decisions of the issue (alpibrusl/cancho-ml#21) are binding and are quoted where they apply. Stage 1 (`docs/categorical.md`) gave the reader, the
vocabulary, the ids and the limits; this stage gives the **steps that read them**: one-hot and label encoding of a text column, a rare-category bucket,
frequency encoding and hashing, with a plan grammar, a fitted state that carries the vocabulary, and a byte layout for that state.

**What this stage does not do.** It does not write a categorical column into a model file: the string table (`docs/stringtable.md`) does, and the sections below that say "before stage 3" describe this stage as it was merged. `FitVocab`, the one explicit object that holds everything a categorical step has to store, has an encoder and a decoder here; the string table stores it as it is. Target encoding, native categorical splits in trees and free text are not in this stage.

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
| E9 | **The file of a categorical fit.** A fit with a categorical step writes the model file (the string table with it); with `--split` it scores the held-out rows in process and writes the file too; `--dry-run` fits, reports the file it would write (`planned_actions`) and writes nothing; an existing file is refused without `--overwrite`. `predict` and `eval` of a model fitted without a categorical step are unchanged | `tests/conformance/test_enc.py`, `test_stringtable.py` | the file not written; `--overwrite` ignored; a dry run that writes |
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
  (id -2 of a stored vocabulary) cannot be hashed and is an all-zero block that **counts as unknown** (`enc-check` meets this; `ml predict` does not: the reader of a model file hashes the cell itself,
  `docs/stringtable.md` section 5, so an unseen string is in its bucket and is not counted).
* A hashed column still passes through the reader's dictionary at stage 2, so the reader's bounds (`--max-categories` up to 1,024, the pool) apply to it. The privacy point of the issue
  ("the hashing encoder is the option that does not store strings") holds for the **step**: its fitted state is `B` and the column's name and **holds no category** (section 5). The reader
  mode that hashes without interning is the reader of a model file (`docs/stringtable.md` section 5); the reader of a fit still interns.

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
A step of kind 7 to 10 is outside `transforms.steps.decode` (kinds 1 to 6 only). In a model file the bytes are an entry of the string table and the `steps` section holds a stub (`docs/stringtable.md` sections 2 and 3); they are the payload of `Fitted` and are what `enc-check` prints and `oracle_enc.py` writes again.

### 5.3 What `decode` checks (before anything sized by the bytes is allocated)

The layout number, `k`, every length against the bytes that remain, the parameter ranges, `unknown` and `missing` in {0, 1}, names non-empty and distinct within the step,
strings strictly ascending bytewise within a column, each count between 1 and `rows`, the sum of a column's counts at most `rows`, `ncat` zero exactly for kind 10, and no trailing byte. Anything
else is `ml.model.malformed`.

### 5.4 What the string table serialises

`docs/stringtable.md` stores: (1) the `FitVocab` of every categorical step as section 5.2 writes it; (2) the schema type code 3 (`kinds.schema_type_categorical`) for each categorical input column, and the `--na-values`
markers and the reader limits that built the vocabulary (a marker honoured at fit and not at predict would skew silently); (3) nothing else, the reader's vocabulary at predict being built from the steps' `FitVocab`s by
name, and `apply` mapping ids by string. For `hash` columns the reader of a model file has a mode that hashes the cell bytes and interns nothing (section 3.5). The checksum, the bounds and the damaged-file fuzz of the table
are its own gates (S1 to S13); `pipeline.file.decode` knows kinds 7 to 10 through the stubs and the chain of widths (a categorical step adds columns).

## 6. The command line, and what `fit` does about files

`--cat NAMES` and its flags are stage 1's. With `--cat`, a planned model (`--model linear|logistic|tree|forest`, `--plan`, `--step`) reads the file with the categorical loader, fits the plan on the **training rows** (all, or
the `--split` train side) with the categorical steps, and fits the estimator on the numeric matrix the steps leave. `--model constant` has no steps: it is refused as before
(`ml.input.categorical-not-encoded`).

**With a categorical step `fit` writes the model file** (`docs/stringtable.md` section 6), under the rules of every fit (`--output`, `--overwrite`, `--dry-run`, `--max-bytes`, atomic write):

| invocation | result |
|---|---|
| a categorical step | the fit on all the rows, the file written with its string table, `written: true`, `path`, `bytes`, `sha256`, and in the summary `steps_sha256`, the `categorical` columns' sizes and `string_table` (entries, strings, pool bytes, bytes, SHA-256 of the section) |
| a categorical step and `--split train=F [--seed N]` | the fit on the train rows, the held-out metrics from the **fitted steps** applied to the test rows (the same `apply` that `predict` uses), and the file written; `eval --split` with the same seed on the file gives the same metrics string for string |
| a categorical step and `--dry-run` | as the first row without writing; exit 9 as every dry run, with `planned_actions` |
| a plan with no categorical step (and no `--cat`) | unchanged: `pipeline.run`, the model file and every byte as before (E10); a declared column that no step reads is `ml.input.categorical-not-encoded` |

**A tree or a forest follows the same table.** The steps leave a numeric matrix, and the estimator of the blob interface (`docs/wiring-trees.md` section 13) is fitted on it; the answer has the `categorical` list, `string_table` and the `forest` header (`trees`, `nodes`, `bins`, `max_depth`, `min_leaf`, `mtry`) in place of the coefficients. A missing numeric cell is accepted by the tree and forest as in a numeric plan (the encoders handle a missing text cell by their own `missing` rule). The fixtures `fit-tree-categorical-writes-the-file`, `fit-forest-categorical-writes-the-file`, `fit-forest-categorical-split-scores-held-out-rows`, `fit-tree-categorical-split-takes-a-missing-cell` and `fit-tree-categorical-dry-run` hold it.

The held-out metrics path is `fit`'s own (`wiring.md` W7): the test side is `ceil(n * (1 - F))` rows, the same split as `eval --split`. `predict` and `eval` read the categorical columns from the string table (`docs/stringtable.md` section 5).

## 7. Refusals

| tag | exit | when | `detail` | repair |
|---|---|---|---|---|
| `ml.plan.column-kind` (new) | 8 | section 4.2 | `step`, `column` (the name), `needs` (`numeric` or `categorical`) | `retry` with `--step "encode column=NAME kind=onehot; ..."` for a numeric-only verb on a categorical column from `--step`; else none |
| `ml.input.reserved-category` (new) | 8 | `min-count` >= 2 and the vocabulary holds `__other__` | `step`, `column` | none: rename the value or use `min-count=1` |
| `ml.model.string-table`, `ml.limit.string-table` (the string table) | 8 | `docs/stringtable.md` section 7 | | |
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

Everything below was printed by a script of this repository (`sh scripts/check.sh` runs all of it; `docs/bench/gate_enc.log` and `docs/bench/mutants_enc.log` are the logs of the
gate and of the mutants), on one machine: this Mac, arm64, macOS, compiler `a4572ea`, scikit-learn 1.8.0, numpy 2.4.6, Python 3.14.5. The machine was **not quiet** (1-minute load average 3 to 6.6 during the
runs, other agents building); correctness gates do not depend on the load and no timing is a claim. **No benchmark was run**, and the cost of the per-cell remap tables, of the second pass of the reader and
of the appended columns is therefore not measured.

| gate | result |
|---|---|
| E1/E4/E5 oracle, bit for bit | `tools/enc-check` against `scripts/oracle_enc.py` (own counting from the cells, `hashlib`, `struct`): the transformed matrix, the names, the payload bytes, the unknown count and every refusal (tag, step, column, row) agree in **515 runs, 0 differences**: 400 seeded random tables (166 answered, 234 refused by the oracle's own verdict: the reader's limits, a plan mistake, an unseen category with `unknown=refuse` in the predict file, ...) with random plans over the four encoders, `min-count` 1 to 50, `unknown`, `missing`, `max-categories`, a numeric min-max scale in between, a second file read through the first one's vocabulary (80 percent of the tables); and the corner families: counts N-1, N and N+1 at the threshold for N in 1 to 7 and 100, the bucket among kept strings that sort before and after `_`, `__other__` as a real category at `min-count` 1 and 2, one category, all cells missing, NFC against NFD, Latin `a` against Cyrillic `a`, CJK, emoji, a NUL and quotes in a cell, markers, two columns in two orders, the `column-kind` and unknown-column refusals, 9, 10 and 30 strings against `max-categories`, 65 strings against the reader's bound of 64, 1,024 and 1,025 distinct strings, and four all-unseen predict files |
| E2 row order | 50 tables (35 accepted at every order), 5 seeded permutations each: the payload bytes are identical and every row of the matrix follows its row: **250 runs, 0 differences**. The mutant `e2-first-appearance` (a vocabulary in the order the rows meet the strings, every cell still found) is killed by this gate alone |
| E3 thread count | 8 fits that were answered (37 tables tried: a fit that is refused says nothing about threads, and the first version of this gate counted refusals; see below), `--threads` 1, 2, 4, 8, 16: one standard output each: **40 runs, 0 differences** |
| E6 hash | 6 values of B, 36 tables of 900+ strings each (random UTF-8, strings that differ in one byte, `\u00a0`, emoji): the matrix equals the oracle's and the position of the 1 equals `int.from_bytes(sha256(s)[:8], "big") % B` read by hand: **72 runs, 0 differences**; `tests/enc.cho` has the known answers (`abc`, the empty string, `a`) |
| E7 payload | `tests/enc.cho`, 13 tests (12 of the encoders, one of a file with a string table): hand-computed matrices for one-hot, label and frequency, the unseen and missing rules, the rare bucket (threshold, place, collision, bound), the bytes of a step written by hand and decoded again, 40 fields edited out of their bound, every truncation, every byte flipped three ways and 3,000 seeded damages (each is refused or encodes back to the same bytes), name and category bounds exactly, the fit against columns that are not there, and the mixed pipeline fitted, stored and applied again to the same bits |
| E8 plan | `tests/t_pipeline.cho` (2 new tests), `oracle_pipe.py --plans` (the Python grammar now knows `freq`, `hash`, `min-count`, `missing` and the `~default` rule: 1,500 respelled plans, 1,500 fixed points, 1,194 float spellings, 1,500 garbage plans with the same verdict and position), `tests/conformance/test_enc.py` (20 tests) |
| E9 the file of a categorical fit | `test_enc.py` (class `Files`: the file written and what the summary says of it, an existing file kept without `--overwrite`, `--split` writes and scores, a dry run reports the file and writes nothing) and the fixtures of `docs/stringtable.md` section 10 |
| E10 numeric inputs | the 161 fixtures of `golden_numeric.json` and `golden_constant.json`: status, standard output and files are the record's, **but one**: see "the one change of a numeric case" below |
| E11 no trap | `gate_enc.py fuzz`: **3,024 runs** (hostile plans and files through `enc-check` and `ml fit --cat`, 4,097 categorical columns, a 70,000-byte cell, 2^20 distinct values, 1,025 distinct values at the ceiling, a million empty cells, a plan of 70 lines, invalid UTF-8): 0 traps, 0 malformed answers; the damage tests of E7 |
| E12 tables | 2 new rules (`ml.plan.column-kind`, `ml.input.reserved-category`), 35 flags (none new), 13 limits; every one reached by a fixture (`test_tables.py`); the registries (66 refusal codes in 4 bands, 10 step kinds) agree with the sources and with `design.md` |
| E13 scikit-learn | 3 generated datasets (1,500 rows, 3 categorical columns of 4, 6 and 3 values and 2 numeric ones), the library's seeded split (`--split train=0.75 --seed 7`, 1,125 and 375 rows; the rows from `oracle_metrics.py`), `l2` 0.1 against `LogisticRegression(C = 1 / (0.1 * 1125), lbfgs, tol 1e-12)` on `StandardScaler` + `OneHotEncoder(handle_unknown="ignore")` of the same rows. **Tolerances were fixed in section 1 before the run, and all PASS**: held-out accuracy equal (0.776, 0.7467, 0.76), ROC-AUC equal to the last digit shown (0.86667, 0.82407, 0.86184), log-loss relative difference 3.1e-10, 1.4e-8 and 2.8e-9 (bound 1e-6), coefficients and intercept (15 numbers) at most 1.7e-8, 7.4e-8 and 1.5e-8 apart (bound 1e-5) |

**Mutants** (`scripts/mutants_enc.py`, `docs/bench/mutants_enc.log`): **57 patterns, 56 killed, 1 survivor explained, 0 unexplained**; the unmutated control passes every gate. They are on the order of the
vocabulary (first appearance, descending, a label from the row), the rare bucket (the threshold `<=`, `+ 1`, `- 1`; the bucket last and first; the collision not checked; merged counts not added; a merged category taken for unseen;
the bound on strings instead of effective categories), the encoders (a label from 1, a reversed one-hot block, the frequency by a reciprocal and with a denominator of rows plus one, the merged category's own count, the
column order, the appended columns before the others), the unknown and missing rules (never and always refused, not counted, a missing cell taken for an unseen string, `missing=zero` ignored, an unseen label of 0.0, an unseen string
mapped to the nearest category, an unseen id taken for a category), the hash (little endian, the top bit dropped, the first four bytes, the modulus B - 1, the last eight bytes, the bucket from the id), the decoder (unsorted strings,
counts over the rows, a count of 0, trailing bytes, the field order of the payload, the name length, the category ceiling), the plan and the columns (the default written, an explicit default kept, `*` of `encode` reaching categorical
columns, the kind check removed, the lookup order, an encoded column staying alive, an unread declared column silent, `min-count` on a numeric `encode` accepted), the files (the refusal removed, the file written, a split fit trained on all
rows, the held-out rows scored with the training ids), the thread count in the answer, a step kind claimed twice, a tag without a fixture, and the bound on the columns of a step. **The required one is killed by the row-order gate
alone:** `e2-first-appearance` fits the vocabulary in the order the rows meet the strings and searches it linearly, so every lookup still succeeds and only the order of the rows decides the payload; `gate_enc.py order` reports `DIFF payload bytes`.
**The survivor:** `e8-numeric-first-lookup-reversed` is equivalent (a name is both a numeric column and a not-yet-encoded categorical one only if an encoder wrote a numeric column with another declared column's name; the encoders write `NAME`, which is then no longer categorical,
or `NAME#i`, and a plan cannot name a column that holds `#`). **What the mutants found about the gates, and what was done:** the first run had 16 results that were not kills: 11 mutants did not build because `cancho install` of a
concurrent `check.sh` rewrote `build/deps` while the copies were made (re-run alone: all killed); the thread gate was vacuous (every fit it ran was refused for a missing numeric cell, so a thread count in the answer survived), and now it counts
the answered fits and fails if there are too few; three bounds had no test (a name of 256 bytes, 1,025 categories, a column index past the vocabulary), and `tests/enc.cho` now has them; a mutant's gate list named gates that could not see it.

**The one change of a numeric case.** `golden_numeric.json` is not rewritten. One of its 161 fixtures, `ml-plan-unknown-step`, prints the text of `ml.plan.unknown-step`, which lists the plan verbs; stage 2 adds `freq` and `hash`, and an
unchanged text would be false. `test_cat_baseline.py` now pins the SHA-256 of the new standard output for that one case beside the record, with the reason, and a test says that the record still holds the old hash and that the exception
list is that one case. The other 160 fixtures, and the model files among them, are byte for byte what the binary of `7d6183d` wrote.

**What the gates do not claim.** The oracle is written by the same author from the same document, with its own counting and its own payload writer, but the same understanding of the rules: it checks the code against the document, not the
document against the world; scikit-learn is the outside check, within bounds, for one-hot encoding with a penalty (not for the rare bucket, the frequency, the hash or the label encoding, which scikit-learn has no equivalent for, or not the same one).
The reading of the CSV is the stage-1 oracle's (judged against the reader by C1). The hash is checked against Python's `hashlib`, not against a second implementation of SHA-256 in this repository (`std.crypto`'s is the compiler's). Nothing was run on
a second machine or with more than the threads the kernels use (no step here is threaded; E3 says the answer does not change with the flag). The damaged-file fuzz of the string table is gate S10 of `docs/stringtable.md`.
Predict-time alignment of the numeric columns by name is the reader's and the schema's job (`docs/stringtable.md` gate S8): `pipeline.mixed.apply_steps` takes the numeric matrix in the fitted order.

## 10. Open questions

1. A second encoding of one column (section 2) and a plan key to name the outputs.
2. A distinct missing indicator (section 3.2), if a use needs it.
3. The hashing reader mode and the string table are `docs/stringtable.md`.
4. The speed of the per-cell remap tables is not measured; no speed is claimed.

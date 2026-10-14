# cancho-ml string table (stage 3 of issue 21)

**Status: this document is written before the code it judges. Sections 1 to 9 fix the layout, the bounds, the determinism, the refusals and the gates; section 10 holds the results and is
filled only with what a script of this repository printed, with its conditions.** Tags as in `design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from a
document or the code, **[measured]** printed by a script here, **[open]** not decided. The decisions of the issue (alpibrusl/cancho-ml#21) are binding: raw category strings may be
stored in the model file (a privacy point), the vocabulary is sorted bytewise, every limit is refused before allocation with a rule tag, the hashing encoder is the option that stores no
string. `docs/categorical.md` gave the reader, the vocabulary and the limits; `docs/categorical-encoders.md` gave the steps and the object that carries their state (`FitVocab`, section 5).
This stage stores that object in the model file, so that `fit` writes a model with categorical steps and `predict` and `eval` read it.

## 1. What is stored, and where

A model with a categorical step (`encode`, `freq` or `hash` of a declared `--cat` column) needs, to be applied to new rows, four things the numeric file does not hold:

1. the **fitted vocabulary** of every categorical step: its parameters, the name of each column, the sorted strings that occur in the training rows and their integer counts;
2. the **type of each categorical input column** in the schema (type code 3, reserved by `wiring.md` section 3 and `kinds.schema_type_categorical()`);
3. the **missing-value markers** (`--na-values`) and the **reader limits** the vocabulary was built with (`--max-categories`, `--max-string-bytes`, `--max-pool-bytes`, `--max-cat-columns`);
4. nothing else: the effective categories, the rare bucket, the label ids and the frequencies are computed from (1) by the rules of `categorical-encoders.md` section 3, never stored.

All of it goes in **one new section of the model file, the string table (section tag 6)**, and one stub per categorical step in the `steps` section. [choice] The alternative, to put the
vocabulary inside the step records of the `steps` section (the bytes `pipeline.mixed.fit` already makes), has no second section, but it spreads the only privacy-relevant bytes of the file
over a section whose other content is numbers, and gives the reader of the file no single place to bound, checksum and refuse before it allocates. One section with its own header, bounds and
digest is the smaller thing to audit. The cost, a reassembly of the steps payload at load time, is a copy of at most 12 MiB (section 4).

**A file without a categorical step is the file it always was** (`design.md` 4.1, byte for byte): five sections, `sections = 5`, no tag 6, no type 3. The table is present only when a step needs
it, and then `sections = 6`. This is the registry rule: a reader that knows five sections refuses a six-section file as `ml.model.malformed` (`mfile.cho`: the constant model reader checks the
count), so no old reader applies a categorical model without its vocabulary.

### 1.1 The registry entries (`lib/core/kinds.cho`, `design.md` 4.1)

| number | registry | meaning |
|---|---|---|
| section tag 6 | section tags | `strings`: the string table (this document) |
| schema type 3 | column types | a categorical input column (`scale` 0) |
| steps layout 2 | step layouts | the stub of a categorical step in the `steps` section (section 3) |
| string table layout 1 | table layouts | section 2 |
| code 328, `ml.model.string-table` | refusal codes (`transforms.codes`) | a stored string table or its stubs are damaged or disagree with the file |
| code 329, `ml.limit.string-table` | refusal codes | a fit would write a table over the bounds of section 4 |

A number is never reused. `tests/t_registry.cho` and `scripts/check_registry.py` compare these with the code and with `design.md`.

## 2. The byte layout of the string table (section tag 6; little endian)

```
u32 layout            1
u32 entries           1 .. 64                      one per categorical step, ascending by step
u64 pool_bytes        0 .. 8,388,608               sum of the lengths of every category string of the table
u64 strings           0 .. 131,072                 sum of the number of category strings of every column of every entry
u32 na_len            0 .. 16,383                  then na_len bytes: the --na-values list as given, markers separated by commas
u32 max_categories    1 .. 1,024                   the reader limits that were in force at fit
u32 max_string_bytes  1 .. 65,536
u32 max_pool_bytes    1 .. 8,388,608
u32 max_cat_columns   1 .. 4,096
entries times:
  u32 step            1 .. 64                      the 1-based place of the step in the plan, strictly ascending
  u64 length          bytes of the payload         at most the bytes that remain before the digest
  payload             the step payload of categorical-encoders.md section 5.2, layout 1, verbatim: kind (7 to 10), layout, k, unknown, missing, param, rows,
                      then per column: name_len, name, ncat, per string: len, bytes, count
32 bytes              SHA-256 of every byte of the table before these 32
```

* `FitVocab.encode` (`catenc.encode_payload`) is the one function that writes an entry; `catenc.decode` reads it. The table adds the header and the digest around them. [choice]
* The `na` list is stored as the text given, not canonicalised: it is a list of byte markers compared with the cell bytes, and the order of the markers does not change the result, but the
  bytes of the list are the user's. The reader applies it as `--na-values` does (docs/categorical.md section 5.1). At most 64 markers of 1 to 255 bytes, none empty: the limits of the flag.
* The four limits are stored because a vocabulary built with a limit is a statement about the data ("at most 64 categories") and because `predict` reads with `max_string_bytes`: a cell
  longer than any string the model could have seen is refused with the same rule as at fit. `max_categories`, `max_pool_bytes` and `max_cat_columns` are checked against the table's content
  (every column has at most `max_categories` strings, the pool is at most `max_pool_bytes`, the columns are at most `max_cat_columns`) and are not used to intern anything at predict.
* **Determinism [choice]:** the table has no field that depends on the order of the training rows, on the thread count, on a clock or on a map's iteration order. Strings are bytewise ascending
  inside a column, columns are in the order of the step's `column` list (ascending by header place), entries ascend by step. The same training rows in any order give the same table, schema, plan, meta and
  categorical steps, and a fit at any thread count gives the same file (gate S4). The sections that hold floats computed by sums over the rows (a standardising or imputing step, the estimator) follow the order of the rows,
  as the sums of the libraries do (`docs/linalg.md` section 2): the gate measures and prints how often the whole file is identical under a permutation, and asserts nothing about it.

## 3. The `steps` section, the schema and the meta

* **A categorical step in the `steps` section is a stub of 8 bytes:** `u32 kind` (7 to 10) and `u32 layout = 2`. The entry `step = its place` of the table is its payload. Numeric steps (kinds 1 to 6)
  are stored as before. A file never holds a categorical step with layout 1 in the `steps` section: one representation per file.
* **Reassembly.** `file.decode` rebuilds the **effective steps payload** by replacing each stub with `u64 length | payload` of its table entry. The result is byte for byte the payload
  `pipeline.mixed.fit` returns (`Fitted.payload`); `run.apply_steps`'s mixed twin, `mixed.apply_steps`, is given this payload and nothing else changes in how a step is applied (gate S6).
* **The schema** lists every column of the training file that the model reads, in file order, with the target: numeric columns keep type 0, 1 or 2; a categorical input column has type 3 and
  scale 0. The target is never type 3. The set of type 3 columns equals the set of column names of the table, each once (checked at decode).
* **The meta** is unchanged: `columns` is the number of columns of the schema (categorical ones included), `rows` the training rows. A model of kinds 4 and above keeps its `sum_order`.
* **The chain of widths.** The numeric steps start from the numeric feature columns (the schema's columns that are neither the target nor type 3, there may be none); a categorical step appends the
  columns it writes (`categorical-encoders.md` section 2). The width of the estimator's input is the width after the last step, as before.

## 4. Bounds, all checked before the thing they bound is allocated

| quantity | bound | why |
|---|---|---|
| table payload (the section length) | at most 12 MiB = 12,582,912 | worst case of a valid table is about 11.1 MiB (below), and the file as a whole must fit `--max-bytes` (default 16 MiB, ceiling 256 MiB) with the rest of the model |
| pool bytes (all strings) | at most 8 MiB = 8,388,608 | the reader's pool ceiling (`dict.ceiling_pool_bytes()`); a vocabulary the reader could build fits |
| strings in all | at most 131,072 = 2^17 | 12 bytes of framing per string (`u32` length, `u64` count): at most 1.5 MiB |
| entries | 1 to 64 | the plan has at most 64 lines (`plan.max_lines()`) |
| columns of an entry | 1 to 4,096 | `catenc.max_columns()` |
| strings of a column | 1 to 1,024 (0 for `hash`) | `catenc.max_categories()` |
| one string | 1 to 65,536 bytes | `catenc.max_string()` |
| a column name | 1 to 255 bytes | `catenc.max_name()` |
| `na_len` | at most 16,383 | 64 markers of 255 bytes and 63 commas |
| `rows` of an entry, a count | 1 to 2^31 - 1, a count at most `rows` | `catenc` |

The worst valid table: 8,388,608 (pool) + 131,072 x 12 (framing) + 4,096 x 263 (names and their framing) + about 40,000 (headers, `na`, digest) = 11,078,000 bytes, below the 12 MiB cap, which
therefore never refuses a valid table and only guards the allocation. **A fit whose table would pass a bound is refused with `ml.limit.string-table`** (exit 8, detail `what`: `strings`, `pool_bytes` or `bytes`, `size`, `max`), after the
steps were fitted and before any file is made; a fit that is held out (`--split`) or a dry run reports the same refusal, so a dry run says what a real run would do. A table within these
bounds is within `--max-bytes` at its default of 16 MiB in itself (12 MiB < 16 MiB); the file as a whole is still refused with `limit.input-too-large` at fit when it passes `--max-bytes`, as for every model.

**The decoder** reads the table in this order and stops at the first failure: (1) `length` of the section against the cap (done by the file structure check, before the section is touched); (2) the
digest of the table against its bytes; (3) the header fields against their ranges, the `na` list against its bytes and its marker limits; (4) the entries: `step` strictly ascending and in range,
`length` against the bytes that remain, then the payload by `catenc.decode` (which checks every inner length against the bytes that remain **before** it allocates anything sized by the bytes)
and the table totals (`pool_bytes`, `strings`) against the sums of the entries; (5) UTF-8 of every name and every string (`dict.utf8_offset`, the validation the reader applies to a cell); (6) the stored limits against the
content. Nothing is allocated from a count or a length that has not been compared with the bytes that exist, and every counter is compared with its bound before it is added to a sum, so a
count of 2^63 - 1 cannot overflow one. The digest comes first on purpose: a damaged table is reported as damaged, not as whichever field the damage landed on; a table whose digest was repaired by
an editor reaches the field checks, which are the defence against that. Gate S10 flips every byte, truncates at every length, repairs the digests, and writes counts of 2^63 - 1.

## 5. Reading a file with a table

* **The reader vocabulary** of `predict` and `eval` is built from the table by column **name**: the columns of every entry, in the order of the steps and, inside a step, of its column list; each
  column's strings are already ascending, so the vocabulary is the one `dict.freeze` would give for those strings (`dict.valid` holds). Its ids are the ranks of the strings of the *training* vocabulary.
  A column of a `hash` entry has no string; the reader marks it as **hash mode** with the step's number of buckets.
* **Hash mode (the hash reader).** For a cell of a hash-mode column the reader decides *missing* as for any categorical cell (empty or a marker), refuses a cell over `max_string_bytes` or not
  UTF-8 as for any cell, and otherwise writes the **bucket** `SHA-256(cell)[0..8)` big-endian modulo `B` as the id. It interns nothing, looks nothing up and counts nothing as unseen. The encoder reads the
  id as the bucket. So an unseen string hashes into its bucket exactly as in training (`categorical-encoders.md` section 3.5 stated this as the thing the reader had to do). The reader of a fit is not
  changed: a hashed column is still interned at fit and the fit's bounds apply to it; the bytes of the table hold none of its strings.
* **Encode, label and freq** keep counting unseen categories (id -2 from the reader's lookup, a zero block, `-1.0` or `0.0` by the step's `unknown=ignore`, or `ml.input.unknown-category` with step, column and row
  for `unknown=refuse`); the count is `unknown_categories` in the answer.
* **Alignment is by name, for every model.** The numeric features are found in the file by the schema's names, the categorical columns by the table's names, in any order of the file's columns. A
  column of the model that the file lacks, a column the file has that the model does not (and `--drop` does not ignore), and a column of another type are `ml.schema.mismatch`, with the repair of the numeric
  rule (`--drop` the extra columns when that is all that differs; otherwise the missing and mistyped columns by name in the hint). A numeric model's predict already aligned by name; the gate S8 covers all
  four models and both kinds of column.
* `predict` and `eval` take no flag of the categorical reader (`--cat`, `--na-values`, the four limits): the file says what to read with. The flags are `fit` flags and stay so.

## 6. What `fit` does

A fit of `linear`, `logistic`, `tree` or `forest` with categorical steps **writes the model file** (the table with it) under the rules of every fit: `--output`, `--overwrite`, `--dry-run`, `--max-bytes`,
atomic write. `--split train=F` fits on the train side, scores the held-out rows through the fitted steps **in process**, and writes the file too; the answer has `written: true` unless it is a dry run.
`ml.model.vocabulary-not-stored` **is removed**: nothing reaches it (the table can hold what the steps hold, up to the bounds of section 4, and a fit over those bounds has its own rule, `ml.limit.string-table`).
The answer's summary keeps `steps_sha256` (the digest of the effective steps payload) and `categorical`, and gains `string_table`: `entries`, `strings`, `pool_bytes`, `bytes` and `sha256` of the section.

**Privacy [read, documented in `categorical.md`, the README and `cli.md`].** The file holds the raw category strings of the training rows (every distinct value of every encoded column that is not the target,
with its count), the column names, and the markers: whoever has the model file has those values. `hash` stores none (a hashed column's entry holds its name and `B`); `encode min-count=N` still stores every
distinct string, because the counts that decide the bucket are part of the state. A pipeline that must not carry its category values hashes them.

## 7. Refusals

| tag | exit | when | `detail` | repair |
|---|---|---|---|---|
| `ml.model.string-table` (code 328) | 8 | a stored table is damaged, fails a bound of section 4, or disagrees with the steps, the plan or the schema of its own file | `reason` (a word below), `step` (the step, when one), `section` 6 | none: refit |
| `ml.limit.string-table` (code 329) | 8 | a fit would write a table over a bound | `what`, `size`, `max` | none: fewer or shorter categories, `min-count` cannot help (the counts are state), or hash the column |
| `ml.schema.mismatch` (reused) | 8 | the file lacks a column of the model, has an extra one, or one of another type | `missing`, `extra`, `mistyped` | `--drop` when only extras differ |
| `ml.input.unknown-category` (reused) | 8 | an unseen string in a step with `unknown=refuse` | `step`, `column`, `row` | none |
| `ml.model.checksum`, `ml.model.truncated`, `ml.model.malformed`, `ml.model.version`, `ml.model.bad-magic` (reused) | 8 | the file level checks of `design.md` 4.1, in that order | | |

Reasons of `ml.model.string-table`, each with a fixture: `digest`, `layout`, `entries`, `header` (pool, strings, na, limits out of range), `step` (order or range), `length`, `payload` (`catenc.decode`
refused the entry), `utf8`, `totals` (the header sums disagree with the entries), `limits` (content over a stored limit), `stub` (a stub without an entry or an entry without a stub, a stub of the wrong kind
or layout), `schema` (the type 3 columns and the table's names differ), `plan` (the plan's steps and the stored ones differ), `trailing`.

The order of the checks of the whole file is `design.md` 4.1's (magic, version, structure, checksum, sum order, then the fields); the table is checked after the schema and before the steps, because the steps are
read through it.

## 8. The file-level changes, and the numeric files they do not touch

* `pipeline.file.encode` takes an optional table payload; with none it is the code that wrote five sections. `decode` reads five or six sections; the sixth must have tag 6.
* `peek_kind` (the router between the constant reader and the pipeline) reads five or six sections.
* `mfile.cho` (constant models) still requires five.
* `pipeline.schema.decode` accepts type 3. `pipeline.mixed` gains `check_steps` and `agrees` for kinds 7 to 10 (width chain, and the plan against the stored steps), and `vocab_of` (the reader vocabulary of a payload).
* `csv.cho` / `dict.cho` / `bucket.cho`: the hash mode of the predict reader (a per-column bucket count in the vocabulary, scratch for the digest and the stored markers in the state, one `bucket_of` shared with the encoder), and `CatCfg.lenient`, with which a vocabulary column the file lacks is the schema check's `ml.schema.mismatch` (missing, extra and mistyped columns in one answer, `--drop` as the repair) and not the reader's status 29.
* **Golden records.** `golden_numeric.json`, `golden_constant.json`, `golden_planned.json` and the model files they pin keep their bytes. The fixtures that pinned `ml.model.vocabulary-not-stored`
  (`ml-model-vocabulary-not-stored`, `-tree`, `-forest`) and the ones that pinned `written: false` after `--split` with a categorical step become fixtures of the new behaviour; they are listed in section 10.

## 9. The gates (written before the code; each can fail; section 10 shows the verdicts)

A gate is a script or a test run by `scripts/check.sh`. "Can fail" is shown by a **mutant** (`scripts/mutants_st.py`: a one-line change to a source of this repository, built and run in a frozen
copy of the tree). A survivor is listed with its reason in section 10.

| id | gate | judged by | a mutant it must kill |
|---|---|---|---|
| S1 | **Oracle, byte for byte.** `scripts/oracle_st.py` implements section 2 and the file layout again in Python, from the **cells** of the CSV: its own counting, its own bytewise sort, its own payload writer (`struct`), its own SHA-256 (`hashlib`), its own schema and steps stubs, and the whole `CML1` file for a given plan (the plan text, the schema, the steps, the model section taken from the tool because the estimator is not under test, the meta, the string table, the digest). `ml fit` writes the file; the bytes of the table, of the steps section, of the schema, and of the file with the model section substituted, are equal. Files: 300 seeded random tables (1 to 4 categorical columns, a numeric id and x, 0 to 300 rows, cardinality 1 to 70, missing cells, markers, random plans over the four encoders and min-max scaling, fitted with `tree`: the estimator is not under test here, and S2, S8 and S10 cover all four models), the corner families of `categorical-encoders.md` (the rare bucket at its threshold, `__other__`, one category, all cells missing, NFC against NFD, 65 and 1,024 and 1,025 strings, ...), and the reader's adversarial files (unicode, emoji, a NUL and quotes in a cell, strings of 1 and of 65,536 bytes, markers, a BOM, quoted line breaks) through each encoder | `scripts/gate_st.py diff` | two limits written in the other order; the digest over the wrong span; the markers not stored; a limit not stored; the step number of an entry off by one; the stub layout; the type code 3 not written |
| S2 | **Round trip, bit for bit.** For every fit of S1 with `--split`, the held-out metrics `eval --split` (same seed) prints on the written file equal the in-process held-out metrics of the fit, string for string (the doubles are shortest round-trip strings, so equal strings are equal bits); and `eval` on a CSV of the held-out rows alone (taken by the oracle's replica of the split) gives the same metrics, with `predict` answering on them. Plans include one whose steps read the columns in an order other than the header's | `scripts/gate_st.py roundtrip` | a column of a step looked up in the reader's vocabulary by its place in the step; the stored markers ignored |
| S3 | **Golden.** Every numeric and categorical-free fixture and every model file of `golden_*.json` gives the same status, standard output and bytes; a model without a categorical step is a five-section file with the bytes of before; the fixtures replaced by this stage are exactly those listed in section 10 | `test_*_baseline.py`, `tests/t_pipeline.cho` | a default section written for a numeric model; the section count 6 for every file |
| S4 | **Row order and thread count.** For each S1 file, five seeded permutations of the data rows give the same schema, steps (min-max scaling is order free) and table bytes, and `--threads` 1, 2, 4, 8, 16 give the same whole file; for the four models one more probe prints how often the whole file is the same under a permutation (not asserted: see section 2); a vocabulary in first-appearance order must fail this | `scripts/gate_st.py order` | `s4-first-appearance` (the vocabulary in the order the rows meet the strings) |
| S5 | **Unseen and missing at predict.** A file whose categorical column holds unseen categories, a marker, an empty cell, columns in another order and a column absent: unseen are counted (`unknown_categories`) for `encode` and `freq`, refused with step, column and row for `unknown=refuse`; a missing cell follows `missing=pass|zero`; a `hash` column never has an unseen string, and an unseen string lands in the bucket it hashes to | `scripts/gate_st.py predict`, `tests/conformance/test_stringtable.py` | the hash mode off (an unseen string gives a zero block, or a bucket looked up as a string); a bucket equal to the number of buckets accepted; the markers of the fit not applied; the bucket computed from the halves of the digest without the weight of the high one |
| S6 | **Reassembly.** The effective steps payload of a file equals `Fitted.payload` of the fit (the summary's `steps_sha256` equals the SHA-256 of the reassembled payload read by the oracle); `catenc` applied to it gives the matrix of the fit | `scripts/gate_st.py diff`, `tests/strtab.cho` | a stub that picks the entry of another step |
| S7 | **Bounds before allocation.** Each row of section 4 at its limit and one past it, at the decoder (`tests/strtab.cho`) and at the fit (`gate_st.py limits`: 131,072 strings accepted if the pool allows, 131,073 refused with `ml.limit.string-table`); a length or a count larger than the bytes that exist, up to 2^63 - 1, in any field of a table is refused (an allocation sized by such a field would trap or exhaust memory, so the test passing is the proof that none is made) | `tests/strtab.cho`, `scripts/gate_st.py limits` | a bound removed or moved by one (each of the bounds of section 4 has a mutant) |
| S8 | **Alignment by name.** For `linear`, `logistic`, `tree` and `forest`: the columns of the predict file in each of several permutations give identical predictions (bytes of the answer); a missing, an extra and a retyped column are `ml.schema.mismatch` with the repair of the numeric rule; a categorical column dropped by `--drop` is missing | `scripts/gate_st.py align`, `test_stringtable.py` | the numeric features taken by position; a column of a step looked up by its place; a missing categorical column that is the reader's refusal and not the schema check's |
| S9 | **Privacy.** A byte string put in a categorical column occurs in the model file exactly when the step stores strings: for `encode` and `freq` it is found in the table; for `hash` the file holds no byte of any category (searched for every string of the column) | `scripts/gate_st.py privacy` | `hash` storing the strings |
| S10 | **Damaged file, never a trap.** Every byte of a file with a table flipped (three ways), every truncation, the checksum repaired after each edit (file digest, and table digest), every `u32` and `u64` field of the table set to its extremes (0, 1, 2^16, 2^31 - 1, 2^31, 2^32 - 1 and, for a `u64`, 2^32, 2^62, 2^63 - 1, -1, -2^63), bytes that are not UTF-8 in names and strings, and seeded damage (changed, cut, inserted and appended bytes); each is answered by one valid JSON line, never a trap status, and with the tag the oracle names when its file-level or table checks refuse the file. Duplicate names, unsorted strings and counts over `rows` are in `tests/enc.cho` and `tests/strtab.cho`, which do the same flips, cuts and extremes on the decoder alone | `scripts/gate_st.py fuzz`, `tests/strtab.cho` | any bound removed; the digest check removed; the UTF-8 check removed |
| S11 | **scikit-learn, end to end.** Generated data with 3 categorical and 2 numeric columns; a seeded train/test split written as two CSV files; `ml fit` on the train file writes a model file; `ml predict` and `ml eval` on the **test file from the file** agree with `OneHotEncoder` + `LogisticRegression` and `OneHotEncoder` + `RandomForestClassifier` fitted on the same train file. Tolerances fixed here, before any run: logistic (l2 > 0, `C = 1 / (l2 n_train)`, `lbfgs`, `tol 1e-12`): predicted labels identical on every test row, probabilities within 1e-6 absolute, accuracy equal, ROC-AUC within 1e-6 absolute, log-loss within 1e-6 relative. Random forest: the two libraries grow different trees (different bootstraps and split search), so equality is not claimed; the tolerance is a choice fixed here: held-out accuracy within 0.05 absolute and ROC-AUC within 0.05 absolute of scikit-learn's, on three datasets; and `ml eval --split` on the written file equals the held-out metrics of `ml fit --split` on the same rows, string for string (S2) | `scripts/gate_st_sklearn.py` | a tolerance exceeded |
| S12 | **Tables.** The new rules and the new section tag are in `spec.cho`, `introspect`, the schemas, the registries and the fixtures once; the removed rule is in none; every tag of section 7 is reached by a fixture | `test_tables.py`, `t_registry.cho`, `check_registry.py` | a tag without a fixture; the removed rule still in a table |
| S13 | **Prose.** `check_prose.py`, `check_links.py`, `check_docs.sh` green; every program output in the README and the pages is generated by `scripts/site.py` | `scripts/check_docs.sh` | |

Nothing in a gate is "expected to pass": the mutants are the rows of `scripts/mutants_st.py`.

## 10. Results

Everything below was printed by a script of this repository (`sh scripts/check.sh` runs all of it; `docs/bench/gate_st.log` and `docs/bench/mutants_st.log` are the logs of the gates and of
the mutants), on one machine: this Mac (Apple M4 Max, arm64, macOS 26.2), compiler `a4572ea`, scikit-learn 1.8.0, numpy 2.4.6, Python 3.14.5. The machine was **not quiet** (1-minute load average 3 to 14 during
the runs, other agents building); correctness gates do not depend on the load and **no timing was taken**: the cost of decoding a table, of building the reader vocabulary at each `predict` and of the
reassembly of the steps payload is not measured, and no speed is claimed. The damaged-file gate flips every byte of four files; `check.sh` flips every fifth byte outside the table (`ST_STRIDE=5`) and
every byte of the table.

| gate | result |
|---|---|
| S1, S6 oracle, byte for byte | `gate_st.py diff`: **545 runs, 0 differences**. 300 seeded random tables (140 the oracle accepts and the tool writes a file for, 160 the oracle refuses by its own verdict: a reader limit, a plan mistake, a declared column that no step reads, the tool refusing with the tag the oracle gives), the corner families of `categorical-encoders.md` and the reader's adversarial files through each encoder. The whole file (plan, schema, steps with the stubs, model section, meta, table, digest) equals the oracle's, the plan, the model section and the compiler pin being the tool's own; the summary's `steps_sha256` equals the SHA-256 of the oracle's payload, and `string_table.sha256` that of the table |
| S2 round trip | `gate_st.py roundtrip`: **120 runs, 0 differences**. 60 fits with `--split` (4 models x 5 plans: one-hot with a rare bucket, mixed one-hot, frequency and hash, hash, label, and one whose steps read the columns in an order that is not the header's, x 3 tables of 40 to 200 rows), written and read back: `eval --split` on the file prints the held-out metrics of the fit string for string, and so does `eval` of the held-out rows as a file |
| S4 row order and threads | `gate_st.py order`: **496 runs, 0 differences**. 40 tables x 5 permutations of the rows (same schema, steps with min-max scaling, and table bytes) and 5 thread counts (same whole file). Printed, not asserted: of 8 permuted triples per model the whole file was identical in 0 for the tree, the forest, linear and logistic, because the estimators draw by row index or sum in row order |
| S5 unseen, missing, markers, hashing | `gate_st.py predict`: **440 runs, 0 differences**. 400 fits from generated tables and a predict file each (another column order, unseen strings, markers, empty cells, a column absent): `unknown_categories` (1,559 unseen strings in all) and each of the 95 refusals the oracle gives (tag, step, column, row) agree; 40 linear models with a hash step and 9 unseen strings each predict intercept plus the coefficient of the bucket `hashlib` names, to 1e-9 relative |
| S7 bounds | `gate_st.py limits`: 7 runs, 0 differences: 128 columns of 1,024 strings (131,072) in three modes are written, read back and applied; 129 columns (132,096, or 131,838 under `--split train=0.999`) are `ml.limit.string-table` (`what` strings, `size`, `max` 131,072) with no file. `tests/strtab.cho` holds each bound of section 4 at its limit and one past it. The pool and size bounds are not reachable from the command line (the reader's own pool ceiling, 8 MiB, refuses first, and a table of 12 MiB needs more strings than the strings bound allows): they guard the decoder against a file no `fit` wrote |
| S8 alignment by name | `gate_st.py align`: **156 runs, 0 differences**. The four models with categorical columns, six permutations of the predict file's columns each (and the target in any place): identical answer bytes; a missing numeric column, a missing and a dropped categorical column, an extra numeric and an extra text column (with the retry repair, which gives the answer of the fitted columns), a numeric column holding text. The four models without a categorical column in four orders |
| S9 privacy | `gate_st.py privacy`: 8 fits, 0 differences: tokens put in 3 categorical columns are in the file exactly when the step stores strings |
| S10 damaged files | `gate_st.py fuzz`: **61,155 runs, 0 differences**, no trap, one JSON line each. Four files (one per model; hash and label steps among them): 19,497 byte flips (three ways, raw, with the file digest repaired, with both digests repaired), every truncation, every `u32` and `u64` field of the table at 6 and 8 extreme values, non-UTF-8 bytes in names and strings, 3,000 seeded damages. The oracle's file-level and table verdicts equal the tool's tag and `reason` wherever the oracle gives one. **The first full run found a trap that is not the table's**: a logistic or tree file whose class label is outside the 12 digits of an integer column (a label of -2^63) trapped when `predict` printed it; the labels record is now checked at decode (`ml.model.malformed`, `lib/models/record.cho`), with a test in `test_stringtable.py` |
| S11 scikit-learn | `gate_st_sklearn.py`: all PASS. 3 generated datasets (1,500 rows, 3 categorical and 2 numeric columns), the library's seeded split written as two files, `ml fit` on the train file, `ml predict` and `ml eval` on the test file from the file. Logistic: predicted labels identical on every test row (0 differ of 375), probability at most 3.4e-8 from scikit-learn's (bound 1e-6), accuracy equal, ROC-AUC at most 1.1e-16 apart (bound 1e-6), log-loss relative difference at most 8.3e-9 (bound 1e-6); the third dataset has 4 test rows with a category the train file never held. Forest: accuracy 0.008, 0.021 and 0.003 from scikit-learn's, ROC-AUC 0.005, 0.009 and 0.018 (bound 0.05 for both); `eval --split` on the written file equals the held-out metrics of the fit for both models |
| S12 tables | the two new rules and the removed one are in `spec.cho`, `introspect`, the schemas and the fixtures once (`test_tables.py`, `test_rules.py`); `check_registry.py`: 68 refusal codes in 4 bands, section tag 6 |
| unit programs | `tests/strtab.cho`: 15 tests (the table written by hand and compared byte for byte, every field out of its range, every bound at its limit and one past it, every flipped byte and cut with the digests repaired and not, 4,000 seeded damages, 2^63 - 1 in every field, the cross checks, the hashed ids); `tests/enc.cho`: 13 tests, one of them a file with a table decoded and its steps joined back |
| golden | `golden_numeric.json`, `golden_constant.json` and `golden_planned.json` are not touched and every fixture still gives their record; a model without a categorical step has five sections |

**Mutants** (`scripts/mutants_st.py`, `docs/bench/mutants_st.log`): **62 patterns, 60 killed, 2 survive and are explained, 0 unexplained**; the unmutated control passes every gate. They are on the layout (the layout
number, the entry count, the pool and string totals, the markers not stored or not passed, two limits swapped, a limit stored as a constant, the digest span, the step number of an entry, the entry length, the stub layout, the schema type
code, the meta columns, a table written for every file), the bounds (each of the seven bounds of section 4 moved by one, a size compared with `>=`, the digest not compared, the UTF-8 checks, the duplicate names, the stored limits against the
content, the totals, the step order, the three checks of the join), the file (the routing of a six-section file, a type 3 without a table, a table without a type 3, the names of the table against the schema, the plan against the stored steps in
five ways, the label range), the reader (the hash mode off, the bucket from the halves of the digest, an id equal to the number of buckets, the hash-bucket check of the step, the markers ignored, the string limit ignored, a column looked up by
its place, the numeric features taken by place, a missing vocabulary column as the reader's refusal, a hashed column of 2 buckets, the decoder accepting a name twice), privacy (a hash step that stores the strings) and the registry (a code used twice).
**`s4-first-appearance` is the mutant the issue requires the row-order gate to kill**: a vocabulary in the order the rows meet the strings. The gate `order` kills it alone, and it does so by a refusal: `pack` decodes the payload it is given, and
the decoder requires ascending strings, so the fit is refused with `ml.model.string-table` before a file can differ.
The two survivors are equivalent: `b-limit-columns-in-the-loop` (the comparison of the columns of an entry with the stored limit inside the loop gives the same reason as the final comparison; its job is to stop the quadratic comparison of names before
the count grows past 4,096) and `f-table-without-type3` (a table with no categorical column in the schema is refused with the same reason by the comparison of names, which comes next).
What the mutants showed about the gates: the first run had 7 results that were not kills. One mutant did not build (the edit left a function with an unused heap); four had no test (the plan listing a column the step did not read, a name twice in one payload,
an id equal to the number of buckets, the code constant of the new refusal, which no registry test read), and one mutant (`p-hash-stores-strings`) changed code that cannot run without a second edit; each got its test, its second edit or its reason.

**The fixtures this stage changes**, all in `tests/conformance/fixtures.py` and `test_enc.py`; the three golden records are not rewritten. Replaced: `ml-model-vocabulary-not-stored` by `fit-linear-categorical-writes-the-file`,
`-tree` by `fit-tree-categorical-writes-the-file`, `-forest` by `fit-forest-categorical-writes-the-file`. Changed expectations (`written: true`, a `path`, a `string_table`): `fit-forest-categorical-split-scores-held-out-rows`,
`fit-tree-categorical-split-takes-a-missing-cell`; `fit-tree-categorical-dry-run` stays `written: false`, exit 9, and `planned_actions` now names the file. New: `predict-from-a-categorical-model-file`, `eval-from-a-categorical-model-file`,
`predict-from-a-categorical-tree-file`, `ml-schema-mismatch-categorical-column-missing`, `ml-model-string-table-digest` (the rule `ml.model.string-table`) and `ml-limit-string-table` (the rule `ml.limit.string-table`, 129 columns of 1,024 strings).
`test_enc.py`'s class `Files` (E9) tests the new behaviour in place of the refusal. `tests/conformance/test_stringtable.py` (18 tests) holds the layout read by the oracle's reader, every reason of a refused table, the stubs, the plan against the stored
steps, alignment by name, markers, the string limit, `unknown=refuse` at predict, the hashed bucket, the model with no numeric feature, and the label range.

**Choices made** (each overturnable): the table is a sixth section (tag 6) and a categorical step is an 8-byte stub in `steps` (section 1); the `FitVocab` payload of `categorical-encoders.md` 5.2 is stored verbatim, one entry per step; the
digest is the table's own; the markers and the four reader limits are stored; the bounds are 2^17 strings, 8 MiB of string bytes and 12 MiB; `ml.model.vocabulary-not-stored` is removed (nothing reaches it); `predict` and `eval` take no flag
of the categorical reader; the hashed reader mode is for stored models only; a vocabulary column the file lacks is the schema check's refusal (`CatCfg.lenient`).

**What is not done.** A hashed column is interned while the fit reads the file (open question 1). The table stores the merged strings of a rare bucket (open question 2). There is no verb that lists the categories of a model file. Nothing was timed. Nothing was
run on a second machine. Titanic and House Prices were not downloaded and not run.

## 11. Open questions

1. A hashed column is interned at fit (the fit's bounds apply). Hashing at fit without interning needs the plan before the header is read; it would let a hashed column exceed `--max-categories`. [open]
2. The table stores every string of an `encode min-count=N` column, including the ones that are merged into `__other__`, because the counts are the state. A table that stores only the kept strings and the
   merged total would be smaller and would leak less; it changes the semantics of `unknown` (a merged string would be unseen, not the bucket) and is left. [open]
3. A second reader of the same string table (an `inspect` verb that lists the categories without applying the model). [open]

# cancho-ml run tracking and train/serve skew checks (issue 43, milestone M22)

**Status: this document is written before the code it judges.** Sections 1 to 9 fix the file layout, the bounds, the answers, the run record, the id, the verbs, the refusals, the authority and the gates; section 10
holds the results and is filled only with what a script of this repository printed, with its conditions. Tags as in `design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from a
document or the code, **[measured]** printed by a script here, **[open]** not decided. The text of the issue (alpibrusl/cancho-ml#43) is the scope.

The two problems the issue names are one: a model file does not remember what it was fitted on. `eval` scores the file a model was fitted on without saying so, a feature far outside the range seen in
training is applied silently, and nobody can list, compare or replay a fit. Because `fit` is deterministic (same inputs, same bytes, at any thread count) the fix is to store the identity of the training
data in the model file (section 2) and to write a run record that is a pure function of the inputs (section 5): a run is reproducible by construction, and `ml runs verify` checks it.

## 1. Decisions in one table

| # | question | decision | why |
|---|---|---|---|
| D1 | where the training identity lives | an optional **provenance section**, tag 7, in the model file (section 2) | the issue asks for it in "the model's meta"; `meta` is checked by exact length in every reader, so a new tail would change the rule for kinds 1 to 3 and break `ml.model.malformed` for every older reader; a section is skipped by the structure walk exactly as the string table (tag 6) is |
| D2 | written always, or on request | **always written by `fit`** of this version, for every model kind; readers keep reading a file without it | the skew checks are worth nothing if they need a flag nobody passes; the cost is one deliberate re-record of the model-file goldens (section 8) |
| D3 | a file with no section | `provenance: absent` in `describe`, `"on":"unknown"` in `eval`, `"out_of_range":null` in `predict` and `eval`; `--max-out-of-range` is then refused (`ml.predict.no-provenance`) | a guard that cannot be evaluated must not pass silently |
| D4 | the constant models (kinds 1 to 3) | they carry the section with **zero feature records**: the file identity (SHA-256, size, rows) only; no range is stored and `predict` of a constant model reports no `out_of_range` | a constant ignores its features: a range of a feature it never uses is not a skew signal; `eval` still says `"on"` |
| D5 | the statistics of a feature | count of non-missing training cells, min, max, mean, population standard deviation, of the cells **as the reader read them, before any step of the plan** (section 2.3) | the issue; the steps are in the file already, a range after the steps would describe the encoder, not the data |
| D6 | the run id | SHA-256 over a canonical description of the inputs that determine the model bytes (section 5.2); not the thread count | the thread count cannot change a byte, so refitting at another count must give the same id |
| D7 | where runs live | a directory given by `--track DIR`, which must exist and lie under `--root` | the pinned compiler has no directory-creation primitive (`dir_*` has open, enter, list, stat, rename, remove, no make): the directory of `--output` must exist too; the issue's "created if absent" is **not done**, and the refusal is `io.not-found` with the path |
| D8 | the eval record | a separate file `DIR/<run id>.eval-<scored sha256>.json` | one record is immutable once written, so an eval is never an edit of it; the same eval twice is the same file |
| D9 | authority | none added (section 7) | `ml` already reads and writes files beneath `--root`; `ml-predict` and `ml-serve` only read the new section |

## 2. The provenance section

### 2.1 Placement and the version policy **[choice]**

The model file keeps `version = 1` and the layout of `design.md` 4.1. Section tags: 1 plan, 2 schema, 3 steps, 4 model, 5 meta, then, **in this order and only when present**, 6 strings (the string table,
`stringtable.md`) and **7 provenance**. The section count in the header is 5, 6 or 7 and says nothing by itself: the reader walks the tags. Rules of the walk, all checked before any payload is read:

* the first five tags are 1 to 5 in order (as before);
* each further section has a tag greater than the one before it, and the tag is 6 or 7; so a file has `{}`, `{6}`, `{7}` or `{6, 7}` after the core sections, and a count of 6 can mean either of the two;
* the section lengths cover the body exactly (`pos == body`), as before.

A file without tag 7 is byte for byte what the previous version wrote (a test): **a model fitted by an older `ml` is read by this one, and says `provenance: absent`**. A file with tag 7 is read by an older reader
only if the older reader tolerated a seventh section; it does not: it refuses the count (`ml.model.malformed`, section 0). That is the one-way door of D2 and the reason the golden re-record exists.
The `meta` section, the sum order and the kinds are untouched. `kinds.section_provenance()` is 7, `kinds.section_count()` is 7 and `section_at(7)` is 7 (`scripts/check_registry.py`,
`tests/t_registry.cho`).

### 2.2 Byte layout (all integers little endian; a double is its bits)

```
u32  layout           1
u64  file_bytes       size of the training file in bytes, 0 .. 2^62
32 B file_sha256      SHA-256 of the training file's bytes
u64  file_rows        data rows in the training file, 1 .. 10^9
u64  rows             rows the fit trained on (= `meta` rows; less than file_rows under --split), 1 .. file_rows
u32  features         number of feature records, 0 .. 8192
features * {
  u64 count           non-missing training cells of the column, 0 .. rows
  f64 min  f64 max    exact, as the reader read the cells
  f64 mean            the exact mean, rounded once (2.3)
  f64 std             population standard deviation (2.3)
}
```

The payload is exactly `64 + 40 * features` bytes. The feature records are the **numeric feature columns of the schema in schema order** (not the target, not a categorical column): the record `i` is the `i`-th
schema column whose role is feature and whose type is 0, 1 or 2. A categorical column has no range (its unseen strings are counted by `unknown_categories` already). A model with no numeric feature
(only categorical columns, or a constant model) has `features = 0`.

### 2.3 The statistics: the rule, with the rounding stated **[choice]**

Over the training rows (all rows, or the train side of `--split`) and the non-missing cells of one column, `x_1 .. x_n` doubles as the reader holds them in the matrix (a decimal column holds the nearest double of its
text; the reader is cancho-table's exact one):

* `count` = n. When n = 0 the four doubles are +0.0 (bits 0).
* `min`, `max`: the smallest and the largest; a zero is stored as +0.0 whatever its sign (there is no -0 in the reader).
* `mean` = the exact sum divided by n, **rounded once** to the nearest double, ties to even: the sum is accumulated exactly (`facc`, cancho-table's superaccumulator, which no order of the cells and no thread
  count can change) and `facc.mean_key` rounds once.
* `std` = `sqrt(v)` with `v = S / n` in IEEE double division, where `S = sum (x_i - mean)^2` is accumulated **exactly** (`ml.fsq`, the exact sum of squared differences of doubles) with the *stored* `mean` as the
  centre, and rounded once. So `std` is the population standard deviation about the stored mean: it differs from the exact population deviation by at most `(mu - mean)^2`, a relative 2^-106 of the variance
  for a column whose mean is not near zero. This is a definition, not an approximation claim: the oracle (`scripts/oracle_track.py`) computes `S` with Python fractions, rounds it once with `float(Fraction)`,
  divides with the same IEEE division and takes `math.sqrt`, and compares the bits. A `S` beyond the largest double is `ml.fit.non-finite` (a column of values near 1e154).
* a column of one distinct value has `min = max = mean` and `std = 0.0` exactly (a check of the decoder, 2.4).

Why exact and not the lane sums of the package: these numbers are stored in the file, so they must not depend on the order of the cells, on the thread count or on the summation order of the model kind.

### 2.4 Bounds and the decoder, in the order they are checked

The reader (`pipeline.provsec.validate`, called by `pipeline.file.decode` and by `mfile.decode`) checks, **before it reads a record and before it allocates anything**:

1. the length is `64 + 40 * features` for the `features` the payload says, and `features <= 8192`; a length that disagrees (including a huge count) is refused with nothing read past the payload;
2. `layout == 1`; `file_bytes` in range; `1 <= rows <= file_rows <= 10^9`;
3. `rows` equals the `rows` of `meta` (the file disagrees with itself otherwise);
4. `features` equals the number of numeric feature columns of the schema (zero for kinds 1 to 3);
5. per record: `count <= rows`; with `count = 0` all four doubles are bits 0; with `count > 0` the four doubles are finite, `min <= mean <= max`, `std` is finite with the sign bit clear, and `min == max`
   implies `mean == min` and `std` is +0.0.

A failure is `ml.model.malformed` with detail `section: 7` (the section tag), as every other section. The checksum is checked first (before any of this), as for every section. The decoder allocates nothing sized by
the file: the validation walks the slice. `describe` and the server read the stored records with accessors over the same bytes.

## 3. The answers

### 3.1 `eval`: which data was scored

`eval` hashes the CSV it scores (streamed; the file is not held for it) and compares the SHA-256 with `file_sha256` of the model. The answer gains, after `metrics`:

* `"on":"train"` when the two are equal (the file is the file the model was fitted on; with `--split` some of its rows were not training rows, and `eval --split` says `train` all the same: the model does not store
  the split, the record does);
* `"on":"other"` when they differ;
* `"on":"unknown"` when the model has no provenance section.

`--format text` prints the same as a line `on train`. `fit --split` scores the held-out rows through the stored steps already (`holdout`); that object gains `"on":"held-out"`.

### 3.2 `predict`, `eval`, `ml-serve`: how far outside the training data

For a model with feature records, every **non-missing** cell of a numeric feature column that is strictly below its `min` or strictly above its `max` is *out of range*. A missing cell is never out of range; the target
column is not a feature and is never compared; a column with `count = 0` is never compared; a categorical column has no range. The comparison is on the doubles the reader produced, before the steps.

The answer gains, beside `unknown_categories`:

```
"out_of_range": {"cells": C, "rows": R, "columns": [{"name": "age", "count": 3}, ...]}   or   null
```

`C` is the number of out-of-range cells, `R` the number of rows with at least one, `columns` lists **only the columns with a nonzero count**, in schema order (an array of objects, not a JSON object keyed by
column name: a column name is arbitrary bytes and a key would need a second escaping rule). `null` when the model has no provenance, or is a constant model. Counted over **every row of the scored input**, not only the
page `--limit`/`--from` selects (as `unknown_categories`); for `eval --split` over the rows scored.

`predict` also gains `"ignored_columns":[...]`: the columns of the file the model did not use: the target column when the file carries it (the file may, and the model ignores it), in file order; empty otherwise.
(An extra column that is not the target is `ml.schema.mismatch` as before; a column named by `--drop` is the caller's own and is not listed.)

### 3.3 `--max-out-of-range N` (predict, eval; also a start flag of `ml-serve`)

If more than `N` rows are out of range, the call is refused: rule `ml.predict.out-of-range`, exit 8, detail `rows` (how many), `max` (N), `column` (the first column, in schema order, that has an out-of-range
cell in the first offending row), `row` (that row, counting data rows from 1), `value`-free: no cell is quoted. `N` is 0 to 10^9; absent means no bound (the counts are still reported). `ml-serve` applies the bound to
each request: a request over it gets the HTTP 422 problem with the rule. A model with no provenance and `--max-out-of-range`: `ml.predict.no-provenance` (exit 8).

### 3.4 `describe`

`describe` gains a `provenance` member: `null` when the model file has no section, else
`{"file_sha256":..., "file_bytes":..., "file_rows":..., "rows":..., "features":[{"name":..., "count":..., "min":"...", "max":"...", "mean":"...", "std":"..."}]}`
(doubles as strings, the shortest decimal that reads back, as every double in the answers). The OpenAPI documents of `ml-serve` (`docs/serving/*.openapi.json`) describe the new member of the answer and
are regenerated and re-validated (O1).

## 4. Counting is a pure function

`out_of_range` and the statistics are functions of the matrix the reader produced and of the stored records; neither depends on the thread count, on the order of the rows, or on the model kind. The gates compare
them with an independent computation (section 9).

## 5. Tracking

### 5.1 `ml fit --track DIR`

`DIR` is a directory beneath `--root` that exists (D7). `fit` writes `DIR/<run id>.json` and the model. The record is canonical JSON: one line, keys in the fixed order of 5.3, no spaces, a final newline; **no
timestamp, no host name, no absolute path, no payload** (no row, no cell, no category string). It names the training file as given (a path relative to `--root`, so that `verify` can find it) and the model file as
given. Column names appear (in the plan, the target and the lists `--drop`, `--float`, `--cat`): they are the schema, which the model file holds too; the markers of `--na-values` appear because they are stored in
the model's string table.

Bounds **[choice]**: at most **1000 records** in `DIR` (`ml.runs.too-many`, counted over every `*.json` of the directory, before anything is written); a record of at most **262144 bytes**
(`ml.runs.record-too-large`; the plan alone may be 65536). Both are refused before the model is written.

Idempotence: if `DIR/<id>.json` exists, it is read and compared with the record about to be written **with the `threads` member removed**; equal: nothing is written (`written:false`); different: the call is
refused with `ml.runs.conflict` and nothing is written. The second case is not a normal event: it means the same inputs gave another model (a determinism fault) or the directory was edited. Order of effects: every check
(bounds, conflict, the output path, `--overwrite`) is made before either file is written; the model is written, then the record. `--dry-run` writes neither and lists both in `planned_actions`.

### 5.2 The run id **[choice]**

`id = SHA-256(canonical text)` in lowercase hex, where the canonical text is the bytes of the `inputs` object of 5.3 exactly as the record holds them (the same writer produces both). `inputs` holds what
determines the model bytes and nothing else:

* the training file's SHA-256, size and row count (not its name: the same bytes under another name are the same run);
* the target column name (null for a clustering), the canonical plan text (which holds the model and every hyperparameter), the task;
* the seed and the split as `{num, den}` reduced to lowest terms (`train=0.8` and `train=0.80` are one split) or null;
* the lists `--drop`, `--float` and `--cat` as given, `--na-values` and the four reader limits of `--cat` (they are stored in the string table, so they change bytes);
* the compiler pin (`meta` stores it, so another compiler is another model file) and the `sum_order` of the model kind.

Not in `inputs`: `threads` (recorded beside it, `threads`, as the count of the run that first wrote the record), the caps `--max-rows`, `--max-columns`, `--max-cells`, `--max-line-bytes` (they can refuse a fit, never change its
bytes), `--output`, `--overwrite`, `--log`, `--format`. `verify` reads with the ceilings of those caps.

### 5.3 The record (`ml.run.v1`)

```
{"schema":"ml.run.v1","id":"<64 hex>",
 "inputs":{"data":{"path":"iris_train.csv","bytes":N,"sha256":"..","rows":N},
           "target":"virginica"|null,"task":"classification"|"regression"|"clustering",
           "plan":"<canonical plan text>","seed":0,"split":{"num":4,"den":5}|null,
           "drop":"..","float":"..","cat":"..","na_values":"..","reader":{"max_categories":64,"max_string_bytes":255,"max_pool_bytes":1048576,"max_cat_columns":128}|null,
           "compiler":"<pin>","sum_order":2|3|null},
 "threads":1,
 "model":{"path":"m.ml","bytes":N,"sha256":".."},
 "counts":{"file_rows":N,"train_rows":N,"test_rows":N,"features":N,"steps":N},
 "metrics":{"on":"train"|"held-out","rows":N,"values":{...}}|null}
```

(`drop`, `float`, `cat`, `na_values` are the strings as given, empty when absent; `reader` is null when `--cat` was not given. `sum_order` is null for the constant kinds.)

`metrics` is the metric object `eval` prints for the same rows: the **held-out rows** when `--split` was given (`"on":"held-out"`), else the **training rows** (`"on":"train"`, scored through the stored steps by the same
code as `eval`). A constant model has `metrics` null (its summary, the mean or the classes, is in the answer of `fit`; scoring a constant on its training rows is `ml eval`).
The record is built from numbers `fit` already holds; it never contains a row, a cell or a category string.

`ml eval --track DIR` writes `DIR/<run id>.eval-<scored sha256>.json`, schema `ml.eval.v1`: `{"schema":"ml.eval.v1","run":"<id>","model_sha256":"..","data":{"path":..,"bytes":..,"sha256":..,"rows":..},"on":"train"|"other"|"unknown","split":{..}|null,"positive":..|null,"metrics":{...}}`.
The run is found by the model file's SHA-256 among the records of `DIR`; none is `ml.runs.unknown-model`. The same eval twice writes the same bytes (idempotent as above).

### 5.4 The verbs (`ml runs ...`)

All four go through the flag table, `introspect`, `skill`, the schemas and the module contract as the other verbs. Flags: `--root`, `--format`, `--limit` (list: records per page, default 1000), `--max-bytes`, and for
`verify` also `--threads`.

* `ml runs list DIR`: `{"dir":"DIR","count":N,"runs":[{"id":..,"model":..,"task":..,"data_sha256":..,"model_sha256":..,"on":..,"evals":N}]}`, sorted by id (bytewise); eval records are counted, not listed. A record that
  does not validate is `ml.runs.damaged` (the file name in detail), not skipped.
* `ml runs show DIR ID`: `{"record":{...}}` the record, validated (the id recomputed from `inputs` must equal the name and `id`).
* `ml runs compare DIR ID ID`: `{"a":..,"b":..,"inputs":[{"key":"seed","a":0,"b":1},...],"metrics":[{"name":"rmse","a":"..","b":"..","equal":false}],"model_equal":bool}`: the `inputs` members that differ (a
  dotted key for the nested ones) and the metrics side by side (a member missing on one side is null there).
* `ml runs verify DIR ID`: reads the record, checks the SHA-256 of the training file named in it (`ml.runs.data-changed` if it differs, detail: recorded and found), **refits in memory** (the fit code of `fit`, with
  the inputs of the record, at `--threads`, default 1; no file is written) and compares the SHA-256 of the resulting model file bytes with the record's: equal gives `{"verified":true,...}`; different is
  `ml.runs.model-differs` with the two hashes and the two compiler pins (the usual cause) in detail.

The columns that a fit read as `--drop`, `--float` or `--cat` are replayed from the lists of the record by position in the header of the training file, which the replay reads first. **Limit [choice]**: the replay of a
column that carries one of these three flags needs its header position below 62 (a bit mask in the reader's configuration, because the reader's configuration is a plain structure that cannot hold a runtime
string at the pinned compiler); a record that needs more is refused with `ml.runs.not-replayable` (detail: the column). Fits with such a flag on a column past the 62nd are not replayable by `verify`.

### 5.5 Rule tags

New tags, all registered in `design.md` 8.1, `tools/ml/spec.tmpl`, `lib/core/kinds.cho` (numbers) and the schemas: `ml.predict.out-of-range` (8), `ml.predict.no-provenance` (8), `ml.runs.conflict` (8),
`ml.runs.too-many` (8), `ml.runs.record-too-large` (8), `ml.runs.not-found` (3), `ml.runs.unknown-model` (8), `ml.runs.damaged` (8), `ml.runs.data-changed` (8), `ml.runs.model-differs` (8),
`ml.runs.not-replayable` (8), `ml.input.changed` (8: the training file changed between the reads of one fit). Existing tags reused: `io.not-found` for a missing `DIR`, `ml.fit.non-finite` for an
unrepresentable statistic, `ml.model.malformed` for a damaged section 7.

### 5.6 Privacy

A record holds hashes, counts, flags, the plan and the names the plan and the schema hold. It never holds a row, a cell, a category string (the vocabulary lives in the model file, not in the record) or a path outside
`--root`. The training file's **name as given** (relative to `--root`) is stored, so that `verify` can find the file; an absolute path is refused at the command line already (`path.absolute`). A test scans the
records of every fixture for every cell and every category of its training file.

## 6. Where the fit is computed

`fit` hashes the training file before it reads it and again after, and refuses (`ml.input.changed`) if the two differ: the SHA-256 in the section is then the hash of the bytes that were read, and the claim
"`verify` re-reads the file and gets the same hash" is a claim about the same bytes. The statistics are computed from the matrix the fit holds (no extra read of the file). Cost: two sequential reads of the training
file plus one pass over the matrix per feature (exact accumulation), measured in section 10.

## 7. Authority

`ml` reads and writes files beneath `--root` already (`manifests/ml.authority.json`: `dir_read`, `dir_write`, `file_read`, `file_write`): the run record is another file in that set, and the listing of `DIR`
is `dir_read`. **No capability is added to any binary.** `ml-predict` and `ml-serve` only **read** the new section and hash nothing they were not already allowed to read; they gain no write label. The pinned
manifests are re-derived by `scripts/manifest.py --check` and compared with `scripts/check_bins.py` (a build of the three binaries from the same sources), and the diff of `manifests/*.authority.json` is empty in the commit
that adds the code (section 10).

## 8. The goldens: one deliberate re-record **[choice]**

Every model file `fit` writes gains section 7, so its bytes and its SHA-256 change, and so does the `bytes` and `sha256` of the answer of `fit`; `describe` gains a member; `predict` and `eval` gain members.
The re-record is **one commit**, made after the code works and before the docs, and it lists in its message every file it changes and why. The list is in section 10 (filled from the diff).
Nothing else in a golden may change: the commit is checked by comparing the re-recorded files with the previous ones after removing the new members, and a model file with section 7 removed (and the
header count and the checksum repaired) with the previous bytes.

## 9. The gates (written before the code; each can fail)

| id | gate | how it can fail |
|---|---|---|
| T1 | **section oracle**: `scripts/oracle_track.py` computes section 7 for fitted models from the CSV alone (exact fractions) and compares **bytes** with the section in the file `ml fit` wrote, for every model kind, categorical and `--split` fits included | any rounding rule, order of records, layout or count differing by a bit |
| T2 | **record oracle**: the same script builds the run record and the id from the command line's inputs and compares bytes with `DIR/<id>.json` | a key, an order, a canonical form or the id definition differing |
| T3 | **replay equality**: `runs verify` reproduces the model SHA-256 for every kind (constant regression and classification, linear, logistic, tree, forest, kmeans), with categorical plans and `--split`, at `--threads` 1 and 8 | any input the record omits that changes the bytes |
| T4 | **thread invariance**: the whole model file, section 7 included, and the record (modulo `threads`) are the same bytes at `--threads` 1, 2, 8 | a statistic that depends on a thread or an order |
| T5 | **`on`**: `"on":"train"` on the training file and on a copy of it under another name; `"other"` on a file that differs by one byte; `"unknown"` on a pre-provenance file | an equality that looks at the name, or always says train |
| T6 | **out-of-range**: counts equal an independent computation, on generated data with ties at the boundary (a cell equal to `min` or `max` is not out), missing cells, a target column in the file and `--split` | off by one on a bound, a missing cell counted, the target counted |
| T7 | **bound**: `--max-out-of-range N` refuses at N+1 rows and accepts at N, naming the first column and row | wrong comparison, wrong row |
| T8 | **serve equality**: `ml-serve` answers carry the bytes of `ml predict` for `out_of_range` (as the other answer members), and the OpenAPI documents validate | any divergence |
| T9 | **fuzz**: every byte of a section 7 flipped (with and without a repaired checksum), every truncation, counts of `2^32-1` and `2^63-1`: always a refusal with a tag, never a trap; damaged run records (every byte flipped, truncations, duplicated keys, wrong id): `ml.runs.damaged`, never a trap | a bound checked after use |
| T10 | **idempotence and conflict**: refit with the same inputs writes nothing and gives the same bytes; a record edited in place is `ml.runs.conflict`; the 1001st record and an oversized record are refused before anything is written | an overwrite, a partial write |
| T11 | **privacy**: no record contains a cell or a category string of its training file | a payload leak |
| T12 | **mutants** (`scripts/mutants_track.py`): the id includes the thread count; the write overwrites; `on` is always `train`; a range comparison is off by one at `min` and at `max`; a missing cell is counted; the target is counted; the sum is a naive float sum; the section decoder skips a bound (each one) | a mutant that survives is explained or the gate is strengthened |
| T13 | **registers**: `check_registry.py`, `test_tables.py`, the schemas, `check_bins.py` and `manifest.py --check` see the new tag, flags, verbs and rules | a place that does not know them |

The existing suites (`scripts/check.sh`) run unchanged apart from the goldens of section 8.

## 10. Results

(filled when the gates have run)

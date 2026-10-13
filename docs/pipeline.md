# cancho-ml pipeline (`lib/pipeline`, stage M4 of v1)

**Status: sections 1 to 5 (the gates, the grammar, the semantics, the file) were written before any code of `lib/pipeline`. Section 6
holds the results; a number or a verdict is in it only if a script in `scripts/` printed it, with its conditions.** Tags as in
`design.md`: **[choice]** is a decision the maintainer can overturn, **[open]** is not decided.

An internal library, **not wired into the CLI**: `lib/pipeline/*.cho` on top of `lib/transforms` (`docs/transforms.md`), a driver
`tools/pipe-check`, tests in `tests/t_pipeline*.cho`. The wiring stage (the 4-lane order change has landed, `docs/lanes.md`) adds the plan flags
to `fit`, the stored steps to `predict` and `eval`, and the estimators of kinds 4 to 7 behind the hook of section 5.4.

## 1. The gates (fixed before the code; each can fail, section 6 shows it failing)

| id | gate | judged by | a mutant it must kill |
|---|---|---|---|
| P1 | **Plan text.** Equivalent spellings (key order, blanks, comments, CRLF, `+`-less numbers with leading zeros, `1e-1` and `0.1`) give the same canonical bytes; canonicalising a canonical plan changes nothing; every malformed plan is refused with a tag and the **line and column** in `detail`; one fixture per tag; floats are read with the reader of cancho-table (`ml.csvcell`, pinned) and written back in its shortest form | `tests/t_pipeline.cho`, `oracle_pipe.py --plans` | canonical key order, a default not materialised, a number not normalised, line/column off by one |
| P2 | **Fit on the training rows only.** A pipeline fitted on the training matrix has the same parameter bytes whatever the test rows are (fitted on `train`, the step records are equal to those of a fit on `train` run beside a different `test`, and differ from a fit on `train + test`); the pipeline never receives the test rows at fit | `tests/t_pipeline.cho`, `oracle_pipe.py --pipeline` | fit on all rows |
| P3 | **Each step feeds the next.** A two-step plan (impute then standardise; encode then scale) gives the output of applying step 2 to the output of step 1 fitted on the training rows: the parameters of step 2 are those of the *imputed* training column | `oracle_pipe.py --pipeline` | step 2 fitted on the raw matrix |
| P4 | **One payload.** The file holds the plan, the schema, the steps (every fitted parameter) and the estimator payload in one CML1 file; encode then decode returns the same bytes; a file with no steps is **byte for byte** the file `tools/ml/mfile.cho` writes for the same constant model; sections 1 to 3 and 5 are read by an independent reader (`oracle_pipe.py --file`, Python) that recomputes the transform of a test matrix from the stored parameters alone and agrees bit for bit | `tests/t_pipeline.cho`, `oracle_pipe.py --file` | section order, a length, a missing field |
| P5 | **No plan is applied without its stored parameters.** (a) Every `pub fn` of `lib/pipeline` and `lib/transforms` that transforms a matrix takes the stored step bytes, and none takes a `Plan`: `scripts/check_pipeline_api.py` scans the sources; (b) the only constructor of the object that `apply` takes is `file.decode`, which checks the checksum and every field; (c) editing the text of the plan section in a file whose checksum is repaired changes no output (the plan text is a description) but a mismatch between the plan and the steps is `ml.model.malformed`; (d) editing a stored parameter and repairing the checksum **changes the output**, and editing it without repairing the trailer is `ml.model.checksum` | `tests/t_pipeline.cho`, `check_pipeline_api.py` | apply reading the plan; the checksum not checked; a parameter read from a copy |
| P6 | **Schema by name.** A prediction input with the fitted columns in another order gives the same output as in the fitted order; a missing column, an extra column and a column of another type are `ml.schema.mismatch` with a repair text naming the missing, extra and mistyped columns; the target (if the schema has one) is not asked for | `tests/t_pipeline.cho`, `oracle_pipe.py --pipeline` | positions instead of names, the repair on the wrong side |
| P7 | **No trap.** Fuzz: random and hostile plans (bytes, huge numbers, deep keys, 10^5 lines), random column lists and names, matrices with empty, constant and all-missing columns, and corrupted model files (every truncation, every flipped byte with and without a repaired trailer, bounded random damage of the steps and plan sections) never trap; every answer is a code of `transforms.codes` | `tests/t_pipeline_fuzz.cho` | a removed bound |
| P8 | **Limits before allocation.** Plan bytes, plan lines, listed columns, steps, categories, bins and output width are compared with their caps before anything sized by them is allocated | `tests/t_pipeline_fuzz.cho` | a removed cap |
| P9 | **Registries.** The step kinds, section tags and the 301 to 325 band of refusal codes are in `lib/core/kinds.cho` and `lib/transforms/codes.cho`, in `design.md` 4.1, 8 and 8.1 and in `tools/ml/spec.cho` `pending()`, and `tests/t_registry.cho` and `scripts/check_registry.py` fail on any disagreement | `t_registry.cho`, `check_registry.py` | a number claimed twice |

## 2. The plan text (design.md 4.2, made precise)

One step per line, `verb key=value key=value ...`, a final `model` line. **Input** may have blanks (spaces, tabs) between tokens and at
the ends of a line, CRLF or LF, blank lines, and lines whose first non-blank character is `#`; the keys of a line may come in any order;
omitted keys with a default take it. **Canonical form** (what is stored, `plan` section): no blanks but one space between tokens, `LF` ends every line, no comment and no
blank line, the keys in the order of the table, every default written, numbers in one spelling, the `column` list sorted bytewise
ascending.

| verb | keys in canonical order | values |
|---|---|---|
| `impute` | `column` `strategy` `value` | column list or `*`; `mean` `median` `constant`; `value` (a float) **only** with `constant`, required there |
| `scale` | `column` `kind` `ddof` | list or `*`; `minmax` `standard`; `ddof` `0` `1`, **only** with `standard` (written `ddof=0` by default) |
| `encode` | `column` `kind` `max-categories` `unknown` | list or `*`; `onehot` `label`; `1 .. 1024` (default 64); `ignore` `refuse` (default `ignore`) |
| `bucket` | `column` `kind` `bins` | list or `*`; `width` `quantile`; `2 .. 256` (required) |
| `model` | `kind`, then the keys of the kind (table in `lib/pipeline/plan.cho`, **provisional: the wiring stage owns it**) | `constant` (`task`), `linear` (`l2`), `logistic` (`iterations` `l2`), `kmeans` (`k` `iterations`), `tree` / `forest` (`task`, `max-depth` ...) |

* **Numbers.** An int is `[0-9]+` (at most 9 digits, leading zeros allowed and dropped). A float is read by `ml.csvcell.read` /
  `double` (cancho-table's `flt.parse_float` underneath: correctly rounded, ties to even; an exponent, a leading or trailing point and more
  than 12 digits are fine; `nan`, `inf` and an overflowing cell are refused) and written by cancho-table's `flt.put_float`
  (the shortest decimal that reads back, always with a point: `0.1`, `5.0`, `1e21`). So `rate=0.10`, `rate=1e-1` and `rate=.1` are `0.1`.
  There is no second float parser here.
* **Columns** are header names. A name must be non-empty and hold no blank, `,`, `=` or `#`; a header that has one is addressed with
  `*` or not at all (a limit, **[open]**: quoting). `*` means every column of the matrix the step receives.
* **Limits** (before anything is allocated): 65,536 bytes, 64 lines, 4,096 names in one list, 255 bytes in one name.
* **The `model` line is last and exactly once**; a plan without it, with it twice or before a step is `ml.plan.model-line`.

### 2.1 Refusals of the parser (`detail` = line and column, both 1-based; the column is the first byte of the token at fault)

| tag | when |
|---|---|
| `ml.plan.syntax` | a token that is not `verb` or `key=value`; an empty key or value; a byte outside printable ASCII in a token; a `=` inside a value |
| `ml.plan.unknown-step` | the verb is not in the table |
| `ml.plan.unknown-key` | the key is not a key of the verb (or of the model kind) |
| `ml.plan.missing-key` | a required key is absent (`column`, `strategy`, `kind`, `value` with `constant`, `bins`) |
| `ml.plan.duplicate-key` | the same key twice on a line |
| `ml.plan.bad-value` | an enum value not in its list, an integer or float that does not parse, `value` with a strategy that has none, a duplicate or ill-formed column name |
| `ml.param.out-of-range` | an integer parameter outside its range (`bins`, `max-categories`, `ddof`) |
| `ml.plan.model-line` | the `model` line is missing, not last, or repeated |
| `ml.limit.plan` | more bytes, lines or listed names than the limits above |

At fit, resolving a list against the current column names: `ml.plan.unknown-column` (the step in `detail`), `ml.input.too-many-columns`
when an encoding widens the matrix past the cap.

## 3. Semantics

* **Fit** (`pipeline.run.fit`): for each step line in order, resolve its columns against the names of the matrix it receives, fit on that
  matrix (the training rows), write the step record, apply it to the same matrix (which cannot meet an unknown category: every category
  was learned from it), and hand the result to the next step. The result is the transformed training matrix, its column names (a
  one-hot column `c` of `k` categories becomes `c#0 .. c#k-1`; other names are kept) and the `steps` payload. The plan is
  parsed once, before any data is looked at, so a bad plan costs nothing.
* **Apply** (`pipeline.run.apply`): takes the decoded file (never the plan), the input matrix and its column names and types, aligns the
  columns by name (section 3.1), then applies each stored step in order. It returns the matrix and a report (the count of unknown
  categories per run). The estimator is applied by the caller to that matrix (section 5.4).
* **Threads.** Nothing here spawns a thread. The only decisive sums are in `transforms.dsum`.

### 3.1 The schema

The fitted schema is the `schema` section (as `tools/ml/mfile.cho` writes it: `u32 columns`, `u32 target`, per column name, type, scale;
type 0 int, 1 dec, 2 float). The *features* are the columns but the target. An input with `names` and `types` is **aligned** to the features
by name: the fitted feature `j` is found in the input by exact byte equality of the name, and its type and scale must equal the fitted ones.
Order may differ; extra input columns, missing ones and mistyped ones are all reported together: `ml.schema.mismatch` with the
repair text `missing: a,b; extra: c; mistyped: d` (names, comma separated, sorted bytewise, truncated after 8 KiB). A duplicate input name
is also a mismatch. The target column, if present in the input, is *extra* unless the caller removed it (the wiring decides which; the
library takes the list of names it is given).

## 4. Determinism

The same plan, training matrix and caps give the same bytes (the steps payload and the file). Nothing depends on a thread, a clock or a
path. The file contains no timestamp. D2 (the same bytes on another machine) is untested here, as in each stage.

## 5. The model file

`CML1`, version 1, five sections in order, SHA-256 trailer (`design.md` 4.1). What this stage defines:

### 5.1 `steps` (section 3)
`u32 n` then `n` times `u64 length | payload`, a payload being a step (`docs/transforms.md` section 4). `n = 0` is the four bytes
`00 00 00 00` that `mfile.cho` writes: **a file without steps is the file it was.** The reader of `mfile.cho` still reads only that
form of `steps`; reading the steps there is the wiring stage's job and needs no other change to that file.

### 5.2 `plan` (section 1)
The canonical text of section 2, with its final LF. For the constant model the plan is `model kind=constant task=regression\n`, exactly what
`mfile.cho` writes, so the canonical parser reproduces it (a test).

### 5.3 What `decode` checks (in this order, as `mfile.cho`)
magic, version, section count and tags in order and every length against the bytes there (truncated), the SHA-256 trailer (checksum),
then, **for a model kind of 4 or above, the sum order** at the end of `meta` (`ml.model.version` unless it is 2; `design.md` section 4.1, added at `v1-post`),
then every field: the schema (as `mfile.cho`), the plan (parses, and is canonical: its canonical form is itself), the steps (each payload
by the rules of `docs/transforms.md` section 4, chained: the input width of step `i+1` is the output width of step `i`, and the first is
the number of feature columns), the **agreement of the plan with the steps** (as many transform lines as steps, and the kind of each
step is the kind its line asks for), the model payload (its leading `u32 kind` is a registered model kind and its length is at least
4; **the estimator's own layout is checked by the estimator's reader**), the meta section (as `mfile.cho`, `hyperparameters = 0`; for a model kind of 4 or above it is 4 bytes longer: the `u32 sum_order` = 2 after the compiler pin, which `file.encode` writes for such a kind and `decode` requires; kinds 1 to 3 keep the exact bytes of `mfile.cho`).
Anything else is a code of `transforms.codes` (`ml.model.*`).

### 5.4 The estimator hook
`file.encode(plan, schema, steps, model_payload, meta)` takes the `model` section payload as bytes (`u32 kind` then the records of
`models.record`, `tree.model` or `mfile.cho`). `file.decode` returns a view of those bytes (offset and length); the caller decodes them
with the estimator's own reader, with the width the steps leave (`pipeline.run.output_width`). Wiring kinds 4 to 7 means calling those
readers there and teaching `mfile.cho` the section; nothing in this library names an estimator but the table of `model` lines.

## 6. Results

The gates P1 to P9 are run by `sh scripts/gate_pipe.sh` (all green in this session; `PIPE_FAST=1` is what the mutants run) and the numbers of the
oracle and the mutants are in `docs/transforms.md` section 6, which judges the same code.

| gate | evidence |
|---|---|
| P1 plan | `tests/t_pipeline.cho` (canonical forms by hand, floats `1e-1` `.10` `-0` `0.30000000000000004`, fixed point, 45 refusals with code, line and column); `oracle_pipe.py --plans`: 1,500 respelled plans (key order, blanks, tabs, CRLF, comments, leading zeros, equivalent floats), 1,500 fixed points, 1,188 float spellings against a Python canonicaliser, and 1,500 garbage plans whose verdict and position equal a second implementation of the grammar written in Python |
| P2, P3 | `test_fit_uses_the_training_rows_only`, `test_each_step_feeds_the_next`; chains in `oracle_pipe.py --pipeline` (600); mutants `fit-on-the-test-rows`, `step-two-fitted-on-the-raw-matrix` killed |
| P4 | `file.encode`/`decode` round trip; a file with no steps is accepted by the independent reader of `tests/conformance/test_model_file.py` and equals the layout of `mfile.cho` written again in Python, byte for byte (2 files); 150 files with steps read by an independent Python reader that applies the stored parameters and equals the driver's predict bit for bit. **Not compared with a file written by the `ml` binary**: it needs `cancho install`, which was not allowed here (the hash printed in `docs/cli.md` section 3 for the sales example is not reproduced by this layout with any of the pins and scales tried; the size, 316 bytes, is) |
| P5 | `scripts/check_pipeline_api.py` (no `pub fn` that applies takes a plan or a spec or mentions a fit; `Loaded` is built by `file.decode` only; every matrix-returning `pub fn` is classified); `test_editing_a_stored_parameter_changes_the_prediction` (divisor 10.0 -> 20.0 with the trailer repaired: the output halves; without repair `ml.model.checksum`; 0.0 and -10.0: `ml.model.malformed`); `test_the_plan_must_agree_with_the_steps` |
| P6 | `test_columns_are_matched_by_name`, `test_a_reordered_input_gives_the_same_output`; `oracle_pipe.py --pipeline`: 300 alignments (missing, extra, mistyped, duplicate, reordered) against a Python reference, text of the repair included |
| P7, P8 | `tests/t_pipeline_fuzz.cho` 4 passed: 4,000 mutated plans, the limits at both sides of 64 KiB, 64 lines, 4,096 names, 255 bytes; 1,500 fits through the pipeline with hostile plans and caps; every truncation, every byte flipped three ways with and without a repaired trailer, and 600 seeded multi-byte damages of a 3-step file |
| P9 | `tests/t_registry.cho` (7), `scripts/check_registry.py`: 7 model kinds, 6 stream kinds, 6 step kinds, 5 section tags, 59 refusal codes in 4 bands |

## 7. Open questions

1. **The `model` line's keys are provisional** (`lib/pipeline/plan.cho` `defs_of`): constant, linear, logistic, kmeans, tree and forest with the keys the libraries
   have today. The design example `rate=0.1` for logistic regression is not accepted because the Newton fit has no rate; the wiring stage owns this table.
2. **Names with a blank, `,`, `=` or `#`** cannot be addressed in a plan (only by `*`). Quoting is not designed.
3. **Text categories** are not supported (the reader gives numbers). A text column must be dropped.
4. **`unknown=ignore` for `label` gives -1.0**, a value no category has; a downstream estimator sees it as a number.
5. The summation order is the package's, through `transforms.dsum`. When the lane-order change lands, `oracle_pipe.py --sums` fails until `ordered_sum`
   there is updated (one function), and the stored step bytes of standard scalers and mean imputers change with it: model files fitted before the change are
   not byte-identical to files fitted after it, though they still apply (the stored parameters are read, not recomputed).
6. The O(n^2) name matching (`names.find`, `first_bad`, `schema.align`) is bounded by 8,192 columns; it was not timed.
7. `mfile.cho` still reads only `steps = u32 0`; the wiring stage extends it (or switches to `pipeline.file`).
8. D2 (the same bytes on another machine) is untested, as in every stage.

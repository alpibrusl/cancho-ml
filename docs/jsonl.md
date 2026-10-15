# JSON lines as an input of `fit`, `predict` and `eval` (issue #54)

**Status: sections 1 to 6 are the design and the gates, written before any code of this stage (`tools/ml/jsonl.cho`). Section 7 holds what the scripts printed; a
number or a verdict is in it only if a script of this repository printed it, with its conditions.** Tags as in `design.md`: **[choice]** a decision the
maintainer can overturn, **[read]** read from a source file or a document at a named revision, **[measured]** printed by a command, with its conditions,
**[open]** not decided.

The question: can `ml fit|predict|eval` read a file (or standard input) of JSON lines, one object per line, and give **exactly** the model and the answers
the equivalent CSV gives, with the refusals of a tool an agent can read, and without a copy of anyone's code?

## 1. The decision: a reader of `ml`'s own (B), not cancho-table's reader as a store (A)

cancho-table reads JSON lines since PR #47 (R2; `readers.md` section 12 in that repository). The two ways to have that in `ml` are (A) pin it as a store, as the
five numeric modules are pinned (`docs/reader.md` section 1), or (B) a small reader of `ml` over `std.json`. **Measured** at cancho-table `0a1b7dc`
(`main`), by reading the `import` lines of `tools/table/*.cho` and following them (a script in this section's reproduction below):

| root module | modules in its import closure | lines in the closure | declares `struct Query` (the table's plan) | non-`std` imports that are not modules of the repository |
|---|---:|---:|---|---|
| `jsonl` (the input of the table binary) | 30 | 15,777 | yes (`query`) | `toolbox.cli fail limit lines out path place text` |
| `jrec` (one record: the line to the cells) | 18 | 9,553 | yes (`query`) | `toolbox.fail out text` |
| `jscan` (the range scan) | 20 | 10,328 | yes (`query`) | `toolbox.fail lines out text` |
| `jfields` (the declaration of `--fields`) | 1 | 255 | no | none |
| `jflat` (the flat-object scanner, an accelerator) | 1 | 202 | no | none |

* **[read] `jrec.extract` is not a stable interface.** Its signature is `extract(heap, line, tape, st, table, sbytes, nfields, flat, nodes, absent, qdelim,
  cells, rec0, a)`: `cells` is the engine's triple of ints per field, `a` is the engine's tally (`k_records`, `k_abort`, `k_err_col`... "as `engine.cho`
  names them"), and its refusals are abort codes of the engine (50 to 55). It imports `engine`, which imports `agg`, `expr`, `query`, `report`, `sorter`,
  `jout` and `writer`: the table's plan, its aggregates and its writer. `scripts/publish_numeric.py` refuses to publish a module that declares `Query` or
  imports a module that is not published (`numeric-package.md` in cancho-table, the `query` withdrawal of #44), and `jrec` would need 14 more
  modules to be published or cut. The two modules that are free of the plan (`jfields`, `jflat`) are the declaration of dotted paths and an accelerator;
  neither holds the reading of a value.
* **[read] The minimal interface** ("for a line and a declared list of paths, the text of each cell, or a refusal with a rule and a position") does not
  exist in table's sources: it would be a rewrite of the 566 lines of `jrec.cho` with the engine taken out, plus a new gate set in table, plus a
  publish script with `--check`, plus a store consumed by `ml`. That is the "few hundred lines" the extraction has to stay under, and it does not: the
  extraction is a new module, and `ml` would pin a module written for it.
* **[read, measured] The compiler.** `std.json.parse_with` is the call table's reader is built on. It is in cancho `e47d647` (`std/json.cho` line 594) and
  **not** in `a4572ea`, the compiler `ml` pins (`grep -c parse_with std/json.cho` is 0 there). A store of `jrec` would not type-check with `ml`'s compiler
  (the same finding as `flt` and `float_of_bits` in `numeric-package.md` there): choosing (A) moves the compiler pin of `ml`, which changes the bytes
  of every store `ml` publishes (`docs/stores.md`) and every model file's `compiler` field, for a reader of 500 lines.

**Decision: (B).** `tools/ml/jsonl.cho` is an independent reader written from table's specification (`readers.md` section 12.2 to 12.5 of cancho-table: strict RFC 8259,
numbers are their exact token text, strings are decoded, a duplicate key at a declared path is refused, a line that is not an object is refused, limits
before allocation), over `std.json.parse` of `a4572ea`. **No table code is copied.** The conformance oracle is Python's `json`, not table. Nothing is
pinned from cancho-table beyond what `ml` pins today, and no branch of cancho-table is needed (`jsonl-store` was not made).

Where (B) differs from table on purpose: `--fields` items are **flat names** (the key of the top-level object, literally: `a.b` is the key `"a.b"`, never
the path `a` then `b`; table's dotted and indexed paths are not read), because the issue's contract is "never guessed" and a flat name is the only reading
that needs no rule about arrays and nulls on the way; and the values are typed by the column (section 3), because `ml` reads typed columns where table
reads text. A user who wants `user.age` writes `table --input-format jsonl --fields id,user.age,spend --select '#2,#3' --format csv FILE | ml fit ... -` (section 7 shows the pipe answering).

**Reproduction of the closure table** (`ci`-free, no compiler): `python3 scripts/jsonl_closure.py DIR_OF_TABLE_CHECKOUT` prints the rows above.

## 2. The surface

| flag | verbs | default | |
|---|---|---|---|
| `--input-format csv\|jsonl\|ndjson` | `fit`, `predict`, `eval` | `csv` | never guessed from a name or from the bytes; `ndjson` is `jsonl` |
| `--fields NAMES` | `fit` | none | the columns, flat names separated by commas; **required** with `jsonl` at `fit` (`args.required-flag`); `args.conflict` with `csv` and for `predict` and `eval` |
| `--absent refuse\|empty` | `fit`, `predict`, `eval` | `refuse` | what a declared key that the line does not hold is; `args.conflict` with `csv` |
| `--max-json-nodes N` | `fit`, `predict`, `eval` | 4096 | the most JSON nodes a line may hold, 1 to 1,048,576 (`ml.param.out-of-range` outside); `args.conflict` with `csv` |

The existing limits keep their meaning for JSON lines: `--max-line-bytes` (a longer line is `limit.line-too-long`, checked before the line is parsed),
`--max-rows`, `--max-columns`, `--max-cells`, and `--max-bytes`, which for JSON lines also bounds the **input** (a file of JSON lines is read whole, up to
`--max-bytes`, default 16 MiB, ceiling 256 MiB; a CSV file is not, it is streamed) and the CSV that the reader makes of it (section 5).

**The fields.** `fit` and `cv` (when the splits stage merges; `fit --split` is the case here) declare the columns with `--fields`: the header of the table
is those names in that order. `predict` and `eval` take their fields **from the model file**: the stored schema names the columns a caller must give
(`ml describe`, `request.header`), so no flag is needed and none is accepted. `predict` reads the feature columns (the target is skipped, as in CSV);
`eval` reads the feature columns and the target (a clustering has no target). The names and the order of the stored schema are used, so a line is read
**by key**, in any order, and **keys the model does not name are ignored**. This differs from CSV on purpose: a CSV header with a column the model does not
know is `ml.schema.mismatch` (repair `--drop`), because a CSV row has no names but the header; a JSON object carries its names with its values and an
object with more keys than the model reads is the normal case (an event with ids and timestamps). The ignored keys are not checked (not even for
duplicates), as table does for keys the plan does not look up.

**Standard input.** `fit -` and `eval -` read standard input as `predict -` does (`docs/serving.md` section 2): up to `--max-bytes` bytes, kept in memory
(`limit.input-too-large` past it), CSV or JSON lines alike. The operand is required for `fit` and `eval` (no operand is `args.missing-operand`, as before):
`predict` is the only verb where no operand means standard input. Memory: the bytes once, and for JSON lines the CSV made from them as well while the
conversion runs (the JSON bytes are released before the verb runs), so the peak is below three times `--max-bytes` (the JSON lines, the CSV, and the copy of the CSV that the verb reads). `fit - --split ...` needs no
second read (the scan already runs twice over the bytes in memory). `fit --cv` does not exist on `main`; when it merges it reads the same bytes in
memory, which is why `fit -` was built with a bounded buffer and not as a stream. `--track` (not on `main`): the training file is `-`; **[choice] the
tracking stage records the name `-` and the SHA-256 of the bytes read from standard input** (for JSON lines, of the JSON lines bytes, before the
conversion), since the bytes are in memory when the hash is taken; it does not refuse. This section is the decision; the tracking stage builds it.

**The model file records nothing about the input format.** A model fitted from JSON lines and one fitted from the equivalent CSV are the same bytes
(gate G1); `predict` and `eval` take either format whatever the fit read. `ml-predict` takes `--input-format`, `--absent` and `--max-json-nodes` for
`predict` (section 6, G6): the reader is the same source and adds no capability (standard input and files are what it already reads).

## 3. What a cell is: the value-to-cell table

A line is one JSON value of the strict syntax of RFC 8259 (no comments, trailing commas, leading zeros, `NaN`; strings valid UTF-8 with no raw control
byte and no lone surrogate escape; nesting at most 128 deep), and it must be an object. The table below is applied to the value of each declared key.
A record is a line; a blank line (only space, tab, CR) is skipped; a byte order mark at the start of the input and a CR before the LF are dropped; the
last line may have no LF.

"Numeric" below is a column that the reader reads as a number: at `fit`, a column that is not named by `--cat` (and is not the target); at `predict` and
`eval`, a column the stored schema types `int`, `dec` or `float`. "Text" is a column named by `--cat` (at `fit`) or typed categorical by the stored
schema.

| the value at the key | the cell | refusal |
|---|---|---|
| a number | its **exact token text** (`5`, `5.0`, `-0`, `1e3`, `1E-7`, 17 digits, 40 digits): never through a double. The reader then reads it as it reads a CSV cell, so `1e3` in an `int` column, a decimal with more digits than the column's scale, and `1e400` are the CSV path's refusals (`ml.input.*`, `ml.schema.mismatch`), not a rounding | none of its own |
| a string, in a text column | its decoded text (escapes, `\u0000`, surrogate pairs; UTF-8). An empty string is an empty cell: **missing**, as in CSV. Quoted as CSV when it holds a comma, a quote, a CR or an LF | none |
| a string, in a numeric column | not a cell | `ml.jsonl.type-mismatch` (`found: "string"`). A JSON string `"5"` is **text**, a number `5` is a number: the reader never turns one into the other |
| `true`, `false`, in a text column | the word `true` or `false` (text) | none |
| `true`, `false`, in a numeric column | not a cell | `ml.jsonl.type-mismatch` (`found: "boolean"`): never guessed as 0 and 1; encode them upstream, or declare the column with `--cat` |
| `null` | an empty cell: **missing** | none (the models decide what a missing cell is, as for CSV) |
| the key is not in the object | `--absent empty`: an empty cell. `--absent refuse` (default): not a cell | `ml.jsonl.missing-field`, repair `retry` with `--absent empty` |
| an object or an array | not a cell | `ml.jsonl.not-scalar` (`kind`): `ml` has no nested columns; flatten with `table` first |
| the key twice in the object | not a cell | `ml.jsonl.duplicate-key`, wherever the second is (the two keys compared decoded: `"a"` and `"a"` are one key). Only keys that are declared fields are judged |
| a line that is not valid JSON | not a record | `ml.jsonl.syntax` (`offset` in the line, `code`, `reason`) |
| a line that is valid JSON but not an object | not a record | `ml.jsonl.not-object` (`kind`) |
| a line with more than `--max-json-nodes` nodes | not a record | `ml.jsonl.too-many-nodes` (`limit`), refused when the tape is full: nothing sized by the line exists beyond the tape of `--max-json-nodes` |
| a line longer than `--max-line-bytes` | not a record | `limit.line-too-long`, before the line is parsed |

**One column, one kind.** A column holds values of one JSON kind (number, string or boolean; null and absent are the missing cell and have no kind).
When a text column has a value of a different kind than its first non-null value, that is `ml.jsonl.mixed-kinds` naming the column, the line and kind of
the first value, and the line and kind of the one that differs ("a column with both numbers and strings in different lines"). The hazard is
numbers and text that print alike: a categorical column with `5` on one line and `"5"` on the next would be one category in the CSV and two kinds of
value in the file. A numeric column with a string or a boolean never gets as far: it is `ml.jsonl.type-mismatch` at the first one. A text column of
numbers only (`1`, `2`, `3` as classes) is accepted; its cells are the token text and the reader makes them categories (`007` cannot be a JSON number,
so a code with a leading zero is a string).

**How numeric-looking text is protected.** Not by quoting in the CSV (a CSV has no types and a quoted `5` is the number 5 for a numeric column) but by
the rule above: the converter knows the type of every column before it writes a cell, so a string never reaches a numeric column. In a text column the
cell is a category whatever it looks like, because the reader reads that column as text (`docs/categorical.md` section 2: "`007` and `7` are two
categories").

**Rule precedence**, first match, by line in file order: the line is longer than `--max-line-bytes`; it is not strict JSON, or has more nodes than `--max-json-nodes` (the parser fills its tape as it goes, so a line that is both too big and malformed further on is the node bound's); it is not an object; a declared key occurs twice (the first repeat in key order); then each declared field in declaration order: absent or null, nested, type, mixed kinds. The first refusal is the answer; the other refusals of the file are not searched for. A file larger than `--max-bytes` is refused before any line is read, and a CSV larger than `--max-bytes` after the line that passes it.

## 4. How the reader is built: the conversion to the CSV the reader already reads

`tools/ml/jsonl.cho` turns the JSON lines into the **CSV bytes** that the equivalent CSV file would hold, in memory, and `ml` then runs every verb on
those bytes as it runs `predict -` on standard input (`paths.of_mem`: the `Mem` source of `ml.csv`, which the two passes of the reader already read).
This is the design because the gate G1 (same model bytes, same answers) is then true by construction for everything downstream of the reader, and
because no verb, model or plan is touched: the change is one function between "resolve the paths" and "run the verb" in `tools/ml/main.cho`.

The CSV it writes: the header is the declared names in order (quoted as CSV when a name holds a comma, a quote, a CR or an LF); a record is the cells in
the order of the header, separated by commas, ended by LF; a cell is quoted (`"` doubled) only when it holds a comma, a quote, a CR or an LF; an empty
cell of a table of one column is written `""` (a blank line would not be a record); a blank line of the input is a blank line of the output (the CSV
reader counts blank lines, so a line number is kept). Line numbers: a refusal of the JSON lines reader names the `line` of the input and the `row` (the
record's number); a refusal of the CSV reader behind it (`ml.input.non-numeric`, `ml.schema.mismatch`...) names the `line` of the **generated CSV**,
which is the input line plus one (the header) while no string in the file holds a line break. This is a limit, not a feature, and is said in
`cli.md`.

The names of a stored schema (`predict`, `eval`) are read from the model file by the same decoder the verbs use (`pipeline.schema`, `ml.mfile` for a
model with no steps), before the data is read. A model file that cannot be read leaves the file alone: the verb then refuses on the model, as before.

Lookup of the keys of a line is one pass over the pairs of the top-level object with a small hash table of the declared names (open addressing on a
31-bit hash, comparing the decoded key to the name), so a line with 50,000 keys and 4,096 declared fields costs the keys and not their product.

## 5. Limits and costs

* The input is bounded by `--max-bytes`; a line by `--max-line-bytes` (default 1 MiB, ceiling 16 MiB) before parsing; a line's tape by `--max-json-nodes`
  (3 ints a node, once, allocated before the first line); the generated CSV by `--max-bytes` (`--absent empty` with 4,096 fields turns a 2-byte line `{}`
  into 4,096 bytes; the bound is checked before each append, `limit.input-too-large` with `detail.what: "csv"`); the number of fields by
  `--max-columns` (checked on the flag, before anything is allocated).
* Depth: the parser's 128. Invalid UTF-8 anywhere, a raw control byte in a string and a lone surrogate escape are `ml.jsonl.syntax`.
* A 10 MB line, 100,000 empty lines, deep nesting to the node limit, huge numbers: gate G4.
* Time is the parser's plus one pass; the cost is measured in section 7 and not claimed against anything.

## 6. The gates (written before the code; each is shown failing by a mutant)

| id | gate | judged by | a mutant it must kill |
|---|---|---|---|
| G1 | **Equivalence.** For iris, diabetes, wine and breast cancer (bundled) and for generated tables, `ml fit` on the JSON lines and on the equivalent CSV give the **same model file bytes**, and `predict` and `eval` give the same standard output bytes, for every model kind (constant, linear, logistic, tree, forest, kmeans), with plans using scale, impute, bucket, encode, freq, hash and `--cat` text columns, missing cells written as `null` and as absent keys (`--absent empty`), `fit --split`, `predict` and `eval` from a JSON lines file with the keys in another order and with extra keys, standard input for `fit -`, `eval -` and `predict -`. `ml cv` is judged when the splits stage is on `main` | `tests/conformance/test_jsonl_equiv.py` | the converter writing one cell too few; a different column order; `null` read as the text `null` |
| G2 | **Cell text against Python's `json`.** `jsonl-check` (`tools/jsonl-check`) prints the CSV the converter makes; the test builds the expected bytes from `json.loads(parse_float=str, parse_int=str)` and its own CSV writer, and compares byte for byte for numbers (`1`, `1.0`, `-0`, `1e3`, `1E-7`, 17 digits, subnormal, huge, `1e400` read as text and refused downstream), strings (escapes, surrogate pairs, NUL, quotes, commas, CR, LF), big integers, booleans and nulls, and the refusals of nested values | `tests/conformance/test_jsonl_oracle.py` | the exact-text rule replaced by a double round trip; a decoder that skips surrogate pairs |
| G3 | **Refusals.** One fixture per new rule tag with its repair run, and the `args.*` conflicts | `tests/conformance/test_jsonl_rules.py`, `fixtures.py`, `test_tables.py`, `test_rules.py` | a rule whose message or repair is changed |
| G4 | **Fuzz and hostile lines.** Huge numbers, nesting to the node limit and one past it, invalid UTF-8, a BOM, CRLF, a missing final newline, a 10 MB line, 100,000 empty lines, lines that are not objects, random bytes and mutated valid files: an answer of the contract (exit status of a rule or 0), never a trap, no hang (a timeout is a failure); thread invariance of `fit --threads` | `tests/conformance/test_jsonl_fuzz.py` | the line bound not applied; the node limit off by one |
| G5 | **Mutants** of the mapping, absent and null, numbers, the string and number protection, the limits, the field order, standard input | `scripts/mutants_jsonl.py` (`--check`: each edit applies; the run rebuilds a copy of the tree) | all killed, or each survivor explained in section 7 |
| G6 | **Authority.** `ml-predict` and `ml-serve` gain no capability; the manifests are what `scripts/check_bins.py` derives; `ml` needs no new label for standard input at `fit` (`io_read` is already there) | `scripts/check_bins.py`, `manifests/` | the existing mutants of `check_bins.py --mutants` |
| G7 | **The fast CI job** runs `scripts/run_conformance.py --no-oracles` in a Python without numpy: the new tests import only the standard library and `jsonschema`, so none is "not run" | the runner in a venv with `ci/requirements-fast.txt` only | a test that imports numpy at top level (it would be listed NOT RUN) |
| H | **Cross-platform fingerprint.** `scripts/xplat_check.py` is equal to `docs/bench/xplat_fp.txt` on the Mac and on gram; two cases (a JSON lines fit and its predict) are appended to the fingerprint without changing its existing lines | `scripts/xplat_check.py` on both machines | a converter that depends on the platform |

## 7. What was built and measured

**Built.** `tools/ml/jsonl.cho` (the converter, 488 lines: the declaration of the fields, one record, the input), `tools/ml/jsonlcli.cho` (the flags, the fields of a model file, the refusals, the replacement of the input, 434 lines), `tools/jsonl-check/main.cho` (the driver of the gates, a `[[bin]]` of `cancho.toml`), and edits of shared files, all small: `tools/ml/main.cho` and `tools/ml-predict/main.cho` (two lines each: the flag check and the conversion between "resolve the paths" and "run the verb"), `tools/ml/app.cho` (`stdin_wanted` takes `fit -` and `eval -`), `tools/ml/wire.cho` and `tools/ml/catcli.cho` (the loaders take the `Rp` of the input, so the bytes in memory reach them: three call sites), `tools/ml/spec.tmpl` (four flags, eight rules, one limit; `spec.cho`, the schemas, the manifests and `generated/*/built.cho` are regenerated from it), `cancho.toml` (two source lists and the `jsonl-check` bin), `tools.toml` (a reason), `tests/conformance/fixtures.py` (`_jsonl_cases`, appended), `scripts/site.py`, `scripts/xplat_check.py` (two cases appended), `scripts/check.sh`, `.github/workflows/ci.yml` (one build line), and the patterns of `scripts/mutants_cli.py` and `scripts/mutants_serve.py` that named the lines these edits changed.

**The decision, measured** (section 1): `python3 scripts/jsonl_closure.py DIR` on cancho-table `0a1b7dc`: the input module `jsonl` has 30 modules and 15,777 lines in its import closure, `jrec` (one record) 18 and 9,553, both with the table's `Query`; `jfields` and `jflat` (255 and 202 lines) are free of the plan and hold neither the reading of a value nor an interface to it. `std.json.parse_with` is in cancho `e47d647` and not in `a4572ea`. No branch of cancho-table was made.

**The gates, run on the final tree** (Mac, macOS 26, arm64, `a4572ea`; the machine was loaded, load average above 50 during the cost run):

| gate | result |
|---|---|
| G1 `test_jsonl_equiv.py` | 20 tests pass |
| G2 `test_jsonl_oracle.py` | 26 tests pass; 250 seeded random files agree with `jsonl_ref.py`; the CSV bound is exact to the byte; depth 127 and 128 are accepted and 129 refused, as the reference says |
| G3 `test_jsonl_rules.py` | 12 tests pass; 24 fixtures (`_jsonl_cases`) reach the 8 new rules, 6 `args.*` and limit rules of the new flags |
| G4 `test_jsonl_fuzz.py` | 11 tests pass: 0 traps, 0 hangs; 120 random and mutated files; hostile field lists of 8,192 and 8,193 names, and of 400,000 empty names, are answered quickly (a first version of the field list parser allocated a region of the size the list chose and trapped at 4,097 names; the test that found it is `test_hostile_field_lists_cost_their_length`) |
| G5 `scripts/mutants_jsonl.py` | run on the final tree: 47 mutants, 46 killed, 1 explained (`cli-shown-name-lost`: the name of the converted input is read by no answer). `docs/bench/mutants_jsonl.log` |
| G6 | `scripts/check_bins.py` and `scripts/manifest.py --check` pass; the derived authority of the three binaries is the same set of labels as before (`ml-predict`: `args`, `dir_read`, `err_write`, `file_read`, `fs_read("")`, `heap`, `io_read`, `io_write`); `ml-serve` compiles the new modules and its `main` reaches none of them |
| G7 | in a Python with `ci/requirements-fast.txt` only (no numpy): `scripts/run_conformance.py --no-oracles` runs the four new modules, 68 tests at the time, none "not run" |
| H | `scripts/xplat_check.py` prints `ok, 96 lines for 24 fits` on the Mac; the first 88 lines of `docs/bench/xplat_fp.txt` are unchanged and the 8 appended lines carry the hashes of the CSV cases (the model file and the predict answer of `logistic-iris-jsonl` equal those of `logistic-iris-t1`, and of `forest-pets-jsonl-cat` those of `forest-pets-hash-t1`) |
| `table ... | ml predict -` | `table --input-format jsonl --fields id,user.age,spend --select '#2,#3' --format csv events.jsonl | ml predict --model-file m.ml -` answers (cancho-table built at `0a1b7dc` with its pinned compiler `e47d647`, a scratch clone): the CSV `table` prints is the header `user.age,spend` and the values, and `ml` reads it |
| Linux x86-64 (`gram`, Ubuntu, `taskset -c 0-6`, `a4572ea` built there) | `scripts/check.sh` ran once, whole: every step before the conformance suite passed (the pins, the stores, the unit and numeric gates, the three binaries, `check_bins`, the reader, categorical and encoder gates); the suite ran 508 tests in 1,649 s and 5 failed, all tests that the change made wrong or new: the field list test passed one argument of 400,000 bytes, which Linux refuses (128 KiB per argument); `test_wire_baseline` took the 24 new fixtures for constant-model fixtures (the flags are now in the filter of `scripts/golden_constant.py`); `test_serving` listed the flags of `ml-predict` and said that `fit -` reads a file named `-` (it reads standard input now; `./-` is the file). Fixed, rerun there with the other touched modules (125 tests pass), then the steps of `check.sh` after the suite run by themselves because `check.sh` stops at the first failure (the mutant checks, the non-finite, HTTP, wiring, paths, encoder, string table and pipeline gates, `check_docs.sh` with `site.py --check` against the Linux binary): all pass. `xplat_check.py --require-predict` prints `ok, 96 lines for 24 fits equal docs/bench/xplat_fp.txt (linux, x86_64)` and its 15 mutants are refused |
| the mutant suites on `gram` | `mutants_jsonl.py` 47: 46 killed, 1 explained, baseline passes; `mutants_cli.py` 52 of 52 killed, baseline passes; `mutants_serve.py` 60 of 60 killed (one mutant that fed `eval -` to standard input became equivalent when `eval -` began to read it, and is replaced by two that read `eval -` and `fit -` as files; both killed), baseline passes. Each baseline is the whole conformance suite on the unmutated copy: 508 tests |

**Cost** (one run each, minimum of 3, Mac with a load average of 56, so the figures are an upper bound and mean nothing against another machine): a file of 120,000 lines with 13 numeric keys (18,480,000 bytes), and the same table as a CSV (10,320,040 bytes). `jsonl-check` (the conversion alone) 0.43 s, about 43 MB/s of JSON lines; `ml fit --model constant` on the CSV 0.11 s, and on the JSON lines 0.30 s; the two model files are equal. Nothing is claimed against another reader or another library.

**Not done.** `ml cv` and `--track`: not on `main` (the decisions are in section 2). Dotted or indexed paths in `--fields`: not read (section 1). The line of a CSV-reader refusal is the generated CSV's (section 4); a mapping back to the input line would need the reader to carry the line map. `--max-bytes` for a JSON lines file bounds memory (the file is read whole), where a CSV file is streamed.

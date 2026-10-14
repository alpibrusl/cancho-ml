# cancho-ml serving, first half: standard input, `ml-predict` and `describe` (issues #11 and #16)

**Status: this document is written before the code it judges. Sections 1 to 8 fix the behaviour, the authority rows, the contract of `describe` and the gates; section 9
holds the results and is filled only with what a script of this repository printed, with its conditions. Section 10 lists what the server (`ml-serve`, issue #16, second half) has to decide.**
Tags as in `design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from the code or from a document of the pinned compiler, **[measured]** printed by a
script here, **[open]** not decided.

Three pieces, in the order a server needs them:

1. `ml predict ... -` (and `ml predict ...` with no file operand) reads the CSV from standard input.
2. `ml-predict`, a second binary built from the same sources, which can predict and describe and can do nothing else: its authority row has no write of any kind and no thread.
3. `ml describe --model-file M` prints what a model file requires of its caller. A server turns it into a request schema, so its field names are a contract (section 5).

## 1. The gates (written first; each one must be able to fail)

| id | gate | test or script | mutants that must be killed |
|---|---|---|---|
| V1 | `ml predict` of a CSV on standard input prints the bytes `ml predict` prints for the same bytes in a file, over a corpus of stored models (kinds 1 to 7, categorical steps, missing cells, unseen categories, hash steps) and inputs, in JSON and in text | `tests/conformance/test_serving.py` | stdin read as a different number of bytes; a last line without newline dropped; a byte order mark kept; `--limit` and `--from` ignored through stdin |
| V2 | every refusal of a file is the refusal of the same bytes on standard input, equal but for `detail.path` (`-` for standard input): ragged row, bad quote, unterminated quote, over `--max-rows`, `--max-columns`, `--max-cells`, `--max-line-bytes`, schema mismatch, a cell that is not a number, no data | `test_serving.py` | a refusal that reads the file name of standard input; `--max-line-bytes` not applied |
| V3 | the bound of standard input: more than `--max-bytes` bytes is `limit.input-too-large` before any parse, an empty standard input is the refusal of an empty file, a closed standard input is an empty one | `test_serving.py` | the bound off by one; a read past the bound |
| V4 | the operand: `-` and no operand mean standard input, `./-` is a file named `-`, `ml fit ... -` and `ml eval ... -` read a file named `-` as before, `--root` does not apply to standard input and is still checked as a value | `test_serving.py` | `./-` read as standard input; `-` read as a file |
| V5 | the authority row of `ml` is `io_read` more than before, with a reason in `tools.toml`; the manifest check fails when any other label is added to the derived row, and when the ceiling or its reason is edited | `test_authority.py`, `scripts/mutants_serve.py` | a ceiling without `io_read`; an `io_read` without its reason; a derived row with `net_out`, `clock`, `ffi`, `fs_write`, `conc`, `file_write` added (in `test_authority.py`, on the derived row) |
| P1 | `ml-predict predict` prints the bytes `ml predict` prints, for every case of V1 and every refusal reachable by `predict`, from a file and from standard input (but for the program name inside a `retry` repair) | `test_serving.py` | `describe` mapped to another function in one binary; a table of one binary with another default |
| P2 | `ml-predict` takes `predict`, `describe`, `introspect` and `skill` and refuses `fit` and `eval` (`args.unknown-command`) and every flag of `fit` and `eval` (`args.unknown-flag`) | `test_serving.py` | `fit` reachable; a fit flag in the table |
| P3 | the authority row of `ml-predict` is the committed `manifests/ml-predict.authority.json`, within the ceiling of `tools.toml`, and has no `conc`, `dir_write`, `file_write`; the check fails on a mutant that adds a capability | `test_authority.py`, `scripts/manifest.py --check`, `scripts/mutants_serve.py` | a write in a reachable function; a thread in a reachable function; a stale manifest |
| P4 | the two binaries are built from the same sources: the file list of `ml-predict` is the file list of `ml` with the entry point replaced | `scripts/check_bins.py` | a copy of a module under `tools/ml-predict`; a file of `tools/ml` missing from the list |
| P5 | `ml-predict` writes no file: a run in a directory nobody can write to leaves the directory as it was (the byte-exact listing) | `test_serving.py` | none (the row says it); the check is that the row and the behaviour agree |
| D1 | `describe` agrees field by field with an independent reader of the file (`scripts/oracle_describe.py`) over corpora of models of all kinds | `tests/conformance/test_describe.py` | a field dropped, renamed, off by one, a type code misread, a category count wrong |
| D2 | damaged files: every byte flipped, every truncation, counts of 2^63 - 1, a repaired checksum: `describe` answers a refusal as data, never a trap, never an allocation sized by a number read from the file before it was checked | `test_describe.py` | a size read and allocated before the check |
| D3 | `ml-predict describe` prints the bytes `ml describe` prints | `test_serving.py` | none |
| D4 | the field names of `describe` are the contract of section 5: the JSON Schema of the answer is generated from the table of section 5, and a golden answer for a fixture model of each kind is committed | `test_describe.py`, `tests/conformance/golden_describe.json` | a renamed field |
| T1 | the one-flag-table gate covers the new verb and the new binary: every flag in `introspect` of `ml-predict` is a flag of `ml` with the same kind, role and default; every flag a verb of `ml-predict` takes is in its table; `skill` and the schema list them | `test_tables.py`, `test_serving.py` | a flag in the table of one binary and not the other |
| G | every existing fixture and every golden file is unchanged, except the texts that list verbs and binaries (re-recorded in one commit, with the reason) | the existing suite | |

The mutants of the new code are in `scripts/mutants_serve.py`; each is killed or a survivor is listed with its reason in section 9.

## 2. `predict` reads standard input

### 2.1 What the command does

`ml predict [flags] -` and `ml predict [flags]` (no file operand) read the CSV from standard input. **[choice]** Both spellings are accepted, as in `cancho-table`
(the design of its standard input, section 3 of its readers document): `-` is explicit, no operand is what a forgotten operand of an agent's subprocess does (an empty standard input, which is the refusal of an
empty file, `ml.input.empty`, not a hang on a terminal that no agent has). A file named `-` is `./-`. Only `predict` reads standard input: `fit` and `eval` need to read the
file twice or more and keep their operand a file; `ml fit ... -` is the file named `-`, as it always was.

`--root` does not apply to standard input: there is no path. It is still checked as a value, so a malformed one is refused as for a file. `--max-bytes` bounds it (2.3).

### 2.2 Why the bytes are kept

**[read]** The reader makes two passes over the CSV: the first checks every record and infers the column kinds without keeping a value, the second writes the doubles into a matrix
sized by the first (`reader.md` section 2.3). A file is opened twice. A pipe cannot be read twice, so **the bytes of standard input are read once into memory, up to `--max-bytes`
bytes, and the two passes run over that memory**. The reader code is the same: only the supply of 64 KiB chunks differs (a chunk from the file, or the next 64 KiB of the buffer).
Memory is the input plus what the file case needs. This is the price of two passes; a one-pass streaming predict is not built (section 10).

### 2.3 Bounds and refusals

* More than `--max-bytes` bytes (default 16 MiB, ceiling 256 MiB, the flag that bounds a model file) is refused as `limit.input-too-large` (exit 8), with `detail` `{"size": N, "max": M}`
  (the detail a too large model file has) where `size` is the bytes read when the bound was found (`max + 1`), not the length of the stream. Reading stops there: nothing past the bound is kept. **[choice]** The rule is the one a model file over the flag has; a CSV file has no such
  bound (it is read in chunks), so **a CSV that is accepted from a file and is larger than `--max-bytes` is refused from standard input**. This is the one difference, and it is documented in `introspect` (the flag's help).
* Every other check is the file's, in the file's order: `--max-line-bytes`, `--max-columns`, `--max-rows`, `--max-cells`, the quote and ragged-row refusals, the schema check, the missing cells.
* **A read error of standard input is the end of the input**: `getchar` answers the same value for an error and for the end, so the bytes read so far are the input (a truncated stream is then a refusal of the reader if it ends inside a record, `parse.csv-unterminated-quote` or `parse.csv-ragged-row`, or a shorter input). A closed standard input is an empty one, so `ml.input.empty`.
* **The order of the refusals [choice]:** the command line first (as before), then standard input is read (so a model file that is missing is found after the bytes were read), then the model file, then the parse. A caller that sends
  standard input to a command that is going to refuse its flags is not made to wait: the command line is checked before any read.
* The reads are `getchar` calls on `Io` (`io_read`): **[read]** the compiler of the pin has no bulk read of standard input (`read_bytes` is edition 9, cancho #363, after the pin). `cancho-table` measured `getchar`
  at 71 MB/s on the Mac and 264 MB/s on Linux. The speed of standard input here is not claimed; section 9 prints what was measured. Moving the pin to a compiler with `read_bytes` is section 10.

### 2.4 The authority of `ml`

`io_read` is a new label in `manifests/ml.authority.json` and in the ceiling of `ml` in `tools.toml`, with a reason (`[ml.reasons]`), and nowhere else: the check refuses every other
label added (the mutants of V5). **[choice]** Standard input is an authority of its own, `io_read`, and not `fs_read("")` on a path such as `/dev/stdin`: the second would hide it in the label
every file read already has. The row of `ml` is then `args, conc, dir_read, dir_write, err_write, file_read, file_write, fs_read(""), heap, io_read, io_write`.

## 3. `ml-predict`

### 3.1 What it is

A second binary, `ml-predict`, built by `cancho build --bin ml-predict` from **the same sources** as `ml`. It has its own `main` (`tools/ml-predict/main.cho`, about 100 lines of the same shape as
`tools/ml/main.cho`: it opens the world, checks the command line, runs one verb and answers). Everything else, the command line checks (`tools/ml/app.cho`), the reader, the pipeline, the predictors and the
answer, is the code of `ml`: P4 compares the file lists. **[read]** The authority of a program is what its `main` reaches (the compiler's derivation is over the functions reachable from `main`), so a binary whose
`main` never calls `fit` has a row without what `fit` needs. That is the whole mechanism: nothing is `#ifdef`ed out, nothing is copied.

### 3.2 The surface

* Verbs: `predict`, `describe`; and `introspect` and `skill` as every tool of the family has. `fit` and `eval` are `args.unknown-command`, as is anything else.
* Flags: the flags of `ml` that `predict` and `describe` take: `--root`, `--model-file`, `--drop`, `--limit`, `--from`, `--proba`, `--max-rows`, `--max-columns`, `--max-cells`, `--max-line-bytes`, `--max-bytes`, `--format`.
  **`--log` and `--log-max-bytes` are not flags of `ml-predict`.** **[choice]** The log is a file append (`file_write`, `dir_write`): the point of this binary is that its row has neither. A server that wants a log
  writes one itself, from the answers. Every other flag of `ml` is `args.unknown-flag`, with the nearest flag suggested as for any unknown flag.
* The table of flags of `ml-predict` is generated from the table of `ml` (`scripts/gen_models.py`) by name, so a flag has one definition; T1 compares the two.
* The answer is the answer of `ml`: the envelope says `"command": "ml"` and `"schema": "ml.v1"` in both (the bytes of an answer are the same from either binary; P1). `introspect` says `"tool": "ml-predict"`. **[choice]**
  One exception: the `retry` repair of a refusal is the command line to run again, and starts with the name the process was run by; the two binaries differ there and nowhere else (the gate P1 replaces one name by the other).
* Threads: prediction is single-threaded. **[read]** `wireio.predict` and the predictors it calls declare no `conc` (their signatures end `[heap, args, fs_read(""), dir_read, file_read]`); the threads of `ml` are
  `fit --threads` and `eval --threads`. So the row of `ml-predict` has no `conc`, and `--threads` is not one of its flags.

### 3.3 The authority row

The row is derived by `cancho authority` and committed in `manifests/ml-predict.authority.json`; the ceiling is in `tools.toml`:

`args, dir_read, err_write, file_read, fs_read(""), heap, io_read, io_write`

| label | why |
|---|---|
| `args` | the command line |
| `fs_read("")`, `dir_read`, `file_read` | the model file and the CSV file, opened below `--root` by name (`toolbox.place`), as `ml` does |
| `io_read` | the CSV on standard input |
| `io_write` | the answer, one line |
| `err_write` | the usage line on a refused command line, as the family's tools do |
| `heap` | the working data |

Not in the row: `file_write`, `dir_write` (no model, no log, no output file), `conc` (no thread), `fs_write`, `exec`, `ffi`, `net_out`, `net_in`, `clock`, `env`. `scripts/manifest.py --check` fails when the binary's
embedded row, the committed file or the ceiling disagree; `scripts/mutants_serve.py` plants a capability in a reachable function and in `main` and the check must fail for each.

## 4. `describe`

`ml describe --model-file M` and `ml-predict describe --model-file M` read the model file and print what it requires of a caller. No CSV operand: `ml describe FILE` is `args.too-many-operands`.
The file is decoded by the code that `predict` uses (one decoder), then described; the description is a function of the file's bytes alone (no path, no time, no host).

Refusals are data, exit codes and rules as for `predict` on the same file: `ml.model.bad-magic`, `ml.model.version`, `ml.model.truncated`, `ml.model.checksum`, `ml.model.malformed`, `ml.model.string-table`, `limit.input-too-large`,
`io.not-found`. A damaged file never traps (D2). **Nothing is allocated from a size read from the file before that size has been compared with the bytes that exist and with the file's bound** (the decoder of the model file checks
every length against the bytes that remain, `docs/stringtable.md` section 4); the categories listed are read from the decoded table, and the number listed is bounded by `--limit` (default 1,000, ceiling 1,000,000) per column.

## 5. The contract: the fields of `describe`

One JSON object in `data` of the envelope `ml.v1`. A field is always present; a value that does not apply is `null`. **[choice]** Names are stable: a rename is a new schema version. `TB` is the toolbox `text_or_bytes`
(a string, or `{"b64": ...}` for bytes that are not UTF-8).

| field | type | meaning |
|---|---|---|
| `model` | `constant`, `linear`, `logistic`, `tree`, `forest` | the estimator the plan names |
| `task` | `regression`, `classification` | |
| `kind` | integer 1 to 7 | the registry number of the file's model section (`lib/core/kinds.cho`): 1 and 3 constant regression, 2 constant classification, 4 linear, 5 logistic, 7 tree and forest |
| `plan` | string | the canonical plan text, section 1 of the file |
| `rows` | integer | training rows |
| `file` | object | `bytes` (integer), `sha256` (hex of the file's bytes, the digest of the whole file), `format` (`"CML1"`), `version` (1), `sections` (5, or 6 with a string table), `compiler` (the pin the file records, TB), `sum_order` (the summation order the file records for a model of kind 4 or above, 2 or 3; `null` for the constant kinds) |
| `target` | object | `name` (TB), `type` (`int`, `dec`, `float`), `scale` (integer for `dec`, else `null`) |
| `columns` | array | one object per column of the schema, in the file's order: `name` (TB), `type` (`int`, `dec`, `float`, `categorical`), `scale` (integer for `dec`, else `null`), `role` (`feature` or `target`), `required` (a feature: true; the target: false) |
| `request` | object | `header` (array of TB: the columns a prediction input holds, the schema's columns without the target, in the file's order), `optional` (array of TB: the target, which an input may hold or omit, content ignored), `extra_columns` (`"refused"`: a column that is not the model's is `ml.schema.mismatch`, unless `--drop` ignores it), `missing_cells` (`"refused"` for constant, `"needs_impute"` for linear and logistic: a missing cell of a column no step imputes is refused, `"handled"` for tree and forest) |
| `categorical` | array | one object per categorical input column, in the order of the steps: `column` (TB), `encoder` (`onehot`, `label`, `freq`, `hash`), `categories` (the number of strings the file holds for it; 0 for `hash`), `buckets` (the hash buckets for `hash`, else `null`), `listed` (the number of strings in `values`), `truncated` (`listed < categories`), `values` (array of `{"value": TB, "count": integer}`, bytewise ascending, at most `--limit`) |
| `na_values` | array of TB | the missing-value markers the file records (`--na-values` at fit), in the order given; `[]` for a model without a string table |
| `reader` | object or `null` | the reader limits recorded with the vocabulary: `max_categories`, `max_string_bytes`, `max_pool_bytes`, `max_cat_columns`; `null` without a string table |
| `classes` | array or `null` | classification: the two labels, ascending |
| `positive` | integer or `null` | a classification other than constant: the label whose probability `--proba` prints (the larger of the two for a model fitted without `--positive`; the file stores it). `null` for a constant model, which has no positive label |
| `output` | object | `columns` (the names of the columns of a prediction row: `["prediction"]` for a regression, `["label"]` for a classification, `["label", "probability"]` for a constant classification, which always prints its probability), `with_proba` (the columns with `--proba`: `["label", "probability"]` for a classification other than constant, else `null`, because a regression and a constant model refuse `--proba`), `label_threshold` (`"0.5"` for a classification other than constant, else `null`: the label is `classes[1]` when the probability is above it) |
| `limits` | object | `values_per_column` (the `--limit` in force) and `max_bytes` (the `--max-bytes` in force) |

The JSON Schema of the answer (`schemas/ml.v1.json`, the `describe` member of `data`) is generated from this table by `scripts/schemas.py`; D4 compares the schema, the table and the answers.

## 6. How one request row becomes the CSV the predictor reads

`ml-serve` takes JSON bodies (`--input-format` is not a flag of `ml`). It calls the same library code in-process, so the rule is: **build the CSV text `describe` says, and give it to the predictor as standard input would**.

1. The header is `request.header` (plus, optionally, the target), joined by commas; a name that holds a comma, a quote, a CR or an LF is written between double quotes with each quote doubled (RFC 4180).
2. One data row follows, the cells in the header's order, then one LF.
3. A cell by the column's type: `int`: the decimal digits of an integer (no fraction part, no exponent, no sign other than `-`); `dec` with scale `s`: a decimal with at most `s` digits after the point; `float`: any finite number the reader parses (`nan` and `inf` are refused); `categorical`: the string, between quotes when it holds a comma, a quote or a newline; an empty cell, or a string in `na_values`, is a missing value.
4. **The reader infers a column's type from the cells it reads, and checks it against the file's type** (`ml.schema.mismatch` with `mistyped`): a one-row CSV with `1.0` in an `int` column is a `dec` column and is refused. So the server renders an `int` as an integer and a `dec` or a `float` as a number, and a JSON value of the wrong type is a `problem+json` of its own before the CSV exists.
5. A batch is the same with more rows; `--max-rows`, `--max-cells` and `--max-line-bytes` bound it exactly as for a file.
6. The answer is `predict`'s document (`rows` as strings, `columns` from `describe.output`).

The server does not have to re-implement the check of step 4: whatever it does not catch, the predictor refuses with the rule of the file case.

## 7. The authority of the two binaries, together

| binary | row | what it can do |
|---|---|---|
| `ml` | `args, conc, dir_read, dir_write, err_write, file_read, file_write, fs_read(""), heap, io_read, io_write` | fit, predict, eval, describe; threads for the fits; writes the model and the log it is told to |
| `ml-predict` | `args, dir_read, err_write, file_read, fs_read(""), heap, io_read, io_write` | predict and describe; reads files and standard input; writes the answer |

No tool holds `ffi`, `net_out`, `net_in` or `clock` (`tools.toml`). `max_tools` is 10.

## 8. Not in scope

`--input-format` and any JSON body; a server; one-pass streaming prediction; a compiler move; `fit` from standard input; changing a model file's bytes (a file written before this document is described as it is).

## 9. Results

Machine: Mac, arm64, Apple M4 Max, macOS 26.2, compiler `a4572ea`, load about 8 while the suites ran. `scripts/check.sh` exits 0: 288 conformance tests pass in 524 seconds, none skipped.

| gate | result | script |
|---|---|---|
| V1 to V4 | 16 models x 30 inputs, from a file and from standard input: same status, same bytes on standard output and standard error (a refusal differs only where it names the file); inputs of 181,486 bytes (three 64 KiB chunks) equal; the bound exact at `--max-bytes` and one below; `-`, `./-`, no operand, empty and closed standard input as section 2 says; a refused command line answers while standard input stays open | `tests/conformance/test_serving.py` (28 tests, with P1, P2, P5, D3, T1) |
| P1, P2, P5, D3, T1 | `ml-predict` equals `ml` over the same pairs from a file and from standard input (the program name inside a `retry` repair is the one difference) and for `describe` of every model; it refuses `fit`, `eval` and 19 flags of them; its 12 flags are those of `predict` and `describe` without the log, each with the kind, role and default of `ml`'s; a run in a directory nobody can write to leaves it as it was | the same |
| V5, P3 | the derived rows are committed and embedded; the ceilings are exact; a derived row with `conc`, `file_write`, `dir_write`, `net_out`, `clock`, `ffi` or `fs_write` added fails `manifest.gate`; a source change that appends to a log from the `main` of `ml-predict` fails `manifest.py --tree` | `test_authority.py`, `mutants_serve.py` |
| P4 | the lists agree; 7 mutants refused | `scripts/check_bins.py` |
| D1 | 16 models x 3 page sizes (1,000, 5 and 1 strings per column): every field equal to `scripts/oracle_describe.py`; the corpus covers kinds 1, 2, 3, 4, 5 and 7, the four encoders, the four column types, 5 and 6 sections, the three `missing_cells` values, markers, truncated vocabularies, a float target | `tests/conformance/test_describe.py` |
| D2 | 28,718 damaged files, 0 traps; 1,763 with a repaired checksum are another valid file, equal to the oracle's description of it | the same, `ST_STRIDE=1`; `check.sh` runs it at that stride |
| D4 | schema, table of section 5 and answer agree on 16 fields in one order; golden answers of the 16 models | the same, `scripts/golden_describe.py` |
| mutants | 55 patterns of `scripts/mutants_serve.py`: 55 killed. A first run killed 53: a chunk that skips a byte (a small input never reached a chunk boundary: the multi-chunk input was added) and a ceiling wider than the row (the mutant ran only the manifest check: it now runs the tests) | `scripts/mutants_serve.py` |
| existing fixtures | every fixture and golden unchanged but `args-missing-verb` and `args-unknown-command` of `golden_constant.json` and `golden_numeric.json`, whose messages list the verbs and now name `describe`; re-recorded in one commit | `tests/conformance` |
| binary size | `ml` 1,062,592 bytes, `ml-predict` 706,296 bytes (66.5 percent), `cancho build` on the Mac. Linux not measured | `ls -l build/` |
| standard input throughput | a 10,182,226-byte CSV, `predict --limit 1` with a constant model: 0.06 s from a file, 0.20 s from standard input (`getchar`), minimum of 3, load about 8. A figure of this run, no claim | `/usr/bin/time` |

## 10. Open for the server

1. **Bulk read of standard input.** The pinned compiler has `getchar` only. `read_bytes` (cancho #363, edition 9) is the bulk read; using it is a move of the pin of the whole repository (its sources are `edition 5`). The in-process server needs no standard input at all: it can hand the CSV text to the same library call. Whether `ml-serve` links `ml.csv` with `Source::Mem` directly (no `io_read`) is a decision for that stage.
2. **The library boundary.** `ml-serve` should call `wireio.predict`-level code in-process. Today that function takes `Rp` paths, `Args` and `Fs`; a memory-in, memory-out entry (a CSV text and a model file's bytes in, the document out) is the missing piece and is a small refactor of `tools/ml/wireio.cho` and `verbs.cho` (the `Source::Mem` plumbing is already there).
3. **Single-row type inference.** Section 6 step 4: the reader types a column from its cells, so a one-row request must render `int` columns as integers and `dec` columns within their scale. The server can pre-check against `describe` or let the reader refuse.
4. **`describe` size.** Bounded by the file (a table of 12 MiB can list about that much); `--limit` caps the strings per column. A server would call it once at start.
5. **`ml predict` with no operand on a terminal** waits for input like `cat` does; the MCP generator should always pass `-` or a file (`docs/cli.md` section 9).
6. **Hot reload and the model digest.** `describe.file.sha256` is the digest a response should carry.

Files other branches also edit: `tools/ml/main.cho` (the shared command-line code moved to `tools/ml/app.cho`), `tools/ml/spec.tmpl` (the template of `spec.cho`), `scripts/gen_models.py`, `scripts/manifest.py`, `scripts/site.py`, `scripts/mutants_cli.py` and `scripts/mutants_cat.py` (patterns retargeted), `cancho.toml`, `tools.toml`, `tests/conformance/harness.py` (a `stdin` argument), `tests/conformance/golden_constant.json` and `golden_numeric.json`.

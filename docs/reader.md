# cancho-ml reader: float and decimal columns into the matrix, by cancho-table's exact reader

**Status: the gates (section 5) were written before any code of this stage. Sections 6 onward are filled in as the code lands; a
number or a verdict appears in them only if a script of this repository printed it, with the conditions.** Tags: **[read]** read from
a document or a source file at a named commit, **[measured]** printed by a script here, **[choice]** a decision the maintainer can
overturn, **[open]** not decided.

The question this stage answers: can `ml` read `:float` and `:dec` CSV columns exactly (correctly rounded, bit for bit) **without a
copy of cancho-table's code**, and what does cancho-table have to do for that to be a pinned dependency and not a local convenience.

## 1. The dependency

cancho-table's numeric modules are a **pinned package dependency** of `ml`: five `[dependencies.*]` in `cancho.toml` (`pow5`, `dec`, `flt`,
`facc`, `reader`), one rev, no copy. `ml` imports `flt`, `facc` and `reader`; `flt` imports `dec` and `pow5`. Sections 1.1 and 1.2 are how
that became possible (what was found, what cancho-table did); 1.3 is the pin; 1.4 is how to move it.

### 1.1 What was found (at cancho-table `7f36608`, before the stores existed)

* **The reader is merged [read].** cancho-table PR #32 ("N3b: an exact float reader without allocation (Eisel-Lemire, 3 tiers)")
  is merged (2026-10-08, merge commit `7cb2348`): `tools/table/flt.cho` (607 lines), `pow5.cho`, `dec.cho`, `facc.cho`, `reader.cho`.
  `design.md` section 3, `cli.md` section 2.3 and `cancho.toml` said it was unmerged: corrected in place (section 9).
* **A cancho dependency is a `cancho-vcs` store at a pinned commit [read, measured].** `cancho.toml` `[dependencies.NAME]` has `git`,
  `rev` (a full commit hash) and `path` (the store inside the repository: `.cancho-vcs/<module>`); `cancho install` fetches the store
  at that commit, re-parses, re-typechecks and re-hashes every declaration, writes the sources to `build/deps/<hash>.cho`, and
  `cancho build --bin` compiles them with the program (`cancho` `docs/package-system.md` sections 7 and 8; `crates/cancho/src/project.rs`).
  A dependency's own dependencies travel in the store (`requires/`). There is **no dependency form without a git origin and a
  store**.
* **cancho-table had no store at any commit [measured].** `git ls-tree -r 7f36608` has no `.cancho-vcs`; cancho-table is a binary
  repository (`[[bin]] table`). The reason was exactly "no store is committed", not "the code cannot be a package".
* **The code is a package unchanged [measured].** A draft publisher (`docs/upstream/publish_numeric.py`) published the six modules of
  `7f36608` (the five above and cancho-table's `query`) into one store per module (2, 10, 13, 41, 9 and 4 declarations), each requiring the stores of the modules it imports, into a
  scratch git repository; a scratch project with `[dependencies.flt|query|facc|reader]` ran `cancho install` (1 + 3 + 4 + 1 files: the
  closure through `requires/`), built with compiler `a4572ea` and ran `flt.parse_float` and `query.parse_dec`. The same with stores
  **published by compiler `e47d647`** (the one cancho-table is pinned to) and **consumed by `a4572ea`** (this repository's).

### 1.2 What cancho-table did

cancho-table PR #44 ("Publish the numeric modules as .cancho-vcs stores another repository can pin", merged) committed
`.cancho-vcs/{pow5,dec,flt,query,facc,reader}` (published by compiler `e47d647`) and `scripts/publish_numeric.py`, and its
`docs/numbers.md` section 1.1 lists the 15 public functions a consumer pins against: `flt.parse_float`, `flt.key_of`, `flt.float_of_key`,
`flt.put_float`, `query.parse_int`, `query.parse_dec`, `query.pow10`, `facc.width`, `facc.add_bits`, `facc.normalize`, `facc.finalize_key`,
`facc.mean_key`, `reader.scan`, `reader.fields` and `reader.split_header`; `ml` calls those of `flt`, `facc` and `reader`. A change to one of them is a change to a pinned interface, and the
stores are republished in the same commit (its words). The proposal text that asked for this is `docs/upstream/cancho-table-numeric-package.md`
(kept as the record; it was not filed by this repository).

### 1.3 The pin

cancho-table rev: `202b214372a03ec6ed8ca841357f3d2ed4de8aa0`

This is the merge commit of cancho-table PR #44. `cancho.toml` has, for each of `pow5`, `dec`, `flt`, `facc` and `reader`:

```
[dependencies.flt]
git = "https://github.com/alpibrusl/cancho-table"
rev = "202b214372a03ec6ed8ca841357f3d2ed4de8aa0"
path = ".cancho-vcs/flt"
```

the same form as the `cancho-tools` contract packages. `[[bin]] ml` lists only this repository's sources; `cancho install` fetches the
stores into `build/deps` (not committed; **it needs the network**), and the scripts that build from a file list (`build_reader.sh`,
`gate_pipe.sh`, `check.sh`'s registry test, the mutants) pass `build/deps/*.cho`. `scripts/check_table_pin.sh` (gate RD1) checks that
the five entries agree, that no other cancho-table store is listed, that the rev is the one on the line above, that the RD1 row of section 5
names the same five, that no `sources` entry leaves the tree and that no `.cho` file under `lib/`, `tools/` or `tests/` declares one of the five
modules; `CHECK_REMOTE=1` also asks GitHub whether the rev has the five stores. Nothing of cancho-table is copied here. `build/ml` built with cancho-table's `query` store also listed and built without it (the five above): SHA-256 `6c17f7e89cf8a8b2ae380d0ac3906913a080f794b7af867ec03c4da59eab9ef5` and `6c17f7e89cf8a8b2ae380d0ac3906913a080f794b7af867ec03c4da59eab9ef5`, 811,232 bytes each, and the 12 files of a seeded `fit`, `predict` and `eval` of the three models are byte-identical. Local-development arrangement removed: `table.pin`, the clone at `../cancho-table-main` and `$CANCHO_TABLE`.

### 1.4 Moving the pin

1. Choose a cancho-table commit that has the stores (`CHECK_REMOTE=1 sh scripts/check_table_pin.sh` after step 2 asks GitHub) and read what changed in
   the fifteen functions of 1.2 (its `docs/numbers.md` section 1.1 and the commit messages).
2. Change `rev` in the five blocks of `cancho.toml` and the line in 1.3 to the same 40 hex digits.
3. `rm -rf build && cancho install && cancho build --bin ml`, then `scripts/check.sh` (it runs RD1 to RD8) and `scripts/mutants_reader.py`.
4. Show that the reader's bytes did not change (or say that they did and why): the dumps of `reader-check --mode dump` and the model files of `ml fit`
   on the same inputs, before and after (section 6 says how this was done for the first pin).
5. Add a line to section 6 and commit the change as its own commit.

## 2. What is read

### 2.1 The cell and the column

A cell is read by one function (`ml.cell.read`, `tools/ml/cell.cho`) in this order:

1. empty: **missing** (section 3). 2. `nan`, `inf`, `infinity`, any case, optionally signed: refused, `ml.input.non-finite`.
3. The decimal grammar `[+-]? digits [. digits]` with at most 12 significant digits (leading zeros of the integer part not counted):
an **int** (no point) or a **dec(S)** (S digits after the point), kept as an integer mantissa and S (this is the path of the CLI slice,
unchanged). 4. Anything else goes to cancho-table's `flt.parse_float`: grammar `[+-]? digits? [. digits?] ([eE] [+-]? digits)?` with at
least one digit in the mantissa, ASCII only, no space, no separator, at most 1,100 bytes. The answer is a **float** cell, the nearest
double, ties to even (three tiers: Clinger, Eisel-Lemire, std.json's bignum reader) **[read]**. Refusals of step 4: not a number
(`ml.input.non-numeric`, reason "not a number"), past the range of a double (`ml.input.non-finite`, reason "out of range"), a non-zero
cell that would read as zero (`ml.input.non-numeric`, reason "underflows to zero"), longer than 1,100 bytes (`ml.input.non-numeric`,
reason "longer than 1,100 bytes").

The value of an int or dec cell is `float(m) / 10^S`: both operands are exact doubles (m is below 10^12, S is at most 12), so the one division
rounds once, which is the correctly rounded value of the decimal (Clinger's fast path; the same argument `flt.cho` makes). **There is no
negative zero:** `-0`, `-0.0` and `-0e5` read as `+0.0` (cancho-table's rule B2, `numbers.md`); Python reads `-0.0`, so the differential gate maps it.

A **column's type** is inferred, and only widens: **int < dec(S) < float**; S is the most digits after the point the column holds. A
column whose integer digits plus S exceed 12 becomes **float** (the CLI slice refused it with `ml.input.unsupported-type`). `ml fit
--float NAMES` forces the named columns to float (a declared type; the only one: an int or a dec column is exact when it can be, and
widening is always safe, so there is nothing to declare downward). A column with no cell yet (all missing) has kind 0.

### 2.2 The file

Comma separated only. RFC 4180 quoting through cancho-table's `reader.fields` / `reader.scan` / `reader.split_header`: a field may be
quoted, `""` inside is one quote, a quoted field may hold a delimiter or a line break; a record that spans lines is assembled in a buffer
bounded by `--max-line-bytes` (`limit.line-too-long` past it). The header is the first record that is not blank; the first line's UTF-8
byte order mark is dropped; a CR before the LF is dropped; blank lines outside a quote are skipped. A quote closing and followed by
anything but a comma or the end is `parse.csv-bad-quote`, a quote open at the end of the file `parse.csv-unterminated-quote` (both
cancho-table's tags). A numeric cell that was quoted is read from the text between the quotes (`"1.5"` is 1.5; a cell that holds a `""` is
not a number). A quoted newline's CR is not kept (irrelevant to a number). `ml.input.unsupported-csv` is removed: nothing reaches it.

### 2.3 Limits, before any allocation

`--max-rows`, `--max-columns`, `--max-cells` (rows times columns, checked by division, never a product), `--max-line-bytes` are checked
**as the file is read** by the pass that only counts and infers types (no matrix exists yet): the header's column count before the
header is split; each row's number against the row and cell caps before its cells are read. The matrix is allocated by the second
pass after `ml.limits.check_matrix` accepted the shape pass 1 found. The missing-cell bitmap is `columns * ceil(rows / 32)` words, sized by the same
counts. `ml.matrix.make` (lib/core) calls `check_matrix` itself, so no caller can allocate without the check.

## 3. The matrix and the missing bitmap

`lib/core/matrix.cho` (`ml.matrix`): a **rank-2 `linalg.tensor.Tensor` descriptor** (`matrix(0, rows, cols)`, row-major, strides `(cols, 1)`)
beside a `[float]` of `rows * cols` and a **per-column bitmap**: column `j`'s bit for row `i` is bit `i mod 32` of word
`j * ceil(rows / 32) + i / 32` (one 32-bit group per `int`, so no sign bit is touched). A missing cell holds `0.0` in the data and a 1 in the
bitmap; nothing but the bitmap says it is missing **[choice, `design.md` section 3: not NaN]**. NaN and infinity are never stored: they are refused.

`csv.load` (the second pass of `csv.scan`) fills the matrix from the file, columns being the kept (not `--drop`ped) columns in file
order. **The CLI's `--model constant` streams and builds no matrix** (it needs no feature value: 8 bytes a cell would be waste): its
cells go through the same `ml.cell.read`, and the matrix is built by `reader-check` (`tools/reader-check`) and will be by the first
model that needs one. In the CLI an empty cell is still `ml.input.non-numeric` (reason "empty"): there is no imputation step yet, and
ignoring a cell silently would be the wrong answer presented as an answer.

## 4. The CLI

`--model constant` accepts float features and a float target. For a regression whose target column is float, the model stores `n` and
the mean as the bits of a double: **the exact sum of the doubles (`facc`, 73 limbs of 32 bits, no rounding on the way), divided by `n`
and rounded once** (`facc.mean_key`). `predict` prints that double as the shortest decimal that reads back to it (`flt.put_float`),
`eval` prints `sse` = the exact `sum (y - P)^2` rounded once to a double (`lib/core/fsq.cho`, a wide exact accumulator of squared
differences: a difference of two doubles is an integer multiple of 2^-1074 of up to 2098 bits, its square an integer of 4196 bits, the
sum is exact; the rounding is half to even, subnormals included) and `rmse` = `sqrt(sse / n)` in two IEEE operations. A classification
target must still be an int column (a float target is `ml.label.not-integer`). The model file gains one column type (2, float) and one
model kind (3, constant float regression); a file with only int and dec columns is **byte-identical** to what the CLI slice wrote
(tested). `ml.metric.undefined` becomes reachable (`sse` beyond the largest double), with a fixture.

## 5. The gates (written first; each one must be able to fail)

A gate is a script or a test run by `scripts/check.sh`. "Can fail" is shown by a **mutant** (`scripts/mutants_reader.py`: a one-line change
to a source of this repository, or to the installed text of a cancho-table module in a **copy** of the tree's `build/deps`; the tree's own `build/deps` is never touched). A survivor is
listed with its reason in section 6.

| id | gate | test | mutants that must be killed |
|---|---|---|---|
| RD1 | **Pinned, not copied.** `cancho.toml` has `[dependencies.X]` for the five modules `pow5`, `dec`, `flt`, `facc` and `reader`, each `git` = cancho-table, `path` = `.cancho-vcs/X`, one `rev` of 40 hex digits, the same in all five and equal to the one documented in section 1.3; no other `[dependencies.X]` has `git` = cancho-table (a sixth store is refused); this row names the five and no other; no `sources` entry leaves this tree; no `.cho` file under `lib/`, `tools/` or `tests/` declares one of the five modules; after `cancho install` the five modules are in `build/deps`; with `CHECK_REMOTE=1` the rev has the five stores on GitHub (skipped offline). | `scripts/check_table_pin.sh` | a rev that differs in one module; a rev the docs do not document; a short rev; the path of another module; a missing dependency; a dropped module; a sixth store added; a document row that names a sixth; a `../cancho-table-main` sources entry; a module declared in `tools/ml` or in `lib/core` |
| RD2 | **Differential, zero differences.** `reader-check` reads files of generated and adversarial cells and prints every cell's 64 bits; Python's `float()` (and `Fraction` for the halfway cells) gives the expected bits; the number of differences is 0 over at least 2,000,000 cells, which hold: 17-digit and shortest-repr doubles, every power of ten from 1e-330 to 1e310 with the digits 1, 9, 17-digit neighbours, the midpoint between adjacent doubles written exactly (up to 767 digits) and its neighbours one unit of the last digit either side, subnormals (the smallest, half of it, the largest, one below normal), `2^53 + 1`, exponents (`e`, `E`, `+`, `-`, leading zeros, 20-digit exponents, huge ones), signed zeros, `.5`, `5.`, `+5`, leading and trailing zeros, ints and decs with up to 12 digits. Expected refusals (one file each, about 200: `nan`, `inf`, `infinity` in all cases and signs, `1e400`, `1e-400`, `0x10`, `1_0`, ` 1`, `1 `, `1e`, `e5`, `.`, `-`, `+`, `1.2.3`, `--1`, 1,101 bytes, an empty exponent, a NUL) have the tag and reason the spec says. | `scripts/gate_reader.py diff` | Clinger limit 22 to 23; Eisel-Lemire's sticky retry removed; ties not to even in the exact tier; a negative sign dropped; the int/dec path multiplying by the reciprocal; the mantissa digits capped at 11; `-0` kept as a negative zero; the bitmap bit written to the next column; NaN stored; `.5` refused |
| RD3 | **Matrix and bitmap.** For the same files with empty cells, the missing set printed by `reader-check` equals the set Python computed, per column and in total; a missing cell's data is `0.0`; the kept columns are the non-dropped; types inferred equal Python's (int < dec(S) < float). | `scripts/gate_reader.py matrix` | empty cell treated as 0 with no bit; bit word/shift off by one at row 31/32 and at 1 row; types not widening; `--drop` column kept |
| RD4 | **Limits, before allocation.** Each cap has a file one over it that is refused with its tag (rows, columns, cells, line, and the 1,100-byte cell), and the refusal is reached with the matrix not built: peak memory of the refusal of a 200 MB file stays under 16 MB. `ml.matrix.make` with `2^31 - 1` rows and 4,096 columns answers a refusal code and allocates nothing (a unit test). | `scripts/gate_reader.py limits`, `tests/matrix.cho` | the row check after the row is stored; the cell cap by a product that overflows; the column cap off by one; the line cap not applied to a record that spans lines |
| RD5 | **CSV shape.** CRLF, BOM, blank lines, quoted fields (a comma, a doubled quote, a line break in them), a record spanning lines, a quoted number, a quoted empty cell: the matrix equals the one Python's `csv` module gives for the same bytes, at every chunk-edge cut of a 64 KiB read; bad and unterminated quotes are refused with their tags and the right line. | `scripts/gate_reader.py csv` | CR kept in the cell; BOM kept in the first name; a quote closing mid-field accepted; the line number of a spanning record taken from its last line |
| RD6 | **No trap.** A seeded fuzz: random bytes, every byte of valid files flipped, truncated, duplicated, cells of every length up to 1,300 bytes, quotes in every position, 100,000 digits, 10^6 exponents, a 5 MB line, invalid UTF-8; `reader-check` and `ml fit/eval` never end with a trap status (132, 134, 136, 139, a negative status), and `ml` answers one valid JSON line each time. At least 5,000 runs. | `scripts/gate_reader.py fuzz` | a bound check removed in the cell reader (reading `cell[0]` of an empty cell); the field count of a record not checked before the fields are read |
| RD7 | **CLI oracle.** `fit`, `predict` and `eval` with a float target and float features agree, bit for bit, with `fractions.Fraction`: the mean (exact sum over `n`, `float()` once), `sse` (exact `sum (y - P)^2`, `float()` once) and `rmse` (`math.sqrt(sse / n)`), on seeded data whose sums leave 64 bits and whose values cancel (1e308, -1e308, 1e-320), with every summand order. The existing 88 conformance tests pass, except the ones that asserted the refusal this stage removes (listed in section 6). A model with only int and dec columns is byte-identical to the one the previous commit wrote. | `tests/conformance/test_float.py`, `scripts/check.sh` | the sum rounded at each step (a float `+=`); the mean rounded to 6 digits; the square of the rounded difference; the mean divided by n - 1; a difference rounded before squaring; `float` column compatible with `int` model |
| RD8 | **Tables.** The new flag (`--float`), the new rules (`parse.csv-bad-quote`, `parse.csv-unterminated-quote`, `ml.metric.undefined`) and the removed ones (`ml.input.unsupported-type`, `ml.input.unsupported-csv`) are in `spec.cho` once and in `introspect`, `skill`, the schemas and the fixtures, as gate T1 of `cli.md` already checks. | `test_tables.py` | as T1 |
| RD9 | **Speed measured, not claimed.** `scripts/bench_reader.py`: cells per second of `reader-check --hash` (pass 1 and the whole load) and of `table` on the same file, one thread, min of 7 interleaved, load recorded at the start and the end of every timing. | `scripts/bench_reader.py` | n/a (a measurement, no pass/fail threshold) |

Nothing in a gate is "expected to pass": the mutants above are each a row of `scripts/mutants_reader.py`, and section 6 reports the verdict of each,
with the survivors. A mutant of a cancho-table module is an edit of the installed text of the module in a **copy** of `build/deps`, never of `build/deps` itself.

## 6. Results

Everything below was printed by a script of this repository (`scripts/check.sh` exits 0; the logs are in `docs/bench/`), on one machine: this
Mac, arm64, macOS, compiler `a4572ea`. The gate figures below were last printed after the switch to the pinned package (cancho-table `202b214`, section 6.1); the speed figures (RD9) predate it (section 6.1). The machine was **not quiet**: the 1-minute load average was 3 to 12
during the gates (other processes), 3.6 to 4.8 during the speed run, and it reached 47 at the end of the last fuzz. Nothing here is a
benchmark of anything but the binaries named.

| gate | result |
|---|---|
| RD1 pin | ok; 11 mutants (a rev that differs in one module, a rev the docs do not document, a short rev, the path of another module, a missing dependency, a dropped dependency, a sixth store added, a document row that names a sixth, one that leaves a module out, a `../` sources entry, a module declared in `tools/ml`, one in `lib/core`) all caught |
| RD2 differential | **1,966,143 accepted cells compared bit for bit with Python `float()` and 34,380 refusals compared (status, aux): 0 differences** (seed 1) |
| RD3 matrix | 192 random files (missing cells, `--drop`, `--float`, rows 1 to 200 across the 32-row word edges), 28,518 cells, values, missing bits, kinds, hash; 216 kind/scale checks: 0 differences |
| RD4 limits | 36 checks, 0 differences. A 200 MB file refused at its 11th row, at the cell cap, and at `matrix.make` peaks at 1.5 MB (16 MB allowed); read whole into the matrix the same file peaks at 414 MB, so an allocation before the check would show |
| RD5 csv | 277 files, 11,331 numeric cells against Python's `csv` module, 17 refusal cases (lines of errors in records that span lines): 0 differences |
| RD6 fuzz | 5,014 inputs through `reader-check`, 5,115 runs of `ml`; 14 large cases (a 100,000-digit cell, a 10^6 exponent, a 5 MB line, a 5 MB header, a million blank lines...): **0 traps, 0 malformed answers** |
| RD7 CLI oracle | `tests/conformance/test_float.py`, 11 tests (240 seeded datasets, exact ties in the mean and in `sse`, row-order independence): all pass |
| conformance suite | 101 tests, all pass, 1 skipped (the `--threads` branch of D1). Of the 88 of the CLI slice, **2 were rewritten because they asserted the refusal this stage removes** (`test_float_text_is_refused_not_approximated`, `test_the_hint_names_the_missing_dependency`); the other 86 are untouched |

**Mutants.** `scripts/mutants_reader.py`: 40 on the reader gates + 11 on the pin, `docs/bench/mutants_reader.log`: **36 killed, 4 survivors, all
explained**: `rd2-nan-one-layer` (nan and inf are refused twice, by `ml.csvcell` and by `flt`: equivalent), `rd4-cell-cap-by-product` (every cap is at most
2^31 - 1 so `rows * cols` cannot overflow: equivalent), `rd5-blank-line-in-quote-skipped` (the text of a quoted field is never read as a number: the only
observable is the byte length against the cap), `rd6-fields-count-ignored` (`reader.scan` has counted the same fields: equivalent). Mutants of a
cancho-table module are made in the installed text of the module in a copy of `build/deps` (Clinger limit 23, mantissa 2^53 + 1, the sticky retry, the halfway test, truncation, the sign, signed
zero, subnormals; the bad-quote flag of `reader`). `scripts/mutants_cli.py` (the CLI slice's 34 and 11 new for RD7): **45 killed, 0 survived**,
`docs/bench/mutants_cli_reader.log`. **What the mutants found about the gates**: five survived the first run and each was a gate that checked less
than it said: the differential generated no empty cell and too few mantissas of 2^53 + 1 (now a family of 610 and the empty cell); the matrix gate
compared kinds only on columns that were always float (now 216 single-cell and ordered-column checks); the allocation of the limits mutant was
optimised away by the compiler (the mutant now uses it), and the model-file test had no float model (now it has one). One kill is a trap, not a
value (`rd2-clinger-limit-23` indexes past a 23-entry table): it is still a kill.

**Found about the dependency while doing it.** `table`'s `main` cannot be built by `a4572ea` (`bulk.cho` is `edition 9`; `a4572ea` knows 1 to 7), but
its numeric modules (editions 5 and 6 features) compile and run under both compilers; the reader-check built by `e47d647` is the same size and speed
as the `a4572ea` one (the speed log has both).

**Speed (RD9)** (measured before the switch to the pinned package, section 6.1; not re-run), `docs/bench/bench_reader.log`, 500,000 rows x 8 columns (4.0 M cells), one thread, minimum of 7 interleaved rounds, load 3.6 to 4.8.
`table` is the binary of cancho-table `main` built by its own pin `e47d647`. Cells per second = cells scanned / seconds:

| file | `reader-check` pass 1 (every cell read, none kept) | `reader-check` whole load (2 passes + matrix) | `table --agg max:c0..c7:float` | `table --agg max:c0:float` | `table --agg count` |
|---|---|---|---|---|---|
| float17 (94 MB, `%.16e`) | 17.1 M/s | 7.6 M/s | 19.6 M/s | 51.6 M/s (6.5 M parsed/s) | 67.3 M/s |
| short (30 MB, `%.2f`, the int/dec path) | 41.8 M/s | 19.9 M/s | 40.7 M/s | 91.4 M/s (11.4 M parsed/s) | 104.7 M/s |
| mixed (62 MB) | 22.2 M/s | 9.9 M/s | 23.4 M/s | 58.1 M/s (7.3 M parsed/s) | 71.5 M/s |

So, on this file and machine, reading every cell of a float file in pass 1 is **13% slower than cancho-table's** (17.1 against 19.6) and on short
decimals equal; the whole load is about 2.3 times a pass (it reads every cell twice, by design: the shape must be known before the matrix is
allocated). Where the 13% goes was not profiled: ml copies each record into a buffer and searches it for a quote and a comma count before
splitting, which table's fused reader does not (docs/readers.md of cancho-table); that is a hypothesis, not a measurement.

### 6.1 The switch from the local clone to the pinned package (cancho-table `7f36608` to `202b214`)

* **What changed in the code compiled.** Between the two commits `pow5.cho`, `dec.cho`, `flt.cho`, `facc.cho` and `reader.cho` of cancho-table are
  unchanged (`git diff --stat 7f36608 202b214 -- tools/table/<module>.cho` is empty for each); `query.cho` differs by 34 insertions and 21 deletions, all
  of them a new field `sort` of the `Query` struct (the `--sort` flag of `table`); `ml` imports nothing from
  `query`. The stores are the re-rendered text of those files (published by compiler `e47d647`), not the files.
* **Identical bytes [measured].** Before the switch (clone at `7f36608`, `build/ml` and `build/reader-check` built from it) and after (stores at `202b214`),
  on the same inputs, `shasum -a 256` of 20 files is the same list: `reader-check --mode dump`, `--mode hash` and `--mode shape` (with `--float a,b,y`
  and without) of three seeded files (`mixed`: 20,000 rows, floats of every spelling, subnormals, 2^53 + 1, empty cells; `floaty` and `decs`: 5,000 rows each),
  plus the model files of `ml fit --target y` (as a float regression on `floaty` and on `decs`, and as a decimal regression on `decs`) and the answers of `fit` and `eval`.
  The hash of the list of hashes is `35bd3f5b0b3b1f3994e0e3e7770ee556d79d0ec6e47e1042392a5dfc6167286c` in both runs. The two `build/ml`
  binaries themselves differ (as files); only what they print and write was compared. The scripts were throwaway, not committed.
* **Every gate the same [measured].** `scripts/check.sh` before (the old tree, in a scratch copy with the sibling clone) and after: the same
  figures line for line: RD2 1,966,143 accepted cells and 34,380 refusals, 0 differences; RD3, RD4, RD5, RD6 0 differences; 105 conformance tests,
  1 skipped; 17 Cancho test programs ok; every `--check` of the mutant lists applies (`mutants_reader` 40, `mutants_cli` 53, `mutants_paths` 19, `mutants_pipe` 73, ...).
  The RD1 line changed from `6 sources at 7f36608` to `6 dependencies at 202b214`.
* **Mutant suites run after the switch.** `mutants_reader` 40 reader mutants: 36 killed, 4 survive and are explained (the same four as before), plus the 8 RD1 mutants (4 before) all killed; `mutants_cli` 53 killed of 53; `mutants_paths` 19 killed of 19; `mutants_pipe` 72 killed, 1 survives and is explained. Their lists are the same as before the switch except the RD1 ones.
* **`cancho install`.** It needs the network: it fetches `cancho-tools` and `cancho-table` at their revs. With the six dependencies of that
  time it printed 15 `installed` lines and took 1.4 seconds (wall, once, after the cancho-tools stores had been fetched by an earlier call); 17 files in `build/deps`
  (6,907 lines). With the five that `cancho.toml` has now, `build/deps` holds 16 files.
* **Speed.** RD9 was not re-run: the machine was not quiet (1-minute load 7 to 9 during this work, other agents building). The numbers of
  section 6 predate the switch. They should not move (the reader modules are unchanged), but that is an expectation, not a measurement.

## 7. Open questions

1. **CI**: the file list no longer needs a sibling clone, but this repository has no workflow yet; a CI run would need the network for `cancho install`.
2. **The CLI builds no matrix for the constant model** *(answered at W1 for the other models: `linear` and `logistic` read the file with the loader, `tools/ml/csvload.cho` `load_t`; mode 9 of the scan skips a target column that predict does not need)*.
3. **Missing cells.** The loader records them; the constant model still refuses an empty cell. *(At W1 the pipeline models take it as a missing value that the plan must `impute`; if it is not imputed the fit is refused with `ml.input.missing-value` and a repair.)*
   One consequence of "a blank line is not a record": in a one-column file an empty cell must be written `""`.
4. **`-0` reads as `0.0`** (cancho-table's rule B2). A model that wants the sign of zero needs another reader.
5. **A float target's `mean` is a double, an int or decimal target's is a decimal with six more digits.** `--float` on an int target therefore changes the
   output form. Is that wanted, or should every regression be a double?
6. **Compiler pins differ**: cancho-ml `0207556`, cancho-table `e47d647`. The committed stores (published by `e47d647`) install and build under `0207556` (measured with `cancho install` and `cancho build --bin ml`; section 6 measured the same under `a4572ea`), but a single
   pin would remove the question.
7. **`describe`'s `integers_only: true`** is a constant of the pinned contract package. It is true of the JSON `ml` prints (every decimal is a string) and
   not of its code, which now reads and sums floats.
8. **Speed**: the 13% on float cells (section 6) and the second pass of the loader; a loader that grows its matrix in one pass would trade the
   exact-size allocation and the early refusal for one parse per cell.
9. **D2 on a second machine** is still untested (no float operation in the model path but `float_of`, one division, `ldexp` and `sqrt`).

## 9. Statements corrected in place

`design.md` sections 3, 10 and 12, `cli.md` sections 2.3 and 10, `cancho.toml` and the `unsupported-type` text of `spec.cho` called
cancho-table's float reader unmerged; it is merged (PR #32). Each now says what section 1 says.

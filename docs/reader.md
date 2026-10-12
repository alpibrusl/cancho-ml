# cancho-ml reader: float and decimal columns into the matrix, by cancho-table's exact reader

**Status: the gates (section 5) were written before any code of this stage. Sections 6 onward are filled in as the code lands; a
number or a verdict appears in them only if a script of this repository printed it, with the conditions.** Tags: **[read]** read from
a document or a source file at a named commit, **[measured]** printed by a script here, **[choice]** a decision the maintainer can
overturn, **[open]** not decided.

The question this stage answers: can `ml` read `:float` and `:dec` CSV columns exactly (correctly rounded, bit for bit) **without a
copy of cancho-table's code**, and what does cancho-table have to do for that to be a pinned dependency and not a local convenience.

## 1. The dependency

### 1.1 What was found

* **The reader is merged [read].** cancho-table PR #32 ("N3b: an exact float reader without allocation (Eisel-Lemire, 3 tiers)")
  is merged (2026-10-08, merge commit `7cb2348`); `main` at `7f36608` has `tools/table/flt.cho` (607 lines), `pow5.cho`, `dec.cho`,
  `facc.cho`, `reader.cho`, and `docs/numbers.md` / `docs/readers.md`. `design.md` section 3, `cli.md` section 2.3 and
  `cancho.toml` said it was unmerged: corrected in place (section 9).
* **A cancho dependency is a `cancho-vcs` store at a pinned commit [read, measured].** `cancho.toml` `[dependencies.NAME]` has `git`,
  `rev` (a full commit hash) and `path` (the store inside the repository: `.cancho-vcs/<module>`); `cancho install` fetches the store
  at that commit, re-parses, re-typechecks and re-hashes every declaration, writes the sources to `build/deps/<hash>.cho`, and
  `cancho build --bin` compiles them with the program (`cancho` `docs/package-system.md` sections 7 and 8; `crates/cancho/src/project.rs`).
  A dependency's own dependencies travel in the store (`requires/`). There is **no dependency form without a git origin and a
  store**.
* **cancho-table has no store at any commit [measured].** `git ls-tree -r 7f36608` has no `.cancho-vcs`; `cancho-table` is a binary
  repository (`[[bin]] table`, `sources = ["tools/table", "generated/table"]`). So **the git form cannot be used today**, and the
  reason is exactly "no store is committed", not "the code cannot be a package".
* **The code can be a package unchanged [measured].** `scripts` of this stage (`docs/upstream/publish_numeric.py`, modelled on the
  compiler's `scripts/publish_packages.py`) publish `pow5`, `dec`, `flt`, `query`, `facc` and `reader` of the pinned clone into
  one store per module (2, 10, 13, 41, 9 and 4 declarations), each requiring the stores of the modules it
  imports, into a scratch git repository outside both projects (500 KB). A scratch project with
  `[dependencies.flt|query|facc|reader] git = <that repository> rev = <its commit> path = .cancho-vcs/<module>` ran `cancho install` (it
  fetched 1 + 3 + 4 + 1 files: the closure through `requires/`), built with compiler `a4572ea` and ran: `flt.parse_float`,
  `query.parse_dec` return the expected values. The same with the stores **published by compiler `e47d647`** (the one cancho-table
  is pinned to) and **consumed by `a4572ea`** (this repository's): it installs, builds and runs. No source file of cancho-table was moved or
  edited. (A scratch repository proves the mechanism; it is not a pin anyone else can fetch.)

### 1.2 What cancho-table must do (the minimal proposal; not filed, not pushed)

Nothing in `tools/table/*.cho` changes. Three additions to the repository, in one pull request:

1. `scripts/publish_numeric.py` (a draft is `docs/upstream/publish_numeric.py` here): publishes the six modules into
   `.cancho-vcs/{pow5,dec,flt,query,facc,reader}` at the repository root, one store per module, requirements derived from the `import` lines.
2. The six committed stores (about 500 KB), and `scripts/publish_numeric.py --check` in CI: a fresh publish must equal the committed
   stores byte for byte (publishing is deterministic, `package-system.md` 7.4), so a source edited without its store fails CI.
3. A sentence in `docs/numbers.md`: the stores are part of the interface, and a change of a signature of `flt.parse_float`,
   `flt.float_of_key`, `flt.key_of`, `flt.put_float`, `query.parse_dec`, `query.parse_int`, `facc.add_bits`, `facc.mean_key`,
   `reader.scan/fields/split_header` is a change a consumer pins against.

`query` is the module that holds `parse_int`, `parse_dec` and `pow10` and also 38 other declarations (the `--where` grammar's
values); a smaller `num` module with only the three would make the dependency smaller. That is an optimisation, not a requirement,
and it is not asked for. Then `cancho add flt https://github.com/alpibrusl/cancho-table --rev <commit> --path .cancho-vcs/flt`
(and the other three) is the whole consumer side. The note and the issue text are `docs/upstream/cancho-table-numeric-package.md`.

### 1.3 What `ml` does until then (LOCAL-DEV fallback, marked as such)

`cancho.toml`'s `[[bin]] ml` (and the other binaries that read cells) list the **files** of a read-only clone of cancho-table as
extra sources: `../cancho-table-main/tools/table/{pow5,dec,flt,query,facc,reader}.cho`. Nothing is copied into this repository.
This is pinned by `table.pin` (the commit and the SHA-256 of each of the six files) and checked by `scripts/check_table_pin.sh`,
which `scripts/check.sh` runs first: a clone at another commit, a dirty clone, or a source whose hash differs fails the check. The
check also fails if any file under `lib/`, `tools/` or `tests/` declares one of the six module names, or has the bytes of one of
them (the anti-vendoring gate, RD1). The clone is `../cancho-table-main` or `$CANCHO_TABLE`. **This cannot be CI** (CI has no
sibling clone); it is a development arrangement whose end is section 1.2.

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
to a source of this repository, or to a **scratch copy** of a cancho-table module, the clone itself is never touched). A survivor is
listed with its reason in section 6.

| id | gate | test | mutants that must be killed |
|---|---|---|---|
| RD1 | **Pinned, not copied.** The clone is at the commit of `table.pin`, clean, and each of the six sources has the pinned SHA-256; no file of this repository declares one of the six modules or contains one's bytes. | `scripts/check_table_pin.sh` | pin file with another hash; a pin at another commit; `tools/ml/flt.cho` a copy of the module |
| RD2 | **Differential, zero differences.** `reader-check` reads files of generated and adversarial cells and prints every cell's 64 bits; Python's `float()` (and `Fraction` for the halfway cells) gives the expected bits; the number of differences is 0 over at least 2,000,000 cells, which hold: 17-digit and shortest-repr doubles, every power of ten from 1e-330 to 1e310 with the digits 1, 9, 17-digit neighbours, the midpoint between adjacent doubles written exactly (up to 767 digits) and its neighbours one unit of the last digit either side, subnormals (the smallest, half of it, the largest, one below normal), `2^53 + 1`, exponents (`e`, `E`, `+`, `-`, leading zeros, 20-digit exponents, huge ones), signed zeros, `.5`, `5.`, `+5`, leading and trailing zeros, ints and decs with up to 12 digits. Expected refusals (one file each, about 200: `nan`, `inf`, `infinity` in all cases and signs, `1e400`, `1e-400`, `0x10`, `1_0`, ` 1`, `1 `, `1e`, `e5`, `.`, `-`, `+`, `1.2.3`, `--1`, 1,101 bytes, an empty exponent, a NUL) have the tag and reason the spec says. | `scripts/gate_reader.py diff` | Clinger limit 22 to 23; Eisel-Lemire's sticky retry removed; ties not to even in the exact tier; a negative sign dropped; the int/dec path multiplying by the reciprocal; the mantissa digits capped at 11; `-0` kept as a negative zero; the bitmap bit written to the next column; NaN stored; `.5` refused |
| RD3 | **Matrix and bitmap.** For the same files with empty cells, the missing set printed by `reader-check` equals the set Python computed, per column and in total; a missing cell's data is `0.0`; the kept columns are the non-dropped; types inferred equal Python's (int < dec(S) < float). | `scripts/gate_reader.py matrix` | empty cell treated as 0 with no bit; bit word/shift off by one at row 31/32 and at 1 row; types not widening; `--drop` column kept |
| RD4 | **Limits, before allocation.** Each cap has a file one over it that is refused with its tag (rows, columns, cells, line, and the 1,100-byte cell), and the refusal is reached with the matrix not built: peak memory of the refusal of a 200 MB file stays under 16 MB. `ml.matrix.make` with `2^31 - 1` rows and 4,096 columns answers a refusal code and allocates nothing (a unit test). | `scripts/gate_reader.py limits`, `tests/matrix.cho` | the row check after the row is stored; the cell cap by a product that overflows; the column cap off by one; the line cap not applied to a record that spans lines |
| RD5 | **CSV shape.** CRLF, BOM, blank lines, quoted fields (a comma, a doubled quote, a line break in them), a record spanning lines, a quoted number, a quoted empty cell: the matrix equals the one Python's `csv` module gives for the same bytes, at every chunk-edge cut of a 64 KiB read; bad and unterminated quotes are refused with their tags and the right line. | `scripts/gate_reader.py csv` | CR kept in the cell; BOM kept in the first name; a quote closing mid-field accepted; the line number of a spanning record taken from its last line |
| RD6 | **No trap.** A seeded fuzz: random bytes, every byte of valid files flipped, truncated, duplicated, cells of every length up to 1,300 bytes, quotes in every position, 100,000 digits, 10^6 exponents, a 5 MB line, invalid UTF-8; `reader-check` and `ml fit/eval` never end with a trap status (132, 134, 136, 139, a negative status), and `ml` answers one valid JSON line each time. At least 5,000 runs. | `scripts/gate_reader.py fuzz` | a bound check removed in the cell reader (reading `cell[0]` of an empty cell); the field count of a record not checked before the fields are read |
| RD7 | **CLI oracle.** `fit`, `predict` and `eval` with a float target and float features agree, bit for bit, with `fractions.Fraction`: the mean (exact sum over `n`, `float()` once), `sse` (exact `sum (y - P)^2`, `float()` once) and `rmse` (`math.sqrt(sse / n)`), on seeded data whose sums leave 64 bits and whose values cancel (1e308, -1e308, 1e-320), with every summand order. The existing 88 conformance tests pass, except the ones that asserted the refusal this stage removes (listed in section 6). A model with only int and dec columns is byte-identical to the one the previous commit wrote. | `tests/conformance/test_float.py`, `scripts/check.sh` | the sum rounded at each step (a float `+=`); the mean rounded to 6 digits; the square of the rounded difference; the mean divided by n - 1; a difference rounded before squaring; `float` column compatible with `int` model |
| RD8 | **Tables.** The new flag (`--float`), the new rules (`parse.csv-bad-quote`, `parse.csv-unterminated-quote`, `ml.metric.undefined`) and the removed ones (`ml.input.unsupported-type`, `ml.input.unsupported-csv`) are in `spec.cho` once and in `introspect`, `skill`, the schemas and the fixtures, as gate T1 of `cli.md` already checks. | `test_tables.py` | as T1 |
| RD9 | **Speed measured, not claimed.** `scripts/bench_reader.py`: cells per second of `reader-check --hash` (pass 1 and the whole load) and of `table` on the same file, one thread, min of 7 interleaved, load recorded at the start and the end of every timing. | `scripts/bench_reader.py` | n/a (a measurement, no pass/fail threshold) |

Nothing in a gate is "expected to pass": the mutants above are each a row of `scripts/mutants_reader.py`, and section 6 reports the verdict of each,
with the survivors. A mutant of a cancho-table module is a **copy in a scratch directory** with one edit, never an edit of the clone.

## 6. Results

*(Filled in as the stage lands; see the end of the commit history of this file.)*

## 7. Open questions

*(Filled in at the end.)*

## 9. Statements corrected in place

`design.md` sections 3, 10 and 12, `cli.md` sections 2.3 and 10, `cancho.toml` and the `unsupported-type` text of `spec.cho` called
cancho-table's float reader unmerged; it is merged (PR #32). Each now says what section 1 says.

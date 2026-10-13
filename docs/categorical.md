# cancho-ml categorical columns: the reader, the matrix and the limits (stage 1 of issue 21)

**Status: sections 1 to 8 (the gates, the declaration, the representation, the limits, the refusals, the cell rules, the loader, the command line) were
written before any code of this stage. Section 9 holds the results; a number or a verdict is in it only if a script of this repository printed it, with
the conditions.** Tags as in `design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from a document or the code, **[measured]**
printed by a script here, **[open]** not decided. The decisions of the issue (alpibrusl/cancho-ml#21) are binding and are quoted where they apply.

Stage 1 of four. This stage reads **declared** text columns into a bounded dictionary and an integer id matrix, with limits and refusals. It does **not**
encode anything (stage 2), store anything in a model file (stage 3) or split on categories (stage 4). Until stage 2 a declared categorical column is read,
checked and then refused by `ml fit` with `ml.input.categorical-not-encoded` (section 8); the dictionary and the ids are reached through the library, the
loader API (section 7) and `reader-check`, which is how the gates see them.

## 1. The gates (fixed before the code; each can fail, section 9 shows it failing)

A gate is a script or a test run by `scripts/check.sh`. "Can fail" is shown by a **mutant** (`scripts/mutants_cat.py`: a one-line change to a source of this
repository, built and run in a copy of the tree). A survivor is listed with its reason in section 9.

| id | gate | judged by | a mutant it must kill |
|---|---|---|---|
| C1 | **Vocabulary and ids against Python.** `reader-check --mode cat` prints, for each categorical column, the vocabulary (count, then each category as hex of its bytes) and the id of every cell; `scripts/oracle_cat.py` implements section 6 independently (its own RFC 4180 reading of the same bytes, its own unescaping, bytewise sort by Python `bytes` order, its own limit and UTF-8 rules) and the two agree on every file: vocabulary bytes, ids, missing set, and for a refused file the status, the line and the column. Files: seeded random tables (1 to 6 categorical columns, 0 to 300 rows, cardinality 1 to 70) and the adversarial family of C5 to C7 | `scripts/gate_cat.py diff` | first-appearance order; descending order; the sort key lower-cased; the sort key trimmed; ids taken from the position in the file |
| C2 | **Row order.** For each file, five permutations of its data rows (seeded) give the same vocabulary bytes, and a row's id is the same whatever its position (the id of row `i` in the permuted file equals the id of the same row in the original); an accepted file is accepted in every order and a refused file is refused in every order (the first violation, so the line and possibly which limit's tag, may differ: section 5.2) | `scripts/gate_cat.py order` | **the dictionary ordered by first appearance must fail this gate** (`c2-first-appearance`); the id of a row taken from a hash of its position |
| C3 | **Thread count.** `ml fit --cat ... --threads T` for T in 1, 2, 4, 8, 16 writes the same standard output, byte for byte, including the digest of the vocabulary in the refusal's `detail` (section 8), and `reader-check` dumps of the same file are byte-identical on 5 repeated runs. The reader is single-threaded; this gate says that no later stage of the verb makes the dictionary depend on `--threads` | `scripts/gate_cat.py threads` | the thread count mixed into the digest; a dictionary built per thread |
| C4 | **Limits, each refused before allocation.** For each of the five limits (categories per column, bytes per category, total pool bytes, categorical columns, and the shape caps with ids counted) a file one over is refused with its tag, a file at the limit is accepted, and the refusal is reached with the id matrix not built: the peak memory of the refusal of a 200 MB file stays under 16 MB (`/usr/bin/time -l`, as RD4); `catmat.make` with `2^31 - 1` rows and 4,096 columns answers a code and allocates nothing (a unit test) | `scripts/gate_cat.py limits`, `tests/cat.cho` | each limit off by one (`>` for `>=`); the shape check after the allocation; the cell count by a product |
| C5 | **UTF-8.** Valid: every boundary (U+007F, U+0080, U+07FF, U+0800, U+FFFF, U+10000, U+10FFFF), emoji, a lone NUL, a BOM inside a cell, the same text in composed and decomposed form (two categories). Refused with `ml.input.invalid-utf8`, the line, the column and the byte offset: a lone continuation byte, a truncated sequence at the end of the cell, an overlong form (`C0 80`, `E0 80 80`), a surrogate (`ED A0 80`), a code point past U+10FFFF (`F4 90 80 80`, `F5`), `FF`, Latin-1 text. Python's `bytes.decode("utf-8")` is the oracle of validity and of the offset | `scripts/gate_cat.py utf8` | no check; the first byte only; surrogates accepted; the check on the pool and not on the cell |
| C6 | **Missing.** An empty cell, a quoted empty cell and a cell equal to a marker of `--na-values` are missing (bit set, id -1) and are not categories; the marker is compared with the whole cell, bytewise, after unquoting; `NA` is a category when it is not a marker; a cell of one space is a category (leading and trailing spaces are kept); a column with no non-missing cell has an empty vocabulary | `scripts/gate_cat.py diff` | an empty cell as a category; a marker matched by prefix; a marker matched without regard to case; a cell trimmed before the comparison |
| C7 | **Round trip.** `reader-check --mode cat --predict FILE2` builds the vocabulary from `FILE`, then reads `FILE2` through the **stored** vocabulary: ids equal Python's lookups, a string the vocabulary does not hold is id -2 and is counted per column, the file may hold any rows and the columns in another order; a vocabulary column absent from `FILE2` is `ml.schema.mismatch`; `FILE` read through its own vocabulary gives the ids of the fit | `scripts/gate_cat.py roundtrip`, `tests/cat.cho` | predict adding unseen strings to the vocabulary; an unseen string mapped to the nearest id |
| C8 | **No trap.** A seeded fuzz of `reader-check --mode cat` and of `ml fit --cat`: random bytes, every byte of valid files flipped, truncated and duplicated, quotes in every position, cells of 65,536 and 70,000 bytes, 2^20 distinct values in one column (a refusal, not an allocation), invalid UTF-8 in every column, 4,097 categorical columns; at least 3,000 runs; no trap status (132, 134, 136, 139, a negative status) and `ml` answers one valid JSON line each time | `scripts/gate_cat.py fuzz` | a bound removed in the dictionary (an index past the slots of a column) |
| C9 | **Numeric inputs do not change.** `tests/conformance/golden_numeric.json` was written by the binary of `v1-next` before this stage (`7d6183d`, `scripts/golden_numeric.py`): for each of the 161 fixtures that exist before this stage (the constant model, `linear`, `logistic` with imputed, scaled, bucketed and one-hot steps, and every refusal of the reader) the exit status, the SHA-256 of standard output and of every file in the sandbox are the same. W1 (`golden_constant.json`) still holds | `test_cat_baseline.py`, `test_wire_baseline.py` | a changed default that reaches a numeric plan; a numeric column read as categorical |
| C10 | **Tables.** The new flags and rules are in `spec.cho` once and in `introspect`, `skill`, the schemas and the fixtures, as T1 of `cli.md` checks | `test_tables.py` | as T1 |
| C11 | **Declared, not inferred.** A text column that is not declared is `ml.input.non-numeric` with a `retry --drop` repair (unchanged bytes, C9); `--cat` on the target, on a dropped column or on a `--float` column is `args.conflict`; `--cat` names a column that is not there: `ml.input.unknown-column`; a declared column no step encodes is `ml.input.categorical-not-encoded` | `tests/conformance/test_cat.py` | the inference of a text column; `--cat` on the target accepted |
| C12 | **The structure.** Unit tests of the dictionary, the frozen vocabulary and the id matrix (`tests/cat.cho`): sorted and strictly ascending, ids and counts consistent, the missing bitmap and the id -1 agree, every refusal code has a tag and a fixture, the registries (`tests/t_registry.cho`, `scripts/check_registry.py`) | `tests/cat.cho`, `tests/t_registry.cho` | the freeze not sorting; a bitmap bit without the id -1 |

Nothing in a gate is "expected to pass": the mutants are the rows of `scripts/mutants_cat.py`, and section 9 reports the verdict of each, with the survivors.

## 2. Declaring a column

`ml` reads a CSV column as int, dec or float and **infers** which, widening only (`docs/reader.md` section 2.1); `--float NAMES` is the one declaration
and only forces a type that inference would not choose. A text column has no numeric reading, so it can never be inferred from the cells: **it is
declared, or it is refused.** The same mechanism is extended: a flag that takes header names separated by commas.

| flag | verbs | meaning |
|---|---|---|
| `--cat NAMES` | fit | columns to read as **categorical text**: every cell is a string, bytewise; the column is not a column of the float matrix. NAMES are header names, as `--drop` and `--float`. A name twice is the same column. The target cannot be named (`args.conflict`; text targets are a later stage), nor a dropped column or a `--float` column (`args.conflict`) |
| `--na-values LIST` | fit | with `--cat`: strings (separated by commas, so a marker has no comma) that mean "missing" in a categorical column, in addition to the empty cell. At most 64 markers of 1 to 255 bytes. Compared with the whole cell, bytewise (`NA` is not `na`, ` NA` is not `NA`) |
| `--max-categories N` | fit | with `--cat`: the most distinct non-missing values of one categorical column, 1 to 1,024. Default 64 |
| `--max-string-bytes N` | fit | with `--cat`: the longest cell of a categorical column, in bytes, 1 to 65,536. Default 255 |
| `--max-pool-bytes N` | fit | with `--cat`: the most bytes of all the categories of all the columns together, 1 to 8,388,608. Default 1,048,576 |
| `--max-cat-columns N` | fit | with `--cat`: the most categorical columns, 1 to 4,096. Default 128 |

`--na-values` and the four limits without `--cat` are `args.conflict` (a limit that governs nothing is a mistake the caller should hear about). They are **fit
flags**: `predict` and `eval` take none of them. The reason is the same for all: what a model needs to read a text column again (which columns, which
strings, which markers) is stored with it from stage 3, and a flag on `predict` would be a second source of truth that could differ from the fit's
(`docs/wiring.md` W2 for the same rule about plans). Until stage 3 no model holds a categorical column, and `predict`/`eval` of a file with a text column are
refused as before (`ml.input.non-numeric`, whose repair is `--drop`).

**Why not a type suffix in the header or a `NAME:cat` spelling [choice].** cancho-table types a column in its `--agg` and `--where` operands (`c0:float`).
`ml` has never typed columns in a header or in an operand; it names them in a flag (`--float`, `--drop`). A new spelling would be a second mechanism for
one idea, and a header the file's author wrote is not ours to edit. `--cat` is `--float`'s sibling.

**What is not done: no inference [read, the issue].** A column whose cells are all non-numeric is not guessed to be categorical. It is `ml.input.non-numeric`,
at the first cell, with the existing repair (`--drop NAME`); the summary of the rule in `introspect` names `--cat` as the other way out. (The text of the
refusal itself is unchanged on purpose: W1 and C9 pin its bytes.) A column of numbers that the user wants as categories (zip codes, `1/2/3` classes)
is declared with `--cat` too: the cells are then read as strings (`007` and `7` are two categories) and never as numbers.

**An empty cell [read, the issue; choice for the "unless declared otherwise"].** An empty cell is missing, not a category. The issue allows declaring
otherwise; this stage does not: a categorical `none` is a string the file spells (the `NA` of House Prices stays a category when it is not a marker), or it is
made by stage 2's `impute strategy=constant` on a categorical column. Both are explicit, and neither changes what an empty cell means.

## 3. The representation

A categorical column is **not a column of the float matrix.** The float matrix (`ml.matrix`, `docs/reader.md` section 3) holds the numeric kept columns,
unchanged, in file order. Beside it:

* **The vocabulary** (`ml.dict.Vocab`): per categorical column, in **file order of the header**, the distinct non-missing strings, **sorted bytewise
  ascending** (the order of `std.bytes.compare`: unsigned byte by byte, a prefix before its extensions). It does not depend on the row order, the
  thread count, the order of the names in `--cat`, or the hash of anything: there is no hash. The category with rank `i` has id `i`.
* **The ids** (`ml.catmat.Ids`): a **rank-2 `i64` tensor** (`linalg.tensor` descriptor, row-major, strides `(ncat, 1)`) of `rows * ncat` integers, column `j`
  holding the ids of the `j`-th categorical column. An id is `0 .. k-1` for a category, **`-1` for a missing cell**, and **`-2` for a string the
  vocabulary does not hold** (only when the vocabulary is given, section 7; never at fit).
* **The missing bitmap** (the same layout as the float matrix's: column `j`'s bit for row `i` is bit `i mod 32` of word `j * ceil(rows / 32) + i / 32`).
  A bit is set exactly when the id is -1; `ml.catmat` keeps both and a unit test (C12) says they agree. The id -1 is redundant on purpose: a consumer that
  forgets the bitmap meets a negative index and traps in a test, not a valid category.

The shape's kind of a categorical column is **4** (`csv.kind`; 0 none, 1 int, 2 dec, 3 float; the 4 the reader reserved, `docs/wiring.md` section 3). The schema type
code 3, reserved by the same section, is the file's code for it and stage 3 writes it; this stage writes no file. Both numbers are in `lib/core/kinds.cho`
(`column_*`) so two stages cannot claim them.

**Interning is a sorted array, not a hash [choice].** A column's dictionary during the first pass is a sorted array of at most `--max-categories` entries
(offset and length in a pool of the bytes), searched by binary search with `bytes.compare` and inserted into by a shift. A hash table would need a
specified hash, a collision rule and a growth rule, and its iteration order would be a thing to keep out of the result; a sorted array has none of
these, and its cost is bounded by the limit that is checked first: at most `k` shifts of `k <= 1,024` entries per new category, and `log2 k <= 10` byte
comparisons per cell. The frozen vocabulary is rebuilt from it in sorted order (the pool in first-seen order is not kept), so no byte of the result is a
function of the order the strings were met in. The cost is **not measured here**; no speed is claimed.

**Two passes, as the reader already has [read].** Pass 1 infers the numeric kinds, counts rows and builds the dictionaries (refusing on a limit as it goes);
`freeze` makes the vocabulary; pass 2 fills the float matrix and the ids, looking each cell up by binary search in the frozen vocabulary. A cell that is
not found in pass 2 at fit means the file changed between the passes: `io.read-failed`, as for numbers.

## 4. The limits

Each limit is checked **before the thing it limits exists**: a cell before it is copied into the pool, a new category before the slot is used, the column
count before any dictionary is allocated, and the id matrix is allocated only after pass 1 has finished and `limits.check_matrix` has accepted
`rows x (numeric + categorical columns)` (the id matrix counts in `--max-cells` at 8 bytes a cell, like the doubles).

| limit | flag | default | ceiling | tag | why these numbers **[choice]** |
|---|---|---|---|---|---|
| categories per column | `--max-categories` | 64 | 1,024 | `ml.limit.categories` | The numbers of `encode`'s `max-categories` (`docs/transforms.md`), as the issue says ("bounds stay"). The same tag: it is the same rule met earlier. A one-hot block of 1,024 columns is already most of the default `--max-columns` 4,096 |
| bytes per category | `--max-string-bytes` | 255 | 65,536 | `ml.limit.string-bytes` | 255 is the longest name a plan can hold (`ml.limit.plan` refuses a name over 255 bytes), so a default category can always be written back into a plan or an error. 65,536 keeps a stored length in 17 bits in the dictionary entry and is far under `--max-line-bytes` |
| total pool bytes | `--max-pool-bytes` | 1,048,576 | 8,388,608 | `ml.limit.string-pool` | The sum of the bytes of all the distinct categories of all the columns; it bounds the memory of the dictionary and, from stage 3, the model file. 1 MiB is 128 columns at 64 categories of 128 bytes. The ceiling is half of the default `--max-bytes` of `predict` (16 MiB), so a model that fits at the ceiling is still readable by default |
| categorical columns | `--max-cat-columns` | 128 | 4,096 | `ml.limit.categorical-columns` | House Prices has about 40 text columns; 128 is three times that. 4,096 is the default `--max-columns`. With the other ceilings the dictionary slots are at most `4,096 x 1,024` words = 32 MiB, allocated only after this check and the header are done |
| the id matrix | `--max-rows`, `--max-columns`, `--max-cells` | as before | as before | `ml.input.too-many-*` | `rows x (numeric + categorical)` against `--max-cells`, by division. The scan already divides the cap by **every** column of the header, dropped ones too (status 13), so a file that reaches `load_cat`'s own check is under the cap; that second check is the one `catmat.make` repeats before it allocates (mutant `c4-ids-not-counted-in-cells` survives for that reason) |

The pool bound also bounds the **number** of distinct strings (each is at least one byte), so no count can exceed 8,388,608 whatever the other flags say.
A file of 2^20 distinct values in one column is refused at the 65th (default) and allocates nothing for the other 1,048,512 (C4, C8).

What is **not** limited separately: the bytes of a cell that is missing (it is empty or a marker of at most 255 bytes), and the length of a column's name
(the header is read as before). A cell longer than `--max-line-bytes` is `limit.line-too-long` before this stage is reached.

## 5. Refusals

Every refusal has a rule tag, an exit code (8) and the details below; `line` and `column` count from 1 and are those of the cell (`line` is where the record
begins, as for numbers). The libraries answer a code (`ml.refuse`, band 1 to 99, codes 16 to 20); the wiring translates by tag (`design.md` 8.1).

| tag | when | `detail` | repair |
|---|---|---|---|
| `ml.limit.categories` | a column meets its `max + 1`-th distinct value | `line`, `column`, `name`, `max` | none (raise `--max-categories` up to 1,024, or `bucket`/`--drop` the column) |
| `ml.limit.string-bytes` | a cell of a categorical column is longer than the limit | `line`, `column`, `name`, `length`, `max` | none |
| `ml.limit.string-pool` | a new category would take the pool past its limit | `line`, `column`, `name`, `size`, `max` | none |
| `ml.limit.categorical-columns` | more declared columns than the limit; seen at the header, before any row | `columns`, `max` | none |
| `ml.input.invalid-utf8` | a cell of a categorical column is not well-formed UTF-8 | `line`, `column`, `name`, `offset` (byte offset of the first bad byte in the cell) | `retry` with `--drop NAME` when that is plain; the hint says to re-encode the file as UTF-8 |
| `ml.input.unknown-column` (reused, exit 3 as before) | `--cat` names a column that is not in the header | `flag`, `position`, `name` | none |
| `args.conflict` (reused) | `--cat` names the target, a dropped column or a `--float` column; `--na-values` or a limit without `--cat` | `flags` | none |
| `ml.input.categorical-not-encoded` | a declared column and no step reads text columns (all of fit until stage 2) | `columns` (the names), `categories` (their counts), `vocabulary_sha256` | none: the hint names `encode column=NAME kind=onehot` (stage 2) and `--drop NAME` |
| `ml.schema.mismatch` (reused; library only in this stage) | the loader of `predict` is given a vocabulary with a column the file does not have | the missing names | none |
| `args.bad-value` (reused) | a bad `--na-values` list: an empty marker, more than 64, one over 255 bytes | `flag`, `reason` | none |

`ml.limit.categories` is the tag `encode` already answers at fit (`docs/transforms.md`), with a different `detail` (the step and column there, the
line and name here); both are the one rule. `ml.input.invalid-utf8` is stated with a repair because the cause is usually the file's encoding: the
statement in the issue ("refused with a repair, not accepted silently") is met by the hint and, where it is safe, by the `--drop` retry; dropping
a column changes the model, which is why it is a `retry` the caller may take, as for `ml.input.non-numeric`.

### 5.1 The order of the checks of one cell of a categorical column

1. unquote (a quoted field's `""` is one `"`; section 6);
2. empty: missing; 3. equal to a marker: missing;
4. longer than `--max-string-bytes`: `ml.limit.string-bytes`; 5. not well-formed UTF-8: `ml.input.invalid-utf8`;
6. already in the dictionary: done; 7. a new category: `ml.limit.categories` first, then `ml.limit.string-pool`; else it is inserted.

The cells of a row are read left to right, and the first problem of the file in that order stops the read (as the numeric cells do).

### 5.2 What is invariant under the order of the rows

Whether a file is accepted, and when it is, the vocabulary bytes and the id of every row. **Not invariant:** the line of a refusal, and which of two limits
is met first when a file breaks both (a column over its categories in one order may break the pool first in another). A gate that asserted more would be
asserting something false (C2 judges exactly the first sentence).

## 6. What a cell is

The rules are those of `docs/reader.md` section 2.2, restated for text because here the bytes of the cell are the value:

* The file is comma separated, RFC 4180. A field may be quoted; **inside quotes `""` is one `"`**; the quotes are not part of the value. A quote in the middle of an
  unquoted field is text (`ab"c` is the four bytes). The reader's cell for a quoted field is the text between its quotes with `""` still doubled; the loader
  undoes the doubling and only that.
* **A line break inside quotes is one LF**: the reader joins the lines of a record with LF and drops one CR before each LF, so `a\r\nb` inside quotes is the three
  bytes `a`, LF, `b`. A CR that is not before an LF is kept. (Python's `csv` keeps the CRLF; the oracle implements the reader's rule, not `csv`'s.)
* **A byte order mark** on the first line is dropped by the reader; a BOM anywhere else is the three bytes `EF BB BF`, valid UTF-8, part of the cell.
* **NUL is an ordinary byte** (valid UTF-8, U+0000): a category may contain it. Nothing here treats a NUL as the end of anything; names are printed in the
  JSON of an error through `toolbox.text.put`, which escapes it.
* **Nothing is normalised and nothing is folded** [read, the issue]: no Unicode normalisation (NFC and NFD forms of one letter are two categories), no case folding
  (`Male` and `male` are two), no trimming (`" A"`, `"A"` and `"A "` are three), no collapsing of runs of spaces. Those are explicit steps of a later stage if they
  exist at all, never implicit.
* **Validity is std's strict UTF-8** (`std.utf8.decode`: no overlong forms, no surrogates, nothing past U+10FFFF, maximal-subpart error runs). The first bad byte's offset is reported.
* **Equality is byte equality.** Two strings that look the same and differ in bytes (a Latin `a` and a Cyrillic `a`, a non-breaking space) are two categories.
* **The vocabulary is a set of strings**: a string met a million times is one entry. There is no count in it (stage 2's frequency encoding counts in its own fit).

## 7. The loader

`ml.csvload` gains `load_cat` beside `load_t` (which is unchanged: a numeric-only call is the code of before, and C9 judges its bytes):

```
load_cat(heap, fs, root, rel, full, cfg, cat: csv.CatCfg, target, skip, vocab: Option vocabulary)
    -> (Matrix numeric, Ids, Vocab, Shape, Report, code, unseen-counts)
```

* **Fit** (`vocab` none): the categorical columns are those named in `cat.cols`. Pass 1 builds the dictionaries; `dict.freeze` returns the `Vocab`; pass 2
  fills the matrix and the ids. The `Vocab` is returned to the caller: the one fit-time value stage 3 stores.
* **Predict** (`vocab` given): the categorical columns are the vocabulary's **by name**, whatever `cat.cols` says; the cells are checked as at fit (steps 1 to
  5 of section 5.1, so invalid UTF-8 and a long cell are refused the same way) and then looked up in the **stored** vocabulary: a string it holds is its id; a string
  it does not is **-2, counted per column, never added**; the vocabulary is read-only and the columns may be in another order than at fit. A vocabulary column
  the file lacks is `ml.schema.mismatch`. The unseen counts are what stage 2's `unknown=ignore` adds to its own count and `unknown=refuse` turns into
  `ml.input.unknown-category`; this stage decides nothing about them.
* The **in-memory round trip** is tested: fit on `FILE`, read `FILE` through its own vocabulary and get the ids of the fit; read `FILE2` and get Python's.
* `Vocab` has the accessors a serialiser needs (`ncols`, `name`, `count`, `string`) and `dict.dump`, the **canonical byte form used by the gates**
  (`u32` column count; per column `u32` name length, name, `u32` category count, per category `u32` length, bytes; little endian). It is a dump for hashing and
  diffing. **It is not the model file format and does not decide it**: stage 3 owns the section, its checksum and its bounds.

## 8. The command line in this stage

`ml fit --cat NAMES ...` reads the file with `load_cat`, so every refusal of sections 4 and 5 is reachable from the command line, in the order of the file.
If the read succeeds, the fit is refused with `ml.input.categorical-not-encoded` before any model is fitted or written: **no step of this stage reads a
text column**, and fitting without the column would drop it silently, which the issue's rule of declared kinds forbids. The refusal carries the names, the
number of categories of each and `vocabulary_sha256` (SHA-256 of the canonical dump of section 7), so a caller and the gates can see that the vocabulary was built and what it
holds without the file format existing. Its `repair` is `none`, with the hint: name the column in `encode column=NAME kind=onehot|label` (a step that
reads text columns from stage 2 of issue 21), or `--drop` it. `--dry-run` changes nothing.

Without `--cat` nothing changes: no numeric path calls the new code (the numeric functions are not edited; `scan` is the new scan with the categorical part
switched off), and C9 says the bytes are the same.

## 9. Results

Everything below was printed by a script of this repository (`scripts/check.sh` runs all of it; the one step of it that does not exit 0 is named at the end of this section), on one machine:
this Mac, arm64, macOS, compiler `a4572ea`, `build/reader-check` and `build/ml` built from this tree. The machine was **not quiet** (1-minute load average 2.5 to 3.3 during the gates,
other agents building); correctness gates do not depend on the load and no timing is a claim. **No benchmark was run**, and the cost of the sorted array, of the extra reader arguments on the
numeric path and of the second pass is therefore not measured.

| gate | result |
|---|---|
| C1/C6 vocabulary, ids, missing, refusals | 44 adversarial files (unicode and emoji, NFC against NFD, NUL, quotes, quoted line breaks and CRLF, duplicate-looking strings, spaces kept, all-empty, one category, markers, numeric-looking categories, BOM, strings at, over and far over every byte limit, invalid UTF-8, bad and unterminated quotes, ragged rows, `--cat` on the target, a dropped and a float column, flag order against file order) and 1,500 seeded random files (711 accepted, 789 refused by the oracle's verdict, random limits): `reader-check --mode cat` and `scripts/oracle_cat.py` agree on every line (vocabulary bytes, the canonical dump, every id with its bitmap bit, the numeric cells beside them, and for a refusal the status, line, column and aux): **0 differences** in 1,544 runs |
| C2 row order | 150 seeded tables (80 accepted at every order), 5 seeded permutations of the rows each: the vocabulary lines are identical, the ids and numeric cells of each row follow the row, and accepted/refused never changes: **0 differences** in 750 runs. The mutants that keep first-appearance order are killed by this gate alone (section below) |
| C3 thread count | 9 tables, `ml fit --model linear --cat ... --threads` at 1, 2, 4, 8 and 16: one answer (status and bytes) per table, and its `vocabulary_sha256` is the SHA-256 of the oracle's canonical dump; 5 reader runs per table byte-identical: **0 differences** in 90 runs |
| C4 limits | 75 runs: categories at 1, 2, 3, 64, 65, 1,024 and 1,025; strings of 1, 5, 255, 256, 1,000, 65,535 and 65,536 bytes, each at and one under its limit; the pool at 7, 8, 9 bytes; 1 to 4 categorical columns against limits of 1 to 3; each equal to the oracle and, through `ml`, to the right tag. A 203 MB file of 2,800,000 distinct values in one categorical column is refused at its 66th line (default) and 1,026th (`--max-categories 1024`) with a peak resident size of **1.5 MB and 1.6 MB** (16 MB allowed); a file of 2^20 distinct values is refused at the same two lines (its memory was not measured). `tests/cat.cho` refuses `catmat.make` of `2^31 - 1` rows by 4,096 columns without allocating. **0 differences** |
| C5 UTF-8 | 387 valid strings (in chunks) and 4,445 invalid ones, each alone and after ASCII (every lead byte against nine second bytes, random byte strings, the named cases): validity and offset equal Python's strict decoder in 8,901 runs; a dropped column is not read: **0 differences** |
| C7 round trip | 150 tables each read again through their own vocabulary (ids equal the fit's) and through a second file with the columns in another order, unseen strings, markers, empty cells, a missing vocabulary column and nine adversarial predict files (invalid UTF-8, too long, NUL, quotes, no rows, the target present with garbage): ids, `-2` and the unseen counts equal the oracle's in 459 runs: **0 differences** |
| C8 no trap | 3,618 runs of `reader-check --mode cat` and `ml fit --cat` (random bytes, mutated valid files, quotes in every position, 4,097 categorical columns, a 70,000-byte and a 65,536-byte cell, 2^20 distinct values, a million quotes, a quoted cell of a million doubled quotes, a million empty cells, NULs, invalid UTF-8 in every column): **0 traps, 0 malformed answers** |
| C9 numeric inputs | 161 fixtures of the binary of `7d6183d`: exit status, standard output and every file the same, byte for byte (`test_cat_baseline.py`); W1's golden record of the constant model (`test_wire_baseline.py`) the same; the whole conformance suite **156 tests OK, none skipped** (141 before this stage plus 15 for C11) |
| C10 tables | 35 flags (6 new), 72 rules (5 new), 4 log records, 183 fixtures (20 new) reach each other as T1 requires; `scripts/schemas.py --check`, `gen_models.py --check` and `manifest.py --check` clean (the authority row is unchanged; the declarations in `manifests/ml.authority.json` and `generated/ml/built.cho` are regenerated) |
| C11 declared, not inferred | 15 tests of `test_cat.py` and the fixtures: an undeclared text column is `ml.input.non-numeric` with the `retry --drop` repair, bytes unchanged (C9); the summary of the rule in `introspect` names `--cat`; every conflict, range and list error of section 2; the digest in the refusal equals the oracle's |
| C12 structure | 10 tests of `tests/cat.cho` (sorted and independent of the order including a hand-written dump, each limit exact and leaving the dictionary unchanged, UTF-8 forms with offsets, find and valid, the id/bitmap agreement, refusals before allocation), `tests/core.cho` (the five new codes) and `tests/t_registry.cho` (column kinds), `scripts/check_registry.py`: 64 refusal codes in 4 bands, 5 column kinds |

**Mutants** (`scripts/mutants_cat.py`, `docs/bench/mutants_cat.log`): **42 patterns, 40 killed, 2 survivors explained, 0 unexplained**; the unmutated control passes every gate. They are on the order of
the vocabulary (first appearance, descending, reversed freeze, ids from the row, trimming), the quotes, the missing rule (an empty cell as a category, markers not applied, by prefix, ignoring case, the
last marker skipped), UTF-8 (no check in the dictionary, no check in the reader, errors stepped over, the width of a sequence, the offset), every limit (categories, bytes, pool, columns, each off by
one in the dictionary and in the reader, the order of the byte and UTF-8 checks, a repeated string charged to the pool), the stored vocabulary (unseen mapped to an id, not counted, columns taken from the
flag), the thread count in the answer, and the command line (a wrong rule, `--cat` on the target, a limit without `--cat`, an empty marker, the hook removed, an unknown name ignored).
**The required one is killed by the row-order gate alone:** `c2-first-appearance` keeps the vocabulary in the order the strings are met (a linear search in the dictionary and in the frozen
vocabulary, insertion at the end, so every lookup still succeeds and only the row order decides the ids); `scripts/gate_cat.py order` reports `DIFF ids or cells of a row`, and the variant that
leaves the binary search in the frozen vocabulary is killed by the same gate as a file accepted in one order and refused in another. **The survivors:** `c4-ids-not-counted-in-cells`
(equivalent: section 4, the scan's cap already counts every header column) and `c7-fit-lookup-miss-ignored` (**a gap**: the branch that answers `io.read-failed` when pass 2 of a fit meets a
string pass 1 did not see is only reached if the file changes between the passes, which no gate can make happen deterministically; it is read, not tested). **What the mutants found about the
gates:** the first definition of `c2-first-appearance` replaced only the dictionary and was killed by the order gate for the wrong reason (lookups failed and the file was refused), which is why the
mutant was redefined to keep the lookups consistent and the first definition was kept as `c2-first-appearance-dictionary-only`.

**What the gates do not claim.** The oracle is written by the same author from the same document, with its own reading of RFC 4180 but the same understanding of the cell rules: it checks the
code against the document, not the document against the world. Python's `bytes` order and strict UTF-8 decoder are the outside references for sorting and validity. The numeric columns beside the categorical ones are
compared only for ints and short decimals (the numeric reader has its own gates, RD2 to RD9, all re-run by `scripts/check.sh` and unchanged). Nothing was run on a second machine, with more than one
thread (the reader is single-threaded), or on data that ships with the tools (stage 5 of the issue). The command line never writes a model for a categorical column in this stage, so a model file with a
vocabulary does not exist to be damaged: the damaged-file fuzz is stage 3.

**The one step of `scripts/check.sh` that does not exit 0 on this branch alone** is `scripts/check_docs.sh`: `scripts/site.py --check` compares the generated blocks of `README.md` and `docs/index.html`
with the binary, and the counts they print (`flags 29`, `limits 9`, `rules 67`, and the tables of limits and rules in the page) are now 35, 13 and 72. This branch does not edit those two pages (another
branch does); after they are merged, `python3 scripts/site.py` rewrites exactly those lines (checked in a scratch copy: it changes three lines of `README.md` and the generated tables of the page, after which
`check_docs.sh` passes: 237 links, 0 problems; the sitemap consistent).

## 10. Open questions and what stage 2 may assume

1. **Stage 2 may assume**: `Ids` and `Vocab` as in sections 3 and 7; ids `0 .. k-1` in bytewise order; -1 missing (bit set), -2 unseen only when a vocabulary was
   given; `unseen` counted per column; every limit already enforced, so an `encode` over a categorical column needs no limit of its own beyond its output width
   (`k` columns for one-hot against `--max-columns`); the float matrix holds only numeric columns, so a plan's column resolution must learn that a name may
   be a categorical column (the schema has the kind 4 for it); `--cat` is the only way a column becomes categorical.
2. **Whether `encode`'s own `max-categories` should remain** now that the reader holds a bound of the same name. Today both exist and the smaller wins; stage 2 decides.
3. **Empty as a category** (the issue's "unless declared otherwise") is not built (section 2).
4. **`--na-values` for numeric columns** is not built: the stored model cannot carry markers until stage 3, and a marker honoured at fit and not at predict would skew silently.
5. **Speed of the sorted array** is not measured.

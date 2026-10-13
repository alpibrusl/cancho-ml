# cancho-ml command line (`ml`): M9 built early as a vertical slice

**Status: the gates (section 1) were written before any code of `tools/ml`. Sections 2 onward are filled in as the code lands;
a number or a verdict appears in them only if a script in `scripts/` printed it, with the conditions.** Tags: **[read]** read
from a document of the pinned compiler or of cancho-table, **[choice]** a decision the maintainer can overturn, **[open]** not
decided.

## 0. What this slice is

The agent-visible surface of cancho-ml is the command line (and later an MCP server). `ml` is one binary with the verbs
`fit`, `predict`, `eval`, `introspect` and `skill`, built the way `cancho-table`'s `table` is: one flag table drives the
parser, `introspect` and `skill`; every refusal is data with a `<area>.<name>` tag, an exit code and, where safe, a repair;
the authority is derived by the compiler and gated against a committed manifest. The linear-algebra and tensor packages stay
internal: nothing in this slice imports them, because the one model built here (`--model constant`) needs none.

It is a **vertical slice**: one real model end to end, not a framework. There are no fake models: the `--model` flag's choice list (in `introspect`) is the list of what is built, and a test pins it.

## 1. The gates (written first; each one must be able to fail)

A gate is a test in `tests/conformance/` run by `scripts/check.sh`. "Can fail" is shown by a **mutant** in
`scripts/mutants_cli.py` (a one-line change to the source, or to the table, the schema or the fixtures, that the gate must
catch); a survivor is listed with its reason in section 7.

| id | gate | test | mutants that must be killed |
|---|---|---|---|
| T1 | **Tables.** Every flag of `flag_table()` is in `introspect`, in the text of `skill`, in the `flag` enum of `schemas/ml.v1.json`, and used by at least one fixture of `tests/conformance/fixtures.py`. Every rule tag of `rules` is in `introspect`, in `skill`, in the `rule` enum of the schema, and reached by a fixture whose first error is that tag with the exit code of the table. Every record of `log_table()` is in `schemas/log.v1.json` and appears in the log of some fixture. The schemas are generated from the tables by `scripts/schemas.py` and a stale file fails. | `test_tables.py` | flag added to the table with no fixture; flag removed from the schema; rule added with no fixture; fixture removed; log record added with no fixture; skill text missing a flag |
| R1 | **Refusals.** Every refusal is exit code of D4 (agent-toolbox.md), one JSON line on stdout valid against the schema, stderr empty in JSON mode, `error` equal to `errors[0]`, `ok:false` exactly when the exit is not 0 or 9. | `test_rules.py` | exit code of a tag changed; the answer rendered as text in JSON mode; `ml fit` with no file not refused |
| N1 | **No trap.** A seeded fuzz of argument vectors (missing values and operands, empty strings, `..`, absolute paths, over-long paths, duplicated and unknown flags, every flag with every verb) never produces a trap status (132, 134, 136, 139, a negative status) and always one valid JSON line. Spike F found `SIGILL` on a missing argument and on `..`: both are fixtures, and the fuzz found a third (two dots in a row, section 2.1). The vectors run with and without `--root` (without it the builtins see the whole path). | `test_fuzz_args.py` | argv[1] read before counting the arguments; the two-dots check removed |
| L1 | **No payload is logged.** Fit and predict inputs carry canary strings (cell values, column names, a path component); the log file and stderr never contain a canary; every log line is valid against `schemas/log.v1.json` (a closed object, the key set and order of the table); the log is bounded (the bytes one run adds are asserted, and a log at its cap is refused before any work with `limit.log-full`). | `test_log.py` | a column name written into a log record; the cap check removed |
| M1 | **Model file.** The writer and the reader round-trip; bad magic, a version the reader does not know, a truncated file at every length, a flipped bit anywhere and a wrong checksum are refused with `ml.model.bad-magic`, `ml.model.version`, `ml.model.truncated`, `ml.model.checksum`; a structurally corrupt file whose checksum was recomputed is `ml.model.malformed`. A corrupted-file fuzz (every truncation, every single-byte flip of a small model, seeded multi-byte damage, seeded structural damage with a repaired checksum) never traps and never reports success for a file whose checksum is wrong. | `test_model_file.py` | checksum not compared; a section length not checked against the file size; version not compared |
| S1 | **Schema check.** A prediction input whose columns differ from the fitted schema is `ml.schema.mismatch` with the missing, extra and mistyped columns in `detail` and a repair that names them; column order may differ. | `test_schema.py` | the check removed; the repair listing the wrong side |
| O1 | **Oracle.** The model is real and right: `fit`, `predict` and `eval` agree, value for value, with an exact-rational oracle in Python (`fractions.Fraction`) on seeded data that includes values whose sums leave 64 bits. | `test_oracle.py` | mean divided by n-1; tie broken the other way; rounding of the prediction changed; a carry dropped in the exact sum; accuracy counted wrongly; eval using the wrong scale |
| D1 | **Determinism.** The same input and plan give the same model bytes, whatever the working directory, the spelling of the path or the number of repetitions; the same command gives the same stdout bytes. If a `--threads` flag ever appears in the table, the test runs it at 1, 2, 8 and 64 and compares bytes; today there is none and the test says so. | `test_determinism.py` | a stand-in binary whose bytes depend on a `--threads` value (the test must fail on it) |
| A1 | **Authority.** `scripts/manifest.py` derives the authority with `cancho authority`, writes `manifests/ml.authority.json` and `generated/ml/built.cho`, and `--check` fails if the committed manifest, the embedded one or the binary's `introspect` differ from a fresh derivation, or if a derived label is outside the ceiling of `tools.toml`. | `test_authority.py`, `scripts/manifest.py --check` | a mutant that adds a capability (a file write by path, a `Clock`, `Ffi`) must fail the check |
| H1 | **Honesty.** the `--model` choice list is exactly the built models; asking for another is `args.bad-value`; a float column is read to the nearest double by cancho-table's exact reader (`docs/reader.md`), never approximately. *(Corrected in place: this row said a float column was refused with `ml.input.unsupported-type`, because the float reader was believed unmerged; it is merged, PR #32, and the rule is gone.)* | `test_honesty.py` | a model name added to the table without an implementation |

Nothing in a gate is "expected to pass": each row names the mutant that proves the gate can fail, and section 7 reports the
verdict of each, with the survivors.

## 2. The command line

```
ml [FLAGS] fit FILE       fit a model on a CSV and write it
ml [FLAGS] predict FILE   apply a stored model to a CSV
ml [FLAGS] eval FILE      score a stored model on a CSV
ml introspect             the tool as data (flags, rules, limits, authority, schema)
ml skill                  the same as a SKILL.md for an agent
```

The first operand is the verb, the second the CSV file. Flags may stand anywhere. A flag a verb does not take is
`args.conflict` (the lists are `allowed()` in `tools/ml/spec.cho`); an unknown flag is `args.unknown-flag`, never ignored. The
flag table, with kinds, defaults and help, is `ml introspect | jq .flags` and is not copied here, because a copy is a second
table to drift (gate T1 checks the one table against its projections).

| verb | needs | takes also |
|---|---|---|
| `fit` | `--target NAME --output MODEL` | `--model constant`, `--task regression\|classification`, `--drop`, `--float`, `--overwrite`, `--dry-run`, the `--max-*` caps, `--log`, `--format` |
| `predict` | `--model-file MODEL` | `--drop`, `--limit`, `--from`, `--max-*` caps, `--max-bytes`, `--log`, `--format` |
| `eval` | `--model-file MODEL` | `--drop`, `--max-*` caps, `--max-bytes`, `--log`, `--format` |

**Output.** One JSON object on one line on standard output, schema `ml.v1` (`schemas/ml.v1.json`): `{ok, command, schema, data,
error, errors, meta}`, errors `{code, rule, message, hint, repair, detail}`, no timestamp, no duration, no float (every decimal is a
string whose digits are exact). In `--format text` a success is a few lines and an error is a sentence on standard error.

**Exit codes** (cancho `docs/agent-toolbox.md` D4, written out independently in `test_exit_codes_are_the_d4_classes`): 0 done, 1 the
operating system failed (`io.read-failed`, `io.write-failed`), 2 the invocation is malformed, 3 a named input does not exist,
4 refused (permission, outside the root, a symbolic link), 5 conflict (`conflict.exists`), 8 the answer is no or a limit was
reached (`ml.*`, `limit.*`, `parse.*`), 9 a `--dry-run` completed. With several errors the first, in input order, decides the code and
`errors` lists them all.

**Dry run.** `fit --dry-run` reads, checks and fits, runs every check the real write would (the directory, an existing output,
`--overwrite`), and answers `dry_run: true` and `planned_actions: [{op: create|replace, path, bytes, sha256}]` with exit 9. It opens the
directory and the output name to look, and creates, renames and removes nothing (`fsio.write_atomic` with `dry`). A test shows the
file is absent afterwards and unchanged when it exists. *Not checked under `strace`:* that is the claim of the contract package's
`dry_run` guarantee and it is not re-measured here.

### 2.1 Paths and confinement

* `--root DIR` is the directory FILE, `--model-file`, `--output` and `--log` are resolved beneath. Each is checked lexically by the
  pinned `toolbox.path` (`path.empty`, `path.too-long`, `path.dotdot`, `path.absolute`, `path.outside-root`) and then opened one
  component at a time with no link followed (`toolbox.place`): a link anywhere below the root is `path.symlink`.
* Without `--root` a path is the working directory's, and links are followed (`confinement: none` for readers). Writers still work
  beneath the directory that holds the file.
* **A path-valued flag takes its value as an argument of its own** (`--output FILE`, not `--output=FILE`; the second is
  `args.bad-value`). The lexical checks of `toolbox.path` take an argument by index. `--root=DIR` is accepted.
* **Two dots in a row are refused wherever a whole path reaches a builtin.** Found by the fuzz (gate N1): `open_read` and `open_dir`
  trap on any path that contains `..`, not only on a `..` component, while `toolbox.path.has_dotdot` looks for the component alone, so
  `a..b` and `.../x` passed the check and ended the process with `SIGILL`. Under `--root` only the root string and, without a root, the
  whole path reach those builtins; `ml` refuses them with `path.dotdot` (`paths.has_double_dot`). Under `--root` a name such as
  `a..b.csv` is an ordinary name (a fixture). The same hole is in the pinned package: see section 8.
* The output is written to `<name>.ml-tmp` beside the final name, synced, renamed over it and the directory synced, so a reader never
  sees half a model (`fsio.write_atomic`). A leftover temporary from a crash makes the next write `io.write-failed`
  (`O_EXCL`); it is not removed by the tool.

### 2.2 Limits

Every number sized by input has a flag, a default and a ceiling, listed by `introspect.limits` (the table is `limits()` in
`spec.cho`, and gate T1 checks it against the flags): `--max-rows` (default 10,000,000, ceiling 1,000,000,000), `--max-columns`
(4,096 / 8,192), `--max-cells`, `--max-line-bytes` (1 MiB / 16 MiB), `--max-bytes` for a model file (16 MiB / 256 MiB), `--limit`,
`--from`, `--log-max-bytes`. A value outside its range is `ml.param.out-of-range` with the range in `detail`. The ceilings keep every
sum in the exact accumulators below 2^63 (section 3.4); they are **choices**, not measurements, and section 9 of `design.md` still
waits for the speed factor.

### 2.3 The CSV reader (`tools/ml/csv.cho`, `csvcell.cho`, `csvload.cho`)

*(Rewritten by the reader stage; the first version of this section said cancho-table's float reader was unmerged and listed a 12-digit
reader of its own. `docs/reader.md` is the full text, with its gates.)* A cell is read by `ml.csvcell.read`: an int or a decimal of at most 12
significant digits exactly as before (an integer mantissa and a scale), and everything else (an exponent, `.5`, `5.`, more than 12 digits) by
cancho-table's `flt.parse_float`, the nearest double, ties to even. A column is `int`, `dec(S)` or `float`, widening only; `--float NAMES`
forces float. Records are RFC 4180 (cancho-table's `reader` module): quoted fields, doubled quotes, line breaks inside quotes, a BOM, CRLF,
blank lines. `nan` and `inf` are `ml.input.non-finite`, and so is a number past the range of a double; a cell that is empty, is not a number,
reads as zero though it is not zero, or is longer than 1,100 bytes is `ml.input.non-numeric` with the reason in `detail`. A quote closed and
followed by anything but a comma is `parse.csv-bad-quote`, a quote open at the end `parse.csv-unterminated-quote`.

The modules of cancho-table are **not copied** and, today, **not a `[dependencies.*]`** either: cancho-table commits no `.cancho-vcs` store
(`docs/reader.md` section 1). The binaries compile the files of a read-only clone pinned by `table.pin` (LOCAL-DEV), and
`docs/upstream/cancho-table-numeric-package.md` is the proposal that ends it.

One function, `csv.scan`, reads the file once in bounded memory (one 64 KiB chunk, the record being read up to `--max-line-bytes`, the header,
the offsets of one record's fields, four ints a column). `fit` scans twice: pass 1 for the shape, pass 2 for the target folded into exact
accumulators (an int or dec target: `ml.exact`; a float target: `facc`, the exact sum of the doubles), and a cell that no longer fits what pass 1
saw (the file changed between the passes) is an error, not a trap. `predict` and `eval` scan once; `eval` folds the target while it infers the
shape and discards the fold if the schema check fails. The matrix loader (`csv.load`, `tools/reader-check`) is the same scan with a matrix
as its sink; the CLI's constant model builds none.

### 2.4 A session

(Run on this machine with the binary of this commit; the hash is the file's SHA-256.)

```
$ ml --root . fit --target y --output sales.cml sales.csv           # sales.csv: id,x,y / 1,2.5,10 / 2,3.5,20 / 3,1.0,30 / 4,0.25,41
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":4,"columns":3,"target":"y","path":"sales.cml","bytes":316,"sha256":"9b6a34144d1351f141f1ed2f1238f053afef57d815f29d047892239b63afbd05","written":true,"summary":{"mean":"25.250000"}},"meta":{"version":"0.1.0"}}
$ ml --root . predict --model-file sales.cml new.csv                # new.csv: id,x / 5,2.0 / 6,4.5   (no target column needed)
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","columns":["prediction"],"rows":[["25.250000"],["25.250000"]],"row_count":2,"total_rows":2,"truncated":false,"next":null},"meta":{"version":"0.1.0"}}
$ ml --root . eval --model-file sales.cml check.csv                 # check.csv = sales.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":4,"metrics":{"rmse":"11.519006","sse":"530.750000000000","prediction":"25.250000"}},"meta":{"version":"0.1.0"}}
$ ml --root . fit --target y --output sales.cml sales.csv           # again
{"ok":false,...,"error":{"code":"CONFLICT","rule":"conflict.exists","message":"the output file exists","hint":"use --overwrite to replace it, or name another file","repair":null,"detail":{"path":"sales.cml"}},...}     # exit 5
$ ml --root . fit --dry-run --overwrite --target y --output sales.cml sales.csv
{"ok":true,...,"data":{...,"written":false,...},"dry_run":true,"planned_actions":[{"op":"replace","path":"sales.cml","bytes":316,"sha256":"9b6a...bd05"}],...}   # exit 9
$ ml --root . fit --target y --output f.cml flt.csv                 # flt.csv: id,x,y / 1,1e5,10 / 2,0.1,2.75   (the CLI slice refused the 1e5 with ml.input.unsupported-type)
{"ok":true,...,"data":{...,"bytes":316,"summary":{"mean":"6.37500000"}},...}                  # x is a float feature; y is a decimal target: the exact path, 8 digits as before
$ ml --root . fit --target y --output g.cml flt2.csv                # flt2.csv: id,x,y / 1,1e5,1e1 / 2,0.1,2.75   (a float target)
{"ok":true,...,"data":{...,"bytes":300,"summary":{"mean":"6.375"}},...}                       # the exact sum of the doubles over n, rounded once, the shortest decimal back
```

## 3. The constant model and its arithmetic

`--model constant` is the only model built, and it is a real one: it is the baseline every other model is compared with, and the
slice is judged on whether the whole path (CSV, schema, file, prediction, score) is right, not on how clever the estimator is.

### 3.1 Regression

The prediction is the training mean of the target. The target is a column `int` or `dec(S)`; every value is an integer mantissa
`m_i` (below 10^12 in size) at scale `S`. The sum `T = sum m_i` is exact: it is kept in two limbs (`sum_q = sum trunc(m_i / 10^9)`,
`sum_r = sum (m_i mod 10^9)`, `ml.exact.sum_add`) because 10^7 values of 12 digits already leave 64 bits. The prediction is

    P = round_half_away_from_zero( T * 10^6 / n )        a scaled integer at scale S + 6, printed as a decimal

computed by long division in base-10^9 limbs (`ml.exact.mean_scaled`), no float. So the answer has six more fractional digits than
the target, rounded, and **that rounded value is the prediction**: `predict` prints it, `eval` scores against it, and the test oracle
(`fractions.Fraction`) agrees with every digit.

### 3.2 Classification

The target is an `int` column with exactly two values `c0 < c1` (more is `ml.label.not-binary`, fewer `ml.label.single-class`, a
decimal column `ml.label.not-integer`). The model stores the counts `n0`, `n1`. The predicted label is the class with more rows, the
smaller label on a tie (rule 4 of `design.md` section 6: lowest index wins), and `probability` is `n1 / n` = P(label = `c1`) rounded
half up to six digits (`ml.exact.fraction_scaled`). The format reserves the class count (the tensor has `K = 2`), so more classes
change no existing byte.

### 3.3 Eval

* Classification: `correct` rows whose target equals the predicted label, `total` rows, `accuracy = correct / total` to six digits.
  A label the model has not seen is a wrong row, not an error.
* Regression: each target cell is shifted to the model's scale (an input with fewer fractional digits is fine, more is a schema
  mismatch), then `e_i = y_i * 10^6 - P` is an exact integer below 2 * 10^18, and `SSE = sum e_i^2` is an exact integer kept in six
  limbs (`ml.exact.square_add`), printed **exactly** as `sse` with `2 (S + 6)` fractional digits. `rmse` is **not** exact: it is
  `trunc(sqrt(float(SSE) / float(n)) + 0.5)` in units of `10^-(S+6)`, where `float(SSE)` is the limbs folded left to right in binary64
  (`x = x * 1e9 + limb`). That is a fixed sequence of IEEE operations with no `fma` and no library call (`sqrt` is an instruction), so
  the same bits come out on every machine (`design.md` D2, **not yet measured on a second machine**); the oracle repeats the steps in
  Python and also bounds the result within 1 unit and 1e-12 relative of the true square root.

### 3.4 Bounds

A cell has at most 12 digits, a file at most 10^9 rows (`--max-rows` ceiling), so: `|sum_q| <= 1000 n`, `|sum_r| <= (10^9 - 1) n`,
each squared-error limb gains less than 5 * 10^9 a row (below 2^63 for 10^9 rows), and the long division keeps every partial
dividend below `n * 10^9 < 2^63`. The model reader checks the same bounds on the numbers in a file before they reach the
arithmetic (a repaired checksum does not make a file trustworthy: `test_every_field_check_has_a_file_that_reaches_it`).

## 4. The model file

`design.md` section 4.1 is the format and is updated to match this slice: magic `CML1`, `u32` version 1, `u32` section count 5, five
sections (`plan`, `schema`, `steps`, `model`, `meta`), then a **32-byte SHA-256 trailer over every byte before it**. (Section 4.1 first
put the digest inside `meta`; a digest inside the bytes it covers is circular for the whole file, so it is a trailer.) Every numeric
block is a tensor record `dtype u32 | rank u32 | shape (rank * u64) | elements`; the constant model's are `i64`, rank 1. The file holds
no timestamp, no host, no path; it holds the compiler pin it was built with (informational, as section 3 chose), so **two builds with
different compilers write different bytes for the same data**. A test pins the layout with an independent reader written from the
document. **The plan** (section 4.2) is the one canonical line `model kind=constant task=regression\n` (or `classification`): fixed key
order, one spelling, built from `--model` and `--task` alone, since no transform step is built yet (impute, scale and encode are M3
and M4); the reader accepts those two byte strings and nothing else.

The reader refuses in this order: `ml.model.bad-magic`, `ml.model.version`, `ml.model.truncated` (the structure: five sections in order
whose lengths fit the bytes that are there, checked before anything is allocated from them; a length that runs past the end, even
2^62, is this), `ml.model.checksum`, `ml.model.malformed` (a field outside what the format allows). Because the structure is walked
before the checksum, a flipped bit in a length field reports `truncated` or `malformed` rather than `checksum`; no damaged file is
ever accepted (gate M1 flips every byte three ways).

*Added at `v1-post` (rule in `design.md` section 4.1).* A file of a model kind of 4 to 7 (written by the libraries, not by `ml`) has a `u32 sum_order` as the last four bytes of `meta`; `ml` still builds kinds 1 to 3, whose `meta` has no such field and is unchanged. The reader checks that field after the checksum and before the other fields: a kind 4 to 7 file with a sum order other than 2 is `ml.model.version` (detail `sum_order`, `known`); with order 2 it is `ml.model.malformed` in section 4 (kind not built by `ml`).

## 5. The log

`--log FILE` appends one NDJSON record at the start of the run (`start`), after the read (`read`: rows, columns, bytes of the CSV), after
a real write (`write`: bytes and SHA-256 of the model) and at the end (`end`: ok, exit, the first rule), in `spec.log_table()`. The
fields are counts, flags, rule tags and a hash: the writers in `ml.logrec` take no cell, no column name and no path, so none can be
logged; gate L1 plants canaries in all three and in every kind of run. Bounded: at most four records a run, each under 256 bytes
(`per_run` is asserted at 2048), and a log already `--log-max-bytes` long is refused **before any work** with `limit.log-full`
(so the file can pass the cap by one run at most). No timestamp, because there is no clock (a record with the time would need the
`clock` label the ceiling forbids): order is `seq`, the position in the run.

## 6. Authority and determinism

`tools.toml` is the ceiling and `manifests/ml.authority.json` the derivation (`scripts/manifest.py`): `args`, `dir_read`, `dir_write`,
`err_write`, `file_read`, `file_write`, `fs_read("")`, `heap`, `io_write` and nothing else. In words: it reads the files named on its
command line; it creates, appends to, renames and removes only beneath a directory it opened itself, and holds no `fs_write`, so no
path write; it has no standard input, no clock, no environment, no network, no foreign code. `design.md` section 7 expected `conc`; there
is none, because the slice has no `--threads`. **The label arrives with the first threaded kernel, in the same diff as the test that
no thread count changes a byte** (gate D1 already contains that test and skips today, with the reason). `fs_read("")` is the only
extent the compiler can state: the directory is a run-time value (agent-toolbox.md L5), so the row names the verb, not the place.

Determinism is tested from the outside (gate D1): the same data and plan give the same model bytes whatever the working directory, the
spelling of the path (`./`, `//`), the line ending, a BOM, blank lines, the caps given, or the repetition; the same command gives
the same stdout bytes five times.

## 7. Gate verdicts

Everything below was printed by a script of this repository, on one machine (this Mac, arm64, macOS, the pinned compiler `a4572ea`),
under a 1-minute load average of 3.8 to 6.0 (other processes on the machine; not a quiet-machine run, and no figure here is a benchmark).

**Conformance tests** *(as of the CLI slice; the reader stage's count and verdicts are in `docs/reader.md` section 6)* (`cd tests/conformance && python3 -m unittest discover -p 'test_*.py'`): 88 tests, all pass, 1 skipped (the
`--threads` branch of D1, because there is no such flag; the skip says so). The last run took 28.2 s. 90 fixtures
(`tests/conformance/fixtures.py`) reach all 19 flags, all 48 rules and all 4 log records of the tables. The numeric gates of stage 1
and the unit tests of `lib/core/exact.cho` (7) pass in `scripts/check.sh`, which exits 0.

**Mutants** (`python3 scripts/mutants_cli.py`, log `docs/bench/mutants_cli.log`): 34 mutants, each a copy of the tree with one change and
the binary rebuilt, run against the tests of the gate it targets; **all 34 killed, none survived**, and the unchanged tree passes.

| gate | mutants | killed | survived | the mutants |
|---|---|---|---|---|
| T1 tables | 6 | 6 | 0 | `t1-flag-without-fixture`, `t1-flag-missing-from-the-schema`, `t1-rule-without-fixture`, `t1-fixture-reaches-another-rule`, `t1-log-record-without-fixture`, `t1-skill-lacks-a-flag` |
| R1 refusals | 3 | 3 | 0 | `r1-exit-code-changed`, `r1-text-in-json-mode`, `r1-missing-operand-not-refused` |
| N1 no trap | 2 | 2 | 0 | `n1-verb-read-before-counting`, `n1-double-dot-check-removed` |
| L1 log | 2 | 2 | 0 | `l1-column-name-logged`, `l1-cap-not-checked` |
| M1 model file | 4 | 4 | 0 | `m1-checksum-not-compared`, `m1-section-length-not-checked`, `m1-version-not-compared`, `m1-trailing-bytes-accepted` |
| S1 schema check | 2 | 2 | 0 | `s1-check-removed`, `s1-repair-names-the-wrong-side` |
| O1 oracle | 8 | 8 | 0 | `o1-mean-divides-by-n-minus-1`, `o1-tie-goes-to-the-larger-label`, `o1-half-rounds-down`, `o1-mean-truncates`, `o1-carry-dropped`, `o1-sum-in-one-int`, `o1-accuracy-counts-the-wrong-rows`, `o1-eval-ignores-the-cell-scale` |
| D1 determinism | 2 | 2 | 0 | `d1-model-holds-the-byte-count`, `d1-thread-count-changes-a-byte` |
| A1 authority | 3 | 3 | 0 | `a1-reads-standard-input`, `a1-reads-the-clock`, `a1-manifest-is-stale` |
| H1 honesty | 2 | 2 | 0 | `h1-a-model-in-the-table-that-is-not-built`, `h1-an-exponent-is-read-as-a-prefix` |

How the gates were checked to be able to fail, and what they found:

* **Bugs the gates found in the code** (both fixed, both fixtures now): (1) `eval` of a classification model computed the regression
  mean from the class labels before choosing the metric, and trapped on 12-digit labels (found by the oracle gate, O1); (2) a path
  with two dots in a row (`a..b`, `.../x`) passed `toolbox.path` and ended the process with `SIGILL` in `open_read`/`open_dir`, with
  no `--root` and as a `--root` value (found by the fuzz, N1, once the vectors ran without `--root`: with it they did not). The
  seeded fuzz of 2,500 vectors plus about 3,200 systematic ones (every flag with every one-flag mistake, in every verb) found nothing else.
* **A gate that was weaker than it looked.** The first full run had one survivor, `m1-trailing-bytes-accepted`, which was written down
  as an equivalent mutant. It was not: a crafted file with junk before the trailer and the SHA-256 of all of it passes the checksum and
  only the structure check refuses it, and no test built such a file. A test now does and the mutant is killed. (Appended bytes after
  the trailer do fail the checksum; that is what made the first test pass and the mutant look equivalent.)
* **A mutant that did not do what its name said.** `n1-missing-operand-check-removed` survived the fuzz: removing the check does not
  trap, because the contract package's accessors answer a harmless value for an operand that is not there (`ml fit` alone becomes a
  read of argument 0). It is now `r1-missing-operand-not-refused` (killed by the fixtures, gate R1), and the trap class that
  spike F found is `n1-verb-read-before-counting` (argv[1] read before there is one), killed by the fuzz.
* **What the gates do not claim.** O1 compares with an oracle written by the same author from the same text; it checks the tool against
  the document, not the document against anything outside it (the oracle's RMSE repeats the tool's three IEEE steps in Python).
  D2, the same bytes on another machine, is not tested here. T1 checks that the tables, the schema, the skill text and the fixtures
  agree; it does not check that a help sentence is true. The `--threads` branch of D1 is shown able to fail only by a stand-in binary.
  The `dry_run` guarantee ("no mutating system call") was not run under `strace`. No gate measures memory or speed.

## 8. What the pinned contract package says that is wrong or too narrow for `ml`

`ml` uses `toolbox.describe` as pinned (rev `8dab9176`), so `introspect` and `skill` repeat some generic sentences. They are the
package's, not this repository's claims, and are listed here so that nobody reads them as measured for `ml`. Each is an upstream
issue for cancho-tools, not worked around by copying the module.

| where | the package prints | for `ml` |
|---|---|---|
| `introspect.guarantees.concurrency`, `skill` | "read-only: it never writes, so any number may run at once", or, with `atomic`, "locked: of writers racing ... exactly one wins" | `fit` writes. Its write is built to be atomic for a reader (temporary, sync, rename) but there is no lock and the temporary has a fixed name, so two writers racing on one output can fail one of them; **neither the atomicity nor the race was tested**, so `atomic` is not claimed |
| `introspect.evidence.offline` and `offline_note` | gates M1 to M9 "in cancho-tools/tests/conformance, run by CI on the commit this binary was built from" | the gates of `ml` are in this repository (`tests/conformance`), section 1 |
| `skill`, the authority paragraph | "Symlinks are followed." | not under `--root` (`path.symlink`); followed without a root |
| the shared summary of `conflict.exists` | "--create was given and the path exists" | the flag is `--overwrite`'s absence; the tag's meaning (the output exists) is the same |
| `toolbox.path.has_dotdot` | refuses a `..` *component* | the builtins that take a whole path (`open_read`, `open_dir`) trap on `..` anywhere in the string (`a..b`, `.../x`): found by gate N1, worked around in `tools/ml/paths.cho`, and a reproducer for cancho-tools is `ml --target y --output o.cml a..b` with the workaround removed (mutant `n1-double-dot-check-removed` is exactly that) |
| `describe` | no field for a tool-specific `confinement_note` or `concurrency` | one sentence each would let a tool state its own |

## 9. The MCP server: not started; how it will be derived (stub for the M9 issue)

cancho-table's `server/mcp.cho` derives its one tool from `table introspect`; the server for `ml` should do the same and hold no
description of its own. From the flag table:

* **Three tools**, `ml_fit`, `ml_predict`, `ml_eval`, one per verb, each with the properties of `spec.allowed(verb)` and `FILE` as
  `file`. The input schema comes from each flag's `kind` (`nat` is an integer with the minimum of the check and the `introspect.limits`
  ceiling, `choice:a/b` an enum, `path` a string, `bool` a boolean, `text` a non-empty string) and its `default`.
* **Roles decide what is exposed.** `root` is the server's (`ml_* --root DIR`, first and once; a `root` property is refused as in
  table). `guard` flags (`--overwrite`, `--dry-run`) are the only way to widen or to narrow what a call does, so the server
  does not offer `overwrite` unless its own command line allowed it. `path-write` flags (`--output`, `--log`) stay confined by the tool
  itself; the server adds no path check of its own.
* **The argv is fixed in form.** Every flag value is its own argument. Table's server uses `--name=value`; `ml` refuses that form for
  path flags (section 2.1), so the generator must emit two arguments for flags of role `path-*`. The alternative, teaching `ml` to take
  `--output=FILE`, needs a text form of `toolbox.path.operand` that the pinned package does not have.
* **Result.** `structuredContent` is the one JSON document, `isError` is `ok == false`, and `meta.version` is the tool's. The
  generator refuses a tool whose derived authority holds `io_read` (the server gives the child an empty standard input).
* **Gate (when built):** the same shape as table's `test_mcp.py`: the argv a stand-in binary receives, no shell metacharacter, the
  schema of every tool equal to a projection of the flag table (T1 extended with the MCP schema), and a mutant per rule.

## 10. Open questions

1. **`--threads` and `conc`.** When the first threaded kernel is wired in, does `fit` get `--threads` (as table does) or does the
   thread count stay internal? Section 6 of `design.md` says the block size must not depend on it; gate D1 is ready either way.
2. **Quoted CSV and the shared reader.** *(Answered by the reader stage: quoted CSV is read through cancho-table's `reader` module; the
   shared code is a pinned dependency in intent and a pinned local file list in fact, until cancho-table commits its stores: `docs/reader.md`
   section 1.)*
3. **Twelve digits.** *(Answered by the reader stage: the 12-digit cap keeps the exact int/dec path; a column that needs more is a float,
   read by cancho-table's reader and summed by `facc`. See `docs/reader.md`.)*
4. **`rmse` as a decimal.** *(For an int or dec target.)* It is three IEEE steps after an exact sum. An exact integer square root over the limbs would make it
   exact too, at about 60 lines; is the exact sum enough for "decisive sums are exact" (`design.md` section 6 rule 5)?
5. **The package's `describe` text** (section 8): upstream issues to file, not filed from here.
6. **A second machine.** D2 (same bytes on Mac aarch64 and Linux x86-64) is claimed by construction (no libm, no `fma`) and tested only
   on this Mac. The Linux run is the gram box's job and was not done here.

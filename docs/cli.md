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

It is a **vertical slice**: one real model end to end, not a framework. There are no fake models: `introspect.models` lists
what is built and a test pins that list.

## 1. The gates (written first; each one must be able to fail)

A gate is a test in `tests/conformance/` run by `scripts/check.sh`. "Can fail" is shown by a **mutant** in
`scripts/mutants_cli.py` (a one-line change to the source, or to the table, the schema or the fixtures, that the gate must
catch); a survivor is listed with its reason in section 7.

| id | gate | test | mutants that must be killed |
|---|---|---|---|
| T1 | **Tables.** Every flag of `flag_table()` is in `introspect`, in the text of `skill`, in the `flag` enum of `schemas/ml.v1.json`, and used by at least one fixture of `tests/fixtures/cli.json`. Every rule tag of `rules` is in `introspect`, in `skill`, in the `rule` enum of the schema, and reached by a fixture whose first error is that tag with the exit code of the table. Every record of `log_table()` is in `schemas/log.v1.json` and appears in the log of some fixture. The schemas are generated from the tables by `scripts/schemas.py` and a stale file fails. | `test_tables.py` | flag added to the table with no fixture; flag removed from the schema; rule added with no fixture; fixture removed; log record added with no fixture; skill text missing a flag |
| R1 | **Refusals.** Every refusal is exit code of D4 (agent-toolbox.md), one JSON line on stdout valid against the schema, stderr empty in JSON mode, `error` equal to `errors[0]`, `ok:false` exactly when the exit is not 0 or 9. | `test_rules.py` | exit code of a tag changed; an error written to stderr |
| N1 | **No trap.** A seeded fuzz of argument vectors (missing values and operands, empty strings, `..`, absolute paths, over-long paths, duplicated and unknown flags, every flag with every verb) never produces a trap status (132, 134, 136, 139, a negative status) and always one valid JSON line. Spike F found `SIGILL` on a missing argument and on `..`: both are fixtures. | `test_fuzz_args.py` | the missing-operand check removed; the `..` check removed |
| L1 | **No payload is logged.** Fit and predict inputs carry canary strings (cell values, column names, a path component); the log file, stdout of a failing run and stderr never contain a canary where only a count or a tag may be; every log line is valid against `schemas/log.v1.json` (a closed object, a fixed key set); the log is bounded (records per run, bytes per record, and a refusal `limit.log-full` at the file cap). | `test_log.py` | a column name written into a log record; the cap check removed |
| M1 | **Model file.** The writer and the reader round-trip; bad magic, a version the reader does not know, a truncated file at every length, a flipped bit anywhere and a wrong checksum are refused with `ml.model.bad-magic`, `ml.model.version`, `ml.model.truncated`, `ml.model.checksum`; a structurally corrupt file whose checksum was recomputed is `ml.model.malformed`. A corrupted-file fuzz (every truncation, every single-byte flip of a small model, seeded multi-byte damage, seeded structural damage with a repaired checksum) never traps and never reports success for a file whose checksum is wrong. | `test_model_file.py` | checksum not compared; a section length not checked against the file size; version not compared |
| S1 | **Schema check.** A prediction input whose columns differ from the fitted schema is `ml.schema.mismatch` with the missing, extra and mistyped columns in `detail` and a repair that names them; column order may differ. | `test_schema.py` | the check removed; the repair listing the wrong side |
| O1 | **Oracle.** The model is real and right: `fit`, `predict` and `eval` agree, value for value, with an exact-rational oracle in Python (`fractions.Fraction`) on seeded data that includes values whose sums leave 64 bits. | `test_oracle.py` | mean divided by n-1; tie broken the other way; rounding of the prediction changed; a carry dropped in the exact sum; accuracy counted wrongly; eval using the wrong scale |
| D1 | **Determinism.** The same input and plan give the same model bytes, whatever the working directory, the spelling of the path or the number of repetitions; the same command gives the same stdout bytes. If a `--threads` flag ever appears in the table, the test runs it at 1, 2, 8 and 64 and compares bytes; today there is none and the test says so. | `test_determinism.py` | a stand-in binary whose bytes depend on a `--threads` value (the test must fail on it) |
| A1 | **Authority.** `scripts/manifest.py` derives the authority with `cancho authority`, writes `manifests/ml.authority.json` and `generated/ml/built.cho`, and `--check` fails if the committed manifest, the embedded one or the binary's `introspect` differ from a fresh derivation, or if a derived label is outside the ceiling of `tools.toml`. | `test_authority.py`, `scripts/manifest.py --check` | a mutant that adds a capability (a file write by path, a `Clock`, `Ffi`) must fail the check |
| H1 | **Honesty.** `introspect.models` is exactly the built models; asking for another is `args.bad-value`; a float column is refused (`ml.input.unsupported-type`), never read approximately. | `test_honesty.py` | a model name added to the table without an implementation |

Nothing in a gate is "expected to pass": each row names the mutant that proves the gate can fail, and section 7 reports the
verdict of each, with the survivors.

## 2. to 6. (filled in below as the code lands)

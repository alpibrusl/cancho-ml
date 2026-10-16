# cancho-ml as an MCP server: `ml-mcp` (issue #45)

**Status: sections 1 to 9 are written before the code they judge and fix the protocol revision, the tools, what a call does, the bounds, the authority and the gates; a place where the code shows the text wrong is corrected in the text.
Section 10 holds the results and is filled only with what a script of this repository printed, with its conditions. Section 11 lists what is not done.**
Tags as in `design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from a document or from the code, **[measured]** printed by a script here, **[open]** not decided.

`ml-mcp` is the fourth binary built from the sources of `ml` (after `ml-predict`, `ml-serve`; `docs/serving.md` section 3, `docs/serving-http.md`). It speaks the Model Context Protocol over standard input and
output. Its tools are derived from the flag table of `tools/ml/spec.cho` and the schemas in `schemas/`; it holds no description of a flag of its own. A tool call runs the library code of `ml` in the same process: no
subprocess, no `exec`.

## 1. The protocol revision **[read]**

The specification pages at https://modelcontextprotocol.io/specification/ were read for this stage: the latest revision there is `2026-07-28`, which has **no handshake** (every request carries its
version and capabilities in `_meta`; a server must answer `server/discover`). The revision before the handshake went away that holds everything the issue asks for (structured tool results, an output schema, the
`initialize` / `initialized` lifecycle, `notifications/cancelled`, newline-delimited stdio) is `2025-06-18`.

**Pinned: `2025-06-18`** **[choice]**. `scripts/check_mcp.py` and `tests/conformance/mcp_client.py` hold the string. A client that offers another version in `initialize` is answered with `2025-06-18`
(the spec: the server answers the version it supports when it does not support the requested one; the client disconnects if it cannot use it). The modern era of `2026-07-28` (`server/discover`, per-request `_meta`) is
not implemented; a request carrying `_meta` of that era is served as a request of the pinned revision (the unknown member is ignored), and `server/discover` is the JSON-RPC error `-32601`.

Implemented:

| piece | behaviour |
|---|---|
| transport | stdio: one JSON-RPC message per line on standard input, one per line on standard output, UTF-8, no embedded newline in an output message (the writer escapes), nothing but messages on standard output; the usage line and fatal start errors go to standard error |
| lifecycle | `initialize` (answers `protocolVersion`, `capabilities` `{"tools": {"listChanged": false}}`, `serverInfo`, `instructions`), `notifications/initialized`; shutdown is the end of standard input |
| `ping` | answered with `{}` at any time after the handshake began (also before it, as the spec allows) |
| `tools/list` | the whole list in one page; a `cursor` is accepted only if absent or empty (`-32602` otherwise) |
| `tools/call` | section 4 |
| `notifications/cancelled` | read and ignored: section 8.2 |
| errors | JSON-RPC 2.0 error objects: `-32700` parse error, `-32600` invalid request (including a batch and a message over the limit), `-32601` method not found, `-32602` invalid params (unknown tool, a wrong argument type, a call before `initialize` is `-32002`, **[choice]** the code the SDKs use for "server not initialized"), `-32603` internal |

**Batches are refused**: a JSON array is `-32600` with `id: null`. The `2025-06-18` revision removed batching from the base protocol (it was in `2025-03-26`); a server that accepted one would answer in a shape the pinned
revision no longer defines. **[choice, from the changelog as remembered: the page fetched for this stage does not say it; the gate D2 shows the refusal, and a client that sends batches learns it at the first one]**

Not implemented, and declared absent by the capabilities: resources, prompts, completions, logging, sampling (client feature), roots (client feature), elicitation, tasks, progress notifications, `listChanged` notifications,
pagination, streamable HTTP (a second transport would need a listener; the authority row of `ml-mcp` has none, section 7), authorization.

## 2. The tools: derived, not written **[choice]**

Ten tools, one per verb and one per `runs` subcommand: `fit`, `predict`, `eval`, `cv`, `describe`, `introspect`, `runs_list`, `runs_show`, `runs_compare`, `runs_verify`. `skill` is not a tool (it is prose for an agent that
is not speaking MCP). The names and the set come from the `usage` lines of the flag table (`fit:`, `predict:`, ... and the `runs` forms); `scripts/mcp_tools.py` reads the table and emits `docs/mcp/tools.json`. For each tool:

* `description`: the verb's line of `usage`, the summary of the flag table, and the help of each property is the help of its flag (the text of `ml introspect`, not a copy: the generator reads it).
* `inputSchema` (JSON Schema 2020-12): one property per flag the verb takes (`allowed(verb)` of `spec.cho`) except those the server owns (`root`, `format`, `log`, `log-max-bytes`, and the guard flags when the server
  is started read-only), plus the operands: `file` (the CSV), and for `runs_*` `dir` and `id` / `id_a`, `id_b`. By kind: `nat` is an `integer` with `minimum` 0 and, for a flag with an entry in `limits`, the ceiling as `maximum` and the
  default as `default`; `choice:a/b` an `enum`; `bool` a `boolean`; `text` and `path` a `string` (`minLength` 1). `required`: the flags the usage line of the verb writes outside brackets (`fit`: `output`; `target` is optional there because `--model kmeans` has none), the operands the verb needs (`file` for `fit`, `eval`, `cv`; `dir` and the ids for `runs_*`), and `model-file` for `predict`, `eval`, `describe`; the library refuses what is missing as the command line does. `additionalProperties: false`. No conflict between flags is stated in the schema (the schema has no `dependentRequired` or `not`): `plan` against `step`, `split-by` against `stratify` and the hyperparameters a model does not take are `args.conflict` from the library, and the description of each tool says so.
* `predict` takes `file` **or** `records` (section 4.3): `oneOf` over the two.
* `outputSchema`: the envelope of `schemas/ml.v1.json` with the `data` member narrowed to the variants of that verb (chosen by the property names of each variant, `scripts/mcp_tools.py`), self-contained.
* `annotations`: `readOnlyHint` true for every tool but `fit` (and `eval`/`cv` never write unless `track`/`output` are given, which the tools offer only with `--allow-write`); `destructiveHint` false;
  `idempotentHint` true (the same input and seed give the same bytes); `openWorldHint` false.

**The drift check.** `docs/mcp/tools.json` is committed. `scripts/check_mcp_tools.py` regenerates it from the table and compares byte for byte; it fails when a flag, a model or a rule is added to the table and the file is not
regenerated (so a flag added by the refusals or the boosting changes lands with a regenerated file or CI fails). `tools/ml-mcp/tools.cho` (the `tools/list` answer and the call table as constants) is generated from the same
file and checked the same way. The projection is compared with `ml introspect` of the built binary (flags, kinds, defaults, limits), so the three (table, `introspect`, tools) cannot disagree silently.

## 3. Read-only by default **[choice]**

`ml-mcp [--root DIR] [--allow-write] [--max-message-bytes N]` and `ml-mcp introspect | skill`. Without `--allow-write` the server lists only the tools that write nothing: `fit` is absent, and `eval`, `cv` and `predict` do not
offer `track`, `output`, `overwrite`, `dry-run` (`dry-run` is a flag of `fit` only, so it goes with `fit`). A `tools/call` of `fit` without the flag is a **tool
error** (`isError: true`) with the rule `mcp.read-only` and exit code 4 (permission), and a property of the schema that the read-only list does not offer is the same refusal, so that a client that cached the
write-enabled list gets a rule and not a silent write. With the flag, `fit` writes its model file, and `--track DIR` writes run records, only beneath `--root`, by the code of the CLI (`toolbox.place`, no link followed,
`O_EXCL` temporaries). `--root` is required with `--allow-write` (`args.required-flag`): a server that can write must have a root to confine the write to. **[choice]** The server never reads `--log`.

## 4. What a call does

### 4.1 The library runs in process; why the library changes

The command line and the library are joined by `Args`, the capability that holds the command line the process was started with, and by `&static [byte]` slices of it: `cli.text(args, parsed, table, name)` returns a
slice of `argv`, and `Opts` (`tools/ml/verbs.cho`) holds `target`, `drops`, `step`, `positive`, ... as `&static [byte]`. A call arriving as JSON has no `Args`, and a string read from the wire is not `static`; the pinned
compiler has no way to build either (the compiler's document on arguments: argv is written once before any code runs; a struct cannot carry a region). `ml serve` avoided it because it needs numbers only; `ml runs verify`
avoided it with masks and by passing texts to `wire.make` as parameters. A tool call needs all the flags, so **the library is changed so that nothing below `main` reads `Args`**:

* `ml.argv` (`tools/ml/argv.cho`): a parsed command line is one byte slice, `src`, built once by `main` from the real `Args` and `cli.Parsed` (`argv.from_cli`), or by `ml-mcp` from the JSON arguments
  (`argv.from_call`). It holds the verb and operands, each flag given with its value and the argument index it had, and the argument vector (for the argv a repair repeats). `argv.has`, `argv.text`, `argv.nat`,
  `argv.operand`, `argv.operand_count`, `argv.value_index`, `argv.value_is_whole` have the signatures of the `cli` functions they replace, with `src` for `(args, parsed)`.
* Every function that took `(args, parsed)` takes `src`. `Opts` no longer carries a text; the texts are read from `src` where used. The paths are checked by `ml.paths` over the text (the checks of
  `toolbox.path` that take text: `longest`, `has_dotdot`, `under`, `lexical`), producing the same rule tags and details.
* Nothing else changes: the verbs' answers are the same bytes (the conformance suite of the CLI runs unchanged, and the goldens are not touched), the binaries of the CLI are built from the same sources with the same
  authority rows (`args` stays in `main` only).

So a call is **the same checks, the same code, the same refusals**: the call is turned into the `src` its equivalent command line would parse to, and the argument vector in it is that command line
(`ml --root DIR fit --target y ... FILE`, `argv[0]` is `ml`), so that a `repair` of kind `retry` is a command the agent can run with the CLI and `ml-mcp` can run again after dropping the flag.

### 4.2 The result

`structuredContent` is the document the CLI prints on one line (`schemas/ml.v1.json`: `ok`, `command`, `schema`, `data` or `errors`, `meta`), parsed. `content[0]` is `{"type":"text","text":...}` with the text form
(`--format text`) of a success, and for a refusal the document itself (the sentences of a refusal are what `--format text` writes to standard error, which a tool has none of). Byte equality is tested on the document: the server builds it with the same function (`toolbox.out.document`) from the same
buffers, and the test client compares the **exact bytes** of the `structuredContent` member of the response line with the line the CLI prints (gate A).

* A **success** is `isError: false`, with `_meta` `{"ml/exit": 0}`. A dry run (`fit` with `dry-run`) is a success too: the document says `dry_run: true`, and `_meta` says `{"ml/exit": 9}`, the exit code of the equivalent command **[choice]**.
* A **refusal** (`ok: false`: any `errors` of the CLI: `args.*`, `path.*`, `io.*`, `ml.*`, `limit.*`, `parse.*`, `conflict.exists`) is a **tool result** with `isError: true`, `structuredContent` the document with `error` / `errors`
  (rule, code, message, hint, repair, detail) **exactly** as the CLI prints it, and `_meta` `{"ml/exit": N}` with the exit code of its first error.
* A **protocol problem** is a JSON-RPC error: malformed JSON `-32700`; not a request object, a batch, a message over the limit, an `id` that is an object or an array `-32600`; unknown method `-32601`; unknown tool, `arguments`
  not an object, an argument of the wrong JSON type (a string for a `nat`, a number for a `path`), a property the schema does not list, `-32602`. The `data` of the error holds the JSON Pointer of the first fault.
  A property outside the schema is **not** the CLI's `args.unknown-flag`: the JSON-RPC layer owns the shape; the library owns the values (ranges, conflicts, existence).
* The tool-level checks the schema cannot state come from the library: `ml.param.out-of-range` with the range in `detail`, `args.conflict`, `args.required-flag`, `path.outside-root`, `path.symlink`, `io.not-found`.

### 4.3 Paths and inline records

Paths are relative to `--root` and resolved as in the CLI: checked lexically (`path.empty`, `path.too-long`, `path.dotdot`, `path.absolute`, `path.outside-root`) and opened one component at a time with no link followed
(`path.symlink`). Without `--root` a path is the working directory's, as for the CLI (`confinement: none` for readers) **[choice]: `ml-mcp` without `--root` refuses `--allow-write`, and a read-only server without a root reads
as the CLI does**.

`predict` takes `records`: a JSON array of at most `--max-batch-rows`-style bound (`limit`, default 1000, ceiling 1,000,000, the `--limit` of the verb) objects. Each is rendered to the CSV the predictor reads by the **byte rule** of
`docs/serving-http.md` section 4, with the code of `tools/ml-serve/cols.cho` and `num.cho` (compiled into `ml-mcp` from the same files, not copied): the header is the columns the model requires (from `describe`, in the
model's order), a number is judged from its text and never from a double (`serve.number-not-exact`), a string with CR LF is `serve.string-not-renderable`, `null` and an absent key are an empty cell. The two rules are added to
`ml-mcp`'s rule list and are tool errors with the JSON Pointer in `detail`, as in `ml-serve`. The CSV is given to `wireio.predict` as bytes in memory (the plumbing of `predict -`).

## 5. Bounds **[choice]**

| limit | flag | default | ceiling | refused with, before allocation |
|---|---|---|---|---|
| one message | `--max-message-bytes` | 1,048,576 | 16,777,216 | no more than the limit is kept; the rest of the line is read and dropped as it arrives; the answer is the JSON-RPC error `-32600` (the message names `--max-message-bytes`) with `id: null` |
| JSON nesting | the depth limit of `std.json` | its own | | `-32700` (parse error) |
| inline records | the `limit` of `predict` | 1000 | 1,000,000 | `ml.param.out-of-range` |
| every limit of the CLI | `max-rows`, `max-bytes`, `max-line-bytes`, ... | as `introspect` | as `introspect` | by the library, with its rule |

One request at a time: the server reads a line, answers it, reads the next. Requests per second are not limited.

## 6. Cancellation and time **[choice]**

### 6.1 `notifications/cancelled`
The server has one thread and reads its input only between calls. A `notifications/cancelled` that names a call in flight is read after the call has answered: it names a request that is no longer in progress, which the
specification lets a receiver ignore, and the server does. A cancellation that arrives before the request it names is also ignored (unknown id). The call is never interrupted: `fit` of a large file runs to its end or to the
limit that bounds it. This is stated, tested (the client sends the cancellation after the request and checks that the answer arrives and the server lives) and listed in section 11.

### 6.2 `--timeout-seconds` is not offered
A deadline needs the clock, and a clock is a label (`clock`) the authority row of `ml-mcp` does not have; adding it for a deadline that cannot interrupt a running call would widen the row for nothing. The work of one call is bounded by the
limits of the library (rows, columns, cells, bytes, trees, nodes, depth), not by time. A host that needs a deadline kills the process, which is how an MCP stdio server is shut down (the spec's `SIGTERM` step).

## 7. Authority **[choice]**

`ml-mcp` is a `[[bin]]` of `cancho.toml` with the file list of `ml` (all of `tools/ml` except `main.cho`, since `ml-mcp` has its own) and `tools/ml-mcp`, plus `tools/ml-serve/cols.cho` and `num.cho`. Its row is derived by
`cancho authority` and committed as `manifests/ml-mcp.authority.json`. Expected: `args` (its own flags), `conc` (the `fit` and `cv` code it shares with `ml` spawns the threads of the numeric kernels; the row of `ml` has it for the
same reason and the flag `threads` is offered), `dir_read`, `dir_write`, `err_write`, `file_read`, `file_write`, `fs_read("")`, `heap`, `io_read` (the protocol is read from standard input), `io_write`. Not in the row: `net_in`, `net_out`,
`conn_*`, `poll`, `clock`, `exec`, `ffi`, `env`, `fs_write`. `tools.toml` has the ceiling `[ml-mcp]` with a reason for each label beyond the base row. `scripts/check_bins.py` is extended (the file list of a fourth binary, the row
against its ceiling and against the committed file), and its mutants include one that adds a capability to the derived row and one that adds `net_in` to the reachable code: both are refused.
Admission (`scripts/admit.py`) is for models: `ml-mcp` is not one.

## 8. The gates (written before the code; each must be able to fail)

| id | gate | script | mutants that must be killed |
|---|---|---|---|
| A | a protocol client (Python, standard library only) starts `ml-mcp`, does the handshake, lists the tools, calls **every** tool on the bundled data and compares `structuredContent` to the CLI's JSON of the equivalent command: parsed equal and the canonical text equal | `tests/conformance/mcp_client.py`, `tests/conformance/test_mcp.py` | the id not echoed; `isError` always false; a key dropped from the document; the exit code wrong in `_meta` |
| B | the generated tools equal `introspect` and the table (drift); every `inputSchema` and `outputSchema` is a valid Draft 2020-12 schema (`jsonschema`'s meta-validation); a generated valid argument set per tool is accepted by the schema and by the server, an invalid one is refused by the schema and, sent to the server directly, by the library's own refusal; the real output of every tool validates against its `outputSchema` | `scripts/check_mcp_tools.py`, `tests/conformance/test_mcp_schema.py` | a flag dropped from the generator; a kind mapped to the wrong type; an enum changed; a ceiling off by one |
| C | every class of refusal reaches the client as data with `rule`, `code`, exit in `_meta` and the repair; `runs_verify` of a record runs and answers `verified: true`; the read-only server refuses `fit` with `mcp.read-only` | `tests/conformance/test_mcp.py` | the read-only switch removed; the repair dropped |
| D | malformed and hostile input: invalid JSON; a message over the limit; a batch; an unknown method; a call before `initialize`; a wrong argument type; the same id twice; nesting past the depth; invalid UTF-8; an `id` of each JSON type (number, string, null, true, object, array, float); input in one-byte chunks; EOF in the middle of a message; a call whose CLI exit is each of 0, 2, 3, 4, 5, 8, 9: every one gets an answer of the contract (or, for a notification, none), the process neither traps nor hangs, and the next request is served | `tests/conformance/test_mcp_hostile.py` | a trap on an empty line; the size check removed; the batch accepted; the id coerced |
| E | mutants of the server: the framing, the id echo, `isError`, the schema generation, the read-only switch, the root resolution, the inline-records rendering, the limits; run on a copy of the tree; baselines pass first | `scripts/mutants_mcp.py` | all killed or each survivor explained in section 10 |
| F | the authority: the derived row equals the committed file and is inside the ceiling; the mutants of `check_bins.py` are refused | `scripts/check_bins.py`, `scripts/manifest.py` | a capability added to the row; a file list that drifts |
| G | CI: the fast job runs `scripts/check_mcp.py` (the protocol client, under a minute) and the drift check; the numpy-free runner runs the conformance without an oracle | `.github/workflows/ci.yml`, `scripts/check_workflow.py` | a step removed |
| H | a real session (`initialize`, `tools/list`, `fit`, `predict` with inline records, `runs_list`) is generated into the documentation (`gen:mcp-session` of `scripts/site.py`) and checked against a fresh run | `scripts/site.py --check` | a byte of the transcript changed |

The library change of section 4.1 has its own gate: every existing conformance test and golden of the CLI, `ml-predict` and `ml-serve` passes unchanged (G of `docs/serving-http.md`), and `scripts/check_bins.py`
shows the three older rows unchanged.

## 9. Not done

Section 11 lists what the server does not do and what was not run.

## 10. Results

Everything below was printed by a script of this repository on one machine: a Mac (arm64, Apple M4 Max, macOS 26.2) with compiler `0207556`, shared with other sessions, at a load average of 70 to 95 while the
scripts ran. A time is therefore an upper bound for a quiet machine. The Linux runs wait (section 11).

| gate | what ran | result |
|---|---|---|
| A, the fixtures through the tools | `tests/conformance/test_mcp.py` `CaseGate`: each of the 339 fixtures of the command line that can be run as a plain command is run by the command line with the arguments in the order the server builds them (`argv[0]` `ml`), by the server and by the command line in `--format text`, in one scratch directory wiped between the runs | 303 fixtures can be said as a tool call; for all 303 the `structuredContent` is the line the command line printed, byte for byte (repairs with their argv included), `_meta` `ml/exit` is the exit code, `isError` is whether the command refused, and for a success `content[0].text` is the `--format text` output. By exit code: 0 (86 fixtures), 2 (84), 3 (11), 4 (4), 5 (1), 8 (112), 9 (5). The 36 that cannot be said are a flag the server owns (`--format`, `--log`, `--root`), a form the JSON does not hold (`--flag=value`, `-`, an unknown flag) or standard input and output closed. Exit 1 has two fixtures and neither can be said as a call, so it is not tested through a call. 23 s |
| B, the tools | `scripts/check_mcp_tools.py --introspect`: the 10 tools (9 read-only) regenerate to the committed bytes of `docs/mcp/tools.json` and `tools/ml-mcp/tools.cho`, every `inputSchema` and `outputSchema` passes the Draft 2020-12 meta-schema, the kind, role, default, ceiling and help of every property equal `ml introspect`, every flag of `introspect` is the property of a tool or one the server owns, a valid argument set (the required ones; every property at its lower and at its upper boundary) is accepted, and a wrong type, a value outside an enum, a negative number, a number above its ceiling, a malformed run id, a missing required property and an unknown property are each refused; `test_mcp_schema.py`: the same sets sent to the server get a tool result and never a protocol error, a number above its ceiling is refused with the refusal of the command line byte for byte (57 properties), and the real answers of 21 calls and 6 refusals validate against their `outputSchema` | pass; `--mutants`: 8 of 8 edits of the generator or of the file are refused |
| C, refusals as data | `test_mcp.py` `Tools`: 12 refusals of 11 rules with rule, exit code in `_meta`, hint and repair; the read-only refusal; `runs_verify` of a record answers `verified: true`; the repair of a refusal is a command the command line runs | pass |
| D, hostile input | `test_mcp_hostile.py`: 28 tests: invalid JSON (12 forms), non-messages, blank lines, a batch, unknown methods, a call before `initialize`, `initialize` with another revision, 17 wrong argument sets and 8 wrong `params`, a key twice, the same id twice, an id of each JSON type (7 echoed as written, 9 refused with `id: null`), the `jsonrpc` member, notifications and response-shaped messages (no answer), cancellation, the size limit at `cap - 1`, `cap`, `cap + 1` and 1,000,000 bytes, the default limit, the limit flag, nesting to 100,000, a million elements, invalid UTF-8 (6 forms), NUL and control bytes, CR LF, one-byte chunks, EOF in the middle of a message, EOF at once, one call for each exit code of the command line that a call can reach, and 400 random lines assembled from 25 fragments, each followed by a ping | pass; the process never traps or hangs, standard error stays empty |
| E, mutants | `scripts/mutants_mcp.py`, 51 mutants of the framing, the id, the envelope, the lifecycle, the arguments, the read-only switch, the root, the records, the generator and the authority, on a copy of the tree with every built executable carried | @MUT@ |
| F, authority | `scripts/manifest.py --check`; `test_authority.AuthorityMcp`; `scripts/check_bins.py` (17 mutants refused, 6 of them for `ml-mcp`) | the row of `ml-mcp` is `args`, `conc`, `dir_read`, `dir_write`, `err_write`, `file_read`, `file_write`, `fs_read("")`, `heap`, `io_read`, `io_write`; bounded; no foreign symbol; the rows of `ml`, `ml-predict` and `ml-serve` are unchanged; three mutants of the authority (the clock read in `main`, a ceiling narrower than the row, a network label in the ceiling) are refused |
| G, CI | `scripts/check_mcp.py` runs the 48 tests of the four modules above (`test_mcp`, `test_mcp_schema`, `test_mcp_hostile`, `test_authority.AuthorityMcp`); `scripts/check_workflow.py` and its 26 mutants | 48 tests in 32 s; the workflow checks pass |
| H, a real session | `scripts/site.py` runs `initialize`, `tools/list`, `fit` with `track`, `predict` with inline records and `runs_list` against `ml-mcp` and writes the lines into the README and the guide; `site.py --check` compares them with a fresh run | current |

The change to the library (section 4.1) has its own evidence: the determinism gate `scripts/xplat_check.py --require-predict` prints 88 lines for 22 fits equal to the committed golden (darwin, arm64; `ml-predict` equal to `ml`), so every model file and
every answer of the 22 fits is the byte sequence the golden holds; the 343 fixtures and the golden answers of the conformance suite pass on the three binaries; `ml`, `ml-predict` and `ml-serve` have the authority rows they had.
@CONF@

Sizes: `tools/list` is 126,876 bytes with `--allow-write` and 91,883 bytes without (compact JSON; the output schemas are 64 percent of the first and 67 percent of the second); `build/ml-mcp` is 1,682,504 bytes and `build/ml` 1,396,288.

## 11. Limits and what is not done

* **A running call is not cancelled and has no deadline.** The server reads its input between calls; `notifications/cancelled` for a call in progress is read after the answer (tested). There is no clock in the authority row, so
  no `--timeout-seconds`. A host that needs a deadline kills the process. The work of a call is bounded by the limits of the library.
* **Revision.** `2025-06-18` only. The stateless revision `2026-07-28` (`server/discover`, `_meta` on every request) is not implemented. A JSON-RPC batch is refused with `-32600`; the pinned revision removed batching
  (from the changelog as remembered, not from a page fetched for this work).
* **Not implemented:** resources, prompts, completions, logging, sampling, roots, elicitation, tasks, progress notifications, `listChanged` notifications, pagination, authorization, the streamable HTTP transport.
* **`content[0].text`** is the text form of a success and the document for a refusal; the specification asks a tool that returns structured content to also return its serialisation as text, which a refusal does and a success does not.
  `_meta` carries `ml/exit`; the name is a **choice**.
* **The size of `tools/list`** is 92 to 127 KB: a client that keeps the list in a prompt pays for it. The output schemas are narrowed to the variants of the verb and self-contained.
* **`introspect`** answers the document `ml introspect` prints with the authority row of `ml-mcp` (the row of `ml-mcp`'s own binary); the other members are equal (tested).
* **`--log` and `--format` are not offered.** The server owns the format (the text form is in the result); a log file is the command line's.
* **Exit code 1** (the operating system failed) is not reached by a test through a call: its two fixtures close standard output or read a device, which a call cannot state.
* **`toolbox.path` is not called by `ml`.** Its `root` and `operand` read `Args` by index, so `tools/ml/paths.cho` has their rules over text (about 100 lines, the pinned `lexical`, `under`, `has_dotdot` and `longest` doing the work).
  The conformance fixtures of the path rules and the fuzz of the arguments pass on the result; the pinned functions and the port were not run side by side on one corpus.
* **What was not run here (Linux and the long suites):** the full `scripts/check.sh`; the conformance modules that need numpy, scipy or scikit-learn; `ST_STRIDE=1` of `test_describe`; the mutant scripts of the older gates
  (their patterns apply: each `--check` passes, but the 1,542 mutants of their last runs were not rebuilt after the change of section 4.1); the `full-gate` of the workflow and the determinism fingerprint on Linux x86-64 and aarch64;
  `scripts/check_web_pin.py --bins`. `docs/status.md` lists them for the Linux run.

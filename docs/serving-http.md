# cancho-ml serving over HTTP: `ml-serve` (issue #16)

**Status: this document is written before the code it judges. Sections 1 to 9 fix the dependency, the endpoints, the byte rule, the limits, the authority and the gates; section 10 holds the
results and is filled only with what a script of this repository printed, with its conditions; section 11 lists what the stage does not do.**
Tags as in `design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from the code or from a document of a pinned dependency, **[measured]** printed by a script
here, **[open]** not decided.

`ml-serve` serves ONE trained model file over HTTP with cancho-web. It is a third binary, built from the same sources as `ml` and `ml-predict` (`docs/serving.md` section 3), with its own
`main`. It is not a subprocess of `ml predict` (that would need `exec`): it calls the library code of `ml` in the same process, with the CSV and the model file as bytes in memory.

## 1. The dependency: cancho-web, cancho-schema and `http.server`, as pinned packages

### 1.1 What was found

* **cancho-web publishes `web` as a store [read, measured].** `alpibrusl/cancho-web` at `5f404a7` has `.cancho-vcs` (the store of `src/web.cho`, published by `scripts/publish.sh`, whose `--check` CI runs);
  `web` imports `schema`, so its `requires/` records `cancho-schema` at `59a70285ff23af63a226bc56ff978feccb4cf8ca` (the `SCHEMA_REV` of its `ci.yml`).
* **cancho-schema publishes `schema` as a store [read, measured].** `alpibrusl/cancho-schema` has `.cancho-vcs` at `59a7028` (the commit its store was last republished at; later commits do not change it).
* **`http.server` is a store of the compiler's repository [read, measured].** `alpibrusl/cancho` at `a4572ea3b506e4bcec1c4be43deda1043c11ba9f`, the compiler these sources are written for, has
  `packages/http-server/.cancho-vcs`.
* **A project pins the three the way it pins `cancho-table` and `cancho-tools` [measured].** A scratch project outside this repository with the three `[dependencies.*]` blocks below ran `cancho install`
  (`installed schema (59a70285ff23): 1 file(s)`, `installed server (a4572ea3b506): 1 file(s)`, `installed web (5f404a777018): 2 file(s)`), imported `http.server`, `schema` and `web` and
  built with compiler `a4572ea`. **No upstream change is needed and nothing is copied**; there is no LOCAL-DEV arrangement.

### 1.2 The pin

```
[dependencies.web]
git = "https://github.com/alpibrusl/cancho-web"
rev = "5f404a777018cf42b5c713daa4972c938c9229f4"
path = ".cancho-vcs"

[dependencies.schema]
git = "https://github.com/alpibrusl/cancho-schema"
rev = "59a70285ff23af63a226bc56ff978feccb4cf8ca"
path = ".cancho-vcs"

[dependencies.server]
git = "https://github.com/alpibrusl/cancho"
rev = "a4572ea3b506e4bcec1c4be43deda1043c11ba9f"
path = "packages/http-server/.cancho-vcs"
```

`web` records its requirement on `schema` at `59a7028`, and a consumer must also name `schema` in its own `cancho.toml` (cancho-web's `scripts/publish.sh` says so); the two revs are the same. The module of the
`server` store is `http.server`. `scripts/check_web_pin.py` (gate W1) checks that the three blocks have the git URLs above, a 40-digit rev and the path, that the `schema` rev is the `SCHEMA_REV` this document
names, that the `server` rev is the compiler pin of `[package] cancho`, that no other store of these three repositories is listed, that no `.cho` file under `lib/`, `tools/` or `tests/` declares the modules `web`,
`schema` or `http.server`, and (`CHECK_REMOTE=1`) that GitHub has the three stores at those revs. Moving a pin is the act of editing the revs in `cancho.toml` and the lines of this section together.

cancho-web rev: `5f404a777018cf42b5c713daa4972c938c9229f4`

cancho-schema rev: `59a70285ff23af63a226bc56ff978feccb4cf8ca` (the `SCHEMA_REV` `5f404a7`'s `ci.yml` names: `SCHEMA_REV` `59a70285ff23af63a226bc56ff978feccb4cf8ca`)

cancho http-server rev: `a4572ea3b506e4bcec1c4be43deda1043c11ba9f`

Only `ml-serve` imports these modules; `cancho build --bin ml` and `--bin ml-predict` are not changed by them (gate W2 compares the bytes of the two binaries built before and after the three blocks
are added; the other changes of this stage that reach their sources are listed in section 10).

### 1.3 What the pinned libraries do, read before the design

* **One thread.** `http.server` is one `Poller` loop (`wait`, then `next` until it answers -1, `respond`). Requests are answered one at a time, in order, per connection; pipelined requests are supported **[read]**.
* **Limits of the loop [read].** The buffer of one connection (`size` in `server.open`) holds the head and the body: a request whose head and body do not fit is answered by the loop itself, before the handler, `413`
  (`{"error":"request too large"}`) or, for a head with no end in `size` bytes, `431`; a malformed head is `400`. These three bodies are `application/json` `{"error": ...}`, not `problem+json`: **the loop
  writes them, and the pinned package has no hook for them** (section 5.3 says what that means for the documented refusals). The idle timeout of a connection is an argument of `server.open` (seconds; it reads the
  clock, so the row has `clock`). At most 1,024 connections, and 256 MiB of buffers in all: a larger `size` means fewer connections.
* **The output queue of a connection is 65,536 bytes [read].** An answer is handed to the kernel at once; what the kernel does not take is queued, and **a connection whose queue would hold more than 64 KiB is
  closed**. So every answer of `ml-serve` is bounded to 60,000 bytes by construction (section 5.2) and the documents that could be larger are refused or reduced at start (sections 3.5 and 5.2).
* **`tcp_listen` binds every address of the host [read].** `cancho` `crates/cancho-codegen/src/body/net.rs` (`bind`): the socket address is `INADDR_ANY`, the bound of the capability is the port alone, and there is no host argument.
  **The pinned compiler cannot bind `127.0.0.1`.** Section 5.4 is the consequence.
* **Numbers [read].** `std.json` parses strictly (RFC 8259: no `NaN`, no `Infinity`, no BOM, no trailing comma; a control character in a string, invalid UTF-8 and nesting past its depth limit are errors; the
  error carries the byte position) into a tape whose number nodes point at the number's text; cancho-schema's `integer` accepts a whole number written as a float (`150.0`, `1.5e2`) and its `number` accepts any JSON number.
  A duplicate key in an object is not an error of either (`json.get` finds one of them). Section 3 does not rely on either for the bytes of a cell.

## 2. The surface

```
ml-serve --model-file MODEL [--root DIR] [--port N] [--token-file FILE | --no-token] [--max-body-bytes N] [--max-batch-rows N] [--idle-seconds N] [--max-bytes N]
ml-serve introspect | skill
```

One model file per process, read once at start. **Reload is a restart** (**[choice]**; the optional guarded `POST /admin/reload` of the issue is not built: nothing in the pinned loop makes a swap
atomic without a second copy of the model, and a service that wants zero downtime starts the new process on the new file behind a proxy). `--model-file`, `--token-file` and `--root` are checked as every path of the
family is (`toolbox.path`: below `--root`, no `..`, no symbolic link followed).

| flag | meaning | default | ceiling |
|---|---|---|---|
| `--model-file` | the model file (required) | | |
| `--root` | resolve `--model-file` and `--token-file` beneath this directory | none | |
| `--port` | the TCP port (1 to 65535) | 8080 | 65535 |
| `--token-file` | a file holding the bearer token (section 5.4) | none | |
| `--no-token` | serve with no token, on every address of the host (section 5.4) | off | |
| `--max-body-bytes` | the largest request body | 1,048,576 | 16,777,216 |
| `--max-batch-rows` | the most records in one batch | 100 | 1,000 |
| `--idle-seconds` | a connection that makes no progress for this long is closed | 10 | 3,600 |
| `--max-bytes` | the largest model file | 16,777,216 | 268,435,456 |

## 3. The endpoints

Each is one declaration with `web.operation` and its parameters, body and responses, and the document at `/openapi.json` is generated from them by `web.openapi` at start (`docs/design.md` of cancho-web).

| operation | route | request | success |
|---|---|---|---|
| `predictOne` | `POST /predict` | one record: a JSON object with one member per request column | 200, section 3.2 |
| `predictBatch` | `POST /predict/batch` | `{"rows": [record, ...]}`, 1 to `--max-batch-rows` records | 200, section 3.2 |
| `getModel` | `GET /model` | none | 200, the answer of `describe` (section 3.3) |
| `getOpenapi` | `GET /openapi.json` | none | 200, the document |
| `health` | `GET /health` | none | 200, `{"ok":true,"model_sha256":"..."}` |

**[choice] Two predict routes, not one.** The issue's text gives one route taking a record or `{"rows": [...]}`. A schema node of cancho-schema is one type: there is no `oneOf`, and a body declared
as "an object" would document no column at all, which is the point of the document. A model may also have a column named `rows`. So the record and the batch are two operations with two body schemas, and
the OpenAPI document says exactly what each accepts. Neither takes a query parameter: **[choice]** there is no `proba` flag; the answer always carries what the model has (3.2).

### 3.1 The request schema, from `describe`

At start the server calls the code of `ml describe` on the model bytes (the same function, section 6) and reads the answer (`docs/serving.md` section 5); the schema is built from these fields only:

* **The record** is an object, `additionalProperties: false` (`extra_columns` is `"refused"` in every file; the server has no `--drop`). A field per column of `request.header`, in the file's order, required, named by the
  column. A column whose name is not UTF-8 (a `{"b64"}` name) cannot be a JSON key: the model is refused at start (`serve.column-name`, section 5.6). The optional columns of `request.optional` (the target) are
  declared as optional members of any JSON type, **and ignored**: the server does not put them in the CSV (`describe` says their content is ignored).
* **`int` column:** `integer`, with the bounds ±999,999,999,999 (the 12 digits of the reader's `int`, `docs/reader.md` section 2.1). **`dec` with scale `s`:** `number`. **`float`:** `number`.
* **`categorical`:** `string`. When `categorical[].truncated` is false, the encoder is not `hash`, every `values[].value` is UTF-8 text and the document stays within its bound (3.5), the node has the `enum` of
  the file's categories: **a string outside the enum is refused (`choice`, 422)** for such a column. For a `hash` column, a column with a truncated list and a column with a non-UTF-8 category there is no `enum`, and any
  string is accepted and counted by the predictor as `unknown_categories` when it is not in the vocabulary, as the command line does. **[choice]** The file's own vocabulary is the contract a model
  was trained on, and a request outside it is a request the model never saw: where the file lists the whole vocabulary the document says so and the server refuses; where it cannot, the CLI's behaviour stands.
* **Nullable** exactly when `request.missing_cells` is `"handled"` (tree and forest): the node accepts `null`, which is an empty cell. For `"refused"` and `"needs_impute"` no column is nullable (a plan can impute
  a column, and `describe` does not say which: the document says the strict thing). An empty string in a categorical column is an empty cell too, and so is a string equal to one of `na_values`: the predictor reads them as
  missing, as it does from a file.
* **The batch** is `{"rows": array of records}`, `minItems` 1, `maxItems` `--max-batch-rows`, strict.
* A model with a column named `rows` is **served**: the record and the batch are different routes.

### 3.2 The answer

```
{"model_sha256":"<64 hex digits>","model":"linear","task":"regression","columns":["prediction"],"rows":[[36.17963203363425]],"row_count":1,"unknown_categories":0}
```

* `columns` is `describe.output.columns` for a regression (`["prediction"]`), `["label","probability"]` for every classification (the probability of the positive label; a constant classification has both
  always, the others are asked for with the `--proba` option of the same code), `["cluster","distance"]` for a clustering (the squared distance to the centre of its cluster, the `--distance` option).
* Each row is an array with a value per column. **The value is the text the command line prints, as a JSON number**: the document of `ml predict` holds each value as a string made by
  `flt.put_float` (shortest decimal that reads back to the double, positional from 1e-6 to 1e21 with a point, scientific outside) or `exact.put_scaled` (a label, an integer), and the server writes the same string
  between no quotes after checking that it is a JSON number (`json.put_fragment` parses it: a text that is not one is `serve.internal`, 500, never a quoted string). The decimal text is the same; nothing is printed twice. 
* `model_sha256` is `describe.file.sha256` and the response header `X-Model-Sha256` carries it too.
* `unknown_categories` is the predictor's count over the request.
* Every answer is at most 60,000 bytes; the 1,000-row ceiling and the values (at most about 56 bytes a row) make that true by construction, and the server checks it.

### 3.3 `GET /model`

The `data` object of `ml describe --model-file` for the served file, as it is. Its size is bounded to 60,000 bytes by choosing, at start, the largest `--limit` of 1,000, 100, 10, 1 and 0 whose answer fits; the
fields `categorical[].listed` and `truncated` say what was cut. `describe.file.sha256` is the digest.

### 3.4 `GET /openapi.json`

The document `web.openapi` makes from the declarations, OpenAPI 3.1, title `ml-serve`, version `0.1.0`, with the components `Record`, `Batch`, `Prediction`, `Problem` (3.4.1). For a fixture model it is a checked-in
file (`docs/serving/<fixture>.openapi.json`, gate O1); a model change is a change to that file. The document is held to be what the server does by the end-to-end tests (every documented response is
validated against it) and, if the tool can be installed, by Schemathesis.

#### 3.4.1 The refusals in the document

`Problem` is an object with `type`, `title`, `status` (required), `rule`, `detail`, `count`, `errors` (`pointer`, `code`, `detail`) and `error` (any). Documented per operation: 200; 401 and 403 when a token is set (`web.require`);
400, 413, 415, 422 and 500 on the predict routes; 500 on `GET /model`. The 404 and 405 of the router are cancho-web's own; the 413, 431 and 400 of the loop are the loop's (section 5.3).

### 3.5 Sizes of the documents

A document, or the `GET /model` answer, larger than 60,000 bytes cannot be sent safely (section 1.3). At start: the document is built with every `enum`; if it is larger than 60,000 bytes it is built again
with none (and the schema the requests are judged by is the schema without them: the document and the validator are one object); if it is still larger, the model is refused at start (`serve.document-too-large`).

## 4. The byte rule: how a record becomes the CSV the predictor reads

The server renders each record to the CSV text that the command line would read, and gives it to **the same function as `ml predict`**: `wireio.predict` (for the model kinds that module takes) or `verbs.predict` (the
constant models, the older reader), with the CSV as bytes in memory (`paths.Rp` holding the bytes, `csv.Source::Mem`: the plumbing of `predict -`) and the model file as bytes in memory (the same `Rp` carrying the bytes
the server read at start). No temporary file, no subprocess, no `exec`. The answer document is the document `ml predict` prints, parsed, and its values are written again between no quotes.

**The CSV.** The header is `request.header`, joined by commas, each name written between double quotes with each quote doubled when it holds a comma, a quote, CR or LF (RFC 4180); then one line per record, in the
request's order, with one cell per header column, then LF. The cells of a column, by its type:

| column | the cell | refused with 422 (`serve.number-not-exact`, the JSON Pointer of the value) |
|---|---|---|
| `int` | the integer's decimal digits, `-` for a negative, no leading zeros, `0` for zero (`-0` is `0`) | a number that is not exactly an integer of at most 12 digits: `1.5`, `1e13`, `150.0000000000000000001` |
| `dec`, scale `s` | the value with exactly `s` digits after the point (`s = 0`: no point) | a number with more than `s` significant digits after the point, or more than 12 digits in all (`dec` is an exact 12-digit decimal) |
| `float` | `flt.put_float` of the nearest double (ties to even; the shortest decimal that reads back to it); `0.0` for any zero (the reader has no negative zero) | a number past the range of a double (`1e999`), or non-zero and smaller than the smallest subnormal |
| `categorical` | the string as UTF-8, between double quotes with quotes doubled when it holds a comma, a quote or LF | a string with a CR or NUL (a CSV reader does not keep a CR inside a quoted field: the cell could not be read back as sent) |
| any, `null` | an empty cell | |

The number is judged from its text (the tape's byte range), by a decimal reader of its own that keeps the digits and the exponent, and never from a double: a request is never rounded to a number it did not
send. A `float` column is a double, and the nearest double to the text is the value of the cell; the refusal for it is range only. **[choice]** A whole number written `150.0` or `1.5e2` is accepted for an `int`
column (cancho-schema's rule, and the document says `integer`) and rendered `150`.

Names and strings reach the predictor only through this rendering, so a name with a newline or a quote cannot add a column or a row.

### 4.1 The byte-equality gate (S1)

For a corpus of stored models of all kinds (the 19 models of `tests/conformance/serving_corpus.py`, kinds 1 to 7, k-means included, categorical, hash and missing-cell cases) and records taken from the
corpus' input CSVs: for every row, the value of each column in the HTTP answer is **text-identical** to the cell of `ml predict` on the same data. The comparison is exact: the CLI's rows are
`rows[i][j]`, strings, of the document of `ml predict --model-file M --proba|--distance CSV`; the server's are the JSON number tokens of the response body, extracted by a JSON reader that keeps number tokens as
text (`parse_float=str`, `parse_int=str` in Python), and `str == str` is the test. A record is built from the CSV row by the *column type* (an `int` cell is sent as a JSON integer, a `dec` or `float` cell as the
JSON number with the same text, a categorical cell as a string, an empty cell as `null` for a tree or forest and omitted from the comparison otherwise). The mutants that must fail S1: the number formatting changed in
the server's rows (a digit dropped, a quote added), the column order of the CSV (two columns swapped), the null rendering (an empty cell written as the string `null`), the `dec` scale padding, the quoting of a category.

## 5. Safety

### 5.1 One thread, bounded everything

Single thread (the `Poller` of the loop); a request is parsed and answered before the next. Bounded: the body (`--max-body-bytes`, enforced by the loop's buffer: head plus body past it is 413), the head
(16,384 bytes, the loop's `431`), the rows (`--max-batch-rows`), the cells (rows times columns, against `--max-cells` of the predictor, set to the product of the model's columns and the row limit), the JSON
nesting (`std.json`'s depth), the time of an idle connection (`--idle-seconds`), the number of connections (1,024 less what the buffers need). There is **no request timeout beyond the idle one**: the pinned loop
has no deadline per request, and a handler that runs long (a large forest over 1,000 rows) holds every other connection; the load test (section 10) measures the time of a request at the ceiling.

### 5.2 The size of an answer

See 1.3 and 3.2. A batch of the ceiling's 1,000 rows of two columns is at most 1,000 times about 56 bytes; the server checks the final length against 60,000 and answers 500 `serve.internal` otherwise (it cannot happen: the check is a
backstop with a mutant).

### 5.3 The refusals and their statuses

Refusals are `application/problem+json` (RFC 9457) with a `rule` field, **except the three that the loop and the router write before the handler is reached**: `400` (malformed HTTP head), `413` (head and body past the
buffer), `431` (head too large) are `application/json` `{"error": "..."}` from `http.server`, and `405` is `application/json` `{"error":"method not allowed"}` with `Allow` from cancho-web's `dispatch` **[read]**. Everything
else, including the `404` for a path that is not a route (`web.dispatch`), is `problem+json`.

| status | when | `rule` |
|---|---|---|
| 400 | the body is not JSON (syntax, a BOM, invalid UTF-8, NaN or Infinity, a control character, nesting past the depth limit, trailing bytes), or has a duplicate key in a record | `serve.body-not-json`, `serve.duplicate-key` |
| 401, 403 | a token is set and the request has none, or an unknown one (401), or one that does not list the scheme (403): `web.guard`'s answers | none (cancho-web's body) |
| 404 | the path is not a route | none (cancho-web's body, `problem+json`) |
| 405 | the method is not allowed on the route | none (cancho-web's `{"error"}` with `Allow`) |
| 413 | the batch has more than `--max-batch-rows` records (`serve.batch-too-large`); the body is larger than `--max-body-bytes` (the loop, no `rule`) | |
| 415 | `Content-Type` of a predict POST is not `application/json` (parameters allowed) | `serve.media-type` |
| 422 | the body breaks the request schema: a missing column, an unknown column, a value of the wrong type, a string outside the `enum`, an empty or missing `rows`; **all errors, each with its JSON Pointer** | `ml.schema.mismatch` |
| 422 | a number that cannot be rendered exactly for its column (3 above), a string that cannot be rendered | `serve.number-not-exact`, `serve.string-not-renderable` |
| 422 | the predictor refuses the CSV: `ml.input.non-numeric`, `ml.input.non-finite`, `ml.input.missing-value`, `ml.input.categorical-*`, `ml.input.unseen-category`, `ml.input.empty`, `ml.input.too-many-rows`, `ml.input.unsupported-type`, every other `ml.input.*`, `ml.label.*`, `ml.schema.*`, `parse.*` | the predictor's tag, and the predictor's error object as `error` |
| 413 | the predictor refuses a limit: `limit.line-too-long`, `limit.input-too-large`, `limit.too-many-rows`, `limit.too-many-columns`, `limit.too-many-cells` | the predictor's tag |
| 500 | anything the request cannot cause: `ml.model.*` (the file was checked at start), `ml.limit.*`, `ml.metric.*`, `ml.tensor.*`, `ml.linalg.*`, `io.*`, `path.*`, `args.*`, an answer the server cannot print as JSON | the predictor's tag, or `serve.internal` |

The mapping is a function of the tag's prefix and is exercised by the end-to-end tests with a representative of every row; a gate (S9) checks that every tag in `ml introspect` has a row.

### 5.4 Where it listens, and the token

**[read, 1.3]** `ml-serve` listens on every address of the host: the pinned `tcp_listen` has no way to ask for `127.0.0.1` and the capability's bound is the port alone. So there is no loopback default, and the
rule "a bind that is not loopback requires a token" applies to **every** start: **`ml-serve` refuses to start (`serve.no-token`, exit 2) unless `--token-file FILE` or `--no-token` is given.**
`--no-token` is an explicit acknowledgement that every host that can reach the port can call the model; it is for a machine whose firewall is the access control. A move of the compiler pin to a
`tcp_listen` with a host argument is the change that lets a loopback default come back (section 11).

With `--token-file`: the file is read once at start, below `--root`, at most 4,096 bytes; one trailing LF (and CR) is removed; an empty token, a token with a byte outside `!` to `~` (a space inside, a control byte,
non-ASCII) is refused at start (`serve.token`). The token is held as data in `web.tokens` (`web.token`), compared by `web.guard` in constant time (cancho-web's `same_secret`), required by **every operation except `GET /health`**
(`web.require`/`web.no_auth`; `web.next_undecided` must find none at start, else the server refuses to run, as cancho-web's `guarded` example does). The token is never logged and is not in `/model` or `/openapi.json`
(the document names the scheme). Transport is plain HTTP: a bearer token on the wire is as private as the network is; TLS is for a proxy in front.

### 5.5 What is logged

One line per request on standard error: `METHOD path status milliseconds` (`web.log_line`, which leaves the query string out), and the line `listening on PORT` at start. Never a body, a cell, a column
value, a token or a header. **[choice]** The clock is read for the idle timeout of the loop and for the milliseconds of that line.

### 5.6 Refusals at start

`ml-serve` reads the model once and does everything that can be refused before it listens: the command line (`args.*`), the path (`path.*`), the model file (`io.*`, `limit.input-too-large`, `ml.model.*`,
`ml.model.string-table`: a damaged file is refused with its tag and exit code, exactly the refusal of `ml describe`), the token (`serve.token`), a column name that is not UTF-8 (`serve.column-name`), the size of the
documents (3.5), the port (`serve.listen`, exit 4, when the port is in use). Exit codes follow `docs/cli.md`: usage 2, input 3 to 8 as for `describe`, `serve.listen` 4.

## 6. Authority

`ml-serve` is a third `[[bin]]` of `cancho.toml` with the file list of `ml-predict` (the same sources) and `tools/ml-serve` (its own `main` and the modules of the server) in place of `tools/ml-predict`; a new file
of `tools/ml` is a source of the three, and `scripts/check_bins.py` fails on a missing or copied file (P4 extended).

Its row is derived by `cancho authority` and committed in `manifests/ml-serve.authority.json`. Expected, and checked against the derived row: `args`, `clock`, `conn_accept`, `conn_read`, `conn_write`,
`dir_read`, `err_write`, `file_read`, `fs_read("")`, `heap`, `io_write`, `net_in`, `poll`.

| label | why |
|---|---|
| `net_in`, `conn_accept`, `conn_read`, `conn_write`, `poll` | listen on a port and serve the connections of the loop |
| `clock` | the idle timeout of `http.server` and the milliseconds of the log line (cancho-web's `users` example has the same two uses) |
| `args` | the command line |
| `fs_read("")`, `dir_read`, `file_read` | the model file and the token file, read below `--root` once at start (and the library signatures the in-process call keeps) |
| `io_write` | `introspect` and `skill` |
| `err_write` | the usage line, `listening on PORT`, the access log |
| `heap` | the working data |

Not in the row: `file_write`, `dir_write`, `fs_write` (no write of any file), `exec` (no subprocess), `ffi` (no foreign code), `net_out`, `conc`, `env`, `io_read`. `tools.toml` has the ceiling `[ml-serve]` with the reason of every
label beyond `ml-predict`'s row, and `scripts/manifest.py` no longer forbids `net_in` and `clock` for a tool whose ceiling names them with a reason (the other tools keep the ban). A check fails when the binary's
embedded row, the committed file or the ceiling disagree; mutants that add `file_write`, `dir_write`, `exec`, `ffi`, `net_out` or `conc` to the derived row, or to a reachable function, must be caught.

## 7. The library boundary: what `wireio`, `verbs`, `mdesc` and `paths` change

The in-process call needs a memory-in entry, and the entry is the one the command line already has for standard input. The changes to `tools/ml`, each a few lines and each covered by the existing
suite (gate G: every fixture and golden file of the other binaries is unchanged):

* `paths.of_mem(bytes, shown)`: an `Rp` that holds bytes (the `stdin` Rp of `predict -`, built from a buffer that is not standard input).
* `wireio.peek_model`, `verbs.load_model` and `mdesc.describe` take the bytes of an `Rp` that holds memory instead of opening the path, and open the path otherwise. **A model read from memory and a model read
  from the file are the same bytes, decoded by the same function**; the file case is unchanged.

The model file is read once at start (`fsio.read_whole`, below `--root`, bounded by `--max-bytes`); each request builds an `Rp` over a copy of the bytes (a copy of the file's length per request: measured in section 10,
not claimed to be free) and decodes it, so a request does not depend on any other and no state is shared between requests. **[choice]** Decoding per request is the price of one code path; caching the decoded model
is an optimisation that adds a second path to the gate and is not done here.

## 8. What the server does with a request, in order

1. `web.dispatch`: route, method, the parameters (none are declared); 404/405 written by cancho-web.
2. `web.guard` (a token is set): 401/403.
3. `Content-Type` of a predict POST: 415.
4. `json.parse` on the body: 400 `serve.body-not-json` with the error's position; duplicate keys: 400 `serve.duplicate-key`.
5. For the batch: the count of `rows` (413 past the limit, before any validation of the records).
6. `schema.validate`: 422 `ml.schema.mismatch` with every error and its pointer (`schema.problem`, with the `rule` member added).
7. The rendering of 4: 422 `serve.number-not-exact` or `serve.string-not-renderable`, with the pointer of the first value.
8. The predictor on the CSV: its refusal mapped by 5.3, or its document.
9. The answer of 3.2.

## 9. The gates (written before the code; each must be able to fail)

| id | gate | test or script | mutants that must be killed |
|---|---|---|---|
| W1 | the three dependencies are pinned, consistent and not copied | `scripts/check_web_pin.py` | a rev edited in one block; a block removed; a copy of `web` under `lib/`; a sixth store of cancho-web |
| W2 | the three blocks do not change the bytes of `ml` and `ml-predict` | `scripts/check_web_pin.py --bins` (or the run recorded in section 10) | |
| S1 | the byte-equality of 4.1 over the 19-model corpus | `tests/conformance/test_serve_bytes.py` | number formatting, column order, null rendering, `dec` padding, category quoting |
| S2 | end-to-end over real sockets: readiness, every endpoint, every refusal with its status and `problem+json` body, keep-alive, pipelining (5 requests in one write, answers in order), a request split across 1-byte writes, a body larger than the limit (413), wrong content type (415), invalid JSON (400), JSON with duplicate keys, NaN, Infinity, 1e999, nesting of 100,000, a BOM, invalid UTF-8; unknown, missing and extra columns; unseen categories (`enum` and no `enum`); null; one record, a batch at the limit and over | `tests/conformance/test_serve.py` | a status mapped to another; a refusal as 200; the `rule` field dropped; the row limit off by one |
| S3 | the JSON number differential: the cell the server renders for a number equals Python's `Decimal`/`float` of the same text (int exact or refused, dec exact or refused, float correctly rounded or refused) over generated and adversarial numbers: 17-digit mantissas, subnormals, halfway cases, exponents, `-0`, `1e999` | `tests/conformance/test_serve_json.py` | the rounding of a float taken for an int; an exponent ignored; a dec scale off by one |
| S4 | the fuzz: request bodies, paths and headers (random bytes, truncations and byte flips of valid requests) never trap and never hang (every request is answered or the connection is closed within the idle time) | `tests/conformance/test_serve_fuzz.py` | none (the property is "no trap"); the harness is checked by a planted trap |
| O1 | the served OpenAPI document of every fixture model is a checked-in file and the check fails on drift; it validates with `openapi-spec-validator` | `scripts/gate_serve_openapi.py`, `docs/serving/*.openapi.json` | a field dropped from the document; an `enum` removed |
| O2 | Schemathesis against the served document of fixture models, if it can be installed in a virtual environment inside this clone | `scripts/gate_serve_schemathesis.py` | |
| A1 | the authority row is the committed `manifests/ml-serve.authority.json`, within the ceiling; a derived row with `file_write`, `dir_write`, `exec`, `ffi`, `net_out`, `conc` added fails | `tests/conformance/test_authority.py`, `scripts/manifest.py --check`, `scripts/mutants_http.py` | a write, an `exec`, a thread in a reachable function |
| A2 | `ml-serve` is built from the same sources: P4 extended | `scripts/check_bins.py` | a copied or missing file |
| T1 | the flag table of `ml-serve` is checked as the other tools' (introspect, skill, schemas) | `tests/conformance/test_tables.py` | a flag in the table and not in the parser |
| L1 | a load test: requests per second and latency percentiles at stated concurrency, the machine and N; no claim beyond the measurement | `scripts/bench_serve.py` | |
| G | every other fixture and golden file is unchanged | the existing suite | |

The mutants of the new code are `scripts/mutants_http.py` (counts in section 10).

## 10. Results

(filled from runs of this repository's scripts)

## 11. Not done, limits

* Reload (restart only). A loopback-only bind (the pinned compiler has none, 5.4). Per-request deadline (the loop has none). Caching of the decoded model. TLS. Per-column nullability for a model with an
  imputing plan. `problem+json` for the 400, 413 and 431 of the loop and the 405 of the router (not hookable at the pins).

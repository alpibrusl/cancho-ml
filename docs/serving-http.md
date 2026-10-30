# cancho-ml serving over HTTP: `ml-serve` (issue #16)

**Status: sections 1 to 9 were written before the code they judge and fix the dependency, the endpoints, the byte rule, the limits, the authority and the gates; a place where the code showed the
text wrong is corrected in place (the facts found while building are tagged [measured] or [read]). Section 10 holds the results and is filled only with what a script of this repository printed, with
its conditions; section 11 lists what the stage does not do.**
Tags as in `design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from the code or from a document of a pinned dependency, **[measured]** printed by a script
here, **[open]** not decided.

`ml-serve` serves ONE trained model file over HTTP with cancho-web. It is a third binary, built from the same sources as `ml` and `ml-predict` (`docs/serving.md` section 3), with its own
`main`. It is not a subprocess of `ml predict` (that would need `exec`): it calls the library code of `ml` in the same process, with the CSV and the model file as bytes in memory.

## 1. The dependency: cancho-web, cancho-schema and `http.server`, as pinned packages

### 1.1 What was found

* **cancho-web publishes `web` as a store [read, measured].** `alpibrusl/cancho-web` at `5f404a7` has `.cancho-vcs` (the store of `src/web.cho`, published by `scripts/publish.sh`, whose `--check` CI runs);
  `web` imports `schema`, so its `requires/` records `cancho-schema` at `59a70285ff23af63a226bc56ff978feccb4cf8ca` (the `SCHEMA_REV` of its `ci.yml`).
* **cancho-schema publishes `schema` as a store [read, measured].** `alpibrusl/cancho-schema` has `.cancho-vcs` at `59a7028` (the commit its store was last republished at; later commits do not change it).
* **`http.server` is a store of the compiler's repository [read, measured].** `alpibrusl/cancho` at `2bd251f7f9354fe43d48055b3d9fe4d4af007a5c`, the compiler these sources are written for, has
  `packages/http-server/.cancho-vcs`. Its content differs from the store at `a4572ea` (the module gained streaming request bodies, `Expect: 100-continue` and a byte-fed mode, `docs/http-server.md` sections 11 and 12 of the compiler repository); the functions `ml-serve` calls (`open`, `close`, `wait`, `next`, `head`, `parsed`, `body`, `respond`, `reply_with`, `reply_as`) keep their signatures.
* **A project pins the three the way it pins `cancho-table` and `cancho-tools` [measured].** A scratch project outside this repository with the three `[dependencies.*]` blocks below ran `cancho install`
  (`installed schema (59a70285ff23): 1 file(s)`, `installed server (0207556b4029): 1 file(s)`, `installed web (5f404a777018): 2 file(s)`), imported `http.server`, `schema` and `web` and
  built with compiler `0207556` (`cancho build --bin ml`, `--bin ml-predict` and `--bin ml-serve` all build; `web` and `schema` need no change). **No upstream change is needed and nothing is copied**; there is no LOCAL-DEV arrangement.

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
rev = "2bd251f7f9354fe43d48055b3d9fe4d4af007a5c"
path = "packages/http-server/.cancho-vcs"
```

`web` records its requirement on `schema` at `59a7028`, and a consumer must also name `schema` in its own `cancho.toml` (cancho-web's `scripts/publish.sh` says so); the two revs are the same. The module of the
`server` store is `http.server`. `scripts/check_web_pin.py` (gate W1) checks that the three blocks have the git URLs above, a 40-digit rev and the path, that the `schema` rev is the `SCHEMA_REV` this document
names, that the `server` rev is the compiler pin of `[package] cancho`, that no other store of these three repositories is listed, that no `.cho` file under `lib/`, `tools/` or `tests/` declares the modules `web`,
`schema` or `http.server`, and (`CHECK_REMOTE=1`) that GitHub has the three stores at those revs. Moving a pin is the act of editing the revs in `cancho.toml` and the lines of this section together.

cancho-web rev: `5f404a777018cf42b5c713daa4972c938c9229f4`

cancho-schema rev: `59a70285ff23af63a226bc56ff978feccb4cf8ca` (the `SCHEMA_REV` `5f404a7`'s `ci.yml` names: `SCHEMA_REV` `59a70285ff23af63a226bc56ff978feccb4cf8ca`)

cancho http-server rev: `2bd251f7f9354fe43d48055b3d9fe4d4af007a5c`

Only `ml-serve` imports these modules; `cancho build --bin ml` and `--bin ml-predict` are not changed by them (gate W2 compares the bytes of the two binaries built before and after the three blocks
are added; the other changes of this stage that reach their sources are listed in section 10).

### 1.3 What the pinned libraries do, read before the design

* **One thread.** `http.server` is one `Poller` loop (`wait`, then `next` until it answers -1, `respond`). Requests are answered one at a time, in order, per connection; pipelined requests are supported **[read]**.
* **Limits of the loop [read, measured].** The buffer of one connection (`size` in `server.open`) holds the head and the body. The head is bounded by `std.http` (65,536 bytes and 64 headers: more is the loop's `400`);
  a request whose head and body do not fit `size` is answered by the loop itself, before the handler, `413` (`{"error":"request too large"}`) or, for a head with no end in `size` bytes, `431`; a malformed head
  is `400`. These bodies are `application/json` `{"error": ...}`, not `problem+json`: **the loop writes them, and the pinned package has no hook for them** (section 5.3 says what that means for the documented
  refusals). `ml-serve` sets `size` to `--max-body-bytes` plus 16 KiB, and judges the body's length itself, so a body past the flag that the buffer still holds is a `problem+json` 413. The idle timeout of a connection is an argument
  of `server.open` (seconds; it reads the clock, so the row has `clock`). At most 1,024 connections, and 256 MiB of buffers in all: a larger `size` means fewer connections.
* **The output queue of a connection is 65,536 bytes [read].** An answer is handed to the kernel at once; what the kernel does not take is queued, and **a connection whose queue would hold more than 64 KiB is
  closed**. So every answer of `ml-serve` is bounded to 60,000 bytes by construction (section 5.2) and the documents that could be larger are refused or reduced at start (sections 3.5 and 5.2).
* **`tcp_listen` binds every address of the host, `tcp_listen_on` binds one [read].** `tcp_listen(net, port, backlog, flags)` is `INADDR_ANY`. `tcp_listen_on(net, host, port, backlog, flags)` (edition 9, `docs/native-sockets.md` section 12 of the
  compiler repository, `alpibrusl/cancho` #503) takes the host as a string literal in dotted-quad text, so the host is a choice between literals in the source and never a value read at run time. Section 5.4 is the consequence.
* **Numbers [read].** `std.json` parses strictly (RFC 8259: no `NaN`, no `Infinity`, no BOM, no trailing comma; a control character in a string, invalid UTF-8 and nesting past its depth limit are errors; the
  error carries the byte position) into a tape whose number nodes point at the number's text; cancho-schema's `integer` accepts a whole number written as a float (`150.0`, `1.5e2`) and its `number` accepts any JSON number.
  A duplicate key in an object is not an error of either (`json.get` finds one of them). Section 3 does not rely on either for the bytes of a cell.

## 2. The surface

```
ml-serve --model-file MODEL [--root DIR] [--port N] [--bind loopback|all] [--token-file FILE] [--max-body-bytes N] [--max-batch-rows N] [--idle-seconds N] [--max-bytes N]
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
| `--bind` | `loopback` listens on 127.0.0.1, `all` on 0.0.0.0 (section 5.4) | `loopback` | |
| `--token-file` | a file holding the bearer token (section 5.4); required with `--bind all` | none | |
| `--no-token` | accepted with the loopback bind, where it changes nothing; refused with `--bind all` (section 5.4) | off | |
| `--max-body-bytes` | the largest request body | 1,048,576 | 4,194,304 |
| `--max-batch-rows` | the most records in one batch | 100 | 1,000 |
| `--idle-seconds` | a connection that makes no progress for this long is closed (the only deadline of a request) | 10 | 3,600 |
| `--max-bytes` | the largest model file | 16,777,216 | 268,435,456 |
| `--max-out-of-range` | refuse a request with more rows outside the training range than this (0 refuses any); absent: no bound | none | 1,000,000,000 |

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
{"model_sha256":"<64 hex digits>","model":"linear","task":"regression","columns":["prediction"],"rows":[[36.17963203363425]],"row_count":1,"unknown_categories":0,"out_of_range":{"cells":0,"rows":0,"columns":[]}}
```

* `columns` is `describe.output.columns` for a regression (`["prediction"]`), `["label","probability"]` for every classification (the probability of the positive label; a constant classification has both
  always, the others are asked for with the `--proba` option of the same code), `["cluster","distance"]` for a clustering (the squared distance to the centre of its cluster, the `--distance` option), and `["label","probabilities"]` for a model of three or more classes (`softmax`, [multiclass.md](multiclass.md) section 3.4): the second value of a row is then an object whose keys are the labels, in ascending order, and whose values are the probabilities as JSON numbers, and the document names a cell as anything (`{}`) for that model only.
* A **ranking** model (`describe.task` is `ranking`) answers `["prediction","rank"]` for `POST /predict/batch`: the records of the batch are one group, `rank` is the 1-based position of a record by score descending, ties in the order of the records, and the cells are the text `ml predict --rank --group COL` prints for the same rows with one group value (docs/ranking.md section 7). `POST /predict` answers `["prediction"]` alone; a model that does not rank never has the column.
* Each row is an array with a value per column. **The value is the text the command line prints, as a JSON number**: the document of `ml predict` holds each value as a string made by
  `flt.put_float` (shortest decimal that reads back to the double, positional from 1e-6 to 1e21 with a point, scientific outside) or `exact.put_scaled` (a label, an integer), and the server writes the same string
  between no quotes after checking that it is a JSON number (`json.put_fragment` parses it: a text that is not one is `serve.internal`, 500, never a quoted string). The decimal text is the same; nothing is printed twice. 
* `model_sha256` is `describe.file.sha256` and the response header `X-Model-Sha256` carries it too.
* `unknown_categories` is the predictor's count over the request (the constant models count none; the server writes 0).
* `out_of_range` is the predictor's member of `ml predict` (`docs/tracking.md` section 3.2): the cells of the request outside the range of the training rows stored in the model file, the rows that hold one and the columns that do; `null` for a constant model and for a model file with no provenance section. `--max-out-of-range N` at start refuses a request with more than N such rows: HTTP 422, rule `ml.predict.out-of-range`, with the rows, the bound, the first column and the first row in the error. `ignored_columns` is not in the answer: the server builds the CSV from the columns of the model.
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

| column | the cell | refused with 422 (`serve.number-not-exact` or `serve.string-not-renderable`, with the JSON Pointer of the value) |
|---|---|---|
| `int` | the integer's decimal digits, `-` for a negative, no leading zeros, `0` for zero (`-0` is `0`) | a number that is not exactly an integer of at most 12 digits: `1.5`, `1e13`, `150.0000000000000000001` |
| `dec`, scale `s` | the value with exactly `s` digits after the point (`s = 0`: no point) | a number with more than `s` significant digits after the point, or more than 12 digits in all (`dec` is an exact 12-digit decimal) |
| `float` | `flt.put_float` of the nearest double (ties to even; the shortest decimal that reads back to it); `0.0` for any zero (the reader has no negative zero) | a number past the range of a double (`1e999`), or non-zero and smaller than the smallest subnormal |
| `categorical` | the string as UTF-8, between double quotes with quotes doubled when it holds a comma, a quote, CR or LF; an empty string is an empty cell | a string with a CR directly before an LF (the reader drops that CR inside a quoted field, so the cell could not be read back as sent; a lone CR and U+0000 are kept and are written as sent **[measured]**) |
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

Single thread (the `Poller` of the loop); a request is parsed and answered before the next. Bounded, and where each bound comes from:

| what | the bound | by |
|---|---|---|
| the body | `--max-body-bytes` (default 1 MiB, ceiling 4 MiB): more is `413 serve.body-too-large` (problem+json) when the loop's buffer still holds the request, and the loop's own `413` when head and body past the buffer (`--max-body-bytes` plus 16 KiB); the connection is closed after the loop's | the handler, then `http.server` |
| the head | 65,536 bytes and 64 headers (`std.http`'s `max_head` and the table of the loop): more is the loop's `400 {"error": ...}` | the pinned `std.http` **[measured]**: 17,000 bytes served, 70,000 refused; 60 headers served, 70 refused |
| the records | `--max-batch-rows` (default 100, ceiling 1,000): more is `413 serve.batch-too-large`, judged before the records are | the handler |
| the cells | the predictor's `--max-cells` is set to the rows times 8,192 and its `--max-columns` to 8,192 (the width of an encoded matrix, a one-hot block included), `--max-rows` to the rows of the request | the predictor |
| the JSON | nesting past the depth `std.json` allows is `400 serve.body-not-json` | `std.json` |
| the tape and the CSV of one request | about 24 bytes of tape for each byte of body, and the CSV (about the body's size): a body of the ceiling's 4 MiB takes about 100 MiB while it is judged and none after | the handler |
| an idle connection | `--idle-seconds` (default 10, ceiling 3,600): a connection that makes no progress is closed | `http.server` |
| the connections | the loop keeps one buffer of `--max-body-bytes` + 16 KiB for each connection within 256 MiB, and 1,024 at most: at the default 1 MiB about 250 are served **[measured]**; a connection past that is not served and the server goes on (S4) | `http.server` |

There is **no request timeout beyond the idle one**: the pinned loop has no deadline for a request, and a handler that runs long (a large forest over 1,000 rows) holds every other connection while it runs. The
load test (section 10) reports the time of a request of the ceiling.

### 5.2 The size of an answer

See 1.3 and 3.2. A batch of the ceiling's 1,000 rows of two columns is at most 1,000 times about 56 bytes; the server checks the final length against 60,000 and answers `500 serve.internal` otherwise (it cannot
happen: the check is a backstop). `GET /model` and `GET /openapi.json` are chosen to fit at start (3.3, 3.5).

### 5.3 The refusals and their statuses

Refusals are `application/problem+json` (RFC 9457) with a `rule` field and the `model_sha256`, **except the answers that the loop and the router write before the handler is reached**: `400` (malformed HTTP head), `413`
(head and body past the buffer) and `431` are `application/json` `{"error": "..."}` from `http.server`, `405` is `application/json` `{"error":"method not allowed"}` with `Allow` from cancho-web's `dispatch`, and
`401`/`403` are cancho-web's `guard` bodies **[read]**. They have no `rule` and (but for the loop's) the head of every answer has the `X-Model-Sha256` header. The `404` for a path that is not a route (`web.dispatch`) is
`problem+json` without a `rule`.

| status | when | `rule` |
|---|---|---|
| 400 | the body is not JSON (syntax, a BOM, invalid UTF-8, NaN or Infinity, a control character, nesting past the depth limit, trailing bytes), or has a key twice in a record or twice `rows` | `serve.body-not-json`, `serve.duplicate-key` |
| 401, 403 | a token is set and the request has none or an unknown one (401), or one that does not list the scheme (403) | none (cancho-web's body) |
| 404 | the path is not a route | none (cancho-web's body) |
| 405 | the method is not allowed on the route | none (cancho-web's `{"error"}` with `Allow`) |
| 413 | the batch has more than `--max-batch-rows` records; the body is longer than `--max-body-bytes` | `serve.batch-too-large`, `serve.body-too-large` |
| 415 | `Content-Type` of a predict POST is not `application/json` (parameters allowed, case ignored) | `serve.media-type` |
| 422 | the body breaks the request schema: a missing column, an unknown column, a value of the wrong type, a string outside the `enum`, null for a column that takes none, an empty or missing `rows`, an integer outside 12 digits; **all errors (the first 16 stored, the count of all), each with its JSON Pointer** | `ml.schema.mismatch` |
| 422 | a number that cannot be written exactly for its column (4), a category with CR LF | `serve.number-not-exact`, `serve.string-not-renderable` |
| 422 | the predictor refuses the CSV because of the data in it: `ml.schema.mismatch`, `ml.input.non-numeric`, `ml.input.non-finite`, `ml.input.missing-value`, `ml.input.unknown-category`, `ml.input.invalid-utf8`, `ml.input.reserved-category`, `ml.input.duplicate-column`, `ml.input.unknown-column`, `ml.limit.string-bytes`, `ml.limit.string-pool`, `ml.limit.categories`, `ml.limit.categorical-columns`, `ml.predict.out-of-range` (more rows of the request hold a feature outside the training range than `--max-out-of-range` allows, `docs/tracking.md`), and `ml.predict.non-finite` (the model's answer for a row of the batch is not a finite number: a prediction, a probability or a squared distance; the command line refuses it too, before it writes a cell, `docs/nonfinite.md`; `error.detail` holds the 1-based `row` of the batch and the `column`) | the predictor's tag, and its error object as `error` |
| 413 | the predictor refuses a limit of the size of the input: `limit.line-too-long`, `limit.input-too-large`, `ml.input.too-many-rows`, `ml.input.too-many-columns`, `ml.input.too-many-cells` | the predictor's tag, and its error object as `error` |
| 500 | everything else, which a request cannot cause: `ml.model.*` (the file was checked at start), `ml.plan.*`, `ml.fit.*`, `ml.linalg.*`, `ml.metric.*`, `ml.tensor.*`, `ml.split.*`, `ml.label.*`, `ml.param.*`, `ml.input.empty`, `parse.*`, `io.*`, `path.*`, `args.*`, `conflict.*`, an answer the server cannot write | the predictor's tag, or `serve.internal` |

The mapping is a function of the tag, two lists in `tools/ml-serve/srv.cho` (`status_of`) that gate S9 compares with this table and with the tags of `ml introspect`; the end-to-end tests reach a representative of
each row they can (the schema, the number and the string rules, a refusal of the predictor with its status and its rule equal to the command line's, the size rules). The schema's `ml.schema.mismatch` of the
predictor itself (a column the CSV lacks) cannot happen: the server writes every column.

### 5.4 Where it listens, and the token

**[read, 1.3]** `ml-serve` listens on **127.0.0.1** by default, so only the host it runs on can reach the port. `--bind loopback|all` chooses (default `loopback`): `loopback` calls
`tcp_listen_on(net, "127.0.0.1", port, 1024, 0)` and `all` calls `tcp_listen_on(net, "0.0.0.0", port, 1024, 0)`, the two literal calls of `open_listener` in `tools/ml-serve/main.cho` (the only file at `edition 9`).
The host cannot come from the command line: the compiler requires a literal, so `--bind` is a choice between two calls, and no flag takes an address.

| `--bind` | token flags | result |
|---|---|---|
| `loopback` (the default) | none | starts, listens on 127.0.0.1, no token needed |
| `loopback` | `--token-file FILE` | starts; every operation but `GET /health` needs the token |
| `loopback` | `--no-token` | starts; the flag is accepted and changes nothing |
| `all` | `--token-file FILE` | starts, listens on 0.0.0.0, the token is enforced |
| `all` | none | refused: `serve.no-token`, exit 2 |
| `all` | `--no-token` | refused: `args.conflict`, exit 2 |
| a value other than `loopback` and `all` | | refused: `args.bad-value`, exit 2 |

A server on every address is open to every host that can reach the port, so it requires a token (`serve.no-token`, with the repair in its hint: give `--token-file FILE`, or drop `--bind all`). `--token-file` together with `--no-token`
is `args.conflict`. The start-up line on standard error names the address: `listening on 127.0.0.1:PORT` or `listening on 0.0.0.0:PORT`.

**What the authority row proves, and what it does not [read, choice].** `main` keeps an unnarrowed `Net("")`, because one binary chooses between two literal calls, so the row stays `net_in("")` and `scripts/manifest.py`'s check passes.
`net_in("")` does not say which address the server listens on: the row alone cannot prove loopback. The proof of the default is the socket test of gate S10 (`tests/conformance/test_serve_bind.py`): the default server accepts on 127.0.0.1
and is refused (`ConnectionRefusedError`) on the machine's non-loopback address.

**A stronger alternative, described and not built [choice for the maintainer].** A second binary whose `main` narrows its network capability to `Net("127.0.0.1:")` (any port on 127.0.0.1, compiler `docs/native-sockets.md` section 12.3). The compiler would then refuse at compile
time every `tcp_listen` and every `tcp_listen_on` with another host, and `cancho authority` would print the loopback bound in the row (`net_in("127.0.0.1:")`), so the claim "this binary cannot listen on a network address" would be a fact of the row and not of a test.
The price is two binaries (one without the `all` call, one with it), two rows in `manifests/` and one more entry in `scripts/check_bins.py`'s lists. Until the maintainer chooses it, the default is proved by the socket test only.

With `--token-file`: the file is read once at start, below `--root` (a file of more than 1 MiB is `limit.input-too-large`); one trailing LF (and CR) is removed, and the rest is at most 4,096 bytes; an empty token, a token with a byte outside `!` to `~` (a space inside, a control byte,
non-ASCII) is refused at start (`serve.token`). The token is held as data in `web.tokens` (`web.token`), compared by `web.guard` in constant time (cancho-web's `same_secret`), required by **every operation except `GET /health`**
(`web.require`/`web.no_auth`; `web.next_undecided` must find none at start, else the server refuses to run, as cancho-web's `guarded` example does). The token is never logged and is not in `/model` or `/openapi.json`
(the document names the scheme). Transport is plain HTTP: a bearer token on the wire is as private as the network is, which is why `--bind all` is the form to put behind a proxy that terminates TLS, and loopback is the form for a proxy on the same host.

### 5.5 What is logged

One line per request on standard error: `METHOD path status milliseconds` (`web.log_line`, which leaves the query string out), and the line `listening on ADDRESS:PORT` at start (`127.0.0.1` or `0.0.0.0`). Never a body, a cell, a column
value, a token or a header. **[choice]** The clock is read for the idle timeout of the loop and for the milliseconds of that line.

### 5.6 Refusals at start

`ml-serve` reads the model once and does everything that can be refused before it listens: the command line (`args.*`, including a `--bind` value other than `loopback` and `all`, and the token rule of 5.4: `serve.no-token`), the path (`path.*`), the model file (`io.*`, `limit.input-too-large`, `ml.model.*`,
`ml.model.string-table`: a damaged file is refused with its tag and exit code, exactly the refusal of `ml describe`), the token (`serve.token`), a column name that is not UTF-8 (`serve.column-name`), the size of the
documents (3.5), the port (`serve.listen`, exit 5, when the port is in use or needs a privilege). Exit codes follow `docs/cli.md`: usage 2, a missing input 3, a limit or a refused file 8, `serve.listen` 5.

## 6. Authority

`ml-serve` is a third `[[bin]]` of `cancho.toml` with the file list of `ml-predict` (the same sources) and `tools/ml-serve` (its own `main` and the modules of the server) in place of `tools/ml-predict`; a new file
of `tools/ml` is a source of the three, and `scripts/check_bins.py` fails on a missing or copied file (P4 extended).

Its row is derived by `cancho authority` and committed in `manifests/ml-serve.authority.json`. Expected, and checked against the derived row: `args`, `clock`, `conn_accept`, `conn_read`, `conn_write`,
`dir_read`, `err_write`, `file_read`, `fs_read("")`, `heap`, `io_write`, `net_in`, `poll`.

| label | why |
|---|---|
| `net_in`, `conn_accept`, `conn_read`, `conn_write`, `poll` | listen on a port and serve the connections of the loop; the row is `net_in("")`, unnarrowed (5.4 says what that does not prove) |
| `clock` | the idle timeout of `http.server` and the milliseconds of the log line (cancho-web's `users` example has the same two uses) |
| `args` | the command line |
| `fs_read("")`, `dir_read`, `file_read` | the model file and the token file, read below `--root` once at start; the predictor's signature keeps them for every call, and a call over bytes in memory opens nothing |
| `io_write` | `introspect` and `skill` |
| `err_write` | the usage line, `listening on ADDRESS:PORT`, the access log |
| `heap` | the working data |

Not in the row: `file_write`, `dir_write`, `fs_write` (no write of any file), `exec` (no subprocess), `ffi` (no foreign code), `net_out`, `conc`, `env`, `io_read`. `tools.toml` has the ceiling `[ml-serve]` with the reason of every
label beyond `ml-predict`'s row, and `scripts/manifest.py` no longer forbids `net_in` and `clock` for a tool whose ceiling names them with a reason (the other tools keep the ban). A check fails when the binary's
embedded row, the committed file or the ceiling disagree; mutants that add `file_write`, `dir_write`, `exec`, `ffi`, `net_out` or `conc` to the derived row, or to a reachable function, must be caught.

## 7. The library boundary: what `wireio`, `verbs`, `mdesc`, `fsio` and `paths` change

The in-process call needs a memory-in entry, and the entry is the one the command line already has for standard input. The changes to `tools/ml`, each small and each covered by the existing suite (gate G: every fixture and
golden file of the other binaries is unchanged, and `test_serving.py` and `test_describe.py` pass):

* `paths.of_mem(bytes, shown)`: an `Rp` that holds bytes (the `stdin` Rp of `predict -`, built from a slice that is not standard input).
* `fsio.read_model`: the bytes of a model file for `wireio.predict`, `verbs.predict` and `mdesc.describe`: a copy of the bytes an `Rp` holds in memory, or the file below `--root` otherwise. **A model read from memory and
  a model read from the file are the same bytes, decoded by the same function**; the file case is unchanged.
* `wireio.predict` is `load_stored`, then `wireio.predict_loaded`, then `drop_stored`: the second function is the rest of the old body, taking the decoded model by reference. The command line calls the three in a row; the server
  decodes its model **once at start** and calls `predict_loaded` for each request, so the answer is the answer of the same function and a request does not decode the file again. The constant models of the older reader
  (`verbs.predict`, kinds 1 to 3) are tiny and are decoded for each request from the bytes held in memory.

The model file is read once at start (`fsio.read_whole`, below `--root`, bounded by `--max-bytes`). **[measured]** Without the single decode a request on a 33 MB forest of 200 trees took 155 ms (1 client); with it 9.5 ms
(section 10): the decode was the cost, and the design of one decode at start is the one measured.

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
| W2 | the three blocks do not change the bytes of `ml` and `ml-predict` | `scripts/check_web_pin.py --bins` | a changed byte |
| S1 | the byte-equality of 4.1 over the 19-model corpus | `tests/conformance/test_serve_bytes.py` | number formatting, column order, null rendering, `dec` padding, category quoting |
| S2 | end-to-end over real sockets: readiness, every endpoint, every refusal with its status and `problem+json` body, keep-alive, pipelining (5 requests in one write, answers in order), a request split across 1-byte writes, a body larger than the limit (413), wrong content type (415), invalid JSON (400), JSON with duplicate keys, NaN, Infinity, 1e999, nesting of 100,000, a BOM, invalid UTF-8; unknown, missing and extra columns; unseen categories (`enum` and no `enum`); null; one record, a batch at the limit and over | `tests/conformance/test_serve.py` | a status mapped to another; a refusal as 200; the `rule` field dropped; the row limit off by one |
| S3 | the JSON number differential: the cell the server renders for a number equals Python's `Decimal`/`float` of the same text (int exact or refused, dec exact or refused, float correctly rounded or refused) over generated and adversarial numbers: 17-digit mantissas, subnormals, halfway cases, exponents, `-0`, `1e999` | `tests/conformance/test_serve_json.py` | the rounding of a float taken for an int; an exponent ignored; a dec scale off by one |
| S4 | the fuzz: request bodies, paths and headers (random bytes, truncations, byte flips, insertions and deletions of valid requests, hostile paths and headers, hostile JSON) never trap and never hang (every request is answered or the connection is closed within the idle time); the resident set does not grow | `tests/conformance/test_serve_fuzz.py` | a planted trap for a body of a size; a planted spin on a path; an idle time in milliseconds |
| O1 | the served OpenAPI document of every fixture model is a checked-in file and the check fails on drift; it validates with `openapi-spec-validator` | `scripts/gate_serve_openapi.py`, `docs/serving/*.openapi.json` | a field dropped from the document; an `enum` removed |
| O2 | Schemathesis against the served document of six fixture models, in a virtual environment inside this clone (`build/venv`); it reported what the document could not say, and the exclusion is in the script | `scripts/gate_serve_schemathesis.py` | (the findings: a `dec` number and a CR LF the document accepts and the server refuses) |
| A1 | the authority row is the committed `manifests/ml-serve.authority.json`, within the ceiling; a derived row with `file_write`, `dir_write`, `exec`, `ffi`, `net_out`, `conc` added fails | `tests/conformance/test_authority.py`, `scripts/manifest.py --check`, `scripts/mutants_http.py` | a write, an `exec`, a thread in a reachable function |
| A2 | `ml-serve` is built from the same sources: P4 extended | `scripts/check_bins.py` | a copied or missing file |
| T1 | the flag table of `ml-serve` lists the flags the parser takes, `introspect` and `skill` answer, and each range is refused outside | `tests/conformance/test_serve.py` (`Start`) | a flag ignored (`--max-batch-rows`, `--max-body-bytes`) |
| L1 | a load test: requests per second and latency percentiles at stated concurrency, the machine and N; no claim beyond the measurement | `scripts/bench_serve.py` | |
| S10 | where it listens, on real sockets: the default and `--bind loopback` accept on 127.0.0.1 and are refused (`ConnectionRefusedError`) on the machine's non-loopback address, without a token; `--bind all` without `--token-file` and `--bind all --no-token` are refused at start; `--bind all --token-file` answers on both addresses and enforces the token on both; a token on loopback is enforced; a `--bind` value outside the two is `args.bad-value` | `tests/conformance/test_serve_bind.py` | the default flipped to `all`; `all` bound to loopback; the token rule dropped for `all`; the host literal changed; `--no-token` accepted with `all`; the old `tcp_listen` for loopback |
| S9 | every refusal rule of the predictor has a status (5.3), and the lists in the source, the document and `ml introspect` agree | `tests/conformance/test_serve.py` (`Statuses`) | a rule moved between lists; a status changed |
| G | every other fixture and golden file is unchanged | the existing suite | |

The mutants of the new code are `scripts/mutants_http.py` (counts in section 10).

## 10. Results

Machines: the Mac (arm64, Apple M4 Max, 16 cores, macOS 26.2, Python 3.14.5), `gram` (Linux x86-64, `taskset -c 0-6` for `scripts/check.sh`) and the Pi (Linux aarch64, the fingerprint only). Compiler `0207556b4029` on all three (`alpibrusl/cancho` main, the newest commit whose `ci` workflow succeeded and that contains #503, `tcp_listen_on`).

`scripts/check.sh` on gram, one pass, exit 0 (`~/Workspace/xplat-ml/loop-check.log`, 57 min 9 s elapsed, the full `ci/requirements-full.txt` environment): 520 conformance tests pass and none is skipped. On the Mac the modules that import without numpy (`scripts/run_conformance.py --no-oracles`, an environment with `ci/requirements-fast.txt` only) pass: 31 modules, 484 tests, 0 failed, and the six modules that need numpy are reported as not run. The load test is not repeated (below).

| gate | result | script |
|---|---|---|
| W1 | three dependencies pinned (`web` 5f404a7, `schema` 59a7028, `server` 0207556), none copied; the `server` store is the compiler repository at the compiler pin | `scripts/check_web_pin.py` |
| W2 | `ml` and `ml-predict` built from a copy without the three blocks and without the `ml-serve` bin are the bytes of the ones built with them; a flipped byte is refused | `scripts/check_web_pin.py --bins` |
| S1 | 19 models x their inputs: 118 model/input pairs, 4,962 rows and 7,668 values compared as text, one request each and the whole input as a batch; 29 rows refused for a category the model lists in full, and the command line agrees on each (it accepts, or refuses with `ml.input.unknown-category` under `unknown=refuse`); 20 rows not sendable (an empty cell for a model that takes none) | `tests/conformance/test_serve_bytes.py` |
| S2 | 46 tests: the end-to-end tests over real sockets (every endpoint, every refusal, keep-alive, pipelined requests, a request in one-byte writes, a chunked body, the token, the refusals at start, the log) and the status table (S9); the authority tests are 18 in their own module | `tests/conformance/test_serve.py` |
| S10 | 10 tests over real sockets, none skipped, on the Mac (non-loopback address 192.168.1.165) and on gram (192.168.1.82): the default and `--bind loopback` accept on 127.0.0.1 and are refused (`ConnectionRefusedError`) on the non-loopback address; the token is enforced on loopback when given; `--bind all` without a token is `serve.no-token`, with `--no-token` `args.conflict`, a bad value `args.bad-value`; `--bind all --token-file` answers on both addresses and enforces the token on both; `--bind` is in `introspect` with the default `loopback` | `tests/conformance/test_serve_bind.py` |
| S3 | 18,111 generated and adversarial numbers through `serve-check`: int, 976 rendered and 17,135 refused as `Decimal` says; dec at 6 scales; float 18,031 rendered (same double and as many digits as `repr`) and 80 refused; then 4,500 requests: 1,654 answered as the command line answers on Python's rendering, 2,842 refused with 422, 4 refused because the model's answer is infinite (`ml.predict.non-finite`) | `tests/conformance/test_serve_json.py` |
| S4 | 3,001 cases (random bytes, flips, truncations, edits, hostile paths, headers, versions, chunked and JSON bodies) on 48 threads: no trap, no hang, every case answered or closed; 252 of 400 simultaneous connections served at the default body limit; the resident set does not grow (the figures are in the log of the run) | `tests/conformance/test_serve_fuzz.py` |
| O1 | seven documents checked in (`docs/serving/*.openapi.json`), each served by a process, equal to its file and valid for `openapi-spec-validator` (7 of 7, `ok` for each). The seven files name the SHA-256 of a model file, which moves with the pin (10.1) | `scripts/gate_serve_openapi.py` |
| O2 | Schemathesis 4.31.1, seed 20261015, six fixtures, run on gram (not skipped): the default of the script (100 examples for each operation) gives 321 to 355 cases for each fixture, 0 failures; 1,000 examples for each operation gives 2,121 to 2,155 cases for each, 0 failures. `positive_data_acceptance` is not run for the models with a `dec` column or a category with no `enum`, for the reason in 3.4 and 11 | `scripts/gate_serve_schemathesis.py` |
| A1 | the derived row is `args, clock, conn_accept, conn_read, conn_write, dir_read, err_write, file_read, fs_read(""), heap, io_write, net_in(""), poll` (the same for either `--bind`: 5.4), bounded, no foreign symbol, within the exact ceiling; a row with `file_write`, `dir_write`, `conc`, `net_out`, `ffi`, `exec`, `env`, `io_read` or `fs_write` added fails, and so does `net_in` or `clock` for `ml` and `ml-predict` | `tests/conformance/test_authority.py`, `scripts/manifest.py --check` |
| A2 | the list of `ml-serve` is the list of `ml-predict` with its own directories; 11 mutants of the lists refused | `scripts/check_bins.py` |
| G | the fixtures and the records of the other binaries: the conformance suite and the four records (`golden_describe.json`, `golden_constant.json`, `golden_planned.json`, `golden_numeric.json`) pass with the digests of this pin (10.1) | the existing suite |

### 10.1 The compiler pin moves no model byte

The model file holds the compiler pin it was built with (informational, `docs/cli.md` section 4), so the digest of a model file moves with the text of the pin and with nothing else the compiler change does. The cross-platform fingerprint (`scripts/xplat_check.py`, 88 lines: the model file, `predict` and `eval` of every model kind at 1 and 8 threads on the bundled data) shows it:

* with the compiler at `0207556` and the pin text `a4572ea` in `generated/ml/built.cho` the 88 lines are the 88 lines of the golden written by the compiler at `a4572ea` (measured on the Mac on the merged tree);
* with the pin text of this tree the 66 lines of `predict` and `eval` and of the exit statuses are the same and the 22 lines of `model-file` differ, because the pin text is inside the file; the Mac, gram and the Pi (`ml` built there with the compiler at `0207556`) print the same 88 lines, which are the committed `docs/bench/xplat_fp.txt`;
* the records `golden_describe.json`, `golden_constant.json`, `golden_planned.json` and `golden_numeric.json` (97, 274, 60 and 19 cases), the seven `docs/serving/*.openapi.json` and the digest pinned in `test_honesty.py` carry model-file digests and are re-recorded with the pin text, in one commit; with the compiler at `0207556` and the pin text `a4572ea`, the generators print the old `golden_constant.json` (97 of 97 cases equal) and, for the numeric and planned records, every status and every written file of the old records (274 and 60 cases); the 8 numeric and 21 planned answers that differ are the ones that gained `split.spec`, whose answers `test_cat_baseline.py` and `test_planned_baseline.py` pinned beside the records and the records now hold.

**Stores.** The 23 stores are published with the new compiler and are the committed bytes (`docs/stores.md` section 7.1).

**Mutants** (`scripts/mutants_http.py`, on a frozen copy of the tree, on gram with the compiler at `0207556`, 26 min): 86 patterns, 82 killed and 4 survive, each explained in the script: `s-answer-limit` (the check of 60,000 bytes is a backstop that no request reaches), `s-unknown-count-dropped` (a category the model has not seen is refused by its `enum` before the predictor counts it), `s-non-finite-guard-dropped` (the predictor refuses a row that is not finite before a cell is written) and `c-an-empty-string-is-text` (two quotes and nothing are one empty cell to the reader). The unmutated copy passes. `scripts/mutants_serve.py` (the serving suite, 66 min): 59 killed of 59, the unmutated copy passes.
The ten patterns of the bind and the token rule (S10) are all killed by `test_serve_bind.py`: `m-all-needs-no-token`, `m-all-accepts-no-token`, `m-the-token-rule-is-everywhere`, `b-the-default-is-all`, `b-all-is-the-default-in-the-code`, `b-all-binds-loopback`, `b-loopback-binds-all`, `b-the-host-literal-is-another-loopback-address`, `b-the-start-up-line-lies` and `b-the-old-call` (the old `tcp_listen` for the loopback case). `b-all-binds-loopback` and `b-loopback-binds-all` need a non-loopback address on the machine to tell the two binds apart; gram has one.

**Load test**, not repeated: the figures below were measured with the compiler at `a4572ea` and a server that listened on every address, and no claim about speed is made for the pin or for the loopback bind (`scripts/bench_serve.py --n 2000 --repeat 3`, best of 3 by throughput; closed loop, one keep-alive connection per Python client thread; Apple M4 Max, load average 7.2 before and 11.3 after; the Python client takes
part of the time, the server is one thread, and the figures say nothing beyond this client on this machine):

| scenario | clients | requests | req/s | p50 ms | p90 ms | p99 ms | max ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `GET /health` | 1 | 2,000 | 41,122 | 0.022 | 0.031 | 0.043 | 0.054 |
| `GET /health` | 4 | 8,000 | 54,327 | 0.064 | 0.125 | 0.205 | 0.376 |
| `GET /health` | 16 | 32,000 | 42,041 | 0.331 | 0.686 | 1.122 | 2.245 |
| `POST /predict`, linear model (645 bytes) | 1 | 2,000 | 22,028 | 0.043 | 0.055 | 0.068 | 0.088 |
| `POST /predict`, linear | 4 | 8,000 | 37,658 | 0.080 | 0.160 | 0.170 | 0.217 |
| `POST /predict`, linear | 16 | 32,000 | 36,613 | 0.401 | 0.579 | 0.949 | 1.828 |
| `POST /predict/batch`, 100 records | 1 | 200 | 807 | 1.229 | 1.273 | 1.351 | 1.647 |
| `POST /predict/batch`, 100 records | 4 | 800 | 845 | 3.569 | 7.144 | 7.168 | 7.194 |
| `POST /predict`, forest of 200 trees, depth 12 (33,044,881 bytes) | 1 | 200 | 105 | 9.452 | 9.819 | 10.300 | 10.761 |
| `POST /predict`, same forest | 4 | 800 | 106 | 28.152 | 56.642 | 58.371 | 59.819 |

The model is read from disk and decoded once: before the single decode was made, the same forest took 155 ms for a request at one client (30 requests, an earlier run of this script, load about 9). One `ml predict`
process on one record took 6.6 ms at the minimum and 6.8 ms at the median over 30 runs. The resident set of the server after 32,000 requests is 3.2 MiB for the small model and 164 MiB for the forest (its decoded copy and the bytes).
With four clients the p50 of the forest is four times the time of a request: the loop answers one at a time.

## 11. Not done, limits

* **Reload** (restart only). **A binary that cannot listen on a network address at all** (a `main` with `Net("127.0.0.1:")`, 5.4: described, not built). **An address chosen on the command line** (the compiler takes a literal host only). **A per-request deadline** (the loop has none). **TLS.** **A listing of every refusal in problem+json:** the 400, 413 and
  431 of the loop and the 405, 401 and 403 of cancho-web are theirs (5.3).
* **Per-column nullability** for a model with an imputing plan: `describe` says `needs_impute` and not which columns, so no column takes `null` (3.1).
* **What the document cannot state.** cancho-schema at the pin has no bound or multiple for a `number` and no pattern but one for U+0000, so the document says `number` for a `dec` column whose cells take at most its
  scale's digits and 12 in all (a request outside is `422 serve.number-not-exact`), and `string` for a category with no `enum` that may not hold CR LF. Schemathesis finds exactly these two (section 10), and `positive_data_acceptance` is
  excluded for the models that have them. The change that would state them is in cancho-schema (bounds and `multipleOf` for `number`, a pattern for a string).
* **Memory of a large body.** The tape of a request is about 24 bytes for each byte of body; the allocator keeps what it took, so after a body of 4 MiB the server's resident set stays near 105 MiB **[measured]**.
* **A prediction that is not finite is one rule on the command line and on the server** (`ml.predict.non-finite`, `docs/nonfinite.md`): `ml predict`, `ml-predict` and `ml-serve` raise it from the same function before any cell is written, and the server answers it with 422 and the predictor's error object. cancho-table's `flt.put_float` takes finite doubles only (an infinity is written as `i.nf`); every call in this repository goes through one door that cannot be handed such a double.
* **Tie-breaking of the shortest decimal.** cancho-table's printer and Python's `repr` may choose different shortest decimals for the same double when two are equally near (`-902899553826474.3` and `...4.2`); the server prints
  what the command line prints, and the differential test compares the number of digits and the double.

# cancho-ml model modules: the contract, the generator, the admission script (issue #18)

**Status: sections 1 to 7 were written before any code of this stage (`scripts/gen_models.py`, `scripts/admit.py`, the manifests, `tools/ml/est.cho`).
Section 8 holds the results; a number or a verdict is in it only if a script of this repository printed it, with its conditions.** Tags as in
`design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from a document or tried against the pinned compiler, **[open]** not decided.

A model that `ml` can fit is a *module*: a directory under `lib/models/`, a manifest in it, an adapter with a fixed interface, and a document with
its gates. Nothing else in the tree names the model by hand: the dispatch, the flags, the plan keys, the schemas and the documentation tables
are generated from the manifests, and a module cannot add a float order, a capability, an unregistered refusal or a kind number of its own
without `scripts/admit.py` failing. The five models of the libraries (constant, linear, logistic, k-means, tree and forest) are the first users:
three are wired in this stage (`docs/wiring.md`), two have a manifest and `wired = false`.

## 1. Function values and traits: tried, and not relied on [read]

`docs/function-values.md` of the pinned compiler (`a4572ea`) says a function value is **captureless, non-generic and names a top-level function**, with
the row of the function fixed in its type; row abstraction was "answered no". Tried here (scratch programs, not committed): a plain
`fn(int) -> [] int` passed to a function builds and runs; a region-polymorphic function (`[&s](xs: &s [int])`) passed as `fn(&static [int])` also built in the one trivial
case tried. The fit and predict functions of the models take `heap: &!h Heap` and carry `[heap, conc]` rows that depend on that region parameter, and the
documented rule is that a *generic* value is refused, so a table of function values for the estimators is **not relied on**: no behaviour here depends on a case
the document says is refused. There are no traits. **The dispatch is a generated chain of `if kind == N` over the kind numbers** (defunctionalization,
`function-values.md` section 3): one function per operation in `tools/ml/est.cho`, written by `scripts/gen_models.py`, one branch per wired model, and a
test that every wired kind is a branch of every operation (section 5, Q1).

## 2. The manifest: `lib/models/<name>/model.toml`

One file per model, parsed by `scripts/gen_models.py` with Python's `tomllib`. An unknown key, a missing key or a value of the wrong type is an error.

| key | type | meaning |
|---|---|---|
| `name` | string | the plan name and the `--model` value; equal to the directory name |
| `kind` | int | the `u32` that opens the `model` section; **from `lib/core/kinds.cho`** (the registry in code) |
| `extra_kinds` | list of int, optional | more kinds the model writes (`constant` writes 1, 2 and 3: regression, classification, float regression) |
| `kind_fn` | string | the function of `kinds.cho` that returns it (`model_linear`); the generator checks the body |
| `shares_kind_with` | string, optional | `tree` and `forest` share kind 7 (the task is a field of the payload); both name each other, no other pair may |
| `task` | `regression` or `classification` | what the model predicts (`clustering` for k-means, which is never wired) |
| `tier` | `experimental` or `supported` | section 6 |
| `wired` | bool | callable from `ml`. `false`: the manifest holds the kind and the plan keys, the dispatch has no branch |
| `adapter` | string | the module that implements the interface of section 4 (required when `wired`); `builtin` for `constant`, which `tools/ml/verbs.cho` builds exactly and which has no adapter |
| `plan_order` | int | the place of the model in the plan grammar's list (`0 .. n-1`, once each): `constant`, `linear`, `logistic`, `kmeans`, `tree`, `forest` are 0 to 5 |
| `payload_version` | int | the version of the records after the `u32 kind` (linear 1, logistic 2: it gained the labels record, `docs/wiring.md` section 4) |
| `doc` | string | the document that holds the model's gates, written before its code (`admit` checks it exists, has a `Gates` section and was committed first) |
| `files` | list of strings | the source files of the module; `admit` applies the size, sum and tag checks to them |
| `[limits]` | table | named integer limits (`max_features = 256`): data, shown by `introspect` through the doc table |
| `[[hyperparameter]]` | table list | `name`, `type` (`int`: `min`, `max`; `float`: `nonneg = true`, the only float the plan grammar has; `enum`: `values`), `default` (string, the canonical spelling) or `required = true`, `doc` (one sentence without `;`, `\|` or a quote: it is a cell of the flag table) |
| `[tags]` | table | `emits`: every rule tag the module can answer; each must be in the tables of `spec.cho` (`extra()` or `pending()`), and each reachable one has a fixture |
| `[authority]` | table | `adds`: labels the module needs beyond the base row of `lib/models/contract.toml`, `reason`: why (empty when `adds` is empty) |
| `[sums]` | table | `decisive`: the files that hold decisive float sums; each must route them through `lib/linalg/order.cho` (section 5, Q4) |
| `[oracle]` | table | `script` and `args`: the Python oracle `admit` runs, `tolerance`: `bits` or a stated number for the comparison with scikit-learn |

`lib/models/contract.toml` holds what is common: the base authority row (`args`, `dir_read`, `dir_write`, `err_write`, `file_read`, `file_write`, `fs_read("")`, `heap`, `io_write`),
the file-size limit (2,000 lines), the thread counts the identity check runs (1 to 16), and the paths of the registries.

## 3. What is generated, from the manifests of the wired models

`python3 scripts/gen_models.py` writes the files below; `--check` writes nothing and exits 1 when any of them differs from a fresh generation
(cancho-table's drift check). `scripts/check.sh` runs `--check`.

| generated file | what it holds | the hand-written part |
|---|---|---|
| `tools/ml/spec.cho` | the whole file: the flag table (the `--model` choice list, one flag per hyperparameter with its default), `allowed()` per verb, `limits()` (one row per int hyperparameter), and the text of `tool()` | `tools/ml/spec.tmpl`: the flags, rules and texts that are not a model's; placeholders `{{...}}` |
| `tools/ml/est.cho` | module `ml.est`: `wired`, `task`, `name`, `tier`, `fit`, `predict`, `proba`, `payload_size`, `encode`, `decode`, `diag_len`, `hparams` (the model line's values) | nothing |
| `lib/pipeline/modeldefs.cho` | module `pipeline.modeldefs`: the keys of the `model` line (`kind:enum=...`, `model:linear` = `l2:fnn:=0.0` ...), the model names and their count | nothing |
| `docs/models.md` | the table of models, tiers, hyperparameters and tags | the text around it |

So **one table still drives the parser, `introspect`, `skill`, the schemas and the plan grammar** (`docs/cli.md` gate T1): a hyperparameter is written once, in the
manifest, and reaches `--l2` on the command line, `l2=` in a plan, the `limits` row, the schema and the document. `schemas/ml.v1.json` follows because
`scripts/schemas.py` reads the generated `spec.cho` and the manifests.

## 4. The interface of an adapter (fixed)

`adapter` names a module in the model's directory (`models.adapter_linear` in `lib/models/linear/adapter.cho`). It exports exactly these functions, with
these signatures; `scripts/admit.py` parses the file and compares them with the template in `scripts/gen_models.py`:

```
pub fn diag_len(d: int, ip: &[int]) -> int                       // size of the diagnostics buffer the fit needs
pub fn out_len(d: int) -> int                                    // number of parameters (coefficients and the intercept)
pub fn fit(heap, threads, xt: Tensor, xd: &[float], yd: &[float], ip: &[int], fp: &[float], seed: int, out: &![float], diag: &![float]) -> [heap, conc] int
pub fn predict(xt: Tensor, xd: &[float], wv: &[float], out: &![float]) -> int
pub fn proba(xt: Tensor, xd: &[float], wv: &[float], out: &![float]) -> int   // regression: answers a refusal
pub fn payload_size(d: int) -> int
pub fn encode(dst: &![byte], at: int, d: int, wv: &[float], c0: int, c1: int) -> int
pub fn decode(src: &[byte], d: int, out: &![float]) -> Decoded                  // code, c0, c1, end
```

`ip` and `fp` are the integer and float hyperparameters in manifest order; a code is a code of `ml.refuse` (1 to 99) as every library answers. The adapter
calls the library (`models.linreg.fit`, `models.logreg.fit`) and nothing else numeric: **every decisive sum is the library's, through `linalg.order`**.
`seed` is part of the contract so a stochastic model (a forest, k-means) fits through the same door; linear and logistic ignore it.

## 5. Admission: `python3 scripts/admit.py <model>`

Each check prints `PASS`, `FAIL` or `SKIP (reason)` and the script exits 1 on any `FAIL`. A module may be `supported` only if all pass. The cheap checks
run in seconds; `--fast` leaves out the three that re-run long scripts (Q9, Q10, Q11), which the mutant-module test set uses. **Every row names the mutant module that
must make it fail** (section 7); a check no mutant module can fail is not a gate.

| id | check | fails on |
|---|---|---|
| Q0 | **Doc first.** The `doc` file exists, has a `Gates` section, and its first commit is an ancestor of the first commit of every file in `files` (an uncommitted tree is `SKIP`: the real admission runs in a committed tree) | a module with no doc section |
| Q1 | **Manifest and generated glue.** The manifest parses with no unknown key; `gen_models.py --check` is clean; every wired kind is a branch of every operation of `est.cho` | a stale generated file; a wired model with a missing branch |
| Q2 | **Registry.** The kind is unique among manifests (except a declared `shares_kind_with` pair), equals the number `kinds.cho` returns for `kind_fn`, and is below `model_count()` | `mutant-kind-reuse` |
| Q3 | **Tags.** `emits` is a subset of the registered tags (`spec.cho` `extra()` and `pending()`); the module's sources contain no tag-shaped string literal and call no refusal code outside `emits`; each reachable tag has a fixture in `tests/conformance/fixtures.py` | `mutant-unregistered-tag` |
| Q4 | **Sums.** A file of the module that accumulates floats (`x = x + ...` on a float variable) must import `linalg.order` or a `linalg` kernel, or the sum is outside the package order | `mutant-sum-outside-order` |
| Q5 | **Authority.** The authority of `ml` derived from the tree (`scripts/manifest.py --tree`) equals the committed manifest, and every label beyond the base row is declared in some manifest's `[authority] adds` with a reason | `mutant-capability` |
| Q6 | **Thread identity.** `ml fit --threads t` for every `t` in 1 to 16 on a seeded data set gives the same model bytes as `t = 1` | `mutant-threads-one-count` (differs at 11 only) |
| Q7 | **Corrupted files.** Every truncation, a byte flip at every offset of a small model, and seeded multi-byte damage with and without a repaired checksum are answered with a rule tag, never a trap | a decoder that indexes past the payload |
| Q8 | **Size.** No file of the module is longer than 2,000 lines | a module file of 2,001 lines |
| Q9 | **Oracle.** The module's Python oracle agrees bit for bit with the `ml` binary (the model file bytes), on the cases the oracle script generates | an oracle of another order |
| Q10 | **Mutants.** `docs/bench/mutants_<name>.json` (written by the mutant script) says every mutant is killed or explained, and its pattern count equals the live count of `--check` (a stale result fails) | a survivor with no reason |
| Q11 | **scikit-learn.** `scripts/gate_wire_sklearn.py <model>` agrees within the tolerances the manifest states, printed with the numbers | a tolerance exceeded |

## 6. Tiers [choice]

`experimental`: listed by `introspect` with that label in the `--model` help, never the default, not mentioned in the README's list of what works, no stability promise for its
payload. `supported`: all of Q0 to Q11 pass at the commit that sets the tier, the model file layout is frozen until a `payload_version` bump, and the README may list it.
`constant` is `supported` by the gates of `docs/cli.md`; `linear` and `logistic` are set `supported` only if their `admit` run in section 8 is all `PASS`.

## 7. The mutant modules: the gate of the gate (`tests/modules/`)

`tests/modules/run.py` copies the tree to a scratch directory, adds the defective module (`tests/modules/mutants.py`: a new model directory, its entry in the registry, and for the capability mutant one edit of `main.cho`), runs
`gen_models.py`, `schemas.py` and `manifest.py` there as a module's author would, and runs `admit.py --fast` on it (building `ml` first when the check needs the binary). The run passes only if the check named in the table **fails** for that mutant, and the
control (an honest twin of `linear` under another name and kind) is admitted by every check that runs.

| mutant module | what it does | the check that must fail |
|---|---|---|
| `mutant-capability` | a function of the adapter's file reads the clock, and `tools/ml/main.cho` hands it one | Q5 |
| `mutant-kind-reuse` | a model with kind 4, the number of `linear` | Q2 |
| `mutant-unregistered-tag` | emits `ml.mutant.not-registered` | Q3 |
| `mutant-sum-outside-order` | a float accumulation in the adapter, no `linalg.order` | Q4 |
| `mutant-threads-one-count` | a fit whose bytes are the same at one thread count only (any count above 1 changes a byte) | Q6 |
| `mutant-threads-eleven` | a fit that is linear's at every thread count but 11 (a check at 1, 2, 4, 8 and 16 alone would pass it) | Q6 |

## 8. Results

*(Filled in when the code and its runs exist; see the end of this document.)*

## 9. Open questions

1. `tree` and `forest` share one kind number by a declared pair; if a third model ever wants to share, the rule needs thought (section 2).
2. Q4 is a static scan of the source text, not a proof: a sum written through a helper is not seen. Q6 and Q9 are the dynamic backstop (a changed order changes bytes or the oracle).
3. The `files` list is trusted: a source file not listed is not scanned. Q8 scans the whole directory of the module to cover that.

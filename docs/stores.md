# The numeric stores other repositories pin (issue #46)

**Status: sections 1 to 6 are the design and the gates, written before `scripts/publish_stores.py`. Section 7 holds what the scripts printed.**
Tags: **[read]** read from a source file or a document, **[choice]** a decision the maintainer can overturn, **[measured]** printed by a script of this
repository, with its conditions.

The question: can `cancho-stats` (and any later consumer) use the random streams, the exact sums, the matrix and its missing bitmap, the Cholesky
factor and the solves, and the ranks and splits of `ml` **without a copy of their code**? The rule of the project is that code is pinned and never
copied. `cancho-table` solved the same problem in the other direction with committed `.cancho-vcs` stores (cancho-table PR #44, `docs/reader.md`
section 1); this repository does the same for the modules below. `ml` itself does not consume its own stores.

## 1. The list, and why

The consumer's needs (`cancho-stats` `docs/design.md`): seeded derived random streams, the matrix type and its missing bitmap, exact sums, the
Cholesky factor and the solves, ranks, a k-fold split. The issue proposed `lib/core` (`exact`, `fbits`, `fsq`, `matrix`, `rng`, `limits`, `refuse`),
all of `lib/linalg` and `lib/metrics` (`sums`, `rank`, `regress`, `ratio`, `splits`). The rule that fixes the rest is **the import closure**: a
published module imports `std.*` and published modules only, otherwise a store set is not self-contained. `scripts/publish_stores.py --closure`
computes it from the sources and prints it; the publisher refuses a list that is not closed.

**[choice] The list is 23 modules.**

| Folder | Published | Why |
|---|---|---|
| `lib/core` | `ml.refuse`, `ml.limits`, `ml.rng`, `ml.exact`, `ml.fbits`, `ml.fsq`, `ml.matrix` | the issue's seven; `matrix` imports `limits`, `refuse` and `linalg.tensor`, `limits` imports `refuse` |
| `lib/linalg` | `linalg.order`, `linalg.tensor`, `linalg.dot`, `linalg.axpy`, `linalg.gemv`, `linalg.syrk`, `linalg.gemm`, `linalg.chol`, `linalg.par` | all nine: `chol` and the solves are the point; `par` imports `dot`, `gemv`, `syrk` and `gemm` |
| `lib/metrics` | `metrics.rules`, `metrics.ratio`, `metrics.classify`, `metrics.sums`, `metrics.regress`, `metrics.rank`, `metrics.splits` | the issue's five, plus `rules` and `classify`, which `regress`, `rank` and `splits` import (the closure) |

Not published, on purpose: `ml.kinds` (the registry of model kinds and stream kinds of `ml`: a consumer registers its own stream kinds in its own
registry, and two programs that share a seed must not share a kind, `docs/rng.md`), `ml.dict`, `ml.catmat`, `ml.bucket`, `ml.pathcheck`,
`ml.testdata` (the categorical columns and test helpers of `ml`), and `metrics.logloss` (nothing a consumer needs; adding it later changes no pin).
Everything under `lib/models`, `lib/transforms`, `lib/pipeline` and `tools` is the model file, the plan and the command line of `ml`, and is not an
interface for another program.

A module stays out of the list until a consumer needs it and its `pub` surface is something the project is willing to keep.

### 1.1 Names

A store is one directory per module, **named by the module's full name**, at `.cancho-vcs/<module>`: `.cancho-vcs/ml.rng`,
`.cancho-vcs/linalg.chol`, `.cancho-vcs/metrics.rank` (the way `cancho-tools` commits `.cancho-vcs/toolbox.cli`; `cancho-table`'s modules have
undotted names, so its stores are `.cancho-vcs/flt`). Each store holds the module's declarations and a `requires/` entry for each published module it
imports, so a consumer that pins `linalg.chol` reaches `linalg.tensor`, `linalg.order` and `ml.refuse` through it. The name of a `[dependencies.X]`
block is the consumer's label (dots would nest a TOML table): this document uses the module name with the dot written as `-`, e.g. `ml-rng`.

## 2. What is API

* **The API is the `pub` declarations of the 23 modules.** Their names, parameter types, result types and effect rows are in
  `docs/stores_api.txt`, one line per `pub` function, written by `scripts/publish_stores.py` from the sources. `--check` fails when the file
  differs from the sources, so a changed signature is a visible change of a committed file.
* **A change to a `pub` declaration of a listed module is a pin move for every consumer**: the stores are republished in the same commit, the
  consumer reads the diff of `docs/stores_api.txt`, and bumps its `rev` as its own commit. A new `pub` function is an addition. A function that is
  not `pub`, a module not listed, and everything in `tools/` or `lib/models` is not API and changes without notice.
* **Behaviour is API too:** the bytes of a result (`rng`, `exact`, the lane order of `linalg.order` and the kernels, `docs/linalg.md` section 2)
  are what the consumers' determinism rests on. The gates of those documents run on the sources the stores publish; a change that moves a bit is
  republished and announced like a signature.
* Refusal codes (`ml.refuse`) are `pub` functions, so a code number is API; a number is never reused (`docs/design.md`).

## 3. How a consumer pins

A consumer lists one `[dependencies.*]` block per module it imports **and** per module it wants named in its own `cancho.toml` (the
`requires/` entries bring the closure of each store, but a consumer that lists only the modules it imports still builds; section 7 says what was
measured). All blocks carry one `rev`, a full commit hash of this repository that contains the stores, the same way `cancho-ml` pins
`cancho-table` (`docs/reader.md` section 1.3):

```
[dependencies.ml-rng]
git = "https://github.com/alpibrusl/cancho-ml"
rev = "<40 hex digits>"
path = ".cancho-vcs/ml.rng"

[dependencies.linalg-chol]
git = "https://github.com/alpibrusl/cancho-ml"
rev = "<40 hex digits>"
path = ".cancho-vcs/linalg.chol"
```

and `import ml.rng;`, `import linalg.chol;` in its sources. `cancho install` (it needs the network) fetches the stores into `build/deps`; a build
from a file list passes `build/deps/*.cho`. Before the repository is pushed, a `file://` URL of a local clone at a commit works the same way, and
`scripts/consume_stores.py` uses it (section 5).

**The commit a consumer pins.** The stores are committed with the sources they were published from, in one commit, and `--check` is part of
`scripts/check.sh`, so **at every commit of the main branch the committed stores are what a fresh publish of that commit's sources gives**.
A consumer pins any commit that contains the stores. A later commit that did not change a listed module has the same stores byte for byte
(`git diff <pinned> <later> -- .cancho-vcs` is empty), so it is a pin move with no change; a later commit that changed one is a pin move with a
change, and `docs/stores_api.txt` and the diff say which. `cancho-table` handled the question the same way: its stores are in the commit of the
sources and the rev a consumer pins is a commit that contains them (its PR #44 note). There is no bootstrap commit: the stores do not contain
themselves, so publishing and committing is one step.

**The compiler.** The bytes of a store depend on the compiler that published it, so the publisher refuses a compiler other than the
`cancho = "<rev>"` of `cancho.toml` (as cancho-table's does). A consumer's compiler type-checks the store's source; it need not be the pin, but it
must be new enough for the modules (section 7 says which compilers were tried).

**No circular dependency.** `cancho.toml` of this repository lists no block whose `git` is this repository, and `[[bin]]` entries build from the
folders `lib/core`, `lib/linalg`, `lib/metrics`. A change to a source needs no pin move here; the stores are republished from the source.

## 4. The publisher: `scripts/publish_stores.py`

```
python3 scripts/publish_stores.py            # rewrite .cancho-vcs/<module> and docs/stores_api.txt
python3 scripts/publish_stores.py --check    # change nothing; exit 1 if a committed store or the API file is not what a fresh publish gives
python3 scripts/publish_stores.py --closure  # print the modules, their imports and the closure; exit 1 if the list is not closed
```

* **Deterministic:** the order is a topological order of the list (ties by name), every store is published by `cancho vcs publish --std` with a
  `--requires <lock>:<store>` per imported published module, and the output depends on the sources and the compiler only, not on the working
  directory, the output directory or the time.
* `--check` publishes into a temporary directory and compares the trees byte for byte: a changed, missing or extra file or store fails.
* It refuses: a compiler other than the pin; a listed module whose file does not declare its name; a listed module that imports a module that is not
  listed (and not `std.*`); an import cycle; a module that is not in `lib/core`, `lib/linalg` or `lib/metrics`.

## 5. The consumption proof: `scripts/consume_stores.py`

It makes a scratch package in a **temporary directory outside the repository**, with one `[dependencies.*]` block per store, `git = "file://<this
repository>"` and `rev` the current `HEAD` (or `--rev`), runs `cancho install`, builds a program that imports and calls each family, and compares
the output with values written in the script (computed from the definitions, not from this code):

| Call | Expected |
|---|---|
| `ml.rng`: the first outputs of seed 0 (`e220a8397b1dcdaf` first), `derive(derive(42,1),0)` (`d8005cdd08a0d146`), `unit` | the C reference's vectors (`docs/rng.md`) |
| `ml.matrix` with a missing bitmap: make a 3 by 2 matrix, mark a cell missing, read the bit, the cell, the count | the bitmap rule of `docs/reader.md` section 3 |
| `ml.exact`, `ml.fsq`: an exact sum of `1e16, 1, -1e16` (a double sum gives 0) and an exact sum of squares | 1 and the exactly rounded value |
| `linalg.chol`: factor a 3 by 3 positive definite matrix and solve a system | the solution, to the bits of a hand computation with integers |
| `metrics.rank`: ranks with ties; `metrics.splits`: a k-fold of a seed | the ranks; every row in exactly one test fold, equal sizes |
| the closure: `linalg.par`, `metrics.regress` and `metrics.sums` are imported and called once | the values |

The script exits 1 when `cancho install` or the build fails (a store missing or stale: the proof publishes nothing itself) and when any value
differs. It prints the output and its SHA-256 so that two machines can be compared. `--url` replaces the `file://` URL by the GitHub URL once the
commit is pushed; the proof for a pushed commit is `python3 scripts/consume_stores.py --url https://github.com/alpibrusl/cancho-ml --rev <hash>`.

## 6. The gates (written before the code; each can fail)

| Gate | Fails when |
|---|---|
| PS1 publish twice | two publishes of one tree, from different working directories into different output directories, differ by one byte |
| PS2 `--check`, source mutant | one byte of a listed source is changed (a comment byte) and the stores are not republished |
| PS3 `--check`, store mutant | one byte of a committed store file is changed, a store file is missing, a store or a file is extra, `docs/stores_api.txt` differs |
| PS4 closure | `--closure` finds a listed module with an unlisted import; the mutant lists `ml.matrix` without `linalg.tensor`, and `linalg.chol` without `linalg.order`, and both must be refused |
| PS5 no copy | `scripts/check_stores_pin.sh`: a `.cho` file outside `lib/core`, `lib/linalg`, `lib/metrics` declares one of the 23 modules; a `cancho.toml` block names this repository as its `git` (no circular dependency); `docs/stores.md` lists a module that is not in the publisher, or the reverse. Its `CONSUMER` mode reads another repository's `cancho.toml` and `.cho` files: every block on this repository has the 40-hex `rev`, one rev, a path `.cancho-vcs/<module>` of a listed module, and no file of the consumer declares a listed module |
| PS6 consumption | `scripts/consume_stores.py` fails to install, build or match a value; mutants: a store removed from the scratch repository, a store with one byte changed (the install must refuse it), a wrong expected value |
| PS7 two machines | the proof run on this Mac and on the Linux host (a clean clone, `taskset -c 0-6`) prints different output |
| PS8 the repository | `scripts/check.sh` fails (it runs `--check`, PS5 and the PS4 and PS2 mutants) |
| PS9 the documents | `check_docs.sh`, `check_prose.py`, `check_links.py`, `check_doc_refs.py` fail |

`scripts/mutants_stores.py` runs PS2 to PS6's mutants and reports each as killed or surviving.

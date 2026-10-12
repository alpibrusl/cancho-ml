# Proposal for cancho-table: publish the numeric modules as `.cancho-vcs` stores

**Status: a draft written by cancho-ml's reader stage. NOT filed, NOT pushed, nothing in cancho-table was edited.** The facts below
were measured on a read-only clone of `alpibrusl/cancho-table` at `7f3660881750f772bfd38de01792263fda930519` (main, after PR #32), with
the compilers `a4572ea` (cancho-ml's pin) and `e47d647` (cancho-table's pin). The tool that proves it is `publish_numeric.py` beside this file.

## Why

cancho-ml (`docs/reader.md`) wants cancho-table's exact readers (`:int`, `:dec(S)`, `:float`) and its exact float sum, as a **pinned
dependency and not a copy** (`CLAUDE.md`: the `contract` package is pinned, never copied; the same rule). A cancho dependency is a
`cancho-vcs` store at a pinned commit of a git repository (`[dependencies.NAME] git, rev, path = ".cancho-vcs/<module>"`,
`cancho docs/package-system.md` sections 7 and 8). cancho-table is a binary repository: no commit of it holds a store, so there is
nothing to pin.

## What was measured

* `vcs publish` of `tools/table/{pow5,dec,flt,query,facc,reader}.cho`, one store per module, requirements derived from the `import` lines:
  `pow5` 2 declarations, `dec` 10, `flt` 13, `query` 41, `facc` 9, `reader` 4; 500 KB for the six. No source needed a change.
* A project depending on `flt`, `query`, `facc`, `reader` by `git` (a scratch repository holding the stores) and `rev` ran `cancho install`,
  `cancho build --bin` and a program calling `flt.parse_float` and `query.parse_dec`: correct. `flt` pulled `dec` and `pow5` through `requires/`.
* Stores published by `e47d647` installed, built and ran under `a4572ea` (cancho-ml's compiler): the compiler pin of the consumer need not equal the publisher's.
* Not measured: a store fetched from GitHub (the scratch origin was a local git repository; the origin form is the same).

## Proposal (one pull request)

1. `scripts/publish_numeric.py` (from the draft): `--src . --out .` writes `.cancho-vcs/{pow5,dec,flt,query,facc,reader}`; `--check` compares a fresh publish with the committed stores byte for byte.
2. The six stores, committed. CI runs the `--check`.
3. `docs/numbers.md`: one paragraph naming the public functions a consumer pins against (`flt.parse_float`, `flt.key_of`, `flt.float_of_key`,
   `flt.put_float`, `query.parse_int`, `query.parse_dec`, `query.pow10`, `facc.width`, `facc.add_bits`, `facc.normalize`, `facc.finalize_key`,
   `facc.mean_key`, `reader.scan`, `reader.fields`, `reader.split_header`) and that a change to any of them is a change to a pinned interface.

Optional, not required: move `parse_int`, `parse_dec`, `pow10` and `int_max/int_min` of `query.cho` into a `num` module, so that a consumer of
the cell parsers does not carry the 38 other declarations of `query`.

## Draft issue text (cancho-table)

> **Publish the numeric modules as stores so another repository can pin them**
>
> cancho-ml reads CSV columns with `flt.parse_float`, `query.parse_dec`, `facc` and `reader`. It cannot depend on them as a package because
> cancho-table commits no `.cancho-vcs` store. The modules need no change: `vcs publish` of the six files works on `e47d647` and `a4572ea`
> (500 KB, deterministic), and a consumer's `cancho install` resolves them through `requires/`. Please add `scripts/publish_numeric.py` (draft
> attached), commit the stores, and check them in CI with `--check`. Names a consumer will pin: ... (the list above). Until then cancho-ml compiles
> the files of a clone of this repository at a pinned commit, marked as a local-dev arrangement (`docs/reader.md` 1.3).

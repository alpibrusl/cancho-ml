# The lane order and the path check (`v1-lanes`)

**Status: two tasks on the merged tree `v1-int`, in this order, each its own commits: section 1 (the check tools and `..`), then
sections 2 onward (one summation order with L = 4 lanes). Every number below was printed by the script named next to it, in this
session, on one machine: Mac arm64 (Darwin 25.2.0), cancho `a4572ea`, Python 3.14.5, numpy 2.4.6. Benchmarks were not run.**

## 1. The check tools and a path with two dots

**The defect.** `open_read` and `open_dir` end the process with SIGILL (exit 132, no message) on a path that holds two dots in a row
anywhere (`a..b`, `.../x`, `x..`), not only on a `..` component; `toolbox.path.has_dotdot` looks at components only. `ml` already
refuses these (`tools/ml/paths.cho`, `path.dotdot`, exit 2). Reproducer from `integration.md` section 7 item 2: `models-check linreg
../ml-data/n1000/x.bin y.bin n=1000 d=20` exited 132. The runtime-side reproducer is `docs/upstream/dotdot_path.cho`.

**Every place that opens a path from arguments** (grep of `open_read`, `open_dir`, `open_write`, `Fs(`, `file_read` over `lib/` and
`tools/`): `lib/` has none (the libraries take bytes and slices; no library entry opens a path). `tools/`: `ml` (checked since stage 2),
`models-check` (operands 2 and 3), `tree-check` (`x=`, `y=`, `miss=`), `reader-check` (the file; it already refused `..`, but with the
usage text and no tag). `linalg-check` and `metrics-check` release `fs` and open nothing.

**The fix.** One helper, not a copy of the contract package: `lib/core/pathcheck.cho` (`has_double_dot`, `tag()` = `path.dotdot`,
`exit_code()` = 2). `tools/ml/paths.cho` keeps its `has_double_dot` as a one-line wrapper over it, so `ml` and the three drivers run the same
function; the drivers cannot import `toolbox.*` (they are built from explicit file lists with no `build/deps`). A driver prints the
line `refused path.dotdot` and exits 2 before any builtin sees the path. The tag is the one `ml` already has (`path.dotdot`, exit 2); it
is not in the numeric registries of `integration.md` section 4 because those hold the library codes, and these drivers are test tools.

**Missing operands.** The gate found two more traps that no one had named: `tree-check` read `m[5]` of `mode=` and `[4]` of `out=` (an
index past the end), and `metrics-check` read `arg(g, k)` for `k >= 2` with fewer arguments (`metrics-check ratio`, `... ratio 1`,
`... split`, `... metric`, `... terms`: SIGILL). Both are fixed (`len` guards; `operand(g, k)` answering the empty text) and each is a fixture.

**The gate (`scripts/gate_paths.py`, in `scripts/check.sh`).** It builds the five drivers and runs: (a) `tests/paths/fixtures.json`, 69
fixtures (good paths still work, a missing file is still its old answer, every dotdot form is refused with exactly `refused path.dotdot`
and exit 2, 37 missing-operand cases end in a status the fixture names); (b) a sweep of every prefix of each valid vector and every
truncation of every argument (595 vectors); (c) a seeded fuzz of 400 argument vectors per tool, 2,000 in all (`FUZZ_SEED`, `FUZZ_N`),
by mutation of valid vectors with path-like fragments. A trap, a negative status, a hang over 60 s, a dotdot path that is not refused, or a
refusal where there is no dotdot path fails the gate; the expectation is computed by a Python model of each tool's own argument parser.
Result on the final tree: `paths gate: 69 fixtures, 595 sweep vectors, 2000 fuzz vectors (seed 20261013), 0 failures`.

**Mutants (`scripts/mutants_paths.py`, 19; `--check` in `check.sh`).** The helper always false, missing a trailing or a leading pair, looking
for the wrong pair; the tag changed; the exit code changed; in each of `models-check`, `tree-check` and `reader-check` the check off, each
operand unchecked, the refusal ignored, the reader's exit changed; the two length guards of `tree-check`; the operand guard of
`metrics-check`. **19 of 19 killed, none survives** (control passes first). `mutants_cli.py` (`n1-double-dot-check-removed`) now aims at
`lib/core/pathcheck.cho`, where the check lives; its 45 patterns still apply.

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

## 2. The lane order: what was built

The definition is `docs/linalg.md` section 2 (written first, commit `32f0074`); `docs/design.md` 3.1 and 17 record the decision and what it does
to bytes. Code: `lib/linalg/order.cho` (`lanes()`, `panel()`, `lane_combine`, `block_partial`), `dot.cho` (4 lanes, two blocks together), `gemv.cho`
(`gemv_n`: 4 rows x 4 lanes; `gemv_t`: four lane vectors per block, row `i` to lane `i mod 4`), `syrk.cho` and `gemm.cho` (2 x 2 tiles of four lanes:
sixteen accumulators; the old 4 x 4 tile would have needed sixty-four), `chol.cho` (panels of 32), `par.cho` (workspace sizes), `lib/models/linear/npass.cho`
and `lib/models/kmeans/kpass.cho` (the decisive sums of logistic regression and k-means), `lib/metrics/sums.cho` and `logloss.cho` (use `order.block_partial`).
Unchanged on purpose: `axpy`, the triangular solves, a row's score and distance (left folds, `models-linear.md` 3), the k-means++ total, the trees' regression
sums (sequential, one thread per tree), the command line (constant models are exact integer and `facc` arithmetic: no float order at all; `tools/ml` imports
nothing from `linalg`, and its 101 conformance tests pass unchanged).

Three independent implementations of the order agree bit for bit: the Cancho reference `tests/lref.cho`, the numpy oracles (`scripts/oracle_linalg.py`,
`oracle_models.py`; `oracle_metrics.py` already had this rule) and the C kernels of `scripts/c_parity/parity.c` (rewritten to the lane order; its hashes
equal `linalg-check`'s for dot, gemv_n, gemv_t, syrk and chol on the shapes tried). The gate could fail: before the kernels were touched, `t_dot` and `t_chol`
failed against the new reference (3 of 8 and 3 of 5 tests).

## 3. Gates whose expectation changed, and why

Only these. No tolerance was loosened anywhere; every comparison of two implementations is still bit for bit.

| what | why |
|---|---|
| `tests/lref.cho`, `scripts/oracle_linalg.py`, `scripts/oracle_models.py` | they ARE the definition of the order and were rewritten to the new one first |
| `tests/p_models.cho` `linreg_refusals` | **a real consequence.** Two equal columns with `l2 = 0` were refused (`ml.linalg.not-positive-definite`) under the old order and are **fitted** under the lane order (code 0 at 1 and at 4 threads): the pivot test `s > 0.0` on an exactly singular Gram matrix sees whatever the rounding leaves. The test now asks for what does not depend on the order: an all-zero column is refused (pivot exactly `+0.0`), and for equal columns one and four threads agree and the answer is the refusal or a fit. The claim "a rank-deficient X is refused" in `models-linear.md` 3.1 was false as written and is corrected in place there. Use `l2 > 0` |
| `scripts/gate_par.sh` | not forced by the order: its pipeline ended in `exit 0` after searching the oracle output for `MISMATCH`, so an oracle that crashed passed. It now requires the oracle's exit status and the line `suite: all bits equal` |
| `scripts/check.sh` | adds `t_order` (new test file: the constants, the lane combine on values where the association shows, `block_partial` against the reference for every length 0 to 260), `gate_paths.py` and `mutants_paths.py --check` |
| `manifests/ml.authority.json`, `generated/ml/built.cho` | task 1: the shared path helper adds one function to `ml`'s function list; the derived labels are the same (`manifest.py --check` and the ceiling pass) |
| mutant lists (`mutants_linalg`, `mutants_models`, `mutants_metrics`, `mutants_paths`) | patterns re-aimed at code that moved; new mutants for the constants (below) |
| `scripts/mutants_cli.py` | one pattern moved from `tools/ml/paths.cho` to `lib/core/pathcheck.cho` |

Documents that carried the old order and were corrected in place: `linalg.md` section 5 (a banner: every figure there is from the old order and none was re-measured),
5.3 item 2, 5.7 item 1; `metrics.md` 2.4 and 7 item 1; `models-linear.md` 3 and 3.1; `design.md` 16.

## 4. What changed in bytes (old tree `v1-int` against this tree, same arguments, hash lines of the drivers)

Printed by the two drivers built from each tree; "same" means the same hash.

* **Different:** `dot` n >= 5 (n = 200 and 1,000,000 tried), `gemv_n` with d >= 5 (d = 20, 300), `gemv_t` n >= 5, `syrk` n >= 5, `gemm` n = 33, `chol` d = 21 and 64
  (and `fwd`, `bwd` at d = 64: their own order is unchanged, they inherit the difference from L).
* **Same:** `axpy`; any reduction of at most 2 terms and, on the data tried, of 3 and 4 terms (`dot` n = 1 to 4, `gemv_n` d = 3, `syrk` n = 4, `gemm` n = 1, 2, `chol` d = 1, 2).
* **Models** (`models-check`, generated data): linear regression: parameters, raw sums and predictions all differ (n = 1,000, d = 5: 5 of 6 parameters, up to 3 ulp; n = 100,000, d = 20: 11 of 21, up to 6 ulp);
  logistic (n = 1,000, d = 5, 4 iterations): 3 of 6 parameters, up to 11 ulp, the raw sums differ, the predicted classes are the same; k-means (n = 1,000, d = 5, k = 4): 9 of 20 centroid values, up to 5 ulp, labels the same.
  With 3 rows or 2 clusters of 2 rows everything is the same.
* **Metrics, trees, forests, the random generator and splits, the command line, model files of kinds 1 to 3:** same bytes (not re-measured one by one: the gates that pin them, below, pass unchanged).

A model or a prediction made before this change therefore differs in the last bits from one made after. There are no shipped models of kinds 4 to 7; `design.md` 17 records that the model file has no field for the order and proposes `meta` + `u32 sum order` (not implemented, format untouched).

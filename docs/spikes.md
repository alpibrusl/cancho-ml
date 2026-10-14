# Spikes (M0): the record of the benchmark that decided the design

This file is a **record**. It holds what was written and measured for issue
[alpibrusl/cancho-ml#1](https://github.com/alpibrusl/cancho-ml/issues/1) (M0: benchmark first), in the order it happened:
the gate, written before any spike code; spike L (logistic regression by gradient descent), spike S (the same problem by
Newton and by L-BFGS), spike F (a histogram random forest), the Q-det study (what it costs to make a sum independent of the
thread count) and the ceiling of SIMD kernels written in C. It may carry dates, commits and conditions. It is **not a speed
claim for `ml`**: every timing was taken on a loaded, shared Mac with the programs named in the section, all before the 4-lane
summation order of `docs/lanes.md`, so they are the record of a decision and say nothing about the speed of `ml` today. A
number that is not in a source named below is not in this file.

## Index

| section | content | measured (Mac, cancho `a4572ea`) |
|---|---|---|
| 0 to 7 | the gate: questions, fixed inputs, determinism arms, correctness gates, the decision rule | written 2026-10-12 18:41, before any spike code |
| 8, 9 | spike L: arm A, oracle, mutants, timing, the tuning ladder, quiet-moment rerun; gaps found | 2026-10-12 |
| 10 to 12 | spike S: Newton and L-BFGS, passes over X; gaps found | 2026-10-12 |
| 13, 14 | spike F: the histogram forest against its oracle, timing against scikit-learn, profile, gaps | 2026-10-12 |
| 15 | Q-det: arms A, B (exact accumulator), C (fixed block tree) on a Newton step | 2026-10-12 |
| 16 | the SIMD ceiling in C (summary; the document is `docs/simd-ceiling.md`) | 2026-10-12 evening to 2026-10-13 |
| 17 | gaps found for upstream, with reproducers | |
| 18 | M0 outcome: the factor per spike against its reference, and the decision rule row of each | |

## Provenance: where each part was committed, and what was changed to put it here

The spikes were run in separate clones of this repository (siblings of this one on the maintainer's machine, named below);
none of those clones is a reference a reader can follow, and the files they hold are copied into this repository. Commits are of
the clone's branch; dates are the commit dates (all 2026-10-12 unless shown, +0200).

| part | clone and branch | commit (date) | scripts and data in this repository |
|---|---|---|---|
| sections 0 to 7, the gate | `cancho-ml-opt`, `m0-opt`; the gate is also on origin as `m0-gate` | `54fcfc1` (18:41) | the text only |
| sections 8, 9 (spike L, first slice; tuning) | `cancho-ml-opt`, `m0-opt` | `8922077` (19:07), `8ed24a6` (20:09), `4de2cc4` (20:09, quiet-moment rerun) | `scripts/spikes/{gen.py, oracle_l.py, sklearn_l.py, check_oracle.py, bench_l.py, bench_variants.py, spike_l.cho, spike_l_tuned.cho, expcheck.cho}`, `scripts/spikes/variants/` |
| sections 10 to 12 (spike S) | `cancho-ml-solver`, `m0-solver` | `1679230` (20:06) | `scripts/spikes/{spike_s.cho, oracle_s.py, check_oracle_s.py, bench_s.py}` |
| sections 13, 14 (spike F; were 10 and 11 in its branch) | `cancho-ml-forest`, `m0-forest` | gate `9c0d087` (20:25), oracle `e7cbb9e` (20:25), results `e63ab76` (20:35); ablation harness `c60c77f` | `scripts/spikes/{spike_f.cho, oracle_f.py, check_f.py, bench_f.py, ablate_f.py, splitmix64_ref.c}`, `scripts/spikes/variants/spike_f_branchless_partition.cho` |
| section 15 (Q-det; was `docs/linalg.md` section 6 in its branch) | `cancho-ml-armb`, `armb` | gate `f4a1c4e` (23:42), results `3cb826d` (23:57), wording `d9c419c` (23:57) | `scripts/spikes/q_det/`, `docs/bench/q_det.jsonl`, `docs/bench/q_det_gates.txt`, `docs/bench/q_det_rerun.txt` |
| section 16 and `docs/simd-ceiling.md` | `cancho-ml-simd`, `simd-ceiling` | gate `bac2412` (22:22), kernels `4152248` (23:41), results `5814443` (2026-10-13 00:04) | `scripts/simd/`, `docs/bench/simd-*` |

What was changed on the way in, and nothing else:

* **Renumbering.** The branches numbered their spike sections independently (spike S and spike F both used 10 and 11). Here spike S keeps
  10 to 12, spike F is 13 and 14, Q-det is 15. The references in other documents (`docs/models-linear.md`, `docs/models-tree.md`,
  `docs/design.md`) and the section numbers in the comments of the copied scripts were changed to match. Sections 0 to 9 keep their numbers.
* **Notes added to a source text** are in square brackets and begin with "Record note". Corrections made in place by the original authors
  ("Corrected 2026-10-12 in place") are kept as they were.
* **Two paths** in `scripts/spikes/q_det/build.sh` and `scripts/spikes/q_det/compare_models.py` named a sibling clone (`../cancho-ml-solver/...`);
  they name `scripts/spikes/spike_s.cho` and `scripts/spikes/` of this repository.
* **Comments of copied scripts** that cite a section (`docs/spikes.md 10.x` in the files of spike F, `docs/linalg.md section 6` in those of Q-det)
  were changed to the numbers of this file. `scripts/spikes/q_det/facc.cho` is cancho-table's exact accumulator, vendored unmodified for arm B (section 15.7
  quotes its hash), and was not touched: its header still says `docs/linalg.md section 6`, which is section 15 here.
* **Two copies of `bench_l.py`** existed (47 lines in `m0-solver`, 50 in `m0-opt` and `m0-forest`); the 50-line one is here. Spike S's
  `bench_s.py` does not import it.
* **Not copied:** the generated data (`ml-data/`, one `x.bin`, `y.bin` and `meta.json` per size, up to 168 MB), `build/` and `out/` of the
  clones. `python3 scripts/spikes/gen.py [OUTDIR]` writes the data again from the seed (20261012); the SHA-256 of each file is in
  section 8 and in each `meta.json`. The scripts name `/Users/alfonso/Workspace/alpibrusl/lexsys/ml-data` as the data directory
  (a `--data` flag in most; `ML_DATA` in `scripts/spikes/q_det/`; a constant in `check_f.py`, `bench_f.py` and `ablate_f.py`). No file over 2 MB was left out: the largest copied file is `docs/bench/simd-gate.jsonl`
  (67,500 bytes); the largest source is `scripts/spikes/q_det/spike_q.cho` (34,854 bytes).
* The clone `cancho-ml-det` holds an older merge of `m0-opt` and `m0-solver` with uncommitted edits; nothing in this file comes from it.
  `cancho-ml-core` holds an older state of what `main` has.

Not in any source, so not in this file: a measurement on `gram` (Linux x86-64); a measurement of `predict` for spike L or S; a run of the
gate's determinism arms with threads on a machine other than the Mac; whether the gaps of section 17 were filed as issues upstream.

---

*Sections 0 to 7 are the gate as committed at `54fcfc1` (2026-10-12 18:41), before any spike code. Its first line then read: "Status: gate only. No spike has been written or run; there are no numbers in this file." Two later corrections were made in place by the authors and are kept where they stand (the data are binary files, not CSV, in section 2; the default backend, in section 8). The gate text is otherwise as written, including the choices marked **(choice)**. Compiler documents named in section 0 (`docs/floating-point.md`, `docs/float-math.md`, `docs/compile-time.md`, `docs/ROADMAP.md`) are documents of the cancho compiler repository, not files of this one.*

## 0. What is already known (read from the pinned compiler's docs, not measured here)

Source: `cancho` at `a4572ea`, `docs/floating-point.md`, `docs/float-math.md`, `docs/compile-time.md`, `docs/ROADMAP.md`.

* `float` is native IEEE-754 binary64, round-to-nearest-even; the optimiser may not reassociate; `a * b + c` is not
  contracted to `fma` (checked in the disassembly, `compile-time.md`). So one thread's float result is a function of the
  source order of operations, and the same on every machine.
* `exp` and `log` are cancho code in `std/math.cho` (fdlibm's algorithms), 1 ulp from glibc on the ranges measured there,
  not calls to a platform libm. They are therefore the same bytes on Mac and Linux. Their *speed* is not known.
* No SIMD: the checked backend does not vectorise (overflow traps and bounds checks block it). The LLVM backend vectorises
  only the wrapping variants, and its document lists `Type::Float` as a gap at the time it was written (unverified for
  `a4572ea`). Which backend M0 uses is part of M0.
* Not yet known: the cost of an exact accumulator in an inner loop; the cost of a fixed reduction tree; thread scaling of a
  float loop; the cost of `exp`/`log` per call.

## 1. Questions M0 answers

1. **Q-speed.** How far is cancho from scikit-learn on one core, for fit and for predict, on logistic regression and on a
   random forest?
2. **Q-det.** What does each way of making the result independent of the thread count cost, on the same loop?
3. **Q-gaps.** Which cancho gaps did the spikes hit (multiply-high, SIMD, float formatting, anything else)? Each becomes an
   issue upstream, with the numbers that justify it.

## 2. Fixed inputs

* **Data.** 1,000,000 rows by 20 `float` columns plus a binary label, generated by a script with a seed and written as CSV
  with the shortest round-trip decimal. Both sides read the same file bytes. Its SHA-256 goes in the results. A 100,000-row
  prefix is run too, to see whether the ratio depends on size.
  *(Correction 2026-10-12: the data are not CSV. `scripts/spikes/gen.py` writes `x.bin` (row-major little-endian float64) and
  `y.bin` (float64 0/1) and a `meta.json` with the SHA-256 of each; both sides read the same file bytes, which is the
  property the gate wanted. Reason: cancho's `std` has no text-to-float (`std.fmt` only prints), so a CSV would have timed
  or validated a parser, not the fit. The 100,000- and 1,000-row sets are exact prefixes of the 1,000,000-row one.)*
* **Logistic regression, spike L.** Batch gradient descent on the mean log-loss, zero initial weights, fixed learning rate
  and iteration count (flags, no stopping rule), no regularisation. Written in cancho and in numpy with the *same
  algorithm*.
* **Random forest, spike F.** T = 10 trees, depth 8, histogram split search with 64 bins per feature, `sqrt(20)` features
  per split, seeded bootstrap. The cancho RNG is its own specified stream, so the trees will **not** equal scikit-learn's;
  this is stated, and the comparison below is on the metric, not on the trees.
* **Machines.** The Mac (Apple M4 Max) and `gram` (Linux x86-64; `taskset -c 0-5`, `nice`). No GPU.

## 3. The arms for Q-det (all on spike L's gradient sum, the one loop that decides the result)

| arm | what it is | claims |
|---|---|---|
| A | plain sequential `float` sum | fastest; correct at one thread only; the baseline |
| B | exact accumulator for every decisive sum (the cancho-table accumulator) | same bytes at any thread count |
| C | fixed-shape reduction: rows cut into blocks of a fixed size, each summed in order, the partial sums combined in a fixed tree | same bytes at any thread count if the block size does not depend on the thread count |

Cost is reported as the arm's time divided by A's, at 1 thread, and the scaling of B and C at 1, 2, 4, 6 threads (gram) and
1, 2, 4, 8, 16 (Mac).

## 4. Correctness gates (a result is void if any fails)

* **G-oracle (spike L).** The cancho coefficients after the fixed iterations equal the numpy fixed-iteration descent's to
  a stated tolerance, **1e-9 relative on every coefficient (choice)**, at one thread with arm A. If arm A and numpy differ
  by more, the difference is explained (summation order) before any timing is reported.
* **G-det.** Arms B and C produce byte-identical models at every thread count tried, on both machines; and the Mac and gram
  models are byte-identical to each other. Arm A is expected to fail this at more than one thread and the failing case is
  recorded (a gate that cannot fail is not a gate).
* **G-metric vs scikit-learn.** `LogisticRegression` uses a different solver (L-BFGS), so it reaches a different
  point. A timing comparison against it is reported only if the test AUC differs by at most **0.002 (choice)**; otherwise
  only the same-algorithm numpy comparison is reported. For the forest, the test AUC must differ by at most **0.005
  (choice)** from `RandomForestClassifier` at the same T, depth, features per split and bootstrap.
* **G-mutants.** The G-det check is run against a mutant of arm B and of arm C that adds in thread-arrival order. Both must
  fail G-det at more than one thread. A mutant that survives means the gate is void.
* **G-noise.** Every timing is the minimum of 7 interleaved runs, with the load average at start and end; a cell whose
  spread (max over min) exceeds 1.25 is rerun, not reported.

## 5. Decision rule, fixed now (choice, to be confirmed before the first run)

Let R be cancho's time divided by scikit-learn's, fit, one core, same machine, minimum of N, for each spike that passed
G-metric:

| R | what the pitch may say |
|---|---|
| R <= 2 | speed may be stated, with machine, size and threads |
| 2 < R <= 10 | lead with reproducibility; say nothing about speed; document the factor in `docs/spikes.md` |
| R > 10 | reproducibility only, and M1 sets a smaller row limit than the matrix could hold; the factor is recorded as the gap |

If arm B costs more than **3x (choice)** arm A at one thread and arm C does not, M1 specifies C. If both cost less than 3x,
M1 picks the simpler to specify and says why. If both cost more, M1 reopens the determinism promise instead of weakening
it silently.

## 6. Output

Numbers in a table here, with: machine, cores, threads, load at start and end, minimum of N, compiler commit, backend, the
data SHA-256, and the sentence "the gap is a factor of X". Not done until those numbers are in this file. The cancho gaps
found go to `alpibrusl/cancho` as issues, one each, linked from here.

## 7. Open before the first run

* `gram` has no scikit-learn (`ModuleNotFoundError`). Installing it is a download; it needs the maintainer's yes, and goes in
  a virtual environment under the home directory, not system Python.
* Which backend: Cranelift (the default) and LLVM if `Type::Float` builds on `a4572ea`. Both are reported if both build.

## 8. Spike L results (2026-10-12; arm A, Mac only; sources `cancho-ml-opt` `m0-opt` `8922077` and `8ed24a6`, scripts in `scripts/spikes/`)

Everything below was measured here, once, by the scripts in `scripts/spikes/`. **Not measured:** `gram`, threads, arms B/C,
spike F, `predict`, G-det, G-mutants. [Record note: this list is as of the commits of spike L alone. Spike F, arms B and C with threads, and G-det
for a Newton step were measured later on the Mac and are sections 13 to 15; `gram` and the `predict` of spikes L and S were not measured anywhere.]

**Conditions.** Apple M4 Max, macOS 26.2 (arm64), cancho `a4572ea` (edition 5), `--std`, the default backend and
`--backend llvm`; *(corrected 2026-10-12, see 8.3: `cancho --help` of `a4572ea` says the default is LLVM "since #127"; the table rows labelled "Cranelift" below were the default build, i.e. LLVM, and that is why they equal the `--backend llvm` rows. `--backend cranelift` is a different, slower build, timed in 8.3.)* numpy 2.4.6, scikit-learn 1.8.0, Python 3.14. The machine was **shared and loaded** (5 users; load average
6.5 to 8 throughout, listed per row), so the times are upper-bound-ish and the Mac is not a quiet machine. Each timing is the
minimum of N=7 interleaved runs (cancho, cancho-llvm, numpy in turn), spread = max/min. numpy and scikit-learn were run with
`VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; whether Accelerate honoured it was not verified (an
earlier unpinned run gave the same times within 3%). cancho's clock is `clock_ms` (1 ms resolution), around the iterations only; the
file read and decode are outside it. Algorithm: lr 0.5, 100 iterations, zero start, train = first 80% of rows, test = last 20%.

**Data** (`gen.py`, seed 20261012, positive rate 0.54; the `y.bin` hashes are in each `meta.json`): SHA-256 of `x.bin`:
* 1,000,000 rows: x `a5d34dd36c0a457467df59f76926cad40f1eac10976bf05229dc879e327072ca`, y `844e27d7ad4bf019d71b418a79d395e7e47544818040e62bd914f36a050b2ef9`
* 100,000 rows: x `896ce7c83bb6a8a276e905babe247859b618f1a77f72d314ddfd9fc5a4a5508c`
* 1,000 rows: x `8b01ef54aad11cae4fc02fec8e80ce7a36b487df7e2d0f2d156a5c3d1a3141f7`

### 8.1 G-oracle: passed, at 1e-9, with room to spare

`check_oracle.py` compares the 21 coefficients (20 weights, bias) of cancho arm A with numpy's fixed-iteration descent:

| set | numpy order | worst relative difference | worst distance |
|---|---|---|---|
| 1,000 rows | BLAS (`X @ w`, `X.T @ r`) | 4.1e-16 | 2 ulp |
| 100,000 rows | BLAS | 5.3e-16 | 3 ulp |
| 1,000,000 rows | BLAS | 4.6e-16 | 4 ulp |
| 1,000 rows | sequential (columns added in order, `np.cumsum`) | 2.1e-16 | 1 ulp |
| 100,000 rows | sequential | 2.3e-16 | 2 ulp |

The difference is not zero, and the reasons are two, of which only one was isolated. (1) **Summation order**: numpy's BLAS and
pairwise sums add in another order than cancho's left-to-right loops; the sequential-order numpy run, which adds in cancho's
order, brings 2 to 4 ulp down to 1 to 2 ulp. (2) What is left (1 to 2 ulp) is *not* separately isolated; the likely cause is
`exp`, which is cancho code (fdlibm, 1 ulp from the C library per `float-math.md`) against numpy's own `exp`. That is an
inference, not a measurement. The tolerance was not loosened.

**The gate can fail** (`check_oracle.py --selftest`, 1,000 rows, same conditions; the run exits 1 if a must-reject mutant is accepted):

| mutant | verdict | worst relative difference |
|---|---|---|
| cancho: one fewer iteration | rejected | 1.5e-3 |
| cancho: lr x (1+1e-6) | rejected | 1.6e-7 |
| cancho: lr x (1+1e-8) | rejected | 1.6e-9 |
| cancho: lr x (1+1e-10) | **accepted** | 1.6e-11 |
| cancho: lr x (1+1e-12) | **accepted** | 1.6e-13 |
| numpy side: one weight x (1+1e-6) / x (1+1e-8) | rejected / rejected | 1.0e-6 / 1.0e-8 |
| numpy side: bias moved by 100 ulp | accepted | 1.3e-14 |

So the gate **resolves a relative learning-rate error of about 1e-8 and larger** (a change of relative size e in lr moves the
coefficients by about 0.16 e here), and does *not* see 1e-10 or smaller. A mutation of lr by 1e-12, which the brief asked for, is
below the 1e-9 tolerance and is accepted; it would take a tighter tolerance to reject it, and bit equality is not
expected between libraries with different `exp`.

### 8.2 Timing, fit only, one thread (iterations 100, lr 0.5)

| set | variant | min of 7 | spread | test AUC | load avg at start / end (1, 5, 15 min) |
|---|---|---|---|---|---|
| 100,000 | cancho, default build (= LLVM, see 8.3) | 0.295 s | 1.027 | 0.885563 | 6.70 6.99 5.59 / 6.56 6.96 5.59 |
| 100,000 | cancho, LLVM | 0.294 s | 1.027 | 0.885563 | same run |
| 100,000 | numpy GD, same algorithm | 0.066 s | 1.021 | 0.885563 | same run |
| 100,000 | scikit-learn L-BFGS (6 iterations), C=inf | 0.0093 s | 1.20 | 0.885568 | 6.68 6.98 5.60 (both ends) |
| 1,000,000 | cancho, default build (= LLVM, see 8.3) | 2.958 s | 1.011 | 0.886683 | 6.68 6.98 5.60 / 6.76 6.99 5.68 |
| 1,000,000 | cancho, LLVM | 2.945 s | 1.023 | 0.886683 | same run |
| 1,000,000 | numpy GD, same algorithm | 0.733 s | 1.039 | 0.886683 | same run |
| 1,000,000 | scikit-learn L-BFGS (6 iterations), C=inf | 0.1053 s | 1.06 | 0.886683 | 6.54 6.94 5.67 (both ends) |

(The load average the OS reports is refreshed every few seconds, so start and end are equal for the short scikit-learn runs.)
The 100,000-row scikit-learn cell has spread 1.20 (first run slower), under the 1.25 limit of G-noise.

* **G-metric** (test AUC within 0.002 of scikit-learn): passes on both sizes (5e-6 and under 1e-6 apart), so the comparison below
  is allowed. It compares *different algorithms* (100 full-batch gradient steps against 6 L-BFGS iterations); it is the
  decision-rule quantity R of section 5 only if the maintainer accepts that.
* cancho arm A fit / numpy same-algorithm fit: **4.5x** at 100,000 rows and **4.0x** at 1,000,000, one thread, this Mac.
* cancho arm A fit / scikit-learn fit: **32x** at 100,000 rows and **28x** at 1,000,000 rows, one thread, this Mac, loaded. Per
  section 5 that is the R > 10 row, on a loaded machine and on arm A, the fastest arm. The gap is stated, not diagnosed:
  where the time goes was not profiled *when this was written*; section 8.3 does that, and the 4.0x / 4.5x here are superseded by it.
* **LLVM backend.** `Type::Float` builds on `a4572ea` with `--backend llvm` (no error), the output coefficients are
  byte-identical to the Cranelift build on the 100,000-row set, and the fit time is the same within noise (0.294 vs 0.295 s;
  2.945 vs 2.958 s). The statement in section 0 that the LLVM document lists `Type::Float` as a gap is stale for this commit.
* The loader (outside the timed region) took 7 ms for the 100,000-row set (`#load_ms` includes the file read); every decoded
  value is checked against its source bytes with `bits_of`, and the run stops on any difference, inf or NaN.

### 8.3 Arm A, profiled and tuned (2026-10-12; `m0-opt` `8ed24a6`, quiet-moment rerun `4de2cc4`)

Everything here is new, measured on the same machine as section 8 and by `scripts/spikes/bench_variants.py` / `bench_l.py`.
**Conditions.** Apple M4 Max, macOS 26.2 arm64, one thread, cancho `a4572ea` with `--std`, default build (LLVM) unless a row
says `--backend cranelift`; numpy 2.4.6 and scikit-learn 1.8.0 pinned to one thread as in section 8 (not verified that
Accelerate obeys). Fit only, 100 iterations, lr 0.5, minimum of N=7 interleaved runs, spread = max/min. **The machine is shared
(5 users) and was never quiet: load average 2.5 to 3.4 (1 min) during every run below, listed per table. No run had load < 2,
so none is marked quiet.** Data and algorithm as in section 8 (same SHA-256s). The ablation rows are subtractive
(a part removed, outputs therefore wrong, time kept), so their pieces overlap and do not add up exactly.

**Correctness of every kept variant.** All the kept variants produce the **same 21 output lines, bit for bit**, as the
original `spike_l.cho` (`same_bits_as_first True` in `bench_variants.py`, on 100,000 and 1,000,000 rows), so nothing about
G-oracle moved: `check_oracle.py --bin tuned` gives PASS at 1e-9 with the same worst differences as section 8.1 (blas order:
2, 3, 4 ulp at 1,000 / 100,000 / 1,000,000 rows; seq order on `v5`, the same code as tuned: 1 and 2 ulp at 1,000 / 100,000). No summation
order was changed, no `fma`, no reassociation, no change of IEEE results. The selftest of the gate still rejects all its
must-reject mutants on the tuned binary.

#### Where the time went (original arm A, 100,000 rows, 8.0 M row-iterations, min of 7, load 2.5 start / 2.5 end)

| variant (outputs wrong where marked) | time | reading |
|---|---|---|
| original `spike_l.cho` | 0.275 s | 34 ns per row-iteration |
| exp replaced by `-z` (wrong) | 0.133 s | **exp and the division: about 0.14 s, about half** |
| gradient loop removed (wrong) | 0.109 s | gradient pass: about 0.17 s |
| dot loop removed (wrong) | 0.163 s | dot product: about 0.11 s |
| dot and exp removed (wrong) | 0.072 s | what is left: the gradient pass and the loop |

So: about half of the time was `std.math.exp` (cancho code, not inlined away), the rest roughly evenly the two 20-wide loops.
The disassembly (`otool -tvV`) of the original loops shows why they were slow: per element, an overflow-checked
`base + j` (`adds`, `b.vs`), a bounds check (`cmp`, `b.hs`) and the load, i.e. 4 of 11 instructions are checks; the dot loop is a
single dependent chain of 20 `fadd`s per row, with `exp` and two divides after it, one row at a time. The check cost is the one
`check-cost.md` predicts ("provable is free, loaded is not": `xs[base + j]` is not provable from the loop, the subslice is).
Inside `exp`, `pow2` is a loop of squarings (called twice per `exp`) and `round_to_int` is a data-dependent branch on the sign
of `z`, whose sign varies from row to row; removing that branch is the 0.081 -> 0.061 step below (that the cause is branch misprediction is an inference, no hardware counters were read).

#### Steps, one change at a time (100,000 rows; one batch, load 2.5 / 2.5; every row bit-identical to the original)

| variant | change | time | spread |
|---|---|---|---|
| v0 | original `spike_l.cho` | 0.275 s | 1.065 |
| v1 | `exp_t`: std's exp, same operations, 2^k from the exact power table the loader already has (instead of `pow2`'s loop) | 0.231 s | 1.026 |
| v1w | v1, and the index `base + j` written `wrapping_add(base, j)` (bounds check kept) | 0.166 s | 1.054 |
| v2 | v1, and each row read through `let row = xs[i*20..i*20+20]` (one check per row instead of per element) | 0.160 s | 1.056 |
| v2m | v2 with std's `math.exp` (so: subslice alone) | 0.238 s | 1.063 |
| v3_2 / v3_4 / v3_8 / v3_16 | v2, rows taken 2 / 4 / 8 / 16 at a time: independent dot chains and exps in flight, each row's own additions in the original order, gradient rows added in row order per weight | 0.141 / 0.105 / 0.081 / 0.077 s | 1.05 / 1.02 / 1.04 / 1.07 |
| v3m_8 | 8 rows at a time with std's `math.exp` | 0.125 s | 1.048 |
| v4 | v3_8, and the rounding step in `exp_t` without a branch (`h = 0.5; if kf < 0 { h = -0.5 }; truncate(kf + h)`, identical for every kf including -0.0) | 0.061 s | 1.066 |
| **tuned** (= v5) | v4, and one table lookup `y * tab[k + 1075]` instead of two (exact either way for \|x\| <= 660, outside it `exp_t` calls `math.exp`) | **0.061 s** | 1.033 |
| v5_b1 | tuned without the 8-row blocking (one row at a time) | 0.102 s | 1.059 |

Reading: no single change is the story. The 8-row blocking is worth 0.160 -> 0.081 s on v2 and 0.102 -> 0.061 s on the final
exp (v5_b1 -> tuned); the branchless round 1.3x (v3_8 -> v4); the table `pow2` 1.2x (v0 -> v1); the row subslice 1.45x on top of v1
(v1 -> v2), and `wrapping_add` on the index alone gets the same (v1w 0.166 s vs v2 0.160 s). **Did not help:** 16 rows instead of 8 (0.077 vs 0.081 s in the batch above, 0.084 vs 0.085 s in an
earlier pair; within the 1.04 to 1.07 spreads); the second table lookup removal (v4 -> v5: 0.061 -> 0.061 s, noise; kept because it is
fewer operations, not because it was shown faster); `--backend cranelift` (below, slower).
Not tried: any change of summation order (the 4-chain unrolled dot product would change z's rounding and need its own oracle),
`fma`, a sigmoid that avoids `exp` (it would change bits and needs a ulp study), threads.

Split of the tuned program (subtractive, wrong outputs, 100,000 rows, same batch): total 0.061 s; exp removed 0.036 s (exp and
its division about 0.025 s, **about 40%**); dot removed 0.042 s (dot about 0.019 s); gradient removed 0.046 s (gradient about
0.015 s); exp and gradient removed 0.022 s (the dot, the loop and loads). `exp` is still the largest single piece.

**`exp_t` is bit-identical to `math.exp`**, tested (`scripts/spikes/expcheck.cho`): 114,000,002 points (a 1e-4 sweep of
[-700, 700], a 1e-6 sweep of [-40, 40], 20 M xorshift points in [-710, 710]), 0 differing bits. That test found a bug in my first
version (guard of 700 with a single table lookup indexes out of the table above k = 971; the guard is now 660), and it can fail:
a mutant with one polynomial coefficient scaled by (1 + 1e-6 / 1e-12 / 1e-15) differs in 111 M / 82 M / 0.75 M points.
The fit itself only reaches \|z\| well below 660.

#### Final comparison, one thread, this Mac (load 3.0 to 3.4 at start / 2.8 to 3.4 at end of each; not quiet)

| set | variant | min of 7 | spread | cancho tuned / this |
|---|---|---|---|---|
| 100,000 | cancho original (default build) | 0.278 s | 1.058 | |
| 100,000 | **cancho tuned** | **0.061 s** | 1.049 | |
| 100,000 | numpy GD, same algorithm | 0.061 s | 1.076 | **1.0x** (tied) |
| 100,000 | scikit-learn L-BFGS (6 iterations) | 0.0084 s | 1.18 | **7.3x** |
| 100,000 | cancho original, `--backend cranelift` | 0.466 s | 1.034 | |
| 100,000 | cancho tuned, `--backend cranelift` | 0.228 s | 1.044 | |
| 1,000,000 | cancho original (default build) | 2.794 s | 1.060 | |
| 1,000,000 | **cancho tuned** | **0.615 s** | 1.057 | |
| 1,000,000 | numpy GD, same algorithm | 0.690 s | 1.067 | **0.89x** (cancho faster) |
| 1,000,000 | scikit-learn L-BFGS (6 iterations) | 0.0974 s | 1.08 | **6.3x** |
| 1,000,000 | cancho original, `--backend cranelift` | 4.689 s | 1.057 | |
| 1,000,000 | cancho tuned, `--backend cranelift` | 2.304 s | 1.030 | |

(Load for the 1,000,000 cancho/numpy batch: 3.20 3.76 3.67 / 2.83 3.53 3.59. For the 100,000 batch: 3.01 3.75 3.67 / 3.40 3.80 3.69.
The scikit-learn runs were N=7 right after, load 2.83 / 2.83 as the OS reported it. The scikit-learn 100,000-row spread is
max/min of its 7 runs, 0.0099 / 0.0084.) The numpy time moved between sessions (0.733 s in section 8, 0.690 s here); the cancho
original moved from 2.958 to 2.794 s. Both are the same code under different shared-machine load, which is the reason the ratios
are quoted from one interleaved batch only. **The gap to numpy, same algorithm, one thread, was 4.0x at 1,000,000 rows (4.5x at 100,000)
and is now 0.89x (1.0x): it is closed on this machine, at this load.** The gap to scikit-learn is not the same kind of number:
it compares 100 full gradient steps with 6 L-BFGS iterations, and what remains (6.3x to 7.3x, down from 28x and 32x) is about
the algorithm, not the loop: this program makes one pass over the matrix per iteration, 100 passes; L-BFGS did 6 iterations (the number of function evaluations was not counted).
Per section 5 this stays in the 2 < R <= 10 row on the scikit-learn ratio (R = 6.3 at 1,000,000 rows, 7.3 at 100,000), loaded
machine, arm A, arm A being the fastest arm. Whether numpy's own time is limited by two memory passes and a vectorised `exp` was not
profiled here.

**Quiet-moment rerun (load average below 2 on the 1-minute figure, which is still not an idle machine: 5 users logged in).**
Same binaries, N=7 interleaved, run right after the commit when load fell: 1,000,000 rows, load 1.81 2.91 3.33 / 1.96 2.83 3.29:
original 2.746 s (spread 1.036), **tuned 0.611 s (1.021)**, numpy same algorithm 0.686 s (1.026), so tuned/numpy **0.89x**, original/numpy 4.0x;
scikit-learn 0.0970 s (spread 1.06, N=7), so tuned/scikit-learn **6.3x**. 100,000 rows, load 1.96 2.83 3.29 / 1.97 2.82 3.28:
original 0.271 s (1.048), **tuned 0.060 s (1.050)**, numpy 0.061 s (1.081), tuned/numpy **0.98x**, original/numpy 4.4x; scikit-learn 0.0081 s (spread 1.23, first run slow, under the 1.25 limit), tuned/scikit-learn **7.4x**. The ratios agree with the loaded batch above to the second digit.

**`--backend cranelift` finding.** `cancho --help` at `a4572ea` says the default is LLVM; section 8 called the default build
"Cranelift" and so its two rows were the same backend. Explicit `--backend cranelift` is 1.7x slower on the original (4.689 s vs
2.794 s) and 3.7x slower than the LLVM tuned build; its output bits are the same.

**What was not done.** Not measured: `gram`, threads, predict, arms B/C (the blocking changes nothing about how a determinism arm
would be written, but B and C need their own measurement; B's accumulator per row will not be free at 8 rows in flight).
Whether the tuned loop has the same ratio on gram's x86-64 was not measured. All timings in this section come from one loaded Mac.

## 9. Gaps found in spike L

1. **No `float_of_bits`.** (Known, `floating-point.md` 4.1.) Workaround for the binary loader: integers plus `float_of` times
   a table of 2046 exact powers of two (`math.ldexp(1.0, e - 1075)`), subnormals through `ldexp(float_of(m), -1074)`; verified bit for bit
   with `bits_of` on every value of every run. Cost: 40 lines in `spike_l.cho`, no measurable time (7 ms for 2.0 million values).
2. **No text-to-float in `std`** (`std.fmt` only prints). Reproducer: the learning rate cannot be passed as `0.5` on the command line.
   Workaround: the argument is the 16 hex digits of its bit pattern, decoded by the same exact decoder. Cost: the
   caller (`check_oracle.py`, `bench_l.py`) converts. Also the reason for the binary data files (section 2 correction above).
3. **Clock resolution is 1 ms** (`clock_ms`). Enough for fits of 0.3 s and up (0.3% resolution at the shortest); not for predict or per-iteration timing.
4. **Not a gap, a note for the next spike:** a `res` value (`Buffer`) cannot be declared before the `borrow mut heap` block that
   creates it, so the program's allocation, loading and fit all live inside one `borrow mut heap` block; `fs: &c Fs` is not a
   type, only `Fs(p)`. Neither cost anything but a rewrite.
5. **`std.math.exp` is slow for two reasons, both in `std/math.cho` (found by 8.3).** (a) `pow2(k)` is a loop of squarings and
   `exp` calls it twice; (b) `round_to_int` is `if v >= 0.0 { truncate(v + 0.5) } else { truncate(v - 0.5) }`, which compiles to a
   branch (that it is mispredicted is inferred from the timing, not read from counters). Fixing only these, with identical bits (`expcheck.cho`: 114 M points, 0 differing),
   takes the logistic fit loop from 0.125 s to 0.061 s at 100,000 rows with 8-row blocking (v3m_8 vs tuned in 8.3; the table
   part alone, 1.2x). Reproducer for (b): `scripts/spikes/variants/v3_8.cho` vs `v4.cho`, which differ only in `exp_t`'s rounding, `if kf >= 0.0 { k = truncate(kf + 0.5); } else { k = truncate(kf - 0.5); }` against `var h = 0.5; if kf < 0.0 { h = 0.0 - 0.5; } let k = truncate(kf + h);` (0.081 s -> 0.061 s, bit-identical); this is the shape of std's `round_to_int`. For (a): `v0` vs `v1`.
   Proposed: write `round_to_int` as the select, and take 2^k from a 2046-entry table or from a future `float_of_bits`.
6. **Per-element checks on `xs[base + j]` inside a 20-wide loop are not discharged** (`base = i * 20` comes from the outer loop).
   Every element pays an overflow-checked add, a compare and a branch. Reading the row as `xs[i*20..i*20+20]` once, or writing the
   index with `wrapping_add`, gives 1.45x (v1 0.231 s -> v2 0.160 s, v1w 0.166 s) on the fit; I found no way in the source to tell the loop "this index is
   below the length" for a fixed-width row, other than the subslice. Reproducer: `v1.cho` vs `v1w.cho` / `v2.cho` in `scripts/spikes/variants/`.
7. **`--backend cranelift` is 1.7x slower than the default LLVM on this float loop** (section 8.3), and this document's section 8
   called the default "Cranelift" while `cancho --help` of `a4572ea` says LLVM is the default (section 8 corrected).
8. **No fused multiply-add and no SIMD for the float loops** is not a gap that cost here: the 8-row blocking reached numpy
   without either. Whether a vectorised `exp` (numpy has one) would help the remaining 40% was not tried.

## 10. Spike S gate: a better solver for the same problem (2026-10-12; written before `spike_s.cho`)

**Question.** Section 8 compared cancho's 100 gradient-descent steps with scikit-learn's L-BFGS, which stops after 6
iterations: part of the 28x to 32x is *algorithmic* (fewer passes over X), not the speed of cancho's loop. Spike S gives
cancho a solver that needs few passes, fixed in advance, and measures again. Same data, same split, same problem
(mean log-loss, zero start, no penalty) as spike L.

**Order of events, stated plainly.** The parameters below (K, m, trials) were chosen **after** a short exploration in numpy
(the `oracle_s.py` transcription of the algorithms, 1,000 and 100,000 rows, K = 6 to 20; nothing was run in cancho yet).
What that exploration showed: Newton's coefficients differ from its own 12-step result by 1e-6 (K=5) and 1e-13 (K=6), so
K=6 is the first count at which the model is converged *and* K-1 is still distinguishable at 1e-9; L-BFGS (m=10) has the
same loss as scikit-learn at K=6, never backtracked on these data, and never reached a history longer than K. The gate
below was written from that, before the cancho code. The numbers are choices, not findings.

### 10.1 The algorithms (exact; both fixed iteration count, no stopping rule, no threads)

Common. Train rows = first n - n/5. Parameters w[0..19], bias w[20]; z_i = w[20] + x_i0 w[0] + ... + x_i19 w[19]
(added left to right, starting from the bias). With e = exp(-|z|): p = 1/(1+e) if z >= 0 else e/(1+e); row loss
l = max(z,0) + log1p(e) - y z. f = (sum of l over rows, ascending)/n. Gradient g_j = (sum_i (p_i - y_i) a_ij)/n, with a_i =
(x_i, 1). One evaluation of f and g = **one pass over X** (a single fused loop over the rows).

**Newton (N), K iterations (flag), no line search.** Each iteration is one pass: p = 1/(1+exp(-z)), g as above, s = p(1-p),
H_ab = (sum_i (s_i a_ia) a_ib)/n for b <= a (lower triangle, rows ascending); then d = H^-1 g by Cholesky H = L L^T (row by
row, every sum in ascending index order, pivot must be > 0 else the run refuses), forward then backward substitution;
w <- w - d. No damping, no regularisation, no jitter. Passes = K. **K = 6 (choice).**

**L-BFGS (L), K iterations, memory m, backtracking.** One pass first (f, g at w = 0). Iteration k: d = -H_k g by the
standard two-loop recursion over the stored pairs (s_i, y_i), newest first then oldest first, H0 = gamma I with gamma =
(s.y)/(y.y) of the newest pair (d = -g when there is no pair). Trial step t0 = 1/||g||_2 times 2^T0E at k = 0 (T0E a flag,
0 by default, used only by the stress case), t0 = 1 afterwards. Backtracking: at most **T = 10 trials**; trial j evaluates
f, g at w + t d (one pass each); accept at the first trial with f_new <= f + c1 t (g.d), c1 = 1e-4; else t <- t/2; **if the
10th trial also fails it is accepted anyway** (deterministic, no failure branch). Then s = w_new - w, y = g_new - g; the pair
is stored only if s.y > 0 (else skipped and counted), and only the m newest are kept. Passes = 1 + total trials.
**K = 8, m = 5 (choice;** m=5 and not 10 so that the history is actually full for K > m, otherwise the memory could not be
tested; a larger m costs no extra pass).

Both oracle and cancho add in the orders written above. The numpy oracle (`oracle_s.py`) uses BLAS for the O(n) sums, so
last-ulp differences from summation order are expected (section 8.1 saw 1 to 4 ulp for 100 steps); both solvers contract
errors (Newton quadratically), so the difference at the end should be well under 1e-9 and is measured, not assumed.

### 10.2 Gates (a result is void if any fails; no tolerance is loosened after a run, a change is a dated edit here)

* **G-oracle-S.** For N (K=6) and L (K=8, m=5): the 21 coefficients of cancho against `oracle_s.py` (same algorithm,
  nothing from scipy), **1e-9 relative per coefficient**, at 1,000 and 100,000 rows (1,000,000 as an extra, reported). The
  stress case is also checked: L with T0E = 10 on 1,000 rows, which must make the oracle backtrack (the check prints the
  backtrack count; if it is 0 the stress case is void).
* **G-mutants-S.** Cancho mutants (the source with one edit, rebuilt) and numpy-side mutants are compared against the
  unmutated oracle; **all must be rejected** at 1,000 and 100,000 rows, else the gate is void:
  N: K-1; s = p instead of p(1-p); update w <- w + d; update w <- w - 0.9 d; Cholesky back substitution reading L[i,k] instead of L[k,i].
  L: K-1; m = 4; m = 6; `alpha - beta` changed to `alpha + beta`; gamma scaling dropped; c1 = 0.5 (stress case); halving
  factor 0.5 -> 0.6 (stress case); trials = 1 (stress case).
  **Expected survivors, explained:** "skip pairs with s.y <= 0 removed" and "trials 10 -> 9" cannot be seen on data where the
  skip count is 0 and no run needs 10 trials; they are listed as survivors if they survive. A K+1 mutant is also expected to
  survive for N (converged to 1e-13 at K=6), so the gate pins K from below only, and says so.
* **G-metric-S.** Against scikit-learn `LogisticRegression(C=inf, solver="lbfgs", max_iter=1000)` fit on the same train rows:
  **|test AUC difference| <= 1e-5** and **|train log-loss difference| <= 1e-6** (absolute, mean log-loss) **(both choices;
  tighter than section 4's 0.002, because the aim here is the same solution, not the same ranking)**, at 1,000, 100,000 and
  1,000,000 rows. Also printed, not gated: both against the Newton K=12 optimum.
* **Dated edit, 2026-10-12, after the first run of the gate (which FAILED on this line at 1,000 rows):** the AUC tolerance is
  `max(1e-5, 1/(n_pos n_neg))` of the test set, i.e. never tighter than one swapped test pair. At 1,000 rows the test set has
  200 rows (about 108 positive), one swapped pair moves the AUC by 1.07e-4, and the first run showed exactly that
  (|dAUC| = 1.00e-4, Newton at the optimum against scikit-learn 1e-3 away from it in the coefficients). At 100,000 and
  1,000,000 rows the bound is 1e-5 as written above. The log-loss tolerance was not touched.
* **G-passes.** Reported for cancho (N: K; L: 1 + trials, printed by the program) and for scikit-learn: `n_iter_`, and the
  `nfev` of the scipy `L-BFGS-B` result obtained from the *actual* `fit` call by wrapping `scipy.optimize.minimize` (no
  reconstruction of the loss). One pass is one evaluation of loss and gradient; note scikit-learn's evaluation reads X
  twice (a matrix-vector product and its transpose) where cancho's fused loop reads it once, which is stated next to the
  numbers.
* **G-noise** unchanged: min of N >= 7 interleaved, load average at start and end, spread <= 1.25 or rerun.

### 10.3 Timing protocol

Fit only (file read and decode outside), 100,000 and 1,000,000 rows, interleaved rounds of: cancho N, cancho L, numpy N,
numpy L (same algorithms, BLAS), scikit-learn; one thread each (`VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1`; honoured by Accelerate unverified, as in section 8). cancho's clock is `clock_ms`. The Mac is shared:
the load average is printed with every row. If cancho does not reach scikit-learn, the results say so in a sentence.

## 11. Spike S results (2026-10-12; Mac only, one session; source `cancho-ml-solver` `m0-solver` `1679230`)

**Conditions.** Apple M4 Max, macOS 26.2 (arm64), cancho `a4572ea` (edition 5), `--std`, Cranelift (default) and
`--backend llvm`; [Record note: the default is LLVM, not Cranelift, as section 8.3 found with `cancho --help` of `a4572ea`; the two backends gave the same times within 3% here, and the rows "Cranelift (LLVM)" below are the default build followed by `--backend llvm`;] numpy 2.4.6, scipy 1.17.1, scikit-learn 1.8.0, Python 3.14; one thread (`VECLIB_MAXIMUM_THREADS=1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`, honoured by Accelerate unverified). Shared machine: load average 3.2 3.8 3.7 (1, 5,
15 min) at the start and end of both timing runs (the OS refreshes it every few seconds; printed in
the run output). Data as section 8 (same SHA-256). Scripts: `spike_s.cho`, `oracle_s.py`, `check_oracle_s.py`, `bench_s.py`.

### 11.1 Gates (`check_oracle_s.py`, run twice; the second after the dated edit of 10.2)

* **G-oracle-S: passed.** Worst relative coefficient difference to the numpy oracle: Newton 9.7e-16 (1,000), 2.2e-15 (100,000),
  8.2e-16 (1,000,000); L-BFGS 7.8e-16, 3.4e-15, 2.0e-15; stress case (T0E=10) 1.0e-15 and 6.5e-15, with 8 and 9 backtracks, equal in
  oracle and cancho, as are all pass counts. Tolerance 1e-9; room of 5 to 6 orders of magnitude.
* **G-mutants-S: every must-reject mutant rejected** (worst relative difference at 1,000 / 100,000 rows):
  N one fewer iteration 9.8e-7 / 2.2e-7; N s=p 0.69 / 0.85; N w+d inf / inf; N w-0.9d 1.8e-4 / 1.2e-4; N back-substitution index
  1.5e-2 / 4.2e-3; L one fewer iteration 1.0e-3 / 6.3e-5; L m=4 1.3e-4 / 5.6e-6; L m=6 6.0e-5 / 1.0e-6; L alpha+beta 1.1e-2 / 4.0e-4;
  L no gamma 5.5e-2 / 2.8e-2; L c1=0.5 (stress) 7.5e-3 / 3.1e-3; L halving 0.6 (stress) 4.7e-2 / 9.5e-2; L trials=1 (stress) 1.7e3 / 1.8e9.
  **Survivors, all predicted in 10.2:** N with one MORE iteration (7.1e-13 / 3.3e-14: Newton is converged at K=6, so the gate
  pins K from below only); L trials=9 in the stress case (1.0e-15: no run needs the 10th trial); L with the `s.y > 0` test
  removed (7.8e-16: that test is never taken on these data, `skipped` = 0). So the curvature-skip rule and the 10-trial
  cap are **untested by this gate**.
* **G-metric-S: passed** (after the dated edit; the first run FAILED, see 10.2). Against scikit-learn: test AUC differences
  1.0e-4 (1,000 rows, Newton; 0 for L-BFGS), 3.6e-7 / 3.9e-7 (100,000), 1.4e-7 / 1.5e-7 (1,000,000); train log-loss differences
  (cancho minus scikit-learn) -2.1e-8 / -2.0e-8, -5.0e-8 / -5.0e-8, -1.5e-7 / -1.5e-7, all negative: cancho's point has the
  *lower* loss; scikit-learn stops about 1e-7 above the optimum (Newton K=12 is identical to K=6 in the printed digits).

### 11.2 Passes over X and fit time (`bench_s.py`, N=7 interleaved, fit only)

Passes: cancho Newton 6; cancho L-BFGS 9 (1,000 and 100,000 rows) and 10 (1,000,000); scikit-learn `n_iter_` 6 with scipy
L-BFGS-B `nfev` 7 (100,000) and 8 (1,000,000) (from the real `fit` call, `scipy.optimize.minimize` wrapped to record the
result). scikit-learn's evaluation reads X twice (matrix-vector product and its transpose); cancho's fused loop reads it once.

| rows | variant | passes | min of 7 | spread | per pass |
|---|---|---|---|---|---|
| 100,000 | cancho Newton, Cranelift (LLVM) | 6 | 0.0340 s (0.0330) | 1.09 (1.09) | 5.7 ms |
| 100,000 | cancho L-BFGS, Cranelift (LLVM) | 9 | 0.0440 s (0.0450) | 1.11 (1.09) | 4.9 ms |
| 100,000 | numpy Newton, same algorithm | 6 | 0.0123 s | 1.15 | 2.1 ms |
| 100,000 | numpy L-BFGS, same algorithm | 9 | 0.0083 s | 1.10 | 0.9 ms |
| 100,000 | scikit-learn lbfgs, C=inf | 7 | 0.0094 s | 1.15 | 1.3 ms |
| 1,000,000 | cancho Newton, Cranelift (LLVM) | 6 | 0.3590 s (0.3620) | 1.04 (1.03) | 60 ms |
| 1,000,000 | cancho L-BFGS, Cranelift (LLVM) | 10 | 0.5310 s (0.5240) | 1.02 (1.04) | 53 ms |
| 1,000,000 | numpy Newton, same algorithm | 6 | 0.1217 s | 1.08 | 20 ms |
| 1,000,000 | numpy L-BFGS, same algorithm | 10 | 0.0978 s | 1.05 | 9.8 ms |
| 1,000,000 | scikit-learn lbfgs, C=inf | 8 | 0.1035 s | 1.03 | 13 ms |

(Per pass = min time / passes, rounded; train rows are 80% of the set.) All spreads are under G-noise's 1.25.
For reference, section 8 measured cancho GD at 0.295 s (100,000) and 2.958 s (1,000,000) in 100 passes; that was another
session at load 6.5 to 8, so the ratio below is indicative: it is not an interleaved measurement.

* **The algorithmic gap is closed: cancho needs 6 (Newton) to 10 (L-BFGS) passes against scikit-learn's 7 to 8**, and reaches
  the same or a lower loss, the same test AUC to 1.5e-7 at 100,000 rows or more, with every result checked against a same-algorithm numpy oracle.
* **cancho does NOT reach scikit-learn in seconds.** Best variant, Newton, one thread, this Mac, loaded: **3.6x slower than
  scikit-learn at 100,000 rows (0.034 vs 0.0094 s) and 3.5x at 1,000,000 rows (0.359 vs 0.1035 s)**; L-BFGS is 4.7x and 5.1x.
  The factor R of section 5 for the best solver is 3.5 to 3.6, in the 2 < R <= 10 row (the earlier GD arm was 28x to 32x). It is
  about 8x less time than cancho GD at 1,000,000 rows (0.359 vs 2.958 s, not same-session).
* What remains is the **cost per pass**: 53 to 60 ms against 13 ms (scikit-learn) and 10 to 20 ms (numpy, same algorithm) at 800,000
  rows, i.e. cancho's scalar fused loop is 4 to 5x slower per pass than BLAS plus numpy's vectorised `exp`/`log1p`. Where it
  goes was not profiled; cancho's Newton pass (231 multiply-adds per row on top of the gradient) costs only 13% more than its L-BFGS pass
  (which has `exp` and `log1p`), so the loop arithmetic is not the larger part. That is an observation of the two timings, not a diagnosis.
* The LLVM backend again gives the same times (within 3%) as Cranelift.
* Not measured: the 1,000-row timing, `gram`, threads, arm B/C (determinism) for these solvers, predict.

## 12. Gaps found in spike S

1. **No `float_of_bits` / text-to-float**: unchanged from section 9 (the loader and the flags still use the integer route).
2. **No linear algebra in `std`.** The 21x21 Cholesky, the triangular solves and the two-loop recursion are 120 lines of
   hand-written index arithmetic on one flat `[float]` workspace with offsets: there is no matrix type, no 2-D slice, no
   `dot`. It works and is checked, but every offset is a magic number (a typo shows up only as a failed gate).
3. **`fn` and other keywords cannot be identifiers** (`fn` as a variable name; trivial, found while writing). No `min`/`max` for
   floats beyond `std.math.fmin/fmax`; not needed here.
4. **`!` and `||` in conditions worked as expected; a `res` (heap) allocation still cannot be hoisted** (section 9, item 4).

## 13. Spike F gate (written 2026-10-12 before any spike F code; binding like sections 0 to 7; source `cancho-ml-forest` `m0-forest` `9c0d087`, oracle `e7cbb9e`)

Spike F is the random forest of section 2: T = 10 trees, depth 8, 64 bins per feature, `floor(sqrt(20))` = 4 features per
split, seeded bootstrap, 1,000,000 x 20 binary-label data of `ml-data/`, train = first `n - n/5` rows, test = the last
`n/5` (as spike L). One thread. Everything below is a **choice** (the algorithm is ours, not scikit-learn's) unless marked.
The oracle (`scripts/spikes/oracle_f.py`, numpy/Python) implements exactly this text, and the trees and probabilities of
cancho must be **identical to it, not close**.

### 13.1 Algorithm

Notation: `n` = number of training rows `ntr`; `X[i][f]` float64; `y[i]` in {0,1}; `D = 20`, `B = 64`, `T = 10`,
`MAXD = 8`, `MTRY = 4`; `seed` an int in [0, 2^63). Arithmetic on 64-bit words is modulo 2^64 where it says *wrapping*.
Refused (not run) if `n > 2^20` (the exact comparison of 13.5 is proven for node sizes up to 2^20), if a value is
non-finite, or a label is not 0 or 1.

**13.1.1 RNG.** SplitMix64, reference: Vigna, `prng.di.unimi.it/splitmix64.c` (public domain):
`state += 0x9E3779B97F4A7C15; z = state; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
return z ^ (z >> 31)`, all unsigned 64-bit wrapping, `>>` logical. `mix(z)` is the last three lines (the output function).
`uniform(m)` = `(next() >> 1) mod m` (63 bits, then remainder; a modulo bias below 2^-43 for m <= 2^20, accepted, **choice**).

**13.1.2 Derived streams.** `derive(seed, c1, c2, ...)`: `h = seed`; for each component `c`: `h = mix((h ^ c) + 0x9E3779B97F4A7C15)`,
i.e. the first output of a SplitMix64 whose state is `h ^ c`. The stream is a SplitMix64 started with state `h`.
Streams: bootstrap of tree t (t = 0..9): `derive(seed, 1, t)`. Features of node `id` of tree t: `derive(seed, 2, t, id)`.
Node ids are heap numbers: the root is 1, the children of `id` are `2 id` (left) and `2 id + 1` (right); depth(root) = 0.
No draw depends on the order in which nodes are visited, so any traversal order gives the same trees.

**13.1.3 Bins.** Per feature f, the training values of that column (all `n` rows, not the bootstrap) sorted ascending as
IEEE order (`v[0..n)`); edge `e[f][k] = v[floor(k n / 64)]` for k = 1..63 (integer division; an actual data value, so ties
and repeated edges can occur). `bin(x) = #{k in 1..63 : e[f][k] <= x}` in 0..63 (an element equal to an edge goes to the
**upper** bin). A split "at bin b" (b = 0..62) sends a row left iff `bin <= b`, equivalently iff `x < e[f][b+1]`; prediction
uses that float comparison, training uses bins.

**13.1.4 Bootstrap.** For tree t, draws `d_j = uniform(n)` from stream `(1, t)`, j = 0..n-1 in order; the tree's root row
multiset is `{d_j}` (repeats count). All node statistics are counts over this multiset with multiplicity.

**13.1.5 Node.** A node has row multiset R (size `m`), class counts `n0, n1` (n0 + n1 = m), depth d. It is a **leaf** iff
`d = 8`, or `n0 = 0`, or `n1 = 0`, or no valid candidate exists (below). `min_samples_leaf = 1`, `min_samples_split = 2`
(implied by 1 row per side), `min_impurity_decrease = 0` and **a split with zero decrease is taken** if it is the best
valid one (no gain threshold). Otherwise: draw the feature subset from stream `(2, t, id)`: array `a = [0..19]`; for
i = 0..3 { `j = i + uniform(20 - i)`; swap `a[i], a[j]` }; the subset is `a[0..4)` sorted ascending. For each subset
feature (ascending) and each b = 0..62 (ascending) the candidate has left counts `nl = sum_{k<=b} cnt[k]`,
`pl = sum_{k<=b} pos[k]` (rows of R with that bin / of class 1), right `nr = m - nl`, `pr = n1 - pl`; it is **valid** iff
`nl >= 1` and `nr >= 1`.

**13.1.6 Criterion and ties.** Gini. The weighted child impurity is `m - S` with `S = (pl^2+ql^2)/nl + (pr^2+qr^2)/nr`
(`ql = nl - pl`, `qr = nr - pr`), so the best split **maximises S**. `S = N / Dn` with the integers
`N = (pl^2+ql^2) nr + (pr^2+qr^2) nl`, `Dn = nl nr`. Candidate 1 beats candidate 2 iff `N1 * Dn2 > N2 * Dn1`, evaluated
**exactly in integers** (up to 2^97, so 128 bit; no float is involved in any split decision). Candidates are visited in the
order feature ascending, then bin ascending, and the best is replaced **only if strictly better**: so among exactly equal
gains the **lowest feature index, then the lowest bin** wins. The chosen split sends rows with `bin <= b` to child `2 id` and
the others to `2 id + 1`; children get their counts from `nl, pl`.

**13.1.7 Leaf and prediction.** Leaf value = the integer counts `(n0, n1)`. Tree probability `p_t = float(n1) / float(n0+n1)`
(one IEEE division). Forest probability of a row: `acc = 0.0; for t = 0..9 in order: acc = acc + p_t; result = acc / 10.0`.
The traversal compares the float `x[f] < e[f][b+1]`.

**13.1.8 Output format (the checker's input).** Text, one item per line: `E f k <hex bits of e[f][k]>` for all 20 x 63 edges;
`N t id feat bin n0 n1` for every node that exists, ascending t then id (feat = -1 and bin = -1 for a leaf); `P i <hex bits>`
for each test row i; then non-compared lines starting with `#` (`#fit_ms`, `#predict_ms`, `#load_ms`). The checker compares
all `E`, `N`, `P` lines **byte for byte**.

### 13.2 Reference vectors for SplitMix64

`scripts/spikes/splitmix64_ref.c` is Vigna's published `next()` verbatim, compiled with the system `cc` and run for the
seeds below (this is the verification: the constants and the function were read from the published page, then the C was
run, and both the Python oracle and the cancho program must print the same five values). The values for seed 0 also equal
the ones widely quoted for splitmix64 (`e220a8397b1dcdaf`, `6e789e6aa1b965f4`, `06c45d188009454f`, `f88bb8a8724c81ec`).

| seed (initial state) | outputs 1 to 5 |
|---|---|
| 0 | e220a8397b1dcdaf 6e789e6aa1b965f4 06c45d188009454f f88bb8a8724c81ec 1b39896a51a8749b |
| 1234567 | 599ed017fb08fc85 2c73f08458540fa5 883ebce5a3f27c77 3fbef740e9177b3f e3b8346708cb5ecd |
| 2^64-1 | e4d971771b652c20 e99ff867dbf682c9 382ff84cb27281e9 6d1db36ccba982d2 b4a0472e578069ae |

(1234567's first output, 0x599ed017fb08fc85 = 6457827717110365317, is also the value quoted for that seed in the common
descriptions of the generator; that equality is from memory, not re-fetched. *Correction 2026-10-12 in place: an earlier
draft of this table was shifted by one output for the last two seeds; the C program's 15 lines are 5 per seed.*)

(`derive` and `uniform` have no external reference: the Python and the cancho versions are compared with each other on
fixed inputs, printed by `oracle_f.py --vectors` and by `spike_f --vectors`.)

### 13.3 Oracle and the identity gate

* **G-F-ident.** At 1,000 rows and at 100,000 rows (and at 1,000,000 if the oracle finishes), seed 20261012: every `E`, `N`
  and `P` line of `spike_f` equals `oracle_f.py`'s byte for byte. Any difference fails the gate and voids spike F's timing.
* **G-F-mutants.** Each mutant below is a one-site source edit of `spike_f.cho` (script `scripts/spikes/mutants_f.py`,
  each edit asserted to match exactly once), built and run at 1,000 and 100,000 rows; it must be **rejected** by G-F-ident
  at one of the two sizes (a mutant rejected at neither survives and voids the gate, or is explained).
  1. tie rule: replace only if better-or-equal (so the highest feature, then the highest bin wins);
  2. bin boundary off by one: `e <= x` becomes `e < x` in the bin assignment (equal goes to the lower bin);
  3. bootstrap: draw j = 0 of tree 0 is `uniform(n)` plus 1 (mod n);
  4. direction: rows with `bin <= b` go right (the children swapped), in training and in prediction;
  5. depth 7 instead of 8;
  6. 3 features per split instead of 4.
  Also a ties-occur check: `oracle_f.py` counts the exactly-equal-gain comparisons it saw (the mutant 1 can bite only if
  there are some) and the count is printed with the verdict.
* **G-F-metric vs scikit-learn** (section 4): `RandomForestClassifier(n_estimators=10, max_depth=8, max_features=4,
  bootstrap=True, n_jobs=1, random_state=0)`; the test AUC of cancho (mean probability) must be within **0.005** of it
  at 100,000 and at 1,000,000 rows. Stated plainly: **the trees differ** (scikit-learn uses exact thresholds on unbinned
  values, its own RNG, `min_samples_leaf=1`, and keeps drawing features if the sampled ones have no valid split; ours uses
  quantile bins, SplitMix64, and does not). Depth 8 and 4 features per split match; sklearn's `n_estimators`, bootstrap size
  `n` and Gini criterion match; its leaf value is the same class proportion; its probability is the mean of tree
  probabilities (same idea, its own summation order, not bit-compared).

### 13.4 Timing (section 4 G-noise and section 5 apply)

`fit` = from the loaded float arrays to the 10 trees, **including** the quantile edges and the bin matrix (scikit-learn
needs none of those; that is a cost of this algorithm and is reported separately by ablation); `predict` = the 10 trees
over the test rows to the probabilities in memory. Both timed inside the program by `clock_ms` (1 ms). Min of 7 interleaved,
load at start and end, spread; sklearn `fit` is `RandomForestClassifier.fit` on the train rows, `predict` is
`predict_proba` on the test rows, `n_jobs=1`, thread-pinned env as in section 8. Sets: 100,000 and 1,000,000 rows. The
row of section 5 is computed from fit R only (that is what it says); the predict ratio is reported beside it.

### 13.5 What the exact comparison costs, and its bound (a design statement, measured in section 14)

`N` is below `m^3 <= 2^60` and `Dn <= m^2/4 < 2^38` for `m <= 2^20`, so `N1*Dn2` is below 2^98: it does not fit a 64-bit
`int` (cancho has no multiply-high; gap candidate). The program multiplies in base 2^30 limbs (three limbs of product),
exact, and compares the limbs. It runs once per candidate (<= 4 x 63 per node), against a histogram pass that touches every
row of the node 4 times, so its share is small by count; the ablation in 14.5 replaces it by a float comparison
(outputs then not comparable) to measure the difference rather than assume it.

## 14. Spike F results (2026-10-12; Mac only, one thread, one session; source `m0-forest` `e63ab76`)

**Conditions.** Apple M4 Max, macOS 26.2 arm64; cancho `a4572ea` (edition 5), `--std`, default backend (LLVM); numpy 2.4.6,
scikit-learn 1.8.0, Python 3.14; `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1`, `n_jobs=1` (that sklearn
really ran on one thread was not checked with a counter). Data and splits as section 8 (same SHA-256s). Seed 20261012.
**The machine was shared (5 users) and the one-minute load was 1.9 to 2.9 during every run below; no run had load < 2 at both
ends, so none is marked quiet.** Scripts: `oracle_f.py`, `check_f.py`, `bench_f.py`, `ablate_f.py`, `spike_f.cho`.

### 14.1 G-F-ident: passed, at 1,000, 100,000 and 1,000,000 rows

Every `E` (1,260 edge bit patterns), `N` (every node: feature, bin, class counts) and `P` (every test probability's bit
pattern) line of `spike_f` equals the oracle's, byte for byte (`check_f.py`): 1,000 rows (2,974 lines), 100,000 (26,344), 1,000,000
(206,370; 5,110 nodes = 10 full trees of 511). SplitMix64 / `derive` / `uniform` vectors (28 lines) are identical between
`oracle_f.py --vectors` and `spike_f --vectors`; the SplitMix64 lines were also checked against the published C (13.2).
The oracle is separate code (numpy, Python big integers for the comparison, a vectorised bootstrap), not a transcription of the cancho.
Exactly-equal-gain comparisons that the oracle saw: 15,759 (1,000 rows), 2,989 (100,000), 5 (1,000,000), so the tie rule is
exercised at the two smaller sizes and barely at 1,000,000.

### 14.2 G-F-mutants: all six rejected, no survivor

`check_f.py --mutants` (each a one-site edit of the cancho source, asserted unique, rebuilt and run against the oracle):

| mutant | result at 1,000 rows | result at 100,000 rows |
|---|---|---|
| tie rule: better-or-equal replaces | rejected | rejected |
| bin boundary `e <= x` -> `e < x` | rejected | rejected |
| tree 0 draw 0 plus one | rejected | rejected |
| children swapped (training and prediction) | rejected | rejected |
| depth 7 | rejected | rejected |
| 3 features per split | rejected | rejected |

(The mtry mutant scans 3 of the 4 sorted drawn features; it is not the oracle's `--mutant mtry`, which draws 3. Both change the
trees.) 1,000,000 rows was not run for the mutants.

### 14.3 G-F-metric vs scikit-learn: passed

Test AUC, cancho vs `RandomForestClassifier(n_estimators=10, max_depth=8, max_features=4, bootstrap=True, n_jobs=1,
random_state=0)`: 100,000 rows 0.826189 vs 0.826105 (difference 0.000084); 1,000,000 rows 0.826176 vs 0.827210 (**0.001033**),
both within the 0.005 limit. **The trees differ** (13.3): sklearn's thresholds are exact, ours are 64 quantile bins, and the RNGs
differ. Depth matches: all 10 sklearn trees reach depth 8 at 1,000,000 rows (checked with `get_depth()`).

### 14.4 Timing, one thread, this Mac (load listed; not quiet), min of 7 interleaved rounds

| set | what | min | spread (max/min) | load avg 1-min at start / end |
|---|---|---|---|---|
| 100,000 (80,000 train, 20,000 test) | cancho fit | 0.054 s | 1.074 | 2.44 / 2.83 |
| | sklearn fit | 1.031 s | 1.017 | |
| | cancho predict | 0.007 s | 1.143 | |
| | sklearn predict_proba | 0.0087 s | 1.108 | |
| 1,000,000 (800,000 train, 200,000 test) | cancho fit | 0.673 s | 1.046 | 2.83 / 2.55 |
| | sklearn fit | 13.222 s | 1.004 | |
| | cancho predict | 0.071 s | 1.085 | |
| | sklearn predict_proba | 0.0876 s | 1.076 | |

Ratios cancho / sklearn: **fit 0.052 at 100,000 and 0.051 at 1,000,000; predict 0.80 at 100,000 and 0.81 at 1,000,000.**
The 100,000-row predict cell is 7 ms on a 1 ms clock (+-14%). **Row of section 5** (R = fit ratio, G-metric passed): R <= 2,
"speed may be stated, with machine, size and threads" (this Mac, one thread, loaded, 100,000 and 1,000,000 rows). Read the next
paragraph before quoting it.

*What the ratio compares.* cancho does 64-bin histogram splits and sklearn does exact-threshold splits (sorting each node's
values per feature); the task and parameters are the same, the algorithm is not, and AUC agrees to 0.001. The comparison is
"fit this model on this data", not "the same algorithm in two languages". The cancho fit includes the quantile edges and the bin
matrix (below). No numpy same-algorithm timing exists for spike F (the oracle was written for exactness, took 2.3 s fit at
1,000,000 rows in the same session, and was not tuned or timed in rounds).

### 14.5 Where the time goes (cancho, 1,000,000 rows, min of 7 interleaved, load 1.86 / 2.01; `ablate_f.py`)

Phases printed by the program: quantile edges (key conversion, radix sort of each training column, 20 columns) 0.20 s; bin
matrix 0.026 s; growing the 10 trees 0.446 s; thresholds and probabilities 0 ms; **fit 0.677 s**; predict 0.071 s.
Ablation by doing one piece twice (outputs unchanged; the delta is that piece's cost, and is a *lower bound* because the second
pass runs on warm caches), `fit_ms` column, base 677:

| piece done twice | fit | delta |
|---|---|---|
| histogram build (4 features per row pass) | 874 | **+197 ms** |
| radix sort of the columns | 807 | +130 ms |
| partition | 727 | +50 ms |
| bootstrap draws | 685 | +8 ms (noise-level) |
| candidate scan (cumulative sums, comparisons) | 679 | +2 ms (noise) |
| bin matrix | 701 | +24 ms |
| **exact comparison replaced by a float cross-product** (trees may differ) | 669 | **-8 ms, inside the 1.04 to 1.06 spread** |

So **the exact integer comparison cost nothing measurable** here (it runs 4 x 63 times per node, 5,110 nodes). About 197 + 130 +
50 + 24 + 8 = 410 of 677 ms are attributed; the rest of `grow` (about 190 ms) is not separated: the attributed pieces were
measured warm, and the cold first pass over `idx`/`bins` is most likely where it sits, but that is an inference. A branchless
partition (`scripts/spikes/variants/spike_f_branchless_partition.cho`) gave 441 vs 445 ms of grow (no gain; not kept). Not
profiled: predict (71 ms for 200,000 rows x 10 trees), the bounds checks of the gathers `c0[r]`, `yb[r]`, `wk[i0]`, whether the
histogram loop is limited by the load-increment-store dependency; nothing tried to speed it up beyond the 4-features-per-pass
structure and subslices for the bin columns.

### 14.6 Gaps found for upstream (spike F)

1. **No multiply-high / 128-bit product.** The exact Gini comparison needs `N1*D2 > N2*D1` with values up to 2^98; `gt` in
   `spike_f.cho` is 22 lines of base-2^30 limbs. Cost in time: none measured (table above). Cost in effort: the bound proof
   (n <= 2^20) and the limb code. A `mul_hi`/`wide_mul` builtin would remove both; no reproducer needed beyond `gt`.
2. **No logical right shift** (`>>` is arithmetic, `bitwise.md`): every SplitMix64 shift is `(z >> k) & mask` (`mix`, `uni`).
   Cost: no time measured; a readability and review cost, and a place for a wrong mask (the vectors catch it).
3. **No sort in `std` for floats or ints.** Wrote an LSD radix sort (about 55 lines, `radix`); it is 130 ms of the 677 ms fit
   at 1,000,000 rows (20 columns of 800,000). A `std` sort or a selection routine for order statistics would be needed by any tabular package.
4. **`clock_ms` resolution 1 ms** (known, gap 3 of section 9): the 100,000-row predict is 7 ms, +-14%.
5. **A path containing `..` and a missing argument both end the process with SIGILL (exit 132)** instead of a refusal
   (observed with `./build/spike_f ../ml-data/n1000/x.bin ...` and with no arguments; a nonexistent absolute path is handled
   by my code). The `..` case is presumably the filesystem path check of `check-cost.md` (not confirmed here); the missing-argument case is my `arg(g, 3)`
   out of range. Relevant to the "no input reaches a panic" rule for M2. No time cost.
6. **Boilerplate, not speed:** `main` is about 150 lines of nested `borrow` blocks to pass nine boxes to five functions; a function
   cannot take the capability or a struct of slices (struct reference fields, cancho#251). Cost: rewrite time only.
7. Already known and met again: no `float_of_bits` (loader of section 9 gap 1 reused unchanged).

### 14.7 What this does not establish

Only the Mac, one thread, load 1.9 to 2.9; sklearn's thread use unverified by counter; `gram` and threads not run (the
single-thread bootstrap/streams are derived per tree and per node, which is what a later parallel version needs, but no
parallel version exists, so D1 for the forest is **not tested**); the ratio is between different split algorithms (14.4);
the 1,000,000-row mutants were not run; zero-gain splits are taken (13.1.5) and `-0.0` versus `0.0` ties in the sort were not
exercised (the data have no exact zeros); `uniform(m)` has the stated modulo bias; the edges are exact quantiles of all
training rows, which costs 30% of fit and a sampled edge rule would be cheaper and is a different (untested) algorithm.

## 15. Q-det on a Newton step: arm A, arm B (exact accumulator) and arm C (the fixed block tree). The gate, written before the spike

**Status: sections 15.1 to 15.6 were written (commit `f4a1c4e`, 2026-10-12 23:42, branch `armb` of `cancho-ml-armb`) before `scripts/spikes/q_det/`; section 15.7 (results, commits `3cb826d` and `d9c419c`, 2026-10-12 23:57) was appended afterwards and is the only
part that holds numbers.** It answers sections 3 to 5 of this document for the loop that decides the result of
logistic regression by Newton's method, and it is the measurement behind open question 4 of `docs/linalg.md` 5.7 (`order.cho` as the swap point).
Tags as in `design.md`: **[choice]** is a decision the maintainer can overturn.

### 15.1 What is summed (the decisive sums)

One Newton step at weights `w` (21 values: 20 weights and the bias `w[20]`), on the first `n - n/5` rows of the M0 data
(`ml-data/n100000`, `ml-data/n1000000` (written by `scripts/spikes/gen.py`): 20 columns, SHA-256 in each `meta.json`), exactly the step of
`scripts/spikes/spike_s.cho` (section 10) (`oracle_s.py` is its numpy twin). Per row `i`, with `x_i` the 20 features and `x_i[20] = 1.0`:

* `z = w[20] + x_0 w_0 + ... + x_19 w_19` (ascending, as spike S); `p = 1 / (1 + exp(-z))`; `r = p - y`; `s = p (1 - p)`.
* **gradient**: 21 sums `g_a = sum_i r * x_a` (the 21st, the bias, is `sum r`).
* **Hessian**: the lower triangle, 231 sums `H_ab = sum_i (s * x_a) * x_b`, `b <= a` (two rounded products, as spike S).
* **log-loss**: 1 sum `sum_i ((pos + log1p(e)) - y z)`, `e = exp(-|z|)`, `pos = max(z, 0)` (spike S's L-BFGS form; spike S does not
  sum it in Newton, here it is the third kind of decisive sum and is summed at every iteration, at the `w` the step starts from).

That is **253 sums** (21 + 231 + 1), not the 231+ of the brief: spike S has a bias (d = 20 features, 21 parameters). Everything
else in the step (the 21 x 21 Cholesky solve and the update) is 21-dimensional, serial, identical in every arm, and is not
a decisive sum (its order is fixed by `docs/linalg.md` section 2.1). The fit is **6 Newton iterations from w = 0**, no stopping rule.
`exp` is spike S's `math.exp` bit for bit through the table form `exp_t` of `spike_l_tuned.cho` (2^k from the exact table,
same operations; `scripts/spikes/q_det/` checks it against `math.exp`). `n` in {100,000; 1,000,000} (train rows 80,000 and 800,000).

### 15.2 The arms (all three use the same per-row code; only the accumulation differs)

| arm | accumulation | thread story |
|---|---|---|
| A | one running `float` per sum, rows ascending from `+0.0` (spike S's order) | none: one thread (the baseline) |
| B | one **exact accumulator** per sum (`facc.cho` of cancho-table, vendored by copy: 73 integers = 72 limbs of 32 bits + a counter per sum; 253 x 73 integers = 148 KB of state, rounded once at the end) | any partition of rows; merge = limb-wise integer addition, exact |
| C | **`docs/linalg.md` section 2 as it stood on branch `armb` (before the lanes of `docs/lanes.md`)**: blocks of K = 256 rows, ascending from `+0.0`, partials combined by the fixed binary tree (`order.settle`/`settle_root`/`finish`, width 253); threads take whole chunks of G = 8 blocks (2,048 rows) and return chunk roots | partition independent of the thread count |
| N | reference only: per-row terms computed, three terms added (`r`, `s`, loss); **no 253-sum accumulation**. It is not an arm; it measures what is not the sums | one thread |

The thread story is shared with `lib/linalg/par.cho`: a thread owns **copies** of its rows (cancho cannot lend a shared
read-only slice, U2 of `docs/linalg.md` 5.6), made per Newton iteration, and the copy is part of the measured time. The partition of rows into
chunks of 2,048 is a function of the row count only. A threaded Newton iteration is: copy, spawn, join, combine, Cholesky.

### 15.3 Correctness gates (a result is void if any fails)

* **G-A** (the baseline is the baseline): arm A's 21 coefficients equal `spike_s` (`scripts/spikes/spike_s.cho`, unchanged) bit for bit at
  both sizes (Newton, 6 iterations). If not, the "plain sequential" arm is not spike S's order and every ratio to A is void.
* **G-B-exact** (B against exactly rounded sums): for one step from a fixed `w0` (`scripts/spikes/q_det/oracle_q.py`), the
  253 raw sums of arm B, bit for bit, against `math.fsum` of the same 253 per-row term columns (the terms are dumped as bits by the
  spike, not recomputed). B must be **bit-identical in all 253**, at n = 100,000 (and at 1,000,000 if the 1.6 GB dump
  is practical; else said so).
* **G-C-order** (C against the order-matched numpy reference): the same 253 sums of arm C, bit for bit, against block rule + tree
  of `oracle_linalg.tree` (K = 256) on the same dumped terms. Arm A's sums likewise against the sequential left-to-right order
  (numpy, row loop).
* **G-det**: arms B and C produce **byte-identical models** (21 coefficients) **and byte-identical loss lines** at threads
  1, 2, 4, 8, 16, at both sizes. (The Mac only here: a second machine, `gram`, is not used in this stage; the cross-machine
  claim is therefore **not made**.)
* **G-mutants**: a build of B that finalises each thread's accumulators (rounding them) and adds the thread results as doubles in
  thread order, and a build of C that adds one partial per thread in thread order, must each **fail** G-det at more than one
  thread (identical to the real build at one thread). A mutant that passes voids G-det.
* **G-noise**: every timing is the minimum of 7 samples taken **interleaved** (A, B, C, A, B, C ...), `uptime` load average at the
  start and the end of the whole run and at each cell's end, spread = max / min; **a cell with load >= 2 at either end is flagged**,
  and a cell with spread > 1.25 is rerun once and, if it stays, printed with the spread marked. Timed region: the fit (the 6
  iterations, including thread copies) after the data are loaded and decoded; load time is outside.

### 15.4 What is reported (and what is not)

Per arm: fit time at one thread, ratios **B/A** and **C/A** at one thread (both sizes); B and C at 1, 2, 4, 8, 16 threads (time
and byte check); the distance in ulps of the A, B and C coefficients from each other; test AUC and test log-loss (the last
`n/5` rows) of the models of A, B and C, to say whether exactness changes the model in practice; a memory-traffic account for B.
Not reported: any comparison with another library's speed (not the question here), any claim about `gram` or another core count.

### 15.5 Decision rule (from section 5 of this document, restated, **[choice]**)

If B costs more than **3x** A at one thread and C does not, M1 (`docs/design.md` section 6) specifies C. If both cost less than 3x, M1 picks the
simpler to specify and says why. If both cost more, M1 reopens the determinism promise rather than weakening it silently. The
ratio used is the fit time, one thread, minimum of 7, at the size where they differ; if the two sizes straddle 3x, both are
said. The rule is applied to the **fit time** (the user-visible cost), and the same ratio on the sums alone (minus arm N) is
shown as well, because the per-row `exp` and `log1p` are not a cost of the determinism rule.

### 15.6 What this does not say

It is one loop on one machine (Apple M4 Max, 16 cores, macOS), one compiler (`a4572ea`, default backend), one data set. It does
not say what B costs on another loop (e.g. one sum per row instead of 253, or a sum over a column read once), nor on `gram`.
A is not claimed to be deterministic across thread counts, and it is not timed with threads.

### 15.7 Results (2026-10-12; `scripts/spikes/q_det/`, raw lines in `docs/bench/q_det.jsonl`, `q_det_gates.txt`, `q_det_rerun.txt`)

**Conditions.** Apple M4 Max (16 cores; 12 performance + 4 efficiency per the specification, not checked here), macOS 26.2, cancho `a4572ea`, default backend (LLVM, section 8.3), `cancho build --std`;
data `ml-data/n100000` (x.bin SHA-256 `896ce7c8...`) and `n1000000` (`a5d34dd3...`). Fit only (6 Newton iterations from w = 0, 20
features + bias, 253 sums per step, thread copies included), `clock_ms` (1 ms). Minimum of 7 samples, interleaved (A, B, C, N in
turn; B and C by thread count in turn), spread = max / min. `facc.cho` vendored unmodified from cancho-table `c85da3b` (file SHA-256
`f199fef4...` includes a 3-line header). **Load: the machine was shared (5 users) and busy. The 1-minute load average was 5.2 at the start
and 5.6 at the end of the n = 100,000 run, 6.3 at the start and 8.4 at the end of the n = 1,000,000 run: EVERY cell is flagged (load >= 2).**
These are not quiet-machine numbers. The spreads are nevertheless small (below); a ratio taken inside one interleaved run shares its load.

**One thread (ms, min of 7; spread in brackets).**

| n | A | B | C | N (terms only) | B / A | C / A | (B - N) / (A - N) | (C - N) / (A - N) |
|---|---|---|---|---|---|---|---|---|
| 100,000 (80,000 train rows) | 46 (1.04) | 636 (1.01) | 46 (1.04) | 16 (1.06) | **13.8** | **1.00** | 20.7 | 1.00 |
| 1,000,000 (800,000 train rows) | 468 (1.03) | 6,350 (1.01) | 466 (1.01) | 165 (1.02) | **13.6** | **1.00** | 20.4 | 0.99 |

At 1,000,000 rows B costs 5.2 ns per added term (1.21e9 terms), A and C 0.25 ns per term beyond arm N's 0.14 ns per term of row work.
Phase 2 re-measured B and C at one thread inside the thread table (B 639 / 6,370, C 46 / 456 ms): same within 1 to 2 %.

**Threads (ms, min of 7; spread in brackets where above 1.05).** Every model byte string at every thread count and every sample equals the
one-thread bytes of the same arm (`bytes_distinct = 1` in every cell, all 7 samples).

| n | arm | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| 100,000 | B | 639 | 326 | 167 | 87 (1.08) | 86 |
| 100,000 | C | 46 | 25 | 13 (1.23) | 8 (1.5, marked; rerun of 15 samples: min 11, spread 1.36, stays above 1.25) | 8 (1.25) |
| 1,000,000 | B | 6,370 | 3,216 | 1,630 | 851 | 604 (1.22) |
| 1,000,000 | C | 456 | 252 | 141 (1.23) | 83 | 67 (1.21) |

Copy + spawn + join only (MUT = 2, threads do no work, min of 3): 2 to 6 ms (n = 100,000) and 21 to 27 ms (n = 1,000,000) for the
6 iterations, whatever the arm and thread count: about 37 % of C's 67 ms at 16 threads, 4 % of B's 604 ms. The 8 to 16 thread
step is small for both arms (12 performance cores, other users' load).

**Correctness gates** (`oracle_q.py`, `gate_q.py`, `mutants_q.py`; all passed at both sizes):

* G-A: arm A's 21 coefficients equal the unchanged `spike_s` bit for bit (both sizes).
* G-B-exact: from a fixed W0 (arm C after 3 iterations), 253 / 253 sums of arm B equal `math.fsum` of the 253 dumped term columns,
  bit for bit (80,000 and 800,000 rows; the terms are the 253 x 800,000 doubles B added, dumped by the spike).
* G-C-order: 253 / 253 sums of arm C equal block rule (K = 256) + `oracle_linalg.tree` on the same terms; G-A-order: 253 / 253 of arm A
  equal rows ascending from +0.0. (The terms of A and C are computed by their own code; they enter only through these equalities.)
* G-det: B and C, threads 1, 2, 4, 8, 16: model bytes (21 coefficients + 6 loss lines) identical, and the 253 raw sums of the first
  step identical, at both sizes.
* G-mutants (determinism): the thread-order mutants differ from the real bytes at every thread count above 1 on the raw sums, both arms,
  both sizes, and on the model bytes in these cases: B: all four counts at both sizes; C: all four at n = 100,000, but only 2, 8
  and 16 at n = 1,000,000 (equal at 4). **The coefficients alone do not always separate them**: at n = 1,000,000 neither mutant changes any of
  the 21 coefficients at any count (Newton's last step damps the noise below the last bit), and at n = 100,000 the B mutant changes
  them only at 2 and 16 threads. This is why the gate compares the loss lines and the raw sums too. Both mutants are equal to the real arm at one thread, as designed.
* G-mutants (correctness): 7 source mutants of `spike_q.cho` / `order.cho` (a 255-row block; the tree as a left fold; a lost sign flip in
  the exact decode; a term into the wrong accumulator; a term perturbed by one ulp in B; a perturbed Hessian product in A; a perturbed loss term in A)
  all killed by the oracle gates; control passes; no survivor.

**How far apart are the models** (1 thread, `compare_models.py`): distance of the 21 coefficients in ulps, A vs B and A vs C: max 13
(n = 100,000) and 7 (n = 1,000,000); **B vs C: max 1 and 0** (identical at 1,000,000). Against the float64 numpy Newton of `oracle_s.py`
(a third order): A 2.2e-15 and 8.2e-16 relative, B 1.9e-16 and 2.2e-16, C 1.8e-16 and 2.2e-16. The sums themselves differ more than the model:
at the first step from W0, the largest of the 253 sums is 781 ulps (n = 100,000) and 2,023 ulps (n = 1,000,000) from the exact one for A, and 48 and 111 for C.
**Test AUC and test mean log-loss are identical to the last printed digit for A, B and C** (n = 100,000: AUC 0.8855675278341977, log-loss
0.4234154629542849; n = 1,000,000: 0.8866827408632275, 0.42042014227342633). On this problem exactness changes the coefficients by a few ulps and the model's quality by nothing measurable.

**Memory traffic of B (estimate, not measured: there is no hardware counter here).** State 253 x 73 x 8 = 147,752 bytes against a
2,024-byte accumulator block for A and C; per added term B loads and stores four 8-byte words (three limbs and the counter) where A
loads and stores one, plus about 30 integer operations (shifts, masks, a signed add) and two bounds-checked `vec` accesses per limb.
The measured 5.2 ns per term is about 20 cycles at 4 GHz **[estimate of the clock]**. Whether the cost is the instruction count or the
cache footprint was not separated (no variant with a small footprint was built). B uses cancho-table's code unmodified; a tuned accumulator
(unchecked indices, 64-bit limbs, a carry-save sparse form) might be several times faster, but not 4.5 times, the factor needed to reach 3x.

**Record note (2026-10-14, a rerun of the correctness gates only, no timing).** On the tree of this repository, whose `lib/linalg/order.cho` has
the four lanes of `docs/lanes.md` (the branch `armb` had the order before them), `sh scripts/spikes/q_det/build.sh`, `python3 scripts/spikes/q_det/oracle_q.py 100000`
and `python3 scripts/spikes/q_det/gate_q.py 100000 1 2 4` build and print the same verdicts as above for n = 100,000: G-A, G-B-exact (253/253),
G-C-order (253/253) and G-A-order (253/253) pass, with the same 781 ulps (A against exact) and 48 (C against exact); G-det holds at 1, 2 and 4 threads
for B and C and the thread-order mutants of both are killed (coefficients alone differ at 2 threads for B and at 2 and 4 for C, in this run). The
spike's sums do not use the lanes. Not rerun: 1,000,000 rows, 8 and 16 threads, the source mutants, any timing.

**Verdicts.** (1) Decision rule (15.5): B / A = 13.8 and 13.6 at one thread, above 3x; C / A = 1.00 at both sizes, below 3x. **The rule
selects C for M1 (`docs/design.md` section 6)** (a recommendation; the rule is the maintainer's). The same conclusion holds on the sums alone (20x against 1.0x). (2) Both
arms keep the bytes at 1 to 16 threads; the mutants that must fail do. (3) Exactness buys nothing visible in the model here.

**What was not done / does not hold.** The Mac only: no `gram`, no cross-machine byte identity (G-det asked for it; not claimed). Arm A was
not timed with threads. 253 sums, not 231 (a bias). All cells are load-flagged and the spread of C at 8 and 16 threads (n = 100,000)
and of B at 16 threads (n = 1,000,000) is above 1.2, one of them above 1.25. Thread copies are per iteration (U2). The row-term
code is duplicated in three loops (A and C share one); equality of the terms is checked only through the sums. `rows_exact` has a
`dumpon` branch per term in the timed path (predictable; its cost was not isolated). One loop, one data set, one compiler.

## 16. The SIMD ceiling in C (summary; the document is `docs/simd-ceiling.md`)

An upper-bound experiment, not one of the spikes of issue #1: how fast the cancho-ml kernels could get on this Mac if SIMD were
available and the summation order were SIMD-friendly, with **C and NEON intrinsics compiled by Apple clang** (no cancho compiler is
involved in the kernels). It is a ceiling for a C compiler given hand-written intrinsics, not a prediction of what cancho would emit.

* **Source.** `cancho-ml-simd`, branch `simd-ceiling`: gate and lane orders `bac2412` (2026-10-12 22:22), kernels and checkers
  `4152248` (23:41), results `5814443` (2026-10-13 00:04). The document, `docs/simd-ceiling.md`, carries no date of its own.
  Scripts: `scripts/simd/` (`simd.c`, `kern.h`, `bench_simd.py`, `report.py`, `bw.c`, `peak.c`). Raw data: `docs/bench/simd-gate.jsonl`,
  `docs/bench/simd-gate.log`, `docs/bench/simd-gate.load.txt`, `docs/bench/simd-check.log`, `docs/bench/simd-probes.txt`, the tables
  `docs/bench/simd-tables.md`, and the preliminary loaded run `docs/bench/simd-gate-prelim-loaded.jsonl` (kept as raw data, not used).
* **Conditions** (its section 3): Apple M4 Max, macOS 26.2, Apple clang 21.0.0 with `-O3 -mcpu=native -fno-fast-math -ffp-contract=off`,
  one thread; numpy 2.4.6 / scipy 1.17.1 with Accelerate and the thread variables set to 1 by the script. Load average (1 min) 2.2 to 3.2
  for the cells from `dot` to `syrk` 1e6 x 64 and 5.3 to 6.3 for `syrk` 1e6 x 256 and the three Cholesky cells: **every cell is flagged
  `LOAD>=2`**. Minimum of 7 interleaved samples; one cell (the scalar C timing of gemv_n 1e6 x 256) has a spread of 1.37 after its reruns.
* **Result** (its section 0, quoted as the document states it): 23 gate cells; each C kernel bit-identical to a scalar reference of a
  documented lane order for L in {1, 2, 4, 8, 16} (162 mutants killed, 22 equivalent and explained, none survive); matches numpy/scipy
  in 4 of 23 cells (gemv_n at d = 20, Cholesky d = 21 and 64) and does not in 19: dot 2.35x slower, axpy 2.12x, gemv_n d >= 64 1.47x to
  1.68x, gemv_t 1.43x to 2.39x, syrk 2.85x to 13.6x, Cholesky d = 256 1.67x; fma flips no verdict. syrk and Cholesky are at 75% to 98% of
  the measured NEON compute peak; the large-d gemv cells are bounded by single-core read bandwidth (114 to 121 GB/s measured) below
  numpy's own 114 to 163 GB/s. **L = 4** is the lane count it recommends; `docs/design.md` section 17 and `docs/lanes.md` record that
  four lanes were adopted for the summation order.
* Its references to `linalg.md` sections (1.1, 1.2, 1.3, 2, 5.2, 5.3, 5.6, 5.7) are to `docs/linalg.md` as it stood on branch `simd-ceiling`
  (328 lines), whose section 2 was the order **before** the lanes; the same section numbers exist in the current file, with section 2 revised.

## 17. Gaps found for upstream, with reproducers

Each row cites the section that found it and quotes its measured cost. Reproducers that exist as files in `docs/upstream/` are linked and
not copied here; reproducers that exist only as variants of the spike programs are in `scripts/spikes/variants/`. The gate asked for these
to be filed as issues in `alpibrusl/cancho`; **whether they were is not recorded in any source of this file.**

| gap | found in | reproducer | measured cost, as the section states it |
|---|---|---|---|
| no `float_of_bits`; workaround is a table of 2046 exact powers of two | 9 item 1 (known: `floating-point.md` 4.1 of the compiler) | the loader in `scripts/spikes/spike_l.cho` | 40 lines; no measurable time (7 ms for 2.0 million values) |
| no text-to-float in `std` (`std.fmt` only prints) | 9 item 2 | none beyond the statement (a learning rate cannot be passed as `0.5`) | also the reason the data are binary files, not CSV |
| `clock_ms` has a resolution of 1 ms | 9 item 3, 14.6 item 4 | `docs/upstream/clock_resolution.cho` | 0.3% at a 0.3 s fit; 14% at a 7 ms predict |
| `std.math.exp`: `pow2` is a loop of squarings, `round_to_int` compiles to a branch | 9 item 5 (found by 8.3) | `scripts/spikes/spike_l.cho` against `scripts/spikes/variants/v1.cho` (table `pow2`), `variants/v3_8.cho` against `variants/v4.cho` (the rounding); `scripts/spikes/expcheck.cho` | 0.125 s to 0.061 s at 100,000 rows with 8-row blocking, bit-identical on 114,000,002 points |
| per-element bounds and overflow checks on `xs[base + j]` in a 20-wide loop | 9 item 6 | `variants/v1.cho` against `variants/v1w.cho` and `variants/v2.cho` | 1.45x (0.231 s to 0.160 s) |
| `--backend cranelift` slower than the default (LLVM) | 8.3, 9 item 7 | `cancho build --backend cranelift` on `scripts/spikes/spike_l.cho` | 1.7x on the original (4.689 s against 2.794 s) |
| no fused multiply-add, no SIMD for float loops | 9 item 8 | `docs/linalg.md` 5.6 U1; the C ceiling of section 16 | not a gap that cost in spike L; see section 16 for the kernels |
| no linear algebra in `std`: the 21 x 21 Cholesky and the L-BFGS recursion are hand-written index arithmetic | 12 item 2 | `scripts/spikes/spike_s.cho` | effort only |
| keywords as identifiers (`fn` there; `res` and `val` in the reproducer) | 12 item 3 | `docs/upstream/keywords.cho` | usability |
| a `res` allocation cannot be hoisted out of its `borrow mut heap` block | 9 item 4, 12 item 4 | none | a rewrite |
| no multiply-high: the exact Gini comparison needs a product up to 2^98 | 14.6 item 1 | `docs/upstream/mulhi.cho` | none measured (the exact comparison cost about nothing, 14.5) |
| no logical right shift | 14.6 item 2 | `docs/upstream/lshr.cho` | none measured |
| no sort in `std` | 14.6 item 3 | `docs/upstream/sort_std.cho` | the radix sort is 130 ms of the 677 ms fit at 1,000,000 rows |
| a path with `..` and a missing argument end the process with SIGILL (exit 132) | 14.6 item 5 | `docs/upstream/dotdot_path.cho` | none |
| about 150 lines of nested `borrow` blocks in `main`; a function cannot take a struct of slices (cancho#251) | 14.6 item 6 | none | rewrite time only |
| a thread cannot be given a shared read-only slice, so threaded code copies its rows | 15.2, 15.7 (U2 of `docs/linalg.md` 5.6) | `docs/upstream/spawn_shared_slice.cho` | the copy is inside the timed region: 2 to 6 ms (n = 100,000) and 21 to 27 ms (n = 1,000,000) for the 6 iterations |
| `copy_into` takes bytes only | `docs/linalg.md` 5.6 U3 | `docs/upstream/copy_into_float.cho` | the loop is the copy above |
| `cancho test` cannot run a test that spawns | `docs/linalg.md` 5.6 U4 | `docs/upstream/test_runner_conc.cho` | |

The proposal to cancho-table that came out of the same work is `docs/upstream/cancho-table-numeric-package.md`.

## 18. M0 outcome: the answer to the gate of issue #1

Nothing in this section is new: every figure is quoted from the section named, with the conditions of that section. **All timings are
from one Mac (Apple M4 Max, macOS 26.2), one thread unless the row says otherwise, cancho `a4572ea`, on 2026-10-12, with the machine
shared and loaded (the one-minute load average is in the last column; no run is quiet except where marked), and all of them were taken
before the 4-lane summation order of `docs/lanes.md`. They are the record of the decision, not claims about the speed of `ml` today.**
`gram` (the Linux x86-64 machine the issue also asks for) was not used for any of them.

### 18.1 The factor against the reference, per spike

R is cancho's fit time divided by the reference's, one thread, minimum of N = 7 interleaved runs, 1,000,000 rows (800,000 train rows) unless
shown; spread = max/min of the 7 runs.

| spike and variant | reference | factor R | load average (1 min) at start / end | spread (cancho / reference) | metric gate (G-metric) | section |
|---|---|---|---|---|---|---|
| L, original arm A (100 gradient steps) | scikit-learn L-BFGS, 6 iterations | **28x** at 1,000,000 rows, 32x at 100,000 | 6.68 / 6.76 (cancho), 6.54 (scikit-learn) | 1.011 / 1.06 (1M) | passed: test AUC 5e-6 and under 1e-6 apart | 8.2 |
| L, original arm A | numpy, same algorithm | 4.0x (1M), 4.5x (100,000) | as above | 1.011 / 1.039 | | 8.2, 8.3 |
| L, **tuned arm A** (bit-identical output) | numpy, same algorithm | **0.89x** (1M), 1.0x (100,000): cancho is not slower | 3.20 / 2.83 (1M) | 1.057 / 1.067 | | 8.3 |
| L, **tuned arm A** | scikit-learn L-BFGS | **6.3x** (1M), 7.3x (100,000) | 3.20 / 2.83 | 1.057 / 1.08 | passed | 8.3 |
| L, tuned arm A, **quiet-moment rerun** | scikit-learn | 6.3x (1M), 7.4x (100,000) | 1.81 / 1.96 (1M); 1.96 / 1.97 (100,000) | 1.021 / 1.06 | passed | 8.3 |
| S, **Newton**, 6 passes | scikit-learn L-BFGS, `C=inf`, 8 function evaluations | **3.5x** (1M), 3.6x (100,000) | 3.2 / 3.2 | 1.04 / 1.03 | passed: test AUC 1.4e-7 apart at 1M (1.0e-4 at 1,000 rows, after the dated edit of 10.2) | 11.2 |
| S, L-BFGS, 10 passes | scikit-learn | 5.1x (1M), 4.7x (100,000) | 3.2 / 3.2 | 1.02 / 1.03 | passed | 11.2 |
| F, histogram forest, fit | scikit-learn `RandomForestClassifier` | **0.051** at 1M, 0.052 at 100,000 (cancho's fit takes that fraction of the time; **the algorithms differ**) | 2.83 / 2.55 (1M); 2.44 / 2.83 (100,000) | 1.046 / 1.004 (1M) | passed: test AUC 0.001033 apart at 1M (limit 0.005) | 14.3, 14.4 |
| F, predict | scikit-learn `predict_proba` | 0.81 at 1M, 0.80 at 100,000 (the 100,000-row cell is 7 ms on a 1 ms clock) | as above | 1.085 / 1.076 (1M) | | 14.4 |
| Q-det, arm B (exact accumulator), 6 Newton iterations | arm A (plain sequential sum) | **13.8x** (100,000), **13.6x** (1M) | 5.2 / 5.6 (100,000); 6.3 / 8.4 (1M) | 1.01 / 1.04 (100,000) | bytes identical at 1 to 16 threads | 15.7 |
| Q-det, arm C (fixed block tree) | arm A | **1.00x** at both sizes | as above | 1.04 / 1.04 (100,000) | bytes identical at 1 to 16 threads | 15.7 |

Cores: the sources state the core count only in section 15 (16, 12 performance and 4 efficiency "per the specification, not checked here").
Spikes L, S and F ran one thread on the shared machine and do not state a core count. numpy and scikit-learn were pinned to one thread by
environment variables; that Accelerate and scikit-learn obeyed was not verified (sections 8, 14).

The sentence the issue asks for, in the terms of the sources. *Logistic regression:* after tuning, cancho's gradient descent takes 0.89 of
numpy's time for the same algorithm at 1,000,000 rows (4.0 before), and **the gap to scikit-learn's L-BFGS is a factor of 6.3 at 1,000,000
rows for gradient descent (28 before tuning) and 3.5 for Newton's method**, the remainder being the cost per pass over the data: 53 to 60 ms
for cancho against 13 ms for scikit-learn and 10 to 20 ms for numpy at 800,000 rows (11.2; where it goes was not profiled).
*Random forest:* cancho's fit takes 0.051 of scikit-learn's time at 1,000,000 rows on 64-bin histogram splits against exact thresholds, with
test AUC 0.001 apart; it is not a same-algorithm comparison (14.4).

### 18.2 The decision rule (section 5), applied to each spike

| R (fit, one thread, minimum of N, against scikit-learn) | what the pitch may say |
|---|---|
| R <= 2 | speed may be stated, with machine, size and threads |
| 2 < R <= 10 | lead with reproducibility; say nothing about speed; document the factor in `docs/spikes.md` |
| R > 10 | reproducibility only, and M1 sets a smaller row limit than the matrix could hold; the factor is recorded as the gap |

| spike | row | the reading in the source |
|---|---|---|
| L, original arm A (R = 28 and 32) | R > 10 | 8.2: "on a loaded machine and on arm A, the fastest arm" |
| L, tuned arm A (R = 6.3 and 7.3; 7.4 at the quiet moment) | 2 < R <= 10 | 8.3: "this stays in the 2 < R <= 10 row"; it compares 100 gradient steps with 6 L-BFGS iterations |
| S, Newton (R = 3.5 and 3.6) | 2 < R <= 10 | 11.2 |
| F (R = 0.051 and 0.052) | R <= 2 | 14.4: "speed may be stated, with machine, size and threads (this Mac, one thread, loaded, 100,000 and 1,000,000 rows). Read the next paragraph before quoting it": the algorithms differ |

Determinism arms (section 3, the rule after the table in section 5: if B costs more than 3x A at one thread and C does not, M1 specifies C):
B / A = 13.8 and 13.6 (above 3x), C / A = 1.00 at both sizes, so **the rule selects C, the fixed block tree** (15.7: "a recommendation; the
rule is the maintainer's"). B and C keep the model bytes at 1, 2, 4, 8 and 16 threads and the thread-order mutants of each fail the check
at more than one thread; on this problem the exact sum changes the coefficients by a few ulps and the test AUC and log-loss not at all
(15.7). The cross-machine claim (the Mac and `gram` giving the same bytes) was **not** tested.

### 18.3 Where the sources disagree, or where one source corrects another

* **Backend label.** Section 8 (as first written) and section 11 (spike S, branch `m0-solver`) call the default backend Cranelift. Section 8.3,
  later, reads `cancho --help` of `a4572ea` and says the default is LLVM; the rows labelled "Cranelift (default)" in sections 8.2 and 11 were the
  default build, which is why they equal the `--backend llvm` rows. `--backend cranelift` is a different build, 1.7x slower on the original
  arm A (8.3).
* **Section 8.2 against 8.3.** 28x/32x and 4.0x/4.5x (session at load 6.5 to 8) are superseded by 6.3x/7.3x and 0.89x/1.0x (load 2.8 to 3.4) after
  tuning; the same code under different load gave 2.958 s and 2.794 s for the original, which is why 8.3 quotes ratios only from one interleaved batch.
* **Newton at 1,000,000 rows.** Spike S's Newton fit took 0.359 s (section 11.2, load 3.2); the Q-det arm A, whose 21 coefficients equal spike S's
  bit for bit (15.7, G-A), took 468 ms for the same 6 iterations (load 6.3 to 8.4). The two were not measured together, the loads differ, and
  neither source explains the difference.
* **Numbering of sections.** The three result branches each used 10 and 11 for their own spike; see the provenance table above.
* **Spike F's gate names a script, `scripts/spikes/mutants_f.py`** (13.3), that was never committed; the six mutants are run by
  `scripts/spikes/check_f.py --mutants` (14.2).
* **Spike S's tolerance.** The gate of 10.2 was edited on 2026-10-12, after its first run failed at 1,000 rows (|dAUC| = 1.00e-4 against 1e-5);
  the AUC tolerance became `max(1e-5, 1/(n_pos n_neg))`. The edit is kept in 10.2 and the log-loss tolerance was not touched.

### 18.4 What the record does not settle

Only the Mac; one thread for the speed ratios; no quiet-machine run except the one rerun of 8.3 (1-minute load under 2 is not an idle machine:
five users were logged in); no `gram`; no `predict` for L or S; no threaded timing of arm A; arms B and C only on the Newton step of logistic
regression (one loop, one data set, one compiler); the forest has no thread test (14.7); the forest timing compares different algorithms and
no same-algorithm numpy timing of it exists. The cost of the lane order in scalar cancho was not measured by any of these (`docs/lanes.md`
section 6).

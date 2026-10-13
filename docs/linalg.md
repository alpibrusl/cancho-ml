# cancho-ml linear algebra (`lib/linalg`, stage 1 of v1)

**Status: the gate (section 1) and the specification (sections 2 to 4) were written before any kernel. Section 5 holds the
measurements; a number is in it only if a script in `scripts/` printed it, with its conditions.** Tags as in `design.md`:
**[choice]** is a decision the maintainer made or can overturn, **[estimate]** is not measured.

## 1. The gate (fixed before any kernel code)

The target, from the maintainer: the linear algebra must **match or improve** current libraries.

### 1.1 Reference and protocol

* **Reference** = numpy and scipy as installed on the machine under test, called from Python, **one thread**:
  `VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` in the environment of the process. On this Mac the
  BLAS is Accelerate (numpy reports `blas: accelerate`). Whether Accelerate honours the variable is **checked, not assumed**:
  `scripts/bench_linalg.py` prints `cpu seconds / wall seconds` of a 1024 gemm run in that environment (1.0 means one
  thread); the figure goes next to every table in section 5.
* **Same machine**, same moment: cancho and numpy samples are **interleaved** (one numpy sample, one cancho process, in turn).
* **Statistic**: the **minimum of 7** samples per cell; the **spread** (max over min) is printed; a cell with spread > 1.25 is
  rerun, and if it stays above, it is reported with the spread marked. The load average (1 minute) is read at the start and at
  the end of each cell. **A cell with load >= 2 at either end is flagged**; a flagged cell is not a quiet-machine number.
* A cancho sample times the kernel only: data generation, allocation and result hashing are outside the timed region
  (`clock_ms`, 1 ms resolution, so a sample runs the kernel `R` times and reports the mean; `R` is printed). A numpy sample
  is the same `R` calls (`time.perf_counter`), with preallocated outputs where the API allows it (`out=`, `overwrite_*`), so
  that allocation is not charged to either side where it can be avoided. Where it cannot (`np.dot` of vectors returns a
  scalar; `scipy.linalg.cholesky` allocates), the allocation stays in numpy's time and is said so.
* The compiler is the pinned one (`a4572ea`), default LLVM backend. Everything is `float` (binary64), no `fma`, no SIMD
  (cancho has neither, `design.md` section 3).

### 1.2 Kernels and shapes

| kernel | numpy / scipy call | shapes | role |
|---|---|---|---|
| `dot` | `np.dot(x, y)` | n = 1e6 | gate |
| `axpy` | `scipy.linalg.blas.daxpy(x, y, a=a)` (one pass, writes y in place: checked, the returned array is y; `y += a * x` is two passes in numpy) | n = 1e6 | gate |
| `gemv_n` y = X w | `np.dot(X, w, out=y)` | X is n x d, n in {1e5, 1e6}, d in {20, 64, 256} | gate |
| `gemv_t` g = X^T r | `np.dot(r, X, out=g)` | same | gate |
| `syrk` C = X^T X (one triangle) | `scipy.linalg.blas.dsyrk(1.0, X, trans=1)` (BLAS `dsyrk`: one triangle only) | same | gate |
| `cholesky` | `scipy.linalg.cholesky(A, lower=True, check_finite=False)` | d in {21, 64, 256} | gate |
| `trsv` forward, backward | `scipy.linalg.solve_triangular(L, b, lower=True, trans=0 / 1, check_finite=False)` | d in {21, 64, 256} | gate |
| `gemm` | `np.dot(A, B, out=C)` | square 256, 512, 1024 | **reference only** |

n = 1e6 with d = 256 is a 2 GB matrix (8 bytes x 2.56e8). If a shape cannot be run (memory, time) its cell says "not run" and why;
it is not dropped.

### 1.3 What "match" and "improve" mean here

* **MATCH** for a cell = `cancho time / numpy time <= 1.0` by the protocol above, on the **tall-skinny** shapes (`dot`, `axpy`,
  `gemv_n`, `gemv_t`, `syrk`) and on Cholesky and `trsv` for d <= 256. **Target stated**: ratio <= 1.0 on every one of
  those cells at one thread. A cell above 1.0 is **NOT MATCH** with the factor; it is not rounded down and not hidden. If the
  fixed-order rule (section 2) is what makes a shape slower, section 5 says so and measures how much (an ablation with the
  order rule removed, outputs then wrong, time kept, as `spikes.md` section 8.3 does).
* **Square gemm is reference only.** It is expected not to match (no SIMD, no `fma`; Accelerate uses the AMX unit). The honest
  ratio is reported and no claim is made on it.
* **IMPROVE** = **byte-identical results at every thread count 1, 2, 4, 8, 16** (checked for every kernel that has a threaded
  variant: `dot`, `gemv_t`, `syrk`, `gemm`; the rest are single-threaded by construction and have nothing to vary) and,
  when a second machine is available, **across machines**. A BLAS does not give this (its blocking and its thread split depend
  on the CPU and the thread count). The gate **must be able to fail**: a mutant that adds the partial sums **in thread order**
  must fail the check at more than one thread (section 4); a mutant that survives voids the gate.
* **Correctness** comes first. Every kernel has an **order-matched oracle** in numpy (the same additions in the same order,
  `scripts/oracle_linalg.py`): the cancho result and the oracle are compared **bit for bit**, not by a tolerance, wherever
  the order is identical. A comparison to the library's own result (BLAS order) is reported as a distance in ulps, never as
  a pass.

### 1.4 What the gate does not say

It does not say cancho is as fast as numpy on shapes outside the table, on other machines, or with several threads. It does not
compare to a vendor BLAS with its threads on. A verdict is for the cells measured, on the machine and load printed.

## 2. The normative summation order **[choice; revised at `v1-lanes`: four lanes inside a block]**

Every reduction has an order that is part of the API and does not depend on the thread count. **The order changed once, at the
branch `v1-lanes`:** until then a block was one ascending chain; now a block is four interleaved chains (the lane rule below). The
text of this section is the single definition; `lib/linalg/order.cho` holds its constants and `lib/metrics/sums.cho` no longer has
a second one (before: `order.cho` one chain per block, `sums.lanes()` = 4 four lanes; `docs/metrics.md` 2.4 already had this rule).
The evidence for the lane count is `cancho-ml-simd/docs/simd-ceiling.md` section 11 (a C ceiling experiment with NEON intrinsics,
bit-identical to a scalar reference of this very rule for L in {1, 2, 4, 8, 16}; L = 4 recommended: worst cell 1.20 of the best L,
geometric mean 1.07 over 22 cells), **not** a measurement of cancho. What it buys is stated there: one definition that a scalar
cancho, a future vector cancho and C can all follow bit for bit. What it does not buy: speed of the scalar code (not measured).

* **Block rule.** A reduction over `n` terms `t_0 .. t_{n-1}` is cut into **blocks of `K` terms** (block `b` holds
  `t_{bK} .. t_{min(n,(b+1)K)-1}`). `K = 256` (`order.block()`), a constant of the package, never a function of the thread count
  or of `n`.
* **Lane rule (inside a block).** `L = 4` (`order.lanes()`). A block with terms `u_0 .. u_{m-1}` (`m <= K`, numbered from the
  start of the block) has four accumulators `s_0 .. s_3`, each **starting at `+0.0`**; **term `u_i` is added to lane `i mod 4`**,
  each lane in ascending `i`: `s_l <- s_l + u_i`. `K` is a multiple of `L`, so the lane of a term is also its global index mod 4.
  A lane with no term holds `+0.0`. **The lane combine** (`order.lane_combine`, fixed, written down): the block partial is
  `(s_0 + s_1) + (s_2 + s_3)`, the left operand first in each addition, the level-wise tree of the four lanes. A block of `m <= 3`
  terms still takes this combine (`m = 1`: `(u_0 + 0.0) + (0.0 + 0.0)`), which only matters for the sign of a zero.
* **Tree rule (across blocks, unchanged).** The block partials `p_0 .. p_{m-1}` are combined by a **fixed binary tree by block
  index**: level by level, `q_i = p_{2i} + p_{2i+1}` (left operand first), and when the count is odd the last partial is carried
  up unchanged. `m = 1` gives `p_0` with no further addition; `n = 0` gives `0.0`.
* **One swappable function.** The combine of two partials is `order.combine` (a scalar add); the lane combine and the whole
  combining step of every kernel go through `lib/linalg/order.cho`. The exact-accumulator alternative replaces that file's
  combine and partial type; no kernel is edited.
* **Streaming form, same tree.** The tree is evaluated by a stack of at most `log2(m) + 1` partials (push each block at level 0,
  merge while the two top entries have equal level, finish by merging right to left). `scripts/oracle_linalg.py --tree` checks
  that it equals the level-wise definition above for every `m` up to 4096. A **chunk** is `G = 8` consecutive blocks starting
  at a multiple of `G`, which is an aligned subtree; a thread is given whole chunks, returns each chunk's root, and the parent
  continues the same stack with those roots. That is why the thread count cannot change a bit (`G` is a package constant too).
* **Products are not fused**: a term is `fl(a*b)`, then added (`floating-point.md`: no contraction). The oracle multiplies and
  adds separately in numpy as well.
* **What is a function of what.** The order is a function of the term index (block = `i div 256`, lane = `i mod 4`) and of the
  constants `K`, `L`, `G` and (Cholesky) `NB`; not of the thread count, the vector width, the strides or the fast path taken.

### 2.1 Per kernel

| kernel | term and reduction axis | order |
|---|---|---|
| `dot(x, y)` | `x_i * y_i`, i = 0..n-1 | block rule, lane rule inside each block, tree rule |
| `axpy` | none (`y_i = y_i + a * x_i`) | elementwise; no reduction; one rounding for the product, one for the add; **unchanged** |
| `gemv_n` (y = X w) | for row i: `X_ij * w_j`, j = 0..d-1 | per row exactly `dot` of the row and `w` (one block when d <= K: lane = j mod 4) |
| `gemv_t` (g = X^T r) | for column j: `X_ij * r_i`, i = 0..n-1 | block rule over rows i, lane = (i - block start) mod 4 = i mod 4, tree rule, independently per column |
| `syrk` (C = X^T X, entries j <= k) | `X_ij * X_ik`, i = 0..n-1 | as `gemv_t`, per entry; the other triangle is not written |
| `gemm` (C = A B) | `A_ik * B_kj`, k = 0..p-1 | per entry exactly `dot` of a row of A and a column of B. Follows from the definition; the SIMD experiment did not run `gemm` |
| `cholesky` (A = L L^T, lower) | `L_ik * L_jk`, k = 0..j-1 | **panels of `NB = 32` columns** (`order.panel()`), aligned at multiples of 32. For the entry `(i, j)`, `i >= j`: `s = A_ij`; for every full panel `p = 0 .. c-1` (`c = j div 32`, so every `k` of it is `< j`) `s = s - P_p` where `P_p` is the **lane rule** (4 lanes, lane = `(k - 32p) mod 4`, lane combine as above) over `L_ik * L_jk` for `k` in the panel; then once more `s = s - P_c` with `P_c` the lane rule over `k = 32c .. j-1` (empty = `+0.0`). Then `L_ij = s / L_jj`, or for `i = j`, `L_ii = sqrt(s)` with the pivot test `s > 0.0`, else refusal. This is `ref_chol` of `cancho-ml-simd/scripts/simd/simd.c`, the order a blocked right-looking algorithm computes. **It is not a recurrence over single terms any more: `NB` and `L` are part of the order** |
| `trsv` forward (L x = b) | `L_ik * x_k`, k = 0..i-1 | `s = b_i; s = s - L_ik * x_k` for k ascending; `x_i = s / L_ii`; **unchanged** (a recurrence; not in the SIMD experiment, so no evidence for another order) |
| `trsv` backward (L^T x = b) | `L_ki * x_k`, k = d-1 down to i+1 | `s = b_i; s = s - L_ki * x_k` for k **descending** from d-1; `x_i = s / L_ii`; i from d-1 down; **unchanged** *(corrected in place earlier: the first draft said ascending k; descending lets four columns be summed together)* |

**What the evidence supports, and what was not adopted.** `simd-ceiling.md` shows (its section 4) the lane rule for `dot`, `gemv_n`,
`gemv_t` and `syrk` and the panel rule for Cholesky each bit-identical to a scalar reference, on 11 to 17 shapes per kernel, with
five runtime mutants of the reference (lane count, lane combine order, element-to-lane map, block size, panel width) all detected
(162 killed, 22 equivalent and explained). Its section 11 recommends `L = 4`, `K = 256` and Cholesky panels of 32; it records that
the Cholesky order differs from the one-running-value order (its caveat 5). **Adopted:** exactly that, with `NB = 32` as the
experiment fixed it. **Not adopted, because no evidence exists:** a different `NB` (only 16 was tried, as a mutant that must differ),
a different order for `trsv` (outside the experiment), any change to `axpy`. No performance claim is made here: the experiment's
speed figures are for C with intrinsics, one session, loaded; the scalar cancho kernels of this branch were not benchmarked.

**The decisive sums of the models** (`docs/models-linear.md` 3) follow the block and lane rule over the rows of a block: the logistic
gradient, Hessian and loss, the k-means cluster sums, counts and inertia, and the linear-regression `G` and `b` (through `syrk` and
`gemv_t`). **Not under the rule, as before and for the reason given there:** a row's score `z`, a row-to-centroid distance, the
k-means++ total (left folds, a row is never split among threads); the regression sums of the trees (sequential, one thread per
tree, `models-tree.md` 3.2); the exact sums of the command line (`ml`: integer limbs and `facc`, order-free).

## 3. API **[choice]**

* **Descriptor.** `linalg.tensor.Tensor { off, rank, s0, s1, s2, s3, t0, t1, t2, t3 }`, all `int`, `R_MAX = 4`; unused shape
  entries are 1 and unused strides are 0. The elements are a separate `[float]` passed beside it. A matrix is `rank = 2`,
  row-major strides `(cols, 1)`.
* **Operations** (descriptor edits, no copy): `matrix`, `vector`, `slice` (one axis), `transpose`, `reshape` (only when the strides
  are contiguous), `broadcast` (stride 0), `row`, `column`. Each answers a `Made` (`Ok(Tensor)` or `Err(code)`).
* **Refusals** (rule tags, exit 8): `ml.tensor.rank` (rank 0 or > 4), `ml.tensor.shape` (a negative extent, a reshape
  that does not preserve the element count or is not contiguous, an element count that overflows `int`, a broadcast of an axis
  that is not 1), `ml.tensor.bounds` (the descriptor addresses an element outside the data it is given, or a slice lies outside
  its axis). `tensor.check(t, len)` is the single check every kernel entry calls first; no kernel indexes with an unchecked
  descriptor.
* **Kernels** take the descriptor(s) and the data beside. Contiguous operands (stride 1 on the reduction axis, row stride = cols)
  take the fast path; any other stride takes the generic path; **both have the same bits** (a test).
* **Threaded variants** (`*_par`) take a thread count `t >= 1`; `t = 1` is the sequential kernel.

## 4. Tests and mutants (all gates in `tests/` and `scripts/`; `scripts/check.sh` runs the fast ones)

1. `cancho test` unit tests (`tests/t_*.cho`, `tests/core.cho`): each kernel against the definition of section 2 written separately
   in `tests/lref.cho`, on random and adversarial inputs (cancellation, mixed magnitudes, subnormals, signed zeros), row- and
   column-major storage, strided vectors; the refusals of every entry.
2. `scripts/oracle_linalg.py`: the same inputs (SplitMix64 generator, `docs/rng.md`) through the compiled `linalg-check` binary
   and through numpy emulating the section 2 order; bit comparison (`--suite`, `--thread-suite --threads ...`, `--tree`).
3. `scripts/mutants_core.py`, `scripts/mutants_linalg.py`: each mutant is a source edit to a copy of the tree; the gate of its group
   must fail on it (and the unmutated control must pass first). Includes the thread-order mutant. Survivors are listed with a
   reason in the scripts and in section 5.1.
4. Fuzz (`tests/t_fuzz.cho`): random descriptors with out-of-range ints, negative strides, overflowing products, un-normalised
   axes, through every kernel entry; no call may trap. The threaded variants are a program (`tests/t_par.cho`, `scripts/gate_par.sh`)
   because `cancho test` cannot run a test that spawns (section 5.6, U4).
5. `scripts/c_parity/parity.c`: the same algorithms in C; its hashes must equal the driver's.

## 5. Results

**Every figure and every bit hash in sections 5 and 6 below was produced with the previous order (one ascending chain per block, a one-running-value Cholesky), before `v1-lanes` (section 2). None was re-measured on the lane order: no benchmark was run on that branch (`docs/lanes.md`). The correctness statements of 5.1 stand for the order they name; the bits of today are checked by the gates listed in `docs/lanes.md`.**

Everything here was printed by a script of this repository; the raw lines are in `docs/bench/`. **Conditions.** Apple M4 Max (16
cores), macOS 26.2, cancho `a4572ea` (default LLVM backend), numpy 2.4.6, scipy 1.17.1, Python 3.14, Accelerate BLAS, Apple clang
21.0.0 for the C column. **The machine is shared (5 users logged in) and was never quiet: the load average (1 minute) was between
3.8 and 10 during every table below, so every cell is flagged `LOAD>=2` and none is a quiet-machine number.** The numpy
side ran with `VECLIB_MAXIMUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1`; **the variable is honoured**: `cpu seconds / wall seconds`
of a 1024 gemm is 0.999 (`docs/bench/gate-1thread.load.txt`), and 1.00 for the gemv, dsyrk, dot and daxpy calls of the gate
(measured in the session that wrote this section). A cancho sample is a separate process of `build/linalg-check` timing `R`
runs with `clock_ms`; a numpy sample is `R'` calls in one Python process. Minimum of 7 interleaved samples; spread = max / min;
a cell with spread > 1.25 is rerun up to twice and the best attempt is kept (three cells still exceed 1.25 and are marked).

### 5.1 Correctness: bits, not tolerances

| gate | what it checks | result |
|---|---|---|
| unit tests (`scripts/check.sh`) | 53 test functions in 8 files: every kernel against the section 2 definition written separately in `tests/lref.cho` (generic strides, block partials stored then reduced level by level, not the stack), on plain, wide-magnitude, signed-zero and subnormal data, ragged and aligned block counts, row- and column-major storage, strided vectors | all pass |
| numpy oracle (`scripts/oracle_linalg.py --suite`) | 46 cases (0 to 1e6 rows, d on both sides of 256) through the compiled driver against numpy performing the same additions in the same order (never numpy's BLAS) | all bits equal |
| tree (`--tree`) | the stack of `order.cho` equals the level-wise tree for every block count 1..4096 | equal |
| third implementation (`scripts/c_parity/parity.c`) | the same algorithms in C, `-ffp-contract=off` | same hash as cancho on every kernel it implements (dot, axpy, gemv_n, gemv_t, syrk, chol) |
| fuzz (`tests/t_fuzz.cho`: 8 x 20,000 random descriptors per kernel, extremes included) | no kernel entry traps; a refusal is a code of `ml.refuse` | passes; **found two holes** that are fixed with tests and mutants: an unused axis holding extent 0 hid a huge used one from `tensor.check`, and `axpy` sliced with the offset of an empty operand |
| mutants (`scripts/mutants_core.py`, `scripts/mutants_linalg.py`) | 136 source mutants (an operation, a bound, a block length, an index, a tree rule, a thread-order combine) | **132 killed, 4 survive and are equivalent** (explained in the scripts: `fbits-load-check`, `limits-mul`, `order-blocks`, `order-level-eq`), 0 unexplained; `docs/bench/mutants_*.log` |

**Not done:** a second machine. The brief allows `gram` only with ssh, which this task may not use, so "the same bytes across
machines" is **not tested**; what exists is agreement of cancho, an independent numpy emulation and a C build on this one machine.

### 5.2 The gate, one thread (`scripts/bench_linalg.py --gate --c`, `docs/bench/gate-1thread.jsonl`)

| kernel | shape | cancho | numpy / scipy | cancho / numpy | verdict | C (same algorithm) | cancho / C | spread c / n | load start / end | R c / n |
|---|---|---|---|---|---|---|---|---|---|---|
| dot | n=1,000,000 | 278.00 us | 76.75 us | 3.62 | NOT MATCH (3.62x) LOAD>=2 | 257.80 us | 1.08 | 1.07 / 1.20 | 8.3 / 8.5 | 500 / 674 |
| axpy | n=1,000,000 | 187.10 us | 97.59 us | 1.92 | NOT MATCH (1.92x) LOAD>=2 | 191.06 us | 0.98 | 1.08 / 1.12 | 8.5 / 8.5 | 775 / 456 |
| gemv_n | n=100,000 d=20 | 313.73 us | 295.16 us | 1.06 | NOT MATCH (1.06x) LOAD>=2 | 311.32 us | 1.01 | 1.03 / 1.11 | 8.5 / 8.5 | 459 / 184 |
| gemv_t | n=100,000 d=20 | 296.52 us | 221.57 us | 1.34 | NOT MATCH (1.34x) LOAD>=2 | 264.56 us | 1.12 | 1.07 / 1.13 | 8.5 / 8.5 | 489 / 150 |
| syrk | n=100,000 d=20 | 1.880 ms | 547.88 us | 3.43 | NOT MATCH (3.43x) LOAD>=2 | 1.939 ms | 0.97 | 1.10 / 1.04 | 8.5 / 8.7 | 50 / 89 |
| gemv_n | n=100,000 d=64 | 1.067 ms | 556.15 us | 1.92 | NOT MATCH (1.92x) LOAD>=2 | 1.121 ms | 0.95 | 1.23 / 1.25 | 8.7 / 9.1 | 120 / 95 |
| gemv_t | n=100,000 d=64 | 900.00 us | 494.48 us | 1.82 | NOT MATCH (1.82x) spread>1.25 LOAD>=2 | 896.21 us | 1.00 | 1.27 / 1.05 | 9.2 / 9.3 | 150 / 130 |
| syrk | n=100,000 d=64 | 17.375 ms | 1.721 ms | 10.10 | NOT MATCH (10.10x) LOAD>=2 | 17.406 ms | 1.00 | 1.03 / 1.08 | 9.3 / 9.3 | 8 / 27 |
| gemv_n | n=100,000 d=256 | 5.400 ms | 2.350 ms | 2.30 | NOT MATCH (2.30x) LOAD>=2 | 5.690 ms | 0.95 | 1.19 / 1.09 | 9.3 / 9.4 | 20 / 21 |
| gemv_t | n=100,000 d=256 | 4.520 ms | 1.684 ms | 2.68 | NOT MATCH (2.68x) LOAD>=2 | 4.454 ms | 1.01 | 1.14 / 1.24 | 9.4 / 9.4 | 25 / 25 |
| syrk | n=100,000 d=256 | 290.000 ms | 19.515 ms | 14.86 | NOT MATCH (14.86x) LOAD>=2 | 301.970 ms | 0.96 | 1.06 / 1.05 | 9.4 / 9.3 | 1 / 2 |
| gemv_n | n=1,000,000 d=20 | 3.643 ms | 3.485 ms | 1.05 | NOT MATCH (1.05x) LOAD>=2 | 3.130 ms | 1.16 | 1.07 / 1.17 | 9.5 / 9.7 | 42 / 9 |
| gemv_t | n=1,000,000 d=20 | 3.143 ms | 2.382 ms | 1.32 | NOT MATCH (1.32x) LOAD>=2 | 2.947 ms | 1.07 | 1.23 / 1.08 | 9.7 / 9.8 | 42 / 21 |
| syrk | n=1,000,000 d=20 | 22.333 ms | 6.114 ms | 3.65 | NOT MATCH (3.65x) LOAD>=2 | 23.035 ms | 0.97 | 1.10 / 1.09 | 9.8 / 10.0 | 6 / 7 |
| gemv_n | n=1,000,000 d=64 | 13.000 ms | 6.022 ms | 2.16 | NOT MATCH (2.16x) LOAD>=2 | 12.683 ms | 1.03 | 1.08 / 1.08 | 10.0 / 9.9 | 10 / 8 |
| gemv_t | n=1,000,000 d=64 | 11.667 ms | 4.750 ms | 2.46 | NOT MATCH (2.46x) LOAD>=2 | 11.401 ms | 1.02 | 1.14 / 1.20 | 9.9 / 9.5 | 12 / 10 |
| syrk | n=1,000,000 d=64 | 180.000 ms | 16.902 ms | 10.65 | NOT MATCH (10.65x) LOAD>=2 | 183.263 ms | 0.98 | 1.03 / 1.10 | 9.5 / 9.1 | 1 / 2 |
| gemv_n | n=1,000,000 d=256 | 131.000 ms | 23.562 ms | 5.56 | NOT MATCH (5.56x) spread>1.25 LOAD>=2 | 134.234 ms | 0.98 | 1.50 / 1.10 | 8.7 / 8.8 | 1 / 2 |
| gemv_t | n=1,000,000 d=256 | 95.000 ms | 14.417 ms | 6.59 | NOT MATCH (6.59x) spread>1.25 LOAD>=2 | 97.397 ms | 0.98 | 1.21 / 1.53 | 8.6 / 8.6 | 1 / 2 |
| syrk | n=1,000,000 d=256 | 2.734 s | 189.494 ms | 14.43 | NOT MATCH (14.43x) LOAD>=2 | 2.860 s | 0.96 | 1.02 / 1.03 | 7.9 / 5.3 | 1 / 1 |
| chol | d=21 | 0.58 us | 2.55 us | 0.23 | MATCH LOAD>=2 | 0.38 us | 1.53 | 1.03 / 1.05 | 5.3 / 5.3 | 259615 / 1574 |
| fwd | d=21 | 0.13 us | 3.68 us | 0.04 | MATCH LOAD>=2 | | | 1.03 / 1.02 | 5.3 / 5.3 | 1687500 / 1534 |
| bwd | d=21 | 0.14 us | 3.56 us | 0.04 | MATCH LOAD>=2 | | | 1.05 / 1.04 | 5.3 / 5.3 | 1125000 / 4838 |
| chol | d=64 | 8.76 us | 8.93 us | 0.98 | MATCH LOAD>=2 | 7.52 us | 1.16 | 1.02 / 1.02 | 5.3 / 5.3 | 22500 / 3529 |
| fwd | d=64 | 0.45 us | 5.10 us | 0.09 | MATCH LOAD>=2 | | | 1.04 / 1.04 | 5.3 / 5.3 | 337500 / 4152 |
| bwd | d=64 | 0.64 us | 5.22 us | 0.12 | MATCH LOAD>=2 | | | 1.02 / 1.03 | 5.3 / 5.3 | 225000 / 4152 |
| chol | d=256 | 490.26 us | 151.82 us | 3.23 | NOT MATCH (3.23x) LOAD>=2 | 483.09 us | 1.01 | 1.01 / 1.04 | 5.3 / 5.4 | 308 / 290 |
| fwd | d=256 | 5.91 us | 16.45 us | 0.36 | MATCH LOAD>=2 | | | 1.02 / 1.02 | 5.4 / 5.4 | 22500 / 2108 |
| bwd | d=256 | 8.84 us | 16.97 us | 0.52 | MATCH LOAD>=2 | | | 1.06 / 1.03 | 5.4 / 5.4 | 22500 / 2272 |
| gemm | 256^3 | 1.320 ms | 80.90 us | 16.32 | reference only LOAD>=2 | | | 1.01 / 1.03 | 5.4 / 5.3 | 150 / 606 |
| gemm | 512^3 | 10.846 ms | 656.16 us | 16.53 | reference only LOAD>=2 | | | 1.02 / 1.01 | 5.3 / 5.3 | 13 / 76 |
| gemm | 1024^3 | 90.000 ms | 6.528 ms | 13.79 | reference only LOAD>=2 | | | 1.14 / 1.11 | 5.3 / 5.3 | 1 / 7 |

**Verdict.** Against the calls a user writes (`np.dot`, `scipy.linalg.cholesky`, `solve_triangular`, ...): **8 of the 29 gate cells
MATCH and 21 are NOT MATCH**. MATCH: Cholesky d = 21 and d = 64, forward and backward solves at d = 21, 64, 256. NOT MATCH: dot
(3.62x), axpy (1.92x), gemv_n (1.05x to 5.56x), gemv_t (1.32x to 6.59x), syrk (3.43x to 14.86x), Cholesky d = 256 (3.23x). Square
gemm (reference only, no claim): 16.3x, 16.5x, 13.8x slower than Accelerate at 256, 512, 1024. Two cells are within 6% of 1.0 and
should not be read as near-misses either way: gemv_n at d = 20 (1.06x and 1.05x). **numpy's time on that cell moved by 2.4x
between sessions with cancho's unchanged:** 119 us seen once during development (load 11.5, not in a saved table) against 230
to 300 us in five other runs (loads 4 to 9), `cpu/wall` 1.00 throughout; the cause was not found, so the ratio on that cell is
1.06 in the table and would have been 2.6 against the 119 us figure.

**Against the thinnest scipy call** (the f2py wrappers `scipy.linalg.lapack.dpotrf` / `dtrtrs` on Fortran-ordered input, no
validation, no transposing copy; second session, load 3.8 to 4.0, `docs/bench/gate-lapack.jsonl`): the solves still MATCH
(0.36 to 0.75 of the time of `dtrtrs`) and **Cholesky does not at any d**: 1.09x at d = 21, 1.64x at d = 64, 10.67x at d = 256
(`scipy.linalg.cholesky` costs 2.5, 8.9 and 150 us where `dpotrf` costs 0.5, 5.2 and 47 us in a separate microbenchmark; the
difference is the wrapper's own work, not itemised here, and it is what makes d = 21 and d = 64 MATCH above). Strictest reading: **6 of 29 cells MATCH (the solves
only)**.

| kernel | shape | cancho | numpy / scipy | cancho / numpy | verdict | C (same algorithm) | cancho / C | raw LAPACK | cancho / raw | verdict vs raw | spread c / n | load start / end | R c / n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chol | d=21 | 0.57 us | 2.46 us | 0.23 | MATCH LOAD>=2 | 0.38 us | 1.50 | 0.52 us | 1.09 | NOT MATCH (1.09x) | 1.08 / 1.06 | 3.8 / 3.8 | 241071 / 2850 |
| chol | d=64 | 8.80 us | 8.91 us | 0.99 | MATCH LOAD>=2 | 7.46 us | 1.18 | 5.37 us | 1.64 | NOT MATCH (1.64x) | 1.02 / 1.04 | 3.8 / 3.8 | 11250 / 2696 |
| chol | d=256 | 496.67 us | 151.67 us | 3.27 | NOT MATCH (3.27x) LOAD>=2 | 484.43 us | 1.03 | 46.55 us | 10.67 | NOT MATCH (10.67x) | 1.01 / 1.04 | 3.8 / 3.8 | 300 / 322 |
| fwd | d=21 | 0.13 us | 3.62 us | 0.04 | MATCH LOAD>=2 | | | 0.35 us | 0.36 | MATCH | 1.03 / 1.03 | 4.0 / 4.0 | 1125000 / 2307 |
| fwd | d=64 | 0.45 us | 5.13 us | 0.09 | MATCH LOAD>=2 | | | 1.13 us | 0.40 | MATCH | 1.03 / 1.03 | 4.0 / 4.0 | 337500 / 3157 |
| fwd | d=256 | 5.87 us | 16.65 us | 0.35 | MATCH LOAD>=2 | | | 10.99 us | 0.53 | MATCH | 1.02 / 1.02 | 4.0 / 4.0 | 22500 / 831 |
| bwd | d=21 | 0.14 us | 3.53 us | 0.04 | MATCH LOAD>=2 | | | 0.38 us | 0.36 | MATCH | 1.03 / 1.04 | 4.0 / 4.0 | 1125000 / 4724 |
| bwd | d=64 | 0.64 us | 5.15 us | 0.12 | MATCH LOAD>=2 | | | 1.27 us | 0.50 | MATCH | 1.02 / 1.05 | 4.0 / 3.9 | 225000 / 2474 |
| bwd | d=256 | 8.80 us | 17.15 us | 0.51 | MATCH LOAD>=2 | | | 11.69 us | 0.75 | MATCH | 1.05 / 1.02 | 3.9 / 3.9 | 11250 / 712 |

### 5.3 Why the misses are misses (profiled by ablation, not guessed)

1. **cancho is at the level of plain C for the same algorithm.** `cancho / C` (clang -O3, the same blocks, the same order, no fma)
   is 0.95 to 1.16 for dot, axpy, gemv_n, gemv_t, syrk and Cholesky at d = 256 (column `cancho / C` above; 1.5 and 1.18 at
   Cholesky d = 21 and 64 where a call is under 10 us). The same C is 0.9x to 6.8x the time of numpy on the dot, axpy and gemv cells (0.9x on one cell, gemv_n 1e5 x 20)
   and 3.5x to 15.5x on syrk. So the gap is the vendor library, not cancho's code generation, and it cannot be closed in this language
   without something the language does not have (5.6). *Not diagnosed:* what the vendor kernels do (hand-written NEON, the AMX
   unit, prefetching): nothing here reads them.
2. **The summation order costs the dot product, and only it.** `scripts/ablate/dot.sh`: n = 1e6, the block rule 200 to 250 us,
   the same scalar code over 8 adjacent lanes (another order) 140 us, in cancho; C with NEON intrinsics over 8 lanes 156 us
   (`scripts/c_parity/dot.sh`), numpy 69 to 77 us. The rule costs **1.7x** on the dot product (scalar adds, because the blocks are
   2 KB apart and cannot share a vector register); even with a free order cancho would be 1.8x to 2.0x from numpy. gemv_t, syrk
   and gemm vectorise across independent columns or entries and pay nothing for the rule; in gemv_n the row sum is one chain
   either way. (An alternative order that is SIMD-friendly, lanes inside the block, is a decision for the maintainer: it would
   change every dot-shaped reduction; see the open questions.)
3. **fma is not the reason.** `scripts/c_parity/fma.sh`: the same C allowed to fuse runs between 0.85x and 1.05x of the time
   without (syrk 0.91 and 0.95, gemv_n 0.86, gemv_t 1.05, Cholesky 1.04, dot 0.96).
4. **The large-d gemv cells are about streaming.** gemv_n at n = 1e6: d = 64 moves 512 MB in 13.0 ms (39 GB/s) against numpy's
   85 GB/s; d = 256, 2.05 GB in 131 ms (15.6 GB/s) against 87 GB/s; plain C is the same (134 ms). That is a one-core scalar
   loop's limit here; rows in flight matter: **8 rows were 2.2x slower than 4 at d = 256** (11.1 against 5.1 ms for n = 1e5;
   `scripts/ablate/gemv_rows.sh`), because eight row streams 2 KB apart alias; the kernel uses 4 rows.
5. **syrk is compute-bound and 10x to 15x behind.** 4 x 4 register tiles reach plain C (`cancho / C` 0.96 to 1.00). A found
   cancho-side cost: the per-iteration `xs[i * rs + j0 .. ]` multiplies were overflow-checked (a `mul`/`smulh` pair per slice),
   1.34x behind C at n = 1e5, d = 20 (2.52 against 1.89 ms); carrying the offset with `wrapping_add` took it to 0.97 (commit
   `523a144`). Gemm went from 39 to 32 ms at 512 from the same change (one measurement, three repetitions).
6. **Cholesky d = 256 is 3.2x** (and 10.7x against raw `dpotrf`): plain C is the same time (`cancho / C` 1.02).
7. **The solves MATCH** and the reason is partly the reference: forward at d = 256 is 5.9 us in cancho (4 rows in flight) against
   16.6 us for `solve_triangular` and 11.0 us for `dtrtrs`. The order matters here: **backward substitution with k ascending
   (the first draft of the spec) is one serial chain: 25.1 us at d = 256, against 8.8 us for the descending order that is
   implemented** (`scripts/ablate/solve.sh`); ascending would have been NOT MATCH against the raw wrapper (11.7 us).

**What blocks matching, specifically.** (a) No vector type: the 2-lane code LLVM finds by itself (`fmul.2d` on adjacent
products, SLP over a 4 x 4 tile) is all there is, and a rule that adds blocks far apart cannot use it. (b) Whatever the vendor
kernels use beyond plain NEON (a plain NEON dot is still 2.1x slower than numpy's). (c) One scalar core at memory-streaming
rates. Fma and the summation order are *not* the dominant causes except the order for the dot product.

### 5.4 Determinism across thread counts: the "improve" gate

* **Byte-identical at 1, 2, 3, 4, 8 and 16 threads**: `scripts/gate_par.sh` (the program `tests/t_par.cho` and the driver against
  the numpy oracle on 15 shapes of `dot`, `gemv_t`, `syrk`, `gemv_n`, `gemm` with several chunks, ragged and aligned) passes;
  `docs/bench/threads.txt`: one distinct result hash over all 5 thread counts and 5 runs on every kernel timed.
* **The gate can fail.** `scripts/thread_mutant_demo.py` builds the mutant of `par-thread-order` (each thread keeps a running
  sum of its chunk roots, the parent adds the threads' sums in thread order, as a library reduction does) and runs the same
  cases at 1, 2, 4, 8, 16 threads:

```
real build
  dot     n=1000000  threads 1,2,4,8,16 -> dc9330f251fa9baa dc9330f251fa9baa dc9330f251fa9baa dc9330f251fa9baa dc9330f251fa9baa
  gemv_t  n=100000   d=64   ... -> d5506f96ddb1e687 (all five)
  syrk    n=30000    d=20   ... -> e89af4ef8b078d1d (all five)
mutant: partials added in thread order
  dot     -> dc9330f251fa9baa 88e14ecdd8cbb028 7c6b5e9ccbe5e164 3fee63a39a47aeb1 3fee63a39a47aeb1   (4 distinct)
  gemv_t  -> d5506f96ddb1e687 b8d33a7416d4da67 1d45bb9ea3cf5ece 0ceaaff9f8824e1e d29446e390d0ff5b   (5 distinct)
  syrk    -> e89af4ef8b078d1d a608bac174edb144 50f632d29a4c4f02 2f0beb194b41b608 1b5d681367db0129   (5 distinct)
```
  (the mutant agrees with the real build at one thread, which runs the sequential kernel). Eight further mutants of the threaded
  code (a misaligned chunk, no final merge, a wrong row partition, a wrong copy, ...) are all killed.
* **A BLAS does not give this, in part** (`scripts/blas_thread_variance.py`, Accelerate through numpy, 1/2/4/8/16 threads, one
  run): `gemv_t` 1e5 x 64 **differs** between 1 thread and 2 or more (two distinct results); `dot` 1e6, `gemv_n` 1e5 x 64,
  `dsyrk` 2e5 x 20 and `gemm` 512 were identical at every thread count on these inputs. So the claim "a BLAS does not give this"
  is shown for one of five operations here, not for all.
* **Speed of the threaded variants (honest numbers, `docs/bench/threads.txt`, load 4.2, min of 5):** `syrk` 100000 x 64 takes
  17.5 ms at 1 thread and 3.75 ms at 16 (4.7x); `gemm` 512: 10.8 to 2.25 ms (4.8x); `gemm` 1024: 89.5 to 40.0 ms (2.2x). **`dot`,
  `gemv_t` and `gemv_n` get slower** (0.38x to 0.65x at 2 to 16 threads) because every thread owns a copy of its input rows:
  cancho cannot hand a thread a shared read-only slice (5.6, U2), so a memory-bound kernel pays a copy that costs more than the
  kernel. The threads are for compute-bound kernels.

### 5.5 Other measured facts

* Tuning that the C comparison found: the gemv row count and the syrk/gemm index multiplications (5.3 items 4 and 5).
* `cancho build` in package mode was **not run** (it needs `cancho install`, a network fetch); `scripts/build.sh` passes the same
  file list. The `cancho.toml` pins are copied from cancho-table; `cancho-table` is **not** a dependency yet. *(Corrected in place: this said
  "its float reader is unmerged". It is merged, PR #32; the real blocker is that cancho-table commits no `.cancho-vcs` store, `docs/reader.md`
  section 1.)*
* `cancho fmt --check lib tests tools` is clean; no file is over 900 lines.

### 5.6 Upstream gaps, with reproducers (`docs/upstream/`)

| id | gap | reproducer | measured cost |
|---|---|---|---|
| U1 | no SIMD vector type or lane operations: the only SIMD is what LLVM finds; a reduction that must add far-apart blocks in order is scalar | `scripts/ablate/dot.sh`, `scripts/c_parity/dot.sh` | dot 1e6: 200 to 250 us for the block rule, 140 us for 8 adjacent lanes, 156 us for C with NEON intrinsics, 69 to 77 us numpy |
| U2 | a thread cannot be given a shared read-only slice (`thread-payloads.md` is a design); threaded kernels copy their input | `docs/upstream/spawn_shared_slice.cho` | dot / gemv at 2 to 16 threads 0.38x to 0.65x of one thread |
| U3 | `copy_into` takes bytes only; a float copy is a loop | `docs/upstream/copy_into_float.cho` | (the loop is the copy of U2) |
| U4 | `cancho test` generates `main` with the row `[]`, so a test that spawns is refused; two test files that both `import std.test` clash | `docs/upstream/test_runner_conc.cho` | `tests/t_par.cho` is a program, not tests |
| U5 | `res` and `val` are keywords, refused as variable, parameter and field names (found three times) | `docs/upstream/keywords.cho` | usability |
| U6 | the overflow check of `i * rs + c` inside a counted loop is not strength-reduced | `git show 523a144 -- lib/linalg/syrk.cho` | syrk 1e5 x 20: 1.34x of C before, 0.97 after |

Not a gap (measured): a 2-D slice. Row sub-slices (`xs[i * rs .. i * rs + d]`) discharge their checks and reach C parity.

### 5.7 Open questions for the maintainer

1. **The reduction order.** The block rule costs the dot product 1.7x (5.3 item 2). An order with lanes inside the block (say 8
   accumulators over adjacent elements, then the block tree) would be SIMD-friendly and equally thread-independent; it changes
   `dot`, `gemv_n` (rows) and the leaf of `gemm`. Not done, because the rule is the maintainer's.
2. **Which numpy call is "the reference"?** `scipy.linalg.cholesky` (what is written) or the raw wrapper (the library): the
   Cholesky verdict flips at d = 21 and d = 64 (5.2).
3. **Is MATCH reachable at all?** Not with the current language for dot, axpy, gemv, syrk (5.3); it is for the triangular solves.
   The honest options are to drop the tall-skinny MATCH target, to ask for U1, or to treat the vendor library as the fast path
   behind a flag that gives up D1/D2.
4. The exact-accumulator swap (`order.cho` is the one place) was **not exercised**: only the fixed tree exists, so "swappable" is
   a structure, not a tested claim.
5. Cross-machine byte identity is untested (5.1).

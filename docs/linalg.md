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
| `axpy` | `scipy.linalg.blas.daxpy(x, y, a=a, overwrite_y=1)` (one pass, in place; `y += a * x` is two passes in numpy) | n = 1e6 | gate |
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

## 2. The normative summation order **[choice, provisional]**

Every reduction has an order that is part of the API and does not depend on the thread count. The maintainer's provisional rule:

* **Block rule.** A reduction over `n` terms `t_0 .. t_{n-1}` is cut into **blocks of `K` terms** (block `b` holds
  `t_{bK} .. t_{min(n,(b+1)K)-1}`). Inside a block the sum is **ascending index starting from `+0.0`**:
  `((0.0 + t_a) + t_{a+1}) + ...`. `K = 256` (`order.block()`), a constant of the package, never a function of the thread
  count or of `n`.
* **Tree rule.** The block partials `p_0 .. p_{m-1}` are combined by a **fixed binary tree by block index**: level by level,
  `q_i = p_{2i} + p_{2i+1}` (left operand first), and when the count is odd the last partial is carried up unchanged. `m = 1`
  gives `p_0` with no further addition; `n = 0` gives `0.0`.
* **One swappable function.** The combine of two partials is `order.combine` (a scalar add); the whole combining step of every
  kernel goes through `lib/linalg/order.cho`. The exact-accumulator alternative (being measured elsewhere) replaces that file's
  combine and partial type; no kernel is edited.
* **Streaming form, same tree.** The tree is evaluated by a stack of at most `log2(m) + 1` partials (push each block at level 0,
  merge while the two top entries have equal level, finish by merging right to left). `scripts/oracle_linalg.py --tree` checks
  that it equals the level-wise definition above for every `m` up to 4096. A **chunk** is `G = 8` consecutive blocks starting
  at a multiple of `G`, which is an aligned subtree; a thread is given whole chunks, returns each chunk's root, and the parent
  continues the same stack with those roots. That is why the thread count cannot change a bit (`G` is a package constant too).
* **Products are not fused**: a term is `fl(a*b)`, then added (`floating-point.md`: no contraction). The oracle multiplies and
  adds separately in numpy as well.

### 2.1 Per kernel

| kernel | term and reduction axis | order |
|---|---|---|
| `dot(x, y)` | `x_i * y_i`, i = 0..n-1 | block rule, then tree rule |
| `axpy` | none (`y_i = y_i + a * x_i`) | elementwise; no reduction; one rounding for the product, one for the add |
| `gemv_n` (y = X w) | for row i: `X_ij * w_j`, j = 0..d-1 | per row exactly `dot` of the row and `w` (one block when d <= K) |
| `gemv_t` (g = X^T r) | for column j: `X_ij * r_i`, i = 0..n-1 | block rule over rows i, tree rule, independently per column |
| `syrk` (C = X^T X, entries j <= k) | `X_ij * X_ik`, i = 0..n-1 | as `gemv_t`, per entry; the other triangle is not written |
| `gemm` (C = A B) | `A_ik * B_kj`, k = 0..p-1 | per entry exactly `dot` of a row of A and a column of B |
| `cholesky` (A = L L^T, lower) | `L_ik * L_jk`, k = 0..j-1 | **one running value, ascending, no blocks**: `s = A_ij; s = s - L_ik * L_jk` for k ascending; then `L_ij = s / L_jj`, or for j = i, `L_ii = sqrt(s)` with the pivot test `s > 0.0`, else refusal. It is a recurrence, not a free reduction, so the block rule does not apply |
| `trsv` forward (L x = b) | `L_ik * x_k`, k = 0..i-1 | `s = b_i; s = s - L_ik * x_k` for k ascending; `x_i = s / L_ii` |
| `trsv` backward (L^T x = b) | `L_ki * x_k`, k = d-1 down to i+1 | `s = b_i; s = s - L_ki * x_k` for k **descending** from d-1; `x_i = s / L_ii`; i from d-1 down. *(Corrected in place: the first draft said ascending k. Ascending k makes the first term of row i the x just computed, so the whole solve is one serial chain; descending k lets four columns be summed together. Both are valid fixed orders; this one is the one implemented, and `scripts/bench_linalg.py` measures the difference.)* |

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

## 4. Tests and mutants (all gates in `tests/` and `scripts/`)

1. `cancho test` unit tests (`tests/*.cho`): each kernel against a naive implementation written in the section 2 order in the
   same test, on random and adversarial inputs (cancellation, mixed magnitudes, subnormals, signed zeros).
2. `scripts/oracle_linalg.py`: the same inputs (SplitMix64 generator, `docs/rng.md`) through the compiled `linalg-check`
   binary and through numpy emulating the section 2 order; bit comparison.
3. `scripts/mutants_linalg.py`: each mutant is a source edit to a copy of a kernel; the gates must fail on it. Includes the
   thread-order mutant. Survivors are listed in section 5 with a reason.
4. Fuzz (`scripts/fuzz_tensor.py`): random descriptors with out-of-range ints, negative strides, overflowing products; the process
   must refuse with a tag and never trap.

## 5. Results

*Not yet measured.*

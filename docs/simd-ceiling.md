# SIMD ceiling for the cancho-ml linear algebra (experiment, no cancho change)

**Status: section 1 (the gate) and section 2 (the lane orders) were written before any kernel code. Results are in section 3
onwards and only contain numbers a script of this repository printed.**

**Record note (added when this file was copied into this repository).** Source: clone `cancho-ml-simd`, branch `simd-ceiling`: gate and
lane orders `bac2412` (2026-10-12 22:22), kernels and checkers `4152248` (23:41), this document `5814443` (2026-10-13 00:04). The measurements are of
2026-10-12 evening, on the machine and under the load stated in section 3; the document carries no other date. Its references to `linalg.md`
sections are to `docs/linalg.md` as it stood on that branch (328 lines), where section 2 was the order **before** the four lanes of
`docs/lanes.md` were adopted (the lane count this experiment recommends); the same section numbers exist in the current file. The scripts
are `scripts/simd/`, the raw data `docs/bench/simd-*` (largest file `docs/bench/simd-gate.jsonl`, 67,500 bytes), and `docs/spikes.md` section 16
is a summary of this document. `scripts/c_parity/parity.c`, `scripts/bench_linalg.py`, `scripts/oracle_linalg.py` and `tests/lref.cho` named below
are files of this repository.

**What this is, and is not.** An upper-bound experiment: *how fast can the cancho-ml kernels get on this Mac if SIMD were
available and the summation order were SIMD-friendly*, using **C with NEON intrinsics compiled by Apple clang**. It is a ceiling
for **a C compiler given hand-written intrinsics**. It is **not** a prediction of what cancho would emit: cancho has no vector
type (`linalg.md` 5.6 U1), a cancho SIMD backend might do better (a smarter register allocator is unlikely, but a vendor library
trick such as AMX is out of reach for both) or worse (cancho code is 0.95 to 1.16 of plain C, `linalg.md` 5.3 item 1, and
nothing says that carries over to vector code). Nothing here changes the cancho compiler or `lib/linalg`.

## 0. Result in one paragraph

Measured on an Apple M4 Max, one thread, load average 2.2 to 6.3 (every cell flagged), 23 gate cells: a C-with-NEON implementation of dot, axpy,
gemv_n, gemv_t, syrk and a blocked right-looking Cholesky, **each bit-identical to a scalar reference of a documented lane order for
L in {1, 2, 4, 8, 16}** (162 mutants killed, 22 equivalent and explained, none survive), **matches numpy/scipy (Accelerate, 1 thread)
in 4 of 23 cells** (gemv_n at d = 20, Cholesky d = 21 and 64) **and does not in 19**: dot 2.35x slower, axpy 2.12x, gemv_n d >= 64
1.47x to 1.68x, gemv_t 1.43x to 2.39x, syrk 2.85x to 13.6x, Cholesky d = 256 1.67x. Adding fma (bits change) flips no verdict.
syrk and Cholesky are at 75% to 98% of the measured NEON compute peak (Cholesky 76%), so they are bounded; the large-d gemv cells are bounded by single-core
NEON read bandwidth (114 to 121 GB/s measured) below numpy's own 114 to 163 GB/s. L = 4 is the recommended lane count. This is a ceiling
for a C compiler with intrinsics, not a prediction of what cancho would emit. Sections 3 to 13 give the evidence.

## 1. The gate (fixed before any kernel code)

### 1.1 What is measured

Six kernels of `docs/linalg.md` section 1.2, in C, one thread, `clang -O3 -mcpu=native -fno-fast-math -ffp-contract=off`
(Apple clang 21.0.0, Apple M4 Max, arm64):

| kernel | C implementations timed |
|---|---|
| `dot`, `axpy`, `gemv_n`, `gemv_t`, `syrk` (X^T X, one triangle), `chol` (blocked, right-looking) | **CS** = plain scalar C of the *current* cancho order (`scripts/c_parity/parity.c`, unchanged, `linalg.md` section 2); **N(L)** = NEON intrinsics, fixed lane order of section 2 below, `L` in {2, 4, 8, 16} (plus `L = 1` where the kernel vectorises across independent outputs); then **F(L)** = the same with `vfmaq_f64`, **flagged as changing bits**, reported apart and never used in the verdict |

`trsv` and `gemm` are outside this experiment. Shapes are the shapes of `linalg.md` 1.2: `dot` and `axpy` n = 1e6; `gemv_n`,
`gemv_t`, `syrk` with n in {1e5, 1e6}, d in {20, 64, 256}; `chol` d in {21, 64, 256} (d = 256 is the one the task names; 21 and 64
are cheap and show the small-d side). A shape that cannot be run is reported as "not run" with the reason, never dropped.

### 1.2 Protocol (the protocol of `linalg.md` 1.1, reused)

* **Reference** = numpy / scipy as installed, **one thread** (`VECLIB_MAXIMUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1`),
  called exactly as `scripts/bench_linalg.py` calls them (it is imported, not copied: `numpy_case`, `raw_case`, `GATE`). For
  Cholesky the raw `lapack.dpotrf` is timed too.
* **Statistic**: minimum of **7** samples per cell, the numpy sample and the C process **interleaved** (one numpy sample, then one
  C process which times every C variant of the cell in turn). **Spread** = max / min printed per variant; a cell with spread
  > 1.25 is rerun up to twice and the best attempt kept, as in `linalg.md`. **Load average (1 min) read at the start and at the
  end of each cell; a cell with load >= 2 at either end is flagged.** A flagged cell is not a quiet-machine number.
* A C sample times the kernel only (data generation, allocation, hashing outside), the mean of `R` runs chosen so a sample is at
  least 40 ms (a numpy sample at least 50 ms, as in the harness).
* **Cancho column** = the figure printed in `docs/bench/gate-1thread.jsonl` (minimum of 7, a different session, a different load):
  it is *copied*, not re-measured, so a ratio against it mixes sessions and says so. The *same-session* anchor is CS (plain C of
  the same order), and `linalg.md` 5.3 measured cancho / CS = 0.95 to 1.16.
* Inputs: the SplitMix64 generator of `docs/rng.md`, seed 1, the same bits numpy gets in `bench_linalg.py`.

### 1.3 Correctness gate (bits, not tolerances)

For every kernel, every `L`, both the unfused and the fma variant, the NEON result must be **byte-identical** to a plain
scalar C reference written from the *definition* of the order (section 2), with runtime `L`, level-wise trees, no vector code,
on a shape list that includes empty, ragged, aligned, odd and `> K` sizes (`simd check`). Each kernel also runs at bench shapes
whenever the reference is cheap enough (the caps are stated in section 4). **The gate must be able to fail**: the reference carries runtime mutants
(lane count doubled; lane tree replaced by a left fold; element-to-lane map changed; block size changed; panel width changed for
`chol`) and for each (kernel, `L`) the mutated reference must be **different** from the NEON result on at least one input of the
suite. A mutant that is not detected voids the gate for that (kernel, L) unless it is provably equivalent, and then the reason is
written next to it (the one expected case: for `L = 2` the tree and a left fold are the same additions in the same order).

### 1.4 What "MATCH numpy" means here

The `linalg.md` 1.3 rule: **time(C-NEON best L, unfused) / time(numpy) <= 1.0** on the cell. Reported next to it: the same ratio
for CS, for F(best) (never in the verdict), and for cancho (copied figure). For the memory-bound `gemv` cells at large d the
bandwidth in GB/s (bytes of X / time) of the best N(L) and of numpy.

### 1.5 What the gate does not say

Not another machine, not several threads, not a prediction of cancho (see the top). The thread-count independence of the order is
argued from its definition in section 2 (the order is a function of indices only) and **not measured here**: this experiment is
single-threaded. The load on this shared machine was not quiet (see section 3); every number is conditioned on it.

## 2. The fixed lane orders (normative for the experiment)

`K = 256` as in `linalg.md` 2. **L** is a power of two. All sums start from `+0.0`; a product is rounded before it is added
(`fl(a*b)`, then `fl(acc + p)`); the fma variant instead does `fma(a, b, acc)` for the *lane accumulation only*, everything else
unfused.

**Lane rule (inside a block).** A reduction over terms `t_0 .. t_{m-1}` of one block (`m <= K`, the terms numbered from the block
start) has **L accumulators** `s_0 .. s_{L-1}`; **term `i` goes to lane `i mod L`**, each lane adding its terms in ascending
index order. The lanes are combined by the **level-wise tree** `q_i = s_{2i} + s_{2i+1}` (left operand first), repeated until one
value remains (L is a power of two, so there is no odd carry). An empty lane holds `+0.0`. The block partial is that value.
**Block rule and tree rule across blocks: unchanged** (`linalg.md` 2): blocks of K terms, partials combined by the fixed binary
tree by block index. Because `K` is a multiple of every `L`, the lane of a term is also `(i mod L)` counted from the start of the
reduction. *The order is a function of the term index only*: not of the vector width (a 2-lane register holds lane pairs; with
a 4-lane register the same lanes would be the same accumulators), not of the thread count (a chunk of whole blocks is an aligned
subtree, as in `linalg.md` 2 "streaming form").

| kernel | order |
|---|---|
| `dot(x, y)` | lane rule over i, then block tree |
| `axpy` | none (elementwise `y_i + fl(a * x_i)`) |
| `gemv_n` (row i of X times w) | per row exactly `dot` of the row and w (d <= K: one block, lane rule over j) |
| `gemv_t` (column j: sum over rows i of `X_ij r_i`) | per column exactly `dot` of the column and r: lane rule over rows i within each block of K rows, block tree |
| `syrk` (entry j <= k: sum over rows of `X_ij X_ik`) | per entry as `gemv_t` |
| `chol` (lower; entry (i, j), i >= j) | panels of **NB = 32** columns, aligned at multiples of 32. Let `P_p(i, j)` = lane-rule sum of `L_ik L_jk` over `k` in panel `p` (lane = `(k - panel start) mod L`) restricted to `k < j`. With `c` the panel that contains `j`: `s = A_ij; s = s - P_0; s = s - P_1; ... (every full panel before c, ascending); s = s - P_c(i, j)` (the in-panel terms, possibly empty = `+0.0`); then `L_ij = s / L_jj`, or for `i = j`, `L_ii = sqrt(s)` with the test `s > 0.0`. This is the order a blocked right-looking algorithm computes (trailing update `A22 -= L21 L21^T` per panel, panel factored with the in-panel sums) and it is **different from** the one-running-value order of `linalg.md` 2.1: **NB and L are part of the order** |

Why `chol` changes order: a blocked algorithm cannot subtract term by term in ascending `k`; the panel sum has to be formed
first. That is a decision to be made by whoever writes the normative order, not a detail.

`L = 1` (the order of `linalg.md` 2 for `gemv_t`, `syrk`) is run as an extra, vectorised across independent outputs; for `chol`,
`L = 1` is the lane rule with one lane per panel (the NB-panel order). `dot` and `gemv_n` have no `L = 1` NEON kernel (the
reduction runs along the contiguous axis).

**"Free order" comparators** (what the fixed order costs): `dot` with 16 lanes and no blocks, one tree at the end (`free16`,
bits differ from every fixed order); and for the other kernels the cost is read as N(L) against N(1) (the current order,
vectorised) and as the difference between L values. Where no free comparator exists the cell says so.

## 3. What was run, and under what conditions

* **Machine**: Apple M4 Max, macOS 26.2, Apple clang 21.0.0, `-O3 -mcpu=native -fno-fast-math -ffp-contract=off`, one thread,
  numpy 2.4.6 / scipy 1.17.1 with Accelerate and `VECLIB_MAXIMUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1` (set by
  `scripts/simd/bench_simd.py` itself; whether Accelerate honours it was checked for the same stack in `linalg.md` section 5).
* **Load**: the machine is shared (5 users logged in). The 1-minute load average was **2.2 to 3.2 for every cell from `dot` to `syrk` 1e6 x 64, and 5.3 to 6.3 for `syrk` 1e6 x 256 and the three
  Cholesky cells** (`docs/bench/simd-gate.jsonl`, per-cell values
  in the last table of section 6). **Every cell is flagged `LOAD>=2`; none is a quiet-machine number.** A preliminary run of the
  same script (`docs/bench/simd-gate-prelim-loaded.*`, load 4 to 12, *before* the `gemv_t` and `syrk` kernels were tuned, so its
  kernels are not the ones in this document) gave the same four MATCH cells and the same NOT MATCH for the rest; it is kept as raw
  data and is not used below. The single-core read bandwidth I measured moved by 2x between a loaded and a quieter moment
  (section 5), so absolute GB/s here carry that uncertainty.
* **Protocol**: `linalg.md` 1.1 as written in section 1.2 above: minimum of 7 interleaved samples, spread max/min, a cell with spread
  > 1.25 rerun up to twice (best attempt kept; attempts are in the last table). One cell has a spread above 1.25 after its reruns: the
  CS (scalar C) timing of gemv_n 1e6 x 256 (1.37; numpy 1.08 and the NEON variant 1.09 are fine).
* **cancho column** = the figure in `docs/bench/gate-1thread.jsonl`, copied (a different session, load 3.8 to 10). Ratios against it mix
  sessions. The same-session scalar anchor is **CS**.

## 4. Correctness: bit equality, and the mutants that must be caught

`build/simd check` (`docs/bench/simd-check.log`). For each of the 10 implementations (L in {1, 2, 4, 8, 16} x {unfused, fma}; `L = 1`
exists only for `gemv_t`, `syrk`, `chol`) and each kernel the NEON result is compared **bit for bit** with a plain scalar C
reference written from the definition of section 2 (runtime `L`, level-wise trees, no vector code), on 11 to 17 shapes per kernel (lists in `scripts/simd/simd.c`):
empty, 1, ragged, aligned, `K` +/- 1, several blocks (odd and even block counts), `d` not a multiple of anything (1, 3, 7, 17, 20,
21, 33, 257, 300, 600, 1000), `d > K`, Cholesky `d` = 1, 2, 5, 16, 31, 32, 33, 64, 100, 255, 256, on inputs with magnitudes from
2^-20 to 2^20 and exact zeros (so the order shows). **All 46 (kernel, implementation) pairs are identical on every shape; axpy
(no order) is identical too.** At the bench shapes the benchmark driver repeats the comparison on its first sample (always for `dot`, `gemv_n`, `chol`; `gemv_t` when
n*d < 3e8; `syrk` when n*d*d/2 < 1.5e9, which is `syrk` at d = 20 for both n and d = 64 for n = 1e5; column `bits` of the tables says `ok`); `syrk` at d = 256 (both n) and n = 1e6, d = 64 are too expensive for the reference
and are covered by the suite shapes (255 and 700 rows at d = 256, 257 rows at d = 64) only (`-` in the table).

**Mutants.** The reference carries five runtime mutants: lane count doubled, lanes combined by a left fold instead of the tree,
element-to-lane map `(i/2) mod L`, block size 128 instead of 256, Cholesky panel 16 instead of 32. For every (kernel,
implementation) the mutated reference must **differ** from the NEON result on some suite shape:
**162 killed, 22 equivalent, 0 survive.** The 22 equivalent are explained, not hidden: for `L <= 2` a left fold of at most two
lanes *is* the tree (a + b = b + a bit for bit), 10 cases; for `L = 1` the lane map `(i/2) mod 1` and the left fold of one lane do not
change anything, 12 cases. The gate can therefore fail, and it fails where the order is changed.

The thread-count independence of the order is by construction (a function of indices only, section 2) and **was not measured**:
nothing here is threaded.

## 5. Two probes that bound the kernels (not kernels; `scripts/simd/bw.c`, `scripts/simd/peak.c`, `docs/bench/simd-probes.txt`)

Run once after the benchmark, load average 5.0, so they are conditions-specific.

* **Single-core read bandwidth of NEON loads** (sum of an array with 8 accumulators; the best of plain 16-byte loads, `ld1`
  of four registers, 2, 4 and 8 concurrent streams): **121 GB/s at 16 MB, 119 GB/s at 51 MB, 114 GB/s at 205 MB, 113.5 GB/s at 2 GB**.
  A plain single stream gave 120 / 110 / 94 / 90 GB/s; software prefetch made it *slower* (95 / 97 / 90 / 86 GB/s). At an
  earlier, busier moment (load average about 3 to 6, a first run of the same program, not saved) the same loop gave 105 / 63 / 69 / 56 GB/s: the bandwidth of this machine depends on what
  the other users do.
* **Register-only vector throughput** (16 independent chains, nothing hoisted): the unfused pair `vmulq + vaddq` reaches
  **7.47 G vector multiply-adds/s = 14.9 GMAC/s** (2 lanes); `vfmaq_f64` reaches **11.97 G/s = 23.9 GMAC/s**, i.e. fma is
  **1.6x** the unfused rate, not 2x.

## 6. Results per kernel and shape

Tables are generated by `scripts/simd/report.py` from `docs/bench/simd-gate.jsonl` (`docs/bench/simd-tables.md` has the same text).
Columns: **CS** = plain scalar C of the current cancho order (`parity.c`); **N`L`** = NEON, unfused, lane order of section 2 with
`L` lanes (bits identical to the scalar reference of that order); **numpy** = the call of `linalg.md` 1.2, same session, interleaved;
**cancho (copied)** from `docs/bench`; the ratios are `N(best L) / numpy`, `CS / numpy` and `cancho / numpy` (above 1.0 = slower than numpy).
`F` tables are the fma variants: **they change bits** and are *not* used in any verdict.

#### dot (unfused, bits identical to the scalar reference of each order)

| shape | CS | N2 | N4 | N8 | N16 | best L | numpy | cancho (copied) | N(best) / numpy | CS / numpy | cancho / numpy | verdict | bits | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=1000000 | 166.16 us | 227.59 us | 143.14 us | 137.65 us | 134.16 us | N16 | 57.11 us | 278.00 us | 2.35 | 2.91 | 4.87 | NOT MATCH (2.35x) | ok | LOAD>=2 |


#### dot, fma variant (CHANGES BITS; not in the verdict)

| shape | F2 | F4 | F8 | F16 | best | F(best) / numpy | F(best) / N(best) | bits |
|---|---|---|---|---|---|---|---|---|
| n=1000000 | 270.75 us | 142.93 us | 135.27 us | 131.26 us | F16 | 2.30 | 0.98 | ok |


#### axpy (unfused, bits identical to the scalar reference of each order)

| shape | C scalar (CS) | C NEON | numpy | cancho (copied) | NEON / numpy | CS / numpy | cancho / numpy | verdict | notes |
|---|---|---|---|---|---|---|---|---|---|
| n=1000000 | 152.71 us | 156.28 us | 73.75 us | 187.10 us | 2.12 | 2.07 | 2.54 | NOT MATCH (2.12x) | LOAD>=2 |


#### gemv_n (unfused, bits identical to the scalar reference of each order)

| shape | CS | N2 | N4 | N8 | N16 | best L | numpy | cancho (copied) | N(best) / numpy | CS / numpy | cancho / numpy | verdict | bits | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=100000 d=20 | 233.63 us | 158.91 us | 163.88 us | 436.35 us | 746.79 us | N2 | 179.43 us | 313.73 us | 0.89 | 1.30 | 1.75 | MATCH | ok | LOAD>=2 |
| n=100000 d=64 | 777.73 us | 561.96 us | 548.46 us | 546.65 us | 563.28 us | N8 | 325.63 us | 1.067 ms | 1.68 | 2.39 | 3.28 | NOT MATCH (1.68x) | ok | LOAD>=2 |
| n=100000 d=256 | 3.856 ms | 2.689 ms | 2.563 ms | 2.496 ms | 2.512 ms | N8 | 1.652 ms | 5.400 ms | 1.51 | 2.33 | 3.27 | NOT MATCH (1.51x) | ok | LOAD>=2 |
| n=1000000 d=20 | 2.389 ms | 1.794 ms | 1.817 ms | 4.357 ms | 7.553 ms | N2 | 2.121 ms | 3.643 ms | 0.85 | 1.13 | 1.72 | MATCH | ok | LOAD>=2 |
| n=1000000 d=64 | 8.076 ms | 6.288 ms | 6.170 ms | 6.072 ms | 6.165 ms | N8 | 3.836 ms | 13.000 ms | 1.58 | 2.11 | 3.39 | NOT MATCH (1.58x) | ok | LOAD>=2 |
| n=1000000 d=256 | 47.919 ms | 28.157 ms | 27.561 ms | 27.680 ms | 26.532 ms | N16 | 18.028 ms | 0.131 s | 1.47 | 2.66 | 7.27 | NOT MATCH (1.47x) | ok | spread 1.37, LOAD>=2 |


#### gemv_n, fma variant (CHANGES BITS; not in the verdict)

| shape | F2 | F4 | F8 | F16 | best | F(best) / numpy | F(best) / N(best) | bits |
|---|---|---|---|---|---|---|---|---|
| n=100000 d=20 | 157.47 us | 163.98 us | 431.13 us | 737.69 us | F2 | 0.88 | 0.99 | ok |
| n=100000 d=64 | 536.32 us | 527.83 us | 511.53 us | 535.00 us | F8 | 1.57 | 0.94 | ok |
| n=100000 d=256 | 2.665 ms | 2.461 ms | 2.403 ms | 2.439 ms | F8 | 1.45 | 0.96 | ok |
| n=1000000 d=20 | 1.751 ms | 1.782 ms | 4.300 ms | 7.351 ms | F2 | 0.83 | 0.98 | ok |
| n=1000000 d=64 | 6.136 ms | 5.929 ms | 5.800 ms | 5.933 ms | F8 | 1.51 | 0.96 | ok |
| n=1000000 d=256 | 28.333 ms | 26.524 ms | 26.590 ms | 25.537 ms | F16 | 1.42 | 0.96 | ok |


#### gemv_t (unfused, bits identical to the scalar reference of each order)

| shape | CS | N1 | N2 | N4 | N8 | N16 | best L | numpy | cancho (copied) | N(best) / numpy | CS / numpy | cancho / numpy | verdict | bits | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=100000 d=20 | 189.51 us | 194.79 us | 185.01 us | 191.97 us | 207.82 us | 217.98 us | N2 | 118.83 us | 296.52 us | 1.56 | 1.59 | 2.50 | NOT MATCH (1.56x) | ok | LOAD>=2 |
| n=100000 d=64 | 583.53 us | 594.63 us | 620.17 us | 624.86 us | 812.71 us | 847.20 us | N1 | 319.29 us | 900.00 us | 1.86 | 1.83 | 2.82 | NOT MATCH (1.86x) | ok | LOAD>=2 |
| n=100000 d=256 | 2.649 ms | 2.682 ms | 3.161 ms | 3.216 ms | 3.377 ms | 3.345 ms | N1 | 1.252 ms | 4.520 ms | 2.14 | 2.12 | 3.61 | NOT MATCH (2.14x) | ok | LOAD>=2 |
| n=1000000 d=20 | 2.003 ms | 2.049 ms | 1.983 ms | 1.998 ms | 2.176 ms | 2.556 ms | N2 | 1.389 ms | 3.143 ms | 1.43 | 1.44 | 2.26 | NOT MATCH (1.43x) | ok | LOAD>=2 |
| n=1000000 d=64 | 6.334 ms | 6.452 ms | 6.610 ms | 6.815 ms | 8.228 ms | 8.511 ms | N1 | 3.281 ms | 11.667 ms | 1.97 | 1.93 | 3.56 | NOT MATCH (1.97x) | ok | LOAD>=2 |
| n=1000000 d=256 | 27.650 ms | 30.776 ms | 32.001 ms | 32.935 ms | 34.138 ms | 34.418 ms | N1 | 12.903 ms | 95.000 ms | 2.39 | 2.14 | 7.36 | NOT MATCH (2.39x) | ok | LOAD>=2 |


#### gemv_t, fma variant (CHANGES BITS; not in the verdict)

| shape | F1 | F2 | F4 | F8 | F16 | best | F(best) / numpy | F(best) / N(best) | bits |
|---|---|---|---|---|---|---|---|---|---|
| n=100000 d=20 | 204.67 us | 184.86 us | 190.03 us | 203.54 us | 203.75 us | F2 | 1.56 | 1.00 | ok |
| n=100000 d=64 | 586.68 us | 605.89 us | 609.93 us | 812.42 us | 835.83 us | F1 | 1.84 | 0.99 | ok |
| n=100000 d=256 | 2.537 ms | 3.096 ms | 3.210 ms | 3.337 ms | 3.322 ms | F1 | 2.03 | 0.95 | ok |
| n=1000000 d=20 | 2.120 ms | 1.963 ms | 1.963 ms | 2.142 ms | 2.431 ms | F4 | 1.41 | 0.99 | ok |
| n=1000000 d=64 | 6.407 ms | 6.447 ms | 6.749 ms | 8.140 ms | 8.383 ms | F1 | 1.95 | 0.99 | ok |
| n=1000000 d=256 | 27.005 ms | 31.629 ms | 32.119 ms | 34.040 ms | 34.180 ms | F1 | 2.09 | 0.88 | ok |


#### syrk (unfused, bits identical to the scalar reference of each order)

| shape | CS | N1 | N2 | N4 | N8 | N16 | best L | numpy | cancho (copied) | N(best) / numpy | CS / numpy | cancho / numpy | verdict | bits | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=100000 d=20 | 1.593 ms | 4.234 ms | 1.661 ms | 1.795 ms | 1.499 ms | 2.095 ms | N8 | 525.82 us | 1.880 ms | 2.85 | 3.03 | 3.58 | NOT MATCH (2.85x) | ok | LOAD>=2 |
| n=100000 d=64 | 14.924 ms | 17.471 ms | 17.368 ms | 18.278 ms | 16.863 ms | 23.679 ms | N8 | 1.486 ms | 17.375 ms | 11.34 | 10.04 | 11.69 | NOT MATCH (11.34x) | ok | LOAD>=2 |
| n=100000 d=256 | 0.269 s | 0.225 s | 0.234 s | 0.257 s | 0.264 s | 0.423 s | N1 | 17.762 ms | 0.290 s | 12.68 | 15.14 | 16.33 | NOT MATCH (12.68x) | - | LOAD>=2 |
| n=1000000 d=20 | 18.247 ms | 44.127 ms | 20.741 ms | 20.579 ms | 17.570 ms | 23.565 ms | N8 | 5.540 ms | 22.333 ms | 3.17 | 3.29 | 4.03 | NOT MATCH (3.17x) | ok | LOAD>=2 |
| n=1000000 d=64 | 0.159 s | 0.196 s | 0.195 s | 0.203 s | 0.186 s | 0.259 s | N8 | 14.989 ms | 0.180 s | 12.44 | 10.63 | 12.01 | NOT MATCH (12.44x) | - | LOAD>=2 |
| n=1000000 d=256 | 2.840 s | 2.423 s | 2.463 s | 2.707 s | 2.747 s | 4.437 s | N1 | 0.178 s | 2.734 s | 13.60 | 15.95 | 15.35 | NOT MATCH (13.60x) | - | LOAD>=2 |


#### syrk, fma variant (CHANGES BITS; not in the verdict)

| shape | F1 | F2 | F4 | F8 | F16 | best | F(best) / numpy | F(best) / N(best) | bits |
|---|---|---|---|---|---|---|---|---|---|
| n=100000 d=20 | 4.516 ms | 1.013 ms | 1.234 ms | 1.202 ms | 2.067 ms | F2 | 1.93 | 0.68 | ok |
| n=100000 d=64 | 11.134 ms | 11.035 ms | 13.689 ms | 13.900 ms | 23.036 ms | F2 | 7.42 | 0.65 | ok |
| n=100000 d=256 | 0.134 s | 0.147 s | 0.210 s | 0.227 s | 0.414 s | F1 | 7.57 | 0.60 | - |
| n=1000000 d=20 | 47.382 ms | 12.812 ms | 15.626 ms | 14.506 ms | 22.895 ms | F2 | 2.31 | 0.73 | ok |
| n=1000000 d=64 | 0.133 s | 0.131 s | 0.157 s | 0.159 s | 0.251 s | F2 | 8.75 | 0.70 | - |
| n=1000000 d=256 | 1.417 s | 1.510 s | 2.218 s | 2.397 s | 4.386 s | F1 | 7.96 | 0.58 | - |


#### chol (unfused, bits identical to the scalar reference of each order)

| shape | CS | N1 | N2 | N4 | N8 | N16 | best L | numpy | cancho (copied) | N(best) / numpy | CS / numpy | cancho / numpy | verdict | bits | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d=21 | 0.35 us | 0.40 us | 0.40 us | 0.34 us | 0.38 us | 0.57 us | N4 | 2.38 us | 0.58 us | 0.14 | 0.15 | 0.24 | MATCH | ok | LOAD>=2; vs raw dpotrf 0.49 us: 0.70x |
| d=64 | 7.23 us | 4.55 us | 4.36 us | 4.36 us | 4.65 us | 8.24 us | N2 | 8.90 us | 8.76 us | 0.49 | 0.81 | 0.98 | MATCH | ok | LOAD>=2; vs raw dpotrf 5.20 us: 0.84x |
| d=256 | 481.81 us | 246.79 us | 245.14 us | 274.92 us | 264.37 us | 490.69 us | N2 | 146.76 us | 490.26 us | 1.67 | 3.28 | 3.34 | NOT MATCH (1.67x) | ok | LOAD>=2; vs raw dpotrf 47.25 us: 5.19x |


#### chol, fma variant (CHANGES BITS; not in the verdict)

| shape | F1 | F2 | F4 | F8 | F16 | best | F(best) / numpy | F(best) / N(best) | bits |
|---|---|---|---|---|---|---|---|---|---|
| d=21 | 0.34 us | 0.35 us | 0.31 us | 0.36 us | 0.55 us | F4 | 0.13 | 0.90 | ok |
| d=64 | 3.38 us | 3.44 us | 3.87 us | 4.14 us | 7.70 us | F1 | 0.38 | 0.78 | ok |
| d=256 | 171.06 us | 185.64 us | 296.13 us | 253.08 us | 468.71 us | F1 | 1.17 | 0.70 | ok |


#### Spreads and load per cell

| kernel | shape | spread numpy | spread CS | spread N(best) | load start / end | attempts |
|---|---|---|---|---|---|---|
| dot | n=1000000 | 1.18 | 1.07 | 1.03 | 2.3 / 2.3 | 2 |
| axpy | n=1000000 | 1.05 | 1.05 | 1.08 | 2.3 / 2.3 | 1 |
| gemv_n | n=100000 d=20 | 1.02 | 1.07 | 1.04 | 2.3 / 2.3 | 1 |
| gemv_n | n=100000 d=64 | 1.23 | 1.14 | 1.04 | 2.3 / 2.2 | 1 |
| gemv_n | n=100000 d=256 | 1.15 | 1.10 | 1.06 | 2.4 / 2.4 | 3 |
| gemv_n | n=1000000 d=20 | 1.04 | 1.03 | 1.01 | 2.3 / 2.3 | 1 |
| gemv_n | n=1000000 d=64 | 1.09 | 1.01 | 1.04 | 2.4 / 2.4 | 1 |
| gemv_n | n=1000000 d=256 | 1.08 | 1.37 | 1.09 | 3.2 / 3.2 | 2 |
| gemv_t | n=100000 d=20 | 1.17 | 1.02 | 1.07 | 2.3 / 2.3 | 1 |
| gemv_t | n=100000 d=64 | 1.04 | 1.04 | 1.06 | 2.2 / 2.2 | 1 |
| gemv_t | n=100000 d=256 | 1.01 | 1.05 | 1.12 | 2.4 / 2.4 | 1 |
| gemv_t | n=1000000 d=20 | 1.01 | 1.12 | 1.08 | 2.3 / 2.3 | 1 |
| gemv_t | n=1000000 d=64 | 1.01 | 1.05 | 1.12 | 2.4 / 2.3 | 1 |
| gemv_t | n=1000000 d=256 | 1.07 | 1.05 | 1.15 | 3.1 / 3.0 | 1 |
| syrk | n=100000 d=20 | 1.02 | 1.09 | 1.11 | 2.3 / 2.3 | 1 |
| syrk | n=100000 d=64 | 1.05 | 1.06 | 1.04 | 2.2 / 2.4 | 1 |
| syrk | n=100000 d=256 | 1.01 | 1.05 | 1.06 | 2.4 / 2.3 | 1 |
| syrk | n=1000000 d=20 | 1.04 | 1.05 | 1.05 | 2.3 / 2.2 | 1 |
| syrk | n=1000000 d=64 | 1.01 | 1.06 | 1.06 | 3.0 / 3.2 | 3 |
| syrk | n=1000000 d=256 | 1.05 | 1.15 | 1.02 | 6.1 / 6.3 | 2 |
| chol | d=21 | 1.16 | 1.10 | 1.03 | 6.3 / 5.5 | 1 |
| chol | d=64 | 1.03 | 1.07 | 1.06 | 5.5 / 5.3 | 1 |
| chol | d=256 | 1.08 | 1.05 | 1.05 | 5.3 / 5.3 | 1 |



## 7. Verdict: would a SIMD cancho at this ceiling MATCH numpy?

#### Verdict table: would a SIMD cancho at this ceiling MATCH numpy? (unfused, bit-identical to the scalar reference; fma shown apart)

| kernel | shape | numpy | N(best L) | N / numpy | verdict | min(CS,N) / numpy | F(best) / numpy (bits change) | cancho (copied) / N(best) | notes |
|---|---|---|---|---|---|---|---|---|---|
| dot | n=1000000 | 57.11 us | N16 134.16 us | 2.35 | NOT MATCH | 2.35 | 2.30 | 2.07x | LOAD>=2 |
| axpy | n=1000000 | 73.75 us | N 156.28 us | 2.12 | NOT MATCH | 2.07 | 2.08 | 1.20x | LOAD>=2 |
| gemv_n | n=100000 d=20 | 179.43 us | N2 158.91 us | 0.89 | MATCH | 0.89 | 0.88 | 1.97x | LOAD>=2 |
| gemv_n | n=100000 d=64 | 325.63 us | N8 546.65 us | 1.68 | NOT MATCH | 1.68 | 1.57 | 1.95x | LOAD>=2 |
| gemv_n | n=100000 d=256 | 1.652 ms | N8 2.496 ms | 1.51 | NOT MATCH | 1.51 | 1.45 | 2.16x | LOAD>=2 |
| gemv_n | n=1000000 d=20 | 2.121 ms | N2 1.794 ms | 0.85 | MATCH | 0.85 | 0.83 | 2.03x | LOAD>=2 |
| gemv_n | n=1000000 d=64 | 3.836 ms | N8 6.072 ms | 1.58 | NOT MATCH | 1.58 | 1.51 | 2.14x | LOAD>=2 |
| gemv_n | n=1000000 d=256 | 18.028 ms | N16 26.532 ms | 1.47 | NOT MATCH | 1.47 | 1.42 | 4.94x | spread 1.37, LOAD>=2 |
| gemv_t | n=100000 d=20 | 118.83 us | N2 185.01 us | 1.56 | NOT MATCH | 1.56 | 1.56 | 1.60x | LOAD>=2 |
| gemv_t | n=100000 d=64 | 319.29 us | N1 594.63 us | 1.86 | NOT MATCH | 1.83 | 1.84 | 1.51x | LOAD>=2 |
| gemv_t | n=100000 d=256 | 1.252 ms | N1 2.682 ms | 2.14 | NOT MATCH | 2.12 | 2.03 | 1.69x | LOAD>=2 |
| gemv_t | n=1000000 d=20 | 1.389 ms | N2 1.983 ms | 1.43 | NOT MATCH | 1.43 | 1.41 | 1.59x | LOAD>=2 |
| gemv_t | n=1000000 d=64 | 3.281 ms | N1 6.452 ms | 1.97 | NOT MATCH | 1.93 | 1.95 | 1.81x | LOAD>=2 |
| gemv_t | n=1000000 d=256 | 12.903 ms | N1 30.776 ms | 2.39 | NOT MATCH | 2.14 | 2.09 | 3.09x | LOAD>=2 |
| syrk | n=100000 d=20 | 525.82 us | N8 1.499 ms | 2.85 | NOT MATCH | 2.85 | 1.93 | 1.25x | LOAD>=2 |
| syrk | n=100000 d=64 | 1.486 ms | N8 16.863 ms | 11.34 | NOT MATCH | 10.04 | 7.42 | 1.03x | LOAD>=2 |
| syrk | n=100000 d=256 | 17.762 ms | N1 0.225 s | 12.68 | NOT MATCH | 12.68 | 7.57 | 1.29x | LOAD>=2 |
| syrk | n=1000000 d=20 | 5.540 ms | N8 17.570 ms | 3.17 | NOT MATCH | 3.17 | 2.31 | 1.27x | LOAD>=2 |
| syrk | n=1000000 d=64 | 14.989 ms | N8 0.186 s | 12.44 | NOT MATCH | 10.63 | 8.75 | 0.97x | LOAD>=2 |
| syrk | n=1000000 d=256 | 0.178 s | N1 2.423 s | 13.60 | NOT MATCH | 13.60 | 7.96 | 1.13x | LOAD>=2 |
| chol | d=21 | 2.38 us | N4 0.34 us | 0.14 | MATCH | 0.14 | 0.13 | 1.69x | LOAD>=2; vs raw dpotrf: 0.70 |
| chol | d=64 | 8.90 us | N2 4.36 us | 0.49 | MATCH | 0.49 | 0.38 | 2.01x | LOAD>=2; vs raw dpotrf: 0.84 |
| chol | d=256 | 146.76 us | N2 245.14 us | 1.67 | NOT MATCH | 1.67 | 1.17 | 2.00x | LOAD>=2; vs raw dpotrf: 5.19 |



* **MATCH in 4 of 23 cells**: gemv_n at d = 20 (0.89 and 0.85 of numpy's time, where numpy is itself slow: 89 and 75 GB/s against
  its 114 to 157 GB/s elsewhere) and Cholesky d = 21 and d = 64 (0.14 and 0.49 of `scipy.linalg.cholesky`; against the raw
  `lapack.dpotrf` 0.70 and 0.84, so they hold against the strictest reference too; at these sizes the scipy wrapper's own cost is
  most of numpy's time, `linalg.md` 5.2, and plain scalar C already beats it: CS 0.15 and 0.81).
* **NOT MATCH in 19 of 23 cells**: dot 2.35x; axpy 2.12x; gemv_n d >= 64 1.47x to 1.68x; gemv_t 1.43x to 2.39x; syrk 2.85x to 13.6x;
  Cholesky d = 256 1.67x (5.19x against the raw `dpotrf`).
* **The fma variant does not change a single verdict.** Its best ratios: gemv_n 1.42x to 1.57x (d >= 64), gemv_t 1.41x to 2.09x,
  syrk 1.93x to 8.75x, Cholesky d = 256 1.17x, dot 2.30x, axpy 2.08x. It helps only the compute-bound kernels (syrk F/N =
  0.58 to 0.73, Cholesky d = 256 0.70) and nothing that waits on memory.
* **Where SIMD would help cancho** (against the copied cancho figure, a different session; against CS, same session): dot 2.07x
  over cancho, 1.24x over CS; gemv_n 1.95x to 2.16x over cancho (4.9x at 1e6 x 256), 1.33x to 1.81x over CS; Cholesky d = 256 2.0x over both;
  gemv_t 1.5x to 3.1x over cancho (3.1x only at 1e6 x 256, where the copied cancho figure is 95 ms) but **0.90 to 1.02 of CS** (no gain over what clang already does with the scalar order); syrk
  0.97x to 1.29x over cancho and **0.85 to 1.19 of CS** (nothing); axpy 1.20x over cancho and 0.98 of CS (nothing).

**This is a ceiling for a C compiler with intrinsics** on this machine at this load; it is **not** a prediction of what cancho would emit.

## 8. The memory-bound cells: bandwidth in GB/s

#### Bandwidth of the memory-bound cells (bytes of X read / time; dot counts both vectors, axpy counts x read, y read and written)

| kernel | shape | bytes | N(best) GB/s | CS GB/s | numpy GB/s | cancho GB/s (copied) | best read probe GB/s (size class) | time at probe rate / numpy |
|---|---|---|---|---|---|---|---|---|
| dot | n=1000000 | 16.0 MB | 119.3 (N16) | 96.3 | 280.2 | 57.6 | 121.2 (N=2000000) | 2.31 |
| axpy | n=1000000 | 24.0 MB | 153.6 (N) | 157.2 | 325.4 | 128.3 | 121.2 (N=2000000) | 2.68 |
| gemv_n | n=100000 d=20 | 16.0 MB | 100.7 (N2) | 68.5 | 89.2 | 51.0 | 121.2 (N=2000000) | 0.74 |
| gemv_n | n=100000 d=64 | 51.2 MB | 93.7 (N8) | 65.8 | 157.2 | 48.0 | 118.8 (N=6400000) | 1.32 |
| gemv_n | n=100000 d=256 | 204.8 MB | 82.1 (N8) | 53.1 | 123.9 | 37.9 | 114.1 (N=25600000) | 1.09 |
| gemv_n | n=1000000 d=20 | 160.0 MB | 89.2 (N2) | 67.0 | 75.4 | 43.9 | 114.1 (N=25600000) | 0.66 |
| gemv_n | n=1000000 d=64 | 512.0 MB | 84.3 (N8) | 63.4 | 133.5 | 39.4 | 114.1 (N=25600000) | 1.17 |
| gemv_n | n=1000000 d=256 | 2048.0 MB | 77.2 (N16) | 42.7 | 113.6 | 15.6 | 113.5 (N=256000000) | 1.00 |
| gemv_t | n=100000 d=20 | 16.0 MB | 86.5 (N2) | 84.4 | 134.6 | 54.0 | 121.2 (N=2000000) | 1.11 |
| gemv_t | n=100000 d=64 | 51.2 MB | 86.1 (N1) | 87.7 | 160.4 | 56.9 | 118.8 (N=6400000) | 1.35 |
| gemv_t | n=100000 d=256 | 204.8 MB | 76.4 (N1) | 77.3 | 163.5 | 45.3 | 114.1 (N=25600000) | 1.43 |
| gemv_t | n=1000000 d=20 | 160.0 MB | 80.7 (N2) | 79.9 | 115.2 | 50.9 | 114.1 (N=25600000) | 1.01 |
| gemv_t | n=1000000 d=64 | 512.0 MB | 79.4 (N1) | 80.8 | 156.0 | 43.9 | 114.1 (N=25600000) | 1.37 |
| gemv_t | n=1000000 d=256 | 2048.0 MB | 66.5 (N1) | 74.1 | 158.7 | 21.6 | 113.5 (N=256000000) | 1.40 |



* At large d the gemv cells move 205 MB to 2 GB per call. **N(best) reaches 77 to 94 GB/s on gemv_n and 66 to 86 GB/s on gemv_t,
  numpy 114 to 163 GB/s** (gemv_n 1e6 x 256: 77 against 114; gemv_t 1e6 x 256: 66 against 159). numpy's figures for gemv_t
  (156 to 163 GB/s at 51 MB to 2 GB) are **above the best NEON read loop I wrote (114 to 121 GB/s)**: whatever Accelerate does
  there, a loop of NEON loads is not limited the same way. *Not diagnosed*; a matrix coprocessor is a guess, nothing here measures it.
* The last column is the time the cell would take **if the kernel read X at the best probe rate** (bytes / probe GB/s) over numpy's
  time: 1.00 (gemv_n 1e6 x 256), 1.09 (gemv_n 1e5 x 256), 1.17 to 1.43 (gemv_n d = 64, gemv_t d >= 64); at d = 20 it is 0.66 to 0.74
  (gemv_n) and 1.01 to 1.11 (gemv_t). **So even a perfect NEON streaming kernel would at best tie numpy on the biggest gemv_n cell and
  lose by 1.1x to 1.4x on the other d >= 64 cells**; the NEON kernels here run 1.3x to 1.7x above that bound (gemv_n 1e5 x 256: 82 GB/s
  achieved against 114; gemv_t 1e6 x 256: 66 against 114), which is the room my kernels leave.
* dot and axpy are 16 and 24 MB: cache-resident, not DRAM-bound; numpy's 280 and 325 GB/s (bytes counted) are 2.3x to 2.7x above
  what a NEON read loop reached (121 GB/s). Same remark, same lack of diagnosis.
* **The cancho copy of the gemv cells is far from this bound (15.6 to 57 GB/s)**, which is where SIMD would help cancho: gemv_n 1e6 x
  256 is 131 ms in cancho, 26.5 ms here (4.9x), but the ceiling is still 1.47x numpy.

## 9. The compute-bound cells: syrk and Cholesky against the measured peak

#### Compute-bound cells against the measured register-only peak (scripts/simd/peak.c: unfused 14.9 GMAC/s, fma 23.9 GMAC/s)

| kernel | shape | useful MACs | N(best) | GMAC/s | % of unfused peak | F(best) | GMAC/s | % of fma peak | CS GMAC/s |
|---|---|---|---|---|---|---|---|---|---|
| syrk | n=100000 d=20 | 2.1e+07 | N8 | 14.01 | 94% | F2 | 20.72 | 87% | 13.18 |
| syrk | n=100000 d=64 | 2.08e+08 | N8 | 12.33 | 83% | F2 | 18.85 | 79% | 13.94 |
| syrk | n=100000 d=256 | 3.29e+09 | N1 | 14.61 | 98% | F1 | 24.47 | 102% | 12.23 |
| syrk | n=1000000 d=20 | 2.1e+08 | N8 | 11.95 | 80% | F2 | 16.39 | 68% | 11.51 |
| syrk | n=1000000 d=64 | 2.08e+09 | N8 | 11.16 | 75% | F2 | 15.86 | 66% | 13.06 |
| syrk | n=1000000 d=256 | 3.29e+10 | N1 | 13.58 | 91% | F1 | 23.21 | 97% | 11.58 |
| chol | d=21 | 1.54e+03 | N4 | 4.49 | 30% | F4 | 4.98 | 21% | 4.37 |
| chol | d=64 | 4.37e+04 | N2 | 10.03 | 67% | F1 | 12.91 | 54% | 6.05 |
| chol | d=256 | 2.8e+06 | N2 | 11.41 | 76% | F1 | 16.35 | 68% | 5.80 |



* **syrk**: the best NEON variant runs at **75% to 98% of the measured unfused register peak** (14.9 GMAC/s); the scalar C
  (CS) is already at 11.5 to 13.9 GMAC/s: clang autovectorises the 4 x 4 tile. There is no unfused headroom left in a 2-lane
  NEON; fma lifts it to 66% to 102% of its own (higher) peak. Whatever numpy does (17.8 ms for 3.29e9 MACs at n = 1e5, d = 256 is
  185 GMAC/s, 12x the unfused NEON peak), it is not a 2-lane vector loop.
* **Cholesky d = 256**: 76% of the unfused peak (counting the useful d^3/6 multiply-adds; the time includes the diagonal-tile waste in the
  trailing update, the panel factorisation and the copy of A into the workspace). Splitting the time with the
  trailing update removed (a profiling switch in `kern.h`, `CHOL_NOTRAIL`, outputs then wrong): the rest (copy of A, panel
  factorisation with its vector divisions) took 86 to 92 us for every L in that one run (load 4 to 5), of ~245 us in the full run.

## 10. What the fixed lane order costs

* **dot**: 16 free lanes with no blocks 128.6 us; the fixed order with L = 16, 134.2 us (**1.04x**); L = 8, 137.7 us (1.07x); L = 4,
  143.1 us (**1.11x**); L = 2, 227.6 us (1.77x: one accumulator register, a 3-cycle add chain per 2 elements).
* **Against the current cancho order in plain C (CS)** the lane order is *faster* on the reductions it can vectorise along the
  contiguous axis: dot 1.24x, gemv_n 1.33x to 1.81x (CS is the 4-row, one-chain-per-row loop).
* **Against the same kernel with the current order vectorised across outputs (N1)**: gemv_t is within 0% to 5% of the best L (N1 / N(best) = 1.00 to 1.05) and syrk at d >= 64 within 0% to 5%; syrk at d = 20 is 2.5x to 2.8x
  *better* with lanes (N1 has one add chain per accumulator); Cholesky N1 / N(best) is 1.01 to 1.17.
  I.e. **choosing L >= 2 costs nothing on the cells where N1 exists, and wins where the chain latency shows**.
* **Costs seen with large L** are register pressure in the tiles (syrk and Cholesky at L = 16: 1.9x to 2.0x slower than best, one
  accumulator tile of 1 x 2 outputs), the horizontal reduction and the zero-padded tail per 20-element row of gemv_n (worst cell, d = 20: L = 8 2.75x,
  L = 16 4.70x of the best L), and an L1 conflict heuristic in gemv_t (below). The first two are properties of my kernels, not necessarily of any order.

#### Which L: time of N(L) over the time of the best L, per cell (1.00 = best); worst case and geometric mean over the cells of each kernel

| kernel | L=1 worst / geomean | L=2 worst / geomean | L=4 worst / geomean | L=8 worst / geomean | L=16 worst / geomean |
|---|---|---|---|---|---|
| dot | n/a | 1.70 / 1.70 | 1.07 / 1.07 | 1.03 / 1.03 | 1.00 / 1.00 |
| gemv_n | n/a | 1.08 / 1.03 | 1.04 / 1.02 | 2.75 / 1.38 | 4.70 / 1.66 |
| gemv_t | 1.05 / 1.01 | 1.18 / 1.05 | 1.20 / 1.07 | 1.37 / 1.20 | 1.42 / 1.26 |
| syrk | 2.82 / 1.41 | 1.18 / 1.07 | 1.20 / 1.13 | 1.17 / 1.05 | 1.88 / 1.52 |
| chol | 1.17 / 1.07 | 1.17 / 1.05 | 1.12 / 1.04 | 1.11 / 1.09 | 2.00 / 1.85 |
| all cells where L exists for every kernel (dot, gemv_n, gemv_t, syrk, chol) | n/a (no dot, gemv_n) | 1.70 / 1.07 | 1.20 / 1.07 | 2.75 / 1.18 | 4.70 / 1.49 |



## 11. Recommendation: the lane count for the normative order

**L = 4** (two NEON registers, four independent accumulation chains per reduction; blocks `K = 256`, tree rule across blocks as in `linalg.md`
2, Cholesky panels of 32). Measured reasons (table above): its worst cell is 1.20x of the best L of that cell and its geometric mean is 1.07
over all 22 cells where it exists; **L = 2** has the same geometric mean but loses 1.70x on dot; **L = 8** is 1.03x on dot but 2.75x on
gemv_n at d = 20 in my kernel; **L = 16** is best only on dot (L = 8 is 3% behind, L = 4 7% behind) and its worst cells are 1.9x to 4.7x of the best on
gemv_n, syrk and Cholesky. L = 1 does not exist for dot (the reduction is along the contiguous axis).

What it buys: the three implementations (cancho scalar, a future SIMD cancho, this C) can all produce the same bits for L = 4
because the scalar reference of section 2 is four accumulators and a tree. What it does **not** buy: speed parity of the scalar
code (a scalar cancho with four accumulators per reduction is still scalar; not measured here), and portability of the *optimum*:
I measured only this machine; L = 4 fills one 256-bit register or two 128-bit ones, but I did not run x86.

Changes of bits against today's order (`linalg.md` 2): every dot-shaped kernel (dot, gemv_n rows, gemv_t, syrk, gemm leaf) moves to
the lane rule, and Cholesky moves from "one running value" to "panel sums of 32 subtracted in ascending panel order"; the numpy oracle
(`scripts/oracle_linalg.py`) and `tests/lref.cho` would need the lane rule and the panel rule. That is a decision for the maintainer,
as `linalg.md` 5.7 item 1 already says.

## 12. Caveats and what was not done

1. **Ceiling for a C compiler with hand-written intrinsics, not a cancho prediction.** cancho has no vector type (`linalg.md` 5.6 U1).
2. **My kernels are tuned, not exhaustively.** I tried, measured and kept or dropped: a register-strip `gemv_t` with chunking of the rows (worse than row
   streaming: 1.5 to 3.4 ms at d = 64, n = 1e5, against 1.1 to 1.35 ms for row streaming measured right after; one run each, not saved), software prefetch (slower in the probe), 4-row-per-lane groups in `gemv_t`
   (kept, except where `L * d * 8` is a multiple of 4 KB, where rows alias: falls back to row streaming; this is why N2 to N16 are
   18% to 26% behind N1 at d = 256), packing of the syrk block into 8-column panels for d > 32 (kept: at d = 256, n = 1e5 the preliminary run had N2 0.30 s, N4 0.54 s, N8 0.92 s,
   N16 1.82 s, and after packing they are 0.23, 0.26, 0.26, 0.42 s; the loss had been power-of-two row aliasing in L1; different load). A real expert kernel
   could be faster, so every N column is a lower bound on the intrinsics ceiling, and the cells where N(best) is still above the
   read-probe bound (section 8) have room I did not take.
3. **gemv_t gives no gain over CS** (plain C of the current order compiled by clang, which per `linalg.md` 5.3 already uses 2-lane code where it finds it) in my implementation: 0.90 to 1.02 of CS.
4. **Everything is flagged LOAD>=2.** The probe bandwidth moved 2x between moments. Ratios to numpy are within-session and interleaved;
   comparisons with the copied cancho figures are across sessions.
5. **Cholesky timing includes** copying the lower triangle of A into a padded workspace and leaves L in the workspace (leading
   dimension `dp`); scipy's wrapper also allocates and copies. The Cholesky order here (panels) is **not** the order of `linalg.md`
   2.1, so CS and N compute different bits; both are exact to their own references.
6. syrk d = 256 (and n = 1e6, d = 64) bits are verified on the suite shapes only (section 4). Threads: not measured. Accuracy of the
   result against the true sum: not measured (the order is deterministic; that is all that is claimed).
7. `trsv`, `gemm`, other machines, x86, AVX: not run.

## 13. Reproduce

```
mkdir -p build        # from the root of this repository
clang -O3 -mcpu=native -fno-fast-math -ffp-contract=off -o build/simd scripts/simd/simd.c -lm
build/simd check                                            # section 4 (docs/bench/simd-check.log)
clang -O3 -mcpu=native -o build/bw scripts/simd/bw.c && clang -O3 -mcpu=native -ffp-contract=off -o build/peak scripts/simd/peak.c
python3 scripts/simd/bench_simd.py --gate --out docs/bench/simd-gate.jsonl   # section 6 (about 20 minutes)
python3 scripts/simd/report.py > docs/bench/simd-tables.md
```

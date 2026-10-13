# cancho-ml models: linear regression, logistic regression, k-means (`lib/models`, stage M6 of v1)

**Status: sections 1 to 5 (the gate and the specification) were written before any code of `lib/models`. Section 6 holds the
results; a number is in it only if a script in `scripts/` printed it, with its conditions.** Tags as in `design.md`:
**[choice]** is a decision, **[estimate]** is not measured. Library only: nothing here is wired into the `ml` command line
(`docs/cli.md` still has only `--model constant`).

## 1. What is built

Three estimators on a matrix given as a tensor descriptor (`linalg.tensor`, rank 2, unit column stride) beside its `[float]`:

| estimator | module(s) | what it computes |
|---|---|---|
| linear regression (OLS, optional ridge) | `lib/models/linear/linreg.cho` | normal equations with the package's `syrk` and `gemv_t`, then Cholesky and two triangular solves |
| logistic regression, binary | `lib/models/linear/logreg.cho`, `npass.cho` | Newton, a **fixed** number of iterations, no line search, no stopping rule, optional l2 |
| k-means | `lib/models/kmeans/kmeans.cho`, `kpass.cho` | seeded init (k-means++ style or random), a **fixed** number of Lloyd iterations |
| shared | `lib/models/common.cho` (checks), `score.cho` (the score order), `record.cho` (tensor records of the CML1 model file), `mexp.cho` (`exp`, `log1p`), `par.cho` (the threaded pass of Newton and Lloyd) | |

Features are assumed already transformed (the pipeline stage scales and encodes). Missing values do not exist at this level:
a non-finite cell is refused. The target of the regression is a `[float]`; the target of the classification is a `[float]` that is
exactly `0.0` or `1.0` (the CLI will map its integer column to that).

## 2. The gate (fixed before the code; every line must be able to fail)

* **G-unit** (`tests/p_models.cho`, a program): the tie rule, the empty-cluster rule, the init draws, the score order,
  the record encoder against hand-written bytes, every refusal, each against a definition written separately in the test.
* **G-oracle** (bit for bit, not a tolerance): for each estimator, `scripts/oracle_models.py` performs the **same IEEE operations
  in the same order** as section 3 in numpy (it never calls numpy's BLAS, `exp`, `log1p`, `linalg` or `sum`; `exp`, `log` and
  `log1p` are ports of `std/math.cho`), on data generated from the SplitMix64 streams of `docs/rng.md`, and compares to the
  compiled driver `build/models-check`: every output parameter, **every raw decisive sum of every iteration**, and every loss or
  inertia line. Shapes: row counts on both sides of 256 and 2048 (ragged and aligned blocks and chunks), `d` odd and even, `l2`
  zero and positive, `k` 1 to 9.
* **G-det** the same comparison at **1, 2, 4, 8 and 16 threads**; all outputs, **the raw sums and the loss lines included**
  (the coefficients alone do not always separate a thread-order mistake: `docs/linalg.md` 6.7 of `cancho-ml-armb` found a
  mutant that changed no coefficient at 1,000,000 rows). The gate is void unless the thread-order mutant (section 5) fails it.
* **G-sklearn** (a tolerance, stated now, explained in section 6): on the M0 data (`ml-data`, first `n - n/5` rows to fit, the rest
  to test, as spike S) and on a generated regression problem and a generated clustering problem,
  * linear and ridge, against `sklearn.linear_model.LinearRegression` / `Ridge(alpha=l2, solver="cholesky")`: every coefficient
    `|dw_j| <= 1e-8 * max(1, max_j |w_j|)`; test RMSE relative difference `<= 1e-9`;
  * logistic, against `LogisticRegression(penalty=None | "l2", C = 1 / (l2 * n_train), solver="lbfgs", tol=1e-12, max_iter=10000)`:
    every coefficient `|dw_j| <= 1e-6`, **train objective** `|df| <= 1e-9` (absolute, mean log-loss plus the penalty),
    test AUC `|dAUC| <= max(1e-5, 1 / (n_pos n_neg))` of the test set (the spike S rule: never tighter than one swapped pair);
  * k-means, against `KMeans(init=<the cancho initial centroids>, n_init=1, algorithm="lloyd", max_iter=ITERS, tol=0)`: every
    centroid component `|dc| <= 1e-9 * max(1, |c|)`, inertia relative difference `<= 1e-9`, labels equal on every row but at
    most `1e-5 * n` (a near tie decided by a different summation order of the distance), the count printed.
  These numbers are choices made before the first run. If a run fails, the failure is reported and a change is a dated edit here,
  never a silent loosening. The logistic tolerances are those of spike S for the points that are the same optimum; K is chosen so
  that Newton has converged (`K = 8`, section 3.2), and the converged optimum is also printed against `K + 4`.
* **G-mutants** (`scripts/mutants_models.py`, each an edit of a copy of the sources, run through G-oracle and G-det; **every
  one must be killed**, a survivor is listed with its reason): the iteration count (K - 1, K + 1), the Newton update sign, l2 applied
  to the bias, l2 dropped from the Hessian, the gradient sign, the Hessian product, the tie rule (`<` to `<=`), the empty-cluster
  rule, the seed-derive index (center `j` drawn from stream `j + 1`), the derive kind, the k-means++ cumulative comparison, the
  centroid division, the ridge on the intercept (linear), the block length, the tree combine in thread order (G-det), a thread
  chunk misalignment.
* **G-refusal** every refusal has a rule tag and a fixture (`tests/p_models.cho`, a program): input rows, columns, features, `k`,
  iterations, `l2`, threads, labels, non-finite input, a matrix that is not positive definite (a duplicated column with `l2 = 0`:
  also a case of the oracle suite, `dup=1`), overflowing sums (`ml.fit.non-finite`: features of `1e200`, whose squares overflow).
* **G-fuzz** (`tests/p_models.cho`, a program): random descriptors, extents, strides, offsets, short data, `k`, `iters`, `l2`
  (NaN, inf, negative, huge) through every public entry; no call may trap, an answer is `0` or a code of `ml.refuse`.
* **G-limits** every allocation sized by the input is compared to a limit first (rows, columns, cells from `lib/core/limits.cho`;
  features at most 256 for the linear models, `k * d + k + 1 <= 131072` for k-means, `k <= 1024`); the fixtures include a descriptor
  claiming `2^40` rows that must be refused with nothing allocated.
* **Dated edits to this section, 2026-10-13, made while writing the code, none of them loosening a gate:** (1) the unit, refusal, limit, thread
  and fuzz tests are one program, `tests/p_models.cho`, not `cancho test` files, because the estimators take a thread count and the
  test runner's `main` has no `conc` (as `tests/t_par.cho`); (2) the oracle suite also covers `k` up to 16, the identical-points and
  two-points data (`dup=2`, `dup=3`), the collinear design (`dup=1`) and the refusal of the normal equations; (3) gates added that
  were not in the first list: `mexp.log1p` against `std`'s on a dense grid (a cheaper form with `std`'s bits, section 3.2), the
  `+inf` padding centroids of k-means (section 3.3), a row stride larger than `d` with an offset (a view) giving the bits of the
  contiguous copy, and the mutants that go with them (64 in all, the list in `scripts/mutants_models.py`).
* **G-speed** is a report, not a pass/fail: fit time at 100,000 and 1,000,000 rows by 20, one thread, minimum of 7 interleaved,
  load average at the start and end of each cell, against numpy performing **the same algorithm** (vectorised, BLAS sums: it does
  not reproduce the bits) and against scikit-learn. Whether cancho matches is stated as a ratio, whatever it is.

## 3. Specification (what the oracle and the code must both do)

Common. `n` rows, `d` features, `P = d + 1` parameters (the intercept or bias is **last**: `w[d]`). A float is binary64; no
operation is fused; `a*b` is rounded before it is added. `sum` below always means **the package rule of `docs/linalg.md`
section 2** (revised at `v1-lanes`): cut the rows into blocks of 256; inside a block the terms go to 4 lanes (term `i` of the block to lane
`i mod 4`, each lane ascending from `+0.0`), the block's partial is `(l0 + l1) + (l2 + l3)`, and the partials are combined by the fixed binary
tree by block index (`linalg.order`); a thread is given whole chunks of 8 blocks and the thread count never changes a bit. (Before `v1-lanes`
a block was one ascending chain; every number in section 6 below was measured with that order.) A **term that is not present (a row not in the cluster) is `+0.0`**, which cannot
change a lane that starts at `+0.0` (a lane is a sum, and `x + 0.0 = x` for every `x` but `-0.0`, which no lane can hold: it starts at `+0.0`), so a lane is the ascending sum over the rows of the lane that are present.

**Score order** (used by `predict`, `predict_proba`, the logistic pass, the linear `predict`): `z = w[d]; z = z + x[i][j] * w[j]`
for `j = 0 .. d-1` ascending. It is a left fold over the features, not the block rule: one row's score is never split among
threads, so there is no partition to be independent of **[choice; exception to "all sums use the block rule", stated here]**.

### 3.1 Linear regression

Terms: `G[j][k] = sum_i x_ij x_ik` (`j <= k < d`, `linalg.syrk`), `G[j][d] = sum_i x_ij * 1.0` (`gemv_t` with a vector of ones),
`G[d][d] = float(n)` (`sum of 1.0*1.0`, exact), `b[j] = sum_i x_ij y_i` (`gemv_t`), `b[d] = sum_i y_i * 1.0`. The lower triangle of `G` is the
mirror of the upper. These raw sums are the **decisive sums** of this estimator (`diag` of the fit holds `G` then `b`).
Ridge: `G[j][j] = G[j][j] + l2` for `j < d` only (**the intercept is not penalised**), once, after the sums. The objective is
`||y - Xw - b||^2 + l2 ||w||^2` (scikit-learn's `Ridge(alpha = l2)`, not divided by `n`). Solve `G = L L^T`
(`linalg.chol`, one running value, ascending; pivot `s > 0.0` else `ml.linalg.not-positive-definite`), `L u = b` forward, `L^T w = u`
backward with `k` **descending** (`docs/linalg.md` 2.1). The model is `coef = w[0..d)`, `intercept = w[d]`. A non-finite result is
`ml.fit.non-finite`.

No QR and no SVD: a rank-deficient `X` (with `l2 = 0`) is **refused when the pivot test sees it**, not solved by a minimum-norm fallback
(`ml.linalg.not-positive-definite`, repair "set l2 > 0 or drop the dependent column"). *(Corrected in place at `v1-lanes`: this sentence said
"is refused" without condition. The test is `s > 0.0` on a rounded pivot, so it fires on an exactly dependent column only when the rounding
leaves `s <= 0.0`: an all-zero column always gives exactly `+0.0`, but two equal columns were refused under the old order by rounding luck and
are fitted, with an ill-conditioned result, under the lane order. `tests/p_models.cho` `linreg_refusals` now asks for the zero column and, for
the equal columns, only that one and four threads agree. Use `l2 > 0`.)* The normal equations square the condition
number; this is the cost of the fixed, thread-independent order and is stated in section 6.

### 3.2 Logistic regression (binary), Newton, fixed iterations

`w = 0` (all `P` values, bias included). For iteration `it = 0 .. K-1`, with `a_i = (x_i, 1.0)`:

* per row: `z` by the score order; `az = |z|`; `e = exp(-az)`; `p = 1/(1+e)` if `z >= 0` else `e/(1+e)`; `r = p - y`;
  `s = p * (1 - p)`; `loss = (max(z, 0) + log1p(e)) - y*z` (`max(z,0)` is `z` when `z >= 0` else `0.0`).
  `exp` is the table form of `std/math.cho`'s `exp` (`lib/models/mexp.cho`: the same operations, `2^k` read from an exact table, `std`'s
  own `exp` outside `|x| <= 660`); `log1p` is `std`'s, through `mexp.log1p`, which on `[0, 1]` (the range of `e`) computes `std`'s operations without its
  exponent-splitting code and with both branches of `log` selected, not branched on (a mispredicted branch per row was measurable); it has `std`'s bits, checked on a dense grid by `tests/p_models.cho` and by the oracle, which ports `std`'s `log1p`.
* decisive sums, `W = P + P(P+1)/2 + 1` of them, in this slot layout: `g_a = sum_i r * a_ia` (`a < P`, so `g_d = sum_i r`);
  `h_ab = sum_i (s * a_ia) * a_ib` for `b <= a < P` at index `P + a(a+1)/2 + b` (two rounded products); `ell = sum_i loss` at `W-1`.
* then: `nf = float(n)`; `g_a = sums_a / nf`; `H_ab = sums_ab / nf`; **l2**: `g_a = g_a + l2 * w_a` and `H_aa = H_aa + l2` for `a < d`
  **only** (the bias is not penalised). The objective of the iteration is `f = ell/nf + (0.5*l2) * q`, `q = sum_{j<d} w_j * w_j` ascending
  from `0.0` (the "loss line"; it is the value at the `w` the iteration started from). The penalty is `(l2/2) ||w||^2` on the mean
  log-loss, so scikit-learn's `C` is `1 / (l2 n)`.
* `H = L L^T` (`chol`), `L u = g`, `L^T dl = u` (descending), `w_a = w_a - dl_a`. No damping, no jitter, no line search, no
  convergence test: **the number of iterations is a flag** (`1 .. 1000`). A non-finite `w` is `ml.fit.non-finite`; a pivot that is not
  `> 0.0` is `ml.linalg.not-positive-definite`.
* `K = 8` is the value used by the gates **[choice]**: spike S found `K = 6` converged to `1e-13` on this data; 8 leaves margin for
  other data. The library takes `K` from the caller and has no default.
* The labels must be `0.0` or `1.0` exactly (`ml.label.not-binary`) and both present (`ml.label.single-class`).

`predict_proba`: `p` as above from the score. `predict`: `1.0` if `z > 0.0` else `0.0` (scikit-learn's decision rule; `z == 0` is class 0 **[choice]**).

### 3.3 k-means

`root = rng.derive(seed, 3)` (kind 3 = k-means **[choice; kinds are listed in docs/rng.md]**); the draw for center `j` is
`z_j = rng.derive(root, j)`. `uniform(z, m) = (z >> 1 & 0x7FFF...) % m` (`docs/rng.md`); `u01(z) = float((z >> 11) & (2^53 - 1)) / 2^53`
(exact, in `[0, 1)`).

* **Distance**: `dist2(x, c) = sum over j ascending of t*t, t = x_j - c_j`, a left fold from `+0.0` (one row against one
  centroid is never split, as the score).
* **Init `random`**: center `j` is row `uniform(derive(z_j, a), n)` for the first attempt `a = 0, 1, ...` whose row is not already chosen
  (at most 64 attempts; then the first unchosen row at or after the last attempt's row, cyclically; `k <= n` guarantees one).
* **Init `plusplus`**: center 0 is row `uniform(z_0, n)`. After center `j-1` is chosen, `mind_i = dist2(x_i, c_{j-1})` for `j = 1`,
  else `min(mind_i, dist2(x_i, c_{j-1}))` (`d < m` takes `d`); `total = sum_i mind_i` and `cum_i` are left folds ascending from `0.0`
  (single thread: `O(n k d)`, sequential by design, no partition to vary); `u = u01(z_j) * total`; center `j` is the **lowest `i` with
  `cum_i > u`**; if there is none (`total == 0.0`, or rounding) it is row `uniform(z_j, n)`. A row with `mind_i = 0` is never picked
  by the scan.
* **Assignment**: `label_i = argmin_c dist2(x_i, c_c)` over `c` ascending with a **strict** `<`: the **lowest index wins a tie**.
  *(Implementation, no change of meaning: the centroids are held in groups of four, padded to a multiple of four with centroids whose
  coordinates are all `+inf`, so that eight distances are formed in registers per feature loop; the distance of a finite row to a padding
  centroid is `+inf`, and a strict `<` never selects `+inf`, so a padding centroid is never the nearest even when every real distance
  overflows to `+inf`. `tests/p_models.cho` has the fixture; the mutant that pads with `1e300` is killed by it.)*
* **Lloyd step** (decisive sums, `W = k*d + k + 1`): per cluster `c` and feature `j`, `sum_i [label_i = c] x_ij` (at `c*d + j`); per cluster
  `cnt_c = sum_i [label_i = c] * 1.0` (at `k*d + c`, exact); `inertia = sum_i dist2(x_i, c_label_i)` (at `W-1`); package rule.
  Update: `c_c[j] = sum_cj / float(cnt_c)` if `cnt_c > 0`; **empty-cluster rule: a cluster with `cnt_c == 0` keeps its previous
  centroid unchanged** **[choice]**. Alternatives (scikit-learn moves the farthest point to it) cost another pass and a rule for ties among
  farthest points; the cost of this choice is that a dead centroid stays dead, so `inertia` can be higher than
  scikit-learn's on data where a cluster empties; the number of empty clusters at every iteration is in `diag`, and G-sklearn data
  has none.
* `ITERS` Lloyd steps (`1 .. 1000`), then one final assignment pass at the final centroids (it gives the final labels and
  `inertia_final`, which scikit-learn also reports). The per-iteration inertia is that of the assignment **before** the update.
* `predict`: the labels (the assignment); `transform`: `sqrt(dist2(x_i, c_c))` for every row and cluster (n x k).

### 3.4 Records (CML1, `design.md` 4.1)

A parameter block is a **tensor record**: `u32` dtype, `u32` rank, `rank` x `u64` shape, then the elements row-major, all
little-endian. `dtype 1 = i64` (already used by `tools/ml/mfile.cho`), **`dtype 2 = f64` (new)**. The estimator payload of the `model` section is `u32 kind`
then records: `kind 4` linear: `coef` (rank 1, `d`), `intercept` (rank 1, `1`); `kind 5` logistic: the same two; `kind 6` k-means:
`centroids` (rank 2, `k x d`). Hyperparameters (`l2`, iterations, `k`, init, seed) belong to `meta`'s hyperparameter field. `record.cho` encodes and
decodes (a decoder refuses a dtype, rank, shape or length that does not fit, before allocating). *(Renumbered at integration: this section first said 3, 4, 5, and `tools/ml/mfile.cho` already used 3 for the constant float regression; the registry is `design.md` section 4.1.)* Adding these kinds to `mfile.cho` is
the CLI's next step, not this one's.

### 3.5 Limits and refusals

Checked in this order, before anything sized by the input is allocated: threads `1..64` (`ml.param.threads`); `iters`, `k`, `l2`
(`ml.param.out-of-range`; `l2` must be finite and `>= 0`); `limits.check_matrix` (`ml.input.empty`, `too-many-rows`, `too-many-columns`,
`too-many-cells`); features `<= 256` (linear, logistic) and `k*d + k + 1 <= 131072` (k-means) (`ml.input.too-many-columns`, `ml.param.out-of-range`); the descriptor
against the data (`ml.tensor.*`, unit column stride and row stride `>= d` else `ml.tensor.shape`); non-finite cells of `X` or `y`
(`ml.input.non-finite`); labels (`ml.label.*`); then the fit. New tags (codes 12 to 15 of `lib/core/refuse.cho`): `ml.input.non-finite`,
`ml.label.single-class`, `ml.label.not-binary` (exit 8, as `design.md` section 8) and `ml.fit.non-finite` (exit 8).

## 4. API (all functions answer a refusal code, `0` when done; output buffers are the caller's, checked against their length; on a refusal the contents of `out`, `diag` and `labels` are unspecified)

```
linreg.fit(heap, threads, xt, xd, y, l2, out[d+1], diag[P*P+P])
linreg.predict(xt, xd, w[d+1], out[n])
logreg.fit(heap, threads, xt, xd, y, iters, l2, out[d+1], diag[iters*(W+1)])     diag: per iteration W raw sums then the loss line
logreg.predict_proba(xt, xd, w[d+1], out[n])       logreg.predict(xt, xd, w[d+1], out[n])
kmeans.fit(heap, threads, xt, xd, k, iters, seed, init, out[k*d], diag[(iters+1)*W], labels[n])     W = k*d + k + 1: per Lloyd iteration the raw sums
                                                                                                  (sums, counts, inertia), then the final assignment's
kmeans.predict(xt, xd, centroids, k, labels[n])    kmeans.transform(xt, xd, centroids, k, out[n*k])
record.* : encode_f64 / decode_f64 / model payloads
```

## 5. Mutants and survivors, the thread story

The mutant list is `scripts/mutants_models.py`; its control (the unmutated tree) must pass first. The thread-order mutant adds the
threads' sums in thread order, as a library reduction does, and must fail G-det at more than one thread on the raw sums. Threads: a
thread cannot be given a shared read-only slice (`docs/linalg.md` 5.6 U2), so each owns a copy of its rows, **made at every
iteration**: the threaded fits are for determinism and are slower than one thread on memory-bound shapes; the table in section 6
says by how much.

## 6. Results

Everything here was printed by a script of this repository (`scripts/gate_models.sh`, `scripts/mutants_models.py`,
`scripts/sklearn_models.py`, `scripts/bench_models.py`, `scripts/bench_models_threads.py`) in the session that wrote this section; raw
lines are in `docs/bench/models-*.txt`.

### 6.1 Conditions and data

Apple M4 Max (16 cores), macOS 26.2, cancho `a4572ea` (default LLVM backend, `cancho build --std`), numpy 2.4.6, scipy 1.17.1,
scikit-learn 1.8.0, Python 3.14.5, Accelerate BLAS. Timings: one thread (`VECLIB_MAXIMUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1`,
set by the script, which re-executes itself). **The machine is shared (5 users logged in) and was not quiet:** during the speed tables the
1-minute load average was 2.4 to 3.7 and the 5- and 15-minute averages 6.4 to 7.3 (printed with each table in `docs/bench/models-bench.txt`);
the correctness gates do not depend on it. No number below is a quiet-machine number.

Generated data of the oracle suites (`tools/models-check/main.cho` `gen_data`, `scripts/oracle_models.py` `gen`; `s1, s2, s3 = derive(seed, 1..3)`,
`element(s, i)` as in `docs/rng.md`): `X[i][j] = element(s1, i*d + j)`; `wt[j] = element(s3, j)`; `e_i = element(s2, i)`;
`z_i = 0.0 + sum_j X[i][j] * wt[j]` (a left fold); linear target `y_i = (z_i + 0.5 e_i) + 2.0`; logistic target `y_i = 1.0` iff
`(z_i + 6.0 e_i) + 0.25 > 0.0`; k-means: `g_i = trunc((e_i + 1.0) * 2.5)` in 0..4 and `X[i][j] += float(((g_i*7 + j*3) mod 11) - 5) * 1.5`.
`dup=1` copies column 1 into column 0; `dup=2` makes every row row 0; `dup=3` makes row `i` row `i mod 2`. Real data: the M0 sets of
`ml-data` (`x.bin`, `y.bin`, 20 columns, SHA-256 in each `meta.json`), read by the driver with the bit-exact loader. Timings fit on **all** `n`
rows of the file (not the 80% of spike S); the linear target is `X w + 3 + 0.5 N(0,1)` from numpy's `default_rng(7)`.

### 6.2 Correctness verdicts (`scripts/gate_models.sh`, `scripts/mutants_models.py`; raw lines in `docs/bench/models-gate.txt`, `models-mutants.log`)

| gate | what it checks | result |
|---|---|---|
| G-unit, G-refusal, G-limits, G-fuzz (`tests/p_models.cho`) | 23 tests: `exp` and `log1p` forms against `std` (about 300,000 and 700,000 points and the edges), the score order (a test that fails if the order is reversed), the record bytes against hand-written ones, every truncation and five damaged fields, ordinary fits, **every refusal of every entry** (`ml.input.*`, `ml.label.*`, `ml.param.*`, `ml.tensor.*`, `ml.linalg.not-positive-definite`, `ml.fit.non-finite`), the tie rule, the empty-cluster rule, the padding centroids, `k = n`, the base-rate bias with a huge l2, limit claims of `2^40` rows and of `9223372036854775807 x 9223372036854775807`, views (row stride 5, offset 2) at 1 and 4 threads, the bits at 2, 3, 4, 7, 8, 16 threads, 4,000 fuzz iterations x 8 entries with wild descriptors, `l2` (NaN, inf, negative), `k`, iterations and thread counts | all pass; no call traps |
| G-oracle (bit for bit) | 55 cases at one thread (15 linear, 12 logistic, 28 k-means; 0 to 100,000 rows, `d` 1 to 21, `k` 1 to 16, l2 zero and positive, both inits, the collinear and the identical and two-point data): parameters, every raw decisive sum of every iteration, every loss and inertia line, hashes of `predict`, `predict_proba`, labels and `transform`, against numpy performing the same operations in the same order | **all bits equal** |
| G-det | 13 cases x {1, 2, 4, 8, 16} threads = 65 runs, the same outputs | **all bits equal**; the thread-order mutant fails it |
| G-mutants (67) | see section 6.3 | **65 killed, 2 survivors, both explained** |
| G-sklearn | 39 comparisons, section 6.4 | all pass |

The numpy side of G-oracle is not a transcription of cancho's code: it is vectorised across rows, it builds the block partials by adding
position by position across blocks, and its `exp` and `log1p` are ports of `std/math.cho`; that the two agree to the bit on 55 shapes
(including the identical-points data that exercises every tie) is the evidence that the order of section 3 is what both do.

### 6.3 Mutants

The list is `scripts/mutants_models.py` (67 edits of a copy of the sources, each run through `scripts/gate_models.sh`; a mutant that does not
compile is reported as invalid, not as killed, `scripts/mutlib.py`). **65 killed.** The two survivors:

* `log-sigmoid-boundary` (`z >= 0` to `z > 0` in the sigmoid): equivalent. `z == 0.0` exactly happens only at the first iteration (`w = 0`),
  where `e = 1` and both forms give `0.5` with the same bits, and `pos` is `0.0` either way; no other row has `z` exactly zero.
* `par-models-ragged` (the clamp of the last chunk's block count removed): equivalent. The extra blocks are all-zero partials (`m <= 0`
  rows) and `x + 0.0 = x`, so the tree of the real blocks keeps its bits; the clamp stays so that no block is made of nothing.

Killed among others: the iteration count (`K - 1`, `K + 1`), the update sign and a damped update, l2 on the bias in the gradient and in the
Hessian, l2 dropped from either, the gradient sign, a Hessian product, the bias row, `s = p`, the loss sign, the sigmoid branch, the bias
start of the score, the block length, the prediction threshold (`z >= 0`: killed by `logreg_predict_at_zero`), the `exp` table index;
the k-means tie rule in each lane, group and the single-group path, the empty-cluster rule (`>=`, and a reset to zero), the seed-derive
index, the stream kind, the attempt index of the random init, the k-means++ comparison (`>=`, killed by two-point data on which every
weight vanishes: `dup=3`), the min of the running distances, the scale of `u01`, the first center, the fallback draw, the iteration count,
the final pass, the centroid division, the padding (`1e300` instead of `+inf`, killed by `kmeans_padding_never_wins`), the group index of the
layout, the distance accumulation, the block length, the row stride of a view (three places), the thread-order combine
(`par-thread-order`: the parent adds the threads' sums in thread order; measured here against the thread suite: 0 of 13 cases differ at one thread, 12 of 13 at two, 6 of 13 at 4, 8 and 16: the cases with at most three chunks of 2,048 rows cannot show it, since a left fold of three roots is the tree), a chunk misalignment, the thread copies of rows, labels, targets and weights.

### 6.4 scikit-learn (`scripts/sklearn_models.py`, `docs/bench/models-sklearn.txt`; all 39 comparisons pass)

Worst case over the three sets (1,000, 100,000 and 1,000,000 rows; fit on the first 80%) against the tolerance written in section 2:

| comparison | worst measured | tolerance |
|---|---|---|
| `LinearRegression` coefficients, relative to `max(1, max|w|)` | 5.2e-15 | 1e-8 |
| `LinearRegression` test RMSE, relative | 1.1e-15 | 1e-9 |
| `Ridge(alpha=25, solver="cholesky")` coefficients | 1.2e-14 | 1e-8 |
| `Ridge` test RMSE, relative | 1.2e-15 | 1e-9 |
| `LogisticRegression(penalty=None, lbfgs, tol=1e-12)` coefficients | 5.7e-9 | 1e-6 |
| same, train objective `|df|` | 5.6e-17 | 1e-9 |
| same, test AUC | 0 | `max(1e-5, 1/(n_pos n_neg))` |
| `LogisticRegression(penalty="l2", C = 1/(l2 n))`, `l2 = 0.01`, coefficients | 1.2e-7 | 1e-6 |
| same, train objective | 4.4e-15 | 1e-9 |
| same, test AUC | 1.9e-9 | `max(1e-5, 1/(n_pos n_neg))` |
| `KMeans` (same initial centroids, `lloyd`, `max_iter=10`, `tol=0`) centroids, relative | 1.1e-15 | 1e-9 |
| same, inertia, relative | 2.6e-14 | 1e-9 |
| same, rows with another label | 0 of up to 800,000 (no empty cluster) | 1e-5 n |

Why the logistic coefficients differ by 1e-9 to 1e-7 and the objective by 1e-15: cancho's Newton at `K = 8` and at `K = 12` agree to 2e-16, so its
point is the optimum to rounding; scikit-learn's L-BFGS stops about 1e-7 away in the coefficients even at `tol = 1e-12`, and the objective is flat there
(a quadratic: 1e-7 squared). The tolerances were written before the first run and are looser than the agreement by 3 to 7 orders of magnitude; the bit-level
claim is G-oracle's, not this table's. The linear coefficients agree to rounding although the two algorithms differ (normal equations against
scikit-learn's centred least squares) because the data are well conditioned (standard normal columns); on an ill-conditioned design the normal
equations square the condition number and would not agree, and are refused only when a pivot is not positive (section 3.1).

### 6.5 Speed (`scripts/bench_models.py`; raw lines in `docs/bench/models-bench.txt`)

Fit only, one thread, all `n` rows, 20 features, **minimum of 7 interleaved samples** (spread = max / min in the raw lines, 1.01 to 1.14), load as in 6.1.
`numpy same algorithm` = the same method written with numpy and Accelerate BLAS (Newton K = 6 with the loss; normal equations with Cholesky; Lloyd 10 with expanded
distances), which does not reproduce cancho's bits. Ratios are cancho time / the other time (**above 1 = cancho slower**).

| model | rows | cancho | numpy, same algorithm | cancho / numpy | scikit-learn | cancho / scikit-learn |
|---|---|---|---|---|---|---|
| logistic, Newton K = 6, loss each iteration | 100,000 | 22.8 ms | 15.0 ms | **1.52** | 10.4 ms (lbfgs, defaults) / 16.3 ms (`tol=1e-12`) | **2.19** / **1.40** |
| logistic | 1,000,000 | 236 ms | 170 ms | **1.39** | 126 ms (defaults) / 223 ms (`tol=1e-12`) | **1.87** / **1.06** |
| linear (normal equations) | 100,000 | 2.85 ms | 1.29 ms | **2.21** | 13.2 ms (`LinearRegression`) | 0.22 |
| linear | 1,000,000 | 32.0 ms | 16.1 ms | **1.99** | 182 ms | 0.18 |
| k-means (k = 8, 10 iterations + the final pass, k-means++ included) | 100,000 | 33.8 ms | 62.4 ms | 0.54 | 32.7 ms (same init) | **1.03** |
| k-means | 1,000,000 | 306 ms | 1,148 ms | 0.27 | 330 ms | 0.93 |

**Does it match?** k-means: yes, within the noise (1.03 and 0.93 against scikit-learn; scikit-learn runs the same 10 iterations with the same initial centroids).
Linear regression: it is 4.6 and 5.7 times faster than `LinearRegression` but **not** than the same algorithm in numpy (2.0 to 2.2 times slower): scikit-learn solves the
least squares by an SVD-based `lstsq` with centring, which is the slower algorithm, and cancho's `syrk` is the package's kernel, 3.4 times behind `dsyrk` at 100,000 x 20
(`docs/linalg.md` 5.2). Logistic regression: **it does not match** the default scikit-learn call (1.9 to 2.2 times slower) or numpy's same algorithm (1.4 to 1.5 times slower); it is at parity (1.06) with scikit-learn run to `tol=1e-12`
at 1,000,000 rows, but that is not the comparison the user makes. Pass counts: cancho 6 Newton passes; scikit-learn's default 6 iterations, 7 (100,000) and 8 (1,000,000)
evaluations (`scipy` `nfev`, measured here; each evaluation reads `X` twice), 11 iterations and 12 or 15 evaluations at `tol=1e-12`. Per pass, cancho 39 ms against scikit-learn's 15.7 ms per
evaluation at 1,000,000 rows. Spike S (`cancho-ml-solver` `docs/spikes.md` 11.2) measured its Newton at 3.5 times scikit-learn; this is 1.9, and the data and the loop differ (80% of the rows there, the loss was not summed).

### 6.6 What the tuned loop did to the Newton pass (1,000,000 rows, K = 6, same session, min of 5, load 3.5 / 6.5 / 6.5; all four binaries print the same parameters and sums)

| variant | fit time |
|---|---|
| per-row rank-1 update of the 253 sums (the loop of `cancho-ml-armb`), branchy `row_terms`, `std` `log1p` | 621 ms |
| the same with the branch-free `row_terms` and `mexp.log1p` | 588 ms |
| the block staged: rows read eight at a time (eight chains of `z` and eight `exp`), copied to a padded scratch, `SX = s * A`, the Hessian in 4 x 4 register tiles held across the 256 rows of the block (the tiles of `linalg.syrk`); branchy `row_terms` | 293 ms |
| **staged, branch-free `row_terms`, `mexp.log1p`** (what is in the tree) | **234 ms** |

2.65 times in all (for scale: `cancho-ml-armb` measured its per-row arm C at 466 ms for 800,000 rows, 583 ms per 1,000,000, in another session). Ablations of the staged loop before the branch-free terms
(subtractive: wrong outputs, time kept, 1,000,000 rows, six passes, a different session at load 7 to 8, so only the proportions of the then 305 ms): removing `log1p` (the loss) took 100 ms, removing `exp` 85 ms,
and the Hessian tiles are about 90 of 256 ms (a run with a large l2, so that the zeroed Hessian still factorises).
A two-pass version (all `z` first, then all `exp` in a separate loop) was 336 ms against 245 ms: the interleaving of eight rows is what hides the latency of `exp`. Four rows in flight
instead of eight: 245 against 240 ms (no difference). What the loop cannot do, in this language: the 16 multiply-adds of a tile cannot be fused (no `fma`, by D2) and a reduction over the rows
cannot be vectorised without changing the order; the tiles are vectorised across the 4 x 4 outputs (two lanes), `cancho / C` for `syrk` being 0.96 to 1.00 (`docs/linalg.md` 5.3).
K-means went from 787 ms to 375 ms by grouping the centroids in fours (`kpass`: four distances per feature load, in registers, padded with `+inf` centroids), then to 310 ms with eight chains
in flight (one group alone is bound by the latency of its two vector adds).

### 6.7 Threads (`scripts/bench_models_threads.py`; every output of every run identical to the one-thread run: no cell is marked DIFFERENT)

| model | rows | 1 | 2 | 4 | 8 | 16 threads |
|---|---|---|---|---|---|---|
| logistic | 100,000 | 23 ms | 15 | 10 | 7 | 6 (x3.7) |
| logistic | 1,000,000 | 240 ms | 158 | 102 | 72 | 59 (x4.1) |
| linear | 100,000 | 3 ms | 3 | 3 | 2 | 3 (x1.1) |
| linear | 1,000,000 | 34 ms | 34 | 28 | 25 | 24 (x1.4) |
| k-means | 100,000 | 34 ms | 25 | 19 | 17 | 16 (x2.2) |
| k-means | 1,000,000 | 319 ms | 227 | 161 | 129 | 113 (x2.8) |

Every thread owns a copy of its rows, made at every pass (a thread cannot be given a shared read-only slice, `docs/linalg.md` 5.6 U2), and the copy is in the time; that is why
the speed-up is far from linear and why linear regression (one pass) hardly gains. One cell of an earlier run of this script printed 515 ms for k-means at 1,000,000 rows and one thread
(the rerun above, 319 ms, agrees with the 306 to 320 ms of every other measurement of that cell); the cause was not found and the first run's line is kept in the raw log.

### 6.8 Open questions for the maintainer

1. **`exp` and `log1p` are the compiler's (`std/math.cho`)**: a change to them changes model bytes (`design.md` section 3); the oracle ports them, so it fails on such a change, which is the point.
   The cheaper `mexp.log1p` has `std`'s bits on `[0, 1]` today; if `std` changes it, `tests/p_models.cho` says so.
2. **The convergence count**: K is the caller's. K = 8 agrees with K = 12 to 2e-16 on the M0 data (K = 5 is 1.8e-7 from K = 12 and K = 6 is 2.6e-14, measured on 100,000 rows), and with scikit-learn run to `tol=1e-12` to 6e-9 without l2 and 1.2e-7 with it; on badly scaled or nearly separable data Newton without damping can fail (a pivot that is not
   positive, or a non-finite step: `ml.linalg.not-positive-definite` / `ml.fit.non-finite`). The pipeline's scaling stage is what makes it safe; whether the CLI should refuse unscaled input is not decided here.
3. **Empty clusters keep their centroid** (section 3.3). scikit-learn relocates the farthest point; the cost of this choice is a possible dead centroid. The alternative needs one more pass and a tie rule among farthest points.
4. **Logistic regression is not as fast as scikit-learn's default call** (6.5). Closing it needs a faster Hessian tile (the tile loop is about 35% of the pass and is bound by the vector pipes without `fma`) or fewer passes (K = 6 is already 2.6e-14 from the K = 12 point on this data and K = 5 is 1.8e-7; the library takes K and does not decide it).
5. **Not done**: a second machine (the bytes across Mac and Linux are untested, as in `docs/linalg.md` 5.1); `cancho build --bin models-check` in package mode (needs `cancho install`; `cancho.toml` has the entry, unbuilt); multiclass; QR or SVD; `tools/ml/mfile.cho` does not know the model kinds 3 to 5 or `dtype 2`.

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
| logistic regression, binary | `lib/models/linear/logreg.cho`, `npass.cho`, `lpar.cho` | Newton, a **fixed** number of iterations, no line search, no stopping rule, optional l2 |
| k-means | `lib/models/kmeans/kmeans.cho`, `kpass.cho` | seeded init (k-means++ style or random), a **fixed** number of Lloyd iterations |
| shared | `lib/models/score.cho`, `lib/models/record.cho`, `lib/models/mexp.cho` | the score order, the tensor records of the CML1 model file, `exp`/`log1p` |

Features are assumed already transformed (the pipeline stage scales and encodes). Missing values do not exist at this level:
a non-finite cell is refused. The target of the regression is a `[float]`; the target of the classification is a `[float]` that is
exactly `0.0` or `1.0` (the CLI will map its integer column to that).

## 2. The gate (fixed before the code; every line must be able to fail)

* **G-unit** `cancho test` (and programs for what spawns): the tie rule, the empty-cluster rule, the init draws, the score order,
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
* **G-refusal** every refusal has a rule tag and a fixture (`tests/t_models.cho`, `scripts/oracle_models.py --refusals`, which
  runs the driver): input rows, columns, features, `k`, iterations, `l2`, threads, labels, non-finite input, a matrix that is not
  positive definite (a duplicated column with `l2 = 0`: a fixture), a separable problem whose Newton step overflows
  (`ml.fit.non-finite`: a fixture).
* **G-fuzz** (`tests/t_models.cho`, a program): random descriptors, extents, strides, offsets, short data, `k`, `iters`, `l2`
  (NaN, inf, negative, huge) through every public entry; no call may trap, an answer is `0` or a code of `ml.refuse`.
* **G-limits** every allocation sized by the input is compared to a limit first (rows, columns, cells from `lib/core/limits.cho`;
  features at most 256 for the linear models, `k * d + k + 1 <= 131072` for k-means, `k <= 1024`); the fixtures include a descriptor
  claiming `2^40` rows that must be refused with nothing allocated.
* **G-speed** is a report, not a pass/fail: fit time at 100,000 and 1,000,000 rows by 20, one thread, minimum of 7 interleaved,
  load average at the start and end of each cell, against numpy performing **the same algorithm** (vectorised, BLAS sums: it does
  not reproduce the bits) and against scikit-learn. Whether cancho matches is stated as a ratio, whatever it is.

## 3. Specification (what the oracle and the code must both do)

Common. `n` rows, `d` features, `P = d + 1` parameters (the intercept or bias is **last**: `w[d]`). A float is binary64; no
operation is fused; `a*b` is rounded before it is added. `sum` below always means **the package rule of `docs/linalg.md`
section 2**: cut the rows into blocks of 256, each block's partial is the ascending sum from `+0.0` of the terms of the rows
of the block, the partials are combined by the fixed binary tree by block index (`linalg.order`); a thread is given whole chunks of 8
blocks and the thread count never changes a bit. A **term that is not present (a row not in the cluster) is `+0.0`**, which cannot
change a sum that starts at `+0.0`, so a partial is the ascending sum over the rows that are present.

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

No QR and no SVD: a rank-deficient `X` (with `l2 = 0`) is **refused**, not solved by a minimum-norm fallback
(`ml.linalg.not-positive-definite`, repair "set l2 > 0 or drop the dependent column"). The normal equations square the condition
number; this is the cost of the fixed, thread-independent order and is stated in section 6.

### 3.2 Logistic regression (binary), Newton, fixed iterations

`w = 0` (all `P` values, bias included). For iteration `it = 0 .. K-1`, with `a_i = (x_i, 1.0)`:

* per row: `z` by the score order; `az = |z|`; `e = exp(-az)`; `p = 1/(1+e)` if `z >= 0` else `e/(1+e)`; `r = p - y`;
  `s = p * (1 - p)`; `loss = (max(z, 0) + log1p(e)) - y*z` (`max(z,0)` is `z` when `z >= 0` else `0.0`).
  `exp` is the table form of `std/math.cho`'s `exp` (`lib/models/mexp.cho`: the same operations, `2^k` read from an exact table, `std`'s
  own `exp` outside `|x| <= 660`); `log1p` is `std`'s.
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
then records: `kind 3` linear: `coef` (rank 1, `d`), `intercept` (rank 1, `1`); `kind 4` logistic: the same two; `kind 5` k-means:
`centroids` (rank 2, `k x d`). Hyperparameters (`l2`, iterations, `k`, init, seed) belong to `meta`'s hyperparameter field. `record.cho` encodes and
decodes (a decoder refuses a dtype, rank, shape or length that does not fit, before allocating). Adding these kinds to `mfile.cho` is
the CLI's next step, not this one's.

### 3.5 Limits and refusals

Checked in this order, before anything sized by the input is allocated: threads `1..64` (`ml.param.threads`); `iters`, `k`, `l2`
(`ml.param.out-of-range`; `l2` must be finite and `>= 0`); `limits.check_matrix` (`ml.input.empty`, `too-many-rows`, `too-many-columns`,
`too-many-cells`); features `<= 256` (linear, logistic) and `k*d + k + 1 <= 131072` (k-means) (`ml.input.too-many-columns`, `ml.param.out-of-range`); the descriptor
against the data (`ml.tensor.*`, unit column stride and row stride `>= d` else `ml.tensor.shape`); non-finite cells of `X` or `y`
(`ml.input.non-finite`); labels (`ml.label.*`); then the fit. New tags (codes 12 to 15 of `lib/core/refuse.cho`): `ml.input.non-finite`,
`ml.label.single-class`, `ml.label.not-binary` (exit 8, as `design.md` section 8) and `ml.fit.non-finite` (exit 8).

## 4. API (all functions answer a refusal code, `0` when done; output buffers are the caller's, checked against their length)

```
linreg.fit(heap, threads, xt, xd, y, l2, out[d+1], diag[P*P+P])
linreg.predict(xt, xd, w[d+1], out[n])
logreg.fit(heap, threads, xt, xd, y, iters, l2, out[d+1], diag[iters*(W+1)])     diag: per iteration W raw sums then the loss line
logreg.predict_proba(xt, xd, w[d+1], out[n])       logreg.predict(xt, xd, w[d+1], out[n])
kmeans.fit(heap, threads, xt, xd, k, iters, seed, init, out[k*d], diag[(iters+1)*(W+1) ... ], labels[n])
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

(filled in after the measurements)

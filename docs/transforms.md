# cancho-ml transforms (`lib/transforms`, stage M3 of v1)

**Status: sections 1 to 4 (the gates, the rules, the specification, the records) were written before any code of `lib/transforms`.
Section 6 holds the results; a number or a verdict is in it only if a script in `scripts/` printed it, with its conditions.** Tags as in
`design.md`: **[choice]** is a decision the maintainer can overturn, **[open]** is not decided.

An internal library, **not wired into the CLI** (a later stage does that, once the lane-order change has landed):
`lib/transforms/*.cho`, a driver `tools/pipe-check` that the gates call, tests in `tests/t_transforms*.cho`, scripts in
`scripts/*pipe*`. The pipeline that strings transforms together, the plan text and the model-file payload are `docs/pipeline.md`.

## 0. What a transform is

A **transform** has a **fit** (learn parameters from the *training* rows of a matrix: `ml.matrix.Matrix`, rows times columns doubles
and a bitmap of missing cells) and an **apply** (use the stored parameters on any matrix). Fit never sees the test rows (the pipeline
calls it on the training matrix only; a mutant that fits on all rows must be killed, gate P2). The fitted parameters are **tensor
records** of the model file (dtype tags of `design.md` section 4.1: 1 = i64, 2 = f64), and apply reads **only** them: there is no apply
that takes the plan (`docs/pipeline.md` gate P5).

Six transforms, five plan verbs: `impute` (mean, median, constant), `scale` (`minmax`, `standard`), `encode` (`onehot`, `label`),
`bucket` (equal `width`, `quantile`).

## 1. The gates (written first; each one must be able to fail, section 5 shows it failing)

| id | gate | judged by | a mutant it must kill |
|---|---|---|---|
| X1 | **Hand-computed fixtures.** For every transform, small matrices whose parameters and outputs are written by hand in the test (not computed by the code under test): mean, median (odd and even count), constant, min-max, standard with `ddof` 0 and 1, one-hot and label (sorted categories, unknown category with `ignore` and `refuse`), bucket width and quantile (an exact edge goes to the upper bucket) | `tests/t_transforms.cho` | parameter sign, `ddof`, category order, quantile rule, edge rule |
| X2 | **The written rules for empty, constant, all-missing and missing columns** (section 2.2): each has a fixture with the expected refusal tag or the expected bytes | `tests/t_transforms.cho` | constant divisor `1.0` removed, all-missing check removed, missing passed through as a value |
| X3 | **Differential oracle.** `scripts/oracle_pipe.py` re-implements section 3 in Python (own ordered sum, own sort, own median, own edges) and compares, for generated data (including empty-of-missing, constant, all-missing, one-row and heavily tied columns), the fitted records and the transformed training and test matrices **bit for bit**: fit on train and apply to test equals the reference's fit on train and apply to test. The decisive sums use the package's order through one function (`transforms.dsum`); the oracle has one function (`ordered_sum`) with the same order, and it is the only place the oracle names an order | `oracle_pipe.py --transforms` | any sign, ddof, edge, tie, missing rule |
| X4 | **References.** Against `scikit-learn` `SimpleImputer`, `MinMaxScaler`, `StandardScaler`, `OneHotEncoder`, `OrdinalEncoder`, `KBinsDiscretizer` (the version on the machine, printed): bit equality where the algorithm and the order are the same (imputation with a constant, the median, one-hot and ordinal encoding of the same categories, bucket indices away from an edge), and otherwise a stated bound in ulps with the observed maximum printed and the reason (section 3.7) | `oracle_pipe.py --sklearn` | as X3 |
| X5 | **Records.** The step payload equals bytes written by hand for a fixed input; encode then decode gives the same parameters bit for bit; a decode refuses every truncation and every field out of its bound with a code, before allocating anything sized by the file | `tests/t_transforms.cho`, `t_transforms_fuzz.cho` | field order, dtype tag, shape |
| X6 | **No input reaches a trap; limits before allocation.** Hostile matrices (huge, non-finite values passed as data, columns out of range, categories over the bound, huge `bins`), hostile parameters and corrupted step bytes through every entry: the answer is a code of `transforms.codes`, never a trap; a matrix or an output wider than the cap is refused before the output is allocated | `tests/t_transforms_fuzz.cho` | a removed bound |
| X7 | **Tags.** Every refusal is a code of `lib/transforms/codes.cho` with a tag, an exit code and a table row; every tag is reached by a fixture; `design.md` section 8 and 8.1 and `tools/ml/spec.cho` `pending()` list them | `tests/t_registry.cho`, `scripts/check_registry.py` | tag renamed, exit changed, a tag with no fixture |

Mutants (section 5): at least 25 single edits of `lib/transforms` (and the plan and pipeline files for the P gates), each applied to a
copy of the tree, the gates run on it, killed if one fails. A survivor is explained in `scripts/mutants_pipe.py`.

## 2. Rules

### 2.1 What every transform does with a missing cell **[choice]**

* **Fit** looks at the non-missing cells of a column only, in ascending row order.
* **Apply**: *only `impute` removes a missing cell.* Every other transform passes it through: the output cell is missing (value `0.0`,
  bit set), and for `onehot` the whole block of the column is missing. The estimator decides what a missing cell means (the trees
  take it, the linear models refuse it with `ml.input.non-finite`'s relative, section 8 of `design.md`).
  Alternative considered: a missing cell is an unknown category / a zero. Rejected: it is an invented value.

### 2.2 Empty, constant, all-missing (each has a fixture)

| case | rule | tag |
|---|---|---|
| **empty**: a matrix with no rows | fit refuses | `ml.input.empty` |
| **all-missing column**: every cell of a selected column is missing | fit refuses, for every transform *except* `impute strategy=constant` (which learns nothing from the data). The step and the column are in `detail` | `ml.fit.all-missing` |
| **constant column** (all non-missing cells equal; a column with one non-missing cell is constant) | `scale minmax`: the divisor is `1.0` (range 0), so the training output is exactly `0.0`. `scale standard`: the mean *is* the constant (not the rounded sum over `n`) and the divisor is `1.0`, so the training output is exactly `0.0`. `bucket width`: every edge equals the constant, so every value at or above it is in the last bucket. `bucket quantile`: the same. `onehot`/`label`: one category. `impute`: the constant is the mean and the median | none |
| a **standard** column with `n - ddof < 1` non-missing cells (for example one cell and `ddof=1`) | the variance is taken as `0`, so the divisor is `1.0` | none |
| a computed parameter that is not finite (a sum or a range overflows) | fit refuses | `ml.fit.non-finite` |
| a value at apply time **outside** the training range (min-max, bucket) | not clipped: `scale` extrapolates, `bucket` uses the first or last bucket | none |

"Constant" is decided by **comparing the values** (`min == max`), never by a tolerance on the variance: with a tolerance the rule would
depend on a threshold, and without the comparison the rounded mean of a constant column (`0.1` three times is not exactly `0.1`) would
turn a constant column into noise of size 1. scikit-learn uses a tolerance (`_is_constant_feature`); the difference is an absolute error
of about `n * eps * |c|` on a constant column and is reported in X4.

### 2.3 Categories **[choice]**

* The categories of a column are its **distinct non-missing values**, sorted ascending as numbers (`-0.0` and `0.0` are one category,
  stored as `+0.0`). Equality is `==` on the double. There is no text category in this slice: the reader (`docs/reader.md`) gives
  numbers, and a text column is `--drop`ped.
* **Bound.** `max-categories` (default 64, `1 .. 1024`): more distinct values is `ml.limit.categories` at fit. The sorted copy is
  `n` doubles, `n` already under the matrix caps; the output width (`columns - selected + categories`) is compared with the column cap
  by `matrix.make` before the output is allocated (`ml.input.too-many-columns`).
* **Unknown category at apply** (a non-missing value that is not in the stored set): `unknown=ignore` (default) gives an all-zero block
  (`onehot`) or the code `-1.0` (`label`: no category has it) and **adds one to the unknown count** that `apply` returns and the
  pipeline reports; it never guesses a nearest category. `unknown=refuse` is `ml.input.unknown-category` (step, column and row in
  `detail`). The mode is stored in the step record, so it is part of the model.
* `onehot` output: the block of a column `j` with categories `c_0 < ... < c_{k-1}` is `k` columns, in place of the column (the other
  columns keep their order); `label` output is one column with `float(i)` for category `i`.

### 2.4 Buckets **[choice]**

`bins` is `2 .. 256`; there are `bins - 1` interior edges `e_1 <= ... <= e_{bins-1}`. The bucket of `x` is **the number of edges that are
`<= x`** (an exact edge goes to the upper bucket; the convention of `np.searchsorted(edges, x, side="right")`, which
`KBinsDiscretizer` uses), as a double `0.0 .. bins-1`.

* `width`: `lo = min`, `range = max - lo`, `e_i = lo + (range * float(i)) / float(bins)`.
* `quantile`: the sorted non-missing values `v[0 .. m)`; for `i = 1 .. bins-1`: `num = (m - 1) * i`, `q = num div bins`, `r = num mod bins`;
  `e_i = v[q]` if `r = 0`, else `v[q] + (v[q+1] - v[q]) * (float(r) / float(bins))` (linear interpolation, numpy's default method; the
  form `a + (b - a) t`).
* Each `e_i` is raised to `e_{i-1}` if rounding made it smaller (so the edges are non-decreasing, and the stored record is checked
  to be). A non-finite edge is `ml.fit.non-finite`.

### 2.5 `standard`: the std and the sums **[choice]**

`mean = S1 / float(m)` with `S1` the decisive sum of the `m` non-missing values; `var = S2 / float(m - ddof)` with `S2` the decisive sum of
`(x - mean)^2`; `std = sqrt(var)` (the builtin, one correctly rounded instruction). **`ddof` is `0` (population, the default: numpy,
scikit-learn's `StandardScaler`) or `1` (sample: pandas)**, a key of the plan, stored in the step record. Both sums go through
`transforms.dsum`, which calls the package's reduction (`metrics.sums.reduce`, kinds `val` and `dev`); nothing in `lib/transforms` adds
decisive terms in any other place. The order of that reduction is the package's and may change (the lane-order change in progress);
no code or test here names a lane count, and the oracle's `ordered_sum` is the one line to follow it.

The median sorts the non-missing values with a comparison sort whose result does not depend on its algorithm (a total order on finite
doubles; ties are equal values), so there is no order to specify: `(v[m/2 - 1] + v[m/2]) / 2.0` for even `m`, `v[m/2]` for odd.

## 3. Specification of each transform

Notation: a step selects `k` columns (ascending input indices `c_0 < ... < c_{k-1}`); `m_j` is the count of non-missing cells of column `c_j`.

### 3.1 impute `(strategy: mean | median | constant)`
fit: fill `f_j` = mean (`S1/m_j`), median, or the constant `value` of the plan (a finite double, the same for every selected column).
apply: a missing cell of a selected column becomes `f_j` and is no longer missing; all else is unchanged.

### 3.2 scale `kind=minmax`
fit: `lo_j = min`, `d_j = max - lo_j`; if `d_j == 0.0` then `d_j = 1.0`. apply: `(x - lo_j) / d_j`.

### 3.3 scale `kind=standard`
fit: section 2.5; if the column is constant, `mean_j = the constant`, `d_j = 1.0`; else `d_j = std` (and `d_j = 1.0` if `std == 0.0`).
apply: `(x - mean_j) / d_j`.

### 3.4 encode `kind=onehot | label`
Section 2.3. Lookup is a binary search over the sorted categories with `==` on the double.

### 3.5 bucket `kind=width | quantile`
Section 2.4.

### 3.6 Output shape
All but `onehot` keep the number of columns and rows. Every output cell is a finite double or missing: a cell of the input that is not finite, and a scaled value that overflows, are `ml.input.non-finite` (step, column and row in `detail`), never stored.

### 3.7 Where scikit-learn differs, stated before measuring
* `StandardScaler` computes mean and variance with numpy's pairwise algorithm and treats near-constant columns by tolerance:
  expect differences of a few ulps in `mean`/`std`, and an absolute difference on constant columns.
* `MinMaxScaler` computes `x * scale_ + min_` with `scale_ = 1 / range`: the division becomes a multiplication by a rounded reciprocal,
  so up to a few ulps from `(x - lo) / d`.
* `KBinsDiscretizer` edges come from `np.linspace` / `np.percentile` (different rounding in the interpolation): edges within a few
  ulps; bucket indices equal unless a value lies within those ulps of an edge (the generated data avoids it, and the check says so).
* `SimpleImputer(mean)` uses `np.ma` sums; `median` and `constant` are exact.
The bounds are measured, not assumed: X4 prints the maximum observed per transform.

## 4. The records (the `steps` payload of the model file)

One step = `u32 kind | ints record | floats record`, a record being `u32 dtype | u32 rank | rank x u64 shape | elements`, all little-endian
(`design.md` 4.1). Step kinds (`lib/core/kinds.cho`, `design.md` 4.1): 1 impute, 2 minmax, 3 standard, 4 onehot, 5 label, 6 bucket.

`ints` (dtype 1, rank 1) = `[k, cols_in, c_0 .. c_{k-1}, extra...]`; `floats` (dtype 2):

| kind | `extra` of ints | floats record |
|---|---|---|
| 1 impute | `strategy` (0 mean, 1 median, 2 constant) | rank 1 `[k]` fills |
| 2 minmax | none | rank 2 `[k, 2]` rows `(lo, d)` |
| 3 standard | `ddof` | rank 2 `[k, 2]` rows `(mean, d)` |
| 4 onehot, 5 label | `unknown` (0 ignore, 1 refuse), `n_0 .. n_{k-1}` category counts | rank 1 `[sum n]` the categories, column by column, ascending |
| 6 bucket | `bins`, `method` (0 width, 1 quantile) | rank 2 `[k, bins-1]` the interior edges |

A decode checks, before allocating anything sized by the file: `1 <= k <= cols_in <= 8192`, columns strictly ascending and `< cols_in`,
the extras in range (`ddof` 0/1, `unknown` 0/1, `1 <= n_j <= 1024` and `sum n_j <= 2^20`, `2 <= bins <= 256`), the floats shape equal to
the one the ints imply, every double finite, every divisor `> 0`, categories strictly ascending within a column, edges non-decreasing,
and the output width within the cap. Anything else is `ml.model.malformed` (the step in `detail`).

## 5. Mutants

Listed and run by `scripts/mutants_pipe.py` (section 6 of this file has the table): parameter sign, `ddof` 0/1 swap, unknown-category rule
(ignore vs refuse, zero vs -1), category order (descending, first-seen), quantile rule (`<` vs `<=` edge, `q+1` for `q`, `r/bins`),
fit on test rows (all rows), missing handling (a missing cell as a value, impute not clearing the bit, an all-missing column accepted),
constant-column divisor, median even-count average, a decisive sum added outside `dsum`, a bound removed.

## 6. Results

*(filled in by the stage after the gates ran; see `docs/pipeline.md` section 7 for the pipeline gates)*

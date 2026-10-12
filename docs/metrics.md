# cancho-ml metrics and splits (`lib/metrics`, stage M5 of v1)

**Status: sections 1 to 4 (the gates, the specification, the refusals) were written before any code of `lib/metrics`. Section 5
holds the results; a number or a verdict is in it only if a script in `scripts/` printed it, with its conditions.** Tags as in
`design.md`: **[choice]** is a decision the maintainer can overturn, **[open]** is not decided.

An internal library, **not wired into the CLI** (a later stage does that): `lib/metrics/*.cho`, a driver `tools/metrics-check`
that the gates call, tests in `tests/t_metrics*.cho`, scripts in `scripts/*metrics*`.

## 1. The gates (fixed before the code; each can fail, section 6 shows it failing)

| id | gate | judged by | a mutant it must kill |
|---|---|---|---|
| S1 | **Splits equal an independent implementation of the written spec, bit for bit.** `scripts/oracle_metrics.py` re-implements section 2 in Python (own SplitMix64 on Python ints, own rejection sampling) and compares the whole assignment vector of train/test split, k-fold and stratified k-fold over a grid of (n, k, seed, repeat) including n = k, n = k + 1, sizes straddling every power of two the draw rule touches, and 3 fixed reference vectors written into this document | `oracle_metrics.py --splits` | seed derive index, kind, fold boundary, stratification deal, draw-to-index rule |
| S2 | **Split properties** (not only equality with a copy of itself): every row in exactly one fold; fold sizes are `floor(n/k)` or `ceil(n/k)` with the larger ones first; stratified: per class and fold the count is `floor` or `ceil` of `n_c / k`; the train/test split has exactly `n_test` test rows; the same seed twice gives the same bytes; repeat `r` computed before or after repeat `r'` gives the same bytes (no shared state) | `tests/t_metrics_split.cho` | fold boundary, stratification deal |
| S3 | **Unbiased bounded draw.** Rejection sampling: the number of accepted 63-bit values is a multiple of `m`. Checked by an exact count on small moduli with a reduced-width twin of the rule, and the rule's threshold is compared with the Python threshold for 10,000 moduli up to `2^62` | `tests/t_metrics_split.cho`, `oracle_metrics.py --draw` | threshold off by one, plain modulus |
| M1 | **Counts-based metrics are exact.** accuracy, precision, recall, F1 are one division of two integers each below `2^31`, so one IEEE division is the correctly rounded value. The result must equal Python's `a / b` on the same integers, bit for bit, on generated cases and on the edge cases of section 3 | `oracle_metrics.py --metrics` | wrong positive class, TP/FP/FN mixed up, F1 formula |
| M2 | **ROC-AUC is exact up to one rounding.** Numerator and denominator are exact integers (section 2.4). The result must equal `float(Fraction(num, den))` computed in Python from `scipy.stats.rankdata` midranks (an independent route to the same number), bit for bit, on random, heavily tied, all-tied, single-positive, and extreme-value data | `oracle_metrics.py --auc` | tie rule, midrank off by one, wrong positive class, a dropped tie group |
| M3 | **Decisive sums follow the written order.** For RMSE, MAE, R-squared and log-loss the result must equal numpy applying the same additions in the same order (section 2.2: 4 lanes in 256-term blocks, the level-wise tree over blocks), bit for bit, on sizes on both sides of 4, 256, 1024, 65536 and on wide-magnitude data. For log-loss the terms come from `std.math.log`, which is not Python's `math.log`: the gate takes the terms printed by the driver (checked to be within 2 ulp of the Python terms, with the count of exact ones reported) and checks that the printed result is the ordered sum of those terms | `oracle_metrics.py --sums` | lane count, lane map, tree shape, block length, left fold, clip |
| R1 | **References.** Against `sklearn.metrics` (the version on the machine, printed), and `numpy`/`math.fsum`: bit equality where the order or the rounding is identical (counts-based metrics, AUC if sklearn's trapezoid happens to agree), and otherwise a stated bound, with the observed maximum printed. Cases: ties, a single class, empty input, all-equal predictions, extreme probabilities (0, 1, `1 - 2^-53`, subnormals), constant targets | `oracle_metrics.py --sklearn` | any metric changed by one ulp-level rule (clip bound, denominator) |
| T1 | **Tags.** Every refusal is a code of `lib/metrics/rules.cho` with a `<area>.<name>` tag, an exit code and a summary in the table convention of `tools/ml/spec.cho` (`tag|exit|repairable|summary`), every tag is reached by at least one fixture through the driver with the right tag and exit code, no tag is unreached, and the table is checked against `design.md` section 8 for the tags that exist there | `scripts/gate_metrics.py --tags` | a tag renamed, an exit changed, a fixture removed, an error returned as another code |
| F1 | **No input reaches a trap; limits before allocation.** A fuzz program (`tests/t_metrics_fuzz.cho`) feeds random lengths, NaN, infinities, subnormals, extreme labels (`INT_MIN`, `INT_MAX`), huge `n_test`, `k`, and caps, to every entry; the driver is run on the same classes of input. A trap fails the run. An allocation sized by input is compared with a cap first: the fuzz runs with a cap of 1,000 rows and inputs of 1,001 rows must be refused before any allocation (the heap counter, `heap_used`, is read around the call) | `t_metrics_fuzz.cho`, `gate_metrics.py --fuzz` | a removed cap check, a removed finite check |
| P1 | **Speed, reported not promised.** AUC and log-loss on 1,000,000 predictions against `sklearn.metrics.roc_auc_score`/`log_loss` and against a thin numpy version (`argsort` + `cumsum`; `np.log` + sum). One thread, minimum of 7 interleaved samples, spread and load printed | `scripts/bench_metrics.py` | none: whatever it prints is the verdict, "matches" only if the number says so |

Mutants (section 6): 20 or more source edits of `lib/metrics`, each applied to a copy of the tree, the gates S1, S2, S3, M1 to M3, T1
and F1 run on it, and the mutant is killed if one fails. A survivor is explained in `scripts/mutants_metrics.py` or the gate is fixed.

## 2. Specification

### 2.1 The stream and the bounded draw

A **stream** is named `stream(seed, kind, index) = derive(derive(seed, kind), index)` with `derive` of `docs/rng.md`
(`derive(h, c) = mix((h ^ c) + 0x9E3779B97F4A7C15)`, wrapping). Its state starts at that value `s_0`, and **draw `t`** (`t = 1, 2, ...`)
is `z_t = mix(s_0 + t * 0x9E3779B97F4A7C15)` (wrapping): the SplitMix64 output, `docs/rng.md`.

| kind | used for | index |
|---|---|---|
| 1 | train/test split | the repeat `r >= 0` |
| 2 | k-fold | the repeat `r >= 0` |
| 3 | stratified k-fold | `2 * r + c`, `c` the class ordinal (0 for the smaller label, 1 for the larger) |

`seed`, `r` are non-negative ints; `seed` is an arbitrary int (it is only XORed). Streams are never shared: nothing about a split
depends on another split, on a thread, or on an order of calls.

**Bounded integer** `bounded(stream, m)` for `2 <= m <= 2^62`: let `v` be the next draw's top 63 bits, `v = z >> 1` (logical shift, `0 <= v < 2^63`).
Let `r = 2^63 mod m` (computed as `((2^63 - 1) mod m + 1) mod m`, which fits in an int). **Reject** `v` if `v >= 2^63 - r` (that is,
`v > (2^63 - 1) - r`) and draw again; otherwise the answer is `v mod m`. The number of accepted values is `2^63 - r`, a multiple
of `m`, so the answer is exactly uniform. Every attempt, accepted or not, consumes one draw. `rng.uniform` of `docs/rng.md` (plain
modulus, bias below `m / 2^63`) is **not** used here: the splits are the place where "a documented bias" would be hardest to
defend, and the rejection costs nothing measurable at these sizes. The rejection probability per attempt is `r / 2^63 < m / 2^63`;
for `m <= 2^31` that is below `2^-32`. There is no cap on the number of attempts (a cap would be a second rule); the loop ends with
probability 1, and for a fixed stream it ends after a number of draws that is the same on every machine.

### 2.2 The shuffle

`shuffle(list of length L, stream)` is **Fisher-Yates from the end**: for `i = L - 1` down to `1`: `j = bounded(stream, i + 1)`
(so `0 <= j <= i`), swap `list[i]` and `list[j]`. A list of length 0 or 1 draws nothing.

### 2.3 Splits (all return an assignment vector: element `i` is the group of row `i`)

* **Train/test** `(n, n_test, seed, r)`: needs `n >= 2` and `1 <= n_test <= n - 1`. `perm` = `[0, 1, ..., n-1]` shuffled with
  `stream(seed, 1, r)`. Row `perm[p]` is test for `p < n_test`, train otherwise. Group 1 = test, 0 = train. A test size from a
  fraction `num/den` (`1 <= num < den <= 10^6`) is `n_test = ceil(n * num / den)` in integers (`(n*num + den - 1) / den`), as
  scikit-learn does.
* **K-fold** `(n, k, seed, r)`: needs `2 <= k <= n`. `perm` as above with `stream(seed, 2, r)`. Fold `f` (`0 <= f < k`) holds the
  `size(f) = n div k + (1 if f < n mod k else 0)` entries of `perm` that start at `start(f) = f * (n div k) + min(f, n mod k)`.
  Row `perm[p]` is in fold `f` for `start(f) <= p < start(f) + size(f)`.
* **Stratified k-fold** `(labels, k, seed, r)`: labels are ints with exactly two distinct values (one value is
  `ml.label.single-class`, three or more `ml.label.not-binary`); class ordinal 0 is the smaller value. Needs `2 <= k` and every class
  to have at least `k` rows (else `ml.split.stratum-too-small`; this implies `k <= n`). For each class `c` in ordinal order, `members_c`
  = the rows of that class in ascending row order, shuffled (2.2) with `stream(seed, 3, 2 r + c)`. Let `L` = `members_0` followed by `members_1`.
  **The element at position `t` of `L` is in fold `t mod k`.** Consequences, checked by S2: fold sizes are `floor` or `ceil` of
  `n / k`, and for each class the fold counts are `floor` or `ceil` of `n_c / k` (class 0 starts at fold 0, class 1 at fold `n_0 mod k`). The rule has
  no rounding of fractions, by design.

All sizes are compared with the row cap (default 10,000,000, ceiling `2^30`) **before** the permutation is allocated. Splits do not
spawn threads and have no `conc` effect (the compiler's effect row enforces it), so the thread count cannot change a bit.

### 2.4 Sums: the order of every decisive sum **[choice]**

A decisive sum of terms `t_0 .. t_{n-1}` (`n >= 1`) is cut into **blocks of K = 256** terms (`order.block()`); inside a block the terms go to **4 lanes**:
term `i` of the block (counted from the block start) is added to lane `i mod 4`, each lane ascending from `+0.0`; the block partial is
`(lane0 + lane1) + (lane2 + lane3)`. The block partials are combined by the fixed binary tree of `docs/linalg.md` section 2
(level by level, left operand first, an odd last one carried up; evaluated by the stack of `lib/linalg/order.cho`, which `oracle_linalg.py --tree`
checks equal to the level-wise definition). Products are not fused. This is the lane rule of `docs/simd-ceiling.md` section 2
(L = 4) applied to the metrics. **`lib/linalg/order.cho` of this tree still has one chain per block (L = 1)**; the metrics have
their own lane constant `sums.lanes()`, so that the choice of L for linalg (open, `docs/linalg.md` 5.7) cannot change a metric
silently, and the block length and the tree are the shared ones. If the maintainer chooses L = 4 for the linalg package the two
orders coincide and this note is deleted. For `n = 0` a sum is not made: the metric is undefined (a refusal).

Why not an exact sum **[choice]**: the terms of RMSE and MAE are already rounded (`fl(e*e)`), and those of log-loss carry the error of
`log` (1 ulp, `design.md` section 3); an exact sum of rounded terms is not a more correct metric, only a more expensive one. What
the tree buys is the property that matters here: the same bytes whatever the thread count a later stage chooses, because a
thread can be given whole aligned chunks of blocks (`linalg.md` 2). The cost of the tree against a plain left fold is measured in
section 5 (a number, not an assumption).

### 2.5 Counts-based metrics

Labels are `int`. `pos` is the **positive label**, stated by the caller (the CLI will name it; nothing here guesses "the larger one").
For `y_true` and `y_pred`: positive means equal to `pos`. A binary metric needs the set `{y_true} U {y_pred} U {pos}` to have at most 2
distinct values, else `ml.label.not-binary` (so a wrong `pos` that is in neither array is caught when both labels are present).
`TP` = both positive, `FP` = predicted positive only, `FN` = true positive only, `TN` = neither.

| metric | value | undefined (`ml.metric.undefined`) when |
|---|---|---|
| accuracy | `(rows with y_pred == y_true) / n` (any labels, no binary requirement) | `n = 0` |
| precision | `TP / (TP + FP)` | `TP + FP = 0` (nothing predicted positive) |
| recall | `TP / (TP + FN)` | `TP + FN = 0` (no true positive) |
| F1 | `2 TP / (2 TP + FP + FN)` | `2 TP + FP + FN = 0` |

The numerators and denominators are exact ints (each below the row cap `2^30`, so exactly representable), so each value is one IEEE division,
correctly rounded. scikit-learn returns 0 with a warning in the undefined cases (`zero_division` default); this library refuses
**[choice]**: a fold with no predicted positive must not look like a fold with a bad model.

### 2.6 ROC-AUC by the midrank rule, in integers

`y_true` labels (exactly two distinct values, `pos` one of them; else `ml.label.single-class` or `ml.label.not-binary`), `score` finite floats
(`-0.0` equals `+0.0`). `P` = number of positives, `N` = number of negatives, both `>= 1`. Rank all `n = P + N` scores ascending; a group of
`g` equal scores that occupy ranks `a + 1 .. a + g` gets the **midrank** `a + (g + 1) / 2` for every member (the average of its
ranks). The Mann-Whitney statistic is `U = R+ - P (P + 1) / 2`, `R+` the sum of the midranks of the positives, and

    AUC = U / (P N).

In integers: sort by score; for each distinct score value `v` let `p` and `q` be its positives and negatives and `b` the number of negatives with a score below `v`.
Then `2 U = sum over v of (2 p b + p q)` (a positive beats the `b` negatives below it and ties the `q` equal ones for half a point;
this is the midrank formula regrouped: for a group, the sum of the positives' midranks minus the positives' rank offsets is `p b + p q / 2`), and

    AUC = (2 U) / (2 P N),

both exact ints (`2 U <= 2 P N <= n^2 / 2 <= 2^59`), **one correctly rounded division at the end** (`ratio`, section 2.7). The implementation sorts the
scores of the positives and of the negatives apart (an LSD radix sort on the order-preserving integer key of the float, with `-0.0`
mapped to `+0.0`; no comparison sort, so no data-dependent trap) and walks the two sorted lists together. A tie between a positive and a negative counts
half, by the rule above; a tie between two positives (or two negatives) counts nothing. A float in `{+-inf}` or NaN is `ml.input.non-finite`.

### 2.7 The correctly rounded division of two big ints

`ratio(a, b)` for `0 <= a <= b < 2^62`, `b >= 1`: the double nearest to `a / b` (ties to even), by integer long division, no float
intermediate. `a = 0` gives `+0.0`; `a = b` gives `1.0`. Otherwise generate the quotient bits MSB first (`r = a`; repeat: `r = 2 r`; bit
`= r >= b`; `r = r - b` if the bit is 1) until 53 bits starting at the first 1-bit are known, take the next bit as the round bit and the
non-zero-ness of the remaining `r` (and of any later bit) as the sticky bit, round to nearest even, and scale the 53-bit integer by an exact power of two.
`2 r < 2^63` keeps the loop inside `int`. The Python gate computes `float(Fraction(a, b))`, which is correctly rounded by construction.

### 2.8 Regression metrics and log-loss

`y` and `p` are finite floats of equal length `n >= 1`. Sums are the decisive sums of 2.4.

| metric | value |
|---|---|
| RMSE | `sqrt(SSE / n)`, `SSE = sum (y_i - p_i)^2` with `fl(e*e)`, `e = fl(y_i - p_i)`; `n` as float (exact) |
| MAE | `sum |y_i - p_i| / n` |
| R-squared | `1.0 - SSE / SST`, `SST = sum (y_i - m)^2`, `m = (sum y_i) / n` (both sums decisive, the mean first). If every `y_i` equals `y_0` (a direct comparison, not `SST == 0`, because the computed mean of equal values is not always equal to them) or `SST` is `0.0`: `ml.metric.undefined`. A result that is not finite (an overflow of a square): `ml.metric.undefined` |
| log-loss | `p_i` must be in `[0, 1]` (else `ml.metric.bad-probability`; NaN is `ml.input.non-finite`). With `eps = 2^-52` and `clip(x) = min(max(x, eps), 1 - eps)` (`1 - eps` is exact): the term is `-log(clip(p_i))` for a positive row and `-log(clip(1.0 - p_i))` for a negative row (`1.0 - p_i` rounded, then clipped, as scikit-learn does with the columns `[1 - p, p]`); `std.math.log`; the value is `(sum of terms) / n` |

`ml.metric.undefined` for `n = 0`. Differences from scikit-learn that are choices: undefined cases are refusals (not 0.0 or NaN with a warning); the
clip is the same; the sum order is ours.

## 3. References and what "equal" means (R1)

| metric | reference route | expected |
|---|---|---|
| accuracy, precision, recall, F1 | Python `a / b` on exact ints; and `sklearn.metrics.*_score` | bit equal to Python; to sklearn: measured, stated in section 5 |
| ROC-AUC | `float(Fraction(2U, 2PN))` from `scipy.stats.rankdata` midranks; and `sklearn.metrics.roc_auc_score` | bit equal to the Fraction; sklearn: bound below |
| RMSE, MAE, R-squared | numpy in the order of 2.4 (explicit loops over lanes, level-wise tree); and `sklearn.metrics.*` | bit equal to the ordered numpy; sklearn: bound below |
| log-loss | numpy in the order of 2.4 on the driver's printed terms; and `sklearn.metrics.log_loss`, `math.fsum` of the same terms | bit equal to the ordered sum of the printed terms; sklearn: bound below |

**The bound against sklearn / numpy when the order differs.** Both orders add `n` terms of one sign (squares, absolute values, `-log` of a probability in `[eps, 1-eps]`) in a binary-tree-like order, so
each result carries a relative error of at most `(d + 2) u` against the exact sum of the *same rounded terms*, with `u = 2^-53` and `d` the
depth of the sum (for ours `d <= 3 + ceil(log2(n/256)) + 255`... bounded loosely by the length of the longest chain, 64 + 8 + 1 terms: 4
lanes of 64 terms, 2 adds, up to 31 tree levels). The gate therefore asserts a relative difference of at most `300 u` for sums of one sign, and prints the
observed maximum in ulps of the result. R-squared is `1 - SSE/SST`: the error of the ratio is relative to the ratio, so its bound is absolute, `300 u` times `max(1, SSE/SST)`. For
log-loss the terms also differ by the `log` of two libraries (1 ulp each): the bound adds 2 u. AUC against sklearn's trapezoid over `(fpr, tpr)`: relative `300 u`
(the integers are exact; sklearn divides them once and sums trapezoids). A bound that is exceeded fails the gate; one that is far from
reached is reported as such, not tightened to look good.

## 4. Rule tags (T1)

Registered in `lib/metrics/rules.cho` in the table convention of `tools/ml/spec.cho` (`tag|exit|repairable|summary`, entries separated by `;`), the
exit codes of `design.md` section 8 (arguments 2, a limit or a parse failure 8). `design.md` is **not** edited by this stage (other stages edit it
in parallel); the wiring stage appends these rows to its section 8 and to `spec.cho`'s `extra()`.

| tag | exit | when | in design.md section 8? |
|---|---|---|---|
| `ml.input.empty` | 8 | a split of 0 rows | yes |
| `ml.input.too-many-rows` | 8 | `n` over the cap | yes |
| `ml.input.non-finite` | 8 | NaN or an infinity in a score, target or prediction | yes |
| `ml.label.single-class` | 8 | AUC, or a stratified split, with one class | yes |
| `ml.label.not-binary` | 8 | more than two distinct labels where two are needed | yes |
| `ml.metric.undefined` | 8 | empty input, a constant target for R-squared, a ratio with a zero denominator, a non-finite result | yes |
| `ml.param.out-of-range` | 2 | `k < 2`, `n_test` or a fraction out of range, a cap below 1 or above its ceiling, a negative repeat | yes |
| `ml.metric.length-mismatch` | 8 | two arrays of different lengths | **new** |
| `ml.metric.bad-probability` | 8 | a probability outside `[0, 1]` | **new** |
| `ml.split.empty-fold` | 8 | `k > n`, or a train/test split that would leave a side empty | **new** |
| `ml.split.stratum-too-small` | 8 | a class with fewer than `k` rows | **new** |

## 5. Results

(Filled in by the scripts' output; see below.)

## 6. Mutants

(Filled in below.)

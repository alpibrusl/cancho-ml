# cancho-ml metrics and splits (`lib/metrics`, stage M5 of v1)

**Status: sections 1 to 4 (the gates, the specification, the refusals) were written before any code of `lib/metrics`. Section 5
holds the results; a number or a verdict is in it only if a script in `scripts/` printed it, with its conditions.** Tags as in
`design.md`: **[choice]** is a decision the maintainer can overturn, **[open]** is not decided.

A library, **wired into `ml eval` and `--split` at stage W1** (`docs/wiring.md`; for the constant models `eval` keeps its exact arithmetic of `docs/cli.md`): `lib/metrics/*.cho`, a driver `tools/metrics-check`
that the gates call, tests in `tests/t_metrics*.cho`, scripts in `scripts/*metrics*`.

## 1. The gates (fixed before the code; each can fail, section 6 shows it failing)

| id | gate | judged by | a mutant it must kill |
|---|---|---|---|
| S1 | **Splits equal an independent implementation of the written spec, bit for bit.** `scripts/oracle_metrics.py` re-implements section 2 in Python (own SplitMix64 on Python ints, own rejection sampling) and compares the whole assignment vector of train/test split, k-fold and stratified k-fold over a grid of (n, k, seed, repeat) including n = k, n = k + 1, sizes straddling every power of two the draw rule touches, and 3 fixed reference vectors written into this document | `oracle_metrics.py --splits` | seed derive index, kind, fold boundary, stratification deal, draw-to-index rule |
| S2 | **Split properties** (not only equality with a copy of itself): every row in exactly one fold; fold sizes are `floor(n/k)` or `ceil(n/k)` with the larger ones first; stratified: per class and fold the count is `floor` or `ceil` of `n_c / k`; the train/test split has exactly `n_test` test rows; the same seed twice gives the same bytes; repeat `r` computed before or after repeat `r'` gives the same bytes (no shared state) | `tests/t_metrics_split.cho` | fold boundary, stratification deal |
| S3 | **Unbiased bounded draw.** The accepted 63-bit values are a multiple of `m`: `reject_limit(m) mod m = m - 1` and the values lost are fewer than `m`, for moduli from 2 to `2^62` in growing steps (unit test); the Python implementation of the rule gives the same successive values for 490 (seed, modulus) pairs including moduli above `2^62`, where about a quarter of the attempts are rejected; and two streams constructed with the inverse of the SplitMix64 finaliser put the first draw exactly on the largest accepted value and exactly one above it | `tests/t_metrics_split.cho`, `oracle_metrics.py --draw` | threshold off by one, plain modulus, `<` for `<=` |
| M1 | **Counts-based metrics are exact.** accuracy, precision, recall, F1 are one division of two integers each below `2^31`, so one IEEE division is the correctly rounded value. The result must equal Python's `a / b` on the same integers, bit for bit, on generated cases and on the edge cases of section 3 | `oracle_metrics.py --metrics` | wrong positive class, TP/FP/FN mixed up, F1 formula |
| M2 | **ROC-AUC is exact up to one rounding.** Numerator and denominator are exact integers (section 2.4). The result must equal `float(Fraction(num, den))` computed in Python from `scipy.stats.rankdata` midranks (an independent route to the same number), bit for bit, on random, heavily tied, all-tied, single-positive, and extreme-value data | `oracle_metrics.py --auc` | tie rule, midrank off by one, wrong positive class, a dropped tie group |
| M3 | **Decisive sums follow the written order.** For RMSE, MAE, R-squared and log-loss the result must equal numpy applying the same additions in the same order (section 2.4: 4 lanes in 256-term blocks, the level-wise tree over blocks), bit for bit, on sizes on both sides of 4, 256, 1024, 65536 and on wide-magnitude data. For log-loss the terms come from `std.math.log`, which is not Python's `math.log`: the gate takes the terms printed by the driver (checked to be within 2 ulp of the Python terms, with the count of exact ones reported) and checks that the printed result is the ordered sum of those terms | `oracle_metrics.py --sums` | lane count, lane map, tree shape, block length, left fold, clip |
| R1 | **References.** Against `sklearn.metrics` (the version on the machine, printed), and `numpy`/`math.fsum`: bit equality where the order or the rounding is identical (counts-based metrics, AUC if sklearn's trapezoid happens to agree), and otherwise a stated bound, with the observed maximum printed. Cases: ties, a single class, empty input, all-equal predictions, extreme probabilities (0, 1, `1 - 2^-53`, subnormals), constant targets | `oracle_metrics.py --sklearn` | any metric changed by a one-ulp rule (clip bound, denominator) |
| T1 | **Tags.** Every refusal is a code of `lib/metrics/rules.cho` with a `<area>.<name>` tag, an exit code and a summary in the table convention of `tools/ml/spec.cho` (`tag|exit|repairable|summary`), every tag is reached by at least one fixture through the driver with the right tag and exit code, no tag is unreached, and the table is checked against `design.md` section 8 for the tags that exist there | `scripts/gate_metrics.py --tags` | a tag renamed, an exit changed, a fixture removed, an error returned as another code |
| F1 | **No input reaches a trap; limits before allocation.** A fuzz program (`tests/t_metrics_fuzz.cho`) feeds random lengths, NaN, infinities, subnormals, extreme labels (`INT_MIN`, `INT_MAX`), huge `n`, `n_test`, `k`, caps and repeats, to every entry (it must also reach the value paths, and fails if it does not); the driver is run on the same classes of input (`gate_metrics.py --fuzz`, exit codes only 0, 2, 8, no signal). Every allocation sized by input is preceded by a comparison with the cap: a call with `n` from `2^40` to `INT_MAX` and an output slice of 3 entries must answer `ml.input.too-many-rows` (an allocation of that size, or a trap, would end the test) | `t_metrics_fuzz.cho`, `gate_metrics.py --fuzz` | a removed cap check, a removed finite check |
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
| 4 | train/test split | the repeat `r >= 0` |
| 5 | k-fold | the repeat `r >= 0` |
| 6 | stratified k-fold | `2 * r + c`, `c` the class ordinal (0 for the smaller label, 1 for the larger) |

`seed`, `r` are non-negative ints; `seed` is an arbitrary int (it is only XORed). Streams are never shared: nothing about a split
depends on another split, on a thread, or on an order of calls.

**Bounded integer** `bounded(stream, m)` for `2 <= m <= 2^63 - 1` (the callers never pass more than `2^30`; the full range is tested because only moduli above `2^62` reject often): let `v` be the next draw's top 63 bits, `v = z >> 1` (logical shift, `0 <= v < 2^63`).
Let `r = 2^63 mod m` (computed as `((2^63 - 1) mod m + 1) mod m`, which fits in an int). **Reject** `v` if `v >= 2^63 - r` (that is,
`v > (2^63 - 1) - r`) and draw again; otherwise the answer is `v mod m`. The number of accepted values is `2^63 - r`, a multiple
of `m`, so the answer is exactly uniform. Every attempt, accepted or not, consumes one draw. `rng.uniform` of `docs/rng.md` (plain
modulus, bias below `m / 2^63`) is **not** used here: the splits are the place where "a documented bias" would be hardest to
defend, and the rejection costs nothing measurable at these sizes (section 5.4 does not time the splits). The rejection probability per attempt is `r / 2^63 < m / 2^63`;
for `m <= 2^31` that is below `2^-32`. There is no cap on the number of attempts (a cap would be a second rule); the loop ends with
probability 1, and for a fixed stream it ends after a number of draws that is the same on every machine.

### 2.2 The shuffle

`shuffle(list of length L, stream)` is **Fisher-Yates from the end**: for `i = L - 1` down to `1`: `j = bounded(stream, i + 1)`
(so `0 <= j <= i`), swap `list[i]` and `list[j]`. A list of length 0 or 1 draws nothing.

### 2.3 Splits (all return an assignment vector: element `i` is the group of row `i`)

* **Train/test** `(n, n_test, seed, r)`: needs `n >= 2` and `1 <= n_test <= n - 1`. `perm` = `[0, 1, ..., n-1]` shuffled with
  `stream(seed, 4, r)`. Row `perm[p]` is test for `p < n_test`, train otherwise. Group 1 = test, 0 = train. A test size from a
  fraction `num/den` (`1 <= num < den <= 10^6`) is `n_test = ceil(n * num / den)` in integers (`(n*num + den - 1) / den`), as
  scikit-learn does.
* **K-fold** `(n, k, seed, r)`: needs `2 <= k <= n`. `perm` as above with `stream(seed, 5, r)`. Fold `f` (`0 <= f < k`) holds the
  `size(f) = n div k + (1 if f < n mod k else 0)` entries of `perm` that start at `start(f) = f * (n div k) + min(f, n mod k)`.
  Row `perm[p]` is in fold `f` for `start(f) <= p < start(f) + size(f)`.
* **Stratified k-fold** `(labels, k, seed, r)`: labels are ints with exactly two distinct values (one value is
  `ml.label.single-class`, three or more `ml.label.not-binary`); class ordinal 0 is the smaller value. Needs `2 <= k` and every class
  to have at least `k` rows (else `ml.split.stratum-too-small`; this implies `k <= n`). For each class `c` in ordinal order, `members_c`
  = the rows of that class in ascending row order, shuffled (2.2) with `stream(seed, 6, 2 r + c)`. Let `L` = `members_0` followed by `members_1`.
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
(L = 4), which since `v1-lanes` is the one rule of the whole package: `docs/linalg.md` section 2 defines it and `lib/linalg/order.cho` holds its
constants (`lanes()`, `lane_combine`, `block_partial`); `lib/metrics/sums.cho` no longer has a constant of its own (before `v1-lanes` it had
`sums.lanes()` while `order.cho` had one chain per block, and the two orders differed). The metrics' bits did not change at that merge: the
rule they already followed is the one that was adopted. For `n = 0` a sum is not made: the metric is undefined (a refusal).

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

**The bound against sklearn / numpy when the order differs.** Every sum here adds terms of one sign (squares, absolute values, `-log` of a number in `[eps, 1 - eps]`), so
a result carries a relative error of at most `k u / (1 - k u)` against the exact sum of the same rounded terms, `u = 2^-53`, `k` the largest number of
additions any term passes through. Ours: 63 in its lane (a block has 256 terms, 64 per lane), 2 to combine the lanes, and at most 22 tree
levels for `n <= 2^30` (`2^22` blocks), so `k <= 88`. numpy's pairwise sum: a leaf of 128 terms in 8 accumulators (16 additions), then at most 13 levels for `10^6` terms:
`k <= 32` (from numpy's documented algorithm, blocks of 128 summed in 8 accumulators, not re-read in its source here). The gate therefore asserts a relative difference of at most `(88 + 32 + 10) u = 130 u` for
a sum of one sign (10 u for the division, the square root, the terms' own rounding and, for log-loss, 1 ulp of `log` in each of two libraries). R-squared is `1 - SSE/SST`: both sums carry the
bound, so the bound is absolute, `2 * 130 u * max(1, SSE/SST)`. AUC against sklearn's trapezoid over `(fpr, tpr)`: the counts are exact integers, sklearn divides each once and sums
the trapezoids pairwise: relative `130 u`. A bound that is exceeded fails the gate; the largest gap observed is printed next to it (section 5.2) and is far smaller, which
is reported as such and not used to tighten the bound.

## 4. Rule tags (T1)

Registered in `lib/metrics/rules.cho` in the table convention of `tools/ml/spec.cho` (`tag|exit|repairable|summary`, entries separated by `;`), the
exit codes of `design.md` section 8 (arguments 2, a limit or a parse failure 8). `design.md` was **not** edited by this stage (other stages edited it
in parallel); the integration registered the four new rows in its section 8 and in `spec.cho`'s `pending()` (not `extra()`: no verb called the
library, so no fixture reached them); the wiring stage (W1) moved `ml.split.empty-fold` to `extra()` and `rules()` with a fixture, and `ml.metric.length-mismatch`, `ml.metric.bad-probability` and `ml.split.stratum-too-small` stay in `pending()` because no input reaches them (`docs/wiring.md` section 5). The numeric codes are 201 to 211 (design.md 8.1).

| tag | exit | when | in design.md section 8? |
|---|---|---|---|
| `ml.input.empty` | 8 | a split of 0 rows | yes |
| `ml.input.too-many-rows` | 8 | `n` over the cap | yes |
| `ml.input.non-finite` | 8 | NaN or an infinity in a score, target or prediction | yes |
| `ml.label.single-class` | 8 | AUC, or a stratified split, with one class | yes |
| `ml.label.not-binary` | 8 | more than two distinct labels where two are needed | yes |
| `ml.metric.undefined` | 8 | empty input, a constant target for R-squared, a ratio with a zero denominator, a non-finite result | yes |
| `ml.param.out-of-range` | 2 | `k < 2`, `n_test` or a fraction out of range, a cap below 1 or above its ceiling, a negative repeat | yes |
| `ml.metric.length-mismatch` | 8 | two arrays of different lengths | yes (registered at integration) |
| `ml.metric.bad-probability` | 8 | a probability outside `[0, 1]` | yes (registered at integration) |
| `ml.split.empty-fold` | 8 | `k > n`, or a train/test split that would leave a side empty | yes (registered at integration) |
| `ml.split.stratum-too-small` | 8 | a class with fewer than `k` rows | yes (registered at integration) |
| `ml.label.unseen` | 8 | a true or predicted label is not one of the classes of a multiclass metric (**new**, stage A of `multiclass.md`) | yes |
| `ml.limit.classes` | 8 | more than 64 classes (**new**, stage A of `multiclass.md`) | yes |

## 5. Results

Machine: Mac, arm64 (Darwin 25.2.0, 16 cores), cancho `a4572ea` (default LLVM backend), Python 3.14.5, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.8.0.
Every number below was printed by a script of `scripts/` in this session; `docs/bench/` holds the outputs.

### 5.1 The gates (`sh scripts/gate_metrics.sh`, 20 s)

| gate | what ran | result |
|---|---|---|
| unit tests | 24 tests in `tests/t_metrics.cho` (10), `t_metrics_split.cho` (11), `t_metrics_fuzz.cho` (3) | all pass |
| S1 splits | `oracle_metrics.py --splits`: 2,978 (kind, n, k or n_test, seed, repeat) cases, whole assignment vectors against the independent Python implementation of section 2: 21 sizes from 2 to 4097 (every power-of-two edge), 7 seeds including negative and `-2^63`, repeats 0, 1, 3, 5; stratified with 0/1 labels, with 5/9 and -3/4 labels; 3 reference vectors (below) | bit equal on all |
| S3 draw | `--draw`: 492 sequences of 40 to 60 successive `bounded` calls, moduli from 2 to `2^63 - 1`; 150 of them with moduli above `2^62`, where 3,336 of 12,336 attempts are rejected (so the rejection branch is exercised, not assumed); 2 streams built with the inverse of `mix` so that the first draw is exactly the largest accepted value and exactly one above it | all equal; `reject_limit` satisfies `limit mod m = m - 1` and `top - limit < m` for moduli from 2 to `2^62` in growing steps (unit test) |
| ratio | `--ratio`: 3,066 pairs (random `a <= b < 2^62`, exact ties, quotients just below 1) | bit equal to Python `a / b` |
| M1 counts | `--metrics`: accuracy, precision, recall, F1 on 7 sizes x 2 positive labels, and 12 hand cases (labels other than 0/1, nothing positive) | 68 of 68 bit equal to Python `a / b` |
| M2 AUC | `--auc`: 109 cases (sizes 2 to 20,000; uniform, 16-valued, constant, signed-zero, negative, extreme-probability scores; 3% and 97% positives; both positive labels; extremes up to `+-1.79e308` and `5e-324`) | 109 of 109 bit equal to `float(Fraction)` from scipy midranks |
| M3 sums | `--sums`: 1,033 cases of RMSE, MAE, R-squared at sizes 1 to 70, around every block edge up to 2,051, and up to 100,003, on wide-magnitude data and on data of one magnitude; 9 log-loss cases | 1,033 of 1,033 bit equal to numpy in the written order; log-loss terms: 13,407 of 13,896 bit equal to Python `math.log` terms, the rest within 1.00 ulp |
| T1 tags | `gate_metrics.py --tags`: 11 tags registered, `table()` equal to `tag()`, exit codes equal to `design.md` section 8 where the tag is there, the others marked new here | 11 of 11 reached by 45 fixtures through the driver |
| F1 fuzz | in-language: 4,000 rounds over the nine metric entries (rmse succeeded in at least 1,000, AUC and log-loss in at least 100), 6,000 rounds over the three splits, 20 huge-`n` calls; driver: 400 random invocations | no trap; 85 driver runs succeeded, 315 were refused with a tag; no exit code outside 0, 2, 8 |

**Reference vectors** (also in `oracle_metrics.py`; `stream(42, 4, 0) = f94f852667e9fcae`, which is `derive(derive(42, 4), 0)` with `derive` of `docs/rng.md`):

| call | assignment (element `i` = group of row `i`) |
|---|---|
| k-fold `n = 10, k = 3, seed = 42, repeat = 0` | `0 0 2 2 0 2 1 1 0 1` |
| train/test `n = 10, n_test = 3, seed = 42, repeat = 0` | `0 0 1 0 0 0 0 1 1 0` |
| stratified `labels = 0 0 0 0 0 0 1 1 1 1, k = 3, seed = 7, repeat = 0` | `1 2 0 2 1 0 0 2 1 0` |

### 5.2 Against sklearn (`oracle_metrics.py --sklearn`; the gate bound is `130 u`, section 3)

Cases: n from 2 to 100,000; counts on balanced and rare labels; AUC on uniform, 16-valued (ties), all-equal, signed-zero and negative scores at 50% and 2% positives; regression on
wide-magnitude, informative, uniform and constant predictions; log-loss on uniform, tied, extreme (0, 1, `2^-52`, `1 - 2^-53`, `1e-300`, `5e-324`) and all-equal probabilities.

| metric | cases | bit equal to sklearn | largest relative gap otherwise | bound | order matched |
|---|---|---|---|---|---|
| accuracy | 8 | 8 | 0 | 0 (one division of exact ints) | Python `a / b`: all bit equal |
| precision | 8 | 8 | 0 | 0 | same |
| recall | 8 | 8 | 0 | 0 | same |
| F1 | 8 | 8 | 0 | 0 | same (bit equal in the 8 cases) |
| ROC-AUC | 35 | 29 | 1.98 u | 130 u | `float(Fraction)`: all bit equal |
| RMSE | 20 | 15 | 1.67 u | 130 u | numpy in the written order: all bit equal |
| MAE | 20 | 14 | 2.66 u | 130 u | same |
| R-squared | 20 | 12 | 8.30 u (relative to the value) | `2 * 130 u * max(1, SSE/SST)` absolute | same |
| log-loss | 32 | 10 | 4.57 u | 132 u | the ordered sum of the library's own terms: all bit equal |

**Why equal where it is and bounded where it is not.** The first four are one correctly rounded division of the same two exact integers on both sides, so equality is a theorem, and
the cases confirm it. AUC differs from sklearn only in the last bits because sklearn divides each count once and sums trapezoids; the integers are exact on our side and the
division is the only rounding. The sums differ from numpy's pairwise sum only in the association of the same rounded terms. Log-loss also differs by the `log` of two libraries (`std.math.log`, 1 ulp of the platform `log` as `float-math.md` section 9 says; the
measured gap per term is at most 1.00 ulp, and 96.5% of the terms are bit equal).

**Where this library refuses and sklearn does not** (printed by the same script; these are the choices of section 2.5 and 2.8):

| call | sklearn 1.8.0 | here |
|---|---|---|
| `roc_auc_score`, one class | returns `nan` | `ml.label.single-class` |
| `precision_score`, nothing predicted positive | returns `0.0` (warning) | `ml.metric.undefined` |
| `recall_score`, nothing truly positive | returns `0.0` | `ml.metric.undefined` |
| `f1_score`, nothing positive | returns `0.0` | `ml.metric.undefined` |
| `r2_score`, constant target | `0.0`, or `1.0` for a perfect prediction | `ml.metric.undefined` |
| `accuracy_score`, `root_mean_squared_error`, empty input | raises `ValueError` | `ml.metric.undefined` |
| `log_loss`, probability 1.5 | raises `ValueError` | `ml.metric.bad-probability` |
| `roc_auc_score`, a NaN score | raises `ValueError` | `ml.input.non-finite` |
| `accuracy_score`, lengths 3 and 4 | raises `ValueError` | `ml.metric.length-mismatch` |

### 5.3 What the gates found while they were being written

* **The gate's own harness hid infinities.** The first driver decoded a float from its bits with `fbits.decode`, which answers NaN for an infinity too; no infinity ever reached the library, and the mutant that tests `v != v` instead of `v - v != 0` survived.
  Found by that mutant; the driver now builds infinities, and a unit test and fixtures reach them on every entry.
* **Gaps in the unit and fixture sets** that a first run of 147 mutants (126 killed, 21 survivors) exposed, each closed with a test: a third label equal to 0 slipped past a sloppy "other label" test on four routes; lengths that differ only in the second direction; an output slice longer than `n`; one row of one label in a stratified split; a non-finite prediction (not target) in the regression metrics; an `SSE / SST` that overflows although both are finite.
* **Wide-magnitude data cannot see a change of lane.** With terms spread over `2^-40` (the first sums oracle), one term dominates the sum and a tail term added to lane 3 instead of lane 2 gave the same bits on every case. The oracle now also runs sizes 1 to 70 and every block edge on data of one magnitude.
* **One of my mutants was a no-op** (`auc-layout`, `nn + 0`) and one was equivalent by construction (`b + a + 0.0 * a`): removed and replaced.

### 5.4 Speed (P1): `scripts/bench_metrics.py`, `docs/bench/metrics-1thread.txt`, `metrics-sum-cost.txt`

One thread. Minimum of 7 interleaved samples (one Python call, one cancho process, in turn); the cancho sample times 3 calls of the metric with `clock_ms` (1 ms resolution, the data
generated before the clock starts, the allocation inside the metric included); the Python sample is one call. `n = 1,000,000`, 50% positives for AUC, 30% for log-loss. **Load: the machine was not
quiet: the 1-minute load average was 4.5 to 5.5 during these runs (other users and a virtual machine, not these scripts; it was 8 to 9 right after the mutants, and these were
run after it had come down, not after it reached 1), so every cell is flagged `load >= 2` and none is a quiet-machine number.** Spreads (max / min) are 1.02 to 1.12.

| cell | reference ms | cancho ms | cancho / reference | verdict |
|---|---|---|---|---|
| AUC, uniform scores, `sklearn.roc_auc_score` | 146.59 | 24.00 | 0.16 | faster |
| AUC, uniform scores, thin numpy (`argsort` + `cumsum`, integer trapezoid) | 86.21 | 24.33 | 0.28 | faster |
| AUC, 16 distinct scores, `sklearn.roc_auc_score` | 78.54 | 22.67 | 0.29 | faster |
| AUC, 16 distinct scores, thin numpy | 32.16 | 23.00 | 0.72 | faster |
| log-loss, uniform probabilities, `sklearn.log_loss` | 51.88 | 16.00 | 0.31 | faster |
| log-loss, uniform probabilities, thin numpy (`clip`, `np.log`, `sum`) | 5.15 | 15.67 | 3.04 | **does not match: 3.0x slower** |
| log-loss, 16 distinct values, `sklearn.log_loss` | 56.69 | 16.33 | 0.29 | faster |
| log-loss, 16 distinct values, thin numpy | 5.43 | 16.00 | 2.95 | **does not match: 2.9x slower** |

The AUC is faster than both references in these cells. It sorts the two classes apart with a radix sort (8 passes of 8 bits, constant bytes skipped); what makes `argsort` slower here was not investigated. The skip of a constant byte saves
11 ms (22.33 against 33.33) on 16-valued scores and 0.3 ms on uniform ones (measured with the `auc-skip-none` mutant, minimum of 7). The results are equal to the references to the gaps of section 5.2 (0 relative gap on the AUC and log-loss cells). The log-loss gap to the thin numpy version
is `std.math.log`: the same terms summed by a single left fold take 14.20 ms of the 16.00, so about 14 ns per row is the clip and the `log`; the thin numpy version does the same work in 5 ms in all, and why was not investigated.
**This library is 3x slower than a thin numpy log-loss and 3x faster than the sklearn call;** which of the two a user compares with is theirs to say. A faster `std.math.log` is an upstream matter (`design.md` section 11 asks for a faster `exp` for the same reason).

**Cost of the fixed tree against the alternatives** (`--sum-cost`, same conditions, load 4.8 to 4.9):

| sum of the 1,000,000 log-loss terms | time | error against `math.fsum` of the same terms |
|---|---|---|
| tree: 4 lanes x 256 x fixed tree (the library) | 16.00 ms in the whole metric | 0.0 ulp |
| one left fold (driver-only `logloss_naive`) | 14.20 ms in the whole metric | 413.0 ulp |
| numpy `np.sum` (pairwise) | not timed | 0.0 ulp |

So the tree costs 1.8 ms of 16 (1.13x of the left fold) and has the order that does not depend on the thread count; the exact accumulator was not built, and on this evidence it would buy nothing: the tree is already
within 0 ulp of the exact sum of the rounded terms on this input (one input; the worst case of the tree is the bound of section 3, not 0).

## 6. Mutants (`scripts/mutants_metrics.py`, 5 shards in parallel; the log is `docs/bench/mutants_metrics.log`)

146 one-place edits of `lib/metrics` and of `lib/linalg/order.cho` (the shared tree), each applied to a copy of the tree, with the whole gate of section 1 run on it (the unmutated control passes first):
by file: splits, stream names, draw, shuffle, folds, caps, stratification (44), the counts-based metrics (13), ROC-AUC and its radix sort (30), the correctly rounded division (8), the lane rule and block length (12), the shared tree `lib/linalg/order.cho` (3), RMSE, MAE, R-squared (14), log-loss (15), the registry (7).

**138 killed, 8 survive, each explained, 0 unexplained.** The ones named in the brief: tie rule (`auc-tie-zero`, `auc-tie-one`, `auc-tie-half-pos`), midrank off by one (`auc-below-ties`, `auc-midrank-off-by-one`, `auc-below-minus`), wrong positive class (`auc-wrong-positive`, `auc-positives-count`,
`cls-tp-class`, `ll-label-pos`, `ll-neg-class`), fold boundary (`fold-start-no-extra`, `fold-start-min-max`, `fold-start-always-mod`, `kfold-last-fold`, `kfold-hi-bound`), stratification rounding / deal (`strat-deal-offset`, `strat-deal-per-class`, `strat-deal-blocks`), seed derive index (`seed-derive-index`, `seed-derive-swap`, `seed-derive-single`, `strat-class1-stream`, `strat-class0-stream`): all killed.

| survivor | why it cannot change a result |
|---|---|
| `auc-run-neg`, `auc-run-pos` | `a[ni] <= v` and `a[ni] == v`: `v` is the smaller of the two sorted heads, so the head is never below it |
| `auc-merge-tie-pick` | `<=` against `<` when the heads are equal: either choice gives the same `v` and both runs are consumed |
| `auc-skip-none` | a counting pass over a byte that is the same for every key is a stable copy; the skip only saves time (11 ms on 16-valued scores, section 5.4) |
| `r2-sst-zero` | `SST = 0` gives an infinity or NaN in `SSE / SST`, which the finite test on the result refuses with the same tag |
| `ratio-first-bit` | `r == b` at the first bit happens only for a quotient that is an exact power of two, and the 52 bits that follow round up to the same value (the oracle tests `1 / 2^61`, `2^61 - 1` over `2^61` and the exact-power quotients `2^e + 1` over `2^(53 + e)`) |
| `ratio-one` | `a == b` is a shortcut; the general path also returns `1.0` (the oracle tests `1 / 1`, which with the mutant takes the general path) |
| `sum-tree-level` | stack levels strictly decrease from the bottom, so a top level above the one below never happens (the same survivor as `docs/linalg.md` 5.1 `order-level-eq`) |

## 7. Open questions for the maintainer

1. **The lane count. Settled at `v1-lanes`:** L = 4 for the whole package, one definition (`docs/linalg.md` section 2), `sums.lanes()` is gone and `lib/linalg/order.cho` holds the constants. *(It read: the metrics use 4 lanes inside the 256-term block, `lib/linalg/order.cho` has one chain per block, and `docs/linalg.md` leaves L open; `sums.lanes()` was this module's own constant so that deciding for linalg could not change a metric silently. L = 4 was adopted, and the metrics' bits did not move.)*
2. **Refuse or return 0 for a ratio with a zero denominator** (precision with nothing predicted positive, and the like): refusing is the choice here; sklearn's default is 0.0 with a warning, and a cross-validation loop that hits one bad fold now gets `ml.metric.undefined` for the fold. The wiring stage may want a `--zero-division` flag; it is a decision, not a bug.
3. **Four new tags** (`ml.metric.length-mismatch`, `ml.metric.bad-probability`, `ml.split.empty-fold`, `ml.split.stratum-too-small`) are in `lib/metrics/rules.cho` and not in `design.md` section 8 or `tools/ml/spec.cho`; the wiring stage appends them (the table has the `spec.cho` format). `ml.input.empty` for a split of zero rows against `ml.metric.undefined` for an empty metric is the split I made; say if one tag is preferred.
4. **Multiclass and a stratified split of more than two classes** are refused (`ml.label.not-binary`), as `design.md` section 1 recommends for M2 to M6. The format of the class ordinal (the smaller label is class 0) is in section 2.3.
5. **Memory.** AUC allocates two arrays of `n + 1` ints and 256 counters (16 MB at `n = 10^6`); a split allocates one permutation of `n + 1` ints beside the caller's output. All behind the cap.
6. **No threaded variant.** The order of every sum is the order a threaded version would keep (whole aligned chunks of blocks), but none is built; the 16 ms log-loss is a single-thread number.
7. **Speed against a thin numpy log-loss is not matched** (3.0x slower); AUC is faster than both references. Both under load 4.5 to 5.5, so the ratios are indicative; rerun on a quiet machine before quoting them.

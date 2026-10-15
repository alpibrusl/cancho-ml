# cancho-ml gradient-boosted trees (`lib/models/boost`, issue 36, stage M15)

**Status: sections 1 to 9 are the specification and the gates, written BEFORE any code of this stage (the first commit of the branch holds only this file).
Section 10 (results) is filled in after the gates have run; a number or a verdict is in it only if a script of this repository printed it, with its
conditions.** Tags as in `design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from a document or the code, **[open]** not
decided or not yet checked. A claim found false is corrected in place and marked.

`--model boost` fits additive trees to the gradients and hessians of a loss: squared error for a regression, logistic loss for a binary classification.
It is wired through the module contract (`docs/modules.md`) as a blob model of both tasks, like the tree and the forest (`docs/wiring-trees.md`), with its
own model kind, **8**, and its own library in `lib/models/boost/`. It reuses the quantile bins of `lib/models/tree` (`tree.bins`, `tree.pbins`, the
reader of the bitmap of missing cells, the refusal codes of `tree.deny`) and nothing else of the tree: the growth, the criterion, the leaf value and the
payload are new, because a boosting round fits a Newton step, not a mean or a Gini impurity.

## 1. Scope

| item | what | notes |
|---|---|---|
| regression | squared error `l = (F - y)^2 / 2`, gradient `F - y`, hessian 1 | section 2 |
| binary classification | logistic loss, `p = 1 / (1 + exp(-F))`, gradient `p - y`, hessian `p (1 - p)` | section 2; labels as logistic and the tree: 0.0 and 1.0 inside the library |
| trees | histogram splits on the quantile bins of `tree.bins`, depth-limited, grown depth first in preorder | section 2.4 |
| leaf value | the Newton step `-G / (H + lambda)` times the learning rate | section 2.5 |
| missing cells | accepted; a learned default side per split, the tree's variants 0 and 1 | section 2.4 |
| row sampling | seeded `--subsample`, one stream per round | section 2.3 |
| early stopping | `--early-stopping N`, refused unless `--split` is declared | section 2.7 |
| flags | `--rounds --learning-rate --max-depth --min-leaf --bins --l2 --subsample --early-stopping --max-nodes` and the existing `--task --seed --threads --split --positive` | section 6 |

**Not in this stage:** multiclass (waits for M16, issue 37), column sampling per tree or per node (**[choice]**: not cheap here, a second stream and a
changed histogram loop; `docs/models-boost.md` section 10 lists it as open), categorical splits, sample weights, feature importance (M23 reads the splits),
monotone constraints, a `huber` or absolute loss, dart, a threaded round. The fit of the rounds is sequential (section 7).

## 2. The algorithm (normative; `scripts/oracle_boost.py` implements exactly this text)

Notation: `n` training rows, `d` columns, `B` value bins (`bins`, 2..255), `R` rounds, `D` = `max_depth`, `ML` = `min_leaf`, `L` = `l2`, `eta` = `learning_rate`,
`seed` an int in [0, 2^63). `X[i][f]` is a float64, finite unless the cell is missing. Task 0 is classification (`y` exactly 0.0 or 1.0), task 1 regression
(`y` finite, `|y| <= 1e100`, the tree's bound). SplitMix64 and `derive` are `docs/rng.md`: `derive(h, c) = mix((h ^ c) + 0x9E3779B97F4A7C15)`; a draw advances
the state by `0x9E3779B97F4A7C15` and mixes; `uniform(m) = (z >> 1) mod m`.

### 2.1 Bins

The edges and the bin matrix are those of `docs/models-tree.md` section 2.3, computed once from **all `n` rows** given to the fit (the validation rows of
2.7 included **[choice]**: the bins are a function of the data matrix alone, so they do not depend on the validation split). A missing cell has bin `B`. A
split "at bin `b`" sends a present row left iff `bin <= b`, equivalently `x < e[f][b+1]`; training uses bins, prediction the float comparison with the stored
threshold.

### 2.2 Streams [choice]

Two stream kinds are appended to the registry (`lib/core/kinds.cho`, `docs/rng.md`; the numbers are append-only): **7**, the row sample of a round: state
`derive(derive(seed, 7), round)` with `round` 0-based; **8**, the validation split: state `derive(seed, 8)`. No draw depends on a thread.

### 2.3 Rows and sampling

* The **fit rows** `R0` are all `n` rows in ascending order, or, with early stopping, the rows that remain after the validation rows of 2.7 are taken out,
  in ascending order.
* `K = truncate(subsample * 1000000.0 + 0.5)` (one IEEE multiply, one add, a truncation toward zero), `1 <= K <= 1000000`. With `K = 1000000` the sample of
  every round is `R0` and no draw is made. Otherwise, in round `r`, walking `R0` in ascending order, the `k`-th row is in the sample `S_r` iff the `k`-th
  draw of the round's stream satisfies `uniform(1000000) < K`. `S_r` stays in ascending row order. A sample with fewer than `2 ML` rows is grown as a root leaf
  (2.4); an empty sample gives a leaf of value 0.0 (the Newton step of nothing).
* The **gradients and hessians are computed for every fit row** each round (the scores `F` of the rows outside the sample change too, 2.6), but only the rows
  of `S_r` enter a histogram, a node total or a leaf value.

### 2.4 Base score, gradients, one tree

* **Base score** `F0`. Regression: `Sy / float(m)` with `Sy` the sequential sum of `y` over `R0` in row order (+0.0 first) and `m = |R0|`. Classification:
  with `n1` = positive rows of `R0`, `F0 = log(float(n1) / float(m - n1))` (one division, then `std.math.log`); `n1 = 0` or `n1 = m` is `ml.label.single-class`.
* **Scores** `F[i] = F0` for all `n` rows at the start. In round `r`: regression `g_i = F_i - y_i`, `h_i = 1.0`; classification `p_i = sigmoid(F_i)` and `g_i =
  p_i - y_i`, `h_i = p_i * (1.0 - p_i)`, where `sigmoid` is `models.logreg.proba_of` (`e = exp(-|z|)`; `1 / (1 + e)` for `z >= 0`, else `e / (1 + e)`), the function
  `logistic` already uses.
* **A tree** is grown on `S_r` from the root, depth first, **preorder** (the node's index in the node arrays is its preorder rank; the left subtree is built
  completely before the right sibling), rows kept in ascending order inside a node by a **stable partition**. A node of `m` rows (counted with the sample),
  depth `dep`, holds `G = sum g_i`, `H = sum h_i` in row order from +0.0.
* **Leaf rule.** A node is a leaf iff `dep = D`, or `m < 2 ML`, or no valid candidate has a positive gain, or the tree would pass the node budget (the last
  case is a refusal, not a leaf: `ml.limit.nodes`, as the tree, which never cuts a tree).
* **Candidates**, in the order *feature ascending, bin `b` ascending (`b = 0 .. B-2`), then variant*: variant 0 sends the node's missing rows right, variant 1
  left (evaluated only if the node has at least one missing row in that column). The histogram of a (node, feature) is the count, `sum g` and `sum h` of each of the
  `B + 1` bins in row order from +0.0. Left side: `nl`, `GL`, `HL` are the sequential cumulative sums over bins `0 .. b` (`((0.0 + s_0) + s_1) + ...`), plus the
  missing bin for variant 1; right side `nr = m - nl`, `GR = G - GL`, `HR = H - HL` (separate roundings). **Valid** iff `nl >= ML`, `nr >= ML`, `HL >= HMIN` and
  `HR >= HMIN`, with `HMIN = 1e-3` **[choice, constant: the floor that keeps a Newton step from dividing by a hessian that rounding or a saturated probability
  made zero]**. Score `S = GL*GL / (HL + L) + GR*GR / (HR + L)`: the products and the quotients are separate roundings, no `fma`, then the sum. A candidate replaces
  the best only if `S` is strictly greater (`>`), so among equal scores the lowest feature, lowest bin and variant 0 win (`models-tree.md` 2.4 item 4).
* **Split rule.** The node is split on the best candidate iff `S_best - S_parent > 0.0`, where `S_parent = G*G / (H + L)` (same roundings). **[choice]**: a gain
  that is zero or negative is not taken (unlike the tree, which takes a zero decrease), because a boosting round whose residual is exactly constant must not
  grow nodes from rounding noise. The test is on the float difference and is deterministic; two splits whose true gains differ by less than the rounding can
  be ordered either way, identically at every thread count (the tree's own statement, `models-tree.md` 3.2).
* Rows go to the children as the tree's 2.4 item 5: a present row left iff `bin <= b`, a missing row left iff the winner is variant 1 (stored as `dleft`).

### 2.5 Leaf values and shrinkage

The Newton step of a leaf with totals `G`, `H`: `lv = -G / (H + L)` when `H + L >= HMIN`, else `lv = 0.0` (the negation is `0.0 - G`, so a node with no gradient has the value +0.0, then one division). The **stored value is
`eta * lv`** (one multiplication, rounded once), so the learning rate is applied at fit time and the model file holds the values that predict adds. An internal
node's `val` is its threshold `e[f][b+1]`, as in the tree.

### 2.6 Updating the scores

After the tree of round `r` is built, every one of the `n` rows (fit rows, validation rows, rows outside the sample) walks it **by bins** (a missing bin follows
`dleft`) to a leaf and `F_i = F_i + val` (one addition). Prediction is `acc = F0; for t = 0 .. T-1 in order: acc = acc + val_t`, the same additions in the same
order, so **the score `predict` computes for a training row equals the training score `F_i` of the fit, bit for bit** (a row's bin walk and the threshold
walk agree because `bin <= b` iff `x < e[b+1]`). That identity is a gate (GB3). Regression: the prediction is `acc`. Classification: the probability is `sigmoid(acc)` (`models.logreg.proba_of`) and the class is 1 iff that probability is above 0.5, a tie going to the other class: the rule of the tree and of the forest, so `describe`'s `label_threshold` of 0.5 is true of the printed probability. (A score above 0 so small that its sigmoid rounds to exactly 0.5 is therefore class 0.)

### 2.7 Early stopping [choice]

`early_stopping = P` with `P = 0` off. `P > 0` is **refused unless a held-out part is declared**: the wiring refuses `--early-stopping` (or a plan with `early-stopping`
above 0) without `--split train=F` with `args.conflict`. The `--split` rows are the test rows of the held-out metrics; **early stopping never reads them**: it takes
its own validation rows out of the *train* side, so the reported held-out metrics stay an honest estimate. Rule:

1. `nv = n / 10` (integer division) validation rows, `n >= 10` else `ml.param.out-of-range`; the rows are the first `nv` entries of a partial Fisher-Yates shuffle of
   `0 .. n-1` driven by the stream of kind 8: for `i = 0 .. nv-1`: `j = i + uniform(n - i)`; swap `a[i], a[j]`. The fit rows are the others, in ascending order.
2. After round `r` (`rr = r + 1` rounds done) the validation loss `V_rr` is the sequential sum over the validation rows in ascending row order of the per-row loss,
   divided by `float(nv)`. Regression: `(F_i - y_i) * (F_i - y_i)`. Classification: `max(F,0) - y*F + log1p(exp(-|F|))`, the logistic loss as `models.logreg` writes it
   (`mexp.exp`, `mexp.log1p`).
3. `best` starts at +infinity, `best_round` 0, `since` 0. If `V_rr < best` (strict): `best = V_rr`, `best_round = rr`, `since = 0`; else `since = since + 1`; if
   `since >= P` the fit stops after round `rr`.
4. The model keeps the trees of rounds `1 .. best_round` (the later trees are dropped, so the model is the best round's, not the last). If the rounds ran out
   without a stop, `best_round` still applies.

Deterministic: the validation rows are a function of `(seed, n)`, the loss of the fit rows' trees, the rule of fixed comparisons. The payload records the number
of rounds requested and the number kept (section 5).

## 3. Numeric decisions

### 3.1 The order of every sum, and `sum_order` **[choice]**

Every float sum of this model is a **sequential** left fold from +0.0 in a fixed order that is a function of the algorithm alone: the node totals and the bin
sums in the (ascending, stable) row order of the node; the cumulative sums over bins ascending; the base score over `R0` ascending; the validation loss over the
validation rows ascending; the score of a row over the rounds in order. **No sum is split across threads** (section 7), so the four-lane block order of
`docs/linalg.md` section 2 plays no part and the file would claim a property it does not have if it said `sum_order = 2`. So **kind 8 records `sum_order = 3`**
("sequential, fixed row order"), the value the tree and the forest record for the same reason (`docs/wiring-trees.md` section 4, registered in `lib/core/kinds.cho`
`sum_order_sequential()`), and `kinds.sum_order_for(8)` is 3. The rule of the reader is the one `ml.model.version` already applies: a file of kind 8 that says
another order is refused with `ml.model.version` (`sum_order` and `known = 3` in detail), and a kind 4 or 5 file that says 3 is refused with `known = 2`. The manifest
declares `[sums] sequential` for the files that sum floats (`scripts/admit.py` Q4).

**Why not exact arithmetic [choice].** The tree's Gini decision is exact because counts are integers. Here a gradient is a float (`F - y`, `p - y`, `p (1 - p)`), and an exact sum would need the gradients quantised to a fixed point (as the scaled-integer histograms of some libraries do), which changes the values the model learns. The decision is instead a float comparison whose inputs are sums in a fixed order that the algorithm defines, so the result is the same in every run, at every thread count and in an independent implementation that adds in the same order (the oracle checks it bit for bit); two splits whose true gains differ by less than the rounding can be ordered either way, identically every time (`models-tree.md` 3.2 says the same of its regression criterion).

No histogram subtraction (the sibling's histogram as the parent's minus the child's) is used: it would give sums that differ from the direct sums in the last bits and
would make the result depend on which child is built first. Every histogram is summed directly from its rows. This costs about one extra pass over the node's rows per
level (**[estimate]**; the speed is not a claim of this document).

### 3.2 `exp` and `log` [read]

`sigmoid` and the logistic loss use `models.mexp` (`exp`, `log1p`), whose bits are those of `std/math.cho`, an fdlibm port written in cancho (so the result does not
depend on the platform's libm; `docs/models-linear.md` 3.2). The base score uses `std.math.log` on a positive finite quotient (also the cancho port). The oracle
ports the same functions in Python (`scripts/oracle_models.py` has `exp_c`, `log1p_c`; this stage adds `log_c` for any positive double) and does not call `math.exp`
or `math.log`.

### 3.3 Overflow, non-finite values, the hessian floor

* A regression target is bounded by 1e100 (`ml.input.non-finite` above, the tree's rule), `n <= 2^20`, so `G <= 2^20 * 1e101` and `G*G <= 1e214`: no sum or square
  overflows, and `F` stays within `1e100 * (1 + R eta)`. A classification score is bounded by `R * eta * |lv|` with `|lv| <= m / HMIN`, finite.
* **`ml.fit.non-finite`** (the rule that exists since the logistic fit) is checked after each round on the fit rows' scores: a non-finite score is refused with it, nothing
  is written. **[read]** from the bounds above it is not reachable from a command line: no fixture reaches it, the check is the defence of a bound that someone
  might later widen, and the mutant that removes it is explained as such in section 10 unless a case is found.
* A leaf's `H + L < HMIN` gives value 0.0 (a saturated probability, `p (1 - p)` rounding to zero, and `L = 0`); a candidate whose side has `H < HMIN` is invalid.
  `sigmoid` of a very large `|F|` is exactly 0.0 or 1.0 (`e` underflows), hence `h = 0.0`; nothing divides by it.
* A score that is not finite at **prediction** (impossible by the bounds, a handcrafted file could try) is `ml.predict.non-finite`, the rule of issue 33: the check
  of `wiredata.first_bad_score` covers every wired kind.

## 4. Parameters, limits and refusals

`Params`: `task`, `rounds` (1..10,000), `learning_rate` (finite, `0 < eta <= 1`), `max_depth` (1..30; 31 to 1000 accepted by the flag and refused with
`ml.limit.depth`), `min_leaf` (1..n), `bins` (2..255), `l2` (finite, 0 to 1e100), `subsample` (finite, `0 < s <= 1`, and `K >= 1`), `early_stopping` (0..1,000),
`max_nodes` (per tree), `seed`, `threads`. Every check is made **before any allocation sized by input**, in the order of `tree.params.check`: threads, the matrix caps, the row
bound (`n <= 2^20` **[choice]**: the tree's exactness bound is not needed here, but the wiring's one check of the rows, its message and its fixtures are shared), then each parameter, then the node allocation.

| tag | exit | when |
|---|---|---|
| `ml.param.out-of-range` | 2 | `rounds`, `learning_rate`, `l2`, `subsample`, `early_stopping`, `min_leaf`, `bins` outside the ranges above (a float flag outside its range is checked here: the flag table can only say "0 or more"); `early_stopping > 0` with `n < 10`; a negative seed |
| `ml.limit.depth` | 8 | `max_depth` above 30 |
| `ml.limit.nodes` | 8 | `max_nodes` above 4,194,304; `rounds * cap` above `2^22` (`cap = min(max_nodes, 2^(D+1)-1, 2m-1)`: **the model of all rounds is held in memory and written whole**, so the total is bounded at 4,194,304 nodes, a quarter of the forest's, **[choice]**); a tree that needs more than `max_nodes` nodes while growing (never cut) |
| `ml.input.non-finite` | 8 | a present `X` cell not finite; a regression `y` not finite or above 1e100 |
| `ml.input.too-many-columns` | 8 | more than 4096 columns after the steps, though `--max-columns` allows up to 8192: the reader of the payload refuses more than 4096, and a fit does not write a file `predict` cannot read **[choice]** |
| `ml.label.not-binary`, `ml.label.single-class` | 8 | as the tree; a classification whose fit rows hold one class (after the validation rows are taken out) |
| `ml.fit.non-finite` | 8 | section 3.3 |
| `ml.param.threads` | 2 | threads outside 1..64, or the copies the binning would make above 2^34 bytes (the tree's formula, the same `pbins`) |
| `ml.model.truncated`, `ml.model.malformed`, `ml.model.version` | 8 | damaged files (section 5) |

## 5. The model file: kind 8

The `model` section is `u32 kind = 8`, then tensor records (`u32 dtype` 1 = i64 or 2 = f64, `u32 rank` = 1, `u64` extent, the elements; the tree's framing, all
little-endian, no padding):

| # | dtype | extent | content |
|---|---|---|---|
| 1 | i64 | 10 | header: `task, rounds, d, bins, total_nodes, max_depth, min_leaf, early_stopping, rounds_requested, subsample_k` |
| 2 | f64 | 3 | `base_score, learning_rate, l2` |
| 3 | i64 | rounds+1 | `tree_off`: node offset of each tree, `tree_off[0] = 0`, `tree_off[rounds] = total_nodes` |
| 4 | i64 | N | `feat` (-1 leaf) |
| 5 | i64 | N | `bin` (-1 leaf) |
| 6 | i64 | N | `dleft` (0 or 1; 0 for a leaf) |
| 7 | i64 | N | `left` (tree-local index, preorder: this node's index + 1; -1 leaf) |
| 8 | i64 | N | `right` (-1 leaf) |
| 9 | f64 | N | `val`: the threshold of an internal node, the **shrunk** value `eta * lv` of a leaf |

(`N = total_nodes`.) For a **classification** one more record follows, the one logistic and the tree write: `labels`, i64 [2], the negative and the positive label
(`payload_version` 2 of the wired layout, `docs/wiring-trees.md` section 4). Size: `4 + 96 + 40 + 16 + 8 (rounds + 1) + 6 (16 + 8 N)` (+ 32 for the labels).

**Reader** (`boost.model.read`, wrapped by `decode_blob`): the header's ranges first, with overflow-free comparisons and **before any allocation**: `task` 0 or 1,
`1 <= rounds <= 10,000`, `1 <= d <= 4096`, `2 <= bins <= 255`, `rounds <= total_nodes <= 2^22`, `1 <= max_depth <= 30`, `min_leaf >= 1`, `0 <= early_stopping <= 1,000`,
`rounds <= rounds_requested <= 10,000`, `early_stopping = 0` implies `rounds = rounds_requested`, `1 <= subsample_k <= 1,000,000`; then the byte length must equal the size computed
from `(rounds, total_nodes, task)` (`ml.model.truncated` if shorter, `ml.model.malformed` if longer or any record header wrong); then `base_score` finite, `0 < learning_rate <= 1`,
`l2` finite and `>= 0`, every `val` finite (`fbits.load` refuses NaN and infinities), the offsets strictly increasing from 0 to `total_nodes`, each node's fields in range,
and **each tree a well-formed binary tree in preorder of depth at most `max_depth`**, checked by the tree's one walk with a stack of the right children still to come
(`models-tree.md` section 7: that is all a walk needs to end inside the tree). Every failure of the section is `ml.model.malformed` at the wiring. A file may say anything;
the reader never traps. The task of the plan's model line must be the payload's (a plan that says classification over a regression payload is malformed); the other
hyperparameters of the plan are a description.

**`--max-bytes`.** `fit` compares the length of the file it would write with `--max-bytes` and refuses a longer one with `limit.input-too-large` before writing
(`docs/wiring-trees.md` section 13: shared code, every model), so a file `predict` cannot read is not written. A boosted model of 10,000 rounds of depth 8 can exceed the 16 MiB default;
`--max-bytes` raises it on both sides.

## 6. The surface

* `ml fit --model boost [--task regression|classification] [--rounds N] [--learning-rate X] [--max-depth N] [--min-leaf N] [--bins N] [--l2 X] [--subsample X]
  [--early-stopping N] [--max-nodes N] [--seed N] [--threads N] [--split train=F] [--positive LABEL] [--step TEXT | --plan FILE] ...`. The plan form is
  `model kind=boost task=regression rounds=100 learning-rate=0.1 max-depth=4 min-leaf=10 bins=64 l2=1.0 subsample=1.0 early-stopping=0 max-nodes=4194304` (the
  canonical text lists every key in the order of the manifest). The defaults are those of the manifest **[choice]**: 100 rounds, learning rate 0.1, depth 4, 10 rows per
  leaf, 64 bins, `l2` 1.0, no subsample, no early stopping; the flag table gives no default of its own to a flag several models share when their defaults differ (`max-depth`, `min-leaf`,
  `l2` here), and the help names each model's.
* `--task` chooses the task, default `regression`, never inferred (`docs/wiring-trees.md` section 3). `--positive` names the positive label of a classification.
* `ml predict` prints the label (classification) or the value; `--proba` adds the probability of the positive label (a regression is `args.conflict`). `ml eval` prints the
  metrics of the task. `ml describe` prints the kind (8), the task, the plan (the hyperparameters) and `sum_order` 3, like every planned model, **and an `ensemble` object after `file`: `trees` (the rounds the file holds), `nodes` and `rounds_requested`** (the object is present for kind 8 only, so the answers of the other kinds keep their bytes; `scripts/oracle_describe.py` writes it too). `ml-predict` and `ml-serve` answer with the bytes `ml predict` prints. The fit summary has a `boost` object: `rounds`
  (kept), `rounds_requested`, `nodes`, `learning_rate`, `base_score`, and `best_round` when early stopping ran.
* Introspect, skill, the schemas and the OpenAPI documents follow from the manifest through `scripts/gen_models.py`, `scripts/schemas.py` and the server's documents.
* The admission file is `lib/models/boost/model.toml`: `kind = 8`, `task = "both"`, `shape = "blob"`, `plan_order = 6`, `tier = "supported"` since all twelve checks pass (section 9, section 10).

## 7. Threads **[choice]**

**The rounds are sequential and one thread builds each tree**: a round needs the previous round's scores, and splitting a tree's histograms across threads would
split sums or force a combine order. `--threads N` reaches only the binning (`tree.pbins`: the columns are cut among the threads and the arrays are the same at any
count, `models-tree.md` section 5). The file is therefore the same at every thread count by construction; GB2 shows it at 1, 2, 4, 8, 16 anyway, and the mutants of the
thread mapping are expected to survive only where no byte can depend on the count (explained in section 10). **No claim of a speed-up with threads is made.** A parallel
histogram build is open (section 10).

## 8. Gates (fixed now; each must be able to fail; a gate that cannot is not a gate)

| id | gate | judged by | a defect it must catch |
|---|---|---|---|
| GB1 | **Oracle, bytes.** `scripts/oracle_boost.py` (Python/numpy, no cancho code, `exp`/`log` ported) writes the model **payload** and `scripts/oracle_wire_boost.py` the whole **file** (header, schema, stored steps, string table, meta, trailer) and both equal what `ml` writes, byte for byte, on generated data: regression and classification, missing cells, constant and empty columns, scale/impute/bucket steps, categorical plans, many shapes, 1,000 rows and 100,000 rows, rounds 1 to 50, depth 1 to 8, subsample on and off, early stopping on and off; `predict`, `predict --proba` and every `eval` metric equal the oracle's doubles | `test_wire_boost_oracle.py`, `oracle_wire_boost.py --suite` | a gradient formula, the leaf value, the shrinkage rounding, the default side, the subsample stream, a sum order |
| GB2 | **Threads.** The whole file at `--threads` 1, 2, 4, 8, 16 is one file, both tasks, with missing cells, a split, subsample and early stopping | `test_determinism.py`, `admit.py` Q6 | a count that reaches a byte |
| GB3 | **Scores.** `predict` on the training rows equals the fit's own scores; a one-round model is `base + eta * lv` by hand; a model of `eta` 1 and one round on a regression with `l2` 0 and a single leaf is the mean | `test_wire_boost.py` | the update not applied to a row outside the sample; the shrinkage applied twice |
| GB4 | **scikit-learn, bounds fixed here, section 8.1.** | `scripts/gate_wire_sklearn.py boost` | a bound exceeded; the controls |
| GB5 | **Admission** `scripts/admit.py boost` Q0 to Q11, all twelve, no `--fast`; the six mutant modules are still refused; the control is admitted | `admit.py`, `tests/modules/run.py` | the six mutant modules |
| GB6 | **Fuzz.** Damaged model files (every truncation, every byte flipped with two masks, repaired checksums, seeded multi-byte damage, huge counts such as 2^63 - 1 in every header field and every record header, a rule broken at a time) give a tag and never a trap; argument fuzz on the new flags; hostile data (huge magnitudes, constant targets, duplicated rows, one row, subnormals) | `test_wire_boost_file.py`, `test_fuzz_args.py`, `admit.py` Q7 | a removed bound of the reader |
| GB7 | **Mutants** of the new code and of the wiring (`scripts/mutants_boost.py`, `scripts/mutants_wire_boost.py`): the gradient and hessian formulas, the shrinkage, the leaf value, the default side, the subsample stream, the validation split, the early-stopping rule, the sum order, every bound, the decoder, the flag mapping, the kind number: killed or each survivor explained; run on a copy of the tree; the baseline passes | the scripts | itself |
| GB8 | **Serving.** `ml-predict` equals `ml predict` byte for byte; `describe` equals its Python oracle (`scripts/oracle_describe.py`); `ml-serve` equals `ml predict` byte for byte; the OpenAPI document validates and Schemathesis runs against the server for a boosted model | `test_serving.py`, `test_serve_bytes.py`, `gate_serve_openapi.py`, `gate_serve_schemathesis.py` | a different byte on any door |
| GB9 | **Cross-platform fingerprint.** `scripts/xplat_fp.py` gains boost cases (regression, classification, 1 and 8 threads, a categorical plan) and `docs/bench/xplat_fp.txt` gains lines **appended** after its 88 existing ones; `scripts/xplat_check.py` is extended and its own mutants still refuse; the new lines are reproduced bit for bit on gram (Linux x86-64, `taskset -c 0-6`) | `scripts/xplat_check.py`, a run on gram | a line that differs on another machine |
| GB10 | **Golden.** The golden records of the existing fixtures (`golden_constant.json`, `golden_numeric.json`, `golden_planned.json`, `golden_describe.json` for the models that existed) are unchanged | `test_wire_baseline.py`, `golden_describe.py --check` | a changed byte of an existing model |
| GB11 | **Tables and tags.** Every new flag and rule tag is reached by a fixture; the generated files are fresh; the registry (kinds 8, streams 7 and 8, `sum_order_for`) agrees with `docs/design.md` 4.1 | `test_tables.py`, `gen_models.py --check`, `check_registry.py` | a flag with no fixture |

### 8.1 The scikit-learn comparison: the bounds, fixed before the run [choice]

Data: the shipped train/test files of `docs/examples/data`, the same files to both programs: breast cancer (binary, 426 / 143 rows, 30 features), iris versicolor against
virginica (75 / 25, 4), wine class 0 against the rest (133 / 45, 13), diabetes (regression, 331 / 111, 10). Settings: `ml --model boost --rounds 100 --learning-rate 0.1
--max-depth 3 --min-leaf 10 --bins 64 --l2 1.0 --seed 0` against `HistGradientBoostingClassifier/Regressor(max_iter=100, learning_rate=0.1, max_depth=3,
max_leaf_nodes=None, min_samples_leaf=10, l2_regularization=1.0, max_bins=64, early_stopping=False, random_state=0)`.

**What is aligned [read; each checked against the installed version when the gate runs]:** the loss (squared error, log-loss), the base score (the mean, the log-odds of
the training rate), the Newton leaf `-G / (H + l2)` times the learning rate, the number of rounds, the depth limit, the minimum rows per leaf, the L2 term, the number of
bins, a learned side for missing cells, the minimum hessian for a split (1e-3 in both). **What is not:** the bins (scikit-learn places an edge midway between distinct values
when a column has few of them and takes quantiles of a subsample of at most 200,000 rows otherwise; here an edge is an actual data value at a fixed rank of all the rows, and a
value on an edge goes up), so the candidate thresholds differ and **the trees are not the same trees**; the tie rule; the gain test; the order in which scikit-learn grows
nodes (best first) against the depth-first preorder here (the same tree when no leaf limit applies, but not claimed); the float summation order. **Equality is not claimed
and not tested**; only the quality of the result is compared, on the test files, within bounds:

| quantity | bound |
|---|---|
| test accuracy | within 0.05 absolute of scikit-learn's (a row of the 25-row iris file is 0.04) |
| test ROC-AUC | within 0.03 absolute |
| test log-loss | within 15% relative |
| test RMSE (diabetes) | within 8% relative |

**Controls (must be able to fail):** (C1) `--max-depth 1` against scikit-learn at depth 3 must fall outside at least one bound per task, or the gate reports that it cannot
fail on these files (the tree gate's C1 did stay inside; the lesson is written there, `wiring-trees.md` section 11); (C2) the same model fitted on a **permuted training
target** must fall outside every accuracy and RMSE bound on every dataset. If a bound is exceeded the gate fails and is reported as it is; the bounds are not widened after
the run. The report prints the metrics of both libraries, not only the verdicts. No speed claim against scikit-learn is made unless this document's section 10 prints the
machine, the load and the minimum of N runs.

## 9. Admission and the module contract

* The adapter is `lib/models/boost/adapter.cho` (module `models.adapter_boost`, the four functions of `docs/modules.md` section 4b) over the library files
  `lib/models/boost/{params,grow,model,predict,fit}.cho`; no float arithmetic lives in the adapter. The manifest lists the library files in `files`, and `[sums] sequential`
  names the files that sum floats (Q4).
* Q6 and Q7 fit with each task; Q9 runs `scripts/oracle_wire_boost.py`; Q10 reads `docs/bench/mutants_boost.json` (written by the mutant script, pattern count equal to the live
  count); Q11 runs `scripts/gate_wire_sklearn.py boost`. The tier is `supported` only if all twelve pass at the commit that sets it; until then the manifest says
  `experimental` and section 10 names the failing checks.
* The mutant modules of `tests/modules/` are run again; the control (a twin of `linear` under another name) must be admitted. Kind 8 was the number the control used (`docs/modules.md`
  section 8); this stage gives the number to `boost`, so the control moves to the next free number, a change of `tests/modules/mutants.py` listed among the shared files.

## 10. Open questions and the results

*Open:* (1) column sampling per tree; (2) a parallel histogram build (the per-feature scans of a node are independent; each thread would need its own copy of the bins, the cost the
forest already pays); (4) a three-way early stopping that uses an explicit validation file.

## 10 (continued). Results

Machines: a Mac (arm64, Darwin 25.2.0, Apple M4 Max) and a Linux x86-64 workstation (Ubuntu 26.04.1, Intel Core i7-1260P, `ssh gram`, runs pinned with `taskset -c 0-6`), cancho `a4572ea`, Python 3.14.5 on the Mac and the workstation's virtual environment, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.8.0. Both machines were shared with other jobs (the Mac's load average was 20 to 90 during these runs, the workstation's 1 to 7), so no time here is a claim except where a table says what it is. Every figure was printed by the script or test named.

| gate | evidence |
|---|---|
| GB1 oracle, bytes | `scripts/oracle_wire_boost.py --suite 60` on the Mac: **120 cases, 115 fitted and 5 refusals the oracle predicts, all byte for byte** (the whole file: header, schema, stored steps, payload, labels record, meta, trailer; then `predict`, `predict --proba` on other rows with the columns reversed, and every `eval` metric for both positive labels, as doubles). Covered by the cases: 46 with missing cells left in (8,292 cells), 2,989 splits whose default side is left, 3,528 trees, 44,500 nodes, 72 with a split, 91 with stored steps, 4 with a column that has no cell, 47 with a row sample, 28 with early stopping of which 15 stopped before the rounds ran out. The same suite with 30 cases per task on the workstation: 60 cases, all equal. `--large` on the workstation: 100,000 rows, 2 cases per task, **4 cases equal to the oracle and the file equal at `--threads` 1, 2, 4, 8 and 16**. `tests/conformance/test_wire_boost_oracle.py` runs 30 cases per task, a test of the cases it generates, one that pins the sample boundary (a draw equal to K) and the rounding of K, and one that shows the oracle can fail (a changed sum-order field, learning rate, sample scale). The oracle is written by the same author from the same document, and reuses the transform, split and metric oracles of earlier stages and the ports of `exp`, `log` and `log1p` of `std/math.cho`: it checks the code against the document, not the document against the world. The "categorical plans" of the oracle are an ordinal column one-hot encoded by a step; the text columns (`--cat`, with a string table) are covered by `scripts/gate_st.py` (round trip of the held-out metrics, row order, alignment by name) and by the fingerprint, not by a byte oracle |
| GB2 threads | `test_wire_boost.py`: the model file and the answer at 1, 2, 4, 8 and 16 threads, four configurations (a row sample and a split, a classification, early stopping, bins 255 with a step): one file each. `admit.py` Q6: `--threads` 1 to 16, one file per task (126,037 and 120,987 bytes). **The threads only bin the columns; the rounds are sequential. No speed-up with threads is claimed**: `scripts/bench_boost.py` on the workstation (100,000 rows, 20 columns, 50 rounds, depth 6, minimum of 5): 1.96 s at 1 thread and 1.90 s at 8 for a regression, 1.94 s and 2.15 s for a classification, whole `ml fit` processes with the CSV read |
| GB3 scores | `test_wire_boost.py`: a round of a known dataset equals the base plus the shrunk Newton step by hand (5.0 - 0.5 * 5.0 = 2.5 and 7.5; with `l2` 1 and a learning rate of 1: 5 -/+ 150 / 31); a constant feature leaves the mean; a perfectly separable classification is fitted at `l2` 0 and 1 without a refusal; the default side is learned from the training rows that miss. The identity "the score of a training row is the fit's own" is not tested from outside (the fit's scores are not printed); it follows from the same additions in the same order, and the oracle, which predicts with the stored values, equals the binary |
| GB4 scikit-learn | `scripts/gate_wire_sklearn_boost.py`, bounds of section 8.1, 10 comparisons, **all PASS**: breast cancer accuracy 0.9580 against 0.9580 (distance 0), ROC-AUC 0.9828 against 0.9868 (0.0040, bound 0.03), log-loss 0.1482 against 0.1495 (0.8%, bound 15%); iris accuracy 1.0000 against 1.0000, ROC-AUC 1.0000 against 1.0000, log-loss 0.0344 against 0.0344; wine accuracy 0.9778 against 0.9778, ROC-AUC 0.9956 against 0.9956, log-loss 0.0693 against 0.0667 (3.9%); diabetes RMSE 61.38 against 63.40 (3.2%, bound 8%). Controls: C1 (`--max-depth 1`) is outside a bound on iris and wine (classification) and on diabetes (regression), so the gate can fail; C2 (the train target permuted) is outside the accuracy or RMSE bound on all four data sets. What is aligned and what is not is in section 8.1: the trees are not the same trees, and equality is not claimed. The small test files make the accuracy bounds coarse (one row of the 25-row iris file is 0.04) |
| GB5 admission | `scripts/admit.py boost`, no `--fast`, on the workstation at the final code: **ADMITTED, 12 of 12 PASS** (Q6 one file at `--threads` 1 to 16; Q7 9,072 damaged files, none trapped, none accepted with a wrong checksum, 9,043 refused with a tag; Q9 60 oracle cases; Q10 175 mutant patterns in 2 suites; Q11 10 comparisons). On the Mac `--fast` (Q0 to Q8): 9 PASS, 3 skipped by the flag. `tests/modules/run.py`: the control (kind 9) admitted and the six mutant modules refused on the check each was written for (`mutant-kind-reuse` also fails Q1) |
| GB6 fuzz | `test_wire_boost_file.py` (17 tests) on three files (a classification with steps, a regression with a row sample, an early-stopped regression; 3,167, 3,370 and 4,339 bytes): **10,876 truncations, 21,752 flipped bytes (2 masks), 3,000 seeded damages** (1,500 of any section with the checksum repaired, 1,500 of whole words of the model section at boundary values such as 2^63 - 1), the header values, the floats, the record headers, the offsets, every node field, the labels record, the length, the plan/payload task and counts that wrap the size computation (rounds + 2^61, total + 2^60 with the record extents edited to match), each rule one at a time: none traps, none with a wrong checksum is accepted, each rule is `ml.model.malformed`. `test_fuzz_args.py`: every flag of the boosted model in every mistake (5 bases, 17 flags, 31 values, two spellings) and 1,250 seeded vectors. `test_wire_boost.py`: hostile data (one row, 1e300 magnitudes, a constant target, duplicated rows, a target above 1e100), more than 4,096 columns, the exact edge of the total of nodes (8,192 rounds of 512 nodes allowed, 8,193 refused), the boundary of ten training rows for early stopping, the 2^20 rows bound. It found no trap in the library |
| GB7 mutants | `scripts/mutants_boost.py` (151 patterns) and `scripts/mutants_wire_boost.py` (24), run in slices on a copy of the tree on the workstation (`docs/bench/mutants_boost.log`, `.json`): **175 patterns, 153 killed, 22 survivors, each explained, 0 unexplained**, and the unchanged copy passes the whole conformance suite (the baseline of each suite, `PASSED` on the workstation). **The first run left 9 mutants alive that had no reason**; tests were added for them (the sample boundary and the rounding of K, the wrapped counts, early stopping with ten training rows, the exact total of nodes, the summary and describe fields, a lowered max-depth in the header), seven then died, and two (the reader's ceilings on rounds and nodes) are explained as the adapter's identical check. The explained survivors are: equivalents (a threads count that changes no byte, a subsample of 1.0 that skips draws it would never fail, a validation loss scaled by a constant, a node pretest that is an optimisation), second lines of defence behind another check of the same bound (the reader behind the adapter and the length, the label check behind the wiring's, the class and probability guards), and bounds no input reaches (the non-finite check of a fit, the copies ceiling at 2.3e8 cells, a depth of exactly 30). |
| GB8 serving | `test_serving`, `test_serve`, `test_serve_bytes`, `test_serve_json`, `test_serve_fuzz`, `test_describe`, `test_nonfinite` and the others on three boosted models in the corpus (`boost_reg`, `boost_cls`, `cat_boost_hash`): `ml-predict` equal to `ml` and `ml-serve` equal to `ml predict` byte for byte; `describe` equal to `scripts/oracle_describe.py` (the golden answers gained 3 entries, the 19 before unchanged). The two OpenAPI documents of the boosted fixtures validate with `openapi-spec-validator`, and Schemathesis 4.31.1 against them: 335 and 345 generated cases, no issue (the six documents before are unchanged and still pass) |
| GB9 fingerprint | `docs/bench/xplat_fp.txt` has 24 lines more, appended after the 88 (they are unchanged); `scripts/xplat_check.py` prints `ok, 112 lines for 28 fits` on the Mac and, from a clean clone, on the workstation with `taskset -c 0-6`, `ml-predict` equal to `ml`; its mutants are 17 (two new ones on a boost fit), all refused. The Raspberry Pi was not run |
| GB10 golden | `golden_constant.json`, `golden_numeric.json`, `golden_planned.json` unchanged and reproduced; `golden_describe.json` gained three entries and no entry changed |
| GB11 tables | `test_tables`: 49 flags, 79 rules, 23 limits, all reached by 317 fixtures (39 are the boosted model's); `gen_models.py --check`, `schemas.py --check`, `manifest.py --check`, `check_registry.py` clean |
| the whole | `sh scripts/check.sh` on the Mac: 497 conformance tests pass, none skipped, in 2,297 s (the machine shared); see `docs/status.md` for the rest of the gate |

**A real session** (`docs/examples/data`, the `ml` binary of this tree): `ml fit --model boost --task classification --target benign --rounds 100 --learning-rate 0.1 --max-depth 3 --seed 1` on `breast_cancer_train.csv` writes 63,327 bytes (100 rounds, 1,272 nodes, base score 0.5183) and `eval` on the test file prints accuracy 0.958, precision 0.9667, recall 0.9667, F1 0.9667, ROC-AUC 0.9828, log-loss 0.1482. On `diabetes_train.csv` (regression, depth 3, 100 rounds) the test RMSE is 61.38 (MAE 46.68, R2 0.241) against 63.40 for scikit-learn's histogram boosting with the same settings; with `--rounds 400 --early-stopping 10 --split train=0.8` the fit stops at round 28 (378 nodes, 19,137 bytes) and the 67 held-out rows score RMSE 59.48.

**Speed, measured here, not a claim.** `scripts/bench_boost.py --sklearn` on the workstation (load 1 before and 1.5 after, 7 pinned cores, minimum of 5 runs; 100,000 rows x 20 columns, 5 percent of two columns missing, 50 rounds, depth 6, min-leaf 10, 64 bins): `ml fit --model boost` as a process (CSV read, binning, 50 rounds, file written) 1.96 s for a regression and 1.94 s for a classification; scikit-learn's `HistGradientBoosting*` `fit` alone on the matrix in memory, one OpenMP thread, 0.69 s and 0.86 s. The two do not do the same work (the CSV read is on one side only, the bins and growth order differ), so the ratio of about 2.3 to 2.8 against `ml` is the ratio of these two programs on these inputs, and `ml` is the slower one. A histogram built from four features per pass of the node's rows (the tree library's layout) would give the same bits and is open.

**What the gates do not claim.** Only these two machines were used. The oracle shares its author and its document with the code. The mutants prove the tests notice 153 one-site edits, not that the tests are complete. The scikit-learn comparison is a bound on quality on four small files, not a proof that the algorithm is the best one: ties, bins and the gain test differ. A model of 10,000 deep rounds can exceed the 16 MiB default of `--max-bytes`; `fit` then refuses to write it. There is no column sampling, no multiclass, no feature importance, and the rounds are not threaded.

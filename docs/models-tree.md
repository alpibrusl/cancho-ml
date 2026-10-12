# cancho-ml decision trees and random forests (`lib/models/tree`, stage M7)

**Status: sections 1 to 9 are the specification and the gates, written BEFORE any code of this stage (the first commit of
the branch holds only this file). Section 10 (results) is filled in after the gates have run; a number is in it only if a
script of this repository printed it, with its conditions.** Tags as in `design.md`: **[choice]** is a decision the
maintainer can overturn, **[estimate]** is not measured, **[read]** is read from a document. A claim found false is
corrected in place and marked.

The library is **internal**: it is not wired into the `ml` command line yet. It productionises spike F
(`cancho-ml-forest`, `docs/spikes.md` sections 10 and 11) and keeps its algorithm where this document does not say otherwise.

## 1. Scope

| item | what | notes |
|---|---|---|
| CART classification | **binary** labels 0.0 / 1.0; Gini | design.md section 1 (multiclass is not in v1) |
| CART regression | variance reduction | section 3 |
| split search | **histogram**, quantile bins, up to 255 value bins plus one missing bin | section 2.3 |
| limits | `max_depth`, `min_samples_leaf`, `max_nodes` | section 4; rule tags `ml.limit.depth`, `ml.limit.nodes` |
| random forest | seeded bagging, a feature subset per node, parallel over trees | sections 2.4, 5 |
| prediction | `predict_proba` (class 1), `predict` (class), regression value | aggregation order in 2.6 |
| missing values | a per-column bitmap; a learned default side per split | section 6 |
| model bytes | node arrays as tensor records for the `model` section of CML1 | section 7 |

A decision tree is a forest of one tree with `bootstrap = 0` and `mtry = d` (all columns). There is one entry point.
**Not in this stage**: the command line, the pipeline, multiclass, categorical splits, sample weights, feature importance,
out-of-bag scores, boosting.

## 2. The algorithm (normative; `oracle_tree.py` implements exactly this text)

Notation: `n` training rows, `d` columns, `B` value bins (flag `bins`, 2..255), `T` trees, `MAXD` = `max_depth`, `ML` =
`min_samples_leaf`, `MTRY` features per node, `seed` an int in [0, 2^63). `X[i][f]` is a float64 that is finite unless the
cell is missing (section 6). Arithmetic on 64-bit words is wrapping where it says so. Task 0 is classification (`y` is
exactly 0.0 or 1.0), task 1 is regression (`y` finite, `|y| <= 1e100`).

### 2.1 RNG and derived streams

SplitMix64 as `docs/rng.md`: `derive(h, c) = mix((h ^ c) + 0x9E3779B97F4A7C15)`, a draw is `s += 0x9E3779B97F4A7C15;
z = mix(s)`, `uniform(m) = (z >> 1) mod m`. Streams, never shared (design.md 6.2):

* bootstrap of tree `t`: start state `derive(derive(seed, 1), t)`;
* features of node `id` of tree `t`: start state `derive(derive(derive(seed, 2), t), id)`, where `id` is the **heap number**
  of the node (root 1, children `2 id` and `2 id + 1`; `id < 2^31` because `MAXD <= 30`).

No draw depends on which thread builds the tree or on the order in which nodes are visited.

### 2.2 Bootstrap

With `bootstrap = 1`, tree `t` draws `d_j = uniform(n)` for `j = 0 .. n-1` from the bootstrap stream, in order; its root row
sequence is `d_0, d_1, ...` (repeats count, every statistic below counts with multiplicity). With `bootstrap = 0` the root row
sequence is `0 .. n-1`.

### 2.3 Bins (quantile, deterministic)

For column `f`, let `v[0 .. nv)` be the **non-missing** training values of the column (all `n` rows, not the bootstrap) sorted
ascending (IEEE order; `-0.0` sorts before `0.0` by the bit-pattern key and compares equal to it in `<=`). Edge `e[f][k] =
v[floor(k nv / B)]` for `k = 1 .. B-1` (integer division; an actual data value, repeats can occur; a column with `nv = 0` has
all edges `0.0`). `bin(x) = #{k in 1..B-1 : e[f][k] <= x}` in `0 .. B-1` (a value equal to an edge goes to the **upper** bin);
a missing cell has bin `B`. A split "at bin `b`" (`b = 0 .. B-2`) sends a non-missing row left iff `bin <= b`, equivalently iff
`x < e[f][b+1]`; training uses bins, prediction uses the float comparison with the stored threshold `e[f][b+1]`.

### 2.4 Node

A node holds a row sequence `R` (size `m`, in the order given below), depth `dep` (root 0), and its heap number `id`. Nodes are
created in **preorder** (left subtree completely before the right sibling is visited); a node's index in the tree's node arrays
is its preorder rank, root 0. Splitting stably: `R` keeps its order inside each child (the root's order is the bootstrap order).

A node is a **leaf** iff `dep = MAXD`, or `m < 2 ML`, or the task is classification and one class count is 0, or the task is
regression and all `y` of `R` are equal (`ymin == ymax`), or no valid candidate exists. Otherwise:

1. **Feature subset.** `a = [0 .. d-1]`; for `i = 0 .. MTRY-1`: `j = i + uniform(d - i)`; swap `a[i], a[j]`; the subset is
   `a[0 .. MTRY)` sorted ascending. (When `MTRY = d` the draws are still made; the subset is all columns.)
2. **Candidates**, in the order *feature ascending, bin `b` ascending, then variant*: variant 0 sends the node's missing rows
   **right**, variant 1 sends them **left**. A variant 1 is evaluated only if the node has at least one missing row in that
   column (otherwise it equals variant 0 and could not win a strict comparison, so skipping it changes nothing; the oracle
   evaluates it anyway, which checks this). Left side `(nl, ...)`, right side `m - nl`; a candidate is **valid** iff `nl >= ML`
   and `nr >= ML`.
3. **Criterion.** Classification (Gini), with `pl` = class-1 count on the left, `ql = nl - pl`, `pr = n1 - pl`, `qr = nr - pr`:
   maximise `S = (pl^2 + ql^2)/nl + (pr^2 + qr^2)/nr`, i.e. `N / Dn` with the integers `N = (pl^2+ql^2) nr + (pr^2+qr^2) nl`,
   `Dn = nl nr`. Candidate 1 beats candidate 2 iff `N1 Dn2 > N2 Dn1`, evaluated **exactly in integers** (section 3.1); no
   float takes part in a classification split decision. Regression: maximise `S = SL^2/nl + SR^2/nr` in float64, with the
   sums of section 3.2; candidate 1 beats candidate 2 iff `S1 > S2` (IEEE `>`).
4. **Ties.** The best candidate is replaced **only if strictly better**: among exactly equal scores the lowest feature, then the
   lowest bin, then variant 0 wins (design.md 6.4). The best valid candidate is taken even if its decrease is zero.
5. Rows go to the children: a non-missing row left iff `bin <= b`; a missing row left iff the winner is variant 1. The winner's
   variant is stored as `dleft`.

### 2.5 Leaf values

Classification: the integer counts `(n0, n1)`; the leaf probability is `float(n1) / float(n0 + n1)` (one IEEE division).
Regression: `Sm / float(m)` with `Sm` the section 3.2 sum of the leaf's rows (one IEEE division). Both are stored as the node's
`val`; an internal node's `val` is its threshold `e[f][b+1]`.

### 2.6 Prediction and aggregation order

A row walks each tree from the root: at an internal node with column `f`, a missing cell goes to the child `dleft` names (left
if `dleft = 1`), a present cell goes left iff `x < val`. Forest value: `acc = 0.0; for t = 0 .. T-1 in order: acc = acc + v_t;
result = acc / float(T)`, where `v_t` is the reached leaf's `val`. This order is part of the API and does not depend on the
thread count. Classification: `predict_proba` is `result` (the probability of class 1); `predict` is 1 iff `result > 0.5`
(a tie goes to class 0, the lowest class index). Regression: `result`.

## 3. Numeric decisions

### 3.1 Gini by exact integer arithmetic

`N < m^3` and `Dn <= m^2/4`, so for `m <= 2^20` the products `N1 Dn2` stay below `2^98`: they do not fit an `int` (cancho has no
multiply-high, upstream gap U1). The comparison multiplies in base `2^30` limbs, exactly. **Row ceiling [choice]: `n <= 2^20 =
1,048,576`** training rows (`ml.input.too-many-rows` above it; the bound is the proof above, and raising it needs a three-limb
numerator). The ceiling applies to `n`, which is also the root size of every tree (the bootstrap draws `n` rows). A unit test
compares the comparison to Python big integers on random pairs and on pairs at the bounds (`scripts/check_tree.sh`).

### 3.2 Regression sums: float64 in a fixed row order (not the linalg block order, not exact integers) **[choice]**

Every regression sum is a **sequential** sum from `+0.0`, in the node's row order (section 2.4: stable, so a function of the
bootstrap order alone): the node total `Sm`, each bin's sum `s_b`, the cumulative `L_b = ((0.0 + s_0) + s_1) + ... + s_b` over
bins ascending, and `SL = L_b` (variant 0) or `SL = L_b + s_miss` (variant 1), `SR = Sm - SL`, `S = SL*SL/float(nl) +
SR*SR/float(nr)` (the products and quotients are separate roundings; there is no `fma`, `floating-point.md`).

*Why not the package order of `docs/linalg.md` section 2:* that order exists so that a sum split across threads has one
answer. No regression sum is split across threads here: **a tree is built by one thread, whole**, so the only order that
matters is the row order inside the node, which the algorithm fixes. A block/tree sum would add cost and change nothing the
gates can see.
*Why not exact integer scaled sums:* it would quantise the target (`y * 2^k` rounded), which changes what the model learns,
and the exact comparison of squared sums needs more than 128 bits at useful scales. *The gate's logic* (section 8, G-ident):
the oracle must reproduce **bits**; sequential IEEE sums are reproduced exactly by `numpy.bincount` with weights and
`numpy.cumsum` (both sequential; `numpy.sum` is pairwise and is NOT used), so the float choice is checkable, and the stable
partition is what makes the order a function of the algorithm instead of the implementation. The cost is that the
split search is only as exact as float64 is: two candidates whose true scores differ by less than the rounding can be ordered
either way, **identically in every run and at every thread count**; this is the definition, not a defect to test away.
`ml.fit.non-finite` cannot occur: `|y| <= 1e100` and `n <= 2^20` keep every square below `1e206`.

## 4. Parameters, limits and refusals

`Params`: `task`, `trees` (1..1024), `max_depth` (1..30), `min_leaf` (1..n), `max_nodes` (per tree), `bins` (2..255), `mtry`
(1..d; 0 means the default: classification `floor(sqrt(d))`, regression `max(1, d/3)`), `bootstrap` (0 or 1), `seed`, `threads`
(1..64). Every check is made **before any allocation sized by input**.

| tag | exit | when | repair |
|---|---|---|---|
| `ml.input.empty` / `ml.input.too-many-rows` / `too-many-columns` / `too-many-cells` | 8 | `ml.limits` caps; rows also above 2^20 (3.1) | the flag, or a row sample |
| `ml.param.out-of-range` | 2 | any parameter outside its range above, a negative seed, a bitmap of the wrong length | the range |
| `ml.param.threads` | 2 | threads outside 1..64 | 1 to 64 |
| `ml.limit.depth` | 8 | `max_depth` above 30 (the ceiling that keeps the heap number inside 31 bits) | a depth of at most 30 |
| `ml.limit.nodes` | 8 | `max_nodes` above 2^22; the forest's allocation `trees * cap` above 2^24 nodes (`cap = min(max_nodes, 2^(MAXD+1)-1, 2n-1)`); **or a tree that needs more than `max_nodes` nodes while growing** | `--max-nodes`, a smaller depth or more `min_samples_leaf` |
| `ml.input.non-finite` | 8 | a present `X` cell that is NaN or infinite; a regression `y` that is not finite or above `1e100` | impute, mark it missing, or drop the row |
| `ml.label.not-binary` | 8 | a classification `y` that is not exactly 0.0 or 1.0 | encode the label |
| `ml.label.single-class` | 8 | classification with all labels equal | none |
| `ml.schema.mismatch` | 8 | prediction columns differ from the model's | the difference |
| `ml.model.truncated` / `ml.model.malformed` | 8 | section 7 | none |

`max_nodes` is a **refusal, not a truncation** **[choice]**: stopping a depth-first growth at a node count would make the
model depend on the visiting order and would change it silently; the alternative (best-first growth with a leaf budget) is a
different algorithm. A refusal found in a thread is reported as the refusal of the **lowest tree index**, so it is the same at
every thread count.

## 5. Threads (design.md 6.3, `cancho-table` `docs/parallel.md` [read])

* `fit` with `threads = t`: the number of threads used is `min(t, T)`. Thread `k` (0-based) builds trees `[k T / t', (k+1) T / t')`
  with `t'` the number used: a **static partition of trees**. Each tree is built wholly by one thread with its own streams
  (2.1), so a tree's bytes are a function of `(data, params, seed, t)` only.
* A cancho thread gets one pointer-width payload and no shared reads (`docs/upstream/spawn_shared_slice.cho`): each thread
  **owns a copy** of the bin matrix and the labels (bytes: `copy_into`), builds into its own boxes (fixed slots, `cap` nodes per
  tree), and the parent merges the trees **in tree order** after `join`. The copies are timed. Binning (edges, bin matrix) runs
  in the parent before the threads start; it is the serial part of `fit` and is reported separately.
* `predict` with `threads = t`: rows are cut into `t` contiguous pieces by a static partition; each row's value depends only on
  that row and the forest (2.6), so any partition gives the same bits. Pieces are copied into the threads like the bins.
* D1 for this stage: **the packed forest and every prediction are bit-identical at 1, 2, 4, 8 and 16 threads** (G-thread).

## 6. Missing values

The matrix carries a **per-column bitmap**: column `c` occupies `ceil(n/8)` bytes; row `i` is bit `i & 7` (least significant
first) of byte `i >> 3`; set means missing. An empty bitmap means no missing cell. The value stored in a missing cell is
**never read** (it may be NaN). Bits beyond the last row are ignored. **Rule [choice]: a missing cell goes to a learned
default side.** For each candidate split the best side for the node's missing rows is chosen by the same criterion (variants
0 and 1 of 2.4); the side is stored per node as `dleft`. At prediction time a missing cell follows `dleft`, whatever the
column; a node with no training-time missing rows in its column has `dleft = 0` (variant 0 wins the tie). Missing is not an
error at predict time and has no repair; the alternative (refusing with a tag) would make a model unusable on its own
training data's kind. A missing **target** is not representable: `y` has no bitmap, and a non-finite `y` is refused
(`ml.input.non-finite`).
The bin of a missing cell is `B`; edges ignore missing cells (2.3). A column that is missing in every training row has no
valid split.

## 7. The model: tensor records for the CML1 `model` section (design.md 4.1)

The payload (the content of one section; the envelope, the plan, the schema and the checksum belong to the command line) is
little-endian with no padding: `u32 kind` = 3 (tree ensemble), then **tensor records** as design.md 4.1 says: `u32 dtype`
(1 = i64, 2 = f64), `u32 rank`, `rank` `u64` extents, then the elements row-major. The records, in this order:

| # | dtype | rank | shape | content |
|---|---|---|---|---|
| 1 | i64 | 1 | 8 | `task, trees, d, bins, total_nodes, max_depth, min_leaf, mtry` |
| 2 | i64 | 1 | trees+1 | `tree_off`: node offset of each tree; `tree_off[0] = 0`, `tree_off[trees] = total_nodes` |
| 3 | i64 | 1 | N | `feat` (-1 leaf) |
| 4 | i64 | 1 | N | `bin` (-1 leaf) |
| 5 | i64 | 1 | N | `dleft` (0 or 1; 0 for a leaf) |
| 6 | i64 | 1 | N | `left` (tree-local index; -1 leaf) |
| 7 | i64 | 1 | N | `right` (-1 leaf) |
| 8 | i64 | 1 | N | `na` (class 0 count / row count) |
| 9 | i64 | 1 | N | `nb` (class 1 count / 0) |
| 10 | f64 | 1 | N | `val` (threshold bits / leaf value bits) |

(`N = total_nodes`.) **Reader** `model.read(bytes, limits)`: the structure is checked against the byte length **before
anything is allocated**: kind, then every record's dtype, rank and extents against the expected ones, `N <= max_nodes_total`
(`ml.limit.nodes`), `8 * (8 + trees + 1 + 7 N) ...` summed with overflow-free comparisons must equal the payload length
(`ml.model.truncated` if the payload is shorter than the records need, `ml.model.malformed` if longer or any extent wrong).
Then, per node: `-1 <= feat < d`; a leaf has `feat = bin = left = right = -1`, `dleft = 0`; an internal node has `0 <= bin <= bins - 2`,
`dleft` 0 or 1, `left = i + 1` (preorder), counts `>= 0`; a classification leaf has `na + nb >= 1`, both counts at most 2^20, and a
`val` whose bits are those of `float(nb) / float(na + nb)`; a regression leaf has `na >= 1`, `nb = 0`; every `val` decodes to a
finite float (`fbits`); tree offsets strictly increase from 0 to `total_nodes`; the header values are inside their ranges (`trees`,
`d`, `bins`, `task`, `max_depth`, `mtry`). The trees must be **well-formed binary trees in preorder of depth at most `max_depth`**,
checked by one walk with a stack of the right children still to come: a node that follows a leaf must be exactly the innermost
pending right child, the depth is the number of ancestors, and nothing may be pending at the end. That is all that is needed for a
walk to end inside the tree (every step goes to a later node; an accepted tree has no cycle and no node outside it). *A first draft
also checked `right < tree_size`, `right > left` and "the last node is a leaf"; the mutants of those checks survived, and the
reason is that each is implied by the stack rule (a right child outside the tree, at or before the left child, or pushed by a last
internal node can never be popped by a node of that index), so they were removed instead of being kept as checks no test can
distinguish.* Anything else is `ml.model.malformed`. A file is input: it may say anything. The envelope's own tags (`bad-magic`,
`version`, `checksum`) are the command line's.

## 8. Gates (fixed now; each must be able to fail; a gate that cannot is not a gate)

Data: `ml-data/n1000`, `n100000` (20 float64 columns, labels 0/1, the files of spike F), and **derived inputs written by
`scripts/gen_tree_data.py` from them** (the same files feed the oracle and cancho, so the generator is not part of the
algorithm): a regression target (a fixed nonlinear function of the columns plus a seeded noise), and a missing bitmap (about 12%
of the cells in 7 columns, the probability depending on the label so that the default side matters).

**G-ident.** For every configuration of the table, `tree-check` (the compiled library) and `oracle_tree.py` (Python/numpy,
separate code, big integers for the Gini comparison) print, byte for byte equal: every edge (`E`), every node (`N t i feat bin
dleft left right na nb valbits`) and every test prediction (`P i bits`). The test rows are the last `n/5`; the training rows
are the first `n - n/5`. Sizes: 1,000 and 100,000 rows. Any difference fails the gate.

| id | task | T | depth | ML | bins | mtry | boot | missing |
|---|---|---|---|---|---|---|---|---|
| C1 | class | 10 | 8 | 1 | 64 | 4 | 1 | no |
| C2 | class | 7 | 10 | 5 | 32 | 5 | 1 | yes |
| C3 | class | 1 | 6 | 1 | 64 | 20 | 0 | no |
| R1 | regression | 8 | 9 | 3 | 48 | 7 | 1 | yes |
| R2 | regression | 1 | 7 | 2 | 64 | 20 | 0 | no |

C1 is the configuration of spike F; its nodes and probabilities must also equal spike F's `N` and `P` lines (same data, seed
20261012): a regression check of this port against the spike's own gate (heap numbers are recomputed from the preorder).
The oracle prints the number of exactly equal score comparisons it saw (so the tie rule is shown to be exercised) and the
number of nodes whose winner is variant 1 (so the missing rule is exercised).

**G-mutants.** Each mutant is a one-site edit of the library (`scripts/mutants_tree.py`, each pattern asserted to occur
exactly once, the unmutated control must pass first, built and run against the cached oracle output at both sizes). A mutant
must be **killed** (some line differs, the program refuses, or it does not finish); a survivor is explained or voids the gate.
The list (group, mutant): tie rule (replace on equal); equal tie among variants (variant 1 first); bin boundary (`e <= x`
becomes `e < x`); bootstrap draw (draw 0 of tree 0 plus one); child swap (training and prediction); depth (`MAXD - 1`); feature
subset size (`MTRY - 1`); aggregation order (trees summed from the last to the first); missing side (missing always left;
missing always right; missing bin treated as bin 0); `min_samples_leaf` boundary (`nl > ML`); leaf rule (`m < 2 ML` becomes
`m <= 2 ML`); edge index (`k nv / B` becomes `k nv / (B+1)`); the Gini comparison (an operand swap, and a limb shift); regression
sum order (the node total summed in reverse; cumulative from `-0.0` is NOT a mutant, it is equal); stability of the partition
(unstable swap partition); regression purity test removed; leaf value (`Sm / m` becomes `Sm / (m + 1)`); stream index (the
feature stream uses the node's preorder rank instead of its heap number); the thread group below.

**G-thread.** `tree-check` prints a hash of the packed forest (the section 7 payload) and of all predictions. For C1, C2, R1
and for T = 10 and T = 32 at 100,000 rows, the hashes at 1, 2, 4, 8 and 16 threads (also 3 and 5, which do not divide T) are
equal. Mutants that must be killed by it: the tree's bootstrap stream indexed by the thread-local rather than the global tree
number; the merge in reverse thread order; the partition `k T / t` replaced by `k * ceil(T/t)` with the extra trees dropped
(a model with missing trees); the prediction partition giving the last piece one row less (a row left unwritten).

**G-fuzz.** `tree-check fuzz`: a model payload of a real forest (classification with missing, and regression) is damaged in
every way listed, and for each damaged payload `model.read` and, if it accepts, `predict` on 200 rows (some missing) must
return; nothing traps and nothing runs without end. Damage: every prefix (every truncation); every byte replaced by 0x00,
0xFF and its complement; every byte flipped in each of 8 bits for the first 512 bytes; 4,000 seeded multi-byte damages. The
expected result of each class of damage is also checked where it is determined (a truncation is `ml.model.truncated`; an
undamaged payload is accepted and equals the forest). Mutants that must be killed: any one of the validations of section 7
removed (child order, feature range, leaf form, tree offsets, the length equation, the `val` finiteness, `dleft`); the process
dying (SIGILL), an unexpected acceptance or a hang (timeout 120 s) is a kill.

**G-refuse.** One fixture per tag of section 4 (a unit test: `cancho test`), including the limits checked before allocation
(a request with `trees * cap` over the ceiling is refused without allocating, shown by passing a huge `n` with an empty slice),
and a **refusal reproducible at any thread count** (`ml.limit.nodes` from a tree of the last thread, same tag at 1 and 8 threads).

**G-ref (scikit-learn).** `RandomForestClassifier` / `RandomForestRegressor` with `n_estimators`, `max_depth`,
`max_features`, `min_samples_leaf`, `bootstrap` matching, `n_jobs = 1`, `random_state = 0`, on the same train/test split.
**Different algorithm** (exact thresholds, its own RNG, a feature subset is redrawn when no split exists, the criterion is
float), so only the quality of the result is compared, within a tolerance fixed now: classification test AUC within **0.005**
at 100,000 rows (spike F measured 0.0001 at 100,000 and 0.001 at 1,000,000 for C1); regression test RMSE within **3% relative**
at 100,000 rows. A tolerance that no mutant could break is not evidence of correctness; the oracle gate is, and this gate is
only a sanity bound against a shared misreading of the algorithm. Its ability to fail is shown with a deliberately broken
control (a forest of depth 1) that must be outside the tolerance.

**G-limits.** `max_depth` 31 and `max_nodes` over the ceiling are refused; a tree stopped by `max_nodes` growth is refused with
`ml.limit.nodes` and never silently truncated; memory is bounded: the largest allocation of `fit` is computed from the
parameters (reported by the driver), not from the data content.

## 9. Timing protocol (fixed now)

* One machine: Apple M4 Max, macOS, 16 cores (10 performance, 4 efficiency, 2 more by the OS count; the OS decides the core
  class of a thread: **thread scaling beyond the performance cores is not linear and is reported as measured**). The load
  average (1 min) is read at the start and the end of every cell; a cell with load >= 2 at either end is flagged.
* `fit` = from the loaded float arrays to the packed forest, **including** edges, bin matrix, the thread copies and the merge;
  `predict` = from the loaded test matrix to the probabilities in memory (including the piece copies when threaded). Timed in
  the program with `clock_ms` (1 ms): a cell below 100 ms is repeated `R` times inside the program and divided, `R` printed.
* **Minimum of 7 interleaved** samples (one cancho sample, one competitor sample, in turn); spread (max over min) printed; a
  cell with spread > 1.25 is rerun and, if it stays, marked.
* Sizes 100,000 and 1,000,000 rows x 20 columns (train = first 80%, test = last 20%), classification `T = 10, depth 8, 64 bins,
  mtry 4` for the comparison with scikit-learn at one thread; scaling at `T = 32` (it divides 1, 2, 4, 8, 16) for 1, 2, 4, 8, 16
  threads, with `T = 10` as well, whose partition is uneven (reported as it is).
* **What is fair.** scikit-learn `RandomForest*` is an **exact-split** forest, a different algorithm, so the ratio is the time
  to produce a model of this family, not the speed of the same algorithm; **no speed claim is made against it**, only the two
  columns side by side. A histogram forest is a fair competitor; the installed packages are checked and listed in section 10
  (no installation is allowed). `HistGradientBoosting*` is gradient boosting (leaf-wise growth, second-order leaf values, no
  bagging); with one iteration it builds one histogram tree and is listed only as context, never as an equivalent.

## 10. Results

*(filled in after the gates have run)*

# cancho-ml decision trees and random forests (`lib/models/tree`, stage M7)

**Status: sections 1 to 9 are the specification and the gates, written BEFORE any code of this stage (the first commit of
the branch holds only this file). Section 10 (results) is filled in after the gates have run; a number is in it only if a
script of this repository printed it, with its conditions.** Tags as in `design.md`: **[choice]** is a decision the
maintainer can overturn, **[estimate]** is not measured, **[read]** is read from a document. A claim found false is
corrected in place and marked.

The library is **internal**: it is not wired into the `ml` command line yet. It productionises spike F
(`scripts/spikes/spike_f.cho`, `docs/spikes.md` sections 13 and 14) and keeps its algorithm where this document does not say otherwise.

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
multiply-high: `docs/upstream/mulhi.cho`). The comparison multiplies in base `2^30` limbs, exactly. **Row ceiling [choice]: `n <= 2^20 =
1,048,576`** training rows (`ml.input.too-many-rows` above it; the bound is the proof above, and raising it needs a three-limb
numerator). The ceiling applies to `n`, which is also the root size of every tree (the bootstrap draws `n` rows). `tree-check mode=gt` prints the
comparison of 2,006 pairs (2,000 pseudo-random from the SplitMix64 stream, 6 at the bounds) and `scripts/gate_tree.py --exact` checks
every answer against Python integers. (A tighter bound is `N <= m nl nr <= 2^58`; the products are below 2^96.)

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
| `ml.param.threads` | 2 | threads outside 1..64; or the copies the threads would make (`n d 9 + threads (n d + 24 n)` bytes, section 5) above 2^34 | 1 to 64, or fewer threads |
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
  tree), and the parent merges the trees **in tree order** after `join`. The copies are timed. The memory they take is bounded
  before anything is allocated: `n d 9 + threads (n d + 24 n)` bytes (the bin matrix and about 24 bytes of scratch a row per
  thread, and one more gather of the matrix by the binning) must not exceed 2^34 [choice, provisional], else `ml.param.threads`.
* **Binning in threads (added after the first benchmark showed the serial binning as the limit of the scaling):** with more than
  one thread and more than one column the columns are cut among the threads (`tree.pbins`). A column's edges and bins depend on
  that column and its missing bits alone, so any partition of the columns gives the same arrays; the parent gathers each thread's
  columns column-major, the thread sorts and bins them, the parent copies the results into the global arrays. With one thread the
  whole matrix is binned in the parent (`bins.make_edges`, `bins.make_bins`: two implementations of the same definition, and G-thread
  compares their bytes at 1 and at 2 or more threads).
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
little-endian with no padding: `u32 kind` = 7 (tree ensemble; the registry is design.md 4.1: 3 was claimed by the float constant model of the CLI), then **tensor records** as design.md 4.1 says: `u32 dtype`
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

| id | task | T | depth | ML | bins | mtry | boot | missing | data |
|---|---|---|---|---|---|---|---|---|---|
| C1 | class | 10 | 8 | 1 | 64 | 4 | 1 | no | ml-data |
| C2 | class | 7 | 10 | 5 | 32 | 5 | 1 | yes | ml-data |
| C3 | class | 1 | 6 | 1 | 64 | 20 | 0 | no | ml-data |
| C4 | class | 3 | 8 | 2 | 200 | 6 | 1 | yes | ml-data |
| C5 | class | 2 | 7 | 1 | 255 | 8 | 1 | no | ml-data |
| C6 | class | 4 | 4 | 1 | 2 | 20 | 1 | yes | ml-data |
| R1 | regression | 8 | 9 | 3 | 48 | 7 | 1 | yes | ml-data |
| R2 | regression | 1 | 7 | 2 | 64 | 20 | 0 | no | ml-data |
| E1 | class | 6 | 6 | 2 | 16 | default (2 of 8) | 1 | yes | edge set |
| E2 | regression | 5 | 7 | 1 | 32 | 3 | 1 | yes | edge set (noisy target) |
| E3 | class | 1 | 5 | 1 | 8 | 8 | 0 | yes | edge set |
| E4 | regression | 4 | 6 | 1 | 16 | 4 | 1 | yes | edge set (integer target 0..5: many nodes whose targets are all equal) |

*(Corrected in place: the first draft listed C1 to C3, R1, R2. C4 to C6 were added when the bin search was specialised for 64 and for
256 edges, so that both code paths and the smallest bin count are covered; E1 to E4 when the regression-purity and the variant-tie
mutants showed that the large inputs never reach those cases.)* The **edge-case set** (`scripts/gen_tree_data.py`, 1,000 rows x 8
columns, only at 1,000 rows): a constant column, a column of only `-0.0` and `0.0`, a column of small integers, a column missing
in every row, a column missing in 95% of the rows, a column of `1e300`, `-1e300` and subnormals, a normal column with 5% missing,
and a column of heavy duplicates; so repeated edges, equal scores in large numbers, `-0.0` before `0.0` in the sort, a column with no
valid split and extreme magnitudes are all exercised. Missing cells hold NaN in the file given to both programs (a missing cell is
never read).

C1 is the configuration of spike F; its nodes and probabilities must also equal spike F's `N` and `P` lines (same data, seed
20261012): a regression check of this port against the spike's own gate (heap numbers are recomputed from the preorder).
The oracle prints the number of exactly equal score comparisons it saw (so the tie rule is shown to be exercised) and the
number of nodes whose winner is variant 1 (so the missing rule is exercised).

**G-mutants.** Each mutant is a list of one-site edits of the library (`scripts/mutants_tree.py`, each pattern asserted to occur
exactly once, applied to a copy of `lib/`, `tools/` and `tests/`). The unmutated control must pass every gate first. A mutant is
**killed** if the build of the driver succeeds and then one of the gates fails, times out or traps, in this order: `unit`
(`tests/t_tree.cho`), `exact`, `fuzz`, `ident` at 1,000 rows, `ident` at 100,000 rows, `threads`. A survivor must be explained
(an equivalent mutant) in the script or the gate does not pass; a mutant that does not build is reported and fails the run. The
list covers, at least: the tie rule (an equal score replaces, for both criteria), the tie between the two sides for missing rows
(variant 1 tried first, for both criteria), the bin boundary (`e <= x` becomes `e < x`), the bootstrap draw (draw 0 of tree 0 plus
one; the stream of tree + 1), the child direction (training, prediction, both), the depth (`MAXD - 1`), the feature subset size
(`MTRY - 1`; the first `MTRY - 1` marked; a draw over all `d`; the stream index), the aggregation order (trees summed from the last
to the first; a multiplication by `1/T`; the sum started from `-0.0`), the missing side (always left, always right, flipped in
prediction, missing in bin 0, variant 1 never tried, variant 1 without its sums or counts, missing cells entering the edges, the
default side not stored), the `min_samples_leaf` boundary and the leaf rule, the edge index and the threshold index, the leaf
values, the Gini comparison (an operand, a limb shift, the carry, the high part), the regression sums (the node total summed last to
first, the bin sums last to first, the cumulative sum from the last bin, the right sum perturbed), the stability of the partition
(an unstable swap partition replaces the stable one), the regression purity test, the node budget and every limit of section 4 at its
boundary, the label, target and finiteness checks, the thread group below, the binning in threads, and every validation of the
reader of section 7.
*(Corrected in place: the first draft's list had 18 items; the final one has 98 mutants, the added ones found by running the
earlier mutants and reading which checks nothing could distinguish.)* A second run restricts the gates to the oracle and the thread
gate (`--gates ident1000,ident100000,threads`) so that what the oracle alone kills is on record next to what needs the unit tests,
the exact comparison and the fuzz.

**G-thread.** `tree-check` prints a SHA-256 of the packed forest (the section 7 payload) and a hash of all predictions. For C1,
C2, R1, C4 at 100,000 rows with T = 10 and T = 32 trees where listed, and C2, C5, E1, E2, E4 at 1,000 rows, the hashes at 1, 2, 3,
4, 5, 8 and 16 threads (3 and 5 do not divide T) are equal. *(Corrected in place: the first draft said 1, 2, 4, 8, 16 and three
configurations.)* The prediction uses the same thread count as the fit. `tests/t_tree.cho` also compares the payload bytes and the
predictions at 1 to 7 threads for both tasks, and with more threads than trees and than rows. Mutants that must be killed by it: the
tree's stream indexed by the thread-local rather than the global tree number; the partition `k T / t` replaced by `k * ceil(T/t)`,
or the last group one tree short; the merge into the wrong slot; the prediction pieces leaving a row; the bitmap copy of a prediction
piece at a wrong column; every copy of the binning in threads.

**G-fuzz.** `tree-check mode=fuzz` fits two small forests (classification with missing cells, 4 trees of at most 6 levels; and
regression, 3 trees) on the 1,000-row inputs, writes their payloads (13,836 and 11,588 bytes) and damages them. For every damaged
payload `model.read` must return; an accepted forest is walked by `predict` on 32 rows with a missing cell in every 7th position; a
trap or a hang (timeout) is a failure. The damage: **every prefix** (answer must be `ml.model.truncated`); **every byte** replaced by
0x00, 0xFF and its complement, and each of its 8 bits flipped for the first 2,048 bytes; **8,000 seeded multi-byte damages**;
**every field of every node** (records 3 to 9) set to 0, 1, 2, the node count, 2^40, -2, -1, 2^62 and the original plus one; the
`val` of every node set to NaN and to infinities; the eight header values; every tree offset; the kind and every record header. For
those with a determined answer the reader's answer is compared with a rule written in the driver independently of the reader (for
example: a leaf's `bin` must be -1, an internal node's `left` must be its index plus one, its `right` must be the original, a class
leaf's `val` must be the ratio of its counts, a header `max_depth` below the real depth must be refused); a disagreement counts as
`unexpected`. And **payloads built to break one rule at a time** that a single-field damage cannot reach: an empty tree, nodes that
belong to no tree, the last node of a tree made internal. The undamaged payload must be accepted and write back the same bytes.
*(Corrected in place: the first draft judged only "does not trap". The first mutant run showed that removing a validation was not
detected by that, so the answers are now judged.)*

**G-refuse.** One fixture per tag of section 4 and per parameter range, in `tests/t_tree.cho` (a program, not `cancho test`: the
library spawns and the test runner's `main` has no `conc` in its row, `docs/upstream/test_runner_conc.cho`), including the limits
checked before allocation (a shape that says millions of rows or a forest of 2^24 nodes is refused while the data slices passed are
empty, so nothing sized by the shape can have been allocated), the boundary of the node ceiling (1024 trees of 16,384 nodes
allowed, of 16,385 refused), and a **refusal reproducible at any thread count** (`ml.limit.nodes` from a growing tree: the same
tag at 1, 2, 4 and 8 threads, and the same fit succeeds with room). The tags are checked to have a name and an exit code.

**G-ref (scikit-learn).** `RandomForestClassifier` / `RandomForestRegressor` with `n_estimators`, `max_depth`,
`max_features`, `min_samples_leaf`, `bootstrap` matching, `n_jobs = 1`, `random_state = 0`, on the same train/test split.
**Different algorithm** (exact thresholds, its own RNG, a feature subset is redrawn when no split exists, the criterion is
float), so only the quality of the result is compared, within a tolerance fixed now: classification test AUC within **0.005**
at 100,000 rows (spike F measured 0.0001 at 100,000 and 0.001 at 1,000,000 for C1); regression test RMSE within **3% relative**
at 100,000 rows. A tolerance that no mutant could break is not evidence of correctness; the oracle gate is, and this gate is
only a sanity bound against a shared misreading of the algorithm. Its ability to fail is shown with a deliberately broken
control (a forest of depth 1) that must be outside the tolerance.

**G-limits.** `max_depth` 31 and `max_nodes` over the ceiling are refused; a tree that needs more than `max_nodes` is refused with
`ml.limit.nodes` and never silently truncated; every allocation of `fit` is a formula of `(n, d, trees, cap, bins, threads)` (the
slot arrays `trees * 7 * cap` ints and `trees * cap` floats, the bin matrix `n * d` bytes, `n`-sized scratch per thread) and none
depends on the content of the data. (These are the fixtures of G-refuse.)

## 9. Timing protocol (fixed now)

* One machine: Apple M4 Max, macOS 26.2, 16 cores (12 performance and 4 efficiency, `sysctl hw.perflevel*`; 64 GB; the OS decides the
  core class of a thread: **thread scaling beyond 12 threads is not linear and is reported as measured**; the machine is shared with
  other users and processes, so the load is whatever the load average says, printed with every table). The load
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

*(Sections 10.1 to 10.11 are filled from the output of the scripts named in each; the raw lines of the timing are in
`docs/bench/tree_<size>.*` and `docs/bench/tree_<size>_b.*`, the mutant logs in `docs/bench/mutants_tree*.log`.)*

### 10.1 Conditions

Apple M4 Max (12 performance and 4 efficiency cores, 64 GB), macOS 26.2, cancho `a4572ea` (default LLVM backend, `--std`, edition 5),
Python 3.14.5, numpy 2.4.6, scikit-learn 1.8.0. Seed 20261012 everywhere. **The machine is shared with other users and with other
agents' builds and benchmarks: the one-minute load average was 3 to 12 during every timing below (and 8 to 24 during part of the
mutant runs), never the "about 1" of the start of the stage. Every timing cell is flagged `load >= 2`; none is a quiet-machine number.**
Gates other than timing do not depend on the load.

### 10.2 G-ident: passed, 20 pairs (`scripts/gate_tree.py --ident`)

The library and `scripts/oracle_tree.py` print the same edges (`E`), the same nodes (`N`) and the same test predictions (`P`), byte
for byte, for every configuration of section 8 at 1,000 and 100,000 rows (the edge-case set at 1,000 rows). What the oracle saw
(`ties` = comparisons of two exactly equal scores that were not the harmless duplicate of a variant 1; `v1` = splits whose winner
sends the missing rows left):

| config | rows | nodes | ties | v1 |
|---|---|---|---|---|
| C1 | 1,000 / 100,000 | 1,514 / 5,084 | 15,759 / 2,989 | 0 / 0 |
| C2 | 1,000 / 100,000 | 975 / 11,943 | 3,052 / 19,998 | 78 / 786 |
| C3 | 1,000 / 100,000 | 93 / 127 | 740 / 0 | 0 / 0 |
| C4 | 1,000 / 100,000 | 447 / 1,491 | 17,667 / 18,658 | 37 / 77 |
| C5 | 1,000 / 100,000 | 228 / 508 | 11,007 / 1,370 | 0 / 0 |
| C6 | 1,000 / 100,000 | 124 / 124 | 4 / 0 | 8 / 9 |
| R1 | 1,000 / 100,000 | 1,542 / 6,646 | 11,027 / 22,437 | 137 / 686 |
| R2 | 1,000 / 100,000 | 119 / 253 | 1,148 / 307 | 0 / 0 |
| E1, E2, E3, E4 | 1,000 | 184, 577, 55, 234 | 390, 3,635, 64, 629 | 13, 40, 4, 11 |

So the tie rule is exercised (C3 at 100,000 rows has none; the others have many) and so is the missing rule (`v1` > 0 wherever
there are missing cells). Not exercised by the oracle configurations: an exact tie between the two sides for the missing rows that is
also the winner (the mutant "variant 1 first" survives the oracle alone); the unit test `tie_variants` covers it with a three-row case.
The identity at 1,000,000 rows was not run (the oracle's cost; spike F ran it for C1, `docs/spikes.md` 14.1).

### 10.3 The port of spike F: identical (`scripts/gate_tree.py --spike`)

C1 (spike F's configuration, seed 20261012) gives spike F's own program's output: every edge (1,260 bit patterns), every node (1,514
at 1,000 rows, 5,084 at 100,000, compared by heap number, recomputed from the preorder) and every prediction (200 and 20,000), equal.

### 10.4 G-thread: passed, 11 rows x 7 thread counts (`scripts/gate_tree.py --threads`)

The SHA-256 of the packed forest and the hash of the predictions are equal at 1, 2, 3, 4, 5, 8 and 16 threads for C1, C2, R1 (T = 10),
C1 and R1 (T = 32), C4 (T = 3) at 100,000 rows, and C2 (T = 7), C5 (T = 2), E1 (T = 6), E2 (T = 5), E4 (T = 4) at 1,000 rows. The
prediction runs with the same thread count as the fit. The benchmark scripts check the same equality at 100,000 and 1,000,000 rows
(T = 10 and 32, threads 1, 2, 4, 8, 16: `IDENTICAL` in all four tables of 10.9). Thread counts above the number of trees, rows or
columns are covered by `tests/t_tree.cho` (64 threads on a 2-tree, 40-row fit).

### 10.5 G-mutants: passed, no unexplained survivor

100 mutants (`scripts/mutants_tree.py`; log `docs/bench/mutants_tree.log`). The unmutated control passes every gate. **94 killed,
6 survivors, all explained, no unexplained survivor, none failed to build.** By the first gate that failed: 58 by the unit tests, 16 by the fuzz, 13 by the oracle at 1,000 rows, 4 by the thread gate, 1 by the exact comparison, 1 by the oracle at 100,000 rows, 1 by the unit tests (a hang). Survivors
(each is an equivalent mutant or unobservable by construction, with the argument in the script):

* `score: swapped operands`: IEEE addition is commutative;
* `aggregation: sum starts from the first tree` (the sum starts at `-0.0`): the same bits unless a leaf value is `-0.0`, and none is
  (sums start from `+0.0`, `0.0 / m` is `+0.0`, class values are ratios of counts);
* `threads: another static partition of the prediction rows` and `pbins: another static partition of the columns`: rows and columns
  are independent, so another partition gives the same bits; these mutants **are** another partition, which is the property the gate
  relies on;
* `threads: refusal of the highest tree`: the only refusal a tree returns is `ml.limit.nodes`, so which tree's is reported cannot change
  the tag today;
* `pbins: a column with no value has edge 1.0`: such a column has only missing cells, its edges are never read and are not part of
  the model (the serial path's `0.0` is killed by the E lines of the edge-case set: `a column with no value has edge 1.0 (serial)`).

**What the oracle alone kills.** A second run with only the oracle identity (1,000 and 100,000 rows) and the thread gate
(`--gates ident1000,ident100000,threads`, log `docs/bench/mutants_tree_oracle.log`): 62 killed (48 by identity at 1,000 rows, 2 at
100,000 rows, 11 by the thread gate, 1 of them a hang). Every mutant of the algorithm named in the stage's list is among them: the tie
rule (both criteria), the bin boundary, the bootstrap draw, the child direction (training, prediction, both), the depth, the
feature subset size, the aggregation order, the missing side (always left, always right, flipped, missing in bin 0), the `min_samples_leaf`
boundary, the sums of the regression criterion in their order, the unstable partition and the thread group. The oracle does **not** kill:
the two ties between the sides for missing rows (the unit test does), the checks of limits and ranges, and the reader's validations
(the unit tests, the exact comparison and the fuzz do): that is the division of work, and it is on record, not assumed.


### 10.6 G-fuzz: passed (`scripts/gate_tree.py --fuzz`, about 0.6 s). Payloads of 13,836 bytes (classification, 4 trees, 212 nodes, with missing cells)
and 11,588 bytes (regression, 3 trees, 177 nodes). Byte-level and seeded damage: 6,699 accepted and 43,771 refused (classification),
9,062 accepted and 36,343 refused (regression), plus the structured set (every field of every node at 9 values, every `val` at 5, the
header, the offsets, the record headers) and the crafted payloads; **no trap, no hang, no answer that disagrees with the rules
(`unexpected 0`)**. Many accepted payloads are legitimate: a threshold or a count changed to another valid value; the file checksum of
the envelope is what catches those, not the reader. The reader accepts what it can walk safely. The undamaged payload is accepted and
writes back the same bytes, and every prefix is `ml.model.truncated`.
*(Corrected in place at W3a, the stage that wired the tree into `ml`. The sentence "no trap" above was false: the structured set never set a field to 2^63 - 1, and a class leaf whose count `nb` was
2^63 - 1 made the reader compute `na + nb` before it compared the counts with their bound of 2^20; cancho traps on an overflow, so a damaged payload ended the process (exit -4). It was found by the
seeded damage by whole words of `tests/conformance/test_wire_trees_file.py`. The bounds are compared first now (`check_tree` in `lib/models/tree/model.cho`), the structured set also tries 2^63 - 1, and
the mutant "reader: counts bounded before they are added" is killed by the fuzz. Nothing else of the library changed, and the payload bytes did not.)*


### 10.7 Unit tests and refusals (`tests/t_tree.cho`, `scripts/gate_tree.py --unit`)

21 tests, all passing: every parameter range and every tag of section 4 (including `ml.limit.nodes` from a growing tree at 1, 2,
4 and 8 threads, and the same fit succeeding with room), the limits compared before allocation, the node and the memory ceilings at
their boundaries, the hand-worked trees (a classification split between 2 and 3 with predictions 0 and 1; regression leaves
2.0 and 12.0), the unsplittable pair (probability 0.5, class 0), the missing rule (default side learned right when the missing rows
are class 1, left when they are class 0; a NaN stored in a missing cell never read), the tie between the two sides (variant 0 wins),
the aggregation order, the bytes at 1 to 7 threads for both tasks, the model reader's refusals, the bins and the radix order
(`-0.0` before `0.0`) and the bitmap layout.

### 10.8 G-ref against scikit-learn: **the pre-registered gate FAILED for the regression forest R1; the cause is attributed, not fixed by moving the tolerance**

`scripts/gate_sklearn.py --big` (output: `docs/bench/gate_sklearn.txt`), scikit-learn 1.8.0 `RandomForestClassifier` /
`RandomForestRegressor`, `n_jobs = 1`, `random_state = 0`, matching trees, depth, `max_features`, `min_samples_leaf`, bootstrap, the
same split. **A different algorithm** (exact thresholds on raw values, its own RNG, a feature subset redrawn when no split exists).
Tolerances fixed in section 8 before the first run: AUC within 0.005, RMSE within 3% relative.

| config | rows | metric | cancho | scikit-learn | distance | within | control (depth-1 forest) distance |
|---|---|---|---|---|---|---|---|
| C1 | 100,000 | AUC | 0.826189 | 0.826105 | 0.000084 | yes | 0.0844, outside |
| C2 (missing) | 100,000 | AUC | 0.865853 | 0.866141 | 0.000288 | yes | 0.0681, outside |
| C3 (1 tree) | 100,000 | AUC | 0.767285 | 0.767211 | 0.000074 | yes | 0.1324, outside |
| C1 | 1,000,000 | AUC | 0.826176 | 0.827210 | 0.001033 | yes | 0.0833, outside |
| R1 (missing) | 100,000 | RMSE | 2.784408 | 3.029622 | **8.1%** | **NO** | 12.8%, outside |
| R2 (1 tree) | 100,000 | RMSE | 3.045111 | 3.117527 | 2.3% | yes | 7.1%, outside |

The classification forests agree to 0.001 AUC or better (spike F's 1,000,000-row figure is reproduced: 0.826176 against 0.827210). The
depth-1 control is far outside the tolerance in every row, so the gate could fail. **R1 fails: the library's forest is 8.1% better
(lower RMSE) than scikit-learn's.** Diagnosis, run after the failure and labelled as such in the script: scikit-learn fitted on the
bin indices of the library's own edges (the same candidate thresholds) gives 2.8737 (3.1% from cancho; without missing cells,
2.6986 against cancho's 2.6829, 0.6%), and on the raw values 3.0037 without missing cells (the forest of raw values is worse than the
one of bin indices at the same depth). So most of the gap is the thresholds (quantile bins against every value) on this target,
which has a product term `x3 x5` whose extreme values an exact splitter isolates; the remaining 3.1% with missing cells is **not
attributed** (the two handle missing cells differently: they both learn a side per split, scikit-learn also considers the
split of present against missing cells, which the library does not: section 6 says the missing rows travel with a threshold
split). Nothing here shows the library to be wrong (it is bit-identical to an independent oracle of its own specification, 10.2);
it shows the 3% rule of section 8 was a bad pre-registration for a different algorithm on a target like this one. **The maintainer
decides** whether to keep the rule (and the gate stays red for R1), to compare against scikit-learn on bin indices only, or to drop
the regression comparison. At 1,000 rows the classification AUC differs by 0.017 (scikit-learn is lower); the 1,000-row rows are
reported, not gated.

### 10.9 Timing (`scripts/bench_tree.py`, `scripts/bench_report_tree.py`; raw: `docs/bench/tree_<size>.{txt,jsonl}`)

Minimum of 7 interleaved rounds (each round runs every entry once, in turn); spread = maximum over minimum. Classification, depth 8,
64 bins, `mtry` 4, bootstrap, 20 columns, train = first 80% of the rows, test = last 20%. cancho fit includes the edges, the bin
matrix, the thread copies and the merge; predict includes the piece copies. cancho times are read in the program (`clock_ms`): 100,000
rows: mean of 10 fits / 10 predicts; 1,000,000 rows: 1 fit, mean of 3 predicts. scikit-learn times: `perf_counter` around `fit` and
`predict_proba`. **Two sessions per size, each of 7 rounds; the load average (1 min) over the rounds: 100,000 rows 3.3 to 4.9 (first) and 5.2 to 8.4
(replicate); 1,000,000 rows 4.9 to 11.9 (first) and 2.6 to 5.0 (second). All flagged.** The figures at 16 threads on a loaded machine
with 12 performance cores should be read as "no worse than", not as a scaling law.

**100,000 rows** (train 80,000, test 20,000), the first session (lower load). size 100000; 7 rounds; load average (1 min) at the start / end of each round: 3.3/3.6 3.6/3.5 3.5/3.5 3.5/3.6 3.6/3.5 3.5/4.6 4.6/4.9
(* spread above the 1.25 of the protocol; the cell was not rerun)

| entry | trees | threads | fit min (s) | spread | fit speedup | predict min (s) | spread | predict speedup |
|---|---|---|---|---|---|---|---|---|
| cancho | 10 | 1 | 0.0516 | 1.05 | 1.00x | 0.0092 | 1.07 | 1.00x |
| cancho | 10 | 2 | 0.0260 | 1.04 | 1.98x | 0.0052 | 1.04 | 1.77x |
| cancho | 10 | 4 | 0.0158 | 1.04 | 3.27x | 0.0027 | 1.07 | 3.41x |
| cancho | 10 | 8 | 0.0121 | 1.08 | 4.26x | 0.0016 | 1.06 | 5.75x |
| cancho | 10 | 16 | 0.0100 | 1.09 | 5.16x | 0.0013 | 1.31 * | 7.08x |
| cancho | 32 | 1 | 0.1197 | 1.03 | 1.00x | 0.0362 | 1.12 | 1.00x |
| cancho | 32 | 2 | 0.0605 | 1.04 | 1.98x | 0.0200 | 1.05 | 1.81x |
| cancho | 32 | 4 | 0.0312 | 1.04 | 3.84x | 0.0104 | 1.04 | 3.48x |
| cancho | 32 | 8 | 0.0188 | 1.07 | 6.37x | 0.0055 | 1.04 | 6.58x |
| cancho | 32 | 16 | 0.0181 | 1.06 | 6.61x | 0.0039 | 1.33 * | 9.28x |
| scikit-learn RandomForestClassifier (exact splits) | 10 | 1 | 1.0600 | 1.10 |  | 0.0092 | 1.18 |  |
| scikit-learn RandomForestClassifier (exact splits) | 10 | 16 | 0.1316 | 1.13 |  | 0.0134 | 1.23 |  |
| scikit-learn HistGradientBoosting, 1 iteration (context only) | 1 | 1 | 0.1055 | 1.08 |  | 0.0007 | 1.15 |  |

The replicate session of the same rows (load 5 to 8), minimum fits and predicts only. size 100000_b; 7 rounds; load average (1 min) at the start / end of each round: 8.4/8.1 8.1/7.5 7.5/6.6 6.6/6.1 6.1/5.8 5.8/5.6 5.6/5.2

| entry | trees | what | 1 thread | 2 | 4 | 8 | 16 | worst spread |
|---|---|---|---|---|---|---|---|---|
| cancho | 10 | fit (s) | 0.0473 | 0.0238 | 0.0151 | 0.0120 | 0.0096 | 1.07 |
| cancho | 10 | predict (s) | 0.0081 | 0.0046 | 0.0027 | 0.0016 | 0.0013 | 1.15 |
| cancho | 32 | fit (s) | 0.1093 | 0.0563 | 0.0304 | 0.0187 | 0.0176 | 1.07 |
| cancho | 32 | predict (s) | 0.0339 | 0.0175 | 0.0101 | 0.0055 | 0.0039 | 1.33 |
| scikit-learn RF | 10 | fit (s) | 1.0163 |  |  |  | 0.1347 | 1.05 |
| scikit-learn RF | 10 | predict (s) | 0.0087 |  |  |  | 0.0161 | 1.02 |
| scikit-learn HGB 1 iteration | 1 | fit (s) | 0.1000 |  |  |  |  | 1.04 |
| scikit-learn HGB 1 iteration | 1 | predict (s) | 0.0006 |  |  |  |  | 1.07 |

**1,000,000 rows** (train 800,000, test 200,000), the second session (lower load, tighter spreads). size 1000000; 7 rounds; load average (1 min) at the start / end of each round: 5.0/4.8 4.8/4.5 4.5/3.8 3.8/3.0 3.0/2.6 2.6/3.2 3.2/3.5
(* spread above the 1.25 of the protocol; the cell was not rerun)

| entry | trees | threads | fit min (s) | spread | fit speedup | predict min (s) | spread | predict speedup |
|---|---|---|---|---|---|---|---|---|
| cancho | 10 | 1 | 0.5830 | 1.04 | 1.00x | 0.0863 | 1.11 | 1.00x |
| cancho | 10 | 2 | 0.3060 | 1.07 | 1.91x | 0.0453 | 1.11 | 1.90x |
| cancho | 10 | 4 | 0.2330 | 1.06 | 2.50x | 0.0263 | 1.03 | 3.28x |
| cancho | 10 | 8 | 0.2070 | 1.05 | 2.82x | 0.0147 | 1.02 | 5.89x |
| cancho | 10 | 16 | 0.2210 | 1.02 | 2.64x | 0.0103 | 1.29 * | 8.35x |
| cancho | 32 | 1 | 1.2890 | 1.04 | 1.00x | 0.3563 | 1.03 | 1.00x |
| cancho | 32 | 2 | 0.7590 | 1.06 | 1.70x | 0.1807 | 1.03 | 1.97x |
| cancho | 32 | 4 | 0.5710 | 1.01 | 2.26x | 0.1020 | 1.02 | 3.49x |
| cancho | 32 | 8 | 0.3860 | 1.04 | 3.34x | 0.0533 | 1.01 | 6.68x |
| cancho | 32 | 16 | 0.3810 | 1.06 | 3.38x | 0.0337 | 1.25 | 10.58x |
| scikit-learn RandomForestClassifier (exact splits) | 10 | 1 | 12.6230 | 1.03 |  | 0.0845 | 1.05 |  |
| scikit-learn RandomForestClassifier (exact splits) | 10 | 16 | 1.8624 | 1.02 |  | 0.0139 | 1.30 * |  |
| scikit-learn HistGradientBoosting, 1 iteration (context only) | 1 | 1 | 0.6020 | 1.02 |  | 0.0058 | 1.20 |  |

The first session of the same rows (load 5 to 12). size 1000000; 7 rounds; load average (1 min) at the start / end of each round: 4.9/5.5 5.5/6.0 6.0/6.4 6.4/7.5 7.5/10.1 10.1/9.7 9.7/11.9

| entry | trees | what | 1 thread | 2 | 4 | 8 | 16 | worst spread |
|---|---|---|---|---|---|---|---|---|
| cancho | 10 | fit (s) | 0.6390 | 0.3350 | 0.2220 | 0.2160 | 0.2340 | 1.19 |
| cancho | 10 | predict (s) | 0.0940 | 0.0517 | 0.0277 | 0.0150 | 0.0113 | 1.32 |
| cancho | 32 | fit (s) | 1.4780 | 0.7770 | 0.4950 | 0.3700 | 0.3890 | 1.23 |
| cancho | 32 | predict (s) | 0.4050 | 0.2070 | 0.1037 | 0.0540 | 0.0343 | 1.28 |
| scikit-learn RF | 10 | fit (s) | 15.7432 |  |  |  | 1.9526 | 1.33 |
| scikit-learn RF | 10 | predict (s) | 0.0999 |  |  |  | 0.0152 | 2.00 |
| scikit-learn HGB 1 iteration | 1 | fit (s) | 0.6702 |  |  |  |  | 1.06 |
| scikit-learn HGB 1 iteration | 1 | predict (s) | 0.0059 |  |  |  |  | 1.17 |


**What the comparison is, and is not.** scikit-learn `RandomForestClassifier` is an **exact-split** forest: a different algorithm,
so its row is the time to produce a model of the same family, not the speed of the same algorithm, and **no speed claim is made
against it** (the columns are side by side; the quality agreement is 10.8). A histogram forest would be a fair competitor: the
installed Python packages were checked (`lightgbm`, `xgboost`, `catboost`, `polars`, `h2o`, `tensorflow`, `jax` are absent;
scikit-learn, numba, pandas, torch are present) and **no histogram random forest is installed**, so there is none to compare
with; nothing was installed. `HistGradientBoostingClassifier` with one iteration is a histogram tree, but gradient boosting (leaf-wise
growth, second-order leaf values, no bagging, no feature subsets, 63 bins here to match), so its row is context only: it builds one
tree where the library builds 10 or 32. The same machine, the same data, one thread each (`OMP_NUM_THREADS` and the BLAS variables 1).
That scikit-learn's `n_jobs = 1` and the histogram implementation really ran on one thread was not checked with a counter.

**Where the time goes** (the macOS `sample` profiler, 1 ms, one run of 6 fits and 1 predict at 1,000,000 rows, 1 thread, T = 10, load 3
to 8, taken before the last two small changes (the partition and the 64-entry search); the share of 1,268 samples): the histogram
pass 28%, the rest of the tree growing (partition, scans, bootstrap) 31%, the radix sort of the edges 17%, the conversion of values to
sort keys 11%, the bin search 6%, prediction 5%. The binning work a 1-thread fit does serially, measured once by the driver
(`phase=1`): edges 0.221 s and bin matrix 0.038 s at 1,000,000 rows (0.018 s and 0.004 s at 100,000); with threads they are done
column by column in threads.

**Thread scaling and its limit.** Speed-up of the fit against the same configuration at 1 thread (minimum of 7), T = 32: 100,000 rows
1.98x, 3.84x, 6.37x, 6.61x at 2, 4, 8, 16 threads (replicate session 1.94x, 3.60x, 5.84x, 6.21x); 1,000,000 rows 1.70x, 2.26x, 3.34x,
3.38x (first session 1.90x, 2.99x, 3.99x, 3.80x). T = 10 does not divide by 4, 8 or 16 (groups of 2 or 3 trees; at most 10 threads busy): at
1,000,000 rows 1.91x, 2.50x, 2.82x, 2.64x. Prediction scales better (the piece copies are small against the work): 10.6x at 16 threads
at 1,000,000 rows, 9.3x at 100,000 (T = 32). Read as one number, the 1,000,000-row T = 32 fit behaves like a program with about a quarter
of its 1-thread time not parallel (an Amdahl fit to 3.38x at 16 threads: a fit of the table, not a measurement of any part).
**What I think bounds it is the job creation, not the algorithm** (an explanation by elimination: the trees are independent and the binning
is in threads; the cost itself was not measured): cancho gives a thread one pointer and no shared read, so the parent builds, for each
thread, a copy of the bin matrix (16 MB at 800,000 rows), the labels and the `n`-sized scratch (about 13 MB), sequentially, while the earlier
threads already run (`docs/upstream/spawn_shared_slice.cho`). The bytes are the same at every thread count (10.4).

**Memory.** A fit of 800,000 x 20 at 16 threads holds about 16 x (16 MB + 19 MB) of copies besides the matrix; the ceiling of
section 5 (2^34 bytes) refuses what would exceed it.

### 10.10 What the work found (corrections and findings, in the order they were made)

* **A `region` is one 64 KiB arena chunk.** The first 100,000-row fit died with SIGILL and no message although 1,000 rows worked; the
  cause was `alloc_slice` of 80,000 values in a `region` (`docs/upstream/region_trap.cho`). All large scratch is on the heap now.
* **A bin search written naturally was 12x slower than the one the optimiser can see through** (`docs/upstream/bounds_branch.cho`,
  33 ms against 2.8 ms and 5.4 ms for 80,000 x 20 cells). Measured at load 3 to 10 in the middle of the work, not in the protocol of
  10.9: the bin matrix of 100,000 rows 32 -> 5 ms and the fit 94 -> 64 ms; at 1,000,000 rows, 54 -> 37 ms after the specialisation
  of the search for at most 64 bins.
* **The first reader checked more than it needed.** Three of its checks survived their mutants; they were implied by the stack rule
  (section 7) and were removed; two other mutants survived because the fuzz judged only "no trap", and the fuzz now judges the answers.
* **A mutant of my own list did nothing** (`bb - bb / (bb + 1)` is `bb`; found by reading it, replaced), and the mutant runs found
  **cases the large inputs never reach**: an exact tie between the two sides for missing rows that wins (needs a node
  of three rows), and a regression node whose targets are all equal but whose rows differ (needs integer targets). Both are in the
  edge-case set or the unit tests now.
* **The target of R1 favours binned thresholds** over exact ones at the same depth (10.8), a result about the target, not about the code.
* **Scaling is bounded by job creation**, not by the algorithm (10.9), and cancho's thread payload is the cause.

### 10.11 What this does not establish

* Only this Mac, aarch64; no second machine (D2 across machines is **not tested** for the forest; the arithmetic is integer, IEEE
  `+ - * /` without `fma`, and `fbits`, so the expectation is the same bytes, unverified). The gram host was not used.
* The oracle identity is at 1,000 and 100,000 rows; the 1,000,000-row identity was not run. The thread equality at 1,000,000 rows is by hash of
  the benchmark runs, without an oracle.
* The timings were taken on a loaded, shared machine (3 to 12 load average); 16-thread cells in particular are upper bounds.
  scikit-learn's thread use was not checked with a counter.
* Quality against scikit-learn: the regression tolerance fixed in advance is not met on R1 (10.8); nothing was run on real data beyond the
  synthetic `ml-data`.
* A tree fit is limited to 2^20 rows (the exact comparison); the cap that `ml` uses (10,000,000 rows) is above it, so the verbs will need
  the sampled-fit repair or a wider comparison before they call the library. *(W3a: `ml` counts the training rows before it copies them and refuses a tree or forest fit above 2^20 with
  `ml.input.too-many-rows` and a hint to sample or to hold rows out with `--split`; `docs/wiring-trees.md` section 6. The wider comparison is still not built.)*
* The memory ceiling of the thread copies (2^34 bytes) and the node ceilings (2^22 per tree, 2^24 per forest) are provisional.
* Not built: the command line verbs, multiclass, a model-file envelope around the payload (the checksum, the plan, the schema), feature
  importance, out-of-bag scores.


## 11. API and effect rows

`tree.forest.fit(heap, params, n, d, xs, bm, ys, rows_cap, cols_cap, cells_cap) -> (Forest, code)` with row `[heap, conc]`;
`forest.drop` `[heap]`; `tree.predict.predict(heap, forest, xs, bm, rows, d, threads, out) -> code` and `predict_class`, `[heap, conc]`;
`tree.model.write(heap, forest) -> Buffer` and `model.read(heap, bytes, max_total) -> (Forest, code)`, `[heap]`; `tree.params.check`,
`tree.bins.*` (except the threaded `tree.pbins.run`, `[heap, conc]`) and `tree.exact.gt` have the empty row. The compiler checks a row
exactly, so this is also the authority a program that uses the library needs: **heap and conc, and nothing else** (no clock, no
file, no console, no network). `res struct Forest` holds three boxes (`toff`, `ni`, `nv`) and eight sizes, and is dropped by the caller; a refusal
returns an empty forest that must be dropped too.

## 12. Upstream gaps found in this stage (reproducers in `docs/upstream/`)

| file | gap | cost here |
|---|---|---|
| `mulhi.cho` | no multiply-high: the exact Gini comparison forms 98-bit products in 30-bit limbs (`tree.exact`) | effort and review, not time; and the 2^20 row ceiling (section 3.1) |
| `sort_std.cho` | no sort in `std` | a 50-line radix sort; it is 17% of a fit at 1,000,000 rows (profile below) |
| `lshr.cho` | `>>` is arithmetic: every SplitMix64 shift is `>> k & mask` | a place for a wrong mask |
| `clock_resolution.cho` | `clock_ms` has 1 ms resolution: 182 of 200 runs of a 0.2 ms job read 0 | a cell under ~100 ms is repeated in the program and divided |
| `region_trap.cho` | a `region` is one 64 KiB arena chunk; `alloc_slice` beyond it is a silent trap, **at a size that depends on the data** | found by bisection after the fit died at 100,000 rows and ran at 1,000 |
| `dotdot_path.cho` | `open_read` on a path with `..` is SIGILL, not `Opened::Failed` | the driver needs absolute paths |
| `bounds_branch.cho` | a guarded bounds-checked load makes the bin search 12x (6x against the masked form) slower | measured, section 10.9 |
| `copy_into_float.cho`, `spawn_shared_slice.cho`, `test_runner_conc.cho` | (met again) float `copy_into`; no shared read-only slice for threads; `cancho test` cannot spawn | job creation is the serial part of a threaded fit (10.9); the thread tests are a program |

Frictions that are not gaps: `let S { a: b } = x` does not rename; `val` is a keyword; `main` and helpers must carry exact rows
(`err_write` in a row that does not perform it is an error); a boolean test of a `res` value must be consumed on every path.

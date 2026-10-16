# cancho-ml interpretability: feature importance and a dump of a tree (issue 44, milestone M23)

**Status: this document is written before the code it judges (the first commit of the branch holds only this file).** Sections 1 to 9 fix what the model files hold, the
definitions, the verbs, the answers, the refusals and the gates. Section 10 (results) is filled in after the gates have run, with only what a script of this repository printed
and its conditions. Tags as in `design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from a document or from the code, **[measured]** printed by a
command whose line is given, **[open]** not decided. The text of the issue (alpibrupa/cancho-ml#44) is the scope.

An outside review put the problem in one sentence: *an agent that cannot explain a forest can only trust the score*. This document gives `ml` two verbs that explain a fitted
model without a second tool: `ml importance` (which columns the model relies on, and by how much) and `ml dump` (what the model is, as text an agent can read).

## 1. Decisions in one table

| # | question | decision | why |
|---|---|---|---|
| D1 | does the tree payload hold what an impurity importance needs | **yes, for both tasks, and the file does not change** (section 2.2): every node, internal or leaf, stores its row counts (`na`, `nb`); a classification node stores the class-0 and class-1 counts of the rows that reached it, a regression node its row count, and a regression leaf its mean | the Gini decrease of a split is a function of the six counts of a node and its two children; the variance reduction is a function of the two child row counts and the two child means, and a child mean is the count-weighted mean of the leaves below it. No byte of any model file moves, so no model digest and no golden re-keys |
| D2 | which importances exist | (a) **standardised coefficients** for linear and logistic models; (b) **impurity decrease** (mean decrease in impurity, the quantity scikit-learn calls `feature_importances_`) for trees and forests, computed from the stored counts; (c) **split counts** as a second, differently named table beside (b); (d) **permutation importance** on a held-out file (`--data`), for every model with a target | (a) and (b) need no data, so `describe` and `GET /model` carry them; (c) is NOT an impurity importance and is named `splits` and `split_rows`; (d) needs data and measures what a model loses when a column carries no information |
| D3 | a file change | **not made.** Section 3.3 says what a payload version would add (the exact node impurity and the exact node sum of the target for a regression) and leaves it as an open question for the maintainer | the importance of D1 is exact for classification and exact up to the rounding of the stored leaf means for regression; a new payload would not remove that rounding from any file already written |
| D4 | what a table row is | one row per **column the model sees** (after the steps), with the **raw input column** it comes from in `source`; the permutation table has one row per **raw input column** | a one-hot column `c#2` has a coefficient of its own, a standardised coefficient of a sum of indicators is not a number anyone can read; a permutation breaks a raw column and re-applies the stored steps, so it can only be per raw column |
| D5 | the std in a standardised coefficient | the standard deviation of the **column the model sees**, over the training rows (population, `ddof` 0), taken from the plan's own `scale standard` step when that step made the column (it is 1 by construction), else the provenance section's std of the raw column divided by the stored divisors of the steps; **refused with `ml.importance.no-scale` and a repair when neither is stored** (a file fitted before run tracking, or a column made by an encoder) | no number is guessed; `describe` (which must keep answering for old files) says `available: false` and the tag instead of failing |
| D6 | the stream of a permutation | stream kind **11**: `derive(derive(derive(seed, 11), repeat), column)`, then the Fisher-Yates shuffle of `lib/metrics/splits.cho` (`shuffle`) over the row indices, where `column` is the **index of the raw column in the model's schema**; kinds 9 and 10 are the draft boosting PR's (#62) and are listed as reserved so that the registry stays a contiguous list | the thread count, the order of the columns in the data file and the order of the loop never reach a draw; the schema index does not depend on the data file |
| D7 | the metric difference | `--metric` names the metric; a **loss** (`rmse`, `mae`, `log_loss`) gives `permuted - baseline`, a **score** (`r2`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc`) gives `baseline - permuted`; so a **larger importance always means the model lost more**; the answer states `direction`. Per column: the mean over the repeats (sequential sum, repeat 0 first) and the population standard deviation (two passes) | one reading rule for every metric |
| D8 | where the code lives | a function per model kind behind one dispatch (`interp.table`): arms for the kinds 4 and 5 (standardised), 7 (impurity), and the place for kind 8 (boosting, draft PR #62: its payload has the same node layout and a per-node gain it can store, so its function is `boost_table` and one new arm; nothing else changes) | boosting plugs in later without touching the verbs |
| D9 | no new capability | `importance` and `dump` read files the way `eval` does; `ml-predict` and `ml-serve` gain the data-free table only, in `describe` (they read nothing new) | `scripts/check_bins.py` and `manifest.py --check` stay unchanged for the three authority rows |
| D10 | threads | `--threads` is accepted and checked, as in `eval`; nothing in `importance` is threaded, and the gate shows the bytes equal at 1 and at 8 | the stream design (D6) already makes the result independent of the order of work, so a threaded loop can be added later without changing a byte |

## 2. What the model files hold (read from the payloads and the code, then measured)

### 2.1 The kinds, one by one **[read]**

| kind | model | `model` section | what an explanation can use |
|---|---|---|---|
| 1 to 3 | constant | the constant(s) of `mfile.cho` | nothing per feature: no importance, no tree. `importance` and `dump` refuse with `ml.importance.unsupported-model` and `ml.dump.unsupported-model` |
| 4 | linear | `coef` `f64 [d]`, `intercept` `f64 [1]` (`docs/wiring.md` section 4) | the coefficients, in the units of the **column the model sees**; the steps (`scale` stores mean and divisor per column) and the provenance (section 7 of the file: count, min, max, mean and population std of every numeric raw column, before the steps) give the unit |
| 5 | logistic | the same plus the two labels | the same; the coefficients are log-odds per unit |
| 6 | k-means | `centroids` `f64 [k, d]` | the centroids, in the space the model sees; with the stored steps a centroid can be shown by raw column when the steps are scalings (section 4.3). No importance: a clustering has no target to lose |
| 7 | tree, forest | ten records (`docs/models-tree.md` section 7): `header` (task, trees, d, bins, total_nodes, max_depth, min_leaf, mtry), `tree_off`, then per node `feat`, `bin`, `dleft`, `left`, `right`, `na`, `nb`, `val` | section 2.2 |

### 2.2 The tree payload, measured **[measured]**

A tree and a regression tree were fitted on `docs/examples/data/iris_train.csv` (75 rows) and their payloads read with `scripts/oracle_st.py` and the layout of
`docs/models-tree.md` section 7 (no code of `tools/ml`; the reader is 15 lines of Python). The root of the classification tree (`--task classification --max-depth 3`) and its two children:

```
node 0  feat 3  left 1   right 8   na 37  nb 38   val 1.8      (internal: val is the threshold)
node 1  feat 2  left 2   right 5   na 36  nb 5                  (the left child)
node 8  feat 2  left 9   right 12  na 1   nb 33                 (the right child)
```

`37 + 38 = 75` rows reach the root, `36 + 1 = 37` class-0 rows and `5 + 33 = 38` class-1 rows go to the children: **an internal node of a classification tree stores
the class counts of the rows that reached it.** In the regression tree (`--task regression --max-depth 2`) the root has `na = 75` and its children `37` and `38`: **an internal
node of a regression tree stores its row count**, a leaf stores its count and `val` = the mean of its rows (`docs/models-tree.md` section 2.5). What is **not** stored: the node
impurity, the node sum of the target, the gain, a sample weight. With a bootstrap the counts are the in-bag multiplicities, because that is what the tree was grown on.

Consequences, all derived in section 3.2 and checked by the gates:

* the **Gini decrease of a classification split** needs only `(na, nb)` of the node and of its two children: exact, no rounding of an input;
* the **variance reduction of a regression split** is `nL nR / n (mL - mR)^2`, where `mL` and `mR` are the means of the rows of the two children. A child that is a leaf stores
  its mean; a child that is an internal node has the mean `(sum over its leaves of count * mean) / count`, a sum of values already rounded once when the tree was grown, so
  the result differs from the training value by the rounding of the stored leaf means (a relative `1e-16` per leaf) and by nothing else.

### 2.3 What an old file lacks

A file fitted before run tracking has no provenance section; a file fitted without a `scale standard` step has no stored unit for its columns. Standardised coefficients of such
a file are refused with a tag and a repair (D5). The impurity importance of a tree needs no section beyond the model, so every tree file has it.

## 3. The definitions (normative; `scripts/oracle_interp.py` implements exactly this text)

Notation: a model sees `w` columns (after the steps); `source(j)` is the raw column of column `j` (a column made by an encoder has the raw column it encodes); floats are IEEE
double, every sum is **sequential from +0.0 in the order stated**, there is no `fma`, a quotient is one rounding.

### 3.1 Standardised coefficients (kinds 4 and 5)

For model column `j` with stored coefficient `b_j`:

* **sd_j**, the standard deviation of the column the model sees over the training rows: (1) if the last step that changed the column is `scale standard`, then `1.0` for `ddof` 0
  and `sqrt((n - 1) / n)` for `ddof` 1 with `n` the training rows (the scaler divides by the deviation of its own input, so the deviation of its output is known without data),
  or `0.0` when the provenance says the raw column is constant; (2) else, when the provenance section holds the raw column, `std_raw / D_j`, where `D_j` is the product of the stored
  divisors of every `scale` step applied to the column (a `minmax` divisor is the range; no step: `D_j = 1.0`); an `impute` step does not change `D_j` and the provenance std is that of the **present** cells
  (stated in the answer as `std_source: "provenance"`); (3) else the column has no stored deviation (an encoder made it, or the file has no provenance): `ml.importance.no-scale`.
* `standardised_j = b_j * sd_j` (one rounding); `|standardised_j|` is the magnitude of the effect of one standard deviation of the column.
* **importance_j** (the "share", what ranks): `|standardised_j| / T` with `T = |standardised_0| + |standardised_1| + ...` in column order from +0.0; `total` is `1.0`, or `0.0` when `T` is 0
  (every coefficient 0: every share is `0.0`).
* **rank**: 1 for the largest `importance`; ties go to the **lower column index** (a stable sort on `(-importance, j)`): ranks are distinct, 1 to `w`.

What this measures: how many standard deviations of the model's output (for a linear model: units of the target; for a logistic model: log-odds) a one-standard-deviation change of
the column moves it, **with the other columns held fixed**. It is not a measure of what the model would lose without the column (that is permutation importance), and with correlated
columns the coefficients are shared arbitrarily between them.

### 3.2 Impurity decrease (kind 7, trees and forests) **[choice]: the scikit-learn definition**

For a tree, take its nodes in preorder. For an **internal** node with counts `(a, b)` (classification: class 0 and class 1; regression: `a` is the row count and `b = 0`), left child
`(aL, bL)` and right child `(aR, bR)`, and `n = a + b` (classification) or `n = a` (regression), `nL`, `nR` likewise:

* classification, with `D = bL * nR - bR * nL` (an exact integer, `|D| <= 2^40`): `dec = 2.0 * (float(D) * float(D)) / (float(n) * (float(nL) * float(nR)))`.
  This is `n G - nL G_L - nR G_R` for the Gini impurity `G = 2 f (1 - f)` (`f` the class-1 fraction), written without a difference of two large quotients, so it is never negative.
* regression: `S` of a node is the sum of the target over its rows as the leaves state it: a leaf's `S = float(count) * val`, an internal node's `S = S_left + S_right` (a pairwise tree sum, the
  recursion of the tree). `m = S / float(n)`. `dec = (float(nL) * float(nR) / float(n)) * ((mL - mR) * (mL - mR))`, left to right.
* the feature of the node gets `dec` added in preorder order: `t[f] = t[f] + dec`, from +0.0.

A tree's importance vector is `t[f] / (t[0] + t[1] + ...)` (features ascending) when the sum is positive, else all zeros. A forest's is the tree vectors **added in tree order**, then divided by the sum of the result
(features ascending); `total` is `1.0`, or `0.0` when no tree splits. This is `RandomForest*.feature_importances_` (average of the normalised tree vectors, renormalised) computed
from our counts.

The second table, **named differently because it measures something else**: `splits[f]` = the number of internal nodes, in all trees, whose feature is `f`; `split_rows[f]` = the sum over
those nodes of the node's row count `n` (the stored counts, so bootstrap multiplicities). "How often and how high in the tree a feature splits" is what these two say. They are integers and are
printed as integers. They are not an importance and are not normalised.

**Ties** (all deterministic): the rank is a stable sort on `(-importance, j)`; two columns with exactly equal importance (typically both 0.0) are ordered by column index.

**What the number is and is not.** It is computed on the training rows the trees were grown on (in-bag, with the bootstrap multiplicities), so it is biased towards columns with many distinct
values and it gives a pure-noise column a small positive share (the noise gates of section 8 measure that). It is not what the model would lose on new data: that is permutation importance.

### 3.3 What a file change would add **[open]**

A payload version (readers keep reading the old one, as `docs/tracking.md` section 2.1 does for the provenance section) could store per node the exact impurity (Gini or variance) and the exact
sum of the target; then the regression importance would be exact to the bit instead of exact up to the rounding of the leaf means, and a boosting model could store its gain. The cost is the
one-way door of every file-format change in this repository: the digest of every model file changes (D2 of `docs/tracking.md`), so every golden is re-recorded in one commit. The measured
size of the rounding (section 10) is the evidence for the maintainer's decision; nothing here depends on it.

### 3.4 Permutation importance (`--data FILE`; any model with a target)

Inputs: the model file, a CSV with the model's columns **and the target**, a metric, `R` repeats (`--repeats`, 1 to 100, default 5), a seed (`--seed`, default 0).

1. Read the file as `eval` does (same reader, same schema check, same `--drop`); all `n` rows are used (`--split` is not a flag of `importance`). A missing target cell, or a classification label the model does not know, is `eval`'s refusal.
2. **Baseline** `B` = the metric of the model's predictions on the rows as read.
3. For every **raw feature column `c`** of the schema (numeric or categorical, in schema order) and every repeat `r = 0 .. R-1`:
   * the shuffle: `a = [0 .. n-1]`; `splits.shuffle(a, 0, n, derive(derive(derive(seed, 11), r), c))` (Fisher-Yates from the end, rejection sampling, `docs/metrics.md` 2.1), where `c` is the schema index of the column;
   * the column in the matrix (or in the id matrix for a categorical column) is replaced by itself permuted: row `i` takes the **cell** (value, or missing flag, or category id with its unseen flag) of row `a[i]` of the **same file**;
     a missing cell travels with the shuffle, so the number of missing cells is unchanged;
   * the stored steps are applied to the permuted raw input (the encoders and scalers see the permuted column, as they would see any column), the model predicts, the metric is computed: `P[c][r]`;
   * `d[c][r] = P - B` for a loss and `B - P` for a score (D7).
4. Per column: `mean = (d[0] + d[1] + ...) / R` (sequential from +0.0, repeat 0 first); `std = sqrt(((d[0]-mean)^2 + (d[1]-mean)^2 + ...) / R)` (population, `R = 1` gives 0.0). A column the model does not read has `d = 0.0` exactly.
5. Rank: a stable sort on `(-mean, c)`.

Metrics: regression `rmse` (default), `mae`, `r2`; classification `roc_auc` (default), `accuracy`, `precision`, `recall`, `f1`, `log_loss`; the library's own definitions (`lib/metrics`), with the model's positive label.
A metric with no value on the baseline or on a permuted run (a precision with no predicted positive, an R-squared of a constant target) is `ml.importance.undefined-metric`.

A permutation importance is **per raw input column** (D4): a one-hot encoded column is permuted as the raw column it came from, and the table says so (`source` is the column itself; the answer's
`columns` member of the data-free table names the model columns). A column the model reads through several encoders gets one row.

**What it measures**, stated so that nobody reads more: the drop of the metric when this column's values are exchanged between rows of the file, with every other column untouched. It is a property of
(model, file): on a file where two columns are copies of each other, shuffling one leaves the other to carry the signal and both look unimportant. On the file the model was trained on it measures
what the model memorised.

## 4. The verbs

```
ml importance --model-file M [--data FILE [--metric NAME] [--repeats R] [--seed S] [--drop COLS]] [--format text]
ml dump       --model-file M [--tree I] [--max-nodes N] [--format text]
```

### 4.1 Flags **[choice]** (the table is `ml introspect | jq .flags`; the gates check it)

| flag | verbs | meaning |
|---|---|---|
| `--data FILE` | importance | the held-out CSV for permutation importance (a path under `--root`, read as `eval` reads its file) |
| `--metric NAME` | importance | section 3.4; default `rmse` (regression) or `roc_auc` (classification); a name that does not fit the task is `ml.importance.bad-metric` |
| `--repeats R` | importance | permutations per column, 1 to 100 (the flag `cv` already has; the range is the verb's) |
| `--seed S` | importance | the seed of the permutation streams (an existing flag) |
| `--max-work N` | importance | refuse (`ml.limit.work`) a permutation run that would score more than `N` row evaluations (`rows * raw columns * repeats`), before the first shuffle; default 4,294,967,296, ceiling 2^40 (the limits table, like `--max-cells`) |
| `--tree I` | dump | the tree to print, 0-based (0 for a tree); `ml.param.out-of-range` when `I >= trees` |
| `--max-nodes N` | dump | refuse (`ml.limit.nodes`) a tree with more than `N` nodes, **before** the node list is built; default 10,000 when not given, ceiling 4,194,304 (the flag `fit` has) |
| `--drop`, `--max-rows`, `--max-columns`, `--max-cells`, `--max-line-bytes`, `--max-bytes`, `--threads`, `--log`, `--log-max-bytes`, `--format`, `--root`, `--model-file` | as `eval` | `--threads` is accepted and changes no byte (D10) |

The model file names its target, so `importance` takes no `--target` (the issue's sketch lists one; `eval` reads the target of the file by the stored name, and a second spelling of the same
fact could only disagree with it).

### 4.2 The answer of `importance`

`data` (one JSON object; doubles are strings in the shortest decimal that reads back, as `provenance` is; counts are integers):

```
{"model":"forest","task":"classification","kind":7,
 "importance":   {"method":"impurity_decrease","available":true,"measure":"...","trees":T,"total":"1.0",
                  "columns":[{"column":"x1","source":"x1","importance":"0.5","rank":1,"splits":12,"split_rows":340}, ...]},
 "permutation":  null | {"metric":"roc_auc","direction":"score","repeats":5,"seed":0,"rows":300,"baseline":"0.97",
                  "columns":[{"column":"x1","importance":"0.31","std":"0.02","rank":1}, ...]}}
```

`importance` is **the same object `describe` prints** (D5 and section 6) and a gate compares the bytes. For a linear or logistic model `method` is `standardised_coefficient` and a row is
`{"column","source","coefficient","std","std_source","standardised","importance","rank"}`; `std_source` is `"scale"` or `"provenance"`. `--format text` prints the table as lines.

### 4.3 The answer of `dump`

* tree or forest: `{"model","task","kind","trees","tree":I,"nodes":N,"depth":D,"list":[...]}`, where `list` holds the nodes in preorder (the stored order); each node is
  `{"id":i,"depth":k,"leaf":false,"column":"x1","source":"x1","threshold":"1.8","cell":"1.8","missing":"right","left":1,"right":8,"counts":[37,38],"rows":75}` for an internal node
  (`column` is the model column name, `source` the raw column; `threshold` is the stored float in the shortest decimal; `cell` is the text a data cell of the raw column would have for that value (an integer
  for an `int` column, the fixed-point text with the column's scale for a `dec` column, the shortest decimal for a `float` column) and is `null` when the steps changed the column; `missing` is the side
  a missing cell takes; a row goes left when `x < threshold`), and `{"id":i,"depth":k,"leaf":true,"value":"0.3333333333333333","counts":[2,1],"rows":3}` for a leaf (`value` is the stored leaf value: a
  probability of class 1 for a classification, the mean for a regression). `--format text` prints the tree indented, one node a line.
* linear and logistic: the stored `intercept` and for each model column its `coefficient`.
* k-means: for each centroid (cluster 0 to k-1) the value of each model column, and `raw` (the value in the units of the raw column) when every step that made the column is a scaling.
* constant: `ml.dump.unsupported-model`.

The node list is built only after `nodes <= --max-nodes` has been checked on the number the payload states (the payload's own checks bound that number by the file size).

## 5. Refusals (new tags, exit 8 unless stated)

| tag | exit | when | repair |
|---|---|---|---|
| `ml.importance.unsupported-model` | 8 | the model is a constant or a k-means | none |
| `ml.importance.no-scale` | 8 | a standardised coefficient needs the deviation of a column and the file stores none (D5); the columns are in `detail` | none: the hint names `--data FILE`, the permutation importance, which needs no stored deviation |
| `ml.importance.bad-metric` | 8 | `--metric` is not a metric of the task, or `--metric`, `--repeats` or `--seed` without `--data` (`args.conflict`) | the metrics of the task are in `detail` |
| `ml.importance.undefined-metric` | 8 | the metric has no value on the baseline or on a permuted run | choose another metric |
| `ml.importance.too-few-rows` | 8 | `--data` holds fewer than 2 rows | none |
| `ml.limit.work` | 8 | `columns * repeats * rows` above `--max-work` (default 2^32: the permutation would score that many row evaluations) | none: fewer repeats, a smaller file or a higher `--max-work` (said in the hint) |
| `ml.dump.unsupported-model` | 8 | the model is a constant | none |
| `ml.limit.nodes` | 8 | `dump`: the tree has more nodes than `--max-nodes` | raise `--max-nodes` (existing tag; second fixture) |
| `ml.param.out-of-range` | 2 | `--tree I` with `I >= trees`, `--repeats` outside 1 to 100 (existing tag) | the range is in `detail` |

`args.conflict` (existing) answers a flag the verb does not take. The file reader's tags (`ml.model.*`, `ml.schema.mismatch`, the input tags of the reader) are the ones `eval` and `describe` give.
New tags are registered in `tools/ml/spec.tmpl` (`extra()` and `rules()`), `docs/design.md` sections 8 and 8.1, with one fixture each in `tests/conformance/fixtures.py`.

## 6. `describe` and `GET /model`

`describe` gains one member, `importance`, after `limits` **[choice]**: always present; the object of section 4.2 for kinds 4, 5 and 7, `null` for kinds 1 to 3 and 6; for a model whose
standardised coefficients cannot be computed (D5) the object `{"method":"standardised_coefficient","available":false,"reason":"ml.importance.no-scale","columns":[...]}` so that `describe` keeps
answering for every file it answered for before. No existing member changes: a gate removes the member from the new golden answers and requires the old ones, byte for byte (section 8, I9).
`docs/serving.md` section 5 gets a row; `schemas/ml.v1.json` and the seven `docs/serving/*.openapi.json` documents are regenerated, and a gate shows that the only difference is the new member.
`ml-serve` returns `describe` for `GET /model`, so the table is served with no new code in the server.

## 7. Boosting plugs in later (draft PR #62, model kind 8, **not merged here**)

One function per model kind, one dispatch. `tools/ml/interp.cho` has `table(kind, ...)`; the arms are `4 | 5 -> standardised`, `7 -> impurity`. A gradient-boosted model reuses the tree walk of
`impurity` over its own payload (its node layout is the same plus a shrunk leaf value), with the gain per node if its payload stores it, or the same count-based decrease on the stored counts;
its function and one arm are the whole change. `dump` has the same dispatch (`dump.nodes` per kind). The permutation path is model-agnostic: it calls the estimator's predict.

## 8. The gates (fixed now; each must be able to fail; a gate that cannot is not a gate)

Independent code: `scripts/oracle_interp.py` reads the model file with `scripts/oracle_st.py`'s reader of the file level and the layouts of this repository's documents, and computes every
table from the bytes, with none of the code of `tools/ml`, `lib/pipeline` or `lib/transforms`. Predictions and metrics for the permutation oracle come from `scripts/oracle_wire.py`,
`scripts/oracle_wire_tree.py` and `scripts/oracle_metrics.py` (the oracles of the earlier stages, themselves compared with the libraries bit for bit); the permutation stream is re-implemented
in Python from `docs/rng.md` and `docs/metrics.md` 2.1.

| id | gate | judged by | a mutant it must kill |
|---|---|---|---|
| I1 | **standardised coefficients, bytes.** For linear and logistic models (plain, imputed, scaled with ddof 0 and 1, minmax, scaled twice), the printed `coefficient`, `std`, `standardised`, `importance`, `rank` and `total` equal the oracle's, byte for byte | `test_interp.py` | `b * sd` replaced by `b / sd`; the `ddof` 1 value; the provenance std of the wrong feature; the divisor product |
| I2 | **impurity decrease, bytes.** For trees and forests of both tasks, with missing cells, with and without bootstrap, the printed `importance`, `splits`, `split_rows` and `total` equal the oracle's, byte for byte; `splits` and `split_rows` are integers | `test_interp.py` | the Gini identity's operands; the preorder accumulation; the tree sum of the leaf values; the per-tree normalisation (skipped); the forest renormalisation |
| I3 | **sums to 1.** `total` is `1.0` and the printed shares add up to it within `w` ulps for every table with a positive sum; `0.0` and all-zero shares when no tree splits (a one-leaf tree) and when every coefficient is 0 | `test_interp.py` | a missing normalisation |
| I4 | **scikit-learn, on our trees.** The structure of a `bootstrap=0` tree of ours is loaded into scikit-learn's `Tree`, the node impurities are computed from the data routed down our tree, and `compute_feature_importances` is compared with ours: relative `1e-12` for classification, `1e-9` for regression (the stored leaf means), plus the same on a forest of 20 trees; and `RandomForest*.feature_importances_` on the same data (a **different algorithm**: exact thresholds, its own streams): the top feature is the same, the Spearman rank correlation is at least **0.9**, and no share differs by more than **0.1**, on the three data sets of section 8.1; controls: scikit-learn's importances of labels shuffled against the features must be outside those bounds | `test_interp_sklearn.py` | the Gini identity's factor 2; the counts of the wrong child |
| I5 | **a noise column ranks last.** A column of seeded noise added to each data set of section 8.1 ranks **last** in the impurity importance of a forest of 100 trees and in the permutation importance on held-out rows, where its importance is also less than a fifth of the weakest signal's (*corrected in place: the first draft required it to be within three standard errors of 0, which is false: a forest splits on a noise column, and the measurement of section 10 shows a small positive value*); the **mutant that permutes the target instead of the column makes the permutation gate fail** (every column then has the same importance, the noise column is not last) | `test_interp.py` | permute the target; permute a fixed column; the metric difference reversed |
| I6 | **permutation, bits.** For linear, logistic, tree and forest models on files with missing cells, categorical columns (`--cat`), an `onehot`, a `scale`, an `impute` and a `bucket` step, `baseline`, every `importance` and every `std` equal the Python re-implementation (stream, shuffle, steps, predictions, metric) **bit for bit** for the 3 seeds, `R` of 1, 3 and 7 and every metric of the task | `test_interp.py` | the stream kind; the repeat index; the column index; the shuffle's bound; the mean or std formula; the direction of a loss; permuting after the steps instead of before |
| I7 | **scikit-learn permutation importance on the same fitted model.** `sklearn.inspection.permutation_importance` (n_repeats 30, its own stream) with an estimator whose `predict` is `ml predict` on the permuted data, against `ml importance --repeats 30`: **not the same stream**, so means are compared within bounds fixed now: for every column `|ours - theirs| <= 4 * sqrt((std_o^2 + std_s^2) / 30) + 1e-3 * |baseline|`, and the order of any two columns whose means differ by more than 4 standard errors is the same; aligned: the metric, the rows, the model, the positive label; not aligned: the permutations. Control: scikit-learn's importances of a model with the target permuted are outside the bounds | `test_interp_sklearn.py` | a metric difference of the wrong sign |
| I8 | **thread invariance.** The answers of `importance` (both tables, with and without `--data`) and of `dump` are the same bytes at `--threads` 1 and 8 | `test_interp.py` | a thread count reaching the stream |
| I9 | **describe changes only by one member.** The new `golden_describe.json` with the `importance` member removed equals the old one byte for byte; `describe.importance` equals `importance.importance` of the verb; the OpenAPI documents differ only by the member; `oracle_describe.py` computes the member and `test_describe.py` compares | `golden_describe.py --check`, `test_describe.py`, `gate_serve_openapi.py` | a member before `limits` |
| I10 | **dump.** The nodes, the counts, the thresholds (the `cell` text), the default sides and the leaf values of a dumped tree equal the oracle's reading of the payload; a tree walked by hand from the dump gives `ml predict`'s answer for 100 rows; `--tree I` picks the I-th tree of a forest; `--max-nodes` refuses at `nodes - 1` and accepts at `nodes`, before the list is built; k-means centroids equal the payload and `raw` equals the inverse of the stored scalings | `test_interp.py` | an off-by-one tree; the threshold of the wrong node; the default side flipped; the bound |
| I11 | **no trap.** Every truncation and every flipped byte of a tree, a forest, a linear, a logistic and a k-means file (with and without a repaired checksum) is refused or answered by `dump` and `importance` and `describe`, never a trap; the argument fuzz (`test_fuzz_args.py`) includes the new flags | `test_interp.py`, `test_fuzz_args.py` | a removed bound |
| I12 | **tables.** Every new flag, verb and rule is reached by a fixture (`test_tables.py`), the registry and the schemas agree (`check_registry.py`, `schemas.py --check`, `gen_models.py --check`), the authority rows of the three binaries are unchanged (`manifest.py --check`, `check_bins.py`) and `ml-predict` and `ml-serve` take neither verb | the same | a verb added to `ml-predict` |
| I13 | **mutants** (`scripts/mutants_interp.py`): every pattern is killed, or each survivor is explained in the script; the unmutated tree passes first | `mutants_interp.py` | |

### 8.1 The data sets of the gates **[choice]**

Fixed now, generated by seeded code in `scripts/oracle_interp.py` (no download): `cls`, 2,000 rows, 6 informative numeric columns (a logistic rule on a random projection plus noise) and one
`NOISE` column; `reg`, 2,000 rows, a nonlinear target of 4 of 6 numeric columns plus noise, and a `NOISE` column; `mixed`, 1,500 rows, numeric columns with 8 percent missing cells and a categorical column of 5 values
that decides the target. The noise column is `uniform(-1, 1)` from `derive(seed, 99)`, placed in the middle of the columns (after `x2`), so that an importance of zero for every column, which ranks by index, cannot put it last by accident. In `cls` and `reg` every column but `NOISE` is a signal of weight 2.0, 1.5, 1.0, 0.7, 0.5 and 0.35 (the first draft had two more noise columns, which makes "the noise column ranks last" a coin flip; the weights were fixed before the gate ran). These sets feed I4, I5, I6 and I7.

## 9. Limits and authority

* Before anything sized by the input is allocated: `--max-nodes` against the stated node count; `columns * repeats * rows` against `--max-work` (before the first shuffle); `--repeats` 1 to 100; the file is read with `--max-bytes`.
* `importance` and `dump` are verbs of `ml` only. They open the model file and the data file as `eval` does (`file_read`, `dir_read`); the authority row of `ml` is unchanged, as are the rows of `ml-predict`
  and `ml-serve` (the data-free table is computed by the code of `describe`, which they already contain).
* No file over 2,000 lines; the new code is in `tools/ml/interp.cho` (lineage and tables, in the sources of all three binaries), `tools/ml/idump.cho` and `tools/ml/perm.cho` (`ml` only).

## 10. Results

*(filled in after the gates have run)*

## 11. Open questions

1. A payload version with the exact node impurity (section 3.3).
2. `--threads` is inert in `importance`; a threaded loop over columns is a change that keeps the bytes (D6, D10).
3. Standardised coefficients of encoder columns (the deviation of an indicator is `sqrt(p (1 - p))` with `p` the category frequency, which the string table holds for `--cat` columns); refused in this version.
4. Permutation importance with groups of correlated columns, and conditional permutation, are not offered.

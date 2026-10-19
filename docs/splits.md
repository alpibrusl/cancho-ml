# Split strategies and cross-validation from the command line (issue 42)

**Status: this document and its gates (section 9) were written before any code of the stage. Sections 10 onward are filled in as the
code lands; a number or a verdict appears in them only if a script of this repository printed it, with the conditions.** Tags: **[choice]**
a decision the maintainer can overturn, **[read]** read from a document or a source file, **[measured]** printed by a script here,
**[open]** not decided.

Before this stage `ml fit --split train=F --seed N` was the only split the tool exposed: a seeded random holdout. `lib/metrics/splits.cho`
had a seeded k-fold and a stratified k-fold with gates (`docs/metrics.md` section 2), and nothing in the command line reached them. People use the
command line, not the library, so every strategy of this document is a flag of `ml`, parsed, refused, repaired and described by `introspect`
and `skill` like every other flag.

## 1. The surface **[choice]**

One flag table (`tools/ml/spec.tmpl`) drives the parser, `introspect`, `skill` and the schemas. Six flags and one verb are new.

| flag | kind | verbs | meaning |
|---|---|---|---|
| `--stratify` | bool | fit, eval, cv | keep the class proportions (a classification target only) |
| `--split-by time:COLUMN` | text | fit, eval, cv | order the rows by COLUMN: the oldest train, the newest test |
| `--split-group COLUMN` | text | fit, eval, cv | all the rows of a group on one side (or in one fold) |
| `--folds K` | nat, 2 to 20, default 5 | cv | the number of folds |
| `--repeats R` | nat, 1 to 10, default 1 | cv | the number of repetitions of the fold assignment |
| `--emit-folds` | bool | fit, eval, cv | put the assignment of the rows in the answer, as data |

**A new verb, `ml cv`, not `fit --cv K`.** `cv` never writes a model: it scores. A flag on `fit` that makes the verb stop writing
the file it was asked to `--output` would be a verb in disguise, `--output` would be required by the verb and refused by the flag, and a
script that reads "fit wrote a model" would be wrong. `cv` takes everything `fit` takes except `--output`, `--overwrite`, `--dry-run`
and `--split`; the `ml` authority gains nothing (it reads the file `fit` reads and writes what `fit` writes only through `fit`). `ml-predict` and
`ml-serve` are unaffected: a split is a training-time concept, none of these flags is in their tables (`scripts/check_bins.py` checks it).

**The holdout variants** are flags of `fit --split train=F` (and of `eval --split train=F`, which must be able to replay the rows a fit held out: without it
`eval --split train=0.8 --seed 3` would score rows a stratified fit trained on). Without `--split` they are `args.conflict`. `--seed` is
the seed of the split as it is now, and of the model.

**Which combinations are legal.** At most one of `--stratify`, `--split-group`, `--split-by` per command: `StratifiedGroupKFold` is a different
algorithm (balance of classes and no shared group are two constraints that no deal satisfies in general, and an approximation would need a
rule of its own, with its own gate), and a group time series split is another (groups that straddle the cut). They are **refused**, not approximated:
`ml.split.combination`. `cv` with `--split` is the same tag. `--repeats` above 1 with `--split-by` is the same tag (an expanding window has
nothing random to repeat). So is `--early-stopping` (a boosted model) with `--split-group` or `--split-by`, in `fit --split` and in `cv`: early stopping validates on a
random tenth of the training rows, which a group split or a time split cannot allow (`docs/models-boost.md` section 2.8); the repair drops `--early-stopping`.

## 2. The canonical split text **[choice]**

One function, `splitspec.text` (`tools/ml/splitspec.cho`), produces one line for any split, and nothing else writes one. The run record of `fit --track` and the eval
record of `eval --track` store it as the `split` of their inputs (`docs/tracking.md` section 5.2), and `ml runs verify` reads it back with `splitspec.parse`; this section is the contract.

```
none
holdout train=F [stratify | group=C | time=C] seed=S
cv folds=K repeats=R [stratify | group=C | time=C] seed=S
```

Tokens are single-space separated, in that order, no trailing space. `F` is the train fraction as typed, normalised: no leading `.`-less form, no
trailing zeros (`0.80` and `.8` are `0.8`; at most 6 digits). `S` is the decimal seed. `C` is the column name with every byte outside
`A-Z a-z 0-9 _ . -` written as `%` and two upper-case hex digits (so the line has no space, no `=`, no control byte, and a name with a non-ASCII byte is
still one token). Examples: `holdout train=0.8 stratify seed=1`, `cv folds=5 repeats=1 group=customer seed=3`, `holdout train=0.75 time=order%20date seed=0`.
The line is in the answer of `fit` (`summary.split.spec`), `eval` and `cv` (`split`). `none` is for a fit without `--split`.

## 3. The assignment, strategy by strategy **[choice]**

All draws are the library's (`docs/metrics.md` 2.1 and 2.2, `docs/rng.md`): `stream(seed, kind, index) = derive(derive(seed, kind), index)`,
`bounded` by rejection on 63 bits, `shuffle` Fisher-Yates from the end. Two stream kinds are new (`docs/rng.md`):

| kind | stream | index |
|---|---|---|
| 7 | stratified holdout | the class rank (0 for the smaller label) |
| 8 | group shuffle | the repeat (0 for a holdout) |

`n` rows, `n_test = ceil(n * (1 - F))` as before (`splits.n_test_for`: `num/den` with `den <= 10^6`), `1 <= n_test <= n - 1` else `ml.split.empty-fold`.
"Rank" of a group is its position among the distinct group texts sorted bytewise.

**Holdout.** The answer is a vector of 0 (train) and 1 (test) per row.
* plain: `splits.train_test` (kind 4, unchanged: the bytes of every existing fit are the same).
* `--stratify`: the classes, by ascending label; `members_c` is the rows of class `c` in ascending row order, shuffled with `stream(seed, 7, c)`; `L` is
  `members_0` then `members_1`; the element at position `t` of `L` is a **test row iff `floor((t + 1) * n_test / n) - floor(t * n_test / n) = 1`** (integers).
  The test rows are exactly `n_test`; the test rows of class `c` are `floor` or `ceil` of `n_c * n_test / n` (the difference of two floors over a window of
  `n_c` positions), the property gate B1 checks. A class of fewer than 2 rows, or one that would have no training row, is `ml.split.stratum-too-small`.
* `--split-group C`: groups sorted bytewise, `G` of them (`G >= 2` else `ml.split.too-few-groups`); `perm` = the ranks shuffled with `stream(seed, 8, 0)`; walking `perm`, a
  group goes to the test side while the test side has fewer than `n_test` rows and fewer than `G - 1` groups. The test side has at least `n_test` rows and
  overshoots by less than the size of one group; the answer states the real count. The training side is never empty.
* `--split-by time:C`: the rows ordered by `(key, row)` (a stable order); the first `n - n_test` are train, the rest test. Ties at the cut are cut in file order.

**Cross-validation.** The answer per repeat is a fold number per row. Repeat `r` is computed alone (no stream is shared).
* plain: `splits.kfold(seed, r)` (kind 5): a shuffled permutation cut at `fold_start(n, K, f)`. Needs `K <= n`.
* `--stratify`: `splits.stratified(seed, r)` (kind 6, index `2r + c`): `L` as in `docs/metrics.md` 2.3, position `t` is in fold `t mod K`. Every class has at least `K` rows.
* `--split-group C`: `G >= K` groups (else `ml.split.too-few-groups`); `perm` = ranks shuffled with `stream(seed, 8, r)`; walking `perm`, each group goes to the fold with the
  fewest rows so far (ties: the lowest fold number). Every fold is non-empty (the first `K` groups meet empty folds). Group sizes can make the folds unequal; `cv` reports them.
* `--split-by time:C`: the rows ordered as above are cut into `K + 1` blocks at `fold_start(n, K + 1, b)`. **Fold `i` (0 to K - 1) trains on blocks `0 .. i` and tests on block `i + 1`**
  (an expanding window; block 0 is only ever training; later blocks are in neither side of fold `i`). The emitted assignment is `block - 1` per row: `-1` for block 0. Needs `n >= K + 1` else `ml.split.empty-fold`.

**What depends on the order of the rows.** Plain holdout and k-fold: the permutation is of positions, so the assignment depends on the file's row order (as in the library).
Stratified: the members of a class are listed in row order, so it does too. Group: the groups are sorted by their text and sized by counting, so the assignment of a row depends on its group's text,
the set of groups, their sizes and the seed, **not on the row order**. Time: the order of the keys, and the file order among equal keys only. None of them depends on the thread count or on anything
else: the assignment is computed before any fit and by no thread.

## 4. The columns **[choice]**

`--split-by` and `--split-group` name a column of the header (`ml.input.unknown-column`, exit 3, if it is not there; `ml.split.column-target` if it is the target). The column is read **a second time,
as text**, by `tools/ml/splitcol.cho`, with the dialect of the reader (`docs/reader.md` section 2: a byte order mark dropped, CR before LF dropped, blank lines are not records, RFC 4180 quotes), so a text
column that the numeric reader cannot take is usable: the column is also a feature unless `--drop` removes it (the usual case for a date or an id, and the refusal `ml.input.non-numeric` carries the `--drop` repair).
The file is held in memory once, up to `--max-bytes` (`limit.input-too-large` beyond). An empty cell is `ml.split.missing-cell` (row in `detail`). With `--input-format jsonl` the bytes read are the CSV made of the lines (`docs/jsonl.md` section 4), so the column is the one the CSV holds: a string date is text (name the column with `--drop`, or `--cat`), a number is a number, and the cuts and the answers equal the CSV's byte for byte (`tests/conformance/test_jsonl_cv.py`).

**Groups.** The identity of a group is the bytes of its cell after the CSV quotes are removed, compared bytewise, no trimming, no case folding, no number parsing: `7` and `07` are two groups. The column may be text
or integer for this reason alone. No bound on the group count except the row count.

**Time.** The kind of the column is the kind of its first cell, and every cell must be of it (`ml.split.time-format`, reason `mixed`). A **number** is a cell of the reader's grammar (integer, decimal, float; `docs/reader.md`
2.1), its key is the order-preserving key of its double (`csvcell.key_of`), compared as integers: `20240115` is a date to this rule, in order. A **date or datetime** is, exactly,
`YYYY-MM-DD`, or `YYYY-MM-DD` then `T` or one space, then `hh:mm`, optionally `:ss`, optionally `.` and 1 to 9 digits, optionally one final `Z` (UTC). Calendar rules are checked
(month 1 to 12, day within the month, leap years by the Gregorian rule, hour 0 to 23, minute and second 0 to 59, year 0000 to 9999). Its key is `(YYYYMMDDhhmmss, nanoseconds)` compared lexicographically, so `2024-01-05`,
`2024-01-05T00:00:00` and `2024-01-05 00:00:00Z` are the same instant (a tie), and `10:00:00.25` is before `10:00:00.5`. **Refused** with the reason in `detail`: a numeric offset (`+01:00`: no two offsets compare by bytes
and a conversion would be a rule of its own), a `Z` on a bare date, a week or ordinal date, a 24:00, a second 60 (`reason`: `offset`, `form`, `range`), and everything else (`not-a-time`).

## 5. Leakage: what is fitted on what **[choice]**

Every fitted step of the plan (impute, scale, bucket, encode, freq, hash, the categorical vocabulary) is fitted on the training rows of the fold by `wire.train`, the function `fit` uses, and the held-out rows are
transformed by the **stored** steps of that fit, the function `fit --split` uses. `cv` has no other path to the data: there is no step fitted on all rows to leave out. The model of a fold uses `--seed` as `fit --seed` does, so the fitted model of a fold is the model
`ml fit --seed S` writes from the training rows of the fold (gate C1 compares them). Folds run one after the other, each fit threaded as it is now (`--threads`); nothing about the answer depends on the thread count.

A training fold that lacks a class of a classification target is `ml.label.single-class` (the fit of that fold would be the refusal of `fit` on that file).

## 6. The answer of `ml cv` **[choice]**

```
{"model":..,"task":..,"target":..,"rows":N,"columns":..,"plan":"..","split":"cv folds=5 repeats=1 seed=0","strategy":"kfold",
 "results":[{"repeat":0,"fold":0,"train":..,"test":..,"metrics":{..}}, ...],
 "summary":{"count":M,"std":"sample","metrics":{"rmse":{"mean":"..","std":"..","n":M}, ..}},
 "assignment":{"repeats":R,"rows":N,"folds":[[..], ..]}}                 (with --emit-folds)
```
`metrics` of a fold is the object `ml eval` prints for the held-out rows (same code, same strings). In `summary.metrics`, every scalar member of the metrics (a string that reads as a finite number) is aggregated over the `M = K * R` results in the order
(repeat, fold): the **sum** is the package's one order (`metrics.sums`: blocks of 256, four lanes, the fixed tree), `mean = sum / M` (one IEEE division), `std = sqrt(sum((x - mean)^2) / (M - 1))` with the same sum: the
**sample** standard deviation (divisor `M - 1`; the population one would understate the spread of `M` scores that estimate one thing), `M >= 2` always. A member that is `null` in some fold (an undefined metric) has `mean` and `std` `null` and
`n` the number of folds that had it. Arrays (the sizes of the clusters) are not aggregated. `--format text` prints one line per fold and the summary.

`--emit-folds` puts the fold of each row in the answer: `folds[r][i]` is the fold of row `i` (0-based, file order) in repeat `r` (`-1` for the initial block of a time split). In `fit` and `eval` it is `summary.split.assignment` /
`split.assignment`, a vector of 0 and 1 (1 = held out). Bounded: at most 1,048,576 entries (`rows * repeats`), else `ml.split.emit-too-large`.

## 7. Refusals **[choice]**

Tags are registered in `tools/ml/spec.tmpl`, `docs/design.md` sections 8 and 8.1, with a fixture each (`tests/conformance/fixtures.py`).

| tag | exit | when | repair |
|---|---|---|---|
| `ml.split.stratify-target` | 2 | `--stratify` with a regression or a clustering | `retry` without `--stratify` |
| `ml.split.stratum-too-small` | 8 | a class with fewer rows than `--folds` (cv), or fewer than 2 (holdout) | `retry` with `--folds` = the smallest class, when it is at least 2 |
| `ml.split.too-few-groups` | 8 | fewer groups than `--folds`, or fewer than 2 | `retry` with `--folds` = the number of groups, when it is at least 2 |
| `ml.split.time-format` | 8 | a time cell that is not a number or an accepted date (`line`, `column`, `reason`) | none |
| `ml.split.missing-cell` | 8 | an empty cell in the time or group column (`row`, `column`) | none |
| `ml.split.combination` | 2 | two strategies, `cv` with `--split`, `--repeats` above 1 with `--split-by`, `--early-stopping` with `--split-group` or `--split-by` | `choose` between the two ways of dropping a flag; `retry` with `--repeats 1`, or without `--early-stopping`, when that is the conflict |
| `ml.split.out-of-range` | 2 | `--folds` outside 2 to 20, `--repeats` outside 1 to 10 | none (the range is in `detail`) |
| `ml.split.column-target` | 2 | the group or time column is the target | none |
| `ml.split.emit-too-large` | 8 | `--emit-folds` over 1,048,576 entries | `retry` without `--emit-folds` |
| `ml.split.empty-fold` | 8 | (existing) a side or a fold would be empty: `K > n`, `n < K + 1` for time, `n_test < 1` | none |

## 8. Determinism, bounds, and what is not claimed

* The same file, seed and flags give the same assignment, at any thread count and on any machine: it is made before any fit, from the library's stream, with no clock and no thread (gate D1, `--threads 1` and `8`).
* All sizes are compared with the caps before an allocation: rows with `--max-rows`, the column text with `--max-bytes`, folds with 20, repeats with 10, the emitted entries with 1,048,576. No input reaches a trap (gate F1).
* **Not claimed: equality with scikit-learn's assignment.** Its generator (`numpy.random.RandomState` and `permutation`) is not SplitMix64, so `KFold(shuffle=True, random_state=s)` and `--seed s` give different assignments. What is compared with scikit-learn is
  the **invariants** (gate B1: class counts, no group on two sides, time order) and the **metrics on the same explicit fold files** (gate S1: the folds of `--emit-folds` given to scikit-learn's logistic regression and linear regression, a
  bound fixed in advance). Nothing is claimed about speed.
* Constant (`--model constant`) is **not** scored by `cv`: its fit and its eval are the exact-decimal code of `docs/cli.md`, written for a whole file read by the scanner, with no path that takes a subset of rows; `ml cv --model constant` is `args.conflict`.
  The baseline of a comparison is `ml fit`/`ml eval` on the fold files `--emit-folds` gives.

## 9. The gates (written before the code; each can fail) **[choice]**

All are run by `scripts/check.sh` through `tests/conformance/test_splits.py` (S1 through `tests/conformance/test_splits_sklearn.py`, which imports numpy and scikit-learn and which the fast job of the CI workflow lists as not run) and `scripts/oracle_split.py`; the mutants of section 9.1 prove that each can fail.

| id | gate | would fail on |
|---|---|---|
| A1 | **The assignment, bytes.** `scripts/oracle_split.py` re-implements the stream (SplitMix64, bounded, shuffle), the CSV dialect, the group and time rules and every strategy of section 3 from this text, and compares the vector of `--emit-folds` with its own, on generated and fixture files, all strategies, several seeds and repeats | a draw, an index, a kind, a rounding, a tie rule, the fold rule |
| B1 | **Invariants, on generated and adversarial data** (not a copy of the rule): stratified class counts per fold (holdout and cv) within 1 of the exact proportion; no group on both sides, every group in one fold; max train time <= min test time (every fold of a time cv); every row tested exactly once (k-fold, stratified, group); fold sizes of k-fold `floor`/`ceil` | a rule that is the oracle's twin and wrong |
| C1 | **End to end, no leak.** For each fold the oracle writes the training and held-out CSV, runs the real `ml fit` and `ml eval` on them, and the metrics must equal the per-fold `metrics` of `ml cv` **string for string**, for linear, logistic, tree, forest and kmeans, with plans that have `scale`, `impute`, `bucket`, `encode`, `freq`, `hash` and categorical columns | any step fitted on rows outside the training fold |
| C2 | **The planted leak.** A build in which the steps are fitted on all rows (`scripts/mutants_split.py`, mutant `leak-steps-fitted-on-all-rows`) must FAIL C1 | C1 not able to see a leak |
| D1 | **Thread invariance.** The whole answer of `ml cv` and of every holdout is identical at `--threads 1` and `8` | a thread-dependent assignment or fit |
| E1 | **Refusals.** One fixture per tag of section 7, with the repair applied: the retry runs and succeeds | a tag, an exit code or a repair changed |
| F1 | **Fuzz.** Random argument lists over the new flags; hostile data (one class, one row per group, a group per row, ties in the time column, missing cells, 5,000 groups, quoted cells with commas and newlines, BOM, CRLF): an answer of the contract and no trap | a trap, a hang |
| G1 | **Golden.** Every existing golden, fixture and model file is unchanged except the ones listed in section 10 | an accidental change |
| S1 | **scikit-learn on the same folds.** Folds from `--emit-folds` given to `sklearn` (`LogisticRegression(C=huge)`, `LinearRegression`): the per-fold metric within a bound fixed here before the first run: 1e-6 relative for regression coefficients' metrics, 1e-4 absolute for logistic accuracy and log-loss | the folds not being the folds the metrics use |
| M1 | **Mutants** (`scripts/mutants_split.py`) for the assignment, the stratification, the group rule, the time order, the refusals, the aggregate and the leak: killed, or each survivor explained | a gate that does not fail |

### 9.1 Mutants

`scripts/mutants_split.py` rebuilds the tool once per pattern, as `scripts/mutants_cli.py` does, and runs `tests/conformance/test_splits.py` against it.

## 10. Results

Machine: a Mac (arm64, Apple M4 Max, macOS 26.2), compiler `a4572ea`, `build/ml`; the machine was shared. Correctness gates do not depend on the load. Counts are those of the run that printed them.

| gate | result | command |
|---|---|---|
| A1 | 286 cases: the assignment of `--emit-folds` and the canonical text equal the oracle's | `python3 scripts/oracle_split.py --assign` |
| B1 | 381 cases of invariants on generated and hostile data | `--invariants` |
| C1 | 246 folds: `ml cv` metrics equal `ml fit` then `ml eval` on the oracle's fold files, string for string; plans with impute, scale, bucket, encode, freq, hash and text columns; linear, logistic, tree, forest, boost (without early stopping, which is refused under the group and time strategies), kmeans; the aggregate equals the oracle's | `--e2e` |
| C2 | the mutant `leak-steps-fitted-on-all-rows` fails C1 (killed) | `python3 scripts/mutants_split.py --only leak` |
| D1 | 18 answers identical at `--threads 1` and `8` | `--threads` |
| S1 | linear: worst relative gap 6.13e-16 (bound 1e-6); logistic: accuracy gap 0 and log-loss gap 1.44e-08 (bound 1e-4 absolute) | `--sklearn` |
| E1, F1 | the fixtures of every tag, with the repairs, and the hostile data and fuzz of `tests/conformance/test_splits.py` | `python3 -m unittest test_splits test_rules` |
| M1 | 47 mutants, 47 killed, 0 survive | `python3 scripts/mutants_split.py` |

Limits found while running: a held-out fold with one class has no ROC-AUC, and `ml.label.single-class` refuses the cross-validation as `ml eval` does on that file (a group or time split of few groups meets it; `--stratify` or fewer folds avoid it); `--model constant` is not cross-validated; `fit` and `eval` answers with `--split` gained `split.spec`, so three golden answers of the old records changed their hash and are pinned in `test_cat_baseline.py` and `test_planned_baseline.py` (status and files unchanged).

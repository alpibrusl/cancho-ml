# Wiring the recommender into `ml fit | predict | eval | describe` (stage C2 of issue 70)

**Status: sections 1 to 10 are written BEFORE any code of this stage; the first commit of the branch holds only this file. Section 11 holds the results; a number or a verdict is in it only if a script of this repository printed it, with its conditions.** Tags as in `design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from a document or the code, **[estimate]** a figure derived on paper, **[open]** not decided. A claim found false is corrected in place and marked.

`docs/recommend.md` fixes what the recommender is (the log, ALS explicit and implicit, the neighbours, the metrics, the limits, the gates RC1 to RC14). Stage C1 (merged) gave the pure parts: the metrics (`lib/metrics/recommend.cho`) and the compressed forms of the log and the split per user (`lib/models/recommend/log.cho`), each with an oracle. This stage wires ALS **through the same door as the other models** (`docs/modules.md` is the contract, `docs/wiring.md` and `docs/wiring-kmeans.md` the flow): a manifest, the dispatch written by `scripts/gen_models.py`, the plan keys, the flag table and the schemas, and `fit` stores the model in one `CML1` file.

The recommender differs from every model wired before in the way that shapes the stage: **it has no feature matrix and no target.** Its input is a log of `user, item[, rating][, time]`; its answer is, for a user, a ranked list of items; its file holds ids and the history of the log. Everything the other models share that depends on a matrix (steps, imputation, categorical encoders, `--target`, `--positive`, `--drop`, `--cat`, `--float`) does not apply, and each of those flags is refused with `args.conflict`, not ignored **[choice]**. What does apply is the rest: the reader and its limits, `--threads`, `--seed`, `--split`, the plan, `--track`, the CML1 envelope and the toolbox's errors.

**In this stage:** ALS (explicit and implicit), `fit`, `describe`, `predict --for` and `--like`, `eval`. **Not in this stage:** the neighbour method (C4), threads over the solves (C3: the serial version first, `--threads` is accepted and changes nothing, which a test shows), cold start (`--cold`, C5), `ml-serve`, `ml-mcp` and JSON lines (C5).

## 1. The gates (fixed before the code; each can fail, section 11 shows it failing)

| id | gate | judged by | a mutant it must kill |
|---|---|---|---|
| WR1 | **Tables from the manifest.** The flags (section 2) are data in `lib/models/recommend/model.toml` and the verb tables; `gen_models.py --check`, `schemas.py --check` and the table gate T1 are clean; every new flag is reached by a fixture and every new rule by one with its exit code; the `--model` choice list gains `recommend` | `gen_models.py --check`, `test_tables.py` | a flag without a fixture; a rule without an exit code |
| WR2 | **The log reader.** `--user`, `--item`, `--rating`, `--time` name columns of the header (`ml.input.unknown-column`); a column named twice, or a log column that is also `--time` of another role, is `args.conflict`; the ids are read as text and numbered by the bytewise rank of the distinct cells, so a shuffled log gives the same file **byte for byte**; ids that look like numbers (`007`, `7`, `1e3`) are three ids; an empty id cell is `ml.recommend.missing-id` | `test_recommend_fit.py`, `oracle_wire_recommend.py` | the numbering by first appearance; a numeric reading of an id |
| WR3 | **Oracle, bit for bit.** A Python implementation of the plain ALS of `recommend.md` section 4 (dense NumPy, the same accumulation order, the Cholesky of `scripts/oracle_linalg.py`) writes the whole model file; it equals the file `ml fit` wrote, header, plan, schema, ids, history, factors and trailer included, on seeded logs (up to 40 users, 30 items, factors up to 6, both forms, with and without `--alpha`) | `scripts/oracle_wire_recommend.py` | the item order of a sum; `lambda` once instead of per solve; the implicit `c` for `c - 1`; `Y^T Y` per user |
| WR4 | **The implicit form against the formula** and **the loss falls** (RC4, RC5): the closed form solved densely for each user within a bound fixed in the script before the run; the regularised loss after each half-pass is not above its value before it, within a rounding bound fixed in the script | `scripts/gate_recommend_loss.py` | the preference taken from the confidence; a skipped term |
| WR5 | **The verbs.** `fit` prints the method, the users, the items, the interactions, the factors, `lambda`, the passes, the loss after the last pass and the file's size and digest; `predict --for COL` prints, for each input user, its `--top N` items not in the user's history as rows `user, rank, item, score`; `--keep-seen` keeps them; `--like COL` prints the `N` nearest items by the cosine of the factor vectors; `eval` prints the metrics of `recommend.md` section 3; `describe` prints the method, the counts, the density and that the file holds the log | `test_recommend_fit.py`, `test_recommend_predict.py` | the history not excluded; ties broken by the larger item; the rank counted from 0 |
| WR6 | **Cold start is a refusal** in this stage: an unknown user (item) is `ml.recommend.unknown-user` (`unknown-item`), exit 8, with the repair that names `--cold popular`, which exists from C5 and is reported as not yet available **[choice: a refusal with a repair that does not exist yet would lie]** | `test_recommend_predict.py` | the refusal dropped; a silent empty answer |
| WR7 | **Refusals reachable from `ml`, before any allocation sized by the data.** The limits of `recommend.md` section 2 (`ml.limit.users`, `ml.limit.items`, `ml.limit.interactions`), `--factors` outside 1 to 256, `--iterations` outside 1 to 1,000, `--l2` not positive, `--alpha` not positive, a value out of range (`ml.recommend.bad-rating`), a repeated explicit pair (`ml.recommend.duplicate-pair`), `ml.linalg.not-positive-definite` with the repair that names `--l2`, `--max-rows`, `--threads`, `ml.plan.*` | `test_rules.py`, fixtures | a check dropped; a limit tested after the allocation |
| WR8 | **Thread invariance of the serial stage.** `fit --threads t` for t in 1, 2, 4, 8 writes one file and prints one answer | `test_determinism.py` | a thread count that reaches the sums |
| WR9 | **Damaged files.** Every count, offset and sort order of the payload (section 6) broken in turn, at the stride of the other file gates: each is `ml.model.malformed` or `ml.model.truncated` with its section, no trap, no allocation sized by an unchecked count | `test_wire_recommend_file.py` | a check removed |
| WR10 | **The flags that do not apply** (`--target`, `--drop`, `--cat`, `--float`, `--positive`, `--stratify`, `--split-by`, `--split-group`, `--na-values`, `--step`) are `args.conflict` with `--model recommend`; the flags of the recommender with another model are `args.conflict` too; `ml cv` with `--model recommend` is `args.conflict` that names `fit --split-user-last` | `test_recommend_fit.py` | a flag accepted and ignored |
| WR11 | **Splits.** `fit --split-user-last N --time COL` holds out each user's `N` latest interactions and prints the metrics of the held-out part; `--split train=F` is a random split of interactions and the answer says `leaks_future: true`; `--split-by time:COL` is the global time split of `splits.md`; the answers equal the oracle's masks (RC12) | tests, `oracle_recommend.py --log` | the order of ties at the cut; the leak not reported |
| WR12 | **Track and replay.** `--track` records the columns, the method, the seed and the data digest; `ml runs verify` replays the fit and gets the same file digest | `test_track.py` | the column roles dropped from the record |
| WR13 | **Admission.** `scripts/admit.py recommend`: the checks Q0 to Q11 pass for `supported`, else the manifest says `experimental` and section 11 lists the failures | `admit.py` | the module mutants of `tests/modules/run.py` |
| WR14 | **Mutants** of the code of this stage: all killed, or each survivor explained | `scripts/mutants_recommend_wire.py` | itself |

## 2. The command line [choice]

```
ml fit --model recommend --user COL --item COL [--rating COL] [--time COL]
       [--implicit] [--factors F] [--l2 L] [--iterations N] [--alpha A]
       [--split-user-last N | --split train=F] [--seed S] [--threads T] --output M LOG
ml predict --model-file M (--for COL | --like COL) [--top N] [--keep-seen] IDS
ml eval    --model-file M --user COL --item COL [--rating COL] [--top K] TEST
ml describe --model-file M
```

* The four column flags are the first flags that name roles of columns rather than columns to drop or encode; they are common to `fit` and `eval`. `--rating` is optional: without it every row is an event of value 1 and the fit is implicit (`--implicit` is then implied; with `--rating` and no `--implicit`, the fit is explicit). `--time` is required by `--split-user-last` and `--split-by time:`, and refused (`args.conflict`) otherwise **[choice: a column the fit does not use is a mistake worth saying]**.
* Defaults (`recommend.md` section 6): `--factors 32`, `--l2 0.1`, `--iterations 10`, `--alpha 40.0` (implicit only; with the explicit form `--alpha` is `args.conflict`). Ranges: `--factors` 1 to 256, `--iterations` 1 to 1,000, `--l2` and `--alpha` positive and finite.
* `predict --for COL` reads the file `IDS` and the column `COL` of it (one id per row; other columns are ignored); `--like COL` the same for items. `--top` is 1 to 1,000 (default 10) and at most the number of items; the rank is 1-based. `--limit` and `--from` page the answer as for the other models (the rows are `users x top`).
* `eval` reads a test log; for each user with a held-out item, the model ranks all items except the user's training items (`recommend.md` section 3). `--top` is the `k` of the metrics (default 10). Users the model has never seen are counted in `users_cold`, not scored and not refused **[choice: a test log of a real split always has some]**; items it has never seen can never be recommended and count as misses for the user who has them.
* `describe` reads the file only (as for every model).

## 3. The flow [choice]

`fit`: parse the flags (the conflicts of WR10 before the file is opened) → read the log through the existing reader with the three or four named columns (`splitcol.extract` names a column and returns its cells; the ids are ranked by `splitcol.group_ranks`, the function that gives a group column its numbers) → the checks and limits of `recommend.md` section 2 (`log.build` and the tool's own tags) → a split, if asked (`log.split_user_last`, or the existing holdout/time machinery on the interaction rows) → the ALS passes → the file → the answer. `--dry-run` plans without writing, as for every model. There is **no step stage**: a recommender has no matrix to impute, scale or encode, and the `steps` section of its file is the empty list (`u32 0`).

`predict` and `eval` do not go through the matrix path either: the ids of the input are looked up in the file's sorted id tables by binary search, and the scores are `p_u . q_i`. This is a second flow beside the matrix flow of `wireio.cho`, behind the same verbs; it lives in its own module (`tools/ml/recwire.cho`, `tools/ml/recio.cho`) so that no existing file grows past its ceiling.

## 4. The plan [choice]

The plan line is `model kind=recommend method=als implicit=0 factors=32 l2=0.1 iterations=10 alpha=0` (keys in the order of the manifest, defaults printed, so the same plan is the same bytes). `ml fit --plan FILE` works as for the other models. The column roles are **not** in the plan: like the group column of a ranking fit, they belong to the call and the run record, not to the bytes of the model **[choice]**; the schema (section 5) records the *names* of the id columns so that `predict` can say what the file expects.

## 5. The model file [choice]

A `CML1` file with the sections of `wiring.md` section 4: `plan`, `schema`, `steps`, `model`, `meta`.

* `schema`: the columns of the log the model was fitted on, in the order user, item, rating (when explicit), with the user and item columns typed 3 (categorical, reserved by `wiring.md` section 3) and the rating a float; the schema has no target. `describe` reads the roles from the order **[choice]**.
* `steps`: `u32 0`.
* `meta`: `rows` = the interactions kept after duplicates, `columns` = 2 or 3, the seed, the pin, then `u32 sum_order = 2`.
* `model`: one payload, little-endian, **8 bytes a word** [choice; corrected from the first draft, which mixed 4- and 8-byte fields and wrote the history's item numbers as `u32`]: the header of ten words (the kind `kinds.model_recommend()` = 10; the method, 0 explicit and 1 implicit; the factors `f`; the iterations; `U`; `I`; `nnz`; `lambda` and `alpha` as the bits of the doubles, `alpha` 0.0 when explicit; the seed), then the user ids (the byte count `B`, `U + 1` offsets from 0 to `B`, the `B` bytes), the item ids (the same), the history (`U + 1` offsets from 0 to `nnz`, `nnz` item numbers, `nnz` values as bits), `P` (`U x f` doubles) and `Q` (`I x f` doubles); no byte after (`lib/models/recommend/model.cho` has the layout with the offsets). The reader computes the size the payload must have **before it allocates**, from the header and the two byte counts, and checks: every field against its range and the limits of `recommend.md` section 2, `lambda` positive and finite, `alpha` positive and finite when implicit and 0.0 when explicit, each id 1 to 255 bytes and the ids of a table **strictly ascending bytewise**, each offset table non-decreasing from 0 to its total (an offset past the table is refused before it is used), every user at least one item, the items of a user strictly ascending and below `I`, every item in some history, every value finite and within 1e6 (implicit: above 0), and every double of `P` and `Q` finite and reproduced bit for bit. A payload shorter than its fields say is `ml.model.truncated`; every other failure, including a payload longer than its fields say, is `ml.model.malformed` with `section` 4. The plan's method and factors must agree with the payload's (checked by the verbs, with the plan).
* The file holds the ids and the interactions of the log. `describe` and the answer of `fit` say so, and `docs/stringtable.md` section 1's privacy point applies in full **[the issue]**.
* **The summation order field** records **3** for this kind (`lib/core/kinds.cho` `sum_order_for`): the sums that make the factors, the normal equations and the losses, and the dot products of the scores, are sequential left folds in a fixed order (`recommend.md` section 4). The Cholesky and the two triangular solves inside each solve are the library's and use its lane order (`docs/linalg.md` section 2.1); that order depends on the size of the system, which the file fixes, and on no thread count, and the field does not claim otherwise, as the k-means file's 2 does not claim there is no sequential sum **[choice, settled at the registry commit of this stage]**. A reader that does not know the order 3 refuses the file with `ml.model.version`.

## 6. Refusals [choice]

| tag | exit | when |
|---|---|---|
| `ml.recommend.missing-id` | 8 | an empty id cell (`row`, `column`) |
| `ml.recommend.bad-rating` | 8 | a rating that is not a number, not finite, above 1e6 in magnitude, or, implicit, not above 0 (`row`) |
| `ml.recommend.duplicate-pair` | 8 | an explicit pair twice (`row`, `row2`); the repair says to aggregate the log first |
| `ml.recommend.unknown-user`, `ml.recommend.unknown-item` | 8 | a user or an item of `predict` the file does not hold (`row`, `id`); the repair names `--cold popular` as not yet available |
| `ml.limit.users`, `ml.limit.items`, `ml.limit.interactions` | 8 | the limits of `recommend.md` section 2, with the number and the ceiling |
| `ml.limit.eval-work` | 8 | `users x items x factors` above the ceiling of `recommend.md` section 3 |
| `ml.linalg.not-positive-definite` | 8 | existing; the repair names `--l2` |

The rest are existing tags (`args.*`, `ml.input.*`, `ml.param.*`, `ml.model.*`, `ml.plan.*`, `ml.schema.mismatch`). Each new tag has a fixture and a row in the table of rules, and `status.md` counts them when they exist.

## 7. Threads

The serial stage does not use them: `--threads` is accepted (1 to 16) and changes no byte (WR8). Threads over the independent solves of a half-pass are C3, with the same file as bytes.

## 8. The oracle

`scripts/oracle_wire_recommend.py` reuses `oracle_recommend.py` for the log (the numbering, the sums of duplicates) and `scripts/oracle_linalg.py` for the Cholesky; the normal equations are accumulated in the order of `recommend.md` section 4, the solves are the library's Cholesky (ported), the initial factors are the draws of stream kind 12 (ported from `scripts/oracle_rng.py`). It writes the whole file and the answers of `predict` and `eval`.

## 9. Admission and the module contract

`lib/models/recommend/model.toml` declares `shape = "blob"` **[open: the blob adapter assumes a matrix `xt` of features; the recommender has none, so the manifest probably needs a new shape, `log`, with the dispatch generated by `gen_models.py` calling `recommend.adapter.fit_log`/`predict_log`; the first commit that has code settles this and this section is corrected]**, its limits, its tags, its flags and its oracle; `scripts/admit.py recommend` runs the checks of `docs/modules.md`.

## 10. Open questions

* **[open]** The new manifest shape of section 9.
* **[open]** Whether `eval` should offer a sample of users (`--sample N`) before the ceiling `ml.limit.eval-work` is measured.

## 11. Results

Empty: filled by the commits that have code, from what their scripts printed.

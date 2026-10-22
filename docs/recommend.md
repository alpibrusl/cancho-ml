# cancho-ml recommendation (issue 70, stage M30)

**Status: a specification, written BEFORE any code of this stage.** Every number that a script of this repository has not printed is a **[choice]** or an **[estimate]**; nothing here is a measurement. Claims about other libraries are not made. Tags as in `design.md`: **[choice]** a decision the maintainer can overturn, **[read]** read from a document or the code, **[estimate]** a figure derived on paper, to be measured by the stage named, **[open]** not decided.

The decisions of the issue (alpibrusl/cancho-ml#70) are binding: two classical, batch methods (ALS and item-item neighbours), deterministic from the log, the seed and the plan; every sum in a fixed order; the same bytes at any thread count; every limit refused before allocation; ids interned in sorted order; the ranking metrics shared with `docs/ranking.md`. Online updating, a neural retriever, session sequence models and side information are not in this stage.

## 1. Scope and order

| stage | what | why first |
|---|---|---|
| C1 | **The log and its metrics.** The reader of `user, item[, rating][, time]` with the interning of ids, the limits, the splits per user (`--split-user-last N`), and the recommendation metrics (hit rate, recall, NDCG, MRR at `k`) as a library with its oracle. No model yet | everything after judges itself with these; the metrics are the part that can be wrong without anyone noticing |
| C2 | **ALS, explicit and implicit**, serial, `ml fit --model recommend --method als`, the model file, `describe`, `predict --for` (top-n for a user), `eval`. The oracle compares the factors as bytes | the serial version is the reference the threaded one is held to |
| C3 | **Threads** over the independent solves of a half-pass, the same bytes as C2 at 1, 2, 4, 8 threads | invariance by construction, judged by a test |
| C4 | **Neighbours**: item-item cosine with shrinkage, the k-nearest lists in the file, `predict --for` and `--like` | the second method, on the reader, the file and the surfaces C2 made |
| C5 | **Surfaces**: cold start (`--cold`), `ml-serve` by user id, `ml-mcp`, JSON lines, the tool schemas | the surfaces that exist already, each with its byte-equality gate |

Each stage ends with its results in section 10, from a script that printed them. A stage that finds a claim here false corrects it in place.

## 2. The data

A row of the log is `user, item[, rating][, time]`. The columns are named by flags: `--user COL`, `--item COL`, `--rating COL` (explicit feedback; without it every row is an implicit event of value 1), `--time COL` (needed only by the splits per user, section 7). These columns are the model's whole input: there is no feature matrix and no target, so `--target` is `args.conflict` with `--model recommend` **[choice]**.

**Ids** are read as text, whatever they look like: `007` and `7` are two ids, `1e3` is not a number to this model **[choice]**, so that no id depends on a numeric reading. The distinct user ids and the distinct item ids are each **sorted bytewise and numbered from 0**; the number of an id never depends on the order of the file. A missing id (an empty cell) is `ml.recommend.missing-id` (exit 8, `row` and `column` in `detail`), never an id of its own.

**Ratings** (explicit) are finite numbers with `|r| <= 1e6` **[choice]**; anything else, a NaN, an infinity or a cell that is not a number is `ml.recommend.bad-rating` (exit 8, `row`). The implicit form reads the rating column, if given, as the count or strength of the event and requires `r > 0` (`ml.recommend.bad-rating` otherwise); without the column every event is 1.

**Duplicates.** A pair `(user, item)` that occurs twice: with explicit ratings it is `ml.recommend.duplicate-pair` (exit 8, the two rows in `detail`; the repair says to aggregate the log first, since no rule here is right for every log); with implicit feedback the values are **summed** in file order, a count of events being what the form of Hu, Koren and Volinsky takes as input **[choice]**.

**The history.** The log, as a list of items per user sorted by item number with their values, is kept in the model file (section 5): `predict` excludes a user's seen items from the top-n and the neighbour method scores from them. **A model file of this kind therefore holds the ids and the interactions of its log**; the privacy point of `stringtable.md` 1 applies, stronger: the file is as sensitive as the log. `describe` says so **[choice]**.

**Limits**, each refused before anything sized by it is allocated, with a rule tag and the number in `introspect` **[choice; to be measured at C1]**: users `<= 2^20` (`ml.limit.users`), items `<= 2^20` (`ml.limit.items`), interactions `<= 2^25` (`ml.limit.interactions`), factors `<= 256`, iterations `<= 1000`, neighbours per item `<= 256`. The memory of a fit is, on paper **[estimate]**: the log twice (once by user, once by item, for the two half-passes), 16 bytes an interaction in each (an index and a value, 8 bytes each in cancho's integers and doubles), so `32 * nnz`; the factors `8 * f * (U + I)`; and one `f x f` work matrix per thread. At the ceiling of `2^25` interactions that is 1 GiB for the log copies before the reader's own text. C1 measures the real figure of the reader and of a fit on this machine and corrects this paragraph and the ceilings from the numbers.

## 3. The metrics (stage C1)

For each user with at least one **held-out** item (the "test items" of the user), the model ranks **all items except the user's training items**, by score descending, ties by **item number ascending** (that is, by id in bytewise order). With `T_u` the set of the user's test items, `R_u` the top-`k` list:

* **hit rate@k**: 1 if `R_u` holds any item of `T_u`, else 0. The mean over users is reported.
* **recall@k**: `|R_u ∩ T_u| / |T_u|`.
* **NDCG@k**: binary gain, the discount `1 / log2(r + 1)` of `docs/ranking.md` section 3 (`std.math.log`, the same table), `IDCG` over the first `min(k, |T_u|)` ranks.
* **MRR**: `1 / r` for the best-ranked test item over the **whole** ranking (not cut at `k`), 0 never: a user whose test items are all in the list gets a rank.

Each mean is over the users **with a held-out item**; the users in the test file without one, and users the model has never seen (cold), are not scored and are counted (`users`, `users_scored`, `users_cold`); the rule for a user who is cold is in section 6. Sums are sequential in user-number order, so the metric is a function of the data and the model. The code of `lib/metrics/ranking.cho` (the sort, the discount table, the gains) is reused where it fits and not copied; the new function takes a ranked list of item numbers and a set of test items.

**The cost of `eval`** is `users * items * factors` multiply-adds to score every item for every user. A ceiling `ml.limit.eval-work` of `2^34` multiply-adds **[estimate, to be measured at C2]** is refused before the first score; the repair says to evaluate a sample of users (`--users-from`/`--users-to` of the held-out file, i.e. a smaller file).

## 4. The methods

### 4.1 ALS, explicit (stage C2)

Factors `P` (users x f) and `Q` (items x f). The loss is `sum over observed (u, i) of (r_ui - p_u . q_i)^2 + lambda (|P|^2 + |Q|^2)`, `lambda = --l2 > 0` (a ridge term; `--l2` must be positive **[choice]**, so that every solve is positive definite). One **pass** is a user half-pass then an item half-pass:

* user `u`: `A = sum over its items i of q_i q_i^T + lambda I`, `b = sum of r_ui q_i`, the items in **item-number ascending order**, every accumulator a sequential sum from +0.0; `p_u = A^-1 b` by the Cholesky of `lib/linalg` (`linalg.cholesky` then `solve`) **[read]**;
* item `i`: the same with the users in user-number order.

The number of passes is `--iterations`, **fixed**: there is no convergence test, no tolerance the caller did not give **[the issue]**. The first user half-pass uses `Q_0`, the initial item factors; `P` has no initial value. `Q_0[i][c]` is the `c`-th draw (`c = 0 .. f - 1`) of the stream `derive(derive(seed, 12), i)`, as `unit(draw) / sqrt(f)` **[choice]**; stream kind 12 is the next one of `docs/rng.md` and is registered with this stage. A user or an item with no interaction cannot occur (the ids come from the log). A user with a single item is a rank-one system made positive definite by `lambda`.

A Cholesky that fails is `ml.linalg.not-positive-definite` (existing, exit 8); here its repair names `--l2` (increase it) **[the issue]**.

### 4.2 ALS, implicit (stage C2)

The form of Hu, Koren and Volinsky: preference `p_ui = 1` where the user has an event, 0 elsewhere; confidence `c_ui = 1 + alpha * r_ui` (`r_ui` = 0 where there is no event). `--alpha > 0`, default 40.0 **[choice: the figure the form is usually given with; no claim is made about its quality here]**. For the user half-pass, with `Y` the item factors,

`x_u = (Y^T Y + sum over the user's events of (c_ui - 1) y_i y_i^T + lambda I)^-1 * sum over the user's events of c_ui y_i`

so that the cost is not `users * items`: `Y^T Y` is computed **once per half-pass**, in item-number order, and a user adds only its events, in item-number order. The item half-pass is the transpose. Everything else is as in 4.1. The method is chosen by `--method als` with `--implicit` (a flag with no value) **[choice]**; without it, the explicit form.

### 4.3 Neighbours (stage C4)

Item-item cosine with shrinkage: with `v_i` the item's vector of values over the users (a missing value is 0; explicit ratings are used as they are, implicit values as summed),

`sim(i, j) = (v_i . v_j) / (|v_i| * |v_j| + s)`, `s = --shrinkage >= 0`, default 0 **[choice]**

and the `K = --neighbours` items with the largest `sim` (ties: the smaller item number), excluding the item itself and any item with `sim <= 0`, are stored in the file in order. The dot products come from the users' lists: for each user in user-number order, for each pair of its items in item-number order, the products are added to the pair's accumulator, so each accumulator is a sequential sum in user-number order. The work is `sum over users of d_u^2`, `d_u` the user's number of items; a log whose `sum d_u^2 / 2` pairs exceed `2^30` is `ml.limit.cooccurrence` **[choice, to be measured at C4]**.

The **score** of item `i` for user `u` is `sum over j in history(u) of sim(i, j) * r_uj`, taken only over pairs that are in `j`'s stored list (a neighbour list is all the file knows), the items in item-number order. The scores of an item that is nobody's neighbour are 0.

### 4.4 Threads (stage C3)

The solves of one half-pass are independent. They are divided among the threads in fixed contiguous blocks of users (or items), each block written to its own rows of the factor matrix, no sum shared between threads; so the result is the same at any thread count **by construction**, and a test holds it to that. `Y^T Y` is computed by one thread. The serial version is gated first.

## 5. The model file

One new model kind (the next number of `lib/core/kinds.cho`, section 4.1 of `design.md`) for both methods; the method is a field of the first record, as the task is for the tree **[choice: the id tables, the history and the checks are the same; two kinds would duplicate them]**. The payload, after the common sections (`wiring.md`):

1. header: method (0 ALS explicit, 1 ALS implicit, 2 neighbours), `U`, `I`, `nnz`, `f` (0 for neighbours), `K`, the doubles `lambda`, `alpha`, `s`, the iterations kept and requested, the seed;
2. the **user ids** and **item ids**: the string table of `stringtable.md` (sorted bytewise, length-prefixed), reused;
3. the **history**: for each user, its items (item numbers, ascending) and values (doubles), as offsets and arrays;
4. ALS: `P` and `Q` as doubles, row-major (the bits of the doubles, so the bytes are a function of the computation); neighbours: for each item, its `K` neighbours (item numbers) and similarities, in order;
5. the digest and the checks of the common sections.

The reader checks, before it allocates, every count against the limits of section 2 and against the file's length, that the id tables are strictly sorted, every history is strictly ascending with numbers below `I`, every neighbour list likewise, and that every double is finite; a damaged file is `ml.model.malformed` (existing) at the first check that fails. A file of another kind keeps its bytes.

## 6. The surfaces

* **`fit`**: `ml fit --model recommend --user U --item I [--rating R] [--time T] --method als|neighbours [--implicit] --factors F --l2 L --iterations N [--alpha A] [--neighbours K] [--shrinkage S] [--split-user-last N] --seed S --output M LOG`. `--threads` applies from C3. `--factors`, `--l2`, `--iterations` are bounded flags with the ranges of section 2 (`args.bad-value` outside). Defaults **[choice]**: `--factors 32`, `--l2 0.1`, `--iterations 10`, `--neighbours 50`.
* **`predict`**: the input is a file with a column of ids. **`--for COL`**: the column holds users; the answer has, for each input row, its `--top N` items (default 10) not in the user's history, as rows `user, rank, item, score` (`columns: ["user","rank","item","score"]`, ids as strings, rank 1 = best, ties by item number); `--keep-seen` keeps the user's own items. **`--like COL`**: the column holds items; the answer is, for each, its `N` nearest items (`item, rank, neighbour, score`; for ALS the cosine of the factor vectors, for neighbours the stored list). The shape of the answer (several rows for one input row) is new; `output.columns` of `describe` says it.
* **Cold start.** A user (or item) the model has never seen is `ml.recommend.unknown-user` (`unknown-item`), exit 8, with the repair `--cold popular`. With **`--cold popular`** the answer for such a user is the `N` most interacted-with items (counts from the stored history, ties by item number), each such row marked `"cold": true`, never silently **[choice]**; an unknown item for `--like` has no fallback and stays a refusal.
* **`eval`**: `ml eval --model-file M TEST --user U --item I [--rating R] [--top K]` computes the metrics of section 3 on a test log; also `fit --split-user-last N` and `cv` report them on the held-out part.
* **`describe`**: the method, `U`, `I`, `nnz`, the density `nnz / (U * I)` (a count, not a statement of quality), `f`, `lambda`, `alpha`, `K`, and that the file holds the log.
* **`ml-serve`** (C5): the answer by user id on `POST /predict` and by a list of item ids on a new route for a session, read-only on one file; its own addendum to `serving-http.md`, written before the code **[open]**. **`ml-mcp`** and the JSON-lines mode follow the generated tool list.
* **`--track`** records the columns, the method and the seed; `runs verify` replays the fit.

## 7. Splits

The splits of `docs/splits.md` split rows; a recommender needs to hold out interactions **per user**. `--split-user-last N` (needs `--time T`, else `args.required-flag`) holds out each user's `N` latest interactions (ties in file order); a user with `N` or fewer interactions stays wholly in training and is not scored. This is the "leave-one-out per user" of the issue for `N = 1` **[choice]**. A **random** split of interactions is allowed (`--split train=F`) and the answer says `"leaks_future": true` **[the issue]**. The global `--split-by time:COL` of `splits.md` applies as written. Cross-validation by this split is not offered; the refusal is `ml.split.group-required`'s sibling, with its own tag, decided at C1.

## 8. Rule tags

New **[choice]**, each with an exit code, a repair where one is safe, and a fixture in the table of rules; `status.md` counts them when they exist: `ml.recommend.missing-id`, `ml.recommend.bad-rating`, `ml.recommend.duplicate-pair`, `ml.recommend.unknown-user`, `ml.recommend.unknown-item`, `ml.limit.users`, `ml.limit.items`, `ml.limit.interactions`, `ml.limit.cooccurrence`, `ml.limit.eval-work`. The refusals of `lib/linalg` and of the reader apply unchanged.

## 9. Gates (written first; each can fail)

| id | gate | judged by | a mutant that must die |
|---|---|---|---|
| RC1 | **The metrics against an independent oracle.** A Python implementation from the definitions of section 3 (no cancho code), on generated logs: users with one to many test items, ties in the scores, test items outside the top-`k`, users with none, `k` below, at and above the number of items. Every per-user value and every mean equals the oracle's **bits** | `scripts/oracle_recommend.py`, a check driver | the discount `log2(r)`; the tie broken by the larger item; the history not excluded; MRR cut at `k`; an unscored user counted as 0 |
| RC2 | **Ids and the reader.** Ids that look like numbers (`007`, `7`, `1e3`), the same id in two spellings, ids with quotes, commas, UTF-8, an empty id; the numbering equal to the sorted order whatever the order of the file (a shuffled log gives the same file) | tests, the oracle's interning | the numbering by first appearance |
| RC3 | **ALS against a dense NumPy oracle, bytes of the factors.** The oracle does the same updates in the same order (the same accumulation order, the Cholesky of `oracle_linalg.py`) for a fixed pass count on small logs (up to 40 users, 30 items, rank up to 6), explicit and implicit, with and without `--alpha`; `P` and `Q` equal as bytes | `scripts/oracle_wire_recommend.py` | the item order of a sum; `lambda` once instead of per solve; the implicit `c` for `c - 1`; `Y^T Y` per user |
| RC4 | **The implicit form against the formula**, the closed form solved densely for each user with all items (`users * items` small), within a bound fixed in the script before the run | the same script | the preference taken from the confidence |
| RC5 | **The loss falls.** The regularised loss after each half-pass is not above its value before it, by more than a rounding bound fixed in the script (a pass of exact ALS cannot raise it); a mutant that skips the last item of a sum fails it | `scripts/gate_recommend_loss.py` | a skipped term; the wrong sign of `b` |
| RC6 | **Planted structure.** A log drawn from a known low-rank `P*Q*^T` with noise; the held-out RMSE of the explicit fit within a bound **fixed in the script before the run**, and the planted items of an implicit log among the top-`k` for most users, bound likewise; a log of no structure must not meet it | the script | the factors not updated; one pass instead of `N` |
| RC7 | **Thread invariance.** The same bytes at 1, 2, 4, 8 threads, for both forms and neighbours | a test | a sum shared across blocks |
| RC8 | **Neighbours against the oracle**, bytes of the lists, and the score of a user against a direct sum | the oracle | the shrinkage in the numerator; ties by the larger item; the item as its own neighbour |
| RC9 | **Cold start and seen-items controls**: an unknown user is `ml.recommend.unknown-user`; `--cold popular` equals the popularity list; a user's seen items are never in the top-n unless `--keep-seen` | tests | the history not excluded; the fallback silent |
| RC10 | **Hostile logs** (fuzz): duplicate pairs, a user with one item, an item with one user, a rating outside its range, `NaN`, ids that look like numbers, a log over each limit (refused before allocation), a single user, a single item, all ratings equal | `test_fuzz_*` | a check removed |
| RC11 | **Damaged files**: every count, every offset and every sort order of section 5 broken in turn, each a tag, no trap | `test_wire_recommend_file.py` (the stride of the other file gates) | a check removed |
| RC12 | **Splits**: `--split-user-last` holds out exactly each user's latest `N`, a user with at most `N` stays in training, the leak is reported for a random split | tests, the oracle | the order of ties at the cut |
| RC13 | **Mutants** of the code of every stage: all killed, or each survivor explained | `scripts/mutants_recommend.py` | itself |
| RC14 | **`ml-serve` byte equality** with the command line, `describe`, the tool schemas of `ml-mcp`, the tables regenerated and checked, admission as for every model | existing checks | |

No claim of quality against another library is made unless a script here measured it on named data and a named machine.

## 10. Results

Empty: filled by each stage, from what its scripts printed.

## 11. Open questions

* **[open]** The ceilings of section 2 (`2^20` users and items, `2^25` interactions, `2^34` multiply-adds for `eval`, `2^30` co-occurrence pairs) are estimates; C1 and C2 measure the reader, a fit and an evaluation on one core and correct this document.
* **[open]** Whether an explicit-feedback fit should offer the weighted regularisation (`lambda` times the user's or the item's number of ratings) as an option; not built, and no claim is made for or against it.
* **[open]** Whether duplicate explicit ratings should have an aggregation flag (`--duplicates last|mean`) rather than a refusal.
* **[open]** The session route of `ml-serve` (section 6) and the shape of its request.
* **[open]** Storing the history is what makes a file as sensitive as the log; whether `predict` should also accept the history in the input, with a model that stores none, is a later option.

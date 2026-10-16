<p align="center"><img src="docs/assets/cancho-ml-logo.jpg" alt="cancho-ml" width="220"></p>

# cancho-ml

**Classical machine learning on tabular data, as a command line an agent can read.** `ml` fits, applies and scores a model on a CSV with explicit semantics, exact sums where a sum decides the result, bounded resources, structured refusals and a machine-verifiable authority. It is a command line, not a chat box: you (or an agent) give it flags, it gives back one JSON line, or a named refusal with a repair. One small binary, written in [cancho](https://github.com/alpibrusl/cancho), built the way [cancho-table](https://github.com/alpibrusl/cancho-table) is. The [project page](https://alpibrusl.github.io/cancho-ml/) has the same in pages.

**Status: early, alpha, not for production.** `ml` builds seven models. `--model constant` is the baseline: the training mean for a regression, the majority class for a classification. `--model linear` (least squares, ridge with `--l2`), `--model logistic` (binary, Newton's method), `--model tree` (a CART decision tree with histogram splits) and `--model forest` (bagged trees, threaded over trees; `--task`, `--trees`, `--max-depth`, `--min-leaf`, `--bins`, `--mtry`, `--bootstrap`, `--max-nodes`) `--model boost` (gradient-boosted trees: the squared-error or the logistic loss, `--task`, `--rounds`, `--learning-rate`, `--max-depth`, `--min-leaf`, `--bins`, `--l2`, `--subsample`, `--early-stopping`) and `--model kmeans` (a clustering with no target: `--k` centres, a fixed number of Lloyd `--iterations`, a seeded `--init`) go through a pipeline of steps (`impute`, `scale`, `encode`, `bucket`, and `freq` and `hash` for text) that `--step` or `--plan` describes and that is fitted on the training rows and stored, with the model, in **one** file: `predict` and `eval` apply the stored steps, and no flag can apply another transformation. `ml` reads integer, decimal and float CSV columns, takes an empty cell as a missing value that the plan must impute, holds rows out with a seeded `--split`, threads `fit` with `--threads`, gives probabilities with `predict --proba`, and scores with accuracy, precision, recall, F1, ROC-AUC, log-loss, RMSE, MAE and R2 (and, for k-means, the inertia). Text columns declared with `--cat` are read into a bounded dictionary, and the steps `encode` (one-hot, label), `freq` (the share of the training rows) and `hash` (SHA-256, a fixed number of buckets) turn them into numbers, with a rare-category bucket (`min-count`, the category `__other__`) and rules for an unseen string (`unknown`) and an empty cell (`missing`). The model file of such a fit holds a **string table**: the fitted vocabulary of every categorical step, with the training strings and their counts, so `predict` and `eval` read the text columns by name from the file alone, count a string the training rows never held, and hash it into its bucket under a `hash` step. A declared column that no step reads is refused (`ml.input.categorical-not-encoded`). A tree, forest or boosted model takes a missing cell itself and learns a side for it at each split, where linear and logistic refuse a missing cell that no `impute` step covers (`ml.input.missing-value`); a tree, forest or boosted fit takes at most 2^20 (1,048,576) training rows (`ml.input.too-many-rows`). A tree or forest file is model kind 7, and a boosted file is model kind 8, both with the summation order 3 (sequential sums: no lane sum reaches their bytes). k-means has no target: `fit` needs no `--target` and refuses one, the file's schema has no target column, `predict` prints the cluster of each row (and with `--distance` the squared distance to its centre) and `eval` the inertia, the rows of each cluster and the mean distance of each; a missing cell is refused as for linear and logistic. A k-means file is model kind 6, with the summation order 2. A model is a module (a manifest, a fixed adapter, generated glue, an admission script) so that the next one cannot add a float order, a capability or an unregistered refusal: [docs/modules.md](docs/modules.md). One pre-registered gate is red (tree regression against scikit-learn, below). The whole list, with what each item was measured as: [docs/status.md](docs/status.md).

A GitHub Actions workflow, `.github/workflows/ci.yml` ([docs/ci.md](docs/ci.md)), builds the pinned compiler and `ml`, `ml-predict` and `ml-serve` on `ubuntu-24.04` and `ubuntu-26.04` (Linux x86-64), `ubuntu-24.04-arm` (Linux aarch64) and `macos-26` (macOS arm64) and runs the fast gates and a cross-platform determinism gate on each; the full gate runs on request and nightly, and a tag `v*` publishes the binaries. `scripts/check.sh` runs every gate, and `scripts/check_docs.sh` checks this README and the project pages (see Contributing).

## One complete flow

Fit, predict, score, get refused with a repair, run the repair. The repair is the command with the mistake taken out, and the output below is what the built program prints, on the small files shown under "What you can do".

<!-- gen:flow -->
```console
# Fit the training mean of price. area has a decimal point: it is a float column, read exactly
$ ml fit --target price --output houses.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":5,"columns":4,"target":"price","path":"houses.ml","bytes":416,"sha256":"df669ebf009e182d179668967f9c248bc3bfed3383141089bb8759e5c569c62a","written":true,"summary":{"mean":"218000.000000"}},"meta":{"version":"0.1.0"}}
# Predict for two new rows, which have no price
$ ml predict --model-file houses.ml new.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","columns":["prediction"],"rows":[["218000.000000"],["218000.000000"]],"row_count":2,"total_rows":2,"truncated":false,"next":null},"meta":{"version":"0.1.0"}}
# Score the model on the file it was fitted on
$ ml eval --model-file houses.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":5,"on":"train","metrics":{"rmse":"71035.202541","sse":"25230000000.000000000000","prediction":"218000.000000"}},"meta":{"version":"0.1.0"}}
# New rows with a column of text are refused
$ ml predict --model-file houses.ml listings.csv
{"rule": "ml.input.non-numeric", "hint": "drop the column with --drop, or fix the cell", "repair": {"kind": "retry", "argv": ["ml", "predict", "--model-file", "houses.ml", "listings.csv", "--drop", "city"]}}
# exit status 8; the rule, hint, repair of the JSON line it prints
# The tool offered the corrected command. Run it:
$ ml predict --model-file houses.ml listings.csv --drop city
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","columns":["prediction"],"rows":[["218000.000000"],["218000.000000"]],"row_count":2,"total_rows":2,"truncated":false,"next":null},"meta":{"version":"0.1.0"}}
```
<!-- /gen:flow -->

## A model through the pipeline

A float CSV with an empty cell. A plan has to say what to do with it, or the fit is refused with a repair. Then the plan, written out: impute, standardise, a logistic regression; predict with probabilities (the empty cell of the new rows is filled with the mean **stored in the model**); score. The output is what the built program prints, on the files in `scripts/site.py`.

<!-- gen:pflow -->
```console
# A float CSV with an empty cell: the plan must say what to do with it. The refusal offers a repair
$ ml fit --model logistic --target bought --output shop2.ml shop2.csv
{"rule": "ml.input.missing-value", "hint": "add an impute step for the columns, for example impute column=a,b strategy=mean.", "repair": {"kind": "retry", "argv": ["ml", "fit", "--model", "logistic", "--target", "bought", "--output", "shop2.ml", "shop2.csv", "--step", "impute column=spend strategy=mean"]}}
# exit status 8; the rule, hint, repair of the JSON line it prints
# The repair works, but a plan can say more: impute, standardise, then a logistic regression (50 Newton iterations, ridge 0.1)
$ ml fit --model logistic --overwrite --step 'impute column=spend strategy=mean;scale column=visits,spend kind=standard' --iterations 50 --l2 0.1 --target bought --output shop2.ml shop2.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","rows":12,"columns":4,"target":"bought","path":"shop2.ml","bytes":885,"sha256":"eda24be438ac59a45d63112c4c910fabdb9ba1ec009d4b34e5c870f6e107f4c6","written":true,"summary":{"plan":"impute column=spend strategy=mean\nscale column=spend,visits kind=standard ddof=0\nmodel kind=logistic iterations=50 l2=0.1\n","steps":2,"features":3,"coefficients":[{"name":"id","value":"-0.08218688808973369"},{"name":"visits","value":"0.25761586355528143"},{"name":"spend","value":"1.1474347278761494"}],"intercept":"0.6023363690462818","classes":[0,1],"positive":1}},"meta":{"version":"0.1.0"}}
# Predict with probabilities for new rows (the empty cell is filled with the mean STORED in the model)
$ ml predict --model-file shop2.ml --proba new2.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","columns":["label","probability"],"rows":[["0","0.36645031865861677"],["1","0.8502846389524489"],["0","0.0906646088784972"]],"row_count":3,"total_rows":3,"truncated":false,"next":null,"unknown_categories":0,"out_of_range":{"cells":3,"rows":3,"columns":[{"name":"id","count":3}]},"ignored_columns":[]},"meta":{"version":"0.1.0"}}
# Score it: accuracy, precision, recall, F1, ROC-AUC, log-loss
$ ml eval --model-file shop2.ml shop2.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","rows":12,"on":"train","positive":1,"metrics":{"accuracy":"0.8333333333333334","precision":"0.8333333333333334","recall":"0.8333333333333334","f1":"0.8333333333333334","roc_auc":"0.9444444444444444","log_loss":"0.4012542324769021"},"unknown_categories":0,"out_of_range":{"cells":0,"rows":0,"columns":[]}},"meta":{"version":"0.1.0"}}
```
<!-- /gen:pflow -->

## k-means: clusters, and no target

k-means has no target: `fit` needs no `--target` and refuses one, `--k` (1 to 1,024) is required, `--iterations` (default 10) is a fixed number of Lloyd iterations, `--init kmeans++|random` draws the first centres from `--seed`. A file with a label column is used without it (`--drop`), the features are standardised by a step, and a cell that is still missing after the steps is refused, as for linear and logistic. `fit` prints the inertia, the rows of each cluster and the mean distance of each; `predict` prints the cluster of each row (`--distance` adds the squared distance to its centre); `eval` prints the same three numbers for another file, or for the rows `--split` holds out. The inertia is a sum of squared distances in the package's lane order, so the same data, plan and seed give the same bytes at every `--threads`. On the committed `docs/examples/data/iris_train.csv` and `iris_test.csv` (copy them next to where you run it):

<!-- gen:kmflow -->
```console
# k-means has no target: it clusters the rows by their features. The label column of iris is not given as one
$ ml fit --model kmeans --k 3 --target virginica --output iris-km.ml iris_train.csv
{"rule": "args.conflict", "hint": "see the verbs and their flags in introspect usage", "detail": {"flag": "target"}}
# exit status 2; the rule, hint, detail of the JSON line it prints
# k is required
$ ml fit --model kmeans --drop virginica --output iris-km.ml iris_train.csv
{"rule": "ml.plan.missing-key", "hint": "add the key.", "detail": {"line": 1, "column": 1}}
# exit status 8; the rule, hint, detail of the JSON line it prints
# Drop the label, standardise the four measurements, fit three clusters (seed 1): the answer holds the inertia, the rows of each cluster and the mean distance of each
$ ml fit --model kmeans --k 3 --seed 1 --drop virginica --step 'scale column=* kind=standard' --output iris-km.ml iris_train.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"kmeans","task":"clustering","rows":75,"columns":4,"target":null,"path":"iris-km.ml","bytes":916,"sha256":"ae4d43ad9f70c284200db9e3fd385d7a1e32870fd86495297ae2dfd165e82056","written":true,"summary":{"plan":"scale column=* kind=standard ddof=0\nmodel kind=kmeans k=3 iterations=10 init=kmeans++\n","steps":1,"features":4,"clusters":{"k":3,"iterations":10,"init":"kmeans++","rows":75,"metrics":{"inertia":"114.01824275090962","sizes":[26,31,18],"mean_distance":["1.2398733702280837","1.0644974060896137","1.1737313271562995"]}}}},"meta":{"version":"0.1.0"}}
# The cluster of five test rows and the squared distance of each to its centre
$ ml predict --model-file iris-km.ml --distance --drop virginica --limit 5 iris_test.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"kmeans","task":"clustering","columns":["cluster","distance"],"rows":[["0","9.531664854091233"],["0","2.5630208525939238"],["1","2.976542017495812"],["1","0.887839091686247"],["2","0.33871718188300526"]],"row_count":5,"total_rows":25,"truncated":true,"next":{"from":5},"unknown_categories":0,"out_of_range":{"cells":3,"rows":2,"columns":[{"name":"sepal_length_cm","count":1},{"name":"sepal_width_cm","count":2}]},"ignored_columns":[]},"meta":{"version":"0.1.0"}}
# The inertia of the 25 test rows, the rows of each cluster and the mean distance of each
$ ml eval --model-file iris-km.ml --drop virginica iris_test.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"kmeans","task":"clustering","rows":25,"on":"other","clusters":3,"metrics":{"inertia":"47.3765627878906","sizes":[9,11,5],"mean_distance":["1.568635332727916","1.1181940244601365","0.6890412730032826"]},"unknown_categories":0,"out_of_range":{"cells":3,"rows":2,"columns":[{"name":"sepal_length_cm","count":1},{"name":"sepal_width_cm","count":2}]}},"meta":{"version":"0.1.0"}}
# The same, as text, on the training rows
$ ml --format text eval --model-file iris-km.ml --drop virginica iris_train.csv
on train
inertia 114.01824275090962
sizes 26 31 18
mean_distance 1.2398733702280837 1.0644974060896137 1.1737313271562995
```
<!-- /gen:kmflow -->

## Track runs and check for skew

A model file remembers what it was fitted on. Every `fit` writes a **provenance section** (tag 7 of the file): the SHA-256 and size of the training file, its rows, and for each numeric feature the count, minimum, maximum, exact mean and population standard deviation of the training cells as the reader read them (exact sums, rounded once, so no order of rows and no thread count changes a byte). `eval` prints `"on":"train"` when the file it scores is the file the model was fitted on, `"other"` when it is not, `"unknown"` for a model file with no section; `predict`, `eval` and `ml-serve` count the non-missing cells strictly outside the stored range (`out_of_range`, with the columns that have any), `--max-out-of-range N` refuses above N rows (`ml.predict.out-of-range`, naming the first column and row), and `predict` lists the target column it ignored (`ignored_columns`). Files without a section are read and say so.

`fit --track DIR` (a directory under `--root` that exists) writes `DIR/RUNID.json`: canonical JSON, no timestamp, no host name, no payload, whose id is the SHA-256 of the inputs that determine the model bytes (the training file's hash, the plan, the seed, the split, the lists, the compiler pin; not the thread count). The same inputs write the same record; a different record under an existing id is `ml.runs.conflict`. `ml runs list|show|compare DIR` read the records, and `ml runs verify DIR ID` refits in memory at any `--threads` and compares the SHA-256 of the model file. The design and its gates are in [docs/tracking.md](docs/tracking.md). [docs/board.html](docs/board.html) is a read-only page for these records: drop `DIR/*.json`, or the output of `ml runs list` or `show`, on it for a table, a comparison and a chart; it recomputes every run id in the browser and sends nothing (design: [docs/board.md](docs/board.md)).

<!-- gen:trackflow -->
```console
# Every fit can write a run record: canonical JSON, named by the hash of the inputs that determine the model (not the thread count), no timestamp, no payload
$ ml fit --model logistic --target virginica --iterations 25 --l2 0 --track runs --output iris-lr.ml iris_train.csv
{"path": "iris-lr.ml", "bytes": 704, "sha256": "e855b04f8717184c26345ebbfd395c47bb981ce3c74b188afcef699e71484195", "run": {"id": "519a70c001eb5f1e4e0c9e9165bce2e9dc7368d508556c389ad134b315e2a476", "dir": "runs", "file": "519a70c001eb5f1e4e0c9e9165bce2e9dc7368d508556c389ad134b315e2a476.json", "written": true, "sha256": "0448ad81f3a93b561258ecf6fd0077a92e4cfdbfbcbdc20bf0d3b377cdeb5a2b"}}
$ ml fit --model forest --task classification --trees 20 --seed 2 --target virginica --track runs --output iris-rf.ml iris_train.csv
{"path": "iris-rf.ml", "bytes": 17994, "sha256": "b9297048d4ff4f2d69950d6a6a6ef7e5ed992810f06f08dff46d69741bf2c98e", "run": {"id": "bc01713e1d29df7e77a5f6f59333a9a2a58d0024be1740a91057c3fc69dd5686", "dir": "runs", "file": "bc01713e1d29df7e77a5f6f59333a9a2a58d0024be1740a91057c3fc69dd5686.json", "written": true, "sha256": "6b756757e8ba99a6fa35c695205cbdc9c4b3d05a1a4efc9eb419fe42081a2cb7"}}
# The directory lists the runs; a run is shown whole, and two are compared by the inputs that differ and the metrics side by side
$ ml runs list runs
{"count": 2, "runs": [{"id": "519a70c001eb5f1e4e0c9e9165bce2e9dc7368d508556c389ad134b315e2a476", "model": "logistic", "task": "classification", "data_sha256": "e510e80e5dccc8297dd2901ca624f7a81da274db462699dfab4eb961610a854b", "model_sha256": "e855b04f8717184c26345ebbfd395c47bb981ce3c74b188afcef699e71484195", "train_rows": 75, "metrics_on": "train", "evals": 0}, {"id": "bc01713e1d29df7e77a5f6f59333a9a2a58d0024be1740a91057c3fc69dd5686", "model": "forest", "task": "classification", "data_sha256": "e510e80e5dccc8297dd2901ca624f7a81da274db462699dfab4eb961610a854b", "model_sha256": "b9297048d4ff4f2d69950d6a6a6ef7e5ed992810f06f08dff46d69741bf2c98e", "train_rows": 75, "metrics_on": "train", "evals": 0}]}
$ ml runs show runs 519a70c001eb5f1e4e0c9e9165bce2e9dc7368d508556c389ad134b315e2a476
{"ok":true,"command":"ml","schema":"ml.v1","data":{"record":{"schema":"ml.run.v1","id":"519a70c001eb5f1e4e0c9e9165bce2e9dc7368d508556c389ad134b315e2a476","inputs":{"data":{"path":"iris_train.csv","bytes":1422,"sha256":"e510e80e5dccc8297dd2901ca624f7a81da274db462699dfab4eb961610a854b","rows":75},"target":"virginica","task":"classification","plan":"model kind=logistic iterations=25 l2=0.0\n","seed":0,"positive":null,"…
$ ml runs compare runs 519a70c001eb5f1e4e0c9e9165bce2e9dc7368d508556c389ad134b315e2a476 bc01713e1d29df7e77a5f6f59333a9a2a58d0024be1740a91057c3fc69dd5686
{"model_equal": false, "inputs": [{"key": "plan", "a": "model kind=logistic iterations=25 l2=0.0\n", "b": "model kind=forest task=classification trees=20 max-depth=8 min-leaf=1 bins=64 mtry=0 bootstrap=1 max-nodes=4194304\n"}, {"key": "seed", "a": 0, "b": 2}, {"key": "sum_order", "a": 2, "b": 3}]}
# A run is replayable: verify refits in memory and compares the SHA-256 of the model file, at any thread count
$ ml runs verify runs 519a70c001eb5f1e4e0c9e9165bce2e9dc7368d508556c389ad134b315e2a476
{"verified": true, "model_sha256": "e855b04f8717184c26345ebbfd395c47bb981ce3c74b188afcef699e71484195", "threads": 1}
$ ml runs verify --threads 8 runs bc01713e1d29df7e77a5f6f59333a9a2a58d0024be1740a91057c3fc69dd5686
{"verified": true, "model_sha256": "b9297048d4ff4f2d69950d6a6a6ef7e5ed992810f06f08dff46d69741bf2c98e", "threads": 8}
# eval says whether it scored the file the model was fitted on, and counts the cells outside the range of the training rows
$ ml eval --model-file iris-lr.ml --track runs iris_train.csv
{"rows": 75, "on": "train", "metrics": {"accuracy": "0.9733333333333334", "precision": "0.9736842105263158", "recall": "0.9736842105263158", "f1": "0.9736842105263158", "roc_auc": "0.9950213371266002", "log_loss": "0.07256993734328859"}, "out_of_range": {"cells": 0, "rows": 0, "columns": []}}
$ ml eval --model-file iris-lr.ml --track runs iris_test.csv
{"rows": 25, "on": "other", "metrics": {"accuracy": "1.0", "precision": "1.0", "recall": "1.0", "f1": "1.0", "roc_auc": "1.0", "log_loss": "0.024399970263418697"}, "out_of_range": {"cells": 3, "rows": 2, "columns": [{"name": "sepal_length_cm", "count": 1}, {"name": "sepal_width_cm", "count": 2}]}}
# predict counts them too, and names the target column it ignored
$ ml predict --model-file iris-lr.ml --limit 2 iris_test.csv
{"row_count": 2, "unknown_categories": 0, "out_of_range": {"cells": 3, "rows": 2, "columns": [{"name": "sepal_length_cm", "count": 1}, {"name": "sepal_width_cm", "count": 2}]}, "ignored_columns": ["virginica"]}
# A bound on the rows outside the range refuses, naming the first column and row
$ ml predict --model-file iris-lr.ml --max-out-of-range 0 iris_test.csv
{"rule": "ml.predict.out-of-range", "hint": "check where the data comes from, or raise --max-out-of-range.", "detail": {"rows": 2, "max": 0, "column": "sepal_width_cm", "row": 1}}
# exit status 8; the rule, hint, detail of the JSON line it prints
```
<!-- /gen:trackflow -->

## Boosted trees: `--model boost`

`--model boost` fits additive trees to the gradients and hessians of a loss, round by round: squared error for a regression (`--task regression`, the default) and logistic loss for a binary classification (`--task classification`). Each round grows one histogram tree on the quantile bins of the tree library, with a Newton step `-G / (H + l2)` in each leaf, multiplied by `--learning-rate`; a cell that is missing is accepted and each split learns the side it takes. `--rounds` (default 100), `--learning-rate` (0.1), `--max-depth` (4), `--min-leaf` (10), `--bins` (64), `--l2` (1.0), `--subsample` (the fraction of the rows each round fits on, seeded from `--seed` and the round, never from the thread count) and `--early-stopping N` are the flags. Early stopping is refused unless `--split train=F` declares a held-out part, and the stopping rule never reads it: it takes a random tenth of the training rows for its own validation loss and keeps the best round. A random tenth is not a group split or a time split of the training rows, so `--early-stopping` with `--split-group` or `--split-by` (in `fit --split` and in `ml cv`) is `ml.split.combination`, and its repair drops `--early-stopping`; with a plain or a `--stratify` split, or in `ml cv` with random folds, it runs. Every sum is a sequential sum in a fixed row order, so the file records the summation order 3, as the tree and the forest do, and the same data, plan and seed give the same bytes at every `--threads` (the threads only bin the columns: the rounds are sequential). The file is model kind 8; a boosted model holds all its trees, so `fit` refuses to write a file longer than `--max-bytes`. On the committed `docs/examples/data/breast_cancer_train.csv` and `breast_cancer_test.csv` (copy them next to where you run it):

<!-- gen:bflow -->
```console
# early stopping keeps rows out of the fit, so it is refused unless a held-out part is declared with --split
$ ml fit --model boost --target benign --early-stopping 10 --output cancer-boost.ml breast_cancer_train.csv
{"rule": "args.conflict", "hint": "see the verbs and their flags in introspect usage", "detail": {"flag": "early-stopping"}}
# exit status 2; the rule, hint, detail of the JSON line it prints
# 100 rounds of depth-3 trees on 80 percent of the rows each, learning rate 0.1: the summary holds the rounds, the nodes and the base score
$ ml fit --model boost --task classification --target benign --rounds 100 --learning-rate 0.1 --max-depth 3 --subsample 0.8 --seed 1 --output cancer-boost.ml breast_cancer_train.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"boost","task":"classification","rows":426,"columns":31,"target":"benign","path":"cancer-boost.ml","bytes":57979,"sha256":"338fb4cd281bf7702a53467dfdb8156f2f43592c89ea13aa6e808d01f963d8c3","written":true,"summary":{"plan":"model kind=boost task=classification rounds=100 learning-rate=0.1 max-depth=3 min-leaf=10 bins=64 l2=1.0 subsample=0.8 early-stopping=0 max-nodes=4194304\n","steps":0,"features":30,"boost":{"rounds":100,"rounds_requested":100,"nodes":1134,"bins":64,"max_depth":3,"min_leaf":10,"learning_rate":"0.1","base_score":"0.518344456180018"},"classes":[0,1],"positive":1}},"meta":{"version":"0.1.0"}}
# the label and the probability of the positive label for three test rows
$ ml predict --model-file cancer-boost.ml --proba --limit 3 breast_cancer_test.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"boost","task":"classification","columns":["label","probability"],"rows":[["1","0.9989532673025714"],["0","0.0014020009583737808"],["0","0.0006485175287276667"]],"row_count":3,"total_rows":143,"truncated":true,"next":{"from":3},"unknown_categories":0,"out_of_range":{"cells":27,"rows":10,"columns":[{"name":"mean_radius","count":1},{"name":"mean_perimeter","count":1},{"name":"mean_area","count":1},{"name":"mean_smoothness","count":2},{"name":"mean_concavity","count":3},{"name":"mean_concave_points","count":1},{"name":"texture_error","count":1},{"name":"smoothness_error","count":1},{"name":"concavity_error","count":1},{"name":"concave_points_error","count":1},{"name":"symmetry_error","count":1},{"name":"fractal_dimension_error","count":2},{"name":"worst_radius","count":1},{"name":"worst_perimeter","count":1},{"name":"worst_area","count":1},{"name":"worst_smoothness","count":1},{"name":"worst_compactness","count":2},{"name":"worst_concave_points","count":1},{"name":"worst_symmetry","count":1},{"name":"worst_fractal_dimension","count":3}]},"ignored_columns":["benign"]},"meta":{"version":"0.1.0"}}
# accuracy, precision, recall, F1, ROC-AUC and log-loss of the 143 test rows
$ ml eval --model-file cancer-boost.ml breast_cancer_test.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"boost","task":"classification","rows":143,"on":"other","positive":1,"metrics":{"accuracy":"0.958041958041958","precision":"0.9666666666666667","recall":"0.9666666666666667","f1":"0.9666666666666667","roc_auc":"0.9872117400419287","log_loss":"0.131972703636791"},"unknown_categories":0,"out_of_range":{"cells":27,"rows":10,"columns":[{"name":"mean_radius","count":1},{"name":"mean_perimeter","count":1},{"name":"mean_area","count":1},{"name":"mean_smoothness","count":2},{"name":"mean_concavity","count":3},{"name":"mean_concave_points","count":1},{"name":"texture_error","count":1},{"name":"smoothness_error","count":1},{"name":"concavity_error","count":1},{"name":"concave_points_error","count":1},{"name":"symmetry_error","count":1},{"name":"fractal_dimension_error","count":2},{"name":"worst_radius","count":1},{"name":"worst_perimeter","count":1},{"name":"worst_area","count":1},{"name":"worst_smoothness","count":1},{"name":"worst_compactness","count":2},{"name":"worst_concave_points","count":1},{"name":"worst_symmetry","count":1},{"name":"worst_fractal_dimension","count":3}]}},"meta":{"version":"0.1.0"}}
# up to 500 rounds, stopped after 10 without a better loss on a tenth of the training rows; the test side of --split is never read by the stopping rule
$ ml fit --model boost --task classification --target benign --rounds 500 --learning-rate 0.2 --early-stopping 10 --split train=0.8 --seed 1 --output cancer-early.ml breast_cancer_train.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"boost","task":"classification","rows":340,"columns":31,"target":"benign","path":"cancer-early.ml","bytes":15020,"sha256":"6b601401f69856c6c889f5cfc9c48362aa4bf67f074d6bda1f488e503d27606b","written":true,"summary":{"plan":"model kind=boost task=classification rounds=500 learning-rate=0.2 max-depth=4 min-leaf=10 bins=64 l2=1.0 subsample=1.0 early-stopping=10 max-nodes=4194304\n","steps":0,"features":30,"boost":{"rounds":22,"rounds_requested":500,"nodes":252,"bins":64,"max_depth":4,"min_leaf":10,"learning_rate":"0.2","base_score":"0.37018329396352456","best_round":22},"classes":[0,1],"positive":1,"split":{"train":340,"test":86,"seed":1,"spec":"holdout train=0.8 seed=1"},"holdout":{"rows":86,"on":"held-out","metrics":{"accuracy":"0.9418604651162791","precision":"0.9666666666666667","recall":"0.9508196721311475","f1":"0.9586776859504132","roc_auc":"0.9954098360655738","log_loss":"0.09890371252033758"}}}},"meta":{"version":"0.1.0"}}
# describe says the kind (8), the sum order (in file) and the number of trees of the file
$ ml describe --model-file cancer-boost.ml
{"model": "boost", "task": "classification", "kind": 8, "ensemble": {"trees": 100, "nodes": 1134, "rounds_requested": 100}, "classes": [0, 1], "output": {"columns": ["label"], "with_proba": ["label", "probability"], "with_distance": null, "label_threshold": "0.5"}}
```
<!-- /gen:bflow -->

## Worked examples

Four classic datasets, as bundled with scikit-learn (breast cancer Wisconsin, diabetes progression, iris restricted to two species, wine), solved by the real `ml` binary and compared with scikit-learn on the **same two files** each. [docs/examples.md](docs/examples.md) fixes what is compared and the tolerances, and was written before any example was run; the commands and their output are on [the examples page](https://alpibrusl.github.io/cancho-ml/examples.html), with two failures shown (a penalty-free logistic fit on separable data; a column whose type differs between two files). The `ml` output is regenerated by `scripts/check_docs.sh`; the scikit-learn column is read from `docs/examples/reference.json`, written by `python3 scripts/examples_data.py` (needs scikit-learn), and **is not computed by the check**. Nothing here is a claim about speed.

One complete flow, on the committed files `docs/examples/data/iris_train.csv` and `iris_test.csv` (copy them next to where you run it):

<!-- gen:ex-flow -->
```console
# Logistic regression, no penalty, on the raw features: these two species overlap, so the optimum is finite
$ ml fit --model logistic --target virginica --iterations 25 --l2 0 --output iris_fit.ml iris_train.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","rows":75,"columns":5,"target":"virginica","path":"iris_fit.ml","bytes":704,"sha256":"e855b04f8717184c26345ebbfd395c47bb981ce3c74b188afcef699e71484195","written":true,"summary":{"plan":"model kind=logistic iterations=25 l2=0.0\n","steps":0,"features":4,"coefficients":[{"name":"sepal_length_cm","value":"-2.346667939035462"},{"name":"sepal_width_cm","value":"-6.929972260305374"},{"name":"petal_length_cm","value":"9.539045048622665"},{"name":"petal_width_cm","value":"17.159779108027216"}],"intercept":"-41.54877417104498","classes":[0,1],"positive":1}},"meta":{"version":"0.1.0"}}
# Score it on the training rows first: not every one is right
$ ml eval --model-file iris_fit.ml iris_train.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","rows":75,"on":"train","positive":1,"metrics":{"accuracy":"0.9733333333333334","precision":"0.9736842105263158","recall":"0.9736842105263158","f1":"0.9736842105263158","roc_auc":"0.9950213371266002","log_loss":"0.07256993734328859"},"unknown_categories":0,"out_of_range":{"cells":0,"rows":0,"columns":[]}},"meta":{"version":"0.1.0"}}
# Score the held-out rows
$ ml eval --model-file iris_fit.ml iris_test.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","rows":25,"on":"other","positive":1,"metrics":{"accuracy":"1.0","precision":"1.0","recall":"1.0","f1":"1.0","roc_auc":"1.0","log_loss":"0.024399970263418697"},"unknown_categories":0,"out_of_range":{"cells":3,"rows":2,"columns":[{"name":"sepal_length_cm","count":1},{"name":"sepal_width_cm","count":2}]}},"meta":{"version":"0.1.0"}}
# Probabilities for the first five test rows
$ ml predict --model-file iris_fit.ml --proba iris_test.csv --limit 5
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","columns":["label","probability"],"rows":[["1","0.9999998497708621"],["1","0.9999999991884112"],["0","0.000012445257356600849"],["0","0.000053161429567075556"],["0","0.00004866552586998901"]],"row_count":5,"total_rows":25,"truncated":true,"next":{"from":5},"unknown_categories":0,"out_of_range":{"cells":3,"rows":2,"columns":[{"name":"sepal_length_cm","count":1},{"name":"sepal_width_cm","count":2}]},"ignored_columns":["virginica"]},"meta":{"version":"0.1.0"}}
```
<!-- /gen:ex-flow -->

The headline numbers of every example, `ml` against scikit-learn on the same test file. The row "largest coefficient difference" is the largest absolute difference over the coefficients and the intercept printed by `ml fit`. A difference outside the tolerance is reported on the page, with its cause.

<!-- gen:ex-summary -->
| example, run | metric | ml | scikit-learn 1.8.0 | absolute difference | tolerance of docs/examples.md |
|---|---|---:|---:|---:|---|
| Breast cancer, fit | accuracy | 0.951048951048951 | 0.951048951048951 | 0 | within |
| Breast cancer, fit | roc_auc | 0.9939203354297694 | 0.9939203354297694 | 0 | within |
| Breast cancer, fit | log_loss | 0.09862906835282033 | 0.09862905898568457 | 9.4e-09 | within |
| Breast cancer, fit | largest coefficient difference |  |  | 4.2e-07 | within |
| Diabetes, ols | rmse | 56.39290423586858 | 56.39290423586856 | 1.4e-14 | within |
| Diabetes, ols | r2 | 0.35940880381777107 | 0.3594088038177712 | 1.1e-16 | within |
| Diabetes, ols | largest coefficient difference |  |  | 5.2e-12 | within |
| Diabetes, ridge | rmse | 56.27420487723617 | 56.27420487723619 | 1.4e-14 | within |
| Diabetes, ridge | r2 | 0.3621026796026515 | 0.3621026796026512 | 3.3e-16 | within |
| Diabetes, ridge | largest coefficient difference |  |  | 3.8e-12 | within |
| Iris, fit | accuracy | 1.0 | 1.0 | 0 | within |
| Iris, fit | roc_auc | 1.0 | 1.0 | 0 | within |
| Iris, fit | log_loss | 0.024399970263418697 | 0.024399937949542223 | 3.2e-08 | within |
| Iris, fit | largest coefficient difference |  |  | 1.1e-04 | **differs** |
| Wine, fit | accuracy | 1.0 | 1.0 | 0 | within |
| Wine, fit | roc_auc | 1.0 | 1.0 | 0 | within |
| Wine, fit | log_loss | 0.05163738931014777 | 0.05163738956541201 | 2.6e-10 | within |
| Wine, fit | largest coefficient difference |  |  | 4.7e-08 | within |

* **Iris, fit.** Outside the tolerance of docs/examples.md section 3: largest coefficient or intercept difference (5 values). The cause is the reference solver and it is measured here: scikit-learn's own two solvers differ from each other by up to 2.3e-05 in a coefficient (L-BFGS stops short of the optimum), while ml differs from scikit-learn's Newton solver by at most 6.2e-10, inside the same tolerance. The tolerance was not moved.
<!-- /gen:ex-summary -->

## Why cancho-ml?

Most ML tools optimise for how many models they offer. `ml` is built to be predictable first, and offers seven models and a pipeline.

* **Explicit types.** A column is an integer, a decimal (an exact mantissa and a scale) or a float (the nearest double, ties to even, read by cancho-table's exact reader). `nan`, `inf`, an empty cell and a cell that is not a number are refused, with the line and the column; `--drop` ignores a column. [docs/reader.md](docs/reader.md), [docs/cli.md](docs/cli.md)
* **Exact where a sum decides.** The mean of an integer or decimal target and the sum of squared errors are exact integers, kept in limbs, not floats; a float target is summed exactly and rounded once. A test compares every digit with Python's `fractions.Fraction`. The one step that is not exact, the `rmse` (three IEEE operations after the exact sum), is said so in the document. [docs/cli.md](docs/cli.md) section 3
* **The same bytes.** A model file holds no timestamp, host or path. The same data, plan and seed give the same model bytes whatever the working directory, the spelling of the path or the repetition, and at every `--threads` from 1 to 16 (tested on one Mac, for linear, logistic, k-means, tree, forest and boost); across machines it holds for 28 fits on macOS arm64 and Linux x86-64 and is **not tested** on any other platform (see the limits below). [docs/cli.md](docs/cli.md) section 6, [docs/linalg.md](docs/linalg.md) section 5.4
* **Bounded resource use.** Rows, columns, cells, line length, model file size, log size and the categorical dictionary each have a flag, a default and a ceiling, checked before anything sized by the input is allocated. Peak memory is measured for one case, the reader's refusal of a 200 MB file (1.5 MB, `docs/reader.md` section 6); no gate measures the memory of a fit. `ml introspect` lists the limits. [docs/cli.md](docs/cli.md) section 2.2
* **No implicit network access.** The authority row, derived by `cancho authority`, lists what the program can reach: no `net_out`, `net_in`, `ffi` or `clock`, no standard input, and no write by path. `scripts/manifest.py --check` fails if the binary differs from the committed [`manifests/ml.authority.json`](manifests/ml.authority.json). [docs/cli.md](docs/cli.md) section 6
* **Machine-readable errors.** One JSON line against a schema (`ml.v1`); a refusal is `{code, rule, message, hint, repair, detail}` with a stable `<area>.<name>` rule tag. [docs/design.md](docs/design.md) section 8
* **Repair hints.** A refusal says how to fix the request, as an argv to run, or why there is none. [docs/cli.md](docs/cli.md)
* **A model file that checks itself.** `CML1`: sections, tensor records and a SHA-256 trailer. Every truncation and every flipped byte of a small model is refused, and no damaged file is accepted. [docs/cli.md](docs/cli.md) section 4
* **Capability introspection.** `ml introspect` prints the flags, limits, rules and authority; `ml skill` prints a guide for an agent. Both come from the tables the parser runs on.

## Two-minute quick start

You need `git`, Rust, `clang` and `python3`, and the machine needs network access for the clones and for `cancho install`, which fetches the pinned modules into `build/deps`: the `cancho-tools` contract package and cancho-table's five numeric modules (`pow5`, `dec`, `flt`, `facc`, `reader`, the stores that [cancho-table#44](https://github.com/alpibrusl/cancho-table/pull/44) committed), all `[dependencies.*]` of `cancho.toml`, each at one commit. The cancho-table rev is `202b214372a0`; nothing of cancho-table is cloned or copied, and `scripts/check_table_pin.sh` checks that the pin agrees with [docs/reader.md](docs/reader.md) section 1.3, which also says how to move it.

```sh
git clone https://github.com/alpibrusl/cancho                           # the compiler
git clone https://github.com/alpibrusl/cancho-ml && cd cancho-ml
REV=$(sed -n 's/^cancho *= *"\([0-9a-f]*\)".*/\1/p' cancho.toml)        # the compiler these sources need
(cd ../cancho && git fetch -q origin && git checkout "$REV" && cargo build --release -p cancho)
sh scripts/check_table_pin.sh                                           # the pin agrees with docs/reader.md
export PATH=$PWD/../cancho/target/release:$PWD/build:$PATH
cancho install && cancho build --bin ml                                 # install fetches the pins (network); builds build/ml

cat > houses.csv <<'EOF'
id,rooms,area,price
1,2,48.5,180000
2,3,71.25,240000
3,3,64,215000
4,4,98.5,335000
5,1,31,120000
EOF
printf 'id,rooms,area\n6,2,52.5\n7,3,80\n' > new.csv
printf 'id,rooms,area,city\n6,2,52.5,Lyon\n7,3,80,Porto\n' > listings.csv
```

On a 16-core Mac at a load average of 3 to 5, with the compiler built, `cancho install` took 1.4 seconds (once, from GitHub) and `cancho build --bin ml` 5.0 seconds (once). The compiler build and the clones are extra.

Fit, predict, score, and a refusal:

<!-- gen:qs -->
```console
$ ml fit --target price --output houses.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":5,"columns":4,"target":"price","path":"houses.ml","bytes":416,"sha256":"df669ebf009e182d179668967f9c248bc3bfed3383141089bb8759e5c569c62a","written":true,"summary":{"mean":"218000.000000"}},"meta":{"version":"0.1.0"}}
$ ml predict --model-file houses.ml new.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","columns":["prediction"],"rows":[["218000.000000"],["218000.000000"]],"row_count":2,"total_rows":2,"truncated":false,"next":null},"meta":{"version":"0.1.0"}}
$ ml eval --model-file houses.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":5,"on":"train","metrics":{"rmse":"71035.202541","sse":"25230000000.000000000000","prediction":"218000.000000"}},"meta":{"version":"0.1.0"}}
$ ml predict --model-file houses.ml listings.csv
{"rule": "ml.input.non-numeric", "hint": "drop the column with --drop, or fix the cell", "repair": {"kind": "retry", "argv": ["ml", "predict", "--model-file", "houses.ml", "listings.csv", "--drop", "city"]}}
# exit status 8; the rule, hint, repair of the JSON line it prints
```
<!-- /gen:qs -->

## What you can do

Every example runs on the small files above and the ones shown with it, in this order, and the output is what the program prints. The first examples use the constant model, whose prediction is the same for every row; the next two go through the pipeline, and the last has text columns.

### Fit, predict and score a regression

`fit` reads the CSV, takes `--target` as the column to predict and writes the model with `--output`. `price` is a whole-number column, so its mean is an exact decimal with six more fractional digits than the column has. `predict` needs the same columns as the fit, minus the target; `eval` scores a model on a CSV that has the target.

<!-- gen:t-fit -->
```console
$ ml fit --target price --output houses.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":5,"columns":4,"target":"price","path":"houses.ml","bytes":416,"sha256":"df669ebf009e182d179668967f9c248bc3bfed3383141089bb8759e5c569c62a","written":true,"summary":{"mean":"218000.000000"}},"meta":{"version":"0.1.0"}}
$ ml predict --model-file houses.ml new.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","columns":["prediction"],"rows":[["218000.000000"],["218000.000000"]],"row_count":2,"total_rows":2,"truncated":false,"next":null},"meta":{"version":"0.1.0"}}
$ ml eval --model-file houses.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":5,"on":"train","metrics":{"rmse":"71035.202541","sse":"25230000000.000000000000","prediction":"218000.000000"}},"meta":{"version":"0.1.0"}}
```
<!-- /gen:t-fit -->

### Fit and score a classification

`--task classification` needs an integer target with exactly two values. The prediction is the class with more rows (the smaller label on a tie) and the probability is the share of the larger label's rows, to six digits. `predict` comes in pages: `--limit` and `--from`, and `next` is where to continue. This second example uses `shop.csv`.

```csv
id,visits,spend,bought
1,3,12.50,1
2,1,0.00,0
3,7,40.25,1
4,2,5.00,0
5,5,22.75,1
6,4,18.00,1
```

<!-- gen:t-class -->
```console
$ ml fit --task classification --target bought --output shop.ml shop.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"classification","rows":6,"columns":4,"target":"bought","path":"shop.ml","bytes":439,"sha256":"b58cc59fc9716b5b05d8b2ab36c5801748cd112041f007a1505125f18dfed2c7","written":true,"summary":{"classes":[0,1],"counts":[2,4],"majority":1,"probability":"0.666667"}},"meta":{"version":"0.1.0"}}
$ ml predict --model-file shop.ml shop.csv --limit 2 --from 1
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"classification","columns":["label","probability"],"rows":[["1","0.666667"],["1","0.666667"]],"row_count":2,"total_rows":6,"truncated":true,"next":{"from":3}},"meta":{"version":"0.1.0"}}
$ ml eval --model-file shop.ml shop.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"classification","rows":6,"on":"train","metrics":{"accuracy":"0.666667","correct":4,"total":6,"prediction":1}},"meta":{"version":"0.1.0"}}
```
<!-- /gen:t-class -->

### Read floats exactly

A cell with an exponent, a leading or trailing point or more than 12 digits is a float without being named (`--float NAMES` forces it). It is read to the nearest double, ties to even, by cancho-table's reader. The mean of a float target is the exact sum of the doubles over the count, rounded once, and written as the shortest decimal that reads back to the same double. This example uses `sensor.csv`.

```csv
id,x,reading
1,1.5,1e1
2,2.5,2.75
3,0.1,3e-1
4,4,5.5
```

<!-- gen:t-float -->
```console
$ ml fit --target reading --output sensor.ml sensor.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":4,"columns":3,"target":"reading","path":"sensor.ml","bytes":382,"sha256":"c6dd144995f2780cf54dab2532be15a41bf4687b29e7c5904c3ee1e983be3ffc","written":true,"summary":{"mean":"4.6375"}},"meta":{"version":"0.1.0"}}
$ ml eval --model-file sensor.ml sensor.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":4,"on":"train","metrics":{"rmse":"3.6012801543340114","sse":"51.876875","prediction":"4.6375"}},"meta":{"version":"0.1.0"}}
```
<!-- /gen:t-float -->

### Check before you write

An existing output is refused unless `--overwrite` is given. `--dry-run` reads, checks and fits, runs every check the write would, and answers with the file it would write: exit 9, and nothing created or changed. The file is written to a temporary name beside the final one, synced and renamed over it; that this keeps a reader from seeing half a model, and two writers racing on one output, were not tested.

<!-- gen:t-dry -->
```console
$ ml fit --target price --output houses.ml houses.csv
{"rule": "conflict.exists", "hint": "use --overwrite to replace it, or name another file", "detail": {"path": "houses.ml"}}
# exit status 5; the rule, hint, detail of the JSON line it prints
$ ml fit --dry-run --overwrite --target price --output houses.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","rows":5,"columns":4,"target":"price","path":"houses.ml","bytes":416,"sha256":"df669ebf009e182d179668967f9c248bc3bfed3383141089bb8759e5c569c62a","written":false,"summary":{"mean":"218000.000000"}},"dry_run":true,"planned_actions":[{"op":"replace","path":"houses.ml","bytes":416,"sha256":"df669ebf009e182d179668967f9c248bc3bfed3383141089bb8759e5c569c62a"}],"meta":{"version":"0.1.0"}}
# exit status 9; a dry run: nothing was written
```
<!-- /gen:t-dry -->

### A damaged model is refused

The model file ends with the SHA-256 of every byte before it. `damaged.ml` is `houses.ml` with one byte altered.

<!-- gen:t-damaged -->
```console
$ ml eval --model-file damaged.ml houses.csv
{"rule": "ml.model.checksum", "hint": "the file changed after it was written"}
# exit status 8; the rule, hint of the JSON line it prints
```
<!-- /gen:t-damaged -->

### It says what it does not do

The `--model` list is the list of what is built. There are no stand-in models, and a test pins that. A prediction that is not a finite number (a feature near the largest double times a coefficient above 1) is a refusal too, `ml.predict.non-finite`, with the row and the column in `detail` and no cell written; `predict`, `ml-predict`, `eval` and `ml-serve` raise it from one function, over every row scored and not only the page ([docs/nonfinite.md](docs/nonfinite.md)).

<!-- gen:t-honest -->
```console
$ ml fit --model knn --target price --output knn.ml houses.csv
{"rule": "args.bad-value", "detail": {"flag": "--model", "value": "knn", "expected": "constant/linear/logistic/kmeans/tree/forest/boost"}}
# exit status 2; the rule, detail of the JSON line it prints
$ ml fit --target prize --output typo.ml houses.csv
{"rule": "ml.input.unknown-column", "hint": "names are compared exactly, byte for byte", "detail": {"flag": "--target", "name": "prize"}}
# exit status 3; the rule, hint, detail of the JSON line it prints
$ ml predict --model-file houses-big.ml --drop id huge.csv
{"rule": "ml.predict.non-finite", "hint": "scale the inputs, or check or drop the row.", "detail": {"row": 1, "column": "prediction"}}
# exit status 8; the rule, hint, detail of the JSON line it prints
```
<!-- /gen:t-honest -->

### A fit that has no answer is refused by name

Two designs have no unique answer: a column that the columns before it explain (the one-hot block of a text column beside the intercept, an equal column, a constant column), and classes that a plane separates (logistic regression with `--l2 0`). `ml` refuses them with `ml.fit.collinear` and `ml.fit.separable`, not with huge coefficients or a late `ml.linalg.not-positive-definite`. The first names the column, the dependency (`(intercept)` = 1 x `species#0` + ...), the relative pivot and the tolerance `2^-40`; the second the pass, the summed loss against its bound `0.6931` (ln 2: every row has a positive margin) and the norm of the coefficients. Both carry the repair as an argv, a ridge sized from the data; with `--plan` the repair is `none`, because `l2` is a key of the plan's model line. A positive `--l2` is never refused as separable. The decisions, the tolerance and the measurements: [docs/refusals-fit.md](docs/refusals-fit.md).

<!-- gen:refuse -->
```console
# The text column species is one-hot encoded: its columns add up to the intercept, so the normal equations of the first Newton pass are singular. The refusal names the column and the dependency
$ ml fit --model logistic --cat species --step 'encode column=species kind=onehot' --drop id,city --target adopted --output pets-col.ml pets.csv
{"rule": "ml.fit.collinear", "hint": "set l2 above 2^-30 times the largest diagonal element, or drop one of the columns named in detail.", "repair": {"kind": "retry", "argv": ["ml", "fit", "--model", "logistic", "--cat", "species", "--step", "encode column=species kind=onehot", "--drop", "id,city", "--target", "adopted", "--output", "pets-col.ml", "pets.csv", "--l2", "1e-8"]}, "detail": {"model": "logistic", "pass": 0, "pivot": 8, "column": "(intercept)", "relative_pivot": "3.8163916471489756e-17", "tolerance": "9.094947017729282e-13", "l2": "0.0", "largest_diagonal": "7.417763157894737", "involved": 6, "columns": ["species#0", "species#1", "species#2", "species#3", "species#4", "species#5"], "coefficients": ["1.000000", "1.000000", "1.000000", "1.000000", "1.000000", "1.000000"]}}
# exit status 8; the rule, hint, repair, detail of the JSON line it prints
# The repair the tool offered is a ridge just above the tolerance. Run it
$ ml fit --model logistic --cat species --step 'encode column=species kind=onehot' --drop id,city --target adopted --output pets-col.ml pets.csv --l2 1e-8
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","rows":76,"columns":4,"target":"adopted","path":"pets-col.ml","bytes":933,"sha256":"853d842b964e39b4a351d4aefec3e946c003eb6174f8038137f7c73239ffa73d","written":true,"summary":{"plan":"encode column=species kind=onehot max-categories=64 unknown=ignore\nmodel kind=logistic iterations=100 l2=1e-8\n","steps":1,"steps_sha256":"c9cffeb5a63745b49f568539ef8e9939a07ec8308e02e6959643c789ff88e2c0","categorical":[{"step":1,"kind":"onehot","column":"species","categories":6,"output_columns":6}],"string_table":{"entries":1,"strings":6,"pool_bytes":29,"bytes":236,"sha256":"fb5a45737bc8f57d329c8bd330b0f24dd6e7ce2b3c1737eb585dd88a5afa6946"},"features":8,"coefficients":[{"name":"age","value":"-0.5941654182959529"},{"name":"visits","value":"0.4910489127624334"},{"name":"species#0","value":"-2.003961367096021"},{"name":"species#1","value":"-0.026512480558186693"},{"name":"species#2","value":"0.6355451771815215"},{"name":"species#3","value":"0.08426814350645216"},{"name":"species#4","value":"2.036084076946871"},{"name":"species#5","value":"-0.7254235494784796"}],"intercept":"1.3834201860087134","classes":[0,1],"positive":1}},"meta":{"version":"0.1.0"}}
# The two classes of shop.csv are separated by a plane, so with no penalty the best fit is at infinity. It is refused at the pass where every row is right
$ ml fit --model logistic --drop id --target bought --output shop-sep.ml shop.csv
{"rule": "ml.fit.separable", "hint": "set l2 above 0, for example the value in the repair.", "repair": {"kind": "retry", "argv": ["ml", "fit", "--model", "logistic", "--drop", "id", "--target", "bought", "--output", "shop-sep.ml", "shop.csv", "--l2", "1"]}, "detail": {"reason": "loss", "pass": 3, "loss_sum": "0.39914766019334014", "loss_bound": "0.6931", "coefficient_norm": "3.8404580607230154", "predictor_scale": "6.3091994880700435", "scale_bound": "50.0", "l2": "0.0", "iterations": 100}}
# exit status 8; the rule, hint, repair, detail of the JSON line it prints
# The repair is a penalty. Run it
$ ml fit --model logistic --drop id --target bought --output shop-sep.ml shop.csv --l2 1
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","rows":6,"columns":3,"target":"bought","path":"shop-sep.ml","bytes":535,"sha256":"ba16320f84807cc18b7cc32d3539d629e32aae3f3b413a0c959439835e565e64","written":true,"summary":{"plan":"model kind=logistic iterations=100 l2=1.0\n","steps":0,"features":2,"coefficients":[{"name":"visits","value":"0.05247869247963008"},{"name":"spend","value":"0.36360045523940554"}],"intercept":"-3.314175317672299","classes":[0,1],"positive":1}},"meta":{"version":"0.1.0"}}
```
<!-- /gen:refuse -->

### Hold rows out, with a seed

`--split train=0.8 --seed 3` chooses the rows by the seeded train/test split of the metrics library (the stream registry, kind 4): the test side is the rows the fit does not see, `ceil(n * 0.2)` of them. The fit reports the held-out rows' metrics, scored through the stored steps; `eval --split train=0.8 --seed 3` on the same file scores the same rows. With 5 rows the test side is 1 row, so this is only to show the shape of the answer.

<!-- gen:t-split -->
```console
$ ml fit --model linear --l2 0.1 --split train=0.8 --seed 3 --target price --output houses-lin.ml houses.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"linear","task":"regression","rows":4,"columns":4,"target":"price","path":"houses-lin.ml","bytes":545,"sha256":"cd95890af137326e63fcfab54087d154dc499e7b1553b5108a2bc4585f725f32","written":true,"summary":{"plan":"model kind=linear l2=0.1\n","steps":0,"features":3,"coefficients":[{"name":"id","value":"-1905.5649303915627"},{"name":"rooms","value":"-15667.071024324228"},{"name":"area","value":"3843.571373934476"}],"intercept":"25324.69546154733","split":{"train":4,"test":1,"seed":3,"spec":"holdout train=0.8 seed=3"},"holdout":{"rows":1,"on":"held-out","metrics":{"rmse":"8366.812920622935","mae":"8366.812920622935","r2":null}}}},"meta":{"version":"0.1.0"}}
```
<!-- /gen:t-split -->

### Splits and cross-validation

`--split train=F` is a seeded random holdout; three flags change which rows are held out, and a verb scores a model by k-fold cross-validation. The reasons, the algorithms and the gates are in [docs/splits.md](docs/splits.md).

| flag | what it does |
|---|---|
| `--stratify` | keeps the class proportions: per class the held-out rows (or the rows of a fold) are the floor or the ceiling of its exact share. A classification target only |
| `--split-group COLUMN` | all the rows of a group are on one side (or in one fold). The column may be text or an integer; groups are compared byte for byte |
| `--split-by time:COLUMN` | orders the rows by COLUMN, a number or an ISO-8601 date or datetime: the oldest rows train and the newest test; with `cv` the folds are an expanding window. Ties keep the order of the file |
| `ml cv --folds K --repeats R` | `K` from 2 to 20 folds, `R` from 1 to 10 repeats; scores the model on every held-out fold and writes no file |
| `--emit-folds` | puts the fold of every row (file order) in the answer, so a run can be replayed |

At most one of the three strategies per command: they are different algorithms, and `ml.split.combination` refuses two. A text column that a rule reads still needs `--drop` when it is not a feature. Every fitted step (impute, scale, bucket, encode, freq, hash and the categorical vocabulary) is fitted on the training rows of each fold alone, by the function `fit` uses, and the held-out rows go through the stored steps, as `eval` does: a test in the repository cuts the fold files itself, runs `ml fit` and `ml eval` on them and compares the metrics of every fold with those of `ml cv`, string for string. A mutant that fits the steps on all rows fails that test. `summary.split.spec` and `split` carry one line that names any split (`holdout train=0.8 stratify seed=1`, `cv folds=5 repeats=1 group=customer seed=3`).

The standard deviation is the sample one (divisor `K * R - 1`). The assignment is the library's seeded stream, not scikit-learn's, so the folds differ from `KFold(shuffle=True)`; what is compared with scikit-learn is the invariants and the metrics on the same explicit folds. `visits.csv` is generated data (`docs/examples/data/gen_visits.py`): 12 customers, 10 visits each, so the metrics show how the commands work and nothing about customers.

<!-- gen:splitflow -->
```console
# A holdout that keeps the class proportions: 426 rows, 20 percent held out, per class the floor or the ceiling of its share
$ ml fit --model tree --task classification --max-depth 4 --target benign --split train=0.8 --stratify --seed 7 --output bc.ml breast_cancer_train.csv
{"split":{"train":340,"test":86,"seed":7,"spec":"holdout train=0.8 stratify seed=7"},"holdout":{"rows":86,"on":"held-out","metrics":{"accuracy":"0.9418604651162791","precision":"0.9454545454545454","recall":"0.9629629629629629","f1":"0.9541284403669725","roc_auc":"0.9461805555555556","log_loss":"1.3089414342386103"}}}
# visits.csv: 12 customers with 10 visits each, an ISO date, a basket, the item count and whether the customer came back. The newest quarter of the visits is held out (the date column is dropped as a feature, not as a rule)
$ ml fit --model tree --task classification --max-depth 3 --target returning --drop customer,visit_date --split train=0.75 --split-by time:visit_date --output visits-time.ml visits.csv
{"split":{"train":90,"test":30,"seed":0,"spec":"holdout train=0.75 time=visit_date seed=0"},"holdout":{"rows":30,"on":"held-out","metrics":{"accuracy":"0.4","precision":"0.0","recall":"0.0","f1":"0.0","roc_auc":"0.5429864253393665","log_loss":"2.255239329249321"}}}
# No customer on both sides: whole customers are held out
$ ml fit --model tree --task classification --max-depth 3 --target returning --drop customer,visit_date --split train=0.75 --split-group customer --seed 2 --output visits-group.ml visits.csv
{"split":{"train":90,"test":30,"seed":2,"spec":"holdout train=0.75 group=customer seed=2"},"holdout":{"rows":30,"on":"held-out","metrics":{"accuracy":"0.43333333333333335","precision":"0.3333333333333333","recall":"0.7","f1":"0.45161290322580644","roc_auc":"0.5425","log_loss":"3.080977491776084"}}}
# Two rules for one split are refused, not blended
$ ml fit --model tree --task classification --target returning --drop customer,visit_date --split train=0.75 --stratify --split-by time:visit_date --output x.ml visits.csv
{"rule": "ml.split.combination", "hint": "keep one of them.", "repair": {"kind": "none", "reason": "keep one of the two: they are different algorithms, and dropping a group or a time rule can leak rows between the sides"}, "detail": {"flag": "--stratify", "other": "--split-by"}}
# exit status 2; the rule, hint, repair, detail of the JSON line it prints
# Five-fold cross-validation, stratified: every step is fitted on the four training folds of each fold alone. Per-fold metrics, then the mean and the sample standard deviation
$ ml --format text cv --model logistic --task classification --step 'scale column=* kind=standard' --l2 0.1 --target virginica --folds 5 --seed 1 --stratify iris_train.csv
cv: logistic classification, 75 rows, cv folds=5 repeats=1 stratify seed=1
repeat 0 fold 0: train 60 test 15 accuracy=0.9333333333333333 precision=0.875 recall=1.0 f1=0.9333333333333333 roc_auc=0.9821428571428571 log_loss=0.2558531837774219
repeat 0 fold 1: train 60 test 15 accuracy=0.9333333333333333 precision=1.0 recall=0.8571428571428571 f1=0.9230769230769231 roc_auc=1.0 log_loss=0.27371716218030023
repeat 0 fold 2: train 60 test 15 accuracy=1.0 precision=1.0 recall=1.0 f1=1.0 roc_auc=1.0 log_loss=0.20536677751917837
repeat 0 fold 3: train 60 test 15 accuracy=0.8666666666666667 precision=0.8 recall=1.0 f1=0.8888888888888888 roc_auc=0.9821428571428571 log_loss=0.3373620786165024
repeat 0 fold 4: train 60 test 15 accuracy=0.9333333333333333 precision=1.0 recall=0.875 f1=0.9333333333333333 roc_auc=1.0 log_loss=0.29570178889623133
accuracy mean 0.9333333333333332 std 0.04714045207910316
precision mean 0.9349999999999999 std 0.09287087810503354
recall mean 0.9464285714285715 std 0.07362688617174395
f1 mean 0.9357264957264956 std 0.04029984772479705
roc_auc mean 0.9928571428571429 std 0.009780759955449421
log_loss mean 0.27360019819792686 std 0.04879761163429506
$ ml --format text cv --model linear --step 'scale column=* kind=standard' --target progression --folds 5 --seed 1 diabetes_train.csv
cv: linear regression, 331 rows, cv folds=5 repeats=1 seed=1
repeat 0 fold 0: train 264 test 67 rmse=51.26915634092274 mae=40.079441504420245 r2=0.5444848475040903
repeat 0 fold 1: train 265 test 66 rmse=56.47637996728807 mae=46.982995159007956 r2=0.4944111209775581
repeat 0 fold 2: train 265 test 66 rmse=55.34388490502349 mae=46.384195252405554 r2=0.5269757178495083
repeat 0 fold 3: train 265 test 66 rmse=55.00427633532207 mae=43.8457621841562 r2=0.5251682642236672
repeat 0 fold 4: train 265 test 66 rmse=50.79903754016721 mae=41.62632028610556 r2=0.5706964245749535
rmse mean 53.77854701774472 std 2.569314888294932
mae mean 43.7837428772191 std 2.9739592721548647
r2 mean 0.5323472750259555 std 0.028006791780639296
# An expanding window: fold i trains on everything before block i + 1 and tests on that block
$ ml cv --model tree --task classification --max-depth 3 --target returning --drop customer,visit_date --split-by time:visit_date --folds 4 visits.csv
{"split":"cv folds=4 repeats=1 time=visit_date seed=0","strategy":"time","summary":{"count":4,"std":"sample","metrics":{"accuracy":{"mean":"0.625","std":"0.21783871029664056","n":4},"precision":{"mean":null,"std":null,"n":3},"recall":{"mean":"0.47619047619047616","std":"0.3868589716493314","n":4},"f1":{"mean":"0.44940476190476186","std":"0.33872830848485314","n":4},"roc_auc":{"mean":"0.6960930205415499","std":"0.18341201780759583","n":4},"log_loss":{"mean":"3.206846846909722","std":"1.3505716186412535","n":4}}}}
# Whole customers per fold, and the assignment of every row (file order) as data, so that a run can be replayed
$ ml cv --model tree --task classification --max-depth 3 --target returning --drop customer,visit_date --split-group customer --folds 4 --seed 5 --emit-folds visits.csv
{"split":"cv folds=4 repeats=1 group=customer seed=5","strategy":"group","assignment":{"repeats":1,"rows":120,"folds":[[1,1,1,0,3,1,3,2,3,0,0,1,0,2,3,2,3,3,0,0,0,3,1,2,2,0,2,1,0,2,2,3,3,3,0,0,3,0,0,2,1,3,1,1,2,3,2,3,2,1,2,2,0,0,2,0,3,3,3,2,2,3,1,0,0,1,0,1,2,2,1,1,3,0,1,2,1,0,3,2,1,1,0,3,1,1,0,1,0,3,0,0,3,1,3,3,1,2,1,3,3,2,2,1,2,3,2,2,2,0,0,3,1,1,3,0,0,2,2,1]]}}
# A regression has no classes to keep in proportion. The refusal offers the corrected command
$ ml cv --model linear --step 'scale column=* kind=standard' --target progression --stratify diabetes_train.csv
{"rule": "ml.split.stratify-target", "hint": "drop --stratify, or use --task classification with a model that has a class target.", "repair": {"kind": "retry", "argv": ["ml", "cv", "--model", "linear", "--step", "scale column=* kind=standard", "--target", "progression", "diabetes_train.csv"]}}
# exit status 2; the rule, hint, repair of the JSON line it prints
```
<!-- /gen:splitflow -->

### Text columns: encode, fit, then predict and eval from the file

`--cat NAMES` declares columns of text. A step has to read each declared column: `encode column=NAME kind=onehot|label` (a block of columns, or one label), `freq column=NAME` (the share of the training rows that hold the category, one rounding) or `hash column=NAME buckets=B` (the bucket is the first 8 bytes of SHA-256, big-endian, modulo B). `min-count=N` puts the categories seen fewer than N times in the training rows into one bucket, `__other__`; `unknown=refuse` refuses a string the training rows never held (the default gives an all-zero block, `0.0` for `freq`, `-1.0` for a label); `missing=zero` turns an empty cell into zeros, where the default leaves the encoded cells missing and the model refuses them (`ml.input.missing-value`). The vocabulary is built from the training rows only, sorted bytewise, so the table does not depend on the row order or on `--threads`. The model file stores the table, the `--na-values` markers and the reader limits of the fit, so `predict` and `eval` take no `--cat` flag: the file says what to read. This example uses `docs/examples/data/pets.csv` (76 rows: `age`, `species`, `city`, `visits` and the target `adopted`; copy it next to where you run it). It is generated data, so the metrics say how the commands work and nothing about pets.

<!-- gen:catflow -->
```console
# pets.csv: 76 rows; species and city are text. Declared with --cat but read by no step, they are refused
$ ml fit --model logistic --cat species,city --target adopted --output pets.ml pets.csv
{"rule": "ml.input.categorical-not-encoded", "hint": "name the column in an encode, freq or hash step (for example encode column=NAME kind=onehot), or remove it from --cat and drop it with --drop.", "detail": {"columns": "species,city", "categories": "6,3", "vocabulary_sha256": "95cdde428dd56f2e7a696e7ffaf4b9fc65312b5e3c674b34e7e6d065f5b5cd55"}}
# exit status 8; the rule, hint, detail of the JSON line it prints
# Encoded, the fit writes the model file. species is one-hot (the categories seen fewer than 4 times share __other__), city is its frequency; the file holds the training strings
$ ml fit --model logistic --cat species,city --step 'encode column=species kind=onehot min-count=4;freq column=city missing=zero;scale column=age,visits kind=standard' --drop id --iterations 50 --l2 0.1 --target adopted --output pets.ml pets.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","rows":76,"columns":5,"target":"adopted","path":"pets.ml","bytes":1333,"sha256":"d1a63c6466cf54c23ec1a7f8d0093ff045017de123aabece4a33811df66386c6","written":true,"summary":{"plan":"encode column=species kind=onehot max-categories=64 unknown=ignore min-count=4\nfreq column=city max-categories=64 unknown=ignore min-count=1 missing=zero\nscale column=age,visits kind=standard ddof=0\nmodel kind=logistic iterations=50 l2=0.1\n","steps":3,"steps_sha256":"8f910f6f5390602504628d4c90061912cd906bc60a7fd09959b610d8f766094b","categorical":[{"step":1,"kind":"onehot","column":"species","categories":6,"output_columns":6},{"step":2,"kind":"freq","column":"city","categories":3,"output_columns":1}],"string_table":{"entries":2,"strings":9,"pool_bytes":43,"bytes":342,"sha256":"c597f59764a5f40e89018b535f62668e14dd81a901782bf34383fee32c813295"},"features":9,"coefficients":[{"name":"age","value":"-0.7824228926137594"},{"name":"visits","value":"0.4073813210802055"},{"name":"species#0","value":"-0.020165373363988392"},{"name":"species#1","value":"-0.4065920239943293"},{"name":"species#2","value":"-0.06722805346853337"},{"name":"species#3","value":"0.22592934448065002"},{"name":"species#4","value":"-0.003848538940300602"},{"name":"species#5","value":"0.27190464528650155"},{"name":"city","value":"0.10100137512530218"}],"intercept":"0.3296864511790136","classes":[0,1],"positive":1}},"meta":{"version":"0.1.0"}}
# predict reads the model file and nothing else; the text columns are found by name
$ ml predict --model-file pets.ml --drop id --proba --limit 3 pets.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","columns":["label","probability"],"rows":[["1","0.7517148972048996"],["1","0.8107096739731363"],["0","0.45457790834086403"]],"row_count":3,"total_rows":76,"truncated":true,"next":{"from":3},"unknown_categories":0,"out_of_range":{"cells":0,"rows":0,"columns":[]},"ignored_columns":["adopted"]},"meta":{"version":"0.1.0"}}
# New rows: the columns in another order, a city and a species the training rows never held (counted, not guessed), an empty city cell that missing=zero accepts
$ ml predict --model-file pets.ml --proba newpets.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","columns":["label","probability"],"rows":[["1","0.779143555391799"],["1","0.5062375189261414"],["1","0.7357314548827897"]],"row_count":3,"total_rows":3,"truncated":false,"next":null,"unknown_categories":2,"out_of_range":{"cells":0,"rows":0,"columns":[]},"ignored_columns":[]},"meta":{"version":"0.1.0"}}
# eval scores the file it was fitted on
$ ml eval --model-file pets.ml --drop id pets.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","rows":76,"on":"train","positive":1,"metrics":{"accuracy":"0.8289473684210527","precision":"0.8444444444444444","recall":"0.8636363636363636","f1":"0.8539325842696629","roc_auc":"0.8742897727272727","log_loss":"0.4847004611714956"},"unknown_categories":0,"out_of_range":{"cells":0,"rows":0,"columns":[]}},"meta":{"version":"0.1.0"}}
# species hashed to 8 columns and city as a label: the file holds no string of species, and an unseen species lands in the bucket it hashes to
$ ml fit --model logistic --cat species,city --step 'hash column=species buckets=8 missing=zero;encode column=city kind=label missing=zero;scale column=age,visits kind=standard' --drop id --split train=0.75 --seed 3 --iterations 50 --l2 0.1 --target adopted --output pets-hash.ml pets.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","rows":57,"columns":5,"target":"adopted","path":"pets-hash.ml","bytes":1213,"sha256":"78f6713b5f0626b0fcf23af86277cee3b85a345308c9377d8fca12fc9efbd453","written":true,"summary":{"plan":"hash column=species buckets=8 missing=zero\nencode column=city kind=label max-categories=64 unknown=ignore missing=zero\nscale column=age,visits kind=standard ddof=0\nmodel kind=logistic iterations=50 l2=0.1\n","steps":3,"steps_sha256":"2c9c79220429bb0793ac19d974f565566b77a08b03492bda2ccc1f26b3eefa86","categorical":[{"step":1,"kind":"hash","column":"species","categories":0,"output_columns":8},{"step":2,"kind":"label","column":"city","categories":3,"output_columns":1}],"string_table":{"entries":2,"strings":3,"pool_bytes":14,"bytes":241,"sha256":"b7ca8fa50923607db62ab75bf2e7ecd7267edaa4d78704e3af296ef7f1747a9f"},"features":11,"coefficients":[{"name":"age","value":"-0.6875723955849204"},{"name":"visits","value":"0.4151426140617374"},{"name":"species#0","value":"0.25559046581946027"},{"name":"species#1","value":"0.0"},{"name":"species#2","value":"0.0"},{"name":"species#3","value":"-0.07144881420245953"},{"name":"species#4","value":"0.2628888663105544"},{"name":"species#5","value":"0.0"},{"name":"species#6","value":"0.0"},{"name":"species#7","value":"-0.44703051792755505"},{"name":"city","value":"-0.35100088010733876"}],"intercept":"0.5909043953829989","classes":[0,1],"positive":1,"split":{"train":57,"test":19,"seed":3,"spec":"holdout train=0.75 seed=3"},"holdout":{"rows":19,"on":"held-out","metrics":{"accuracy":"0.7894736842105263","precision":"0.8181818181818182","recall":"0.8181818181818182","f1":"0.8181818181818182","roc_auc":"0.9318181818181818","log_loss":"0.4278208227654049"}}}},"meta":{"version":"0.1.0"}}
$ ml predict --model-file pets-hash.ml --proba newpets.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","columns":["label","probability"],"rows":[["1","0.8560974272954618"],["1","0.5502566884050817"],["1","0.8204410175517777"]],"row_count":3,"total_rows":3,"truncated":false,"next":null,"unknown_categories":1,"out_of_range":{"cells":0,"rows":0,"columns":[]},"ignored_columns":[]},"meta":{"version":"0.1.0"}}
```
<!-- /gen:catflow -->

**The strings are in the file.** The string table of the model file ([docs/stringtable.md](docs/stringtable.md)) holds the **raw category strings of the training rows**, with their counts, the column names and the missing-value markers: whoever has the model file has those values. A `hash` step stores none (its entry holds the column name and the number of buckets, and `predict` hashes the cells it reads), and `encode min-count=N` still stores every distinct string, because the counts are the state. A pipeline that must not carry its category values hashes them. A file with no categorical step has no table and is the file of a numeric model. The table is bounded (131,072 strings, 8 MiB of string bytes, 12 MiB in all: a fit over a bound is `ml.limit.string-table`), checksummed, and refused with `ml.model.string-table` when it is damaged. [docs/categorical.md](docs/categorical.md), [docs/categorical-encoders.md](docs/categorical-encoders.md)

### Predict from standard input and with ml-predict

`ml predict` reads the CSV from standard input for the operand `-` or for no operand; a file named `-` is `./-`. The bytes are read once, up to `--max-bytes` (16 MiB by default), and the reader's two passes run over them, so the answer for the bytes on standard input is, byte for byte, the answer for the same bytes in a file (a refusal that names a file says `-` instead). `ml describe --model-file M` prints, as data, what a model file requires of a caller: the model and its task, the plan, the columns by name and type, the vocabularies with counts, the missing-value markers, the reader limits, the classes and the columns of an answer. Its field names are a contract ([docs/serving.md](docs/serving.md) section 5): a server builds its request schema from them. `ml-predict` is a second binary built from the same sources. It has `predict`, `describe`, `introspect` and `skill` and nothing else, and its authority row has no write of any kind and no thread; its answers are the answers of `ml predict`, apart from the program name inside a repair.

<!-- gen:sflow -->
```console
# predict reads the CSV from standard input for the operand - (or no operand): the answer is the file's answer
$ ml predict --model-file houses.ml - < new.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"constant","task":"regression","columns":["prediction"],"rows":[["218000.000000"],["218000.000000"]],"row_count":2,"total_rows":2,"truncated":false,"next":null},"meta":{"version":"0.1.0"}}
# a forgotten operand with nothing on standard input is the refusal of an empty file
$ ml predict --model-file houses.ml < empty.csv
{"rule": "ml.input.empty", "hint": "give a CSV whose first row is the header"}
# exit status 8; the rule, hint of the JSON line it prints
# describe: what the model file requires of a caller. Excerpt: the model, the columns of a request, the classes, the columns of an answer
$ ml describe --model-file pets.ml --limit 2
{"model": "logistic", "task": "classification", "request": {"header": ["age", "species", "city", "visits"], "optional": ["adopted"], "extra_columns": "refused", "missing_cells": "needs_impute"}, "classes": [0, 1], "output": {"columns": ["label"], "with_proba": ["label", "probability"], "with_distance": null, "label_threshold": "0.5"}}
# ml-predict predicts and describes, from the same sources, with a row that has no write and no thread
$ ml-predict predict --model-file pets.ml --proba - < newpets.csv
{"ok":true,"command":"ml","schema":"ml.v1","data":{"model":"logistic","task":"classification","columns":["label","probability"],"rows":[["1","0.779143555391799"],["1","0.5062375189261414"],["1","0.7357314548827897"]],"row_count":3,"total_rows":3,"truncated":false,"next":null,"unknown_categories":2,"out_of_range":{"cells":0,"rows":0,"columns":[]},"ignored_columns":[]},"meta":{"version":"0.1.0"}}
$ ml-predict eval --model-file houses.ml houses.csv
{"rule": "args.unknown-command", "hint": "ml-predict has the verbs predict and describe, and introspect and skill describe it", "detail": {"command": "eval"}}
# exit status 2; the rule, hint, detail of the JSON line it prints
$ ml-predict predict --model-file houses.ml --log run.ndjson new.csv
{"rule": "args.unknown-flag", "hint": "the tool's flags are listed by `introspect`; POSIX spellings are not promised", "detail": {"flag": "--log", "position": 4}}
# exit status 2; the rule, hint, detail of the JSON line it prints
```
<!-- /gen:sflow -->

### Serve a model over HTTP

`ml-serve` serves one model file with [cancho-web](https://github.com/alpibrusl/cancho-web) ([docs/serving-http.md](docs/serving-http.md)). It is a third binary from the same sources, with its own entry point, and it is not a subprocess of `ml predict`: it calls the same code in the same process, with the CSV and the model file as bytes in memory. It reads the model file once and decodes it once, builds the request schema from what `describe` says (one member for each column of the model, the `enum` of the categories the file lists in full, `null` only where the model takes missing cells) and the OpenAPI 3.1 document from the same declaration, and answers one request at a time on one thread.

`POST /predict` takes a record, `POST /predict/batch` takes `{"rows": [...]}`, `GET /model` is the answer of `describe`, `GET /openapi.json` is the document and `GET /health` is liveness. A classification answers `label` and `probability`, a regression `prediction`, a clustering `cluster` and `distance`. Every answer carries the SHA-256 of the model file in its head (`X-Model-Sha256`).

The values in an answer are the text `ml predict` prints, written as JSON numbers. A number that cannot be written exactly as the type of its column (an `int` takes a whole number of at most 12 digits, a `dec` the digits of its scale) is refused with its JSON Pointer and never rounded. A refusal is `application/problem+json` with a `rule` field (`ml.schema.mismatch` lists every error at once; a refusal of the predictor carries its own rule and error object), and the body, the number of records, the head and the idle time are bounded.

`ml-serve` listens on `127.0.0.1` by default, so only the host it runs on can reach it, and needs no token there (`--token-file FILE` is accepted and then enforced: a bearer token, read once, required by every operation but `GET /health`). `--bind all` listens on every address of the host and requires `--token-file`; `--bind all --no-token` is refused. The commands below run `ml-serve` on `pets.ml`, the model of the categorical flow, and talk to it. `scripts/demo.sh` runs a whole session with the built binaries in a temporary directory (two tracked experiments, `ml runs compare` and `ml runs verify`, `ml eval` on the training file and on another, `ml-serve` on a free port with no flag about the address, and two `curl` requests, one of them a refusal); `scripts/check_demo.py` runs it and asserts that the model the server answers with is the model `runs verify` verified, and refuses ten edits of the script.

<!-- gen:hflow -->
```console
# ml-serve reads the model file once and answers over HTTP, one request at a time (the demo runs on a free port; the text shows 8080). With no flag about the address it listens on 127.0.0.1 and needs no token
$ ml-serve --model-file pets.ml --port 8080 &
listening on 127.0.0.1:8080
# liveness, and the SHA-256 of the model file being served
$ curl -s localhost:8080/health
{"ok":true,"model_sha256":"d1a63c6466cf54c23ec1a7f8d0093ff045017de123aabece4a33811df66386c6"}
# one record: a member for every column of the model. The head carries the digest too
$ curl -s -i -XPOST -H 'Content-Type: application/json' -d '{"age":3,"species":"cat","city":"Turin","visits":4}' localhost:8080/predict
HTTP/1.1 200 OK
X-Model-Sha256: d1a63c6466cf54c23ec1a7f8d0093ff045017de123aabece4a33811df66386c6
Content-Type: application/json
Content-Length: 278
Connection: keep-alive

{"model_sha256":"d1a63c6466cf54c23ec1a7f8d0093ff045017de123aabece4a33811df66386c6","model":"logistic","task":"classification","columns":["label","probability"],"rows":[[1,0.6996810090936468]],"row_co…
# a batch: a row of the answer for each record, in order
$ curl -s -XPOST -H 'Content-Type: application/json' -d '{"rows":[{"age":3,"species":"cat","city":"Turin","visits":4},{"age":1,"species":"dog","city":"Lyon","visits":7}]}' localhost:8080/predict/batch
{"model_sha256":"d1a63c6466cf54c23ec1a7f8d0093ff045017de123aabece4a33811df66386c6","model":"logistic","task":"classification","columns":["label","probability"],"rows":[[1,0.6996810090936468],[1,0.915…
# a refusal: every error at once, each with its JSON Pointer, as application/problem+json with the rule
$ curl -s -i -XPOST -H 'Content-Type: application/json' -d '{"age":"two","species":"axolotl","city":"Lyon","visits":4,"zip":1}' localhost:8080/predict
HTTP/1.1 422 Unprocessable Content
X-Model-Sha256: d1a63c6466cf54c23ec1a7f8d0093ff045017de123aabece4a33811df66386c6
Content-Type: application/problem+json
Content-Length: 412
Connection: keep-alive

{"rule":"ml.schema.mismatch","model_sha256":"d1a63c6466cf54c23ec1a7f8d0093ff045017de123aabece4a33811df66386c6","type":"about:blank","title":"Unprocessable Content","status":422,"count":3,"errors":[{"pointer":"/age","code":"type","detail":"has the wrong type"}…
# what the model requires of a caller (excerpt), and the OpenAPI 3.1 document made from the same columns (excerpt)
$ curl -s localhost:8080/model
{"model":"logistic","task":"classification","request":{"header":["age","species","city","visits"],"optional":["adopted"],"extra_columns":"refused","missing_cells":"needs_impute"},"classes":[0,1],"out…
$ curl -s localhost:8080/openapi.json
{"openapi":"3.1.0","paths":["/health","/model","/openapi.json","/predict","/predict/batch"],"Record":{"type":"object","properties":{"age":{"type":"integer","minimum":-999999999999,"maximum":999999999…
# the form for a network: every address of the host, and a token (a request without it is a 401; the token is the only line of the file)
$ ml-serve --model-file pets.ml --bind all --token-file token.txt --port 8081 &
listening on 0.0.0.0:8081
$ curl -s -i localhost:8081/model
HTTP/1.1 401 Unauthorized
X-Model-Sha256: d1a63c6466cf54c23ec1a7f8d0093ff045017de123aabece4a33811df66386c6
Content-Type: application/problem+json
Content-Length: 104
Connection: keep-alive
WWW-Authenticate: Bearer

{"type":"about:blank","title":"Unauthorized","status":401,"detail":"send Authorization: Bearer <token>"}
$ curl -s -H 'Authorization: Bearer demo-token' localhost:8081/model
{"model":"logistic","task":"classification","classes":[0,1]}
```
<!-- /gen:hflow -->

The model is read at start and changed by restarting. There is no TLS, no reload and no request deadline beyond the idle time. [docs/serving-http.md](docs/serving-http.md) has the endpoints, the status of every refusal, the limits and the gates.

### The numeric stores other repositories pin

The random generator, the matrix and its missing-cell bitmap, the exact sums, the linear algebra (Cholesky and the solves among it) and the ranks and splits are published as 23 pinned stores under `.cancho-vcs`, one per module (`ml.rng`, `ml.matrix`, `linalg.chol`, `metrics.splits` and the others), so that another repository, `cancho-stats` first, pins them and copies nothing. `python3 scripts/publish_stores.py` writes them, `--check` fails when a committed store differs from a fresh publish of the sources, and `python3 scripts/consume_stores.py` builds a scratch package outside the repository that pins the stores of a commit and calls every module. A consumer writes, for each module it imports:

```
[dependencies.ml-rng]
git = "https://github.com/alpibrusl/cancho-ml"
rev = "<a commit that holds the stores>"
path = ".cancho-vcs/ml.rng"
```

The `pub` functions of the 23 modules are the interface (`docs/stores_api.txt`); a change to one is a pin move for the consumers. `ml` itself builds from `lib/core`, `lib/linalg` and `lib/metrics` and does not consume its own stores. [docs/stores.md](docs/stores.md)

## Built to be read by an agent

`ml introspect` and `ml skill` come from the same tables the code parses its flags with, so they cannot disagree with the program: the interface, the limits, the rules and the authority can be learned from the binary itself. A test checks that every flag and every rule is in the table, in `introspect`, in `skill`, in the schema and in a fixture. [docs/cli.md](docs/cli.md) section 1.

<!-- gen:agentio -->
```console
$ ml introspect          # one JSON document; excerpts, long lines cut

tool        ml 0.1.0   (compiler 0207556b)
flags       57, for example
  --root        path             resolve FILE, --model-file, --output and --log bene…
  --target      text             fit: the name of the column to predict, as the head…
  --output      path             fit: where to write the model file. The directory m…
  --model-file  path             predict, eval, describe: the model file to read
  --drop        text             columns to ignore, header names separated by commas…
  --dry-run     bool             fit: read, check and fit, then report what would be…
limits      24, for example
  --max-rows         default 10,000,000   ceiling 1,000,000,000
  --max-columns      default 4,096        ceiling 8,192
  --max-line-bytes   default 1,048,576    ceiling 16,777,216
  --max-bytes        default 16,777,216   ceiling 268,435,456
rules       101, for example
  args.unknown-flag        exit 2   a repair: sometimes
  ml.input.non-numeric     exit 8   a repair: sometimes
  ml.schema.mismatch       exit 8   a repair: sometimes
  ml.model.checksum        exit 8   a repair: never
  conflict.exists          exit 5   a repair: never
  limit.line-too-long      exit 8   a repair: sometimes
reads environment: false   reads clock: false
authority   args, conc, dir_read, dir_write, err_write, file_read, file_write, fs_read(""), heap, io_read, io_write
            bounded: true   foreign symbols: 0
```
<!-- /gen:agentio -->

* **Rule tags.** Match on `error.rule`, not on the message. A rule is `<area>.<name>`; the table in `introspect` says each rule's exit code and whether a repair can exist (`never`, `sometimes`, `always`). The nine tags of the libraries that no input can make `ml` answer (descriptors, a solve the pivot test never lets fail, trees, the stratified split) are registered in [docs/design.md](docs/design.md) section 8.1 and [docs/wiring.md](docs/wiring.md) section 5, and are not in `introspect`.
* **Repairs.** `repair` is `null`, or `{"kind":"retry","argv":[...]}`: an argv that can be run as it stands and that never widens what the tool may touch.
* **`--dry-run`.** Only `fit` writes; `--dry-run` answers what it would write (`planned_actions`) and exits 9. It was not run under `strace`, so "no mutating system call" is the contract package's claim here and not re-measured ([docs/cli.md](docs/cli.md) section 2).
* **Exit codes.** One per class, from cancho's agent-toolbox; with several errors the first decides the code and `errors` lists them all.

<!-- gen:exit-codes -->
| exit | `code` | meaning |
|---:|---|---|
| 0 | `SUCCESS` | what was asked was done; zero matches is success |
| 1 | `GENERAL_ERROR` | the operating system failed mid-operation, or the tool caught a bug in itself |
| 2 | `INVALID_ARGS` | the invocation is malformed or names something of the wrong kind; nothing was read |
| 3 | `NOT_FOUND` | a named input does not exist |
| 4 | `PERMISSION_DENIED` | the operating system refused, or the tool's own confinement did |
| 5 | `CONFLICT` | state is not what the caller said it would be |
| 8 | `PRECONDITION_FAILED` | the answer is no, or a limit was reached |
| 9 | `DRY_RUN` | a --dry-run completed; planned_actions lists what an apply would do |
<!-- /gen:exit-codes -->

* **The authority row.** What the compiler says the program can reach. It reads the files named on its command line and, for `predict -`, standard input (`io_read`), creates, appends to, renames and removes only beneath a directory it opened itself, and keeps its working data on the heap. `ml-predict` has the same row without `conc`, `dir_write` and `file_write`. `ml-serve` has the row of `ml-predict` without standard input and with the sockets of a server (`net_in`, `conn_accept`, `conn_read`, `conn_write`, `poll`) and `clock` (the idle timeout and the access log): no write of any file, no thread, no process, no foreign code, no outbound connection. `conc` is in the row of `ml` because `fit --threads N` calls the threaded kernels of `lib/linalg`; `tools.toml` states the reason, and the test that no thread count changes a byte is in `tests/conformance` ([docs/wiring.md](docs/wiring.md) section 6).

<!-- gen:authority -->
```console
# manifests/ml.authority.json, manifests/ml-predict.authority.json and manifests/ml-serve.authority.json
ml          args, conc, dir_read, dir_write, err_write, file_read, file_write, fs_read(""), heap, io_read, io_write
ml-predict  args, dir_read, err_write, file_read, fs_read(""), heap, io_read, io_write
ml-serve    args, clock, conn_accept, conn_read, conn_write, dir_read, err_write, file_read, fs_read(""), heap, io_write, net_in(""), poll
# bounded: true; foreign symbols: none; ffi, net_out: absent from all three; net_in, clock: ml-serve only
```
<!-- /gen:authority -->

* **Where the output is the contract package's text, not a measurement for `ml`.** `introspect.guarantees` and parts of `skill` are written by the pinned `cancho-tools` package and are wrong or too narrow for `ml` in six places, for example "read-only: it never writes" although `fit` writes. They are listed in [docs/cli.md](docs/cli.md) section 8 so that nobody reads them as claims.
* **An MCP server is not built.** `docs/cli.md` section 9 says how it would be derived from `introspect`.

## What it cannot do

* **Seven models are reachable.** `constant`, `linear`, `logistic`, `kmeans`, `tree`, `forest` and `boost` ([docs/wiring-trees.md](docs/wiring-trees.md), [docs/wiring-kmeans.md](docs/wiring-kmeans.md), [docs/models-boost.md](docs/models-boost.md)). A model is wired by its adapter and `wired = true`; the admission script is what checks it ([docs/modules.md](docs/modules.md)). k-means is not scored against a label, runs a fixed number of iterations with no stopping rule, scales nothing by itself (add a `scale` step), refuses a missing cell, and keeps the centroid of a cluster that empties where scikit-learn moves the farthest point to it. A tree, forest or boosted fit takes at most 1,048,576 training rows, the bound of the tree library's exact Gini comparison (the boosted fit keeps the same bound for the one check the wiring makes); a forest of many trees, or a boosted model of many deep rounds, can write a file longer than the 16 MiB default of `--max-bytes`, which `fit` then refuses to write (`limit.input-too-large`) because `predict` could not read it. `--seed` is one flag: it is the seed of `--split`, of a forest's bootstrap and of a boosted model's row sample and validation split. A boosted fit is sequential round by round, so `--threads` only bins the columns and speeds nothing else; `--subsample` is a seeded sample of rows, not of columns (there is no column sampling). Linear and logistic report a prediction column that is empty in every row as a missing column (`ml.schema.mismatch`); a tree or forest predicts with it missing.
* **The model file of a categorical step holds the training strings, within bounds.** `fit --cat NAMES` reads declared text columns into a dictionary sorted bytewise and an id matrix, with limits and refusals ([docs/categorical.md](docs/categorical.md)), and `encode`, `freq` and `hash` turn them into numbers ([docs/categorical-encoders.md](docs/categorical-encoders.md)). The file stores them in a string table ([docs/stringtable.md](docs/stringtable.md)): at most 131,072 strings, 8 MiB of string bytes and 12 MiB, else `ml.limit.string-table`; only the strings bound is reachable from the command line, because the reader's own limits refuse the others first. A hashed column is still interned while the fit reads the file, so `--max-categories` bounds it there; the file stores none of its strings. `predict` and `eval` take no `--cat`, `--na-values` or limit flag. A declared column that no step reads is `ml.input.categorical-not-encoded`; `--model constant` has no steps, so it refuses every declared column; a text column that is not declared is `ml.input.non-numeric`, with `--drop` as the repair. The same rows in any order give the same string table, the same schema and the same categorical steps; the numeric steps that sum (standardise, mean imputation) and the estimator's section follow the order of the rows, as the sums of the libraries do. Target encoding, native categorical splits in trees and free text are not built.
* **Binary classification only.** More than two labels is `ml.label.not-binary`.
* **`ml cv` scores the models of the pipeline only.** `--model constant` is `args.conflict`: its exact-decimal fit and eval read a whole file and take no subset of rows, so the baseline of a comparison is `ml fit` and `ml eval` on the fold files `--emit-folds` describes. A held-out side with one class has no ROC-AUC, and `ml.label.single-class` refuses the cross-validation as `ml eval` refuses that side; `--stratify` or fewer folds avoid it. A stratified group split (`StratifiedGroupKFold`) and a grouped time series are not built: they are refused as combinations.
* **`--threads` is for `fit` only, in effect.** `eval` accepts and checks the flag, and nothing in it is threaded. The scores of `linear` and `logistic` are single-threaded.
* **One red gate.** The regression forest against scikit-learn's: RMSE 8.1% apart where the tolerance is 3% (`docs/models-tree.md` section 10.8). The algorithms differ (histogram against exact splits) and most of the gap is attributed to that; the tolerance is kept and the maintainer has not decided. [docs/models-tree.md](docs/models-tree.md)
* **Three platforms agree; no other is tested.** macOS arm64, Linux x86-64 and Linux aarch64 agree on 112 fingerprint lines for 28 fits (every model kind, 1 and 8 threads: the SHA-256 of each model file and of each `predict` and `eval` answer, `docs/bench/xplat_fp.txt`, checked by `scripts/xplat_check.py`), with compiler `0207556` (the first 88 lines are those the golden had before the boosted model, unchanged; the 24 lines of its 6 fits are appended; the model file holds the pin text, `docs/ci.md` section 3): a Mac (macOS 26.2), an Ubuntu 26.04 workstation (Intel Core i7-1260P) and a Raspberry Pi 5 (Linux 6.18, aarch64). The workflow repeats the comparison on GitHub's runners for every push and pull request, on `ubuntu-24.04` and `ubuntu-26.04` (Linux x86-64), `ubuntu-24.04-arm` (Linux aarch64) and `macos-26` (macOS arm64), and has not run at this pin; the Raspberry Pi run was done by hand with `scripts/xplat_check.py`, with the compiler and `ml` built there from source, and is not in the workflow. Any other platform is untested. Three conformance cases whose answer holds an errno or depends on `/dev/stdin` run on macOS only ([docs/ci.md](docs/ci.md) section 5).
* **The CI covers the fast gates only, and the network is needed.** The workflow runs on every push and pull request what needs no numerical oracle; `scripts/check.sh` (30 to 40 minutes) runs on request and nightly in a separate job, and its result gates no pull request. `cancho install` and the compiler clone need the network ([docs/reader.md](docs/reader.md) section 1). The runner images are named, never `-latest`. The compiler is read from `cancho.toml` and built on the runner: the first run of a pin builds it uncached on each of the four runners, and the cache serves the later ones. The last run of this matrix on GitHub (2026-10-15, run 38739883040) was at the previous pin, `a4572ea`: it passed on `ubuntu-24.04` in 13 min, `ubuntu-26.04` in 17 min, `ubuntu-24.04-arm` (Linux aarch64) in 12 min and `macos-26` in 21 min, with 88 lines equal to that golden on each. No run exists at `0207556`.
* **`ml-serve` is one thread and has no TLS.** A long request holds the others, the idle timeout is the only deadline, the loopback default is proved by a socket test and not by the authority row (`net_in("")`), and a `dec` column's digits and a category's CR LF are refusals the OpenAPI document cannot state ([docs/serving-http.md](docs/serving-http.md) sections 5 and 11). An MCP server is not built.
* **No neural networks and no GPU.** They need matrix kernels, a GPU path and automatic differentiation, which cancho does not have. Inference of small exported models may come later; it is not planned in v1.
* **Not built, not promised here:** multiclass, boosting, naive Bayes, k-NN, feature importance, out-of-bag scores, a stopping rule or a choice of k for k-means.

## How fast

**No speed is claimed for `ml`: the command line has not been timed.** The libraries under it were timed, on one Mac (arm64, 16 cores, macOS 26.2; the documents that name the chip say Apple M4 Max) that other users and other builds were also using. **The load average was 2.4 or more in every table, so every cell is flagged `load >= 2` and none is a quiet-machine number: read the numbers below as the conditions of one afternoon, not as claims. The linear-algebra and model timings were taken with the summation order of one chain per block; the 4-lane order the libraries use today has not been timed.**

Fitting 1,000,000 rows of 20 features on one thread, in milliseconds, minimum of 7 interleaved samples (spread 1.01 to 1.07), load average 2.4 to 3.2 at the start and end of each cell, and 6.6 to 7.1 as the 5- and 15-minute averages ([docs/bench/models-bench.txt](docs/bench/models-bench.txt), [docs/models-linear.md](docs/models-linear.md) section 6.5):

| model | cancho | the same algorithm in numpy (Accelerate) | scikit-learn |
|---|---:|---:|---:|
| logistic regression, Newton, 6 passes | 236 | 170 | 126 (lbfgs, defaults), 223 (`tol=1e-12`) |
| linear regression, normal equations | 32.0 | 16.1 | 182 (`LinearRegression`, a different algorithm) |
| k-means, k = 8, 10 iterations | 306 | 1,148 | 330 (same initial centroids) |

**Where it is slower.** Dense float linear algebra: on the 29 shapes of the gate, one thread, against the calls a user writes (`np.dot`, `scipy.linalg.cholesky`), 8 match and 21 do not, from 1.05 times slower to 14.9 times slower (`syrk` at 100,000 by 256 is the worst), and square `gemm` is 13.8 to 16.5 times slower than Accelerate. The same algorithm written in C, with fused multiply-add off, takes 0.95 to 1.16 times as long as cancho on dot, axpy, gemv, syrk and Cholesky at 256 columns, so the gap is the vendor library and the language (no SIMD vector type), not the compiler's code generation ([docs/linalg.md](docs/linalg.md) section 5.2 and 5.3; load average 3.8 to 10). Logistic regression is 1.4 to 1.5 times slower than numpy's same algorithm and 1.9 to 2.2 times slower than scikit-learn's default call. A log-loss over a million probabilities is 3.0 times slower than a thin numpy version ([docs/metrics.md](docs/metrics.md) section 5.4).

**Where it is not.** The triangular solves, and Cholesky at 21 and 64 columns, match; k-means is level with scikit-learn (306 against 330 ms); the AUC over a million scores took 24 ms against 86 ms for a thin numpy version and 147 ms for `sklearn.roc_auc_score`. The libraries are threaded and keep their bytes: logistic regression on 1,000,000 rows takes 240 ms on one thread and 59 ms on sixteen, linear regression gains 1.4 times, because every thread owns a copy of its rows ([docs/bench/models-bench.txt](docs/bench/models-bench.txt)).

**Forests.** A forest of 10 trees on 100,000 rows fit in 0.0516 s on one thread and 0.0100 s on sixteen, where scikit-learn's `RandomForestClassifier` took 1.06 s and 0.132 s. **This is not a speed claim:** scikit-learn's forest looks for exact thresholds on the raw values and this one uses quantile bins (64 in this benchmark), so they are different algorithms making models of the same family. The quality of the classification forests agrees to within 0.0011 AUC; the regression forest does not (the red gate above).

The tables, the load of each cell and how to rerun them: [project page](https://alpibrusl.github.io/cancho-ml/benchmarks.html).

## Learn more

| | |
|---|---|
| [project page](https://alpibrusl.github.io/cancho-ml/) | the landing page: a replayed run, what is checked, the measured numbers and how to build it |
| [guide](https://alpibrusl.github.io/cancho-ml/guide.html) | the flows with their real output, the agent view and the quick start, in pages |
| [worked examples](https://alpibrusl.github.io/cancho-ml/examples.html) | four classic datasets solved by `ml` and compared with scikit-learn on the same files |
| [evidence](https://alpibrusl.github.io/cancho-ml/evidence.html) | the gates, the oracles and the mutants, and what each proves and does not prove |
| [benchmarks](https://alpibrusl.github.io/cancho-ml/benchmarks.html) | what was timed, with machine, load and minimum of N; not claims |
| [docs/status.md](docs/status.md) | what is built, what is not, what was measured, with the source document of each |
| [docs/examples.md](docs/examples.md) | the gate of the worked examples: what is compared, the `--l2` mapping, the tolerances and why |
| [docs/design.md](docs/design.md) | the plan, the promises in tiers, the model file, the refusals and the registries |
| [docs/cli.md](docs/cli.md) | the command line: gates, reference, arithmetic of the constant model, model file, log, authority |
| [docs/wiring.md](docs/wiring.md), [docs/wiring-trees.md](docs/wiring-trees.md), [docs/wiring-kmeans.md](docs/wiring-kmeans.md), [docs/modules.md](docs/modules.md), [docs/models.md](docs/models.md) | linear and logistic, then tree and forest, then k-means (no target), through the pipeline; the module contract and the admission script; the table of models |
| [docs/linalg.md](docs/linalg.md), [docs/rng.md](docs/rng.md) | the tensor and linear-algebra package and its summation order; the random generator |
| [docs/reader.md](docs/reader.md) | the float and decimal CSV reader and the dependency on cancho-table |
| [docs/metrics.md](docs/metrics.md) | metrics and splits |
| [docs/stores.md](docs/stores.md) | the 23 numeric stores other repositories pin: the list and its import closure, the API, how a consumer pins, the gates |
| [docs/models-linear.md](docs/models-linear.md), [docs/models-tree.md](docs/models-tree.md), [docs/models-boost.md](docs/models-boost.md) | linear and logistic regression and k-means; trees and forests; the red gate; gradient-boosted trees, their gates and their comparison with scikit-learn |
| [docs/spikes.md](docs/spikes.md), [docs/simd-ceiling.md](docs/simd-ceiling.md) | what was measured while deciding the design, with machine, load and minimum of N; the record of the decision, not a speed claim for `ml` |
| [docs/upstream/](docs/upstream/) | reproducers for what cancho lacks, and the proposal to cancho-table |

## Contributing

Design before code, in `docs/design.md`, with claims measured; a claim found false is edited in the document that made it. Gates are written before the code they judge and must be able to fail: a mutant per gate, all killed or each survivor explained. See [CLAUDE.md](CLAUDE.md).

`scripts/check.sh` runs the pin check, the format check, the unit tests, every library gate, the registry checks, the command line's conformance tests, the generator's drift check, the admission script on the wired models with the mutant modules it must refuse, a check that every mutant edit still applies, `scripts/check_demo.py` (the demo script and its mutants) and `scripts/check_docs.sh`. It needs the compiler at the pinned revision, the network for `cancho install` and the data files of the tree library's gate in `../ml-data`. The mutant runs (`scripts/mutants_*.py`) rebuild the tool many times and run on request.

This README and the project pages (the landing page and the guide, among others) carry program output between paired `gen` comment markers; the figures of the landing page are read from `docs/status.md`, the golden and `ml introspect` by `scripts/site.py`. `python3 scripts/site.py` rewrites them from the built `build/ml`; `scripts/check_docs.sh` runs `site.py --check`, which fails if either file differs from what the binary prints, and `scripts/check_links.py`, which fails on a link to a file that is not in the repository. `scripts/check_prose.py`, run by the same script, fails when this README, `docs/status.md` or a project page holds a phrase of history (the list is in the script, with a mutant per phrase). The prose, and every benchmark and gate figure, is written by hand next to its source document.

## Licence

[EUPL-1.2](LICENSE). Open source, provided as is, without warranty.

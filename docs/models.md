# The models (generated)

This page is the table of `lib/models/*/model.toml`, written by `scripts/gen_models.py` (`docs/modules.md`). It is not edited by hand; `scripts/check.sh` fails when it is stale.

<!-- gen:models -->
| model | kind | task | tier | wired | hyperparameters (name, range, default) | rule tags it can emit |
|---|---|---|---|---|---|---|
| `constant` | 1, 2, 3 | regression | supported | yes | `task` (regression/classification, required) | `ml.label.single-class`, `ml.label.not-binary`, `ml.label.not-integer`, `ml.metric.undefined`, `ml.model.malformed` |
| `linear` | 4 | regression | supported | yes | `l2` (finite, 0 or more, default `0.0`) | `ml.tensor.rank`, `ml.tensor.shape`, `ml.tensor.bounds`, `ml.input.non-finite`, `ml.input.too-many-columns`, `ml.param.out-of-range`, `ml.linalg.not-positive-definite`, `ml.linalg.singular`, `ml.param.threads`, `ml.fit.non-finite` |
| `logistic` | 5 | classification | supported | yes | `iterations` (1 to 1000, default `100`); `l2` (finite, 0 or more, default `0.0`) | `ml.tensor.rank`, `ml.tensor.shape`, `ml.tensor.bounds`, `ml.input.non-finite`, `ml.input.too-many-columns`, `ml.param.out-of-range`, `ml.linalg.not-positive-definite`, `ml.linalg.singular`, `ml.param.threads`, `ml.fit.non-finite`, `ml.label.single-class`, `ml.label.not-binary` |
| `kmeans` | 6 | clustering | experimental | no (library) | `k` (1 to 1024, required); `iterations` (1 to 1000, default `10`) | `ml.input.non-finite`, `ml.param.out-of-range`, `ml.param.threads` |
| `tree` | 7 | classification | experimental | no (library) | `task` (classification/regression, required); `max-depth` (1 to 30, default `8`); `min-leaf` (1 to 1000000, default `1`) | `ml.input.non-finite`, `ml.label.not-binary`, `ml.label.single-class`, `ml.limit.depth`, `ml.limit.nodes`, `ml.model.truncated`, `ml.model.malformed`, `ml.schema.mismatch` |
| `forest` | 7 | classification | experimental | no (library) | `task` (classification/regression, required); `trees` (1 to 1024, default `100`); `max-depth` (1 to 30, default `8`); `min-leaf` (1 to 1000000, default `1`) | `ml.input.non-finite`, `ml.label.not-binary`, `ml.label.single-class`, `ml.limit.depth`, `ml.limit.nodes`, `ml.param.threads` |
<!-- /gen:models -->

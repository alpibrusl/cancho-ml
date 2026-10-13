# cancho-ml

**Status: early.** The repository holds the plan (one epic and the issues under it, `docs/design.md`), the linear algebra of stage 1
(`lib/linalg`, `docs/linalg.md`) and a vertical slice of the command line (`tools/ml`, `docs/cli.md`): `ml fit | predict | eval |
introspect | skill` with one real model, `--model constant` (the training mean, or the majority class), a versioned model file, refusals
as data with rule tags, a bounded log and a checked authority manifest. The libraries of the other stages (the CSV reader, metrics and splits, linear and logistic regression, k-means, decision trees and
forests) are merged and gated but **no verb calls them yet**: see `docs/integration.md` for what is wired, the registries of model
kinds, stream kinds and refusal codes, and the result of every gate on the merged tree; `docs/cli.md` section 1 has the gates of the
command line and section 7 their verdicts. The transforms (impute, scale, encode, bucket) and the pipeline (plan text, fitted steps in the model file,
schema by name) are libraries too, not wired yet: `docs/transforms.md`, `docs/pipeline.md`.

A machine-learning library and command line written in [cancho](https://github.com/alpibrusl/cancho), for classical ML on
tabular data: transformations, pipelines, training, prediction and evaluation. It follows the shape of
[cancho-table](https://github.com/alpibrusl/cancho-table): a pure library underneath, an agent-first command line on top
(flags drive the parser, `introspect` and `skill`; refusals are data with a rule tag and a repair; bounded resources; a
checkable authority report).

## What it is for

* **Reproducible results.** The same data, the same seed and the same plan give the same bytes in the model and the
  predictions, at any thread count. Sums that decide a result are exact (the accumulators come from `cancho-table`).
* **Pipelines as data.** A pipeline is a small declarative plan (`impute`, `scale`, `encode`, then a model). The fitted
  parameters of each step are stored with the model, so training and prediction apply the same transformation.
* **Bounded and refusing.** Every model type has a size limit and a rule when the data exceeds it.

## What it is not

Deep learning. Training neural networks needs matrix kernels, a GPU path and automatic differentiation, which cancho does
not have. Inference of small exported models may come later. Nothing here is a claim about speed against scikit-learn
until the benchmark in the epic has been run and its numbers are in `docs/`.

## Where the plan is

The epic and its children are the issues of this repository. `CLAUDE.md` has the rules that apply to every change.

# cancho-ml

**Status: planned. No code yet.** This repository holds the plan: one epic and the issues under it.

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

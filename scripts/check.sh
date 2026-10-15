#!/bin/sh
# What CI runs: fmt check, unit tests, the numeric gates of stage 1, then the command line (stage 2, docs/cli.md section 1):
# package-mode build of `ml`, the schemas and the authority manifest checked against the sources, the conformance gates, and a
# check that every edit of the mutant list still applies. The gates of lib/metrics and
# lib/models/tree and the registry check are run here too. The mutants themselves (scripts/mutants_cli.py) rebuild the tool
# once per pattern, 52 of them, and run on request, as the stage 1 mutants do (the wiring stage's are scripts/mutants_wire.py: 40 patterns; the serving mutants, standard input,
# ml-predict and describe, are scripts/mutants_serve.py).
set -e
cd "$(dirname "$0")/.."
# absolute, because the conformance tests run from tests/conformance
CANCHO=${CANCHO:-$(cd .. && pwd)/cancho-compiler-a4572ea/target/release/cancho}
export CANCHO
# RD1: the cancho-table modules the reader compiles are a pinned package dependency and are not copied here (docs/reader.md section 1)
sh scripts/check_table_pin.sh
# the pinned packages, fetched into build/deps (needs the network the first time; build/deps is not committed)
$CANCHO install
$CANCHO fmt --check lib tests tools
$CANCHO test --std lib/core/*.cho lib/linalg/tensor.cho tests/core.cho
$CANCHO test --std lib/core/exact.cho tests/exact.cho
# the categorical columns (docs/categorical.md): the dictionary, the frozen vocabulary and the id matrix
$CANCHO test --std lib/core/*.cho lib/linalg/tensor.cho tests/cat.cho
$CANCHO test --std lib/core/*.cho lib/linalg/tensor.cho tests/matrix.cho
$CANCHO test --std lib/core/fsq.cho tests/fsq.cho
for T in t_dot t_gemv t_syrk t_chol t_gemm t_fuzz t_order; do
  $CANCHO test --std lib/core/*.cho lib/linalg/*.cho tests/lref.cho tests/$T.cho
done
python3 scripts/mutants_core.py --check
python3 scripts/mutants_linalg.py --check
sh scripts/build.sh
python3 scripts/oracle_linalg.py --tree
python3 scripts/oracle_linalg.py --suite
sh scripts/gate_par.sh
# lib/models (docs/models-linear.md): the program of unit, refusal, limit, thread and fuzz tests and the oracle gates at 1 to 16 threads
# (the mutants of the gate, scripts/mutants_models.py, rebuild the driver once per pattern, 87 of them, and run on request, as the other mutants do)
python3 scripts/mutants_models.py --check
sh scripts/gate_models.sh
# integration (docs/integration.md): the registries of model kinds, stream kinds and refusal codes (two stages must not claim one number), then
# the gates of the stages that were merged after the first two: lib/metrics (docs/metrics.md section 1) and lib/models/tree
# (docs/models-tree.md section 8; needs ../ml-data, the files of spike F; its mutants and the scikit-learn comparison run on request)
# (t_registry imports the transforms and the pipeline too since the merge, so they and the table decimal modules are in the program, as in gate_pipe.sh)
$CANCHO test --std lib/core/*.cho lib/linalg/*.cho lib/metrics/*.cho lib/models/*.cho lib/models/*/*.cho lib/transforms/*.cho lib/pipeline/*.cho tools/ml/csvcell.cho build/deps/*.cho tests/t_registry.cho
python3 scripts/check_registry.py
python3 scripts/mutants_metrics.py --check
sh scripts/gate_metrics.sh
sh scripts/check_tree.sh
# stage 2: the command line. Package mode: `cancho install` fetches the pinned contract modules into build/deps (not committed).
$CANCHO build --bin ml
# the prediction binary (docs/serving.md): the same sources, its own main
$CANCHO build --bin ml-predict
python3 scripts/check_bins.py
python3 scripts/check_bins.py --mutants
# the reader's gates (docs/reader.md section 5): reader-check against Python on millions of cells, the matrix and bitmap, the limits,
# the CSV shape and the fuzz
sh scripts/build_reader.sh
python3 scripts/gate_reader.py all
python3 scripts/mutants_reader.py --check
# the categorical columns of the reader against their oracle (docs/categorical.md section 1: C1 to C8; the mutants, scripts/mutants_cat.py, 42 patterns, run on request)
python3 scripts/mutants_cat.py --check
python3 scripts/gate_cat.py all
python3 scripts/gen_models.py --check
python3 scripts/schemas.py --check
python3 scripts/golden_describe.py --check
python3 scripts/manifest.py --check
(cd tests/conformance && python3 -m unittest discover -p 'test_*.py')
python3 scripts/mutants_cli.py --check
python3 scripts/mutants_serve.py --check
# the model module contract and the wiring (docs/modules.md, docs/wiring.md): the admission script on the wired models, the mutant modules it must refuse,
# the byte-for-byte oracle (also run by the conformance tests above, with its own cases) and the patterns of the wiring mutants. The mutants and the
# scikit-learn comparison (scripts/mutants_wire.py, scripts/gate_wire_sklearn.py, scripts/admit.py without --fast) run on request.
python3 scripts/admit.py constant --fast
python3 scripts/admit.py linear --fast
python3 scripts/admit.py logistic --fast
# the tree and the forest (docs/wiring-trees.md): the same admission through the blob interface, their wiring mutants (scripts/mutants_wire_trees.py, run on request) and
# the library payload of a model file against the Python oracle (also run by the conformance tests above, with its own cases)
python3 scripts/admit.py tree --fast
python3 scripts/admit.py forest --fast
# k-means (docs/wiring-kmeans.md): a clustering through the blob interface, with no target; the same admission, its wiring mutants (scripts/mutants_wire_kmeans.py, run on request)
# and the oracle (scripts/oracle_wire_kmeans.py) that the conformance tests above also run
python3 scripts/admit.py kmeans --fast
python3 tests/modules/run.py
python3 scripts/mutants_wire.py --check
python3 scripts/mutants_wire_trees.py --check
python3 scripts/mutants_wire_kmeans.py --check
# the check tools open no path that holds two dots in a row and trap on no missing operand (docs/lanes.md section 1)
python3 scripts/gate_paths.py
python3 scripts/mutants_paths.py --check
# the encoders of categorical columns (docs/categorical-encoders.md, stage 2 of issue 21): the unit program, the driver and the Python gates against the oracle
# (E1 to E6, E11, E13; the mutants, scripts/mutants_enc.py, run on request)
python3 scripts/mutants_enc.py --check
sh scripts/gate_enc.sh
# the string table of the model file (docs/stringtable.md, stage 3 of issue 21): the unit program, the Python gates against scripts/oracle_st.py and scikit-learn (S1 to S11;
# the damaged-file gate flips every fifth byte outside the table here, ST_STRIDE=1 flips every byte; the mutants, scripts/mutants_st.py, run on request)
python3 scripts/mutants_st.py --check
ST_STRIDE=5 sh scripts/gate_st.sh
# stage M3/M4: lib/transforms and lib/pipeline (docs/transforms.md, docs/pipeline.md): unit, fixture and fuzz programs, the driver, the API scan, the
# registries and the Python oracles (scripts/oracle_pipe.py --sklearn and scripts/mutants_pipe.py run on request, as the other mutants do)
python3 scripts/mutants_pipe.py --check
sh scripts/gate_pipe.sh
# the documentation checks (docs: program output in README and the page, links, the pages' basics); they need build/ml, built above
sh scripts/check_docs.sh

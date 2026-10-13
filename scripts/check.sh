#!/bin/sh
# What CI runs: fmt check, unit tests, the numeric gates of stage 1, then the command line (stage 2, docs/cli.md section 1):
# package-mode build of `ml`, the schemas and the authority manifest checked against the sources, the conformance gates, and a
# check that every edit of the mutant list still applies. The gates of lib/metrics and
# lib/models/tree and the registry check are run here too. The mutants themselves (scripts/mutants_cli.py) rebuild the tool
# 45 times and run on request, as the stage 1 mutants do.
set -e
cd "$(dirname "$0")/.."
# absolute, because the conformance tests run from tests/conformance
CANCHO=${CANCHO:-$(cd .. && pwd)/cancho-compiler-a4572ea/target/release/cancho}
export CANCHO
# RD1: the cancho-table modules the reader compiles are the pinned ones and are not copied here (LOCAL-DEV, docs/reader.md section 1.3)
sh scripts/check_table_pin.sh
$CANCHO fmt --check lib tests tools
$CANCHO test --std lib/core/*.cho lib/linalg/tensor.cho tests/core.cho
$CANCHO test --std lib/core/exact.cho tests/exact.cho
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
# (the mutants of the gate, scripts/mutants_models.py, rebuild the driver 59 times and run on request, as the other mutants do)
python3 scripts/mutants_models.py --check
sh scripts/gate_models.sh
# integration (docs/integration.md): the registries of model kinds, stream kinds and refusal codes (two stages must not claim one number), then
# the gates of the stages that were merged after the first two: lib/metrics (docs/metrics.md section 1) and lib/models/tree
# (docs/models-tree.md section 8; needs ../ml-data, the files of spike F; its mutants and the scikit-learn comparison run on request)
$CANCHO test --std lib/core/*.cho lib/linalg/*.cho lib/metrics/*.cho lib/models/*.cho lib/models/*/*.cho tests/t_registry.cho
python3 scripts/check_registry.py
python3 scripts/mutants_metrics.py --check
sh scripts/gate_metrics.sh
sh scripts/check_tree.sh
# stage 2: the command line. Package mode: `cancho install` fetches the pinned contract modules into build/deps (not committed).
$CANCHO build --bin ml
# the reader's gates (docs/reader.md section 5): reader-check against Python on millions of cells, the matrix and bitmap, the limits,
# the CSV shape and the fuzz
sh scripts/build_reader.sh
python3 scripts/gate_reader.py all
python3 scripts/mutants_reader.py --check
python3 scripts/schemas.py --check
python3 scripts/manifest.py --check
(cd tests/conformance && python3 -m unittest discover -p 'test_*.py')
python3 scripts/mutants_cli.py --check
# the check tools open no path that holds two dots in a row and trap on no missing operand (docs/lanes.md section 1)
python3 scripts/gate_paths.py
python3 scripts/mutants_paths.py --check

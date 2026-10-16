#!/bin/sh
# What CI should run for stage M7 (docs/models-tree.md section 8): the format check, then every gate except the mutants and the
# scikit-learn comparison: unit tests, the exact comparison, the identity gate against the Python oracle, the thread gate, the fuzz
# of the model reader, C1 against spike F, and a check that every mutant edit still applies. Mutants (scripts/mutants_tree.py, about
# 90 builds) and G-ref (scripts/gate_sklearn.py) run on request.
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-$(cd .. && pwd)/cancho-compiler-0207556/target/release/cancho}
export CANCHO
$CANCHO fmt --check lib/models tests/t_tree.cho tools/tree-check
python3 scripts/gen_tree_data.py
python3 scripts/gate_tree.py
python3 scripts/mutants_tree.py --check

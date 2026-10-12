#!/bin/sh
# What CI runs: fmt check, unit tests, the numeric gates of stage 1, then the command line (stage 2, docs/cli.md section 1):
# package-mode build of `ml`, the schemas and the authority manifest checked against the sources, the conformance gates, and a
# check that every edit of the mutant list still applies. The mutants themselves (scripts/mutants_cli.py) rebuild the tool
# 30 times and run on request, as the stage 1 mutants do.
set -e
cd "$(dirname "$0")/.."
# absolute, because the conformance tests run from tests/conformance
CANCHO=${CANCHO:-$(cd .. && pwd)/cancho-compiler-a4572ea/target/release/cancho}
export CANCHO
$CANCHO fmt --check lib tests tools
$CANCHO test --std lib/core/*.cho lib/linalg/tensor.cho tests/core.cho
$CANCHO test --std lib/core/exact.cho tests/exact.cho
for T in t_dot t_gemv t_syrk t_chol t_gemm t_fuzz; do
  $CANCHO test --std lib/core/*.cho lib/linalg/*.cho tests/lref.cho tests/$T.cho
done
python3 scripts/mutants_core.py --check
python3 scripts/mutants_linalg.py --check
sh scripts/build.sh
python3 scripts/oracle_linalg.py --tree
python3 scripts/oracle_linalg.py --suite
sh scripts/gate_par.sh
# stage 2: the command line. Package mode: `cancho install` fetches the pinned contract modules into build/deps (not committed).
$CANCHO build --bin ml
python3 scripts/schemas.py --check
python3 scripts/manifest.py --check
(cd tests/conformance && python3 -m unittest discover -p 'test_*.py')
python3 scripts/mutants_cli.py --check

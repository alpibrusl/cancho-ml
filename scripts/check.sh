#!/bin/sh
# What CI runs for stage 1: fmt check, unit tests, mutant pattern check. Heavier gates (mutants, oracle, benchmarks) are separate.
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-../cancho-compiler-a4572ea/target/release/cancho}
$CANCHO fmt --check lib tests tools
$CANCHO test --std lib/core/*.cho lib/linalg/tensor.cho tests/core.cho
for T in t_dot t_gemv t_syrk t_chol t_gemm t_fuzz; do
  $CANCHO test --std lib/core/*.cho lib/linalg/*.cho tests/lref.cho tests/$T.cho
done
python3 scripts/mutants_core.py --check
python3 scripts/mutants_linalg.py --check
sh scripts/build.sh
python3 scripts/oracle_linalg.py --tree
python3 scripts/oracle_linalg.py --suite

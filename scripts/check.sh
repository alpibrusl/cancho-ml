#!/bin/sh
# What CI runs for stage 1: fmt check, unit tests, mutant pattern check. Heavier gates (mutants, oracle, benchmarks) are separate.
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-../cancho-compiler-a4572ea/target/release/cancho}
$CANCHO fmt --check lib tests tools
$CANCHO test --std lib/core/*.cho lib/linalg/tensor.cho tests/core.cho
python3 scripts/mutants_core.py --check

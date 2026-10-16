#!/bin/sh
# Builds tools/enc-check (docs/categorical-encoders.md gate E1): the categorical encoders and the mixed pipeline on a CSV file, printed as bits, for
# scripts/oracle_enc.py. The reader's sources, lib/transforms, lib/pipeline and the pinned packages in build/deps (run `cancho install` once).
# ML_DEPS overrides build/deps (the mutants run in a copy with one module edited).
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-../cancho-compiler-0207556/target/release/cancho}
mkdir -p build
$CANCHO build --std lib/core/*.cho lib/linalg/tensor.cho lib/linalg/order.cho lib/metrics/sums.cho lib/transforms/*.cho lib/pipeline/*.cho \
  tools/ml/csv.cho tools/ml/csvcell.cho tools/ml/csvload.cho tools/ml/paths.cho \
  ${ML_DEPS:-build/deps}/*.cho tools/enc-check/main.cho -o ${1:-build/enc-check}

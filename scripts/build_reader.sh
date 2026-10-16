#!/bin/sh
# Builds tools/reader-check (docs/reader.md section 5) from an explicit file list: the reader's own sources, the pinned contract
# modules and cancho-table's five numeric modules, both pinned packages in build/deps (run `cancho install` once; docs/reader.md section 1).
# ML_DEPS overrides build/deps (the mutants run in a copy with one module edited).
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-../cancho-compiler-0207556/target/release/cancho}
mkdir -p build
$CANCHO build --std lib/core/*.cho lib/linalg/tensor.cho tools/ml/csv.cho tools/ml/csvcell.cho tools/ml/csvload.cho tools/ml/paths.cho \
  ${ML_DEPS:-build/deps}/*.cho tools/reader-check/main.cho -o ${1:-build/reader-check}

#!/bin/sh
# Builds tools/jsonl-check (docs/jsonl.md section 6) from an explicit file list: the JSON lines reader's own source, the file reader and the pinned modules in
# build/deps (run `cancho install` once). ML_DEPS overrides build/deps (the mutants run in a copy).
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-../cancho-compiler-0207556/target/release/cancho}
mkdir -p build
$CANCHO build --std lib/core/*.cho lib/linalg/tensor.cho tools/ml/jsonl.cho tools/ml/fsio.cho tools/ml/paths.cho \
  ${ML_DEPS:-build/deps}/*.cho tools/jsonl-check/main.cho -o ${1:-build/jsonl-check}

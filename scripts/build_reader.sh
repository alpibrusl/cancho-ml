#!/bin/sh
# Builds tools/reader-check (docs/reader.md section 5) from an explicit file list: the reader's own sources, the pinned contract
# modules in build/deps (run `cancho build --bin ml` once to install them), and the six numeric modules of the cancho-table clone
# named by table.pin (LOCAL-DEV, docs/reader.md section 1.3). TABLE_SRC overrides the directory of the six files (the mutants use a
# scratch copy with one edit); the clone itself is never written.
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-../cancho-compiler-a4572ea/target/release/cancho}
T=${TABLE_SRC:-${CANCHO_TABLE:-../cancho-table-main}/tools/table}
mkdir -p build
$CANCHO build --std lib/core/*.cho lib/linalg/tensor.cho tools/ml/csv.cho tools/ml/csvcell.cho tools/ml/csvload.cho tools/ml/paths.cho \
  build/deps/*.cho $T/pow5.cho $T/dec.cho $T/flt.cho $T/query.cho $T/facc.cho $T/reader.cho tools/reader-check/main.cho -o ${1:-build/reader-check}

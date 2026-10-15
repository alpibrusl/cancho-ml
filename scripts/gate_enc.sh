#!/bin/sh
# The gates of docs/categorical-encoders.md section 1: the unit program (hand-computed matrices, the rare bucket, the hash, the bytes of a step, the decoder
# against every damage, the mixed pipeline in memory), the driver `enc-check`, and the Python gates that compare it and `ml` with scripts/oracle_enc.py
# (E1 to E6, E11, E13). ENC_FAST=1 runs them on fewer cases (the mutants of scripts/mutants_enc.py). Needs build/ml (`cancho build --bin ml`) and the pinned
# packages in build/deps.
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-$(cd .. && pwd)/cancho-compiler-0207556/target/release/cancho}
export CANCHO
DEPS=${ML_DEPS:-$(pwd)/build/deps}
LIB="lib/core/*.cho lib/linalg/tensor.cho lib/linalg/order.cho lib/metrics/sums.cho lib/transforms/*.cho lib/pipeline/*.cho tools/ml/csvcell.cho $DEPS/*.cho tests/pt.cho"
D=$(mktemp -d)
trap 'rm -rf $D' EXIT
$CANCHO test --std $LIB tests/enc.cho > $D/enc.out 2>&1 || true
if grep -q "error:" $D/enc.out; then echo BUILD_ERROR; cat $D/enc.out; exit 1; fi
if grep -q FAILED $D/enc.out || ! grep -q "test result: ok" $D/enc.out; then cat $D/enc.out; exit 1; fi
echo "tests/enc.cho: $(grep 'test result' $D/enc.out)"
sh scripts/build_enc.sh build/enc-check > $D/build.out 2>&1 || { echo BUILD_ERROR; cat $D/build.out; exit 1; }
python3 scripts/gate_enc.py ${1:-all}

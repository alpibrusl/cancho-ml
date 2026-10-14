#!/bin/sh
# The gates of docs/stringtable.md section 9: the unit program of the string table (the bytes written by hand, every field out of its range, the bounds at the limit and
# one past it, every flipped byte and truncation with the digests repaired and not) and the Python gates that run `ml` against scripts/oracle_st.py (S1, S2, S4, S5, S7, S8,
# S9, S10) and scikit-learn (S11). ST_STRIDE=N flips and cuts every N-th byte of the files outside the string table in the damaged-file gate (default 1: every byte);
# ST_FAST=1 runs the gates on fewer cases (the mutants of scripts/mutants_st.py). Needs build/ml (`cancho build --bin ml`) and the pinned packages in build/deps.
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-$(cd .. && pwd)/cancho-compiler-a4572ea/target/release/cancho}
export CANCHO
DEPS=${ML_DEPS:-$(pwd)/build/deps}
LIB="lib/core/*.cho lib/linalg/tensor.cho lib/linalg/order.cho lib/metrics/sums.cho lib/transforms/*.cho lib/pipeline/*.cho tools/ml/csvcell.cho $DEPS/*.cho tests/pt.cho"
D=$(mktemp -d)
trap 'rm -rf $D' EXIT
$CANCHO test --std $LIB tests/strtab.cho > $D/st.out 2>&1 || true
if grep -q "error:" $D/st.out; then echo BUILD_ERROR; cat $D/st.out; exit 1; fi
if grep -q FAILED $D/st.out || ! grep -q "test result: ok" $D/st.out; then cat $D/st.out; exit 1; fi
echo "tests/strtab.cho: $(grep 'test result' $D/st.out)"
python3 scripts/oracle_st.py
python3 scripts/gate_st.py ${1:-all}
if [ -z "$ST_FAST" ] && [ "${1:-all}" = all ]; then
  python3 scripts/gate_st_sklearn.py
fi

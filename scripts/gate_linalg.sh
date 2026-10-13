#!/bin/sh
# The correctness gate of one kernel group: unit tests against the reference, then the compiled driver against the numpy
# oracle. Usage: gate_linalg.sh TESTFILE KERNEL...   (exit non-zero on any failure). Run from the repository (or a copy of it).
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-../cancho-compiler-a4572ea/target/release/cancho}
T=$1; shift
OUT=$($CANCHO test --std lib/core/*.cho lib/linalg/*.cho tests/lref.cho tests/$T 2>&1) || { echo "$OUT" | tail -5; exit 1; }
echo "$OUT" | grep -q FAILED && { echo "$OUT" | grep FAILED; exit 1; }
BIN=$(mktemp -d)/linalg-check
$CANCHO build --std lib/core/*.cho lib/linalg/*.cho tools/linalg-check/main.cho -o $BIN || exit 1
for K in "$@"; do
  LINALG_CHECK=$BIN python3 scripts/oracle_linalg.py --suite --kernel $K || exit 1
done

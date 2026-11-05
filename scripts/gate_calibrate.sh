#!/bin/sh
# The gate of the calibration table (docs/calibration.md section 5), cheapest first: the unit test worked by hand, the compiled driver against the independent oracle (scripts/gate_calibrate.py), the refusals. This is what
# scripts/mutants_calibrate.py runs on each mutant. CASES (default 200) sets the number of random tables. Exit non-zero on any failure.
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-$(cd .. && pwd)/cancho-compiler-0207556/target/release/cancho}
D=$(mktemp -d)
trap 'rm -rf $D' EXIT
LIB="lib/core/*.cho lib/linalg/tensor.cho lib/linalg/order.cho lib/metrics/*.cho"
perl -e 'alarm 120; exec @ARGV' $CANCHO test --std $LIB tests/mt.cho tests/t_calibrate.cho > $D/t.out 2>&1 || { cat $D/t.out; exit 1; }
grep -q FAILED $D/t.out && { cat $D/t.out; exit 1; }
$CANCHO build --std $LIB tools/metrics-check/*.cho -o $D/metrics-check
METRICS_CHECK=$D/metrics-check python3 scripts/gate_calibrate.py --cases "${CASES:-200}" --no-controls

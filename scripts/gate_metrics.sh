#!/bin/sh
# The whole gate of lib/metrics (docs/metrics.md section 1), cheapest first, stopping at the first failure: unit tests, the
# compiled driver against the independent oracles (splits, draw, ratio, counts, AUC, ordered sums, sklearn), the tag fixtures and the
# fuzz. This is what scripts/mutants_metrics.py runs on each mutant. Exit non-zero on any failure.
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-$(cd .. && pwd)/cancho-compiler-a4572ea/target/release/cancho}
D=$(mktemp -d)
trap 'rm -rf $D' EXIT
LIB="lib/core/*.cho lib/linalg/tensor.cho lib/linalg/order.cho lib/metrics/*.cho"
for T in t_metrics t_metrics_split t_metrics_fuzz; do
  $CANCHO test --std $LIB tests/mt.cho tests/$T.cho > $D/$T.out 2>&1 || { cat $D/$T.out; exit 1; }
  grep -q FAILED $D/$T.out && { cat $D/$T.out; exit 1; }
done
$CANCHO build --std $LIB tools/metrics-check/*.cho -o $D/metrics-check
export METRICS_CHECK=$D/metrics-check
python3 scripts/gate_metrics.py --tags
python3 scripts/oracle_metrics.py --ratio --draw --metrics --auc --sums
python3 scripts/oracle_metrics.py --splits
python3 scripts/oracle_metrics.py --sklearn
python3 scripts/gate_metrics.py --fuzz

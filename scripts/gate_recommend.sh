#!/bin/sh
# The gate of the pure parts of the recommender (docs/recommend.md section 9, gates RC1 to RC5, RC10, RC12): the metrics driver and the log driver against the independent oracle,
# cheapest first, stopping at the first failure. scripts/check.sh runs it; the mutants (scripts/mutants_recommend.py, mutants_recommend_log.py) run the same oracle commands.
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-$(cd .. && pwd)/cancho-compiler-0207556/target/release/cancho}
D=$(mktemp -d)
trap 'rm -rf $D' EXIT
LIB="lib/core/*.cho lib/linalg/tensor.cho lib/linalg/order.cho lib/metrics/*.cho"
$CANCHO build --std $LIB tools/metrics-check/*.cho -o $D/metrics-check
$CANCHO build --std lib/core/*.cho lib/linalg/*.cho lib/metrics/*.cho lib/models/common.cho lib/models/recommend/*.cho tools/recommend-check/*.cho -o $D/recommend-check
export METRICS_CHECK=$D/metrics-check
export RECOMMEND_CHECK=$D/recommend-check
python3 scripts/oracle_recommend.py --controls --driver --metrics 300
python3 scripts/oracle_recommend.py --log-controls --log 300
python3 scripts/oracle_recommend.py --als-controls --als 300
python3 scripts/oracle_recommend.py --payload 20
python3 scripts/mutants_recommend.py --check
python3 scripts/mutants_recommend_log.py --check
python3 scripts/mutants_recommend_als.py --check
python3 scripts/mutants_recommend_model.py --check

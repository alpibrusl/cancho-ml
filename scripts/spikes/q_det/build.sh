#!/bin/sh
# Build spike_q (docs/spikes.md section 15) and the unchanged spike S (arm A's oracle, G-A) into build/. Default backend, as section 1.1.
set -e
cd "$(dirname "$0")/../../.."
CANCHO=${CANCHO:-../cancho-compiler-0207556/target/release/cancho}
SPIKE_S=${SPIKE_S:-scripts/spikes/spike_s.cho}
mkdir -p build
$CANCHO build --std lib/linalg/order.cho scripts/spikes/q_det/facc.cho scripts/spikes/q_det/spike_q.cho -o build/spike_q
$CANCHO build --std "$SPIKE_S" -o build/spike_s

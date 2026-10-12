#!/bin/bash
# Ablation: backward substitution, k descending (implemented, 4 columns in flight) against k ascending (one serial chain).
# Prints microseconds per solve, minimum of 7 runs, d = 21, 64, 256. docs/linalg.md section 5.
set -e
cd "$(dirname "$0")/../.."
scripts/ablate/variant.sh bwd_asc scripts/ablate/edit_bwd_ascending.py >/dev/null
for d in 21 64 256; do
  reps=$((2000000 / d)); line="d=$d:"
  for v in "build/linalg-check:fwd(asc,4 rows)" "build/linalg-check:bwd(desc,4 cols)" "build/ablate/bwd_asc/linalg-check:bwd(asc,serial)"; do
    bin=${v%%:*}; label=${v##*:}; kern=fwd; case "$label" in bwd*) kern=bwd;; esac; best=999999999
    for i in 1 2 3 4 5 6 7; do ms=$($bin $kern $d 0 1 $reps 1 | awk '/^ms/{print $2}'); [ "$ms" -lt "$best" ] && best=$ms; done
    line="$line $label=$(echo "scale=3; $best*1000/$reps" | bc)us"
  done
  echo "$line"
done

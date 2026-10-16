#!/bin/bash
# Ablation: rows in flight of gemv_n (docs/linalg.md section 5). Builds 2..6-row variants of the current 4-row kernel and times
# 100,000 x d for d = 20, 64, 128, 256 (100 reps below d = 128, 40 above, 7 runs each, minimum); the 8-row figure is the kernel as first written
# (git show 432591a:lib/linalg/gemv.cho). Prints ms per rep.
set -e
cd "$(dirname "$0")/../.."
for n in 2 3 5 6; do NROWS=$n scripts/ablate/variant.sh rows$n scripts/ablate/edit_gemv_rows.py >/dev/null; done
# the 8-row kernel as first written (commit 432591a)
D=build/ablate/rows8; mkdir -p $D; cp -r lib tools $D/; git show 432591a:lib/linalg/gemv.cho > $D/lib/linalg/gemv.cho
../cancho-compiler-0207556/target/release/cancho build --std $D/lib/core/*.cho $D/lib/linalg/*.cho $D/tools/linalg-check/main.cho -o $D/linalg-check
for d in 20 64 128 256; do
  reps=100; [ $d -ge 128 ] && reps=40
  line="d=$d:"
  for v in "build/linalg-check:current(4)" "build/ablate/rows8/linalg-check:rows8" $(for n in 2 3 5 6; do echo "build/ablate/rows$n/linalg-check:rows$n"; done); do
    bin=${v%%:*}; label=${v##*:}; best=999999
    for i in 1 2 3 4 5 6 7; do ms=$($bin gemv_n 100000 $d 1 $reps 1 | awk '/^ms/{print $2}'); [ "$ms" -lt "$best" ] && best=$ms; done
    line="$line $label=$(echo "scale=3; $best/$reps" | bc)ms"
  done
  echo "$line"
done

#!/bin/bash
# Ablation: what would fused multiply-add (and nothing else) change, in C? Same code built with -ffp-contract=off (the rule) and
# -ffp-contract=fast (fma allowed); the bits differ, only the time is looked at. Minimum of 7 runs, ms per repetition.
set -e
cd "$(dirname "$0")/../.."
mkdir -p build
cc -O3 -ffp-contract=off  -mcpu=native -o build/parity scripts/c_parity/parity.c -lm
cc -O3 -ffp-contract=fast -mcpu=native -o build/parity_fma scripts/c_parity/parity.c -lm
best() { local b=999999999; for i in 1 2 3 4 5 6 7; do m=$("$@" | awk '/^ms/{printf "%d", $2*1000000}'); [ "$m" -lt "$b" ] && b=$m; done; echo $b; }
for c in "syrk 20000 64 1 5" "syrk 100000 20 1 5" "gemv_n 100000 64 1 20" "gemv_t 100000 64 1 20" "chol 256 0 1 100" "dot 1000000 0 1 300" "dot_lanes 1000000 0 1 300"; do
  set -- $c; reps=$5
  off=$(best build/parity $c); fma=$(best build/parity_fma $c)
  echo "$1 n=$2 d=$3: no-fma $(echo "scale=3; $off/1000000/$reps" | bc) ms   fma $(echo "scale=3; $fma/1000000/$reps" | bc) ms   fma/no-fma $(echo "scale=2; $fma/$off" | bc)"
done

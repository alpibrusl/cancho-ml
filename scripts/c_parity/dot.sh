#!/bin/bash
# Ablation of the dot product, n = 1,000,000, in C (clang -O3 -ffp-contract=off): the block rule (as implemented), the same scalar
# code over 8 adjacent lanes (another order), one chain, and NEON intrinsics over 8 lanes (4 vector accumulators). Minimum of 7
# runs of 300, microseconds per dot. The numpy figure of the same moment is printed for reference.
set -e
cd "$(dirname "$0")/../.."
mkdir -p build; cc -O3 -ffp-contract=off -mcpu=native -o build/parity scripts/c_parity/parity.c -lm
for k in dot dot_lanes dot_chain dot_neon; do
  b=999999999; for i in 1 2 3 4 5 6 7; do m=$(build/parity $k 1000000 0 1 300 | awk '/^ms/{printf "%d", $2*1000000}'); [ "$m" -lt "$b" ] && b=$m; done
  echo "C $k: $(echo "scale=1; $b/300/1000" | bc) us"
done
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 - <<'PY'
import sys, time, numpy as np
sys.path.insert(0, "scripts"); import oracle_linalg as O
x, y = O.elements(O.derive(1, 1), 1000000), O.elements(O.derive(1, 2), 1000000)
best = 1e9
for _ in range(7):
    t = time.perf_counter()
    for _ in range(300): np.dot(x, y)
    best = min(best, (time.perf_counter() - t) / 300)
print("numpy np.dot: %.1f us" % (best * 1e6))
PY

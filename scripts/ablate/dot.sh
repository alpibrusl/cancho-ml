#!/bin/bash
# Ablation: the block rule of the dot product against the same scalar code without it (docs/linalg.md section 5).
#   blocks  CxL : C blocks of length L summed together (the rule with L = 256, C = 8 is the implemented one, in a loop that does
#                 nothing else); the C partials are added at the end in order
#   lanes   C   : C accumulators over ADJACENT elements (the usual unrolled dot product; a different summation order)
# Prints microseconds per dot of n = 1,000,000, minimum of 7 runs of 100. Outputs of the lanes variants are NOT the rule's.
set -e
cd "$(dirname "$0")/../.."
C=${CANCHO:-../cancho-compiler-a4572ea/target/release/cancho}
D=build/ablate/dot; mkdir -p $D
for cfg in "256 8" "256 16" "240 16"; do set -- $cfg
  python3 scripts/ablate/dot_blocks_gen.py $1 $2 > $D/b_$1_$2.cho; $C build --std lib/core/rng.cho $D/b_$1_$2.cho -o $D/b_$1_$2
  echo "blocks $2 x $1 : $($D/b_$1_$2 1000000 100 | tail -1)"
done
for cfg in "256 4" "256 8" "256 16"; do set -- $cfg
  python3 scripts/ablate/dot_lanes_gen.py $1 $2 > $D/l_$1_$2.cho; $C build --std lib/core/rng.cho $D/l_$1_$2.cho -o $D/l_$1_$2
  echo "lanes  $2 : $($D/l_$1_$2 1000000 100 | tail -1)"
done

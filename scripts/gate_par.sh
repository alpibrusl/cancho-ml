#!/bin/sh
# The determinism gate of the threaded kernels: the unit program (threads against sequential, bits) and the compiled driver
# against the numpy oracle at 1, 2, 3, 4, 8 and 16 threads. Exit non-zero on any difference.
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-../cancho-compiler-0207556/target/release/cancho}
D=$(mktemp -d)
$CANCHO build --std lib/core/*.cho lib/linalg/*.cho tests/lref.cho tests/t_par.cho -o $D/t_par
$D/t_par | grep -q FAILED && { $D/t_par; exit 1; }
$CANCHO build --std lib/core/*.cho lib/linalg/*.cho tools/linalg-check/main.cho -o $D/linalg-check
# the oracle's own exit status decides (it prints "suite: all bits equal" and exits 0, or MISMATCH lines and exits 1); a crash of the
# oracle is a failure too (before v1-lanes this line only searched the output for MISMATCH, so an oracle that died passed)
LINALG_CHECK=$D/linalg-check python3 scripts/oracle_linalg.py --suite --thread-suite --threads ${THREADS:-1,2,3,4,8,16} > $D/par.out || { grep -v " ok$" $D/par.out | head -8; exit 1; }
grep -q "suite: all bits equal" $D/par.out || { cat $D/par.out | head -8; exit 1; }
exit 0

#!/bin/sh
# The determinism gate of the threaded kernels: the unit program (threads against sequential, bits) and the compiled driver
# against the numpy oracle at 1, 2, 3, 4, 8 and 16 threads. Exit non-zero on any difference.
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-../cancho-compiler-a4572ea/target/release/cancho}
D=$(mktemp -d)
$CANCHO build --std lib/core/*.cho lib/linalg/*.cho tests/lref.cho tests/t_par.cho -o $D/t_par
$D/t_par | grep -q FAILED && { $D/t_par; exit 1; }
$CANCHO build --std lib/core/*.cho lib/linalg/*.cho tools/linalg-check/main.cho -o $D/linalg-check
LINALG_CHECK=$D/linalg-check python3 scripts/oracle_linalg.py --suite --thread-suite --threads ${THREADS:-1,2,3,4,8,16} | grep -v " ok$" | grep -q MISMATCH && {
  LINALG_CHECK=$D/linalg-check python3 scripts/oracle_linalg.py --suite --thread-suite --threads ${THREADS:-1,2,3,4,8,16} | grep MISMATCH | head -5; exit 1; }
exit 0

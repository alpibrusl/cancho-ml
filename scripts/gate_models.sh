#!/bin/sh
# The numeric gates of lib/models (docs/models-linear.md section 2): the program of unit, refusal, limit, thread and fuzz tests,
# and the compiled driver against the order-matched numpy oracles (G-oracle) and at 1, 2, 4, 8 and 16 threads (G-det). Exit
# non-zero on any failing test or any bit that differs. `scripts/mutants_models.py` runs this script on every mutant of the sources.
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-../cancho-compiler-0207556/target/release/cancho}
D=build/gate-models
mkdir -p $D
FILES="lib/core/*.cho lib/linalg/*.cho lib/models/*.cho lib/models/*/*.cho"
# a source that does not compile is not a failing gate: say so (scripts/mutlib.py does not count it as a kill)
$CANCHO build --std $FILES tools/models-check/main.cho -o $D/models-check || { echo BUILD_ERROR; exit 3; }
$CANCHO build --std $FILES tests/p_models.cho -o $D/p_models || { echo BUILD_ERROR; exit 3; }
$D/p_models
# the unit, refusal, limit and fuzz tests of the softmax regression (docs/multiclass.md section 8)
$CANCHO build --std $FILES tests/p_softmax.cho -o $D/p_softmax || { echo BUILD_ERROR; exit 3; }
$D/p_softmax
MODELS_CHECK=$D/models-check python3 scripts/oracle_models.py --suite
MODELS_CHECK=$D/models-check python3 scripts/oracle_models.py --suite --thread-suite --threads ${THREADS:-1,2,4,8,16}

#!/bin/sh
# What `cancho build` in package mode would pass, from an explicit file list (package mode needs `cancho install`: network).
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-../cancho-compiler-a4572ea/target/release/cancho}
mkdir -p build
$CANCHO build --std lib/core/*.cho lib/linalg/*.cho tools/linalg-check/main.cho -o ${1:-build/linalg-check}

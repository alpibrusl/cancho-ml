#!/bin/sh
# What `cancho build` in package mode would pass, from an explicit file list. The stage 1 driver (linalg-check) imports nothing
# from the contract package, so it is built from lib/ alone; `ml` is built in package mode (`cancho build --bin ml`, see check.sh).
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-../cancho-compiler-a4572ea/target/release/cancho}
mkdir -p build
$CANCHO build --std lib/core/*.cho lib/linalg/*.cho tools/linalg-check/main.cho -o ${1:-build/linalg-check}

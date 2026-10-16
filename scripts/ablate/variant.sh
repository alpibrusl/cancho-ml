#!/bin/bash
# Build a variant of build/linalg-check: variant.sh NAME EDIT.py   (EDIT.py edits the copy of lib/ in its working directory)
# The copy lives in build/ablate/NAME; the binary is build/ablate/NAME/linalg-check. Used for the ablations of docs/linalg.md.
set -e
cd "$(dirname "$0")/../.."
ROOT=$PWD
C=${CANCHO:-$ROOT/../cancho-compiler-0207556/target/release/cancho}
name=$1; edit=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
rm -rf build/ablate/$name; mkdir -p build/ablate/$name; cp -r lib tools build/ablate/$name/
(cd build/ablate/$name && python3 "$edit")
$C build --std build/ablate/$name/lib/core/*.cho build/ablate/$name/lib/linalg/*.cho build/ablate/$name/tools/linalg-check/main.cho -o build/ablate/$name/linalg-check

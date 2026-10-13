#!/bin/sh
# The whole gate of lib/transforms and lib/pipeline (docs/transforms.md section 1, docs/pipeline.md section 1), cheapest first, stopping at the
# first failure: the unit, fixture and fuzz programs, the driver, the API scan, the registries and the Python oracles. This is what
# scripts/mutants_pipe.py runs on each mutant. PIPE_FAST=1 runs the oracles on fewer cases (the mutants). Exit non-zero on any failure; a compile
# error prints BUILD_ERROR (a mutant that does not compile is not a kill).
set -e
cd "$(dirname "$0")/.."
CANCHO=${CANCHO:-$(cd .. && pwd)/cancho-compiler-a4572ea/target/release/cancho}
# the pinned packages (cancho-table's numeric modules among them), installed by `cancho install`; ML_DEPS names another directory (the mutants)
DEPS=${ML_DEPS:-$(pwd)/build/deps}
D=$(mktemp -d)
trap 'rm -rf $D' EXIT
LIB="lib/core/*.cho lib/linalg/tensor.cho lib/linalg/order.cho lib/metrics/sums.cho lib/transforms/*.cho lib/pipeline/*.cho tools/ml/csvcell.cho $DEPS/*.cho tests/pt.cho"
for TST in t_transforms t_transforms_fuzz t_pipeline t_pipeline_fuzz; do
  $CANCHO test --std $LIB tests/$TST.cho > $D/$TST.out 2>&1 || true
  if grep -q "error:" $D/$TST.out; then echo BUILD_ERROR; cat $D/$TST.out; exit 1; fi
  if grep -q FAILED $D/$TST.out || ! grep -q "test result: ok" $D/$TST.out; then cat $D/$TST.out; exit 1; fi
  echo "$TST: $(grep 'test result' $D/$TST.out)"
done
# the registry program: all libraries in one program (module names must not collide)
$CANCHO test --std lib/core/*.cho lib/linalg/*.cho lib/metrics/*.cho lib/models/*.cho lib/models/*/*.cho lib/transforms/*.cho lib/pipeline/*.cho tools/ml/csvcell.cho $DEPS/*.cho tests/t_registry.cho > $D/reg.out 2>&1 || true
if grep -q "error:" $D/reg.out; then echo BUILD_ERROR; cat $D/reg.out; exit 1; fi
if grep -q FAILED $D/reg.out || ! grep -q "test result: ok" $D/reg.out; then cat $D/reg.out; exit 1; fi
echo "t_registry: $(grep 'test result' $D/reg.out)"
python3 scripts/check_registry.py
python3 scripts/check_pipeline_api.py
$CANCHO build --std $LIB tools/pipe-check/main.cho -o $D/pipe-check > $D/build.out 2>&1 || { echo BUILD_ERROR; cat $D/build.out; exit 1; }
export PIPE_CHECK=$D/pipe-check
if [ -n "$PIPE_FAST" ]; then
  export PIPE_PLANS=${PIPE_PLANS:-250} PIPE_CASES=${PIPE_CASES:-500} PIPE_CHAINS=${PIPE_CHAINS:-150} PIPE_FILES=${PIPE_FILES:-40}
fi
python3 scripts/oracle_pipe.py --sums --plans --transforms --pipeline --file

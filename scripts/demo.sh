#!/bin/sh
# A reproducible demo of cancho-ml for a screen recording: two tracked experiments, their comparison and replay, the train/held-out marker, a model served over
# HTTP and a curl request. Every line printed is real output of the binaries.
#
#   BIN=/path/to/build sh scripts/demo.sh            (BIN holds ml and ml-serve; default ./build)
#
# It works in a temporary directory, picks a free port, and stops only the server it started. ml-serve listens on 127.0.0.1 by default, so it needs no token
# and no flag besides the model and the port; SERVE_FLAGS adds flags (for example `--bind all --token-file FILE`).
set -eu
BIN=${BIN:-$(cd "$(dirname "$0")/.." && pwd)/build}
DATA=${DATA:-$(cd "$(dirname "$0")/.." && pwd)/docs/examples/data}
SERVE_FLAGS=${SERVE_FLAGS:-}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/cancho-ml-demo.XXXXXX")
SERVER_PID=
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT INT TERM
mkdir -p "$WORK/runs"
cp "$DATA/iris_train.csv" "$DATA/iris_test.csv" "$WORK/"
cd "$WORK"
ml() { "$BIN/ml" --root . "$@"; }
say() { printf '\n$ %s\n' "$*"; }
pick() { python3 -c "import json,sys; d=json.load(sys.stdin); d=d.get('data',d); print($1)"; }

say 'ml fit --model logistic --target virginica --step "scale column=* kind=standard" --l2 0.1 --split train=0.8 --seed 3 --track runs --output iris.ml iris_train.csv'
A=$(ml fit --model logistic --target virginica --step 'scale column=* kind=standard' --l2 0.1 --split train=0.8 --seed 3 --track runs --output iris.ml iris_train.csv | pick "d['run']['id']")
echo "run $A"

say 'ml fit ... --l2 1.0 --track runs --output iris_l2_1.ml iris_train.csv'
C=$(ml fit --model logistic --target virginica --step 'scale column=* kind=standard' --l2 1.0 --split train=0.8 --seed 3 --track runs --output iris_l2_1.ml iris_train.csv | pick "d['run']['id']")
echo "run $C"

say 'ml runs compare runs A C'
ml runs compare runs "$A" "$C" | pick "'\n'.join(['differs: '+i['key']+'  '+repr(i['a'].splitlines()[-1])+' -> '+repr(i['b'].splitlines()[-1]) for i in d['inputs']]+['%-10s %s -> %s' % (m['name'], m['a'][:6], m['b'][:6]) for m in d['metrics']])"

say 'ml runs verify runs A        # re-fits from the record and compares the model SHA-256'
ml runs verify runs "$A" | pick "'verified: %s  model %s' % (d['verified'], d['model_sha256'])"

say 'ml runs show runs A          # the record names the model file by its SHA-256 too'
ml runs show runs "$A" | pick "'record: model %s' % d['record']['model']['sha256']"

say 'ml eval --model-file iris.ml iris_train.csv   (and iris_test.csv)'
for f in iris_train.csv iris_test.csv; do
  ml eval --model-file iris.ml "$f" | pick "'$f  on: %s  rows: %s  out_of_range cells: %s  accuracy: %s' % (d['on'], d['rows'], d['out_of_range']['cells'], d['metrics']['accuracy'])"
done

PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
say "ml-serve --model-file iris.ml --port $PORT $SERVE_FLAGS &"
# shellcheck disable=SC2086
"$BIN/ml-serve" --root . --model-file iris.ml --port "$PORT" $SERVE_FLAGS > serve.log 2>&1 &
SERVER_PID=$!
i=0; until grep -q listening serve.log 2>/dev/null || [ $i -ge 50 ]; do sleep 0.1; i=$((i+1)); done
cat serve.log

say "curl -s localhost:$PORT/predict -d '{\"sepal_length_cm\":5.6,\"sepal_width_cm\":2.8,\"petal_length_cm\":4.9,\"petal_width_cm\":2.0}'"
curl -s --max-time 5 -X POST "localhost:$PORT/predict" -H 'content-type: application/json' \
  -d '{"sepal_length_cm":5.6,"sepal_width_cm":2.8,"petal_length_cm":4.9,"petal_width_cm":2.0}'
echo

say "curl -s localhost:$PORT/predict -d '{\"sepal_length_cm\":\"abc\"}'     # a refusal is data too"
curl -s --max-time 5 -X POST "localhost:$PORT/predict" -H 'content-type: application/json' -d '{"sepal_length_cm":"abc"}' | cut -c1-260
echo

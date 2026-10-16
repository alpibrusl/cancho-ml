#!/usr/bin/env python3
"""Regenerate the parts of the README and of the project pages that are program
output, from the built binary, so that they cannot drift from it.

    python3 scripts/site.py            # rewrite the generated regions
    python3 scripts/site.py --check    # change nothing; exit 1 if a file differs
    python3 scripts/site.py --bin build/ml

The mechanism is cancho-table's (scripts/site.py there). A generated region is a
pair of comment lines in a file,

    <!-- gen:NAME -->
    ...whatever the script writes...
    <!-- /gen:NAME -->

and nothing outside a pair is touched: **the prose of README.md, docs/index.html,
docs/evidence.html and docs/benchmarks.html is hand-written, and so is every
number in them that is not program output** (the gate and benchmark figures are
copied from the document that cites them, next to their conditions). What is
generated:

    t-NAME        README.md, index.html   the example for one task: the command and
                                          what the built binary prints for it
    flow          README.md, index.html   one complete flow: fit, predict, eval, a
                                          refusal with its repair, the repair run
    pflow         README.md, index.html   the same for a model that goes through the pipeline: a
                                          float CSV with an empty cell, a refusal with its repair,
                                          impute + standardise + logistic, predict --proba, eval
    qs            README.md, index.html   the quick start's commands and output
    kmflow        README.md, index.html   k-means on docs/examples/data/iris_train.csv and iris_test.csv: no target (a --target is refused), k required, the label dropped,
                                          the features standardised, three clusters, predict with the squared distance, eval
    catflow       README.md, index.html   the categorical flow on docs/examples/data/pets.csv: a declared column that no step
                                          reads is refused, a fit with categorical steps writes the model file (one-hot, rare bucket,
                                          frequency), predict and eval read it, new rows with unseen categories in another column
                                          order, and a hashed column that stores no string
    splitflow     README.md, index.html   the split strategies and cross-validation (docs/splits.md): a stratified holdout on docs/examples/data/breast_cancer_train.csv, a time split and
                                          a group split on docs/examples/data/visits.csv, a refused combination, `ml cv --folds 5` on iris and diabetes with the metrics of
                                          each fold and their mean and standard deviation, a time cv, a group cv with --emit-folds, and a refusal with its repair
    hero          index.html              the first example
    agentio       README.md, index.html   excerpts of `ml introspect`
    skill         index.html              the first lines of `ml skill`
    authority     README.md, index.html   the row of manifests/ml.authority.json
                                          (checked against `ml introspect`, and
                                          for net, ffi and clock)
    exit-codes    README.md               the exit codes `ml introspect` lists
    limits, rules, nrules   index.html    the limits and rules `ml introspect` lists

Two whole files are also written (and checked) from the list of pages below, so
that the pages and the sitemap cannot disagree: docs/sitemap.xml and
docs/robots.txt.

The commands run in a temporary directory on fixed small files, in the order of
DEMOS, as `ml` found on PATH (so a repair that names the tool says `ml`). Each
command runs once and its output is reused wherever it is shown; a model file
written by one is read by the next. Standard output and standard error are one
stream, as in a terminal. Needs the standard library and the built binary
(`cancho install && cancho build --bin ml`, see the README).
"""

import argparse
import difflib
import html
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import site_examples as sx  # noqa: E402  (the worked examples: docs/examples.md)
from site_examples import Drift  # noqa: E402

FILES = {
    # four houses' worth of numbers; area is a float column (a decimal point, so it is read as a decimal or a double)
    "houses.csv": "id,rooms,area,price\n1,2,48.5,180000\n2,3,71.25,240000\n3,3,64,215000\n4,4,98.5,335000\n5,1,31,120000\n",
    # new rows to predict for: no price column is needed
    "new.csv": "id,rooms,area\n6,2,52.5\n7,3,80\n",
    # a row with an area no house has: the model's answer is past the largest double
    "huge.csv": "id,rooms,area\n8,2,1e308\n",
    # no byte at all: the standard input of a forgotten operand
    "empty.csv": "",
    # the same with a column of text
    "listings.csv": "id,rooms,area,city\n6,2,52.5,Lyon\n7,3,80,Porto\n",
    # a two-class target
    "shop.csv": "id,visits,spend,bought\n1,3,12.50,1\n2,1,0.00,0\n3,7,40.25,1\n4,2,5.00,0\n5,5,22.75,1\n6,4,18.00,1\n",
    # a float target (an exponent makes a cell a float)
    "sensor.csv": "id,x,reading\n1,1.5,1e1\n2,2.5,2.75\n3,0.1,3e-1\n4,4,5.5\n",
    # the pipeline flow: float columns, an empty cell, two classes that overlap (so a logistic regression has a finite fit)
    "shop2.csv": "id,visits,spend,bought\n1,3,12.5,1\n2,1,0.0,0\n3,7,,1\n4,2,5.25,0\n5,5,22.75,1\n6,4,18.0,1\n7,1,3.5,0\n8,6,9.75,0\n9,2,14.0,1\n10,3,2.25,0\n11,4,6.0,1\n12,5,4.5,0\n",
    "new2.csv": "id,visits,spend\n13,3,\n14,6,20.0\n15,1,1.5\n",
    # new rows for the model fitted on pets.csv: the columns in another order, a city and a species the training rows never held, and an empty city cell
    "newpets.csv": "city,species,age,visits\nOslo,rabbit,2,3\nLyon,axolotl,4,1\n,cat,1,2\n",
}

# What is shown is what is typed; the argv run is shlex.split of it. The order is the order of running.
DEMOS = [
    ("fit", "ml fit --target price --output houses.ml houses.csv"),
    ("predict", "ml predict --model-file houses.ml new.csv"),
    ("eval", "ml eval --model-file houses.ml houses.csv"),
    ("bad", "ml predict --model-file houses.ml listings.csv"),
    # "fixed" is not typed here: it is the argv of the repair of "bad", run as the tool offered it
    ("class_fit", "ml fit --task classification --target bought --output shop.ml shop.csv"),
    ("class_page", "ml predict --model-file shop.ml shop.csv --limit 2 --from 1"),
    ("class_eval", "ml eval --model-file shop.ml shop.csv"),
    ("float_fit", "ml fit --target reading --output sensor.ml sensor.csv"),
    ("float_eval", "ml eval --model-file sensor.ml sensor.csv"),
    ("exists", "ml fit --target price --output houses.ml houses.csv"),
    ("dry", "ml fit --dry-run --overwrite --target price --output houses.ml houses.csv"),
    ("damaged", "ml eval --model-file damaged.ml houses.csv"),
    ("other_model", "ml fit --model knn --target price --output knn.ml houses.csv"),
    ("typo", "ml fit --target prize --output typo.ml houses.csv"),
    ("big_fit", "ml fit --model linear --float area,price --drop id --l2 0.1 --target price --output houses-big.ml houses.csv"),
    ("big", "ml predict --model-file houses-big.ml --drop id huge.csv"),
    # the models that go through the pipeline (docs/wiring.md): a refusal with its repair first, then the plan written out
    ("p_bad", "ml fit --model logistic --target bought --output shop2.ml shop2.csv"),
    ("p_fit", "ml fit --model logistic --overwrite --step 'impute column=spend strategy=mean;scale column=visits,spend kind=standard' --iterations 50 --l2 0.1 --target bought --output shop2.ml shop2.csv"),
    ("p_predict", "ml predict --model-file shop2.ml --proba new2.csv"),
    ("p_eval", "ml eval --model-file shop2.ml shop2.csv"),
    ("p_split", "ml fit --model linear --l2 0.1 --split train=0.8 --seed 3 --target price --output houses-lin.ml houses.csv"),
    # text columns declared with --cat (docs/categorical.md, docs/categorical-encoders.md, docs/stringtable.md) on docs/examples/data/pets.csv
    ("c_bad", "ml fit --model logistic --cat species,city --target adopted --output pets.ml pets.csv"),
    ("c_fit", "ml fit --model logistic --cat species,city --step 'encode column=species kind=onehot min-count=4;freq column=city missing=zero;scale column=age,visits kind=standard' --drop id --iterations 50 --l2 0.1 --target adopted --output pets.ml pets.csv"),
    ("c_predict", "ml predict --model-file pets.ml --drop id --proba --limit 3 pets.csv"),
    ("c_new", "ml predict --model-file pets.ml --proba newpets.csv"),
    ("c_eval", "ml eval --model-file pets.ml --drop id pets.csv"),
    ("c_hash", "ml fit --model logistic --cat species,city --step 'hash column=species buckets=8 missing=zero;encode column=city kind=label missing=zero;scale column=age,visits kind=standard' --drop id --split train=0.75 --seed 3 --iterations 50 --l2 0.1 --target adopted --output pets-hash.ml pets.csv"),
    ("c_hash_new", "ml predict --model-file pets-hash.ml --proba newpets.csv"),
    # standard input, describe and ml-predict (docs/serving.md); the last field of a demo is the file given on standard input
    ("s_stdin", "ml predict --model-file houses.ml -", "new.csv"),
    ("s_empty", "ml predict --model-file houses.ml", "empty.csv"),
    ("s_describe", "ml describe --model-file pets.ml --limit 2"),
    ("s_mp", "ml-predict predict --model-file pets.ml --proba -", "newpets.csv"),
    ("s_mp_fit", "ml-predict eval --model-file houses.ml houses.csv"),
    ("s_mp_flag", "ml-predict predict --model-file houses.ml --log run.ndjson new.csv"),
    # k-means (docs/wiring-kmeans.md) on docs/examples/data/iris_train.csv and iris_test.csv: no target, the label column dropped, the features standardised
    ("k_target", "ml fit --model kmeans --k 3 --target virginica --output iris-km.ml iris_train.csv"),
    ("k_nok", "ml fit --model kmeans --drop virginica --output iris-km.ml iris_train.csv"),
    ("k_fit", "ml fit --model kmeans --k 3 --seed 1 --drop virginica --step 'scale column=* kind=standard' --output iris-km.ml iris_train.csv"),
    ("k_predict", "ml predict --model-file iris-km.ml --distance --drop virginica --limit 5 iris_test.csv"),
    ("k_eval", "ml eval --model-file iris-km.ml --drop virginica iris_test.csv"),
    ("k_text", "ml --format text eval --model-file iris-km.ml --drop virginica iris_train.csv"),
    # split strategies and cross-validation (docs/splits.md)
    ("sp_strat", "ml fit --model tree --task classification --max-depth 4 --target benign --split train=0.8 --stratify --seed 7 --output bc.ml breast_cancer_train.csv"),
    ("sp_time", "ml fit --model tree --task classification --max-depth 3 --target returning --drop customer,visit_date --split train=0.75 --split-by time:visit_date --output visits-time.ml visits.csv"),
    ("sp_group", "ml fit --model tree --task classification --max-depth 3 --target returning --drop customer,visit_date --split train=0.75 --split-group customer --seed 2 --output visits-group.ml visits.csv"),
    ("sp_combo", "ml fit --model tree --task classification --target returning --drop customer,visit_date --split train=0.75 --stratify --split-by time:visit_date --output x.ml visits.csv"),
    ("sp_cv_iris", "ml --format text cv --model logistic --task classification --step 'scale column=* kind=standard' --l2 0.1 --target virginica --folds 5 --seed 1 --stratify iris_train.csv"),
    ("sp_cv_diab", "ml --format text cv --model linear --step 'scale column=* kind=standard' --target progression --folds 5 --seed 1 diabetes_train.csv"),
    ("sp_cv_time", "ml cv --model tree --task classification --max-depth 3 --target returning --drop customer,visit_date --split-by time:visit_date --folds 4 visits.csv"),
    ("sp_cv_group", "ml cv --model tree --task classification --max-depth 3 --target returning --drop customer,visit_date --split-group customer --folds 4 --seed 5 --emit-folds visits.csv"),
    ("sp_bad", "ml cv --model linear --step 'scale column=* kind=standard' --target progression --stratify diabetes_train.csv"),
]
# demos shown as the members of the JSON line that matter, since the line is long
# demos whose answer is shown as the members of `data` that matter, since the document is long
SUMMARY_DATA = {"s_describe": ["model", "task", "request", "classes", "output"]}
# demos whose answer is shown as members found by a path into `data`: the last name of a path is the key shown
PICK_DATA = {
    "sp_strat": [("summary", "split"), ("summary", "holdout")],
    "sp_time": [("summary", "split"), ("summary", "holdout")],
    "sp_group": [("summary", "split"), ("summary", "holdout")],
    "sp_cv_time": [("split",), ("strategy",), ("summary",)],
    "sp_cv_group": [("split",), ("strategy",), ("assignment",)],
}
SUMMARY = {
    "bad": ["rule", "hint", "repair"],
    "exists": ["rule", "hint", "detail"],
    "damaged": ["rule", "hint"],
    "other_model": ["rule", "detail"],
    "typo": ["rule", "hint", "detail"],
    "big": ["rule", "hint", "detail"],
    "p_bad": ["rule", "hint", "repair"],
    "c_bad": ["rule", "hint", "detail"],
    "s_empty": ["rule", "hint"],
    "s_mp_fit": ["rule", "hint", "detail"],
    "s_mp_flag": ["rule", "hint", "detail"],
    "k_target": ["rule", "hint", "detail"],
    "k_nok": ["rule", "hint", "detail"],
    "sp_combo": ["rule", "hint", "repair", "detail"],
    "sp_bad": ["rule", "hint", "repair"],
}
# task name -> the demos shown for it
TASKS = {
    "fit": ["fit", "predict", "eval"],
    "class": ["class_fit", "class_page", "class_eval"],
    "float": ["float_fit", "float_eval"],
    "dry": ["exists", "dry"],
    "damaged": ["damaged"],
    "honest": ["other_model", "typo", "big"],
    "split": ["p_split"],
}
QS = ["fit", "predict", "eval", "bad"]
# a demo whose refusal carries a repair that is run as offered: the key of the repair's own result
REPAIRED = {"bad": "fixed", "p_bad": "p_fixed", "sp_bad": "sp_fixed"}
FORBIDDEN = {"ffi", "net_out", "net_in", "clock"}


def e(s):
    return html.escape(s, quote=False)


# --- running things ----------------------------------------------------------


def run_words(words, binary, cwd, stdin=None):
    """The command, run as `words` say (argv[0] is what the tool prints in a repair) with `binary`, or its sibling `ml-predict` for that command;
    `stdin` is the name of a file in `cwd` given on standard input (none: an empty one)."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"}
    exe = Path(binary).parent / words[0] if words and words[0] in ("ml-predict", "ml-serve") else binary
    data = (Path(cwd) / stdin).read_bytes() if stdin else b""
    r = subprocess.run(words, executable=str(exe), cwd=cwd, env=env, input=data,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return r.stdout.decode("utf-8"), r.returncode


# The HTTP flow (docs/serving-http.md): `ml-serve` is started on pets.ml, written by the demo c_fit, on a free port; the requests are sent over a socket and the answer is printed as curl -i
# prints it. The text shows port 8080 where the run used another (the one thing that differs). A person typed the commands below; the bytes after them are the server's.
HTTP_PORT_SHOWN = 8080
HTTP_ONE = '{"age":3,"species":"cat","city":"Turin","visits":4}'
HTTP_BATCH = '{"rows":[{"age":3,"species":"cat","city":"Turin","visits":4},{"age":1,"species":"dog","city":"Lyon","visits":7}]}'
HTTP_BAD = '{"age":"two","species":"axolotl","city":"Lyon","visits":4,"zip":1}'
JSON_POST = "curl -s -i -XPOST -H 'Content-Type: application/json' -d '%s' localhost:8080%s"
HTTP_DEMOS = [
    ("h_health", "curl -s localhost:8080/health", "GET", "/health", None, "body"),
    ("h_one", JSON_POST % (HTTP_ONE, "/predict"), "POST", "/predict", HTTP_ONE, "all"),
    ("h_batch", "curl -s -XPOST -H 'Content-Type: application/json' -d '%s' localhost:8080/predict/batch" % HTTP_BATCH, "POST", "/predict/batch", HTTP_BATCH, "body"),
    ("h_bad", JSON_POST % (HTTP_BAD, "/predict"), "POST", "/predict", HTTP_BAD, "all"),
    ("h_model", "curl -s localhost:8080/model", "GET", "/model", None, "model"),
    ("h_openapi", "curl -s localhost:8080/openapi.json", "GET", "/openapi.json", None, "openapi"),
]


HTTP_TOKEN = "demo-token"
HTTP_ALL_PORT_SHOWN = 8081


def http_request(port, method, path, body, token=None):
    """The response as curl -i prints it (status line, headers, a blank line, the body), with the line ends of a terminal: the head, and the body of the length it says."""
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        head = "%s %s HTTP/1.1\r\nHost: localhost:%d\r\nAccept: */*\r\n" % (method, path, port)
        if token is not None:
            head += "Authorization: Bearer %s\r\n" % token
        if body is not None:
            head += "Content-Type: application/json\r\nContent-Length: %d\r\n" % len(body.encode())
        s.sendall((head + "\r\n").encode() + (body or "").encode())
        got = b""
        while b"\r\n\r\n" not in got:
            d = s.recv(65536)
            if not d:
                break
            got += d
        raw_head, _, raw_body = got.partition(b"\r\n\r\n")
        length = 0
        for line in raw_head.decode().split("\r\n")[1:]:
            k, _, v = line.partition(":")
            if k.lower() == "content-length":
                length = int(v.strip())
        while len(raw_body) < length:
            d = s.recv(65536)
            if not d:
                break
            raw_body += d
    finally:
        s.close()
    return raw_head.decode().replace("\r\n", "\n"), raw_body.decode()


def run_http_demos(binary, tmp):
    serve = Path(binary).parent / "ml-serve"
    if not serve.exists():
        raise Drift("%s does not exist: run `cancho build --bin ml-serve`" % serve)
    if not (Path(tmp) / "pets.ml").exists():
        raise Drift("the HTTP demos need pets.ml, which the demo c_fit writes")
    s0 = socket.socket()
    s0.bind(("127.0.0.1", 0))
    port = s0.getsockname()[1]
    s0.close()
    log = open(Path(tmp) / "serve.log", "w+")
    proc = subprocess.Popen([str(serve), "--model-file", "pets.ml", "--port", str(port)], cwd=tmp, stdout=subprocess.PIPE, stderr=log, stdin=subprocess.DEVNULL,
                            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"})
    out = {}
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise Drift("ml-serve did not listen")
        log.seek(0)
        started = log.read().strip().replace(str(port), str(HTTP_PORT_SHOWN))
        out["h_start"] = ("ml-serve --model-file pets.ml --port %d &" % HTTP_PORT_SHOWN, started, 0)
        for key, typed, method, path, body, shape in HTTP_DEMOS:
            head, text = http_request(port, method, path, body)
            status = int(head.split()[1])
            if shape == "all":
                shown = head + "\n\n" + text
            elif shape == "model":
                d = json.loads(text)
                shown = json.dumps({k: d[k] for k in ("model", "task", "request", "classes", "output")}, ensure_ascii=False, separators=(",", ":"))
            elif shape == "openapi":
                d = json.loads(text)
                shown = json.dumps({"openapi": d["openapi"], "paths": sorted(d["paths"]), "Record": d["components"]["schemas"]["Record"]}, ensure_ascii=False, separators=(",", ":"))
            else:
                shown = text
            if status not in (200, 422):
                raise Drift("the HTTP demo %s answered %d" % (key, status))
            out[key] = (typed, shown, status)
        out.update(run_all_demo(serve, tmp))
    finally:
        proc.terminate()
        proc.wait(5)
        if proc.stdout:
            proc.stdout.close()
        log.close()
    return out


def run_all_demo(serve, tmp):
    """The form for a network: `--bind all --token-file`, started on a free port (the text shows 8081); a request without the token is a 401, with it the answer."""
    (Path(tmp) / "token.txt").write_text(HTTP_TOKEN + "\n")
    s0 = socket.socket()
    s0.bind(("127.0.0.1", 0))
    port = s0.getsockname()[1]
    s0.close()
    log = open(Path(tmp) / "serve-all.log", "w+")
    proc = subprocess.Popen([str(serve), "--model-file", "pets.ml", "--bind", "all", "--token-file", "token.txt", "--port", str(port)], cwd=tmp, stdout=subprocess.PIPE, stderr=log,
                            stdin=subprocess.DEVNULL, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"})
    out = {}
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise Drift("ml-serve --bind all did not listen")
        log.seek(0)
        out["h_all_start"] = ("ml-serve --model-file pets.ml --bind all --token-file token.txt --port %d &" % HTTP_ALL_PORT_SHOWN, log.read().strip().replace(str(port), str(HTTP_ALL_PORT_SHOWN)), 0)
        head, text = http_request(port, "GET", "/model", None)
        if int(head.split()[1]) != 401:
            raise Drift("the demo without the token answered %s, not 401" % head.split()[1])
        out["h_all_denied"] = ("curl -s -i localhost:%d/model" % HTTP_ALL_PORT_SHOWN, head + "\n\n" + text, 0)
        head, text = http_request(port, "GET", "/model", None, token=HTTP_TOKEN)
        if int(head.split()[1]) != 200:
            raise Drift("the demo with the token answered %s, not 200" % head.split()[1])
        d = json.loads(text)
        out["h_all_ok"] = ("curl -s -H 'Authorization: Bearer %s' localhost:%d/model" % (HTTP_TOKEN, HTTP_ALL_PORT_SHOWN),
                           json.dumps({k: d[k] for k in ("model", "task", "classes")}, ensure_ascii=False, separators=(",", ":")), 0)
    finally:
        proc.terminate()
        proc.wait(5)
        if proc.stdout:
            proc.stdout.close()
        log.close()
    return out


def make_context(binary, tmp):
    """Write the files, run every demo once, in order. The results are kept."""
    for name, text in FILES.items():
        (Path(tmp) / name).write_text(text)
    # the categorical example is a committed file, as the worked examples' CSVs are
    (Path(tmp) / "pets.csv").write_bytes((ROOT / "docs" / "examples" / "data" / "pets.csv").read_bytes())
    # the k-means example: the iris files of the worked examples
    for part in ("train", "test"):
        (Path(tmp) / ("iris_%s.csv" % part)).write_bytes((ROOT / "docs" / "examples" / "data" / ("iris_%s.csv" % part)).read_bytes())
    # the split examples: the committed files of docs/examples/data
    for name in ("breast_cancer_train.csv", "diabetes_train.csv", "visits.csv"):
        (Path(tmp) / name).write_bytes((ROOT / "docs" / "examples" / "data" / name).read_bytes())
    results = {}

    def run(words, stdin=None):
        return run_words(words, binary, tmp, stdin)

    for key, typed, *given in DEMOS:
        if key == "damaged":
            # a copy of houses.ml with one byte changed, inside the part the SHA-256 trailer covers
            b = bytearray((Path(tmp) / "houses.ml").read_bytes())
            b[-40] ^= 1
            (Path(tmp) / "damaged.ml").write_bytes(bytes(b))
        out, status = run(shlex.split(typed), given[0] if given else None)
        results[key] = (typed + (" < %s" % given[0] if given else ""), out, status)
        if key == "c_bad" and (Path(tmp) / "pets.ml").exists():
            raise Drift("the demo `c_bad` wrote pets.ml: a declared column that no step reads is refused")
        if key in ("c_fit", "c_hash") and not (Path(tmp) / ("pets.ml" if key == "c_fit" else "pets-hash.ml")).exists():
            raise Drift("the demo `%s` wrote no model file" % key)
        if key in REPAIRED:
            doc = json.loads(out)
            repair = doc["error"]["repair"]
            if repair is None or repair["kind"] != "retry":
                raise Drift("the demo `%s` no longer carries a retry repair" % key)
            argv = repair["argv"]
            out2, status2 = run(argv)
            if status2 != 0:
                raise Drift("the repair of the demo `%s` does not run" % key)
            results[REPAIRED[key]] = (shlex.join(argv), out2, status2)
    run_track_demos(run, tmp, results)
    http_results = run_http_demos(binary, tmp)
    ref = sx.reference()
    examples = sx.run_examples(run, tmp, ref)
    r = subprocess.run(["ml", "introspect"], executable=str(binary), cwd=tmp, stdout=subprocess.PIPE, check=True,
                       env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"})
    return {"results": results, "http": http_results, "introspect": json.loads(r.stdout), "run": run, "examples": examples, "ref": ref}


# Run tracking and the skew checks (docs/tracking.md) on docs/examples/data/iris_train.csv and iris_test.csv. The ids are the hashes of the inputs, so they are the same on every run; the
# commands that name them are typed from the answers of the fits.
TRACK_FITS = [
    ("t_fit", "ml fit --model logistic --target virginica --iterations 25 --l2 0 --track runs --output iris-lr.ml iris_train.csv"),
    ("t_fit2", "ml fit --model forest --task classification --trees 20 --seed 2 --target virginica --track runs --output iris-rf.ml iris_train.csv"),
]
SUMMARY_DATA.update({
    "t_fit": ["path", "bytes", "sha256", "run"], "t_fit2": ["path", "bytes", "sha256", "run"], "t_list": ["count", "runs"], "t_compare": ["model_equal", "inputs"],
    "t_eval_train": ["rows", "on", "metrics", "out_of_range"], "t_eval_test": ["rows", "on", "metrics", "out_of_range"],
    "t_predict": ["row_count", "unknown_categories", "out_of_range", "ignored_columns"], "t_verify": ["verified", "model_sha256", "threads"],
    "t_verify8": ["verified", "model_sha256", "threads"],
})
SUMMARY.update({"t_bound": ["rule", "hint", "detail"], "t_conflict": ["rule", "detail"]})


def run_track_demos(run, tmp, results):
    (Path(tmp) / "runs").mkdir()
    ids = []
    for key, typed in TRACK_FITS:
        out, status = run(shlex.split(typed))
        if status != 0:
            raise Drift("the demo %s fails" % key)
        results[key] = (typed, out, status)
        ids.append(json.loads(out)["data"]["run"]["id"])
    a, b = ids
    for key, typed in [
        ("t_list", "ml runs list runs"),
        ("t_show", "ml runs show runs %s" % a),
        ("t_compare", "ml runs compare runs %s %s" % (a, b)),
        ("t_verify", "ml runs verify runs %s" % a),
        ("t_verify8", "ml runs verify --threads 8 runs %s" % b),
        ("t_eval_train", "ml eval --model-file iris-lr.ml --track runs iris_train.csv"),
        ("t_eval_test", "ml eval --model-file iris-lr.ml --track runs iris_test.csv"),
        ("t_predict", "ml predict --model-file iris-lr.ml --limit 2 iris_test.csv"),
        ("t_bound", "ml predict --model-file iris-lr.ml --max-out-of-range 0 iris_test.csv"),
    ]:
        out, status = run(shlex.split(typed))
        results[key] = (typed, out, status)


def track_items(ctx):
    """Track runs and check for skew: fit with --track, list, show, compare, verify; eval says what it scored; predict counts the cells outside the training range and can refuse."""
    show = ctx["results"]["t_show"]
    return [("# Every fit can write a run record: canonical JSON, named by the hash of the inputs that determine the model (not the thread count), no timestamp, no payload", None, 0, None),
            item(ctx, "t_fit"),
            item(ctx, "t_fit2"),
            ("# The directory lists the runs; a run is shown whole, and two are compared by the inputs that differ and the metrics side by side", None, 0, None),
            item(ctx, "t_list"),
            (show[0], cut(show[1].strip(), 420), show[2], None),
            item(ctx, "t_compare"),
            ("# A run is replayable: verify refits in memory and compares the SHA-256 of the model file, at any thread count", None, 0, None),
            item(ctx, "t_verify"),
            item(ctx, "t_verify8"),
            ("# eval says whether it scored the file the model was fitted on, and counts the cells outside the range of the training rows", None, 0, None),
            item(ctx, "t_eval_train"),
            item(ctx, "t_eval_test"),
            ("# predict counts them too, and names the target column it ignored", None, 0, None),
            item(ctx, "t_predict"),
            ("# A bound on the rows outside the range refuses, naming the first column and row", None, 0, None),
            item(ctx, "t_bound")]


def item(ctx, key):
    """(typed command, what is printed, status, note). A refusal is shown as the
    members of its JSON line that matter."""
    typed, out, status = ctx["results"][key]
    if key in SUMMARY_DATA:
        doc = json.loads(out)
        if not doc["ok"]:
            raise Drift("the demo %s refuses" % key)
        out = json.dumps({k: doc["data"][k] for k in SUMMARY_DATA[key]}, ensure_ascii=False)
        return typed, out, status, None
    if key in PICK_DATA:
        doc = json.loads(out)
        if not doc["ok"]:
            raise Drift("the demo %s refuses" % key)
        picked = {}
        for path in PICK_DATA[key]:
            v = doc["data"]
            for step in path:
                v = v[step]
            picked[path[-1]] = v
        return typed, json.dumps(picked, ensure_ascii=False, separators=(",", ":")), status, None
    if key in SUMMARY:
        doc = json.loads(out)
        if doc["ok"] or status == 0:
            raise Drift("the demo %s no longer refuses" % key)
        err = doc["error"]
        out = json.dumps({k: err[k] for k in SUMMARY[key]}, ensure_ascii=False)
        return typed, out, status, "the %s of the JSON line it prints" % ", ".join(SUMMARY[key])
    if status != 0 and key not in SUMMARY:
        if json.loads(out).get("dry_run"):
            return typed, out, status, "a dry run: nothing was written"
        raise Drift("the demo %s fails with status %d" % (key, status))
    return typed, out, status, None


def flow_items(ctx):
    """Fit, predict, eval, be refused with a repair, run the repair."""
    return [("# Fit the training mean of price. area has a decimal point: it is a float column, read exactly", None, 0, None),
            item(ctx, "fit"),
            ("# Predict for two new rows, which have no price", None, 0, None),
            item(ctx, "predict"),
            ("# Score the model on the file it was fitted on", None, 0, None),
            item(ctx, "eval"),
            ("# New rows with a column of text are refused", None, 0, None),
            item(ctx, "bad"),
            ("# The tool offered the corrected command. Run it:", None, 0, None),
            item(ctx, "fixed")]


def pflow_items(ctx):
    """The pipeline flow: a float CSV with an empty cell, the refusal and its repair, then impute + standardise + logistic, predict with probabilities, eval."""
    return [("# A float CSV with an empty cell: the plan must say what to do with it. The refusal offers a repair", None, 0, None),
            item(ctx, "p_bad"),
            ("# The repair works, but a plan can say more: impute, standardise, then a logistic regression (50 Newton iterations, ridge 0.1)", None, 0, None),
            item(ctx, "p_fit"),
            ("# Predict with probabilities for new rows (the empty cell is filled with the mean STORED in the model)", None, 0, None),
            item(ctx, "p_predict"),
            ("# Score it: accuracy, precision, recall, F1, ROC-AUC, log-loss", None, 0, None),
            item(ctx, "p_eval")]


def catflow_items(ctx):
    """The categorical flow: a refusal, a fit that writes the model file with the strings of its categories, predict and eval from the file alone, new rows with unseen
    categories in another column order, and a hashed column that stores no string."""
    return [("# pets.csv: 76 rows; species and city are text. Declared with --cat but read by no step, they are refused", None, 0, None),
            item(ctx, "c_bad"),
            ("# Encoded, the fit writes the model file. species is one-hot (the categories seen fewer than 4 times share __other__), city is its frequency; the file holds the training strings", None, 0, None),
            item(ctx, "c_fit"),
            ("# predict reads the model file and nothing else; the text columns are found by name", None, 0, None),
            item(ctx, "c_predict"),
            ("# New rows: the columns in another order, a city and a species the training rows never held (counted, not guessed), an empty city cell that missing=zero accepts", None, 0, None),
            item(ctx, "c_new"),
            ("# eval scores the file it was fitted on", None, 0, None),
            item(ctx, "c_eval"),
            ("# species hashed to 8 columns and city as a label: the file holds no string of species, and an unseen species lands in the bucket it hashes to", None, 0, None),
            item(ctx, "c_hash"),
            item(ctx, "c_hash_new")]


def sflow_items(ctx):
    """Standard input, describe and ml-predict."""
    return [("# predict reads the CSV from standard input for the operand - (or no operand): the answer is the file's answer", None, 0, None),
            item(ctx, "s_stdin"),
            ("# a forgotten operand with nothing on standard input is the refusal of an empty file", None, 0, None),
            item(ctx, "s_empty"),
            ("# describe: what the model file requires of a caller. Excerpt: the model, the columns of a request, the classes, the columns of an answer", None, 0, None),
            item(ctx, "s_describe"),
            ("# ml-predict predicts and describes, from the same sources, with a row that has no write and no thread", None, 0, None),
            item(ctx, "s_mp"),
            item(ctx, "s_mp_fit"),
            item(ctx, "s_mp_flag")]
def kmflow_items(ctx):
    """k-means: no target, a refusal that says so, a fit without k refused, the iris features standardised and clustered, the cluster and the distance of each test row, the metrics."""
    return [("# k-means has no target: it clusters the rows by their features. The label column of iris is not given as one", None, 0, None),
            item(ctx, "k_target"),
            ("# k is required", None, 0, None),
            item(ctx, "k_nok"),
            ("# Drop the label, standardise the four measurements, fit three clusters (seed 1): the answer holds the inertia, the rows of each cluster and the mean distance of each", None, 0, None),
            item(ctx, "k_fit"),
            ("# The cluster of five test rows and the squared distance of each to its centre", None, 0, None),
            item(ctx, "k_predict"),
            ("# The inertia of the 25 test rows, the rows of each cluster and the mean distance of each", None, 0, None),
            item(ctx, "k_eval"),
            ("# The same, as text, on the training rows", None, 0, None),
            item(ctx, "k_text")]


def splitflow_items(ctx):
    """Splits and cross-validation: a stratified holdout, a time split, a group split, a refused combination, k-fold cv of two models with the metrics of every fold and their mean and
    standard deviation, a time cv, a group cv with the folds as data, and a refusal with its repair."""
    return [("# A holdout that keeps the class proportions: 426 rows, 20 percent held out, per class the floor or the ceiling of its share", None, 0, None),
            item(ctx, "sp_strat"),
            ("# visits.csv: 12 customers with 10 visits each, an ISO date, a basket, the item count and whether the customer came back. The newest quarter of the visits is held out (the date column is dropped as a feature, not as a rule)", None, 0, None),
            item(ctx, "sp_time"),
            ("# No customer on both sides: whole customers are held out", None, 0, None),
            item(ctx, "sp_group"),
            ("# Two rules for one split are refused, not blended", None, 0, None),
            item(ctx, "sp_combo"),
            ("# Five-fold cross-validation, stratified: every step is fitted on the four training folds of each fold alone. Per-fold metrics, then the mean and the sample standard deviation", None, 0, None),
            item(ctx, "sp_cv_iris"),
            item(ctx, "sp_cv_diab"),
            ("# An expanding window: fold i trains on everything before block i + 1 and tests on that block", None, 0, None),
            item(ctx, "sp_cv_time"),
            ("# Whole customers per fold, and the assignment of every row (file order) as data, so that a run can be replayed", None, 0, None),
            item(ctx, "sp_cv_group"),
            ("# A regression has no classes to keep in proportion. The refusal offers the corrected command", None, 0, None),
            item(ctx, "sp_bad")]


def hflow_items(ctx):
    """Serve a model over HTTP: start, a record, a batch, a refusal with every error, what the model requires, and the document."""
    h = ctx["http"]

    def one(key, width=200):
        typed, out, status = h[key]
        return (typed, "\n".join(cut(l, width) for l in out.splitlines()), 0, None)

    return [("# ml-serve reads the model file once and answers over HTTP, one request at a time (the demo runs on a free port; the text shows 8080). With no flag about the address it listens on 127.0.0.1 and needs no token", None, 0, None),
            one("h_start"),
            ("# liveness, and the SHA-256 of the model file being served", None, 0, None),
            one("h_health"),
            ("# one record: a member for every column of the model. The head carries the digest too", None, 0, None),
            one("h_one"),
            ("# a batch: a row of the answer for each record, in order", None, 0, None),
            one("h_batch"),
            ("# a refusal: every error at once, each with its JSON Pointer, as application/problem+json with the rule", None, 0, None),
            one("h_bad", 260),
            ("# what the model requires of a caller (excerpt), and the OpenAPI 3.1 document made from the same columns (excerpt)", None, 0, None),
            one("h_model"),
            one("h_openapi"),
            ("# the form for a network: every address of the host, and a token (a request without it is a 401; the token is the only line of the file)", None, 0, None),
            one("h_all_start"),
            one("h_all_denied"),
            one("h_all_ok")]


def render(items, html_mode):
    out = []
    for display, text, status, note in items:
        if text is None:
            out.append(('<span class="c">%s</span>' % e(display)) if html_mode else display)
            continue
        if html_mode:
            out.append('<span class="c">$</span> ' + e(display))
            out.append(e(text.rstrip("\n")))
            if status != 0:
                out.append('<span class="r"># exit status %d%s</span>' % (status, "; " + e(note) if note else ""))
        else:
            out.append("$ " + display)
            out.append(text.rstrip("\n"))
            if status != 0:
                out.append("# exit status %d%s" % (status, "; " + note if note else ""))
    return "\n".join(out)


def md_block(items):
    return "```console\n" + render(items, False) + "\n```"


def html_pre(items):
    return '<pre class="code" tabindex="0">' + render(items, True) + "</pre>"


def task_items(ctx, task):
    return [item(ctx, k) for k in TASKS[task]]


def hero(ctx):
    return ('<div class="term" role="img" aria-label="One command on a small file and what it prints">'
            '<div class="bar"><i></i><i></i><i></i></div>\n<pre>' + render([item(ctx, "eval")], True) + "</pre></div>")


def qs_items(ctx):
    return [item(ctx, k) for k in QS]


def predict_authority(ctx):
    manifest = json.loads((ROOT / "manifests" / "ml-predict.authority.json").read_text())
    out, status = ctx["run"](["ml-predict", "introspect"])
    if status != 0 or manifest != json.loads(out)["authority"]:
        raise Drift("manifests/ml-predict.authority.json is not what `ml-predict introspect` says: run scripts/manifest.py")
    effects = {lab["name"] for lab in manifest["labels"]}
    if not manifest["bounded"] or manifest["foreign_symbols"] or effects & (FORBIDDEN | {"conc", "file_write", "dir_write", "fs_write"}):
        raise Drift("the authority of ml-predict has a write, a thread, a forbidden label or a foreign symbol, or is unbounded")
    return ", ".join(l["name"] + ('("%s")' % l["argument"] if l["argument"] is not None else "") for l in manifest["labels"])


def serve_authority(ctx):
    manifest = json.loads((ROOT / "manifests" / "ml-serve.authority.json").read_text())
    out, status = ctx["run"](["ml-serve", "introspect"])
    if status != 0 or manifest != json.loads(out)["authority"]:
        raise Drift("manifests/ml-serve.authority.json is not what `ml-serve introspect` says: run scripts/manifest.py")
    effects = {lab["name"] for lab in manifest["labels"]}
    if not manifest["bounded"] or manifest["foreign_symbols"] or effects & {"ffi", "net_out", "conc", "exec", "env", "io_read", "file_write", "dir_write", "fs_write"}:
        raise Drift("the authority of ml-serve has a write, a thread, an exec, a foreign symbol or an outbound connection, or is unbounded")
    return ", ".join(l["name"] + ('("%s")' % l["argument"] if l["argument"] is not None else "") for l in manifest["labels"])


def authority_row(ctx):
    manifest = json.loads((ROOT / "manifests" / "ml.authority.json").read_text())
    if manifest != ctx["introspect"]["authority"]:
        raise Drift("manifests/ml.authority.json is not what `ml introspect` says: run scripts/manifest.py")
    names = []
    for lab in manifest["labels"]:
        names.append(lab["name"] + ('("%s")' % lab["argument"] if lab["argument"] is not None else ""))
    effects = {lab["name"] for lab in manifest["labels"]}
    if not manifest["bounded"] or manifest["foreign_symbols"] or effects & FORBIDDEN:
        raise Drift("the authority has a forbidden label, a foreign symbol, or is unbounded")
    return manifest


def authority_md(ctx):
    m = authority_row(ctx)
    names = ", ".join(l["name"] + ('("%s")' % l["argument"] if l["argument"] is not None else "") for l in m["labels"])
    return ("```console\n# manifests/ml.authority.json, manifests/ml-predict.authority.json and manifests/ml-serve.authority.json\nml          %s\nml-predict  %s\nml-serve    %s\n"
            "# bounded: %s; foreign symbols: none; ffi, net_out: absent from all three; net_in, clock: ml-serve only\n```" % (names, predict_authority(ctx), serve_authority(ctx), str(m["bounded"]).lower()))


def authority(ctx):
    m = authority_row(ctx)
    names = ", ".join(l["name"] + ('("%s")' % l["argument"] if l["argument"] is not None else "") for l in m["labels"])
    return ('<pre class="code" tabindex="0"><span class="c"># manifests/ml.authority.json, manifests/ml-predict.authority.json and manifests/ml-serve.authority.json</span>\n'
            'ml          %s\nml-predict  %s\nml-serve    %s\n<span class="c"># bounded: %s; foreign symbols: none; ffi, net_out: absent from all three; net_in, clock: ml-serve only</span></pre>'
            % (e(names), e(predict_authority(ctx)), e(serve_authority(ctx)), str(m["bounded"]).lower()))


def limits(ctx):
    rows = "".join("<tr><th><code>--%s</code></th><td class=\"num\">%s</td><td class=\"num\">%s</td></tr>"
                   % (e(l["name"]), format(l["default"], ","), format(l["ceiling"], ","))
                   for l in ctx["introspect"]["limits"])
    return ('<div class="fitwrap"><table class="fit"><thead><tr><th>limit</th><th>default</th><th>ceiling</th></tr></thead><tbody>'
            + rows + "</tbody></table></div>")


def rules_html(ctx):
    rs = ctx["introspect"]["rules"]

    def short(s):
        return s.split(" Hint:")[0]
    rows = "".join("<tr><th><code>%s</code></th><td class=\"num\">%d</td><td>%s</td><td>%s</td></tr>"
                   % (e(r["rule"]), r["exit"], e(r["repairable"]), e(short(r["summary"]))) for r in rs)
    return ('<div class="fitwrap"><table class="fit"><thead><tr><th>rule</th><th>exit</th><th>a repair?</th><th>what it refuses</th></tr></thead><tbody>%s</tbody></table></div>'
            % rows)


def nrules(ctx):
    return str(len(ctx["introspect"]["rules"]))


def cut(s, n):
    return s if len(s) <= n else s[: n - 1] + "…"


SAMPLE_RULES = ["args.unknown-flag", "ml.input.non-numeric", "ml.schema.mismatch", "ml.model.checksum", "conflict.exists", "limit.line-too-long"]
SAMPLE_FLAGS = ["--root", "--target", "--output", "--model-file", "--drop", "--dry-run"]


def agentio_lines(ctx):
    d = ctx["introspect"]
    flags = {f["name"]: f for f in d["flags"]}
    rules = {r["rule"]: r for r in d["rules"]}
    lim = {l["name"]: l for l in d["limits"]}
    try:
        L = ["$ ml introspect          # one JSON document; excerpts, long lines cut", "",
             "tool        %s %s   (compiler %s)" % (d["tool"], d["version"], d["compiler"][:8]),
             "flags       %d, for example" % len(d["flags"])]
        for n in SAMPLE_FLAGS:
            f = flags[n]
            L.append("  %-13s %-16s %s" % (n, f["kind"], cut(f["help"], 52)))
        L.append("limits      %d, for example" % len(d["limits"]))
        for n in ("max-rows", "max-columns", "max-line-bytes", "max-bytes"):
            L.append("  --%-16s default %-12s ceiling %s" % (n, format(lim[n]["default"], ","), format(lim[n]["ceiling"], ",")))
        L.append("rules       %d, for example" % len(d["rules"]))
        for n in SAMPLE_RULES:
            r = rules[n]
            L.append("  %-24s exit %d   a repair: %s" % (n, r["exit"], r["repairable"]))
        L += ["reads environment: %s   reads clock: %s"
              % (str(d["reads_environment"]).lower(), str(d["reads_clock"]).lower()),
              "authority   " + ", ".join(l["name"] + ('("%s")' % l["argument"] if l["argument"] is not None else "") for l in d["authority"]["labels"]),
              "            bounded: %s   foreign symbols: %d" % (str(d["authority"]["bounded"]).lower(), len(d["authority"]["foreign_symbols"]))]
    except KeyError as exc:
        raise Drift("`ml introspect` no longer lists %s" % exc)
    return "\n".join(L)


def agentio(ctx):
    return '<pre class="code" tabindex="0">' + e(agentio_lines(ctx)) + "</pre>"


def agentio_md(ctx):
    return "```console\n" + agentio_lines(ctx) + "\n```"


def skill(ctx):
    out, status = ctx["run"](["ml", "skill"])
    if status != 0:
        raise Drift("`ml skill` failed")
    lines = out.splitlines()
    start = lines.index("## Output")
    body = lines[start:start + 6]
    return ('<pre class="code" tabindex="0">' + e("$ ml skill          # a guide in Markdown; the front matter and one section, long lines cut\n"
            + "\n".join(cut(l, 100) for l in lines[:4]) + "\n...\n" + "\n".join(cut(l, 100) for l in body)) + "</pre>")


def exit_codes(ctx):
    rows = "".join("| %d | `%s` | %s |\n" % (c["code"], c["name"], c["meaning"]) for c in ctx["introspect"]["exit_codes"])
    return "| exit | `code` | meaning |\n|---:|---|---|\n" + rows.rstrip("\n")


REGIONS = {
    "README.md": dict({"t-" + t: (lambda c, t=t: md_block(task_items(c, t))) for t in TASKS},
                      flow=lambda c: md_block(flow_items(c)), pflow=lambda c: md_block(pflow_items(c)), qs=lambda c: md_block(qs_items(c)),
                      catflow=lambda c: md_block(catflow_items(c)), sflow=lambda c: md_block(sflow_items(c)), hflow=lambda c: md_block(hflow_items(c)), kmflow=lambda c: md_block(kmflow_items(c)), splitflow=lambda c: md_block(splitflow_items(c)), trackflow=lambda c: md_block(track_items(c)),
                      agentio=agentio_md, authority=authority_md, **{"exit-codes": exit_codes}),
    "docs/index.html": dict({"t-" + t: (lambda c, t=t: html_pre(task_items(c, t))) for t in TASKS},
                            hero=hero, flow=lambda c: html_pre(flow_items(c)), pflow=lambda c: html_pre(pflow_items(c)), qs=lambda c: html_pre(qs_items(c)),
                            catflow=lambda c: html_pre(catflow_items(c)), sflow=lambda c: html_pre(sflow_items(c)), hflow=lambda c: html_pre(hflow_items(c)), kmflow=lambda c: html_pre(kmflow_items(c)), splitflow=lambda c: html_pre(splitflow_items(c)), trackflow=lambda c: html_pre(track_items(c)),
                            agentio=agentio, skill=skill, authority=authority, limits=limits,
                            rules=rules_html, nrules=nrules),
}


def example_regions():
    """The regions of docs/examples.html and of the README that come from the worked examples."""
    regions = {"ex-meta": lambda c: sx.meta_html(c["ref"]),
               "ex-runit": lambda c: '<pre class="code" tabindex="0">' + e("\n".join(sx.runit_lines())) + "</pre>"}
    for key, ex in sx.EXAMPLES.items():
        regions["ex-%s-facts" % key] = lambda c, k=key: sx.facts_html(c["ref"], k)
        regions["ex-%s-cmds" % key] = lambda c, k=key: html_pre(sx.cmd_items(c["examples"], k))
        if any("same_as" in r for r in ex["runs"]):
            regions["ex-%s-conv" % key] = lambda c, k=key: sx.conv_html(c["examples"], c["ref"], k)
        for r in ex["runs"]:
            if r.get("ref"):
                regions["ex-%s-table-%s" % (key, r["id"])] = lambda c, k=key, i=r["id"]: sx.table_html(c["examples"], c["ref"], k, i)
                regions["ex-%s-pred-%s" % (key, r["id"])] = lambda c, k=key, i=r["id"]: sx.pred_html(c["examples"], c["ref"], k, i)
                regions["ex-%s-coef-%s" % (key, r["id"])] = lambda c, k=key, i=r["id"]: sx.coef_html(c["examples"], c["ref"], k, i)
    return regions


REGIONS["docs/examples.html"] = example_regions()
REGIONS["README.md"]["ex-flow"] = lambda c: md_block(sx.cmd_items(c["examples"], "iris", only=["fit"]))
REGIONS["docs/examples.md"] = {"ex-results": lambda c: sx.full_md(c["examples"], c["ref"])}
REGIONS["README.md"]["ex-summary"] = lambda c: sx.summary_md(c["examples"], c["ref"])


SITE_BASE = "https://alpibrusl.github.io/cancho-ml/"
# the pages that are meant to be found
PAGES = ["", "examples.html", "evidence.html", "benchmarks.html"]


def sitemap(_ctx=None):
    urls = "".join("  <url><loc>%s%s</loc></url>\n" % (SITE_BASE, p) for p in PAGES)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + urls + "</urlset>\n")


def robots(_ctx=None):
    return "User-agent: *\nAllow: /\n\nSitemap: %ssitemap.xml\n" % SITE_BASE


WHOLE_FILES = {"docs/sitemap.xml": sitemap, "docs/robots.txt": robots}


def splice(text, name, body):
    start = "<!-- gen:%s -->\n" % name
    end = "<!-- /gen:%s -->" % name
    i = text.find(start)
    j = text.find(end, i + 1)
    if i < 0 or j < 0:
        raise Drift("no <!-- gen:%s --> region in the file" % name)
    return text[:i] + start + body + "\n" + text[j:]


def regions_in(text):
    """Every region name that appears in a file, to catch one nobody generates."""
    names = []
    i = 0
    while True:
        i = text.find("<!-- gen:", i)
        if i < 0:
            return names
        j = text.find(" -->", i)
        names.append(text[i + len("<!-- gen:"):j])
        i = j


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", default=str(ROOT / "build" / "ml"))
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 if a file differs")
    args = ap.parse_args()
    binary = Path(args.bin).resolve()
    if not binary.exists():
        sys.exit("site.py: %s does not exist: run `cancho install && cancho build --bin ml` first" % binary)
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_context(binary, tmp)
        for rel, regions in REGIONS.items():
            path = ROOT / rel
            old = path.read_text()
            for name in regions_in(old):
                if name not in regions:
                    raise Drift("%s has a region gen:%s that nothing generates" % (rel, name))
            new = old
            for name, fn in regions.items():
                new = splice(new, name, fn(ctx))
            if new == old:
                continue
            if args.check:
                bad += 1
                sys.stderr.write("site.py: %s differs from what the binary produces; run `python3 scripts/site.py`\n" % rel)
                for k, line in enumerate(difflib.unified_diff(old.splitlines(), new.splitlines(), "committed", "generated", lineterm="", n=0)):
                    if k >= 30:
                        sys.stderr.write("...\n")
                        break
                    sys.stderr.write(line[:200] + "\n")
            else:
                path.write_text(new)
                print("wrote", rel)
    for rel, fn in WHOLE_FILES.items():
        path = ROOT / rel
        new = fn()
        old = path.read_text() if path.exists() else None
        if new == old:
            continue
        if args.check:
            bad += 1
            sys.stderr.write("site.py: %s differs from what the page list produces; run `python3 scripts/site.py`\n" % rel)
        else:
            path.write_text(new)
            print("wrote", rel)
    if args.check and not bad:
        print("site.py: generated regions are current")
    return 1 if bad else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Drift as exc:
        sys.exit("site.py: %s" % exc)

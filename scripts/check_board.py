#!/usr/bin/env python3
"""The gates of the experiments board (docs/board.html, docs/board.md section 9): the page's logic is run with node, headlessly, on records made by the built `ml`.

    python3 scripts/check_board.py [--bin build/ml]       # make the runs, run the gates G1 to G10; exit 1 with the reasons
    python3 scripts/check_board.py --fixtures DIR         # use the runs and the tool's answers already written by --make-fixtures (no binary needed)
    python3 scripts/check_board.py --make-fixtures DIR    # only write them
    python3 scripts/check_board.py --only G5,G6           # some gates

The runs are made here, in a temporary directory, by the commands of scripts/demo.sh (its two fits) and a larger set (regression, clustering, a constant model, a forest,
text columns with different reader limits, a stratified split, evals), all with `--track`. The tool's answers (`ml runs list`, `show` of every run, `compare` of every ordered
pair) are written beside the records. Every expected value below is computed here, from those files and the documented rules, never by the page's code.
The logic of the page is the part of its one script between `/* LOGIC begin` and `/* LOGIC end */` (scripts/board_run.js loads only that part).
Standard library and node.
"""
import argparse
import base64
import hashlib
import itertools
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
PAGE = ROOT / "docs" / "board.html"
DATA = ROOT / "docs" / "examples" / "data"

BUDGET_PAGE = 72 * 1024
BUDGET_SCRIPT = 56 * 1024
BUDGET_WITH_CSS = 96 * 1024
PRIMARY = {"classification": "accuracy", "regression": "rmse", "clustering": "inertia"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")

# --- the runs --------------------------------------------------------------------------------------------------------------------------------------------------------

# (label, argv after `ml fit`); the first two are the fits of scripts/demo.sh, the third is the sample's k-means. Files are in the work directory.
FITS = [
    ("demo-a", "--model logistic --target virginica --step 'scale column=* kind=standard' --l2 0.1 --split train=0.8 --seed 3 --output iris.ml iris_train.csv"),
    ("demo-c", "--model logistic --target virginica --step 'scale column=* kind=standard' --l2 1.0 --split train=0.8 --seed 3 --output iris_l2_1.ml iris_train.csv"),
    ("kmeans3", "--model kmeans --k 3 --drop virginica --output km3.ml iris_train.csv"),
    ("kmeans2", "--model kmeans --k 2 --drop virginica --seed 4 --output km2.ml iris_train.csv"),
    ("linear", "--model linear --float area,price --drop id --target price --split train=0.75 --seed 1 --output houses.ml houses.csv"),
    ("constant", "--target price --drop id --output const.ml houses.csv"),
    ("forest", "--model forest --task classification --trees 20 --seed 2 --target virginica --output forest.ml iris_train.csv"),
    ("strat", "--model logistic --target virginica --split train=0.8 --stratify --seed 5 --output strat.ml iris_train.csv"),
    ("pets-cat", "--model logistic --cat species,city --step 'encode column=species kind=onehot min-count=4;freq column=city missing=zero;scale column=age,visits kind=standard' --drop id --iterations 50 --l2 0.1 --target adopted --output pets.ml pets.csv"),
    ("pets-cat32", "--model logistic --cat species,city --max-categories 32 --step 'encode column=species kind=onehot min-count=4;freq column=city missing=zero;scale column=age,visits kind=standard' --drop id --iterations 50 --l2 0.1 --target adopted --output pets32.ml pets.csv"),
    ("pets-plain", "--model logistic --target adopted --drop id,species,city --output pets_plain.ml pets.csv"),
    ("utf8", "--model linear --float '\u00e1rea,precio_\u20ac' --drop id --target 'precio_\u20ac' --output u.ml utf.csv"),
]
# evals written with --track: (label of the run, argv after `ml eval --model-file M --track runs`)
EVALS = [("demo-a", "iris_test.csv"), ("demo-a", "iris_train.csv"), ("strat", "--split train=0.5 iris_train.csv")]
# the runs of the page's sample (docs/board.md section 7): by label, and the one eval it holds; scripts/site.py writes them into docs/board.html
BOARD_SAMPLE = ("demo-a", "demo-c", "kmeans3")
BOARD_SAMPLE_EVAL = "iris_test.csv"
# non-ASCII column names: the record holds them as UTF-8 bytes, and the id is the hash of those bytes
UTF8_CSV = "id,\u00e1rea,precio_\u20ac\n1,48.5,180000\n2,71.25,240000\n3,64.0,215000\n4,98.5,335000\n5,31.0,120000\n6,50.0,170000\n"
HOUSES = "id,rooms,area,price\n1,2,48.5,180000\n2,3,71.25,240000\n3,3,64,215000\n4,4,98.5,335000\n5,1,31,120000\n6,2,50,170000\n7,3,80,260000\n8,4,90,300000\n"


def run_ml(binary, work, argv):
    cmd = [str(binary), "--root", "."] + shlex.split(argv)
    p = subprocess.run(cmd, cwd=work, capture_output=True, text=True, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"})
    return p.stdout, p.returncode


def make_fixtures(binary, out):
    """Make the runs and the tool's answers; write them to out/fixtures.json and out/runs/*.json. -> the fixtures dict."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        work = Path(tmp)
        (work / "runs").mkdir()
        for name in ("iris_train.csv", "iris_test.csv", "pets.csv"):
            shutil.copy(DATA / name, work / name)
        (work / "houses.csv").write_text(HOUSES)
        (work / "utf.csv").write_text(UTF8_CSV, encoding="utf-8")
        ids, models = {}, {}
        for label, argv in FITS:
            text, status = run_ml(binary, work, "fit " + argv + " --track runs")
            doc = json.loads(text)
            if status != 0 or not doc["ok"]:
                raise SystemExit("check_board: the fit %s failed: %s" % (label, text[:300]))
            ids[label] = doc["data"]["run"]["id"]
            models[label] = shlex.split(argv)[shlex.split(argv).index("--output") + 1]
        evals = []
        for label, argv in EVALS:
            text, status = run_ml(binary, work, "eval --model-file %s --track runs %s" % (models[label], argv))
            if status != 0 or not json.loads(text)["ok"]:
                raise SystemExit("check_board: the eval %s %s failed: %s" % (label, argv, text[:300]))
        files = {p.name: p.read_text(encoding="utf-8") for p in sorted((work / "runs").glob("*.json"))}
        listing, status = run_ml(binary, work, "runs list runs")
        runs = []
        for label, rid in ids.items():
            show, _ = run_ml(binary, work, "runs show runs " + rid)
            runs.append({"label": label, "id": rid, "file": rid + ".json", "show": show})
        compares = {}
        for a, b in itertools.product(ids.values(), repeat=2):
            text, status = run_ml(binary, work, "runs compare runs %s %s" % (a, b))
            compares["%s %s" % (a, b)] = json.loads(text)
        bad_show, _ = run_ml(binary, work, "runs show runs " + "0" * 64)
        compare_text = compares["%s %s" % (ids["demo-a"], ids["demo-c"])]
        compare_raw, _ = run_ml(binary, work, "runs compare runs %s %s" % (ids["demo-a"], ids["demo-c"]))
        pin = json.loads(files[ids["demo-a"] + ".json"])["inputs"]["compiler"]
        fx = {"files": files, "list": listing, "runs": runs, "compares": compares, "refusal": bad_show, "compare_raw": compare_raw, "compiler": pin,
              "evals": sorted(n for n in files if ".eval-" in n)}
        (out / "fixtures.json").write_text(json.dumps(fx), encoding="utf-8")
    return fx


# --- node ------------------------------------------------------------------------------------------------------------------------------------------------------

JSC = "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"


def engine_argv():
    """node when it starts; else Apple's jsc (the same script, scripts/board_run.js, runs under both)."""
    global _ENGINE
    if _ENGINE is None:
        node_path = shutil.which("node")
        if node_path and subprocess.run([node_path, "--version"], capture_output=True).returncode == 0:
            _ENGINE = [node_path]
        elif os.path.exists(JSC):
            _ENGINE = [JSC, "--"]  # after the script: [JSC, script, "--", request]
        else:
            raise RuntimeError("neither a working node nor jsc is available")
    return _ENGINE


_ENGINE = None


def node(jobs, page=None):
    req = {"html": str(page or PAGE), "jobs": jobs}
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=os.environ.get("TMPDIR"), delete=False) as f:
        json.dump(req, f)
        path = f.name
    try:
        eng = engine_argv()
        argv = [eng[0], str(HERE / "board_run.js")] + eng[1:] + [path]
        p = subprocess.run(argv, capture_output=True, text=True)
    finally:
        os.unlink(path)
    if p.returncode != 0:
        raise RuntimeError("the script engine failed: " + (p.stderr or p.stdout)[-600:])
    out = json.loads(p.stdout)
    for job, r in zip(jobs, out):
        if not r["ok"]:
            raise RuntimeError("the page's logic failed on %s: %s" % (job["op"], r["error"][:500]))
    return [r["result"] for r in out]


def one(job):
    return node([job])[0]


def modes():
    return ("subtle", "pure") if one({"op": "engine"})["subtle"] else ("pure",)


def analyze(sources, mode="pure"):
    return one({"op": "analyze", "sources": sources, "mode": mode})


def b64(b):
    return base64.b64encode(b).decode()


def codes(result):
    return sorted(p["code"] for p in result["problems"])


def row_by_id(result, rid):
    rows = [r for r in result["rows"] if r["id"] == rid]
    return rows[0] if rows else None


# --- the gates -------------------------------------------------------------------------------------------------------------------------------------------------

def g1_sha256(fx, errs):
    rng = random.Random(7)
    vectors = [b"", b"abc", "pr\u00e9cio \u20ac \U0001f600".encode()] + [bytes(rng.randrange(256) for _ in range(n)) for n in (55, 56, 57, 63, 64, 65, 119, 120, 128)]
    vectors.append(bytes(rng.randrange(256) for _ in range(1 << 20)))
    eng = one({"op": "engine"})
    jobs = [{"op": "sha", "b64": b64(v), "mode": "pure"} for v in vectors]
    if eng["subtle"]:
        jobs += [{"op": "sha", "b64": b64(v), "mode": "subtle"} for v in vectors]
    got = node(jobs)
    for i, v in enumerate(vectors):
        want = hashlib.sha256(v).hexdigest()
        if got[i] != want:
            errs.append("G1: the built-in SHA-256 of a %d-byte vector is %s, hashlib says %s" % (len(v), got[i], want))
        if eng["subtle"] and got[len(vectors) + i] != want:
            errs.append("G1: crypto.subtle through makeDigest gives %s for a %d-byte vector, hashlib says %s" % (got[len(vectors) + i], len(v), want))
    # the routing: a crypto.subtle is used when the environment has one, the built-in routine when it has none or when the digest call is refused
    r = one({"op": "routing", "b64": b64(b"abc")})
    want_abc = hashlib.sha256(b"abc").hexdigest()
    if r["kinds"] != ["subtle", "subtle", "pure", "pure"]:
        errs.append("G1: makeDigest picks %s for (a subtle, a subtle that refuses, a crypto without subtle, nothing), want subtle, subtle, pure, pure" % r["kinds"])
    if r["viaSubtle"] != "".join("%02x" % i for i in range(32)):
        errs.append("G1: makeDigest did not return the digest of crypto.subtle (%s)" % r["viaSubtle"])
    for k in ("afterReject", "none", "undef"):
        if r[k] != want_abc:
            errs.append("G1: makeDigest without a working crypto.subtle (%s) gives %s, hashlib says %s" % (k, r[k], want_abc))
    return eng


def sources_files(fx, names=None):
    names = names or list(fx["files"])
    return [{"name": n, "text": fx["files"][n]} for n in names]


FIELDS = ["id", "model", "task", "trainRows", "split", "metricsOn", "metricName", "metricText", "dataSha", "modelSha", "evals", "verdict"]


def pick(row):
    return {k: row.get(k) for k in FIELDS}


def g2_ids_and_forms(fx, errs):
    files = fx["files"]
    list_doc = json.loads(fx["list"])["data"]
    list_rows = {r["id"]: r for r in list_doc["runs"]}
    run_ids = [r["id"] for r in fx["runs"]]
    forms = {
        "files": sources_files(fx),
        "list+files": [{"name": "list.json", "text": fx["list"]}] + sources_files(fx),
        "show": [{"name": "show-%s.txt" % r["id"][:12], "text": r["show"]} for r in fx["runs"]] + [{"name": n, "text": files[n]} for n in fx["evals"]],
        "pasted": [{"name": "pasted text", "text": "\n".join(files[n].strip() for n in files)}],
    }
    results = {}
    for mode in modes():
        for name, srcs in forms.items():
            r = analyze(srcs, mode)
            results[(name, mode)] = r
            if r["problems"]:
                errs.append("G2: form %s (%s digest) gives problems %s on honest records" % (name, mode, codes(r)))
            if sorted(x["id"] for x in r["rows"]) != sorted(run_ids):
                errs.append("G2: form %s (%s digest) shows runs %s, want %s" % (name, mode, [x["id"][:12] for x in r["rows"]], [x[:12] for x in run_ids]))
                continue
            for row in r["rows"]:
                if row["verdict"] != "ok":
                    errs.append("G2: form %s (%s): run %s has verdict %s" % (name, mode, row["id"][:12], row["verdict"]))
                if row["kind"] != "record":
                    errs.append("G2: form %s: run %s is not a record row" % (name, row["id"][:12]))
    base = {r["id"]: pick(r) for r in results[("files", modes()[0])]["rows"]}
    for (name, mode), r in results.items():
        for row in r["rows"]:
            if pick(row) != base.get(row["id"]):
                errs.append("G2: form %s (%s) reads run %s differently from the files: %s against %s" % (name, mode, row["id"][:12], pick(row), base.get(row["id"])))
    # the fields against `ml runs list`
    for row in results[("files", modes()[0])]["rows"]:
        lr = list_rows[row["id"]]
        for mine, theirs in (("model", "model"), ("task", "task"), ("dataSha", "data_sha256"), ("modelSha", "model_sha256"), ("trainRows", "train_rows"), ("metricsOn", "metrics_on"), ("evals", "evals")):
            if row[mine] != lr[theirs]:
                errs.append("G2: run %s: %s is %r, `ml runs list` says %s %r" % (row["id"][:12], mine, row[mine], theirs, lr[theirs]))
        rec = json.loads(files[row["id"] + ".json"])
        if row["split"] != rec["inputs"]["split"]:
            errs.append("G2: run %s: split text %r, record says %r" % (row["id"][:12], row["split"], rec["inputs"]["split"]))
    # a list alone: rows from the list, flagged as unchecked
    lonly = analyze([{"name": "list.json", "text": fx["list"]}])
    if sorted(x["id"] for x in lonly["rows"]) != sorted(run_ids):
        errs.append("G2: the list alone shows %d runs, want %d" % (len(lonly["rows"]), len(run_ids)))
    for row in lonly["rows"]:
        lr = list_rows[row["id"]]
        if row["kind"] != "list" or row["verdict"] != "list-only":
            errs.append("G2: list row %s is %s/%s, want list/list-only" % (row["id"][:12], row["kind"], row["verdict"]))
        for mine, theirs in (("model", "model"), ("task", "task"), ("dataSha", "data_sha256"), ("trainRows", "train_rows"), ("metricsOn", "metrics_on"), ("evals", "evals")):
            if row[mine] != lr[theirs]:
                errs.append("G2: list row %s: %s is %r, the list says %r" % (row["id"][:12], mine, row[mine], lr[theirs]))
    if codes(lonly) != ["list-only"]:
        errs.append("G2: the list alone gives problems %s, want exactly one list-only note" % codes(lonly))
    # the list in the middle of the order is the same list (sorted by id)
    if [r["id"] for r in list_doc["runs"]] != sorted(run_ids):
        errs.append("G2: `ml runs list` is not in id order (the default order of the page rests on it)")
    # the commands to copy
    rid, rid2 = run_ids[0], run_ids[1]
    cmds = one({"op": "commands", "dir": "runs", "id": rid, "a": rid, "b": rid2})
    want = {"verify": "ml runs verify runs " + rid, "compare": "ml runs compare runs %s %s" % (rid, rid2), "show": "ml runs show runs " + rid}
    if cmds != want:
        errs.append("G2: the commands are %s, want %s" % (cmds, want))
    q = one({"op": "commands", "dir": "my runs/it's", "id": rid, "a": rid, "b": rid2})["verify"]
    if shlex.split(q) != ["ml", "runs", "verify", "my runs/it's", rid]:
        errs.append("G2: a directory name with a space and a quote is not quoted for a shell: %s" % q)
    return results


def flip(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def g3_tamper(fx, errs):
    a = fx["runs"][0]
    text = fx["files"][a["file"]]
    rid = a["id"]
    rec = json.loads(text)

    def flags_of(t, name="pasted text"):
        r = analyze([{"name": name, "text": t}])
        row = row_by_id(r, rid)
        return r, row

    cases = []
    # one byte of `inputs` changed: seed 3 -> 4 (the seed appears twice in `inputs`: in its member and in the split text; change the member)
    cases.append(("an inputs byte (seed)", flip(text, '"seed":3,"positive"', '"seed":4,"positive"'), {"id-mismatch", "check-mismatch"}, "id-check-mismatch"))
    cases.append(("an inputs byte (data hash)", text.replace(rec["inputs"]["data"]["sha256"], rec["inputs"]["data"]["sha256"][:-1] + ("0" if rec["inputs"]["data"]["sha256"][-1] != "0" else "1"), 1),
                  {"id-mismatch", "check-mismatch"}, "id-check-mismatch"))
    cases.append(("a metric byte", flip(text, '"accuracy":"0.9333333333333333"', '"accuracy":"0.9433333333333333"'), {"check-mismatch"}, "check-mismatch"))
    cases.append(("the model hash", text.replace(rec["model"]["sha256"], "0" * 64, 1), {"check-mismatch"}, "check-mismatch"))
    idm = '"id":"' + rid[:-1] + ("0" if rid[-1] != "0" else "1") + '"'
    cases.append(("the id member", flip(text, '"id":"' + rid + '"', idm), {"id-mismatch", "check-mismatch"}, "id-check-mismatch"))
    chk = rec["check"]
    cases.append(("the check member", flip(text, '"check":"' + chk + '"', '"check":"' + chk[:-1] + ("0" if chk[-1] != "0" else "1") + '"'), {"check-mismatch"}, "check-mismatch"))
    for what, t, want_flags, verdict in cases:
        r, row = flags_of(t)
        if row is None and verdict != "id-check-mismatch":
            errs.append("G3: %s: the record is not shown" % what)
            continue
        got = set(p["code"] for p in r["problems"])
        if got != want_flags:
            errs.append("G3: %s: problems %s, want %s" % (what, sorted(got), sorted(want_flags)))
        shown = [x for x in r["rows"]]
        if len(shown) != 1:
            errs.append("G3: %s: %d rows shown, want 1 (a damaged record is shown, marked)" % (what, len(shown)))
        elif shown[0]["verdict"] != verdict:
            errs.append("G3: %s: verdict %s, want %s" % (what, shown[0]["verdict"], verdict))
        for p in r["problems"]:
            if p["severity"] != "error":
                errs.append("G3: %s: problem %s is a %s, want an error" % (what, p["code"], p["severity"]))
    # tampering in a show answer (the record nested in an envelope)
    shown = a["show"]
    r, row = flags_of(flip(shown, '"seed":3,"positive"', '"seed":4,"positive"'))
    if set(p["code"] for p in r["problems"]) != {"id-mismatch", "check-mismatch"}:
        errs.append("G3: an inputs byte changed in a `show` answer gives %s" % codes(r))
    # a reformatted copy: pretty printed JSON of the same record is not forged
    pretty = json.dumps(rec, indent=2)
    r, row = flags_of(pretty)
    if codes(r) != ["reformatted"] or r["problems"][0]["severity"] != "note" or row is None or row["verdict"] != "reformatted":
        errs.append("G3: a pretty-printed copy gives %s / %s, want one reformatted note" % (codes(r), row and row["verdict"]))
    # a pretty-printed copy with a changed seed is still caught
    rec2 = json.loads(text)
    rec2["inputs"]["seed"] = 4
    r, row = flags_of(json.dumps(rec2, indent=2))
    if "id-mismatch" not in codes(r):
        errs.append("G3: a pretty-printed copy with a changed input is not an id-mismatch: %s" % codes(r))
    # a record under another id's file name
    other = fx["runs"][1]["id"]
    r, row = flags_of(text, other + ".json")
    if codes(r) != ["name-mismatch"]:
        errs.append("G3: a record under another id's file name gives %s, want name-mismatch" % codes(r))
    r, row = flags_of(text, rid + ".json")
    if codes(r) != []:
        errs.append("G3: the record under its own file name gives %s" % codes(r))
    # the problems name their source
    r, row = flags_of(flip(text, '"seed":3,"positive"', '"seed":4,"positive"'), "damaged-copy.json")
    if not r["problems"] or not all(p["source"] == "damaged-copy.json" for p in r["problems"]):
        errs.append("G3: a problem does not name its source file")
    # both digests agree on the damaged record
    r2 = analyze([{"name": "damaged-copy.json", "text": flip(text, '"seed":3,"positive"', '"seed":4,"positive"')}], "pure")
    if codes(r2) != codes(r):
        errs.append("G3: the pure digest and crypto.subtle disagree on a damaged record")
    # the computed id is reported
    mism = [p for p in r["problems"] if p["code"] == "id-mismatch"]
    if mism:
        calc = hashlib.sha256(re.search(r'"inputs":(\{.*?\}),"threads"', flip(text, '"seed":3,"positive"', '"seed":4,"positive"')).group(1).encode()).hexdigest()
        if calc[:12] not in mism[0]["message"]:
            errs.append("G3: the id-mismatch message does not give the first 12 digits of the recomputed id (%s): %s" % (calc[:12], mism[0]["message"]))


def g4_problems(fx, errs):
    a, c = fx["runs"][0], fx["runs"][1]
    ta, tc = fx["files"][a["file"]], fx["files"][c["file"]]
    ev_name = [n for n in fx["evals"] if n.startswith(a["id"])][0]
    tev = fx["files"][ev_name]

    def case(what, sources, want_codes, want_rows, sev=None):
        r = analyze(sources)
        if codes(r) != sorted(want_codes):
            errs.append("G4: %s: problems %s, want %s" % (what, codes(r), sorted(want_codes)))
        if len(r["rows"]) != want_rows:
            errs.append("G4: %s: %d rows, want %d" % (what, len(r["rows"]), want_rows))
        for p in r["problems"]:
            if not p.get("source"):
                errs.append("G4: %s: problem %s has no source" % (what, p["code"]))
            if p["severity"] not in ("error", "note"):
                errs.append("G4: %s: problem %s has severity %r" % (what, p["code"], p["severity"]))
        if sev:
            for p in r["problems"]:
                if sev.get(p["code"]) and p["severity"] != sev[p["code"]]:
                    errs.append("G4: %s: %s is a %s, want %s" % (what, p["code"], p["severity"], sev[p["code"]]))
        return r

    case("one honest record", [{"name": a["file"], "text": ta}], [], 1)
    case("the same record twice", [{"name": "one.json", "text": ta}, {"name": "two.json", "text": ta}], ["duplicate-id"], 1, {"duplicate-id": "note"})
    copy = flip(ta, '"accuracy":"0.9333333333333333"', '"accuracy":"0.9433333333333333"')
    r = case("two records with one id that differ", [{"name": "one.json", "text": ta}, {"name": "two.json", "text": copy}], ["duplicate-id", "check-mismatch"], 1, {"duplicate-id": "error"})
    if r["rows"] and r["rows"][0]["verdict"] != "ok":
        errs.append("G4: the first of two records with one id is not the one shown")
    case("a record of schema ml.run.v2", [{"name": "v2.json", "text": ta.replace("ml.run.v1", "ml.run.v2")}], ["schema-version"], 0)
    case("an eval record of schema ml.eval.v2", [{"name": a["file"], "text": ta}, {"name": "e.json", "text": tev.replace("ml.eval.v1", "ml.eval.v2")}], ["schema-version"], 1)
    case("an answer of another schema", [{"name": "x.json", "text": fx["list"].replace('"schema":"ml.v1"', '"schema":"ml.v9"')}], ["schema-version"], 0)
    case("a JSON array", [{"name": "x.json", "text": "[1,2,3]"}], ["not-a-record"], 0)
    case("an object without schema", [{"name": "x.json", "text": '{"a":1}'}], ["not-a-record"], 0)
    case("a number", [{"name": "x.json", "text": "42"}], ["not-json"], 0)
    case("a refusal of the tool", [{"name": "x.json", "text": fx["refusal"]}], ["refusal"], 0)
    case("an answer of ml runs compare", [{"name": "x.json", "text": fx["compare_raw"]}], ["compare-answer"], 0)
    case("a CSV", [{"name": "iris.json", "text": "a,b\n1,2\n3,4\n"}], ["not-json"], 0)
    case("an empty file", [{"name": "empty.json", "text": ""}], ["empty"], 0)
    case("a file of blanks", [{"name": "blank.json", "text": " \n\n"}], ["empty"], 0)
    case("a cut record", [{"name": "cut.json", "text": ta[:200]}], ["not-json"], 0)
    case("not JSON inside braces", [{"name": "bad.json", "text": '{"a": nope}'}], ["not-json"], 0)
    r = case("a prompt line around a record", [{"name": "pasted text", "text": "$ ml runs show runs %s\n%s\n$\n" % (a["id"], a["show"])}], ["junk"], 1, {"junk": "note"})
    case("a record and a second record on one line", [{"name": "pasted text", "text": ta.strip() + " " + tc.strip()}], [], 2)
    case("an eval whose run is not loaded", [{"name": ev_name, "text": tev}], ["eval-orphan"], 0)
    case("an eval with its run", [{"name": a["file"], "text": ta}, {"name": ev_name, "text": tev}], [], 1)
    case("the same eval twice", [{"name": a["file"], "text": ta}, {"name": ev_name, "text": tev}, {"name": "again.json", "text": tev}], ["duplicate-eval"], 1, {"duplicate-eval": "note"})
    eb = flip(tev, '"accuracy":"1.0"', '"accuracy":"0.5"')
    case("an eval with a changed metric", [{"name": a["file"], "text": ta}, {"name": ev_name, "text": eb}], ["check-mismatch"], 1)
    # a malformed record: the training row count removed from `counts`
    mal = ta.replace('"train_rows":60,', "")
    case("a record without counts.train_rows", [{"name": "m.json", "text": mal}], ["check-mismatch", "malformed"], 1)
    case("a record with a short hash", [{"name": "m.json", "text": ta.replace(a["id"], a["id"][:60], 1)}], ["check-mismatch", "id-mismatch", "malformed"], 1)
    # a list that disagrees with its record: the first run of the list that has 60 training rows, its record loaded
    ld = json.loads(fx["list"])
    i60 = next(i for i, x in enumerate(ld["data"]["runs"]) if x["train_rows"] == 60)
    bad = json.loads(fx["list"])
    bad["data"]["runs"][i60]["train_rows"] = 61
    rid60 = ld["data"]["runs"][i60]["id"]
    nlist = len(ld["data"]["runs"])
    case("a list row that disagrees with its record", [{"name": "list.json", "text": json.dumps(bad, separators=(",", ":"))}, {"name": rid60 + ".json", "text": fx["files"][rid60 + ".json"]}],
         ["list-disagrees", "list-only"], nlist, {"list-disagrees": "error", "list-only": "note"})
    j = next(i for i, x in enumerate(ld["data"]["runs"]) if x["evals"] >= 1)
    bad = json.loads(fx["list"])
    bad["data"]["runs"][j]["evals"] += 1
    case("a list whose eval count is not the number of eval records loaded", [{"name": "list.json", "text": json.dumps(bad, separators=(",", ":"))}] + sources_files(fx), ["evals-count"], len(fx["runs"]),
         {"evals-count": "note"})
    # a split text in the sources that holds many documents: the order of the rows is the order of reading (records first, then list-only rows)
    r = analyze([{"name": "list.json", "text": fx["list"]}, {"name": c["file"], "text": tc}])
    kinds = [x["kind"] for x in r["rows"]]
    if kinds[0] != "record" or kinds.count("record") != 1:
        errs.append("G4: with a list and one record the rows are %s, want the record first" % kinds)
    # a pasted text holding every kind at once
    big = "\n".join([ta, "noise line", fx["list"], a["show"], fx["refusal"]])
    r = analyze([{"name": "pasted text", "text": big}])
    want = ["duplicate-id", "junk", "refusal"]
    if not set(want) <= set(codes(r)):
        errs.append("G4: a pasted text with a duplicate, noise and a refusal gives %s, want at least %s" % (codes(r), want))
    # splitDocs
    sp = one({"op": "split", "text": 'x {"a":"}{"} [1,{"b":2}] y\n{"c":1}'})
    if [d["raw"] for d in sp["docs"]] != ['{"a":"}{"}', '[1,{"b":2}]', '{"c":1}']:
        errs.append("G4: splitDocs gives %s on text with braces inside a string" % [d["raw"] for d in sp["docs"]])


def flat(prefix, v, out):
    if isinstance(v, dict):
        for k, x in v.items():
            flat(prefix + k + ".", x, out)
    else:
        out[prefix[:-1]] = v
    return out


def g5_compare(fx, errs):
    r = analyze(sources_files(fx))
    rows = {x["id"]: x for x in r["rows"]}
    jobs, keys = [], []
    for (key, want) in fx["compares"].items():
        a, b = key.split()
        jobs.append({"op": "compare", "a": rows[a], "b": rows[b]})
        keys.append(key)
    got = node(jobs)
    n_diff_inputs = n_delta = 0
    for key, g in zip(keys, got):
        want = fx["compares"][key]["data"]
        a, b = key.split()
        mine = json.loads(json.dumps(g))
        deltas = [m.pop("delta", "MISSING") for m in mine["metrics"]]
        if mine != want:
            def first(x, y, path=""):
                if type(x) != type(y):
                    return path
                if isinstance(x, dict):
                    for k in list(x) + [k for k in y if k not in x]:
                        if x.get(k, "<none>") != y.get(k, "<none>"):
                            return first(x.get(k), y.get(k), path + "/" + k)
                if isinstance(x, list):
                    if len(x) != len(y):
                        return path + "[len %d vs %d]" % (len(x), len(y))
                    for i, (p, q) in enumerate(zip(x, y)):
                        if p != q:
                            return first(p, q, path + "[%d]" % i)
                return path
            errs.append("G5: compare %s %s differs from `ml runs compare` at %s: page %s, tool %s" % (a[:12], b[:12], first(mine, want), json.dumps(mine)[:200], json.dumps(want)[:200]))
            continue
        n_diff_inputs += len(want["inputs"])
        for m, d in zip(want["metrics"], deltas):
            expect = None
            if isinstance(m["a"], str) and isinstance(m["b"], str):
                expect = float(m["b"]) - float(m["a"])
            if d != expect:
                errs.append("G5: compare %s %s: delta of %s is %r, b - a is %r" % (a[:12], b[:12], m["name"], d, expect))
            n_delta += expect is not None
    if len(keys) < 100:
        errs.append("G5: only %d ordered pairs were compared" % len(keys))
    # the cases that make the rules differ: nested keys, a null against an object, lists
    kinds = {"data.sha256": 0, "reader": 0, "reader.max_categories": 0, "sizes": 0}
    for key in fx["compares"]:
        d = fx["compares"][key]["data"]
        for i in d["inputs"]:
            if i["key"] in kinds:
                kinds[i["key"]] += 1
        for m in d["metrics"]:
            if m["name"] == "sizes" and m["equal"] is False and m["a"] and m["b"]:
                kinds["sizes"] += 1
    for k, n in kinds.items():
        if n == 0:
            errs.append("G5: the runs of this gate never exercise the compare key %s" % k)
    return n_diff_inputs, n_delta, len(keys)


def py_flat_inputs(rec):
    return flat("", rec["inputs"], {}) if False else _flat_inputs(rec["inputs"])


def _flat_inputs(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat_inputs(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def metric_of(rec):
    m = rec["metrics"]
    name = PRIMARY.get(rec["inputs"]["task"])
    if not m or name is None or name not in m["metrics"]:
        return None, None
    v = m["metrics"][name]
    try:
        return name, (v if isinstance(v, str) else repr(v))
    except Exception:
        return name, None


def sort_key_value(row, key):
    return {"id": row["id"], "model": row["model"], "task": row["task"], "train_rows": row["trainRows"], "split": row["split"], "metrics_on": row["metricsOn"],
            "metric": row["metricValue"], "data": row["dataSha"], "evals": row["evals"], "verdict": row["verdict"]}[key]


def py_sort(rows, key, direction):
    present = [r for r in rows if sort_key_value(r, key) is not None]
    missing = [r for r in rows if sort_key_value(r, key) is None]
    present = sorted(present, key=lambda r: r["id"])
    present = sorted(present, key=lambda r: sort_key_value(r, key), reverse=(direction == "desc"))
    return present + sorted(missing, key=lambda r: r["id"])


def g6_table(fx, errs):
    # records of ten runs and a list of all: two rows are list-only (missing metric, model...)
    names = [r["file"] for r in fx["runs"][:-2]]
    srcs = [{"name": "list.json", "text": fx["list"]}] + sources_files(fx, names)
    res = analyze(srcs)
    rows = res["rows"]
    if len([r for r in rows if r["kind"] == "list"]) != 2:
        errs.append("G6: the table holds %d list-only rows, want 2" % len([r for r in rows if r["kind"] == "list"]))
    keys = ["id", "model", "task", "train_rows", "split", "metrics_on", "metric", "data", "evals", "verdict"]
    jobs = [{"op": "sort", "rows": rows, "key": k, "dir": d} for k in keys for d in ("asc", "desc")]
    got = node(jobs)
    i = 0
    for k in keys:
        for d in ("asc", "desc"):
            want = [r["uid"] for r in py_sort(rows, k, d)]
            if got[i] != want:
                errs.append("G6: sort by %s %s gives %s, want %s" % (k, d, [u for u in got[i]][:6], want[:6]))
            i += 1
    # the default (id ascending) is the order of `ml runs list`
    listed = [r["id"] for r in json.loads(fx["list"])["data"]["runs"]]
    uid_to_id = {r["uid"]: r["id"] for r in rows}
    if [uid_to_id[u] for u in got[0]] != listed:
        errs.append("G6: the default order is not the order of `ml runs list`")
    # a sort does not change its input and is stable for equal keys: sort by task twice gives the same
    again = node([{"op": "sort", "rows": rows, "key": "task", "dir": "asc"}])[0]
    if again != got[keys.index("task") * 2]:
        errs.append("G6: sorting the same rows twice gives two orders")
    # the metric column: runs without one are last in both directions
    for d in ("asc", "desc"):
        order = got[keys.index("metric") * 2 + (0 if d == "asc" else 1)]
        vals = [next(r for r in rows if r["uid"] == u)["metricValue"] for u in order]
        seen_none = False
        for v in vals:
            if v is None:
                seen_none = True
            elif seen_none:
                errs.append("G6: a run without the metric is not last (%s)" % d)
                break
    # filters
    recs = {r["id"]: json.loads(fx["files"][r["id"] + ".json"]) for r in rows if r["kind"] == "record"}
    models = sorted(set(r["model"] for r in rows))
    datas = sorted(set(r["dataSha"] for r in rows))
    cases = []
    for m in models:
        cases.append(({"model": m}, [r for r in rows if r["model"] == m]))
    for dsha in datas:
        cases.append(({"data": dsha}, [r for r in rows if r["dataSha"] == dsha]))
    cases.append(({"model": "logistic", "data": datas[0]}, [r for r in rows if r["model"] == "logistic" and r["dataSha"] == datas[0]]))
    cases.append(({}, rows))
    cases.append(({"model": "no-such-model"}, []))
    plan_text = lambda rec: rec["inputs"]["plan"].rstrip("\n").replace("\n", " ; ")
    for r0 in [r for r in rows if r["kind"] == "record"][:4]:
        val = plan_text(recs[r0["id"]])
        cases.append(({"key": "plan", "value": val}, [r for r in rows if r["kind"] == "record" and plan_text(recs[r["id"]]) == val]))
    seed_vals = sorted(set(json.dumps(recs[r["id"]]["inputs"]["seed"]) for r in rows if r["kind"] == "record"))
    for sv in seed_vals:
        cases.append(({"key": "seed", "value": sv}, [r for r in rows if r["kind"] == "record" and json.dumps(recs[r["id"]]["inputs"]["seed"]) == sv]))
    got = node([{"op": "filter", "rows": rows, "f": f} for f, _ in cases])
    for (f, want), g in zip(cases, got):
        w = [r["uid"] for r in want]
        if sorted(g) != sorted(w):
            errs.append("G6: filter %s gives %d rows, want %d" % (json.dumps(f), len(g), len(w)))
        elif g != w:
            errs.append("G6: filter %s does not keep the order of its input" % json.dumps(f))
    # the inputs that vary
    varying = node([{"op": "varying", "rows": rows}])[0]
    flats = [_flat_inputs(recs[r["id"]]["inputs"]) for r in rows if r["kind"] == "record"]
    keys_all = []
    for f in flats:
        for k in f:
            if k not in keys_all:
                keys_all.append(k)
    want_vary = [k for k in keys_all if len(set(json.dumps(f.get(k, "<missing>")) for f in flats)) > 1]
    if varying != want_vary:
        errs.append("G6: the inputs that vary are %s, want %s" % (varying, want_vary))
    if "plan" not in varying or "data.sha256" not in varying or "seed" not in varying:
        errs.append("G6: the runs of this gate do not vary in plan, seed and data.sha256")
    # the text of an input
    texts = node([{"op": "inputValue", "rows": [r for r in rows if r["kind"] == "record"], "key": "plan"}])[0]
    want_t = [plan_text(recs[r["id"]]) for r in rows if r["kind"] == "record"]
    if texts != want_t:
        errs.append("G6: the text of the plan input is %s, want %s" % (texts[:2], want_t[:2]))


def g7_primary(fx, errs):
    res = analyze(sources_files(fx))
    seen = set()
    for row in res["rows"]:
        rec = json.loads(fx["files"][row["id"] + ".json"])
        name, text = metric_of(rec)
        seen.add((rec["inputs"]["task"], name))
        if row["metricName"] != name or row["metricText"] != text:
            errs.append("G7: run %s (%s, %s): primary metric %s=%s, want %s=%s" % (row["id"][:12], rec["inputs"]["task"], row["model"], row["metricName"], row["metricText"], name, text))
        if text is not None and row["metricValue"] != float(text):
            errs.append("G7: run %s: metric value %r, want %r" % (row["id"][:12], row["metricValue"], float(text)))
        if text is None and row["metricValue"] is not None:
            errs.append("G7: run %s has no metric text but a value" % row["id"][:12])
    for want in (("classification", "accuracy"), ("regression", "rmse"), ("clustering", "inertia"), ("regression", None)):
        if want not in seen:
            errs.append("G7: the runs of this gate hold no %s" % (want,))
    # edge cases of the rule
    cases = [({"task": "classification", "metrics": {"accuracy": "0.5", "f1": "0.1"}}, {"name": "accuracy", "text": "0.5", "value": 0.5}),
             ({"task": "regression", "metrics": {"mae": "1", "rmse": "2.5"}}, {"name": "rmse", "text": "2.5", "value": 2.5}),
             ({"task": "clustering", "metrics": {"inertia": "30.5", "sizes": [1, 2]}}, {"name": "inertia", "text": "30.5", "value": 30.5}),
             ({"task": "regression", "metrics": {"mae": "1"}}, None), ({"task": "regression", "metrics": None}, None), ({"task": "ranking", "metrics": {"accuracy": "1"}}, None),
             ({"task": "classification", "metrics": {"accuracy": 0.25}}, {"name": "accuracy", "text": "0.25", "value": 0.25}),
             ({"task": "classification", "metrics": {"accuracy": "abc"}}, {"name": "accuracy", "text": "abc", "value": None}),
             ({"task": "classification", "metrics": {"accuracy": [1, 2]}}, None)]
    got = node([{"op": "primary", "task": c["task"], "metrics": c["metrics"]} for c, _ in cases])
    for (c, want), g in zip(cases, got):
        if g != want:
            errs.append("G7: primaryMetric(%s, %s) = %s, want %s" % (c["task"], json.dumps(c["metrics"]), g, want))


def g8_chart(fx, errs):
    res = analyze(sources_files(fx))
    rows = res["rows"]
    rec = {r["id"]: json.loads(fx["files"][r["id"] + ".json"]) for r in rows}

    def scalar(rr, name):
        m = rr["metrics"]
        if not m or name not in m["metrics"]:
            return None
        v = m["metrics"][name]
        if isinstance(v, (str, int, float)) and not isinstance(v, bool):
            try:
                return float(v)
            except ValueError:
                return None
        return None

    names = []
    for r in rows:
        m = rec[r["id"]]["metrics"]
        for k in (m["metrics"] if m else {}):
            if k not in names and any(scalar(rec[x["id"]], k) is not None for x in rows):
                names.append(k)
    mn = node([{"op": "metricNames", "rows": rows}])[0]
    if mn != names:
        errs.append("G8: the metric names are %s, want %s" % (mn, names))
    for metric in ["accuracy", "rmse", "inertia", "f1"]:
        g = node([{"op": "chart", "rows": rows, "metric": metric}])[0]
        want_bars = []
        left = []
        for r in rows:
            v = scalar(rec[r["id"]], metric)
            if v is None:
                left.append(r["id"])
            else:
                want_bars.append((r["id"], v, rec[r["id"]]["metrics"]["on"]))
        got_bars = [(b["id"], b["value"], b["on"]) for b in g["bars"]]
        if got_bars != want_bars:
            errs.append("G8: chart %s has bars %s, want %s" % (metric, got_bars[:3], want_bars[:3]))
        if sorted(g["left"]) != sorted(left):
            errs.append("G8: chart %s leaves out %d runs, want %d" % (metric, len(g["left"]), len(left)))
        vals = [v for _, v, _ in want_bars]
        if vals and (g["lo"] != min(0.0, min(vals)) or g["hi"] != max(0.0, max(vals))):
            errs.append("G8: chart %s range is %s..%s, want %s..%s" % (metric, g["lo"], g["hi"], min(0.0, min(vals)), max(0.0, max(vals))))
        if vals:
            lay = node([{"op": "layout", "data": g, "width": 400}])[0]
            for b, v, l in zip(g["bars"], vals, lay):
                span = g["hi"] - g["lo"]
                want_w = abs(v) / span * 400 if span else 0
                if abs(l["w"] - want_w) > 1e-6 or l["uid"] != b["uid"]:
                    errs.append("G8: bar of %s in chart %s is %.3f wide, want %.3f" % (b["id"][:12], metric, l["w"], want_w))
                    break
    empty = node([{"op": "chart", "rows": rows, "metric": "no-such-metric"}])[0]
    if empty["bars"] != [] or len(empty["left"]) != len(rows):
        errs.append("G8: a metric nobody has gives bars %s" % empty["bars"])
    # the default metric: the primary metric most runs share, ties by first appearance
    counts = {}
    order = []
    for r in rows:
        name, _ = metric_of(rec[r["id"]])
        if name:
            counts[name] = counts.get(name, 0) + 1
            if name not in order:
                order.append(name)
    want_default = max(order, key=lambda n: (counts[n], -order.index(n))) if order else None
    dm = node([{"op": "defaultMetric", "rows": rows}])[0]
    if dm != want_default:
        errs.append("G8: the default metric is %s, want %s" % (dm, want_default))
    # negative values: the axis starts at the smallest negative
    fake = [dict(rows[0], uid="u1", id="a" * 64, metricName="r2"), dict(rows[1], uid="u2", id="b" * 64, metricName="r2")]
    for f, v in zip(fake, ("-0.5", "2")):
        f["record"] = {"metrics": {"rows": 1, "on": "held-out", "metrics": {"r2": v}}}
    g = node([{"op": "chart", "rows": fake, "metric": "r2"}])[0]
    if (g["lo"], g["hi"]) != (-0.5, 2.0):
        errs.append("G8: a chart with a negative value has the range %s..%s, want -0.5..2" % (g["lo"], g["hi"]))
    lay = node([{"op": "layout", "data": g, "width": 250}])[0]
    if abs(lay[0]["x"] - 0.0) > 1e-6 or abs(lay[0]["w"] - 50.0) > 1e-6 or abs(lay[1]["x"] - 50.0) > 1e-6 or abs(lay[1]["w"] - 200.0) > 1e-6:
        errs.append("G8: the bars of a chart with a negative value are placed at %s" % lay)


class SampleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cur = None
        self.blocks = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "script" and a.get("id") == "sample":
            self.cur = a
            self.blocks["attrs"] = a
            self.blocks["text"] = ""

    def handle_endtag(self, tag):
        if tag == "script":
            self.cur = None

    def handle_data(self, d):
        if self.cur is not None:
            self.blocks["text"] += d


def g9_sample(fx, errs, html):
    p = SampleParser()
    p.feed(html)
    if "text" not in p.blocks:
        errs.append("G9: the page has no <script type=\"text/plain\" id=\"sample\"> block")
        return
    if p.blocks["attrs"].get("type") != "text/plain":
        errs.append("G9: the sample block is not type text/plain (the browser must not run or parse it)")
    text = p.blocks["text"]
    lines = [l for l in text.split("\n") if l.strip()]
    r = analyze([{"name": "sample", "text": text}])
    if r["problems"]:
        errs.append("G9: the page's own logic finds problems in the sample: %s" % codes(r))
    if len(r["rows"]) != 3:
        errs.append("G9: the sample shows %d runs, want 3" % len(r["rows"]))
    if sum(row["evals"] for row in r["rows"]) != 1:
        errs.append("G9: the sample holds %d evals, want 1" % sum(row["evals"] for row in r["rows"]))
    ids = {x["label"]: x["id"] for x in fx["runs"]}
    want_ids = sorted([ids["demo-a"], ids["demo-c"], ids["kmeans3"]])
    if sorted(x["id"] for x in r["rows"]) != want_ids:
        errs.append("G9: the sample's runs are %s, want those of a fresh run of the same commands %s" % ([x["id"][:12] for x in r["rows"]], [x[:12] for x in want_ids]))
    fresh = [fx["files"][i + ".json"].strip() for i in want_ids]
    ev = [fx["files"][n].strip() for n in fx["evals"] if n.startswith(ids["demo-a"]) and json.loads(fx["files"][n])["data"]["path"] == "iris_test.csv"]
    if lines != fresh + ev:
        errs.append("G9: the sample's records are not, byte for byte and in the order of run id then eval, the records of a fresh run (%d lines against %d)" % (len(lines), len(fresh + ev)))
    if len(text.encode()) > 12 * 1024:
        errs.append("G9: the sample is %d bytes, over its 12 KB" % len(text.encode()))
    if "</script" in text.lower():
        errs.append("G9: the sample holds a closing script tag")


class Structure(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.attrs = []
        self.h1 = 0
        self.lang = None
        self.title = ""
        self._in_title = False
        self.desc = None
        self.scripts = []        # (attrs, text)
        self._script = None
        self.styles = []
        self._style = None
        self.labels_for = set()
        self.controls = []
        self.in_label = 0
        self.skip = None
        self.main = False
        self.viewport = False
        self.text = []
        self.in_head = True
        self.svg = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.append(tag)
        self.attrs.append((tag, a))
        if tag == "html":
            self.lang = a.get("lang")
        if tag == "h1":
            self.h1 += 1
        if tag == "svg":
            self.svg += 1
        if tag == "title" and not self.svg:
            self._in_title = True
        if tag == "meta" and a.get("name") == "description":
            self.desc = a.get("content", "")
        if tag == "meta" and a.get("name") == "viewport":
            self.viewport = "width=device-width" in a.get("content", "")
        if tag == "script":
            self._script = [a, ""]
        if tag == "style":
            self._style = ""
        if tag == "label":
            self.in_label += 1
            if a.get("for"):
                self.labels_for.add(a["for"])
        if tag in ("input", "select", "textarea"):
            self.controls.append((tag, a, self.in_label > 0))
        if tag == "a" and "skip" in (a.get("class") or "").split():
            self.skip = a.get("href")
        if tag == "main" and a.get("id") == "main":
            self.main = True
        if tag == "body":
            self.in_head = False

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag == "svg":
            self.svg -= 1
        if tag == "label":
            self.in_label -= 1
        if tag == "script" and self._script is not None:
            self.scripts.append(tuple(self._script))
            self._script = None
        if tag == "style" and self._style is not None:
            self.styles.append(self._style)
            self._style = None

    def handle_data(self, d):
        if self._in_title:
            self.title += d
        if self._script is not None:
            self._script[1] += d
        elif self._style is not None:
            self._style += d
        else:
            self.text.append(d)


FORBIDDEN_APIS = [r"\bfetch\s*\(", r"XMLHttpRequest", r"\bWebSocket\b", r"sendBeacon", r"\bimport\s*\(", r"\beval\s*\(", r"new\s+Function\b", r"\binnerHTML\b", r"\binsertAdjacentHTML\b",
                  r"document\.write", r"\blocalStorage\b", r"\bsessionStorage\b", r"\bindexedDB\b", r"\bEventSource\b", r"\bserviceWorker\b", r"\bimportScripts\b", r"https?://", r"\bdocument\.cookie\b",
                  r"\bnavigator\.sendBeacon\b", r"\bnew\s+Worker\b", r"\bpostMessage\b", r"\bwindow\.open\b"]


def g10_structure(fx, errs, html, path):
    size = len(html.encode())
    css = (path.parent / "assets" / "site.css").stat().st_size
    if size > BUDGET_PAGE:
        errs.append("G10: board.html is %d bytes, over its budget of %d" % (size, BUDGET_PAGE))
    if size + css > BUDGET_WITH_CSS:
        errs.append("G10: board.html and site.css are %d bytes, over %d" % (size + css, BUDGET_WITH_CSS))
    p = Structure()
    p.feed(html)
    if p.h1 != 1:
        errs.append("G10: %d <h1>, want 1" % p.h1)
    if not p.lang:
        errs.append("G10: no lang")
    if p.skip != "#main" or not p.main:
        errs.append("G10: no skip link to <main id=\"main\">")
    if not p.viewport:
        errs.append("G10: no viewport")
    if not (50 <= len(p.title.strip()) <= 60):
        errs.append("G10: the title is %d characters, want 50 to 60" % len(p.title.strip()))
    if not p.desc or len(p.desc) > 160:
        errs.append("G10: the description is missing or over 160 characters")
    js = [(a, t) for a, t in p.scripts if not a.get("type") or a.get("type") == "text/javascript"]
    if len(js) != 1:
        errs.append("G10: %d logic scripts, want exactly one" % len(js))
        return
    code = js[0][1]
    if len(code.encode()) > BUDGET_SCRIPT:
        errs.append("G10: the logic script is %d bytes, over %d" % (len(code.encode()), BUDGET_SCRIPT))
    for a, t in p.scripts:
        if a.get("src"):
            errs.append("G10: a <script src> (%s)" % a["src"])
        if a not in [x for x, _ in js] and a.get("type") not in ("text/plain", "application/ld+json"):
            errs.append("G10: a script of type %r" % a.get("type"))
    for tag, a in p.attrs:
        for k, v in a.items():
            if k.startswith("on"):
                errs.append("G10: an inline handler %s on <%s>" % (k, tag))
            if k == "src":
                errs.append("G10: a src attribute on <%s>: %s" % (tag, v))
            if k == "href" and re.match(r"(https?:)?//", v or "") and tag != "a" and not (tag == "link" and a.get("rel") == "canonical"):
                errs.append("G10: <%s href=%s> reaches another host" % (tag, v))
            if k == "tabindex" and v not in ("0", "-1"):
                errs.append("G10: tabindex %s on <%s> (only 0 and -1)" % (v, tag))
    for pat in FORBIDDEN_APIS:
        m = re.search(pat, code)
        if m:
            errs.append("G10: the script holds %r (%s)" % (m.group(0), pat))
    css_all = "\n".join(p.styles)
    if re.search(r"@import|url\((?!#)", css_all):
        errs.append("G10: the inline stylesheet imports or loads something (@import or url())")
    # no colour literal: every colour is a token of site.css, whose pairs scripts/check_contrast.py measures
    if re.search(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(", css_all):
        errs.append("G10: the inline stylesheet holds a colour literal (use the tokens of site.css)")
    if "prefers-color-scheme" not in (path.parent / "assets" / "site.css").read_text():
        errs.append("G10: site.css has no dark scheme")
    if not re.search(r"@media\s*\(max-width:\s*(40|36|30|23\.?\d*)rem\)|@media\s*\(max-width:\s*\d+px\)", css_all):
        errs.append("G10: the inline stylesheet has no narrow-screen rule")
    for tag, a, wrapped in p.controls:
        if a.get("type") in ("hidden",):
            continue
        if not (wrapped or a.get("id") in p.labels_for or a.get("aria-label") or a.get("aria-labelledby")):
            errs.append("G10: a <%s> with no label (id=%s)" % (tag, a.get("id")))
    # the whole script compiles (the logic is run by the other gates; the DOM code is compiled here and run in the browser)
    try:
        one({"op": "syntax"})
    except RuntimeError as exc:
        errs.append("G10: the script does not compile: %s" % str(exc)[:300])
    # the logic part touches no document
    a = code.find("/* LOGIC begin")
    b = code.find("/* LOGIC end */")
    if a < 0 or b < a:
        errs.append("G10: the script has no LOGIC begin / LOGIC end markers")
    else:
        logic = code[a:b]
        for w in (r"\bdocument\b", r"\bwindow\b", r"\bnavigator\b", r"\blocation\b", r"\bsetTimeout\b", r"\bconsole\b"):
            if re.search(w, logic):
                errs.append("G10: the logic part uses %s (it must run in node)" % w)
        if re.search(r"\bcreateElement\b|\btextContent\b|\bgetElementById\b|\bquerySelector", logic):
            errs.append("G10: the logic part touches the DOM")
    # DOM code writes text only
    if "textContent" not in code:
        errs.append("G10: the DOM code never uses textContent")
    # statements the page must make
    text = " ".join(p.text) + " " + html
    low = re.sub(r"\s+", " ", text.lower())
    for needle, what in (("nothing is sent", "that nothing is sent anywhere"), ("ml runs verify", "the verify command"), ("only the tool", "that only the tool can verify"),
                         ("cannot verify", "that the page cannot verify"), ("real records", "the sample's caption (real records)"), ("sample", "the sample button"),
                         ("ml runs list", "the input it reads: ml runs list"), ("ml runs show", "the input it reads: ml runs show"), ("drop", "drag and drop"), ("paste", "paste"),
                         ("crypto.subtle", "the SHA-256 it uses"), ("error", "the word error for a problem"), ("note", "the word note for a problem")):
        if needle not in low:
            errs.append("G10: the page does not say %s (looked for %r)" % (what, needle))
    for needle in ('id="problems"', 'role="status"', 'type="file"', 'aria-sort', '<caption', 'id="sample"', '<table'):
        if needle not in html and needle != "aria-sort":
            errs.append("G10: the page has no %s" % needle)
    if "aria-sort" not in code:
        errs.append("G10: the table headers carry no aria-sort")
    if "multiple" not in html:
        errs.append("G10: the file input does not take several files")
    for ev in ("dragover", "drop", "paste", "change"):
        if '"%s"' % ev not in code:
            errs.append("G10: no handler for %s" % ev)


GATES = {"G1": g1_sha256, "G2": g2_ids_and_forms, "G3": g3_tamper, "G4": g4_problems, "G5": g5_compare, "G6": g6_table, "G7": g7_primary, "G8": g8_chart}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", default=str(ROOT / "build" / "ml"))
    ap.add_argument("--fixtures")
    ap.add_argument("--make-fixtures")
    ap.add_argument("--only")
    ap.add_argument("--page")
    args = ap.parse_args(argv)
    global PAGE
    if args.page:
        PAGE = Path(args.page)
    if args.make_fixtures:
        make_fixtures(Path(args.bin), args.make_fixtures)
        print("check_board: fixtures written to", args.make_fixtures)
        return 0
    if not PAGE.is_file():
        print("check_board: %s does not exist" % PAGE)
        return 1
    if args.fixtures:
        fx = json.loads((Path(args.fixtures) / "fixtures.json").read_text(encoding="utf-8"))
    else:
        if not Path(args.bin).exists():
            print("check_board: %s does not exist: run `cancho install && cancho build --bin ml` first" % args.bin)
            return 1
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
            fx = make_fixtures(Path(args.bin), td)
    only = set(args.only.split(",")) if args.only else None
    errs = []
    html = PAGE.read_text()
    stats = {}
    for name, fn in GATES.items():
        if only and name not in only:
            continue
        before = len(errs)
        try:
            r = fn(fx, errs)
            if name == "G5":
                stats["compare"] = r
        except RuntimeError as exc:
            errs.append("%s: %s" % (name, exc))
        print("check_board: %s %s" % (name, "ok" if len(errs) == before else "FAILED"))
    if not only or "G9" in only:
        before = len(errs)
        g9_sample(fx, errs, html)
        print("check_board: G9 %s" % ("ok" if len(errs) == before else "FAILED"))
    if not only or "G10" in only:
        before = len(errs)
        g10_structure(fx, errs, html, PAGE)
        print("check_board: G10 %s" % ("ok" if len(errs) == before else "FAILED"))
    for e in errs:
        print("check_board:", e)
    if not errs:
        c = stats.get("compare")
        print("check_board: ok (%d runs and %d evals made by the tool; %s; board.html %d bytes of %d)"
              % (len(fx["runs"]), len(fx["evals"]), ("%d ordered pairs equal `ml runs compare`" % c[2]) if c else "compare not run", len(html.encode()), BUDGET_PAGE))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

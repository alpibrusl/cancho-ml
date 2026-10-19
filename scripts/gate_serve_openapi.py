#!/usr/bin/env python3
"""O1 (docs/serving-http.md section 9): the OpenAPI document `ml-serve` serves for a fixture model is a checked-in file, and a model change is a change to that file.

    python3 scripts/gate_serve_openapi.py            fail when a served document differs from docs/serving/<id>.openapi.json, or does not validate
    python3 scripts/gate_serve_openapi.py --update   write the files (after reading what changed)

The fixture models are fitted by `ml fit` from the seeded data of tests/conformance/serving_corpus.py, so their files, and the SHA-256 the document names, are the same on every run of the same
sources. Each document is served by a real process (`build/ml-serve`, or $ML_SERVE_BIN), read over HTTP, and written in a canonical form (two-space indent, keys in the order the server wrote them,
a final newline). It is validated with `openapi-spec-validator` when that is importable (the virtual environment build/venv of this clone is tried when it is not), and the check says plainly when it
could not be: the comparison with the checked-in file still runs.
"""

import difflib
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
OUT = ROOT / "docs" / "serving"

# id -> (model id of the corpus, extra flags, guarded)
FIXTURES = {
    "regression-linear": ("linear", [], False),
    "regression-linear-guarded": ("linear", [], True),
    "classification-logistic": ("logistic", [], False),
    "clustering-kmeans": ("kmeans", [], False),
    "categorical-onehot": ("cat_linear_onehot", [], False),
    "missing-cells-forest": ("cat_forest_hash", [], False),
    "constant-classification": ("const_cls", [], False),
    "classification-boost": ("boost_cls", [], False),
    "categorical-boost": ("cat_boost_hash", [], False),
}


def validator():
    try:
        import openapi_spec_validator as v

        return v
    except ImportError:
        pass
    venv = ROOT / "build" / "venv" / "bin" / "python3"
    if venv.exists():
        r = subprocess.run([str(venv), "-c", "import openapi_spec_validator"], capture_output=True)
        if r.returncode == 0:
            return str(venv)
    return None


def validate(doc, v):
    if v is None:
        return "openapi-spec-validator is not importable: the document was not validated"
    if isinstance(v, str):
        r = subprocess.run([v, "-c", "import json,sys,openapi_spec_validator as v; v.validate(json.load(sys.stdin))"], input=json.dumps(doc).encode(), capture_output=True)
        return None if r.returncode == 0 else r.stderr.decode()[-600:]
    try:
        v.validate(doc)
    except Exception as e:  # noqa: BLE001
        return repr(e)[:600]
    return None


def served_documents():
    from serve_lib import Server, corpus

    sb, models, _ = corpus()
    docs = {}
    try:
        for fid, (mid, flags, guarded) in FIXTURES.items():
            kw = {"token": "fixture-token"} if guarded else {}
            with Server(sb.dir, models[mid][0], *flags, **kw) as srv:
                headers = {"Authorization": "Bearer fixture-token"} if guarded else {}
                r = srv.request("GET", "/openapi.json", headers=headers)
                assert r.status == 200, r
                docs[fid] = r.text()
    finally:
        sb.cleanup()
    return docs


def canonical(text):
    return json.dumps(json.loads(text), indent=2, ensure_ascii=False) + "\n"


def main():
    update = "--update" in sys.argv
    v = validator()
    bad = 0
    docs = served_documents()
    for fid, text in docs.items():
        doc = json.loads(text)
        want = canonical(text)
        path = OUT / (fid + ".openapi.json")
        why = validate(doc, v)
        if why:
            print("%s: %s" % (fid, why))
            if v is not None:
                bad += 1
        if update:
            OUT.mkdir(exist_ok=True)
            path.write_text(want)
            print("wrote %s" % path.relative_to(ROOT))
            continue
        if not path.exists():
            print("FAIL %s: %s is not there (run with --update)" % (fid, path.relative_to(ROOT)))
            bad += 1
        elif path.read_text() != want:
            print("FAIL %s: the served document is not %s" % (fid, path.relative_to(ROOT)))
            sys.stdout.writelines(list(difflib.unified_diff(path.read_text().splitlines(True), want.splitlines(True), "checked in", "served", n=1))[:40])
            bad += 1
        else:
            print("ok %s%s" % (fid, "" if v is not None else " (not validated)"))
    stale = sorted(p.name for p in OUT.glob("*.openapi.json") if p.name[: -len(".openapi.json")] not in FIXTURES) if OUT.exists() else []
    if stale and not update:
        print("FAIL files with no fixture: %s" % stale)
        bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

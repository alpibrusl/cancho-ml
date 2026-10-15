#!/usr/bin/env python3
"""O2 (docs/serving-http.md section 9): Schemathesis against the OpenAPI document `ml-serve` serves, for fixture models.

    python3 scripts/gate_serve_schemathesis.py [--max-examples N] [--require] [FIXTURE ...]

Schemathesis is looked for on PATH and in the virtual environment build/venv of this clone (`python3 -m venv build/venv && build/venv/bin/pip install schemathesis openapi-spec-validator`). When it is not
there the script says so and exits 0, or 1 with --require. Every check runs, with these settings and no others:

* the seed is fixed, `--max-examples` is 100 for each operation;
* `negative_data_rejection` expects 413 among the rejections, as it does 400 and 422 (a batch with more records than the document's `maxItems` is 413, and 413 is a rejection);
* for a model that has a `dec` column, `positive_data_acceptance` is not run: cancho-schema at the pin states a `number` with no bounds and no multiple, so the document cannot say that a `dec` column
  takes at most its scale's digits and 12 in all, and Schemathesis (rightly) sends numbers the server refuses with 422 `serve.number-not-exact`. The script prints this as the one exclusion.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

# fixture id -> (model id of the corpus, token, has a dec column)
FIXTURES = {
    "categorical-onehot": ("cat_linear_onehot", None, False),
    "missing-cells-forest": ("cat_forest_hash", None, False),
    "regression-linear": ("linear", None, True),
    "classification-logistic": ("logistic", None, True),
    "clustering-kmeans": ("kmeans", None, True),
    "regression-linear-guarded": ("linear", "fixture-token", True),
}

CONFIG = """[checks.negative_data_rejection]
expected-statuses = ["400", "401", "403", "404", "405", "406", "409", "413", "415", "422", "428", "429", "5xx"]
"""


def tool():
    p = shutil.which("schemathesis")
    if p:
        return p
    v = ROOT / "build" / "venv" / "bin" / "schemathesis"
    return str(v) if v.exists() else None


def main():
    args = sys.argv[1:]
    require = "--require" in args
    examples = "100"
    if "--max-examples" in args:
        examples = args[args.index("--max-examples") + 1]
    wanted = [a for a in args if a in FIXTURES]
    st = tool()
    if st is None:
        print("schemathesis is not installed (python3 -m venv build/venv && build/venv/bin/pip install schemathesis): not run")
        return 1 if require else 0
    from serve_lib import Server, corpus

    version = subprocess.run([st, "--version"], capture_output=True, text=True).stdout.strip()
    sb, models, _ = corpus()
    work = pathlib.Path(tempfile.mkdtemp(prefix="st-"))
    (work / "schemathesis.toml").write_text(CONFIG)
    bad = 0
    try:
        for fid, (mid, token, has_dec) in FIXTURES.items():
            if wanted and fid not in wanted:
                continue
            kw = {"token": token} if token else {}
            with Server(sb.dir, models[mid][0], **kw) as srv:
                cmd = [st, "--config-file", str(work / "schemathesis.toml"), "run", "http://127.0.0.1:%d/openapi.json" % srv.port, "--seed", "20261015", "--max-examples", examples,
                       "--generation-database", "none", "--no-color", "--report-dir", str(work / "report")]
                if has_dec:
                    cmd += ["--exclude-checks", "positive_data_acceptance"]
                if token:
                    cmd += ["-H", "Authorization: Bearer " + token]
                p = subprocess.run(cmd, capture_output=True, text=True, cwd=work)
                tail = [l for l in p.stdout.splitlines() if l.strip()][-14:]
                print("== %s (%s)%s: exit %d" % (fid, version, ", positive_data_acceptance excluded (dec column)" if has_dec else "", p.returncode))
                if p.returncode != 0:
                    print(p.stdout[-6000:])
                    bad += 1
                else:
                    print("\n".join("   " + l for l in tail))
                if not srv.alive():
                    print("FAIL the server ended")
                    bad += 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
        sb.cleanup()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

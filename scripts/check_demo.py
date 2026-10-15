#!/usr/bin/env python3
"""The demo script (scripts/demo.sh, README "Serving over HTTP") is a claim about the built binaries, so it is run and its key facts are asserted.

    python3 scripts/check_demo.py [--bin build]     # run scripts/demo.sh with BIN=build; exit 1 with the reasons when a fact fails
    python3 scripts/check_demo.py --mutants         # each fact is shown able to fail: an edited demo must be refused

The facts, in the order the demo prints them:

  D1  the demo exits 0 and takes under 60 seconds
  D2  two experiments were tracked: two different 64-digit run ids, and `runs compare` names the plan as the input that differs
  D3  `runs verify` prints `verified: True` and the SHA-256 of the model file
  D4  `runs show` prints the same SHA-256 (the run record names the model file by it)
  D5  `eval` on the training file prints `on: train` and on another file `on: other`
  D6  ml-serve started on the free port the demo chose, printed that it listens on 127.0.0.1 and printed no other address
  D7  the first curl answer is a prediction (one row) whose `model_sha256` is the SHA-256 of D3 and D4
  D8  the second curl answer is the refusal `ml.schema.mismatch`, status 422, also carrying that `model_sha256`
  D9  after the demo has ended nothing listens on that port (it stopped the server it started)
"""
import argparse
import os
import pathlib
import re
import socket
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DEMO = HERE / "demo.sh"
HEX64 = r"[0-9a-f]{64}"


def run_demo(script, binary, tmp):
    """-> (exit status, output, seconds). The demo runs in its own TMPDIR."""
    env = dict(os.environ, BIN=str(binary), DATA=str(ROOT / "docs" / "examples" / "data"), TMPDIR=str(tmp))
    t0 = time.time()
    p = subprocess.run(["sh", str(script)], capture_output=True, text=True, env=env, timeout=120)
    return p.returncode, p.stdout + p.stderr, time.time() - t0


def listening(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def facts(rc, out, secs):
    """-> (list of failures, the port the demo used or 0)."""
    bad = []
    if rc != 0:
        bad.append("D1 the demo exited %d" % rc)
    if secs >= 60:
        bad.append("D1 the demo took %.1f seconds" % secs)
    ids = re.findall(r"^run (%s)$" % HEX64, out, re.M)
    if len(ids) != 2 or ids[0] == ids[1]:
        bad.append("D2 expected two different run ids, got %r" % ids)
    if not re.search(r"^differs: plan .*l2=0\.1.* -> .*l2=1\.0", out, re.M):
        bad.append("D2 `runs compare` does not name the plan with l2 0.1 -> 1.0 as the input that differs")
    v = re.findall(r"^verified: True  model (%s)$" % HEX64, out, re.M)
    if len(v) != 1:
        bad.append("D3 expected one `verified: True  model <sha256>` line, got %d" % len(v))
    sha = v[0] if v else None
    r = re.findall(r"^record: model (%s)$" % HEX64, out, re.M)
    if len(r) != 1 or (sha and r[0] != sha):
        bad.append("D4 the run record's model sha256 %r is not the verified one %r" % (r, sha))
    if not re.search(r"^iris_train\.csv  on: train  ", out, re.M):
        bad.append("D5 no `on: train` line for iris_train.csv")
    if not re.search(r"^iris_test\.csv  on: other  ", out, re.M):
        bad.append("D5 no `on: other` line for iris_test.csv")
    m = re.search(r"^\$ ml-serve --model-file iris\.ml --port (\d+)", out, re.M)
    port = int(m.group(1)) if m else 0
    if not m:
        bad.append("D6 the demo does not start `ml-serve --model-file iris.ml --port N` with the default bind")
    ls = re.findall(r"^listening on (\S+)$", out, re.M)
    if not (m and ls == ["127.0.0.1:%d" % port]):
        bad.append("D6 expected `listening on 127.0.0.1:%s`, got %r" % (port, ls))
    if "0.0.0.0" in out:
        bad.append("D6 the output names 0.0.0.0")
    pred = re.search(r'^\{"model_sha256":"(%s)".*"row_count":1,' % HEX64, out, re.M)
    if not pred:
        bad.append("D7 the first curl did not answer a one-row prediction")
    elif sha and pred.group(1) != sha:
        bad.append("D7 the served model_sha256 %s is not the verified %s" % (pred.group(1), sha))
    ref = re.search(r'^\{"rule":"ml\.schema\.mismatch","model_sha256":"(%s)","type":"about:blank","title":"[^"]*","status":422,' % HEX64, out, re.M)
    if not ref:
        bad.append("D8 the second curl is not the refusal ml.schema.mismatch with status 422")
    elif sha and ref.group(1) != sha:
        bad.append("D8 the refusal names model_sha256 %s, not %s" % (ref.group(1), sha))
    if port and listening(port):
        bad.append("D9 something still listens on port %d after the demo" % port)
    return bad, port


def check(script, binary):
    with tempfile.TemporaryDirectory(prefix="check-demo.") as tmp:
        rc, out, secs = run_demo(script, binary, tmp)
    bad, _ = facts(rc, out, secs)
    return bad, out, secs


# name, old, new: `old` occurs once in scripts/demo.sh
MUTANTS = [
    ("serves-another-model", '"$BIN/ml-serve" --root . --model-file iris.ml', '"$BIN/ml-serve" --root . --model-file iris_l2_1.ml'),
    ("verifies-another-run", 'ml runs verify runs "$A"', 'ml runs verify runs "$C"'),
    ("shows-another-run", 'ml runs show runs "$A"', 'ml runs show runs "$C"'),
    ("held-out-file-is-the-training-file", "for f in iris_train.csv iris_test.csv", "for f in iris_train.csv iris_train.csv"),
    ("the-experiments-are-equal", "--l2 1.0 --split", "--l2 0.1 --split"),
    ("binds-every-address-with-a-token", "SERVE_FLAGS=${SERVE_FLAGS:-}", "SERVE_FLAGS=${SERVE_FLAGS:---bind all --token-file serve.token}"),
    ("binds-every-address-without-a-token", "SERVE_FLAGS=${SERVE_FLAGS:-}", "SERVE_FLAGS=${SERVE_FLAGS:---bind all}"),
    ("the-refusal-request-is-valid", """-d '{"sepal_length_cm":"abc"}'""", """-d '{"sepal_length_cm":5.6,"sepal_width_cm":2.8,"petal_length_cm":4.9,"petal_width_cm":2.0}'"""),
    ("the-server-is-not-stopped", 'kill "$SERVER_PID"', 'true "$SERVER_PID"'),
    ("a-step-fails", 'ml runs compare runs "$A"', 'ml runs compare nowhere "$A"'),
]


def mutants(binary):
    text = DEMO.read_text()
    bad = 0
    with tempfile.TemporaryDirectory(prefix="check-demo-mutants.") as d:
        for name, old, new in MUTANTS:
            if text.count(old) != 1:
                print("MUTANT %s: the pattern occurs %d times in scripts/demo.sh (must be 1)" % (name, text.count(old)), file=sys.stderr)
                bad += 1
                continue
            script = pathlib.Path(d) / (name + ".sh")
            script.write_text(text.replace(old, new))
            with tempfile.TemporaryDirectory(dir=d) as tmp:
                rc, out, secs = run_demo(script, binary, tmp)
            fails, port = facts(rc, out, secs)
            # a mutant that leaves its server up must not leave it up after this run: the one process that listens on the port the demo chose, killed by its pid
            if port and listening(port):
                pids = subprocess.run(["lsof", "-t", "-iTCP:%d" % port, "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()
                for pid in pids:
                    os.kill(int(pid), 15)
            if fails:
                print("killed   %-40s %s" % (name, fails[0]))
            else:
                print("SURVIVED %s" % name)
                bad += 1
    print("check_demo --mutants: %d mutants, %d survived" % (len(MUTANTS), bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", default=str(ROOT / "build"), help="the directory that holds ml and ml-serve")
    ap.add_argument("--mutants", action="store_true")
    a = ap.parse_args()
    binary = pathlib.Path(a.bin).resolve()
    for exe in ("ml", "ml-serve"):
        if not (binary / exe).exists():
            print("check_demo: %s is not built in %s" % (exe, binary), file=sys.stderr)
            return 2
    if a.mutants:
        return mutants(binary)
    bad, out, secs = check(DEMO, binary)
    if bad:
        print("check_demo: FAILED\n  " + "\n  ".join(bad), file=sys.stderr)
        print("---- output of the demo ----\n" + out, file=sys.stderr)
        return 1
    print("check_demo: ok (%d facts hold in %.1f s)" % (9, secs))
    return 0


if __name__ == "__main__":
    sys.exit(main())

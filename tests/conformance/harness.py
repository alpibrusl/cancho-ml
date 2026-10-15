"""Shared pieces of the conformance gates (docs/cli.md section 1), after cancho-table's harness.

The binary is build/ml, built by `cancho build --bin ml` before the tests run (ML_BIN overrides the path). Every test runs a real
process and judges it from outside: exit status, the bytes on standard output and standard error, the files left behind.
"""

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import jsonschema

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import spec  # noqa: E402

BIN = os.environ.get("ML_BIN") or str(ROOT / "build" / "ml")
TRAPS = {132, 134, 136, 139, -4, -6, -8, -11}

_cache = {}


def binary():
    return BIN


class Result:
    def __init__(self, argv, status, stdout, stderr):
        self.argv = argv
        self.status = status
        self.stdout = stdout
        self.stderr = stderr

    def doc(self):
        lines = self.stdout.decode("utf-8").splitlines()
        assert len(lines) == 1, "a document tool writes one line: %r (stderr %r)" % (self.stdout[:300], self.stderr[:200])
        return json.loads(lines[0])

    def data(self):
        return self.doc().get("data")

    def rules(self):
        return [e["rule"] for e in self.doc().get("errors", [])]

    def first_rule(self):
        r = self.rules()
        return r[0] if r else None

    def error(self):
        return self.doc()["error"]

    def __repr__(self):
        return "<ml %r -> %d %r %r>" % (self.argv[1:], self.status, self.stdout[:300], self.stderr[:120])


PREDICT_BIN = os.environ.get("ML_PREDICT_BIN") or str(ROOT / "build" / "ml-predict")


def runs_here(c):
    """A case marked darwin_only has a platform-specific answer (the errno of a refusal, or what /dev/stdin is when descriptor 0 is closed): it is run on macOS only
    (docs/ci.md section 5). Every other case runs everywhere."""
    return not c.get("darwin_only") or sys.platform == "darwin"


def run(*args, cwd=None, timeout=120, close_stdin=False, close_stdout=False, binary_path=None, stdin=None):
    """stdin: the bytes to give the process on standard input (None: the process gets an empty one, /dev/null)."""
    argv = [binary_path or BIN, *[str(a) for a in args]]

    def child():
        if close_stdin:
            os.close(0)
        if close_stdout:
            os.close(1)

    p = subprocess.run(argv, capture_output=True, cwd=cwd, timeout=timeout, preexec_fn=child if (close_stdin or close_stdout) else None,
                       input=stdin if stdin is not None else (None if close_stdin else b""))
    return Result(argv, p.returncode, p.stdout, p.stderr)


def schema():
    if "schema" not in _cache:
        _cache["schema"] = json.loads((ROOT / "schemas" / "ml.v1.json").read_text())
    return _cache["schema"]


def log_schema():
    if "log" not in _cache:
        _cache["log"] = json.loads((ROOT / "schemas" / "log.v1.json").read_text())
    return _cache["log"]


def introspect():
    if "introspect" not in _cache:
        p = subprocess.run([BIN, "introspect"], capture_output=True, check=True)
        _cache["introspect"] = json.loads(p.stdout)
    return _cache["introspect"]


def skill():
    if "skill" not in _cache:
        _cache["skill"] = subprocess.run([BIN, "skill"], capture_output=True, check=True).stdout.decode()
    return _cache["skill"]


def package_catalogue():
    """tag -> (exit, repairable) of the installed contract package's rule catalogue (build/deps)."""
    if "catalogue" in _cache:
        return _cache["catalogue"]
    out = {}
    for f in sorted((ROOT / "build" / "deps").glob("*.cho")):
        text = f.read_text()
        if "module toolbox.rules;" not in text:
            continue
        body = re.search(r'pub fn catalogue\(\) -> \[\] &static \[byte\] \{\s*return "(.*?)";', text, re.S).group(1)
        for entry in body.split(";"):
            tag, code, repairable, summary = entry.split("|")
            out[tag] = (int(code), repairable)
    _cache["catalogue"] = out
    return out


def expected_exit(tag):
    """The exit code the tables give a tag: ml's own (spec.cho) or the package's."""
    t = spec.tables()
    if tag in t["extra"]:
        return t["extra"][tag]["exit"]
    return package_catalogue()[tag][0]


def validate(result, expect_stderr_empty=True):
    """The output is valid against the schema, the status is declared, nothing is on stderr in JSON mode. A list of problems."""
    problems = []
    if result.status in TRAPS or result.status < 0:
        return ["trapped: status %d" % result.status]
    declared = {c["code"] for c in introspect()["exit_codes"]}
    if result.status not in declared:
        problems.append("status %d not in the declared table %s" % (result.status, sorted(declared)))
    try:
        text = result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return problems + ["stdout is not UTF-8"]
    if not text.endswith("\n"):
        problems.append("output does not end with a newline")
    lines = text.splitlines()
    if len(lines) != 1:
        problems.append("a document tool wrote %d lines" % len(lines))
    validator = jsonschema.Draft202012Validator(schema())
    for line in lines:
        try:
            value = json.loads(line)
        except ValueError as e:
            problems.append("not JSON: %s" % e)
            continue
        for err in validator.iter_errors(value):
            problems.append("schema: %s at %s" % (err.message[:200], list(err.absolute_path)))
        if value.get("ok") is False and result.status in (0, 9):
            problems.append("ok:false with exit %d" % result.status)
        if value.get("ok") is True and result.status not in (0, 9):
            problems.append("ok:true with exit %d" % result.status)
        if value.get("errors") and value["error"] != value["errors"][0]:
            problems.append("error is not errors[0]")
        if value.get("errors"):
            want = expected_exit(value["error"]["rule"])
            if result.status != want:
                problems.append("rule %s has exit %d in the tables, the process exited %d" % (value["error"]["rule"], want, result.status))
    if expect_stderr_empty and result.stderr:
        problems.append("stderr is not empty in JSON mode: %r" % result.stderr[:200])
    return problems


class Sandbox:
    """A scratch directory that is the --root of a test."""

    def __init__(self):
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="ml-"))

    def write(self, name, data):
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        return p

    def read(self, name):
        return (self.dir / name).read_bytes()

    def run(self, *args, **kw):
        return run("--root", self.dir, *args, **kw)

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


# --------------------------------------------------------------------------------------------------- the model file

SECTION_TAGS = (1, 2, 3, 4, 5, 6)


def sections(data):
    """[(tag, payload_start, payload_len)] of a model file's sections: five, or six when the file has a string table."""
    out = []
    pos = 12
    for _ in range(int.from_bytes(data[8:12], "little")):
        tag = int.from_bytes(data[pos:pos + 4], "little")
        n = int.from_bytes(data[pos + 4:pos + 12], "little")
        out.append((tag, pos + 12, n))
        pos += 12 + n
    return out


def fix_checksum(data):
    body = bytes(data[:-32])
    return body + hashlib.sha256(body).digest()


def set_bytes(data, at, new):
    b = bytearray(data)
    b[at:at + len(new)] = new
    return bytes(b)

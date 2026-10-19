"""Shared pieces of the JSON lines gates (docs/jsonl.md section 6): the driver `jsonl-check`, the reference (jsonl_ref.py) and a compare. Standard library only."""
import os
import pathlib
import subprocess
import tempfile

import jsonl_ref

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECK = os.environ.get("JSONL_CHECK_BIN") or str(ROOT / "build" / "jsonl-check")


def run_check(data, fields, cats=(), absent_empty=False, max_nodes=4096, max_line=1048576, max_bytes=16777216, timeout=60):
    """("ok", csv bytes) or ("refused", (code, line, row, col, a, b, c)) from the driver; a trap or an unexpected status fails the caller"""
    with tempfile.TemporaryDirectory(prefix="jl-") as d:
        p = pathlib.Path(d) / "in.jsonl"
        p.write_bytes(data)
        argv = [CHECK, "--fields", ",".join(fields), "--max-json-nodes", str(max_nodes), "--max-line-bytes", str(max_line), "--max-bytes", str(max_bytes)]
        if cats:
            argv += ["--cat", ",".join(cats)]
        if absent_empty:
            argv += ["--absent", "empty"]
        argv.append(str(p))
        r = subprocess.run(argv, capture_output=True, timeout=timeout)
    if r.returncode == 0:
        return "ok", r.stdout
    if r.returncode == 8 and r.stdout.startswith(b"refused "):
        return "refused", tuple(int(x) for x in r.stdout.split()[1:])
    raise AssertionError("jsonl-check exited %d: %r %r" % (r.returncode, r.stdout[:200], r.stderr[:200]))


def agree(got, want):
    """problems (a list of sentences) where the driver's answer is not the reference's"""
    if got[0] != want[0]:
        return ["driver %s, reference %s: %r vs %r" % (got[0], want[0], got[1], want[1])]
    if got[0] == "ok":
        return [] if got[1] == want[1] else ["csv differs: %r vs %r" % (got[1][:300], want[1][:300])]
    g, w = got[1], want[1]
    if w[0] == 1:  # a syntax error: the offset and the parser's code are the parser's, not compared (docs/jsonl.md)
        return [] if g[:4] == w[:4] else ["syntax refusal differs: %r vs %r" % (g, w)]
    return [] if g == w else ["refusal differs: %r vs %r" % (g, w)]


def check(test, data, fields, **kw):
    got = run_check(data, fields, **kw)
    want = jsonl_ref.expected(data, fields, **kw)
    problems = agree(got, want)
    test.assertEqual(problems, [], "input %r fields %r %r" % (data[:200], fields, kw))
    return got

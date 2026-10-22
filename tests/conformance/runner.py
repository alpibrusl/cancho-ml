"""Run one fixture case (fixtures.py) in a scratch directory and judge it. Used by test_tables, test_rules and the fuzz."""

import os

from harness import Sandbox, expected_exit, run, validate


def apply_setup(sb, steps):
    for step in steps:
        kind = step[0]
        if kind == "write":
            sb.write(step[1], step[2])
        elif kind == "fit":
            r = sb.run("fit", *step[1:])
            assert r.status == 0, "setup fit failed: %r" % r
        elif kind == "mutate":
            sb.write(step[2], step[3](sb.read(step[1])))
        elif kind == "mkdir":
            (sb.dir / step[1]).mkdir(parents=True, exist_ok=True)
        elif kind == "symlink":
            os.symlink(sb.dir / step[1], sb.dir / step[2])
        elif kind == "chmod":
            os.chmod(sb.dir / step[1], step[2])
        elif kind == "raw":
            step[1](sb)
        else:
            raise AssertionError("unknown setup step %r" % (kind,))


def first_run_id(sb):
    """the id of the first run record of the directory `runs` of the sandbox ("" when there is none)"""
    d = sb.dir / "runs"
    names = sorted(p.name for p in d.iterdir() if p.name.endswith(".json") and ".eval-" not in p.name and len(p.name) == 69) if d.is_dir() else []
    return names[0][:64] if names else ""


def substitute(argv, sb):
    return [a.replace("@ROOT", str(sb.dir)).replace("@ID", first_run_id(sb)) for a in argv]


def run_case(c, sb=None):
    """(sandbox, result) after the setup and the command of case `c`."""
    sb = sb or Sandbox()
    apply_setup(sb, c["setup"])
    argv = substitute(c["argv"], sb)
    kw = {"close_stdin": c.get("close_stdin", False), "close_stdout": c.get("close_stdout", False)}
    if c.get("noroot"):
        return sb, run(*argv, cwd=sb.dir, **kw)
    return sb, sb.run(*argv, **kw)


def subset(want, got, path=""):
    """Problems where `got` does not contain `want` (dicts by key, lists exactly)."""
    if isinstance(want, dict):
        if not isinstance(got, dict):
            return ["%s: expected an object, got %r" % (path, got)]
        out = []
        for k, v in want.items():
            if k not in got:
                out.append("%s.%s missing" % (path, k))
            else:
                out += subset(v, got[k], path + "." + k)
        return out
    if want != got:
        return ["%s: expected %r, got %r" % (path, want, got)]
    return []


def judge(c, sb, r):
    """A list of problems: the expectation of the case against the result."""
    ex = c["expect"]
    problems = []
    rule = ex["rule"]
    want_exit = ex["exit"]
    if want_exit is None:
        want_exit = expected_exit(rule) if rule else 0
    if r.status != want_exit:
        problems.append("exit %d, expected %d: %r" % (r.status, want_exit, r))
    if c.get("no_json") or ex.get("stdout_starts") is not None:
        pass
    else:
        problems += validate(r)
        if not problems or r.status == want_exit:
            try:
                doc = r.doc()
            except Exception as e:  # noqa: BLE001
                return problems + ["no document: %s" % e]
            if rule is not None and doc.get("error", {}).get("rule") != rule:
                problems.append("first rule %r, expected %r" % (doc.get("error", {}).get("rule"), rule))
            if rule is None and not doc.get("ok"):
                problems.append("expected success, got %r" % doc.get("error"))
            if ex.get("data") is not None:
                problems += subset(ex["data"], doc.get("data"), "data")
            d = doc.get("data")
            if isinstance(d, dict) and isinstance(d.get("rows"), list) and isinstance(d.get("row_count"), int) and len(d["rows"]) != d["row_count"]:
                problems.append("row_count %d but %d rows are written" % (d["row_count"], len(d["rows"])))
            if ex.get("dry_run") and doc.get("dry_run") is not True:
                problems.append("dry_run is not true")
            if ex.get("repair_retry"):
                rep = doc["error"]["repair"]
                if not (rep and rep["kind"] == "retry"):
                    problems.append("expected a retry repair, got %r" % rep)
            if ex.get("repair_runs"):
                rep = doc["error"]["repair"]
                if not (rep and rep["kind"] in ("retry", "choose")):
                    problems.append("expected a retry or choose repair, got %r" % rep)
            if ex.get("repair_has"):
                rep = doc["error"]["repair"]
                argvs = [rep["argv"]] if rep and rep["kind"] == "retry" else [o["argv"] for o in (rep or {}).get("options", [])]
                for pair in ex["repair_has"]:
                    if not any(any(a[i:i + len(pair)] == pair for i in range(len(a))) for a in argvs):
                        problems.append("no repair argv holds %r: %r" % (pair, argvs))
            if ex.get("repair_none"):
                rep = doc["error"]["repair"]
                if not (rep and rep["kind"] == "none" and rep.get("reason")):
                    problems.append("expected a repair of kind none with a reason, got %r" % rep)
            if ex.get("detail") is not None:
                problems += subset(ex["detail"], doc.get("error", {}).get("detail") or {}, "detail")
    if ex.get("stdout_starts") is not None and not r.stdout.decode().startswith(ex["stdout_starts"]):
        problems.append("stdout starts %r, expected %r" % (r.stdout[:80], ex["stdout_starts"]))
    if ex.get("stderr_has") and ex["stderr_has"] not in r.stderr.decode():
        problems.append("stderr %r lacks %r" % (r.stderr[:200], ex["stderr_has"]))
    if ex.get("no_file") and (sb.dir / ex["no_file"]).exists():
        problems.append("%s exists after the command" % ex["no_file"])
    return problems

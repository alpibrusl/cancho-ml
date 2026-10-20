"""The tables of tools/ml/spec.cho, read from the source: the one place the flag table, the rules, the log records and the
limits are written. Everything that must agree with them (the schemas, the gates, the skill checks) reads them through here,
so a table edited in the source is seen by every check at once."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = ROOT / "tools" / "ml" / "spec.cho"


def _literal(text, fn):
    """The string literal returned by `pub fn fn()` (a single return), unescaped."""
    m = re.search(r"pub fn %s\(\)[^{]*\{\s*return \"((?:[^\"\\]|\\.)*)\";" % re.escape(fn), text)
    if not m:
        raise SystemExit("spec.py: no `pub fn %s() ... return \"...\"` in %s" % (fn, SPEC))
    return m.group(1).replace('\\"', '"').replace("\\\\", "\\")


def tables(path=SPEC):
    text = pathlib.Path(path).read_text()
    flags = []
    for entry in _literal(text, "flag_table").split(";"):
        name, short, kind, role, default, help_ = entry.split("|", 5)
        flags.append({"name": name, "short": short, "kind": kind, "role": role, "default": default, "help": help_})
    extra = {}
    for entry in _literal(text, "extra").split(";"):
        tag, exit_, repairable, summary = entry.split("|", 3)
        extra[tag] = {"exit": int(exit_), "repairable": repairable, "summary": summary}
    rules = _literal(text, "rules").split(";")
    log = {}
    for entry in _literal(text, "log_table").split(";"):
        event, fields = entry.split("|")
        log[event] = fields.split(",")
    limits = {}
    for entry in _literal(text, "limits").split(";"):
        name, default, ceiling = entry.split("|")
        limits[name] = (int(default), int(ceiling))
    allowed = {}
    m = re.search(r"pub fn allowed\(verb: int\)[^{]*\{(.*?)\n\}", text, re.S)
    found = re.findall(r"return \"([^\"]*)\";", m.group(1))
    for verb, names in zip(("fit", "predict", "describe", "cv", "runs", "importance", "dump", "eval"), found):
        allowed[verb] = names.split(",")
    pflags = []
    for entry in _literal(text, "predict_flag_table").split(";"):
        name, short, kind, role, default, help_ = entry.split("|", 5)
        pflags.append({"name": name, "short": short, "kind": kind, "role": role, "default": default, "help": help_})
    plimits = {}
    for entry in _literal(text, "predict_limits").split(";"):
        name, default, ceiling = entry.split("|")
        plimits[name] = (int(default), int(ceiling))
    allowed_all = {}
    for verb, names in zip(("fit", "predict", "describe", "cv", "runs", "importance", "dump", "eval"), found):
        allowed_all[verb] = names.split(",")
    tool_m = re.search(r"pub fn tool\(\)[^{]*\{(.*?)\n\}", text, re.S)
    usage = re.search(r'usage: "((?:[^"\\]|\\.)*)"', tool_m.group(1)).group(1).replace("\\n", "\n")
    summary = re.search(r'summary: "((?:[^"\\]|\\.)*)"', tool_m.group(1)).group(1)
    return {"usage": usage, "summary": summary, "allowed_all": allowed_all, "flags": flags, "extra": extra, "rules": rules, "log": log, "limits": limits, "allowed": allowed, "predict_flags": pflags, "predict_limits": plimits}

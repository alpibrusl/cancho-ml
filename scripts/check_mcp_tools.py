#!/usr/bin/env python3
"""Gate B of docs/mcp.md: the tools are what the flag table gives, and they are valid JSON Schema.

    python3 scripts/check_mcp_tools.py              # drift (regenerate and compare), schema validity, generated arguments
    python3 scripts/check_mcp_tools.py --introspect # also compare the flags, kinds, defaults and limits with `ml introspect` of build/ml (ML_BIN)
    python3 scripts/check_mcp_tools.py --mutants    # the check can fail: break the generator in eight ways, each must be refused

1. Drift: scripts/mcp_tools.py regenerates docs/mcp/tools.json and tools/ml-mcp/tools.cho; both must equal the committed bytes.
2. Every inputSchema and outputSchema is a valid Draft 2020-12 schema (jsonschema's meta-validation).
3. For every tool a valid argument set (the required properties; and every property at a boundary value) is accepted by its inputSchema, and a set of invalid ones
   (a wrong type per property, a missing required property, an unknown property, a number above its ceiling) is refused.
4. With --introspect: each property is a flag of `ml introspect` with the same kind, default and ceiling, and every flag a verb takes is a property of its tool.
"""

import copy
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mcp_tools  # noqa: E402

try:
    import jsonschema
except ImportError:  # the numpy-free runner still has jsonschema (ci/requirements-fast.txt)
    sys.exit("check_mcp_tools: jsonschema is needed (pip install -r ci/requirements-fast.txt)")

ROOT = pathlib.Path(__file__).resolve().parent.parent
VALIDATOR = jsonschema.Draft202012Validator


def sample(prop, boundary):
    t = prop.get("type")
    if "enum" in prop:
        return prop["enum"][-1] if boundary else prop["enum"][0]
    if t == "integer":
        return prop.get("maximum", 1) if boundary else prop.get("minimum", 0)
    if t == "boolean":
        return True
    if t == "array":
        return [{}]
    if "pattern" in prop:
        return "a" * 64
    return "x"


def wrong(prop):
    t = prop.get("type")
    out = []
    if t != "string" or "enum" in prop:
        out.append(("a number where a string is wanted" if t == "string" else "a string", "oops"))
    if t == "string":
        out.append(("a number", 7))
    if "enum" in prop:
        out.append(("a value outside the enum", "zzz-not-in-enum"))
    if t == "integer":
        out.append(("a float", 1.5))
        out.append(("negative", -1))
        if "maximum" in prop:
            out.append(("above the ceiling", prop["maximum"] + 1))
    if t == "boolean":
        out.append(("a string", "true"))
    if t == "array":
        out.append(("an object", {}))
    if "pattern" in prop:
        out.append(("not a run id", "XYZ"))
    return out


def base_args(schema):
    args = {}
    for r in schema.get("required", []):
        args[r] = sample(schema["properties"][r], False)
    if "oneOf" in schema:
        args.setdefault("file", "x.csv") if "file" in schema["properties"] else None
    return args


def check_tools(d, errors):
    names = [t["name"] for t in d["tools"]]
    if names != mcp_tools.ORDER:
        errors.append("tool names/order %r != %r" % (names, mcp_tools.ORDER))
    for lst in ("tools", "readonly_tools"):
        for t in d[lst]:
            for key in ("inputSchema", "outputSchema"):
                try:
                    VALIDATOR.check_schema(t[key])
                except jsonschema.SchemaError as e:
                    errors.append("%s %s of %s is not a valid schema: %s" % (lst, key, t["name"], e.message[:120]))
    for t in d["tools"]:
        try:
            VALIDATOR.check_schema(t["inputSchema"])
        except jsonschema.SchemaError:
            continue
        s = t["inputSchema"]
        v = VALIDATOR(s)
        ok = base_args(s)
        if list(v.iter_errors(ok)):
            errors.append("%s: the required arguments %r are refused: %s" % (t["name"], ok, [e.message for e in v.iter_errors(ok)][:2]))
            continue
        for boundary in (False, True):
            full = dict(ok)
            for p, ps in s["properties"].items():
                if p == "records" and "file" in full:
                    continue
                full[p] = sample(ps, boundary)
            if "records" in full and "file" in full:
                del full["records"]
            errs = [e.message for e in v.iter_errors(full)]
            if errs:
                errors.append("%s: a valid argument set (boundary=%s) is refused: %s" % (t["name"], boundary, errs[:2]))
        for p, ps in s["properties"].items():
            for why, val in wrong(ps):
                bad = dict(ok)
                bad[p] = val
                if not list(v.iter_errors(bad)):
                    errors.append("%s: %s=%r (%s) is accepted" % (t["name"], p, val, why))
        for r in s.get("required", []):
            bad = {k: x for k, x in ok.items() if k != r}
            if not list(v.iter_errors(bad)):
                errors.append("%s: missing required %s is accepted" % (t["name"], r))
        bad = dict(ok)
        bad["no-such-property"] = 1
        if not list(v.iter_errors(bad)):
            errors.append("%s: an unknown property is accepted" % t["name"])


def check_introspect(d, errors):
    binary = os.environ.get("ML_BIN") or str(ROOT / "build" / "ml")
    doc = json.loads(subprocess.run([binary, "introspect"], capture_output=True, check=True).stdout)
    flags = {f["name"][2:]: f for f in doc["flags"]}
    limits = {x["name"]: x for x in doc["limits"]}
    for t in d["tools"]:
        c = d["calls"][t["name"]]
        for p in c["props"]:
            f = flags.get(p["name"])
            if f is None:
                errors.append("%s: property %s is not a flag of introspect" % (t["name"], p["name"]))
                continue
            if f["kind"] != p["kind"] or f["role"] != p["role"]:
                errors.append("%s: %s kind/role %s/%s != introspect %s/%s" % (t["name"], p["name"], p["kind"], p["role"], f["kind"], f["role"]))
            ps = t["inputSchema"]["properties"][p["name"]]
            if f["default"] is not None and str(ps.get("default")) != str(f["default"]):
                errors.append("%s: %s default %r != introspect %r" % (t["name"], p["name"], ps.get("default"), f["default"]))
            if p["name"] in limits and ps.get("maximum") != limits[p["name"]]["ceiling"]:
                errors.append("%s: %s maximum %r != introspect ceiling %r" % (t["name"], p["name"], ps.get("maximum"), limits[p["name"]]["ceiling"]))
            if ps.get("description") != f["help"]:
                errors.append("%s: %s help differs from introspect" % (t["name"], p["name"]))
    # every flag the table lists is the property of some tool (or one the server owns)
    covered = {p["name"] for c in d["calls"].values() for p in c["props"]} | mcp_tools.OWNED
    for name in flags:
        if name not in covered:
            errors.append("flag --%s of introspect is the property of no tool" % name)


def run(argv):
    errors = []
    d = mcp_tools.build()
    js = json.dumps(d, indent=1, ensure_ascii=False) + "\n"
    cho = mcp_tools.render_cho(d)
    for path, want in ((mcp_tools.OUT_JSON, js), (mcp_tools.OUT_CHO, cho)):
        have = path.read_text() if path.exists() else None
        if have != want:
            errors.append("stale: %s differs from the regeneration (python3 scripts/mcp_tools.py)" % path.relative_to(ROOT))
    committed = json.loads(mcp_tools.OUT_JSON.read_text()) if mcp_tools.OUT_JSON.exists() else d
    check_tools(committed, errors)
    if "--introspect" in argv:
        check_introspect(committed, errors)
    return errors, d


MUTANTS = [
    ("a flag dropped from the generator", lambda d: d["tools"][0]["inputSchema"]["properties"].pop("seed")),
    ("a kind mapped to the wrong type", lambda d: d["tools"][0]["inputSchema"]["properties"]["seed"].update(type="string")),
    ("an enum value changed", lambda d: d["tools"][0]["inputSchema"]["properties"]["model"]["enum"].__setitem__(0, "constan")),
    ("a ceiling off by one", lambda d: d["tools"][0]["inputSchema"]["properties"]["trees"].update(maximum=1025)),
    ("a required property dropped", lambda d: d["tools"][0]["inputSchema"]["required"].remove("output")),
    ("additionalProperties allowed", lambda d: d["tools"][1]["inputSchema"].update(additionalProperties=True)),
    ("an invalid schema (type misspelt)", lambda d: d["tools"][2]["inputSchema"]["properties"]["seed"].update(type="integr")),
    ("a tool removed", lambda d: d["tools"].pop(3)),
]


def mutants():
    d0 = mcp_tools.build()
    killed = 0
    for name, mutate in MUTANTS:
        d = copy.deepcopy(d0)
        mutate(d)
        errors = []
        check_tools(d, errors)
        if (ROOT / "build" / "ml").exists():
            check_introspect(d, errors)
        # the drift half: the mutated tools differ from the committed bytes
        if json.dumps(d, indent=1, ensure_ascii=False) + "\n" != json.dumps(d0, indent=1, ensure_ascii=False) + "\n":
            errors.append("drift")
        status = "KILLED" if errors else "SURVIVED"
        killed += bool(errors)
        print("%-40s %s%s" % (name, status, "" if not errors else "  (%s)" % errors[0][:70]))
    print("mcp tool mutants: %d of %d killed" % (killed, len(MUTANTS)))
    return killed == len(MUTANTS)


def main():
    if "--mutants" in sys.argv:
        sys.exit(0 if mutants() else 1)
    errors, d = run(sys.argv)
    for e in errors:
        print("FAIL", e)
    if errors:
        sys.exit(1)
    print("mcp tools: %d tools (%d read-only), drift, schema validity and generated arguments pass%s" % (len(d["tools"]), len(d["readonly_tools"]), "; equal to introspect" if "--introspect" in sys.argv else ""))


if __name__ == "__main__":
    main()

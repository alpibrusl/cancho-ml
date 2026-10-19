#!/usr/bin/env python3
"""Derive the tools of `ml-mcp` (docs/mcp.md section 2) from the flag table of tools/ml/spec.cho and the schemas of schemas/ml.v1.json.

Nothing here names a flag. The verbs and the `runs` subcommands come from the `usage` lines of the table, the properties of a tool from `allowed(verb)`
and the kind of each flag, the required ones from the usage line (a flag written outside brackets), the ceilings from `limits`, the help from the
table, the output schema from the variants of `data` in schemas/ml.v1.json (chosen by the names of their members).

    python3 scripts/mcp_tools.py            # write docs/mcp/tools.json and tools/ml-mcp/tools.cho
    python3 scripts/mcp_tools.py --check    # exit 1 if either committed file differs from the regeneration
"""

import copy
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import spec  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "docs" / "mcp" / "tools.json"
OUT_CHO = ROOT / "tools" / "ml-mcp" / "tools.cho"
REVISION = "2025-06-18"
DIALECT = "https://json-schema.org/draft/2020-12/schema"

# flags the server owns: the root is its own, the answer is always the document (the text form is content[0]), the log is the CLI's own record of a run
OWNED = {"root", "format", "log", "log-max-bytes"}
ID_PATTERN = "^[0-9a-f]{64}$"
# the order tools are listed in
ORDER = ["fit", "predict", "eval", "cv", "describe", "introspect", "runs_list", "runs_show", "runs_compare", "runs_verify"]


def usage_lines(usage):
    """{verb: line} for the lines `verb:  ml ...`."""
    out = {}
    for line in usage.split("\n"):
        m = re.match(r"^([a-z]+):\s+ml (.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def required_flags(line):
    """The flags a usage line writes outside brackets."""
    depth, cur, outside = 0, "", ""
    for ch in line:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif depth == 0:
            outside += ch
    return re.findall(r"--([a-z-]+)", outside)


def runs_forms(line):
    """[(sub, [operand names])] from `runs list DIR | runs show DIR ID | ...`."""
    forms = []
    for part in re.findall(r"runs ([a-z]+)((?: [A-Z]+)*)", line):
        forms.append((part[0], part[1].split()))
    return forms


def prop_schema(flag, limits, desc_prefix=""):
    kind = flag["kind"]
    name = flag["name"]
    help_ = flag["help"]
    s = {}
    if kind == "nat":
        s["type"] = "integer"
        s["minimum"] = 0
        if name in limits:
            s["maximum"] = limits[name][1]
    elif kind == "bool":
        s["type"] = "boolean"
    elif kind.startswith("choice:"):
        s["type"] = "string"
        s["enum"] = kind[len("choice:"):].split("/")
    else:  # text, path
        s["type"] = "string"
        s["minLength"] = 1
    if flag["default"]:
        d = flag["default"]
        s["default"] = int(d) if kind == "nat" else d
    s["description"] = help_
    return s


def writes(flag):
    return flag["role"] in ("path-write", "guard")


def data_variants(schema, verb):
    """The variants of $defs.data that belong to a verb, by the names of their members."""
    keep = {
        "fit": lambda p: "written" in p,
        "predict": lambda p: "row_count" in p,
        "eval": lambda p: "metrics" in p and "on" in p,
        "cv": lambda p: "folds" in p and "results" in p,
        "describe": lambda p: "provenance" in p and "request" in p,
        "runs_list": lambda p: "runs" in p and "count" in p,
        "runs_show": lambda p: "record" in p,
        "runs_compare": lambda p: "model_equal" in p,
        "runs_verify": lambda p: "verified" in p,
    }[verb]
    return [v for v in schema["$defs"]["data"]["oneOf"] if keep(v.get("properties", {}))]


def refs(node, acc):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "$ref":
                acc.add(v.split("/")[-1])
            else:
                refs(v, acc)
    elif isinstance(node, list):
        for v in node:
            refs(v, acc)


def own_rules():
    """The tags `ml-mcp` raises itself (tools/ml-mcp/mspec.cho `extra()`): they are rules of the answer of a tool beside the rules of ml."""
    text = (ROOT / "tools" / "ml-mcp" / "mspec.cho").read_text()
    m = re.search(r'pub fn extra\(\)[^{]*\{\s*return "((?:[^"\\]|\\.)*)";', text)
    return [entry.split("|")[0] for entry in m.group(1).split(";")]


def output_schema(schema, verb):
    """The envelope of ml.v1 with `data` narrowed to the verb's variants; the definitions it reaches, no others."""
    if verb == "introspect":
        return {"$schema": DIALECT, "type": "object",
                "description": "the tool as data: the document `ml introspect` prints (flags, rules, limits, authority, schema)",
                "required": ["tool", "version", "flags", "rules", "limits", "authority"]}
    defs = copy.deepcopy(schema["$defs"])
    defs["rule"]["enum"] = defs["rule"]["enum"] + [r for r in own_rules() if r not in defs["rule"]["enum"]]
    variants = data_variants(schema, verb)
    assert variants, verb
    defs["data"] = {"oneOf": variants} if len(variants) > 1 else variants[0]
    env = {k: copy.deepcopy(v) for k, v in schema.items() if k not in ("$defs", "$id", "title")}
    env["$schema"] = DIALECT
    env["title"] = "ml.v1 " + verb
    need = set()
    refs(env, need)
    seen = set()
    while need - seen:
        n = (need - seen).pop()
        seen.add(n)
        refs(defs[n], need)
    env["$defs"] = {k: defs[k] for k in sorted(need)}
    return env


def build():
    t = spec.tables()
    schema = json.loads((ROOT / "schemas" / "ml.v1.json").read_text())
    flags = {f["name"]: f for f in t["flags"]}
    limits = t["limits"]
    lines = usage_lines(t["usage"])
    tools = []
    calls = {}
    summary = t["summary"].split(". ")[0] + "."
    for name in ORDER:
        if name == "introspect":
            tools.append({"name": "introspect", "title": "introspect",
                          "description": "The tool as data: every flag with its kind, default and help, the rules with their exit codes, the limits with their ceilings, the authority row and the schemas. " + summary,
                          "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
                          "outputSchema": output_schema(schema, name),
                          "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}})
            calls[name] = {"verb": "introspect", "sub": None, "operands": [], "props": [], "writes": [], "needs_write": False}
            continue
        sub = None
        verb = name
        operand_names = []
        if name.startswith("runs_"):
            verb, sub = "runs", name[5:]
            forms = dict(runs_forms(lines["runs"]))
            operand_names = forms[sub]
            allowed = []
            for f in t["allowed_all"]["runs"]:
                h = flags[f]["help"]
                if "runs" in h and ("runs " + sub) not in h:
                    continue
                allowed.append(f)
            line = lines["runs"]
            req = []
        else:
            allowed = list(t["allowed_all"][verb])
            line = lines[verb]
            req = required_flags(line)
            if verb == "predict" or verb == "fit" or verb == "eval" or verb == "cv":
                operand_names = ["FILE"]
        props = {}
        order = []
        writing = []
        for f in allowed:
            if f in OWNED:
                continue
            fl = flags[f]
            props[f] = prop_schema(fl, limits)
            order.append(f)
            if writes(fl):
                writing.append(f)
        required = []
        opnames = {"FILE": "file", "DIR": "dir", "ID": "id"}
        op_props = []
        ids = [o for o in operand_names if o == "ID"]
        idx = 0
        for o in operand_names:
            pname = opnames[o]
            if o == "ID" and len(ids) == 2:
                pname = "id_a" if idx == 0 else "id_b"
                idx += 1
            if o == "FILE":
                props[pname] = {"type": "string", "minLength": 1, "description": "the CSV file, relative to the server's root (predict: or give records)" if verb == "predict" else "the CSV file, relative to the server's root"}
            elif o == "DIR":
                props[pname] = {"type": "string", "minLength": 1, "description": "the directory of run records, relative to the server's root"}
            else:
                props[pname] = {"type": "string", "pattern": ID_PATTERN, "description": "a run id: 64 lowercase hexadecimal digits"}
            op_props.append(pname)
            if not (verb == "predict" and o == "FILE"):
                required.append(pname)
        for f in req:
            if f in props and f not in required:
                required.append(f)
        schema_in = {"type": "object", "properties": props, "required": required, "additionalProperties": False}
        if verb == "predict":
            rec = {"type": "array", "minItems": 1, "maxItems": limits["limit"][1], "items": {"type": "object"},
                   "description": "rows to predict inline: objects keyed by the columns the model requires (describe lists them); rendered to the CSV the predictor reads by the byte rule of docs/serving-http.md section 4. Give this or file"}
            props["records"] = rec
            schema_in["oneOf"] = [{"required": ["file"]}, {"required": ["records"]}]
        wr = sorted(set(writing))
        desc = lines[verb if verb != "runs" else "runs"]
        desc = "ml " + desc
        description = "%s Usage of the command line this tool is: `%s`. Values the library refuses (ranges, conflicts between flags, unknown columns) come back as a tool error with the rule, the exit code and the repair." % (summary, desc if verb != "runs" else "ml runs %s ..." % sub)
        tools.append({"name": name, "title": name.replace("_", " "), "description": description, "inputSchema": schema_in,
                      "outputSchema": output_schema(schema, name),
                      "annotations": {"readOnlyHint": not (verb == "fit"), "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}})
        kinds = {p: (flags[p]["kind"], flags[p]["role"]) for p in order}
        calls[name] = {"verb": verb, "sub": sub, "operands": op_props, "props": [{"name": p, "kind": kinds[p][0], "role": kinds[p][1]} for p in order],
                       "writes": wr, "needs_write": verb == "fit"}
    ro = []
    for tl in tools:
        c = calls[tl["name"]]
        if c["needs_write"]:
            continue
        tl2 = copy.deepcopy(tl)
        for w in c["writes"]:
            tl2["inputSchema"]["properties"].pop(w, None)
        ro.append(tl2)
    return {"revision": REVISION, "generator": "scripts/mcp_tools.py", "tools": tools, "readonly_tools": ro, "calls": calls}


def cho_string(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def chunk_const(name, text, comment):
    # a function per document; a literal is one line, so a document is cut in pieces joined by the reader (std has no literal concatenation)
    return "// %s\npub fn %s() -> [] &static [byte] {\n    return %s;\n}\n" % (comment, name, cho_string(text))


def render_cho(d):
    full = {"tools": d["tools"]}
    ro = {"tools": d["readonly_tools"]}
    full_s = json.dumps(full, separators=(",", ":"), ensure_ascii=False)
    ro_s = json.dumps(ro, separators=(",", ":"), ensure_ascii=False)
    rows = []
    for name in ORDER:
        c = d["calls"][name]
        props = ";".join("%s|%s|%s" % (p["name"], p["kind"], p["role"]) for p in c["props"])
        rows.append("%s|%s|%s|%s|%s|%s|%d" % (name, c["verb"], c["sub"] or "", ",".join(c["operands"]), ",".join(c["writes"]), props.replace("|", "~"), 1 if c["needs_write"] else 0))
    text = (
        "edition 5;\n\nmodule mcp.tools;\n\n"
        "// GENERATED by scripts/mcp_tools.py from the flag table (docs/mcp.md section 2); do not edit. scripts/check_mcp_tools.py regenerates it and fails on a difference.\n\n"
        + chunk_const("revision", d["revision"], "the protocol revision implemented")
        + "\n" + chunk_const("list_write", full_s, "the `tools` member of the answer to tools/list when the server was started with --allow-write")
        + "\n" + chunk_const("list_readonly", ro_s, "the same, read-only")
        + "\n" + chunk_const("calls", "\n".join(rows), "one line per tool: name|verb|sub|operands|write flags|props|needs write; props are name~kind~role separated by ;")
    )
    return text


def main():
    d = build()
    js = json.dumps(d, indent=1, ensure_ascii=False) + "\n"
    cho = render_cho(d)
    if "--check" in sys.argv:
        bad = 0
        for path, want in ((OUT_JSON, js), (OUT_CHO, cho)):
            have = path.read_text() if path.exists() else None
            if have != want:
                print("stale: %s differs from the regeneration (python3 scripts/mcp_tools.py)" % path.relative_to(ROOT))
                bad = 1
        if bad:
            sys.exit(1)
        print("mcp tools: %d tools, docs/mcp/tools.json and tools/ml-mcp/tools.cho are what the flag table gives" % len(d["tools"]))
        return
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CHO.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(js)
    OUT_CHO.write_text(cho)
    print("wrote", OUT_JSON.relative_to(ROOT), OUT_CHO.relative_to(ROOT))


if __name__ == "__main__":
    main()

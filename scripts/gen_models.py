#!/usr/bin/env python3
"""The generator of the model module contract (docs/modules.md): reads `lib/models/<name>/model.toml` for every model and writes

    tools/ml/spec.cho            from tools/ml/spec.tmpl: the flag table, allowed(), limits() and the text of tool()
    tools/ml/est.cho             the dispatch over the kind numbers of the wired models (module ml.est)
    lib/pipeline/modeldefs.cho   the keys of the `model` line of a plan (module pipeline.modeldefs)
    docs/models.md               the table of models (between the markers)

    python3 scripts/gen_models.py            write
    python3 scripts/gen_models.py --check    write nothing; exit 1 if a file differs from a fresh generation, or a manifest is wrong
    python3 scripts/gen_models.py --root DIR the same on the tree under DIR (admit.py and the mutant modules use it)

A manifest with an unknown key, a missing key or a value of the wrong type is an error (`ManifestError`). Everything generated is a pure
function of the manifests and tools/ml/spec.tmpl: no clock, no environment, no iteration order that is not sorted.
"""

import pathlib
import re
import sys
import tomllib

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent

TOP_KEYS = {
    "name": str, "kind": int, "extra_kinds": list, "kind_fn": str, "shares_kind_with": str, "task": str, "tier": str, "wired": bool,
    "adapter": str, "payload_version": int, "doc": str, "files": list, "plan_order": int, "limits": dict, "hyperparameter": list,
    "tags": dict, "authority": dict, "sums": dict, "oracle": dict,
}
REQUIRED = {"name", "kind", "kind_fn", "task", "tier", "wired", "payload_version", "doc", "files", "plan_order", "tags", "authority", "sums"}
HP_KEYS = {"name": str, "type": str, "min": int, "max": int, "values": list, "default": str, "required": bool, "doc": str, "nonneg": bool}
TASKS = {"regression", "classification", "clustering"}
TIERS = {"experimental", "supported"}
HP_TYPES = {"int", "float", "enum"}
NAME = re.compile(r"^[a-z][a-z0-9-]*$")


class ManifestError(Exception):
    pass


def _check_type(where, key, value, want):
    if want is int and isinstance(value, bool):
        raise ManifestError("%s: %s must be an integer" % (where, key))
    if not isinstance(value, want):
        raise ManifestError("%s: %s must be %s" % (where, key, want.__name__))


def parse_manifest(path):
    where = str(path)
    try:
        m = tomllib.loads(pathlib.Path(path).read_text())
    except tomllib.TOMLDecodeError as e:
        raise ManifestError("%s: %s" % (where, e))
    for k in m:
        if k not in TOP_KEYS:
            raise ManifestError("%s: unknown key %r" % (where, k))
        _check_type(where, k, m[k], TOP_KEYS[k])
    for k in sorted(REQUIRED - set(m)):
        raise ManifestError("%s: missing key %r" % (where, k))
    if not NAME.match(m["name"]):
        raise ManifestError("%s: name %r is not a plan name" % (where, m["name"]))
    if pathlib.Path(path).parent.name != m["name"]:
        raise ManifestError("%s: name %r is not the directory name" % (where, m["name"]))
    if m["task"] not in TASKS:
        raise ManifestError("%s: task %r" % (where, m["task"]))
    if m["tier"] not in TIERS:
        raise ManifestError("%s: tier %r" % (where, m["tier"]))
    if m["wired"] and "adapter" not in m:
        raise ManifestError("%s: a wired model names its adapter" % where)
    for k in ("emits",):
        if k not in m["tags"]:
            raise ManifestError("%s: [tags] needs %r" % (where, k))
    for k in m["tags"]:
        if k != "emits":
            raise ManifestError("%s: [tags] has unknown key %r" % (where, k))
    _check_type(where, "tags.emits", m["tags"]["emits"], list)
    for k, v in m["authority"].items():
        if k not in ("adds", "reason"):
            raise ManifestError("%s: [authority] has unknown key %r" % (where, k))
    m["authority"].setdefault("adds", [])
    m["authority"].setdefault("reason", "")
    if m["authority"]["adds"] and not m["authority"]["reason"]:
        raise ManifestError("%s: [authority] adds labels and states no reason" % where)
    for k in m["sums"]:
        if k != "decisive":
            raise ManifestError("%s: [sums] has unknown key %r" % (where, k))
    m["sums"].setdefault("decisive", [])
    for k in m.get("oracle", {}):
        if k not in ("script", "args", "tolerance"):
            raise ManifestError("%s: [oracle] has unknown key %r" % (where, k))
    hps = []
    for h in m.get("hyperparameter", []):
        for k in h:
            if k not in HP_KEYS:
                raise ManifestError("%s: hyperparameter has unknown key %r" % (where, k))
            _check_type(where, "hyperparameter." + k, h[k], HP_KEYS[k])
        for k in ("name", "type", "doc"):
            if k not in h:
                raise ManifestError("%s: hyperparameter %r lacks %r" % (where, h.get("name"), k))
        if h["type"] not in HP_TYPES:
            raise ManifestError("%s: hyperparameter %s: type %r" % (where, h["name"], h["type"]))
        if not NAME.match(h["name"]):
            raise ManifestError("%s: hyperparameter name %r" % (where, h["name"]))
        has_default = "default" in h
        if has_default == h.get("required", False):
            raise ManifestError("%s: hyperparameter %s: exactly one of default and required = true" % (where, h["name"]))
        if h["type"] == "int" and ("min" not in h or "max" not in h):
            raise ManifestError("%s: hyperparameter %s: an int has min and max" % (where, h["name"]))
        if h["type"] == "int" and has_default:
            if not re.fullmatch(r"[0-9]{1,9}", h["default"]) or not h["min"] <= int(h["default"]) <= h["max"]:
                raise ManifestError("%s: hyperparameter %s: default %r outside its range" % (where, h["name"], h["default"]))
        if h["type"] == "float" and not h.get("nonneg", False):
            raise ManifestError("%s: hyperparameter %s: only nonneg floats exist in the plan grammar" % (where, h["name"]))
        if h["type"] == "enum" and not h.get("values"):
            raise ManifestError("%s: hyperparameter %s: an enum has values" % (where, h["name"]))
        if ";" in h["doc"] or "|" in h["doc"] or "\"" in h["doc"] or "\\" in h["doc"]:
            raise ManifestError("%s: hyperparameter %s: the doc holds a character of the table syntax" % (where, h["name"]))
        hps.append(h)
    m["hyperparameter"] = hps
    m["_path"] = str(path)
    m["_dir"] = str(pathlib.Path(path).parent)
    return m


def load_manifests(root):
    root = pathlib.Path(root)
    models = []
    for p in sorted((root / "lib" / "models").glob("*/model.toml")):
        models.append(parse_manifest(p))
    names = [m["name"] for m in models]
    if len(set(names)) != len(names):
        raise ManifestError("two manifests with one name")
    orders = [m["plan_order"] for m in models]
    if sorted(orders) != list(range(len(models))):
        raise ManifestError("plan_order must be 0 .. %d, once each: %s" % (len(models) - 1, orders))
    return sorted(models, key=lambda m: m["plan_order"])


def wired(models):
    return sorted([m for m in models if m["wired"] and m.get("adapter") != "builtin"], key=lambda m: m["kind"])


# ------------------------------------------------------------------------------------------------ the registry (kinds.cho)

def registry(root):
    text = (pathlib.Path(root) / "lib" / "core" / "kinds.cho").read_text()
    fns = {}
    for m in re.finditer(r"pub fn (model_\w+)\(\) -> \[\] int \{\s*return (\d+);\s*\}", text):
        fns[m.group(1)] = int(m.group(2))
    return fns


def check_registry(models, root):
    """Problems (strings) in the kind numbers: unique, equal to the registry function, below model_count()."""
    problems = []
    reg = registry(root)
    count = reg.get("model_count")
    seen = {}
    for m in models:
        kinds = [m["kind"]] + list(m.get("extra_kinds", []))
        fn = m["kind_fn"]
        if fn not in reg:
            problems.append("%s: kind_fn %s is not a function of lib/core/kinds.cho" % (m["name"], fn))
        elif reg[fn] != m["kind"]:
            problems.append("%s: kind %d but %s() returns %d" % (m["name"], m["kind"], fn, reg[fn]))
        for k in kinds:
            if count is not None and not 1 <= k <= count:
                problems.append("%s: kind %d is outside 1 .. model_count() = %s" % (m["name"], k, count))
            if k in seen:
                other = seen[k]
                pair = m.get("shares_kind_with") == other["name"] and other.get("shares_kind_with") == m["name"]
                if not pair:
                    problems.append("%s: kind %d is already %s's" % (m["name"], k, other["name"]))
            else:
                seen[k] = m
    return problems


# ------------------------------------------------------------------------------------------------ the pieces of text

def lit(text):
    """A cancho string literal (one line; the six escapes)."""
    out = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ord(ch) < 32:
            raise ManifestError("a control character in a literal")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def hp_flag_entries(models):
    """name|short|kind|role|default|help of every hyperparameter flag of a wired model (one entry per name; two models may share a flag
    when its type, range, default and doc agree)."""
    seen = {}
    for m in wired(models):
        for h in m["hyperparameter"]:
            if h["type"] == "enum":
                raise ManifestError("%s: an enum hyperparameter has no flag yet" % m["name"])
            kind = "nat" if h["type"] == "int" else "text"
            default = h["default"]
            entry = "%s||%s|none|%s|%s" % (h["name"], kind, default, h["doc"])
            if h["name"] in seen and seen[h["name"]][0] != entry:
                raise ManifestError("hyperparameter %s is declared differently by two models" % h["name"])
            seen.setdefault(h["name"], (entry, []))[1].append(m["name"])
    return [seen[k][0] for k in sorted(seen)], {k: seen[k][1] for k in seen}


def model_flag_entry(models):
    names = [m["name"] for m in models if m["wired"]]
    tiers = ", ".join("%s (%s)" % (m["name"], m["tier"]) for m in models if m["wired"])
    help_ = ("the model fit builds: %s. constant (the default) is the training mean for a regression and the majority class with its probability for a "
             "classification. linear fits by least squares and logistic by Newton's method, both through the pipeline of --plan or --step. A name outside "
             "this list is args.bad-value" % tiers)
    return "model||choice:%s|none|constant|%s" % ("/".join(names), help_)


def limit_entries(models):
    out = {}
    for m in wired(models):
        for h in m["hyperparameter"]:
            if h["type"] == "int":
                out[h["name"]] = "%s|%s|%d" % (h["name"], h["default"], h["max"])
    return [out[k] for k in sorted(out)]


def render_spec(models, root):
    tmpl = (pathlib.Path(root) / "tools" / "ml" / "spec.tmpl").read_text()
    entries, _ = hp_flag_entries(models)
    hyper_names = [e.split("|")[0] for e in entries]
    subs = {
        "MODEL_FLAG": model_flag_entry(models) + ";",
        "HYPER_FLAGS": "".join(e + ";" for e in entries),
        "HYPER_ALLOW": "".join("," + n for n in hyper_names),
        "LIMITS_HYPER": "".join(";" + e for e in limit_entries(models)),
        "MODEL_LIST": ", ".join("%s (%s)" % (m["name"], m["tier"]) for m in models if m["wired"]),
        "MODEL_CHOICES": "|".join(m["name"] for m in models if m["wired"]),
    }
    out = tmpl
    for k, v in subs.items():
        out = out.replace("{{%s}}" % k, v)
    left = re.findall(r"\{\{(\w+)\}\}", out)
    if left:
        raise ManifestError("spec.tmpl: unfilled placeholders %s" % sorted(set(left)))
    header = "// GENERATED by scripts/gen_models.py from tools/ml/spec.tmpl and lib/models/*/model.toml -- do not edit; edit the template or a manifest.\n"
    lines = out.split("\n", 1)
    # the header goes after the `module` line
    mod = re.search(r"^module ml\.spec;\n", out, re.M)
    return out[:mod.end()] + "\n" + header + out[mod.end():]


def plan_defs(h):
    if h["type"] == "int":
        base = "%s:int=%d,%d" % (h["name"], h["min"], h["max"])
    elif h["type"] == "float":
        base = "%s:fnn" % h["name"]
    else:
        base = "%s:enum=%s" % (h["name"], ",".join(h["values"]))
    return base + (":req" if h.get("required") else ":=" + h["default"])


def render_modeldefs(models):
    out = ["module pipeline.modeldefs;", "", "// GENERATED by scripts/gen_models.py from lib/models/*/model.toml -- do not edit.", "",
           "// The keys of the `model` line of a plan (docs/pipeline.md section 2, docs/modules.md section 3): the grammar of lib/pipeline/plan.cho reads these def",
           "// strings (`name:type:rule`). The order is the manifests' plan_order; the keys of a model are in the order of its [[hyperparameter]] tables.", ""]
    out += ["pub fn count() -> [] int {", "    return %d;" % len(models), "}", ""]
    out += ["// the model names, in plan order", "pub fn name(i: int) -> [] &static [byte] {"]
    for i, m in enumerate(models[:-1]):
        out += ["    if i == %d {" % i, "        return %s;" % lit(m["name"]), "    }"]
    out += ["    return %s;" % lit(models[-1]["name"]), "}", ""]
    kinds = ",".join(m["name"] for m in models)
    out += ["pub fn defs[&s](verb: &s [byte]) -> [] &static [byte] {", "    if bytes.equal(verb, \"model\") {",
            "        return %s;" % lit("kind:enum=%s:req" % kinds), "    }"]
    for m in models:
        out += ["    if bytes.equal(verb, %s) {" % lit("model:" + m["name"]),
                "        return %s;" % lit(" ".join(plan_defs(h) for h in m["hyperparameter"])), "    }"]
    out += ["    return \"\";", "}", ""]
    text = "\n".join(out)
    return text.replace("module pipeline.modeldefs;\n", "module pipeline.modeldefs;\n\nimport std.bytes;\n", 1)


# ------------------------------------------------------------------------------------------------ est.cho

def _chain(ws, body):
    """body(m) -> list of lines for the branch of wired model m; the last model's branch is the fall through."""
    out = []
    for m in ws:
        out.append("    if kind == %d {" % m["kind"])
        out += ["        " + l for l in body(m)]
        out.append("    }")
    return out


def render_est(models):
    ws = wired(models)
    imports = ["import std.bytes;", "import linalg.tensor;", "import ml.csvcell;", "import models.record;"]
    imports += ["import %s;" % m["adapter"] for m in ws]
    o = ["edition 5;", "", "module ml.est;", "", "// GENERATED by scripts/gen_models.py from lib/models/*/model.toml -- do not edit (docs/modules.md section 3).",
         "// The dispatch of ml over the kind numbers of the wired models: one chain of `if kind == N` per operation, one branch per model. Function values",
         "// and traits are not used (docs/modules.md section 1). The adapters are the fixed interface of docs/modules.md section 4.", ""]
    o += sorted(imports, key=lambda s: (not s.startswith("import std"), s))
    o.append("")
    kinds = " || ".join("kind == %d" % m["kind"] for m in ws) or "false"
    o += ["pub fn wired(kind: int) -> [] bool {", "    return %s;" % kinds, "}", ""]
    o += ["// the kind of the wired model called `name` (0 if there is none)", "pub fn kind_named[&n](name: &n [byte]) -> [] int {"]
    for m in ws:
        o += ["    if bytes.equal(name, %s) {" % lit(m["name"]), "        return %d;" % m["kind"], "    }"]
    o += ["    return 0;", "}", ""]

    def simple(fname, ret, f, default):
        lines = ["pub fn %s(kind: int) -> [] %s {" % (fname, ret)]
        lines += _chain(ws, lambda m: ["return %s;" % f(m)])
        lines += ["    return %s;" % default, "}", ""]
        return lines

    o += simple("name", "&static [byte]", lambda m: lit(m["name"]), '""')
    o += simple("tier", "&static [byte]", lambda m: lit(m["tier"]), '""')
    o += ["// 1 regression, 2 classification, 0 not a wired model"]
    o += simple("task", "int", lambda m: {"regression": "1", "classification": "2"}.get(m["task"], "0"), "0")
    o += simple("nints", "int", lambda m: str(sum(1 for h in m["hyperparameter"] if h["type"] == "int")), "0")
    o += simple("nfloats", "int", lambda m: str(sum(1 for h in m["hyperparameter"] if h["type"] == "float")), "0")
    o += ["// whether the model has a hyperparameter called `flag`", "pub fn takes[&f](kind: int, flag: &f [byte]) -> [] bool {"]
    for m in ws:
        o.append("    if kind == %d {" % m["kind"])
        for h in m["hyperparameter"]:
            o += ["        if bytes.equal(flag, %s) {" % lit(h["name"]), "            return true;", "        }"]
        o.append("    }")
    o += ["    return false;", "}", ""]
    o += ["// whether `flag` is a hyperparameter of some wired model (its flag is then a flag of fit)", "pub fn hparam_flag[&f](flag: &f [byte]) -> [] bool {"]
    names = sorted({h["name"] for m in ws for h in m["hyperparameter"]})
    for n in names:
        o += ["    if bytes.equal(flag, %s) {" % lit(n), "        return true;", "    }"]
    o += ["    return false;", "}", ""]

    # the readers of the values of the model line (used by hparams and hparam_ok below)
    o += [
        "// the value of the token `key=value` after its `=`",
        "fn value_of[&t](tok: &t [byte]) -> [] &t [byte] {",
        "    var i = 0;",
        "    while i < len(tok) && int_of(tok[i]) != '=' {",
        "        i = i + 1;",
        "    }",
        "    if i >= len(tok) {",
        "        return tok[len(tok)..len(tok)];",
        "    }",
        "    return tok[i + 1..len(tok)];",
        "}",
        "",
        "// a plan integer (1 to 9 digits, as the plan parser wrote it), or -1",
        "fn int_value[&t](v: &t [byte]) -> [] int {",
        "    if len(v) < 1 || len(v) > 9 {",
        "        return 0 - 1;",
        "    }",
        "    var n = 0;",
        "    var i = 0;",
        "    while i < len(v) {",
        "        let c = int_of(v[i]);",
        "        if c < '0' || c > '9' {",
        "            return 0 - 1;",
        "        }",
        "        n = n * 10 + c - '0';",
        "        i = i + 1;",
        "    }",
        "    return n;",
        "}",
        "",
        "// a plan float, as flt.put_float wrote it and ml.csvcell reads it; -1.0 when it is not one (a plan float is never negative)",
        "fn float_value[&t](v: &t [byte]) -> [] float {",
        "    let (st, kind, m, frac, idig, key) = csvcell.read(v);",
        "    if st != 0 {",
        "        return 0.0 - 1.0;",
        "    }",
        "    return csvcell.double(kind, m, frac, key);",
        "}",
        "",
    ]

    # the value of a hyperparameter flag, checked before the plan is built (so a bad --l2 is args.bad-value, not a plan error)
    o += ["// whether `value` is a value of the hyperparameter `name`: an integer inside its range, or a finite float of 0 or more",
          "pub fn hparam_ok[&n, &v](name: &n [byte], value: &v [byte]) -> [] bool {"]
    seen = set()
    for m in ws:
        for h in m["hyperparameter"]:
            if h["name"] in seen:
                continue
            seen.add(h["name"])
            o.append("    if bytes.equal(name, %s) {" % lit(h["name"]))
            if h["type"] == "int":
                o += ["        let v = int_value(value);", "        return v >= %d && v <= %d;" % (h["min"], h["max"])]
            else:
                o += ["        return float_value(value) >= 0.0;"]
            o.append("    }")
    o += ["    return false;", "}", ""]

    o += ["pub fn diag_len[&i](kind: int, d: int, ip: &i [int]) -> [] int"]
    o[-1] += " {"
    o += _chain(ws, lambda m: ["return %s.diag_len(d, ip);" % m["adapter"].split(".")[-1]])
    o += ["    return 0 - 1;", "}", ""]
    o += ["pub fn out_len(kind: int, d: int) -> [] int {"]
    o += _chain(ws, lambda m: ["return %s.out_len(d);" % m["adapter"].split(".")[-1]])
    o += ["    return 0 - 1;", "}", ""]
    o += ["pub fn payload_size(kind: int, d: int) -> [] int {"]
    o += _chain(ws, lambda m: ["return %s.payload_size(d);" % m["adapter"].split(".")[-1]])
    o += ["    return 0 - 1;", "}", ""]
    o += ["// 0, or a code of ml.refuse (ml.param.out-of-range for a kind that is not wired)",
          "pub fn fit[&h, &x, &y, &i, &f, &q, &g](heap: &!h Heap, kind: int, threads: int, xt: tensor.Tensor, xd: &x [float], yd: &y [float], ip: &i [int], fp: &f [float], seed: int, out: &!q [float], diag: &!g [float]) -> [heap, conc] int {"]
    o += _chain(ws, lambda m: ["return %s.fit(heap, threads, xt, xd, yd, ip, fp, seed, out, diag);" % m["adapter"].split(".")[-1]])
    o += ["    return 8;", "}", ""]
    for fn in ("predict", "proba"):
        o += ["pub fn %s[&x, &w, &q](kind: int, xt: tensor.Tensor, xd: &x [float], wv: &w [float], out: &!q [float]) -> [] int {" % fn]
        o += _chain(ws, lambda m, fn=fn: ["return %s.%s(xt, xd, wv, out);" % (m["adapter"].split(".")[-1], fn)])
        o += ["    return 8;", "}", ""]
    o += ["pub fn encode[&b, &w](kind: int, dst: &!b [byte], at: int, d: int, wv: &w [float], c0: int, c1: int) -> [] int {"]
    o += _chain(ws, lambda m: ["return %s.encode(dst, at, d, wv, c0, c1);" % m["adapter"].split(".")[-1]])
    o += ["    return 0 - 1;", "}", ""]
    o += ["pub fn decode[&s, &q](kind: int, src: &s [byte], d: int, out: &!q [float]) -> [] record.Decoded {"]
    o += _chain(ws, lambda m: ["return %s.decode(src, d, out);" % m["adapter"].split(".")[-1]])
    o += ["    return record.decoded(1, 0, 0, 0);", "}", ""]

    # the hyperparameters of the canonical model line: `model kind=NAME k=v k=v` (docs/pipeline.md section 2)
    o += [
        "// The integer and float hyperparameters of the model line, in the order of the manifest, into ip and fp. 0, or the place (from 1) of the token that is not",
        "// a value of its kind. `line` is the canonical model line without its LF.",
        "pub fn hparams[&l, &i, &f](kind: int, line: &l [byte], ip: &!i [int], fp: &!f [float]) -> [] int {",
    ]
    for m in ws:
        o.append("    if kind == %d {" % m["kind"])
        ni = nf = 0
        for k, h in enumerate(m["hyperparameter"]):
            tok = "bytes.field(line, ' ', %d)" % (3 + k)
            if h["type"] == "int":
                o += ["        ip[%d] = int_value(value_of(%s));" % (ni, tok), "        if ip[%d] < 0 {" % ni, "            return %d;" % (k + 1), "        }"]
                ni += 1
            else:
                o += ["        fp[%d] = float_value(value_of(%s));" % (nf, tok), "        if fp[%d] < 0.0 {" % nf, "            return %d;" % (k + 1), "        }"]
                nf += 1
        o += ["        return 0;", "    }"]
    o += ["    return 0 - 1;", "}", ""]
    return "\n".join(o)


# ------------------------------------------------------------------------------------------------ docs/models.md

MD_BEGIN = "<!-- gen:models -->"
MD_END = "<!-- /gen:models -->"


def render_models_table(models):
    rows = ["| model | kind | task | tier | wired | hyperparameters (name, range, default) | rule tags it can emit |", "|---|---|---|---|---|---|---|"]
    for m in models:
        hp = []
        for h in m["hyperparameter"]:
            if h["type"] == "int":
                r = "%d to %d" % (h["min"], h["max"])
            elif h["type"] == "float":
                r = "finite, 0 or more"
            else:
                r = "/".join(h["values"])
            hp.append("`%s` (%s, %s)" % (h["name"], r, "required" if h.get("required") else "default `%s`" % h["default"]))
        kinds = str(m["kind"]) + ("".join(", %d" % k for k in m.get("extra_kinds", [])))
        rows.append("| `%s` | %s | %s | %s | %s | %s | %s |" % (m["name"], kinds, m["task"], m["tier"], "yes" if m["wired"] else "no (library)",
                                                         "; ".join(hp) or "none", ", ".join("`%s`" % t for t in m["tags"]["emits"]) or "none"))
    return "\n".join(rows)


def render_models_md(models, root):
    path = pathlib.Path(root) / "docs" / "models.md"
    head = ("# The models (generated)\n\nThis page is the table of `lib/models/*/model.toml`, written by `scripts/gen_models.py` (`docs/modules.md`). It is not edited by hand; "
            "`scripts/check.sh` fails when it is stale.\n\n")
    return head + MD_BEGIN + "\n" + render_models_table(models) + "\n" + MD_END + "\n"


def generate(root=DEFAULT_ROOT):
    """{relative path: text} of every generated file."""
    root = pathlib.Path(root)
    models = load_manifests(root)
    problems = check_registry(models, root)
    if problems:
        raise ManifestError("; ".join(problems))
    return {
        "tools/ml/spec.cho": render_spec(models, root),
        "tools/ml/est.cho": render_est(models),
        "lib/pipeline/modeldefs.cho": render_modeldefs(models),
        "docs/models.md": render_models_md(models, root),
    }


def main(argv):
    root = DEFAULT_ROOT
    if "--root" in argv:
        root = pathlib.Path(argv[argv.index("--root") + 1]).resolve()
    check = "--check" in argv
    try:
        files = generate(root)
    except ManifestError as e:
        print("gen_models: " + str(e))
        return 1
    stale = []
    for rel, text in files.items():
        p = root / rel
        if check:
            if not p.exists() or p.read_text() != text:
                stale.append(rel)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    if stale:
        print("gen_models: stale generated files (run scripts/gen_models.py): " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

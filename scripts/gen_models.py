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
    "adapter": str, "shape": str, "payload_version": int, "doc": str, "files": list, "plan_order": int, "limits": dict, "hyperparameter": list,
    "tags": dict, "authority": dict, "sums": dict, "oracle": dict, "target_classes": int, "weighted": bool, "ranking": bool,
}
REQUIRED = {"name", "kind", "kind_fn", "task", "tier", "wired", "payload_version", "doc", "files", "plan_order", "tags", "authority", "sums"}
HP_KEYS = {"name": str, "type": str, "min": int, "max": int, "values": list, "default": str, "required": bool, "doc": str, "nonneg": bool}
TASKS = {"regression", "classification", "clustering", "both"}
SHAPES = {"vector", "blob"}
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
    m.setdefault("shape", "vector")
    if m["shape"] not in SHAPES:
        raise ManifestError("%s: shape %r (vector or blob)" % (where, m["shape"]))
    if m["shape"] == "blob" and m["task"] not in ("both", "clustering", "classification"):
        raise ManifestError("%s: a blob model serves both tasks, is a clustering or a classifier" % where)
    if m["shape"] == "blob" and m["task"] == "both" and not any(h.get("name") == "task" and h.get("type") == "enum" for h in m.get("hyperparameter", [])):
        raise ManifestError("%s: a blob model of both tasks has a `task` enum hyperparameter" % where)
    if m["task"] == "both" and m["shape"] != "blob":
        raise ManifestError("%s: task = both needs shape = blob" % where)
    if m["task"] == "clustering" and m["wired"] and m["shape"] != "blob":
        raise ManifestError("%s: a wired clustering is a blob model (its payload size depends on k)" % where)
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
        if k not in ("decisive", "sequential"):
            raise ManifestError("%s: [sums] has unknown key %r" % (where, k))
    m["sums"].setdefault("decisive", [])
    m["sums"].setdefault("sequential", [])
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
        if h["type"] == "enum" and h["name"] != "task" and not all(re.fullmatch(r"[a-z0-9+-]+", v) for v in h["values"]):
            raise ManifestError("%s: hyperparameter %s: a value of an enum is a flag value: lower-case letters, digits, + and -" % (where, h["name"]))
        if h["type"] == "enum" and has_default and h["default"] not in h["values"]:
            raise ManifestError("%s: hyperparameter %s: default %r is not one of its values" % (where, h["name"], h["default"]))
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


def flag_hps(m):
    """the hyperparameters of a model that have a flag: every one but the `task` of a model of both tasks (the existing --task)"""
    return [h for h in m["hyperparameter"] if not (h["type"] == "enum" and h["name"] == "task")]


def hp_shape(h):
    """what two models must agree on to share a flag: type, range or values"""
    return (h["type"], h.get("min"), h.get("max"), tuple(h.get("values", ())), h.get("nonneg"))


def hp_flag_entries(models):
    """name|short|kind|role|default|help of every hyperparameter flag of a wired model (one entry per name; two models may share a flag when its type,
    range and values agree: the flag table then has no default of its own when they differ, and the help names each model's)."""
    seen = {}
    for m in wired(models):
        for h in flag_hps(m):
            seen.setdefault(h["name"], []).append((m, h))
    out, users = [], {}
    for name in sorted(seen):
        group = seen[name]
        if any(hp_shape(h) != hp_shape(group[0][1]) for _, h in group):
            raise ManifestError("hyperparameter %s is declared differently by two models" % name)
        h0 = group[0][1]
        kind = {"int": "nat", "float": "text"}.get(h0["type"]) or "choice:" + "/".join(h0["values"])
        defaults = {h.get("default", "") for _, h in group}
        default = defaults.pop() if len(defaults) == 1 else ""
        docs = {h["doc"] for _, h in group}
        if len(docs) == 1:
            doc = docs.pop()
        else:
            doc = ". ".join("%s: %s" % (m["name"], h["doc"].rstrip(".")) + (" (default %s)" % h["default"] if "default" in h else "") for m, h in group)
        out.append("%s||%s|none|%s|%s" % (name, kind, default, doc))
        users[name] = [m["name"] for m, _ in group]
    return out, users


def model_flag_entry(models):
    names = [m["name"] for m in models if m["wired"]]
    tiers = ", ".join("%s (%s)" % (m["name"], m["tier"]) for m in models if m["wired"])
    how = {"linear": "linear fits by least squares", "logistic": "logistic by Newton's method",
           "kmeans": "kmeans clusters the rows around k centres by a fixed number of Lloyd iterations and needs no target (--k, --iterations, --init, --seed)",
           "tree": "tree is a CART decision tree with histogram splits (--task chooses classification or regression)",
           "forest": "forest is a bagged random forest of such trees, parallel over trees (--task, --trees, --seed)",
           "boost": "boost is a gradient-boosted ensemble of such trees, fitted round by round to the gradients of the squared-error or the logistic loss (--task, --rounds, --learning-rate, --subsample, --early-stopping, --seed)"}
    sentences = [how[n] for n in names if n in how]
    help_ = ("the model fit builds: %s. constant (the default) is the training mean for a regression and the majority class with its probability for a "
             "classification. %s. All but constant are fitted through the pipeline of --plan or --step. A name outside "
             "this list is args.bad-value" % (tiers, " and ".join(sentences) if sentences else "the others go through the pipeline"))
    return "model||choice:%s|none|constant|%s" % ("/".join(names), help_)


def limit_entries(models):
    out, defaults = {}, {}
    for m in wired(models):
        for h in m["hyperparameter"]:
            if h["type"] == "int":
                defaults.setdefault(h["name"], set()).add(h.get("default", str(h["min"])))
                out[h["name"]] = (h["name"], h["max"])
    return ["%s|%s|%d" % (n, next(iter(defaults[n])) if len(defaults[n]) == 1 else str(min(int(v) for v in defaults[n])), out[n][1]) for n in sorted(out)]


# the flags of `ml-predict` (docs/serving.md section 3.2): those of `predict` and `describe` that write nothing, taken by name from the table of `ml`
PREDICT_DROPS = {"log", "log-max-bytes"}


def predict_tables(spec_text):
    """(flag table, limits) of ml-predict, filtered from the generated table of ml by name: one definition of every flag."""
    table = re.search(r'pub fn flag_table\(\) -> \[\] &static \[byte\] \{\n    return "(.*?)";\n\}', spec_text, re.S).group(1)
    limits = re.search(r'pub fn limits\(\) -> \[\] &static \[byte\] \{\n    return "(.*?)";\n\}', spec_text, re.S).group(1)
    allowed = set()
    for verb in ("2", "4"):
        m = re.search(r"if verb == %s \{\n        return \"(.*?)\";" % verb, spec_text)
        allowed |= set(m.group(1).split(","))
    keep = allowed - PREDICT_DROPS
    flags = [e for e in table.split(";") if e.split("|")[0] in keep]
    if {e.split("|")[0] for e in flags} != keep:
        raise ManifestError("predict flags: %s are not in the table" % sorted(keep - {e.split('|')[0] for e in flags}))
    lims = [e for e in limits.split(";") if e.split("|")[0] in keep]
    return ";".join(flags), ";".join(lims)


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
        "HYPER_USAGE": " ".join("[--%s %s]" % (n, {"int": "N", "float": "X", "enum": "NAME"}[next(h for m in wired(models) for h in flag_hps(m) if h["name"] == n)["type"]]) for n in hyper_names),
        "HYPER_FLAG_NAMES": ", ".join("--" + n for n in hyper_names),
    }
    out = tmpl
    for k, v in subs.items():
        out = out.replace("{{%s}}" % k, v)
    pflags, plimits = predict_tables(out)
    out = out.replace("{{PREDICT_FLAGS}}", pflags).replace("{{PREDICT_LIMITS}}", plimits)
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
    out += ["// the keys of the model called `name`, or \"\" when there is no such model", "pub fn model_defs[&s](name: &s [byte]) -> [] &static [byte] {"]
    for m in models:
        out += ["    if bytes.equal(name, %s) {" % lit(m["name"]), "        return %s;" % lit(" ".join(plan_defs(h) for h in m["hyperparameter"])), "    }"]
    out += ["    return \"\";", "}", ""]
    text = "\n".join(out)
    return text.replace("module pipeline.modeldefs;\n", "module pipeline.modeldefs;\n\nimport std.bytes;\n", 1)


# ------------------------------------------------------------------------------------------------ est.cho

def keys_of(ws):
    """{model name: dispatch key}. The key of a model is its kind; the second of a declared pair that shares a kind (tree and forest) is the kind plus 100, so that
    a fit can tell them apart (different hyperparameters) while a file, which has only the kind, goes to the first (docs/wiring-trees.md section 10)."""
    keys, seen = {}, set()
    for m in ws:
        k = m["kind"]
        keys[m["name"]] = k + 100 if k in seen else k
        seen.add(k)
    return keys


def is_blob(m):
    return m.get("shape", "vector") == "blob"


def _chain(ws, body, keys):
    """body(m) -> list of lines for the branch of wired model m, tested on its dispatch key"""
    out = []
    for m in ws:
        out.append("    if kind == %d {" % keys[m["name"]])
        out += ["        " + l for l in body(m)]
        out.append("    }")
    return out


def int_like(m):
    """the hyperparameters stored in ip: the ints and the enums (an enum as the index of its value), in manifest order"""
    return [h for h in m["hyperparameter"] if h["type"] in ("int", "enum")]


def render_est(models):
    ws = wired(models)
    keys = keys_of(ws)
    vec = [m for m in ws if not is_blob(m)]
    blob = [m for m in ws if is_blob(m)]
    imports = ["import std.bytes;", "import linalg.tensor;", "import ml.csvcell;", "import models.record;"]
    imports += ["import %s;" % m["adapter"] for m in ws]
    o = ["edition 5;", "", "module ml.est;", "", "// GENERATED by scripts/gen_models.py from lib/models/*/model.toml -- do not edit (docs/modules.md section 3).",
         "// The dispatch of ml over the models wired: one chain of `if kind == N` per operation, one branch per model. Function values and traits are not used",
         "// (docs/modules.md section 1). `kind` is the dispatch key of a model: its kind number, plus 100 for the second of a pair that shares a kind (kind_of gives the",
         "// number a file has). The adapters are the two fixed interfaces of docs/modules.md sections 4 (vector models) and 4b (blob models).", ""]
    o += sorted(imports, key=lambda s: (not s.startswith("import std"), s))
    o.append("")
    kinds = " || ".join("kind == %d" % k for k in sorted({m["kind"] for m in ws})) or "false"
    o += ["// whether a file whose model section opens with this kind is one `ml` builds", "pub fn wired(kind: int) -> [] bool {", "    return %s;" % kinds, "}", ""]
    o += ["// the kind a file has for the dispatch key of a model (0 if there is none)", "pub fn kind_of(kind: int) -> [] int {"]
    o += _chain(ws, lambda m: ["return %d;" % m["kind"]], keys)
    o += ["    return 0;", "}", ""]
    o += ["// the dispatch key of the wired model called `name` (0 if there is none)", "pub fn kind_named[&n](name: &n [byte]) -> [] int {"]
    for m in ws:
        o += ["    if bytes.equal(name, %s) {" % lit(m["name"]), "        return %d;" % keys[m["name"]], "    }"]
    o += ["    return 0;", "}", ""]

    def simple(fname, ret, f, default):
        lines = ["pub fn %s(kind: int) -> [] %s {" % (fname, ret)]
        lines += _chain(ws, lambda m: ["return %s;" % f(m)], keys)
        lines += ["    return %s;" % default, "}", ""]
        return lines

    o += simple("name", "&static [byte]", lambda m: lit(m["name"]), '""')
    o += simple("tier", "&static [byte]", lambda m: lit(m["tier"]), '""')
    o += ["// 1 regression, 2 classification, 3 either (the `task` of the model line, task_of), 4 clustering (no target), 0 not a wired model"]
    o += simple("task", "int", lambda m: {"regression": "1", "classification": "2", "both": "3", "clustering": "4"}.get(m["task"], "0"), "0")
    o += simple("nints", "int", lambda m: str(len(int_like(m))), "0")
    o += simple("nfloats", "int", lambda m: str(sum(1 for h in m["hyperparameter"] if h["type"] == "float")), "0")
    o += ["// 1 regression or 2 classification: the task of the model whose integer hyperparameters are `ip` (a model of both tasks has it in `task`)",
          "pub fn task_of[&i](kind: int, ip: &i [int]) -> [] int {"]
    for m in ws:
        if m["task"] == "both":
            idx = [h["name"] for h in int_like(m)].index("task")
            vals = next(h for h in m["hyperparameter"] if h["name"] == "task")["values"]
            o += ["    if kind == %d {" % keys[m["name"]], "        if ip[%d] == %d {" % (idx, vals.index("classification")), "            return 2;", "        }", "        return 1;", "    }"]
    o += ["    return task(kind);", "}", ""]
    o += ["// whether the model has a hyperparameter called `flag` (the `task` of a model of both tasks is the --task flag, not a hyperparameter flag)",
          "pub fn takes[&f](kind: int, flag: &f [byte]) -> [] bool {"]
    for m in ws:
        o.append("    if kind == %d {" % keys[m["name"]])
        for h in flag_hps(m):
            o += ["        if bytes.equal(flag, %s) {" % lit(h["name"]), "            return true;", "        }"]
        o.append("    }")
    o += ["    return false;", "}", ""]
    o += ["// whether `flag` is a hyperparameter of some wired model (its flag is then a flag of fit)", "pub fn hparam_flag[&f](flag: &f [byte]) -> [] bool {"]
    names = sorted({h["name"] for m in ws for h in flag_hps(m)})
    for n in names:
        o += ["    if bytes.equal(flag, %s) {" % lit(n), "        return true;", "    }"]
    o += ["    return false;", "}", ""]
    o += ["// whether the model keeps its fitted state as a byte payload that its adapter builds and reads (the blob interface), not as a vector of floats",
          "pub fn is_blob(kind: int) -> [] bool {"]
    o += ["    return %s;" % (" || ".join("kind == %d" % keys[m["name"]] for m in blob) or "false"), "}", ""]
    o += ["// whether the model takes a cell that is still missing after the steps (a blob model of both tasks: the tree and the forest); the others are refused with ml.input.missing-value",
          "pub fn takes_missing(kind: int) -> [] bool {"]
    o += ["    return %s;" % (" || ".join("kind == %d" % keys[m["name"]] for m in blob if m["task"] == "both") or "false"), "}", ""]

    # a model whose integer hyperparameter `early-stopping` is above 0 keeps rows out of its fit and needs a declared held-out part (--split)
    o += ["// whether the hyperparameters `ip` of the model ask for early stopping, which is refused unless --split declares held-out rows", "pub fn needs_split[&i](kind: int, ip: &i [int]) -> [] bool {"]
    for m in ws:
        for k, h in enumerate(int_like(m)):
            if h["name"] == "early-stopping":
                o += ["    if kind == %d {" % keys[m["name"]], "        return ip[%d] > 0;" % k, "    }"]
    o += ["    return false;", "}", ""]

    # the least value of an integer hyperparameter flag (1 unless its manifest says 0: the generic check of the nat flags is 1 to the ceiling)
    mins = {}
    for m in ws:
        for h in m["hyperparameter"]:
            if h["type"] == "int":
                mins.setdefault(h["name"], h["min"])
    o += ["// the least value the command line accepts for the integer hyperparameter flag `name` (the ceiling is in spec.limits())", "pub fn hparam_min[&n](name: &n [byte]) -> [] int {"]
    for n in sorted(mins):
        if mins[n] != 1:
            o += ["    if bytes.equal(name, %s) {" % lit(n), "        return %d;" % mins[n], "    }"]
    o += ["    return 1;", "}", ""]

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
            if (h["type"] == "enum" and h["name"] == "task") or h["name"] in seen:
                continue
            seen.add(h["name"])
            o.append("    if bytes.equal(name, %s) {" % lit(h["name"]))
            if h["type"] == "enum":
                o += ["        // a value of an enum is checked by the choice kind of its flag (args.bad-value)", "        return true;"]
            elif h["type"] == "int":
                o += ["        // the range of an integer is checked by the nat check of the command line (ml.param.out-of-range)", "        return true;"]
            else:
                o += ["        return float_value(value) >= 0.0;"]
            o.append("    }")
    o += ["    return false;", "}", ""]

    # ---- the vector interface (docs/modules.md section 4)
    ad = lambda m: m["adapter"].split(".")[-1]
    o += ["pub fn diag_len[&i](kind: int, d: int, ip: &i [int]) -> [] int {"]
    o += _chain(vec, lambda m: ["return %s.diag_len(d, ip);" % ad(m)], keys)
    o += ["    return 0 - 1;", "}", ""]
    # a vector model that takes a weight per row (docs/weights.md): `weighted = true` in its manifest and a fit_weighted in its adapter
    wtd = [m for m in vec if m.get("weighted")]
    wtd_all = [m for m in ws if m.get("weighted")]
    o += ["// whether the fit of this kind takes a weight per row (docs/weights.md section 2.4)", "pub fn weighted(kind: int) -> [] bool {"]
    o += _chain(wtd_all, lambda m: ["return true;"], keys)
    o += ["    return false;", "}", ""]
    o += ["// 0, or a code of ml.refuse (ml.param.out-of-range for a kind that has no weighted fit); wd: one weight per row",
          "pub fn fit_weighted[&h, &x, &y, &w, &i, &f, &q, &g](heap: &!h Heap, kind: int, threads: int, xt: tensor.Tensor, xd: &x [float], yd: &y [float], wd: &w [float], ip: &i [int], fp: &f [float], seed: int, out: &!q [float], diag: &!g [float]) -> [heap, conc] int {"]
    o += _chain(wtd, lambda m: ["return %s.fit_weighted(heap, threads, xt, xd, yd, wd, ip, fp, seed, out, diag);" % ad(m)], keys)
    o += ["    return 8;", "}", ""]
    o += ["pub fn out_len(kind: int, d: int) -> [] int {"]
    o += _chain(vec, lambda m: ["return %s.out_len(d);" % ad(m)], keys)
    o += ["    return 0 - 1;", "}", ""]
    o += ["pub fn payload_size(kind: int, d: int) -> [] int {"]
    o += _chain(vec, lambda m: ["return %s.payload_size(d);" % ad(m)], keys)
    o += ["    return 0 - 1;", "}", ""]
    o += ["// 0, or a code of ml.refuse (ml.param.out-of-range for a kind that is not wired)",
          "pub fn fit[&h, &x, &y, &i, &f, &q, &g](heap: &!h Heap, kind: int, threads: int, xt: tensor.Tensor, xd: &x [float], yd: &y [float], ip: &i [int], fp: &f [float], seed: int, out: &!q [float], diag: &!g [float]) -> [heap, conc] int {"]
    o += _chain(vec, lambda m: ["return %s.fit(heap, threads, xt, xd, yd, ip, fp, seed, out, diag);" % ad(m)], keys)
    o += ["    return 8;", "}", ""]
    for fn in ("predict", "proba"):
        o += ["pub fn %s[&x, &w, &q](kind: int, xt: tensor.Tensor, xd: &x [float], wv: &w [float], out: &!q [float]) -> [] int {" % fn]
        o += _chain(vec, lambda m, fn=fn: ["return %s.%s(xt, xd, wv, out);" % (ad(m), fn)], keys)
        o += ["    return 8;", "}", ""]
    o += ["pub fn encode[&b, &w](kind: int, dst: &!b [byte], at: int, d: int, wv: &w [float], c0: int, c1: int) -> [] int {"]
    o += _chain(vec, lambda m: ["return %s.encode(dst, at, d, wv, c0, c1);" % ad(m)], keys)
    o += ["    return 0 - 1;", "}", ""]
    o += ["pub fn decode[&s, &q](kind: int, src: &s [byte], d: int, out: &!q [float]) -> [] record.Decoded {"]
    o += _chain(vec, lambda m: ["return %s.decode(src, d, out);" % ad(m)], keys)
    o += ["    return record.decoded(1, 0, 0, 0);", "}", ""]

    # ---- the blob interface (docs/modules.md section 4b): the fitted model is a payload of bytes
    o += ["// 0, or a code of ml.refuse or of the library (the payload is then empty; ml.param.out-of-range for a kind that has no blob adapter)",
          "pub fn fit_blob[&h, &x, &m, &y, &i, &f](heap: &!h Heap, kind: int, threads: int, xt: tensor.Tensor, xd: &x [float], miss: &m [int], wpc: int, yd: &y [float], ip: &i [int], fp: &f [float], seed: int, c0: int, c1: int, rows_cap: int, cols_cap: int, cells_cap: int) -> [heap, conc] (Box[[byte]], int) {"]
    o += _chain(blob, lambda m: ["return %s.fit_blob(heap, threads, xt, xd, miss, wpc, yd, ip, fp, seed, c0, c1, rows_cap, cols_cap, cells_cap);" % ad(m)], keys)
    o += ["    return (box_slice(heap, 1, byte_of(0)), 8);", "}", ""]
    wblob = [m for m in blob if m.get("weighted")]
    o += ["// as fit_blob with a weight per row (docs/weights.md section 2.5); wd: one weight per row",
          "pub fn fit_blob_weighted[&h, &x, &m, &y, &w, &i, &f](heap: &!h Heap, kind: int, threads: int, xt: tensor.Tensor, xd: &x [float], miss: &m [int], wpc: int, yd: &y [float], wd: &w [float], ip: &i [int], fp: &f [float], seed: int, c0: int, c1: int, rows_cap: int, cols_cap: int, cells_cap: int) -> [heap, conc] (Box[[byte]], int) {"]
    o += _chain(wblob, lambda m: ["return %s.fit_blob_weighted(heap, threads, xt, xd, miss, wpc, yd, wd, ip, fp, seed, c0, c1, rows_cap, cols_cap, cells_cap);" % ad(m)], keys)
    o += ["    return (box_slice(heap, 1, byte_of(0)), 8);", "}", ""]
    rblob = [m for m in blob if m.get("ranking")]
    o += ["// the pairwise ranking fit (docs/ranking.md section 4): as fit_blob with the group number of every row; ng groups; wtd says whether wd holds the weights",
          "pub fn fit_blob_rank[&h, &x, &m, &y, &w, &u, &i, &f](heap: &!h Heap, kind: int, threads: int, xt: tensor.Tensor, xd: &x [float], miss: &m [int], wpc: int, yd: &y [float], wd: &w [float], wtd: bool, gid: &u [int], ng: int, ip: &i [int], fp: &f [float], seed: int, c0: int, c1: int, rows_cap: int, cols_cap: int, cells_cap: int) -> [heap, conc] (Box[[byte]], int) {"]
    o += _chain(rblob, lambda m: ["return %s.fit_blob_rank(heap, threads, xt, xd, miss, wpc, yd, wd, wtd, gid, ng, ip, fp, seed, c0, c1, rows_cap, cols_cap, cells_cap);" % ad(m)], keys)
    o += ["    return (box_slice(heap, 1, byte_of(0)), 8);", "}", ""]
    o += ["// whether the kind has a ranking objective (--task ranking)", "pub fn ranking(kind: int) -> [] bool {"]
    o += _chain(rblob, lambda m: ["return true;"], keys)
    o += ["    return false;", "}", ""]
    for fn in ("predict", "proba"):
        o += ["pub fn %s_blob[&h, &b, &x, &m, &q](heap: &!h Heap, kind: int, blob: &b [byte], xt: tensor.Tensor, xd: &x [float], miss: &m [int], wpc: int, out: &!q [float]) -> [heap] int {" % fn]
        o += _chain(blob, lambda m, fn=fn: ["return %s.%s_blob(heap, blob, xt, xd, miss, wpc, out);" % (ad(m), fn)], keys)
        o += ["    return 8;", "}", ""]
    o += ["// the payload read and checked: the code, the two labels and in `end` the task (1 regression, 2 classification)",
          "pub fn decode_blob[&h, &s](heap: &!h Heap, kind: int, src: &s [byte], d: int) -> [heap] record.Decoded {"]
    o += _chain(blob, lambda m: ["return %s.decode_blob(heap, src, d);" % ad(m)], keys)
    o += ["    return record.decoded(1, 0, 0, 0);", "}", ""]

    # the hyperparameters of the canonical model line: `model kind=NAME k=v k=v` (docs/pipeline.md section 2)
    o += [
        "// The integer and float hyperparameters of the model line, in the order of the manifest, into ip and fp (an enum as the index of its value, in ip). 0, or the place",
        "// (from 1) of the token that is not a value of its kind. `line` is the canonical model line without its LF.",
        "pub fn hparams[&l, &i, &f](kind: int, line: &l [byte], ip: &!i [int], fp: &!f [float]) -> [] int {",
    ]
    for m in ws:
        o.append("    if kind == %d {" % keys[m["name"]])
        ni = nf = 0
        for k, h in enumerate(m["hyperparameter"]):
            tok = "bytes.field(line, ' ', %d)" % (3 + k)
            if h["type"] == "int":
                o += ["        ip[%d] = int_value(value_of(%s));" % (ni, tok), "        if ip[%d] < 0 {" % ni, "            return %d;" % (k + 1), "        }"]
                ni += 1
            elif h["type"] == "enum":
                first = True
                for vi, v in enumerate(h["values"]):
                    o += ["        %sif bytes.equal(value_of(%s), %s) {" % ("" if first else "} else ", tok, lit(v)), "            ip[%d] = %d;" % (ni, vi)]
                    first = False
                o += ["        } else {", "            return %d;" % (k + 1), "        }"]
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

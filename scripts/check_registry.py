#!/usr/bin/env python3
"""The registries of the numbers and tags two stages could claim twice, read from the sources and compared with the documents
(docs/design.md 4.1 and 8.1, docs/rng.md, tools/ml/spec.cho). The in-language half is tests/t_registry.cho; this is the half that
needs the documents and the places that still write a bare number (tree.grow's two stream kinds, tools/ml/mfile.cho's three model kinds).
Exit 1 and a line per disagreement. usage: check_registry.py [--root DIR]"""
import pathlib, re, sys

ROOT = pathlib.Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else pathlib.Path(__file__).resolve().parents[1]
bad = []


def fail(msg):
    bad.append(msg)


def read(p):
    return (ROOT / p).read_text()


def consts(path):
    """name -> int of every `pub fn name() -> [] int { return N; }`"""
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"pub fn (\w+)\(\) -> \[\] int \{\s*return (\d+);", read(path))}


def table_rows(text, header_start):
    """the rows of the markdown table whose header line starts with `header_start`, as lists of stripped cells"""
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if l.startswith(header_start):
            rows = []
            for r in lines[i + 2:]:
                if not r.startswith("|"):
                    break
                rows.append([c.strip() for c in r.strip().strip("|").split("|")])
            return rows
    fail("no table starting %r" % header_start)
    return []


def unique(label, values):
    seen = {}
    for name, v in values:
        if v in seen:
            fail("%s: %s and %s are both %d" % (label, seen[v], name, v))
        seen[v] = name


# ---- model kinds
K = consts("lib/core/kinds.cho")
models = [(n, v) for n, v in K.items() if n.startswith("model_") and n != "model_count"]
unique("model kinds (lib/core/kinds.cho)", models)
if K.get("model_count") != len(models) or sorted(v for _, v in models) != list(range(1, len(models) + 1)):
    fail("model kinds are not 1..model_count without gaps")
design = read("docs/design.md")
doc_kinds = [int(r[0]) for r in table_rows(design, "| kind | model |")]
if doc_kinds != sorted(v for _, v in models):
    fail("design.md 4.1 kind table %s differs from lib/core/kinds.cho %s" % (doc_kinds, sorted(v for _, v in models)))
rec = consts("lib/models/record.cho")
for fn, name in (("kind_linear", "model_linear"), ("kind_logistic", "model_logistic"), ("kind_kmeans", "model_kmeans")):
    if rec.get(fn) != K[name]:
        fail("lib/models/record.cho %s = %s, registry %s = %s" % (fn, rec.get(fn), name, K[name]))
if consts("lib/models/tree/model.cho").get("kind") != K["model_tree"]:
    fail("lib/models/tree/model.cho kind() is not the registry's model_tree")
mf = read("tools/ml/mfile.cho")
if "1 constant regression of an int or dec target, 2 constant classification, 3 constant regression of a float" not in mf.replace("\n//", ""):
    fail("tools/ml/mfile.cho no longer documents kinds 1, 2, 3 as shipped")
if not re.search(r"mk < 1 \|\| mk > 3", mf):
    fail("tools/ml/mfile.cho: the accepted kinds are no longer 1 to 3 (update the registry and design.md 4.1 first)")

# ---- stream kinds
streams = [(n, v) for n, v in K.items() if n.startswith("stream_") and n != "stream_count"]
unique("stream kinds (lib/core/kinds.cho)", streams)
doc_streams = [int(r[0]) for r in table_rows(read("docs/rng.md"), "| kind | stream |")]
if doc_streams != sorted(v for _, v in streams) or K.get("stream_count") != len(streams):
    fail("docs/rng.md stream table %s differs from lib/core/kinds.cho %s" % (doc_streams, sorted(v for _, v in streams)))
grow = read("lib/models/tree/grow.cho")
if "rng.derive(rng.derive(p.seed, %d), tree)" % K["stream_tree_bootstrap"] not in grow:
    fail("lib/models/tree/grow.cho: the bootstrap stream is not derive(derive(seed, %d), tree)" % K["stream_tree_bootstrap"])
if "rng.derive(rng.derive(rng.derive(p.seed, %d), tree), id)" % K["stream_tree_features"] not in grow:
    fail("lib/models/tree/grow.cho: the feature stream is not derive(derive(derive(seed, %d), tree), id)" % K["stream_tree_features"])
for fn, name in (("kind_train_test", "stream_train_test"), ("kind_kfold", "stream_kfold"), ("kind_stratified", "stream_stratified")):
    if consts("lib/metrics/splits.cho").get(fn) != K[name]:
        fail("lib/metrics/splits.cho %s is not the registry's %s" % (fn, name))
if consts("lib/models/kmeans/kmeans.cho").get("kind") != K["stream_kmeans"]:
    fail("lib/models/kmeans/kmeans.cho kind() is not the registry's stream_kmeans")
oracle = read("scripts/oracle_metrics.py")
for kind, where in ((K["stream_train_test"], "py_train_test"), (K["stream_kfold"], "py_kfold"), (K["stream_stratified"], "py_strat")):
    body = oracle.split("def %s" % where)[1].split("\ndef ")[0]
    if "stream(seed, %d," % kind not in body:
        fail("scripts/oracle_metrics.py %s does not use stream kind %d" % (where, kind))


# ---- refusal codes: the three libraries' tag(), exit_code(), against design.md 8.1
def tag_fn(path):
    src = read(path).split("pub fn tag")[1].split("pub fn exit_code")[0]
    return {int(c): t for c, t in re.findall(r'if code == (\d+) \{\s*return "([^"]+)";', src)}


def exit_fn(tag):
    return 2 if tag in ("ml.param.out-of-range", "ml.param.threads") else 8


libs = {"ml.refuse": ("lib/core/refuse.cho", 1, 99), "tree.deny": ("lib/models/tree/deny.cho", 101, 199), "metrics.rules": ("lib/metrics/rules.cho", 201, 299)}
code_tag = {}
for lib, (path, lo, hi) in libs.items():
    for c, t in tag_fn(path).items():
        if not lo <= c <= hi:
            fail("%s: code %d is outside the band %d..%d" % (lib, c, lo, hi))
        if c in code_tag:
            fail("code %d is claimed by %s and by %s" % (c, code_tag[c][1], lib))
        code_tag[c] = (t, lib)
    tags = list(tag_fn(path).values())
    if len(set(tags)) != len(tags):
        fail("%s registers a tag twice" % lib)
doc_codes = {}
for r in table_rows(design, "| code | tag |"):
    c = int(r[0])
    doc_codes[c] = (r[1].strip("`"), int(r[2]), r[3].strip("`"))
if {c: (t, exit_fn(t), lib) for c, (t, lib) in code_tag.items()} != doc_codes:
    only_src = sorted(set(code_tag) - set(doc_codes))
    only_doc = sorted(set(doc_codes) - set(code_tag))
    diff = sorted(c for c in set(code_tag) & set(doc_codes) if (code_tag[c][0], exit_fn(code_tag[c][0]), code_tag[c][1]) != doc_codes[c])
    fail("design.md 8.1 differs from the sources: only in the sources %s, only in the document %s, different %s" % (only_src, only_doc, diff))
# one tag, one exit code, in every library
by_tag = {}
for c, (t, lib) in code_tag.items():
    by_tag.setdefault(t, set()).add(exit_fn(t))
# the exit code each library's exit_code() really answers is tested in tests/t_registry.cho; the metrics table() carries its own
met_table = re.search(r'pub fn table\(\)[^{]*\{\s*return "(.*)";', read("lib/metrics/rules.cho")).group(1)
for e in met_table.split(";"):
    t, ex = e.split("|")[:2]
    if int(ex) != exit_fn(t):
        fail("lib/metrics/rules.cho table(): %s has exit %s, the registry says %d" % (t, ex, exit_fn(t)))

# ---- tools/ml/spec.cho: the rules the libraries add that ml cannot reach yet (`pending`) against the libraries and design.md section 8
spec = read("tools/ml/spec.cho")
m = re.search(r'pub fn pending\(\)[^{]*\{\s*return "(.*)";', spec)
if m:
    pending = {}
    for e in m.group(1).split(";"):
        t, ex, rep, summ = e.split("|", 3)
        pending[t] = int(ex)
        if t in re.search(r'pub fn extra\(\)[^{]*\{\s*return "(.*)";', spec).group(1) and t != "":
            pass
    extra_tags = {e.split("|")[0] for e in re.search(r'pub fn extra\(\)[^{]*\{\s*return "(.*)";', spec).group(1).split(";")}
    rules_tags = set(re.search(r'pub fn rules\(\)[^{]*\{\s*return "(.*)";', spec).group(1).split(";"))
    for t, ex in pending.items():
        if t in extra_tags or t in rules_tags:
            fail("spec.cho pending(): %s is already reachable (in extra or rules); it belongs in extra" % t)
        if ex != exit_fn(t):
            fail("spec.cho pending(): %s has exit %d, the registry says %d" % (t, ex, exit_fn(t)))
    lib_tags = {t for (t, _) in code_tag.values()}
    for t in sorted(lib_tags - extra_tags - rules_tags - set(pending)):
        fail("the tag %s of a library is neither reachable from ml (extra, rules) nor in spec.cho pending()" % t)
    for t in sorted(pending):
        if t not in lib_tags:
            fail("spec.cho pending(): %s is a tag no library has" % t)
    # design.md section 8: every library tag has a row with the same exit code, outside code blocks
    sec8 = design.split("## 8. Refusals")[1].split("## 8.1")[0]
    for t in sorted(lib_tags):
        rows = [l for l in sec8.splitlines() if l.startswith("|") and "`%s`" % t in l.split("|")[1:2][0] if len(l.split("|")) > 2]
        exits = {int(l.split("|")[2].strip()) for l in rows if l.split("|")[2].strip().isdigit()}
        if not rows:
            fail("design.md section 8 has no row for the library tag %s" % t)
        elif exits and exits != {exit_fn(t)}:
            fail("design.md section 8: %s has exit %s, the registry says %d" % (t, sorted(exits), exit_fn(t)))
else:
    print("check_registry: tools/ml/spec.cho has no pending() yet; the rule tags are not compared", file=sys.stderr)

if bad:
    print("\n".join("check_registry: " + b for b in bad))
    sys.exit(1)
print("check_registry: %d model kinds, %d stream kinds, %d refusal codes in %d bands: tables, sources and documents agree" % (len(models), len(streams), len(code_tag), len(libs)))

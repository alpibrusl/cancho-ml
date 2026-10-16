"""The mutant modules (docs/modules.md section 7): a model module with one defect each, added to a scratch copy of the tree. `scripts/admit.py` must REFUSE each one,
on the check named, and must ADMIT the copy with no defect (the control).

A module is a directory of lib/models with a manifest and an adapter. The defects:

  mutant-capability           the adapter's file has a function that reads the clock and the tool's main hands it a clock: the authority row grows (Q5)
  mutant-kind-reuse           a model that takes the kind number of `linear` (Q2)
  mutant-unregistered-tag     a model that writes and lists a rule tag no table registers (Q3)
  mutant-sum-outside-order    a model whose adapter adds floats in its own loop, importing no linalg kernel (Q4)
  mutant-threads-one-count    a model whose bytes are the same at one thread count only: any count above 1 changes a byte (Q6)
  mutant-threads-eleven       a model whose bytes are those of linear at every thread count but 11 (Q6: a check at 1, 2, 4, 8 and 16 alone would not see it)
"""

import re

KINDS = "lib/core/kinds.cho"
CANCHO = "cancho.toml"


def manifest(name, kind, kind_fn, emits=None, adds=None, reason=""):
    emits = emits if emits is not None else ["ml.tensor.rank", "ml.tensor.shape", "ml.tensor.bounds", "ml.input.non-finite", "ml.input.too-many-columns", "ml.param.out-of-range",
                                             "ml.linalg.not-positive-definite", "ml.linalg.singular", "ml.param.threads", "ml.fit.non-finite"]
    return '''name = "%(name)s"
kind = %(kind)d
kind_fn = "%(kind_fn)s"
task = "regression"
tier = "experimental"
wired = true
adapter = "models.adapter_%(name)s"
payload_version = 1
doc = "docs/models-linear.md"
files = ["lib/models/%(name)s/adapter.cho"]
plan_order = 7

[tags]
emits = [%(emits)s]

[authority]
adds = [%(adds)s]
reason = "%(reason)s"

[sums]
decisive = []

[oracle]
script = "scripts/oracle_wire.py"
args = "linear"
tolerance = "bits"

[[hyperparameter]]
name = "l2"
type = "float"
nonneg = true
default = "0.0"
doc = "the ridge penalty, a finite number, 0 or more. The intercept is not penalised"
''' % dict(name=name, kind=kind, kind_fn=kind_fn, emits=", ".join('"%s"' % t for t in emits), adds=", ".join('"%s"' % a for a in adds or []), reason=reason)


def adapter(base, name, kind, fit_body=None, extra=""):
    text = base.replace("models.adapter_linear", "models.adapter_" + name).replace("record.kind_linear()", str(kind))
    if fit_body:
        text = text.replace("    return linreg.fit(heap, threads, xt, xd, yd, fp[0], out, diag);", fit_body)
    return text + extra


def add_kind(tree, fn, number):
    """a new entry of the registry (lib/core/kinds.cho): the function and model_count()"""
    p = tree / KINDS
    s = p.read_text()
    s = s.replace("pub fn model_count() -> [] int {", "pub fn %s() -> [] int {\n    return %d;\n}\n\npub fn model_count() -> [] int {" % (fn, number), 1)
    s = re.sub(r"(pub fn model_count\(\) -> \[\] int \{\n    return )\d+;", r"\g<1>%d;" % number, s, count=1)
    p.write_text(s)


def add_model(tree, name, kind, kind_fn, new_kind=True, fit_body=None, extra="", **mf):
    base = (tree / "lib/models/linear/adapter.cho").read_text()
    d = tree / "lib/models" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.toml").write_text(manifest(name, kind, kind_fn, **mf))
    (d / "adapter.cho").write_text(adapter(base, name, kind, fit_body, extra))
    if new_kind:
        add_kind(tree, kind_fn, kind)
    c = tree / CANCHO
    s = c.read_text()
    s = s.replace('"lib/models/logistic",', '"lib/models/logistic", "lib/models/%s",' % name)
    c.write_text(s)


def capability(tree):
    add_model(tree, "capmut", 9, "model_capmut",
              extra="\n// a model that wants the time: not in its manifest\npub fn stamp[&c](clock: &c Clock) -> [clock] int {\n    return clock_ms(clock);\n}\n")
    main = tree / "tools/ml/main.cho"
    s = main.read_text()
    s = s.replace("import ml.est;\n", "import ml.est;\nimport models.adapter_capmut;\n", 1)
    s = s.replace("    release(clock);\n", "    borrow clock as &c in {\n        adapter_capmut.stamp(c);\n    }\n    release(clock);\n", 1)
    main.write_text(s)


def kind_reuse(tree):
    add_model(tree, "dupkind", 4, "model_linear", new_kind=False)


def unregistered_tag(tree):
    add_model(tree, "badtag", 9, "model_badtag", emits=["ml.tensor.rank", "ml.mutant.not-registered"],
              extra='\npub fn why() -> [] &static [byte] {\n    return "ml.mutant.not-registered";\n}\n')


def sum_outside_order(tree):
    body = ("    let c = linreg.fit(heap, threads, xt, xd, yd, fp[0], out, diag);\n"
            "    var s = 0.0;\n    var i = 0;\n    while i < len(yd) {\n        s = s + yd[i];\n        i = i + 1;\n    }\n"
            "    out[0] = out[0] + s - s;\n    return c;")
    add_model(tree, "ownsum", 9, "model_ownsum", fit_body=body)


def threads_one_count(tree):
    body = ("    let c = linreg.fit(heap, threads, xt, xd, yd, fp[0], out, diag);\n"
            "    if threads > 1 {\n        out[0] = out[0] + 1.0;\n    }\n    return c;")
    add_model(tree, "t1only", 9, "model_t1only", fit_body=body)


def threads_eleven(tree):
    body = ("    let c = linreg.fit(heap, threads, xt, xd, yd, fp[0], out, diag);\n"
            "    if threads == 11 {\n        out[0] = out[0] + 1.0;\n    }\n    return c;")
    add_model(tree, "t11", 9, "model_t11", fit_body=body)


# id, the model it adds, the function that adds it, the check that must fail, whether the check needs the binary
MUTANTS = [
    ("mutant-capability", "capmut", capability, "Q5", False),
    ("mutant-kind-reuse", "dupkind", kind_reuse, "Q2", False),
    ("mutant-unregistered-tag", "badtag", unregistered_tag, "Q3", False),
    ("mutant-sum-outside-order", "ownsum", sum_outside_order, "Q4", False),
    ("mutant-threads-one-count", "t1only", threads_one_count, "Q6", True),
    ("mutant-threads-eleven", "t11", threads_eleven, "Q6", True),
]

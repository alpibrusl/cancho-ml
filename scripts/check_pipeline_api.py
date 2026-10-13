#!/usr/bin/env python3
"""Gate P5(a) and (b) (docs/pipeline.md): there is no code path that applies a plan without its stored parameters.

Read from the sources (no compiler):
  1. a `pub fn` that takes a `plan.Plan` is one of the allow-list: the fit (`run.fit`), the check that a plan describes stored steps (`run.agrees`),
     the plan's own accessors (module `pipeline.plan`) -- nothing else, in particular no function that also takes steps bytes or a Step and returns a matrix;
  2. the apply functions (`transforms.apply.apply`, `run.apply_steps`) do not mention `plan.`, `fit.` or `Plan` in their bodies, and their modules do
     not import `transforms.fit` or `pipeline.plan` (apply.cho) -- the parameters of an apply can only come from the stored bytes it is given;
  3. `Loaded` is built only inside `pipeline/file.cho` (`decode` and `empty`), and a `Step` only in `transforms/steps.cho` and by `steps.make` in fit;
  4. every `pub fn` whose result holds a `matrix.Matrix` is on the list below, each with the kind of its inputs (fit / apply / copy), so that a new
     one fails this gate until it is classified here.
Exit 1 and a line per violation."""
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
bad = []


def read(p):
    return (ROOT / p).read_text()


def pub_fns(path):
    src = read(path)
    out = []
    for m in re.finditer(r"pub fn (\w+)(\[[^\]]*\])?\((.*?)\) -> (\[[^\]]*\]) ([^{]*)\{", src, re.S):
        name, _, params, effects, ret = m.groups()
        start = m.end()
        end = src.find("\n}\n", start)
        out.append((name, params, ret.strip(), src[start:end]))
    return out


files = {"transforms/apply.cho": "apply", "transforms/fit.cho": "fit", "transforms/steps.cho": "steps", "pipeline/run.cho": "run", "pipeline/file.cho": "file", "pipeline/plan.cho": "plan", "pipeline/schema.cho": "schema", "pipeline/names.cho": "names", "pipeline/mixed.cho": "mixed", "transforms/catenc.cho": "catenc"}
# the mixed pipeline (docs/categorical-encoders.md) has its own fit, and the numeric fit's column resolver is shared with it
ALLOW_PLAN_PARAM = {("run", "fit"), ("run", "agrees"), ("run", "resolve"), ("mixed", "fit")}
# result holds a matrix: (module, function) -> kind
MATRIX_FNS = {("apply", "apply"): "apply", ("run", "apply_steps"): "apply", ("run", "fit"): "fit", ("run", "clone"): "copy", ("run", "permute"): "copy",
             ("mixed", "fit"): "fit", ("mixed", "apply_steps"): "apply", ("mixed", "hstack"): "copy", ("catenc", "apply"): "apply"}

for rel, mod in files.items():
    for name, params, ret, body in pub_fns("lib/" + rel):
        if "plan.Plan" in params and mod != "plan" and (mod, name) not in ALLOW_PLAN_PARAM:
            bad.append("lib/%s: pub fn %s takes a plan.Plan but is not on the allow-list (only the fit and the agreement check may)" % (rel, name))
        if "Matrix" in ret and (mod, name) not in MATRIX_FNS:
            bad.append("lib/%s: pub fn %s returns a matrix and is not classified in scripts/check_pipeline_api.py" % (rel, name))
        if (mod, name) in MATRIX_FNS and MATRIX_FNS[(mod, name)] == "apply":
            if re.search(r"\bplan\.|\bfit\.|\bPlan\b", body):
                bad.append("lib/%s: the apply function %s mentions the plan or a fit in its body" % (rel, name))
            if "plan.Plan" in params or "fit.Spec" in params:
                bad.append("lib/%s: the apply function %s takes a plan or a spec" % (rel, name))

apply_src = read("lib/transforms/apply.cho")
for imp in ("transforms.fit", "pipeline.plan"):
    if re.search(r"^import %s;" % re.escape(imp), apply_src, re.M):
        bad.append("lib/transforms/apply.cho imports %s" % imp)

# Loaded and Step have one constructor each
for p in list((ROOT / "lib").rglob("*.cho")) + list((ROOT / "tools").rglob("*.cho")):
    rel = str(p.relative_to(ROOT))
    t = p.read_text()
    if "Loaded {" in t and rel != "lib/pipeline/file.cho":
        # a pattern in a `let Loaded { ... } =` destructuring is not a construction; a struct literal is `Loaded { name: value`
        if re.search(r"Loaded \{ \w+:", t):
            bad.append("%s builds a Loaded; only file.decode may (lib/pipeline/file.cho)" % rel)
    if re.search(r"\bStep \{ \w+:", t) and rel != "lib/transforms/steps.cho":
        bad.append("%s builds a Step literal; only lib/transforms/steps.cho may (fit goes through steps.make)" % rel)
file_src = read("lib/pipeline/file.cho")
ctor = [m.start() for m in re.finditer(r"Loaded \{ \w+:", file_src)]
fn_starts = [(m.start(), m.group(1)) for m in re.finditer(r"(?:pub )?fn (\w+)", file_src)]
for pos in ctor:
    owner = [n for s, n in fn_starts if s < pos][-1]
    if owner not in ("decode", "empty"):
        bad.append("lib/pipeline/file.cho: a Loaded is built in %s (only decode and empty may)" % owner)

if bad:
    print("\n".join("check_pipeline_api: " + b for b in bad))
    sys.exit(1)
print("check_pipeline_api: no apply takes a plan or fits; Loaded is built by file.decode only; every matrix-returning pub fn is classified")

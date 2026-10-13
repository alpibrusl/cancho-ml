#!/usr/bin/env python3
"""The admission script of a model module (docs/modules.md section 5, issue alpibrusl/cancho-ml#18).

    python3 scripts/admit.py <model> [--root DIR] [--fast] [--only Q2,Q5] [--build]

Every check prints PASS, FAIL or SKIP (with the reason) and the script exits 1 on any FAIL. A model may be `supported` only if none fails.
`--root` names the tree to judge (default: this repository; tests/modules/run.py judges a scratch copy with one defect added). `--build` builds the binary of the tree
first (`build/ml`, from the explicit file list of the manifests), which the scratch copies need; the repository's own `build/ml` is built by `cancho build --bin ml`.
`--fast` leaves out the checks that re-run long scripts: Q9 (the oracle), Q10 (the mutant record, a read) and Q11 (scikit-learn).

    Q0  doc first          Q1  manifest and generated glue     Q2  registry numbers          Q3  rule tags
    Q4  sums               Q5  authority                       Q6  thread identity 1..16     Q7  damaged files
    Q8  file sizes         Q9  the oracle, bit for bit         Q10 mutants                   Q11 scikit-learn
"""

import glob
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_models  # noqa: E402

CHECKS = ["Q%d" % i for i in range(12)]
TRAPS = {132, 134, 136, 139}

# the operations of the dispatch, each with the kind argument first (docs/modules.md section 3)
EST_OPS = ["name", "tier", "task", "nints", "nfloats", "takes", "diag_len", "out_len", "payload_size", "fit", "predict", "proba", "encode", "decode", "hparams"]

# the interface of an adapter (docs/modules.md section 4), types without regions
ADAPTER = {
    "diag_len": (["d: int", "ip: &[int]"], "int"),
    "out_len": (["d: int"], "int"),
    "fit": (["heap: &!Heap", "threads: int", "xt: Tensor", "xd: &[float]", "yd: &[float]", "ip: &[int]", "fp: &[float]", "seed: int", "out: &![float]", "diag: &![float]"], "int"),
    "predict": (["xt: Tensor", "xd: &[float]", "wv: &[float]", "out: &![float]"], "int"),
    "proba": (["xt: Tensor", "xd: &[float]", "wv: &[float]", "out: &![float]"], "int"),
    "payload_size": (["d: int"], "int"),
    "encode": (["dst: &![byte]", "at: int", "d: int", "wv: &[float]", "c0: int", "c1: int"], "int"),
    "decode": (["src: &[byte]", "d: int", "out: &![float]"], "Decoded"),
}


def compiler(root):
    here = pathlib.Path(root).parent / "cancho-compiler-a4572ea" / "target" / "release" / "cancho"
    cand = [os.environ.get("CANCHO"), str(here), str(HERE.parent.parent / "cancho-compiler-a4572ea" / "target" / "release" / "cancho")]
    for c in cand:
        if c and pathlib.Path(c).exists():
            return c
    return "cancho"


# ------------------------------------------------------------------------------------------------ the binary of a tree
def table_sources():
    t = pathlib.Path(os.environ.get("CANCHO_TABLE", str(HERE.parent.parent / "cancho-table-main"))) / "tools" / "table"
    return [str(t / (m + ".cho")) for m in ("pow5", "dec", "flt", "query", "facc", "reader")]


def ml_sources(root, models):
    """The explicit file list of `cancho build --bin ml` for the tree: the libraries, the directories of the wired models, the tool, the dependencies"""
    root = pathlib.Path(root)

    def g(pat):
        return sorted(glob.glob(str(root / pat)))

    dirs = {"lib/models/kmeans"}
    for m in gen_models.wired(models):
        for f in m["files"]:
            dirs.add(str(pathlib.PurePosixPath(f).parent))
        dirs.add("lib/models/" + m["name"])
    files = g("lib/core/*.cho") + g("lib/linalg/*.cho") + g("lib/metrics/*.cho") + g("lib/models/*.cho")
    for d in sorted(dirs):
        files += g(d + "/*.cho")
    files += g("lib/transforms/*.cho") + g("lib/pipeline/*.cho") + g("build/deps/*.cho") + [str(root / "generated/ml/built.cho")] + g("tools/ml/*.cho") + table_sources()
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def build_ml(root):
    root = pathlib.Path(root)
    models = gen_models.load_manifests(root)
    (root / "build").mkdir(exist_ok=True)
    out = root / "build" / "ml"
    p = subprocess.run([compiler(root), "build", "--std", *ml_sources(root, models), "-o", str(out)], capture_output=True, text=True, cwd=root)
    return (out if p.returncode == 0 else None), p.stdout + p.stderr


# ------------------------------------------------------------------------------------------------ helpers
class Ctx:
    def __init__(self, root, name):
        self.root = pathlib.Path(root).resolve()
        self.name = name
        self.models = gen_models.load_manifests(self.root)
        self.model = next((m for m in self.models if m["name"] == name), None)
        if self.model is None:
            raise SystemExit("admit: no model called %r under %s/lib/models" % (name, self.root))
        with open(self.root / "lib" / "models" / "contract.toml", "rb") as f:
            self.contract = tomllib.load(f)
        self.bin = pathlib.Path(os.environ.get("ML_BIN", str(self.root / "build" / "ml")))

    def read(self, rel):
        return (self.root / rel).read_text()

    def ml(self, *args, cwd=None, timeout=300):
        p = subprocess.run([str(self.bin), *[str(a) for a in args]], capture_output=True, cwd=cwd, timeout=timeout)
        return p.returncode, p.stdout, p.stderr


def git(root, *args):
    p = subprocess.run(["git", *args], capture_output=True, text=True, cwd=root)
    return p.returncode, p.stdout.strip()


def first_commit(root, rel):
    rc, out = git(root, "log", "--diff-filter=A", "--format=%H", "--follow", "--", rel)
    lines = out.splitlines() if rc == 0 else []
    return lines[-1] if lines else None


def adapter_file(c):
    """the file that declares the adapter's module"""
    mod = c.model["adapter"]
    for f in c.model["files"]:
        p = c.root / f
        if p.exists() and re.search(r"^module %s;" % re.escape(mod), p.read_text(), re.M):
            return p
    return None


def norm_params(params):
    out = []
    depth = 0
    cur = ""
    for ch in params:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    res = []
    for p in out:
        name, _, ty = p.partition(":")
        ty = ty.strip()
        ty = re.sub(r"&(!?)[a-z][A-Za-z0-9]* ", r"&\1", ty)       # &x [float] -> &[float]
        ty = re.sub(r"&!h Heap", "&!Heap", ty)
        ty = ty.replace("tensor.Tensor", "Tensor").replace("record.Decoded", "Decoded")
        res.append("%s: %s" % (name.strip(), ty))
    return res


def signature(text, fn):
    m = re.search(r"pub fn %s(?:\[[^\]]*\])?\((.*?)\) -> \[([^\]]*)\] ([\w.&!\[\] ]+?) \{" % re.escape(fn), text, re.S)
    if not m:
        return None
    ret = m.group(3).replace("record.Decoded", "Decoded")
    return norm_params(m.group(1)), ret, m.group(2)


# ------------------------------------------------------------------------------------------------ the checks
def q0(c):
    m = c.model
    doc = c.root / m["doc"]
    if not doc.exists():
        return "FAIL", "the doc %s does not exist" % m["doc"]
    if not re.search(r"^#+ .*\bgates?\b", doc.read_text(), re.I | re.M):
        return "FAIL", "%s has no section on gates" % m["doc"]
    rc, out = git(c.root, "rev-parse", "--is-inside-work-tree")
    if rc != 0:
        return "SKIP", "not a git checkout"
    d = first_commit(c.root, m["doc"])
    if d is None:
        return "SKIP", "the doc is not committed yet: commit it before the code (the real admission runs in a committed tree)"
    for f in m["files"]:
        if not (c.root / f).exists():
            return "FAIL", "%s is listed in files and does not exist" % f
        fc = first_commit(c.root, f)
        if fc is None:
            return "SKIP", "%s is not committed yet" % f
        rc, _ = git(c.root, "merge-base", "--is-ancestor", d, fc)
        if rc != 0:
            return "FAIL", "%s was committed before (or apart from) its doc %s" % (f, m["doc"])
    return "PASS", "the doc %s (a section on gates) was committed before all %d files" % (m["doc"], len(m["files"]))


def q1(c):
    try:
        fresh = gen_models.generate(c.root)
    except gen_models.ManifestError as e:
        return "FAIL", "manifest: %s" % e
    stale = [rel for rel, text in fresh.items() if not (c.root / rel).exists() or (c.root / rel).read_text() != text]
    if stale:
        return "FAIL", "stale generated files: %s (run scripts/gen_models.py)" % ", ".join(stale)
    m = c.model
    if not m["wired"] or m.get("adapter") == "builtin":
        return "PASS", "manifest parses, generated files current (%s has no adapter to compare)" % m["name"]
    est = c.read("tools/ml/est.cho")
    for op in EST_OPS:
        mm = re.search(r"pub fn %s(?:\[[^\]]*\])?\(.*?\n\}\n" % op, est, re.S)
        if not mm:
            return "FAIL", "tools/ml/est.cho has no function %s" % op
        if "kind == %d" % m["kind"] not in mm.group(0) and op not in ("hparams", "takes", "nints", "nfloats", "name", "tier", "task") and "kind == %d" % m["kind"] not in est:
            return "FAIL", "est.%s has no branch for kind %d" % (op, m["kind"])
        if op in ("fit", "predict", "proba", "encode", "decode", "diag_len", "out_len", "payload_size", "hparams") and "kind == %d" % m["kind"] not in mm.group(0):
            return "FAIL", "est.%s has no branch for kind %d" % (op, m["kind"])
    af = adapter_file(c)
    if af is None:
        return "FAIL", "no file of the module declares %s" % m["adapter"]
    text = af.read_text()
    for fn, (params, ret) in ADAPTER.items():
        sig = signature(text, fn)
        if sig is None:
            return "FAIL", "the adapter has no `pub fn %s`" % fn
        if sig[0] != params or sig[1] != ret:
            return "FAIL", "the adapter's %s is %s -> %s, the contract is %s -> %s" % (fn, sig[0], sig[1], params, ret)
    cancho = (c.root / "cancho.toml").read_text() if (c.root / "cancho.toml").exists() else ""
    d = str(pathlib.PurePosixPath(m["files"][0]).parent)
    if cancho and '"%s"' % d not in cancho:
        return "FAIL", "cancho.toml's sources for `ml` do not name the module's directory %s" % d
    return "PASS", "manifest parses, generated files current, every operation of est.cho has the branch, the adapter has the %d functions of the contract" % len(ADAPTER)


def q2(c):
    problems = gen_models.check_registry(c.models, c.root)
    if problems:
        return "FAIL", "; ".join(problems)
    return "PASS", "kind %d%s is unique and is what %s() returns in lib/core/kinds.cho" % (c.model["kind"], "".join(", %d" % k for k in c.model.get("extra_kinds", [])), c.model["kind_fn"])


def known_tags(c):
    """the rule tags ml answers: spec.tmpl's extra() and pending(), by their tables"""
    text = c.read("tools/ml/spec.tmpl")
    tags = set()
    for fn in ("extra", "pending"):
        m = re.search(r"pub fn %s\(\)[^{]*\{\s*return \"((?:[^\"\\]|\\.)*)\";" % fn, text)
        tags |= {e.split("|")[0] for e in m.group(1).split(";")}
    pend = re.search(r"pub fn pending\(\)[^{]*\{\s*return \"((?:[^\"\\]|\\.)*)\";", text)
    return tags, {e.split("|")[0] for e in pend.group(1).split(";")}


def refuse_tags(c):
    """{module alias: {function: tag}} for ml.refuse (lib/core/refuse.cho), metrics.rules and transforms.codes"""
    out = {}
    for alias, rel in (("refuse", "lib/core/refuse.cho"), ("rules", "lib/metrics/rules.cho"), ("codes", "lib/transforms/codes.cho")):
        text = c.read(rel)
        codes = {}
        for m in re.finditer(r"pub fn (\w+)\(\) -> \[\] int \{\s*return (\d+);", text):
            codes[int(m.group(2))] = m.group(1)
        tags = {}
        for m in re.finditer(r"if code == (\d+) \{\s*return \"([a-z.\-]+)\";", text):
            tags[codes.get(int(m.group(1)))] = m.group(2)
        out[alias] = tags
    return out


def q3(c):
    m = c.model
    known, pending = known_tags(c)
    bad = [t for t in m["tags"]["emits"] if t not in known]
    if bad:
        return "FAIL", "emits tags no table registers: %s" % ", ".join(bad)
    emits = set(m["tags"]["emits"])
    by = refuse_tags(c)
    fixtures = c.read("tests/conformance/fixtures.py") if (c.root / "tests/conformance/fixtures.py").exists() else ""
    for t in sorted(emits):
        if t not in pending and 'bad("%s"' % t not in fixtures and t not in ("ml.model.malformed", "ml.label.single-class", "ml.label.not-binary", "ml.label.not-integer", "ml.metric.undefined"):
            return "FAIL", "%s is emitted, is not in pending() and no fixture reaches it" % t
    for rel in m["files"]:
        text = (c.root / rel).read_text()
        for lit in re.findall(r'"((?:ml|args|path|io|parse|limit|conflict)\.[a-z0-9.\-]+)"', text):
            if lit not in emits:
                return "FAIL", "%s writes the tag %s, which the manifest does not list in emits" % (rel, lit)
        for alias, fn in re.findall(r"\b(refuse|rules|codes)\.(\w+)\(\)", text):
            tag = by.get(alias, {}).get(fn)
            if tag and tag not in emits:
                return "FAIL", "%s calls %s.%s(), the tag %s, which is not in emits" % (rel, alias, fn, tag)
    return "PASS", "%d tags, all registered; each reachable one has a fixture; the sources write and call no tag outside emits" % len(emits)


def float_vars(text):
    return set(re.findall(r"\bvar (\w+) = -?[0-9]+\.[0-9]+;", text)) | set(re.findall(r"\bvar (\w+): float\b", text))


def q4(c):
    m = c.model
    scanned = 0
    for rel in sorted(set(m["files"]) | set(m["sums"]["decisive"])):
        p = c.root / rel
        if not p.exists():
            continue
        text = p.read_text()
        scanned += 1
        fv = float_vars(text)
        hits = [v for v in fv if re.search(r"^\s*%s = %s [+-] " % (re.escape(v), re.escape(v)), text, re.M)]
        if hits and not re.search(r"^import linalg\.(order|dot|gemv|syrk|gemm|axpy|par);", text, re.M) and "linalg" not in text.split("\n\n")[0]:
            return "FAIL", "%s accumulates the float variable(s) %s in its own order and imports no linalg kernel or linalg.order" % (rel, ", ".join(sorted(hits)))
    return "PASS", "%d files scanned: no float is accumulated outside the package order" % scanned


def q5(c):
    mf = c.root / "scripts" / "manifest.py"
    p = subprocess.run([sys.executable, str(mf), "--check"], capture_output=True, text=True, cwd=c.root,
                       env=dict(os.environ, CANCHO=compiler(c.root)))
    if p.returncode != 0:
        return "FAIL", "manifest.py --check: " + " | ".join(l for l in p.stdout.splitlines() if l.startswith("FAIL"))[:300]
    base = set(c.contract["authority"]["base"])
    committed = json.loads((c.root / "manifests" / "ml.authority.json").read_text())
    labels = {l["name"] if l["argument"] is None else '%s("%s")' % (l["name"], l["argument"]) for l in committed["labels"]}
    declared = set()
    for mm in c.models:
        declared |= set(mm["authority"]["adds"])
    undeclared = labels - base - declared
    if undeclared:
        return "FAIL", "the authority of ml has %s, which no manifest declares in [authority] adds" % ", ".join(sorted(undeclared))
    mine = set(c.model["authority"]["adds"])
    return "PASS", "the derived authority equals the committed manifest; %s" % ("this module adds %s (%s)" % (", ".join(sorted(mine)), c.model["authority"]["reason"][:60]) if mine else "this module adds no label")


def sample_csv(task, n=1500):
    rng = random.Random(1234)
    rows = []
    for i in range(n):
        x = [rng.uniform(-3, 3) for _ in range(5)]
        z = x[0] * 1.5 - x[1] + 0.5 * x[2] + rng.gauss(0, 0.7)
        if task == "regression":
            rows.append("%s,%.6f" % (",".join("%.6f" % v for v in x), z + 2.0))
        else:
            rows.append("%s,%d" % (",".join("%.6f" % v for v in x), 1 if z + rng.gauss(0, 1.5) > 0 else 0))
    head = "f0,f1,f2,f3,f4," + ("y" if task == "regression" else "lab")
    return head + "\n" + "\n".join(rows) + "\n"


def needs_binary(c):
    if not c.bin.exists():
        return "SKIP", "no binary at %s (build it, or pass --build)" % c.bin
    return None


def fit_args(c, out, extra=()):
    task = c.model["task"]
    return ["fit", "--model", c.name, "--target", "y" if task == "regression" else "lab", "--output", out, "--step", "scale column=* kind=standard", *extra, "d.csv"]


def q6(c):
    if not c.model["wired"] or c.model.get("adapter") == "builtin":
        return "SKIP", "%s has no threaded fit" % c.name
    s = needs_binary(c)
    if s:
        return s
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        (tmp / "d.csv").write_text(sample_csv(c.model["task"]))
        files, stdouts = {}, {}
        for t in c.contract["limits"]["thread_counts"]:
            rc, out, err = c.ml("--root", tmp, *fit_args(c, "m%d.cml" % t, ["--threads", t, "--overwrite"]))
            if rc != 0:
                return "FAIL", "fit --threads %d exited %d: %s" % (t, rc, out[:200].decode(errors="replace"))
            files[t] = (tmp / ("m%d.cml" % t)).read_bytes()
            stdouts[t] = out.replace(b"m%d.cml" % t, b"M")
        ref = files[1]
        differing = [t for t, b in files.items() if b != ref]
        if differing:
            return "FAIL", "the model file at --threads %s differs from the one at --threads 1" % ", ".join(map(str, differing))
        outs = [t for t, o in stdouts.items() if o != stdouts[1]]
        if outs:
            return "FAIL", "the answer at --threads %s differs" % ", ".join(map(str, outs))
    return "PASS", "fit --threads 1..16 gives one model file (%d bytes) and one answer" % len(ref)


def q7(c):
    if not c.model["wired"] or c.model.get("adapter") == "builtin":
        return "SKIP", "no model file of its own"
    s = needs_binary(c)
    if s:
        return s
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        (tmp / "d.csv").write_text(sample_csv(c.model["task"], 200))
        rc, out, err = c.ml("--root", tmp, *fit_args(c, "m.cml"))
        if rc != 0:
            return "FAIL", "fit exited %d: %s" % (rc, out[:200].decode(errors="replace"))
        data = (tmp / "m.cml").read_bytes()
        import hashlib

        def seal(b):
            b = bytes(b[:-32])
            return b + hashlib.sha256(b).digest()

        cases = [(data[:k], False) for k in range(0, len(data), 3)]
        rng = random.Random(7)
        for at in range(0, len(data), 2):
            b = bytearray(data)
            b[at] ^= 0x01 << rng.randrange(8)
            cases.append((bytes(b), False))
        for _ in range(250):
            b = bytearray(data)
            for _ in range(rng.randint(1, 4)):
                b[rng.randrange(len(b))] = rng.randrange(256)
            cases.append((seal(b), True))
        bad = 0
        for i, (blob, repaired) in enumerate(cases):
            (tmp / "x.cml").write_bytes(blob)
            rc, out, err = c.ml("--root", tmp, "predict", "--model-file", "x.cml", "--limit", "2", "d.csv")
            if rc in TRAPS or rc < 0:
                return "FAIL", "a damaged model file trapped (status %d, case %d)" % (rc, i)
            try:
                doc = json.loads(out)
            except ValueError:
                return "FAIL", "a damaged model file got no JSON document (status %d, case %d)" % (rc, i)
            if rc == 0 and not repaired:
                return "FAIL", "a damaged file with a wrong checksum was accepted (case %d)" % i
            if rc != 0 and "rule" not in doc.get("error", {}):
                return "FAIL", "a refusal without a rule tag (case %d)" % i
            bad += rc != 0
    return "PASS", "%d damaged files (truncations, flips, repaired-checksum damage): none trapped, none accepted with a wrong checksum, %d refused with a tag" % (len(cases), bad)


def q8(c):
    limit = c.contract["limits"]["max_file_lines"]
    files = {str(c.root / f) for f in c.model["files"]} | set(glob.glob(str(c.root / "lib" / "models" / c.name / "*.cho")))
    big = []
    for f in sorted(files):
        if os.path.exists(f):
            n = len(pathlib.Path(f).read_text().splitlines())
            if n > limit:
                big.append("%s (%d lines)" % (os.path.relpath(f, c.root), n))
    if big:
        return "FAIL", "over %d lines: %s" % (limit, ", ".join(big))
    return "PASS", "%d files, none over %d lines" % (len(files), limit)


def q9(c):
    o = c.model.get("oracle")
    if not o:
        return "FAIL", "no [oracle] in the manifest"
    s = needs_binary(c)
    if s:
        return s
    script = c.root / o["script"]
    if not script.exists():
        return "FAIL", "the oracle %s does not exist" % o["script"]
    if script.name.startswith("test_"):
        # a unittest module of the conformance tests (the constant model's exact-rational oracle)
        p = subprocess.run([sys.executable, "-m", "unittest", script.stem], capture_output=True, text=True, env=dict(os.environ, ML_BIN=str(c.bin)), cwd=script.parent)
        if p.returncode != 0:
            return "FAIL", (p.stdout + p.stderr)[-300:]
        return "PASS", "%s: %s" % (o["script"], (p.stderr.strip().splitlines() or ["ok"])[-1])
    p = subprocess.run([sys.executable, str(script), "--suite", "30", o["args"]], capture_output=True, text=True, env=dict(os.environ, ML_BIN=str(c.bin)), cwd=c.root)
    if p.returncode != 0:
        return "FAIL", (p.stdout + p.stderr)[-300:]
    return "PASS", p.stdout.strip().splitlines()[-1]


def q10(c):
    rec = c.root / "docs" / "bench" / ("mutants_%s.json" % c.name)
    if not rec.exists():
        return "FAIL", "no mutant record %s (run the mutant scripts and write it)" % rec.relative_to(c.root)
    data = json.loads(rec.read_text())
    total = 0
    for suite in data["suites"]:
        script = c.root / suite["script"]
        p = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, cwd=c.root)
        m = re.search(r"(\d+) (?:\w+ )?mutants?", p.stdout + p.stderr)
        live = int(m.group(1)) if m else -1
        if live != suite["patterns"]:
            return "FAIL", "%s has %d patterns and the record says %d: the record is stale" % (suite["script"], live, suite["patterns"])
        unexplained = [s["id"] for s in suite["survivors"] if not s.get("why")]
        if unexplained:
            return "FAIL", "survivors with no reason: %s" % ", ".join(unexplained)
        if suite["killed"] + len(suite["survivors"]) != suite["patterns"]:
            return "FAIL", "%s: killed + survivors != patterns" % suite["script"]
        total += suite["patterns"]
    return "PASS", "%d patterns in %d suites, all killed or explained (record from %s)" % (total, len(data["suites"]), data.get("machine", "?"))


def q11(c):
    if c.model.get("adapter") == "builtin":
        return "SKIP", "%s is exact integer arithmetic, compared with fractions.Fraction (Q9); scikit-learn's DummyRegressor would add nothing with a stated tolerance" % c.name
    s = needs_binary(c)
    if s:
        return s
    script = c.root / "scripts" / "gate_wire_sklearn.py"
    if not script.exists():
        return "SKIP", "no scikit-learn gate for this model"
    p = subprocess.run([sys.executable, str(script), c.name], capture_output=True, text=True, env=dict(os.environ, ML_BIN=str(c.bin)), cwd=c.root)
    if p.returncode != 0:
        return "FAIL", (p.stdout + p.stderr)[-300:]
    return "PASS", "%d comparisons within the stated tolerances" % p.stdout.count("observed")


FUNCS = {"Q0": q0, "Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5, "Q6": q6, "Q7": q7, "Q8": q8, "Q9": q9, "Q10": q10, "Q11": q11}
NAMES = {"Q0": "doc first", "Q1": "manifest and generated glue", "Q2": "registry numbers", "Q3": "rule tags", "Q4": "sums", "Q5": "authority", "Q6": "thread identity",
         "Q7": "damaged files", "Q8": "file sizes", "Q9": "oracle", "Q10": "mutants", "Q11": "scikit-learn"}
SLOW = {"Q9", "Q10", "Q11"}


def admit(root, name, fast=False, only=None, build=False, out=print):
    """{check: (status, message)}; also printed"""
    if build:
        binary, log = build_ml(root)
        if binary is None:
            out("build FAIL " + log[:600])
            return {"build": ("FAIL", log[:600])}
    c = Ctx(root, name)
    res = {}
    for q in CHECKS:
        if only and q not in only:
            continue
        if fast and q in SLOW:
            res[q] = ("SKIP", "--fast")
        else:
            try:
                res[q] = FUNCS[q](c)
            except Exception as e:  # noqa: BLE001  a check that cannot run is a failure, not a pass
                res[q] = ("FAIL", "the check itself failed: %s: %s" % (type(e).__name__, e))
        out("%-4s %-4s %-28s %s" % (q, res[q][0], NAMES[q], res[q][1]))
    return res


def main(argv):
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    name = argv[0]
    root = pathlib.Path(argv[argv.index("--root") + 1]) if "--root" in argv else HERE.parent
    only = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None
    res = admit(root, name, fast="--fast" in argv, only=only, build="--build" in argv)
    failed = [q for q, (s, _) in res.items() if s == "FAIL"]
    print("admit %s: %s" % (name, "ADMITTED (%d PASS, %d SKIP)" % (sum(1 for s, _ in res.values() if s == "PASS"), sum(1 for s, _ in res.values() if s == "SKIP")) if not failed else "REFUSED: " + ", ".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

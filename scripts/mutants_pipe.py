#!/usr/bin/env python3
"""Mutants of lib/transforms and lib/pipeline (docs/transforms.md section 5, docs/pipeline.md section 6). A mutant is a name and one or more edits
(file, old, new); every `old` must occur exactly once in its file. Each is applied to a COPY of lib/, tests/, tools/ and scripts/ in a temporary
directory (the sources are never touched), `sh scripts/gate_pipe.sh` is run in the copy with PIPE_FAST=1, and the mutant is killed if the gate exits
non-zero. A mutant that does not compile is INVALID, not killed. The unmutated copy must pass first. A survivor needs a reason in ALLOWED.

    python3 scripts/mutants_pipe.py --check          every pattern applies once (no compiler)
    python3 scripts/mutants_pipe.py [name-substring ...]
Exit 1 if a mutant survives unexplained, is invalid, or cannot apply."""
import os, pathlib, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CANCHO = os.environ.get("CANCHO", str(ROOT.parent / "cancho-compiler-a4572ea" / "target" / "release" / "cancho"))
TABLE = os.environ.get("TABLE_SRC", str(ROOT.parent / "cancho-table-main" / "tools" / "table"))

F = "lib/transforms/fit.cho"
A = "lib/transforms/apply.cho"
S = "lib/transforms/steps.cho"
D = "lib/transforms/dsum.cho"
SO = "lib/transforms/sortf.cho"
C = "lib/transforms/codes.cho"
P = "lib/pipeline/plan.cho"
SC = "lib/pipeline/schema.cho"
FI = "lib/pipeline/file.cho"
R = "lib/pipeline/run.cho"
K = "lib/core/kinds.cho"

HELP_MINMAX = """fn colmin[&m](x: &m matrix.Matrix, c: int, n: int) -> [] float {
    var lo = matrix.value(x, 0, c);
    var r = 0;
    while r < n {
        if matrix.value(x, r, c) < lo {
            lo = matrix.value(x, r, c);
        }
        r = r + 1;
    }
    return lo;
}

fn colmax[&m](x: &m matrix.Matrix, c: int, n: int) -> [] float {
    var hi = matrix.value(x, 0, c);
    var r = 0;
    while r < n {
        if matrix.value(x, r, c) > hi {
            hi = matrix.value(x, r, c);
        }
        r = r + 1;
    }
    return hi;
}

fn gone[&h]"""

M = [
    # ---- fit: parameters
    ("impute-constant-column-is-not-its-own-mean", [(F, "                if constant {\n                    fill = lo;", "                if false {\n                    fill = lo;")]),
    ("median-even-count-not-averaged", [(F, "fill = (vals[cnt / 2 - 1] + vals[cnt / 2]) / 2.0;", "fill = vals[cnt / 2];")]),
    ("median-parity-swapped", [(F, "if cnt % 2 == 1 {", "if cnt % 2 == 0 {")]),
    ("impute-mean-divides-by-count-plus-one", [(F, "fill = dsum.total(heap, vals, cnt) / float_of(cnt);", "fill = dsum.total(heap, vals, cnt) / float_of(cnt + 1);")]),
    ("minmax-parameter-sign", [(F, "var d = hi - lo;", "var d = lo - hi;")]),
    ("minmax-constant-divisor-removed", [(F, "if d == 0.0 {", "if d == 2.0 {")]),
    ("standard-ddof-ignored", [(F, "let den = cnt - sp.a;", "let den = cnt;")]),
    ("standard-ddof-swapped", [(F, "let den = cnt - sp.a;", "let den = cnt - (1 - sp.a);")]),
    ("standard-constant-rule-removed", [(F, "let constant = lo == hi;", "let constant = false;")]),
    ("standard-zero-std-guard-removed", [(F, "if sd > 0.0 {", "if sd >= 0.0 {")]),
    ("standard-mean-sign", [(F, "mean = dsum.total(heap, vals, cnt) / float_of(cnt);", "mean = 0.0 - dsum.total(heap, vals, cnt) / float_of(cnt);")]),
    ("quantile-position-rule", [(F, "let num = (m - 1) * i;", "let num = m * i;")]),
    ("quantile-no-interpolation", [(F, "return v[q] + (v[q + 1] - v[q]) * (float_of(r) / float_of(bins));", "return v[q + 1];")]),
    ("quantile-fraction-denominator", [(F, "(float_of(r) / float_of(bins));", "(float_of(r) / float_of(bins + 1));")]),
    ("width-edge-index-shifted", [(F, "let e = lo + range * float_of(i) / float_of(bins);", "let e = lo + range * float_of(i + 1) / float_of(bins);")]),
    ("width-edge-denominator", [(F, "let e = lo + range * float_of(i) / float_of(bins);", "let e = lo + range * float_of(i) / float_of(bins + 1);")]),
    ("all-missing-column-accepted", [(F, "if cnt == 0 && !(kind == 1 && sp.a == 2) {", "if false {")]),
    ("categories-bound-removed", [(F, "if dn > maxcat {", "if false {")]),
    ("non-finite-input-accepted-by-fit", [(F, "            if !finite(v) {\n                return (0 - 1, 0.0, 0.0);\n            }", "            if false {\n                return (0 - 1, 0.0, 0.0);\n            }")]),
    ("quantile-edge-clamp-removed", [(F, "if i > 1 && e < fl[j * w + i - 2] {", "if false {")]),
    # ---- apply
    ("bucket-edge-goes-to-the-lower-bucket", [(A, "if f[from + mid] <= v {", "if f[from + mid] < v {")]),
    ("unknown-category-rule-swapped", [(A, "if a[base] == 1 {", "if a[base] == 0 {")]),
    ("unknown-label-code-is-zero", [(A, "matrix.put(contents(dw), cols, r, o, 0.0 - 1.0);", "matrix.put(contents(dw), cols, r, o, 0.0);")]),
    ("unknown-category-not-counted", [(A, "unknown = unknown + 1;", "unknown = unknown + 0;")]),
    ("impute-leaves-the-cell-missing", [(A, "if j >= 0 && kind == 1 {", "if j >= 0 && kind == 9 {")]),
    ("missing-cell-passed-as-a-value", [(A, "} else {\n                                        matrix.mark_missing(contents(mw), wpc, r, o);", "} else {\n                                        matrix.put(contents(dw), cols, r, o, 0.0);")]),
    ("onehot-missing-block-is-one-cell", [(A, "matrix.mark_missing(contents(mw), wpc, r, o + t2);", "matrix.mark_missing(contents(mw), wpc, r, o);")]),
    ("category-lookup-descending", [(A, "if e < v {", "if e > v {")]),
    ("minmax-multiplies-by-the-divisor", [(A, "(v - f[2 * j]) / f[2 * j + 1];", "(v - f[2 * j]) * f[2 * j + 1];")]),
    ("overflow-of-a-scaled-value-not-refused", [(A, "if !(z - z == 0.0) {", "if false {")]),
    ("fit-on-the-test-rows", [(A, "fn gone[&h]", HELP_MINMAX), (A, "let z = (v - f[2 * j]) / f[2 * j + 1];", "let z = (v - colmin(x, c, n)) / (colmax(x, c, n) - colmin(x, c, n));")]),
    # ---- records
    ("decode-columns-order-unchecked", [(S, "if c <= prev || c >= cin {", "if false {")]),
    ("decode-divisor-unchecked", [(S, "if !(f[2 * j + 1] > 0.0) {", "if false {")]),
    ("decode-category-order-unchecked", [(S, "if !(f[at + t - 1] < f[at + t]) {", "if false {")]),
    ("decode-category-count-bound-removed", [(S, "if n < 1 || n > max_categories() {", "if n < 1 {")]),
    ("encode-floats-dtype-tag", [(S, "    o = put32(heap, o, 2);\n    o = put32(heap, o, rank);", "    o = put32(heap, o, 3);\n    o = put32(heap, o, rank);")]),
    ("onehot-width-forgets-the-replaced-column", [(S, "return a[1] - a[0] + category_total(a);", "return a[1] + category_total(a);")]),
    # ---- decisive sums
    ("dsum-total-adds-magnitudes", [(D, "return run(heap, sums.kind_val(), vals, 0.0, n);", "return run(heap, sums.kind_abs(), vals, 0.0, n);")]),
    ("dsum-total-in-another-order", [(D, "return run(heap, sums.kind_val(), vals, 0.0, n);", "var s = 0.0;\n    var i = 0;\n    while i < n {\n        s = s + vals[i];\n        i = i + 1;\n    }\n    return s;")]),
    ("sort-descending", [(SO, "if a[root] < a[child] {", "if a[root] > a[child] {")]),
    # ---- plan
    ("plan-key-order-changed", [(P, "return \"column:cols:req kind:enum=minmax,standard:req ddof:int=0,1:=0@kind=standard\";", "return \"kind:enum=minmax,standard:req column:cols:req ddof:int=0,1:=0@kind=standard\";")]),
    ("plan-default-not-written", [(P, "ddof:int=0,1:=0@kind=standard", "ddof:int=0,1:opt@kind=standard")]),
    ("plan-number-spelling-kept", [(P, "return (buffer.push_nat(heap, out, n), 0, n, 0.0, 0);", "return (buffer.append(heap, out, v), 0, n, 0.0, 0);")]),
    ("plan-column-off-by-one", [(P, "let vcol = tk[0] - ls + 1;", "let vcol = tk[0] - ls;")]),
    ("plan-column-list-not-sorted", [(P, "contents(s)[2 * key + 1]]) > 0 {", "contents(s)[2 * key + 1]]) < 0 {")]),
    ("plan-step-after-model-accepted", [(P, "            if model_seen {", "            if false {")]),
    ("plan-unknown-key-ignored", [(P, "if !has_key(defs1, key) && !has_key(defs2, key) {", "if false {")]),
    ("plan-duplicate-key-ignored", [(P, "            if t2 > 0 {", "            if false {")]),
    # ---- schema
    ("schema-by-position", [(SC, "let at = names.find(given, nm);", "let at = f;")]),
    ("schema-repair-names-the-wrong-side", [(SC, "d = buffer.append(heap, d, \"missing: \");", "d = buffer.append(heap, d, \"extra: \");"), (SC, "d = buffer.append(heap, d, \"extra: \");\n        borrow extra", "d = buffer.append(heap, d, \"missing: \");\n        borrow extra")]),
    ("schema-type-ignored", [(SC, "if given_types[2 * at] != type_of(sc, j) || given_types[2 * at + 1] != scale_of(sc, j) {", "if false {")]),
    # ---- file
    ("file-checksum-not-verified", [(FI, "if !checksum_ok(heap, data) {", "if false {")]),
    ("file-plan-and-steps-may-disagree", [(FI, "    if !agree {", "    if false {")]),
    ("file-section-order-unchecked", [(FI, "if tag != kinds.section_at(s + 1) {", "if false {")]),
    ("file-model-family-unchecked", [(FI, " || !model_family_ok(pkind, mk)", "")]),
    ("file-plan-need-not-be-canonical", [(FI, "    if !canon {", "    if false {")]),
    # ---- run
    ("step-two-fitted-on-the-raw-matrix", [(R, "fit.fit(heap, plan.spec_of(pl, s), curr, contents(cr), s + 1)", "fit.fit(heap, plan.spec_of(pl, s), x, contents(cr), s + 1)")]),
    ("unknown-count-dropped-by-the-pipeline", [(R, "unknown = unknown + u;", "unknown = unknown + 0;")]),
    ("permutation-ignored", [(R, "let from = perm[c];", "let from = c;")]),
    # ---- registries
    ("step-kind-registered-twice", [(K, "pub fn step_label() -> [] int {\n    return 5;", "pub fn step_label() -> [] int {\n    return 4;")]),
    ("exit-code-of-the-parameter-tag-moved", [(C, "    if code == 303 {\n        return 2;", "    if code == 304 {\n        return 2;")]),
    ("tag-renamed", [(C, 'return "ml.plan.syntax";', 'return "ml.plan.syntax-error";')]),
]

# a survivor and the reason it cannot be killed by any input
ALLOWED = {
    "quantile-edge-clamp-removed": "the same clamp runs again over the whole edge row after the loop (the width edges need it): the first one is redundant by construction, so removing it changes no output",
}


def problems():
    out, names = [], set()
    cache = {}
    for name, edits in M:
        if name in names:
            out.append("mutant %s: named twice" % name)
        names.add(name)
        for file, old, new in edits:
            p = ROOT / file
            if not p.exists():
                out.append("mutant %s: no such file %s" % (name, file)); continue
            t = cache.setdefault(file, p.read_text())
            n = t.count(old)
            if n != 1:
                out.append("mutant %s: pattern occurs %d times in %s (must be exactly 1): %r" % (name, n, file, old[:70]))
            if old == new:
                out.append("mutant %s: replacement equals pattern" % name)
    return out


def tree(tmp):
    for d in ("lib", "tests", "tools", "scripts"):
        if (ROOT / d).exists():
            shutil.copytree(ROOT / d, pathlib.Path(tmp) / d)


def gate(tmp):
    env = dict(os.environ, CANCHO=CANCHO, TABLE_SRC=TABLE, PIPE_FAST="1")
    r = subprocess.run(["sh", "scripts/gate_pipe.sh"], cwd=tmp, capture_output=True, text=True, env=env, timeout=1800)
    return r


def main():
    bad = problems()
    if bad:
        print("\n".join(bad)); return 1
    if "--check" in sys.argv:
        print("mutants_pipe: %d mutants, every pattern applies once" % len(M)); return 0
    want = [a for a in sys.argv[1:] if not a.startswith("--")]
    with tempfile.TemporaryDirectory() as tmp:
        tree(tmp)
        r = gate(tmp)
        if r.returncode != 0:
            print("mutants_pipe: the unmutated control FAILS the gate; refusing to run mutants\n" + r.stdout[-1500:] + r.stderr[-800:]); return 1
        print("mutants_pipe: control (unmutated) passes", flush=True)
    killed, explained, survivors = 0, 0, []
    for name, edits in M:
        if want and not any(w in name for w in want):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tree(tmp)
            for file, old, new in edits:
                f = pathlib.Path(tmp) / file
                f.write_text(f.read_text().replace(old, new, 1))
            r = gate(tmp)
        if "BUILD_ERROR" in r.stdout:
            survivors.append(name); print("%-48s INVALID (does not compile)\n%s" % (name, r.stdout[-700:]), flush=True)
        elif r.returncode != 0:
            killed += 1; print("%-48s killed" % name, flush=True)
        elif name in ALLOWED:
            explained += 1; print("%-48s SURVIVED (explained: %s)" % (name, ALLOWED[name]), flush=True)
        else:
            survivors.append(name); print("%-48s SURVIVED" % name, flush=True)
    print("mutants_pipe: %d killed, %d survived and explained, %d unexplained" % (killed, explained, len(survivors)))
    if survivors:
        print("SURVIVORS: " + ", ".join(survivors)); return 1
    return 0


sys.exit(main())

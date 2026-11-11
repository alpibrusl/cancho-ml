#!/usr/bin/env python3
"""Mutants of lib/models/recommend/als.cho (docs/recommend.md gate RC13): a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change to the recommender ALS, the driver rebuilt from the copy, and the oracle's checks run against it
(`oracle_recommend.py --als-controls` and `--als 120`). KILLED: one of them failed. SURVIVED: both passed, and the mutant must carry a reason (`why`).

    python3 scripts/mutants_recommend_als.py              all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_recommend_als.py --only ID
    python3 scripts/mutants_recommend_als.py --list
    python3 scripts/mutants_recommend_als.py --check      every `old` text is found where it is expected (no build, no test)
CANCHO names the compiler."""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
CANCHO = os.environ.get("CANCHO", str(ROOT.parent / "cancho-compiler-0207556" / "target" / "release" / "cancho"))
PY = sys.executable
RK = "lib/models/recommend/als.cho"

# (id, what, old, new, count, why)
MUTANTS = [
    ("initial-factors-unscaled", "the initial item factors are not divided by sqrt(f)", "rng.unit(rng.output(st)) / sqrt(float_of(f));", "rng.unit(rng.output(st));", 1, None),
    ("initial-stream-kind", "the initial factors use stream kind 13", "    return 12;", "    return 13;", 1, None),
    ("initial-draw-not-advanced", "the first draw is the seed state, not its successor", "                                                st = rng.advance(st);", "                                                st = st;", 1, None),
    ("lambda-doubled", "the ridge term is added twice", "am[s * f + s] = am[s * f + s] + lambda;", "am[s * f + s] = am[s * f + s] + 2.0 * lambda;", 1, None),
    ("diagonal-of-the-sum-dropped", "the product of a factor with itself is left out of the system", "while q <= p {", "while q < p {", 1, None),
    ("right-hand-side-without-values", "the right-hand side ignores the ratings", "bv[p] = bv[p] + w * yp;", "bv[p] = bv[p] + yp;", 1, None),
    ("implicit-confidence-without-one", "the implicit right-hand side weighs c - 1 instead of c", "w = 1.0 + coef;", "w = coef;", 1, None),
    ("implicit-alpha-dropped", "the implicit confidence ignores alpha", "coef = alpha * vals[k];", "coef = vals[k];", 1, None),
    ("implicit-gram-not-used", "the implicit system starts from zero instead of Y^T Y", "am[s] = gm[s];", "am[s] = 0.0;", 1, None),
    ("implicit-term-rounded-differently", "the implicit term is (c y_p) y_q, not c (y_p y_q)", "coef * (yp * other[base + q])", "coef * yp * other[base + q]", 1, None),
    ("item-step-uses-the-item-gram", "the item half-pass takes the Gram matrix of its own factors", "                                                gram(pf, nu, f, g);", "                                                gram(qf, ni, f, g);", 1, None),
    ("backward-solve-forward", "the second triangular solve is the first", "code = chol.solve(1, common.mat(f, f), lm, common.vec(f), zv, common.vec(f), xv);", "code = chol.solve(0, common.mat(f, f), lm, common.vec(f), zv, common.vec(f), xv);", 1, None),
    ("one-pass-fewer", "the last pass is not made", "while pass < iterations && code == 0 {", "while pass < iterations - 1 && code == 0 {", 1, None),
    ("no-mid-pass-loss", "the loss after the user half-pass is not recorded", "if code == 0 && trace {", "if false {", 1, None),
    ("last-loss-after-the-first-pass", "without a trace the loss is that of the first pass", "(trace || pass == iterations - 1)", "(trace || pass == 0)", 1, None),
    ("loss-without-q-penalty", "the penalty leaves out the item factors", "return data + lambda * (sp + sq);", "return data + lambda * sp;", 1, None),
    ("implicit-loss-without-quadratic", "the implicit loss leaves out x^T G x", "            data = data + quad;", "            data = data + 0.0;", 1, None),
    ("implicit-loss-confidence-without-one", "the implicit loss confidence is alpha r", "let cc = 1.0 + alpha * uval[k];", "let cc = alpha * uval[k];", 1, None),
    ("factors-over-the-limit", "more than 256 factors are not refused", "f > max_factors()", "f > 100000", 1, None),
    ("iterations-over-the-limit", "more than 1000 iterations are not refused", "iterations > max_iterations()", "iterations > 100000", 1, None),
    ("infinite-lambda-accepted", "an infinite lambda is not refused", "return x - x == 0.0 && x > 0.0;", "return x > 0.0;", 1, None),
    ("alpha-read-by-the-explicit-form", "the explicit form refuses alpha = 0", "|| implicit && !is_finite_pos(alpha) {", "|| !is_finite_pos(alpha) {", 1, None),
    ("alpha-not-checked", "the implicit form accepts alpha = 0", "|| implicit && !is_finite_pos(alpha) {", " {", 1, None),
    ("pivot-code-swallowed", "a failed Cholesky is not reported", "        if code != 0 {\n            return code;\n        }", "        if code != 0 {\n            return 0;\n        }", 1, None),
]


def build(tree):
    lib = [str(p) for p in sorted((tree / "lib" / "core").glob("*.cho"))] + [str(p) for p in sorted((tree / "lib" / "linalg").glob("*.cho"))] + [str(p) for p in sorted((tree / "lib" / "metrics").glob("*.cho"))]
    lib += [str(tree / "lib/models/common.cho")] + [str(p) for p in sorted((tree / "lib" / "models" / "recommend").glob("*.cho"))]
    out = tree / "recommend-check"
    r = subprocess.run([CANCHO, "build", "--std", *lib, *[str(p) for p in sorted((tree / "tools" / "recommend-check").glob("*.cho"))], "-o", str(out)], capture_output=True, text=True, cwd=tree)
    return out if r.returncode == 0 else None, r.stdout + r.stderr


def judge(out):
    env = dict(os.environ, RECOMMEND_CHECK=str(out))
    for args in (["--als-controls"], ["--als", "120"]):
        r = subprocess.run([PY, str(ROOT / "scripts" / "oracle_recommend.py"), *args], capture_output=True, text=True, env=env, timeout=1800)
        if r.returncode != 0:
            return "KILLED", (r.stdout.strip().splitlines() or ["failed"])[0][:120]
    return "SURVIVED", ""


def make_tree(td):
    tree = pathlib.Path(td) / "tree"
    for d in ("lib/core", "lib/linalg", "lib/metrics", "lib/models/recommend", "tools/recommend-check"):
        shutil.copytree(ROOT / d, tree / d)
    shutil.copy(ROOT / "lib/models/common.cho", tree / "lib/models/common.cho")
    return tree


def apply(tree, m):
    path = tree / RK
    s = path.read_text()
    old, new, count = m[2], m[3], m[4]
    if s.count(old) < 1 or (count == 1 and s.count(old) != 1):
        return "`old` found %d times" % s.count(old)
    path.write_text(s.replace(old, new) if count == 0 else s.replace(old, new, count))
    return None


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        for m in MUTANTS:
            print("%-34s %s" % (m[0], m[1]))
        return 0
    if "--check" in argv:
        bad = 0
        with tempfile.TemporaryDirectory() as td:
            tree = make_tree(td)
            for m in MUTANTS:
                p = apply(tree, m)
                if p:
                    print("INVALID %s: %s" % (m[0], p))
                    bad += 1
                shutil.copy(ROOT / RK, tree / RK)
        print("%d recommend mutants, %d with an edit that does not apply" % (len(MUTANTS), bad))
        return 1 if bad else 0
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    killed, survivors, invalid = 0, [], []
    for m in MUTANTS:
        if only and m[0] != only:
            continue
        with tempfile.TemporaryDirectory() as td:
            tree = make_tree(td)
            p = apply(tree, m)
            if p:
                invalid.append(m[0])
                print("%-9s %-34s %s" % ("INVALID", m[0], p), flush=True)
                continue
            out, log = build(tree)
            if out is None:
                invalid.append(m[0])
                print("%-9s %-34s build failed: %s" % ("INVALID", m[0], log.strip()[:100]), flush=True)
                continue
            v, d = judge(out)
        print("%-9s %-34s %s" % (v, m[0], d), flush=True)
        if v == "KILLED":
            killed += 1
        else:
            survivors.append(m)
    if not only:
        with tempfile.TemporaryDirectory() as td:
            tree = make_tree(td)
            out, log = build(tree)
            v, d = judge(out) if out else ("INVALID", log[:100])
            print("%-9s %-34s %s" % ("baseline", "(the unchanged tree)", "passes" if v == "SURVIVED" else "FAILS: " + d), flush=True)
            if v != "SURVIVED":
                return 1
    print("\n%d mutants: %d killed, %d survived, %d invalid" % (killed + len(survivors) + len(invalid), killed, len(survivors), len(invalid)))
    for m in survivors:
        print("SURVIVOR %s: %s" % (m[0], m[5] or "NO REASON GIVEN"))
    return 1 if invalid or any(not m[5] for m in survivors) else 0


if __name__ == "__main__":
    sys.exit(main())

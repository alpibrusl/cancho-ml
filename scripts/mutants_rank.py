#!/usr/bin/env python3
"""Mutants of lib/metrics/ranking.cho (docs/ranking.md gate RK11): a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change to the ranking metrics, the driver rebuilt from the copy, and the oracle's two checks run against it
(`oracle_rank.py --metrics 150` and `--driver`). KILLED: one of them failed. SURVIVED: both passed, and the mutant must carry a reason (`why`).

    python3 scripts/mutants_rank.py              all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_rank.py --only ID
    python3 scripts/mutants_rank.py --list
    python3 scripts/mutants_rank.py --check      every `old` text is found where it is expected (no build, no test)
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
RK = "lib/metrics/ranking.cho"

# (id, what, old, new, count) -- count 0 replaces every occurrence, n the first n
MUTANTS = [
    ("discount-of-r", "the discount is 1 / log2(r) for 1 / log2(r + 1)", "math.log(float_of(r + 1))", "math.log(float_of(r))", 1, None),
    ("gain-without-minus-one", "the gain is 2^rel", "float_of((1 << rel) - 1)", "float_of(1 << rel)", 1, None),
    ("ties-by-the-later-row", "equal scores keep the later row first", "keys[idx[a]] >= keys[idx[b]]", "keys[idx[a]] > keys[idx[b]]", 0, None),
    ("ndcg-divides-by-dcg", "the NDCG is DCG over the DCG of the predicted order", "sum_ndcg + dcg / idcg", "sum_ndcg + dcg / dcg", 1, None),
    ("ap-denominator-is-k", "AP divides by min(k, R) without the min", "                                        if kk < nrel {\n                                            apd = kk;", "                                        if false {\n                                            apd = kk;", 1, None),
    ("ndcg-mean-over-all-groups", "an empty group counts as 0 in the NDCG mean", "ndcg: sum_ndcg / float_of(scored)", "ndcg: sum_ndcg / float_of(ngroups)", 1, None),
    ("metrics-past-k", "the sums run over the whole group, not the first k", "                                        if r2 < kk {", "                                        if r2 < g {", 1, None),
    ("precision-divides-by-the-group", "precision@k divides by the group size", "                                        pden = k;", "                                        pden = g;", 1, None),
    ("recall-divides-by-k", "recall divides by the cut-off", "sum_rec + float_of(hits) / float_of(nrel)", "sum_rec + float_of(hits) / float_of(kk)", 1, None),
    ("mrr-of-the-last-relevant", "the reciprocal rank is that of the last relevant row", "                                                if first == 0 {", "                                                if true {", 1, None),
    ("map-averages-over-ranks", "AP sums the precision at every rank, not at the relevant ones", "                                            if rl > 0 {\n                                                hits = hits + 1;", "                                            if true {\n                                                hits = hits + 1;", 1, None),
    ("non-finite-score-accepted", "a non-finite score is not refused", "if regress.all_finite(score) == false {", "if false {", 1, None),
    ("relevance-above-30-accepted", "a relevance above 30 is not refused", " || rel[i] > max_relevance() {", " {", 1, None),
    ("negative-relevance-accepted", "a negative relevance is not refused", "|| rel[i] < 0 || rel[i] > max_relevance()", "|| rel[i] > max_relevance()", 1, None),
    ("group-over-the-cap-accepted", "a group of more than 4,096 rows is not refused", "if st[gi + 1] > max_group() {", "if st[gi + 1] > 100000 {", 1, None),
    ("k-over-the-cap-accepted", "k above 4,096 is not refused", "k > max_group() ||", "k > 100000 ||", 1, None),
    ("length-mismatch-accepted", "arrays of different lengths are not refused", "if len(rel) != n || len(score) != n {", "if len(rel) != n && len(score) != n {", 1, None),
    ("row-cap-ignored", "the row cap is not enforced", "if n > cap || n > classify.ceiling_rows() {", "if n > classify.ceiling_rows() {", 1, None),
    ("empty-group-accepted", "a group number with no row is accepted", "                        if st[gi + 1] == 0 {\n                            empty_group = true;", "                        if false {\n                            empty_group = true;", 1, None),
]


def build(tree):
    lib = [str(p) for p in sorted((tree / "lib" / "core").glob("*.cho"))] + [str(tree / "lib/linalg/tensor.cho"), str(tree / "lib/linalg/order.cho")] + [str(p) for p in sorted((tree / "lib" / "metrics").glob("*.cho"))]
    out = tree / "metrics-check"
    r = subprocess.run([CANCHO, "build", "--std", *lib, *[str(p) for p in sorted((tree / "tools" / "metrics-check").glob("*.cho"))], "-o", str(out)], capture_output=True, text=True, cwd=tree)
    return out if r.returncode == 0 else None, r.stdout + r.stderr


def judge(out):
    env = dict(os.environ, METRICS_CHECK=str(out))
    for args in (["--driver"], ["--metrics", "150"]):
        r = subprocess.run([PY, str(ROOT / "scripts" / "oracle_rank.py"), *args], capture_output=True, text=True, env=env, timeout=1800)
        if r.returncode != 0:
            return "KILLED", (r.stdout.strip().splitlines() or ["failed"])[0][:120]
    return "SURVIVED", ""


def make_tree(td):
    tree = pathlib.Path(td) / "tree"
    for d in ("lib/core", "lib/linalg", "lib/metrics", "tools/metrics-check"):
        shutil.copytree(ROOT / d, tree / d)
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
        print("%d rank mutants, %d with an edit that does not apply" % (len(MUTANTS), bad))
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

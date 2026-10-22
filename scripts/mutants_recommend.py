#!/usr/bin/env python3
"""Mutants of lib/metrics/recommend.cho (docs/recommend.md gate RC13): a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change to the recommendation metrics, the driver rebuilt from the copy, and the oracle's two checks run against it
(`oracle_recommend.py --metrics 150` and `--driver`). KILLED: one of them failed. SURVIVED: both passed, and the mutant must carry a reason (`why`).

    python3 scripts/mutants_recommend.py              all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_recommend.py --only ID
    python3 scripts/mutants_recommend.py --list
    python3 scripts/mutants_recommend.py --check      every `old` text is found where it is expected (no build, no test)
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
RK = "lib/metrics/recommend.cho"
SD = "lib/metrics/ranking.cho"

# (id, what, old, new, count, why, file) -- count 0 replaces every occurrence, n the first n; the file defaults to recommend.cho
MUTANTS = [
    ("discount-of-the-next-rank", "the discount table is shifted one rank", "ds[q] = ranking.discount(q, ln2);", "ds[q] = ranking.discount(q + 1, ln2);", 1, None),
    ("ranking-ascending", "the items are ranked from the lowest score", "ks[it] = rank.key_of(scores[u * nitems + it]);", "ks[it] = 0 - rank.key_of(scores[u * nitems + it]);", 1, None),
    ("ranking-not-sorted", "the items are left in item order", "    ranking.sort_desc(keys, idx, tmp, m);", "    m = m + 0;", 1, None),
    ("ties-by-the-later-item", "equal scores keep the later item first", "keys[idx[a]] >= keys[idx[b]]", "keys[idx[a]] > keys[idx[b]]", 0, None, SD),
    ("history-not-excluded", "the items of the history stay in the ranking", "        if p < len(hist) && hist[p] == it {", "        if false {", 1, None),
    ("mrr-cut-at-k", "the reciprocal rank is judged within the first k", "            if first == 0 {\n                first = r + 1;", "            if first == 0 && r < kk {\n                first = r + 1;", 1, None),
    ("ndcg-divides-by-dcg", "the NDCG is the DCG over itself", "ndcg: dcg / idcg", "ndcg: dcg / dcg", 1, None),
    ("ideal-dcg-not-cut-at-k", "the ideal DCG runs over all test items, not min(k, |T|)", "    if k < ideal {\n        ideal = k;\n    }", "    if false {\n        ideal = k;\n    }", 1, None),
    ("recall-divides-by-k", "recall divides by the cut-off", "recall: float_of(found) / float_of(len(test))", "recall: float_of(found) / float_of(kk)", 1, None),
    ("hit-is-the-count", "the hit is the number found", "        hit = 1.0;", "        hit = float_of(found);", 1, None),
    ("metrics-past-k", "the sums run over the whole ranking", "            if r < kk {\n                dcg", "            if r < m {\n                dcg", 1, None),
    ("marks-not-cleared", "the test marks of a user stay for the next", "        mark[test[j]] = 0;", "        mark[test[j]] = 1;", 1, None),
    ("user-without-test-scored", "a user without a test item is scored", "                            if toff[u + 1] > toff[u] {", "                            if true {", 1, None),
    ("mean-over-all-users", "the means divide by the users, not the users scored", "ndcg: sum_ndcg / n", "ndcg: sum_ndcg / float_of(nusers)", 1, None),
    ("non-finite-score-accepted", "a non-finite score is not refused", "if regress.all_finite(scores) == false {", "if false {", 1, None),
    ("k-zero-accepted", "k = 0 is not refused", "if k < 1 || k > nitems {", "if k > nitems {", 1, None),
    ("k-above-items-accepted", "k above the number of items is not refused", "if k < 1 || k > nitems {", "if k < 1 {", 1, None),
    ("cap-ignored", "the cap on users times items is not enforced", " || nusers * nitems > cap {", " {", 1, None),
    ("score-length-accepted", "a score array of the wrong length is not refused", "if len(scores) != nusers * nitems {", "if false {", 1, None),
    ("test-in-history-accepted", "a test item that is in the history is not refused", "if disjoint(hoff[u], hoff[u + 1], hist, toff[u], toff[u + 1], test) == false {", "if false {", 1, None),
    ("list-not-ascending-accepted", "a list that is not strictly ascending is accepted", " || j > off[u] && list[j] <= list[j - 1] {", " {", 1, None),
    ("offset-past-the-list-accepted", "an offset beyond the list is not refused", " || off[u + 1] > len(list) {", " {", 1, None),
    ("decreasing-offset-accepted", "a decreasing offset is not refused", "if off[u + 1] < off[u] || off[u + 1] > len(list) {", "if off[u + 1] > len(list) {", 1, None),
    ("offset-end-accepted", "offsets that do not end at the list are accepted", " || off[nusers] != len(list) {", " {", 1, None),
    ("item-out-of-range-accepted", "an item number out of range is not refused", "if list[j] < 0 || list[j] >= nitems ||", "if list[j] < 0 ||", 1, None),
]


def build(tree):
    lib = [str(p) for p in sorted((tree / "lib" / "core").glob("*.cho"))] + [str(tree / "lib/linalg/tensor.cho"), str(tree / "lib/linalg/order.cho")] + [str(p) for p in sorted((tree / "lib" / "metrics").glob("*.cho"))]
    out = tree / "metrics-check"
    r = subprocess.run([CANCHO, "build", "--std", *lib, *[str(p) for p in sorted((tree / "tools" / "metrics-check").glob("*.cho"))], "-o", str(out)], capture_output=True, text=True, cwd=tree)
    return out if r.returncode == 0 else None, r.stdout + r.stderr


def judge(out):
    env = dict(os.environ, METRICS_CHECK=str(out))
    for args in (["--controls"], ["--driver"], ["--metrics", "150"]):
        r = subprocess.run([PY, str(ROOT / "scripts" / "oracle_recommend.py"), *args], capture_output=True, text=True, env=env, timeout=1800)
        if r.returncode != 0:
            return "KILLED", (r.stdout.strip().splitlines() or ["failed"])[0][:120]
    return "SURVIVED", ""


def make_tree(td):
    tree = pathlib.Path(td) / "tree"
    for d in ("lib/core", "lib/linalg", "lib/metrics", "tools/metrics-check"):
        shutil.copytree(ROOT / d, tree / d)
    return tree


def apply(tree, m):
    path = tree / (m[6] if len(m) > 6 else RK)
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
                shutil.copy(ROOT / SD, tree / SD)
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

#!/usr/bin/env python3
"""Mutants of lib/models/recommend/log.cho (docs/recommend.md gate RC13): a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change to the recommender log, the driver rebuilt from the copy, and the oracle's checks run against it
(`oracle_recommend.py --log-controls` and `--log 150`). KILLED: one of them failed. SURVIVED: both passed, and the mutant must carry a reason (`why`).

    python3 scripts/mutants_recommend_log.py              all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_recommend_log.py --only ID
    python3 scripts/mutants_recommend_log.py --list
    python3 scripts/mutants_recommend_log.py --check      every `old` text is found where it is expected (no build, no test)
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
RK = "lib/models/recommend/log.cho"

# (id, what, old, new, count, why) -- count 0 replaces every occurrence, n the first n
MUTANTS = [
    ("duplicate-implicit-keeps-the-last", "a repeated implicit pair keeps its last value", "            vl[k] = vl[k] + vals[o[r]];", "            vl[k] = vals[o[r]];", 1, None),
    ("duplicate-explicit-accepted", "a repeated explicit pair is not refused", "if dup_a >= 0 && implicit == false {", "if false {", 1, None),
    ("duplicate-rows-swapped", "the two rows of a duplicate are reported the other way round", "                    dup_a = o[r - 1];\n                    dup_b = o[r];", "                    dup_a = o[r];\n                    dup_b = o[r - 1];", 1, None),
    ("duplicate-of-the-last-pair", "the last repeated pair is reported, not the first", "                if dup_a < 0 {", "                if true {", 1, None),
    ("implicit-zero-accepted", "an implicit value of zero or less is not refused", " || implicit && !(x > 0.0) {", " {", 1, None),
    ("rating-bound-one-sided", "a rating below -1e6 is not refused", "x > max_value() || x < 0.0 - max_value()", "x > max_value()", 1, None),
    ("non-finite-rating-accepted", "a NaN or infinite rating is not refused", "if !(x - x == 0.0) {", "if false {", 1, None),
    ("user-range-not-checked", "a user number out of range is accepted", "if uid[r] < 0 || uid[r] >= nusers || iid[r] < 0 || iid[r] >= nitems {", "if iid[r] < 0 || iid[r] >= nitems {", 1, None),
    ("item-range-not-checked", "an item number out of range is accepted", "if uid[r] < 0 || uid[r] >= nusers || iid[r] < 0 || iid[r] >= nitems {", "if uid[r] < 0 || uid[r] >= nusers {", 1, None),
    ("empty-id-accepted", "a user or an item with no row is accepted", "    if empty_id {", "    if false {", 1, None),
    ("users-over-the-limit", "more users than the limit are not refused", "    if nusers > max_users() {", "    if false {", 1, None),
    ("items-over-the-limit", "more items than the limit are not refused", "    if nitems > max_items() {", "    if false {", 1, None),
    ("rows-over-the-cap", "more rows than the cap are not refused", "    if n > max_interactions() || n > cap {", "    if n > max_interactions() {", 1, None),
    ("rows-over-the-limit", "more rows than the limit are not refused", "    if n > max_interactions() || n > cap {", "    if n > cap {", 1, "the tool passes `cap` (its own bound, never above the limit), so the library's constant is a second line of defence for a caller that passes more; reaching it needs more than 2^25 rows in an argument vector, which no control here builds"),
    ("ids-over-the-rows", "more ids than rows are not refused", " || nusers > n || nitems > n {", " {", 1, "with more ids than rows some id has no row, and the check for an id with no row refuses with the same code and reason; the early check only avoids allocating the work arrays for a log that is going to be refused (the allocation is bounded by the limits of the line above)"),
    ("items-not-ascending-in-a-user", "the first sort is by user, so the items of a user are not ascending", "sort_by_key(iid, a, contents(o1), contents(cw), n, nitems);", "sort_by_key(uid, a, contents(o1), contents(cw), n, nusers);", 1, None),
    ("users-of-an-item-wrong", "the by-item form lists the same user for every entry", "contents(iu)[cu[it[k]]] = q;", "contents(iu)[cu[it[k]]] = 0;", 1, None),
    ("values-of-an-item-wrong", "the by-item form carries a constant value", "contents(ivw)[cu[it[k]]] = vl[k];", "contents(ivw)[cu[it[k]]] = 1.0;", 1, None),
    ("split-ties-cut-the-other-way", "equal times hold out the earlier row", "                            rv[q] = n - 1 - q;", "                            rv[q] = q;", 1, None),
    ("split-user-with-exactly-last", "a user with exactly N rows is held out whole", "                            if end - pos > last {", "                            if end - pos >= last {", 1, None),
    ("split-one-too-many", "N + 1 rows of a user are held out", "while t < pos + last {", "while t < pos + last + 1 && t < end {", 1, None),
    ("split-user-range-not-checked", "a user number out of range is accepted by the split", "        if uid[r] < 0 || uid[r] >= nusers {\n            return (box_slice(heap, 1, 0), refuse.param_out_of_range());", "        if false {\n            return (box_slice(heap, 1, 0), refuse.param_out_of_range());", 1, None),
    ("split-last-zero-accepted", "N = 0 is not refused", "|| last < 1 ||", "||", 1, None),
]


def build(tree):
    lib = [str(p) for p in sorted((tree / "lib" / "core").glob("*.cho"))] + [str(p) for p in sorted((tree / "lib" / "linalg").glob("*.cho"))] + [str(p) for p in sorted((tree / "lib" / "metrics").glob("*.cho"))]
    lib += [str(tree / "lib/models/common.cho")] + [str(p) for p in sorted((tree / "lib" / "models" / "recommend").glob("*.cho"))]
    out = tree / "recommend-check"
    r = subprocess.run([CANCHO, "build", "--std", *lib, *[str(p) for p in sorted((tree / "tools" / "recommend-check").glob("*.cho"))], "-o", str(out)], capture_output=True, text=True, cwd=tree)
    return out if r.returncode == 0 else None, r.stdout + r.stderr


def judge(out):
    env = dict(os.environ, RECOMMEND_CHECK=str(out))
    for args in (["--log-controls"], ["--log", "150"]):
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

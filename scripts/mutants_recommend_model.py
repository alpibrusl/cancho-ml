#!/usr/bin/env python3
"""Mutants of lib/models/recommend/model.cho (docs/recommend.md gate RC13): a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change to the recommender model payload, the driver rebuilt from the copy, and the oracle's checks run against it
(`oracle_recommend.py --payload 6`). KILLED: one of them failed. SURVIVED: both passed, and the mutant must carry a reason (`why`).

    python3 scripts/mutants_recommend_model.py              all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_recommend_model.py --only ID
    python3 scripts/mutants_recommend_model.py --list
    python3 scripts/mutants_recommend_model.py --check      every `old` text is found where it is expected (no build, no test)
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
RK = "lib/models/recommend/model.cho"

# (id, what, old, new, count, why)
MUTANTS = [
    ("kind-not-checked", "a payload of another kind is read", "    if fbits.get64(src, 0) != kinds.model_recommend() {", "    if false {", 1, None),
    ("method-2-accepted", "a method above 1 is read", "method < 0 || method > 1 ||", "method < 0 ||", 1, None),
    ("factors-zero-accepted", "zero factors are read", "f < 1 || f > als.max_factors()", "f > als.max_factors()", 1, None),
    ("factors-over-the-limit", "more factors than the limit are read", "f < 1 || f > als.max_factors()", "f < 1", 1, None),
    ("iterations-zero-accepted", "zero iterations are read", "|| iterations < 1 ||", "||", 1, None),
    ("iterations-over-the-limit", "more iterations than the limit are read", "|| iterations > als.max_iterations() {", " {", 1, None),
    ("users-zero-accepted", "zero users are read", "    if nu < 1 || nu > log.max_users()", "    if nu > log.max_users()", 1, None),
    ("nnz-over-the-limit", "more interactions than the limit are read", " || nnz > log.max_interactions() ||", " ||", 1, None),
    ("more-users-than-interactions", "more users than interactions are read", " || nu > nnz || ni > nnz {", " {", 1, "with more users than interactions some user has an empty history range, and the check of the history refuses it with the same code; the early check only avoids allocating arrays sized by the counts of a payload that is going to be refused (they are bounded by the limits of the line above)"),
    ("lambda-not-positive", "a lambda of 0 or less is read", "if !(lambda - lambda == 0.0 && lambda > 0.0) ||", "if !(lambda - lambda == 0.0) ||", 1, None),
    ("lambda-encoding-not-canonical", "a lambda whose bits do not survive decoding is read", " || bits_of(lambda) != fbits.get64(src, 56) || bits_of(alpha) != fbits.get64(src, 64) {", " {", 1,
     "fbits.decode answers NaN for an infinity and a NaN, which the finiteness test refuses; every other pattern tried (the subnormal and normal extremes and 16 random ones per payload, the named cases `lambda bits ...` of oracle_recommend.py) is read back as written, so on those the comparison of the bits refuses nothing the finiteness test does not. This is evidence on the patterns tried, not a proof"),
    ("explicit-alpha-not-zero", "an explicit payload with an alpha is read", " || method == 0 && bits_of(alpha) != 0 {", " {", 1, None),
    ("implicit-alpha-not-positive", "an implicit payload with an alpha of 0 is read", "if method == 1 && !(alpha - alpha == 0.0 && alpha > 0.0) ||", "if", 1, None),
    ("user-bytes-too-many", "a byte count above 255 an id is read", "if ub < nu || ub > nu * max_id_bytes() {", "if ub < nu {", 1, None),
    ("item-bytes-too-many", "an item byte count above 255 an id is read", "if ib < ni || ib > ni * max_id_bytes() {", "if ib < ni {", 1, None),
    ("extra-bytes-accepted", "a payload longer than its fields say is read", "    if len(src) > size {", "    if false {", 1, None),
    ("short-payload-accepted", "a payload shorter than its fields say is read", "    if len(src) < size {\n        return bad(heap, 2);\n    }\n    if len(src) > size", "    if false {\n        return bad(heap, 2);\n    }\n    if len(src) > size", 1, None),
    ("ids-not-ascending", "ids that are not ascending are read", "            if order > 0 {", "            if order > 1 {", 1, None),
    ("empty-id-accepted", "an empty id is read", "        if width < 1 || width > max_id_bytes() || offs[i + 1] > total {", "        if width > max_id_bytes() || offs[i + 1] > total {", 1, None),
    ("long-id-accepted", "an id of more than 255 bytes is read", "        if width < 1 || width > max_id_bytes() || offs[i + 1] > total {", "        if width < 1 || offs[i + 1] > total {", 1, None),
    ("offset-past-the-bytes", "an id offset past the table is not refused before it is used", "        if width < 1 || width > max_id_bytes() || offs[i + 1] > total {", "        if width < 1 || width > max_id_bytes() {", 1, None),
    ("table-end-not-checked", "an offset table that does not end at its byte count is read", "    if offs[0] != 0 || offs[n] != total {", "    if offs[0] != 0 {", 1, None),
    ("history-end-not-checked", "history offsets that do not end at nnz are read", "contents(how)[0] != 0 || contents(how)[nu] != nnz", "contents(how)[0] != 0", 1, None),
    ("history-start-not-checked", "history offsets that do not start at 0 are read", "contents(how)[0] != 0 || contents(how)[nu] != nnz", "contents(how)[nu] != nnz", 1, None),
    ("user-without-history-accepted", "a user with no history is read", "if hi <= lo || hi > nnz {", "if hi > nnz {", 1, None),
    ("item-number-out-of-range-accepted", "an item number out of range is read", "if it < 0 || it >= ni || j > lo && it <= contents(hiw)[j - 1] {", "if j > lo && it <= contents(hiw)[j - 1] {", 1, None),
    ("history-not-ascending-accepted", "items of a user that are not ascending are read", "if it < 0 || it >= ni || j > lo && it <= contents(hiw)[j - 1] {", "if it < 0 || it >= ni {", 1, None),
    ("value-bound-one-sided", "a value below -1e6 is read", "v > log.max_value() || v < 0.0 - log.max_value() ||", "v > log.max_value() ||", 1, None),
    ("implicit-zero-value-accepted", "an implicit value of 0 or less is read", "|| method == 1 && !(v > 0.0) {", " {", 1, None),
    ("unseen-item-accepted", "an item no history holds is read", "                                                if contents(sw)[it2] != 1 {", "                                                if false {", 1, None),
    ("factors-not-checked", "a non-finite factor is read", "                                        if fbits.load(src, at, contents(pw), nu * f) != 0 {", "                                        if false {", 1, None),
]


def build(tree):
    lib = [str(p) for p in sorted((tree / "lib" / "core").glob("*.cho"))] + [str(p) for p in sorted((tree / "lib" / "linalg").glob("*.cho"))] + [str(p) for p in sorted((tree / "lib" / "metrics").glob("*.cho"))]
    lib += [str(tree / "lib/models/common.cho")] + [str(p) for p in sorted((tree / "lib" / "models" / "recommend").glob("*.cho"))]
    out = tree / "recommend-check"
    r = subprocess.run([CANCHO, "build", "--std", *lib, *[str(p) for p in sorted((tree / "tools" / "recommend-check").glob("*.cho"))], "-o", str(out)], capture_output=True, text=True, cwd=tree)
    return out if r.returncode == 0 else None, r.stdout + r.stderr


def judge(out):
    env = dict(os.environ, RECOMMEND_CHECK=str(out))
    for args in (["--payload", "6"],):
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

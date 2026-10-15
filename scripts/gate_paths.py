#!/usr/bin/env python3
"""G-paths -- no path and no missing operand reaches a trap in a check tool (docs/lanes.md section 1).

The five drivers under tools/ (models-check, tree-check, reader-check, linalg-check, metrics-check) are built from the sources,
then (1) the fixtures of tests/paths/fixtures.json run: a path with two dots in a row anywhere (`../x`, `a..b`, `.../x`, `..`,
`a/../b`) must be refused with the tag path.dotdot and exit 2 and no output but that line, a good path must still work, a missing
operand must end in a status the fixture names and never in a trap; (2) a seeded fuzz of argument vectors per tool (mutations of
a valid vector: every prefix, every truncated argument, random replacements, insertions, deletions, path-like fragments) must
never end in a trap (132, 134, 136, 139, a negative status) or a hang, and a vector the model of the tool's own argument parser says
carries a dotdot path must end in exactly the refusal. Exit 1 on any failure.

    python3 scripts/gate_paths.py            # FUZZ_SEED (default 20261013), FUZZ_N (vectors per tool, default 400)
    python3 scripts/gate_paths.py --tool models-check

scripts/mutants_paths.py removes each check in turn and requires this script to fail. Environment: CANCHO, ML_DEPS (the installed
packages, default build/deps), as in scripts/build_reader.sh."""
import json
import os
import random
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANCHO = os.environ.get("CANCHO", os.path.join(os.path.dirname(ROOT), "cancho-compiler-0207556", "target", "release", "cancho"))
TRAPS = {132, 134, 136, 139}
TAG = "path.dotdot"
SEED = int(os.environ.get("FUZZ_SEED", "20261013"))
N = int(os.environ.get("FUZZ_N", "400"))


def files(*pats):
    out = []
    for p in pats:
        if p.endswith("/*.cho"):
            d = os.path.join(ROOT, p[:-6])
            out += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".cho")]
        else:
            out.append(os.path.join(ROOT, p))
    return out


def build(tmp):
    exes = {}
    lists = {
        "linalg-check": files("lib/core/*.cho", "lib/linalg/*.cho", "tools/linalg-check/main.cho"),
        "models-check": files("lib/core/*.cho", "lib/linalg/*.cho", "lib/models/*.cho", "lib/models/linear/*.cho",
                              "lib/models/kmeans/*.cho", "tools/models-check/main.cho"),
        "tree-check": files("lib/core/*.cho", "lib/linalg/tensor.cho", "lib/models/tree/*.cho", "tools/tree-check/main.cho"),
        "metrics-check": files("lib/core/*.cho", "lib/linalg/tensor.cho", "lib/linalg/order.cho", "lib/metrics/*.cho",
                               "tools/metrics-check/main.cho", "tools/metrics-check/gen.cho"),
    }
    for name, fl in lists.items():
        exe = os.path.join(tmp, name)
        r = subprocess.run([CANCHO, "build", "--std", *fl, "-o", exe], capture_output=True, text=True)
        if r.returncode:
            print("BUILD_ERROR %s\n%s%s" % (name, r.stdout[-1500:], r.stderr[-1500:]))
            sys.exit(3)
        exes[name] = exe
    exe = os.path.join(tmp, "reader-check")
    env = dict(os.environ, CANCHO=CANCHO)
    r = subprocess.run(["sh", os.path.join(ROOT, "scripts", "build_reader.sh"), exe], capture_output=True, text=True, env=env, cwd=ROOT)
    if r.returncode:
        print("BUILD_ERROR reader-check\n%s%s" % (r.stdout[-1500:], r.stderr[-1500:]))
        sys.exit(3)
    exes["reader-check"] = exe
    return exes


def run(exe, argv, cwd):
    try:
        r = subprocess.run([exe] + argv, capture_output=True, cwd=cwd, timeout=60)
    except subprocess.TimeoutExpired:
        return None, b"", b""
    return r.returncode, r.stdout, r.stderr


def trap(status):
    return status is None or status < 0 or status in TRAPS


class Gate:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def fail(self, msg):
        self.failures.append(msg)
        if len(self.failures) <= 30:
            print("FAIL " + msg)


def scratch(tmp):
    d = os.path.join(tmp, "work")
    os.makedirs(os.path.join(d, "sub"))
    xs = [((i * 7 + 3) % 11) / 4.0 for i in range(20)]
    ys = [float(i % 2) + 0.5 * (i % 3) for i in range(10)]
    px, py = os.path.join(d, "x.bin"), os.path.join(d, "y.bin")
    open(px, "wb").write(struct.pack("<20d", *xs))
    open(py, "wb").write(struct.pack("<10d", *ys))
    open(os.path.join(d, "m.bin"), "wb").write(bytes(2 * 2))
    open(os.path.join(d, "a.csv"), "w").write("a,b\n1,2\n3,4\n5,6\n")
    return d, {"{x}": px, "{y}": py, "{csv}": os.path.join(d, "a.csv"), "{miss}": os.path.join(d, "m.bin")}


def subst(argv, sub):
    out = []
    for a in argv:
        for k, v in sub.items():
            a = a.replace(k, v)
        out.append(a)
    return out


# ---- the models of each tool's own argument parsing: which operands are paths, as the tool reads them ----

def models_paths(argv):
    # argv here is the vector WITHOUT the program name: MODEL X Y ...; the tool needs arg_count >= 4, the program name included
    if len(argv) < 3:
        return []
    if argv[1] == "-":
        return []
    out = [argv[1]]
    if argv[0] != "kmeans":
        out.append(argv[2])
    return out


def tree_paths(argv):
    out = []
    for key in ("x", "y", "miss"):
        for a in argv:
            if a.startswith(key + "="):
                out.append(a[len(key) + 1:])
                break
    return out


def reader_paths(argv):
    # the tool's loop: known flags take a value when one follows; any other "--" argument is an error; the last other argument is the file
    value_flags = {"--mode", "--max-rows", "--max-columns", "--max-cells", "--max-line-bytes", "--drop", "--float"}
    i, bad, f = 0, False, ""
    while i < len(argv):
        a = argv[i]
        if a in value_flags and i + 1 < len(argv):
            if a == "--mode" and argv[i + 1] not in ("shape", "dump", "hash"):
                bad = True
            i += 2
        elif len(a) > 1 and a[0] == "-" and a[1] == "-":
            bad = True
            i += 1
        else:
            f = a
            i += 1
    return [] if bad else [f]


def no_paths(argv):
    return []


PATHS = {"models-check": models_paths, "tree-check": tree_paths, "reader-check": reader_paths,
         "linalg-check": no_paths, "metrics-check": no_paths}

VALID = {
    "models-check": [["linreg", "{x}", "{y}", "n=10", "d=2"], ["kmeans", "{x}", "{y}", "n=10", "d=2", "k=2", "iters=2", "threads=2"],
                     ["logreg", "{x}", "{y}", "n=10", "d=2", "ntr=8", "l2m=1"], ["linreg", "-", "-", "n=10", "d=2", "seed=3"]],
    "tree-check": [["mode=fit", "x={x}", "y={y}", "n=10", "d=2", "ntr=8", "task=0", "trees=2", "depth=2", "out=hash"],
                   ["mode=fuzz", "x={x}", "y={y}", "miss={miss}", "n=10", "d=2", "ntr=8"], ["mode=gt"]],
    "reader-check": [["--mode", "hash", "--max-rows", "100", "{csv}"], ["--drop", "a", "--float", "b", "{csv}"], ["{csv}"]],
    "linalg-check": [["dot", "100", "1", "3"], ["gemv_n", "5", "4", "1"], ["chol", "4", "1"], ["gemm", "4", "1", "1", "1", "2"]],
    "metrics-check": [["ratio", "1", "2"], ["testcount", "10", "2", "1"], ["bounded", "1", "10", "3"],
                      ["split", "strat", "0,1,0,1", "2", "1", "0", "4"]],
}

FRAGMENTS = ["..", "../x", "a..b", ".../x", "x..", "..x", "a/../b", "./x", "/", "/dev/null", "/nonexistent", "", "-", "--", "-x", "=",
             "x=..", "y=a..b", "miss=../m", "mode=", "mode=fit", "mode=fuzz", "mode=gt", "out=", "out=hash", "n=", "n=0", "n=-1",
             "n=3", "d=1", "d=0", "k=0", "k=1", "threads=0", "threads=3", "1", "0", "-1", "99999999999999999999", "linreg", "logreg",
             "kmeans", "nomodel", "x.bin", "y.bin", "a.csv", "nofile.bin", "sub", "sub/x", "a" * 600, "\xe9", "\x01", "--mode", "--drop",
             "--float", "--max-rows", "--max-cells", "--mode=hash", "hash", "shape", "dump", "dot", "ratio", "split", "strat", "0,1,1"]


def mutate(rnd, base, exe_name, sub):
    v = list(base)
    for _ in range(rnd.choice([1, 1, 2, 3])):
        op = rnd.randrange(7)
        if op == 0 and v:
            del v[rnd.randrange(len(v))]
        elif op == 1 and v:
            i = rnd.randrange(len(v))
            v[i] = v[i][:rnd.randrange(len(v[i]) + 1)]
        elif op == 2 and v:
            v[rnd.randrange(len(v))] = rnd.choice(FRAGMENTS)
        elif op == 3:
            v.insert(rnd.randint(0, len(v)), rnd.choice(FRAGMENTS))
        elif op == 4 and v:
            i = rnd.randrange(len(v))
            if "=" in v[i]:
                k = v[i].index("=")
                v[i] = v[i][:k + 1] + rnd.choice(FRAGMENTS)
        elif op == 5 and v:
            i = rnd.randrange(len(v))
            j = rnd.randint(0, len(v[i]))
            v[i] = v[i][:j] + rnd.choice(["..", ".", "/..", "../"]) + v[i][j:]
        elif op == 6:
            v = v[:rnd.randint(0, len(v))]
    return v


def expected_refusal(name, argv):
    return any(".." in p for p in PATHS[name](argv))


def check_run(g, name, exe, cwd, argv, what):
    status, out, err = run(exe, argv, cwd)
    g.checks += 1
    shown = repr(argv)[:300]
    if trap(status):
        g.fail("%s %s: TRAP/hang status %s on %s" % (name, what, status, shown))
        return status, out
    want = expected_refusal(name, argv)
    line = ("refused " + TAG + "\n").encode()
    if want and (status != 2 or out != line):
        g.fail("%s %s: a dotdot path was not refused with %s and exit 2 (status %s, stdout %r) on %s" % (name, what, TAG, status, out[:120], shown))
    if not want and out.startswith(b"refused " + TAG.encode()):
        g.fail("%s %s: refused %s with no dotdot path, on %s" % (name, what, TAG, shown))
    return status, out


def fixtures(g, exes, cwd, sub, only):
    fx = json.load(open(os.path.join(ROOT, "tests", "paths", "fixtures.json")))["tools"]
    for name, groups in fx.items():
        if only and name != only:
            continue
        exe = exes[name]
        for group, cases in groups.items():
            for case in cases:
                argv = subst(case["argv"], sub)
                status, out, err = run(exe, argv, cwd)
                g.checks += 1
                if trap(status):
                    g.fail("fixture %s/%s %r: TRAP/hang status %s" % (name, group, case["argv"], status))
                    continue
                if group == "dotdot":
                    if status != 2 or out != ("refused " + TAG + "\n").encode():
                        g.fail("fixture %s/dotdot %r: want 'refused %s' exit 2, got status %s stdout %r" % (name, case["argv"], TAG, status, out[:120]))
                    continue
                if case.get("status") is not None and status != case["status"]:
                    g.fail("fixture %s/%s %r: want status %s, got %s (%r)" % (name, group, case["argv"], case["status"], status, out[:120]))
                if "stdout" in case and case["stdout"].encode() not in out:
                    g.fail("fixture %s/%s %r: stdout lacks %r (%r)" % (name, group, case["argv"], case["stdout"], out[:120]))


def sweeps(g, exes, cwd, sub, only):
    """Exhaustive: every prefix of each valid vector, and every argument truncated to every length."""
    for name, vecs in VALID.items():
        if only and name != only:
            continue
        for base in vecs:
            base = subst(base, sub)
            for k in range(len(base) + 1):
                check_run(g, name, exes[name], cwd, base[:k], "prefix")
            for i, a in enumerate(base):
                for j in range(0, len(a), max(1, len(a) // 12)):
                    check_run(g, name, exes[name], cwd, base[:i] + [a[:j]] + base[i + 1:], "truncate")


def fuzz(g, exes, cwd, sub, only):
    for name, vecs in VALID.items():
        if only and name != only:
            continue
        rnd = random.Random("%s/%d" % (name, SEED))
        for _ in range(N):
            base = subst(rnd.choice(vecs), sub)
            argv = mutate(rnd, base, name, sub)
            check_run(g, name, exes[name], cwd, argv, "fuzz")


def main():
    only = None
    if "--tool" in sys.argv:
        only = sys.argv[sys.argv.index("--tool") + 1]
    with tempfile.TemporaryDirectory() as tmp:
        exes = build(tmp)
        cwd, sub = scratch(tmp)
        g = Gate()
        fixtures(g, exes, cwd, sub, only)
        n_fix = g.checks
        sweeps(g, exes, cwd, sub, only)
        n_sw = g.checks - n_fix
        fuzz(g, exes, cwd, sub, only)
        n_fz = g.checks - n_fix - n_sw
    print("paths gate: %d fixtures, %d sweep vectors, %d fuzz vectors (seed %d), %d failures" % (n_fix, n_sw, n_fz, SEED, len(g.failures)))
    if g.failures:
        print("GATE FAILED")
        sys.exit(1)
    print("paths gate: all tools refuse a dotdot path with %s exit 2 and none traps" % TAG)


if __name__ == "__main__":
    main()

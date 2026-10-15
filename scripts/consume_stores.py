#!/usr/bin/env python3
"""PS6 (docs/stores.md section 5): the consumption proof. A scratch package OUTSIDE this repository pins the 23 stores of `.cancho-vcs` by git,
rev and path, runs `cancho install`, builds scripts/consume_stores/main.cho (it imports and calls every published module) and compares its
output with values computed here in Python from the definitions (SplitMix64, Fractions, the k-fold specification), never with this code's own answers.

    python3 scripts/consume_stores.py                       # url file://<this repository>, rev HEAD
    python3 scripts/consume_stores.py --rev HASH            # another commit of this repository
    python3 scripts/consume_stores.py --url https://github.com/alpibrusl/cancho-ml --rev HASH    # the pushed form
    python3 scripts/consume_stores.py --url file:///path/to/clone --rev HASH --no-check          # leave the freshness check to `cancho install`
    python3 scripts/consume_stores.py --out FILE            # also write the program output (two machines compare these files)

Steps: (1) clone URL at REV into a temporary directory; (2) `publish_stores.py --check` there: the stores at REV are what a fresh publish of REV's sources
gives (skipped by --no-check); (3) write the scratch package; (4) `cancho install`; (5) `cancho build --bin proof`; (6) run it and compare.
Exit 1 with the reason at the first step that fails. The compiler is $CANCHO or `cancho` on PATH, and must be the one cancho.toml pins.
"""
import argparse
import fractions
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import publish_stores  # noqa: E402  (the list of modules is one place)

M64 = (1 << 64) - 1
GOLD = 0x9E3779B97F4A7C15
URL = "https://github.com/alpibrusl/cancho-ml"


# ---------------------------------------------------------------- the expected values, from the definitions
def mix(z):
    z &= M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def output(s):
    return mix(s & M64)


def derive(h, c):
    return mix((((h & M64) ^ (c & M64)) + GOLD) & M64)


def unit(z):
    return ((z >> 11) & ((1 << 53) - 1)) * 2.0**-52 - 1.0


def element(seed, i):
    return unit(output(seed + (i + 1) * GOLD))


def bits(x):
    return "%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]


def kfold(n, k, seed, rep):
    """the specification of docs/metrics.md: Fisher-Yates from the end with a rejection-sampled bounded draw on the stream derive(derive(seed, 5), rep)"""
    state = [derive(derive(seed, 5), rep)]

    def draw():
        state[0] = (state[0] + GOLD) & M64
        return mix(state[0])

    def bounded(m):
        r = (1 << 63) % m
        while True:
            v = draw() >> 1
            if v < (1 << 63) - r:
                return v % m

    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = bounded(i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    out = [0] * n
    q, r = divmod(n, k)
    p = 0
    for f in range(k):
        for _ in range(q + (1 if f < r else 0)):
            out[perm[p]] = f
            p += 1
    return out


def expected():
    """{key: [values]} for the keys of main.cho whose value is fixed; `dot.one` is checked separately (a bound), the rest are exact"""
    F = fractions.Fraction
    fsq = F(10**16 - 1) ** 2 + F(1, 16)
    e = {
        "rng.output0": [bits_int(output(GOLD))],
        "rng.derive": [bits_int(derive(derive(42, 1), 0))],
        "rng.unit": [bits(unit(output(GOLD)))],
        "rng.element": [bits(element(7, 3))],
        "fbits.roundtrip": [bits(1.5)],
        "limits.ok": ["0"],
        "limits.refused": ["1"],
        "matrix.make": ["0"],
        "matrix.rows": ["3"],
        "matrix.cols": ["2"],
        "matrix.wpc": ["1"],
        "matrix.v00": [bits(1.5)],
        "matrix.v20": [bits(-2.25)],
        "matrix.v01": [bits(0.1)],
        "matrix.missing_mask": [str((1 << (1 * 2 + 0)) + (1 << (2 * 2 + 1)))],  # cells (1,0) and (2,1)
        "matrix.missing_col0": ["1"],
        "matrix.missing_col1": ["1"],
        "fsq.sum": [bits(float(fsq))],  # correctly rounded: float(Fraction) rounds once
        "fsq.status": ["0"],
        "exact.mean_pos": ["2333333"],  # round(7 * 10^6 / 3)
        "exact.mean_neg": ["-2333333"],
        "exact.mean_big": [str(round(F(4995 * 10**9 + 4999999995, 5) * 10**6))],
        "chol.code": ["0"],
        "chol.l": [bits(v) for v in (2.0, 6.0, 1.0, -8.0, 5.0, 3.0)],  # L of [[4,12,-16],[12,37,-43],[-16,-43,98]], row by row
        "chol.forward_code": ["0"],
        "chol.backward_code": ["0"],
        "chol.x0": [bits(1.0)],
        "chol.x1": [bits(2.0)],
        "chol.x2": [bits(3.0)],
        "par.equal": ["1", "1", "1"],
        "rank.auc_code": ["0"],
        "rank.auc": [bits(0.875)],  # labels 1 0 1 0, scores 9 8 8 1: (3 wins + 1 tie / 2) / 4 pairs
        "regress.rmse_code": ["0"],
        "regress.rmse": [bits(1.0)],
        "sums.sq": [bits(4.0)],
        "splits.code": ["0"],
        "splits.kfold": [" ".join(str(v) for v in kfold(10, 3, 7, 0))],
        "order.block": ["256"],
        "order.lanes": ["4"],
        "refuse.singular": ["ml.linalg.singular"],
        "ratio.third": [bits(1 / 3)],
        "gemm.code": ["0"],
        "gemm.c00": [bits(19.0)],
        "gemm.c01": [bits(22.0)],
        "gemm.c10": [bits(43.0)],
        "gemm.c11": [bits(50.0)],
        "gemv.code": ["0"],
        "gemv.y0": [bits(3.0)],
        "gemv.y1": [bits(7.0)],
        "axpy.code": ["0"],
        "axpy.y0": [bits(3.0)],
        "axpy.y1": [bits(5.0)],
        "syrk.code": ["0"],
        "syrk.c00": [bits(10.0)],
        "syrk.c01": [bits(14.0)],
        "syrk.c11": [bits(20.0)],
        "classify.code": ["0"],
        "classify.accuracy": [bits(0.75)],
    }
    return e


def bits_int(v):
    return "%016x" % v


def dot_bound():
    """the exact dot product of the two generated vectors, and the sum of |x y|: the lane-ordered double sum must lie within 1e-12 of it, relatively"""
    F = fractions.Fraction
    n = 5000
    xs = [F(element(1, k)) for k in range(n)]
    ys = [F(element(2, k)) for k in range(n)]
    return sum(x * y for x, y in zip(xs, ys)), sum(abs(x * y) for x, y in zip(xs, ys))


# ---------------------------------------------------------------- the scratch package
def run(args, cwd=None, env=None):
    return subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def compiler(pin):
    exe = os.environ.get("CANCHO") or shutil.which("cancho")
    if not exe:
        raise SystemExit("no compiler: set CANCHO, or put the cancho named in cancho.toml on PATH")
    got = subprocess.run([exe, "--version"], stdout=subprocess.PIPE, text=True, check=True).stdout
    m = re.search(r"rev ([0-9a-f]+)", got)
    if not m or m.group(1) != pin:
        raise SystemExit(f"{exe} is {m.group(1) if m else got.strip()}, cancho.toml pins {pin}")
    return exe


def manifest(pin, url, rev):
    lines = [f'[package]\nname = "stores-proof"\ncancho = "{pin}"\n']
    for module in sorted(publish_stores.MODULES):
        label = module.replace(".", "-")
        lines.append(f'[dependencies.{label}]\ngit = "{url}"\nrev = "{rev}"\npath = ".cancho-vcs/{module}"\n')
    lines.append('[[bin]]\nname = "proof"\nsources = ["src"]\nstd = true\nout = "build/proof"\n')
    return "\n".join(lines)


def fail(msg):
    print("consume_stores: FAILED: " + msg, file=sys.stderr)
    sys.exit(1)


def compare(text):
    want = expected()
    got = {}
    for line in text.splitlines():
        key, _, val = line.partition(" ")
        got.setdefault(key, []).append(val)
    problems = []
    for key, vals in want.items():
        if got.get(key) != vals:
            problems.append(f"{key}: got {got.get(key)}, want {vals}")
    for key in got:
        if key not in want and key != "dot.one":
            problems.append(f"{key}: not expected")
    if "dot.one" not in got:
        problems.append("dot.one: missing")
    else:
        exact, scale = dot_bound()
        b = int(got["dot.one"][0], 16)
        value = fractions.Fraction(struct.unpack("<d", struct.pack("<Q", b))[0])
        if abs(value - exact) > scale * fractions.Fraction(1, 10**12):
            problems.append(f"dot.one: {float(value)} is not within 1e-12 of the exact {float(exact)}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="file://" + ROOT)
    ap.add_argument("--rev")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    rev = args.rev or subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        fail(f"--rev {rev} is not a full commit hash")
    with tempfile.TemporaryDirectory(prefix="stores-proof-") as tmp:
        clone = os.path.join(tmp, "source")
        r = run(["git", "clone", "-q", args.url, clone])
        if r.returncode != 0:
            fail("git clone " + args.url + ":\n" + r.stdout)
        r = run(["git", "-C", clone, "checkout", "-q", rev])
        if r.returncode != 0:
            fail(f"{rev} is not a commit of {args.url}:\n{r.stdout}")
        pin = re.search(r'^cancho\s*=\s*"([0-9a-f]+)"', open(os.path.join(clone, "cancho.toml")).read(), re.M).group(1)
        exe = compiler(pin)
        env = dict(os.environ, CANCHO=exe)
        if not os.path.isdir(os.path.join(clone, ".cancho-vcs")):
            fail(f"{rev} has no .cancho-vcs: pin a commit that holds the stores")
        if not args.no_check:
            r = run([sys.executable, os.path.join(clone, "scripts", "publish_stores.py"), "--check", "--src", clone, "--out", clone], env=env)
            if r.returncode != 0:
                fail("the stores at " + rev[:12] + " are not what a fresh publish of its sources gives:\n" + r.stdout[-1500:])
        pkg = os.path.join(tmp, "package")
        os.makedirs(os.path.join(pkg, "src"))
        open(os.path.join(pkg, "cancho.toml"), "w").write(manifest(pin, args.url, rev))
        shutil.copy(os.path.join(ROOT, "scripts", "consume_stores", "main.cho"), os.path.join(pkg, "src", "main.cho"))
        r = run([exe, "install"], cwd=pkg, env=env)
        if r.returncode != 0:
            fail("cancho install:\n" + r.stdout[-1500:])
        installed = len([f for f in os.listdir(os.path.join(pkg, "build", "deps")) if f.endswith(".cho")])
        r = run([exe, "build", "--bin", "proof"], cwd=pkg, env=env)
        if r.returncode != 0:
            fail("cancho build --bin proof:\n" + r.stdout[-1500:])
        r = run([os.path.join(pkg, "build", "proof")], cwd=pkg)
        if r.returncode != 0:
            fail(f"the proof program exited {r.returncode}:\n" + r.stdout[-1500:])
        problems = compare(r.stdout)
        if problems:
            fail("the output differs from the definitions:\n  " + "\n  ".join(problems))
        if args.out:
            open(args.out, "w").write(r.stdout)
        print(r.stdout, end="")
        digest = hashlib.sha256(r.stdout.encode()).hexdigest()
        print(f"consume_stores: ok: {len(publish_stores.MODULES)} stores at {rev[:12]} from {args.url}, {installed} installed sources, "
              f"{len(r.stdout.splitlines())} lines, output sha256 {digest}")


if __name__ == "__main__":
    main()

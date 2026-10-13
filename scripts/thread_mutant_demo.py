#!/usr/bin/env python3
"""Show the thread-order mutant failing the byte-identity check (docs/linalg.md sections 1.3 and 5.5).
Builds linalg-check from the tree and from the tree with the `par-thread-order` mutant of scripts/mutants_linalg.py (partials
added in thread order), runs the same cases at 1, 2, 4, 8 and 16 threads, and prints the result hash of each. The real build
prints one hash per case; the mutant prints several (and the numpy order-matched oracle agrees with the real one only)."""
import os, pathlib, shutil, subprocess, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import mutlib, mutants_linalg as M

ROOT = mutlib.ROOT
CASES = [("dot", 1000000, 0), ("gemv_t", 100000, 64), ("syrk", 30000, 20)]
THREADS = [1, 2, 4, 8, 16]


def build(srcdir, out):
    files = sorted(str(p) for d in ("lib/core", "lib/linalg") for p in (pathlib.Path(srcdir) / d).glob("*.cho"))
    files.append(str(pathlib.Path(srcdir) / "tools/linalg-check/main.cho"))
    subprocess.run([mutlib.cancho(), "build", "--std", *files, "-o", out], check=True)


def hashes(binary):
    rows = {}
    for kernel, n, d in CASES:
        hs = []
        for t in THREADS:
            r = subprocess.run([binary, kernel, str(n), str(d), "1", "1", str(t)], capture_output=True, text=True)
            hs.append(r.stdout.split("\n")[0].split()[-1])
        rows[(kernel, n, d)] = hs
    return rows


def main():
    mut = [m for m in M.GROUPS["par"][1] if m[0] == "par-thread-order"][0]
    with tempfile.TemporaryDirectory() as tmp:
        good = os.path.join(tmp, "good"); bad = os.path.join(tmp, "bad")
        build(str(ROOT), good)
        shutil.copytree(ROOT / "lib", pathlib.Path(tmp) / "m" / "lib"); shutil.copytree(ROOT / "tools", pathlib.Path(tmp) / "m" / "tools")
        f = pathlib.Path(tmp) / "m" / mut[1]
        f.write_text(f.read_text().replace(mut[2], mut[3], 1))
        build(str(pathlib.Path(tmp) / "m"), bad)
        for label, b in (("real build", good), ("mutant: partials added in thread order", bad)):
            print(label)
            for (k, n, d), hs in hashes(b).items():
                print("  %-7s n=%-8d d=%-3d threads %s -> %s" % (k, n, d, ",".join(map(str, THREADS)), " ".join(hs)))
                print("      %s" % ("identical at every thread count" if len(set(hs)) == 1 else "DIFFERENT: %d distinct results; the byte-identity check FAILS" % len(set(hs))))

if __name__ == "__main__":
    main()

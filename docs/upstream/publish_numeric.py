#!/usr/bin/env python3
"""DRAFT for cancho-table (docs/upstream/cancho-table-numeric-package.md). NOT filed, NOT in cancho-table.

Publish the numeric modules of cancho-table's tools/table as `.cancho-vcs` stores, one store per module, each requiring the
stores of the modules it imports (docs/package-system.md 7.4, the way the compiler's scripts/publish_packages.py does it for
packages/x509 and packages/tls).

    python3 publish_numeric.py --src TABLE_CHECKOUT --out DIR            # (re)build DIR/.cancho-vcs/<module>
    python3 publish_numeric.py --src TABLE_CHECKOUT --out DIR --check    # fail if DIR/.cancho-vcs is not what a fresh publish gives

In cancho-table the proposal is `--src . --out .` (the stores are committed at `.cancho-vcs/<module>` at the repository root, as
cancho-tools commits `.cancho-vcs/toolbox.*`). Here it is run from a scratch directory against a read-only clone to PROVE that
the form works: nothing is written under --src.

The compiler is $CANCHO or ./target/release/cancho.
"""
import filecmp
import os
import re
import shutil
import subprocess
import sys
import tempfile

# the modules cancho-ml reads cells with; `facc` (the exact float sum) and `reader` (RFC 4180 records) are leaves of their own
MODULES = ["pow5", "dec", "flt", "query", "facc", "reader"]
STORE = ".cancho-vcs"


def compiler():
    exe = os.environ.get("CANCHO") or "target/release/cancho"
    if not os.path.exists(exe):
        raise SystemExit("set CANCHO to the compiler named in cancho.toml")
    return exe


def sources(src):
    """{module: (file, [imports that are not std])}"""
    found = {}
    for m in MODULES:
        path = os.path.join(src, "tools", "table", m + ".cho")
        text = open(path).read()
        decl = re.search(r"^module\s+(\w+)\s*;", text, re.M)
        if not decl or decl.group(1) != m:
            raise SystemExit(f"{path} does not declare module {m}")
        imports = [i for i in re.findall(r"^import\s+([\w.]+)\s*;", text, re.M) if not i.startswith("std.")]
        for i in imports:
            if i not in MODULES:
                raise SystemExit(f"{m} imports {i}, which is not published: add it to MODULES or cut the import")
        found[m] = (path, imports)
    return found


def order(found):
    done, out = set(), []

    def visit(m, path):
        if m in done:
            return
        if m in path:
            raise SystemExit("import cycle: " + " -> ".join(path + [m]))
        for dep in found[m][1]:
            visit(dep, path + [m])
        done.add(m)
        out.append(m)

    for m in MODULES:
        visit(m, [])
    return out


def publish(src, out):
    exe = compiler()
    found = sources(src)
    root = os.path.join(out, STORE)
    shutil.rmtree(root, ignore_errors=True)
    with tempfile.TemporaryDirectory() as locks:
        for m in order(found):
            path, deps = found[m]
            args = [exe, "vcs", "publish", "--store", os.path.join(root, m), "--std"]
            for dep in deps:
                lock = os.path.join(locks, dep + ".lock")
                if not os.path.exists(lock):
                    subprocess.run([exe, "vcs", "lock", "--store", os.path.join(root, dep), "-o", lock, "--all"], check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                args += ["--requires", f"{lock}:{os.path.join(root, dep)}"]
            r = subprocess.run(args + [path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if r.returncode != 0:
                raise SystemExit(f"publishing {m} failed:\n{r.stdout}")
            print(f"{m}: {r.stdout.count('published ')} declarations")


def differences(a, b):
    cmp = filecmp.dircmp(a, b)
    bad = [os.path.join(a, n) for n in cmp.left_only] + [os.path.join(b, n) for n in cmp.right_only]
    bad += [os.path.join(a, n) for n in cmp.diff_files + cmp.funny_files]
    for sub in cmp.common_dirs:
        bad += differences(os.path.join(a, sub), os.path.join(b, sub))
    return bad


def main():
    args = sys.argv[1:]
    if "--src" not in args or "--out" not in args:
        raise SystemExit(__doc__)
    src = args[args.index("--src") + 1]
    out = args[args.index("--out") + 1]
    if "--check" not in args:
        publish(src, out)
        return
    with tempfile.TemporaryDirectory() as scratch:
        publish(src, scratch)
        committed = os.path.join(out, STORE)
        if not os.path.isdir(committed):
            raise SystemExit(committed + " is missing")
        bad = differences(committed, os.path.join(scratch, STORE))
        if bad:
            print("the committed stores are not what a fresh publish gives:")
            print("\n".join("  " + p for p in bad[:20]))
            sys.exit(1)
        print("the committed stores match a fresh publish")


if __name__ == "__main__":
    main()

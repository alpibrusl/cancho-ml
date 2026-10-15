#!/usr/bin/env python3
"""Publish selected modules of lib/core, lib/linalg and lib/metrics as `.cancho-vcs` stores, one store per module, each requiring the stores of
the published modules it imports (cancho docs/package-system.md 7.4). docs/stores.md says what is published, why, and what is checked.

    python3 scripts/publish_stores.py            # rewrite .cancho-vcs/<module> and docs/stores_api.txt
    python3 scripts/publish_stores.py --check    # change nothing; exit 1 if a committed store or the API file is not what a fresh publish gives
    python3 scripts/publish_stores.py --closure  # print each module, its imports and the closure; exit 1 if the list is not closed

--modules A,B,C narrows the list to those modules of MODULES (the closure mutants list a module without its import).

--src DIR and --out DIR (both default to the repository root) read the sources from, and write the stores to, other places.
The compiler is $CANCHO, or `cancho` on PATH; it must be the commit cancho.toml names, because the bytes of a store depend on it.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = ".cancho-vcs"
API = os.path.join("docs", "stores_api.txt")
# the published modules (docs/stores.md section 1): module name -> folder. The order here is not the publishing order (that is topological).
MODULES = {
    "ml.refuse": "lib/core", "ml.limits": "lib/core", "ml.rng": "lib/core", "ml.exact": "lib/core", "ml.fbits": "lib/core",
    "ml.fsq": "lib/core", "ml.matrix": "lib/core",
    "linalg.order": "lib/linalg", "linalg.tensor": "lib/linalg", "linalg.dot": "lib/linalg", "linalg.axpy": "lib/linalg",
    "linalg.gemv": "lib/linalg", "linalg.syrk": "lib/linalg", "linalg.gemm": "lib/linalg", "linalg.chol": "lib/linalg",
    "linalg.par": "lib/linalg",
    "metrics.rules": "lib/metrics", "metrics.ratio": "lib/metrics", "metrics.classify": "lib/metrics", "metrics.sums": "lib/metrics",
    "metrics.regress": "lib/metrics", "metrics.rank": "lib/metrics", "metrics.splits": "lib/metrics",
}
FOLDERS = ("lib/core", "lib/linalg", "lib/metrics")


def compiler(src):
    exe = os.environ.get("CANCHO") or shutil.which("cancho")
    if not exe:
        raise SystemExit("no compiler: set CANCHO, or put the cancho named in cancho.toml on PATH")
    pin = re.search(r'^cancho\s*=\s*"([0-9a-f]+)"', open(os.path.join(src, "cancho.toml")).read(), re.M)
    got = subprocess.run([exe, "--version"], stdout=subprocess.PIPE, text=True, check=True).stdout
    rev = re.search(r"rev ([0-9a-f]+)", got)
    if not pin or not rev or rev.group(1) != pin.group(1):
        raise SystemExit(f"{exe} is {rev.group(1) if rev else got.strip()}, cancho.toml pins {pin.group(1) if pin else '?'}")
    return exe


def file_of(src, module):
    return os.path.join(src, MODULES[module], module.split(".")[-1] + ".cho")


def read_module(src, module, modules=None):
    """(path, text, [imports that are not std]) of a listed module; refuses a file that does not declare its name"""
    path = file_of(src, module)
    if not os.path.isfile(path):
        raise SystemExit(f"{path} does not exist (module {module})")
    text = open(path).read()
    decl = re.search(r"^module\s+([\w.]+)\s*;", text, re.M)
    if not decl or decl.group(1) != module:
        raise SystemExit(f"{path} does not declare module {module}")
    imports = [i for i in re.findall(r"^import\s+([\w.]+)\s*;", text, re.M) if not i.startswith("std.")]
    return path, text, imports


def sources(src, modules=None):
    """{module: (file, text, [imports that are not std])}; refuses a list that is not closed, or a module outside the three folders"""
    modules = MODULES if modules is None else modules
    found, problems = {}, []
    for m in modules:
        if modules[m] not in FOLDERS:
            problems.append(f"{m} is in {modules[m]}, not in one of {', '.join(FOLDERS)}")
            continue
        found[m] = read_module(src, m)
        problems += [f"{m} imports {i}, which is not published: add it to MODULES or cut the import" for i in found[m][2] if i not in modules]
    if problems:
        raise SystemExit("the published set is not self-contained:\n" + "\n".join("  " + p for p in problems))
    return found


def order(found):
    """the modules, each after the modules it imports (ties by name)"""
    done, out = set(), []

    def visit(m, path):
        if m in done:
            return
        if m in path:
            raise SystemExit("import cycle: " + " -> ".join(path + [m]))
        for dep in sorted(found[m][2]):
            visit(dep, path + [m])
        done.add(m)
        out.append(m)

    for m in sorted(found):
        visit(m, [])
    return out


def closure(found):
    """{module: sorted transitive published imports}"""
    out = {}
    for m in order(found):
        s = set()
        for d in found[m][2]:
            s |= {d} | set(out[d])
        out[m] = sorted(s)
    return out


def api_lines(found):
    """one line per `pub` declaration of the published modules: `module: pub ...` up to the body or the semicolon, whitespace collapsed"""
    lines = []
    for m in sorted(found):
        text = found[m][1]
        for decl in re.finditer(r"^pub\s[^{;]*", text, re.M):
            lines.append(f"{m}: " + " ".join(decl.group(0).split()))
    return lines


def write_api(found, out):
    path = os.path.join(out, API)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# The pub declarations of the published modules (scripts/publish_stores.py writes it; docs/stores.md section 2).\n")
        f.write("".join(l + "\n" for l in api_lines(found)))


def publish(src, out, quiet=False):
    exe = compiler(src)
    found = sources(src)
    root = os.path.join(out, STORE)
    shutil.rmtree(root, ignore_errors=True)
    with tempfile.TemporaryDirectory() as locks:
        for m in order(found):
            path, _, deps = found[m]
            args = [exe, "vcs", "publish", "--store", os.path.join(root, m), "--std"]
            for dep in sorted(deps):
                lock = os.path.join(locks, dep + ".lock")
                if not os.path.exists(lock):
                    subprocess.run([exe, "vcs", "lock", "--store", os.path.join(root, dep), "-o", lock, "--all"], check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                args += ["--requires", f"{lock}:{os.path.join(root, dep)}"]
            r = subprocess.run(args + [path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if r.returncode != 0:
                raise SystemExit(f"publishing {m} failed:\n{r.stdout}")
            if not quiet:
                print(f"{m}: {r.stdout.count('published ')} declarations")
    write_api(found, out)


def tree(root):
    """{relative path: bytes} of every file under root (a symlink or a special file is read as what it is, so it differs)"""
    files = {}
    for here, dirs, names in os.walk(root):
        for n in dirs + names:
            full = os.path.join(here, n)
            rel = os.path.relpath(full, root)
            if os.path.islink(full) or not (os.path.isdir(full) or os.path.isfile(full)):
                files[rel] = b"<link or special file>"
            elif os.path.isfile(full):
                files[rel] = open(full, "rb").read()
            else:
                files[rel] = None  # a directory (an empty extra one differs too)
    return files


def differences(committed, fresh):
    """what is in one tree only, or has other bytes (never the file times: contents are compared)"""
    a, b = tree(committed), tree(fresh)
    bad = [f"{p} (not in a fresh publish)" for p in sorted(a.keys() - b.keys())]
    bad += [f"{p} (missing)" for p in sorted(b.keys() - a.keys())]
    bad += [f"{p} (differs)" for p in sorted(a.keys() & b.keys()) if a[p] != b[p]]
    return bad


def print_closure(src):
    found = sources(src)
    clo = closure(found)
    for m in order(found):
        print(f"{m:18} imports {', '.join(found[m][2]) or '-':45} closure {', '.join(clo[m]) or '-'}")
    print(f"{len(found)} modules; every import of a published module is published (std.* aside)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=ROOT)
    ap.add_argument("--out", default=ROOT)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--closure", action="store_true")
    ap.add_argument("--modules")
    args = ap.parse_args()
    if args.modules:
        names = args.modules.split(",")
        unknown = [n for n in names if n not in MODULES]
        if unknown:
            raise SystemExit("not in MODULES: " + ", ".join(unknown))
        for m in [m for m in MODULES if m not in names]:
            del MODULES[m]
    if args.closure:
        print_closure(args.src)
        return
    if not args.check:
        publish(args.src, args.out)
        return
    committed = os.path.join(args.out, STORE)
    if not os.path.isdir(committed):
        raise SystemExit(committed + " is missing")
    api = os.path.join(args.out, API)
    with tempfile.TemporaryDirectory() as scratch:
        publish(args.src, scratch, quiet=True)
        bad = differences(committed, os.path.join(scratch, STORE))
        if not os.path.isfile(api):
            bad.append(API + " (missing)")
        elif open(api, "rb").read() != open(os.path.join(scratch, API), "rb").read():
            bad.append(API + " (differs)")
    if bad:
        print("the committed stores are not what a fresh publish gives:")
        print("\n".join("  " + p for p in bad[:20]))
        sys.exit(1)
    print("the committed stores match a fresh publish")


if __name__ == "__main__":
    main()

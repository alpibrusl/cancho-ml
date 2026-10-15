#!/usr/bin/env python3
"""Run the conformance tests of tests/conformance, one process per test module, a few at a time (docs/ci.md section 5).

    python3 scripts/run_conformance.py [-j N] [--no-oracles] [MODULE ...]

The same tests as `cd tests/conformance && python3 -m unittest discover -p 'test_*.py'`, which runs them one after the other (about 25 minutes on a
16-core Linux machine). Each module runs in its own `python3 -m unittest` process; the modules run N at a time (default: the CPUs, at most 4), and the output of a failing module is
printed in full. A module that cannot be imported fails the run, with one exception, named with --no-oracles: a module whose import needs numpy, scipy or scikit-learn (the
oracles of the numerical gates) is listed as NOT RUN, which is what the fast job of the workflow does, because it installs none of them. The full gate installs them and
runs everything. Exit 0 only if every module that ran passed.
"""
import argparse
import concurrent.futures
import os
import pathlib
import re
import subprocess
import sys
import time

CONF = pathlib.Path(__file__).resolve().parent.parent / "tests" / "conformance"
ORACLE_PKGS = ("numpy", "scipy", "sklearn")


def run_module(mod, no_oracles):
    t = time.time()
    p = subprocess.run([sys.executable, "-m", "unittest", mod], cwd=CONF, capture_output=True, text=True)
    out = p.stdout + p.stderr
    dt = time.time() - t
    if p.returncode != 0 and no_oracles:
        m = re.search(r"ModuleNotFoundError: No module named '(\w+)", out)
        if m and m.group(1) in ORACLE_PKGS and "Failed to import test module" in out:
            return mod, "not-run", dt, "needs " + m.group(1)
    ran = re.search(r"Ran (\d+) tests?", out)
    return mod, "ok" if p.returncode == 0 else "FAIL", dt, (ran.group(1) + " tests" if ran else "") + ("" if p.returncode == 0 else "\n" + out)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--no-oracles", action="store_true")
    ap.add_argument("modules", nargs="*")
    a = ap.parse_args(argv)
    mods = a.modules or sorted(p.stem for p in CONF.glob("test_*.py"))
    if not mods:
        print("run_conformance: no test modules in %s" % CONF, file=sys.stderr)
        return 2
    t0 = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.j) as ex:
        for r in ex.map(lambda m: run_module(m, a.no_oracles), mods):
            results.append(r)
            print("%-8s %-34s %6.1fs  %s" % (r[1], r[0], r[2], r[3].split("\n")[0]), flush=True)
    bad = [r for r in results if r[1] == "FAIL"]
    for r in bad:
        print("\n==== %s ====\n%s" % (r[0], r[3]))
    nr = [r[0] for r in results if r[1] == "not-run"]
    tests = sum(int(re.match(r"(\d+)", r[3]).group(1)) for r in results if r[1] == "ok" and re.match(r"(\d+)", r[3]))
    print("run_conformance: %d modules, %d passed (%d tests), %d failed, %d not run%s, %.0fs" % (
        len(results), len(results) - len(bad) - len(nr), tests, len(bad), len(nr), (" (" + ", ".join(nr) + ")") if nr else "", time.time() - t0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

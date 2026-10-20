#!/usr/bin/env python3
"""Run the conformance tests of tests/conformance, one process per test module, a few at a time (docs/ci.md section 5).

    python3 scripts/run_conformance.py [-j N] [--no-oracles] [MODULE ...]

The same tests as `cd tests/conformance && python3 -m unittest discover -p 'test_*.py'`, which runs them one after the other (about 25 minutes on a
16-core Linux machine). Each module runs in its own `python3 -m unittest` process; the modules run N at a time (default: the CPUs, at most 4), and the output of a failing module is
printed in full. A module in SHARDS runs as several processes (`ST_SHARD=I/N`, docs/ci.md section 5.2) and is reported when its last shard ends. The tasks start in the order of
EXPECTED, the longest first, and a module is printed when it ends. A module that cannot be imported fails the run, with one exception, named with --no-oracles: a module whose import needs numpy, scipy or scikit-learn (the
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
# seconds per module, measured on a GitHub-hosted ubuntu-24.04 (4 cores, run 39256778633, ST_STRIDE=5 FUZZ_N=500; test_fuzz_interp: 221 s alone on the build machine of the PR, one run): only an order of start, never what runs (docs/ci.md 5.2)
EXPECTED = {"test_describe": 1650, "test_fuzz_args": 1022, "test_interp": 625, "test_fuzz_interp": 421, "test_wire_boost_file": 503, "test_wire_trees_file": 389, "test_model_file": 163, "test_wire_kmeans_file": 134,
            "test_serving": 89, "test_wire_file": 88, "test_serve_fuzz": 73, "test_track": 68, "test_nonfinite": 66, "test_splits": 54, "test_serve_json": 39}
SHARDS = {"test_describe": 4, "test_fuzz_args": 3}


def make_tasks(mods, expected, shards):
    """(module, shard index, shard count) for every module and shard, the largest expected time of a task first, the others in the order given"""
    tasks = [(m, i, shards.get(m, 1)) for m in mods for i in range(shards.get(m, 1))]
    return sorted(tasks, key=lambda t: -expected.get(t[0], 0) / t[2])


def run_task(mod, i, n, no_oracles):
    """one process of one module (shard i of n) -> (status, seconds, first line or tests, output)"""
    t = time.time()
    env = dict(os.environ, ST_SHARD="%d/%d" % (i, n)) if n > 1 else None
    p = subprocess.run([sys.executable, "-m", "unittest", mod], cwd=CONF, capture_output=True, text=True, env=env)
    out = p.stdout + p.stderr
    dt = time.time() - t
    if p.returncode != 0 and no_oracles:
        m = re.search(r"ModuleNotFoundError: No module named '(\w+)", out)
        if m and m.group(1) in ORACLE_PKGS and "Failed to import test module" in out:
            return "not-run", dt, "needs " + m.group(1), ""
    ran = re.search(r"Ran (\d+) tests?", out)
    return ("ok" if p.returncode == 0 else "FAIL"), dt, (ran.group(1) + " tests" if ran else ""), ("" if p.returncode == 0 else out)


def merge(mod, parts):
    """the result of a module from the results of its shards: (module, status, seconds, text); the text of a FAIL is the output of its failing shards"""
    bad = [p for p in parts if p[0] == "FAIL"]
    if bad:
        return mod, "FAIL", max(p[1] for p in parts), "\n" + "\n".join(p[3] for p in bad)
    nr = [p for p in parts if p[0] == "not-run"]
    if nr:
        return mod, "not-run", max(p[1] for p in parts), nr[0][2]
    tests = sum(int(re.match(r"(\d+)", p[2]).group(1)) for p in parts if re.match(r"(\d+)", p[2]))
    return mod, "ok", max(p[1] for p in parts), ("%d tests" % tests) if tests else ""


def run_all(tasks, jobs, task_runner, no_oracles, emit):
    """start the tasks in order, `jobs` at a time; call emit(result) for a module as soon as its last shard has ended; return the results in the order they ended"""
    count = {}
    for m, _, n in tasks:
        count[m] = n
    parts = {m: [] for m in count}
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(task_runner, m, i, n, no_oracles): m for m, i, n in tasks}
        for f in concurrent.futures.as_completed(futs):
            m = futs[f]
            parts[m].append(f.result())
            if len(parts[m]) == count[m]:
                r = merge(m, parts[m])
                results.append(r)
                emit(r)
    return results


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
    results = run_all(make_tasks(mods, EXPECTED, SHARDS), a.j, run_task, a.no_oracles,
                      lambda r: print("%-8s %-34s %6.1fs  %s" % (r[1], r[0], r[2], r[3].split("\n")[0]), flush=True))
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

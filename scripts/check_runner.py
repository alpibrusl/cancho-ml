#!/usr/bin/env python3
"""The gates R2, R3 and R4 of docs/ci.md section 5.2, on scripts/run_conformance.py, and the mutants that show they can fail.

    python3 scripts/check_runner.py            # the gates on the script
    python3 scripts/check_runner.py --mutants  # each edit of the script must be caught by a gate

R1 (the shards partition the damaged files of `test_describe`) is a test of that module, `test_the_shards_partition_the_damage`.
No test module runs here: the runner is driven with stand-ins for the process that runs a task.
"""
import importlib.util
import pathlib
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE / "run_conformance.py"


def load(path):
    spec = importlib.util.spec_from_file_location("run_conformance_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gates(rc):
    """-> a list of what is wrong; empty when R2, R3 and R4 hold"""
    bad = []
    # R2: the order of the tasks
    order = [(m, i, n) for m, i, n in rc.make_tasks(["a", "b", "c", "d"], {"b": 100, "c": 50}, {"b": 4})]
    want = [("c", 0, 1)] + [("b", i, 4) for i in range(4)] + [("a", 0, 1), ("d", 0, 1)]
    if order != want:
        bad.append("R2: the tasks are %r, not %r" % (order, want))
    sh = rc.make_tasks(["x", "y"], {}, {"x": 3})
    if sorted(sh) != [("x", 0, 3), ("x", 1, 3), ("x", 2, 3), ("y", 0, 1)]:
        bad.append("R2: a module is not present once per shard: %r" % (sh,))
    # R3: a module that ends is printed before a slower one that began earlier
    def run_task(mod, i, n, no_oracles):
        time.sleep({"slow": 0.8, "fast": 0.05}[mod])
        return ("ok", 0.0, "3 tests", "")
    seen = []
    rc.run_all([("slow", 0, 1), ("fast", 0, 1)], 2, run_task, False, lambda r: seen.append(r[0]))
    if seen != ["fast", "slow"]:
        bad.append("R3: printed in the order %r, not fast then slow" % (seen,))
    # R4: a module with shards
    def run_shards(mod, i, n, no_oracles):
        if i == 2:
            return ("FAIL", 1.0, "", "the output of shard 2")
        return ("ok", 1.0 + i, "%d tests" % (5 + i), "")
    seen = []
    rc.run_all([("m", i, 3) for i in range(3)], 3, run_shards, False, lambda r: seen.append(r))
    if len(seen) != 1 or seen[0][0] != "m":
        bad.append("R4: a module with three shards is reported %d times" % len(seen))
    elif seen[0][1] != "FAIL" or "the output of shard 2" not in seen[0][3]:
        bad.append("R4: a failing shard does not fail the module with its output: %r" % (seen[0],))
    def run_ok(mod, i, n, no_oracles):
        return ("ok", 1.0 + i, "%d tests" % (5 + i), "")
    seen = []
    rc.run_all([("m", i, 3) for i in range(3)], 3, run_ok, False, lambda r: seen.append(r))
    if not seen or seen[0][1] != "ok" or not seen[0][3].startswith("18 tests"):
        bad.append("R4: the tests of the shards are not summed (5 + 6 + 7): %r" % (seen,))
    return bad


def main(argv):
    if "--mutants" in argv:
        return mutants()
    bad = gates(load(SCRIPT))
    for b in bad:
        print("check_runner: " + b)
    if bad:
        return 1
    print("check_runner: ok (R2 order, R3 printing, R4 shards of a module)")
    return 0


MUTANTS = [  # (name, old, new): each text occurs once in scripts/run_conformance.py
    ("R2 the smallest first", "key=lambda t: -expected.get(t[0], 0) / t[2])", "key=lambda t: expected.get(t[0], 0) / t[2])"),
    ("R2 a shard costs as much as its module", "key=lambda t: -expected.get(t[0], 0) / t[2])", "key=lambda t: -expected.get(t[0], 0))"),
    ("R2 the last shard of a module is dropped", "for i in range(shards.get(m, 1))]", "for i in range(max(1, shards.get(m, 1) - 1))]"),
    ("R3 printed in the order of submission", "for f in concurrent.futures.as_completed(futs):", "for f in futs:"),
    ("R4 a module is reported at every shard", "if len(parts[m]) == count[m]:", "if len(parts[m]) >= 1:"),
    ("R4 a failing shard does not fail the module", '    if bad:\n        return mod, "FAIL"', '    if False:\n        return mod, "FAIL"'),
    ("R4 the output of the failing shard is dropped", 'max(p[1] for p in parts), "\\n" + "\\n".join(p[3] for p in bad)', 'max(p[1] for p in parts), "\\n"'),
    ("R4 only the first shard's tests are counted", "for p in parts if re.match", "for p in parts[:1] if re.match"),
]


def mutants():
    text = SCRIPT.read_text()
    survived = 0
    for name, old, new in MUTANTS:
        if text.count(old) != 1:
            print("check_runner: the edit %r applies %d times, not once" % (name, text.count(old)))
            return 1
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "run_conformance.py"
            p.write_text(text.replace(old, new))
            try:
                bad = gates(load(p))
            except Exception as e:  # a mutant that crashes is caught as well
                bad = ["crashed: %r" % (e,)]
        print("%-52s %s" % (name, "caught" if bad else "SURVIVED"))
        survived += not bad
    print("check_runner --mutants: %d mutants, %d survived" % (len(MUTANTS), survived))
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

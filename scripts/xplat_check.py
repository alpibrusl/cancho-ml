#!/usr/bin/env python3
"""The cross-platform determinism gate (docs/ci.md section 2): the fingerprint of `ml` against a committed golden.

The fingerprint is one tab-separated line per artefact, for 22 fits of every model kind on the bundled example data (docs/examples/data):
`case  fit-exit  rc`, `case  model-file  sha256`, `case  predict  rc  sha256`, `case  eval  rc  sha256`. A model file is hashed whole; a
predict or eval answer is hashed as the bytes the program printed. The golden, docs/bench/xplat_fp.txt, is those 88 lines as written on
macOS arm64 and read back on Linux x86-64 (docs/ci.md section 3). Every platform of the CI matrix runs this script on its own build and must
print the golden's bytes: equal to the golden is equal to each other.

    python3 scripts/xplat_check.py [--bin build] [--golden docs/bench/xplat_fp.txt] [--data docs/examples/data]
    python3 scripts/xplat_check.py --update    # rewrite the golden: only when the model bytes change on purpose, in a commit that says why
    python3 scripts/xplat_check.py --mutants   # the checks are shown able to fail

Besides the golden, the run itself must satisfy (each is a failure with its own message, whatever the golden says):

* every fit, predict and eval exits 0 (a failed fit fails the gate loudly; it is never only printed);
* the model file exists, and the cases that differ only in `--threads` (1 and 8) produce the same lines (thread invariance);
* `ml-predict`, when it is built next to `ml`, prints the same bytes as `ml` for every predict (it has no eval: docs/serving.md section 3);
* the line count and the case names are the golden's (a missing or an extra case is a failure).

Exit 0: all of that holds. Exit 1: a readable report with a diff. Exit 2: the script could not run (no binary, no golden).
"""
import argparse
import difflib
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TIMEOUT = 300  # seconds per command: a hang is a failure, not a stuck job

STD = "scale column=* kind=standard"
CAT = "encode column=species kind=onehot min-count=4;freq column=city;scale column=age,visits kind=standard"
HASH = "hash column=species buckets=8;encode column=city kind=label;scale column=age,visits kind=standard"


def cases():
    """(name, train, test, fit args, predict args) in the golden's order."""
    out = []
    for t in ("1", "8"):
        out += [
            ("linear-diabetes-t" + t, "diabetes_train", "diabetes_test", ["--model", "linear", "--step", STD, "--l2", "0.05", "--threads", t], []),
            ("logistic-iris-t" + t, "iris_train", "iris_test", ["--model", "logistic", "--step", STD, "--l2", "0.1", "--threads", t], ["--proba"]),
            ("logistic-wine-t" + t, "breast_cancer_train", "breast_cancer_test", ["--model", "logistic", "--step", STD, "--l2", "0.1", "--threads", t], ["--proba"]),
            ("tree-diabetes-t" + t, "diabetes_train", "diabetes_test", ["--model", "tree", "--max-depth", "6", "--threads", t], []),
            ("forest-diabetes-t" + t, "diabetes_train", "diabetes_test", ["--model", "forest", "--trees", "20", "--seed", "7", "--threads", t], []),
            ("forest-cancer-t" + t, "breast_cancer_train", "breast_cancer_test", ["--model", "forest", "--task", "classification", "--trees", "20", "--seed", "7", "--threads", t], ["--proba"]),
            ("kmeans-iris-pp-t" + t, "iris_train", "iris_test", ["--model", "kmeans", "--k", "3", "--seed", "1", "--step", STD, "--drop", "virginica", "--threads", t], ["--distance", "--drop", "virginica"]),
            ("kmeans-iris-rand-t" + t, "iris_train", "iris_test", ["--model", "kmeans", "--k", "4", "--init", "random", "--seed", "5", "--step", STD, "--drop", "virginica", "--threads", t], ["--distance", "--drop", "virginica"]),
        ]
    for t in ("1", "8"):
        out += [
            ("logistic-pets-cat-t" + t, "pets", "pets", ["--model", "logistic", "--cat", "species,city", "--step", CAT, "--drop", "id", "--l2", "0.1", "--iterations", "50", "--threads", t], ["--proba", "--drop", "id"]),
            ("forest-pets-hash-t" + t, "pets", "pets", ["--model", "forest", "--task", "classification", "--trees", "10", "--seed", "3", "--cat", "species,city", "--step", HASH, "--drop", "id", "--threads", t], ["--proba", "--drop", "id"]),
        ]
    out.append(("constant-reg", "diabetes_train", "diabetes_test", ["--model", "constant"], []))
    out.append(("constant-cls", "iris_train", "iris_test", ["--model", "constant"], []))
    return out


def sha(b):
    return hashlib.sha256(b).hexdigest()


def last_column(path):
    with open(path) as f:
        return f.readline().strip().split(",")[-1]


def run(binary, work, args):
    """(exit code, stdout bytes). A binary that hangs, or cannot start, is a failure of the caller's, never an exception out of here."""
    try:
        p = subprocess.run([binary, "--root", "."] + args, cwd=work, capture_output=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return -1, b"timeout after %d s" % TIMEOUT
    except OSError as e:
        return -2, str(e).encode()
    return p.returncode, p.stdout


def fingerprint(ml, data, ml_predict=None):
    """-> (lines, failures). lines are the golden's form; failures are strings, each a run-level fault (not a golden difference)."""
    lines, failures = [], []
    work = tempfile.mkdtemp(prefix="xplat-")
    try:
        for f in os.listdir(data):
            if f.endswith(".csv"):
                shutil.copy(os.path.join(data, f), work)
        for name, train, test, fit_args, pred_args in cases():
            tf, sf, out = train + ".csv", test + ".csv", name + ".ml"
            if not (os.path.exists(os.path.join(work, tf)) and os.path.exists(os.path.join(work, sf))):
                failures.append("%s: data file %s or %s is not in %s" % (name, tf, sf, data))
                continue
            fa = ["fit"] + fit_args
            if fit_args[fit_args.index("--model") + 1] != "kmeans":
                fa += ["--target", last_column(os.path.join(work, tf))]
            rc, so = run(ml, work, fa + ["--overwrite", "--output", out, tf])
            lines.append("%s\tfit-exit\t%d" % (name, rc))
            if rc != 0:
                failures.append("%s: fit exited %d: %s" % (name, rc, so[:300].decode("utf-8", "replace")))
                continue
            try:
                with open(os.path.join(work, out), "rb") as fh:
                    lines.append("%s\tmodel-file\t%s" % (name, sha(fh.read())))
            except OSError as e:
                failures.append("%s: fit exited 0 but wrote no model file: %s" % (name, e))
                continue
            steps = [("predict", ["predict", "--model-file", out, "--limit", "100000"] + pred_args + [sf])]
            ev = ["eval", "--model-file", out]
            if "--drop" in pred_args:
                ev += ["--drop", pred_args[pred_args.index("--drop") + 1]]
            steps.append(("eval", ev + [sf]))
            for what, args in steps:
                rc, so = run(ml, work, args)
                lines.append("%s\t%s\t%d\t%s" % (name, what, rc, sha(so)))
                if rc != 0:
                    failures.append("%s: %s exited %d: %s" % (name, what, rc, so[:300].decode("utf-8", "replace")))
                if ml_predict and what == "predict":
                    rc2, so2 = run(ml_predict, work, args)
                    if (rc2, so2) != (rc, so):
                        failures.append("%s: ml-predict %s differs from ml (exit %d against %d, %d bytes against %d)" % (name, what, rc2, rc, len(so2), len(so)))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    failures += thread_faults(lines)
    return lines, failures


def thread_faults(lines):
    """The cases `X-t1` and `X-t8` differ only in --threads: every line after the case name must be equal."""
    by = {}
    for ln in lines:
        name, rest = ln.split("\t", 1)
        by.setdefault(name, []).append(rest)
    out = []
    for name in by:
        if name.endswith("-t1"):
            other = name[:-1] + "8"
            if other not in by:
                out.append("thread invariance: %s has no partner %s" % (name, other))
            elif by[name] != by[other]:
                diff = [r for r in by[name] if r not in by[other]]
                out.append("thread invariance: %s and %s differ (1 and 8 threads give other bytes): %s" % (name, other, "; ".join(d.replace("\t", " ")[:60] for d in diff)))
    return out


def compare(lines, golden_text):
    """-> list of report lines, empty when the fingerprint is the golden's."""
    gold = golden_text.splitlines()
    if lines == gold:
        return []
    rep = []
    gn = {l.split("\t")[0] for l in gold}
    ln = {l.split("\t")[0] for l in lines}
    for c in sorted(gn - ln):
        rep.append("case missing from this run: " + c)
    for c in sorted(ln - gn):
        rep.append("case not in the golden: " + c)
    if len(lines) != len(gold):
        rep.append("%d lines, the golden has %d" % (len(lines), len(gold)))
    rep.append("diff (golden -> this run):")
    rep += [d.rstrip("\n") for d in difflib.unified_diff(gold, lines, "golden", "this run", lineterm="", n=0)]
    return rep


def check(lines, failures, golden_text):
    rep = ["run fault: " + f for f in failures] + compare(lines, golden_text)
    return rep


def locate(bin_dir, name):
    p = os.path.join(bin_dir, name)
    return p if os.path.isfile(p) else None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", default=os.path.join(ROOT, "build"), help="the directory with ml (and ml-predict next to it)")
    ap.add_argument("--golden", default=os.path.join(ROOT, "docs", "bench", "xplat_fp.txt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "docs", "examples", "data"))
    ap.add_argument("--update", action="store_true", help="rewrite the golden (only for a deliberate change of model bytes)")
    ap.add_argument("--mutants", action="store_true", help="show that the check fails when it must")
    ap.add_argument("--require-predict", action="store_true", help="fail when ml-predict is not built next to ml (CI)")
    ap.add_argument("--print", action="store_true", help="print the fingerprint and exit (no golden read)")
    a = ap.parse_args(argv)
    ml = locate(a.bin, "ml")
    if not ml:
        print("xplat_check: no %s/ml (cancho install && cancho build --bin ml)" % a.bin, file=sys.stderr)
        return 2
    ml_predict = locate(a.bin, "ml-predict")
    if a.require_predict and not ml_predict:
        print("xplat_check: no %s/ml-predict (cancho build --bin ml-predict)" % a.bin, file=sys.stderr)
        return 2
    if not os.path.isdir(a.data):
        print("xplat_check: no data directory " + a.data, file=sys.stderr)
        return 2
    lines, failures = fingerprint(os.path.abspath(ml), os.path.abspath(a.data), os.path.abspath(ml_predict) if ml_predict else None)
    if a.print:
        print("\n".join(lines))
        return 1 if failures else 0
    if a.mutants:
        return mutants(a, lines, failures)
    if a.update:
        if failures:
            print("xplat_check --update refused: the run has faults, and a golden is written only from a clean run:", file=sys.stderr)
            for f in failures:
                print("  " + f, file=sys.stderr)
            return 1
        old = open(a.golden).read() if os.path.exists(a.golden) else ""
        with open(a.golden, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        changed = [d for d in compare(lines, old)] if old else ["(no golden before)"]
        print("xplat_check: wrote %s (%d lines)" % (a.golden, len(lines)))
        print("\n".join(changed))
        return 0
    if not os.path.exists(a.golden):
        print("xplat_check: no golden " + a.golden, file=sys.stderr)
        return 2
    rep = check(lines, failures, open(a.golden).read())
    if rep:
        print("xplat_check: FAIL (%s, %s)" % (sys.platform, os.uname().machine))
        print("\n".join(rep))
        return 1
    print("xplat_check: ok, %d lines for %d fits equal %s (%s, %s%s)" % (
        len(lines), len({l.split("\t")[0] for l in lines}), os.path.relpath(a.golden, ROOT), sys.platform, os.uname().machine,
        "; ml-predict equal to ml" if ml_predict else "; no ml-predict to compare"))
    return 0


# ---- the mutants: each one must make the check fail ----

WRAP = """#!/bin/sh
# a mutant of ml: %s
case "$*" in %s) %s ;; esac
exec %s "$@"
"""


# after the real fit, overwrite one byte of the file named by --output
FLIP = ("case \"$*\" in %s) prev=; for x in \"$@\"; do [ \"$prev\" = --output ] && f=$x; prev=$x; done; "
        "printf 'x' | dd of=\"$f\" bs=1 seek=40 conv=notrunc 2>/dev/null;; esac\n")


def mutants(a, lines, failures):
    if failures:
        print("xplat_check --mutants: the unmutated run has faults, so it cannot be mutated:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    golden = "\n".join(lines) + "\n"
    results = []

    def expect_fail(label, rep, needle):
        ok = bool(rep) and any(needle in r for r in rep)
        results.append((label, ok, rep[:2]))

    results.append(("control: the unmutated run passes", check(lines, [], golden) == [], []))
    # golden side: one hex digit of a stored hash, a missing case, a missing line, an extra case, a reordered pair, a changed exit code
    g = golden.splitlines()
    i = next(k for k, l in enumerate(g) if "\tmodel-file\t" in l)
    flip = g[i][:-1] + ("0" if g[i][-1] != "0" else "1")
    expect_fail("one hex digit of a stored model hash changed", check(lines, [], "\n".join(g[:i] + [flip] + g[i + 1:]) + "\n"), "diff")
    j = next(k for k, l in enumerate(g) if "\teval\t" in l)
    flip = g[j][:-1] + ("0" if g[j][-1] != "0" else "1")
    expect_fail("one hex digit of a stored eval hash changed", check(lines, [], "\n".join(g[:j] + [flip] + g[j + 1:]) + "\n"), "diff")
    first = g[0].split("\t")[0]
    expect_fail("a whole case missing from the golden", check(lines, [], "\n".join(l for l in g if l.split("\t")[0] != first) + "\n"), "case not in the golden: " + first)
    expect_fail("a whole case missing from the run", check([l for l in lines if l.split("\t")[0] != first], [], golden), "case missing from this run: " + first)
    expect_fail("one line missing from the run", check(lines[:5] + lines[6:], [], golden), "lines")
    expect_fail("an extra case in the golden", check(lines, [], golden + "extra-case\tfit-exit\t0\n"), "case missing from this run: extra-case")
    expect_fail("two lines swapped", check(lines[:1] + [lines[2], lines[1]] + lines[3:], [], golden), "diff")
    expect_fail("an empty golden", check(lines, [], ""), "case not in the golden")
    # run side: a fit that fails, a model file that differs by one byte, a predict answer that differs, ml-predict that differs
    real = os.path.abspath(os.path.join(a.bin, "ml"))
    real_p = os.path.join(a.bin, "ml-predict")
    tmp = tempfile.mkdtemp(prefix="xplat-mut-")
    data = os.path.abspath(a.data)
    try:
        def wrapper(name, desc, pattern, action, target):
            p = os.path.join(tmp, name)
            with open(p, "w") as fh:
                fh.write(WRAP % (desc, pattern, action, target))
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
            return p

        w = wrapper("ml-fitfail", "every k-means fit exits 1", "*fit*kmeans*", "echo '{\"ok\":false}'; exit 1", real)
        l2, f2 = fingerprint(w, data)
        expect_fail("a fit exits 1", check(l2, f2, golden), "fit exited 1")
        # one byte of a model file: the wrapper runs the real fit and then edits the first byte of the output of the forest-pets cases
        w = os.path.join(tmp, "ml-flipbyte")
        with open(w, "w") as fh:
            fh.write("#!/bin/sh\n# a mutant of ml: the model file of a tree fit loses its last byte's value\n"
                     "%s \"$@\" || exit $?\n" % real + FLIP % "*'fit --model tree'*")
        os.chmod(w, 0o755)
        l2, f2 = fingerprint(w, data)
        expect_fail("one byte of a stored tree model differs", check(l2, f2, golden), "diff")
        w = wrapper("ml-nopred", "predict prints a changed answer", "*predict*", "echo '{\"ok\":true}'; exit 0", real)
        l2, f2 = fingerprint(w, data)
        expect_fail("a predict answer differs", check(l2, f2, golden), "diff")
        if os.path.isfile(real_p):
            w = wrapper("ml-predict-other", "ml-predict prints another answer", "*predict*", "echo '{\"ok\":true}'; exit 0", real_p)
            l2, f2 = fingerprint(real, data, w)
            expect_fail("ml-predict differs from ml", check(l2, f2, golden), "ml-predict")
        # thread variance: the 8-thread forest fit writes another byte, the 1-thread fit does not; the golden is the unmutated one, but the thread check has its own message
        w = os.path.join(tmp, "ml-threads")
        with open(w, "w") as fh:
            fh.write("#!/bin/sh\n# a mutant of ml: a fit with --threads 8 of a forest writes another byte\n"
                     "%s \"$@\" || exit $?\n" % real + FLIP % "*'fit --model forest'*'--threads 8'*")
        os.chmod(w, 0o755)
        l2, f2 = fingerprint(w, data)
        expect_fail("a fit differs between 1 and 8 threads (the thread check, not only the golden)", f2 and ["run fault: " + x for x in f2] or [], "thread invariance")
        # a binary that does not exist
        l2, f2 = fingerprint(os.path.join(tmp, "nonexistent"), data)
        expect_fail("the binary cannot start", check(l2, f2, golden), "fit exited -2")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bad = 0
    for label, ok, sample in results:
        print("%-4s %s" % ("ok" if ok else "FAIL", label))
        if not ok:
            bad += 1
            for s in sample:
                print("       got: " + s[:200])
    print("xplat_check --mutants: %d checks, %d wrong" % (len(results), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

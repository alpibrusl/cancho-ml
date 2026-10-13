#!/usr/bin/env python3
"""Mutants for the gates of docs/cli.md section 1: a gate that cannot fail judges nothing.

Each mutant is a copy of the tree with one change (an edit to a source, a table, a schema or a fixture, or a stand-in binary),
the binary rebuilt from the copy, and the tests of the gate it targets run against it. KILLED means a test failed (or the
authority check did); SURVIVED means every test passed, and the mutant must then carry a reason: an equivalent mutant, or a
gap the section 7 of docs/cli.md names.

    python3 scripts/mutants_cli.py              # all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_cli.py --only ID    # one
    python3 scripts/mutants_cli.py --list
    python3 scripts/mutants_cli.py --check      # every `old` text is found exactly where it is expected (no build, no test)

The copy is built with the explicit file list that package mode would pass (`cancho build --std lib/core/*.cho build/deps/*.cho
generated/ml/built.cho tools/ml/*.cho`), so no mutant runs `cancho install`.
"""

import glob
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def compiler():
    here = ROOT.parent / "cancho-compiler-a4572ea" / "target" / "release" / "cancho"
    return os.environ.get("CANCHO") or (str(here) if here.exists() else "cancho")


# id, gate, what the mutant does, edits [(file, old, new, nth occurrence from 1)], tests (unittest targets), options
# options: regen (run scripts/schemas.py in the copy after the edits), kind ("wrapper:NAME", "authority", "manifest"), why (for a
# mutant expected to survive)
def M(id, gate, what, edits, tests, **opt):
    return dict(id=id, gate=gate, what=what, edits=edits, tests=tests, opt=opt)


MUTANTS = [
    # ------------------------------------------------------------------ T1: the tables
    M("t1-flag-without-fixture", "T1", "a flag is added to the table (schemas regenerated); no fixture uses it",
      [("tools/ml/spec.cho", ";format||choice:json/text|none|json|", ";seed||nat|none|0|a flag no fixture uses;format||choice:json/text|none|json|", 1)],
      ["test_tables"], regen=True),
    M("t1-flag-missing-from-the-schema", "T1", "the schema's flag enum loses --limit",
      [("schemas/ml.v1.json", '"--limit",', "", 1)], ["test_tables"]),
    M("t1-rule-without-fixture", "T1", "a rule is added to the table (schemas regenerated); no fixture reaches it",
      [("tools/ml/spec.cho", ";ml.model.malformed\";", ";ml.model.malformed;ml.input.too-wide\";", 1),
       ("tools/ml/spec.cho", ";limit.log-full|8|never|", ";ml.input.too-wide|8|never|a rule nothing reaches. Hint: none. Repair: none.;limit.log-full|8|never|", 1)],
      ["test_tables"], regen=True),
    M("t1-fixture-reaches-another-rule", "T1", "the only fixture of ml.model.malformed now claims another rule",
      [("tests/conformance/fixtures.py", 'bad("ml.model.malformed")', 'bad("ml.model.checksum")', 1)], ["test_tables"]),
    M("t1-log-record-without-fixture", "T1", "a log record is added to the table (schemas regenerated); no run writes it",
      [("tools/ml/spec.cho", ";end|event,seq,ok,exit,rule\";", ";end|event,seq,ok,exit,rule;skip|event,seq\";", 1)],
      ["test_tables"], regen=True),
    M("t1-skill-lacks-a-flag", "T1", "a stand-in binary whose `skill` text drops the --limit line", [], ["test_tables"], kind="wrapper:skill"),
    # ------------------------------------------------------------------ R1: refusals
    M("r1-exit-code-changed", "R1", "ml.input.empty exits 3 instead of 8", [("tools/ml/spec.cho", "ml.input.empty|8|", "ml.input.empty|3|", 1)],
      ["test_rules"]),
    M("r1-text-in-json-mode", "R1", "the answer is always rendered as text",
      [("tools/ml/main.cho", 'bytes.equal(cli.text(args, parsed, table, "format"), "text")', "true", 1)], ["test_rules"]),
    M("r1-missing-operand-not-refused", "R1", "`ml fit` with no file is not refused (no trap follows: the contract's accessors answer a harmless value)",
      [("tools/ml/main.cho", "} else if cli.operand_count(parsed) == 1 {", "} else if false {", 1)], ["test_rules"]),
    # ------------------------------------------------------------------ N1: no trap
    M("n1-verb-read-before-counting", "N1", "argv[1] is read before there is known to be one (spike F's bug)",
      [("tools/ml/main.cho", "    var verb = 0;\n    if cli.operand_count(parsed) == 0 {", "    var verb = verb_of(arg(args, 1));\n    if cli.operand_count(parsed) == 0 {", 1)],
      ["test_fuzz_args"]),
    M("n1-double-dot-check-removed", "N1", "a path with two dots in a row reaches the builtins",
      [("lib/core/pathcheck.cho", "        if int_of(p[i]) == '.' && int_of(p[i + 1]) == '.' {\n            return true;", "        if int_of(p[i]) == '.' && int_of(p[i + 1]) == '.' {\n            return false;", 1)],
      ["test_fuzz_args"]),
    # ------------------------------------------------------------------ L1: the log
    M("l1-column-name-logged", "L1", "the write record carries a column name where the hash goes",
      [("tools/ml/verbs.cho", "lg = logrec.write(heap, lg, len(data), digest);", "lg = logrec.write(heap, lg, len(data), mfile.name(mr, 1));", 1)],
      ["test_log"]),
    M("l1-cap-not-checked", "L1", "a log at its cap is appended to anyway", [("tools/ml/main.cho", "if lsize >= cap {", "if false {", 1)], ["test_log"]),
    # ------------------------------------------------------------------ M1: the model file
    M("m1-checksum-not-compared", "M1", "the SHA-256 trailer is not checked", [("tools/ml/mfile.cho", "if !checksum_ok(heap, data) {", "if false {", 1)],
      ["test_model_file"]),
    M("m1-section-length-not-checked", "M1", "a section length is checked for sign only, not against the file",
      [("tools/ml/mfile.cho", "if length < 0 || length > body - pos - 12 {", "if length < 0 {", 1)], ["test_model_file"]),
    M("m1-version-not-compared", "M1", "any version is accepted", [("tools/ml/mfile.cho", "if v != 1 {", "if false {", 1)], ["test_model_file"]),
    M("m1-trailing-bytes-accepted", "M1", "bytes between the last section and the trailer are not an error (the file is rebuilt with a valid checksum)",
      [("tools/ml/mfile.cho", "if pos != body {", "if false {", 1)], ["test_model_file"]),
    # ------------------------------------------------------------------ S1: the schema check
    M("s1-check-removed", "S1", "a different set of columns is accepted",
      [("tools/ml/verbs.cho", "if missing + mistyped + extras == 0 {", "if true {", 1)], ["test_schema"]),
    M("s1-repair-names-the-wrong-side", "S1", "the --drop repair lists the model's columns, not the extra ones",
      [("tools/ml/verbs.cho", "        if !csv.dropped(sh, j) && !in_model(mo, sh, j) {\n            let nm = csv.name(sh, j);",
        "        if !csv.dropped(sh, j) && in_model(mo, sh, j) {\n            let nm = csv.name(sh, j);", 1)], ["test_schema", "test_rules"]),
    # ------------------------------------------------------------------ O1: the oracle
    M("o1-mean-divides-by-n-minus-1", "O1", "the mean divides by n - 1",
      [("tools/ml/verbs.cho", "mfile.stat(mo, 3), mfile.stat(mo, 0));", "mfile.stat(mo, 3), mfile.stat(mo, 0) - 1);", 1)], ["test_oracle", "test_rules"]),
    M("o1-tie-goes-to-the-larger-label", "O1", "a tie is broken the other way",
      [("tools/ml/verbs.cho", "if mfile.stat(mo, 5) > mfile.stat(mo, 4) {", "if mfile.stat(mo, 5) >= mfile.stat(mo, 4) {", 1)], ["test_oracle"]),
    M("o1-half-rounds-down", "O1", "a half is rounded toward zero, not away", [("lib/core/exact.cho", "if 2 * rem >= n {", "if 2 * rem > n {", 1)], ["test_oracle"]),
    M("o1-mean-truncates", "O1", "the mean is truncated, not rounded", [("lib/core/exact.cho", "        q0 = q0 + 1;\n", "", 1)], ["test_oracle"]),
    M("o1-carry-dropped", "O1", "the remainder sum is not carried before it is scaled", [("lib/core/exact.cho", "let carry = sr / b;", "let carry = 0;", 1)],
      ["test_oracle"]),
    M("o1-sum-in-one-int", "O1", "the target is summed in one 64-bit int (a trap past 9.2e18)",
      [("lib/core/exact.cho", "    acc[at] = acc[at] + q;\n", "    acc[at] = acc[at] + q - q;\n", 1),
       ("lib/core/exact.cho", "acc[at + 1] = acc[at + 1] + (m - q * 1000000000);", "acc[at + 1] = acc[at + 1] + m;", 1)], ["test_oracle"]),
    M("o1-accuracy-counts-the-wrong-rows", "O1", "accuracy counts the rows that differ", [("tools/ml/csv.cho", "if m == pred {", "if m != pred {", 1)], ["test_oracle", "test_rules"]),
    M("o1-eval-ignores-the-cell-scale", "O1", "an eval cell is not shifted to the model's scale",
      [("tools/ml/csv.cho", "let y = m * exact.pow10(tscale - frac);", "let y = m * exact.pow10(tscale);", 1)], ["test_oracle"]),
    # ------------------------------------------------------------------ D1: determinism
    M("d1-model-holds-the-byte-count", "D1", "the stored row count is the CSV's byte count",
      [("tools/ml/verbs.cho", "rows: csv.rows(sh), cols: kept", "rows: csv.bytes_read(sh), cols: kept", 1)], ["test_determinism"]),
    M("d1-thread-count-changes-a-byte", "D1", "a stand-in with a --threads flag whose model bytes depend on the count",
      [("tools/ml/spec.cho", ";format||choice:json/text|none|json|", ";threads||nat|none|1|a stand-in;format||choice:json/text|none|json|", 1)],
      ["test_determinism"], kind="wrapper:threads", regen=True),
    # ------------------------------------------------------------------ A1: authority
    M("a1-reads-standard-input", "A1", "run() reads a byte of standard input (io_read)",
      [("tools/ml/main.cho", 'fs: &f Fs(""), io: &!i Io) -> [heap, args, fs_read(""), dir_read, dir_write, file_read, file_write, io_write, err_write] int {\n    let which',
        'fs: &f Fs(""), io: &!i Io) -> [heap, args, fs_read(""), dir_read, dir_write, file_read, file_write, io_write, io_read, err_write] int {\n    getchar(io);\n    let which', 1)],
      [], kind="authority"),
    M("a1-reads-the-clock", "A1", "main() reads the clock", [("tools/ml/main.cho", "    release(clock);\n", "    borrow clock as &c in {\n        clock_ms(c);\n    }\n    release(clock);\n", 1)],
      [], kind="authority"),
    M("a1-manifest-is-stale", "A1", "the committed manifest lacks a label the binary has",
      [("manifests/ml.authority.json", '"name": "heap",', '"name": "heapx",', 1)], ["test_authority"], kind="manifest"),
    # ------------------------------------------------------------------ H1: honesty
    M("h1-a-model-in-the-table-that-is-not-built", "H1", "logreg is a choice of --model, with no implementation",
      [("tools/ml/spec.cho", "model||choice:constant|", "model||choice:constant/logreg|", 1)], ["test_honesty", "test_tables"], regen=True),
    M("h1-an-exponent-is-read-as-a-prefix", "H1", "1e5 is read as 1", [("tools/ml/csvcell.cho", "return (4, 0, 0, 0);", "return (0, 1, 0, 1);", 1)],
      ["test_honesty"]),
    # ------------------------------------------------------------------ RD7: the constant model on floats (docs/reader.md section 5)
    M("rd7-mean-divides-by-n-plus-1", "RD7", "the float mean is the exact sum over n + 1",
      [("tools/ml/verbs.cho", "facc.mean_key(fw, 0, n)", "facc.mean_key(fw, 0, n + 1)", 1)], ["test_float"]),
    M("rd7-sum-is-not-exact", "RD7", "the float sum drops the low half of every double (a float accumulator's rounding, in effect)",
      [("tools/ml/csv.cho", "facc.add_bits(fv, 0, csvcell.bits_of_key(csvcell.key_of(kind, m, frac, key)));", "facc.add_bits(fv, 0, csvcell.bits_of_key(csvcell.key_of(kind, m, frac, key)) & (0 - 4096));", 1)], ["test_float"]),
    M("rd7-sse-carry", "RD7", "the square of a difference is carried in 25 bits, not 26",
      [("lib/core/fsq.cho", "        acc[352 + i] = v & mask();\n        carry = v >> 26;", "        acc[352 + i] = v & mask();\n        carry = v >> 25;", 1)], ["test_float"]),
    M("rd7-sse-rounds-half-up", "RD7", "the exact sum of squares is rounded half up, not half to even",
      [("lib/core/fsq.cho", "if guard == 1 && (sticky || mant & 1 == 1) {", "if guard == 1 {", 1)], ["test_float"]),
    M("rd7-sse-ignores-the-sticky-bit", "RD7", "a guard bit set and bits below it set is read as an exact tie",
      [("lib/core/fsq.cho", "if guard == 1 && (sticky || mant & 1 == 1) {", "if guard == 1 && mant & 1 == 1 {", 1)], ["test_float"]),
    M("rd7-rmse-from-two-roots", "RD7", "rmse is sqrt(sse) / sqrt(n)",
      [("tools/ml/verbs.cho", "let rmse = sqrt(sse / float_of(rows));", "let rmse = sqrt(sse) / sqrt(float_of(rows));", 1)], ["test_float"]),
    M("rd7-sse-against-a-neighbour-of-the-mean", "RD7", "the squared errors are taken against the double above the mean",
      [("tools/ml/csv.cho", "fsq.add_square(acc, csvcell.bits_of_key(csvcell.key_of(kind, m, frac, key)), pred);", "fsq.add_square(acc, csvcell.bits_of_key(csvcell.key_of(kind, m, frac, key)), pred + 1);", 1)], ["test_float"]),
    M("rd7-float-column-fits-an-int-model", "RD7", "an int model column accepts float data",
      [("tools/ml/verbs.cho", "    if mfile.type_of(mo, i) == 0 {\n        return k == 1;", "    if mfile.type_of(mo, i) == 0 {\n        return true;", 1)], ["test_float"]),
    M("rd7-int-model-bytes-change", "RD7", "a model of ints and decimals writes a seed of 1 (its bytes are no longer the CLI slice's)",
      [("tools/ml/mfile.cho", "    me = put64(heap, me, 0);\n    me = put32(heap, me, 0);", "    me = put64(heap, me, 1);\n    me = put32(heap, me, 0);", 1)], ["test_honesty"]),
    M("rd7-float-target-as-class-label", "RD7", "a float target is accepted as a class label",
      [("tools/ml/verbs.cho", "} else if o.task == 2 && csv.kind(sr, csv.tcol(sr)) != 1 {", "} else if o.task == 2 && csv.kind(sr, csv.tcol(sr)) == 0 {", 1)], ["test_float"]),
    M("rd7-float-model-kind-checked-as-int", "RD7", "a float regression model file is accepted with an int target column",
      [("tools/ml/mfile.cho", "if types[target] != 2 || s[1] >> 52 & 2047 == 2047 {", "if s[1] >> 52 & 2047 == 2047 {", 1)], ["test_model_file"]),
]


def copy_tree(dest):
    ignore = shutil.ignore_patterns(".git", "build", "__pycache__", "*.pyc")
    shutil.copytree(ROOT, dest, ignore=ignore, symlinks=True)
    # the numeric modules of cancho-table sit beside the tree, where cancho.toml's LOCAL-DEV sources name them
    table = pathlib.Path(os.environ.get("CANCHO_TABLE", str(ROOT.parent / "cancho-table-main")))
    link = dest.parent / "cancho-table-main"
    if not link.exists():
        link.symlink_to(table)
    deps = ROOT / "build" / "deps"
    if deps.is_dir():
        (dest / "build").mkdir(exist_ok=True)
        shutil.copytree(deps, dest / "build" / "deps")


def apply_edits(tree, m, strict=True):
    problems = []
    for item in m["edits"]:
        f, old, new, nth = item
        path = tree / f
        text = path.read_text()
        parts = text.split(old)
        if len(parts) - 1 < nth:
            problems.append("%s: %r found %d times, wanted occurrence %d" % (f, old[:60], len(parts) - 1, nth))
            continue
        # replace the nth occurrence
        head = old.join(parts[:nth])
        tail = old.join(parts[nth:])
        path.write_text(head + new + tail)
    return problems


def table_sources():
    t = pathlib.Path(os.environ.get("CANCHO_TABLE", str(ROOT.parent / "cancho-table-main"))) / "tools" / "table"
    return [str(t / (m + ".cho")) for m in ("pow5", "dec", "flt", "query", "facc", "reader")]


def build(tree):
    files = (sorted(glob.glob(str(tree / "lib/core/*.cho"))) + [str(tree / "lib/linalg/tensor.cho")] + sorted(glob.glob(str(tree / "build/deps/*.cho"))) +
             [str(tree / "generated/ml/built.cho")] + sorted(glob.glob(str(tree / "tools/ml/*.cho"))) + table_sources())
    out = tree / "build" / "ml"
    p = subprocess.run([compiler(), "build", "--std", *files, "-o", str(out)], capture_output=True, text=True, cwd=tree)
    return out if p.returncode == 0 else None, p.stdout + p.stderr


WRAPPERS = {
    # skill text without the --limit line
    "skill": '''#!/usr/bin/env python3
import subprocess, sys
real = sys.argv[0] + ".real"
p = subprocess.run([real] + sys.argv[1:], capture_output=True)
out = p.stdout
if sys.argv[1:2] == ["skill"]:
    out = b"".join(l for l in out.splitlines(True) if b"`--limit" not in l)
sys.stdout.buffer.write(out)
sys.stderr.buffer.write(p.stderr)
sys.exit(p.returncode)
''',
    # a --threads flag: removed before the real binary sees it; with a count other than 1 the written model differs by a byte
    "threads": '''#!/usr/bin/env python3
import subprocess, sys
real = sys.argv[0] + ".real"
args = sys.argv[1:]
threads = 1
if "--threads" in args:
    i = args.index("--threads")
    threads = int(args[i + 1])
    del args[i:i + 2]
p = subprocess.run([real] + args, capture_output=True)
if p.returncode == 0 and threads != 1 and "--output" in args and "--root" in args:
    path = args[args.index("--root") + 1] + "/" + args[args.index("--output") + 1]
    data = bytearray(open(path, "rb").read())
    data[-1] ^= 1
    open(path, "wb").write(bytes(data))
sys.stdout.buffer.write(p.stdout)
sys.stderr.buffer.write(p.stderr)
sys.exit(p.returncode)
''',
}


def make_wrapper(tree, name):
    real = tree / "build" / "ml"
    shutil.move(str(real), str(real) + ".real")
    real.write_text(WRAPPERS[name])
    real.chmod(0o755)


def run_tests(tree, targets):
    env = dict(os.environ, ML_BIN=str(tree / "build" / "ml"), PYTHONDONTWRITEBYTECODE="1", CANCHO=compiler())
    p = subprocess.run([sys.executable, "-m", "unittest", *targets], capture_output=True, text=True, cwd=tree / "tests" / "conformance", env=env)
    return p.returncode, p.stdout + p.stderr


def failing_tests(output):
    names = []
    for line in output.splitlines():
        if line.startswith("FAIL: ") or line.startswith("ERROR: "):
            names.append(line.split(": ", 1)[1].split(" ")[0])
    return names


def run_one(m):
    with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        copy_tree(tree)
        problems = apply_edits(tree, m)
        if problems:
            return "INVALID", "; ".join(problems)
        if m["opt"].get("regen"):
            r = subprocess.run([sys.executable, str(tree / "scripts" / "schemas.py")], capture_output=True, text=True, cwd=tree)
            if r.returncode != 0:
                return "INVALID", "schemas.py: " + r.stdout + r.stderr
        kind = m["opt"].get("kind", "")
        if kind == "authority":
            # the manifest is derived from the mutated sources and compared with the committed one and the ceiling
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "manifest.py"), "--tree", str(tree)], capture_output=True, text=True)
            return ("KILLED", "manifest.py --tree: " + " | ".join(l for l in r.stdout.splitlines() if l.startswith("FAIL"))[:300]) if r.returncode != 0 else ("SURVIVED", "")
        binary, log = build(tree)
        if binary is None:
            return "INVALID", "build failed: " + log[:400]
        if kind.startswith("wrapper:"):
            make_wrapper(tree, kind.split(":", 1)[1])
        if kind == "manifest":
            r = subprocess.run([sys.executable, str(tree / "scripts" / "manifest.py"), "--check"], capture_output=True, text=True, cwd=tree)
            if r.returncode != 0:
                return "KILLED", "manifest.py --check: " + " | ".join(l for l in r.stdout.splitlines() if l.startswith("FAIL"))[:300]
        code, out = run_tests(tree, m["tests"])
        if code != 0:
            return "KILLED", "tests: " + ", ".join(failing_tests(out)[:4]) + ("" if failing_tests(out) else out.strip().splitlines()[-1][:200])
        return "SURVIVED", ""


def baseline():
    with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        copy_tree(tree)
        binary, log = build(tree)
        if binary is None:
            return "INVALID", "build failed: " + log[:400]
        code, out = run_tests(tree, ["discover", "-p", "test_*.py"])
        return ("PASSED", "") if code == 0 else ("FAILED", out[-600:])


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        for m in MUTANTS:
            print("%-44s %s  %s" % (m["id"], m["gate"], m["what"]))
        return 0
    if "--check" in argv:
        bad = 0
        with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
            tree = pathlib.Path(td) / "tree"
            copy_tree(tree)
            for m in MUTANTS:
                fresh = tree
                problems = apply_edits(tree, m)
                if problems:
                    print("INVALID %s: %s" % (m["id"], "; ".join(problems)))
                    bad += 1
                # undo by recopying only the files edited
                for item in m["edits"]:
                    shutil.copy(ROOT / item[0], tree / item[0])
        print("%d mutants, %d with an edit that does not apply" % (len(MUTANTS), bad))
        return 1 if bad else 0
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    results = []
    for m in MUTANTS:
        if only and m["id"] != only:
            continue
        verdict, detail = run_one(m)
        results.append((m, verdict, detail))
        print("%-9s %-44s %s  %s" % (verdict, m["id"], m["gate"], detail), flush=True)
    if not only:
        v, d = baseline()
        print("%-9s %-44s %s  %s" % (v, "baseline (the unchanged tree)", "-", d), flush=True)
    survivors = [(m, d) for m, v, d in results if v == "SURVIVED"]
    invalid = [m["id"] for m, v, d in results if v == "INVALID"]
    killed = sum(1 for m, v, d in results if v == "KILLED")
    print("\n%d mutants: %d killed, %d survived, %d invalid" % (len(results), killed, len(survivors), len(invalid)))
    for m, d in survivors:
        print("SURVIVOR %s: %s" % (m["id"], m["opt"].get("why", "NO REASON GIVEN")))
    unexplained = [m["id"] for m, d in survivors if not m["opt"].get("why")]
    return 1 if (unexplained or invalid) else 0


if __name__ == "__main__":
    sys.exit(main())

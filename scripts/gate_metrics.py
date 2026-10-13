#!/usr/bin/env python3
"""Gates T1 and F1 of docs/metrics.md: the rule tags (table, design.md, a fixture for every tag through the compiled driver) and the
fuzz of the driver.

    python3 scripts/gate_metrics.py --tags
    python3 scripts/gate_metrics.py --fuzz
Exit 1 on any failure. METRICS_CHECK names the driver."""
import os, re, subprocess, sys, pathlib, random

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = os.environ.get("METRICS_CHECK", str(ROOT / "build" / "metrics-check"))
NAN = "7ff8000000000000"
ONE_AND_HALF = "3ff8000000000000"
HALF = "3fe0000000000000"
FAILS = []


def fail(msg):
    FAILS.append(msg)
    print("FAIL", msg)


def run(*args):
    r = subprocess.run([BIN] + [str(a) for a in args], capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr


def table():
    src = (ROOT / "lib" / "metrics" / "rules.cho").read_text()
    m = re.search(r"pub fn table\(\) -> \[\] &static \[byte\] \{\s*return \"(.*?)\";", src, re.S)
    rows = []
    for ent in m.group(1).split(";"):
        tag, exit_, rep, summary = ent.split("|", 3)
        rows.append((tag, int(exit_), rep, summary))
    return rows, src


FIXTURES = {
    "ml.input.empty": ("split", "kfold", 0, 2, 1, 0, 100),
    "ml.input.too-many-rows": ("split", "kfold", 1000, 2, 1, 0, 999),
    "ml.input.non-finite": ("metric", "auc", "L:1,0,1", "H:%s,3fe0000000000000,3fe0000000000000" % NAN, 1, 100, 1),
    "ml.label.single-class": ("metric", "auc", "L:1,1,1", "H:%s" % ",".join([HALF] * 3), 1, 100, 1),
    "ml.label.not-binary": ("metric", "f1", "L:0,1,2", "L:0,1,1", 1, 100, 1),
    "ml.metric.undefined": ("metric", "r2", "H:%s" % ",".join([HALF] * 4), "H:%s" % ",".join([HALF] * 4), 0, 100, 1),
    "ml.param.out-of-range": ("split", "kfold", 10, 1, 1, 0, 100),
    "ml.metric.length-mismatch": ("metric", "accuracy", "L:1,0", "L:1", 1, 100, 1),
    "ml.metric.bad-probability": ("metric", "logloss", "L:1,0", "H:%s,%s" % (ONE_AND_HALF, HALF), 1, 100, 1),
    "ml.split.empty-fold": ("split", "kfold", 3, 5, 1, 0, 100),
    "ml.split.stratum-too-small": ("split", "strat", "L:0,0,0,1", 2, 1, 0, 100),
}
# more routes to a tag, to show that it is the rule and not one input that raises it
MORE = [
    ("ml.metric.undefined", ("metric", "precision", "L:0,1", "L:0,0", 1, 100, 1)),
    ("ml.metric.undefined", ("metric", "recall", "L:0,0", "L:0,1", 1, 100, 1)),
    ("ml.metric.undefined", ("metric", "f1", "L:0,0,0", "L:0,0,0", 1, 100, 1)),
    ("ml.metric.undefined", ("metric", "rmse", "H:", "H:", 0, 100, 1)),
    ("ml.metric.undefined", ("metric", "accuracy", "L:", "L:", 0, 100, 1)),
    ("ml.metric.undefined", ("metric", "auc", "L:", "H:", 1, 100, 1)),
    ("ml.metric.bad-probability", ("metric", "logloss", "L:1", "H:3ff0000000000001", 1, 100, 1)),
    ("ml.metric.bad-probability", ("metric", "logloss", "L:1", "H:bff0000000000000", 1, 100, 1)),
    ("ml.metric.bad-probability", ("metric", "logloss", "L:1", "H:8000000000000001", 1, 100, 1)),
    ("ml.metric.undefined", ("metric", "r2", "H:3fb999999999999a,3fb999999999999a,3fb999999999999a", "H:%s" % ",".join([HALF] * 3), 0, 100, 1)),
    ("ml.input.non-finite", ("metric", "rmse", "H:%s" % NAN, "H:%s" % HALF, 0, 100, 1)),
    ("ml.input.non-finite", ("metric", "rmse", "H:%s" % HALF, "H:%s" % NAN, 0, 100, 1)),
    ("ml.input.non-finite", ("metric", "r2", "H:%s,%s" % (HALF, "3ff0000000000000"), "H:7ff0000000000000,%s" % HALF, 0, 100, 1)),
    ("ml.input.non-finite", ("metric", "auc", "L:1,0", "H:fff0000000000000,%s" % HALF, 1, 100, 1)),
    ("ml.input.non-finite", ("metric", "mae", "H:7ff0000000000000", "H:%s" % HALF, 0, 100, 1)),
    ("ml.input.non-finite", ("metric", "logloss", "L:1", "H:fff0000000000000", 1, 100, 1)),
    ("ml.label.single-class", ("split", "strat", "L:3,3,3,3", 2, 1, 0, 100)),
    ("ml.label.single-class", ("split", "strat", "L:5", 2, 1, 0, 100)),
    ("ml.label.not-binary", ("split", "strat", "L:1,2,0,1,2,0", 2, 1, 0, 100)),
    ("ml.label.not-binary", ("metric", "auc", "L:1,2,0,1", "H:%s" % ",".join([HALF] * 4), 1, 100, 1)),
    ("ml.label.not-binary", ("metric", "f1", "L:1,2,0,1", "L:1,1,1,1", 1, 100, 1)),
    ("ml.label.not-binary", ("metric", "recall", "L:1,2,2,2", "L:1,2,0,2", 1, 100, 1)),
    ("ml.metric.length-mismatch", ("metric", "accuracy", "L:1,0,1", "L:1,0,1,0", 1, 100, 1)),
    ("ml.metric.length-mismatch", ("metric", "mae", "H:%s,%s" % (HALF, HALF), "H:%s" % HALF, 0, 100, 1)),
    ("ml.metric.undefined", ("metric", "r2", "H:0000000000000000,1eb67e9c127b6e74", "H:%s" % ",".join(["5f138d352e5096af"] * 2), 0, 100, 1)),
    ("ml.label.single-class", ("metric", "auc", "L:0,0", "H:%s" % ",".join([HALF] * 2), 1, 100, 1)),
    ("ml.label.not-binary", ("split", "strat", "L:0,1,2,0,1,2", 2, 1, 0, 100)),
    ("ml.label.not-binary", ("metric", "logloss", "L:0,1,2", "H:%s" % ",".join([HALF] * 3), 1, 100, 1)),
    ("ml.input.too-many-rows", ("metric", "auc", "L:1,0,1", "H:%s" % ",".join([HALF] * 3), 1, 2, 1)),
    ("ml.param.out-of-range", ("split", "train_test", 10, 0, 1, 0, 100)),
    ("ml.param.out-of-range", ("split", "kfold", 10, 2, 1, 0 - 1, 100)),
    ("ml.param.out-of-range", ("metric", "accuracy", "L:1", "L:1", 1, 0, 1)),
    ("ml.split.empty-fold", ("split", "train_test", 5, 5, 1, 0, 100)),
    ("ml.split.empty-fold", ("split", "train_test", 1, 1, 1, 0, 100)),
]


def tags_gate():
    rows, src = table()
    # 1. the table and the code agree: same tags, same order as tag(code), exit codes as exit_code(code)
    tag_fn = re.findall(r"if code == (\d+) \{\s*return \"([a-z.\-]+)\";", src.split("pub fn tag")[1].split("pub fn exit_code")[0])
    if [t for _, t in tag_fn] != [r[0] for r in rows] or [int(c) for c, _ in tag_fn] != list(range(201, 201 + len(rows))):   # the band of metrics.rules (design.md section 8.1)
        fail("table() and tag() disagree")
    if len(set(r[0] for r in rows)) != len(rows):
        fail("a tag is registered twice")
    for tag, exit_, rep, summary in rows:
        want = 2 if tag == "ml.param.out-of-range" else 8
        if exit_ != want:
            fail("%s: exit %d in the table, expected %d" % (tag, exit_, want))
        if not re.match(r"^([a-z]+\.)+[a-z\-]+$", tag):
            fail("%s is not <area>.<name>" % tag)
        if "Hint:" not in summary or "Repair:" not in summary:
            fail("%s: the summary does not end with Hint and Repair as in tools/ml/spec.cho" % tag)
    # 2. design.md section 8: a tag that is there has the same exit code; one that is not is marked new in docs/metrics.md
    docs = pathlib.Path(os.environ.get("DOCS_ROOT", str(ROOT))) / "docs"     # the mutant runner copies lib/, tests/, tools/, scripts/ only
    design = (docs / "design.md").read_text()
    metrics_md = (docs / "metrics.md").read_text()
    for tag, exit_, _, _ in rows:
        found = None
        for line in design.splitlines():
            if line.startswith("|") and ("`%s`" % tag in line.split("|")[1]):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) > 1 and cells[1].isdigit():
                    found = int(cells[1])
        row = [l for l in metrics_md.splitlines() if l.startswith("| `%s` |" % tag)]
        if found is not None:
            if found != exit_:
                fail("%s: exit %d here, %d in design.md section 8" % (tag, exit_, found))
            if row and "**new**" in row[0]:
                fail("%s is in design.md section 8 but docs/metrics.md calls it new" % tag)
        else:
            if not row or "**new**" not in row[0]:
                fail("%s is not in design.md section 8 and docs/metrics.md does not mark it new" % tag)
    # 3. every tag is reached by a fixture through the driver, with its exit code and nothing else
    reached = set()
    todo = [(t, a) for t, a in FIXTURES.items()] + MORE
    for tag, args in todo:
        code, out, err = run(*args)
        want_exit = dict((r[0], r[1]) for r in rows).get(tag)
        if ("refused %s" % tag) not in out.split("\n") or code != want_exit or err.strip():
            fail("fixture %s %s -> exit %d, %r" % (tag, args[:3], code, out.strip()))
        else:
            reached.add(tag)
    for tag in [r[0] for r in rows]:
        if tag not in reached:
            fail("no fixture reaches " + tag)
    # 4. the driver says nothing else on a refusal (no value line, no partial output)
    code, out, err = run(*FIXTURES["ml.metric.undefined"])
    if "value" in out:
        fail("a refusal printed a value")
    print("tags: %d registered, %d reached by %d fixtures, exit codes and design.md section 8 consistent" % (len(rows), len(reached), len(todo)))


def fuzz_gate():
    rnd = random.Random(20261013)
    nums = [0, 1, 2, 3, 5, 10, 63, 64, 100, 255, 256, 257, 1000, 4097, 2 ** 20, 2 ** 31, 2 ** 31 - 1, 2 ** 62, 2 ** 63 - 1, -1, -2, -(2 ** 31), -(2 ** 63)]
    hexes = ["0000000000000000", "8000000000000000", "3ff0000000000000", "bff0000000000000", "7ff0000000000000", "fff0000000000000", NAN,
             "0000000000000001", "7fefffffffffffff", "ffefffffffffffff", HALF, "3c90000000000000", "3fefffffffffffff", ONE_AND_HALF, "bfd0000000000000"]
    names_i = ["accuracy", "precision", "recall", "f1"]
    names_f = ["rmse", "mae", "r2"]
    names_s = ["auc", "logloss"]

    def ints_spec():
        k = rnd.randrange(0, 12)
        pool = [0, 1, 2, -1, 2 ** 63 - 1, -(2 ** 63), 7]
        if rnd.random() < 0.2:
            return "GL:%d:%d:%d" % (rnd.choice([0, 1, 5, 300, 20000]), rnd.randrange(0, 100), rnd.choice([0, 1, 500, 999, 1000]))
        if rnd.random() < 0.7:
            pool = [0, 1]
        return "L:" + ",".join(str(rnd.choice(pool)) for _ in range(k))

    def floats_spec():
        if rnd.random() < 0.3:
            return "GF:%d:%d:%d:%d" % (rnd.choice([0, 1, 2, 3, 4, 5, 6, 10, 99]), rnd.choice([0, 1, 7, 300, 20000]), rnd.randrange(0, 50), rnd.randrange(0, 50))
        k = rnd.randrange(0, 12)
        return "H:" + ",".join(rnd.choice(hexes) for _ in range(k))

    bad_exit = 0
    seen_ok = seen_refused = 0
    runs = 0
    for _ in range(400):
        kind = rnd.randrange(0, 6)
        if kind == 0:
            args = ("metric", rnd.choice(names_i), ints_spec(), ints_spec(), rnd.choice(nums), rnd.choice([100, 5, 0, -1, 2 ** 40, 10 ** 7]), 1)
        elif kind == 1:
            args = ("metric", rnd.choice(names_f), floats_spec(), floats_spec(), 0, rnd.choice([100, 5, 0, -1, 2 ** 40, 10 ** 7]), 1)
        elif kind == 2:
            args = ("metric", rnd.choice(names_s), ints_spec(), floats_spec(), rnd.choice([0, 1, 2]), rnd.choice([100, 5, 0, -1, 2 ** 40, 10 ** 7]), 1)
        elif kind == 3:
            args = ("split", rnd.choice(["kfold", "train_test"]), rnd.choice(nums), rnd.choice(nums), rnd.choice(nums), rnd.choice([0, 1, 5, -1, 2 ** 62]), rnd.choice([100, 5, 0, -1, 2 ** 40, 10 ** 7, 2 ** 30, 2 ** 30 + 1]))
        elif kind == 4:
            args = ("split", "strat", ints_spec(), rnd.choice(nums), rnd.choice(nums), rnd.choice([0, 1, 5, -1]), rnd.choice([100, 5, 0, -1, 2 ** 30 + 1, 10 ** 7]))
        else:
            args = ("ratio", rnd.choice(nums), rnd.choice(nums))
        runs += 1
        if kind == 3 and isinstance(args[2], int) and 20000000 < args[2] and False:
            continue
        code, out, err = run(*args)
        if code not in (0, 2, 8):
            bad_exit += 1
            fail("exit %d for %s: %r %r" % (code, args, out[-200:], err[-200:]))
            continue
        if code == 0:
            seen_ok += 1
        else:
            seen_refused += 1
            m = re.search(r"refused (\S+)", out)
            if not m:
                fail("exit %d without a refused line for %s" % (code, args))
    if seen_ok < 40 or seen_refused < 40:
        fail("the fuzz did not reach both outcomes (ok %d, refused %d)" % (seen_ok, seen_refused))
    print("fuzz: %d driver runs, %d ok, %d refused with a tag, no exit outside 0/2/8, no signal" % (runs, seen_ok, seen_refused))


if __name__ == "__main__":
    if not os.path.exists(BIN):
        print("no driver at " + BIN)
        sys.exit(1)
    a = sys.argv[1:] or ["--tags", "--fuzz"]
    if "--tags" in a:
        tags_gate()
    if "--fuzz" in a:
        fuzz_gate()
    if FAILS:
        print("%d FAILURES" % len(FAILS))
        sys.exit(1)
    print("gates T1/F1 pass")

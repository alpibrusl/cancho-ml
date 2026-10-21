#!/usr/bin/env python3
"""The oracle of the wiring of the ranking objective (docs/ranking.md gate RK4): the cases of oracle_wire_boost.py turned into ranking files (a group column, a relevance target of
integers) and compared with `ml fit --model boost --task ranking --group qid` byte for byte, and `ml predict` against the oracle's scores.

    python3 scripts/oracle_wire_rank.py --suite [N] [--weighted]     N seeded cases (default 24) against build/ml, 30 to 700 rows
Exit 1 on any mismatch. ML_BIN names the binary."""

import pathlib
import random
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_rank as orank  # noqa: E402
import oracle_wire as ow  # noqa: E402
import oracle_wire_boost as owb  # noqa: E402
import oracle_wire_tree as owt  # noqa: E402
import json  # noqa: E402

COVER = {"cases": 0, "groups": 0, "rows": 0}


def gen_case(seed, rows=None, weighted=False):
    """a regression case of the boosting oracle with its target turned into a relevance (an integer grade by the rank of the value, 0 to a ceiling of 1, 3 or 30, or all 0) and a
    group of every row (blocks of 1 to 60 rows, labels as text, the rows shuffled in 60 percent of the cases)"""
    case = owb.gen_case(seed, "regression", rows=rows, weighted=weighted)
    rng = random.Random(seed * 17 + 3)
    n = len(case.rows)
    ys = [float(r[-1]) for r in case.rows]
    order = sorted(range(n), key=lambda i: (ys[i], i))
    ceiling = rng.choice([1, 3, 3, 30, 0])
    rel = [0] * n
    for pos, i in enumerate(order):
        rel[i] = min(ceiling, (pos * (ceiling + 1)) // n)
    for i, r in enumerate(case.rows):
        r[-1] = str(rel[i])
    # groups: consecutive blocks of the file, then (maybe) the rows shuffled, so a group's rows need not be adjacent
    sizes = []
    left = n
    maxg = rng.choice([3, 8, 20, 60])
    while left > 0:
        s = min(left, rng.randint(1, maxg))
        sizes.append(s)
        left -= s
    labels = []
    style = rng.choice(["q%d", "%d", "g%03d"])
    for g, s in enumerate(sizes):
        labels += [style % g] * s
    perm = list(range(n))
    if rng.random() < 0.6:
        rng.shuffle(perm)
    case.rows = [case.rows[i] for i in perm]
    if weighted and case.weights is not None:
        case.weights = [case.weights[i] for i in perm]
    case.groups = [labels[i] for i in perm]
    case.task = "ranking"
    case.split = None
    case.positive = None
    case.hp.pop("early-stopping", None)
    COVER["cases"] += 1
    COVER["groups"] += len(sizes)
    COVER["rows"] += n
    return case


def check_eval(case, tmp, k, gain):
    """`ml eval --group qid` on the file against the metrics of the oracle over the scores of the oracle's model (all rows; docs/ranking.md section 3)"""
    expect, ctx = owb.fit_oracle(case)
    if ctx is None:
        return []
    names, rows = case.names, case.rows
    t = names.index(case.target)
    feat = [j for j in range(len(names)) if j != t]
    X = [[ow.to_float(r[j]) for j in feat] for r in rows]
    scores = owb.predict_oracle(case, ctx, X)
    rel = [int(r[t]) for r in rows]
    want = orank.evaluate(case.groups, rel, scores, k, gain)
    extra = ["--drop", "wt"] if case.weights is not None else []
    rc, out = ow.ml("--root", tmp, "eval", "--model-file", "m.cml", "--group", "qid", "--rank-k", str(k), "--gain", gain, *extra, "d.csv")
    doc = json.loads(out)
    if not doc.get("ok"):
        return ["eval refused: %s" % out[:300]]
    got = doc["data"]["metrics"]
    problems = []
    for key, name in (("ndcg", "ndcg"), ("map", "map"), ("mrr", "mrr"), ("precision", "prec"), ("recall", "recall")):
        w = want[name]
        if w is None:
            if got.get(key, "missing") is not None:
                problems.append("eval %s: expected null, got %r" % (key, got.get(key)))
        elif got.get(key) != ow.put_float(w):
            problems.append("eval %s (k=%d, %s): %r vs %r" % (key, k, gain, got.get(key), ow.put_float(w)))
    if got.get("groups") != want["groups"] or got.get("groups_scored") != want["scored"]:
        problems.append("eval groups %r/%r vs %r/%r" % (got.get("groups"), got.get("groups_scored"), want["groups"], want["scored"]))
    return problems


def main():
    argv = sys.argv[1:]
    n = 24
    if "--suite" in argv:
        at = argv.index("--suite") + 1
        if len(argv) > at and argv[at].isdigit():
            n = int(argv[at])
    weighted = "--weighted" in argv
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for seed in range(1, n + 1):
            case = gen_case(seed * 7919 + 211, weighted=weighted)
            pr = owb.run_case(case, tmp, threads=1 + seed % 4)
            if not pr and (tmp / "m.cml").exists():
                for k, gain in ((10, "exp"), (3, "linear"), (0, "exp"), (1, "exp")):
                    pr += check_eval(case, tmp, k, gain)
            (tmp / "m.cml").unlink(missing_ok=True)
            if pr:
                bad += 1
                print("MISMATCH rank seed %d: %s" % (seed, "; ".join(pr)))
    print("oracle_wire_rank covered: %s, not judged %d" % (", ".join("%s %d" % (k, v) for k, v in COVER.items()), len(owb.UNJUDGED)))
    print("oracle_wire_rank: %d cases, %s" % (n, "all agree byte for byte" if not bad else "%d mismatches" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

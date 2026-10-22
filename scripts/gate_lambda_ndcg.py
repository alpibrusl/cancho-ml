#!/usr/bin/env python3
"""Gate LR3 of docs/ranking.md: the held-out NDCG of `--task ranking` and `--task lambdarank` on seeded synthetic files. Nothing is asserted about which is better; the numbers are printed.

    python3 scripts/gate_lambda_ndcg.py [SEEDS]      default 10 seeds; ML_BIN names the binary
Each seed makes a train file (120 groups) and a test file (60 groups) of 5 to 40 rows a group, 6 features, a relevance 0 to 4 from a nonlinear score and noise. Both objectives use the
same hyperparameters (100 rounds, depth 4, min-leaf 10, learning rate 0.1) on the same file and are judged on the test file by `ml eval --group qid` (ndcg at 10, mean over groups)."""
import json
import os
import pathlib
import random
import subprocess
import sys
import tempfile

ML = os.environ.get("ML_BIN", str(pathlib.Path(__file__).resolve().parent.parent / "build" / "ml"))


def make(path, seed, groups):
    r = random.Random(seed)
    with open(path, "w") as f:
        f.write("qid,x0,x1,x2,x3,x4,x5,rel\n")
        for g in range(groups):
            qf = r.gauss(0, 1)
            for _ in range(r.randint(5, 40)):
                x = [r.gauss(0, 1) for _ in range(6)]
                z = 1.2 * x[0] + 0.8 * x[1] * x[2] + (0.9 if x[3] > 0.5 else 0.0) + 0.3 * qf + r.gauss(0, 0.7)
                rel = max(0, min(4, int(z + 1.5)))
                f.write("q%d,%s,%d\n" % (g, ",".join("%.4f" % v for v in x), rel))


def ml(*a):
    p = subprocess.run([ML, *a], capture_output=True, stdin=subprocess.DEVNULL)
    return json.loads(p.stdout)


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for s in range(1, seeds + 1):
            make(td + "/tr.csv", s, 120)
            make(td + "/te.csv", s + 1000, 60)
            res = []
            for task in ("ranking", "lambdarank"):
                d = ml("--root", td, "fit", "--model", "boost", "--task", task, "--group", "qid", "--target", "rel", "--rounds", "100", "--max-depth", "4", "--min-leaf", "10", "--output", "m.cml", "--overwrite", "tr.csv")
                assert d["ok"], d
                e = ml("--root", td, "eval", "--model-file", "m.cml", "--group", "qid", "te.csv")
                assert e["ok"], e
                res.append(float(e["data"]["metrics"]["ndcg"]))
            rows.append(res)
            print("seed %2d  ranking %.6f  lambdarank %.6f  diff %+.6f" % (s, res[0], res[1], res[1] - res[0]))
    mp = sum(r[0] for r in rows) / len(rows)
    ml_ = sum(r[1] for r in rows) / len(rows)
    wins = sum(1 for r in rows if r[1] > r[0])
    print("mean ndcg@10 over %d seeds: ranking %.6f, lambdarank %.6f (lambdarank higher on %d of %d)" % (len(rows), mp, ml_, wins, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

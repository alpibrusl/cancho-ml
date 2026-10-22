#!/usr/bin/env python3
"""The cost of the pair loop of docs/ranking.md section 4 (the measurement the two ceilings of section 2 wait for).

    python3 scripts/bench_rank.py [N]       N timed runs of each cell (default 3), the minimum reported; ML_BIN names the binary
For each group size g, a file of 20,000 rows in groups of g rows (relevance 0 to 4, 6 features), fitted with --threads 1, 20 rounds, depth 4, min-leaf 10, for both objectives and for
a regression on the same rows (the cost of everything but the pairs). The pairs of a round are about rows * g * 0.4 (the share of ordered pairs with different relevance is measured
from the file). Prints the machine, the load average before the run, and the minimum of N wall-clock seconds per cell. Nothing is asserted."""
import os
import pathlib
import random
import subprocess
import sys
import tempfile
import time

ML = os.environ.get("ML_BIN", str(pathlib.Path(__file__).resolve().parent.parent / "build" / "ml"))
ROWS = 20000


def make(path, g, seed=1):
    r = random.Random(seed)
    pairs = 0
    with open(path, "w") as f:
        f.write("qid,x0,x1,x2,x3,x4,x5,rel,y\n")
        done = 0
        q = 0
        while done < ROWS:
            size = min(g, ROWS - done)
            rels = []
            for _ in range(size):
                x = [r.gauss(0, 1) for _ in range(6)]
                z = 1.2 * x[0] + 0.8 * x[1] * x[2] + r.gauss(0, 0.7)
                rel = max(0, min(4, int(z + 1.5)))
                rels.append(rel)
                f.write("q%d,%s,%d,%.4f\n" % (q, ",".join("%.4f" % v for v in x), rel, z))
            cnt = [rels.count(v) for v in range(5)]
            pairs += (size * size - sum(c * c for c in cnt)) // 2
            done += size
            q += 1
    return pairs


def run(td, *args):
    t = time.perf_counter()
    p = subprocess.run([ML, "--root", td, "fit", "--threads", "1", "--rounds", "20", "--max-depth", "4", "--min-leaf", "10", "--output", "m.cml", "--overwrite", *args], capture_output=True, stdin=subprocess.DEVNULL)
    dt = time.perf_counter() - t
    return dt, p.returncode, p.stdout[:200]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print("machine:", " ".join(os.uname()[:1] + os.uname()[2:3]), "cpus", os.cpu_count(), "load before", os.getloadavg(), "N", n)
    print("%6s %12s %10s %10s %10s" % ("g", "pairs/round", "regress s", "ranking s", "lambda s"))
    with tempfile.TemporaryDirectory() as td:
        for g in (10, 50, 100, 250, 500, 1000, 2000, 4096):
            pairs = make(td + "/d.csv", g)
            cells = []
            for args in (("--model", "boost", "--target", "y", "--drop", "qid,rel", "d.csv"),
                         ("--model", "boost", "--task", "ranking", "--group", "qid", "--target", "rel", "--drop", "y", "d.csv"),
                         ("--model", "boost", "--task", "lambdarank", "--group", "qid", "--target", "rel", "--drop", "y", "d.csv")):
                best, bad = None, None
                for _ in range(n):
                    dt, rc, out = run(td, *args)
                    if rc != 0:
                        bad = (rc, out)
                        break
                    best = dt if best is None else min(best, dt)
                cells.append("refused" if bad else "%.2f" % best)
                if bad:
                    cells[-1] = "rc%d" % bad[0]
            print("%6d %12d %10s %10s %10s" % (g, pairs, *cells), flush=True)


if __name__ == "__main__":
    main()

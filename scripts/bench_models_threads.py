#!/usr/bin/env python3
"""Thread scaling of the fits of lib/models and the check that every output is the same at every count (docs/models-linear.md 6.4).

    python3 scripts/bench_models_threads.py [--sets n100000,n1000000] [-N 5] [--threads 1,2,4,8,16] [--only logreg,linreg,kmeans]

For each model and thread count: the minimum of N runs of `reps` fits (the driver's own clock_ms), and whether the `params`, `diag`
and prediction lines of the run are byte-identical to the one-thread run (the raw sums and loss lines included). Min of N
interleaved over the thread counts; the load average at the start and the end of each table."""
import os, subprocess, sys, tempfile, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
BIN = os.environ.get("MODELS_CHECK", str(HERE.parent / "build" / "models-check"))
DATA = pathlib.Path(os.environ.get("ML_DATA", "/Users/alfonso/Workspace/alpibrusl/lexsys/ml-data"))


def run(args, threads, reps):
    r = subprocess.run([BIN] + args + ["threads=%d" % threads, "reps=%d" % reps], capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        k, _, v = line.partition(" ")
        out[k] = v
    return out


def main():
    a = sys.argv[1:]
    sets = a[a.index("--sets") + 1].split(",") if "--sets" in a else ["n100000", "n1000000"]
    N = int(a[a.index("-N") + 1]) if "-N" in a else 5
    threads = [int(t) for t in a[a.index("--threads") + 1].split(",")] if "--threads" in a else [1, 2, 4, 8, 16]
    only = a[a.index("--only") + 1].split(",") if "--only" in a else None
    for s in sets:
        n = int(s[1:])
        d = 20
        xf, yf = str(DATA / s / "x.bin"), str(DATA / s / "y.bin")
        X = np.fromfile(xf, dtype="<f8").reshape(n, d)
        rng = np.random.default_rng(7)
        ylin = X @ rng.standard_normal(d) + 3.0 + 0.5 * rng.standard_normal(n)
        td = tempfile.mkdtemp()
        yl = os.path.join(td, "ylin.bin")
        ylin.astype("<f8").tofile(yl)
        jobs = {"logreg": ["logreg", xf, yf, "n=%d" % n, "d=%d" % d, "iters=6"],
                "linreg": ["linreg", xf, yl, "n=%d" % n, "d=%d" % d],
                "kmeans": ["kmeans", xf, yf, "n=%d" % n, "d=%d" % d, "k=8", "iters=10", "seed=3", "init=1"]}
        for name, args in jobs.items():
            if only and name not in only:
                continue
            reps = {"logreg": (5, 1), "linreg": (20, 3), "kmeans": (3, 1)}[name][0 if n <= 100000 else 1]
            la0 = os.getloadavg()
            best = {t: 1e30 for t in threads}
            same = {t: True for t in threads}
            ref = None
            for _ in range(N):
                for t in threads:
                    o = run(args, t, reps)
                    ms = int(o["ms"])
                    best[t] = min(best[t], ms / reps)
                    sig = tuple(o.get(k) for k in ("params", "diag", "pred", "proba", "labels", "transform"))
                    if ref is None:
                        ref = sig
                    same[t] &= (sig == ref)
            la1 = os.getloadavg()
            print("%s %s n=%d (load start %.2f end %.2f): " % (name, s, n, la0[0], la1[0]) + "  ".join("t=%d %6.0f ms (x%.2f)%s" % (t, best[t], best[threads[0]] / best[t], "" if same[t] else " DIFFERENT") for t in threads), flush=True)


if __name__ == "__main__":
    main()

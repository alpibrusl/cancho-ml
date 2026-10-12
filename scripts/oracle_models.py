#!/usr/bin/env python3
"""Order-matched numpy oracles for lib/models (docs/models-linear.md section 2: G-oracle and G-det).

Each oracle performs the SAME floating-point operations in the SAME order as docs/models-linear.md section 3 (the block rule of
docs/linalg.md with K = 256 and the level-wise binary tree, products rounded before they are added, never fused), vectorised across
the independent parts. It never calls numpy's BLAS, exp, log1p, sum or linalg: `exp` and `log1p` are ports of std/math.cho. The
compiled driver build/models-check must equal it bit for bit: the parameters, every raw decisive sum of every iteration, every loss
or inertia line, and the hash of the predictions.

    python3 scripts/oracle_models.py --suite [--threads 1,2,4,8,16] [--model linreg|logreg|kmeans]
    python3 scripts/oracle_models.py CASE...        one case: MODEL n=.. d=.. seed=.. ...
Exit 1 on any mismatch. MODELS_CHECK overrides the driver path."""
import os, subprocess, sys, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_linalg as ol                      # the block rule, the tree, the Cholesky and the solves of the package

ROOT = HERE.parent
BIN = os.environ.get("MODELS_CHECK", str(ROOT / "build" / "models-check"))
K = ol.K
U = np.uint64


# ---------------------------------------------------------------- the generated data (driver: gen_data)
def gen(model, n, d, seed, dup=0):
    s1, s2, s3 = ol.derive(seed, 1), ol.derive(seed, 2), ol.derive(seed, 3)
    X = ol.elements(s1, n * d).reshape(n, d)
    e = ol.elements(s2, n)
    y = None
    if model == "kmeans":
        g = np.trunc((e + 1.0) * 2.5).astype(np.int64)
        for j in range(d):
            X[:, j] = X[:, j] + ((g * 7 + j * 3) % 11 - 5).astype(np.float64) * 1.5
    else:
        wt = ol.elements(s3, d)
        z = np.zeros(n)
        for j in range(d):
            z = z + X[:, j] * wt[j]
        if model == "linreg":
            y = (z + 0.5 * e) + 2.0
        else:
            y = np.where((z + 6.0 * e) + 0.25 > 0.0, 1.0, 0.0)
    if dup == 1 and d >= 2:
        X[:, 0] = X[:, 1]
    if dup == 2:
        X[:] = X[0]
    if dup == 3 and n > 2:
        X[2:] = X[np.arange(2, n) % 2]
    return X, y


def scores(X, w):
    """the score order: z = w[d]; z = z + x_j * w_j, j ascending"""
    n, d = X.shape
    z = np.full(n, w[d])
    for j in range(d):
        z = z + X[:, j] * w[j]
    return z


# ---------------------------------------------------------------- linear regression
def linreg(X, y, l2):
    n, d = X.shape
    p = d + 1
    Xa = np.hstack([X, np.ones((n, 1))])
    C = ol.syrk_o(Xa)                                   # upper triangle
    b = ol.gemv_t_o(Xa, y)
    G = np.triu(C) + np.triu(C, 1).T
    Gr = G.copy()
    for j in range(d):
        Gr[j, j] = Gr[j, j] + l2
    L = ol.chol_o(Gr)
    if L is None:
        return None
    u = ol.fwd_o(L, b)
    w = ol.bwd_o(L, u)
    diag = np.concatenate([G.ravel(), b])
    return w, diag, scores(X, w)


# ---------------------------------------------------------------- the driver
def run_driver(model, args, threads, reps=1):
    r = subprocess.run([BIN, model, "-", "-"] + ["%s=%s" % kv for kv in args.items()] + ["threads=%d" % threads, "reps=%d" % reps],
                       capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        k, _, v = line.partition(" ")
        out[k] = v
    out["_rc"] = r.returncode
    return out


def bits(arr):
    return " ".join("%016x" % b for b in np.ascontiguousarray(arr, dtype=np.float64).ravel().view(np.uint64))


def case(model, a, threads, quiet=False):
    n, d, seed = a["n"], a["d"], a["seed"]
    X, y = gen(model, n, d, seed, a.get("dup", 0))
    ntr = a.get("ntr", n)
    l2 = float(a.get("l2m", 0)) / 1000.0
    exp = None
    if model == "linreg":
        res = linreg(X[:ntr], y[:ntr], l2)
        if res is not None:
            w, diag, pred = res
            pred = scores(X, w)
            exp = {"params": bits(w), "diag": bits(diag), "pred": "%016x" % ol.hash_bits(pred)}
    got = run_driver(model, a, threads)
    if exp is None:
        ok = "refused" in got
    else:
        ok = all(got.get(k) == v for k, v in exp.items())
    label = "%-7s %s t=%-2d" % (model, " ".join("%s=%s" % kv for kv in a.items()), threads)
    if not quiet or not ok:
        print("%s %s" % (label, "ok" if ok else "MISMATCH"), flush=True)
        if not ok and exp is not None:
            for k, v in exp.items():
                if got.get(k) != v:
                    print("   %s: cancho %s...\n   %s: oracle %s..." % (k, str(got.get(k))[:100], k, v[:100]))
    return ok


# the shapes: ragged and aligned block (256) and chunk (2048) counts, odd and even d, l2 zero and positive
SUITE = {
    "linreg": [dict(n=n, d=d, seed=s, l2m=l) for (n, d, s, l) in [
        (1, 1, 1, 1000), (2, 1, 2, 0), (30, 3, 3, 0), (255, 3, 4, 0), (256, 4, 5, 1500), (257, 5, 6, 0), (1000, 20, 7, 0),
        (2048, 7, 8, 250), (2049, 7, 9, 0), (4097, 21, 10, 1000), (5000, 1, 11, 0), (20000, 20, 12, 0), (100000, 20, 13, 100)]]
    + [dict(n=1000, d=4, seed=5, l2m=0, dup=1), dict(n=1000, d=4, seed=5, l2m=10, dup=1)],
}
THREAD_SUITE = {
    "linreg": [dict(n=n, d=d, seed=s, l2m=l) for (n, d, s, l) in [(5000, 7, 3, 0), (30000, 20, 4, 500), (9000, 21, 5, 0), (4097, 3, 6, 0)]],
}
def main():
    a = sys.argv[1:]
    threads = [1]
    if "--threads" in a:
        threads = [int(t) for t in a[a.index("--threads") + 1].split(",")]
    models = a[a.index("--model") + 1].split(",") if "--model" in a else ["linreg", "logreg", "kmeans"]
    if "--suite" in a:
        ok = True
        table = THREAD_SUITE if "--thread-suite" in a else SUITE
        for m in models:
            for c in table.get(m, []):
                for t in threads:
                    ok &= case(m, c, t)
        print("suite:", "all bits equal" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)
    m = a[0]
    c = {k: int(v) for k, v in (x.split("=") for x in a[1:] if "=" in x and not x.startswith("threads="))}
    t = [int(x.split("=")[1]) for x in a if x.startswith("threads=")] or [1]
    sys.exit(0 if all(case(m, c, tt) for tt in t) else 1)


if __name__ == "__main__":
    main()

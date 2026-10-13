#!/usr/bin/env python3
"""Order-matched numpy oracles for lib/models (docs/models-linear.md section 2: G-oracle and G-det).

Each oracle performs the SAME floating-point operations in the SAME order as docs/models-linear.md section 3 (the block and lane rule of
docs/linalg.md section 2: K = 256, four lanes inside a block, the level-wise binary tree, products rounded before they are added, never fused), vectorised across
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


# ---------------------------------------------------------------- exp and log1p of std/math.cho (ports, elementwise)
def exp_c(x):
    """lib/models/mexp.cho exp: |x| <= 660 only (asserted); the same operations as std's exp, 2^k exact"""
    x = np.asarray(x, dtype=np.float64)
    assert np.all(np.abs(x) <= 660.0), "exp argument outside the table form"
    kf = x * 1.44269504088896338700e+00
    h = np.where(kf < 0.0, -0.5, 0.5)
    k = np.trunc(kf + h)
    hi = x - k * 6.93147180369123816490e-01
    lo = k * 1.90821492927058770002e-10
    r = hi - lo
    t = r * r
    c = r - t * (1.66666666666666019037e-01 + t * (-2.77777777770155933842e-03 + t * (6.61375632143793436117e-05 + t * (-1.65339022054652515390e-06 + t * 4.13813679705723846039e-08))))
    y = 1.0 - (lo - r * c / (2.0 - c) - hi)
    return y * np.ldexp(1.0, k.astype(np.int64))


def log_c(u):
    """std log for u in [1, 2] (normal numbers): fdlibm's __ieee754_log as written in std/math.cho"""
    u = np.asarray(u, dtype=np.float64)
    raw = (u.view(np.uint64) >> np.uint64(52)) & np.uint64(0x7ff)
    k = raw.astype(np.int64) - 1023
    m = u / np.ldexp(1.0, k)
    big = m > 1.4142135623730951
    m = np.where(big, m * 0.5, m)
    k = k + big.astype(np.int64)
    f = m - 1.0
    s = f / (2.0 + f)
    z = s * s
    w = z * z
    t1 = w * (3.999999999940941908e-01 + w * (2.222219843214978396e-01 + w * 1.531383769920937332e-01))
    t2 = z * (6.666666666666735130e-01 + w * (2.857142874366239149e-01 + w * (1.818357216161805012e-01 + w * 1.479819860511658591e-01)))
    tail = s * (0.5 * f * f + (t2 + t1))
    n = k.astype(np.float64)
    r0 = f - (0.5 * f * f - tail)
    r1 = n * 6.93147180369123816490e-01 - (0.5 * f * f - (tail + n * 1.90821492927058770002e-10) - f)
    return np.where(k == 0, r0, r1)


def log1p_c(x):
    x = np.asarray(x, dtype=np.float64)
    u = 1.0 + x
    return np.where(u == 1.0, x, log_c(np.where(u == 1.0, 2.0, u)) * x / (u - 1.0))


# ---------------------------------------------------------------- the block rule over a term matrix
def block_sums(T):
    """T: (n, W) terms. The sums per column by the package rule: blocks of 256 rows, four lanes inside a block (row i of the block to lane i mod 4, each lane ascending from +0.0), level-wise tree."""
    n, W = T.shape
    if n == 0:
        return np.zeros(W)
    nb = -(-n // K)
    P = np.zeros((nb * K, W))
    P[:n] = T
    P3 = P.reshape(nb, K, W)
    acc = ol.lane_partial(lambda t: P3[:, t, :], K, (nb, W))
    return ol.tree(acc)


# ---------------------------------------------------------------- logistic regression
def sigmoid_terms(z, y):
    az = np.abs(z)
    e = exp_c(-az)
    p = np.where(z >= 0.0, 1.0 / (1.0 + e), e / (1.0 + e))
    pos = np.where(z >= 0.0, z, 0.0)
    r = p - y
    s = p * (1.0 - p)
    loss = (pos + log1p_c(e)) - y * z
    return r, s, loss


def newton_terms(X, y, w):
    """(n, W) terms of the Newton pass at w: g (P), H lower by rows (P (P+1)/2), the loss"""
    n, d = X.shape
    P = d + 1
    z = scores(X, w)
    r, s, loss = sigmoid_terms(z, y)
    A = np.hstack([X, np.ones((n, 1))])
    cols = [r * A[:, a] for a in range(P)]
    for a in range(P):
        sa = s * A[:, a]
        for b in range(a + 1):
            cols.append(sa * A[:, b])
    cols.append(loss)
    return np.stack(cols, axis=1)


def logreg(X, y, iters, l2, rows_chunk=16384):
    n, d = X.shape
    P = d + 1
    W = P + P * (P + 1) // 2 + 1
    nf = np.float64(n)
    w = np.zeros(P)
    diag = []
    for it in range(iters):
        # whole blocks per group so the partials are those of the global rule; groups of rows_chunk rows
        parts = []
        for r0 in range(0, n, rows_chunk):
            T = newton_terms(X[r0:r0 + rows_chunk], y[r0:r0 + rows_chunk], w)
            m = T.shape[0]
            nb = -(-m // K)
            Pm = np.zeros((nb * K, W))
            Pm[:m] = T
            P3 = Pm.reshape(nb, K, W)
            acc = ol.lane_partial(lambda t: P3[:, t, :], K, (nb, W))
            parts.append(acc)
        sums = ol.tree(np.concatenate(parts, axis=0))
        q = np.float64(0.0)
        for j in range(d):
            q = q + w[j] * w[j]
        f = sums[W - 1] / nf + (0.5 * l2) * q
        diag.append(np.concatenate([sums, [f]]))
        g = sums[:P] / nf
        for a in range(d):
            g[a] = g[a] + l2 * w[a]
        H = np.zeros((P, P))
        for a in range(P):
            for b in range(a + 1):
                H[a, b] = sums[P + a * (a + 1) // 2 + b] / nf
        for a in range(d):
            H[a, a] = H[a, a] + l2
        L = ol.chol_o(H)
        if L is None:
            return None
        u = ol.fwd_o(L, g)
        dl = ol.bwd_o(L, u)
        w = w - dl
    z = scores(X, w)
    az = np.abs(z)
    e = exp_c(-az)
    prob = np.where(z >= 0.0, 1.0 / (1.0 + e), e / (1.0 + e))
    lab = np.where(z > 0.0, 1.0, 0.0)
    return w, np.concatenate(diag), lab, prob


# ---------------------------------------------------------------- k-means
M64 = (1 << 64) - 1


def uniform(z, m):
    return ((z >> 1) & 0x7FFFFFFFFFFFFFFF) % m if m > 0 else 0


def u01(z):
    return np.float64((z >> 11) & 0x1FFFFFFFFFFFFF) / 9007199254740992.0


def dist2_to(X, c):
    """squared distance of every row to one centroid: a left fold over j ascending from +0.0"""
    acc = np.zeros(X.shape[0])
    for j in range(X.shape[1]):
        t = X[:, j] - c[j]
        acc = acc + t * t
    return acc


def chosen_rows(X, k, seed, init):
    n, d = X.shape
    root = ol.derive(seed, 3)
    ch = []
    mind = None
    for j in range(k):
        zj = ol.derive(root, j)
        if init == 0:
            row, a, found = 0, 0, False
            while a < 64 and not found:
                row = uniform(ol.derive(zj, a), n)
                found = row not in ch
                a += 1
            while not found:
                row = row + 1
                if row >= n:
                    row = 0
                found = row not in ch
            ch.append(row)
        elif j == 0:
            ch.append(uniform(zj, n))
        else:
            dd = dist2_to(X, X[ch[j - 1]])
            mind = dd if j == 1 else np.where(dd < mind, dd, mind)
            total = np.float64(0.0)
            cum = np.zeros(n)
            c = np.float64(0.0)
            for i in range(n):          # left folds from 0.0 (a Python loop: the init is sequential by design)
                c = c + mind[i]
                cum[i] = c
            total = c
            u = u01(zj) * total
            hit = np.nonzero(cum > u)[0]
            ch.append(int(hit[0]) if hit.size else uniform(zj, n))
    return ch


def kmeans_terms(X, lab, cen):
    n, d = X.shape
    k = cen.shape[0]
    cols = []
    for c in range(k):
        m = lab == c
        for j in range(d):
            cols.append(np.where(m, X[:, j], 0.0))
    for c in range(k):
        cols.append(np.where(lab == c, 1.0, 0.0))
    return cols


def assign(X, cen):
    k = cen.shape[0]
    D = np.stack([dist2_to(X, cen[c]) for c in range(k)], axis=1)
    lab = np.argmin(D, axis=1)           # the first minimum: the lowest index wins a tie
    return lab, D[np.arange(X.shape[0]), lab], D


def sums_chunked(n, W, make, rows_chunk=16384):
    parts = []
    for r0 in range(0, n, rows_chunk):
        T = make(r0, min(n, r0 + rows_chunk))
        m = T.shape[0]
        nb = -(-m // K)
        Pm = np.zeros((nb * K, W))
        Pm[:m] = T
        P3 = Pm.reshape(nb, K, W)
        acc = ol.lane_partial(lambda t: P3[:, t, :], K, (nb, W))
        parts.append(acc)
    return ol.tree(np.concatenate(parts, axis=0))


def kmeans(X, k, iters, seed, init):
    n, d = X.shape
    W = k * d + k + 1
    ch = chosen_rows(X, k, seed, init)
    cen = X[ch].copy()
    diag = []
    for it in range(iters + 1):
        def make(r0, r1):
            lab, mn, _ = assign(X[r0:r1], cen)
            cols = kmeans_terms(X[r0:r1], lab, cen) + [mn]
            return np.stack(cols, axis=1)
        sums = sums_chunked(n, W, make)
        diag.append(sums)
        if it < iters:
            for c in range(k):
                cnt = sums[k * d + c]
                if cnt > 0.0:
                    cen[c] = sums[c * d:c * d + d] / cnt
    lab, _, D = assign(X, cen)
    return cen, np.concatenate(diag), lab, np.sqrt(D)


def hash_ints(v):
    v = np.asarray(v, dtype=np.int64).astype(np.uint64)
    idx = np.arange(v.size, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return int(np.sum(ol.mix(v ^ ol.mix(idx)), dtype=np.uint64))


# ---------------------------------------------------------------- the driver
def logreg_pred(X, w):
    z = scores(X, w)
    e = exp_c(-np.abs(z))
    prob = np.where(z >= 0.0, 1.0 / (1.0 + e), e / (1.0 + e))
    return w, None, np.where(z > 0.0, 1.0, 0.0), prob


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
    if model == "logreg":
        res = logreg(X[:ntr], y[:ntr], a.get("iters", 8), l2)
        if res is not None:
            w, diag, _, _ = res
            _, _, lab, prob = logreg_pred(X, w)
            exp = {"params": bits(w), "diag": bits(diag), "pred": "%016x" % ol.hash_bits(lab), "proba": "%016x" % ol.hash_bits(prob)}
    if model == "kmeans":
        cen, diag, lab, tr = kmeans(X[:ntr], a["k"], a["iters"], seed, a.get("init", 1))
        labp, _, Dp = assign(X, cen)
        _, _, _, _ = None, None, None, None
        exp = {"params": bits(cen), "diag": bits(diag), "labels": "%016x" % hash_ints(lab), "pred": "%016x" % hash_ints(labp),
               "transform": "%016x" % ol.hash_bits(np.sqrt(Dp))}
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
SUITE["logreg"] = [dict(n=n, d=d, seed=s, l2m=l, iters=i) for (n, d, s, l, i) in [
    (40, 1, 1, 0, 6), (300, 2, 2, 0, 8), (255, 3, 3, 1000, 8), (256, 4, 4, 0, 8), (257, 5, 5, 500, 8), (1000, 20, 6, 0, 8),
    (1000, 20, 7, 2000, 6), (2048, 7, 8, 0, 8), (2049, 7, 9, 100, 8), (4097, 21, 10, 0, 8), (5000, 3, 11, 0, 1), (20000, 20, 12, 0, 8)]]
SUITE["kmeans"] = [dict(n=n, d=d, seed=s, k=k, iters=i, init=ini, dup=du) for (n, d, s, k, i, ini, du) in [
    (9, 1, 1, 1, 3, 1, 0), (30, 2, 2, 3, 5, 1, 0), (255, 3, 3, 4, 4, 0, 0), (256, 4, 4, 5, 6, 1, 0), (257, 5, 5, 3, 5, 0, 0),
    (1000, 20, 6, 5, 8, 1, 0), (1000, 7, 7, 9, 6, 0, 0), (2048, 7, 8, 6, 5, 1, 0), (2049, 7, 9, 5, 5, 1, 0), (4097, 10, 10, 8, 4, 0, 0),
    (5000, 3, 11, 7, 1, 1, 0), (20000, 20, 12, 8, 6, 1, 0),
    # identical points: every distance ties (the lowest index wins) and clusters 1.. are empty (they keep their centroid)
    (8, 2, 13, 3, 3, 1, 2), (300, 3, 14, 4, 3, 0, 2), (3000, 2, 15, 5, 4, 1, 2), (500, 2, 24, 9, 3, 1, 2), (700, 3, 25, 13, 3, 0, 2), (600, 2, 26, 8, 3, 1, 2), (800, 3, 27, 16, 2, 1, 2), (300, 3, 28, 4, 3, 1, 2),
    # two distinct points: after the two centers every k-means++ weight is 0, the scan finds nothing and the draw of the fallback decides
    (40, 2, 16, 3, 3, 1, 3), (40, 2, 17, 3, 3, 1, 3), (41, 2, 18, 4, 3, 1, 3), (200, 3, 19, 5, 3, 1, 3), (200, 3, 20, 3, 2, 1, 3), (64, 2, 21, 4, 3, 1, 3),
    (64, 2, 22, 3, 3, 0, 3), (999, 2, 23, 6, 3, 1, 3)]]
THREAD_SUITE = {
    "linreg": [dict(n=n, d=d, seed=s, l2m=l) for (n, d, s, l) in [(5000, 7, 3, 0), (30000, 20, 4, 500), (9000, 21, 5, 0), (4097, 3, 6, 0)]],
}
THREAD_SUITE["logreg"] = [dict(n=n, d=d, seed=s, l2m=l, iters=i) for (n, d, s, l, i) in [
    (5000, 7, 3, 0, 4), (30000, 20, 4, 500, 6), (9000, 21, 5, 0, 5), (4097, 3, 6, 0, 8)]]


THREAD_SUITE["kmeans"] = [dict(n=n, d=d, seed=s, k=k, iters=i, init=ini, dup=du) for (n, d, s, k, i, ini, du) in [
    (5000, 7, 3, 5, 4, 1, 0), (30000, 20, 4, 6, 5, 1, 0), (9000, 10, 5, 8, 3, 0, 0), (4097, 3, 6, 4, 6, 1, 0), (5000, 2, 7, 5, 3, 1, 2)]]


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

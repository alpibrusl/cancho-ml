#!/usr/bin/env python3
"""Order-matched numpy oracle for lib/linalg (docs/linalg.md sections 2 and 4).

Each oracle performs the SAME floating-point operations in the SAME order as the normative rule (block rule with K = 256 and the
level-wise binary tree; products rounded before they are added; never fused), vectorised across the independent
parts, so the result must equal the compiled kernel bit for bit. numpy's own BLAS result is NOT used here.

    python3 scripts/oracle_linalg.py --tree                    stack == level-wise tree, every m up to 4096 (Python floats)
    python3 scripts/oracle_linalg.py --suite [--threads 1,2..] build/linalg-check against the oracles on the shape list
    python3 scripts/oracle_linalg.py KERNEL N D SEED [THREADS]  one case
    ORACLE_MUTANT=name python3 ...                             a deliberately wrong oracle (scripts/mutants_linalg.py)

Exit 1 on any mismatch."""
import os, subprocess, sys, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = os.environ.get("LINALG_CHECK", str(ROOT / "build" / "linalg-check"))
K = 256
MUT = os.environ.get("ORACLE_MUTANT", "")
U = np.uint64
GOLD = U(0x9E3779B97F4A7C15)


def mix(z):
    z = np.asarray(z, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> U(30))) * U(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> U(27))) * U(0x94D049BB133111EB)
        return z ^ (z >> U(31))


def derive(h, c):
    with np.errstate(over="ignore"):
        return int(mix(U((int(h) ^ int(c)) & 0xFFFFFFFFFFFFFFFF) + GOLD))


def elements(seed, n):
    """element i of the data set of `seed` (lib/core/rng.cho `element`)"""
    i = np.arange(1, n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = mix(U(seed) + i * GOLD)
    return (z >> U(11)).astype(np.float64) / 4503599627370496.0 - 1.0


def hash_bits(arr):
    a = np.ascontiguousarray(arr, dtype=np.float64).ravel()
    bits = a.view(np.uint64)
    idx = np.arange(a.size, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return int(np.sum(mix(bits ^ mix(idx)), dtype=np.uint64))


# ---------------------------------------------------------------- the rule
def tree(p):
    """level-wise binary tree over axis 0; an odd last element is carried up"""
    if p.shape[0] == 0:
        return np.zeros(p.shape[1:])
    while p.shape[0] > 1:
        m = p.shape[0]
        h = m // 2
        if MUT == "tree-leftfold":
            acc = p[0]
            for i in range(1, m):
                acc = acc + p[i]
            return acc
        q = p[0:2 * h:2] + p[1:2 * h:2]
        if m % 2:
            if MUT == "tree-carry-add":
                q = np.concatenate([q[:-1], q[-1:] + p[m - 1:m]], axis=0)
            else:
                q = np.concatenate([q, p[m - 1:m]], axis=0)
        p = q
    return p[0]


def kk():
    return 255 if MUT == "block-255" else K


def dot_o(x, y):
    n = x.size
    if n == 0:
        return np.float64(0.0)
    k = kk()
    nb = -(-n // k)
    p = np.zeros(nb * k)
    p[:n] = x * y
    P = p.reshape(nb, k)
    acc = np.zeros(nb)
    cols = range(k - 1, -1, -1) if MUT == "dot-reverse" else range(k)
    for j in cols:
        acc = acc + P[:, j]
    return tree(acc)


def gemv_n_o(X, w):
    n, d = X.shape
    if d == 0:
        return np.zeros(n)
    nb = -(-d // K)
    parts = []
    for b in range(nb):
        acc = np.zeros(n)
        for j in range(b * K, min(d, (b + 1) * K)):
            acc = acc + X[:, j] * w[j]
        parts.append(acc)
    return tree(np.array(parts))


def gemv_t_o(X, r):
    n, d = X.shape
    if n == 0:
        return np.zeros(d)
    k = kk()
    nb = -(-n // k)
    acc = np.zeros((nb, d))
    base = np.arange(nb) * k
    ts = range(k - 1, -1, -1) if MUT == "gemvt-reverse" else range(k)
    for t in ts:
        idx = base + t
        ok = idx < n
        ic = np.minimum(idx, n - 1)
        acc = acc + X[ic] * np.where(ok, r[ic], 0.0)[:, None]
    return tree(acc)


def syrk_o(X):
    """upper triangle (j <= k) of the d x d result, lower triangle zero; chunks of 8 blocks are aligned subtrees"""
    n, d = X.shape
    out = np.zeros((d, d))
    if n == 0:
        return out
    nb = -(-n // K)
    roots = []
    for g0 in range(0, nb, 8):
        g1 = min(nb, g0 + 8)
        base = np.arange(g0, g1) * K
        acc = np.zeros((g1 - g0, d, d))
        for t in range(K):
            idx = base + t
            ok = idx < n
            ic = np.minimum(idx, n - 1)
            R = X[ic] * ok[:, None]
            acc = acc + R[:, :, None] * R[:, None, :]
        roots.append(tree(acc))
    C = tree(np.array(roots))
    iu = np.triu_indices(d)
    out[iu] = C[iu]
    return out


def gemm_o(A, B):
    m, p = A.shape
    n = B.shape[1]
    if p == 0:
        return np.zeros((m, n))
    parts = []
    for b0 in range(0, p, K):
        acc = np.zeros((m, n))
        for k in range(b0, min(p, b0 + K)):
            acc = acc + A[:, k:k + 1] * B[k:k + 1, :]
        parts.append(acc)
    return tree(np.array(parts))


def chol_o(A):
    d = A.shape[0]
    L = np.zeros((d, d))
    for j in range(d):
        s = A[j, j]
        for k in range(j):
            s = s - L[j, k] * L[j, k]
        if not s > 0.0:
            return None
        ljj = np.sqrt(s)
        L[j, j] = ljj
        if j + 1 < d:
            s = A[j + 1:, j].copy()
            for k in range(j):
                s = s - L[j + 1:, k] * L[j, k]
            L[j + 1:, j] = s / ljj
    return L


def fwd_o(L, b):
    d = L.shape[0]
    x = np.zeros(d)
    for i in range(d):
        s = b[i]
        for k in range(i):
            s = s - L[i, k] * x[k]
        x[i] = s / L[i, i]
    return x


def bwd_o(L, b):
    d = L.shape[0]
    x = np.zeros(d)
    for i in range(d - 1, -1, -1):
        s = b[i]
        ks = range(i + 1, d) if MUT == "bwd-ascending" else range(d - 1, i, -1)
        for k in ks:
            s = s - L[k, i] * x[k]
        x[i] = s / L[i, i]
    return x


def spd(d, seed):
    E = elements(seed, d * d).reshape(d, d)
    A = 0.5 * (E + E.T)
    idx = np.arange(d)
    A[idx, idx] = (float(d) + 1.0) + E[idx, idx]
    return A


def oracle(kernel, n, d, seed):
    """the expected result array (flattened) of linalg-check KERNEL N D SEED REPS=1"""
    s1, s2, s3 = derive(seed, 1), derive(seed, 2), derive(seed, 3)
    if kernel == "dot":
        return np.array([dot_o(elements(s1, n), elements(s2, n))])
    if kernel == "axpy":
        a = elements(s3, 1)[0]
        return elements(s2, n) + a * elements(s1, n)
    if kernel == "gemv_n":
        return gemv_n_o(elements(s1, n * d).reshape(n, d), elements(s2, d))
    if kernel == "gemv_t":
        return gemv_t_o(elements(s1, n * d).reshape(n, d), elements(s2, n))
    if kernel == "syrk":
        return syrk_o(elements(s1, n * d).reshape(n, d))
    if kernel == "chol":
        return chol_o(spd(n, s1))
    if kernel in ("fwd", "bwd"):
        L = chol_o(spd(n, s1))
        b = elements(s2, n)
        return fwd_o(L, b) if kernel == "fwd" else bwd_o(L, b)
    if kernel == "gemm":
        return gemm_o(elements(s1, n * n).reshape(n, n), elements(s2, n * n).reshape(n, n))
    raise SystemExit("unknown kernel " + kernel)


def run_cancho(kernel, n, d, seed, reps=1, threads=1):
    r = subprocess.run([BIN, kernel, str(n), str(d), str(seed), str(reps), str(threads)], capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        k, _, v = line.partition(" ")
        out[k] = v
    out["_rc"] = r.returncode
    return out


def check(kernel, n, d, seed, threads=1, quiet=False):
    exp = oracle(kernel, n, d, seed)
    got = run_cancho(kernel, n, d, seed, 1, threads)
    want = "%016x" % hash_bits(exp)
    first = " ".join("%016x" % b for b in np.ascontiguousarray(exp, dtype=np.float64).ravel().view(np.uint64)[:4])
    ok = got.get("hash") == want and got.get("first") == first and int(got.get("count", -1)) == np.asarray(exp).size
    if not quiet or not ok:
        print("%-7s n=%-8d d=%-4d seed=%-3d t=%-2d %s%s" % (kernel, n, d, seed, threads, "ok" if ok else "MISMATCH",
              "" if ok else "   cancho %s %s / oracle %s %s" % (got.get("hash"), got.get("first"), want, first)), flush=True)
    return ok


# the shapes: ragged and aligned block counts, d on both sides of 256, small and odd d for the fringe tiles
SUITE = [
    ("dot", 0, 0), ("dot", 1, 0), ("dot", 255, 0), ("dot", 256, 0), ("dot", 257, 0), ("dot", 2048, 0), ("dot", 2049, 0),
    ("dot", 10000, 0), ("dot", 123457, 0), ("dot", 1000000, 0),
    ("axpy", 0, 0), ("axpy", 1, 0), ("axpy", 1000, 0), ("axpy", 1000000, 0),
    ("gemv_n", 1000, 20), ("gemv_n", 777, 64), ("gemv_n", 301, 256), ("gemv_n", 50, 300), ("gemv_n", 100000, 20),
    ("gemv_t", 1000, 20), ("gemv_t", 2049, 64), ("gemv_t", 4097, 21), ("gemv_t", 300, 256), ("gemv_t", 100000, 20), ("gemv_t", 100000, 64),
    ("syrk", 1000, 20), ("syrk", 2049, 21), ("syrk", 4100, 64), ("syrk", 3000, 7), ("syrk", 20000, 20),
    ("chol", 1, 0), ("chol", 21, 0), ("chol", 64, 0), ("chol", 100, 0), ("chol", 256, 0),
    ("fwd", 21, 0), ("fwd", 64, 0), ("fwd", 256, 0), ("bwd", 21, 0), ("bwd", 64, 0), ("bwd", 256, 0),
    ("gemm", 4, 0), ("gemm", 33, 0), ("gemm", 64, 0), ("gemm", 256, 0), ("gemm", 300, 0),
]


# kernels with a threaded variant, shapes with several chunks (2048 rows each), ragged and aligned
THREAD_SUITE = [
    ("dot", 4097, 0), ("dot", 123457, 0), ("dot", 1000000, 0),
    ("gemv_t", 5000, 21), ("gemv_t", 100000, 20), ("gemv_t", 100000, 64),
    ("syrk", 4097, 7), ("syrk", 30000, 20), ("syrk", 9000, 21), ("syrk", 20000, 64),
    ("gemv_n", 100000, 20), ("gemv_n", 20001, 64),
    ("gemm", 64, 0), ("gemm", 129, 0), ("gemm", 300, 0),
]


def tree_check():
    """the stack of lib/linalg/order.cho (re-implemented) equals the level-wise tree, in Python floats, for m up to 4096"""
    rng = np.random.default_rng(7)
    bad = 0
    for m in range(1, 4097):
        v = (rng.standard_normal(m) * 10.0 ** rng.integers(-8, 8, m)).tolist()
        lv, st = [], []
        for x in v:
            st.append(x); lv.append(0)
            while len(st) >= 2 and lv[-1] == lv[-2]:
                b = st.pop(); lv.pop()
                st[-1] = st[-1] + b; lv[-1] += 1
        while len(st) >= 2:
            b = st.pop()
            st[-1] = st[-1] + b
        a = np.array(v)
        while len(a) > 1:
            h = len(a) // 2
            q = a[0:2 * h:2] + a[1:2 * h:2]
            if len(a) % 2:
                q = np.concatenate([q, a[-1:]])
            a = q
        if st[0] != a[0] or np.float64(st[0]).tobytes() != a[0].tobytes():
            bad += 1
    print("tree: stack == level-wise for m = 1..4096: %s" % ("ok" if bad == 0 else "%d MISMATCHES" % bad))
    return bad == 0


def main():
    a = sys.argv[1:]
    if "--tree" in a:
        sys.exit(0 if tree_check() else 1)
    if "--suite" in a:
        threads = [1]
        if "--threads" in a:
            threads = [int(t) for t in a[a.index("--threads") + 1].split(",")]
        ok = True
        only = a[a.index("--kernel") + 1].split(",") if "--kernel" in a else None
        for kernel, n, d in (THREAD_SUITE if "--thread-suite" in a else SUITE):
            if only and kernel not in only:
                continue
            for t in threads:
                ok &= check(kernel, n, d, 3, t)
        print("suite:", "all bits equal" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)
    kernel, n, d, seed = a[0], int(a[1]), int(a[2]), int(a[3])
    t = int(a[4]) if len(a) > 4 else 1
    sys.exit(0 if check(kernel, n, d, seed, t) else 1)


if __name__ == "__main__":
    main()

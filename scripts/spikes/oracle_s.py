#!/usr/bin/env python3
"""Spike S oracle: the two fixed-iteration solvers of docs/spikes.md section 10, in numpy.  No scipy, no black box.

usage: oracle_s.py DATADIR newton|lbfgs ITERS OUT_HEX [--m 10] [--trials 10] [--c1 1e-4] [--t0e 0]

Both minimise f(w) = mean_i [ softplus(z_i) - y_i z_i ],  z_i = x_i . w[0:20] + w[20],  on the train rows (first n - n//5),
starting at w = 0, no penalty.  The 21 parameters are w[0..19] then the bias w[20].  The exact algorithms are in section 10.1;
this file is their transcription.  Passes over X are counted in `passes` (each evaluation of the gradient = one pass).
"""
import sys, numpy as np
from sklearn.metrics import roc_auc_score
import oracle_l

D = 20; P = 21

def augment(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])

# ---------------------------------------------------------------- shared pieces
def logistic_terms(A, y, w):
    """z, p, per-row loss with the one-exp stable forms of section 10.1 (e = exp(-|z|))."""
    z = A @ w
    e = np.exp(-np.abs(z))
    p = np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    loss = np.maximum(z, 0.0) + np.log1p(e) - y * z
    return z, p, loss

def loss_grad(A, y, w):
    n = A.shape[0]
    z, p, loss = logistic_terms(A, y, w)
    return loss.sum() / n, (A.T @ (p - y)) / n

# ---------------------------------------------------------------- Newton
def cholesky_solve(H, g):
    """Solve H d = g by Cholesky H = L L^T, plain loops, ascending index in every sum (the order cancho uses).
    H: 21x21 (lower triangle used). Raises if a pivot is not > 0."""
    L = np.zeros((P, P))
    for i in range(P):
        for j in range(i + 1):
            s = H[i, j]
            for k in range(j):
                s = s - L[i, k] * L[j, k]
            if i == j:
                if not s > 0.0:
                    raise ValueError("not positive definite")
                L[i, i] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    u = np.zeros(P)
    for i in range(P):
        s = g[i]
        for k in range(i):
            s = s - L[i, k] * u[k]
        u[i] = s / L[i, i]
    d = np.zeros(P)
    for i in range(P - 1, -1, -1):
        s = u[i]
        for k in range(i + 1, P):
            s = s - L[k, i] * d[k]
        d[i] = s / L[i, i]
    return d

def newton(A, y, iters, step=1.0):
    n = A.shape[0]; w = np.zeros(P); passes = 0
    for _ in range(iters):
        z = A @ w
        p = 1.0 / (1.0 + np.exp(-z))
        g = (A.T @ (p - y)) / n
        s = p * (1.0 - p)
        H = ((A * s[:, None]).T @ A) / n
        passes += 1
        d = cholesky_solve(H, g)
        w = w - step * d
    return w, passes

# ---------------------------------------------------------------- L-BFGS
def two_loop(g, S, Y):
    """H_k g with the standard two-loop recursion; S, Y lists oldest first; gamma = s.y / y.y of the newest pair."""
    q = g.copy(); m = len(S); alpha = [0.0] * m; rho = [1.0 / float(Y[i] @ S[i]) for i in range(m)]
    for i in range(m - 1, -1, -1):
        alpha[i] = rho[i] * float(S[i] @ q)
        q = q - alpha[i] * Y[i]
    if m > 0:
        q = q * (float(S[-1] @ Y[-1]) / float(Y[-1] @ Y[-1]))
    for i in range(m):
        beta = rho[i] * float(Y[i] @ q)
        q = q + (alpha[i] - beta) * S[i]
    return q

def lbfgs(A, y, iters, m=10, trials=10, c1=1e-4, t0e=0):
    w = np.zeros(P); f, g = loss_grad(A, y, w); passes = 1; S = []; Y = []; skipped = 0; backtracks = 0
    for k in range(iters):
        d = -two_loop(g, S, Y)
        gd = float(g @ d)
        t = 1.0 / float(np.sqrt(g @ g)) if k == 0 else 1.0
        if k == 0:
            for _e in range(t0e): t = t * 2.0
        for tr in range(trials):
            wn = w + t * d
            fn, gn = loss_grad(A, y, wn); passes += 1
            if fn <= f + c1 * t * gd or tr == trials - 1:
                break
            t = t * 0.5; backtracks += 1
        s = wn - w; yv = gn - g
        if float(s @ yv) > 0.0:
            S.append(s); Y.append(yv)
            if len(S) > m: S.pop(0); Y.pop(0)
        else:
            skipped += 1
        w, f, g = wn, fn, gn
    return w, passes, dict(skipped=skipped, backtracks=backtracks)

def solve(A, y, algo, iters, m=10, trials=10, c1=1e-4, t0e=0):
    if algo == "newton":
        w, passes = newton(A, y, iters); return w, passes, {}
    return lbfgs(A, y, iters, m, trials, c1, t0e)

def hexes(w):
    return ["%016x" % int(u) for u in np.asarray(w, dtype="<f8").view("<u8")]

def main():
    a = [s for s in sys.argv[1:] if not s.startswith("--")]
    opt = {k: v for k, v in zip(sys.argv[1:], sys.argv[2:]) if k.startswith("--")}
    m = int(opt.get("--m", 10)); trials = int(opt.get("--trials", 10)); c1 = float(opt.get("--c1", 1e-4)); t0e = int(opt.get("--t0e", 0))
    d, algo, iters, out = a[0], a[1], int(a[2]), a[3]
    X, y = oracle_l.load(d); ntr = oracle_l.split(y.size)
    A = augment(X[:ntr]); w, passes, info = solve(A, y[:ntr], algo, iters, m, trials, c1, t0e)
    open(out, "w").write("\n".join(hexes(w)) + "\n")
    f, _ = loss_grad(A, y[:ntr], w)
    print(algo, "passes", passes, info, "train_logloss %.12f test_auc %.6f" % (f, roc_auc_score(y[ntr:], augment(X[ntr:]) @ w)))

if __name__ == "__main__":
    main()

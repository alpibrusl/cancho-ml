#!/usr/bin/env python3
"""Spike L oracle: batch gradient descent on the mean log-loss, numpy.  (docs/spikes.md section 2)

usage: oracle_l.py DATADIR LR ITERS OUT_HEX [--order blas|seq]

Algorithm (the same in spike_l.cho, arm A):
  split: train = first n - n//5 rows, test = the last n//5 rows (fixed).
  w = 0 (20), b = 0.
  repeat ITERS times, on the train rows:
     z_i = b + x_i0*w0 + x_i1*w1 + ... (terms added left to right, starting from b)
     p_i = 1 / (1 + exp(-z_i));  r_i = p_i - y_i
     gw_j = sum_i r_i * x_ij  (rows added in order);  gb = sum_i r_i
     w_j = w_j - LR * (gw_j / ntr);  b = b - LR * (gb / ntr)
  no regularisation, no stopping rule.
--order blas (default): z = X @ w + b and g = X.T @ r: numpy's own summation order (BLAS / pairwise).
--order seq : z by adding columns left to right and g by np.cumsum (sequential), i.e. cancho arm A's order of additions.
OUT_HEX: 21 lines, 16 lowercase hex digits each: the IEEE bit patterns of w[0..19], then b. Prints train/test AUC.
"""
import sys, numpy as np
from sklearn.metrics import roc_auc_score

D = 20

def load(d):
    x = np.fromfile(d + "/x.bin", dtype="<f8")
    y = np.fromfile(d + "/y.bin", dtype="<f8")
    n = y.size
    return x.reshape(n, D), y

def split(n):
    return n - n // 5

def fit(X, y, lr, iters, order="blas"):
    n = X.shape[0]
    w = np.zeros(D); b = 0.0
    nf = float(n)
    for _ in range(iters):
        if order == "blas":
            z = X @ w + b
        else:
            z = np.full(n, b)
            for j in range(D):
                z = z + X[:, j] * w[j]
        p = 1.0 / (1.0 + np.exp(-z))
        r = p - y
        if order == "blas":
            gw = X.T @ r; gb = r.sum()
        else:
            gw = np.array([np.cumsum(r * X[:, j])[-1] for j in range(D)]); gb = np.cumsum(r)[-1]
        w = w - lr * (gw / nf)
        b = b - lr * (gb / nf)
    return w, b

def hexes(w, b):
    v = np.concatenate([w, [b]]).astype("<f8")
    return ["%016x" % int(u) for u in v.view("<u8")]

def main():
    a = [s for s in sys.argv[1:] if not s.startswith("--")]
    order = "seq" if "--order" in sys.argv and sys.argv[sys.argv.index("--order") + 1] == "seq" else "blas"
    a = [s for s in a if s not in ("seq", "blas")]
    d, lr, iters, out = a[0], float(a[1]), int(a[2]), a[3]
    X, y = load(d)
    ntr = split(y.size)
    w, b = fit(X[:ntr], y[:ntr], lr, iters, order)
    open(out, "w").write("\n".join(hexes(w, b)) + "\n")
    s = lambda Xs: Xs @ w + b
    print("order", order, "train_auc %.6f test_auc %.6f" % (roc_auc_score(y[:ntr], s(X[:ntr])), roc_auc_score(y[ntr:], s(X[ntr:]))))

if __name__ == "__main__":
    main()

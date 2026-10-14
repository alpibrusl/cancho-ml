#!/usr/bin/env python3
"""How far apart are the A, B and C models, and does it matter? (docs/spikes.md section 15.4)

usage: compare_models.py N

Runs arms A, B, C (1 thread, 6 iterations), prints for each pair the distance in ulps of the 21 coefficients (max and list), the
relative distance of each arm to the numpy Newton oracle of spike S (scripts/spikes/oracle_s.py) (oracle_s.newton, float64, BLAS sums: a different order
again), and the test AUC and test mean log-loss (rows n - n/5 .. n) of each model, in full precision, with differences."""
import sys, pathlib
import numpy as np
import qlib

SOLVER = qlib.ROOT / "scripts" / "spikes"
sys.path.insert(0, str(SOLVER))


def metrics(X, y, w):
    z = X @ w[:20] + w[20]
    ll = float(np.mean(np.logaddexp(0.0, z) - y * z))
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, z)), ll


def main():
    n = int(sys.argv[1])
    ntr = qlib.NTRAIN(n)
    d = qlib.datadir(n)
    X = np.fromfile(d / "x.bin", dtype="<f8").reshape(n, 20)
    y = np.fromfile(d / "y.bin", dtype="<f8")
    ws = {}
    for name, arm in (("A", 0), ("B", 1), ("C", 2)):
        r = qlib.parse(qlib.run_q(n, arm, 1, 6))
        ws[name] = qlib.hexes_to_array(r["w"])
    import oracle_s
    A = oracle_s.augment(X[:ntr])
    wo, _ = oracle_s.newton(A, y[:ntr], 6)
    print(f"n={n}: train {ntr} rows, test {n - ntr} rows")
    for p, q in (("A", "B"), ("A", "C"), ("B", "C")):
        u = [qlib.ulps(ws[p][i], ws[q][i]) for i in range(21)]
        print(f"ulps {p} vs {q}: max {max(u)}, per coefficient {u}")
    for k in "ABC":
        rel = np.max(np.abs(ws[k] - wo) / np.maximum(np.abs(wo), 1e-300))
        print(f"max relative distance of {k} to the numpy Newton oracle: {rel:.3e}")
    m = {k: metrics(X[ntr:], y[ntr:], ws[k]) for k in "ABC"}
    for k in "ABC":
        print(f"{k}: test AUC {m[k][0]!r}  test log-loss {m[k][1]!r}")
    for p, q in (("A", "B"), ("C", "B"), ("A", "C")):
        print(f"{p} - {q}: dAUC {m[p][0] - m[q][0]:.3e}  dlogloss {m[p][1] - m[q][1]:.3e}")
    wm = np.max(np.abs(ws["B"] - ws["C"]))
    print(f"max |B - C| absolute over coefficients: {wm:.3e}")


main()

#!/usr/bin/env python3
"""C2a (the scikit-learn half of gate WG5, docs/weights.md section 5) -- one weighted tree of lib/models/tree against `DecisionTreeClassifier` and `DecisionTreeRegressor` with
`sample_weight`, with the bound fixed BEFORE this script was run:

    on the training rows of 4 planted data sets (2 weight kinds, a classification and a regression each), the two trees predict within 1e-9 on at least 95 percent of the rows

The data are 4 columns of 6 values each (so the quantile bins separate every value and the thresholds of the two libraries cut the same partitions), 600 rows, depth 4, a leaf of one
row, no bootstrap, every feature at every node. The control: scikit-learn fitted WITHOUT the weights must agree on less than 95 percent of the rows (else the weights do not decide the
tree and the comparison judges nothing). Exit 1 when a bound is exceeded or a control stays inside it; the bound is never widened.

    python3 scripts/gate_weights_tree_sklearn.py [PATH-TO-tree-check]"""
import os, subprocess, struct, sys, pathlib, warnings
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import oracle_tree as ot

EXE = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "build" / "tree-check")
SEED = "20261012"
AGREE_MIN = 0.95
TOL = 1e-9
N, D = 600, 4


def data(task, seed):
    r = np.random.default_rng(seed)
    x = r.integers(0, 6, size=(N, D)).astype(np.float64)
    z = x[:, 0] * 1.5 - x[:, 1] + 0.7 * (x[:, 2] > 2) + r.normal(0, 1.0, N)
    y = (z > np.median(z)).astype(np.float64) if task == 0 else z
    return x, y


def ours(x, y, task, kind, wt_dir):
    os.makedirs(wt_dir, exist_ok=True)
    xa, ya = np.concatenate([x, x]), np.concatenate([y, y])
    xa.astype("<f8").tofile(f"{wt_dir}/x.bin")
    ya.astype("<f8").tofile(f"{wt_dir}/y.bin")
    a = [EXE, "mode=fit", f"x={wt_dir}/x.bin", f"y={wt_dir}/y.bin", "n=%d" % (2 * N), "ntr=%d" % N, "d=%d" % D, "task=%d" % task, "seed=" + SEED, "trees=1", "depth=4", "leaf=1",
         "bins=64", "mtry=%d" % D, "boot=0", "wt=%d" % kind, "out=full"]
    out = subprocess.run(a, capture_output=True, text=True, check=True).stdout
    return np.array([struct.unpack("<d", struct.pack("<Q", int(l.split()[2], 16)))[0] for l in out.splitlines() if l.startswith("P ")])


def main():
    ok = True
    for task in (0, 1):
        for kind in (3, 4):
            x, y = data(task, 31 + task)
            w = ot.gen_weights(int(SEED), N, kind)
            mine = ours(x, y, task, kind, str(ROOT / "build" / "wsk"))
            Est = DecisionTreeClassifier if task == 0 else DecisionTreeRegressor
            kw = dict(max_depth=4, min_samples_leaf=1, random_state=0)
            m = Est(**kw).fit(x, y, sample_weight=w)
            sk = m.predict_proba(x)[:, 1] if task == 0 else m.predict(x)
            c = Est(**kw).fit(x, y)
            ck = c.predict_proba(x)[:, 1] if task == 0 else c.predict(x)
            agree = float(np.mean(np.abs(mine - sk) <= TOL))
            cagree = float(np.mean(np.abs(mine - ck) <= TOL))
            good = agree >= AGREE_MIN and cagree < AGREE_MIN
            ok &= good
            print("%-14s wt=%d  agreement %.4f (bound %.2f)  control (no weights) %.4f  %s" % ("classification" if task == 0 else "regression", kind, agree, AGREE_MIN, cagree, "ok" if good else "FAIL"))
    print("gate_weights_tree_sklearn:", "ok" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

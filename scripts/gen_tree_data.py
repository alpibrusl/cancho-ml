#!/usr/bin/env python3
"""Inputs of the tree gates (docs/models-tree.md section 8), written to build/tree-data/ from ml-data/ (read-only).

For each size n in 1000, 100000 (the files of spike F: 20 float64 columns, labels 0/1):
    n<N>/reg_y.bin    a regression target: a fixed nonlinear function of the columns plus seeded noise
    n<N>/xm.bin       the same matrix with NaN written into the missing cells (the library must never read them)
    n<N>/miss.bin     the missing bitmap: d columns of ceil(n/8) bytes, bit i & 7 of byte i >> 3 (LSB first), set = missing;
                      about 12% of the cells of 7 columns, the probability depending on the label in two of them
and the edge-case set (1,000 rows x 8 columns) edge/ : x.bin xm.bin miss.bin y.bin (class) yr.bin (regression, noisy) yq.bin (regression, integers 0..5):
    column 0 constant, 1 only -0.0 and 0.0, 2 small integers, 3 missing in every row, 4 missing in 95% of the rows,
    5 values 1e300 / -1e300 / subnormals, 6 normal, 7 heavy duplicates (so many exactly equal scores and repeated edges).
usage: gen_tree_data.py [OUT]  (default build/tree-data)"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("ML_DATA", os.path.join(os.path.dirname(ROOT), "ml-data"))
MISS_COLS = (1, 3, 5, 7, 11, 13, 17)


def bitmap(mask):  # mask: (n, d) bool -> bytes, column-major bits LSB first
    n, d = mask.shape
    out = bytearray()
    for c in range(d):
        out += np.packbits(mask[:, c].astype(np.uint8), bitorder="little").tobytes()
    return bytes(out)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "tree-data")
    for n in (1000, 100000):
        d = f"{out}/n{n}"
        os.makedirs(d, exist_ok=True)
        X = np.fromfile(f"{DATA}/n{n}/x.bin", "<f8").reshape(n, 20)
        y = np.fromfile(f"{DATA}/n{n}/y.bin", "<f8")
        rng = np.random.default_rng(7)
        reg = 2.0 * X[:, 0] - 3.0 * X[:, 3] * X[:, 5] + np.sin(3.0 * X[:, 7]) + 0.5 * np.abs(X[:, 11]) + 0.3 * rng.standard_normal(n)
        reg.astype("<f8").tofile(f"{d}/reg_y.bin")
        rng = np.random.default_rng(11)
        mask = np.zeros((n, 20), dtype=bool)
        for c in MISS_COLS:
            p = 0.08 + (0.25 if c in (3, 7) else 0.0) * y
            mask[:, c] = rng.random(n) < p
        open(f"{d}/miss.bin", "wb").write(bitmap(mask))
        Xm = X.copy()
        Xm[mask] = np.nan
        Xm.astype("<f8").tofile(f"{d}/xm.bin")
        print("n%d: missing fraction %.4f over all cells" % (n, mask.mean()))
    d = f"{out}/edge"
    os.makedirs(d, exist_ok=True)
    n = 1000
    rng = np.random.default_rng(23)
    X = np.zeros((n, 8))
    X[:, 0] = 3.0
    X[:, 1] = np.where(rng.random(n) < 0.5, -0.0, 0.0)
    X[:, 2] = rng.integers(0, 5, n).astype(float)
    X[:, 3] = rng.standard_normal(n)
    X[:, 4] = rng.standard_normal(n)
    X[:, 5] = rng.choice(np.array([1e300, -1e300, 5e-324, -5e-324, 1e-310, 0.0, 2.5]), n)
    X[:, 6] = rng.standard_normal(n)
    X[:, 7] = rng.choice(np.array([0.5, 1.5, 1.5, 1.5, 7.25]), n)
    mask = np.zeros((n, 8), dtype=bool)
    mask[:, 3] = True
    mask[:, 4] = rng.random(n) < 0.95
    mask[:, 6] = rng.random(n) < 0.05
    y = ((X[:, 2] >= 2) ^ (X[:, 6] > 0.3) ^ (rng.random(n) < 0.1)).astype(float)
    yr = X[:, 2] * 1.5 - X[:, 7] + 0.25 * rng.standard_normal(n)
    yq = X[:, 2] + (X[:, 7] > 1.0)  # integers 0 .. 5: many nodes whose targets are all equal
    X.astype("<f8").tofile(f"{d}/x.bin")
    Xm = X.copy()
    Xm[mask] = np.nan
    Xm.astype("<f8").tofile(f"{d}/xm.bin")
    open(f"{d}/miss.bin", "wb").write(bitmap(mask))
    y.astype("<f8").tofile(f"{d}/y.bin")
    yr.astype("<f8").tofile(f"{d}/yr.bin")
    yq.astype("<f8").tofile(f"{d}/yq.bin")
    print("edge: 1000 x 8, %d missing cells" % mask.sum())


main()

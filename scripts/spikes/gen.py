#!/usr/bin/env python3
"""M0 data generator (docs/spikes.md section 2).

usage: gen.py [OUTDIR]     (default /Users/alfonso/Workspace/alpibrusl/lexsys/ml-data)

One stream, seed SEED, makes N_MAX = 1,000,000 rows. The smaller sets are PREFIXES of it (the first 100,000 and the first
1,000 rows), as the gate asks ("a 100,000-row prefix"). Per set, OUTDIR/n<rows>/ holds
  x.bin   rows x 20 float64, row-major, little-endian (standard normal)
  y.bin   rows float64 0.0/1.0
  meta.json  n, d, seed, sha256 of each file, the hidden model
Hidden model: w_true = 0.5 * N(0,1) (20 values) drawn first from default_rng(SEED), b_true = 0.3. Then, for each of the
10 chunks of CHUNK = 100,000 rows in turn: X = standard_normal((CHUNK, 20)), then u = random(CHUNK); y = 1.0 if
u < sigmoid(X @ w_true + b_true) else 0.0. The chunk size is part of the definition (it fixes the stream order). The
1,000-row set is the first 1,000 rows of chunk 0.
Files are written once and never rewritten; sha256 is of the bytes on disk.
"""
import hashlib, json, os, sys
import numpy as np

SEED = 20261012
D = 20
CHUNK = 100_000
SIZES = [1_000_000, 100_000, 1_000]
B_TRUE = 0.3

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/Users/alfonso/Workspace/alpibrusl/lexsys/ml-data"
    rng = np.random.default_rng(SEED)
    w_true = 0.5 * rng.standard_normal(D)
    xs, ys = [], []
    for _ in range(SIZES[0] // CHUNK):
        x = rng.standard_normal((CHUNK, D))
        u = rng.random(CHUNK)
        p = 1.0 / (1.0 + np.exp(-(x @ w_true + B_TRUE)))
        xs.append(x)
        ys.append((u < p).astype(np.float64))
    X = np.ascontiguousarray(np.concatenate(xs)).astype('<f8')
    Y = np.concatenate(ys).astype('<f8')
    for n in SIZES:
        d = os.path.join(out, f"n{n}")
        os.makedirs(d, exist_ok=True)
        xb = X[:n].tobytes(); yb = Y[:n].tobytes()
        with open(os.path.join(d, "x.bin"), "wb") as f: f.write(xb)
        with open(os.path.join(d, "y.bin"), "wb") as f: f.write(yb)
        meta = {"n": n, "d": D, "seed": SEED, "chunk": CHUNK, "b_true": B_TRUE, "w_true": w_true.tolist(),
                "positive_rate": float(Y[:n].mean()),
                "x_sha256": hashlib.sha256(xb).hexdigest(), "y_sha256": hashlib.sha256(yb).hexdigest(),
                "numpy": np.__version__}
        with open(os.path.join(d, "meta.json"), "w") as f: json.dump(meta, f, indent=1)
        print(n, meta["x_sha256"], meta["y_sha256"], meta["positive_rate"])

if __name__ == "__main__":
    main()

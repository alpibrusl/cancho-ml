#!/usr/bin/env python3
"""Correctness gates G-A, G-B-exact, G-C-order of docs/spikes.md section 15.3.

usage: oracle_q.py N [--dump]     N in 100000, 1000000

1. G-A: arm A (spike_q, 1 thread, 6 iterations) against spike S (unchanged), coefficients bit for bit.
2. A start point W0 is taken from arm C after 3 iterations. From W0 one step is run with arm B and OUT=2: the spike writes the per-row
   terms (ntr x 253 float64) it adds, and the 253 raw sums. Then
   G-B-exact: the 253 sums of B against math.fsum (exactly rounded) of the 253 term columns, bit for bit.
   G-C-order: the 253 sums of arm C (OUT=1, same W0) against the block rule (K = 256, ascending from +0.0) and the level-wise
              tree of oracle_linalg.tree, on the same terms, bit for bit.
   G-A-order: the 253 sums of arm A against rows ascending from +0.0 (numpy, one vector add per row), bit for bit.
   The terms of arms A and C are computed by their own code; they are compared to the terms of B only through these sums."""
import math, sys, struct
import numpy as np
import qlib
sys.path.insert(0, str(qlib.ROOT / "scripts"))
import oracle_linalg

K = 256


def bits(x):
    return int(np.float64(x).view(np.uint64))


def main():
    n = int(sys.argv[1])
    ntr = qlib.NTRAIN(n)
    ok = True
    # G-A
    d = qlib.datadir(n)
    import subprocess
    s = subprocess.run([qlib.SPIKE_S, str(d / "x.bin"), str(d / "y.bin"), str(n), "0", "6", "10", "10", "0"], capture_output=True, text=True, check=True).stdout
    sw = [l for l in s.splitlines() if not l.startswith("#")][:21]
    a = qlib.parse(qlib.run_q(n, 0, 1, 6))
    ga = sw == a["w"]
    print(f"G-A  n={n}: arm A coefficients == spike S bit for bit: {ga}")
    ok &= ga
    # W0
    c3 = qlib.parse(qlib.run_q(n, 2, 1, 3))
    w0 = "".join(c3["w"])
    assert len(w0) == 336
    raw = qlib.run_q(n, 1, 1, 1, 0, w0, 2, raw=True)
    nb = ntr * 253 * 8
    terms = np.frombuffer(raw[:nb], dtype="<f8").reshape(ntr, 253)
    b = qlib.parse(raw[nb:].decode())
    assert len(b["sums"]) == 253
    # G-B-exact
    mism = 0
    for k in range(253):
        e = math.fsum(terms[:, k].tolist())
        if bits(e) != int(b["sums"][k], 16):
            mism += 1
    print(f"G-B-exact n={n}: {253 - mism}/253 sums of arm B bit-identical to math.fsum of the dumped terms ({ntr} rows)")
    ok &= mism == 0
    # G-C-order
    c = qlib.parse(qlib.run_q(n, 2, 1, 1, 0, w0, 1))
    nbk = (ntr + K - 1) // K
    part = np.zeros((nbk, 253))
    full = ntr // K
    if full:
        blk = terms[:full * K].reshape(full, K, 253)
        acc = np.zeros((full, 253))
        for j in range(K):
            acc = acc + blk[:, j, :]
        part[:full] = acc
    if full < nbk:
        acc = np.zeros(253)
        for i in range(full * K, ntr):
            acc = acc + terms[i]
        part[full] = acc
    ref_c = oracle_linalg.tree(part)
    mc = sum(1 for k in range(253) if bits(ref_c[k]) != int(c["sums"][k], 16))
    print(f"G-C-order n={n}: {253 - mc}/253 sums of arm C bit-identical to block rule (K=256) + tree on the dumped terms")
    ok &= mc == 0
    # G-A-order
    a1 = qlib.parse(qlib.run_q(n, 0, 1, 1, 0, w0, 1))
    acc = np.zeros(253)
    for i in range(ntr):
        acc = acc + terms[i]
    ma = sum(1 for k in range(253) if bits(acc[k]) != int(a1["sums"][k], 16))
    print(f"G-A-order n={n}: {253 - ma}/253 sums of arm A bit-identical to rows ascending from +0.0 on the dumped terms")
    ok &= ma == 0
    # how different are the three sets of sums (info)
    def u(x, y):
        return max(qlib.ulps(qlib.hex_to_float(p), qlib.hex_to_float(q)) for p, q in zip(x, y))
    print(f"info: max ulps over the 253 sums: A vs exact {u(a1['sums'], b['sums'])}, C vs exact {u(c['sums'], b['sums'])}")
    print("ALL GATES PASSED" if ok else "GATE FAILED (a result below is void)")
    sys.exit(0 if ok else 1)


main()

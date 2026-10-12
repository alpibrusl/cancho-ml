#!/usr/bin/env python3
"""G-oracle (docs/spikes.md section 4): cancho arm A vs numpy, coefficients to 1e-9 relative.

usage: check_oracle.py --bin SPIKE_L_EXE [--data DIR] [--lr 0.5] [--iters 100] [--sets n1000 n100000] [--order blas|seq]
                       [--selftest]

Gate: for each of the 21 coefficients (20 weights, then the bias), |cancho - numpy| <= TOL * |numpy|  (TOL = 1e-9; a
numpy coefficient of exactly 0 must be matched exactly). Exit 0 if every set passes, 1 if any coefficient fails.
Also prints the largest relative difference and its ulp distance.

--selftest runs the gate against MUTANTS first (the gate must be able to fail): the cancho side is run with one
fewer iteration, with the learning rate scaled by 1+1e-6, 1+1e-8, 1+1e-10, 1+1e-12, and the numpy side with a weight
flipped by one part in 1e-8 / 1e-10 (a corrupted coefficient file). Mutants at or above the gate's resolution
must be REJECTED; the table says which scale the gate resolves. The selftest exits 1 if a mutant that must be
rejected is accepted, and 0 otherwise (then the real comparison runs).
"""
import argparse, os, struct, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oracle_l

TOL = 1e-9

def bits_of(x): return struct.unpack("<Q", struct.pack("<d", x))[0]
def of_bits(u): return struct.unpack("<d", struct.pack("<Q", u))[0]

def run_cancho(exe, d, n, lr, iters):
    x, y = os.path.join(d, "x.bin"), os.path.join(d, "y.bin")
    out = subprocess.run([exe, x, y, str(n), "%016x" % bits_of(lr), str(iters)], capture_output=True, text=True, check=True).stdout
    lines = [l for l in out.splitlines() if not l.startswith("#")]
    if len(lines) != 21:
        raise SystemExit("cancho output is not 21 coefficient lines:\n" + out)
    return np.array([of_bits(int(l, 16)) for l in lines])

def run_numpy(d, lr, iters, order):
    X, y = oracle_l.load(d)
    w, b = oracle_l.fit(X[:oracle_l.split(y.size)], y[:oracle_l.split(y.size)], lr, iters, order)
    return np.concatenate([w, [b]])

def compare(c, o):
    """-> (ok, worst relative difference, worst ulp distance)"""
    worst, ulps, ok = 0.0, 0, True
    for a, b in zip(c, o):
        d = abs(a - b)
        rel = 0.0 if d == 0 else (float("inf") if b == 0 else d / abs(b))
        if rel > TOL: ok = False
        worst = max(worst, rel)
        ulps = max(ulps, abs(bits_of(a) - bits_of(b)) if (a < 0) == (b < 0) else 1 << 63)
    return ok, worst, ulps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--data", default="/Users/alfonso/Workspace/alpibrusl/lexsys/ml-data")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--sets", nargs="+", default=["n1000", "n100000"])
    ap.add_argument("--order", default="blas")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sizes = {s: int(s[1:]) for s in a.sets}

    if a.selftest:
        d = os.path.join(a.data, a.sets[0]); n = sizes[a.sets[0]]
        base_o = run_numpy(d, a.lr, a.iters, a.order)
        print("selftest on %s (lr %g, %d iters); '-' = must be rejected, '+' = may be accepted" % (a.sets[0], a.lr, a.iters))
        muts = [("cancho: one fewer iteration", "-", lambda: run_cancho(a.bin, d, n, a.lr, a.iters - 1), base_o)]
        for e in (1e-6, 1e-8, 1e-10, 1e-12):
            muts.append(("cancho: lr * (1+%g)" % e, "-" if e >= 1e-8 else "+", (lambda e=e: run_cancho(a.bin, d, n, a.lr * (1 + e), a.iters)), base_o))
        good_c = run_cancho(a.bin, d, n, a.lr, a.iters)
        for e in (1e-6, 1e-8):
            bad_o = base_o.copy(); bad_o[7] *= 1 + e
            muts.append(("numpy: w[7] * (1+%g)" % e, "-", (lambda g=good_c: g), bad_o))
        bad_o = base_o.copy(); bad_o[20] = of_bits(bits_of(bad_o[20]) + 100)
        muts.append(("numpy: bias + 100 ulp", "+", (lambda g=good_c: g), bad_o))
        failed = False
        for name, must, f, o in muts:
            ok, rel, ulps = compare(f(), o)
            verdict = "ACCEPTED" if ok else "rejected"
            flag = "" if (ok is False or must == "+") else "  <-- SURVIVOR: gate void"
            if flag: failed = True
            print("  %-34s %-9s worst rel %.3e  worst ulps %d%s" % (name, verdict, rel, ulps, flag))
        if failed: sys.exit(1)

    rc = 0
    for s in a.sets:
        d = os.path.join(a.data, s); n = sizes[s]
        c = run_cancho(a.bin, d, n, a.lr, a.iters)
        o = run_numpy(d, a.lr, a.iters, a.order)
        ok, rel, ulps = compare(c, o)
        print("%-9s order=%s lr=%g iters=%d  %s  worst rel %.3e  worst ulps %d" % (s, a.order, a.lr, a.iters, "PASS" if ok else "FAIL", rel, ulps))
        if not ok: rc = 1
    sys.exit(rc)

if __name__ == "__main__":
    main()

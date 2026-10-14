#!/usr/bin/env python3
"""G-det and G-mutants of docs/spikes.md section 15.3.

usage: gate_q.py N [THREADS ...]       (default threads 1 2 4 8 16)

For arms B and C: the model bytes (21 coefficients + 6 loss lines) at every thread count must be identical (and equal to the
one-thread bytes); the 253 raw sums of one step from a fixed W0 likewise. Then the same for the thread-order MUTANT (MUT=1), which must
differ from the real bytes at more than one thread count (it equals them at one thread by construction). Exit 1 if G-det fails or a mutant survives."""
import sys
import qlib


def main():
    n = int(sys.argv[1])
    ts = [int(x) for x in sys.argv[2:]] or [1, 2, 4, 8, 16]
    c3 = qlib.parse(qlib.run_q(n, 2, 1, 3))
    w0 = "".join(c3["w"])
    ok = True
    for arm, name in ((1, "B"), (2, "C")):
        real, mut, rsum, msum, wr, wm = {}, {}, {}, {}, {}, {}
        for t in ts:
            r = qlib.parse(qlib.run_q(n, arm, t, 6, 0)); real[t] = qlib.model_bytes(r); wr[t] = " ".join(r["w"])
            r = qlib.parse(qlib.run_q(n, arm, t, 6, 1)); mut[t] = qlib.model_bytes(r); wm[t] = " ".join(r["w"])
            rsum[t] = " ".join(qlib.parse(qlib.run_q(n, arm, t, 1, 0, w0, 1))["sums"])
            msum[t] = " ".join(qlib.parse(qlib.run_q(n, arm, t, 1, 1, w0, 1))["sums"])
        ref, refs = real[ts[0]], rsum[ts[0]]
        same = all(real[t] == ref for t in ts)
        same_s = all(rsum[t] == refs for t in ts)
        print(f"G-det  arm {name} n={n}: model bytes identical at threads {ts}: {same}; 253 raw sums from W0 identical: {same_s}")
        ok &= same and same_s
        mdiff = [t for t in ts if mut[t] != ref]
        sdiff = [t for t in ts if msum[t] != refs]
        eq1 = [t for t in ts if mut[t] == ref]
        print(f"G-mutants arm {name} n={n}: thread-order mutant differs from the real model bytes at threads {mdiff} (equal at {eq1}); raw sums differ at {sdiff}")
        killed = len([t for t in mdiff if t > 1]) >= 1 and len([t for t in sdiff if t > 1]) >= 1
        allgt1 = all(t in sdiff for t in ts if t > 1)
        print(f"        mutant killed: {killed}; fails at every thread count > 1 (raw sums): {allgt1}; (model bytes: {all(t in mdiff for t in ts if t > 1)}); coefficients alone differ at {[t for t in ts if wm[t] != wr[ts[0]]]}")
        ok &= killed
    print("G-det / G-mutants PASSED" if ok else "G-det / G-mutants FAILED")
    sys.exit(0 if ok else 1)


main()

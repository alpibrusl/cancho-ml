#!/usr/bin/env python3
"""Spike F oracle: the algorithm of docs/spikes.md section 13.1, in numpy/Python, EXACTLY.
usage: oracle_f.py DATADIR SEED OUT        (writes the E/N/P lines of 13.1.8, prints AUC, tie count and times)
       oracle_f.py --vectors               (SplitMix64 / derive / uniform vectors, same text as `spike_f --vectors`)
Options: --mutant NAME (tie|bin|boot|dir|depth|mtry): the oracle itself with a one-site change, used to test the checker.
"""
import sys, time
import numpy as np

M = (1 << 64) - 1
G = 0x9E3779B97F4A7C15
C1 = 0xBF58476D1CE4E5B9
C2 = 0x94D049BB133111EB
D, B, T, MAXD, MTRY = 20, 64, 10, 8, 4

def mix(z):
    z &= M
    z = ((z ^ (z >> 30)) * C1) & M
    z = ((z ^ (z >> 27)) * C2) & M
    return z ^ (z >> 31)

class SM:
    def __init__(self, s): self.s = s & M
    def next(self):
        self.s = (self.s + G) & M
        return mix(self.s)
    def uniform(self, m): return (self.next() >> 1) % m

def derive(seed, *cs):
    h = seed & M
    for c in cs: h = mix((h ^ c) + G)
    return h

def vectors():
    out = []
    for lab, s in (("0", 0), ("1234567", 1234567), ("max", M)):
        r = SM(s)
        for k in range(5): out.append("sm %s %d %016x" % (lab, k + 1, r.next()))
    for args in ((42, 1, 0), (42, 2, 3, 77), (20261012, 1, 9)):
        out.append("derive %s %016x" % (",".join(map(str, args)), derive(*args)))
    for lab, args, m in (("a1000", (42, 1, 0), 1000), ("b20", (42, 2, 3, 77), 20)):
        r = SM(derive(*args))
        for k in range(5): out.append("uni %s %d %d" % (lab, k + 1, r.uniform(m)))
    return out

def bootstrap(seed, t, n, mut):
    s0 = derive(seed, 1, t)
    k = np.arange(1, n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(s0) + k * np.uint64(G)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(C1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(C2)
        z = z ^ (z >> np.uint64(31))
    d = ((z >> np.uint64(1)) % np.uint64(n)).astype(np.int64)
    if mut == "boot" and t == 0: d[0] = (d[0] + 1) % n
    return d

def main():
    if sys.argv[1] == "--vectors":
        print("\n".join(vectors())); return
    mut = None
    if "--mutant" in sys.argv:
        i = sys.argv.index("--mutant"); mut = sys.argv[i + 1]; del sys.argv[i:i + 2]
    d, seed, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    maxd = 7 if mut == "depth" else MAXD
    mtry = 3 if mut == "mtry" else MTRY
    X = np.fromfile(d + "/x.bin", dtype="<f8"); y = np.fromfile(d + "/y.bin", dtype="<f8")
    n_all = y.size; X = X.reshape(n_all, D)
    ntr = n_all - n_all // 5
    assert ntr <= 1 << 20 and np.isfinite(X).all() and set(np.unique(y)) <= {0.0, 1.0}
    Xtr, Xte = X[:ntr], X[ntr:]; ytr = y[:ntr].astype(np.int64)
    t0 = time.perf_counter()
    edges = np.empty((D, B - 1))
    bins = np.empty((D, ntr), dtype=np.uint8)
    for f in range(D):
        v = np.sort(Xtr[:, f]); e = np.array([v[(k * ntr) // B] for k in range(1, B)])
        edges[f] = e
        bins[f] = (np.searchsorted(e, Xtr[:, f], side="left" if mut == "bin" else "right")).astype(np.uint8)
    ties = 0
    trees = []
    for t in range(T):
        rows0 = bootstrap(seed, t, ntr, mut)
        nodes = {}
        stack = [(1, rows0, 0)]
        while stack:
            nid, rows, dep = stack.pop()
            yr = ytr[rows]; m = rows.size; n1 = int(yr.sum()); n0 = m - n1
            leaf = dep == maxd or n0 == 0 or n1 == 0
            best = None
            if not leaf:
                r = SM(derive(seed, 2, t, nid))
                a = list(range(D))
                for i in range(mtry):
                    j = i + r.uniform(D - i); a[i], a[j] = a[j], a[i]
                sub = sorted(a[:mtry])
                for f in sub:
                    c = np.bincount(bins[f][rows].astype(np.int64) * 2 + yr, minlength=2 * B)
                    pos = c[1::2]; cnt = c[0::2] + pos
                    NL = np.cumsum(cnt)[:B - 1].tolist(); PL = np.cumsum(pos)[:B - 1].tolist()
                    for b in range(B - 1):
                        nl, pl = NL[b], PL[b]; nr = m - nl
                        if nl < 1 or nr < 1: continue
                        pr = n1 - pl; ql = nl - pl; qr = nr - pr
                        N = (pl * pl + ql * ql) * nr + (pr * pr + qr * qr) * nl; Dn = nl * nr
                        if best is None: best = (N, Dn, f, b, nl, pl); continue
                        lhs, rhs = N * best[1], best[0] * Dn
                        if lhs == rhs: ties += 1
                        if lhs > rhs or (mut == "tie" and lhs == rhs): best = (N, Dn, f, b, nl, pl)
            if best is None:
                nodes[nid] = (-1, -1, n0, n1); continue
            _, _, f, b, nl, pl = best
            nodes[nid] = (f, b, n0, n1)
            goleft = bins[f][rows] <= b
            if mut == "dir": goleft = ~goleft
            stack.append((2 * nid, rows[goleft], dep + 1)); stack.append((2 * nid + 1, rows[~goleft], dep + 1))
        trees.append(nodes)
    fit_s = time.perf_counter() - t0
    # predict
    nte = Xte.shape[0]; acc = np.zeros(nte)
    for t in range(T):
        nodes = trees[t]
        ids = np.ones(nte, dtype=np.int64); prob = np.zeros(1024); feat = np.full(1024, -1); thr = np.zeros(1024)
        for nid, (f, b, n0, n1) in nodes.items():
            prob[nid] = n1 / (n0 + n1); feat[nid] = f
            if f >= 0: thr[nid] = edges[f][b]      # e[f][b+1] in 1-based: index b
        for _ in range(maxd + 1):
            fe = feat[ids]; act = fe >= 0
            if not act.any(): break
            xv = Xte[np.arange(nte), np.where(act, fe, 0)]
            left = xv < thr[ids]
            if mut == "dir": left = ~left
            ids = np.where(act, np.where(left, 2 * ids, 2 * ids + 1), ids)
        acc = acc + prob[ids]
    pf = acc / 10.0
    with open(outp, "w") as o:
        for f in range(D):
            for k in range(1, B): o.write("E %d %d %016x\n" % (f, k, int(edges[f][k - 1].view(np.uint64))))
        for t in range(T):
            for nid in sorted(trees[t]):
                f, b, n0, n1 = trees[t][nid]; o.write("N %d %d %d %d %d %d\n" % (t, nid, f, b, n0, n1))
        for i, v in enumerate(pf.view(np.uint64).tolist()): o.write("P %d %016x\n" % (i, v))
    from sklearn.metrics import roc_auc_score
    print("oracle mutant=%s ntr=%d nte=%d ties=%d fit_s=%.2f nodes=%d test_auc=%.6f" % (mut, ntr, nte, ties, fit_s, sum(len(t) for t in trees), roc_auc_score(y[ntr:], pf)))

if __name__ == "__main__":
    main()

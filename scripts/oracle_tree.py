#!/usr/bin/env python3
"""The oracle of the tree gates: docs/models-tree.md section 2, in Python/numpy, EXACTLY (separate code from lib/models/tree).
usage: oracle_tree.py x=X.bin y=Y.bin [miss=M.bin] n=N d=D [ntr=T] task=0|1 seed=S trees=T depth=D leaf=L bins=B mtry=K boot=0|1 out=FILE
Writes the E / N / P lines that `tree-check mode=fit` prints (byte for byte the same), and prints one summary line:
    ties = comparisons of two exactly equal scores (the tie rule is exercised only if > 0)
    v1 = splits whose winner sends the missing rows left
    equal_v = candidates where the variant 1 equals the variant 0 (no missing row), to show the skipped duplicate is harmless.
Exact integer arithmetic (Python ints) for the Gini comparison; IEEE float64 sequential sums (np.bincount with weights, a Python
loop over bins, np.add.accumulate) for regression; np.sum is never used (pairwise)."""
import struct
import sys
import numpy as np

M64 = (1 << 64) - 1
GOLD = 0x9E3779B97F4A7C15
C1 = 0xBF58476D1CE4E5B9
C2 = 0x94D049BB133111EB


def mix(z):
    z &= M64
    z = ((z ^ (z >> 30)) * C1) & M64
    z = ((z ^ (z >> 27)) * C2) & M64
    return z ^ (z >> 31)


def derive(h, c):
    return mix((h ^ c) + GOLD)


class SM:
    def __init__(self, s):
        self.s = s & M64

    def uniform(self, m):
        self.s = (self.s + GOLD) & M64
        return (mix(self.s) >> 1) % m


def bits_hex(f):
    return "%016x" % int(np.float64(f).view(np.uint64))


def arg_int(a, k, d=None):
    return int(a[k]) if k in a else d


class Forest:
    pass


def key_array(x):
    u = x.view(np.uint64)
    return np.where((u >> np.uint64(63)) == 1, ~u, u ^ np.uint64(1 << 63))


def float_of_key(k):
    k = int(k)
    bits = (~k & M64) if (k >> 63) == 0 else (k ^ (1 << 63))
    return np.uint64(bits).view(np.float64)


def fit(Xtr, mtr, ytr, task, T, maxd, ML, B, mtry, boot, seed, stats):
    n, d = Xtr.shape
    edges = np.zeros((d, B - 1))
    binsm = np.zeros((d, n), dtype=np.int64)
    for f in range(d):
        present = ~mtr[:, f]
        col = Xtr[present, f]
        nv = col.size
        if nv > 0:
            ks = np.sort(key_array(np.ascontiguousarray(col)))
            for k in range(1, B):
                edges[f, k - 1] = float_of_key(ks[(k * nv) // B])
        x = np.where(present, Xtr[:, f], 0.0)
        b = np.searchsorted(edges[f], x, side="right")
        binsm[f] = np.where(present, b, B)
    trees = []
    for t in range(T):
        if boot:
            s0 = derive(derive(seed, 1), t)
            k = np.arange(1, n + 1, dtype=np.uint64)
            with np.errstate(over="ignore"):
                z = np.uint64(s0) + k * np.uint64(GOLD)
                z = (z ^ (z >> np.uint64(30))) * np.uint64(C1)
                z = (z ^ (z >> np.uint64(27))) * np.uint64(C2)
                z = z ^ (z >> np.uint64(31))
            rows0 = ((z >> np.uint64(1)) % np.uint64(n)).astype(np.int64)
        else:
            rows0 = np.arange(n, dtype=np.int64)
        nodes = []  # [feat, bin, dleft, left, right, na, nb, val]
        stack = [(1, rows0, 0, -1, 0)]
        while stack:
            hid, rows, dep, parent, side = stack.pop()
            ni = len(nodes)
            nodes.append(None)
            if parent >= 0:
                nodes[parent][3 if side == 0 else 4] = ni
            m = rows.size
            if task == 0:
                yr = ytr[rows].astype(np.int64)
                n1 = int(yr.sum())
                n0 = m - n1
                can = dep < maxd and m >= 2 * ML and n0 > 0 and n1 > 0
            else:
                yv = ytr[rows]
                sm = float(np.add.accumulate(np.concatenate(([0.0], yv)))[-1])
                can = dep < maxd and m >= 2 * ML and bool(yv.min() != yv.max())
            best = None
            if can:
                r = SM(derive(derive(derive(seed, 2), t), hid))
                a = list(range(d))
                for i in range(mtry):
                    j = i + r.uniform(d - i)
                    a[i], a[j] = a[j], a[i]
                sub = sorted(a[:mtry])
                for f in sub:
                    bf = binsm[f][rows]
                    if task == 0:
                        c = np.bincount(bf * 2 + yr, minlength=2 * (B + 1))
                        pos = c[1::2]
                        cnt = c[0::2] + pos
                        NL = np.cumsum(cnt[: B - 1]).tolist()
                        PL = np.cumsum(pos[: B - 1]).tolist()
                        mc, mp = int(cnt[B]), int(pos[B])
                        for b in range(B - 1):
                            for dl in (0, 1):
                                nl, pl = NL[b], PL[b]
                                if dl == 1:
                                    nl += mc
                                    pl += mp
                                nr = m - nl
                                if nl < ML or nr < ML:
                                    continue
                                if dl == 1 and mc == 0:
                                    stats["equal_v"] += 1
                                pr = n1 - pl
                                ql = nl - pl
                                qr = nr - pr
                                N = (pl * pl + ql * ql) * nr + (pr * pr + qr * qr) * nl
                                Dn = nl * nr
                                if best is None:
                                    best = (N, Dn, f, b, dl, nl, pl)
                                    continue
                                lhs, rhs = N * best[1], best[0] * Dn
                                if lhs == rhs and not (dl == 1 and mc == 0):
                                    stats["ties"] += 1
                                if lhs > rhs:
                                    best = (N, Dn, f, b, dl, nl, pl)
                    else:
                        cnt = np.bincount(bf, minlength=B + 1).tolist()
                        sums = np.bincount(bf, weights=yv, minlength=B + 1).tolist()
                        mc, ms = cnt[B], sums[B]
                        nlc = 0
                        sl = 0.0
                        for b in range(B - 1):
                            nlc += cnt[b]
                            sl = sl + sums[b]
                            for dl in (0, 1):
                                nl, SL = nlc, sl
                                if dl == 1:
                                    nl = nlc + mc
                                    SL = sl + ms
                                nr = m - nl
                                if nl < ML or nr < ML:
                                    continue
                                if dl == 1 and mc == 0:
                                    stats["equal_v"] += 1
                                SR = sm - SL
                                sc = SL * SL / float(nl) + SR * SR / float(nr)
                                if best is None:
                                    best = (sc, f, b, dl, nl)
                                    continue
                                if sc == best[0] and not (dl == 1 and mc == 0):
                                    stats["ties"] += 1
                                if sc > best[0]:
                                    best = (sc, f, b, dl, nl)
            if best is None:
                if task == 0:
                    nodes[ni] = [-1, -1, 0, -1, -1, n0, n1, n1 / (n0 + n1)]
                else:
                    nodes[ni] = [-1, -1, 0, -1, -1, m, 0, sm / float(m)]
                continue
            if task == 0:
                _, _, f, b, dl, nl, pl = best
            else:
                _, f, b, dl, nl = best
            if dl == 1:
                stats["v1"] += 1
            bf = binsm[f][rows]
            goleft = np.where(bf == B, dl == 1, bf <= b)
            nodes[ni] = [f, b, dl, -1, -1, n0 if task == 0 else m, n1 if task == 0 else 0, edges[f, b]]
            stack.append((2 * hid + 1, rows[~goleft], dep + 1, ni, 1))
            stack.append((2 * hid, rows[goleft], dep + 1, ni, 0))
        trees.append(nodes)
    return edges, trees


def payload(trees, task, d, B, maxd, ML, mtry):
    """The model payload of docs/models-tree.md section 7, in Python: `u32 kind = 7` then the ten tensor records (u32 dtype, u32 rank = 1, u64 extent, the
    elements): the header of eight i64, the tree offsets, feat, bin, dleft, left, right, na, nb as i64 and val as f64. `trees` is what `fit` returns."""
    total = sum(len(t) for t in trees)

    def rec(dtype, vals, fmt):
        return struct.pack("<IIQ", dtype, 1, len(vals)) + struct.pack("<%d%s" % (len(vals), fmt), *vals)

    offs = [0]
    for t in trees:
        offs.append(offs[-1] + len(t))
    out = struct.pack("<I", 7) + rec(1, [task, len(trees), d, B, total, maxd, ML, mtry], "q") + rec(1, offs, "q")
    for k in range(7):
        out += rec(1, [int(nd[k]) for t in trees for nd in t], "q")
    out += rec(2, [float(nd[7]) for t in trees for nd in t], "d")
    return out


def predict(trees, Xte, mte, T):
    nte = Xte.shape[0]
    acc = np.zeros(nte)
    for nodes in trees:
        feat = np.array([x[0] for x in nodes])
        dleft = np.array([x[2] for x in nodes])
        left = np.array([x[3] for x in nodes])
        right = np.array([x[4] for x in nodes])
        thr = np.array([x[7] for x in nodes])
        cur = np.zeros(nte, dtype=np.int64)
        while True:
            fe = feat[cur]
            act = fe >= 0
            if not act.any():
                break
            idx = np.arange(nte)
            fc = np.where(act, fe, 0)
            xv = Xte[idx, fc]
            ms = mte[idx, fc]
            gl = np.where(ms, dleft[cur] == 1, xv < thr[cur])
            cur = np.where(act, np.where(gl, left[cur], right[cur]), cur)
        acc = acc + thr[cur]  # leaf value is nodes[.][7]
    return acc / float(T)


def main():
    a = dict(s.split("=", 1) for s in sys.argv[1:])
    n, d = int(a["n"]), int(a["d"])
    ntr = arg_int(a, "ntr", n - n // 5)
    X = np.fromfile(a["x"], "<f8").reshape(n, d)
    y = np.fromfile(a["y"], "<f8")
    mask = np.zeros((n, d), dtype=bool)
    if "miss" in a:
        raw = np.fromfile(a["miss"], dtype=np.uint8)
        st = (n + 7) // 8
        for c in range(d):
            mask[:, c] = np.unpackbits(raw[c * st:(c + 1) * st], bitorder="little")[:n].astype(bool)
    task, T, maxd = int(a["task"]), int(a["trees"]), int(a["depth"])
    ML, B, boot, seed = arg_int(a, "leaf", 1), int(a["bins"]), arg_int(a, "boot", 1), arg_int(a, "seed", 20261012)
    mtry = arg_int(a, "mtry", 0)
    if mtry == 0:
        mtry = max(1, int(d ** 0.5 + 1e-9)) if task == 0 else max(1, d // 3)
        if task == 0:
            while mtry * mtry > d: mtry -= 1
            while (mtry + 1) * (mtry + 1) <= d: mtry += 1
    stats = {"ties": 0, "v1": 0, "equal_v": 0}
    with np.errstate(invalid="ignore"):
        edges, trees = fit(X[:ntr], mask[:ntr], y[:ntr], task, T, maxd, ML, B, mtry, boot, seed, stats)
        pf = predict(trees, X[ntr:], mask[ntr:], T)
    with open(a["out"], "w") as o:
        for f in range(d):
            for k in range(1, B):
                o.write("E %d %d %s\n" % (f, k, bits_hex(edges[f, k - 1])))
        for t, nodes in enumerate(trees):
            for i, nd in enumerate(nodes):
                o.write("N %d %d %d %d %d %d %d %d %d %s\n" % (t, i, nd[0], nd[1], nd[2], nd[3], nd[4], nd[5], nd[6], bits_hex(nd[7])))
        for i, v in enumerate(pf):
            o.write("P %d %s\n" % (i, bits_hex(v)))
    print("oracle ntr=%d nte=%d nodes=%d ties=%d v1=%d equal_v=%d" % (ntr, n - ntr, sum(len(t) for t in trees), stats["ties"], stats["v1"], stats["equal_v"]))


if __name__ == "__main__":
    main()

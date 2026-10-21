#!/usr/bin/env python3
"""The oracle of the boosting library: docs/models-boost.md section 2 in Python/numpy, EXACTLY (separate code from lib/models/boost). It writes the model
PAYLOAD of section 5 byte for byte, and predicts. It calls neither math.exp nor math.log: `exp`, `log` and `log1p` are the ports of std/math.cho that
`oracle_models.py` holds. The arithmetic is IEEE float64 elementwise in numpy (no fma) and every sum is sequential (`np.bincount` with weights and
`np.add.accumulate` add in order; `np.sum` is never used, it is pairwise).

    from oracle_boost import fit, payload, predict_scores
    res = fit(X, M, y, task, R, eta, maxd, ML, B, L, subsample, early, maxnodes, seed)   # X (n, d) float64, M (n, d) bool missing mask, y floats
    payload(res, d, task, ML, maxd, B)                                                 # the bytes of the model section
"""
import struct
import sys
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from oracle_tree import mix, derive, SM, key_array, float_of_key, GOLD, M64, C1, C2  # noqa: E402
from oracle_models import exp_c, log_c, log1p_c  # noqa: E402

HMIN = 0.001
SCALE = 1000000


class LimitNodes(Exception):
    pass


def seqsum(v):
    """a sequential left fold from +0.0"""
    return float(np.add.accumulate(np.concatenate(([0.0], np.asarray(v, dtype=np.float64))))[-1])


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    az = np.abs(z)
    e = exp_c(0.0 - az)
    return np.where(z >= 0.0, 1.0 / (1.0 + e), e / (1.0 + e))


def make_bins(X, M, B):
    n, d = X.shape
    edges = np.zeros((d, B - 1))
    binsm = np.zeros((d, n), dtype=np.int64)
    for f in range(d):
        present = ~M[:, f]
        col = X[present, f]
        nv = col.size
        if nv > 0:
            ks = np.sort(key_array(np.ascontiguousarray(col)))
            for k in range(1, B):
                edges[f, k - 1] = float_of_key(ks[(k * nv) // B])
        x = np.where(present, X[:, f], 0.0)
        b = np.searchsorted(edges[f], x, side="right")
        binsm[f] = np.where(present, b, B)
    return edges, binsm


def rank_grad(F, rel, groups, wts=None):
    """the gradient and the hessian of the pairwise ranking loss (docs/ranking.md section 4). groups: for each group, the row numbers in file order; rel: the relevance of
    every row. For every ordered pair (i, j) of one group with rel[i] > rel[j], i ascending then j ascending, rho = sigmoid(F[j] - F[i]) = 1 / (1 + exp(F[i] - F[j])) and
    tg = w * rho, th = w * (rho * (1 - rho)) with w = wts[i] * wts[j] (or 1): g[i] -= tg, g[j] += tg, h[i] += th, h[j] += th. Every accumulator is a sequential sum in the
    order the pair loop reaches its terms, from +0.0."""
    n = len(F)
    g = np.zeros(n)
    h = np.zeros(n)
    F = np.asarray(F, dtype=np.float64)
    for rows in groups:
        rows = np.asarray(rows, dtype=np.int64)
        for ai in range(rows.size):
            i = int(rows[ai])
            js = rows[rel[rows] < rel[i]]
            if js.size == 0:
                continue
            rho = sigmoid(F[js] - F[i])
            tg = rho
            th = rho * (1.0 - rho)
            if wts is not None:
                w = wts[i] * wts[js]
                tg = w * rho
                th = w * (rho * (1.0 - rho))
            g[i] = np.add.accumulate(np.concatenate(([g[i]], 0.0 - tg)))[-1]
            h[i] = np.add.accumulate(np.concatenate(([h[i]], th)))[-1]
            g[js] = g[js] + tg
            h[js] = h[js] + th
    return g, h


def leaf_value(eta, L, G, H):
    lv = 0.0
    if H + L >= HMIN:
        lv = (0.0 - G) / (H + L)
    return eta * lv


def grow(rows, g, h, binsm, edges, B, maxd, ML, L, eta, cap, stats):
    """one tree over the sample `rows` (ascending): nodes [feat, bin, dleft, left, right, val] in preorder"""
    d = binsm.shape[0]
    nodes = []
    stack = [(rows, 0, -1, 0)]
    while stack:
        rws, dep, parent, side = stack.pop()
        ni = len(nodes)
        nodes.append(None)
        if len(nodes) > cap:
            raise LimitNodes()
        if parent >= 0:
            nodes[parent][3 if side == 0 else 4] = ni
        m = rws.size
        gr, hr = g[rws], h[rws]
        G, H = seqsum(gr), seqsum(hr)
        best = None
        if dep < maxd and m >= 2 * ML:
            for f in range(d):
                bf = binsm[f][rws]
                cnt = np.bincount(bf, minlength=B + 1)
                sg = np.bincount(bf, weights=gr, minlength=B + 1)
                sh = np.bincount(bf, weights=hr, minlength=B + 1)
                mc, mg, mh = int(cnt[B]), float(sg[B]), float(sh[B])
                nlc = np.add.accumulate(cnt[:B - 1])
                cg = np.add.accumulate(sg[:B - 1])
                ch = np.add.accumulate(sh[:B - 1])
                S = np.full((B - 1, 2), -np.inf)
                for dl in (0, 1):
                    if dl == 1 and mc == 0:
                        continue
                    nl = nlc + mc if dl else nlc
                    gl = cg + mg if dl else cg
                    hl = ch + mh if dl else ch
                    nr = m - nl
                    gr_ = G - gl
                    hr_ = H - hl
                    with np.errstate(all="ignore"):
                        sc = gl * gl / (hl + L) + gr_ * gr_ / (hr_ + L)
                    ok = (nl >= ML) & (nr >= ML) & (hl >= HMIN) & (hr_ >= HMIN)
                    S[:, dl] = np.where(ok, sc, -np.inf)
                flat = S.reshape(-1)
                k = int(np.argmax(flat))
                if flat[k] == -np.inf:
                    continue
                if best is None or flat[k] > best[0]:
                    best = (float(flat[k]), f, k // 2, k % 2)
        found = False
        if best is not None:
            with np.errstate(all="ignore"):
                ps = G * G / (H + L)
            found = best[0] - ps > 0.0
        if not found:
            nodes[ni] = [-1, -1, 0, -1, -1, leaf_value(eta, L, G, H)]
            continue
        _, f, b, dl = best
        bf = binsm[f][rws]
        goleft = np.where(bf == B, dl == 1, bf <= b)
        if dl == 1:
            stats["v1"] = stats.get("v1", 0) + 1
        nodes[ni] = [f, b, dl, -1, -1, float(edges[f, b])]
        stack.append((rws[~goleft], dep + 1, ni, 1))
        stack.append((rws[goleft], dep + 1, ni, 0))
    return nodes


def walk_bins(nodes, binsm, B):
    n = binsm.shape[1]
    feat = np.array([x[0] for x in nodes])
    bn = np.array([x[1] for x in nodes])
    dleft = np.array([x[2] for x in nodes])
    left = np.array([x[3] for x in nodes])
    right = np.array([x[4] for x in nodes])
    val = np.array([x[5] for x in nodes])
    cur = np.zeros(n, dtype=np.int64)
    idx = np.arange(n)
    while True:
        fe = feat[cur]
        act = fe >= 0
        if not act.any():
            break
        b = binsm[np.where(act, fe, 0), idx]
        gl = np.where(b == B, dleft[cur] == 1, b <= bn[cur])
        cur = np.where(act, np.where(gl, left[cur], right[cur]), cur)
    return val[cur]


def row_loss(task, F, y):
    if task == 1:
        return (F - y) * (F - y)
    mx = np.where(F < 0.0, 0.0, F)
    af = np.where(F < 0.0, 0.0 - F, F)
    return (mx - y * F) + log1p_c(exp_c(0.0 - af))


def node_cap(maxnodes, m, maxd):
    return min(maxnodes, 2 * m - 1, (1 << (maxd + 1)) - 1)


def fit(X, M, y, task, R, eta, maxd, ML, B, L, subsample, early, maxnodes, seed, stats=None, wts=None):
    """-> dict(base, trees, kept, requested, subk, edges); task 0 classification (y in {0.0, 1.0}), 1 regression. Raises LimitNodes.
    wts: None, or the weight of every row (docs/weights.md section 2.6): the base score and the gradients of the fit rows are weighted, the validation loss is not."""
    if stats is None:
        stats = {}
    n, d = X.shape
    edges, binsm = make_bins(X, M, B)
    K = int(subsample * 1000000.0 + 0.5)
    nv = n // 10 if early > 0 else 0
    if nv == 0:
        frows = np.arange(n, dtype=np.int64)
        vrows = np.zeros(0, dtype=np.int64)
    else:
        sm = SM(derive(seed, 10))
        perm = list(range(n))
        for i in range(nv):
            j = i + sm.uniform(n - i)
            perm[i], perm[j] = perm[j], perm[i]
        mark = np.zeros(n, dtype=bool)
        mark[perm[:nv]] = True
        vrows = np.nonzero(mark)[0].astype(np.int64)
        frows = np.nonzero(~mark)[0].astype(np.int64)
    m = frows.size
    yf = np.asarray(y, dtype=np.float64)
    wf = None if wts is None else np.asarray(wts, dtype=np.float64)
    if task == 1:
        if wf is None:
            base = seqsum(yf[frows]) / float(m)
        else:
            base = seqsum(wf[frows] * yf[frows]) / seqsum(wf[frows])
    elif wf is None:
        n1 = int(yf[frows].sum())
        assert 0 < n1 < m
        base = float(log_c(np.array([float(n1) / float(m - n1)]))[0])
    else:
        w1 = seqsum(wf[frows][yf[frows] == 1.0])
        w0 = seqsum(wf[frows][yf[frows] == 0.0])
        assert w1 > 0.0 and w0 > 0.0
        base = float(log_c(np.array([w1 / w0]))[0])
    F = np.full(n, base)
    cap = node_cap(maxnodes, m, maxd)
    trees = []
    best, best_round, since = 0.0, 0, 0
    for r in range(R):
        g = np.zeros(n)
        h = np.zeros(n)
        if task == 1:
            g[frows] = F[frows] - yf[frows]
            h[frows] = 1.0
        else:
            p = sigmoid(F[frows])
            g[frows] = p - yf[frows]
            h[frows] = p * (1.0 - p)
        if wf is not None:
            g[frows] = wf[frows] * g[frows]
            h[frows] = wf[frows] * h[frows]
        if K == SCALE:
            sample = frows
        else:
            s0 = derive(derive(seed, 9), r)
            k = np.arange(1, m + 1, dtype=np.uint64)
            with np.errstate(over="ignore"):
                z = np.uint64(s0) + k * np.uint64(GOLD)
                z = (z ^ (z >> np.uint64(30))) * np.uint64(C1)
                z = (z ^ (z >> np.uint64(27))) * np.uint64(C2)
                z = z ^ (z >> np.uint64(31))
            keep = ((z >> np.uint64(1)) % np.uint64(SCALE)) < np.uint64(K)
            sample = frows[keep]
        nodes = grow(sample, g, h, binsm, edges, B, maxd, ML, L, eta, cap, stats)
        trees.append(nodes)
        F = F + walk_bins(nodes, binsm, B)
        assert np.all(np.isfinite(F))
        if early > 0:
            vl = seqsum(row_loss(task, F[vrows], yf[vrows])) / float(nv)
            if best_round == 0 or vl < best:
                best, best_round, since = vl, r + 1, 0
            else:
                since += 1
                if since >= early:
                    break
    kept = best_round if early > 0 else len(trees)
    stats["rounds_run"] = len(trees)
    return {"base": base, "trees": trees[:kept], "kept": kept, "requested": R, "subk": K, "edges": edges, "eta": eta, "L": L, "early": early,
            "task": task, "maxd": maxd, "ML": ML, "B": B}


def payload(res, d):
    """the bytes of the model section (docs/models-boost.md section 5): u32 kind = 8, then nine tensor records"""
    trees = res["trees"]
    total = sum(len(t) for t in trees)

    def rec(dtype, vals, fmt):
        return struct.pack("<IIQ", dtype, 1, len(vals)) + struct.pack("<%d%s" % (len(vals), fmt), *vals)

    offs = [0]
    for t in trees:
        offs.append(offs[-1] + len(t))
    out = struct.pack("<I", 8)
    out += rec(1, [res["task"], len(trees), d, res["B"], total, res["maxd"], res["ML"], res["early"], res["requested"], res["subk"]], "q")
    out += rec(2, [res["base"], res["eta"], res["L"]], "d")
    out += rec(1, offs, "q")
    for k in range(5):
        out += rec(1, [int(nd[k]) for t in trees for nd in t], "q")
    out += rec(2, [float(nd[5]) for t in trees for nd in t], "d")
    return out


def predict_scores(res, X, M):
    """the score of each row: base, then the leaf value of each tree added in order"""
    n = X.shape[0]
    acc = np.full(n, res["base"])
    idx = np.arange(n)
    for nodes in res["trees"]:
        feat = np.array([x[0] for x in nodes])
        dleft = np.array([x[2] for x in nodes])
        left = np.array([x[3] for x in nodes])
        right = np.array([x[4] for x in nodes])
        thr = np.array([x[5] for x in nodes])
        cur = np.zeros(n, dtype=np.int64)
        while True:
            fe = feat[cur]
            act = fe >= 0
            if not act.any():
                break
            fc = np.where(act, fe, 0)
            xv = X[idx, fc]
            ms = M[idx, fc]
            with np.errstate(invalid="ignore"):
                gl = np.where(ms, dleft[cur] == 1, xv < thr[cur])
            cur = np.where(act, np.where(gl, left[cur], right[cur]), cur)
        acc = acc + thr[cur]
    return acc

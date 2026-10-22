#!/usr/bin/env python3
"""Gate RK3 of docs/ranking.md: the gradient and the hessian of the pairwise ranking loss (section 4) against finite differences of the loss itself, on small groups.

    python3 scripts/gate_rank_grad.py            the oracle's rank_grad against the numerical derivatives, and the controls (a wrong gradient must be rejected)
The loss is  sum over pairs (i, j) of one group with rel_i > rel_j of  w_ij * log(1 + exp(-(F_i - F_j))). The bounds are fixed here, before any run:
    gradient: central difference with step 1e-6, relative error at most 1e-6 (an absolute floor of 1e-9)
    hessian (the diagonal): second central difference of the loss with step 1e-3, relative error at most 1e-5 (an absolute floor of 1e-7)
Exit 1 on any failure."""
import math
import pathlib
import random
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_boost as ob  # noqa: E402

G_STEP, G_REL, G_ABS = 1e-6, 1e-6, 1e-9
H_STEP, H_REL, H_ABS = 1e-3, 1e-5, 1e-7


def loss(F, rel, groups, wts):
    total = 0.0
    for rows in groups:
        for i in rows:
            for j in rows:
                if rel[i] > rel[j]:
                    w = 1.0 if wts is None else wts[i] * wts[j]
                    z = F[i] - F[j]
                    total += w * (max(-z, 0.0) + math.log1p(math.exp(-abs(z))))
    return total


def cases():
    rng = random.Random(7)
    for c in range(60):
        ng = rng.choice([1, 2, 3])
        groups, rel = [], []
        n = 0
        for _ in range(ng):
            size = rng.choice([1, 2, 3, 5, 8])
            groups.append(list(range(n, n + size)))
            rel += [rng.choice([0, 0, 1, 2, 3]) for _ in range(size)]
            n += size
        # rows of a group need not be adjacent in the file: shuffle the row numbers inside the groups' index lists
        perm = list(range(n))
        rng.shuffle(perm)
        groups = [[perm[i] for i in g] for g in groups]
        relv = [0] * n
        k = 0
        for rows in groups:
            for i in rows:
                relv[i] = rel[k]
                k += 1
        scale = rng.choice([0.01, 1.0, 5.0])
        F = [rng.uniform(-1, 1) * scale for _ in range(n)]
        if c % 7 == 0 and n >= 2:
            F[1] = F[0]                       # a pair with equal scores
        if c % 11 == 0:
            F = [0.0] * n                      # every score equal
        wts = None if c % 3 else [rng.choice([0.0, 0.5, 1.0, 2.0, 3.0]) for _ in range(n)]
        yield F, relv, groups, wts


def numerical(F, rel, groups, wts):
    n = len(F)
    g = []
    h = []
    base = loss(F, rel, groups, wts)
    for i in range(n):
        Fp = list(F); Fm = list(F)
        Fp[i] += G_STEP; Fm[i] -= G_STEP
        g.append((loss(Fp, rel, groups, wts) - loss(Fm, rel, groups, wts)) / (2 * G_STEP))
        Fp = list(F); Fm = list(F)
        Fp[i] += H_STEP; Fm[i] -= H_STEP
        h.append((loss(Fp, rel, groups, wts) - 2 * base + loss(Fm, rel, groups, wts)) / (H_STEP * H_STEP))
    return np.array(g), np.array(h)


def close(a, b, rel_tol, abs_tol):
    return bool(np.all(np.abs(a - b) <= rel_tol * np.maximum(np.abs(a), np.abs(b)) + abs_tol))


def judge(grad_fn):
    bad = 0
    for F, rel, groups, wts in cases():
        g, h = grad_fn(np.array(F), np.array(rel), groups, None if wts is None else np.array(wts))
        ng, nh = numerical(F, rel, groups, wts)
        if not close(g, ng, G_REL, G_ABS) or not close(h, nh, H_REL, H_ABS):
            bad += 1
    return bad


def main():
    bad = judge(ob.rank_grad)
    print("gate_rank_grad: oracle gradient and hessian against finite differences: %s" % ("all 60 cases agree" if not bad else "%d of 60 cases disagree" % bad))
    rc = 1 if bad else 0
    # the controls: each of these is wrong in one way, and the gate must reject every one
    def flipped(F, rel, groups, wts):
        g, h = ob.rank_grad(F, rel, groups, wts)
        return 0.0 - g, h

    def hess_rho(F, rel, groups, wts):
        g, h = ob.rank_grad(F, rel, groups, wts)
        return g, h * 2.0

    def once(F, rel, groups, wts):
        g, h = ob.rank_grad(F, rel, groups, wts)
        return g * 0.5, h

    for name, fn in (("a sign flipped", flipped), ("the hessian doubled", hess_rho), ("a gradient halved", once)):
        if judge(fn) == 0:
            print("FAIL control: %s was not rejected" % name)
            rc = 1
        else:
            print("control rejected: %s" % name)
    return rc


if __name__ == "__main__":
    sys.exit(main())

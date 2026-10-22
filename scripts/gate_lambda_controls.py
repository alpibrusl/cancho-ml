#!/usr/bin/env python3
"""Gate LR2 of docs/ranking.md: the LambdaRank weights (section 4) of the oracle against values worked out by hand and against a second, direct implementation with Python's own log2.

    python3 scripts/gate_lambda_controls.py
The literal cases are written here before the run; the bound of the random cases is a relative error of 1e-12 (the oracle uses the port of std.math's log, this file math.log2). A wrong
version of the weights (the discount log2(r), the gain 2^rel, the ranks ascending) must be rejected. Exit 1 on any failure."""
import math
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import oracle_boost as ob  # noqa: E402

BOUND = 1e-12


def direct(F, rel, groups, wts=None, disc=lambda r: 1.0 / math.log2(r + 1), gain=lambda v: 2.0 ** v - 1.0, descending=True):
    n = len(F)
    g, h = [0.0] * n, [0.0] * n
    for rows in groups:
        order = sorted(rows, key=lambda i: (-F[i] if descending else F[i], i))
        rank = {i: p + 1 for p, i in enumerate(order)}
        ideal = sorted((rel[i] for i in rows), reverse=True)
        idcg = sum(gain(v) * disc(p + 1) for p, v in enumerate(ideal))
        if idcg <= 0.0:
            continue
        for i in rows:
            for j in rows:
                if rel[i] > rel[j]:
                    rho = 1.0 / (1.0 + math.exp(F[i] - F[j]))
                    w = abs(gain(rel[i]) - gain(rel[j])) * abs(disc(rank[i]) - disc(rank[j])) / idcg
                    if wts is not None:
                        w *= wts[i] * wts[j]
                    g[i] -= w * rho
                    g[j] += w * rho
                    h[i] += w * rho * (1 - rho)
                    h[j] += w * rho * (1 - rho)
    return g, h


def near(a, b):
    return abs(a - b) <= BOUND * max(1.0, abs(a), abs(b))


def judge(fn):
    """the failures of `fn(F, rel, groups, wts) -> (g, h)` on the literal cases and on 200 random ones"""
    bad = []
    # two rows, rel 1 and 0, equal scores: the first row is rank 1; IDCG = 1; delta = 1 - 1/log2(3); rho = 1/2
    g, h = fn([0.0, 0.0], [1, 0], [[0, 1]], None)
    delta = 1.0 - 1.0 / math.log2(3)
    want = ([-delta / 2, delta / 2], [delta / 4, delta / 4])
    if not all(near(x, y) for x, y in zip(list(g) + list(h), want[0] + want[1])):
        bad.append("two rows: %r %r vs %r" % (list(g), list(h), want))
    # a group where nothing is relevant has no pair, and one of a single relevance has none
    g, h = fn([0.3, -0.2], [0, 0], [[0, 1]], None)
    if any(x != 0.0 for x in list(g) + list(h)):
        bad.append("all-zero group has a pair")
    # three rows, rel 2 1 0, scores putting them at ranks 3 1 2: the pair (rel 2, rel 0) swaps ranks 3 and 2
    F = [-1.0, 1.0, 0.0]
    g, h = fn(F, [2, 1, 0], [[0, 1, 2]], None)
    idcg = 3.0 / math.log2(2) + 1.0 / math.log2(3)
    d01 = abs(3.0 - 1.0) * abs(1 / math.log2(4) - 1 / math.log2(2)) / idcg
    d02 = abs(3.0 - 0.0) * abs(1 / math.log2(4) - 1 / math.log2(3)) / idcg
    d12 = abs(1.0 - 0.0) * abs(1 / math.log2(2) - 1 / math.log2(3)) / idcg
    rho = lambda a, b: 1.0 / (1.0 + math.exp(F[a] - F[b]))  # noqa: E731
    want_g = [-(d01 * rho(0, 1) + d02 * rho(0, 2)), d01 * rho(0, 1) - d12 * rho(1, 2), d02 * rho(0, 2) + d12 * rho(1, 2)]
    if not all(near(x, y) for x, y in zip(g, want_g)):
        bad.append("three rows: %r vs %r" % (list(g), want_g))
    rng = random.Random(11)
    for c in range(200):
        ng = rng.choice([1, 2, 3])
        groups, rel, k = [], [], 0
        for _ in range(ng):
            s = rng.randint(1, 9)
            groups.append(list(range(k, k + s)))
            rel += [rng.choice([0, 1, 2, 3, 30]) for _ in range(s)]
            k += s
        F = [rng.choice([rng.uniform(-3, 3), 0.5]) for _ in range(k)]
        wts = [rng.uniform(0.5, 3.0) for _ in range(k)] if c % 2 else None
        g, h = fn(F, rel, groups, wts)
        wg, wh = direct(F, rel, groups, wts)
        if not all(near(x, y) for x, y in zip(list(g) + list(h), wg + wh)):
            bad.append("random case %d differs" % c)
            break
    return bad


def main():
    bad = judge(lambda F, rel, groups, wts: ob.lambda_grad(F, rel, groups, wts))
    for line in bad:
        print("FAIL", line)
    wrong = {
        "discount of rank r + 0.0001": lambda F, rel, groups, wts: direct(F, rel, groups, wts, disc=lambda r: 1.0 / math.log2(r + 1.0001)),
        "gain 2^rel": lambda F, rel, groups, wts: direct(F, rel, groups, wts, gain=lambda v: 2.0 ** v),
        "ranks ascending": lambda F, rel, groups, wts: direct(F, rel, groups, wts, descending=False),
        "pairwise (no delta)": lambda F, rel, groups, wts: ob.rank_grad(F, np.asarray(rel), groups, None if wts is None else np.asarray(wts)),
    }
    for name, fn in wrong.items():
        if not judge(lambda F, rel, groups, wts, fn=fn: tuple(np_list(fn(F, rel, groups, wts)))):
            print("FAIL the gate accepts a wrong version: %s" % name)
            bad.append(name)
    print("gate_lambda_controls: %s" % ("all pass, 4 wrong versions rejected" if not bad else "%d failures" % len(bad)))
    return 1 if bad else 0


def np_list(pair):
    return [list(map(float, pair[0])), list(map(float, pair[1]))]


if __name__ == "__main__":
    sys.exit(main())

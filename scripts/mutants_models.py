#!/usr/bin/env python3
"""Mutants of lib/models (docs/models-linear.md section 2, G-mutants): each is one source edit of a copy of the tree; the gate
scripts/gate_models.sh (unit/refusal/limit/fuzz program, the numpy oracles bit for bit, and 1, 2, 4, 8, 16 threads) must fail on it.
Every mutant must be killed; a survivor is listed in ALLOWED with the reason, or the run fails.

    python3 scripts/mutants_models.py --check        every pattern applies exactly once (no compiler)
    python3 scripts/mutants_models.py [substring ...]
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mutlib

LIN = "lib/models/linear/linreg.cho"
LOG = "lib/models/linear/logreg.cho"
NP = "lib/models/linear/npass.cho"
KM = "lib/models/kmeans/kmeans.cho"
KP = "lib/models/kmeans/kpass.cho"
PAR = "lib/models/par.cho"
LPAR = "lib/linalg/par.cho"
LCH = "lib/linalg/chol.cho"
SC = "lib/models/score.cho"
MX = "lib/models/mexp.cho"
RC = "lib/models/record.cho"

M = [
    # ---- linear regression
    ("lin-ridge-on-intercept", LIN, "            while j < d {\n                gv[j * p + j] = gv[j * p + j] + l2;", "            while j < p {\n                gv[j * p + j] = gv[j * p + j] + l2;"),
    ("lin-ridge-sign", LIN, "gv[j * p + j] = gv[j * p + j] + l2;", "gv[j * p + j] = gv[j * p + j] - l2;"),
    ("lin-ridge-dropped", LIN, "gv[j * p + j] = gv[j * p + j] + l2;", "gv[j * p + j] = gv[j * p + j];"),
    ("lin-count", LIN, "diag[d * p + d] = float_of(n);", "diag[d * p + d] = float_of(n) - 1.0;"),
    ("lin-sum-y", LIN, "diag[p * p + d] = sy;", "diag[p * p + d] = 0.0;"),
    ("lin-backward-as-forward", LIN, "code = chol.solve(1, common.mat(p, p), contents(lr), common.vec(p), contents(ur), common.vec(p), contents(wwr));", "code = chol.solve(0, common.mat(p, p), contents(lr), common.vec(p), contents(ur), common.vec(p), contents(wwr));"),
    ("lin-mirror", LIN, "diag[k * p + j] = cc[j * d + k];", "diag[k * p + j] = 0.0;"),
    ("lin-intercept-out", LIN, "out[j] = w[j];", "out[j] = w[p - 1 - j];"),
    # ---- the relative pivot test of the factorisation (docs/models-linear.md 3.1)
    ("lin-pivot-tol-small", LCH, "return 1.0 / 1099511627776.0;", "return 1.0 / 1152921504606846976.0;"),
    ("lin-pivot-tol-large", LCH, "return 1.0 / 1099511627776.0;", "return 1.0 / 1048576.0;"),
    ("lin-pivot-absolute", LCH, "return s > pivot_tol() * a;", "return s > pivot_tol();"),
    ("lin-pivot-none", LCH, "return s > pivot_tol() * a;", "return s > 0.0;"),
    ("lin-pivot-inverted", LCH, "return s > pivot_tol() * a;", "return !(s > pivot_tol() * a);"),
    ("lin-pivot-wrong-diag", LCH, "if !pivot_ok(s, av[j * as_ + j]) {", "if !pivot_ok(s, av[0]) {"),
    # ---- logistic regression
    ("log-iters-minus-one", LOG, "while it < iters && code == 0 {", "while it < iters - 1 && code == 0 {"),
    ("log-iters-plus-one", LOG, "while it < iters && code == 0 {", "while it < iters + 1 && code == 0 {"),
    ("log-update-sign", LOG, "contents(wm)[a] = contents(wm)[a] - contents(dr)[a];", "contents(wm)[a] = contents(wm)[a] + contents(dr)[a];"),
    ("log-update-damped", LOG, "contents(wm)[a] = contents(wm)[a] - contents(dr)[a];", "contents(wm)[a] = contents(wm)[a] - 0.9 * contents(dr)[a];"),
    ("log-l2-on-bias-gradient", LOG, "                                if a < d {\n                                    gg = gg + l2 * w[a];", "                                if a < p {\n                                    gg = gg + l2 * w[a];"),
    ("log-l2-on-bias-hessian", LOG, "if a < d && b == a {", "if a < p && b == a {"),
    ("log-l2-dropped-hessian", LOG, "hh = hh + l2;", "hh = hh;"),
    ("log-l2-dropped-gradient", LOG, "gg = gg + l2 * w[a];", "gg = gg;"),
    ("log-l2-objective", LOG, "sums[ww - 1] / nf + 0.5 * l2 * q;", "sums[ww - 1] / nf + l2 * q;"),
    ("log-gradient-sign", NP, 'ga[j] = ga[j] + ri * v;', 'ga[j] = ga[j] - ri * v;'),
    ("log-hessian-product", NP, 'c100 = c100 + p0[1] * q0[0];', 'c100 = c100 + p0[0] * q0[0];'),
    ("log-hessian-bias-row", NP, 'hbv[lane] = hbv[lane] + si;', 'hbv[lane] = hbv[lane] + ri;'),
    ("log-hessian-s", NP, "let s = p * (1.0 - p);", "let s = p;"),
    ("log-loss-sign", NP, "let loss = pos + mexp.log1p(e) - y * z;", "let loss = pos + mexp.log1p(e) + y * z;"),
    ("log-sigmoid-negative-branch", NP, "    var num = e;", "    var num = 1.0;"),
    ("log-sigmoid-boundary", NP, "    if z >= 0.0 {\n        num = 1.0;", "    if z > 0.0 {\n        num = 1.0;"),
    ("log-bias-start", NP, "var z0 = wv[d];", "var z0 = 0.0;"),
    ("log-score-bias-rows8", NP, "    let b = wv[d];\n    var z0 = b;", "    let b = 0.0;\n    var z0 = b;"),
    ("log-block-length", PAR, "var hi = (b + 1) * 256;\n            if hi > rows {", "var hi = (b + 1) * 255;\n            if hi > rows {"),
    ("log-predict-threshold", LOG, "        if out[i] > 0.0 {\n            out[i] = 1.0;", "        if out[i] >= 0.0 {\n            out[i] = 1.0;"),
    ("log-proba-branch", LOG, "    if z >= 0.0 {\n        return 1.0 / (1.0 + e);\n    }\n    return e / (1.0 + e);", "    if z >= 0.0 {\n        return e / (1.0 + e);\n    }\n    return 1.0 / (1.0 + e);"),
    ("log-exp-table", MX, "return y * pow2tab[k + 1075];", "return y * pow2tab[k + 1074];"),
    # ---- the logistic payload and its labels (docs/wiring.md section 4, added at W1)
    ("rec-labels-equal-accepted", RC, "    if a == b {\n        return decoded(1, 0, 0, at);", "    if a == b {\n        return decoded(0, a, b, at + 32);"),
    ("rec-labels-dtype-unchecked", RC, "    if get32(src, at) != dtype_i64() || get32(src, at + 4) != 1 ||", "    if get32(src, at + 4) != 1 ||"),
    ("rec-labels-swapped", RC, "    return decoded(0, a, b, at + 32);", "    return decoded(0, b, a, at + 32);"),
    ("rec-payload-trailing-bytes-ok", RC, "    if !labels {\n        if r2.end != len(src) {", "    if !labels {\n        if false {"),
    ("rec-labels-end-unchecked", RC, "    if r3.code != 0 || r3.end != len(src) {", "    if r3.code != 0 {"),
    ("rec-kind-unchecked", RC, "    if len(src) < 4 || get32(src, 0) != kind || d < 1 {", "    if len(src) < 4 || d < 1 {"),
    # ---- k-means
    ("km-tie-rule-lane", KP, "        if a4 < bd {", "        if a4 <= bd {"),
    ("km-tie-rule-group", KP, "a7 = a7 + u7 * u7;\n            j = j + 1;\n        }\n        if a0 < bd {", "a7 = a7 + u7 * u7;\n            j = j + 1;\n        }\n        if a0 <= bd {"),
    ("km-tie-rule-single-group", KP, "        if a3 < bd {\n            bd = a3;\n            best = g * 4 + 3;\n        }\n    }\n    return (best, bd);", "        if a3 <= bd {\n            bd = a3;\n            best = g * 4 + 3;\n        }\n    }\n    return (best, bd);"),
    ("km-padding-finite", KP, "var v = 1.0 / 0.0;", "var v = 1.0e300;"),
    ("km-padding-index", KP, "out[(c / 4 * d + j) * 4 + c % 4] = v;", "out[(c / 4 * d + j) * 4 + (3 - c % 4)] = v;"),
    ("km-empty-cluster", KM, "if cnt > 0.0 {", "if cnt >= 0.0 {"),
    ("km-empty-cluster-zero", KM, "                        if cnt > 0.0 {\n                            var j = 0;\n                            while j < d {\n                                cen[c * d + j] = sums[c * d + j] / cnt;", "                        if cnt > 0.0 {\n                            var j = 0;\n                            while j < d {\n                                cen[c * d + j] = sums[c * d + j] / cnt;\n                                j = j + 1;\n                            }\n                        } else {\n                            var j = 0;\n                            while j < d {\n                                cen[c * d + j] = 0.0;"),
    ("km-seed-derive-index", KM, "let zj = rng.derive(root, j);", "let zj = rng.derive(root, j + 1);"),
    ("km-derive-kind", KM, "pub fn kind() -> [] int {\n    return 3;", "pub fn kind() -> [] int {\n    return 2;"),
    ("km-attempt-index", KM, "row = rng.uniform(rng.derive(zj, a), n);", "row = rng.uniform(rng.derive(zj, a + 1), n);"),
    ("km-plusplus-compare", KM, "if cum > u {", "if cum >= u {"),
    ("km-plusplus-min", KM, "if j == 1 || dd < mind[i] {", "if j == 1 || dd > mind[i] {"),
    ("km-u01-scale", KM, "/ 9007199254740992.0;", "/ 4503599627370496.0;"),
    ("km-first-center", KM, "ch[0] = rng.uniform(zj, n);", "ch[0] = rng.uniform(zj, n - 1);"),
    ("km-fallback", KM, "pick = rng.uniform(zj, n);", "pick = 0;"),
    ("km-iters-minus-one", KM, "} else if it < iters {", "} else if it < iters - 1 {"),
    ("km-update-division", KM, "cen[c * d + j] = sums[c * d + j] / cnt;", "cen[c * d + j] = sums[c * d + j] / (cnt + 1.0);"),
    ("km-sum-sign", KP, "row[j] = row[j] + xr[j];", "row[j] = row[j] - xr[j];"),
    ("km-count", KP, 'la[base + k * d + best] = la[base + k * d + best] + 1.0;', 'la[base + k * d + best] = la[base + k * d + best] + 2.0;'),
    ("km-inertia", KP, 'la[base + ww - 1] = la[base + ww - 1] + bd;', 'la[base + ww - 1] = la[base + ww - 1] + 0.0;'),
    ("km-distance", KP, "a7 = a7 + u7 * u7;", "a7 = a7 + u7;"),
    ("km-distance-single", KP, "a3 = a3 + u3 * u3;\n            j = j + 1;\n        }\n        if a0 < bd {", "a3 = a3 + u3;\n            j = j + 1;\n        }\n        if a0 < bd {"),
    ("km-distance-start", KP, "var a7 = 0.0;", "var a7 = 1.0e-300;"),
    ("km-block-length", PAR, 'kpass.block(xs, rs, b * 256, hi, d, k, wv, lab, sc, st[sp * ww..sp * ww + ww]);', 'kpass.block(xs, rs, b * 256, hi - 1, d, k, wv, lab, sc, st[sp * ww..sp * ww + ww]);'),
    ("km-block-length-sequential", PAR, 'kpass.block(xs, rs, b * 256, hi, d, kc, wv, lab, sc, st[sp * ww..sp * ww + ww]);', 'kpass.block(xs, rs, b * 256, hi - 1, d, kc, wv, lab, sc, st[sp * ww..sp * ww + ww]);'),
    ("km-final-pass", KM, "while it <= iters && code == 0 {", "while it < iters && code == 0 {"),
    ("km-transform-sqrt", KM, "out[i * k + c] = sqrt(kpass.dist2(", "out[i * k + c] = (kpass.dist2("),
    # ---- the lane rule of the decisive sums (docs/linalg.md section 2): lane count, lane map, lane combine order, lane start
    ("log-lane-map", NP, "let lane = i % 4;", "let lane = i % 2;"),
    ("log-lane-map-halves", NP, "let lane = i % 4;", "let lane = i / 2 % 4;"),
    ("log-lane-combine-g", NP, "slot[a] = order.lane_combine(gacc[a], gacc[pd + a], gacc[2 * pd + a], gacc[3 * pd + a]);", "slot[a] = order.lane_combine(gacc[a], gacc[2 * pd + a], gacc[pd + a], gacc[3 * pd + a]);"),
    ("log-lane-combine-hbias", NP, 'brow[b] = order.lane_combine(hb[b], hb[pd + b], hb[2 * pd + b], hb[3 * pd + b]);', 'brow[b] = order.lane_combine(hb[b], hb[2 * pd + b], hb[pd + b], hb[3 * pd + b]);'),
    ("log-lane-combine-gb", NP, "slot[d] = order.lane_combine(gbv[0], gbv[1], gbv[2], gbv[3]);", "slot[d] = order.lane_combine(gbv[0], gbv[2], gbv[1], gbv[3]);"),
    ("log-lane-combine-tile", NP, "hacc[(a0 + 1) * pd + b0 + 1] = order.lane_combine(c110, c111, c112, c113);", "hacc[(a0 + 1) * pd + b0 + 1] = order.lane_combine(c110, c112, c111, c113);"),
    ("log-lane-tile-map", NP, "c102 = c102 + p2[1] * q2[0];", "c101 = c101 + p2[1] * q2[0];"),
    ("log-lane-tile-tail", NP, "c001 = c001 + p[0] * q[0];", "c000 = c000 + p[0] * q[0];"),
    ("log-lane-tile-init", NP, 'var c113 = 0.0;', 'var c113 = 1.0e10;'),
    ("log-lane-loss-fold", NP, "slot[p + p * (p + 1) / 2] = order.block_partial(ll, m);", "slot[p + p * (p + 1) / 2] = ll[0];"),
    ("log-lane-acc-init", NP, "    while j < 4 * pd {\n        gacc[j] = 0.0;", "    while j < 3 * pd {\n        gacc[j] = 0.0;"),
    ("km-lane-map", KP, "let base = (i - r0) % 4 * ww;", "let base = (i - r0) % 2 * ww;"),
    ("km-lane-combine", KP, "slot[z] = order.lane_combine(la[z], la[ww + z], la[2 * ww + z], la[3 * ww + z]);", "slot[z] = order.lane_combine(la[z], la[2 * ww + z], la[ww + z], la[3 * ww + z]);"),
    ("km-lane-zero", KP, "    while q < 4 * ww {\n        la[q] = 0.0;", "    while q < 3 * ww {\n        la[q] = 0.0;"),
    # ---- the threads (shared by the three)
    ("par-thread-order", LPAR, "    var sp = 0;\n    var c = 0;\n    while c < nchunks {\n        sp = order.settle_root(st, lv, sp, w, roots, c * w, 3);\n        c = c + 1;\n    }\n    order.finish(st, sp, w);\n    return 0;", "    var j = 0;\n    while j < w {\n        st[j] = 0.0;\n        j = j + 1;\n    }\n    var k = 0;\n    while k < t {\n        j = 0;\n        while j < w {\n            st[w + j] = 0.0;\n            j = j + 1;\n        }\n        var c = bound(k, t, nchunks);\n        let c1 = bound(k + 1, t, nchunks);\n        while c < c1 {\n            j = 0;\n            while j < w {\n                st[w + j] = st[w + j] + roots[c * w + j];\n                j = j + 1;\n            }\n            c = c + 1;\n        }\n        j = 0;\n        while j < w {\n            st[j] = st[j] + st[w + j];\n            j = j + 1;\n        }\n        k = k + 1;\n    }\n    return 0;"),
    ("par-models-chunk-misaligned", PAR, "let r0 = c0 * 2048;", "let r0 = c0 * 2047;"),
    ("par-models-row-copy", PAR, "copy_floats(jx[i * d..i * d + d], xs[(r0 + i) * rs..(r0 + i) * rs + d], d);", "copy_floats(jx[i * d..i * d + d], xs[(r0 + i + 1) * rs..(r0 + i + 1) * rs + d], d);"),
    ("view-stride-threaded-copy", PAR, "xs[(r0 + i) * rs..(r0 + i) * rs + d], d);", "xs[(r0 + i) * d..(r0 + i) * d + d], d);"),
    ("view-stride-newton", NP, "let x0 = xs[(r0 + i0 + 0) * rs..(r0 + i0 + 0) * rs + d];", "let x0 = xs[(r0 + i0 + 0) * d..(r0 + i0 + 0) * d + d];"),
    ("view-stride-lloyd", KP, "let xr = xs[i * rs..i * rs + d];", "let xr = xs[i * d..i * d + d];"),
    ("par-models-y-copy", PAR, "copy_floats(contents(jk.jy), ys[r0..r1], rows);", "copy_floats(contents(jk.jy), ys[r0 + 1..r1 + 1], rows);"),
    ("par-models-labels", PAR, "dst[i] = src[i];", "dst[i] = 0;"),
    ("par-models-weights", PAR, "copy_floats(contents(jk.jw), wv, len(wv));", "copy_floats(contents(jk.jw), wv, len(wv) - 1);"),
    ("par-models-ragged", PAR, "        var b1 = c * 8 + 8;\n        if b1 > nb {\n            b1 = nb;\n        }\n        var sp = 0;", "        var b1 = c * 8 + 8;\n        var sp = 0;"),
]

# survivors that are equivalent mutants, each with the reason
ALLOWED = {
    "par-models-ragged": "without the clamp the last chunk runs blocks past the end; each is an all-zero partial (m <= 0 rows), and x + 0.0 = x, so the tree of the real blocks keeps its bits: an equivalent mutant (the clamp is kept so that no block is made of nothing)",
    "log-sigmoid-boundary": "z == 0 happens only at the first iteration (w = 0), where e = 1 and 1/(1+e) = e/(1+e) = 0.5 with the same bits; no other row has z exactly 0",
}

if __name__ == "__main__":
    sys.exit(mutlib.run([m for m in M if len(sys.argv) < 2 or sys.argv[1] == "--check" or any(a in m[0] for a in sys.argv[1:])],
                        ["sh", "scripts/gate_models.sh"], "mutants_models", allowed=ALLOWED))

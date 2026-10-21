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
CM = "lib/models/common.cho"
SC = "lib/models/score.cho"
MX = "lib/models/mexp.cho"
RC = "lib/models/record.cho"
SP = "lib/models/softmax/spass.cho"
SM = "lib/models/softmax/softmax.cho"
RF = "lib/core/refuse.cho"

M = [
    # ---- linear regression
    ("lin-ridge-on-intercept", LIN, "            while j < d {\n                gv[j * p + j] = gv[j * p + j] + l2;", "            while j < p {\n                gv[j * p + j] = gv[j * p + j] + l2;"),
    ("lin-ridge-sign", LIN, "gv[j * p + j] = gv[j * p + j] + l2;", "gv[j * p + j] = gv[j * p + j] - l2;"),
    ("lin-ridge-dropped", LIN, "gv[j * p + j] = gv[j * p + j] + l2;", "gv[j * p + j] = gv[j * p + j];"),
    ("lin-count", LIN, "diag[d * p + d] = corner;", "diag[d * p + d] = corner - 1.0;"),
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
    ("rec-labels-equal-accepted", RC, "    if a == b || a < 0 - label_max()", "    if a < 0 - label_max()"),
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
    # ---- softmax regression (docs/multiclass.md 3.2)
    ("sm-loss-sign", SP, "let loss = mx + math.log(sum) - zy;", "let loss = mx + math.log(sum) + zy;"),
    ("sm-loss-max", SP, "let loss = mx + math.log(sum) - zy;", "let loss = math.log(sum) - zy;"),
    ("sm-loss-reference", SP, "if yi < kc {\n            zy = zv[yi];", "if yi < kc - 1 {\n            zy = zv[yi];"),
    ("sm-max-start", SP, "var mx = 0.0;\n        var k = 0;", "var mx = 0.0 - 1.0e9;\n        var k = 0;"),
    ("sm-max-compare", SP, "            if z > mx {\n                mx = z;", "            if z < mx {\n                mx = z;"),
    ("sm-reference-exp", SP, "let eref = mexp.exp(0.0 - mx);", "let eref = 1.0;"),
    ("sm-exp-argument", SP, "ev[k] = mexp.exp(zv[k] - mx);", "ev[k] = mexp.exp(zv[k]);"),
    ("sm-sum-missing-reference", SP, "sum = sum + eref;", "sum = sum + 0.0;"),
    ("sm-probability", SP, "let pk = ev[k] / sum;", "let pk = ev[k] / (sum + 1.0);"),
    ("sm-target", SP, "if yi == k {\n                tk = 1.0;", "if yi == k + 1 {\n                tk = 1.0;"),
    ("sm-s-diagonal", SP, "sm[k * kc + jj] = pv[k] * (1.0 - pv[k]);", "sm[k * kc + jj] = pv[k];"),
    ("sm-s-offdiagonal-sign", SP, "sm[k * kc + jj] = 0.0 - pv[k] * pv[jj];", "sm[k * kc + jj] = pv[k] * pv[jj];"),
    ("sm-gradient-sign", SP, "la[k * dd + a] = la[k * dd + a] + rk * av[a];", "la[k * dd + a] = la[k * dd + a] - rk * av[a];"),
    ("sm-hessian-triangle", SP, "bmax = a + 1;", "bmax = a;"),
    ("sm-hessian-class-pair", SP, "let sa = sm[k * kc + jj] * xa;", "let sa = sm[k * kc + k] * xa;"),
    ("sm-hessian-row-offset", SP, "let base = p + ai * (ai + 1) / 2;", "let base = p + ai * ai / 2;"),
    ("sm-hessian-class-offset", SP, "let off = base + jj * dd;", "let off = base + jj;"),
    ("sm-hessian-factor", SP, "la[off + b] = la[off + b] + sa * av[b];", "la[off + b] = la[off + b] + sa * av[a];"),
    ("sm-q-first", SP, "var q = zv[0] * zv[0];", "var q = zv[0];"),
    ("sm-q-square", SP, "q = q + zv[k] * zv[k];", "q = q + zv[k];"),
    ("sm-score-bias", SP, "var z = wv[k * dd + d];", "var z = 0.0;"),
    ("sm-score-weight", SP, "z = z + row[j] * wv[k * dd + j];", "z = z + row[j] * wv[j];"),
    ("sm-label-row", SP, "let yi = truncate(ys[r0 + i]);", "let yi = truncate(ys[r0]);"),
    ("sm-bias-column", SP, "av[d] = 1.0;", "av[d] = 0.0;"),
    ("sm-lane-map", SP, "let la = lanes[i % 4 * ww..i % 4 * ww + ww];", "let la = lanes[i % 2 * ww..i % 2 * ww + ww];"),
    ("sm-lane-combine", SP, "slot[u] = order.lane_combine(lanes[u], lanes[ww + u], lanes[2 * ww + u], lanes[3 * ww + u]);", "slot[u] = order.lane_combine(lanes[u], lanes[2 * ww + u], lanes[ww + u], lanes[3 * ww + u]);"),
    ("sm-lane-zero", SP, "    while t < 4 * ww {\n        lanes[t] = 0.0;", "    while t < 3 * ww {\n        lanes[t] = 0.0;"),
    ("sm-width", SP, "return p + p * (p + 1) / 2 + 2;", "return p + p * (p + 1) / 2 + 3;"),
    ("sm-iters-minus-one", SM, "while it < iters && code == 0 {", "while it < iters - 1 && code == 0 {"),
    ("sm-iters-plus-one", SM, "while it < iters && code == 0 {", "while it < iters + 1 && code == 0 {"),
    ("sm-update-sign", SM, "contents(wm)[a] = contents(wm)[a] - contents(dr)[a];", "contents(wm)[a] = contents(wm)[a] + contents(dr)[a];"),
    ("sm-l2-bias-gradient", SM, "                                if a % dd < d {\n                                    gg = gg + l2 * w[a];", "                                if a % dd < dd {\n                                    gg = gg + l2 * w[a];"),
    ("sm-l2-bias-hessian", SM, "if a % dd < d && b == a {", "if a % dd < dd && b == a {"),
    ("sm-l2-dropped-hessian", SM, "hh = hh + l2;", "hh = hh;"),
    ("sm-l2-dropped-gradient", SM, "gg = gg + l2 * w[a];", "gg = gg;"),
    ("sm-l2-objective", SM, "sums[ww - 1] / nf + 0.5 * l2 * q;", "sums[ww - 1] / nf + l2 * q;"),
    ("sm-l2-objective-bias", SM, "                            if a % dd < d {\n                                q = q + w[a] * w[a];", "                            if a % dd < dd {\n                                q = q + w[a] * w[a];"),
    ("sm-mean-gradient", SM, "var gg = sums[a] / nf;", "var gg = sums[a];"),
    ("sm-mean-hessian", SM, "var hh = sums[p + a * (a + 1) / 2 + b] / nf;", "var hh = sums[p + a * (a + 1) / 2 + b];"),
    ("sm-separable-bound", SM, "return 0.6931;", "return 0.0;"),
    ("sm-scale-bound", SM, "return 50.0;", "return 1.0e300;"),
    ("sm-scale-formula", SM, "} else if math.sqrt(qsum / (nf * float_of(kc))) > scale_bound() {", "} else if math.sqrt(qsum / nf) > scale_bound() {"),
    ("sm-separable-only-with-ridge", SM, "        if code == 0 && l2 == 0.0 {\n            // separation (docs/multiclass.md 3.2)", "        if code == 0 && l2 == 99.0 {\n            // separation (docs/multiclass.md 3.2)"),
    ("sm-collinear-first-pass", SM, "                if it == 0 {\n                    // the Hessian of the first pass is a Gram-like matrix", "                if it == 7 {\n                    // the Hessian of the first pass is a Gram-like matrix"),
    ("sm-max-classes", SM, "pub fn max_classes() -> [] int {\n    return 64;", "pub fn max_classes() -> [] int {\n    return 65;"),
    ("sm-max-params", SM, "pub fn max_params() -> [] int {\n    return 2048;", "pub fn max_params() -> [] int {\n    return 2049;"),
    ("sm-budget", SM, "return 67108864;", "return 67108865;"),
    ("sm-memory-chunks", SM, "return ww * (2 * nchunks + depth + 2 + 10 * te) + 2 * p * p + n * (d + 1);", "return ww * (nchunks + depth + 2 + 10 * te) + 2 * p * p + n * (d + 1);"),
    ("sm-memory-matrix", SM, "+ 2 * p * p + n * (d + 1);", "+ p * p + n * (d + 1);"),
    ("sm-memory-threads", SM, "    if te > 64 {\n        te = 64;\n    }", "    if te > 63 {\n        te = 63;\n    }"),
    ("sm-rows-ceiling", SM, "if n > 1073741824 {", "if n > 1073741825 {"),
    ("sm-classes-first", SM, "    if kk > max_classes() {\n        return refuse.limit_classes();\n    }\n    var c = common.check_iters(iters);", "    var c = common.check_iters(iters);"),
    ("sm-label-range", SM, "if v < 0.0 || v > float_of(kk - 1) || float_of(truncate(v)) != v {", "if v < 0.0 || v > float_of(kk) || float_of(truncate(v)) != v {"),
    ("sm-label-fraction", SM, "|| float_of(truncate(v)) != v {", "|| false {"),
    ("sm-absent-class", SM, "if bad != 0 || present != kk {", "if bad != 0 || present < 2 {"),
    ("sm-predict-tie", SM, "                if z[k] > bv {\n                    bv = z[k];", "                if z[k] >= bv {\n                    bv = z[k];"),
    ("sm-predict-reference", SM, "            if 0.0 > bv {\n                best = kc;", "            if 0.0 >= bv {\n                best = kc;"),
    ("sm-predict-reference-missing", SM, "            if 0.0 > bv {\n                best = kc;\n            }", "            if false {\n                best = kc;\n            }"),
    ("sm-proba-reference", SM, "out[i * kk + kc] = eref / s;", "out[i * kk + kc] = 0.0;"),
    ("sm-proba-layout", SM, "out[i * kk + k] = z[k] / s;", "out[k * n + i] = z[k] / s;"),
    ("sm-proba-sum", SM, "            s = s + eref;\n            k = 0;\n            while k < kc {\n                out[i * kk + k]", "            k = 0;\n            while k < kc {\n                out[i * kk + k]"),
    ("sm-predict-class-count", SM, "    if kk < 3 || kk > max_classes() {\n        return refuse.param_out_of_range();\n    }\n    var c = common.check_x", "    var c = common.check_x"),
    ("sm-par-dispatch", PAR, "            } else if kind == 3 {\n                spass.block(xs, rs, ys, b * 256, hi, d, k, wv, sc, st[sp * ww..sp * ww + ww]);", "            } else if kind == 3 {\n                spass.block(xs, rs, ys, b * 256, hi, d, k + 1, wv, sc, st[sp * ww..sp * ww + ww]);"),
    ("sm-par-labels-copy", PAR, "        if kind != 2 {\n            copy_floats(contents(jk.jy), ys[r0..r1], rows);", "        if kind == 1 {\n            copy_floats(contents(jk.jy), ys[r0..r1], rows);"),
    ("sm-refuse-classes-code", RF, "pub fn limit_classes() -> [] int {\n    return 24;", "pub fn limit_classes() -> [] int {\n    return 25;"),
    # ---- sample weights (docs/weights.md section 2.2 and 2.3)
    ("w-lin-no-sqrt", LIN, "let si = sqrt(wd[i]);", "let si = wd[i];"),
    ("w-lin-corner-n", LIN, "        corner = r.v;", "        corner = float_of(n);"),
    ("w-lin-y-unscaled", LIN, "contents(yw)[i] = si * yd[i];", "contents(yw)[i] = yd[i];"),
    ("w-lin-x-unscaled", LIN, "contents(xw)[i * d + j] = si * xd[xt.off + i * xt.t0 + j * xt.t1];", "contents(xw)[i * d + j] = xd[xt.off + i * xt.t0 + j * xt.t1];"),
    ("w-lin-view-stride", LIN, "xd[xt.off + i * xt.t0 + j * xt.t1];", "xd[xt.off + i * d + j];"),
    ("w-lin-weighted-flag", LIN, "if code == 0 && weighted {", "if code == 0 && false {"),
    ("w-check-negative", CM, "if wd[i] < 0.0 || wd[i] > max_weight() {", "if wd[i] < 0.0 - 1.0 || wd[i] > max_weight() {"),
    ("w-check-max", CM, "|| wd[i] > max_weight() {", "|| wd[i] >= max_weight() {"),
    ("w-check-max-value", CM, "return 1000000000000.0;", "return 10000000000000.0;"),
    ("w-check-nonfinite", CM, "    if count_bad_vec(wd, n) != 0 {\n        return refuse.input_non_finite();\n    }\n    var any = false;", "    var any = false;"),
    ("w-check-all-zero", CM, "    if !any {\n        return refuse.param_out_of_range();", "    if false {\n        return refuse.param_out_of_range();"),
    ("w-log-nf-rows", LOG, "        sw = r.v;", "        sw = float_of(n);"),
    ("w-log-wmin-ignored", LOG, "contents(yr), n, d, iters, l2, sw, wmin, out, diag);", "contents(yr), n, d, iters, l2, sw, 1.0, out, diag);"),
    ("w-log-classes-all-rows", LOG, "        if wd[i] > 0.0 {\n            if yd[i] == 1.0 {", "        if true {\n            if yd[i] == 1.0 {"),
    ("w-log-kind", LOG, "code = newton(heap, t, 4, xd[xt.off..len(xd)]", "code = newton(heap, t, 1, xd[xt.off..len(xd)]"),
    ("w-log-weights-after-labels", LOG, "contents(yq)[n + k] = wd[k];", "contents(yq)[n + k] = 1.0;"),
    ("w-np-r", NP, "rr[q] = rr[q] * wq;", "rr[q] = rr[q];"),
    ("w-np-s", NP, "ss[q] = ss[q] * wq;", "ss[q] = ss[q];"),
    ("w-np-loss", NP, "ll[q] = ll[q] * wq;", "ll[q] = ll[q];"),
    ("w-np-row-offset", NP, "let wq = ys[wo + r0 + q];", "let wq = ys[wo + q];"),
    ("w-par-weights-copy", PAR, "ys[n + r0..n + r1], rows);", "ys[r0..r1], rows);"),
    ("w-par-piece-offset", PAR, "let wo = contents(job.jp)[6] * rows;", "let wo = contents(job.jp)[6] * (rows - 1);"),
    ("w-par-sequential-offset", PAR, "npass.block(xs, rs, ys, b * 256, hi, d, wv, n, sc, st[sp * ww..sp * ww + ww]);", "npass.block(xs, rs, ys, b * 256, hi, d, wv, 0, sc, st[sp * ww..sp * ww + ww]);"),
    ("w-par-kind4-dispatch", PAR, "if kind == 1 || kind == 4 {", "if kind == 1 {"),
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

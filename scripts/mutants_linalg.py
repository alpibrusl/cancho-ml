#!/usr/bin/env python3
"""Source mutants of lib/linalg, per kernel group. Each mutant is applied to a copy of the tree; the group's gate
(scripts/gate_linalg.sh: unit tests against the reference, then the compiled driver against the numpy oracle) must fail.

    python3 scripts/mutants_linalg.py --check                pattern check only
    python3 scripts/mutants_linalg.py [group|name-substring ...]
Groups: dot axpy gemv syrk chol gemm par order. Survivors are listed with the reason in ALLOWED."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import mutlib

D, A, G, S, C, M, O = ("lib/linalg/dot.cho", "lib/linalg/axpy.cho", "lib/linalg/gemv.cho", "lib/linalg/syrk.cho",
                       "lib/linalg/chol.cho", "lib/linalg/gemm.cho", "lib/linalg/order.cho")

GROUPS = {}
GROUPS["dot"] = (["sh", "scripts/gate_linalg.sh", "t_dot.cho", "dot"], [
    ("dot-lane-init", D, "var a3 = 0.0;", "var a3 = 1.0e-16;"),
    ("dot-lane-map", D, "s1 = s1 + xr[j + 1] * yr[j + 1];", "s0 = s0 + xr[j + 1] * yr[j + 1];"),
    ("dot-lane-tail", D, "s1 = s1 + xr[j] * yr[j];", "s2 = s2 + xr[j] * yr[j];"),
    ("dot-pair-second", D, "b3 = b3 + x1[j + 3] * y1[j + 3];", "b3 = b3 + x1[j + 2] * y1[j + 2];"),
    ("dot-pair-combine", D, 'order.lane_combine(b0, b1, b2, b3)', 'order.lane_combine(b0, b2, b1, b3)'),
    ("dot-ragged-block", D, "var m = n - b * 256;\n            if m > 256 {\n                m = 256;", "var m = n - b * 256;\n            if m > 255 {\n                m = 255;"),
    ("dot-pair-guard", D, "if b + 2 <= b1 && (b + 2) * 256 <= n {", "if b + 2 <= b1 && (b + 2) * 256 <= n + 256 {"),
    ("dot-strided-lane", D, "let l = j % 4;", "let l = j / 2 % 4;"),
    ("dot-strided-block", D, "if m > 256 {\n        m = 256;\n    }\n    var s0 = 0.0;", "if m > 255 {\n        m = 255;\n    }\n    var s0 = 0.0;"),
    ("dot-strided-init", D, 'var s3 = 0.0;\n    var j = 0;\n    while j < m {\n        let i = b * 256 + j;', 'var s3 = 1.0e30;\n    var j = 0;\n    while j < m {\n        let i = b * 256 + j;'),
    ("dot-strided-ystride", D, "yd[yo + i * ys]", "yd[yo + i]"),
    ("dot-empty", D, "if n == 0 {\n        return tensor.fres(0, 0.0);\n    }", "if n == 0 {\n        return tensor.fres(0, 1.0);\n    }"),
    ("dot-shape-check", D, "if xt.s0 != yt.s0 {", "if xt.s0 > yt.s0 {"),
    ("order-no-cascade", O, "lv[sp - 2] = lv[sp - 2] + 1;\n        sp = sp - 1;\n    }\n    return sp;\n}\n\n// Like settle", "lv[sp - 2] = 0;\n        sp = sp - 1;\n    }\n    return sp;\n}\n\n// Like settle"),
    ("order-finish-left", O, "    while sp >= 2 {\n        let a = (sp - 2) * w;\n        let b = (sp - 1) * w;\n        var j = 0;\n        while j < w {\n            st[a + j] = combine(st[a + j], st[b + j]);\n            j = j + 1;\n        }\n        sp = sp - 1;\n    }\n    return 0;", "    var q = 1;\n    while q < sp {\n        var j = 0;\n        while j < w {\n            st[j] = combine(st[j], st[q * w + j]);\n            j = j + 1;\n        }\n        q = q + 1;\n    }\n    return 0;"),
    ("order-blocks", O, "return (n - 1) / block() + 1;", "return n / block() + 1;"),
    ("order-level-eq", O, "while sp >= 2 && lv[sp - 1] == lv[sp - 2] {\n        let a = (sp - 2) * w;\n        let b = (sp - 1) * w;\n        var j = 0;\n        while j < w {\n            st[a + j] = combine(st[a + j], st[b + j]);\n            j = j + 1;\n        }\n        lv[sp - 2] = lv[sp - 2] + 1;\n        sp = sp - 1;\n    }\n    return sp;\n}\n\n// Like", "while sp >= 2 && lv[sp - 1] >= lv[sp - 2] {\n        let a = (sp - 2) * w;\n        let b = (sp - 1) * w;\n        var j = 0;\n        while j < w {\n            st[a + j] = combine(st[a + j], st[b + j]);\n            j = j + 1;\n        }\n        lv[sp - 2] = lv[sp - 2] + 1;\n        sp = sp - 1;\n    }\n    return sp;\n}\n\n// Like"),
])
GROUPS["axpy"] = (["sh", "scripts/gate_linalg.sh", "t_dot.cho", "axpy"], [
    ("axpy-sign", A, "yr[i] = yr[i] + a * xr[i];", "yr[i] = yr[i] - a * xr[i];"),
    ("axpy-skip-last", A, "let xr = xs[0..n];\n    let yr = ys[0..n];\n    var i = 0;\n    while i < n {", "let xr = xs[0..n];\n    let yr = ys[0..n];\n    var i = 0;\n    while i + 1 < n {"),
    ("axpy-strided-x", A, "yd[q] = yd[q] + a * xd[xt.off + i * xt.t0];", "yd[q] = yd[q] + a * xd[xt.off + i];"),
    ("axpy-strided-y", A, "let q = yt.off + i * yt.t0;", "let q = yt.off + i;"),
    ("axpy-writable", A, "c = tensor.writable(yt);", "c = 0;"),
    ("axpy-empty", A, "    let n = xt.s0;\n    if n == 0 {\n        return 0;\n    }\n    if xt.t0 == 1", "    let n = xt.s0;\n    if xt.t0 == 1"),
    ("axpy-shape", A, "if xt.s0 != yt.s0 {", "if xt.s0 < yt.s0 {"),
])
GROUPS["gemv"] = (["sh", "scripts/gate_linalg.sh", "t_gemv.cho", "gemv_n", "gemv_t"], [
    ("gemvn-row-mix", G, "z33 = z33 + r3[j + 3] * w3;", "z33 = z33 + r2[j + 3] * w3;"),
    ("gemvn-init", G, "var z33 = 0.0;", "var z33 = 1.0e-16;"),
    ("gemvn-lane-map", G, "z01 = z01 + r0[j + 1] * w1;", "z00 = z00 + r0[j + 1] * w1;"),
    ("gemvn-lane-tail", G, "z22 = z22 + r2[j + 2] * wj;", "z23 = z23 + r2[j + 2] * wj;"),
    ("gemvn-lane-tail-0", G, "z10 = z10 + r1[j + 0] * wj;", "z11 = z11 + r1[j + 0] * wj;"),
    ("gemvn-store", G, "ys[i + 2] = order.lane_combine(z20, z21, z22, z23);", "ys[i + 2] = order.lane_combine(z30, z31, z32, z33);"),
    ("gemvn-combine", G, "ys[i + 1] = order.lane_combine(z10, z11, z12, z13);", "ys[i + 1] = order.lane_combine(z10, z12, z11, z13);"),
    ("gemvn-row1-tail", G, "z2 = z2 + r[j + 2] * wr[j + 2];\n    }\n    return", "z1 = z1 + r[j + 2] * wr[j + 2];\n    }\n    return"),
    ("gemvn-row1-init", G, "var z3 = 0.0;\n    var j = 0;\n    while j + 4 <= d {\n        z0 = z0 + r[j] * wr[j];", "var z3 = 1.0e-16;\n    var j = 0;\n    while j + 4 <= d {\n        z0 = z0 + r[j] * wr[j];"),
    ("gemvn-fast-guard", G, "&& d <= 256 && xt.t0 >= 0 {", "&& d <= 300 && xt.t0 >= 0 {"),
    ("gemvn-generic-row", G, "dot.block_strided(xd, xt.off + i * xt.t0, xt.t1,", "dot.block_strided(xd, xt.off, xt.t1,"),
    ("gemvn-generic-stride", G, "wd, wt.off, wt.t0, b, d)", "wd, wt.off, 1, b, d)"),
    ("gemvn-d0", G, "if d == 0 {\n        var i = 0;\n        while i < n {\n            yd[yt.off + i * yt.t0] = 0.0;", "if d == 0 {\n        var i = 0;\n        while i < n {\n            yd[yt.off + i * yt.t0] = 1.0;"),
    ("gemvt-acc-init", G, "acc[j] = 0.0;\n        j = j + 1;\n    }\n    let l0", "acc[j] = 1.0e-16;\n        j = j + 1;\n    }\n    let l0"),
    ("gemvt-lane-map", G, "l1[j] = l1[j] + a1[j] * c1;\n            l2[j]", "l0[j] = l0[j] + a1[j] * c1;\n            l2[j]"),
    ("gemvt-row-coef", G, "l3[j] = l3[j] + a3[j] * c3;", "l3[j] = l3[j] + a3[j] * c2;"),
    ("gemvt-tail-lane", G, "let c2 = rv[i + 2];\n        j = 0;\n        while j < d {\n            l2[j] = l2[j] + a2[j] * c2;", "let c2 = rv[i + 2];\n        j = 0;\n        while j < d {\n            l1[j] = l1[j] + a2[j] * c2;"),
    ("gemvt-tail-row", G, "let c0 = rv[i];\n        j = 0;", "let c0 = rv[r0];\n        j = 0;"),
    ("gemvt-combine", G, 'order.lane_combine(st[base + j], st[base + d + j], st[base + 2 * d + j], st[base + 3 * d + j])', 'order.lane_combine(st[base + j], st[base + 2 * d + j], st[base + d + j], st[base + 3 * d + j])'),
    ("gemvt-stack", G, "return 44 * d;", "return 41 * d;"),
    ("gemvt-block-hi", G, "var hi = (b + 1) * 256;\n        if hi > n {\n            hi = n;\n        }\n        block_lanes", "var hi = (b + 1) * 255;\n        if hi > n {\n            hi = n;\n        }\n        block_lanes"),
    ("gemvt-generic-hi", G, "var hi = (b + 1) * 256;\n                    if hi > n {\n                        hi = n;\n                    }\n                    j = 0;", "var hi = (b + 1) * 255;\n                    if hi > n {\n                        hi = n;\n                    }\n                    j = 0;"),
    ("gemvt-generic-lane", G, "let lane = (i - b * 256) % 4;", "let lane = (i - b * 256) % 2;"),
    ("gemvt-generic-index", G, "xd[xt.off + i * xt.t0 + j * xt.t1] * ri", "xd[xt.off + i * xt.t0 + j] * ri"),
    ("gemvt-finish", G, "order.finish(contents(sw), sp, d);\n            j = 0;", "j = 0;"),
])
GROUPS["syrk"] = (["sh", "scripts/gate_linalg.sh", "t_syrk.cho", "syrk"], [
    ("syrk-tile-entry", S, "c010 = c010 + p0[0] * q0[1];", "c010 = c010 + p0[1] * q0[0];"),
    ("syrk-tile-row", S, "c113 = c113 + p3[1] * q3[1];", "c113 = c113 + p3[0] * q3[1];"),
    ("syrk-tile-init", S, "var c113 = 0.0;", "var c113 = 1.0e-16;"),
    ("syrk-tile-lane", S, 'c001 = c001 + p1[0] * q1[0];', 'c000 = c000 + p1[0] * q1[0];'),
    ("syrk-tile-store", S, 'order.lane_combine(c110, c111, c112, c113);', 'order.lane_combine(c110, c112, c111, c113);'),
    ("syrk-tile-tail", S, 'c001 = c001 + p[0] * q[0];', 'c000 = c000 + p[0] * q[0];'),
    ("syrk-edge-init", S, "var s3 = 0.0;\n    var i = r0;\n    while i + 4 <= r1 {", "var s3 = 1.0e-16;\n    var i = r0;\n    while i + 4 <= r1 {"),
    ("syrk-edge-index", S, "while i + 4 <= r1 {\n        s0 = s0 + xs[i * rs + ja] * xs[i * rs + kb];", "while i + 4 <= r1 {\n        s0 = s0 + xs[i * rs + ja] * xs[i * rs + ja];"),
    ("syrk-edge-tail-lane", S, "if i + 1 < r1 {\n        s1 = s1 +", "if i + 1 < r1 {\n        s0 = s0 +"),
    ("syrk-block-hi", S, "var hi = (b + 1) * 256;\n        if hi > n {\n            hi = n;\n        }\n        block_tiles", "var hi = (b + 1) * 255;\n        if hi > n {\n            hi = n;\n        }\n        block_tiles"),
    ("syrk-tile-start", S, "var k0 = j0;\n        while k0 < d {", "var k0 = j0 + 2;\n        while k0 < d {"),
    ("syrk-generic-index", S, "xd[xo + i * rs + k * cs]", "xd[xo + i * rs + k]"),
    ("syrk-generic-lane", S, "let l = (i - r0) % 4;", "let l = (i - r0) % 2;"),
    ("syrk-n0", S, "if n == 0 {\n                // the sum over no rows", "if n == 1 {\n                // the sum over no rows"),
    ("syrk-store-upper", S, "var k = j;\n        while k < d {\n            cd[ct.off", "var k = j + 1;\n        while k < d {\n            cd[ct.off"),
    ("syrk-finish", S, "order.finish(contents(sw), sp, d * d);", "sp = sp;"),
])
GROUPS["chol"] = (["sh", "scripts/gate_linalg.sh", "t_chol.cho", "chol", "fwd", "bwd"], [
    ("chol-row-mix", C, "a31 = a31 + r3[k + 1] * c1;", "a31 = a31 + r2[k + 1] * c1;"),
    ("chol-store", C, "lv[(i0 + 3) * ls + j] = s3 / ljj;", "lv[(i0 + 3) * ls + j] = s2 / ljj;"),
    ("chol-init", C, "var s2 = av[(i0 + 2) * as_ + j];", "var s2 = av[(i0 + 3) * as_ + j];"),
    ("chol-rows4-sign", C, "s0 = s0 - order.lane_combine(a00, a01, a02, a03);", "s0 = s0 + order.lane_combine(a00, a01, a02, a03);"),
    ("chol-rows4-lane", C, "a12 = a12 + r1[k + 2] * c2;\n            a13", "a11 = a11 + r1[k + 2] * c2;\n            a13"),
    ("chol-rows4-tail-lane", C, 'a01 = a01 + r0[k + 1] * c;', 'a00 = a00 + r0[k + 1] * c;'),
    ("chol-rows4-panel", C, "var e = p0 + 32;\n        if e > j {\n            e = j;\n        }\n        var a00 = 0.0;", "var e = p0 + 16;\n        if e > j {\n            e = j;\n        }\n        var a00 = 0.0;"),
    ("chol-entry-panel", C, "var s = aij;\n    var p0 = 0;\n    while p0 < j {\n        var e = p0 + 32;\n        if e > j {\n            e = j;\n        }\n        var a0 = 0.0;\n        var a1 = 0.0;\n        var a2 = 0.0;\n        var a3 = 0.0;\n        var k = p0;\n        while k + 4 <= e {", "var s = aij;\n    var p0 = 0;\n    while p0 < j {\n        var e = p0 + 16;\n        if e > j {\n            e = j;\n        }\n        var a0 = 0.0;\n        var a1 = 0.0;\n        var a2 = 0.0;\n        var a3 = 0.0;\n        var k = p0;\n        while k + 4 <= e {"),
    ("chol-generic-panel", C, "var e = p0 + 32;\n        if e > j {\n            e = j;\n        }\n        var a0 = 0.0;\n        var a1 = 0.0;\n        var a2 = 0.0;\n        var a3 = 0.0;\n        var k = p0;\n        while k < e {", "var e = p0 + 16;\n        if e > j {\n            e = j;\n        }\n        var a0 = 0.0;\n        var a1 = 0.0;\n        var a2 = 0.0;\n        var a3 = 0.0;\n        var k = p0;\n        while k < e {"),
    ("chol-entry-step", C, "s = s - order.lane_combine(a0, a1, a2, a3);\n        p0 = p0 + 32;\n    }\n    return s;\n}\n\n// four rows", "s = s - order.lane_combine(a0, a1, a2, a3);\n        p0 = p0 + 16;\n    }\n    return s;\n}\n\n// four rows"),
    ("chol-entry-sign", C, "s = s - order.lane_combine(a0, a1, a2, a3);\n        p0 = p0 + 32;\n    }\n    return s;\n}\n\n// four rows", "s = s + order.lane_combine(a0, a1, a2, a3);\n        p0 = p0 + 32;\n    }\n    return s;\n}\n\n// four rows"),
    ("chol-entry-tail", C, "if k + 1 < e {\n            a1 = a1 + ri[k + 1] * rj[k + 1];", "if k + 1 < e {\n            a0 = a0 + ri[k + 1] * rj[k + 1];"),
    ("chol-entry-term", C, "a3 = a3 + ri[k + 3] * rj[k + 3];", "a3 = a3 + ri[k + 3] * ri[k + 3];"),
    ("chol-pivot", C, "if !(s > 0.0) {\n            return refuse.not_positive_definite();\n        }\n        let ljj", "if !(s >= 0.0) {\n            return refuse.not_positive_definite();\n        }\n        let ljj"),
    ("chol-rows4-guard", C, "while i + 4 <= d {\n            rows4(", "while i + 3 <= d {\n            rows4("),
    ("chol-generic-pivot", C, "if i == j {\n                if !(s > 0.0) {", "if i == j {\n                if !(s >= 0.0) {"),
    ("chol-generic-index", C, "lv[lt.off + j * lt.t0 + k * lt.t1];", "lv[lt.off + j * lt.t0 + k];"),
    ("chol-generic-lane", C, "let ln = (k - p0) % 4;", "let ln = (k - p0) % 2;"),
    ("fwd-group-term", C, "s3 = s3 - lv[(i + 3) * ls + i + 1] * x1;", "s3 = s3 - lv[(i + 3) * ls + i + 1] * x0;"),
    ("fwd-group-order", C, "s2 = s2 - lv[(i + 2) * ls + i] * x0;\n        s2 = s2 - lv[(i + 2) * ls + i + 1] * x1;", "s2 = s2 - lv[(i + 2) * ls + i + 1] * x1;\n        s2 = s2 - lv[(i + 2) * ls + i] * x0;"),
    ("fwd-tail", C, "s = s - r[k] * xs[k];\n            k = k + 1;\n        }\n        xv[i] = s / lv[i * ls + i];", "s = s - r[k] * xs[k];\n            k = k + 2;\n        }\n        xv[i] = s / lv[i * ls + i];"),
    ("bwd-group-term", C, "s1 = s1 - lv[(i + 2) * ls + i + 1] * x2;", "s1 = s1 - lv[(i + 2) * ls + i + 1] * x3;"),
    ("bwd-ascending", C, "var k = d - 1;\n        while k >= top {\n            let r = lv[k * ls + i..k * ls + i + 4];\n            let c = xv[k];\n            s0 = s0 - r[0] * c;\n            s1 = s1 - r[1] * c;\n            s2 = s2 - r[2] * c;\n            s3 = s3 - r[3] * c;\n            k = k - 1;\n        }", "var k = top;\n        while k < d {\n            let r = lv[k * ls + i..k * ls + i + 4];\n            let c = xv[k];\n            s0 = s0 - r[0] * c;\n            s1 = s1 - r[1] * c;\n            s2 = s2 - r[2] * c;\n            s3 = s3 - r[3] * c;\n            k = k + 1;\n        }"),
    ("bwd-tail-term", C, "s = s - lv[k * ls + i] * xv[k];\n            k = k - 1;\n        }\n        xv[i] = s / lv[i * ls + i];\n        i = i - 1;\n    }\n    return 0;\n}\n\n// The solves.", "s = s - lv[i * ls + k] * xv[k];\n            k = k - 1;\n        }\n        xv[i] = s / lv[i * ls + i];\n        i = i - 1;\n    }\n    return 0;\n}\n\n// The solves."),
    ("solve-singular", C, "if !(ld[lt.off + i * lt.t0 + i * lt.t1] != 0.0) {", "if !(ld[lt.off + i * lt.t0 + i * lt.t1] != 1.0) {"),
    ("solve-generic-bwd", C, "s = s - ld[lt.off + k * lt.t0 + i * lt.t1] * xd[xt.off + k * xt.t0];", "s = s - ld[lt.off + i * lt.t0 + k * lt.t1] * xd[xt.off + k * xt.t0];"),
])
GROUPS["order"] = (["sh", "scripts/gate_linalg.sh", "t_order.cho"], [
    ("order-lanes-2", O, "pub fn lanes() -> [] int {\n    return 4;", "pub fn lanes() -> [] int {\n    return 2;"),
    ("order-lanes-8", O, "pub fn lanes() -> [] int {\n    return 4;", "pub fn lanes() -> [] int {\n    return 8;"),
    ("order-panel-16", O, "pub fn panel() -> [] int {\n    return 32;", "pub fn panel() -> [] int {\n    return 16;"),
    ("order-block-128", O, "pub fn block() -> [] int {\n    return 256;", "pub fn block() -> [] int {\n    return 128;"),
    ("order-group-4", O, "pub fn group() -> [] int {\n    return 8;", "pub fn group() -> [] int {\n    return 4;"),
    ("order-lane-combine-fold", O, "return combine(combine(l0, l1), combine(l2, l3));", "return combine(combine(combine(l0, l1), l2), l3);"),
    ("order-lane-combine-pairs", O, "return combine(combine(l0, l1), combine(l2, l3));", "return combine(combine(l0, l2), combine(l1, l3));"),
    ("order-partial-lane-map", O, "s1 = s1 + buf[j + 1];\n        s2 = s2 + buf[j + 2];", "s1 = s1 + buf[j + 2];\n        s2 = s2 + buf[j + 1];"),
    ("order-partial-tail", O, "    if j < m {\n        s0 = s0 + buf[j];\n        j = j + 1;\n    }", "    if j < m {\n        s1 = s1 + buf[j];\n        j = j + 1;\n    }"),
    ("order-partial-init", O, "var s0 = 0.0;\n    var s1 = 0.0;\n    var s2 = 0.0;\n    var s3 = 0.0;\n    var j = 0;\n    while j + 4 <= m {\n        s0 = s0 + buf[j];", "var s0 = 0.0;\n    var s1 = 0.0;\n    var s2 = 0.0;\n    var s3 = 1.0e-16;\n    var j = 0;\n    while j + 4 <= m {\n        s0 = s0 + buf[j];"),
])
GROUPS["fuzz"] = (["sh", "scripts/gate_linalg.sh", "t_fuzz.cho"], [  # the check-normalised mutant is judged by tests/core.cho (mutants_core.py)
    ("fuzz-sane-in-check", "lib/linalg/tensor.cho", "    if sane(t) == false {\n        return refuse.tensor_bounds();\n    }\n    // each (extent - 1)", "    // each (extent - 1)"),
    ("fuzz-reach-guard", "lib/linalg/tensor.cho", "if too_big(shape_at(t, a) - 1, abs_c(stride_at(t, a))) == 1 {\n            return refuse.tensor_bounds();", "if false {\n            return refuse.tensor_bounds();"),
])
GROUPS["par"] = (["sh", "scripts/gate_par.sh"], [
    # THE required mutant: the partial sums are added in thread order, as a library reduction does (each thread keeps a running
    # sum of its own chunks' roots, the parent adds the threads' sums in thread order). It must fail at more than one thread.
    ("par-thread-order", "lib/linalg/par.cho", "    var sp = 0;\n    var c = 0;\n    while c < nchunks {\n        sp = order.settle_root(st, lv, sp, w, roots, c * w, 3);\n        c = c + 1;\n    }\n    order.finish(st, sp, w);\n    return 0;", "    var j = 0;\n    while j < w {\n        st[j] = 0.0;\n        j = j + 1;\n    }\n    var k = 0;\n    while k < t {\n        j = 0;\n        while j < w {\n            st[w + j] = 0.0;\n            j = j + 1;\n        }\n        var c = bound(k, t, nchunks);\n        let c1 = bound(k + 1, t, nchunks);\n        while c < c1 {\n            j = 0;\n            while j < w {\n                st[w + j] = st[w + j] + roots[c * w + j];\n                j = j + 1;\n            }\n            c = c + 1;\n        }\n        j = 0;\n        while j < w {\n            st[j] = st[j] + st[w + j];\n            j = j + 1;\n        }\n        k = k + 1;\n    }\n    return 0;"),
    ("par-chunk-misaligned", "lib/linalg/par.cho", "r0 = c0 * 2048;\n        r1 = c1 * 2048;", "r0 = c0 * 2047;\n        r1 = c1 * 2047;"),
    ("par-no-finish", "lib/linalg/par.cho", "    order.finish(st, sp, w);\n    return 0;\n}\n\n// the threads actually used", "    return 0;\n}\n\n// the threads actually used"),
    ("par-ragged-clamp", "lib/linalg/par.cho", "if b1 > nb {\n            b1 = nb;\n        }\n        var sp = 0;", "var sp = 0;"),
    ("par-gemm-rows", "lib/linalg/par.cho", "r0 = bound(k, t, tiles) * 4;", "r0 = bound(k, t, tiles) * 4 + 1;"),
    ("par-copy-row", "lib/linalg/par.cho", "xd[(r0 + i) * rs..(r0 + i) * rs + acols]", "xd[(r0 + i + 1) * rs..(r0 + i + 1) * rs + acols]"),
    ("par-slot-copy", "lib/linalg/par.cho", "copy_floats(slots[c0 * w..c1 * w], contents(ow), (c1 - c0) * w);", "copy_floats(slots[c0 * w..c1 * w], contents(ow), (c1 - c0) * w - 1);"),
    ("par-bound", "lib/linalg/par.cho", "return k * nchunks / t;", "return k * nchunks / t + k / 3;"),
    ("par-dot-second-operand", "lib/linalg/par.cho", "copy_floats(contents(jk.b), yd[r0..r1], rows);\n        } else {", "copy_floats(contents(jk.b), xd[r0..r1], rows);\n        } else {"),
])
GROUPS["gemm"] = (["sh", "scripts/gate_linalg.sh", "t_gemm.cho", "gemm"], [
    ("gemm-tile-entry", M, "c113 = c113 + p1[k + 3] * q3[1];", "c113 = c113 + p0[k + 3] * q3[1];"),
    ("gemm-tile-col", M, "c101 = c101 + p1[k + 1] * q1[0];", "c101 = c101 + p1[k + 1] * q1[1];"),
    ("gemm-tile-init", M, "var c103 = 0.0;", "var c103 = 1.0e-16;"),
    ("gemm-tile-lane", M, "c012 = c012 + p0[k + 2] * q2[1];", "c011 = c011 + p0[k + 2] * q2[1];"),
    ("gemm-tile-tail", M, 'c001 = c001 + p0[k + 1] * q[0];', 'c000 = c000 + p0[k + 1] * q[0];'),
    ("gemm-store", M, "out[3] = order.lane_combine(c110, c111, c112, c113);", "out[3] = order.lane_combine(c100, c101, c102, c103);"),
    ("gemm-combine", M, "out[1] = order.lane_combine(c010, c011, c012, c013);", "out[1] = order.lane_combine(c010, c012, c011, c013);"),
    ("gemm-block", M, "var b1 = b0 + 256;\n                    if b1 > p {\n                        b1 = p;\n                    }\n                    tile22", "var b1 = b0 + 255;\n                    if b1 > p {\n                        b1 = p;\n                    }\n                    tile22"),
    ("gemm-edge-term", M, "st[sp] = lane_run(av, q * ars, 1, bv, w, brs, b0, b1);", "st[sp] = lane_run(av, q * ars, 1, bv, j0, brs, b0, b1);"),
    ("gemm-edge-block", M, "var b1 = b0 + 256;\n                            if b1 > p {\n                                b1 = p;\n                            }\n                            st[sp] = lane_run(av", "var b1 = b0 + 255;\n                            if b1 > p {\n                                b1 = p;\n                            }\n                            st[sp] = lane_run(av"),
    ("gemm-generic-block", M, "var b1 = b0 + 256;\n                            if b1 > p {\n                                b1 = p;\n                            }\n                            let s = lane_run(ad", "var b1 = b0 + 255;\n                            if b1 > p {\n                                b1 = p;\n                            }\n                            let s = lane_run(ad"),
    ("gemm-generic-term", M, "bt.off + j * bt.t1, bt.t0, b0, b1)", "bt.off + j, bt.t0, b0, b1)"),
    ("gemm-run-lane", M, "let l = (k - k0) % 4;", "let l = (k - k0) % 2;"),
    ("gemm-run-init", M, "var s3 = 0.0;\n    var k = k0;", "var s3 = 1.0e-16;\n    var k = k0;"),
    ("gemm-tile-guard", M, "if i0 + 2 <= r1 && j0 + 2 <= n {", "if i0 + 2 <= r1 && j0 + 1 <= n {"),
])

ALLOWED = {
    "order-blocks": "equivalent: for n a multiple of 256 it counts one extra EMPTY block, whose partial is +0.0 (a block sum starts at +0.0, so it is never -0.0); adding +0.0 to a partial changes no bit",
    "order-level-eq": "equivalent: the levels on the stack are strictly decreasing from bottom to top, so the top is never greater than the entry below it; >= and == select the same merges",
}

# Deliberately wrong ORACLES (scripts/oracle_linalg.py, ORACLE_MUTANT): the suite against the unmutated binary must FAIL under each,
# which shows that the oracle itself would see a kernel that took the other choice (the constants of the lane order among them).
ORACLE_MUTANTS = ["tree-leftfold", "tree-carry-add", "block-255", "dot-reverse", "gemvt-reverse", "bwd-ascending",
                  "lanes-2", "lanes-8", "lane-map", "lane-combine-fold", "chol-panel-16"]


def oracle_mutants():
    import os, subprocess, tempfile
    cancho = mutlib.cancho()
    d = tempfile.mkdtemp()
    exe = d + "/linalg-check"
    files = sorted(str(f) for f in (mutlib.ROOT / "lib/core").glob("*.cho")) + sorted(str(f) for f in (mutlib.ROOT / "lib/linalg").glob("*.cho"))
    r = subprocess.run([cancho, "build", "--std", *files, str(mutlib.ROOT / "tools/linalg-check/main.cho"), "-o", exe], capture_output=True, text=True)
    if r.returncode:
        print("BUILD_ERROR oracle mutants\n" + r.stdout[-500:] + r.stderr[-500:])
        return 1
    ok = subprocess.run([sys.executable, str(mutlib.ROOT / "scripts/oracle_linalg.py"), "--suite"], env=dict(os.environ, LINALG_CHECK=exe), capture_output=True, text=True)
    if ok.returncode != 0:
        print("mutants_linalg/oracle: the unmutated oracle FAILS the suite; refusing to run its mutants")
        return 1
    survivors = []
    for m in ORACLE_MUTANTS:
        r = subprocess.run([sys.executable, str(mutlib.ROOT / "scripts/oracle_linalg.py"), "--suite"], env=dict(os.environ, LINALG_CHECK=exe, ORACLE_MUTANT=m), capture_output=True, text=True)
        print("%-40s %s" % (m, "killed" if r.returncode != 0 else "SURVIVED"), flush=True)
        if r.returncode == 0:
            survivors.append(m)
    print("mutants_linalg/oracle: %d oracle mutants, %d survive" % (len(ORACLE_MUTANTS), len(survivors)))
    return 1 if survivors else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "oracle" in args:
        sys.exit(oracle_mutants() if "--check" not in sys.argv else 0)
    rc = 0
    whole = [a for a in args if a in GROUPS]
    subs = [a for a in args if a not in GROUPS]
    for g in GROUPS:
        cmd, muts = GROUPS[g]
        if whole and not subs and g not in whole:
            continue
        if g in whole:
            pick = muts
        elif not args:
            pick = muts
        else:
            pick = [m for m in muts if any(a in m[0] for a in subs)]
        if pick:
            rc |= mutlib.run(pick, cmd, "mutants_linalg/" + g, allowed=ALLOWED)
    if not args and "--check" not in sys.argv:
        rc |= oracle_mutants()
    sys.exit(rc)

#!/usr/bin/env python3
"""Source mutants of lib/linalg, per kernel group. Each mutant is applied to a copy of the tree; the group's gate
(scripts/gate_linalg.sh: unit tests against the reference, then the compiled driver against the numpy oracle) must fail.

    python3 scripts/mutants_linalg.py --check                pattern check only
    python3 scripts/mutants_linalg.py [group|name-substring ...]
Groups: dot axpy gemv syrk chol gemm par. Survivors are listed with the reason in ALLOWED."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import mutlib

D, A, G, S, C, M, O = ("lib/linalg/dot.cho", "lib/linalg/axpy.cho", "lib/linalg/gemv.cho", "lib/linalg/syrk.cho",
                       "lib/linalg/chol.cho", "lib/linalg/gemm.cho", "lib/linalg/order.cho")

GROUPS = {}
GROUPS["dot"] = (["sh", "scripts/gate_linalg.sh", "t_dot.cho", "dot"], [
    ("dot-chain-init", D, "var s3 = 0.0;", "var s3 = 1.0e-16;"),
    ("dot-chain-swap", D, "out[1] = s1;\n    out[2] = s2;", "out[1] = s2;\n    out[2] = s1;"),
    ("dot-ragged-block", D, "var m = n - b * 256;\n            if m > 256 {\n                m = 256;", "var m = n - b * 256;\n            if m > 255 {\n                m = 255;"),
    ("dot-chunk-guard", D, "if b + 8 <= b1 && (b + 8) * 256 <= n {", "if b + 8 <= b1 && (b + 8) * 256 <= n + 256 {"),
    ("dot-block1-init", D, "var s = 0.0;\n    var j = 0;\n    while j < m {\n        s = s + xr[j] * yr[j];", "var s = 1.0e-16;\n    var j = 0;\n    while j < m {\n        s = s + xr[j] * yr[j];"),
    ("dot-strided-block", D, "var m = n - b * 256;\n    if m > 256 {\n        m = 256;\n    }\n    var s = 0.0;\n    var j = 0;\n    while j < m {\n        let i = b * 256 + j;", "var m = n - b * 256;\n    if m > 255 {\n        m = 255;\n    }\n    var s = 0.0;\n    var j = 0;\n    while j < m {\n        let i = b * 256 + j;"),
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
    ("gemvn-row-mix", G, "z3 = z3 + r3[j] * wj;", "z3 = z3 + r2[j] * wj;"),
    ("gemvn-init", G, "var z3 = 0.0;", "var z3 = 1.0e-16;"),
    ("gemvn-tail-init", G, "var z = 0.0;\n    var j = 0;\n    while j < d {\n        z = z + r[j] * wr[j];", "var z = 1.0e-16;\n    var j = 0;\n    while j < d {\n        z = z + r[j] * wr[j];"),
    ("gemvn-store", G, "ys[i + 2] = z2;\n    ys[i + 3] = z3;", "ys[i + 2] = z3;\n    ys[i + 3] = z2;"),
    ("gemvn-fast-guard", G, "&& d <= 256 && xt.t0 >= 0 {", "&& d <= 300 && xt.t0 >= 0 {"),
    ("gemvn-generic-block", G, "var m = d - b * 256;\n                    if m > 256 {\n                        m = 256;", "var m = d - b * 256;\n                    if m > 255 {\n                        m = 255;"),
    ("gemvn-generic-stride", G, "wd[wt.off + jj * wt.t0]", "wd[wt.off + jj]"),
    ("gemvn-d0", G, "if d == 0 {\n        var i = 0;\n        while i < n {\n            yd[yt.off + i * yt.t0] = 0.0;", "if d == 0 {\n        var i = 0;\n        while i < n {\n            yd[yt.off + i * yt.t0] = 1.0;"),
    ("gemvt-acc-init", G, "acc[j] = 0.0;\n        j = j + 1;\n    }\n    var i = r0;", "acc[j] = 1.0e-16;\n        j = j + 1;\n    }\n    var i = r0;"),
    ("gemvt-row-order", G, "s = s + a1[j] * c1;\n            s = s + a2[j] * c2;", "s = s + a2[j] * c2;\n            s = s + a1[j] * c1;"),
    ("gemvt-row-coef", G, "s = s + a3[j] * c3;", "s = s + a3[j] * c2;"),
    ("gemvt-tail-row", G, "ac[j] = ac[j] + a0[j] * c0;", "ac[j] = ac[j] + a0[j] * rv[r0];"),
    ("gemvt-block-hi", G, "var hi = (b + 1) * 256;\n        if hi > n {\n            hi = n;\n        }\n        block_acc", "var hi = (b + 1) * 255;\n        if hi > n {\n            hi = n;\n        }\n        block_acc"),
    ("gemvt-generic-hi", G, "var hi = (b + 1) * 256;\n                    if hi > n {\n                        hi = n;\n                    }\n                    let a = sp * d;", "var hi = (b + 1) * 255;\n                    if hi > n {\n                        hi = n;\n                    }\n                    let a = sp * d;"),
    ("gemvt-generic-index", G, "xd[xt.off + i * xt.t0 + j * xt.t1] * ri", "xd[xt.off + i * xt.t0 + j] * ri"),
    ("gemvt-finish", G, "order.finish(contents(sw), sp, d);\n            j = 0;", "j = 0;"),
])
GROUPS["syrk"] = (["sh", "scripts/gate_linalg.sh", "t_syrk.cho", "syrk"], [
    ("syrk-tile-entry", S, "c01 = c01 + a0 * b1;", "c01 = c01 + a1 * b0;"),
    ("syrk-tile-row", S, "c23 = c23 + a2 * b3;", "c23 = c23 + a2 * b2;"),
    ("syrk-tile-init", S, "var c33 = 0.0;", "var c33 = 1.0e-16;"),
    ("syrk-tile-store", S, "o3[3] = c33;", "o3[3] = c32;"),
    ("syrk-edge-init", S, "var c = 0.0;\n            var i = r0;\n            while i < r1 {\n                c = c + xs[i * rs + j0 + j]", "var c = 1.0e-16;\n            var i = r0;\n            while i < r1 {\n                c = c + xs[i * rs + j0 + j]"),
    ("syrk-edge-index", S, "xs[i * rs + k0 + k];", "xs[i * rs + k0];"),
    ("syrk-block-hi", S, "var hi = (b + 1) * 256;\n        if hi > n {\n            hi = n;\n        }\n        block_tiles", "var hi = (b + 1) * 255;\n        if hi > n {\n            hi = n;\n        }\n        block_tiles"),
    ("syrk-tile-start", S, "var k0 = j0;\n        while k0 < d {", "var k0 = j0 + 4;\n        while k0 < d {"),
    ("syrk-generic-index", S, "xd[xo + i * rs + k * cs]", "xd[xo + i * rs + k]"),
    ("syrk-n0", S, "if n == 0 {\n                // the sum over no rows", "if n == 1 {\n                // the sum over no rows"),
    ("syrk-store-upper", S, "var k = j;\n        while k < d {\n            cd[ct.off", "var k = j + 1;\n        while k < d {\n            cd[ct.off"),
    ("syrk-finish", S, "order.finish(contents(sw), sp, d * d);", "sp = sp;"),
])
GROUPS["chol"] = (["sh", "scripts/gate_linalg.sh", "t_chol.cho", "chol", "fwd", "bwd"], [
    ("chol-row-mix", C, "s5 = s5 - r5[k] * c;", "s5 = s5 - r4[k] * c;"),
    ("chol-store", C, "lv[(i0 + 7) * ls + j] = s7 / ljj;", "lv[(i0 + 7) * ls + j] = s6 / ljj;"),
    ("chol-init", C, "var s2 = av[(i0 + 2) * as_ + j];", "var s2 = av[(i0 + 3) * as_ + j];"),
    ("chol-diag-term", C, "s = s - rj[k] * rj[k];\n        k = k + 1;\n    }\n    return s;", "s = s + rj[k] * rj[k];\n        k = k + 1;\n    }\n    return s;"),
    ("chol-pivot", C, "if !(s > 0.0) {\n            return refuse.not_positive_definite();\n        }\n        let ljj", "if !(s >= 0.0) {\n            return refuse.not_positive_definite();\n        }\n        let ljj"),
    ("chol-tail-term", C, "t = t - ri[k] * rj[k];", "t = t - ri[k] * ri[k];"),
    ("chol-rows8-guard", C, "while i + 8 <= d {\n            rows8(", "while i + 7 <= d {\n            rows8("),
    ("chol-generic-pivot", C, "if i == j {\n                if !(s > 0.0) {", "if i == j {\n                if !(s >= 0.0) {"),
    ("chol-generic-index", C, "lv[lt.off + j * lt.t0 + k * lt.t1];", "lv[lt.off + j * lt.t0 + k];"),
    ("fwd-group-term", C, "s3 = s3 - lv[(i + 3) * ls + i + 1] * x1;", "s3 = s3 - lv[(i + 3) * ls + i + 1] * x0;"),
    ("fwd-group-order", C, "s2 = s2 - lv[(i + 2) * ls + i] * x0;\n        s2 = s2 - lv[(i + 2) * ls + i + 1] * x1;", "s2 = s2 - lv[(i + 2) * ls + i + 1] * x1;\n        s2 = s2 - lv[(i + 2) * ls + i] * x0;"),
    ("fwd-tail", C, "s = s - r[k] * xs[k];\n            k = k + 1;\n        }\n        xv[i] = s / lv[i * ls + i];", "s = s - r[k] * xs[k];\n            k = k + 2;\n        }\n        xv[i] = s / lv[i * ls + i];"),
    ("bwd-group-term", C, "s1 = s1 - lv[(i + 2) * ls + i + 1] * x2;", "s1 = s1 - lv[(i + 2) * ls + i + 1] * x3;"),
    ("bwd-ascending", C, "var k = d - 1;\n        while k >= top {\n            let r = lv[k * ls + i..k * ls + i + 4];\n            let c = xv[k];\n            s0 = s0 - r[0] * c;\n            s1 = s1 - r[1] * c;\n            s2 = s2 - r[2] * c;\n            s3 = s3 - r[3] * c;\n            k = k - 1;\n        }", "var k = top;\n        while k < d {\n            let r = lv[k * ls + i..k * ls + i + 4];\n            let c = xv[k];\n            s0 = s0 - r[0] * c;\n            s1 = s1 - r[1] * c;\n            s2 = s2 - r[2] * c;\n            s3 = s3 - r[3] * c;\n            k = k + 1;\n        }"),
    ("bwd-tail-term", C, "s = s - lv[k * ls + i] * xv[k];\n            k = k - 1;\n        }\n        xv[i] = s / lv[i * ls + i];\n        i = i - 1;\n    }\n    return 0;\n}\n\n// The solves.", "s = s - lv[i * ls + k] * xv[k];\n            k = k - 1;\n        }\n        xv[i] = s / lv[i * ls + i];\n        i = i - 1;\n    }\n    return 0;\n}\n\n// The solves."),
    ("solve-singular", C, "if !(ld[lt.off + i * lt.t0 + i * lt.t1] != 0.0) {", "if !(ld[lt.off + i * lt.t0 + i * lt.t1] != 1.0) {"),
    ("solve-generic-bwd", C, "s = s - ld[lt.off + k * lt.t0 + i * lt.t1] * xd[xt.off + k * xt.t0];", "s = s - ld[lt.off + i * lt.t0 + k * lt.t1] * xd[xt.off + k * xt.t0];"),
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
    ("gemm-tile-entry", M, "c13 = c13 + a1 * b3;", "c13 = c13 + a2 * b3;"),
    ("gemm-tile-col", M, "c30 = c30 + a3 * b0;", "c30 = c30 + a3 * b1;"),
    ("gemm-tile-init", M, "var c22 = 0.0;", "var c22 = 1.0e-16;"),
    ("gemm-store", M, "out[15] = c33;", "out[15] = c32;"),
    ("gemm-block", M, "var b1 = b0 + 256;\n                    if b1 > p {\n                        b1 = p;\n                    }\n                    tile44", "var b1 = b0 + 255;\n                    if b1 > p {\n                        b1 = p;\n                    }\n                    tile44"),
    ("gemm-edge-term", M, "s = s + av[q * ars + k] * bv[k * brs + w];", "s = s + av[q * ars + k] * bv[k * brs + j0];"),
    ("gemm-edge-block", M, "var b1 = b0 + 256;\n                            if b1 > p {\n                                b1 = p;\n                            }\n                            var s = 0.0;\n                            var k = b0;\n                            while k < b1 {\n                                s = s + av[q", "var b1 = b0 + 255;\n                            if b1 > p {\n                                b1 = p;\n                            }\n                            var s = 0.0;\n                            var k = b0;\n                            while k < b1 {\n                                s = s + av[q"),
    ("gemm-generic-block", M, "var b1 = b0 + 256;\n                            if b1 > p {\n                                b1 = p;\n                            }\n                            var s = 0.0;\n                            var k = b0;\n                            while k < b1 {\n                                s = s + ad[", "var b1 = b0 + 255;\n                            if b1 > p {\n                                b1 = p;\n                            }\n                            var s = 0.0;\n                            var k = b0;\n                            while k < b1 {\n                                s = s + ad["),
    ("gemm-generic-term", M, "bd[bt.off + k * bt.t0 + j * bt.t1]", "bd[bt.off + k * bt.t0 + j]"),
    ("gemm-tile-guard", M, "if i0 + 4 <= r1 && j0 + 4 <= n {", "if i0 + 4 <= r1 && j0 + 3 <= n {"),
])

ALLOWED = {
    "order-blocks": "equivalent: for n a multiple of 256 it counts one extra EMPTY block, whose partial is +0.0 (a block sum starts at +0.0, so it is never -0.0); adding +0.0 to a partial changes no bit",
    "order-level-eq": "equivalent: the levels on the stack are strictly decreasing from bottom to top, so the top is never greater than the entry below it; >= and == select the same merges",
}

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
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
    sys.exit(rc)

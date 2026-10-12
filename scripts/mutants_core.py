#!/usr/bin/env python3
"""Mutants of lib/core and lib/linalg/tensor.cho: tests/core.cho must fail on each."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import mutlib

T = "lib/linalg/tensor.cho"
M = [
 ("rng-gold",        "lib/core/rng.cho", "0 - 7046029254386353131", "0 - 7046029254386353130"),
 ("rng-shift30",     "lib/core/rng.cho", "z >> 30 & 0x3FFFFFFFF", "z >> 29 & 0x7FFFFFFFF"),
 ("rng-mult1",       "lib/core/rng.cho", "0 - 4658895280553007687", "0 - 4658895280553007689"),
 ("rng-mult2",       "lib/core/rng.cho", "0 - 7723592293110705685", "0 - 7723592293110705687"),
 ("rng-final-shift", "lib/core/rng.cho", "z >> 31 & 0x1FFFFFFFF", "z >> 32 & 0xFFFFFFFF"),
 ("rng-derive-xor",  "lib/core/rng.cho", "mix(wrapping_add(h ^ c, gold()))", "mix(wrapping_add(h + c, gold()))"),
 ("rng-uniform-shift", "lib/core/rng.cho", "(z >> 1 & 0x7FFFFFFFFFFFFFFF) % m", "(z & 0x7FFFFFFFFFFFFFFF) % m"),
 ("rng-uniform-empty", "lib/core/rng.cho", "if m <= 0 {\n        return 0;\n    }", "if m < 0 {\n        return 0;\n    }"),
 ("rng-unit-offset", "lib/core/rng.cho", "4503599627370496.0 - 1.0", "4503599627370496.0 - 0.999"),
 ("rng-unit-mask",   "lib/core/rng.cho", "z >> 11 & 0x1FFFFFFFFFFFFF", "z >> 12 & 0x1FFFFFFFFFFFFF"),
 ("fbits-scale",     "lib/core/fbits.cho", "math.ldexp(1.0, e - 1075)", "math.ldexp(1.0, e - 1074)"),
 ("fbits-implicit",  "lib/core/fbits.cho", "float_of(m + 4503599627370496) * scale[e]", "float_of(m) * scale[e]"),
 ("fbits-sign",      "lib/core/fbits.cho", "if bits < 0 {\n        v = v * (0.0 - 1.0);\n    }", "if bits < 0 && v != 0.0 {\n        v = v * (0.0 - 1.0);\n    }"),
 ("fbits-subnormal", "lib/core/fbits.cho", "math.ldexp(float_of(m), 0 - 1074)", "math.ldexp(float_of(m), 0 - 1073)"),
 ("fbits-nan",       "lib/core/fbits.cho", "if e == 2047 {\n        return 0.0 / 0.0;\n    }", "if e == 2048 {\n        return 0.0 / 0.0;\n    }"),
 ("fbits-endian",    "lib/core/fbits.cho", "var k = 7;\n    while k >= 0 {", "var k = 6;\n    while k >= 0 {"),
 ("fbits-put-shift", "lib/core/fbits.cho", "bits >> 8 * k & 255", "bits >> 7 * k & 255"),
 ("fbits-load-check", "lib/core/fbits.cho", "} else if bits_of(v) != bits {\n            bad = bad + 1;\n        }", "}"),
 ("limits-rows",     "lib/core/limits.cho", "if rows > rows_cap {", "if rows >= rows_cap {"),
 ("limits-cells",    "lib/core/limits.cho", "if rows > cells_cap / cols {", "if rows >= cells_cap / cols {"),
 ("limits-mul",      "lib/core/limits.cho", "if rows > cells_cap / cols {", "if rows * cols > cells_cap {"),
 ("limits-empty",    "lib/core/limits.cho", "if rows == 0 || cols == 0 {", "if rows == 0 {"),
 ("limits-ceiling",  "lib/core/limits.cho", "|| cells_cap > ceiling_cells()", "|| cells_cap > ceiling_cells() + 1"),
 ("limits-threads",  "lib/core/limits.cho", "if t < 1 || t > 64 {", "if t < 1 || t > 65 {"),
 ("refuse-exit",     "lib/core/refuse.cho", "if code == 8 || code == 11 {", "if code == 8 {"),
 ("refuse-dup-tag",  "lib/core/refuse.cho", "return \"ml.input.too-many-cells\";", "return \"ml.input.too-many-rows\";"),
 ("refuse-tag-rank", "lib/core/refuse.cho", "return \"ml.tensor.rank\";", "return \"ml.tensor.rnk\";"),
 ("tensor-rank-hi",  T, "if rank < 1 || rank > 4 {\n        return Made::Err(refuse.tensor_rank());\n    }\n    var a0", "if rank < 1 || rank > 5 {\n        return Made::Err(refuse.tensor_rank());\n    }\n    var a0"),
 ("tensor-negshape", T, "if a0 < 0 || a1 < 0 || a2 < 0 || a3 < 0 {", "if a0 < 0 || a1 < 0 || a2 < 0 {"),
 ("tensor-count-guard", T, "    if too_big(count, a3) == 1 {\n        return Made::Err(refuse.tensor_shape());\n    }\n    // a stride", "    // a stride"),
 ("tensor-normalise", T, "    if rank < 3 {\n        a2 = 1;\n        b2 = 0;\n    }\n    if rank < 2 {\n        a1 = 1;\n        b1 = 0;\n    }\n    if a0", "    if rank < 2 {\n        a1 = 1;\n        b1 = 0;\n    }\n    if a0"),
 ("tensor-too-big",  T, "if a > ceiling() / b {", "if a >= ceiling() / b {"),
 ("tensor-reach-lo", T, "if reach_lo(t) < 0 || reach_hi(t) >= len {", "if reach_hi(t) >= len {"),
 ("tensor-reach-hi", T, "if reach_lo(t) < 0 || reach_hi(t) >= len {", "if reach_lo(t) < 0 || reach_hi(t) > len {"),
 ("tensor-neg-stride", T, "        if st < 0 {\n            lo = lo + (shape_at(t, a) - 1) * st;", "        if st < 1 {\n            lo = lo + (shape_at(t, a) - 1) * st + st;"),
 ("tensor-empty",    T, "    if t.s0 == 0 || t.s1 == 0 || t.s2 == 0 || t.s3 == 0 {\n        return 0;\n    }\n    if sane(t) == false {\n        return refuse.tensor_bounds();", "    if sane(t) == false {\n        return refuse.tensor_bounds();"),
 ("tensor-sane-off", T, "|| abs_c(t.off) > c {", "{"),
 ("tensor-slice-hi", T, "if lo < 0 || hi < lo || hi > ext {", "if lo < 0 || hi < lo || hi > ext + 1 {"),
 ("tensor-slice-off", T, "let off = t.off + lo * st;", "let off = t.off + lo;"),
 ("tensor-transpose", T, "return make(t.off, 2, t.s1, t.s0, 1, 1, t.t1, t.t0, 0, 0);", "return make(t.off, 2, t.s1, t.s0, 1, 1, t.t0, t.t1, 0, 0);"),
 ("tensor-contig",   T, "if ext != 1 && stride_at(t, a) != want {", "if stride_at(t, a) != want {"),
 ("tensor-reshape-count", T, "if count(t) < 0 || m0 * m1 * m2 * m3 != count(t) {", "if count(t) < 0 {"),
 ("tensor-reshape-strides", T, "let r1 = m3 * m2;", "let r1 = m3 * m2 * 2;"),
 ("tensor-bcast-extent", T, "|| shape_at(t, axis) != 1 {", "{"),
 ("tensor-bcast-stride", T, "    if axis == 0 {\n        n0 = n;\n        b0 = 0;\n    }", "    if axis == 0 {\n        n0 = n;\n    }"),
 ("tensor-row-bounds", T, "    if i < 0 || i >= t.s0 {", "    if i < 0 || i > t.s0 {"),
 ("tensor-col-off", T, "return make(t.off + j * t.t1, 1, t.s0, 1, 1, 1, t.t0, 0, 0, 0);", "return make(t.off + j, 1, t.s0, 1, 1, 1, t.t0, 0, 0, 0);"),
 ("tensor-check-normalised", T, "|| t.s3 < 0 || normalised(t) == false {", "|| t.s3 < 0 {"),
 ("tensor-normalised-axes", T, "if t.rank < 2 && (t.s1 != 1 || t.t1 != 0) {", "if t.rank < 2 && (t.t1 != 0) {"),
 ("tensor-abs-clamp", T, "return ceiling() + 1;\n    }\n    if v < 0 {", "return 0 - ceiling();\n    }\n    if v < 0 {"),
]
ALLOWED = {
 "fbits-load-check": "equivalent: the roundtrip check in load() can only fire if decode() is wrong, which test_fbits_* would catch first; it is defence in depth",
 "limits-mul": "equivalent: the caps are bounded by ceiling_dim (2^31 - 1), so rows * cols is below 2^62 and cannot overflow; the division form is defence in depth",
}
sys.exit(mutlib.run(M, ["{cancho}", "test", "--std", "lib/core/rng.cho", "lib/core/fbits.cho", "lib/core/refuse.cho", "lib/core/limits.cho", "lib/linalg/tensor.cho", "tests/core.cho"], "mutants_core", allowed=ALLOWED))

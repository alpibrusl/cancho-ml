#!/usr/bin/env python3
"""Source mutants of lib/metrics (docs/metrics.md section 6). Each is a one-place edit applied to a copy of the tree; the whole gate
(scripts/gate_metrics.sh: unit tests, the driver against the independent oracles, tag fixtures, fuzz) must fail on it.

    python3 scripts/mutants_metrics.py --check          every pattern applies exactly once (no compiler)
    python3 scripts/mutants_metrics.py [name-substring ...]
Survivors are listed with a reason in ALLOWED; an unexplained survivor fails the script."""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import mutlib

os.environ.setdefault("DOCS_ROOT", str(mutlib.ROOT))

SP, CL, RK, RG, LL, SU, RA, RU, OR = ("lib/metrics/splits.cho", "lib/metrics/classify.cho", "lib/metrics/rank.cho", "lib/metrics/regress.cho",
                                      "lib/metrics/logloss.cho", "lib/metrics/sums.cho", "lib/metrics/ratio.cho", "lib/metrics/rules.cho",
                                      "lib/linalg/order.cho")
M = [
    # ---- seeds, streams, the draw
    ("seed-derive-index", SP, "return rng.derive(rng.derive(seed, kind), index);", "return rng.derive(rng.derive(seed, kind), index + 1);"),
    ("seed-derive-swap", SP, "return rng.derive(rng.derive(seed, kind), index);", "return rng.derive(rng.derive(seed, index), kind);"),
    ("seed-derive-single", SP, "return rng.derive(rng.derive(seed, kind), index);", "return rng.derive(seed, kind + index);"),
    ("kind-kfold-collides", SP, "pub fn kind_kfold() -> [] int {\n    return 5;", "pub fn kind_kfold() -> [] int {\n    return 4;"),
    ("kind-stratified", SP, "pub fn kind_stratified() -> [] int {\n    return 6;", "pub fn kind_stratified() -> [] int {\n    return 7;"),
    ("strat-class1-stream", SP, "stream(seed, kind_stratified(), 2 * rep + 1)", "stream(seed, kind_stratified(), 2 * rep)"),
    ("strat-class0-stream", SP, "stream(seed, kind_stratified(), 2 * rep));", "stream(seed, kind_stratified(), rep));"),
    ("draw-plain-modulus", SP, "return top - r;\n}", "return top;\n}"),
    ("draw-limit-off-by-one", SP, "return top - r;\n}", "return top - r - 1;\n}"),
    ("draw-accept-above", SP, "if v <= limit {\n            going = false;", "if v <= limit + 1 {\n            going = false;"),
    ("draw-accept-strict", SP, "if v <= limit {\n            going = false;", "if v < limit {\n            going = false;"),
    ("draw-r-formula", SP, "let r = (top % m + 1) % m;", "let r = top % m;"),
    ("draw-shift", SP, "v = rng.output(s) >> 1 & 9223372036854775807;", "v = rng.output(s) & 9223372036854775807;"),
    ("draw-no-advance", SP, "        s = rng.advance(s);\n        v = rng.output(s)", "        v = rng.output(s)"),
    ("shuffle-stop", SP, "var i = len - 1;\n    while i >= 1 {", "var i = len - 1;\n    while i >= 2 {"),
    ("shuffle-range", SP, "bounded(s, i + 1);", "bounded(s, i);"),
    ("shuffle-swap", SP, "a[lo + i] = a[lo + j];\n        a[lo + j] = t;", "a[lo + i] = a[lo + j];\n        a[lo + i] = t;"),
    ("shuffle-offset", SP, "let t = a[lo + i];", "let t = a[i];"),
    # ---- folds and sizes
    ("fold-start-no-extra", SP, "return f * (n / k) + extra;", "return f * (n / k);"),
    ("fold-start-min-max", SP, "if n % k < extra {\n        extra = n % k;", "if n % k > extra {\n        extra = n % k;"),
    ("fold-start-always-mod", SP, "var extra = f;\n    if n % k < extra {\n        extra = n % k;\n    }", "var extra = n % k;"),
    ("kfold-last-fold", SP, "var f = 0;\n        while f < k {\n            let lo = fold_start(n, k, f);", "var f = 0;\n        while f < k - 1 {\n            let lo = fold_start(n, k, f);"),
    ("kfold-hi-bound", SP, "let hi = fold_start(n, k, f + 1);", "let hi = fold_start(n, k, f + 1) - 1;"),
    ("kfold-k-gt-n", SP, "if k > n {\n        return rules.empty_fold();", "if k > n + 1 {\n        return rules.empty_fold();"),
    ("kfold-k-lt-2", SP, "if k < 2 {\n        return rules.out_of_range();\n    }\n    if k > n {", "if k < 1 {\n        return rules.out_of_range();\n    }\n    if k > n {"),
    ("tt-count", SP, "while p < n_test {", "while p <= n_test {"),
    ("tt-empty-train", SP, "if n < 2 || n_test > n - 1 {", "if n < 2 || n_test > n {"),
    ("tt-zero-test", SP, "if n_test < 1 {\n        return rules.out_of_range();", "if n_test < 0 {\n        return rules.out_of_range();"),
    ("tt-uses-kfold-kind", SP, "shuffle(a, 0, n, stream(seed, kind_train_test(), rep));", "shuffle(a, 0, n, stream(seed, kind_kfold(), rep));"),
    ("tt-ceil", SP, "return (n * num + den - 1) / den;", "return n * num / den;"),
    ("cap-exclusive", SP, "if n > cap {\n        return rules.too_many_rows();", "if n >= cap {\n        return rules.too_many_rows();"),
    ("cap-ceiling", SP, "if cap < 1 || cap > ceiling_rows() || rep < 0 || n < 0 {", "if cap < 1 || cap > ceiling_rows() + 1 || rep < 0 || n < 0 {"),
    ("cap-rep", SP, "if cap < 1 || cap > ceiling_rows() || rep < 0 || n < 0 {", "if cap < 1 || cap > ceiling_rows() || n < 0 {"),
    ("out-len", SP, "if out_len != n {\n        return rules.length_mismatch();", "if out_len < n {\n        return rules.length_mismatch();"),
    # ---- stratification
    ("strat-deal-offset", SP, "out[a[t]] = t % k;", "out[a[t]] = (t + 1) % k;"),
    ("strat-deal-per-class", SP, "out[a[t]] = t % k;", "if t < n0 {\n                out[a[t]] = t % k;\n            } else {\n                out[a[t]] = (t - n0) % k;\n            }"),
    ("strat-deal-blocks", SP, "out[a[t]] = t % k;", "out[a[t]] = t * k / n;"),
    ("strat-class-order", SP, "if second < first {\n        lo_label = second;", "if second > first {\n        lo_label = second;"),
    ("strat-shuffle-class1", SP, "shuffle(a, n0, n1, stream", "shuffle(a, n0, n1 - 1, stream"),
    ("strat-no-shuffle", SP, "shuffle(a, 0, n0, stream(seed, kind_stratified(), 2 * rep));", "// no shuffle of class 0"),
    ("strat-stratum", SP, "if n0 < k || n1 < k {", "if n0 < k && n1 < k {"),
    ("strat-third-label", SP, "} else if v != second {\n                third = true;", "} else if v != second && v != 0 {\n                third = true;"),
    ("strat-single", SP, "if seen2 == false {\n        return rules.single_class();", "if seen2 == false && n > 1 {\n        return rules.single_class();"),
    ("strat-members-order", SP, "a[p0] = i;\n                p0 = p0 + 1;", "a[p0] = n - 1 - i;\n                p0 = p0 + 1;"),
    # ---- counts
    ("cls-tp-class", CL, "if t == pos && p == pos {\n            tp = tp + 1;", "if t != pos && p != pos {\n            tp = tp + 1;"),
    ("cls-fp-fn-swap", CL, "} else if p == pos {\n            fp = fp + 1;\n        } else if t == pos {\n            fneg = fneg + 1;", "} else if p == pos {\n            fneg = fneg + 1;\n        } else if t == pos {\n            fp = fp + 1;"),
    ("cls-precision-den", CL, "return rules.ok(exact(k.tp, k.tp + k.fp));", "return rules.ok(exact(k.tp, k.tp + k.fneg));"),
    ("cls-recall-den", CL, "return rules.ok(exact(k.tp, k.tp + k.fneg));", "return rules.ok(exact(k.tp, k.tp + k.fp));"),
    ("cls-f1-num", CL, "return rules.ok(exact(2 * k.tp, 2 * k.tp + k.fp + k.fneg));", "return rules.ok(exact(k.tp, 2 * k.tp + k.fp + k.fneg));"),
    ("cls-f1-den", CL, "return rules.ok(exact(2 * k.tp, 2 * k.tp + k.fp + k.fneg));", "return rules.ok(exact(2 * k.tp, k.tp + k.fp + k.fneg));"),
    ("cls-f1-zero", CL, "if 2 * k.tp + k.fp + k.fneg == 0 {", "if k.tp + k.fp + k.fneg == 1 {"),
    ("cls-precision-zero", CL, "if k.tp + k.fp == 0 {", "if k.tp == 0 {"),
    ("cls-accuracy-eq", CL, "if y_true[i] == y_pred[i] {\n            hit = hit + 1;", "if y_true[i] <= y_pred[i] {\n            hit = hit + 1;"),
    ("cls-empty", CL, "if n1 == 0 {\n        return rules.undefined();", "if n1 == 0 {\n        return rules.input_empty();"),
    ("cls-length", CL, "if n1 != n2 {", "if n1 > n2 {"),
    ("cls-third-true", CL, "} else if t != other {\n                return Counts { code: rules.not_binary()", "} else if t != other && t != 0 {\n                return Counts { code: rules.not_binary()"),
    ("cls-third-pred", CL, "} else if p != other {\n                return Counts { code: rules.not_binary()", "} else if p != other && p != 0 {\n                return Counts { code: rules.not_binary()"),
    # ---- AUC
    ("auc-tie-zero", RK, "two_u = two_u + 2 * p * below + p * q;", "two_u = two_u + 2 * p * below;"),
    ("auc-tie-one", RK, "two_u = two_u + 2 * p * below + p * q;", "two_u = two_u + 2 * p * below + 2 * p * q;"),
    ("auc-tie-half-pos", RK, "two_u = two_u + 2 * p * below + p * q;", "two_u = two_u + 2 * p * below + p * p;"),
    ("auc-below-ties", RK, "let below = ni - q;", "let below = ni;"),
    ("auc-midrank-off-by-one", RK, "let below = ni - q;", "let below = ni - q + 1;"),
    ("auc-below-minus", RK, "let below = ni - q;", "let below = ni - q - 1;"),
    ("auc-den", RK, "ratio.ratio(two_u, 2 * np * nn)", "ratio.ratio(two_u, 2 * np * nn + 1)"),
    ("auc-den-halved", RK, "ratio.ratio(two_u, 2 * np * nn)", "ratio.ratio(two_u / 2, np * nn)"),
    ("auc-wrong-positive", RK, "if y_true[i] == pos {\n                        a[pp] = key_of(score[i]);", "if y_true[i] != pos {\n                        a[pp] = key_of(score[i]);"),
    ("auc-positives-count", RK, "if t == pos {\n            np = np + 1;", "if t != pos {\n            np = np + 1;"),
    ("auc-single-class", RK, "if np == 0 || nn == 0 {", "if np == 0 && nn == 0 {"),
    ("auc-single-class-neg", RK, "if np == 0 || nn == 0 {", "if np == 0 || nn == 1 {"),
    ("auc-not-binary", RK, "} else if t != other {\n            return rules.fail(rules.not_binary());", "} else if t != other && t != 0 {\n            return rules.fail(rules.not_binary());"),
    ("auc-nonfinite", RK, "if regress.all_finite(score) == false {", "if false {"),
    ("auc-signed-zero", RK, "if x == 0.0 {\n        return 0;\n    }", "if x == 0.0 && bits_of(x) == 0 {\n        return 0;\n    }"),
    ("auc-negative-keys", RK, "return b ^ 9223372036854775807;", "return b;"),
    ("auc-negative-keys-all", RK, "return b ^ 9223372036854775807;", "return b ^ 4611686018427387903;"),
    ("auc-top-digit", RK, "if d == 7 {\n        return v ^ 128;", "if d == 6 {\n        return v ^ 128;"),
    ("auc-digit-shift", RK, "let v = k >> 8 * d & 255;", "let v = k >> 7 * d & 255;"),
    ("auc-copy-back", RK, "if in_a == false {\n        var i = lo;", "if in_a == true {\n        var i = lo;"),
    ("auc-pass-parity", RK, "if moved == 1 {\n            in_a = in_a == false;", "if moved == 0 {\n            in_a = in_a == false;"),
    ("auc-skip-wrong", RK, "if cnt[digit(src[lo], d)] == hi - lo {", "if cnt[digit(src[lo], d)] >= hi - lo - 1 {"),
    ("auc-skip-none", RK, "if cnt[digit(src[lo], d)] == hi - lo {\n        return 0;\n    }", "if cnt[digit(src[lo], d)] == hi - lo + 1 {\n        return 0;\n    }"),
    ("auc-prefix", RK, "cnt[b] = run;\n        run = run + k;", "run = run + k;\n        cnt[b] = run;"),
    ("auc-sort-lo", RK, "sort_range(a, t, k, nn, n);", "sort_range(a, t, k, nn + 1, n);"),
    ("auc-sort-neg", RK, "sort_range(a, t, k, 0, nn);", "sort_range(a, t, k, 1, nn);"),
    ("auc-merge-pick", RK, "} else if a[ni] < a[pj] {\n                        v = a[ni];", "} else if a[ni] > a[pj] {\n                        v = a[ni];"),
    ("auc-merge-tie-pick", RK, "} else if a[ni] < a[pj] {", "} else if a[ni] <= a[pj] {"),
    ("auc-run-neg", RK, "while ni < nn && a[ni] == v {", "while ni < nn && a[ni] <= v {"),
    ("auc-run-pos", RK, "while pj < n && a[pj] == v {", "while pj < n && a[pj] <= v {"),
    # ---- ratio
    ("ratio-bits", RA, "while i < 52 {", "while i < 51 {"),
    ("ratio-scale", RA, "0 - (e + 52)", "0 - (e + 51)"),
    ("ratio-sticky", RA, "if r != 0 || m % 2 == 1 {", "if m % 2 == 1 {"),
    ("ratio-tie-even", RA, "if r != 0 || m % 2 == 1 {", "if r != 0 || m % 2 == 0 {"),
    ("ratio-round-bit", RA, "r = 2 * r;\n    var round_up = false;\n    if r >= b {", "r = 2 * r;\n    var round_up = false;\n    if r > b {"),
    ("ratio-first-bit", RA, "    var found = false;\n    while found == false {\n        r = 2 * r;\n        e = e + 1;\n        if r >= b {", "    var found = false;\n    while found == false {\n        r = 2 * r;\n        e = e + 1;\n        if r > b {"),
    ("ratio-one", RA, "if a == b {\n        return 1.0;", "if a == b + 1 {\n        return 1.0;"),
    ("ratio-range", RA, "if a < 0 || b < 1 || a > b || b >= 4611686018427387904 {", "if a < 0 || b < 1 || b >= 4611686018427387904 {"),
    # ---- sums, regression
    ("sum-lane-map", OR, "s1 = s1 + buf[j + 1];\n        s2 = s2 + buf[j + 2];", "s1 = s1 + buf[j + 2];\n        s2 = s2 + buf[j + 1];"),
    ("sum-lane-combine", OR, "return lane_combine(s0, s1, s2, s3);", "return lane_combine(s0, s2, s1, s3);"),
    ("sum-lane-fold", OR, "return lane_combine(s0, s1, s2, s3);", "return combine(combine(combine(s0, s1), s2), s3);"),
    ("sum-lane-tail", OR, "    if j < m {\n        s2 = s2 + buf[j];\n    }", "    if j < m {\n        s3 = s3 + buf[j];\n    }"),
    ("sum-lane-tail-1", OR, "    if j < m {\n        s0 = s0 + buf[j];\n        j = j + 1;\n    }", "    if j < m {\n        s1 = s1 + buf[j];\n        j = j + 1;\n    }"),
    ("sum-lane-init", OR, "var s3 = 0.0;", "var s3 = 1.0e-30;"),
    ("sum-block", SU, "if m > order.block() {\n            m = order.block();", "if m > 128 {\n            m = 128;"),
    ("sum-block-up", SU, "if m > order.block() {\n            m = order.block();", "if m > 260 {\n            m = 260;"),
    ("sum-sq", SU, "buf[j] = e * e;\n            j = j + 1;\n        }\n    } else if kind == 2 {", "buf[j] = e * a[at + j];\n            j = j + 1;\n        }\n    } else if kind == 2 {"),
    ("sum-abs", SU, "if e < 0.0 {\n                buf[j] = 0.0 - e;", "if e < 1.0 {\n                buf[j] = 0.0 - e;"),
    ("sum-dev-center", SU, "let e = a[at + j] - center;", "let e = a[at + j];"),
    ("sum-val", SU, "buf[j] = a[at + j];", "buf[j] = a[at + j] * 1.0000000000000002;"),
    ("sum-tree-combine", OR, "pub fn combine(a: float, b: float) -> [] float {\n    return a + b;", "pub fn combine(a: float, b: float) -> [] float {\n    return (a + b) * 1.0000000000000002;"),
    ("sum-tree-level", OR, "while sp >= 2 && lv[sp - 1] == lv[sp - 2] {\n        let a = (sp - 2) * w;\n        let b = (sp - 1) * w;\n        var j = 0;\n        while j < w {\n            st[a + j] = combine(st[a + j], st[b + j]);\n            j = j + 1;\n        }\n        lv[sp - 2] = lv[sp - 2] + 1;\n        sp = sp - 1;\n    }\n    return sp;\n}\n\n// Like", "while sp >= 2 && lv[sp - 1] >= lv[sp - 2] {\n        let a = (sp - 2) * w;\n        let b = (sp - 1) * w;\n        var j = 0;\n        while j < w {\n            st[a + j] = combine(st[a + j], st[b + j]);\n            j = j + 1;\n        }\n        lv[sp - 2] = lv[sp - 2] + 1;\n        sp = sp - 1;\n    }\n    return sp;\n}\n\n// Like"),
    ("sum-tree-finish", OR, "    while sp >= 2 {\n        let a = (sp - 2) * w;\n        let b = (sp - 1) * w;\n        var j = 0;\n        while j < w {\n            st[a + j] = combine(st[a + j], st[b + j]);\n            j = j + 1;\n        }\n        sp = sp - 1;\n    }\n    return 0;", "    var q = 1;\n    while q < sp {\n        var j = 0;\n        while j < w {\n            st[j] = combine(st[j], st[q * w + j]);\n            j = j + 1;\n        }\n        q = q + 1;\n    }\n    return 0;"),
    ("rmse-sqrt", RG, "let v = math.sqrt(sse / float_of(len(y)));", "let v = sse / float_of(len(y));"),
    ("rmse-den", RG, "let v = math.sqrt(sse / float_of(len(y)));", "let v = math.sqrt(sse / float_of(len(y) - 1));"),
    ("mae-den", RG, "let v = dsum(heap, sums.kind_abs(), y, p, 0.0) / float_of(len(y));", "let v = dsum(heap, sums.kind_abs(), y, p, 0.0) / float_of(len(y) + 1);"),
    ("mae-kind", RG, "let v = dsum(heap, sums.kind_abs(), y, p, 0.0)", "let v = dsum(heap, sums.kind_sq(), y, p, 0.0)"),
    ("r2-formula", RG, "let v = 1.0 - sse / sst;", "let v = sse / sst;"),
    ("r2-mean-den", RG, "let mean = dsum(heap, sums.kind_val(), y, p, 0.0) / float_of(n);", "let mean = dsum(heap, sums.kind_val(), y, p, 0.0) / float_of(n - 1);"),
    ("r2-sst-den", RG, "let sst = dsum(heap, sums.kind_dev(), y, p, mean);", "let sst = dsum(heap, sums.kind_sq(), y, p, mean);"),
    ("r2-constant-check", RG, "if y[i] != y[0] {\n            constant = false;", "if y[i] > y[0] {\n            constant = false;"),
    ("r2-no-constant-check", RG, "    if constant {\n        return rules.fail(rules.undefined());\n    }\n", ""),
    ("r2-sst-zero", RG, "|| sst == 0.0 {", "{"),
    ("r2-nonfinite-result", RG, "    let v = 1.0 - sse / sst;\n    if finite(v) == false {", "    let v = 1.0 - sse / sst;\n    if false {"),
    ("fin-check", RG, "if v[i] - v[i] != 0.0 {", "if v[i] != v[i] {"),
    ("reg-nonfinite-y", RG, "if all_finite(y) == false || all_finite(p) == false {", "if all_finite(y) == false {"),
    ("reg-nonfinite-p", RG, "if all_finite(y) == false || all_finite(p) == false {", "if all_finite(p) == false {"),
    # ---- log-loss
    ("ll-eps", LL, "return 2.220446049250313e-16;", "return 1.1102230246251565e-16;"),
    ("ll-neg-class", LL, "if is_pos == false {\n        x = 1.0 - p;", "if is_pos {\n        x = 1.0 - p;"),
    ("ll-no-complement", LL, "x = 1.0 - p;\n    }\n    if x < eps()", "x = p;\n    }\n    if x < eps()"),
    ("ll-clip-low", LL, "if x < eps() {\n        x = eps();\n    }", "if x < 0.0 {\n        x = eps();\n    }"),
    ("ll-clip-high", LL, "if x > 1.0 - eps() {\n        x = 1.0 - eps();\n    }", "if x > 1.0 {\n        x = 1.0 - eps();\n    }"),
    ("ll-clip-high-off", LL, "if x > 1.0 - eps() {\n        x = 1.0 - eps();\n    }", "if x > 1.0 - eps() {\n        x = 1.0 - 2.0 * eps();\n    }"),
    ("ll-sign", LL, "return 0.0 - math.log(x);", "return math.log(x);"),
    ("ll-mean", LL, "return rules.ok(total / float_of(n));", "return rules.ok(total / float_of(n + 1));"),
    ("ll-prob-high", LL, "if prob[i] < 0.0 || prob[i] > 1.0 {", "if prob[i] < 0.0 || prob[i] >= 1.0 {"),
    ("ll-prob-low", LL, "if prob[i] < 0.0 || prob[i] > 1.0 {", "if prob[i] <= 0.0 || prob[i] > 1.0 {"),
    ("ll-prob-high-wide", LL, "if prob[i] < 0.0 || prob[i] > 1.0 {", "if prob[i] < 0.0 || prob[i] > 1.5 {"),
    ("ll-finite", LL, "if regress.all_finite(prob) == false {", "if false {"),
    ("ll-order-probs", LL, "let t = labels[i];\n        if t != pos {\n            if have == false {", "let t = labels[i];\n        if t != pos && t != 0 {\n            if have == false {"),
    ("ll-label-pos", LL, "buf[j] = term(labels[at + j] == pos, prob[at + j]);", "buf[j] = term(labels[at + j] != pos, prob[at + j]);"),
    ("ll-fill-offset", LL, "buf[j] = term(labels[at + j] == pos, prob[at + j]);", "buf[j] = term(labels[at + j] == pos, prob[j]);"),
    # ---- the registry
    ("rules-exit-param", RU, "if code == 207 {\n        return 2;", "if code == 208 {\n        return 2;"),
    ("rules-exit-range", RU, "if code >= 201 && code <= 211 {\n        return 8;", "if code >= 201 && code <= 210 {\n        return 8;"),
    ("rules-dup-tag", RU, "return \"ml.split.stratum-too-small\";", "return \"ml.split.empty-fold\";"),
    ("rules-tag-rename", RU, "return \"ml.metric.undefined\";", "return \"ml.metric.undefine\";"),
    ("rules-table-exit", RU, "ml.metric.length-mismatch|8|never", "ml.metric.length-mismatch|2|never"),
    ("rules-table-drop", RU, "ml.metric.bad-probability|8|never|a probability outside 0 to 1. Hint: log-loss needs the probability of the positive class. Repair: none.;", ""),
    ("rules-code-swap", RU, "pub fn undefined() -> [] int {\n    return 206;", "pub fn undefined() -> [] int {\n    return 205;"),
]

ALLOWED = {
    "auc-run-neg": "equivalent: v is the smaller of the two heads, so a[ni] >= v always; `a[ni] <= v` and `a[ni] == v` select the same entries",
    "auc-run-pos": "equivalent: same argument for the positive head",
    "auc-merge-tie-pick": "equivalent: when the two heads are equal either one gives the same v, and both runs are then consumed",
    "auc-skip-none": "equivalent in the result: a counting pass over a byte that is the same for every key is a stable copy; the skip only saves time (11 ms on 16-valued scores, docs/metrics.md 5.4)",
    "r2-sst-zero": "equivalent: a zero SST makes SSE / SST an infinity or a NaN, and the finite test on the result refuses it with the same tag",
    "ratio-first-bit": "equivalent: r == b at the first bit happens only when the quotient is an exact power of two, and the 52 bits that follow then round up to the same value",
    "ratio-one": "equivalent: a == b is a shortcut; the general path also returns 1.0 (checked for a = b above)",
    "sum-tree-level": "equivalent: stack levels strictly decrease from the bottom, so `lv[sp-1] > lv[sp-2]` never happens (docs/linalg.md 5.1 lists the same survivor, order-level-eq)",
}

# --shard I/N runs the mutants whose position is I mod N (so that N processes can share the list); a name-substring argument keeps
# only the mutants whose name contains it
args = [a for a in sys.argv[1:] if not a.startswith("--")]
for i, a in enumerate(sys.argv[1:]):
    if a == "--shard":
        si, sn = [int(x) for x in sys.argv[i + 2].split("/")]
        M = [m for j, m in enumerate(M) if j % sn == si]
        args = [x for x in args if x != sys.argv[i + 2]]
if args:
    M = [m for m in M if any(a in m[0] for a in args)]
sys.exit(mutlib.run(M, ["sh", "scripts/gate_metrics.sh"], "mutants_metrics", allowed=ALLOWED, timeout=900))

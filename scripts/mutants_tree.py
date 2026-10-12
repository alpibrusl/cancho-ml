#!/usr/bin/env python3
"""Mutants of lib/models/tree (docs/models-tree.md section 8, G-mutants). A mutant is a list of source edits (file, old, new), each
`old` asserted to occur exactly once, applied to a COPY of lib/ and tools/ and tests/. It must be killed by a gate: the build of the
driver must succeed (a mutant that does not build is reported, not counted), then, in this order, the gates stop at the first that
fails, times out or traps: unit (tests/t_tree.cho), exact, fuzz, ident at 1,000 rows, ident at 100,000 rows, threads.
The unmutated control must pass every gate first. A survivor must be explained in ALLOWED.

    python3 scripts/mutants_tree.py --check            every pattern applies once (no compiler)
    python3 scripts/mutants_tree.py [-j N] [substring ...]   run (default 4 parallel workers)
Exit 1 if a mutant survives unexplained, fails to apply, or does not build."""
import concurrent.futures as cf, os, shutil, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_tree as G

G_ = "lib/models/tree/grow.cho"
B_ = "lib/models/tree/bins.cho"
P_ = "lib/models/tree/predict.cho"
F_ = "lib/models/tree/forest.cho"
M_ = "lib/models/tree/model.cho"
X_ = "lib/models/tree/exact.cho"
Q_ = "lib/models/tree/params.cho"
D_ = "lib/models/tree/deny.cho"

M = [
 # --- the ties, bins, bootstrap, direction, depth, subset size, aggregation order, missing side (the list of the gate)
 ("tie-class: equal replaces", [(G_, "exact.gt(nn, dd, wk[ob + 1], wk[ob + 2])", "exact.gt(wk[ob + 1], wk[ob + 2], nn, dd) == false")]),
 ("tie-reg: equal replaces", [(G_, "sc > wf[ofb]", "sc >= wf[ofb]")]),
 ("tie-variant: variant 1 before 0 (class)", [(G_, "        cand_c(wk, ob, nl, pl, 0, b, fi, m, n1, ml);\n        if mc > 0 {\n            cand_c(wk, ob, nl + mc, pl + mp, 1, b, fi, m, n1, ml);\n        }",
                                              "        if mc > 0 {\n            cand_c(wk, ob, nl + mc, pl + mp, 1, b, fi, m, n1, ml);\n        }\n        cand_c(wk, ob, nl, pl, 0, b, fi, m, n1, ml);")]),
 ("tie-variant: variant 1 before 0 (reg)", [(G_, "        cand_r(wk, wf, ob, ofb, nl, sl, 0, b, fi, m, sm, ml);\n        if mc > 0 {\n            cand_r(wk, wf, ob, ofb, nl + mc, sl + ms, 1, b, fi, m, sm, ml);\n        }",
                                             "        if mc > 0 {\n            cand_r(wk, wf, ob, ofb, nl + mc, sl + ms, 1, b, fi, m, sm, ml);\n        }\n        cand_r(wk, wf, ob, ofb, nl, sl, 0, b, fi, m, sm, ml);")]),
 ("bin boundary: e < x", [(B_, "if e[c & 63] <= x {", "if e[c & 63] < x {"), (B_, "if e[c & 255] <= x {", "if e[c & 255] < x {")]),
 ("bin search: a step missing (64)", [(B_, "if e[c + 7 & 63] <= x {\n        c = c + 8;\n    }", "")]),
 ("bin search: a step missing (256)", [(B_, "if e[c + 15 & 255] <= x {\n        c = c + 16;\n    }", "")]),
 ("bin padding: not +inf", [(B_, "ep[f * ps + k] = inf;", "ep[f * ps + k] = 0.0;")]),
 ("bootstrap: draw 0 of tree 0 plus one", [(G_, "            r = rng.uniform(rng.output(s), n);", "            r = rng.uniform(rng.output(s), n);\n            if tree == 0 && k == 0 {\n                r = (r + 1) % n;\n            }")]),
 ("bootstrap: stream of tree + 1", [(G_, "var s = rng.derive(rng.derive(p.seed, 1), tree);", "var s = rng.derive(rng.derive(p.seed, 1), tree + 1);")]),
 ("child swap in training", [(G_, "                if v <= bb {\n                    lf = 1;", "                if v > bb {\n                    lf = 1;")]),
 ("child swap in prediction", [(P_, "left = row[f] < nv[g];", "left = row[f] >= nv[g];"), (P_, "                    if row[f] < nv[g] {", "                    if row[f] >= nv[g] {")]),
 ("child swap in both", [(G_, "                if v <= bb {\n                    lf = 1;", "                if v > bb {\n                    lf = 1;"), (P_, "left = row[f] < nv[g];", "left = row[f] >= nv[g];"), (P_, "                    if row[f] < nv[g] {", "                    if row[f] >= nv[g] {")]),
 ("depth - 1", [(G_, "var can = dep < p.max_depth && m", "var can = dep + 1 < p.max_depth && m")]),
 ("subset size - 1 (the caller's mtry)", [(F_, "let mtry = params.mtry_of(p, d);", "let mtry = params.mtry_of(p, d) - 1;")]),
 ("subset: the first mtry - 1 marked", [(G_, "            j = 0;\n            while j < mtry {\n                wk[o_mark + wk[o_perm + j]] = 1;", "            j = 0;\n            while j + 1 < mtry {\n                wk[o_mark + wk[o_perm + j]] = 1;")]),
 ("subset stream: the node's preorder rank", [(G_, "rng.derive(rng.derive(rng.derive(p.seed, 2), tree), id)", "rng.derive(rng.derive(rng.derive(p.seed, 2), tree), ni)")]),
 ("subset draw: uniform(d) every time", [(G_, "let q = j + rng.uniform(rng.output(fs), d - j);", "let q = rng.uniform(rng.output(fs), d);")]),
 ("aggregation: trees summed last to first", [(P_, "        var t = 0;\n        while t < trees {", "        var t = trees - 1;\n        while t >= 0 {"), (P_, "            acc = acc + nv[g];\n            t = t + 1;", "            acc = acc + nv[g];\n            t = t - 1;"), (P_, "            var t = 0;\n            while t < trees {", "            var t = trees - 1;\n            while t >= 0 {"), (P_, "                acc = acc + nv[g];\n                t = t + 1;", "                acc = acc + nv[g];\n                t = t - 1;")]),
 ("aggregation: multiply by 1 / T instead of dividing", [(P_, "                t = t + 1;\n            }\n            out[i] = acc / ftrees;", "                t = t + 1;\n            }\n            out[i] = acc * (1.0 / ftrees);"), (P_, "            t = t + 1;\n        }\n        out[i] = acc / ftrees;", "            t = t + 1;\n        }\n        out[i] = acc * (1.0 / ftrees);")]),
 ("aggregation: sum starts from the first tree", [(P_, "        var acc = 0.0;\n        var t = 0;", "        var acc = 0.0 - 0.0;\n        var t = 0;"), (P_, "            var acc = 0.0;\n            var t = 0;", "            var acc = 0.0 - 0.0;\n            var t = 0;")]),
 ("missing side: always left in training", [(G_, "lf = bdl;", "lf = 1;")]),
 ("missing side: always right in training", [(G_, "lf = bdl;", "lf = 0;")]),
 ("missing side: flipped in prediction", [(P_, "left = dl[g] == 1;", "left = dl[g] == 0;")]),
 ("missing bin: bin 0", [(B_, "                var c = nbins;\n                if len(bm) == 0 || !is_missing(bm, st, f, i) {\n                    c = bin_of64(", "                var c = 0;\n                if len(bm) == 0 || !is_missing(bm, st, f, i) {\n                    c = bin_of64(")]),
 ("missing: variant 1 never tried (class)", [(G_, "        cand_c(wk, ob, nl, pl, 0, b, fi, m, n1, ml);\n        if mc > 0 {", "        cand_c(wk, ob, nl, pl, 0, b, fi, m, n1, ml);\n        if mc > 1000000000 {")]),
 ("missing: variant 1 never tried (reg)", [(G_, "        cand_r(wk, wf, ob, ofb, nl, sl, 0, b, fi, m, sm, ml);\n        if mc > 0 {", "        cand_r(wk, wf, ob, ofb, nl, sl, 0, b, fi, m, sm, ml);\n        if mc > 1000000000 {")]),
 ("missing: variant 1 counts but not the sums (reg)", [(G_, "nl + mc, sl + ms, 1,", "nl + mc, sl, 1,")]),
 ("missing: variant 1 class counts (class)", [(G_, "nl + mc, pl + mp, 1,", "nl + mc, pl, 1,")]),
 ("missing: edges include missing cells", [(B_, "                if !is_missing(bm, st, f, i) {\n                    ka[nv]", "                if true {\n                    ka[nv]")]),
 ("missing: default side not stored", [(G_, "            oi[2 * cap + ni] = bdl;", "            oi[2 * cap + ni] = 0;")]),
 # --- limits and leaves
 ("min_samples_leaf: nl > ml (class)", [(G_, "    let nr = m - nl;\n    if nl >= ml && nr >= ml {\n        let pr", "    let nr = m - nl;\n    if nl > ml && nr >= ml {\n        let pr")]),
 ("min_samples_leaf: nr > ml (reg)", [(G_, "    let nr = m - nl;\n    if nl >= ml && nr >= ml {\n        let sr", "    let nr = m - nl;\n    if nl >= ml && nr > ml {\n        let sr")]),
 ("leaf rule: m > 2 ml", [(G_, "m >= 2 * p.min_leaf;", "m > 2 * p.min_leaf;")]),
 ("leaf rule: pure class node is split", [(G_, "can = can && n0 > 0 && n1 > 0;", "can = can;")]),
 ("leaf rule: regression purity test removed", [(G_, "can = can && ymin != ymax;", "can = can;")]),
 ("edge index: k nv / (B + 1)", [(B_, "ka[k * nv / nbins]", "ka[k * nv / (nbins + 1)]")]),
 ("edge: the last edge one value lower", [(B_, "ka[k * nv / nbins]", "ka[k * nv / nbins - k / (nbins - 1)]")]),
 ("threshold: edge of bin b + 1 instead of b", [(G_, "ov[ni] = ed[bf * ne + bb];", "ov[ni] = ed[bf * ne + bb + 1];")]),
 ("leaf value: class 0 fraction", [(G_, "ov[ni] = float_of(n1) / float_of(n0 + n1);", "ov[ni] = float_of(n0) / float_of(n0 + n1);")]),
 ("leaf value: mean over m + 1", [(G_, "ov[ni] = sm / float_of(m);", "ov[ni] = sm / float_of(m + 1);")]),
 ("exact: operand of the second product", [(X_, "let (hq, lq) = mul_hi_lo(n2, d1);", "let (hq, lq) = mul_hi_lo(n2, d2);")]),
 ("exact: limb shift 29", [(X_, "let a1 = a >> 30;", "let a1 = a >> 29;")]),
 ("exact: low parts only", [(X_, "    if hp != hq {\n        return hp > hq;\n    }\n    return lp > lq;", "    return lp > lq;")]),
 ("exact: carry dropped", [(X_, "let t = a1 * b0 + a0 * b1 + (lo >> 30);", "let t = a1 * b0 + a0 * b1;")]),
 # --- sums of the regression criterion
 ("sum order: node total summed last to first", [(G_, "            k = lo;\n            while k < hi {\n                let v = yv[idx[k]];\n                sm = sm + v;", "            k = hi - 1;\n            while k >= lo {\n                let v = yv[idx[k]];\n                sm = sm + v;"),
                                                 (G_, "                if v > ymax {\n                    ymax = v;\n                }\n                k = k + 1;", "                if v > ymax {\n                    ymax = v;\n                }\n                k = k - 1;")]),
 ("sum order: bin sums last to first (hist1r)", [(G_, "    var k = lo;\n    while k < hi {\n        let r = idx[k];\n        let i0 = base + int_of(c0[r]);\n        hc[i0] = hc[i0] + 1;\n        hf[i0] = hf[i0] + yv[r];\n        k = k + 1;\n    }",
                                                 "    var k = hi - 1;\n    while k >= lo {\n        let r = idx[k];\n        let i0 = base + int_of(c0[r]);\n        hc[i0] = hc[i0] + 1;\n        hf[i0] = hf[i0] + yv[r];\n        k = k - 1;\n    }")]),
 ("sum order: cumulative from the last bin", [(G_, "        sl = sl + wf[fb + b];", "        sl = sl + wf[fb + (nbins - 2 - b)];")]),
 ("sum: right sum off by 1e-12", [(G_, "let sr = sm - sl;", "let sr = sm - sl - 0.0 + 1.0e-12;")]),
 ("score: swapped operands (an equivalent: + commutes)", [(G_, "let sc = sl * sl / float_of(nl) + sr * sr / float_of(nr);", "let sc = sr * sr / float_of(nr) + sl * sl / float_of(nl);")]),
 ("partition: unstable swap partition", [(G_, "            var a = lo;\n            var z = 0;\n            k = lo;\n            while k < hi {\n                let r = idx[k];\n                let v = int_of(col[r]);\n                var lf = 0;\n                if v <= bb {\n                    lf = 1;\n                }\n                if v == nbins {\n                    lf = bdl;\n                }\n                // branch-free: write the row to both lists, advance one of them (a <= k, so idx[a] is never a row still to be read)\n                idx[a] = r;\n                tmp[z] = r;\n                a = a + lf;\n                z = z + 1 - lf;\n                k = k + 1;\n            }\n            k = 0;\n            while k < z {\n                idx[a + k] = tmp[k];\n                k = k + 1;\n            }",
                                    "            var a = lo;\n            var zz = hi - 1;\n            while a <= zz {\n                let r = idx[a];\n                let v = int_of(col[r]);\n                var lf = 0;\n                if v <= bb {\n                    lf = 1;\n                }\n                if v == nbins {\n                    lf = bdl;\n                }\n                if lf == 1 {\n                    a = a + 1;\n                } else {\n                    idx[a] = idx[zz];\n                    idx[zz] = r;\n                    zz = zz - 1;\n                }\n            }")]),
 # --- nodes
 ("node budget: > instead of >=", [(G_, "        if cnt > cap {", "        if cnt > cap + 1 {")]),
 ("node cap: 2^depth - 1", [(Q_, "c = min2(c, (1 << p.max_depth + 1) - 1);", "c = min2(c, (1 << p.max_depth) - 1);")]),
 ("depth ceiling >= 31 accepted", [(Q_, "if p.max_depth > depth_max() {", "if p.max_depth > depth_max() + 1 {")]),
 ("node total check off by one", [(Q_, "if p.trees > nodes_total_max() / node_cap(p, n) {", "if p.trees > nodes_total_max() / node_cap(p, n) + 1 {")]),
 ("row bound of the exact comparison", [(Q_, "if n > rows_max() {", "if n > rows_max() + 1 {")]),
 ("min_leaf range", [(Q_, "if p.min_leaf < 1 || p.min_leaf > n {", "if p.min_leaf < 1 {")]),
 ("single class not refused", [(F_, "if ones == 0 || ones == n {", "if ones == n + 1 {")]),
 ("label check: 2.0 accepted as 1", [(F_, "            } else if v != 0.0 {\n                return deny.not_binary();", "            } else if v != 0.0 && v != 2.0 {\n                return deny.not_binary();")]),
 ("target bound 1e100", [(F_, "v <= big && v >= 0.0 - big", "v <= big")]),
 ("non-finite X not seen", [(F_, "if bins.count_bad_cells(xs, bm, n, d) != 0 {", "if bins.count_bad_cells(xs, bm, n, d) != 0 && false {")]),
 ("missing cell read as non-finite", [(B_, "} else if len(bm) == 0 || !is_missing(bm, st, f, i) {", "} else if true {")]),
 # --- threads
 ("threads: tree stream from the thread-local index", [(F_, "let r = grow.grow(p, n, d, mtry, cap, t, bins,", "let r = grow.grow(p, n, d, mtry, cap, j, bins,")]),
 ("threads: partition ceil(T / t)", [(F_, "    return k * trees / te;", "    return k * ((trees + te - 1) / te);")]),
 ("threads: the last group one tree short", [(F_, "    return k * trees / te;", "    if k == te {\n        return trees - 1;\n    }\n    return k * trees / te;")]),
 ("threads: merge copies group k at slot k", [(F_, "copy_ints(so[t0 * 7 * cap..t1 * 7 * cap]", "copy_ints(so[k * 7 * cap..(k + t1 - t0) * 7 * cap]")]),
 ("threads: refusal of the highest tree", [(F_, "            if cs[t] < 0 && refusal == 0 {\n                refusal = 0 - cs[t];\n            }", "            if cs[t] < 0 {\n                refusal = 0 - cs[t];\n            }")]),
 ("threads: predict pieces leave a row", [(P_, "    var r1 = b1 * 8;", "    var r1 = b1 * 8 - 1;")]),
 ("threads: another static partition of the prediction rows", [(P_, "    return k * nblk / te;", "    return (k * nblk + 1) / te;")]),
 ("threads: copy of the bitmap at a wrong column offset", [(P_, "gm[c * st + b0..c * st + b1]", "gm[c * st + b0 + (1 - 1 / (1 + b0))..c * st + b1 + (1 - 1 / (1 + b0))]")]),
 # --- the model reader (G-fuzz)
 ("reader: left child must be i + 1 not checked", [(M_, "if lc != i + 1 || cur >= max_depth {", "if cur >= max_depth {")]),
 ("reader: depth bound", [(M_, "if lc != i + 1 || cur >= max_depth {", "if lc != i + 1 {")]),
 ("reader: feature range", [(M_, "if feat < 0 || feat >= d || bin < 0", "if feat < 0 || bin < 0")]),
 ("reader: bin range", [(M_, "bin < 0 || bin > nbins - 2 ||", "bin < 0 ||")]),
 ("reader: default side", [(M_, " || dl != 0 && dl != 1 {", " {")]),
 ("reader: leaf form", [(M_, "if bin != 0 - 1 || dl != 0 || lc != 0 - 1 || rc != 0 - 1 {", "if bin != 0 - 1 {")]),
 ("reader: counts non-negative", [(M_, "if na < 0 || nb < 0 {", "if na < -5 {")]),
 ("reader: finite values (load)", [(M_, "            if bad_vals != 0 {", "            if bad_vals != 0 && false {")]),
 ("reader: class leaf value equals the counts", [(M_, " || bits_of(v) != bits_of(float_of(nb) / float_of(na + nb)) {", " {")]),
 ("reader: tree offsets increase", [(M_, "if tof[t + 1] <= tof[t] {", "if tof[t + 1] < tof[t] {")]),
 ("reader: offsets end at total", [(M_, "if tof[0] != 0 || tof[trees] != total {", "if tof[0] != 0 {")]),
 ("reader: length equation (longer)", [(M_, "    if n > size {\n        return bad(heap, deny.model_malformed());\n    }\n", "")]),
 ("reader: length equation (shorter)", [(M_, "    if n < size {\n        return bad(heap, deny.model_truncated());\n    }\n", "")]),
 ("reader: record headers", [(M_, "        if !rec_ok(data, record_at(r, trees, total), dt, total) {\n            return bad(heap, deny.model_malformed());\n        }", "")]),
 ("reader: header ranges (d)", [(M_, "d < 1 || d > 4096 ||", "d < 1 ||")]),
 ("reader: node ceiling", [(M_, "    if total > max_total || total > params.nodes_total_max() {", "    if total > params.nodes_total_max() * 1000 {")]),
 ("reader: every tree is checked", [(M_, "while t < trees && code == 0 {\n                        code = check_tree(", "while t + 1 < trees && code == 0 {\n                        code = check_tree(")]),
 ("reader: pending right children empty", [(M_, "    if stack != 0 {", "    if false {")]),
 ("reader: a right child must be popped by the node of that index", [(M_, "                if st[stack] != i {\n                    return deny.model_malformed();\n                }", "")]),
 ("reader: kind", [(M_, "if get32(data, 0) != kind() || !rec_ok(data, 4, 1, 8) {", "if !rec_ok(data, 4, 1, 8) {")]),
]
# a survivor is allowed only with its reason
ALLOWED = {
 "score: swapped operands (an equivalent: + commutes)": "equivalent: IEEE addition is commutative, a + b and b + a have the same bits",
 "aggregation: sum starts from the first tree": "equivalent: acc starts at -0.0 instead of +0.0; the first addition gives the same bits unless a leaf value is -0.0, and no leaf value is -0.0 (sums start from +0.0, 0.0 / m is +0.0, class values are ratios of counts)",
 "threads: another static partition of the prediction rows": "equivalent by design: rows are independent, so any partition of them gives the same bits; this mutant IS another partition (it is the property G-thread relies on)",
 "threads: refusal of the highest tree": "not observable today: the only refusal a tree returns is ml.limit.nodes, so which tree's refusal is reported cannot change the tag; the lowest-tree rule is kept for the day a second code exists",
}
GATES = ["unit", "exact", "fuzz", "ident1000", "ident100000", "threads"]


def problems():
    out, names, cache = [], set(), {}
    for name, edits in M:
        if name in names:
            out.append("mutant %r named twice" % name)
        names.add(name)
        for f, old, new in edits:
            p = os.path.join(G.ROOT, f)
            if not os.path.exists(p):
                out.append("%s: no file %s" % (name, f)); continue
            t = cache.setdefault(f, open(p).read())
            c = t.count(old)
            if c != 1:
                out.append("%s: pattern occurs %d times in %s: %r" % (name, c, f, old[:60]))
            if old == new:
                out.append("%s: replacement equals pattern" % name)
    return out


def copy_tree(dst):
    for d in ("lib", "tools", "tests"):
        shutil.copytree(os.path.join(G.ROOT, d), os.path.join(dst, d))
    os.makedirs(os.path.join(dst, "build"), exist_ok=True)


def gate_all(src, only=None):
    """name of the first gate that fails, 'build' if it does not build, or None if all pass"""
    exe = os.path.join(src, "build", "tree-check")
    try:
        G.build(src, exe)
    except RuntimeError as e:
        return "build", str(e)[-300:]
    tests = [
        ("unit", lambda: G.unit(src, verbose=False, timeout=120)),
        ("exact", lambda: G.exact(exe, verbose=False)),
        ("fuzz", lambda: G.fuzz(exe, verbose=False, timeout=60)),
        ("ident1000", lambda: G.ident(exe, list(G.CONFIGS), ["1000"], verbose=False)),
        ("ident100000", lambda: G.ident(exe, list(G.CONFIGS), ["100000"], verbose=False)),
        ("threads", lambda: G.thread_gate(exe, verbose=False, timeout=120)),
    ]
    for name, fn in tests:
        try:
            if fn():
                return name, ""
        except subprocess.TimeoutExpired:
            return name + " (timeout)", ""
        except RuntimeError as e:
            return "unit build", str(e)[-200:]
    return None, ""


def one(item):
    name, edits = item
    with tempfile.TemporaryDirectory() as tmp:
        copy_tree(tmp)
        for f, old, new in edits:
            p = os.path.join(tmp, f)
            t = open(p).read()
            open(p, "w").write(t.replace(old, new, 1))
        t0 = time.time()
        g, msg = gate_all(tmp)
    return name, g, msg, time.time() - t0


def main():
    bad = problems()
    if bad:
        print("\n".join(bad)); return 1
    if "--check" in sys.argv:
        print("mutants_tree: %d mutants, every pattern applies once" % len(M)); return 0
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    jobs = 4
    if "-j" in sys.argv:
        jobs = int(sys.argv[sys.argv.index("-j") + 1]); args = [a for a in args if a != str(jobs)]
    sel = [m for m in M if not args or any(a in m[0] for a in args)]
    with tempfile.TemporaryDirectory() as tmp:
        copy_tree(tmp)
        g, msg = gate_all(tmp)
    if g is not None:
        print("the unmutated control FAILS gate %s %s; refusing to run mutants" % (g, msg)); return 1
    print("mutants_tree: control (unmutated) passes all gates", flush=True)
    killed, survivors, explained, nobuild = {}, [], 0, []
    with cf.ThreadPoolExecutor(jobs) as ex:
        for name, g, msg, dt in ex.map(one, sel):
            if g == "build":
                nobuild.append(name); print("%-58s DOES NOT BUILD %s" % (name, msg), flush=True)
            elif g is None and name in ALLOWED:
                explained += 1; print("%-58s SURVIVED (explained: %s)" % (name, ALLOWED[name]), flush=True)
            elif g is None:
                survivors.append(name); print("%-58s SURVIVED" % name, flush=True)
            else:
                killed[g] = killed.get(g, 0) + 1; print("%-58s killed by %-14s (%.0f s)" % (name, g, dt), flush=True)
    print("mutants_tree: %d killed %s, %d survived and explained, %d unexplained, %d did not build" % (sum(killed.values()), dict(sorted(killed.items())), explained, len(survivors), len(nobuild)))
    if survivors:
        print("SURVIVORS: " + "; ".join(survivors))
    return 1 if survivors or nobuild else 0


sys.exit(main())

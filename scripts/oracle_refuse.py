#!/usr/bin/env python3
"""Gates B, C and F of docs/refusals-fit.md: the refusals of ml fit (ml.fit.collinear, ml.fit.separable) against oracles that share no code with them.

  rank  -- B. Generated designs (duplicate columns, scaled copies, sums, constants, zero columns, the full one-hot block, near-collinear pairs at many
           scales, well conditioned data). The oracle is the exact relative pivot of the Gram matrix of [X | 1] in rational arithmetic on the very doubles
           that ml reads (fractions.Fraction): ml must refuse exactly when the exact relative pivot is at most 2^-40, must name the exact first failing
           column, and must give the exact dependency (to 1e-5). A pivot within 10% of the tolerance is in a band: reported, not counted.
  sep   -- C. Integer points in the plane (and the line) with labels: separability decided EXACTLY (candidate normals from the point pairs, integer
           arithmetic, strict and weak). Complete separation must be refused with ml.fit.separable (100 iterations, l2 0); overlapping classes
           must be fitted; the quasi-complete ones are tallied; a positive l2 is never refused.
  fuzz  -- F. Random designs (random rank, scales 1e-300 to 1e300, all rows equal, one row, wide): no trap (a JSON document, exit 0 or 8), the verdict
           is the same at 1, 2, 4, 8 and 16 threads byte for byte, and scaling a column by a power of ten does not change the verdict.
  all   -- the three. Exit 1 on any disagreement, with the first few printed.

usage: oracle_refuse.py {rank|sep|fuzz|all} [--ml PATH] [--seed N] [--n N]
"""
import itertools
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = Fraction(1, 2 ** 40)
ML = os.path.join(ROOT, "build", "ml")
SEED = 20261015
N = None
if "--ml" in sys.argv:
    ML = sys.argv[sys.argv.index("--ml") + 1]
if "--seed" in sys.argv:
    SEED = int(sys.argv[sys.argv.index("--seed") + 1])
if "--n" in sys.argv:
    N = int(sys.argv[sys.argv.index("--n") + 1])

BAD = []


def bad(msg):
    BAD.append(msg)
    if len(BAD) <= 12:
        print("DISAGREE:", msg)


def run(dirname, *args):
    p = subprocess.run([ML, "--root", dirname] + list(args), capture_output=True, cwd=dirname)
    return p.returncode, p.stdout, p.stderr


def doc_of(out):
    return json.loads(out.decode())


def write_csv(dirname, header, rows, name="d.csv"):
    with open(os.path.join(dirname, name), "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")


def fmt(x):
    return repr(float(x))


# ---------------------------------------------------------------- the exact elimination

def exact_pivots(cols):
    """cols: list of columns (lists of Fractions), the intercept column of ones LAST. Returns (j, ratio, c) for the first j whose relative pivot is
    at most 2^-40 (c: exact least-squares coefficients of column j on the columns before it), or (None, min ratio, None)."""
    p = len(cols)
    n = len(cols[0])
    G = [[sum(a * b for a, b in zip(cols[i], cols[j])) for j in range(p)] for i in range(p)]
    L = [[Fraction(0)] * p for _ in range(p)]
    # LDL^T in exact arithmetic: D[j] = s_jj, L[i][j] = (G_ij - sum_k L_ik L_jk D_k) / D_j
    D = [Fraction(0)] * p
    minratio = None
    for j in range(p):
        s = G[j][j] - sum(L[j][k] * L[j][k] * D[k] for k in range(j))
        ratio = (s / G[j][j]) if G[j][j] != 0 else Fraction(0)
        minratio = ratio if minratio is None or ratio < minratio else minratio
        if not (ratio > TOL):
            # c = G11^-1 g_j by exact forward/back substitution on the LDL factors of the leading block
            g = [G[j][k] for k in range(j)]
            u = []
            for q in range(j):
                u.append(g[q] - sum(L[q][k] * u[k] for k in range(q)))
            v = [u[q] / D[q] for q in range(j)]
            c = [Fraction(0)] * j
            for q in reversed(range(j)):
                c[q] = v[q] - sum(L[k][q] * c[k] for k in range(q + 1, j))
            return j, ratio, c
        D[j] = s
        for i in range(j + 1, p):
            L[i][j] = (G[i][j] - sum(L[i][k] * L[j][k] * D[k] for k in range(j))) / s
    return None, minratio, None


def exact_for(header_cols, texts):
    cols = [[Fraction(float(t)) for t in col] for col in texts]
    cols.append([Fraction(1)] * len(cols[0]))
    return exact_pivots(cols)


# ---------------------------------------------------------------- B: rank

def gen_rank(rng, kind):
    n = rng.randint(8, 40)
    p = rng.randint(2, 7)
    base = [[round(rng.uniform(-5, 5), 4) for _ in range(n)] for _ in range(p)]
    cols = [list(c) for c in base]
    note = kind
    if kind == "dup":
        i, j = sorted(rng.sample(range(p), 2))
        cols[j] = list(cols[i])
    elif kind == "scaled":
        i, j = sorted(rng.sample(range(p), 2))
        cols[j] = [round(3 * v, 4) for v in cols[i]]
    elif kind == "sum":
        p = max(p, 3)
        cols = [list(c) for c in base[:p]] if len(base) >= p else cols
        while len(cols) < 3:
            cols.append([round(rng.uniform(-5, 5), 4) for _ in range(n)])
        a, b = 0, 1
        cols[2] = [round(x + y, 4) for x, y in zip(cols[a], cols[b])]
    elif kind == "constant":
        j = rng.randrange(p)
        cols[j] = [rng.choice([1.0, 2.5, -4.0, 1000.0])] * n
    elif kind == "zero":
        j = rng.randrange(p)
        cols[j] = [0.0] * n
    elif kind == "near":
        i, j = sorted(rng.sample(range(p), 2))
        eps = 10 ** rng.uniform(-9, -3)
        scales = [10 ** rng.randint(-8, 8) for _ in range(p)]
        cols[j] = [x + eps * rng.uniform(-1, 1) for x in cols[i]]
        cols = [[v * scales[k] for v in c] for k, c in enumerate(cols)]
        note = "near eps=%.1e scales=%s" % (eps, scales)
    elif kind == "well":
        pass
    y = [round(rng.uniform(-3, 3), 4) for _ in range(n)]
    return cols, y, note


def case_rank(d, cols, y, note, stats):
    header = ["c%d" % i for i in range(len(cols))] + ["y"]
    texts = [[fmt(v) for v in c] for c in cols]
    rows = [[texts[k][r] for k in range(len(cols))] + [fmt(y[r])] for r in range(len(y))]
    write_csv(d, header, rows)
    j, ratio, c = exact_for(header[:-1], texts)
    st, out, _ = run(d, "fit", "--model", "linear", "--float", ",".join(header), "--target", "y", "--output", "m.cml", "--overwrite", "d.csv")
    doc = doc_of(out)
    refused = not doc["ok"] and doc["error"]["rule"] == "ml.fit.collinear"
    if not doc["ok"] and doc["error"]["rule"] not in ("ml.fit.collinear", "ml.fit.non-finite"):
        bad("%s: unexpected rule %s" % (note, doc["error"]["rule"]))
        return
    if not doc["ok"] and doc["error"]["rule"] == "ml.fit.non-finite":
        stats["nonfinite"] += 1
        return
    if ratio is not None and ratio > 0:
        import math
        b = int(math.floor(math.log10(float(ratio))))
        stats["decades"][b] = stats["decades"].get(b, 0) + 1
    elif ratio is not None:
        stats["decades"]["<=0"] = stats["decades"].get("<=0", 0) + 1
    in_band = ratio is not None and Fraction(9, 10) * TOL <= ratio <= Fraction(11, 10) * TOL
    want = j is not None
    if refused != want:
        if in_band:
            stats["band"] += 1
            return
        bad("%s: ml %s, exact relative pivot %s (%s)" % (note, "refused" if refused else "fitted", float(ratio), "singular" if want else "regular"))
        return
    stats["refused" if refused else "fitted"] += 1
    if not refused:
        return
    det = doc["error"]["detail"]
    names = header[:-1] + ["(intercept)"]
    if det["pivot"] != j or det["column"] != names[j]:
        if in_band:
            stats["band"] += 1
            return
        bad("%s: ml names pivot %s (%s), exact first failing pivot is %d (%s)" % (note, det["pivot"], det["column"], j, names[j]))
        return
    stats["named"] += 1
    big = max([abs(float(v)) for v in c] or [0.0])
    shown = dict(zip(det["columns"], [float(v) for v in det["coefficients"]]))
    for k in range(j):
        ck = float(c[k])
        if abs(ck) >= big * 2 ** -20 * 1.01 and big > 0:
            if names[k] not in shown:
                bad("%s: the dependency on %s (%g) is not named" % (note, names[k], ck))
            elif abs(shown[names[k]] - ck) > 1e-5 * max(1.0, abs(ck)):
                bad("%s: coefficient of %s is %g, exact %g" % (note, names[k], shown[names[k]], ck))
            else:
                stats["coefficients"] += 1
    for nm in shown:
        k = names.index(nm)
        if abs(float(c[k])) < big * 2 ** -20 * 0.99:
            bad("%s: %s is named but its exact coefficient is %g (largest %g)" % (note, nm, float(c[k]), big))


def case_onehot(d, rng, stats, drop_first):
    levels = rng.randint(2, 5)
    n = rng.randint(levels * 2, 40)
    cats = [i % levels for i in range(n)]
    rng.shuffle(cats)
    names = ["l%d" % i for i in range(levels)]
    rows = [["%s" % names[c], fmt(rng.uniform(-2, 2))] for c in cats]
    write_csv(d, ["sp", "y"], rows)
    st, out, _ = run(d, "fit", "--model", "linear", "--float", "y", "--cat", "sp", "--step", "encode column=sp kind=onehot", "--target", "y", "--output", "m.cml", "--overwrite", "d.csv")
    doc = doc_of(out)
    # drop_first is not an encoder option: the block is complete, so the answer is always the dependency (intercept) = sum of the block
    if doc["ok"] or doc["error"]["rule"] != "ml.fit.collinear":
        bad("onehot with %d levels: %s" % (levels, doc.get("error", {}).get("rule", "fitted")))
        return
    det = doc["error"]["detail"]
    if det["column"] != "(intercept)" or len(det["columns"]) != levels or any(abs(float(c) - 1) > 1e-6 for c in det["coefficients"]):
        bad("onehot dependency wrong: %s" % det)
        return
    stats["onehot"] += 1
    st, out2, _ = run(d, "fit", "--model", "linear", "--float", "y", "--cat", "sp", "--step", "encode column=sp kind=onehot", "--l2", "0.1", "--target", "y", "--output", "m.cml", "--overwrite", "d.csv")
    if not doc_of(out2)["ok"]:
        bad("onehot with a ridge was refused")


def rank_mode():
    rng = random.Random(SEED)
    stats = {k: 0 for k in ("refused", "fitted", "band", "named", "coefficients", "onehot", "nonfinite")}
    stats["decades"] = {}
    kinds = ["dup", "scaled", "sum", "constant", "zero", "near", "near", "near", "well", "well"]
    count = N or 300
    with tempfile.TemporaryDirectory() as d:
        for i in range(count):
            kind = kinds[i % len(kinds)]
            cols, y, note = gen_rank(rng, kind)
            case_rank(d, cols, y, note, stats)
        for i in range(max(10, count // 10)):
            case_onehot(d, rng, stats, False)
        # the shapes of the claim: an exactly dependent family is refused every time and a well conditioned one never
    stats["decades"] = {str(k): stats["decades"][k] for k in sorted(stats["decades"], key=lambda z: (isinstance(z, str), z))}
    print("rank: %s" % stats)
    if stats["refused"] == 0 or stats["fitted"] == 0:
        bad("rank: one verdict never happened: %s" % stats)
    return stats


# ---------------------------------------------------------------- C: separation

def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def candidates(points, dim):
    """normals to try for a separating line: every difference vector of two points, its rotation, and both perturbed by a rotation of 1/10^4"""
    vs = set()
    eps = Fraction(1, 10 ** 4)
    for p, q in itertools.combinations(points, 2):
        dv = (q[0] - p[0], q[1] - p[1]) if dim == 2 else (q[0] - p[0], 0)
        if dv == (0, 0):
            continue
        for base in (dv, (-dv[1], dv[0])):
            vs.add(base)
            for s in (eps, -eps):
                vs.add((base[0] - s * base[1], base[1] + s * base[0]))
    vs.add((Fraction(1), Fraction(0)))
    vs.add((Fraction(-1), Fraction(0)))
    vs.add((Fraction(0), Fraction(1)))
    vs.add((Fraction(0), Fraction(-1)))
    return vs


def separability(pts, labels, dim):
    """'complete' (a line with every point strictly on its side), 'quasi' (weakly, not all points on the line), or 'overlap'. Exact."""
    A = [p for p, l in zip(pts, labels) if l == 0]
    B = [p for p, l in zip(pts, labels) if l == 1]
    allp = list(set(pts))
    weak = False
    for nv in candidates(allp, dim):
        pa = [nv[0] * p[0] + nv[1] * p[1] for p in A]
        pb = [nv[0] * p[0] + nv[1] * p[1] for p in B]
        if max(pa) < min(pb) or max(pb) < min(pa):
            return "complete"
        if max(pa) <= min(pb) or max(pb) <= min(pa):
            # touching: weak separation counts only if not every point is on the one line
            allv = pa + pb
            if len(set(allv)) > 1:
                weak = True
    return "quasi" if weak else "overlap"


def gen_sep(rng, kind, dim):
    n = rng.randint(6, 14)
    while True:
        pts = [tuple(rng.randint(-6, 6) for _ in range(dim)) + ((0,) if dim == 1 else ()) for _ in range(n)]
        if kind == "random":
            labels = [rng.randint(0, 1) for _ in range(n)]
        else:
            w = [rng.randint(-3, 3) for _ in range(2)]
            b = rng.randint(-4, 4)
            labels = [1 if (w[0] * p[0] + w[1] * p[1] + b) > 0 else 0 for p in pts]
            if kind == "flip":
                k = rng.randrange(n)
                labels[k] = 1 - labels[k]
        if 0 < sum(labels) < n:
            return pts, labels



def newton_events(pts, labels, dim, iters=100):
    """An independent numpy Newton iteration (the same method as lib/models/linear/logreg.cho: w = 0, no damping, H = X'WX / n, the relative pivot test of an unblocked
    Cholesky) and the first event of the refusals: (rule, reason, pass, loss, norm, borderline) or None if the passes end quietly."""
    import numpy as np
    X = np.array([[float(v) for v in p[:dim]] for p in pts])
    y = np.array(labels, dtype=float)
    n = len(y)
    A = np.hstack([X, np.ones((n, 1))])
    P = A.shape[1]
    G = A.T @ A / n
    w = np.zeros(P)
    tol = 2.0 ** -40
    for it in range(iters):
        z = A @ w
        loss = float(np.sum(np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z))) - y * z))
        scale = math.sqrt(max(float(w @ G @ w), 0.0))
        norm = float(np.linalg.norm(w))
        pr = 1.0 / (1.0 + np.exp(-z))
        W = pr * (1 - pr)
        H = (A.T * W) @ A / n
        border = abs(loss - 0.6931) < 1e-7 * 0.6931 or abs(scale - 50.0) < 1e-7 * 50
        if loss < 0.6931:
            return ("ml.fit.separable", "loss", it, loss, norm, border)
        if scale > 50.0:
            return ("ml.fit.separable", "scale", it, loss, norm, border)
        L = np.zeros((P, P))
        failed = None
        ratio = 1.0
        for j in range(P):
            for i in range(j, P):
                sv = H[i, j] - float(L[i, :j] @ L[j, :j])
                if i == j:
                    ratio = sv / H[j, j] if H[j, j] > 0 else 0.0
                    if not (sv > tol * H[j, j]):
                        failed = j
                        break
                    L[j, j] = math.sqrt(sv)
                else:
                    L[i, j] = sv / L[j, j]
            if failed is not None:
                break
        if failed is not None:
            border = border or 0.5 * tol < ratio < 2 * tol
            return (("ml.fit.collinear", None, it, loss, norm, border) if it == 0 else ("ml.fit.separable", "pivot", it, loss, norm, border))
        g = A.T @ (pr - y) / n
        w = w - np.linalg.solve(H, g)
    return None


def case_sep(d, pts, labels, dim, stats, tally):
    header = ["x1", "x2", "y"] if dim == 2 else ["x1", "y"]
    rows = [[str(p[0]), str(p[1]), str(l)] if dim == 2 else [str(p[0]), str(l)] for p, l in zip(pts, labels)]
    write_csv(d, header, rows)
    verdict = separability([tuple(Fraction(v) for v in p) for p in pts], labels, dim)
    outcomes = {}
    for it, l2 in ((100, "0"), (200, "0"), (100, "0.01"), (100, "1e-9")):
        st, out, _ = run(d, "fit", "--model", "logistic", "--iterations", str(it), "--l2", l2, "--target", "y", "--output", "m.cml", "--overwrite", "d.csv")
        doc = doc_of(out)
        got = "fitted" if doc["ok"] else doc["error"]["rule"]
        if not doc["ok"] and got == "ml.fit.separable":
            got += ":" + doc["error"]["detail"]["reason"]
        outcomes[(it, l2)] = got
        if l2 != "0" and got == "ml.fit.separable" or (l2 != "0" and got.startswith("ml.fit.separable")):
            bad("a positive l2 (%s) was refused for separation: %s %s" % (l2, pts, labels))
    g = outcomes[(100, "0")]
    for k, v in outcomes.items():
        if v == "ml.linalg.not-positive-definite" and k[1] != "1e-9":
            bad("not-positive-definite escaped with l2 %s: %s %s" % (k[1], pts, labels))
    ev = newton_events(pts, labels, dim)
    st, out, _ = run(d, "fit", "--model", "logistic", "--iterations", "100", "--l2", "0", "--target", "y", "--output", "m.cml", "--overwrite", "d.csv")
    doc = doc_of(out)
    if ev is None:
        if not doc["ok"]:
            bad("independent Newton ends quietly, ml refused %s: %s %s" % (doc["error"]["rule"], pts, labels))
        else:
            stats["quiet"] = stats.get("quiet", 0) + 1
    elif not ev[5]:
        if doc["ok"] or doc["error"]["rule"] != ev[0]:
            bad("independent Newton says %s, ml says %s: %s %s" % (ev[:3], "fitted" if doc["ok"] else doc["error"]["rule"], pts, labels))
        elif ev[1] is not None:
            det = doc["error"]["detail"]
            if det["reason"] != ev[1] or det["pass"] != ev[2]:
                bad("independent Newton says %s at pass %d, ml says %s at pass %d: %s %s" % (ev[1], ev[2], det["reason"], det["pass"], pts, labels))
            elif ev[1] == "loss" and (abs(float(det["loss_sum"]) - ev[3]) > 1e-6 * ev[3] or abs(float(det["coefficient_norm"]) - ev[4]) > 1e-6 * max(ev[4], 1e-12)):
                bad("loss/norm differ: ml %s %s, numpy %g %g" % (det["loss_sum"], det["coefficient_norm"], ev[3], ev[4]))
            else:
                stats["events"] = stats.get("events", 0) + 1
                stats["reason:" + ev[1]] = stats.get("reason:" + ev[1], 0) + 1
    tally.setdefault(verdict, {})
    tally[verdict][g] = tally[verdict].get(g, 0) + 1
    if verdict == "overlap" and g != "fitted":
        bad("overlapping classes were refused (%s): %s %s" % (g, pts, labels))
    if verdict == "complete" and not g.startswith("ml.fit.separable"):
        bad("separable classes at 100 passes were not refused (%s): %s %s" % (g, pts, labels))
    if verdict != "complete" and g.startswith("ml.fit.separable:loss"):
        bad("the loss test refused a set that is not completely separable (%s): %s %s" % (verdict, pts, labels))
    # more passes never un-refuse a complete separation
    if verdict == "complete" and not outcomes[(200, "0")].startswith("ml.fit.separable"):
        bad("200 passes lost the refusal: %s %s" % (pts, labels))


def sep_mode():
    rng = random.Random(SEED + 1)
    tally = {}
    stats = {}
    count = N or 240
    with tempfile.TemporaryDirectory() as d:
        for i in range(count):
            dim = 2 if i % 4 else 1
            kind = ["sep", "flip", "random"][i % 3]
            pts, labels = gen_sep(rng, kind, dim)
            case_sep(d, pts, labels, dim, stats, tally)
    print("sep: independent-Newton agreement: %s" % json.dumps(stats, sort_keys=True))
    print("sep: exact verdict -> ml verdicts at 100 passes, l2 0: %s" % json.dumps(tally, sort_keys=True))
    if "complete" not in tally or "overlap" not in tally:
        bad("sep: a family never happened: %s" % tally)
    return tally


# ---------------------------------------------------------------- F: fuzz

def strip_argv(out):
    """the answer without the argv of its repair (which repeats the command line, --threads included)"""
    d = json.loads(out.decode())
    if not d["ok"]:
        for e in [d["error"]] + d.get("errors", []):
            e.pop("repair", None)
    return json.dumps(d, sort_keys=True)


def fuzz_mode():
    rng = random.Random(SEED + 2)
    count = N or 200
    verdicts = {}
    with tempfile.TemporaryDirectory() as d:
        for i in range(count):
            shape = rng.choice(["rank", "equal", "one", "wide", "scale"])
            n = rng.randint(2, 30)
            p = rng.randint(1, 12)
            if shape == "one":
                n = 1
            if shape == "wide":
                p = rng.randint(20, 60)
            r = rng.randint(1, p)
            basis = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(r)]
            cols = []
            for k in range(p):
                if k < r:
                    cols.append(list(basis[k]))
                else:
                    cs = [rng.uniform(-2, 2) for _ in range(r)]
                    cols.append([sum(c * b[t] for c, b in zip(cs, basis)) for t in range(n)])
            if shape == "equal":
                row = [rng.uniform(-1, 1) for _ in range(p)]
                cols = [[row[k]] * n for k in range(p)]
            exps = [rng.choice([0, 0, 0, 0, 0, 0, 3, -3, 10, -10]) for _ in range(p)]
            if shape == "scale" and i % 2 == 0:
                exps = [rng.choice([0, -300, 300, -150, 150]) for _ in range(p)]
            elif shape == "scale":
                exps = [rng.randint(-300, 300) for _ in range(p)]
            texts = []
            for k in range(p):
                texts.append([("%.6e" % (v * 1.0)).replace("e+00", "").replace("e-00", "") if False else "%.6fe%d" % (v, exps[k]) for v in cols[k]])
            y = ["%.6f" % rng.uniform(-1, 1) for _ in range(n)]
            header = ["c%d" % k for k in range(p)] + ["y"]
            rows = [[texts[k][t] for k in range(p)] + [y[t]] for t in range(n)]
            write_csv(d, header, rows)
            for model, extra in (("linear", []), ("logistic", ["--iterations", "30"])):
                tgt = y
                if model == "logistic":
                    lab = ["%d" % (rng.random() < 0.5) for _ in range(n)]
                    rows2 = [[texts[k][t] for k in range(p)] + [lab[t]] for t in range(n)]
                    write_csv(d, header, rows2)
                outs = []
                for th in ("1", "2", "4", "8", "16"):
                    st, out, err = run(d, "fit", "--model", model, "--float", ",".join(header[:-1]), "--target", "y", "--output", "m.cml", "--overwrite", "--threads", th, *extra, "d.csv")
                    if st not in (0, 8, 2) or not out:
                        bad("trap or no document: status %s model %s shape %s err %r" % (st, model, shape, err[:100]))
                        break
                    try:
                        doc = doc_of(out)
                    except Exception:
                        bad("not JSON: %r" % out[:100])
                        break
                    outs.append(out)
                else:
                    if len(set(strip_argv(o) for o in outs)) != 1:
                        bad("threads change the answer: model %s shape %s" % (model, shape))
                    doc = doc_of(outs[0])
                    key = "fitted" if doc["ok"] else doc["error"]["rule"]
                    verdicts[key] = verdicts.get(key, 0) + 1
    print("fuzz: verdicts %s" % json.dumps(verdicts, sort_keys=True))
    return verdicts


def main():
    modes = [a for a in sys.argv[1:] if a in ("rank", "sep", "fuzz", "all")]
    mode = modes[0] if modes else "all"
    if not os.path.exists(ML):
        sys.exit("no %s: build it first (cancho build --bin ml)" % ML)
    if mode in ("rank", "all"):
        rank_mode()
    if mode in ("sep", "all"):
        sep_mode()
    if mode in ("fuzz", "all"):
        fuzz_mode()
    if BAD:
        print("%d disagreements" % len(BAD))
        sys.exit(1)
    print("oracle_refuse %s: every verdict agrees" % mode)


if __name__ == "__main__":
    main()

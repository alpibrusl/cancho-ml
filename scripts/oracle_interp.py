#!/usr/bin/env python3
"""The oracle of interpretability (docs/interpret.md): `ml importance`, `ml dump` and the `importance` member of `describe`, written again in Python from the bytes of a
model file and the definitions of docs/interpret.md sections 3 and 4, with none of the code of tools/ml, lib/pipeline or lib/transforms.

What it reads, on its own: the file (scripts/oracle_st.py walks the sections and checks the checksum; the schema, the step records, the string table, the provenance section
and the model payloads are read here from docs/design.md 4.1, docs/transforms.md 4, docs/stringtable.md 2, docs/tracking.md 2.2, docs/models-linear.md 3.4 and
docs/models-tree.md 7). What it computes: the lineage of every column the model sees (the raw column it comes from, the stored scalings), the standardised coefficients, the
impurity decrease of a tree and of a forest from the stored counts, the dump of a tree and of the k-means centroids, and the permutation importance (the stream and the shuffle
of docs/rng.md and docs/metrics.md 2.1, the stored steps through scripts/oracle_pipe.py's `ref_apply`, the predictions and the metrics from the written definitions).

Every table is a list of dicts whose strings are the text `ml` prints: a double is the shortest decimal that reads back (`fnum`), a count is an int.

    from oracle_interp import read_model, importance_member, tree_dump, kmeans_dump, permutation, fnum
"""

import csv
import io
import math
import re
import struct
import sys
import pathlib
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_st as S  # noqa: E402

M64 = (1 << 64) - 1
GOLD = 0x9E3779B97F4A7C15
STREAM_PERMUTATION = 11
MAX_FLOAT = 1.7976931348623157e308
TYPES = {0: "int", 1: "dec", 2: "float", 3: "categorical"}

METHOD_STD = "standardised_coefficient"
METHOD_MDI = "impurity_decrease"
MEASURE_STD = "coefficient times the standard deviation of the column the model sees; the importance is the share of the absolute standardised coefficients"
MEASURE_MDI = "mean decrease in impurity from the stored node counts (Gini for a classification, variance for a regression), normalised per tree and averaged over the trees"

LOSS_METRICS = {"rmse", "mae", "log_loss"}
METRICS = {"regression": ["rmse", "mae", "r2"], "classification": ["roc_auc", "accuracy", "precision", "recall", "f1", "log_loss"]}
DEFAULT_METRIC = {"regression": "rmse", "classification": "roc_auc"}


class Refusal(Exception):
    def __init__(self, tag, **detail):
        super().__init__(tag)
        self.tag = tag
        self.detail = detail


# ---------------------------------------------------------------- numbers as `ml` prints them
def fnum(v):
    """a double as flt.put_float prints it: the shortest decimal that reads back, positional from 1e-6 to 1e21 with a point; zero (either sign) is 0.0"""
    if v == 0:
        return "0.0"
    if not 1e-6 <= abs(v) < 1e21:
        mant, _, exp = repr(v).partition("e")
        if not exp:
            mant, exp = ("%e" % v).split("e")
        return "%se%d" % (mant, int(exp))
    from decimal import Decimal

    s = format(Decimal(repr(v)), "f")
    return s if "." in s else s + ".0"


def fin(x):
    return x if math.isfinite(x) else 0.0


def tb(b):
    """text_or_bytes"""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        import base64
        return {"b64": base64.b64encode(b).decode()}


# ---------------------------------------------------------------- reading a model file
def read_schema(sc):
    cols, target = struct.unpack_from("<II", sc, 0)
    pos = 8
    out = []
    for _ in range(cols):
        nl = struct.unpack_from("<I", sc, pos)[0]
        pos += 4
        name = sc[pos:pos + nl]
        pos += nl
        ty, scale = struct.unpack_from("<II", sc, pos)
        pos += 8
        out.append((name, ty, scale))
    return out, target


def read_record(buf, pos):
    dt, rk = struct.unpack_from("<II", buf, pos)
    pos += 8
    ext = struct.unpack_from("<" + "Q" * rk, buf, pos)
    pos += 8 * rk
    n = 1
    for e in ext:
        n *= e
    vals = struct.unpack_from("<%d%s" % (n, "q" if dt == 1 else "d"), buf, pos)
    return list(vals), list(ext), pos + 8 * n


def read_steps(payload, table_entries):
    """the steps: dicts with kind and (for 1 to 6) ints and fl, (for 7 to 10) the payload dict of the string table"""
    n = struct.unpack_from("<I", payload, 0)[0]
    pos = 4
    out = []
    cat = 0
    for _ in range(n):
        ln = struct.unpack_from("<Q", payload, pos)[0]
        pos += 8
        body = payload[pos:pos + ln]
        pos += ln
        kind = struct.unpack_from("<I", body, 0)[0]
        if kind >= 7:
            out.append({"kind": kind, "cat": table_entries[cat][1]})
            cat += 1
        else:
            ints, _, p = read_record(body, 4)
            fl, _, p = read_record(body, p)
            out.append({"kind": kind, "ints": ints, "fl": fl})
    return out


def read_table_entries(t):
    layout, entries, pool, strings, na_len = struct.unpack_from("<IIQQI", t, 0)
    pos = 28 + na_len + 16
    out = []
    for _ in range(entries):
        step, ln = struct.unpack_from("<Iq", t, pos)
        out.append((step, S.parse_payload(t[pos + 12:pos + 12 + ln])))
        pos += 12 + ln
    return out


def read_provenance(p):
    """{rows, file_rows, features: [dict(count, min, max, mean, std)]} (docs/tracking.md 2.2)"""
    layout, = struct.unpack_from("<I", p, 0)
    file_bytes = struct.unpack_from("<Q", p, 4)[0]
    file_rows, rows = struct.unpack_from("<QQ", p, 44)
    nf = struct.unpack_from("<I", p, 60)[0]
    feats = []
    for k in range(nf):
        c, mn, mx, me, sd = struct.unpack_from("<Qdddd", p, 64 + 40 * k)
        feats.append({"count": c, "min": mn, "max": mx, "mean": me, "std": sd})
    return {"rows": rows, "file_rows": file_rows, "features": feats}


def read_model(data):
    """the model file as a dict: kind, model name, task, plan, cols, target, steps, prov, rows and the estimator's arrays"""
    sec = S.parse_file(data)
    M = {"data": data}
    plan = sec[1].decode("utf-8", "replace")
    M["plan"] = plan
    m = re.search(r"^model kind=(\w+)(?: task=(\w+))?", plan, re.M)
    M["model"] = m.group(1)
    M["cols"], M["target"] = read_schema(sec[2])
    M["table"] = read_table_entries(sec[6]) if 6 in sec else []
    M["steps"] = read_steps(sec[3], M["table"])
    M["prov"] = read_provenance(sec[7]) if 7 in sec else None
    meta = sec[5]
    M["rows"] = struct.unpack_from("<q", meta, 0)[0]
    mdl = sec[4]
    M["kind"] = struct.unpack_from("<I", mdl, 0)[0]
    M["model_bytes"] = mdl
    task = {"linear": "regression", "logistic": "classification", "kmeans": "clustering"}.get(M["model"])
    if M["model"] in ("tree", "forest"):
        recs = []
        pos = 4
        while pos < len(mdl):
            vals, _, pos = read_record(mdl, pos)
            recs.append(vals)
        h = recs[0]
        task = "classification" if h[0] == 0 else "regression"
        M["tree"] = {"task": h[0], "trees": h[1], "d": h[2], "bins": h[3], "total": h[4], "max_depth": h[5], "min_leaf": h[6], "mtry": h[7],
                     "off": recs[1], "feat": recs[2], "bin": recs[3], "dleft": recs[4], "left": recs[5], "right": recs[6], "na": recs[7], "nb": recs[8], "val": recs[9]}
        if len(recs) > 10:
            M["labels"] = tuple(recs[10])
    elif M["model"] in ("linear", "logistic"):
        coef, _, pos = read_record(mdl, 4)
        icpt, _, pos = read_record(mdl, pos)
        M["coef"], M["intercept"] = coef, icpt[0]
        if M["model"] == "logistic":
            labs, _, pos = read_record(mdl, pos)
            M["labels"] = tuple(labs)
    elif M["model"] == "kmeans":
        cen, ext, pos = read_record(mdl, 4)
        M["centroids"], M["k"], M["d"] = cen, ext[0], ext[1]
    M["task"] = task
    return M


# ---------------------------------------------------------------- the lineage of the columns the model sees
def numeric_features(M):
    return [j for j, (n, ty, sc) in enumerate(M["cols"]) if j != M["target"] and ty != 3]


def lineage(M):
    """one dict per column the model sees: name (bytes), source (bytes), src (schema index), k (index among the numeric features, or None), derived, scaled, off, div, unit (ddof of a
    last standard scaler, else None). An impute step changes none of them: the stored deviation is that of the present cells."""
    cols = M["cols"]
    cur = []
    for k, j in enumerate(numeric_features(M)):
        cur.append({"name": cols[j][0], "source": cols[j][0], "src": j, "k": k, "derived": False, "scaled": False, "off": 0.0, "div": 1.0, "unit": None})
    for st in M["steps"]:
        kind = st["kind"]
        if kind <= 6:
            ints, fl = st["ints"], st["fl"]
            k = ints[0]
            sel = ints[2:2 + k]
            if kind == 1:
                pass
            elif kind in (2, 3):
                for j, c in enumerate(sel):
                    a, d = fl[2 * j], fl[2 * j + 1]
                    col = cur[c]
                    col["off"] = col["off"] + a * col["div"]
                    col["div"] = col["div"] * d
                    col["scaled"] = True
                    col["unit"] = ints[2 + k] if kind == 3 else None
            elif kind == 4:
                cnts = ints[3 + k:3 + 2 * k]
                new = []
                at = {c: j for j, c in enumerate(sel)}
                for c, col in enumerate(cur):
                    if c in at:
                        for t in range(cnts[at[c]]):
                            nc = dict(col)
                            nc["name"] = col["name"] + b"#" + str(t).encode()
                            nc["derived"] = True
                            new.append(nc)
                    else:
                        new.append(col)
                cur = new
            else:
                for c in sel:
                    cur[c]["derived"] = True
        else:
            p = st["cat"]
            for name, strs in p["cols"]:
                j = [i for i, (n, ty, sc) in enumerate(cols) if n == name][0]
                if p["kind"] == 7:
                    w = len(cat_effective(strs, p["par"])[0])
                elif p["kind"] == 10:
                    w = p["par"]
                else:
                    w = 1
                blocked = p["kind"] in (7, 10)
                for t in range(w):
                    cur.append({"name": name + (b"#" + str(t).encode() if blocked else b""), "source": name, "src": j, "k": None, "derived": True, "scaled": False,
                                "off": 0.0, "div": 1.0, "unit": None})
    return cur


def cat_effective(strs, min_count):
    """strs: [(string, count)] bytewise ascending -> (effective strings ascending, {string: index}, counts)"""
    OTHER = b"__other__"
    kept = [(s, c) for s, c in strs if c >= min_count]
    merged = [(s, c) for s, c in strs if c < min_count]
    names = list(kept)
    if merged:
        names.append((OTHER, sum(c for _, c in merged)))
    names.sort(key=lambda t: t[0])
    index = {s: i for i, (s, _) in enumerate(names)}
    mapping = {s: index[s] for s, _ in kept}
    for s, _ in merged:
        mapping[s] = index[OTHER]
    return [s for s, _ in names], mapping, [c for _, c in names]


# ---------------------------------------------------------------- ranks
def ranks(imps):
    order = sorted(range(len(imps)), key=lambda j: (-imps[j], j))
    r = [0] * len(imps)
    for pos, j in enumerate(order):
        r[j] = pos + 1
    return r


# ---------------------------------------------------------------- the standardised coefficients (docs/interpret.md 3.1)
def column_sd(M, col):
    """(sd, source word) or None"""
    if col["derived"]:
        return None
    prov = M["prov"]
    pk = prov["features"][col["k"]] if prov and col["k"] is not None and col["k"] < len(prov["features"]) else None
    if col["unit"] is not None:
        if pk is not None and pk["std"] == 0.0:
            return 0.0, "scale"
        n = M["rows"]
        return (1.0 if col["unit"] == 0 else math.sqrt((n - 1) / n)), "scale"
    if pk is not None:
        return pk["std"] / col["div"], "provenance"
    return None


def listing(imps, limit):
    """the indices of the rows a table lists and whether it is cut: all of them in column order, or (when --limit cuts it) the `limit` best ranks, best first"""
    w = len(imps)
    listed = min(w, limit)
    if listed == w:
        return list(range(w)), False
    rk = ranks(imps)
    return sorted(range(w), key=lambda j: rk[j])[:listed], True


def std_member(M, limit=1000):
    cols = lineage(M)
    missing = [c["name"] for c in cols if column_sd(M, c) is None]
    if missing:
        shown = min(len(missing), limit)
        return {"method": METHOD_STD, "available": False, "reason": "ml.importance.no-scale", "measure": MEASURE_STD, "listed": shown, "truncated": shown < len(missing),
                "missing": [tb(n) for n in missing[:shown]]}
    rows = []
    st = []
    for j, c in enumerate(cols):
        sd, src = column_sd(M, c)
        b = M["coef"][j]
        z = b * sd
        st.append((b, sd, src, z))
    t = 0.0
    for b, sd, src, z in st:
        t = t + abs(z)
    imps = [abs(z) / t if t > 0.0 else 0.0 for b, sd, src, z in st]
    rk = ranks(imps)
    order, cut = listing(imps, limit)
    for j in order:
        c = cols[j]
        b, sd, src, z = st[j]
        rows.append({"column": tb(c["name"]), "source": tb(c["source"]), "coefficient": fnum(b), "std": fnum(sd), "std_source": src, "standardised": fnum(z), "importance": fnum(imps[j]), "rank": rk[j]})
    return {"method": METHOD_STD, "available": True, "measure": MEASURE_STD, "total": "1.0" if t > 0.0 else "0.0", "listed": len(rows), "truncated": cut, "columns": rows}


# ---------------------------------------------------------------- the impurity decrease (docs/interpret.md 3.2)
MAX_COUNT = 1 << 30


def tree_decreases(T, t):
    """-> [(node index in the tree, feature, decrease)] for the internal nodes of tree t in preorder; and the node rows"""
    lo, hi = T["off"][t], T["off"][t + 1]
    cls = T["task"] == 0
    na, nb, val = T["na"], T["nb"], T["val"]
    left, right, feat = T["left"], T["right"], T["feat"]
    S_ = {}
    if not cls:
        for i in range(hi - 1, lo - 1, -1):
            if feat[i] < 0:
                S_[i] = float(na[i]) * val[i]
            else:
                S_[i] = S_[lo + left[i]] + S_[lo + right[i]]
    out = []
    for i in range(lo, hi):
        if feat[i] < 0:
            continue
        L, R = lo + left[i], lo + right[i]
        if cls:
            n, nl, nr = na[i] + nb[i], na[L] + nb[L], na[R] + nb[R]
            if any(c < 0 or c > MAX_COUNT for c in (na[i], nb[i], na[L], nb[L], na[R], nb[R])) or n < 1 or nl < 1 or nr < 1:
                dec = 0.0
            else:
                D = nb[L] * nr - nb[R] * nl
                df = float(D)
                dec = (2.0 * (df * df)) / (float(n) * (float(nl) * float(nr)))
        else:
            n, nl, nr = na[i], na[L], na[R]
            if n < 1 or nl < 1 or nr < 1 or n > MAX_COUNT or nl > MAX_COUNT or nr > MAX_COUNT:
                dec = 0.0
            else:
                ml_, mr = S_[L] / float(nl), S_[R] / float(nr)
                d = ml_ - mr
                dec = ((float(nl) * float(nr)) / float(n)) * (d * d)
        out.append((i - lo, feat[i], fin(dec), (na[i] + nb[i]) if cls else na[i]))
    return out


def tree_scores(M):
    """-> (imps, splits, split_rows) per model column"""
    T = M["tree"]
    d = T["d"]
    acc = [0.0] * d
    splits = [0] * d
    srows = [0] * d
    for t in range(T["trees"]):
        tf = [0.0] * d
        for _i, f, dec, rows in tree_decreases(T, t):
            nv = tf[f] + dec
            tf[f] = nv if math.isfinite(nv) else tf[f]
            splits[f] += 1
            srows[f] += rows
        tt = 0.0
        for f in range(d):
            tt = tt + tf[f]
        if not math.isfinite(tt) or not tt > 0.0:
            continue
        for f in range(d):
            acc[f] = acc[f] + tf[f] / tt
    s = 0.0
    for f in range(d):
        s = s + acc[f]
    imps = [acc[f] / s if (math.isfinite(s) and s > 0.0) else 0.0 for f in range(d)]
    return imps, splits, srows, (math.isfinite(s) and s > 0.0)


def tree_member(M, limit=1000):
    cols = lineage(M)
    T = M["tree"]
    imps, splits, srows, positive = tree_scores(M)
    rk = ranks(imps)
    width = min(len(imps), len(cols))
    order, cut = listing(imps[:width], limit) if width == len(imps) else (list(range(width)), False)
    rows = []
    for j in order:
        c = cols[j]
        rows.append({"column": tb(c["name"]), "source": tb(c["source"]), "importance": fnum(imps[j]), "rank": rk[j], "splits": splits[j], "split_rows": srows[j]})
    return {"method": METHOD_MDI, "available": True, "measure": MEASURE_MDI, "trees": T["trees"], "total": "1.0" if positive else "0.0", "listed": len(rows), "truncated": cut, "columns": rows}


def importance_member(M, limit=1000):
    """the `importance` member of describe (the rows listed are cut at `limit`, as the categories are), or None for the kinds that have none"""
    if M["model"] in ("linear", "logistic"):
        return std_member(M, limit)
    if M["model"] in ("tree", "forest"):
        return tree_member(M, limit)
    return None


# ---------------------------------------------------------------- the dump (docs/interpret.md 4.3)
def cell_text(M, col, v):
    """the text a data cell of the raw column would have for the value v, or None when the steps changed the column"""
    if col["derived"] or col["scaled"]:
        return None
    ty, scale = M["cols"][col["src"]][1], M["cols"][col["src"]][2]
    if ty == 0:
        if v == math.floor(v) and abs(v) < 9.0e18:
            return str(int(v))
        return fnum(v)
    if ty == 1:
        t = v * (10.0 ** scale)
        if abs(t) < 2.0 ** 50:
            k = int(math.floor(t + 0.5)) if t >= 0.0 else -int(math.floor(-t + 0.5))
            neg = k < 0
            digits = str(abs(k)).rjust(scale + 1, "0")
            body = digits[:-scale] + "." + digits[-scale:] if scale > 0 else digits
            return ("-" if neg else "") + body
        return fnum(v)
    return fnum(v)


def tree_dump(M, i, max_nodes=10000):
    T = M["tree"]
    if i < 0 or i >= T["trees"]:
        raise Refusal("ml.param.out-of-range", tree=i, trees=T["trees"])
    lo, hi = T["off"][i], T["off"][i + 1]
    n = hi - lo
    if n > max_nodes:
        raise Refusal("ml.limit.nodes", nodes=n, max=max_nodes)
    cols = lineage(M)
    cls = T["task"] == 0
    depth = {0: 0}
    nodes = []
    maxd = 0
    for g in range(lo, hi):
        li = g - lo
        dp = depth[li]
        maxd = max(maxd, dp)
        a, b = T["na"][g], T["nb"][g]
        counts = [a, b] if cls else [a]
        rows = a + b if cls else a
        if T["feat"][g] < 0:
            nodes.append({"id": li, "depth": dp, "leaf": True, "value": fnum(T["val"][g]), "counts": counts, "rows": rows})
        else:
            f = T["feat"][g]
            c = cols[f]
            thr = T["val"][g]
            ct = cell_text(M, c, thr)
            depth[T["left"][g]] = dp + 1
            depth[T["right"][g]] = dp + 1
            nodes.append({"id": li, "depth": dp, "leaf": False, "column": tb(c["name"]), "source": tb(c["source"]), "threshold": fnum(thr), "cell": ct,
                          "missing": "left" if T["dleft"][g] == 1 else "right", "left": T["left"][g], "right": T["right"][g], "counts": counts, "rows": rows})
    return {"model": M["model"], "task": M["task"], "kind": M["kind"], "trees": T["trees"], "tree": i, "nodes": n, "depth": maxd, "list": nodes}


def kmeans_dump(M):
    cols = lineage(M)
    out = []
    d, k = M["d"], M["k"]
    for c in range(k):
        vals = []
        for j in range(d):
            z = M["centroids"][c * d + j]
            col = cols[j]
            raw = None
            if not col["derived"]:
                raw = fnum(z * col["div"] + col["off"])
            vals.append({"column": tb(col["name"]), "source": tb(col["source"]), "value": fnum(z), "raw": raw})
        out.append({"cluster": c, "values": vals})
    return {"model": "kmeans", "task": "clustering", "kind": M["kind"], "k": k, "centroids": out}


def linear_dump(M):
    cols = lineage(M)
    return {"model": M["model"], "task": M["task"], "kind": M["kind"], "intercept": fnum(M["intercept"]),
            "coefficients": [{"column": tb(c["name"]), "source": tb(c["source"]), "coefficient": fnum(M["coef"][j])} for j, c in enumerate(cols)]}


# ---------------------------------------------------------------- predictions of a stored model on raw cells (numeric and categorical)
def apply_cat_step(p, cells_by_col, nrows):
    """the rows of the block a categorical step writes (scripts/oracle_enc.py's rules, written again): cells_by_col {name: [bytes or None]}"""
    out = [[] for _ in range(nrows)]
    for name, strs in p["cols"]:
        kind = p["kind"]
        if kind != 10:
            names, mapping, ecounts = cat_effective(strs, p["par"])
            known = {s for s, _ in strs}
            width = len(names) if kind == 7 else 1
        else:
            import hashlib
            width = p["par"]
        for r, cell in enumerate(cells_by_col[name]):
            if cell is None:
                if p["missing"] == 0:
                    out[r] += [None] * width
                else:
                    out[r] += [-1.0] if kind == 8 else [0.0] * width
                continue
            if kind == 10:
                row = [0.0] * width
                row[int.from_bytes(hashlib.sha256(cell).digest()[:8], "big") % width] = 1.0
                out[r] += row
                continue
            if cell not in known:
                out[r] += [-1.0] if kind == 8 else [0.0] * width
                continue
            e = mapping[cell]
            if kind == 7:
                row = [0.0] * width
                row[e] = 1.0
                out[r] += row
            elif kind == 8:
                out[r] += [float(e)]
            else:
                out[r] += [ecounts[e] / p["rows"]]
    return out


def through_steps(M, num_rows, cat_cells):
    """num_rows: rows of the numeric features (floats or None) in schema order; cat_cells: {name: [bytes or None]} -> the rows the estimator sees"""
    import oracle_pipe as op
    cur = num_rows
    n = len(num_rows) if num_rows else (len(next(iter(cat_cells.values()))) if cat_cells else 0)
    for st in M["steps"]:
        if st["kind"] <= 6:
            cur, _u = op.ref_apply(st["kind"], st["ints"], st["fl"], cur)
        else:
            blk = apply_cat_step(st["cat"], cat_cells, n)
            if not cur:
                cur = [[] for _ in range(n)]
            cur = [a + b for a, b in zip(cur, blk)]
    return cur


def proba_of(z):
    import oracle_models as omod
    az = abs(z)
    e = float(omod.exp_c(-az))
    return 1.0 / (1.0 + e) if z >= 0.0 else e / (1.0 + e)


def predict(M, X):
    """-> ('prediction', values) or ('label', labels, probabilities of the positive label) for the rows X the estimator sees"""
    import numpy as np
    model = M["model"]
    if model in ("linear", "logistic"):
        import oracle_models as omod
        Xn = np.array(X, dtype=np.float64).reshape(len(X), len(M["coef"]))
        w = list(M["coef"]) + [M["intercept"]]
        z = omod.scores(Xn, w)
        if model == "linear":
            return "prediction", [float(v) for v in z]
        c0, c1 = M["labels"]
        return "label", [c1 if v > 0.0 else c0 for v in z], [proba_of(float(v)) for v in z]
    T = M["tree"]
    out = []
    for row in X:
        acc = 0.0
        for t in range(T["trees"]):
            lo = T["off"][t]
            i = lo
            while T["feat"][i] >= 0:
                x = row[T["feat"][i]]
                if x is None:
                    go_left = T["dleft"][i] == 1
                else:
                    go_left = x < T["val"][i]
                i = lo + (T["left"][i] if go_left else T["right"][i])
            acc = acc + T["val"][i]
        out.append(acc / float(T["trees"]))
    if M["task"] == "regression":
        return "prediction", out
    c0, c1 = M["labels"]
    return "label", [c1 if v > 0.5 else c0 for v in out], out


# ---------------------------------------------------------------- the metrics (docs/metrics.md; lane order of docs/linalg.md section 2)
K, LANES = 256, 4


def _tree_sum(p):
    p = list(p)
    while len(p) > 1:
        m = len(p)
        h = m // 2
        q = [p[2 * i] + p[2 * i + 1] for i in range(h)]
        if m % 2:
            q.append(p[m - 1])
        p = q
    return p[0]


def ordered_sum(t):
    n = len(t)
    nb = -(-n // K)
    pad = list(t) + [0.0] * (nb * K - n)
    part = []
    for b in range(nb):
        lanes = [0.0] * LANES
        base = b * K
        for j in range(K // LANES):
            for l in range(LANES):
                lanes[l] = lanes[l] + pad[base + j * LANES + l]
        part.append((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
    return _tree_sum(part)


def metric_value(name, y, pred, pos=None, labels=None, prob=None):
    """the value of a metric, or None when it has no value; y floats (regression) or ints; pred values or labels; prob the positive label's probability"""
    n = len(y)
    if name == "rmse":
        return math.sqrt(ordered_sum([(a - b) * (a - b) for a, b in zip(y, pred)]) / float(n))
    if name == "mae":
        return ordered_sum([abs(a - b) for a, b in zip(y, pred)]) / float(n)
    if name == "r2":
        if all(v == y[0] for v in y):
            return None
        mean = ordered_sum(list(y)) / float(n)
        sst = ordered_sum([(a - mean) * (a - mean) for a in y])
        sse = ordered_sum([(a - b) * (a - b) for a, b in zip(y, pred)])
        if sst == 0.0:
            return None
        r = 1.0 - sse / sst
        return r if math.isfinite(r) else None
    yt = [int(v) for v in y]
    if name == "accuracy":
        return sum(1 for a, b in zip(yt, pred) if a == b) / n
    tp = sum(1 for a, b in zip(yt, pred) if a == pos and b == pos)
    fp = sum(1 for a, b in zip(yt, pred) if a != pos and b == pos)
    fn = sum(1 for a, b in zip(yt, pred) if a == pos and b != pos)
    if name == "precision":
        return tp / (tp + fp) if tp + fp else None
    if name == "recall":
        return tp / (tp + fn) if tp + fn else None
    if name == "f1":
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    if name == "roc_auc":
        return auc(yt, prob, pos)
    if name == "log_loss":
        eps = 2.220446049250313e-16
        terms = []
        for a, q in zip(yt, prob):
            x = q if a == pos else 1.0 - q
            x = min(max(x, eps), 1.0 - eps)
            terms.append(-math.log(x))
        return ordered_sum(terms) / float(n)
    raise ValueError(name)


def auc(yt, score, pos):
    """float(Fraction) of the Mann-Whitney statistic with midranks; None when a class is empty"""
    order = sorted(range(len(yt)), key=lambda i: score[i])
    ranks2 = [0] * len(yt)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and score[order[j + 1]] == score[order[i]]:
            j += 1
        # doubled midrank of the 1-based positions i+1 .. j+1: (i + 1) + (j + 1)
        for q in range(i, j + 1):
            ranks2[order[q]] = (i + 1) + (j + 1)
        i = j + 1
    P = sum(1 for v in yt if v == pos)
    N = len(yt) - P
    if P == 0 or N == 0:
        return None
    two_rplus = sum(ranks2[i] for i in range(len(yt)) if yt[i] == pos)
    two_u = two_rplus - P * (P + 1)
    return float(Fraction(two_u, 2 * P * N))


# ---------------------------------------------------------------- the stream (docs/rng.md, docs/metrics.md 2.1)
def mix(z):
    z &= M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def derive(h, c):
    return mix((((h & M64) ^ (c & M64)) + GOLD) & M64)


def permutation_of(n, seed, repeat, column):
    s = derive(derive(derive(seed, STREAM_PERMUTATION), repeat), column)
    a = list(range(n))
    for i in range(n - 1, 0, -1):
        m = i + 1
        r = (1 << 63) % m
        while True:
            s = (s + GOLD) & M64
            v = mix(s) >> 1
            if v < (1 << 63) - r:
                break
        j = v % m
        a[i], a[j] = a[j], a[i]
    return a


# ---------------------------------------------------------------- the data, as `eval` reads it, and the permutation importance (docs/interpret.md 3.4)
def read_data(M, text):
    """-> (num_rows, cat_cells, y, n): the numeric feature cells in schema order (floats or None), the categorical cells by name, the target (floats)"""
    rd = csv.reader(io.StringIO(text))
    header = next(rd)
    rows = [r for r in rd]
    names = [h.encode() for h in header]
    cols = M["cols"]
    idx = {n: i for i, n in enumerate(names)}
    feats = numeric_features(M)
    num = []
    for r in rows:
        num.append([None if r[idx[cols[j][0]]] == "" else float(r[idx[cols[j][0]]]) for j in feats])
    cats = {}
    for j, (n, ty, sc) in enumerate(cols):
        if ty == 3 and j != M["target"]:
            cats[n] = [None if r[idx[n]] == "" else r[idx[n]].encode() for r in rows]
    t = cols[M["target"]][0]
    y = [float(r[idx[t]]) for r in rows]
    return num, cats, y, len(rows)


def permutation(M, text, metric=None, repeats=5, seed=0):
    """the `permutation` object of `ml importance --data`, as a dict of strings and ints"""
    task = M["task"]
    if task not in METRICS:
        raise Refusal("ml.importance.unsupported-model")
    metric = metric or DEFAULT_METRIC[task]
    if metric not in METRICS[task]:
        raise Refusal("ml.importance.bad-metric", metric=metric)
    num, cats, y, n = read_data(M, text)
    if n < 2:
        raise Refusal("ml.importance.too-few-rows", rows=n)
    pos = M["labels"][1] if task == "classification" else None

    def score(num_rows, cat_cells):
        X = through_steps(M, num_rows, cat_cells)
        out = predict(M, X)
        if task == "regression":
            v = metric_value(metric, y, out[1])
        else:
            v = metric_value(metric, y, out[1], pos=pos, prob=out[2])
        if v is None:
            raise Refusal("ml.importance.undefined-metric", metric=metric)
        return v

    base = score(num, cats)
    loss = metric in LOSS_METRICS
    feats = numeric_features(M)
    cols = M["cols"]
    results = []
    for j, (name, ty, sc) in enumerate(cols):
        if j == M["target"]:
            continue
        ds = []
        for r in range(repeats):
            a = permutation_of(n, seed, r, j)
            if ty == 3:
                cc = dict(cats)
                cc[name] = [cats[name][a[i]] for i in range(n)]
                p = score(num, cc)
            else:
                k = feats.index(j)
                nn = [list(row) for row in num]
                for i in range(n):
                    nn[i][k] = num[a[i]][k]
                p = score(nn, cats)
            ds.append((p - base) if loss else (base - p))
        mean = 0.0
        for d in ds:
            mean = mean + d
        mean = mean / float(repeats)
        var = 0.0
        for d in ds:
            var = var + (d - mean) * (d - mean)
        results.append((name, mean, math.sqrt(var / float(repeats))))
    rk = ranks([m for _, m, _ in results])
    return {"metric": metric, "direction": "loss" if loss else "score", "repeats": repeats, "seed": seed, "rows": n, "baseline": fnum(base),
            "columns": [{"column": tb(nm), "importance": fnum(m), "std": fnum(s), "rank": rk[i]} for i, (nm, m, s) in enumerate(results)]}


# ---------------------------------------------------------------- the data sets of the gates (docs/interpret.md 8.1)
def unit_stream(seed):
    """a generator of floats in [-1, 1): the unit() of docs/rng.md on the stream that starts at `seed`"""
    s = seed & M64
    while True:
        s = (s + GOLD) & M64
        z = mix(s)
        yield ((z >> 11) & ((1 << 53) - 1)) * 2.0 ** -52 - 1.0


def gen_dataset(name, n, seed):
    """-> (header, rows) of text cells. cls / reg: 6 numeric columns x0..x5 and NOISE; mixed: numeric columns with 8 percent missing cells and a categorical column"""
    u = unit_stream(derive(seed, 7))
    nz = unit_stream(derive(seed, 99))
    rows = []
    if name in ("cls", "reg"):
        for i in range(n):
            x = [next(u) for _ in range(6)]
            noise = next(nz)
            e = next(u) * (1.5 if name == "cls" else 0.3)
            if name == "cls":
                z = 2.0 * x[0] - 1.5 * x[1] + 1.0 * x[2] + 0.7 * x[3] + 0.5 * x[4] + 0.35 * x[5] + e
                y = "1" if z > 0.0 else "0"
            else:
                y = repr(3.0 * x[0] + 2.0 * x[1] * x[1] + (1.0 if x[2] > 0.0 else -1.0) + 0.7 * x[3] + 0.5 * x[4] + 0.35 * x[5] + e)
            cells = [repr(v) for v in x]
            rows.append(cells[:3] + [repr(noise)] + cells[3:] + [y])
        return ["x0", "x1", "x2", "NOISE", "x3", "x4", "x5", "y"], rows
    if name == "disc":
        # an integer column g with four values, two numeric columns: a one-hot encoding of g is a plan with a numeric encoder
        eff = [0.0, 2.0, -1.0, 1.0]
        for i in range(n):
            g = int((next(u) + 1.0) * 2.0) % 4
            x = [next(u), next(u)]
            e = next(u) * 0.2
            rows.append([str(g), repr(x[0]), repr(x[1]), repr(eff[g] + 1.2 * x[0] + e)])
        return ["g", "x0", "x1", "y"], rows
    if name == "rare":
        # a classification with a handful of positives: some bootstrap samples hold none, and the tree grown on one is a single leaf
        for i in range(n):
            x = [next(u) for _ in range(3)]
            rows.append([repr(v) for v in x] + ["1" if i in (7, 211) else "0"])
        return ["x0", "x1", "x2", "y"], rows
    if name == "mixed":
        cats = ["red", "green", "blue", "gold", "teal"]
        eff = {"red": 2.0, "green": -1.0, "blue": 0.5, "gold": -2.0, "teal": 1.0}
        for i in range(n):
            x = [next(u) for _ in range(3)]
            c = cats[int((next(u) + 1.0) * 2.5) % 5]
            e = next(u) * 0.2
            y = repr(eff[c] + 1.5 * x[0] + 0.7 * x[1] + 0.5 * x[2] + e)
            cells = [repr(v) for v in x]
            for k in (0, 1, 2):
                if (next(nz) + 1.0) * 0.5 < 0.08:
                    cells[k] = ""
            rows.append(cells + [c, repr(next(nz)), y])
        return ["a", "b", "c", "color", "NOISE", "y"], rows
    raise ValueError(name)


def to_csv(header, rows):
    return ",".join(header) + "\n" + "".join(",".join(r) + "\n" for r in rows)

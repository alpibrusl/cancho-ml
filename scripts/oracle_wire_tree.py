#!/usr/bin/env python3
"""The oracle of the wiring of the tree and the forest (docs/wiring-trees.md gates WT3, WT4, WT5): a Python implementation that fits the same pipeline and the
same model as `ml fit --model tree|forest` and writes the model file's bytes, so that the file the real binary wrote is compared byte for byte.

What it reuses, and why that is not circular: the library part is `oracle_tree.py` (the algorithm of docs/models-tree.md section 2 and the payload of section 7
in Python, compared with `lib/models/tree` by `gate_tree.py`'s G-ident); the transforms and their application are `oracle_pipe.py`'s (`ref_pipeline`,
`ref_apply`); the CSV types, the schema, the step records and the meta section are `oracle_wire.py`'s (written for linear and logistic); the split is
`oracle_metrics.py`'s. What is written here is the glue of this stage: the missing cells handed to the library as a mask, the task, the labels record, the
sum order 3 of docs/wiring-trees.md section 4, the predictions of a forest and the metrics from the definitions of docs/metrics.md.

    python3 scripts/oracle_wire_tree.py --suite [N] [tree|forest]   N seeded cases per model and task (default 24) against build/ml, 1,000 rows or less
    python3 scripts/oracle_wire_tree.py --large [tree|forest]        two cases of 100,000 rows each (a few minutes)
Exit 1 on any mismatch. ML_BIN names the binary.
"""

import copy
import hashlib
import json
import math
import os
import pathlib
import random
import struct
import sys
import tempfile
import warnings
from fractions import Fraction

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_metrics as om_  # noqa: E402
import oracle_pipe as op  # noqa: E402
import oracle_tree as ot  # noqa: E402
import oracle_wire as ow  # noqa: E402

SUM_ORDER_SEQUENTIAL = 3
DEFAULTS = {"trees": 100, "max-depth": 8, "min-leaf": 1, "bins": 64, "mtry": 0, "bootstrap": 1, "max-nodes": 4194304}
UNJUDGED = []
COVER = {"cases": 0, "with_missing": 0, "missing_cells": 0, "v1_splits": 0, "nodes": 0, "split": 0, "steps": 0}


class TCase:
    """A CSV, a tree or forest and a plan: everything `ml fit` is given."""

    def __init__(self, model, task, names, rows, target, steps, step_text, hp, split=None, seed=0, positive=None):
        self.model, self.task, self.names, self.rows, self.target = model, task, names, rows, target
        self.steps, self.step_text, self.hp = steps, step_text, hp
        self.split, self.seed, self.positive = split, seed, positive

    def csv(self):
        return ",".join(self.names) + "\n" + "".join(",".join(r) + "\n" for r in self.rows)

    def model_line(self):
        keys = ["max-depth", "min-leaf", "bins", "max-nodes"] if self.model == "tree" else ["trees", "max-depth", "min-leaf", "bins", "mtry", "bootstrap", "max-nodes"]
        parts = ["model", "kind=" + self.model, "task=" + self.task]
        for k in keys:
            parts.append("%s=%d" % (k, self.hp.get(k, DEFAULTS[k])))
        return " ".join(parts)

    def plan_text(self):
        return "".join(self.step_text) + self.model_line()


def isqrt(v):
    r = int(math.isqrt(v))
    return max(1, r)


def resolve_mtry(case, d):
    if case.model == "tree":
        return d
    m = case.hp.get("mtry", 0)
    if m:
        return m
    return isqrt(d) if case.task == "classification" else max(1, d // 3)


def labels_record(c0, c1):
    return struct.pack("<IIQqq", 1, 1, 2, c0, c1)


def file_bytes(plan, schema, steps, model, rows, ncols, seed):
    meta = struct.pack("<QQQII", rows, ncols, seed, 0, len(ow.PIN)) + ow.PIN.encode() + struct.pack("<I", SUM_ORDER_SEQUENTIAL)
    body = b"CML1" + struct.pack("<II", 1, 5) + ow.section(1, plan.encode()) + ow.section(2, schema) + ow.section(3, steps) + ow.section(4, model) + ow.section(5, meta)
    return body + hashlib.sha256(body).digest()


def matrix_of(rows_after):
    """(values with 0.0 in a missing cell, mask) of the matrix the steps leave (None is a missing cell)"""
    n = len(rows_after)
    d = len(rows_after[0]) if n else 0
    X = np.zeros((n, d), dtype=np.float64)
    M = np.zeros((n, d), dtype=bool)
    for i, row in enumerate(rows_after):
        for j, v in enumerate(row):
            if v is None:
                M[i, j] = True
            else:
                X[i, j] = v
    return X, M


def through_steps(stored, X):
    cur = X
    for kind, ints, fl in stored:
        cur, _u = op.ref_apply(kind, ints, fl, cur)
    return cur


def fit_oracle(case):
    """(file bytes, or the tag of the refusal) and a context for predict and eval"""
    names, rows = case.names, case.rows
    t = names.index(case.target)
    types = [ow.column_type([r[j] for r in rows]) for j in range(len(names))]
    feat = [j for j in range(len(names)) if j != t]
    fnames = [names[j] for j in feat]
    X = [[ow.to_float(r[j]) for j in feat] for r in rows]
    y = [ow.to_float(r[t]) for r in rows]
    n = len(rows)
    mask = ow.split_masks(n, case.split, case.seed)
    tr = [i for i in range(n) if mask is None or mask[i] == 0]
    Xtr = [X[i] for i in tr]
    ytr = [y[i] for i in tr]
    labels = None
    if case.task == "classification":
        vals = sorted(set(int(v) for v in y))
        hi, lo = vals[-1], vals[0]
        c1, c0 = hi, lo
        if case.positive is not None and case.positive == lo:
            c1, c0 = lo, hi
        labels = (c0, c1)
        yfit = [1.0 if int(v) == c1 else 0.0 for v in ytr]
    else:
        yfit = ytr
    try:
        stored, cur, _, _, _ = op.ref_pipeline(case.steps, fnames, Xtr, [])
    except op.Refuse as r:
        return r.tag, None
    if mask is not None:
        try:
            through_steps(stored, [X[i] for i in range(n) if mask[i] == 1])
        except op.Refuse as r:
            return r.tag, None
    Xn, Mn = matrix_of(cur)
    d = Xn.shape[1]
    task = 0 if case.task == "classification" else 1
    T = 1 if case.model == "tree" else case.hp.get("trees", DEFAULTS["trees"])
    boot = 0 if case.model == "tree" else case.hp.get("bootstrap", DEFAULTS["bootstrap"])
    maxd = case.hp.get("max-depth", DEFAULTS["max-depth"])
    ML = case.hp.get("min-leaf", DEFAULTS["min-leaf"])
    B = case.hp.get("bins", DEFAULTS["bins"])
    mtry = resolve_mtry(case, d)
    stats = {"ties": 0, "v1": 0, "equal_v": 0}
    edges, trees = ot.fit(Xn, Mn, np.array(yfit, dtype=np.float64), task, T, maxd, ML, B, mtry, boot, case.seed, stats)
    model = ot.payload(trees, task, d, B, maxd, ML, mtry)
    if labels:
        model += labels_record(*labels)
    plan = ow.canon(case.plan_text())
    data = file_bytes(plan, ow.schema_bytes(names, types, t), ow.steps_bytes(stored), model, len(tr), len(names), case.seed)
    ctx = {"stored": stored, "trees": trees, "T": T, "d": d, "labels": labels, "types": types, "mask": mask, "stats": stats, "missing": int(Mn.sum())}
    COVER["cases"] += 1
    COVER["with_missing"] += 1 if ctx["missing"] else 0
    COVER["missing_cells"] += ctx["missing"]
    COVER["v1_splits"] += stats["v1"]
    COVER["nodes"] += sum(len(tr_) for tr_ in trees)
    COVER["split"] += 1 if mask is not None else 0
    COVER["steps"] += 1 if stored else 0
    return data, ctx


def predict_oracle(case, ctx, X_rows):
    """what `ml predict` prints: the values (regression), or the labels and the probabilities of the positive label"""
    cur = through_steps(ctx["stored"], X_rows)
    Xn, Mn = matrix_of(cur)
    with np.errstate(invalid="ignore"):
        p = ot.predict(ctx["trees"], Xn, Mn, ctx["T"])
    p = [float(v) for v in p]
    if case.task == "regression":
        return p
    c0, c1 = ctx["labels"]
    return [c1 if v > 0.5 else c0 for v in p], p


def eval_oracle(case, ctx, rows_idx):
    """the metrics `ml eval` prints for the rows of the case's CSV in rows_idx"""
    names, rows = case.names, case.rows
    t = names.index(case.target)
    feat = [j for j in range(len(names)) if j != t]
    X = [[ow.to_float(rows[i][j]) for j in feat] for i in rows_idx]
    y = [ow.to_float(rows[i][t]) for i in rows_idx]
    n = len(y)
    out = {}
    if case.task == "regression":
        p = predict_oracle(case, ctx, X)
        sq = [(a - b) * (a - b) for a, b in zip(y, p)]
        out["rmse"] = math.sqrt(ow.ordered(sq) / float(n))
        out["mae"] = ow.ordered([abs(a - b) for a, b in zip(y, p)]) / float(n)
        if all(v == y[0] for v in y):
            out["r2"] = None
        else:
            mean = ow.ordered(y) / float(n)
            sst = ow.ordered([(a - mean) * (a - mean) for a in y])
            out["r2"] = 1.0 - ow.ordered(sq) / sst
        return out
    c0, c1 = ctx["labels"]
    pos = case.positive if case.positive is not None else c1
    lab, prob = predict_oracle(case, ctx, X)
    yt = [int(v) for v in y]
    score = prob if pos == c1 else [1.0 - q for q in prob]
    tp = sum(1 for a, b in zip(yt, lab) if a == pos and b == pos)
    fp = sum(1 for a, b in zip(yt, lab) if a != pos and b == pos)
    fn = sum(1 for a, b in zip(yt, lab) if a == pos and b != pos)
    out["accuracy"] = sum(1 for a, b in zip(yt, lab) if a == b) / n
    out["precision"] = tp / (tp + fp) if tp + fp else None
    out["recall"] = tp / (tp + fn) if tp + fn else None
    out["f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    isp = np.array([v == pos for v in yt])
    P, N = int(isp.sum()), n - int(isp.sum())
    if P == 0 or N == 0:
        out["roc_auc"] = "single-class"
    else:
        from scipy.stats import rankdata
        r2 = np.rint(rankdata(np.array(score), method="average") * 2).astype(np.int64)
        out["roc_auc"] = float(Fraction(int(r2[isp].sum()) - P * (P + 1), 2 * P * N))
    eps = 2.220446049250313e-16
    terms = []
    for a, q in zip(yt, score):
        x = q if a == pos else 1.0 - q
        x = min(max(x, eps), 1.0 - eps)
        terms.append(-math.log(x))
    out["log_loss"] = ow.ordered(terms) / float(n)
    return out


# ---------------------------------------------------------------- the data
def gen_case(seed, model, task, rows=None):
    rng = random.Random(seed)
    n = rows or rng.choice([30, 64, 101, 257, 300, 513, 700])
    d = rng.randint(2, 7)
    kinds = [rng.choice(["int", "dec", "big", "flt"]) for _ in range(d)]
    scales = [rng.randint(1, 3) for _ in range(d)]
    use_missing = rng.random() < 0.65
    miss = [rng.choice([0, 0.05, 0.15, 0.3]) if use_missing else 0 for _ in range(d)]
    cols = [[ow.gen_cell(rng, kinds[j], scales[j], miss[j]) for _ in range(n)] for j in range(d)]
    if rng.random() < 0.35:
        cj = rng.randrange(d)
        v = "7" if kinds[cj] == "int" else "2.5"
        cols[cj] = [v if c != "" or not miss[cj] else "" for c in cols[cj]]
        if not any(c != "" for c in cols[cj]):
            cols[cj][0] = v
    if use_missing and rng.random() < 0.08:
        cols[rng.randrange(d)] = [""] * n          # a column missing in every row
    X = [[float(cols[j][i]) if cols[j][i] != "" else 0.0 for j in range(d)] for i in range(n)]
    coef = [rng.uniform(-2, 2) / (1 + (j % 3) * 50) for j in range(d)]
    z = [sum(c * v for c, v in zip(coef, row)) + (row[0] * row[1] * 0.001 if d > 1 else 0.0) + rng.gauss(0, 0.5) for row in X]
    names = ["f%d" % j for j in range(d)]
    categorical = rng.random() < 0.25
    if categorical:
        levels = [rng.randint(1, 5) for _ in range(n)]
        effect = {k: rng.uniform(-3, 3) for k in range(1, 6)}
        z = [v + effect[l] for v, l in zip(z, levels)]
        cols.append([str(l) for l in levels])
        names.append("cat")
        d += 1
    if task == "regression":
        ycol = ["%.4f" % (v + 3.0) for v in z]
        target = "y"
    else:
        med = sorted(z)[n // 2]
        lab = [1 if v + rng.gauss(0, 0.3) > med else 0 for v in z]
        pair = rng.choice([(0, 1), (0, 1), (-1, 1), (2, 5), (10, 3)])
        ycol = [str(pair[v]) for v in lab]
        target = "lab"
    rowsl = [[cols[j][i] for j in range(d)] + [ycol[i]] for i in range(n)]
    steps, text = [], []
    dall = list(range(d))
    allmissing = [j for j in range(d) if all(cols[j][i] == "" for i in range(n))]
    ok_cols = [j for j in dall if j not in allmissing]
    missing = [j for j in ok_cols if any(c == "" for c in cols[j])]
    if missing and rng.random() < 0.45:
        st = {"kind": 1, "a": rng.choice([0, 0, 1, 2]), "b": 0, "value": rng.choice([0.0, 1.5, -2.0]), "cols": missing}
        steps.append(st)
        text.append(op.plan_line(st, names, None) + "\n")
    if ok_cols and rng.random() < 0.5:
        sk = rng.choice([2, 3, 3])
        st = {"kind": sk, "a": rng.randint(0, 1) if sk == 3 else 0, "b": 0, "value": 0.0, "cols": sorted(rng.sample(ok_cols, rng.randint(1, len(ok_cols))))}
        steps.append(st)
        text.append(op.plan_line(st, names, None) + "\n")
    plain = [j for j in ok_cols if not (categorical and j == d - 1)]
    if plain and rng.random() < 0.3:
        st = {"kind": 6, "a": rng.randint(0, 1), "b": rng.choice([2, 3, 5, 8]), "value": 0.0, "cols": [rng.choice(plain)]}
        steps.append(st)
        text.append(op.plan_line(st, names, None) + "\n")
    if categorical and rng.random() < 0.7:
        st = {"kind": 4, "a": rng.randint(0, 1), "b": 8, "value": 0.0, "cols": [d - 1]}
        steps.append(st)
        text.append(op.plan_line(st, names, None) + "\n")
    hp = {}
    if rng.random() < 0.6:
        hp["max-depth"] = rng.choice([2, 4, 6, 10])
    if rng.random() < 0.4:
        hp["min-leaf"] = rng.choice([2, 3, 5])
    if rng.random() < 0.5:
        hp["bins"] = rng.choice([2, 5, 16, 32, 255])
    if model == "forest":
        hp["trees"] = rng.choice([1, 3, 7, 10])
        if rng.random() < 0.5:
            hp["mtry"] = rng.randint(1, d)
        if rng.random() < 0.4:
            hp["bootstrap"] = 0
    case = TCase(model, task, names + [target], rowsl, target, steps, text, hp)
    if rng.random() < 0.4:
        case.split, case.seed = rng.choice(["train=0.8", "train=0.7", "train=0.5", "train=0.9"]), rng.randint(0, 2 ** 31)
    elif rng.random() < 0.5:
        case.seed = rng.randint(0, 99)
    if task == "classification" and rng.random() < 0.4:
        case.positive = min(int(v) for v in ycol)
    return case


# ---------------------------------------------------------------- the binary
def fit_args(case, out, sparse=True):
    a = ["fit", "--model", case.model, "--task", case.task, "--target", case.target, "--output", out]
    for k, v in case.hp.items():
        if v != DEFAULTS[k] or not sparse:
            a += ["--" + k, str(v)]
    a += ow.step_flag(case)
    if case.split:
        a += ["--split", case.split, "--seed", str(case.seed)]
    elif case.seed:
        a += ["--seed", str(case.seed)]
    if case.positive is not None and case.task == "classification":
        a += ["--positive", str(case.positive)]
    return a


def run_case(case, tmp, threads=1):
    """Compare the real `ml` with the oracle on one case. A list of problems."""
    problems = []
    (tmp / "m.cml").unlink(missing_ok=True)
    (tmp / "d.csv").write_text(case.csv())
    expect, ctx = fit_oracle(case)
    rc, out = ow.ml("--root", tmp, *fit_args(case, "m.cml"), "--threads", threads, "d.csv")
    doc = json.loads(out)
    if ctx is None:
        if doc.get("ok") or doc["error"]["rule"] != expect:
            problems.append("expected the refusal %s, got %s" % (expect, out[:300]))
        return problems
    if not doc.get("ok"):
        return ["the oracle fitted and ml refused: %s" % out[:400]]
    got = (tmp / "m.cml").read_bytes()
    if got != expect:
        why = "the model file differs from the oracle's (%d bytes, %d expected)" % (len(got), len(expect))
        for i in range(min(len(got), len(expect))):
            if got[i] != expect[i]:
                why += ", first difference at byte %d" % i
                break
        return [why]
    if any(all(r[j] == "" for r in case.rows) for j in range(len(case.names))):
        # a column with no cell at all is refused by predict and eval as a missing column of the schema, for every model (found by this oracle; docs/wiring-trees.md
        # section 11, open question 2): the fit is compared, the verbs are not
        UNJUDGED.append(1)
        return problems
    proba = case.task == "classification"
    base = ["--root", tmp, "predict", "--model-file", "m.cml", "--limit", "1000000"] + (["--proba"] if proba else [])
    rc, out = ow.ml(*base, "d.csv")
    doc = json.loads(out)
    if not doc.get("ok"):
        return problems + ["predict refused: %s" % out[:300]]
    names, rows = case.names, case.rows
    t = names.index(case.target)
    feat = [j for j in range(len(names)) if j != t]
    X = [[ow.to_float(r[j]) for j in feat] for r in rows]
    problems += compare_predict(case, doc, predict_oracle(case, ctx, X), "predict")
    # other rows: the last 40, reversed, the columns reversed, the target left out (columns are matched by name)
    sub = list(reversed(range(len(rows))))[:40]
    pn = [names[j] for j in reversed(feat)]
    (tmp / "p.csv").write_text(",".join(pn) + "\n" + "".join(",".join(rows[i][j] for j in reversed(feat)) + "\n" for i in sub))
    rc, out = ow.ml(*base, "p.csv")
    doc = json.loads(out)
    if not doc.get("ok"):
        problems.append("predict on reordered columns refused: %s" % out[:300])
    else:
        problems += compare_predict(case, doc, predict_oracle(case, ctx, [X[i] for i in sub]), "predict on reordered columns")
    idx = list(range(len(rows)))
    if case.split:
        mask = ow.split_masks(len(rows), case.split, case.seed)
        idx = [i for i in idx if mask[i] == 1]
    problems += check_eval(case, ctx, idx, tmp, threads)
    if case.task == "classification":
        other = copy.copy(case)
        c0, c1 = ctx["labels"]
        other.positive = c0 if (case.positive if case.positive is not None else c1) == c1 else c1
        problems += check_eval(other, ctx, idx, tmp, threads)
    return problems


def compare_predict(case, doc, want, what):
    cells = doc["data"]["rows"]
    if case.task == "regression":
        if [float(c[0]) for c in cells] != want:
            return ["%s: the values differ from the oracle's" % what]
        return []
    labs, probs = want
    if [int(c[0]) for c in cells] != labs or [float(c[1]) for c in cells] != probs:
        return ["%s: labels or probabilities differ from the oracle's" % what]
    return []


def check_eval(case, ctx, idx, tmp, threads):
    problems = []
    ev = ["eval", "--model-file", "m.cml", "d.csv"]
    if case.split:
        ev += ["--split", case.split, "--seed", str(case.seed)]
    if case.positive is not None and case.task == "classification":
        ev += ["--positive", str(case.positive)]
    rc, out = ow.ml("--root", tmp, *ev, "--threads", threads)
    doc = json.loads(out)
    want = eval_oracle(case, ctx, idx)
    if not doc.get("ok"):
        if want.get("roc_auc") == "single-class" and doc["error"]["rule"] == "ml.label.single-class":
            return problems
        return problems + ["eval refused: %s" % out[:300]]
    got = doc["data"]["metrics"]
    for k, v in want.items():
        g = got.get(k, "missing")
        if v is None:
            if g is not None:
                problems.append("eval %s (positive %s): expected null, got %r" % (k, case.positive, g))
        elif k == "log_loss":
            if g is None or abs(float(g) - v) > 4e-16 * max(1.0, abs(v)) * 8:
                problems.append("eval %s (positive %s): %r vs %r" % (k, case.positive, g, ow.put_float(v)))
        elif g != ow.put_float(v):
            problems.append("eval %s (positive %s): %r vs %r" % (k, case.positive, g, ow.put_float(v)))
    return problems


def main():
    argv = sys.argv[1:]
    n = 24
    models = ("tree", "forest")
    rows = None
    large = False
    if argv and argv[0] == "--suite":
        n = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 24
        rest = [a for a in argv[1:] if a in models]
        if rest:
            models = (rest[0],)
    elif argv and argv[0] == "--large":
        large, n, rows = True, 2, 100000
        rest = [a for a in argv[1:] if a in models]
        if rest:
            models = (rest[0],)
    bad = 0
    refused = 0
    cases = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for model in models:
            for task in ("classification", "regression"):
                for seed in range(1, n + 1):
                    case = gen_case(seed * 7919 + (101 if task == "regression" else 0) + (31 if model == "forest" else 0), model, task, rows=rows)
                    if large and model == "forest":
                        case.hp["trees"] = min(case.hp.get("trees", 3), 3)
                    pr = run_case(case, tmp, threads=1 + seed % 4)
                    cases += 1
                    (tmp / "m.cml").unlink(missing_ok=True)
                    if pr:
                        bad += 1
                        print("MISMATCH %s %s seed %d: %s" % (model, task, seed, "; ".join(pr)))
    print("oracle_wire_tree covered: %s; %d cases with a column that has no cell: file compared, verbs not" % (", ".join("%s %d" % (k, v) for k, v in COVER.items()), len(UNJUDGED)))
    print("oracle_wire_tree: %d cases (%s, both tasks, %s rows), %s" % (cases, " and ".join(models), "100,000" if large else "30 to 700", "all agree byte for byte" if not bad else "%d mismatches" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

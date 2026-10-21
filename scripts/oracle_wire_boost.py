#!/usr/bin/env python3
"""The oracle of the wiring of the boosted trees (docs/models-boost.md gates GB1, GB2, GB3): a Python implementation that fits the same pipeline and the same
model as `ml fit --model boost` and writes the model file's bytes, so that the file the real binary wrote is compared byte for byte.

What it reuses, and why that is not circular: the library part is `oracle_boost.py` (the algorithm of docs/models-boost.md section 2 and the payload of section 5 in
Python, with no cancho code, `exp` and `log` ported from std/math.cho); the case generator, the CSV types, the schema, the step records, the meta section and the
metrics are the ones the tree and linear oracles use (`oracle_wire_tree.py`, `oracle_wire.py`, `oracle_pipe.py`, `oracle_metrics.py`). What is written here is the glue
of this model: the plan line, the task, the labels record, the sum order 3, the scores and the class and probability of a score, and the metrics of a boosted model.

    python3 scripts/oracle_wire_boost.py --suite [N]     N seeded cases per task (default 24) against build/ml, 30 to 700 rows
    python3 scripts/oracle_wire_boost.py --large         two cases per task of 100,000 rows each, at 1, 2, 4, 8, 16 threads (several minutes)
    python3 scripts/oracle_wire_boost.py boost           what scripts/admit.py Q9 runs (the suite with a fixed small N)
Exit 1 on any mismatch. ML_BIN names the binary.
"""

import copy
import json
import math
import pathlib
import random
import sys
import tempfile
import warnings
from fractions import Fraction

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_boost as ob  # noqa: E402
import oracle_pipe as op  # noqa: E402
import oracle_wire as ow  # noqa: E402
import oracle_wire_tree as owt  # noqa: E402

DEFAULTS = {"rounds": 100, "learning-rate": "0.1", "max-depth": 4, "min-leaf": 10, "bins": 64, "l2": "1.0", "subsample": "1.0", "early-stopping": 0, "max-nodes": 4194304}
ORDER = ["rounds", "learning-rate", "max-depth", "min-leaf", "bins", "l2", "subsample", "early-stopping", "max-nodes"]
COVER = {"cases": 0, "with_missing": 0, "missing_cells": 0, "v1_splits": 0, "nodes": 0, "trees": 0, "split": 0, "steps": 0, "empty_columns": 0, "subsample": 0, "early": 0,
         "early_cut": 0, "refusals": 0}


class BCase(owt.TCase):
    def model_line(self):
        parts = ["model", "kind=boost", "task=" + self.task]
        for k in ORDER:
            parts.append("%s=%s" % (k, self.hp.get(k, DEFAULTS[k])))
        return " ".join(parts)


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
            owt.through_steps(stored, [X[i] for i in range(n) if mask[i] == 1])
        except op.Refuse as r:
            return r.tag, None
    Xn, Mn = owt.matrix_of(cur)
    d = Xn.shape[1]
    task = 0 if case.task == "classification" else 1
    hp = {k: case.hp.get(k, DEFAULTS[k]) for k in ORDER}
    stats = {}
    yv = np.array(yfit, dtype=np.float64)
    if task == 0 and (yv.sum() == 0 or yv.sum() == len(yv)):
        return "ml.label.single-class", None
    wts = None
    if case.weights is not None:
        # the first row of a class with no weight above 0 on the training side gets weight 1 (the refusal of such a side is judged by the tests of the command line)
        w = list(case.weights)
        for v in ((0.0, 1.0) if task == 0 else (None,)):
            g = [i for i in range(len(tr)) if v is None or yfit[i] == v]
            if g and not any(w[tr[i]] > 0 for i in g):
                w[tr[g[0]]] = 1.0
        case.weights = w
        wts = np.array([w[i] for i in tr], dtype=np.float64)
    try:
        res = ob.fit(Xn, Mn, yv, task, int(hp["rounds"]), float(hp["learning-rate"]), int(hp["max-depth"]), int(hp["min-leaf"]), int(hp["bins"]), float(hp["l2"]),
                     float(hp["subsample"]), int(hp["early-stopping"]), int(hp["max-nodes"]), case.seed, stats, wts)
    except ob.LimitNodes:
        return "ml.limit.nodes", None
    except AssertionError:
        return "unjudged", None
    model = ob.payload(res, d)
    if labels:
        model += owt.labels_record(*labels)
    plan = ow.canon(case.plan_text())
    data = owt.file_bytes(plan, ow.schema_bytes(names, types, t), ow.steps_bytes(stored), model, len(tr), len(names), case.seed,
                         ow.provenance_of(case, [names[j] for j in range(len(names)) if j != t], mask))
    ctx = {"stored": stored, "res": res, "d": d, "labels": labels, "types": types, "mask": mask, "stats": stats, "missing": int(Mn.sum())}
    COVER["cases"] += 1
    COVER["with_missing"] += 1 if ctx["missing"] else 0
    COVER["missing_cells"] += ctx["missing"]
    COVER["v1_splits"] += stats.get("v1", 0)
    COVER["nodes"] += sum(len(tr_) for tr_ in res["trees"])
    COVER["trees"] += len(res["trees"])
    COVER["split"] += 1 if mask is not None else 0
    COVER["steps"] += 1 if stored else 0
    COVER["subsample"] += 1 if float(hp["subsample"]) < 1.0 else 0
    COVER["early"] += 1 if int(hp["early-stopping"]) > 0 else 0
    COVER["early_cut"] += 1 if res["kept"] < int(hp["rounds"]) else 0
    COVER["empty_columns"] += 1 if any(all(r[j] == "" for r in rows) for j in range(len(names))) else 0
    return data, ctx


def scores_oracle(ctx, X_rows):
    cur = owt.through_steps(ctx["stored"], X_rows)
    Xn, Mn = owt.matrix_of(cur)
    return [float(v) for v in ob.predict_scores(ctx["res"], Xn, Mn)]


def predict_oracle(case, ctx, X_rows):
    s = scores_oracle(ctx, X_rows)
    if case.task == "regression":
        return s
    c0, c1 = ctx["labels"]
    p = [float(v) for v in ob.sigmoid(np.array(s))]
    return [c1 if q > 0.5 else c0 for q in p], p


def eval_oracle(case, ctx, rows_idx):
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
def gen_case(seed, task, rows=None, weighted=False):
    """the generator of the tree oracle (missing cells, constant and empty columns, int/dec/big/float columns, an ordinal category, impute, scale, bucket and encode
    steps, a split, --positive), with the hyperparameters of a boosted model"""
    base = owt.gen_case(seed, "tree", task, rows=rows)
    rng = random.Random(seed * 31 + 7)
    hp = {}
    big = rows is not None and rows >= 50000
    if rng.random() < 0.8:
        hp["rounds"] = rng.choice([1, 2, 5, 12, 30, 50] if not big else [5, 10, 20])
    if rng.random() < 0.6:
        hp["learning-rate"] = rng.choice(["0.05", "0.3", "1.0", "0.1", "0.5"])
    if rng.random() < 0.7:
        hp["max-depth"] = rng.choice([1, 2, 3, 5, 8] if not big else [3, 6, 8])
    if rng.random() < 0.5:
        hp["min-leaf"] = rng.choice([1, 2, 5, 20])
    if rng.random() < 0.5:
        hp["bins"] = rng.choice([2, 5, 16, 32, 255])
    if rng.random() < 0.5:
        hp["l2"] = rng.choice(["0.0", "0.5", "1.0", "10.0"])
    if rng.random() < 0.35:
        hp["subsample"] = rng.choice(["0.5", "0.8", "0.25", "0.999999"])
    if rng.random() < 0.15:
        hp["max-nodes"] = rng.choice([3, 7, 15, 1000])
    case = BCase("boost", task, base.names, base.rows, base.target, base.steps, base.step_text, hp)
    case.split, case.seed, case.positive = base.split, base.seed, base.positive
    if rng.random() < 0.25:
        if not case.split:
            case.split = rng.choice(["train=0.8", "train=0.9"])
            case.seed = rng.randint(0, 99)
        hp["early-stopping"] = rng.choice([1, 2, 5])
        hp["rounds"] = max(hp.get("rounds", 100), 30)
        if len(case.rows) * float(case.split[6:]) < 20:
            hp.pop("early-stopping")
    if weighted:
        wr = random.Random(seed ^ 0x5EED)
        case.weights = [wr.choice(owt.WEIGHTS) for _ in range(len(case.rows))]
    return case


# ---------------------------------------------------------------- the binary
def fit_args(case, out, sparse=True):
    a = ["fit", "--model", "boost", "--task", case.task, "--target", case.target, "--output", out]
    if case.weights is not None:
        a += ["--weight", "wt"]
    for k, v in case.hp.items():
        if str(v) != str(DEFAULTS[k]) or not sparse:
            a += ["--" + k, str(v)]
    a += ow.step_flag(case)
    if case.split:
        a += ["--split", case.split, "--seed", str(case.seed)]
    elif case.seed:
        a += ["--seed", str(case.seed)]
    if case.positive is not None and case.task == "classification":
        a += ["--positive", str(case.positive)]
    return a


def run_case(case, tmp, threads=1, all_threads=False):
    """Compare the real `ml` with the oracle on one case. A list of problems."""
    problems = []
    (tmp / "m.cml").unlink(missing_ok=True)
    expect, ctx = fit_oracle(case)
    (tmp / "d.csv").write_text(case.csv())
    rc, out = ow.ml("--root", tmp, *fit_args(case, "m.cml"), "--threads", threads, "d.csv")
    doc = json.loads(out)
    if ctx is None:
        if expect == "unjudged":
            UNJUDGED.append(1)
            return []
        COVER["refusals"] += 1
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
    if all_threads:
        for t in (1, 2, 4, 8, 16):
            (tmp / "th.cml").unlink(missing_ok=True)
            rc, out = ow.ml("--root", tmp, *fit_args(case, "th.cml"), "--threads", t, "d.csv")
            if rc != 0 or (tmp / "th.cml").read_bytes() != expect:
                problems.append("the file at --threads %d differs from the oracle's" % t)
    proba = case.task == "classification"
    base = ["--root", tmp, "predict", "--model-file", "m.cml", "--limit", "1000000"] + (["--proba"] if proba else [])
    rc, out = ow.ml(*base, *(["--drop", "wt"] if case.weights is not None else []), "d.csv")
    doc = json.loads(out)
    if not doc.get("ok"):
        return problems + ["predict refused: %s" % out[:300]]
    names, rows = case.names, case.rows
    t = names.index(case.target)
    feat = [j for j in range(len(names)) if j != t]
    X = [[ow.to_float(r[j]) for j in feat] for r in rows]
    problems += owt.compare_predict(case, doc, predict_oracle(case, ctx, X), "predict")
    sub = list(reversed(range(len(rows))))[:40]
    pn = [names[j] for j in reversed(feat)]
    (tmp / "p.csv").write_text(",".join(pn) + "\n" + "".join(",".join(rows[i][j] for j in reversed(feat)) + "\n" for i in sub))
    rc, out = ow.ml(*base, "p.csv")
    doc = json.loads(out)
    if not doc.get("ok"):
        problems.append("predict on reordered columns refused: %s" % out[:300])
    else:
        problems += owt.compare_predict(case, doc, predict_oracle(case, ctx, [X[i] for i in sub]), "predict on reordered columns")
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


UNJUDGED = []


def check_eval(case, ctx, idx, tmp, threads):
    problems = []
    ev = ["eval", "--model-file", "m.cml", *(["--drop", "wt"] if case.weights is not None else []), "d.csv"]
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
    rows = None
    large = False
    if argv and argv[0] == "--suite":
        n = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 24
    elif argv and argv[0] == "--large":
        large, n, rows = True, 2, 100000
    elif argv and argv[0] == "boost":
        n = 6
    weighted = "--weighted" in argv
    bad = 0
    cases = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for task in ("classification", "regression"):
            for seed in range(1, n + 1):
                case = gen_case(seed * 7919 + (101 if task == "regression" else 0) + 57, task, rows=rows, weighted=weighted)
                pr = run_case(case, tmp, threads=1 + seed % 4, all_threads=large)
                cases += 1
                (tmp / "m.cml").unlink(missing_ok=True)
                if pr:
                    bad += 1
                    print("MISMATCH boost %s seed %d: %s" % (task, seed, "; ".join(pr)))
    print("oracle_wire_boost covered: %s, not judged %d" % (", ".join("%s %d" % (k, v) for k, v in COVER.items()), len(UNJUDGED)))
    print("oracle_wire_boost: %d cases (both tasks, %s rows), %s" % (cases, "100,000" if large else "30 to 700", "all agree byte for byte" if not bad else "%d mismatches" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

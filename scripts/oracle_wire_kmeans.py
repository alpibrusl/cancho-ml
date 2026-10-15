#!/usr/bin/env python3
"""The oracle of the wiring of k-means (docs/wiring-kmeans.md gates WK3, WK4, WK5, WK6): a Python implementation that fits the same pipeline and the same model as
`ml fit --model kmeans` and writes the model file's bytes, so that the file the real binary wrote is compared byte for byte.

What it reuses, and why that is not circular: the library part is `oracle_models.py` (`kmeans`: the initialisation from the stream of kind 3, the Lloyd sums in the
package's lane order, the empty-cluster rule, written in numpy and compared bit for bit with `lib/models/kmeans` by `gate_models.sh`); the transforms and their
application are `oracle_pipe.py`'s (compared with the library by `gate_pipe.sh`); the CSV types, the step records, the sections and the meta are `oracle_wire.py`'s; the
split is `oracle_metrics.py`'s. What is written here is the glue of this stage: a schema with no target (the target index is the number of columns), the payload of
kind 6, the refusal of a cell still missing, the predictions (cluster, squared distance) and the metrics (inertia, sizes, mean distances) from the definitions of
docs/wiring-kmeans.md section 3, each sum through `oracle_metrics.ordered_sum`, the package's order.

    python3 scripts/oracle_wire_kmeans.py --suite [N]     N seeded cases per initialisation against build/ml (default 30), 30 to 700 rows
    python3 scripts/oracle_wire_kmeans.py --large         two cases of 100,000 rows per initialisation, the file also at 1, 2, 4, 8 and 16 threads (a few minutes)
Exit 1 on any mismatch. ML_BIN names the binary.
"""

import json
import math
import os
import pathlib
import random
import struct
import sys
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_metrics as om_  # noqa: E402
import oracle_models as omod  # noqa: E402
import oracle_pipe as op  # noqa: E402
import oracle_wire as ow  # noqa: E402

KIND = 6
INITS = ("random", "kmeans++")
COVER = {"cases": 0, "with_missing": 0, "missing_cells": 0, "split": 0, "steps": 0, "label_dropped": 0, "empty_clusters": 0, "refused": 0}


class KCase:
    """A CSV, a k-means and a plan: everything `ml fit` is given. `extra` is the values of a label column that the fit drops."""

    def __init__(self, names, rows, extra, steps, step_text, k, iters, init, split=None, seed=0, init_flag=True):
        self.names, self.rows, self.extra = names, rows, extra
        self.steps, self.step_text = steps, step_text
        self.k, self.iters, self.init, self.split, self.seed, self.init_flag = k, iters, init, split, seed, init_flag

    def header(self):
        return self.names + (["lab"] if self.extra is not None else [])

    def csv(self):
        out = ",".join(self.header()) + "\n"
        for i, r in enumerate(self.rows):
            out += ",".join(r + ([self.extra[i]] if self.extra is not None else [])) + "\n"
        return out

    def model_line(self):
        return "model kind=kmeans k=%d iterations=%d init=%s" % (self.k, self.iters, self.init)

    def plan_text(self):
        return "".join(self.step_text) + self.model_line()


def payload(cen):
    k, d = cen.shape
    return struct.pack("<I", KIND) + struct.pack("<IIQQ", 2, 2, k, d) + struct.pack("<%dd" % (k * d), *[float(v) for v in cen.reshape(-1)])


def file_bytes(plan, schema, steps, model, rows, ncols, seed, prov=None):
    return ow.file_bytes(plan, schema, steps, model, rows, ncols, seed, prov)


def matrix_of(rows_after):
    return np.array([[np.nan if v is None else v for v in row] for row in rows_after], dtype=np.float64).reshape(len(rows_after), -1)


def fit_oracle(case):
    """(file bytes, or the tag of the refusal) and a context for predict and eval"""
    names, rows = case.names, case.rows
    types = [ow.column_type([r[j] for r in rows]) for j in range(len(names))]
    X = [[ow.to_float(r[j]) for j in range(len(names))] for r in rows]
    n = len(rows)
    mask = ow.split_masks(n, case.split, case.seed)
    tr = [i for i in range(n) if mask is None or mask[i] == 0]
    try:
        stored, cur, _, _, _ = op.ref_pipeline(case.steps, names, [X[i] for i in tr], [])
    except op.Refuse as r:
        return r.tag, None
    if any(v is None for row in cur for v in row):
        return "ml.input.missing-value", None
    if mask is not None:
        try:
            held = through_steps(stored, [X[i] for i in range(n) if mask[i] == 1])
        except op.Refuse as r:
            return r.tag, None
        if any(v is None for row in held for v in row):
            return "ml.input.missing-value", None
    d = len(cur[0])
    ntr = len(tr)
    if case.k > ntr or case.k * d + case.k + 1 > 131072:
        return "ml.param.out-of-range", None
    Xn = np.array(cur, dtype=np.float64).reshape(ntr, d)
    init = 0 if case.init == "random" else 1
    cen, diag, lab, D = omod.kmeans(Xn, case.k, case.iters, case.seed, init)
    plan = ow.canon(case.plan_text())
    # the schema: the kept columns and no target; the target index is the number of columns
    data = file_bytes(plan, ow.schema_bytes(names, types, len(names)), ow.steps_bytes(stored), payload(cen), ntr, len(names), case.seed, ow.provenance_of(case, list(names), mask))
    ctx = {"stored": stored, "cen": cen, "d": d, "types": types, "mask": mask, "Xn": Xn, "init_rows": omod.chosen_rows(Xn, case.k, case.seed, init)}
    COVER["cases"] += 1
    COVER["with_missing"] += 1 if any(c == "" for r in rows for c in r) else 0
    COVER["missing_cells"] += sum(1 for r in rows for c in r if c == "")
    COVER["split"] += 1 if mask is not None else 0
    COVER["steps"] += 1 if stored else 0
    COVER["label_dropped"] += 1 if case.extra is not None else 0
    cnt = np.bincount(lab, minlength=case.k)
    COVER["empty_clusters"] += int((cnt == 0).sum())
    return data, ctx


def through_steps(stored, X):
    cur = X
    for kind, ints, fl in stored:
        cur, _u = op.ref_apply(kind, ints, fl, cur)
    return cur


def assign_rows(ctx, X_rows):
    """(clusters, squared distances) of the rows for the fitted centroids, through the stored steps"""
    cur = through_steps(ctx["stored"], X_rows)
    Xn = matrix_of(cur)
    lab, mn, _ = omod.assign(Xn, ctx["cen"])
    return [int(v) for v in lab], [float(v) for v in mn]


def metrics_of(ctx, k, lab, d2):
    """what `ml eval` prints: the inertia, the rows of each cluster and the mean Euclidean distance of each cluster (None for an empty one)"""
    inertia = ow.ordered(d2)
    sizes = [sum(1 for c in lab if c == j) for j in range(k)]
    means = []
    for j in range(k):
        dist = [math.sqrt(v) for c, v in zip(lab, d2) if c == j]
        means.append(ow.ordered(dist) / float(len(dist)) if dist else None)
    return {"inertia": inertia, "sizes": sizes, "mean_distance": means}


def eval_oracle(case, ctx, rows_idx):
    X = [[ow.to_float(case.rows[i][j]) for j in range(len(case.names))] for i in rows_idx]
    lab, d2 = assign_rows(ctx, X)
    return metrics_of(ctx, case.k, lab, d2)


# ---------------------------------------------------------------- the data
def gen_case(seed, rows=None, force_impute=False):
    rng = random.Random(seed)
    n = rows or rng.choice([30, 64, 101, 257, 300, 513, 700])
    d = rng.randint(1, 6)
    kinds = [rng.choice(["int", "dec", "big", "flt"]) for _ in range(d)]
    scales = [rng.randint(1, 3) for _ in range(d)]
    use_missing = rng.random() < 0.55
    if d == 1:
        use_missing = False          # a missing cell of a one-column file is a blank line, which the reader skips
    miss = [rng.choice([0, 0.05, 0.15, 0.3]) if use_missing else 0 for _ in range(d)]
    # clustered data: each row belongs to one of a few groups whose centres differ, so that the clusters are not empty or all the same
    groups = rng.randint(1, 5)
    g = [rng.randrange(groups) for _ in range(n)]
    cols = []
    for j in range(d):
        col = []
        for i in range(n):
            c = ow.gen_cell(rng, kinds[j], scales[j], miss[j])
            if c != "" and kinds[j] in ("int", "dec"):
                v = float(c) + g[i] * rng.choice([30.0, 60.0, 90.0])
                c = str(int(v)) if kinds[j] == "int" else "%.*f" % (scales[j], v)
            col.append(c)
        cols.append(col)
    if rng.random() < 0.3:
        cj = rng.randrange(d)
        v = "7" if kinds[cj] == "int" else "2.5"
        cols[cj] = [v if c != "" or not miss[cj] else "" for c in cols[cj]]
        if not any(c != "" for c in cols[cj]):
            cols[cj][0] = v
    names = ["f%d" % j for j in range(d)]
    ok_cols = [j for j in range(d) if any(c != "" for c in cols[j])]
    for j in range(d):
        if j not in ok_cols:
            cols[j][0] = "1"
            ok_cols.append(j)
    missing = [j for j in ok_cols if any(c == "" for c in cols[j])]
    steps, text = [], []
    if missing and (force_impute or rng.random() < 0.93):
        st = {"kind": 1, "a": rng.choice([0, 0, 1, 2]), "b": 0, "value": rng.choice([0.0, 1.5, -2.0]), "cols": missing}
        steps.append(st)
        text.append(op.plan_line(st, names, None) + "\n")
    if rng.random() < 0.7:
        sk = rng.choice([2, 3, 3])
        st = {"kind": sk, "a": rng.randint(0, 1) if sk == 3 else 0, "b": 0, "value": 0.0, "cols": sorted(rng.sample(ok_cols, rng.randint(1, len(ok_cols))))}
        steps.append(st)
        text.append(op.plan_line(st, names, None) + "\n")
    if ok_cols and rng.random() < 0.2:
        st = {"kind": 6, "a": rng.randint(0, 1), "b": rng.choice([2, 3, 5, 8]), "value": 0.0, "cols": [rng.choice(ok_cols)]}
        steps.append(st)
        text.append(op.plan_line(st, names, None) + "\n")
    rowsl = [[cols[j][i] for j in range(d)] for i in range(n)]
    extra = [str(g[i] % 2) for i in range(n)] if rng.random() < 0.4 else None
    k = rng.randint(1, 12)
    iters = rng.choice([1, 2, 5, 10, 10, 10, 20])
    init = rng.choice(INITS)
    case = KCase(names, rowsl, extra, steps, text, k, iters, init)
    if rng.random() < 0.4:
        case.split, case.seed = rng.choice(["train=0.8", "train=0.7", "train=0.5", "train=0.9"]), rng.randint(0, 2 ** 31)
    elif rng.random() < 0.5:
        case.seed = rng.randint(0, 99)
    case.init_flag = not (init == "kmeans++" and iters == 10 and rng.random() < 0.5)
    return case


# ---------------------------------------------------------------- the binary
def fit_args(case, out):
    a = ["fit", "--model", "kmeans", "--k", str(case.k), "--output", out]
    if case.iters != 10 or case.init_flag:
        a += ["--iterations", str(case.iters)]
    if case.init_flag:
        a += ["--init", case.init]
    if case.extra is not None:
        a += ["--drop", "lab"]
    a += ow.step_flag(case)
    if case.split:
        a += ["--split", case.split, "--seed", str(case.seed)]
    elif case.seed:
        a += ["--seed", str(case.seed)]
    return a


def run_case(case, tmp, threads=1, all_threads=False):
    """Compare the real `ml` with the oracle on one case. A list of problems."""
    problems = []
    (tmp / "m.cml").unlink(missing_ok=True)
    (tmp / "d.csv").write_text(case.csv())
    expect, ctx = fit_oracle(case)
    rc, out = ow.ml("--root", tmp, *fit_args(case, "m.cml"), "--threads", threads, "d.csv")
    doc = json.loads(out)
    if ctx is None:
        COVER["refused"] += 1
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
            rc2, out2 = ow.ml("--root", tmp, *fit_args(case, "th.cml"), "--threads", t, "d.csv")
            if rc2 != 0 or (tmp / "th.cml").read_bytes() != expect:
                problems.append("the file at --threads %d differs from the oracle's" % t)
    names, rows = case.names, case.rows
    # the answer of the fit: the metrics of the training rows, and of the held-out rows
    mask = ctx["mask"]
    n = len(rows)
    tr_idx = [i for i in range(n) if mask is None or mask[i] == 0]
    s = doc["data"]["summary"]
    problems += compare_metrics("fit training", s["clusters"]["metrics"], eval_oracle(case, ctx, tr_idx))
    if mask is not None:
        te_idx = [i for i in range(n) if mask[i] == 1]
        problems += compare_metrics("fit holdout", s["holdout"]["metrics"], eval_oracle(case, ctx, te_idx))
    # predict: every row, with the distance; the label column is dropped like in the fit
    drop = ["--drop", "lab"] if case.extra is not None else []
    base = ["--root", tmp, "predict", "--model-file", "m.cml", "--limit", "1000000", "--distance"] + drop
    rc, out = ow.ml(*base, "d.csv")
    pdoc = json.loads(out)
    if not pdoc.get("ok"):
        return problems + ["predict refused: %s" % out[:300]]
    X = [[ow.to_float(r[j]) for j in range(len(names))] for r in rows]
    problems += compare_predict(pdoc, assign_rows(ctx, X), "predict")
    # the same rows with one cluster column only
    rc, out = ow.ml("--root", tmp, "predict", "--model-file", "m.cml", "--limit", "1000000", *drop, "d.csv")
    p1 = json.loads(out)
    if not p1.get("ok") or p1["data"]["columns"] != ["cluster"] or [int(c[0]) for c in p1["data"]["rows"]] != assign_rows(ctx, X)[0]:
        problems.append("predict without --distance: %s" % out[:200])
    # other rows: the last 40, reversed, the columns reversed, the label column left out (columns are matched by name)
    sub = list(reversed(range(len(rows))))[:40]
    pn = list(reversed(names))
    (tmp / "p.csv").write_text(",".join(pn) + "\n" + "".join(",".join(rows[i][j] for j in reversed(range(len(names)))) + "\n" for i in sub))
    rc, out = ow.ml("--root", tmp, "predict", "--model-file", "m.cml", "--limit", "1000000", "--distance", "p.csv")
    q = json.loads(out)
    if not q.get("ok"):
        problems.append("predict on reordered columns refused: %s" % out[:300])
    else:
        problems += compare_predict(q, assign_rows(ctx, [X[i] for i in sub]), "predict on reordered columns")
    # eval: on all rows, or on the test side of the split
    idx = list(range(n))
    ev = ["eval", "--model-file", "m.cml", "d.csv"] + drop
    if case.split:
        idx = [i for i in idx if mask is not None and mask[i] == 1] if mask is not None else idx
        ev += ["--split", case.split, "--seed", str(case.seed)]
    else:
        pass
    rc, out = ow.ml("--root", tmp, *ev, "--threads", threads)
    edoc = json.loads(out)
    if not edoc.get("ok"):
        return problems + ["eval refused: %s" % out[:300]]
    if edoc["data"]["clusters"] != case.k or edoc["data"]["rows"] != len(idx):
        problems.append("eval: clusters or rows differ")
    problems += compare_metrics("eval", edoc["data"]["metrics"], eval_oracle(case, ctx, idx))
    return problems


def compare_predict(doc, want, what):
    cells = doc["data"]["rows"]
    lab, d2 = want
    if doc["data"]["columns"] != ["cluster", "distance"]:
        return ["%s: the columns are %r" % (what, doc["data"]["columns"])]
    if [int(c[0]) for c in cells] != lab:
        return ["%s: the clusters differ from the oracle's" % what]
    # compared as doubles: at 17 digits the shortest decimal that reads back is not unique and cancho-table's printer may end on another digit than Python's
    if [float(c[1]) for c in cells] != d2:
        return ["%s: the distances differ from the oracle's" % what]
    return []


def compare_metrics(what, got, want):
    problems = []
    if float(got["inertia"]) != want["inertia"]:
        problems.append("%s inertia: %r vs %r" % (what, got["inertia"], ow.put_float(want["inertia"])))
    if got["sizes"] != want["sizes"]:
        problems.append("%s sizes: %r vs %r" % (what, got["sizes"], want["sizes"]))
    gm = [None if v is None else float(v) for v in got["mean_distance"]]
    if gm != want["mean_distance"]:
        problems.append("%s mean_distance: %r vs %r" % (what, got["mean_distance"], want["mean_distance"]))
    return problems


def main():
    argv = sys.argv[1:]
    n = 30
    rows = None
    large = False
    if argv and argv[0] == "--suite":
        n = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 30
    elif argv and argv[0] == "--large":
        large, n, rows = True, 2, 100000
    bad = 0
    cases = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for init in INITS:
            for seed in range(1, n + 1):
                case = gen_case(seed * 7919 + (101 if init == "random" else 0), rows=rows, force_impute=large)
                case.init = init
                case.init_flag = True
                if large:
                    case.k = 2 + (seed * 5) % 11
                    case.iters = 10
                    case.split = None
                pr = run_case(case, tmp, threads=1 + seed % 4, all_threads=large)
                cases += 1
                (tmp / "m.cml").unlink(missing_ok=True)
                if pr:
                    bad += 1
                    print("MISMATCH %s seed %d: %s" % (init, seed, "; ".join(pr)))
    print("oracle_wire_kmeans covered: %s" % ", ".join("%s %d" % (k, v) for k, v in COVER.items()))
    print("oracle_wire_kmeans: %d cases (both initialisations, k 1 to 12, %s rows), %s" % (cases, "100,000" if large else "30 to 700", "all agree byte for byte" if not bad else "%d mismatches" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

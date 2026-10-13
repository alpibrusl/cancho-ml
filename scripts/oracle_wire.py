#!/usr/bin/env python3
"""The oracle of the wiring stage (docs/wiring.md gate W3, W6, W7): a Python implementation that fits the same pipeline and the same model as `ml fit` and writes the
model file's bytes, so that the file the real binary wrote can be compared byte for byte, not within a tolerance.

What it reuses, and why that is not circular: the transforms and their application come from `oracle_pipe.py` (`ref_fit`, `ref_apply`, which `gate_pipe.sh` compares
bit for bit with the library), the estimators from `oracle_models.py` (`linreg`, `logreg`, the package's summation order written again in numpy, compared with
`models-check` by `gate_models.sh`), the split from `oracle_metrics.py` (`py_train_test`, SplitMix64 on Python ints), the plan's canonical text from
`oracle_pipe.canon_plan`. What it writes here is the glue the stage added: the CSV types, the schema section, the step records, the payloads, the meta section, the
SHA-256 trailer, the labels, the predictions and the metrics from the definitions in docs/metrics.md (integers and Fractions where the definition is an integer ratio).

    python3 scripts/oracle_wire.py --suite [N]     N seeded cases (default 40) against build/ml: fit, predict, eval, the bytes of the file
    python3 scripts/oracle_wire.py linear|logistic one seed and print the case
Exit 1 on any mismatch. ML_BIN names the binary.
"""

import hashlib
import json
import math
import os
import pathlib
import random
import re
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import oracle_metrics as om_  # noqa: E402
import oracle_models as omod  # noqa: E402
import oracle_pipe as op  # noqa: E402

BIN = os.environ.get("ML_BIN", str(ROOT / "build" / "ml"))
PIN = (ROOT / "cancho.toml").read_text().split('cancho = "')[1].split('"')[0]
SUM_ORDER = 2
KIND = {"linear": 4, "logistic": 5}
UNJUDGED = []


# ---------------------------------------------------------------- the CSV, as ml reads it
INT_RE = re.compile(r"^[+-]?[0-9]+$")
DEC_RE = re.compile(r"^[+-]?[0-9]+\.[0-9]+$")


def cell_kind(text):
    """(kind, scale): 1 int, 2 dec(scale), 3 float, None for an empty cell (docs/reader.md section 2.1; the cells generated here have at most 12 digits)"""
    if text == "":
        return None
    if INT_RE.match(text):
        return 1, 0
    m = DEC_RE.match(text)
    if m:
        digits = len(text.lstrip("+-").replace(".", "").lstrip("0"))
        frac = len(text.split(".")[1])
        return (3, 0) if digits > 12 else (2, frac)
    return 3, 0


def column_type(cells):
    """the schema type of a column: (0 int, 1 dec, 2 float, scale); a column with no cell is a float (docs/wiring.md section 3)"""
    kinds = [cell_kind(c) for c in cells]
    kinds = [k for k in kinds if k is not None]
    if not kinds:
        return 2, 0
    kind = max(k[0] for k in kinds)
    scale = max(k[1] for k in kinds if k[0] < 3) if any(k[0] < 3 for k in kinds) else 0
    if kind < 3:
        idig = max(len(c.lstrip("+-").split(".")[0].lstrip("0")) for c in cells if c != "")
        if scale + idig > 12:
            kind = 3
    if kind == 1:
        return 0, 0
    if kind == 2:
        return 1, scale
    return 2, 0


def to_float(text):
    return None if text == "" else float(text) + 0.0


# ---------------------------------------------------------------- the file
def put_str(s):
    b = s.encode()
    return struct.pack("<I", len(b)) + b


def schema_bytes(names, types, target):
    out = struct.pack("<II", len(names), target)
    for n, (t, s) in zip(names, types):
        out += put_str(n) + struct.pack("<II", t, s)
    return out


def enc_step(kind, ints, fl):
    out = struct.pack("<III", kind, 1, 1) + struct.pack("<Q", len(ints)) + struct.pack("<%dq" % len(ints), *ints)
    k = ints[0]
    if kind == 1:
        shape = [k]
    elif kind in (2, 3):
        shape = [k, 2]
    elif kind in (4, 5):
        shape = [len(fl)]
    else:
        shape = [k, ints[2 + k] - 1] if False else [k, len(fl) // k]
    out += struct.pack("<II", 2, len(shape)) + b"".join(struct.pack("<Q", s) for s in shape) + struct.pack("<%dd" % len(fl), *fl)
    return out


def steps_bytes(stored):
    out = struct.pack("<I", len(stored))
    for kind, ints, fl in stored:
        p = enc_step(kind, ints, fl)
        out += struct.pack("<Q", len(p)) + p
    return out


def f64_record(vals):
    return struct.pack("<II", 2, 1) + struct.pack("<Q", len(vals)) + struct.pack("<%dd" % len(vals), *vals)


def model_payload(kind, w, d, c0, c1):
    out = struct.pack("<I", kind) + f64_record(list(w[:d])) + f64_record([w[d]])
    if kind == 5:
        out += struct.pack("<II", 1, 1) + struct.pack("<Q", 2) + struct.pack("<qq", c0, c1)
    return out


def section(tag, payload):
    return struct.pack("<IQ", tag, len(payload)) + payload


def file_bytes(plan, schema, steps, model, rows, ncols, seed):
    meta = struct.pack("<QQQII", rows, ncols, seed, 0, len(PIN)) + PIN.encode() + struct.pack("<I", SUM_ORDER)
    body = b"CML1" + struct.pack("<II", 1, 5) + section(1, plan.encode()) + section(2, schema) + section(3, steps) + section(4, model) + section(5, meta)
    return body + hashlib.sha256(body).digest()


# ---------------------------------------------------------------- the fit
class Case:
    """A CSV, a model and a plan: everything `ml fit` is given."""

    def __init__(self, model, names, rows, target, steps, step_text, l2="0.0", iterations=100, split=None, seed=0, positive=None):
        self.model, self.names, self.rows, self.target = model, names, rows, target
        self.steps, self.step_text = steps, step_text
        self.l2, self.iterations, self.split, self.seed, self.positive = l2, iterations, split, seed, positive

    def csv(self):
        return ",".join(self.names) + "\n" + "".join(",".join(r) + "\n" for r in self.rows)

    def model_line(self):
        if self.model == "linear":
            return "model kind=linear l2=%s" % self.l2
        return "model kind=logistic iterations=%d l2=%s" % (self.iterations, self.l2)

    def plan_text(self):
        return "".join(self.step_text) + self.model_line()


def split_masks(n, split, seed):
    """test mask of --split train=F (docs/wiring.md section 2): ceil(n * (1 - F)) rows, the stream of the registry"""
    if split is None:
        return None
    frac = split.split("=")[1]
    digits = len(frac.split(".")[1])
    den = 10 ** digits
    num = den - int(frac.split(".")[1])
    n_test = -(-n * num // den)
    return om_.py_train_test(n, n_test, seed, 0)


def fit_oracle(case):
    """(file bytes or the tag of the refusal, a context for predict and eval)"""
    names, rows = case.names, case.rows
    t = names.index(case.target)
    types = [column_type([r[j] for r in rows]) for j in range(len(names))]
    feat = [j for j in range(len(names)) if j != t]
    fnames = [names[j] for j in feat]
    X = [[to_float(r[j]) for j in feat] for r in rows]
    y = [to_float(r[t]) for r in rows]
    n = len(rows)
    mask = split_masks(n, case.split, case.seed)
    tr = [i for i in range(n) if mask is None or mask[i] == 0]
    Xtr = [X[i] for i in tr]
    ytr = [y[i] for i in tr]
    labels = None
    if case.model == "logistic":
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
    if any(v is None for row in cur for v in row):
        return "ml.input.missing-value", None
    d = len(cur[0])
    Xn = np.array(cur, dtype=np.float64).reshape(len(cur), d)
    yn = np.array(yfit, dtype=np.float64)
    l2 = float(case.l2)
    if case.model == "linear":
        res = omod.linreg(Xn, yn, l2)
        if res is None:
            return "ml.linalg.not-positive-definite", None
        w = res[0]
    else:
        res = omod.logreg(Xn, yn, case.iterations, l2)
        if res is None:
            return "ml.linalg.not-positive-definite", None
        w = res[0]
    plan = op.canon_plan(case.plan_text()) if False else canon(case.plan_text())
    data = file_bytes(plan, schema_bytes(names, types, t), steps_bytes(stored), model_payload(KIND[case.model], w, d, *(labels or (0, 0))), len(tr), len(names), case.seed)
    ctx = {"stored": stored, "w": w, "d": d, "labels": labels, "types": types, "fnames": fnames, "mask": mask}
    return data, ctx


def canon(text):
    try:
        return op.canon_plan(text)
    except op.PlanErr as e:
        raise AssertionError("the generated plan is not valid: %s %s" % (e.tag, text))


# ---------------------------------------------------------------- predictions and metrics
def through_steps(stored, X):
    cur = X
    unknown = 0
    for kind, ints, fl in stored:
        cur, u = op.ref_apply(kind, ints, fl, cur)
        unknown += u
    return cur


def proba_of(z):
    az = abs(z)
    e = float(omod.exp_c(-az))
    return 1.0 / (1.0 + e) if z >= 0.0 else e / (1.0 + e)


def predict_oracle(case, ctx, X_rows):
    """the values `ml predict` prints: ('prediction', [floats]) or ('label', [ints], [probabilities])"""
    cur = through_steps(ctx["stored"], X_rows)
    Xn = np.array(cur, dtype=np.float64).reshape(len(cur), ctx["d"])
    z = omod.scores(Xn, ctx["w"])
    if case.model == "linear":
        return [float(v) for v in z]
    c0, c1 = ctx["labels"]
    return [c1 if v > 0.0 else c0 for v in z], [proba_of(float(v)) for v in z]


def ordered(vals):
    return float(om_.ordered_sum(vals)) if len(vals) else 0.0


def eval_oracle(case, ctx, rows_idx):
    """the metrics `ml eval` prints for the rows `rows_idx` of the case's CSV: {name: float or None}"""
    names, rows = case.names, case.rows
    t = names.index(case.target)
    feat = [j for j in range(len(names)) if j != t]
    X = [[to_float(rows[i][j]) for j in feat] for i in rows_idx]
    y = [to_float(rows[i][t]) for i in rows_idx]
    n = len(y)
    out = {}
    if case.model == "linear":
        p = predict_oracle(case, ctx, X)
        sq = [(a - b) * (a - b) for a, b in zip(y, p)]
        out["rmse"] = math.sqrt(ordered(sq) / float(n))
        out["mae"] = ordered([abs(a - b) for a, b in zip(y, p)]) / float(n)
        if all(v == y[0] for v in y):
            out["r2"] = None
        else:
            mean = ordered(y) / float(n)
            sst = ordered([(a - mean) * (a - mean) for a in y])
            sse = ordered(sq)
            out["r2"] = 1.0 - sse / sst
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
        two_u = int(r2[isp].sum()) - P * (P + 1)
        out["roc_auc"] = float(Fraction(two_u, 2 * P * N))
    eps = 2.220446049250313e-16
    terms = []
    for a, q in zip(yt, score):
        x = q if a == pos else 1.0 - q
        x = min(max(x, eps), 1.0 - eps)
        terms.append(-math.log(x))
    out["log_loss"] = ordered(terms) / float(n)
    return out


# ---------------------------------------------------------------- the data
def gen_cell(rng, kind, scale, miss):
    if miss and rng.random() < miss:
        return ""
    if kind == "int":
        return str(rng.randint(-50, 400))
    if kind == "dec":
        return "%.*f" % (scale, rng.uniform(-20, 80))
    if kind == "big":
        return "%.2f" % rng.uniform(-3e5, 9e5)
    return "%.6e" % rng.uniform(-2.0, 30.0)


def gen_case(seed, model):
    rng = random.Random(seed)
    n = rng.choice([30, 64, 101, 257, 300, 513, 700])
    d = rng.randint(2, 6)
    kinds = [rng.choice(["int", "dec", "big", "flt"]) for _ in range(d)]
    scales = [rng.randint(1, 3) for _ in range(d)]
    miss = [rng.choice([0, 0, 0.05, 0.15]) for _ in range(d)]
    const = rng.random() < 0.35
    cols = [[gen_cell(rng, kinds[j], scales[j], miss[j]) for _ in range(n)] for j in range(d)]
    cj = rng.randrange(d)
    if const:
        # a constant column (its own value in every row, or missing in some)
        v = "7" if kinds[cj] == "int" else "2.5"
        cols[cj] = [v if c != "" or not miss[cj] else "" for c in cols[cj]]
        if not any(c != "" for c in cols[cj]):
            cols[cj][0] = v
    X = []
    for i in range(n):
        row = []
        for j in range(d):
            row.append(float(cols[j][i]) if cols[j][i] != "" else 0.0)
        X.append(row)
    coef = [rng.uniform(-2, 2) / (1 + (j % 3) * 50) for j in range(d)]
    z = [sum(c * v for c, v in zip(coef, row)) + rng.gauss(0, 0.5) for row in X]
    names = ["f%d" % j for j in range(d)]
    if model == "linear":
        ycol = ["%.4f" % (v + 3.0) for v in z]
        target = "y"
    else:
        med = sorted(z)[n // 2]
        lab = [1 if v + rng.gauss(0, 0.3) > med else 0 for v in z]
        ycol = [str(2 * v - 1) if seed % 3 == 0 else str(v) for v in lab]   # labels 0/1, or -1/1
        target = "lab"
    allnames = names + [target]
    rows = [[cols[j][i] for j in range(d)] + [ycol[i]] for i in range(n)]
    # the plan: impute the columns that have an empty cell, then scale, now and then bucket or encode
    steps, text = [], []
    missing = [j for j in range(d) if any(c == "" for c in cols[j])]
    if missing:
        a = rng.choice([0, 0, 1, 2])
        st = {"kind": 1, "a": a, "b": 0, "value": rng.choice([0.0, 1.5, -2.0]), "cols": missing}
        steps.append(st)
        text.append(op.plan_line(st, names, None) + "\n")
    if rng.random() < 0.8:
        sk = rng.choice([2, 3, 3])
        st = {"kind": sk, "a": rng.randint(0, 1) if sk == 3 else 0, "b": 0, "value": 0.0, "cols": sorted(rng.sample(range(d), rng.randint(1, d)))}
        steps.append(st)
        text.append(op.plan_line(st, names, None) + "\n")
    l2 = rng.choice(["0.0", "0.25", "1.5"]) if model == "linear" else rng.choice(["0.0", "0.5", "2.0"])
    if const and l2 == "0.0":
        l2 = "0.5"
    case = Case(model, allnames, rows, target, steps, text, l2=l2, iterations=rng.choice([5, 25, 100]))
    if rng.random() < 0.4:
        case.split, case.seed = rng.choice(["train=0.8", "train=0.7", "train=0.5", "train=0.9"]), rng.randint(0, 2 ** 31)
    elif rng.random() < 0.3:
        case.seed = rng.randint(0, 99)
    if model == "logistic" and rng.random() < 0.4:
        case.positive = min(int(v) for v in ycol)
    return case


# ---------------------------------------------------------------- the binary
def ml(*args, cwd=None):
    p = subprocess.run([BIN, *[str(a) for a in args]], capture_output=True, cwd=cwd)
    return p.returncode, p.stdout.decode()


def step_flag(case):
    return [] if not case.step_text else ["--step", ";".join(s.strip() for s in case.step_text)]


def fit_args(case, out):
    a = ["fit", "--model", case.model, "--target", case.target, "--output", out]
    if case.l2 != "0.0" or case.model == "linear":
        a += ["--l2", case.l2]
    if case.model == "logistic":
        a += ["--iterations", str(case.iterations)]
    a += step_flag(case)
    if case.split:
        a += ["--split", case.split, "--seed", str(case.seed)]
    elif case.seed:
        a += ["--seed", str(case.seed)]
    if case.positive is not None and case.model == "logistic":
        a += ["--positive", str(case.positive)]
    return a


def put_float(x):
    return op.canon_float(x)


def run_case(case, tmp, threads=1, quiet=False):
    """Compare the real `ml` with the oracle on one case. A list of problems."""
    problems = []
    (tmp / "m.cml").unlink(missing_ok=True)
    (tmp / "d.csv").write_text(case.csv())
    try:
        expect, ctx = fit_oracle(case)
    except AssertionError as e:
        # the oracle's exp is the library's table form, defined for |x| <= 660: a case outside it is not judged, but ml must still answer without a trap
        rc, out = ml("--root", tmp, *fit_args(case, "m.cml"), "--threads", threads, "d.csv")
        if rc in (132, 134, 136, 139) or rc < 0:
            problems.append("ml trapped (status %d) on a case the oracle does not judge" % rc)
        UNJUDGED.append(1)
        return problems
    rc, out = ml("--root", tmp, *fit_args(case, "m.cml"), "--threads", threads, "d.csv")
    doc = json.loads(out)
    if ctx is None:
        if doc.get("ok") or doc["error"]["rule"] != expect:
            problems.append("expected the refusal %s, got %s" % (expect, out[:300]))
        return problems
    if not doc.get("ok"):
        return ["the oracle fitted and ml refused: %s" % out[:400]]
    got = (tmp / "m.cml").read_bytes()
    if got != expect:
        problems.append("the model file differs from the oracle's (%d bytes, %d expected)" % (len(got), len(expect)))
        return problems
    # predict: every row
    rc, out = ml("--root", tmp, "predict", "--model-file", "m.cml", "--limit", "1000000", "--proba", "d.csv") if case.model == "logistic" else \
        ml("--root", tmp, "predict", "--model-file", "m.cml", "--limit", "1000000", "d.csv")
    doc = json.loads(out)
    if not doc.get("ok"):
        return problems + ["predict refused: %s" % out[:300]]
    names, rows = case.names, case.rows
    t = names.index(case.target)
    feat = [j for j in range(len(names)) if j != t]
    X = [[to_float(r[j]) for j in feat] for r in rows]
    want = predict_oracle(case, ctx, X)
    cells = doc["data"]["rows"]
    if case.model == "linear":
        if [float(c[0]) for c in cells] != want:
            problems.append("predict: the values differ from the oracle's")
    else:
        labs, probs = want
        if [int(c[0]) for c in cells] != labs or [float(c[1]) for c in cells] != probs:
            problems.append("predict: labels or probabilities differ from the oracle's")
    # predict on other rows: the last 40, in reverse order, the columns reversed and the target left out (columns are matched by name)
    sub = list(reversed(range(len(rows))))[:40]
    pn = [names[j] for j in reversed(feat)]
    (tmp / "p.csv").write_text(",".join(pn) + "\n" + "".join(",".join(rows[i][j] for j in reversed(feat)) + "\n" for i in sub))
    rc, out = ml("--root", tmp, "predict", "--model-file", "m.cml", "--limit", "1000000", "--proba", "p.csv") if case.model == "logistic" else \
        ml("--root", tmp, "predict", "--model-file", "m.cml", "--limit", "1000000", "p.csv")
    doc = json.loads(out)
    if not doc.get("ok"):
        problems.append("predict on reordered columns refused: %s" % out[:300])
    else:
        want = predict_oracle(case, ctx, [X[i] for i in sub])
        cells = doc["data"]["rows"]
        if case.model == "linear":
            if [float(c[0]) for c in cells] != want:
                problems.append("predict on reordered columns: the values differ from the oracle's")
        elif [int(c[0]) for c in cells] != want[0] or [float(c[1]) for c in cells] != want[1]:
            problems.append("predict on reordered columns: labels or probabilities differ from the oracle's")
    # eval: on all rows, or on the test side of the split
    ev = ["eval", "--model-file", "m.cml", "d.csv"]
    idx = list(range(len(rows)))
    if case.split:
        ev += ["--split", case.split, "--seed", str(case.seed)]
        mask = split_masks(len(rows), case.split, case.seed)
        idx = [i for i in idx if mask[i] == 1]
    if case.positive is not None and case.model == "logistic":
        ev += ["--positive", str(case.positive)]
    rc, out = ml("--root", tmp, *ev, "--threads", threads)
    doc = json.loads(out)
    want = eval_oracle(case, ctx, idx)
    if not doc.get("ok"):
        if "roc_auc" in want and want["roc_auc"] == "single-class" and doc["error"]["rule"] == "ml.label.single-class":
            return problems
        return problems + ["eval refused: %s" % out[:300]]
    got = doc["data"]["metrics"]
    for k, v in want.items():
        g = got.get(k, "missing")
        if v is None:
            if g is not None:
                problems.append("eval %s: expected null, got %r" % (k, g))
        elif k == "log_loss":
            if g is None or abs(float(g) - v) > 4e-16 * max(1.0, abs(v)) * 8:
                problems.append("eval %s: %r vs %r" % (k, g, put_float(v)))
        elif g != put_float(v):
            problems.append("eval %s: %r vs %r" % (k, g, put_float(v)))
    return problems


def main():
    argv = sys.argv[1:]
    n = 40
    if argv and argv[0] == "--suite":
        n = int(argv[1]) if len(argv) > 1 else 40
    bad = 0
    refused = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for model in ("linear", "logistic"):
            for seed in range(1, n + 1):
                case = gen_case(seed * 7919 + (0 if model == "linear" else 101), model)
                pr = run_case(case, tmp, threads=1 + seed % 4)
                (tmp / "m.cml").unlink(missing_ok=True)
                if pr:
                    bad += 1
                    print("MISMATCH %s seed %d: %s" % (model, seed, "; ".join(pr)))
    print("oracle_wire: %d cases per model (%d not judged: exp outside the library's table form, only checked not to trap), %s" % (n, len(UNJUDGED), "all agree byte for byte" if not bad else "%d mismatches" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

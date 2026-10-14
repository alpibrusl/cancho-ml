#!/usr/bin/env python3
"""WK9 -- `ml --model kmeans` against scikit-learn's KMeans with the initial centres FIXED (docs/wiring-kmeans.md section 10).

Each program draws its own initial centres (ml from its stream of kind 3, scikit-learn from its generator), so a comparison with each model's own seeded initialisation compares
two random choices and is NOT made. Here scikit-learn is given the initial centres of ml: the rows the initialisation chooses, taken from the oracle (scripts/oracle_models.py),
which the byte-for-byte gate of scripts/oracle_wire_kmeans.py shows equal to the rows ml chooses:

    KMeans(n_clusters=k, init=<those rows as an array>, n_init=1, algorithm='lloyd', max_iter=<iterations>, tol=0)

on the same matrix: the train file without its label, after the standard-scaling step (ddof 0) that ml applies (the matrix is the oracle's, bit for bit the one ml fits). The
tolerances are the ones of section 10, written before the first run and not changed to fit a result:

    inertia, relative                                                     1e-9   (the bound of the library's gate, docs/models-linear.md)
    centroids, the largest absolute difference over the largest |value|   1e-9
    the cluster of every row                                              equal  (a row within 1e-9 relative of equidistant is reported with its margin: the one allowed exception)

Data: the shipped iris (k 3), wine (k 3) and breast-cancer (k 2) train files of docs/examples/data without their label, and a seeded set of 2,000 rows and 6 features with 5
groups (k 5); both initialisations; 10 iterations. scikit-learn moves the farthest point to an empty cluster and ml keeps the centroid: a configuration where a cluster
empties is outside the gate (the script checks that none does).

The controls: the same comparison with scikit-learn given ONE Lloyd iteration (max_iter=1) must fall outside the bounds on at least one data set, and so must a run whose
initial centres are the rows chosen by the OTHER initialisation, or the gate cannot fail and the script says so and exits 1.

    python3 scripts/gate_wire_sklearn_kmeans.py       exit 1 when a bound is exceeded, a cluster empties, or a control stays inside
"""

import json
import os
import pathlib
import subprocess
import struct
import sys
import tempfile
import warnings

import numpy as np
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import oracle_wire_kmeans as owk  # noqa: E402
import oracle_wire as ow  # noqa: E402
import oracle_models as omod  # noqa: E402

BIN = os.environ.get("ML_BIN", str(ROOT / "build" / "ml"))
DATA = ROOT / "docs" / "examples" / "data"
INERTIA, CENTROIDS = 1e-9, 1e-9
ITERS = 10
SETS = [("iris", "virginica", 3), ("wine", "class_0", 3), ("breast_cancer", "benign", 2), ("blobs", None, 5)]


def ml(*args):
    p = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True)
    return json.loads(p.stdout)


def load(name, label):
    if name == "blobs":
        rng = np.random.default_rng(11)
        centres = rng.uniform(-6, 6, (5, 6))
        g = rng.integers(0, 5, 2000)
        X = centres[g] + rng.normal(0, 1.5, (2000, 6))
        names = ["f%d" % j for j in range(6)]
        return names, [["%.6f" % v for v in row] for row in X]
    lines = (DATA / ("%s_train.csv" % name)).read_text().strip().split("\n")
    head = lines[0].split(",")
    j = head.index(label)
    names = [h for i, h in enumerate(head) if i != j]
    rows = [[c for i, c in enumerate(l.split(",")) if i != j] for l in lines[1:]]
    return names, rows


def centroids_of(path, k, d):
    data = pathlib.Path(path).read_bytes()
    pos, secs = 12, {}
    for _ in range(5):
        tag, n = struct.unpack_from("<IQ", data, pos)
        secs[tag] = data[pos + 12:pos + 12 + n]
        pos += 12 + n
    m = secs[4]
    kk, dd = struct.unpack_from("<QQ", m, 12)
    assert (kk, dd) == (k, d)
    return np.array(struct.unpack_from("<%dd" % (k * d), m, 28)).reshape(k, d)


def compare(sk, cen, inertia, lab_ml, Xn):
    """(centroid error, inertia error, rows whose cluster differs, the smallest margin among them)"""
    ce = float(np.max(np.abs(sk.cluster_centers_ - cen)) / np.max(np.abs(cen)))
    ie = abs(sk.inertia_ - inertia) / inertia
    diff = np.nonzero(sk.labels_ != lab_ml)[0]
    margin = None
    for i in diff:
        d = np.sort(((Xn[i] - sk.cluster_centers_) ** 2).sum(axis=1))
        m = (d[1] - d[0]) / max(d[0], 1e-300)
        margin = m if margin is None else min(margin, m)
    return ce, ie, len(diff), margin


def run(name, label, k, init, seed, td):
    names, rows = load(name, label)
    case = owk.KCase(names, rows, None, [], [], k, ITERS, init, None, seed)
    case.steps = [{"kind": 3, "a": 0, "b": 0, "value": 0.0, "cols": list(range(len(names)))}]
    case.step_text = ["scale column=* kind=standard ddof=0\n"]
    expect, ctx = owk.fit_oracle(case)
    assert ctx is not None, expect
    (td / "d.csv").write_text(case.csv())
    a = ["fit", "--model", "kmeans", "--k", k, "--iterations", ITERS, "--init", init, "--seed", seed, "--step", "scale column=* kind=standard", "--output", "m.cml", "--overwrite", "d.csv"]
    d = ml("--root", td, *a)
    assert d["ok"], d
    assert (td / "m.cml").read_bytes() == expect, "the file of ml is not the oracle's: the initial centres below would not be ml's"
    X, init_rows = ctx["Xn"], ctx["init_rows"]
    cen = centroids_of(td / "m.cml", k, X.shape[1])
    inertia = float(d["data"]["summary"]["clusters"]["metrics"]["inertia"])
    sizes = d["data"]["summary"]["clusters"]["metrics"]["sizes"]
    p = ml("--root", td, "predict", "--model-file", "m.cml", "--limit", 1000000, "d.csv")
    lab_ml = np.array([int(r[0]) for r in p["data"]["rows"]])
    return X, init_rows, cen, inertia, sizes, lab_ml


def sk_fit(X, centres, iters):
    return KMeans(n_clusters=len(centres), init=centres, n_init=1, algorithm="lloyd", max_iter=iters, tol=0).fit(X)


def main():
    ok = True
    worst = {"centroids": 0.0, "inertia": 0.0}
    control_out = {"one iteration": False, "other initialisation": False}
    with tempfile.TemporaryDirectory() as tmp:
        td = pathlib.Path(tmp)
        for name, label, k in SETS:
            for init in owk.INITS:
                for seed in (0, 7):
                    X, rows_, cen, inertia, sizes, lab_ml = run(name, label, k, init, seed, td)
                    if min(sizes) == 0:
                        print("%-14s %-9s seed %d  a cluster is empty: outside the gate" % (name, init, seed))
                        ok = False
                        continue
                    sk = sk_fit(X, X[rows_], ITERS)
                    ce, ie, nd, margin = compare(sk, cen, inertia, lab_ml, X)
                    good = ce <= CENTROIDS and ie <= INERTIA and (nd == 0 or (margin is not None and margin <= 1e-9))
                    ok &= good
                    worst["centroids"], worst["inertia"] = max(worst["centroids"], ce), max(worst["inertia"], ie)
                    print("%-14s %-9s seed %d  k %d  centroids observed %.3e (bound %.0e)  inertia observed %.3e (bound %.0e)  rows with another cluster %d%s  %s"
                          % (name, init, seed, k, ce, CENTROIDS, ie, INERTIA, nd, "" if nd == 0 else " (margin %.1e)" % margin, "PASS" if good else "FAIL"))
                    # the controls: a wrong number of iterations, and the other initialisation's centres, on the same ml model
                    sk1 = sk_fit(X, X[rows_], 1)
                    c1 = compare(sk1, cen, inertia, lab_ml, X)
                    if c1[0] > CENTROIDS or c1[1] > INERTIA or c1[2] > 0:
                        control_out["one iteration"] = True
                    other = "random" if init == "kmeans++" else "kmeans++"
                    rows2 = omod.chosen_rows(X, k, seed, 0 if other == "random" else 1)
                    sk2 = sk_fit(X, X[rows2], ITERS)
                    c2 = compare(sk2, cen, inertia, lab_ml, X)
                    if c2[0] > CENTROIDS or c2[1] > INERTIA or c2[2] > 0:
                        control_out["other initialisation"] = True
    for c, outside in control_out.items():
        print("control (%s): %s" % (c, "outside the bounds on at least one data set, as it must be" if outside else "INSIDE the bounds on every data set: the gate cannot fail"))
        ok &= outside
    print("worst of the 16 configurations: centroids %.3e (bound %.0e), inertia %.3e (bound %.0e)" % (worst["centroids"], CENTROIDS, worst["inertia"], INERTIA))
    print("sklearn gate (kmeans, initial centres fixed): %s" % ("all PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

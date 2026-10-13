#!/usr/bin/env python3
"""G-speed of docs/models-linear.md section 2: fit time of lib/models against numpy performing the same algorithm and against
scikit-learn, one thread, minimum of 7 interleaved samples, the load average at the start and at the end of the run.

    python3 scripts/bench_models.py [--sets n100000,n1000000] [--only logreg,linreg,kmeans] [-N 7]

Fit only: the files are read, decoded and (for scikit-learn and numpy) loaded before the clock. cancho's sample is `reps` fits in one
process timed with clock_ms (1 ms resolution) and divided; numpy's and scikit-learn's are `reps` fits timed with perf_counter,
divided. `reps` is chosen from a first run so that a sample lasts about 150 ms. All fits use ALL n rows of the file.
Environment: MODELS_CHECK, ML_DATA. The process re-executes itself with VECLIB_MAXIMUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1
(and the other thread variables) so that numpy's BLAS, scikit-learn's OpenMP code and the scipy it calls use one thread."""
import os, sys
THREADS_ENV = {k: "1" for k in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
if os.environ.get("VECLIB_MAXIMUM_THREADS") != "1" or os.environ.get("OMP_NUM_THREADS") != "1":
    env = dict(os.environ)
    env.update(THREADS_ENV)
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

import platform, subprocess, tempfile, time, warnings, pathlib
import numpy as np
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scipy, sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
import oracle_models as om
import oracle_linalg as ol

BIN = os.environ.get("MODELS_CHECK", str(HERE.parent / "build" / "models-check"))
DATA = pathlib.Path(os.environ.get("ML_DATA", "/Users/alfonso/Workspace/alpibrusl/lexsys/ml-data"))
TARGET_S = 0.15


def cancho(args, reps):
    r = subprocess.run([BIN] + args + ["reps=%d" % reps], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("ms "):
            return int(line.split()[1]) / 1000.0 / reps
    raise SystemExit("driver failed: %s %s" % (args, r.stdout[:300]))


# ---- numpy performing the same algorithms (BLAS sums, numpy exp/log1p: not the same bits)
def np_logreg(A, y, iters):
    n, P = A.shape
    w = np.zeros(P)
    nf = float(n)
    for _ in range(iters):
        z = A @ w
        e = np.exp(-np.abs(z))
        p = np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
        loss = (np.maximum(z, 0.0) + np.log1p(e) - y * z).sum() / nf
        g = (A.T @ (p - y)) / nf
        s = p * (1.0 - p)
        H = ((A * s[:, None]).T @ A) / nf
        L = np.linalg.cholesky(H)
        u = np.linalg.solve(L, g)
        w = w - np.linalg.solve(L.T, u)
    return w, loss


def np_linreg(X, y):
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))])
    G = A.T @ A
    b = A.T @ y
    L = np.linalg.cholesky(G)
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


def np_kmeans(X, cen0, iters):
    cen = cen0.copy()
    k = cen.shape[0]
    x2 = (X * X).sum(1)
    for _ in range(iters):
        D = x2[:, None] - 2.0 * (X @ cen.T) + (cen * cen).sum(1)[None, :]
        lab = D.argmin(1)
        cnt = np.bincount(lab, minlength=k).astype(float)
        sums = np.zeros_like(cen)
        for j in range(X.shape[1]):
            sums[:, j] = np.bincount(lab, weights=X[:, j], minlength=k)
        nz = cnt > 0
        cen[nz] = sums[nz] / cnt[nz, None]
    return cen


def timed(f, reps):
    t = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t) / reps


def reps_for(f):
    t = timed(f, 1)
    return max(1, int(TARGET_S / max(t, 1e-6)))


def main():
    sets = ["n100000", "n1000000"]
    only = None
    N = 7
    a = sys.argv[1:]
    if "--sets" in a:
        sets = a[a.index("--sets") + 1].split(",")
    if "--only" in a:
        only = a[a.index("--only") + 1].split(",")
    if "-N" in a:
        N = int(a[a.index("-N") + 1])
    print("machine %s %s, numpy %s scipy %s scikit-learn %s, python %s, threads env %s" % (platform.machine(), platform.platform(terse=True), np.__version__, scipy.__version__, sklearn.__version__, platform.python_version(), ",".join("%s=%s" % kv for kv in sorted(THREADS_ENV.items())[:3])))
    try:
        cfg = subprocess.run([sys.executable, "-c", "import numpy; numpy.show_config()"], capture_output=True, text=True).stdout
        print("numpy blas:", [l.strip() for l in cfg.splitlines() if "name:" in l][:2])
    except Exception:
        pass
    for s in sets:
        n = int(s[1:])
        d = 20
        xf, yf = str(DATA / s / "x.bin"), str(DATA / s / "y.bin")
        X = np.fromfile(xf, dtype="<f8").reshape(n, d)
        y = np.fromfile(yf, dtype="<f8")
        A = np.hstack([X, np.ones((n, 1))])
        rng = np.random.default_rng(7)
        ylin = X @ rng.standard_normal(d) + 3.0 + 0.5 * rng.standard_normal(n)
        td = tempfile.mkdtemp()
        yl = os.path.join(td, "ylin.bin")
        ylin.astype("<f8").tofile(yl)
        ch = om.chosen_rows(X, 8, 3, 1)
        cen0 = X[ch].copy()
        jobs = {}
        base = ["-" if False else None]
        # name -> (cancho args, [(label, callable)])
        jobs["logreg"] = (["logreg", xf, yf, "n=%d" % n, "d=%d" % d, "iters=6"], [
            ("numpy same algorithm (Newton K=6, loss, BLAS)", lambda: np_logreg(A, y, 6)),
            ("scikit-learn lbfgs, penalty=None, defaults", lambda: LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X, y)),
            ("scikit-learn lbfgs, penalty=None, tol=1e-12", lambda: LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000, tol=1e-12).fit(X, y))])
        jobs["linreg"] = (["linreg", xf, yl, "n=%d" % n, "d=%d" % d], [
            ("numpy same algorithm (normal equations, BLAS)", lambda: np_linreg(X, ylin)),
            ("scikit-learn LinearRegression", lambda: LinearRegression().fit(X, ylin))])
        jobs["kmeans"] = (["kmeans", xf, yf, "n=%d" % n, "d=%d" % d, "k=8", "iters=10", "seed=3", "init=1"], [
            ("numpy same algorithm (Lloyd 10, expanded distances, BLAS)", lambda: np_kmeans(X, cen0, 10)),
            ("scikit-learn KMeans lloyd, same init, max_iter=10, tol=0", lambda: KMeans(n_clusters=8, init=cen0, n_init=1, algorithm="lloyd", max_iter=10, tol=0.0).fit(X))])
        for name in ("logreg", "linreg", "kmeans"):
            if only and name not in only:
                continue
            cargs, others = jobs[name]
            # reps from a first run of each
            p = subprocess.run([BIN] + cargs + ["reps=1"], capture_output=True, text=True)
            t1 = [int(l.split()[1]) for l in p.stdout.splitlines() if l.startswith("ms ")]
            creps = max(1, int(TARGET_S * 1000 / max(t1[0], 1))) if t1 else 1
            oreps = [reps_for(f) for _, f in others]
            labels = ["cancho 1 thread"] + [l for l, _ in others]
            times = [[] for _ in labels]
            la0 = os.getloadavg()
            for _ in range(N):
                times[0].append(cancho(cargs, creps))
                for i, (_, f) in enumerate(others):
                    times[i + 1].append(timed(f, oreps[i]))
            la1 = os.getloadavg()
            print("\n== %s, %s: n=%d d=%d (all rows), N=%d interleaved, load start %.2f %.2f %.2f end %.2f %.2f %.2f" % ((name, s, n, d, N) + la0 + la1))
            m0 = min(times[0])
            for lab, t, r in zip(labels, times, [creps] + oreps):
                print("  %-62s min %9.2f ms  spread %.3f  reps %-4d  cancho/this %.2f" % (lab, 1000 * min(t), max(t) / min(t), r, m0 / min(t)))
            print("  all (ms): " + " | ".join("%s: %s" % (l.split()[0], " ".join("%.1f" % (1000 * v) for v in t)) for l, t in zip(labels, times)))


if __name__ == "__main__":
    main()

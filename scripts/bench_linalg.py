#!/usr/bin/env python3
"""cancho vs numpy/scipy, one thread, the gate of docs/linalg.md section 1.

    VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/bench_linalg.py [options] [cell ...]

A cell is KERNEL:N:D (D is 0 where it does not apply), e.g. gemv_t:100000:64. Options:
    --samples N     samples per cell (default 7); a sample = one numpy timing and one cancho process, interleaved
    --out FILE      append one JSON line per cell
    --threads T     cancho threads (default 1)
    --gate          the cells of the gate (section 1.2)
    --c             also time the C version of the same algorithm (scripts/c_parity; build/parity), interleaved
    --check-env     print cpu/wall of a numpy gemm 1024 (is the single-thread variable honoured?) and exit

Each sample is the mean over R repetitions (R chosen so that a sample is long enough for clock_ms: cancho >= 100 ms, numpy
>= 50 ms). Reported: min over samples, spread = max/min, load average (1 min) at the start and the end of the cell
(flag when >= 2), and the ratio cancho_min / numpy_min."""
import json, os, platform, subprocess, sys, time, resource
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import oracle_linalg as O

BIN = O.BIN
ENV_VARS = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")


def machine():
    cpu = ""
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    if not cpu:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    return "%s, %s, %d cores" % (cpu, platform.platform(), os.cpu_count())


def compiler_commit():
    p = os.path.join(O.ROOT, "cancho.toml")
    for line in open(p):
        if line.startswith("cancho"):
            return line.split('"')[1][:7]
    return "?"


def numpy_case(kernel, n, d):
    """(callable, setup description); data built once with the oracle's generators (the same bits cancho builds)"""
    seed = 1
    s1, s2, s3 = O.derive(seed, 1), O.derive(seed, 2), O.derive(seed, 3)
    import scipy.linalg as sl
    from scipy.linalg import blas
    if kernel == "dot":
        x, y = O.elements(s1, n), O.elements(s2, n)
        return lambda: np.dot(x, y)
    if kernel == "axpy":
        x, y = O.elements(s1, n), O.elements(s2, n)
        a = float(O.elements(s3, 1)[0])
        return lambda: blas.daxpy(x, y, a=a)
    if kernel == "gemv_n":
        X, w = O.elements(s1, n * d).reshape(n, d), O.elements(s2, d)
        y = np.zeros(n)
        return lambda: np.dot(X, w, out=y)
    if kernel == "gemv_t":
        X, r = O.elements(s1, n * d).reshape(n, d), O.elements(s2, n)
        g = np.zeros(d)
        return lambda: np.dot(r, X, out=g)
    if kernel == "syrk":
        X = O.elements(s1, n * d).reshape(n, d)
        XT = X.T  # Fortran-contiguous view: no copy inside f2py
        return lambda: blas.dsyrk(1.0, XT, trans=0)
    if kernel == "chol":
        A = O.spd(n, s1)
        return lambda: sl.cholesky(A, lower=True, check_finite=False)
    if kernel in ("fwd", "bwd"):
        L = O.chol_o(O.spd(n, s1))
        b = O.elements(s2, n)
        t = 0 if kernel == "fwd" else 1
        return lambda: sl.solve_triangular(L, b, lower=True, trans=t, check_finite=False)
    if kernel == "gemm":
        A, B = O.elements(s1, n * n).reshape(n, n), O.elements(s2, n * n).reshape(n, n)
        C = np.zeros((n, n))
        return lambda: np.dot(A, B, out=C)
    raise SystemExit("unknown kernel " + kernel)


def raw_case(kernel, n):
    """the thinnest scipy call for the kernels where scipy.linalg's wrapper adds Python-level work (validation, a copy, the
    clean-up of the other triangle): the f2py wrapper of LAPACK itself, on Fortran-ordered input (no transposing copy)"""
    from scipy.linalg import lapack
    seed = 1
    s1, s2 = O.derive(seed, 1), O.derive(seed, 2)
    if kernel == "chol":
        Af = np.asfortranarray(O.spd(n, s1))
        return lambda: lapack.dpotrf(Af, lower=1, clean=0)
    if kernel in ("fwd", "bwd"):
        Lf = np.asfortranarray(O.chol_o(O.spd(n, s1)))
        b = O.elements(s2, n)
        t = 0 if kernel == "fwd" else 1
        return lambda: lapack.dtrtrs(Lf, b, lower=1, trans=t)
    return None


def time_numpy(fn, reps):
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


def cancho_ms(kernel, n, d, reps, threads):
    r = O.run_cancho(kernel, n, d, 1, reps, threads)
    if "ms" not in r:
        raise SystemExit("cancho failed: %s" % r)
    return int(r["ms"]) / reps


CBIN = os.environ.get("PARITY_BIN", os.path.join(O.ROOT, "build", "parity"))


def c_ms(kernel, n, d, reps):
    r = subprocess.run([CBIN, kernel, str(n), str(d), "1", str(reps)], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("ms "):
            return float(line.split()[1]) / reps
    raise SystemExit("parity failed: " + r.stdout + r.stderr)


C_KERNELS = ("dot", "axpy", "gemv_n", "gemv_t", "syrk", "chol")  # the kernels scripts/c_parity/parity.c implements


def cell(kernel, n, d, samples, threads, with_c=False):
    with_c = with_c and kernel in C_KERNELS
    fn = numpy_case(kernel, n, d)
    fn()  # warm
    raw = raw_case(kernel, n) if kernel in ("chol", "fwd", "bwd") else None
    if raw:
        raw()
    # calibrate reps
    t1 = time_numpy(fn, 1)
    rn = max(1, int(0.05 / max(t1, 1e-7)))
    rc = 1
    while True:
        ms = cancho_ms(kernel, n, d, rc, threads) * rc
        if ms >= 100 or rc >= 1 << 22:
            break
        rc = max(rc * 2, int(rc * 150 / max(ms, 1)))
    load0 = os.getloadavg()[0]
    np_t, ca_t, c_t, raw_t = [], [], [], []
    for _ in range(samples):
        np_t.append(time_numpy(fn, rn))
        if raw:
            raw_t.append(time_numpy(raw, rn))
        ca_t.append(cancho_ms(kernel, n, d, rc, threads) / 1000.0)
        if with_c:
            c_t.append(c_ms(kernel, n, d, rc) / 1000.0)
    load1 = os.getloadavg()[0]
    cm, nm = min(ca_t), min(np_t)
    extra = {}
    if raw:
        extra.update(raw_s=min(raw_t), raw_spread=max(raw_t) / min(raw_t), raw_ratio=cm / min(raw_t), all_raw=raw_t)
    if with_c:
        extra.update(c_s=min(c_t), c_spread=max(c_t) / min(c_t), all_c=c_t, cancho_over_c=cm / min(c_t), c_over_numpy=min(c_t) / nm)
    return dict(extra, **dict(kernel=kernel, n=n, d=d, threads=threads, samples=samples, reps_cancho=rc, reps_numpy=rn,
                cancho_s=cm, numpy_s=nm, ratio=cm / nm, spread_cancho=max(ca_t) / cm, spread_numpy=max(np_t) / nm,
                load_start=load0, load_end=load1, flag_load=bool(max(load0, load1) >= 2.0),
                all_cancho=ca_t, all_numpy=np_t))


GATE = []
for n_ in (1_000_000,):
    GATE += [("dot", n_, 0), ("axpy", n_, 0)]
for n_ in (100_000, 1_000_000):
    for d_ in (20, 64, 256):
        GATE += [("gemv_n", n_, d_), ("gemv_t", n_, d_), ("syrk", n_, d_)]
for d_ in (21, 64, 256):
    GATE += [("chol", d_, 0), ("fwd", d_, 0), ("bwd", d_, 0)]
for n_ in (256, 512, 1024):
    GATE += [("gemm", n_, 0)]


def check_env():
    a = np.random.default_rng(1).standard_normal((1024, 1024))
    b = np.random.default_rng(2).standard_normal((1024, 1024))
    np.dot(a, b)
    r0 = resource.getrusage(resource.RUSAGE_SELF); t0 = time.perf_counter()
    for _ in range(5):
        np.dot(a, b)
    r1 = resource.getrusage(resource.RUSAGE_SELF); t1 = time.perf_counter()
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    print(json.dumps(dict(env={k: os.environ.get(k) for k in ENV_VARS}, gemm1024_cpu_over_wall=cpu / (t1 - t0), wall_s=t1 - t0)))


def main():
    a = sys.argv[1:]
    if "--check-env" in a:
        check_env(); return
    samples = int(a[a.index("--samples") + 1]) if "--samples" in a else 7
    threads = int(a[a.index("--threads") + 1]) if "--threads" in a else 1
    out = a[a.index("--out") + 1] if "--out" in a else None
    with_c = "--c" in a
    cells = []
    skip = set()
    for i, x in enumerate(a):
        if x in ("--samples", "--threads", "--out"):
            skip.add(i); skip.add(i + 1)
    for i, x in enumerate(a):
        if i in skip or x.startswith("--"):
            continue
        k, n, d = x.split(":"); cells.append((k, int(n), int(d)))
    if "--gate" in a:
        cells = GATE + cells
    meta = dict(machine=machine(), compiler=compiler_commit(), numpy=np.__version__,
                env={k: os.environ.get(k) for k in ENV_VARS}, python=platform.python_version())
    for k, n, d in cells:
        # protocol: a cell with spread > 1.25 is rerun (up to 2 more times); the attempt with the smallest worst spread is kept
        best = None
        for attempt in range(3):
            res = cell(k, n, d, samples, threads, with_c)
            res["attempt"] = attempt + 1
            worst = max(res["spread_cancho"], res["spread_numpy"])
            if best is None or worst < max(best["spread_cancho"], best["spread_numpy"]):
                best = res
            if worst <= 1.25:
                break
        res = best
        res["meta"] = meta
        print("%-7s n=%-8d d=%-4d t=%d  cancho %.6fs  numpy %.6fs  ratio %.2f  spread %.2f/%.2f  load %.1f/%.1f%s  R=%d/%d" % (
            k, n, d, threads, res["cancho_s"], res["numpy_s"], res["ratio"], res["spread_cancho"], res["spread_numpy"],
            res["load_start"], res["load_end"], " LOAD>=2" if res["flag_load"] else "", res["reps_cancho"], res["reps_numpy"]) + (("  attempts %d" % res["attempt"] if res["attempt"] > 1 else "")) + (
            ("   C %.6fs  cancho/C %.2f  C/numpy %.2f" % (res["c_s"], res["cancho_over_c"], res["c_over_numpy"]) if "c_s" in res else "")) + (
            "   raw LAPACK %.6fs  cancho/raw %.2f" % (res["raw_s"], res["raw_ratio"]) if "raw_s" in res else ""), flush=True)
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()

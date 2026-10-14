#!/usr/bin/env python3
"""C scalar / C NEON (fixed lane orders) vs numpy/scipy, one thread: the protocol of docs/linalg.md 1.1 (and docs/simd-ceiling.md 1.2).

    python3 scripts/simd/bench_simd.py [--out FILE] [--samples 7] [--gate | KERNEL:N:D ...]

One sample = one numpy timing, then one run of `build/simd bench` (which times every C variant of the cell in turn and, on the first
sample of the first attempt, checks every NEON result against the scalar reference of its order). Statistic: minimum over the
samples per variant, spread = max/min, load average at the start and the end of the cell (flag when >= 2). A cell whose worst
spread (numpy, CS, best NEON variant) exceeds 1.25 is rerun up to twice; the attempt with the smallest worst spread is kept.
Reuses scripts/bench_linalg.py (numpy_case, raw_case, time_numpy, GATE, machine) unchanged."""
import os, sys, json, subprocess, time
for k in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[k] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import bench_linalg as B

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BIN = os.path.join(ROOT, "build", "simd")
KERNELS = ("dot", "axpy", "gemv_n", "gemv_t", "syrk", "chol")


def c_run(kernel, n, d, verify):
    r = subprocess.run([BIN, "bench", kernel, str(n), str(d), "1", "verify" if verify else "noverify"], capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        p = line.split()
        if p and p[0] == "res":
            out[p[1]] = dict(s=float(p[2]), hash=p[3], ver=p[4])
    if not out:
        raise SystemExit("simd bench failed: %s %s" % (r.stdout, r.stderr))
    return out


def cell(kernel, n, d, samples, first):
    fn = B.numpy_case(kernel, n, d)
    fn()
    raw = B.raw_case(kernel, n) if kernel == "chol" else None
    if raw:
        raw()
    t1 = B.time_numpy(fn, 1)
    rn = max(1, int(0.05 / max(t1, 1e-7)))
    load0 = os.getloadavg()[0]
    np_t, raw_t, var = [], [], {}
    for s in range(samples):
        np_t.append(B.time_numpy(fn, rn))
        if raw:
            raw_t.append(B.time_numpy(raw, rn))
        res = c_run(kernel, n, d, verify=(first and s == 0))
        for k, v in res.items():
            e = var.setdefault(k, dict(all=[], hash=v["hash"], ver="-"))
            e["all"].append(v["s"])
            if v["ver"] not in ("-", "noverify"):
                e["ver"] = v["ver"]
            if e["hash"] != v["hash"] and kernel != "axpy":
                e["hash_unstable"] = True
    load1 = os.getloadavg()[0]
    for e in var.values():
        e["min"] = min(e["all"]); e["spread"] = max(e["all"]) / min(e["all"])
    nm = min(np_t)
    best_n = min((k for k in var if k.startswith("N") and k[1:].isdigit()), key=lambda k: var[k]["min"], default=None)
    worst = max([max(np_t) / nm, var["CS"]["spread"]] + ([var[best_n]["spread"]] if best_n else []))
    r = dict(kernel=kernel, n=n, d=d, samples=samples, reps_numpy=rn, numpy_s=nm, numpy_spread=max(np_t) / nm, all_numpy=np_t,
             variants=var, best_n=best_n, worst_spread=worst, load_start=load0, load_end=load1, flag_load=bool(max(load0, load1) >= 2.0))
    if raw:
        r.update(raw_s=min(raw_t), raw_spread=max(raw_t) / min(raw_t))
    return r


def main():
    a = sys.argv[1:]
    samples = int(a[a.index("--samples") + 1]) if "--samples" in a else 7
    out = a[a.index("--out") + 1] if "--out" in a else None
    skip = set()
    for i, x in enumerate(a):
        if x in ("--samples", "--out"):
            skip.update((i, i + 1))
    cells = []
    for i, x in enumerate(a):
        if i in skip or x.startswith("--"):
            continue
        k, n, d = x.split(":"); cells.append((k, int(n), int(d)))
    if "--gate" in a:
        cells = [c for c in B.GATE if c[0] in KERNELS and not (c[0] == "chol" and c[1] not in (21, 64, 256))] + cells
    meta = dict(machine=B.machine(), numpy=B.np.__version__, env={k: os.environ.get(k) for k in B.ENV_VARS}, python=sys.version.split()[0],
                clang=subprocess.run(["clang", "--version"], capture_output=True, text=True).stdout.splitlines()[0])
    for k, n, d in cells:
        best = None; first_ver = None
        for attempt in range(3):
            res = cell(k, n, d, samples, first=(attempt == 0))
            res["attempt"] = attempt + 1
            if attempt == 0:
                first_ver = {k2: e["ver"] for k2, e in res["variants"].items()}
            if best is None or res["worst_spread"] < best["worst_spread"]:
                best = res
            if res["worst_spread"] <= 1.25:
                break
        res = best; res["meta"] = meta
        for k2, e in res["variants"].items():
            e["ver"] = first_ver.get(k2, e["ver"])
        v = res["variants"]
        line = "%-7s n=%-8d d=%-4d numpy %.6fs  CS %.6fs  bestN %s %.6fs  ratio N/numpy %.2f  spread %.2f  load %.1f/%.1f%s  att %d" % (
            k, n, d, res["numpy_s"], v["CS"]["min"], res["best_n"], v[res["best_n"]]["min"] if res["best_n"] else 0,
            (v[res["best_n"]]["min"] / res["numpy_s"]) if res["best_n"] else 0, res["worst_spread"], res["load_start"], res["load_end"],
            " LOAD>=2" if res["flag_load"] else "", res["attempt"])
        print(line, flush=True)
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()

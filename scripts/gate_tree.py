#!/usr/bin/env python3
"""The identity, thread and spike gates of lib/models/tree (docs/models-tree.md section 8): G-ident, G-thread, G-spike.
usage: gate_tree.py [--ident] [--threads] [--spike] [--sizes 1000,100000] [--configs C1,R1] [--cancho PATH] [--src DIR]
Builds tools/tree-check (from --src, default the repository), runs every configuration, and compares every E, N and P line,
byte for byte, with the oracle's (scripts/oracle_tree.py, cached in build/oracle/). Exit 1 on any difference.
Also importable: scripts/mutants_tree.py uses `build`, `run`, `CONFIGS`, `oracle` and `same`."""
import argparse, os, subprocess, sys, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANCHO = os.environ.get("CANCHO", os.path.join(os.path.dirname(ROOT), "cancho-compiler-0207556", "target", "release", "cancho"))
ML = os.environ.get("ML_DATA", os.path.join(os.path.dirname(ROOT), "ml-data"))
TD = os.path.join(ROOT, "build", "tree-data")
SEED = "20261012"

# id: task, trees, depth, leaf, bins, mtry, boot, missing, dataset ("n" = the ml-data sizes, "edge" = the edge-case set)
CONFIGS = {
    "C1": dict(task=0, trees=10, depth=8, leaf=1, bins=64, mtry=4, boot=1, miss=False, data="n"),
    "C2": dict(task=0, trees=7, depth=10, leaf=5, bins=32, mtry=5, boot=1, miss=True, data="n"),
    "C3": dict(task=0, trees=1, depth=6, leaf=1, bins=64, mtry=20, boot=0, miss=False, data="n"),
    "C4": dict(task=0, trees=3, depth=8, leaf=2, bins=200, mtry=6, boot=1, miss=True, data="n"),
    "C5": dict(task=0, trees=2, depth=7, leaf=1, bins=255, mtry=8, boot=1, miss=False, data="n"),
    "C6": dict(task=0, trees=4, depth=4, leaf=1, bins=2, mtry=20, boot=1, miss=True, data="n"),
    "R1": dict(task=1, trees=8, depth=9, leaf=3, bins=48, mtry=7, boot=1, miss=True, data="n"),
    "R2": dict(task=1, trees=1, depth=7, leaf=2, bins=64, mtry=20, boot=0, miss=False, data="n"),
    "E1": dict(task=0, trees=6, depth=6, leaf=2, bins=16, mtry=0, boot=1, miss=True, data="edge"),
    "E2": dict(task=1, trees=5, depth=7, leaf=1, bins=32, mtry=3, boot=1, miss=True, data="edge"),
    "E3": dict(task=0, trees=1, depth=5, leaf=1, bins=8, mtry=8, boot=0, miss=True, data="edge"),
    "E4": dict(task=1, trees=4, depth=6, leaf=1, bins=16, mtry=4, boot=1, miss=True, data="edge", ytag="yq"),
}


def inputs(cfg, size):
    """(args for x, y, miss, n, d) of a configuration at a size"""
    if cfg["data"] == "edge":
        d = f"{TD}/edge"
        return dict(x=f"{d}/xm.bin", y=f"{d}/{cfg.get('ytag', 'yr' if cfg['task'] else 'y')}.bin", miss=f"{d}/miss.bin", n=1000, d=8)
    nd = f"{TD}/n{size}"
    y = f"{nd}/reg_y.bin" if cfg["task"] else f"{ML}/n{size}/y.bin"
    x = f"{nd}/xm.bin" if cfg["miss"] else f"{ML}/n{size}/x.bin"
    r = dict(x=x, y=y, n=int(size), d=20)
    if cfg["miss"]:
        r["miss"] = f"{nd}/miss.bin"
    return r


def argv(cfg, size, threads=1, out="full", extra=()):
    i = inputs(cfg, size)
    a = ["x=" + i["x"], "y=" + i["y"], "n=%d" % i["n"], "d=%d" % i["d"]]
    if "miss" in i:
        a.append("miss=" + i["miss"])
    a += ["task=%d" % cfg["task"], "seed=" + SEED, "trees=%d" % cfg["trees"], "depth=%d" % cfg["depth"], "leaf=%d" % cfg["leaf"],
          "bins=%d" % cfg["bins"], "mtry=%d" % cfg["mtry"], "boot=%d" % cfg["boot"], "threads=%d" % threads, "out=" + out]
    return a + list(extra)


def build(src, exe, cancho=CANCHO, extra=()):
    files = sorted(os.listdir(f"{src}/lib/core")) and [f"{src}/lib/core/{f}" for f in sorted(os.listdir(f"{src}/lib/core")) if f.endswith(".cho")]
    files += [f"{src}/lib/linalg/tensor.cho"]  # lib/core/matrix.cho imports linalg.tensor (integration)
    files += [f"{src}/lib/models/tree/{f}" for f in sorted(os.listdir(f"{src}/lib/models/tree")) if f.endswith(".cho")]
    files += [f"{src}/tools/tree-check/main.cho"]
    r = subprocess.run([cancho, "build", "--std", *files, "-o", exe], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("build failed:\n" + r.stdout + r.stderr)


def run(exe, cfg, size, threads=1, out="full", timeout=300, extra=()):
    r = subprocess.run([exe, "mode=fit"] + argv(cfg, size, threads, out, extra), capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout


def oracle(cid, size):
    cfg = CONFIGS[cid]
    os.makedirs(f"{ROOT}/build/oracle", exist_ok=True)
    p = f"{ROOT}/build/oracle/{cid}_{size}.txt"
    script = f"{ROOT}/scripts/oracle_tree.py"
    if not os.path.exists(p) or os.path.getmtime(p) < os.path.getmtime(script):
        i = inputs(cfg, size)
        a = ["x=" + i["x"], "y=" + i["y"], "n=%d" % i["n"], "d=%d" % i["d"], "task=%d" % cfg["task"], "seed=" + SEED, "trees=%d" % cfg["trees"],
             "depth=%d" % cfg["depth"], "leaf=%d" % cfg["leaf"], "bins=%d" % cfg["bins"], "mtry=%d" % cfg["mtry"], "boot=%d" % cfg["boot"], "out=" + p]
        if "miss" in i:
            a.append("miss=" + i["miss"])
        r = subprocess.run([sys.executable, script] + a, capture_output=True, text=True, check=True)
        open(p + ".summary", "w").write(r.stdout)
    return open(p).read().splitlines(), open(p + ".summary").read().strip()


def lines_of(out):
    return [l for l in out.splitlines() if l[:2] in ("E ", "N ", "P ")]


def same(a, b):
    """None if equal, else a short description of the first difference"""
    if len(a) != len(b):
        return "line counts %d vs %d" % (len(a), len(b))
    for i in range(len(a)):
        if a[i] != b[i]:
            return "line %d: %r vs %r" % (i, a[i], b[i])
    return None


def pairs(cids, sizes):
    for cid in cids:
        for size in (["1000"] if CONFIGS[cid]["data"] == "edge" else sizes):
            yield cid, size


def ident(exe, cids, sizes, verbose=True):
    bad = 0
    for cid, size in pairs(cids, sizes):
        code, out = run(exe, CONFIGS[cid], size)
        want, summ = oracle(cid, size)
        d = "program exit %d" % code if code else same(lines_of(out), want)
        if verbose:
            print("ident  %-3s n=%-7s %s   [%s]" % (cid, size, "IDENTICAL" if d is None else "DIFFERENT: " + d, summ))
        bad += d is not None
    return bad


def hashes(out):
    return {l.split()[1]: l.split()[2] for l in out.splitlines() if l.startswith("H ")}


def thread_gate(exe, sizes_trees=(("C1", "100000", 10), ("C2", "100000", 10), ("R1", "100000", 10), ("C1", "100000", 32), ("R1", "100000", 32), ("C2", "1000", 7), ("C4", "100000", 3), ("C5", "1000", 2), ("E1", "1000", 6), ("E2", "1000", 5), ("E4", "1000", 4)),
                counts=(1, 2, 3, 4, 5, 8, 16), verbose=True, timeout=300):
    bad = 0
    for cid, size, trees in sizes_trees:
        cfg = dict(CONFIGS[cid], trees=trees)
        ref = None
        row = []
        for t in counts:
            code, out = run(exe, cfg, size, threads=t, out="hash", timeout=timeout)
            h = hashes(out) if code == 0 else {"exit": str(code)}
            row.append("%d:%s" % (t, h.get("forest", "?")[:8]))
            if ref is None:
                ref = h
            elif h != ref or code != 0:
                bad += 1
        if verbose:
            print("thread %-3s n=%-7s T=%-3d %s   %s" % (cid, size, trees, "EQUAL" if all(r.split(":")[1] == row[0].split(":")[1] for r in row) and bad == 0 else "DIFFERENT", " ".join(row)))
    return bad


def unit(src, cancho=CANCHO, verbose=True, timeout=300):
    """tests/t_tree.cho: the refusal fixtures and the hand-worked cases; fails on a FAILED line, a trap or a non-zero exit"""
    files = [f"{src}/lib/core/{f}" for f in sorted(os.listdir(f"{src}/lib/core")) if f.endswith(".cho")]
    files += [f"{src}/lib/linalg/tensor.cho"]  # lib/core/matrix.cho imports linalg.tensor (integration)
    files += [f"{src}/lib/models/tree/{f}" for f in sorted(os.listdir(f"{src}/lib/models/tree")) if f.endswith(".cho")] + [f"{src}/tests/t_tree.cho"]
    exe = f"{src}/build/t_tree"
    os.makedirs(f"{src}/build", exist_ok=True)
    r = subprocess.run([cancho, "build", "--std", *files, "-o", exe], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("build failed:\n" + r.stdout + r.stderr)
    r = subprocess.run([exe], capture_output=True, text=True, timeout=timeout)
    oks = [l for l in r.stderr.splitlines() if l.startswith("ok ")]
    bad = r.returncode != 0 or "FAILED" in r.stderr
    if verbose:
        print("unit   %d tests ok, exit %d%s" % (len(oks), r.returncode, "   " + " ".join(l for l in r.stderr.splitlines() if l.startswith("FAILED")) if bad else ""))
    return int(bad)


def exact(exe, verbose=True):
    """tree-check mode=gt: the limb comparison of tree.exact against Python big integers on 2,006 pairs"""
    r = subprocess.run([exe, "mode=gt"], capture_output=True, text=True, timeout=60)
    bad, n, ones = 0, 0, 0
    for l in r.stdout.splitlines():
        t = l.split()
        if t and t[0] == "G":
            a, b, c, d, got = map(int, t[1:6])
            want = int(a * d > c * b)
            n += 1
            ones += want
            bad += got != want
    bad += n < 2000
    if verbose:
        print("exact  %d pairs against Python integers (%d with a > c), %d wrong" % (n, ones, bad))
    return int(bad > 0)


def fuzz(exe, verbose=True, timeout=180):
    """tree-check mode=fuzz: every truncation and many damaged payloads; a trap, a hang or an unexpected result fails"""
    i = inputs(dict(task=0, miss=True, data="n"), "1000")
    try:
        r = subprocess.run([exe, "mode=fuzz", "x=" + i["x"], "y=" + i["y"], "miss=" + i["miss"], "n=1000", "d=20"], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if verbose: print("fuzz   TIMEOUT (a payload made the reader or predict run without end)")
        return 1
    ok = r.returncode == 0 and "fuzz ok" in r.stdout
    if verbose:
        print("fuzz   %s   %s" % ("ok" if ok else "FAILED (exit %d)" % r.returncode, " | ".join(l for l in r.stdout.splitlines() if l.startswith("fuzz ") and l != "fuzz ok")))
    return int(not ok)


def spike_gate(cancho, verbose=True):
    """C1 against spike F's own program (scripts/spikes/spike_f.cho): heap numbers recomputed from the preorder"""
    sp = os.path.join(ROOT, "scripts", "spikes", "spike_f.cho")
    exe = f"{ROOT}/build/spike_f"
    r = subprocess.run([cancho, "build", sp, "--std", "-o", exe], capture_output=True, text=True)
    if r.returncode:
        print("spike build failed", r.stderr); return 1
    bad = 0
    for size in ("1000", "100000"):
        s = subprocess.run([exe, f"{ML}/n{size}/x.bin", f"{ML}/n{size}/y.bin", size, SEED], capture_output=True, text=True).stdout
        S, E, P = {}, [], []
        for l in s.splitlines():
            t = l.split()
            if t and t[0] == "N": S[(int(t[1]), int(t[2]))] = tuple(map(int, t[3:7]))
            elif t and t[0] == "E": E.append(l)
            elif t and t[0] == "P": P.append(l)
        want, _ = oracle("C1", size)
        nodes, E2, P2 = {}, [l for l in want if l.startswith("E ")], [l for l in want if l.startswith("P ")]
        for l in want:
            t = l.split()
            if t[0] == "N": nodes.setdefault(int(t[1]), []).append(list(map(int, t[2:10])))
        M = {}
        for tr, ns in nodes.items():
            heap = {0: 1}
            for i, (_, feat, b, dl, lc, rc, na, nb) in enumerate(ns):
                h = heap[i]; M[(tr, h)] = (feat, b, na, nb)
                if feat >= 0: heap[lc] = 2 * h; heap[rc] = 2 * h + 1
        ok = S == M and E == E2 and P == P2
        if verbose:
            print("spike  C1  n=%-7s %s (nodes %d, edges %d, predictions %d)" % (size, "IDENTICAL to spike F" if ok else "DIFFERENT", len(S), len(E), len(P)))
        bad += not ok
    return bad


# ---------------------------------------------------------------- the weighted growth (docs/weights.md section 2.5)

WEIGHTED = ("E1", "E2", "E3", "E4")


def weighted(exe, verbose=True):
    """G-weights: for each edge configuration and each weight kind (1 unit, 2 integers, 3 floats, 4 floats with zeros) the weighted driver equals the weighted oracle line for line
    (E, N and P), and the weighted forest is the same bytes at 1, 2, 4 and 8 threads. Also, with unit weights a regression forest is the unweighted one (the sums of 1.0 are exact)"""
    bad = 0
    for cid in WEIGHTED:
        cfg = CONFIGS[cid]
        i = inputs(cfg, "1000")
        for kind in (1, 2, 3, 4, 5):
            code, out = run(exe, cfg, "1000", extra=["wt=%d" % kind])
            p = f"{ROOT}/build/oracle/w{kind}_{cid}.txt"
            os.makedirs(f"{ROOT}/build/oracle", exist_ok=True)
            a = ["x=" + i["x"], "y=" + i["y"], "n=%d" % i["n"], "d=%d" % i["d"], "task=%d" % cfg["task"], "seed=" + SEED, "trees=%d" % cfg["trees"], "depth=%d" % cfg["depth"],
                 "leaf=%d" % cfg["leaf"], "bins=%d" % cfg["bins"], "mtry=%d" % cfg["mtry"], "boot=%d" % cfg["boot"], "miss=" + i["miss"], "wt=%d" % kind, "out=" + p]
            summ = subprocess.run([sys.executable, f"{ROOT}/scripts/oracle_tree.py"] + a, capture_output=True, text=True, check=True).stdout.strip()
            d = "program exit %d" % code if code else same(lines_of(out), open(p).read().splitlines())
            if verbose:
                print("wident %-3s wt=%d %s   [%s]" % (cid, kind, "IDENTICAL" if d is None else "DIFFERENT: " + d, summ))
            bad += d is not None
            if kind == 3 or kind == 4:
                ref = None
                for t in (1, 2, 4, 8):
                    c2, o2 = run(exe, dict(cfg), "1000", threads=t, out="hash", extra=["wt=%d" % kind])
                    h = hashes(o2) if c2 == 0 else {"exit": str(c2)}
                    if ref is None:
                        ref = h
                    elif h != ref:
                        bad += 1
                        print("wthread %-3s wt=%d DIFFERENT at %d threads" % (cid, kind, t))
        if cfg["task"] == 1:
            c1, o1 = run(exe, cfg, "1000")
            c2, o2 = run(exe, cfg, "1000", extra=["wt=1"])
            same_unit = c1 == 0 and c2 == 0 and lines_of(o1) == lines_of(o2)
            if verbose:
                print("wunit  %-3s unit weights equal the unweighted regression forest: %s" % (cid, "EQUAL" if same_unit else "DIFFERENT"))
            bad += not same_unit
    return bad


def few_valued(n, d, seed, task):
    """a data set whose columns take 6 values each (so the quantile bins separate every value whatever the duplication), a class or a number by a planted rule"""
    import numpy as np
    r = np.random.default_rng(seed)
    x = r.integers(0, 6, size=(n, d)).astype(np.float64)
    z = x[:, 0] * 1.5 - x[:, 1] + 0.7 * (x[:, 2] > 2) + r.normal(0, 1.0, n)
    y = (z > np.median(z)).astype(np.float64) if task == 0 else z
    return x, y


def node_rows(out):
    rows = [l.split() for l in out.splitlines() if l.startswith("N ")]
    # the bin number is left out: several bins of a column with 6 values share one edge, and the bin that holds an edge is a detail of the counts; the threshold (the last field of a node that splits) says the same
    return [(int(t[3]), int(t[5]), int(t[6]), int(t[7]), t[10]) for t in rows]


def dup_gate(exe, verbose=True):
    """G-dup (docs/weights.md WG3): a single tree on integer weights predicts, for every training row, what the tree of the file with the rows repeated predicts (a weight of 0 is a row
    left out), within 1e-12 relative (the sums are made in another order). The training rows are given again as the rows to predict. The trees themselves may differ in a
    node where two splits make the same partition (their scores are equal in real arithmetic and rounding orders them), which is why predictions and not nodes are compared: the
    number of nodes that agree is printed. The bins are the same because every column has 6 values and each has far more than n / bins rows"""
    import struct
    import numpy as np
    sys.path.insert(0, f"{ROOT}/scripts")
    import oracle_tree as ot
    os.makedirs(f"{ROOT}/build/wdup", exist_ok=True)
    bad = 0

    def bits_to_f(h):
        return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]

    def preds(out):
        return [bits_to_f(l.split()[2]) for l in out.splitlines() if l.startswith("P ")]

    for task in (0, 1):
        for kind in (2, 5):
            n, d = 400, 4
            x, y = few_valued(n, d, 17 + task, task)
            w = ot.gen_weights(int(SEED), n, kind)
            keep = []
            for i in range(n):
                keep += [i] * int(w[i])
            xa, ya = np.concatenate([x, x]), np.concatenate([y, y])
            xd, yd = np.concatenate([x[keep], x]), np.concatenate([y[keep], y])
            for name, arr in (("x", xa), ("y", ya), ("xd", xd), ("yd", yd)):
                arr.astype("<f8").tofile(f"{ROOT}/build/wdup/{name}.bin")
            common = ["trees=1", "depth=6", "leaf=1", "bins=64", "mtry=%d" % d, "boot=0", "d=%d" % d, "task=%d" % task, "seed=" + SEED, "out=full"]
            r1 = subprocess.run([exe, "mode=fit", f"x={ROOT}/build/wdup/x.bin", f"y={ROOT}/build/wdup/y.bin", "n=%d" % (2 * n), "ntr=%d" % n, "wt=%d" % kind] + common, capture_output=True, text=True)
            r2 = subprocess.run([exe, "mode=fit", f"x={ROOT}/build/wdup/xd.bin", f"y={ROOT}/build/wdup/yd.bin", "n=%d" % (len(keep) + n), "ntr=%d" % len(keep)] + common, capture_output=True, text=True)
            a1, a2 = preds(r1.stdout), preds(r2.stdout)
            ok = r1.returncode == 0 and r2.returncode == 0 and len(a1) == n and len(a2) == n
            worst = max((abs(u - v) / max(1.0, abs(v)) for u, v in zip(a1, a2)), default=1.0) if ok else 1.0
            ok = ok and worst <= 1e-12
            na, nb = node_rows(r1.stdout), node_rows(r2.stdout)
            agree = sum(1 for u, v in zip(na, nb) if u[:4] == v[:4]) if len(na) == len(nb) else 0
            if verbose:
                print("wdup   task=%d wt=%d rows %d -> %d, %d training predictions %s (largest difference %.2e, bound 1e-12; nodes %d and %d, %d agree in place)" % (task, kind, n, len(keep), n, "EQUAL" if ok else "DIFFERENT", worst, len(na), len(nb), agree))
            bad += not ok
    return bad


def unit_gate(exe, verbose=True):
    """G-unit (docs/weights.md WG2), measured and not asserted: with unit weights a classification forest uses a float score where the unweighted one uses an exact integer comparison;
    the number of the configurations whose N, E and P lines are equal is printed. The regression forests are asserted equal in `weighted`"""
    same_n = 0
    tot = 0
    for cid in ("E1", "E3"):
        c1, o1 = run(exe, CONFIGS[cid], "1000")
        c2, o2 = run(exe, CONFIGS[cid], "1000", extra=["wt=1"])
        tot += 1
        e = c1 == 0 and c2 == 0 and lines_of(o1) == lines_of(o2)
        same_n += e
        if verbose:
            print("wunit  %-3s classification, unit weights against the unweighted forest: %s" % (cid, "EQUAL" if e else "DIFFERENT (a float tie or rounding ordered two candidates differently)"))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ident", action="store_true"); ap.add_argument("--threads", action="store_true"); ap.add_argument("--spike", action="store_true")
    ap.add_argument("--unit", action="store_true"); ap.add_argument("--fuzz", action="store_true"); ap.add_argument("--exact", action="store_true"); ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--sizes", default="1000,100000"); ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--cancho", default=CANCHO); ap.add_argument("--src", default=ROOT)
    a = ap.parse_args()
    if not (a.ident or a.threads or a.spike or a.unit or a.fuzz or a.exact or a.weighted):
        a.ident = a.threads = a.spike = a.unit = a.fuzz = a.exact = a.weighted = True
    if not os.path.exists(TD):
        subprocess.run([sys.executable, f"{ROOT}/scripts/gen_tree_data.py"], check=True)
    os.makedirs(f"{ROOT}/build", exist_ok=True)
    exe = f"{ROOT}/build/tree-check"
    build(a.src, exe, a.cancho)
    bad = 0
    if a.unit:
        bad += unit(a.src, a.cancho)
    if a.exact:
        bad += exact(exe)
    if a.ident:
        bad += ident(exe, a.configs.split(","), a.sizes.split(","))
    if a.threads:
        bad += thread_gate(exe)
    if a.weighted:
        bad += weighted(exe)
        bad += dup_gate(exe)
        unit_gate(exe)
    if a.fuzz:
        bad += fuzz(exe)
    if a.spike:
        bad += spike_gate(a.cancho)
    print("GATE", "FAILED (%d)" % bad if bad else "passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

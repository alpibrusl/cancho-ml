#!/usr/bin/env python3
"""Timing for docs/spikes.md section 15 (protocol of 15.3 G-noise).

usage: bench_q.py N [--rounds 7] [--out FILE.jsonl]

Phase 1 (one thread): arms A, B, C and the reference N, interleaved, `rounds` rounds. Phase 2: B and C at threads 1,2,4,8,16
interleaved. Phase 3: the copy/spawn overhead (MUT=2: threads do no work), minimum of 3. The statistic is the minimum of the
rounds of `#fit_ms` (the fit only: data loading and decoding are outside); the spread is max / min. The 1-minute load average is read
before and after every sample; a cell is FLAGGED if any reading of its samples is >= 2. Every threaded sample's model bytes are
compared with the one-thread bytes of the same arm."""
import argparse, json, subprocess, sys, time, platform
import qlib

THREADS = [1, 2, 4, 8, 16]


def sample(n, arm, t, mut=0, w0="-", iters=6):
    l0 = qlib.load_avg()
    r = qlib.parse(qlib.run_q(n, arm, t, iters, mut, w0, 0))
    l1 = qlib.load_avg()
    return r, l0, l1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    n = a.n
    start_load = qlib.load_avg()
    print(f"# n={n} machine={platform.platform()} cores={subprocess.run(['sysctl','-n','hw.ncpu'],capture_output=True,text=True).stdout.strip()} load at start={start_load:.2f} compiler=a4572ea default backend")
    cells = {}

    def rec(key, r, l0, l1):
        c = cells.setdefault(key, dict(ms=[], loads=[], bytes=set()))
        c["ms"].append(r["fit_ms"])
        c["loads"] += [l0, l1]
        c["bytes"].add(qlib.model_bytes(r))

    # W0 for arm N: the final weights of arm A
    ra, _, _ = sample(n, 0, 1)
    w0 = "".join(ra["w"])
    for rd in range(a.rounds):
        for key in (("A", 1), ("B", 1), ("C", 1), ("N", 1)):
            arm = dict(A=0, B=1, C=2, N=3)[key[0]]
            r, l0, l1 = sample(n, arm, 1, 0, w0 if key[0] == "N" else "-")
            rec(key, r, l0, l1)
    for rd in range(a.rounds):
        for t in THREADS:
            for name, arm in (("B", 1), ("C", 2)):
                if t == 1:
                    r, l0, l1 = sample(n, arm, 1)
                    rec((name, 1, "phase2"), r, l0, l1)
                else:
                    r, l0, l1 = sample(n, arm, t)
                    rec((name, t), r, l0, l1)
    ovh = {}
    for t in THREADS[1:]:
        for name, arm in (("B", 1), ("C", 2)):
            ms = []
            for _ in range(3):
                r, l0, l1 = sample(n, arm, t, 2)
                ms.append(r["fit_ms"])
            ovh[(name, t)] = min(ms)
    end_load = qlib.load_avg()
    rows = []
    for key, c in cells.items():
        mn, mx = min(c["ms"]), max(c["ms"])
        rows.append(dict(n=n, cell="/".join(map(str, key)), min_ms=mn, max_ms=mx, spread=round(mx / mn, 3), samples=c["ms"], max_load=round(max(c["loads"]), 2),
                         flagged_load_ge_2=max(c["loads"]) >= 2, spread_gt_1_25=(mx / mn) > 1.25, bytes_distinct=len(c["bytes"])))
    for key, v in ovh.items():
        rows.append(dict(n=n, cell="overhead/" + "/".join(map(str, key)), min_ms=v))
    print(f"# load at end={end_load:.2f}")
    for r in rows:
        print(json.dumps(r))
    # bytes check across threads
    for name in ("B", "C"):
        bs = set()
        for t in THREADS:
            k = (name, t) if t > 1 else (name, 1)
            bs |= cells[k]["bytes"]
        bs |= cells[(name, 1, "phase2")]["bytes"]
        print(f"# bytes check arm {name}: distinct model byte strings over threads {THREADS} and all samples = {len(bs)}")
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(dict(header=True, n=n, start_load=start_load, end_load=end_load, time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")


main()

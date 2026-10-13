#!/usr/bin/env python3
"""Wall time of the threaded kernels at 1, 2, 4, 8, 16 threads (linalg-check), and whether the result hash is the same.
    python3 scripts/bench_threads.py [--samples N] [kernel:n:d ...]
Minimum of N runs (default 5) per point, each run `reps` repetitions chosen for >= 100 ms at one thread; the load average at the
start and the end of the table. The time includes the copy of each thread's input rows (cancho cannot share a slice with a
thread: docs/upstream/spawn_shared_slice.cho), and for the reductions the one-thread time is the sequential kernel."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import oracle_linalg as O

CELLS = ["dot:1000000:0", "gemv_t:1000000:64", "gemv_n:1000000:64", "syrk:100000:64", "syrk:100000:20", "gemm:512:0", "gemm:1024:0"]
THREADS = [1, 2, 4, 8, 16]

def main():
    a = sys.argv[1:]
    samples = int(a[a.index("--samples") + 1]) if "--samples" in a else 5
    cells = [x for i, x in enumerate(a) if ":" in x] or CELLS
    print("load start %.1f; machine has %d cores; min of %d runs" % (os.getloadavg()[0], os.cpu_count(), samples))
    for c in cells:
        k, n, d = c.split(":"); n, d = int(n), int(d)
        rc = 1
        while True:
            r = O.run_cancho(k, n, d, 1, rc, 1)
            if int(r["ms"]) >= 100 or rc > 1 << 20: break
            rc *= 2
        row, hashes = [], set()
        for t in THREADS:
            best = None
            for _ in range(samples):
                r = O.run_cancho(k, n, d, 1, rc, t)
                ms = int(r["ms"]) / rc
                best = ms if best is None else min(best, ms)
                hashes.add(r["hash"])
            row.append(best)
        print("%-7s n=%-8d d=%-3d reps=%-4d " % (k, n, d, rc) + "  ".join("t=%d %.3f ms (%.2fx)" % (t, ms, row[0] / ms) for t, ms in zip(THREADS, row)) + "   hashes: %d distinct" % len(hashes), flush=True)
    print("load end %.1f" % os.getloadavg()[0])

if __name__ == "__main__":
    main()

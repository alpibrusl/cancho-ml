#!/usr/bin/env python3
"""RD9 (docs/reader.md section 5): cells per second of the reader of this repository and of cancho-table's own `table` on the SAME file,
one thread, minimum of N interleaved runs, the 1-minute load average recorded before and after every round.

    python3 scripts/bench_reader.py --rows 500000 --runs 7 [--table PATH ...] [--reader PATH ...] [--keep DIR]

What is timed (wall time of the whole process, `time.perf_counter` around subprocess.run; stdout is a few lines):
  reader-check --mode shape   pass 1 of ml.csv.scan: every cell of every kept column is read (ml.csvcell.read), nothing is stored
  reader-check --mode hash    the whole load: pass 1, the matrix, pass 2 (every cell read a second time and stored), one hash
  table --agg max:c0:float,...,max:c7:float   cancho-table reads every cell of the 8 columns with flt.parse_float and keeps the maximum
  table --agg max:c0:float                    one column parsed of 8 (the other cells are only scanned past), for the per-parsed-cell figure
  table --agg count                           no cell parsed: the cost of the record reader alone
Cells per second = rows * 8 / seconds for every line (cells scanned); the parsed-cell rate is given where fewer cells are parsed.
The file: `kind` float17 (a random double printed with 17 significant digits), short (two decimals, the int/dec path), mixed.
Nothing here is a claim about another library: the only comparison is between binaries run by this script on this machine.
"""
import os
import pathlib
import random
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load():
    return os.getloadavg()[0]


def make_file(path, rows, kind, seed=1):
    rnd = random.Random(seed)
    with open(path, "w") as f:
        f.write(",".join("c%d" % j for j in range(8)) + "\n")
        buf = []
        for _ in range(rows):
            cells = []
            for j in range(8):
                if kind == "float17":
                    x = rnd.uniform(-1, 1) * 10 ** rnd.randint(-20, 20)
                    cells.append("%.16e" % x)
                elif kind == "short":
                    cells.append("%.2f" % rnd.uniform(-1000, 1000))
                else:
                    r = rnd.random()
                    cells.append("%.16e" % (rnd.uniform(-1, 1) * 10 ** rnd.randint(-20, 20)) if r < 0.5 else ("%.2f" % rnd.uniform(-1000, 1000) if r < 0.9 else str(rnd.randint(-10 ** 6, 10 ** 6))))
            buf.append(",".join(cells))
            if len(buf) >= 20000:
                f.write("\n".join(buf) + "\n")
                buf = []
        if buf:
            f.write("\n".join(buf) + "\n")


def main():
    argv = sys.argv[1:]

    def opt(name, default):
        return type(default)(argv[argv.index(name) + 1]) if name in argv else default

    def multi(name):
        return [argv[i + 1] for i, a in enumerate(argv) if a == name]

    rows = opt("--rows", 500000)
    runs = opt("--runs", 7)
    keep = pathlib.Path(opt("--keep", str(ROOT / "build" / "bench")))
    keep.mkdir(parents=True, exist_ok=True)
    readers = multi("--reader") or [str(ROOT / "build" / "reader-check")]
    tables = multi("--table")
    for kind in ("float17", "short", "mixed"):
        path = keep / ("%s_%d.csv" % (kind, rows))
        if not path.exists():
            make_file(path, rows, kind)
        size = os.path.getsize(path)
        cases = []
        for r in readers:
            tag = "reader-check[%s]" % pathlib.Path(r).name
            cases.append((tag + " shape (pass 1)", [r, "--mode", "shape", "--max-rows", str(rows + 10), str(path)], 8 * rows, 8 * rows))
            cases.append((tag + " hash (whole load)", [r, "--mode", "hash", "--max-rows", str(rows + 10), "--max-cells", "400000000", str(path)], 8 * rows, 8 * rows))
        for t in tables:
            tag = "table[%s]" % pathlib.Path(t).name
            cases.append((tag + " max:c0..c7:float", [t, "--agg", ",".join("max:c%d:float" % j for j in range(8)), "--max-rows", str(rows + 10), str(path)], 8 * rows, 8 * rows))
            cases.append((tag + " max:c0:float", [t, "--agg", "max:c0:float", "--max-rows", str(rows + 10), str(path)], 8 * rows, rows))
            cases.append((tag + " count", [t, "--agg", "count", "--max-rows", str(rows + 10), str(path)], 8 * rows, 0))
        times = {c[0]: [] for c in cases}
        loads = []
        l0 = load()
        for rnd in range(runs):
            la = load()
            for name, argv_, scanned, parsed in cases:
                t0 = time.perf_counter()
                p = subprocess.run(argv_, capture_output=True)
                dt = time.perf_counter() - t0
                if p.returncode not in (0,):
                    print("   !! %s exited %d: %s" % (name, p.returncode, p.stdout[:120]))
                times[name].append(dt)
            loads.append((la, load()))
        l1 = load()
        print("file %s: %d rows x 8 columns = %.1f M cells, %.1f MB; machine %s; load %.2f before, %.2f after; %d interleaved rounds, one thread" %
              (path.name, rows, 8 * rows / 1e6, size / 1e6, os.uname().machine, l0, l1, runs))
        print("   load per round (before, after): %s" % " ".join("%.1f/%.1f" % x for x in loads))
        for name, argv_, scanned, parsed in cases:
            ts = times[name]
            best = min(ts)
            med = statistics.median(ts)
            line = "   %-52s min %.3f s  median %.3f s  %6.1f M cells/s (min)" % (name, best, med, scanned / best / 1e6)
            if parsed and parsed != scanned:
                line += "  (%.1f M parsed cells/s)" % (parsed / best / 1e6)
            print(line)


if __name__ == "__main__":
    main()

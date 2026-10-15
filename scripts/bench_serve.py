#!/usr/bin/env python3
"""L1 (docs/serving-http.md section 9): a small load test of `ml-serve`: requests per second and latency percentiles, with the machine, the load and N, and no claim beyond the measurement.

    python3 scripts/bench_serve.py [--n N] [--repeat R] [--json FILE]

Scenarios, each on a keep-alive connection per client thread, each client sending its next request when the answer to the last has come (closed loop):

* `health`: GET /health, the cost of the loop and the transport with no model work;
* `one`: POST /predict with one record, the linear model of the serving corpus (60 training rows, 645 bytes);
* `batch100`: POST /predict/batch with 100 records, the same model;
* `one-forest`: POST /predict with one record on a forest of 200 trees, depth 12, fitted on 4,000 rows (the model file is megabytes: the decode of the model is made for each request, docs/serving-http.md section 7).

at 1, 4 and 16 client threads. N is the number of requests per client thread in each run (the default is small; the scenario says how many). The best of R repeats by throughput is printed with the others in
the JSON. The client is Python threads over sockets, so the client takes part of the time, and the server is one thread: a figure here is a figure of this client against this server on this machine, with
this load, and nothing more. The script also prints the resident set of the server before and after, and the time of one `ml predict` process on the same record, for scale.
"""

import json
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from serve_lib import Conn, Server  # noqa: E402
from harness import Sandbox, BIN  # noqa: E402
import serving_corpus as C  # noqa: E402

REC = b'{"a":3,"b":2.50,"c":0.5}'
JSON = {"Content-Type": "application/json"}


def percentile(sorted_ms, q):
    if not sorted_ms:
        return 0.0
    i = min(len(sorted_ms) - 1, int(q * len(sorted_ms)))
    return sorted_ms[i]


def run_once(port, request, clients, n):
    lat = [[] for _ in range(clients)]
    errors = []
    barrier = threading.Barrier(clients + 1)

    def work(k):
        c = Conn(port, 30)
        try:
            barrier.wait()
            for _ in range(n):
                t0 = time.perf_counter()
                c.send(request)
                r = c.read_response()
                lat[k].append((time.perf_counter() - t0) * 1000)
                if r.status != 200:
                    errors.append(r.status)
        finally:
            c.close()

    ts = [threading.Thread(target=work, args=(k,)) for k in range(clients)]
    for t in ts:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    allms = sorted(x for l in lat for x in l)
    return {"clients": clients, "requests": len(allms), "seconds": wall, "rps": len(allms) / wall, "p50_ms": percentile(allms, 0.50), "p90_ms": percentile(allms, 0.90),
            "p99_ms": percentile(allms, 0.99), "max_ms": allms[-1] if allms else 0.0, "errors": len(errors)}


def rss(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) if out else 0


def machine():
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "cpus": os.cpu_count(), "loadavg_before": os.getloadavg()}
    try:
        info["model"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return info


def big_forest(sb):
    rng = random.Random(5)
    rows = ["a,b,c,y"]
    for i in range(4000):
        a = rng.randint(-20, 40)
        b = rng.randint(-5000, 5000) / 100.0
        c = rng.uniform(-3, 3)
        rows.append("%d,%.2f,%r,%.1f" % (a, b, c, round(1.5 * a - 0.7 * b + 4 * c + rng.uniform(-2, 2), 1)))
    sb.write("big.csv", "\n".join(rows) + "\n")
    r = sb.run("fit", "--model", "forest", "--task", "regression", "--trees", "200", "--max-depth", "12", "--seed", "3", "--target", "y", "--threads", "4", "--max-bytes", "134217728", "--output", "bigforest.cml", "big.csv")
    assert r.status == 0, r
    return "bigforest.cml", len(sb.read("bigforest.cml"))


def main():
    args = sys.argv[1:]
    n = int(args[args.index("--n") + 1]) if "--n" in args else 1000
    repeat = int(args[args.index("--repeat") + 1]) if "--repeat" in args else 3
    out = {"machine": machine(), "n_per_client": n, "repeat": repeat, "scenarios": {}}
    sb = Sandbox()
    try:
        sb.write("num.csv", C.num_csv(random.Random(17)))
        r = sb.run("fit", "--model", "linear", "--l2", "0.5", "--step", "impute column=* strategy=mean;scale column=a,b kind=standard", "--target", "y", "--drop", "lab", "--output", "linear.cml", "num.csv")
        assert r.status == 0, r
        forest, size = big_forest(sb)
        batch = b'{"rows":[' + b",".join([REC] * 100) + b"]}"
        scenarios = [
            ("health", "linear.cml", Conn.build("GET", "/health"), [1, 4, 16], n),
            ("one", "linear.cml", Conn.build("POST", "/predict", body=REC, headers=JSON), [1, 4, 16], n),
            ("batch100", "linear.cml", Conn.build("POST", "/predict/batch", body=batch, headers=JSON), [1, 4], max(10, n // 10)),
            ("one-forest", forest, Conn.build("POST", "/predict", body=REC, headers=JSON), [1, 4], max(10, n // 10)),
        ]
        for name, model, request, levels, count in scenarios:
            with Server(sb.dir, model, "--max-batch-rows", "100", "--max-bytes", "134217728") as srv:
                rss0 = rss(srv.proc.pid)
                runs = []
                for clients in levels:
                    reps = [run_once(srv.port, request, clients, count) for _ in range(repeat)]
                    best = max(reps, key=lambda x: x["rps"])
                    runs.append({"best": best, "all_rps": [round(x["rps"], 1) for x in reps]})
                out["scenarios"][name] = {"model": model, "model_bytes": len(sb.read(model)), "request_bytes": len(request), "n_per_client": count, "runs": runs, "rss_kib_before": rss0, "rss_kib_after": rss(srv.proc.pid)}
        # one process of `ml predict` on one record, for scale
        sb.write("rec.csv", "a,b,c\n3,2.50,0.5\n")
        t = []
        for _ in range(30):
            t0 = time.perf_counter()
            subprocess.run([BIN, "--root", str(sb.dir), "predict", "--model-file", "linear.cml", "rec.csv"], capture_output=True)
            t.append((time.perf_counter() - t0) * 1000)
        out["cli_process_ms"] = {"n": len(t), "min": min(t), "median": statistics.median(t)}
        out["machine"]["loadavg_after"] = os.getloadavg()
    finally:
        sb.cleanup()
    print("machine: %s, %s, %d cpus, python %s; load average before %s, after %s" % (out["machine"].get("model", "?"), out["machine"]["platform"], out["machine"]["cpus"], out["machine"]["python"],
          tuple(round(x, 2) for x in out["machine"]["loadavg_before"]), tuple(round(x, 2) for x in out["machine"]["loadavg_after"])))
    print("%-11s %8s %8s %9s %9s %8s %8s %8s %8s %6s" % ("scenario", "clients", "N", "req/s", "p50 ms", "p90 ms", "p99 ms", "max ms", "model B", "errors"))
    for name, s in out["scenarios"].items():
        for run in s["runs"]:
            b = run["best"]
            print("%-11s %8d %8d %9.0f %9.3f %8.3f %8.3f %8.3f %8d %6d" % (name, b["clients"], b["requests"], b["rps"], b["p50_ms"], b["p90_ms"], b["p99_ms"], b["max_ms"], s["model_bytes"], b["errors"]))
        print("            resident set of the server: %d KiB before, %d KiB after" % (s["rss_kib_before"], s["rss_kib_after"]))
    print("one `ml predict` process on one record: minimum %.1f ms, median %.1f ms over %d runs" % (out["cli_process_ms"]["min"], out["cli_process_ms"]["median"], out["cli_process_ms"]["n"]))
    if "--json" in args:
        pathlib = __import__("pathlib")
        pathlib.Path(args[args.index("--json") + 1]).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

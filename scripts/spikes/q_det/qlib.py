"""Shared helpers of the Q-det scripts (docs/spikes.md section 15). Paths with `..` are refused by the cancho runtime (it traps), so
every path handed to a binary is absolute."""
import os, subprocess, sys, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = pathlib.Path(os.environ.get("ML_DATA", ROOT.parent / "ml-data")).resolve()
SPIKE_Q = os.environ.get("Q_BIN", str(ROOT / "build" / "spike_q"))
SPIKE_S = str(ROOT / "build" / "spike_s")
NTRAIN = lambda n: n - n // 5


def datadir(n):
    return DATA / f"n{n}"


def run_q(n, arm, threads=1, iters=6, mut=0, w0="-", out=0, raw=False):
    d = datadir(n)
    p = subprocess.run([SPIKE_Q, str(d / "x.bin"), str(d / "y.bin"), str(n), str(arm), str(threads), str(iters), str(mut), w0, str(out)],
                       capture_output=True, check=True)
    return p.stdout if raw else p.stdout.decode()


def parse(text):
    """-> dict(w=[21 hex], loss=[hex], sums=[253 hex], fit_ms=int)"""
    r = dict(w=[], loss=[], sums=[], fit_ms=None, load_ms=None)
    for l in text.splitlines():
        if l.startswith("#loss "):
            r["loss"].append(l.split()[1])
        elif l.startswith("#sum "):
            r["sums"].append(l.split()[1])
        elif l.startswith("#fit_ms"):
            r["fit_ms"] = int(l.split()[1])
        elif l.startswith("#load_ms"):
            r["load_ms"] = int(l.split()[1])
        elif l.startswith("#"):
            raise RuntimeError(l)
        elif l.strip():
            r["w"].append(l.strip())
    return r


def model_bytes(r):
    """the model: the 21 coefficients and the loss lines"""
    return " ".join(r["w"]) + "|" + " ".join(r["loss"])


def hex_to_float(h):
    return np.frombuffer(int(h, 16).to_bytes(8, "little"), dtype="<f8")[0]


def hexes_to_array(hs):
    return np.array([hex_to_float(h) for h in hs])


def ulps(a, b):
    """distance in units in the last place between two doubles of the same sign class (ordered-integer map)"""
    def key(x):
        i = int(np.float64(x).view(np.int64))
        return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)
    return abs(key(a) - key(b))


def load_avg():
    return os.getloadavg()[0]

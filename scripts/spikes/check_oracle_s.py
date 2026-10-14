#!/usr/bin/env python3
"""G-oracle-S, G-mutants-S and G-metric-S of docs/spikes.md section 10.

usage: check_oracle_s.py --cancho CANCHO_COMPILER [--data DIR] [--sets n1000 n100000] [--extra n1000000] [--skip-mutants]

1. G-oracle-S: spike_s (Newton K=6; L-BFGS K=8 m=5) against oracle_s.py, 1e-9 relative per coefficient; plus the stress case
   (L-BFGS T0E=10 on the first set, which must make the oracle backtrack).
2. G-mutants-S: each mutant (one edit of spike_s.cho, rebuilt, or a changed flag) is compared with the UNMUTATED oracle; every
   one must be rejected at the sizes listed, except those marked expected survivors, which are reported as such.
3. G-metric-S: |test AUC diff| <= max(1e-5, one swapped test pair = 1/(n_pos n_neg)) and |train log-loss diff| <= 1e-6 against scikit-learn (C=inf, lbfgs, max_iter=1000).
Exit 1 if any gate fails or a must-reject mutant survives."""
import argparse, os, struct, subprocess, sys, tempfile, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oracle_l, oracle_s
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

TOL = 1e-9; AUC_TOL = 1e-5; LOSS_TOL = 1e-6
NEWTON = dict(algo=0, iters=6, m=0, trials=0, t0e=0)
LBFGS = dict(algo=1, iters=8, m=5, trials=10, t0e=0)
STRESS = dict(LBFGS, t0e=10)
SRC = os.path.join(HERE, "spike_s.cho")

def bits_of(x): return struct.unpack("<Q", struct.pack("<d", x))[0]
def of_bits(u): return struct.unpack("<d", struct.pack("<Q", u))[0]

def build(compiler, src, out):
    subprocess.run([compiler, "build", src, "--std", "-o", out], check=True, capture_output=True, text=True)

def run_cancho(exe, data, name, p):
    d = os.path.join(data, name); n = int(name[1:])
    out = subprocess.run([exe, d + "/x.bin", d + "/y.bin", str(n)] + [str(p[k]) for k in ("algo", "iters", "m", "trials", "t0e")],
                         capture_output=True, text=True, check=True).stdout
    lines = [l for l in out.splitlines() if not l.startswith("#")]
    if len(lines) != 21 or "#error" in out:
        return None, {}
    info = {l.split()[0][1:]: int(l.split()[1]) for l in out.splitlines() if l.startswith("#")}
    return np.array([of_bits(int(l, 16)) for l in lines]), info

_cache = {}
def oracle(data, name, p):
    key = (name,) + tuple(sorted(p.items()))
    if key not in _cache:
        d = os.path.join(data, name); X, y = oracle_l.load(d); ntr = oracle_l.split(y.size)
        w, passes, info = oracle_s.solve(oracle_s.augment(X[:ntr]), y[:ntr], "newton" if p["algo"] == 0 else "lbfgs", p["iters"], p["m"] or 10, p["trials"] or 10, 1e-4, p["t0e"])
        _cache[key] = (w, passes, info)
    return _cache[key]

def compare(c, o):
    if c is None: return False, float("inf")
    worst, ok = 0.0, True
    for a, b in zip(c, o):
        rel = 0.0 if a == b else (float("inf") if b == 0 or not np.isfinite(a) else abs(a - b) / abs(b))
        if not rel <= TOL: ok = False
        worst = max(worst, rel)
    return ok, worst

MUT = [  # name, kind, edit (old, new) or flag change, cases, expected survivor?
 ("N: one fewer iteration", dict(NEWTON, iters=5), None, "N", False),
 ("N: s = p instead of p(1-p)", NEWTON, ("let s = p * (1.0 - p);", "let s = p;"), "N", False),
 ("N: update w + d", NEWTON, ("ws[j] = ws[j] - ws[945 + j];", "ws[j] = ws[j] + ws[945 + j];"), "N", False),
 ("N: update w - 0.9 d", NEWTON, ("ws[j] = ws[j] - ws[945 + j];", "ws[j] = ws[j] - 0.9 * ws[945 + j];"), "N", False),
 ("N: back substitution reads L[i,k]", NEWTON, ("s = s - ws[483 + k * 21 + i] * ws[945 + k];", "s = s - ws[483 + i * 21 + k] * ws[945 + k];"), "N", False),
 ("N: one MORE iteration (K=7)", dict(NEWTON, iters=7), None, "N", True),
 ("L: one fewer iteration", dict(LBFGS, iters=7), None, "L", False),
 ("L: m = 4", dict(LBFGS, m=4), None, "L", False),
 ("L: m = 6", dict(LBFGS, m=6), None, "L", False),
 ("L: alpha + beta", LBFGS, ("let c = ws[ao + slot] - beta;", "let c = ws[ao + slot] + beta;"), "L", False),
 ("L: gamma scaling dropped", LBFGS, ("ws[105 + j] = ws[105 + j] * gamma;", "ws[105 + j] = ws[105 + j] * 1.0;"), "L", False),
 ("L: c1 = 0.5 (stress)", STRESS, ("let c1 = 1.0 / 10000.0;", "let c1 = 0.5;"), "S", False),
 ("L: halving factor 0.6 (stress)", STRESS, ("t = t * 0.5;", "t = t * 0.6;"), "S", False),
 ("L: trials = 1 (stress)", dict(STRESS, trials=1), None, "S", False),
 ("L: trials = 9 (stress)", dict(STRESS, trials=9), None, "S", True),
 ("L: s.y > 0 test removed", LBFGS, ("if sy > 0.0 {", "if sy > 0.0 - 1.0e300 {"), "L", True),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancho", required=True)
    ap.add_argument("--data", default="/Users/alfonso/Workspace/alpibrusl/lexsys/ml-data")
    ap.add_argument("--sets", nargs="+", default=["n1000", "n100000"])
    ap.add_argument("--extra", nargs="*", default=["n1000000"])
    ap.add_argument("--skip-mutants", action="store_true")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="spike_s_")
    exe = os.path.join(tmp, "spike_s"); build(a.cancho, SRC, exe)
    fail = False

    print("== G-oracle-S (tolerance %g relative)" % TOL)
    cases = [("Newton K=6", NEWTON, a.sets + a.extra), ("L-BFGS K=8 m=5", LBFGS, a.sets + a.extra), ("L-BFGS stress T0E=10", STRESS, a.sets[:1] + a.sets[1:])]
    for label, p, sets in cases:
        for s in sets:
            c, info = run_cancho(exe, a.data, s, p); o, passes, oinfo = oracle(a.data, s, p)
            ok, rel = compare(c, o)
            extra = ""
            if label.endswith("T0E=10"):
                extra = "  oracle backtracks %s cancho backtracks %s" % (oinfo.get("backtracks"), info.get("backtracks"))
                if not oinfo.get("backtracks"): ok = False; extra += "  STRESS CASE VOID (no backtrack)"
            if passes != info.get("passes") or oinfo.get("backtracks", 0) != info.get("backtracks", 0): ok = False; extra += "  PASS COUNT DIFFERS (oracle %d, cancho %s)" % (passes, info.get("passes"))
            print("  %-22s %-9s %s worst rel %.3e passes %s%s" % (label, s, "PASS" if ok else "FAIL", rel, info.get("passes"), extra))
            fail |= not ok

    if not a.skip_mutants:
        print("== G-mutants-S (all compared with the unmutated oracle; rejected = good)")
        for name, p, edit, grp, expected in MUT:
            e2 = exe
            if edit:
                src = open(SRC).read()
                if src.count(edit[0]) != 1: sys.exit("mutant anchor not unique/absent: " + edit[0])
                msrc = os.path.join(tmp, "m.cho"); open(msrc, "w").write(src.replace(edit[0], edit[1]))
                e2 = os.path.join(tmp, "mut"); build(a.cancho, msrc, e2)
            base = {"N": NEWTON, "L": LBFGS, "S": STRESS}[grp]
            res = []
            for s in a.sets:
                c, _ = run_cancho(e2, a.data, s, p); o, _, _ = oracle(a.data, s, base)
                ok, rel = compare(c, o); res.append((s, ok, rel))
            survived_all = all(ok for _, ok, _ in res)
            verdict = "REJECTED" if not survived_all else "survived"
            flag = ""
            if survived_all and not expected: flag = "  <-- UNEXPECTED SURVIVOR: gate void"; fail = True
            if (not survived_all) and expected: flag = "  (expected survivor was rejected)"
            print("  %-36s %-8s %s%s" % (name, verdict, "  ".join("%s %.1e" % (s, r) for s, _, r in res), flag))

    print("== G-metric-S vs scikit-learn (|dAUC| <= %g, |dLogLoss| <= %g)" % (AUC_TOL, LOSS_TOL))
    for s in a.sets + a.extra:
        d = os.path.join(a.data, s); X, y = oracle_l.load(d); ntr = oracle_l.split(y.size)
        A = oracle_s.augment(X[:ntr]); At = oracle_s.augment(X[ntr:])
        npos = float(y[ntr:].sum()); auc_tol = max(AUC_TOL, 1.0001 / (npos * (y.size - ntr - npos)))  # one swapped pair is the AUC's granularity
        mk = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000).fit(X[:ntr], y[:ntr])
        wk = np.r_[mk.coef_[0], mk.intercept_]
        wopt, _ = oracle_s.newton(A, y[:ntr], 12)
        fk = oracle_s.loss_grad(A, y[:ntr], wk)[0]; fo = oracle_s.loss_grad(A, y[:ntr], wopt)[0]; ak = roc_auc_score(y[ntr:], At @ wk); ao = roc_auc_score(y[ntr:], At @ wopt)
        for label, p in (("Newton", NEWTON), ("L-BFGS", LBFGS)):
            c, info = run_cancho(exe, a.data, s, p)
            fc = oracle_s.loss_grad(A, y[:ntr], c)[0]; ac = roc_auc_score(y[ntr:], At @ c)
            ok = abs(ac - ak) <= auc_tol and abs(fc - fk) <= LOSS_TOL
            print("  %-7s %-9s %s  test AUC %.8f (sklearn %.8f, K=12 Newton %.8f) dAUC %.2e | train logloss %.10f (sklearn %.10f, K=12 Newton %.10f) dLL %.2e | passes %s" % (label, s, "PASS" if ok else "FAIL", ac, ak, ao, ac - ak, fc, fk, fo, fc - fk, info["passes"]))
            fail |= not ok
    print("RESULT:", "FAIL" if fail else "all gates pass")
    sys.exit(1 if fail else 0)

if __name__ == "__main__":
    main()

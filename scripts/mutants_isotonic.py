#!/usr/bin/env python3
"""Source mutants of the isotonic fit (docs/calibration.md section 5): lib/metrics/isotonic.cho. Each is a one-place edit applied to a copy of the tree; scripts/gate_isotonic.sh (the unit tests worked by hand, the driver
against the independent oracles, the refusals) must fail on it.

    python3 scripts/mutants_isotonic.py --check          every pattern applies exactly once (no compiler)
    python3 scripts/mutants_isotonic.py [name-substring ...]
Survivors are listed with a reason in ALLOWED; an unexplained survivor fails the script."""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import mutlib

C = "lib/metrics/isotonic.cho"
M = [
    ("violation-strict", C, "var violation = below > top;", "var violation = below >= top;"),
    ("violation-direction", C, "var violation = below > top;", "var violation = below < top;"),
    ("decreasing-direction", C, "violation = below < top;", "violation = below > top;"),
    ("merge-sum", C, "contents(bsw)[blocks - 2] = contents(bsw)[blocks - 2] + contents(bsw)[blocks - 1];", "contents(bsw)[blocks - 2] = contents(bsw)[blocks - 2] - contents(bsw)[blocks - 1];"),
    ("merge-weight", C, "contents(bww)[blocks - 2] = contents(bww)[blocks - 2] + contents(bww)[blocks - 1];", "contents(bww)[blocks - 2] = contents(bww)[blocks - 2] + 1.0;"),
    ("merge-count", C, "contents(bcw)[blocks - 2] = contents(bcw)[blocks - 2] + contents(bcw)[blocks - 1];", "contents(bcw)[blocks - 2] = contents(bcw)[blocks - 2] + 1;"),
    ("merge-pop", C, "                                                blocks = blocks - 1;\n                                            } else {", "                                                blocks = blocks - 0;\n                                            } else {"),
    ("merge-loop", C, "while going && blocks > 1 {", "while going && blocks > 2 {"),
    ("block-mean-below", C, "let below = contents(bsw)[blocks - 2] / contents(bww)[blocks - 2];", "let below = contents(bsw)[blocks - 2];"),
    ("block-mean-top", C, "let top = contents(bsw)[blocks - 1] / contents(bww)[blocks - 1];", "let top = contents(bsw)[blocks - 1] / contents(bww)[blocks - 2];"),
    ("value-weight", C, "var swy = ws[r] * ys[r];", "var swy = ys[r];"),
    ("tie-value", C, "swy = swy + ws[q] * ys[q];", "swy = swy + ys[q];"),
    ("tie-weight", C, "sw = sw + ws[q];", "sw = sw + 1.0;"),
    ("tie-test", C, "while j < n && xs[contents(iw)[j]] == xs[r] {", "while j < n && xs[contents(iw)[j]] <= xs[r] {"),
    ("tie-pooling-off", C, "while j < n && xs[contents(iw)[j]] == xs[r] {", "while j < i && xs[contents(iw)[j]] == xs[r] {"),
    ("output-block", C, "out_block[at] = b;", "out_block[at] = b + 1;"),
    ("output-x", C, "out_x[at] = contents(pxw)[at];", "out_x[at] = contents(pww)[at];"),
    ("output-weight", C, "out_w[at] = contents(pww)[at];", "out_w[at] = 1.0;"),
    ("output-fit", C, "out_fit[at] = value;", "out_fit[at] = contents(pvw)[at];"),
    ("weight-positive", C, "if ws[i] <= 0.0 {", "if ws[i] < 0.0 {"),
    ("weight-length", C, "if len(ws) != n {", "if len(ws) > n {"),
    ("weight-finite", C, "regress.all_finite(xs) == false || regress.all_finite(ys) == false || regress.all_finite(ws) == false", "regress.all_finite(xs) == false || regress.all_finite(ys) == false"),
    ("x-finite", C, "regress.all_finite(xs) == false || regress.all_finite(ys) == false || regress.all_finite(ws) == false", "regress.all_finite(ys) == false || regress.all_finite(ws) == false"),
    ("info-blocks", C, "info[1] = float_of(blocks);", "info[1] = float_of(points);"),
    ("info-rows", C, "info[2] = float_of(n);", "info[2] = float_of(points);"),
]
ALLOWED = {
    "tie-test": "equivalent: the rows are sorted by x ascending, so every later row has x >= the first of the group, and `<=` selects exactly the rows that `==` selects",
}

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if args:
    M = [m for m in M if any(a in m[0] for a in args)]
sys.exit(mutlib.run(M, ["sh", "scripts/gate_isotonic.sh"], "mutants_isotonic", allowed=ALLOWED, timeout=900))

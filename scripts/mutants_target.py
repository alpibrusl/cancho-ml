#!/usr/bin/env python3
"""Source mutants of the out-of-fold target encoding (docs/calibration.md section 5): lib/metrics/targetenc.cho. Each is a one-place edit applied to a copy of the tree; scripts/gate_target.sh (the unit tests worked by hand, the driver
against the independent oracles, the refusals) must fail on it.

    python3 scripts/mutants_target.py --check          every pattern applies exactly once (no compiler)
    python3 scripts/mutants_target.py [name-substring ...]
Survivors are listed with a reason in ALLOWED; an unexplained survivor fails the script."""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import mutlib

C = "lib/metrics/targetenc.cho"
M = [
    ("count-positives", C, "total_b = total_b + label[i];", "total_b = total_b + 1;"),
    ("count-fit", C, "total_n = total_n + 1;", "total_n = total_n + 2;"),
    ("count-fit-only", C, "                if fit[i] == 1 {\n                    total_n = total_n + 1;", "                if fit[i] == 0 {\n                    total_n = total_n + 1;"),
    ("count-vocabulary", C, "                    if cat[i] >= 0 {\n                        let f = (fold[i] % folds + folds) % folds;\n                        contents(nw)", "                    if cat[i] >= 1 {\n                        let f = (fold[i] % folds + folds) % folds;\n                        contents(nw)"),
    ("fold-negative-count", C, "let f = (fold[i] % folds + folds) % folds;\n                        contents(nw)", "let f = fold[i] % folds;\n                        contents(nw)"),
    ("fold-negative-row", C, "let f = (fold[i] % folds + folds) % folds;\n                    let nn", "let f = fold[i] % folds;\n                    let nn"),
    ("fold-modulus", C, "let f = (fold[i] % folds + folds) % folds;\n                    let nn", "let f = (fold[i] % (folds + 1) + folds + 1) % (folds + 1);\n                    let nn"),
    ("p0-floor", C, "            if den < 1 {\n                den = 1;", "            if den < 0 {\n                den = 1;"),
    ("p0-denominator", C, "p0 = float_of(total_b) / float_of(den);", "p0 = float_of(total_b) / float_of(den + 1);"),
    ("category-te", C, "cat_te[cc] = (float_of(sb) + k * p0) / (float_of(sn) + k);", "cat_te[cc] = (float_of(sb) + k * p0) / float_of(sn + 1);"),
    ("category-logn", C, "cat_logn[cc] = math.log1p(float_of(sn));", "cat_logn[cc] = math.log1p(float_of(sn) + 1.0);"),
    ("category-sum", C, "sn = sn + contents(nw)[f * ncat + cc];", "sn = sn + contents(nw)[cc];"),
    ("row-out-of-fold-n", C, "let nn = cat_n[cat[i]] - contents(nw)[f * ncat + cat[i]];", "let nn = cat_n[cat[i]];"),
    ("row-out-of-fold-b", C, "let bb = cat_b[cat[i]] - contents(bw)[f * ncat + cat[i]];", "let bb = cat_b[cat[i]];"),
    ("row-prior", C, "te[i] = (float_of(bb) + k * p0) / (float_of(nn) + k);", "te[i] = (float_of(bb) + p0) / (float_of(nn) + k);"),
    ("row-logn", C, "logn[i] = math.log1p(float_of(nn));", "logn[i] = math.log1p(float_of(nn + 1));"),
    ("outside-te", C, "                    te[i] = p0;\n                    logn[i] = 0.0;", "                    te[i] = 0.0;\n                    logn[i] = 0.0;"),
    ("outside-logn", C, "                    te[i] = p0;\n                    logn[i] = 0.0;", "                    te[i] = p0;\n                    logn[i] = 1.0;"),
    ("check-length", C, "if len(fit) != n || len(fold) != n {", "if len(fit) != n {"),
    ("check-ncat-low", C, "if ncat < 1 || ncat > max_categories()", "if ncat < 0 || ncat > max_categories()"),
    ("check-folds-low", C, "folds < 2 || folds > max_folds()", "folds < 1 || folds > max_folds()"),
    ("check-folds-high", C, "folds < 2 || folds > max_folds()", "folds < 2 || folds > max_folds() + 1"),
    ("check-k", C, "!(k > 0.0) || k > 1e300", "k < 0.0 || k > 1e300"),
    ("check-code-low", C, "if cat[i] < 0 - 1 || cat[i] >= ncat {", "if cat[i] < 0 - 2 || cat[i] >= ncat {"),
    ("check-code-high", C, "if cat[i] < 0 - 1 || cat[i] >= ncat {", "if cat[i] < 0 - 1 || cat[i] > ncat {"),
    ("check-fit-flag", C, "if fit[i] != 0 && fit[i] != 1 {", "if fit[i] != 0 && fit[i] != 1 && fit[i] != 2 {"),
    ("check-label-fit", C, "if fit[i] == 1 && label[i] != 0 && label[i] != 1 {", "if label[i] != 0 && label[i] != 1 {"),
    ("check-label", C, "if fit[i] == 1 && label[i] != 0 && label[i] != 1 {", "if fit[i] == 1 && label[i] != 0 {"),
    ("info-fit", C, "info[1] = float_of(total_n);", "info[1] = float_of(n);"),
    ("info-positives", C, "info[2] = float_of(total_b);", "info[2] = float_of(total_n);"),
]
ALLOWED = {
}

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if args:
    M = [m for m in M if any(a in m[0] for a in args)]
sys.exit(mutlib.run(M, ["sh", "scripts/gate_target.sh"], "mutants_target", allowed=ALLOWED, timeout=900))

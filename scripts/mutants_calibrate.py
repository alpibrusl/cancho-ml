#!/usr/bin/env python3
"""Source mutants of the calibration table (docs/calibration.md section 5): lib/metrics/calibrate.cho. Each is a one-place edit applied to a copy of the tree; scripts/gate_calibrate.sh (the unit test worked by hand, the driver
against the independent oracle, the refusals) must fail on it.

    python3 scripts/mutants_calibrate.py --check          every pattern applies exactly once (no compiler)
    python3 scripts/mutants_calibrate.py [name-substring ...]
Survivors are listed with a reason in ALLOWED; an unexplained survivor fails the script."""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import mutlib

C = "lib/metrics/calibrate.cho"
M = [
    ("edge-start", C, "let e0 = bi * n / bins;", "let e0 = (bi * n + 1) / bins;"),
    ("edge-end", C, "let e1 = (bi + 1) * n / bins;", "let e1 = ((bi + 1) * n + 1) / bins;"),
    ("sort-stable", C, "p[idx[a]] <= p[idx[b]]", "p[idx[a]] < p[idx[b]]"),
    ("sort-direction", C, "p[idx[a]] <= p[idx[b]]", "p[idx[a]] >= p[idx[b]]"),
    ("mean-count", C, "let mean_pred = sum_p / float_of(m);", "let mean_pred = sum_p / float_of(m + 1);"),
    ("observed-count", C, "let observed = ratio.ratio(ys, m);", "let observed = ratio.ratio(ys, m + 1);"),
    ("ratio-direction", C, "rat = observed / mean_pred;", "rat = mean_pred / observed;"),
    ("ratio-nan", C, "if mean_pred > 0.0 {\n                                            rat = observed / mean_pred;", "if mean_pred >= 0.0 {\n                                            rat = observed / mean_pred;"),
    ("variance-floor", C, "if v < 1e-12 {", "if v < 1e-6 {"),
    ("variance-term", C, "contents(sqw)[i] = prob[r] * (1.0 - prob[r]);", "contents(sqw)[i] = prob[r] * prob[r];"),
    ("z-sign", C, "let z = (float_of(ys) - sum_p) / sd;", "let z = (sum_p - float_of(ys)) / sd;"),
    ("judged", C, "if sum_p >= min_expected {", "if sum_p > min_expected {"),
    ("band-upper", C, "if judged > 0.5 && !(rat >= band_lo && rat <= band_hi) {", "if judged > 0.5 && !(rat >= band_lo) {"),
    ("band-lower", C, "if judged > 0.5 && !(rat >= band_lo && rat <= band_hi) {", "if judged > 0.5 && !(rat <= band_hi) {"),
    ("z-condition", C, "if az > zmax {", "if az > zmax + 1.0 {"),
    ("z-absolute", C, "var az = z;\n                                            if az < 0.0 {\n                                                az = 0.0 - az;\n                                            }", "var az = z;"),
    ("overall-denominator", C, "let overall = float_of(positives) / denom;", "let overall = float_of(positives) / float_of(n);"),
    ("overall-floor", C, "if denom < 1e-12 {", "if denom < 1e-3 {"),
    ("ok-bad-bins", C, "if overall >= overall_lo && overall <= overall_hi && nbad == 0 {", "if overall >= overall_lo && overall <= overall_hi {"),
    ("ok-overall-upper", C, "if overall >= overall_lo && overall <= overall_hi && nbad == 0 {", "if overall >= overall_lo && nbad == 0 {"),
    ("bins-equal-rows", C, "bins > n {", "bins >= n {"),
    ("bins-zero", C, "if bins < 1 || bins > max_bins() || bins > n {", "if bins < 0 || bins > max_bins() || bins > n {"),
    ("probability-low", C, "if prob[i] < 0.0 || prob[i] > 1.0 {", "if prob[i] > 1.0 {"),
    ("probability-high", C, "if prob[i] < 0.0 || prob[i] > 1.0 {", "if prob[i] < 0.0 {"),
    ("label-binary", C, "if label[i] != 0 && label[i] != 1 {", "if label[i] != 0 && label[i] != 1 && label[i] != 2 {"),
    ("non-finite", C, "    if regress.all_finite(prob) == false {\n        return rules.non_finite();\n    }\n    var i = 0;", "    var i = 0;"),
    ("level-old-count", C, "        if days[i] <= cut {\n            n_old = n_old + 1;", "        if days[i] < cut {\n            n_old = n_old + 1;"),
    ("level-old-fill", C, "                                if days[i] <= cut {\n                                    contents(pow)[a] = prob[i];", "                                if days[i] < cut {\n                                    contents(pow)[a] = prob[i];"),
    ("level-median-even", C, "out = (days[contents(iw)[n / 2 - 1]] + days[contents(iw)[n / 2]]) / 2.0;", "out = days[contents(iw)[n / 2]];"),
    ("level-median-odd", C, "out = days[contents(iw)[n / 2]];\n            } else {", "out = days[contents(iw)[n / 2 - 1]];\n            } else {"),
    ("level-factor", C, "factor = float_of(yo_pos) / sum_old;", "factor = float_of(yo_pos) / float_of(n_old);"),
    ("level-clip-high", C, "if v > 1.0 {\n                    v = 1.0;", "if v > 2.0 {\n                    v = 1.0;"),
    ("level-empty-sum", C, "if n_old > 0 && n_new > 0 && sum_old > 0.0 {", "if n_old > 0 && n_new > 0 && sum_old >= 0.0 {"),
    ("level-empty-new", C, "if n_old > 0 && n_new > 0 && sum_old > 0.0 {", "if n_old > 0 && sum_old > 0.0 {"),
    ("level-positives", C, "yo_pos = yo_pos + label[i];", "yo_pos = yo_pos + 1;"),
]
ALLOWED = {
}

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if args:
    M = [m for m in M if any(a in m[0] for a in args)]
sys.exit(mutlib.run(M, ["sh", "scripts/gate_calibrate.sh"], "mutants_calibrate", allowed=ALLOWED, timeout=900))

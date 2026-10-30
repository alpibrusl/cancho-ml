#!/usr/bin/env python3
"""Generates docs/stories.html and docs/stories/*: eleven end-to-end stories of increasing complexity, from the baseline every model has to beat to a run that anyone can replay at any thread count, then four short ones from
the sciences. Every command is RUN against build/ml on the data of docs/stories/ (small files, CPU, each fit in well under a second) and every number, excerpt and figure is drawn from what it printed; the data are invented
(seeded generators below), the runs are real, and the narration is checked against the numbers: a sentence that stops being true stops the build of the page.

    python3 scripts/gen_stories.py [--bin build/ml] [--check]

`--check` regenerates in memory and fails if docs/stories.html or the data files differ. scripts/storylib.py is the same file as in cancho-stats and cancho-table."""
import argparse
import hashlib
import json
import math
import pathlib
import random
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from storylib import W, STORY_CSS, base_css, claim, pretty, num, esc, axis, svg, wrap_cmd, term, story, fig_hashes  # noqa: E402

DOCS = ROOT / "docs"


# ------------------------------------------------------------------------------------------------ the invented data
def csv_of(header, rows):
    return ",".join(header) + "\n" + "\n".join(",".join(str(x) for x in row) for row in rows) + "\n"


def make_rent():
    """240 days of a bicycle-hire shop: temperature, humidity, weekday, holiday, rentals; the first 180 days train, the last 60 test"""
    r = random.Random(21)
    rows = []
    for _ in range(240):
        t = round(r.uniform(2, 32), 1)
        h = int(round(r.uniform(30, 95)))
        wd = r.choice([0, 1, 1, 1, 1, 0, 0])
        hol = 1 if r.random() < 0.05 else 0
        y = 40 + 7 * t - 0.15 * t * t - 0.4 * h + 25 * wd - 30 * hol + r.gauss(0, 18)
        rows.append((t, h, wd, hol, max(0, int(round(y)))))
    return rows[:180], rows[180:]


def make_visits():
    """400 customers of the shop: visits last month, spend, coupon sent, did they come back"""
    r = random.Random(8)
    rows = []
    for _ in range(400):
        v = r.randint(0, 9)
        spend = round(max(0.0, r.gauss(40 + 6 * v, 25)), 1)
        coupon = 1 if r.random() < 0.4 else 0
        z = -2.4 + 0.45 * v + 0.012 * spend + 0.9 * coupon + r.gauss(0, 0.6)
        rows.append((v, spend, coupon, 1 if z > 0.25 else 0))
    return rows[:300], rows[300:]


def make_ring():
    """points in the square; the ones inside a circle are class 1"""
    r = random.Random(5)
    rows = []
    for _ in range(400):
        x, y = round(r.uniform(-1, 1), 3), round(r.uniform(-1, 1), 3)
        rows.append((x, y, 1 if x * x + y * y < 0.45 else 0))
    return rows[:300], rows[300:]


def make_expression():
    """tissue samples: four genes' expression (log2); genes a and b differ between tumour and normal, c and d do not"""
    r = random.Random(41)
    rows = []
    for _ in range(160):
        t = 1 if r.random() < 0.5 else 0
        a = round(r.gauss(7.0 + 1.6 * t, 1.0), 2)
        b = round(r.gauss(9.0 - 1.0 * t, 1.0), 2)
        c = round(r.gauss(5.0, 1.0), 2)
        d = round(r.gauss(8.0, 1.2), 2)
        rows.append((a, b, c, d, t))
    return rows[:120], rows[120:]


def make_calibration():
    """a Beer-Lambert calibration: absorbance of twelve standards (three replicates of four concentrations) and of five unknown samples whose true concentrations the generator keeps"""
    r = random.Random(3)
    std = []
    for c in (0.1, 0.2, 0.4, 0.8):
        for _ in range(3):
            std.append((round(0.002 + 0.74 * c + r.gauss(0, 0.004), 4), c))
    unknown = []
    truth = []
    for c in (0.15, 0.3, 0.5, 0.6, 0.7):
        unknown.append((round(0.002 + 0.74 * c + r.gauss(0, 0.004), 4),))
        truth.append(c)
    return std, unknown, truth


def make_events():
    """pulses from a detector: pulse height and rise time; two kinds of particle, unlabelled in the file (the generator keeps the labels)"""
    r = random.Random(17)
    rows, kind = [], []
    for _ in range(240):
        k = 1 if r.random() < 0.4 else 0
        if k:
            rows.append((round(r.gauss(52, 6), 1), round(r.gauss(18, 2.0), 1)))
        else:
            rows.append((round(r.gauss(30, 5), 1), round(r.gauss(9, 1.6), 1)))
        kind.append(k)
    return rows, kind


def make_town():
    """600 residents: age, income (thousand euros), years of schooling, whether they support a new tram line (more support among the young and the better off)"""
    r = random.Random(29)
    rows = []
    for _ in range(600):
        age = r.randint(18, 80)
        inc = round(max(8.0, r.gauss(32, 12)), 1)
        edu = r.randint(6, 20)
        z = 1.2 - 0.035 * (age - 18) + 0.03 * (inc - 32) + 0.04 * (edu - 12) + r.gauss(0, 0.8)
        rows.append((age, inc, edu, 1 if z > 0 else 0))
    return rows[:450], rows[450:]


def make_uplift(seed, n):
    """customers of a shop, half of them (a coin) shown an ad: days since the last order, orders in the last year, arm, bought. Buying depends on both; the ad helps the lapsed and does not help (here, slightly hurts) the loyal"""
    r = random.Random(seed)
    sig = lambda z: 1 / (1 + math.exp(-z))
    rows = []
    for _ in range(n):
        rec = r.randint(1, 365)
        orders = min(14, int(r.expovariate(1 / 3.5)))
        arm = r.random() < 0.5
        z = -3.6 + 0.28 * orders - 0.004 * rec + ((0.3 + 0.004 * (rec - 180) - 0.06 * (orders - 4)) if arm else 0)
        rows.append((rec, orders, "ads" if arm else "holdout", 1 if r.random() < sig(z) else 0))
    return rows


BIG = {}  # data files above the 2,000-line limit of a committed file: regenerated on demand (`--data-out DIR`), never committed


# ------------------------------------------------------------------------------------------------ real runs
class Runner:
    def __init__(self, binary, folder):
        self.binary = binary
        self.folder = folder

    def raw(self, args):
        return subprocess.run([self.binary] + args + ["--root", str(self.folder)], capture_output=True, text=True, timeout=600)

    def doc(self, args):
        p = self.raw(args)
        if p.returncode != 0:
            raise SystemExit("gen_stories: `ml %s` exited %d: %s" % (" ".join(args), p.returncode, (p.stdout + p.stderr)[:300]))
        return json.loads(p.stdout)["data"]

    def refusal(self, args):
        p = self.raw(args)
        if p.returncode == 0:
            raise SystemExit("gen_stories: `ml %s` was expected to be refused" % " ".join(args))
        return p.returncode, json.loads(p.stdout)["error"]

    def predictions(self, model, file, proba=False, limit=1000):
        d = self.doc(["predict", "--model-file", model, "--limit", str(limit)] + (["--proba"] if proba else []) + [file])
        return d["rows"]


def shown(args):
    return "ml " + " ".join(a if re.fullmatch(r"[\w.,:=/@+*-]+", a) else "'%s'" % a for a in args)


def tidy(err, keys):
    """the refusal as the tool printed it, with the program and the --root of this run taken out of the repair"""
    out = {k: err[k] for k in keys if k in err}
    rep = json.loads(json.dumps(err["repair"])) if err.get("repair") else None
    if rep and "argv" in rep:
        res, skip = ["ml"], False
        for a in rep["argv"][1:]:
            if skip:
                skip = False
            elif a == "--root":
                skip = True
            else:
                res.append(a)
        rep["argv"] = res
    out["repair"] = rep
    return out


def short(doc, drop=("path", "bytes")):
    return {k: v for k, v in doc.items() if k not in drop}


# ------------------------------------------------------------------------------------------------ figures
def fig_hbars(items, fmt="%.2f", label_w=130, unit=""):
    mx = max(v for _, v, _ in items)
    body = []
    for k, (name, v, cls) in enumerate(items):
        y = 26 + k * 34
        w = (W - label_w - 120) * v / mx
        body.append('<text class="nm" x="%g" y="%g" text-anchor="end">%s</text><rect class="pbar %s" x="%g" y="%g" width="%g" height="20" rx="3"/><text class="tk" x="%g" y="%g">%s%s</text>' % (label_w - 8, y + 5, esc(name), cls, label_w, y - 10, w, label_w + w + 8, y + 5, esc(fmt % v), unit))
    return svg(26 + 34 * len(items) + 6, "\n".join(body), "Bars")


def fig_pred_actual(pairs, lo, hi, label):
    ax, sx = axis(70, 330, 290, lo, hi, [0, 50, 100, 150, 200], "predicted")
    sy = lambda v: 270 - (v - lo) / (hi - lo) * 250
    body = [ax, '<line class="ax" x1="70" y1="20" x2="70" y2="290"/>']
    for t in (0, 50, 100, 150, 200):
        body.append('<text class="tk" x="62" y="%g" text-anchor="end">%d</text>' % (sy(t) + 4, t))
    body.append('<text class="lb" x="16" y="150" transform="rotate(-90 16 150)" text-anchor="middle">rentals that day</text>')
    body.append('<line class="zero" x1="%g" y1="%g" x2="%g" y2="%g"/>' % (sx(lo), sy(lo), sx(hi), sy(hi)))
    for p, a in pairs:
        body.append('<circle class="dot pa" cx="%g" cy="%g" r="3.8"/>' % (sx(p), sy(a)))
    body.append('<text class="nm" x="350" y="40">%s</text><text class="tk" x="350" y="60">dashed: a perfect prediction</text>' % esc(label))
    return svg(330, "\n".join(body), "Predicted against actual rentals on the held-out days")


def fig_two_scatter(left, right, lo, hi, ll, rl, ticks=(0, 50, 100, 150, 200)):
    body = []
    for off, (pairs, label) in ((0, (left, ll)), (320, (right, rl))):
        ax, sx = axis(40 + off, 300 + off, 270, lo, hi, list(ticks), label)
        sy = lambda v: 250 - (v - lo) / (hi - lo) * 230
        body.append(ax)
        body.append('<line class="ax" x1="%d" y1="20" x2="%d" y2="270"/>' % (40 + off, 40 + off))
        body.append('<line class="zero" x1="%g" y1="%g" x2="%g" y2="%g"/>' % (sx(lo), sy(lo), sx(hi), sy(hi)))
        for p, a in pairs:
            body.append('<circle class="dot %s" cx="%g" cy="%g" r="3.4"/>' % ("pa" if off == 0 else "pg", sx(p), sy(a)))
    return svg(320, "\n".join(body), "Predicted against actual rentals for two models")


def fig_flow(steps):
    n = len(steps)
    bw = (W - 40 - (n - 1) * 34) / n
    body = ['<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="context-stroke"/></marker></defs>']
    for k, (title, sub, cls) in enumerate(steps):
        x = 20 + k * (bw + 34)
        body.append('<rect class="fbox %s" x="%g" y="20" width="%g" height="92" rx="10"/><text class="nm" x="%g" y="48" text-anchor="middle">%s</text>' % (cls, x, bw, x + bw / 2, esc(title)))
        for i, line in enumerate(sub):
            body.append('<text class="tk" x="%g" y="%g" text-anchor="middle">%s</text>' % (x + bw / 2, 70 + i * 16, esc(line)))
        if k < n - 1:
            body.append('<line class="arrow" x1="%g" y1="66" x2="%g" y2="66"/>' % (x + bw + 4, x + bw + 30))
    return svg(132, "\n".join(body), "A request, its refusal, the repair and the answer")


def fig_confusion(tp, fp, fn, tn):
    body = []
    cells = [(tn, 0, 0, "neg"), (fp, 1, 0, "bad"), (fn, 0, 1, "bad"), (tp, 1, 1, "pos")]
    mx = max(tn, fp, fn, tp, 1)
    for v, cx, cy, cls in cells:
        x, y = 200 + cx * 130, 40 + cy * 80
        body.append('<rect class="cm %s" x="%g" y="%g" width="124" height="74" rx="6" style="opacity:%.2f"/><text class="digits" x="%g" y="%g" text-anchor="middle" style="fill:var(--fg)">%d</text>' % (cls, x, y, 0.25 + 0.75 * v / mx, x + 62, y + 46, v))
    body.append('<text class="tk" x="327" y="14" text-anchor="middle">what the model said</text><text class="nm" x="262" y="32" text-anchor="middle">away</text><text class="nm" x="392" y="32" text-anchor="middle">back</text>')
    body.append('<text class="nm" x="190" y="82" text-anchor="end">stayed away</text><text class="nm" x="190" y="162" text-anchor="end">returned</text>')
    return svg(210, "\n".join(body), "The counts of the four kinds of answer")


def fig_roc(points, auc):
    ax, sx = axis(60, 340, 290, 0, 1, [0, 0.25, 0.5, 0.75, 1], "false positive rate")
    sy = lambda v: 270 - v * 250
    body = [ax, '<line class="ax" x1="60" y1="20" x2="60" y2="290"/>', '<line class="zero" x1="%g" y1="%g" x2="%g" y2="%g"/>' % (sx(0), sy(0), sx(1), sy(1))]
    for t in (0, 0.5, 1):
        body.append('<text class="tk" x="52" y="%g" text-anchor="end">%g</text>' % (sy(t) + 4, t))
    body.append('<text class="lb" x="14" y="150" transform="rotate(-90 14 150)" text-anchor="middle">true positive rate</text>')
    body.append('<polyline class="ln la" points="%s"/>' % " ".join("%g,%g" % (sx(x), sy(y)) for x, y in points))
    body.append('<text class="nm" x="370" y="60">area under the curve</text><text class="digits" x="370" y="96" style="fill:var(--accent)">%.4f</text><text class="tk" x="370" y="120">dashed: a coin toss</text>' % auc)
    return svg(340, "\n".join(body), "The ROC curve of the held-out customers")


def fig_classes(rows, ll, title):
    ax, sx = axis(70, 330, 290, -1, 1, [-1, -0.5, 0, 0.5, 1], "x")
    sy = lambda v: 270 - (v + 1) / 2 * 250
    body = [ax, '<line class="ax" x1="70" y1="20" x2="70" y2="290"/>']
    for x, y, c in rows:
        body.append('<circle class="dot %s" cx="%g" cy="%g" r="3.4"/>' % ("pb" if c else "pa", sx(x), sy(y)))
    body.append('<text class="nm" x="360" y="50">%s</text><circle class="dot pa" cx="364" cy="74" r="4"/><text class="tk" x="374" y="78">outside the circle</text><circle class="dot pb" cx="364" cy="98" r="4"/><text class="tk" x="374" y="102">inside</text>' % esc(title))
    return svg(330, "\n".join(body), "The training points, coloured by class")


def fig_cv(depths):
    """accuracy of every fold of the cross-validation, by tree depth, and the mean"""
    ax, sx = axis(70, 600, 250, 0.5, len(depths) + 0.5, list(range(1, len(depths) + 1)), "maximum depth of the tree", fmt=lambda v: "%d" % v)
    sy = lambda v: 230 - (v - 0.5) / 0.5 * 200
    body = [ax, '<line class="ax" x1="70" y1="30" x2="70" y2="250"/>']
    for t in (0.5, 0.75, 1.0):
        body.append('<text class="tk" x="62" y="%g" text-anchor="end">%g</text>' % (sy(t) + 4, t))
    body.append('<text class="lb" x="16" y="140" transform="rotate(-90 16 140)" text-anchor="middle">accuracy on a held-out fold</text>')
    means = []
    for k, accs in enumerate(depths):
        for i, a in enumerate(accs):
            body.append('<circle class="dot pa" cx="%g" cy="%g" r="3.4"/>' % (sx(k + 1) + ((i % 5) - 2) * 5, sy(a)))
        m = sum(accs) / len(accs)
        means.append(m)
        body.append('<line class="md" x1="%g" y1="%g" x2="%g" y2="%g"/>' % (sx(k + 1) - 14, sy(m), sx(k + 1) + 14, sy(m)))
    return svg(290, "\n".join(body), "Accuracy of each held-out fold by the depth of the tree"), means


def fig_imp(items):
    """importance shares as bars"""
    return fig_hbars([(n, v, "pa") for n, v in items], fmt="%.3f", label_w=100)


def fig_calibration(std, line, unknown_pred):
    lo, hi = 0.0, 0.8
    ax, sx = axis(70, 330, 290, 0.0, 0.7, [0, 0.2, 0.4, 0.6], "absorbance")
    sy = lambda c: 270 - c / hi * 250
    body = [ax, '<line class="ax" x1="70" y1="20" x2="70" y2="290"/>']
    for t in (0, 0.2, 0.4, 0.6, 0.8):
        body.append('<text class="tk" x="62" y="%g" text-anchor="end">%g</text>' % (sy(t) + 4, t))
    body.append('<text class="lb" x="14" y="150" transform="rotate(-90 14 150)" text-anchor="middle">concentration, mM</text>')
    body.append('<line class="ciline" x1="%g" y1="%g" x2="%g" y2="%g"/>' % (sx(line[0][0]), sy(line[0][1]), sx(line[1][0]), sy(line[1][1])))
    for a, c in std:
        body.append('<circle class="dot pa" cx="%g" cy="%g" r="4"/>' % (sx(a), sy(c)))
    for a, c in unknown_pred:
        body.append('<path class="mn" d="M%g %g l6 6 l-6 6 l-6 -6 z"/>' % (sx(a), sy(c) - 6))
    body.append('<text class="nm" x="360" y="50">standards</text><circle class="dot pa" cx="456" cy="46" r="4"/><text class="nm" x="360" y="74">unknowns</text><path class="mn" d="M456 68 l6 6 l-6 6 l-6 -6 z"/>')
    return svg(330, "\n".join(body), "The calibration line, the standards and the unknown samples")


def fig_clusters(rows, labels):
    ax, sx = axis(70, 600, 290, 15, 70, [20, 30, 40, 50, 60, 70], "pulse height")
    sy = lambda v: 270 - (v - 3) / 21 * 250
    body = [ax, '<line class="ax" x1="70" y1="20" x2="70" y2="290"/>']
    for t in (5, 10, 15, 20):
        body.append('<text class="tk" x="62" y="%g" text-anchor="end">%d</text>' % (sy(t) + 4, t))
    body.append('<text class="lb" x="14" y="150" transform="rotate(-90 14 150)" text-anchor="middle">rise time, ns</text>')
    for (h, rt), c in zip(rows, labels):
        body.append('<circle class="dot %s" cx="%g" cy="%g" r="3.4"/>' % ("pb" if c else "pa", sx(h), sy(rt)))
    return svg(330, "\n".join(body), "The pulses, coloured by the cluster k-means found")


def fig_curve(xs, ys, xlabel, ylabel):
    ax, sx = axis(70, 600, 250, min(xs), max(xs), [20, 30, 40, 50, 60, 70, 80], xlabel, fmt=lambda v: "%d" % v)
    sy = lambda v: 230 - v * 200
    body = [ax, '<line class="ax" x1="70" y1="30" x2="70" y2="250"/>']
    for t in (0, 0.5, 1):
        body.append('<text class="tk" x="62" y="%g" text-anchor="end">%g</text>' % (sy(t) + 4, t))
    body.append('<text class="lb" x="14" y="140" transform="rotate(-90 14 140)" text-anchor="middle">%s</text>' % esc(ylabel))
    body.append('<line class="zero" x1="70" y1="%g" x2="600" y2="%g"/>' % (sy(0.5), sy(0.5)))
    body.append('<polyline class="ln la" points="%s"/>' % " ".join("%g,%g" % (sx(x), sy(y)) for x, y in zip(xs, ys)))
    return svg(290, "\n".join(body), ylabel)


def fig_signed(items, unit):
    """bars that may point left of zero: (label, value)"""
    lo, hi = min(0, min(v for _, v in items)), max(0, max(v for _, v in items))
    ax, sx = axis(150, 600, 26 + 34 * len(items) + 4, lo - 0.5, hi + 0.8, [t for t in range(int(math.floor(lo)), int(math.ceil(hi)) + 1)], unit, fmt=lambda v: "%d" % v)
    body = [ax, '<line class="zero" x1="%g" y1="10" x2="%g" y2="%g"/>' % (sx(0), sx(0), 26 + 34 * len(items))]
    for k, (name, v) in enumerate(items):
        y = 26 + k * 34
        x0, x1 = sorted((sx(0), sx(v)))
        body.append('<text class="nm" x="142" y="%g" text-anchor="end">%s</text><rect class="pbar %s" x="%g" y="%g" width="%g" height="20" rx="3"/><text class="tk" x="%g" y="%g" text-anchor="%s">%+.1f</text>' % (y + 5, esc(name), "pa" if v >= 0 else "bad", x0, y - 10, max(x1 - x0, 1), (x1 + 6) if v >= 0 else (x0 - 6), y + 5, "start" if v >= 0 else "end", v))
    return svg(26 + 34 * len(items) + 44, "\n".join(body), "Realized lift of each fifth of the customers")


def fig_gain(curves, xlabel, ylabel):
    """cumulative incremental buyers when contacting the top x% by each ranking; curves: (label, class, points) with points (x%, y)"""
    ymax = max(y for _, _, pts in curves for _, y in pts)
    ymin = min(0, min(y for _, _, pts in curves for _, y in pts))
    ax, sx = axis(70, 600, 250, 0, 100, [0, 20, 40, 60, 80, 100], xlabel, fmt=lambda v: "%d" % v)
    sy = lambda v: 230 - (v - ymin) / (ymax - ymin) * 200
    body = [ax, '<line class="ax" x1="70" y1="30" x2="70" y2="250"/>', '<line class="zero" x1="70" y1="%g" x2="600" y2="%g"/>' % (sy(0), sy(0))]
    body.append('<text class="lb" x="14" y="140" transform="rotate(-90 14 140)" text-anchor="middle">%s</text>' % esc(ylabel))
    for t in sorted({0, round(ymax / 2), round(ymax)}):
        body.append('<text class="tk" x="62" y="%g" text-anchor="end">%d</text>' % (sy(t) + 4, t))
    for k, (label, cls, pts) in enumerate(curves):
        body.append('<polyline class="ln %s" points="%s"/>' % (cls, " ".join("%g,%g" % (sx(x), sy(y)) for x, y in pts)))
        body.append('<text class="nm" x="%g" y="%g" style="fill:var(--%s)">%s</text>' % (84, 44 + k * 18, "accent" if cls == "la" else ("good" if cls == "lg" else "muted"), esc(label)))
    return svg(290, "\n".join(body), ylabel)


EXTRA_CSS = """
.fig .pbar.pa { fill: var(--accent); } .fig .pbar.pb { fill: var(--good); } .fig .pbar.bad { fill: var(--bad); }
.fig .fbox { fill: var(--surface); stroke-width: 2; } .fig .fbox.q { stroke: var(--accent); } .fig .fbox.r { stroke: var(--bad); } .fig .fbox.ok { stroke: var(--good); }
.fig .arrow { stroke: var(--muted); stroke-width: 1.8; marker-end: url(#ar); }
.fig .ln { fill: none; stroke-width: 2.5; } .fig .ln.la { stroke: var(--accent); } .fig .ln.lg { stroke: var(--good); } .fig .ln.lm { stroke: var(--muted); stroke-dasharray: 5 4; }
.fig .cm.neg { fill: var(--good); } .fig .cm.pos { fill: var(--good); } .fig .cm.bad { fill: var(--bad); }
.fig .digits { fill: var(--fg); font: 600 22px var(--mono); }
.fig .mn { fill: var(--warn); } .fig .md { stroke: var(--fg); stroke-width: 3; }
.fig .dot.pa { fill: var(--accent); opacity: .8; } .fig .dot.pb { fill: var(--bad); opacity: .8; } .fig .dot.pg { fill: var(--good); opacity: .8; }
.fig .ciline { stroke: var(--accent); stroke-width: 3; }
"""


# ------------------------------------------------------------------------------------------------ the page
def metric(doc, key):
    return float(doc["metrics"][key])


def build(binary):
    rent_tr, rent_te = make_rent()
    vis_tr, vis_te = make_visits()
    ring_tr, ring_te = make_ring()
    ex_tr, ex_te = make_expression()
    std, unknown, truth = make_calibration()
    events, kinds = make_events()
    town_tr, town_te = make_town()
    rh = ["temp_c", "humidity", "weekday", "holiday", "rentals"]
    gaps = [list(r) for r in rent_tr]
    for i in range(len(gaps)):
        if i % 17 == 3:
            gaps[i][1] = ""
    files = {
        "rent_train.csv": csv_of(rh, rent_tr), "rent_test.csv": csv_of(rh, rent_te), "rent_gaps.csv": csv_of(rh, gaps),
        "visits_train.csv": csv_of(["visits", "spend", "coupon", "returned"], vis_tr), "visits_test.csv": csv_of(["visits", "spend", "coupon", "returned"], vis_te),
        "ring_train.csv": csv_of(["x", "y", "inside"], ring_tr), "ring_test.csv": csv_of(["x", "y", "inside"], ring_te),
        "expression_train.csv": csv_of(["gene_a", "gene_b", "gene_c", "gene_d", "tumour"], ex_tr), "expression_test.csv": csv_of(["gene_a", "gene_b", "gene_c", "gene_d", "tumour"], ex_te),
        "standards.csv": csv_of(["absorbance", "conc_mM"], std), "unknowns.csv": csv_of(["absorbance"], unknown),
        "pulses.csv": csv_of(["pulse_height", "rise_ns"], events),
        "town_train.csv": csv_of(["age", "income_k", "schooling", "supports"], town_tr), "town_test.csv": csv_of(["age", "income_k", "schooling", "supports"], town_te),
    }
    with tempfile.TemporaryDirectory() as t:
        folder = pathlib.Path(t)
        for name, text in files.items():
            (folder / name).write_text(text)
        R = Runner(binary, folder)
        S = []
        # 1 baseline ---------------------------------------------------------------------------------------
        a1 = ["fit", "--model", "constant", "--target", "rentals", "--output", "base.ml", "rent_train.csv"]
        d1 = R.doc(a1)
        a1b = ["eval", "--model-file", "base.ml", "rent_test.csv"]
        d1b = R.doc(a1b)
        rmse_base = metric(d1b, "rmse")
        claim(abs(float(d1["summary"]["mean"]) - sum(r[4] for r in rent_tr) / len(rent_tr)) < 1e-5, "story 1: the constant model is the training mean")
        S.append(story(1, "The number to beat", "start here", ["fit --model constant", "eval"],
            "&#8220;I rent bicycles. Can a model predict how many I will rent tomorrow? Before I believe one, what does <em>not</em> knowing anything score?&#8221; &#8212; Pilar",
            ["<p>Every model is judged against the dumbest one: always answer the average of the training days. <code>ml fit --model constant</code> is exactly that, and <code>ml eval</code> scores it on days it has never seen (the last 60 of 240; the first 180 train). The model is a file; the score is a document.</p>",
             term(shown(a1), pretty(short(d1))),
             term(shown(a1b), pretty(short(d1b, ("path",)))),
             fig_hbars([("always the mean", rmse_base, "pa")], unit=" bicycles"),
             '<div class="read"><b>Reading it.</b> The mean of the 180 training days is %s rentals, and answering it every day is wrong by %s bicycles on a typical held-out day (the root-mean-square error). That is the line a real model has to get under, in the same units, on the same days.</div>' % (d1["summary"]["mean"], num(rmse_base, 4))]))
        # 2 linear -----------------------------------------------------------------------------------------
        a2 = ["fit", "--model", "linear", "--target", "rentals", "--output", "lin.ml", "rent_train.csv"]
        d2 = R.doc(a2)
        a2b = ["eval", "--model-file", "lin.ml", "rent_test.csv"]
        d2b = R.doc(a2b)
        pred_lin = [float(r_[0]) for r_ in R.predictions("lin.ml", "rent_test.csv")]
        pred_base = [float(r_[0]) for r_ in R.predictions("base.ml", "rent_test.csv")]
        actual = [r_[4] for r_ in rent_te]
        rm = math.sqrt(sum((p - a) ** 2 for p, a in zip(pred_lin, actual)) / len(actual))
        claim(abs(rm - metric(d2b, "rmse")) < 1e-6 and metric(d2b, "rmse") < rmse_base, "story 2: the printed rmse is the rmse of the predictions, and it beats the mean")
        coefs = {c["name"]: float(c["value"]) for c in d2["summary"]["coefficients"]}
        claim(coefs["temp_c"] > 0 > coefs["humidity"] and coefs["weekday"] > 0 > coefs["holiday"], "story 2: signs of the coefficients")
        S.append(story(2, "A line through the days", "a first real model", ["fit --model linear", "predict", "eval"],
            "&#8220;Warm, dry weekdays are busy; wet holidays are not. Can it learn that?&#8221;",
            ["<p>A linear model finds one weight per column. The fit prints them, with the intercept; <code>ml predict</code> applies the stored model to the held-out days and <code>ml eval</code> scores it. The arithmetic of the fit is a normal equation solved exactly where a sum decides the result, and the file records the SHA-256 of what it learnt.</p>",
             term(shown(a2), pretty(short(d2)["summary"] if False else {"model": d2["model"], "rows": d2["rows"], "summary": {"features": d2["summary"]["features"], "coefficients": d2["summary"]["coefficients"], "intercept": d2["summary"]["intercept"]}})),
             term(shown(a2b), pretty(short(d2b, ("path",)))),
             fig_two_scatter(list(zip(pred_base, actual)), list(zip(pred_lin, actual)), -20, 220, "the mean: every day the same", "a line through the days"),
             '<div class="read"><b>Reading it.</b> The weights say what the model learnt: each degree adds %s rentals, each point of humidity takes away %s, a weekday adds %s and a holiday takes away %s. On the 60 days it had not seen, its error is %s bicycles against %s for the mean, and it explains %s of the variance (<code>r2</code>). In the right panel the points hug the dashed line; on the left they are a horizontal column, the mean.</div>' % (num(coefs["temp_c"], 3), num(-coefs["humidity"], 3), num(coefs["weekday"], 3), num(-coefs["holiday"], 3), num(metric(d2b, "rmse"), 4), num(rmse_base, 4), num(metric(d2b, "r2"), 3))]))
        # 3 plan -------------------------------------------------------------------------------------------
        a3 = ["fit", "--model", "linear", "--target", "rentals", "--output", "gap.ml", "rent_gaps.csv"]
        rc3, err3 = R.refusal(a3)
        claim(err3["rule"] == "ml.input.missing-value" and err3["repair"]["kind"] == "retry", "story 3: the empty cell is refused with a retry repair")
        fix3 = [a for a in err3["repair"]["argv"][1:]]
        out3, skip = [], False
        for a in fix3:
            if skip:
                skip = False
            elif a == "--root":
                skip = True
            else:
                out3.append(a)
        d3 = R.doc(out3)
        a3c = ["fit", "--model", "linear", "--target", "rentals", "--step", "impute column=humidity strategy=mean;scale column=temp_c,humidity kind=standard", "--output", "gap2.ml", "rent_gaps.csv"]
        d3c = R.doc(a3c)
        d3e = R.doc(["eval", "--model-file", "gap2.ml", "rent_test.csv"])
        d3d = R.doc(["describe", "--model-file", "gap2.ml"])
        claim(metric(d3e, "rmse") < rmse_base, "story 3: the model fitted on the file with gaps beats the mean")
        S.append(story(3, "Empty cells are a decision", "a plan, a refusal, a repair", ["ml.input.missing-value", "--step impute;scale"],
            "&#8220;Eleven days the humidity sensor was down. The cells are empty.&#8221;",
            ["<p>A model cannot guess a missing number, and neither does <code>ml</code>: the fit is refused with a rule, a hint and the command that repairs it. Then the plan says more: impute the humidity with the mean (<em>of the training rows, stored in the model file</em>, so that <code>predict</code> fills gaps with the same number) and standardise two columns. Every step is fitted on the training rows and stored with the estimator in one file; there is no way to predict with a different transformation.</p>",
             term(shown(a3), "exit %d\n%s" % (rc3, pretty(tidy(err3, ("code", "rule", "message", "hint", "detail"))))),
             fig_flow([("fit", ["a column has", "empty cells"], "q"), ("refused", [err3["rule"].split(".")[1] + "." + err3["rule"].split(".")[2], "exit %d" % rc3], "r"), ("repair", ["--step impute", "column=humidity"], "ok"), ("fitted", ["%d rows" % d3["rows"]], "ok")]),
             term(shown(out3), pretty({"model": d3["model"], "rows": d3["rows"], "plan": d3["summary"]["plan"]})),
             term(shown(a3c), pretty({"model": d3c["model"], "plan": d3c["summary"]["plan"], "steps": d3c["summary"]["steps"]})),
             term(shown(["eval", "--model-file", "gap2.ml", "rent_test.csv"]), pretty(short(d3e, ("path",)))),
             '<div class="read"><b>Reading it.</b> The refusal named the column (<code>humidity</code>) and carried the repair as an argument vector; running it fitted the model. A plan with an impute and a scale step costs one flag, and its error on the held-out days is %s, still under the mean&#8217;s %s. The model file describes itself: <code>ml describe</code> lists the steps it will apply.</div>' % (num(metric(d3e, "rmse"), 4), num(rmse_base, 4))]))
        # 4 classification --------------------------------------------------------------------------------
        a4 = ["fit", "--model", "logistic", "--target", "returned", "--output", "ret.ml", "visits_train.csv"]
        d4 = R.doc(a4)
        a4b = ["eval", "--model-file", "ret.ml", "visits_test.csv"]
        d4b = R.doc(a4b)
        pr = R.predictions("ret.ml", "visits_test.csv", proba=True)
        y = [r_[3] for r_ in vis_te]
        lab = [int(r_[0]) for r_ in pr]
        prob = [float(r_[1]) for r_ in pr]
        tp = sum(1 for l_, a in zip(lab, y) if l_ == 1 and a == 1)
        fp = sum(1 for l_, a in zip(lab, y) if l_ == 1 and a == 0)
        fn = sum(1 for l_, a in zip(lab, y) if l_ == 0 and a == 1)
        tn = sum(1 for l_, a in zip(lab, y) if l_ == 0 and a == 0)
        acc = (tp + tn) / len(y)
        claim(abs(acc - metric(d4b, "accuracy")) < 1e-12 and abs(tp / (tp + fp) - metric(d4b, "precision")) < 1e-12, "story 4: accuracy and precision are those of the counts")
        # ROC from the probabilities of the run
        order = sorted(range(len(y)), key=lambda i: -prob[i])
        P, N = sum(y), len(y) - sum(y)
        pts, tpc, fpc = [(0.0, 0.0)], 0, 0
        for i in order:
            if y[i]:
                tpc += 1
            else:
                fpc += 1
            pts.append((fpc / N, tpc / P))
        auc = sum((pts[i + 1][0] - pts[i][0]) * (pts[i + 1][1] + pts[i][1]) / 2 for i in range(len(pts) - 1))
        claim(abs(auc - metric(d4b, "roc_auc")) < 1e-9, "story 4: the area under the drawn ROC curve is the printed roc_auc")
        S.append(story(4, "Will they come back?", "a yes/no question", ["fit --model logistic", "predict --proba", "eval"],
            "&#8220;I mailed a coupon to some of my customers. Who is likely to come back, and how sure can I be?&#8221;",
            ["<p>Logistic regression answers yes or no with a probability. The fit is Newton&#8217;s method on the exact log-loss; <code>predict --proba</code> gives the label and its probability for each customer the model has not seen, and <code>eval</code> scores them: accuracy, precision, recall, F1, the area under the ROC curve and the log-loss.</p>",
             term(shown(a4), pretty({"model": d4["model"], "rows": d4["rows"], "features": d4["summary"].get("features"), "coefficients": d4["summary"].get("coefficients")})),
             term(shown(a4b), pretty(short(d4b, ("path",)))),
             fig_confusion(tp, fp, fn, tn),
             fig_roc(pts, auc),
             '<div class="read"><b>Reading it.</b> Of %d held-out customers the model was right %d times (accuracy %s). It said &#8220;returns&#8221; %d times and was right on %d of them (precision %s); it caught %d of the %d who really returned (recall %s). The curve is drawn from the probabilities that <code>predict</code> printed and its area, %s, is the <code>roc_auc</code> printed by <code>eval</code>: a recomputation from the run, checked when this page is generated. <code>eval</code> also says that %d of the held-out cells (in the column <code>%s</code>) lay outside the range the model was fitted on: a model is only as good as the new rows resemble the old ones, and the answer says when they do not.</div>' % (len(y), tp + tn, num(acc, 3), tp + fp, tp, num(tp / (tp + fp), 3), tp, tp + fn, num(tp / (tp + fn), 3), num(auc, 4), d4b["out_of_range"]["cells"], ", ".join(c["name"] for c in d4b["out_of_range"]["columns"]))]))
        # 5 non-linear -------------------------------------------------------------------------------------
        accs = {}
        runs5 = []
        for nm, extra in (("logistic", ["--model", "logistic"]), ("tree, depth 4", ["--model", "tree", "--task", "classification", "--max-depth", "4"]), ("forest, 50 trees", ["--model", "forest", "--task", "classification", "--trees", "50", "--max-depth", "6", "--seed", "1"])):
            fa = ["fit"] + extra + ["--target", "inside", "--output", "ring-%d.ml" % len(accs), "ring_train.csv"]
            R.doc(fa)
            ev = ["eval", "--model-file", "ring-%d.ml" % len(accs), "ring_test.csv"]
            e5 = R.doc(ev)
            accs[nm] = metric(e5, "accuracy")
            runs5.append((fa, ev, e5))
        base_rate = sum(1 for r_ in ring_te if r_[2] == 0) / len(ring_te)
        claim(accs["logistic"] <= base_rate + 0.05 and accs["tree, depth 4"] > 0.85 and accs["forest, 50 trees"] > 0.85, "story 5: the line does no better than always answering outside; the tree and the forest are above 0.85")
        S.append(story(5, "When a line is not enough", "non-linear models", ["--model tree", "--model forest"],
            "&#8220;The points inside the circle are one class. Logistic regression gives me a straight line. What now?&#8221;",
            ["<p>A logistic model draws a straight boundary; no straight line separates a disc from its surroundings. A decision tree (CART with histogram splits) and a forest of bagged trees (parallel over trees, seeded) can. The same data, the same held-out points, three models:</p>",
             fig_classes([(x, y_, c) for x, y_, c in ring_tr], "inside", "the 300 training points"),
             term(shown(runs5[0][0]), "..."),
             term(shown(runs5[0][1]), pretty({"model": "logistic", "metrics": runs5[0][2]["metrics"]})),
             term(shown(runs5[2][1]), pretty({"model": "forest", "metrics": runs5[2][2]["metrics"]})),
             fig_hbars([(k, v, "pb" if v > 0.85 else "bad") for k, v in accs.items()], fmt="%.2f", label_w=130),
             '<div class="read"><b>Reading it.</b> %s of the held-out points are outside the circle, so a model that always says &#8220;outside&#8221; scores %s: the logistic model gets %s (its precision is <code>null</code> because it never said &#8220;inside&#8221;). The tree gets %s and the forest %s. The model has to fit the shape of the problem; the evaluation was the same for all three.</div>' % (num(base_rate, 3), num(base_rate, 3), num(accs["logistic"], 3), num(accs["tree, depth 4"], 3), num(accs["forest, 50 trees"], 3))]))
        # 6 cross-validation -------------------------------------------------------------------------------
        folds_by_depth = []
        cv_docs = []
        for depth in range(1, 7):
            a6 = ["cv", "--model", "tree", "--task", "classification", "--max-depth", str(depth), "--target", "inside", "--folds", "5", "--repeats", "2", "--seed", "3", "ring_train.csv"]
            d6 = R.doc(a6)
            cv_docs.append((a6, d6))
            folds_by_depth.append([float(x["metrics"]["accuracy"]) for x in d6["results"]])
        fig6, means6 = fig_cv(folds_by_depth)
        best = max(range(6), key=lambda i: means6[i]) + 1
        claim(means6[0] < means6[3] and len(folds_by_depth[0]) == 10, "story 6: ten folds per depth, and depth 4 beats depth 1")
        S.append(story(6, "Do not trust one split", "cross-validation", ["cv --folds 5 --repeats 2"],
            "&#8220;The tree scored well on my test points. Was I lucky with which points I held out? And how deep should it be?&#8221;",
            ["<p><code>ml cv</code> fits the model on every training fold and scores it on the fold held out, <code>K</code> folds and <code>R</code> repeats, and writes no file. For each depth of the tree, five folds twice, with a seed: ten scores per depth.</p>",
             term(shown(cv_docs[3][0]), pretty({"strategy": cv_docs[3][1]["strategy"], "folds": cv_docs[3][1]["folds"], "repeats": cv_docs[3][1]["repeats"], "results": cv_docs[3][1]["results"][:1] + ["..."]})),
             fig6,
             '<div class="read"><b>Reading it.</b> A tree of depth 1 scores %s on average, depth 2 %s, depth %d the best, %s. The dots are the ten folds of each depth: the spread tells how much one held-out set can mislead. A single test split gives one dot; the cross-validation gives ten, and the same seed gives the same folds on any machine.</div>' % (num(means6[0], 3), num(means6[1], 3), best, num(means6[best - 1], 3))]))
        # 7 reproducibility --------------------------------------------------------------------------------
        (folder / "runs").mkdir(exist_ok=True)
        a7 = ["fit", "--model", "forest", "--task", "classification", "--trees", "40", "--max-depth", "6", "--seed", "2", "--target", "inside", "--track", "runs", "--output", "run.ml", "ring_train.csv"]
        d7 = R.doc(a7)
        runid = d7["run"]["id"]
        v1 = R.doc(["runs", "verify", "runs", runid])
        v8 = R.doc(["runs", "verify", "--threads", "8", "runs", runid])
        claim(v1["verified"] and v8["verified"] and v1["model_sha256"] == v8["model_sha256"] == d7["sha256"], "story 7: the run verifies at 1 and 8 threads with the model's hash")
        digs = []
        for th in (1, 2, 4, 8):
            fa = ["fit", "--model", "forest", "--task", "classification", "--trees", "40", "--max-depth", "6", "--seed", "2", "--target", "inside", "--threads", str(th), "--output", "t%d.ml" % th, "ring_train.csv"]
            dd = R.doc(fa)
            digs.append(("--threads %d: model file" % th, dd["sha256"][:48]))
        claim(len({h for _, h in digs}) == 1 and digs[0][1] == d7["sha256"][:48], "story 7: the same model file at 1, 2, 4 and 8 threads")
        S.append(story(7, "Same model, any machine", "reproducibility", ["fit --track", "runs verify", "--threads"],
            "&#8220;My colleague refit it on the server with sixteen cores. Is it the same model?&#8221;",
            ["<p>A forest is a bag of random trees, which is the usual way to get a result nobody can reproduce. Here every tree has its own random stream derived from the seed and the tree&#8217;s index, so the number of threads cannot change a tree. <code>fit --track DIR</code> writes a <em>run record</em>: canonical JSON, named by the hash of the inputs that determine the model, with no timestamp and no host name. <code>runs verify</code> refits from the record, in memory, and compares the SHA-256 of the model file.</p>",
             term(shown(a7), pretty({"model": d7["model"], "sha256": d7["sha256"], "run": {"id": d7["run"]["id"], "dir": d7["run"]["dir"]}})),
             term(shown(["runs", "verify", "runs", runid]), pretty(v1)),
             term(shown(["runs", "verify", "--threads", "8", "runs", runid]), pretty(v8)),
             "<p>The same forest, fitted on one, two, four and eight threads; the SHA-256 of each model file:</p>",
             fig_hashes(digs),
             '<div class="read"><b>Reading it.</b> The run record verifies on one thread and on eight, with the model hash %s&hellip;. The four model files are the same bytes. That is the claim a result has to make before anyone else can rely on it, and it is the claim the project&#8217;s CI checks across Linux and macOS.</div>' % d7["sha256"][:12]]))
        S.append('<section class="part" id="lab"><div class="wrap"><h2>Part II &middot; In the lab and in the field</h2><p class="lead">The same tool on four small datasets from the sciences: tumour or normal tissue from gene expression, a calibration curve, two kinds of particle in a detector, a town survey. The data are invented; the runs are real.</p></div></section>')
        # 8 biology ----------------------------------------------------------------------------------------
        a8 = ["fit", "--model", "logistic", "--target", "tumour", "--step", "scale column=* kind=standard", "--output", "tum.ml", "expression_train.csv"]
        d8 = R.doc(a8)
        a8b = ["eval", "--model-file", "tum.ml", "expression_test.csv"]
        d8b = R.doc(a8b)
        a8c = ["importance", "--model-file", "tum.ml"]
        d8c = R.doc(a8c)
        imp = [(c["column"], float(c["importance"])) for c in d8c["importance"]["columns"]]
        claim({n for n, _ in sorted(imp, key=lambda x: -x[1])[:2]} == {"gene_a", "gene_b"} and metric(d8b, "accuracy") > 0.8, "story 8: genes a and b carry the importance and the held-out accuracy is above 0.8")
        S.append(story(8, "Tumour or normal?", "biology: which features matter", ["fit --model logistic", "importance"],
            "&#8220;Four genes measured in 120 tissue samples. Can they tell tumour from normal? And which ones are doing the telling?&#8221; &#8212; a bioinformatician",
            ["<p>The columns are standardised (a <code>scale</code> step stored in the model), a logistic regression is fitted, and then <code>ml importance</code> says which columns the model relies on: for a linear model, the stored coefficient times the standard deviation of the column the model sees, as a share of the total.</p>",
             term(shown(a8b), pretty(short(d8b, ("path",)))),
             term(shown(a8c), pretty({"importance": {"method": d8c["importance"]["method"], "columns": [{k: c[k] for k in ("column", "standardised", "importance", "rank")} for c in d8c["importance"]["columns"]]}})),
             fig_imp(sorted(imp, key=lambda x: -x[1])),
             '<div class="read"><b>Reading it.</b> On the %d held-out samples the model is right %s of the time (area under the curve %s). Genes <code>%s</code> and <code>%s</code> carry %s of the importance together; the other two, which the generator made independent of the tissue, carry the rest. A score alone would not tell; the importance does, and it is computed from the stored model, not from a refit.</div>' % (d8b["rows"], num(metric(d8b, "accuracy"), 3), num(metric(d8b, "roc_auc"), 3), sorted(imp, key=lambda x: -x[1])[0][0], sorted(imp, key=lambda x: -x[1])[1][0], num(sum(v for _, v in sorted(imp, key=lambda x: -x[1])[:2]), 3))]))
        # 9 chemistry --------------------------------------------------------------------------------------
        a9 = ["fit", "--model", "linear", "--target", "conc_mM", "--output", "cal.ml", "standards.csv"]
        d9 = R.doc(a9)
        a9b = ["predict", "--model-file", "cal.ml", "unknowns.csv"]
        d9b = R.doc(a9b)
        pred9 = [float(r_[0]) for r_ in d9b["rows"]]
        a9c = ["eval", "--model-file", "cal.ml", "standards.csv"]
        d9c = R.doc(a9c)
        errs = [abs(p - t_) for p, t_ in zip(pred9, truth)]
        claim(max(errs) < 0.02 and metric(d9c, "r2") > 0.999, "story 9: the unknowns are within 0.02 mM of the truth and r2 on the standards is above 0.999")
        co = {c["name"]: float(c["value"]) for c in d9["summary"]["coefficients"]}
        icpt = float(d9["summary"]["intercept"])
        S.append(story(9, "A calibration curve", "chemistry: fit once, apply to unknowns", ["fit --model linear", "predict", "eval"],
            "&#8220;Twelve standards, four concentrations, three replicates each. Five unknowns. What are their concentrations?&#8221;",
            ["<p>A calibration is a regression of concentration on absorbance, fitted on the standards and <em>applied</em> to the unknowns. <code>ml fit</code> stores the line in a model file; <code>ml predict</code> applies it to the file of unknowns, and the file cannot be applied with a different transformation than the one it learnt.</p>",
             term(shown(a9), pretty({"rows": d9["rows"], "coefficients": d9["summary"]["coefficients"], "intercept": d9["summary"]["intercept"]})),
             term(shown(a9c), pretty(short(d9c, ("path",)))),
             term(shown(a9b), pretty({"columns": d9b["columns"], "rows": d9b["rows"]})),
             fig_calibration([(a, c) for a, c in std], ((0.0, icpt), (0.62, icpt + co["absorbance"] * 0.62)), [(u[0], p) for u, p in zip(unknown, pred9)]),
             '<div class="read"><b>Reading it.</b> The line explains %s of the variance of the standards (r2). The five unknowns come out at %s mM; the generator knows their true concentrations (%s), and the largest difference is %s mM. In a real laboratory the truth is the one thing you do not have, which is why the standards are replicated; here the page can show the check.</div>' % (num(metric(d9c, "r2"), 5), ", ".join("%.3f" % p for p in pred9), ", ".join("%g" % t_ for t_ in truth), num(max(errs), 2))]))
        # 10 physics ---------------------------------------------------------------------------------------
        a10 = ["fit", "--model", "kmeans", "--k", "2", "--seed", "1", "--step", "scale column=* kind=standard", "--output", "km.ml", "pulses.csv"]
        d10 = R.doc(a10)
        a10b = ["predict", "--model-file", "km.ml", "--limit", "1000", "pulses.csv"]
        d10b = R.doc(a10b)
        cl = [int(r_[0]) for r_ in d10b["rows"]]
        agree = max(sum(1 for c, k in zip(cl, kinds) if c == k), sum(1 for c, k in zip(cl, kinds) if c != k)) / len(kinds)
        a10c = ["eval", "--model-file", "km.ml", "pulses.csv"]
        d10c = R.doc(a10c)
        claim(agree > 0.95 and sorted(d10c["metrics"]["sizes"]) == sorted([cl.count(0), cl.count(1)]), "story 10: the clusters agree with the kinds the generator kept on more than 95% of the pulses, and the sizes printed are the sizes counted")
        S.append(story(10, "Two kinds of pulse", "physics: clusters with no labels", ["fit --model kmeans", "predict", "eval"],
            "&#8220;The detector records the height and the rise time of every pulse. I think there are two kinds of particle. Without labels, can the data say so?&#8221;",
            ["<p>k-means has no target: <code>fit</code> needs none and refuses one. The columns are standardised first (a pulse height in the tens and a rise time in the tens of nanoseconds have different units), <code>--k 2</code> asks for two centres, and the initial centres are drawn from <code>--seed</code>, so the clusters are the same on every run.</p>",
             term(shown(a10), pretty({"model": d10["model"], "rows": d10["rows"], "target": d10["target"]})),
             term(shown(a10c), pretty({"clusters": d10c["clusters"], "metrics": d10c["metrics"]})),
             fig_clusters(events, cl),
             '<div class="read"><b>Reading it.</b> Two clusters of %s and %s pulses. The generator kept the true kind of each pulse (the file does not have it): the clusters agree with it on %s of the pulses. With real data there is no such check, and the honest statement is the one the tool prints: the inertia, the sizes and the mean distance of each cluster, not that these are two particles.</div>' % (d10c["metrics"]["sizes"][0], d10c["metrics"]["sizes"][1], num(agree, 3))]))
        # 11 social ----------------------------------------------------------------------------------------
        a11 = ["fit", "--model", "logistic", "--target", "supports", "--step", "scale column=* kind=standard", "--output", "town.ml", "town_train.csv"]
        d11 = R.doc(a11)
        a11b = ["eval", "--model-file", "town.ml", "town_test.csv"]
        d11b = R.doc(a11b)
        a11c = ["importance", "--model-file", "town.ml"]
        d11c = R.doc(a11c)
        imp11 = {c["column"]: float(c["importance"]) for c in d11c["importance"]["columns"]}
        grid_ages = list(range(18, 81, 4))
        med_inc = sorted(r_[1] for r_ in town_tr)[len(town_tr) // 2]
        med_edu = sorted(r_[2] for r_ in town_tr)[len(town_tr) // 2]
        (folder / "grid.csv").write_text(csv_of(["age", "income_k", "schooling"], [(a, med_inc, med_edu) for a in grid_ages]))
        pg = R.predictions("town.ml", "grid.csv", proba=True)
        curve = [float(r_[1]) for r_ in pg]
        claim(all(curve[i] >= curve[i + 1] for i in range(len(curve) - 1)) and curve[0] > 0.5 > curve[-1], "story 11: the predicted support falls with age, from above one half to below it")
        claim(imp11["age"] == max(imp11.values()), "story 11: age is the most important column")
        S.append(story(11, "Who supports the tram line?", "social science: a model to read, not to use", ["fit --model logistic", "importance", "predict --proba"],
            "&#8220;Six hundred residents were asked. I do not want to predict anyone. I want to know what the answers depend on.&#8221; &#8212; a council analyst",
            ["<p>A model can be a way of describing a survey: which columns the answer depends on, and how. The logistic regression is fitted on 450 residents with standardised columns, scored on 150 it has not seen, and then asked two questions: which column matters (<code>importance</code>) and what the model says about a resident of the median income and schooling at each age (<code>predict --proba</code> on a grid of ages).</p>",
             term(shown(a11b), pretty(short(d11b, ("path",)))),
             term(shown(a11c), pretty({"importance": {"columns": [{k: c[k] for k in ("column", "standardised", "importance", "rank")} for c in d11c["importance"]["columns"]]}})),
             fig_curve(grid_ages, curve, "age, years", "predicted probability of support"),
             '<div class="read"><b>Reading it.</b> Held-out accuracy %s; the model is not the point. The ranking is: <code>age</code> carries %s of the importance, income %s, schooling %s. The curve is what the model says for a resident of median income (%s thousand) and schooling (%s years): support falls from %s at 18 to %s at 80, crossing one half in between. It describes this sample; it does not say that growing older causes anything.</div>' % (num(metric(d11b, "accuracy"), 3), num(imp11["age"], 3), num(imp11["income_k"], 3), num(imp11["schooling"], 3), med_inc, med_edu, num(curve[0], 2), num(curve[-1], 2))]))
        # 12 marketing: uplift --------------------------------------------------------------------------------
        up_tr, up_te = make_uplift(61, 14000), make_uplift(62, 14000)
        uh = ["recency_days", "orders_12m", "bought"]
        for nm, text in (("uplift_holdout_train.csv", csv_of(uh, [(a, b, d) for a, b, c, d in up_tr if c == "holdout"])), ("uplift_ads_train.csv", csv_of(uh, [(a, b, d) for a, b, c, d in up_tr if c == "ads"])),
                         ("uplift_test.csv", csv_of(["recency_days", "orders_12m", "arm", "bought"], up_te)), ("uplift_test_x.csv", csv_of(["recency_days", "orders_12m"], [(a, b) for a, b, c, d in up_te]))):
            BIG[nm] = text
            (folder / nm).write_text(text)
        u1 = ["fit", "--model", "logistic", "--target", "bought", "--output", "u_holdout.ml", "uplift_holdout_train.csv"]
        u2 = ["fit", "--model", "logistic", "--target", "bought", "--output", "u_ads.ml", "uplift_ads_train.csv"]
        du1, du2 = R.doc(u1), R.doc(u2)
        p0 = [float(r_[1]) for r_ in R.predictions("u_holdout.ml", "uplift_test_x.csv", proba=True, limit=20000)]
        p1 = [float(r_[1]) for r_ in R.predictions("u_ads.ml", "uplift_test_x.csv", proba=True, limit=20000)]
        claim(len(p0) == len(up_te) == len(p1), "story 12: one probability per test customer from each model")
        upl = [b - a for a, b in zip(p0, p1)]

        def fifths(score):
            idx = sorted(range(len(up_te)), key=lambda i: -score[i])
            out = []
            for k in range(5):
                grp = idx[k * len(idx) // 5:(k + 1) * len(idx) // 5]
                t = [up_te[i][3] for i in grp if up_te[i][2] == "ads"]
                c = [up_te[i][3] for i in grp if up_te[i][2] == "holdout"]
                out.append(100 * (sum(t) / len(t) - sum(c) / len(c)))
            return out
        fu, fr = fifths(upl), fifths(p1)
        claim(fu[0] > 1.5 and fu[0] > fu[2] and fu[4] < 0 and fu[4] < fu[0] - 3, "story 12: the fifth ranked highest by predicted uplift has a clearly positive realized lift and the lowest one a negative lift")
        claim(max(fr) - min(fr) < 0.7 * (max(fu) - min(fu)), "story 12: ranking by the response model spreads the realized lift less than ranking by uplift")
        te_t = [r_[3] for r_ in up_te if r_[2] == "ads"]
        te_c = [r_[3] for r_ in up_te if r_[2] == "holdout"]
        ate = 100 * (sum(te_t) / len(te_t) - sum(te_c) / len(te_c))
        S.append(story(12, "Who does the ad actually move?", "marketing: uplift, two models and a randomized test", ["fit --model logistic", "predict --proba"],
            "&#8220;The ad lifts sales by about one point overall. Is it one point for everybody, or a lot for some and nothing &#8212; or worse &#8212; for others?&#8221; &#8212; the same head of growth, a month later",
            ["<p>Because the ad was shown to a random half of the customers (the stories of cancho-stats showed why that matters), the customers who saw it and those who did not are comparable <em>in every respect</em>, and so is a model fitted to each. The simplest way to ask who is moved is to fit two models, one on the customers shown the ad and one on the held-out customers, and to subtract: for a new customer, the predicted chance of buying <em>with</em> the ad minus the predicted chance <em>without</em>. That difference is the predicted uplift. Two fits, one per file:</p>",
             term(shown(u1), pretty({"model": du1["model"], "rows": du1["rows"], "coefficients": du1["summary"].get("coefficients")})),
             term(shown(u2), pretty({"model": du2["model"], "rows": du2["rows"], "coefficients": du2["summary"].get("coefficients")})),
             "<p>Then <code>predict --proba</code> with each model on a second, later group of %d customers, who were also randomly split, and the uplift of each is the difference of the two probabilities. A prediction is only a claim until it is checked, and here it can be checked against the truth: sort the new customers from the highest predicted uplift (the top fifth) to the lowest (the bottom fifth), cut them into five equal groups, and in each group compare the buyers among those who were shown the ad with those who were not.</p>" % len(up_te),
             fig_signed([("top fifth", fu[0]), ("second", fu[1]), ("third", fu[2]), ("fourth", fu[3]), ("bottom fifth", fu[4])], "realized lift in each fifth, points of conversion (measured on the held-out test)"),
             '<div class="read"><b>Reading it.</b> Over all %d test customers the ad lifts conversion by %+.1f points. But in the fifth of customers that the models picked as the most movable it lifts it by %+.1f points, and in the fifth picked as the least movable by %+.1f: for them the ad does nothing, or slightly harms. The models learned that from one number per customer, how long ago they last bought and how often they buy. The groups hold %d customers each, so each bar carries a margin of about a point or two (the middle ones are not reliably ordered), and that is how a model like this is to be read: the ends, not the middle. The models are linear in the log odds; a forest per arm (<code>--model forest</code>) is the usual next step, and the same test on randomized data is how to know it helped.</div>' % (len(up_te), ate, fu[0], fu[4], len(up_te) // 5)]))
        # 13 marketing: who to spend on ------------------------------------------------------------------------
        def gain(score):
            idx = sorted(range(len(up_te)), key=lambda i: -score[i])
            pts = []
            tt = [r_ for r_ in up_te]
            nt = sum(1 for r_ in tt if r_[2] == "ads")
            nc = len(tt) - nt
            for q in range(0, 101, 5):
                grp = idx[:q * len(idx) // 100]
                bt = sum(up_te[i][3] for i in grp if up_te[i][2] == "ads")
                bc = sum(up_te[i][3] for i in grp if up_te[i][2] == "holdout")
                ng = len(grp)
                at = sum(1 for i in grp if up_te[i][2] == "ads")
                ac = ng - at
                pts.append((q, (bt / at - bc / ac) * ng if at and ac else 0.0))
            return pts
        gu, gr = gain(upl), gain(p1)
        full = gu[-1][1]
        rnd = [(q, full * q / 100) for q in range(0, 101, 5)]
        at30 = lambda pts: dict(pts)[30]
        claim(abs(gu[-1][1] - gr[-1][1]) < 1e-9 and at30(gu) > at30(gr) and at30(gu) > at30(rnd) and max(y for _, y in gu) > full * 1.2, "story 13: both rankings end at the same total; at 30% the uplift ranking has more incremental buyers than the response ranking and than chance; the uplift curve rises above the total (some contacts lose sales)")
        best = max(gu, key=lambda p: p[1])
        S.append(story(13, "Spend the budget on the right third", "marketing: targeting by response or by uplift", ["predict --proba", "eval"],
            "&#8220;We can only afford to contact a third of them. Take the customers most likely to buy, right?&#8221; &#8212; the finance partner, who wants a number",
            ["<p>There are two natural rankings. <em>Most likely to buy</em> uses the model of the customers shown the ad, and puts the loyal first. <em>Most movable</em> uses the uplift of the previous story. The test group was randomized, so a ranking can be scored honestly: contact the top q percent by that ranking, and the buyers the contact adds are the difference in conversion between those shown the ad and those not, times the size of the group. This is a <em>gain curve</em>: how many extra buyers each plan buys as the contacted share grows from zero to everyone.</p>",
             fig_gain([("ranked by uplift", "la", gu), ("ranked by chance to buy", "lg", gr), ("random order", "lm", rnd)], "share of customers contacted, %", "extra buyers"),
             '<div class="read"><b>Reading it.</b> Contacting everybody buys %.0f extra buyers in the %d customers of the test. Contacting the top 30%% by chance to buy gets %.0f: not much more than the %.0f that contacting 30%% at random would. The same 30%% chosen by uplift gets %.0f, and the uplift curve peaks at %.0f extra buyers when %d%% are contacted, <em>above</em> the total of contacting everyone: the last customers on the list lose the shop sales when they are contacted, and a plan that leaves them out does better than a plan that does not. The curves are estimates on one random split (the stats stories explain how wide such estimates are, and a bootstrap of the gain is one way to put an interval on them), so the ordering is clear and the exact peak is not. <code>ml</code> fitted the models and made the predictions; the arithmetic of the curve is the page&#8217;s, from the printed probabilities and the arm and outcome of each test customer.</div>' % (full, len(up_te), at30(gr), at30(rnd), at30(gu), best[1], best[0])]))
    return S, files


def page(sections):
    css = base_css("#6d3fc0", "#b79bf5", "#ffffff", "#150b2b") + STORY_CSS + EXTRA_CSS
    nav = " ".join('<a class="opt" href="#s%d">%d</a>' % (i, i) for i in range(1, 14))
    names = ("Baseline", "A line", "Plans", "Yes or no", "Non-linear", "Cross-validation", "Same model", "Tumour", "Calibration", "Pulses", "Survey", "Uplift", "Targeting")
    toc = "".join('<a href="#s%d">%d &middot; %s</a>' % (i, i, esc(nm)) for i, nm in enumerate(names, 1))
    data = ", ".join('<a href="stories/%s">%s</a>' % (n, n) for n in sorted(FILES_LIST))
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ml: thirteen stories, from a baseline to the lab</title>
<meta name="description" content="Thirteen end-to-end examples of the ml command, in increasing complexity: bicycle hire, then biology, chemistry, physics and a social survey. Every number and figure is drawn from a real run.">
<meta name="color-scheme" content="light dark">
<style>%s</style>
</head>
<body>
<header class="top"><div class="wrap">
  <a class="brand" href=".">ml</a>
  <nav>%s<a href=".">Overview</a><a href="https://github.com/alpibrusl/cancho-ml">GitHub</a></nav>
</div></header>
<div class="wrap hero small">
  <h1>Thirteen stories, from a baseline to the lab.</h1>
  <p class="sub">Pilar rents bicycles and wants to know what a model can tell her; each story asks a little more of <code>ml</code> than the last. Then four short ones from the sciences and two from marketing (who does an ad move, and whom to spend on). Every number and every figure is drawn from a real run of <code>ml</code> on the data you can download (small files, one CPU, each fit well under a second). The data are invented; the runs are real, and the narration is checked against the numbers whenever the page is generated.</p>
  <div class="toc">%s</div>
  <p class="data">Data: %s. The four files of stories 12 and 13 (<code>uplift_*.csv</code>, 14,000 customers a file at most) are not committed: <code>python3 scripts/gen_stories.py --data-out DIR</code> writes them, from the same seeds. Rebuild this page: <code>python3 scripts/gen_stories.py</code>.</p>
</div>
%s
<footer><div class="wrap">ml &middot; EUPL-1.2 &middot; the data are invented; the runs, the digits and the figures are real &middot; <a href=".">overview</a></div></footer>
</body>
</html>
""" % (css, nav, toc, data, "\n".join(sections))


FILES_LIST = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(ROOT / "build" / "ml"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--data-out", help="also write the data files too large to commit into this directory")
    a = ap.parse_args()
    sections, files = build(a.bin)
    if a.data_out:
        pathlib.Path(a.data_out).mkdir(parents=True, exist_ok=True)
        for name, text in BIG.items():
            (pathlib.Path(a.data_out) / name).write_text(text)
    FILES_LIST.extend(files)
    out = {DOCS / "stories.html": page(sections)}
    for name, text in files.items():
        out[DOCS / "stories" / name] = text
    if a.check:
        stale = [str(p.relative_to(ROOT)) for p, text in out.items() if not p.exists() or p.read_text() != text]
        if stale:
            print("gen_stories: stale: " + ", ".join(stale))
            return 1
        print("gen_stories: up to date (%d files)" % len(out))
        return 0
    for p, text in out.items():
        p.parent.mkdir(exist_ok=True)
        p.write_text(text)
    print("gen_stories: wrote %d files" % len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Writes the cards of the gallery (docs/gallery.html, between the paired gen:cards comment lines) from real runs of the built `ml`.

    python3 scripts/gallery.py              write the cards
    python3 scripts/gallery.py --check      change nothing; exit 1 if the page differs from what the binary produces
    python3 scripts/gallery.py --bin PATH   the binary (default build/ml)

Every number and every shape in a card is read from an answer of `ml`: the points of a scatter are the rows of `ml predict`, the tree is the nodes of `ml dump`, the bars are the
metrics of `ml eval` or the columns of `ml importance`. The data are the files of docs/examples/data, except for the two cases that need data the bundled files do not have (three
classes, and queries with relevance grades): those are written here from a seed, the card says "synthetic", and the file is the same on every run. A model the binary does not build is
left out. Standard library only; nothing is drawn that a run did not print.
"""

import argparse
import hashlib
import html
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "examples" / "data"
PAGE = ROOT / "docs" / "gallery.html"
STD = "scale column=* kind=standard"


class Drift(Exception):
    pass


def esc(s):
    return html.escape(str(s), quote=True)


class Lab:
    """A scratch directory with the example files, in which `ml` is run."""

    def __init__(self, binary, tmp):
        self.bin, self.dir = binary, Path(tmp)
        for f in DATA.glob("*.csv"):
            shutil.copy(f, self.dir / f.name)

    def write(self, name, text):
        (self.dir / name).write_text(text)

    def run(self, cmd):
        """cmd: the command as a person types it ("ml fit ..."). -> (status, parsed JSON of the answer or None, raw output)"""
        words = shlex.split(cmd)
        assert words[0] == "ml", cmd
        p = subprocess.run([str(self.bin)] + words[1:], cwd=self.dir, capture_output=True, text=True, timeout=300)
        try:
            doc = json.loads(p.stdout.splitlines()[0]) if p.stdout.strip() else None
        except ValueError:
            doc = None
        return p.returncode, doc, p.stdout

    def ok(self, cmd):
        st, doc, out = self.run(cmd)
        if st not in (0, 9) or not doc or not doc.get("ok"):
            raise Drift("the gallery command failed (exit %d): %s\n%s" % (st, cmd, out[:400]))
        return doc["data"] if "data" in doc else doc

    def csv(self, name):
        lines = (self.dir / name).read_text().splitlines()
        head = lines[0].split(",")
        return head, [l.split(",") for l in lines[1:]]


def f(x, nd=4):
    """a decimal string as the short number a card prints"""
    return ("%." + str(nd) + "g") % float(x)


# --------------------------------------------------------------------------------------- the drawings (inline SVG, colours from the page's tokens)

COLORS = ["var(--accent)", "var(--good)", "var(--warn)", "var(--bad)"]


def svg(title, desc, body, w=320, h=190):
    n = int(hashlib.sha256(title.encode()).hexdigest()[:6], 16) % 100000
    return ('<svg viewBox="0 0 %d %d" role="img" aria-labelledby="t%d d%d"><title id="t%d">%s</title><desc id="d%d">%s</desc>%s</svg>'
            % (w, h, n, n, n, esc(title), n, esc(desc), body))


def scale(vals, lo, hi):
    a, b = min(vals), max(vals)
    if b == a:
        b = a + 1.0
    return lambda v: lo + (float(v) - a) / (b - a) * (hi - lo)


def scatter(xs, ys, cls, title, desc, xlab, ylab, diag=False, rings=(), w=320, h=190):
    sx, sy = scale(xs + [r[0] for r in rings], 34, w - 12), scale(ys + [r[1] for r in rings], h - 28, 12)
    b = ['<line class="ax" x1="34" y1="%d" x2="%d" y2="%d"/><line class="ax" x1="34" y1="12" x2="34" y2="%d"/>' % (h - 28, w - 12, h - 28, h - 28)]
    if diag:
        lo, hi = min(xs + ys), max(xs + ys)
        s = scale([lo, hi], 34, w - 12), scale([lo, hi], h - 28, 12)
        b.append('<line class="dg" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (s[0](lo), s[1](lo), s[0](hi), s[1](hi)))
        sx, sy = s
    for x, y, c in zip(xs, ys, cls):
        b.append('<circle class="p c%d" cx="%.1f" cy="%.1f" r="3.2"/>' % (c % 4, sx(x), sy(y)))
    for x, y, c in rings:
        b.append('<circle class="rg c%d" cx="%.1f" cy="%.1f" r="9"/>' % (c % 4, sx(x), sy(y)))
    b.append('<text class="lb" x="%d" y="%d" text-anchor="end">%s</text><text class="lb" x="38" y="11">%s</text>' % (w - 12, h - 10, esc(xlab), esc(ylab)))
    return svg(title, desc, "".join(b), w, h)


def bars(items, title, desc, unit="", w=320, h=190, hi=None):
    """items: [(label, value)]"""
    top = hi if hi is not None else max(v for _, v in items) or 1.0
    rowh = (h - 14) / len(items)
    b = []
    for i, (lab, v) in enumerate(items):
        y = 8 + i * rowh
        wd = max(1.0, (w - 150) * v / top)
        b.append('<text class="lb" x="104" y="%.1f" text-anchor="end">%s</text><rect class="bar c%d" x="110" y="%.1f" width="%.1f" height="%.1f" rx="3"/><text class="lb" x="%.1f" y="%.1f">%s</text>'
                 % (y + rowh * 0.62, esc(lab), i % 2, y + 2, wd, rowh - 6, 114 + wd, y + rowh * 0.62, esc(f(v, 3) + unit)))
    return svg(title, desc, "".join(b), w, h)


def matrix(m, labels, title, desc, w=320, h=190):
    n = len(m)
    cell = min(40, (h - 40) / n)
    mx = max(max(r) for r in m) or 1
    b = []
    ox, oy = (w - cell * n) / 2, 24
    for i, row in enumerate(m):
        for j, v in enumerate(row):
            op = 0.12 + 0.88 * v / mx
            b.append('<rect class="%s" x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="3" style="opacity:%.2f"/><text class="lb mid" x="%.1f" y="%.1f" text-anchor="middle">%d</text>'
                     % ("cell hit" if i == j else "cell", ox + j * cell, oy + i * cell, cell - 3, cell - 3, op if i == j else max(op, 0.25) if v else 0.1, ox + j * cell + (cell - 3) / 2, oy + i * cell + cell / 2 + 3, v))
    b.append('<text class="lb" x="%.1f" y="12" text-anchor="middle">predicted</text><text class="lb" x="8" y="%.1f">actual</text>' % (w / 2, oy + cell * n / 2))
    return svg(title, desc, "".join(b), w, h)


def short(col):
    parts = [p for p in col.split("_") if p != "cm"]
    return "".join(p[0] for p in parts) if len(parts) > 1 else col[:4]


def tree_svg(nodes, title, desc, w=320, h=190):
    by = {n["id"]: n for n in nodes}
    pos = {}
    maxd = max(n["depth"] for n in nodes)
    leaves = [n["id"] for n in nodes if n["leaf"]]
    order = []

    def walk(i):
        n = by[i]
        if n["leaf"]:
            order.append(i)
        else:
            walk(n["left"])
            walk(n["right"])
    walk(0)
    for k, i in enumerate(order):
        pos[i] = (20 + k * (w - 40) / max(1, len(order) - 1), 0)

    def place(i):
        n = by[i]
        if n["leaf"]:
            return pos[i][0]
        x = (place(n["left"]) + place(n["right"])) / 2
        pos[i] = (x, 0)
        return x
    place(0)
    b = []
    for n in nodes:
        x, _ = pos[n["id"]]
        y = 18 + n["depth"] * (h - 46) / max(1, maxd)
        pos[n["id"]] = (x, y)
    for n in nodes:
        if not n["leaf"]:
            for c in (n["left"], n["right"]):
                b.append('<line class="ax" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (pos[n["id"]][0], pos[n["id"]][1], pos[c][0], pos[c][1]))
    for n in nodes:
        x, y = pos[n["id"]]
        if n["leaf"]:
            c = 1 if float(n["value"]) >= 0.5 else 0
            b.append('<circle class="p c%d" cx="%.1f" cy="%.1f" r="7"/><text class="lb mid" x="%.1f" y="%.1f" text-anchor="middle">%d</text>' % (c, x, y, x, y + 3, n["rows"]))
        else:
            b.append('<rect class="nd" x="%.1f" y="%.1f" width="58" height="16" rx="5"/><text class="lb mid" x="%.1f" y="%.1f" text-anchor="middle">%s ≤ %s</text>'
                     % (x - 29, y - 8, x, y + 3, esc(short(n["column"])), esc(n["threshold"])))
    return svg(title, desc, "".join(b), w, h)


# --------------------------------------------------------------------------------------- the cards


class Card:
    def __init__(self, kind, key, title, blurb, cmds, svg_, facts, note="", status=""):
        self.status = status
        self.kind, self.key, self.title, self.blurb, self.cmds, self.svg, self.facts, self.note = kind, key, title, blurb, cmds, svg_, facts, note

    def html(self):
        cmd = "\n".join('<span class="c">$</span> %s' % esc(c.split("  #")[0]) for c in self.cmds)
        facts = "".join("<li>%s</li>" % x for x in self.facts)
        note = '<p class="note">%s</p>' % esc(self.note) if self.note else ""
        return ('<article class="gcard" data-kind="%s" id="%s"><div class="viz">%s</div><div class="tx"><span class="tag">%s</span><h3>%s</h3><p>%s</p>%s<ul class="facts">%s</ul>'
                '<pre class="code" tabindex="0">%s</pre></div></article>'
                % (self.kind, self.key, self.svg, self.status or ("model" if self.kind == "model" else "option"), esc(self.title), self.blurb, note, facts, cmd))


def models_available(lab):
    st, doc, out = lab.run("ml introspect")
    flags = {x["name"]: x for x in doc["flags"]}
    return set(flags["--model"]["kind"].split(":")[1].split("/"))


def three_blobs(seed=7, n=40):
    rng = random.Random(seed)
    rows = []
    centres = [(1.0, 1.0), (4.0, 1.5), (2.5, 4.2)]
    for c, (cx, cy) in enumerate(centres):
        for _ in range(n):
            rows.append("%.3f,%.3f,%d" % (rng.gauss(cx, .7), rng.gauss(cy, .7), c))
    rng.shuffle(rows)
    return "x,y,class\n" + "\n".join(rows[: 2 * n]) + "\n", "x,y,class\n" + "\n".join(rows[2 * n:]) + "\n"


def ranking_data(seed=11, queries=30, size=8):
    rng = random.Random(seed)
    out = ["qid,f1,f2,rel"]
    for q in range(queries):
        for _ in range(size):
            a, b = rng.random(), rng.random()
            rel = min(3, max(0, int(round(2.2 * a + 1.2 * b - 0.9 + rng.gauss(0, .35)))))
            out.append("q%02d,%.3f,%.3f,%d" % (q, a, b, rel))
    return "\n".join(out) + "\n"


def rating_log(seed=5, users=30, items=12):
    """A log of users and items in three taste groups: a user rates about half the items, high where the item belongs to the user's group."""
    rnd = random.Random(seed)
    rows = ["user,item,rating"]
    for u in range(users):
        for i in range(items):
            if rnd.random() < 0.55:
                base = 5 if i % 3 == u % 3 else 2
                rows.append("u%02d,i%02d,%d" % (u, i, max(1, min(5, base + rnd.choice([-1, 0, 0, 1])))))
    return "\n".join(rows) + "\n"


def make_cards(lab):
    have = models_available(lab)
    cards = []
    # --- the baseline ladder: constant, linear and forest on the diabetes test file
    base = ["ml fit --model constant --target progression --output c.ml diabetes_train.csv", "ml fit --model linear --target progression --step \"%s\" --l2 0.1 --output l.ml diabetes_train.csv" % STD,
            "ml fit --model forest --task regression --target progression --trees 50 --seed 1 --output f.ml diabetes_train.csv", "ml fit --model boost --task regression --target progression --rounds 100 --learning-rate 0.1 --seed 1 --output b.ml diabetes_train.csv"]
    rm = {}
    for c, name in zip(base, ("constant", "linear", "forest", "boost")):
        if name in have:
            lab.ok(c)
            rm[name] = float(lab.ok("ml eval --model-file %s.ml diabetes_test.csv" % name[0])["metrics"]["rmse"])
    cards.append(Card("model", "constant", "constant: the baseline to beat", "The mean of the target, fitted and scored like any model. If a model cannot beat it, it learned nothing.",
                      [base[0], "ml eval --model-file c.ml diabetes_test.csv"],
                      bars([(k, v) for k, v in rm.items()], "RMSE on the diabetes test file, by model", "Bars of the RMSE of the constant, linear, forest and boost models, from ml eval.", w=320, h=150),
                      ["RMSE of the constant model <b>%s</b>; of the best of the four <b>%s</b>" % (f(rm["constant"], 4), f(min(rm.values()), 4))]))
    # --- linear
    d = lab.ok("ml predict --model-file l.ml --drop progression diabetes_test.csv --limit 200")
    head, rows = lab.csv("diabetes_test.csv")
    actual = [float(r[head.index("progression")]) for r in rows]
    pred = [float(r[0]) for r in d["rows"]]
    ev = lab.ok("ml eval --model-file l.ml diabetes_test.csv")["metrics"]
    cards.append(Card("model", "linear", "linear: least squares, ridge with --l2", "Exact sums in a fixed order, so the coefficients are the same on every platform.",
                      [base[1], "ml predict --model-file l.ml diabetes_test.csv"],
                      scatter(actual, pred, [0] * len(actual), "Predicted against actual progression, 112 test rows", "Each dot is a test row; the line is where a perfect prediction falls.", "actual", "predicted", diag=True),
                      ["R² <b>%s</b>, RMSE <b>%s</b> on rows the model never saw" % (f(ev["r2"], 3), f(ev["rmse"], 4))]))
    if "boost" in rm:
        lab.ok("ml predict --model-file b.ml --drop progression diabetes_test.csv --limit 200")
    # --- logistic on iris
    lab.ok("ml fit --model logistic --target virginica --step \"%s\" --l2 0.1 --output g.ml iris_train.csv" % STD)
    d = lab.ok("ml predict --model-file g.ml --proba --drop virginica iris_test.csv --limit 100")
    head, rows = lab.csv("iris_test.csv")
    xs, ys = [float(r[2]) for r in rows], [float(r[3]) for r in rows]
    labs = [int(r[0]) for r in d["rows"]]
    ev = lab.ok("ml eval --model-file g.ml iris_test.csv")["metrics"]
    cards.append(Card("model", "logistic", "logistic: Newton's method, with the probability", "Two classes, a label and the probability of the positive one with <code>--proba</code>.",
                      ["ml fit --model logistic --target virginica --step \"%s\" --l2 0.1 --output g.ml iris_train.csv" % STD, "ml predict --model-file g.ml --proba iris_test.csv"],
                      scatter(xs, ys, labs, "Iris test rows coloured by the predicted class", "Petal length against petal width; blue is class 0, green is class 1, as predicted.", "petal length", "petal width"),
                      ["accuracy <b>%s</b>, ROC-AUC <b>%s</b>, log-loss <b>%s</b>" % (f(ev["accuracy"], 3), f(ev["roc_auc"], 3), f(ev["log_loss"], 3))]))
    # --- softmax on three synthetic blobs
    if "softmax" in have:
        tr, te = three_blobs()
        lab.write("blobs_train.csv", tr)
        lab.write("blobs_test.csv", te)
        lab.ok("ml fit --model softmax --target class --l2 0.1 --output s.ml blobs_train.csv")
        d = lab.ok("ml predict --model-file s.ml --drop class blobs_test.csv --limit 100")
        head, rows = lab.csv("blobs_test.csv")
        ev = lab.ok("ml eval --model-file s.ml blobs_test.csv")["metrics"]
        pl = [int(r[0]) for r in d["rows"]]
        cards.append(Card("model", "softmax", "softmax: three or more classes", "One model for any number of classes, with a confusion matrix.",
                          ["ml fit --model softmax --target class --l2 0.1 --output s.ml blobs_train.csv", "ml eval --model-file s.ml blobs_test.csv"],
                          scatter([float(r[0]) for r in rows], [float(r[1]) for r in rows], pl, "Three blobs coloured by the predicted class", "Test rows of three seeded blobs; the colour is the predicted class.", "x", "y"),
                          ["accuracy <b>%s</b>; confusion matrix in the answer: %s" % (f(ev["accuracy"], 3), "".join("[%s]" % ",".join(str(v) for v in r) for r in ev["confusion"]))],
                          note="Synthetic: three seeded blobs written by scripts/gallery.py, because the bundled files have two classes."))
    # --- tree
    lab.ok("ml fit --model tree --task classification --target virginica --max-depth 3 --output t.ml iris_train.csv")
    d = lab.ok("ml dump --model-file t.ml")
    ev = lab.ok("ml eval --model-file t.ml iris_test.csv")["metrics"]
    cards.append(Card("model", "tree", "tree: a CART tree you can print", "<code>ml dump</code> prints every node: the column, the threshold and the rows it holds.",
                      ["ml fit --model tree --task classification --target virginica --max-depth 3 --output t.ml iris_train.csv", "ml dump --model-file t.ml"],
                      tree_svg(d["list"], "The tree fitted on iris, as ml dump prints it", "Each box is a split on a column and a threshold; each circle is a leaf with the number of rows it holds."),
                      ["%d nodes, depth %d; accuracy <b>%s</b> on the test file" % (d["nodes"], d["depth"], f(ev["accuracy"], 3))]))
    # --- forest importance
    lab.ok("ml fit --model forest --task classification --target benign --trees 50 --seed 1 --output rf.ml breast_cancer_train.csv")
    d = lab.ok("ml importance --model-file rf.ml")
    cols = d["importance"]["columns"][:6]
    ev = lab.ok("ml eval --model-file rf.ml breast_cancer_test.csv")["metrics"]
    cards.append(Card("model", "forest", "forest: bagged trees, and which columns they used", "Trees fitted in parallel, with the same bytes at any thread count. <code>ml importance</code> says which columns mattered.",
                      ["ml fit --model forest --task classification --target benign --trees 50 --seed 1 --output rf.ml breast_cancer_train.csv", "ml importance --model-file rf.ml"],
                      bars([(c["column"].replace("mean_", "")[:14], float(c["importance"])) for c in cols], "The six columns the forest relied on most", "Bars of the impurity decrease of each column, from ml importance.", w=320, h=170),
                      ["accuracy <b>%s</b> on 143 held-out rows; the six columns carry <b>%s</b> of the decrease" % (f(ev["accuracy"], 3), f(sum(float(c["importance"]) for c in cols), 3))]))
    # --- boost
    if "boost" in have:
        cards.append(Card("model", "boost", "boost: gradient-boosted trees, with early stopping", "Rounds fitted to the gradients of the loss; <code>--early-stopping</code> stops when a held-out part stops improving.",
                          [base[3], "ml eval --model-file b.ml diabetes_test.csv"],
                          scatter(actual, [float(r[0]) for r in lab.ok("ml predict --model-file b.ml --drop progression diabetes_test.csv --limit 200")["rows"]], [1] * len(actual), "Boost: predicted against actual progression", "Each dot is a test row; the line is a perfect prediction.", "actual", "predicted", diag=True),
                          ["RMSE <b>%s</b>, against <b>%s</b> for the linear model and <b>%s</b> for the forest" % (f(rm["boost"], 4), f(rm["linear"], 4), f(rm["forest"], 4))]))
    # --- kmeans
    lab.ok("ml fit --model kmeans --k 3 --seed 1 --output k.ml --drop virginica iris_train.csv")
    d = lab.ok("ml predict --model-file k.ml --drop virginica iris_test.csv --limit 100")
    cen = lab.ok("ml dump --model-file k.ml")["centroids"]
    head, rows = lab.csv("iris_test.csv")
    ci = lambda c, name: float(next(v["value"] for v in cen[c]["values"] if v["column"] == name))
    cards.append(Card("model", "kmeans", "kmeans: clusters with no target", "A seeded initialisation and a fixed number of passes: the same clusters every time.",
                      ["ml fit --model kmeans --k 3 --seed 1 --drop virginica --output k.ml iris_train.csv", "ml predict --model-file k.ml --drop virginica iris_test.csv"],
                      scatter([float(r[2]) for r in rows], [float(r[3]) for r in rows], [int(r[0]) for r in d["rows"]], "Iris test rows by cluster, with the three centres", "Petal length against petal width; the colour is the cluster, the rings are the centres from ml dump.", "petal length", "petal width",
                              rings=[(ci(c, "petal_length_cm"), ci(c, "petal_width_cm"), c) for c in range(3)]),
                      ["k = 3; the centres are the rings; each test row gets its cluster, and <code>--distance</code> adds its squared distance"]))
    # --- ranking
    if "boost" in have:
        lab.write("rank.csv", ranking_data())
        res = {}
        for task in ("ranking", "lambdarank"):
            lab.ok("ml fit --model boost --task %s --group qid --target rel --rounds 60 --seed 1 --output r_%s.ml rank.csv" % (task, task))
            m = lab.ok("ml eval --model-file r_%s.ml --group qid --rank-k 5 rank.csv" % task)["metrics"]
            res[task] = float(m["ndcg"])
        cards.append(Card("model", "ranking", "ranking: pairwise or LambdaRank", "Rows ranked inside a group (a query), scored by NDCG. The group stays on one side of every split.",
                          ["ml fit --model boost --task lambdarank --group qid --target rel --rounds 60 --output r.ml rank.csv", "ml eval --model-file r.ml --group qid --rank-k 5 rank.csv"],
                          bars([("pairwise", res["ranking"]), ("lambdarank", res["lambdarank"])], "NDCG@5 of the two ranking objectives", "Two bars: the NDCG at 5 of the pairwise objective and of LambdaRank, from ml eval.", w=320, h=110, hi=1.0),
                          ["NDCG@5 <b>%s</b> (pairwise) and <b>%s</b> (LambdaRank), on the training queries" % (f(res["ranking"], 3), f(res["lambdarank"], 3))],
                          note="Synthetic: 30 seeded queries of 8 rows, written by scripts/gallery.py; scored on the rows it was fitted on, to show the flags."))
    # --- recommender
    if "recommend" in have:
        lab.write("ratings.csv", rating_log())
        lab.write("who.csv", "who\nu00\n")
        d = lab.ok("ml fit --model recommend --user user --item item --rating rating --factors 3 --iterations 10 --seed 1 --output rec.ml ratings.csv")
        a = lab.ok("ml predict --model-file rec.ml --for who --top 5 who.csv")
        sm = d["summary"]
        cards.append(Card("model", "recommend", "recommend: what a user has not seen yet", "Matrix factorisation (ALS) from a log of user, item, rating; <code>predict --for</code> answers the top items a user has not rated.",
                          ["ml fit --model recommend --user user --item item --rating rating --factors 3 --seed 1 --output rec.ml ratings.csv", "ml predict --model-file rec.ml --for who --top 5 who.csv"],
                          bars([(r[2], float(r[3])) for r in a["rows"]], "The top five items for user u00, by score", "Five bars: the predicted score of the five items the model puts first for user u00, from ml predict.", "", 320, 190),
                          ["%d users, %d items, %d interactions; the loss is <b>%s</b>" % (sm["users"], sm["items"], sm["interactions"], f(float(sm["loss"]), 2)),
                           "items the user already rated are left out (<code>--keep-seen</code> keeps them)"],
                          note="Synthetic: 30 users in three taste groups, written by scripts/gallery.py."))
    # --- options: text columns
    lab.ok("ml fit --model logistic --target adopted --cat species,city --step \"encode column=species,city kind=onehot\" --step \"scale column=* kind=standard\" --l2 0.5 --drop id --output p.ml pets.csv") if False else None
    st, doc, out = lab.run("ml fit --model logistic --target adopted --drop id --cat species,city --step 'encode column=species,city kind=onehot; scale column=* kind=standard' --l2 0.5 --output p.ml pets.csv")
    if st == 0:
        ds = lab.ok("ml describe --model-file p.ml")
        cats = ds["categorical"]
        items = [(c["column"], c["categories"]) for c in cats]
        cards.append(Card("option", "text", "text columns, by a written rule", "<code>--cat</code> names the text columns; a step encodes them; a new category at predict time is counted, never guessed.",
                          ["ml fit --model logistic --target adopted --drop id --cat species,city --step 'encode column=species,city kind=onehot; scale column=* kind=standard' --output p.ml pets.csv", "ml describe --model-file p.ml"],
                          bars([(a, float(b)) for a, b in items], "Categories the model file holds, by text column", "Bars of how many distinct values the file stores for each text column.", w=320, h=100),
                          ["the model file holds %s; <code>describe</code> lists them" % ", ".join("%d %s" % (b, a) for a, b in items)]))
    # --- options: time split
    st, doc, out = lab.run("ml fit --model linear --target basket --drop customer --drop visit_date --split-by time:visit_date --split train=0.75 --output v.ml visits.csv")
    st2, doc2, out2 = lab.run("ml fit --model linear --target basket --drop customer,visit_date --split-by time:visit_date --split train=0.75 --output v.ml visits.csv")
    if st2 == 0:
        sp = doc2["data"]["summary"]["split"]
        cards.append(Card("option", "split", "splits that cannot leak", "<code>--split-by time:COLUMN</code> trains on the past and scores the newest rows; <code>--stratify</code> and <code>--split-group</code> keep classes and groups apart.",
                          ["ml fit --model linear --target basket --drop customer,visit_date --split-by time:visit_date --split train=0.75 --output v.ml visits.csv"],
                          bars([("train (older)", float(sp["train"])), ("test (newest)", float(sp["test"]))], "Rows on each side of a time split", "Two bars: the older rows the model trains on and the newest rows it is scored on.", w=320, h=100),
                          ["%d older rows train, %d newer rows are scored: the model never sees the future" % (sp["train"], sp["test"])]))
    # --- options: cross-validation
    st, doc, out = lab.run("ml cv --model logistic --target virginica --step '%s' --l2 0.1 --folds 5 --seed 3 iris_train.csv" % STD)
    if st == 0:
        res = doc["data"]["results"][:5]
        agg = doc["data"]["summary"]["metrics"]["accuracy"]
        cards.append(Card("option", "cv", "cross-validation, fitted per fold", "Every step is fitted on the training rows of each fold only, so a fold cannot leak into its own scoring.",
                          ["ml cv --model logistic --target virginica --step \"%s\" --l2 0.1 --folds 5 --seed 3 iris_train.csv" % STD],
                          bars([("fold %d" % r["fold"], float(r["metrics"]["accuracy"])) for r in res], "Accuracy of each of the five folds", "Five bars: the accuracy of the model on each held-out fold.", w=320, h=150, hi=1.0),
                          ["mean accuracy <b>%s</b>, sample std <b>%s</b> over %d folds" % (f(agg["mean"], 3), f(agg["std"], 3), agg["n"])]))
    # --- options: threads
    shas = []
    for t in (1, 8):
        d = lab.ok("ml fit --model forest --task classification --target benign --trees 50 --seed 1 --threads %d --overwrite --output th%d.ml breast_cancer_train.csv" % (t, t))
        shas.append(d["sha256"])
    cards.append(Card("option", "threads", "the same bytes at any thread count", "<code>--threads</code> changes the time, not one byte of the model file.",
                      ["ml fit --model forest ... --seed 1 --threads 1 --output th1.ml breast_cancer_train.csv", "ml fit --model forest ... --seed 1 --threads 8 --output th8.ml breast_cancer_train.csv"],
                      svg("The SHA-256 of the model file at 1 and at 8 threads", "Two rows with the first hex digits of the model file's SHA-256 at one thread and at eight; they are equal.",
                          '<text class="lb big" x="20" y="60">1 thread</text><text class="lb big good" x="130" y="60">%s…</text><text class="lb big" x="20" y="110">8 threads</text><text class="lb big good" x="130" y="110">%s…</text><text class="lb" x="20" y="160">%s</text>'
                          % (shas[0][:12], shas[1][:12], "equal" if shas[0] == shas[1] else "DIFFERENT"), 320, 180),
                      ["SHA-256 <b>%s…</b> at one thread and at eight: <b>%s</b>" % (shas[0][:12], "equal" if shas[0] == shas[1] else "different")]))
    if shas[0] != shas[1]:
        raise Drift("the forest differs between 1 and 8 threads: a bug, not a gallery problem")
    # --- options: dry run
    d = lab.ok("ml fit --model linear --target progression --output dry.ml --dry-run diabetes_train.csv")
    cards.append(Card("option", "dry", "look before it writes", "<code>--dry-run</code> reads, checks and fits, then says what it would write and exits 9. Nothing is written.",
                      ["ml fit --model linear --target progression --output dry.ml --dry-run diabetes_train.csv"],
                      svg("A dry run plans a file and writes none", "A box for the planned file with its size and the words not written.",
                          '<rect class="nd" x="40" y="40" width="240" height="90" rx="12" style="stroke-dasharray:6 5"/><text class="lb big" x="160" y="82" text-anchor="middle">dry.ml</text><text class="lb" x="160" y="108" text-anchor="middle">%d bytes planned · not written</text>' % d["bytes"], 320, 170),
                      ["exit status <b>9</b>; <code>planned_actions</code> names the file, its size and its SHA-256"]))
    return cards


# What is asked for and not built. Each entry is run for real: the answer it gets today is printed on the card, and a command that starts to work is a Drift
# (the card then becomes a real one). `status` is the tag: "soon" has an open issue and a design; "spike" and "not planned" are said as they are.
SOON = [
    ("knn", "k-nearest neighbours", "soon", 39, "Exact, bounded search with a stated tie rule; the model file holds the training rows, so its size is a limit.",
     "ml fit --model knn --target progression --output k.ml diabetes_train.csv", "the neighbours: three of the training rows closest to a new one"),
    ("nb", "naive Bayes", "soon", 38, "Gaussian, multinomial and categorical, counted exactly before the last logarithm.",
     "ml fit --model nb --target virginica --output k.ml iris_train.csv", "class bars"),
    ("lasso", "lasso and elastic net", "soon", 40, "<code>--l1</code> by coordinate descent in a fixed cyclic order; coefficients driven to exactly zero.",
     "ml fit --model linear --target progression --l1 0.1 --output k.ml diabetes_train.csv", "coefficient bars with zeros"),
    ("pca", "PCA, polynomial features, robust scaling", "soon", 41, "More pipeline steps, fitted on the training rows and stored in the model file like every step.",
     'ml fit --model linear --target progression --step "pca components=2" --output k.ml diabetes_train.csv', "axes"),
    ("forecast", "forecasting", "soon", 58, "Lag and rolling features, naive baselines to beat, multi-step forecasts with intervals. Today <code>--split-by time</code> and <code>ml cv</code> judge a forecast made elsewhere honestly.",
     "ml forecast --horizon 7 sales.csv", "a line with an interval band"),
    ("bandit", "bandits and off-policy evaluation", "soon", 59, "Sample weights, <code>ml ope</code> (the value of a policy from a log of decisions) and policies learned by reduction. Online learning in the server is out of scope.",
     "ml ope --model-file k.ml iris_test.csv", "arms"),
    ("recommend", "recommender", "soon", 70, "Matrix factorisation from a log of user, item, rating, with top-n answers per user. In review, not yet released.",
     "ml fit --model recommend --target rating --output k.ml iris_train.csv", "grid"),
    ("gpu", "GPU kernels", "spike", 47, "Only a measured spike, and a separate numerical tier if it comes: single precision cannot give the CPU's bytes. No promise is made.",
     "ml fit --model linear --target progression --device gpu --output k.ml diabetes_train.csv", "chip"),
    ("deep", "neural networks", "not planned", 13, "A deep-learning framework is not the aim: no GPU path, no autodiff. The classical models are what is built.",
     "ml fit --model mlp --target progression --output k.ml diabetes_train.csv", "layers"),
]

SKETCH = {
    "knn": '<circle class="p c0" cx="70" cy="70" r="6"/><circle class="p c0" cx="95" cy="95" r="6"/><circle class="p c1" cx="190" cy="60" r="6"/><circle class="p c1" cx="215" cy="100" r="6"/><circle class="p c0" cx="120" cy="60" r="6"/><circle class="rg" cx="140" cy="85" r="9"/><circle class="dg" cx="140" cy="85" r="48"/>',
    "nb": '<rect class="bar" x="60" y="40" width="150" height="16" rx="3"/><rect class="bar" x="60" y="70" width="90" height="16" rx="3"/><rect class="bar" x="60" y="100" width="40" height="16" rx="3"/>',
    "lasso": '<line class="ax" x1="40" y1="85" x2="280" y2="85"/>' + "".join('<rect class="cell" x="%d" y="%d" width="22" height="%d"/>' % (50 + 34 * i, 85 - h if h > 0 else 85, abs(h)) for i, h in enumerate([50, -30, 0, 38, 0, -22, 0])),
    "pca": '<line class="ax" x1="40" y1="130" x2="280" y2="130"/><line class="ax" x1="40" y1="130" x2="40" y2="20"/><line class="dg" x1="50" y1="120" x2="260" y2="40"/><circle class="p c0" cx="90" cy="105" r="5"/><circle class="p c0" cx="150" cy="85" r="5"/><circle class="p c0" cx="200" cy="62" r="5"/><circle class="p c0" cx="120" cy="90" r="5"/>',
    "forecast": '<path class="dg" d="M40 110 L80 90 L120 100 L160 70 L200 80" fill="none"/><path d="M200 80 L280 50 L280 100 Z" class="cell" opacity=".25"/><line class="rg" x1="200" y1="80" x2="280" y2="75"/>',
    "bandit": ''.join('<circle class="p c%d" cx="%d" cy="85" r="%d"/>' % (i, 70 + 70 * i, r) for i, r in enumerate([18, 28, 12])),
    "recommend": ''.join('<rect class="cell" opacity="%s" x="%d" y="%d" width="28" height="22" rx="3"/>' % (o, 70 + 34 * c, 40 + 28 * r_) for r_, row in enumerate([[.9, .2, .5, .2], [.3, .8, .2, .6], [.6, .3, .9, .3]]) for c, o in enumerate(row)),
    "gpu": '<rect class="nd" x="100" y="45" width="120" height="80" rx="10"/><text class="lb big" x="160" y="92" text-anchor="middle">f32</text>',
    "deep": ''.join('<circle class="p c3" cx="%d" cy="%d" r="7"/>' % (70 + 60 * l, 45 + 40 * k) for l in range(4) for k in range(3)),
}


def soon_cards(lab, have):
    out = []
    for key, title, status, issue, blurb, cmd, _ in SOON:
        if key == "recommend" and "recommend" in have:
            continue
        st, doc, raw = lab.run(cmd)
        if st == 0 or not doc or doc.get("ok"):
            raise Drift("%s now works (exit %d): turn its card into a real one: %s" % (title, st, cmd))
        rule = doc["error"]["rule"]
        label = {"soon": "coming soon", "spike": "spike, not promised", "not planned": "not planned"}[status]
        viz = svg("%s: not built yet" % title, "A sketch of %s with the words %s." % (title, label),
                  '<g opacity=".8">%s</g><text class="lb big" x="160" y="156" text-anchor="middle">%s</text>' % (SKETCH[key], label), 320, 170)
        out.append(Card("soon", key, title, blurb, [cmd], viz,
                        ["today: refused, exit <b>%d</b>, rule <code>%s</code>" % (st, esc(rule)),
                         'tracked in <a href="https://github.com/alpibrusl/cancho-ml/issues/%d">#%d</a>' % (issue, issue)], status=status))
    return out


def render(binary):
    with tempfile.TemporaryDirectory() as tmp:
        lab = Lab(binary, tmp)
        cards = make_cards(lab)
        cards += soon_cards(lab, models_available(lab))
    return "\n".join(c.html() for c in cards)


def splice(text, body):
    a, b = "<!-- gen:cards -->\n", "<!-- /gen:cards -->"
    i, j = text.find(a), text.find(b)
    if i < 0 or j < 0:
        raise Drift("docs/gallery.html has no gen:cards region")
    return text[: i + len(a)] + body + "\n" + text[j:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", default=str(ROOT / "build" / "ml"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    old = PAGE.read_text()
    new = splice(old, render(Path(args.bin).resolve()))
    if new == old:
        print("gallery.py: the cards are current")
        return 0
    if args.check:
        sys.stderr.write("gallery.py: docs/gallery.html differs from what the binary produces; run `python3 scripts/gallery.py`\n")
        return 1
    PAGE.write_text(new)
    print("wrote docs/gallery.html")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Drift as exc:
        sys.exit("gallery.py: %s" % exc)

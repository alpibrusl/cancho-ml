"""The worked examples for scripts/site.py (docs/examples.md): run every `ml` command of the examples on the committed
CSVs, read scikit-learn's results from docs/examples/reference.json (written once by scripts/examples_data.py; **this
module never runs scikit-learn**), and produce the regions of docs/examples.html and README.md. Standard library only.
"""
import hashlib
import html
import json
import re
import shlex
import shutil
from pathlib import Path

import examples_common as common

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = {e["key"]: e for e in common.EXAMPLES}


class Drift(Exception):
    pass


def e(s):
    return html.escape(s, quote=False)


def reference():
    return json.loads((ROOT / common.REFERENCE).read_text())


def sk_label(ref):
    return "scikit-learn " + ref["generated_by"]["scikit-learn"]


# --- running ------------------------------------------------------------------------------------------------------


def run_examples(run, tmp, ref):
    """Copy the committed CSVs into the working directory, check them against the reference, run every command once.
    Returns {(key, run_id): {"fit": (typed, out, status), "eval": ..., "eval_train": ..., "predict": ...}}."""
    res = {}
    for key, ex in EXAMPLES.items():
        for side in ("train", "test"):
            meta = ref["examples"][key]["files"][side]
            src = ROOT / meta["path"]
            if hashlib.sha256(src.read_bytes()).hexdigest() != meta["sha256"]:
                raise Drift("%s is not the file docs/examples/reference.json was computed on: run scripts/examples_data.py" % meta["path"])
            if src.read_bytes().count(b"\n") - 1 != meta["rows"]:
                raise Drift("%s has not %d rows" % (meta["path"], meta["rows"]))
            shutil.copy(src, Path(tmp) / ("%s_%s.csv" % (key, side)))
        for r in ex["runs"]:
            d = {}
            argv = common.fit_argv(ex, r)
            d["fit"] = (shlex.join(argv), *run(argv))
            if d["fit"][2] == 0:
                if r.get("refusal"):
                    raise Drift("%s %s was expected to be refused" % (key, r["id"]))
                a = common.eval_argv(ex, r)
                d["eval"] = (shlex.join(a), *run(a))
                if r.get("eval_refusal"):
                    if d["eval"][2] != 8:
                        raise Drift("%s %s: the eval was expected to be refused" % (key, r["id"]))
                    res[(key, r["id"])] = d
                    continue
                if r.get("eval_train"):
                    a = common.eval_argv(ex, r)
                    a[-1] = "%s_train.csv" % key
                    d["eval_train"] = (shlex.join(a), *run(a))
                if r.get("ref"):
                    a = common.proba_argv(ex, r)
                    if ex["task"] == "regression":
                        a.remove("--proba")
                    d["predict"] = (shlex.join(a), *run(a))
            elif not r.get("refusal"):
                raise Drift("%s %s fails with status %d" % (key, r["id"], d["fit"][2]))
            res[(key, r["id"])] = d
    return res


# --- reading the outputs -------------------------------------------------------------------------------------------


def doc_of(entry):
    return json.loads(entry[1])


def fit_values(res, key, rid):
    s = doc_of(res[(key, rid)]["fit"])["data"]["summary"]
    return [(c["name"], c["value"]) for c in s["coefficients"]], s["intercept"]


def eval_metrics(res, key, rid, name="eval"):
    return doc_of(res[(key, rid)][name])["data"]["metrics"]


def fnum(x):
    return "%.1e" % x if x != 0 else "0"


# --- comparison with the reference ---------------------------------------------------------------------------------


def compare(res, ref, key, rid):
    """Rows of the comparison for one reference run: metrics, coefficients, predictions."""
    ex = EXAMPLES[key]
    run = next(r for r in ex["runs"] if r["id"] == rid)
    rr = ref["examples"][key]["runs"][rid]
    kind = run["model"]
    rows = []
    order = common.REG_ORDER if ex["task"] == "regression" else common.CLASS_ORDER
    mlm = eval_metrics(res, key, rid)
    for m in order:
        mv, sv = float(mlm[m]), rr["metrics"][m]
        d, ok = common.within(common.METRIC_TOL[m], mv, sv)
        rows.append(dict(name=m, ml=mlm[m], sk=repr(sv), diff=d, ok=ok, tol=common.allowed(common.METRIC_TOL[m], sv)))
    coef, icpt = fit_values(res, key, rid)
    names = [n for n, _ in coef]
    if names != [c["name"] for c in rr["coef"]]:
        raise Drift("%s %s: the coefficient names of ml and of the reference differ" % (key, rid))
    crow = []
    for (n, v), c in zip(coef, rr["coef"]):
        d, ok = common.within(common.COEF_TOL[kind], float(v), c["value"])
        crow.append(dict(name=n, ml=v, sk=repr(c["value"]), diff=d, ok=ok))
    d, ok = common.within(common.COEF_TOL[kind], float(icpt), rr["intercept"])
    crow.append(dict(name="intercept", ml=icpt, sk=repr(rr["intercept"]), diff=d, ok=ok))
    if "newton_coef" in rr:
        nv = rr["newton_coef"] + [rr["newton_intercept"]]
        for row, v in zip(crow, nv):
            row["nsk"] = repr(v)
            row["ndiff"] = abs(float(row["ml"]) - v)
            row["nok"] = common.within(common.COEF_TOL[kind], float(row["ml"]), v)[1]
    worst = max(crow, key=lambda r: r["diff"])
    tol = common.COEF_TOL[kind]
    rows.append(dict(name="largest coefficient or intercept difference (%d values)" % len(crow), ml="", sk="",
                     diff=worst["diff"], ok=all(r["ok"] for r in crow), tol="%g relative" % tol[0], at=worst["name"]))
    # the first rows of predict
    prow = []
    pr = doc_of(res[(key, rid)]["predict"])["data"]["rows"]
    for i, cells in enumerate(pr):
        if ex["task"] == "regression":
            sv = rr["pred_first"][i]
            d, ok = common.within(common.METRIC_TOL["rmse"], float(cells[0]), sv)
            prow.append(dict(i=i + 1, ml=cells[0], sk=repr(sv), diff=d, ok=ok))
        else:
            sv = rr["proba_first"][i]
            d, ok = common.within(common.METRIC_TOL["log_loss"], float(cells[1]), sv)
            ok = ok and int(cells[0]) == rr["label_first"][i]
            prow.append(dict(i=i + 1, ml=cells[1], sk=repr(sv), diff=d, ok=ok, mlabel=cells[0], slabel=str(rr["label_first"][i])))
    return dict(rows=rows, coef=crow, pred=prow, ref=rr, kind=kind)


KNOWN = {}  # (key, run id) -> an explanation written by hand for a difference this module cannot explain itself


def verdicts(cmp, key, rid):
    """The note that goes under the table when anything is outside its tolerance (docs/examples.md section 5)."""
    bad = [r["name"] for r in cmp["rows"] if not r["ok"]] + ["prediction row %d" % p["i"] for p in cmp["pred"] if not p["ok"]]
    if not bad:
        return None
    rr = cmp["ref"]
    if "newton_coef" in rr:
        wide = max(abs(c["value"] - n) for c, n in zip(rr["coef"], rr["newton_coef"]))
        nd = max(r["ndiff"] for r in cmp["coef"])
        if all(r["nok"] for r in cmp["coef"]):
            return ("Outside the tolerance of docs/examples.md section 3: %s. The cause is the reference solver and it is measured here: "
                    "scikit-learn's own two solvers differ from each other by up to %s in a coefficient (L-BFGS stops short of the "
                    "optimum), while ml differs from scikit-learn's Newton solver by at most %s, inside the same tolerance. "
                    "The tolerance was not moved." % (", ".join(bad), fnum(wide), fnum(nd)))
    if (key, rid) in KNOWN:
        return "Outside the tolerance: %s. %s" % (", ".join(bad), KNOWN[(key, rid)])
    raise Drift("%s %s: %s outside the tolerance and unexplained: write the cause in KNOWN (docs/examples.md section 5)" % (key, rid, ", ".join(bad)))


# --- the commands -------------------------------------------------------------------------------------------------

COMMENT = {
    ("breast_cancer", "naive"): "# The obvious command: no penalty (--l2 0), standardised features, 25 Newton iterations. The classes are separable, so it is refused, with the repair as an argv",
    ("breast_cancer", "fit"): "# The fix: a small ridge penalty, --l2 0.01. The optimum is now finite",
    ("breast_cancer", "fit", "eval"): "# Score the 143 held-out rows",
    ("breast_cancer", "fit", "predict"): "# Probabilities for the first five test rows",
    ("breast_cancer", "more"): "# Four times the iterations, to see whether the iteration count still matters",
    ("diabetes", "ols"): "# Least squares: no penalty",
    ("diabetes", "ols", "eval"): "# Score the held-out rows",
    ("diabetes", "ols", "predict"): "# The first five predictions",
    ("diabetes", "ridge"): "# Ridge: the penalty l2 is added to the diagonal of the normal equations, the intercept is not penalised",
    ("diabetes", "ridge", "eval"): "# Score the held-out rows",
    ("diabetes", "ridge", "predict"): "# The first five predictions",
    ("iris", "fit"): "# Logistic regression, no penalty, on the raw features: these two species overlap, so the optimum is finite",
    ("iris", "fit", "eval_train"): "# Score it on the training rows first: not every one is right",
    ("iris", "fit", "eval"): "# Score the held-out rows",
    ("iris", "fit", "predict"): "# Probabilities for the first five test rows",
    ("iris", "more"): "# Four times the iterations",
    ("wine", "typed"): "# First attempt: a small ridge penalty. The fit works",
    ("wine", "typed", "eval"): "# but scoring the test file is refused: color_intensity has 2 decimals in the train file and 6 in the test file, so the two files give it different types",
    ("wine", "naive"): "# The fix is in the fit: --float reads the column as a float. Now the obvious command, no penalty. The classes are separable, so it is refused, with the repair as an argv",
    ("wine", "fit"): "# The same small ridge penalty as the first attempt, with --float",
    ("wine", "fit", "eval"): "# Score the 45 held-out rows",
    ("wine", "fit", "predict"): "# Probabilities for the first five test rows",
    ("wine", "more"): "# Four times the iterations",
}


def shorten_fit(out):
    """The long `coefficients` array of a fit is cut after three members when it has more than ten; the page says so."""
    m = re.search(r'("coefficients":\[)(.*?)(\],"intercept")', out, re.S)
    if not m:
        return out
    parts = m.group(2).split("},{")
    if len(parts) <= 10:
        return out
    cut = "},{".join(parts[:3]) + "},{ ... %d more ... }" % (len(parts) - 3)
    return out[:m.start(2)] + cut + out[m.end(2):]


def cmd_items(res, key, only=None):
    """Items for scripts/site.py's renderer: (typed, output, status, note), with the comment lines between."""
    ex = EXAMPLES[key]
    items = []
    for r in ex["runs"]:
        if only is not None and r["id"] not in only:
            continue
        d = res[(key, r["id"])]
        for part in ("fit", "eval_train", "eval", "predict"):
            if part not in d:
                continue
            c = COMMENT.get((key, r["id"], part)) or (COMMENT.get((key, r["id"])) if part == "fit" else None)
            if c:
                items.append((c, None, 0, None))
            typed, out, status = d[part]
            if status != 0:
                err = json.loads(out)["error"]
                keys = [k for k in ("rule", "hint", "repair", "detail") if err.get(k)]
                out = json.dumps({k: err[k] for k in keys}, ensure_ascii=False)
                items.append((typed, out, status, "the %s of the JSON line it prints" % ", ".join(keys)))
                continue
            if part == "fit":
                out = shorten_fit(out)
            items.append((typed, out, 0, None))
    return items


def has_cut(res, key):
    for r in EXAMPLES[key]["runs"]:
        d = res[(key, r["id"])]
        if d["fit"][2] == 0 and shorten_fit(d["fit"][1]) != d["fit"][1]:
            return True
    return False


# --- tables ----------------------------------------------------------------------------------------------------------


def verdict_html(ok):
    return '<td class="%s">%s</td>' % ("win" if ok else "lose", "within" if ok else "differs")


def table_html(res, ref, key, rid):
    c = compare(res, ref, key, rid)
    sk = sk_label(ref)
    out = ['<div class="fitwrap"><table class="fit wide"><thead><tr><th>test metric</th><th class="num">ml</th><th class="num">%s</th>'
           '<th class="num">absolute difference</th><th class="num">tolerance</th><th>verdict</th></tr></thead><tbody>' % e(sk)]
    for r in c["rows"]:
        tol = r["tol"] if isinstance(r["tol"], str) else fnum(r["tol"])
        name = r["name"] if "at" not in r else r["name"] + " (at %s)" % r["at"]
        out.append('<tr><th>%s</th><td class="num">%s</td><td class="num">%s</td><td class="num">%s</td><td class="num">%s</td>%s</tr>'
                   % (e(name), e(r["ml"]), e(r["sk"]), fnum(r["diff"]), e(tol), verdict_html(r["ok"])))
    out.append("</tbody></table></div>")
    out.append('<p class="fn">%s was fitted as <code>%s</code>.</p>' % (e(sk), e(c["ref"]["sklearn_params"])))
    note = verdicts(c, key, rid)
    if note:
        out.append('<p class="note"><b>A difference, reported.</b> %s</p>' % e(note))
    return "".join(out)


def summary_md(res, ref):
    """The README's comparison table: the headline metrics of every reference run, and the cause of any difference."""
    out = ["| example, run | metric | ml | %s | absolute difference | tolerance of docs/examples.md |" % sk_label(ref),
           "|---|---|---:|---:|---:|---|"]
    notes = []
    for key, ex in EXAMPLES.items():
        head = ["rmse", "r2"] if ex["task"] == "regression" else ["accuracy", "roc_auc", "log_loss"]
        for r in ex["runs"]:
            if not r.get("ref"):
                continue
            c = compare(res, ref, key, r["id"])
            note = verdicts(c, key, r["id"])
            if note:
                notes.append("* **%s, %s.** %s" % (ex["short"], r["id"], note))
            for row in c["rows"]:
                if row["name"] in head or "at" in row:
                    name = row["name"] if "at" not in row else "largest coefficient difference"
                    out.append("| %s, %s | %s | %s | %s | %s | %s |" % (ex["short"], r["id"], name, row["ml"], row["sk"], fnum(row["diff"]),
                                                                    "within" if row["ok"] else "**differs**"))
    return "\n".join(out + ([""] + notes if notes else []))


def pred_html(res, ref, key, rid):
    ex = EXAMPLES[key]
    c = compare(res, ref, key, rid)
    sk = sk_label(ref)
    cls = ex["task"] == "classification"
    head = "<th>test row</th>" + ('<th class="num">ml label</th><th class="num">%s label</th>' % e(sk) if cls else "")
    head += '<th class="num">ml %s</th><th class="num">%s %s</th><th class="num">absolute difference</th><th>verdict</th>' % (
        "probability" if cls else "prediction", e(sk), "probability" if cls else "prediction")
    out = ['<div class="fitwrap"><table class="fit wide"><thead><tr>%s</tr></thead><tbody>' % head]
    for p in c["pred"]:
        lab = '<td class="num">%s</td><td class="num">%s</td>' % (e(p["mlabel"]), e(p["slabel"])) if cls else ""
        out.append('<tr><th>%d</th>%s<td class="num">%s</td><td class="num">%s</td><td class="num">%s</td>%s</tr>'
                   % (p["i"], lab, e(p["ml"]), e(p["sk"]), fnum(p["diff"]), verdict_html(p["ok"])))
    out.append("</tbody></table></div>")
    return "".join(out)


def coef_html(res, ref, key, rid):
    c = compare(res, ref, key, rid)
    sk = sk_label(ref)
    newton = "ndiff" in c["coef"][0]
    head = '<th>coefficient</th><th class="num">ml</th><th class="num">%s</th><th class="num">absolute difference</th><th>verdict</th>' % e(sk)
    if newton:
        head += '<th class="num">%s, newton-cholesky</th><th class="num">absolute difference</th><th>verdict</th>' % e(sk)
    out = ['<div class="fitwrap"><table class="fit wide"><thead><tr>%s</tr></thead><tbody>' % head]
    for r in c["coef"]:
        row = '<tr><th>%s</th><td class="num">%s</td><td class="num">%s</td><td class="num">%s</td>%s' % (
            e(r["name"]), e(r["ml"]), e(r["sk"]), fnum(r["diff"]), verdict_html(r["ok"]))
        if newton:
            row += '<td class="num">%s</td><td class="num">%s</td>%s' % (e(r["nsk"]), fnum(r["ndiff"]), verdict_html(r["nok"]))
        out.append(row + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def conv_html(res, ref, key):
    """The iteration check: each run that names `same_as` against the run it repeats with more iterations."""
    ex = EXAMPLES[key]
    rows = []
    for r in ex["runs"]:
        if "same_as" not in r:
            continue
        a, ai = fit_values(res, key, r["same_as"])
        b, bi = fit_values(res, key, r["id"])
        base = next(x for x in ex["runs"] if x["id"] == r["same_as"])
        d = max(abs(float(x[1]) - float(y[1])) for x, y in zip(a, b))
        d = max(d, abs(float(ai) - float(bi)))
        big = max(abs(float(x[1])) for x in a)
        rows.append('<tr><th><code>--l2 %s</code></th><td class="num">%d</td><td class="num">%d</td><td class="num">%s</td><td class="num">%s</td></tr>'
                    % (e(r["l2"]), base["iterations"], r["iterations"], fnum(d), "%.4g" % big))
    if not rows:
        return ""
    return ('<div class="fitwrap"><table class="fit"><thead><tr><th>penalty</th><th class="num">iterations</th><th class="num">compared with</th>'
            '<th class="num">largest change in a coefficient or the intercept</th><th class="num">largest coefficient</th></tr></thead><tbody>%s</tbody></table></div>'
            % "".join(rows))


def facts_html(ref, key):
    ex = EXAMPLES[key]
    f = ref["examples"][key]
    s = "%d training rows and %d test rows, %d features (<code>%s</code>, <code>%s</code>)" % (
        f["files"]["train"]["rows"], f["files"]["test"]["rows"], f["features"],
        e(Path(f["files"]["train"]["path"]).name), e(Path(f["files"]["test"]["path"]).name))
    if "positive_share_train" in f:
        s += "; target <code>%s</code>, the positive class (label 1) is %.1f%% of the training rows" % (e(ex["target"]), 100 * f["positive_share_train"])
    else:
        s += "; target <code>%s</code>" % e(ex["target"])
    return "<p class=\"fn\">%s.</p>" % s


def runit_lines():
    out = ["# in a scratch directory, with the built ml on PATH", "mkdir -p work && cd work && cp ../docs/examples/data/*.csv ."]
    for key, ex in EXAMPLES.items():
        for r in ex["runs"]:
            if not (r.get("ref") or r.get("shown_only")) or r.get("refusal"):
                continue
            out.append("# %s, %s" % (ex["short"], r["id"]))
            out.append(shlex.join(common.fit_argv(ex, r)))
            out.append(shlex.join(common.eval_argv(ex, r)))
            if r.get("ref"):
                a = common.proba_argv(ex, r)
                if ex["task"] == "regression":
                    a.remove("--proba")
                out.append(shlex.join(a))
    return out


def meta_html(ref):
    g = ref["generated_by"]
    return ("<p class=\"fn\">%s, numpy %s, scipy %s, Python %s. Split: <code>%s</code>. Command that wrote the files: <code>%s</code>. "
            "Read from <code>docs/examples/reference.json</code>; <b>not re-run by the check.</b></p>"
            % (e(sk_label(ref)), e(g["numpy"]), e(g["scipy"]), e(g["python"]), e(g["split"]), e(g["command"])))


def full_md(res, ref):
    """Section 6 of docs/examples.md: every test metric of every reference run, then the iteration checks."""
    out = ["| example, run | metric | ml | %s | absolute difference | verdict |" % sk_label(ref), "|---|---|---:|---:|---:|---|"]
    notes = []
    for key, ex in EXAMPLES.items():
        for r in ex["runs"]:
            if not r.get("ref"):
                continue
            c = compare(res, ref, key, r["id"])
            note = verdicts(c, key, r["id"])
            if note:
                notes.append("* %s, %s: %s" % (ex["short"], r["id"], note))
            for row in c["rows"]:
                name = row["name"] if "at" not in row else "largest coefficient or intercept difference (at %s)" % row["at"]
                out.append("| %s, %s | %s | %s | %s | %s | %s |" % (ex["short"], r["id"], name, row["ml"], row["sk"], fnum(row["diff"]),
                                                                "within" if row["ok"] else "**differs**"))
            worst = max(p["diff"] for p in c["pred"])
            out.append("| %s, %s | first %d test rows, largest prediction or probability difference | | | %s | %s |"
                       % (ex["short"], r["id"], common.PROBA_ROWS, fnum(worst), "within" if all(p["ok"] for p in c["pred"]) else "**differs**"))
    out.append("")
    out += notes
    out.append("")
    out.append("Iteration checks (each run against the run it repeats with more iterations; the largest change in a coefficient or the intercept):")
    out.append("")
    for key, ex in EXAMPLES.items():
        for r in ex["runs"]:
            if "same_as" not in r:
                continue
            a, ai = fit_values(res, key, r["same_as"])
            b, bi = fit_values(res, key, r["id"])
            base = next(x for x in ex["runs"] if x["id"] == r["same_as"])
            d = max([abs(float(x[1]) - float(y[1])) for x, y in zip(a, b)] + [abs(float(ai) - float(bi))])
            out.append("* %s, `--l2 %s`, %d iterations against %d: %s" % (ex["short"], r["l2"], base["iterations"], r["iterations"], fnum(d)))
    return "\n".join(out)

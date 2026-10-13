#!/usr/bin/env python3
"""Regenerate the parts of the README and of the project pages that are program
output, from the built binary, so that they cannot drift from it.

    python3 scripts/site.py            # rewrite the generated regions
    python3 scripts/site.py --check    # change nothing; exit 1 if a file differs
    python3 scripts/site.py --bin build/ml

The mechanism is cancho-table's (scripts/site.py there). A generated region is a
pair of comment lines in a file,

    <!-- gen:NAME -->
    ...whatever the script writes...
    <!-- /gen:NAME -->

and nothing outside a pair is touched: **the prose of README.md, docs/index.html,
docs/evidence.html and docs/benchmarks.html is hand-written, and so is every
number in them that is not program output** (the gate and benchmark figures are
copied from the document that cites them, next to their conditions). What is
generated:

    t-NAME        README.md, index.html   the example for one task: the command and
                                          what the built binary prints for it
    flow          README.md, index.html   one complete flow: fit, predict, eval, a
                                          refusal with its repair, the repair run
    qs            README.md, index.html   the quick start's commands and output
    hero          index.html              the first example
    agentio       README.md, index.html   excerpts of `ml introspect`
    skill         index.html              the first lines of `ml skill`
    authority     README.md, index.html   the row of manifests/ml.authority.json
                                          (checked against `ml introspect`, and
                                          for net, ffi and clock)
    exit-codes    README.md               the exit codes `ml introspect` lists
    limits, rules, nrules   index.html    the limits and rules `ml introspect` lists

Two whole files are also written (and checked) from the list of pages below, so
that the pages and the sitemap cannot disagree: docs/sitemap.xml and
docs/robots.txt.

The commands run in a temporary directory on fixed small files, in the order of
DEMOS, as `ml` found on PATH (so a repair that names the tool says `ml`). Each
command runs once and its output is reused wherever it is shown; a model file
written by one is read by the next. Standard output and standard error are one
stream, as in a terminal. Needs the standard library and the built binary
(`cancho install && cancho build --bin ml`, see the README).
"""

import argparse
import difflib
import html
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    # four houses' worth of numbers; area is a float column (a decimal point, so it is read as a decimal or a double)
    "houses.csv": "id,rooms,area,price\n1,2,48.5,180000\n2,3,71.25,240000\n3,3,64,215000\n4,4,98.5,335000\n5,1,31,120000\n",
    # new rows to predict for: no price column is needed
    "new.csv": "id,rooms,area\n6,2,52.5\n7,3,80\n",
    # the same with a column of text
    "listings.csv": "id,rooms,area,city\n6,2,52.5,Lyon\n7,3,80,Porto\n",
    # a two-class target
    "shop.csv": "id,visits,spend,bought\n1,3,12.50,1\n2,1,0.00,0\n3,7,40.25,1\n4,2,5.00,0\n5,5,22.75,1\n6,4,18.00,1\n",
    # a float target (an exponent makes a cell a float)
    "sensor.csv": "id,x,reading\n1,1.5,1e1\n2,2.5,2.75\n3,0.1,3e-1\n4,4,5.5\n",
}

# What is shown is what is typed; the argv run is shlex.split of it. The order is the order of running.
DEMOS = [
    ("fit", "ml fit --target price --output houses.ml houses.csv"),
    ("predict", "ml predict --model-file houses.ml new.csv"),
    ("eval", "ml eval --model-file houses.ml houses.csv"),
    ("bad", "ml predict --model-file houses.ml listings.csv"),
    # "fixed" is not typed here: it is the argv of the repair of "bad", run as the tool offered it
    ("class_fit", "ml fit --task classification --target bought --output shop.ml shop.csv"),
    ("class_page", "ml predict --model-file shop.ml shop.csv --limit 2 --from 1"),
    ("class_eval", "ml eval --model-file shop.ml shop.csv"),
    ("float_fit", "ml fit --target reading --output sensor.ml sensor.csv"),
    ("float_eval", "ml eval --model-file sensor.ml sensor.csv"),
    ("exists", "ml fit --target price --output houses.ml houses.csv"),
    ("dry", "ml fit --dry-run --overwrite --target price --output houses.ml houses.csv"),
    ("damaged", "ml eval --model-file damaged.ml houses.csv"),
    ("other_model", "ml fit --model linear --target price --output linear.ml houses.csv"),
    ("typo", "ml fit --target prize --output typo.ml houses.csv"),
]
# demos shown as the members of the JSON line that matter, since the line is long
SUMMARY = {
    "bad": ["rule", "hint", "repair"],
    "exists": ["rule", "hint", "detail"],
    "damaged": ["rule", "hint"],
    "other_model": ["rule", "detail"],
    "typo": ["rule", "hint", "detail"],
}
# task name -> the demos shown for it
TASKS = {
    "fit": ["fit", "predict", "eval"],
    "class": ["class_fit", "class_page", "class_eval"],
    "float": ["float_fit", "float_eval"],
    "dry": ["exists", "dry"],
    "damaged": ["damaged"],
    "honest": ["other_model", "typo"],
}
QS = ["fit", "predict", "eval", "bad"]
FORBIDDEN = {"ffi", "net_out", "net_in", "clock"}


class Drift(Exception):
    pass


def e(s):
    return html.escape(s, quote=False)


# --- running things ----------------------------------------------------------


def run_words(words, binary, cwd):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"}
    r = subprocess.run(words, executable=str(binary), cwd=cwd, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return r.stdout.decode("utf-8"), r.returncode


def make_context(binary, tmp):
    """Write the files, run every demo once, in order. The results are kept."""
    for name, text in FILES.items():
        (Path(tmp) / name).write_text(text)
    results = {}

    def run(words):
        return run_words(words, binary, tmp)

    for key, typed in DEMOS:
        if key == "damaged":
            # a copy of houses.ml with one byte changed, inside the part the SHA-256 trailer covers
            b = bytearray((Path(tmp) / "houses.ml").read_bytes())
            b[-40] ^= 1
            (Path(tmp) / "damaged.ml").write_bytes(bytes(b))
        out, status = run(shlex.split(typed))
        results[key] = (typed, out, status)
        if key == "bad":
            doc = json.loads(out)
            repair = doc["error"]["repair"]
            if repair is None or repair["kind"] != "retry":
                raise Drift("the demo `bad` no longer carries a retry repair")
            argv = repair["argv"]
            out2, status2 = run(argv)
            if status2 != 0:
                raise Drift("the repair of the demo `bad` does not run")
            results["fixed"] = (shlex.join(argv), out2, status2)
    r = subprocess.run(["ml", "introspect"], executable=str(binary), cwd=tmp, stdout=subprocess.PIPE, check=True,
                       env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"})
    return {"results": results, "introspect": json.loads(r.stdout), "run": run}


def item(ctx, key):
    """(typed command, what is printed, status, note). A refusal is shown as the
    members of its JSON line that matter."""
    typed, out, status = ctx["results"][key]
    if key in SUMMARY:
        doc = json.loads(out)
        if doc["ok"] or status == 0:
            raise Drift("the demo %s no longer refuses" % key)
        err = doc["error"]
        out = json.dumps({k: err[k] for k in SUMMARY[key]}, ensure_ascii=False)
        return typed, out, status, "the %s of the JSON line it prints" % ", ".join(SUMMARY[key])
    if status != 0 and key not in SUMMARY:
        if json.loads(out).get("dry_run"):
            return typed, out, status, "a dry run: nothing was written"
        raise Drift("the demo %s fails with status %d" % (key, status))
    return typed, out, status, None


def flow_items(ctx):
    """Fit, predict, eval, be refused with a repair, run the repair."""
    return [("# Fit the training mean of price. area has a decimal point: it is a float column, read exactly", None, 0, None),
            item(ctx, "fit"),
            ("# Predict for two new rows, which have no price", None, 0, None),
            item(ctx, "predict"),
            ("# Score the model on the file it was fitted on", None, 0, None),
            item(ctx, "eval"),
            ("# New rows with a column of text are refused", None, 0, None),
            item(ctx, "bad"),
            ("# The tool offered the corrected command. Run it:", None, 0, None),
            item(ctx, "fixed")]


def render(items, html_mode):
    out = []
    for display, text, status, note in items:
        if text is None:
            out.append(('<span class="c">%s</span>' % e(display)) if html_mode else display)
            continue
        if html_mode:
            out.append('<span class="c">$</span> ' + e(display))
            out.append(e(text.rstrip("\n")))
            if status != 0:
                out.append('<span class="r"># exit status %d%s</span>' % (status, "; " + e(note) if note else ""))
        else:
            out.append("$ " + display)
            out.append(text.rstrip("\n"))
            if status != 0:
                out.append("# exit status %d%s" % (status, "; " + note if note else ""))
    return "\n".join(out)


def md_block(items):
    return "```console\n" + render(items, False) + "\n```"


def html_pre(items):
    return '<pre class="code" tabindex="0">' + render(items, True) + "</pre>"


def task_items(ctx, task):
    return [item(ctx, k) for k in TASKS[task]]


def hero(ctx):
    return ('<div class="term" role="img" aria-label="One command on a small file and what it prints">'
            '<div class="bar"><i></i><i></i><i></i></div>\n<pre>' + render([item(ctx, "eval")], True) + "</pre></div>")


def qs_items(ctx):
    return [item(ctx, k) for k in QS]


def authority_row(ctx):
    manifest = json.loads((ROOT / "manifests" / "ml.authority.json").read_text())
    if manifest != ctx["introspect"]["authority"]:
        raise Drift("manifests/ml.authority.json is not what `ml introspect` says: run scripts/manifest.py")
    names = []
    for lab in manifest["labels"]:
        names.append(lab["name"] + ('("%s")' % lab["argument"] if lab["argument"] is not None else ""))
    effects = {lab["name"] for lab in manifest["labels"]}
    if not manifest["bounded"] or manifest["foreign_symbols"] or effects & FORBIDDEN:
        raise Drift("the authority has a forbidden label, a foreign symbol, or is unbounded")
    return manifest


def authority_md(ctx):
    m = authority_row(ctx)
    names = ", ".join(l["name"] + ('("%s")' % l["argument"] if l["argument"] is not None else "") for l in m["labels"])
    return ("```console\n# manifests/ml.authority.json\nml    %s\n"
            "# bounded: %s; foreign symbols: none; net_out, net_in, ffi, clock: absent\n```" % (names, str(m["bounded"]).lower()))


def authority(ctx):
    m = authority_row(ctx)
    names = ", ".join(l["name"] + ('("%s")' % l["argument"] if l["argument"] is not None else "") for l in m["labels"])
    return ('<pre class="code" tabindex="0"><span class="c"># manifests/ml.authority.json</span>\n'
            'ml    %s\n<span class="c"># bounded: %s; foreign symbols: none; net_out, net_in, ffi, clock: absent</span></pre>'
            % (e(names), str(m["bounded"]).lower()))


def limits(ctx):
    rows = "".join("<tr><th><code>--%s</code></th><td class=\"num\">%s</td><td class=\"num\">%s</td></tr>"
                   % (e(l["name"]), format(l["default"], ","), format(l["ceiling"], ","))
                   for l in ctx["introspect"]["limits"])
    return ('<div class="fitwrap"><table class="fit"><thead><tr><th>limit</th><th>default</th><th>ceiling</th></tr></thead><tbody>'
            + rows + "</tbody></table></div>")


def rules_html(ctx):
    rs = ctx["introspect"]["rules"]

    def short(s):
        return s.split(" Hint:")[0]
    rows = "".join("<tr><th><code>%s</code></th><td class=\"num\">%d</td><td>%s</td><td>%s</td></tr>"
                   % (e(r["rule"]), r["exit"], e(r["repairable"]), e(short(r["summary"]))) for r in rs)
    return ('<div class="fitwrap"><table class="fit"><thead><tr><th>rule</th><th>exit</th><th>a repair?</th><th>what it refuses</th></tr></thead><tbody>%s</tbody></table></div>'
            % rows)


def nrules(ctx):
    return str(len(ctx["introspect"]["rules"]))


def cut(s, n):
    return s if len(s) <= n else s[: n - 1] + "…"


SAMPLE_RULES = ["args.unknown-flag", "ml.input.non-numeric", "ml.schema.mismatch", "ml.model.checksum", "conflict.exists", "limit.line-too-long"]
SAMPLE_FLAGS = ["--root", "--target", "--output", "--model-file", "--drop", "--dry-run"]


def agentio_lines(ctx):
    d = ctx["introspect"]
    flags = {f["name"]: f for f in d["flags"]}
    rules = {r["rule"]: r for r in d["rules"]}
    lim = {l["name"]: l for l in d["limits"]}
    try:
        L = ["$ ml introspect          # one JSON document; excerpts, long lines cut", "",
             "tool        %s %s   (compiler %s)" % (d["tool"], d["version"], d["compiler"][:8]),
             "flags       %d, for example" % len(d["flags"])]
        for n in SAMPLE_FLAGS:
            f = flags[n]
            L.append("  %-13s %-16s %s" % (n, f["kind"], cut(f["help"], 52)))
        L.append("limits      %d, for example" % len(d["limits"]))
        for n in ("max-rows", "max-columns", "max-line-bytes", "max-bytes"):
            L.append("  --%-16s default %-12s ceiling %s" % (n, format(lim[n]["default"], ","), format(lim[n]["ceiling"], ",")))
        L.append("rules       %d, for example" % len(d["rules"]))
        for n in SAMPLE_RULES:
            r = rules[n]
            L.append("  %-24s exit %d   a repair: %s" % (n, r["exit"], r["repairable"]))
        L += ["reads environment: %s   reads clock: %s"
              % (str(d["reads_environment"]).lower(), str(d["reads_clock"]).lower()),
              "authority   " + ", ".join(l["name"] + ('("%s")' % l["argument"] if l["argument"] is not None else "") for l in d["authority"]["labels"]),
              "            bounded: %s   foreign symbols: %d" % (str(d["authority"]["bounded"]).lower(), len(d["authority"]["foreign_symbols"]))]
    except KeyError as exc:
        raise Drift("`ml introspect` no longer lists %s" % exc)
    return "\n".join(L)


def agentio(ctx):
    return '<pre class="code" tabindex="0">' + e(agentio_lines(ctx)) + "</pre>"


def agentio_md(ctx):
    return "```console\n" + agentio_lines(ctx) + "\n```"


def skill(ctx):
    out, status = ctx["run"](["ml", "skill"])
    if status != 0:
        raise Drift("`ml skill` failed")
    lines = out.splitlines()
    start = lines.index("## Output")
    body = lines[start:start + 6]
    return ('<pre class="code" tabindex="0">' + e("$ ml skill          # a guide in Markdown; the front matter and one section, long lines cut\n"
            + "\n".join(cut(l, 100) for l in lines[:4]) + "\n...\n" + "\n".join(cut(l, 100) for l in body)) + "</pre>")


def exit_codes(ctx):
    rows = "".join("| %d | `%s` | %s |\n" % (c["code"], c["name"], c["meaning"]) for c in ctx["introspect"]["exit_codes"])
    return "| exit | `code` | meaning |\n|---:|---|---|\n" + rows.rstrip("\n")


REGIONS = {
    "README.md": dict({"t-" + t: (lambda c, t=t: md_block(task_items(c, t))) for t in TASKS},
                      flow=lambda c: md_block(flow_items(c)), qs=lambda c: md_block(qs_items(c)),
                      agentio=agentio_md, authority=authority_md, **{"exit-codes": exit_codes}),
    "docs/index.html": dict({"t-" + t: (lambda c, t=t: html_pre(task_items(c, t))) for t in TASKS},
                            hero=hero, flow=lambda c: html_pre(flow_items(c)), qs=lambda c: html_pre(qs_items(c)),
                            agentio=agentio, skill=skill, authority=authority, limits=limits,
                            rules=rules_html, nrules=nrules),
}


SITE_BASE = "https://alpibrusl.github.io/cancho-ml/"
# the pages that are meant to be found
PAGES = ["", "evidence.html", "benchmarks.html"]


def sitemap(_ctx=None):
    urls = "".join("  <url><loc>%s%s</loc></url>\n" % (SITE_BASE, p) for p in PAGES)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + urls + "</urlset>\n")


def robots(_ctx=None):
    return "User-agent: *\nAllow: /\n\nSitemap: %ssitemap.xml\n" % SITE_BASE


WHOLE_FILES = {"docs/sitemap.xml": sitemap, "docs/robots.txt": robots}


def splice(text, name, body):
    start = "<!-- gen:%s -->\n" % name
    end = "<!-- /gen:%s -->" % name
    i = text.find(start)
    j = text.find(end, i + 1)
    if i < 0 or j < 0:
        raise Drift("no <!-- gen:%s --> region in the file" % name)
    return text[:i] + start + body + "\n" + text[j:]


def regions_in(text):
    """Every region name that appears in a file, to catch one nobody generates."""
    names = []
    i = 0
    while True:
        i = text.find("<!-- gen:", i)
        if i < 0:
            return names
        j = text.find(" -->", i)
        names.append(text[i + len("<!-- gen:"):j])
        i = j


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", default=str(ROOT / "build" / "ml"))
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 if a file differs")
    args = ap.parse_args()
    binary = Path(args.bin).resolve()
    if not binary.exists():
        sys.exit("site.py: %s does not exist: run `cancho install && cancho build --bin ml` first" % binary)
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_context(binary, tmp)
        for rel, regions in REGIONS.items():
            path = ROOT / rel
            old = path.read_text()
            for name in regions_in(old):
                if name not in regions:
                    raise Drift("%s has a region gen:%s that nothing generates" % (rel, name))
            new = old
            for name, fn in regions.items():
                new = splice(new, name, fn(ctx))
            if new == old:
                continue
            if args.check:
                bad += 1
                sys.stderr.write("site.py: %s differs from what the binary produces; run `python3 scripts/site.py`\n" % rel)
                for k, line in enumerate(difflib.unified_diff(old.splitlines(), new.splitlines(), "committed", "generated", lineterm="", n=0)):
                    if k >= 30:
                        sys.stderr.write("...\n")
                        break
                    sys.stderr.write(line[:200] + "\n")
            else:
                path.write_text(new)
                print("wrote", rel)
    for rel, fn in WHOLE_FILES.items():
        path = ROOT / rel
        new = fn()
        old = path.read_text() if path.exists() else None
        if new == old:
            continue
        if args.check:
            bad += 1
            sys.stderr.write("site.py: %s differs from what the page list produces; run `python3 scripts/site.py`\n" % rel)
        else:
            path.write_text(new)
            print("wrote", rel)
    if args.check and not bad:
        print("site.py: generated regions are current")
    return 1 if bad else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Drift as exc:
        sys.exit("site.py: %s" % exc)

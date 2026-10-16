#!/usr/bin/env python3
"""Mutants of the landing page and of the site checks: each is applied to a COPY of docs/, scripts/, manifests/ and README.md in a temporary directory (never to
the tree), the gates are run there, and the mutant is killed if a gate that must catch it exits non-zero. A control (the unmutated copy) must pass every gate first.

    python3 scripts/mutants_site.py [--bin build/ml] [NAME ...]     # the mutants whose name contains one of the words (default: all)
    python3 scripts/mutants_site.py --check                         # only check that every pattern applies once (no gate run)

The gates: `landing` (scripts/check_landing.py: the figures against the sources, the header of the five pages, the shape of the three replays, the experiments and
the server in words, the runners of the workflow, the build commands, the limits and the basics), `demo` (check_landing.py --demo: the first two replays against a fresh
run of scripts/demo.sh, which needs ml-serve and curl), `site` (scripts/site.py --check: the generated regions and the figures against the built binary and the documents), `links`
(scripts/check_links.py) and `seo` (scripts/seo_check.py). The colour pairs have their own mutants (scripts/check_contrast.py --mutants). Needs the built binary
(`cancho install && cancho build --bin ml`). Exit 1 if the control fails, a pattern does not apply, or a mutant survives.
"""
import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable
BINARY = ROOT / "build" / "ml"

# (name, file, old, new, the gates that must catch it)
IDX = "docs/index.html"


def _fig(name):
    """The figure the landing page shows now (a pattern that names a figure is written from it, so a change of the real number does not break the mutant)."""
    m = re.search(r'<span data-n="%s">([^<]+)</span>' % name, (ROOT / IDX).read_text())
    return m.group(1)


def _more(value):
    return "{:,}".format(int(value.replace(",", "")) + 1)


def _less(value):
    return "{:,}".format(int(value.replace(",", "")) - 1)


TESTS, KILLED, RULES = _fig("tests"), _fig("mut_killed"), _fig("rules")
MUTANTS = [
    ("figure-tests-typed", IDX, '<span data-n="tests">%s</span>' % TESTS, '<span data-n="tests">%s</span>' % _more(TESTS), {"landing", "site"}),
    ("figure-mutants-typed", IDX, '<span data-n="mut_killed">%s</span>' % KILLED, '<span data-n="mut_killed">%s</span>' % _more(KILLED), {"landing", "site"}),
    ("figure-rules-typed", IDX, '<b><span data-n="rules">%s</span></b>' % RULES, '<b><span data-n="rules">%s</span></b>' % _more(RULES), {"landing", "site"}),
    ("figure-red-gap-typed", IDX, '<span data-n="red_gap">8.1</span>', '<span data-n="red_gap">3.1</span>', {"landing", "site"}),
    ("figure-name-unknown", IDX, '<span data-n="red_tol">3</span>', '<span data-n="red_tolerance">3</span>', {"landing", "site"}),
    ("source-tests-edited", "docs/status.md", "%s conformance tests pass, none skipped" % TESTS, "%s conformance tests pass, none skipped" % _more(TESTS), {"landing", "site"}),
    ("source-golden-line-removed", "docs/bench/xplat_fp.txt",
     "linear-diabetes-t1\teval\t0\t6553db4528e032fd15cc7c61fae26f6a332fad84f73d9ebc25ecef417225a08d\n", "", {"landing", "site"}),
    ("source-rule-count-stale", "docs/status.md", "| refusals | %s rule tags in `introspect`" % RULES, "| refusals | %s rule tags in `introspect`" % _less(RULES), {"landing"}),
    ("evidence-tests-differ", "docs/evidence.html", "%s tests pass" % TESTS, "%s tests pass" % _less(TESTS), {"landing"}),
    ("replay-number-edited", IDX, "<span>accuracy   0.9333 -&gt; 0.8666</span>", "<span>accuracy   0.9334 -&gt; 0.8666</span>", {"demo", "site"}),
    ("replay-hash-edited", IDX, "verified: True  model 3d69715e3577", "verified: True  model 3d69715e3578", {"landing", "demo", "site"}),
    ("replay-whole-hash-shown", IDX, '<span>run d659c3217b4c\u2026</span>', '<span>run d659c3217b4c0f1e2d3c4b5a69788796a5b4c3d2e1f00112233445566778899a</span>', {"landing", "demo", "site"}),
    ("replay-port-leaks", IDX, "localhost:8080/predict -d '{\"sepal_length_cm\":\"abc\"}'", "localhost:63440/predict -d '{\"sepal_length_cm\":\"abc\"}'", {"landing", "demo", "site"}),
    ("replay-temp-path-leaks", IDX, "<span>run 519af537b8b7\u2026</span>", "<span>run 519af537b8b7\u2026 /var/folders/x/cancho-ml-demo.AbC</span>", {"landing", "demo", "site"}),
    ("replay-serve-not-loopback", IDX, "listening on 127.0.0.1:8080", "listening on 0.0.0.0:8080", {"landing", "demo", "site"}),
    ("replay-serve-flag-shown", IDX, "$ ml-serve --model-file iris.ml --port 8080</span>", "$ ml-serve --model-file iris.ml --port 8080 --no-token</span>", {"landing", "demo", "site"}),
    ("replay-train-marker-wrong", IDX, "iris_train.csv  on: train", "iris_train.csv  on: other", {"landing", "demo", "site"}),
    ("replay-compare-removed", IDX, '<span class="a">$ ml runs compare runs A C</span>\n', "", {"landing", "demo", "site"}),
    ("replay-tab-removed", IDX, '<pre id="screen-serve"', '<pre id="screen-sv"', {"landing", "demo", "site"}),
    ("replay-prediction-edited", IDX, '"rows":[[1,0.678121207688661]]', '"rows":[[1,0.678121207688662]]', {"demo", "site"}),
    ("replay-repair-not-the-argv", IDX, "strategy=mean'</span>\n<span class=\"g\">", "strategy=median'</span>\n<span class=\"g\">", {"landing", "site"}),
    ("replay-refusal-no-exit-status", IDX, '<span class="b"># exit status 8</span>\n', "", {"landing", "site"}),
    ("replay-control-not-a-button", IDX, '<button id="replay" type="button" hidden>&#8635; replay</button>', '<div id="replay">&#8635; replay</div>', {"landing"}),
    # the header is generated, so a header mutant edits the list of links in site.py (or one page's region, which site.py --check refuses)
    ("nav-link-missing-file", "scripts/site.py", '("Guide", "guide.html", "", "guide.html")', '("Guide", "guide2.html", "", "guide.html")', {"links", "landing"}),
    ("nav-link-dropped-on-a-page", "docs/evidence.html", '<a href="benchmarks.html" class="opt">Benchmarks</a>', "", {"landing", "site"}),
    ("nav-dead-fragment", "scripts/site.py", '("How it works", "index.html#how", "opt", "index.html")', '("How it works", "index.html#hows", "opt", "index.html")', {"links"}),
    ("nav-page-not-current", "docs/guide.html", 'aria-current="page">Guide</a>', ">Guide</a>", {"landing", "site"}),
    ("budget-over", IDX, "<!--\n  The landing page.", "<!-- " + "x" * 6000 + " -->\n<!--\n  The landing page.", {"landing"}),
    ("two-h1", IDX, '<div class="cta">', '<h1>Another</h1><div class="cta">', {"landing", "seo"}),
    ("skip-link-removed", IDX, '<a class="skip" href="#main">Skip to content</a>\n', "", {"landing"}),
    ("svg-title-removed", IDX, '<title id="flow-t">From a CSV to a model file, and the refusal loop</title>', "", {"landing"}),
    ("svg-in-words-removed", IDX, "In words:", "Briefly:", {"landing"}),
    ("external-script", IDX, "</head>", '<script src="https://example.com/x.js"></script>\n</head>', {"landing", "seo"}),
    ("external-stylesheet", IDX, '<link rel="stylesheet" href="assets/site.css">', '<link rel="stylesheet" href="https://example.com/site.css">', {"landing"}),
    ("red-gate-removed", IDX, "One pre-registered gate is red:", "One gate differs:", {"landing"}),
    ("no-speed-statement-removed", IDX, "No speed is claimed:", "Speed is fine:", {"landing"}),
    ("build-commands-drift", "README.md", "export PATH=$PWD/../cancho/target/release:$PWD/build:$PATH", "export PATH=$PWD/../cancho/target/debug:$PWD/build:$PATH", {"landing"}),
    ("sitemap-page-missing", "docs/sitemap.xml", "  <url><loc>https://alpibrusl.github.io/cancho-ml/guide.html</loc></url>\n", "", {"site"}),
    ("seo-title-too-long", IDX, "<title>cancho-ml: machine learning on a CSV that refuses clearly</title>",
     "<title>cancho-ml: machine learning on a CSV that refuses clearly and repeats</title>", {"landing", "seo"}),
    ("link-to-missing-file", IDX, '<a href="evidence.html">Evidence</a></h3>', '<a href="evidenz.html">Evidence</a></h3>', {"links"}),
    # the server, the experiments and the platforms, in words
    ("serve-text-every-address", IDX, "It listens on 127.0.0.1 unless you pass <code>--bind all</code>, which needs a token file (<code>--token-file</code>).",
     "It listens on every address of the host and so needs <code>--token-file</code> or <code>--no-token</code>.", {"landing"}),
    ("serve-text-no-token-added", IDX, "which needs a token file (<code>--token-file</code>)", "which needs a token file (<code>--token-file</code> or <code>--no-token</code>)", {"landing"}),
    ("serve-text-token-file-removed", IDX, " (<code>--token-file</code>)", "", {"landing"}),
    ("serve-comment-says-every-address", "scripts/site.py", '"# the same file over HTTP: it listens on 127.0.0.1 unless --bind all is given"',
     '"# the same file over HTTP: it listens on every address of the host"', {"landing"}),
    ("experiments-card-removed", IDX, "<h3>Experiments you can replay</h3>", "<h3>Experiments</h3>", {"landing"}),
    ("experiments-verify-claim-removed", IDX, "<code>ml runs verify</code> re-fits and reproduces the model&rsquo;s hash.", "<code>ml runs verify</code> checks it.", {"landing"}),
    ("experiments-compare-claim-removed", IDX, "shows what differs and the metrics side by side;", "compares them;", {"landing"}),
    ("experiments-on-marker-removed", IDX, "(<code>on</code>) and counts the cells outside its range", "and counts the cells outside the range", {"landing"}),
    ("experiments-track-removed", IDX, "(<code>--track</code>)", "", {"landing"}),
    # the line about explaining a model (docs/interpret.md)
    ("explain-dump-claim-removed", IDX, "and <code>ml dump</code> prints a tree.", "", {"landing"}),
    ("explain-importance-claim-removed", IDX, "<code>ml importance</code> says", "It says", {"landing"}),
    ("explain-link-removed", IDX, 'href="guide.html#explain"', 'href="guide.html"', {"landing"}),
    ("explain-section-of-the-guide-renamed", "docs/guide.html", 'id="explain"', 'id="explained"', {"landing"}),
    ("explain-readme-section-renamed", "README.md", "## Explain a model", "## Explanations", {"landing"}),
    ("platforms-runner-dropped-from-text", IDX, "<code>ubuntu-24.04-arm</code>, ", "", {"landing"}),
    ("platforms-figure-typed", IDX, '<span data-n="runners">4</span>', '<span data-n="runners">3</span>', {"landing", "site"}),
    ("platforms-matrix-grows", ".github/workflows/ci.yml", "          - {os: macos-26, label: darwin-arm64}\n",
     "          - {os: macos-26, label: darwin-arm64}\n          - {os: windows-2025, label: windows-x86_64}\n", {"landing", "site"}),
    ("platforms-pin-moves", "cancho.toml", 'cancho = "0207556b4029a3ee73196cdfdb48e8efa7e9c519"', 'cancho = "1234567b4029a3ee73196cdfdb48e8efa7e9c519"', {"landing"}),
    ("lang-removed", IDX, '<html lang="en">', "<html>", {"landing", "seo"}),
    # the experiments board (docs/board.html): its card on the landing page, its link in the header of every page, its place in the sitemap and in the SEO checks
    ("board-card-removed", IDX, '<h3><a href="board.html">Experiments board</a></h3>', "<h3>Experiments board</h3>", {"landing"}),
    ("board-card-link-missing-file", IDX, '<h3><a href="board.html">Experiments board</a></h3>', '<h3><a href="boards.html">Experiments board</a></h3>', {"links", "landing"}),
    ("board-card-privacy-claim-removed", IDX, "Nothing is sent anywhere.", "Nothing leaves.", {"landing"}),
    ("board-card-what-it-shows-removed", IDX, "a table, a comparison, a chart and a list of what does not add up", "a table", {"landing"}),
    ("board-nav-link-dropped-on-a-page", "docs/evidence.html", '<a href="board.html" class="opt">Board</a>', "", {"landing", "site"}),
    ("board-nav-not-current", "docs/board.html", 'aria-current="page">Board</a>', ">Board</a>", {"landing", "site"}),
    ("board-nav-link-missing-file", "scripts/site.py", '("Board", "board.html", "opt", "board.html")', '("Board", "boards.html", "opt", "board.html")', {"links", "landing"}),
    ("board-sitemap-entry-missing", "docs/sitemap.xml", "  <url><loc>https://alpibrusl.github.io/cancho-ml/board.html</loc></url>\n", "", {"site"}),
    ("board-title-too-long", "docs/board.html", "<title>cancho-ml experiments board: read run records locally</title>", "<title>cancho-ml experiments board: read run records locally, compare them</title>", {"landing", "seo"}),
    ("board-canonical-wrong", "docs/board.html", '<link rel="canonical" href="https://alpibrusl.github.io/cancho-ml/board.html">', '<link rel="canonical" href="https://alpibrusl.github.io/cancho-ml/boards.html">', {"seo"}),
    ("board-skip-link-removed", "docs/board.html", '<a class="skip" href="#main">Skip to content</a>\n', "", {"landing"}),
    ("board-external-script", "docs/board.html", "</head>", '<script src="https://example.com/x.js"></script>\n</head>', {"landing", "seo"}),
    ("board-sample-not-generated", "docs/board.html", "<!-- gen:sample -->\n", "<!-- gen:sample -->\n<!-- edited -->\n", {"site"}),
]


def copy_tree(dst):
    for d in ("docs", "scripts", "manifests", ".github"):
        shutil.copytree(ROOT / d, dst / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "mutants_*.log"))
    shutil.copy(ROOT / "README.md", dst / "README.md")
    shutil.copy(ROOT / "cancho.toml", dst / "cancho.toml")
    # the other files are only linked to (check_links.py asks that they exist): link them, so that nothing is copied that no mutant edits
    for entry in ROOT.iterdir():
        if entry.name not in ("docs", "scripts", "manifests", ".github", "README.md", "cancho.toml", "build", ".git") and not (dst / entry.name).exists():
            (dst / entry.name).symlink_to(entry)


def gates(tree, binary):
    cmds = {"landing": [PY, "scripts/check_landing.py"], "site": [PY, "scripts/site.py", "--check", "--bin", str(binary)],
            "links": [PY, "scripts/check_links.py"], "seo": [PY, "scripts/seo_check.py"],
            "demo": [PY, "scripts/check_landing.py", "--demo", "--bin", str(binary)]}
    failed = set()
    for name, cmd in cmds.items():
        r = subprocess.run(cmd, cwd=tree, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode != 0:
            failed.add(name)
    return failed


def apply(tree, mutant):
    name, file, old, new, _ = mutant
    p = tree / file
    t = p.read_text()
    if t.count(old) != 1:
        raise SystemExit("mutant %s: the pattern occurs %d times in %s, want 1" % (name, t.count(old), file))
    if old == new:
        raise SystemExit("mutant %s: replacement equals pattern" % name)
    p.write_text(t.replace(old, new, 1))
    if file == "scripts/site.py":
        # a mutant of the generator: the pages are regenerated with it, so that site.py --check agrees and the other gates are the ones that must see the fault
        subprocess.run([PY, "scripts/site.py", "--bin", str(BINARY)], cwd=tree, check=True, stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(ROOT / "build" / "ml"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    binary = pathlib.Path(args.bin).resolve()
    global BINARY
    BINARY = binary
    chosen = [m for m in MUTANTS if not args.names or any(w in m[0] for w in args.names)]
    if args.check:
        with tempfile.TemporaryDirectory() as td:
            tree = pathlib.Path(td)
            copy_tree(tree)
            for m in chosen:
                t = tree / m[1]
                if t.read_text().count(m[2]) != 1 or m[2] == m[3]:
                    print("mutant %s does not apply" % m[0])
                    return 1
        print("%d mutants, every pattern applies once" % len(chosen))
        return 0
    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td) / "control"
        base.mkdir()
        copy_tree(base)
        failed = gates(base, binary)
        print("%-34s %s" % ("control", "passes every gate" if not failed else "FAILS " + ",".join(sorted(failed))))
        if failed:
            return 1
        bad = 0
        for i, m in enumerate(chosen):
            tree = pathlib.Path(td) / ("m%d" % i)
            tree.mkdir()
            copy_tree(tree)
            apply(tree, m)
            failed = gates(tree, binary)
            expected = m[4]
            ok = expected <= failed
            print("%-34s %-8s caught by %s%s" % (m[0], "killed" if ok else "SURVIVED", ",".join(sorted(failed)) or "nothing", "" if ok else "; expected " + ",".join(sorted(expected))))
            shutil.rmtree(tree)
            bad += 0 if ok else 1
        print("%d mutants, %d killed, %d not" % (len(chosen), len(chosen) - bad, bad))
        return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

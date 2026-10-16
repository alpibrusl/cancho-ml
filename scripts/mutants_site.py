#!/usr/bin/env python3
"""Mutants of the landing page and of the site checks: each is applied to a COPY of docs/, scripts/, manifests/ and README.md in a temporary directory (never to
the tree), the gates are run there, and the mutant is killed if a gate that must catch it exits non-zero. A control (the unmutated copy) must pass every gate first.

    python3 scripts/mutants_site.py [--bin build/ml] [NAME ...]     # the mutants whose name contains one of the words (default: all)
    python3 scripts/mutants_site.py --check                         # only check that every pattern applies once (no gate run)

The gates: `landing` (scripts/check_landing.py: the figures against the sources, the header of the five pages, the shape of the replay, the build commands, the limits
and the basics), `site` (scripts/site.py --check: the generated regions and the figures against the built binary and the documents), `links`
(scripts/check_links.py) and `seo` (scripts/seo_check.py). The colour pairs have their own mutants (scripts/check_contrast.py --mutants). Needs the built binary
(`cancho install && cancho build --bin ml`). Exit 1 if the control fails, a pattern does not apply, or a mutant survives.
"""
import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable
BINARY = ROOT / "build" / "ml"

# (name, file, old, new, the gates that must catch it)
IDX = "docs/index.html"
MUTANTS = [
    ("figure-tests-typed", IDX, '<span data-n="tests">439</span>', '<span data-n="tests">440</span>', {"landing", "site"}),
    ("figure-mutants-typed", IDX, '<span data-n="mut_killed">1,144</span>', '<span data-n="mut_killed">1,145</span>', {"landing", "site"}),
    ("figure-rules-typed", IDX, '<b><span data-n="rules">79</span></b>', '<b><span data-n="rules">80</span></b>', {"landing", "site"}),
    ("figure-red-gap-typed", IDX, '<span data-n="red_gap">8.1</span>', '<span data-n="red_gap">3.1</span>', {"landing", "site"}),
    ("figure-name-unknown", IDX, '<span data-n="red_tol">3</span>', '<span data-n="red_tolerance">3</span>', {"landing", "site"}),
    ("source-tests-edited", "docs/status.md", "439 conformance tests pass, none skipped", "440 conformance tests pass, none skipped", {"landing", "site"}),
    ("source-golden-line-removed", "docs/bench/xplat_fp.txt",
     "linear-diabetes-t1\teval\t0\tb7b0b80d1e2a9722d14a493dbf65840db560bb462ab6147d4be7d670be17bc7b\n", "", {"landing", "site"}),
    ("source-rule-count-stale", "docs/status.md", "| refusals | 99 rule tags in `introspect`", "| refusals | 77 rule tags in `introspect`", {"landing"}),
    ("evidence-tests-differ", "docs/evidence.html", "439 tests pass", "438 tests pass", {"landing"}),
    ("replay-number-edited", IDX, '"accuracy":"0.9333333333333333"', '"accuracy":"0.9333"', {"site"}),
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
    ("lang-removed", IDX, '<html lang="en">', "<html>", {"landing", "seo"}),
]


def copy_tree(dst):
    for d in ("docs", "scripts", "manifests"):
        shutil.copytree(ROOT / d, dst / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "mutants_*.log"))
    shutil.copy(ROOT / "README.md", dst / "README.md")
    shutil.copy(ROOT / "cancho.toml", dst / "cancho.toml")
    # the other files are only linked to (check_links.py asks that they exist): link them, so that nothing is copied that no mutant edits
    for entry in ROOT.iterdir():
        if entry.name not in ("docs", "scripts", "manifests", "README.md", "cancho.toml", "build", ".git") and not (dst / entry.name).exists():
            (dst / entry.name).symlink_to(entry)


def gates(tree, binary):
    cmds = {"landing": [PY, "scripts/check_landing.py"], "site": [PY, "scripts/site.py", "--check", "--bin", str(binary)],
            "links": [PY, "scripts/check_links.py"], "seo": [PY, "scripts/seo_check.py"]}
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

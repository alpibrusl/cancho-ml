#!/usr/bin/env python3
"""Checks of the landing page (docs/index.html) and of the pages' shared header, that need no binary (standard library only).

    python3 scripts/check_landing.py

What fails it:

  * the figures: every <b|span data-n="NAME"> of the page equals the value computed here from the sources (docs/status.md, docs/bench/xplat_fp.txt, and the
    rule count of the guide's generated table), independently of scripts/site.py; and the counts the page repeats (tests, mutants) are the ones docs/evidence.html
    states, and the rule count is the one docs/status.md states. A figure typed by hand and wrong, or a source edited and the page not, is a failure;
  * the navigation: the header of the five pages is the same list of links, every link is a file of docs/ (or an https address) and an in-page link names an id,
    and the link of the page itself is marked aria-current (the landing page's "How it works" is an in-page link and is not);
  * the replayed runs: three <pre class="screen"> panels (fit and compare, serve and curl, a refusal and its repair) hold their lines as spans; in the last one
    the refusal names a rule and an exit status, the repair is a command, and the command that follows it is the repair's own argv; the first two carry the
    experiment commands and the loopback line of ml-serve, show no whole hash, no port but 8080 and no temporary path; with --demo they are compared with a fresh
    run of scripts/demo.sh (the built binaries), line by line;
  * the experiments and the server, in words: the card "Experiments you can replay" and the commands it names (`--track`, `ml runs compare`, `ml runs verify`,
    `on`, the cells outside the range) are on the page; the text about ml-serve says it listens on 127.0.0.1, that `--bind all` needs `--token-file`, and never
    says it listens on every address by default or mentions `--no-token`; every runner of the matrix of .github/workflows/ci.yml is named on the page, and the
    figure "runners" is their number;
  * the build commands of "Try it" are the lines of the README's quick start (the same sh block);
  * the honest limits are on the page: the red gate with its two numbers, "No speed is claimed", the platforms that are not covered;
  * the basics: one <h1>, `lang`, a skip link to <main id="main">, a <title> and a description inside the SEO limits, the diagram has a <title> and a <desc> and the
    page a sentence that says it in words, the replay control is a <button>, a <noscript>-free fallback (the full log is in the HTML), no external request
    (no <script src>, no stylesheet or image from another host, no @import), an inline script of at most 3 KB;
  * the size budget: docs/index.html is at most 28 KB (28,672 bytes) and, with docs/assets/site.css, at most 45 KB.

    python3 scripts/check_landing.py --budget     # print the sizes
    python3 scripts/check_landing.py --demo       # also run scripts/demo.sh (needs build/ml, build/ml-serve, curl) and compare the replay with it
    python3 scripts/check_landing.py --demo --bin DIR   # the directory that holds ml and ml-serve
"""
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PAGES = ["index.html", "guide.html", "examples.html", "evidence.html", "benchmarks.html"]
BUDGET_INDEX = 28 * 1024
BUDGET_WITH_CSS = 45 * 1024
BUDGET_SCRIPT = 3 * 1024


def src(root=ROOT):
    return Path(root)


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.h1 = 0
        self.lang = None
        self.ids = set()
        self.links = []          # (href, in_header, attrs)
        self.header_links = []
        self.in_header = False
        self.srcs = []
        self.buttons = []
        self.svg_title = self.svg_desc = 0
        self.skip = None
        self.main = False
        self.scripts = []
        self._script = None
        self.data_n = []
        self._stack = []
        self._n = None
        self.title = ""
        self._in_title = False
        self.desc = None
        self.inline_styles = ""
        self._style = False
        self.noscript = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "html":
            self.lang = a.get("lang")
        if tag == "h1":
            self.h1 += 1
        if tag == "header":
            self.in_header = True
        if tag == "main" and a.get("id") == "main":
            self.main = True
        if tag == "a" and a.get("href") is not None:
            self.links.append((a["href"], a))
            if self.in_header:
                self.header_links.append((a["href"], a))
            if "skip" in (a.get("class") or "").split():
                self.skip = a["href"]
        if tag in ("script", "img", "iframe", "source", "audio", "video") and a.get("src"):
            self.srcs.append((tag, a["src"]))
        if tag == "link" and a.get("rel") in ("stylesheet", "icon", "preload", "prefetch"):
            self.srcs.append(("link", a.get("href", "")))
        if tag == "button":
            self.buttons.append(a)
        if tag == "title" and not self._stack_has("svg"):
            self._in_title = True
        if tag == "title" and self._stack_has("svg"):
            self.svg_title += 1
        if tag == "desc":
            self.svg_desc += 1
        if tag == "meta" and a.get("name") == "description":
            self.desc = a.get("content", "")
        if tag == "noscript":
            self.noscript += 1
        if tag == "style":
            self._style = True
        if tag == "script" and not a.get("src") and a.get("type") != "application/ld+json":
            self._script = ""
        if "data-n" in a and tag in ("b", "span"):
            self._n = [a["data-n"], ""]
        if tag not in ("meta", "link", "br", "img", "input", "path", "circle", "rect", "defs", "marker"):
            self._stack.append(tag)

    def _stack_has(self, t):
        return t in self._stack

    def handle_endtag(self, tag):
        if tag == "header":
            self.in_header = False
        if tag == "title":
            self._in_title = False
        if tag == "style":
            self._style = False
        if tag == "script" and self._script is not None:
            self.scripts.append(self._script)
            self._script = None
        if self._n is not None and tag in ("b", "span"):
            self.data_n.append(tuple(self._n))
            self._n = None
        if tag in self._stack:
            while self._stack and self._stack.pop() != tag:
                pass

    def handle_data(self, d):
        if self._in_title:
            self.title += d
        if self._script is not None:
            self._script += d
        if self._style:
            self.inline_styles += d
        if self._n is not None:
            self._n[1] += d


def parse(path):
    p = Page()
    p.feed(path.read_text())
    return p


# --- the figures ---------------------------------------------------------------------------------------------------


def _int(s):
    return int(s.replace(",", ""))


def figures(root):
    """name -> value, from the sources, computed here."""
    status = (root / "docs" / "status.md").read_text()
    fp = (root / "docs" / "bench" / "xplat_fp.txt").read_text().splitlines()
    out = {"fp_lines": str(len(fp)), "fp_fits": str(sum(1 for l in fp if l.split("\t")[1:2] == ["fit-exit"])), "platforms": "3"}
    row = next(l for l in status.splitlines() if l.startswith("| other platforms"))
    named = [p for p in ("macOS arm64", "Linux x86-64", "Linux aarch64") if p in row]
    out["platforms"] = str(len(named))
    out["runners"] = str(len(ci_runners(root)))
    out["tests"] = re.search(r"([\d,]+) conformance tests pass, none skipped", status).group(1)
    m = re.search(r"([\d,]+) patterns across [\w-]+ scripts: ([\d,]+) killed, (\d+) survive", status)
    out["mut_total"], out["mut_killed"], out["mut_survive"] = m.groups()
    m = re.search(r"RMSE [\d.]+ against scikit-learn's [\d.]+, ([\d.]+)% apart; the tolerance is (\d+)%", status)
    out["red_gap"], out["red_tol"] = m.groups()
    guide = (root / "docs" / "guide.html").read_text()
    out["models"] = model_words(root)[0]
    out["rules"] = re.search(r"<!-- gen:nrules -->\n(\d+)\n<!-- /gen:nrules -->", guide).group(1)
    return out


WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
DISPLAY = {"kmeans": "k-means"}


def model_words(root):
    """(the number of wired models in words, their names as the page spells them), from the manifests of lib/models"""
    names = [p.parent.name for p in sorted((root / "lib" / "models").glob("*/model.toml")) if re.search(r"^wired *= *true", p.read_text(), re.M)]
    return WORDS[len(names)], [DISPLAY.get(n, n) for n in names]


def ci_runners(root):
    """The runner images of the matrix of the workflow, in order."""
    return re.findall(r"- \{os: ([\w.-]+),", (root / ".github" / "workflows" / "ci.yml").read_text())


def check_figures(root, p, errs):
    want = figures(root)
    seen = {}
    for name, val in p.data_n:
        if name not in want:
            errs.append("index.html: data-n=%r is a figure nobody computes" % name)
            continue
        seen[name] = val
        if val != want[name]:
            errs.append("index.html: %s is %r, the sources say %r" % (name, val, want[name]))
    for name in ("fp_lines", "tests", "mut_killed", "mut_total", "rules", "red_gap", "red_tol", "platforms", "runners", "models"):
        if name not in seen:
            errs.append("index.html: the figure %s is missing" % name)
    status = (root / "docs" / "status.md").read_text()
    m = re.search(r"(\d+) rule tags in `introspect`", status)
    if not m or m.group(1) != want["rules"]:
        errs.append("docs/status.md says %s rule tags in `introspect`, the guide's table has %s" % (m and m.group(1), want["rules"]))
    ev = (root / "docs" / "evidence.html").read_text()
    if "%s tests pass" % want["tests"] not in ev:
        errs.append("docs/evidence.html does not say '%s tests pass'" % want["tests"])
    for k in ("mut_total", "mut_killed"):
        if ">%s<" % want[k] not in ev:
            errs.append("docs/evidence.html has no cell %s (%s)" % (want[k], k))


# --- the navigation ------------------------------------------------------------------------------------------------


def check_nav(root, errs):
    sets = {}
    for name in PAGES:
        p = parse(root / "docs" / name)
        nav = [(h, a) for h, a in p.header_links if not (a.get("class") or "") == "brand"]
        sets[name] = [(re.sub(r"^index\.html(?=#)", "", h), ) for h, _ in nav]
        for h, a in p.header_links:
            if h.startswith("http"):
                continue
            path, _, frag = h.partition("#")
            target = root / "docs" / (path or name) if path not in ("", "./") else (root / "docs" / (name if not path and frag else "index.html"))
            if not target.is_file():
                errs.append("%s: the header link %r names no file" % (name, h))
                continue
            if frag and frag not in parse(target).ids:
                errs.append("%s: the header link %r names no id" % (name, h))
        current = [h for h, a in nav if a.get("aria-current") == "page"]
        if name == "index.html":
            if current:
                errs.append("index.html: the in-page link must not be marked current: %r" % current)
        elif current != [name]:
            errs.append("%s: the link of the page itself is %r, want [%r]" % (name, current, name))
    # the same links in the same order on every page, up to the form of the link to the landing page
    ref = [x[0].lstrip("#").replace("index.html#", "") for x in sets["index.html"]]
    for name, s in sets.items():
        got = [x[0].lstrip("#").replace("index.html#", "") for x in s]
        if got != ref:
            errs.append("%s: the header links are %r, the landing page's are %r" % (name, got, ref))
    if len(ref) < 5 or len(ref) > 6:
        errs.append("the header has %d links, want 5 or 6" % len(ref))


# --- the replay ----------------------------------------------------------------------------------------------------


class Replay(HTMLParser):
    """The panels of the replay: id of each <pre class="screen"> -> its lines [(class, text)]."""

    def __init__(self):
        super().__init__()
        self.panel = None
        self.panels = {}
        self._cls = None
        self._buf = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "pre" and "screen" in (a.get("class") or "").split():
            self.panel = a.get("id")
            self.panels[self.panel] = []
        elif tag == "span" and self.panel:
            self._cls, self._buf = a.get("class", ""), ""

    def handle_endtag(self, tag):
        if tag == "pre":
            self.panel = None
        if tag == "span" and self.panel and self._buf is not None:
            self.panels[self.panel].append((self._cls, self._buf))
            self._buf = None

    def handle_data(self, d):
        if self._buf is not None:
            self._buf += d


PANELS = ["screen-fit", "screen-serve", "screen-fix"]


def check_replay(root, text, errs):
    r = Replay()
    r.feed(text)
    if list(r.panels) != PANELS:
        errs.append("index.html: the replay panels are %r, want %r (the full logs are in the HTML, for a browser without JavaScript)" % (list(r.panels), PANELS))
        return
    for tab in ("tab-fit", "tab-serve", "tab-fix"):
        if 'id="%s"' % tab not in text:
            errs.append("index.html: the replay has no tab %s" % tab)
    fit, serve, fix = (r.panels[k] for k in PANELS)
    every = fit + serve + fix
    if len(fit) < 15 or len(serve) < 8 or len(fix) < 5:
        errs.append("index.html: the replay panels hold %d, %d and %d lines, want at least 15, 8 and 5" % (len(fit), len(serve), len(fix)))
    for c, t in every:
        if re.search(r"\b[0-9a-f]{64}\b", t):
            errs.append("index.html: the replay shows a whole 64-digit hash (it shows 12 digits): %s" % t[:60])
        if re.search(r"/var/folders|/tmp/|cancho-ml-demo|/Users/", t):
            errs.append("index.html: the replay shows a temporary or local path: %s" % t[:60])
        for port in re.findall(r"(?:localhost|127\.0\.0\.1):(\d+)|--port (\d+)", t):
            if (port[0] or port[1]) != "8080":
                errs.append("index.html: the replay shows a port other than 8080: %s" % t[:60])
    # fit and compare: two tracked fits, compare, verify, eval train against other
    cmds = [t for c, t in fit if c == "a"]
    for want in ("$ ml fit ", "--track runs", "$ ml runs compare ", "$ ml runs verify ", "$ ml eval "):
        if not any(want in t for t in cmds):
            errs.append("index.html: the first replay has no command with %r" % want)
    ftext = "\n".join(t for c, t in fit)
    for want in ("verified: True", "on: train", "on: other", "differs: plan"):
        if want not in ftext:
            errs.append("index.html: the first replay does not show %r" % want)
    ids = re.findall(r"^run ([0-9a-f]{12})…$", ftext, re.M)
    if len(ids) != 2 or ids[0] == ids[1]:
        errs.append("index.html: the first replay shows %r as run ids, want two different ones" % ids)
    # serve and curl: loopback by default, a prediction and a refusal that name the model's hash
    scmds = [t for c, t in serve if c == "a"]
    if len(scmds) != 3 or not scmds[0].startswith("$ ml-serve --model-file iris.ml") or not all(t.startswith("$ curl ") for t in scmds[1:]):
        errs.append("index.html: the second replay is not ml-serve and two curl requests: %r" % scmds)
    elif re.search(r"--bind|--no-token|--token-file", scmds[0]):
        errs.append("index.html: the second replay starts ml-serve with a flag besides the model and the port (it shows the default)")
    stext = "\n".join(t for c, t in serve)
    if "listening on 127.0.0.1:8080" not in stext or "0.0.0.0" in stext:
        errs.append("index.html: the second replay does not show that ml-serve listens on 127.0.0.1")
    if '"rule":"ml.schema.mismatch"' not in stext or '"status":422' not in stext or '"row_count":1' not in stext:
        errs.append("index.html: the second replay shows no prediction and no refusal (ml.schema.mismatch, 422)")
    shas = re.findall(r'"model_sha256":"([0-9a-f]{12})…"', stext)
    verified = re.findall(r"model ([0-9a-f]{12})…", ftext)
    if len(shas) != 2 or len(set(shas) | set(verified)) != 1:
        errs.append("index.html: the model hash of the requests (%r) is not the one the first replay verified (%r)" % (shas, verified))
    # a refusal and its repair
    L = fix
    cmds = [t for c, t in L if c == "a"]
    if not all(t.startswith("$ ml ") for t in cmds) or len(cmds) != 2:
        errs.append("index.html: the third replay's commands are not two `$ ml ...` lines (%d)" % len(cmds))
    ref = [(i, t) for i, (c, t) in enumerate(L) if c == "b" and t.startswith('{"ok":false')]
    if len(ref) != 1:
        errs.append("index.html: the third replay holds %d refusals, want 1" % len(ref))
        return
    i, line = ref[0]
    body = line[:-2] if line.endswith(" …") else line
    try:
        doc = json.loads(body)
    except ValueError as exc:
        errs.append("index.html: the refusal line is not JSON: %s" % exc)
        return
    err = doc["error"]
    if not re.fullmatch(r"[a-z]+(\.[a-z-]+)+", err["rule"]):
        errs.append("index.html: the refusal has no rule tag")
    if not any(t.startswith("# exit status ") and int(t.split()[-1]) != 0 for c, t in L[i + 1:i + 2]):
        errs.append("index.html: the refusal is not followed by its exit status")
    rep = err["repair"]
    if not rep or rep.get("kind") != "retry":
        errs.append("index.html: the refusal carries no retry repair")
        return
    import shlex
    nxt = [t for c, t in L[i + 1:] if c == "a"]
    if not nxt or shlex.split(nxt[0][2:]) != rep["argv"]:
        errs.append("index.html: the command after the refusal is not its repair's argv")
    ok_after = [t for c, t in L[i + 1:] if c == "g"]
    if not ok_after or not ok_after[0].startswith('{"ok":true'):
        errs.append("index.html: the repaired run is not shown as a success")


def load_site(root):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cancho_site", root / "scripts" / "site.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root / "scripts"))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def check_demo(root, text, binary, errs):
    """The first two logs are a fresh run of scripts/demo.sh, cut as the page's caption says (the cuts are scripts/site.py's: demo_lines)."""
    import tempfile
    if not (binary.parent / "ml-serve").exists() or not binary.exists():
        errs.append("check_landing --demo: %s and ml-serve are needed (cancho build --bin ml; cancho build --bin ml-serve)" % binary)
        return
    mod = load_site(root)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            out = mod.run_demo(binary, tmp)
        except Exception as exc:  # noqa: BLE001  (the demo's own failure is the finding)
            errs.append("check_landing --demo: scripts/demo.sh failed: %s" % exc)
            return
    r = Replay()
    r.feed(text)
    for tab, panel in (("fit", "screen-fit"), ("serve", "screen-serve")):
        want = mod.demo_lines(out, tab)
        got = r.panels.get(panel, [])
        if want != got:
            k = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
            errs.append("index.html: the replay %r differs from a fresh run of scripts/demo.sh at line %d: page %r, demo %r (run python3 scripts/site.py)"
                        % (tab, k + 1, got[k] if k < len(got) else None, want[k] if k < len(want) else None))
    # independent of the cuts: the facts of the raw output are on the page
    page = "\n".join(t for p in r.panels.values() for c, t in p)
    for h in sorted(set(re.findall(r"\b[0-9a-f]{64}\b", out))):
        if h[:12] + "…" not in page:
            errs.append("index.html: the hash %s... of the fresh demo is not on the page" % h[:12])
    m = re.search(r"listening on (\S+):\d+", out)
    if not m or m.group(1) != "127.0.0.1":
        errs.append("check_landing --demo: ml-serve printed %r, not a listening line on 127.0.0.1" % (m and m.group(0)))
    for m in re.finditer(r"^(\S+\.csv)  on: (\w+)  rows: (\d+)  out_of_range cells: (\d+)  accuracy: (\S+)$", out, re.M):
        if m.group(0) not in page:
            errs.append("index.html: the eval line %r of the fresh demo is not on the page" % m.group(0))


# --- the rest ------------------------------------------------------------------------------------------------------


def check_basics(root, path, p, text, errs, landing=True):
    n = path.name
    if p.h1 != 1:
        errs.append("%s: %d <h1>, want 1" % (n, p.h1))
    if not p.lang:
        errs.append("%s: no lang" % n)
    if p.skip != "#main" or not p.main:
        errs.append("%s: a skip link to <main id=\"main\"> is needed" % n)
    if "main" not in p.ids:
        errs.append("%s: no id main" % n)
    if not (50 <= len(p.title.strip()) <= 60):
        errs.append("%s: title is %d characters" % (n, len(p.title.strip())))
    if not p.desc or len(p.desc) > 160:
        errs.append("%s: description is %s characters, at most 160" % (n, p.desc and len(p.desc)))
    for tag, s in p.srcs:
        if re.match(r"(https?:)?//", s) or s.startswith("data:") and tag == "script":
            errs.append("%s: an external or data request: <%s> %s" % (n, tag, s))
        if tag == "script":
            errs.append("%s: <script src> is not allowed" % n)
    if re.search(r"@import|url\(\s*['\"]?https?:", p.inline_styles + text):
        errs.append("%s: a CSS import or url() of another host" % n)
    if landing and (p.svg_title < 1 or p.svg_desc < 1):
        errs.append("%s: an SVG without <title> and <desc>" % n)
    for s in p.scripts:
        if len(s.encode()) > BUDGET_SCRIPT:
            errs.append("%s: the inline script is %d bytes, over %d" % (n, len(s.encode()), BUDGET_SCRIPT))


EXPERIMENT_CLAIMS = [("Experiments you can replay", "the card of the experiments"), ("<code>--track</code>", "the flag that writes a run record"),
                     ("ml runs compare", "the comparison of runs"), ("ml runs verify", "the replay of a run"), ("re-fits and reproduces the model", "the claim that verify reproduces the model's hash"),
                     ("shows what differs and the metrics side by side", "the claim of what compare shows"),
                     ("(<code>on</code>)", "the marker that says which file eval scored"), ("outside its range", "the count of cells outside the training range"),
                     ("scripts/demo.sh", "the demo the first two logs are")]
SERVE_REQUIRED = [("127.0.0.1", "that ml-serve listens on 127.0.0.1"), ("--bind all", "the flag that opens it to every address"),
                  ("--token-file", "that a token file goes with --bind all")]
SERVE_FORBIDDEN = [(r"listens on every address", "ml-serve is said to listen on every address"), (r"--no-token", "the flag --no-token is shown"),
                   (r"0\.0\.0\.0", "the address 0.0.0.0 is shown"), (r"needs\s+<code>--token-file</code>\s+or\s+<code>--no-token", "the old rule of the token flags")]


def check_claims(root, text, errs):
    # the models: the page names every model the manifests wire, in a sentence that starts with their number, and says nothing of another count
    word, names = model_words(root)
    m = re.search(r"One verb, <span data-n=\"models\">\w+</span> models: ([^.<]*)\.", text)
    if not m:
        errs.append("index.html: the sentence 'One verb, <N> models: ...' is missing")
    else:
        listed = re.split(r", | and ", m.group(1))
        if sorted(listed) != sorted(names):
            errs.append("index.html: the models listed are %s, the manifests of lib/models wire %s" % (listed, names))
    for stale in set(WORDS) - {word}:
        if re.search(r"\b(%s) models\b" % stale, text, re.I) and stale not in ("two", "three", "four", "five"):
            errs.append("index.html: it says '%s models', the manifests wire %s" % (stale, word))
    for needle, what in EXPERIMENT_CLAIMS:
        if needle not in text:
            errs.append("index.html: %s is missing (looked for %r)" % (what, needle))
    for needle, what in SERVE_REQUIRED:
        if needle not in text:
            errs.append("index.html: the text says nothing of %s (looked for %r)" % (what, needle))
    for pat, what in SERVE_FORBIDDEN:
        if re.search(pat, text):
            errs.append("index.html: %s (matched %r); ml-serve listens on 127.0.0.1 by default" % (what, pat))
    # the platforms: every runner of the matrix is named, with nothing else in its place
    for os_name in ci_runners(root):
        if "<code>%s</code>" % os_name not in text:
            errs.append("index.html: the runner %s of .github/workflows/ci.yml is not named on the page" % os_name)
    pins = re.search(r'^cancho *= *"([0-9a-f]{7})', (root / "cancho.toml").read_text(), re.M)
    if pins and "<code>%s</code>" % pins.group(1) not in text:
        errs.append("index.html: the compiler pin %s of cancho.toml is not named on the page" % pins.group(1))
    for old in re.findall(r"<code>([0-9a-f]{7})</code>", text):
        if pins and old != pins.group(1) and "previous" not in text:
            errs.append("index.html: a compiler revision %s other than the pin is named without saying it is the previous one" % old)


def check_landing_page(root, errs):
    path = root / "docs" / "index.html"
    text = path.read_text()
    p = parse(path)
    check_basics(root, path, p, text, errs)
    check_figures(root, p, errs)
    check_replay(root, text, errs)
    if not any(a.get("id") == "replay" for a in p.buttons) or not all(True for _ in p.buttons):
        errs.append("index.html: the replay control must be a <button id=\"replay\">")
    if "In words:" not in text:
        errs.append("index.html: the diagram has no sentence in words")
    for needle, what in (("No speed is claimed", "the statement that no speed is claimed"), ("is red", "the red gate"),
                         ("Any other platform", "the platforms that are not covered"), ("not a live session", "the replay's caption"),
                         ("no release is published", "what exists of releases")):
        if needle not in text:
            errs.append("index.html: %s is missing (looked for %r)" % (what, needle))
    check_claims(root, text, errs)
    size = len(text.encode())
    css = len((root / "docs" / "assets" / "site.css").read_bytes())
    if size > BUDGET_INDEX:
        errs.append("index.html is %d bytes, over its budget of %d" % (size, BUDGET_INDEX))
    if size + css > BUDGET_WITH_CSS:
        errs.append("index.html and site.css are %d bytes, over %d" % (size + css, BUDGET_WITH_CSS))
    # the build commands are the README's
    m = re.search(r'<pre class="code" tabindex="0" id="build-cmds"><code>(.*?)</code></pre>', text, re.S)
    readme = (root / "README.md").read_text()
    blk = re.search(r"```sh\n(git clone https://github.com/alpibrusl/cancho .*?)\n\ncat >", readme, re.S)
    if not m or not blk:
        errs.append("index.html or README.md: the build commands could not be found")
    else:
        import html as H
        page_lines = [l.split("  #")[0].rstrip() for l in H.unescape(m.group(1)).splitlines()]
        rd_lines = [l for l in blk.group(1).splitlines() if not l.startswith("sh scripts/check_table_pin.sh")]
        rd_lines = [l.split("  #")[0].rstrip() for l in rd_lines]
        if page_lines != rd_lines:
            errs.append("index.html: the build commands are not the README's quick start")


def main(argv):
    root = ROOT
    errs = []
    binary = ROOT / "build" / "ml"
    if "--bin" in argv:
        binary = Path(argv[argv.index("--bin") + 1]).resolve()
    if "--budget" in argv:
        t = (DOCS / "index.html").read_bytes()
        c = (DOCS / "assets" / "site.css").read_bytes()
        print("index.html %d bytes (budget %d), site.css %d bytes, together %d (budget %d)" % (len(t), BUDGET_INDEX, len(c), len(t) + len(c), BUDGET_WITH_CSS))
        return 0
    check_landing_page(root, errs)
    if "--demo" in argv:
        check_demo(root, (DOCS / "index.html").read_text(), binary, errs)
    check_nav(root, errs)
    for name in PAGES[1:]:
        p = parse(root / "docs" / name)
        check_basics(root, root / "docs" / name, p, (root / "docs" / name).read_text(), errs, landing=False)
    for e in errs:
        print("check_landing:", e)
    if not errs:
        t = len((DOCS / "index.html").read_bytes())
        print("check_landing: ok (figures, header of %d pages, replays%s, experiments, server, runners, build commands, limits, basics; index.html %d bytes of %d)"
              % (len(PAGES), " against a fresh run of the demo" if "--demo" in argv else "", t, BUDGET_INDEX))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

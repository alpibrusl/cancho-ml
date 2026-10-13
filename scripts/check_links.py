#!/usr/bin/env python3
"""Check that the links of the README, of docs/status.md and of the project pages
resolve to something that is in this repository.

    python3 scripts/check_links.py          # README.md, docs/status.md, docs/*.html
    python3 scripts/check_links.py --all    # also every other docs/*.md (informational: they are other stages' documents)

What is checked, standard library only:

  * a relative link (href, src, a Markdown link or image) names a file or a
    directory that exists, relative to the file that holds the link;
  * a link to the project page's own address (https://alpibrusl.github.io/cancho-ml/...)
    names a file of docs/;
  * a link to this repository's files (https://github.com/alpibrusl/cancho-ml/blob/main/PATH or
    /tree/main/PATH) names a path that exists in the working tree;
  * a fragment (#name) on an HTML page, or on a link to an HTML page, names an id of that page.

Other addresses are not fetched; they are counted. Exit 1 if any link fails.
"""
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
PAGE = "https://alpibrusl.github.io/cancho-ml/"
REPO = "https://github.com/alpibrusl/cancho-ml/"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        for key in ("href", "src"):
            if a.get(key):
                self.links.append(a[key])


MD_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
MD_FENCE = re.compile(r"```.*?```", re.S)


def md_links(text):
    text = MD_FENCE.sub("", text)
    text = re.sub(r"`[^`\n]*`", "", text)
    return MD_LINK.findall(text)


def ids_of(path, cache={}):
    if path not in cache:
        p = Links()
        p.feed(path.read_text())
        cache[path] = p.ids
    return cache[path]


def check_file(path, problems, counts):
    text = path.read_text()
    if path.suffix == ".html":
        p = Links()
        p.feed(text)
        links = p.links
    else:
        links = md_links(text) + re.findall(r'<img[^>]*\bsrc="([^"]+)"', text)
    for link in links:
        parts = urlsplit(link)
        if parts.scheme in ("mailto", "javascript", "data"):
            continue
        target = None
        if parts.scheme in ("http", "https"):
            if link.startswith(PAGE):
                rel = unquote(parts.path[len(urlsplit(PAGE).path):])
                target = ROOT / "docs" / (rel or "index.html")
                if target.is_dir():
                    target = target / "index.html"
            elif link.startswith(REPO + "blob/main/") or link.startswith(REPO + "tree/main/"):
                rel = unquote(parts.path.split("/main/", 1)[1])
                target = ROOT / rel
            else:
                counts["external"] += 1
                continue
        elif not parts.path:
            target = path   # a fragment on this page
        else:
            target = (path.parent / unquote(parts.path)).resolve()
            if target.is_dir() and (target / "index.html").exists() and parts.path.endswith("/"):
                target = target / "index.html"
            if parts.path in (".", "./"):
                target = ROOT / "docs" / "index.html"
        counts["checked"] += 1
        if not target.exists():
            problems.append("%s: %s -> no such file in the repository" % (path.relative_to(ROOT), link))
            continue
        if parts.fragment and target.suffix == ".html":
            if parts.fragment not in ids_of(target):
                problems.append("%s: %s -> no id %r in %s" % (path.relative_to(ROOT), link, parts.fragment, target.relative_to(ROOT)))


def main():
    files = [ROOT / "README.md", ROOT / "docs" / "status.md"] + sorted((ROOT / "docs").glob("*.html"))
    if "--all" in sys.argv:
        files += [f for f in sorted((ROOT / "docs").glob("*.md")) if f.name != "status.md"]
    problems, counts = [], {"checked": 0, "external": 0}
    for f in files:
        if not f.exists():
            problems.append("%s: the file does not exist" % f.relative_to(ROOT))
            continue
        check_file(f, problems, counts)
    for p in problems:
        sys.stderr.write("check_links: " + p + "\n")
    print("check_links: %d links in %d files checked, %d external addresses not fetched, %d problems"
          % (counts["checked"], len(files), counts["external"], len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

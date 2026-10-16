#!/usr/bin/env python3
"""PS5 (docs/stores.md section 6): the numeric stores are published from the source folders and are copied nowhere.

    python3 scripts/check_stores_pin.py                 # this repository
    CONSUMER=DIR python3 scripts/check_stores_pin.py    # another repository (cancho-stats): its pins of the stores

This repository fails when
* a .cho file other than the module's own file under lib/ declares a published module (a copy), or a published module's file does not declare it;
* cancho.toml has a [dependencies.X] whose git is this repository (cancho-ml does not consume its own stores);
* the modules docs/stores.md section 1 lists are not the modules of scripts/publish_stores.py, or docs/stores_api.txt is missing;
* the publisher's list is not closed under imports (the same test as `publish_stores.py --closure`).

A consumer fails when
* a [dependencies.X] on this repository has a rev that is not 40 hex digits, a rev that differs from another such block, or a path that is not
  `.cancho-vcs/<module>` of a published module;
* a .cho file of the consumer (build/ and .cancho-vcs/ aside) declares a published module (a copy).
ROOT, TOML and DOC override the files read (the mutants).
"""
import os
import pathlib
import re
import sys
import tomllib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import publish_stores  # noqa: E402

ROOT = pathlib.Path(os.environ.get("ROOT") or HERE.parent)
URL = "https://github.com/alpibrusl/cancho-ml"
LOCAL = re.compile(r"^(?:https://github\.com/alpibrusl/cancho-ml(?:\.git)?|ssh://git@github\.com/alpibrusl/cancho-ml(?:\.git)?|file://.*)$")


def declares(text, module):
    return re.search(r"^module\s+" + re.escape(module) + r"\s*;", text, re.M) is not None


def source_files(base, skip):
    for here, dirs, names in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".git")]
        for n in names:
            if n.endswith(".cho"):
                yield pathlib.Path(here) / n


def check_repository():
    problems = []
    modules = publish_stores.MODULES
    own = {publish_stores.file_of(str(ROOT), m): m for m in modules}
    for path in source_files(ROOT, {"build", ".cancho-vcs", "generated", "node_modules"}):
        text = path.read_text()
        for m in modules:
            if declares(text, m) and str(path) != publish_stores.file_of(str(ROOT), m):
                problems.append(f"{path.relative_to(ROOT)} declares module {m}, which is published from {os.path.relpath(publish_stores.file_of(str(ROOT), m), ROOT)}: a copy")
    for f, m in own.items():
        if not os.path.isfile(f) or not declares(open(f).read(), m):
            problems.append(f"{os.path.relpath(f, ROOT)} does not declare module {m}")
    toml = pathlib.Path(os.environ.get("TOML") or ROOT / "cancho.toml")
    deps = tomllib.loads(toml.read_text()).get("dependencies", {})
    for name, d in deps.items():
        if LOCAL.match(d.get("git", "")) and "cancho-ml" in d.get("git", ""):
            problems.append(f"[dependencies.{name}] has git = {d['git']}: cancho-ml does not consume its own stores")
    doc = pathlib.Path(os.environ.get("DOC") or ROOT / "docs" / "stores.md").read_text()
    section = doc.split("## 1. The list, and why", 1)[-1].split("### 1.1 Names", 1)[0]
    # the table rows of section 1 (before "Not published, on purpose"): the first backticked names of each row
    table = section.split("Not published, on purpose", 1)[0]
    listed = set()
    for row in table.splitlines():
        if row.startswith("| `lib/"):
            cells = row.split("|")
            listed |= set(re.findall(r"`([a-z]+\.[a-z0-9_]+)`", cells[2]))
    if listed != set(modules):
        extra, missing = sorted(listed - set(modules)), sorted(set(modules) - listed)
        problems.append(f"docs/stores.md lists modules the publisher does not publish: {extra}; publishes and does not list: {missing}")
    if not (ROOT / publish_stores.API).is_file():
        problems.append(publish_stores.API + " is missing")
    try:
        publish_stores.sources(str(ROOT))
    except SystemExit as e:
        problems.append(str(e))
    return problems


def check_consumer(base):
    base = pathlib.Path(base)
    problems = []
    toml = base / "cancho.toml"
    if not toml.is_file():
        return [f"{toml} does not exist"]
    deps = tomllib.loads(toml.read_text()).get("dependencies", {})
    revs = {}
    for name, d in deps.items():
        git = d.get("git", "")
        if not (re.search(r"/cancho-ml(\.git)?$", git) or git.startswith("file://") and "cancho-ml" in git):
            continue
        rev = d.get("rev", "")
        if not re.fullmatch(r"[0-9a-f]{40}", rev):
            problems.append(f"[dependencies.{name}]: rev is not a full commit hash")
        revs[name] = rev
        m = re.fullmatch(r"\.cancho-vcs/([\w.]+)", d.get("path", ""))
        if not m or m.group(1) not in publish_stores.MODULES:
            problems.append(f"[dependencies.{name}]: path {d.get('path')!r} is not .cancho-vcs/<a published module>")
    if len(set(revs.values())) > 1:
        problems.append("the blocks on cancho-ml pin different revs: " + ", ".join(sorted(set(revs.values()))))
    for path in source_files(base, {"build", ".cancho-vcs", "node_modules"}):
        text = path.read_text()
        for m in publish_stores.MODULES:
            if declares(text, m):
                problems.append(f"{path.relative_to(base)} declares module {m}, which cancho-ml publishes: a copy")
    return problems


def main():
    consumer = os.environ.get("CONSUMER")
    problems = check_consumer(consumer) if consumer else check_repository()
    if problems:
        print("check_stores_pin: " + "\n  ".join(problems), file=sys.stderr)
        sys.exit(1)
    if consumer:
        print("check_stores_pin: ok (consumer pins are well formed, none copied)")
    else:
        print(f"check_stores_pin: ok ({len(publish_stores.MODULES)} published modules, each declared once, none consumed back, the list closed)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""W1 and W2 (docs/serving-http.md sections 1 and 9): cancho-web, cancho-schema and the compiler's http-server are pinned package dependencies of `ml-serve`, and are not copied here.

    python3 scripts/check_web_pin.py             W1: the pins
    python3 scripts/check_web_pin.py --bins      W2: build/ml and build/ml-predict are the bytes a build without the three dependencies makes (needs the network and the compiler)
    CHECK_REMOTE=1 python3 scripts/check_web_pin.py     also asks GitHub whether the three revs have their stores, and whether the `web` store requires the `schema` rev
    TOML=FILE DOC=FILE ROOT=DIR    read other files (the mutants of scripts/mutants_http.py)

W1 fails when
* cancho.toml lacks [dependencies.web], [dependencies.schema] or [dependencies.server], or one of them has another git URL, another path, or a rev that is not 40 hex digits;
* the revs are not the ones docs/serving-http.md names (the lines "cancho-web rev: `...`", "cancho-schema rev: `...`", "cancho http-server rev: `...`"), or the server rev is not
  the compiler pin of [package] cancho;
* another [dependencies.X] points at one of the three repositories (a store that ml-serve does not import is not pinned);
* a .cho file under lib/, tools/ or tests/ declares module web, schema or http.server (a copy), or a file outside tools/ml-serve imports one of them;
* build/deps exists and lacks an installed module of the three.
"""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = pathlib.Path(os.environ.get("ROOT") or pathlib.Path(__file__).resolve().parent.parent)
TOML = pathlib.Path(os.environ.get("TOML") or ROOT / "cancho.toml")
DOC = pathlib.Path(os.environ.get("DOC") or ROOT / "docs" / "serving-http.md")

DEPS = {
    "web": ("https://github.com/alpibrusl/cancho-web", ".cancho-vcs", "cancho-web rev", "web"),
    "schema": ("https://github.com/alpibrusl/cancho-schema", ".cancho-vcs", "cancho-schema rev", "schema"),
    "server": ("https://github.com/alpibrusl/cancho", "packages/http-server/.cancho-vcs", "cancho http-server rev", "http.server"),
}
MODULES = [v[3] for v in DEPS.values()]


def check():
    problems = []
    proj = tomllib.loads(TOML.read_text())
    deps = proj.get("dependencies", {})
    text = DOC.read_text()
    for name, (url, path, label, module) in DEPS.items():
        d = deps.get(name)
        if d is None:
            problems.append("cancho.toml has no [dependencies.%s]" % name)
            continue
        if d.get("git") != url:
            problems.append("[dependencies.%s]: git is %r, not %s" % (name, d.get("git"), url))
        if d.get("path") != path:
            problems.append("[dependencies.%s]: path is %r, not %s" % (name, d.get("path"), path))
        rev = d.get("rev", "")
        if not re.fullmatch(r"[0-9a-f]{40}", rev):
            problems.append("[dependencies.%s]: rev is not a full commit hash" % name)
            continue
        m = re.search(r"%s: `([0-9a-f]{40})`" % re.escape(label), text)
        if not m:
            problems.append("%s has no '%s: `<40 hex digits>`'" % (DOC.name, label))
        elif m.group(1) != rev:
            problems.append("[dependencies.%s]: rev %s is not the documented %s" % (name, rev, m.group(1)))
    pin = proj.get("package", {}).get("cancho", "")
    if "server" in deps and deps["server"].get("rev") != pin:
        problems.append("[dependencies.server]: rev %s is not the compiler pin %s of [package] cancho" % (deps["server"].get("rev"), pin))
    if "schema" in deps and "web" in deps:
        # `web` records its requirement on schema at SCHEMA_REV of cancho-web's ci.yml, and docs/serving-http.md names that rev with the schema pin
        m = re.search(r"SCHEMA_REV[^`\n]*`([0-9a-f]{40})`", text)
        if m and m.group(1) != deps["schema"].get("rev"):
            problems.append("[dependencies.schema]: rev %s is not the SCHEMA_REV %s that the web store requires" % (deps["schema"].get("rev"), m.group(1)))
    urls = {v[0]: k for k, v in DEPS.items()}
    for name, d in deps.items():
        if d.get("git") in urls and name not in DEPS:
            problems.append("[dependencies.%s] is a store of %s that ml-serve does not import (the documented list is web, schema, server)" % (name, d.get("git")))
        if name in DEPS and d.get("git") not in urls:
            pass
    for entry in proj.get("bin", []):
        for s in entry.get("sources", []):
            if s.startswith("..") or "/../" in s:
                problems.append("a sources entry of [[bin]] %s reaches outside this tree: %s" % (entry["name"], s))
    for sub in ("lib", "tools", "tests"):
        for f in sorted((ROOT / sub).rglob("*.cho")) if (ROOT / sub).exists() else []:
            body = f.read_text()
            for m in MODULES:
                if re.search(r"^module %s;" % re.escape(m), body, re.M):
                    problems.append("%s declares module %s, which is a pinned package's" % (f.relative_to(ROOT), m))
            if "tools/ml-serve/" not in str(f.relative_to(ROOT)) + "/" and not str(f.relative_to(ROOT)).startswith("tools/ml-serve/"):
                for m in MODULES:
                    if re.search(r"^import %s;" % re.escape(m), body, re.M):
                        problems.append("%s imports %s: only tools/ml-serve does" % (f.relative_to(ROOT), m))
    bd = ROOT / "build" / "deps"
    if bd.exists() and list(bd.glob("*.cho")):
        have = "\n".join(p.read_text() for p in bd.glob("*.cho"))
        for m in MODULES:
            if not re.search(r"^module %s;" % re.escape(m), have, re.M):
                problems.append("build/deps has no installed module %s (run cancho install)" % m)
    return problems


def remote(proj):
    problems = []
    if shutil.which("gh") is None:
        print("check_web_pin: remote check skipped (no gh)")
        return problems
    deps = proj["dependencies"]
    for name, (url, path, label, module) in DEPS.items():
        repo = url.replace("https://github.com/", "")
        rev = deps[name]["rev"]
        r = subprocess.run(["gh", "api", "repos/%s/contents/%s/manifest.json?ref=%s" % (repo, path, rev), "--jq", ".name"], capture_output=True, text=True)
        if r.returncode != 0:
            if "Could not resolve" in r.stderr or "dial tcp" in r.stderr or "connect" in r.stderr:
                print("check_web_pin: remote check skipped (offline)")
                return problems
            problems.append("%s has no %s/manifest.json at %s on GitHub" % (repo, path, rev))
    r = subprocess.run(["gh", "api", "repos/alpibrusl/cancho-web/contents/.cancho-vcs/requires/0.json?ref=" + deps["web"]["rev"], "--jq", ".content"], capture_output=True, text=True)
    if r.returncode == 0:
        import base64
        req = json.loads(base64.b64decode(r.stdout))
        got = req["lock"]["origin"]["rev"]
        if got != deps["schema"]["rev"]:
            problems.append("the web store at %s requires schema %s, not the pinned %s" % (deps["web"]["rev"], got, deps["schema"]["rev"]))
    return problems


def bins():
    """W2: ml and ml-predict, built from a copy of this tree without the three dependency blocks and without the ml-serve bin, are the bytes of build/ml and build/ml-predict."""
    cancho = os.environ.get("CANCHO") or str(ROOT.parent / "cancho-compiler-a4572ea" / "target" / "release" / "cancho")
    work = pathlib.Path(tempfile.mkdtemp(prefix="webpin-"))
    try:
        for d in ("lib", "tools", "generated", "schemas"):
            shutil.copytree(ROOT / d, work / d)
        text = TOML.read_text()
        for name in DEPS:
            text = re.sub(r"\[dependencies\.%s\]\n(?:[^\[\n].*\n)*" % name, "", text)
        text = re.sub(r"\[\[bin\]\]\nname = \"ml-serve\"\n(?:[^\[\n].*\n)*", "", text)
        (work / "cancho.toml").write_text(text)
        for cmd in ([cancho, "install"], [cancho, "build", "--bin", "ml"], [cancho, "build", "--bin", "ml-predict"]):
            r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
            if r.returncode != 0:
                return ["%s failed: %s" % (" ".join(cmd[1:]), (r.stdout + r.stderr)[-400:])]
        problems = []
        for b in ("ml", "ml-predict"):
            a = (ROOT / "build" / b).read_bytes()
            c = (work / "build" / b).read_bytes()
            if a != c:
                problems.append("build/%s differs from the build without the three dependencies (%d and %d bytes)" % (b, len(a), len(c)))
        return problems
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    problems = check()
    proj = tomllib.loads(TOML.read_text())
    if not problems and os.environ.get("CHECK_REMOTE"):
        problems += remote(proj)
    if "--bins" in sys.argv and not problems:
        problems += bins()
    if problems:
        for p in problems:
            print("check_web_pin: " + p, file=sys.stderr)
        sys.exit(1)
    print("check_web_pin: ok (3 dependencies pinned, none copied%s)" % (", ml and ml-predict unchanged" if "--bins" in sys.argv else ""))


if __name__ == "__main__":
    main()

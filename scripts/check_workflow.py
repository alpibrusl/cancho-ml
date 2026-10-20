#!/usr/bin/env python3
"""Structural checks of .github/workflows/ci.yml (docs/ci.md section 6), for a machine without `actionlint`. Nothing here runs the workflow.

    python3 scripts/check_workflow.py [FILE]       # exit 1 with the reasons when one of the rules below is broken
    python3 scripts/check_workflow.py --mutants    # each rule is shown able to fail: a copy of the workflow with one fault must be refused

The rules:
  W1  the top-level `permissions` is exactly {contents: read}
  W2  a job other than `release` has no `permissions` that grants write (its own block, if any, grants read only)
  W3  `release` grants exactly {contents: write} and runs only for a tag (`if` names refs/tags/)
  W4  every `uses:` is an action pinned to a 40-hex commit SHA (a local `./` action is allowed), with a version comment
  W5  no `pull_request_target`, and no `secrets.` other than `secrets.GITHUB_TOKEN` (the workflow uses `github.token`)
  W6  the compiler revision is read from cancho.toml, and no 40-hex digit string is a literal in the workflow
  W7  every script the workflow runs (`scripts/...`, `ci/...`) is a file of the repository
  W8  triggers: push, pull_request and workflow_dispatch; `full-gate` runs only for workflow_dispatch and schedule
  W9  the matrix of `build-test` has a Linux x86-64 runner, a Linux arm64 runner (`-arm`) and a macOS runner
  W10 every job has `timeout-minutes`
  W11 no `runs-on` and no matrix `os` is a `-latest` label (a moving label changes the platform under a green run)
"""
import pathlib
import re
import sys

try:
    import yaml
except ImportError:
    print("check_workflow: PyYAML is not installed (pip install -r ci/requirements-fast.txt)", file=sys.stderr)
    sys.exit(2)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT = ROOT / ".github" / "workflows" / "ci.yml"
SHA = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def load(text):
    d = yaml.safe_load(text)
    if not isinstance(d, dict):
        raise ValueError("not a mapping")
    if True in d and "on" not in d:  # YAML 1.1 reads the key `on` as true
        d["on"] = d.pop(True)
    return d


def perms(p):
    return p if isinstance(p, dict) else None


def check(text, root=ROOT):
    """-> list of (rule, reason)."""
    out = []
    try:
        d = load(text)
    except Exception as e:  # a YAML error is a failure, not a crash
        return [("W0", "the workflow does not parse: %s" % e)]
    jobs = d.get("jobs") or {}
    if perms(d.get("permissions")) != {"contents": "read"}:
        out.append(("W1", "top-level permissions is %r, not {contents: read}" % (d.get("permissions"),)))
    for name, job in jobs.items():
        p = job.get("permissions")
        if name != "release":
            if p is not None and (not isinstance(p, dict) or any(v not in ("read", "none") for v in p.values())):
                out.append(("W2", "job %s grants %r; only `release` may write" % (name, p)))
        else:
            if perms(p) != {"contents": "write"}:
                out.append(("W3", "release permissions is %r, not {contents: write}" % (p,)))
            if "refs/tags/" not in str(job.get("if", "")):
                out.append(("W3", "release is not limited to a tag (its `if` is %r)" % (job.get("if"),)))
        if "timeout-minutes" not in job:
            out.append(("W10", "job %s has no timeout-minutes" % name))
        for i, st in enumerate(job.get("steps") or []):
            u = st.get("uses")
            if u is not None and not u.startswith("./") and not SHA.match(u):
                out.append(("W4", "job %s step %d: `uses: %s` is not pinned to a 40-hex commit SHA" % (name, i + 1, u)))
    for m in re.finditer(r"^\s*-?\s*uses:\s*(\S+)[^\n]*$", text, re.M):
        line = m.group(0)
        if not m.group(1).startswith("./") and "#" not in line:
            out.append(("W4", "`%s` has no version comment" % line.strip()))
    if "pull_request_target" in text:
        out.append(("W5", "pull_request_target is used"))
    for m in re.finditer(r"secrets\.([A-Za-z0-9_]+)", text):
        if m.group(1) != "GITHUB_TOKEN":
            out.append(("W5", "uses secret %s" % m.group(1)))
    if not re.search(r"sed -n 's/\^cancho \*= \*\"[^']*' cancho\.toml", text):
        out.append(("W6", "no step reads the compiler revision from cancho.toml with sed"))
    for m in re.finditer(r"(?<![0-9a-f@])[0-9a-f]{40}(?![0-9a-f])", text):
        out.append(("W6", "a 40-hex literal that is not a pinned action: %s" % m.group(0)))
    # an action pin is `name@sha`: those are allowed; every other 40-hex string above is a failure (W6)
    out = [(r, m) for r, m in out if not (r == "W6" and "not a pinned action" in m and re.search(r"@" + re.escape(m.split(": ")[1]), text))]
    for s in sorted(set(re.findall(r"(?<![\w./-])((?:scripts|ci)/[\w./-]+)", text))):
        s = s.rstrip(".")
        if not (root / s).exists():
            out.append(("W7", "the workflow names %s, which is not a file of the repository" % s))
    on = d.get("on") or {}
    for t in ("push", "pull_request", "workflow_dispatch"):
        if t not in on:
            out.append(("W8", "the trigger %s is missing" % t))
    fg = jobs.get("full-gate")
    if fg is None:
        out.append(("W8", "no full-gate job"))
    else:
        cond = str(fg.get("if", ""))
        if "workflow_dispatch" not in cond or "schedule" not in cond or "push" in cond or "pull_request" in cond:
            out.append(("W8", "full-gate must run only for workflow_dispatch and schedule (its `if` is %r)" % cond))
    bt = jobs.get("build-test")
    matrix = ((bt or {}).get("strategy") or {}).get("matrix", {}) or {}
    labels = [str(e.get("os", "")) for e in (matrix.get("include") or [])] + [str(o) for o in (matrix.get("os") or [])]
    if not any(l.startswith("ubuntu") and not l.endswith("-arm") for l in labels) or not any(l.startswith("ubuntu") and l.endswith("-arm") for l in labels) \
            or not any(l.startswith("macos") for l in labels):
        out.append(("W9", "build-test's matrix needs a ubuntu x86-64 runner, a ubuntu arm runner and a macos runner (%s)" % labels))
    moving = [l for l in labels if l.endswith("-latest")] + [str(j.get("runs-on")) for j in jobs.values() if str(j.get("runs-on", "")).endswith("-latest")]
    if moving:
        out.append(("W11", "a moving runner label: %s" % ", ".join(sorted(set(moving)))))
    return out


def mutants(text):
    """(label, mutated text, rule that must fire)"""
    def rep(a, b, n=1):
        assert a in text, a
        return text.replace(a, b, n)
    ms = [
        ("top-level permissions removed", rep("permissions:\n  contents: read\n\nconcurrency", "concurrency"), "W1"),
        ("top-level permissions write-all", rep("permissions:\n  contents: read\n\nconcurrency", "permissions: write-all\n\nconcurrency"), "W1"),
        ("top-level contents: write", rep("permissions:\n  contents: read\n\nconcurrency", "permissions:\n  contents: write\n\nconcurrency"), "W1"),
        ("a job other than release writes", rep("    timeout-minutes: 75\n", "    timeout-minutes: 75\n    permissions:\n      contents: write\n"), "W2"),
        ("release also writes issues", rep("      contents: write\n", "      contents: write\n      issues: write\n"), "W3"),
        ("release is not limited to a tag", rep("if: startsWith(github.ref, 'refs/tags/v')", "if: true"), "W3"),
        ("an action pinned by tag", rep("actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1", "actions/checkout@v4"), "W4"),
        ("an action pinned by a short SHA", rep("actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9", "actions/cache@55cc834"), "W4"),
        ("an action pinned by a branch", rep("actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97", "actions/setup-python@main"), "W4"),
        ("an action pin with no version comment", rep("actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9 # v7.0.2", "actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9"), "W4"),
        ("pull_request_target", rep("  pull_request:\n", "  pull_request_target:\n"), "W5"),
        ("a secret", rep("GH_TOKEN: ${{ github.token }}", "GH_TOKEN: ${{ secrets.DEPLOY_KEY }}"), "W5"),
        ("the compiler revision as a literal", rep("rev=$(sed -n 's/^cancho *= *\"\\([0-9a-f]*\\)\".*/\\1/p' cancho.toml)", "rev=a4572ea3b506e4bcec1c4be43deda1043c11ba9f"), "W6"),
        ("a revision literal added to the environment", rep("concurrency:", "env:\n  CANCHO_REV: a4572ea3b506e4bcec1c4be43deda1043c11ba9f\n\nconcurrency:"), "W6"),
        ("a script that does not exist", rep("python3 scripts/check_bins.py\n", "python3 scripts/check_nothing.py\n"), "W7"),
        ("a requirements file that does not exist", rep("ci/requirements-fast.txt", "ci/requirements-none.txt", 2), "W7"),
        ("workflow_dispatch removed", rep("  workflow_dispatch:\n", ""), "W8"),
        ("pull_request removed", rep("  pull_request:\n", ""), "W8"),
        ("full-gate runs on every push", rep("if: github.event_name == 'workflow_dispatch' || github.event_name == 'schedule'", "if: github.event_name == 'workflow_dispatch' || github.event_name == 'schedule' || github.event_name == 'push'"), "W8"),
        ("no macOS in the matrix", rep("          - {os: macos-26, label: darwin-arm64}\n", ""), "W9"),
        ("no Linux arm64 in the matrix", rep("          - {os: ubuntu-24.04-arm, label: linux-aarch64-ubuntu24}\n", ""), "W9"),
        ("no Linux x86-64 in the matrix", rep("          - {os: ubuntu-24.04, label: linux-x86_64-ubuntu24}\n          - {os: ubuntu-26.04, label: linux-x86_64-ubuntu26}\n", ""), "W9"),
        ("a moving label in the matrix", rep("{os: macos-26, label: darwin-arm64}", "{os: macos-latest, label: darwin-arm64}"), "W11"),
        ("a moving label for the release job", rep("    runs-on: ubuntu-24.04\n    timeout-minutes: 15", "    runs-on: ubuntu-latest\n    timeout-minutes: 15"), "W11"),
        ("a job without a timeout", rep("    timeout-minutes: 15\n", ""), "W10"),
        ("not YAML", text + "\n  - : [\n", "W0"),
    ]
    return ms


def main(argv):
    mut = "--mutants" in argv
    args = [a for a in argv if not a.startswith("--")]
    path = pathlib.Path(args[0]) if args else DEFAULT
    if not path.exists():
        print("check_workflow: no %s" % path, file=sys.stderr)
        return 2
    text = path.read_text()
    found = check(text)
    if found:
        print("check_workflow: FAIL")
        for r, m in found:
            print("  %s %s" % (r, m))
        return 1
    print("check_workflow: ok (%s)" % path.relative_to(ROOT) if path.is_relative_to(ROOT) else "check_workflow: ok")
    if not mut:
        return 0
    bad = 0
    ms = mutants(text)
    for label, mt, rule in ms:
        got = {r for r, _ in check(mt)}
        ok = rule in got
        print("%-4s %s (%s)" % ("ok" if ok else "FAIL", label, rule))
        bad += 0 if ok else 1
    print("check_workflow --mutants: %d mutants, %d survived" % (len(ms), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

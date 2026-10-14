#!/usr/bin/env python3
"""Check that every `docs/...` path that README.md and docs/*.md name exists in this repository.

    python3 scripts/check_doc_refs.py            # exit 1 if one does not
    python3 scripts/check_doc_refs.py --list     # the files read, the paths found and the allowlist
    python3 scripts/check_doc_refs.py --mutants  # the check can fail: a copy of a document that names a missing file must be refused

What counts as a reference: the text `docs/PATH` where PATH is made of letters, digits, `_`, `-`, `.` and `/`, in backticks, in a Markdown
link target (`](docs/PATH)`, or `](PATH)` relative to the file that holds it, when it begins with `docs/` after resolving) or bare in
prose. A trailing `.`, `,`, `;`, `:` or `)` is not part of the path. A path with `*`, `<`, `>`, `{`, `$` or `...` in it is a pattern or
a placeholder and is not checked. A path that names a directory is satisfied by the directory. Fenced code blocks are read too: a command
that names a file of this repository has to name one that is there.

ALLOW lists the paths that are not files of this repository, each with the document and the reason. Adding one is a decision to be argued
in review.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (document, path, reason). The paths are files of the repository of the cancho compiler or of cancho-table (their docs/), not of this one.
ALLOW = [
    ("docs/spikes.md", "docs/floating-point.md", "a document of the cancho compiler repository at a4572ea (the gate's source list)"),
    ("docs/spikes.md", "docs/float-math.md", "a document of the cancho compiler repository at a4572ea (the gate's source list)"),
    ("docs/spikes.md", "docs/compile-time.md", "a document of the cancho compiler repository at a4572ea (the gate's source list)"),
    ("docs/spikes.md", "docs/ROADMAP.md", "a document of the cancho compiler repository at a4572ea (the gate's source list)"),
    ("docs/cli.md", "docs/agent-toolbox.md", "cancho's document of the agent contract (compiler repository)"),
    ("docs/design.md", "docs/parallel.md", "cancho-table's document on parallel sums"),
    ("docs/models-tree.md", "docs/parallel.md", "cancho-table's document on parallel sums"),
    ("docs/modules.md", "docs/function-values.md", "a document of the cancho compiler at a4572ea"),
    ("docs/reader.md", "docs/package-system.md", "a document of the cancho compiler repository"),
    ("docs/reader.md", "docs/numbers.md", "cancho-table's document of its public functions"),
    ("docs/reader.md", "docs/readers.md", "cancho-table's document on readers"),
]

PATH = re.compile(r"(?<![A-Za-z0-9_./-])docs/[A-Za-z0-9_./-]*[A-Za-z0-9_/-]")
PLACEHOLDER = re.compile(r"[*<>{}$]|\.\.\.")


def documents():
    return [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))


def references(text):
    """(line number, path) for every docs/ path in the text, placeholders excluded."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        for m in PATH.finditer(line):
            tail = line[m.end():m.end() + 3]
            if PLACEHOLDER.search(m.group(0)) or re.match(r"[*<>{}$]", tail[:1]) or tail.startswith("..."):
                continue
            out.append((n, m.group(0)))
    return out


def check(texts):
    """texts: (document name as 'README.md' or 'docs/x.md', its text). The list of (document, line, path) that does not resolve."""
    allow = {(d, p) for d, p, _ in ALLOW}
    bad = []
    for rel, text in texts:
        for n, p in references(text):
            if (rel, p) in allow:
                continue
            if not (ROOT / p).exists():
                bad.append((rel, n, p))
    return bad


def texts_of_repo():
    return [(d.relative_to(ROOT).as_posix(), d.read_text()) for d in documents()]


def mutants():
    """Each mutant adds one sentence to a copy of a document (in memory); the check must refuse it, and must accept the unchanged text."""
    cases = [
        ("a backticked path", "See `docs/no-such-file.md` for the details."),
        ("a Markdown link", "See [the record](docs/no-such-dir/record.md)."),
        ("a bare path", "The numbers are in docs/spikes-missing.md and nowhere else."),
        ("a name that extends a real file", "See `docs/design.md.bak`."),
        ("a missing file in a real directory", "See `docs/bench/no-such-log.txt`."),
    ]
    texts = texts_of_repo()
    if check(texts):
        print("mutants: the unchanged documents already fail")
        return 1
    failed = 0
    for name, sentence in cases:
        mutated = [(r, t + "\n" + sentence + "\n") if r == "docs/design.md" else (r, t) for r, t in texts]
        ok = any(rel == "docs/design.md" for rel, _, _ in check(mutated))
        print(f"mutant {name}: {'refused' if ok else 'ACCEPTED'}")
        failed += 0 if ok else 1
    return 1 if failed else 0


def main(argv):
    if "--mutants" in argv:
        return mutants()
    docs = documents()
    if "--list" in argv:
        for d in docs:
            print(d.relative_to(ROOT), len(references(d.read_text())))
        for d, p, why in ALLOW:
            print("allow", d, p, "--", why)
        return 0
    bad = check(texts_of_repo())
    total = sum(len(references(d.read_text())) for d in docs)
    for rel, n, p in bad:
        print(f"{rel}:{n}: {p} is not in the repository")
    print(f"check_doc_refs: {len(docs)} documents, {total} references, {len(bad)} unresolved")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

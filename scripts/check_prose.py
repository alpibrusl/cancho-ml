#!/usr/bin/env python3
"""Lint: the README, the project pages and docs/status.md describe the current state, in the present tense, and keep no history.

    python3 scripts/check_prose.py            # fail (exit 1) if one of the seven files holds a changelog or stage-history phrase
    python3 scripts/check_prose.py --list     # the phrases, the allowlist and the files
    python3 scripts/check_prose.py --mutants  # the lint can fail: inject each phrase into a copy of a file, it must be refused

The seven files are README.md, docs/index.html, docs/guide.html, docs/evidence.html, docs/benchmarks.html, docs/examples.html and docs/status.md. The rule
(the maintainer's): do not append information, and keep only the information that is current. What happened, in which order and at which
stage of the work is in git; these files say what `ml` does, what it does not do, and what was measured with its conditions.

What is not read: the regions between `<!-- gen:NAME -->` and `<!-- /gen:NAME -->` (program output, written by scripts/site.py and checked
against the binary by scripts/check_docs.sh, including the comments of a demonstration session). Everything else, tags and attributes
included, is read line by line, case-insensitively.

PHRASES is the list; each entry is a regular expression and what it catches. ALLOW is the list of legitimate uses, each with the file, the
exact text and the reason; a use that is not in ALLOW is a failure. Adding a phrase to ALLOW is a decision to be argued in review.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = ["README.md", "docs/index.html", "docs/guide.html", "docs/evidence.html", "docs/benchmarks.html", "docs/examples.html", "docs/status.md"]

# (regex, what it catches, an injected sentence for the mutants: it must be refused)
PHRASES = [
    (r"\bstages?\b", "a stage of the work", "Since stage W1 the verbs call the linear model."),
    (r"\bW\d\b", "a stage label (W1)", "This was answered at W1."),
    (r"\bmerged?\b|\bmerges\b", "the merge history", "The six branches were merged into one tree."),
    (r"\bintegration\b", "the integration record", "The counts are those at integration."),
    (r"corrected in place", "a correction narrated", "The figure was corrected in place."),
    (r"\b(first|second|last) re-?run\b|\bthe re-?run\b|\bre-?ran\b|\b(was|were|not) re-?run\b|\bserial re-?run\b", "a first run and a rerun", "The first run printed 515 ms and the rerun agrees."),
    (r"\bused to\b", "a past habit", "The tool used to refuse floats."),
    (r"\bno longer\b|\banymore\b", "before and after", "The reader no longer needs a sibling clone."),
    (r"\bnow\b", "a time marker", "Logistic regression is now reachable."),
    (r"\bso far\b|\bfor now\b|\bat the moment\b|\bnot started\b", "a progress report", "So far three models are built."),
    (r"\byet\b", "a time marker", "Text columns are not reachable yet."),
    (r"\bpending\b", "a pending item", "A quiet-machine run is pending."),
    (r"\bnewly\b|\brecently\b|\bpreviously\b|\bformerly\b|\boriginally\b|\bearlier\b", "a time adverb", "The column is newly encoded."),
    (r"\b(was|were|been) (changed|added|removed|fixed|replaced|renamed|moved|dropped|rewritten|corrected|found)\b", "a past change", "A flag was added to fit."),
    (r"\bthis commit\b|\bat the merged tree\b", "a point in history", "The suite passes on this commit."),
    (r"\bspike\b", "a spike", "The files come from the spike."),
    (r"\bv1-[a-z]+", "a branch name", "The count is from v1-post."),
    (r"\b(came|comes) first\b|\bat first\b", "an order of events", "The failures that came first are shown."),
    (r"\bpredat(e|es|ed)\b", "before and after", "The timings predate the change."),
    (r"\b(before|after) the (switch|change|rename|rewrite)\b", "before and after", "Measured before the switch."),
]

# (file, exact text, reason). A line that holds the text is not read for the phrase that the text carries.
ALLOW = [
    ("docs/evidence.html", "pending()", "the name of a function in tools/ml/spec.cho, not a pending item"),
]

GEN = re.compile(r"<!-- gen:([A-Za-z0-9_-]+) -->\n.*?<!-- /gen:\1 -->", re.S)


def readable(text):
    """The text with the generated regions blanked, line numbers kept."""
    return GEN.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def allowed(rel, line, m):
    """True if the match lies inside an occurrence of an allowed text of this file (not merely on the same line)."""
    for f, t, _ in ALLOW:
        if f != rel:
            continue
        k = line.find(t)
        while k >= 0:
            if k <= m.start() and m.end() <= k + len(t):
                return True
            k = line.find(t, k + 1)
    return False


def scan(rel, text):
    out = []
    for n, line in enumerate(readable(text).splitlines(), 1):
        for rx, what, _ in PHRASES:
            for m in re.finditer(rx, line, re.I):
                if allowed(rel, line, m):
                    continue
                a, b = max(0, m.start() - 40), min(len(line), m.end() + 40)
                out.append("%s:%d: %s: ...%s..." % (rel, n, what, line[a:b].strip()))
    return out


def check(root=ROOT, files=FILES):
    bad = []
    for rel in files:
        bad += scan(rel, (pathlib.Path(root) / rel).read_text())
    return bad


def mutants():
    """Each phrase is injected into each file's copy, outside a generated region; the lint must refuse it. Also: the control passes, and the
    same phrase inside a generated region is not read (the region is the binary's output, checked elsewhere)."""
    results = []
    base = {rel: (ROOT / rel).read_text() for rel in FILES}
    control = [b for rel in FILES for b in scan(rel, base[rel])]
    results.append(("control", not control, "; ".join(control[:2])))
    for i, (rx, what, sentence) in enumerate(PHRASES):
        rel = FILES[i % len(FILES)]
        text = base[rel]
        k = text.rfind("</main>") if rel.endswith(".html") else len(text)
        mutated = text[:k] + "\n" + sentence + "\n" + text[k:]
        hit = scan(rel, mutated)
        results.append(("%s: %s" % (rel, what), bool(hit), sentence))
    # every file refuses the first phrase
    for rel in FILES:
        text = base[rel]
        k = text.rfind("</main>") if rel.endswith(".html") else len(text)
        results.append(("%s: refuses a phrase" % rel, bool(scan(rel, text[:k] + "\n" + PHRASES[0][2] + "\n" + text[k:])), PHRASES[0][2]))
    # an allowed use does not excuse the phrase elsewhere on its line
    for f, t, _ in ALLOW:
        text = base[f]
        k = text.rfind("</main>") if f.endswith(".html") else len(text)
        mutated = text[:k] + "\n" + t + " is pending, and so is this.\n" + text[k:]
        results.append(("an allowed use (%s) does not excuse its line" % t, bool(scan(f, mutated)), ""))
        ok_text = text[:k] + "\n" + t + "\n" + text[k:]
        results.append(("an allowed use (%s) passes" % t, not scan(f, ok_text), ""))
    # a phrase in a generated region is not read
    rel = "README.md"
    mutated = base[rel].replace("<!-- gen:flow -->\n", "<!-- gen:flow -->\n" + PHRASES[0][2] + "\n", 1)
    results.append(("a generated region is not read", mutated != base[rel] and not scan(rel, mutated), ""))
    return results


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        print("files: " + ", ".join(FILES))
        for rx, what, _ in PHRASES:
            print("  %-60s %s" % (rx, what))
        print("allowlist: %d entries" % len(ALLOW))
        for f, t, why in ALLOW:
            print("  %s: %r: %s" % (f, t, why))
        return 0
    if "--mutants" in argv:
        bad = 0
        for name, ok, text in mutants():
            if name == "control":
                verdict = "passes" if ok else "FAILS"
            elif name.startswith("a generated") or name.endswith("passes"):
                verdict = "not read" if ok else "READ"
            else:
                verdict = "refused" if ok else "NOT REFUSED"
            print("%-72s %s" % (name, verdict))
            bad += 0 if ok else 1
        res = mutants()
        n = sum(1 for name, _, _ in res if name != "control" and not name.startswith("a generated") and not name.endswith("passes"))
        print("%d mutants, %d refused, %d not; control and the exemptions as printed" % (n, n - sum(1 for name, ok, _ in res if not ok and name != "control" and not name.startswith("a generated") and not name.endswith("passes")), bad))
        return 1 if bad else 0
    bad = check()
    if bad:
        print("\n".join(bad))
        print("check_prose: %d history phrase(s) in %s" % (len(bad), ", ".join(sorted({b.split(":")[0] for b in bad}))))
        return 1
    print("check_prose: ok (%d phrases, %d files, %d allowed uses)" % (len(PHRASES), len(FILES), len(ALLOW)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

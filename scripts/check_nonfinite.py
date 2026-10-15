#!/usr/bin/env python3
"""N9 of docs/nonfinite.md: cancho-table's `flt.put_float` takes finite doubles only (an infinity is written as the text `i.nf`), so the only call of it in
`tools/` is the one inside `verbs.put_float_text`, which refuses to hand it bits that are not finite. This script fails when another call is there.

    python3 scripts/check_nonfinite.py            # exit 1 and a line for each call outside the door
    python3 scripts/check_nonfinite.py --root DIR # a copy of the tree (the mutants)

Allowed, each with its reason: the door itself, and tools/ml-serve/num.cho, which writes a double of a request after it refused every number past the range of a double
(`serve.number-not-exact`), so its argument is finite by its own rule (docs/serving-http.md section 4)."""
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else pathlib.Path(__file__).resolve().parents[1]
CALL = re.compile(r"\bflt\.put_float\s*\(")

ALLOW = {
    "tools/ml-serve/num.cho": "a request's double after the range check of the number module",
}


def door_span(text):
    """(start, end) offsets of the body of `pub fn put_float_text`, or None"""
    m = re.search(r"pub fn put_float_text\b", text)
    if not m:
        return None
    i = text.index("{", m.end())
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return (i, j)
    return None


def main():
    bad = []
    door_calls = 0
    for path in sorted((ROOT / "tools").rglob("*.cho")):
        rel = str(path.relative_to(ROOT))
        text = path.read_text()
        span = door_span(text) if rel == "tools/ml/verbs.cho" else None
        for m in CALL.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            code_line = text.splitlines()[line - 1].lstrip()
            if code_line.startswith("//"):
                continue
            if span and span[0] <= m.start() <= span[1]:
                door_calls += 1
                continue
            if rel in ALLOW:
                continue
            bad.append("%s:%d: a call of flt.put_float outside verbs.put_float_text" % (rel, line))
    if door_calls != 1:
        bad.append("tools/ml/verbs.cho: put_float_text holds %d calls of flt.put_float, wanted exactly 1" % door_calls)
    verbs = (ROOT / "tools/ml/verbs.cho").read_text()
    span = door_span(verbs)
    if span and "bits_finite(bits)" not in verbs[span[0]:span[1]]:
        bad.append("tools/ml/verbs.cho: put_float_text does not test bits_finite before it calls flt.put_float")
    if bad:
        print("\n".join(bad))
        return 1
    print("check_nonfinite: one call of flt.put_float in tools/ml (the door, behind bits_finite); tools/ml-serve/num.cho allowed: %s" % ALLOW["tools/ml-serve/num.cho"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

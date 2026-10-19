#!/usr/bin/env python3
"""The import closure of cancho-table's JSON lines modules (docs/jsonl.md section 1): modules, lines, and whether a module of the closure declares the table's plan.

    python3 scripts/jsonl_closure.py DIR      # DIR is a checkout of cancho-table
"""
import os
import re
import sys


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    base = os.path.join(argv[1], "tools", "table")
    seen = {}

    def visit(m):
        if m in seen:
            return
        path = os.path.join(base, m + ".cho")
        if not os.path.exists(path):
            seen[m] = None
            return
        seen[m] = re.findall(r"^import\s+([\w.]+)\s*;", open(path).read(), re.M)
        for i in seen[m]:
            if not i.startswith("std."):
                visit(i)

    for root in ("jsonl", "jrec", "jscan", "jfields", "jflat"):
        seen.clear()
        visit(root)
        mods = sorted(m for m in seen if seen[m] is not None)
        ext = sorted(m for m in seen if seen[m] is None)
        lines = sum(len(open(os.path.join(base, m + ".cho")).read().splitlines()) for m in mods)
        plan = [m for m in mods if re.search(r"struct\s+Query\b", open(os.path.join(base, m + ".cho")).read())]
        print("%-8s %2d modules %6d lines  declares Query: %-8s  imports outside the repository: %s" % (root, len(mods), lines, plan or "no", ext or "none"))
    return 0


sys.exit(main(sys.argv))

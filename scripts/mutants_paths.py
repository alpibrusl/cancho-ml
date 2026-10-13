#!/usr/bin/env python3
"""Mutants of the path gate (scripts/gate_paths.py, docs/lanes.md section 1): remove or weaken each check, and the gate must fail.

    python3 scripts/mutants_paths.py --check      # every pattern applies exactly once
    python3 scripts/mutants_paths.py [substring]  # build and run

The gate builds five drivers per run (about 8 s), so the whole list takes a few minutes."""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mutlib

ROOT = mutlib.ROOT
# the mutated copy has no build/deps and no sibling clone of cancho-table: name them absolutely
os.environ.setdefault("ML_DEPS", str(ROOT / "build" / "deps"))
os.environ.setdefault("CANCHO_TABLE", str(ROOT.parent / "cancho-table-main"))

H = "lib/core/pathcheck.cho"
MC = "tools/models-check/main.cho"
TC = "tools/tree-check/main.cho"
RC = "tools/reader-check/main.cho"
XC = "tools/metrics-check/main.cho"
DD = "if int_of(p[i]) == '.' && int_of(p[i + 1]) == '.' {\n            return true;"
M = [
 ("helper-always-false", H, DD, "if int_of(p[i]) == '.' && int_of(p[i + 1]) == '.' {\n            return false;"),
 ("helper-misses-trailing", H, "while i + 1 < len(p) {", "while i + 2 < len(p) {"),
 ("helper-misses-leading", H, "    var i = 0;\n    while i + 1 < len(p) {", "    var i = 1;\n    while i + 1 < len(p) {"),
 ("helper-single-dot", H, "int_of(p[i]) == '.' && int_of(p[i + 1]) == '.'", "int_of(p[i]) == '.' && int_of(p[i + 1]) == '/'"),
 ("tag-changed", H, 'return "path.dotdot";', 'return "path.dot-dot";'),
 ("exit-code-changed", H, "pub fn exit_code() -> [] int {\n    return 2;", "pub fn exit_code() -> [] int {\n    return 1;"),
 ("models-check-check-off", MC, "                dotdot = 1;", "                dotdot = 0;"),
 ("models-check-x-unchecked", MC, "pathcheck.has_double_dot(arg(g, 2)) || model", "model"),
 ("models-check-y-unchecked", MC, " || model != 3 && pathcheck.has_double_dot(arg(g, 3)) {", " {"),
 ("models-check-refusal-ignored", MC, "    if dotdot != 0 {\n        // a path", "    if false {\n        // a path"),
 ("tree-check-check-off", TC, "            dotdot = true;", "            dotdot = false;"),
 ("tree-check-x-unchecked", TC, "if ix >= 0 && pathcheck.has_double_dot(arg(g, ix)) || ", "if "),
 ("tree-check-y-unchecked", TC, " || iy >= 0 && pathcheck.has_double_dot(arg(g, iy)) ", " "),
 ("tree-check-miss-unchecked", TC, " || im >= 0 && pathcheck.has_double_dot(arg(g, im)) {", " {"),
 ("tree-check-mode-length", TC, "if len(m) >= 7 && m[5] == byte_of('f') && m[6] == byte_of('i') {", "if m[5] == byte_of('f') && m[6] == byte_of('i') {"),
 ("tree-check-out-length", TC, "if len(arg(g, oi)) >= 5 && arg(g, oi)[4] == byte_of('h') {", "if arg(g, oi)[4] == byte_of('h') {"),
 ("reader-check-check-off", RC, "if !bad && pathcheck.has_double_dot(file) {", "if false && pathcheck.has_double_dot(file) {"),
 ("reader-check-exit", RC, "        code = pathcheck.exit_code();\n    } else if bad", "        code = 8;\n    } else if bad"),
 ("metrics-check-operand-unguarded", XC, "    if arg_count(g) <= i {\n        return \"\";", "    if arg_count(g) < 0 {\n        return \"\";"),
]
sys.exit(mutlib.run(M, ["python3", "scripts/gate_paths.py"], "mutants_paths", timeout=900))

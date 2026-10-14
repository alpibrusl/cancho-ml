#!/usr/bin/env python3
"""P4 (docs/serving.md section 1): `ml` and `ml-predict` are built from the same sources.

The file list of `ml-predict` in cancho.toml is the list of `ml` with the directory tools/ml replaced by its files other than main.cho (a directory would bring in the
main of `ml`) and tools/ml-predict (its own main) in place of generated/ml's partner. This script fails when

* a file of tools/ml other than main.cho is missing from the list of ml-predict, or the list names a file that is not there;
* any other source directory differs between the two binaries;
* tools/ml-predict holds anything but a main.cho of at most 150 lines (a second copy of a module would be a second definition);
* a module is defined in two files of the union of the two lists (a copy under another name of the file).

    python3 scripts/check_bins.py
    python3 scripts/check_bins.py --mutants     # each failure above is injected into a copy of the tree: it must be refused
"""

import pathlib
import re
import shutil
import sys
import tempfile
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def bins(root):
    with open(root / "cancho.toml", "rb") as f:
        proj = tomllib.load(f)
    return {b["name"]: b for b in proj["bin"]}


def check(root):
    problems = []
    b = bins(root)
    ml, mp = b["ml"]["sources"], b["ml-predict"]["sources"]
    own = {"tools/ml", "generated/ml", "tools/ml-predict", "generated/ml-predict"}
    if [s for s in ml if s not in own] != [s for s in mp if s not in own and not s.startswith("tools/ml/")]:
        problems.append("the source directories of ml and ml-predict differ: %s / %s" % ([s for s in ml if s not in own], [s for s in mp if s not in own and not s.startswith("tools/ml/")]))
    for want in ("tools/ml", "generated/ml"):
        if want not in ml:
            problems.append("ml does not list %s" % want)
    for want in ("tools/ml-predict", "generated/ml-predict"):
        if want not in mp:
            problems.append("ml-predict does not list %s" % want)
    here = sorted("tools/ml/" + p.name for p in (root / "tools" / "ml").glob("*.cho") if p.name != "main.cho")
    listed = sorted(s for s in mp if s.startswith("tools/ml/"))
    if "tools/ml/main.cho" in listed:
        problems.append("ml-predict lists the main of ml")
    for f in here:
        if f not in listed:
            problems.append("%s is a source of ml and is not in the list of ml-predict" % f)
    for f in listed:
        if f not in here:
            problems.append("ml-predict lists %s, which is not a file" % f)
    files = sorted(p.name for p in (root / "tools" / "ml-predict").glob("*") if p.is_file())
    if files != ["main.cho"]:
        problems.append("tools/ml-predict holds %s: only main.cho (the entry point) belongs there" % files)
    elif len((root / "tools" / "ml-predict" / "main.cho").read_text().splitlines()) > 150:
        problems.append("tools/ml-predict/main.cho is over 150 lines: shared code belongs in tools/ml")
    # one definition of every module across both lists
    seen = {}
    union = list(dict.fromkeys(ml + mp))
    for entry in union:
        p = root / entry
        if entry.startswith("generated/") or not p.exists():
            # toolbox.built is each binary's own module (its authority, schema and pin): the same name in two files is the design
            continue
        for f in sorted(p.glob("*.cho")) if p.is_dir() else [p]:
            m = re.search(r"^module ([\w.]+);", f.read_text(), re.M)
            if m:
                if m.group(1) in seen and seen[m.group(1)] != f:
                    problems.append("module %s is defined in %s and in %s" % (m.group(1), seen[m.group(1)].relative_to(root), f.relative_to(root)))
                seen[m.group(1)] = f
    return problems


def mutants():
    def copy():
        d = pathlib.Path(tempfile.mkdtemp(prefix="bins-"))
        for rel in ("cancho.toml", "tools/ml", "tools/ml-predict", "generated", "lib"):
            src = ROOT / rel
            if src.is_dir():
                shutil.copytree(src, d / rel)
            else:
                (d / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, d / rel)
        return d

    def edit(d, rel, old, new):
        p = d / rel
        t = p.read_text()
        assert old in t, (rel, old)
        p.write_text(t.replace(old, new, 1))

    cases = [
        ("a file of tools/ml left out of the list", lambda d: edit(d, "cancho.toml", '"tools/ml/mdesc.cho", ', "")),
        ("a file in the list that is not there", lambda d: edit(d, "cancho.toml", '"tools/ml-predict", "generated/ml-predict"]', '"tools/ml/nothing.cho", "tools/ml-predict", "generated/ml-predict"]')),
        ("the main of ml in the list", lambda d: edit(d, "cancho.toml", '"tools/ml/mdesc.cho", ', '"tools/ml/mdesc.cho", "tools/ml/main.cho", ')),
        ("a directory missing from ml-predict", lambda d: edit(d, "cancho.toml", '"lib/metrics", "lib/models", "lib/models/kmeans", "lib/models/linear", "lib/models/logistic", "lib/models/tree", "lib/models/treewire", "lib/transforms", "lib/pipeline", "tools/ml/app.cho"', '"lib/models", "lib/models/kmeans", "lib/models/linear", "lib/models/logistic", "lib/models/tree", "lib/models/treewire", "lib/transforms", "lib/pipeline", "tools/ml/app.cho"')),
        ("a copy of a module under tools/ml-predict", lambda d: shutil.copy(d / "tools/ml/paths.cho", d / "tools/ml-predict/paths.cho")),
        ("a copy of a module in tools/ml under another name", lambda d: shutil.copy(d / "tools/ml/paths.cho", d / "tools/ml/paths2.cho")),
        ("an entry point that grew", lambda d: (d / "tools/ml-predict/main.cho").write_text((d / "tools/ml-predict/main.cho").read_text() + "\n" * 200)),
    ]
    bad = 0
    for what, mutate in cases:
        d = copy()
        try:
            mutate(d)
            problems = check(d)
            ok = bool(problems)
            print("%-60s %s" % (what, "refused" if ok else "NOT REFUSED"))
            bad += 0 if ok else 1
        finally:
            shutil.rmtree(d, ignore_errors=True)
    base = check(ROOT)
    print("the tree: %s" % ("clean" if not base else base))
    return 1 if bad or base else 0


def main():
    if "--mutants" in sys.argv[1:]:
        return mutants()
    problems = check(ROOT)
    for p in problems:
        print("FAIL " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

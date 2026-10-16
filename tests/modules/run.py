#!/usr/bin/env python3
"""Judge the admission script with mutant modules (docs/modules.md section 7).

    python3 tests/modules/run.py              the control and the five mutant modules
    python3 tests/modules/run.py --check      every defect applies to a scratch copy (no build, no admit)
    python3 tests/modules/run.py ID           one mutant

The control (the tree as it is, with a copy of `linear` as a second model removed again: no defect) must be ADMITTED by `admit.py --fast`; each mutant must be
REFUSED, and the check named for it must be among the checks that FAIL. Exit 1 otherwise."""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
import admit  # noqa: E402
import mutants  # noqa: E402


def scratch(dest):
    ignore = shutil.ignore_patterns(".git", "build", "__pycache__", "*.pyc")
    shutil.copytree(ROOT, dest, ignore=ignore, symlinks=True)
    comp = dest.parent / "cancho-compiler-0207556"
    if not comp.exists():
        comp.symlink_to(ROOT.parent / "cancho-compiler-0207556")
    deps = ROOT / "build" / "deps"
    (dest / "build").mkdir(exist_ok=True)
    shutil.copytree(deps, dest / "build" / "deps")


def refresh(tree):
    """regenerate what a module changes (the glue, the schemas, the authority record), as its author would; the defect under test must still show"""
    env = dict(os.environ, CANCHO=admit.compiler(tree))
    for cmd in (["scripts/gen_models.py"], ["scripts/schemas.py"], ["scripts/manifest.py"]):
        subprocess.run([sys.executable, str(tree / cmd[0])], capture_output=True, text=True, cwd=tree, env=env)


def judge(mid, name, fn, want, needs_bin):
    with tempfile.TemporaryDirectory(prefix="ml-module-") as td:
        tree = pathlib.Path(td) / "tree"
        scratch(tree)
        fn(tree)
        refresh(tree)
        lines = []
        res = admit.admit(tree, name, fast=True, build=needs_bin, out=lines.append)
        failed = sorted(q for q, (s, _) in res.items() if s == "FAIL")
        ok = want in failed
        return ok, failed, lines


def control():
    with tempfile.TemporaryDirectory(prefix="ml-module-") as td:
        tree = pathlib.Path(td) / "tree"
        scratch(tree)
        # a second, honest model: a copy of linear under another name and kind
        mutants.add_model(tree, "twin", 8, "model_twin")
        refresh(tree)
        lines = []
        res = admit.admit(tree, "twin", fast=True, build=True, out=lines.append)
        failed = sorted(q for q, (s, _) in res.items() if s == "FAIL")
        return not failed, failed, lines


def main(argv):
    if "--check" in argv:
        bad = 0
        for mid, name, fn, want, nb in mutants.MUTANTS:
            with tempfile.TemporaryDirectory(prefix="ml-module-") as td:
                tree = pathlib.Path(td) / "tree"
                scratch(tree)
                try:
                    fn(tree)
                    admit.gen_models.load_manifests(tree)
                except Exception as e:  # noqa: BLE001
                    print("INVALID %s: %s" % (mid, e))
                    bad += 1
        print("%d mutant modules apply" % len(mutants.MUTANTS))
        return 1 if bad else 0
    only = [a for a in argv if not a.startswith("-")]
    bad = 0
    if not only:
        ok, failed, lines = control()
        print("%-26s %s" % ("control (an honest twin)", "ADMITTED" if ok else "REFUSED: " + ", ".join(failed)))
        if not ok:
            print("\n".join("    " + l for l in lines))
            bad += 1
    for mid, name, fn, want, nb in mutants.MUTANTS:
        if only and mid not in only:
            continue
        ok, failed, lines = judge(mid, name, fn, want, nb)
        print("%-26s %s  (must fail %s; failed: %s)" % (mid, "REFUSED" if ok else "NOT REFUSED", want, ", ".join(failed) or "none"))
        if not ok:
            print("\n".join("    " + l for l in lines))
            bad += 1
    print("\n%s" % ("every mutant module refused, the control admitted" if not bad else "%d problems" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

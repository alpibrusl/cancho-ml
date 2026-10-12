"""Shared half of the mutation scripts. A mutant is (name, file, old, new): `old` must occur exactly once in `file` (a path
relative to the repository root) and differ from `new`. Each mutant is applied to a COPY of lib/, tests/, tools/ in a temporary
directory, so the sources are never touched. A check command is run in the copy; the mutant is killed if the command exits
non-zero. `--check` only verifies that every pattern applies (no compiler).

    python3 scripts/X_mutants.py --check
    python3 scripts/X_mutants.py [name-substring ...]

The compiler is $CANCHO or the pinned build next to this repository. Exit 1 if a mutant survives or cannot apply."""
import os, pathlib, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CANCHO = ROOT.parent / "cancho-compiler-a4572ea" / "target" / "release" / "cancho"


def cancho():
    return os.environ.get("CANCHO", str(DEFAULT_CANCHO))


def problems(mutants):
    out, names, cache = [], set(), {}
    for name, file, old, new in mutants:
        if name in names:
            out.append("mutant %s: named twice" % name)
        names.add(name)
        p = ROOT / file
        if not p.exists():
            out.append("mutant %s: no such file %s" % (name, file)); continue
        t = cache.setdefault(file, p.read_text())
        n = t.count(old)
        if n != 1:
            out.append("mutant %s: pattern occurs %d times in %s (must be exactly 1)" % (name, n, file))
        if old == new:
            out.append("mutant %s: replacement equals pattern" % name)
    return out


def run(mutants, command, label, timeout=1800, allowed=None):
    """command: list of argv, run in the mutated copy, with {cancho} replaced. Returns exit code for the script."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bad = problems(mutants)
    if bad:
        print("\n".join(bad)); return 1
    if "--check" in sys.argv:
        print("%s: %d mutants, every pattern applies once" % (label, len(mutants))); return 0
    allowed = allowed or {}
    survivors, killed, explained = [], 0, 0
    for name, file, old, new in mutants:
        if args and not any(a in name for a in args):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            for d in ("lib", "tests", "tools", "scripts"):
                if (ROOT / d).exists():
                    shutil.copytree(ROOT / d, pathlib.Path(tmp) / d)
            f = pathlib.Path(tmp) / file
            f.write_text(f.read_text().replace(old, new, 1))
            argv = [a.replace("{cancho}", cancho()) for a in command]
            r = subprocess.run(argv, cwd=tmp, capture_output=True, text=True, timeout=timeout)
        ok = r.returncode != 0 or "FAILED" in r.stdout
        if ok:
            killed += 1; print("%-40s killed" % name, flush=True)
        elif name in allowed:
            explained += 1; print("%-40s SURVIVED (explained: %s)" % (name, allowed[name]), flush=True)
        else:
            survivors.append(name); print("%-40s SURVIVED" % name, flush=True)
    print("%s: %d killed, %d survived and explained, %d unexplained" % (label, killed, explained, len(survivors)))
    if survivors:
        print("SURVIVORS: " + ", ".join(survivors)); return 1
    return 0

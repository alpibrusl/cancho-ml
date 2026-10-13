#!/usr/bin/env python3
"""The golden record of the constant model (docs/wiring.md gate W1): for every fixture of the CLI slice that does not use a flag of the wiring stage, the
exit status, the SHA-256 of standard output (the sandbox root and the binary's path replaced by @ROOT and @ML) and the SHA-256 of every file the case left
in its sandbox, as written by a given binary. `tests/conformance/test_wire_baseline.py` runs the same cases against the binary under test and compares.

    ML_BIN=/path/to/the/v1-post/binary python3 scripts/golden_constant.py > tests/conformance/golden_constant.json

The record in the repository was written by the binary of `v1-post` (commit 1b550a5, built before this stage); it is never rewritten to make a test pass.
"""

import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
sys.path.insert(0, str(ROOT / "scripts"))

NEW_FLAGS = {"--plan", "--step", "--l2", "--iterations", "--positive", "--split", "--seed", "--proba",
             # the categorical columns of stage 1 of issue 21 (docs/categorical.md)
             "--cat", "--na-values", "--max-categories", "--max-string-bytes", "--max-pool-bytes", "--max-cat-columns",
             # the tree and the forest (docs/wiring-trees.md section 2)
             "--trees", "--max-depth", "--min-leaf", "--bins", "--mtry", "--bootstrap", "--max-nodes"}
# the refusal of an unknown model name lists the models: it changes with the list, by design
LISTS_THE_MODELS = {"args-bad-value-model"}


def uses_new(c):
    argv = [a for a in c["argv"] if isinstance(a, str)]
    if any(a in NEW_FLAGS for a in argv) or "--threads" in argv:
        return True
    if "fit" in argv and "--max-bytes" in argv:
        # --max-bytes limits the file a fit writes (docs/wiring-trees.md section 10); for predict and eval it is older
        return True
    if "linear" in argv or "logistic" in argv or "tree" in argv or "forest" in argv:
        return True
    for s in c["setup"]:
        if s[0] == "fit" and (any(a in NEW_FLAGS or a in ("linear", "logistic", "tree", "forest", "--threads") for a in s[1:])):
            return True
    return False


def record(c):
    from runner import run_case
    sb, r = run_case(c)
    try:
        out = r.stdout.replace(str(sb.dir).encode(), b"@ROOT").replace(os.fsencode(r.argv[0]), b"@ML")
        files = {}
        for p in sorted(sb.dir.rglob("*")):
            if p.is_file() and not p.is_symlink():
                files[str(p.relative_to(sb.dir))] = hashlib.sha256(p.read_bytes()).hexdigest()
        return {"status": r.status, "stdout": hashlib.sha256(out).hexdigest(), "files": files}
    finally:
        for p in sb.dir.rglob("*"):
            try:
                p.chmod(0o755 if p.is_dir() else 0o644)
            except OSError:
                pass
        sb.cleanup()


def cases():
    from fixtures import CASES
    return [c for c in CASES if not uses_new(c) and not c.get("skip_if_root") and c["id"] not in LISTS_THE_MODELS]


def main():
    out = {}
    for c in cases():
        out[c["id"]] = record(c)
    json.dump(out, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

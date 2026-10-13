#!/usr/bin/env python3
"""The golden record of every numeric fixture (docs/categorical.md gate C9): for every case of tests/conformance/fixtures.py that exists
before the categorical stage (its id is in the record), the exit status, the SHA-256 of standard output (the sandbox root and the binary's path
replaced by @ROOT and @ML) and the SHA-256 of every file the case left in its sandbox, as written by the binary of `v1-next` before stage 1 of
issue 21 (commit 7d6183d). It extends golden_constant.json (W1, the constant model only) to the wired models and to the refusals of the reader.
`tests/conformance/test_cat_baseline.py` runs the same cases against the binary under test. The record is never rewritten to make a test pass.

    ML_BIN=/path/to/the/7d6183d/binary python3 scripts/golden_numeric.py > tests/conformance/golden_numeric.json
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
sys.path.insert(0, str(ROOT / "scripts"))

import golden_constant  # noqa: E402

# the refusal of an unknown model name lists the models: it changes with the list, by design (as in W1)
SKIP = {"args-bad-value-model"}


def cases():
    from fixtures import CASES
    return [c for c in CASES if not c.get("skip_if_root") and c["id"] not in SKIP]


def main():
    out = {}
    for c in cases():
        out[c["id"]] = golden_constant.record(c)
    json.dump(out, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

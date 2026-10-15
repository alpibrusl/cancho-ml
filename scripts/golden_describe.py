#!/usr/bin/env python3
"""Record the golden answers of `describe` (tests/conformance/golden_describe.json): for each model of the serving corpus (tests/conformance/serving_corpus.py), the
bytes `ml describe --model-file M --limit 4` prints. The corpus is built from a seeded generator and `ml fit`, so the answers (the file sizes and digests included)
are a function of the sources; a change to them is re-recorded here on purpose, in a commit that says why.

    python3 scripts/golden_describe.py          # write
    python3 scripts/golden_describe.py --check  # exit 1 if the file differs
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import Sandbox  # noqa: E402
import serving_corpus as C  # noqa: E402

OUT = ROOT / "tests" / "conformance" / "golden_describe.json"


def record():
    sb = Sandbox()
    try:
        models, _ = C.build(sb)
        out = {}
        for mid, (mf, kind, fam) in models.items():
            r = sb.run("describe", "--model-file", mf, "--limit", "4")
            assert r.status == 0, r
            out[mid] = r.stdout.decode()
        return json.dumps(out, indent=1, sort_keys=True) + "\n"
    finally:
        sb.cleanup()


if __name__ == "__main__":
    text = record()
    if "--check" in sys.argv[1:]:
        sys.exit(0 if OUT.exists() and OUT.read_text() == text else 1)
    OUT.write_text(text)

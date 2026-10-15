#!/usr/bin/env python3
"""The golden record of the linear and logistic model files (docs/wiring-trees.md gate WT12): for 60 seeded cases of `scripts/oracle_wire.py` (missing cells,
constant columns, imputed, scaled, bucketed and one-hot columns, splits, both positive labels), the exit status, the SHA-256 of the answer of `fit` (the
sandbox path replaced) and the SHA-256 of the model file, as written by the binary of the parent commit `d481593`, before the tree and the forest were wired.
`tests/conformance/test_planned_baseline.py` runs the same cases against the binary under test. The record is never rewritten to make a test pass.

    ML_BIN=/path/to/the/d481593/binary python3 scripts/golden_planned.py > tests/conformance/golden_planned.json
"""

import hashlib
import json
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_wire as ow  # noqa: E402

SEEDS = range(1, 31)


def cases():
    out = []
    for model in ("linear", "logistic"):
        for seed in SEEDS:
            out.append(("%s-%d" % (model, seed), ow.gen_case(seed * 7919 + (0 if model == "linear" else 101), model)))
    return out


def record(case, tmp):
    (tmp / "m.cml").unlink(missing_ok=True)
    (tmp / "d.csv").write_text(case.csv())
    rc, out = ow.ml("--root", tmp, *ow.fit_args(case, "m.cml"), "d.csv")
    f = tmp / "m.cml"
    return {"status": rc, "stdout": hashlib.sha256(out.replace(str(tmp), "@ROOT").replace(str(ow.BIN), "@ML").encode()).hexdigest(),
            "file": hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None}


def main():
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for name, case in cases():
            out[name] = record(case, pathlib.Path(td))
    json.dump(out, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

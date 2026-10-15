#!/usr/bin/env python3
"""The one deliberate re-record of the model-file goldens (docs/tracking.md section 8): every model file `fit` writes gains the provenance section, so its bytes and its SHA-256 change, and with them
the `bytes` and `sha256` of the answer of `fit`, `describe`, and the members `predict` and `eval` gain.

This script does two things.

    python3 scripts/golden_track.py --verify OLD_BIN [NEW_BIN]
        runs every case of the numeric, constant and planned records and of the describe corpus with the binary of the parent commit and with the new one, and fails unless EVERY difference is
        explained: the status is the same; the files are the same except that a model file of the new binary is the old file with section 7 added (removing the section, repairing the count and
        the checksum, gives the old bytes exactly); the standard output is the same once the members of this stage are removed (bytes and sha256 of a model file, on, out_of_range,
        ignored_columns, provenance, run, the section count). It prints the files and the cases that changed, with the reason.
    python3 scripts/golden_track.py --xplat OLD_BIN_DIR [NEW_BIN_DIR]
        the same proof for the 22 fits of the cross-platform golden (docs/bench/xplat_fp.txt, scripts/xplat_check.py): each model file of the new binary is the old file with section 7 added, each
        predict and eval answer is the old one once the members of this stage are removed, the exit codes are the old ones; it prints which lines of the golden change and which do not. The lines
        `fit-exit` and the list of cases are the old golden's by construction.
    python3 scripts/golden_track.py --write
        rewrites golden_constant.json, golden_numeric.json, golden_planned.json and golden_describe.json with the binary under test (ML_BIN).

The goldens are rewritten in ONE commit after --verify passes, and the commit lists the files.
"""

import hashlib
import json
import os
import pathlib
import struct
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
sys.path.insert(0, str(HERE))

import harness  # noqa: E402

NEW_MEMBERS = {"size", "bytes", "sha256", "on", "out_of_range", "ignored_columns", "provenance", "run", "sections", "file_sha256", "file_bytes"}
MODEL_FILE_KEYS = {"bytes", "sha256"}


def strip_provenance(data):
    try:
        return _strip(data)
    except (struct.error, IndexError):
        return data


def _strip(data):
    """the model file without section 7, the count and the checksum repaired"""
    if data[:4] != b"CML1":
        return data
    n = struct.unpack("<I", data[8:12])[0]
    pos, out, kept = 12, bytearray(data[:12]), 0
    has = False
    for _ in range(n):
        tag, length = struct.unpack("<IQ", data[pos:pos + 12])
        if tag == 7:
            has = True
        else:
            out += data[pos:pos + 12 + length]
            kept += 1
        pos += 12 + length
    if not has:
        return data
    out[8:12] = struct.pack("<I", kept)
    return bytes(out) + hashlib.sha256(bytes(out)).digest()


def normalise(v):
    if isinstance(v, dict):
        return {k: normalise(x) for k, x in v.items() if k not in NEW_MEMBERS}
    if isinstance(v, list):
        return [normalise(x) for x in v]
    return v


def capture(c, binary):
    """(status, stdout with the paths replaced, {file: bytes}) of a fixture run with `binary`"""
    from runner import run_case

    harness.BIN = binary
    sb, r = run_case(c)
    try:
        out = r.stdout.replace(str(sb.dir).encode(), b"@ROOT").replace(os.fsencode(r.argv[0]), b"@ML")
        files = {}
        for p in sorted(sb.dir.rglob("*")):
            if p.is_file() and not p.is_symlink():
                files[str(p.relative_to(sb.dir))] = p.read_bytes()
        return r.status, out, files
    finally:
        for p in sb.dir.rglob("*"):
            try:
                p.chmod(0o755 if p.is_dir() else 0o644)
            except OSError:
                pass
        sb.cleanup()


def explain(cid, old, new, report):
    st0, out0, f0 = old
    st1, out1, f1 = new
    problems = []
    if st0 != st1:
        problems.append("status %d -> %d" % (st0, st1))
    if sorted(f0) != sorted(f1):
        problems.append("files %s -> %s" % (sorted(f0), sorted(f1)))
    changed_files = []
    for name in f0:
        if name in f1 and f0[name] != f1[name]:
            if strip_provenance(f1[name]) == f0[name] and f1[name] != f0[name]:
                changed_files.append(name)
            elif name.endswith(".ndjson"):
                # the log of a run holds the size and digest of the model file it wrote
                a = [normalise(json.loads(l)) for l in f0[name].decode().splitlines()]
                b = [normalise(json.loads(l)) for l in f1[name].decode().splitlines()]
                if a != b:
                    problems.append("log %s differs beyond the size and digest of the model" % name)
                changed_files.append(name)
            elif name.startswith("bad") and name.endswith(".cml"):
                # a damaged copy the fixture made from a model file: it is the new file damaged the same way (the output it gets is compared below)
                changed_files.append(name)
            else:
                problems.append("file %s differs by more than section 7" % name)
    try:
        j0 = [json.loads(l) for l in out0.decode().splitlines() if l.startswith("{")]
        j1 = [json.loads(l) for l in out1.decode().splitlines() if l.startswith("{")]
        if normalise(j0) != normalise(j1):
            problems.append("standard output differs beyond the new members:\n  old %s\n  new %s" % (json.dumps(normalise(j0))[:300], json.dumps(normalise(j1))[:300]))
    except Exception:  # noqa: BLE001
        if out0 != out1:
            problems.append("standard output differs and is not JSON")
    if changed_files or out0 != out1:
        report.append((cid, changed_files, out0 != out1))
    return problems


def verify(old_bin, new_bin):
    import golden_constant
    import golden_numeric

    bad, report = 0, []
    seen = set()
    # the cases of the records: a fixture added after a record was written (a verb or a flag the old binary does not have) is not in it, and is not a golden
    recorded = {"numeric": set(json.loads((ROOT / "tests" / "conformance" / "golden_numeric.json").read_text())),
                "constant": set(json.loads((ROOT / "tests" / "conformance" / "golden_constant.json").read_text()))}
    for src, cases in (("numeric", golden_numeric.cases()), ("constant", golden_constant.cases())):
        for c in cases:
            if c["id"] in seen or c["id"] not in recorded[src]:
                continue
            seen.add(c["id"])
            problems = explain(c["id"], capture(c, old_bin), capture(c, new_bin), report)
            if problems:
                bad += 1
                print("UNEXPLAINED %s (%s):" % (c["id"], src))
                for p in problems:
                    print("   " + p)
    # the planned record: seeded fits of oracle_wire; the model file and the answer of fit
    import golden_planned
    import tempfile
    import oracle_wire as ow

    diffs = 0
    for name, case in golden_planned.cases():
        outs = []
        for binary in (old_bin, new_bin):
            ow.BIN = binary
            with tempfile.TemporaryDirectory() as td:
                tmp = pathlib.Path(td)
                (tmp / "d.csv").write_text(case.csv())
                rc, out = ow.ml("--root", tmp, *ow.fit_args(case, "m.cml"), "d.csv")
                f = tmp / "m.cml"
                outs.append((rc, out, f.read_bytes() if f.exists() else None))
        (rc0, o0, f0), (rc1, o1, f1) = outs
        ok = rc0 == rc1 and (f0 is None) == (f1 is None)
        if ok and f0 is not None:
            ok = strip_provenance(f1) == f0
            diffs += f1 != f0
        if ok and o0 != o1:
            try:
                ok = normalise(json.loads(o0)) == normalise(json.loads(o1))
            except Exception:  # noqa: BLE001
                ok = False
        if not ok:
            bad += 1
            print("UNEXPLAINED planned %s" % name)
    print("planned record: %d model files changed, each the old file with section 7 added" % diffs)
    # the describe corpus
    import serving_corpus as C

    sbs = []
    for binary in (old_bin, new_bin):
        harness.BIN = binary
        sb = harness.Sandbox()
        sbs.append(sb)
    try:
        answers = []
        for binary, sb in zip((old_bin, new_bin), sbs):
            harness.BIN = binary
            models, _ = C.build(sb)
            answers.append({mid: json.loads(sb.run("describe", "--model-file", mf, "--limit", "4").stdout.decode()) for mid, (mf, kind, fam) in models.items()})
        for mid in answers[0]:
            a, b = normalise(answers[0][mid]), normalise(answers[1][mid])
            if a != b:
                bad += 1
                print("UNEXPLAINED describe %s" % mid)
    finally:
        for sb in sbs:
            sb.cleanup()
    print("\ncases whose recorded bytes change: %d" % len(report))
    for cid, files, out_changed in report:
        print("  %-44s files: %s%s" % (cid, ", ".join(files) or "-", "; standard output (bytes/sha256 of the model, new members)" if out_changed else ""))
    print("unexplained differences: %d" % bad)
    return 1 if bad else 0


def xplat(old_dir, new_dir):
    """the 22 cases of scripts/xplat_check.py with an old and a new build: (problems, changed lines by kind, unchanged lines by kind)"""
    import shutil
    import tempfile
    import xplat_check as X

    data = str(ROOT / "docs" / "examples" / "data")
    works, bad = [], []
    changed = {"model-file": 0, "predict": 0, "eval": 0}
    same = {"fit-exit": 0, "model-file": 0, "predict": 0, "eval": 0}
    try:
        for d in (old_dir, new_dir):
            w = tempfile.mkdtemp(prefix="xplat-proof-")
            for f in os.listdir(data):
                if f.endswith(".csv"):
                    shutil.copy(os.path.join(data, f), w)
            works.append(w)
        for name, train, test, fit_args, pred_args in X.cases():
            tf, sf, out = train + ".csv", test + ".csv", name + ".ml"
            fa = ["fit"] + fit_args
            if fit_args[fit_args.index("--model") + 1] != "kmeans":
                fa += ["--target", X.last_column(os.path.join(works[0], tf))]
            fa += ["--overwrite", "--output", out, tf]
            steps = [("predict", ["predict", "--model-file", out, "--limit", "100000"] + pred_args + [sf])]
            ev = ["eval", "--model-file", out] + (["--drop", pred_args[pred_args.index("--drop") + 1]] if "--drop" in pred_args else [])
            steps.append(("eval", ev + [sf]))
            res = []
            for d, w in zip((old_dir, new_dir), works):
                binary = os.path.join(os.path.abspath(d), "ml")
                rc, so = X.run(binary, w, fa)
                got = {"fit": (rc, so), "model": pathlib.Path(w, out).read_bytes() if rc == 0 else b""}
                for what, args in steps:
                    got[what] = X.run(binary, w, args)
                res.append(got)
            o, n = res
            if o["fit"][0] != n["fit"][0]:
                bad.append("%s: fit exit %d -> %d" % (name, o["fit"][0], n["fit"][0]))
            same["fit-exit"] += o["fit"][0] == n["fit"][0]
            if o["model"] == n["model"]:
                same["model-file"] += 1
            elif strip_provenance(n["model"]) == o["model"]:
                changed["model-file"] += 1
            else:
                bad.append("%s: the model file differs by more than section 7" % name)
            for what in ("predict", "eval"):
                (r0, b0), (r1, b1) = o[what], n[what]
                if r0 != r1:
                    bad.append("%s %s: exit %d -> %d" % (name, what, r0, r1))
                elif b0 == b1:
                    same[what] += 1
                else:
                    try:
                        ok = normalise(json.loads(b0)) == normalise(json.loads(b1))
                    except Exception:  # noqa: BLE001
                        ok = False
                    if ok:
                        changed[what] += 1
                    else:
                        bad.append("%s %s: the answer differs beyond the new members" % (name, what))
    finally:
        for w in works:
            shutil.rmtree(w, ignore_errors=True)
    return bad, changed, same


def write():
    for name, script in (("golden_constant.json", "golden_constant.py"), ("golden_numeric.json", "golden_numeric.py"), ("golden_planned.json", "golden_planned.py")):
        out = subprocess.run([sys.executable, str(HERE / script)], capture_output=True, text=True, check=True).stdout
        (ROOT / "tests" / "conformance" / name).write_text(out)
        print("wrote", name)
    subprocess.run([sys.executable, str(HERE / "golden_describe.py")], check=True)
    print("wrote golden_describe.json")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--verify"]:
        sys.exit(verify(a[1], a[2] if len(a) > 2 else str(ROOT / "build" / "ml")))
    if a[:1] == ["--xplat"]:
        nd = a[2] if len(a) > 2 else str(ROOT / "build")
        problems, ch, sm = xplat(a[1], nd)
        for p in problems:
            print("UNEXPLAINED " + p)
        print("xplat: lines that change: %s; lines that stay: %s; unexplained: %d" % (ch, sm, len(problems)))
        sys.exit(1 if problems else 0)
    if a[:1] == ["--write"]:
        sys.exit(write())
    print(__doc__)
    sys.exit(2)

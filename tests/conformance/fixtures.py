"""The fixtures of the gates (docs/cli.md section 1): one list of cases, read by test_tables (every flag, rule and log record is
reached by one), test_rules (each case's first error is its rule, with the exit code of the tables, valid against the schema)
and the fuzz (cases are its seeds).

A case: id, setup (steps run in a scratch directory that is the --root), argv (after `ml --root DIR`; "@ROOT" is the
directory), expect {exit, rule, data (a subset the data must contain), stdout_starts, stderr_has}, and the switches
close_stdin, close_stdout, noroot. A step: ("write", name, text) ("fit", args...) ("mutate", src, dst, fn) ("mkdir", name)
("symlink", target, name) ("chmod", name, mode) ("raw", fn(sandbox)).
"""

import hashlib
import os

from harness import fix_checksum, sections, set_bytes

REG = "id,x,y\n1,2.5,10\n2,3.5,20\n3,1.0,30\n4,0.25,41\n"
CLS = "a,b,lab\n1,2,0\n2,3,1\n3,4,1\n4,5,1\n5,6,0\n"
FIT_REG = ["fit", "--target", "y", "--output", "m.cml", "reg.csv"]
FIT_CLS = ["fit", "--task", "classification", "--target", "lab", "--output", "c.cml", "cls.csv"]
BASE = [("write", "reg.csv", REG), ("write", "cls.csv", CLS)]
WITH_MODEL = BASE + [("fit", "--target", "y", "--output", "m.cml", "reg.csv")]
WITH_CLS = BASE + [("fit", "--task", "classification", "--target", "lab", "--output", "c.cml", "cls.csv")]


def ok(data=None, exit=0, **kw):
    d = {"exit": exit, "rule": None, "data": data}
    d.update(kw)
    return d


def bad(rule, exit=None, **kw):
    d = {"exit": exit, "rule": rule, "data": None}
    d.update(kw)
    return d


def case(id, argv, expect, setup=None, **kw):
    c = {"id": id, "argv": argv, "expect": expect, "setup": BASE if setup is None else setup}
    c.update(kw)
    return c


def flip_plan(d):
    b = bytearray(d)
    b[sections(d)[0][1] + 2] ^= 0x20
    return bytes(b)


def model_kind_7(d):
    return fix_checksum(set_bytes(d, sections(d)[3][1], (7).to_bytes(4, "little")))


CASES = [
    # ------------------------------------------------------------------ what works
    case("fit-regression", ["--model", "constant", "--task", "regression", "--drop", "id"] + FIT_REG,
         ok({"model": "constant", "task": "regression", "rows": 4, "columns": 2, "target": "y", "path": "m.cml", "written": True,
             "summary": {"mean": "25.250000"}})),
    case("fit-classification", FIT_CLS,
         ok({"task": "classification", "rows": 5, "columns": 3, "target": "lab",
             "summary": {"classes": [0, 1], "counts": [2, 3], "majority": 1, "probability": "0.600000"}})),
    case("fit-dry-run", ["--dry-run"] + FIT_REG,
         ok({"written": False}, exit=9, dry_run=True, no_file="m.cml")),
    case("fit-dry-run-replace", ["--dry-run", "--overwrite"] + FIT_REG,
         ok({"written": False}, exit=9, dry_run=True, setup_file_unchanged="m.cml"), setup=WITH_MODEL),
    case("fit-overwrite", ["--overwrite"] + FIT_REG, ok({"written": True}), setup=WITH_MODEL),
    case("fit-limits-and-log", ["--max-rows", "100", "--max-columns", "10", "--max-cells", "1000", "--max-line-bytes", "1000",
                                "--log", "lg.ndjson", "--log-max-bytes", "100000"] + FIT_REG, ok({"rows": 4}), log=True),
    case("fit-text", ["--format", "text"] + FIT_REG, ok(None, stdout_starts="fit: constant regression, 4 rows, wrote m.cml")),
    case("predict-page", ["predict", "--model-file", "m.cml", "--limit", "2", "--from", "1", "reg.csv"],
         ok({"columns": ["prediction"], "rows": [["25.250000"], ["25.250000"]], "row_count": 2, "total_rows": 4, "truncated": True,
             "next": {"from": 3}}), setup=WITH_MODEL),
    case("predict-last-page", ["predict", "--model-file", "m.cml", "--from", "3", "--max-bytes", "100000", "reg.csv"],
         ok({"row_count": 1, "truncated": False, "next": None}), setup=WITH_MODEL),
    case("predict-classification", ["predict", "--model-file", "c.cml", "--limit", "1", "cls.csv"],
         ok({"columns": ["label", "probability"], "rows": [["1", "0.600000"]]}), setup=WITH_CLS),
    case("predict-target-may-be-absent", ["predict", "--model-file", "m.cml", "x.csv"],
         ok({"row_count": 2}), setup=WITH_MODEL + [("write", "x.csv", "id,x\n1,2.5\n2,3.5\n")]),
    case("predict-drop-extra", ["predict", "--model-file", "m.cml", "--drop", "z", "z.csv"],
         ok({"row_count": 2}), setup=WITH_MODEL + [("write", "z.csv", "id,x,z\n1,2.5,hello\n2,3.5,world\n")]),
    case("predict-text", ["predict", "--model-file", "c.cml", "--format", "text", "cls.csv"],
         ok(None, stdout_starts="1\t0.600000\n"), setup=WITH_CLS),
    case("eval-regression", ["eval", "--model-file", "m.cml", "reg.csv"],
         ok({"rows": 4, "metrics": {"rmse": "11.519006", "sse": "530.750000000000", "prediction": "25.250000"}}), setup=WITH_MODEL),
    case("eval-classification", ["eval", "--model-file", "c.cml", "cls.csv"],
         ok({"metrics": {"accuracy": "0.600000", "correct": 3, "total": 5, "prediction": 1}}), setup=WITH_CLS),
    case("eval-classification-wide-labels", ["eval", "--model-file", "w.cml", "w.csv"],
         ok({"metrics": {"accuracy": "0.666667", "correct": 2, "total": 3, "prediction": 999999999999}}),
         setup=[("write", "w.csv", "f,lab\n1,999999999999\n1,-999999999999\n1,999999999999\n"),
                ("fit", "--task", "classification", "--target", "lab", "--output", "w.cml", "w.csv")]),
    case("eval-text", ["eval", "--model-file", "m.cml", "--format", "text", "--drop", "id", "reg.csv"],
         ok(None, stdout_starts="sse 530.750000000000\nrmse 11.519006\n"),
         setup=BASE + [("fit", "--drop", "id", "--target", "y", "--output", "m.cml", "reg.csv")]),
    # ------------------------------------------------------------------ arguments
    case("args-unknown-flag", ["fit", "--bogus", "1", "reg.csv"], bad("args.unknown-flag")),
    case("args-missing-value", ["fit", "reg.csv", "--target"], bad("args.missing-value")),
    case("args-bad-value", ["fit", "--max-rows", "many", "--target", "y", "--output", "m.cml", "reg.csv"], bad("args.bad-value")),
    case("args-bad-value-model", ["fit", "--model", "logreg", "--target", "y", "--output", "m.cml", "reg.csv"], bad("args.bad-value")),
    case("args-unexpected-value", ["fit", "--overwrite=1", "--target", "y", "--output", "m.cml", "reg.csv"], bad("args.unexpected-value")),
    case("args-duplicate-flag", ["fit", "--root", "@ROOT", "--target", "y", "--output", "m.cml", "reg.csv"], bad("args.duplicate-flag")),
    case("args-conflict", ["predict", "--target", "y", "--model-file", "m.cml", "reg.csv"], bad("args.conflict"), setup=WITH_MODEL),
    case("args-missing-operand", ["fit", "--target", "y", "--output", "m.cml"], bad("args.missing-operand")),
    case("args-missing-verb", [], bad("args.missing-operand")),
    case("args-required-flag", ["fit", "--target", "y", "reg.csv"], bad("args.required-flag")),
    case("args-too-many-operands", ["fit", "--target", "y", "--output", "m.cml", "reg.csv", "reg.csv"], bad("args.too-many-operands")),
    case("args-unknown-command", ["frobnicate", "reg.csv"], bad("args.unknown-command")),
    case("param-out-of-range", ["predict", "--model-file", "m.cml", "--limit", "0", "reg.csv"], bad("ml.param.out-of-range"),
         setup=WITH_MODEL),
    case("param-out-of-range-ceiling", ["fit", "--max-columns", "9000", "--target", "y", "--output", "m.cml", "reg.csv"],
         bad("ml.param.out-of-range")),
    # ------------------------------------------------------------------ paths
    case("path-empty", ["fit", "--target", "y", "--output", "m.cml", ""], bad("path.empty")),
    case("path-dotdot", ["fit", "--target", "y", "--output", "m.cml", "sub/../reg.csv"], bad("path.dotdot")),
    case("path-dotdot-output", ["fit", "--target", "y", "--output", "../m.cml", "reg.csv"], bad("path.dotdot")),
    case("path-dotdot-model", ["predict", "--model-file", "../m.cml", "reg.csv"], bad("path.dotdot")),
    case("path-dotdot-log", ["fit", "--log", "../lg", "--target", "y", "--output", "m.cml", "reg.csv"], bad("path.dotdot")),
    case("path-double-dot-in-a-name-without-root", ["fit", "--target", "y", "--output", "m.cml", "a..b.csv"], bad("path.dotdot"),
         noroot=True),
    case("path-double-dot-in-the-root", ["fit", "--root", "a..b", "--target", "y", "--output", "m.cml", "reg.csv"], bad("path.dotdot"),
         noroot=True),
    case("path-double-dot-in-a-name-under-the-root-is-a-name", ["fit", "--target", "y", "--output", "m.cml", "a..b.csv"],
         bad("io.not-found")),
    case("path-absolute", ["fit", "--target", "y", "--output", "m.cml", "@ROOT/reg.csv"], bad("path.absolute")),
    case("path-outside-root", ["fit", "--target", "y", "--output", "m.cml", "/etc/hosts"], bad("path.outside-root")),
    case("path-too-long", ["fit", "--target", "y", "--output", "m.cml", "a" * 5000], bad("path.too-long")),
    case("path-symlink", ["fit", "--target", "y", "--output", "m.cml", "link.csv"], bad("path.symlink"),
         setup=BASE + [("symlink", "reg.csv", "link.csv")]),
    case("path-symlink-dir", ["fit", "--target", "y", "--output", "m.cml", "dirlink/reg.csv"], bad("path.symlink"),
         setup=BASE + [("mkdir", "real"), ("symlink", "real", "dirlink")]),
    case("path-bad-name", ["fit", "--target", "y", "--output", ".", "reg.csv"], bad("path.bad-name")),
    case("path-flag-value-with-equals", ["fit", "--target", "y", "--output=m.cml", "reg.csv"], bad("args.bad-value")),
    case("io-not-found", ["fit", "--target", "y", "--output", "m.cml", "missing.csv"], bad("io.not-found")),
    case("io-not-found-model", ["predict", "--model-file", "missing.cml", "reg.csv"], bad("io.not-found")),
    case("io-not-a-directory", ["fit", "--target", "y", "--output", "m.cml", "reg.csv/x"], bad("io.not-a-directory")),
    case("io-is-a-directory", ["fit", "--target", "y", "--output", "m.cml", "sub"], bad("io.is-a-directory"),
         setup=BASE + [("mkdir", "sub")]),
    case("io-permission-denied", ["fit", "--target", "y", "--output", "m.cml", "secret.csv"], bad("io.permission-denied"),
         setup=BASE + [("write", "secret.csv", REG), ("chmod", "secret.csv", 0)], skip_if_root=True),
    case("io-read-failed", ["fit", "--target", "y", "--output", "m.cml", "/dev/stdin"], bad("io.read-failed"), close_stdin=True,
         noroot=True),
    case("io-write-failed-fit", FIT_REG, bad("io.write-failed", exit=1, stderr_has="io.write-failed"), close_stdout=True,
         no_json=True),
    case("io-write-failed-output-directory", ["fit", "--target", "y", "--output", "nodir/m.cml", "reg.csv"], bad("io.not-found")),
    case("conflict-exists", FIT_REG, bad("conflict.exists"), setup=WITH_MODEL),
    # ------------------------------------------------------------------ limits
    case("limit-line-too-long", ["fit", "--max-line-bytes", "5", "--target", "y", "--output", "m.cml", "reg.csv"],
         bad("limit.line-too-long")),
    case("limit-input-too-large", ["predict", "--model-file", "m.cml", "--max-bytes", "10", "reg.csv"], bad("limit.input-too-large"),
         setup=WITH_MODEL),
    case("limit-log-full", ["fit", "--log", "lg.ndjson", "--log-max-bytes", "10", "--target", "y", "--output", "m.cml", "reg.csv"],
         bad("limit.log-full"), setup=BASE + [("write", "lg.ndjson", "x" * 100)], no_file="m.cml"),
    case("ml-input-too-many-rows", ["fit", "--max-rows", "2", "--target", "y", "--output", "m.cml", "reg.csv"],
         bad("ml.input.too-many-rows")),
    case("ml-input-too-many-columns", ["fit", "--max-columns", "2", "--target", "y", "--output", "m.cml", "reg.csv"],
         bad("ml.input.too-many-columns")),
    case("ml-input-too-many-cells", ["fit", "--max-cells", "5", "--target", "y", "--output", "m.cml", "reg.csv"],
         bad("ml.input.too-many-cells")),
    # ------------------------------------------------------------------ the data
    case("ml-input-empty", FIT_REG, bad("ml.input.empty"), setup=[("write", "reg.csv", "id,x,y\n")]),
    case("ml-input-empty-file", FIT_REG, bad("ml.input.empty"), setup=[("write", "reg.csv", "")]),
    case("ml-input-non-numeric", FIT_REG, bad("ml.input.non-numeric", repair_retry=True),
         setup=[("write", "reg.csv", "id,x,y\n1,abc,10\n2,3,20\n")]),
    case("ml-input-non-numeric-empty-cell", FIT_REG, bad("ml.input.non-numeric"),
         setup=[("write", "reg.csv", "id,x,y\n1,,10\n2,3,20\n")]),
    case("ml-input-non-finite", FIT_REG, bad("ml.input.non-finite"), setup=[("write", "reg.csv", "id,x,y\n1,NaN,10\n2,3,20\n")]),
    case("ml-input-non-finite-inf", FIT_REG, bad("ml.input.non-finite"), setup=[("write", "reg.csv", "id,x,y\n1,-inf,10\n")]),
    case("ml-input-unsupported-type-exponent", FIT_REG, bad("ml.input.unsupported-type"),
         setup=[("write", "reg.csv", "id,x,y\n1,1e5,10\n")]),
    case("ml-input-unsupported-type-wide", FIT_REG, bad("ml.input.unsupported-type"),
         setup=[("write", "reg.csv", "id,x,y\n1,0.1234567890123,10\n")]),
    case("ml-input-unsupported-type-column", FIT_REG, bad("ml.input.unsupported-type"),
         setup=[("write", "reg.csv", "id,x,y\n1,123456789012,10\n2,0.5,20\n")]),
    case("ml-input-unsupported-csv", FIT_REG, bad("ml.input.unsupported-csv"),
         setup=[("write", "reg.csv", 'id,x,y\n1,"2",10\n')]),
    case("ml-input-duplicate-column", FIT_REG, bad("ml.input.duplicate-column"), setup=[("write", "reg.csv", "id,x,x,y\n1,2,3,4\n")]),
    case("ml-input-empty-column-name", FIT_REG, bad("ml.input.duplicate-column"), setup=[("write", "reg.csv", "id,,y\n1,2,3\n")]),
    case("ml-input-unknown-column", ["fit", "--target", "nope", "--output", "m.cml", "reg.csv"], bad("ml.input.unknown-column")),
    case("ml-input-unknown-drop", ["fit", "--drop", "id,nope", "--target", "y", "--output", "m.cml", "reg.csv"],
         bad("ml.input.unknown-column")),
    case("args-conflict-target-dropped", ["fit", "--drop", "y", "--target", "y", "--output", "m.cml", "reg.csv"],
         bad("args.conflict")),
    case("parse-csv-ragged-row", FIT_REG, bad("parse.csv-ragged-row"), setup=[("write", "reg.csv", "id,x,y\n1,2,3\n4,5\n")]),
    case("ml-label-single-class", FIT_CLS, bad("ml.label.single-class"), setup=[("write", "cls.csv", "a,lab\n1,1\n2,1\n")]),
    case("ml-label-not-binary", FIT_CLS, bad("ml.label.not-binary"), setup=[("write", "cls.csv", "a,lab\n1,1\n2,2\n3,3\n")]),
    case("ml-label-not-integer", FIT_CLS, bad("ml.label.not-integer"), setup=[("write", "cls.csv", "a,lab\n1,1.5\n2,2\n")]),
    case("ml-schema-mismatch", ["predict", "--model-file", "m.cml", "other.csv"], bad("ml.schema.mismatch"),
         setup=WITH_MODEL + [("write", "other.csv", "p,q\n1,2\n")]),
    case("ml-schema-mismatch-extra-retry", ["predict", "--model-file", "m.cml", "z.csv"], bad("ml.schema.mismatch", repair_retry=True),
         setup=WITH_MODEL + [("write", "z.csv", "id,x,z\n1,2.5,9\n")]),
    case("ml-schema-mismatch-type", ["predict", "--model-file", "m.cml", "t.csv"], bad("ml.schema.mismatch"),
         setup=WITH_MODEL + [("write", "t.csv", "id,x\n1.5,2.5\n2,3.125\n")]),
    case("ml-schema-mismatch-eval-needs-target", ["eval", "--model-file", "m.cml", "x.csv"], bad("ml.schema.mismatch"),
         setup=WITH_MODEL + [("write", "x.csv", "id,x\n1,2.5\n")]),
    # ------------------------------------------------------------------ the model file
    case("model-bad-magic", ["predict", "--model-file", "bad.cml", "reg.csv"], bad("ml.model.bad-magic"),
         setup=WITH_MODEL + [("mutate", "m.cml", "bad.cml", lambda d: b"XXXX" + d[4:])]),
    case("model-version", ["predict", "--model-file", "bad.cml", "reg.csv"], bad("ml.model.version"),
         setup=WITH_MODEL + [("mutate", "m.cml", "bad.cml", lambda d: set_bytes(d, 4, (2).to_bytes(4, "little")))]),
    case("model-truncated", ["predict", "--model-file", "bad.cml", "reg.csv"], bad("ml.model.truncated"),
         setup=WITH_MODEL + [("mutate", "m.cml", "bad.cml", lambda d: d[:len(d) // 2])]),
    case("model-truncated-empty", ["eval", "--model-file", "bad.cml", "reg.csv"], bad("ml.model.truncated"),
         setup=WITH_MODEL + [("mutate", "m.cml", "bad.cml", lambda d: b"")]),
    case("model-checksum", ["predict", "--model-file", "bad.cml", "reg.csv"], bad("ml.model.checksum"),
         setup=WITH_MODEL + [("mutate", "m.cml", "bad.cml", flip_plan)]),
    case("model-malformed", ["predict", "--model-file", "bad.cml", "reg.csv"], bad("ml.model.malformed"),
         setup=WITH_MODEL + [("mutate", "m.cml", "bad.cml", model_kind_7)]),
]


def log_cases():
    return [c for c in CASES if c.get("log")]

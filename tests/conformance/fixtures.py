"""The fixtures of the gates (docs/cli.md section 1): one list of cases, read by test_tables (every flag, rule and log record is
reached by one), test_rules (each case's first error is its rule, with the exit code of the tables, valid against the schema)
and the fuzz (cases are its seeds).

A case: id, setup (steps run in a scratch directory that is the --root), argv (after `ml --root DIR`; "@ROOT" is the
directory), expect {exit, rule, data (a subset the data must contain), stdout_starts, stderr_has}, and the switches
close_stdin, close_stdout, noroot, darwin_only (a platform-specific answer: harness.runs_here). A step: ("write", name, text) ("fit", args...) ("mutate", src, dst, fn) ("mkdir", name)
("symlink", target, name) ("chmod", name, mode) ("raw", fn(sandbox)).
"""

import hashlib
import os

from harness import fix_checksum, sections, set_bytes, without_provenance

REG = "id,x,y\n1,2.5,10\n2,3.5,20\n3,1.0,30\n4,0.25,41\n"
CLS = "a,b,lab\n1,2,0\n2,3,1\n3,4,1\n4,5,1\n5,6,0\n"
FLT = "id,x,y\n1,2.5,1e5\n2,3.5e0,0.1\n3,.25,2.75\n"
FIT_REG = ["fit", "--target", "y", "--output", "m.cml", "reg.csv"]
FIT_FLT = ["fit", "--float", "id", "--target", "y", "--output", "f.cml", "flt.csv"]
FIT_CLS = ["fit", "--task", "classification", "--target", "lab", "--output", "c.cml", "cls.csv"]
BASE = [("write", "reg.csv", REG), ("write", "cls.csv", CLS)]
WITH_MODEL = BASE + [("fit", "--target", "y", "--output", "m.cml", "reg.csv")]
WITH_CLS = BASE + [("fit", "--task", "classification", "--target", "lab", "--output", "c.cml", "cls.csv")]
WITH_FLT = BASE + [("write", "flt.csv", FLT), ("fit", "--float", "id", "--target", "y", "--output", "f.cml", "flt.csv")]


# ------------------------------------------------------------------ the models that go through the pipeline (docs/wiring.md)
LIN = "a,b,y\n1,2,5.1\n2,1,7.9\n3,4,9.3\n4,5,12.2\n5,6,13.8\n6,3,15.1\n7,9,19.7\n8,2,17.1\n"
LINMIS = "a,b,y\n1,2,5.1\n2,,7.9\n3,4,9.3\n4,5,12.2\n5,6,13.8\n6,3,15.1\n7,9,19.7\n8,2,17.1\n"
LCLS = "a,b,lab\n1,2,0\n2,1,0\n3,4,1\n4,1,1\n5,6,0\n6,3,1\n7,9,1\n8,2,0\n9,7,1\n10,1,0\n3,3,0\n6,6,1\n"
CAT = "c,y\n1,2.0\n2,3.1\n3,3.9\n1,2.2\n2,2.9\n3,4.1\n"
STEP_IMP = "impute column=b strategy=mean; scale column=* kind=standard"
FIT_LIN = ["fit", "--model", "linear", "--target", "y", "--output", "lin.cml", "lin.csv"]
FIT_LOG = ["fit", "--model", "logistic", "--target", "lab", "--output", "log.cml", "lcls.csv"]
LBASE = BASE + [("write", "lin.csv", LIN), ("write", "linmis.csv", LINMIS), ("write", "lcls.csv", LCLS)]
WITH_LIN = LBASE + [("fit", "--model", "linear", "--target", "y", "--output", "lin.cml", "lin.csv")]
WITH_LINMIS = LBASE + [("fit", "--model", "linear", "--target", "y", "--step", STEP_IMP, "--output", "lm.cml", "linmis.csv")]
WITH_LOG = LBASE + [("fit", "--model", "logistic", "--target", "lab", "--output", "log.cml", "--step", "scale column=* kind=standard", "lcls.csv")]


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
         ok(None, stdout_starts="on train\nsse 530.750000000000\nrmse 11.519006\n"),
         setup=BASE + [("fit", "--drop", "id", "--target", "y", "--output", "m.cml", "reg.csv")]),
    case("fit-float-regression", FIT_FLT,
         ok({"model": "constant", "task": "regression", "rows": 3, "columns": 3, "target": "y", "path": "f.cml", "written": True,
             "summary": {"mean": "33334.28333333333"}}), setup=BASE + [("write", "flt.csv", FLT)]),
    case("predict-float", ["predict", "--model-file", "f.cml", "--limit", "2", "flt.csv"],
         ok({"columns": ["prediction"], "rows": [["33334.28333333333"], ["33334.28333333333"]], "row_count": 2, "total_rows": 3,
             "truncated": True}), setup=WITH_FLT),
    case("predict-float-model-takes-an-int-column", ["predict", "--model-file", "f.cml", "i.csv"], ok({"row_count": 2}),
         setup=WITH_FLT + [("write", "i.csv", "id,x\n1,2\n2,3\n")]),
    case("eval-float", ["eval", "--model-file", "f.cml", "flt.csv"],
         ok({"rows": 3, "metrics": {"rmse": "47139.78034007536", "sse": "6666476671.531667", "prediction": "33334.28333333333"}}),
         setup=WITH_FLT),
    case("eval-float-text", ["eval", "--model-file", "f.cml", "--format", "text", "flt.csv"],
         ok(None, stdout_starts="on train\nsse 6666476671.531667\nrmse 47139.78034007536\n"), setup=WITH_FLT),
    case("eval-float-model-on-int-data", ["eval", "--model-file", "f.cml", "i.csv"],
         ok({"metrics": {"rmse": "47139.78034007536"}}), setup=WITH_FLT + [("write", "i.csv", "id,x,y\n1,2.5,100000\n2,3.5,0.1\n3,0.25,2.75\n")]),
    case("fit-quoted-and-crlf", ["fit", "--drop", "t", "--target", "y", "--output", "q.cml", "q.csv"],
         ok({"rows": 2, "columns": 2, "summary": {"mean": "3.0"}}),
         setup=BASE + [("write", "q.csv", '\ufeffid,t,y\r\n"1","a,b\nc ""d""","1.5e0"\r\n\r\n2,"x",4.5\r\n')]),
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
         setup=BASE + [("symlink", "reg.csv", "link.csv")], darwin_only=True),
    case("path-symlink-dir", ["fit", "--target", "y", "--output", "m.cml", "dirlink/reg.csv"], bad("path.symlink"),
         setup=BASE + [("mkdir", "real"), ("symlink", "real", "dirlink")], darwin_only=True),
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
         noroot=True, darwin_only=True),
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
    case("ml-input-non-finite-range", FIT_REG, bad("ml.input.non-finite"), setup=[("write", "reg.csv", "id,x,y\n1,1e999,10\n")]),
    case("ml-input-non-numeric-underflow", FIT_REG, bad("ml.input.non-numeric"), setup=[("write", "reg.csv", "id,x,y\n1,1e-999,10\n")]),
    case("ml-input-non-numeric-hex", FIT_REG, bad("ml.input.non-numeric"), setup=[("write", "reg.csv", "id,x,y\n1,0x10,10\n")]),
    case("parse-csv-bad-quote", FIT_REG, bad("parse.csv-bad-quote"), setup=[("write", "reg.csv", 'id,x,y\n1,"2"x,10\n')]),
    case("parse-csv-unterminated-quote", FIT_REG, bad("parse.csv-unterminated-quote"), setup=[("write", "reg.csv", 'id,x,y\n1,"2,10\n')]),
    case("ml-metric-undefined", ["eval", "--model-file", "big.cml", "big.csv"], bad("ml.metric.undefined"),
         setup=BASE + [("write", "big.csv", "y\n1e200\n-1e200\n"), ("fit", "--target", "y", "--output", "big.cml", "big.csv")]),
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
    # ------------------------------------------------------------------ linear and logistic through the pipeline
    case("fit-linear", FIT_LIN, ok({"model": "linear", "task": "regression", "rows": 8, "columns": 3, "target": "y", "path": "lin.cml", "written": True,
                                    "summary": {"plan": "model kind=linear l2=0.0\n", "steps": 0, "features": 2}}), setup=LBASE),
    case("fit-linear-steps-and-l2", ["fit", "--model", "linear", "--l2", "0.5", "--step", STEP_IMP, "--target", "y", "--output", "lm.cml", "linmis.csv"],
         ok({"model": "linear", "rows": 8, "summary": {"plan": "impute column=b strategy=mean\nscale column=* kind=standard ddof=0\nmodel kind=linear l2=0.5\n",
                                                        "steps": 2, "features": 2}}), setup=LBASE),
    case("fit-linear-plan-file", ["fit", "--plan", "p.txt", "--target", "y", "--output", "lm.cml", "linmis.csv"],
         ok({"model": "linear", "summary": {"steps": 1, "plan": "impute column=* strategy=median\nmodel kind=linear l2=0.0\n"}}),
         setup=LBASE + [("write", "p.txt", "# a plan\nimpute column=*   strategy=median\nmodel kind=linear\n")]),
    case("fit-linear-split-seed-threads", ["fit", "--model", "linear", "--split", "train=0.75", "--seed", "7", "--threads", "3", "--target", "y",
                                           "--output", "lin.cml", "lin.csv"],
         ok({"rows": 6, "summary": {"split": {"train": 6, "test": 2, "seed": 7}, "holdout": {"rows": 2}}}), setup=LBASE),
    case("fit-logistic", ["fit", "--model", "logistic", "--iterations", "50", "--l2", "0.1", "--positive", "0", "--step", "scale column=* kind=minmax",
                          "--target", "lab", "--output", "log.cml", "lcls.csv"],
         ok({"model": "logistic", "task": "classification", "rows": 12, "summary": {"classes": [1, 0], "positive": 0, "steps": 1}}), setup=LBASE),
    case("fit-logistic-default-positive", FIT_LOG, ok({"summary": {"classes": [0, 1], "positive": 1}}), setup=LBASE),
    case("fit-logistic-split-holdout", ["fit", "--model", "logistic", "--split", "train=0.7", "--seed", "1", "--l2", "0.01", "--target", "lab", "--output", "log.cml",
                                        "lcls.csv"], ok({"summary": {"split": {"train": 8, "test": 4, "seed": 1}}}), setup=LBASE),
    case("fit-linear-dry-run", ["--dry-run"] + FIT_LIN, ok({"written": False}, exit=9, dry_run=True, no_file="lin.cml"), setup=LBASE),
    case("fit-linear-text", ["--format", "text"] + FIT_LIN, ok(None, stdout_starts="fit: linear regression, 8 rows, wrote lin.cml"), setup=LBASE),
    case("predict-linear", ["predict", "--model-file", "lin.cml", "--limit", "2", "--from", "1", "lin.csv"],
         ok({"model": "linear", "columns": ["prediction"], "row_count": 2, "total_rows": 8, "truncated": True, "next": {"from": 3}, "unknown_categories": 0}),
         setup=WITH_LIN),
    case("predict-linear-reordered-and-without-the-target", ["predict", "--model-file", "lin.cml", "x.csv"], ok({"row_count": 2}),
         setup=WITH_LIN + [("write", "x.csv", "b,a\n2,1\n1,2\n")]),
    case("predict-linear-with-steps", ["predict", "--model-file", "lm.cml", "linmis.csv"], ok({"row_count": 8}), setup=WITH_LINMIS),
    case("predict-logistic", ["predict", "--model-file", "log.cml", "--limit", "3", "lcls.csv"], ok({"columns": ["label"], "row_count": 3}), setup=WITH_LOG),
    case("predict-logistic-proba", ["predict", "--model-file", "log.cml", "--proba", "--limit", "3", "lcls.csv"],
         ok({"model": "logistic", "task": "classification", "columns": ["label", "probability"], "row_count": 3}), setup=WITH_LOG),
    case("predict-logistic-text", ["predict", "--model-file", "log.cml", "--proba", "--format", "text", "--limit", "1", "lcls.csv"],
         ok(None, stdout_starts="0\t0."), setup=WITH_LOG),
    case("eval-linear", ["eval", "--model-file", "lin.cml", "lin.csv"], ok({"model": "linear", "task": "regression", "rows": 8, "unknown_categories": 0}),
         setup=WITH_LIN),
    case("eval-linear-split", ["eval", "--model-file", "lin.cml", "--split", "train=0.75", "--seed", "7", "--threads", "2", "lin.csv"],
         ok({"rows": 2, "split": {"train": 6, "test": 2, "seed": 7}}), setup=WITH_LIN),
    case("eval-linear-text", ["eval", "--model-file", "lin.cml", "--format", "text", "lin.csv"], ok(None, stdout_starts="on train\nrmse "), setup=WITH_LIN),
    case("eval-logistic", ["eval", "--model-file", "log.cml", "lcls.csv"], ok({"model": "logistic", "rows": 12, "positive": 1}), setup=WITH_LOG),
    case("eval-logistic-other-positive-label", ["eval", "--model-file", "log.cml", "--positive", "0", "lcls.csv"], ok({"positive": 0}), setup=WITH_LOG),
    # ------------------------------------------------------------------ their refusals
    case("ml-input-missing-value", FIT_LIN[:-1] + ["linmis.csv"], bad("ml.input.missing-value", repair_retry=True), setup=LBASE),
    case("ml-input-missing-value-in-a-step-plan", ["fit", "--model", "linear", "--step", "scale column=a kind=minmax", "--target", "y", "--output", "m.cml", "linmis.csv"],
         bad("ml.input.missing-value"), setup=LBASE),
    case("ml-input-missing-value-in-predict", ["predict", "--model-file", "lin.cml", "linmis.csv"], bad("ml.input.missing-value"), setup=WITH_LIN),
    case("ml-plan-syntax", ["fit", "--model", "linear", "--step", "impute column", "--target", "y", "--output", "m.cml", "lin.csv"], bad("ml.plan.syntax"), setup=LBASE),
    case("ml-plan-unknown-step", ["fit", "--model", "linear", "--step", "frobnicate column=a", "--target", "y", "--output", "m.cml", "lin.csv"],
         bad("ml.plan.unknown-step"), setup=LBASE),
    case("ml-plan-unknown-key", ["fit", "--model", "linear", "--step", "scale column=a kind=minmax speed=3", "--target", "y", "--output", "m.cml", "lin.csv"],
         bad("ml.plan.unknown-key"), setup=LBASE),
    case("ml-plan-missing-key", ["fit", "--model", "linear", "--step", "scale column=a", "--target", "y", "--output", "m.cml", "lin.csv"],
         bad("ml.plan.missing-key"), setup=LBASE),
    case("ml-plan-bad-value", ["fit", "--model", "linear", "--step", "scale column=a kind=cubic", "--target", "y", "--output", "m.cml", "lin.csv"],
         bad("ml.plan.bad-value"), setup=LBASE),
    case("ml-plan-duplicate-key", ["fit", "--model", "linear", "--step", "scale column=a kind=minmax kind=standard", "--target", "y", "--output", "m.cml", "lin.csv"],
         bad("ml.plan.duplicate-key"), setup=LBASE),
    case("ml-plan-model-line", ["fit", "--model", "linear", "--step", "model kind=linear", "--target", "y", "--output", "m.cml", "lin.csv"],
         bad("ml.plan.model-line"), setup=LBASE),
    case("ml-plan-model-line-missing-in-a-plan-file", ["fit", "--plan", "p.txt", "--target", "y", "--output", "m.cml", "lin.csv"], bad("ml.plan.model-line"),
         setup=LBASE + [("write", "p.txt", "scale column=a kind=minmax\n")]),
    case("ml-plan-unknown-column", ["fit", "--model", "linear", "--step", "scale column=zzz kind=minmax", "--target", "y", "--output", "m.cml", "lin.csv"],
         bad("ml.plan.unknown-column"), setup=LBASE),
    case("ml-limit-plan", ["fit", "--plan", "p.txt", "--target", "y", "--output", "m.cml", "lin.csv"], bad("ml.limit.plan"),
         setup=LBASE + [("write", "p.txt", "# " + "x" * 70000 + "\nmodel kind=linear\n")]),
    case("ml-plan-model-not-built", ["fit", "--plan", "p.txt", "--target", "y", "--output", "m.cml", "lin.csv"], bad("ml.plan.model-not-built"),
         setup=LBASE + [("write", "p.txt", "model kind=constant task=regression\n")]),
    case("ml-fit-all-missing", ["fit", "--model", "linear", "--step", "impute column=* strategy=mean", "--target", "y", "--output", "m.cml", "am.csv"],
         bad("ml.fit.all-missing"), setup=LBASE + [("write", "am.csv", "a,b,y\n1,,3\n2,,4\n3,,5\n")]),
    case("ml-limit-categories", ["fit", "--model", "linear", "--step", "encode column=a kind=onehot max-categories=3", "--target", "y", "--output", "m.cml", "lin.csv"],
         bad("ml.limit.categories"), setup=LBASE),
    case("ml-input-unknown-category", ["predict", "--model-file", "cat.cml", "newcat.csv"], bad("ml.input.unknown-category"),
         setup=LBASE + [("write", "cat.csv", CAT), ("write", "newcat.csv", "c\n1\n9\n"),
                        ("fit", "--model", "linear", "--l2", "0.1", "--step", "encode column=c kind=onehot unknown=refuse", "--target", "y", "--output", "cat.cml", "cat.csv")]),
    # issue 48: two equal columns with l2 = 0 are named now (ml.fit.collinear); the id stays, because the baseline record is keyed by it (test_cat_baseline CHANGED)
    case("ml-linalg-not-positive-definite", ["fit", "--model", "linear", "--target", "y", "--output", "m.cml", "dup.csv"], bad("ml.fit.collinear", repair_runs=True),
         setup=LBASE + [("write", "dup.csv", "a,b,y\n1,1,0\n2,2,1\n3,3,0\n4,4,1\n5,5,1\n")]),
    case("ml-fit-non-finite", ["fit", "--model", "linear", "--target", "y", "--output", "m.cml", "big.csv"], bad("ml.fit.non-finite"),
         setup=LBASE + [("write", "big.csv", "x,y\n1e200,1\n2e200,0\n3e200,1\n-1e200,0\n")]),
    case("ml-predict-non-finite", ["predict", "--model-file", "lf.cml", "bigx.csv"], bad("ml.predict.non-finite"),
         setup=LBASE + [("fit", "--model", "linear", "--float", "a,b,y", "--target", "y", "--output", "lf.cml", "lin.csv"), ("write", "bigx.csv", "a,b\n1,2\n1.7e308,2\n")]),
    case("ml-eval-non-finite-distance", ["eval", "--model-file", "k.cml", "kb.csv"], bad("ml.predict.non-finite"),
         setup=BASE + [("write", "km.csv", "a,b\n1,1\n1.1,0.9\n5,5\n5.1,4.9\n"), ("fit", "--model", "kmeans", "--k", "2", "--float", "a,b", "--output", "k.cml", "km.csv"),
                       ("write", "kb.csv", "a,b\n1,1\n1e200,1\n")]),
    case("ml-split-empty-fold", ["fit", "--model", "linear", "--split", "train=0.01", "--target", "y", "--output", "m.cml", "lin.csv"], bad("ml.split.empty-fold"), setup=LBASE),
    case("ml-param-threads", ["fit", "--model", "linear", "--threads", "0", "--target", "y", "--output", "m.cml", "lin.csv"], bad("ml.param.threads"), setup=LBASE),
    case("ml-param-threads-too-many", ["eval", "--model-file", "lin.cml", "--threads", "65", "lin.csv"], bad("ml.param.threads"), setup=WITH_LIN),
    case("param-out-of-range-iterations", ["fit", "--model", "logistic", "--iterations", "0", "--target", "lab", "--output", "m.cml", "lcls.csv"],
         bad("ml.param.out-of-range"), setup=LBASE),
    case("param-out-of-range-positive-label", ["fit", "--model", "logistic", "--positive", "7", "--target", "lab", "--output", "m.cml", "lcls.csv"],
         bad("ml.param.out-of-range"), setup=LBASE),
    case("args-bad-value-l2", ["fit", "--model", "linear", "--l2", "-1", "--target", "y", "--output", "m.cml", "lin.csv"], bad("args.bad-value"), setup=LBASE),
    case("args-bad-value-split", ["fit", "--model", "linear", "--split", "0.8", "--target", "y", "--output", "m.cml", "lin.csv"], bad("args.bad-value"), setup=LBASE),
    case("args-bad-value-positive", ["fit", "--model", "logistic", "--positive", "yes", "--target", "lab", "--output", "m.cml", "lcls.csv"], bad("args.bad-value"),
         setup=LBASE),
    case("args-conflict-plan-and-model", ["fit", "--plan", "p.txt", "--model", "linear", "--target", "y", "--output", "m.cml", "lin.csv"], bad("args.conflict"),
         setup=LBASE + [("write", "p.txt", "model kind=linear\n")]),
    case("args-conflict-plan-and-step", ["fit", "--plan", "p.txt", "--step", "scale column=a kind=minmax", "--target", "y", "--output", "m.cml", "lin.csv"],
         bad("args.conflict"), setup=LBASE + [("write", "p.txt", "model kind=linear\n")]),
    case("args-conflict-step-needs-a-model", ["fit", "--step", "scale column=a kind=minmax", "--target", "y", "--output", "m.cml", "lin.csv"], bad("args.conflict"),
         setup=LBASE),
    case("args-conflict-hyperparameter-of-another-model", ["fit", "--model", "linear", "--iterations", "5", "--target", "y", "--output", "m.cml", "lin.csv"],
         bad("args.conflict"), setup=LBASE),
    case("args-conflict-task-of-the-other-kind", ["fit", "--model", "linear", "--task", "classification", "--target", "y", "--output", "m.cml", "lin.csv"],
         bad("args.conflict"), setup=LBASE),
    case("args-conflict-positive-for-a-regression", ["fit", "--model", "linear", "--positive", "1", "--target", "y", "--output", "m.cml", "lin.csv"],
         bad("args.conflict"), setup=LBASE),
    case("args-conflict-split-on-a-constant-model", ["eval", "--model-file", "m.cml", "--split", "train=0.5", "reg.csv"], bad("args.conflict"), setup=WITH_MODEL),
    case("args-conflict-proba-on-a-regression", ["predict", "--model-file", "lin.cml", "--proba", "lin.csv"], bad("args.conflict"), setup=WITH_LIN),
    case("args-conflict-plan-flag-in-predict", ["predict", "--model-file", "lin.cml", "--plan", "p.txt", "lin.csv"], bad("args.conflict"), setup=WITH_LIN),
    case("ml-param-threads-zero-in-eval", ["eval", "--model-file", "lin.cml", "--threads", "0", "lin.csv"], bad("ml.param.threads"), setup=WITH_LIN),
    case("ml-input-non-numeric-empty-target-in-a-pipeline-fit", ["fit", "--model", "linear", "--target", "y", "--output", "m.cml", "et.csv"],
         bad("ml.input.non-numeric"), setup=LBASE + [("write", "et.csv", "a,b,y\n1,2,3\n2,3,\n3,5,9\n4,1,2\n")]),
    case("predict-linear-ignores-what-the-target-column-holds", ["predict", "--model-file", "lin.cml", "t.csv"], ok({"row_count": 2}),
         setup=WITH_LIN + [("write", "t.csv", "a,b,y\n1,2,n/a\n2,1,\"x,y\"\n")]),
    case("ml-label-not-binary-eval-label-the-model-never-saw", ["eval", "--model-file", "log.cml", "other.csv"], bad("ml.label.not-binary"),
         setup=WITH_LOG + [("write", "other.csv", "a,b,lab\n1,2,0\n2,3,5\n")]),
    case("ml-schema-mismatch-in-a-pipeline-model", ["predict", "--model-file", "lin.cml", "other.csv"], bad("ml.schema.mismatch"),
         setup=WITH_LIN + [("write", "other.csv", "a,z\n1,2\n")]),
    case("ml-label-single-class-in-a-held-out-side", ["fit", "--model", "logistic", "--split", "train=0.5", "--seed", "0", "--target", "lab", "--output", "m.cml", "one.csv"],
         bad("ml.label.single-class"),
         setup=LBASE + [("write", "one.csv", "a,lab\n1,0\n2,0\n3,0\n4,0\n5,1\n6,0\n7,0\n8,0\n9,0\n10,0\n")]),
    case("ml-model-malformed-pipeline-model", ["predict", "--model-file", "bad.cml", "lin.csv"], bad("ml.model.malformed"),
         setup=WITH_LIN + [("mutate", "lin.cml", "bad.cml", lambda d: fix_checksum(set_bytes(d, sections(d)[3][1] + 4, (0).to_bytes(4, "little"))))]),
    case("ml-model-version-pipeline-model-sum-order", ["predict", "--model-file", "bad.cml", "lin.csv"], bad("ml.model.version"),
         setup=WITH_LIN + [("mutate", "lin.cml", "bad.cml", lambda d: fix_checksum(set_bytes(d, sections(d)[4][1] + sections(d)[4][2] - 4, (1).to_bytes(4, "little"))))]),
]


# ------------------------------------------------------------------ categorical columns (docs/categorical.md): stage 1 reads and checks them and then refuses
TXT = "id,sex,port,y\n1,male,S,2.5\n2,female,C,3\n3,male,,4\n4,\"fe\"\"male\",Q,5\n5,NA,S,6\n"
FIT_TXT = ["fit", "--target", "y", "--output", "m.cml", "--cat", "sex,port", "txt.csv"]
TBASE = BASE + [("write", "txt.csv", TXT)]


def _cat_cases():
    bad_utf8 = ("write", "u.csv", b"id,sex,y\n1,male,2\n2,caf\xe9,3\n")
    return [
        case("ml-input-categorical-not-encoded", FIT_TXT, bad("ml.input.categorical-not-encoded", data=None), setup=TBASE),
        case("ml-input-categorical-not-encoded-with-markers", ["--format", "json"] + FIT_TXT[:-1] + ["--na-values", "NA,?", "txt.csv"],
             bad("ml.input.categorical-not-encoded"), setup=TBASE),
        case("ml-limit-categories-of-a-categorical-column", ["fit", "--target", "y", "--output", "m.cml", "--cat", "sex,port", "--max-categories", "2", "txt.csv"],
             bad("ml.limit.categories"), setup=TBASE),
        case("ml-limit-string-bytes", ["fit", "--target", "y", "--output", "m.cml", "--cat", "sex,port", "--max-string-bytes", "5", "txt.csv"],
             bad("ml.limit.string-bytes"), setup=TBASE),
        case("ml-limit-string-pool", ["fit", "--target", "y", "--output", "m.cml", "--cat", "sex,port", "--max-pool-bytes", "20", "txt.csv"],
             bad("ml.limit.string-pool"), setup=TBASE),
        case("ml-limit-categorical-columns", ["fit", "--target", "y", "--output", "m.cml", "--cat", "sex,port", "--max-cat-columns", "1", "txt.csv"],
             bad("ml.limit.categorical-columns"), setup=TBASE),
        case("ml-input-invalid-utf8", ["fit", "--target", "y", "--output", "m.cml", "--cat", "sex", "u.csv"], bad("ml.input.invalid-utf8"), setup=BASE + [bad_utf8]),
        case("ml-input-unknown-column-of-cat", ["fit", "--target", "y", "--output", "m.cml", "--cat", "sex,nope", "txt.csv"], bad("ml.input.unknown-column"), setup=TBASE),
        case("args-conflict-cat-on-the-target", ["fit", "--target", "y", "--output", "m.cml", "--cat", "y", "txt.csv"], bad("args.conflict"), setup=TBASE),
        case("args-conflict-cat-on-a-dropped-column", ["fit", "--target", "y", "--output", "m.cml", "--cat", "sex", "--drop", "sex", "txt.csv"], bad("args.conflict"), setup=TBASE),
        case("args-conflict-cat-on-a-float-column", ["fit", "--target", "y", "--output", "m.cml", "--cat", "id", "--float", "id", "txt.csv"], bad("args.conflict"), setup=TBASE),
        case("args-conflict-na-values-without-cat", ["fit", "--target", "y", "--output", "m.cml", "--na-values", "NA", "txt.csv"], bad("args.conflict"), setup=TBASE),
        case("args-conflict-max-categories-without-cat", ["fit", "--target", "y", "--output", "m.cml", "--max-categories", "5", "txt.csv"], bad("args.conflict"), setup=TBASE),
        case("args-conflict-max-string-bytes-without-cat", ["fit", "--target", "y", "--output", "m.cml", "--max-string-bytes", "5", "txt.csv"], bad("args.conflict"), setup=TBASE),
        case("args-conflict-max-pool-bytes-without-cat", ["fit", "--target", "y", "--output", "m.cml", "--max-pool-bytes", "5", "txt.csv"], bad("args.conflict"), setup=TBASE),
        case("args-conflict-max-cat-columns-without-cat", ["fit", "--target", "y", "--output", "m.cml", "--max-cat-columns", "5", "txt.csv"], bad("args.conflict"), setup=TBASE),
        case("args-bad-value-cat-empty", ["fit", "--target", "y", "--output", "m.cml", "--cat", "", "txt.csv"], bad("args.bad-value"), setup=TBASE),
        case("args-bad-value-na-values-empty-marker", ["fit", "--target", "y", "--output", "m.cml", "--cat", "sex", "--na-values", "NA,,?", "txt.csv"], bad("args.bad-value"), setup=TBASE),
        case("ml-param-out-of-range-max-categories", ["fit", "--target", "y", "--output", "m.cml", "--cat", "sex", "--max-categories", "1025", "txt.csv"], bad("ml.param.out-of-range"), setup=TBASE),
        case("args-conflict-cat-in-predict", ["predict", "--model-file", "m.cml", "--cat", "sex", "reg.csv"], bad("args.conflict"), setup=WITH_MODEL),
    ]


CASES += _cat_cases()


# ------------------------------------------------------------------ the encoders of categorical columns (docs/categorical-encoders.md, stage 2 of issue 21)
ENCT = "id,sex,port,age,y\n1,male,S,20,1\n2,female,C,30,0\n3,male,S,40,1\n4,female,Q,50,0\n5,male,S,31,1\n6,female,C,22,0\n7,__other__,S,35,1\n8,male,C,44,0\n"
EBASE = BASE + [("write", "enc.csv", ENCT)]
FIT_ENC = ["fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", "sex,port", "--output", "m.cml"]


WIDE_N = 129
WIDE_NAMES = ",".join("c%03d" % j for j in range(WIDE_N))
WIDE_CSV = "id," + WIDE_NAMES + ",y\n" + "".join("%d,%s,%d\n" % (r, ",".join("v%04d" % r for _ in range(WIDE_N)), r % 7) for r in range(1024))


def table_digest_off(d):
    """a byte of the string table changed and the checksum of the file repaired: the digest of the table no longer matches (reason digest)"""
    s = sections(d)[5]
    return fix_checksum(set_bytes(d, s[1] + 40, bytes([d[s[1] + 40] ^ 1])))


def _enc_cases():
    return [
        case("fit-linear-categorical-writes-the-file", FIT_ENC + ["--step", "encode column=sex,port kind=onehot", "enc.csv"],
             ok({"model": "linear", "written": True, "path": "m.cml", "summary": {"steps": 1, "string_table": {"entries": 1, "strings": 6}}}), setup=EBASE),
        case("predict-from-a-categorical-model-file", ["predict", "--model-file", "m.cml", "--drop", "id", "enc.csv"],
             ok({"model": "linear", "columns": ["prediction"], "row_count": 8, "total_rows": 8, "unknown_categories": 0}),
             setup=EBASE + [("fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", "sex,port", "--drop", "id", "--step", "encode column=sex,port kind=onehot", "--output", "m.cml", "enc.csv")]),
        case("eval-from-a-categorical-model-file", ["eval", "--model-file", "m.cml", "--drop", "id", "enc.csv"],
             ok({"model": "linear", "task": "regression", "rows": 8, "unknown_categories": 0}),
             setup=EBASE + [("fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", "sex,port", "--drop", "id", "--step", "encode column=sex,port kind=onehot", "--output", "m.cml", "enc.csv")]),
        case("ml-schema-mismatch-categorical-column-missing", ["predict", "--model-file", "m.cml", "--drop", "id", "nocat.csv"], bad("ml.schema.mismatch"),
             setup=EBASE + [("write", "nocat.csv", "id,sex,age\n1,male,20\n2,female,30\n"),
                            ("fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", "sex,port", "--drop", "id", "--step", "encode column=sex,port kind=onehot", "--output", "m.cml", "enc.csv")]),
        case("ml-model-string-table-digest", ["predict", "--model-file", "bad.cml", "--drop", "id", "enc.csv"], bad("ml.model.string-table"),
             setup=EBASE + [("fit", "--target", "y", "--model", "linear", "--l2", "1", "--cat", "sex,port", "--drop", "id", "--step", "encode column=sex,port kind=onehot", "--output", "m.cml", "enc.csv"),
                            ("mutate", "m.cml", "bad.cml", table_digest_off)]),
        case("ml-limit-string-table", ["fit", "--model", "tree", "--task", "regression", "--target", "y", "--cat", WIDE_NAMES, "--max-categories", "1024", "--max-cat-columns", "200",
                                       "--step", "freq column=* max-categories=1024", "--output", "m.cml", "wide.csv"], bad("ml.limit.string-table"), setup=BASE + [("write", "wide.csv", WIDE_CSV)]),
        case("ml-plan-column-kind-numeric-step-on-a-categorical-column", FIT_ENC + ["--step", "scale column=sex kind=standard", "enc.csv"], bad("ml.plan.column-kind"), setup=EBASE),
        case("ml-plan-column-kind-freq-on-a-numeric-column", FIT_ENC + ["--step", "freq column=age", "enc.csv"], bad("ml.plan.column-kind"), setup=EBASE),
        case("ml-input-reserved-category", FIT_ENC + ["--step", "encode column=sex,port kind=onehot min-count=2", "--split", "train=0.75", "enc.csv"], bad("ml.input.reserved-category"), setup=EBASE),
        case("ml-input-categorical-not-encoded-by-a-planned-model", FIT_ENC + ["--step", "encode column=sex kind=onehot", "--split", "train=0.75", "enc.csv"],
             bad("ml.input.categorical-not-encoded"), setup=EBASE),
    ]


CASES += _enc_cases()


# ------------------------------------------------------------------ the tree and the forest (docs/wiring-trees.md)
FIT_TREE = ["fit", "--model", "tree", "--target", "y", "--output", "t.cml", "lin.csv"]
FIT_FOREST_CLS = ["fit", "--model", "forest", "--task", "classification", "--trees", "7", "--seed", "3", "--target", "lab", "--output", "f.cml", "lcls.csv"]
WITH_TREE = LBASE + [("fit", "--model", "tree", "--target", "y", "--output", "t.cml", "lin.csv")]
WITH_FOREST = LBASE + [("fit", "--model", "forest", "--task", "classification", "--trees", "7", "--seed", "3", "--target", "lab", "--output", "f.cml", "lcls.csv")]


def _tree_cases():
    return [
        case("fit-tree-regression", FIT_TREE,
             ok({"model": "tree", "task": "regression", "rows": 8, "columns": 3, "target": "y", "path": "t.cml", "written": True,
                 "summary": {"plan": "model kind=tree task=regression max-depth=8 min-leaf=1 bins=64 max-nodes=4194304\n", "steps": 0, "features": 2,
                             "forest": {"trees": 1, "bins": 64, "max_depth": 8, "min_leaf": 1, "mtry": 2}}}), setup=LBASE),
        case("fit-tree-classification", ["fit", "--model", "tree", "--task", "classification", "--max-depth", "3", "--min-leaf", "2", "--bins", "16", "--max-nodes", "100",
                                         "--target", "lab", "--output", "t.cml", "lcls.csv"],
             ok({"model": "tree", "task": "classification", "summary": {"classes": [0, 1], "positive": 1, "forest": {"trees": 1, "bins": 16, "max_depth": 3, "min_leaf": 2}}}), setup=LBASE),
        case("fit-forest-regression-every-flag", ["fit", "--model", "forest", "--task", "regression", "--trees", "5", "--max-depth", "4", "--min-leaf", "1", "--bins", "8",
                                                 "--mtry", "1", "--bootstrap", "0", "--max-nodes", "1000", "--seed", "3", "--threads", "2", "--target", "y", "--output", "f.cml",
                                                 "lin.csv"],
             ok({"model": "forest", "task": "regression", "summary": {"forest": {"trees": 5, "bins": 8, "max_depth": 4, "mtry": 1}}}), setup=LBASE),
        case("fit-forest-classification-positive-label", FIT_FOREST_CLS[:-1] + ["--positive", "0", "lcls.csv"],
             ok({"model": "forest", "task": "classification", "summary": {"classes": [1, 0], "positive": 0}}), setup=LBASE),
        case("fit-forest-from-a-plan", ["fit", "--plan", "p.txt", "--target", "lab", "--output", "f.cml", "lcls.csv"],
             ok({"model": "forest", "task": "classification", "summary": {"steps": 1, "forest": {"trees": 3, "max_depth": 5}}}),
             setup=LBASE + [("write", "p.txt", "scale column=* kind=standard\nmodel kind=forest task=classification trees=3 max-depth=5\n")]),
        case("fit-tree-takes-missing-cells-without-an-impute-step", ["fit", "--model", "tree", "--target", "y", "--output", "m.cml", "linmis.csv"],
             ok({"model": "tree", "rows": 8, "summary": {"steps": 0}}), setup=LBASE),
        case("fit-forest-text", ["--format", "text"] + FIT_FOREST_CLS, ok(None, stdout_starts="fit: forest classification, 12 rows, wrote f.cml"), setup=LBASE),
        case("fit-forest-split-holdout", ["fit", "--model", "forest", "--task", "classification", "--trees", "5", "--split", "train=0.7", "--seed", "1", "--target", "lab",
                                          "--output", "f.cml", "lcls.csv"], ok({"summary": {"split": {"train": 8, "test": 4, "seed": 1}}}), setup=LBASE),
        case("predict-tree-regression", ["predict", "--model-file", "t.cml", "--limit", "2", "lin.csv"],
             ok({"model": "tree", "task": "regression", "columns": ["prediction"], "row_count": 2, "total_rows": 8, "truncated": True, "unknown_categories": 0}), setup=WITH_TREE),
        case("predict-forest-proba", ["predict", "--model-file", "f.cml", "--proba", "--limit", "3", "lcls.csv"],
             ok({"model": "forest", "task": "classification", "columns": ["label", "probability"], "row_count": 3}), setup=WITH_FOREST),
        case("predict-forest-label", ["predict", "--model-file", "f.cml", "--limit", "3", "lcls.csv"], ok({"columns": ["label"], "row_count": 3}), setup=WITH_FOREST),
        case("predict-forest-with-a-missing-cell", ["predict", "--model-file", "f.cml", "--limit", "20", "x.csv"], ok({"row_count": 2}),
             setup=WITH_FOREST + [("write", "x.csv", "a,b\n1,\n,2\n")]),
        case("eval-tree-regression", ["eval", "--model-file", "t.cml", "lin.csv"], ok({"model": "tree", "task": "regression", "rows": 8}), setup=WITH_TREE),
        case("eval-forest-classification", ["eval", "--model-file", "f.cml", "lcls.csv"], ok({"model": "forest", "task": "classification", "rows": 12, "positive": 1}), setup=WITH_FOREST),
        case("eval-forest-other-positive-label", ["eval", "--model-file", "f.cml", "--positive", "0", "lcls.csv"], ok({"positive": 0}), setup=WITH_FOREST),
        # ---- refusals reachable from ml
        case("ml-limit-depth", ["fit", "--model", "tree", "--max-depth", "31", "--target", "y", "--output", "t.cml", "lin.csv"], bad("ml.limit.depth"), setup=LBASE),
        case("ml-limit-nodes-a-tree-that-needs-more", ["fit", "--model", "tree", "--max-nodes", "1", "--target", "y", "--output", "t.cml", "lin.csv"], bad("ml.limit.nodes"), setup=LBASE),
        case("ml-limit-nodes-over-the-ceiling", ["fit", "--model", "forest", "--max-nodes", "4194305", "--target", "y", "--output", "t.cml", "lin.csv"], bad("ml.limit.nodes"), setup=LBASE),
        case("ml-limit-nodes-of-a-forest", ["fit", "--model", "forest", "--trees", "1024", "--max-depth", "30", "--max-nodes", "4194304", "--target", "y", "--output", "t.cml", "big.csv"],
             bad("ml.limit.nodes"), setup=LBASE + [("write", "big.csv", "a,y\n" + "".join("%d,%d\n" % (i, (i * 7) % 13) for i in range(1, 8200)))]),
        case("ml-param-out-of-range-mtry-above-d", ["fit", "--model", "forest", "--mtry", "5", "--target", "y", "--output", "f.cml", "lin.csv"], bad("ml.param.out-of-range"), setup=LBASE),
        case("ml-param-out-of-range-min-leaf-above-n", ["fit", "--model", "tree", "--min-leaf", "100", "--target", "y", "--output", "t.cml", "lin.csv"], bad("ml.param.out-of-range"), setup=LBASE),
        case("ml-param-out-of-range-bootstrap", ["fit", "--model", "forest", "--bootstrap", "2", "--target", "y", "--output", "f.cml", "lin.csv"], bad("ml.param.out-of-range"), setup=LBASE),
        case("ml-param-threads-tree", ["fit", "--model", "forest", "--threads", "65", "--target", "y", "--output", "f.cml", "lin.csv"], bad("ml.param.threads"), setup=LBASE),
        case("ml-param-out-of-range-bins-in-a-plan", ["fit", "--plan", "p.txt", "--target", "y", "--output", "t.cml", "lin.csv"], bad("ml.param.out-of-range"),
             setup=LBASE + [("write", "p.txt", "model kind=tree task=regression bins=1\n")]),
        case("ml-param-out-of-range-bins", ["fit", "--model", "tree", "--bins", "1", "--target", "y", "--output", "t.cml", "lin.csv"], bad("ml.param.out-of-range"), setup=LBASE),
        case("ml-plan-missing-key-task-of-a-plan", ["fit", "--plan", "p.txt", "--target", "y", "--output", "f.cml", "lin.csv"], bad("ml.plan.missing-key"),
             setup=LBASE + [("write", "p.txt", "model kind=forest trees=3\n")]),
        case("args-conflict-trees-with-a-tree", ["fit", "--model", "tree", "--trees", "5", "--target", "y", "--output", "t.cml", "lin.csv"], bad("args.conflict"), setup=LBASE),
        case("args-conflict-mtry-with-linear", ["fit", "--model", "linear", "--mtry", "1", "--target", "y", "--output", "m.cml", "lin.csv"], bad("args.conflict"), setup=LBASE),
        case("args-conflict-l2-with-a-forest", ["fit", "--model", "forest", "--l2", "1", "--target", "y", "--output", "f.cml", "lin.csv"], bad("args.conflict"), setup=LBASE),
        case("args-conflict-positive-with-a-regression-forest", ["fit", "--model", "forest", "--positive", "1", "--target", "y", "--output", "f.cml", "lin.csv"], bad("args.conflict"), setup=LBASE),
        case("args-conflict-max-depth-without-a-model", ["fit", "--max-depth", "3", "--target", "y", "--output", "m.cml", "lin.csv"], bad("args.conflict"), setup=LBASE),
        case("args-conflict-proba-of-a-regression-forest", ["predict", "--model-file", "t.cml", "--proba", "lin.csv"], bad("args.conflict"), setup=WITH_TREE),
        case("ml-label-not-integer-for-a-forest", ["fit", "--model", "forest", "--task", "classification", "--target", "y", "--output", "f.cml", "lin.csv"], bad("ml.label.not-integer"), setup=LBASE),
        case("ml-input-missing-value-linear-still-refuses", FIT_LIN[:-1] + ["linmis.csv"], bad("ml.input.missing-value", repair_retry=True), setup=LBASE),
    ]


CASES += _tree_cases()


# ------------------------------------------------------------------ categorical steps with a tree or forest (docs/wiring-trees.md section 13, docs/categorical-encoders.md section 6)
ENCC = ("id,sex,port,age,y\n1,male,S,20,1\n2,female,C,30,0\n3,male,S,40,1\n4,female,Q,50,0\n5,male,S,31,1\n6,female,C,22,0\n"
        "7,male,S,35,1\n8,female,C,44,0\n9,male,Q,28,1\n10,female,S,33,0\n11,male,C,26,1\n12,female,Q,38,0\n")
ENCM = "id,sex,port,age,y\n1,male,S,,1\n2,female,C,30,0\n3,male,S,40,1\n4,female,Q,50,0\n5,male,S,31,1\n6,female,C,22,0\n7,male,S,35,1\n8,female,C,44,0\n"
ECBASE = BASE + [("write", "encc.csv", ENCC), ("write", "encm.csv", ENCM)]
ENC_STEP = ["--cat", "sex,port", "--step", "encode column=sex,port kind=onehot"]


def _enc_tree_cases():
    return [
        case("fit-tree-categorical-writes-the-file", ["fit", "--model", "tree", "--target", "y"] + ENC_STEP + ["--output", "m.cml", "encc.csv"],
             ok({"model": "tree", "written": True, "path": "m.cml", "summary": {"steps": 1, "features": 7, "string_table": {"entries": 1, "strings": 5}}}), setup=ECBASE),
        case("fit-forest-categorical-writes-the-file", ["fit", "--model", "forest", "--task", "classification", "--target", "y"] + ENC_STEP + ["--output", "m.cml", "encc.csv"],
             ok({"model": "forest", "task": "classification", "written": True, "path": "m.cml", "summary": {"string_table": {"entries": 1, "strings": 5}}}), setup=ECBASE),
        case("fit-forest-categorical-split-scores-held-out-rows", ["fit", "--model", "forest", "--task", "classification", "--trees", "5", "--target", "y"] + ENC_STEP
             + ["--split", "train=0.67", "--output", "m.cml", "encc.csv"],
             ok({"model": "forest", "task": "classification", "written": True, "path": "m.cml",
                 "summary": {"steps": 1, "features": 7, "classes": [0, 1], "forest": {"trees": 5}, "split": {"train": 8, "test": 4}, "holdout": {"rows": 4}}}), setup=ECBASE),
        case("fit-tree-categorical-split-takes-a-missing-cell", ["fit", "--model", "tree", "--target", "y"] + ENC_STEP + ["--split", "train=0.5", "--output", "m.cml", "encm.csv"],
             ok({"model": "tree", "task": "regression", "written": True, "summary": {"split": {"train": 4, "test": 4}}}), setup=ECBASE),
        case("fit-tree-categorical-dry-run", ["fit", "--model", "tree", "--target", "y"] + ENC_STEP + ["--dry-run", "--output", "m.cml", "encc.csv"],
             ok({"model": "tree", "written": False}, exit=9, dry_run=True, no_file="m.cml"), setup=ECBASE),
        case("predict-from-a-categorical-tree-file", ["predict", "--model-file", "m.cml", "--drop", "id", "encc.csv"],
             ok({"model": "tree", "row_count": 12, "unknown_categories": 0}), setup=ECBASE + [("fit", "--model", "tree", "--target", "y", "--drop", "id", "--cat", "sex,port", "--step", "encode column=sex,port kind=onehot", "--output", "m.cml", "encc.csv")]),
    ]


CASES += _enc_tree_cases()


# ------------------------------------------------------------------ a model file that predict could not read is not written (docs/wiring-trees.md section 13)
def _max_bytes_cases():
    return [
        case("limit-input-too-large-fit-forest-file-over-max-bytes", ["fit", "--model", "forest", "--task", "classification", "--trees", "7", "--seed", "3", "--max-bytes", "1000",
                                                                      "--target", "lab", "--output", "f.cml", "lcls.csv"], bad("limit.input-too-large"), setup=LBASE),
        case("limit-input-too-large-fit-linear-file-over-max-bytes", ["fit", "--model", "linear", "--max-bytes", "100", "--target", "y", "--output", "m.cml", "lin.csv"],
             bad("limit.input-too-large"), setup=LBASE),
        case("limit-input-too-large-fit-constant-file-over-max-bytes", ["fit", "--max-bytes", "10", "--target", "y", "--output", "m.cml", "reg.csv"],
             bad("limit.input-too-large"), setup=BASE),
        case("limit-input-too-large-fit-dry-run-refuses-too", ["fit", "--model", "tree", "--max-bytes", "100", "--dry-run", "--target", "y", "--output", "t.cml", "lin.csv"],
             bad("limit.input-too-large"), setup=LBASE),
        case("fit-forest-max-bytes-at-the-size-of-the-file", ["fit", "--model", "tree", "--max-bytes", "100000", "--target", "y", "--output", "t.cml", "lin.csv"],
             ok({"model": "tree", "written": True}), setup=LBASE),
    ]


CASES += _max_bytes_cases()


# ------------------------------------------------------------------ k-means (docs/wiring-kmeans.md): no target, the cluster of each row, the inertia
FIT_KM = ["fit", "--model", "kmeans", "--k", "2", "--seed", "3", "--output", "k.cml", "lin.csv"]
WITH_KM = LBASE + [("fit", "--model", "kmeans", "--k", "2", "--seed", "3", "--output", "k.cml", "lin.csv")]


def _kmeans_cases():
    return [
        case("fit-kmeans-every-flag", ["fit", "--model", "kmeans", "--k", "2", "--iterations", "5", "--init", "random", "--seed", "3", "--threads", "2", "--output", "k.cml", "lin.csv"],
             ok({"model": "kmeans", "task": "clustering", "rows": 8, "columns": 3, "target": None, "written": True, "path": "k.cml",
                 "summary": {"features": 3, "clusters": {"k": 2, "iterations": 5, "init": "random", "rows": 8}}}), setup=LBASE),
        case("fit-kmeans-default-initialisation", FIT_KM, ok({"model": "kmeans", "summary": {"clusters": {"k": 2, "iterations": 10, "init": "kmeans++"}}}), setup=LBASE),
        case("fit-kmeans-with-steps", FIT_KM[:-1] + ["--step", "scale column=* kind=standard", "lin.csv"], ok({"model": "kmeans", "summary": {"steps": 1, "features": 3}}), setup=LBASE),
        case("fit-kmeans-from-a-plan", ["fit", "--plan", "p.txt", "--output", "k.cml", "lin.csv"],
             ok({"model": "kmeans", "task": "clustering", "summary": {"steps": 1, "clusters": {"k": 3, "iterations": 4, "init": "random"}}}),
             setup=LBASE + [("write", "p.txt", "scale column=* kind=minmax\nmodel kind=kmeans k=3 iterations=4 init=random\n")]),
        case("fit-kmeans-text", ["--format", "text"] + FIT_KM, ok(None, stdout_starts="fit: kmeans clustering, 8 rows, wrote k.cml"), setup=LBASE),
        case("fit-kmeans-drops-the-label", ["fit", "--model", "kmeans", "--k", "2", "--drop", "lab", "--output", "k.cml", "lcls.csv"], ok({"model": "kmeans", "columns": 2}), setup=LBASE),
        case("fit-kmeans-split-holdout", FIT_KM[:-1] + ["--split", "train=0.75", "lin.csv"],
             ok({"model": "kmeans", "summary": {"split": {"train": 6, "test": 2}, "holdout": {"rows": 2}}}), setup=LBASE),
        case("fit-kmeans-dry-run", FIT_KM[:-1] + ["--dry-run", "lin.csv"], ok({"model": "kmeans", "written": False}, exit=9, dry_run=True, no_file="k.cml"), setup=LBASE),
        case("fit-kmeans-categorical", ["fit", "--model", "kmeans", "--k", "2", "--seed", "1"] + ENC_STEP + ["--drop", "id", "--output", "k.cml", "encc.csv"],
             ok({"model": "kmeans", "task": "clustering", "written": True, "summary": {"string_table": {"entries": 1}}}), setup=ECBASE),
        case("predict-kmeans-cluster", ["predict", "--model-file", "k.cml", "--limit", "3", "lin.csv"], ok({"model": "kmeans", "task": "clustering", "columns": ["cluster"], "row_count": 3}), setup=WITH_KM),
        case("predict-kmeans-distance", ["predict", "--model-file", "k.cml", "--distance", "--limit", "3", "lin.csv"], ok({"columns": ["cluster", "distance"], "row_count": 3}), setup=WITH_KM),
        case("predict-kmeans-without-the-label-column", ["predict", "--model-file", "k.cml", "--drop", "lab", "lcls.csv"], ok({"row_count": 12}),
             setup=LBASE + [("fit", "--model", "kmeans", "--k", "3", "--drop", "lab", "--output", "k.cml", "lcls.csv")]),
        case("eval-kmeans", ["eval", "--model-file", "k.cml", "lin.csv"], ok({"model": "kmeans", "task": "clustering", "rows": 8, "clusters": 2}), setup=WITH_KM),
        case("eval-kmeans-text", ["--format", "text", "eval", "--model-file", "k.cml", "lin.csv"], ok(None, stdout_starts="on train\ninertia "), setup=WITH_KM),
        case("eval-kmeans-split", ["eval", "--model-file", "k.cml", "--split", "train=0.75", "--seed", "1", "--threads", "3", "lin.csv"], ok({"rows": 2, "split": {"train": 6, "test": 2, "seed": 1}}),
             setup=WITH_KM),
        # ---- refusals
        case("args-conflict-target-with-kmeans", ["fit", "--model", "kmeans", "--k", "2", "--target", "y", "--output", "k.cml", "lin.csv"], bad("args.conflict"), setup=LBASE),
        case("args-conflict-target-with-a-kmeans-plan", ["fit", "--plan", "p.txt", "--target", "y", "--output", "k.cml", "lin.csv"], bad("args.conflict"),
             setup=LBASE + [("write", "p.txt", "model kind=kmeans k=2\n")]),
        case("args-required-flag-target-with-a-plan", ["fit", "--plan", "p.txt", "--output", "m.cml", "lin.csv"], bad("args.required-flag"),
             setup=LBASE + [("write", "p.txt", "model kind=linear\n")]),
        case("args-conflict-task-with-kmeans", ["fit", "--model", "kmeans", "--k", "2", "--task", "regression", "--output", "k.cml", "lin.csv"], bad("args.conflict"), setup=LBASE),
        case("args-conflict-positive-with-kmeans", ["fit", "--model", "kmeans", "--k", "2", "--positive", "1", "--output", "k.cml", "lin.csv"], bad("args.conflict"), setup=LBASE),
        case("args-conflict-k-with-linear", ["fit", "--model", "linear", "--k", "2", "--target", "y", "--output", "m.cml", "lin.csv"], bad("args.conflict"), setup=LBASE),
        case("args-conflict-init-with-a-tree", ["fit", "--model", "tree", "--init", "random", "--target", "y", "--output", "m.cml", "lin.csv"], bad("args.conflict"), setup=LBASE),
        case("args-conflict-k-without-a-model", ["fit", "--k", "2", "--target", "y", "--output", "m.cml", "lin.csv"], bad("args.conflict"), setup=LBASE),
        case("args-conflict-proba-of-kmeans", ["predict", "--model-file", "k.cml", "--proba", "lin.csv"], bad("args.conflict"), setup=WITH_KM),
        case("args-conflict-distance-of-linear", ["predict", "--model-file", "lin.cml", "--distance", "lin.csv"], bad("args.conflict"), setup=WITH_LIN),
        case("args-conflict-distance-of-constant", ["predict", "--model-file", "m.cml", "--distance", "reg.csv"], bad("args.conflict"), setup=WITH_MODEL),
        case("args-conflict-positive-at-eval-of-kmeans", ["eval", "--model-file", "k.cml", "--positive", "1", "lin.csv"], bad("args.conflict"), setup=WITH_KM),
        case("args-bad-value-init", ["fit", "--model", "kmeans", "--k", "2", "--init", "forgy", "--output", "k.cml", "lin.csv"], bad("args.bad-value"), setup=LBASE),
        case("ml-plan-missing-key-k", ["fit", "--model", "kmeans", "--output", "k.cml", "lin.csv"], bad("ml.plan.missing-key"), setup=LBASE),
        case("ml-param-out-of-range-k-zero", ["fit", "--model", "kmeans", "--k", "0", "--output", "k.cml", "lin.csv"], bad("ml.param.out-of-range"), setup=LBASE),
        case("ml-param-out-of-range-k-above-the-ceiling", ["fit", "--model", "kmeans", "--k", "1025", "--output", "k.cml", "lin.csv"], bad("ml.param.out-of-range"), setup=LBASE),
        case("ml-param-out-of-range-k-above-the-rows", ["fit", "--model", "kmeans", "--k", "9", "--output", "k.cml", "lin.csv"], bad("ml.param.out-of-range"), setup=LBASE),
        case("ml-param-out-of-range-kmeans-iterations", ["fit", "--model", "kmeans", "--k", "2", "--iterations", "1001", "--output", "k.cml", "lin.csv"], bad("ml.param.out-of-range"), setup=LBASE),
        case("ml-param-threads-kmeans", ["fit", "--model", "kmeans", "--k", "2", "--threads", "0", "--output", "k.cml", "lin.csv"], bad("ml.param.threads"), setup=LBASE),
        case("ml-input-too-many-rows-kmeans", ["fit", "--model", "kmeans", "--k", "2", "--max-rows", "4", "--output", "k.cml", "lin.csv"], bad("ml.input.too-many-rows"), setup=LBASE),
        case("ml-input-missing-value-kmeans", ["fit", "--model", "kmeans", "--k", "2", "--output", "k.cml", "linmis.csv"], bad("ml.input.missing-value", repair_retry=True), setup=LBASE),
        case("ml-input-missing-value-predict-kmeans", ["predict", "--model-file", "k.cml", "holes.csv"], bad("ml.input.missing-value"),
             setup=WITH_KM + [("write", "holes.csv", "a,b,y\n1,,3\n2,2,4\n")]),
        case("ml-model-malformed-kmeans-with-a-target", ["predict", "--model-file", "bad.cml", "lin.csv"], bad("ml.model.malformed"),
             setup=WITH_KM + [("mutate", "k.cml", "bad.cml", lambda d: fix_checksum(set_bytes(d, sections(d)[1][1] + 4, (0).to_bytes(4, "little"))))]),
    ]


CASES += _kmeans_cases()

# ------------------------------------------------------------------ separable data and collinear columns (docs/refusals-fit.md, issue 48)
# Written BEFORE the code, against main: every case below fails there (the fit is not refused with these tags).
DUPC = "a,b,y\n1,1,0\n2,2,1\n3,3,0\n4,4,1\n5,5,1\n6,6,0\n"
CONSTC = "a,k,y\n1,5,0\n2,5,1\n3,5,0\n4,5,1\n5,5,1\n6,5,0\n"
SPC = "id,sp,y\n1,cat,1\n2,dog,0\n3,cat,0\n4,bird,1\n5,dog,1\n6,bird,0\n7,cat,1\n8,dog,0\n9,bird,1\n10,cat,0\n11,dog,1\n12,bird,0\n"
SEP = "x1,x2,y\n-3,1,0\n-2,-1,0\n-1,2,0\n1,-2,1\n2,1,1\n3,-1,1\n"
NEARSEP = SEP + "3,-1,0\n"
QUASI = "x,y\n-2,0\n-1,0\n0,0\n0,1\n1,1\n2,1\n"
NEARCOL = "a,b,y\n1,1.00001,0\n2,2.00002,1\n3,2.99999,0\n4,4.00004,1\n5,5.00003,1\n6,5.99996,0\n7,7.00001,1\n"
# not separable as a whole (a fit with l2 0 is made), separable once the rows of one of three folds are left out; a duplicate column in the first time block only
CVSEP = "x,y\n-3,0\n-2,0\n-1,0\n1,1\n2,1\n3,1\n-2.5,0\n2.5,0\n-1.5,0\n1.5,1\n2.8,1\n-0.5,0\n"
CVCOL = "day,a,b,y\n1,1,1,2\n2,2,2,3\n3,3,3,5\n4,4,5,7\n5,5,3,9\n6,6,8,11\n7,7,1,12\n8,8,9,15\n"
RBASE = [("write", "dup.csv", DUPC), ("write", "const.csv", CONSTC), ("write", "sp.csv", SPC), ("write", "sep.csv", SEP),
         ("write", "nearsep.csv", NEARSEP), ("write", "nearcol.csv", NEARCOL), ("write", "lcls.csv", LCLS)]
FIT_SP = ["--cat", "sp", "--step", "encode column=sp kind=onehot", "--drop", "id", "--target", "y", "--output", "m.cml", "sp.csv"]


def _refuse_cases():
    col = lambda **kw: bad("ml.fit.collinear", **kw)
    sepa = lambda **kw: bad("ml.fit.separable", **kw)
    return [
        case("ml-fit-collinear-duplicate-column", ["fit", "--model", "linear", "--target", "y", "--output", "m.cml", "dup.csv"],
             col(repair_runs=True, repair_has=[["--l2", "1e-7"], ["--drop", "b"]],
                 detail={"model": "linear", "pivot": 1, "column": "b", "columns": ["a"], "coefficients": ["1.000000"]}), setup=RBASE),
        case("ml-fit-collinear-constant-column-and-the-intercept", ["fit", "--model", "linear", "--target", "y", "--output", "m.cml", "const.csv"],
             col(repair_runs=True, detail={"model": "linear", "column": "(intercept)", "columns": ["k"]}), setup=RBASE),
        case("ml-fit-collinear-full-one-hot-and-the-intercept", ["fit", "--model", "linear"] + FIT_SP,
             col(repair_runs=True, detail={"model": "linear", "column": "(intercept)", "columns": ["sp#0", "sp#1", "sp#2"],
                                           "coefficients": ["1.000000", "1.000000", "1.000000"]}), setup=RBASE),
        case("ml-fit-collinear-logistic-duplicate-column", ["fit", "--model", "logistic", "--target", "y", "--output", "m.cml", "dup.csv"],
             col(repair_runs=True, detail={"model": "logistic", "pass": 0, "pivot": 1, "column": "b", "columns": ["a"]}), setup=RBASE),
        case("ml-fit-collinear-logistic-full-one-hot", ["fit", "--model", "logistic"] + FIT_SP,
             col(repair_runs=True, detail={"model": "logistic", "pass": 0, "column": "(intercept)"}), setup=RBASE),
        case("ml-fit-collinear-ridge-too-small", ["fit", "--model", "linear", "--l2", "1e-300", "--target", "y", "--output", "m.cml", "dup.csv"],
             col(repair_runs=True, detail={"l2": "1e-300"}), setup=RBASE),
        case("ml-fit-collinear-with-a-step-offers-only-the-ridge", ["fit", "--model", "linear", "--step", "scale column=a,b kind=standard", "--target", "y", "--output", "m.cml", "dup.csv"],
             col(repair_runs=True), setup=RBASE),
        case("ml-fit-collinear-in-a-plan-has-no-argv", ["fit", "--plan", "p.txt", "--target", "y", "--output", "m.cml", "dup.csv"],
             col(repair_none=True), setup=RBASE + [("write", "p.txt", "model kind=linear l2=0\n")]),
        case("ml-fit-separable", ["fit", "--model", "logistic", "--target", "y", "--output", "m.cml", "sep.csv"],
             sepa(repair_runs=True, repair_has=[["--l2", "1e-2"]], detail={"reason": "loss", "pass": 2, "l2": "0.0", "loss_bound": "0.6931"}), setup=RBASE),
        case("ml-fit-separable-25-iterations", ["fit", "--model", "logistic", "--iterations", "25", "--target", "y", "--output", "m.cml", "sep.csv"],
             sepa(repair_runs=True, detail={"reason": "loss", "pass": 2}), setup=RBASE),
        case("ml-fit-separable-explicit-zero-l2-is-replaced", ["fit", "--model", "logistic", "--l2", "0", "--target", "y", "--output", "m.cml", "sep.csv"],
             sepa(repair_runs=True), setup=RBASE),
        case("ml-fit-separable-in-a-plan-has-no-argv", ["fit", "--plan", "p.txt", "--target", "y", "--output", "m.cml", "sep.csv"],
             sepa(repair_none=True), setup=RBASE + [("write", "p.txt", "model kind=logistic iterations=50 l2=0\n")]),
        case("ml-fit-separable-quasi-complete-is-seen-by-the-predictor-scale", ["fit", "--model", "logistic", "--target", "y", "--output", "m.cml", "quasi.csv"],
             sepa(repair_runs=True, detail={"reason": "scale", "pass": 39, "scale_bound": "50.0"}), setup=RBASE + [("write", "quasi.csv", QUASI)]),
        case("ml-linalg-not-positive-definite-ridge-below-the-tolerance-after-the-first-pass", ["fit", "--model", "logistic", "--l2", "1e-20", "--target", "y", "--output", "m.cml", "sep.csv"],
             bad("ml.linalg.not-positive-definite"), setup=RBASE),
        case("ml-fit-separable-a-positive-l2-is-fitted", ["fit", "--model", "logistic", "--l2", "0.01", "--target", "y", "--output", "m.cml", "sep.csv"],
             ok({"model": "logistic"}), setup=RBASE),
        case("ml-fit-separable-two-passes-are-not-enough-to-see-it", ["fit", "--model", "logistic", "--iterations", "2", "--target", "y", "--output", "m.cml", "sep.csv"],
             ok({"model": "logistic"}), setup=RBASE),
        case("ml-fit-nearly-separable-but-not-is-fitted", ["fit", "--model", "logistic", "--target", "y", "--output", "m.cml", "nearsep.csv"],
             ok({"model": "logistic"}), setup=RBASE),
        case("ml-fit-overlapping-classes-are-fitted", ["fit", "--model", "logistic", "--target", "lab", "--output", "m.cml", "lcls.csv"],
             ok({"model": "logistic"}), setup=RBASE),
        case("ml-fit-near-collinear-columns-are-fitted", ["fit", "--model", "linear", "--target", "y", "--output", "m.cml", "nearcol.csv"],
             ok({"model": "linear"}), setup=RBASE),
        # `ml cv` fits the training rows of a fold: a fold can be separable or collinear when the whole data is not, and the refusal names it (docs/refusals-fit.md section 6)
        case("ml-cv-fold-separable-names-the-fold", ["cv", "--model", "logistic", "--target", "y", "--folds", "3", "cvsep.csv"],
             sepa(repair_runs=True, repair_has=[["--l2", "1e-2"]], detail={"reason": "loss", "repeat": 0, "fold": 1}), setup=RBASE + [("write", "cvsep.csv", CVSEP)]),
        case("ml-cv-fold-collinear-names-the-fold", ["cv", "--model", "linear", "--target", "y", "--drop", "day", "--folds", "2", "--split-by", "time:day", "cvcol.csv"],
             col(repair_runs=True, repair_has=[["--l2", "1e-7"]], detail={"model": "linear", "column": "b", "columns": ["a"], "repeat": 0, "fold": 0}), setup=RBASE + [("write", "cvcol.csv", CVCOL)]),
        case("ml-fit-full-one-hot-with-a-ridge-is-fitted", ["fit", "--model", "linear", "--l2", "0.1"] + FIT_SP, ok({"model": "linear"}), setup=RBASE),
    ]


CASES += _refuse_cases()


# ------------------------------------------------------------------ run tracking and the skew checks (docs/tracking.md)
def _reseal_model_sha(sb):
    """the run record with another model hash, sealed again: what a record made by another compiler holds"""
    p = next(q for q in sorted((sb.dir / "runs").iterdir()) if len(q.name) == 69)
    body = p.read_text().split(',"check"')[0]
    import json as _json

    old = _json.loads(p.read_text())["model"]["sha256"]
    body = body.replace(old, "0" * 64)
    p.write_text(body + ',"check":"%s"}\n' % hashlib.sha256(body.encode()).hexdigest())


def _edit_record(sb):
    p = next(q for q in sorted((sb.dir / "runs").iterdir()) if len(q.name) == 69)
    p.write_text(p.read_text().replace('"steps":0', '"steps":9'))


def _thousand_records(sb):
    for i in range(1000):
        (sb.dir / "runs" / ("x%04d.json" % i)).write_text("{}")


def _wide(sb):
    cols = ["c%02d" % i for i in range(70)]
    sb.write("wide.csv", ",".join(cols) + "\n" + ",".join(str((i * 7) % 5) for i in range(70)) + "\n" + ",".join(str((i * 3 + 1) % 7) for i in range(70)) + "\n" + ",".join(str((i * 5 + 2) % 11) for i in range(70)) + "\n")


# A record over 262,144 bytes (docs/tracking.md section 5.1) from arguments that fit one process on Linux, where a single argument is at most 131,072 bytes: a target name of 100,000 bytes, a list
# of dropped columns and a list of forced float columns of about 122,000 bytes each (the record holds the three)
LONG_TARGET = "t" * 100000
LONG_DROP = [("d%04d_" % i) + "x" * 80 for i in range(1400)]
LONG_FLOAT = [("f%04d_" % i) + "x" * 80 for i in range(1400)]


def long_record_columns():
    return [LONG_TARGET] + LONG_DROP + LONG_FLOAT


def long_record_args():
    return ["--model", "constant", "--target", LONG_TARGET, "--drop", ",".join(LONG_DROP), "--float", ",".join(LONG_FLOAT)]


def _long_names(sb):
    cols = long_record_columns()
    n = len(cols)
    sb.write("wide2.csv", ",".join(cols) + "\n" + ",".join(str(i % 7) for i in range(n)) + "\n" + ",".join(str((i * 3) % 11) for i in range(n)) + "\n")


FIT_TR = ["fit", "--model", "linear", "--target", "y", "--output", "lin.cml", "--track", "runs", "lin.csv"]
WITH_RUN = LBASE + [("mkdir", "runs"), tuple(FIT_TR)]


def _tracking_cases():
    return [
        case("ml-predict-out-of-range", ["predict", "--model-file", "lin.cml", "--max-out-of-range", "0", "far.csv"], bad("ml.predict.out-of-range"), setup=WITH_LIN + [("write", "far.csv", "a,b\n99,2\n")]),
        case("ml-predict-out-of-range-at-eval", ["eval", "--model-file", "lin.cml", "--max-out-of-range", "0", "far.csv"], bad("ml.predict.out-of-range"), setup=WITH_LIN + [("write", "far.csv", "a,b,y\n99,2,5\n")]),
        case("ml-predict-out-of-range-reports-the-counts", ["predict", "--model-file", "lin.cml", "far.csv"],
             ok({"out_of_range": {"cells": 1, "rows": 1, "columns": [{"name": "a", "count": 1}]}, "ignored_columns": []}), setup=WITH_LIN + [("write", "far.csv", "a,b\n99,2\n")]),
        case("ml-predict-no-provenance", ["predict", "--model-file", "old.cml", "--max-out-of-range", "0", "lin.csv"], bad("ml.predict.no-provenance"),
             setup=WITH_LIN + [("mutate", "lin.cml", "old.cml", without_provenance)]),
        case("eval-says-what-it-scored", ["eval", "--model-file", "lin.cml", "lin.csv"], ok({"on": "train"}), setup=WITH_LIN),
        case("eval-of-another-file-says-other", ["eval", "--model-file", "lin.cml", "lin2.csv"], ok({"on": "other"}), setup=WITH_LIN + [("write", "lin2.csv", LIN.replace("8,2,17.1", "8,2,17.2"))]),
        case("ml-runs-fit-track-writes-a-record", FIT_TR, ok({"run": {"written": True}}), setup=LBASE + [("mkdir", "runs")]),
        case("ml-runs-track-directory-missing", FIT_TR, bad("io.not-found"), setup=LBASE),
        case("ml-runs-conflict", FIT_TR + ["--overwrite"], bad("ml.runs.conflict"), setup=WITH_RUN + [("raw", _edit_record)]),
        case("ml-runs-too-many", FIT_TR, bad("ml.runs.too-many"), setup=LBASE + [("mkdir", "runs"), ("raw", _thousand_records)]),
        case("ml-runs-record-too-large", ["fit"] + long_record_args() + ["--output", "w.cml", "--track", "runs", "wide2.csv"],
             bad("ml.runs.record-too-large"), setup=[("mkdir", "runs"), ("raw", _long_names)]),
        case("ml-runs-show-not-found", ["runs", "show", "runs", "0" * 64], bad("ml.runs.not-found"), setup=LBASE + [("mkdir", "runs")]),
        case("ml-runs-unknown-model", ["eval", "--model-file", "lin.cml", "--track", "runs", "lin.csv"], bad("ml.runs.unknown-model"), setup=WITH_LIN + [("mkdir", "runs")]),
        case("ml-runs-damaged", ["runs", "show", "runs", "ab" * 32], bad("ml.runs.damaged"), setup=LBASE + [("mkdir", "runs"), ("write", "runs/" + "ab" * 32 + ".json", "{}")]),
        case("ml-runs-data-changed", ["runs", "verify", "runs", "@ID"], bad("ml.runs.data-changed"), setup=WITH_RUN + [("write", "lin.csv", LIN.replace("8,2,17.1", "8,2,17.3"))]),
        case("ml-runs-model-differs", ["runs", "verify", "runs", "@ID"], bad("ml.runs.model-differs"), setup=WITH_RUN + [("raw", _reseal_model_sha)]),
        case("ml-runs-not-replayable", ["runs", "verify", "runs", "@ID"], bad("ml.runs.not-replayable"),
             setup=[("mkdir", "runs"), ("raw", _wide), ("fit", "--model", "tree", "--task", "regression", "--target", "c00", "--drop", "c69", "--output", "w.cml", "--track", "runs", "wide.csv")]),
        case("ml-cv-track-refused", ["cv", "--model", "linear", "--target", "y", "--track", "runs", "lin.csv"], bad("args.conflict"), setup=LBASE + [("mkdir", "runs")]),
        case("ml-runs-fit-track-split-group", ["fit", "--model", "linear", "--target", "y", "--drop", "g", "--split", "train=0.75", "--split-group", "g", "--output", "lin.cml", "--track", "runs", "grp.csv"],
             ok({"run": {"written": True}, "summary": {"split": {"spec": "holdout train=0.75 group=g seed=0"}}}), setup=[("mkdir", "runs"), ("write", "grp.csv", "g,a,y\n" + "".join("%s,%d,%d\n" % ("abcd"[i % 4], i, 2 * i + 1) for i in range(20)))]),
        case("ml-runs-list", ["runs", "list", "runs"], ok({"count": 1}), setup=WITH_RUN),
        case("ml-runs-verify", ["runs", "verify", "runs", "@ID"], ok({"verified": True}), setup=WITH_RUN),
    ]


CASES += _tracking_cases()

def log_cases():
    return [c for c in CASES if c.get("log")]


# ------------------------------------------------------------------ split strategies and cross-validation (docs/splits.md); the gates are tests/conformance/test_splits.py
def _split_rows():
    grp = ["g1"] * 3 + ["g2"] * 3 + ["g3"] * 3 + ["g4"] * 3
    rows = ["grp,day,a,b,y,r"]
    for i in range(12):
        rows.append("%s,2024-01-%02d,%s,%d,%d,%s" % (grp[i], i + 1, [1.5, 2.25, 0.5, 3.0, 1.0, 2.5][i % 6], (i * 7) % 5, i % 2, [3.5, 1.25, 7.5, 2.0, 9.75, 4.5][i % 6]))
    return "\n".join(rows) + "\n"


SPL = _split_rows()
SBASE = BASE + [("write", "spl.csv", SPL)]
CVT = ["cv", "--model", "tree", "--task", "classification", "--target", "y", "--drop", "grp,day,r"]
CVR = ["cv", "--model", "linear", "--target", "r", "--drop", "grp,day,y"]
FITT = ["fit", "--model", "tree", "--task", "classification", "--target", "y", "--drop", "grp,day,r", "--output", "s.cml", "--split", "train=0.75"]


def _split_cases():
    return [
        case("cv-kfold", CVT + ["--folds", "3", "spl.csv"], ok({"model": "tree", "task": "classification", "rows": 12, "split": "cv folds=3 repeats=1 seed=0", "strategy": "kfold",
                                                               "summary": {"count": 3, "std": "sample"}}), setup=SBASE),
        case("cv-kfold-repeats-and-emit", CVT + ["--folds", "2", "--repeats", "2", "--seed", "4", "--emit-folds", "spl.csv"],
             ok({"split": "cv folds=2 repeats=2 seed=4", "assignment": {"repeats": 2, "rows": 12}}), setup=SBASE),
        case("cv-stratify", CVT + ["--folds", "3", "--stratify", "spl.csv"], ok({"split": "cv folds=3 repeats=1 stratify seed=0", "strategy": "stratified"}), setup=SBASE),
        case("cv-group", CVT + ["--folds", "2", "--split-group", "grp", "spl.csv"], ok({"split": "cv folds=2 repeats=1 group=grp seed=0", "strategy": "group"}), setup=SBASE),
        case("cv-time", CVT + ["--folds", "3", "--split-by", "time:day", "spl.csv"], ok({"split": "cv folds=3 repeats=1 time=day seed=0", "strategy": "time"}), setup=SBASE),
        case("cv-regression-linear", CVR + ["--folds", "3", "spl.csv"], ok({"model": "linear", "task": "regression"}), setup=SBASE),
        case("cv-text", ["--format", "text"] + CVT + ["--folds", "3", "spl.csv"], ok(None, stdout_starts="cv: tree classification, 12 rows, cv folds=3"), setup=SBASE),
        case("fit-split-stratify", FITT + ["--stratify", "spl.csv"], ok({"summary": {"split": {"spec": "holdout train=0.75 stratify seed=0"}}}), setup=SBASE),
        case("fit-split-group", FITT + ["--split-group", "grp", "spl.csv"], ok({"summary": {"split": {"spec": "holdout train=0.75 group=grp seed=0"}}}), setup=SBASE),
        case("fit-split-time", FITT + ["--split-by", "time:day", "spl.csv"], ok({"summary": {"split": {"spec": "holdout train=0.75 time=day seed=0", "train": 9, "test": 3}}}), setup=SBASE),
        case("fit-split-emit-folds", FITT + ["--emit-folds", "--split-by", "time:day", "spl.csv"],
             ok({"summary": {"split": {"assignment": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]}}}), setup=SBASE),
        case("eval-split-replays-the-strategy", ["eval", "--model-file", "s.cml", "--split", "train=0.75", "--split-by", "time:day", "--drop", "grp,day,r", "spl.csv"],
             ok({"rows": 3, "split": {"spec": "holdout train=0.75 time=day seed=0"}}), setup=SBASE + [("fit",) + tuple(FITT[1:]) + ("--split-by", "time:day", "spl.csv")]),
        # ---- refusals
        case("ml-split-stratify-target-regression", CVR + ["--folds", "3", "--stratify", "spl.csv"], bad("ml.split.stratify-target", repair_retry=True), setup=SBASE),
        case("ml-split-stratum-too-small-cv", CVT + ["--folds", "7", "--stratify", "spl.csv"], bad("ml.split.stratum-too-small", repair_retry=True), setup=SBASE),
        case("ml-split-stratum-too-small-holdout", ["fit", "--model", "tree", "--task", "classification", "--target", "y", "--output", "s.cml", "--split", "train=0.75", "--stratify", "one.csv"], bad("ml.split.stratum-too-small"),
             setup=SBASE + [("write", "one.csv", "a,b,y\n1,1,0\n2,2,0\n3,3,0\n4,4,0\n5,5,0\n6,6,0\n7,7,0\n8,8,1\n")]),
        case("ml-split-stratify-target-holdout", ["fit", "--model", "linear", "--target", "r", "--drop", "grp,day,y", "--output", "s.cml", "--split", "train=0.75", "--stratify", "spl.csv"],
             bad("ml.split.stratify-target", repair_retry=True), setup=SBASE),
        case("ml-split-stratum-too-small-low-class", CVT + ["--folds", "3", "--stratify", "rare.csv"], bad("ml.split.stratum-too-small"),
             setup=SBASE + [("write", "rare.csv", "grp,day,a,b,y,r\n" + "".join("g,2024-01-%02d,%d,%d,%d,1.5\n" % (i + 1, i, i % 4, 0 if i == 0 else 1) for i in range(12)))]),
        case("ml-split-stratum-too-small-high-class", CVT + ["--folds", "3", "--stratify", "rare2.csv"], bad("ml.split.stratum-too-small"),
             setup=SBASE + [("write", "rare2.csv", "grp,day,a,b,y,r\n" + "".join("g,2024-01-%02d,%d,%d,%d,1.5\n" % (i + 1, i, i % 4, 1 if i == 0 else 0) for i in range(12)))]),
        case("ml-split-too-few-groups-cv", CVT + ["--folds", "5", "--split-group", "grp", "spl.csv"], bad("ml.split.too-few-groups", repair_retry=True), setup=SBASE),
        case("ml-split-too-few-groups-holdout", ["fit", "--model", "tree", "--task", "classification", "--target", "y", "--drop", "one,day,r", "--output", "s.cml", "--split", "train=0.75", "--split-group", "one", "spl.csv"], bad("ml.split.too-few-groups"),
             setup=SBASE + [("write", "spl.csv", SPL.replace("g1", "g4").replace("g2", "g4").replace("g3", "g4").replace("grp", "one"))]),
        case("ml-split-time-format-not-a-time", CVT + ["--folds", "3", "--split-by", "time:day", "spl.csv"], bad("ml.split.time-format"),
             setup=SBASE + [("write", "spl.csv", SPL.replace("2024-01-05", "last friday"))]),
        case("ml-split-time-format-offset", CVT + ["--folds", "3", "--split-by", "time:day", "spl.csv"], bad("ml.split.time-format"),
             setup=SBASE + [("write", "spl.csv", SPL.replace("2024-01-05", "2024-01-05T10:00+01:00"))]),
        case("ml-split-time-format-mixed-kinds", CVT + ["--folds", "3", "--split-by", "time:day", "spl.csv"], bad("ml.split.time-format"),
             setup=SBASE + [("write", "spl.csv", SPL.replace("2024-01-05", "5"))]),
        case("ml-split-missing-cell-time", CVT + ["--folds", "3", "--split-by", "time:day", "spl.csv"], bad("ml.split.missing-cell"),
             setup=SBASE + [("write", "spl.csv", SPL.replace("2024-01-05", ""))]),
        case("ml-split-missing-cell-group", CVT + ["--folds", "2", "--split-group", "grp", "spl.csv"], bad("ml.split.missing-cell"),
             setup=SBASE + [("write", "spl.csv", SPL.replace("g2,2024-01-05", ",2024-01-05"))]),
        case("ml-split-combination-stratify-group", CVT + ["--folds", "2", "--stratify", "--split-group", "grp", "spl.csv"], bad("ml.split.combination"), setup=SBASE),
        case("ml-split-combination-stratify-time", FITT + ["--stratify", "--split-by", "time:day", "spl.csv"], bad("ml.split.combination"), setup=SBASE),
        case("ml-split-combination-time-group", CVT + ["--split-by", "time:day", "--split-group", "grp", "spl.csv"], bad("ml.split.combination"), setup=SBASE),
        case("ml-split-combination-cv-with-split", CVT + ["--split", "train=0.8", "spl.csv"], bad("ml.split.combination"), setup=SBASE),
        case("ml-split-combination-repeats-with-time", CVT + ["--folds", "3", "--repeats", "2", "--split-by", "time:day", "spl.csv"], bad("ml.split.combination", repair_retry=True), setup=SBASE),
        case("ml-split-out-of-range-folds-one", CVT + ["--folds", "1", "spl.csv"], bad("ml.split.out-of-range"), setup=SBASE),
        case("ml-split-out-of-range-folds-above", CVT + ["--folds", "21", "spl.csv"], bad("ml.split.out-of-range"), setup=SBASE),
        case("ml-split-out-of-range-repeats", CVT + ["--repeats", "11", "spl.csv"], bad("ml.split.out-of-range"), setup=SBASE),
        case("ml-split-column-target-group", ["cv", "--model", "tree", "--task", "classification", "--target", "y", "--drop", "grp,day,r", "--split-group", "y", "spl.csv"],
             bad("ml.split.column-target"), setup=SBASE),
        case("ml-split-emit-too-large", ["cv", "--model", "tree", "--task", "classification", "--target", "y", "--folds", "2", "--repeats", "10", "--emit-folds", "--max-rows", "200000", "big.csv"], bad("ml.split.emit-too-large", repair_retry=True),
             setup=SBASE + [("raw", lambda sb: sb.write("big.csv", "a,b,y\n" + "".join("%d,%d,%d\n" % (i % 97, i % 13, i % 2) for i in range(110000))))]),
        case("ml-split-empty-fold-more-folds-than-rows", ["cv", "--model", "tree", "--task", "classification", "--target", "y", "--folds", "20", "tiny.csv"], bad("ml.split.empty-fold"),
             setup=SBASE + [("write", "tiny.csv", "a,b,y\n1,1,0\n2,2,1\n3,3,0\n")]),
        case("ml-input-unknown-column-split-group", CVT + ["--split-group", "nosuch", "spl.csv"], bad("ml.input.unknown-column"), setup=SBASE),
        case("args-conflict-stratify-without-split", FITT[:-2] + ["--stratify", "spl.csv"], bad("args.conflict"), setup=SBASE),
        case("args-conflict-folds-at-fit", FITT + ["--folds", "3", "spl.csv"], bad("args.conflict"), setup=SBASE),
        case("args-conflict-cv-constant", ["cv", "--target", "y", "spl.csv"], bad("args.conflict"), setup=SBASE),
        case("args-bad-value-split-by", CVT + ["--split-by", "date:day", "spl.csv"], bad("args.bad-value"), setup=SBASE),
        case("args-required-flag-cv-target", ["cv", "--model", "linear", "spl.csv"], bad("args.required-flag"), setup=SBASE),
        case("args-conflict-cv-output", CVT + ["--output", "m.cml", "spl.csv"], bad("args.conflict"), setup=SBASE),
        case("ml-label-single-class-in-a-training-fold", CVT + ["--folds", "2", "--split-group", "grp", "spl.csv"], bad("ml.label.single-class"),
             setup=SBASE + [("write", "spl.csv", "grp,day,a,b,y,r\n" + "".join("g%d,2024-01-%02d,%d,%d,%d,1.5\n" % (i // 3, i + 1, i, i % 4, 1 if i < 3 else 0) for i in range(12)))]),
    ]


CASES += _split_cases()

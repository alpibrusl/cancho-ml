#!/usr/bin/env python3
"""Write schemas/ml.v1.json (the envelope of every verb) and schemas/log.v1.json (the records of --log), from the tables of
tools/ml/spec.cho. The flag names, the rule tags and the log records are enumerated in the schemas FROM the tables, so a
flag, a rule or a record that is in the table and not in the schema cannot exist: the schema is regenerated, and a committed
file that differs from the regeneration is stale (--check, and tests/conformance/test_tables.py).

Adapted from cancho-table's scripts/schemas.py: the envelope, the error object, the repair kinds and `text_or_bytes` are the
toolbox's and are written here once; the file is self-contained because scripts/manifest.py embeds it in the binary.

    python3 scripts/schemas.py           # write
    python3 scripts/schemas.py --check   # exit 1 if a committed file differs
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gen_models  # noqa: E402
import spec  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DIALECT = "https://json-schema.org/draft/2020-12/schema"
NAT = {"type": "integer", "minimum": 0}
CODES = ["GENERAL_ERROR", "INVALID_ARGS", "NOT_FOUND", "PERMISSION_DENIED", "CONFLICT", "PRECONDITION_FAILED"]
SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
DEC = {"type": "string", "pattern": "^-?[0-9]+(\\.[0-9]+)?$"}
FLT = {"type": "string", "pattern": "^-?[0-9]+(\\.[0-9]+)?(e[+-]?[0-9]+)?$"}


def obj(props, required=None):
    return {"type": "object", "properties": props,
            "required": list(props) if required is None else required, "additionalProperties": False}


TB = {"$ref": "#/$defs/text_or_bytes"}


def common_defs(t):
    return {
        "text_or_bytes": {
            "description": "UTF-8 text as a string; other bytes as base64 (RFC 4648 section 4, padded).",
            "oneOf": [{"type": "string"}, obj({"b64": {"type": "string", "pattern": "^[A-Za-z0-9+/]*={0,2}$"}})],
        },
        "flag": {"description": "Every flag of the table in tools/ml/spec.cho, as introspect spells it.",
                 "enum": ["--" + f["name"] for f in t["flags"]]},
        "rule": {"description": "Every rule tag ml can emit (the `rules` of spec.cho).", "enum": t["rules"]},
        "repair": {"oneOf": [
            {"type": "null"},
            obj({"kind": {"const": "retry"}, "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1}}),
            obj({"kind": {"const": "none"}, "reason": {"type": "string"}}),
            obj({"kind": {"const": "choose"}, "options": {"type": "array", "items": obj(
                {"argv": {"type": "array", "items": {"type": "string"}}})}}),
        ]},
        "error": obj({
            "code": {"enum": CODES},
            "rule": {"$ref": "#/$defs/rule"},
            "message": {"type": "string"},
            "hint": {"type": ["string", "null"]},
            "repair": {"$ref": "#/$defs/repair"},
            "detail": {"type": "object"},
        }),
        "meta": obj({"version": {"type": "string"}}),
    }


def wired_models():
    """The names of the wired models other than constant, from the manifests (docs/modules.md)."""
    return [m["name"] for m in gen_models.wired(gen_models.load_manifests(ROOT))]


def metrics_defs():
    # a metric with no value (a zero denominator, a constant target for R-squared) is null: docs/wiring.md section 2
    val = {"oneOf": [FLT, {"type": "null"}]}
    reg = obj({"rmse": val, "mae": val, "r2": val})
    cls = obj({"accuracy": val, "precision": val, "recall": val, "f1": val, "roc_auc": val, "log_loss": val})
    return reg, cls


def data_defs():
    fit_summary = {"oneOf": [
        obj({"mean": DEC}),
        obj({"classes": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
             "counts": {"type": "array", "items": NAT, "minItems": 2, "maxItems": 2},
             "majority": {"type": "integer"}, "probability": DEC}),
    ]}
    fit = obj({
        "model": {"const": "constant"}, "task": {"enum": ["regression", "classification"]},
        "rows": NAT, "columns": NAT, "target": TB, "path": TB, "bytes": NAT, "sha256": SHA,
        "written": {"type": "boolean"}, "summary": fit_summary,
    })
    predict = obj({
        "model": {"const": "constant"}, "task": {"enum": ["regression", "classification"]},
        "columns": {"type": "array", "items": {"type": "string"}},
        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
        "row_count": NAT, "total_rows": NAT, "truncated": {"type": "boolean"},
        "next": {"oneOf": [{"type": "null"}, obj({"from": NAT})]},
    })
    eval_reg = obj({
        "model": {"const": "constant"}, "task": {"const": "regression"}, "rows": NAT,
        "metrics": obj({"rmse": DEC, "sse": DEC, "prediction": DEC}),
    })
    eval_cls = obj({
        "model": {"const": "constant"}, "task": {"const": "classification"}, "rows": NAT,
        "metrics": obj({"accuracy": DEC, "correct": NAT, "total": NAT, "prediction": {"type": "integer"}}),
    })
    # the models that go through the pipeline (docs/wiring.md section 2)
    names = {"enum": wired_models()}
    mreg, mcls = metrics_defs()
    split = obj({"train": NAT, "test": NAT, "seed": NAT})
    holdout_reg = obj({"rows": NAT, "metrics": mreg})
    holdout_cls = obj({"rows": NAT, "metrics": mcls})
    summary_keys = {
        "plan": {"type": "string"}, "steps": NAT, "features": NAT,
        "coefficients": {"type": "array", "items": obj({"name": TB, "value": FLT})}, "intercept": FLT,
    }
    # a tree or a forest has no coefficients: what the header of its payload says (docs/wiring-trees.md section 2)
    forest_keys = {"plan": {"type": "string"}, "steps": NAT, "features": NAT,
                   "forest": obj({"trees": NAT, "nodes": NAT, "bins": NAT, "max_depth": NAT, "min_leaf": NAT, "mtry": NAT})}
    classes = {"classes": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}, "positive": {"type": "integer"}}
    # a fit with categorical steps adds to the summary the digest of the steps payload, the categorical columns the steps read and the string table the file holds
    # (docs/stringtable.md section 6)
    cat_item = obj({"step": NAT, "kind": {"enum": ["onehot", "label", "freq", "hash"]}, "column": TB, "categories": NAT, "output_columns": NAT})
    table_obj = obj({"entries": NAT, "strings": NAT, "pool_bytes": NAT, "bytes": NAT, "sha256": SHA})
    cat_keys = {"steps_sha256": SHA, "categorical": {"type": "array", "items": cat_item}, "string_table": table_obj}
    summary_reg = obj({**summary_keys, **cat_keys, "split": split, "holdout": holdout_reg}, list(summary_keys))
    summary_cls = obj({**summary_keys, **cat_keys, **classes, "split": split, "holdout": holdout_cls}, list(summary_keys) + ["classes", "positive"])
    summary_forest_reg = obj({**forest_keys, **cat_keys, "split": split, "holdout": holdout_reg}, list(forest_keys))
    summary_forest_cls = obj({**forest_keys, **cat_keys, **classes, "split": split, "holdout": holdout_cls}, list(forest_keys) + ["classes", "positive"])
    fit_p = obj({
        "model": names, "task": {"enum": ["regression", "classification"]}, "rows": NAT, "columns": NAT, "target": TB, "path": TB,
        "bytes": NAT, "sha256": SHA, "written": {"type": "boolean"}, "summary": {"oneOf": [summary_reg, summary_cls, summary_forest_reg, summary_forest_cls]},
    })
    predict_p = obj({
        "model": names, "task": {"enum": ["regression", "classification"]},
        "columns": {"type": "array", "items": {"type": "string"}},
        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
        "row_count": NAT, "total_rows": NAT, "truncated": {"type": "boolean"},
        "next": {"oneOf": [{"type": "null"}, obj({"from": NAT})]}, "unknown_categories": NAT,
    })
    eval_reg_p = obj({"model": names, "task": {"const": "regression"}, "rows": NAT, "metrics": mreg, "unknown_categories": NAT, "split": split},
                     ["model", "task", "rows", "metrics", "unknown_categories"])
    eval_cls_p = obj({"model": names, "task": {"const": "classification"}, "rows": NAT, "positive": {"type": "integer"}, "metrics": mcls,
                      "unknown_categories": NAT, "split": split}, ["model", "task", "rows", "positive", "metrics", "unknown_categories"])
    return {"oneOf": [fit, predict, eval_reg, eval_cls, fit_p, predict_p, eval_reg_p, eval_cls_p]}


def planned():
    return {"type": "array", "items": obj({
        "op": {"enum": ["create", "replace"]}, "path": TB, "bytes": NAT, "sha256": SHA})}


def envelope(t):
    props = {
        "ok": {"type": "boolean"},
        "command": {"const": "ml"},
        "schema": {"const": "ml.v1"},
        "data": {"$ref": "#/$defs/data"},
        "dry_run": {"const": True},
        "planned_actions": planned(),
        "error": {"$ref": "#/$defs/error"},
        "errors": {"type": "array", "items": {"$ref": "#/$defs/error"}, "minItems": 1},
        "meta": {"$ref": "#/$defs/meta"},
    }
    defs = common_defs(t)
    defs["data"] = data_defs()
    return {
        "$schema": DIALECT,
        "$id": "https://github.com/alpibrusl/cancho-ml/schemas/ml.v1.json",
        "title": "ml.v1",
        "description": "One JSON object on one line. ok is false exactly when errors is present; error is its first element. "
                       "A dry run (fit --dry-run) has ok true, exit 9, dry_run and planned_actions. No float appears: every "
                       "decimal is a string with its digits exact.",
        "type": "object",
        "properties": props,
        "required": ["ok", "command", "schema", "meta"],
        "additionalProperties": False,
        "allOf": [{"if": {"properties": {"ok": {"const": True}}},
                   "then": {"not": {"anyOf": [{"required": ["error"]}, {"required": ["errors"]}]}},
                   "else": {"required": ["error", "errors"]}}],
        "$defs": defs,
    }


FIELD_TYPES = {
    "event": None, "seq": NAT, "command": {"type": "string"}, "version": {"type": "string"},
    "rows": NAT, "columns": NAT, "bytes": NAT, "sha256": SHA, "ok": {"type": "boolean"},
    "exit": {"type": "integer", "minimum": 0, "maximum": 9}, "rule": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/rule"}]},
}


def log_schema(t):
    variants = []
    for event, fields in t["log"].items():
        props = {}
        for f in fields:
            props[f] = {"const": event} if f == "event" else FIELD_TYPES[f]
        variants.append(obj(props))
    return {
        "$schema": DIALECT,
        "$id": "https://github.com/alpibrusl/cancho-ml/schemas/log.v1.json",
        "title": "log.v1",
        "description": "One record of the NDJSON log (--log): a closed object whose fields are counts, flags, rule tags and "
                       "hashes. No record has a field that could hold a cell, a column name or a path.",
        "oneOf": variants,
        "$defs": {"rule": {"enum": t["rules"]}},
    }


def render(schema):
    return json.dumps(schema, indent=2, sort_keys=False) + "\n"


def all_schemas(path=spec.SPEC):
    t = spec.tables(path)
    return {"ml.v1.json": envelope(t), "log.v1.json": log_schema(t)}


def main():
    check = "--check" in sys.argv[1:]
    out = ROOT / "schemas"
    out.mkdir(exist_ok=True)
    stale = []
    for name, schema in all_schemas().items():
        path = out / name
        text = render(schema)
        if check:
            if not path.exists() or path.read_text() != text:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.write_text(text)
    if stale:
        print("stale schemas (run scripts/schemas.py): " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

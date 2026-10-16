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


def cluster_metrics_def():
    """the metrics of a clustering: the inertia, the rows of each cluster and the mean distance of each (null for an empty cluster): docs/wiring-kmeans.md section 3"""
    val = {"oneOf": [FLT, {"type": "null"}]}
    return obj({"inertia": val, "sizes": {"type": "array", "items": NAT}, "mean_distance": {"type": "array", "items": val}})


ON = {"enum": ["train", "other", "unknown"]}
# the run record of --track (docs/tracking.md section 5.1): the answer of fit names it
RUN = obj({"id": SHA, "dir": TB, "file": TB, "written": {"type": "boolean"}, "sha256": SHA})
# docs/tracking.md section 3.2: null for a model with no provenance
OOR = {"oneOf": [{"type": "null"}, obj({"cells": NAT, "rows": NAT, "columns": {"type": "array", "items": obj({"name": TB, "count": NAT})}})]}


def runs_data():
    """The answers of ml runs list, show, compare and verify (docs/tracking.md section 5.4)."""
    entry = obj({"id": SHA, "model": {"type": "string"}, "task": {"enum": ["regression", "classification", "clustering"]}, "data_sha256": SHA, "model_sha256": SHA,
                 "train_rows": NAT, "metrics_on": {"oneOf": [{"enum": ["train", "held-out"]}, {"type": "null"}]}, "evals": NAT})
    lst = obj({"dir": TB, "count": NAT, "truncated": {"type": "boolean"}, "runs": {"type": "array", "items": entry}})
    show = obj({"record": {"type": "object"}})
    diff = obj({"key": {"type": "string"}, "a": {}, "b": {}})
    met = obj({"name": {"type": "string"}, "a": {}, "b": {}, "equal": {"type": "boolean"}})
    cmp_ = obj({"a": SHA, "b": SHA, "model_equal": {"type": "boolean"}, "inputs": {"type": "array", "items": diff}, "metrics": {"type": "array", "items": met}})
    ver = obj({"id": SHA, "verified": {"const": True}, "model_sha256": SHA, "data_sha256": SHA, "threads": NAT, "compiler": TB})
    return [lst, show, cmp_, ver]


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
        "written": {"type": "boolean"}, "summary": fit_summary, "run": RUN,
    }, ["model", "task", "rows", "columns", "target", "path", "bytes", "sha256", "written", "summary"])
    predict = obj({
        "model": {"const": "constant"}, "task": {"enum": ["regression", "classification"]},
        "columns": {"type": "array", "items": {"type": "string"}},
        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
        "row_count": NAT, "total_rows": NAT, "truncated": {"type": "boolean"},
        "next": {"oneOf": [{"type": "null"}, obj({"from": NAT})]},
    })
    eval_reg = obj({
        "model": {"const": "constant"}, "task": {"const": "regression"}, "rows": NAT, "on": ON,
        "metrics": obj({"rmse": DEC, "sse": DEC, "prediction": DEC}),
    })
    eval_cls = obj({
        "model": {"const": "constant"}, "task": {"const": "classification"}, "rows": NAT, "on": ON,
        "metrics": obj({"accuracy": DEC, "correct": NAT, "total": NAT, "prediction": {"type": "integer"}}),
    })
    # the models that go through the pipeline (docs/wiring.md section 2)
    names = {"enum": wired_models()}
    mreg, mcls = metrics_defs()
    # a holdout says how the rows were chosen: the canonical text of the split (docs/splits.md section 2) and, with --emit-folds, 1 for each held-out row
    split = obj({"train": NAT, "test": NAT, "seed": NAT, "spec": {"type": "string"}, "assignment": {"type": "array", "items": {"enum": [0, 1]}}}, ["train", "test", "seed", "spec"])
    holdout_reg = obj({"rows": NAT, "on": {"const": "held-out"}, "metrics": mreg})
    holdout_cls = obj({"rows": NAT, "on": {"const": "held-out"}, "metrics": mcls})
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
    # a clustering has no target (the target of its fit answer is null) and its summary says what the fit did (docs/wiring-kmeans.md section 3)
    mclu = cluster_metrics_def()
    cluster_keys = {"plan": {"type": "string"}, "steps": NAT, "features": NAT,
                    "clusters": obj({"k": NAT, "iterations": NAT, "init": {"enum": ["random", "kmeans++"]}, "rows": NAT, "metrics": mclu})}
    summary_clu = obj({**cluster_keys, **cat_keys, "split": split, "holdout": obj({"rows": NAT, "on": {"const": "held-out"}, "metrics": mclu})}, list(cluster_keys))
    fit_p = obj({
        "model": names, "task": {"enum": ["regression", "classification", "clustering"]}, "rows": NAT, "columns": NAT, "target": {"oneOf": [TB, {"type": "null"}]}, "path": TB,
        "bytes": NAT, "sha256": SHA, "written": {"type": "boolean"}, "summary": {"oneOf": [summary_reg, summary_cls, summary_forest_reg, summary_forest_cls, summary_clu]}, "run": RUN,
    }, ["model", "task", "rows", "columns", "target", "path", "bytes", "sha256", "written", "summary"])
    predict_p = obj({
        "model": names, "task": {"enum": ["regression", "classification", "clustering"]},
        "columns": {"type": "array", "items": {"type": "string"}},
        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
        "row_count": NAT, "total_rows": NAT, "truncated": {"type": "boolean"},
        "next": {"oneOf": [{"type": "null"}, obj({"from": NAT})]}, "unknown_categories": NAT, "out_of_range": OOR, "ignored_columns": {"type": "array", "items": TB},
    })
    eval_reg_p = obj({"model": names, "task": {"const": "regression"}, "rows": NAT, "on": ON, "metrics": mreg, "unknown_categories": NAT, "out_of_range": OOR, "split": split},
                     ["model", "task", "rows", "on", "metrics", "unknown_categories", "out_of_range"])
    eval_cls_p = obj({"model": names, "task": {"const": "classification"}, "rows": NAT, "on": ON, "positive": {"type": "integer"}, "metrics": mcls,
                      "unknown_categories": NAT, "out_of_range": OOR, "split": split}, ["model", "task", "rows", "on", "positive", "metrics", "unknown_categories", "out_of_range"])
    eval_clu_p = obj({"model": names, "task": {"const": "clustering"}, "rows": NAT, "on": ON, "clusters": NAT, "metrics": mclu, "unknown_categories": NAT, "out_of_range": OOR, "split": split},
                     ["model", "task", "rows", "on", "clusters", "metrics", "unknown_categories", "out_of_range"])
    # `ml cv` (docs/splits.md section 6)
    agg = obj({"mean": {"oneOf": [FLT, {"type": "null"}]}, "std": {"oneOf": [FLT, {"type": "null"}]}, "n": NAT})
    result = obj({"repeat": NAT, "fold": NAT, "train": NAT, "test": NAT, "metrics": {"oneOf": [mreg, mcls, mclu]}})
    cv_p = obj({"model": names, "task": {"enum": ["regression", "classification", "clustering"]}, "target": {"oneOf": [TB, {"type": "null"}]}, "rows": NAT, "columns": NAT,
                "plan": {"type": "string"}, "split": {"type": "string"}, "strategy": {"enum": ["kfold", "stratified", "group", "time"]}, "folds": NAT, "repeats": NAT,
                "results": {"type": "array", "items": result, "minItems": 2},
                "summary": obj({"count": NAT, "std": {"const": "sample"}, "metrics": {"type": "object", "additionalProperties": agg}}),
                "assignment": obj({"repeats": NAT, "rows": NAT, "folds": {"type": "array", "items": {"type": "array", "items": {"type": "integer", "minimum": -1}}}})},
               ["model", "task", "target", "rows", "columns", "plan", "split", "strategy", "folds", "repeats", "results", "summary"])
    return {"oneOf": [fit, predict, eval_reg, eval_cls, fit_p, predict_p, eval_reg_p, eval_cls_p, eval_clu_p, cv_p, describe_data()] + runs_data()}


def describe_data():
    """The answer of `describe` (docs/serving.md section 5): the contract a server builds its request schema from."""
    opt = lambda s: {"oneOf": [{"type": "null"}, s]}  # noqa: E731
    column_type = {"enum": ["int", "dec", "float", "categorical"]}
    column = obj({"name": TB, "type": column_type, "scale": opt(NAT), "role": {"enum": ["feature", "target"]}, "required": {"type": "boolean"}})
    value = obj({"value": TB, "count": NAT})
    cat = obj({"column": TB, "encoder": {"enum": ["onehot", "label", "freq", "hash"]}, "categories": NAT, "buckets": opt(NAT),
               "listed": NAT, "truncated": {"type": "boolean"}, "values": {"type": "array", "items": value}})
    file_ = obj({"bytes": NAT, "sha256": SHA, "format": {"const": "CML1"}, "version": {"const": 1}, "sections": {"enum": [5, 6, 7]},
                 "compiler": TB, "sum_order": opt(NAT)})
    target = obj({"name": TB, "type": {"enum": ["int", "dec", "float"]}, "scale": opt(NAT)})
    request = obj({"header": {"type": "array", "items": TB}, "optional": {"type": "array", "items": TB},
                   "extra_columns": {"const": "refused"}, "missing_cells": {"enum": ["refused", "needs_impute", "handled"]}})
    reader = obj({"max_categories": NAT, "max_string_bytes": NAT, "max_pool_bytes": NAT, "max_cat_columns": NAT})
    output = obj({"columns": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
                  "with_proba": opt({"type": "array", "items": {"type": "string"}}),
                  "with_distance": opt({"type": "array", "items": {"type": "string"}}), "label_threshold": opt({"type": "string"})})
    feature = obj({"name": TB, "count": NAT, "min": FLT, "max": FLT, "mean": FLT, "std": FLT})
    provenance = obj({"file_sha256": SHA, "file_bytes": NAT, "file_rows": NAT, "rows": NAT, "features": {"type": "array", "items": feature}})
    return obj({
        "model": {"enum": ["constant"] + wired_models()}, "task": {"enum": ["regression", "classification", "clustering"]},
        "kind": {"type": "integer", "minimum": 1, "maximum": 7}, "plan": TB, "rows": NAT, "provenance": opt(provenance), "file": file_, "target": opt(target),
        "columns": {"type": "array", "items": column, "minItems": 1}, "request": request,
        "categorical": {"type": "array", "items": cat}, "na_values": {"type": "array", "items": TB}, "reader": opt(reader),
        "classes": opt({"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}), "positive": opt({"type": "integer"}),
        "output": output, "limits": obj({"values_per_column": NAT, "max_bytes": NAT}),
    })


def planned():
    return {"type": "array", "items": {"oneOf": [
        obj({"op": {"enum": ["create", "replace"]}, "path": TB, "bytes": NAT, "sha256": SHA}),
        obj({"op": {"const": "create"}, "dir": TB, "file": TB, "bytes": NAT, "sha256": SHA})]}}


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

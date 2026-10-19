#!/usr/bin/env python3
"""Mutants for the gates of docs/serving-http.md: the code of `ml-serve` and the checks around it. A gate that cannot fail judges nothing.

As scripts/mutants_serve.py: each mutant is a copy of the tree with one change; `ml-serve` (and `serve-check`, when the number module is the one changed) is rebuilt from the copy, the command line
`ml` is the unmutated one (so that the gate of the byte rule compares the server with a tool that is right), and the tests or scripts of the gate the mutant targets run against them.
KILLED means a test or a script failed; SURVIVED means every one passed, and the mutant must then carry a reason (`why`).

    python3 scripts/mutants_http.py              # all mutants, then the baseline (the unchanged copy must pass)
    python3 scripts/mutants_http.py --only ID    # one
    python3 scripts/mutants_http.py --list
    python3 scripts/mutants_http.py --check      # every `old` text is found where it is expected (no build, no test)

Run it on a frozen copy of the tree when `cancho install` may run in the original (it rewrites build/deps).
"""

import glob
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_cli as base  # noqa: E402

ROOT = base.ROOT

NUM = "tools/ml-serve/num.cho"
COLS = "tools/ml-serve/cols.cho"
API = "tools/ml-serve/api.cho"
SRV = "tools/ml-serve/srv.cho"
MAIN = "tools/ml-serve/main.cho"

S1 = ["test_serve_bytes"]
S2 = ["test_serve"]
B = ["test_serve_bind"]
O1 = ["scripts/gate_serve_openapi.py"]


def M(id, gate, what, edits, tests=(), cmds=(), **opt):
    return dict(id=id, gate=gate, what=what, edits=edits, tests=list(tests), cmds=[list(c) for c in cmds], opt=opt)


def e(file, old, new, nth=1):
    return (file, old, new, nth)


MUTANTS = [
    # ------------------------------------------------------------------ S3, S1: the number cells
    M("n-int-drops-the-sign", "S3", "a negative number loses its minus", [e(NUM, "    if neg {\n        o = buffer.push(heap, o, byte_of(45));\n    }", "    if false {\n        o = buffer.push(heap, o, byte_of(45));\n    }")], ["test_serve_json.Cells"], check_bin=True),
    M("n-the-decimal-point-one-late", "S3", "a dec cell has its point one digit late", [e(NUM, "o = buffer.append(heap, o, digits[0..len(digits) - scale]);", "o = buffer.append(heap, o, digits[0..len(digits) - scale + 1]);")], ["test_serve_json.Cells"], check_bin=True),
    M("n-float-is-the-next-double", "S3", "a float cell is the next double", [e(NUM, "return (flt.put_float(heap, out, key), ok());", "return (flt.put_float(heap, out, key + 1), ok());")], ["test_serve_json.Cells"], check_bin=True),
    M("n-a-fraction-is-cut", "S3", "an int cell truncates a fraction instead of refusing it", [e(NUM, "if zeros < 0 || k + zeros > 12 {", "if k + zeros > 12 {")], ["test_serve_json.Cells"], check_bin=True),
    M("n-thirteen-digits", "S3", "an int or dec cell may hold 13 digits", [e(NUM, "if zeros < 0 || k + zeros > 12 {", "if zeros < 0 || k + zeros > 13 {")], ["test_serve_json.Cells"], check_bin=True),
    M("n-the-exponent-is-ignored", "S3", "the exponent of a number is not read", [e(NUM, "let shift = e - (fe - fb) + (n - 1 - last);", "let shift = 0 - (fe - fb) + (n - 1 - last);")], ["test_serve_json.Cells"], check_bin=True),
    M("n-zero-keeps-no-point", "S3", "a zero dec cell is written without its digits", [e(NUM, 'o = buffer.append(heap, o, "0.");\n        return (push_zeros(heap, o, scale), ok());', 'o = buffer.append(heap, o, "0");\n        return (push_zeros(heap, o, 0), ok());')], ["test_serve_json.Cells"], check_bin=True),
    M("n-float-underflow-accepted", "S3", "a float that underflows to zero is accepted", [e(NUM, "if status != 0 {\n        return (out, out_of_range());\n    }\n    return (flt.put_float", "if status != 0 && status != 12 {\n        return (out, out_of_range());\n    }\n    return (flt.put_float")], ["test_serve_json.Cells"], check_bin=True),
    M("n-grammar-accepts-a-leading-zero", "S3", "the grammar check of an answer's number accepts 01", [e(NUM, "    if int_of(text[i]) == 48 {\n        i = i + 1;\n    } else {", "    if false {\n        i = i + 1;\n    } else {")], ["test_serve_json.Cells"], check_bin=True),
    # ------------------------------------------------------------------ S1, S2: the record to a row
    M("c-column-order", "S1", "the cell of column j is read from the mirror column", [e(COLS, "let v = json.get(body, tape, obj, name(cols, j));", "let v = json.get(body, tape, obj, name(cols, cols.count - 1 - j));")], S1),
    M("c-null-as-zero", "S1", "a null is written as the cell 0", [e(COLS, "        if v >= 0 && !json.is_null(tape, v) {", "        if v >= 0 && json.is_null(tape, v) {\n            o = buffer.push(heap, o, byte_of(48));\n        }\n        if v >= 0 && !json.is_null(tape, v) {")], S1),
    M("c-a-comma-is-not-quoted", "S1", "a category with a comma or a quote is not quoted", [e(COLS, "if c == 44 || c == 34 || c == 10 || c == 13 {", "if c == 10 || c == 13 {")], S1),
    M("c-a-quote-is-not-doubled", "S1", "a quote inside a quoted category is not doubled", [e(COLS, "        if int_of(text[i]) == 34 {\n            o = buffer.push(heap, o, byte_of(34));\n        }", "        if false {\n            o = buffer.push(heap, o, byte_of(34));\n        }")], S1),
    M("c-header-names-not-quoted", "S2", "a column name with a comma is not quoted in the header", [e(COLS, "    h = put_field(heap, h, field);", "    h = buffer.append(heap, h, field);")], ["test_serve.Kinds.test_column_names_with_a_comma_a_quote_and_a_slash"]),
    M("c-pointer-slash-not-escaped", "S2", "a / in a column name is not escaped in a pointer", [e(COLS, "        if ch == 126 {\n            o = buffer.append(heap, o, \"~0\");\n        } else if ch == 47 {\n            o = buffer.append(heap, o, \"~1\");", "        if ch == 126 {\n            o = buffer.append(heap, o, \"~0\");\n        } else if ch == 1047 {\n            o = buffer.append(heap, o, \"~1\");")],
      ["test_serve.Kinds.test_column_names_with_a_comma_a_quote_and_a_slash"]),
    M("c-duplicate-keys-pass", "S2", "a key twice in a record is not seen", [e(COLS, "return json.count(tape, obj) == cols.count + present;", "return true;")], ["test_serve.Refusals.test_duplicate_keys"]),
    M("c-crlf-in-a-category-passes", "S2", "a category with CR LF is written as it is", [e(COLS, "if int_of(d[i]) == 13 && int_of(d[i + 1]) == 10 {", "if false {")], ["test_serve.Kinds"]),
    M("c-nul-in-a-category-refused", "S2", "a category with U+0000 is refused", [e(COLS, "if int_of(d[i]) == 13 && int_of(d[i + 1]) == 10 {", "if int_of(d[i]) == 13 && int_of(d[i + 1]) == 10 || int_of(d[i]) == 0 {")], ["test_serve.Kinds"]),
    M("c-an-empty-string-is-text", "S1", "an empty category is written as two quotes", [e(COLS, "    if n <= 0 {\n        return (out, rendered());\n    }\n    var o = out;", "    if n <= 0 {\n        return (buffer.append(heap, out, \"\\\"\\\"\"), rendered());\n    }\n    var o = out;")], S1,
      why="the reader reads two quotes as an empty cell, which is a missing cell, exactly as it reads nothing between two commas: the two spellings are one cell, an equivalent mutant"),
    # ------------------------------------------------------------------ S1, S2, O1: the schema and the document
    M("a-no-enum", "O1", "a complete vocabulary is not the enum of its column", [e(API, "if ty == 3 && enums {", "if ty == 3 && enums && false {")], ["test_serve.Kinds.test_categories_are_a_listed_vocabulary"], cmds=[["python3", "scripts/gate_serve_openapi.py"]]),
    M("a-always-nullable", "S2", "every column takes null", [e(API, "                if nullable {\n                    s = schema.make_nullable(s, node);", "                if true {\n                    s = schema.make_nullable(s, node);")], ["test_serve.Endpoints.test_null_where_the_model_does_not_take_it"], cmds=[["python3", "scripts/gate_serve_openapi.py"]]),
    M("a-never-nullable", "S2", "no column takes null", [e(API, "    let nullable = is_text(desc, tape, request, \"missing_cells\", \"handled\");", "    let nullable = false;")], ["test_serve.Kinds.test_null_where_the_model_takes_it"], cmds=[["python3", "scripts/gate_serve_openapi.py"]]),
    M("a-extra-columns-ignored", "S2", "a key that is not a column is accepted", [e(API, "let (s1, rec) = schema.new_object(heap, s, true);", "let (s1, rec) = schema.new_object(heap, s, false);")], ["test_serve.Endpoints.test_every_error_at_once_with_its_pointer"], cmds=[["python3", "scripts/gate_serve_openapi.py"]]),
    M("a-int-bound-one-wider", "S2", "an int column takes 10**12", [e(API, "schema.new_int(heap, s, 0 - int_limit(), int_limit());", "schema.new_int(heap, s, 0 - int_limit(), int_limit() + 1);")], ["test_serve.Endpoints.test_numbers_that_cannot_be_rendered_exactly"], cmds=[["python3", "scripts/gate_serve_openapi.py"]]),
    M("a-batch-limit-in-the-document", "O1", "the document's maxItems is one more than the limit", [e(API, "schema.new_array(heap, s, rec, 1, max_rows);", "schema.new_array(heap, s, rec, 1, max_rows + 1);")], ["test_serve.Endpoints.test_openapi_is_generated_from_the_model"], cmds=[["python3", "scripts/gate_serve_openapi.py"]]),
    M("a-health-needs-the-token", "S2", "GET /health is not exempt from the token", [e(API, "    if guarded {\n        api = web.no_auth(heap, api, health_op);\n    }", "    if guarded {\n        api = web.require(heap, api, health_op, \"token\");\n    }")], ["test_serve.Token"], cmds=[["python3", "scripts/gate_serve_openapi.py"]]),
    M("a-the-sha-is-not-in-the-document", "O1", "the document does not name the model's digest", [e(API, '    about = buffer.append(heap, about, json.string_view(desc, tape, sha_node));\n', "")], ["test_serve.Endpoints.test_openapi_is_generated_from_the_model"], cmds=[["python3", "scripts/gate_serve_openapi.py"]]),
    M("a-problem-lacks-rule", "O1", "the Problem component does not name the rule", [e(API, '    s = schema.add_field(heap, s, problem, "rule", text, false);\n', "")], ["test_serve.Refusals"], cmds=[["python3", "scripts/gate_serve_openapi.py"]]),
    # ------------------------------------------------------------------ S2: the handler
    M("s-media-type-unchecked", "S2", "the Content-Type is not checked", [e(SRV, 'if !is_json_type(http.header(request, table, "content-type")) {', "if false {")], ["test_serve.Refusals.test_content_type"]),
    M("s-batch-one-over", "S2", "a batch of limit + 1 is served", [e(SRV, "json.count(tape, rows) > sv.max_rows {", "json.count(tape, rows) > sv.max_rows + 1 {")], ["test_serve.Endpoints.test_predict_batch_at_the_limit_and_over"]),
    M("s-batch-one-short", "S2", "a batch of the limit is refused", [e(SRV, "json.count(tape, rows) > sv.max_rows {", "json.count(tape, rows) >= sv.max_rows {")], ["test_serve.Endpoints.test_predict_batch_at_the_limit_and_over"]),
    M("s-body-limit-one-short", "S2", "a body of exactly the limit is refused", [e(SRV, "if len(body) > sv.max_body {", "if len(body) >= sv.max_body {")], ["test_serve.Refusals.test_a_body_over_the_limit"]),
    M("s-body-limit-unchecked", "S2", "the handler does not check the body's length", [e(SRV, "if len(body) > sv.max_body {", "if len(body) > 999999999 {")], ["test_serve.Refusals.test_a_body_over_the_limit"]),
    M("s-rule-is-empty", "S2", "a refusal says no rule", [e(SRV, '    w = json.put_key(heap, w, "rule");\n    w = json.put_string(heap, w, rule);', '    w = json.put_key(heap, w, "rule");\n    w = json.put_string(heap, w, "");')], ["test_serve.Refusals", "test_serve.Endpoints"]),
    M("s-the-digest-is-not-in-a-refusal", "S2", "a refusal carries no model_sha256 field", [e(SRV, '    w = json.put_key(heap, w, "model_sha256");\n    w = json.put_string(heap, w, sha);\n    w = json.put_key(heap, w, "detail");', '    w = json.put_key(heap, w, "model_sha256");\n    w = json.put_string(heap, w, "");\n    w = json.put_key(heap, w, "detail");')], ["test_serve_bytes", "test_serve.Refusals"]),
    M("s-the-digest-header-is-dropped", "S2", "no X-Model-Sha256 header", [e(SRV, "marked = buffer.append(heap, marked, buffer.bytes(sv.sha_header));", 'marked = buffer.append(heap, marked, "");')], ["test_serve.Endpoints", "test_serve_bytes"]),
    M("s-the-digest-field-is-wrong", "S1", "the model_sha256 of an answer is the zero digest", [e(SRV, 'w = json.put_string(heap, w, sha);\n                w = json.put_key(heap, w, "model");', 'w = json.put_string(heap, w, "0");\n                w = json.put_key(heap, w, "model");')], S1),
    M("s-unprocessable-is-500", "S2", "a refusal of the predictor for its data is a 500", [e(SRV, 'ml.limit.categorical-columns,ml.predict.out-of-range", rule) {\n        return 422;', 'ml.limit.categorical-columns,ml.predict.out-of-range", rule) {\n        return 500;')], ["test_serve.Kinds.test_a_refusal_of_the_predictor_is_the_command_lines_rule_with_its_status"]),
    M("s-too-large-is-422", "S9", "the refusals of size are 422", [e(SRV, 'ml.input.too-many-cells", rule) {\n        return 413;', 'ml.input.too-many-cells", rule) {\n        return 422;')], ["test_serve.Statuses"]),
    M("s-a-rule-in-the-source-and-not-in-the-document", "S9", "a rule is added to the 422 list of the source", [e(SRV, "ml.input.unknown-column,ml.limit.string-bytes", "ml.input.unknown-column,ml.fit.non-finite,ml.limit.string-bytes")], ["test_serve.Statuses"]),
    M("s-guard-skipped", "S2", "web.guard is not called", [e(SRV, "    if sv.guarded {\n        let (g2, allowed)", "    if false {\n        let (g2, allowed)")], ["test_serve.Token"]),
    M("s-root-duplicate-unseen", "S2", "a second rows in a batch is not seen", [e(SRV, "if json.count(tape, 0) != 1 {", "if false {")], ["test_serve.Refusals.test_duplicate_keys"]),
    M("s-the-page-is-one-row", "S1", "the predictor is asked for one row of a batch", [e(SRV, "max_bytes: max_bytes, limit: limit, from: 0", "max_bytes: max_bytes, limit: 1, from: 0")], S1),
    M("s-no-probability", "S1", "a classification is asked without its probability", [e(SRV, "proba: proba, distance: distance,", "proba: false, distance: distance,")], S1),
    M("s-no-distance", "S1", "a clustering is asked without its distance", [e(SRV, "proba: proba, distance: distance,", "proba: proba, distance: false,")], S1),
    M("s-body-json-error-is-200", "S2", "a body that is not JSON is judged as an empty record", [e(SRV, "        if nodes < 0 {\n            var detail", "        if nodes < 0 && false {\n            var detail")], ["test_serve.Refusals.test_invalid_json"]),
    M("s-non-finite-status-is-500", "S2", "the rule ml.predict.non-finite is not in the 422 list of status_of: a row that is not finite is a server fault", [e(SRV, 'if app.listed("ml.predict.non-finite,ml.schema.mismatch,', 'if app.listed("ml.schema.mismatch,')], ["test_serve_json.Http"]),
    M("s-non-finite-guard-dropped", "S2", "a document with a cell that is not a number is written as text", [e(SRV, "                    if good && !num.is_json_number(json.string_view(payload, tp, cell)) {\n                        numeric = false;\n                    }", "                    if good && false {\n                        numeric = false;\n                    }")], ["test_serve_json.Http"],
      why="a guard that no request reaches since the predictor refuses a row that is not finite before it writes a cell (docs/nonfinite.md section 4); the refusal itself is judged by s-non-finite-status-is-500 and by the mutants of scripts/mutants_nonfinite.py"),
    M("s-answer-limit", "S2", "an answer past 60,000 bytes is sent", [e(SRV, "if made == 0 && len(buffer.bytes(bo)) <= answer_limit() {", "if made == 0 {")], ["test_serve"],
      why="the check is a backstop: the ceilings (1,000 rows of at most about 56 bytes) keep an answer under 60,000 bytes, so no request reaches it; an equivalent mutant of the tests, not of the design"),
    M("s-unknown-count-dropped", "S1", "unknown_categories is always 0", [e(SRV, "unseen = json.to_int(payload, tp, unknown);", "unseen = 0;")], S1,
      why="a vocabulary the file lists in full is an enum, so a category the model has not seen is refused before the predictor counts it; the count is nonzero only for a hash column, which counts none"),
    M("s-out-of-range-dropped", "S1", "the answer's out_of_range is always null", [e(SRV, "                if oor >= 0 {\n                    w = track.put_node(heap, w, payload, tp, oor);", "                if false {\n                    w = track.put_node(heap, w, payload, tp, oor);")], ["test_serve_bytes", "test_track.Serve"]),
    M("s-out-of-range-bound-ignored", "S1", "--max-out-of-range is not passed to the predictor", [e(SRV, "sv.max_oor, sv.max_oor_given);", "0, false);")], ["test_track.Serve"]),
    # ------------------------------------------------------------------ S2: the process
    M("m-all-needs-no-token", "S10", "--bind all starts with neither --token-file nor --no-token", [e(MAIN, "} else if everywhere && !argv.has(src, \"token-file\") {", "} else if false {")], B),
    M("m-all-accepts-no-token", "S10", "--bind all --no-token is served: an open server on every address", [e(MAIN, "} else if everywhere && argv.has(src, \"no-token\") {", "} else if false {")], B),
    M("m-the-token-rule-is-everywhere", "S10", "the default (loopback) server still wants a token, as it did before the loopback default", [e(MAIN, "binds_everywhere(src);\n    if argv.has(src, \"token-file\") && argv.has(src, \"no-token\") {", "true;\n    if argv.has(src, \"token-file\") && argv.has(src, \"no-token\") {")], B),
    M("b-the-default-is-all", "S10", "the default of --bind is all (flag table)", [e("tools/ml-serve/sspec.cho", "bind||choice:loopback/all|none|loopback|", "bind||choice:loopback/all|none|all|")], B),
    M("b-all-is-the-default-in-the-code", "S10", "binds_everywhere is true whatever the flag says", [e(MAIN, 'return bytes.equal(argv.text(src, "bind"), "all");', 'return bytes.equal(argv.text(src, "bind"), "all") || true;')], B),
    M("b-all-binds-loopback", "S10", "--bind all listens on 127.0.0.1", [e(MAIN, 'return tcp_listen_on(net, "0.0.0.0", port, 1024, 0);', 'return tcp_listen_on(net, "127.0.0.1", port, 1024, 0);')], B),
    M("b-loopback-binds-all", "S10", "the default listens on 0.0.0.0", [e(MAIN, 'return tcp_listen_on(net, "127.0.0.1", port, 1024, 0);', 'return tcp_listen_on(net, "0.0.0.0", port, 1024, 0);')], B),
    M("b-the-host-literal-is-another-loopback-address", "S10", "the default listens on 127.0.0.2, not 127.0.0.1", [e(MAIN, 'return tcp_listen_on(net, "127.0.0.1", port, 1024, 0);', 'return tcp_listen_on(net, "127.0.0.2", port, 1024, 0);')], B),
    M("b-the-start-up-line-lies", "S10", "the start-up line names 127.0.0.1 for --bind all", [e(MAIN, 'host = "0.0.0.0";', 'host = "127.0.0.1";')], B),
    M("b-the-old-call", "S10", "the old tcp_listen (every address) stands in for loopback", [e(MAIN, 'return tcp_listen_on(net, "127.0.0.1", port, 1024, 0);', 'return tcp_listen(net, port, 1024, 0);')], B),
    M("m-the-token-keeps-its-newline", "S2", "the line feed of the token file is part of the token", [e(MAIN, "if len(text) > 0 && int_of(text[len(text) - 1]) == 10 {", "if false {")], ["test_serve.Token"]),
    M("m-the-idle-time-is-a-thousand-times-longer", "S4", "the idle timeout is in milliseconds", [e(MAIN, "var loop_server = server.open(heap, p, listener, size, 0, idle);", "var loop_server = server.open(heap, p, listener, size, 0, idle * 1000);")], ["test_serve_fuzz.Fuzz.test_a_connection_that_sends_nothing_is_closed"]),
    M("m-no-room-for-a-head", "S2", "the buffer of a connection holds the body and no head", [e(MAIN, 'let size = argv.nat(src, "max-body-bytes") + 16384;', 'let size = argv.nat(src, "max-body-bytes");')], ["test_serve.Refusals.test_a_body_over_the_limit"]),
    M("m-batch-flag-ignored", "S2", "--max-batch-rows is read as 100", [e(MAIN, 'let max_rows = argv.nat(src, "max-batch-rows");', "let max_rows = 100;")], ["test_serve.Endpoints.test_predict_batch_at_the_limit_and_over"]),
    M("m-body-flag-ignored", "S2", "--max-body-bytes is read as 1 MiB", [e(MAIN, 'let max_body = argv.nat(src, "max-body-bytes");', "let max_body = 1048576;")], ["test_serve.Refusals.test_a_body_over_the_limit"]),
    M("m-the-model-is-not-checked-at-start", "S2", "a damaged model file is served", [e(MAIN, "    if clean {\n        borrow root as &rr in {\n            srv.drop(heap, served);", "    if clean && false {\n        borrow root as &rr in {\n            srv.drop(heap, served);")], ["test_serve.Start"]),
    M("m-listen-error-is-ignored", "S2", "a port in use is not refused", [e(MAIN, '"serve.listen", "the port cannot be bound"', '"serve.listen", "the port cannot be bound"'.replace("serve.listen", "serve.internal"))], ["test_serve.Start.test_a_port_in_use"]),
    # ------------------------------------------------------------------ S4: the fuzz can see a trap
    M("f-a-trap-for-a-long-body", "S4", "a body of more than 3,000 bytes traps the server", [e(SRV, "    if len(body) > sv.max_body {\n        return simple", "    if len(body) > 3000 && len(body) <= sv.max_body {\n        let trap_now = trap();\n    }\n    if len(body) > sv.max_body {\n        return simple")], ["test_serve_fuzz.Fuzz.test_no_trap_no_hang"]),
    M("f-a-hang-in-a-path", "S4", "a request for /health?x hangs the server for good", [e(SRV, "    if id == api.op_health() {\n        return server.reply_with", "    if id == api.op_health() && len(http.query(request, table)) > 0 {\n        var spin = 0;\n        while spin >= 0 {\n            spin = spin + 0;\n        }\n    }\n    if id == api.op_health() {\n        return server.reply_with")],
      ["test_serve.Endpoints.test_health_and_the_digest", "test_serve.Transport.test_the_log_has_no_data"]),
    # ------------------------------------------------------------------ A1: the authority
    M("x-a-file-is-written", "A1", "a reachable function appends to a log", [
        e(MAIN, 'fn read_file[&h, &f, &r, &pr](heap: &!h Heap, fs: &f Fs(""), root: &r [byte], p: &pr paths.Rp, cap: int, errs: fail.Errors) -> [heap, fs_read(""), dir_read, file_read] (buffer.Buffer, bool, fail.Errors) {',
          'fn read_file[&h, &f, &r, &pr](heap: &!h Heap, fs: &f Fs(""), root: &r [byte], p: &pr paths.Rp, cap: int, errs: fail.Errors) -> [heap, fs_read(""), dir_read, dir_write, file_read, file_write] (buffer.Buffer, bool, fail.Errors) {\n    let (ws, wa) = fsio.append_log(fs, root, "x", "x", "x", 10);'),
        e(MAIN, 'errs: fail.Errors) -> [heap, fs_read(""), dir_read, file_read] (srv.Served, fail.Errors) {', 'errs: fail.Errors) -> [heap, fs_read(""), dir_read, dir_write, file_read, file_write] (srv.Served, fail.Errors) {'),
        e(MAIN, "-> [heap, fs_read(\"\"), dir_read, file_read, io_write, err_write, net_in(\"\"), conn_accept, conn_read, conn_write, poll, clock] int {", "-> [heap, fs_read(\"\"), dir_read, dir_write, file_read, file_write, io_write, err_write, net_in(\"\"), conn_accept, conn_read, conn_write, poll, clock] int {")],
      kind="authority"),
    M("x-the-network-is-dialled", "A1", "the server connects out", [
        e(MAIN, "        match open_listener(net, everywhere, port) {", "        match tcp_connect(net, \"127.0.0.1\", 9) {\n            Dialed::Ok(dc) => {\n                conn_close(dc);\n            }\n            Dialed::Failed(dcode) => {\n            }\n        }\n        match open_listener(net, everywhere, port) {"),
        e(MAIN, "-> [heap, fs_read(\"\"), dir_read, file_read, io_write, err_write, net_in(\"\"), conn_accept, conn_read, conn_write, poll, clock] int {", "-> [heap, fs_read(\"\"), dir_read, file_read, io_write, err_write, net_in(\"\"), net_out(\"\"), conn_accept, conn_read, conn_write, poll, clock] int {")],
      kind="authority"),
    M("x-the-ceiling-lacks-a-reason", "A1", "tools.toml gives no reason for the clock", [e("tools.toml", 'clock = "http.server closes', 'clockX = "http.server closes')], kind="authority"),
    M("x-the-ceiling-allows-a-write", "A1", "the ceiling of ml-serve allows file_write while the row has none (the ceiling must be exact)", [e("tools.toml", 'allow = ["args", "clock", "conn_accept"', 'allow = ["args", "clock", "file_write", "conn_accept"')], ["test_authority.AuthorityServe"]),
    M("x-the-manifest-is-stale", "A1", "the committed row has lost poll", [e("manifests/ml-serve.authority.json", '    {\n      "name": "poll",\n      "argument": null,\n      "bounded": true\n    }', '    {\n      "name": "pollx",\n      "argument": null,\n      "bounded": true\n    }')], kind="check"),
    # ------------------------------------------------------------------ W1: the pins
    M("w-one-rev-moved", "W1", "the schema rev differs from the documented one", [e("cancho.toml", 'rev = "59a70285ff23af63a226bc56ff978feccb4cf8ca"', 'rev = "59a70285ff23af63a226bc56ff978feccb4cf8cb"')], kind="pin"),
    M("w-a-block-is-missing", "W1", "the http-server dependency is removed", [e("cancho.toml", '[dependencies.server]\ngit = "https://github.com/alpibrusl/cancho"\nrev = "0207556b4029a3ee73196cdfdb48e8efa7e9c519"\npath = "packages/http-server/.cancho-vcs"\n', "")], kind="pin"),
    M("w-another-store", "W1", "a store of cancho-web that ml-serve does not import is listed", [e("cancho.toml", "[dependencies.schema]", '[dependencies.users]\ngit = "https://github.com/alpibrusl/cancho-web"\nrev = "5f404a777018cf42b5c713daa4972c938c9229f4"\npath = ".cancho-vcs"\n\n[dependencies.schema]')], kind="pin"),
    M("w-another-origin", "W1", "web comes from another repository", [e("cancho.toml", 'git = "https://github.com/alpibrusl/cancho-web"', 'git = "https://github.com/someone/cancho-web"')], kind="pin"),
    M("w-the-document-names-another-rev", "W1", "docs/serving-http.md names another web rev", [e("docs/serving-http.md", "cancho-web rev: `5f404a777018cf42b5c713daa4972c938c9229f4`", "cancho-web rev: `5f404a777018cf42b5c713daa4972c938c9229f0`")], kind="pin"),
    M("w-server-rev-is-not-the-compiler", "W1", "http-server is pinned to another commit than the compiler", [e("cancho.toml", 'rev = "0207556b4029a3ee73196cdfdb48e8efa7e9c519"\npath = "packages/http-server/.cancho-vcs"', 'rev = "0207556b4029a3ee73196cdfdb48e8efa7e9c518"\npath = "packages/http-server/.cancho-vcs"')], kind="pin"),
    M("w-web-is-imported-elsewhere", "W1", "tools/ml imports web", [e("tools/ml/paths.cho", "import std.buffer;", "import std.buffer;\nimport web;")], kind="pin"),
    M("w-a-copy-of-schema", "W1", "a file under lib declares module schema", [], kind="pin", create=[("lib/core/schema_copy.cho", "module schema;\n")]),
    M("w-a-copy-of-the-server", "W1", "a file under tools declares module http.server", [], kind="pin", create=[("tools/ml-serve/server_copy.cho", "module http.server;\n")]),
    M("w-bins-differ", "W2", "build/ml is not the bytes a build without the three dependencies makes (a byte of the binary is changed)", [], kind="bins", flip=("build/ml", 5000)),
]


def build_serve(tree):
    g = lambda pat: sorted(glob.glob(str(tree / pat)))
    libs = (g("lib/core/*.cho") + g("lib/linalg/*.cho") + g("lib/metrics/*.cho") + g("lib/models/*.cho") + g("lib/models/boost/*.cho") + g("lib/models/kmeans/*.cho") + g("lib/models/linear/*.cho") +
            g("lib/models/logistic/*.cho") + g("lib/models/tree/*.cho") + g("lib/models/treewire/*.cho") + g("lib/transforms/*.cho") + g("lib/pipeline/*.cho") + g("build/deps/*.cho"))
    shared = [f for f in g("tools/ml/*.cho") if not f.endswith("/main.cho")]
    files = libs + [str(tree / "generated/ml-serve/built.cho")] + shared + g("tools/ml-serve/*.cho")
    target = tree / "build" / "ml-serve"
    target.parent.mkdir(exist_ok=True)
    p = subprocess.run([base.compiler(), "build", "--std", *files, "-o", str(target)], capture_output=True, text=True, cwd=tree)
    return (target if p.returncode == 0 else None), p.stdout + p.stderr


def build_check(tree):
    files = [str(tree / "tools/ml-serve/num.cho"), str(tree / "tools/serve-check/main.cho")] + sorted(glob.glob(str(tree / "build/deps/*.cho")))
    target = tree / "build" / "serve-check"
    p = subprocess.run([base.compiler(), "build", "--std", *files, "-o", str(target)], capture_output=True, text=True, cwd=tree)
    return (target if p.returncode == 0 else None), p.stdout + p.stderr


def env_for(tree):
    return dict(os.environ, ML_BIN=str(ROOT / "build" / "ml"), ML_PREDICT_BIN=str(ROOT / "build" / "ml-predict"), ML_SERVE_BIN=str(tree / "build" / "ml-serve"), SERVE_CHECK_BIN=str(tree / "build" / "serve-check"),
                PYTHONDONTWRITEBYTECODE="1", CANCHO=base.compiler(), SERVE_FUZZ_N="1200")


def run_tests(tree, targets):
    p = subprocess.run([sys.executable, "-m", "unittest", *targets], capture_output=True, text=True, cwd=tree / "tests" / "conformance", env=env_for(tree))
    return p.returncode, p.stdout + p.stderr


def run_one(m):
    with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        base.copy_tree(tree)
        problems = base.apply_edits(tree, m)
        if problems:
            return "INVALID", "; ".join(problems)
        for path, text in m["opt"].get("create", []):
            (tree / path).write_text(text)
        kind = m["opt"].get("kind", "")
        if kind == "authority":
            r = subprocess.run([sys.executable, str(tree / "scripts" / "manifest.py"), "--tree", str(tree)], capture_output=True, text=True, cwd=tree)
            if r.returncode != 0:
                return "KILLED", "manifest.py --tree: " + " | ".join(l for l in r.stdout.splitlines() if l.startswith("FAIL"))[:300]
            return "SURVIVED", ""
        if kind == "check":
            r = subprocess.run([sys.executable, str(tree / "scripts" / "manifest.py"), "--check"], capture_output=True, text=True, cwd=tree)
            return ("KILLED", "manifest.py --check: " + " | ".join(l for l in r.stdout.splitlines() if l.startswith("FAIL"))[:300]) if r.returncode != 0 else ("SURVIVED", "")
        if kind == "pin":
            r = subprocess.run([sys.executable, str(tree / "scripts" / "check_web_pin.py")], capture_output=True, text=True, cwd=tree)
            return ("KILLED", "check_web_pin.py: " + r.stderr.strip()[:300]) if r.returncode != 0 else ("SURVIVED", "")
        if kind == "bins":
            (tree / "build").mkdir(exist_ok=True)
            for b in ("ml", "ml-predict"):
                shutil.copy(ROOT / "build" / b, tree / "build" / b)
            fname, at = m["opt"]["flip"]
            data = bytearray((tree / fname).read_bytes())
            data[at] ^= 1
            (tree / fname).write_bytes(bytes(data))
            r = subprocess.run([sys.executable, str(tree / "scripts" / "check_web_pin.py"), "--bins"], capture_output=True, text=True, cwd=tree, env=env_for(tree))
            return ("KILLED", "check_web_pin.py --bins: " + (r.stderr.strip() + r.stdout.strip())[:300]) if r.returncode != 0 else ("SURVIVED", "")
        built, log = build_serve(tree)
        if built is None:
            return "INVALID", "build failed: " + log[:600]
        if m["opt"].get("check_bin"):
            cb, log2 = build_check(tree)
            if cb is None:
                return "INVALID", "serve-check build failed: " + log2[:600]
        if m["tests"]:
            code, out = run_tests(tree, m["tests"])
            if code != 0:
                return "KILLED", "tests: " + ", ".join(base.failing_tests(out)[:4]) + ("" if base.failing_tests(out) else out.strip().splitlines()[-1][:200])
        for cmd in m["cmds"]:
            argv = [sys.executable if a == "python3" else a for a in cmd]
            r = subprocess.run(argv, capture_output=True, text=True, cwd=tree, env=env_for(tree))
            if r.returncode != 0:
                return "KILLED", "%s: %s" % (cmd[-1], (r.stdout + r.stderr).strip().splitlines()[0][:200] if (r.stdout + r.stderr).strip() else "exit %d" % r.returncode)
        return "SURVIVED", ""


def baseline():
    with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
        tree = pathlib.Path(td) / "tree"
        base.copy_tree(tree)
        built, log = build_serve(tree)
        if built is None:
            return "INVALID", "build failed: " + log[:600]
        cb, log2 = build_check(tree)
        if cb is None:
            return "INVALID", "serve-check build failed: " + log2[:600]
        code, out = run_tests(tree, ["test_serve", "test_serve_bytes", "test_serve_json", "test_authority"])
        if code != 0:
            return "FAILED", out[-800:]
        for cmd in (["gate_serve_openapi.py"], ["check_web_pin.py"]):
            r = subprocess.run([sys.executable, str(tree / "scripts" / cmd[0])], capture_output=True, text=True, cwd=tree, env=env_for(tree))
            if r.returncode != 0:
                return "FAILED", cmd[0] + ": " + (r.stdout + r.stderr)[-600:]
        return "PASSED", ""


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        for m in MUTANTS:
            print("%-52s %s  %s" % (m["id"], m["gate"], m["what"]))
        return 0
    if "--check" in argv:
        bad = 0
        with tempfile.TemporaryDirectory(prefix="ml-mutant-") as td:
            tree = pathlib.Path(td) / "tree"
            base.copy_tree(tree)
            for m in MUTANTS:
                problems = base.apply_edits(tree, m)
                if problems:
                    print("INVALID %s: %s" % (m["id"], "; ".join(problems)))
                    bad += 1
                for item in m["edits"]:
                    shutil.copy(ROOT / item[0], tree / item[0])
        ids = [m["id"] for m in MUTANTS]
        if len(ids) != len(set(ids)):
            print("a mutant is named twice")
            bad += 1
        print("%d mutants, %d with an edit that does not apply" % (len(MUTANTS), bad))
        return 1 if bad else 0
    only = [a for a in argv[argv.index("--only") + 1:] if not a.startswith("--")] if "--only" in argv else None
    results = []
    for m in MUTANTS:
        if only and m["id"] not in only:
            continue
        verdict, detail = run_one(m)
        results.append((m, verdict, detail))
        print("%-9s %-52s %s  %s" % (verdict, m["id"], m["gate"], detail), flush=True)
    if not only:
        v, d = baseline()
        print("%-9s %-52s %s  %s" % (v, "baseline (the unchanged tree)", "-", d), flush=True)
    survivors = [(m, d) for m, v, d in results if v == "SURVIVED"]
    invalid = [m["id"] for m, v, d in results if v == "INVALID"]
    killed = sum(1 for m, v, d in results if v == "KILLED")
    print("\n%d mutants: %d killed, %d survived, %d invalid" % (len(results), killed, len(survivors), len(invalid)))
    for m, d in survivors:
        print("SURVIVOR %s: %s" % (m["id"], m["opt"].get("why", "NO REASON GIVEN")))
    unexplained = [m["id"] for m, d in survivors if not m["opt"].get("why")]
    return 1 if (unexplained or invalid) else 0


if __name__ == "__main__":
    sys.exit(main())

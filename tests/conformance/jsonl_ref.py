"""The reference of the JSON lines reader (docs/jsonl.md sections 3 and 4), written from the specification with Python's `json` and nothing of `ml`'s.

`expected(data, fields, ...)` answers what `jsonl-check` (tools/jsonl-check) must print for the bytes `data`: ("ok", the CSV bytes) or ("refused", (code, line, row,
col, a, b, c)). The codes are those of docs/jsonl.md section 4 / tools/ml/jsonl.cho:

    1 syntax (a: offset, not compared; b: the parser's code, not compared)   2 not an object (a: 0 array, 1 string, 2 number, 3 boolean, 4 null)
    3 too many nodes                 4 missing field (col)                    5 not scalar (col; a: 0 object, 1 array)
    6 duplicate key (col)            7 type mismatch (col; a: 1 string, 2 boolean)
    8 mixed kinds (col; a: kind of the first value, b: its line, c: the kind that differs)
    9 line too long (a: the length)  10 too large (a: the cap, b: 0 the input, 1 the CSV)

kinds: 0 number, 1 string, 2 boolean. Only the standard library: this module is imported by tests that the fast CI job runs.
"""
import json


class Pairs(list):
    """an object as its pairs, in file order, duplicates kept"""


def _constant(name):
    raise ValueError("constant " + name)


def _no_lone_surrogate(s):
    for ch in s:
        if 0xD800 <= ord(ch) <= 0xDFFF:
            return False
    return True


def _walk(v, depth, state):
    """node count and depth; refuses a lone surrogate in a string or a key"""
    state["nodes"] += 1
    if depth > state["depth"]:
        state["depth"] = depth
    if isinstance(v, Pairs):
        for k, x in v:
            state["nodes"] += 1
            if not _no_lone_surrogate(k):
                raise ValueError("lone surrogate")
            _walk(x, depth + 1, state)
    elif isinstance(v, list):
        for x in v:
            _walk(x, depth + 1, state)
    elif isinstance(v, str):
        if not _no_lone_surrogate(v):
            raise ValueError("lone surrogate")


def parse_line(line, max_nodes):
    """(value, None) or (None, 'syntax' | 'nodes')"""
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError:
        return None, "syntax"
    try:
        v = json.loads(text, object_pairs_hook=Pairs, parse_float=str, parse_int=str, parse_constant=_constant)
    except (ValueError, RecursionError):
        return None, "syntax"
    state = {"nodes": 0, "depth": 0}
    try:
        _walk(v, 1, state)
    except (ValueError, RecursionError):
        return None, "syntax"
    if state["depth"] > 128:
        return None, "syntax"
    if state["nodes"] > max_nodes:
        return None, "nodes"
    return v, None


def kind_of(v):
    if isinstance(v, Pairs):
        return "object"
    if isinstance(v, list):
        return "array"
    if isinstance(v, str):
        return "string"
    if v is True or v is False:
        return "boolean"
    if v is None:
        return "null"
    return "number"  # unreachable with parse_int / parse_float = str: numbers are str too, see _is_number


def csv_cell(text):
    if any(c in text for c in (",", '"', "\r", "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text


class Num(str):
    pass


def _tag_numbers(text, max_nodes):
    """parse_int / parse_float = str gives a str for a number and for a string alike: a second parse tells them apart with Num"""
    return json.loads(text, object_pairs_hook=Pairs, parse_float=Num, parse_int=Num, parse_constant=_constant)


def expected(data, fields, cats=(), absent_empty=False, max_nodes=4096, max_line=1048576, max_bytes=16777216):
    """see the module's docstring"""
    cats = set(cats)
    out = [",".join(csv_cell(f) for f in fields).encode("utf-8") + b"\n"]
    size = len(out[0])
    if len(data) > max_bytes:
        return "refused", (10, 0, 0, 0, max_bytes, 0, 0)
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    kinds = {}  # field -> (kind, line)
    lineno = 0
    row = 0
    pieces = data.split(b"\n")
    if pieces and pieces[-1] == b"":
        pieces.pop()
    for raw in pieces:
        lineno += 1
        if len(raw) > max_line:
            return "refused", (9, lineno, 0, 0, len(raw), 0, 0)
        if raw.strip(b" \t\r") == b"":
            out.append(b"\n")
            size += 1
            continue
        row += 1
        line = raw[:-1] if raw.endswith(b"\r") else raw
        v, why = parse_line(line, max_nodes)
        if why == "syntax":
            return "refused", (1, lineno, row, 0, 0, 0, 0)
        if why == "nodes":
            return "refused", (3, lineno, row, 0, max_nodes, 0, 0)
        v = _tag_numbers(line.decode("utf-8"), max_nodes)
        if not isinstance(v, Pairs):
            k = {list: 0, str: 1, Num: 2, bool: 3, type(None): 4}[type(v)]
            return "refused", (2, lineno, row, 0, k, 0, 0)
        seen = {}
        for key, val in v:
            if key in fields:
                if key in seen:
                    return "refused", (6, lineno, row, fields.index(key), 0, 0, 0)
                seen[key] = val
        cells = []
        for j, f in enumerate(fields):
            if f not in seen:
                if not absent_empty:
                    return "refused", (4, lineno, row, j, 0, 0, 0)
                cells.append("")
                continue
            val = seen[f]
            if val is None:
                cells.append("")
                continue
            if isinstance(val, Pairs):
                return "refused", (5, lineno, row, j, 0, 0, 0)
            if isinstance(val, list):
                return "refused", (5, lineno, row, j, 1, 0, 0)
            if isinstance(val, Num):
                kind, text = 0, str(val)
            elif isinstance(val, str):
                kind, text = 1, val
            else:
                kind, text = 2, "true" if val else "false"
            if f not in cats and kind != 0:
                return "refused", (7, lineno, row, j, kind, 0, 0)
            if f in kinds and kinds[f][0] != kind:
                return "refused", (8, lineno, row, j, kinds[f][0], kinds[f][1], kind)
            kinds.setdefault(f, (kind, lineno))
            cells.append(text)
        if len(fields) == 1 and cells[0] == "":
            rec = b'""\n'
        else:
            rec = (",".join(csv_cell(c) if c != "" or True else c for c in cells)).encode("utf-8") + b"\n"
        out.append(rec)
        size += len(rec)
        if size > max_bytes:
            return "refused", (10, lineno, row, 0, max_bytes, 1, 0)
    return "ok", b"".join(out)

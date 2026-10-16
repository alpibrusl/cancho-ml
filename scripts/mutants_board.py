#!/usr/bin/env python3
"""Mutants of the experiments board (docs/board.html, docs/board.md section 9): each fault is applied to a COPY of the page (or of assets/site.css), scripts/check_board.py is run
on the copy with the same fixtures, and the mutant is killed if it exits non-zero (and the gate that must catch it is among those that fail). A control, the unmutated page, must
pass first. A pattern must occur exactly once in its file.

    python3 scripts/mutants_board.py [--bin build/ml] [NAME ...]    # the mutants whose name contains one of the words (default: all)
    python3 scripts/mutants_board.py --check                        # only check that every pattern applies once (no gate run)

The runs the gates use are made once, by the built binary, in a temporary directory (scripts/check_board.py --make-fixtures). Exit 1 if the control fails, a pattern does not apply,
or a mutant survives without an explanation in EXPLAINED.
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
PAGE = "docs/board.html"
CSS = "docs/assets/site.css"

# (name, file, old, new, the gates, any of which must catch it)
MUTANTS = [
    # the table: the order
    ("sort-direction-flipped", PAGE, "return (a < b ? -1 : 1) * sign;", "return (a < b ? 1 : -1) * sign;", {"G6"}),
    ("sort-descending-ignored", PAGE, 'sign = dir === "desc" ? -1 : 1;', "sign = 1;", {"G6"}),
    ("sort-ties-by-position", PAGE, "      return idCmp(x.r, y.r) || x.i - y.i;\n    }).map", "      return x.i - y.i;\n    }).map", {"G6"}),
    ("sort-missing-first", PAGE, "return am && bm ? idCmp(x.r, y.r) || x.i - y.i : am ? 1 : -1;", "return am && bm ? idCmp(x.r, y.r) || x.i - y.i : am ? -1 : 1;", {"G6"}),
    ("sort-numbers-as-text", PAGE, "train_rows: function (r) { return r.trainRows; }", "train_rows: function (r) { return r.trainRows === null ? null : String(r.trainRows); }", {"G6"}),
    ("sort-metric-by-text", PAGE, "metric: function (r) { return r.metricValue; }", "metric: function (r) { return r.metricText; }", {"G6"}),
    ("filter-model-inverted", PAGE, "if (f.model && r.model !== f.model) { return false; }", "if (f.model && r.model === f.model) { return false; }", {"G6"}),
    ("filter-data-ignored", PAGE, "if (f.data && r.dataSha !== f.data) { return false; }", "if (false) { return false; }", {"G6"}),
    ("filter-input-ignored", PAGE, "if (f.key && f.value && inputText(r, f.key) !== f.value) { return false; }", "if (false) { return false; }", {"G6"}),
    ("varying-counts-one-value", PAGE, "      return n > 1;", "      return n > 0;", {"G6"}),
    ("input-text-keeps-newlines", PAGE, 'return typeof v === "string" ? v.replace(/\\n+$/, "").replace(/\\n/g, " ; ") : JSON.stringify(v);', 'return typeof v === "string" ? v.replace(/\\n+$/, "") : JSON.stringify(v);', {"G6"}),
    # the id and the check
    ("id-hashes-whole-record", PAGE, "out.calcId = await digest(enc(inRaw));", "out.calcId = await digest(enc(raw));", {"G2", "G3"}),
    ("id-hashes-compact-form-only", PAGE, "out.calcId = await digest(enc(inRaw));", "out.calcId = await digest(enc(JSON.stringify(o.inputs)));", {"G3"}),
    ("id-not-utf8", PAGE, "function enc(s) { return new TextEncoder().encode(s); }", "function enc(s) { return Uint8Array.from(s, function (c) { return c.charCodeAt(0) & 255; }); }", {"G2"}),
    ("id-mismatch-not-flagged", PAGE, "if (chk.idBad) {", "if (false) {", {"G3"}),
    ("check-mismatch-not-flagged", PAGE, "if (chk.checkBad) {", "if (false) {", {"G3"}),
    ("check-covers-one-byte-more", PAGE, "raw.slice(0, pos)", "raw.slice(0, pos + 1)", {"G2", "G3"}),
    ("check-of-eval-ignored", PAGE, "if (epos >= 0 && isHex(eo.check) && (await digest(", "if (false && (await digest(", {"G4"}),
    ("reformatted-called-forged", PAGE, "if (compact !== inRaw && (await digest(enc(compact))) === o.id) { out.reformatted = true; } else { out.idBad = true; }", "out.idBad = true;", {"G3"}),
    ("verdict-id-and-check-lost", PAGE, 'if (f["id-mismatch"] && f["check-mismatch"]) { return "id-check-mismatch"; }', 'if (f["id-mismatch"] && f["check-mismatch"]) { return "id-mismatch"; }', {"G3"}),
    ("name-mismatch-not-flagged", PAGE, "if (nm && nm[1] !== o.id) {", 'if (nm && nm[1] === "x") {', {"G3"}),
    ("malformed-not-flagged", PAGE, 'if (bad.length) { flags.push("malformed");', 'if (false) { flags.push("malformed");', {"G4"}),
    ("repeated-member-not-flagged", PAGE, ".length > 1) { bad.push(", ".length > 9) { bad.push(", {"G4"}),
    # what is read
    ("schema-v2-read-as-v1", PAGE, 'if (sc === "ml.run.v1") {', 'if (sc === "ml.run.v1" || sc === "ml.run.v2") {', {"G4"}),
    ("schema-version-not-flagged", PAGE, "if (/^ml\\.(run|eval)\\b/.test(sc)) {", "if (/^ml\\.(runs|evals)\\b/.test(sc)) {", {"G4"}),
    ("answer-schema-not-checked", PAGE, 'if (obj.schema !== "ml.v1") {', 'if (false) {', {"G4"}),
    ("duplicate-not-flagged", PAGE, "if (has(byId, shownId)) {", "if (false) {", {"G4"}),
    ("duplicate-difference-a-note", PAGE, 'if (!same) { st.problems[st.problems.length - 1].severity = "error"; }', 'if (!same) { st.problems[st.problems.length - 1].severity = "note"; }', {"G4"}),
    ("duplicate-eval-not-flagged", PAGE, "if (has(evHashes, ekey)) {", "if (false) {", {"G4"}),
    ("non-record-not-flagged", PAGE, 'if (!isObj(obj)) { P("not-a-record", Array.isArray(obj) ? "a JSON array, not a run record or an answer of ml runs" : "a JSON value that is not an object"); return; }', "if (!isObj(obj)) { return; }", {"G4"}),
    ("object-without-schema-not-flagged", PAGE, 'if (typeof sc !== "string") { P(', 'if (typeof sc !== "string") { void (', {"G4"}),
    ("refusal-not-flagged", PAGE, "if (obj.ok === false) {", "if (obj.ok === 0) {", {"G4"}),
    ("compare-answer-not-flagged", PAGE, 'if (isObj(obj.data) && has(obj.data, "inputs") && has(obj.data, "a") && has(obj.data, "b")) {', "if (false) {", {"G4"}),
    ("junk-not-reported", PAGE, "if (sp.junk.length) { P(", "if (false) { P(", {"G4"}),
    ("empty-not-reported", PAGE, 'if (text.replace(/^\\uFEFF/, "").trim() === "") {', 'if (text.replace(/^\\uFEFF/, "").trim() === "x") {', {"G4"}),
    ("cut-after-a-record-not-reported", PAGE, "if (sp.cut) { P(", "if (false) { P(", {"G4"}),
    ("show-answer-ignored", PAGE, 'if (isObj(inner.record) && (inner === obj.data || !has(inner, "schema"))) {', "if (false) {", {"G2", "G4"}),
    ("list-rows-ignored", PAGE, 'st.lists.push({ row: r, source: src.name, dir: typeof inner.dir === "string" ? inner.dir : null });', "void 0;", {"G2", "G4"}),
    ("list-disagreement-not-flagged", PAGE, "if (diffs.length) {", "if (false) {", {"G4"}),
    ("list-only-not-reported", PAGE, "if (listOnly.length) { P(", "if (false) { P(", {"G2", "G4"}),
    ("eval-orphan-not-reported", PAGE, "if (!has(byId, rid) && !has(listed, rid)) {", "if (false) {", {"G4"}),
    ("eval-count-not-compared", PAGE, "if (n > 0 && n !== row.listEvals) {", "if (n < 0 && n !== row.listEvals) {", {"G4"}),
    ("evals-counted-once", PAGE, "attached[eo.run] = (attached[eo.run] || 0) + 1;", "attached[eo.run] = 1;", {"G2"}),
    ("model-from-first-line", PAGE, "return m ? m[m.length - 1].slice(11) : null;", 'return m ? m[0].slice(11) : "x";', {"G2"}),
    ("split-ignores-escapes", PAGE, "if (c === 92) { i++; } else if (c === 34) { return i + 1; }", "if (c === 34) { return i + 1; }", {"G4"}),
    # the primary metric
    ("primary-regression-mae", PAGE, 'regression: "rmse"', 'regression: "mae"', {"G7"}),
    ("primary-classification-f1", PAGE, 'classification: "accuracy"', 'classification: "f1"', {"G7"}),
    ("primary-clustering-mean-distance", PAGE, 'clustering: "inertia"', 'clustering: "mean_distance"', {"G7"}),
    ("primary-metric-not-a-number-kept", PAGE, "if (isNum(v)) { return { name: name, text: String(v), value: v }; }\n    return null;", "return { name: name, text: String(v), value: null };", {"G7"}),
    # compare
    ("compare-no-dotted-keys", PAGE, 'if (isObj(xv) && isObj(yv)) { diffObj(prefix + k + ".", xv, yv, out); }', "if (false) { diffObj(prefix + k + \".\", xv, yv, out); }", {"G5"}),
    ("compare-equal-always-true", PAGE, "equal: deepEqual(x, y)", "equal: true", {"G5"}),
    ("compare-model-equal-inverted", PAGE, 'model_equal: at(a, "model", "sha256") === at(b, "model", "sha256")', 'model_equal: at(a, "model", "sha256") !== at(b, "model", "sha256")', {"G5"}),
    ("compare-metric-order-reversed", PAGE, "var names = Object.keys(ma);", "var names = Object.keys(ma).reverse();", {"G5"}),
    ("compare-b-only-metrics-dropped", PAGE, "Object.keys(mb).forEach(function (k) { if (!has(ma, k)) { names.push(k); } });", "void 0;", {"G5"}),
    ("compare-delta-sign", PAGE, "delta: sx !== null && sy !== null ? sy - sx : null", "delta: sx !== null && sy !== null ? sx - sy : null", {"G5"}),
    ("compare-missing-as-undefined", PAGE, "var x = has(ma, k) ? ma[k] : null, y = has(mb, k) ? mb[k] : null,", "var x = has(ma, k) ? ma[k] : undefined, y = has(mb, k) ? mb[k] : undefined,", {"G5"}),
    ("compare-list-metrics-as-numbers", PAGE, "function scalar(v) {\n    if (typeof v === \"string\")", "function scalar(v) {\n    if (Array.isArray(v)) { return v.length; }\n    if (typeof v === \"string\")", {"G5"}),
    ("compare-input-null-against-object-flattened", PAGE, "if (isObj(xv) && isObj(yv)) {", "if (isObj(xv) || isObj(yv)) {", {"G5"}),
    # the chart
    ("chart-axis-low-not-zero", PAGE, "lo: Math.min.apply(null, [0].concat(vals))", "lo: Math.min.apply(null, vals)", {"G8"}),
    ("chart-axis-high-not-zero", PAGE, "hi: Math.max.apply(null, [0].concat(vals))", "hi: Math.max.apply(null, vals)", {"G8"}),
    ("chart-left-out-not-named", PAGE, "if (v === null) { left.push(r.id); return; }", "if (v === null) { return; }", {"G8"}),
    ("chart-bars-reversed", PAGE, "return { metric: metric, bars: bars, left: left,", "return { metric: metric, bars: bars.reverse(), left: left,", {"G8"}),
    ("chart-scale-by-high", PAGE, "scale = span > 0 ? width / span : 0", "scale = data.hi > 0 ? width / data.hi : 0", {"G8"}),
    ("chart-zero-line-at-left", PAGE, "zero = -data.lo * scale", "zero = 0", {"G8"}),
    ("chart-metric-names-include-lists", PAGE, "scalar(mm[k]) !== null; })) { names.push(k); }", "true; })) { names.push(k); }", {"G8"}),
    ("default-metric-least-common", PAGE, "counts[n] > counts[best]", "counts[n] < counts[best]", {"G8"}),
    ("default-metric-ties-to-last", PAGE, "counts[n] > counts[best]", "counts[n] >= counts[best]", {"G8"}),
    # the commands
    ("command-verify-wrong-verb", PAGE, 'return "ml runs verify " + dirOf(dir) + " " + id;', 'return "ml runs show " + dirOf(dir) + " " + id;', {"G2"}),
    ("command-dir-not-quoted", PAGE, "\"'\" + s.replace(/'/g, \"'\\\\''\") + \"'\"", "\"'\" + s + \"'\"", {"G2"}),
    ("command-compare-order", PAGE, 'return "ml runs compare " + dirOf(dir) + " " + a + " " + b;', 'return "ml runs compare " + dirOf(dir) + " " + b + " " + a;', {"G2"}),
    # SHA-256 and its routing
    ("sha-constant-from-square-roots", PAGE, "K.push((Math.cbrt(n) % 1) * 4294967296 >>> 0);", "K.push((Math.sqrt(n) % 1) * 4294967296 >>> 0);", {"G1"}),
    ("sha-rotation-wrong", PAGE, "rotr(w[i - 15], 7)", "rotr(w[i - 15], 8)", {"G1"}),
    ("sha-padding-bit-wrong", PAGE, "buf[n] = 128;", "buf[n] = 129;", {"G1"}),
    ("sha-padding-length-wrong", PAGE, "total = ((n + 9 + 63) >> 6) << 6", "total = ((n + 8 + 63) >> 6) << 6", {"G1"}),
    ("sha-length-low-word", PAGE, "dv.setUint32(total - 4, (n * 8) >>> 0);", "dv.setUint32(total - 4, (n * 8) & 255);", {"G1"}),
    ("digest-never-uses-subtle", PAGE, "var f = sub ? function", "var f = false ? function", {"G1"}),
    ("digest-no-fallback-on-refusal", PAGE, ".then(hexOf, function () { return sha256Hex(bytes); })", ".then(hexOf)", {"G1"}),
    ("digest-hex-unpadded", PAGE, '(x < 16 ? "0" : "")', '""', {"G1"}),
    ("digest-picks-subtle-without-digest", PAGE, 'typeof env.crypto.subtle.digest === "function" ? env.crypto.subtle : null', "env.crypto.subtle ? env.crypto.subtle : null", {"G1"}),
    # the sample
    ("sample-block-renamed", PAGE, '<script type="text/plain" id="sample">', '<script type="text/plain" id="sampl">', {"G9", "G10"}),
    ("sample-record-edited", PAGE, '"accuracy":"0.9333333333333333"', '"accuracy":"0.9433333333333333"', {"G9"}),
    ("sample-is-javascript", PAGE, '<script type="text/plain" id="sample">', '<script id="sample">', {"G9", "G10"}),
    # the structure of the file
    ("fetch-added", PAGE, '  render();\n  if (location.hash === "#sample") { loadSample(); }', '  fetch("/x");\n  render();\n  if (location.hash === "#sample") { loadSample(); }', {"G10"}),
    ("inner-html-used", PAGE, "function clear(e) { while (e.firstChild) { e.removeChild(e.firstChild); } return e; }", 'function clear(e) { e.innerHTML = ""; return e; }', {"G10"}),
    ("external-script", PAGE, "</head>", '<script src="https://example.com/x.js"></script>\n</head>', {"G10"}),
    ("external-stylesheet", PAGE, '<link rel="stylesheet" href="assets/site.css">', '<link rel="stylesheet" href="https://example.com/site.css">', {"G10"}),
    ("colour-literal", PAGE, ".bd .src { color: var(--muted); font-size: .85rem; }", ".bd .src { color: #555555; font-size: .85rem; }", {"G10"}),
    ("css-url-loaded", PAGE, ".bd .chartbox .zero {", ".bd .chartbox .zero { background: url(x.png);", {"G10"}),
    ("skip-link-removed", PAGE, '<a class="skip" href="#main">Skip to content</a>\n', "", {"G10"}),
    ("lang-removed", PAGE, '<html lang="en">', "<html>", {"G10"}),
    ("viewport-removed", PAGE, '<meta name="viewport" content="width=device-width, initial-scale=1">\n', "", {"G10"}),
    ("title-too-long", PAGE, "<title>cancho-ml experiments board: read run records locally</title>", "<title>cancho-ml experiments board: read run records locally and compare them</title>", {"G10"}),
    ("second-h1", PAGE, '<h2 id="h-load">Load runs</h2>', '<h1 id="h-load">Load runs</h1>', {"G10"}),
    ("textarea-unlabelled", PAGE, '<label for="paste">', '<label for="pasted">', {"G10"}),
    ("inline-handler", PAGE, '<button type="button" id="clear-btn">Clear</button>', '<button type="button" id="clear-btn" onclick="void 0">Clear</button>', {"G10"}),
    ("logic-touches-document", PAGE, "var MAX_DOCS = 5000;", "var MAX_DOCS = 5000; var D = typeof document;", {"G10"}),
    ("logic-marker-removed", PAGE, "/* LOGIC end */", "/* end */", {"G10"}),
    ("script-does-not-compile", PAGE, "  render();\n  if (location.hash", "  render(;\n  if (location.hash", {"G10"}),
    ("size-over-budget", PAGE, "<!doctype html>", "<!doctype html>\n<!-- " + "x" * 80000 + " -->", {"G10"}),
    ("nothing-sent-statement-removed", PAGE, "Nothing is sent anywhere: the files are read", "The files are read", {"G10"}),
    ("cannot-verify-statement-removed", PAGE, "The page cannot verify a run: ", "The page checks a run: ", {"G10"}),
    ("only-the-tool-statement-removed", PAGE, "and only the tool can do that, because", "and that is done elsewhere, because", {"G10"}),
    ("sample-caption-removed", PAGE, "The sample is real records:", "The sample is:", {"G10"}),
    ("list-only-statement-removed", PAGE, "is marked <em>list only</em>", "is marked", {"G10"}),
    ("multiple-files-removed", PAGE, '<input id="pick" type="file" multiple>', '<input id="pick" type="file">', {"G10"}),
    ("drop-handler-removed", PAGE, 'window.addEventListener("drop", function (e) {', 'window.addEventListener("drag", function (e) {', {"G10"}),
    ("aria-sort-removed", PAGE, 'th.setAttribute("aria-sort", st.dir === "asc" ? "ascending" : "descending");', "void 0;", {"G10"}),
    ("dark-scheme-removed-from-css", CSS, "prefers-color-scheme: dark", "prefers-color-scheme: darks", {"G10"}),
]

# survivors, each with its reason: a fault that no input of the gates can show, or no input that can exist
EXPLAINED = {
    "sha-length-high-word": "unreachable: the high word of the bit length is not zero only for input of 512 MiB or more, and a record is at most 262,144 bytes (the page reads at most 16 MiB of a file)",
    "compare-input-b-only-keys-dropped": "unreachable: every record of schema ml.run.v1 has the same members of `inputs`, so the second object never has a member the first lacks",
    "model-from-first-line": "a plan holds one `model` line, so the first and the last are the same line",
}
# extra mutants that are only here to hold their explanation (they must apply, and survive)
MUTANTS += [
    ("sha-length-high-word", PAGE, "dv.setUint32(total - 8, Math.floor(n / 536870912));", "dv.setUint32(total - 8, 0);", {"G1"}),
    ("compare-input-b-only-keys-dropped", PAGE, "Object.keys(y).forEach(function (k) { if (!has(x, k)) { keys.push(k); } });", "void 0;", {"G5"}),
]
EXPLAINED.pop("model-from-first-line")  # it is killed (the mutant returns "x" when the first line is not a model line); see above


def problems(root):
    out, names = [], set()
    for name, f, old, new, gates in MUTANTS:
        if name in names:
            out.append("mutant %s is named twice" % name)
        names.add(name)
        text = (root / f).read_text()
        if text.count(old) != 1:
            out.append("mutant %s: the pattern occurs %d times in %s, want 1" % (name, text.count(old), f))
        if old == new:
            out.append("mutant %s: replacement equals pattern" % name)
    return out


def run_check(page_root, fixtures, binary=None):
    """check_board.py on the copy in page_root (which holds docs/); -> (exit status, the set of gates that FAILED, the last lines)."""
    argv = [PY, str(HERE / "check_board.py"), "--page", str(page_root / PAGE), "--fixtures", str(fixtures)]
    r = subprocess.run(argv, capture_output=True, text=True)
    failed = set(re.findall(r"check_board: (G\d+) FAILED", r.stdout))
    if r.returncode != 0 and not failed:
        failed = {"(crash)"}
    return r.returncode, failed, (r.stdout + r.stderr)[-400:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(ROOT / "build" / "ml"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    bad = problems(ROOT)
    if bad:
        print("\n".join(bad))
        return 1
    chosen = [m for m in MUTANTS if not args.names or any(w in m[0] for w in args.names)]
    if args.check:
        print("mutants_board: %d mutants, every pattern applies once" % len(MUTANTS))
        return 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        subprocess.run([PY, str(HERE / "check_board.py"), "--bin", args.bin, "--make-fixtures", str(td / "fx")], check=True, stdout=subprocess.DEVNULL)

        def tree(name):
            t = td / name
            (t / "docs" / "assets").mkdir(parents=True)
            shutil.copy(ROOT / PAGE, t / PAGE)
            shutil.copy(ROOT / CSS, t / CSS)
            return t
        control = tree("control")
        code, failed, tail = run_check(control, td / "fx")
        print("%-44s %s" % ("control", "passes every gate" if code == 0 else "FAILS " + ",".join(sorted(failed)) + "\n" + tail))
        if code != 0:
            return 1
        killed = explained = survivors = 0
        for i, (name, f, old, new, gates) in enumerate(chosen):
            t = tree("m%d" % i)
            p = t / f
            p.write_text(p.read_text().replace(old, new, 1))
            code, failed, tail = run_check(t, td / "fx")
            shutil.rmtree(t)
            ok = code != 0 and bool(gates & failed or "(crash)" in failed)
            if ok:
                killed += 1
                print("%-44s killed by %s" % (name, ",".join(sorted(failed))))
            elif name in EXPLAINED and code == 0:
                explained += 1
                print("%-44s SURVIVED (explained: %s)" % (name, EXPLAINED[name]))
            else:
                survivors += 1
                print("%-44s SURVIVED%s; expected %s, failed %s" % (name, "" if code == 0 else " (killed by another gate)", ",".join(sorted(gates)), ",".join(sorted(failed)) or "none"))
        print("mutants_board: %d mutants, %d killed, %d survived and explained, %d unexplained" % (len(chosen), killed, explained, survivors))
        return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())

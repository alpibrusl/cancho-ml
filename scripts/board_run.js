// Runs the logic of docs/board.html headlessly: the part of its one script between the markers `/* LOGIC begin` and `/* LOGIC end */`, which touches no document.
// Used by scripts/check_board.py, with node when it starts and with Apple's `jsc` (JavaScriptCore's shell) when it does not:
//     node scripts/board_run.js REQUEST.json          jsc scripts/board_run.js -- REQUEST.json
// REQUEST.json: {"html": PATH, "jobs": [{"op": ..}, ..]}. The output is one JSON array on standard output: {"ok": true, "result": ..} or {"ok": false, "error": ..} per job.
// The code is the same for both engines; what an engine lacks (TextEncoder, base64, a crypto object) is supplied here.
var ARGV = typeof process !== "undefined" && process.argv ? process.argv.slice(2) : Array.prototype.slice.call(arguments);
(function () {
  "use strict";
  var isNode = typeof process !== "undefined" && process.versions && process.versions.node;
  var readFile = isNode ? function (p) { return require("fs").readFileSync(p, "utf8"); } : function (p) { return read(p); };
  var out = isNode ? function (s) { process.stdout.write(s); } : function (s) { print(s); };

  function utf8(s) {
    var o = [];
    for (var i = 0; i < s.length; i++) {
      var c = s.charCodeAt(i);
      if (c >= 0xD800 && c < 0xDC00 && i + 1 < s.length && s.charCodeAt(i + 1) >= 0xDC00 && s.charCodeAt(i + 1) < 0xE000) { c = 0x10000 + ((c - 0xD800) << 10) + (s.charCodeAt(i + 1) - 0xDC00); i++; }
      else if (c >= 0xD800 && c < 0xE000) { c = 0xFFFD; }
      if (c < 0x80) { o.push(c); }
      else if (c < 0x800) { o.push(0xC0 | (c >> 6), 0x80 | (c & 63)); }
      else if (c < 0x10000) { o.push(0xE0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63)); }
      else { o.push(0xF0 | (c >> 18), 0x80 | ((c >> 12) & 63), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63)); }
    }
    return new Uint8Array(o);
  }
  function Encoder() {}
  Encoder.prototype.encode = utf8;
  var TE = typeof TextEncoder !== "undefined" ? TextEncoder : Encoder;

  var B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  function unb64(s) {
    var bytes = [], acc = 0, bits = 0;
    for (var i = 0; i < s.length; i++) {
      var v = B64.indexOf(s.charAt(i));
      if (v < 0) { continue; }
      acc = (acc << 6) | v; bits += 6;
      if (bits >= 8) { bits -= 8; bytes.push((acc >> bits) & 255); acc &= (1 << bits) - 1; }
    }
    return new Uint8Array(bytes);
  }

  var html = null;
  function load(htmlPath) {
    html = readFile(htmlPath);
    var a = html.indexOf("/* LOGIC begin"), b = html.indexOf("/* LOGIC end */");
    if (a < 0 || b < a) { throw new Error("the page has no LOGIC begin / LOGIC end markers"); }
    // the logic sees only what is passed here: no document, no window
    return new Function("TextEncoder", '"use strict";\n' + html.slice(a, b + 15) + "\nreturn Board;")(TE);
  }

  var realSubtle = typeof crypto !== "undefined" && crypto && crypto.subtle ? crypto.subtle : null;
  function digestFor(Board, mode) {
    if (mode === "pure") { return Board.makeDigest({}); }
    if (!realSubtle) { throw new Error("no crypto.subtle in this engine"); }
    return Board.makeDigest({ crypto: { subtle: realSubtle } });
  }
  function bufOf(bytes) { var u = new Uint8Array(bytes.length); u.set(bytes); return u.buffer; }

  async function run(job, Board) {
    switch (job.op) {
      case "syntax": {
        // the whole script of the page (the DOM code too) must compile; it is compiled, not run
        var a = html.indexOf("<script>\n"), b = html.indexOf("</script>", a);
        new Function(html.slice(a + 9, b));
        return true;
      }
      case "engine": return { engine: isNode ? "node " + process.version : "jsc", subtle: !!realSubtle };
      case "sha": return await digestFor(Board, job.mode)(unb64(job.b64));
      case "routing": {
        // a stand-in for crypto.subtle: the digest it returns is 32 bytes 0..31, or it rejects; makeDigest must use it, and fall back to its own routine when it rejects
        var ok = { subtle: { digest: function () { var u = new Uint8Array(32); for (var i = 0; i < 32; i++) { u[i] = i; } return Promise.resolve(u.buffer); } } };
        var bad = { subtle: { digest: function () { return Promise.reject(new Error("insecure context")); } } };
        var f1 = Board.makeDigest({ crypto: ok }), f2 = Board.makeDigest({ crypto: bad }), f3 = Board.makeDigest({ crypto: {} }), f4 = Board.makeDigest(undefined), f5 = Board.makeDigest({ crypto: { subtle: {} } });
        var bytes = unb64(job.b64);
        return { kinds: [f1.kind, f2.kind, f3.kind, f4.kind, f5.kind], viaSubtle: await f1(bytes), afterReject: await f2(bytes), none: await f3(bytes), undef: await f4(bytes) };
      }
      case "split": return Board.splitDocs(job.text);
      case "analyze": return await Board.analyze(job.sources, digestFor(Board, job.mode || "pure"));
      case "compare": return Board.compareRows(job.a, job.b);
      case "sort": return Board.sortRows(job.rows, job.key, job.dir).map(function (r) { return r.uid; });
      case "filter": return Board.filterRows(job.rows, job.f).map(function (r) { return r.uid; });
      case "varying": return Board.varyingKeys(job.rows);
      case "inputValue": return job.rows.map(function (r) { return Board.inputText(r, job.key); });
      case "chart": return Board.chartData(job.rows, job.metric);
      case "layout": return Board.layoutBars(job.data, job.width);
      case "defaultMetric": return Board.defaultMetric(job.rows);
      case "metricNames": return Board.metricNames(job.rows);
      case "primary": return Board.primaryMetric(job.task, job.metrics);
      case "commands": return { verify: Board.verifyCommand(job.dir, job.id), compare: Board.compareCommand(job.dir, job.a, job.b), show: Board.showCommand(job.dir, job.id) };
      default: throw new Error("unknown op " + job.op);
    }
  }

  var req = JSON.parse(readFile(ARGV[0]));
  var Board = load(req.html), results = [];
  (async function () {
    for (var i = 0; i < req.jobs.length; i++) {
      try { results.push({ ok: true, result: await run(req.jobs[i], Board) }); }
      catch (e) { results.push({ ok: false, error: String(e && e.stack || e) }); }
    }
    out(JSON.stringify(results));
  })();
})();

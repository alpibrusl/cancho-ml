#!/usr/bin/env node
// Runs the logic of docs/board.html headlessly: the part of its one script between the markers `/* LOGIC begin` and `/* LOGIC end */`, which touches no document.
// Used by scripts/check_board.py. Standard input: {"html": PATH, "jobs": [{"op": ..}, ..]}; standard output: one JSON array, one {"ok": true, "result": ..} or
// {"ok": false, "error": ..} per job, in order. Only node's own modules are used.
"use strict";
const fs = require("fs");
const vm = require("vm");

function extractLogic(html) {
  const a = html.indexOf("/* LOGIC begin");
  const b = html.indexOf("/* LOGIC end */");
  if (a < 0 || b < a) { throw new Error("the page has no LOGIC begin / LOGIC end markers"); }
  return html.slice(a, b + "/* LOGIC end */".length);
}

function load(htmlPath, withCrypto) {
  const code = extractLogic(fs.readFileSync(htmlPath, "utf8"));
  // a context that holds only what a browser's global scope offers to pure code: no document, no window, no navigator
  const ctx = { TextEncoder: TextEncoder, TextDecoder: TextDecoder };
  if (withCrypto) { ctx.crypto = globalThis.crypto; }
  vm.createContext(ctx);
  return vm.runInContext('"use strict";\n' + code + "\n;Board;", ctx);
}

async function run(job, Board) {
  const digestFor = (mode) => {
    if (mode === "pure") { return Board.makeDigest({}); }
    return Board.makeDigest({ crypto: globalThis.crypto });
  };
  switch (job.op) {
    case "sha": {
      const bytes = new Uint8Array(Buffer.from(job.b64, "base64"));
      return await digestFor(job.mode)(bytes);
    }
    case "kind": return digestFor(job.mode).kind;
    case "split": return Board.splitDocs(job.text);
    case "analyze": return await Board.analyze(job.sources, digestFor(job.mode || "subtle"));
    case "compare": return Board.compareRows(job.a, job.b);
    case "sort": return Board.sortRows(job.rows, job.key, job.dir).map((r) => r.uid);
    case "filter": return Board.filterRows(job.rows, job.f).map((r) => r.uid);
    case "varying": return Board.varyingKeys(job.rows);
    case "inputValue": return job.rows.map((r) => Board.inputText(r, job.key));
    case "chart": return Board.chartData(job.rows, job.metric);
    case "layout": return Board.layoutBars(job.data, job.width);
    case "defaultMetric": return Board.defaultMetric(job.rows);
    case "metricNames": return Board.metricNames(job.rows);
    case "primary": return Board.primaryMetric(job.task, job.metrics);
    case "commands": return { verify: Board.verifyCommand(job.dir, job.id), compare: Board.compareCommand(job.dir, job.a, job.b), show: Board.showCommand(job.dir, job.id) };
    case "short": return Board.short12(job.s);
    default: throw new Error("unknown op " + job.op);
  }
}

async function main() {
  let input = "";
  for await (const chunk of process.stdin) { input += chunk; }
  const req = JSON.parse(input);
  const Board = load(req.html, true);
  const out = [];
  for (const job of req.jobs) {
    try {
      out.push({ ok: true, result: await run(job, Board) });
    } catch (e) {
      out.push({ ok: false, error: String(e && e.stack || e) });
    }
  }
  process.stdout.write(JSON.stringify(out));
}

main().catch((e) => { process.stderr.write(String(e && e.stack || e) + "\n"); process.exit(2); });

#!/usr/bin/env node
/* Records the videos of the landing page (docs/index.html) from the real logs it carries: the three replay panels (`fit`, `serve`, `fix`), which scripts/site.py wrote from a
   run of the built binaries (scripts/demo.sh and the refusal). Nothing here invents a line: the page of a terminal is drawn from the spans of the panel, the typed lines are the
   commands (class a), the rest appears a line at a time. Writes docs/assets/media/<name>.mp4, .webm and .jpg (the poster).

       npm install playwright-core            # once, anywhere; or run with NODE_PATH pointing at it
       CHROME=/path/to/chrome node scripts/record_videos.js        # default: the chromium of PLAYWRIGHT_BROWSERS_PATH
   Needs ffmpeg. Run it again after `python3 scripts/site.py` changed a replay panel, and commit the result: the page shows what was recorded. */
const fs = require("fs"), path = require("path"), cp = require("child_process");
const { chromium } = require("playwright-core");
const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "docs", "index.html"), "utf8");
const out = path.join(root, "docs", "assets", "media");
const NAMES = { fit: "Fit and compare", serve: "Serve and curl", fix: "A refusal and its repair" };
const unesc = (s) => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&amp;/g, "&");
function panel(id) {
  const m = html.match(new RegExp('<pre id="screen-' + id + '"[^>]*>([\\s\\S]*?)</pre>'));
  if (!m) throw new Error("no panel " + id);
  return [...m[1].matchAll(/<span(?: class="(\w*)")?>([\s\S]*?)<\/span>/g)].map((x) => [x[1] || "", unesc(x[2])]);
}
const page = (title) => `<!doctype html><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:#0a1013;font:22px/1.5 "DejaVu Sans Mono",ui-monospace,monospace;color:#d6e2e6}
.w{margin:36px 48px;height:calc(100% - 72px);border:1px solid #243038;border-radius:14px;background:#0f1519;overflow:hidden;box-shadow:0 30px 80px -30px #000}
.b{display:flex;gap:10px;align-items:center;padding:14px 18px;border-bottom:1px solid #1f2a30;color:#7d8f96;font-size:17px}.b i{width:14px;height:14px;border-radius:50%;background:#2c3a41;display:block}
#t{padding:22px 28px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:20px;line-height:1.55}
.g{color:#5fd3a6}.b2{color:#ff8a80}.d{color:#7d8f96}.a{color:#8ec0ff}.cur{display:inline-block;width:11px;height:22px;background:#d6e2e6;vertical-align:-4px;margin-left:2px}
</style><div class="w"><div class="b"><i></i><i></i><i></i><span>${title} — a real run, recorded</span></div><div id="t"></div></div>`;
async function record(browser, id) {
  const lines = panel(id), tmp = fs.mkdtempSync("/tmp/vid-");
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 }, recordVideo: { dir: tmp, size: { width: 1280, height: 720 } } });
  const p = await ctx.newPage();
  await p.setContent(page(NAMES[id]));
  await p.evaluate(async (lines) => {
    const t = document.getElementById("t"), sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const cls = { a: "a", g: "g", b: "b2", d: "d", "": "" };
    const cur = () => { const c = document.createElement("span"); c.className = "cur"; return c; };
    for (const [c, text] of lines) {
      const s = document.createElement("span"); s.className = cls[c] || ""; t.appendChild(s);
      if (c === "a") { const k = cur(); t.appendChild(k); for (const ch of text) { s.textContent += ch; await sleep(22); } k.remove(); await sleep(450); }
      else { s.textContent = text; await sleep(c === "d" ? 350 : 520); }
      t.appendChild(document.createTextNode("\n"));
      t.parentNode.scrollTop = t.parentNode.scrollHeight;
      while (t.getBoundingClientRect().height > innerHeight - 190) { t.firstChild.remove(); if (t.firstChild && t.firstChild.nodeType === 3) t.firstChild.remove(); }
    }
    await sleep(2600);
  }, lines);
  const vid = p.video(); await ctx.close();
  const src = await vid.path();
  const base = path.join(out, "run-" + id);
  cp.execSync(`ffmpeg -y -loglevel error -ss 0.4 -i ${src} -c:v libx264 -pix_fmt yuv420p -crf 30 -preset slow -movflags +faststart -an ${base}.mp4`);
  cp.execSync(`ffmpeg -y -loglevel error -ss 0.4 -i ${src} -c:v libvpx-vp9 -crf 38 -b:v 0 -an ${base}.webm`);
  cp.execSync(`ffmpeg -y -loglevel error -sseof -2.2 -i ${base}.mp4 -frames:v 1 -vf scale=960:-1 -q:v 4 ${base}.jpg`);
  fs.rmSync(tmp, { recursive: true });
  console.log("wrote", path.relative(root, base) + ".{mp4,webm,jpg}", fs.statSync(base + ".mp4").size + " + " + fs.statSync(base + ".webm").size + " bytes");
}
(async () => {
  const exe = process.env.CHROME || undefined;
  const browser = await chromium.launch({ executablePath: exe, args: ["--no-sandbox"] });
  for (const id of process.argv.slice(2).length ? process.argv.slice(2) : ["fit", "serve", "fix"]) await record(browser, id);
  await browser.close();
})();

// Records the live footage for the walkthrough from the running Docker stack.
//
//   docker compose up --build -d          (from the Vantara folder)
//   node capture/capture.mjs [scene ...]  (from video/; default: every scene)
//
// Each scene drives a real browser session against the live dashboard or API and is
// saved to public/footage/<scene>.mp4. Frames come from Chrome's screencast (lossless
// PNG) and are timed exactly as they arrived, so the clips play at real speed.
// Cosmetic only: a cursor is drawn so viewers can follow it, and Streamlit's developer
// toolbar ("Deploy" button, running indicator) is hidden.

import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium } from "playwright-core";

const DASHBOARD = process.env.DASHBOARD_URL ?? "http://127.0.0.1:8501";
const API = process.env.API_URL ?? "http://127.0.0.1:8000";
const CHROME =
  process.env.CHROME_PATH ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";
const OUT_DIR = resolve("public/footage");
const FEATURED_CUSTOMER = "14603"; // high risk, dormant, win-back email (see script §5)

const WIDTH = 1920;
const HEIGHT = 1080;

const PREDICT_EXAMPLE = {
  customer_id: 12345,
  frequency: 8,
  total_items_bought: 250,
  unique_skus_bought: 45,
  monetary_total: 1250.5,
  recency: 14.5,
  customer_age_days: 320,
  monetary_avg: 156.31,
  avg_basket_size: 31.25,
  interpurchase_interval_mean: 28.5,
  interpurchase_interval_std: 12.1,
  purchase_frequency_trend: 0.25,
  return_rate: 0.02,
  cancellation_rate: 0,
  seasonal_concentration: 0.35,
  discount_sensitivity: 0.15,
  is_uk_customer: 1,
  engagement_score: 82.5,
  spend_last_90d: 400,
};

// A visible cursor that follows Playwright's mouse, with a ring on click.
const CURSOR = `
(() => {
  const install = () => {
    if (document.getElementById("__cursor")) return;
    const c = document.createElement("div");
    c.id = "__cursor";
    c.innerHTML = '<svg width="28" height="28" viewBox="0 0 24 24"><path d="M4 2l16 9.5-7 1.6-3.6 6.9z" fill="#0F172A" stroke="#fff" stroke-width="1.4" stroke-linejoin="round"/></svg>';
    Object.assign(c.style, { position: "fixed", left: "-50px", top: "-50px", zIndex: 2147483647, pointerEvents: "none", transform: "translate(-4px,-2px)" });
    document.documentElement.appendChild(c);
    addEventListener("mousemove", (e) => { c.style.left = e.clientX + "px"; c.style.top = e.clientY + "px"; }, true);
    addEventListener("mousedown", (e) => {
      const r = document.createElement("div");
      Object.assign(r.style, { position: "fixed", left: e.clientX - 18 + "px", top: e.clientY - 18 + "px", width: "36px", height: "36px", borderRadius: "50%", border: "3px solid #4F46E5", zIndex: 2147483646, pointerEvents: "none", transition: "all .45s ease-out", opacity: "0.9" });
      document.documentElement.appendChild(r);
      requestAnimationFrame(() => { r.style.transform = "scale(1.8)"; r.style.opacity = "0"; });
      setTimeout(() => r.remove(), 500);
    }, true);
  };
  if (document.readyState === "loading") addEventListener("DOMContentLoaded", install); else install();
})();
`;

const HIDE_STREAMLIT_CHROME = `
  [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
  [data-testid="stAppDeployButton"] { visibility: hidden !important; }
`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let cursor = { x: WIDTH / 2, y: HEIGHT / 2 };

async function moveTo(page, x, y, ms = 700) {
  const steps = Math.max(10, Math.round(ms / 16));
  await page.mouse.move(x, y, { steps });
  cursor = { x, y };
}

async function moveToElement(page, locator, ms) {
  await locator.scrollIntoViewIfNeeded();
  const box = await locator.boundingBox();
  if (!box) throw new Error(`no box for ${locator}`);
  await moveTo(page, box.x + box.width / 2, box.y + box.height / 2, ms);
}

async function clickElement(page, locator, ms) {
  await moveToElement(page, locator, ms);
  await sleep(150);
  await page.mouse.down();
  await sleep(80);
  await page.mouse.up();
}

async function smoothScroll(page, dy, ms = 1200) {
  const steps = Math.max(10, Math.round(ms / 16));
  for (let i = 0; i < steps; i++) {
    await page.mouse.wheel(0, dy / steps);
    await sleep(16);
  }
}

async function streamlitIdle(page, settleMs = 600) {
  await page.waitForFunction(
    () => !document.querySelector('[data-testid="stStatusWidget"] [data-testid="stStatusWidgetRunningIcon"], [data-testid="stStatusWidget"] img'),
    null,
    { timeout: 60_000 },
  ).catch(() => {});
  await page.waitForFunction(() => !document.querySelector('[data-testid="stSkeleton"]'), null, { timeout: 60_000 }).catch(() => {});
  await sleep(settleMs);
}

async function openView(page, label) {
  await clickElement(page, page.locator('section[data-testid="stSidebar"] label', { hasText: label }), 800);
  await sleep(400);
  await streamlitIdle(page, 900);
}

async function openDashboard(page) {
  await page.goto(DASHBOARD, { waitUntil: "networkidle" });
  await page.addStyleTag({ content: HIDE_STREAMLIT_CHROME });
  await page.waitForSelector("text=Executive CRM Cockpit", { timeout: 60_000 });
  await streamlitIdle(page, 1500);
  await moveTo(page, WIDTH * 0.62, HEIGHT * 0.55, 10);
}

async function scrollMainTop(page) {
  await page.evaluate(() => {
    const main = document.querySelector('[data-testid="stMain"]') ?? document.scrollingElement;
    main.scrollTo({ top: 0 });
  });
}

// ---------------------------------------------------------------------------
// Recorder: Chrome screencast frames -> timed concat list -> H.264 MP4.
// ---------------------------------------------------------------------------
async function record(page, name, action) {
  const cdp = await page.context().newCDPSession(page);
  const work = mkdtempSync(join(tmpdir(), `vantara-${name}-`));
  const frames = [];
  cdp.on("Page.screencastFrame", async ({ data, metadata, sessionId }) => {
    const file = join(work, `${String(frames.length).padStart(6, "0")}.png`);
    writeFileSync(file, Buffer.from(data, "base64"));
    frames.push({ file, t: metadata.timestamp });
    await cdp.send("Page.screencastFrameAck", { sessionId }).catch(() => {});
  });
  await cdp.send("Page.startScreencast", { format: "png", maxWidth: WIDTH, maxHeight: HEIGHT, everyNthFrame: 1 });
  // Nudge the cursor so the first frame is emitted immediately.
  await page.mouse.move(cursor.x + 1, cursor.y);
  await page.mouse.move(cursor.x, cursor.y);
  await sleep(300);
  const start = Date.now() / 1000;

  await action();

  await sleep(300);
  const end = Date.now() / 1000;
  await cdp.send("Page.stopScreencast");
  await sleep(300);
  await cdp.detach();

  if (frames.length === 0) throw new Error(`${name}: no frames captured`);
  const t0 = Math.max(frames[0].t, start - 0.3);
  const lines = [];
  frames.forEach((f, i) => {
    const next = i + 1 < frames.length ? frames[i + 1].t : end;
    const duration = Math.max(1 / 60, next - Math.max(f.t, t0));
    lines.push(`file '${f.file.replaceAll("\\", "/")}'`, `duration ${duration.toFixed(4)}`);
  });
  lines.push(`file '${frames.at(-1).file.replaceAll("\\", "/")}'`);
  const list = join(work, "frames.txt");
  writeFileSync(list, lines.join("\n"));

  mkdirSync(OUT_DIR, { recursive: true });
  const out = join(OUT_DIR, `${name}.mp4`);
  const ff = spawnSync("ffmpeg", [
    "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", list,
    "-vf", `fps=30,scale=${WIDTH}:${HEIGHT}:flags=lanczos,format=yuv420p`,
    "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-tune", "stillimage", "-movflags", "+faststart", out,
  ], { stdio: "inherit" });
  rmSync(work, { recursive: true, force: true });
  if (ff.status !== 0) throw new Error(`${name}: ffmpeg failed`);
  console.log(`${name}: ${frames.length} frames, ${(end - start).toFixed(1)} s -> ${out}`);
}

// ---------------------------------------------------------------------------
// Scenes (script sections 4-6)
// ---------------------------------------------------------------------------
const scenes = {
  // §4: from the cockpit to the value-at-risk leaderboard.
  async at_risk(page) {
    await openDashboard(page);
    await record(page, "at_risk", async () => {
      await sleep(3500);
      await openView(page, "At-Risk Priority");
      await sleep(2500);
      await moveTo(page, 1250, 520, 900);
      await sleep(800);
      await smoothScroll(page, 420, 1800);
      await sleep(3500);
      await smoothScroll(page, -420, 1200);
      await sleep(1500);
    });
  },

  // §4: campaign planner at the default £500 budget, then the policy comparison.
  async planner(page) {
    await openDashboard(page);
    await openView(page, "Campaign Planner");
    await scrollMainTop(page);
    await record(page, "planner", async () => {
      await sleep(2500);
      await moveToElement(page, page.locator('[data-testid="stMetric"]').nth(3), 900);
      await sleep(2000);
      await moveTo(page, 1250, 700, 700);
      await smoothScroll(page, 470, 1800);
      await sleep(1200);
      const bars = page.locator(".js-plotly-plot .bars path");
      const n = await bars.count();
      for (let i = 0; i < n; i++) {
        await moveToElement(page, bars.nth(i), 650);
        await sleep(1300);
      }
      await moveTo(page, 1650, 760, 600);
      await sleep(800);
      await smoothScroll(page, 760, 2000);
      await sleep(3500);
    });
  },

  // §5: one customer explained, from the header card to SHAP and LIME. Holds follow
  // the narration (public/voice/05-customer.mp3): risk and action until ~13 s,
  // suggestions until ~30 s, then SHAP and LIME.
  async customer360(page) {
    await openDashboard(page);
    await openView(page, "Customer 360");
    await scrollMainTop(page);
    await record(page, "customer360", async () => {
      await sleep(1500);
      const input = page.locator('input[aria-label="Customer ID"]');
      await clickElement(page, input, 900);
      await page.keyboard.press("Control+A");
      await page.keyboard.type(FEATURED_CUSTOMER, { delay: 140 });
      await page.keyboard.press("Enter");
      await sleep(400);
      await streamlitIdle(page, 800);
      await moveTo(page, 1100, 360, 900);
      await sleep(8400);
      await moveTo(page, 1250, 700, 600);
      await smoothScroll(page, 430, 1800);
      await sleep(14600);
      await smoothScroll(page, 560, 1800);
      await sleep(4000);
      await smoothScroll(page, 420, 1500);
      await sleep(4500);
    });
  },

  // §6: a live call to POST /predict in the API's Swagger docs.
  async api_predict(page) {
    await page.goto(`${API}/docs`, { waitUntil: "networkidle" });
    await page.waitForSelector(".opblock", { timeout: 60_000 });
    await moveTo(page, WIDTH * 0.6, HEIGHT * 0.4, 10);
    const block = page.locator('.opblock-post:has(.opblock-summary-path[data-path="/predict"])');
    await record(page, "api_predict", async () => {
      await sleep(2500);
      await clickElement(page, block.locator(".opblock-summary"), 900);
      await sleep(1200);
      await clickElement(page, block.locator("button.try-out__btn"), 800);
      await sleep(600);
      const body = block.locator("textarea.body-param__text");
      await body.fill(JSON.stringify(PREDICT_EXAMPLE, null, 2));
      await moveToElement(page, body, 700);
      await sleep(1800);
      await clickElement(page, block.locator("button.execute"), 900);
      await block.locator(".live-responses-table .highlight-code").first().waitFor({ timeout: 30_000 });
      await sleep(800);
      await page.evaluate(() => {
        document.querySelector('.opblock-post .live-responses-table')?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
      await sleep(1800);
      await moveTo(page, 900, 560, 800);
      await sleep(2500);
      await smoothScroll(page, 500, 2000);
      await sleep(3500);
    });
  },
};

const wanted = process.argv.slice(2);
const names = wanted.length ? wanted : Object.keys(scenes);
const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ["--hide-scrollbars", "--force-color-profile=srgb", "--font-render-hinting=none"],
});
try {
  for (const name of names) {
    if (!scenes[name]) throw new Error(`unknown scene ${name}; choose from ${Object.keys(scenes).join(", ")}`);
    const context = await browser.newContext({
      viewport: { width: WIDTH, height: HEIGHT },
      deviceScaleFactor: 1,
      colorScheme: "light",
    });
    await context.addInitScript(CURSOR);
    const page = await context.newPage();
    cursor = { x: WIDTH / 2, y: HEIGHT / 2 };
    await scenes[name](page);
    await context.close();
  }
} finally {
  await browser.close();
}

// Headless-Chrome smoke of the app shell (not a vitest file): starts Vite, opens index.html with the test pattern,
// checks DPR-independent internal resolution, the Standard view transform on the calibration strip (±1 LSB),
// debug views, probe records, timestamps and fly-camera keys, and writes screenshots.
//   npx tsx tests/app/e2e-app.ts [outDir]
import { mkdirSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';
import type { App } from '../../src/app/app.ts';
import { acquireGpuLock } from '../../validation/harness/gpu-lock.ts';

declare global { interface Window { __app?: App } }

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const OUT = path.resolve(process.argv[2] ?? path.join(ROOT, 'validation/out/app-e2e'));
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const checks: { name: string; ok: boolean; detail?: string }[] = [];
const check = (name: string, ok: boolean, detail?: string) => { checks.push({ name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`); };

/** Decode a PNG in the page (no Node PNG dependency) and return the red channel at fractional positions. */
async function samplePng(page: import('playwright').Page, png: Buffer, pts: [number, number][]): Promise<number[]> {
  return page.evaluate(async ([b64, p]) => {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const bmp = await createImageBitmap(new Blob([bytes], { type: 'image/png' }), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    const c = new OffscreenCanvas(bmp.width, bmp.height);
    const g = c.getContext('2d', { willReadFrequently: true })!;
    g.drawImage(bmp, 0, 0);
    return p.map(([fx, fy]) => g.getImageData(Math.floor(fx * bmp.width), Math.floor(fy * bmp.height), 1, 1).data[0]);
  }, [png.toString('base64'), pts] as const);
}

const srgb = (x: number) => { const c = Math.min(1, Math.max(0, x)); return c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055; };

async function main(): Promise<number> {
  mkdirSync(OUT, { recursive: true });
  const release = await acquireGpuLock('e2e-app');
  const vite = await createServer({ root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), server: { port: 0, host: '127.0.0.1' }, logLevel: 'warn' });
  await vite.listen();
  const addr = vite.httpServer!.address() as { port: number };
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const errors: string[] = [];
  try {
    const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 2 });
    const page = await ctx.newPage();
    page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') errors.push(`[${m.type()}] ${m.text()}`); });
    page.on('pageerror', (e) => errors.push(`[pageerror] ${e.message}`));
    await page.goto(`http://127.0.0.1:${addr.port}/index.html?seed=1&pattern=1`);
    await page.waitForFunction(() => window.__app !== undefined && window.__app.frameCounter > 10, undefined, { timeout: 60_000 });

    const info = await page.evaluate(() => {
      const a = window.__app!;
      return { internal: [a.targets.width, a.targets.height], canvas: [a.canvas.width, a.canvas.height], dpr: devicePixelRatio, ts: a.timestamps.supported, format: a.presenter.format };
    });
    check('internal 540p independent of DPR 2', info.internal[0] === 960 && info.internal[1] === 540, JSON.stringify(info));
    check('canvas backing = CSS x DPR', info.canvas[0] === 2560 && info.canvas[1] === 1440, `${info.canvas}`);

    // Calibration strip under Standard, EV 0, nearest; hide HUD/panel so they don't cover patches.
    await page.evaluate(() => {
      const a = window.__app!;
      a.present.filter = 'nearest'; a.present.exposureEV = 0; a.present.tonemap = 'standard';
      a.hud.setVisible(false);
      (document.querySelector('.panel') as HTMLElement).style.display = 'none';
    });
    await sleep(300);
    const png = await page.screenshot();
    await writeFile(path.join(OUT, 'calibration.png'), png);
    const vals = [0.0, 0.001, 0.01, 0.05, 0.18, 0.5, 1.0, 4.0];
    const got = await samplePng(page, png, vals.map((_, i) => [(i + 0.5) / 8, 0.05] as [number, number]));
    const want = vals.map((v) => Math.round(srgb(v) * 255));
    const maxErr = Math.max(...got.map((g, i) => Math.abs(g - want[i])));
    check('Standard view transform = sRGB OETF (±1 LSB)', maxErr <= 1, `got ${got} want ${want}`);

    await page.evaluate(() => { const a = window.__app!; a.present.exposureEV = 1; });
    await sleep(200);
    const [g5] = await samplePng(page, await page.screenshot(), [[5.5 / 8, 0.05]]);
    check('exposure +1 EV doubles radiance (0.5 -> 1.0)', g5 === 255, `patch 0.5 at EV+1 = ${g5}`);

    // UI screenshot with HUD + panel, beauty.
    await page.evaluate(() => {
      const a = window.__app!;
      a.present.exposureEV = 0; a.present.filter = 'bilinear';
      a.hud.setVisible(true);
      (document.querySelector('.panel') as HTMLElement).style.display = '';
    });
    await sleep(1500);
    await page.screenshot({ path: path.join(OUT, 'ui-beauty.png') });

    // Fly camera: hold W for 400 ms (dt-integrated), check it moved along -Z.
    const z0 = await page.evaluate(() => window.__app!.camera.position[2]);
    await page.locator('#view').focus();
    await page.keyboard.down('KeyW');
    await sleep(400);
    await page.keyboard.up('KeyW');
    const z1 = await page.evaluate(() => window.__app!.camera.position[2]);
    check('W moves the camera forward', z1 < z0 - 0.1, `z ${z0.toFixed(3)} -> ${z1.toFixed(3)}`);
    await page.keyboard.press('Home');
    const z2 = await page.evaluate(() => window.__app!.camera.position[2]);
    check('Home resets the camera', Math.abs(z2 - z0) < 1e-9, `z ${z2}`);

    // Debug views + probe.
    await page.evaluate(() => {
      const a = window.__app!;
      a.selectDebugView(3); // test normal
      a.debugSettings.probeEnabled = true;
      a.debugSettings.probePixel = [480, 400];
      a.probePanel.setVisible(true);
    });
    await sleep(600);
    await page.screenshot({ path: path.join(OUT, 'debug-normal-probe.png') });
    const probe = await page.evaluate(() => {
      const f = window.__app!.probe.latest;
      return f ? { n: f.records.length, tags: f.records.map((r) => r.tag), aov: f.records.find((r) => r.tag === 1)?.value } : undefined;
    });
    check('probe records (test pattern tags 16,17 + aov)', !!probe && probe.tags.includes(16) && probe.tags.includes(17) && probe.tags.includes(1), JSON.stringify(probe));
    check('probe AOV = ground normal (0,1,0)', !!probe?.aov && Math.abs(probe.aov[1] - 1) < 1e-5, JSON.stringify(probe?.aov));

    await page.evaluate(() => { const a = window.__app!; a.selectDebugView(2); a.debugSettings.split = true; a.debugSettings.splitPos = 0.5; });
    await sleep(500);
    await page.screenshot({ path: path.join(OUT, 'debug-depth-split.png') });
    await page.evaluate(() => { const a = window.__app!; a.debugSettings.split = false; a.selectDebugView(4); });
    await sleep(400);
    await page.screenshot({ path: path.join(OUT, 'debug-cell-code.png') });

    // Timestamps (if supported) and counters.
    const hud = await page.evaluate(() => { const a = window.__app!; return { passes: a.timestamps.averages(), totals: a.totals, frames: a.frameCounter, skipped: a.timestamps.skipped }; });
    check('timestamps', !info.ts || hud.passes.some((p) => p.name === 'present' && p.samples > 0), JSON.stringify(hud.passes.map((p) => `${p.name}:${p.ms.toFixed(3)}`)));
    check('NaN/Inf counters zero', hud.totals.nan === 0 && hud.totals.inf === 0, JSON.stringify(hud.totals));

    // Resize: internal res follows aspect, still height 540.
    await page.evaluate(() => window.__app!.selectDebugView(0));
    await page.setViewportSize({ width: 1000, height: 1000 });
    await sleep(400);
    const res2 = await page.evaluate(() => [window.__app!.targets.width, window.__app!.targets.height, window.__app!.canvas.width]);
    check('resize: 1:1 -> 544x540 internal, canvas 2000', res2[0] === 544 && res2[1] === 540 && res2[2] === 2000, `${res2}`);
    await page.screenshot({ path: path.join(OUT, 'ui-square.png') });

    const relevant = errors.filter((e) => !/Download the React DevTools|favicon/.test(e));
    check('no console errors/warnings', relevant.length === 0, relevant.slice(0, 5).join(' | '));
  } finally {
    await browser.close();
    await vite.close();
    release();
  }
  await writeFile(path.join(OUT, 'report.json'), JSON.stringify({ checks, errors }, null, 2));
  const failed = checks.filter((c) => !c.ok).length;
  console.log(`${checks.length - failed}/${checks.length} checks passed; screenshots in ${OUT}`);
  return failed ? 1 : 0;
}

main().then((c) => process.exit(c), (e) => { console.error(e); process.exit(1); });

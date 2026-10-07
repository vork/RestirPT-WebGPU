// Edge study (docs/decisions/denoiser.md Changelog DN-16): the real app (540p, its default jitter, ReSTIR-interactive +
// denoiser) on the Cornell box with a point light added through the editor. Static camera: per-frame denoiser
// sinceChange / flags, the jitter mode and the resolutions; then screenshots of the canvas, static and with the M5.5
// smoke's camera motion.
//   npx tsx validation/harness/edge-app-probe.ts [tag]   → validation/out/edge-aa/app-probe-<tag>/
import { realpathSync, mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';
import { acquireGpuLock } from './gpu-lock.ts';
const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const OUT = path.join(ROOT, 'validation/out/edge-aa/app-probe-' + (process.argv[2] ?? 'base'));
mkdirSync(OUT, { recursive: true });
const vite = await createServer({ root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), logLevel: 'warn',
  server: { port: 0, host: '127.0.0.1', hmr: false, watch: null, fs: { allow: [ROOT, realpathSync(path.join(ROOT, 'node_modules'))] } } });
await vite.listen();
const port = (vite.httpServer!.address() as { port: number }).port;
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
const page = await ctx.newPage();
await page.addInitScript({ content: 'window.__name = (f) => f;' });
page.on('console', (m) => { if (m.type() === 'error') console.log('[page error]', m.text()); });
const release = await acquireGpuLock('edge-app-probe');
const out: Record<string, unknown> = {};
try {
  await page.goto(`http://127.0.0.1:${port}/index.html?seed=1&res=540p&scene=/validation/assets/cornell/cornell.glb`);
  await page.waitForFunction(() => { const w = window as any; const r = w.__integration?.renderer(); const a = w.__app; return !!r && !!a?.scene && r.scene?.scene === a.scene && !r.loading && r.ready; }, undefined, { timeout: 240_000, polling: 100 });
  await page.evaluate(() => {
    const w = window as any, store = w.__editor.editor.store;
    store.add({ type: 'point', power: 30, matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1, 0.35, -0.2, 1]) });
    w.__app.hud.setVisible(false);
  });
  const frames = (n: number) => page.evaluate((k) => new Promise<void>((res) => { const a = (window as any).__app; const f0 = a.frameCounter; const t = () => (a.frameCounter >= f0 + k ? res() : setTimeout(t, 10)); t(); }), n);
  await page.evaluate(async () => {
    const w = window as any, r = w.__integration.renderer();
    r.options.renderMode = 'restir';
    await r.setOptions({ restirMode: 'interactive', temporal: true });
    await r.prepareRestir();
    w.__app.resetTemporalHistory();
  });
  await page.waitForFunction(() => (window as any).__integration.renderer().denoisedLastFrame, undefined, { timeout: 180_000 });
  await frames(80);
  const trace: unknown[] = [];
  for (let i = 0; i < 16; i++) {
    await frames(1);
    trace.push(await page.evaluate(() => { const w = window as any, d = w.__integration.renderer().denoiser; return { since: d.sinceChange, flags: d.flags, jitter: w.__app.render.jitter, res: [w.__app.targets.width, w.__app.targets.height], canvas: [w.__app.canvas.width, w.__app.canvas.height] }; }));
  }
  out.staticTrace = trace;
  await page.locator('#view').screenshot({ path: path.join(OUT, 'static.png') });
  // camera motion as the M5.5 smoke
  await page.evaluate(() => {
    const w = window as any, app = w.__app;
    const b = app.scene.bounds, c = [0, 1, 2].map((i: number) => 0.5 * (b.min[i] + b.max[i]));
    const ext = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
    const eye0 = [...app.camera.position]; let s = 0;
    app.beforeFrame.add((f: any) => { if (!f.advance) return; s++; app.camera.lookAt([eye0[0] + 0.04 * ext * Math.sin(0.05 * s), eye0[1], eye0[2] + 0.03 * ext * Math.cos(0.05 * s)], c); });
  });
  await frames(30);
  out.movingTrace = await page.evaluate(() => { const d = (window as any).__integration.renderer().denoiser; return { since: d.sinceChange, flags: d.flags }; });
  await page.locator('#view').screenshot({ path: path.join(OUT, 'moving.png') });
} finally {
  release();
  writeFileSync(path.join(OUT, 'probe.json'), JSON.stringify(out, null, 1));
  console.log(JSON.stringify(out));
  await browser.close(); await vite.close();
}
process.exit(0);

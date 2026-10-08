// App screenshots for the docs (README hero images) and the UI review (the panel tree as a user sees it on load,
// and fully expanded). The real app in headless Chrome; the GPU lock is taken for the page loads.
//   npx tsx validation/harness/ui-shots.ts [--panels <dir>] [--hero <dir>] [--no-lock]
//     --panels <dir>   panel-default.png (1280x800 page at load), panel-collapsed.png (top-level folders),
//                      panel-expanded.png (every folder expanded), panel-debug.png (Debug views, a ReSTIR code view)
//     --hero <dir>     sponza.png, cornell.png: ReSTIR-interactive at the app defaults, HUD and panel hidden
import { mkdirSync, realpathSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium, type Browser, type Page } from 'playwright';
import { createServer } from 'vite';
import type { App } from '../../src/app/app.ts';
import type { EditorHandle } from '../../src/app/editor/index.ts';
import type { Integration } from '../../src/app/integration.ts';
import { acquireGpuLock } from './gpu-lock.ts';

declare global {
  interface Window { __app?: App; __integration?: Integration; __editor?: EditorHandle }
}

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const { values: args } = parseArgs({ options: { panels: { type: 'string' }, hero: { type: 'string' }, 'no-lock': { type: 'boolean', default: false } } });
const withTimeout = <T>(p: Promise<T>, ms: number) => Promise.race([p, new Promise<void>((r) => setTimeout(r, ms))]);

async function frames(page: Page, n: number): Promise<void> {
  const f0 = await page.evaluate(() => window.__app!.frameCounter);
  await page.waitForFunction((t) => window.__app!.frameCounter >= t, f0 + n, { timeout: 300_000, polling: 50 });
}

async function open(browser: Browser, port: number, query: Record<string, string>, viewport = { width: 1280, height: 800 }): Promise<{ page: Page; close(): Promise<void> }> {
  const ctx = await browser.newContext({ viewport, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.addInitScript({ content: 'window.__name = (f) => f;' });
  page.on('console', (m) => { if (m.type() === 'error') console.log(`[page:error] ${m.text()}`); });
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
  await page.goto(`http://127.0.0.1:${port}/index.html?${new URLSearchParams({ seed: '1', ...query })}`);
  await page.waitForFunction(() => {
    const r = window.__integration?.renderer(); const a = window.__app;
    return !!r && !!a?.scene && r.scene?.scene === a.scene && !r.loading && r.ready;
  }, undefined, { timeout: 300_000, polling: 100 });
  return { page, close: async () => { await withTimeout(page.close(), 20_000); await withTimeout(ctx.close(), 10_000); } };
}

async function panels(browser: Browser, port: number, out: string): Promise<void> {
  const h = await open(browser, port, {});
  try {
    await frames(h.page, 20);
    await h.page.screenshot({ path: path.join(out, 'panel-default.png') });
    // the top-level folders only (every folder collapsed)
    await h.page.evaluate(() => { for (const f of window.__app!.panel!.pane.children as { expanded?: boolean }[]) if (typeof f.expanded === 'boolean') f.expanded = false; });
    await h.page.waitForTimeout(800);
    await h.page.locator('.panel').screenshot({ path: path.join(out, 'panel-collapsed.png') });
    // every folder expanded (recursively), the panel unclipped, the page as tall as the panel
    await h.page.evaluate(() => {
      type F = { expanded?: boolean; children?: F[] };
      const walk = (f: F) => { if (typeof f.expanded === 'boolean') f.expanded = true; for (const c of f.children ?? []) walk(c); };
      walk(window.__app!.panel!.pane as unknown as F);
      const el = document.querySelector('.panel') as HTMLElement;
      el.style.maxHeight = 'none';
      el.style.overflow = 'visible';
    });
    await h.page.waitForTimeout(800); // folder expand transitions
    const height = await h.page.evaluate(() => (document.querySelector('.panel') as HTMLElement).scrollHeight);
    await h.page.setViewportSize({ width: 1280, height: Math.max(800, height + 40) });
    await h.page.waitForTimeout(600); // folder expand transitions
    await h.page.locator('.panel').screenshot({ path: path.join(out, 'panel-expanded.png') });
    console.log(`panels: ${path.join(out, 'panel-default.png')}, panel-expanded.png (${height} px)`);
    // Debug views: ReSTIR on, the category picked through the real <select>, then a code view with its legend
    await h.page.evaluate(async () => {
      const r = window.__integration!.renderer()!;
      r.options.renderMode = 'restir';
      await r.prepareRestir();
      window.__app!.resetHistory();
      window.__app!.panel!.refresh();
    });
    await frames(h.page, 10);
    const sel = h.page.locator('.panel .tp-lblv', { hasText: 'category' }).locator('select');
    if (await sel.count()) {
      await sel.first().selectOption({ label: 'ReSTIR: reservoir' });
      const mode = await h.page.evaluate(() => window.__app!.debugSettings.mode);
      console.log(`category select → view ${mode} (expected the first reservoir view, 400)`);
    }
    await h.page.evaluate(() => { window.__app!.selectDebugView(406); window.__app!.panel!.folders.debug.element.id = 'shot-debug'; });
    await frames(h.page, 4);
    await h.page.locator('#shot-debug').screenshot({ path: path.join(out, 'panel-debug.png') });
  } finally { await h.close(); }
}

type L = { type: 'sun' | 'rect'; power: number; sizeX?: number; sizeY?: number; visibleToCamera?: boolean; matrix: number[] };
async function hero(browser: Browser, port: number, out: string): Promise<void> {
  const shots: { name: string; scene: string; env?: string; camera?: { eye: [number, number, number]; target: [number, number, number] }; lights?: L[]; ceilingLight?: boolean; frames: number }[] = [
    {
      name: 'sponza', scene: '/validation/assets/downloaded/sponza/Sponza.gltf', env: '/validation/assets/downloaded/hdri/kloofendal_48d_partly_cloudy_puresky_1k.hdr',
      camera: { eye: [-9, 2, -0.5], target: [6, 3.5, 0.5] }, frames: 64,
      // a sun through the open roof (rotated about X then Y; emits along −Z)
      lights: [{ type: 'sun', power: 8, matrix: sunMatrix(-62, 90) }],
    },
    // the Cornell glTF has no lights: a camera-visible rect under the ceiling (Mode B)
    { name: 'cornell', scene: '/validation/assets/cornell/cornell.glb', ceilingLight: true, frames: 64 },
  ];
  for (const s of shots) {
    const h = await open(browser, port, { res: '720p', hud: '0', scene: s.scene, ...(s.env ? { env: s.env } : {}) }, { width: 1280, height: 720 });
    try {
      if (s.env) await h.page.waitForFunction(() => !!window.__app!.env && !!window.__integration!.renderer()!.env?.present, undefined, { timeout: 120_000, polling: 100 });
      await h.page.evaluate(async (c) => {
        const r = window.__integration!.renderer()!, a = window.__app!;
        for (const sel of ['.panel', '.timeline-bar']) (document.querySelector(sel) as HTMLElement | null)?.style.setProperty('display', 'none');
        a.render.overlay = false; // no axes / light gizmos
        const store = window.__editor!.editor.store!;
        if (c.lights) for (const l of c.lights) store.add({ ...l, matrix: new Float32Array(l.matrix) } as never);
        if (c.ceilingLight) {
          const b = a.scene!.bounds, x = (b.min[0] + b.max[0]) / 2, z = (b.min[2] + b.max[2]) / 2, y = b.max[1] - 2e-3 * (b.max[1] - b.min[1]);
          // local −Z (emission) → world −Y
          store.add({ type: 'rect', power: 2, sizeX: 0.25, sizeY: 0.2, visibleToCamera: true, matrix: new Float32Array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, x, y, z, 1]) } as never);
        }
        if (c.camera) a.camera.lookAt(c.camera.eye, c.camera.target);
        r.options.renderMode = 'restir';
        await r.setOptions({ restirMode: 'interactive', temporal: true });
        await r.prepareRestir();
        a.resetHistory();
      }, { camera: s.camera, lights: s.lights, ceilingLight: !!s.ceilingLight });
      await frames(h.page, s.frames);
      const file = path.join(out, `${s.name}.png`);
      await h.page.locator('#view').screenshot({ path: file });
      console.log(`hero: ${file}`);
    } finally { await h.close(); }
  }
}

/** Sun orientation: rotate the −Z emission direction by pitch about X, then yaw about Y (degrees). */
function sunMatrix(pitchDeg: number, yawDeg: number): number[] {
  const p = (pitchDeg * Math.PI) / 180, y = (yawDeg * Math.PI) / 180;
  const cp = Math.cos(p), sp = Math.sin(p), cy = Math.cos(y), sy = Math.sin(y);
  // R = Ry · Rx, column-major
  return [cy, 0, -sy, 0, sy * sp, cp, cy * sp, 0, sy * cp, -sp, cy * cp, 0, 0, 0, 0, 1];
}

async function main(): Promise<number> {
  if (!args.panels && !args.hero) { console.error('nothing to do: pass --panels <dir> and/or --hero <dir>'); return 2; }
  const vite = await createServer({
    root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), logLevel: 'warn',
    server: { port: 0, host: '127.0.0.1', hmr: false, watch: null, fs: { allow: [ROOT, realpathSync(path.join(ROOT, 'node_modules')), realpathSync(path.join(ROOT, 'validation/assets/downloaded'))] } },
  });
  await vite.listen();
  try {
    const port = (vite.httpServer!.address() as { port: number }).port;
    const browser = await chromium.launch({ channel: 'chrome', headless: true });
    const release = args['no-lock'] ? () => {} : await acquireGpuLock('ui-shots');
    try {
      if (args.panels) { mkdirSync(args.panels, { recursive: true }); await panels(browser, port, args.panels); }
      if (args.hero) { mkdirSync(args.hero, { recursive: true }); await hero(browser, port, args.hero); }
    } finally {
      release();
      await withTimeout(browser.close(), 30_000);
    }
  } finally {
    await withTimeout(vite.close(), 10_000);
  }
  return 0;
}

main().then((c) => process.exit(c), (e: unknown) => { console.error(e); process.exit(1); });

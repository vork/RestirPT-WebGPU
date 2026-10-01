// SCRATCH (firefly diagnosis, not for merge): the real app in headless Chrome on validation/assets/cornell/cornell.usda
// (704×540 internal), accumulate N frames per configuration and dump the accumulated colour target (linear f32 rgb) +
// HUD lines to validation/out/<run>/<cfg>.{f32,json}. Takes the GPU lock per group of configs (≤ ~10 min each).
//   npx tsx validation/harness/diag-fireflies.ts --run diag-ff --configs interactive,pt --frames 1000
import { mkdirSync, realpathSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium, type Page } from 'playwright';
import { createServer } from 'vite';
import type { App } from '../../src/app/app.ts';
import type { Integration } from '../../src/app/integration.ts';
import type { RestirSettings } from '../../src/core/render/restir/presets.ts';
import { acquireGpuLock } from './gpu-lock.ts';

declare global { interface Window { __app?: App; __integration?: Integration; __stats?: unknown } }

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const { values: args } = parseArgs({
  options: {
    run: { type: 'string', default: 'diag-ff' }, configs: { type: 'string', default: 'interactive' }, frames: { type: 'string', default: '1000' },
    scene: { type: 'string', default: '/validation/assets/cornell/cornell.usda' }, 'lock-min': { type: 'string', default: '10' },
    seed: { type: 'string', default: '1' }, stats: { type: 'string', default: '0' },
  },
});

interface Cfg {
  render: 'pt' | 'restir'; restirMode?: 'interactive' | 'unbiased' | 'criteria2022' | 'offline' | 'initial'; temporal?: boolean;
  over?: Partial<RestirSettings>; watertight?: boolean; ptRr?: boolean; frames?: number;
}
const I = { render: 'restir', restirMode: 'interactive', temporal: true } as const;
const CFG: Record<string, Cfg> = {
  interactive: { ...I },
  interactive_wt: { ...I, watertight: true },
  interactive_noRR: { ...I, over: { rr: false } },
  interactive_noBoost: { ...I, over: { boostSlots: 0 } },
  interactive_2022: { ...I, over: { criteria: '2022' } },
  spatial_only: { ...I, temporal: false },
  temporal_only: { ...I, over: { rounds: 0 } },
  initial_only: { ...I, temporal: false, over: { rounds: 0 } },
  initial_only_noRR: { ...I, temporal: false, over: { rounds: 0, rr: false } },
  unbiased: { render: 'restir', restirMode: 'unbiased', temporal: true },
  spatial_only_noRR: { ...I, temporal: false, over: { rr: false } },
  interactive_cap5: { ...I, over: { cCap: 5 } },
  interactive_tau1e3: { ...I, over: { tau: 1e-3 } },
  interactive_tau1e5: { ...I, over: { tau: 1e-5 } },
  interactive_talbot: { ...I, over: { temporalMis: 'talbot' } },
  temporal_only_talbot: { ...I, over: { rounds: 0, temporalMis: 'talbot' } },
  temporal_only_cap5: { ...I, over: { rounds: 0, cCap: 5 } },
  interactive_talbot_long: { ...I, over: { temporalMis: 'talbot' }, frames: 12000 },
  temporal_only_long: { ...I, over: { rounds: 0 }, frames: 12000 },
  interactive_long: { ...I, frames: 12000 },
  pt: { render: 'pt' },
  pt_rr: { render: 'pt', ptRr: true },
  pt_wt: { render: 'pt', watertight: true },
  pt_ref: { render: 'pt', frames: 65536 },
  pt_eqtime: { render: 'pt', frames: 4200 },
};

async function waitFrames(page: Page, n: number): Promise<number> {
  const t0 = Date.now();
  await page.waitForFunction((t) => window.__app!.frameIndex >= t, n, { timeout: 900_000, polling: 100 });
  return (Date.now() - t0) / 1000;
}

async function configure(page: Page, c: Cfg): Promise<void> {
  await page.evaluate(async (c) => {
    const r = window.__integration!.renderer()!;
    const app = window.__app!;
    app.setPaused(true);
    const wt = !!c.watertight;
    if (r.options.watertight !== wt) await r.setOptions({ watertight: wt });
    if (c.render === 'pt') {
      r.options.renderMode = 'pt';
      await r.setOptions({ rr: !!c.ptRr });
    } else {
      r.options.renderMode = 'restir';
      await r.setOptions({ restirMode: c.restirMode, temporal: c.temporal, rr: false });
      const pass = await r.prepareRestir();
      if (c.over) { pass!.setSettings(c.over); await pass!.prepare(); }
    }
    app.setPaused(false);
  }, c);
  await page.waitForFunction(() => { const r = window.__integration!.renderer()!; return r.ready && !r.loading; }, undefined, { timeout: 180_000, polling: 100 });
  if (c.render === 'restir') {
    await page.evaluate(async (c) => {
      const r = window.__integration!.renderer()!;
      const pass = await r.prepareRestir();
      if (c.over) { pass!.setSettings(c.over); await pass!.prepare(); }
    }, c);
  }
  const f0 = await page.evaluate(() => window.__app!.frameCounter);
  await page.waitForFunction((t) => window.__app!.frameCounter >= t, f0 + 30, { timeout: 180_000, polling: 50 });
  await page.evaluate(() => { window.__app!.setPaused(true); window.__app!.resetTemporalHistory(); });
  await page.waitForFunction(() => window.__app!.frameIndex === 0, undefined, { timeout: 180_000, polling: 10 });
  await page.evaluate(() => window.__app!.setPaused(false));
}

async function readColour(page: Page): Promise<{ w: number; h: number; b64: string; hud: string[]; settings: unknown; frameIndex: number; camera: unknown }> {
  return page.evaluate(async () => {
    const app = window.__app!;
    app.setPaused(true);
    const r = window.__integration!.renderer()!;
    const device = app.device;
    const t = app.targets;
    await device.queue.onSubmittedWorkDone();
    const bpr = Math.ceil((t.width * 16) / 256) * 256;
    const cb = device.createBuffer({ size: bpr * t.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = device.createCommandEncoder();
    e.copyTextureToBuffer({ texture: t.color }, { buffer: cb, bytesPerRow: bpr }, [t.width, t.height]);
    device.queue.submit([e.finish()]);
    await cb.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(cb.getMappedRange().slice(0)); cb.unmap(); cb.destroy();
    const out = new Float32Array(t.width * t.height * 3);
    for (let y = 0; y < t.height; y++) for (let x = 0; x < t.width; x++) for (let c = 0; c < 3; c++) out[3 * (y * t.width + x) + c] = f[(y * bpr) / 4 + x * 4 + c];
    const u8 = new Uint8Array(out.buffer);
    let s = ''; const CH = 0x8000;
    for (let i = 0; i < u8.length; i += CH) s += String.fromCharCode(...u8.subarray(i, i + CH));
    const settings = r.options.renderMode === 'restir' ? r.restir?.settings : { pt: true, rr: r.options.rr };
    return { w: t.width, h: t.height, b64: btoa(s), hud: r.hudLines(), settings, frameIndex: app.frameIndex, camera: { camToWorld: Array.from(app.camera.camToWorld()), yfov: app.camera.yfov } };
  });
}

async function main(): Promise<void> {
  const OUT = path.join(ROOT, 'validation/out', args.run!);
  mkdirSync(OUT, { recursive: true });
  const vite = await createServer({
    root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), logLevel: 'warn',
    server: { port: 0, host: '127.0.0.1', hmr: false, watch: null, fs: { allow: [ROOT, realpathSync(path.join(ROOT, 'node_modules'))] } },
  });
  await vite.listen();
  const port = (vite.httpServer!.address() as { port: number }).port;
  const names = args.configs!.split(',');
  const defFrames = Number(args.frames);
  const lockMs = Number(args['lock-min']) * 60_000;
  const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--disable-gpu-vsync', '--disable-frame-rate-limit'] });
  let release: (() => void) | undefined;
  let lockT0 = 0;
  try {
    const ctx = await browser.newContext({ viewport: { width: 1056, height: 810 } });
    const page = await ctx.newPage();
    await page.addInitScript({ content: 'window.__name = (f) => f;' });
    page.on('console', (m) => { if (m.type() === 'error' || /error/i.test(m.text())) console.log(`[page:${m.type()}] ${m.text()}`); });
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
    release = await acquireGpuLock('diag-fireflies'); lockT0 = Date.now();
    const q = new URLSearchParams({ seed: args.seed!, res: '540p', scene: args.scene! });
    await page.goto(`http://127.0.0.1:${port}/index.html?${q}`);
    await page.waitForFunction(() => {
      const r = window.__integration?.renderer(); const a = window.__app;
      return !!r && !!a?.scene && r.scene?.scene === a.scene && !r.loading && r.ready;
    }, undefined, { timeout: 180_000, polling: 100 });
    const scene = await page.evaluate(() => {
      const s = window.__app!.scene!;
      return { positions: Array.from(s.geometry.positions), indices: Array.from(s.geometry.indices), triMaterial: Array.from(s.geometry.triMaterial),
        materials: s.materials.map((m) => m.name), lights: s.lights.map((l) => ({ ...l, matrix: Array.from(l.matrix) })), warnings: s.warnings,
        camera: { camToWorld: Array.from(window.__app!.camera.camToWorld()), yfov: window.__app!.camera.yfov } };
    });
    writeFileSync(path.join(OUT, 'scene.json'), JSON.stringify(scene));
    for (const name of names) {
      const c = CFG[name];
      if (!c) throw new Error(`unknown config ${name}`);
      const n = c.frames ?? defFrames;
      if (Date.now() - lockT0 > lockMs) {
        release(); release = undefined;
        await page.evaluate(() => window.__app!.setPaused(true));
        await new Promise((r) => setTimeout(r, 20_000));
        release = await acquireGpuLock('diag-fireflies'); lockT0 = Date.now();
      }
      await configure(page, c);
      let secs = await waitFrames(page, n);
      let rd = await readColour(page);
      for (let tries = 0; rd.frameIndex < n && tries < 3; tries++) {
        console.log(`[diag] ${name}: frameIndex ${rd.frameIndex} < ${n} (reset mid-run?), waiting again`);
        await page.evaluate(() => window.__app!.setPaused(false));
        secs = await waitFrames(page, n);
        rd = await readColour(page);
      }
      await page.evaluate(() => window.__app!.setPaused(false));
      writeFileSync(path.join(OUT, `${name}.f32`), Buffer.from(rd.b64, 'base64'));
      const meta = { name, cfg: c, frames: rd.frameIndex, w: rd.w, h: rd.h, secs, fps: rd.frameIndex / secs, hud: rd.hud, settings: rd.settings, camera: rd.camera };
      writeFileSync(path.join(OUT, `${name}.json`), JSON.stringify(meta, null, 1));
      console.log(`[diag] ${name}: ${rd.frameIndex} frames ${rd.w}x${rd.h} in ${secs.toFixed(1)} s`);
      console.log(rd.hud.filter((l) => /ReSTIR|PT|error|light/i.test(l)).join('\n'));
    }
    await ctx.close();
  } finally {
    release?.();
    await browser.close().catch(() => undefined);
    await vite.close();
  }
}
main().then(() => process.exit(0), (e: unknown) => { console.error(e); process.exit(1); });

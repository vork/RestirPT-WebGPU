// M7 app smoke (docs/decisions/m7-api.md §6, M7 Gate 0): the real app in headless Chrome at 540p.
//   Sponza (glTF, normal-mapped)    NORMAL_MAP compiled (arena tangent section); PT and ReSTIR-interactive render finite and
//                                   error-free; the shading-normal views 320–327 are registered, finite, and 320 / 322 /
//                                   324 / 327 are written; with the normal maps on the closure normal differs from Ns
//                                   (view 325 > 0 on some pixels)
//   m7_textured.usda (USD)          UsdUVTexture network incl. a normal texture: NORMAL_MAP compiled; ReSTIR finite
//   Cornell (glTF, no normal map)   NORMAL_MAP absent (the M6 pipelines, U-M7-BITS); views 320–327 finite (322: no tangent)
// Global per page: no console / WebGPU errors, NaN/Inf = 0. The GPU lock is taken for the page loads.
//   npx tsx validation/harness/m7-app-smoke.ts [--run <id>] [--no-lock]
import { mkdirSync, realpathSync, writeFileSync } from 'node:fs';
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
  interface Window { __app?: App; __integration?: Integration; __editor?: EditorHandle; __webgpuErrors?: string[] }
}

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const VIEWS = [320, 321, 322, 323, 324, 325, 326, 327];
const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const { values: args } = parseArgs({ options: { run: { type: 'string' }, 'no-lock': { type: 'boolean', default: false } } });

const checks: { name: string; ok: boolean; detail?: string }[] = [];
const check = (name: string, ok: boolean, detail?: string) => { checks.push({ name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`); return ok; };
const withTimeout = <T>(p: Promise<T>, ms: number) => Promise.race([p, new Promise<void>((r) => setTimeout(r, ms))]);

async function frames(page: Page, n: number): Promise<void> {
  const f0 = await page.evaluate(() => window.__app!.frameCounter);
  await page.waitForFunction((t) => window.__app!.frameCounter >= t, f0 + n, { timeout: 240_000, polling: 50 });
}

/** Colour target (non-finite count, mean) and the active debug AOV (non-finite, written pixels, max of channel 0). */
async function readState(page: Page): Promise<{ nonFinite: number; mean: number; aovNonFinite: number; aovWritten: number; aovMax: number }> {
  return page.evaluate(async () => {
    const app = window.__app!, device = app.device, t = app.targets;
    await device.queue.onSubmittedWorkDone();
    const bpr = Math.ceil((t.width * 16) / 256) * 256;
    const cb = device.createBuffer({ size: bpr * t.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = device.createCommandEncoder();
    e.copyTextureToBuffer({ texture: t.color }, { buffer: cb, bytesPerRow: bpr }, [t.width, t.height]);
    const dbg = app.debug, view = dbg.activeView();
    const L = 64 + 256 * 32;
    const ab = view ? device.createBuffer({ size: dbg.width * dbg.height * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) : undefined;
    if (ab) e.copyBufferToBuffer(dbg.buffer, L, ab, 0, dbg.width * dbg.height * 16);
    device.queue.submit([e.finish()]);
    await cb.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(cb.getMappedRange().slice(0)); cb.unmap(); cb.destroy();
    let nonFinite = 0, sum = 0, n = 0;
    for (let y = 0; y < t.height; y++) for (let x = 0; x < t.width; x++) for (let c = 0; c < 3; c++) {
      const v = f[(y * bpr) / 4 + x * 4 + c];
      if (!Number.isFinite(v)) nonFinite++; else { sum += v; n++; }
    }
    let aovNonFinite = 0, aovWritten = 0, aovMax = 0;
    if (ab && view) {
      await ab.mapAsync(GPUMapMode.READ);
      const a = new Float32Array(ab.getMappedRange().slice(0)); ab.unmap(); ab.destroy();
      const au = new Uint32Array(a.buffer);
      for (let i = 0; i < dbg.width * dbg.height; i++) {
        if (au[4 * i] !== 0 && au[4 * i] !== 0xffffffff) aovWritten++;
        if (view.kind !== 'code') {
          for (let c = 0; c < (view.kind === 'vec3' ? 3 : 1); c++) if (!Number.isFinite(a[4 * i + c])) aovNonFinite++;
          if (Number.isFinite(a[4 * i]) && a[4 * i] > aovMax) aovMax = a[4 * i];
        }
      }
    }
    return { nonFinite, mean: n ? sum / n : 0, aovNonFinite, aovWritten, aovMax };
  });
}

const errState = (page: Page) => page.evaluate(() => {
  const r = window.__integration!.renderer()!;
  return { err: r.lastError, rs: r.restirError, gpu: window.__webgpuErrors ?? [] };
});

async function openScene(browser: Browser, port: number, scene: string, errors: string[]): Promise<{ page: Page; close: () => Promise<void> }> {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.addInitScript({ content: 'window.__name = (f) => f;' });
  page.on('console', (m) => { if (m.type() === 'error') errors.push(`[${scene}] ${m.text()}`); });
  page.on('pageerror', (e) => errors.push(`[${scene}] [pageerror] ${e.message}`));
  const q = new URLSearchParams({ seed: '1', res: '540p', scene: `/${scene}` });
  await page.goto(`http://127.0.0.1:${port}/index.html?${q}`);
  await page.waitForFunction(() => {
    const r = window.__integration?.renderer(); const a = window.__app;
    return !!r && !!a?.scene && r.scene?.scene === a.scene && !r.loading && r.ready;
  }, undefined, { timeout: 300_000, polling: 100 });
  return { page, close: async () => { await withTimeout(page.close(), 20_000); await withTimeout(ctx.close(), 10_000); } };
}

const sceneDefines = (page: Page) => page.evaluate(() => {
  const sg = window.__integration!.renderer()!.scene!;
  const d = sg.defines() as Record<string, unknown>;
  return { normalMap: d.NORMAL_MAP === true, hasKey: 'NORMAL_MAP' in d };
});

async function renderModes(page: Page, tag: string, OUT: string): Promise<void> {
  for (const mode of ['pt', 'restir'] as const) {
    await page.evaluate(async (m) => {
      const r = window.__integration!.renderer()!;
      r.options.renderMode = m;
      if (m === 'restir') { await r.setOptions({ restirMode: 'interactive', temporal: true }); await r.prepareRestir(); }
      window.__app!.resetHistory();
    }, mode);
    await frames(page, mode === 'pt' ? 16 : 30);
    const s = await readState(page), e = await errState(page);
    check(`${tag}: ${mode === 'pt' ? 'PT' : 'ReSTIR-interactive'} renders finite, error-free`, s.nonFinite === 0 && s.mean > 0 && !e.rs && !e.err && e.gpu.length === 0,
      `mean ${s.mean.toExponential(3)}${e.rs ? ` ${e.rs}` : ''}${e.err ? ` ${e.err}` : ''}${e.gpu[0] ? ` ${e.gpu[0]}` : ''}`);
    await page.locator('#view').screenshot({ path: path.join(OUT, `${tag}-${mode}.png`) });
  }
}

async function views(page: Page, tag: string, OUT: string, mustWrite: number[], shots: number[]): Promise<Record<number, { written: number; max: number }>> {
  const bad: string[] = [];
  const res: Record<number, { written: number; max: number }> = {};
  await page.evaluate(() => window.__app!.selectDebugView(320));   // the first selection compiles the shading-debug pass
  await frames(page, 12);
  for (const id of VIEWS) {
    await page.evaluate((v) => window.__app!.selectDebugView(v), id);
    await frames(page, 4);
    const s = await readState(page);
    res[id] = { written: s.aovWritten, max: s.aovMax };
    if (s.aovNonFinite > 0) bad.push(`${id}: ${s.aovNonFinite} non-finite`);
    if (shots.includes(id)) await page.locator('#view').screenshot({ path: path.join(OUT, `${tag}-view-${id}.png`) });
  }
  await page.evaluate(() => window.__app!.selectDebugView(0));
  const reg = await page.evaluate((ids) => window.__app!.debug.registry.list().filter((v) => ids.includes(v.id)).length, VIEWS);
  check(`${tag}: views 320–327 registered and finite`, reg === VIEWS.length && bad.length === 0, `${reg} registered${bad.length ? `; ${bad.join('; ')}` : ''}`);
  if (mustWrite.length) check(`${tag}: views ${mustWrite.join(', ')} written`, mustWrite.every((v) => res[v].written > 0), JSON.stringify(Object.fromEntries(mustWrite.map((v) => [v, res[v].written]))));
  return res;
}

async function main(): Promise<number> {
  const runId = args.run ?? `m7-app-smoke-${stamp()}`;
  const OUT = path.join(ROOT, 'validation/out', runId);
  mkdirSync(OUT, { recursive: true });
  const vite = await createServer({
    root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), logLevel: 'warn',
    server: { port: 0, host: '127.0.0.1', hmr: false, watch: null, fs: { allow: [ROOT, realpathSync(path.join(ROOT, 'node_modules')), realpathSync(path.join(ROOT, 'validation/assets/downloaded'))] } },
  });
  await vite.listen();
  const port = (vite.httpServer!.address() as { port: number }).port;
  const errors: string[] = [];
  try {
    const browser = await chromium.launch({ channel: 'chrome', headless: true });
    const release = args['no-lock'] ? () => {} : await acquireGpuLock('m7-app-smoke');
    try {
      type L = { type: 'point' | 'rect'; power: number; sizeX?: number; sizeY?: number; visibleToCamera?: boolean; matrix: number[] };
      const cornellLights: L[] = [
        { type: 'point', power: 30, matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1, 0.35, -0.2, 1] },
        { type: 'rect', power: 20, sizeX: 0.15, sizeY: 0.1, visibleToCamera: false, matrix: [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0.5, -0.28, 1] },
      ];
      const sponzaLights: L[] = [
        { type: 'point', power: 4000, matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 4, 0, 1] },
        { type: 'rect', power: 6000, sizeX: 3, sizeY: 1.5, visibleToCamera: false, matrix: [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 2, 9, 0, 1] },
      ];
      const scenes: { tag: string; file: string; nm: boolean; write: number[]; shots: number[]; addLights?: L[]; camera?: { eye: [number, number, number]; target: [number, number, number] } }[] = [
        { tag: 'sponza', file: 'validation/assets/downloaded/sponza/Sponza.gltf', nm: true, write: [320, 322, 324, 327], shots: [320, 322, 323, 324, 325], addLights: sponzaLights,
          camera: { eye: [-9, 2, -0.5], target: [6, 3.5, 0.5] } },   // inside the atrium (the file camera frames the bounds from outside)
        { tag: 'usd-textured', file: 'validation/assets/usd-m7/m7_textured.usda', nm: true, write: [320, 324], shots: [320, 323] },
        { tag: 'cornell', file: 'validation/assets/cornell/cornell.glb', nm: false, write: [320, 321], shots: [], addLights: cornellLights },
      ];
      for (const s of scenes) {
        let h: { page: Page; close: () => Promise<void> } | undefined;
        try {
          h = await openScene(browser, port, s.file, errors);
          const d = await sceneDefines(h.page);
          check(`${s.tag}: NORMAL_MAP ${s.nm ? 'compiled (normal-mapped material, arena tangents)' : 'absent (pre-M7 pipelines)'}`, s.nm ? d.normalMap : !d.hasKey, JSON.stringify(d));
          if (s.addLights) {
            // the glTF has no lights: a point and a rect through the editor (as the M6 smoke)
            await h.page.evaluate((L) => {
              const store = window.__editor!.editor.store!;
              for (const l of L) store.add({ ...l, matrix: new Float32Array(l.matrix) });
            }, s.addLights);
          }
          if (s.camera) await h.page.evaluate((c) => { window.__app!.camera.lookAt(c.eye, c.target); window.__app!.resetHistory(); }, s.camera);
          await frames(h.page, 10);
          await renderModes(h.page, s.tag, OUT);
          const v = await views(h.page, s.tag, OUT, s.write, s.shots);
          if (s.tag === 'sponza') check('sponza: the normal maps tilt the closure normal (view 325 angle(N, Ns) max > 1°)', v[325].max > 1, `max ${v[325].max.toFixed(2)}°`);
          if (!s.nm) check(`${s.tag}: no tangent without normal maps (view 322 max 0)`, v[322].max === 0, `max ${v[322].max}`);
          const t = await h.page.evaluate(() => ({ ...window.__app!.runTotals }));
          check(`${s.tag}: NaN/Inf = 0 (debug counters)`, t.nan === 0 && t.inf === 0, JSON.stringify(t));
        } catch (e) {
          check(`${s.tag}: page run completed`, false, e instanceof Error ? e.message.split('\n')[0] : String(e));
          if (h) await withTimeout(h.page.screenshot({ path: path.join(OUT, `${s.tag}-failure.png`) }), 10_000).catch(() => undefined);
        } finally {
          if (h) await h.close();
        }
      }
    } finally {
      release();
      await withTimeout(browser.close(), 30_000);
    }
    check('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  } finally {
    await withTimeout(vite.close(), 10_000);
  }
  writeFileSync(path.join(OUT, 'report.json'), JSON.stringify({ runId, checks }, null, 1));
  const failed = checks.filter((c) => !c.ok).length;
  console.log(`${failed ? 'FAIL' : 'PASS'} m7 app smoke: ${checks.length - failed}/${checks.length} (screenshots: ${path.relative(ROOT, OUT)})`);
  return failed ? 1 : 0;
}

main().then((c) => process.exit(c), (e: unknown) => { console.error(e); process.exit(1); });

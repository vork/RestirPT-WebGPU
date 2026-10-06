// M6 app smoke (restir-m6-api.md §4 Gate 0, MD9, MD13): the real app in headless Chrome at 540p on the Cornell box with a
// point and a rect light added through the editor, ReSTIR-interactive.
//   defaults     light mode B (the product default after rung 3.11, MD9 / Q4) for the PT and ReSTIR; the interactive preset
//                carries the σ 16 pairing maps, RIS-NEE, dual MVs and the duplication map; the HUD shows them
//   toggles      every M6 feature off and on again (pipeline variants recompile; finite image, no error)
//   light modes  A, A′, B in ReSTIR (variants; finite, no error; the HUD names the mode)
//   views        pairing offset / reciprocity and duplication count / cap (471–479) render finite AOVs; 471, 477, 478
//                are written
//   motion       a camera pan with dual MVs: finite
//   Q3           no timestampWrites on any ReSTIR pass
// Global: no console / WebGPU errors, NaN/Inf = 0. The GPU lock is taken for the page load.
//   npx tsx validation/harness/m6-app-smoke.ts [--run <id>] [--no-lock]
import { mkdirSync, realpathSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium, type Page } from 'playwright';
import { createServer } from 'vite';
import type { App } from '../../src/app/app.ts';
import type { EditorHandle } from '../../src/app/editor/index.ts';
import type { Integration } from '../../src/app/integration.ts';
import { acquireGpuLock } from './gpu-lock.ts';

declare global {
  interface Window { __app?: App; __integration?: Integration; __editor?: EditorHandle; __webgpuErrors?: string[]; __tsPasses?: string[]; __m6Pan?: boolean }
}

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const CORNELL = 'validation/assets/cornell/cornell.glb';
const M6_VIEWS = [471, 472, 473, 474, 475, 476, 477, 478, 479];
const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const { values: args } = parseArgs({ options: { run: { type: 'string' }, 'no-lock': { type: 'boolean', default: false } } });

const checks: { name: string; ok: boolean; detail?: string }[] = [];
const check = (name: string, ok: boolean, detail?: string) => { checks.push({ name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`); return ok; };
const withTimeout = <T>(p: Promise<T>, ms: number) => Promise.race([p, new Promise<void>((r) => setTimeout(r, ms))]);

async function frames(page: Page, n: number): Promise<void> {
  const f0 = await page.evaluate(() => window.__app!.frameCounter);
  await page.waitForFunction((t) => window.__app!.frameCounter >= t, f0 + n, { timeout: 240_000, polling: 50 });
}

/** Colour target (non-finite count, mean) and the active debug AOV (non-finite, written pixels). */
async function readState(page: Page): Promise<{ nonFinite: number; mean: number; aovNonFinite: number; aovWritten: number }> {
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
    let aovNonFinite = 0, aovWritten = 0;
    if (ab && view) {
      await ab.mapAsync(GPUMapMode.READ);
      const a = new Float32Array(ab.getMappedRange().slice(0)); ab.unmap(); ab.destroy();
      const au = new Uint32Array(a.buffer);
      for (let i = 0; i < dbg.width * dbg.height; i++) {
        if (au[4 * i] !== 0 && au[4 * i] !== 0xffffffff) aovWritten++;
        if (view.kind !== 'code') for (let c = 0; c < (view.kind === 'vec3' ? 3 : 1); c++) if (!Number.isFinite(a[4 * i + c])) aovNonFinite++;
      }
    }
    return { nonFinite, mean: n ? sum / n : 0, aovNonFinite, aovWritten };
  });
}

const errState = (page: Page) => page.evaluate(() => {
  const r = window.__integration!.renderer()!;
  return { err: r.lastError, rs: r.restirError, gpu: window.__webgpuErrors ?? [] };
});
const hud = (page: Page) => page.evaluate(() => window.__integration!.renderer()!.hudLines?.().join('\n') ?? '');

async function run(page: Page, OUT: string): Promise<void> {
  // ---- defaults
  const d = await page.evaluate(() => {
    const r = window.__integration!.renderer()!;
    const s = r.restirSettings();
    return { lightMode: r.options.lightMode, pairing: s.pairing, risNee: s.risNee, dualMv: s.dualMv, dupmap: s.dupmap, cCap: s.cCap };
  });
  check('light mode B is the default (PT and ReSTIR; MD9 / Q4)', d.lightMode === 'B', JSON.stringify(d));
  check('interactive preset: σ 16 pairing, RIS-NEE, dual MVs, duplication map, c_cap 5 (TD-I1)', d.pairing === 'gauss' && d.risNee === true && d.dualMv === true && d.dupmap === true && d.cCap === 5, JSON.stringify(d));
  await page.evaluate(async () => {
    const r = window.__integration!.renderer()!;
    r.options.renderMode = 'restir';
    await r.setOptions({ restirMode: 'interactive', temporal: true });
    await r.prepareRestir();
    window.__app!.resetTemporalHistory();
  });
  await frames(page, 30);
  const s0 = await readState(page);
  const h0 = await hud(page);
  const e0 = await errState(page);
  check('ReSTIR-interactive (M6) renders finite, error-free', s0.nonFinite === 0 && s0.mean > 0 && !e0.rs && !e0.err && e0.gpu.length === 0, `mean ${s0.mean.toExponential(3)} ${e0.rs ?? ''}`);
  check('HUD shows the M6 line (Mode B, gauss σ 16, RIS-NEE M 32, dual MV, dup map)', /M6: Mode B\s+pairing gauss σ 16\s+RIS-NEE M 32\s+dual MV on\s+dup map on/.test(h0), h0.split('\n').find((l) => /M6:/.test(l)) ?? 'no M6 line');
  await page.locator('#view').screenshot({ path: path.join(OUT, 'cornell-m6-interactive.png') });

  // ---- feature toggles
  for (const [name, off, on] of [
    ['σ 16 pairing', { pairing: 'disk' }, { pairing: 'gauss' }], ['RIS-NEE', { risNee: false }, { risNee: true }],
    ['dual MVs', { dualMv: false }, { dualMv: true }], ['duplication map', { dupmap: false }, { dupmap: true }],
  ] as const) {
    const res: string[] = [];
    for (const o of [off, on]) {
      await page.evaluate(async (x) => { const r = window.__integration!.renderer()!; Object.assign(r.options.restirFeatures, x); await r.setOptions({ restirFeatures: r.options.restirFeatures }); window.__app!.resetHistory(); }, o as Record<string, unknown>);
      await frames(page, 8);
      const s = await readState(page), e = await errState(page);
      res.push(`${JSON.stringify(o)}: ${s.nonFinite === 0 && s.mean > 0 && !e.rs && e.gpu.length === 0 ? 'ok' : `BAD ${s.nonFinite} ${e.rs ?? ''} ${e.gpu[0] ?? ''}`}`);
    }
    check(`toggle ${name} off / on: finite, error-free`, res.every((x) => x.endsWith('ok')), res.join('; '));
  }

  // ---- light modes in ReSTIR
  const lm: string[] = [];
  for (const m of ['A', 'A′', 'B'] as const) {
    await page.evaluate(async (x) => { await window.__integration!.renderer()!.setOptions({ lightMode: x }); window.__app!.resetHistory(); }, m);
    await frames(page, 8);
    const s = await readState(page), e = await errState(page), h = await hud(page);
    lm.push(`${m}: ${s.nonFinite === 0 && s.mean > 0 && !e.rs && e.gpu.length === 0 && h.includes(`M6: Mode ${m}`) ? 'ok' : `BAD ${s.nonFinite} ${e.rs ?? ''} hud ${/M6: Mode (\S+)/.exec(h)?.[1]}`}`);
  }
  check('ReSTIR in light modes A, A′, B: finite, error-free, HUD names the mode', lm.every((x) => x.endsWith('ok')), lm.join('; '));

  // ---- views 471–479
  const bad: string[] = [];
  const written: Record<number, number> = {};
  for (const id of M6_VIEWS) {
    await page.evaluate((v) => window.__app!.selectDebugView(v), id);
    await frames(page, 3);
    const s = await readState(page);
    written[id] = s.aovWritten;
    if (s.aovNonFinite > 0) bad.push(`${id}: ${s.aovNonFinite} non-finite`);
    if ([471, 477, 478, 479].includes(id)) await page.locator('#view').screenshot({ path: path.join(OUT, `cornell-view-${id}.png`) });
  }
  await page.evaluate(() => window.__app!.selectDebugView(0));
  const reg = await page.evaluate((ids) => window.__app!.debug.registry.list().filter((v) => ids.includes(v.id)).length, M6_VIEWS);
  check('views 471–479 registered and finite', reg === M6_VIEWS.length && bad.length === 0, `${reg} registered${bad.length ? `; ${bad.join('; ')}` : ''}`);
  check('views 471 (pair offset), 477 (reciprocity), 478 (dup count) written', written[471] > 0 && written[477] > 0 && written[478] > 0, JSON.stringify(written));

  // ---- motion (dual MVs)
  await page.evaluate(() => {
    const app = window.__app!;
    const b = app.scene!.bounds, c = [0, 1, 2].map((i) => 0.5 * (b.min[i] + b.max[i])) as [number, number, number];
    const ext = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
    const eye0 = [...app.camera.position] as [number, number, number];
    let k = 0;
    window.__m6Pan = true;
    app.beforeFrame.add((f) => { if (!f.advance || !window.__m6Pan) return; const s = k++; app.camera.lookAt([eye0[0] + 0.05 * ext * Math.sin(0.06 * s), eye0[1], eye0[2]], c); });
  });
  await frames(page, 30);
  await page.evaluate(() => { window.__m6Pan = false; });
  const sm = await readState(page);
  check('camera pan with dual MVs: finite', sm.nonFinite === 0 && sm.mean > 0);
}

async function main(): Promise<number> {
  const runId = args.run ?? `m6-app-smoke-${stamp()}`;
  const OUT = path.join(ROOT, 'validation/out', runId);
  mkdirSync(OUT, { recursive: true });
  const vite = await createServer({
    root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), logLevel: 'warn',
    server: { port: 0, host: '127.0.0.1', hmr: false, watch: null, fs: { allow: [ROOT, realpathSync(path.join(ROOT, 'node_modules'))] } },
  });
  await vite.listen();
  const port = (vite.httpServer!.address() as { port: number }).port;
  const errors: string[] = [];
  try {
    const browser = await chromium.launch({ channel: 'chrome', headless: true });
    const release = args['no-lock'] ? () => {} : await acquireGpuLock('m6-app-smoke');
    try {
      const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
      const page = await ctx.newPage();
      await page.addInitScript({
        content: `window.__name = (f) => f; window.__tsPasses = [];
          (() => { const P = globalThis.GPUCommandEncoder && GPUCommandEncoder.prototype; if (!P) return; const orig = P.beginComputePass;
            P.beginComputePass = function (d) { if (d && d.timestampWrites) window.__tsPasses.push(String(d.label)); return orig.call(this, d); }; })();`,
      });
      page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
      page.on('pageerror', (e) => errors.push(`[pageerror] ${e.message}`));
      try {
        const q = new URLSearchParams({ seed: '1', res: '540p', scene: `/${CORNELL}` });
        await page.goto(`http://127.0.0.1:${port}/index.html?${q}`);
        await page.waitForFunction(() => {
          const r = window.__integration?.renderer(); const a = window.__app;
          return !!r && !!a?.scene && r.scene?.scene === a.scene && !r.loading && r.ready;
        }, undefined, { timeout: 240_000, polling: 100 });
        await page.evaluate(() => {
          const store = window.__editor!.editor.store!;
          store.add({ type: 'point', power: 30, matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1, 0.35, -0.2, 1]) });
          store.add({ type: 'rect', power: 20, sizeX: 0.15, sizeY: 0.1, visibleToCamera: false, matrix: new Float32Array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0.5, -0.28, 1]) });
        });
        await frames(page, 10);
        await run(page, OUT);
        const t = await page.evaluate(() => ({ ...window.__app!.runTotals }));
        check('NaN/Inf = 0 (debug counters)', t.nan === 0 && t.inf === 0, JSON.stringify(t));
        const ts = await page.evaluate(() => window.__tsPasses ?? []);
        const badTs = [...new Set(ts.filter((l) => /^rs_|restir/i.test(l)))];
        check('no timestampWrites on a ReSTIR pass (Q3)', badTs.length === 0, badTs.join(', ') || 'none');
      } catch (e) {
        check('page run completed', false, e instanceof Error ? e.message.split('\n')[0] : String(e));
        await withTimeout(page.screenshot({ path: path.join(OUT, 'failure.png') }), 10_000).catch(() => undefined);
      } finally {
        await withTimeout(page.close(), 20_000);
        await withTimeout(ctx.close(), 10_000);
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
  console.log(`${failed ? 'FAIL' : 'PASS'} m6 app smoke: ${checks.length - failed}/${checks.length} (screenshots: ${path.relative(ROOT, OUT)})`);
  return failed ? 1 : 0;
}

main().then((c) => process.exit(c), (e: unknown) => { console.error(e); process.exit(1); });

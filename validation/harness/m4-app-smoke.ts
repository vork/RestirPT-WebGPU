// M4 app smoke (restir-api.md §6.4 step 4, WP-D; PLAN §6 M4 rows): the real app in headless Chrome, render mode
// ReSTIR on (1) the Cornell box with a point and a rect light added through the editor and (2) the same box lit by
// an HDRI (env NEE). Per scene: every app ReSTIR mode renders (ReSTIR-unbiased, ReSTIR-2022-criteria, Offline, initial
// only) with a finite image and the HUD shows f_r; every M4 debug view (400–499; reservoir views at the three stage
// taps) renders with a finite AOV (scalar / vec3 views); the pixel inspector's probe dump decodes (reservoir after
// initial and after spatial, slots) and draws the 3D path overlay. Global: no uncaptured WebGPU / console errors, no
// renderer or ReSTIR compile errors, NaN/Inf = 0 (finalize counters, arena non-finite counters, debug counters).
// Screenshots (beauty per mode, a representative view per group, the inspector) go to validation/out/<run>/.
//   npx tsx validation/harness/m4-app-smoke.ts [--run <id>] [--no-lock] [--views all|some]
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

declare global { interface Window { __app?: App; __integration?: Integration; __editor?: EditorHandle; __webgpuErrors?: string[] } }

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const HDRI = 'validation/assets/downloaded/hdri/studio_small_09_1k.hdr';
const SCREENSHOT_VIEWS = new Set([400, 401, 403, 404, 405, 406, 409, 410, 420, 426, 432, 438, 439, 440, 446, 460, 461, 462, 463, 469, 470]);
const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const { values: args } = parseArgs({ options: { run: { type: 'string' }, 'no-lock': { type: 'boolean', default: false }, views: { type: 'string', default: 'all' } } });

const checks: { name: string; ok: boolean; detail?: string }[] = [];
const check = (name: string, ok: boolean, detail?: string) => { checks.push({ name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`); return ok; };

/** Page / browser teardown can hang behind in-flight GPU work: never hold the GPU lock on it. */
const withTimeout = <T>(p: Promise<T>, ms: number) => Promise.race([p, new Promise<void>((r) => setTimeout(r, ms))]);

async function frames(page: Page, n: number): Promise<void> {
  const f0 = await page.evaluate(() => window.__app!.frameCounter);
  await page.waitForFunction((t) => window.__app!.frameCounter >= t, f0 + n, { timeout: 180_000, polling: 50 });
}
const hud = (page: Page) => page.evaluate(() => window.__integration!.renderer()!.hudLines());

/** In-page GPU readbacks: colour target (finite, mean), debug AOV (non-finite count for scalar/vec3 views), ReSTIR
 *  finalize counters (nonFinite, bvhOverflow, bvhItercap, negative). */
async function readState(page: Page): Promise<{ colour: { nonFinite: number; mean: number }; aovNonFinite: number; aovWritten: number; finalize: number[] }> {
  return page.evaluate(async () => {
    const app = window.__app!;
    const r = window.__integration!.renderer()!;
    const device = app.device;
    const t = app.targets;
    await device.queue.onSubmittedWorkDone();
    const read = async (src: GPUBuffer, bytes: number, off = 0) => {
      const b = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = device.createCommandEncoder(); e.copyBufferToBuffer(src, off, b, 0, bytes); device.queue.submit([e.finish()]);
      await b.mapAsync(GPUMapMode.READ); const out = b.getMappedRange().slice(0); b.unmap(); b.destroy(); return out;
    };
    // colour target (rgba32float or rgba16float): read as float32 when possible
    const bpp = t.colorFormat === 'rgba32float' ? 16 : 8;
    const bpr = Math.ceil((t.width * bpp) / 256) * 256;
    const cb = device.createBuffer({ size: bpr * t.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = device.createCommandEncoder();
    e.copyTextureToBuffer({ texture: t.color }, { buffer: cb, bytesPerRow: bpr }, [t.width, t.height]);
    device.queue.submit([e.finish()]);
    await cb.mapAsync(GPUMapMode.READ);
    const raw = cb.getMappedRange().slice(0); cb.unmap(); cb.destroy();
    let nonFinite = 0, sum = 0, n = 0;
    if (bpp === 16) {
      const f = new Float32Array(raw);
      for (let y = 0; y < t.height; y++) for (let x = 0; x < t.width; x++) for (let c = 0; c < 3; c++) {
        const v = f[(y * bpr) / 4 + x * 4 + c]; if (!Number.isFinite(v)) nonFinite++; else { sum += v; n++; }
      }
    }
    const dbg = app.debug;
    const view = dbg.activeView();
    let aovNonFinite = 0, aovWritten = 0;
    if (view) {
      const L = 64 + 256 * 32;
      const a = new Float32Array(await read(dbg.buffer, dbg.width * dbg.height * 16, L));
      const u = new Uint32Array(a.buffer);
      const comps = view.kind === 'vec3' ? 3 : 1;
      for (let i = 0; i < dbg.width * dbg.height; i++) {
        if (u[4 * i] !== 0) aovWritten++;
        if (view.kind !== 'code') for (let c = 0; c < comps; c++) if (!Number.isFinite(a[4 * i + c])) aovNonFinite++;
      }
    }
    const rs = r.restir;
    const finalize = rs ? Array.from(new Uint32Array(await read(rs.counters, 16))) : [];
    return { colour: { nonFinite, mean: n ? sum / n : 0 }, aovNonFinite, aovWritten, finalize };
  });
}

async function runScene(page: Page, OUT: string, name: string): Promise<void> {
  // ReSTIR on
  await page.evaluate(async () => {
    const r = window.__integration!.renderer()!;
    r.options.renderMode = 'restir';
    await r.prepareRestir();
    window.__app!.resetHistory();
  });
  await page.waitForFunction(() => window.__integration!.renderer()!.hudLines().some((l) => /^ReSTIR f_r/.test(l)), undefined, { timeout: 120_000, polling: 100 });
  check(`${name}: ReSTIR compiled`, await page.evaluate(() => !window.__integration!.renderer()!.restirError), await page.evaluate(() => window.__integration!.renderer()!.restirError ?? ''));

  for (const mode of ['unbiased', 'criteria2022', 'offline', 'initial'] as const) {
    await page.evaluate(async (m) => { const r = window.__integration!.renderer()!; await r.setOptions({ restirMode: m }); window.__app!.resetHistory(); }, mode);
    await frames(page, mode === 'offline' ? 6 : 16);
    const lines = await hud(page);
    const head = lines.find((l) => l.startsWith(`ReSTIR ${mode}:`)) ?? '';
    const fr = lines.find((l) => l.startsWith('ReSTIR f_r')) ?? '';
    const st = await readState(page);
    check(`${name}/${mode}: renders, finite image, HUD f_r`, !!head && /f_r [\d.]+/.test(fr) && st.colour.nonFinite === 0 && st.colour.mean > 0 && st.finalize[0] === 0,
      `${head} | ${fr} | mean ${st.colour.mean.toExponential(3)} nonFinite ${st.colour.nonFinite} finalize ${st.finalize.join(',')}`);
    await page.screenshot({ path: path.join(OUT, `${name}-${mode}.png`) });
  }

  // every M4 view in the unbiased mode (reservoir views at the three stage taps); shift/MIS views also in Offline
  await page.evaluate(async () => { const r = window.__integration!.renderer()!; await r.setOptions({ restirMode: 'unbiased' }); });
  const views = await page.evaluate(() => window.__app!.debug.registry.list().filter((v) => v.id >= 400 && v.id < 500).map((v) => ({ id: v.id, key: v.key, kind: v.kind, tapped: !!v.tapped })));
  check(`${name}: M4 views registered`, views.length >= 45, `${views.length} views`);
  const bad: string[] = [];
  let rendered = 0;
  const some = args.views !== 'all';
  for (const v of views) {
    if (some && !SCREENSHOT_VIEWS.has(v.id)) continue;
    for (const tap of v.tapped ? [1, 3, 0] : [0]) {
      await page.evaluate(([id, tp]) => { const a = window.__app!; a.selectDebugView(id); a.debugSettings.tap = tp; }, [v.id, tap] as const);
      await frames(page, 3);
      const st = await readState(page);
      rendered++;
      if (st.aovNonFinite > 0) bad.push(`${v.key}@tap${tap}: ${st.aovNonFinite} non-finite`);
      if (tap === 0 && SCREENSHOT_VIEWS.has(v.id)) await page.screenshot({ path: path.join(OUT, `${name}-view-${v.id}-${v.key.replace(/[^\w.]+/g, '_')}.png`) });
    }
  }
  check(`${name}: every M4 view renders with a finite AOV`, bad.length === 0, `${rendered} view×tap renders${bad.length ? `; ${bad.slice(0, 4).join('; ')}` : ''}`);
  // written coverage of the key views (the view is produced at all): c, d, shift code, m_c
  for (const [id, what] of [[400, 'c'], [404, 'd'], [420, 'shift code[0]'], [460, 'm_c']] as const) {
    await page.evaluate((x) => { const a = window.__app!; a.selectDebugView(x); a.debugSettings.tap = 0; }, id);
    await frames(page, 3);
    const st = await readState(page);
    check(`${name}: view ${id} (${what}) is written`, st.aovWritten > 0, `${st.aovWritten} px written`);
  }
  await page.evaluate(() => window.__app!.selectDebugView(0));

  // pixel inspector at the image centre
  await page.evaluate(() => {
    const a = window.__app!;
    const ui = window.__integration!.restirUi();
    a.debugSettings.probePixel = [a.targets.width >> 1, Math.floor(a.targets.height * 0.6)];
    a.debugSettings.probeEnabled = true;
    ui.inspector!.setVisible(true);
  });
  await frames(page, 20);
  await page.waitForFunction(() => !!window.__integration!.restirUi().inspector?.latest?.reservoirs.length, undefined, { timeout: 30_000 }).catch(() => undefined);
  const insp = await page.evaluate(() => {
    const d = window.__integration!.restirUi().inspector!.latest;
    const text = window.__integration!.restirUi().inspector!.root.textContent ?? '';
    return d ? {
      taps: d.reservoirs.map((r) => r.tap), cand: d.candidates.length, paths: [...d.paths.keys()], slots: d.slots.length, mis: !!d.mis.canonical,
      lines: window.__app!.overlay.lineCount, text: text.slice(0, 400),
    } : undefined;
  });
  check(`${name}: inspector dump decodes (reservoir after initial + after spatial, slots) and draws paths`,
    !!insp && insp.taps.includes(1) && insp.taps.includes(3) && insp.slots > 0 && insp.lines > 3,
    insp ? `taps ${insp.taps} candidates ${insp.cand} paths ${insp.paths} slots ${insp.slots} MIS ${insp.mis} overlay lines ${insp.lines}` : 'no decode');
  await page.screenshot({ path: path.join(OUT, `${name}-inspector.png`) });
  await page.evaluate(() => { const a = window.__app!; window.__integration!.restirUi().inspector!.setVisible(false); a.debugSettings.probeEnabled = false; });
  const hudLines = await hud(page);
  const err = hudLines.find((l) => l.startsWith('ReSTIR errors')) ?? '';
  check(`${name}: arena error counters 0 (non-finite, pending, mismatch, BVH, overflow)`, /^ReSTIR errors 0 /.test(err), err);
}

async function main(): Promise<number> {
  const runId = args.run ?? `m4-app-smoke-${stamp()}`;
  const OUT = path.join(ROOT, 'validation/out', runId);
  mkdirSync(OUT, { recursive: true });
  const release = args['no-lock'] ? () => {} : await acquireGpuLock('m4-app-smoke');
  // node_modules may be a symlink into another checkout (worktrees): allow its real path; no HMR / watch (edits by
  // other agents must not reload the page mid-run).
  const vite = await createServer({
    root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), logLevel: 'warn',
    server: { port: 0, host: '127.0.0.1', hmr: false, watch: null, fs: { allow: [ROOT, realpathSync(path.join(ROOT, 'node_modules'))] } },
  });
  await vite.listen();
  const port = (vite.httpServer!.address() as { port: number }).port;
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const errors: string[] = [];
  try {
    for (const scene of ['cornell', 'cornell-hdri'] as const) {
      const page = await (await browser.newContext({ viewport: { width: 1280, height: 720 } })).newPage();
      // tsx (esbuild keepNames) wraps named closures of page.evaluate bodies in __name(): define it in the page.
      await page.addInitScript({ content: 'window.__name = (f) => f;' });
      page.on('console', (m) => { if (m.type() === 'error') errors.push(`${scene}: ${m.text()}`); if (/uncaptured|error/i.test(m.text())) console.log(`[page:${m.type()}] ${m.text()}`); });
      page.on('pageerror', (e) => errors.push(`${scene}: [pageerror] ${e.message}`));
      const q = new URLSearchParams({ seed: '1', res: '540p', scene: '/validation/assets/cornell/cornell.glb', ...(scene === 'cornell-hdri' ? { env: `/${HDRI}` } : {}) });
      await page.goto(`http://127.0.0.1:${port}/index.html?${q}`);
      await page.waitForFunction((env) => {
        const r = window.__integration?.renderer(); const a = window.__app;
        return !!r && !!a?.scene && r.scene?.scene === a.scene && !r.loading && r.ready && (!env || r.env.present);
      }, scene === 'cornell-hdri', { timeout: 180_000, polling: 100 });
      // lights through the editor (the glTF carries none): a point light and a downward rect light (NEE-only, Mode A)
      await page.evaluate((hdri) => {
        const store = window.__editor!.editor.store!;
        store.add({ type: 'point', power: hdri ? 10 : 30, matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1, 0.35, -0.2, 1]) });
        store.add({ type: 'rect', power: 20, sizeX: 0.15, sizeY: 0.1, visibleToCamera: false, matrix: new Float32Array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0.5, -0.28, 1]) });
      }, scene === 'cornell-hdri');
      await frames(page, 10);
      await runScene(page, OUT, scene);
      const t = await page.evaluate(() => ({ ...window.__app!.runTotals }));
      check(`${scene}: NaN/Inf = 0 (debug counters)`, t.nan === 0 && t.inf === 0, JSON.stringify(t));
      const st = await page.evaluate(() => ({ err: window.__integration!.renderer()!.lastError, rs: window.__integration!.renderer()!.restirError, gpu: window.__webgpuErrors ?? [] }));
      check(`${scene}: renderer / ReSTIR / WebGPU error-free`, !st.err && !st.rs && st.gpu.length === 0, [st.err, st.rs, ...st.gpu.slice(0, 2)].filter(Boolean).join(' | '));
      await withTimeout(page.close(), 20_000);
    }
    check('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  } finally {
    await withTimeout(browser.close(), 30_000);
    await withTimeout(vite.close(), 10_000);
    release();
  }
  writeFileSync(path.join(OUT, 'report.json'), JSON.stringify({ runId, checks }, null, 1));
  const failed = checks.filter((c) => !c.ok).length;
  console.log(`${failed ? 'FAIL' : 'PASS'} m4 app smoke: ${checks.length - failed}/${checks.length} (screenshots: ${path.relative(ROOT, OUT)})`);
  return failed ? 1 : 0;
}

main().then((c) => process.exit(c), (e: unknown) => { console.error(e); process.exit(1); });

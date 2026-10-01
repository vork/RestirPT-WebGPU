// M5.5 app smoke (docs/decisions/denoiser.md §8–§11): the real app in headless Chrome at 540p, ReSTIR-interactive, on
// (1) the Cornell box with a point and a rect light added through the editor and (2) Sponza under an HDRI. Per scene:
//   defaults      the denoiser is on in ReSTIR-interactive and runs every frame; the image is finite
//   screenshots   denoised vs raw (denoiser off, accumulation off: the 1-frame ReSTIR output) with a static and with a
//                 moving camera (canvas only, HUD hidden)
//   lights        a moving point light (Cornell): the change bits open the λ gate on those frames; finite
//   views         every denoiser view (520–535) renders a finite AOV; 520 / 521 / 522 / 526 are written
//   held frames   paused: the denoised image does not change (DN9)
//   modes         ReSTIR-unbiased forces it off (DN4, T16 of the app side), ReSTIR-2022-criteria defaults off,
//                 back to ReSTIR-interactive restores on, the PT can be denoised when toggled on
//   timing        the HUD shows the GPU time of the separate timing submits; no timestampWrites on any ReSTIR or
//                 denoiser pass of a frame (Q3)
// Global: no console / WebGPU errors, NaN/Inf = 0. The GPU lock is taken per page load.
//   npx tsx validation/harness/m55-app-smoke.ts [--run <id>] [--no-lock] [--scenes cornell,sponza]
import { existsSync, mkdirSync, realpathSync, writeFileSync } from 'node:fs';
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
  interface Window { __app?: App; __integration?: Integration; __editor?: EditorHandle; __webgpuErrors?: string[]; __tsPasses?: string[]; __dnAnim?: { cam: boolean; light: boolean; k: number } }
}

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const SPONZA = 'validation/assets/downloaded/sponza/Sponza.gltf';
const CORNELL = 'validation/assets/cornell/cornell.glb';
const HDRI = 'validation/assets/downloaded/hdri/kloofendal_48d_partly_cloudy_puresky_1k.hdr';
const DN_VIEWS = [520, 521, 522, 523, 524, 525, 526, 527, 530, 531, 532, 533, 534];
const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const { values: args } = parseArgs({ options: { run: { type: 'string' }, 'no-lock': { type: 'boolean', default: false }, scenes: { type: 'string', default: 'cornell,sponza' } } });

const checks: { name: string; ok: boolean; detail?: string }[] = [];
const check = (name: string, ok: boolean, detail?: string) => { checks.push({ name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`); return ok; };
const withTimeout = <T>(p: Promise<T>, ms: number) => Promise.race([p, new Promise<void>((r) => setTimeout(r, ms))]);

async function frames(page: Page, n: number): Promise<void> {
  const f0 = await page.evaluate(() => window.__app!.frameCounter);
  await page.waitForFunction((t) => window.__app!.frameCounter >= t, f0 + n, { timeout: 180_000, polling: 50 });
}

/** Colour target (finite count, mean, a hash of the bytes) and the debug AOV (non-finite, written pixels). */
async function readState(page: Page): Promise<{ nonFinite: number; mean: number; hash: number; aovNonFinite: number; aovWritten: number; denoised: boolean }> {
  return page.evaluate(async () => {
    const app = window.__app!, r = window.__integration!.renderer()!, device = app.device, t = app.targets;
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
    const raw = cb.getMappedRange().slice(0); cb.unmap(); cb.destroy();
    const f = new Float32Array(raw), u = new Uint32Array(raw);
    let nonFinite = 0, sum = 0, n = 0, hash = 2166136261;
    for (let y = 0; y < t.height; y++) for (let x = 0; x < t.width; x++) for (let c = 0; c < 3; c++) {
      const i = (y * bpr) / 4 + x * 4 + c, v = f[i];
      if (!Number.isFinite(v)) nonFinite++; else { sum += v; n++; }
      hash = Math.imul(hash ^ u[i], 16777619) >>> 0;
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
    return { nonFinite, mean: n ? sum / n : 0, hash, aovNonFinite, aovWritten, denoised: r.denoisedLastFrame };
  });
}

async function shot(page: Page, file: string): Promise<void> {
  await page.evaluate(() => window.__app!.hud.setVisible(false));
  await frames(page, 2);
  await page.locator('#view').screenshot({ path: file });
  await page.evaluate(() => window.__app!.hud.setVisible(true));
}

async function scenePass(page: Page, OUT: string, scene: string): Promise<void> {
  // ---- ReSTIR-interactive: the denoiser is on by default
  await page.evaluate(async () => {
    const r = window.__integration!.renderer()!;
    r.options.renderMode = 'restir';
    await r.setOptions({ restirMode: 'interactive', temporal: true });
    await r.prepareRestir();
    window.__app!.resetTemporalHistory();
  });
  await page.waitForFunction(() => window.__integration!.renderer()!.denoisedLastFrame, undefined, { timeout: 180_000, polling: 100 });
  const def = await page.evaluate(() => { const r = window.__integration!.renderer()!; return { on: r.options.denoise, allowed: r.denoiseAllowed, err: r.denoiserError }; });
  check(`${scene}: denoiser on by default in ReSTIR-interactive and running`, def.on && def.allowed && !def.err, JSON.stringify(def));
  await frames(page, 40);
  const s0 = await readState(page);
  check(`${scene}: denoised image finite`, s0.denoised && s0.nonFinite === 0 && s0.mean > 0, `mean ${s0.mean.toExponential(3)}`);
  await shot(page, path.join(OUT, `${scene}-static-denoised.png`));
  // raw = the 1-frame ReSTIR output (denoiser off, accumulation off)
  await page.evaluate(() => { const r = window.__integration!.renderer()!; r.setDenoise(false); r.options.accumulate = false; });
  await frames(page, 6);
  const s1 = await readState(page);
  check(`${scene}: raw 1-frame output (denoiser off) finite`, !s1.denoised && s1.nonFinite === 0, `mean ${s1.mean.toExponential(3)} vs denoised ${s0.mean.toExponential(3)}`);
  await shot(page, path.join(OUT, `${scene}-static-raw.png`));
  // moving camera
  await page.evaluate(() => {
    const app = window.__app!;
    const b = app.scene!.bounds, c = [0, 1, 2].map((i) => 0.5 * (b.min[i] + b.max[i])) as [number, number, number];
    const ext = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
    const eye0 = [...app.camera.position] as [number, number, number];
    window.__dnAnim = { cam: false, light: false, k: 0 };
    app.beforeFrame.add((f) => {
      const a = window.__dnAnim!;
      if (!f.advance || !(a.cam || a.light)) return;
      const s = a.k++;
      if (a.cam) app.camera.lookAt([eye0[0] + 0.04 * ext * Math.sin(0.05 * s), eye0[1], eye0[2] + 0.03 * ext * Math.cos(0.05 * s)], c);
      if (a.light) {
        const store = window.__editor!.editor.store!;
        const pt = store.list().find((l) => l.type === 'point');
        if (pt) { const m = new Float32Array(pt.matrix); m[12] = 0.1 + 0.12 * Math.sin(0.09 * s); store.update(pt.id, { matrix: m }, 'smoke'); }
      }
    });
  });
  await page.evaluate(() => { window.__dnAnim!.cam = true; });
  await frames(page, 24);
  await shot(page, path.join(OUT, `${scene}-moving-raw.png`));
  await page.evaluate(() => { const r = window.__integration!.renderer()!; r.setDenoise(true); r.options.accumulate = true; });
  await frames(page, 24);
  const sm = await readState(page);
  check(`${scene}: moving camera, denoised, finite`, sm.denoised && sm.nonFinite === 0 && sm.mean > 0);
  await shot(page, path.join(OUT, `${scene}-moving-denoised.png`));
  await page.evaluate(() => { window.__dnAnim!.cam = false; });

  // ---- lights: the change bits open the gradient gate
  const hasPoint = await page.evaluate(() => !!window.__editor?.editor.store?.list().some((l) => l.type === 'point'));
  if (hasPoint) {
    await page.evaluate(() => { window.__dnAnim!.light = true; });
    const gates: number[] = [];
    for (let i = 0; i < 12; i++) { await frames(page, 1); gates.push(await page.evaluate(() => window.__integration!.renderer()!.denoiser!.flags)); }
    await page.evaluate(() => { window.__dnAnim!.light = false; });
    await frames(page, 4);
    const sl = await readState(page);
    const open = gates.filter((f) => (f & 32) && (f & 2)).length;
    check(`${scene}: moving light ⇒ gradient passes run and the λ gate opens; finite`, open >= 6 && sl.nonFinite === 0, `flags ${gates.join(',')}`);
    const closed = await page.evaluate(() => window.__integration!.renderer()!.denoiser!.flags);
    check(`${scene}: static lights ⇒ gate closed`, (closed & 2) === 0, `flags ${closed}`);
  }

  // ---- views
  const bad: string[] = [];
  const written: Record<number, number> = {};
  for (const id of DN_VIEWS) {
    await page.evaluate((v) => window.__app!.selectDebugView(v), id);
    await frames(page, 3);
    const s = await readState(page);
    written[id] = s.aovWritten;
    if (s.aovNonFinite > 0) bad.push(`${id}: ${s.aovNonFinite} non-finite`);
    if ([520, 521, 522, 523, 526, 530, 534].includes(id)) await shot(page, path.join(OUT, `${scene}-view-${id}.png`));
  }
  await page.evaluate(() => window.__app!.selectDebugView(0));
  const reg = await page.evaluate(() => window.__app!.debug.registry.list().filter((v) => v.id >= 520 && v.id < 540).length);
  check(`${scene}: every denoiser view renders a finite AOV`, reg === 14 && bad.length === 0, `${reg} registered${bad.length ? `; ${bad.join('; ')}` : ''}`);
  check(`${scene}: views 520 (variance), 521 (history), 522 (α), 526 (reprojection) are written`, written[520] > 0 && written[521] > 0 && written[522] > 0 && written[526] > 0, JSON.stringify(written));

  // ---- held frames (DN9)
  await page.evaluate(() => window.__app!.setPaused(true));
  await frames(page, 3);
  const h1 = await readState(page);
  await frames(page, 4);
  const h2 = await readState(page);
  await page.evaluate(() => window.__app!.setPaused(false));
  check(`${scene}: paused ⇒ the denoised image holds bit for bit`, h1.denoised && h1.hash === h2.hash && h1.nonFinite === 0, `hash ${h1.hash} / ${h2.hash}`);

  // ---- HUD timing (separate timing submits)
  await frames(page, 70);
  const tm = await page.evaluate(() => { const r = window.__integration!.renderer()!; return { avg: r.denoiser!.timingAverage(), line: r.denoiserHudLine(), n: r.denoiser!.timings.length }; });
  check(`${scene}: HUD denoiser timing from separate submits`, !!tm.avg && tm.avg.totalMs > 0 && /GPU \d/.test(tm.line), `${tm.line} (${tm.n} timing submits)`);

  // ---- modes
  const modes = await page.evaluate(async () => {
    const r = window.__integration!.renderer()!, a = window.__app!;
    const wait = (n: number) => new Promise<void>((res) => { const f0 = a.frameCounter; const tick = () => (a.frameCounter >= f0 + n ? res() : setTimeout(tick, 20)); tick(); });
    const out: Record<string, unknown> = {};
    await r.setOptions({ restirMode: 'unbiased' }); a.resetHistory(); await wait(6);
    r.setDenoise(true);                                   // ignored: not allowed
    await wait(4);
    out.unbiased = { wanted: r.denoiseWanted(), ran: r.denoisedLastFrame, allowed: r.denoiseAllowed, hud: r.denoiserHudLine() };
    await r.setOptions({ restirMode: 'criteria2022' }); await wait(6);
    out.criteria = { on: r.options.denoise, ran: r.denoisedLastFrame };
    await r.setOptions({ restirMode: 'interactive' }); await wait(6);
    out.interactive = { on: r.options.denoise, ran: r.denoisedLastFrame };
    r.options.renderMode = 'pt'; a.resetHistory(); await wait(6);
    out.ptDefault = { on: r.options.denoise, ran: r.denoisedLastFrame };
    r.setDenoise(true); a.resetHistory(); await wait(10);
    out.ptOn = { ran: r.denoisedLastFrame, kind: r.denoiser?.kind };
    r.setDenoise(false);
    r.options.renderMode = 'restir'; a.resetHistory(); await wait(6);
    out.back = { on: r.options.denoise, ran: r.denoisedLastFrame };
    return out;
  }) as Record<string, Record<string, unknown>>;
  check(`${scene}: ReSTIR-unbiased forces the denoiser off`, modes.unbiased.allowed === false && modes.unbiased.wanted === false && modes.unbiased.ran === false, JSON.stringify(modes.unbiased));
  check(`${scene}: ReSTIR-2022-criteria defaults off; interactive restores on`, modes.criteria.on === false && modes.criteria.ran === false && modes.interactive.on === true && modes.interactive.ran === true, JSON.stringify([modes.criteria, modes.interactive]));
  check(`${scene}: PT defaults off; toggled on it denoises the 1-spp PT`, modes.ptDefault.on === false && modes.ptDefault.ran === false && modes.ptOn.ran === true && modes.ptOn.kind === 'pt' && modes.back.on === true,
    JSON.stringify([modes.ptDefault, modes.ptOn, modes.back]));
  const sf = await readState(page);
  check(`${scene}: final image finite`, sf.nonFinite === 0 && sf.mean > 0);
}

async function pageLoad(browser: Browser, port: number, OUT: string, scene: string, errors: string[]): Promise<void> {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.addInitScript({
    content: `window.__name = (f) => f; window.__tsPasses = [];
      (() => { const P = globalThis.GPUCommandEncoder && GPUCommandEncoder.prototype; if (!P) return; const orig = P.beginComputePass;
        P.beginComputePass = function (d) { if (d && d.timestampWrites) window.__tsPasses.push(String(d.label)); return orig.call(this, d); }; })();`,
  });
  page.on('console', (m) => { if (m.type() === 'error') errors.push(`${scene}: ${m.text()}`); if (/uncaptured|error/i.test(m.text())) console.log(`[page:${m.type()}] ${m.text()}`); });
  page.on('pageerror', (e) => errors.push(`${scene}: [pageerror] ${e.message}`));
  try {
    const sponza = scene === 'sponza';
    const q = new URLSearchParams({ seed: '1', res: '540p', scene: `/${sponza ? SPONZA : CORNELL}`, ...(sponza ? { env: `/${HDRI}` } : {}) });
    await page.goto(`http://127.0.0.1:${port}/index.html?${q}`);
    await page.waitForFunction((env) => {
      const r = window.__integration?.renderer(); const a = window.__app;
      return !!r && !!a?.scene && r.scene?.scene === a.scene && !r.loading && r.ready && (!env || r.env.present);
    }, sponza, { timeout: 240_000, polling: 100 });
    await page.evaluate((sp) => {
      const app = window.__app!;
      const store = window.__editor!.editor.store!;
      if (!sp) {
        store.add({ type: 'point', power: 30, matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1, 0.35, -0.2, 1]) });
        store.add({ type: 'rect', power: 20, sizeX: 0.15, sizeY: 0.1, visibleToCamera: false, matrix: new Float32Array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0.5, -0.28, 1]) });
      } else {
        // Sponza: camera along the long axis at ~1/5 height (M1 smoke), and a warm point light in the atrium
        const b = app.scene!.bounds, e = [0, 1, 2].map((i) => b.max[i] - b.min[i]), c = [0, 1, 2].map((i) => 0.5 * (b.min[i] + b.max[i]));
        const long = e[0] >= e[2] ? 0 : 2;
        const eye = [...c] as [number, number, number], tgt = [...c] as [number, number, number];
        eye[1] = tgt[1] = b.min[1] + 0.2 * e[1];
        eye[long] = c[long] + 0.35 * e[long]; tgt[long] = c[long] - 0.35 * e[long];
        app.camera.lookAt(eye, tgt);
        store.add({ type: 'point', power: 2000, color: [1, 0.85, 0.7], matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, c[0], b.min[1] + 0.25 * e[1], c[2], 1]) });
      }
    }, sponza);
    await frames(page, 10);
    await scenePass(page, OUT, scene);
    const t = await page.evaluate(() => ({ ...window.__app!.runTotals }));
    check(`${scene}: NaN/Inf = 0 (debug counters)`, t.nan === 0 && t.inf === 0, JSON.stringify(t));
    const st = await page.evaluate(() => ({ err: window.__integration!.renderer()!.lastError, rs: window.__integration!.renderer()!.restirError, dn: window.__integration!.renderer()!.denoiserError, gpu: window.__webgpuErrors ?? [], ts: window.__tsPasses ?? [] }));
    check(`${scene}: renderer / ReSTIR / denoiser / WebGPU error-free`, !st.err && !st.rs && !st.dn && st.gpu.length === 0, [st.err, st.rs, st.dn, ...st.gpu.slice(0, 2)].filter(Boolean).join(' | '));
    const bad = [...new Set(st.ts.filter((l) => /^rs_|restir/i.test(l) || (/^dn_/.test(l) && !/-timing$/.test(l))))];
    const timing = st.ts.filter((l) => /^dn_.*-timing$/.test(l)).length;
    check(`${scene}: no timestampWrites on a ReSTIR or a frame's denoiser pass (Q3); timing submits carry them`, bad.length === 0 && timing > 0, `bad: ${bad.join(', ') || 'none'}; ${timing} timed denoiser passes`);
  } catch (e) {
    check(`${scene}: page run completed`, false, e instanceof Error ? e.message.split('\n')[0] : String(e));
    await withTimeout(page.screenshot({ path: path.join(OUT, `${scene}-failure.png`) }), 10_000).catch(() => undefined);
  } finally {
    await withTimeout(page.close(), 20_000);
    await withTimeout(ctx.close(), 10_000);
  }
}

async function main(): Promise<number> {
  const runId = args.run ?? `m55-app-smoke-${stamp()}`;
  const OUT = path.join(ROOT, 'validation/out', runId);
  mkdirSync(OUT, { recursive: true });
  const vite = await createServer({
    root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), logLevel: 'warn',
    server: { port: 0, host: '127.0.0.1', hmr: false, watch: null, fs: { allow: [ROOT, realpathSync(path.join(ROOT, 'node_modules')), realpathSync(path.join(ROOT, 'validation/assets/downloaded'))] } },
  });
  await vite.listen();
  const port = (vite.httpServer!.address() as { port: number }).port;
  const errors: string[] = [];
  const scenes = args.scenes.split(',').filter((s) => s === 'cornell' || (s === 'sponza' && existsSync(path.join(ROOT, SPONZA))));
  try {
    const browser = await chromium.launch({ channel: 'chrome', headless: true });
    try {
      for (const scene of scenes) {
        const release = args['no-lock'] ? () => {} : await acquireGpuLock('m55-app-smoke');
        const t0 = Date.now();
        try { await pageLoad(browser, port, OUT, scene, errors); } finally { release(); }
        console.log(`[m55-app-smoke] ${scene}: ${((Date.now() - t0) / 1000).toFixed(0)} s`);
      }
    } finally {
      await withTimeout(browser.close(), 30_000);
    }
    check('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  } finally {
    await withTimeout(vite.close(), 10_000);
  }
  writeFileSync(path.join(OUT, 'report.json'), JSON.stringify({ runId, scenes, checks }, null, 1));
  const failed = checks.filter((c) => !c.ok).length;
  console.log(`${failed ? 'FAIL' : 'PASS'} m5.5 app smoke: ${checks.length - failed}/${checks.length} (screenshots: ${path.relative(ROOT, OUT)})`);
  return failed ? 1 : 0;
}

main().then((c) => process.exit(c), (e: unknown) => { console.error(e); process.exit(1); });

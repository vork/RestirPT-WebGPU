// M5 app smoke (restir-temporal-api.md §6.1 "M5 app smoke", §2.10–§2.11, TD19–TD21; T-D): the real app in headless
// Chrome, render mode ReSTIR, mode ReSTIR-interactive (temporal, RR, boost 3), on (1) the Cornell box with a point and
// a rect light added through the editor and (2) the same box lit by an HDRI, each over several page loads (the M4 Q3
// fault appeared only on some loads). Per page load:
//   static        history becomes valid and stays valid (histFrames grows), temporal counters > 0 (q′ valid, forward
//                 OK, P(s = p)), HUD temporal lines, finite image
//   animated      N frames flying the camera, moving the point light (+ rotating / brightening the HDRI): no history
//                 reset from light / env / camera edits (every advance after the first is valid), refresh frames
//                 happen (TF_REFRESH), non-finite / pending / overflow counters 0, finite image every checkpoint
//   views         every M5 view (480–497) and the reservoir views at the tap "after temporal" render a finite AOV
//                 while animating; 480 / 491 are written; the inspector decodes the temporal records
//   pause (TD20)  a paused frame encodes no ReSTIR pass (no advance, held frames counted, image unchanged and finite)
//                 and the history survives the pause
//   resets        freeze seed ⇒ reset every frame; "reset temporal history" ⇒ one reset; a mode switch (config) ⇒
//                 one reset; HDRI map swap (setEnvironment) ⇒ one reset with reason env-map; temporal off / on works
// Global: no uncaptured WebGPU / console errors, no renderer / ReSTIR compile errors, NaN/Inf = 0 (debug counters,
// finalize counters, arena counters), no timestampWrites on any ReSTIR pass (Q3, C8).
// The GPU lock is taken per page load (≤ ~3 min each); page / browser teardown has timeouts (never hold the lock on
// a hung page). Screenshots and report.json go to validation/out/<run>/.
//   npx tsx validation/harness/m5-app-smoke.ts [--run <id>] [--no-lock] [--loads 2] [--frames 60] [--scenes cornell,cornell-hdri]
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

interface AdvLog { hist: boolean; flags: number; reasons: string[]; histFrames: number }
declare global {
  interface Window {
    __app?: App; __integration?: Integration; __editor?: EditorHandle; __webgpuErrors?: string[];
    __adv?: AdvLog[]; __tsPasses?: string[]; __anim?: { on: boolean; k: number; hdri: boolean };
  }
}

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const HDRI = 'validation/assets/downloaded/hdri/studio_small_09_1k.hdr';
const TF = { HIST_VALID: 1, REFRESH: 8, LIGHT_MOVED: 128, ENV_MOVED: 32, ENV_RADIO: 64 } as const;
const M5_VIEWS = Array.from({ length: 18 }, (_, i) => 480 + i);
const RES_VIEWS_T = [400, 401, 403, 404, 409];
const SCREENSHOT_VIEWS = new Set([480, 481, 482, 484, 486, 488, 489, 491, 492, 493, 495, 496, 497]);
const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const { values: args } = parseArgs({
  options: {
    run: { type: 'string' }, 'no-lock': { type: 'boolean', default: false }, loads: { type: 'string', default: '2' }, frames: { type: 'string', default: '60' },
    scenes: { type: 'string', default: 'cornell,cornell-hdri' },
  },
});
const LOADS = Math.max(1, Number(args.loads));
const NFRAMES = Math.max(10, Number(args.frames));

const checks: { name: string; ok: boolean; detail?: string }[] = [];
const check = (name: string, ok: boolean, detail?: string) => { checks.push({ name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`); return ok; };
/** Page / browser teardown can hang behind in-flight GPU work: never hold the GPU lock on it. */
const withTimeout = <T>(p: Promise<T>, ms: number) => Promise.race([p, new Promise<void>((r) => setTimeout(r, ms))]);

async function frames(page: Page, n: number): Promise<void> {
  const f0 = await page.evaluate(() => window.__app!.frameCounter);
  await page.waitForFunction((t) => window.__app!.frameCounter >= t, f0 + n, { timeout: 180_000, polling: 50 });
}
const hud = (page: Page) => page.evaluate(() => window.__integration!.renderer()!.hudLines());
const advLog = (page: Page) => page.evaluate(() => window.__adv!.splice(0));
const temporalState = (page: Page) => page.evaluate(() => {
  const r = window.__integration!.renderer()!;
  const t = r.restirTemporal!;
  const h = r.restirHud!;
  return { temporalFrames: t.temporalFrames, heldFrames: t.heldFrames, totals: { ...h.totals }, errors: h.errorCount(), latest: h.latest ? { rsc: h.latest.rsc, q1: h.latest.queues[1], q2: h.latest.queues[2] } : undefined };
});

/** In-page GPU readbacks: colour target (finite, mean), debug AOV (non-finite count for scalar / vec3 views, written
 *  pixels), ReSTIR finalize counters (nonFinite, bvhOverflow, bvhItercap, negative). */
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
        if (u[4 * i] !== 0 && u[4 * i] !== 0xFFFFFFFF) aovWritten++;
        if (view.kind !== 'code') for (let c = 0; c < comps; c++) if (!Number.isFinite(a[4 * i + c])) aovNonFinite++;
      }
    }
    const rs = r.restir;
    const finalize = rs ? Array.from(new Uint32Array(await read(rs.counters, 16))) : [];
    return { colour: { nonFinite, mean: n ? sum / n : 0 }, aovNonFinite, aovWritten, finalize };
  });
}

async function setAnim(page: Page, on: boolean): Promise<void> {
  await page.evaluate((o) => { window.__anim!.on = o; }, on);
}

/** Install the per-frame animator (camera orbit, point light path, HDRI rotation + strength wobble) and the advance
 *  log (every RestirAdvance of the interactive kernel). */
async function instrument(page: Page, hdri: boolean): Promise<void> {
  await page.evaluate((h) => {
    const app = window.__app!;
    const r = window.__integration!.renderer()!;
    const k = r.restir!.kernel;
    window.__adv = [];
    const orig = k.advanceInteractive.bind(k);
    k.advanceInteractive = (fu, o) => {
      const a = orig(fu, o);
      window.__adv!.push({ hist: a.histValid, flags: a.flags, reasons: [...a.reasons], histFrames: a.temporal.histFrames });
      if (window.__adv!.length > 5000) window.__adv!.shift();
      return a;
    };
    window.__anim = { on: false, k: 0, hdri: h };
    const store = window.__editor!.editor.store!;
    const point = store.list().find((l) => l.type === 'point')!;
    const b = app.scene!.bounds;
    const c = [0, 1, 2].map((i) => 0.5 * (b.min[i] + b.max[i])) as [number, number, number];
    const ext = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
    const eye0 = [...app.camera.position] as [number, number, number];
    app.beforeFrame.add((f) => {
      const a = window.__anim!;
      if (!a.on || !f.advance) return;
      const s = a.k++;
      // camera: small orbit / dolly around the initial eye (a fly path), always looking at the box centre
      const ph = 0.05 * s;
      app.camera.lookAt([eye0[0] + 0.08 * ext * Math.sin(ph), eye0[1] + 0.03 * ext * Math.sin(0.7 * ph), eye0[2] + 0.05 * ext * Math.cos(ph)], c);
      // point light on a small loop inside the box
      const m = new Float32Array(point.matrix);
      m[12] = 0.1 + 0.12 * Math.sin(0.09 * s); m[14] = -0.2 + 0.1 * Math.cos(0.09 * s);
      store.update(point.id, { matrix: m }, 'smoke');
      if (a.hdri) {
        app.envParams.rotationDeg = (app.envParams.rotationDeg + 1.5) % 360;
        app.envParams.strength = 1 + 0.25 * Math.sin(0.2 * s);
        app.envParamsChanged();
      }
    });
  }, hdri);
}

async function scenePass(page: Page, OUT: string, name: string, load: number): Promise<void> {
  const tag = `${name}#${load}`;
  const hdri = name === 'cornell-hdri';
  // ---- ReSTIR on, ReSTIR-interactive with temporal reuse
  await page.evaluate(async () => {
    const r = window.__integration!.renderer()!;
    r.options.renderMode = 'restir';
    await r.setOptions({ restirMode: 'interactive', temporal: true });
    await r.prepareRestir();
    window.__app!.resetTemporalHistory();
  });
  await page.waitForFunction(() => window.__integration!.renderer()!.hudLines().some((l) => /^ReSTIR temporal: q′ valid/.test(l)), undefined, { timeout: 180_000, polling: 100 });
  check(`${tag}: ReSTIR compiled`, await page.evaluate(() => !window.__integration!.renderer()!.restirError), await page.evaluate(() => window.__integration!.renderer()!.restirError ?? ''));
  await instrument(page, hdri);

  // ---- static: history valid, counters > 0
  await frames(page, 24);
  let log = await advLog(page);
  const st0 = await readState(page);
  const ts0 = await temporalState(page);
  const last0 = log[log.length - 1];
  check(`${tag}: static — history valid and growing, finite image`, !!last0 && last0.hist && last0.histFrames >= 10 && st0.colour.nonFinite === 0 && st0.colour.mean > 0 && st0.finalize[0] === 0,
    `last ${JSON.stringify(last0)} mean ${st0.colour.mean.toExponential(3)} finalize ${st0.finalize}`);
  check(`${tag}: static — temporal counters > 0 (q′ valid, forward OK, s = p)`, ts0.totals.tQvalid > 0 && ts0.totals.tFwdOk > 0 && ts0.totals.tSelP > 0,
    `tQvalid ${ts0.totals.tQvalid} tFwdOk ${ts0.totals.tFwdOk} tSelP ${ts0.totals.tSelP} over ${ts0.totals.temporalFrames} frames`);
  const hl = await hud(page);
  check(`${tag}: HUD temporal lines`, hl.some((l) => /^ReSTIR temporal: hist \d+ frames/.test(l)) && hl.some((l) => /^ReSTIR refresh:/.test(l)) && hl.some((l) => /boost 3/.test(l)),
    hl.filter((l) => l.startsWith('ReSTIR')).join(' | '));
  await page.screenshot({ path: path.join(OUT, `${tag.replace('#', '-')}-static.png`) });

  // ---- animated: camera + light (+ env); no resets, refresh frames, counters clean
  await advLog(page);
  await setAnim(page, true);
  let minMean = Infinity, nf = 0;
  for (let i = 0; i < NFRAMES; i += 20) {
    await frames(page, Math.min(20, NFRAMES - i));
    const s = await readState(page);
    nf += s.colour.nonFinite + s.finalize[0];
    minMean = Math.min(minMean, s.colour.mean);
  }
  log = await advLog(page);
  const resets = log.filter((a) => !a.hist);
  const refresh = log.filter((a) => a.flags & TF.REFRESH).length, moved = log.filter((a) => a.flags & TF.LIGHT_MOVED).length;
  const envMoved = log.filter((a) => a.flags & TF.ENV_MOVED).length;
  check(`${tag}: animated — no history reset from camera / light${hdri ? ' / env' : ''} edits`, log.length >= NFRAMES * 0.9 && resets.length === 0,
    `${log.length} advances, ${resets.length} resets ${resets.slice(0, 2).map((a) => a.reasons.join(',')).join(' | ')}; histFrames ${log[0]?.histFrames}→${log[log.length - 1]?.histFrames}`);
  check(`${tag}: animated — refresh frames (TF_REFRESH ${refresh}, LIGHT_MOVED ${moved}${hdri ? `, ENV_MOVED ${envMoved}` : ''})`,
    refresh >= log.length * 0.8 && moved >= log.length * 0.8 && (!hdri || envMoved >= log.length * 0.8));
  const ts1 = await temporalState(page);
  check(`${tag}: animated — finite image, arena error counters 0`, nf === 0 && minMean > 0 && ts1.errors === 0,
    `non-finite ${nf} min mean ${minMean.toExponential(3)} errors ${ts1.errors} ${JSON.stringify({ tNonFinite: ts1.totals.tNonFinite, tPendingLeft: ts1.totals.tPendingLeft, tQueueOverflow: ts1.totals.tQueueOverflow })}`);
  check(`${tag}: animated — temporal counters > 0`, (ts1.latest?.rsc.tQvalid ?? 0) > 0 && ts1.totals.tFwdOk > ts0.totals.tFwdOk && ts1.totals.tSelP > ts0.totals.tSelP,
    JSON.stringify({ qvalid: ts1.latest?.rsc.tQvalid, disocc: ts1.latest?.rsc.tDisocc, refreshRecs: ts1.latest?.rsc.tRefreshRecs, rays: ts1.latest?.rsc.tRefreshRays, q1: ts1.latest?.q1, q2: ts1.latest?.q2 }));
  await page.screenshot({ path: path.join(OUT, `${tag.replace('#', '-')}-animated.png`) });

  // ---- every M5 view (and the reservoir views at the tap "after temporal") while animating
  const bad: string[] = [];
  const written: Record<number, number> = {};
  let rendered = 0;
  const views: [number, number][] = [...M5_VIEWS.map((id) => [id, 0] as [number, number]), ...RES_VIEWS_T.map((id) => [id, 2] as [number, number])];
  for (const [id, tap] of views) {
    const beforeView = await page.evaluate(([v, tp]) => {
      const a = window.__app!; a.selectDebugView(v); a.debugSettings.tap = tp;
      return window.__integration!.renderer()!.restirTemporal!.temporalFrames;
    }, [id, tap] as const);
    // RS_DEBUG_STRIP prepares instrumented shaders on the first view. Screen frames may show the PT fallback;
    // wait for three actual ReSTIR advances before inspecting this view's AOV.
    await page.waitForFunction((n) => (window.__integration!.renderer()!.restirTemporal?.temporalFrames ?? 0) >= n + 3,
      beforeView, { timeout: 180_000, polling: 50 });
    const s = await readState(page);
    rendered++;
    written[id] = s.aovWritten;
    if (s.aovNonFinite > 0) bad.push(`${id}@tap${tap}: ${s.aovNonFinite} non-finite`);
    if (load === 1 && tap === 0 && SCREENSHOT_VIEWS.has(id)) await page.screenshot({ path: path.join(OUT, `${tag.replace('#', '-')}-view-${id}.png`) });
  }
  const registered = await page.evaluate(() => window.__app!.debug.registry.list().filter((v) => v.id >= 480 && v.id < 500).length);
  check(`${tag}: every M5 view (480–497) renders with a finite AOV`, registered === 18 && bad.length === 0, `${registered} registered, ${rendered} renders${bad.length ? `; ${bad.slice(0, 4).join('; ')}` : ''}`);
  check(`${tag}: views 480 (q′ validity), 491 (c_out), 492 (selection) and 400@after-temporal are written`,
    written[480] > 0 && written[491] > 0 && written[492] > 0 && written[400] > 0, `480 ${written[480]} 491 ${written[491]} 492 ${written[492]} 400 ${written[400]} 497 ${written[497]}`);
  await page.evaluate(() => { const a = window.__app!; a.selectDebugView(0); a.debugSettings.tap = 0; });

  // ---- inspector: temporal records at a hit pixel
  await page.evaluate(() => {
    const a = window.__app!;
    a.debugSettings.probePixel = [a.targets.width >> 1, Math.floor(a.targets.height * 0.6)];
    a.debugSettings.probeEnabled = true;
    window.__integration!.restirUi().inspector!.setVisible(true);
  });
  await frames(page, 12);
  await page.waitForFunction(() => !!window.__integration!.restirUi().inspector?.latest?.temporal?.select, undefined, { timeout: 30_000 }).catch(() => undefined);
  const insp = await page.evaluate(() => {
    const d = window.__integration!.restirUi().inspector!.latest;
    return d ? { taps: d.reservoirs.map((r) => r.tap), temporal: d.temporal ? { flags: d.temporal.flagNames, sel: d.temporal.select?.selName, fwd: d.temporal.forward?.code.name, pick: d.temporal.pick } : undefined, paths: [...d.paths.keys()] } : undefined;
  });
  check(`${tag}: inspector decodes the temporal records (pick, header, select; reservoir after temporal)`,
    !!insp?.temporal?.sel && !!insp.temporal.pick && insp.taps.includes(2), JSON.stringify(insp));
  if (load === 1) await page.screenshot({ path: path.join(OUT, `${tag.replace('#', '-')}-inspector.png`) });
  await page.evaluate(() => { window.__integration!.restirUi().inspector!.setVisible(false); window.__app!.debugSettings.probeEnabled = false; });

  // ---- pause (TD20): no ReSTIR pass, frame held, history survives
  await setAnim(page, false);
  await frames(page, 4);
  await advLog(page);
  const beforePause = await temporalState(page);
  await page.evaluate(() => window.__app!.setPaused(true));
  await frames(page, 12);
  const paused = await temporalState(page);
  const stP = await readState(page);
  const advP = await advLog(page);
  check(`${tag}: pause encodes no ReSTIR pass (no advance, frames held, finite image)`,
    paused.temporalFrames - beforePause.temporalFrames <= 1 && advP.length <= 1 && paused.heldFrames - beforePause.heldFrames >= 10 && stP.colour.nonFinite === 0 && stP.colour.mean > 0,
    `temporal frames +${paused.temporalFrames - beforePause.temporalFrames}, advances ${advP.length}, held +${paused.heldFrames - beforePause.heldFrames}, mean ${stP.colour.mean.toExponential(3)}`);
  await page.evaluate(() => window.__app!.step());
  await frames(page, 3);
  const stepLog = await advLog(page);
  await page.evaluate(() => window.__app!.setPaused(false));
  await frames(page, 6);
  const resumed = await advLog(page);
  check(`${tag}: history survives the pause (step and resume advance with valid history)`, stepLog.length === 1 && stepLog[0].hist && resumed.length >= 4 && resumed.every((a) => a.hist),
    `step ${JSON.stringify(stepLog)} resume ${resumed.length} advances, hist ${resumed.map((a) => a.hist ? 1 : 0).join('')}`);

  // ---- resets: freeze seed, reset control, mode switch (config), env map swap, temporal off / on
  await page.evaluate(() => { window.__app!.render.freezeSeed = true; });
  await frames(page, 6);
  const frozen = await advLog(page);
  await page.evaluate(() => { window.__app!.render.freezeSeed = false; });
  await frames(page, 6);
  const unfrozen = await advLog(page);
  check(`${tag}: freeze seed ⇒ history reset every frame; unfreeze ⇒ valid again`,
    frozen.length >= 4 && frozen.slice(1).every((a) => !a.hist && a.reasons.includes('state.reset')) && unfrozen.slice(2).every((a) => a.hist),
    `frozen ${frozen.map((a) => a.hist ? 1 : 0).join('')} unfrozen ${unfrozen.map((a) => a.hist ? 1 : 0).join('')}`);
  await page.evaluate(() => window.__app!.resetTemporalHistory());
  await frames(page, 6);
  const rst = await advLog(page);
  const oneReset = (l: AdvLog[]) => { const i = l.findIndex((a) => !a.hist); return i >= 0 && l.slice(i + 1).length >= 2 && l.slice(i + 1).every((a) => a.hist); };
  check(`${tag}: "reset temporal history" ⇒ exactly one reset`, oneReset(rst) && rst.find((a) => !a.hist)!.reasons.includes('state.reset'),
    rst.map((a) => (a.hist ? '1' : `0(${a.reasons.join(',')})`)).join(' '));
  await page.evaluate(async () => { await window.__integration!.renderer()!.setOptions({ restirMode: 'unbiased' }); });
  await frames(page, 8);
  const sw = await advLog(page);
  const swState = await readState(page);
  check(`${tag}: mode switch (config change) ⇒ one reset, then valid; ReSTIR-unbiased renders`,
    oneReset(sw) && sw.find((a) => !a.hist)!.reasons.some((x) => x === 'config' || x === 'reallocated') && swState.colour.nonFinite === 0 && swState.colour.mean > 0,
    sw.map((a) => (a.hist ? '1' : `0(${a.reasons.join(',')})`)).join(' '));
  if (hdri) {
    await page.evaluate(async () => { const a = window.__app!; await window.__integration!.renderer()!.setEnvironment(a.env); });
    await frames(page, 6);
    const ms = await advLog(page);
    check(`${tag}: HDRI map swap ⇒ one reset (reason env-map)`, oneReset(ms) && ms.find((a) => !a.hist)!.reasons.includes('env-map'),
      ms.map((a) => (a.hist ? '1' : `0(${a.reasons.join(',')})`)).join(' '));
  }
  await page.evaluate(async () => { await window.__integration!.renderer()!.setOptions({ restirMode: 'interactive', temporal: false }); });
  await frames(page, 8);
  const off = await readState(page);
  const offTf = await temporalState(page);
  await page.evaluate(async () => { await window.__integration!.renderer()!.setOptions({ temporal: true }); });
  await frames(page, 10);
  const on = await readState(page);
  const onLog = await advLog(page);
  check(`${tag}: temporal off renders (no temporal frames) and back on resumes temporal reuse`,
    off.colour.nonFinite === 0 && off.colour.mean > 0 && on.colour.nonFinite === 0 && onLog.length >= 5 && onLog[onLog.length - 1].hist,
    `off mean ${off.colour.mean.toExponential(3)} (temporal frames ${offTf.temporalFrames}), on ${onLog.map((a) => (a.hist ? 1 : 0)).join('')}`);
  const fin = await temporalState(page);
  check(`${tag}: arena error counters 0 at the end`, fin.errors === 0, JSON.stringify(fin.totals));
}

async function pageLoad(browser: Browser, port: number, OUT: string, scene: 'cornell' | 'cornell-hdri', load: number, errors: string[]): Promise<void> {
  const tag = `${scene}#${load}`;
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const page = await ctx.newPage();
  // tsx (esbuild keepNames) wraps named closures of page.evaluate bodies in __name(): define it in the page.
  // Record every compute pass that carries timestampWrites (Q3 / C8: none may be a ReSTIR pass).
  await page.addInitScript({
    content: `window.__name = (f) => f; window.__tsPasses = [];
      (() => { const P = globalThis.GPUCommandEncoder && GPUCommandEncoder.prototype; if (!P) return; const orig = P.beginComputePass;
        P.beginComputePass = function (d) { if (d && d.timestampWrites) window.__tsPasses.push(String(d.label)); return orig.call(this, d); }; })();`,
  });
  page.on('console', (m) => { if (m.type() === 'error') errors.push(`${tag}: ${m.text()}`); if (/uncaptured|error/i.test(m.text())) console.log(`[page:${m.type()}] ${m.text()}`); });
  page.on('pageerror', (e) => errors.push(`${tag}: [pageerror] ${e.message}`));
  try {
    const q = new URLSearchParams({ seed: String(load), res: '540p', scene: '/validation/assets/cornell/cornell.glb', ...(scene === 'cornell-hdri' ? { env: `/${HDRI}` } : {}) });
    await page.goto(`http://127.0.0.1:${port}/index.html?${q}`);
    await page.waitForFunction((env) => {
      const r = window.__integration?.renderer(); const a = window.__app;
      return !!r && !!a?.scene && r.scene?.scene === a.scene && !r.loading && r.ready && (!env || r.env.present);
    }, scene === 'cornell-hdri', { timeout: 180_000, polling: 100 });
    await page.evaluate((hdri) => {
      const store = window.__editor!.editor.store!;
      store.add({ type: 'point', power: hdri ? 10 : 30, matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1, 0.35, -0.2, 1]) });
      store.add({ type: 'rect', power: 20, sizeX: 0.15, sizeY: 0.1, visibleToCamera: false, matrix: new Float32Array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0.5, -0.28, 1]) });
    }, scene === 'cornell-hdri');
    await frames(page, 10);
    await scenePass(page, OUT, scene, load);
    const t = await page.evaluate(() => ({ ...window.__app!.runTotals }));
    check(`${tag}: NaN/Inf = 0 (debug counters)`, t.nan === 0 && t.inf === 0, JSON.stringify(t));
    const st = await page.evaluate(() => ({ err: window.__integration!.renderer()!.lastError, rs: window.__integration!.renderer()!.restirError, gpu: window.__webgpuErrors ?? [], ts: window.__tsPasses ?? [] }));
    check(`${tag}: renderer / ReSTIR / WebGPU error-free`, !st.err && !st.rs && st.gpu.length === 0, [st.err, st.rs, ...st.gpu.slice(0, 2)].filter(Boolean).join(' | '));
    const rsTs = [...new Set(st.ts.filter((l) => /^rs_|restir/i.test(l)))];
    check(`${tag}: no timestampWrites on any ReSTIR pass (Q3, C8)`, rsTs.length === 0, `${st.ts.length} timestamped passes; ReSTIR: ${rsTs.join(', ') || 'none'}`);
  } catch (e) {
    check(`${tag}: page run completed`, false, e instanceof Error ? e.message.split('\n')[0] : String(e));
    await withTimeout(page.screenshot({ path: path.join(OUT, `${tag.replace('#', '-')}-failure.png`) }), 10_000).catch(() => undefined);
  } finally {
    await withTimeout(page.close(), 20_000);
    await withTimeout(ctx.close(), 10_000);
  }
}

async function main(): Promise<number> {
  const runId = args.run ?? `m5-app-smoke-${stamp()}`;
  const OUT = path.join(ROOT, 'validation/out', runId);
  mkdirSync(OUT, { recursive: true });
  // node_modules may be a symlink into another checkout (worktrees): allow its real path; no HMR / watch (edits by
  // other agents must not reload the page mid-run).
  const vite = await createServer({
    root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), logLevel: 'warn',
    server: { port: 0, host: '127.0.0.1', hmr: false, watch: null, fs: { allow: [ROOT, realpathSync(path.join(ROOT, 'node_modules'))] } },
  });
  await vite.listen();
  const port = (vite.httpServer!.address() as { port: number }).port;
  const errors: string[] = [];
  const scenes = args.scenes.split(',').filter((s): s is 'cornell' | 'cornell-hdri' => s === 'cornell' || s === 'cornell-hdri');
  try {
    // One fresh browser for the whole run (the Q3 fault showed on non-first page loads of one browser); the GPU lock is
    // taken per page load.
    const browser = await chromium.launch({ channel: 'chrome', headless: true });
    try {
      for (let load = 1; load <= LOADS; load++) {
        for (const scene of scenes) {
          const release = args['no-lock'] ? () => {} : await acquireGpuLock('m5-app-smoke');
          const t0 = Date.now();
          try { await pageLoad(browser, port, OUT, scene, load, errors); } finally { release(); }
          console.log(`[m5-app-smoke] ${scene}#${load}: ${((Date.now() - t0) / 1000).toFixed(0)} s`);
        }
      }
    } finally {
      await withTimeout(browser.close(), 30_000);
    }
    check('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  } finally {
    await withTimeout(vite.close(), 10_000);
  }
  writeFileSync(path.join(OUT, 'report.json'), JSON.stringify({ runId, loads: LOADS, frames: NFRAMES, scenes, checks }, null, 1));
  const failed = checks.filter((c) => !c.ok).length;
  console.log(`${failed ? 'FAIL' : 'PASS'} m5 app smoke: ${checks.length - failed}/${checks.length} (screenshots: ${path.relative(ROOT, OUT)})`);
  return failed ? 1 : 0;
}

main().then((c) => process.exit(c), (e: unknown) => { console.error(e); process.exit(1); });

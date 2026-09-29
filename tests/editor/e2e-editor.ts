// M3a editor end-to-end in headless Chrome (not a vitest file; like validation/harness/m2-app-export.ts):
//   real app + Cornell GLB → place lights on surfaces through the V-buffer (real mouse clicks), pick, drag the
//   translate/rotate gizmos with real pointer events (checked against the pure gizmo math), keyboard delete /
//   duplicate / undo / redo, keyframes + presets + validation-clock playback, editor scene.json round trip,
//   "Export for Cycles" → /api/reference → headless Blender at tiny spp (GPU lock released while Blender renders),
//   and the compare view auto-loading the EXRs (split / relative error / t-map screenshots).
//   npx tsx tests/editor/e2e-editor.ts [--no-blender] [--out validation/out/m3a-editor-e2e]
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium, type Page } from 'playwright';
import { createServer } from 'vite';
import type { App } from '../../src/app/app.ts';
import type { EditorHandle } from '../../src/app/editor/index.ts';
import type { Integration } from '../../src/app/integration.ts';
import { dragScreen, gizmoLayout, type HandleId } from '../../src/app/editor/gizmos.ts';
import { projectPoint, sunScreenPosition, type ViewInfo } from '../../src/app/editor/picking.ts';
import { presetOrbit } from '../../src/core/scene/animation.ts';
import { isRigid } from '../../src/core/scene/scene-package.ts';
import { acquireGpuLock } from '../../validation/harness/gpu-lock.ts';

declare global { interface Window { __app?: App; __integration?: Integration; __editor?: EditorHandle; __webgpuErrors?: string[] } }

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const { values: args } = parseArgs({ options: { 'no-blender': { type: 'boolean', default: false }, out: { type: 'string', default: 'validation/out/m3a-editor-e2e' } } });
const OUT = path.resolve(ROOT, args.out!);

const checks: { name: string; ok: boolean; detail?: string }[] = [];
const check = (name: string, ok: boolean, detail?: string) => { checks.push({ name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`); return ok; };

let release: (() => void) | undefined;
async function lock(): Promise<void> { release ??= await acquireGpuLock('e2e-editor', { pollMs: 2000 }); }
function unlock(): void { release?.(); release = undefined; }

async function frames(page: Page, n: number): Promise<void> {
  const f0 = await page.evaluate(() => window.__app!.frameCounter);
  await page.waitForFunction((t) => window.__app!.frameCounter >= t, f0 + n, { timeout: 60_000, polling: 30 });
}

const viewOf = (page: Page): Promise<ViewInfo> => page.evaluate(() => {
  const v = window.__editor!.editor.view();
  return { camToWorld: Array.from(v.camToWorld), yfov: v.yfov, width: v.width, height: v.height };
});
const lightsOf = (page: Page) => page.evaluate(() => window.__editor!.editor.lights.map((l) => ({ id: l.id, type: l.type, name: l.name, matrix: Array.from(l.matrix), power: l.power })));
const selected = (page: Page) => page.evaluate(() => window.__editor!.editor.selected);
const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

/** Real pointer drag in N steps. */
async function drag(page: Page, x0: number, y0: number, x1: number, y1: number, steps = 8): Promise<void> {
  await page.mouse.move(x0, y0);
  await page.mouse.down({ button: 'left' });
  await sleep(50); // pressAt is async (resolves the drag on the next microtask)
  for (let i = 1; i <= steps; i++) await page.mouse.move(x0 + ((x1 - x0) * i) / steps, y0 + ((y1 - y0) * i) / steps);
  await page.mouse.up({ button: 'left' });
  await frames(page, 2);
}

async function main(): Promise<number> {
  mkdirSync(OUT, { recursive: true });
  await lock();
  const vite = await createServer({ root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), server: { port: 0, host: '127.0.0.1' }, logLevel: 'warn' });
  await vite.listen();
  const port = (vite.httpServer!.address() as { port: number }).port;
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const consoleErrors: string[] = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
    page.on('console', (m) => { if (m.type() === 'error') { consoleErrors.push(m.text()); console.log(`[page:error] ${m.text()}`); } });
    page.on('pageerror', (e) => { consoleErrors.push(e.message); console.log(`[pageerror] ${e.message}`); });
    // tsx/esbuild (keepNames) wraps named closures in __name(); page.evaluate callbacks run in the page without it
    await page.addInitScript(() => { (globalThis as unknown as { __name: (f: unknown) => unknown }).__name = (f) => f; });
    await page.goto(`http://127.0.0.1:${port}/?seed=1&hud=0`);
    await page.waitForFunction(() => !!window.__integration && !!window.__app && !!window.__editor, undefined, { timeout: 60_000 });
    await page.evaluate(() => window.__integration!.sceneReady);
    await frames(page, 20);
    // make the Lights folder area not overlap the viewport centre (panel is on the right, 300 px)
    const st0 = await page.evaluate(() => ({ scene: window.__app!.scene?.name, fileLights: window.__app!.scene?.lights.length, store: window.__editor!.editor.lights.length, bounds: window.__app!.scene!.bounds, vb: !!window.__integration!.renderer()?.vbuffer }));
    check('scene loaded, editor bound to its LightStore', st0.store === st0.fileLights && st0.vb, JSON.stringify(st0));
    const b = st0.bounds;
    const cx = (b.min[0] + b.max[0]) / 2, cz = (b.min[2] + b.max[2]) / 2, cy = (b.min[1] + b.max[1]) / 2;
    let v = await viewOf(page);

    // ---- place a rect on the ceiling (real click → V-buffer readback) ----
    await page.evaluate(() => window.__editor!.editor.beginPlacement('rect'));
    const ceil = projectPoint(v, [cx + 0.1 * (b.max[0] - b.min[0]), b.max[1] - 1e-3, cz])!;
    await page.mouse.click(ceil.x, ceil.y);
    await page.waitForFunction((n) => window.__editor!.editor.lights.length > n, st0.store, { timeout: 10_000 });
    let L = await lightsOf(page);
    const rect = L.find((l) => l.type === 'rect')!;
    const rm = rect.matrix;
    check('rect placed on the ceiling via V-buffer: at the surface, emitting down', Math.abs(rm[13] - b.max[1]) < 0.02 * (b.max[1] - b.min[1]) && rm[9] > 0.99 && isRigid(rm),
      `pos ${rm.slice(12, 15).map((x) => x.toFixed(4))}, emit −Z = ${[-rm[8], -rm[9], -rm[10]].map((x) => x.toFixed(3))}`);

    // ---- place a spot on the back wall ----
    await page.evaluate(() => window.__editor!.editor.beginPlacement('spot'));
    const wall = projectPoint(v, [cx - 0.15 * (b.max[0] - b.min[0]), b.min[1] + 0.8 * (b.max[1] - b.min[1]), b.min[2] + 1e-3])!; // above the blocks
    await page.mouse.click(wall.x, wall.y);
    await page.waitForFunction(() => window.__editor!.editor.lights.some((l) => l.type === 'spot'), undefined, { timeout: 10_000 });
    L = await lightsOf(page);
    const spot = L.find((l) => l.type === 'spot')!;
    check('spot placed on the back wall, emitting toward the room (+Z)', spot.matrix[10] < -0.99 && spot.matrix[14] > b.min[2], `pos ${spot.matrix.slice(12, 15).map((x) => x.toFixed(3))}`);
    check('newly placed light is selected', (await selected(page)) === spot.id);

    // ---- point + sun ----
    const pointId = await page.evaluate(() => window.__editor!.editor.addInFront('point'));
    await page.evaluate(() => window.__editor!.editor.beginPlacement('sun'));
    await frames(page, 2);
    L = await lightsOf(page);
    check('added point and sun (5 light types available)', L.some((l) => l.type === 'point') && L.some((l) => l.type === 'sun'), L.map((l) => `${l.type}#${l.id}`).join(' '));

    // ---- picking (CPU proxies) ----
    await page.evaluate(() => window.__editor!.editor.select(undefined));
    await frames(page, 1);
    const sp = projectPoint(v, spot.matrix.slice(12, 15))!;
    await page.mouse.click(sp.x, sp.y);
    await frames(page, 1);
    check('click on the spot apex selects it (proxy picking)', (await selected(page)) === spot.id);
    const sun = L.find((l) => l.type === 'sun')!;
    const [sx, sy] = sunScreenPosition(0, v);
    await page.mouse.click(sx, sy);
    await frames(page, 1);
    check('click on the sun icon selects the sun', (await selected(page)) === sun.id);
    await page.mouse.click(sp.x, sp.y);
    await frames(page, 1);

    // ---- translate gizmo drag (X axis) vs the pure gizmo math ----
    await page.evaluate(() => window.__editor!.editor.setMode('translate'));
    await frames(page, 2);
    v = await viewOf(page);
    const centre = spot.matrix.slice(12, 15) as [number, number, number];
    const g = gizmoLayout(centre, v);
    const tip = projectPoint(v, [centre[0] + 0.8 * g.size, centre[1], centre[2]])!;
    const h0 = await page.evaluate(([x, y]) => { const e = window.__editor!.editor; e.moveTo(x, y); return (e as unknown as { hot?: string }).hot; }, [tip.x, tip.y]);
    check('X axis handle is hot under the cursor', h0 === 'x', String(h0));
    await drag(page, tip.x, tip.y, tip.x + 120, tip.y + 10);
    L = await lightsOf(page);
    const moved = L.find((l) => l.id === spot.id)!.matrix;
    const expected = dragScreen('x' as HandleId, centre, spot.matrix, v, tip.x, tip.y, tip.x + 120, tip.y + 10)!;
    check('X drag moves the spot along X exactly like the gizmo math (y, z unchanged)', maxAbs(moved, Array.from(Float32Array.from(expected))) < 1e-5 && moved[12] > centre[0] + 0.01,
      `x ${centre[0].toFixed(4)} → ${moved[12].toFixed(4)} (expected ${expected[12].toFixed(4)}), Δyz ${maxAbs(moved.slice(13, 15), centre.slice(1))}`);
    const undoInfo = await page.evaluate(() => window.__editor!.editor.undo.undoLabel);
    check('the drag is one undo step', undoInfo === 'move light', String(undoInfo));
    await page.keyboard.press('Control+KeyZ');
    await frames(page, 1);
    L = await lightsOf(page);
    check('Ctrl+Z restores the start matrix exactly', maxAbs(L.find((l) => l.id === spot.id)!.matrix, spot.matrix) === 0);
    await page.keyboard.press('Control+Shift+KeyZ');
    await frames(page, 1);
    L = await lightsOf(page);
    check('Shift+Ctrl+Z redoes the move', maxAbs(L.find((l) => l.id === spot.id)!.matrix, moved) === 0);
    const camMoved = await page.evaluate(() => window.__app!.camera.held.size);
    check('editor shortcuts do not leak into the fly camera', camMoved === 0);

    // ---- rotate gizmo (Z ring, facing the camera) ----
    await page.keyboard.press('KeyR');
    await frames(page, 2);
    v = await viewOf(page);
    const c2 = moved.slice(12, 15) as [number, number, number];
    const g2 = gizmoLayout(c2, v);
    const a0 = 0.3, a1 = 1.1;
    const r0 = projectPoint(v, [c2[0] + g2.size * Math.cos(a0), c2[1] + g2.size * Math.sin(a0), c2[2]])!;
    const r1 = projectPoint(v, [c2[0] + g2.size * Math.cos(a1), c2[1] + g2.size * Math.sin(a1), c2[2]])!;
    await drag(page, r0.x, r0.y, r1.x, r1.y);
    L = await lightsOf(page);
    const rot = L.find((l) => l.id === spot.id)!.matrix;
    const expRot = dragScreen('rz', c2, moved, v, r0.x, r0.y, r1.x, r1.y)!;
    check('ring drag rotates about world Z like the gizmo math; rigid; position fixed', maxAbs(rot, Array.from(Float32Array.from(expRot))) < 1e-5 && isRigid(rot) && maxAbs(rot.slice(12, 15), c2) === 0,
      `max |Δ| ${maxAbs(rot, Array.from(Float32Array.from(expRot))).toExponential(2)}`);
    await page.keyboard.press('KeyG');

    // ---- delete / duplicate / undo via keyboard (point light) ----
    await page.evaluate((id) => window.__editor!.editor.select(id), pointId);
    await page.keyboard.press('Delete');
    await frames(page, 1);
    const afterDel = await lightsOf(page);
    await page.keyboard.press('Control+KeyZ');
    await frames(page, 1);
    const afterUndo = await lightsOf(page);
    check('Delete removes the point light; Ctrl+Z restores it with the same id', !afterDel.some((l) => l.id === pointId) && afterUndo.some((l) => l.id === pointId));
    await page.evaluate((id) => window.__editor!.editor.select(id), pointId);
    await page.keyboard.press('Control+KeyD');
    await frames(page, 1);
    const afterDup = await lightsOf(page);
    check('Ctrl+D duplicates with a new id', afterDup.length === afterUndo.length + 1 && new Set(afterDup.map((l) => l.id)).size === afterDup.length);
    await page.keyboard.press('Control+KeyZ');

    // ---- property edit + type change (new id) ----
    const retype = await page.evaluate((id) => {
      const e = window.__editor!.editor;
      e.select(id);
      e.updateSelected({ power: 321 });
      e.setType('disk');
      const nid = e.selected!;
      const l = e.selectedLight!;
      e.undoLast();
      return { nid, type: l.type, power: l.power, back: e.selected, backType: e.selectedLight?.type };
    }, rect.id);
    check('type change = new id (power carried over); undo restores the old id', retype.nid !== rect.id && retype.type === 'disk' && retype.power === 321 && retype.back === rect.id && retype.backType === 'rect', JSON.stringify(retype));

    // ---- animation: keys on the spot, orbit preset on the point, validation-clock playback ----
    const pointNow = (await lightsOf(page)).find((l) => l.id === pointId)!;
    const orbit = presetOrbit({ start: pointNow.matrix.slice(12, 15) as [number, number, number], center: [cx, cy, cz], period: 2, turns: 1, samplesPerTurn: 16 });
    const anim = await page.evaluate(([sid, pid, orb]) => {
      const e = window.__editor!.editor;
      e.anim.setSettings({ fps: 24, duration: 2, loop: true });
      e.player.setMode('validation');
      e.player.seekFrame(0);
      e.select(sid);
      e.keyTarget(`light:${sid}`);
      e.player.seekFrame(48);
      const m = Float32Array.from(e.selectedLight!.matrix);
      m[13] -= 0.1;
      e.updateSelected({ matrix: m, power: 500 });
      e.keyTarget(`light:${sid}`);
      e.applyPreset(`light:${pid}`, orb as never, 'orbit preset');
      e.keyTarget('camera');
      e.player.seekFrame(0);
      e.player.play();
      return { targets: e.anim.targets, keys: e.anim.keyTimes(`light:${sid}`) };
    }, [spot.id, pointId, orbit] as const);
    await frames(page, 30);
    const play = await page.evaluate(([sid, pid]) => {
      const e = window.__editor!.editor;
      e.player.pause();
      const t = e.player.time;
      const s = e.anim.evaluate(t, e.baseState());
      const cur = (id: number) => Array.from(e.store!.get(id)!.matrix);
      return { t, frame: e.player.frame, spot: cur(sid), spotExp: Array.from(s.lights.get(sid)!.matrix), pt: cur(pid), ptExp: Array.from(s.lights.get(pid)!.matrix), power: e.store!.get(sid)!.power, powerExp: s.lights.get(sid)!.power };
    }, [spot.id, pointId] as const);
    check('keys + orbit preset created', anim.targets.length >= 3 && anim.keys.length === 2, JSON.stringify(anim));
    check('validation playback: frame counter advanced, t = frame / fps', play.frame > 0 && play.t === play.frame / 24, `frame ${play.frame}, t ${play.t}`);
    check('lights follow the tracks exactly at the player time', maxAbs(play.spot, play.spotExp) === 0 && maxAbs(play.pt, play.ptExp) === 0 && play.power === play.powerExp, `spot power ${play.power.toFixed(3)}`);

    // ---- editor scene.json round trip ----
    const rt = await page.evaluate(() => { const h = window.__editor!; const a = h.saveState(); h.loadState(a); return { same: h.saveState() === a, bytes: a.length }; });
    check('editor scene.json save → load → save is identical', rt.same, `${rt.bytes} bytes`);

    await page.evaluate(() => window.__editor!.editor.player.seekFrame(24));
    await frames(page, 10);
    await page.screenshot({ path: path.join(OUT, 'editor.png') });

    // ---- Export for Cycles → /api/reference → compare view ----
    const blender = process.env.BLENDER ?? '/Applications/Blender.app/Contents/MacOS/Blender';
    if (!args['no-blender'] && existsSync(blender)) {
      await page.evaluate(() => { window.__editor!.editor.select(undefined); });
      const progress: string[] = [];
      await page.exposeFunction('__e2eProgress', (s: string) => { progress.push(s); console.log(`  [reference] ${s}`); });
      unlock(); // Blender takes the GPU lock itself; holding it here would deadlock (the app suspends its frame loop)
      const t0 = Date.now();
      const res = await page.evaluate(async () => {
        const h = window.__editor!;
        try {
          const r = await h.runReference!({ spp: 4, seeds: '0..1', maxBounces: 2, lightMode: 'A', frames: 'current' },
            (p) => { void (window as unknown as { __e2eProgress(s: string): void }).__e2eProgress(`${p.stage} ${p.done ?? ''}/${p.total ?? ''} ${p.message}`); });
          return { ok: true, r: r as { packageDir: string; refDir: string; frame: number; exrs: { url: string }[] }, ref: h.compare.ref.length, w: h.compare.ref[0]?.width, h: h.compare.ref[0]?.height, tw: window.__app!.targets.width, th: window.__app!.targets.height, mode: h.compare.settings.mode };
        } catch (e) { return { ok: false, err: String(e) }; }
      });
      await lock();
      const secs = ((Date.now() - t0) / 1000).toFixed(1);
      if (check('Export for Cycles + Blender reference render succeeded', res.ok === true, res.ok ? `${res.r!.refDir} in ${secs} s` : (res as { err: string }).err)) {
        const r = res as Required<typeof res> & { r: { packageDir: string; refDir: string; frame: number; exrs: { url: string }[] } };
        check('progress was streamed (per-EXR progress lines)', progress.some((p) => p.startsWith('render 1/2')) || progress.some((p) => /cache hit/.test(p)), progress.slice(-3).join(' | '));
        const sj = JSON.parse(readFileSync(path.join(ROOT, r.r.packageDir, 'scene.json'), 'utf8')) as { frames?: { frame: number; lights?: Record<string, unknown> }[]; lights: unknown[]; render: { width: number } };
        check('package carries the edited lights and the resolved frame (CONSTANT)', sj.lights.length >= 4 && sj.frames?.length === 1 && sj.frames[0].frame === 24 && !!sj.frames[0].lights?.[String(spot.id)], `frames ${JSON.stringify(sj.frames?.map((f) => f.frame))}, ${sj.lights.length} lights`);
        check('compare view auto-loaded both seeds at the internal resolution', r.ref === 2 && r.w === r.tw && r.h === r.th && r.mode === 'split', `${r.ref} seeds ${r.w}x${r.h} (internal ${r.tw}x${r.th}), mode ${r.mode}`);
        await frames(page, 30);
        for (const mode of ['split', 'relerr', 'tmap', 'flip'] as const) {
          await page.evaluate((m) => { const c = window.__editor!.compare; c.setMode(m); }, mode);
          await page.evaluate(() => window.__editor!.compare.refreshOurs());
          await sleep(300);
          await page.screenshot({ path: path.join(OUT, `compare-${mode}.png`) });
        }
        const summ = await page.evaluate(async () => { const c = window.__editor!.compare; await c.captureBatch(); return c.summaryText(); });
        check('compare statistics computed', /ours\/ref/.test(summ), summ.replace(/\n/g, ' | '));
      }
    } else console.log('SKIP  Blender reference (no Blender or --no-blender)');

    const errs = await page.evaluate(() => window.__webgpuErrors ?? []);
    check('no uncaptured WebGPU errors', errs.length === 0, errs.slice(0, 3).join(' | '));
    check('no page errors', consoleErrors.filter((e) => !/favicon/.test(e)).length === 0, consoleErrors.slice(0, 3).join(' | '));
  } finally {
    await browser.close();
    unlock();
    await vite.close();
  }
  const ok = checks.every((c) => c.ok);
  console.log(`${checks.filter((c) => c.ok).length}/${checks.length} checks passed`);
  console.log(ok ? 'RESULT: PASS' : 'RESULT: FAIL');
  return ok ? 0 : 1;
}

main().then((c) => process.exit(c), (e: unknown) => { console.error(e); unlock(); process.exit(1); });

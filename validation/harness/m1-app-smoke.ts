// M1 app smoke (plan §5 M1 exit: "Sponza (MASK foliage) flies at the internal resolution", "G-buffer and BVH views
// work"). Starts Vite, opens the real app in headless Chrome (no unsafe flag) with Sponza (Cornell if Sponza is
// missing) + an HDRI, renders ≥ 60 frames at 960×540, drives WASD/E/Q and an RMB drag, asserts no uncaptured
// WebGPU errors, NaN/Inf = 0 and BVH overflow/itercap = 0, saves view screenshots and times the primary pass
// (timestamp queries) at 960×540 and 1920×1080.
//   npx tsx validation/harness/m1-app-smoke.ts [--run <id>] [--no-lock]
import { existsSync, mkdirSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium, type Page } from 'playwright';
import { createServer } from 'vite';
import type { App } from '../../src/app/app.ts';
import type { Integration } from '../../src/app/integration.ts';
import { acquireGpuLock } from './gpu-lock.ts';

declare global { interface Window { __app?: App; __integration?: Integration; __webgpuErrors?: string[] } }

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const SPONZA = 'validation/assets/downloaded/sponza/Sponza.gltf';
const CORNELL = 'validation/assets/cornell/cornell.glb';
const HDRI = 'validation/assets/downloaded/hdri/kloofendal_48d_partly_cloudy_puresky_1k.hdr';
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');

const { values: args } = parseArgs({ options: { run: { type: 'string' }, 'no-lock': { type: 'boolean', default: false } } });

const checks: { name: string; ok: boolean; detail?: string }[] = [];
const warnings: string[] = [];
const check = (name: string, ok: boolean, detail?: string) => { checks.push({ name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`); return ok; };
const warn = (msg: string) => { warnings.push(msg); console.log(`WARN  ${msg}`); };

async function frames(page: Page, n: number, timeoutMs = 60_000): Promise<number> {
  const t0 = Date.now();
  const f0 = await page.evaluate(() => window.__app!.frameCounter);
  await page.waitForFunction((target) => window.__app!.frameCounter >= target, f0 + n, { timeout: timeoutMs, polling: 50 });
  return (Date.now() - t0) / n;
}

const camState = (page: Page) => page.evaluate(() => {
  const c = window.__app!.camera;
  return { p: [...c.position] as number[], yaw: c.yaw, pitch: c.pitch, speed: c.speed };
});

async function holdKey(page: Page, code: string, ms: number) {
  await page.keyboard.down(code);
  await sleep(ms);
  await page.keyboard.up(code);
  await frames(page, 2);
}

// Run totals (never reset by history resets, which resolution/env changes trigger) + frames whose counters were lost.
const totals = (page: Page) => page.evaluate(() => ({ ...window.__app!.runTotals, probeSkipped: window.__app!.probe.skipped, frames: window.__app!.frameCounter }));

async function main(): Promise<number> {
  const runId = args.run ?? `m1-smoke-${stamp()}`;
  if (!/^[A-Za-z0-9._-]+$/.test(runId)) throw new Error(`bad run id ${runId}`);
  const OUT = path.join(ROOT, 'validation/out', runId);
  mkdirSync(OUT, { recursive: true });
  const sponza = existsSync(path.join(ROOT, SPONZA));
  if (!sponza) warn(`${SPONZA} missing (re-fetch: validation/blender/fetch_sponza.py); using the Cornell box`);
  const hdri = existsSync(path.join(ROOT, HDRI));
  if (!hdri) warn(`${HDRI} missing (npx tsx validation/assets/fetch_hdris.ts); running without an environment`);
  const sceneUrl = `/${sponza ? SPONZA : CORNELL}`;

  const release = args['no-lock'] ? () => {} : await acquireGpuLock('m1-app-smoke');

  const vite = await createServer({ root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), server: { port: 0, host: '127.0.0.1' }, logLevel: 'warn' });
  await vite.listen();
  const port = (vite.httpServer!.address() as { port: number }).port;
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const consoleErrors: string[] = [];
  const report: Record<string, unknown> = { runId, scene: sceneUrl, env: hdri ? `/${HDRI}` : null, checks, warnings, chrome: browser.version() };
  try {
    const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
    const page = await ctx.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error') consoleErrors.push(m.text());
      if (m.type() === 'error' || m.type() === 'warning' || /^\[(scene|env)\]/.test(m.text())) console.log(`[page:${m.type()}] ${m.text()}`);
    });
    page.on('pageerror', (e) => consoleErrors.push(`[pageerror] ${e.message}`));
    const q = new URLSearchParams({ seed: '1', res: '540p', scene: sceneUrl, ...(hdri ? { env: `/${HDRI}` } : {}) });
    const t0 = Date.now();
    await page.goto(`http://127.0.0.1:${port}/index.html?${q}`);
    await page.waitForFunction((wantEnv) => {
      const r = window.__integration?.renderer();
      const a = window.__app;
      return !!r && !!a && !!a.scene && r.scene?.scene === a.scene && !r.loading && r.ready && (!wantEnv || r.env.present);
    }, hdri, { timeout: 180_000, polling: 100 });
    const loadMs = Date.now() - t0;
    const info = await page.evaluate(() => {
      const a = window.__app!, r = window.__integration!.renderer()!;
      const g = r.scene!;
      return {
        scene: a.scene!.name, tris: g.stats.triangles, vertices: g.stats.vertices, geometryBytes: g.stats.geometryBytes, bvh: g.bvh.stats, textureBytes: g.stats.textureBytes, env: [r.env.width, r.env.height],
        internal: [a.targets.width, a.targets.height], hud: r.hudLines(), ts: a.timestamps.supported, lastError: r.lastError,
        alphaTris: Array.from(a.scene!.geometry.triFlags).filter((f) => f & 1).length,
      };
    });
    report.load = { ms: loadMs, ...info };
    console.log(`loaded ${info.scene} in ${loadMs} ms: ${info.tris} tris, ${info.vertices} verts, geometry ${(info.geometryBytes / 1e6).toFixed(2)} MB, ${info.alphaTris} alpha-MASK tris, env ${info.env}, internal ${info.internal}`);
    check('internal resolution 960x540', info.internal[0] === 960 && info.internal[1] === 540, `${info.internal}`);
    check('renderer error-free', !info.lastError, info.lastError);
    if (sponza) check('Sponza MASK foliage present', info.alphaTris > 0, `${info.alphaTris} alpha-MASK tris`);

    // Camera inside the scene (Sponza: along the long X axis at ~1/5 height).
    await page.evaluate(() => {
      const a = window.__app!;
      const b = a.scene!.bounds, e = [0, 1, 2].map((i) => b.max[i] - b.min[i]), c = [0, 1, 2].map((i) => 0.5 * (b.min[i] + b.max[i]));
      const long = e[0] >= e[2] ? 0 : 2;
      const eye = [...c] as [number, number, number], tgt = [...c] as [number, number, number];
      eye[1] = tgt[1] = b.min[1] + 0.2 * e[1];
      eye[long] = c[long] + 0.35 * e[long]; tgt[long] = c[long] - 0.35 * e[long];
      a.camera.lookAt(eye, tgt);
      a.camera.initialPose = a.camera.pose();
    });
    const ms60 = await frames(page, 60);
    check('60 frames rendered', true, `${ms60.toFixed(1)} ms/frame wall clock (rAF-limited)`);
    await page.evaluate(() => { window.__app!.hud.setVisible(true); });
    await page.screenshot({ path: path.join(OUT, 'ui-beauty.png') });

    // ---- fly camera: WASD in the horizontal plane, E/Q along world ±Y, RMB drag rotates ----
    await page.locator('#view').focus();
    const moveCheck = async (code: string, want: (d: number[], s: { yaw: number }) => boolean, label: string) => {
      const a = await camState(page);
      await holdKey(page, code, 350);
      const b = await camState(page);
      const d = [0, 1, 2].map((i) => b.p[i] - a.p[i]);
      const len = Math.hypot(...d);
      const expect = a.speed * 0.35;
      check(`${code} ${label}`, want(d, a) && len > 0.3 * expect && len < 3 * expect && b.yaw === a.yaw,
        `Δ=[${d.map((x) => x.toFixed(3))}] |Δ|=${len.toFixed(3)} (~${expect.toFixed(3)} expected)`);
    };
    const fwd = (y: number) => [-Math.sin(y), 0, -Math.cos(y)];
    const rgt = (y: number) => [Math.cos(y), 0, -Math.sin(y)];
    const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const horiz = (d: number[]) => Math.abs(d[1]) < 1e-9;
    await moveCheck('KeyW', (d, s) => horiz(d) && dot(d, fwd(s.yaw)) > 0.99 * Math.hypot(...d), 'moves forward (horizontal)');
    await moveCheck('KeyS', (d, s) => horiz(d) && dot(d, fwd(s.yaw)) < -0.99 * Math.hypot(...d), 'moves back (horizontal)');
    await moveCheck('KeyA', (d, s) => horiz(d) && dot(d, rgt(s.yaw)) < -0.99 * Math.hypot(...d), 'strafes left');
    await moveCheck('KeyD', (d, s) => horiz(d) && dot(d, rgt(s.yaw)) > 0.99 * Math.hypot(...d), 'strafes right');
    await moveCheck('KeyE', (d) => d[1] > 0 && Math.hypot(d[0], d[2]) < 1e-9, 'moves up world +Y');
    await moveCheck('KeyQ', (d) => d[1] < 0 && Math.hypot(d[0], d[2]) < 1e-9, 'moves down world -Y');
    {
      const a = await camState(page);
      await page.mouse.move(960, 540);
      await page.mouse.down({ button: 'right' });
      for (let i = 1; i <= 10; i++) { await page.mouse.move(960 + 20 * i, 540 - 5 * i); await sleep(16); }
      await page.mouse.up({ button: 'right' });
      await frames(page, 2);
      const b = await camState(page);
      const dyaw = b.yaw - a.yaw, dpitch = b.pitch - a.pitch;
      check('RMB drag rotates (yaw right, pitch up), position fixed', dyaw < -0.1 && dpitch > 0.02 && b.p.every((x, i) => x === a.p[i]),
        `Δyaw ${(dyaw * 180 / Math.PI).toFixed(1)}°, Δpitch ${(dpitch * 180 / Math.PI).toFixed(1)}°`);
    }
    await page.keyboard.press('Home');
    await frames(page, 30);
    let t = await totals(page);
    check('NaN/Inf = 0 while flying', t.nan === 0 && t.inf === 0 && t.framesRead > 0, JSON.stringify(t));
    check('BVH overflow/itercap = 0 while flying', t.bvhOverflow === 0 && t.bvhItercap === 0, JSON.stringify(t));

    // ---- debug views ----
    await page.evaluate(() => { const a = window.__app!; a.hud.setVisible(false); (document.querySelector('.panel') as HTMLElement).style.display = 'none'; });
    const views: [string, number][] = [['beauty', 0], ['albedo', 100], ['normals-ns', 101], ['normals-ng', 102], ['depth', 103], ['primid', 104], ['motion', 110], ['bvh-steps', 200], ['bvh-tris', 202]];
    for (const [name, id] of views) {
      await page.evaluate((v) => window.__app!.selectDebugView(v), id);
      if (id >= 200) await page.evaluate(() => window.__integration!.renderer()!.warmup(true));
      if (name === 'motion') await page.keyboard.down('KeyD');
      await frames(page, 8);
      await page.screenshot({ path: path.join(OUT, `${name}.png`) });
      if (name === 'motion') await page.keyboard.up('KeyD');
    }
    // env grid + background mask: look up through the atrium / out of the box
    await page.evaluate(() => {
      const a = window.__app!;
      const b = a.scene!.bounds, c = [0, 1, 2].map((i) => 0.5 * (b.min[i] + b.max[i])) as [number, number, number];
      a.camera.lookAt(c, [c[0] + 0.3, c[1] + 1, c[2] + 0.2]);
      a.camera.yfov = 90 * Math.PI / 180;
    });
    for (const [name, id] of [['env-grid', 300], ['env-bgmask', 301], ['env-beauty', 0]] as const) {
      await page.evaluate((v) => window.__app!.selectDebugView(v), id);
      await frames(page, 8);
      await page.screenshot({ path: path.join(OUT, `${name}.png`) });
    }
    const bg = await page.evaluate(async () => {
      // background fraction from the V-buffer through the probe of a few pixels is overkill: use the G-buffer
      const r = window.__integration!.renderer()!;
      const a = window.__app!;
      const dev = a.device, n = a.targets.width * a.targets.height;
      const st = dev.createBuffer({ size: n * 80, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = dev.createCommandEncoder();
      e.copyBufferToBuffer(r.gbuffer!, 0, st, 0, n * 80);
      dev.queue.submit([e.finish()]);
      await st.mapAsync(GPUMapMode.READ);
      const u = new Uint32Array(st.getMappedRange());
      let miss = 0;
      for (let i = 0; i < n; i++) if ((u[i * 20 + 15] & 1) === 0) miss++;
      st.destroy();
      return miss / n;
    });
    check('env view sees background pixels', !hdri || bg > 0.01, `${(bg * 100).toFixed(1)}% background`);
    await page.evaluate(() => { const a = window.__app!; a.selectDebugView(0); a.camera.yfov = 40 * Math.PI / 180; a.camera.reset(); });
    await page.screenshot({ path: path.join(OUT, 'beauty-after-views.png') });

    // ---- env panel controls are live (rotation / strength / visibility) ----
    if (hdri) {
      const env = await page.evaluate(async () => {
        const a = window.__app!, r = window.__integration!.renderer()!;
        a.envParams.rotationDeg = 30; a.envParams.strength = 2; a.envParamsChanged();
        const p1 = { ...r.env.params };
        a.envParams.rotationDeg = 0; a.envParams.strength = 1; a.envParamsChanged();
        return p1;
      });
      check('env params reach the GPU record', Math.abs(env.rotationZ - Math.PI / 6) < 1e-9 && env.strength === 2, JSON.stringify(env));
    }

    // ---- primary pass timing (timestamp queries) at 960x540 and 1920x1080 ----
    const timing: Record<string, unknown> = {};
    const measure = async (label: string) => {
      await frames(page, 90);
      const m = await page.evaluate(async () => {
        const a = window.__app!, r = window.__integration!.renderer()!;
        const avg = a.timestamps.averages();
        // Back-to-back dispatches in one submit: 2 warm-up rounds, then the median of 32.
        await r.timePrimary(a.debug.bindGroup, 16);
        await r.timePrimary(a.debug.bindGroup, 16);
        const iso = (await r.timePrimary(a.debug.bindGroup, 32)) ?? [];
        const sorted = [...iso].sort((x, y) => x - y);
        return {
          res: [a.targets.width, a.targets.height], hudPrimary: avg.find((p) => p.name === 'primary'), hudTotal: avg.find((p) => p.name === 'total'),
          isolatedMedianMs: sorted[sorted.length >> 1], isolatedMinMs: sorted[0], isolatedMaxMs: sorted[sorted.length - 1],
        };
      });
      timing[label] = m;
      const px = m.res[0] * m.res[1];
      console.log(`TIMING primary ${m.res[0]}x${m.res[1]}: isolated median ${m.isolatedMedianMs.toFixed(3)} ms (min ${m.isolatedMinMs.toFixed(3)}, ` +
        `${(px / m.isolatedMedianMs / 1e3).toFixed(0)} Mrays/s); in-app 32-frame avg ${m.hudPrimary?.ms.toFixed(3)} ms, gpu frame total ${m.hudTotal?.ms.toFixed(3)} ms`);
      return m;
    };
    if (info.ts) {
      await page.evaluate(() => window.__app!.camera.reset());
      const m540 = await measure('960x540');
      await page.evaluate(() => window.__app!.setResolution('1080p'));
      const m1080 = await measure('1920x1080');
      check('primary pass timed at 960x540 and 1920x1080', m540.isolatedMedianMs > 0 && m1080.isolatedMedianMs > 0 && m1080.res[1] === 1080,
        `${m540.isolatedMedianMs.toFixed(2)} ms @960x540, ${m1080.isolatedMedianMs.toFixed(2)} ms @1920x1080`);
      await page.evaluate(() => window.__app!.setResolution('540p'));
      await frames(page, 10);
    } else warn('timestamp-query unavailable: no primary timings');
    report.timing = timing;

    t = await totals(page);
    const gpuErrors = await page.evaluate(() => window.__webgpuErrors ?? []);
    check('no uncaptured WebGPU errors', gpuErrors.length === 0, gpuErrors.slice(0, 3).join(' | '));
    check('NaN/Inf counters 0 (final, whole run)', t.nan === 0 && t.inf === 0, JSON.stringify(t));
    check('every frame\'s counters were read back (probe ring never skipped)', t.probeSkipped === 0 && t.framesRead > 0, `${t.framesRead} read, ${t.probeSkipped} skipped, ${t.frames} frames`);
    check('BVH overflow/itercap 0 (final)', t.bvhOverflow === 0 && t.bvhItercap === 0, JSON.stringify(t));
    check('no console errors / page errors', consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' | '));
    report.totals = t;
  } catch (e) {
    check('smoke ran to completion', false, e instanceof Error ? e.message : String(e));
  } finally {
    await browser.close();
    await vite.close();
    release();
  }
  report.consoleErrors = consoleErrors;
  await writeFile(path.join(OUT, 'report.json'), JSON.stringify(report, null, 2));
  const failed = checks.filter((c) => !c.ok).length;
  console.log(`\n${checks.length - failed}/${checks.length} checks passed, ${warnings.length} warning(s); screenshots + report in ${path.relative(ROOT, OUT)}`);
  return failed ? 1 : 0;
}

main().then((c) => process.exit(c), (e) => { console.error(e); process.exit(1); });

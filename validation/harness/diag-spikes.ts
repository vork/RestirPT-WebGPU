// SCRATCH (firefly diagnosis, not for merge): per-frame spike tracker on the app's ReSTIR per-frame radiance (rs-frame,
// L1 + estimate) against a converged PT luminance reference (validation/out/<refRun>/pt_ref.f32).
// Per run (one config, one debug view, N frames, deterministic seed): ratio histogram (log2 L_t/L_ref, all pixel-frames),
// per-pixel run lengths above K_run, per-pixel sum / sum² / lag-1 / 32-frame block means (temporal autocorrelation),
// and an event list (frame, pixel, L_t, L_ref, raw debug AOV words) for pixel-frames above K_ev.
//   npx tsx validation/harness/diag-spikes.ts --run diag-sp --ref diag-ff --configs temporal_only --views 0,492,486 --frames 400
import { mkdirSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium, type Page } from 'playwright';
import { createServer } from 'vite';
import type { App } from '../../src/app/app.ts';
import type { Integration } from '../../src/app/integration.ts';
import type { RestirSettings } from '../../src/core/render/restir/presets.ts';
import { acquireGpuLock } from './gpu-lock.ts';

declare global { interface Window { __app?: App; __integration?: Integration; __trk?: any } }

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const { values: args } = parseArgs({
  options: {
    run: { type: 'string', default: 'diag-sp' }, ref: { type: 'string', default: 'diag-ff' }, configs: { type: 'string', default: 'temporal_only' },
    views: { type: 'string', default: '0' }, frames: { type: 'string', default: '400' }, seed: { type: 'string', default: '1' },
    kev: { type: 'string', default: '8' }, krun: { type: 'string', default: '4' }, tap: { type: 'string', default: '2' },
    scene: { type: 'string', default: '/validation/assets/cornell/cornell.usda' },
  },
});

interface Cfg { restirMode: 'interactive' | 'unbiased'; temporal: boolean; over?: Partial<RestirSettings> }
const I = { restirMode: 'interactive', temporal: true } as const;
const CFG: Record<string, Cfg> = {
  interactive: { ...I },
  interactive_noRR: { ...I, over: { rr: false } },
  interactive_noBoost: { ...I, over: { boostSlots: 0 } },
  interactive_2022: { ...I, over: { criteria: '2022' } },
  interactive_cap5: { ...I, over: { cCap: 5 } },
  interactive_cap1: { ...I, over: { cCap: 1 } },
  interactive_talbot: { ...I, over: { temporalMis: 'talbot' } },
  interactive_tau1e3: { ...I, over: { tau: 1e-3 } },
  interactive_tau5e3: { ...I, over: { tau: 5e-3 } },
  spatial_only: { ...I, temporal: false },
  temporal_only: { ...I, over: { rounds: 0 } },
  temporal_only_noRR: { ...I, over: { rounds: 0, rr: false } },
  temporal_only_cap5: { ...I, over: { rounds: 0, cCap: 5 } },
  temporal_only_talbot: { ...I, over: { rounds: 0, temporalMis: 'talbot' } },
  temporal_only_tau1e3: { ...I, over: { rounds: 0, tau: 1e-3 } },
  temporal_only_tau5e3: { ...I, over: { rounds: 0, tau: 5e-3 } },
  temporal_only_2022: { ...I, over: { rounds: 0, criteria: '2022' } },
  initial_only: { ...I, temporal: false, over: { rounds: 0 } },
  unbiased: { restirMode: 'unbiased', temporal: true },
};

const WGSL = /* wgsl */`
struct P { w: u32, h: u32, frame: u32, maxEv: u32, kev: f32, krun: f32, aovBase: u32, blockLen: u32 }
@group(0) @binding(0) var ft: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> refL: array<f32>;
@group(0) @binding(2) var<storage, read> dbgb: array<vec4u>;
@group(0) @binding(3) var<storage, read_write> cnt: array<atomic<u32>, 80>;
@group(0) @binding(4) var<storage, read_write> ev: array<vec4u>;
@group(0) @binding(5) var<storage, read_write> st: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.w || g.y >= p.h) { return; }
  let i = g.y * p.w + g.x;
  let c = textureLoad(ft, vec2i(g.xy), 0).rgb;
  let L = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  let Lr = refL[i];
  if (Lr <= 1e-3) { return; }
  let r = L / Lr;
  // histogram: bin = clamp(floor(2*log2 r) + 32, 0, 63); r == 0 -> bin 64
  var bin = 64u;
  if (r > 0.0) { bin = u32(clamp(floor(2.0 * log2(r)) + 32.0, 0.0, 63.0)); }
  atomicAdd(&cnt[8u + bin], 1u);
  let b = i * 12u;
  // runs above krun
  let above = r > p.krun;
  if (above) {
    if (st[b] == 0.0) { st[b + 3u] += 1.0; }
    st[b + 1u] += 1.0; st[b + 2u] = max(st[b + 2u], st[b + 1u]);
  } else { st[b + 1u] = 0.0; }
  st[b] = select(0.0, 1.0, above);
  st[b + 4u] += L; st[b + 5u] += L * L; st[b + 6u] += L * st[b + 7u]; st[b + 7u] = L;
  st[b + 8u] += L;
  if ((p.frame + 1u) % p.blockLen == 0u) { let m = st[b + 8u] / f32(p.blockLen); st[b + 9u] += m; st[b + 10u] += m * m; st[b + 11u] += 1.0; st[b + 8u] = 0.0; }
  if (r > p.kev) {
    let k = atomicAdd(&cnt[0], 1u);
    if (k < p.maxEv) {
      ev[2u * k] = vec4u(p.frame, g.x | (g.y << 16u), bitcast<u32>(L), bitcast<u32>(Lr));
      ev[2u * k + 1u] = dbgb[p.aovBase + i];
    }
  }
}`;

async function waitFrameIndex(page: Page, n: number): Promise<void> {
  await page.waitForFunction((t) => window.__trk.processed >= t, n, { timeout: 900_000, polling: 100 });
}

async function main(): Promise<void> {
  const OUT = path.join(ROOT, 'validation/out', args.run!);
  mkdirSync(OUT, { recursive: true });
  const refRun = path.join(ROOT, 'validation/out', args.ref!);
  const refMeta = JSON.parse(readFileSync(path.join(refRun, 'pt_ref.json'), 'utf8')) as { w: number; h: number };
  const refRgb = new Float32Array(readFileSync(path.join(refRun, 'pt_ref.f32')).buffer.slice(0));
  const refLum = new Float32Array(refMeta.w * refMeta.h);
  for (let i = 0; i < refLum.length; i++) refLum[i] = 0.2126 * refRgb[3 * i] + 0.7152 * refRgb[3 * i + 1] + 0.0722 * refRgb[3 * i + 2];
  const refB64 = Buffer.from(refLum.buffer).toString('base64');

  const vite = await createServer({
    root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), logLevel: 'warn',
    server: { port: 0, host: '127.0.0.1', hmr: false, watch: null, fs: { allow: [ROOT, realpathSync(path.join(ROOT, 'node_modules'))] } },
  });
  await vite.listen();
  const port = (vite.httpServer!.address() as { port: number }).port;
  const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--disable-gpu-vsync', '--disable-frame-rate-limit'] });
  const release = await acquireGpuLock('diag-spikes');
  const t0 = Date.now();
  try {
    const cfgs = args.configs!.split(',');
    const views = args.views!.split(',').map(Number);
    const N = Number(args.frames);
    for (const name of cfgs) {
      const c = CFG[name];
      if (!c) throw new Error(`unknown config ${name}`);
      for (const view of views) {
        // fresh page per run: identical state ⇒ deterministic replay across views
        const ctx = await browser.newContext({ viewport: { width: 1056, height: 810 } });
        const page = await ctx.newPage();
        await page.addInitScript({ content: 'window.__name = (f) => f;' });
        page.on('console', (m) => { if (m.type() === 'error') console.log(`[page:${m.type()}] ${m.text()}`); });
        page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
        const q = new URLSearchParams({ seed: args.seed!, res: '540p', scene: args.scene! });
        await page.goto(`http://127.0.0.1:${port}/index.html?${q}`);
        await page.waitForFunction(() => {
          const r = window.__integration?.renderer(); const a = window.__app;
          return !!r && !!a?.scene && r.scene?.scene === a.scene && !r.loading && r.ready;
        }, undefined, { timeout: 180_000, polling: 100 });
        await page.evaluate(async ({ c, view, tap }) => {
          const r = window.__integration!.renderer()!;
          const app = window.__app!;
          app.setPaused(true);
          r.options.renderMode = 'restir';
          await r.setOptions({ restirMode: c.restirMode, temporal: c.temporal });
          const pass = await r.prepareRestir();
          if (c.over) { pass!.setSettings(c.over); await pass!.prepare(); }
          if (view) { app.selectDebugView(view); app.debugSettings.tap = tap; }
        }, { c, view, tap: Number(args.tap) });
        // let pipelines settle (paused frames), then reset and install the tracker
        const f0 = await page.evaluate(() => window.__app!.frameCounter);
        await page.waitForFunction((t) => window.__app!.frameCounter >= t, f0 + 30, { timeout: 180_000, polling: 50 });
        await page.evaluate(({ wgsl, refB64, kev, krun, N }) => {
          const app = window.__app!;
          const r = window.__integration!.renderer()!;
          const device = app.device;
          const w = app.targets.width, h = app.targets.height;
          const bin = atob(refB64); const u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
          const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
          const refBuf = device.createBuffer({ size: u8.byteLength, usage: S }); device.queue.writeBuffer(refBuf, 0, u8);
          const maxEv = 1 << 20;
          const cnt = device.createBuffer({ size: 80 * 4, usage: S });
          const ev = device.createBuffer({ size: maxEv * 32, usage: S });
          const st = device.createBuffer({ size: w * h * 12 * 4, usage: S });
          const ub = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          const mod = device.createShaderModule({ code: wgsl });
          const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
          const trk = { processed: 0, lastIdx: app.frameIndex, N, cnt, ev, st, w, h, maxEv };
          window.__trk = trk;
          app.beforeFrame.add(() => {
            // frameIndex = number of accumulated frames; frame (frameIndex − 1) is in rs-frame / debug buffer now
            const fi = app.frameIndex;
            if (fi === trk.lastIdx || fi === 0 || trk.processed >= trk.N) return;
            trk.lastIdx = fi;
            const k = r.restir!.kernel;
            const ab = new ArrayBuffer(32); const du = new Uint32Array(ab); const df = new Float32Array(ab);
            du[0] = w; du[1] = h; du[2] = fi - 1; du[3] = maxEv; df[4] = kev; df[5] = krun; du[6] = (64 + 256 * 32) / 16; du[7] = 32;
            device.queue.writeBuffer(ub, 0, ab);
            const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
              { binding: 0, resource: k.resources.frameTex.createView() }, { binding: 1, resource: { buffer: refBuf } },
              { binding: 2, resource: { buffer: app.debug.buffer } }, { binding: 3, resource: { buffer: cnt } }, { binding: 4, resource: { buffer: ev } },
              { binding: 5, resource: { buffer: st } }, { binding: 6, resource: { buffer: ub } }] });
            const e = device.createCommandEncoder(); const cp = e.beginComputePass(); cp.setPipeline(pipe); cp.setBindGroup(0, bg);
            cp.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8)); cp.end(); device.queue.submit([e.finish()]);
            trk.processed++;
            if (trk.processed >= trk.N) app.setPaused(true);
          });
          app.resetTemporalHistory();
          app.seedIndex = 0;
          app.setPaused(false);
        }, { wgsl: WGSL, refB64, kev: Number(args.kev), krun: Number(args.krun), N });
        const ts = Date.now();
        await waitFrameIndex(page, N);
        const res = await page.evaluate(async () => {
          const app = window.__app!; const t = window.__trk; const device = app.device;
          await device.queue.onSubmittedWorkDone();
          const read = async (src: GPUBuffer, bytes: number) => {
            const b = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
            const e = device.createCommandEncoder(); e.copyBufferToBuffer(src, 0, b, 0, bytes); device.queue.submit([e.finish()]);
            await b.mapAsync(GPUMapMode.READ); const out = new Uint8Array(b.getMappedRange().slice(0)); b.unmap(); b.destroy(); return out;
          };
          const cnt = new Uint32Array((await read(t.cnt, 320)).buffer);
          const nEv = Math.min(cnt[0], t.maxEv);
          const ev = nEv ? await read(t.ev, nEv * 32) : new Uint8Array(0);
          const st = await read(t.st, t.w * t.h * 48);
          const b64 = (u8: Uint8Array) => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode(...u8.subarray(i, i + 0x8000)); return btoa(s); };
          const r = window.__integration!.renderer()!;
          t.cnt.destroy(); t.ev.destroy(); t.st.destroy();
          app.setPaused(true);
          return { cnt: Array.from(cnt), nEv, ev: b64(ev), st: b64(st), w: t.w, h: t.h, hud: r.hudLines(), settings: r.restir?.settings, runSeed: app.runSeed };
        });
        const tag = `${name}_v${view}`;
        writeFileSync(path.join(OUT, `${tag}.ev`), Buffer.from(res.ev, 'base64'));
        writeFileSync(path.join(OUT, `${tag}.st`), Buffer.from(res.st, 'base64'));
        writeFileSync(path.join(OUT, `${tag}.json`), JSON.stringify({ name, view, tap: Number(args.tap), frames: N, secs: (Date.now() - ts) / 1000, kev: Number(args.kev), krun: Number(args.krun),
          cnt: res.cnt, nEv: res.nEv, w: res.w, h: res.h, hud: res.hud, settings: res.settings, runSeed: res.runSeed }, null, 1));
        console.log(`[spikes] ${tag}: ${N} frames in ${((Date.now() - ts) / 1000).toFixed(1)} s, events ${res.cnt[0]}  (lock held ${((Date.now() - t0) / 60000).toFixed(1)} min)`);
        await page.goto('about:blank');
        await page.close();
        await ctx.close();
      }
    }
  } finally {
    release();
    await browser.close().catch(() => undefined);
    await vite.close();
  }
}
main().then(() => process.exit(0), (e: unknown) => { console.error(e); process.exit(1); });

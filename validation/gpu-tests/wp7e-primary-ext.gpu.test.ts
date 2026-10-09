// perf2 WP-7e (RS_PRIMARY_EXT; docs/decisions/perf2-plan.md §2 WP-7, perf2-api.md §1): rs_primary_ext reads the
// primary hit from the renderer's M1 V-buffer instead of tracing the camera ray again.
//   U-WP7E-VBUF  the interactive Renderer (ReSTIR-interactive, temporal, denoiser on, the app's options) with the release
//                set + RS_PRIMARY_EXT on all-lights, nm_smooth, glass, alpha and Sponza-lite, 16 frames with camera and
//                light motion: per frame the ReSTIR V-buffer (rsVbuf of the frame's parity) equals the M1 V-buffer exactly
//                in (primId, bits(u), bits(v)); rsVbuf.w = ‖x₁ − x₀‖ is finite on hits; the frame renders finite with
//                accepted > 0. Diagnostic: the same frames without the flag (rs_primary traces its own ray) give the
//                primId mismatches and max |Δbary| of the two separately compiled intersections — the whole results
//                change of the flag (ulp edge ties).
//   Q3           the M1 primary pass carries timestampWrites (as in the app) and rs_primary_ext depends on its V-buffer
//                texture: no ReSTIR compute pass is encoded with timestampWrites and the frame's ReSTIR work is not lost
//                (accepted > 0, q0 capacity set, every frame).
//   U-WP7E-VTRACE  a validation kernel with the flag forced (no M1 pass: rs_vtrace → rs_primary_ext) vs the flag-free
//                kernel on the same frames: rsVbuf, rsGeo, rsL1 compared texel by texel (diagnostic; the Shipped tier of
//                m8-bits records the end-to-end hashes).
import { afterAll, describe, expect, it } from 'vitest';
import { releaseTestGpu, getTestGpu } from './device-factory.ts';
import { DebugResources, DebugViewRegistry, defaultDebugSettings } from '../../src/core/render/debug-views.ts';
import { FRAME_RESET_HISTORY, FrameUniformBuffer, JITTER_IID, boundsDiagonal, computeRenderOrigin, type CameraState } from '../../src/core/render/frame-uniforms.ts';
import { Renderer } from '../../src/core/render/renderer.ts';
import { RELEASE_PERF_FLAGS, normalizePerfFlags, type PerfFlagsInput } from '../../src/core/render/restir/perf-flags.ts';
import { numSlotsOf, restirSettings, INTERACTIVE_PINNED } from '../../src/core/render/restir/presets.ts';
import type { LightData } from '../../src/core/scene/types.ts';
import { m8BitsScene, type M8BitsScene } from './m8-bits.ts';
import { restirRig } from './restir-fixtures.ts';

afterAll(async () => { await releaseTestGpu(); });

const EXT = normalizePerfFlags({ ...RELEASE_PERF_FLAGS, RS_PRIMARY_EXT: 1 });
const BASE = normalizePerfFlags({ ...RELEASE_PERF_FLAGS, RS_PRIMARY_EXT: 0 });

async function readTex(device: GPUDevice, t: GPUTexture): Promise<Uint32Array> {
  const bpr = Math.ceil((t.width * 16) / 256) * 256;
  const b = device.createBuffer({ size: bpr * t.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const e = device.createCommandEncoder();
  e.copyTextureToBuffer({ texture: t }, { buffer: b, bytesPerRow: bpr }, [t.width, t.height]);
  device.queue.submit([e.finish()]);
  await b.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(b.getMappedRange());
  const out = new Uint8Array(t.width * t.height * 16);
  for (let y = 0; y < t.height; y++) out.set(src.subarray(y * bpr, y * bpr + t.width * 16), y * t.width * 16);
  b.unmap(); b.destroy();
  return new Uint32Array(out.buffer);
}

function yawed(cam: { camToWorld: number[]; yfov: number }, a: number): CameraState {
  const m = cam.camToWorld.slice();
  const c = Math.cos(a), s = Math.sin(a);
  for (const col of [0, 4, 8]) { const x = m[col], z = m[col + 2]; m[col] = c * x + s * z; m[col + 2] = -s * x + c * z; }
  return { camToWorld: new Float32Array(m), yfov: cam.yfov, znear: 1e-4 } as unknown as CameraState;
}

interface VbufStats { exactMismatch: number; primMismatch: number; maxBary: number; badW: number; accepted: number[]; tsReStir: number; frames: number }

async function runRenderer(sceneName: M8BitsScene, flags: PerfFlagsInput, W: number, H: number, frames: number): Promise<VbufStats> {
  const ctx = await getTestGpu();
  const { device } = ctx;
  const { scene, cam: cam0 } = await m8BitsScene(sceneName);
  const debug = new DebugResources(device, new DebugViewRegistry());
  await debug.init();
  debug.resize(W, H);
  const glass = sceneName === 'glass';
  const r = await Renderer.create({ device, debugLayout: debug.layout, debug, features: ctx.features, wgslLanguageFeatures: ctx.wgslLanguageFeatures }, {
    watertight: false, renderMode: 'restir', restirMode: 'interactive', temporal: true, accumulate: true, lightMode: 'B', bvhKind: 'auto',
    maxBounces: glass ? 4 : 3, restirKernel: { perfFlags: flags },
  });
  const origin = computeRenderOrigin(scene.bounds, scene.quant);
  const fu = new FrameUniformBuffer(device);
  const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
  const color = device.createTexture({ label: 'wp7e-colour', size: [W, H], format: 'rgba16float', usage });
  const depth = device.createTexture({ label: 'wp7e-depth', size: [W, H], format: 'r32float', usage });
  const qs = device.features.has('timestamp-query') ? device.createQuerySet({ type: 'timestamp', count: 2 }) : undefined;
  const st: VbufStats = { exactMismatch: 0, primMismatch: 0, maxBary: 0, badW: 0, accepted: [], tsReStir: 0, frames: 0 };
  try {
    if (scene.env) await r.setEnvironment(scene.env);
    expect(await r.setScene(scene, origin)).toBeTruthy();
    r.resize({ width: W, height: H, color, colorFormat: 'rgba16float', depth, frameUniforms: fu.buffer });
    r.setDenoise(true);
    expect(await r.prepareRestir()).toBeTruthy();
    expect(await r.prepareDenoiser()).toBeTruthy();
    await r.warmup(false);
    const k = r.restir!.kernel;
    expect(k.primaryPasses()).toEqual(normalizePerfFlags(flags).RS_PRIMARY_EXT ? ['rs_primary_ext'] : ['rs_primary']);
    const sceneDiag = boundsDiagonal(scene.bounds);
    let lights = scene.lights;
    let prev: CameraState | undefined;
    const NS = numSlotsOf(k.settings);
    for (let f = 0; f < frames; f++) {
      const cam = yawed(cam0, f >= 2 ? 0.004 * (f - 1) : 0);
      if (f >= 2) lights = lights.map((l, i) => (i === 0 ? { ...l, matrix: Object.assign(new Float32Array(l.matrix), { 12: l.matrix[12] + 0.02 }) } : l)) as LightData[];
      r.setLights(lights);
      fu.write({
        camera: cam, prevCamera: prev ?? cam, width: W, height: H, frameIndex: f, seedIndex: f, runSeed: 7, flags: f === 0 ? FRAME_RESET_HISTORY : 0,
        jitterMode: JITTER_IID, jitter: [0.5, 0.5], origin, exposure: 1, time: f / 24, dt: 1 / 24, sceneDiag,
      });
      prev = cam;
      debug.update({ ...defaultDebugSettings(), mode: 0 }, f);
      const enc = device.createCommandEncoder({ label: `wp7e-${f}` });
      const bcp = enc.beginComputePass.bind(enc);
      enc.beginComputePass = (d?: GPUComputePassDescriptor) => { if (d?.timestampWrites && d.label !== 'primary') st.tsReStir++; return bcp(d); };
      debug.beginFrame(enc);
      enc.clearBuffer(k.resources.arena, 0, 256);
      expect(r.encode(enc, { advanced: true, debugMode: 0, debugGroup: debug.bindGroup, resetTemporal: f === 0, resetHistory: f === 0 },
        () => (qs ? { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined)), `frame ${f}`).toBe(true);
      device.queue.submit([enc.finish()]);
      const m1 = await readTex(device, r.vbuffer!), rs = await readTex(device, k.resources.vbuf);
      const fm = new Float32Array(m1.buffer), fr = new Float32Array(rs.buffer);
      for (let i = 0; i < W * H; i++) {
        if (m1[4 * i] !== rs[4 * i] || (m1[4 * i] !== 0xFFFFFFFF && (m1[4 * i + 1] !== rs[4 * i + 1] || m1[4 * i + 2] !== rs[4 * i + 2]))) st.exactMismatch++;
        if (m1[4 * i] !== rs[4 * i]) { st.primMismatch++; continue; }
        if (m1[4 * i] === 0xFFFFFFFF) continue;
        st.maxBary = Math.max(st.maxBary, Math.abs(fm[4 * i + 1] - fr[4 * i + 1]), Math.abs(fm[4 * i + 2] - fr[4 * i + 2]));
        if (!(Number.isFinite(fr[4 * i + 3]) && fr[4 * i + 3] > 0)) st.badW++;
      }
      const c = await k.readCounters(true);
      st.accepted.push(c.rsc.accepted);
      if (c.queues[0].capacity !== W * H * NS) st.accepted[st.accepted.length - 1] = -1;
      st.frames++;
    }
    const img = new Uint16Array(await readTex2(device, color));
    let bad = 0; for (const h of img) if ((h & 0x7c00) === 0x7c00) bad++;
    expect(bad, 'non-finite colour texels').toBe(0);
  } finally {
    r.destroy?.(); fu.destroy(); color.destroy(); depth.destroy(); qs?.destroy();
  }
  return st;
}

async function readTex2(device: GPUDevice, t: GPUTexture): Promise<ArrayBuffer> {
  const bpr = Math.ceil((t.width * 8) / 256) * 256;
  const b = device.createBuffer({ size: bpr * t.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const e = device.createCommandEncoder();
  e.copyTextureToBuffer({ texture: t }, { buffer: b, bytesPerRow: bpr }, [t.width, t.height]);
  device.queue.submit([e.finish()]);
  await b.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(b.getMappedRange());
  const out = new Uint8Array(t.width * t.height * 8);
  for (let y = 0; y < t.height; y++) out.set(src.subarray(y * bpr, y * bpr + t.width * 8), y * t.width * 8);
  b.unmap(); b.destroy();
  return out.buffer;
}

const SCENES: [M8BitsScene, number, number][] = [['all', 96, 64], ['nm_smooth', 96, 96], ['glass', 96, 96], ['alpha', 96, 96], ['sponza_lite', 320, 180]];

describe('U-WP7E-VBUF: with RS_PRIMARY_EXT the ReSTIR V-buffer is the M1 V-buffer (16 frames, camera + light motion); Q3', () => {
  for (const [s, W, H] of SCENES) {
    it(s, async () => {
      const ext = await runRenderer(s, EXT, W, H, 16);
      const base = await runRenderer(s, BASE, W, H, 16);
      console.log(`[U-WP7E-VBUF] ${s} ${W}x${H}: ext ${JSON.stringify({ ...ext, accepted: Math.min(...ext.accepted) })}; `
        + `flag-free M1 vs rs_primary: primId mismatches ${base.primMismatch} / ${base.frames * W * H} texels, texel (prim,u,v) mismatches ${base.exactMismatch}, max |Δbary| ${base.maxBary.toExponential(3)}`);
      expect(ext.exactMismatch, 'rsVbuf (prim, u, v) = M1 V-buffer').toBe(0);
      expect(ext.badW, 'rsVbuf.w finite > 0 on hits').toBe(0);
      expect(ext.tsReStir, 'ReSTIR passes with timestampWrites (Q3)').toBe(0);
      for (const r of [ext, base]) expect(Math.min(...r.accepted), 'accepted > 0 and q0 capacity set every frame').toBeGreaterThan(0);
      // the flag-free run reproduces the documented tolerance of the two intersection pipelines (denoise-run.ts)
      expect(base.primMismatch).toBeLessThanOrEqual(1e-4 * base.frames * W * H + 2);
      expect(base.maxBary).toBeLessThan(1e-2);
    }, 600_000);
  }
});

describe('U-WP7E-VTRACE: validation kernel with RS_PRIMARY_EXT forced (rs_vtrace → rs_primary_ext) vs rs_primary', () => {
  for (const [s, preset] of [['all', 'full-m6'], ['alpha', 'full-m6'], ['nm_smooth', 'interactive']] as const) {
    it(`${s} ${preset}`, async () => {
      const { scene, cam } = await m8BitsScene(s);
      const res: Uint32Array[][] = [];
      for (const flags of ['', 'RS_PRIMARY_EXT']) {
        const rig = await restirRig(scene, 64, 64, { preset, settings: preset === 'interactive' ? { maxBounces: 3, ...INTERACTIVE_PINNED } : { maxBounces: 3 }, seed: 31, lightMode: 'B', cam, perfFlags: flags });
        const k = rig.kernel, device = rig.g.device;
        await k.prepare();
        expect(k.primaryPasses()).toEqual(flags ? ['rs_vtrace', 'rs_primary_ext'] : ['rs_primary']);
        const frames: Uint32Array[] = [];
        for (let f = 0; f < 3; f++) {
          if (k.settings.temporal) k.advance({ t: 3 + f, camera: cam, lights: scene.lights });
          k.beginSubmit();
          const enc = device.createCommandEncoder();
          for (const u of k.frameUnits(3 + f, { accum: rig.accum, counters: rig.counters })) u.encode(enc);
          device.queue.submit([enc.finish()]);
          for (const t of [k.resources.vbuf, k.resources.geo, k.resources.l1]) frames.push(await readTex(device, t));
        }
        res.push(frames);
        rig.destroy();
      }
      const names = ['vbuf', 'geo', 'l1'];
      const diffs = res[0].map((a, i) => { let n = 0; for (let j = 0; j < a.length; j++) if (a[j] !== res[1][i][j]) n++; return `${names[i % 3]}@${Math.floor(i / 3)}:${n}`; });
      console.log(`[U-WP7E-VTRACE] ${s} ${preset}: differing words ${diffs.join(' ')}`);
      // same ray and trace function: the hit ids are equal (ulp moves of the barycentrics are allowed)
      for (let i = 0; i < res[0].length; i += 3) {
        let prim = 0; for (let j = 0; j < res[0][i].length; j += 4) if (res[0][i][j] !== res[1][i][j]) prim++;
        expect(prim).toBeLessThanOrEqual(2);
      }
    }, 600_000);
  }
});

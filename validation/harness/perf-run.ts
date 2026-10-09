// M8 performance driver, browser side (docs/decisions/m8-perf.md §1). NOT a validation readback: it drives the
// interactive Renderer (the app's configuration: ReSTIR-interactive, light mode B, every M6 feature, c_cap 5, denoiser
// on, MT intersection) on a scene package or a glTF (+ HDRI) and measures, at a chosen internal resolution:
//   frame     pipelined frame time: blocks of `block` frames encoded and submitted back to back (one submit per frame,
//             as the app), wall time of the block / block (≥ 32 frames in total). The reported target number.
//   latency   isolated frames: one frame per submit, awaited, minus an empty submit's round trip.
//   passes    per-pass breakdown: the same frames encoded through a proxy GPUCommandEncoder that closes the command
//             buffer before every compute pass, then each command buffer is submitted alone and awaited (the
//             one-submit-per-pass method of the M6 / M7 perf tests; Q3: no timestampWrites on or around ReSTIR passes),
//             minus an empty submit's round trip. Passes are grouped by label kind (`rs_spatial_shift[0][0]` → rs_spatial_shift).
//   denoiser  the denoiser's own timing re-run (separate submits with timestamp writes, DN9; Q3 allows it: no ReSTIR pass).
// Moving lights: `lightAnim` moves light `index` sinusoidally along x every frame (the app's timeline playback).
import { describeContext, type GpuContext } from '../../src/core/gpu/device.ts';
import { DebugResources, DebugViewRegistry } from '../../src/core/render/debug-views.ts';
import { DENOISER_DEFAULTS, type DenoiserSettings } from '../../src/core/render/denoise/layout.ts';
import { FrameUniformBuffer, JITTER_IID, boundsDiagonal, computeRenderOrigin, type CameraState } from '../../src/core/render/frame-uniforms.ts';
import { Renderer, type RendererOptions } from '../../src/core/render/renderer.ts';
import type { RestirSettings } from '../../src/core/render/restir/presets.ts';
import { RELEASE_PERF_FLAGS, normalizePerfFlags, perfFlagsKey, type PerfFlagsInput } from '../../src/core/render/restir/perf-flags.ts';
import { fetchScenePackage } from '../../src/core/scene/scene-package.ts';
import { loadScene } from '../../src/core/scene/load-scene.ts';
import { loadEnvironment } from '../../src/core/scene/env/load-env.ts';
import { ensureLightStore } from '../../src/core/scene/light-store.ts';
import type { LightData, SceneData } from '../../src/core/scene/types.ts';

export interface PerfOptions {
  /** Scene package directory URL (`/validation/scenes/cornell_i_512/`) or a glTF URL (`….gltf` / `.glb`). */
  scene: string;
  /** HDRI URL (glTF scenes; packages carry their own env). */
  env?: string;
  /** Sponza-style setup for a glTF without camera / lights: camera along the long axis, one warm point light. */
  autoSetup?: boolean;
  width: number; height: number;
  /** Timed frames (≥ 32) after `warmup` frames; pipelined blocks of `block` frames. */
  frames?: number; warmup?: number; block?: number;
  /** Isolated-latency frames and per-pass split frames (default 16 each; 0 skips). */
  latencyFrames?: number; passFrames?: number;
  /** Renderer options over the app defaults (renderMode restir, interactive, lightMode B, MT). */
  renderer?: Partial<RendererOptions>;
  /** ReSTIR feature / settings overrides (e.g. { slots: 1 }) over the interactive preset. */
  restir?: Partial<RestirSettings>;
  /** perf2 (perf2-api.md): perf flags of this job ('A,B=2', a name list or { NAME: value }); default the app's
   *  (RELEASE_PERF_FLAGS). Applied to the interactive ReSTIR kernel and the M1 primary (renderer.restirKernel.perfFlags);
   *  recorded in the report. */
  perfFlags?: PerfFlagsInput;
  denoise?: boolean;
  denoiser?: Partial<DenoiserSettings>;
  /** Moving light: light `index` moves by amp·sin(0.09·i) metres along world x every frame. */
  lightAnim?: { index: number; amp: number };
  /** Camera pan per frame (metres along the camera's right axis). */
  pan?: number;
  seed?: number;
  label?: string;
  /** Profiling variants: drop material textures / alpha cutouts (MASK → OPAQUE) / the env; scale lights' power. */
  strip?: { textures?: boolean; alpha?: boolean; env?: boolean; normalMaps?: boolean };
}

export interface PerfReport {
  ok: boolean; label: string; errors: string[];
  scene: string; width: number; height: number; triangles: number; lights: number; env: boolean;
  settings: unknown; rendererOptions: unknown;
  /** perf2: the perf-flag key the job ran with (the release set when the job set none; empty = no flags). */
  perfFlags?: string;
  frame: { meanMs: number; medianMs: number; minMs: number; maxMs: number; blocks: number[]; frames: number };
  latency?: { meanMs: number; medianMs: number; rtMs: number; frames: number };
  passes?: { name: string; meanMs: number; count: number }[];
  passTotalMs?: number;
  denoiser?: { totalMs: number; passes: { name: string; ms: number }[] };
  counters?: { fr: number; queues: unknown };
  adapter: unknown; userAgent: string; createdAt: string; wallS: number;
}

const kindOf = (label: string) => label.replace(/[[#\s].*$/, '').replace(/-timing$/, '');
const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
const median = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };

function lookAt(eye: number[], tgt: number[]): number[] {
  const f = [tgt[0] - eye[0], tgt[1] - eye[1], tgt[2] - eye[2]];
  const n = Math.hypot(...f); for (let i = 0; i < 3; i++) f[i] /= n;
  const up = [0, 1, 0];
  const r = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]];
  const rn = Math.hypot(...r); for (let i = 0; i < 3; i++) r[i] /= rn;
  const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
  return [r[0], r[1], r[2], 0, u[0], u[1], u[2], 0, -f[0], -f[1], -f[2], 0, eye[0], eye[1], eye[2], 1];
}

/** The perf setup of a glTF without camera / lights (Sponza): camera along the long axis at 20 % height looking across
 *  70 % of it (yfov 55°), one warm 2000 W point light at the centre (25 % height) unless autoSetup is false, the HDRI
 *  (validation load mode) when given. Shared with the perf2 Sponza-lite bits case (validation/gpu-tests/m8-bits.ts). */
export async function loadGltfPerfScene(url: string, o: { env?: string; autoSetup?: boolean } = {}): Promise<{ scene: SceneData; camera: CameraState }> {
  let scene = (await loadScene(url)).scene;
  const b = scene.bounds, e = [0, 1, 2].map((i) => b.max[i] - b.min[i]), c = [0, 1, 2].map((i) => 0.5 * (b.min[i] + b.max[i]));
  const long = e[0] >= e[2] ? 0 : 2;
  const eye = [...c], tgt = [...c];
  eye[1] = tgt[1] = b.min[1] + 0.2 * e[1];
  eye[long] = c[long] + 0.35 * e[long]; tgt[long] = c[long] - 0.35 * e[long];
  const camera: CameraState = { camToWorld: lookAt(eye, tgt), yfov: (55 * Math.PI) / 180, znear: 1e-4 };
  if (o.autoSetup !== false) {
    const store = ensureLightStore(scene);
    store.add({ type: 'point', power: 2000, color: [1, 0.85, 0.7], matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, c[0], b.min[1] + 0.25 * e[1], c[2], 1]) });
    scene = { ...scene, lights: store.list().map((l) => ({ ...l })) as LightData[] };
  }
  if (o.env) scene = { ...scene, env: (await loadEnvironment(o.env, { mode: 'validation' })).env };
  return { scene, camera };
}

/** Proxy encoder: every compute pass starts a new real command encoder (the previous one is finished and kept). */
function splitEncoder(device: GPUDevice): { enc: GPUCommandEncoder; finish(): { label: string; cb: GPUCommandBuffer }[] } {
  const out: { label: string; cb: GPUCommandBuffer }[] = [];
  let cur = device.createCommandEncoder({ label: 'perf-split' });
  let curLabel = '(pre)';
  let used = false;
  const flush = () => { if (used) out.push({ label: curLabel, cb: cur.finish() }); else cur.finish(); };
  const proxy = new Proxy({} as GPUCommandEncoder, {
    get(_t, prop) {
      if (prop === 'beginComputePass') {
        return (d?: GPUComputePassDescriptor) => {
          if (used && curLabel !== '(pre)') { flush(); cur = device.createCommandEncoder({ label: 'perf-split' }); used = false; }
          curLabel = String(d?.label ?? 'compute');
          used = true;
          return cur.beginComputePass(d);
        };
      }
      const v = (cur as unknown as Record<string | symbol, unknown>)[prop];
      if (typeof v === 'function') return (...args: unknown[]) => { used = true; return (v as (...a: unknown[]) => unknown).apply(cur, args); };
      return v;
    },
  });
  return { enc: proxy, finish() { flush(); return out; } };
}

export async function renderPerf(ctx: GpuContext, o: PerfOptions): Promise<PerfReport> {
  const t0 = performance.now();
  const errors: string[] = [];
  const { device } = ctx;
  const W = o.width, H = o.height;
  const isGltf = /\.(gltf|glb)$/i.test(o.scene);
  let scene: SceneData;
  let camera: CameraState;
  let pkgBounces: number | undefined;
  if (isGltf) {
    ({ scene, camera } = await loadGltfPerfScene(o.scene, { env: o.env, autoSetup: o.autoSetup }));
  } else {
    const p = await fetchScenePackage(o.scene);
    scene = p.scene;
    camera = { camToWorld: Array.from(p.camera.matrix), yfov: p.camera.yfov, znear: 1e-4 };
    pkgBounces = p.render.maxBounces;
  }
  if (o.strip) {
    const st = o.strip;
    scene = { ...scene, materials: scene.materials.map((m) => ({
      ...m,
      ...(st.textures ? { baseColorTexture: undefined, metallicRoughnessTexture: undefined, emissiveTexture: undefined, specularTexture: undefined, specularColorTexture: undefined, transmissionTexture: undefined } : {}),
      ...(st.textures || st.normalMaps ? { normalTexture: undefined } : {}),
      ...(st.alpha ? { alphaMode: 'OPAQUE' as const } : {}),
    })) };
    if (st.env) scene = { ...scene, env: undefined };
  }
  const debug = new DebugResources(device, new DebugViewRegistry());
  await debug.init();
  debug.resize(W, H);
  const hasGlass = scene.materials.some((m) => m.model === 'glass' || m.model === 'refraction' || m.transmissionFactor > 1e-5);
  const ropts: Partial<RendererOptions> = {
    watertight: false, renderMode: 'restir', restirMode: 'interactive', temporal: true, accumulate: false, lightMode: 'B', bvhKind: 'auto',
    maxBounces: Math.max(pkgBounces ?? 3, hasGlass ? 4 : 0),
    restirFeatures: { ...(o.restir ?? {}) } as RendererOptions['restirFeatures'], ...o.renderer,
  };
  if (o.perfFlags !== undefined) ropts.restirKernel = { ...ropts.restirKernel, perfFlags: normalizePerfFlags(o.perfFlags) };
  const r = await Renderer.create({ device, debugLayout: debug.layout, debug, features: ctx.features, wgslLanguageFeatures: ctx.wgslLanguageFeatures }, ropts);
  const origin = computeRenderOrigin(scene.bounds, scene.quant);
  const fu = new FrameUniformBuffer(device);
  const color = device.createTexture({ label: 'perf-colour', size: [W, H], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
  const depth = device.createTexture({ label: 'perf-depth', size: [W, H], format: 'r32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
  const report: PerfReport = {
    ok: false, label: o.label ?? '', errors, scene: o.scene, width: W, height: H, triangles: scene.geometry.indices.length / 3, lights: scene.lights.length, env: !!scene.env,
    settings: undefined, rendererOptions: ropts, perfFlags: perfFlagsKey(ropts.restirKernel?.perfFlags ?? RELEASE_PERF_FLAGS), frame: { meanMs: NaN, medianMs: NaN, minMs: NaN, maxMs: NaN, blocks: [], frames: 0 },
    adapter: describeContext(ctx), userAgent: navigator.userAgent, createdAt: new Date().toISOString(), wallS: 0,
  };
  try {
    if (scene.env) await r.setEnvironment(scene.env);
    const g = await r.setScene(scene, origin);
    if (!g) throw new Error('setScene superseded');
    r.resize({ width: W, height: H, color, colorFormat: 'rgba16float', depth, frameUniforms: fu.buffer });
    const denoise = o.denoise ?? true;
    r.setDenoise(denoise);
    r.setDenoiserSettings({ ...DENOISER_DEFAULTS, ...o.denoiser });
    if (!await r.prepareRestir()) throw new Error(`ReSTIR compile failed: ${r.restirError}`);
    if (denoise && !await r.prepareDenoiser()) throw new Error(`denoiser compile failed: ${r.denoiserError}`);
    await r.warmup(false);
    report.settings = r.restir?.settings;
    const sceneDiag = boundsDiagonal(scene.bounds);
    const base = scene.lights.map((l) => ({ ...l, matrix: new Float32Array(l.matrix) }));
    let prev: CameraState | undefined;
    let frameNo = 0;
    const setup = (i: number) => {
      const cam: CameraState = { camToWorld: camera.camToWorld.slice(), yfov: camera.yfov, znear: 1e-4 };
      if (o.pan) for (let a = 0; a < 3; a++) cam.camToWorld[12 + a] += i * o.pan * cam.camToWorld[a];
      if (o.lightAnim) {
        const la = o.lightAnim;
        r.setLights(base.map((l, k) => {
          if (k !== la.index) return l;
          const m = new Float32Array(l.matrix); m[12] += la.amp * Math.sin(0.09 * i);
          return { ...l, matrix: m };
        }));
      } else if (i === 0) r.setLights(base);
      fu.write({
        camera: cam, prevCamera: prev ?? cam, width: W, height: H, frameIndex: i, seedIndex: i, runSeed: (o.seed ?? 1) >>> 0, flags: 0,
        jitterMode: JITTER_IID, jitter: [0.5, 0.5], origin, exposure: 1, time: i / 24, dt: 1 / 24, sceneDiag,
      });
      prev = cam;
      debug.update({ ...debug.settings, mode: 0 }, i);
    };
    const encodeFrame = (enc: GPUCommandEncoder, i: number) => {
      debug.beginFrame(enc);
      const ok = r.encode(enc, { advanced: true, debugMode: 0, debugGroup: debug.bindGroup, resetTemporal: i === 0, resetHistory: i === 0 });
      if (!ok || r.denoisedLastFrame !== denoise) throw new Error(`frame ${i}: renderer not ready (${r.restirError ?? ''} ${r.denoiserError ?? ''} ${r.lastError ?? ''})`);
    };
    const oneFrame = (i: number) => { setup(i); const enc = device.createCommandEncoder({ label: `perf-${i}` }); encodeFrame(enc, i); device.queue.submit([enc.finish()]); };
    // empty-submit round trip
    const rt: number[] = [];
    for (let k = 0; k < 64; k++) { const a = performance.now(); device.queue.submit([device.createCommandEncoder().finish()]); await device.queue.onSubmittedWorkDone(); rt.push(performance.now() - a); }
    const rtMs = median(rt);
    // warm-up
    const warm = o.warmup ?? 32;
    for (let k = 0; k < warm; k++) { oneFrame(frameNo++); if (k % 8 === 7) await device.queue.onSubmittedWorkDone(); }
    await device.queue.onSubmittedWorkDone();
    // pipelined frames
    const nFrames = Math.max(32, o.frames ?? 64), block = o.block ?? 8;
    const blocks: number[] = [];
    for (let done = 0; done < nFrames; done += block) {
      const a = performance.now();
      for (let k = 0; k < block; k++) oneFrame(frameNo++);
      await device.queue.onSubmittedWorkDone();
      blocks.push((performance.now() - a) / block);
    }
    report.frame = { meanMs: mean(blocks), medianMs: median(blocks), minMs: Math.min(...blocks), maxMs: Math.max(...blocks), blocks, frames: blocks.length * block };
    // isolated latency
    const nLat = o.latencyFrames ?? 16;
    if (nLat > 0) {
      const lat: number[] = [];
      for (let k = 0; k < nLat; k++) { const a = performance.now(); oneFrame(frameNo++); await device.queue.onSubmittedWorkDone(); lat.push(performance.now() - a - rtMs); }
      report.latency = { meanMs: mean(lat), medianMs: median(lat), rtMs, frames: nLat };
    }
    // per-pass split
    const nSplit = o.passFrames ?? 16;
    if (nSplit > 0) {
      const per = new Map<string, number[]>();
      const totals: number[] = [];
      for (let k = 0; k < nSplit; k++) {
        setup(frameNo);
        const s = splitEncoder(device);
        encodeFrame(s.enc, frameNo++);
        const cbs = s.finish();
        const sums = new Map<string, number>();
        let tot = 0;
        for (const { label, cb } of cbs) {
          const a = performance.now();
          device.queue.submit([cb]);
          await device.queue.onSubmittedWorkDone();
          const ms = Math.max(0, performance.now() - a - rtMs);
          const kk = kindOf(label);
          sums.set(kk, (sums.get(kk) ?? 0) + ms);
          tot += ms;
        }
        totals.push(tot);
        for (const [kk, v] of sums) per.set(kk, [...(per.get(kk) ?? []), v]);
      }
      report.passes = [...per].map(([name, v]) => ({ name, meanMs: mean(v), count: v.length }));
      report.passTotalMs = mean(totals);
    }
    if (denoise && r.denoiser) {
      const runs: { totalMs: number; passes: { name: string; ms: number }[] }[] = [];
      for (let k = 0; k < 8; k++) { await device.queue.onSubmittedWorkDone(); const t = await r.denoiser.time(8); if (t) runs.push(t); }
      if (runs.length) {
        const names = runs[0].passes.map((x) => x.name);
        report.denoiser = { totalMs: mean(runs.map((x) => x.totalMs)), passes: names.map((name) => ({ name, ms: mean(runs.map((x) => x.passes.find((q) => q.name === name)?.ms ?? 0)) })) };
      } else errors.push('denoiser timing unavailable (timestamp-query)');
    }
    if (r.restir) {
      try { const c = await r.restir.kernel.readCounters(false); report.counters = { fr: c.fr, queues: c.queues }; } catch (e) { errors.push(`counters: ${(e as Error).message}`); }
    }
    const fin = r.restir ? Array.from(new Uint32Array(await (async () => {
      const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = device.createCommandEncoder(); e.copyBufferToBuffer(r.restir!.counters, 0, b, 0, 16); device.queue.submit([e.finish()]);
      await b.mapAsync(GPUMapMode.READ); const x = b.getMappedRange().slice(0); b.unmap(); b.destroy(); return x;
    })())) : [];
    if (fin[0]) errors.push(`finalize: ${fin[0]} non-finite pixels`);
    report.ok = errors.length === 0;
  } catch (e) {
    errors.push(e instanceof Error ? `${e.message}\n${e.stack ?? ''}` : String(e));
  } finally {
    r.destroy(); debug.destroy(); fu.destroy(); color.destroy(); depth.destroy();
    report.wallS = (performance.now() - t0) / 1000;
  }
  return report;
}

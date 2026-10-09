// Shared fixtures for the ReSTIR GPU tests (restir-api.md §1.3, WP-A; other WPs import these and add helpers in their
// own files): small quad scenes that exercise every light type, V1/V2/mirror materials and an env map; a PT rig with
// deterministic accumulation (U-PT-BITS image hashes); a ReSTIR rig on top of RestirKernel; reservoir decoding.
import { buildBvh } from '../../src/core/bvh/sah-builder.ts';
import { buildCwbvhFromMesh } from '../../src/core/bvh/cwbvh.ts';
import { normalizePerfFlags, type PerfFlagsInput } from '../../src/core/render/restir/perf-flags.ts';
import type { TexturePathMode } from '../../src/core/render/textures-gpu.ts';
import { BatchAccumulator } from '../../src/core/render/batch-accumulator.ts';
import { createEnvResources, destroyEnvResources, type EnvGpuResources } from '../../src/core/render/env-gpu.ts';
import { computeRenderOrigin, JITTER_IID, type JitterMode } from '../../src/core/render/frame-uniforms.ts';
import { PtKernel, type PtEnvOptions, type PtKernelOptions } from '../../src/core/render/pt-kernel.ts';
import { SceneGpu } from '../../src/core/render/scene-gpu.ts';
import type { LightMode } from '../../src/core/render/lights-gpu.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { RestirKernel, type RestirCounters } from '../../src/core/render/restir/kernel.ts';
import { restirSettings, type RestirPresetName, type RestirSettings } from '../../src/core/render/restir/presets.ts';
import type { LightData, MaterialData, SceneData } from '../../src/core/scene/types.ts';
import { getTestGpu } from './device-factory.ts';
import { synthEnvData, sunTexelEnv } from './env-fixtures.ts';
import { lambert, lightMatrixToward, material, quadScene, type V3 } from './pt-fixtures.ts';

export type { V3 };

// ------------------------------------------------------------------------------------------------ scenes

export function light(o: Partial<LightData> & Pick<LightData, 'type'>): LightData {
  return { id: 0, name: o.type, color: [1, 1, 1], power: 100, exposure: 0, matrix: lightMatrixToward([0, -1, 0], [0, 1, 0]), visibleToCamera: false, ...o };
}

export const v1Mix = (d: V3, g: V3, r: number, mix: number): MaterialData =>
  material({ baseColorFactor: [...d, 1], v1: { diffuse: d, glossy: g, roughness: r, mix } });
export const v2 = (base: V3, metallic: number, roughness: number, o: Partial<MaterialData> = {}): MaterialData =>
  material({ model: 'principled', baseColorFactor: [...base, 1], metallicFactor: metallic, roughnessFactor: roughness, specularFactor: 1, ...o });
export const emissive = (Le: V3): MaterialData =>
  material({ emissiveFactor: Le, v1: { diffuse: [0, 0, 0], glossy: [0, 0, 0], roughness: 0.5, mix: 0 } });

/** Materials of the box: 0 floor Lambert, 1 back wall V1 mix (r 0.3), 2 left wall V2 metal (r 0.25), 3 right wall
 *  Lambert (coloured), 4 mirror (V1 glossy r 0), 5 emissive, 6 V2 dielectric (r 0.4). */
export function boxMaterials(Le: V3 = [4, 3, 2]): MaterialData[] {
  return [
    lambert(0.6),
    v1Mix([0.5, 0.45, 0.4], [0.8, 0.8, 0.8], 0.3, 0.5),
    v2([0.9, 0.7, 0.5], 1, 0.25),
    lambert([0.2, 0.5, 0.3]),
    v1Mix([0, 0, 0], [0.9, 0.9, 0.9], 0, 1),
    emissive(Le),
    v2([0.3, 0.3, 0.8], 0, 0.4),
  ];
}

/** Open-top box (x ∈ [−1.5, 1.5], y ∈ [0, 2], z ∈ [−2, 1]) with a mirror, a dielectric block face and optionally an
 *  emissive quad; the camera (boxCamera) looks into it along −Z. Normals face the interior. */
export function boxScene(lights: LightData[], o: { emissiveQuad?: boolean; env?: SceneData['env']; Le?: V3 } = {}): SceneData {
  const quads = [
    { p: [[-1.5, 0, 1], [1.5, 0, 1], [1.5, 0, -2], [-1.5, 0, -2]], mat: 0 },          // floor (+Y)
    { p: [[-1.5, 0, -2], [1.5, 0, -2], [1.5, 2, -2], [-1.5, 2, -2]], mat: 1 },        // back wall (+Z)
    { p: [[-1.5, 0, 1], [-1.5, 0, -2], [-1.5, 2, -2], [-1.5, 2, 1]], mat: 2 },        // left wall (+X)
    { p: [[1.5, 0, -2], [1.5, 0, 1], [1.5, 2, 1], [1.5, 2, -2]], mat: 3 },            // right wall (−X)
    { p: [[0.3, 0.02, -1.2], [1.0, 0.02, -1.6], [1.0, 0.9, -1.6], [0.3, 0.9, -1.2]], mat: 4 },   // mirror
    { p: [[-1.0, 0.0, -0.8], [-0.4, 0.0, -0.8], [-0.4, 0.6, -0.8], [-1.0, 0.6, -0.8]], mat: 6 }, // dielectric face
  ];
  if (o.emissiveQuad) quads.push({ p: [[-0.4, 1.9, -1.2], [0.4, 1.9, -1.2], [0.4, 1.9, -0.6], [-0.4, 1.9, -0.6]], mat: 5 });
  const s = quadScene(quads, boxMaterials(o.Le), lights);
  if (o.env) s.env = o.env;
  return s;
}

/** Column-major camera at (0, 1, 3.2) looking along −Z (slightly down). */
export function boxCamera(): { camToWorld: number[]; yfov: number } {
  const a = -0.12;   // pitch
  const Y: V3 = [0, Math.cos(a), -Math.sin(a)], Z: V3 = [0, Math.sin(a), Math.cos(a)];
  return { camToWorld: [1, 0, 0, 0, ...Y, 0, ...Z, 0, 0, 1, 3.2, 1], yfov: 55 * Math.PI / 180 };
}

export type BitFixture = 'c0c' | 'c0e' | 'c0m' | 'x_quads' | 'c0s';

/** Scenes of the U-PT-BITS image hashes (restir-api.md §6.1): point (C0c-like), rect (C0e), disk (C0m), every light type
 *  + emissive quad + camera-visible rect ((x) quads), env with a hot texel (C0s-like) — all inside the box. */
export function bitFixtureScene(name: BitFixture): SceneData {
  const down = (p: V3) => lightMatrixToward([0, -1, 0], p);
  switch (name) {
    case 'c0c': return boxScene([light({ id: 1, type: 'point', power: 60, matrix: down([0.2, 1.7, -0.5]) })]);
    case 'c0e': return boxScene([light({ id: 1, type: 'rect', power: 80, sizeX: 0.6, sizeY: 0.4, matrix: down([0, 1.95, -0.9]) })]);
    case 'c0m': return boxScene([light({ id: 1, type: 'disk', power: 80, sizeX: 0.5, matrix: down([-0.3, 1.9, -1.0]) })]);
    case 'x_quads': return boxScene([
      light({ id: 1, type: 'point', power: 30, matrix: down([0.8, 1.6, -0.2]) }),
      light({ id: 2, type: 'spot', power: 50, spotSize: 1.0, spotBlend: 0.2, matrix: lightMatrixToward([0.2, -1, -0.3], [-0.5, 1.8, 0.2]) }),
      light({ id: 3, type: 'rect', power: 40, sizeX: 0.5, sizeY: 0.3, visibleToCamera: true, matrix: lightMatrixToward([0, -0.3, 1], [0.3, 1.3, -1.9]) }),
      light({ id: 4, type: 'disk', power: 40, sizeX: 0.3, spread: 1.2, matrix: down([-0.8, 1.9, -1.5]) }),
      light({ id: 5, type: 'sun', power: 2, matrix: lightMatrixToward([0.3, -1, -0.4], [0, 5, 0]) }),
    ], { emissiveQuad: true });
    case 'c0s': return boxScene([], { env: sunTexelEnv(64, 32, 20, 24, 2e3) });
  }
}

/** A scene with every endpoint type: point, spot, rect, disk, sun, emissive triangles and a synthetic env. */
export function allLightsScene(): SceneData {
  const s = bitFixtureScene('x_quads');
  s.env = synthEnvData(64, 32);
  return s;
}

// ------------------------------------------------------------------------------------------------ PT rig

export interface GpuScene { device: GPUDevice; features: Set<string>; wgslLanguageFeatures: Set<string>; gpu: SceneGpu; env: EnvGpuResources; destroy(): void }

/** Scene upload options of the test rigs (default: the validation configuration — BVH2, Woop, exact texels). perf2 WP-0:
 *  the CWBVH-forced bits cases and Sponza-lite set bvhKind 'cwbvh' (and MT intersection, the app's). */
export interface GpuSceneOptions { bvhKind?: 'bvh2' | 'cwbvh'; watertight?: boolean; textureMode?: TexturePathMode }

export async function gpuScene(scene: SceneData, o: GpuSceneOptions = {}): Promise<GpuScene> {
  const { device, features, wgslLanguageFeatures } = await getTestGpu();
  const origin = computeRenderOrigin(scene.bounds, scene.quant);
  const cw = o.bvhKind === 'cwbvh';
  const gpu = await SceneGpu.create(device, scene, origin, {
    textureMode: o.textureMode ?? 'validation', watertight: o.watertight ?? true, features, wgslLanguageFeatures,
    buildBvh: async (p, i, bo) => {
      if (bo?.cwbvh) { const r = buildCwbvhFromMesh(p, i); return { ...r.bvh2, cwbvh: r.cw }; }
      return buildBvh(p, i, { mt: true, woop: true });
    },
    ...(cw ? { bvhKind: 'cwbvh' as const } : {}),
  });
  const env = await createEnvResources(device, scene.env);
  return { device, features, wgslLanguageFeatures, gpu, env, destroy: () => { gpu.destroy(); destroyEnvResources(env); } };
}

/** Deterministic PT image: k = 1 sample per dispatch (fixed f32 summation order), `spp` samples, batch index 0. */
export async function ptImage(scene: SceneData, W: number, H: number, spp: number, opts: PtKernelOptions & { seed?: number; jitterMode?: JitterMode } = {}): Promise<{ mean: Float32Array; counters: number[] }> {
  const g = await gpuScene(scene);
  const cam = boxCamera();
  const kernel = await PtKernel.create(g.device, g.gpu, g.env, { features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures, maxBounces: 3, ...opts });
  kernel.setView({ camera: cam, width: W, height: H, runSeed: opts.seed ?? 11, jitterMode: opts.jitterMode ?? JITTER_IID });
  const acc = new BatchAccumulator(g.device, W, H, { maxSamplesPerDispatch: 1 });
  const b = await acc.runBatch((enc, d, a, c) => kernel.encode(enc, d, a, c), spp, 0);
  kernel.destroy(); acc.destroy(); g.destroy();
  return { mean: b.mean, counters: b.counters };
}

/** FNV-1a (32 bit) over the bit patterns of a float array, as 8 hex digits. */
export function hashF32(a: Float32Array): string {
  const u = new Uint32Array(a.buffer, a.byteOffset, a.length);
  let h = 0x811c9dc5;
  for (let i = 0; i < u.length; i++) {
    let x = u[i];
    for (let k = 0; k < 4; k++) { h ^= x & 0xff; h = Math.imul(h, 0x01000193) >>> 0; x >>>= 8; }
  }
  return h.toString(16).padStart(8, '0');
}

// ------------------------------------------------------------------------------------------------ ReSTIR rig

export interface RestirRigOptions {
  settings?: Partial<RestirSettings>;
  preset?: RestirPresetName;
  cam?: { camToWorld: number[]; yfov: number };
  seed?: number;
  jitterMode?: JitterMode;
  dumpCandidates?: boolean;
  members?: number;
  memberBase?: number;
  env?: PtEnvOptions;
  initialDefines?: Record<string, number | boolean | string>;
  extraSources?: Record<string, string>;
  /** M6 (restir-m6-api.md MD9): light mode of the ReSTIR kernel (default A). */
  lightMode?: LightMode;
  /** M8 (m8-perf.md §8): reservoir plane order (default the validation 'aos'). */
  resLayout?: 'aos' | 'soa';
  /** M8 P-4: compile Mode B as the Mode-A text while no rect / disk light exists (the interactive kernel's option). */
  modeBNeedsAreaLights?: boolean;
  /** perf2 (perf2-api.md): perf flags of the kernel. Default: testPerfFlags() (VITE_PERF_FLAGS; none when unset). */
  perfFlags?: PerfFlagsInput;
  /** perf2: scene upload options (BVH kind, intersection, texture path). */
  gpu?: GpuSceneOptions;
}

/** perf2 (perf2-api.md): flags forced on by the environment for every test rig that does not pass its own —
 *  `VITE_PERF_FLAGS=RS_VIS_MERGE,CW_TRI_BUDGET=2 npx vitest run --project chrome …` runs the T3 fixtures, VITE_STRESS,
 *  the bits suites etc. with the flags on (V-BIT / V-SHIFT / V-UNB of perf2-plan.md §2). */
export function testPerfFlags(): PerfFlagsInput {
  const env = (import.meta as unknown as { env?: Record<string, unknown> }).env;
  return normalizePerfFlags(String(env?.VITE_PERF_FLAGS ?? ''));
}

export interface RestirRig {
  g: GpuScene; kernel: RestirKernel; W: number; H: number; accum: GPUBuffer; counters: GPUBuffer;
  /** Render frames t = base … base+n−1 (one submit each) into a cleared accumulator: mean image (RGB, row 0 = top),
   *  the finalize counters (nonFinite, bvhOverflow, bvhItercap, negative) and the arena counters. */
  frames(n: number, base?: number, perSubmit?: number): Promise<{ mean: Float32Array; counters: number[]; arena: RestirCounters }>;
  destroy(): void;
}

export async function restirRig(scene: SceneData, W: number, H: number, o: RestirRigOptions = {}): Promise<RestirRig> {
  const g = await gpuScene(scene, o.gpu);
  const kernel = await RestirKernel.create(g.device, g.gpu, g.env, {
    settings: restirSettings(o.preset ?? 'initial', o.settings), features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures,
    instrumentation: { dumpCandidates: o.dumpCandidates, initialDefines: o.initialDefines, extraSources: o.extraSources }, env: o.env,
    lightMode: o.lightMode, resLayout: o.resLayout, modeBNeedsAreaLights: o.modeBNeedsAreaLights, perfFlags: o.perfFlags ?? testPerfFlags(),
  });
  kernel.setView({ camera: o.cam ?? boxCamera(), width: W, height: H, runSeed: o.seed ?? 11, jitterMode: o.jitterMode ?? JITTER_IID, members: o.members, memberBase: o.memberBase });
  await kernel.prepare();
  const device = g.device;
  const accum = device.createBuffer({ label: 'rig-accum', size: W * H * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const counters = device.createBuffer({ label: 'rig-counters', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  return {
    g, kernel, W, H, accum, counters,
    async frames(n: number, base = 0, perSubmit = 1) {
      const clear = device.createCommandEncoder();
      clear.clearBuffer(accum);
      clear.clearBuffer(counters);
      clear.clearBuffer(kernel.resources.arena, 0, 256);
      device.queue.submit([clear.finish()]);
      for (let f = 0; f < n; f += perSubmit) {
        kernel.beginSubmit();
        const enc = device.createCommandEncoder({ label: `rig-frame-${base + f}` });
        for (let j = f; j < Math.min(n, f + perSubmit); j++) for (const u of kernel.frameUnits(base + j, { accum, counters })) u.encode(enc);
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
      }
      const sum = new Float32Array(await readBuffer(device, accum, W * H * 16));
      const mean = new Float32Array(W * H * 3);
      for (let i = 0; i < W * H; i++) for (let c = 0; c < 3; c++) mean[3 * i + c] = sum[4 * i + c] / n;
      const cnt = Array.from(new Uint32Array(await readBuffer(device, counters, 16)));
      return { mean, counters: cnt, arena: await kernel.readCounters(false) };
    },
    destroy() { kernel.destroy(); accum.destroy(); counters.destroy(); g.destroy(); },
  };
}

/** Read an atlas-sized rgba32float / rgba32uint texture (row 0 = top) as raw 32-bit words, 4 per pixel. */
export async function readTexture4(device: GPUDevice, tex: GPUTexture): Promise<Uint32Array> {
  const W = tex.width, H = tex.height;
  const bpr = Math.ceil((W * 16) / 256) * 256;
  const buf = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [W, H]);
  device.queue.submit([enc.finish()]);
  const raw = new Uint32Array(await readBuffer(device, buf, bpr * H));
  buf.destroy();
  const out = new Uint32Array(W * H * 4);
  for (let r = 0; r < H; r++) out.set(raw.subarray((r * bpr) / 4, (r * bpr) / 4 + W * 4), r * W * 4);
  return out;
}

/** Storage buffer with initial contents (or zeros). */
export function storageBuffer(device: GPUDevice, data: ArrayBufferView | number, label = 'test'): GPUBuffer {
  const bytes = typeof data === 'number' ? data : data.byteLength;
  const b = device.createBuffer({ label, size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  if (typeof data !== 'number') device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
  return b;
}

/**
 * Test replacement of restir/rc.wgsl (pass it through `extraSources: { 'restir/rc.wgsl': TEST_RC_DR }`): the contract
 * signatures of §3.4 with rcPairTest = D ∧ R only (e_a non-delta with α ≥ α_min; e_b non-delta unless the vertex is
 * a light/env), so the path tree produces every rc case (b, c, d, e, deep) independently of WP-B's predicate.
 */
export const TEST_RC_DR = `
#include "restir/types.wgsl"
#include "material/material-eval.wgsl"
struct RcVertex { pos: vec3f, ng: vec3f, kind: u32, diffuseOnly: u32 }
struct RcEvent  { lobe: u32, delta: u32, alpha: f32, pMarg: f32 }
struct RcResult { ok: bool, margin: f32, term: u32 }
fn vertex_from_ids(prim: u32, u: f32, v: f32, fromPos: vec3f) -> SurfaceHit {
  var s = scene_surface(prim, u, v, vec3f(0.0));
  if (dot(s.ng, fromPos - s.pos) < 0.0) { s.ng = -s.ng; s.ns = -s.ns; s.backfacing = true; }
  return s;
}
fn rc_vertex(s: SurfaceHit, m: MatEval) -> RcVertex { return RcVertex(s.pos, s.ng, RCK_SURFACE, select(0u, 1u, (m.flags & MATEVAL_DIFFUSE_ONLY) != 0u)); }
fn rc_event_bsdf(m: MatEval, lobe: u32, isDelta: bool, pMarg: f32) -> RcEvent { return RcEvent(lobe, select(0u, 1u, isDelta), select(lobe_roughness(m, lobe), 0.0, isDelta), pMarg); }
fn rc_event_nee(m: MatEval, pMarg: f32) -> RcEvent { return RcEvent(LOBE_NEE, 0u, lobe_roughness(m, LOBE_NEE), pMarg); }
fn rc_event_none() -> RcEvent { return RcEvent(LOBE_NONE, 0u, FLT_MAX, 0.0); }
fn rcPairTest(a: RcVertex, ea: RcEvent, b: RcVertex, eb: RcEvent, thr: f32) -> RcResult {
  let okA = ea.delta == 0u && ea.lobe != LOBE_GT && ea.alpha >= rsParams.alphaMin && ea.lobe != LOBE_NONE;
  let okB = b.kind != RCK_SURFACE || (eb.delta == 0u && eb.lobe != LOBE_GT);
  return RcResult(okA && okB, ea.alpha - rsParams.alphaMin, select(RCT_R, RCT_NONE, okA && okB));
}
fn primaryThreshold(camPos: vec3f, x1: vec3f, ng1: vec3f, tau: f32) -> f32 {
  let d = camPos - x1;
  return tau * dot(d, d) * 4.0 * PI / max(abs(dot(ng1, normalize(d))), 1e-6);
}
fn rc_G(a: vec3f, b: vec3f, ngB: vec3f) -> f32 { let d = a - b; let t2 = dot(d, d); return abs(dot(ngB, d)) / (t2 * sqrt(t2)); }
fn kstar_nee(treeRc: u32, B: u32, prevV: RcVertex, prevE: RcEvent, curV: RcVertex, eNee: RcEvent, thr: f32) -> vec2u {
  if (treeRc != 0u) { return vec2u(treeRc, 0u); }
  if (B >= 2u) { let r = rcPairTest(prevV, prevE, curV, eNee, thr); if (r.ok) { return vec2u(B, bitcast<u32>(r.margin)); } }
  return vec2u(B + 1u, 0u);
}
fn kstar_tree_pair(prevV: RcVertex, prevE: RcEvent, curV: RcVertex, eB: RcEvent, thr: f32) -> RcResult { return rcPairTest(prevV, prevE, curV, eB, thr); }
fn kstar_bsdf_end(treeRc: u32, B: u32, curV: RcVertex, eB: RcEvent, endV: RcVertex, thr: f32) -> vec2u {
  if (treeRc != 0u) { return vec2u(treeRc, 0u); }
  let r = rcPairTest(curV, eB, endV, rc_event_none(), thr);
  if (r.ok) { return vec2u(B + 1u, bitcast<u32>(r.margin)); }
  return vec2u(0u, 0u);
}
`;

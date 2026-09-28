// Primary pass (plan §5 M1): V-buffer vs an f64 brute-force reference with the math.md#raster camera (both
// intersectors), oriented Ng/Ns, depth, thr, the alpha-MASK cutout (factor and COLOR_0 alpha), the camera-miss env
// term (strength·tint, visibleToCamera), motion vectors, accumulation, and zero NaN/Inf/BVH-overflow counters.
import { afterAll, describe, expect, it } from 'vitest';
import { getTestGpu, releaseTestGpu } from './device-factory.ts';
import { buildBvh } from '../../src/core/bvh/sah-builder.ts';
import { bruteClosest } from '../../src/core/bvh/cpu-trace.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { DBGC, DEBUG_BUFFER_LAYOUT, DebugResources, defaultDebugSettings } from '../../src/core/render/debug-views.ts';
import {
  FRAME_RESET_HISTORY, FrameUniformBuffer, JITTER_NONE, computeRenderOrigin, recentre, rigidInverse, type FrameUniformInput,
} from '../../src/core/render/frame-uniforms.ts';
import { GBUF_TEXEL_BYTES, Renderer } from '../../src/core/render/renderer.ts';
import { SceneGpu, recentrePositions } from '../../src/core/render/scene-gpu.ts';
import { composeWgsl, createCheckedShaderModule } from '../../src/core/gpu/wgsl-composer.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { loadGltf } from '../../src/core/scene/gltf-loader.ts';
import { JITTER_IID } from '../../src/core/render/frame-uniforms.ts';
import { TRI_ALPHA_MASK, type EnvironmentData, type MaterialData, type SceneData, type TextureData } from '../../src/core/scene/types.ts';

afterAll(releaseTestGpu);

async function readAsset(rel: string): Promise<ArrayBuffer | undefined> {
  if (typeof window !== 'undefined') {
    const res = await fetch(`/${rel}`);
    if (!res.ok || (res.headers.get('content-type') ?? '').includes('text/html')) return undefined;
    return res.arrayBuffer();
  }
  const fsName = 'node:fs/promises';
  const fs = (await import(/* @vite-ignore */ fsName)) as typeof import('node:fs/promises');
  try { const b = await fs.readFile(rel); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer; } catch { return undefined; }
}

const lookAt = (eye: number[], target: number[]): Float64Array => {
  const f = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
  const fl = Math.hypot(...f);
  const back = f.map((x) => -x / fl);
  const up = [0, 1, 0];
  let right = [up[1] * back[2] - up[2] * back[1], up[2] * back[0] - up[0] * back[2], up[0] * back[1] - up[1] * back[0]];
  const rl = Math.hypot(...right);
  right = right.map((x) => x / rl);
  const u2 = [back[1] * right[2] - back[2] * right[1], back[2] * right[0] - back[0] * right[2], back[0] * right[1] - back[1] * right[0]];
  return Float64Array.from([...right, 0, ...u2, 0, ...back, 0, eye[0], eye[1], eye[2], 1]);
};

function material(o: Partial<MaterialData> = {}): MaterialData {
  return {
    name: 'm', baseColorFactor: [0.8, 0.8, 0.8, 1], metallicFactor: 0, roughnessFactor: 0.5, emissiveFactor: [0, 0, 0], emissiveStrength: 1,
    ior: 1.5, specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5,
    doubleSided: true, model: 'principled', ...o,
  };
}

/** Axis-aligned quads in the z = const plane, facing +Z. */
function quads(qs: { z: number; half: number; mat: number; flags?: number; alpha?: number }[], materials: MaterialData[], textures: TextureData[] = []): SceneData {
  const pos: number[] = [], nrm: number[] = [], uv: number[] = [], col: number[] = [], idx: number[] = [], tm: number[] = [], tf: number[] = [];
  for (const q of qs) {
    const b = pos.length / 3;
    for (const [x, y] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      pos.push(x * q.half, y * q.half, q.z); nrm.push(0, 0, 1); uv.push((x + 1) / 2, (1 - y) / 2); col.push(1, 1, 1, q.alpha ?? 1);
    }
    idx.push(b, b + 1, b + 2, b, b + 2, b + 3);
    tm.push(q.mat, q.mat); tf.push(q.flags ?? 0, q.flags ?? 0);
  }
  const P = Float32Array.from(pos);
  const mn = [Infinity, Infinity, Infinity]; const mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < P.length; i += 3) for (let c = 0; c < 3; c++) { mn[c] = Math.min(mn[c], P[i + c]); mx[c] = Math.max(mx[c], P[i + c]); }
  return {
    name: 'quads',
    geometry: { positions: P, normals: Float32Array.from(nrm), tangents: new Float32Array(P.length / 3 * 4), uv0: Float32Array.from(uv), color0: Float32Array.from(col),
      indices: Uint32Array.from(idx), triMaterial: Uint32Array.from(tm), triFlags: Uint32Array.from(tf) },
    materials, textures, lights: [], cameras: [],
    bounds: { min: mn as [number, number, number], max: mx as [number, number, number] }, warnings: [],
  };
}

interface Rig {
  renderer: Renderer; debug: DebugResources; fu: FrameUniformBuffer; color: GPUTexture; depth: GPUTexture;
  W: number; H: number; origin: [number, number, number];
}

async function rig(scene: SceneData, W: number, H: number, watertight = false): Promise<Rig> {
  const { device, features, wgslLanguageFeatures } = await getTestGpu();
  const debug = new DebugResources(device);
  await debug.init();
  debug.resize(W, H);
  debug.update(defaultDebugSettings(), 0);
  const fu = new FrameUniformBuffer(device);
  const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
  const color = device.createTexture({ size: [W, H], format: 'rgba32float', usage });
  const depth = device.createTexture({ size: [W, H], format: 'r32float', usage });
  const renderer = await Renderer.create({
    device, debugLayout: debug.layout, features, wgslLanguageFeatures,
    buildBvh: async (p, i) => buildBvh(p, i, { mt: true, woop: true }),
  }, { watertight });
  renderer.resize({ width: W, height: H, color, colorFormat: 'rgba32float', depth, frameUniforms: fu.buffer });
  const origin = computeRenderOrigin(scene.bounds);
  await renderer.setScene(scene, origin);
  return { renderer, debug, fu, color, depth, W, H, origin };
}

async function frame(r: Rig, cam: ArrayLike<number>, prev: ArrayLike<number>, yfov: number, flags = FRAME_RESET_HISTORY,
  seed: Partial<Pick<FrameUniformInput, 'jitterMode' | 'seedIndex' | 'runSeed'>> = {}) {
  const { device } = await getTestGpu();
  const u: FrameUniformInput = {
    camera: { camToWorld: Array.from(cam), yfov }, prevCamera: { camToWorld: Array.from(prev), yfov }, width: r.W, height: r.H,
    frameIndex: 0, seedIndex: 0, runSeed: 1, flags, jitterMode: JITTER_NONE, jitter: [0.5, 0.5], origin: r.origin,
    exposure: 1, time: 0, dt: 0, sceneDiag: 1, ...seed,
  };
  r.fu.write(u);
  const enc = device.createCommandEncoder();
  r.debug.beginFrame(enc);
  expect(r.renderer.encode(enc, { advanced: true, debugMode: 0, debugGroup: r.debug.bindGroup })).toBe(true);
  device.queue.submit([enc.finish()]);
  const n = r.W * r.H;
  const gb = await readBuffer(device, r.renderer.gbuffer!, n * GBUF_TEXEL_BYTES);
  const counters = new Uint32Array(await readBuffer(device, r.debug.buffer, DEBUG_BUFFER_LAYOUT.counterCount * 4));
  const texel = async (tex: GPUTexture, bpp: number) => {
    const row = Math.ceil((r.W * bpp) / 256) * 256;
    const buf = device.createBuffer({ size: row * r.H, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const e = device.createCommandEncoder();
    e.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: row }, [r.W, r.H]);
    device.queue.submit([e.finish()]);
    const padded = new Uint8Array(await readBuffer(device, buf, row * r.H));
    buf.destroy();
    const out = new Uint8Array(n * bpp);
    for (let y = 0; y < r.H; y++) out.set(padded.subarray(y * row, y * row + r.W * bpp), y * r.W * bpp);
    return out.buffer;
  };
  return {
    g: new Float32Array(gb), gu: new Uint32Array(gb), vb: new Uint32Array(await texel(r.renderer.vbuffer!, 16)),
    color: new Float32Array(await texel(r.color, 16)), depth: new Float32Array(await texel(r.depth, 4)), counters,
  };
}

/** f64 camera ray per math.md#raster (pixel centre), in the recentred frame. */
function cameraRay(c2wWorld: ArrayLike<number>, origin: number[], yfov: number, W: number, H: number, c: number, r: number) {
  const m = recentre(c2wWorld, origin);
  const ty = Math.tan(yfov / 2), tx = ty * W / H;
  const dc = [(2 * (c + 0.5) / W - 1) * tx, (2 * (H - 1 - r + 0.5) / H - 1) * ty, -1];
  const d = [0, 1, 2].map((i) => m[i] * dc[0] + m[4 + i] * dc[1] + m[8 + i] * dc[2]);
  const l = Math.hypot(...d);
  return { o: [m[12], m[13], m[14]], d: d.map((x) => x / l), m };
}

const G = 20; // floats per GBufTexel: ng 0, thr 3, ns 4, viewZ 7, pos 8, matId 11, albedo 12, flags 15, motion 16

describe('primary pass', () => {
  for (const watertight of [false, true]) {
    it(`Cornell (${watertight ? 'Woop' : 'MT'}): V-buffer = f64 brute force, depth, oriented normals, thr, counters`, async () => {
      const bytes = await readAsset('validation/assets/cornell/cornell.glb');
      expect(bytes).toBeDefined();
      const { scene } = await loadGltf({ kind: 'glb', bytes: new Uint8Array(bytes!), name: 'cornell.glb' }, { tangents: false });
      const W = 96, H = 72;
      const r = await rig(scene, W, H, watertight);
      const cam = scene.cameras[0];
      const out = await frame(r, cam.matrix, cam.matrix, cam.yfov);
      const pos = recentrePositions(scene.geometry.positions, r.origin);
      let mismatch = 0, hits = 0, maxDepthErr = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const { o, d, m } = cameraRay(cam.matrix, r.origin, cam.yfov, W, H, x, y);
        const ref = bruteClosest(pos, scene.geometry.indices, o, d);
        const prim = out.vb[4 * i];
        if (prim !== ref.primId) {
          // Accept only coplanar ties at equal t (shared edges of the box quads).
          const other = prim === 0xffffffff ? Infinity : Math.hypot(out.g[G * i + 8] - o[0], out.g[G * i + 9] - o[1], out.g[G * i + 10] - o[2]);
          if (!(Math.abs(other - ref.t) <= 1e-5 * ref.t)) mismatch++;
          continue;
        }
        if (prim === 0xffffffff) continue;
        hits++;
        // barycentrics, depth, normals
        expect(Math.abs(new Float32Array(out.vb.buffer)[4 * i + 1] - ref.u)).toBeLessThan(1e-4);
        const viewZ = ref.t * -(d[0] * m[8] + d[1] * m[9] + d[2] * m[10]);
        maxDepthErr = Math.max(maxDepthErr, Math.abs(out.g[G * i + 7] - viewZ) / viewZ);
        expect(Math.abs(out.depth[i] - out.g[G * i + 7])).toBe(0);
        const ng = [out.g[G * i], out.g[G * i + 1], out.g[G * i + 2]];
        const ns = [out.g[G * i + 4], out.g[G * i + 5], out.g[G * i + 6]];
        expect(ng[0] * d[0] + ng[1] * d[1] + ng[2] * d[2]).toBeLessThanOrEqual(0);          // toward the camera
        expect(Math.abs(Math.hypot(...ns) - 1)).toBeLessThan(1e-5);
        expect(ns[0] * ng[0] + ns[1] * ng[1] + ns[2] * ng[2]).toBeGreaterThan(0.999);        // flat-shaded box
        const cosT = Math.abs(ng[0] * d[0] + ng[1] * d[1] + ng[2] * d[2]);
        const thr = 2e-4 * ref.t * ref.t * 4 * Math.PI / Math.max(cosT, 1e-6);
        expect(Math.abs(out.g[G * i + 3] - thr) / thr).toBeLessThan(1e-3);
        expect(out.gu[G * i + 11]).toBe(scene.geometry.triMaterial[prim]);
        // placeholder beauty = albedo = base colour factor (no textures, no COLOR_0)
        const bc = scene.materials[scene.geometry.triMaterial[prim]].baseColorFactor;
        for (let c = 0; c < 3; c++) expect(Math.abs(out.color[4 * i + c] - bc[c])).toBeLessThan(1e-6);
      }
      console.log(`PRIMARY_CORNELL ${watertight ? 'woop' : 'mt'}: ${hits} hits / ${W * H}, mismatches ${mismatch}, max rel depth err ${maxDepthErr.toExponential(2)}`);
      expect(hits).toBeGreaterThan(W * H * 0.5);
      expect(mismatch).toBe(0);
      expect(maxDepthErr).toBeLessThan(1e-4);
      for (const k of [DBGC.NAN, DBGC.INF, DBGC.BVH_OVERFLOW, DBGC.BVH_ITERCAP]) expect(out.counters[k]).toBe(0);
      r.renderer.destroy();
    });
  }

  it('alpha MASK cutout: baseColorFactor.a and COLOR_0.a against alphaCutoff; opaque back quad', async () => {
    const cam = lookAt([0, 0, 2], [0, 0, 0]);
    const yfov = 40 * Math.PI / 180;
    const W = 32, H = 32;
    const centre = (H / 2) * W + W / 2;
    const cases: [string, number, number, number, number?][] = [
      // name, factor alpha, vertex alpha, expected front (1) / back (0), baseColorTexture alpha byte (1x1, white)
      ['factor 0.3', 0.3, 1, 0], ['factor 0.7', 0.7, 1, 1], ['factor 0.5 = cutoff', 0.5, 1, 1], ['COLOR_0 0.3', 1, 0.3, 0], ['0.8 x 0.6 = 0.48', 0.8, 0.6, 0],
      ['texture a 64/255', 1, 1, 0, 64], ['texture a 192/255', 1, 1, 1, 192], ['texture 192/255 x factor 0.6 = 0.45', 0.6, 1, 0, 192],
    ];
    for (const [name, fa, va, front, ta] of cases) {
      const tex: TextureData[] = ta === undefined ? [] : [{ name: 't', width: 1, height: 1, pixels: Uint8Array.from([255, 255, 255, ta]), wrapS: 'repeat', wrapT: 'repeat', filter: 'linear' }];
      const s = quads([{ z: 0, half: 0.5, mat: 0, flags: TRI_ALPHA_MASK, alpha: va }, { z: -1, half: 2, mat: 1 }],
        [material({ alphaMode: 'MASK', alphaCutoff: 0.5, baseColorFactor: [1, 0, 0, fa], ...(ta === undefined ? {} : { baseColorTexture: { texture: 0, texCoord: 0 } }) }),
          material({ baseColorFactor: [0, 0, 1, 1] })], tex);
      const r = await rig(s, W, H);
      const out = await frame(r, cam, cam, yfov);
      const prim = out.vb[4 * centre];
      expect(prim < 2 ? 1 : 0, name).toBe(front);
      expect(out.color[4 * centre + (front ? 0 : 2)], name).toBeCloseTo(1, 6);
      r.renderer.destroy();
    }
  });

  it('miss: primId NONE + env background (strength·tint), visibleToCamera; motion vectors; accumulation', async () => {
    const s = quads([{ z: 0, half: 0.1, mat: 0 }], [material({ baseColorFactor: [0.5, 0.5, 0.5, 1] })]);
    const W = 64, H = 48;
    const r = await rig(s, W, H);
    const env: EnvironmentData = {
      name: 'const', width: 8, height: 4, texels: new Float32Array(8 * 4 * 4).fill(2), strength: 0.5, tint: [1, 0.5, 0.25], rotationZ: 0.3, visibleToCamera: true,
    };
    await r.renderer.setEnvironment(env);
    const yfov = 50 * Math.PI / 180;
    const cam = lookAt([0.05, 0.02, 1], [0, 0, 0]);
    let out = await frame(r, cam, cam, yfov);
    const corner = 0;
    expect(out.vb[4 * corner]).toBe(0xffffffff);
    expect(out.depth[corner]).toBe(0);
    [1, 0.5, 0.25].forEach((v, k) => expect(out.color[k]).toBeCloseTo(v, 6));
    // hit pixel near the centre: motion vs an f64 reprojection with a translated previous camera
    const prev = Float64Array.from(cam); prev[12] += 0.02; prev[13] -= 0.01;
    out = await frame(r, cam, prev, yfov, 0);
    const c = W / 2, rr = H / 2, i = rr * W + c;
    expect(out.vb[4 * i]).not.toBe(0xffffffff);
    const p = [out.g[G * i + 8], out.g[G * i + 9], out.g[G * i + 10]];
    const proj = (m: ArrayLike<number>) => {
      const w2c = rigidInverse(recentre(m, r.origin));
      const pc = [0, 1, 2].map((k) => w2c[k] * p[0] + w2c[4 + k] * p[1] + w2c[8 + k] * p[2] + w2c[12 + k]);
      const t = Math.tan(yfov / 2);
      return [(pc[0] / (-pc[2] * t * W / H) + 1) * 0.5 * W, H - (pc[1] / (-pc[2] * t) + 1) * 0.5 * H];
    };
    const [cx, cy] = proj(cam), [px, py] = proj(prev);
    expect(Math.abs(out.g[G * i + 16] - (px - cx))).toBeLessThan(2e-3);
    expect(Math.abs(out.g[G * i + 17] - (py - cy))).toBeLessThan(2e-3);
    expect(out.gu[G * i + 15] & 5).toBe(5); // GB_HIT | GB_MOTION_VALID
    expect(Math.abs(px - cx)).toBeGreaterThan(0.1);
    // accumulation (camera unchanged from the previous frame's cur -> no restart): mean of identical samples
    out = await frame(r, cam, cam, yfov, 0);
    expect(out.color[4 * i]).toBeCloseTo(0.5, 6);
    r.renderer.setEnvParams({ visibleToCamera: false });
    out = await frame(r, cam, cam, yfov);
    expect(Array.from(out.color.subarray(0, 3))).toEqual([0, 0, 0]);
    for (const k of [DBGC.NAN, DBGC.INF, DBGC.BVH_OVERFLOW, DBGC.BVH_ITERCAP]) expect(out.counters[k]).toBe(0);
    r.renderer.destroy();
  });
  it('alpha MASK applies to every trace: trace_any, visible() and visibleInf() through scene-data alpha_pass (MT and Woop)', async () => {
    const { device, features, wgslLanguageFeatures } = await getTestGpu();
    const KERNEL = `
#include "scene/scene-data.wgsl"
#include "geom/visible.wgsl"
@group(1) @binding(0) var<storage, read_write> outp: array<vec4u>;
@compute @workgroup_size(1)
fn main() {
  let o = vec3f(0.0, 0.0, 1.0);
  let d = vec3f(0.0, 0.0, -1.0);
  let z = vec3f(0.0);
  outp[0] = vec4u(
    select(0u, 1u, trace_any(o, d, 1.5)),                                               // front quad only (t = 1)
    select(0u, 1u, visible(o, z, BVH_MISS, vec3f(0.0, 0.0, -0.5), z, BVH_MISS)),        // segment through the front quad
    select(0u, 1u, visibleInf(vec3f(0.0, 0.0, -0.5), z, BVH_MISS, vec3f(0.0, 0.0, 1.0))), // to infinity through it
    trace_closest(o, d, FLT_MAX).primId);
}`;
    const cases: [string, number, number | undefined, boolean][] = [
      ['factor 0.3', 0.3, undefined, false], ['factor 0.7', 0.7, undefined, true], ['texture a 64/255', 1, 64, false], ['texture a 192/255', 1, 192, true],
    ];
    for (const watertight of [false, true]) for (const [name, fa, ta, kept] of cases) {
      const tex: TextureData[] = ta === undefined ? [] : [{ name: 't', width: 1, height: 1, pixels: Uint8Array.from([255, 255, 255, ta]), wrapS: 'repeat', wrapT: 'repeat', filter: 'linear' }];
      const s = quads([{ z: 0, half: 0.5, mat: 0, flags: TRI_ALPHA_MASK }, { z: -1, half: 2, mat: 1 }],
        [material({ alphaMode: 'MASK', baseColorFactor: [1, 0, 0, fa], ...(ta === undefined ? {} : { baseColorTexture: { texture: 0, texCoord: 0 } }) }), material()], tex);
      const gpu = await SceneGpu.create(device, s, [0, 0, 0], { textureMode: 'validation', watertight, buildBvh: async (p, i) => buildBvh(p, i, { mt: true, woop: true }), features, wgslLanguageFeatures });
      const shader = composeWgsl('tests/alpha-any.wgsl', { sources: { ...shaderSources, 'tests/alpha-any.wgsl': KERNEL }, defines: gpu.defines(0), features, wgslLanguageFeatures });
      const module = await createCheckedShaderModule(device, shader, 'alpha-any');
      const l0 = device.createBindGroupLayout({ entries: gpu.layoutEntries(GPUShaderStage.COMPUTE) });
      const l1 = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] });
      const pipe = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [l0, l1] }), compute: { module, entryPoint: 'main' } });
      const out = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, device.createBindGroup({ layout: l0, entries: gpu.bindGroupEntries() }));
      pass.setBindGroup(1, device.createBindGroup({ layout: l1, entries: [{ binding: 0, resource: { buffer: out } }] }));
      pass.dispatchWorkgroups(1);
      pass.end();
      device.queue.submit([enc.finish()]);
      const [occ, vis, visInf, prim] = new Uint32Array(await readBuffer(device, out, 16));
      const label = `${name} ${watertight ? 'woop' : 'mt'}`;
      expect(occ, `${label} trace_any`).toBe(kept ? 1 : 0);
      expect(vis, `${label} visible`).toBe(kept ? 0 : 1);
      expect(visInf, `${label} visibleInf`).toBe(kept ? 0 : 1);
      expect(prim < 2, `${label} trace_closest`).toBe(kept);
      out.destroy();
      gpu.destroy();
    }
  });

  it('JITTER_IID: in-pixel, uniform, i.i.d. across frames and runs, reproducible for equal (runSeed, frame)', async () => {
    // A screen-filling plane: the hit position is the jittered camera ray, so the subpixel offset is recoverable.
    const s = quads([{ z: 0, half: 50, mat: 0 }], [material()]);
    const W = 32, H = 24;
    const r = await rig(s, W, H);
    const yfov = 40 * Math.PI / 180, t = Math.tan(yfov / 2);
    const cam = lookAt([0, 0, 1], [0, 0, -1]);
    const jitter = async (runSeed: number, seedIndex: number) => {
      const out = await frame(r, cam, cam, yfov, FRAME_RESET_HISTORY, { jitterMode: JITTER_IID, runSeed, seedIndex });
      const j = new Float64Array(2 * W * H);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const p = [out.g[G * i + 8] + r.origin[0], out.g[G * i + 9] + r.origin[1], out.g[G * i + 10] + r.origin[2]];
        const zc = p[2] - 1; // camera at z = 1 looking down -Z (camera space = world - eye)
        const col = (p[0] / (-zc * t * W / H) + 1) * 0.5 * W;
        const rowTop = H - (p[1] / (-zc * t) + 1) * 0.5 * H;
        j[2 * i] = col - x;               // u (math.md#raster)
        j[2 * i + 1] = 1 - (rowTop - y);  // v: raster y grows upward
      }
      return j;
    };
    const a = await jitter(7, 0), b = await jitter(7, 1), c = await jitter(8, 0), a2 = await jitter(7, 0);
    for (const j of [a, b, c]) {
      const n = j.length;
      let mean = 0, m2 = 0, lo = Infinity, hi = -Infinity;
      for (const v of j) { mean += v / n; lo = Math.min(lo, v); hi = Math.max(hi, v); }
      for (const v of j) m2 += (v - mean) ** 2 / n;
      expect(lo).toBeGreaterThan(-2e-3);
      expect(hi).toBeLessThan(1 + 2e-3);
      expect(Math.abs(mean - 0.5)).toBeLessThan(0.04);          // 1536 samples: SE ≈ 0.0074
      expect(Math.abs(m2 - 1 / 12)).toBeLessThan(0.012);
    }
    expect(Array.from(a2)).toEqual(Array.from(a));               // same run seed + frame → identical
    const corr = (x: Float64Array, y: Float64Array) => {
      let sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0; const n = x.length;
      for (let i = 0; i < n; i++) { sx += x[i]; sy += y[i]; sxy += x[i] * y[i]; sxx += x[i] * x[i]; syy += y[i] * y[i]; }
      return (sxy - sx * sy / n) / Math.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n));
    };
    expect(Math.abs(corr(a, b))).toBeLessThan(0.1);              // next frame: fresh jitter
    expect(Math.abs(corr(a, c))).toBeLessThan(0.1);              // other run: fresh jitter
    r.renderer.destroy();
  });
});

// Shared fixtures for the PT / light GPU tests (pt.gpu.test.ts, lights.gpu.test.ts): quad scenes, a PT rig on top of
// BatchAccumulator, f64 reference helpers (camera rays, polygon irradiance).
import { buildBvh } from '../../src/core/bvh/sah-builder.ts';
import { BatchAccumulator } from '../../src/core/render/batch-accumulator.ts';
import { createEnvResources, destroyEnvResources } from '../../src/core/render/env-gpu.ts';
import { computeRenderOrigin, type JitterMode } from '../../src/core/render/frame-uniforms.ts';
import { PtKernel, type PtKernelOptions } from '../../src/core/render/pt-kernel.ts';
import { SceneGpu } from '../../src/core/render/scene-gpu.ts';
import type { LightData, MaterialData, SceneData } from '../../src/core/scene/types.ts';
import { TRI_EMISSIVE } from '../../src/core/scene/types.ts';
import { getTestGpu } from './device-factory.ts';

export type V3 = [number, number, number];

export function material(o: Partial<MaterialData> = {}): MaterialData {
  return {
    name: 'm', baseColorFactor: [0.5, 0.5, 0.5, 1], metallicFactor: 0, roughnessFactor: 1, emissiveFactor: [0, 0, 0], emissiveStrength: 1,
    ior: 1.5, specularFactor: 0, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5,
    doubleSided: true, model: 'v1', v1: { diffuse: [0.5, 0.5, 0.5], glossy: [0, 0, 0], roughness: 0.5, mix: 0 }, ...o,
  };
}

export const lambert = (rho: number | V3, o: Partial<MaterialData> = {}): MaterialData => {
  const r: V3 = typeof rho === 'number' ? [rho, rho, rho] : rho;
  return material({ baseColorFactor: [...r, 1], v1: { diffuse: r, glossy: [0, 0, 0], roughness: 0.5, mix: 0 }, ...o });
};

/** Quads given by 4 corners each (normal = (p1−p0)×(p3−p0)); material index per quad. */
export function quadScene(quads: { p: number[][]; mat: number }[], materials: MaterialData[], lights: LightData[] = []): SceneData {
  const pos: number[] = [], nrm: number[] = [], idx: number[] = [], tm: number[] = [], tf: number[] = [];
  for (const q of quads) {
    const b = pos.length / 3;
    const e1 = q.p[1].map((x, k) => x - q.p[0][k]), e2 = q.p[3].map((x, k) => x - q.p[0][k]);
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const l = Math.hypot(...n);
    for (const v of q.p) { pos.push(...v); nrm.push(...n.map((x) => x / l)); }
    idx.push(b, b + 1, b + 2, b, b + 2, b + 3);
    const m = materials[q.mat];
    const f = Math.max(...m.emissiveFactor) * m.emissiveStrength > 0 ? TRI_EMISSIVE : 0;
    tm.push(q.mat, q.mat); tf.push(f, f);
  }
  const P = Float32Array.from(pos);
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < P.length; i += 3) for (let c = 0; c < 3; c++) { mn[c] = Math.min(mn[c], P[i + c]); mx[c] = Math.max(mx[c], P[i + c]); }
  return {
    name: 'quads',
    geometry: {
      positions: P, normals: Float32Array.from(nrm), tangents: new Float32Array(P.length / 3 * 4), uv0: new Float32Array(P.length / 3 * 2),
      indices: Uint32Array.from(idx), triMaterial: Uint32Array.from(tm), triFlags: Uint32Array.from(tf),
    },
    materials, textures: [], lights, cameras: [],
    bounds: { min: mn as V3, max: mx as V3 }, warnings: [],
  };
}

/** Ground plane y = 0, half size S, normal +Y. */
export const groundQuad = (S: number, mat = 0) => ({ p: [[-S, 0, S], [S, 0, S], [S, 0, -S], [-S, 0, -S]], mat });

/** Camera at `pos` looking straight down (−Y), image up = world −Z. Column-major camToWorld. */
export const lookDown = (pos: V3): number[] => [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, pos[0], pos[1], pos[2], 1];

/** Column-major rigid matrix from axes X, Y, Z (unit, right-handed) and position. */
export const frameMatrix = (X: V3, Y: V3, Z: V3, p: V3): Float32Array => new Float32Array([...X, 0, ...Y, 0, ...Z, 0, ...p, 1]);

/** A light whose emission axis a_L = −Z_obj points along world direction `aL` (unit). */
export function lightMatrixToward(aL: V3, p: V3): Float32Array {
  const Z: V3 = [-aL[0], -aL[1], -aL[2]];
  const helper: V3 = Math.abs(Z[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const X = norm(cross(helper, Z));
  const Y = cross(Z, X);
  return frameMatrix(X, Y, Z, p);
}

export const cross = (a: readonly number[], b: readonly number[]): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const dot = (a: readonly number[], b: readonly number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const sub = (a: readonly number[], b: readonly number[]): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const norm = (a: readonly number[]): V3 => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };

/** f64 camera ray for pixel (c, r) and subpixel (u, v) (math.md#raster). */
export function cameraRay(camToWorld: number[], yfov: number, W: number, H: number, c: number, r: number, u = 0.5, v = 0.5): { o: V3; d: V3 } {
  const ty = Math.tan(yfov / 2), tx = ty * W / H;
  const dc = [(2 * (c + u) / W - 1) * tx, (2 * (H - 1 - r + v) / H - 1) * ty, -1];
  const m = camToWorld;
  const d = norm([m[0] * dc[0] + m[4] * dc[1] + m[8] * dc[2], m[1] * dc[0] + m[5] * dc[1] + m[9] * dc[2], m[2] * dc[0] + m[6] * dc[1] + m[10] * dc[2]]);
  return { o: [m[12], m[13], m[14]], d };
}

/** Hit point of a camera ray with the plane y = 0. */
export function hitGround(ray: { o: V3; d: V3 }): V3 {
  const t = -ray.o[1] / ray.d[1];
  return [ray.o[0] + t * ray.d[0], 0, ray.o[2] + t * ray.d[2]];
}

/**
 * Irradiance at x (normal n) from a Lambertian polygon of radiance 1 fully above the horizon (the polygon formula,
 * validation-harness §8 C0e): E = ½ |Σ_i acos(û_i·û_{i+1}) · n·normalize(û_i × û_{i+1})|.
 */
export function polygonIrradiance(x: V3, n: V3, verts: V3[]): number {
  let s = 0;
  for (let i = 0; i < verts.length; i++) {
    const a = norm(sub(verts[i], x)), b = norm(sub(verts[(i + 1) % verts.length], x));
    const c = cross(a, b);
    const cl = Math.hypot(...c);
    if (cl === 0) continue;
    s += Math.acos(Math.max(-1, Math.min(1, dot(a, b)))) * dot(n, c) / cl;
  }
  return 0.5 * Math.abs(s);
}

export interface PtRig { kernel: PtKernel; acc: BatchAccumulator; W: number; H: number; destroy(): void }

export async function ptRig(scene: SceneData, W: number, H: number, cam: { camToWorld: number[]; yfov: number },
  opts: PtKernelOptions & { seed?: number; jitterMode?: JitterMode; budget?: ConstructorParameters<typeof BatchAccumulator>[3] } = {}): Promise<PtRig> {
  const { device, features, wgslLanguageFeatures } = await getTestGpu();
  const origin = computeRenderOrigin(scene.bounds);
  const gpu = await SceneGpu.create(device, scene, origin, {
    textureMode: 'validation', watertight: true, buildBvh: async (p, i) => buildBvh(p, i, { mt: true, woop: true }), features, wgslLanguageFeatures,
  });
  const env = await createEnvResources(device, scene.env);
  const kernel = await PtKernel.create(device, gpu, env, { features, wgslLanguageFeatures, ...opts });
  kernel.setView({ camera: cam, width: W, height: H, runSeed: opts.seed ?? 7, jitterMode: opts.jitterMode });
  const acc = new BatchAccumulator(device, W, H, opts.budget);
  return { kernel, acc, W, H, destroy: () => { kernel.destroy(); acc.destroy(); gpu.destroy(); destroyEnvResources(env); } };
}

export const runPt = (r: PtRig, spp: number, index?: number) => r.acc.runBatch((enc, d, a, c) => r.kernel.encode(enc, d, a, c), spp, index);

/** Mean and standard error (over batches) of `f(batchMean)` for B batches. */
export async function batchStats(r: PtRig, B: number, spp: number, f: (mean: Float32Array) => number[]): Promise<{ mean: number[]; se: number[]; counters: number[] }> {
  const vals: number[][] = [];
  const counters = [0, 0, 0, 0];
  for (let b = 0; b < B; b++) {
    const res = await runPt(r, spp, b);
    res.counters.forEach((c, i) => { counters[i] += c; });
    vals.push(f(res.mean));
  }
  const k = vals[0].length;
  const mean = Array.from({ length: k }, (_, j) => vals.reduce((s, v) => s + v[j], 0) / B);
  const se = mean.map((m, j) => Math.sqrt(vals.reduce((s, v) => s + (v[j] - m) ** 2, 0) / (B - 1) / B));
  return { mean, se, counters };
}

export const pixel = (mean: Float32Array, W: number, c: number, r: number): V3 => [mean[3 * (r * W + c)], mean[3 * (r * W + c) + 1], mean[3 * (r * W + c) + 2]];

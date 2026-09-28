// Emission-only kernel (plan §5 M2; emission.wgsl) + BatchAccumulator, both GPU lanes:
// - C0b-like: an emissive quad covering the view → every pixel = L_e exactly (strength·color, two-sided);
// - camera-visible rect light in front of a constant env → L_rect + L_env inside (pass-through, weight 1),
//   L_env outside, one-sided (a light facing away is invisible), spread < π attenuation vs an f64 reference;
// - Cornell box interior (non-emissive walls) + emissive ceiling patch: walls 0, emitter L_e, no NaN/BVH overflow;
// - sub-submit row bands and k-sample dispatches give bit-identical batches (submit-budget splitting is exact).
import { afterAll, describe, expect, it } from 'vitest';
import { getTestGpu, releaseTestGpu } from './device-factory.ts';
import { buildBvh } from '../../src/core/bvh/sah-builder.ts';
import { BatchAccumulator } from '../../src/core/render/batch-accumulator.ts';
import { EmissionKernel, JITTER_NONE, spreadNormalization } from '../../src/core/render/emission-kernel.ts';
import { createEnvResources, destroyEnvResources } from '../../src/core/render/env-gpu.ts';
import { computeRenderOrigin } from '../../src/core/render/frame-uniforms.ts';
import { SceneGpu, emptyScene } from '../../src/core/render/scene-gpu.ts';
import type { EnvironmentData, LightData, MaterialData, SceneData } from '../../src/core/scene/types.ts';
import { TRI_EMISSIVE } from '../../src/core/scene/types.ts';

afterAll(releaseTestGpu);

const I4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function material(o: Partial<MaterialData> = {}): MaterialData {
  return {
    name: 'm', baseColorFactor: [0.8, 0.8, 0.8, 1], metallicFactor: 0, roughnessFactor: 0.5, emissiveFactor: [0, 0, 0], emissiveStrength: 1,
    ior: 1.5, specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5,
    doubleSided: true, model: 'v1', v1: { diffuse: [0.5, 0.5, 0.5], glossy: [0, 0, 0], roughness: 0.5, mix: 0 }, ...o,
  };
}

/** Quads given by 4 corners each (CCW = front); material index per quad. */
function quadScene(quads: { p: number[][]; mat: number }[], materials: MaterialData[]): SceneData {
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
    geometry: { positions: P, normals: Float32Array.from(nrm), tangents: new Float32Array(P.length / 3 * 4), uv0: new Float32Array(P.length / 3 * 2),
      indices: Uint32Array.from(idx), triMaterial: Uint32Array.from(tm), triFlags: Uint32Array.from(tf) },
    materials, textures: [], lights: [], cameras: [],
    bounds: { min: mn as [number, number, number], max: mx as [number, number, number] }, warnings: [],
  };
}

function constEnv(v: [number, number, number], o: Partial<EnvironmentData> = {}): EnvironmentData {
  const W = 8, H = 4;
  const t = new Float32Array(W * H * 4);
  for (let i = 0; i < W * H; i++) t.set([...v, 1], 4 * i);
  return { name: 'const', width: W, height: H, texels: t, strength: 2, tint: [1, 0.5, 1], rotationZ: 0.3, visibleToCamera: true, ...o };
}

interface Rig { kernel: EmissionKernel; acc: BatchAccumulator; W: number; H: number; destroy(): void }

async function rig(scene: SceneData, W: number, H: number, cam: { camToWorld: number[]; yfov: number }, seed = 7,
  jitterMode?: typeof JITTER_NONE, budget?: ConstructorParameters<typeof BatchAccumulator>[3]): Promise<Rig> {
  const { device, features, wgslLanguageFeatures } = await getTestGpu();
  const origin = computeRenderOrigin(scene.bounds);
  const gpu = await SceneGpu.create(device, scene, origin, {
    textureMode: 'validation', watertight: true, buildBvh: async (p, i) => buildBvh(p, i, { mt: true, woop: true }), features, wgslLanguageFeatures,
  });
  const env = await createEnvResources(device, scene.env);
  const kernel = await EmissionKernel.create(device, gpu, env, { features, wgslLanguageFeatures });
  kernel.setView({ camera: cam, width: W, height: H, runSeed: seed, jitterMode });
  const acc = new BatchAccumulator(device, W, H, budget);
  return { kernel, acc, W, H, destroy: () => { kernel.destroy(); acc.destroy(); gpu.destroy(); destroyEnvResources(env); } };
}

const run = (r: Rig, spp: number, index?: number) =>
  r.acc.runBatch((enc, d, a, c) => r.kernel.encode(enc, d, a, c), spp, index);

describe('emission kernel', () => {
  it('C0b-like: emissive quad covering the view → every pixel = L_e exactly, zero counters', async () => {
    // L_e = (0.5, 0.25, 0.125)·2 = (1, 0.5, 0.25); quad faces AWAY from the camera to exercise two-sidedness.
    const S = 50, D = 2;
    const scene = quadScene([{ p: [[-S, -S, -D], [-S, S, -D], [S, S, -D], [S, -S, -D]], mat: 0 }],
      [material({ emissiveFactor: [0.5, 0.25, 0.125], emissiveStrength: 2 })]);
    const r = await rig(scene, 64, 48, { camToWorld: I4, yfov: 40 * Math.PI / 180 });
    const b = await run(r, 16);
    expect(b.counters.slice(0, 3)).toEqual([0, 0, 0]);
    const bad: number[] = [];
    for (let i = 0; i < 64 * 48; i++) if (b.mean[3 * i] !== 1 || b.mean[3 * i + 1] !== 0.5 || b.mean[3 * i + 2] !== 0.25) bad.push(i);
    expect(bad.length, `first bad pixel ${bad[0]}`).toBe(0);
    r.destroy();
  });

  it('camera-visible rect light in front of the env: L_rect + L_env inside (pass-through), L_env outside, one-sided', async () => {
    const W = 64, H = 64, yfov = Math.PI / 3; // tan(30°): the rect (half size 1 at z = −3) spans image x ∈ [13.5, 50.5]
    const Lenv = [0.3 * 2 * 1, 0.6 * 2 * 0.5, 0.9 * 2 * 1];
    const rect = (facing: 1 | -1, o: Partial<LightData> = {}): LightData => ({
      id: 0, name: 'rect', type: 'rect', color: [1, 0.5, 0.25], power: 10, exposure: 1,
      // emission axis −Z_obj: facing +1 → Z_obj = +Z (emits toward −Z, away from the camera at the origin)... so use −facing
      matrix: new Float32Array([1, 0, 0, 0, 0, facing, 0, 0, 0, 0, facing, 0, 0, 0, -3, 1]), sizeX: 2, sizeY: 2, spread: Math.PI,
      visibleToCamera: true, ...o,
    });
    const Lrect = [1, 0.5, 0.25].map((c) => c * 10 * 2 / (Math.PI * 4));
    const check = async (light: LightData, inside: number[]) => {
      const scene = { ...emptyScene('env'), env: constEnv([0.3, 0.6, 0.9]), lights: [light] };
      const r = await rig(scene, W, H, { camToWorld: I4, yfov });
      expect(r.kernel.cameraVisibleLights).toBe(1);
      const b = await run(r, 8);
      expect(b.counters.slice(0, 3)).toEqual([0, 0, 0]);
      const px = (c: number, row: number) => Array.from(b.mean.subarray(3 * (row * W + c), 3 * (row * W + c) + 3));
      for (const [c, row] of [[14, 14], [32, 32], [49, 49], [14, 49], [49, 14]]) {
        px(c, row).forEach((v, k) => expect(v, `inside ${c},${row}`).toBeCloseTo(inside[k], 5));
      }
      for (const [c, row] of [[0, 0], [12, 32], [51, 32], [32, 63], [63, 63]]) {
        px(c, row).forEach((v, k) => expect(v, `outside ${c},${row}`).toBeCloseTo(Lenv[k], 5));
      }
      r.destroy();
    };
    // Z_obj = +Z → a_L = −Z: emits away from the camera → invisible (one-sided)
    await check(rect(1), Lenv);
    // Z_obj = −Z (and Y flipped to stay right-handed... X × Y = Z: X=(1,0,0), Y=(0,−1,0) → Z=(0,0,−1)) → faces the camera
    await check(rect(-1), Lenv.map((e, k) => e + Lrect[k]));
  });

  it('spread < π: pixel-centre rays match the f64 Cycles attenuation', async () => {
    const W = 32, H = 32, yfov = Math.PI / 3, spread = Math.PI / 2;
    const light: LightData = {
      id: 0, name: 'rect', type: 'rect', color: [1, 1, 1], power: 4, exposure: 0,
      matrix: new Float32Array([1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -2, 1]), sizeX: 4, sizeY: 4, spread, visibleToCamera: true,
    };
    const scene = { ...emptyScene('spread'), lights: [light] };
    const r = await rig(scene, W, H, { camToWorld: I4, yfov }, 1, JITTER_NONE);
    const b = await run(r, 1);
    const { tanHalf, norm } = spreadNormalization(spread);
    const L0 = 4 / (Math.PI * 16);
    const ty = Math.tan(yfov / 2);
    let checked = 0;
    for (const [c, row] of [[16, 16], [5, 20], [28, 3], [10, 10]]) {
      const d = [(2 * (c + 0.5) / W - 1) * ty, (2 * (H - 1 - row + 0.5) / H - 1) * ty, -1];
      const l = Math.hypot(...d);
      const cos = -d[2] / l * 1; // a_L = (0,0,+1)·(−d) … emitted direction −d, a_L = +Z (toward the camera)
      const tanT = Math.sqrt(1 - cos * cos) / cos;
      const hitX = d[0] / -d[2] * 2, hitY = d[1] / -d[2] * 2;
      if (Math.abs(hitX) > 2 || Math.abs(hitY) > 2) continue;
      const want = L0 * Math.max((tanHalf - tanT) * norm, 0);
      expect(b.mean[3 * (row * W + c)], `${c},${row}`).toBeCloseTo(want, 5);
      checked++;
    }
    expect(checked).toBe(4);
    r.destroy();
  });

  it('Cornell-like box: emissive ceiling patch = L_e, walls 0, zero counters', async () => {
    const s = 0.2775, h = 0.555;
    const white = material(), light = material({ emissiveFactor: [17, 12, 4], emissiveStrength: 1 });
    const q = (a: number[], b: number[], c: number[], d: number[], mat = 0) => ({ p: [a, b, c, d], mat });
    const scene = quadScene([
      q([-s, 0, s], [s, 0, s], [s, 0, -s], [-s, 0, -s]),        // floor (+Y)
      q([-s, h, -s], [s, h, -s], [s, h, s], [-s, h, s]),        // ceiling (−Y)
      q([-s, 0, -s], [s, 0, -s], [s, h, -s], [-s, h, -s]),      // back wall (+Z)
      q([-s, 0, s], [-s, 0, -s], [-s, h, -s], [-s, h, s]),      // left wall
      q([s, 0, -s], [s, 0, s], [s, h, s], [s, h, -s]),          // right wall
      q([-0.065, h - 1e-3, -0.0525], [0.065, h - 1e-3, -0.0525], [0.065, h - 1e-3, 0.0525], [-0.065, h - 1e-3, 0.0525], 1), // emitter
      q([s, 0, s], [-s, 0, s], [-s, h, s], [s, h, s]),          // front wall (closes the box)
    ], [white, light]);
    // look straight up from inside the box: the ceiling fills the view, the emitter sits in the centre
    const up = [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0.1, 0, 1]; // back = −Y ⇒ camera −Z = world +Y
    const W = 48, H = 48;
    const r = await rig(scene, W, H, { camToWorld: up, yfov: Math.PI / 2 });
    const b = await run(r, 4);
    expect(b.counters.slice(0, 3)).toEqual([0, 0, 0]);
    const c = Array.from(b.mean.subarray(3 * (24 * W + 24), 3 * (24 * W + 24) + 3));
    expect(c).toEqual([17, 12, 4]);
    const corner = Array.from(b.mean.subarray(0, 3));
    expect(corner).toEqual([0, 0, 0]);
    let sum = 0;
    for (let i = 0; i < W * H; i++) sum += b.mean[3 * i];
    expect(sum).toBeGreaterThan(17 * 2);
    r.destroy();
  });

  it('row-band / k-sample sub-submits are bit-identical to single dispatches', async () => {
    // dyadic radiances: every partial sum is exact, so any split of the samples must give the same bits
    const scene = { ...quadScene([{ p: [[-1, -1, -2], [1, -1, -2], [1, 1, -2], [-1, 1, -2]], mat: 0 }], [material({ emissiveFactor: [0.5, 0.75, 1.25], emissiveStrength: 1 })]),
      env: constEnv([0.125, 0.25, 0.375]) };
    const W = 40, H = 36, cam = { camToWorld: I4, yfov: 1.2 };
    const a = await rig(scene, W, H, cam, 99);
    const whole = await run(a, 12, 3);
    const bands = await rig(scene, W, H, cam, 99, undefined, { targetMs: 1e-6 }); // forces k = 1 and 8-row bands
    const split = await run(bands, 12, 3);
    expect(split.submits).toBeGreaterThan(12 * 4);
    expect(new Uint32Array(split.mean.buffer)).toEqual(new Uint32Array(whole.mean.buffer));
    // edge pixels are partially covered: jitter really is per-sample (a proper average between env and emitter)
    const vals = new Set(Array.from(whole.mean.filter((_, i) => i % 3 === 0)).map((x) => x.toFixed(6)));
    expect(vals.size).toBeGreaterThan(2);
    a.destroy(); bands.destroy();
  });
});

// Reference PT with glass and light modes (plan §5 M3b; gap-glass §7.2 G1/G2, gap-light U9/U10), both GPU lanes.
//   - U9: Mode A ≡ B ≡ A′ on a delta-free scene (rough glass, glossy, Lambert; rect + disk lights); A′ ≡ A bitwise there.
//   - mirror positive control: a roughness-0 mirror floor under a rect light: A < B (≫ 10σ) and A′ ≡ B.
//   - U10: pass-through determinism via the PT_PROBE build (vertex/lobe hash, crossing candidates, vertex count).
//   - C0h/G1 slab: smooth glass slab in an L_e = 1 enclosure (emission_sampling NONE): per pixel
//       Principled grey C:  E = F + (1−F)·C·(1−F^N)
//       Glass node colour c: E = cF + c²(1−F)²(1−(cF)^N)/(1−cF)
//     with F = F_diel(cos θ_i, ior), N = max_bounces (f64 supersampled over the pixel footprint).
//   - C0i/G2 emitter inside a smooth glass box: E = (1−F)·√C·L_e (Principled) — detects any 1/η² factor (0.444×).
import { afterAll, describe, expect, it } from 'vitest';
import { JITTER_IID } from '../../src/core/render/frame-uniforms.ts';
import { PT_COUNTERS } from '../../src/core/render/pt-kernel.ts';
import type { LightData, MaterialData } from '../../src/core/scene/types.ts';
import { fresnelDielectric } from '../../tests/material/bsdf-ref.ts';
import { releaseTestGpu } from './device-factory.ts';
import { batchStats, lambert, lightMatrixToward, material, norm, ptRig, quadScene, runPt, type V3 } from './pt-fixtures.ts';

afterAll(releaseTestGpu);

function expectCounters(c: number[]): void {
  expect(c[PT_COUNTERS.nonFinite], 'NaN/Inf samples').toBe(0);
  expect(c[PT_COUNTERS.bvhOverflow], 'BVH overflow').toBe(0);
  expect(c[PT_COUNTERS.bvhItercap], 'BVH iteration cap').toBe(0);
  expect(c[PT_COUNTERS.negative], 'negative samples').toBe(0);
}

type Quad = { p: number[][]; mat: number };
/** Axis-aligned box faces (outward, or inward), 4 corners each (normal = (p1−p0)×(p3−p0)). */
function boxQuads(mn: V3, mx: V3, mat: number, inward = false): Quad[] {
  const [x0, y0, z0] = mn, [x1, y1, z1] = mx;
  const f: number[][][] = [
    [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
    [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]], [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
    [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
  ];
  return f.map((q) => ({ p: inward ? [q[0], q[3], q[2], q[1]] : q, mat }));
}

const emitter = (Le: number, sampling: 'NONE' | 'FRONT_BACK' = 'NONE'): MaterialData => ({
  ...material({ emissiveFactor: [Le, Le, Le], v1: { diffuse: [0, 0, 0], glossy: [0, 0, 0], roughness: 0.5, mix: 0 } }), emissionSampling: sampling,
});
const principledGlass = (C: number, r = 0, ior = 1.5): MaterialData => material({
  model: 'principled', baseColorFactor: [C, C, C, 1], roughnessFactor: r, metallicFactor: 0, transmissionFactor: 1, ior, specularFactor: 1, v1: undefined,
});
const glassNodeMat = (c: V3, r = 0, ior = 1.5): MaterialData => material({ model: 'glass', baseColorFactor: [...c, 1], roughnessFactor: r, ior, v1: undefined });

/** Camera looking at the origin from direction (sin θ, cos θ, 0)·dist; the slab is the plane y = 0. */
function slabCamera(thetaDeg: number, dist = 1): { camToWorld: number[]; yfov: number } {
  const t = thetaDeg * Math.PI / 180;
  const eye: V3 = [Math.sin(t) * dist, Math.cos(t) * dist, 0];
  const z = norm(eye);                                  // camera looks down −Z_cam = toward the origin
  const X: V3 = [0, 0, 1];                              // ⟂ z (eye.z = 0)
  const Y: V3 = [z[1] * X[2] - z[2] * X[1], z[2] * X[0] - z[0] * X[2], z[0] * X[1] - z[1] * X[0]];   // z × X
  return { camToWorld: [...X, 0, ...Y, 0, ...z, 0, ...eye, 1], yfov: 6 * Math.PI / 180 };
}

/** f64 expected pixel value: average of E(cos θ_i) over a 6×6 sub-pixel grid of the box filter. */
function slabExpected(cam: { camToWorld: number[]; yfov: number }, W: number, H: number, c: number, r: number, E: (cosI: number) => number): number {
  const m = cam.camToWorld, ty = Math.tan(cam.yfov / 2), tx = ty * W / H;
  let s = 0;
  const n = 6;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const u = (i + 0.5) / n, v = (j + 0.5) / n;
    const dc = [(2 * (c + u) / W - 1) * tx, (2 * (H - 1 - r + v) / H - 1) * ty, -1];
    const d = norm([m[0] * dc[0] + m[4] * dc[1] + m[8] * dc[2], m[1] * dc[0] + m[5] * dc[1] + m[9] * dc[2], m[2] * dc[0] + m[6] * dc[1] + m[10] * dc[2]]);
    s += E(Math.abs(d[1]));
  }
  return s / (n * n);
}

describe('PT glass: closed-form slab C0h / G1 (smooth glass in an L_e = 1 furnace, emission_sampling NONE)', () => {
  const W = 12, H = 12;
  const slab = boxQuads([-10, -0.01, -10], [10, 0.01, 10], 1);
  const enclosure = boxQuads([-30, -30, -30], [30, 30, 30], 0, true);
  const cases: { label: string; mat: MaterialData; E: (F: number, N: number) => number }[] = [
    { label: 'Principled C 0.5', mat: principledGlass(0.5), E: (F, N) => F + (1 - F) * 0.5 * (1 - F ** N) },
    { label: 'Glass node c 0.8', mat: glassNodeMat([0.8, 0.8, 0.8]), E: (F, N) => 0.8 * F + 0.64 * (1 - F) ** 2 * (1 - (0.8 * F) ** N) / (1 - 0.8 * F) },
  ];
  for (const cs of cases) {
    for (const theta of [0, 45, 75]) {
      it(`${cs.label}, θ ${theta}°, max_bounces 0/1/2/4`, async () => {
        const cam = slabCamera(theta);
        for (const N of [0, 1, 2, 4]) {
          const r = await ptRig(quadScene([...slab, ...enclosure], [emitter(1), cs.mat]), W, H, cam, { maxBounces: N, jitterMode: JITTER_IID });
          const probes = [[2, 2], [6, 6], [9, 3], [3, 10]];
          const st = await batchStats(r, 8, 256, (m) => probes.map(([c, row]) => m[3 * (row * W + c) + 1]));
          expectCounters(st.counters);
          probes.forEach(([c, row], i) => {
            const want = slabExpected(cam, W, H, c, row, (cosI) => cs.E(fresnelDielectric(cosI, 1.5), N));
            expect(Math.abs(st.mean[i] - want), `N ${N} probe ${i}: ${st.mean[i]} vs ${want} (se ${st.se[i]})`).toBeLessThan(5 * st.se[i] + 2e-4 * want);
          });
          r.destroy();
        }
      });
    }
  }
});

describe('PT glass: C0i / G2 emitter inside a smooth glass box (η² convention detector)', () => {
  it('Principled C 0.36: E = (1−F)·√C·L_e through the top face (a 1/η² factor would give 0.444×)', async () => {
    const W = 10, H = 10, C = 0.36, Le = 2;
    // glass box [−1,1]×[−0.5,0.5]×[−1,1]; an emissive quad (two-sided) inside at y = 0 facing up; black world
    const glass = boxQuads([-1, -0.5, -1], [1, 0.5, 1], 1);
    const quad: Quad = { p: [[-0.9, 0, 0.9], [0.9, 0, 0.9], [0.9, 0, -0.9], [-0.9, 0, -0.9]], mat: 0 };
    for (const theta of [0, 30]) {
      const cam = slabCamera(theta, 3);
      const r = await ptRig(quadScene([...glass, quad], [emitter(Le, 'FRONT_BACK'), principledGlass(C)]), W, H, cam, { maxBounces: 1, jitterMode: JITTER_IID });
      const probes = [[4, 4], [5, 5], [3, 6]];
      const st = await batchStats(r, 8, 256, (m) => probes.map(([c, row]) => m[3 * (row * W + c)]));
      expectCounters(st.counters);
      probes.forEach(([c, row], i) => {
        const want = slabExpected(cam, W, H, c, row, (cosI) => (1 - fresnelDielectric(cosI, 1.5)) * Math.sqrt(C) * Le);
        expect(Math.abs(st.mean[i] - want), `θ ${theta} probe ${i}: ${st.mean[i]} vs ${want} (se ${st.se[i]})`).toBeLessThan(5 * st.se[i] + 2e-4 * want);
      });
      r.destroy();
    }
  });
});

// ------------------------------------------------------------------------------------------------ light modes (M3b)

const rectToward = (id: number, aL: V3, p: V3, power: number, size = 0.6): LightData => ({
  id, name: `rect${id}`, type: 'rect', color: [1, 0.9, 0.8], power, exposure: 0, matrix: lightMatrixToward(aL, p), sizeX: size, sizeY: size * 0.7,
  spread: Math.PI, visibleToCamera: false,
});
const diskToward = (id: number, aL: V3, p: V3, power: number, spread = Math.PI / 2): LightData => ({
  id, name: `disk${id}`, type: 'disk', color: [0.8, 0.9, 1], power, exposure: 0, matrix: lightMatrixToward(aL, p), sizeX: 0.5, spread, visibleToCamera: false,
});
const v1Glossy = (r: number): MaterialData => material({ v1: { diffuse: [0.5, 0.4, 0.3], glossy: [0.9, 0.9, 0.9], roughness: r, mix: 0.5 } });
const mirror = (): MaterialData => material({ model: 'principled', baseColorFactor: [0.9, 0.9, 0.9, 1], metallicFactor: 1, roughnessFactor: 0, v1: undefined });

/** Region means (4 quadrants + whole image) of the luminance of a W×H image. */
function regionMeans(m: Float32Array, W: number, H: number): number[] {
  const s = [0, 0, 0, 0, 0], n = [0, 0, 0, 0, 0];
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
    const i = 3 * (r * W + c), y = 0.2126 * m[i] + 0.7152 * m[i + 1] + 0.0722 * m[i + 2];
    const q = (r < H / 2 ? 0 : 2) + (c < W / 2 ? 0 : 1);
    s[q] += y; n[q]++; s[4] += y; n[4]++;
  }
  return s.map((x, k) => x / n[k]);
}

/** A small box room: floor, back wall, left wall (Lambert / V1 glossy), a rough glass slab and a mirror-able floor tile. */
function roomScene(mats: MaterialData[], lights: LightData[], floorMat = 0): ReturnType<typeof quadScene> {
  const quads = [
    { p: [[-2, 0, 2], [2, 0, 2], [2, 0, -2], [-2, 0, -2]], mat: floorMat },                         // floor (up)
    { p: [[-2, 0, -2], [2, 0, -2], [2, 3, -2], [-2, 3, -2]], mat: 1 },                               // back wall (+z)
    { p: [[-2, 0, 2], [-2, 0, -2], [-2, 3, -2], [-2, 3, 2]], mat: 0 },                               // left wall (+x)
    ...boxQuads([0.2, 0.3, -0.8], [1.2, 0.36, 0.2], 2),                                              // glass slab
  ];
  return quadScene(quads, mats, lights);
}
/** Camera tilted down by t (rad) at p: X = +x, Y = (0, cos t, −sin t), Z = (0, sin t, cos t) (looks down −Z). */
const tiltCam = (t: number, p: V3, yfovDeg: number) => ({ camToWorld: [1, 0, 0, 0, 0, Math.cos(t), -Math.sin(t), 0, 0, Math.sin(t), Math.cos(t), 0, ...p, 1], yfov: yfovDeg * Math.PI / 180 });
const ROOM_CAM = tiltCam(20 * Math.PI / 180, [0, 2.2, 4.2], 50);

describe('PT light modes: U9 Mode A ≡ Mode B ≡ Mode A′ on delta-free scenes (gap-light U9; glass §4.3)', () => {
  it('rough glass + glossy + Lambert, rect + disk (spread 90°) lights, max_bounces 3: region means agree within 5σ; A′ ≡ A bitwise', async () => {
    const W = 24, H = 18;
    const mats = [lambert(0.6), v1Glossy(0.3), glassNodeMat([0.9, 0.95, 1], 0.4, 1.45)];
    const lights = [rectToward(0, [0, -1, 0], [0.3, 2.6, -0.4], 40), diskToward(1, norm([0.6, -1, -0.3]), [-1.2, 2.2, 1.0], 25)];
    const scene = roomScene(mats, lights);
    const res: Record<string, { mean: number[]; se: number[] }> = {};
    for (const mode of ['A', 'B', 'A′'] as const) {
      const r = await ptRig(scene, W, H, ROOM_CAM, { maxBounces: 3, lightMode: mode, jitterMode: JITTER_IID, seed: 7 });
      const st = await batchStats(r, 16, 64, (m) => regionMeans(m, W, H));
      expectCounters(st.counters);
      res[mode] = st;
      r.destroy();
    }
    for (let k = 0; k < 5; k++) {
      const d = res.A.mean[k] - res.B.mean[k], se = Math.hypot(res.A.se[k], res.B.se[k]);
      expect(Math.abs(d), `region ${k}: A ${res.A.mean[k]} B ${res.B.mean[k]} (se ${se})`).toBeLessThan(5 * se);
      expect(res['A′'].mean[k], 'A′ without delta lobes is A').toBe(res.A.mean[k]);   // same seeds, no crossing
    }
  });
});

describe('PT light modes: mirror positive control (vi-B / G6-neg): A < B, A′ ≡ B', () => {
  it('roughness-0 metal floor under a rect light: Mode A never shows the light in the mirror, B and A′ do and agree', async () => {
    const W = 20, H = 16;
    const mats = [mirror(), lambert(0.5), glassNodeMat([1, 1, 1], 0.4)];
    const lights = [rectToward(0, [0, -1, 0], [0, 2.4, -0.8], 60, 0.9)];
    const scene = roomScene(mats, lights, 0);
    const res: Record<string, { mean: number[]; se: number[] }> = {};
    for (const mode of ['A', 'B', 'A′'] as const) {
      const r = await ptRig(scene, W, H, ROOM_CAM, { maxBounces: 2, lightMode: mode, jitterMode: JITTER_IID, seed: 11 });
      const st = await batchStats(r, 16, 64, (m) => regionMeans(m, W, H));
      expectCounters(st.counters);
      res[mode] = st;
      r.destroy();
    }
    const k = 4;
    const gap = res.B.mean[k] - res.A.mean[k], seAB = Math.hypot(res.A.se[k], res.B.se[k]);
    expect(gap / seAB, `A ${res.A.mean[k]} < B ${res.B.mean[k]}`).toBeGreaterThan(10);
    for (let q = 0; q < 5; q++) {
      const d = res['A′'].mean[q] - res.B.mean[q], se = Math.hypot(res['A′'].se[q], res.B.se[q]);
      expect(Math.abs(d), `region ${q}: A′ ${res['A′'].mean[q]} vs B ${res.B.mean[q]} (se ${se})`).toBeLessThan(5 * se);
    }
  });
});

describe('PT light modes: U10 pass-through replay determinism (probe: vertex/lobe hash, crossing candidates, vertices)', () => {
  const W = 16, H = 12;
  const cam = tiltCam(20 * Math.PI / 180, [0, 1.6, 2.6], 45);
  const mats = [lambert(0.6), lambert(0.4), glassNodeMat([1, 1, 1], 0.3)];
  // a rect light between the floor and the back wall FACING the viewer side (+z: rays toward the wall cross it
  // front-facing), and one above all geometry facing up (+y: no ray ever travels against its emission axis there,
  // the room has no ceiling), i.e. it can only be crossed from the back
  const front = rectToward(0, [0, 0, 1], [0, 1.2, -1.2], 30, 3.5);
  const back = rectToward(1, [0, 1, 0], [0.3, 3.2, -0.8], 30, 1.4);
  const probeRun = async (lights: LightData[], mode: 'A' | 'B' | 'A′') => {
    const r = await ptRig(roomScene(mats, lights), W, H, cam, { maxBounces: 4, lightMode: mode, probe: true, jitterMode: JITTER_IID, seed: 5 });
    const b = await runPt(r, 1);
    r.destroy();
    return b.mean;
  };
  it('identical vertex sequences with and without lights and across modes; identical candidates on replay; back faces give none', async () => {
    const bFront = await probeRun([front, back], 'B');
    const bAgain = await probeRun([front, back], 'B');
    const aFront = await probeRun([front, back], 'A');
    const none = await probeRun([], 'B');
    const onlyBack = await probeRun([back], 'B');
    let crossings = 0;
    for (let i = 0; i < W * H; i++) {
      expect(bAgain[3 * i]).toBe(bFront[3 * i]);             // replay: same vertex/lobe sequence
      expect(bAgain[3 * i + 1]).toBe(bFront[3 * i + 1]);     // … and the same crossing candidates
      expect(aFront[3 * i]).toBe(bFront[3 * i]);             // the mode never changes the path
      expect(none[3 * i]).toBe(bFront[3 * i]);               // lights (and their crossings) consume no RNG / vertex
      expect(none[3 * i + 2]).toBe(bFront[3 * i + 2]);       // number of scattering vertices unchanged
      expect(aFront[3 * i + 1]).toBe(0);                     // Mode A: never crossed
      expect(onlyBack[3 * i + 1]).toBe(0);                   // back-facing light: no candidate
      crossings += bFront[3 * i + 1];
    }
    expect(crossings).toBeGreaterThanOrEqual(10);
  });
});

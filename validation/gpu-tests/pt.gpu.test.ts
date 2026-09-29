// Reference path tracer (pt.wgsl + lights + measure), both GPU lanes. Analytic checks at maxBounces 0 (validation-
// harness §8 C0c–C0e, C0l–C0n; gap-light U2), furnace T10 / C0f, primary emission U11, three-estimator T9d.
//   - deterministic (pixel-centre rays, delta lights, 1 light): exact per-pixel comparison to the f64 formula;
//   - stochastic (area lights, emissive triangles, furnace): batch means, |Δ| < 5 SE and a relative tolerance.
import { afterAll, describe, expect, it } from 'vitest';
import { buildBvh } from '../../src/core/bvh/sah-builder.ts';
import { createEnvResources, destroyEnvResources } from '../../src/core/render/env-gpu.ts';
import { FRAME_RESET_HISTORY, FrameUniformBuffer, JITTER_IID, JITTER_NONE, computeRenderOrigin } from '../../src/core/render/frame-uniforms.ts';
import { PT_COUNTERS, PtFramePass } from '../../src/core/render/pt-kernel.ts';
import { SceneGpu } from '../../src/core/render/scene-gpu.ts';
import { spotParams, spotProfile } from '../../src/core/render/lights-gpu.ts';
import { spreadNormalization } from '../../src/core/render/emission-kernel.ts';
import type { LightData } from '../../src/core/scene/types.ts';
import { getTestGpu, releaseTestGpu } from './device-factory.ts';
import {
  batchStats, cameraRay, dot, groundQuad, hitGround, lambert, lightMatrixToward, lookDown, material, norm, pixel, polygonIrradiance,
  ptRig, quadScene, runPt, sub, type V3,
} from './pt-fixtures.ts';

afterAll(releaseTestGpu);

const I4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const RHO = 0.5;

function light(o: Partial<LightData> & Pick<LightData, 'type'>): LightData {
  return { id: 0, name: o.type, color: [1, 1, 1], power: 100, exposure: 0, matrix: lightMatrixToward([0, -1, 0], [0, 1, 0]), visibleToCamera: false, ...o };
}

/** Plane + camera looking down from height 3 over a 2.4 m wide view. */
const W = 24, H = 24, CAM_H = 3, YFOV = 2 * Math.atan(1.2 / CAM_H);
const cam = { camToWorld: lookDown([0.1, CAM_H, -0.05]), yfov: YFOV };

function expectCounters(c: number[]): void {
  expect(c[PT_COUNTERS.nonFinite], 'NaN/Inf samples').toBe(0);
  expect(c[PT_COUNTERS.bvhOverflow], 'BVH overflow').toBe(0);
  expect(c[PT_COUNTERS.bvhItercap], 'BVH iteration cap').toBe(0);
  expect(c[PT_COUNTERS.negative], 'negative samples').toBe(0);
}

/** Deterministic per-pixel check: pixel centres, 1 spp, relative tolerance. */
async function checkPixels(lights: LightData[], want: (x: V3) => number, tol = 2e-5, rho = RHO): Promise<number> {
  const scene = quadScene([groundQuad(2)], [lambert(rho)], lights);
  const r = await ptRig(scene, W, H, cam, { maxBounces: 0, jitterMode: JITTER_NONE });
  const b = await runPt(r, 1);
  expectCounters(b.counters);
  let worst = 0, lit = 0;
  for (let row = 0; row < H; row++) {
    for (let c = 0; c < W; c++) {
      const x = hitGround(cameraRay(cam.camToWorld, YFOV, W, H, c, row));
      const w = want(x);
      const got = pixel(b.mean, W, c, row);
      if (w > 0) lit++;
      for (const g of got) {
        const err = Math.abs(g - w) / Math.max(w, 1e-3);
        worst = Math.max(worst, err);
        expect(err, `pixel ${c},${row}: got ${g}, want ${w}`).toBeLessThan(tol);
      }
    }
  }
  r.destroy();
  return lit;
}

describe('PT: length-1 terms (U11)', () => {
  it('camera facing an emissive quad (C0b-like): pixel = L_e exactly at maxBounces 0 and 3', async () => {
    for (const maxBounces of [0, 3]) {
      const S = 50, D = 2;
      const scene = quadScene([{ p: [[-S, -S, -D], [-S, S, -D], [S, S, -D], [S, -S, -D]], mat: 0 }],
        [material({ emissiveFactor: [0.5, 0.25, 0.125], emissiveStrength: 2, v1: { diffuse: [0, 0, 0], glossy: [0, 0, 0], roughness: 0.5, mix: 0 } })]);
      const r = await ptRig(scene, 32, 24, { camToWorld: I4, yfov: 40 * Math.PI / 180 }, { maxBounces });
      const b = await runPt(r, 8);
      expectCounters(b.counters);
      let bad = 0;
      for (let i = 0; i < 32 * 24; i++) if (b.mean[3 * i] !== 1 || b.mean[3 * i + 1] !== 0.5 || b.mean[3 * i + 2] !== 0.25) bad++;
      expect(bad).toBe(0);
      r.destroy();
    }
  });
});

describe('PT: camera-visible analytic area light (length-1, pass-through, weight 1)', () => {
  it('U11(ii): camera looks through a visibleToCamera rect at an emissive wall: L_rect + L_wall inside, one-sided', async () => {
    const Lw: V3 = [0.25, 0.5, 0.75];
    const wall = { p: [[-5, -5, -4], [5, -5, -4], [5, 5, -4], [-5, 5, -4]], mat: 0 };
    const wallMat = material({ emissiveFactor: Lw, v1: { diffuse: [0, 0, 0], glossy: [0, 0, 0], roughness: 0.5, mix: 0 } });
    const Lrect = 10 / (Math.PI * 4);
    for (const facing of [1, -1] as const) {
      // a_L = +Z (toward the camera at the origin) when facing = 1
      const l = light({ id: 2, type: 'rect', power: 10, sizeX: 2, sizeY: 2, visibleToCamera: true, matrix: lightMatrixToward([0, 0, facing], [0, 0, -2]) });
      const r = await ptRig(quadScene([wall], [wallMat], [l]), 16, 16, { camToWorld: I4, yfov: Math.PI / 2 }, { maxBounces: 2, jitterMode: JITTER_NONE });
      const b = await runPt(r, 4);
      expectCounters(b.counters);
      const inside = pixel(b.mean, 16, 8, 8), outside = pixel(b.mean, 16, 0, 0);
      inside.forEach((v, k) => expect(v).toBeCloseTo(Lw[k] + (facing === 1 ? Lrect : 0), 5));
      outside.forEach((v, k) => expect(v).toBeCloseTo(Lw[k], 5));
      r.destroy();
    }
  });
});

describe('PT: analytic direct lighting at maxBounces 0 (pixel centres, exact)', () => {
  it('C0c point light over a Lambert plane: L = ρ P h/(4π² d³)', async () => {
    const P = 100, h = 1, lp: V3 = [0.2, h, -0.3];
    const lit = await checkPixels([light({ type: 'point', power: P, matrix: lightMatrixToward([0, -1, 0], lp) })], (x) => {
      const d = Math.hypot(...sub(x, lp));
      return RHO * P * h / (4 * Math.PI ** 2 * d ** 3);
    });
    expect(lit).toBe(W * H);
  });

  for (const blend of [0.15, 0.5, 0]) {
    it(`C0d spot (60°, blend ${blend}, tilted 20°): L = (ρ/π)(P/4π) S(cosθ') cosθ/d²`, async () => {
      const tilt = 20 * Math.PI / 180;
      const aL = norm([Math.sin(tilt), -Math.cos(tilt), 0]);
      const lp: V3 = [-0.2, 1.2, 0.1];
      const l = light({ type: 'spot', power: 100, spotSize: 60 * Math.PI / 180, spotBlend: blend, matrix: lightMatrixToward(aL as V3, lp) });
      const { cosHalf, smooth } = spotParams(l);
      const want = (x: V3) => {
        const v = sub(x, lp);
        const d = Math.hypot(...v);
        const cosT = dot(v, aL) / d;
        if (blend === 0 && Math.abs(cosT - cosHalf) < 2e-4) return NaN; // hard edge: skip the band
        return (RHO / Math.PI) * (100 / (4 * Math.PI)) * spotProfile(cosT, cosHalf, smooth) * (lp[1] / d) / (d * d);
      };
      const scene = quadScene([groundQuad(2)], [lambert(RHO)], [l]);
      const r = await ptRig(scene, W, H, cam, { maxBounces: 0, jitterMode: JITTER_NONE });
      const b = await runPt(r, 1);
      expectCounters(b.counters);
      let lit = 0, dark = 0;
      for (let row = 0; row < H; row++) for (let c = 0; c < W; c++) {
        const w = want(hitGround(cameraRay(cam.camToWorld, YFOV, W, H, c, row)));
        if (Number.isNaN(w)) continue;
        const g = pixel(b.mean, W, c, row)[0];
        if (w > 0) lit++; else dark++;
        // f32 hit points (barycentrics on the quad) move cosθ' by ~1e-7; the blend band amplifies it by S'·spotSmooth
        expect(Math.abs(g - w), `pixel ${c},${row}: ${g} vs ${w}`).toBeLessThan(1e-4 * Math.max(w, 1e-2));
      }
      expect(lit).toBeGreaterThan(20);
      expect(dark).toBeGreaterThan(20);
      r.destroy();
    });
  }

  it('C0l sun: L = ρ/π · E · cos', async () => {
    const E = 3, sunDir = norm([0.3, 1, -0.4]); // toward the sun
    const l = light({ type: 'sun', power: E, color: [1, 0.5, 0.25], matrix: lightMatrixToward([-sunDir[0], -sunDir[1], -sunDir[2]], [5, 5, 5]) });
    const scene = quadScene([groundQuad(50)], [lambert(RHO)], [l]);
    const r = await ptRig(scene, W, H, cam, { maxBounces: 0, jitterMode: JITTER_NONE });
    const b = await runPt(r, 1);
    expectCounters(b.counters);
    const want = [1, 0.5, 0.25].map((c) => RHO / Math.PI * E * c * sunDir[1]);
    for (const [c, row] of [[0, 0], [12, 12], [23, 5], [3, 20]]) {
      pixel(b.mean, W, c, row).forEach((g, k) => expect(g).toBeCloseTo(want[k], 5));
    }
    r.destroy();
  });

  it('two point lights: the alias selection (realized pmf) averages to the sum (stochastic)', async () => {
    const la: V3 = [0.4, 1, 0], lb: V3 = [-0.5, 0.7, 0.2];
    const lights = [
      light({ id: 3, type: 'point', power: 100, matrix: lightMatrixToward([0, -1, 0], la) }),
      light({ id: 9, type: 'point', power: 30, color: [0.2, 1, 0.5], matrix: lightMatrixToward([0, -1, 0], lb) }),
    ];
    const scene = quadScene([groundQuad(50)], [lambert(RHO)], lights);
    const r = await ptRig(scene, 8, 8, cam, { maxBounces: 0, jitterMode: JITTER_NONE });
    const probes = [[1, 1], [4, 4], [6, 2]];
    const st = await batchStats(r, 8, 256, (m) => probes.flatMap(([c, row]) => pixel(m, 8, c, row)));
    expectCounters(st.counters);
    probes.forEach(([c, row], i) => {
      const x = hitGround(cameraRay(cam.camToWorld, YFOV, 8, 8, c, row));
      const f = (lp: V3, P: number) => { const d = Math.hypot(...sub(x, lp)); return RHO * P * lp[1] / (4 * Math.PI ** 2 * d ** 3); };
      const want = [0, 1, 2].map((k) => f(la, 100) * 1 + f(lb, 30) * [0.2, 1, 0.5][k]);
      want.forEach((w, k) => {
        const j = 3 * i + k;
        expect(Math.abs(st.mean[j] - w), `probe ${i} ch ${k}: ${st.mean[j]} vs ${w} (se ${st.se[j]})`).toBeLessThan(5 * st.se[j] + 1e-6 * w);
      });
    });
    r.destroy();
  });
});

describe('PT: area lights and emissive triangles at maxBounces 0 (stochastic, batch SE)', () => {
  const PROBES = [[2, 2], [8, 5], [13, 13], [4, 12]];
  const W2 = 16, H2 = 16;
  const cam2 = { camToWorld: lookDown([0, CAM_H, 0]), yfov: 2 * Math.atan(1.5 / CAM_H) };
  const xAt = (c: number, row: number) => hitGround(cameraRay(cam2.camToWorld, cam2.yfov, W2, H2, c, row));

  async function check(scene: ReturnType<typeof quadScene>, want: (x: V3) => number[], technique: 'mis' | 'nee' | 'bsdf' = 'mis', relTol = 0.02,
    probes = PROBES, spp = 512) {
    const r = await ptRig(scene, W2, H2, cam2, { maxBounces: 0, jitterMode: JITTER_NONE, technique });
    const st = await batchStats(r, 8, spp, (m) => probes.flatMap(([c, row]) => pixel(m, W2, c, row)));
    expectCounters(st.counters);
    probes.forEach(([c, row], i) => {
      want(xAt(c, row)).forEach((w, k) => {
        const j = 3 * i + k;
        const err = Math.abs(st.mean[j] - w);
        expect(err, `${technique} probe ${i} ch ${k}: ${st.mean[j]} vs ${w} (se ${st.se[j]})`).toBeLessThan(5 * st.se[j] + 1e-6);
        if (w > 0) expect(err / w, `${technique} probe ${i} relative`).toBeLessThan(relTol);
        else expect(st.mean[j], `${technique} probe ${i}: outside the support`).toBe(0);
      });
    });
    r.destroy();
  }

  it('C0e rect light (one-sided): L = (ρ/π)·E_polygon, L_e = P/(πab)', async () => {
    const a = 0.6, bb = 0.4, P = 20, hL = 1.1, c: V3 = [0.1, hL, -0.2];
    const l = light({ type: 'rect', power: P, sizeX: a, sizeY: bb, spread: Math.PI, matrix: lightMatrixToward([0, -1, 0], c) });
    const Le = P / (Math.PI * a * bb);
    // record axes: lightMatrixToward([0,−1,0]) → X = (0,0,1), Y = (1,0,0)
    const verts: V3[] = [[-bb / 2, 0, -a / 2], [bb / 2, 0, -a / 2], [bb / 2, 0, a / 2], [-bb / 2, 0, a / 2]].map((v) => [c[0] + v[0], hL, c[2] + v[2]] as V3);
    const scene = quadScene([groundQuad(50)], [lambert([0.5, 0.25, 0.8])], [l]);
    await check(scene, (x) => [0.5, 0.25, 0.8].map((rho) => rho / Math.PI * Le * polygonIrradiance(x, [0, 1, 0], verts)));
  });

  it('C0e rect light facing away: exactly black (one-sided)', async () => {
    const l = light({ type: 'rect', power: 20, sizeX: 0.6, sizeY: 0.4, matrix: lightMatrixToward([0, 1, 0], [0, 1, 0]) });
    const scene = quadScene([groundQuad(50)], [lambert(RHO)], [l]);
    const r = await ptRig(scene, W2, H2, cam2, { maxBounces: 0, jitterMode: JITTER_NONE });
    const b = await runPt(r, 64);
    expect(b.mean.every((x) => x === 0)).toBe(true);
    r.destroy();
  });

  it('C0m disk light on axis: E = π L_e R²/(R² + h²)', async () => {
    const D = 0.8, P = 15, h = 1.3;
    const x0 = xAt(8, 8); // pixel near the image centre: put the disk right above it
    const l = light({ type: 'disk', power: P, sizeX: D, matrix: lightMatrixToward([0, -1, 0], [x0[0], h, x0[2]]) });
    const Le = P / (Math.PI * Math.PI / 4 * D * D);
    const scene = quadScene([groundQuad(50)], [lambert(RHO)], [l]);
    const r = await ptRig(scene, W2, H2, cam2, { maxBounces: 0, jitterMode: JITTER_NONE });
    const st = await batchStats(r, 8, 512, (m) => pixel(m, W2, 8, 8));
    const R = D / 2;
    const want = RHO / Math.PI * Math.PI * Le * R * R / (R * R + h * h);
    expect(Math.abs(st.mean[0] - want), `${st.mean[0]} vs ${want} se ${st.se[0]}`).toBeLessThan(5 * st.se[0] + 1e-6);
    expect(Math.abs(st.mean[0] - want) / want).toBeLessThan(0.005);
    r.destroy();
  });

  it('C0n spread 30°/90°: matches f64 quadrature of L_e·spread(θ)·cosθ_z cosθ_x/r²', async () => {
    for (const spreadDeg of [90, 30]) {
      const spread = spreadDeg * Math.PI / 180, a = 0.5, bb = 0.5, P = 10, hL = 0.8, c: V3 = [0.2, hL, 0];
      const l = light({ type: 'rect', power: P, sizeX: a, sizeY: bb, spread, matrix: lightMatrixToward([0, -1, 0], c) });
      const Le = P / (Math.PI * a * bb);
      const { tanHalf, norm: Ns } = spreadNormalization(spread);
      const want = (x: V3): number[] => {
        const n = 400;
        let E = 0;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
          const z: V3 = [c[0] + ((i + 0.5) / n - 0.5) * bb, hL, c[2] + ((j + 0.5) / n - 0.5) * a];
          const v = sub(x, z); const r = Math.hypot(...v);
          const cosZ = -v[1] / r; // a_L = (0,−1,0): cosθ_z = a_L·(x − z)/r
          if (cosZ <= 0) continue;
          const tanT = Math.sqrt(1 - cosZ * cosZ) / cosZ;
          const f = Math.max((tanHalf - tanT) * Ns, 0);
          E += Le * f * cosZ * cosZ / (r * r); // cosθ_x = cosθ_z (parallel planes)
        }
        E *= (a * bb) / (n * n);
        return [RHO / Math.PI * E, RHO / Math.PI * E, RHO / Math.PI * E];
      };
      const scene = quadScene([groundQuad(50)], [lambert(RHO)], [l]);
      await check(scene, want, 'mis', 0.05); // the 5-SE test is the gate; near the cone edge the SE is a few %
    }
  });

  it('emissive quad (two-sided, textureless): MIS = NEE-only = BSDF-only = (ρ/π)·L_e·E_polygon (T9d)', async () => {
    const Le: V3 = [2, 1, 0.5], hL = 0.9, s = 0.35;
    // emitter faces UP (away from the plane): two-sidedness makes it light the plane anyway
    const quad = { p: [[-s, hL, s], [s, hL, s], [s, hL, -s], [-s, hL, -s]], mat: 1 };
    const scene = quadScene([groundQuad(50), quad], [lambert(RHO), material({ emissiveFactor: Le, v1: { diffuse: [0, 0, 0], glossy: [0, 0, 0], roughness: 0.5, mix: 0 } })]);
    const verts: V3[] = quad.p.map((p) => p as V3);
    const want = (x: V3) => Le.map((l) => RHO / Math.PI * l * polygonIrradiance(x, [0, 1, 0], verts));
    // probes outside the emitter's image footprint (the camera looks down past it)
    for (const t of ['mis', 'nee', 'bsdf'] as const) await check(scene, want, t, t === 'bsdf' ? 0.2 : 0.02, [[2, 2], [13, 13], [4, 12], [1, 8]], t === 'bsdf' ? 4096 : 512);
  });
});

describe('PT: furnace T10 / C0f (closed emissive enclosure, all bounces)', () => {
  // Inward-facing cube [−1,1]³, every face emissive L_e and Lambert ρ: L = L_e(1 − ρ^{b+2})/(1 − ρ).
  const L_E = 0.5, R = 0.6;
  const cube = () => {
    const q = (a: number[], b: number[], c: number[], d: number[]) => ({ p: [a, b, c, d], mat: 0 });
    const m = material({ emissiveFactor: [L_E, L_E, L_E], baseColorFactor: [R, R, R, 1], v1: { diffuse: [R, R, R], glossy: [0, 0, 0], roughness: 0.5, mix: 0 } });
    return quadScene([
      q([-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]), q([-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]),
      q([-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]), q([1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]),
      q([-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1]), q([-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]),
    ], [m]);
  };
  for (const [b, rr] of [[0, false], [1, false], [3, false], [3, true]] as const) {
    it(`b = ${b}${rr ? ' with Russian roulette from x₁ (T11)' : ''}: image mean = L_e(1 − ρ^{b+2})/(1 − ρ)${b === 0 ? ' = L_e(1 + ρ)' : ''}`, async () => {
      const cam3 = { camToWorld: [0.8, 0, 0.6, 0, -0.36, 0.8, 0.48, 0, -0.48, -0.6, 0.64, 0, 0.1, -0.2, 0.3, 1], yfov: 1.4 };
      const r = await ptRig(cube(), 32, 32, cam3, { maxBounces: b, rr, rrMinBounces: 0 });
      const st = await batchStats(r, 8, 64, (m) => { let s = 0; for (let i = 0; i < m.length; i += 3) s += m[i]; return [s / (m.length / 3)]; });
      expectCounters(st.counters);
      const want = L_E * (1 - R ** (b + 2)) / (1 - R);
      expect(Math.abs(st.mean[0] - want), `${st.mean[0]} vs ${want} (se ${st.se[0]})`).toBeLessThan(5 * st.se[0] + 1e-7);
      expect(Math.abs(st.mean[0] - want) / want).toBeLessThan(0.002);
      r.destroy();
    });
  }
});

describe('PT: interactive pass (PtFramePass, renderer PT mode)', () => {
  it('frame t (reset) ≡ batch sample t bit for bit (same seed, jitter and path code); progressive mean', async () => {
    const { device, features, wgslLanguageFeatures } = await getTestGpu();
    const Le: V3 = [1, 0.5, 0.25];
    const scene = quadScene([groundQuad(3), { p: [[-0.4, 1, 0.4], [0.4, 1, 0.4], [0.4, 1, -0.4], [-0.4, 1, -0.4]], mat: 1 }],
      [lambert(0.6), material({ emissiveFactor: Le })],
      [light({ id: 1, type: 'rect', power: 5, sizeX: 0.3, sizeY: 0.3, matrix: lightMatrixToward([0, -1, 0], [0.8, 1.5, 0]) }),
        light({ id: 4, type: 'point', power: 20, matrix: lightMatrixToward([0, -1, 0], [-0.8, 1.5, 0.3]) })]);
    const Wf = 32, Hf = 24, seed = 99, t = 5;
    const camf = { camToWorld: [1, 0, 0, 0, 0, 0.8, 0.6, 0, 0, -0.6, 0.8, 0, 0.2, 2.2, 2.5, 1], yfov: 1.0 };
    const batch = await ptRig(scene, Wf, Hf, camf, { maxBounces: 3, seed });
    const ref = await batch.acc.runBatch((enc, d, a, c) => batch.kernel.encode(enc, { ...d, sampleBase: t }, a, c), 1, 0);
    expectCounters(ref.counters);
    const origin = computeRenderOrigin(scene.bounds);
    const gpu = await SceneGpu.create(device, scene, origin, {
      textureMode: 'validation', watertight: true, buildBvh: async (p, i) => buildBvh(p, i, { mt: true, woop: true }), features, wgslLanguageFeatures,
    });
    const env = await createEnvResources(device, undefined);
    const pass = await PtFramePass.create(device, gpu, env, 'rgba32float', { maxBounces: 3, features, wgslLanguageFeatures });
    const color = device.createTexture({ size: [Wf, Hf], format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
    const fu = new FrameUniformBuffer(device);
    pass.setTargets({ width: Wf, height: Hf, color, frameUniforms: fu.buffer });
    const frameAt = async (seedIndex: number, flags: number) => {
      fu.write({ camera: camf, prevCamera: camf, width: Wf, height: Hf, frameIndex: seedIndex, seedIndex, runSeed: seed, flags, jitterMode: JITTER_IID,
        jitter: [0.5, 0.5], origin, exposure: 1, time: 0, dt: 0, sceneDiag: 1 });
      const enc = device.createCommandEncoder();
      expect(pass.encode(enc, { advanced: true, accumulate: true })).toBe(true);
      const rowBytes = Wf * 16;
      const buf = device.createBuffer({ size: rowBytes * Hf, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      enc.copyTextureToBuffer({ texture: color }, { buffer: buf, bytesPerRow: rowBytes }, [Wf, Hf]);
      device.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(buf.getMappedRange().slice(0));
      buf.unmap(); buf.destroy();
      return out;
    };
    const f5 = await frameAt(t, FRAME_RESET_HISTORY);
    let diff = 0;
    for (let i = 0; i < Wf * Hf; i++) for (let k = 0; k < 3; k++) if (f5[4 * i + k] !== ref.mean[3 * i + k]) diff++;
    expect(diff).toBe(0);
    // progressive mean: frame t+1 (no reset) = (sample t + sample t+1)/2
    const ref6 = await batch.acc.runBatch((enc, d, a, c) => batch.kernel.encode(enc, { ...d, sampleBase: t + 1 }, a, c), 1, 1);
    const f6 = await frameAt(t + 1, 0);
    let worst = 0;
    for (let i = 0; i < Wf * Hf; i++) worst = Math.max(worst, Math.abs(f6[4 * i] - (ref.mean[3 * i] + ref6.mean[3 * i]) / 2));
    expect(worst).toBeLessThan(1e-5);
    const cnt = new Uint32Array(await (await import('../../src/core/gpu/readback.ts')).readBuffer(device, pass.counters, 16));
    expect(Array.from(cnt)).toEqual([0, 0, 0, 0]);
    pass.destroy(); color.destroy(); fu.destroy(); gpu.destroy(); destroyEnvResources(env); batch.destroy();
  });
});

describe('PT: offline split-dispatch equality and Gate-1 plants (plan §7.4 M3a)', () => {
  const scene = () => quadScene(
    [groundQuad(3), { p: [[-3, 0, -1], [3, 0, -1], [3, 2, -1], [-3, 2, -1]], mat: 2 }, { p: [[-0.4, 1.2, 0.4], [0.4, 1.2, 0.4], [0.4, 1.2, -0.4], [-0.4, 1.2, -0.4]], mat: 1 }],
    [lambert(0.6), material({ emissiveFactor: [1, 0.5, 0.25] }), lambert([0.3, 0.6, 0.4])],
    [light({ id: 1, type: 'rect', power: 5, sizeX: 0.3, sizeY: 0.3, matrix: lightMatrixToward([0, -1, 0], [0.8, 1.5, 0]) }),
      light({ id: 4, type: 'point', power: 20, matrix: lightMatrixToward([0, -1, 0], [-0.8, 1.5, 0.3]) })]);
  const Ws = 40, Hs = 36, cams = { camToWorld: [1, 0, 0, 0, 0, 0.8, -0.6, 0, 0, 0.6, 0.8, 0, 0.2, 2.2, 2.5, 1], yfov: 1.0 }; // looks down (0, −0.6, −0.8)

  it('row-band sub-submits are bit-identical to full-frame dispatches (k = 1); adaptive k splits agree to f32 summation', async () => {
    const one = await ptRig(scene(), Ws, Hs, cams, { maxBounces: 3, seed: 5, budget: { maxSamplesPerDispatch: 1, targetMs: 1e9 } });
    const full = await runPt(one, 12, 2);
    const bands = await ptRig(scene(), Ws, Hs, cams, { maxBounces: 3, seed: 5, budget: { targetMs: 1e-6 } }); // k = 1, 8-row bands
    const split = await runPt(bands, 12, 2);
    expect(split.submits).toBeGreaterThan(12 * 4);
    expect(new Uint32Array(split.mean.buffer)).toEqual(new Uint32Array(full.mean.buffer));
    const adapt = await ptRig(scene(), Ws, Hs, cams, { maxBounces: 3, seed: 5 });
    await runPt(adapt, 12, 0); // warm the adaptive k up
    const k = await runPt(adapt, 12, 2);
    expect(k.submits).toBeLessThan(split.submits);
    let worst = 0, energy = 0;
    for (let i = 0; i < full.mean.length; i++) { worst = Math.max(worst, Math.abs(k.mean[i] - full.mean[i]) / Math.max(Math.abs(full.mean[i]), 1e-3)); energy += full.mean[i]; }
    expect(worst).toBeLessThan(1e-5);
    expect(energy).toBeGreaterThan(0);
    expectCounters(full.counters); expectCounters(split.counters); expectCounters(k.counters);
    one.destroy(); bands.destroy(); adapt.destroy();
  });

  it('plants: emitScale 1.01 scales every sample exactly; drop 1% at vertex 2 only removes energy (never adds)', async () => {
    const base = await ptRig(scene(), Ws, Hs, cams, { maxBounces: 3, seed: 5, budget: { maxSamplesPerDispatch: 1, targetMs: 1e9 } });
    const s = await ptRig(scene(), Ws, Hs, cams, { maxBounces: 3, seed: 5, budget: { maxSamplesPerDispatch: 1, targetMs: 1e9 }, plant: { emitScale: 1.01 } });
    const d = await ptRig(scene(), Ws, Hs, cams, { maxBounces: 3, seed: 5, budget: { maxSamplesPerDispatch: 1, targetMs: 1e9 }, plant: { dropProb: 0.01, dropBounce: 2 } });
    const b0 = await runPt(base, 64, 0), b1 = await runPt(s, 64, 0), b2 = await runPt(d, 64, 0);
    let worst = 0, sum0 = 0, sum2 = 0, less = 0;
    for (let i = 0; i < b0.mean.length; i++) {
      worst = Math.max(worst, Math.abs(b1.mean[i] - 1.01 * b0.mean[i]) / Math.max(b0.mean[i], 1e-3));
      sum0 += b0.mean[i]; sum2 += b2.mean[i];
      expect(b2.mean[i]).toBeLessThanOrEqual(b0.mean[i] * (1 + 1e-6) + 1e-7);
      if (b2.mean[i] < b0.mean[i] * (1 - 1e-6)) less++;
    }
    expect(worst).toBeLessThan(1e-5);
    expect(sum2).toBeLessThan(sum0);
    expect(less).toBeGreaterThan(0);
    base.destroy(); s.destroy(); d.destroy();
  });
});

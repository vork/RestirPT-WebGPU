// M6 plant predictions (restir-m6-api.md §5.3, MD16), derived on CPU BEFORE any GPU measurement (TD29 / E-18 rule):
//   U8-4  J = t_x²/t_y² (RSF_PLANT_U8_T2) on u8_c0c_point_b0, rung 3.2 (offline: 3 rounds × 6 slots, disk R 10):
//         an f64 replica of the scene (camera rays onto the floor, point light, Lambert ρ 0.5, b = 0: every path is the
//         deterministic d = 2 NEE to the point light) under the production pairing maps (pairing.ts pairLayer /
//         pairTransform / pairPartner) and the production paired MIS (mis-ref.ts twins of mis.wgsl). Unplanted: every
//         pixel keeps its value exactly (unbiased, J = 1 at a light vertex). Planted: G_j = F_c·ρ, H_j = F_j/ρ with
//         ρ = t_j²/t_c² ⇒ one pairing scales the canonical by [1 + ρ(A + B)/(ρA + B)]/2, B = A·ρ^−1.5, > 1 iff ρ > 1:
//         pixels near the light's foot point (every partner farther from the light) brighten. M_foot = pixels within
//         24 px of the foot's projection: predicted +. The global sign is computed here and reported (it is predicted
//         only where the toy is unambiguous).
//   U8-8  W_NEE ← W^RIS·p1_σ/q (RSF_PLANT_U8_RIS_MIXED) on u8_c0e_rect_b0, rung 3.1 + RIS: the planted estimator is
//         ∫ F(y)·r²/|cos θ_z| dA instead of ∫ F dA (p1_σ/q = r²/|cos θ_z| for a uniformly sampled rect). The f64
//         quadrature below gives, per pixel and globally, the contribution-weighted mean of r²/|cos θ_z|: the sign of
//         the plant is the sign of (that mean − 1). The light sits 0.5 m above the floor, so near it r²/cos < 1.
//   Consistent-target RIS (math.md §8 [M6 addition]): RIS whose target p̂ is in ANY measure is unbiased when the same
//         ratio r = p̂/q forms the selection and the UCW; only a UCW in a different measure than q (the U8-8 plant) is
//         biased (gap-light §3.8's "trap" wording corrected).
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { pairLayer, pairPartner, pairTransform } from '../../src/core/render/restir/pairing.ts';
import { PAIR_TEX_SIZES } from '../../src/core/render/restir/presets.ts';
import { pairedWeightsRef, type PairedSlotRef } from './mis-ref.ts';

type V3 = [number, number, number];
const ROOT = path.resolve(__dirname, '../..');
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const nrm = (a: V3): V3 => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };

interface Pkg { cam: number[]; yfov: number; W: number; H: number; lights: { type: string; matrix: number[]; sizeX?: number; sizeY?: number }[] }
function pkg(name: string): Pkg {
  const j = JSON.parse(readFileSync(path.join(ROOT, 'validation/scenes', name, 'scene.json'), 'utf8'));
  return { cam: j.camera.matrix, yfov: j.camera.yfov, W: j.render.width, H: j.render.height, lights: j.lights };
}
/** Floor (y = 0) point seen through the centre of pixel (x, y); undefined above the horizon. */
function floorHit(p: Pkg, x: number, y: number): V3 | undefined {
  const m = p.cam, t = Math.tan(p.yfov / 2), asp = p.W / p.H;
  const u = ((x + 0.5) / p.W * 2 - 1) * t * asp, v = (1 - (y + 0.5) / p.H * 2) * t;
  const d: V3 = nrm([u * m[0] + v * m[4] - m[8], u * m[1] + v * m[5] - m[9], u * m[2] + v * m[6] - m[10]]);
  const o: V3 = [m[12], m[13], m[14]];
  if (!(d[1] < 0)) return undefined;
  const s = -o[1] / d[1];
  return [o[0] + s * d[0], 0, o[2] + s * d[2]];
}
/** Pixel of world point q (camera projection), as the gate uses it for M_foot. */
export function project(p: Pkg, q: V3): [number, number] {
  const m = p.cam, rel = sub(q, [m[12], m[13], m[14]]);
  const cx = dot(rel, [m[0], m[1], m[2]]), cy = dot(rel, [m[4], m[5], m[6]]), cz = -dot(rel, [m[8], m[9], m[10]]);
  const t = Math.tan(p.yfov / 2), asp = p.W / p.H;
  return [((cx / cz) / (t * asp) + 1) / 2 * p.W - 0.5, (1 - (cy / cz) / t) / 2 * p.H - 0.5];
}

describe('U8-4 prediction: J = t_x²/t_y² on u8_c0c_point_b0, offline (3 rounds × 6 slots, disk R 10)', () => {
  const P = pkg('u8_c0c_point_b0');
  const L: V3 = [P.lights[0].matrix[12], P.lights[0].matrix[13], P.lights[0].matrix[14]];
  const W = P.W, H = P.H;
  const pos: V3[] = [], F = new Float64Array(W * H), t2 = new Float64Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const q = floorHit(P, x, y)!;
    const d = sub(L, q), r2 = dot(d, d);
    pos.push(q); t2[y * W + x] = r2;
    F[y * W + x] = (0.5 / Math.PI) * (d[1] / Math.sqrt(r2)) / r2;   // ρ/π · cos/t² (I = const)
  }
  /** Mean ratio (estimate / truth) per pixel after the rounds, averaged over `frames` pairing transforms. */
  function run(plant: boolean, frames: number): Float64Array {
    const acc = new Float64Array(W * H);
    const layers = PAIR_TEX_SIZES.slice(0, 6).map((Ws, s) => pairLayer(Ws, 10, s));
    for (let t = 0; t < frames; t++) {
      let Wr = new Float64Array(W * H).fill(1), c = new Float64Array(W * H).fill(1);   // W = 1/p̂·F/p = 1 at the light pick
      for (let round = 0; round < 3; round++) {
        const trs = layers.map((l, s) => pairTransform(4101, 0, t, round, s, l.W));
        const W2 = new Float64Array(W * H), c2 = new Float64Array(W * H);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const i = y * W + x, pc = F[i];
          const slots: PairedSlotRef[] = [];
          layers.forEach((l, s) => {
            const q = pairPartner(l, trs[s], [x, y], W, H);
            if (!q) return;
            const j = q[1] * W + q[0];
            const rho = plant ? t2[j] / t2[i] : 1;   // J_{j→c} planted t_j²/t_c² (correct: 1), J_{c→j} = 1/ρ
            slots.push({ cj: c[j], pj: F[j], Wj: Wr[j], lumG: pc * rho, J: rho, valid: true, lumH: F[j] / rho });
          });
          const r = pairedWeightsRef(c[i], pc, Wr[i], slots);
          W2[i] = r.wSum / pc;                      // every sample is the light point: p̂_c(Y) = F_c
          c2[i] = r.cOut;
        }
        Wr = W2; c = c2;
      }
      for (let i = 0; i < W * H; i++) acc[i] += Wr[i] / frames;   // estimate F·W over truth F
    }
    return acc;
  }
  it('unplanted: every pixel exact; planted: M_foot (24 px around the foot) brighter (+), global sign reported', () => {
    const base = run(false, 2);
    let maxErr = 0;
    for (const v of base) maxErr = Math.max(maxErr, Math.abs(v - 1));
    expect(maxErr).toBeLessThan(1e-12);
    const pl = run(true, 4);
    const [fx, fy] = project(P, [L[0], 0, L[2]]);
    let sf = 0, sfT = 0, sg = 0, sgT = 0, nFoot = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      sg += F[i] * pl[i]; sgT += F[i];
      if ((x - fx) ** 2 + (y - fy) ** 2 <= 24 * 24) { sf += F[i] * pl[i]; sfT += F[i]; nFoot++; }
    }
    const foot = sf / sfT - 1, glob = sg / sgT - 1;
    console.log(`[U8-4 toy] foot pixel (${fx.toFixed(1)}, ${fy.toFixed(1)}), M_foot ${nFoot} px: Δ ${(100 * foot).toFixed(3)} %; global Δ ${(100 * glob).toFixed(3)} %`);
    expect(nFoot).toBeGreaterThan(1000);
    expect(foot).toBeGreaterThan(0.001);
    (globalThis as Record<string, unknown>).__u8_4 = { foot, glob };
  });
});

describe('U8-8 prediction: UCW in mixed measures (W·p1_σ/q = W·r²/|cos θ_z|) on u8_c0e_rect_b0, initial + RIS', () => {
  it('contribution-weighted mean of r²/|cos θ_z| per pixel and globally (sign of the plant)', () => {
    const P = pkg('u8_c0e_rect_b0');
    const l = P.lights[0], m = l.matrix;
    const c: V3 = [m[12], m[13], m[14]], ax: V3 = nrm([m[0], m[1], m[2]]), ay: V3 = nrm([m[4], m[5], m[6]]);
    const n: V3 = nrm([-m[8], -m[9], -m[10]]);            // emission axis −Z_obj
    const hx = (l.sizeX ?? 1) / 2, hy = (l.sizeY ?? 1) / 2, NQ = 16;
    let sF = 0, sFr = 0, pixLess = 0, pixMore = 0;
    for (let y = 0; y < P.H; y += 2) for (let x = 0; x < P.W; x += 2) {
      const q = floorHit(P, x, y);
      if (!q) continue;
      let f = 0, fr = 0;
      for (let a = 0; a < NQ; a++) for (let b = 0; b < NQ; b++) {
        const u = ((a + 0.5) / NQ * 2 - 1) * hx, v = ((b + 0.5) / NQ * 2 - 1) * hy;
        const z: V3 = [c[0] + u * ax[0] + v * ay[0], c[1] + u * ax[1] + v * ay[1], c[2] + u * ax[2] + v * ay[2]];
        const d = sub(z, q), r2 = dot(d, d), w = nrm(d);
        const cx = w[1], cz = -dot(w, n);                    // receiver cos (floor normal +y), light cos (one-sided)
        if (!(cx > 0 && cz > 0)) continue;
        const g = (cx * cz) / r2;                             // F ∝ ρ/π·L_e·G (area measure)
        f += g; fr += g * (r2 / cz);
      }
      sF += f; sFr += fr;
      if (f > 0) { if (fr / f < 1) pixLess++; else pixMore++; }
    }
    const glob = sFr / sF - 1;
    console.log(`[U8-8 quadrature] global Δ = E[r²/cos] − 1 = ${(100 * glob).toFixed(2)} %; pixels with mean < 1: ${pixLess}, > 1: ${pixMore}`);
    // the prediction recorded in restir-m6-api.md §5.3 (Changelog M6-3): the sign of `glob`
    expect(Math.abs(glob)).toBeGreaterThan(0.01);
    (globalThis as Record<string, unknown>).__u8_8 = glob;
  });
});

describe('consistent-target RIS is unbiased in any measure; a UCW in another measure than q is not (math.md §8)', () => {
  // 1D: integrate f(x) = 1 + x² on [0, 1] with M = 4 candidates from q(x) = 2x (inverse CDF √u).
  const f = (x: number) => 1 + x * x;
  const q = (x: number) => 2 * x;
  const truth = 4 / 3;
  function rng(seed: number): () => number {
    let a = seed >>> 0;
    return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  function est(target: (x: number) => number, ucwScale: (x: number) => number, n: number): { mean: number; se: number } {
    const r = rng(7);
    let s = 0, s2 = 0;
    for (let i = 0; i < n; i++) {
      let wSum = 0, sel = 0, rSel = 0;
      for (let m = 0; m < 4; m++) {
        const x = Math.sqrt(r()) || 1e-12;
        const w = target(x) / q(x);
        wSum += w;
        if (r() * wSum < w) { sel = x; rSel = w; }
      }
      const W = rSel > 0 ? (wSum / 4) / target(sel) * ucwScale(sel) : 0;
      const v = f(sel) * W;
      s += v; s2 += v * v;
    }
    const mean = s / n;
    return { mean, se: Math.sqrt(Math.max(0, s2 / n - mean * mean) / n) };
  }
  it('target f (area measure), target f·x² ("solid-angle-like" measure): both unbiased; UCW × (1 + x) biased', () => {
    const a = est(f, () => 1, 400_000);
    const b = est((x) => f(x) * x * x, () => 1, 400_000);
    const c = est(f, (x) => 1 + x, 400_000);
    expect(Math.abs(a.mean - truth) / a.se).toBeLessThan(4);
    expect(Math.abs(b.mean - truth) / b.se).toBeLessThan(4);
    expect((c.mean - truth) / c.se).toBeGreaterThan(20);
  });
});

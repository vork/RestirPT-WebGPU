// Env importance tables and the env as a light (plan §1.4b, §2 rule 15, §7.4 M3c; math.md#env-sampling,
// #light-selection; env §2.4–2.5, §5.1):
//   ENV-U3 (CPU part): the stored pdfUV equals the realized probability recomputed from the stored u16 integers (to 1 f32
//          ulp), Σ pdfUV/(W_m H_m) = 1 ± 1e-6, and the sampler mirror (envSampleCell) reproduces the realized pmf;
//   ENV-U5: ∫ pdf_σ dω = 1 ± 1e-5 by f64 quadrature (8×8 per cell, s from the DIRECTION as for escapes, pole cap);
//   ENV-U8: pmf determinism — rotation-only edit ⇒ pmf bitwise unchanged; strength edit ⇒ P(env) proxy + clamp as
//          specified; identical inputs ⇒ identical bits;
//   plus: grid sizing (W_m ≥ 4, cap), kernel/floor properties, the no-floor and target-pdf options, Φ_env.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { LightsState, NO_ENV_ENTRY, P_ENV_MAX, P_ENV_MIN, lightPowerProxy } from '../../src/core/render/lights-gpu.ts';
import {
  buildEnvImportance, ENV_FLOOR, envMeanLuminance, envPowerProxy, envSampleCell, importanceWidth, realizedFromPacked,
} from '../../src/core/scene/env/env-importance.ts';
import { decodeHdr } from '../../src/core/scene/env/hdr.ts';
import type { LightData, SceneData } from '../../src/core/scene/types.ts';

const f32ulp = (x: number) => { const f = new Float32Array([x]); const u = new Uint32Array(f.buffer); u[0] += 1; return Math.abs(f[0] - Math.fround(x)); };

/** Synthetic map (rows bottom-up): smooth sky gradient, dark ground, a few very bright texels (incl. seam and pole rows). */
export function synthEnv(W = 256, H = 128): Float32Array {
  const t = new Float32Array(W * H * 4);
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const v = (r + 0.5) / H, u = (c + 0.5) / W;
      const sky = v > 0.5 ? 0.5 + 2 * (v - 0.5) + 0.3 * Math.sin(6 * Math.PI * u) ** 2 : 0.05 + 0.02 * u;
      t.set([sky, 0.8 * sky + 0.01, 0.6 * sky + 0.05 * u, 1], 4 * (r * W + c));
    }
  }
  const hot = (c: number, r: number, v: number) => t.set([v, 0.9 * v, 0.7 * v, 1], 4 * (r * W + c));
  hot(Math.round(W * 0.3), Math.round(H * 0.75), 2000);
  hot(0, Math.round(H * 0.6), 500);           // seam column
  hot(W - 1, Math.round(H * 0.6), 500);
  hot(Math.round(W * 0.7), H - 1, 300);       // top (pole-wrap) row
  hot(Math.round(W * 0.1), 0, 100);           // bottom row
  return t;
}

function sceneWith(lights: LightData[] = [], radius = 1): SceneData {
  return {
    name: 's',
    geometry: {
      positions: Float32Array.from([-radius, 0, 0, radius, 0, 0, 0, 0, 0]), normals: new Float32Array(9), tangents: new Float32Array(12), uv0: new Float32Array(6),
      indices: Uint32Array.from([0, 1, 2]), triMaterial: Uint32Array.from([0]), triFlags: Uint32Array.from([0]),
    },
    materials: [{
      name: 'm', baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 1, emissiveFactor: [0, 0, 0], emissiveStrength: 1, ior: 1.5,
      specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true, model: 'v1',
    }],
    textures: [], lights, cameras: [], bounds: { min: [-radius, 0, 0], max: [radius, 0, 0] }, warnings: [],
  };
}

const point = (id: number, power: number): LightData => ({
  id, name: `p${id}`, type: 'point', color: [1, 1, 1], power, exposure: 0, matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]), visibleToCamera: false,
});

describe('env importance grid and weights (math.md#env-sampling)', () => {
  it('W_m = largest power of two ≤ min(W, cap), ≥ 4; H_m = W_m/2', () => {
    expect(importanceWidth(1024)).toBe(1024);
    expect(importanceWidth(1000)).toBe(512);
    expect(importanceWidth(8192)).toBe(4096);
    expect(importanceWidth(8192, 2048)).toBe(2048);
    expect(importanceWidth(2)).toBe(4);
    expect(importanceWidth(64, 256)).toBe(64);
    const t = buildEnvImportance(synthEnv(256, 128), 256, 128, { cap: 64 });
    expect([t.Wm, t.Hm, t.log2W, t.log2H]).toEqual([64, 32, 6, 5]);
  });

  it('ENV-U3 (CPU): stored pdfUV ≡ realized probabilities of the stored u16 integers; Σ pdfUV/(W_m H_m) = 1', () => {
    for (const [W, H, cap] of [[256, 128, 4096], [256, 128, 64], [300, 150, 4096], [8, 4, 4096]] as const) {
      const tex = synthEnv(W, H);
      const t = buildEnvImportance(tex, W, H, { cap });
      const rowP = realizedFromPacked(t.rowAlias, 0, t.Hm);
      let sum = 0, worst = 0;
      for (let i = 0; i < t.Hm; i++) {
        const colP = realizedFromPacked(t.colAlias, i * t.Wm, t.Wm);
        for (let j = 0; j < t.Wm; j++) {
          const want = rowP[i] * colP[j] * t.Wm * t.Hm;
          const got = t.pdfUV[i * t.Wm + j];
          worst = Math.max(worst, Math.abs(got - want) / f32ulp(want));
          sum += got;
          // floors: every cell has a threshold ≥ 64/65536 of its bucket, i.e. a positive realized probability
          expect(got).toBeGreaterThan(0);
        }
      }
      expect(worst, `${W}x${H} cap ${cap}: |stored − realized| in f32 ulps`).toBeLessThanOrEqual(1);
      expect(Math.abs(sum / (t.Wm * t.Hm) - 1)).toBeLessThan(1e-6);
    }
  });

  it('ENV-U3 (CPU): the sampler mirror reproduces the realized pmf exactly (full enumeration of a small table)', () => {
    const t = buildEnvImportance(synthEnv(16, 8), 16, 8);
    // Enumerate every (bucket, 16-bit threshold) pair of the row level, then of each row's column level.
    const count = new Float64Array(t.Wm * t.Hm);
    for (let bi = 0; bi < t.Hm; bi++) {
      for (let lo = 0; lo < 65536; lo += 1) {
        const [i] = envSampleCell(t, ((bi << (32 - t.log2H)) | lo) >>> 0, 0);
        count[i * t.Wm] += 1;   // row histogram (column 0 slot)
      }
    }
    const rowP = realizedFromPacked(t.rowAlias, 0, t.Hm);
    for (let i = 0; i < t.Hm; i++) expect(count[i * t.Wm] / (65536 * t.Hm)).toBe(rowP[i]);
    for (let i = 0; i < t.Hm; i++) {
      const colP = realizedFromPacked(t.colAlias, i * t.Wm, t.Wm);
      const c = new Float64Array(t.Wm);
      // h0 that deterministically selects row i: bucket i with threshold 0 (< q unless q = 0 → alias); find one
      let h0 = -1;
      for (let bi = 0; bi < t.Hm && h0 < 0; bi++) for (const lo of [0, 65535]) { const h = ((bi << (32 - t.log2H)) | lo) >>> 0; if (envSampleCell(t, h, 0)[0] === i) { h0 = h; break; } }
      expect(h0).toBeGreaterThanOrEqual(0);
      for (let bj = 0; bj < t.Wm; bj++) for (let lo = 0; lo < 65536; lo += 1) c[envSampleCell(t, h0, ((bj << (32 - t.log2W)) | lo) >>> 0)[1]] += 1;
      for (let j = 0; j < t.Wm; j++) expect(c[j] / (65536 * t.Wm)).toBe(colP[j]);
    }
  });

  it('ENV-U5: ∫ pdf_σ dω = 1 ± 1e-5 (f64 quadrature, 8×8 per cell, s from the direction, pole cap)', () => {
    for (const [W, H] of [[256, 128], [64, 32]] as const) {
      const t = buildEnvImportance(synthEnv(W, H), W, H);
      const n = 8;
      let integral = 0;
      for (let i = 0; i < t.Hm; i++) {
        for (let j = 0; j < t.Wm; j++) {
          const p = t.pdfUV[i * t.Wm + j];
          for (let a = 0; a < n; a++) {
            const v = (i + (a + 0.5) / n) / t.Hm;
            const theta = Math.PI - Math.PI * v;
            const s = Math.sin(theta);                      // = ‖b.xy‖ of envDir(u, v) (unit b)
            if (s < 1e-6) continue;                         // pole cap
            const dOmega = 2 * Math.PI * Math.PI * s / (t.Wm * t.Hm * n * n);
            integral += n * (p / (2 * Math.PI * Math.PI * s)) * dOmega;
          }
        }
      }
      expect(Math.abs(integral - 1), `${W}x${H}`).toBeLessThan(1e-5);
    }
  });

  it('kernel and floors: a single bright texel spreads to its 3×3 bilinear footprint; every cell ≥ 2⁻¹⁰ of its row mean', () => {
    const W = 64, H = 32;
    const tex = new Float32Array(W * H * 4);
    for (let k = 0; k < W * H; k++) tex[4 * k + 3] = 1;
    tex.set([1000, 1000, 1000, 1], 4 * (20 * W + 10));
    const t = buildEnvImportance(tex, W, H);
    const pdf = (r: number, c: number) => t.pdfUV[r * t.Wm + c];
    // the neighbours carry the 1/8 kernel tails (bilinear support), far cells only the floor
    expect(pdf(20, 11) / pdf(20, 10)).toBeGreaterThan(0.1);
    expect(pdf(21, 10) / pdf(20, 10)).toBeGreaterThan(0.1);
    expect(pdf(20, 13) / pdf(20, 10)).toBeLessThan(0.01);
    let mn = Infinity;
    for (let k = 0; k < t.pdfUV.length; k++) mn = Math.min(mn, t.pdfUV[k]);
    expect(mn).toBeGreaterThan(0);
    // no floors: black cells get exactly 0 (realized) — the negative control relies on BSDF covering them
    const nf = buildEnvImportance(tex, W, H, { floors: false });
    expect(nf.pdfUV[0]).toBe(0);
    expect(nf.pdfUV[20 * nf.Wm + 10]).toBeGreaterThan(t.pdfUV[20 * t.Wm + 10]);
    expect(ENV_FLOOR).toBe(2 ** -10);
  });

  it('target-weight plant stores the texel-centre density (normalized) but keeps the alias', () => {
    const tex = synthEnv(64, 32);
    const a = buildEnvImportance(tex, 64, 32);
    const b = buildEnvImportance(tex, 64, 32, { plantPdfFromTargets: true });
    expect(b.rowAlias).toEqual(a.rowAlias);
    expect(b.colAlias).toEqual(a.colAlias);
    let sum = 0, maxRatio = 0;
    for (let k = 0; k < b.pdfUV.length; k++) { sum += b.pdfUV[k]; maxRatio = Math.max(maxRatio, Math.abs(b.pdfUV[k] / a.pdfUV[k] - 1)); }
    expect(Math.abs(sum / b.pdfUV.length - 1)).toBeLessThan(1e-6);
    expect(maxRatio).toBeGreaterThan(0.05);
    // targetUV (diagnostics) is the floored kernel weights, ≈ realized up to the u16 quantization
    let q = 0;
    for (let k = 0; k < a.pdfUV.length; k++) q = Math.max(q, Math.abs(a.pdfUV[k] / a.targetUV[k] - 1));
    expect(q).toBeLessThan(2e-2);
  });

  it('determinism: identical inputs → identical bits', () => {
    const tex = synthEnv(128, 64);
    const a = buildEnvImportance(tex, 128, 64), b = buildEnvImportance(tex.slice(), 128, 64);
    expect(Buffer.from(a.pdfUV.buffer).equals(Buffer.from(b.pdfUV.buffer))).toBe(true);
    expect(Buffer.from(a.colAlias.buffer).equals(Buffer.from(b.colAlias.buffer))).toBe(true);
    expect(a.meanRgb).toEqual(b.meanRgb);
  });

  it('env mean M = (1/4π)Σ texel·ΔΩ: a constant map has M = the constant', () => {
    const W = 32, H = 16;
    const tex = new Float32Array(W * H * 4);
    for (let k = 0; k < W * H; k++) tex.set([2, 1, 0.5, 1], 4 * k);
    const t = buildEnvImportance(tex, W, H);
    t.meanRgb.forEach((m, k) => expect(m).toBeCloseTo([2, 1, 0.5][k], 12));
    expect(envMeanLuminance(t, 3, [1, 0.5, 2])).toBeCloseTo(3 * (0.2126 * 2 + 0.7152 * 0.5 + 0.0722 * 1), 12);
    expect(envPowerProxy(t, 1, [1, 1, 1], 2)).toBeCloseTo(4 * Math.PI ** 2 * 4 * envMeanLuminance(t, 1, [1, 1, 1]), 10);
  });

  it('real HDRI (studio_small_09 1k): tables build, Σ = 1, all cells positive', () => {
    let bytes: Uint8Array;
    try { bytes = new Uint8Array(readFileSync('validation/assets/downloaded/hdri/studio_small_09_1k.hdr')); } catch { return; }
    const img = decodeHdr(bytes);
    const t = buildEnvImportance(img.texels, img.width, img.height);
    expect([t.Wm, t.Hm]).toEqual([1024, 512]);
    let s = 0, mn = Infinity;
    for (let k = 0; k < t.pdfUV.length; k++) { s += t.pdfUV[k]; mn = Math.min(mn, t.pdfUV[k]); }
    expect(Math.abs(s / t.pdfUV.length - 1)).toBeLessThan(1e-6);
    expect(mn).toBeGreaterThan(0);
  });
});

describe('env as a light: selection, P(env) clamp, determinism (ENV-U8; math.md#light-selection)', () => {
  const table = buildEnvImportance(synthEnv(64, 32), 64, 32);
  const R = 2;

  it('env alone: P(env) = 1; env NEE off: no entry', () => {
    const s = new LightsState(sceneWith([], R), [0, 0, 0], new Float32Array(9));
    s.update([], { table, strength: 1, tint: [1, 1, 1], nee: true });
    expect(s.curSlot.envEntry).toBe(0);
    expect(s.envPmf()).toBe(1);
    s.update([], { table, strength: 1, tint: [1, 1, 1], nee: false });
    expect(s.curSlot.envEntry).toBe(NO_ENV_ENTRY);
    expect(s.curSlot.nEntries).toBe(0);
  });

  it('P(env) = clamp(Φ_env/ΣΦ̃, 0.1, 0.9), others share 1 − P(env) ∝ Φ̃ (realized from the integers)', () => {
    const s = new LightsState(sceneWith([], R), [0, 0, 0], new Float32Array(9));
    const phiEnv = envPowerProxy(table, 1, [1, 1, 1], s.sceneRadius);
    for (const [lp, expectP] of [[1e-6, P_ENV_MAX], [1e9, P_ENV_MIN]] as const) {
      s.update([point(0, lp)], { table, strength: 1, tint: [1, 1, 1], nee: true });
      expect(s.envPmf()).toBeCloseTo(expectP, 4);
    }
    // middle: unclamped ratio (point proxy = 4π·lum(I) = power; total analytic power = Φ_env → P(env) = 1/2)
    const lp = phiEnv / 3;
    const phiL = lightPowerProxy(point(0, lp), s.sceneRadius);
    const P = phiEnv / (phiEnv + phiL * 3);
    expect(P).toBeCloseTo(0.5, 12);
    s.update([point(0, lp), point(1, 2 * lp)], { table, strength: 1, tint: [1, 1, 1], nee: true });
    expect(s.envPmf()).toBeCloseTo(P, 4);
    expect(s.pmf(0) / s.pmf(1)).toBeCloseTo(0.5, 3);
    expect(s.pmf(0) + s.pmf(1) + s.envPmf()).toBeCloseTo(1, 6);
  });

  it('ENV-U8: rotation-only edits never touch the pmf; strength edits rebuild it as specified; bitwise determinism', () => {
    const s = new LightsState(sceneWith([], R), [0, 0, 0], new Float32Array(9));
    const lights = [point(0, 2 * envPowerProxy(table, 1, [1, 1, 1], s.sceneRadius))];   // P(env) = 1/3 at strength 1
    const u0 = s.update(lights, { table, strength: 1, tint: [1, 1, 1], nee: true });
    expect(u0.pmfChanged).toBe(true);
    const pmf0 = [s.pmf(0), s.envPmf()];
    // rotation lives only in the env uniform (cg, sg): the light state sees the same inputs → no rebuild, same bits
    const u1 = s.update(lights, { table, strength: 1, tint: [1, 1, 1], nee: true });
    expect(u1.pmfChanged).toBe(false);
    expect([s.pmf(0), s.envPmf()]).toEqual(pmf0);
    const words = (slot: number) => s.records.slice(s.slotBase[slot], s.slotBase[slot] + s.slotWords);
    // strength ×2 → rebuilt, P(env) follows the proxy
    const u2 = s.update(lights, { table, strength: 2, tint: [1, 1, 1], nee: true });
    expect(u2.pmfChanged).toBe(true);
    const phiL = lightPowerProxy(lights[0], s.sceneRadius), phiE = envPowerProxy(table, 2, [1, 1, 1], s.sceneRadius);
    expect(s.envPmf()).toBeCloseTo(Math.min(P_ENV_MAX, Math.max(P_ENV_MIN, phiE / (phiE + phiL))), 4);
    // back to strength 1 → identical bits as the first build (deterministic rebuild)
    s.update(lights, { table, strength: 1, tint: [1, 1, 1], nee: true });
    const a = words(s.cur);
    const s2 = new LightsState(sceneWith([], R), [0, 0, 0], new Float32Array(9));
    s2.update(lights, { table, strength: 1, tint: [1, 1, 1], nee: true });
    const b = s2.records.slice(s2.slotBase[s2.cur], s2.slotBase[s2.cur] + s2.slotWords);
    expect(Buffer.from(a.buffer).equals(Buffer.from(b.buffer))).toBe(true);
    // tint is radiometric too
    expect(s.update(lights, { table, strength: 1, tint: [0.5, 1, 1], nee: true }).pmfChanged).toBe(true);
  });

  it('env tables live in records at the LightsParams offsets', () => {
    const s = new LightsState(sceneWith([], R), [0, 0, 0], new Float32Array(9));
    s.update([], { table, strength: 1, tint: [1, 1, 1], nee: true });
    const u = new Uint32Array(s.paramsBytes());
    const [rowOff, colOff, pdfOff, log2W] = u.subarray(28, 32);
    expect(log2W).toBe(table.log2W);
    expect(Array.from(s.records.subarray(rowOff, rowOff + table.Hm))).toEqual(Array.from(table.rowAlias));
    expect(Array.from(s.records.subarray(colOff, colOff + 8))).toEqual(Array.from(table.colAlias.subarray(0, 8)));
    expect(new Float32Array(s.records.buffer, 4 * pdfOff, 4)).toEqual(table.pdfUV.subarray(0, 4));
    expect(u[27] & 2).toBe(2);   // LP_ENV_NEE
  });
});

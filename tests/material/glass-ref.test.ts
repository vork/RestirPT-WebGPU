// Glass lobe class G, CPU lane (plan §7.4 M3b = gap-glass §7.3 U-G1…U-G10, CPU parts): the f64 reference
// (tests/material/glass-ref.ts) against the spec's test vectors, quadratures and Tier-2 tables. The GPU lane
// (validation/gpu-tests/glass.gpu.test.ts) checks the WGSL against this reference.
import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import {
  GLASS_LUT_CHECKSUMS, TABLE_GGX_GLASS_E, TABLE_GGX_GLASS_EAVG, TABLE_GGX_GLASS_INV_E, TABLE_GGX_GLASS_INV_EAVG,
} from '../../src/core/render/luts/cycles-glass-luts.ts';
import { F0FromIor, fresnelDielectric, ggxLambda, type V3 } from './bsdf-ref.ts';
import {
  dirDeg, fresnelDielectricT, glassEval, glassFresnel, glassNode, glassSample, glassSampleFormWeight, principledMix,
  refractionNode, sphereQuadrature,
} from './glass-ref.ts';

const sat = (x: number) => Math.min(Math.max(x, 0), 1);

/** Deterministic PRNG. */
function rnd(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('U-G1 Fresnel (glass §2.2)', () => {
  it('F_diel(1, 1.5) = 0.04; F0 symmetric in η ↔ 1/η', () => {
    expect(fresnelDielectric(1, 1.5)).toBeCloseTo(0.04, 12);
    expect(F0FromIor(1 / 1.5)).toBeCloseTo(F0FromIor(1.5), 14);
  });

  it('remapped generalized Schlick == F_diel for 1/3.73 < η < 3.73 (1e-6), clipped at η = 4 and 1/4', () => {
    const remap = (c: number, eta: number) => {
      const F0 = F0FromIor(eta);
      return F0 + (1 - F0) * sat((fresnelDielectric(c, eta) - F0) / (1 - F0));
    };
    let worst = 0;
    for (const eta of [1 / 3.72, 1 / 2.4, 1 / 1.5, 1 / 1.1, 1.05, 1.33, 1.5, 2.4, 3.72]) {
      for (let i = 0; i <= 400; i++) worst = Math.max(worst, Math.abs(remap(i / 400, eta) - fresnelDielectric(i / 400, eta)));
    }
    expect(worst).toBeLessThan(1e-6);
    for (const eta of [4, 1 / 4]) {
      let dmin = Infinity;
      for (let i = 0; i <= 2000; i++) dmin = Math.min(dmin, fresnelDielectric(i / 2000, eta) - F0FromIor(eta));
      expect(dmin, `η = ${eta}: F_diel dips below F0 (the saturate clips)`).toBeLessThan(-1e-3);
    }
  });

  it('TIR: F = 1, the Refraction node has T = 0; below the critical angle cosT = −√(η² − 1 + c²)/η < 0', () => {
    const eta = 1 / 1.5;
    const cCrit = Math.sqrt(1 - eta * eta);
    expect(fresnelDielectricT(cCrit * 0.999, eta).F).toBe(1);
    const r = refractionNode([1, 1, 1], 0.3, 1.5, true);
    expect(glassFresnel(r, cCrit * 0.999).T).toEqual([0, 0, 0]);
    expect(glassFresnel(r, cCrit * 0.999).R).toEqual([0, 0, 0]);
    expect(glassFresnel(r, 0.9).T).toEqual([1, 1, 1]);
    for (const [c, e] of [[0.9, 1.5], [0.3, 1.5], [0.95, 1 / 1.5]]) {
      const { cosT } = fresnelDielectricT(c, e);
      expect(cosT).toBeLessThan(0);
      expect(cosT).toBeCloseTo(-Math.sqrt(e * e - 1 + c * c) / e, 14);
    }
  });
});

describe('U-G2 eval / pdf vectors (glass §3, Glass node, Color 1)', () => {
  // r, η_side, V (θ, φ), L (θ, φ, below), eval·cos, pdf (Cycles), valid
  const V: [number, number, [number, number], [number, number, boolean], number, number, boolean][] = [
    [0.5, 1.5, [30, 0], [40, 180, true], 2.466877e-1, 2.493586e-1, true],
    [0.5, 1.5, [30, 0], [15, 180, true], 1.032414e+1, 1.033565e+1, true],
    [0.5, 1.5, [30, 0], [30, 0, false], 2.579733e-3, 2.593031e-3, true],
    [0.3, 1.5, [60, 0], [20, 180, true], 4.155928e-2, 4.157036e-2, true],
    [0.5, 1 / 1.5, [20, 0], [35, 180, true], 7.401114e+0, 7.457269e+0, true],
    [0.5, 1 / 1.5, [60, 0], [70, 180, true], 4.183677e-1, 4.610455e-1, true],
    [1.0, 1.5, [80, 0], [80, 0, true], 1.097268e-1, 1.869842e-1, false],
    [0.5, 1.5, [30, 0], [60, 0, true], 6.523885e-3, 6.815053e-3, false],
  ];
  for (const [r, eta, v, l, f, pdf, valid] of V) {
    it(`r ${r} η ${eta.toFixed(4)} V ${v} L ${l}: eval ${f}, pdf ${pdf}, valid ${valid}`, () => {
      const back = eta < 1;
      const g = glassNode([1, 1, 1], r, back ? 1 / eta : eta, dirDeg(v[0], v[1])[2], back);
      expect(g.eta).toBeCloseTo(eta, 12);
      const e = glassEval(g, dirDeg(v[0], v[1]), dirDeg(l[0], l[1], l[2]));
      expect(Math.abs(e.f[0] - f) / f).toBeLessThan(1e-5);
      expect(Math.abs(e.pdfC - pdf) / pdf).toBeLessThan(1e-5);
      expect(e.valid).toBe(valid);
      expect(e.pdfV).toBe(valid ? e.pdfC : 0);
    });
  }
});

describe('U-G3 (CPU) the Cycles eval pdf integrates to > 1 in spurious cases; the valid-only pdf ≤ 1 (glass §2.3)', () => {
  // r, μ, η_side, ∫eval(all), ∫eval(valid) = sampler albedo, ∫pdf(all), ∫pdf(valid)
  const T: [number, number, number, number, number, number, number][] = [
    [0.533, 0.4, 1.5568, 1.04343, 0.93690, 1.1126, 0.9758],
    [0.533, 1.0, 1.5568, 1.00060, 0.98417, 1.0129, 0.9866],
    [1.0, 0.2, 1.5568, 1.30700, 0.70644, 1.9213, 0.9751],
    [0.533, 0.4, 0.6423, 0.82299, 0.78911, 0.9421, 0.9029],
    [1.0, 0.2, 0.6423, 0.88241, 0.50593, 1.2388, 0.7640],
    [0.2, 0.7, 1.5, 1.00060, 0.99951, 1.0008, 0.9997],
  ];
  for (const [r, mu, eta, eAll, eValid, pAll, pValid] of T) {
    it(`r ${r} μ ${mu} η ${eta}: ∫pdf_C = ${pAll}, ∫pdf_V = ${pValid}`, () => {
      const back = eta < 1;
      const g = glassNode([1, 1, 1], r, back ? 1 / eta : eta, mu, back);
      const Vd: V3 = [Math.sqrt(1 - mu * mu), 0, mu];
      const n = 1600;   // midpoint grid in (cosθ, φ): converged to ~1e-4 here (μ = 1 puts both lobes at the poles)
      let iE = 0, iEv = 0, iP = 0, iPv = 0;
      sphereQuadrature(n, 2 * n, (L) => {
        const e = glassEval(g, Vd, L);
        iE += e.f[0]; iP += e.pdfC; iPv += e.pdfV; if (e.valid) iEv += e.f[0];
        return 0;
      });
      const w = (2 / n) * (2 * Math.PI / (2 * n));
      const tol = 2e-3;
      expect(Math.abs(iP * w - pAll)).toBeLessThan(tol * pAll);
      expect(Math.abs(iPv * w - pValid)).toBeLessThan(tol * pValid);
      expect(Math.abs(iE * w - eAll)).toBeLessThan(tol * eAll);
      expect(Math.abs(iEv * w - eValid)).toBeLessThan(tol * eValid);
      expect(iPv * w).toBeLessThanOrEqual(1 + tol);
      if (pAll > 1.01) expect(iP * w, 'nobody "fixes" the χ² test by switching to the Cycles pdf').toBeGreaterThan(1.01);
    });
  }
});

describe('U-G4 (CPU) single-scatter albedo of the valid-only sampler (glass §3 table)', () => {
  // η_side, r, [μ, total, R part]...
  const TAB: [number, number, [number, number, number][]][] = [
    [1.5, 0.1, [[1, 1.0000, 0.0396], [0.7, 1.0000, 0.0510], [0.4, 0.9999, 0.1324], [0.1, 0.9959, 0.5610]]],
    [1.5, 0.25, [[1, 0.9995, 0.0398], [0.7, 0.9988, 0.0514], [0.4, 0.9948, 0.1276], [0.1, 0.9295, 0.3915]]],
    [1.5, 0.5, [[1, 0.9920, 0.0370], [0.7, 0.9793, 0.0467], [0.4, 0.9468, 0.0833], [0.1, 0.9163, 0.1717]]],
    [1.5, 1.0, [[1, 0.8932, 0.0127], [0.7, 0.8002, 0.0177], [0.4, 0.7190, 0.0279], [0.1, 0.7413, 0.0583]]],
    [1 / 1.5, 0.1, [[1, 0.9999, 0.0399], [0.7, 0.9998, 0.9957], [0.4, 0.9996, 0.9994], [0.1, 0.9939, 0.9938]]],
    [1 / 1.5, 0.25, [[1, 0.9960, 0.0412], [0.7, 0.9879, 0.8843], [0.4, 0.9807, 0.9745], [0.1, 0.8885, 0.8855]]],
    [1 / 1.5, 0.5, [[1, 0.9290, 0.0481], [0.7, 0.8626, 0.5403], [0.4, 0.8112, 0.7412], [0.1, 0.8105, 0.7911]]],
    [1 / 1.5, 1.0, [[1, 0.4168, 0.0258], [0.7, 0.4964, 0.1121], [0.4, 0.5053, 0.2220], [0.1, 0.5112, 0.4312]]],
    [1.33, 0.25, [[1, 0.9997, 0.0202], [0.7, 0.9989, 0.0287], [0.4, 0.9951, 0.0940], [0.1, 0.9318, 0.3589]]],
    [1.33, 0.5, [[1, 0.9944, 0.0186], [0.7, 0.9816, 0.0261], [0.4, 0.9480, 0.0583], [0.1, 0.9087, 0.1423]]],
    [1.33, 1.0, [[1, 0.9233, 0.0065], [0.7, 0.8181, 0.0094], [0.4, 0.7211, 0.0164], [0.1, 0.7197, 0.0395]]],
  ];
  const N = 120_000;
  for (const [eta, r, cols] of TAB) {
    it(`η ${eta.toFixed(4)} r ${r}: MC albedo (${N} samples/μ, stratified) within 4σ + 2e-4 of the table`, () => {
      const back = eta < 1;
      for (const [mu, total, rPart] of cols) {
        const g = glassNode([1, 1, 1], r, back ? 1 / eta : eta, mu, back);
        const Vd: V3 = [Math.sqrt(1 - mu * mu), 0, mu];
        const R = rnd(Math.round(eta * 1e4) * 31 + Math.round(r * 100) * 7 + Math.round(mu * 10));
        let s = 0, s2 = 0, sr = 0, sr2 = 0;
        const K = Math.floor(Math.sqrt(N));
        const n = K * K;
        for (let i = 0; i < n; i++) {
          // stratified (u_h1, u_h2) grid, random u_rt
          const u1 = (Math.floor(i / K) + R()) / K, u2 = ((i % K) + R()) / K;
          const smp = glassSample(g, Vd, [Math.min(u1, 1 - 1e-12), Math.min(u2, 1 - 1e-12), R()]);
          const w = smp.ok ? smp.weight[0] : 0;
          s += w; s2 += w * w;
          const wr = smp.ok && !smp.isT ? w : 0;
          sr += wr; sr2 += wr * wr;
        }
        const m = s / n, se = Math.sqrt(Math.max(s2 / n - m * m, 0) / n);
        const mr = sr / n, ser = Math.sqrt(Math.max(sr2 / n - mr * mr, 0) / n);
        // the table itself is a 1e6-sample MC rounded to 4 decimals
        expect(Math.abs(m - total), `μ ${mu}: ${m} ± ${se} vs ${total}`).toBeLessThan(4 * Math.hypot(se, se * Math.sqrt(n / 1e6)) + 2e-4);
        expect(Math.abs(mr - rPart), `μ ${mu} R: ${mr} ± ${ser} vs ${rPart}`).toBeLessThan(4 * Math.hypot(ser, ser * Math.sqrt(n / 1e6)) + 2e-4);
      }
    });
  }
});

describe('U-G5 weight consistency (CPU)', () => {
  it('weight == f/pdf (eval form) == Cycles sample form (R|T)/P·(1+Λ_I)/(1+Λ_I+Λ_O) to 1e-9; common forms agree', () => {
    const R = rnd(5);
    let worst = 0, worstCommon = 0, n = 0;
    for (const [r, eta, back] of [[0.3, 1.5, false], [0.5, 1.5, true], [0.8, 1.33, false], [0.2, 2.4, true], [1.0, 1.5, false]] as const) {
      for (let k = 0; k < 400; k++) {
        const mu = 0.05 + 0.95 * R();
        const Vd: V3 = [Math.sqrt(1 - mu * mu), 0, mu];
        const g = glassNode([0.9, 0.7, 0.5], r, eta, mu, back);
        const u: [number, number, number] = [R(), R(), R()];
        const s = glassSample(g, Vd, u);
        const sf = glassSampleFormWeight(g, Vd, u);
        if (!s.ok || !sf.ok) continue;
        n++;
        for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs(s.weight[c] - sf.weight[c]) / Math.max(sf.weight[c], 1e-12));
        // eval-form common = pdf_C·(1 + Λ_I)/(P_R | 1 − P_R) at the re-derived microfacet; sample form from the sampled one
        const e = glassEval(g, Vd, s.L);
        const lI = ggxLambda(g.a2, Vd[2]);
        const Hu = s.isT ? [-(g.eta * s.L[0] + Vd[0]), -(g.eta * s.L[1] + Vd[1]), -(g.eta * s.L[2] + Vd[2])] : [Vd[0] + s.L[0], Vd[1] + s.L[1], Vd[2] + s.L[2]];
        const fr = glassFresnel(g, (Hu[0] * Vd[0] + Hu[1] * Vd[1] + Hu[2] * Vd[2]) / Math.hypot(...Hu));
        const aR = (fr.R[0] + fr.R[1] + fr.R[2]) / 3, aT = (fr.T[0] + fr.T[1] + fr.T[2]) / 3;
        const lobe = s.isT ? aT / (aR + aT) : aR / (aR + aT);
        worstCommon = Math.max(worstCommon, Math.abs(e.pdfC * (1 + lI) / lobe - sf.commonSample) / sf.commonSample);
      }
    }
    expect(n).toBeGreaterThan(1500);
    expect(worst).toBeLessThan(1e-9);
    expect(worstCommon).toBeLessThan(1e-9);
  });

  it('delta weights: (1−F)√C/(1−P_R) and F/P_R (Principled, smooth)', () => {
    const p = principledMix({ base: [0.64, 0.36, 0.16], metallic: 0, roughness: 0, ior: 1.5, specLevel: 0.5, specTint: [1, 1, 1], transmission: 1 }, 0.8);
    const g = p.glass!;
    const Vd: V3 = [0.6, 0, 0.8];
    const F = fresnelDielectric(0.8, 1.5);
    const sq = [0.8, 0.6, 0.4];
    const a = (sq[0] + sq[1] + sq[2]) / 3;
    const PR = F / (F + (1 - F) * a);
    const t = glassSample(g, Vd, [0.5, 0.5, 0.999]);
    expect(t.isDelta && t.isT).toBe(true);
    t.weight.forEach((w, c) => expect(w).toBeCloseTo((1 - F) * sq[c] / (1 - PR), 12));
    const rr = glassSample(g, Vd, [0.5, 0.5, 0.0]);
    expect(rr.isDelta && !rr.isT).toBe(true);
    rr.weight.forEach((w) => expect(w).toBeCloseTo(F / PR, 12));
  });
});

describe('U-G6 lobe-selection weights (glass §2.6 vectors, 1e-4)', () => {
  const P = (t: number, o: Partial<Parameters<typeof principledMix>[0]> = {}) => ({ base: [0.8, 0.8, 0.8] as V3, metallic: 0, roughness: 0.5, ior: 1.5, specLevel: 0.5, specTint: [1, 1, 1] as V3, transmission: t, ...o });
  it('t = 1: sw_G 0.8992 (μ 1), 0.8773 (μ 0.4), 0.1013 (μ 0.4 backfacing, TIR at N); q(G) = 1', () => {
    for (const [mu, back, sw] of [[1, false, 0.8992], [0.4, false, 0.8773], [0.4, true, 0.1013]] as const) {
      const m = principledMix(P(1), mu, back);
      expect(Math.abs(m.swG - sw), `μ ${mu} back ${back}: ${m.swG}`).toBeLessThan(1e-4);
      expect(m.q.G).toBe(1);
    }
  });
  it('t = 0.5: μ 1 → sw_G 0.4496, q = (G 0.5267, S 0.0237, D 0.4496); μ 0.1 → 0.2934, q = (0.4111, 0.1427, 0.4463)', () => {
    const a = principledMix(P(0.5), 1);
    expect(Math.abs(a.swG - 0.4496)).toBeLessThan(1e-4);
    expect(Math.abs(a.q.G - 0.5267)).toBeLessThan(1e-4);
    expect(Math.abs(a.q.S - 0.0237)).toBeLessThan(1e-4);
    expect(Math.abs(a.q.D - 0.4496)).toBeLessThan(1e-4);
    const b = principledMix(P(0.5), 0.1);
    expect(Math.abs(b.swG - 0.2934)).toBeLessThan(1e-4);
    expect(Math.abs(b.q.G - 0.4111)).toBeLessThan(1e-4);
    expect(Math.abs(b.q.S - 0.1427)).toBeLessThan(1e-4);
    expect(Math.abs(b.q.D - 0.4463)).toBeLessThan(1e-4);
  });
  it('C = (0.9, 0.6, 0.3), m 0.25, t 0.5, r 0.3, μ 0.7: sw_M 0.1505, sw_G 0.2891, sw_S 0.0197, sw_D 0.2132, q(G) 0.4299', () => {
    const m = principledMix(P(0.5, { base: [0.9, 0.6, 0.3], metallic: 0.25, roughness: 0.3 }), 0.7);
    expect(Math.abs(m.swM - 0.1505)).toBeLessThan(1e-4);
    expect(Math.abs(m.swG - 0.2891)).toBeLessThan(1e-4);
    expect(Math.abs(m.swS - 0.0197)).toBeLessThan(1e-4);
    expect(Math.abs(m.swD - 0.2132)).toBeLessThan(1e-4);
    expect(Math.abs(m.q.G - 0.4299)).toBeLessThan(1e-4);
  });
});

describe('U-G7 delta per side (CPU)', () => {
  it('r 0.0037 is delta, 0.0038 is not (α² vs 2e-10)', () => {
    expect(glassNode([1, 1, 1], 0.0037, 1.5, 0.8, false).delta).toBe(true);
    expect(glassNode([1, 1, 1], 0.0038, 1.5, 0.8, false).delta).toBe(false);
  });
  it('|η − 1| < 1e-4 with r 0.3: G_T delta (sampled weight T/(1−P_R), no pdf), G_R rough', () => {
    const g = glassNode([1, 1, 1], 0.3, 1.00005, 0.8, false);
    expect(g.delta).toBe(false);
    expect(g.deltaT).toBe(true);
    const Vd: V3 = [0.6, 0, 0.8];
    const R = rnd(9);
    let nT = 0, nR = 0;
    for (let i = 0; i < 2000; i++) {
      const s = glassSample(g, Vd, [R(), R(), R()]);
      if (!s.ok) continue;
      if (s.isT) { nT++; expect(s.isDelta).toBe(true); expect(s.pdf).toBe(0); } else { nR++; expect(s.isDelta).toBe(false); expect(s.pdf).toBeGreaterThan(0); }
    }
    expect(nT).toBeGreaterThan(1500);
    // the Cycles eval still returns a rough (spurious) BTDF there: f > 0, pdf_V = 0
    const e = glassEval(g, Vd, [-0.6, 0, -0.8]);
    expect(e.f[0]).toBeGreaterThan(0);
    expect(e.valid).toBe(false);
    expect(e.pdfV).toBe(0);
  });
});

describe('U-G9 Tier-2 glass tables (gap-glass §2.7; CPU only — the renderer is Tier 1, M7 ports Tier 2)', () => {
  const sha = (a: Float32Array) => createHash('sha256').update(Buffer.from(a.buffer, a.byteOffset, a.byteLength)).digest('hex');
  it('checksums: counts, sums and f32 SHA-256 as extracted', () => {
    for (const [name, t] of [['table_ggx_glass_E', TABLE_GGX_GLASS_E], ['table_ggx_glass_Eavg', TABLE_GGX_GLASS_EAVG],
      ['table_ggx_glass_inv_E', TABLE_GGX_GLASS_INV_E], ['table_ggx_glass_inv_Eavg', TABLE_GGX_GLASS_INV_EAVG]] as const) {
      const c = GLASS_LUT_CHECKSUMS[name];
      expect(t.length).toBe(c.count);
      expect(Math.abs(t.reduce((s, x) => s + x, 0) - c.sum)).toBeLessThan(1e-4);
      expect(sha(t)).toBe(c.sha256F32);
    }
    expect(GLASS_LUT_CHECKSUMS.table_ggx_glass_E.sum).toBeCloseTo(3758.664856, 5);
    expect(GLASS_LUT_CHECKSUMS.table_ggx_glass_inv_Eavg.sum).toBeCloseTo(216.090299, 5);
  });

  it('energy_scale = 1/E at the §2.7 grid (white glass, Fss ≡ 1)', () => {
    const read3 = (t: Float32Array, x: number, y: number, z: number) => {
      const r1 = (off: number, xx: number) => { const xs = sat(xx) * 15, i = Math.min(Math.trunc(xs), 15), j = Math.min(i + 1, 15), f = xs - i; return f === 0 ? t[off + i] : (1 - f) * t[off + i] + f * t[off + j]; };
      const r2 = (off: number, xx: number, yy: number) => { const ys = sat(yy) * 15, i = Math.min(Math.trunc(ys), 15), j = Math.min(i + 1, 15), f = ys - i; const d0 = r1(off + 16 * i, xx); return f === 0 ? d0 : (1 - f) * d0 + f * r1(off + 16 * j, xx); };
      const zs = sat(z) * 15, i = Math.min(Math.trunc(zs), 15), j = Math.min(i + 1, 15), f = zs - i;
      const d0 = r2(256 * i, x, y);
      return f === 0 ? d0 : (1 - f) * d0 + f * r2(256 * j, x, y);
    };
    const scaleOf = (etaSide: number, r: number, mu: number) => {
      let ior = etaSide, t = TABLE_GGX_GLASS_E;
      if (ior < 1) { ior = 1 / ior; t = TABLE_GGX_GLASS_INV_E; }
      const z = Math.sqrt(Math.abs((ior - 1) / (ior + 1)));
      return 1 / read3(t, r, mu, z);
    };
    const want: [number, number, number, number][] = [
      [1.5, 0.25, 1, 1.0005], [1.5, 0.25, 0.4, 1.0057], [1.5, 0.5, 1, 1.0083], [1.5, 0.5, 0.4, 1.0568], [1.5, 1, 1, 1.1200], [1.5, 1, 0.4, 1.3917],
      [1 / 1.5, 0.25, 1, 1.0044], [1 / 1.5, 0.25, 0.4, 1.0191], [1 / 1.5, 0.5, 1, 1.0762], [1 / 1.5, 0.5, 0.4, 1.2132], [1 / 1.5, 1, 1, 2.3706], [1 / 1.5, 1, 0.4, 1.9477],
    ];
    for (const [eta, r, mu, s] of want) expect(Math.abs(scaleOf(eta, r, mu) - s), `η ${eta} r ${r} μ ${mu}`).toBeLessThan(1e-4);
  });

  it('darkening path for Fss ≠ 1: (1 + Fms·missing)/energy_scale ≤ 1, → 1 as Fss → 1, monotone in Fss', () => {
    const E = 0.8, Eavg = 0.75, missing = (1 - E) / E, energy = 1 + missing;
    const dark = (Fss: number) => (1 + (Fss * Eavg / (1 - Fss * (1 - Eavg))) * missing) / energy;
    expect(dark(1)).toBeCloseTo(1, 12);
    let prev = 0;
    for (const f of [0.1, 0.3, 0.5, 0.8, 0.99]) { const d = dark(f); expect(d).toBeLessThanOrEqual(1); expect(d).toBeGreaterThan(prev); prev = d; }
  });
});

describe('U-G10 ggx_lambda with a negative cosine (the gap-bsdf WGSL fix)', () => {
  it('Λ(a2, −c) == Λ(a2, c)', () => {
    for (const a2 of [1e-6, 0.01, 0.2, 1]) for (const c of [0.05, 0.3, 0.7, 1]) expect(ggxLambda(a2, -c)).toBe(ggxLambda(a2, c));
  });
});

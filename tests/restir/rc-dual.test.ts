// CPU lane of the f64 dual (restir-api.md §6.1 T3-D, R6): hand-built configurations of the pair predicate, the primary
// threshold and the f64 pdfs the dual uses (tests/restir/rc-dual.ts). The GPU agreement runs in restir-shift.gpu.test.ts.
import { describe, expect, it } from 'vitest';
import {
  ALPHA_LOGIC, FLT_MAX, LOBE, MARGIN_LOGIC, RCK, RCT, RC_MARGIN_DISCRETE, dualLobes, eventNone, isLogic, lobeRoughnessF64, pairTestF64,
  pdfF64, primaryThresholdF64, type DualMat, type RcEventD, type RcVertexD,
} from './rc-dual.ts';
import { sphereQuadrature } from '../material/glass-ref.ts';
import type { V3 } from '../material/bsdf-ref.ts';

const surf = (pos: V3, ng: V3, diffuseOnly = false): RcVertexD => ({ pos, ng, kind: RCK.SURFACE, diffuseOnly });
const ev = (lobe: number, alpha: number, pMarg: number, delta = false): RcEventD => ({ lobe, delta, alpha, pMarg });
const P = { alphaMin: 0.2 };

describe('pairTestF64 (math.md#rc-predicate)', () => {
  const a = surf([0, 0, 0], [0, 0, 1]);
  const b = surf([0, 0, 2], [0, 0, -1]);
  // t² = 4, cos_a = cos_b = 1: rayFP = 4/p̄_a, invFP = 4/p̄_b
  it('D, R, F, I in order; ≥ at the threshold', () => {
    expect(pairTestF64(a, ev(LOBE.D, 1, 1, true), b, ev(LOBE.D, 1, 1), 1, P)).toMatchObject({ ok: false, term: RCT.D, margin: RC_MARGIN_DISCRETE });
    expect(pairTestF64(a, ev(LOBE.GT, 1, 1), b, ev(LOBE.D, 1, 1), 1, P)).toMatchObject({ ok: false, term: RCT.D });
    expect(pairTestF64(a, ev(LOBE.S, 0.19, 1), b, ev(LOBE.D, 1, 1), 1, P)).toMatchObject({ ok: false, term: RCT.R });
    expect(pairTestF64(a, ev(LOBE.S, 0.21, 8), b, ev(LOBE.D, 1, 1), 1, P)).toMatchObject({ ok: false, term: RCT.F });
    expect(pairTestF64(a, ev(LOBE.S, 0.21, 4), b, ev(LOBE.D, 1, 1), 1, P).ok).toBe(true);           // rayFP = thr exactly
    expect(pairTestF64(a, ev(LOBE.S, 0.21, 1), b, ev(LOBE.S, 0.3, 8), 1, P)).toMatchObject({ ok: false, term: RCT.I });
    expect(pairTestF64(a, ev(LOBE.S, 0.21, 1), b, ev(LOBE.S, 0.3, 1, true), 1, P)).toMatchObject({ ok: false, term: RCT.D });
    expect(pairTestF64(a, ev(LOBE.S, 0.21, 1), b, ev(LOBE.GT, 0.3, 1), 1, P)).toMatchObject({ ok: false, term: RCT.D });
  });
  it('I skipped for LIGHT / ENV / diffuse-only receivers; ENV needs only D ∧ R', () => {
    expect(pairTestF64(a, ev(LOBE.D, 1, 1), surf([0, 0, 2], [0, 0, -1], true), ev(LOBE.D, 1, 1e9), 1, P).ok).toBe(true);
    expect(pairTestF64(a, ev(LOBE.D, 1, 1), { ...b, kind: RCK.LIGHT }, eventNone(), 1, P).ok).toBe(true);
    expect(pairTestF64(a, ev(LOBE.S, 0.3, 1e9), { pos: [0, 0, 0], ng: [0, 0, 0], kind: RCK.ENV, diffuseOnly: false }, eventNone(), 1e9, P).ok).toBe(true);
    expect(pairTestF64(a, ev(LOBE.S, 0.1, 1), { pos: [0, 0, 0], ng: [0, 0, 0], kind: RCK.ENV, diffuseOnly: false }, eventNone(), 1, P).ok).toBe(false);
  });
  it('cosines at the RECEIVING vertex for F and at the sender for I; guards fail the pair', () => {
    const tilted = surf([0, 0, 2], [0, Math.SQRT1_2, -Math.SQRT1_2]);        // cos_b = 1/√2: rayFP = 4√2/p̄
    expect(pairTestF64(a, ev(LOBE.D, 1, 5.6), tilted, ev(LOBE.D, 1, 1), 1, P).ok).toBe(true);
    expect(pairTestF64(a, ev(LOBE.D, 1, 5.7), tilted, ev(LOBE.D, 1, 1), 1, P).ok).toBe(false);
    expect(pairTestF64(a, ev(LOBE.D, 1, 0), b, ev(LOBE.D, 1, 1), 1, P)).toMatchObject({ ok: false, term: RCT.GUARD });
    expect(pairTestF64(a, ev(LOBE.D, 1, 1), surf([0, 0, 1e-7], [0, 0, 1]), ev(LOBE.D, 1, 1), 1e-30, P)).toMatchObject({ ok: false, term: RCT.GUARD });
    expect(pairTestF64(a, ev(LOBE.D, 1, 1), surf([1, 0, 0], [0, 0, 1]), ev(LOBE.D, 1, 1), 1e-30, P)).toMatchObject({ ok: false, term: RCT.GUARD });
  });
  it('2022 criteria: min roughness of both events and d_min', () => {
    const p22 = { alphaMin: 0.2, crit2022: true, dmin: 1 };
    expect(pairTestF64(a, ev(LOBE.D, 1, 1), b, ev(LOBE.S, 0.19, 1), 0, p22)).toMatchObject({ ok: false, term: RCT.R });
    expect(pairTestF64(a, ev(LOBE.D, 1, 1), b, ev(LOBE.S, 0.21, 1), 0, p22).ok).toBe(true);
    expect(pairTestF64(a, ev(LOBE.D, 1, 1), b, ev(LOBE.S, 0.21, 1), 0, { ...p22, dmin: 2.5 })).toMatchObject({ ok: false, term: RCT.F });
    expect(pairTestF64(a, ev(LOBE.D, 1, 1), { ...b, kind: RCK.LIGHT }, eventNone(), 0, p22).ok).toBe(true);
  });
  it('margins classify LOGIC vs FP-BOUNDARY', () => {
    const r = pairTestF64(a, ev(LOBE.D, 1, 4 * (1 + 1e-7)), b, ev(LOBE.D, 1, 1), 1, P);
    expect(r.ok).toBe(false);
    expect(Math.abs(r.margin)).toBeLessThan(MARGIN_LOGIC);
    expect(isLogic(r)).toBe(false);
    expect(isLogic({ ok: false, margin: -0.5, term: RCT.F })).toBe(true);
    expect(isLogic({ ok: false, margin: 0.5 * ALPHA_LOGIC, term: RCT.R })).toBe(false);
  });
});

describe('primaryThresholdF64', () => {
  it('τ·‖x₁ − x₀‖²·4π/|cos|', () => {
    expect(primaryThresholdF64([0, 0, 3], [0, 0, 0], [0, 0, 1], 2e-4)).toBeCloseTo(2e-4 * 9 * 4 * Math.PI, 12);
    expect(primaryThresholdF64([3, 0, 0], [0, 0, 0], [0, 0, 1], 2e-4)).toBeCloseTo(2e-4 * 9 * 4 * Math.PI / 1e-6, 3);
  });
});

describe('f64 pdfs of the dual (V1 / V2)', () => {
  const v1 = (r: number, mix: number): DualMat => ({ model: 0, base: [0.6, 0.6, 0.6], metallic: mix, roughness: r, ior: 1.5, specLevel: 0.5, specTint: [0.9, 0.9, 0.9], transmission: 0 });
  const v2 = (m: number, r: number): DualMat => ({ model: 1, base: [0.8, 0.5, 0.3], metallic: m, roughness: r, ior: 1.5, specLevel: 0.5, specTint: [1, 1, 1], transmission: 0 });
  const ng: V3 = [0, 0, 1];
  const V: V3 = [Math.sin(0.6), 0, Math.cos(0.6)];
  for (const [name, m] of [['V1 mix r 0.3', v1(0.3, 0.5)], ['V2 metal r 0.25', v2(1, 0.25)], ['V2 dielectric r 0.4', v2(0, 0.4)], ['Lambert', v1(0.5, 0)]] as const) {
    it(`${name}: marginal = pD + pS, ∫ p̄ dω = 1 (valid-only density of a proper sampler, up to Ng rejection)`, () => {
      const I = sphereQuadrature(400, 400, (L) => pdfF64(m, ng, ng, V, L).marg);
      expect(I).toBeGreaterThan(0.97);
      expect(I).toBeLessThan(1.0005);
      const p = pdfF64(m, ng, ng, V, [0.1, 0.2, Math.sqrt(1 - 0.05)]);
      expect(p.marg).toBeCloseTo(p.pD + p.pS, 14);
    });
  }
  it('lobe structure and roughness (D 1, S r, singular S 0, NEE max)', () => {
    const l = dualLobes(v1(0.3, 0.5), 0.8);
    expect(l.hasD && l.hasS && !l.diffuseOnly).toBe(true);
    expect(lobeRoughnessF64(v1(0.3, 0.5), LOBE.D, l)).toBe(1);
    expect(lobeRoughnessF64(v1(0.3, 0.5), LOBE.S, l)).toBeCloseTo(0.3, 12);
    expect(lobeRoughnessF64(v1(0.3, 0.5), LOBE.NEE, l)).toBe(1);
    const mir = v1(0, 1);
    expect(lobeRoughnessF64(mir, LOBE.S, dualLobes(mir, 0.8))).toBe(0);
    expect(dualLobes(v1(0.5, 0), 0.8).diffuseOnly).toBe(true);
    expect(lobeRoughnessF64(mir, LOBE.NONE, dualLobes(mir, 0.8))).toBe(FLT_MAX);
  });
  it('two-sided flip: V below Ng evaluates the flipped frame', () => {
    const m = v1(0.5, 0);
    const up = pdfF64(m, ng, ng, V, [0, 0, 1]).marg;
    const dn = pdfF64(m, ng, ng, [V[0], V[1], -V[2]], [0, 0, -1]).marg;
    expect(dn).toBeCloseTo(up, 12);
  });
});

// T8 (CPU part): the f64 reference BSDF against the published vectors (gap-bsdf §13.4–13.5, U-2, U-6, U-9) and
// WGSL composition of the BSDF modules. The GPU lanes (validation/gpu-tests/bsdf.gpu.test.ts) check the WGSL
// against this reference.
import { describe, expect, it } from 'vitest';
import { composeWgsl } from '../../src/core/gpu/wgsl-composer.ts';
import { lutDefines } from '../../src/core/render/luts/lut-layout.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import {
  F0FromIor, f82TintB, fresnelDielectric, fresnelF82, fresnelGenSchlickIor, iorFromF0, prepare, evalLocal, dirFromTheta,
  type MatParams, type V2Params, type V3,
} from './bsdf-ref.ts';
import { albedoQuadrature } from './quadrature.ts';

const v2 = (o: Partial<V2Params> = {}): V2Params => ({ model: 1, base: [0.8, 0.8, 0.8], metallic: 0, roughness: 0.5, ior: 1.5, specLevel: 0.5, specTint: [1, 1, 1], ...o });

describe('Fresnel identities (gap-bsdf U-2)', () => {
  it('dielectric, generalized Schlick == real Fresnel for η ≤ 2+√3, F82 tint, ior_from_F0', () => {
    expect(fresnelDielectric(1, 1.5)).toBeCloseTo(0.04, 12);
    expect(fresnelDielectric(0.1, 1 / 1.5)).toBe(1);   // TIR
    for (const eta of [1.01, 1.33, 1.5, 2, 3, 3.7]) {
      for (let i = 0; i <= 100; i++) {
        const c = i / 100;
        expect(Math.abs(fresnelGenSchlickIor(c, eta, [F0FromIor(eta), F0FromIor(eta), F0FromIor(eta)])[0] - fresnelDielectric(c, eta))).toBeLessThan(1e-9);
      }
    }
    // η = 4: the Brewster dip goes below F0 and the saturate clips (cycles-verify C4)
    let maxDiff = 0;
    for (let i = 0; i <= 1000; i++) { const c = i / 1000; maxDiff = Math.max(maxDiff, Math.abs(fresnelGenSchlickIor(c, 4, [F0FromIor(4), 0, 0])[0] - fresnelDielectric(c, 4))); }
    expect(maxDiff).toBeGreaterThan(1e-3);
    // F82 with tint 1 == Schlick; F82(1/7) == Schlick(1/7)·tint (the definition of B)
    const F0: V3 = [0.9, 0.6, 0.3];
    const tint: V3 = [1, 0.5, 0.25];
    const schlick = (c: number) => F0.map((f) => f + (1 - f) * (1 - c) ** 5);
    expect(fresnelF82(0.3, F0, f82TintB(F0, [1, 1, 1]))).toEqual(schlick(0.3).map((x) => Math.min(Math.max(x, 0), 1)));
    const f82 = fresnelF82(1 / 7, F0, f82TintB(F0, tint));
    schlick(1 / 7).forEach((s, k) => expect(Math.abs(f82[k] - s * tint[k])).toBeLessThan(1e-12));
    for (const eta of [1.01, 1.5, 3, 10, 18.9]) expect(iorFromF0(F0FromIor(eta))).toBeCloseTo(eta, 9);
  });
});

describe('V2 closure weights and q(S) (gap-bsdf §13.4, U-6)', () => {
  // [params, mu, w_D (rgb or grey), Λ, sw_M, sw_S, sw_D, q(S)]
  const rows: [V2Params, number, number[], number, number, number, number, number][] = [
    [v2(), 1.0, [0.7676], 0.9595, 0, 0.0405, 0.7676, 0.0502],
    [v2(), 0.4, [0.7189], 0.8987, 0, 0.1013, 0.7189, 0.1236],
    [v2(), 0.1, [0.6371], 0.7963, 0, 0.2037, 0.6371, 0.2422],
    [v2({ specLevel: 1 }), 1.0, [0.7355], 0.9193, 0, 0.0807, 0.7355, 0.0988],
    [v2({ specLevel: 1 }), 0.1, [0.6061], 0.7576, 0, 0.2424, 0.6061, 0.2857],
    [v2({ specLevel: 0.25 }), 1.0, [0.7837], 0.9796, 0, 0.0204, 0.7837, 0.0254],
    [v2({ base: [0.9, 0.6, 0.3], metallic: 0.5 }), 1.0, [0.4318, 0.2878, 0.1439], 0.9595, 0.3000, 0.0203, 0.2878, 0.5267],
    [v2({ base: [0.9, 0.6, 0.3], metallic: 0.5 }), 0.4, [0.4044, 0.2696, 0.1348], 0.8987, 0.3108, 0.0507, 0.2696, 0.5728],
    [v2({ base: [0.9, 0.6, 0.3], metallic: 0.5 }), 0.1, [0.3584, 0.2389, 0.1195], 0.7963, 0.3344, 0.1018, 0.2389, 0.6461],
    [v2({ base: [0.9, 0.6, 0.3], roughness: 0.3, specTint: [1, 0.5, 0.25] }), 1.0, [0.8639, 0.5759, 0.2880], 0.9599, 0, 0.0234, 0.5759, 0.0391],
    [v2({ base: [0.9, 0.6, 0.3], roughness: 0.3, specTint: [1, 0.5, 0.25] }), 0.1, [0.5615, 0.3743, 0.1872], 0.6239, 0, 0.3653, 0.3743, 0.4939],
  ];
  it.each(rows.map((r, i) => [i, ...r] as const))('row %i', (_i, p, mu, wD, lam, swM, swS, swD, qS) => {
    const c = prepare(p, mu);
    const tol = 1.01e-4;
    const w = wD.length === 1 ? [wD[0], wD[0], wD[0]] : wD;
    w.forEach((x, k) => expect(Math.abs(c.wD[k] - x), `wD[${k}]`).toBeLessThan(tol));
    expect(Math.abs(c.lambda - lam)).toBeLessThan(tol);
    expect(Math.abs(c.swM - swM)).toBeLessThan(tol);
    expect(Math.abs(c.swSpec - swS)).toBeLessThan(tol);
    expect(Math.abs(c.swD - swD)).toBeLessThan(tol);
    expect(Math.abs(c.qS - qS)).toBeLessThan(tol);
    expect(c.qD + c.qS).toBeCloseTo(1, 12);
  });
  it('IOR level: specularFactor 2 → η\' = 1.7888; 0.5 → 1.3294 (gap-bsdf §10.2)', () => {
    expect(prepare(v2({ specLevel: 1 }), 1).etaS).toBeCloseTo(1.7888, 4);
    expect(prepare(v2({ specLevel: 0.25 }), 1).etaS).toBeCloseTo(1.3294, 4);
    const c0 = prepare(v2({ specLevel: 0 }), 0.5);   // η' = 1: no specular closure ⇒ diffuse-only
    expect(c0.hasSpec).toBe(false);
    expect(c0.hasS).toBe(false);
    expect(c0.qD).toBe(1);
  });
});

describe('directional albedo A(μ) (gap-bsdf §13.5, Tier 1, single scatter)', () => {
  const V = (mu: number): V3 => [Math.sqrt(1 - mu * mu), 0, mu];
  it('dielectric C = 0.8, IOR 1.5: A = spec_true + C·Λ', () => {
    const rows: [number, number, number, number][] = [   // r, mu, A, q(S)
      [0.25, 1.0, 0.8078, 0.0496], [0.25, 0.4, 0.8247, 0.1562], [0.25, 0.1, 0.8450, 0.4882],
      [0.5, 1.0, 0.8046, 0.0502], [0.5, 0.7, 0.8040, 0.0657], [0.5, 0.1, 0.8089, 0.2422],
      [1.0, 1.0, 0.7796, 0.0511], [1.0, 0.4, 0.7836, 0.0683], [1.0, 0.1, 0.7956, 0.0964],
    ];
    for (const [r, mu, A, qS] of rows) {
      const a = albedoQuadrature(v2({ roughness: r }), V(mu), 1024);
      expect(Math.abs(a[0] - A), `r ${r} mu ${mu}: ${a[0]}`).toBeLessThan(5e-4);
      expect(Math.abs(prepare(v2({ roughness: r }), mu).qS - qS)).toBeLessThan(1.01e-4);
    }
  });
  it('metal m = 1, base (0.9, 0.6, 0.3), tint white', () => {
    // r = 0.25: gap-bsdf §13.5 lists (0.8948, 0.5965, 0.2983) at μ = 1 and (0.8457, 0.6943, 0.5429) at μ = 0.1, but its
    // 700² VNDF quadrature is ~1e-3 off at r = 0.25 (its F = 1 value 0.9942 vs ggx_E = 0.99560; this quadrature gives
    // 0.99569, and A ≥ F0·E(0.25, 1) = 0.8960 rules 0.8948 out). The r = 0.25 rows use this quadrature's converged
    // values (n = 1024 and 2048 agree to 1e-5); the other rows are gap-bsdf's.
    const rows: [number, number, V3][] = [
      [0.25, 1.0, [0.8961, 0.5974, 0.2987]], [0.25, 0.1, [0.8463, 0.6947, 0.5432]],
      [0.5, 0.4, [0.7639, 0.5227, 0.2814]], [1.0, 0.7, [0.3411, 0.2277, 0.1143]], [1.0, 0.1, [0.6866, 0.4657, 0.2448]],
    ];
    for (const [r, mu, A] of rows) {
      const a = albedoQuadrature(v2({ base: [0.9, 0.6, 0.3], metallic: 1, roughness: r }), V(mu), 1024);
      A.forEach((x, k) => expect(Math.abs(a[k] - x), `r ${r} mu ${mu} [${k}]: ${a[k]}`).toBeLessThan(5e-4));
    }
    // metal sample weight avg(mix(F0, 1, S5)): 0.6 at μ = 1; 0.6687 at r = 0.5, μ = 0.1
    expect(prepare(v2({ base: [0.9, 0.6, 0.3], metallic: 1, roughness: 0.5 }), 1).swM).toBeCloseTo(0.6, 4);
    expect(prepare(v2({ base: [0.9, 0.6, 0.3], metallic: 1, roughness: 0.5 }), 0.1).swM).toBeCloseTo(0.6687, 4);
  });
  it('V1 glossy (F = 1) albedo equals ggx_E (gap-bsdf §13.3 independent quadrature)', () => {
    const p: MatParams = { model: 0, diffuse: [0, 0, 0], glossy: [1, 1, 1], roughness: 0.5, mix: 1 };
    const want = [[1, 0.9157], [0.7, 0.8853], [0.4, 0.8443], [0.1, 0.8916]];
    for (const [mu, e] of want) expect(Math.abs(albedoQuadrature(p, V(mu), 1024)[0] - e)).toBeLessThan(5e-4);
  });
});

describe('reciprocity (gap-bsdf U-9)', () => {
  it('V1 is reciprocal: f(V,L)/(N·L) == f(L,V)/(N·V)', () => {
    const p: MatParams = { model: 0, diffuse: [0.8, 0.5, 0.2], glossy: [0.9, 0.9, 0.9], roughness: 0.3, mix: 0.4 };
    for (const [a, b] of [[10, 50], [0, 80], [45, 45.5], [70, 20]]) {
      const V = dirFromTheta(a, 0.3), L = dirFromTheta(b, 2.5);
      const e1 = evalLocal(p, V, L), e2 = evalLocal(p, L, V);
      for (let k = 0; k < 3; k++) {
        const f1 = (e1.fD[k] + e1.fS[k]) / L[2], f2 = (e2.fD[k] + e2.fS[k]) / V[2];
        expect(Math.abs(f1 / f2 - 1)).toBeLessThan(1e-12);
      }
    }
  });
  it('V2 is NOT reciprocal: f_D(V,L)/f_D(L,V) = Λ(μ_V)/Λ(μ_L) = 1.2049 at μ_V = 1, μ_L = 0.1', () => {
    const V: V3 = [0, 0, 1], L: V3 = [Math.sqrt(1 - 0.01), 0, 0.1];
    const a = evalLocal(v2(), V, L).fD[0] / L[2];
    const b = evalLocal(v2(), L, V).fD[0] / V[2];
    expect(a / b).toBeCloseTo(0.9595 / 0.7963, 3);
  });
});

describe('WGSL composition', () => {
  const base = { ...lutDefines({ declare: { group: 0, binding: 0 } }) };
  it('bsdf.wgsl composes standalone and exposes the contract API', () => {
    const code = composeWgsl('material/bsdf.wgsl', { sources: shaderSources, defines: base }).code;
    for (const fn of [
      'fn bsdf_eval(m: MatEval, V: vec3f, L: vec3f) -> BsdfEval',
      'fn bsdf_sample(m: MatEval, V: vec3f, u: vec4f) -> BsdfSample',
      'fn bsdf_eval_lobe(m: MatEval, V: vec3f, L: vec3f, lobe: u32) -> vec4f',
      'fn bsdf_pdf_marginal(m: MatEval, V: vec3f, L: vec3f) -> f32',
      'fn bsdf_sample_support(m: MatEval, V: vec3f, L: vec3f, lobe: u32) -> bool',
      'fn lobe_roughness(m: MatEval, lobe: u32) -> f32',
    ]) expect(code, fn).toContain(fn);
    for (const c of ['LOBE_D: u32 = 0u', 'LOBE_S: u32 = 1u', 'LOBE_GR: u32 = 2u', 'LOBE_GT: u32 = 3u', 'LOBE_NEE: u32 = 4u', 'LOBE_NONE: u32 = 5u']) expect(code).toContain(c);
  });
  it('material-eval.wgsl composes with the scene group and textures', () => {
    const code = composeWgsl('material/material-eval.wgsl', {
      sources: shaderSources,
      defines: { ...base, SCENE_GROUP: 1, CUSTOM_ALPHA: false, VERTEX_FORMAT: 1, TEX_GROUP: 1, TEX_BINDING_BASE: 8, TEX_ARRAYS: 2, TEX_SAMPLERS: 1 },
    }).code;
    expect(code).toContain('fn material_eval(hit: SurfaceHit, V: vec3f) -> MatEval');
    expect(code.match(/^struct MatEval \{/gm)).toHaveLength(1);
  });
});

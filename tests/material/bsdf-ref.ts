// f64 CPU reference of the V1 / V2 Tier-1 BSDF (math.md#bsdf-v1, #bsdf-v2; gap-bsdf §3–§6). Independent of the
// WGSL (written from the math and the Cycles sources), used by the CPU tests (tests/material/*.test.ts) and as the
// cross-check oracle of validation/gpu-tests/bsdf.gpu.test.ts.
import { lutFloats, LUT_LAYOUT } from '../../src/core/render/luts/lut-layout.ts';

export type V3 = [number, number, number];
const lut = lutFloats();

export const WEIGHT_CUTOFF = 1e-5;
export const ROUGH_SQ_THRESH = 2e-10;

const sat = (x: number) => Math.min(Math.max(x, 0), 1);
const avg = (v: V3) => (v[0] + v[1] + v[2]) / 3;
const max3 = (v: V3) => Math.max(v[0], v[1], v[2]);
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
const mix3 = (a: V3, b: number, t: number): V3 => [mix(a[0], b, t), mix(a[1], b, t), mix(a[2], b, t)];
export const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const norm = (a: V3): V3 => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };

// ---- lookup_table.h (Math.trunc == float_to_int for x ≥ 0)
export function lutRead(x: number, off: number, n: number): number {
  const xs = sat(x) * (n - 1);
  const i = Math.min(Math.trunc(xs), n - 1);
  const j = Math.min(i + 1, n - 1);
  const t = xs - i;
  const d0 = lut[off + i];
  if (t === 0) return d0;
  return (1 - t) * d0 + t * lut[off + j];
}
export function lutRead2D(x: number, y: number, off: number, nx: number, ny: number): number {
  const ys = sat(y) * (ny - 1);
  const i = Math.min(Math.trunc(ys), ny - 1);
  const j = Math.min(i + 1, ny - 1);
  const t = ys - i;
  const d0 = lutRead(x, off + nx * i, nx);
  if (t === 0) return d0;
  return (1 - t) * d0 + t * lutRead(x, off + nx * j, nx);
}
export function lutRead3D(x: number, y: number, z: number, off: number, n: number): number {
  const zs = sat(z) * (n - 1);
  const i = Math.min(Math.trunc(zs), n - 1);
  const j = Math.min(i + 1, n - 1);
  const t = zs - i;
  const d0 = lutRead2D(x, y, off + n * n * i, n, n);
  if (t === 0) return d0;
  return (1 - t) * d0 + t * lutRead2D(x, y, off + n * n * j, n, n);
}
export const S_ior = (r: number, mu: number, z: number) => lutRead3D(r, mu, z, LUT_LAYOUT.ggxGenSchlickIorS, 16);
export const S_s = (r: number, mu: number, z: number) => lutRead3D(r, mu, z, LUT_LAYOUT.ggxGenSchlickS, 16);
export const ggxE = (r: number, mu: number) => lutRead2D(r, mu, LUT_LAYOUT.ggxE, 32, 32);
export const ggxEavg = (r: number) => lutRead(r, LUT_LAYOUT.ggxEavg, 32);

// ---- Fresnel (bsdf_util.h)
export const F0FromIor = (eta: number) => ((eta - 1) / (eta + 1)) ** 2;
export const iorFromF0 = (f0: number) => { const s = Math.sqrt(Math.min(Math.max(f0, 0), 0.99)); return (1 + s) / (1 - s); };
export function fresnelDielectric(cosI: number, eta: number): number {
  const g = eta * eta - (1 - cosI * cosI);
  if (g <= 0) return 1;
  const ci = Math.abs(cosI);
  const ct = -Math.sqrt(g) / eta;
  const rs = (ci + eta * ct) / (ci - eta * ct);
  const rp = (ct + eta * ci) / (eta * ci - ct);
  return 0.5 * (rs * rs + rp * rp);
}
export function fresnelGenSchlickIor(c: number, eta: number, f0: V3): V3 {
  const F0r = F0FromIor(eta);
  const s = sat((fresnelDielectric(c, eta) - F0r) / (1 - F0r));
  return mix3(f0, 1, s);
}
export function fresnelF82(c: number, F0: V3, B: V3): V3 {
  const s = sat(1 - c);
  const s5 = s ** 5;
  return [0, 1, 2].map((k) => sat(mix(F0[k], 1, s5) - B[k] * c * s5 * s)) as V3;
}
export function f82TintB(F0: V3, tint: V3): V3 {
  if (tint[0] === 1 && tint[1] === 1 && tint[2] === 1) return [0, 0, 0];
  const f = 6 / 7;
  const f5 = f ** 5;
  return [0, 1, 2].map((k) => mix(F0[k], 1, f5) * (7 / (f5 * f)) * (1 - tint[k])) as V3;
}

// ---- GGX
export const ggxD = (a2: number, cosNH: number) => { const c2 = Math.min(cosNH * cosNH, 1); return a2 / (Math.PI * ((1 - c2) + a2 * c2) ** 2); };
export const ggxLambda = (a2: number, cosN: number) => { const c = Math.max(Math.abs(cosN), 1e-7); return 0.5 * (Math.sqrt(1 + a2 * Math.max(1 / (c * c) - 1, 0)) - 1); };

// ---- materials
export interface V1Params { model: 0; diffuse: V3; glossy: V3; roughness: number; mix: number }
export interface V2Params { model: 1; base: V3; metallic: number; roughness: number; ior: number; specLevel: number; specTint: V3 }
export type MatParams = V1Params | V2Params;

export interface Ctx {
  model: 0 | 1;
  alpha: number; a2: number; rough: number; singular: boolean;
  hasD: boolean; hasS: boolean; hasMetal: boolean; hasSpec: boolean;
  wD: V3; wG: V3; wM: number; f0M: V3; bM: V3; wS: number; f0S: V3; etaS: number;
  swD: number; swS: number; swM: number; swSpec: number; qD: number; qS: number;
  lambda: number;   // V2 layering factor Λ_S(μ)
}

/** Closures at μ = N·V (local frame, N = +z, V side already chosen). */
export function prepare(p: MatParams, mu: number): Ctx {
  const r = sat(p.roughness);
  const alpha = r * r;
  const a2 = alpha * alpha;
  const c: Ctx = {
    model: p.model, alpha, a2, rough: Math.sqrt(Math.sqrt(a2)), singular: !(a2 > ROUGH_SQ_THRESH),
    hasD: false, hasS: false, hasMetal: false, hasSpec: false,
    wD: [0, 0, 0], wG: [0, 0, 0], wM: 0, f0M: [0, 0, 0], bM: [0, 0, 0], wS: 0, f0S: [0, 0, 0], etaS: 1,
    swD: 0, swS: 0, swM: 0, swSpec: 0, qD: 0, qS: 0, lambda: 1,
  };
  if (p.model === 0) {
    const f = sat(p.mix);
    const wD = p.diffuse.map((x) => Math.max((1 - f) * x, 0)) as V3;
    const wG = p.glossy.map((x) => Math.max(f * x, 0)) as V3;
    if (Math.abs(avg(wD)) >= WEIGHT_CUTOFF) { c.hasD = true; c.wD = wD; c.swD = Math.abs(avg(wD)); }
    if (Math.abs(avg(wG)) >= WEIGHT_CUTOFF) { c.hasS = true; c.wG = wG; c.swS = Math.abs(avg(wG)); }
  } else {
    const C = p.base.map((x) => Math.max(x, 0)) as V3;
    const Cc = C.map((x) => Math.min(x, 1)) as V3;
    const m = sat(p.metallic);
    const ior = Math.max(p.ior, 1e-5);
    const level = Math.max(p.specLevel, 0);
    const tint = p.specTint.map((x) => Math.max(x, 0)) as V3;
    let weight = 1;
    if (m > WEIGHT_CUTOFF) {
      c.hasS = c.hasMetal = true;
      c.wM = m; c.f0M = Cc; c.bM = f82TintB(Cc, tint.map((x) => Math.min(x, 1)) as V3);
      const Fmu = fresnelF82(mu, c.f0M, c.bM);
      const est: V3 = Fmu.some((x) => x !== 0) ? mix3(Cc, 1, S_s(c.rough, mu, 0.5)) : [0, 0, 0];
      c.swM = m * avg(est);
      weight *= 1 - m;
    }
    let eta = ior;
    let f0 = F0FromIor(eta);
    if (level !== 0.5) { f0 *= 2 * level; eta = iorFromF0(f0); if (ior < 1) eta = 1 / eta; }
    if (eta !== 1 && Math.abs(weight) >= WEIGHT_CUTOFF) {
      c.hasS = c.hasSpec = true;
      c.wS = weight; c.etaS = eta; c.f0S = tint.map((x) => sat(f0 * x)) as V3;
      const z = Math.sqrt(Math.abs((eta - 1) / (eta + 1)));
      const Fmu = fresnelGenSchlickIor(mu, eta, c.f0S);
      const est: V3 = Fmu.some((x) => x !== 0) ? mix3(c.f0S, 1, S_ior(c.rough, mu, z)) : [0, 0, 0];
      c.swSpec = weight * avg(est);
      c.lambda = sat(1 - max3(est));
      weight *= c.lambda;
    }
    const wD = C.map((x) => Math.max(x * weight, 0)) as V3;
    if (Math.abs(avg(wD)) >= WEIGHT_CUTOFF) { c.hasD = true; c.wD = wD; c.swD = Math.abs(avg(wD)); }
    c.swS = c.swM + c.swSpec;
  }
  const swd = c.hasD ? Math.max(c.swD, 1e-12) : 0;
  const sws = c.hasS ? Math.max(c.swS, 1e-12) : 0;
  if (swd + sws > 0) { c.qD = swd / (swd + sws); c.qS = sws / (swd + sws); }
  return c;
}

export function FS(c: Ctx, cosHI: number): V3 {
  if (c.model === 0) return c.wG;
  const out: V3 = [0, 0, 0];
  if (c.hasMetal) { const F = fresnelF82(cosHI, c.f0M, c.bM); for (let k = 0; k < 3; k++) out[k] += c.wM * F[k]; }
  if (c.hasSpec) { const F = fresnelGenSchlickIor(cosHI, c.etaS, c.f0S); for (let k = 0; k < 3; k++) out[k] += c.wS * F[k]; }
  return out;
}

export interface RefEval { fD: V3; fS: V3; pD: number; pS: number }   // pD/pS joint (include q)

/** Eval in the local frame (N = +z). V, L unit. */
export function evalLocal(p: MatParams, V: V3, L: V3): RefEval {
  const c = prepare(p, V[2]);
  const e: RefEval = { fD: [0, 0, 0], fS: [0, 0, 0], pD: 0, pS: 0 };
  if (c.hasD) { const k = Math.max(L[2], 0) / Math.PI; e.fD = c.wD.map((x) => x * k) as V3; e.pD = c.qD * k; }
  const cosNI = V[2];
  const cosNO = L[2];
  if (c.hasS && !c.singular && cosNI > 0 && cosNO >= 0) {
    const H = norm([V[0] + L[0], V[1] + L[1], V[2] + L[2]]);
    const D = ggxD(c.a2, H[2]);
    const lI = ggxLambda(c.a2, cosNI);
    const lO = ggxLambda(c.a2, cosNO);
    const common = D / cosNI * 0.25;
    const F = FS(c, dot(H, V));
    e.fS = F.map((x) => x * common / (1 + lI + lO)) as V3;
    e.pS = c.qS * common / (1 + lI);
  }
  return e;
}

/** Local-frame direction from polar angle theta (deg) at azimuth phi (rad). */
export const dirFromTheta = (thetaDeg: number, phi = 0): V3 => {
  const t = (thetaDeg * Math.PI) / 180;
  return [Math.sin(t) * Math.cos(phi), Math.sin(t) * Math.sin(phi), Math.cos(t)];
};

// f64 CPU dual of the reconnection predicate, k* and the offset checks O1–O3 (restir-api.md §6.1 T3-D, R6;
// math.md#rc-predicate, #offset-checks; gap-rc §3.3–§3.5, §6.1, §10.3). Written from the math, independent of the
// WGSL: every quantity is recomputed in float64 from ids (positions from the f32 vertex data and bary bits), the
// stored direction bits, the stored per-pixel thr bits and the material parameters (f64 BSDF reference
// tests/material/bsdf-ref.ts). Pure TS, no Node APIs: it runs in the cpu lane (rc-dual.test.ts, hand-built
// configurations) and inside the Chrome GPU test on dumped round-trip records (restir-shift.gpu.test.ts).
//
// Classification (gap-rc §10.3): a disagreement between a GPU decision and the dual is LOGIC when |margin_f64| ≥ 2⁻¹⁶
// (roughness: |α − α_min| ≥ 1e-6), else FP-BOUNDARY.
import { LUT_LAYOUT } from '../../src/core/render/luts/lut-layout.ts';
import {
  F0FromIor, ROUGH_SQ_THRESH, S_ior, S_s, WEIGHT_CUTOFF, dot, f82TintB, fresnelF82, fresnelGenSchlickIor, ggxD, ggxLambda, iorFromF0,
  type V3,
} from '../material/bsdf-ref.ts';

void LUT_LAYOUT;

export const LOBE = { D: 0, S: 1, GR: 2, GT: 3, NEE: 4, NONE: 5 } as const;
export const RCK = { SURFACE: 0, LIGHT: 1, ENV: 2 } as const;
export const RCT = { NONE: 0, D: 1, R: 2, F: 3, I: 4, GUARD: 5 } as const;
/** Margin of a discrete failure (delta event, G_T lobe, failed guard): never an FP-boundary flip (rc.wgsl). */
export const RC_MARGIN_DISCRETE = -1024;
export const RC_MARGIN_MAX = 1024;
/** LOGIC threshold on |margin| (log2 units) and on |α − α_min| (gap-rc §10.3). */
export const MARGIN_LOGIC = 2 ** -16;
export const ALPHA_LOGIC = 1e-6;
export const FLT_MAX = 3.4028234663852886e38;

export interface RcVertexD { pos: V3; ng: V3; kind: number; diffuseOnly: boolean }
export interface RcEventD { lobe: number; delta: boolean; alpha: number; pMarg: number }
export interface RcResultD { ok: boolean; margin: number; term: number }
export interface RcParams { alphaMin: number; crit2022?: boolean; dmin?: number }

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const clampM = (x: number) => Math.min(Math.max(x, -RC_MARGIN_MAX), RC_MARGIN_MAX);
const log2Ratio = (num: number, den: number) => (!(den > 0) ? RC_MARGIN_MAX : !(num > 0) ? -RC_MARGIN_MAX : clampM(Math.log2(num / den)));
export const eventNone = (): RcEventD => ({ lobe: LOBE.NONE, delta: false, alpha: FLT_MAX, pMarg: 0 });

/** P_k in f64 (math.md#rc-predicate; gap-rc §3.3), same sub-test order and margin convention as rc.wgsl. */
export function pairTestF64(a: RcVertexD, ea: RcEventD, b: RcVertexD, eb: RcEventD, thr: number, p: RcParams): RcResultD {
  if (p.crit2022) return pair2022F64(a, ea, b, eb, p);
  const r: RcResultD = { ok: false, margin: RC_MARGIN_DISCRETE, term: RCT.D };
  if (ea.delta || ea.lobe === LOBE.GT) return r;
  r.margin = clampM(ea.alpha - p.alphaMin);
  if (!(ea.alpha >= p.alphaMin)) { r.term = RCT.R; return r; }
  if (b.kind === RCK.ENV) return { ok: true, margin: r.margin, term: RCT.NONE };
  const dl = sub(b.pos, a.pos);
  const t2 = dot(dl, dl);
  r.term = RCT.GUARD;
  if (!(t2 > 1e-12) || !Number.isFinite(t2)) { r.margin = RC_MARGIN_DISCRETE; return r; }
  const t = Math.sqrt(t2);
  const cb = Math.abs(dot(b.ng, dl)) / t;
  if (!(cb > 0) || !(ea.pMarg > 0) || !Number.isFinite(ea.pMarg)) { r.margin = RC_MARGIN_DISCRETE; return r; }
  const rayFP = t2 / (ea.pMarg * cb);
  r.margin = Math.min(r.margin, log2Ratio(rayFP, thr));
  if (!(rayFP >= thr)) { r.term = RCT.F; return r; }
  if (b.kind === RCK.LIGHT || b.diffuseOnly) return { ok: true, margin: r.margin, term: RCT.NONE };
  if (eb.delta || eb.lobe === LOBE.GT) return { ok: false, margin: RC_MARGIN_DISCRETE, term: RCT.D };
  const ca = Math.abs(dot(a.ng, dl)) / t;
  if (!(ca > 0) || !(eb.pMarg > 0) || !Number.isFinite(eb.pMarg)) return { ok: false, margin: RC_MARGIN_DISCRETE, term: RCT.GUARD };
  const invFP = t2 / (eb.pMarg * ca);
  r.margin = Math.min(r.margin, log2Ratio(invFP, thr));
  if (!(invFP >= thr)) { r.term = RCT.I; return r; }
  return { ok: true, margin: r.margin, term: RCT.NONE };
}

function pair2022F64(a: RcVertexD, ea: RcEventD, b: RcVertexD, eb: RcEventD, p: RcParams): RcResultD {
  const r: RcResultD = { ok: false, margin: RC_MARGIN_DISCRETE, term: RCT.D };
  if (ea.delta || eb.delta || ea.lobe === LOBE.GT || eb.lobe === LOBE.GT) return r;
  const rmin = Math.min(ea.alpha, eb.alpha);
  r.margin = clampM(rmin - p.alphaMin);
  if (!(rmin >= p.alphaMin)) { r.term = RCT.R; return r; }
  if (b.kind === RCK.ENV) return { ok: true, margin: r.margin, term: RCT.NONE };
  const dl = sub(b.pos, a.pos);
  const t2 = dot(dl, dl);
  const d2 = (p.dmin ?? 0) ** 2;
  r.margin = Math.min(r.margin, 0.5 * log2Ratio(t2, d2));
  if (!(t2 >= d2)) { r.term = RCT.F; return r; }
  return { ok: true, margin: r.margin, term: RCT.NONE };
}

/** thr = τ·R²_pri (math.md#rc-predicate), f64. */
export function primaryThresholdF64(camPos: V3, x1: V3, ng1: V3, tau: number): number {
  const d = sub(camPos, x1);
  const d2 = dot(d, d);
  return tau * d2 * 4 * Math.PI / Math.max(Math.abs(dot(ng1, d)) / Math.sqrt(d2), 1e-6);
}

/** Is a GPU-vs-dual disagreement a LOGIC violation (else FP-BOUNDARY)? `term` = the dual's deciding sub-test. */
export function isLogic(r: RcResultD): boolean {
  if (r.term === RCT.R) return Math.abs(r.margin) >= ALPHA_LOGIC;
  return Math.abs(r.margin) >= MARGIN_LOGIC;
}

// ------------------------------------------------------------------------------------------------ f64 BSDF (p̄, joint)

/** Material parameters as the GPU MatEval sees them (untextured materials; V1 / V2 / glass models). */
export interface DualMat {
  model: number;            // bsdf.wgsl BSDF_MODEL_*: 0 V1, 1 V2, 2 V2 glass, 3 Glass node, 4 Refraction node
  base: V3; metallic: number; roughness: number; ior: number; specLevel: number; specTint: V3; transmission: number;
}

const sat = (x: number) => Math.min(Math.max(x, 0), 1);
const avg = (v: V3) => (v[0] + v[1] + v[2]) / 3;
const max3 = (v: V3) => Math.max(v[0], v[1], v[2]);
const mix3 = (a: V3, b: number, t: number): V3 => [a[0] + (b - a[0]) * t, a[1] + (b - a[1]) * t, a[2] + (b - a[2]) * t];
const nrm = (a: V3): V3 => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };

export interface DualLobes { hasD: boolean; hasS: boolean; singular: boolean; diffuseOnly: boolean; qD: number; qS: number; a2: number; supported: boolean }

/** Lobe structure at μ = Ns·V (V1 / V2 without transmission; glass models → supported = false). */
export function dualLobes(m: DualMat, mu: number): DualLobes {
  const r = sat(m.roughness);
  const alpha = r * r;
  const a2 = alpha * alpha;
  const out: DualLobes = { hasD: false, hasS: false, singular: !(a2 > ROUGH_SQ_THRESH), diffuseOnly: false, qD: 0, qS: 0, a2, supported: m.model <= 1 };
  if (!out.supported) return out;
  let swD = 0, swS = 0;
  if (m.model === 0) {
    const f = sat(m.metallic);
    const wD = m.base.map((x) => Math.max((1 - f) * x, 0)) as V3;
    const wG = m.specTint.map((x) => Math.max(f * x, 0)) as V3;
    if (Math.abs(avg(wD)) >= WEIGHT_CUTOFF) { out.hasD = true; swD = Math.abs(avg(wD)); }
    if (Math.abs(avg(wG)) >= WEIGHT_CUTOFF) { out.hasS = true; swS = Math.abs(avg(wG)); }
  } else {
    const rough = Math.sqrt(Math.sqrt(a2));
    const C = m.base.map((x) => Math.max(x, 0)) as V3;
    const Cc = C.map((x) => Math.min(x, 1)) as V3;
    const met = sat(m.metallic);
    const ior = Math.max(m.ior, 1e-5);
    const level = Math.max(m.specLevel, 0);
    const tint = m.specTint.map((x) => Math.max(x, 0)) as V3;
    let weight = 1;
    let swM = 0, swSpec = 0;
    if (met > WEIGHT_CUTOFF) {
      out.hasS = true;
      const B = f82TintB(Cc, tint.map((x) => Math.min(x, 1)) as V3);
      const Fmu = fresnelF82(mu, Cc, B);
      const est: V3 = Fmu.some((x) => x !== 0) ? mix3(Cc, 1, S_s(rough, mu, 0.5)) : [0, 0, 0];
      swM = met * avg(est);
      weight *= 1 - met;
    }
    let eta = ior;
    let f0 = F0FromIor(eta);
    if (level !== 0.5) { f0 *= 2 * level; eta = iorFromF0(f0); if (ior < 1) eta = 1 / eta; }
    if (eta !== 1 && Math.abs(weight) >= WEIGHT_CUTOFF) {
      out.hasS = true;
      const f0S = tint.map((x) => sat(f0 * x)) as V3;
      const z = Math.sqrt(Math.abs((eta - 1) / (eta + 1)));
      const Fmu = fresnelGenSchlickIor(mu, eta, f0S);
      const est: V3 = Fmu.some((x) => x !== 0) ? mix3(f0S, 1, S_ior(rough, mu, z)) : [0, 0, 0];
      swSpec = weight * avg(est);
      weight *= sat(1 - max3(est));
    }
    const wD = C.map((x) => Math.max(x * weight, 0)) as V3;
    if (Math.abs(avg(wD)) >= WEIGHT_CUTOFF) { out.hasD = true; swD = Math.abs(avg(wD)); }
    swS = swM + swSpec;
  }
  const d = out.hasD ? Math.max(swD, 1e-12) : 0;
  const s = out.hasS ? Math.max(swS, 1e-12) : 0;
  if (d + s > 0) { out.qD = d / (d + s); out.qS = s / (d + s); }
  out.diffuseOnly = out.hasD && !out.hasS;
  return out;
}

/** Perceptual roughness of a lobe (bsdf.wgsl lobe_roughness): D 1, S/G_R r (0 if singular), NEE max over lobes. */
export function lobeRoughnessF64(m: DualMat, lobe: number, lob: DualLobes): number {
  const r = sat(m.roughness);
  const rS = lob.singular ? 0 : r;
  switch (lobe) {
    case LOBE.D: return 1;
    case LOBE.S: case LOBE.GR: return rS;
    case LOBE.NEE: return Math.max(lob.hasD ? 1 : 0, lob.hasS ? rS : 0);
    default: return FLT_MAX;
  }
}

export interface DualPdf { pD: number; pS: number; marg: number; supported: boolean; lob: DualLobes }

/**
 * Valid-only per-lobe JOINT pdfs (q(ℓ|V)·p_ℓ(L|V)) and the marginal p̄ at a surface with geometric normal ng and
 * shading normal ns, after the two-sided flip for the evaluating V (bsdf.wgsl conventions): D carries the Lambert
 * sampler's Ng·L > 0 rejection, S the VNDF sampler's Ng·L ≥ 0 (and Ns·V > 0, Ns·L ≥ 0).
 */
export function pdfF64(m: DualMat, ngIn: V3, nsIn: V3, V: V3, L: V3): DualPdf {
  const back = dot(ngIn, V) < 0;
  const ng: V3 = back ? [-ngIn[0], -ngIn[1], -ngIn[2]] : ngIn;
  const ns: V3 = back ? [-nsIn[0], -nsIn[1], -nsIn[2]] : nsIn;
  const mu = dot(ns, V);
  const lob = dualLobes(m, mu);
  const out: DualPdf = { pD: 0, pS: 0, marg: 0, supported: lob.supported, lob };
  if (!lob.supported) return out;
  const cosNO = dot(ns, L);
  if (lob.hasD && dot(ng, L) > 0) out.pD = lob.qD * Math.max(cosNO, 0) / Math.PI;
  if (lob.hasS && !lob.singular && mu > 0 && cosNO >= 0 && dot(ng, L) >= 0) {
    const H = nrm([V[0] + L[0], V[1] + L[1], V[2] + L[2]]);
    const D = ggxD(lob.a2, dot(ns, H));
    const lI = ggxLambda(lob.a2, mu);
    out.pS = lob.qS * (D / mu * 0.25) / (1 + lI);
  }
  out.marg = out.pD + out.pS;
  return out;
}

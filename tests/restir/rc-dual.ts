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

// ------------------------------------------------------------------------------------------------ T3-D: recorded shifts

/** Geometry as the GPU sees it (recentred f32 positions), per-triangle material, materials in dual form. */
export interface DualScene { positions: Float32Array; indices: Uint32Array; triMaterial: Uint32Array; materials: DualMat[]; tau: number; params: RcParams }

export interface DualVertex { pos: V3; ng: V3; mat: DualMat }

const f32v = new Float32Array(1);
const u32v = new Uint32Array(f32v.buffer);
const bf = (u: number) => { u32v[0] = u; return f32v[0]; };
const neg = (a: V3): V3 => [-a[0], -a[1], -a[2]];

/** vertex_from_ids in f64 from the f32 vertex data (flat shading: ns = ng). */
export function dualVertex(s: DualScene, prim: number, u: number, v: number): DualVertex {
  const i0 = s.indices[3 * prim], i1 = s.indices[3 * prim + 1], i2 = s.indices[3 * prim + 2];
  const P = (i: number): V3 => [s.positions[3 * i], s.positions[3 * i + 1], s.positions[3 * i + 2]];
  const a = P(i0), b = P(i1), c = P(i2);
  const w = 1 - u - v;
  const pos: V3 = [w * a[0] + u * b[0] + v * c[0], w * a[1] + u * b[1] + v * c[1], w * a[2] + u * b[2] + v * c[2]];
  const e1 = sub(b, a), e2 = sub(c, a);
  const ng = nrm([e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]);
  return { pos, ng, mat: s.materials[s.triMaterial[prim]] };
}

/** The event leaving vertex x (incoming from `from`, outgoing along L) with lobe code (lobe | delta << 3 | NEE). */
function dualEvent(x: DualVertex, from: V3, L: V3, lobeCode: number): { e: RcEventD; v: RcVertexD; supported: boolean; cosL: number } {
  const V = nrm(sub(from, x.pos));
  const pdf = pdfF64(x.mat, x.ng, x.ng, V, L);
  const lobe = lobeCode & 7, delta = (lobeCode & 8) !== 0;
  const alpha = delta ? 0 : lobeRoughnessF64(x.mat, lobe, pdf.lob);
  // |cos| of the event direction and of V with the geometric normal: the sampler supports (Ng·L > 0 / ≥ 0) and the
  // two-sided flip switch there, so a pair decided within f32 noise of such a tangency is FP-BOUNDARY, not LOGIC.
  const cosL = Math.min(Math.abs(dot(x.ng, L)), Math.abs(dot(x.ng, V)));
  return { e: { lobe, delta, alpha, pMarg: pdf.marg }, v: { pos: x.pos, ng: x.ng, kind: RCK.SURFACE, diffuseOnly: pdf.lob.diffuseOnly }, supported: pdf.supported, cosL };
}

/** Tangency of a pair (a, b): min |cos| of the segment with both geometric normals (surface vertices only). */
function pairTangency(a: RcVertexD, b: RcVertexD): number {
  const dl = sub(b.pos, a.pos);
  const t = Math.hypot(dl[0], dl[1], dl[2]);
  let c = Math.abs(dot(a.ng, dl)) / t;
  if (b.kind !== RCK.ENV) c = Math.min(c, Math.abs(dot(b.ng, dl)) / t);
  return c;
}
/** A disagreement within this tangency (|cos| of the pair segment or of an event direction) is FP-BOUNDARY. */
export const TANGENCY_FP = 1e-5;

export interface DualRecordResult {
  checked: number; logic: number; fp: number; tangentFp: number;
  /** U-11: J of a defined (SC_OK) shift re-derived from JOINT pdfs in f64 (Eq. 2): checked / disagreeing (> 1e-3 rel). */
  jChecked: number; jBad: number; skipped: string | undefined;
  details: string[];
}

const SC = { OK: 0, O0_MISS: 3, O0_LOBE: 4, O0_TECH: 5, O1: 7, O2: 8, O3: 9, OCCLUDED: 10, ZERO: 11, J_INVALID: 12, O0_SUPPORT: 13, NONFINITE: 15 };

/**
 * Re-derive, in f64, every rcPairTest decision the GPU recorded for one shift (restir-shift-fixtures.ts T3 dual
 * records): the replayed pairs (O1 / O3), the last pre-rc pair with EV_RECONNECT(_NEE) (O1), the rc pair (O2) and the
 * ∅ terminal pair (O3), from ids, the stored rcWi / NEE light data, the destination thr and the materials; plus the
 * destination thr itself (recomputed from the camera and y₁) and the completeness of the pair list of a defined shift.
 */
export function dualCheckRecord(s: DualScene, w: Uint32Array, o: number): DualRecordResult {
  const r: DualRecordResult = { checked: 0, logic: 0, fp: 0, tangentFp: 0, jChecked: 0, jBad: 0, skipped: undefined, details: [] };
  const code = w[o], flags = w[o + 1], thr = bf(w[o + 2]);
  const sc = code & 0xff;
  const d = flags & 15, k = (flags >>> 4) & 15, tech = (flags >>> 8) & 3;
  const lkm1 = (flags >>> 14) & 7, lk = (flags >>> 18) & 7, dk = (flags & 0x200000) !== 0;
  const nP = w[o + 15] & 0xff;
  const cam: V3 = [bf(w[o + 74]), bf(w[o + 75]), bf(w[o + 76])];
  const y1 = dualVertex(s, w[o + 3], bf(w[o + 4]), bf(w[o + 5]));
  const empty = k === 0;
  const nB = empty ? d - 1 : (k > 2 ? k - 2 : 0);
  if (nB > 7 || d > 9) { r.skipped = 'deep'; return r; }
  // thr of the destination domain (math.md#rc-predicate), recomputed
  const thrD = primaryThresholdF64(cam, y1.pos, y1.ng, s.tau);
  if (!(Math.abs(thrD / thr - 1) < 1e-4)) { r.logic++; r.details.push(`thr ${thr} vs f64 ${thrD}`); }
  // offset vertices Y[1..]: y₁, then the replayed hits (Y[b+1] = hit of sample b); lobes at Y[b]
  const Y: (DualVertex | undefined)[] = [undefined, y1];
  const lobes: number[] = [0];
  let escape: V3 | undefined;
  for (let b = 1; b <= nB; b++) {
    const q = o + 26 + 4 * (b - 1);
    const prim = w[q];
    lobes[b] = w[q + 3];
    if (prim === 0xFFFFFFFF || prim === 0xFFFFFFFE) { Y[b + 1] = undefined; if (prim === 0xFFFFFFFF) escape = [bf(w[o + 23]), bf(w[o + 24]), bf(w[o + 25])]; break; }
    Y[b + 1] = dualVertex(s, prim, bf(w[q + 1]), bf(w[q + 2]));
  }
  if (Y.some((v) => v && !v.mat)) { r.skipped = 'material'; return r; }
  const prevPos = (b: number): V3 => (b === 1 ? cam : Y[b - 1]!.pos);
  const dirTo = (b: number): V3 | undefined => (Y[b + 1] ? nrm(sub(Y[b + 1]!.pos, Y[b]!.pos)) : escape);
  const replayEvent = (b: number) => { const L = dirTo(b); return L ? dualEvent(Y[b]!, prevPos(b), L, lobes[b]) : undefined; };
  // reconnection data
  const yk1Idx = empty ? 0 : (k > 2 ? k - 1 : 1);
  const yk1 = empty ? undefined : Y[yk1Idx];
  const forced = tech === 0 && k === d;
  const envRc = tech === 3 && k === d, emitRc = tech === 1 && k === d;
  const light = { dir: [bf(w[o + 16]), bf(w[o + 17]), bf(w[o + 18])] as V3, pos: [bf(w[o + 19]), bf(w[o + 20]), bf(w[o + 21])] as V3, inf: w[o + 22] !== 0 };
  const neeDir = (x: V3): V3 => (light.inf ? light.dir : nrm(sub(light.pos, x)));
  const rcWi: V3 = [bf(w[o + 9]), bf(w[o + 10]), bf(w[o + 11])];
  const xk = (!empty && !forced && !envRc) ? dualVertex(s, w[o + 6], bf(w[o + 7]), bf(w[o + 8])) : undefined;
  const reconnectEvent = () => {
    if (!yk1) return undefined;
    const from = prevPos(yk1Idx);
    if (forced) return dualEvent(yk1, from, neeDir(yk1.pos), LOBE.NEE);
    const L = envRc ? rcWi : nrm(sub(xk!.pos, yk1.pos));
    return dualEvent(yk1, from, L, lkm1);
  };
  const seen = new Set<number>();
  for (let i = 0; i < Math.min(nP, 10); i++) {
    const pw = w[o + 54 + 2 * i];
    const j = pw & 15, gOk = ((pw >>> 4) & 1) !== 0, gTerm = pw >>> 5;
    seen.add(j);
    let res: RcResultD | undefined;
    let sup = true;
    let dbg = '';
    let tang = 1;
    if (empty && j === d) {                              // ∅ terminal pair (y_{d−1}, emitter | env)
      const ea = replayEvent(d - 1);
      if (!ea) { r.skipped = 'trace'; return r; }
      sup = ea.supported;
      const bV: RcVertexD = Y[d] ? { pos: Y[d]!.pos, ng: Y[d]!.ng, kind: RCK.LIGHT, diffuseOnly: false } : { pos: [0, 0, 0], ng: [0, 0, 0], kind: RCK.ENV, diffuseOnly: false };
      res = pairTestF64(ea.v, ea.e, bV, eventNone(), thr, s.params);
      tang = Math.min(ea.cosL, bV.kind === RCK.ENV ? 1 : pairTangency(ea.v, bV));
    } else if (empty || j <= nB) {                      // replayed pair (y_{j−1}, y_j | EV_BSDF)
      const ea = replayEvent(j - 1), eb = replayEvent(j);
      if (!ea || !eb) { r.skipped = 'trace'; return r; }
      sup = ea.supported && eb.supported;
      res = pairTestF64(ea.v, ea.e, eb.v, eb.e, thr, s.params);
      tang = Math.min(ea.cosL, eb.cosL, pairTangency(ea.v, eb.v));
    } else if (j === k - 1) {                           // last pre-rc pair, EV_RECONNECT(_NEE) at y_{k−1}
      const ea = replayEvent(k - 2), eb = reconnectEvent();
      if (!ea || !eb) { r.skipped = 'trace'; return r; }
      sup = ea.supported && eb.supported;
      res = pairTestF64(ea.v, ea.e, eb.v, eb.e, thr, s.params);
      tang = Math.min(ea.cosL, eb.cosL, pairTangency(ea.v, eb.v));
    } else if (j === k) {                               // rc pair (O2)
      const ea = reconnectEvent();
      if (!ea) { r.skipped = 'trace'; return r; }
      sup = ea.supported;
      tang = ea.cosL;
      if (envRc) res = pairTestF64(ea.v, ea.e, { pos: [0, 0, 0], ng: [0, 0, 0], kind: RCK.ENV, diffuseOnly: false }, eventNone(), thr, s.params);
      else if (emitRc) {
        const zV: RcVertexD = { pos: xk!.pos, ng: xk!.ng, kind: RCK.LIGHT, diffuseOnly: false };
        res = pairTestF64(ea.v, ea.e, zV, eventNone(), thr, s.params);
        tang = Math.min(tang, pairTangency(ea.v, zV));
      } else {
        const Lk = tech === 0 && k === d - 1 ? neeDir(xk!.pos) : rcWi;
        const code2 = tech === 0 && k === d - 1 ? LOBE.NEE : (lk | (dk ? 8 : 0));
        const eb = dualEvent(xk!, yk1!.pos, Lk, code2);
        sup = sup && eb.supported;
        dbg = `ea=${JSON.stringify(ea.e)} eb=${JSON.stringify(eb.e)} xk=${JSON.stringify(xk)} y=${JSON.stringify(yk1)} Lk=${Lk}`;
        res = pairTestF64(ea.v, ea.e, eb.v, eb.e, thr, s.params);
      }
    }
    if (!res) { r.logic++; r.details.push(`pair ${j} not attributable (d=${d} k=${k})`); continue; }
    if (!sup) { r.skipped = 'material'; return r; }
    r.checked++;
    if (res.ok !== gOk) {
      if (isLogic(res) && !(tang < TANGENCY_FP)) { r.logic++; r.details.push(`pair ${j}: gpu ok=${gOk} term=${gTerm} m=${bf(w[o + 55 + 2 * i]).toExponential(3)}, f64 ${JSON.stringify(res)} (d=${d} k=${k} tech=${tech} lkm1=${lkm1} lk=${lk} sc=${sc}) ${dbg}`); } else { r.fp++; if (tang < TANGENCY_FP) r.tangentFp++; }
    }
  }
  // U-11: J = jNum/jDen with JOINT pdfs of the copied lobes (math.md#jacobian), recomputed in f64
  if (sc === SC.OK && yk1) {
    const gpuJ = bf(w[o + 77]), jDen = bf(w[o + 78]);
    const joint = (x: DualVertex, from: V3, L: V3, lobe: number): number | undefined => {
      const pd = pdfF64(x.mat, x.ng, x.ng, nrm(sub(from, x.pos)), L);
      if (!pd.supported) return undefined;
      return lobe === LOBE.D ? pd.pD : lobe === LOBE.S ? pd.pS : undefined;
    };
    const Gf = (a: V3, b: V3, nb: V3) => { const dl = sub(a, b); const t2 = dot(dl, dl); return Math.abs(dot(nb, dl)) / (t2 * Math.sqrt(t2)); };
    const from = prevPos(yk1Idx);
    let jNum: number | undefined = 1;
    if (forced) jNum = undefined;
    else if (envRc) jNum = joint(yk1, from, rcWi, lkm1);
    else {
      const wP = nrm(sub(xk!.pos, yk1.pos));
      const pY = joint(yk1, from, wP, lkm1);
      const G = Gf(yk1.pos, xk!.pos, xk!.ng);
      if (pY === undefined) jNum = undefined;
      else if (emitRc || (tech === 0 && k === d - 1)) jNum = pY * G;
      else { const pK = joint(xk!, yk1.pos, rcWi, lk); jNum = pK === undefined ? undefined : pY * G * pK; }
    }
    if (jNum !== undefined) {
      r.jChecked++;
      const Jf = jNum / jDen;
      if (!(Math.abs(Jf / gpuJ - 1) < 1e-3)) { r.jBad++; if (r.details.length < 4) r.details.push(`J gpu ${gpuJ} f64 ${Jf} (d=${d} k=${k} tech=${tech} lkm1=${lkm1} lk=${lk})`); }
    }
  }
  // completeness: a shift defined w.r.t. O1–O3 must have evaluated every pair of its list
  const defined = sc === SC.OK || sc === SC.ZERO || sc === SC.OCCLUDED || sc === SC.J_INVALID || sc === SC.O0_LOBE || sc === SC.O0_SUPPORT || sc === SC.NONFINITE;
  if (defined && nP <= 10) {
    const need: number[] = [];
    const hi = empty ? d : (forced ? k - 1 : k);
    for (let j = 2; j <= hi; j++) need.push(j);
    const lobeFailEarly = sc === SC.O0_LOBE && forced;   // forced NEE with no non-delta lobe stops before O1
    if (!lobeFailEarly && need.some((j) => !seen.has(j))) {
      r.logic++; r.details.push(`incomplete pair list: need ${need} seen ${[...seen]} (d=${d} k=${k} tech=${tech} sc=${sc})`);
    }
  }
  return r;
}

/** MaterialData (scene types) → the dual's parameters (bsdf.wgsl MatEval mapping of material_eval, untextured). */
export function dualMaterial(m: {
  model: string; baseColorFactor: number[]; metallicFactor: number; roughnessFactor: number; ior: number; specularFactor: number;
  specularColorFactor: number[]; transmissionFactor: number; v1?: { diffuse: number[]; glossy: number[]; roughness: number; mix: number };
}): DualMat {
  if (m.model === 'v1' && m.v1) {
    return { model: 0, base: m.v1.diffuse as V3, metallic: m.v1.mix, roughness: m.v1.roughness, ior: 1.5, specLevel: 0.5, specTint: m.v1.glossy as V3, transmission: 0 };
  }
  if (m.model === 'glass' || m.model === 'refraction') {
    return { model: m.model === 'glass' ? 3 : 4, base: m.baseColorFactor.slice(0, 3) as V3, metallic: 0, roughness: m.roughnessFactor, ior: m.ior, specLevel: 0.5, specTint: [1, 1, 1], transmission: 0 };
  }
  return {
    model: m.transmissionFactor > 1e-5 ? 2 : 1, base: m.baseColorFactor.slice(0, 3) as V3, metallic: m.metallicFactor, roughness: m.roughnessFactor,
    ior: m.ior, specLevel: 0.5 * m.specularFactor, specTint: m.specularColorFactor.slice(0, 3) as V3, transmission: m.transmissionFactor,
  };
}
void neg;

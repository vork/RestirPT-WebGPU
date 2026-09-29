// f64 CPU reference of the Cycles GGX glass closure, lobe class G (math.md#glass; gap-glass §2, §5; Cycles
// closure/bsdf_microfacet.h:226-357, :423-499, :586-820; svm/closure.h:169-509, :745-832). Written from the Cycles
// sources and the spec, independent of the WGSL (material/glass.wgsl); used by tests/material/glass-ref.test.ts and
// as the oracle of validation/gpu-tests/glass.gpu.test.ts. Local frame: N = Ng = +z, V.z > 0 (V already on the
// flipped side); `back` says whether that side is the backfacing side of the winding normal (η_side = 1/ior).
import {
  F0FromIor, S_ior, S_s, WEIGHT_CUTOFF, ROUGH_SQ_THRESH, dot, f82TintB, fresnelF82, fresnelGenSchlickIor, ggxD, ggxLambda,
  iorFromF0, norm, type V3,
} from './bsdf-ref.ts';

export const ETA_SINGULAR_EPS = 1e-4;
const sat = (x: number) => Math.min(Math.max(x, 0), 1);
const avg = (v: V3) => (v[0] + v[1] + v[2]) / 3;
const max3 = (v: V3) => Math.max(v[0], v[1], v[2]);
const mix3 = (a: V3, b: number, t: number): V3 => [a[0] + (b - a[0]) * t, a[1] + (b - a[1]) * t, a[2] + (b - a[2]) * t];
const scale3 = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const mul3 = (a: V3, b: V3): V3 => [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
const add3 = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const isZero3 = (a: V3) => a[0] === 0 && a[1] === 0 && a[2] === 0;

export interface GlassClosure {
  w: V3; eta: number; F0r: number; f0: V3; rt: V3; tt: V3; fnone: boolean; hasR: boolean;
  alpha: number; a2: number; rough: number; delta: boolean; deltaT: boolean;
  /** Cycles sample weight (|avg(w)|·avg(albedo estimate), or |avg(w)| for the Refraction node). */
  sw: number;
}

/** Real dielectric Fresnel with the signed refracted cosine (bsdf_util.h fresnel_dielectric_polarized). */
export function fresnelDielectricT(cosI: number, eta: number): { F: number; cosT: number } {
  const g = eta * eta - (1 - cosI * cosI);
  if (g <= 0) return { F: 1, cosT: 0 };
  const ci = Math.abs(cosI);
  const ct = -Math.sqrt(g) / eta;
  const rs = (ci + eta * ct) / (ci - eta * ct);
  const rp = (ct + eta * ci) / (eta * ci - ct);
  return { F: 0.5 * (rs * rs + rp * rp), cosT: ct };
}

export function glassFresnel(g: GlassClosure, cosHI: number): { R: V3; T: V3; cosT: number } {
  const { F: Freal, cosT } = fresnelDielectricT(cosHI, g.eta);
  if (g.fnone) return { R: [0, 0, 0], T: Freal === 1 ? [0, 0, 0] : [1, 1, 1], cosT };
  const s = sat((Freal - g.F0r) / (1 - g.F0r));
  const F = mix3(g.f0, 1, s);
  return { R: mul3(F, g.rt), T: mul3(F.map((x) => 1 - x) as V3, g.tt), cosT };
}

function base(roughness: number, ior: number, back: boolean): Pick<GlassClosure, 'alpha' | 'a2' | 'rough' | 'delta' | 'deltaT' | 'eta' | 'F0r'> {
  const r = sat(roughness);
  const alpha = r * r;
  const a2 = alpha * alpha;
  const i = Math.max(ior, 1e-5);
  const eta = back ? 1 / i : i;
  const delta = !(a2 > ROUGH_SQ_THRESH);
  return { alpha, a2, rough: Math.sqrt(Math.sqrt(a2)), delta, deltaT: delta || Math.abs(eta - 1) < ETA_SINGULAR_EPS, eta, F0r: F0FromIor(eta) };
}

/** Generalized-Schlick glass closure (Principled transmission / Glass node) with the albedo-scaled sample weight at μ. */
export function glassSchlick(w: V3, roughness: number, ior: number, f0: V3, rt: V3, tt: V3, mu: number, back: boolean): GlassClosure {
  const g: GlassClosure = { ...base(roughness, ior, back), w, f0: f0.map(sat) as V3, rt, tt, fnone: false, hasR: true, sw: 0 };
  const fr = glassFresnel(g, mu);
  let R = fr.R;
  if (!isZero3(R)) {
    const z = Math.sqrt(Math.abs((g.eta - 1) / (g.eta + 1)));
    R = mul3(mix3(g.f0, 1, S_ior(g.rough, mu, z)), g.rt);
  }
  g.sw = Math.abs(avg(w)) * avg(add3(R, fr.T));
  return g;
}

export function glassNode(color: V3, roughness: number, ior: number, mu: number, back: boolean): GlassClosure {
  const c = color.map((x) => Math.max(x, 0)) as V3;
  const f0 = F0FromIor(Math.max(ior, 1e-5));
  return glassSchlick([1, 1, 1], roughness, ior, [f0, f0, f0], c, c, mu, back);
}

export function refractionNode(color: V3, roughness: number, ior: number, back: boolean): GlassClosure {
  return { ...base(roughness, ior, back), w: color, f0: [0, 0, 0], rt: [0, 0, 0], tt: [1, 1, 1], fnone: true, hasR: false, sw: Math.abs(avg(color)) };
}

export interface GlassEvalRef { f: V3; pdfC: number; pdfV: number; isT: boolean; valid: boolean }

/** Cycles bsdf_microfacet_eval (GGX glass/refraction) + the sampler-support flag. `Ng` defaults to N. */
export function glassEval(g: GlassClosure, V: V3, L: V3, Ng: V3 = [0, 0, 1]): GlassEvalRef {
  const zero: GlassEvalRef = { f: [0, 0, 0], pdfC: 0, pdfV: 0, isT: false, valid: false };
  const cNI = V[2], cNO = L[2];
  if (!(cNI > 0) || g.delta) return zero;
  const isT = cNO < 0;
  if (!isT && !g.hasR) return { ...zero, isT };
  const Hu: V3 = isT ? [-(g.eta * L[0] + V[0]), -(g.eta * L[1] + V[1]), -(g.eta * L[2] + V[2])] : [V[0] + L[0], V[1] + L[1], V[2] + L[2]];
  const len = Math.hypot(...Hu);
  if (!(len > 0)) return { ...zero, isT };
  const invLen = 1 / len;
  const H = scale3(Hu, invLen);
  const cHI = dot(H, V);
  const fr = glassFresnel(g, cHI);
  const aR = avg(fr.R), aT = avg(fr.T);
  if (!(aR + aT > 0)) return { ...zero, isT };
  const D = ggxD(g.a2, H[2]);
  const lI = ggxLambda(g.a2, cNI), lO = ggxLambda(g.a2, cNO);
  const common = D / cNI * (isT ? (g.eta * invLen) ** 2 * Math.abs(cHI * dot(H, L)) : 0.25);
  const pR = aR / (aR + aT);
  const pdfC = common * (isT ? 1 - pR : pR) / (1 + lI);
  const f = scale3(mul3(g.w, isT ? fr.T : fr.R), common / (1 + lI + lO));
  let valid: boolean;
  if (isT) {
    const Hn = H[2] >= 0 ? H : scale3(H, -1);
    valid = !g.deltaT && dot(Ng, L) < 0 && dot(Hn, V) > 0 && dot(Hn, L) < 0 && aT > 0;
  } else valid = dot(Ng, L) >= 0 && aR > 0;
  return { f, pdfC, pdfV: valid ? pdfC : 0, isT, valid };
}

// ---- sampling (Cycles bsdf_microfacet_sample, f64) ----------------------------------------------------------------

/** Concentric square → disk (sample/mapping.h). */
function uniformDisk(u1: number, u2: number): [number, number] {
  const a = 2 * u1 - 1, b = 2 * u2 - 1;
  if (a === 0 && b === 0) return [0, 0];
  let r: number, phi: number;
  if (a * a > b * b) { r = a; phi = (Math.PI / 4) * (b / a); } else { r = b; phi = Math.PI / 2 - (Math.PI / 4) * (a / b); }
  return [r * Math.cos(phi), r * Math.sin(phi)];
}

/** Heitz 2018 VNDF sample, Cycles variant, local frame (z = N). */
export function vndfSample(V: V3, alpha: number, u1: number, u2: number): V3 {
  const Vs = norm([alpha * V[0], alpha * V[1], V[2]]);
  const lensq = Vs[0] * Vs[0] + Vs[1] * Vs[1];
  let T1: V3 = [1, 0, 0], T2: V3 = [0, 1, 0];
  if (lensq > 1e-7) {
    const il = 1 / Math.sqrt(lensq);
    T1 = [-Vs[1] * il, Vs[0] * il, 0];
    T2 = [Vs[1] * T1[2] - Vs[2] * T1[1], Vs[2] * T1[0] - Vs[0] * T1[2], Vs[0] * T1[1] - Vs[1] * T1[0]];
  }
  const t = uniformDisk(u1, u2);
  t[1] = Math.sqrt(Math.max(1 - t[0] * t[0], 0)) + (t[1] - Math.sqrt(Math.max(1 - t[0] * t[0], 0))) * 0.5 * (1 + Vs[2]);
  const tz = Math.sqrt(Math.max(1 - t[0] * t[0] - t[1] * t[1], 0));
  const Hs: V3 = [t[0] * T1[0] + t[1] * T2[0] + tz * Vs[0], t[0] * T1[1] + t[1] * T2[1] + tz * Vs[1], t[0] * T1[2] + t[1] * T2[2] + tz * Vs[2]];
  return norm([alpha * Hs[0], alpha * Hs[1], Math.max(0, Hs[2])]);
}

export interface GlassSampleRef { ok: boolean; L: V3; isT: boolean; isDelta: boolean; weight: V3; pdf: number }

/** One sample of the closure alone (q(G) = 1): weight = f/pdf (non-delta, eval form) or (R|T)/(P_R|1−P_R) (delta). */
export function glassSample(g: GlassClosure, V: V3, u: [number, number, number], Ng: V3 = [0, 0, 1]): GlassSampleRef {
  const none: GlassSampleRef = { ok: false, L: [0, 0, 1], isT: false, isDelta: false, weight: [0, 0, 0], pdf: 0 };
  if (!(V[2] > 0)) return none;
  const H: V3 = g.delta ? [0, 0, 1] : vndfSample(V, g.alpha, u[0], u[1]);
  const cHI = dot(H, V);
  const fr = glassFresnel(g, cHI);
  const aR = avg(fr.R), aT = avg(fr.T);
  if (!(aR + aT > 0)) return none;
  const pR = aR / (aR + aT);
  const refr = u[2] >= pR;
  const inv = 1 / g.eta;
  const L = norm(refr
    ? [(inv * cHI + fr.cosT) * H[0] - inv * V[0], (inv * cHI + fr.cosT) * H[1] - inv * V[1], (inv * cHI + fr.cosT) * H[2] - inv * V[2]]
    : [2 * cHI * H[0] - V[0], 2 * cHI * H[1] - V[1], 2 * cHI * H[2] - V[2]]);
  if ((dot(Ng, L) < 0) !== refr || (L[2] < 0) !== refr) return { ...none, L };
  const isDelta = g.delta || (refr && Math.abs(g.eta - 1) < ETA_SINGULAR_EPS);
  if (isDelta) return { ok: true, L, isT: refr, isDelta, weight: scale3(mul3(g.w, refr ? fr.T : fr.R), 1 / (refr ? 1 - pR : pR)), pdf: 0 };
  const e = glassEval(g, V, L, Ng);
  if (!(e.pdfV > 0)) return { ...none, L };
  return { ok: true, L, isT: refr, isDelta: false, weight: scale3(e.f, 1 / e.pdfV), pdf: e.pdfV };
}

/** Cycles' sample-form weight (R|T)/(P_R|1−P_R)·(1+Λ_I)/(1+Λ_I+Λ_O) at the SAMPLED microfacet (U-G5 cross-check). */
export function glassSampleFormWeight(g: GlassClosure, V: V3, u: [number, number, number]): { ok: boolean; weight: V3; commonSample: number; L: V3; isT: boolean } {
  const H = vndfSample(V, g.alpha, u[0], u[1]);
  const cHI = dot(H, V);
  const fr = glassFresnel(g, cHI);
  const aR = avg(fr.R), aT = avg(fr.T);
  const pR = aR / (aR + aT);
  const refr = u[2] >= pR;
  const inv = 1 / g.eta;
  const L = norm(refr
    ? [(inv * cHI + fr.cosT) * H[0] - inv * V[0], (inv * cHI + fr.cosT) * H[1] - inv * V[1], (inv * cHI + fr.cosT) * H[2] - inv * V[2]]
    : [2 * cHI * H[0] - V[0], 2 * cHI * H[1] - V[1], 2 * cHI * H[2] - V[2]]);
  const ok = (L[2] < 0) === refr && aR + aT > 0;
  const lI = ggxLambda(g.a2, V[2]), lO = ggxLambda(g.a2, L[2]);
  const commonSample = ggxD(g.a2, H[2]) / V[2] * (refr ? Math.abs(cHI * fr.cosT) / (fr.cosT + cHI * inv) ** 2 : 0.25);
  const weight = scale3(mul3(g.w, refr ? fr.T : fr.R), (1 + lI) / ((refr ? 1 - pR : pR) * (1 + lI + lO)));
  return { ok, weight, commonSample, L, isT: refr };
}

// ---- Principled with transmission: closure weights and Cycles sample weights (U-G6; glass §2.1, §2.6) ---------------

export interface PrincipledT { base: V3; metallic: number; roughness: number; ior: number; specLevel: number; specTint: V3; transmission: number }

export interface PrincipledMix {
  swM: number; swS: number; swG: number; swD: number; wD: V3; wS: number; glass?: GlassClosure; q: { D: number; S: number; G: number };
}

/** Principled closure order at μ (metal, glass, IOR level, specular + layering, diffuse) and the sample weights. */
export function principledMix(p: PrincipledT, mu: number, back = false): PrincipledMix {
  const C = p.base.map((x) => Math.max(x, 0)) as V3;
  const Cc = C.map((x) => Math.min(x, 1)) as V3;
  const r = sat(p.roughness);
  const rough = r;
  const m = sat(p.metallic);
  const t = sat(p.transmission);
  const ior = Math.max(p.ior, 1e-5);
  const level = Math.max(p.specLevel, 0);
  const tint = p.specTint.map((x) => Math.max(x, 0)) as V3;
  let weight = 1;
  const out: PrincipledMix = { swM: 0, swS: 0, swG: 0, swD: 0, wD: [0, 0, 0], wS: 0, q: { D: 0, S: 0, G: 0 } };
  if (m > WEIGHT_CUTOFF) {
    const B = f82TintB(Cc, tint.map((x) => Math.min(x, 1)) as V3);
    const Fmu = fresnelF82(mu, Cc, B);
    const est: V3 = Fmu.some((x) => x !== 0) ? mix3(Cc, 1, S_s(rough, mu, 0.5)) : [0, 0, 0];
    out.swM = m * avg(est);
    weight *= 1 - m;
  }
  if (t > WEIGHT_CUTOFF) {
    const f0 = F0FromIor(ior);
    out.glass = glassSchlick([t * weight, t * weight, t * weight], p.roughness, ior, scale3(tint, f0), [1, 1, 1], Cc.map(Math.sqrt) as V3, mu, back);
    out.swG = out.glass.sw;
    weight *= 1 - t;
  }
  let eta = ior;
  let f0 = F0FromIor(eta);
  if (level !== 0.5) { f0 *= 2 * level; eta = iorFromF0(f0); if (ior < 1) eta = 1 / eta; }
  if (eta !== 1 && Math.abs(weight) >= WEIGHT_CUTOFF) {
    const f0S = tint.map((x) => sat(f0 * x)) as V3;
    const z = Math.sqrt(Math.abs((eta - 1) / (eta + 1)));
    const Fmu = fresnelGenSchlickIor(mu, eta, f0S);
    const est: V3 = Fmu.some((x) => x !== 0) ? mix3(f0S, 1, S_ior(rough, mu, z)) : [0, 0, 0];
    out.wS = weight;
    out.swS = weight * avg(est);
    weight *= sat(1 - max3(est));
  }
  const wD = C.map((x) => Math.max(x * weight, 0)) as V3;
  if (Math.abs(avg(wD)) >= WEIGHT_CUTOFF) { out.wD = wD; out.swD = Math.abs(avg(wD)); }
  const s = out.swM + out.swS + out.swG + out.swD;
  out.q = { D: out.swD / s, S: (out.swM + out.swS) / s, G: out.swG / s };
  return out;
}

// ---- quadrature helpers ------------------------------------------------------------------------------------------

/** ∫_{S²} fn(L) dω on a (cosθ, φ) midpoint grid (n_u × n_phi); returns the sum. */
export function sphereQuadrature(nu: number, nphi: number, fn: (L: V3) => number): number {
  let s = 0;
  const du = 2 / nu, dphi = 2 * Math.PI / nphi;
  for (let i = 0; i < nu; i++) {
    const z = -1 + (i + 0.5) * du;
    const st = Math.sqrt(Math.max(0, 1 - z * z));
    for (let j = 0; j < nphi; j++) {
      const phi = (j + 0.5) * dphi;
      s += fn([st * Math.cos(phi), st * Math.sin(phi), z]);
    }
  }
  return s * du * dphi;
}

/** Direction from polar angle (deg) and azimuth (deg); `below` mirrors z. */
export function dirDeg(thetaDeg: number, phiDeg: number, below = false): V3 {
  const t = thetaDeg * Math.PI / 180, p = phiDeg * Math.PI / 180;
  return [Math.sin(t) * Math.cos(p), Math.sin(t) * Math.sin(p), below ? -Math.cos(t) : Math.cos(t)];
}

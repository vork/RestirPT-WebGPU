// Glass lobe class G on the GPU (plan §7.4 M3b = gap-glass §7.3 U-G1…U-G10), both GPU lanes, on the production
// material/*.wgsl modules (bsdf.wgsl + glass.wgsl) against the f64 reference tests/material/glass-ref.ts.
//  U-G1  glass_fresnel == f64 (random cos, η); Refraction node T = 0 under TIR
//  U-G2  eval·cos / Cycles pdf / valid flag == the glass §3 vectors
//  U-G3  χ² of bsdf_sample over the FULL sphere against the valid-only marginal pdf (grid η × r × θ_V), rejected +
//        delta mass == 1 − ∫p_valid; negative controls (η ×1.01, r ×1.04, and the Cycles eval pdf = "B-spur") fail
//  U-G4  E[weight] == the glass §3 albedo table (η 1.5, 1/1.5, 1.33) and table_ggx_glass_E at grid points (η > 1)
//  U-G5  weight == eval_lobe/pdf_joint on every sample; sampler acceptance == support ∧ pdf > 0; delta weights
//  U-G6  Cycles sample weights (glass §2.6 vectors)
//  U-G7  delta per sub-event (r 0.0037 / 0.0038; |η − 1| < 1e-4 ⇒ G_T delta); no eval at delta vertices
//  U-G8  η_side = 1/ior iff the evaluating V is on the backfacing side of the winding normal
//  U-G10 ggx_lambda(a2, −c) == ggx_lambda(a2, c) (exactly on the CPU lane; to 2e-6 rel. under relaxed Metal math)
//  plus eval == f64 on random (V, L) and the non-reciprocity of Cycles' BTDF: f_T(V,L)/|N·L| = η_side(V)²·f_T(L,V)/|N·V|.
// Histogram domain per hemisphere h (h = 0: L.z ≥ 0 — G_R and D/S; h = 1: L.z < 0 — G_T): u = 1 − |cosθ| ∈ [0, 1]
// and ψ = φ − π ∈ [−π, π] (dω = du dψ), bins equal-probability under a Cauchy mixture centred on the lobe (centre and
// width from an f64 pilot of the reference sampler). Expected counts: f64 on the CPU (the reference pdfs), adaptive
// Gauss–Legendre per bin in an asinh substitution — adaptive because the valid-only density jumps to 0 across the
// support boundaries (Hn·V = 0, TIR, Ng·L = 0) inside bins, and f64 because the f32 pdf of a sharp lobe is
// ill-conditioned in L (H is a difference of O(1) vectors near the peak).
import { afterAll, describe, expect, it } from 'vitest';
import { getTestGpu, lane, releaseTestGpu } from './device-factory.ts';
import { composeWgsl, createCheckedShaderModule } from '../../src/core/gpu/wgsl-composer.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { createLutBuffer, lutDefines } from '../../src/core/render/luts/lut-layout.ts';
import { TABLE_GGX_GLASS_E } from '../../src/core/render/luts/cycles-glass-luts.ts';
import { ggxD, ggxLambda, norm, type V3 } from '../../tests/material/bsdf-ref.ts';
import { adaptiveEdges, chi2Test, sidak } from '../../tests/material/chi2.ts';
import {
  dirDeg, glassEval, glassFresnel, glassNode, glassSample, principledMix, refractionNode, type GlassClosure,
} from '../../tests/material/glass-ref.ts';

const NU = 32;
const NPHI = 64;
const NB = NU * NPHI;
const HE = NU + 1 + NPHI + 1;   // edges per hemisphere
const EDGES = 2 * HE;
const THREADS = 4096;
const SPT = 256;
const NSAMPLES = THREADS * SPT;   // 1,048,576 per case
const PROBE_N = 12;
const CASE_BYTES = 80;

const HARNESS = `
#include "material/bsdf.wgsl"
#include "common/rng.wgsl"

struct Case {
  model: u32, roughness: f32, ior: f32, metallic: f32,
  base: vec3f, transmission: f32,
  tint: vec3f, specLevel: f32,
  V: vec3f, flags: u32,        // bit0: MATEVAL_BACKFACING, bit2: albedo only (no histogram)
  seed: u32, pad0: u32, pad1: u32, pad2: u32,
}
struct Params { caseBase: u32, caseCount: u32, threadsPerCase: u32, samplesPerThread: u32 }
struct Query { caseIdx: u32, pad0: u32, pad1: u32, pad2: u32, V: vec4f, L: vec4f, u: vec4f }

const NU: u32 = ${NU}u;
const NPHI: u32 = ${NPHI}u;
const NB: u32 = ${NB}u;
const HE: u32 = ${HE}u;
const EDGES: u32 = ${EDGES}u;
const PROBE_N: u32 = ${PROBE_N}u;

@group(0) @binding(1) var<storage, read> cases: array<Case>;
@group(0) @binding(2) var<storage, read> edges: array<f32>;
@group(0) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> partials: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> prm: Params;
@group(0) @binding(8) var<storage, read> queries: array<Query>;
@group(0) @binding(9) var<storage, read_write> probeOut: array<vec4f>;

fn case_mat(c: Case, V: vec3f) -> MatEval {
  var m: MatEval;
  m.model = c.model;
  m.base_color = c.base;
  m.metallic = c.metallic;
  m.roughness = c.roughness;
  m.ior = c.ior;
  m.specular_level = c.specLevel;
  m.specular_tint = c.tint;
  m.transmission = c.transmission;
  m.ns = vec3f(0.0, 0.0, 1.0);
  m.ng = vec3f(0.0, 0.0, 1.0);
  m.flags = select(0u, MATEVAL_BACKFACING, (c.flags & 1u) != 0u);
  m.flags = bsdf_flags(m, V);
  return m;
}

fn bin_search(off: u32, n: u32, x: f32) -> u32 {
  var lo = 0u;
  var hi = n;
  while (hi - lo > 1u) {
    let mid = (lo + hi) / 2u;
    if (edges[off + mid] <= x) { lo = mid; } else { hi = mid; }
  }
  return lo;
}
fn rel_err(a: f32, b: f32) -> f32 {
  let e = abs(a - b) / max(abs(b), 1e-30);
  return select(3e38, e, e <= 3e38);
}

@compute @workgroup_size(64)
fn k_sample(@builtin(global_invocation_id) gid: vec3u) {
  let local = gid.x / prm.threadsPerCase;
  if (local >= prm.caseCount) { return; }
  let ci = prm.caseBase + local;
  let t = gid.x % prm.threadsPerCase;
  let c = cases[ci];
  let V = c.V;
  let m = case_mat(c, V);
  let eo = ci * EDGES;
  var acc = vec3f(0.0);
  var acc2 = vec3f(0.0);
  var accR = 0.0;
  var accR2 = 0.0;
  var nValid = 0u;
  var nDelta = 0u;
  var nMismatch = 0u;
  var nNonFinite = 0u;
  var nT = 0u;
  var eW = 0.0;
  var eP = 0.0;
  var eM = 0.0;
  for (var i = 0u; i < prm.samplesPerThread; i++) {
    let h = pcg3d(vec3u(c.seed, t, i));
    let h2 = pcg3d(vec3u(c.seed ^ 0x5bd1e995u, t, i));
    let u = vec4f(u32_to_unit(h.x), u32_to_unit(h.y), u32_to_unit(h.z), u32_to_unit(h2.x));
    let s = bsdf_sample(m, V, u);
    if (s.lobe != LOBE_NONE && any(s.L != vec3f(0.0)) && !s.is_delta) {
      // 1_supp multiplies f and the joint pdf: sampler acceptance == support ∧ (joint pdf of the lobe > 0)
      let acc_ok = bsdf_sample_support(m, V, s.L, s.lobe) && bsdf_eval_lobe(m, V, s.L, s.lobe).w > 0.0;
      if (acc_ok != s.valid) { nMismatch++; }
    }
    if (!s.valid) { continue; }
    nValid++;
    if (!all_finite3(s.weight)) { nNonFinite++; continue; }
    acc += s.weight;
    acc2 += s.weight * s.weight;
    if (s.L.z >= 0.0) { accR += s.weight.x; accR2 += s.weight.x * s.weight.x; } else { nT++; }
    if (s.is_delta) { nDelta++; continue; }
    let ev = bsdf_eval_lobe(m, V, s.L, s.lobe);
    let ref_w = ev.xyz / ev.w;
    let den = max(bsdf_max3(abs(ref_w)), 1e-30);
    eW = max(eW, bsdf_max3(abs(s.weight - ref_w)) / den);
    eP = max(eP, rel_err(s.pdf_joint, ev.w));
    eM = max(eM, rel_err(s.pdf_marginal, bsdf_pdf_marginal(m, V, s.L)));
    if ((c.flags & 4u) != 0u) { continue; }
    let L = s.L;
    let hemi = select(1u, 0u, L.z >= 0.0);
    let uu = (L.x * L.x + L.y * L.y) / (1.0 + abs(L.z));   // 1 − |cosθ|, cancellation-free
    let psi = atan2(-L.y, -L.x);
    let off = eo + hemi * HE;
    let b = bin_search(off, NU, uu) * NPHI + bin_search(off + NU + 1u, NPHI, psi);
    atomicAdd(&hist[(ci * 2u + hemi) * NB + b], 1u);
  }
  partials[(ci * prm.threadsPerCase + t) * 2u] = vec4f(acc, accR);
  partials[(ci * prm.threadsPerCase + t) * 2u + 1u] = vec4f(acc2, accR2);
  atomicAdd(&stats[ci * 8u + 0u], nValid);
  atomicAdd(&stats[ci * 8u + 1u], nDelta);
  atomicAdd(&stats[ci * 8u + 2u], nMismatch);
  atomicAdd(&stats[ci * 8u + 3u], nNonFinite);
  atomicMax(&stats[ci * 8u + 4u], bitcast<u32>(eW));
  atomicMax(&stats[ci * 8u + 5u], bitcast<u32>(eP));
  atomicMax(&stats[ci * 8u + 6u], bitcast<u32>(eM));
  atomicAdd(&stats[ci * 8u + 7u], nT);
}

@compute @workgroup_size(64)
fn k_probe(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&queries)) { return; }
  let q = queries[i];
  let c = cases[q.caseIdx];
  let V = q.V.xyz;
  let L = q.L.xyz;
  let m = case_mat(c, V);
  let o = i * PROBE_N;
  probeOut[o + 0u] = bsdf_glass_diag(m, V, L);
  let ev = bsdf_eval(m, V, L);
  probeOut[o + 1u] = vec4f(ev.f_cos, ev.pdf_marginal);
  probeOut[o + 2u] = bsdf_eval_lobe(m, V, L, LOBE_GR);
  probeOut[o + 3u] = bsdf_eval_lobe(m, V, L, LOBE_GT);
  probeOut[o + 4u] = vec4f(select(0.0, 1.0, bsdf_sample_support(m, V, L, LOBE_GR)), select(0.0, 1.0, bsdf_sample_support(m, V, L, LOBE_GT)),
                           lobe_roughness(m, LOBE_GR), lobe_roughness(m, LOBE_GT));
  probeOut[o + 5u] = bsdf_lobe_probs(m, V);
  let sw = bsdf_sample_weights(m, V);
  probeOut[o + 6u] = vec4f(sw.xyz, f32(m.flags));
  let s = bsdf_sample(m, V, q.u);
  probeOut[o + 7u] = vec4f(s.L, f32(s.lobe));
  probeOut[o + 8u] = vec4f(s.weight, s.pdf_joint);
  probeOut[o + 9u] = vec4f(s.pdf_marginal, select(0.0, 1.0, s.is_delta), select(0.0, 1.0, s.valid), lobe_roughness(m, LOBE_NEE));
  let ctx = bsdf_prepare(m, V);
  let fr = glass_fresnel(ctx, q.u.x);
  let lp = ggx_lambda(q.u.y, q.u.z);
  probeOut[o + 10u] = vec4f(fr.R.x, fr.T.x, fr.cosT, (ggx_lambda(q.u.y, -q.u.z) - lp) / max(lp, 1e-30));
  let rev = bsdf_eval(m, L, V);                                   // reversed arguments (reciprocity check)
  probeOut[o + 11u] = vec4f(rev.f_cos, ctx.g_eta);
}
`;

// ------------------------------------------------------------------------------------------------ cases

/** Material of a case (mirrors the WGSL MatEval models 1–4). */
interface Mat {
  model: 1 | 2 | 3 | 4; base: V3; roughness: number; ior: number; metallic?: number; transmission?: number; specLevel?: number; tint?: V3;
}
interface Case {
  mat: Mat; V: V3; back: boolean; label: string; chi2: boolean; albedoOnly?: boolean;
  /** negative control: histogram of `controlOf` vs the expected counts of this case (same axes) */
  controlOf?: string; axesOf?: Case; useCyclesPdf?: boolean; qIor?: number; qRough?: number;
}

const glassMat = (r: number, ior: number, color: V3 = [1, 1, 1]): Mat => ({ model: 3, base: color, roughness: r, ior });
const dirTheta = (deg: number): V3 => { const t = (deg * Math.PI) / 180; return [Math.sin(t), 0, Math.cos(t)]; };

/** f64 reference closure of a case at μ (Glass node, Refraction node, or Principled with transmission 1). */
function refClosure(m: Mat, mu: number, back: boolean): GlassClosure | undefined {
  if (m.model === 3) return glassNode(m.base, m.roughness, m.ior, mu, back);
  if (m.model === 4) return refractionNode(m.base, m.roughness, m.ior, back);
  return principledMix({ base: m.base, metallic: m.metallic ?? 0, roughness: m.roughness, ior: m.ior, specLevel: m.specLevel ?? 0.5, specTint: m.tint ?? [1, 1, 1], transmission: m.transmission ?? 0 }, mu, back).glass;
}

/**
 * Lobe-centred axes per hemisphere (u centre, u width, ψ centre, ψ width) from an f64 pilot of the reference glass
 * sampler (≥ 800 non-delta samples per hemisphere, ≤ 400k draws): centre = median, width = 1.5·MAD (floored);
 * a hemisphere without a concentrated lobe gets uniform bins (width ≥ 1 ⇒ λ = 0). Memoised per case.
 */
const axesCache = new Map<string, { R: number[]; T: number[] }>();
function pilotAxes(c0: Case): { R: number[]; T: number[] } {
  const c = c0.axesOf ?? c0;
  const key = `${c.label}|${c.mat.model}|${c.mat.roughness}|${c.mat.ior}|${c.back}|${c.V}`;
  const hit = axesCache.get(key);
  if (hit) return hit;
  const g = refClosure(c.mat, c.V[2], c.back);
  const us: [number[], number[]] = [[], []], ps: [number[], number[]] = [[], []];
  if (g) {
    let a = 12345;
    const rnd = () => { a = (Math.imul(a, 1103515245) + 12345) >>> 0; return a / 4294967296; };
    for (let i = 0; i < 400_000 && (us[0].length < 800 || us[1].length < 800); i++) {
      const s = glassSample(g, c.V, [rnd(), rnd(), rnd()]);
      if (!s.ok || s.isDelta) continue;
      const h = s.L[2] >= 0 ? 0 : 1;
      if (us[h].length >= 4000) continue;
      us[h].push((s.L[0] ** 2 + s.L[1] ** 2) / (1 + Math.abs(s.L[2])));
      ps[h].push(Math.atan2(-s.L[1], -s.L[0]));
    }
  }
  const med = (x: number[]) => { const y = [...x].sort((p, q) => p - q); return y[y.length >> 1]; };
  const ax = (h: 0 | 1) => {
    if (us[h].length < 100) return [0.5, 1, 0, 4];
    const uc = med(us[h]), pc = med(ps[h]);
    const wu = Math.max(1.5 * med(us[h].map((x) => Math.abs(x - uc))), 1e-12);
    const wp = Math.max(1.5 * med(ps[h].map((x) => Math.abs(x - pc))), 1e-9);
    return [uc, Math.min(wu, 1), pc, Math.min(wp, 4)];
  };
  const out = { R: ax(0), T: ax(1) };
  axesCache.set(key, out);
  return out;
}

/**
 * Histogram edges. The Cauchy width never gets below an angle of 2e-4 rad (u: 2e-4·sinθ, ψ: 2e-4/sinθ): a sampled f32
 * direction carries up to ~2.5e-7 rad of round-off (refraction amplifies it by 1/η), and bins must stay ≳ 100× wider
 * than that or the counts migrate deterministically across bin edges (seen at r = 0.004: ±5σ alternating residuals).
 * So lobes narrower than 2e-4 rad (r ≤ 0.01) are resolved by a few bins only. The quadrature keeps the true (pilot)
 * lobe widths for its substitution.
 */
function caseEdges(c: Case): Float32Array {
  const a = pilotAxes(c);
  const out = new Float32Array(EDGES);
  [a.R, a.T].forEach(([uc, wu, pc, wp], h) => {
    const st = Math.sqrt(Math.max(uc * (2 - uc), 1e-12));
    const A = 2e-4;
    const wuE = Math.max(wu, A * st), wpE = Math.max(wp, A / Math.max(st, 1e-3));
    out.set(adaptiveEdges(NU, 0, 1, uc, wuE, wuE < 0.3 ? 0.75 : 0), h * HE);
    out.set(adaptiveEdges(NPHI, -Math.PI, Math.PI, pc, wpE, wpE < 1 ? 0.75 : 0), h * HE + NU + 1);
  });
  return out;
}

/** f64 reference density of a case's sampler (valid-only marginal over the non-delta lobes; controls may override η, r,
 *  or use the Cycles eval pdf). Local frame N = Ng = +z. */
function refPdf(c: Case): (L: V3) => number {
  const mat: Mat = { ...c.mat, ...(c.qIor ? { ior: c.qIor } : {}), ...(c.qRough ? { roughness: c.qRough } : {}) };
  const V = c.V, mu = V[2];
  const pick = (e: ReturnType<typeof glassEval>) => (c.useCyclesPdf ? e.pdfC : e.pdfV);
  if (mat.model === 3 || mat.model === 4) {
    const g = mat.model === 3 ? glassNode(mat.base, mat.roughness, mat.ior, mu, c.back) : refractionNode(mat.base, mat.roughness, mat.ior, c.back);
    return (L) => pick(glassEval(g, V, L));
  }
  const pm = principledMix({ base: mat.base, metallic: mat.metallic ?? 0, roughness: mat.roughness, ior: mat.ior, specLevel: mat.specLevel ?? 0.5,
    specTint: mat.tint ?? [1, 1, 1], transmission: mat.transmission ?? 0 }, mu, c.back);
  const r = Math.min(Math.max(mat.roughness, 0), 1), a2 = r ** 4;
  const pS = pm.q.S > 0 && a2 > 2e-10 ? pm.q.S / (4 * mu * (1 + ggxLambda(a2, mu))) : 0;
  return (L) => {
    let p = pm.glass ? pm.q.G * pick(glassEval(pm.glass, V, L)) : 0;
    if (L[2] > 0) p += pm.q.D * L[2] / Math.PI;                               // Lambert (sampler: Ng·L > 0)
    if (pS > 0 && L[2] >= 0) p += pS * ggxD(a2, norm([V[0] + L[0], V[1] + L[1], V[2] + L[2]])[2]);   // GGX VNDF (metal ⊕ specular)
    return p;
  };
}

const GL4X = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526];
const GL4W = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538];

/**
 * Expected bin probabilities of both hemispheres: ∫_bin p(L) du dψ in the asinh substitution of the pilot axes, 4×4
 * Gauss–Legendre per cell, a cell split in 4 while the split changes its value by more than 1e-4 (rel.) + 1e-10 (abs)
 * (≤ 6 levels) or while the support boundary crosses it — probes (GL nodes, corners, centre) disagree on p = 0 — with a
 * mass bound above 1e-8, down to 12 levels, so a thin valid sliver at a bin edge is not missed.
 */
function expectedProbs(c: Case, edges: Float32Array): Float64Array {
  const pdf = refPdf(c);
  const a = pilotAxes(c);
  const out = new Float64Array(2 * NB);
  for (let h = 0; h < 2; h++) {
    const [uc, wu, pc, wp] = h === 0 ? a.R : a.T;
    const zs = h === 0 ? 1 : -1;
    const f = (vu: number, vp: number) => {
      const uu = uc + wu * Math.sinh(vu), psi = pc + wp * Math.sinh(vp);
      const st = Math.sqrt(Math.max(uu * (2 - uu), 0));
      return pdf([-st * Math.cos(psi), -st * Math.sin(psi), zs * (1 - uu)]) * wu * Math.cosh(vu) * wp * Math.cosh(vp);
    };
    // 4×4 GL on a cell, plus the number of GL nodes and corners where the density is 0 (support indicator)
    let zeros = 0;
    const gl = (u0: number, u1: number, p0: number, p1: number) => {
      let s = 0;
      const hu = 0.5 * (u1 - u0), hp = 0.5 * (p1 - p0), mu0 = 0.5 * (u0 + u1), mp0 = 0.5 * (p0 + p1);
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
        const v = f(mu0 + hu * GL4X[i], mp0 + hp * GL4X[j]);
        if (v === 0) zeros++;
        s += GL4W[i] * GL4W[j] * v;
      }
      return s * hu * hp;
    };
    // Does the support boundary cross the cell with a mass that could matter? Probes: GL nodes, corners, centre; some
    // 0 and some not ⇒ cut; refine only while the cell's mass bound (area × max probe) exceeds 1e-8 (0.01 counts).
    const mixed = (u0: number, u1: number, p0: number, p1: number, zq: number) => {
      let z = zq, fmax = 0;
      for (const [a, b] of [[u0, p0], [u1, p0], [u0, p1], [u1, p1], [0.5 * (u0 + u1), 0.5 * (p0 + p1)]]) {
        const v = f(a, b);
        if (v === 0) z++;
        fmax = Math.max(fmax, v);
      }
      return z > 0 && z < 21 && fmax * (u1 - u0) * (p1 - p0) > 1e-8;
    };
    const cell = (u0: number, u1: number, p0: number, p1: number, q: number, zq: number, depth: number): number => {
      const mu = 0.5 * (u0 + u1), mp = 0.5 * (p0 + p1);
      const qs: number[] = [], zs: number[] = [];
      for (const [a0, a1, b0, b1] of [[u0, mu, p0, mp], [mu, u1, p0, mp], [u0, mu, mp, p1], [mu, u1, mp, p1]]) {
        zeros = 0; qs.push(gl(a0, a1, b0, b1)); zs.push(zeros);
      }
      const q4 = qs[0] + qs[1] + qs[2] + qs[3];
      const cut = mixed(u0, u1, p0, p1, zq);
      if ((!cut && (depth >= 6 || Math.abs(q4 - q) <= 1e-4 * Math.abs(q4) + 1e-10)) || depth >= 12) return q4;
      return cell(u0, mu, p0, mp, qs[0], zs[0], depth + 1) + cell(mu, u1, p0, mp, qs[1], zs[1], depth + 1)
        + cell(u0, mu, mp, p1, qs[2], zs[2], depth + 1) + cell(mu, u1, mp, p1, qs[3], zs[3], depth + 1);
    };
    for (let iu = 0; iu < NU; iu++) {
      const vu0 = Math.asinh((edges[h * HE + iu] - uc) / wu), vu1 = Math.asinh((edges[h * HE + iu + 1] - uc) / wu);
      for (let ip = 0; ip < NPHI; ip++) {
        const vp0 = Math.asinh((edges[h * HE + NU + 1 + ip] - pc) / wp), vp1 = Math.asinh((edges[h * HE + NU + 2 + ip] - pc) / wp);
        zeros = 0;
        const q0 = gl(vu0, vu1, vp0, vp1);
        out[h * NB + iu * NPHI + ip] = cell(vu0, vu1, vp0, vp1, q0, zeros, 0);
      }
    }
  }
  return out;
}

function packCases(cases: Case[]): ArrayBuffer {
  const buf = new ArrayBuffer(cases.length * CASE_BYTES);
  const dv = new DataView(buf);
  cases.forEach((c, i) => {
    const b = i * CASE_BYTES;
    const f = (o: number, v: number) => dv.setFloat32(b + o, v, true);
    const v3 = (o: number, v: V3) => { f(o, v[0]); f(o + 4, v[1]); f(o + 8, v[2]); };
    const m = c.mat;
    dv.setUint32(b, m.model, true);
    f(4, m.roughness); f(8, m.ior); f(12, m.metallic ?? 0);
    v3(16, m.base); f(28, m.transmission ?? 0);
    v3(32, m.tint ?? [1, 1, 1]); f(44, m.specLevel ?? 0.5);
    v3(48, c.V);
    dv.setUint32(b + 60, (c.back ? 1 : 0) | (c.albedoOnly ? 4 : 0), true);
    dv.setUint32(b + 64, Math.imul(0x9e3779b9, i + 1) >>> 0, true);
  });
  return buf;
}

const ETAS: [string, number, boolean][] = [['1.5', 1.5, false], ['1/1.5', 1.5, true], ['1.33', 1.33, false], ['2.4', 2.4, false], ['1.00005', 1.00005, false]];
const ROUGH = [0.004, 0.01, 0.05, 0.2, 0.5, 1];
const THETAS = [0, 30, 60, 80, 89];

/** §3 albedo table: [η label, ior, back, r, [μ, total, R part][]] */
const ALBEDO: [number, boolean, number, [number, number, number][]][] = [
  [1.5, false, 0.1, [[1, 1.0000, 0.0396], [0.7, 1.0000, 0.0510], [0.4, 0.9999, 0.1324], [0.1, 0.9959, 0.5610]]],
  [1.5, false, 0.25, [[1, 0.9995, 0.0398], [0.7, 0.9988, 0.0514], [0.4, 0.9948, 0.1276], [0.1, 0.9295, 0.3915]]],
  [1.5, false, 0.5, [[1, 0.9920, 0.0370], [0.7, 0.9793, 0.0467], [0.4, 0.9468, 0.0833], [0.1, 0.9163, 0.1717]]],
  [1.5, false, 1.0, [[1, 0.8932, 0.0127], [0.7, 0.8002, 0.0177], [0.4, 0.7190, 0.0279], [0.1, 0.7413, 0.0583]]],
  [1.5, true, 0.1, [[1, 0.9999, 0.0399], [0.7, 0.9998, 0.9957], [0.4, 0.9996, 0.9994], [0.1, 0.9939, 0.9938]]],
  [1.5, true, 0.25, [[1, 0.9960, 0.0412], [0.7, 0.9879, 0.8843], [0.4, 0.9807, 0.9745], [0.1, 0.8885, 0.8855]]],
  [1.5, true, 0.5, [[1, 0.9290, 0.0481], [0.7, 0.8626, 0.5403], [0.4, 0.8112, 0.7412], [0.1, 0.8105, 0.7911]]],
  [1.5, true, 1.0, [[1, 0.4168, 0.0258], [0.7, 0.4964, 0.1121], [0.4, 0.5053, 0.2220], [0.1, 0.5112, 0.4312]]],
  [1.33, false, 0.25, [[1, 0.9997, 0.0202], [0.7, 0.9989, 0.0287], [0.4, 0.9951, 0.0940], [0.1, 0.9318, 0.3589]]],
  [1.33, false, 0.5, [[1, 0.9944, 0.0186], [0.7, 0.9816, 0.0261], [0.4, 0.9480, 0.0583], [0.1, 0.9087, 0.1423]]],
  [1.33, false, 1.0, [[1, 0.9233, 0.0065], [0.7, 0.8181, 0.0094], [0.4, 0.7211, 0.0164], [0.1, 0.7197, 0.0395]]],
];
/** table_ggx_glass_E grid points (x = rough index, y = μ index, z index; η = (1 + z²)/(1 − z²)). */
const GLASS_E_POINTS: [number, number, number][] = [];
for (const x of [4, 8, 12, 15]) for (const y of [3, 8, 15]) for (const z of [3, 7, 11]) GLASS_E_POINTS.push([x, y, z]);

function buildCases(): Case[] {
  const out: Case[] = [];
  for (const [en, ior, back] of ETAS) for (const r of ROUGH) for (const th of THETAS) {
    out.push({ mat: glassMat(r, ior), V: dirTheta(th), back, label: `glass η=${en} r=${r} θ=${th}`, chi2: true });
  }
  // colour, Principled transmission (√C tint, f0·T_s), Refraction node (TIR kills), Principled mixture (marginal D+S+G)
  out.push({ mat: glassMat(0.3, 1.5, [0.9, 0.6, 0.3]), V: dirTheta(45), back: false, label: 'glass colour r=0.3 θ=45', chi2: true });
  out.push({ mat: glassMat(0.3, 1.5, [0.9, 0.6, 0.3]), V: dirTheta(45), back: true, label: 'glass colour back r=0.3 θ=45', chi2: true });
  out.push({ mat: { model: 2, base: [0.9, 0.5, 0.2], roughness: 0.3, ior: 1.45, transmission: 1, tint: [1, 0.8, 0.6] }, V: dirTheta(50), back: false, label: 'principled t=1 r=0.3 θ=50', chi2: true });
  out.push({ mat: { model: 2, base: [0.9, 0.5, 0.2], roughness: 0.3, ior: 1.45, transmission: 1, tint: [1, 0.8, 0.6] }, V: dirTheta(50), back: true, label: 'principled t=1 back r=0.3 θ=50', chi2: true });
  out.push({ mat: { model: 2, base: [0.9, 0.6, 0.3], roughness: 0.3, ior: 1.5, transmission: 0.5, metallic: 0.25 }, V: dirTheta(45), back: false, label: 'principled mix t=0.5 m=0.25 r=0.3 θ=45', chi2: true });
  out.push({ mat: { model: 2, base: [0.8, 0.8, 0.8], roughness: 0.6, ior: 1.5, transmission: 0.5, specLevel: 0.8 }, V: dirTheta(70), back: true, label: 'principled mix back t=0.5 r=0.6 θ=70', chi2: true });
  out.push({ mat: { model: 4, base: [0.8, 0.9, 1.0], roughness: 0.3, ior: 1.5 }, V: dirTheta(60), back: false, label: 'refraction r=0.3 θ=60', chi2: true });
  out.push({ mat: { model: 4, base: [0.8, 0.9, 1.0], roughness: 0.3, ior: 1.5 }, V: dirTheta(60), back: true, label: 'refraction back (TIR) r=0.3 θ=60', chi2: true });
  // negative controls (power): the SAME histogram against a slightly wrong density
  const ctl = (base: string, what: string, o: Partial<Case>) => {
    const b = out.find((c) => c.label === base)!;
    out.push({ ...b, ...o, label: `${base} with ${what}`, chi2: false, controlOf: base, axesOf: b });
  };
  ctl('glass η=1.5 r=0.2 θ=30', 'η ×1.01', { qIor: 1.5 * 1.01 });
  ctl('glass η=1/1.5 r=0.5 θ=60', 'η ×1.01', { qIor: 1.5 * 1.01 });
  ctl('glass η=1.5 r=0.5 θ=60', 'r ×1.04', { qRough: 0.52 });
  ctl('glass η=1.5 r=1 θ=80', 'the Cycles eval pdf (B-spur)', { useCyclesPdf: true });
  ctl('glass η=1/1.5 r=1 θ=60', 'the Cycles eval pdf (B-spur)', { useCyclesPdf: true });
  // albedo-only cases: the §3 table and table_ggx_glass_E grid points
  for (const [ior, back, r, cols] of ALBEDO) for (const [mu] of cols) {
    out.push({ mat: glassMat(r, ior), V: [Math.sqrt(1 - mu * mu), 0, mu], back, label: `albedo η=${back ? '1/' : ''}${ior} r=${r} μ=${mu}`, chi2: false, albedoOnly: true });
  }
  for (const [x, y, z] of GLASS_E_POINTS) {
    const zz = z / 15, mu = y / 15, eta = (1 + zz * zz) / (1 - zz * zz);
    out.push({ mat: glassMat(x / 15, eta), V: [Math.sqrt(1 - mu * mu), 0, mu], back: false, label: `glass_E x=${x} y=${y} z=${z}`, chi2: false, albedoOnly: true });
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ GPU plumbing

interface Harness { device: GPUDevice; lut: GPUBuffer; pSample: GPUComputePipeline; pProbe: GPUComputePipeline }
let harnessP: Promise<Harness> | undefined;
function harness(): Promise<Harness> {
  harnessP ??= (async () => {
    const { device, features, wgslLanguageFeatures } = await getTestGpu();
    const shader = composeWgsl('tests/glass_harness.wgsl', {
      sources: { ...shaderSources, 'tests/glass_harness.wgsl': HARNESS },
      defines: lutDefines({ declare: { group: 0, binding: 0 } }), features, wgslLanguageFeatures,
    });
    const module = await createCheckedShaderModule(device, shader, 'glass-harness');
    const mk = (entryPoint: string) => device.createComputePipelineAsync({ label: entryPoint, layout: 'auto', compute: { module, entryPoint } });
    const [pSample, pProbe] = await Promise.all([mk('k_sample'), mk('k_probe')]);
    return { device, lut: createLutBuffer(device), pSample, pProbe };
  })();
  return harnessP;
}

function storage(device: GPUDevice, data: ArrayBuffer | ArrayBufferView | number, label: string): GPUBuffer {
  const bytes = typeof data === 'number' ? data : data.byteLength;
  const buf = device.createBuffer({ label, size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, mappedAtCreation: typeof data !== 'number' });
  if (typeof data !== 'number') {
    const src = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    new Uint8Array(buf.getMappedRange()).set(src);
    buf.unmap();
  }
  return buf;
}
const bindAll = (device: GPUDevice, p: GPUComputePipeline, entries: [number, GPUBuffer][]) =>
  device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })) });

interface SamplingRun { cases: Case[]; hist: Uint32Array; partials: Float32Array; stats: Uint32Array; statsF: Float32Array }
let runP: Promise<SamplingRun> | undefined;
function samplingRun(): Promise<SamplingRun> {
  runP ??= (async () => {
    const h = await harness();
    const { device } = h;
    const cases = buildCases();
    const n = cases.length;
    const edges = new Float32Array(n * EDGES);
    cases.forEach((c, i) => { if (!c.albedoOnly) edges.set(caseEdges(c), i * EDGES); });
    const bCases = storage(device, packCases(cases), 'cases');
    const bEdges = storage(device, edges, 'edges');
    const bHist = storage(device, n * 2 * NB * 4, 'hist');
    const bPart = storage(device, n * THREADS * 32, 'partials');
    const bStats = storage(device, n * 8 * 4, 'stats');
    const BATCH = 4;
    const t0 = performance.now();
    device.pushErrorScope('validation');
    for (let base = 0; base < n; base += BATCH) {
      const count = Math.min(BATCH, n - base);
      const prm = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(prm, 0, new Uint32Array([base, count, THREADS, SPT]));
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(h.pSample);
      pass.setBindGroup(0, bindAll(device, h.pSample, [[0, h.lut], [1, bCases], [2, bEdges], [3, bHist], [4, bPart], [5, bStats], [7, prm]]));
      pass.dispatchWorkgroups(Math.ceil((count * THREADS) / 64));
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      prm.destroy();
    }
    const err = await device.popErrorScope();
    if (err) throw new Error(`glass sampling: ${err.message}`);
    console.log(`GLASS_SAMPLING ${lane()} ${n} cases x ${NSAMPLES} samples: ${((performance.now() - t0) / 1000).toFixed(2)} s`);
    const rd = async (b: GPUBuffer, bytes: number) => readBuffer(device, b, bytes);
    const statsBuf = await rd(bStats, n * 32);
    const run: SamplingRun = {
      cases,
      hist: new Uint32Array(await rd(bHist, n * 2 * NB * 4)),
      partials: new Float32Array(await rd(bPart, n * THREADS * 32)),
      stats: new Uint32Array(statsBuf), statsF: new Float32Array(statsBuf),
    };
    for (const b of [bCases, bEdges, bHist, bPart, bStats]) b.destroy();
    return run;
  })();
  return runP;
}

interface Query { caseIdx: number; V: V3; L: V3; u: [number, number, number, number] }
interface ProbeOut {
  gF: number; gPdfC: number; gPdfV: number; gIsT: boolean; gValid: boolean;
  f: V3; pdfM: number; lobeGR: number[]; lobeGT: number[]; supGR: boolean; supGT: boolean; rGR: number; rGT: number;
  qD: number; qS: number; qG: number; eta: number; swD: number; swS: number; swG: number; flags: number;
  sL: V3; sLobe: number; sW: V3; sPj: number; sPm: number; sDelta: boolean; sValid: boolean; rNEE: number;
  frR: number; frT: number; frCosT: number; lambdaDiff: number; revF: V3; ctxEta: number;
}
async function probe(cases: Case[], queries: Query[]): Promise<ProbeOut[]> {
  const h = await harness();
  const { device } = h;
  const qb = new ArrayBuffer(queries.length * 64);
  const dv = new DataView(qb);
  queries.forEach((q, i) => {
    const b = i * 64;
    dv.setUint32(b, q.caseIdx, true);
    q.V.forEach((x, k) => dv.setFloat32(b + 16 + 4 * k, x, true));
    q.L.forEach((x, k) => dv.setFloat32(b + 32 + 4 * k, x, true));
    q.u.forEach((x, k) => dv.setFloat32(b + 48 + 4 * k, x, true));
  });
  const bCases = storage(device, packCases(cases.map((c) => ({ ...c, albedoOnly: true }))), 'cases');
  const bQ = storage(device, qb, 'queries');
  const bO = storage(device, queries.length * PROBE_N * 16, 'probeOut');
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(h.pProbe);
  pass.setBindGroup(0, bindAll(device, h.pProbe, [[0, h.lut], [1, bCases], [8, bQ], [9, bO]]));
  pass.dispatchWorkgroups(Math.ceil(queries.length / 64));
  pass.end();
  device.queue.submit([enc.finish()]);
  const o = new Float32Array(await readBuffer(device, bO, queries.length * PROBE_N * 16));
  for (const b of [bCases, bQ, bO]) b.destroy();
  return queries.map((_, i) => {
    const a = (k: number) => Array.from(o.subarray((i * PROBE_N + k) * 4, (i * PROBE_N + k) * 4 + 4));
    const [o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11] = Array.from({ length: PROBE_N }, (_, k) => a(k));
    return {
      gF: o0[0], gPdfC: o0[1], gPdfV: o0[2], gIsT: (o0[3] & 1) === 1, gValid: (o0[3] & 2) === 2,
      f: [o1[0], o1[1], o1[2]], pdfM: o1[3], lobeGR: o2, lobeGT: o3, supGR: o4[0] > 0.5, supGT: o4[1] > 0.5, rGR: o4[2], rGT: o4[3],
      qD: o5[0], qS: o5[1], qG: o5[2], eta: o5[3], swD: o6[0], swS: o6[1], swG: o6[2], flags: o6[3],
      sL: [o7[0], o7[1], o7[2]], sLobe: o7[3], sW: [o8[0], o8[1], o8[2]], sPj: o8[3], sPm: o9[0], sDelta: o9[1] > 0.5, sValid: o9[2] > 0.5, rNEE: o9[3],
      frR: o10[0], frT: o10[1], frCosT: o10[2], lambdaDiff: o10[3], revF: [o11[0], o11[1], o11[2]], ctxEta: o11[3],
    };
  });
}

function xorshift(seed: number) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}
const f32 = (v: V3): V3 => [Math.fround(v[0]), Math.fround(v[1]), Math.fround(v[2])];
const randSphere = (rnd: () => number): V3 => { const z = 2 * rnd() - 1, p = 2 * Math.PI * rnd(), s = Math.sqrt(1 - z * z); return [s * Math.cos(p), s * Math.sin(p), z]; };
const LOBE_GR = 2, LOBE_GT = 3;
const MATEVAL_HAS_NON_DELTA = 1, MATEVAL_HAS_G = 32, MATEVAL_G_DELTA = 64, MATEVAL_BACKFACING = 128;

afterAll(async () => {
  const h = harnessP ? await harnessP : undefined;
  h?.lut.destroy();
  await releaseTestGpu();
});

describe(`glass lobe G (${lane()})`, () => {
  it('U-G1 glass_fresnel == f64 (R, T, cosT); Refraction node T = 0 under TIR; U-G10 Λ(a2, −c) == Λ(a2, c)', async () => {
    const rnd = xorshift(3);
    const cases: Case[] = [
      { mat: glassMat(0.5, 1.5), V: [0, 0, 1], back: false, label: 'front', chi2: false },
      { mat: glassMat(0.5, 1.5), V: [0, 0, 1], back: true, label: 'back', chi2: false },
      { mat: glassMat(0.5, 2.4), V: [0, 0, 1], back: false, label: '2.4', chi2: false },
      { mat: { model: 4, base: [1, 1, 1], roughness: 0.5, ior: 1.5 }, V: [0, 0, 1], back: true, label: 'refraction back', chi2: false },
    ];
    const queries: Query[] = [];
    for (let ci = 0; ci < cases.length; ci++) for (let k = 0; k < 500; k++) {
      queries.push({ caseIdx: ci, V: [0, 0, 1], L: [0, 0, -1], u: [Math.fround(2 * rnd() - 1), Math.fround(rnd() ** 2), Math.fround(rnd()), 0] });
    }
    const out = await probe(cases, queries);
    let worst = 0, tirKills = 0;
    out.forEach((o, i) => {
      const q = queries[i], c = cases[q.caseIdx];
      const g = refClosure(c.mat, 1, c.back)!;
      const fr = glassFresnel(g, q.u[0]);
      worst = Math.max(worst, Math.abs(o.frR - fr.R[0]), Math.abs(o.frT - fr.T[0]), Math.abs(o.frCosT - fr.cosT));
      if (c.mat.model === 4 && fr.T[0] === 0) { tirKills++; expect(o.frT).toBe(0); expect(o.frR).toBe(0); }
      // U-G10: the |cos| fix makes Λ even; the two inlined calls may differ by FMA contraction (relaxed Metal math)
      expect(Math.abs(o.lambdaDiff)).toBeLessThan(2e-6);
    });
    console.log(`U-G1 ${lane()} max |gpu − f64| = ${worst.toExponential(2)}; refraction-node TIR samples ${tirKills}`);
    expect(worst).toBeLessThan(2e-6);
    expect(tirKills).toBeGreaterThan(50);
  });

  it('U-G2 eval·cos, Cycles pdf and valid flag == the glass §3 vectors (Glass node, Color 1)', async () => {
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
    const cases: Case[] = V.map(([r, eta]) => ({ mat: glassMat(r, eta < 1 ? 1 / eta : eta), V: [0, 0, 1], back: eta < 1, label: '', chi2: false }));
    const out = await probe(cases, V.map((v, i) => ({ caseIdx: i, V: f32(dirDeg(v[2][0], v[2][1])), L: f32(dirDeg(v[3][0], v[3][1], v[3][2])), u: [0.5, 0.5, 0.5, 0.5] })));
    out.forEach((o, i) => {
      const [, , , , f, pdf, valid] = V[i];
      expect(Math.abs(o.gF - f) / f, `vector ${i} eval ${o.gF} vs ${f}`).toBeLessThan(2e-5);
      expect(Math.abs(o.gPdfC - pdf) / pdf, `vector ${i} pdf ${o.gPdfC} vs ${pdf}`).toBeLessThan(2e-5);
      expect(o.gValid, `vector ${i} valid`).toBe(valid);
      expect(o.gPdfV).toBe(valid ? o.gPdfC : 0);
      // NEE (bsdf_eval) keeps the spurious eval; the marginal pdf is the valid-only one
      expect(Math.abs(o.f[0] - f) / f).toBeLessThan(2e-5);
      expect(o.pdfM).toBe(valid ? o.gPdfC : 0);
    });
  });

  it('eval / pdfs / validity == f64 reference on random (V, L), all glass sources, both sides; support == valid', async () => {
    const rnd = xorshift(17);
    const cases: Case[] = [];
    for (const back of [false, true]) {
      cases.push({ mat: glassMat(0.35, 1.5, [0.9, 0.6, 0.3]), V: [0, 0, 1], back, label: 'glass', chi2: false });
      cases.push({ mat: glassMat(0.8, 1.33), V: [0, 0, 1], back, label: 'glass', chi2: false });
      cases.push({ mat: { model: 4, base: [0.8, 0.9, 1], roughness: 0.4, ior: 1.45 }, V: [0, 0, 1], back, label: 'refraction', chi2: false });
      cases.push({ mat: { model: 2, base: [0.9, 0.5, 0.2], roughness: 0.3, ior: 1.45, transmission: 1, tint: [1, 0.8, 0.6] }, V: [0, 0, 1], back, label: 'principled', chi2: false });
    }
    const queries: Query[] = [];
    cases.forEach((_, ci) => { for (let k = 0; k < 400; k++) { let V = randSphere(rnd); if (V[2] < 0.02) V = [V[0], V[1], 0.02 + Math.abs(V[2])]; queries.push({ caseIdx: ci, V: f32(norm(V)), L: f32(randSphere(rnd)), u: [0.5, 0.5, 0.5, 0.5] }); } });
    const out = await probe(cases, queries);
    let eF = 0, eP = 0, flips = 0, n = 0;
    out.forEach((o, i) => {
      const q = queries[i], c = cases[q.caseIdx];
      const g = refClosure(c.mat, q.V[2], c.back)!;
      const e = glassEval(g, q.V, q.L);
      const fAvg = (e.f[0] + e.f[1] + e.f[2]) / 3;
      if (fAvg === 0 && o.gF === 0) return;
      n++;
      eF = Math.max(eF, Math.abs(o.gF - fAvg) / Math.max(fAvg, 1e-3));
      eP = Math.max(eP, Math.abs(o.gPdfC - e.pdfC) / Math.max(e.pdfC, 1e-3));
      if (o.gValid !== e.valid) flips++;
      expect(o.gIsT).toBe(e.isT);
      expect(o.supGT).toBe(e.isT && o.gValid);
      if (c.mat.model !== 4) expect(o.supGR).toBe(!e.isT && q.L[2] >= 0);
      expect(Math.abs(o.eta - g.eta) / g.eta).toBeLessThan(1e-6);
    });
    console.log(`GLASS_EVAL_VS_REF ${lane()} ${n} pairs: max rel f ${eF.toExponential(2)} pdf_C ${eP.toExponential(2)}, validity flips ${flips}`);
    expect(n).toBeGreaterThan(2000);
    expect(eF).toBeLessThan(2e-4);
    expect(eP).toBeLessThan(2e-4);
    expect(flips).toBeLessThanOrEqual(2);   // f32 boundary cases of Hn·V / Hn·L only
  });

  it('U-G3 χ² of the glass sampler over the full sphere vs the valid-only marginal pdf (Šidák α = 0.01); controls fail', async () => {
    const run = await samplingRun();
    const chi = run.cases.map((c, ci) => ({ c, ci })).filter((x) => x.c.chi2);
    // χ² of the histogrammed (valid, non-delta) samples CONDITIONAL on their count (expected rescaled to Σobs); the
    // rejected + delta mass is tested separately: bins cut by the support boundary (Ng·L = 0, Hn·V = 0, TIR) carry a
    // quadrature error of up to ~2e-3 of the mass, which would otherwise dominate the outside cell.
    const tests: { label: string; res: ReturnType<typeof chi2Test>; outObs: number; outExp: number }[] = [];
    let nCpu = 0;
    const tq = performance.now();
    for (const { c, ci } of chi) {
      const obs = run.hist.subarray(ci * 2 * NB, (ci + 1) * 2 * NB);
      const exp = expectedProbs(c, caseEdges(c));
      nCpu++;
      const sObs = obs.reduce((a, b) => a + b, 0), sExp = exp.reduce((a, b) => a + b, 0);
      const scaled = exp.map((x) => (x / sExp) * sObs);
      tests.push({ label: c.label, res: chi2Test(obs, scaled, sObs), outObs: NSAMPLES - sObs, outExp: NSAMPLES * (1 - sExp) });
      if (typeof process !== 'undefined' && process.env?.GLASS_DEBUG && tests[tests.length - 1].res.pValue < 1e-4) {
        const ed = caseEdges(c);
        const rs = Array.from(scaled, (e, b) => ({ b, e, o: obs[b], z: (obs[b] - e) / Math.sqrt(Math.max(e, 1)) })).sort((x, y) => Math.abs(y.z) - Math.abs(x.z)).slice(0, 12);
        console.log(`DEBUG ${c.label} axes ${JSON.stringify(pilotAxes(c))}`);
        for (const r of rs) { const h = r.b >= NB ? 1 : 0, bb = r.b % NB, iu = Math.floor(bb / NPHI), ip = bb % NPHI;
          console.log(`  h${h} u[${ed[h * HE + iu].toExponential(4)},${ed[h * HE + iu + 1].toExponential(4)}] ψ[${ed[h * HE + NU + 1 + ip].toExponential(3)},${ed[h * HE + NU + 2 + ip].toExponential(3)}] obs ${r.o} exp ${r.e.toFixed(1)} z ${r.z.toFixed(1)}`); }
      }
    }
    console.log(`GLASS_CHI2 ${lane()} f64 adaptive expected counts for ${nCpu} cases: ${((performance.now() - tq) / 1000).toFixed(1)} s`);
    const alpha = sidak(0.01, tests.length);
    const fails = tests.filter((t) => !(t.res.pValue > alpha));
    const minP = tests.reduce((m, t) => Math.min(m, t.res.pValue), 1);
    console.log(`GLASS_CHI2 ${lane()} ${tests.length} tests, per-test α = ${alpha.toExponential(2)}, min p = ${minP.toExponential(2)}, failures ${fails.length}`);
    for (const t of fails) console.log(`  FAIL ${t.label} χ² = ${t.res.chi2.toFixed(1)} dof ${t.res.dof} p = ${t.res.pValue.toExponential(2)} rejected+delta ${t.outObs}/${t.outExp.toFixed(1)}`);
    expect(fails.map((t) => `${t.label} p=${t.res.pValue.toExponential(2)}`)).toEqual([]);
    const median = tests.map((t) => t.res.pValue).sort((a, b) => a - b)[tests.length >> 1];
    console.log(`GLASS_CHI2 ${lane()} median p = ${median.toFixed(3)}`);
    expect(median).toBeGreaterThan(0.1);
    // rejected + delta mass (outside) == 1 − ∫p_valid, e.g. η ≈ 1 (G_T delta), TIR, grazing: |Δ| ≤ 5σ + 3e-3·N
    let worstOut = 0;
    for (const t of tests) {
      const tol = 5 * Math.sqrt(Math.max(t.outExp, 1)) + 3e-3 * NSAMPLES;
      worstOut = Math.max(worstOut, Math.abs(t.outObs - t.outExp) / tol);
      expect(Math.abs(t.outObs - t.outExp), `${t.label}: rejected+delta ${t.outObs} vs 1 − ∫p_valid ${t.outExp.toFixed(1)}`).toBeLessThan(tol);
    }
    const eta1 = tests.find((x) => x.label === 'glass η=1.00005 r=0.2 θ=30')!;
    console.log(`GLASS_CHI2 ${lane()} rejected+delta mass: worst |Δ|/tol ${worstOut.toFixed(3)}; η≈1 (G_T delta) ${eta1.outObs} vs ${eta1.outExp.toFixed(0)}`);
    expect(eta1.outObs).toBeGreaterThan(0.9 * NSAMPLES);   // G_T is delta at |η − 1| < 1e-4: nearly all mass outside
    const controls = run.cases.map((c, ci) => ({ c, ci })).filter((x) => x.c.controlOf !== undefined);
    const ctl = controls.map(({ c, ci }) => {
      const bi = run.cases.findIndex((d) => d.label === c.controlOf);
      const obs = run.hist.subarray(bi * 2 * NB, (bi + 1) * 2 * NB);
      const exp = expectedProbs(c, caseEdges(c));
      const sObs = obs.reduce((a, b) => a + b, 0), sExp = exp.reduce((a, b) => a + b, 0);
      return { label: c.label, p: chi2Test(obs, exp.map((x) => (x / sExp) * sObs), sObs).pValue };
    });
    console.log(`GLASS_CHI2 ${lane()} negative controls: ${ctl.map((x) => `${x.label} p=${x.p.toExponential(1)}`).join('; ')}`);
    expect(controls.length).toBe(5);
    for (const x of ctl) expect(x.p, x.label).toBeLessThan(alpha);
  }, 900_000);

  it('U-G5 weight == eval_lobe/pdf_joint (every GPU sample); acceptance == support ∧ pdf > 0; no NaN/Inf', async () => {
    const run = await samplingRun();
    let gW = 0, gP = 0, gM = 0, mism = 0;
    const bad: string[] = [];
    run.cases.forEach((c, ci) => {
      gW = Math.max(gW, run.statsF[ci * 8 + 4]); gP = Math.max(gP, run.statsF[ci * 8 + 5]); gM = Math.max(gM, run.statsF[ci * 8 + 6]);
      mism += run.stats[ci * 8 + 2];
      if (run.stats[ci * 8 + 2] > 0) bad.push(`${c.label}: ${run.stats[ci * 8 + 2]} acceptance/support mismatches`);
      expect(run.stats[ci * 8 + 3], `${c.label} non-finite weights`).toBe(0);
    });
    console.log(`GLASS_WEIGHT_CONSISTENCY ${lane()} w ${gW.toExponential(2)} pj ${gP.toExponential(2)} pm ${gM.toExponential(2)}, support mismatches ${mism}`);
    for (const l of bad.slice(0, 10)) console.log(`  ${l}`);
    expect(gW).toBeLessThanOrEqual(1e-4);
    expect(gP).toBeLessThanOrEqual(1e-4);
    expect(gM).toBeLessThanOrEqual(1e-4);
    // f32 boundary flips (a sampled L whose re-derived half-vector lands on the other side of a validity test) are
    // measure-small; allow ≤ 1e-6 of all samples
    expect(mism).toBeLessThanOrEqual(1e-6 * run.cases.length * NSAMPLES);
  }, 900_000);

  it('U-G4 E[weight] == the glass §3 albedo table (total and R part) and table_ggx_glass_E at grid points (η > 1)', async () => {
    const run = await samplingRun();
    const mean = (ci: number) => {
      const s = [0, 0, 0, 0], s2 = [0, 0, 0, 0];
      for (let t = 0; t < THREADS; t++) for (let k = 0; k < 4; k++) { s[k] += run.partials[(ci * THREADS + t) * 8 + k]; s2[k] += run.partials[(ci * THREADS + t) * 8 + 4 + k]; }
      return s.map((x, k) => ({ m: x / NSAMPLES, se: Math.sqrt(Math.max(s2[k] / NSAMPLES - (x / NSAMPLES) ** 2, 0) / NSAMPLES) }));
    };
    let worstTab = 0, worstE = 0;
    for (const [ior, back, r, cols] of ALBEDO) for (const [mu, total, rPart] of cols) {
      const ci = run.cases.findIndex((c) => c.label === `albedo η=${back ? '1/' : ''}${ior} r=${r} μ=${mu}`);
      const a = mean(ci);
      // the table is a 1e6-sample f64 MC rounded to 4 decimals: 4σ (both sides) + 1e-4 (rounding + f32)
      const tol = 4 * Math.SQRT2 * a[0].se + 1e-4, tolR = 4 * Math.SQRT2 * a[3].se + 1e-4;
      worstTab = Math.max(worstTab, Math.abs(a[0].m - total) / tol, Math.abs(a[3].m - rPart) / tolR);
      expect(Math.abs(a[0].m - total), `${run.cases[ci].label}: ${a[0].m} ± ${a[0].se} vs ${total}`).toBeLessThan(tol);
      expect(Math.abs(a[3].m - rPart), `${run.cases[ci].label} R: ${a[3].m} ± ${a[3].se} vs ${rPart}`).toBeLessThan(tolR);
    }
    for (const [x, y, z] of GLASS_E_POINTS) {
      const ci = run.cases.findIndex((c) => c.label === `glass_E x=${x} y=${y} z=${z}`);
      const a = mean(ci);
      const E = TABLE_GGX_GLASS_E[256 * z + 16 * y + x];
      const tol = 3 * a[0].se + 2e-4;
      worstE = Math.max(worstE, Math.abs(a[0].m - E) / tol);
      expect(Math.abs(a[0].m - E), `glass_E(${x},${y},${z}) = ${E}: MC ${a[0].m} ± ${a[0].se}`).toBeLessThan(tol);
    }
    console.log(`U-G4 ${lane()} worst |Δ|/tol: §3 table ${worstTab.toFixed(3)}, table_ggx_glass_E ${worstE.toFixed(3)}`);
  }, 900_000);

  it('U-G5 delta weights (1−F)√C/(1−P_R)/q(G), F/P_R/q(G); U-G6 sample weights (glass §2.6); U-G7 delta per sub-event', async () => {
    const P = (t: number, o: Partial<Mat> = {}): Mat => ({ model: 2, base: [0.8, 0.8, 0.8], roughness: 0.5, ior: 1.5, transmission: t, ...o });
    const cases: Case[] = [
      { mat: P(1), V: [0, 0, 1], back: false, label: 'sw t1 μ1', chi2: false },
      { mat: P(1), V: [Math.sqrt(1 - 0.16), 0, 0.4], back: false, label: 'sw t1 μ.4', chi2: false },
      { mat: P(1), V: [Math.sqrt(1 - 0.16), 0, 0.4], back: true, label: 'sw t1 μ.4 back', chi2: false },
      { mat: P(0.5), V: [0, 0, 1], back: false, label: 'sw t.5 μ1', chi2: false },
      { mat: P(0.5), V: [Math.sqrt(1 - 0.01), 0, 0.1], back: false, label: 'sw t.5 μ.1', chi2: false },
      { mat: P(0.5, { base: [0.9, 0.6, 0.3], metallic: 0.25, roughness: 0.3 }), V: [Math.sqrt(1 - 0.49), 0, 0.7], back: false, label: 'sw mix', chi2: false },
      // delta: smooth Principled t=1 coloured, both R and T
      { mat: P(1, { base: [0.64, 0.36, 0.16], roughness: 0 }), V: [0.6, 0, 0.8], back: false, label: 'delta', chi2: false },
      // U-G7: r 0.0037 (delta) vs 0.0038; η ≈ 1 rough (G_T delta, G_R rough)
      { mat: glassMat(0.0037, 1.5), V: [0.6, 0, 0.8], back: false, label: 'r0037', chi2: false },
      { mat: glassMat(0.0038, 1.5), V: [0.6, 0, 0.8], back: false, label: 'r0038', chi2: false },
      { mat: glassMat(0.3, 1.00005), V: [0.6, 0, 0.8], back: false, label: 'eta1', chi2: false },
    ];
    const Lr: V3 = [-0.6, 0, 0.8], Lt: V3 = f32(norm([-0.4, 0, -0.9]));
    const q = (ci: number, u: [number, number, number, number], L: V3 = Lt): Query => ({ caseIdx: ci, V: f32(cases[ci].V), L, u });
    const out = await probe(cases, [q(0, [0.5, 0.5, 0.5, 0.5]), q(1, [0.5, 0.5, 0.5, 0.5]), q(2, [0.5, 0.5, 0.5, 0.5]), q(3, [0.5, 0.5, 0.5, 0.5]),
      q(4, [0.5, 0.5, 0.5, 0.5]), q(5, [0.5, 0.5, 0.5, 0.5]),
      q(6, [0.5, 0.5, 0.5, 0.999]), q(6, [0.5, 0.5, 0.5, 0.0], Lr),
      q(7, [0.5, 0.5, 0.5, 0.5], Lr), q(8, [0.5, 0.5, 0.5, 0.5], Lr), q(9, [0.5, 0.3, 0.7, 0.99]), q(9, [0.5, 0.3, 0.7, 0.0], Lr)]);
    // U-G6
    const swG = [0.8992, 0.8773, 0.1013, 0.4496, 0.2934];
    swG.forEach((w, i) => expect(Math.abs(out[i].swG - w), `${cases[i].label} sw_G ${out[i].swG}`).toBeLessThan(1e-4));
    for (const i of [0, 1, 2]) expect(out[i].qG).toBeCloseTo(1, 6);
    expect(Math.abs(out[3].qG - 0.5267)).toBeLessThan(1e-4); expect(Math.abs(out[3].qS - 0.0237)).toBeLessThan(1e-4); expect(Math.abs(out[3].qD - 0.4496)).toBeLessThan(1e-4);
    expect(Math.abs(out[4].qG - 0.4111)).toBeLessThan(1e-4); expect(Math.abs(out[4].qS - 0.1427)).toBeLessThan(1e-4); expect(Math.abs(out[4].qD - 0.4463)).toBeLessThan(1e-4);
    expect(Math.abs(out[5].swG - 0.2891)).toBeLessThan(1e-4); expect(Math.abs(out[5].swS - (0.1505 + 0.0197))).toBeLessThan(2e-4);
    expect(Math.abs(out[5].swD - 0.2132)).toBeLessThan(1e-4); expect(Math.abs(out[5].qG - 0.4299)).toBeLessThan(1e-4);
    // U-G5 delta weights (q(G) = 1 here)
    const F = glassFresnel(glassNode([1, 1, 1], 0, 1.5, 0.8, false), 0.8).R[0];
    const sq = [0.8, 0.6, 0.4], a = (sq[0] + sq[1] + sq[2]) / 3, PR = F / (F + (1 - F) * a);
    const dt = out[6], dr = out[7];
    expect(dt.sDelta && dt.sValid && dt.sLobe === LOBE_GT).toBe(true);
    dt.sW.forEach((w, c) => expect(Math.abs(w - (1 - F) * sq[c] / (1 - PR)) / w).toBeLessThan(1e-5));
    expect(dt.sPj).toBe(0); expect(dt.sPm).toBe(0);
    expect(dr.sDelta && dr.sValid && dr.sLobe === LOBE_GR).toBe(true);
    dr.sW.forEach((w) => expect(Math.abs(w - F / PR) / w).toBeLessThan(1e-5));
    // smooth pure glass: no non-delta lobe ⇒ no NEE; eval 0 (no SD_BSDF_HAS_EVAL)
    expect(dt.flags & MATEVAL_HAS_NON_DELTA).toBe(0);
    expect(dt.flags & MATEVAL_G_DELTA).toBe(MATEVAL_G_DELTA);
    expect(Math.max(...dt.f)).toBe(0);
    // U-G7: threshold 2e-10 on α²
    expect(out[8].flags & MATEVAL_G_DELTA).toBe(MATEVAL_G_DELTA);
    expect(out[8].flags & MATEVAL_HAS_NON_DELTA).toBe(0);
    expect(out[8].rGR).toBe(0);
    expect(out[9].flags & MATEVAL_G_DELTA).toBe(0);
    expect(out[9].flags & (MATEVAL_HAS_NON_DELTA | MATEVAL_HAS_G)).toBe(MATEVAL_HAS_NON_DELTA | MATEVAL_HAS_G);
    expect(out[9].rGR).toBeCloseTo(0.0038, 7);
    // η ≈ 1: G_T sampled as delta, lobe roughness 0; G_R rough (r 0.3); the eval still returns the spurious BTDF
    const e1 = out[10], e2 = out[11];
    expect(e1.rGT).toBe(0);
    expect(e1.rGR).toBeCloseTo(0.3, 6);
    expect(e1.rNEE).toBeCloseTo(0.3, 6);
    expect(e1.sLobe === LOBE_GT && e1.sDelta && e1.sValid).toBe(true);
    expect(e2.sLobe === LOBE_GR && !e2.sDelta && e2.sValid && e2.sPj > 0).toBe(true);
    expect(e1.gF).toBeGreaterThan(0);
    expect(e1.gValid).toBe(false);
    expect(e1.gPdfV).toBe(0);
  });

  it('U-G8 η_side = 1/ior iff the evaluating V is on the backfacing side of the winding normal (MATEVAL_BACKFACING ⊕ Ng·V < 0)', async () => {
    const cases: Case[] = [
      { mat: glassMat(0.3, 1.5), V: [0, 0, 1], back: false, label: 'front', chi2: false },
      { mat: glassMat(0.3, 1.5), V: [0, 0, 1], back: true, label: 'back', chi2: false },
    ];
    const up: V3 = f32(norm([0.3, 0, 0.9])), down: V3 = f32(norm([0.3, 0, -0.9]));
    const out = await probe(cases, [
      { caseIdx: 0, V: up, L: down, u: [0.5, 0.5, 0.5, 0.5] }, { caseIdx: 0, V: down, L: up, u: [0.5, 0.5, 0.5, 0.5] },
      { caseIdx: 1, V: up, L: down, u: [0.5, 0.5, 0.5, 0.5] }, { caseIdx: 1, V: down, L: up, u: [0.5, 0.5, 0.5, 0.5] },
    ]);
    // case_mat builds the MatEval with ng = +z for the probe V; a V below the plane flips the normals (two-sided)
    expect(out[0].eta).toBeCloseTo(1.5, 6);
    expect(out[1].eta).toBeCloseTo(1 / 1.5, 6);
    expect(out[2].eta).toBeCloseTo(1 / 1.5, 6);
    expect(out[3].eta).toBeCloseTo(1.5, 6);
    expect(out[0].flags & MATEVAL_BACKFACING).toBe(0);
    expect(out[2].flags & MATEVAL_BACKFACING).toBe(MATEVAL_BACKFACING);
  });

  it("Cycles' BTDF is not reciprocal (no η² radiance scaling): f_T(V,L)/|N·L| = η_side(V)²·f_T(L,V)/|N·V|; G_R reciprocal", async () => {
    const rnd = xorshift(29);
    const cases: Case[] = [0.2, 0.5, 0.9].flatMap((r) => [false, true].map((back) => ({ mat: glassMat(r, 1.5), V: [0, 0, 1] as V3, back, label: `r${r}`, chi2: false })));
    const queries: Query[] = [];
    cases.forEach((_, ci) => { for (let k = 0; k < 200; k++) { const V = randSphere(rnd); const L = randSphere(rnd); queries.push({ caseIdx: ci, V: f32([V[0], V[1], Math.abs(V[2]) + 0.05]), L: f32(L), u: [0.5, 0.5, 0.5, 0.5] }); } });
    const out = await probe(cases, queries.map((q) => ({ ...q, V: f32(norm(q.V)) })));
    let worstT = 0, worstR = 0, nT = 0, nR = 0;
    out.forEach((o, i) => {
      const V = norm(queries[i].V), L = queries[i].L;
      const fwd = o.f[0] / Math.abs(L[2]), rev = o.revF[0] / Math.abs(V[2]);
      if (!(fwd > 1e-3 && rev > 1e-3) || !o.gValid) return;
      if (L[2] < 0) { nT++; worstT = Math.max(worstT, Math.abs(fwd / (o.ctxEta * o.ctxEta * rev) - 1)); }
      else { nR++; worstR = Math.max(worstR, Math.abs(fwd / rev - 1)); }
    });
    console.log(`GLASS_RECIPROCITY ${lane()} T pairs ${nT}: max |ratio/η² − 1| ${worstT.toExponential(2)}; R pairs ${nR}: max |ratio − 1| ${worstR.toExponential(2)}`);
    expect(nT).toBeGreaterThan(100);
    expect(nR).toBeGreaterThan(100);
    expect(worstT).toBeLessThan(1e-3);
    expect(worstR).toBeLessThan(1e-3);
  });
});

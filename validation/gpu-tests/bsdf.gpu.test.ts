// T8 BSDF tests (plan §7.4 M3a = gap-bsdf U-1…U-10), both GPU lanes, on the production material/*.wgsl modules.
//  (a) χ²: bsdf_sample histograms vs the integrated joint pdf per lobe and the marginal pdf of the mixture
//      (Mitsuba-style, Šidák α = 0.01), θ_V ∈ {0,30,60,80,89}°, r ∈ {0.02,…,1.0}, V1 + two V2 materials
//  (b) weight == eval_lobe/pdf_joint (GPU self-consistency on every sample; f64 reference on recorded samples)
//  (c) E[weight] == ∫ f_cos over the hemisphere (GPU quadrature of bsdf_eval; f64 reference quadrature) and ≤ 1
//  (d) LUT reader == CPU port of lookup_table.h; table test vectors
//  (e) pdf_marginal == Σ q·p_ℓ; q(ℓ|V) == Cycles' albedo-scaled sample weights (f64 reference)
//  (f) singular threshold (r = 0.0037 delta, 0.0038 not)
//  (g) bsdf_sample_support == sampler acceptance, flat and smooth shading
//  (h) reciprocity: V1 reciprocal; V2 non-reciprocal by design (q and Λ depend on μ_V; gap-bsdf §6.9)
//  plus material_eval (material table + textures + COLOR_0 → MatEval).
// Histogram domain: upper hemisphere in (u = 1 − cosθ, φ) (dω = du dφ), bins equal-probability under a Cauchy
// mixture centred on the mirror direction (resolves r = 0.02 peaks), expected counts by 15×15 Gauss–Legendre per bin.
import { afterAll, describe, expect, it } from 'vitest';
import { getTestGpu, lane, releaseTestGpu } from './device-factory.ts';
import { composeWgsl, createCheckedShaderModule } from '../../src/core/gpu/wgsl-composer.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { createLutBuffer, lutDefines } from '../../src/core/render/luts/lut-layout.ts';
import { createGpuTextures } from '../../src/core/render/textures-gpu.ts';
import { MATERIAL_LAYOUT, packMaterials, SCENE_BINDING } from '../../src/core/render/scene-gpu.ts';
import type { MaterialData, TextureData } from '../../src/core/scene/types.ts';
import {
  FS, S_ior, S_s, dot, evalLocal, ggxE, ggxEavg, norm, prepare, type MatParams, type V3,
} from '../../tests/material/bsdf-ref.ts';
import { adaptiveEdges, chi2Test, sidak } from '../../tests/material/chi2.ts';
import { albedoQuadrature } from '../../tests/material/quadrature.ts';

const NU = 32;
const NPHI = 64;
const NB = NU * NPHI;
const EDGES = NU + 1 + NPHI + 1;
const THREADS = 4096;
const SPT = 256;
const NSAMPLES = THREADS * SPT;   // 1,048,576 per case
const RECS = 64;
const PROBE_N = 11;

const HARNESS = `
#include "material/bsdf.wgsl"
#include "common/rng.wgsl"

struct Case {
  model: u32, metallic: f32, roughness: f32, ior: f32,
  base: vec3f, specLevel: f32,
  tint: vec3f, v1Mix: f32,
  v1Diffuse: vec3f, seed: u32,
  v1Glossy: vec3f, qUc: f32,     // quadrature substitution centre / width per axis (k_quad)
  V: vec3f, qWu: f32,
  ns: vec3f, qPc: f32,
  ng: vec3f, qWp: f32,
}
struct Params { caseBase: u32, caseCount: u32, threadsPerCase: u32, samplesPerThread: u32 }
struct Query { caseIdx: u32, lobe: u32, pad0: u32, pad1: u32, V: vec4f, L: vec4f, u: vec4f }

const NU: u32 = ${NU}u;
const NPHI: u32 = ${NPHI}u;
const NB: u32 = ${NB}u;
const EDGES: u32 = ${EDGES}u;
const RECS: u32 = ${RECS}u;
const PROBE_N: u32 = ${PROBE_N}u;

@group(0) @binding(1) var<storage, read> cases: array<Case>;
@group(0) @binding(2) var<storage, read> edges: array<f32>;
@group(0) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> partials: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> quad: array<vec4f>;
@group(0) @binding(7) var<uniform> prm: Params;
@group(0) @binding(8) var<storage, read> queries: array<Query>;
@group(0) @binding(9) var<storage, read_write> probeOut: array<vec4f>;
@group(0) @binding(10) var<storage, read_write> recs: array<vec4f>;

fn case_mat(c: Case, V: vec3f) -> MatEval {
  var m: MatEval;
  m.model = c.model;
  m.base_color = select(c.base, c.v1Diffuse, c.model == 0u);   // V1 slots: Diffuse, Glossy colour, mix
  m.metallic = select(c.metallic, c.v1Mix, c.model == 0u);
  m.roughness = c.roughness;
  m.ior = c.ior;
  m.specular_level = c.specLevel;
  m.specular_tint = select(c.tint, c.v1Glossy, c.model == 0u);
  m.ns = c.ns;
  m.ng = c.ng;
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

// Cancellation-free asinh / sinh / cosh (Tint may lower asinh(x) to log(x + sqrt(x² + 1)), which is useless for x ≪ 0).
fn q_asinh(x: f32) -> f32 { let a = abs(x); return sign(x) * log(a + sqrt(a * a + 1.0)); }
fn q_sinh(v: f32) -> f32 {
  if (abs(v) < 1e-2) { return v * (1.0 + v * v * (1.0 / 6.0)); }
  let e = exp(abs(v));
  return sign(v) * 0.5 * (e - 1.0 / e);
}
fn q_cosh(v: f32) -> f32 { let e = exp(abs(v)); return 0.5 * (e + 1.0 / e); }

fn rel_err(a: f32, b: f32) -> f32 {
  let e = abs(a - b) / max(abs(b), 1e-30);
  return select(3e38, e, e <= 3e38);   // NaN / Inf -> huge
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
  var nValid = 0u;
  var nDelta = 0u;
  var nMismatch = 0u;
  var nNonFinite = 0u;
  var eW = 0.0;
  var eP = 0.0;
  var eM = 0.0;
  for (var i = 0u; i < prm.samplesPerThread; i++) {
    let h = pcg3d(vec3u(c.seed, t, i));
    let h2 = pcg3d(vec3u(c.seed ^ 0x5bd1e995u, t, i));
    let u = vec4f(u32_to_unit(h.x), u32_to_unit(h.y), u32_to_unit(h.z), u32_to_unit(h2.x));
    let s = bsdf_sample(m, V, u);
    if (t < RECS && i == 0u) {
      let r = (ci * RECS + t) * 3u;
      recs[r] = vec4f(s.L, f32(s.lobe));
      recs[r + 1u] = vec4f(s.weight, s.pdf_joint);
      recs[r + 2u] = vec4f(s.pdf_marginal, select(0.0, 1.0, s.is_delta), select(0.0, 1.0, s.valid), 0.0);
    }
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
    if (s.is_delta) { nDelta++; continue; }
    // (b) weight == eval_lobe / pdf_joint, pdf_joint == eval_lobe.w, pdf_marginal == bsdf_pdf_marginal
    let ev = bsdf_eval_lobe(m, V, s.L, s.lobe);
    let ref_w = ev.xyz / ev.w;
    let den = max(bsdf_max3(abs(ref_w)), 1e-30);
    eW = max(eW, bsdf_max3(abs(s.weight - ref_w)) / den);
    eP = max(eP, rel_err(s.pdf_joint, ev.w));
    eM = max(eM, rel_err(s.pdf_marginal, bsdf_pdf_marginal(m, V, s.L)));
    // (a) histogram in (u = 1 - cos, phi) of the local frame (N = +z for the chi2 cases)
    let L = s.L;
    let uu = (L.x * L.x + L.y * L.y) / (1.0 + max(L.z, 0.0));
    let psi = atan2(-L.y, -L.x);   // azimuth relative to the mirror azimuth (φ − π), accurate near the lobe centre
    let b = bin_search(eo, NU, uu) * NPHI + bin_search(eo + NU + 1u, NPHI, psi);
    let slot = select(1u, 0u, s.lobe == LOBE_D);
    atomicAdd(&hist[(ci * 3u + slot) * NB + b], 1u);
    atomicAdd(&hist[(ci * 3u + 2u) * NB + b], 1u);
  }
  partials[(ci * prm.threadsPerCase + t) * 2u] = vec4f(acc, 0.0);
  partials[(ci * prm.threadsPerCase + t) * 2u + 1u] = vec4f(acc2, 0.0);
  atomicAdd(&stats[ci * 8u + 0u], nValid);
  atomicAdd(&stats[ci * 8u + 1u], nDelta);
  atomicAdd(&stats[ci * 8u + 2u], nMismatch);
  atomicAdd(&stats[ci * 8u + 3u], nNonFinite);
  atomicMax(&stats[ci * 8u + 4u], bitcast<u32>(eW));
  atomicMax(&stats[ci * 8u + 5u], bitcast<u32>(eP));
  atomicMax(&stats[ci * 8u + 6u], bitcast<u32>(eM));
}

@compute @workgroup_size(64)
fn k_quad(@builtin(global_invocation_id) gid: vec3u) {
  let local = gid.x / NB;
  if (local >= prm.caseCount) { return; }
  let ci = prm.caseBase + local;
  let b = gid.x % NB;
  let c = cases[ci];
  let V = c.V;
  let ctx = bsdf_prepare(case_mat(c, V), V);
  let eo = ci * EDGES;
  let iu = b / NPHI;
  let ip = b % NPHI;
  let u0 = edges[eo + iu];
  let u1 = edges[eo + iu + 1u];
  let p0 = edges[eo + NU + 1u + ip];
  let p1 = edges[eo + NU + 1u + ip + 1u];
  var gx = array<f32, 5>(-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640);
  var gw = array<f32, 5>(0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891);
  let K = 3u;
  var sD = 0.0;
  var sS = 0.0;
  var sF = vec3f(0.0);
  // Integrate in v = q_asinh((x − c)/w) per axis: linear across the lobe core, logarithmic in its 1/x² tails and in the
  // wide uniform bins, so 3 × 5-point Gauss–Legendre per axis stays accurate for r = 0.02 lobes.
  let vu0 = q_asinh((u0 - c.qUc) / c.qWu);
  let hvu = (q_asinh((u1 - c.qUc) / c.qWu) - vu0) / f32(K);
  let vp0 = q_asinh((p0 - c.qPc) / c.qWp);
  let hvp = (q_asinh((p1 - c.qPc) / c.qWp) - vp0) / f32(K);
  for (var ku = 0u; ku < K; ku++) {
    for (var a = 0u; a < 5u; a++) {
      let vu = vu0 + hvu * (f32(ku) + 0.5 + 0.5 * gx[a]);
      let uu = clamp(c.qUc + c.qWu * q_sinh(vu), u0, u1);
      let du = c.qWu * q_cosh(vu);
      let st = sqrt(max(uu * (2.0 - uu), 0.0));
      for (var kp = 0u; kp < K; kp++) {
        for (var bb = 0u; bb < 5u; bb++) {
          let vp = vp0 + hvp * (f32(kp) + 0.5 + 0.5 * gx[bb]);
          let psi = clamp(c.qPc + c.qWp * q_sinh(vp), p0, p1);
          let dp = c.qWp * q_cosh(vp);
          let L = vec3f(-st * cos(psi), -st * sin(psi), 1.0 - uu);
          let wgt = gw[a] * gw[bb] * (0.5 * hvu) * (0.5 * hvp) * du * dp;
          let e = bsdf_eval_ctx(ctx, V, L);
          sD += e.p_d * wgt;
          sS += e.p_s * wgt;
          sF += (e.f_d + e.f_s) * wgt;
        }
      }
    }
  }
  quad[(ci * NB + b) * 2u] = vec4f(sD, sS, sD + sS, 0.0);
  quad[(ci * NB + b) * 2u + 1u] = vec4f(sF, 0.0);
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
  let ev = bsdf_eval(m, V, L);
  probeOut[o + 0u] = vec4f(ev.f_cos, ev.pdf_marginal);
  probeOut[o + 1u] = bsdf_eval_lobe(m, V, L, LOBE_D);
  probeOut[o + 2u] = bsdf_eval_lobe(m, V, L, LOBE_S);
  probeOut[o + 3u] = bsdf_eval_lobe(m, V, L, LOBE_NEE);
  probeOut[o + 4u] = vec4f(bsdf_pdf_marginal(m, V, L), select(0.0, 1.0, bsdf_sample_support(m, V, L, LOBE_D)),
                           select(0.0, 1.0, bsdf_sample_support(m, V, L, LOBE_S)), bsdf_eval_lobe(m, V, L, LOBE_NONE).w);
  let qq = bsdf_lobe_probs(m, V);
  probeOut[o + 5u] = vec4f(qq.x, qq.y, qq.z, f32(m.flags));
  let s = bsdf_sample(m, V, q.u);
  probeOut[o + 6u] = vec4f(s.L, f32(s.lobe));
  probeOut[o + 7u] = vec4f(s.weight, s.pdf_joint);
  probeOut[o + 8u] = vec4f(s.pdf_marginal, select(0.0, 1.0, s.is_delta), select(0.0, 1.0, s.valid), lobe_roughness(m, LOBE_NEE));
  probeOut[o + 9u] = vec4f(lobe_roughness(m, LOBE_D), lobe_roughness(m, LOBE_S), lobe_roughness(m, LOBE_NONE),
                           select(0.0, 1.0, bsdf_sample_support(m, V, L, LOBE_NEE)));
  probeOut[o + 10u] = vec4f(lut_ggx_gen_schlick_ior_s(q.u.x, q.u.y, q.u.z), lut_ggx_gen_schlick_s(q.u.x, q.u.y, q.u.z),
                            lut_ggx_E(q.u.x, q.u.y), lut_ggx_Eavg(q.u.x));
}
`;

// ------------------------------------------------------------------------------------------------ cases

interface Case {
  p: MatParams; V: V3; ns?: V3; ng?: V3; label: string; chi2: boolean;
  /** negative control: expected counts of this (wrong) material over the bins of the case labelled `controlOf` */
  controlOf?: string; axesOf?: Case;
}
/**
 * Lobe-centred axis parameters (histogram edges and the quadrature substitution): centre and width per axis.
 * Axes: u = 1 − cosθ_L ∈ [0, 1] and ψ = φ_L − φ_mirror ∈ [−π, π] (V lies in the xz plane at φ = 0, so the mirror
 * azimuth is π). Tilting H by δ in the plane of incidence moves L by du ≈ 2δ·sinθ_V; tilting it by δ out of the plane
 * moves L by L_y ≈ 2(V·H)δ, i.e. dψ ≈ 2δ(cosθ_V + δ)/sinθ_V — the lobe collapses in ψ at grazing incidence.
 */
function caseAxes(c0: Case) {
  const c = c0.axesOf ?? c0;
  const r = Math.min(Math.max(c.p.roughness, 0), 1);
  const alpha = r * r;
  const sinT = Math.hypot(c.V[0], c.V[1]);
  const uc = (c.V[0] * c.V[0] + c.V[1] * c.V[1]) / (1 + c.V[2]);   // 1 − cosθ_R (mirror), cancellation-free
  const wu = Math.max(2 * alpha * sinT, 2 * alpha * alpha, 1e-9);
  const wpsi = sinT > 1e-6 ? Math.min((2 * alpha * (c.V[2] + alpha)) / sinT, 1e3) : 1e3;
  return { uc, wu, pc: 0, wphi: wpsi };
}
const CASE_BYTES = 128;

const dirTheta = (deg: number): V3 => { const t = (deg * Math.PI) / 180; return [Math.sin(t), 0, Math.cos(t)]; };
const rotY = (v: V3, deg: number): V3 => { const t = (deg * Math.PI) / 180, c = Math.cos(t), s = Math.sin(t); return [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]]; };

const MATS: Record<string, (r: number) => MatParams> = {
  v1: (r) => ({ model: 0, diffuse: [0.8, 0.8, 0.8], glossy: [0.9, 0.9, 0.9], roughness: r, mix: 0.5 }),
  v2d: (r) => ({ model: 1, base: [0.8, 0.8, 0.8], metallic: 0, roughness: r, ior: 1.5, specLevel: 0.5, specTint: [1, 1, 1] }),
  v2m: (r) => ({ model: 1, base: [0.9, 0.6, 0.3], metallic: 0.5, roughness: r, ior: 1.45, specLevel: 1.0, specTint: [1, 0.5, 0.25] }),
};
const THETAS = [0, 30, 60, 80, 89];
const ROUGH = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0];

function buildCases(): Case[] {
  const out: Case[] = [];
  for (const [name, mk] of Object.entries(MATS)) for (const th of THETAS) for (const r of ROUGH) out.push({ p: mk(r), V: dirTheta(th), label: `${name} θ=${th} r=${r}`, chi2: true });
  // smooth shading (Ns ≠ Ng) and a MatEval whose normals face away from V (two-sided flip): support / consistency only
  const z: V3 = [0, 0, 1];
  for (const name of ['v1', 'v2d', 'v2m']) for (const r of [0.05, 0.3]) for (const [th, tilt] of [[0, 25], [60, 25], [85, 25], [-80, 25], [70, -40]]) {
    out.push({ p: MATS[name](r), V: dirTheta(th), ns: rotY(z, tilt), ng: z, label: `${name} smooth θ=${th} tilt=${tilt} r=${r}`, chi2: false });
  }
  out.push({ p: MATS.v2m(0.3), V: dirTheta(40), ns: [0, 0, -1], ng: [0, 0, -1], label: 'v2m flipped normals', chi2: false });
  // negative controls for the χ² power (small, physically plausible parameter errors)
  const ctl = (base: string, p: MatParams, what: string) => {
    const b = out.find((c) => c.label === base)!;
    out.push({ p, V: b.V, label: `${base} with ${what}`, chi2: false, controlOf: base, axesOf: b });
  };
  ctl('v1 θ=30 r=0.3', MATS.v1(0.3 * 1.05), 'r ×1.05');
  ctl('v1 θ=0 r=0.02', MATS.v1(0.021), 'r = 0.021');
  ctl('v1 θ=60 r=1', { ...(MATS.v1(1) as Extract<MatParams, { model: 0 }>), mix: 0.52 }, 'mix 0.52');
  ctl('v2d θ=60 r=0.5', { ...(MATS.v2d(0.5) as Extract<MatParams, { model: 1 }>), ior: 1.6 }, 'IOR 1.6');
  ctl('v2d θ=89 r=0.1', MATS.v2d(0.105), 'r = 0.105');
  ctl('v2m θ=80 r=0.2', { ...(MATS.v2m(0.2) as Extract<MatParams, { model: 1 }>), metallic: 0.55 }, 'metallic 0.55');
  return out;
}

function packCases(cases: Case[]): ArrayBuffer {
  // Case layout (WGSL): model 0, metallic 4, roughness 8, ior 12, base 16, specLevel 28, tint 32, v1Mix 44,
  // v1Diffuse 48, seed 60, v1Glossy 64, V 80, ns 96, ng 112; size 128.
  const buf = new ArrayBuffer(cases.length * CASE_BYTES);
  const dv = new DataView(buf);
  cases.forEach((c, i) => {
    const b = i * CASE_BYTES;
    const f = (o: number, v: number) => dv.setFloat32(b + o, v, true);
    const v3 = (o: number, v: V3) => { f(o, v[0]); f(o + 4, v[1]); f(o + 8, v[2]); };
    const p = c.p;
    dv.setUint32(b, p.model, true);
    f(8, p.roughness);
    if (p.model === 1) { f(4, p.metallic); f(12, p.ior); v3(16, p.base); f(28, p.specLevel); v3(32, p.specTint); }
    else { f(44, p.mix); v3(48, p.diffuse); v3(64, p.glossy); }
    dv.setUint32(b + 60, Math.imul(0x9e3779b9, i + 1) >>> 0, true);
    v3(80, c.V); v3(96, c.ns ?? [0, 0, 1]); v3(112, c.ng ?? [0, 0, 1]);
    const ax = caseAxes(c);
    f(76, ax.uc); f(92, ax.wu); f(108, ax.pc); f(124, ax.wphi);
  });
  return buf;
}

/** Histogram bin edges of one case: u = 1 − cosθ ∈ [0, 1] and φ ∈ [0, 2π], concentrated on the mirror lobe. */
function caseEdges(c: Case): Float32Array {
  const { uc, wu, pc, wphi } = caseAxes(c);
  const out = new Float32Array(EDGES);
  out.set(adaptiveEdges(NU, 0, 1, uc, wu, wu < 0.5 ? 0.75 : 0), 0);
  out.set(adaptiveEdges(NPHI, -Math.PI, Math.PI, pc, wphi, wphi < 2 ? 0.75 : 0), NU + 1);
  return out;
}

// ------------------------------------------------------------------------------------------------ GPU plumbing

interface Harness { device: GPUDevice; lut: GPUBuffer; pSample: GPUComputePipeline; pQuad: GPUComputePipeline; pProbe: GPUComputePipeline }
let harnessP: Promise<Harness> | undefined;
function harness(): Promise<Harness> {
  harnessP ??= (async () => {
    const { device, features, wgslLanguageFeatures } = await getTestGpu();
    const shader = composeWgsl('tests/bsdf_harness.wgsl', {
      sources: { ...shaderSources, 'tests/bsdf_harness.wgsl': HARNESS },
      defines: lutDefines({ declare: { group: 0, binding: 0 } }), features, wgslLanguageFeatures,
    });
    const module = await createCheckedShaderModule(device, shader, 'bsdf-harness');
    const mk = (entryPoint: string) => device.createComputePipelineAsync({ label: entryPoint, layout: 'auto', compute: { module, entryPoint } });
    const [pSample, pQuad, pProbe] = await Promise.all([mk('k_sample'), mk('k_quad'), mk('k_probe')]);
    return { device, lut: createLutBuffer(device), pSample, pQuad, pProbe };
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

interface SamplingRun {
  cases: Case[];
  hist: Uint32Array; partials: Float32Array; stats: Uint32Array; statsF: Float32Array; quad: Float32Array; recs: Float32Array;
}
let runP: Promise<SamplingRun> | undefined;
function samplingRun(): Promise<SamplingRun> {
  runP ??= (async () => {
    const h = await harness();
    const { device } = h;
    const cases = buildCases();
    const n = cases.length;
    const edges = new Float32Array(n * EDGES);
    cases.forEach((c, i) => edges.set(caseEdges(c), i * EDGES));
    const bCases = storage(device, packCases(cases), 'cases');
    const bEdges = storage(device, edges, 'edges');
    const bHist = storage(device, n * 3 * NB * 4, 'hist');
    const bPart = storage(device, n * THREADS * 32, 'partials');
    const bStats = storage(device, n * 8 * 4, 'stats');
    const bQuad = storage(device, n * NB * 32, 'quad');
    const bRecs = storage(device, n * RECS * 48, 'recs');
    const BATCH = 4;
    const t0 = performance.now();
    for (let base = 0; base < n; base += BATCH) {
      const count = Math.min(BATCH, n - base);
      const prm = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(prm, 0, new Uint32Array([base, count, THREADS, SPT]));
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(h.pSample);
      pass.setBindGroup(0, bindAll(device, h.pSample, [[0, h.lut], [1, bCases], [2, bEdges], [3, bHist], [4, bPart], [5, bStats], [7, prm], [10, bRecs]]));
      pass.dispatchWorkgroups(Math.ceil((count * THREADS) / 64));
      pass.setPipeline(h.pQuad);
      pass.setBindGroup(0, bindAll(device, h.pQuad, [[0, h.lut], [1, bCases], [2, bEdges], [6, bQuad], [7, prm]]));
      pass.dispatchWorkgroups(Math.ceil((count * NB) / 64));
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      prm.destroy();
    }
    console.log(`BSDF_SAMPLING ${lane()} ${n} cases x ${NSAMPLES} samples: ${((performance.now() - t0) / 1000).toFixed(2)} s`);
    const rd = async (b: GPUBuffer, bytes: number) => readBuffer(device, b, bytes);
    const statsBuf = await rd(bStats, n * 32);
    const run: SamplingRun = {
      cases,
      hist: new Uint32Array(await rd(bHist, n * 3 * NB * 4)),
      partials: new Float32Array(await rd(bPart, n * THREADS * 32)),
      stats: new Uint32Array(statsBuf), statsF: new Float32Array(statsBuf),
      quad: new Float32Array(await rd(bQuad, n * NB * 32)),
      recs: new Float32Array(await rd(bRecs, n * RECS * 48)),
    };
    for (const b of [bCases, bEdges, bHist, bPart, bStats, bQuad, bRecs]) b.destroy();
    return run;
  })();
  return runP;
}

interface Query { caseIdx: number; V: V3; L: V3; u: [number, number, number, number] }
interface ProbeOut {
  f: V3; pdfM: number; lobeD: number[]; lobeS: number[]; lobeNEE: number[]; pdfMarg: number; supD: boolean; supS: boolean; noneW: number;
  qD: number; qS: number; qG: number; flags: number;
  sL: V3; sLobe: number; sW: V3; sPj: number; sPm: number; sDelta: boolean; sValid: boolean; rNEE: number;
  rD: number; rS: number; rNone: number; supNEE: boolean; lut: number[];
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
  const bCases = storage(device, packCases(cases), 'cases');
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
    const [o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10] = Array.from({ length: PROBE_N }, (_, k) => a(k));
    return {
      f: [o0[0], o0[1], o0[2]], pdfM: o0[3], lobeD: o1, lobeS: o2, lobeNEE: o3, pdfMarg: o4[0], supD: o4[1] > 0.5, supS: o4[2] > 0.5, noneW: o4[3],
      qD: o5[0], qS: o5[1], qG: o5[2], flags: o5[3],
      sL: [o6[0], o6[1], o6[2]], sLobe: o6[3], sW: [o7[0], o7[1], o7[2]], sPj: o7[3], sPm: o8[0], sDelta: o8[1] > 0.5, sValid: o8[2] > 0.5, rNEE: o8[3],
      rD: o9[0], rS: o9[1], rNone: o9[2], supNEE: o9[3] > 0.5, lut: o10,
    };
  });
}

function xorshift(seed: number) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}
const randHemi = (rnd: () => number, minZ = 0.02): V3 => {
  for (;;) {
    const z = rnd(), p = 2 * Math.PI * rnd(), s = Math.sqrt(1 - z * z);
    if (z >= minZ) return [s * Math.cos(p), s * Math.sin(p), z];
  }
};
const f32 = (v: V3): V3 => [Math.fround(v[0]), Math.fround(v[1]), Math.fround(v[2])];
const relErr = (a: number, b: number, floor = 1e-30) => Math.abs(a - b) / Math.max(Math.abs(b), floor);
const maxRel3 = (a: ArrayLike<number>, b: ArrayLike<number>, floor: number) => {
  const den = Math.max(Math.abs(b[0]), Math.abs(b[1]), Math.abs(b[2]), floor);
  return Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]), Math.abs(a[2] - b[2])) / den;
};

// ------------------------------------------------------------------------------------------------ tests

afterAll(async () => {
  const h = harnessP ? await harnessP : undefined;
  h?.lut.destroy();
  await releaseTestGpu();   // once, after every describe of this file (the device is shared)
});

describe(`BSDF T8 (${lane()})`, () => {

  it('(d) LUT reader on the GPU == CPU port of lookup_table.h; table test vectors', async () => {
    const rnd = xorshift(7);
    const cases: Case[] = [{ p: MATS.v1(0.5), V: [0, 0, 1], label: 'lut', chi2: false }];
    const pts: [number, number, number][] = [];
    for (let i = 0; i < 4000; i++) pts.push([-0.1 + 1.2 * rnd(), -0.1 + 1.2 * rnd(), -0.1 + 1.2 * rnd()]);
    for (let i = 0; i <= 15; i++) pts.push([i / 15, (15 - i) / 15, (i % 4) / 15]);    // exact grid nodes (t == 0 branch)
    for (let i = 0; i <= 31; i++) pts.push([i / 31, i / 31, 0.5]);
    const zIor = Math.sqrt(0.5 / 2.5);
    pts.push([0.5, 0.4, zIor], [0.5, 0.1, 0.5], [0.5, 1, 0.5], [1, 1, 0.5]);
    const out = await probe(cases, pts.map((p) => ({ caseIdx: 0, V: [0, 0, 1], L: [0, 0, 1], u: [Math.fround(p[0]), Math.fround(p[1]), Math.fround(p[2]), 0] })));
    let maxErr = 0;
    out.forEach((o, i) => {
      const [x, y, z] = pts[i].map(Math.fround);
      const ref = [S_ior(x, y, z), S_s(x, y, z), ggxE(x, y), ggxEavg(x)];
      for (let k = 0; k < 4; k++) maxErr = Math.max(maxErr, Math.abs(o.lut[k] - ref[k]));
    });
    console.log(`LUT_READER ${lane()} max |gpu − cpu| = ${maxErr.toExponential(2)} over ${pts.length} points`);
    expect(maxErr).toBeLessThan(1e-6);
    const n = out.length;
    expect(Math.abs(out[n - 4].lut[0] - 0.0639)).toBeLessThan(5.01e-5);            // S_ior(0.5, 0.4, IOR 1.5)
    expect(Math.abs(0.04 + 0.96 * out[n - 4].lut[0] - 0.1013)).toBeLessThan(5.01e-5); // E_spec
    expect(Math.abs(out[n - 3].lut[1] - 0.1718)).toBeLessThan(5.01e-5);             // S5(0.5, 0.1)
    expect(Math.abs(out[n - 2].lut[2] - 0.91528)).toBeLessThan(5.01e-6);            // ggx_E(0.5, 1)
    expect(Math.abs(out[n - 2].lut[3] - 0.88204)).toBeLessThan(5.01e-6);            // ggx_Eavg(0.5)
    expect(Math.abs(out[n - 1].lut[3] - 0.40914)).toBeLessThan(5.01e-6);            // ggx_Eavg(1)
  });

  it('(e) eval / pdfs == f64 reference; pdf_marginal == Σ q·p_ℓ; q(ℓ|V) == Cycles sample weights; lobe roughness', async () => {
    const rnd = xorshift(11);
    const cases: Case[] = [];
    for (const name of Object.keys(MATS)) for (const r of [0.02, 0.05, 0.2, 0.5, 1.0]) cases.push({ p: MATS[name](r), V: [0, 0, 1], label: `${name} r=${r}`, chi2: false });
    const queries: Query[] = [];
    cases.forEach((_, ci) => { for (let k = 0; k < 64; k++) queries.push({ caseIdx: ci, V: f32(randHemi(rnd)), L: f32(randHemi(rnd, 0)), u: [0.5, 0.5, 0.5, 0.5] }); });
    // near-mirror pairs (the GGX peak) for the sharp lobes
    cases.forEach((c, ci) => {
      for (let k = 0; k < 16; k++) {
        const V = randHemi(rnd, 0.05);
        const a = c.p.roughness ** 2;
        const R: V3 = [-V[0], -V[1], V[2]];
        queries.push({ caseIdx: ci, V: f32(V), L: f32(norm([R[0] + a * (rnd() - 0.5), R[1] + a * (rnd() - 0.5), R[2]])), u: [0.5, 0.5, 0.5, 0.5] });
      }
    });
    const out = await probe(cases, queries);
    let eF = 0, eP = 0, eQ = 0, eMix = 0;
    out.forEach((o, i) => {
      const q = queries[i];
      const p = cases[q.caseIdx].p;
      const ref = evalLocal(p, q.V, q.L);
      const c = prepare(p, q.V[2]);
      const fRef = [0, 1, 2].map((k) => ref.fD[k] + ref.fS[k]);
      eF = Math.max(eF, maxRel3(o.f, fRef, 1e-3));
      eP = Math.max(eP, relErr(o.pdfM, ref.pD + ref.pS, 1e-3));
      eQ = Math.max(eQ, Math.abs(o.qD - c.qD), Math.abs(o.qS - c.qS));
      // mixture pdf == Σ q·p_ℓ (same code path: exact up to one f32 add), bsdf_pdf_marginal identical
      eMix = Math.max(eMix, relErr(o.pdfM, o.lobeD[3] + o.lobeS[3], 1e-30));
      expect(o.pdfMarg).toBe(o.pdfM);
      expect(Math.abs(o.qD + o.qS - 1)).toBeLessThan(1e-6);
      expect(o.qG).toBe(0);
      // NEE: all lobes, pdf 1; NONE: pdf 1 (p_k := 1)
      [0, 1, 2].forEach((k) => expect(o.lobeNEE[k]).toBe(o.f[k]));
      expect(o.lobeNEE[3]).toBe(1);
      expect(o.noneW).toBe(1);
      expect(o.supNEE).toBe(true);
      // lobe roughness (math.md#lobe-codes)
      expect(o.rD).toBe(1);
      expect(o.rS).toBeCloseTo(p.roughness, 6);
      expect(o.rNone).toBeGreaterThan(1e38);
      expect(o.rNEE).toBe(1);
    });
    // smooth shading: a direction below Ng but above Ns keeps Cycles' eval f > 0, but no sampler can produce it, so
    // every pdf (joint and marginal) is 0 there (support-consistent densities, plan §1.5)
    const zN: V3 = [0, 0, 1];
    const sm: Case[] = [MATS.v1(0.4), MATS.v2d(0.4)].map((p, i) => ({ p, V: zN, ns: norm([0.5, 0, 1]), ng: zN, label: `smooth ${i}`, chi2: false }));
    const Lbelow = f32(norm([1, 0, -0.1]));
    const Vs = f32(norm([-0.6, 0, 1]));
    const os = await probe(sm, sm.map((_, ci) => ({ caseIdx: ci, V: Vs, L: Lbelow, u: [0.5, 0.5, 0.5, 0.5] as [number, number, number, number] })));
    for (const o of os) {
      expect(Math.max(...o.f)).toBeGreaterThan(0);
      expect(o.pdfM).toBe(0);
      expect(o.lobeD[3]).toBe(0);
      expect(o.lobeS[3]).toBe(0);
      expect(Math.max(o.lobeD[0], o.lobeS[0])).toBeGreaterThan(0);
      expect(o.supD).toBe(false);
      expect(o.supS).toBe(false);
    }
    console.log(`EVAL_VS_REF ${lane()} max rel f ${eF.toExponential(2)} pdf ${eP.toExponential(2)} |Δq| ${eQ.toExponential(2)} mix ${eMix.toExponential(2)}`);
    expect(eF).toBeLessThan(2e-4);
    expect(eP).toBeLessThan(2e-4);
    expect(eQ).toBeLessThan(1e-5);
    expect(eMix).toBeLessThan(1e-6);
  });

  it('(a) χ² per lobe (joint pdf) and for the mixture (marginal pdf), Šidák α = 0.01', async () => {
    const run = await samplingRun();
    const chi = run.cases.map((c, ci) => ({ c, ci })).filter((x) => x.c.chi2);
    const tests: { label: string; slot: string; res: ReturnType<typeof chi2Test> }[] = [];
    for (const { c, ci } of chi) {
      for (let slot = 0; slot < 3; slot++) {
        const obs = run.hist.subarray((ci * 3 + slot) * NB, (ci * 3 + slot + 1) * NB);
        const exp = new Float64Array(NB);
        for (let b = 0; b < NB; b++) exp[b] = NSAMPLES * run.quad[(ci * NB + b) * 8 + slot];
        tests.push({ label: c.label, slot: ['D', 'S', 'mix'][slot], res: chi2Test(obs, exp, NSAMPLES) });
      }
    }
    const alpha = sidak(0.01, tests.length);
    const fails = tests.filter((t) => !(t.res.pValue > alpha));
    const minP = tests.reduce((m, t) => Math.min(m, t.res.pValue), 1);
    console.log(`CHI2 ${lane()} ${tests.length} tests, per-test α = ${alpha.toExponential(2)}, min p = ${minP.toExponential(2)}, failures ${fails.length}`);
    for (const t of fails) console.log(`  FAIL ${t.label} [${t.slot}] χ² = ${t.res.chi2.toFixed(1)} dof ${t.res.dof} p = ${t.res.pValue.toExponential(2)} out ${t.res.outsideObs}/${t.res.outsideExp.toFixed(1)}`);
    expect(fails.map((t) => `${t.label} [${t.slot}] p=${t.res.pValue.toExponential(2)}`)).toEqual([]);
    // the p-values of a correct sampler are ~U(0,1): a lower-tail check guards against a degenerate statistic
    const median = tests.map((t) => t.res.pValue).sort((a, b) => a - b)[tests.length >> 1];
    console.log(`CHI2 ${lane()} median p = ${median.toFixed(3)}`);
    expect(median).toBeGreaterThan(0.1);
    // negative controls (power): the histogram of a case against the expected counts of a slightly wrong material,
    // integrated over the SAME bins (control cases share the axes of their base case)
    const controls = run.cases.map((c, ci) => ({ c, ci })).filter((x) => x.c.controlOf !== undefined);
    const ctl = controls.map(({ c, ci }) => {
      const bi = run.cases.findIndex((d) => d.label === c.controlOf);
      const obs = run.hist.subarray((bi * 3 + 2) * NB, (bi * 3 + 3) * NB);
      const exp = new Float64Array(NB);
      for (let b = 0; b < NB; b++) exp[b] = NSAMPLES * run.quad[(ci * NB + b) * 8 + 2];
      return { label: c.label, p: chi2Test(obs, exp, NSAMPLES).pValue };
    });
    console.log(`CHI2 ${lane()} negative controls: ${ctl.map((x) => `${x.label} p=${x.p.toExponential(1)}`).join('; ')}`);
    expect(controls.length).toBeGreaterThanOrEqual(6);
    for (const x of ctl) expect(x.p, x.label).toBeLessThan(alpha);
  }, 600_000);

  it('(b) sample weight == eval_lobe / pdf_joint (GPU, every sample) and == f64 reference (recorded samples)', async () => {
    const run = await samplingRun();
    // f64 cross-check of the recorded samples. The pdf of a sharp lobe is ill-conditioned in the f32 INPUT direction:
    // L carries ~6e-8 absolute error while H moves by δL/(2 V·H) against a lobe width α = r², so the tolerance is
    // 1e-4 + 8·6e-8/(α·(cosθ_V + α)) (a D-lobe sample: 1e-4). Weights do not involve D and must match to 1e-4.
    let gW = 0, gP = 0, gM = 0, rW = 0, worstRatio = 0, worstP = 0, nRec = 0;
    run.cases.forEach((c, ci) => {
      gW = Math.max(gW, run.statsF[ci * 8 + 4]); gP = Math.max(gP, run.statsF[ci * 8 + 5]); gM = Math.max(gM, run.statsF[ci * 8 + 6]);
      expect(run.stats[ci * 8 + 3], `${c.label} non-finite weights`).toBe(0);
      if (!c.chi2) return;
      for (let t = 0; t < RECS; t++) {
        const r = (ci * RECS + t) * 12;
        const L: V3 = [run.recs[r], run.recs[r + 1], run.recs[r + 2]];
        const lobe = run.recs[r + 3];
        if (!(run.recs[r + 10] > 0.5) || run.recs[r + 9] > 0.5) continue;   // invalid or delta
        const ref = evalLocal(c.p, c.V, L);
        const fl = lobe === 0 ? ref.fD : ref.fS;
        const pl = lobe === 0 ? ref.pD : ref.pS;
        rW = Math.max(rW, maxRel3([run.recs[r + 4], run.recs[r + 5], run.recs[r + 6]], fl.map((x) => x / pl), 1e-6));
        const a = c.p.roughness ** 2;
        const tolP = lobe === 0 ? 1e-4 : 1e-4 + (8 * 6e-8) / (a * (c.V[2] + a));
        const eP = Math.max(relErr(run.recs[r + 7], pl), relErr(run.recs[r + 8], ref.pD + ref.pS));
        worstP = Math.max(worstP, eP);
        worstRatio = Math.max(worstRatio, eP / tolP);
        nRec++;
      }
    });
    console.log(`WEIGHT_CONSISTENCY ${lane()} gpu: w ${gW.toExponential(2)} pj ${gP.toExponential(2)} pm ${gM.toExponential(2)}; vs f64 (${nRec} samples): w ${rW.toExponential(2)}, pdfs ${worstP.toExponential(2)} (max err/tol ${worstRatio.toFixed(3)})`);
    expect(gW).toBeLessThanOrEqual(1e-4);
    expect(gP).toBeLessThanOrEqual(1e-4);
    expect(gM).toBeLessThanOrEqual(1e-4);
    expect(nRec).toBeGreaterThan(5000);
    expect(rW).toBeLessThanOrEqual(1e-4);
    expect(worstRatio).toBeLessThanOrEqual(1);
  }, 600_000);

  it('(c) directional albedo: E[weight] == ∫ f_cos dω (GPU quadrature; f64 quadrature) and ≤ 1', async () => {
    const run = await samplingRun();
    let worstZ = 0, worstAbs = 0, worstCpu = 0, maxA = 0;
    const lines: string[] = [];
    run.cases.forEach((c, ci) => {
      if (!c.chi2) return;
      const s = [0, 0, 0], s2 = [0, 0, 0];
      for (let t = 0; t < THREADS; t++) for (let k = 0; k < 3; k++) { s[k] += run.partials[(ci * THREADS + t) * 8 + k]; s2[k] += run.partials[(ci * THREADS + t) * 8 + 4 + k]; }
      const q = [0, 0, 0];
      for (let b = 0; b < NB; b++) for (let k = 0; k < 3; k++) q[k] += run.quad[(ci * NB + b) * 8 + 4 + k];
      for (let k = 0; k < 3; k++) {
        const mean = s[k] / NSAMPLES;
        const sigma = Math.sqrt(Math.max(s2[k] / NSAMPLES - mean * mean, 0) / NSAMPLES);
        const d = Math.abs(mean - q[k]);
        worstAbs = Math.max(worstAbs, d - 3 * sigma);
        worstZ = Math.max(worstZ, d / Math.max(sigma, 1e-6));
        maxA = Math.max(maxA, q[k], mean - 3 * sigma);
        if (!(d <= 1e-3 + 3 * sigma)) lines.push(`${c.label}[${k}] E[w] ${mean.toFixed(5)} ± ${sigma.toExponential(1)} vs ∫f ${q[k].toFixed(5)}`);
      }
      if ([0.05, 0.3, 1.0].includes(c.p.roughness)) {
        const cpu = albedoQuadrature(c.p, c.V, 512);
        for (let k = 0; k < 3; k++) {
          worstCpu = Math.max(worstCpu, Math.abs(cpu[k] - q[k]));
          if (!(Math.abs(cpu[k] - q[k]) <= 1e-3)) lines.push(`${c.label}[${k}] GPU ∫f ${q[k].toFixed(5)} vs f64 ${cpu[k].toFixed(5)}`);
        }
      }
    });
    console.log(`ALBEDO ${lane()} max |E[w] − ∫f| − 3σ = ${worstAbs.toExponential(2)} (max z ${worstZ.toFixed(2)}), max |GPU ∫f − f64| = ${worstCpu.toExponential(2)}, max A = ${maxA.toFixed(5)}`);
    for (const l of lines) console.log(`  ${l}`);
    expect(lines).toEqual([]);
    expect(maxA).toBeLessThanOrEqual(1 + 1e-3);
  }, 600_000);

  it('(g) bsdf_sample_support == sampler acceptance (flat, smooth shading, flipped normals)', async () => {
    const run = await samplingRun();
    let rejectedSmooth = 0;
    run.cases.forEach((c, ci) => {
      expect(run.stats[ci * 8 + 2], `${c.label}: support/acceptance mismatches`).toBe(0);
      if (!c.chi2) rejectedSmooth += NSAMPLES - run.stats[ci * 8];
    });
    console.log(`SUPPORT ${lane()} 0 mismatches; ${rejectedSmooth} rejected samples in the smooth-shading cases`);
    expect(rejectedSmooth).toBeGreaterThan(10000);   // the smooth cases exercise the Ng/Ns rejections
  }, 600_000);

  it('(f) singular threshold: r = 0.0037 is a delta lobe, r = 0.0038 is not', async () => {
    const mats: MatParams[] = [MATS.v1(0.0037), MATS.v2m(0.0037), MATS.v1(0.0038), MATS.v2m(0.0038)];
    const V = f32(dirTheta(30));
    const mirror = f32([-V[0], -V[1], V[2]]);
    const cases: Case[] = mats.map((p, i) => ({ p, V, label: `singular ${i}`, chi2: false }));
    const queries: Query[] = cases.map((_, ci) => ({ caseIdx: ci, V, L: mirror, u: [0.99999, 0.3, 0.7, 0.5] }));
    const out = await probe(cases, queries);
    out.forEach((o, i) => {
      const p = mats[i];
      const c = prepare(p, V[2]);
      const ref = evalLocal(p, V, mirror);
      expect(o.sLobe).toBe(1);
      expect(o.sValid).toBe(true);
      if (i < 2) {
        expect(c.singular).toBe(true);
        expect(o.sDelta).toBe(true);
        mirror.forEach((x, k) => expect(Math.abs(o.sL[k] - x)).toBeLessThan(1e-6));
        const F = FS(c, V[2]);
        expect(maxRel3(o.sW, F.map((x) => x / c.qS), 1e-6)).toBeLessThan(1e-5);   // F/q(S)
        expect(o.sPj).toBe(0);
        expect(o.sPm).toBe(0);
        expect(o.lobeS).toEqual([0, 0, 0, 0]);                                     // never evaluated by NEE
        expect(maxRel3(o.f, ref.fD, 1e-6)).toBeLessThan(1e-5);                     // eval = D part only
        expect(relErr(o.pdfM, c.qD * mirror[2] / Math.PI)).toBeLessThan(1e-5);     // q(S) stays in the normaliser
        expect(o.rS).toBe(0);
        expect((o.flags & 16) !== 0).toBe(true);                                   // MATEVAL_S_DELTA
      } else {
        expect(c.singular).toBe(false);
        expect(o.sDelta).toBe(false);
        expect(o.sPj).toBeGreaterThan(0);
        expect(Number.isFinite(o.sPj)).toBe(true);
        expect(o.sPm).toBeGreaterThanOrEqual(o.sPj);
        expect(o.rS).toBeCloseTo(0.0038, 7);
        expect((o.flags & 16) !== 0).toBe(false);
      }
    });
    // just above the threshold NEE sees the S lobe at the sampled direction
    const q2: Query[] = [2, 3].map((ci) => ({ caseIdx: ci, V, L: out[ci].sL, u: [0.5, 0.5, 0.5, 0.5] }));
    const out2 = await probe(cases, q2);
    for (const o of out2) { expect(o.lobeS[3]).toBeGreaterThan(0); expect(Math.max(o.lobeS[0], o.lobeS[1], o.lobeS[2])).toBeGreaterThan(0); }
  });

  it('(h) V1 is reciprocal; V2 is non-reciprocal by design (q and Λ_S depend on μ_V)', async () => {
    const rnd = xorshift(5);
    const v1: MatParams = { model: 0, diffuse: [0.8, 0.5, 0.2], glossy: [0.9, 0.9, 0.9], roughness: 0.3, mix: 0.4 };
    const cases: Case[] = [{ p: v1, V: [0, 0, 1], label: 'v1', chi2: false }, { p: MATS.v2d(0.5), V: [0, 0, 1], label: 'v2', chi2: false }];
    const pairs: [V3, V3][] = [];
    for (let i = 0; i < 256; i++) pairs.push([f32(randHemi(rnd, 0.05)), f32(randHemi(rnd, 0.05))]);
    const qs: Query[] = [];
    for (const [a, b] of pairs) { qs.push({ caseIdx: 0, V: a, L: b, u: [0, 0, 0, 0] }, { caseIdx: 0, V: b, L: a, u: [0, 0, 0, 0] }); }
    const out = await probe(cases, qs);
    let worst = 0;
    pairs.forEach(([a, b], i) => {
      const f1 = out[2 * i].f, f2 = out[2 * i + 1].f;
      for (let k = 0; k < 3; k++) worst = Math.max(worst, relErr(f1[k] / b[2], f2[k] / a[2]));
    });
    console.log(`RECIPROCITY ${lane()} V1 max rel |f(V,L)/cosL − f(L,V)/cosV| = ${worst.toExponential(2)}`);
    expect(worst).toBeLessThan(1e-5);
    // V2: f_D(V,L)/f_D(L,V) = Λ_S(μ_V)/Λ_S(μ_L) (e.g. 0.9595/0.7963 = 1.2049 at μ_V = 1, μ_L = 0.1)
    const V: V3 = [0, 0, 1], L: V3 = f32([Math.sqrt(1 - 0.01), 0, 0.1]);
    const o2 = await probe(cases, [{ caseIdx: 1, V, L, u: [0, 0, 0, 0] }, { caseIdx: 1, V: L, L: V, u: [0, 0, 0, 0] }]);
    const ratio = (o2[0].lobeD[0] / L[2]) / (o2[1].lobeD[0] / V[2]);
    const want = prepare(MATS.v2d(0.5), 1).lambda / prepare(MATS.v2d(0.5), L[2]).lambda;
    expect(ratio).toBeCloseTo(want, 4);
    expect(ratio).toBeCloseTo(1.2049, 3);
  });
});

// ------------------------------------------------------------------------------------------------ material_eval

const ME_HARNESS = `
#include "material/material-eval.wgsl"
struct MQ { matId: u32, pad0: u32, pad1: u32, pad2: u32, uv: vec4f, color: vec4f, V: vec4f, ns: vec4f, ng: vec4f }
@group(0) @binding(1) var<storage, read> mq: array<MQ>;
@group(0) @binding(2) var<storage, read_write> mo: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&mq)) { return; }
  let q = mq[i];
  var h: SurfaceHit;
  h.ns = q.ns.xyz;
  h.ng = q.ng.xyz;
  h.uv = q.uv.xy;
  h.color = q.color;
  h.matId = q.matId;
  let m = material_eval(h, q.V.xyz);
  let o = i * 8u;
  mo[o] = vec4f(f32(m.model), m.metallic, m.roughness, m.ior);
  mo[o + 1u] = vec4f(m.base_color, m.specular_level);
  mo[o + 2u] = vec4f(m.specular_tint, m.transmission);
  mo[o + 3u] = vec4f(0.0, 0.0, 0.0, f32(m.flags));
  mo[o + 4u] = vec4f(m.base_color, m.metallic);             // V1 slots: Diffuse, mix
  mo[o + 5u] = vec4f(m.specular_tint, 0.0);                 // V1 slot: Glossy colour
  mo[o + 6u] = vec4f(m.ns, 0.0);
  mo[o + 7u] = vec4f(m.ng, 0.0);
}`;

describe(`material_eval (${lane()})`, () => {
  it('material table + textures (LOD 0, sRGB decoded after filtering) + COLOR_0 → MatEval; model / flags', async () => {
    const { device, features, wgslLanguageFeatures } = await getTestGpu();
    const px = (r: number, g: number, b: number, a = 255) => new Uint8Array([r, g, b, a]);
    const tex = (name: string, pixels: Uint8Array): TextureData => ({ name, width: 1, height: 1, pixels, wrapS: 'clamp-to-edge', wrapT: 'clamp-to-edge', filter: 'linear' });
    const textures = [tex('base', px(128, 64, 255)), tex('mr', px(0, 128, 191)), tex('spec', px(255, 255, 255, 153)), tex('emis', px(200, 100, 50))];
    const ref = (texture: number) => ({ texture, texCoord: 0 });
    const mat = (o: Partial<MaterialData>): MaterialData => ({
      name: 'm', baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 0.5, emissiveFactor: [0, 0, 0], emissiveStrength: 1, ior: 1.5,
      specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true, model: 'principled', ...o,
    });
    const materials: MaterialData[] = [
      mat({ baseColorFactor: [0.5, 1, 0.8, 1], baseColorTexture: ref(0), metallicFactor: 0.6, roughnessFactor: 0.9, metallicRoughnessTexture: ref(1),
        ior: 1.45, specularFactor: 1.5, specularTexture: ref(2), specularColorFactor: [1, 0.8, 0.6], emissiveFactor: [1, 2, 3], emissiveStrength: 2, emissiveTexture: ref(3) }),
      mat({ transmissionFactor: 0.5 }),
      mat({ model: 'v1', v1: { diffuse: [0.2, 0.3, 0.4], glossy: [0.5, 0.5, 0.5], roughness: 0.25, mix: 0.3 } }),
      mat({ specularFactor: 0 }),                     // η' = 1: no specular closure ⇒ diffuse-only
      mat({ model: 'v1', v1: { diffuse: [0.2, 0.3, 0.4], glossy: [0.5, 0.5, 0.5], roughness: 0.25, mix: 0 } }),
    ];
    const set = await createGpuTextures(device, { textures, materials }, { mode: 'validation' });
    const shader = composeWgsl('tests/me_harness.wgsl', {
      sources: { ...shaderSources, 'tests/me_harness.wgsl': ME_HARNESS },
      defines: { ...lutDefines({ declare: { group: 0, binding: 0 } }), SCENE_GROUP: 1, CUSTOM_ALPHA: false, VERTEX_FORMAT: 1, ...set.defines(1, SCENE_BINDING.textureBase) },
      features, wgslLanguageFeatures,
    });
    const module = await createCheckedShaderModule(device, shader, 'material-eval');
    const vis = GPUShaderStage.COMPUTE;
    const bgl0 = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: vis, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: vis, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: vis, buffer: { type: 'storage' } },
    ] });
    const bgl1 = device.createBindGroupLayout({ entries: [
      { binding: SCENE_BINDING.materials, visibility: vis, buffer: { type: 'read-only-storage' } },
      ...set.layoutEntries(SCENE_BINDING.textureBase, vis),
    ] });
    const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1] }), compute: { module, entryPoint: 'main' } });
    const Vz: V3 = [0, 0, 1];
    const qs = [
      { matId: 0, uv: [0.3, 0.7], color: [0.5, 0.25, 1, 1], V: norm([0.3, 0.1, 0.9]) },
      { matId: 1, uv: [0, 0], color: [1, 1, 1, 1], V: Vz },
      { matId: 2, uv: [0, 0], color: [0.1, 0.1, 0.1, 1], V: Vz },
      { matId: 3, uv: [0, 0], color: [1, 1, 1, 1], V: Vz },
      { matId: 4, uv: [0, 0], color: [1, 1, 1, 1], V: Vz },
    ];
    const qb = new ArrayBuffer(qs.length * 96);
    const dv = new DataView(qb);
    qs.forEach((q, i) => {
      const b = i * 96;
      dv.setUint32(b, q.matId, true);
      const w = (o: number, v: number[]) => v.forEach((x, k) => dv.setFloat32(b + o + 4 * k, x, true));
      w(16, q.uv); w(32, q.color); w(48, q.V); w(64, [0, 0, 1]); w(80, [0, 0, 1]);
    });
    const lut = createLutBuffer(device);
    const bQ = storage(device, qb, 'mq');
    const bO = storage(device, qs.length * 8 * 16, 'mo');
    const bM = storage(device, packMaterials(materials, set), 'materials');
    expect(bM.size).toBeGreaterThanOrEqual(materials.length * MATERIAL_LAYOUT.size);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: bgl0, entries: [{ binding: 0, resource: { buffer: lut } }, { binding: 1, resource: { buffer: bQ } }, { binding: 2, resource: { buffer: bO } }] }));
    pass.setBindGroup(1, device.createBindGroup({ layout: bgl1, entries: [{ binding: SCENE_BINDING.materials, resource: { buffer: bM } }, ...set.bindGroupEntries(SCENE_BINDING.textureBase)] }));
    pass.dispatchWorkgroups(1);
    pass.end();
    device.queue.submit([enc.finish()]);
    const o = new Float32Array(await readBuffer(device, bO, qs.length * 8 * 16));
    for (const b of [lut, bQ, bO, bM]) b.destroy();
    set.destroy();
    const s2l = (c: number) => (c < 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
    const row = (i: number, k: number) => Array.from(o.subarray((i * 8 + k) * 4, (i * 8 + k) * 4 + 4));
    const close = (a: number[], b: number[], tol = 2e-5) => b.forEach((x, k) => expect(Math.abs(a[k] - x), `${a} vs ${b}`).toBeLessThan(tol));
    // 0: textured Principled
    close(row(0, 0), [1, 0.6 * 191 / 255, 0.9 * 128 / 255, 1.45]);
    close(row(0, 1), [0.5 * s2l(128 / 255) * 0.5, 1 * s2l(64 / 255) * 0.25, 0.8 * 1 * 1, 0.5 * 1.5 * 153 / 255]);
    close(row(0, 2), [1, 0.8, 0.6, 0]);
    // (emission is not part of MatEval since M3b: the PT reads tri_emission(), covered by pt.gpu.test.ts (vii) textures)
    expect(row(0, 3)[3]).toBe(1 | 4 | 8);                       // non-delta, hasD, hasS
    // 1: transmission → model 2 (Principled + glass closure)
    expect(row(1, 0)[0]).toBe(2);
    expect(row(1, 2)[3]).toBeCloseTo(0.5, 6);
    // 2: V1 (COLOR_0 does not touch V1)
    expect(row(2, 0)[0]).toBe(0);
    expect(row(2, 0)[2]).toBeCloseTo(0.25, 6);
    close(row(2, 4), [0.2, 0.3, 0.4, 0.3]);
    close(row(2, 5).slice(0, 3), [0.5, 0.5, 0.5]);
    expect(row(2, 3)[3]).toBe(1 | 4 | 8);
    // 3: specular level 0 ⇒ η' = 1 ⇒ Lambert only ⇒ diffuse-only
    expect(row(3, 3)[3]).toBe(1 | 2 | 4);
    // 4: V1 mix 0 ⇒ diffuse-only
    expect(row(4, 3)[3]).toBe(1 | 2 | 4);
    close(row(0, 6).slice(0, 3), [0, 0, 1]);
    // the texture-less variant (TEX_ARRAYS = 0, e.g. an untextured scene) and an includer-declared records array compile too
    for (const [records, recordsKind] of [['array<f32>', 'f32'], ['array<vec4u>', 'vec4u']] as const) {
      const bare = composeWgsl('tests/me_bare.wgsl', {
        sources: { ...shaderSources, 'tests/me_bare.wgsl': `@group(0) @binding(0) var<storage, read> records: ${records};\n` + ME_HARNESS },
        defines: { ...lutDefines({ base: 64, recordsKind }), SCENE_GROUP: 1, CUSTOM_ALPHA: false, VERTEX_FORMAT: 1, TEX_GROUP: 1, TEX_BINDING_BASE: 8, TEX_ARRAYS: 0, TEX_SAMPLERS: 0 },
        features, wgslLanguageFeatures,
      });
      await createCheckedShaderModule(device, bare, `material-eval-bare-${recordsKind}`);
    }
  });
});

// BSDF library: V1 validation BSDF and V2 Cycles Principled Tier 1 (contract: docs/decisions/bsdf-api.md).
// Normative math: math.md#bsdf-v1, #bsdf-v2, #pdf-conventions, #support-indicator, #lobe-codes;
// derivations and Cycles line references: docs/research/gap-bsdf-lobe-model.md §3–§7, cycles-conventions.md §4.
//
// Conventions (Cycles): V points toward the previous (camera-side) vertex, L toward the next vertex / the light.
// "eval" / f_cos = f_s(V, L)·|Ns·L| (cosine included). Every public function re-applies the two-sided flip for the
// EVALUATING V (Ng·V < 0 ⇒ Ng, Ns negated; geom/shader_data.h:108-116), so a MatEval built for one V can be
// evaluated under another (shifts: V = normalize(y_{k−1} − x_k)); for the V it was built with this is a no-op.
//
// Lobe classes: D = Lambert; S = GGX reflection (V1 glossy, or V2 metal ⊕ dielectric specular — they share Ns, α
// and the VNDF sampler, so merging them is exact: gap-bsdf §6.4). G_R/G_T (glass) arrive in M3b.
// pdf conventions (math.md#pdf-conventions): throughput uses the JOINT pdf q(ℓ|V)·p_ℓ(L|V); MIS and footprints use
// the MARGINAL Σ_{non-delta ℓ} q(ℓ|V)·p_ℓ(L|V). Never mix them.
// f (eval) is Cycles' eval: lobe normal only, NO Ng test. Every pdf returned here is the TRUE sampler density
// (plan §1.5 "support-consistent pdfs"): each lobe's density carries its sampler's Ng rejection (D: Ng·L > 0,
// S: Ng·L ≥ 0). With flat shading (Ns = Ng) this is identical to Cycles' eval pdf; under smooth shading it keeps
// NEE/BSDF MIS a partition of unity where Cycles' is not (gap-bsdf §8.2).
//
// Requires the LUT defines of material/lut.wgsl (src/core/render/luts/lut-layout.ts lutDefines()).
#include "common/math.wgsl"
#include "common/nan.wgsl"
#include "material/lut.wgsl"
#include "material/fresnel.wgsl"
#include "material/lambert.wgsl"
#include "material/ggx.wgsl"
#include "material/v1.wgsl"
#include "material/v2.wgsl"

// ------------------------------------------------------------------------------------------------ contract types

// Lobe codes (math.md#lobe-codes / plan §1.5): 3 bits + delta flag carried separately.
const LOBE_D: u32 = 0u;   const LOBE_S: u32 = 1u;   const LOBE_GR: u32 = 2u;   const LOBE_GT: u32 = 3u;
const LOBE_NEE: u32 = 4u; const LOBE_NONE: u32 = 5u;

const BSDF_MODEL_V1: u32 = 0u;      // Diffuse + Glossy(GGX, F ≡ 1) + constant Mix (+ Emission)
const BSDF_MODEL_V2: u32 = 1u;      // Principled, distribution GGX, Tier 1
const BSDF_MODEL_GLASS: u32 = 2u;   // glass-capable V2 (M3b). STUB: currently identical to V2 with transmission = 0.

// MatEval.flags. Bits 0/1 are the contract; bits 2–4 are informational extensions (hasD / hasS / S singular),
// all evaluated for the V the MatEval was built with (material_eval).
const MATEVAL_HAS_NON_DELTA: u32 = 1u;
const MATEVAL_DIFFUSE_ONLY: u32 = 2u;
const MATEVAL_HAS_D: u32 = 4u;
const MATEVAL_HAS_S: u32 = 8u;
const MATEVAL_S_DELTA: u32 = 16u;

const BSDF_WEIGHT_CUTOFF: f32 = 1e-5;   // CLOSURE_WEIGHT_CUTOFF (svm/types.h:543); part of the BSDF definition
const BSDF_SW_GUARD: f32 = 1e-12;      // allocated lobe with sample weight 0 keeps q > 0 (gap-bsdf §4.3 (a))

// Everything the BSDF needs at a hit, built once per vertex by material_eval(...) in material/material-eval.wgsl.
struct MatEval {
  model: u32,            // 0 = V1 validation, 1 = V2 Principled (Tier 1), 2 = glass-capable V2 (M3b)
  base_color: vec3f,     // after texture × factor × COLOR_0 (linear)
  metallic: f32,
  roughness: f32,        // perceptual r (GGX α = r²); V1: glossy roughness
  ior: f32,
  specular_level: f32,   // Principled "Specular IOR Level" (0.5 neutral)
  specular_tint: vec3f,
  transmission: f32,     // M3b
  emission: vec3f,       // L_e (two-sided), radiance units
  v1_diffuse: vec3f,     // V1 only
  v1_glossy: vec3f,      // V1 only
  v1_mix: f32,           // V1 only: (1-mix)·Diffuse + mix·Glossy
  ns: vec3f,             // shading normal, flipped to the V side (two-sided shading frame)
  ng: vec3f,             // geometric normal, flipped to the V side
  flags: u32,            // bit0: has non-delta lobe; bit1: diffuse-only (Lambert is the only allocated lobe)
}

struct BsdfEval {       // all-lobe evaluation (NEE) — Cycles eval semantics
  f_cos: vec3f,
  pdf_marginal: f32,     // TRUE sampler density of the one-sample mixture (support-consistent), solid angle
}

struct BsdfSample {
  L: vec3f,              // set whenever a direction was generated, also for rejected samples (diagnostics)
  weight: vec3f,         // f_ℓ·cos / (q(ℓ|V)·p_ℓ(L)) using the JOINT pdf (lobe-indexed throughput); delta: F/q(S)
  pdf_joint: f32,        // q(ℓ|V)·p_ℓ(L); 0 for delta
  pdf_marginal: f32,     // Σ_ℓ q(ℓ|V)·p_ℓ(L) over non-delta lobes (MIS, footprints); 0 for delta
  lobe: u32,             // LOBE_*; the class that was picked (LOBE_NONE if no lobe is allocated)
  is_delta: bool,
  valid: bool,           // false if the sampler rejected (e.g. Ng·L on the wrong side, TIR in M3b)
}

// ------------------------------------------------------------------------------------------------ closure context

// A MatEval prepared for one V: the Cycles closures (weights, Fresnel parameters, sample weights) at that V.
struct BsdfCtx {
  model: u32,
  ns: vec3f,
  ng: vec3f,
  alpha: f32,          // α = r²
  a2: f32,             // α_x·α_y
  rough: f32,          // sqrt(sqrt(α_x α_y)) (= r): LUT x axis
  singular: bool,      // !(a2 > 2e-10): S is a delta lobe
  has_d: bool,
  has_s: bool,
  has_metal: bool,     // V2
  has_spec: bool,      // V2
  w_d: vec3f,          // Lambert closure weight (V2: C·(1−m)·Λ_S(μ_V))
  w_g: vec3f,          // V1 glossy closure weight (F ≡ 1)
  w_m: f32,            // V2 metal closure weight (grey)
  f0_m: vec3f,
  b_m: vec3f,
  w_s: f32,            // V2 dielectric specular closure weight (grey)
  f0_s: vec3f,
  eta_s: f32,          // η'
  sw_d: f32,           // Cycles sample weights
  sw_s: f32,
  q_d: f32,            // q(D|V)
  q_s: f32,            // q(S|V)
}

fn bsdf_prepare(m: MatEval, V: vec3f) -> BsdfCtx {
  var c: BsdfCtx;
  c.model = m.model;
  let back = dot(m.ng, V) < 0.0;
  c.ng = select(m.ng, -m.ng, back);
  c.ns = select(m.ns, -m.ns, back);
  let r = saturate(m.roughness);
  c.alpha = r * r;
  c.a2 = c.alpha * c.alpha;
  c.rough = sqrt(sqrt(c.a2));
  c.singular = !(c.a2 > BSDF_ROUGHNESS_SQ_THRESH);
  if (m.model == BSDF_MODEL_V1) {
    c = bsdf_setup_v1(m, c);
  } else {
    // BSDF_MODEL_GLASS (2) is a stub for M3b: V2 with transmission = 0.
    c = bsdf_setup_v2(m, c, dot(V, c.ns));
  }
  var swd = c.sw_d;
  var sws = c.sw_s;
  if (c.has_d) { swd = max(swd, BSDF_SW_GUARD); } else { swd = 0.0; }
  if (c.has_s) { sws = max(sws, BSDF_SW_GUARD); } else { sws = 0.0; }
  let sum = swd + sws;
  if (sum > 0.0) {
    c.q_d = swd / sum;
    c.q_s = sws / sum;
  }
  return c;
}

/// Fresnel-weighted closure weight of class S at the microfacet (V·H = cos_hi).
fn bsdf_F_S(c: BsdfCtx, cos_hi: f32) -> vec3f {
  if (c.model == BSDF_MODEL_V1) { return c.w_g; }
  var F = vec3f(0.0);
  if (c.has_metal) { F += c.w_m * fresnel_f82(cos_hi, c.f0_m, c.b_m); }
  if (c.has_spec) { F += c.w_s * fresnel_gen_schlick_ior(cos_hi, c.eta_s, c.f0_s); }
  return F;
}

struct LobeEvals {
  f_d: vec3f,
  f_s: vec3f,
  p_d: f32,    // q(D|V)·p_D(L)   (joint)
  p_s: f32,    // q(S|V)·p_S(L|V) (joint; 0 if S is delta)
}

/// Per-class Cycles eval (no Ng test on f) and joint pdfs (true sampler densities: with the samplers' Ng tests).
fn bsdf_eval_ctx(c: BsdfCtx, V: vec3f, L: vec3f) -> LobeEvals {
  var e: LobeEvals;
  let cos_ng = dot(c.ng, L);
  if (c.has_d) {
    let k = lambert_eval(c.ns, L);
    e.f_d = c.w_d * k;
    e.p_d = select(0.0, c.q_d * k, cos_ng > 0.0);            // bsdf_diffuse.h:57-63
  }
  if (c.has_s && !c.singular) {
    let g = ggx_refl_eval(c.a2, c.ns, V, L);
    if (g.valid) {
      e.f_s = bsdf_F_S(c, g.cos_hi) * g.g;
      e.p_s = select(0.0, c.q_s * g.pdf, cos_ng >= 0.0);     // bsdf_microfacet.h:757-761
    }
  }
  return e;
}

// ------------------------------------------------------------------------------------------------ public API

/// NEE: all lobes, Cycles eval (no Ng test on L, no support indicator on f); pdf = marginal over non-delta lobes
/// (true sampler density, i.e. with the samplers' Ng tests).
fn bsdf_eval(m: MatEval, V: vec3f, L: vec3f) -> BsdfEval {
  let c = bsdf_prepare(m, V);
  let e = bsdf_eval_ctx(c, V, L);
  var r: BsdfEval;
  r.f_cos = e.f_d + e.f_s;
  r.pdf_marginal = e.p_d + e.p_s;
  return r;
}

/// MIS pdf of the BSDF technique for a BSDF-hit emitter: Σ_{non-delta ℓ} q(ℓ|V)·p_ℓ(L|V).
fn bsdf_pdf_marginal(m: MatEval, V: vec3f, L: vec3f) -> f32 {
  let c = bsdf_prepare(m, V);
  let e = bsdf_eval_ctx(c, V, L);
  return e.p_d + e.p_s;
}

/// Shifts: (f_ℓ·cos, joint pdf q(ℓ|V)·p_ℓ(L|V)) of one lobe. LOBE_NEE: all-lobe f with pdf 1 (p_k := 1 in the PSS
/// Jacobian, math.md#jacobian); LOBE_NONE: (0, 1); a delta S or an unallocated lobe: (0, 0); G_R/G_T: (0, 0) until M3b.
/// f carries no support indicator (multiply by bsdf_sample_support() for BSDF-sampled segments); the pdf is the true
/// sampler density (already 0 where the lobe's sampler rejects on Ng).
fn bsdf_eval_lobe(m: MatEval, V: vec3f, L: vec3f, lobe: u32) -> vec4f {
  if (lobe == LOBE_NONE) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let c = bsdf_prepare(m, V);
  let e = bsdf_eval_ctx(c, V, L);
  switch lobe {
    case LOBE_D: { return vec4f(e.f_d, e.p_d); }
    case LOBE_S: { return vec4f(e.f_s, e.p_s); }
    case LOBE_NEE: { return vec4f(e.f_d + e.f_s, 1.0); }
    default: { return vec4f(0.0); }   // TODO(M3b): G_R / G_T
  }
}

/// Sampler-support indicator 1_supp(ℓ, V, L) (math.md#support-indicator), after the two-sided flip for this V.
fn bsdf_sample_support(m: MatEval, V: vec3f, L: vec3f, lobe: u32) -> bool {
  let back = dot(m.ng, V) < 0.0;
  let ng = select(m.ng, -m.ng, back);
  let ns = select(m.ns, -m.ns, back);
  switch lobe {
    case LOBE_D: { return dot(ng, L) > 0.0; }
    case LOBE_S, LOBE_GR: { return dot(ns, V) > 0.0 && dot(ng, L) >= 0.0 && dot(ns, L) >= 0.0; }
    case LOBE_GT: { return false; }   // TODO(M3b): Ng·L < 0 ∧ Ns·L < 0 ∧ Hn·V > 0 ∧ Hn·L < 0 ∧ no TIR ∧ |η−1| ≥ 1e-4
    default: { return true; }         // NEE / NONE: no indicator (Cycles' NEE has no Ng test)
  }
}

/// Per-lobe perceptual roughness for the reconnection predicate R_k (math.md#lobe-codes; gap-bsdf §7.4).
/// NEE uses the hasD/hasS bits of MatEval.flags (set by material_eval for the path's V).
fn lobe_roughness(m: MatEval, lobe: u32) -> f32 {
  let r = saturate(m.roughness);
  let a = r * r;
  let rS = select(r, 0.0, !(a * a > BSDF_ROUGHNESS_SQ_THRESH));
  switch lobe {
    case LOBE_D: { return 1.0; }
    case LOBE_S, LOBE_GR, LOBE_GT: { return rS; }
    case LOBE_NEE: {
      let rd = select(0.0, 1.0, (m.flags & MATEVAL_HAS_D) != 0u);
      let rs = select(0.0, rS, (m.flags & MATEVAL_HAS_S) != 0u);
      return max(rd, rs);
    }
    default: { return FLT_MAX; }      // light / env vertex: always rough (never IEEE Inf, plan §1.7)
  }
}

/// One-sample mixture sampler. u = (u_lobe, u_h1, u_h2, u_rt) (math.md#rng-layout; u_rt is reserved for glass R/T and
/// unused by V1/V2, but all four dims are always consumed by the caller). Rejections follow Cycles exactly:
/// D needs Ng·L > 0; S needs Ns·V > 0, Ng·L ≥ 0, Ns·L ≥ 0. A delta S returns weight F(Ns·V)/q(S) with pdfs 0.
fn bsdf_sample(m: MatEval, V: vec3f, u: vec4f) -> BsdfSample {
  var s: BsdfSample;
  s.lobe = LOBE_NONE;
  let c = bsdf_prepare(m, V);
  if (!(c.q_d + c.q_s > 0.0)) { return s; }
  if (u.x < c.q_d) {
    s.lobe = LOBE_D;
    s.L = lambert_sample_dir(c.ns, u.yz);
    if (!(dot(c.ng, s.L) > 0.0)) { return s; }                        // bsdf_diffuse.h:57-63
  } else {
    s.lobe = LOBE_S;
    let cos_ni = dot(c.ns, V);
    if (!(cos_ni > 0.0)) { return s; }                                 // bsdf_microfacet.h:693-696
    let H = ggx_sample_h(c.alpha, c.singular, c.ns, V, u.yz);
    let cos_hi = dot(H, V);
    s.L = 2.0 * cos_hi * H - V;
    if (dot(c.ng, s.L) < 0.0 || dot(c.ns, s.L) < 0.0) { return s; }   // bsdf_microfacet.h:757-761
    if (c.singular) {
      // Cycles: eval = F·1e6, pdf = 1e6 (·lobe_prob) ⇒ weight F/q(S) (cyc-verify C2); no density.
      s.is_delta = true;
      s.weight = bsdf_F_S(c, cos_hi) / c.q_s;
      s.valid = true;
      return s;
    }
  }
  let e = bsdf_eval_ctx(c, V, s.L);
  s.pdf_marginal = e.p_d + e.p_s;
  var f = e.f_s;
  s.pdf_joint = e.p_s;
  if (s.lobe == LOBE_D) {
    f = e.f_d;
    s.pdf_joint = e.p_d;
  }
  if (!(s.pdf_joint > 0.0)) { return s; }
  s.weight = f / s.pdf_joint;
  s.valid = true;
  return s;
}

// ------------------------------------------------------------------------------------------------ extensions

/// Lobe-class pmf at V: (q(D|V), q(S|V), q(G|V) (= 0 until M3b), 0). Debug views, tests (gap-bsdf U-6).
fn bsdf_lobe_probs(m: MatEval, V: vec3f) -> vec4f {
  let c = bsdf_prepare(m, V);
  return vec4f(c.q_d, c.q_s, 0.0, 0.0);
}

/// MatEval.flags for view direction V (material_eval calls this once per vertex).
fn bsdf_flags(m: MatEval, V: vec3f) -> u32 {
  let c = bsdf_prepare(m, V);
  var f = 0u;
  if (c.has_d || (c.has_s && !c.singular)) { f |= MATEVAL_HAS_NON_DELTA; }
  if (c.has_d && !c.has_s) { f |= MATEVAL_DIFFUSE_ONLY; }
  if (c.has_d) { f |= MATEVAL_HAS_D; }
  if (c.has_s) { f |= MATEVAL_HAS_S; }
  if (c.has_s && c.singular) { f |= MATEVAL_S_DELTA; }
  return f;
}

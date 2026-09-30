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
// and the VNDF sampler, so merging them is exact: gap-bsdf §6.4); G = the Cycles GGX glass closure (glass.wgsl,
// math.md#glass) with the sub-events G_R (reflection, Ns·L ≥ 0) and G_T (transmission, Ns·L < 0).
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
#include "material/glass.wgsl"

// ------------------------------------------------------------------------------------------------ contract types

// Lobe codes (math.md#lobe-codes / plan §1.5): 3 bits + delta flag carried separately.
const LOBE_D: u32 = 0u;   const LOBE_S: u32 = 1u;   const LOBE_GR: u32 = 2u;   const LOBE_GT: u32 = 3u;
const LOBE_NEE: u32 = 4u; const LOBE_NONE: u32 = 5u;

const BSDF_MODEL_V1: u32 = 0u;      // Diffuse + Glossy(GGX, F ≡ 1) + constant Mix (+ Emission)
const BSDF_MODEL_V2: u32 = 1u;      // Principled, distribution GGX, Tier 1
const BSDF_MODEL_GLASS: u32 = 2u;   // glass-capable V2: Principled with Transmission Weight > 1e-5 (lobe class G)
const BSDF_MODEL_GLASS_NODE: u32 = 3u;       // Cycles Glass BSDF node (Color, Roughness, IOR)
const BSDF_MODEL_REFRACTION_NODE: u32 = 4u;  // Cycles Refraction BSDF node (Color, Roughness, IOR)

// MatEval.flags. Bits 0/1 are the contract; bits 2–6 are informational extensions (hasD / hasS / S singular / hasG /
// G singular), all evaluated for the V the MatEval was built with (material_eval). Bit 7 is an INPUT: the MatEval
// normals are the NEGATED winding normal (the building ray hit the back side), so that the glass η_side can be
// recomputed for any evaluating V (η_side = 1/ior iff V is on the backfacing side of the winding normal).
const MATEVAL_HAS_NON_DELTA: u32 = 1u;
const MATEVAL_DIFFUSE_ONLY: u32 = 2u;
const MATEVAL_HAS_D: u32 = 4u;
const MATEVAL_HAS_S: u32 = 8u;
const MATEVAL_S_DELTA: u32 = 16u;
const MATEVAL_HAS_G: u32 = 32u;
const MATEVAL_G_DELTA: u32 = 64u;
const MATEVAL_BACKFACING: u32 = 128u;

const BSDF_WEIGHT_CUTOFF: f32 = 1e-5;   // CLOSURE_WEIGHT_CUTOFF (svm/types.h:543); part of the BSDF definition
const BSDF_SW_GUARD: f32 = 1e-12;      // allocated lobe with sample weight 0 keeps q > 0 (gap-bsdf §4.3 (a))

// Everything the BSDF needs at a hit, built once per vertex by material_eval(...) in material/material-eval.wgsl.
// Kept small for the same reason as BsdfCtx (the Metal size cliff): V1 reuses the V2 slots (diffuse → base_color,
// glossy colour → specular_tint, mix → metallic), and emission is not part of it (the PT reads tri_emission()).
struct MatEval {
  model: u32,            // 0 = V1 validation, 1 = V2 Principled (Tier 1), 2 = V2 with transmission (glass),
                         // 3 = Glass node, 4 = Refraction node (base_color = Color)
  base_color: vec3f,     // V2: after texture × factor × COLOR_0 (linear) | V1: Lambert albedo | Glass/Refraction: Colour
  metallic: f32,         // V2 metallic | V1: mix factor ((1−mix)·Diffuse + mix·Glossy)
  roughness: f32,        // perceptual r (GGX α = r²); V1: glossy roughness
  ior: f32,
  specular_level: f32,   // Principled "Specular IOR Level" (0.5 neutral)
  specular_tint: vec3f,  // V2 Specular Tint | V1: glossy colour
  transmission: f32,     // Principled Transmission Weight (model 2)
  ns: vec3f,             // shading normal, flipped to the V side (two-sided shading frame)
  ng: vec3f,             // geometric normal, flipped to the V side
  flags: u32,            // bit0: has non-delta lobe; bit1: diffuse-only (Lambert is the only allocated lobe); MATEVAL_*
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
  valid: bool,           // false if the sampler rejected (e.g. Ng·L on the wrong side, no Fresnel energy)
}

// ------------------------------------------------------------------------------------------------ closure context

// A MatEval prepared for one V: the Cycles closures (weights, Fresnel parameters, lobe pmf) at that V.
// COMPACT on purpose (8 × 16 B): the BSDF library is inlined at every call site, and the Apple Metal compiler
// (macOS 26, M5 Pro; Chrome 154 and dawn.node) silently MISCOMPILES kernels whose per-thread state crosses a size
// threshold (wrong values / no writes, no error): the M3a BSDF harness broke with ~4 extra vec3 in this struct.
// So: booleans are bits, α² / roughness / F0(η) / the glass delta flags are derived, and fields shared by
// mutually exclusive models are unions (docs/decisions/bsdf-api.md "BsdfCtx size").
struct BsdfCtx {
  ns: vec3f,           // shading normal, flipped to V's side
  alpha: f32,          // α = r²  (a2 = α·α: bctx_a2; LUT roughness √α: bctx_rough)
  ng: vec3f,           // geometric normal, flipped to V's side
  bits: u32,           // BC_*
  w_d: vec3f,          // Lambert closure weight (V2: C·(1−m)(1−t)·Λ_S(μ_V))
  q_d: f32,            // q(D|V) (holds the raw sample weight sw_D during setup)
  s_a: vec3f,          // class S: V1 glossy closure weight (F ≡ 1) | V2 metal F82 F0 (= Cc)
  q_s: f32,            // q(S|V) (sw_S during setup)
  s_b: vec3f,          // V2 metal F82-tint B term
  w_m: f32,            // V2 metal closure weight (grey)
  f0_s: vec3f,         // V2 dielectric specular generalized-Schlick f0 (saturated)
  w_s: f32,            // V2 dielectric specular closure weight (grey)
  g_f0: vec3f,         // glass generalized-Schlick f0 (saturated)
  eta_s: f32,          // V2 dielectric specular η'
  g_tt: vec3f,         // glass transmission tint (Principled √Cc, Glass node Colour, Refraction node Colour);
                       // the reflection tint is g_tt when BC_G_RT_TT (Glass node), else 1
  q_g: f32,            // q(G|V) (sw_G during setup)
  g_w: f32,            // glass closure weight (grey; Principled t·(1−m), Glass / Refraction node 1)
  g_eta: f32,          // η_side (relative IOR seen from V's side)
  sw_sum: f32,         // Σ guarded Cycles sample weights (tests: sw_ℓ = q_ℓ·sw_sum)
  model: u32,
}

const BC_SINGULAR: u32 = 1u;     // !(α² > 2e-10): S and G are delta (G_T also when |η_side − 1| < 1e-4)
const BC_HAS_D: u32 = 2u;
const BC_HAS_S: u32 = 4u;
const BC_HAS_METAL: u32 = 8u;    // V2
const BC_HAS_SPEC: u32 = 16u;    // V2
const BC_BACK: u32 = 32u;        // the evaluating V is on the backfacing side of the winding normal (η_side = 1/ior)
const BC_HAS_G: u32 = 64u;
const BC_G_HAS_R: u32 = 128u;    // false for the Refraction node
const BC_G_FNONE: u32 = 256u;    // Refraction node: Fresnel NONE
const BC_G_RT_TT: u32 = 512u;    // Glass node: reflection tint = transmission tint = Colour

fn bctx_has(c: BsdfCtx, bit: u32) -> bool { return (c.bits & bit) != 0u; }
fn bctx_a2(c: BsdfCtx) -> f32 { return c.alpha * c.alpha; }
fn bctx_rough(c: BsdfCtx) -> f32 { return sqrt(sqrt(bctx_a2(c))); }   // sqrt(sqrt(α_x α_y)) (= r): LUT x axis
fn bctx_singular(c: BsdfCtx) -> bool { return bctx_has(c, BC_SINGULAR); }

fn bsdf_prepare(m: MatEval, V: vec3f) -> BsdfCtx {
  var c: BsdfCtx;
  c.model = m.model;
  let flip = dot(m.ng, V) < 0.0;
  c.ng = select(m.ng, -m.ng, flip);
  c.ns = select(m.ns, -m.ns, flip);
  if (flip != ((m.flags & MATEVAL_BACKFACING) != 0u)) { c.bits |= BC_BACK; }   // Cycles SD_BACKFACING for this V (triangle.h:39-42)
  let r = saturate(m.roughness);
  c.alpha = r * r;
  if (!(bctx_a2(c) > BSDF_ROUGHNESS_SQ_THRESH)) { c.bits |= BC_SINGULAR; }
  let mu = dot(V, c.ns);
  if (m.model == BSDF_MODEL_V1) {
    c = bsdf_setup_v1(m, c);
  } else if (m.model == BSDF_MODEL_GLASS_NODE) {
    c = bsdf_setup_glass_node(m, c, mu);
  } else if (m.model == BSDF_MODEL_REFRACTION_NODE) {
    c = bsdf_setup_refraction_node(m, c);
  } else {
    c = bsdf_setup_v2(m, c, mu);            // model 1, and model 2 (+ glass closure, v2.wgsl step 2)
  }
  // the setups left the raw Cycles sample weights in q_*: guard allocated lobes (q > 0), then normalise
  let swd = select(0.0, max(c.q_d, BSDF_SW_GUARD), bctx_has(c, BC_HAS_D));
  let sws = select(0.0, max(c.q_s, BSDF_SW_GUARD), bctx_has(c, BC_HAS_S));
  let swg = select(0.0, max(c.q_g, BSDF_SW_GUARD), bctx_has(c, BC_HAS_G));
  let sum = swd + sws + swg;
  c.sw_sum = sum;
  c.q_d = 0.0;
  c.q_s = 0.0;
  c.q_g = 0.0;
  if (sum > 0.0) {
    c.q_d = swd / sum;
    c.q_s = sws / sum;
    c.q_g = swg / sum;
  }
  return c;
}

/// Fresnel-weighted closure weight of class S at the microfacet (V·H = cos_hi).
fn bsdf_F_S(c: BsdfCtx, cos_hi: f32) -> vec3f {
  if (c.model == BSDF_MODEL_V1) { return c.s_a; }
  var F = vec3f(0.0);
  if (bctx_has(c, BC_HAS_METAL)) { F += c.w_m * fresnel_f82(cos_hi, c.s_a, c.s_b); }
  if (bctx_has(c, BC_HAS_SPEC)) { F += c.w_s * fresnel_gen_schlick_ior(cos_hi, c.eta_s, c.f0_s); }
  return F;
}

struct LobeEvals {
  f_d: vec3f,
  f_s: vec3f,
  f_g: vec3f,  // glass (Cycles eval: spurious refraction region included)
  p_d: f32,    // q(D|V)·p_D(L)   (joint)
  p_s: f32,    // q(S|V)·p_S(L|V) (joint; 0 if S is delta)
  p_g: f32,    // q(G|V)·p_G(L|V) (joint, valid-only = true sampler density; 0 if G is delta)
  g_side: u32, // bit0: G sub-event of L is G_T (Ns·L < 0); bit1: (V, L) in the G sampler support
}

/// Per-class Cycles eval (no Ng test on f) and joint pdfs (true sampler densities: with the samplers' Ng tests).
fn bsdf_eval_ctx(c: BsdfCtx, V: vec3f, L: vec3f) -> LobeEvals {
  var e: LobeEvals;
  let cos_ng = dot(c.ng, L);
  if (bctx_has(c, BC_HAS_D)) {
    let k = lambert_eval(c.ns, L);
    e.f_d = c.w_d * k;
    e.p_d = select(0.0, c.q_d * k, cos_ng > 0.0);            // bsdf_diffuse.h:57-63
  }
  if (bctx_has(c, BC_HAS_S) && !bctx_singular(c)) {
    let g = ggx_refl_eval(bctx_a2(c), c.ns, V, L);
    if (g.valid) {
      e.f_s = bsdf_F_S(c, g.cos_hi) * g.g;
      e.p_s = select(0.0, c.q_s * g.pdf, cos_ng >= 0.0);     // bsdf_microfacet.h:757-761
    }
  }
  if (bctx_has(c, BC_HAS_G)) {
    let g = glass_eval(c, V, L);
    e.f_g = g.f;
    e.p_g = c.q_g * g.pdfV;
    e.g_side = select(0u, 1u, g.isT) | select(0u, 2u, g.valid);
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
  r.f_cos = e.f_d + e.f_s + e.f_g;
  r.pdf_marginal = e.p_d + e.p_s + e.p_g;
  return r;
}

/// MIS pdf of the BSDF technique for a BSDF-hit emitter: Σ_{non-delta ℓ} q(ℓ|V)·p_ℓ(L|V).
fn bsdf_pdf_marginal(m: MatEval, V: vec3f, L: vec3f) -> f32 {
  let c = bsdf_prepare(m, V);
  let e = bsdf_eval_ctx(c, V, L);
  return e.p_d + e.p_s + e.p_g;
}

/// Shifts: (f_ℓ·cos, joint pdf q(ℓ|V)·p_ℓ(L|V)) of one lobe. LOBE_NEE: all-lobe f with pdf 1 (p_k := 1 in the PSS
/// Jacobian, math.md#jacobian); LOBE_NONE: (0, 1); a delta lobe or an unallocated lobe: (0, 0). G_R / G_T: the glass
/// eval restricted to its sub-event side (G_R: Ns·L ≥ 0, G_T: Ns·L < 0) with the valid-only joint pdf.
/// f carries no support indicator (multiply by bsdf_sample_support() for BSDF-sampled segments); the pdf is the true
/// sampler density (already 0 where the lobe's sampler rejects on Ng, and outside the glass support).
fn bsdf_eval_lobe(m: MatEval, V: vec3f, L: vec3f, lobe: u32) -> vec4f {
  if (lobe == LOBE_NONE) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let c = bsdf_prepare(m, V);
  let e = bsdf_eval_ctx(c, V, L);
  switch lobe {
    case LOBE_D: { return vec4f(e.f_d, e.p_d); }
    case LOBE_S: { return vec4f(e.f_s, e.p_s); }
    case LOBE_GR: { return select(vec4f(0.0), vec4f(e.f_g, e.p_g), (e.g_side & 1u) == 0u); }
    case LOBE_GT: { return select(vec4f(0.0), vec4f(e.f_g, e.p_g), (e.g_side & 1u) != 0u); }
    case LOBE_NEE: { return vec4f(e.f_d + e.f_s + e.f_g, 1.0); }
    default: { return vec4f(0.0); }
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
    case LOBE_GT: {                   // Ns·V > 0 ∧ Ng·L < 0 ∧ Ns·L < 0 ∧ Hn·V > 0 ∧ Hn·L < 0 ∧ no TIR ∧ |η−1| ≥ 1e-4
      let c = bsdf_prepare(m, V);
      if (!bctx_has(c, BC_HAS_G) || bctx_singular(c)) { return false; }
      let g = glass_eval(c, V, L);
      return g.isT && g.valid;
    }
    default: { return true; }         // NEE / NONE: no indicator (Cycles' NEE has no Ng test)
  }
}

/// ReSTIR query (restir-api.md §3.1; docs/decisions/bsdf-api.md "bsdf_query"): everything a path-tree vertex, a
/// replayed vertex or a reconnection needs for one (V, L) from ONE bsdf_prepare + ONE bsdf_eval_ctx (the Metal
/// state-size cliff and the per-pipeline inline budget, restir-api.md §4.5):
///   f_lobe  = bsdf_eval_lobe(m, V, L, lobe).rgb     (LOBE_NEE: all lobes; no support factor)
///   p_joint = bsdf_eval_lobe(m, V, L, lobe).a       (LOBE_NEE / LOBE_NONE: 1; delta or unallocated lobe: 0)
///   f_all   = bsdf_eval(m, V, L).f_cos,   p_marg = bsdf_eval(m, V, L).pdf_marginal (= bsdf_pdf_marginal)
///   supp    = bsdf_sample_support(m, V, L, lobe)   (true for LOBE_NEE / LOBE_NONE)
/// Every field equals the public function for the same inputs (U-PT-BITS part 2).
struct BsdfQuery {
  f_lobe: vec3f,
  p_joint: f32,
  f_all: vec3f,
  p_marg: f32,
  supp: bool,
}

fn bsdf_query(m: MatEval, V: vec3f, L: vec3f, lobe: u32) -> BsdfQuery {
  var q: BsdfQuery;
  let c = bsdf_prepare(m, V);
  let e = bsdf_eval_ctx(c, V, L);
  q.f_all = e.f_d + e.f_s + e.f_g;
  q.p_marg = e.p_d + e.p_s + e.p_g;
  q.supp = true;
  let suppS = dot(c.ns, V) > 0.0 && dot(c.ng, L) >= 0.0 && dot(c.ns, L) >= 0.0;
  let isT = (e.g_side & 1u) != 0u;
  switch lobe {
    case LOBE_D: { q.f_lobe = e.f_d; q.p_joint = e.p_d; q.supp = dot(c.ng, L) > 0.0; }
    case LOBE_S: { q.f_lobe = e.f_s; q.p_joint = e.p_s; q.supp = suppS; }
    case LOBE_GR: {
      if (!isT) { q.f_lobe = e.f_g; q.p_joint = e.p_g; }
      q.supp = suppS;
    }
    case LOBE_GT: {
      if (isT) { q.f_lobe = e.f_g; q.p_joint = e.p_g; }
      q.supp = bctx_has(c, BC_HAS_G) && !bctx_singular(c) && (e.g_side & 3u) == 3u;
    }
    case LOBE_NEE: { q.f_lobe = q.f_all; q.p_joint = 1.0; }
    case LOBE_NONE: { q.p_joint = 1.0; }
    default: { }
  }
  return q;
}

/// Per-lobe perceptual roughness for the reconnection predicate R_k (math.md#lobe-codes; gap-bsdf §7.4; glass §5.4).
/// NEE uses the hasD/hasS/hasG bits of MatEval.flags (set by material_eval for the path's V). G_T is also delta
/// (r = 0) when |η_side − 1| < 1e-4, with η_side for the V the MatEval was built with (MATEVAL_BACKFACING).
fn lobe_roughness(m: MatEval, lobe: u32) -> f32 {
  let r = saturate(m.roughness);
  let a = r * r;
  let rS = select(r, 0.0, !(a * a > BSDF_ROUGHNESS_SQ_THRESH));
  switch lobe {
    case LOBE_D: { return 1.0; }
    case LOBE_S, LOBE_GR: { return rS; }
    case LOBE_GT: {
      let eta = glass_eta_side(m.ior, (m.flags & MATEVAL_BACKFACING) != 0u);
      return select(rS, 0.0, abs(eta - 1.0) < GLASS_ETA_SINGULAR_EPS);
    }
    case LOBE_NEE: {
      let rd = select(0.0, 1.0, (m.flags & MATEVAL_HAS_D) != 0u);
      let rs = select(0.0, rS, (m.flags & (MATEVAL_HAS_S | MATEVAL_HAS_G)) != 0u);
      return max(rd, rs);
    }
    default: { return FLT_MAX; }      // light / env vertex: always rough (never IEEE Inf, plan §1.7)
  }
}

/// One-sample mixture sampler. u = (u_lobe, u_h1, u_h2, u_rt) (math.md#rng-layout; u_rt is the glass R/T decision,
/// all four dims are always consumed by the caller). Class pick: u_lobe against the cumulative q(D), q(S), q(G).
/// Rejections follow Cycles exactly: D needs Ng·L > 0; S needs Ns·V > 0, Ng·L ≥ 0, Ns·L ≥ 0; G needs Ns·V > 0 and
/// Ng·L, Ns·L on the side of the sub-event. A delta S returns weight F(Ns·V)/q(S), a delta G sub-event
/// w_G·(R|T)/((P_R | 1−P_R)·q(G)), both with pdfs 0 (the next emitter hit gets MIS weight 1).
fn bsdf_sample(m: MatEval, V: vec3f, u: vec4f) -> BsdfSample {
  var s: BsdfSample;
  s.lobe = LOBE_NONE;
  let c = bsdf_prepare(m, V);
  if (!(c.q_d + c.q_s + c.q_g > 0.0)) { return s; }
  if (bctx_has(c, BC_HAS_G) && !(u.x < c.q_d + c.q_s)) {
    s.lobe = LOBE_GR;
    let gs = glass_sample(c, V, u.yzw);
    s.L = gs.L;
    s.lobe = select(LOBE_GR, LOBE_GT, gs.isT);     // the attempted sub-event, also for a rejected sample
    if (!gs.ok) { return s; }
    if (gs.isDelta) {
      s.is_delta = true;
      s.weight = gs.weightDelta;
      s.valid = true;
      return s;
    }
  } else if (u.x < c.q_d) {
    s.lobe = LOBE_D;
    s.L = lambert_sample_dir(c.ns, u.yz);
    if (!(dot(c.ng, s.L) > 0.0)) { return s; }                        // bsdf_diffuse.h:57-63
  } else {
    s.lobe = LOBE_S;
    let cos_ni = dot(c.ns, V);
    if (!(cos_ni > 0.0)) { return s; }                                 // bsdf_microfacet.h:693-696
    let H = ggx_sample_h(c.alpha, bctx_singular(c), c.ns, V, u.yz);
    let cos_hi = dot(H, V);
    s.L = 2.0 * cos_hi * H - V;
    if (dot(c.ng, s.L) < 0.0 || dot(c.ns, s.L) < 0.0) { return s; }   // bsdf_microfacet.h:757-761
    if (bctx_singular(c)) {
      // Cycles: eval = F·1e6, pdf = 1e6 (·lobe_prob) ⇒ weight F/q(S) (cyc-verify C2); no density.
      s.is_delta = true;
      s.weight = bsdf_F_S(c, cos_hi) / c.q_s;
      s.valid = true;
      return s;
    }
  }
  let e = bsdf_eval_ctx(c, V, s.L);
  s.pdf_marginal = e.p_d + e.p_s + e.p_g;
  var f = e.f_s;
  s.pdf_joint = e.p_s;
  if (s.lobe == LOBE_D) {
    f = e.f_d;
    s.pdf_joint = e.p_d;
  } else if (s.lobe == LOBE_GR || s.lobe == LOBE_GT) {
    // weight = f_G/(q(G)·p_G) with the eval form of the same (V, L): exactly f/pdf_joint (U-G5); a sample whose
    // re-derived configuration falls outside the support by f32 round-off (measure-small) is rejected.
    f = e.f_g;
    s.pdf_joint = select(0.0, e.p_g, ((e.g_side & 1u) != 0u) == (s.lobe == LOBE_GT));
  }
  if (!(s.pdf_joint > 0.0)) { return s; }
  s.weight = f / s.pdf_joint;
  s.valid = true;
  return s;
}

// ------------------------------------------------------------------------------------------------ extensions

/// Lobe-class pmf at V: (q(D|V), q(S|V), q(G|V), η_side of G). Debug views, tests (gap-bsdf U-6, glass U-G6/U-G8).
fn bsdf_lobe_probs(m: MatEval, V: vec3f) -> vec4f {
  let c = bsdf_prepare(m, V);
  return vec4f(c.q_d, c.q_s, c.q_g, c.g_eta);
}

/// Unnormalised Cycles sample weights at V: (sw_D, sw_S, sw_G, 0) (glass U-G6 vectors).
fn bsdf_sample_weights(m: MatEval, V: vec3f) -> vec4f {
  let c = bsdf_prepare(m, V);
  return vec4f(c.q_d, c.q_s, c.q_g, 0.0) * c.sw_sum;
}

/// Glass diagnostics for (V, L): (f_G.rgb… as luminance-free avg, Cycles eval pdf pdf_C, valid-only pdf_V, flags)
/// with flags bit0 = isT, bit1 = valid. pdfs WITHOUT q(G) (the closure's own densities). U-G2/U-G3.
fn bsdf_glass_diag(m: MatEval, V: vec3f, L: vec3f) -> vec4f {
  let c = bsdf_prepare(m, V);
  let g = glass_eval(c, V, L);
  return vec4f(bsdf_avg3(g.f), g.pdfC, g.pdfV, f32(select(0u, 1u, g.isT) | select(0u, 2u, g.valid)));
}

/// MatEval.flags for view direction V (material_eval calls this once per vertex).
/// The MATEVAL_BACKFACING input bit of m.flags is preserved.
fn bsdf_flags(m: MatEval, V: vec3f) -> u32 {
  let c = bsdf_prepare(m, V);
  var f = m.flags & MATEVAL_BACKFACING;
  let hasD = bctx_has(c, BC_HAS_D);
  let hasS = bctx_has(c, BC_HAS_S);
  let hasG = bctx_has(c, BC_HAS_G);
  let sing = bctx_singular(c);
  if (hasD || ((hasS || hasG) && !sing)) { f |= MATEVAL_HAS_NON_DELTA; }
  if (hasD && !hasS && !hasG) { f |= MATEVAL_DIFFUSE_ONLY; }
  if (hasD) { f |= MATEVAL_HAS_D; }
  if (hasS) { f |= MATEVAL_HAS_S; }
  if (hasS && sing) { f |= MATEVAL_S_DELTA; }
  if (hasG) { f |= MATEVAL_HAS_G; }
  if (hasG && sing) { f |= MATEVAL_G_DELTA; }
  return f;
}

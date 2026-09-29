// NEE in the product measure μ and the NEE/BSDF MIS weights (plan §2 rules 1–2; math.md#measure, math.md#mis).
//   μ = Σ_{area ∪ tri} δ_L ⊗ A_L + Σ_{delta ∪ sun} δ_(L, c_L) + δ_ENV ⊗ σ_env
//   (the env component, its sampler, p1Env and its MIS weights live in lights/env-sample.wgsl; nee_sample() there
//    dispatches the ENV alias entry and calls light_sample_entry() below for every other entry)
//   q  = P(L)/A_L (area, triangle) | P(L) (point, spot, sun)          P = realized pmf (select.wgsl)
//   F_NEE = T ⊙ ω1·f_cos ⊙ Λ·V/q;   p1 = q·r²/|cosθ_z| appears ONLY inside ω1/ω2
//   ω1 = M(B)p1/(M(B)p1 + p2), ω2 = p2/(M(B)p1 + p2), p2 = marginal BSDF pdf over non-delta lobes at x_{d−1}
// Hard rules by FLAG, never by pdf == 0 arithmetic: ω1 ≡ 1 for delta lights and for Mode-A analytic lights; ω2 = 1
// after a delta lobe; camera rays weight 1. Emissive triangles use two-technique MIS in every mode.
#include "lights/select.wgsl"
#include "lights/point.wgsl"
#include "lights/spot.wgsl"
#include "lights/area.wgsl"
#include "lights/sun.wgsl"
#include "lights/emissive.wgsl"

/// M(B): one shared function (math.md#mis). 32 at B = 1 with RIS-NEE (M6), 1 otherwise.
fn mis_M(B: u32) -> f32 { return 1.0; }

/// ω1 for an NEE sample: M p1/(M p1 + p2). Delta / Mode-A analytic ⇒ 1 by flag.
fn mis_w1(ls: LightSample, p2: f32, B: u32) -> f32 {
  if (ls.isDelta) { return 1.0; }
  if (ls.analytic && (lightsParams.flags & LP_MODE_A) != 0u) { return 1.0; }
  let mp1 = mis_M(B) * ls.p1;
  if (!(mp1 > 0.0)) { return 0.0; }                     // unreachable for a valid sample (q > 0, cos > 0)
  return mp1 / (mp1 + max(p2, 0.0));
}

/// ω2 for a BSDF-sampled endpoint with light pdf p1 (seen from x_{d−1}) and BSDF marginal p2.
fn mis_w2(p1: f32, p2: f32, B: u32) -> f32 {
  let mp1 = mis_M(B) * max(p1, 0.0);
  if (!(mp1 > 0.0)) { return 1.0; }                     // not an NEE entry (or zero pmf): BSDF is the only technique
  return p2 / (mp1 + p2);
}

/// One NEE light sample at shading point x (math.md#measure). hSel/hSel2: slots u_sel/u_sel2, u: (h_l0, h_l1).
/// Analytic lights and emissive triangles only (the env entry is handled by env-sample.wgsl nee_sample()).
fn light_sample(x: vec3f, hSel: u32, hSel2: u32, u: vec2f) -> LightSample {
  let slot = lightsParams.cur;
  if (slot.nEntries == 0u) { return light_sample_none(); }
  return light_sample_entry(x, slot, alias_sample(slot, hSel, hSel2), u);
}

/// NEE sample of a selected analytic / emissive-triangle alias entry (entry < nAnalytic + triCount).
fn light_sample_entry(x: vec3f, slot: LightSlot, entry: u32, u: vec2f) -> LightSample {
  let pmf = light_pmf(slot, entry);
  var s: LightSample;
  if (entry < slot.nAnalytic) {
    let r = light_load(slot, entry);
    switch (r.kind) {
      case LT_POINT: { s = point_sample(r, x); }
      case LT_SPOT: { s = spot_sample(r, x); }
      case LT_SUN: { s = sun_sample(r, x); }
      default: { s = area_sample(r, x, u); }
    }
    s.q = select(pmf * r.invArea, pmf, s.isDelta);
  } else {
    let i = entry - slot.nAnalytic;
    s = emissive_sample(i, x, u);
    s.q = pmf / bitcast<f32>(emissive_tri(i).y);
  }
  s.entry = entry;
  s.p1 = 0.0;                                           // delta lights and the sun: undefined, never used (flags)
  if (!s.isDelta && s.cosZ > 0.0) { s.p1 = s.q * s.dist * s.dist / s.cosZ; }   // no Inf: the sun's dist is never read
  s.valid = s.valid && s.q > 0.0;
  return s;
}

/// p1 of a BSDF-sampled hit on emissive triangle primId at z (unit geometric normal ngz) seen from x (math.md#mis:
/// recomputed at the hit's previous vertex). 0 when the triangle is not an NEE entry.
fn tri_light_p1(x: vec3f, z: vec3f, ngz: vec3f, primId: u32) -> f32 {
  let i = emissive_entry_of_prim(primId);
  if (i == LIGHT_NONE) { return 0.0; }
  let slot = lightsParams.cur;
  if (slot.nEntries == 0u) { return 0.0; }
  let d = z - x;
  let dist2 = dot(d, d);
  let cosZ = abs(dot(ngz, d)) * inverseSqrt(dist2);
  if (!(cosZ > 0.0)) { return 0.0; }
  let q = light_pmf(slot, slot.nAnalytic + i) / bitcast<f32>(emissive_tri(i).y);
  return q * dist2 / cosZ;
}

/// p1 of a BSDF-sampled crossing of analytic area light `entry` (rect/disk) at z, seen from x (Mode B, M3b: the
/// BSDF_ANALYTIC technique). One-sided; 0 behind the light. Same q = P(L)/A as NEE, so ω1 + ω2 = 1 exactly.
fn analytic_area_p1(x: vec3f, entry: u32, z: vec3f) -> f32 {
  let slot = lightsParams.cur;
  let r = light_load(slot, entry);
  let d = z - x;
  let dist2 = dot(d, d);
  let cosZ = -dot(d, r.normal) * inverseSqrt(dist2);
  if (!(cosZ > 0.0)) { return 0.0; }
  return light_pmf(slot, entry) * r.invArea * dist2 / cosZ;
}

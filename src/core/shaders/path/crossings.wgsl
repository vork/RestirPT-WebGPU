// Mode B / Mode A′ pass-through crossings of analytic area lights by a BSDF ray (plan §1.4 "Modes", §1.2 bounces;
// math.md#mis "Modes" and "Pass-throughs", #bounces; gap-light §1.3, §2.1–§2.4; Cycles light.h:258-421,
// shade_light.h:19-104).
//
// A BSDF ray from x_B (origin org, direction ω) crosses every front-facing rect/disk light whose plane it meets at
// t ∈ (0, t_hit) before the first triangle (t_hit = FLT_MAX on a miss). Each crossed light L is its own light-ending
// candidate (d = B+1, BSDF_ANALYTIC, lightId) with contribution
//     β ⊙ L_e(z → x_B)·ω2,   ω2 = p2/(M(B)·p1 + p2)   (Mode B, after a non-delta lobe)
//                            ω2 = 1                  (after a delta lobe: Cycles PATH_RAY_MIS_SKIP; the only case in A′)
// with p1 = pmf(L)/A·r²/cosθ_z measured from x_B (the SAME q as NEE: ω1 + ω2 = 1 exactly) and p2 the marginal BSDF
// pdf of ω at x_B (valid-only). A crossing creates no vertex and consumes no bounce, no RNG dimension and no RR; the
// continuing path is unchanged (the ray origin and the MIS origin stay at x_B). Analytic lights never occlude.
// Modes (LightsParams.flags, lights-gpu.ts lightModeFlags):
//   A   LP_MODE_A                    analytic lights NEE-only (ω1 ≡ 1), never crossed
//   B   LP_CROSS_ALL                 crossed by every BSDF ray, balance MIS with NEE
//   A′  LP_MODE_A | LP_CROSS_DELTA   crossed only by rays leaving a delta lobe, weight 1; NEE keeps ω1 ≡ 1 (a singular
//                                    vertex does no NEE, so the techniques never overlap: same expectation as B)
// Emissive triangles are unaffected (always two-technique MIS, measure.wgsl). Sphere lights (r > 0) are not supported
// (a ray could cross them twice).
#include "lights/measure.wgsl"

const LP_CROSS_ALL: u32 = 4u;     // Mode B
const LP_CROSS_DELTA: u32 = 8u;   // Mode A′

struct AreaCrossing {
  hit: bool,
  z: vec3f,        // crossing point on the light
  Le: vec3f,       // L_e(z → ray origin): one-sided, spread
}

/// Crossing of the ray (o, d) with rect/disk light r at t ∈ (0, tMax), front-facing only (the ray travels against a_L).
fn area_light_cross(r: LightRec, o: vec3f, d: vec3f, tMax: f32) -> AreaCrossing {
  var c: AreaCrossing;
  let dn = dot(d, r.normal);
  if (!(dn < 0.0)) { return c; }                     // one-sided: back faces are never intersected (area.h:405-408)
  let t = dot(r.pos - o, r.normal) / dn;
  if (!(t > 0.0 && t < tMax)) { return c; }
  let p = o + t * d - r.pos;
  let x = dot(p, r.axisU) / r.halfU;
  let y = dot(p, r.axisV) / r.halfV;
  let inside = select(abs(x) <= 1.0 && abs(y) <= 1.0, x * x + y * y <= 1.0, r.kind == LT_DISK);
  if (!inside) { return c; }
  c.Le = area_radiance(r, -normalize(d));
  c.z = o + t * d;
  c.hit = any(c.Le > vec3f(0.0));
  return c;
}

/// Σ over crossed analytic area lights of L_e·ω2 (rgb) and the number of crossing candidates (w) for the BSDF ray
/// (org, dir) from vertex x (index B), up to tHit. p2: marginal BSDF pdf of dir at x; afterDelta: the ray left a delta
/// lobe; bsdfOnly: the T9d BSDF-only estimator (ω2 = 1).
fn pt_crossings(x: vec3f, org: vec3f, dir: vec3f, tHit: f32, p2: f32, B: u32, afterDelta: bool, bsdfOnly: bool) -> vec4f {
  let flags = lightsParams.flags;
  let crossAll = (flags & LP_CROSS_ALL) != 0u;
  if (!(crossAll || ((flags & LP_CROSS_DELTA) != 0u && afterDelta))) { return vec4f(0.0); }
  let slot = lightsParams.cur;
  var sum = vec4f(0.0);
  for (var i = 0u; i < slot.lightCount; i++) {
    let r = light_load(slot, i);
    if (r.kind != LT_RECT && r.kind != LT_DISK) { continue; }
    let c = area_light_cross(r, org, dir, tHit);
    if (!c.hit) { continue; }
    var w2 = 1.0;
    if (crossAll && !afterDelta && !bsdfOnly) { w2 = mis_w2(analytic_area_p1(x, i, c.z), p2, B); }
    sum += vec4f(w2 * c.Le, 1.0);
  }
  return sum;
}

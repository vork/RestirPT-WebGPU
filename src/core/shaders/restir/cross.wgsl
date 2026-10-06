// Mode-B / A′ analytic-light crossings for ReSTIR (restir-m6-api.md MD6–MD8; PLAN §1.4 Modes, rule 3 (d); math.md#mis
// "Pass-throughs", #path-tree [M6 addition]; gap-light §2.1–§2.4, §5.4). Pipeline variant RS_MODE_B only (MD1).
//   A crossing of rect / disk light `entry` by a ray is stored light-locally as planar coordinates xy ∈ [−1, 1]²
//   (z = c + x·halfU·a_u + y·halfV·a_v; disk lights: x² + y² ≤ 1), so a moved light carries its crossing point
//   rigidly (temporal case (d-ana)) and no inverse of the concentric map is needed.
//   cross_end(x, ω, entry, slot) is the ONE end-term evaluation of cases (c-ana), deep, ∅ and of the refresh: the ray
//   (x, ω) re-intersected with the light (front-facing, inside, t > 0), L_e toward x and p1 at the crossing point.
//   Case (d-ana) (the light point is the rc vertex) evaluates the copied z = cross_point(xy) instead.
// Modes (LightsParams.flags; crossings.wgsl): LP_CROSS_ALL (B: every BSDF ray, ω2 = p2/(M p1 + p2) after a non-delta
// lobe, 1 after a delta lobe) | LP_CROSS_DELTA (A′: only rays leaving a delta lobe, ω2 = 1).
#include "path/crossings.wgsl"
#include "restir/m6-types.wgsl"

struct CrossHit { hit: bool, xy: vec2f, t: f32 }
struct CrossEnd { ok: bool, Le: vec3f, p1: f32, z: vec3f }

/// Crossing of ray (o, d) with rect / disk light r at t ∈ (0, tMax), front-facing only: planar light-local xy.
fn cross_ray(r: LightRec, o: vec3f, d: vec3f, tMax: f32) -> CrossHit {
  var c = CrossHit(false, vec2f(0.0), 0.0);
  let dn = dot(d, r.normal);
  if (!(dn < 0.0)) { return c; }                         // one-sided: the ray travels against a_L (area.h:405-408)
  let t = dot(r.pos - o, r.normal) / dn;
  if (!(t > 0.0 && t < tMax)) { return c; }
  let q = o + t * d - r.pos;
  let x = dot(q, r.axisU) / r.halfU;
  let y = dot(q, r.axisV) / r.halfV;
  let inside = select(abs(x) <= 1.0 && abs(y) <= 1.0, x * x + y * y <= 1.0, r.kind == LT_DISK);
  if (!inside) { return c; }
  c.hit = true;
  c.xy = vec2f(x, y);
  c.t = t;
  return c;
}

/// The light point of planar coordinates xy (the stored crossing).
fn cross_point(r: LightRec, xy: vec2f) -> vec3f {
  return r.pos + (xy.x * r.halfU) * r.axisU + (xy.y * r.halfV) * r.axisV;
}

/// Crossings enabled for a BSDF ray leaving a (non-)delta lobe, and whether its ω2 is the MIS weight (Mode B) or 1.
fn cross_enabled(afterDelta: bool) -> bool {
  let f = lightsParams.flags;
  return (f & LP_CROSS_ALL) != 0u || ((f & LP_CROSS_DELTA) != 0u && afterDelta);
}
fn cross_mis(afterDelta: bool) -> bool { return (lightsParams.flags & LP_CROSS_ALL) != 0u && !afterDelta; }

/// End term of a crossing end seen from x along unit direction w under light slot `slot` (re-intersection, MD6).
fn cross_end(x: vec3f, w: vec3f, entry: u32, slot: LightSlot) -> CrossEnd {
  var e = CrossEnd(false, vec3f(0.0), 0.0, vec3f(0.0));
  if (entry >= slot.nAnalytic) { return e; }
  let r = light_load(slot, entry);
  if (r.kind != LT_RECT && r.kind != LT_DISK) { return e; }
  let c = cross_ray(r, x, w, FLT_MAX);
  if (!c.hit) { return e; }
  e.z = cross_point(r, c.xy);
  e.Le = area_radiance(r, -w);
  e.p1 = analytic_area_p1_s(x, entry, e.z, slot);
  e.ok = true;
  return e;
}

/// Light-local words of a crossing endpoint: (RC_TAG_CROSS | entry, bits(x), bits(y)).
fn cross_words(entry: u32, xy: vec2f) -> vec3u { return vec3u(RC_TAG_CROSS | entry, bitcast<u32>(xy.x), bitcast<u32>(xy.y)); }
fn rs_is_cross_words(w: vec3u) -> bool { return (w.x & RC_TAG_MASK) == RC_TAG_CROSS && w.x != RC_ENV_DIR && w.x != RC_NONE; }

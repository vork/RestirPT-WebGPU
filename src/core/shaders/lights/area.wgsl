// Rect / disk area lights (one-sided, spread; plan §1.4, math.md#units-lights and #light-selection "Per-light position
// sampling"; gap-light §3.4; K/light/area.h:106-121, 313, 337-341, 405-408).
// - L_e = Φ/(πA) (record emit) · spread(θ), θ = angle between the emitted direction (z → x) and a_L.
//   spread = π ⇒ factor 1; else max((tan a − tan θ)·N_s, 0) (Cycles area_light_spread_attenuation).
// - One-sided: L_e(z→x) = 0 unless dot(x − z, a_L) > 0.
// - Area-uniform sampling in light-local (u, v) ∈ [0,1)² (the spherical-rectangle sampler is FORBIDDEN, it breaks J = 1):
//   rect: z = c + (u − ½)·len_u·a_u + (v − ½)·len_v·a_v;  disk: Shirley–Chiu concentric map, z = c + ½len_u δx a_u + …
//   p_A = 1/A, q = P(L)/A, Λ = L_e|cosθ_z|/r², p1 = q·r²/|cosθ_z|.
// - Camera rays see a visibleToCamera light as a pass-through length-1 term (area_light_crossing; same maths as the
//   emission kernel's cam_light_radiance). BSDF rays hit them only in Mode B (M3b).
#include "lights/lights.wgsl"

/// Shirley–Chiu concentric map [0,1)² → unit disk (gap-bsdf §11 sample_uniform_disk).
fn light_concentric_disk(u: vec2f) -> vec2f {
  let a = 2.0 * u - vec2f(1.0);
  if (a.x == 0.0 && a.y == 0.0) { return vec2f(0.0); }
  var r: f32;
  var phi: f32;
  if (abs(a.x) > abs(a.y)) {
    r = a.x;
    phi = (PI / 4.0) * (a.y / a.x);
  } else {
    r = a.y;
    phi = (PI / 2.0) - (PI / 4.0) * (a.x / a.y);
  }
  return r * vec2f(cos(phi), sin(phi));
}

/// Spread attenuation for the emitted direction with cos θ = cosT (> 0).
fn area_spread(r: LightRec, cosT: f32) -> f32 {
  if (r.spreadNorm < 0.0) { return 1.0; }
  let tanT = sqrt(max(1.0 - cosT * cosT, 0.0)) / cosT;
  return max((r.tanHalfSpread - tanT) * r.spreadNorm, 0.0);
}

/// Point z on the light for (u, v) ∈ [0,1)².
fn area_point(r: LightRec, u: vec2f) -> vec3f {
  if (r.kind == LT_DISK) {
    let d = light_concentric_disk(u);
    return r.pos + (r.halfU * d.x) * r.axisU + (r.halfV * d.y) * r.axisV;
  }
  return r.pos + ((2.0 * u.x - 1.0) * r.halfU) * r.axisU + ((2.0 * u.y - 1.0) * r.halfV) * r.axisV;
}

/// Radiance leaving z toward a point at unit direction w (from z), one-sided with spread.
fn area_radiance(r: LightRec, w: vec3f) -> vec3f {
  let c = dot(w, r.normal);
  if (!(c > 0.0)) { return vec3f(0.0); }
  return r.emit * area_spread(r, c);
}

fn area_sample(r: LightRec, x: vec3f, u: vec2f) -> LightSample {
  var s = light_sample_none();
  let z = area_point(r, u);
  let d = z - x;
  let dist2 = dot(d, d);
  if (!(dist2 > 0.0)) { return s; }
  let dist = sqrt(dist2);
  let w = d / dist;
  let cosZ = dot(-w, r.normal);                       // cosθ_z = n_z·(x − z)/r, one-sided: must be > 0
  s.valid = true;
  s.kind = r.kind;
  s.pos = z;
  s.nz = r.normal;
  s.dir = w;
  s.dist = dist;
  s.cosZ = max(cosZ, 0.0);
  s.Lambda = select(vec3f(0.0), r.emit * (area_spread(r, cosZ) * cosZ / dist2), cosZ > 0.0);
  s.isDelta = false;
  s.analytic = true;
  return s;
}

/// Radiance of the light if the ray (o, d) crosses it front-facing at t ∈ (0, tMax), else 0 (pass-through).
fn area_light_crossing(r: LightRec, o: vec3f, d: vec3f, tMax: f32) -> vec3f {
  let dn = dot(d, r.normal);
  if (!(dn < 0.0)) { return vec3f(0.0); }             // one-sided: the ray travels against a_L
  let t = dot(r.pos - o, r.normal) / dn;
  if (!(t > 0.0 && t < tMax)) { return vec3f(0.0); }
  let p = o + t * d - r.pos;
  let x = dot(p, r.axisU) / r.halfU;
  let y = dot(p, r.axisV) / r.halfV;
  let inside = select(abs(x) <= 1.0 && abs(y) <= 1.0, x * x + y * y <= 1.0, r.kind == LT_DISK);
  if (!inside) { return vec3f(0.0); }
  return area_radiance(r, -normalize(d));
}

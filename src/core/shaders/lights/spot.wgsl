// Spot light, r = 0 (delta position; math.md#units-lights "Spot profile", gap-light §3.3; K/light/spot.h:15-31).
// I(ω) = Φ/(4π)·S(cosθ'), cosθ' = dot(normalize(x − c), a_L), NOT cone-normalised (on-axis I equals a point light of
// the same Φ). S(t) = smoothstep01((t − cosH)·spotSmooth), spotSmooth = 1/((1 − cosH)·blend); blend = 0 ⇒ hard step
// (t > cosH), the guard for Cycles' (t − cosH)·∞. S is evaluated at the EVALUATING shading point, never cached.
#include "lights/lights.wgsl"

fn spot_smoothstep01(f: f32) -> f32 {
  if (f <= 0.0) { return 0.0; }
  if (f >= 1.0) { return 1.0; }
  return f * f * (3.0 - 2.0 * f);
}

/// S(cosθ') for record r.
fn spot_profile(r: LightRec, cosT: f32) -> f32 {
  if (r.spotSmooth < 0.0) { return select(0.0, 1.0, cosT > r.cosHalf); }
  return spot_smoothstep01((cosT - r.cosHalf) * r.spotSmooth);
}

fn spot_sample(r: LightRec, x: vec3f) -> LightSample {
  var s = light_sample_none();
  let d = r.pos - x;
  let dist2 = dot(d, d);
  if (!(dist2 > 0.0)) { return s; }
  let dist = sqrt(dist2);
  let w = d / dist;                                   // ω_L (toward the light)
  let S = spot_profile(r, dot(-w, r.normal));         // cosθ' = dot(normalize(x − c), a_L)
  s.valid = true;
  s.kind = LT_SPOT;
  s.pos = r.pos;
  s.nz = vec3f(0.0);
  s.dir = w;
  s.dist = dist;
  s.Lambda = r.emit * (S / dist2);
  s.isDelta = true;
  s.analytic = true;
  return s;
}

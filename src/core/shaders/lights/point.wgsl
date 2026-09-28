// Point light, r = 0 (delta position; plan §1.4, math.md#units-lights, gap-light §3.2; K/light/point.h:65-77).
// Radiant intensity I = Φ/(4π) W/sr (record emit). Λ = I/r², q = P(L) (unit atom), ω1 ≡ 1. Never hit by any ray.
// No positional dims are consumed (they stay reserved in the RNG layout).
#include "lights/lights.wgsl"

fn point_sample(r: LightRec, x: vec3f) -> LightSample {
  var s = light_sample_none();
  let d = r.pos - x;
  let dist2 = dot(d, d);
  if (!(dist2 > 0.0)) { return s; }
  let dist = sqrt(dist2);
  s.valid = true;
  s.kind = LT_POINT;
  s.pos = r.pos;
  s.nz = vec3f(0.0);
  s.dir = d / dist;
  s.dist = dist;
  s.Lambda = r.emit / dist2;
  s.isDelta = true;
  s.analytic = true;
  return s;
}

// Sun, angle = 0 (delta direction; math.md#units-lights, gap-light §3.6; cyc §2.5). Irradiance E = Φ (W/m²) on a
// surface perpendicular to the sun direction s_L = +Z_obj = −a_L. Λ = E, q = P(L), ω1 ≡ 1, visibility to infinity
// (visibleInf, t_max = FLT_MAX). Never hit by any ray.
#include "lights/lights.wgsl"

fn sun_sample(r: LightRec, x: vec3f) -> LightSample {
  var s = light_sample_none();
  s.valid = true;
  s.kind = LT_SUN;
  s.pos = x;
  s.nz = vec3f(0.0);
  s.dir = -r.normal;
  s.dist = FLT_MAX;
  s.Lambda = r.emit;
  s.isDelta = true;
  s.isInf = true;
  s.analytic = true;
  return s;
}

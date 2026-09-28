// Fresnel models (math.md#bsdf-v2 "Fresnel models"; gap-bsdf §3.4; Cycles closure/bsdf_util.h, bsdf_microfacet.h).

fn bsdf_avg3(v: vec3f) -> f32 { return (v.x + v.y + v.z) * (1.0 / 3.0); }   // Cycles average()
fn bsdf_max3(v: vec3f) -> f32 { return max(v.x, max(v.y, v.z)); }           // Cycles reduce_max()

/// F0(η) = ((η−1)/(η+1))²   [bsdf_util.h:338-341]
fn fresnel_F0_from_ior(eta: f32) -> f32 { return sqr((eta - 1.0) / (eta + 1.0)); }

/// ior_from_F0(f) = (1+√f)/(1−√f), f clamped to [0, 0.99]   [bsdf_util.h:332-336]
fn fresnel_ior_from_F0(f0: f32) -> f32 {
  let s = sqrt(clamp(f0, 0.0, 0.99));
  return (1.0 + s) / (1.0 - s);
}

/// Exact unpolarised dielectric Fresnel; 1 under TIR   [bsdf_util.h:47-99 fresnel_dielectric_cos]
fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
  let g = eta * eta - (1.0 - cos_i * cos_i);
  if (g <= 0.0) { return 1.0; }
  let ci = abs(cos_i);
  let ct = -sqrt(g) / eta;
  let rs = (ci + eta * ct) / (ci - eta * ct);
  let rp = (ct + eta * ci) / (eta * ci - ct);
  return 0.5 * (rs * rs + rp * rp);
}

/// Generalized Schlick, exponent < 0 mode: the real Fresnel of η remapped from [F0(η), 1] to [f0, 1] (f90 = 1).
/// The saturate is ported literally (it clips for η > 2+√3; cycles-verify C4).   [bsdf_microfacet.h:313-320]
fn fresnel_gen_schlick_ior(cos_hi: f32, eta: f32, f0: vec3f) -> vec3f {
  let F0r = fresnel_F0_from_ior(eta);
  let s = saturate((fresnel_dielectric(cos_hi, eta) - F0r) / (1.0 - F0r));
  return mix(f0, vec3f(1.0), s);
}

/// F82-tint: saturate(F0 + (1−F0)·s⁵ − B·c·s⁶), s = saturate(1 − c)   [bsdf_util.h:178-184]
fn fresnel_f82(c: f32, F0: vec3f, B: vec3f) -> vec3f {
  let s = saturate(1.0 - c);
  let s5 = (s * s) * (s * s) * s;
  return saturate(mix(F0, vec3f(1.0), s5) - B * (c * s5 * s));
}

/// B(F0, tint) = (F0 + (1−F0)f⁵)·(7/f⁶)·(1 − tint), f = 6/7; B = 0 iff tint ≡ 1   [bsdf_util.h:146-158; bsdf_microfacet.h:908-913]
fn fresnel_f82_tint_B(F0: vec3f, tint: vec3f) -> vec3f {
  if (all(tint == vec3f(1.0))) { return vec3f(0.0); }
  let f = 6.0 / 7.0;
  let f5 = (f * f) * (f * f) * f;
  return mix(F0, vec3f(1.0), f5) * (7.0 / (f5 * f)) * (vec3f(1.0) - tint);
}

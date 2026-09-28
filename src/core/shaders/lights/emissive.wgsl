// Emissive triangles (plan §1.4; math.md#units-lights "Emissive triangle", #light-selection, #measure; gap-light §3.5;
// K/closure/emissive.h). Two-sided, opaque, occluding, always hittable. L_e = emissiveFactor·strength ⊙
// sRGB⁻¹(emissive texture at uv) — the same expression as the emission kernel. Area-uniform sampling:
// bary = (1 − √u₁, √u₁(1 − u₂), √u₁u₂) over (v0, v1, v2), p_A = 1/A, q = P(tri)/A, Λ = L_e|cosθ_z|/r².
// The area A comes from the record (identical for NEE q and for the BSDF-hit p1, so the MIS partition is exact).
// Requires scene/scene-data.wgsl (sceneTris, sceneVerts, sceneMaterials, tex_sample).
#include "lights/lights.wgsl"
#include "scene/scene-data.wgsl"

/// Emitted radiance of triangle primId at hit barycentrics (u, v) (0 if not TRI_EMISSIVE). Two-sided, no π.
fn tri_emission(primId: u32, u: f32, v: f32) -> vec3f {
  let t = sceneTris[primId];
  if ((tri_flags(t) & TRI_EMISSIVE) == 0u) { return vec3f(0.0); }
  let m = sceneMaterials[tri_material(t)];
  return m.emission * tex_sample(m.texEmissive, scene_uv0(t, u, v)).rgb;
}

/// Unit geometric (winding) normal of primId, unflipped.
fn tri_normal(primId: u32) -> vec3f {
  let t = sceneTris[primId];
  let a = sceneVerts[t.x].p;
  return normalize(cross(sceneVerts[t.y].p - a, sceneVerts[t.z].p - a));
}

/// Area-uniform sample of emissive-triangle entry i as seen from x. u = (u₁, u₂).
fn emissive_sample(i: u32, x: vec3f, u: vec2f) -> LightSample {
  var s = light_sample_none();
  let e = emissive_tri(i);
  let primId = e.x;
  let t = sceneTris[primId];
  let p0 = sceneVerts[t.x].p;
  let p1 = sceneVerts[t.y].p;
  let p2 = sceneVerts[t.z].p;
  let su = sqrt(u.x);
  let b1 = su * (1.0 - u.y);
  let b2 = su * u.y;
  let z = p0 + b1 * (p1 - p0) + b2 * (p2 - p0);
  let ng = normalize(cross(p1 - p0, p2 - p0));
  let d = z - x;
  let dist2 = dot(d, d);
  if (!(dist2 > 0.0)) { return s; }
  let dist = sqrt(dist2);
  let w = d / dist;
  let cosZ = abs(dot(ng, w));                         // two-sided
  s.valid = true;
  s.kind = LT_TRI;
  s.pos = z;
  s.nz = ng;
  s.prim = primId;
  s.dir = w;
  s.dist = dist;
  s.cosZ = cosZ;
  s.Lambda = tri_emission(primId, b1, b2) * (cosZ / dist2);
  s.isDelta = false;
  s.analytic = false;
  return s;
}

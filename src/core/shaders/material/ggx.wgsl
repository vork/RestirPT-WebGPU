// Isotropic GGX microfacet reflection (math.md#bsdf-v1 "GGX building blocks"; gap-bsdf §3.2;
// Cycles closure/bsdf_microfacet.h:189-220 (VNDF), :500-563 (D, Λ), :586-675 (eval), :686-825 (sample)).
//   a2 = α_x·α_y = α² = r⁴
//   D(c)   = a2 / (π·((1 − c²) + a2·c²)²)
//   Λ(c)   = ½(√(1 + a2·max(1/c² − 1, 0)) − 1),  c := max(|c|, 1e-7)   (|·| fix: glass U-G10)
//   G2     = 1/(1 + Λ(N·V) + Λ(N·L))                  (height-correlated Smith)
//   eval   = F(V·H)·D·G2/(4·N·V)                       (includes |N·L|)
//   p_VNDF = D·G1(V)/(4·N·V) = D/(4·N·V·(1 + Λ(N·V)))
//   valid  : N·V > 0, a2 > 2e-10, N·L ≥ 0 (N·L < 0 is "transmission", rejected for reflection closures)

const BSDF_ROUGHNESS_SQ_THRESH: f32 = 2e-10;   // svm/types.h:545 (singular ⇔ !(a2 > thresh), r ≤ 0.0037606)

/// GGX D. `sin2` = 1 − cos²(N·H) passed in explicitly: callers compute it as |N×H|², which is the same quantity
/// without the catastrophic cancellation of 1 − c² in f32 at low roughness (r ≲ 0.05 near the peak). The value is
/// Cycles' bsdf_D with cos_NH2 = min(c², 1); only the f32 rounding differs.
fn ggx_D(a2: f32, cos2_in: f32, sin2_in: f32) -> f32 {
  let cos2 = min(cos2_in, 1.0);
  let sin2 = clamp(sin2_in, 0.0, 1.0);
  let t = sin2 + a2 * cos2;
  return a2 / (PI * t * t);
}

/// Smith Λ for GGX; |cos| and the 1e-7 floor avoid computing with Inf under relaxed Metal math (plan §1.7; U-G10).
fn ggx_lambda(a2: f32, cos_n: f32) -> f32 {
  let c = max(abs(cos_n), 1e-7);
  return 0.5 * (sqrt(1.0 + a2 * max(1.0 / (c * c) - 1.0, 0.0)) - 1.0);
}

/// Heitz 2018 VNDF sampling in the local frame (z = N), Cycles' variant with the (1 + V_s.z)/2 warp.
fn ggx_sample_vndf(Vl: vec3f, alpha: f32, u: vec2f) -> vec3f {
  // 3.2: stretch to the hemisphere configuration
  let Vs = normalize(vec3f(alpha * Vl.x, alpha * Vl.y, Vl.z));
  // 4.1: orthonormal basis
  let lensq = Vs.x * Vs.x + Vs.y * Vs.y;
  var T1 = vec3f(1.0, 0.0, 0.0);
  var T2 = vec3f(0.0, 1.0, 0.0);
  if (lensq > 1e-7) {
    T1 = vec3f(-Vs.y, Vs.x, 0.0) * inverseSqrt(lensq);
    T2 = cross(Vs, T1);
  }
  // 4.2: parameterization of the projected area
  var t = bsdf_sample_uniform_disk(u);
  t.y = mix(sqrt(max(1.0 - t.x * t.x, 0.0)), t.y, 0.5 * (1.0 + Vs.z));
  // 4.3: reprojection onto the hemisphere
  let Hs = t.x * T1 + t.y * T2 + sqrt(max(1.0 - dot(t, t), 0.0)) * Vs;
  // 3.4: unstretch
  return normalize(vec3f(alpha * Hs.x, alpha * Hs.y, max(0.0, Hs.z)));
}

/// Result of the GGX reflection eval for one (V, L), without Fresnel and closure weight.
struct GgxRefl {
  valid: bool,
  cos_hi: f32,   // H·V (Fresnel argument)
  g: f32,        // D·G2/(4·N·V)       (eval·cos per unit F)
  pdf: f32,      // D·G1(V)/(4·N·V)    (VNDF reflection pdf, solid angle)
}

/// Isotropic GGX reflection eval + VNDF pdf (bsdf_microfacet_eval reflection branch). No Ng test.
fn ggx_refl_eval(a2: f32, N: vec3f, V: vec3f, L: vec3f) -> GgxRefl {
  var r: GgxRefl;
  let cos_ni = dot(N, V);
  let cos_no = dot(N, L);
  if (!(cos_ni > 0.0) || cos_no < 0.0 || !(a2 > BSDF_ROUGHNESS_SQ_THRESH)) { return r; }
  let Hu = V + L;
  let len2 = dot(Hu, Hu);
  if (!(len2 > 0.0)) { return r; }
  let H = Hu * inverseSqrt(len2);
  let cos_nh = dot(N, H);
  let nxh = cross(N, H);
  let D = ggx_D(a2, cos_nh * cos_nh, dot(nxh, nxh));
  let lI = ggx_lambda(a2, cos_ni);
  let lO = ggx_lambda(a2, cos_no);
  let jac = D / cos_ni * 0.25;   // Cycles "common" (a WGSL reserved word)
  r.valid = true;
  r.cos_hi = dot(H, V);
  r.pdf = jac / (1.0 + lI);
  r.g = jac / (1.0 + lI + lO);
  return r;
}

/// Sampled half-vector (world) for V about N: the VNDF sample, or N itself when singular.
fn ggx_sample_h(alpha: f32, singular: bool, N: vec3f, V: vec3f, u: vec2f) -> vec3f {
  if (singular) { return N; }
  let tb = bsdf_make_orthonormals(N);
  let Vl = vec3f(dot(tb[0], V), dot(tb[1], V), dot(N, V));
  let Hl = ggx_sample_vndf(Vl, alpha, u);
  return Hl.x * tb[0] + Hl.y * tb[1] + Hl.z * N;
}

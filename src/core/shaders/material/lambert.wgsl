// Lambert (math.md#bsdf-v1 "Lambert"; gap-bsdf §3.3; Cycles closure/bsdf_diffuse.h, sample/mapping.h).
//   eval·cos = max(N·L, 0)/π,  pdf = max(N·L, 0)/π  (no Ng test in eval)
//   sample   = cosine hemisphere about N via the concentric disk; rejected unless Ng·L > 0 (strict).

/// Concentric (Shirley–Chiu) square → disk map   [sample/mapping.h sample_uniform_disk]
fn bsdf_sample_uniform_disk(u: vec2f) -> vec2f {
  let a = 2.0 * u.x - 1.0;
  let b = 2.0 * u.y - 1.0;
  if (a == 0.0 && b == 0.0) { return vec2f(0.0); }
  var r: f32;
  var phi: f32;
  if (a * a > b * b) {
    r = a;
    phi = (0.25 * PI) * (b / a);
  } else {
    r = b;
    phi = (0.5 * PI) - (0.25 * PI) * (a / b);
  }
  return vec2f(r * cos(phi), r * sin(phi));
}

/// Cycles make_orthonormals (util/math_float3.h:718-741): returns (T, B) with (T, B, N) right-handed.
fn bsdf_make_orthonormals(N: vec3f) -> mat2x3f {
  var a: vec3f;
  if (N.x != N.y || N.x != N.z) { a = vec3f(N.z - N.y, N.x - N.z, N.y - N.x); }
  else { a = vec3f(N.z - N.y, N.x + N.z, -N.y - N.x); }
  a = normalize(a);
  return mat2x3f(a, cross(N, a));
}

/// Lambert eval·cos (per unit closure weight) == its pdf.
fn lambert_eval(N: vec3f, L: vec3f) -> f32 { return max(dot(N, L), 0.0) * INV_PI; }

/// Cosine-hemisphere direction about N (sample_cos_hemisphere). The Ng rejection is the caller's.
fn lambert_sample_dir(N: vec3f, u: vec2f) -> vec3f {
  let d = bsdf_sample_uniform_disk(u);
  let tb = bsdf_make_orthonormals(N);
  return d.x * tb[0] + d.y * tb[1] + sqrt(max(1.0 - dot(d, d), 0.0)) * N;
}

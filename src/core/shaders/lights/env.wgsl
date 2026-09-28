// Environment map: Cycles-identical equirect mapping and the ONE env radiance lookup (plan §1.4b, rule 15).
// math.md#env-mapping. γ = Blender Mapping-node rotation Z (POINT), passed as (cg, sg) = (cos γ, sin γ) so that the
// current and previous env records (temporal reuse) can both be evaluated.
//
// Texture convention: texEnv is rgba32float with rows uploaded BOTTOM-UP (row 0 = nadir, Blender ImBuf order,
// src/core/render/env-gpu.ts). In WebGPU the normalized texture coordinate y = 0 addresses the FIRST uploaded row,
// so Cycles' (u, v) is used UNCHANGED as the texture coordinate: v = 0 → row 0 (nadir), v = 1 → last row (zenith).
// There is deliberately no 1 − v flip (Cycles svm_node_tex_environment has none either). Sampler sEnv is
// repeat/repeat + linear/linear with 1 mip, so within half a texel of a pole the lookup blends the top and bottom
// rows — Cycles' EXTENSION_REPEAT quirk, replicated on purpose (ENV-U7).
//
// Includer must define ENV_GROUP and ENV_BINDING (uniform at ENV_BINDING, texture +1, sampler +2), e.g. via
// envDefines() in env-gpu.ts, or define ENV_NO_BINDINGS to get only the pure mapping functions.
#include "common/math.wgsl"

// Must match ENV_UNIFORM_SIZE / writeEnvUniform in env-gpu.ts (32 B).
struct EnvParams {
  cg: f32,                 // cos γ
  sg: f32,                 // sin γ
  strength: f32,
  visibleToCamera: u32,    // 0/1
  tint: vec3f,
  flags: u32,              // ENV_FLAG_*
}
const ENV_FLAG_PRESENT: u32 = 1u; // an env map is loaded (else texEnv is a 1×1 black placeholder)

#ifndef ENV_NO_BINDINGS
@group($ENV_GROUP) @binding($ENV_BINDING) var<uniform> envParams: EnvParams;
@group($ENV_GROUP) @binding($ENV_BINDING + 1) var texEnv: texture_2d<f32>;
@group($ENV_GROUP) @binding($ENV_BINDING + 2) var sEnv: sampler;
#endif

// b = R_z(γ)·C·d, C = R_x(+90°) (glTF +Y up → Blender +Z up). math.md#env-mapping
fn envToBlender(d: vec3f, cg: f32, sg: f32) -> vec3f {
  return vec3f(cg * d.x + sg * d.z, sg * d.x - cg * d.z, d.y);
}

// Cycles direction_to_equirectangular(b): u = (atan2(b.y,b.x) − π)/(−2π), v = (acos(b.z/|b|) − π)/(−π).
// math.md#env-mapping. Two numerical choices, both mathematically identical to the formula (ENV-U2):
// - θ = atan2(|b.xy|, b.z) instead of acos(clamp(b.z, −1, 1)): Metal's fast acos is off by up to ~1.6e-5 rad
//   near |b.z| → 1 (dv ≈ 5e-6, measured), atan2 is well conditioned everywhere and equals acos(b.z/|b|) exactly
//   in real arithmetic (Cycles normalizes too). |b.xy| ≥ +0, so the nadir gives θ = +π, never −π.
// - At the exact poles (b.xy = 0) WGSL atan2(0, 0) is NaN on Metal; Cycles' atan2f(0, 0) = 0 → u = 0.5.
fn envUV(d: vec3f, cg: f32, sg: f32) -> vec2f {
  let b = envToBlender(d, cg, sg);
  let phi = select(atan2(b.y, b.x), 0.0, b.x == 0.0 && b.y == 0.0);
  let theta = atan2(length(b.xy), b.z);
  let u = (phi - PI) / (-TWO_PI);
  let v = (theta - PI) / (-PI);
  return vec2f(u, v);
}

// Inverse (Cycles equirectangular_to_direction, then C⁻¹·R_z(−γ)). math.md#env-mapping
fn envDir(uv: vec2f, cg: f32, sg: f32) -> vec3f {
  let phi = -TWO_PI * uv.x + PI;
  let theta = -PI * uv.y + PI;
  let st = sin(theta);
  let b = vec3f(st * cos(phi), st * sin(phi), cos(theta));
  let rx = cg * b.x + sg * b.y;       // R_z(−γ)·b
  let ry = -sg * b.x + cg * b.y;
  return vec3f(rx, b.z, -ry);         // C⁻¹(X, Y, Z) = (X, Z, −Y)
}

#ifndef ENV_NO_BINDINGS
// L_env = scale ⊙ texel(uv), scale = strength·tint of the env record in use (cur or prev). math.md#env-mapping
fn envRadianceScaled(uv: vec2f, scale: vec3f) -> vec3f {
  return scale * textureSampleLevel(texEnv, sEnv, uv, 0.0).rgb;
}

// L_env with the current record: strength·tint ⊙ texel(uv). The ONLY env radiance evaluation (plan rule 15).
fn envRadiance(uv: vec2f) -> vec3f {
  return envRadianceScaled(uv, envParams.strength * envParams.tint);
}

// Camera-miss term (length-1 path, weight 1, outside the reservoir): visibleToCamera · L_env(d). math.md#path-tree
fn envBackground(d: vec3f) -> vec3f {
  if ((envParams.flags & ENV_FLAG_PRESENT) == 0u || envParams.visibleToCamera == 0u) { return vec3f(0.0); }
  return envRadiance(envUV(d, envParams.cg, envParams.sg));
}
#endif

// Environment map: Cycles-identical equirect mapping and the ONE env radiance lookup (plan §1.4b, rule 15).
// math.md#env-mapping. γ = Blender Mapping-node rotation Z (POINT), passed as (cg, sg) = (cos γ, sin γ) so that the
// current and previous env records (temporal reuse) can both be evaluated.
//
// Texture convention: texEnv is rgba32float with rows uploaded BOTTOM-UP (row 0 = nadir, Blender ImBuf order,
// src/core/render/env-gpu.ts). In WebGPU the normalized texture coordinate y = 0 addresses the FIRST uploaded row,
// so Cycles' (u, v) is used UNCHANGED as the texture coordinate: v = 0 → row 0 (nadir), v = 1 → last row (zenith).
// There is deliberately no 1 − v flip (Cycles svm_node_tex_environment has none either). The lookup is an explicit
// f32 bilinear (4 × textureLoad, f32 weights) with repeat/repeat wrap, so within half a texel of a pole it blends the
// top and bottom rows — Cycles' EXTENSION_REPEAT quirk, replicated on purpose (ENV-U7). M5 (restir-temporal-api.md
// Changelog C-10): the hardware sampler quantizes the sub-texel weights (8 bit on Apple GPUs), which turned ulp-level
// envUV differences between pipelines into jumps of up to 2e-3 on high-frequency maps (C-9); the explicit lookup is
// continuous in uv and exact in f32 for every texel format. The sampler binding stays in the layout (unused).
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

// Pipeline-stable arithmetic (C-10): every multiply-add below is an explicit fma, and the divisions by 2π / π are
// multiplications by constant reciprocals, so the result cannot depend on how a pipeline contracts or rewrites them
// (Metal fast math); the same direction bits give the same (u, v) bits in every pipeline.
const ENV_INV_TWO_PI: f32 = 0.15915494309189535;
const ENV_INV_PI: f32 = 0.3183098861837907;

// b = R_z(γ)·C·d, C = R_x(+90°) (glTF +Y up → Blender +Z up). math.md#env-mapping
fn envToBlender(d: vec3f, cg: f32, sg: f32) -> vec3f {
  return vec3f(fma(cg, d.x, sg * d.z), fma(sg, d.x, -(cg * d.z)), d.y);
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
  let theta = atan2(sqrt(fma(b.x, b.x, b.y * b.y)), b.z);
  let u = fma(phi, -ENV_INV_TWO_PI, 0.5);             // (φ − π)/(−2π)
  let v = fma(theta, -ENV_INV_PI, 1.0);               // (θ − π)/(−π)
  return vec2f(u, v);
}

// Inverse (Cycles equirectangular_to_direction, then C⁻¹·R_z(−γ)). math.md#env-mapping
fn envDir(uv: vec2f, cg: f32, sg: f32) -> vec3f {
  let phi = fma(-TWO_PI, uv.x, PI);
  let theta = fma(-PI, uv.y, PI);
  let st = sin(theta);
  let b = vec3f(st * cos(phi), st * sin(phi), cos(theta));
  let rx = fma(cg, b.x, sg * b.y);    // R_z(−γ)·b
  let ry = fma(-sg, b.x, cg * b.y);
  return vec3f(rx, b.z, -ry);         // C⁻¹(X, Y, Z) = (X, Z, −Y)
}

#ifndef ENV_NO_BINDINGS
// Bilinear reconstruction of the env map at (u, v): texel centres at (i + ½)/W, (j + ½)/H, repeat/repeat wrap (u seam
// and the pole rows), f32 weights. Non-finite uv (bit test; relaxed math folds x != x) → 0. math.md#env-mapping
fn envTexel(uv: vec2f) -> vec3f {
  _ = sEnv;                                             // keeps the (unused) sampler in auto layouts / bind groups
  let ub = bitcast<vec2u>(uv) & vec2u(0x7f800000u);
  if (any(ub == vec2u(0x7f800000u))) { return vec3f(0.0); }
  let dim = vec2i(textureDimensions(texEnv, 0));
  let x = fma(uv, vec2f(dim), vec2f(-0.5));
  let x0 = floor(x);
  let t = x - x0;
  let i0 = ((vec2i(x0) % dim) + dim) % dim;
  let i1 = (i0 + 1) % dim;
  let a = textureLoad(texEnv, i0, 0).rgb;
  let b = textureLoad(texEnv, vec2i(i1.x, i0.y), 0).rgb;
  let c = textureLoad(texEnv, vec2i(i0.x, i1.y), 0).rgb;
  let d = textureLoad(texEnv, i1, 0).rgb;
  let tx = vec3f(t.x);
  let lo = fma(b - a, tx, a);                           // explicit fma lerps (pipeline-stable, C-10)
  let hi = fma(d - c, tx, c);
  return fma(hi - lo, vec3f(t.y), lo);
}

// L_env = scale ⊙ texel(uv), scale = strength·tint of the env record in use (cur or prev). math.md#env-mapping
fn envRadianceScaled(uv: vec2f, scale: vec3f) -> vec3f {
  return scale * envTexel(uv);
}

// L_env under env record er (frame-selected, restir-temporal-api.md §3.1): er.strength·er.tint ⊙ texel(uv).
fn envRadiance_s(uv: vec2f, er: EnvParams) -> vec3f {
  return envRadianceScaled(uv, er.strength * er.tint);
}

// L_env with the current record: strength·tint ⊙ texel(uv). The ONLY env radiance evaluation (plan rule 15).
fn envRadiance(uv: vec2f) -> vec3f {
  return envRadiance_s(uv, envParams);
}

// Camera-miss term (length-1 path, weight 1, outside the reservoir): visibleToCamera · L_env(d). math.md#path-tree
fn envBackground(d: vec3f) -> vec3f {
  if ((envParams.flags & ENV_FLAG_PRESENT) == 0u || envParams.visibleToCamera == 0u) { return vec3f(0.0); }
  return envRadiance(envUV(d, envParams.cg, envParams.sg));
}
#endif

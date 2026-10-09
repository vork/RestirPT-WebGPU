// Material texture sampling (plan §1.6; gap-bsdf §9 NORMATIVE; math.md#bsdf-v2, math.md#units-lights emissive).
//
// Colour data is stored as rgba8unorm (never -srgb) and filtered by the hardware on the ENCODED bytes;
// sRGB → linear is applied AFTER filtering with the exact EOTF (Cycles svm/image.h:28-38, util/color.h:63-69).
//
// Defines (from textures-gpu.ts `textureDefines()`):
//   TEX_GROUP          bind group of the texture bindings
//   TEX_BINDING_BASE   first binding: arrays at BASE+0..BASE+15, samplers at BASE+16..BASE+23
//   TEX_ARRAYS         number of texture_2d_array bindings actually used (0..16)
//   TEX_SAMPLERS       number of sampler bindings actually used (0..8)
// Binding budget: ≤ 16 sampled textures + ≤ 8 samplers here; the profile allows 48 / 16 per stage, which leaves
// room for the env map (texEnv + sEnv) and debug/V-buffer textures in the same stage.
// Only TEX_ARRAYS × TEX_SAMPLERS textureSampleLevel call sites are compiled.
//
// Validation mode samples LOD 0 (arrays have 1 mip). Interactive mode passes a ray-cone LOD; its arrays hold
// linear-space mips re-encoded to sRGB bytes (textures-gpu.ts), so the same decode-after-filter applies.

// TexSlot (32 B, embedded in material records): uv' = (dot(xf0, [uv,1]), dot(xf1, [uv,1])) (KHR_texture_transform)
//   info bits: [0,4) array index, [4,15) layer, [15,18) sampler index, 18 sRGB, 19 uv set, 31 valid.
struct TexSlot {
  xf0: vec3f,
  info: u32,
  xf1: vec3f,
  pad: u32,
}

const TEX_VALID: u32 = 0x80000000u;
const TEX_SRGB: u32 = 0x40000u;
const TEX_UVSET1: u32 = 0x80000u;

#if TEX_ARRAYS > 0
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 0) var texArr0: texture_2d_array<f32>;
#endif
#if TEX_ARRAYS > 1
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 1) var texArr1: texture_2d_array<f32>;
#endif
#if TEX_ARRAYS > 2
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 2) var texArr2: texture_2d_array<f32>;
#endif
#if TEX_ARRAYS > 3
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 3) var texArr3: texture_2d_array<f32>;
#endif
#if TEX_ARRAYS > 4
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 4) var texArr4: texture_2d_array<f32>;
#endif
#if TEX_ARRAYS > 5
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 5) var texArr5: texture_2d_array<f32>;
#endif
#if TEX_ARRAYS > 6
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 6) var texArr6: texture_2d_array<f32>;
#endif
#if TEX_ARRAYS > 7
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 7) var texArr7: texture_2d_array<f32>;
#endif
#if TEX_ARRAYS > 8
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 8) var texArr8: texture_2d_array<f32>;
#endif
#if TEX_ARRAYS > 9
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 9) var texArr9: texture_2d_array<f32>;
#endif
#if TEX_ARRAYS > 10
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 10) var texArr10: texture_2d_array<f32>;
#endif
#if TEX_ARRAYS > 11
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 11) var texArr11: texture_2d_array<f32>;
#endif
#if TEX_ARRAYS > 12
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 12) var texArr12: texture_2d_array<f32>;
#endif
#if TEX_ARRAYS > 13
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 13) var texArr13: texture_2d_array<f32>;
#endif
#if TEX_ARRAYS > 14
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 14) var texArr14: texture_2d_array<f32>;
#endif
#if TEX_ARRAYS > 15
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 15) var texArr15: texture_2d_array<f32>;
#endif
#if TEX_SAMPLERS > 0
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 16) var texSmp0: sampler;
#endif
#if TEX_SAMPLERS > 1
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 17) var texSmp1: sampler;
#endif
#if TEX_SAMPLERS > 2
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 18) var texSmp2: sampler;
#endif
#if TEX_SAMPLERS > 3
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 19) var texSmp3: sampler;
#endif
#if TEX_SAMPLERS > 4
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 20) var texSmp4: sampler;
#endif
#if TEX_SAMPLERS > 5
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 21) var texSmp5: sampler;
#endif
#if TEX_SAMPLERS > 6
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 22) var texSmp6: sampler;
#endif
#if TEX_SAMPLERS > 7
@group($TEX_GROUP) @binding($TEX_BINDING_BASE + 23) var texSmp7: sampler;
#endif

// Exact sRGB EOTF used by Cycles on Metal (gap-bsdf §9 item 3). Alpha is never decoded.
fn tex_srgb_to_linear(c: f32) -> f32 {
  if (c < 0.04045) { return select(c * (1.0 / 12.92), 0.0, c < 0.0); }
  return pow((c + 0.055) * (1.0 / 1.055), 2.4);
}
fn tex_linear_to_srgb(c: f32) -> f32 {
  if (c < 0.0031308) { return select(c * 12.92, 0.0, c < 0.0); }
  return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}
fn tex_srgb_to_linear4(c: vec4f) -> vec4f {
  return vec4f(tex_srgb_to_linear(c.r), tex_srgb_to_linear(c.g), tex_srgb_to_linear(c.b), c.a);
}

fn tex_slot_valid(s: TexSlot) -> bool { return (s.info & TEX_VALID) != 0u; }
fn tex_slot_uv_set(s: TexSlot) -> u32 { return select(0u, 1u, (s.info & TEX_UVSET1) != 0u); }
fn tex_slot_uv(s: TexSlot, uv: vec2f) -> vec2f {
#if TEX_NO_XFORM
  return uv;                     // perf2 WP-8 (MAT_VARIANTS): every valid slot of the scene is the identity transform
#else
  let h = vec3f(uv, 1.0);
  return vec2f(dot(s.xf0, h), dot(s.xf1, h));
#endif
}
#if MAT_VARIANTS
/// perf2 WP-8: tex_slot_uv from the two transform rows (field-wise material loads, material-eval.wgsl).
fn tex_xf_uv(xf0: vec3f, xf1: vec3f, uv: vec2f) -> vec2f {
  let h = vec3f(uv, 1.0);
  return vec2f(dot(xf0, h), dot(xf1, h));
}
#endif

#if TEX_ARRAYS > 0
// Sampler switch for one array (textures and samplers are legal WGSL function parameters).
fn tex_sample_arr(t: texture_2d_array<f32>, si: u32, uv: vec2f, layer: u32, lod: f32) -> vec4f {
#if MAT_VARIANTS
  var r: vec4f;
  switch si {
#if TEX_SAMPLERS > 1
    case 1u: { r = textureSampleLevel(t, texSmp1, uv, layer, lod); }
#endif
#if TEX_SAMPLERS > 2
    case 2u: { r = textureSampleLevel(t, texSmp2, uv, layer, lod); }
#endif
#if TEX_SAMPLERS > 3
    case 3u: { r = textureSampleLevel(t, texSmp3, uv, layer, lod); }
#endif
#if TEX_SAMPLERS > 4
    case 4u: { r = textureSampleLevel(t, texSmp4, uv, layer, lod); }
#endif
#if TEX_SAMPLERS > 5
    case 5u: { r = textureSampleLevel(t, texSmp5, uv, layer, lod); }
#endif
#if TEX_SAMPLERS > 6
    case 6u: { r = textureSampleLevel(t, texSmp6, uv, layer, lod); }
#endif
#if TEX_SAMPLERS > 7
    case 7u: { r = textureSampleLevel(t, texSmp7, uv, layer, lod); }
#endif
    default: { r = textureSampleLevel(t, texSmp0, uv, layer, lod); }
  }
  return r;
#else
  switch si {
#if TEX_SAMPLERS > 1
    case 1u: { return textureSampleLevel(t, texSmp1, uv, layer, lod); }
#endif
#if TEX_SAMPLERS > 2
    case 2u: { return textureSampleLevel(t, texSmp2, uv, layer, lod); }
#endif
#if TEX_SAMPLERS > 3
    case 3u: { return textureSampleLevel(t, texSmp3, uv, layer, lod); }
#endif
#if TEX_SAMPLERS > 4
    case 4u: { return textureSampleLevel(t, texSmp4, uv, layer, lod); }
#endif
#if TEX_SAMPLERS > 5
    case 5u: { return textureSampleLevel(t, texSmp5, uv, layer, lod); }
#endif
#if TEX_SAMPLERS > 6
    case 6u: { return textureSampleLevel(t, texSmp6, uv, layer, lod); }
#endif
#if TEX_SAMPLERS > 7
    case 7u: { return textureSampleLevel(t, texSmp7, uv, layer, lod); }
#endif
    default: { return textureSampleLevel(t, texSmp0, uv, layer, lod); }
  }
#endif
}
#endif

#if MAT_VARIANTS
/// perf2 WP-8 (MAT_VARIANTS): the raw filtered texel of a VALID slot's (array, layer, sampler) word `info` at st (already
/// transformed); the array / sampler switches assign instead of returning (no tint_volatile_zero guards).
fn tex_fetch(info: u32, st: vec2f, lod: f32) -> vec4f {
#if TEX_ARRAYS > 0
  let ai = info & 0xfu;
  let layer = (info >> 4u) & 0x7ffu;
  let si = (info >> 15u) & 0x7u;
  var r: vec4f;
  switch ai {
#if TEX_ARRAYS > 1
    case 1u: { r = tex_sample_arr(texArr1, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 2
    case 2u: { r = tex_sample_arr(texArr2, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 3
    case 3u: { r = tex_sample_arr(texArr3, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 4
    case 4u: { r = tex_sample_arr(texArr4, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 5
    case 5u: { r = tex_sample_arr(texArr5, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 6
    case 6u: { r = tex_sample_arr(texArr6, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 7
    case 7u: { r = tex_sample_arr(texArr7, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 8
    case 8u: { r = tex_sample_arr(texArr8, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 9
    case 9u: { r = tex_sample_arr(texArr9, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 10
    case 10u: { r = tex_sample_arr(texArr10, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 11
    case 11u: { r = tex_sample_arr(texArr11, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 12
    case 12u: { r = tex_sample_arr(texArr12, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 13
    case 13u: { r = tex_sample_arr(texArr13, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 14
    case 14u: { r = tex_sample_arr(texArr14, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 15
    case 15u: { r = tex_sample_arr(texArr15, si, st, layer, lod); }
#endif
    default: { r = tex_sample_arr(texArr0, si, st, layer, lod); }
  }
  return r;
#else
  return vec4f(1.0);
#endif
}
#endif

/** Raw (still encoded) filtered texel of a slot at the given LOD. Invalid slot → vec4f(1). */
fn tex_sample_raw(s: TexSlot, uv: vec2f, lod: f32) -> vec4f {
#if MAT_VARIANTS
  if (!tex_slot_valid(s)) { return vec4f(1.0); }
  return tex_fetch(s.info, tex_slot_uv(s, uv), lod);
#elif TEX_ARRAYS > 0
  if (!tex_slot_valid(s)) { return vec4f(1.0); }
  let st = tex_slot_uv(s, uv);
  let ai = s.info & 0xfu;
  let layer = (s.info >> 4u) & 0x7ffu;
  let si = (s.info >> 15u) & 0x7u;
  switch ai {
#if TEX_ARRAYS > 1
    case 1u: { return tex_sample_arr(texArr1, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 2
    case 2u: { return tex_sample_arr(texArr2, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 3
    case 3u: { return tex_sample_arr(texArr3, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 4
    case 4u: { return tex_sample_arr(texArr4, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 5
    case 5u: { return tex_sample_arr(texArr5, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 6
    case 6u: { return tex_sample_arr(texArr6, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 7
    case 7u: { return tex_sample_arr(texArr7, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 8
    case 8u: { return tex_sample_arr(texArr8, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 9
    case 9u: { return tex_sample_arr(texArr9, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 10
    case 10u: { return tex_sample_arr(texArr10, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 11
    case 11u: { return tex_sample_arr(texArr11, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 12
    case 12u: { return tex_sample_arr(texArr12, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 13
    case 13u: { return tex_sample_arr(texArr13, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 14
    case 14u: { return tex_sample_arr(texArr14, si, st, layer, lod); }
#endif
#if TEX_ARRAYS > 15
    case 15u: { return tex_sample_arr(texArr15, si, st, layer, lod); }
#endif
    default: { return tex_sample_arr(texArr0, si, st, layer, lod); }
  }
#else
  return vec4f(1.0);
#endif
}

/** Filtered texel at LOD `lod`, sRGB-decoded AFTER filtering for sRGB slots (rgb only). Invalid slot → 1. */
fn tex_sample_lod(s: TexSlot, uv: vec2f, lod: f32) -> vec4f {
  let raw = tex_sample_raw(s, uv, lod);
  if ((s.info & TEX_SRGB) != 0u && tex_slot_valid(s)) { return tex_srgb_to_linear4(raw); }
  return raw;
}

/** Validation-mode lookup: bilinear at LOD 0 (gap-bsdf §9 item 2). */
fn tex_sample(s: TexSlot, uv: vec2f) -> vec4f { return tex_sample_lod(s, uv, 0.0); }

#if TEX_RESAMPLE
// ---- Interactive-path resample / mip kernel (textures-gpu.ts). Filters in LINEAR space for sRGB layers
// (decode → weighted average → re-encode), so mips hold sRGB-encoded bytes of linear averages.
// Each output texel averages tapsX × tapsY bilinear taps spread over its source footprint (clamp-to-edge);
// for an exact 2× reduction the taps land on texel centres, i.e. a 2×2 box filter.
struct TexResampleParams {
  srcSize: vec2u,
  dstSize: vec2u,
  srcLayerBase: u32,
  dstLayerBase: u32,
  tapsX: u32,
  tapsY: u32,
}
@group(0) @binding(0) var rsSrc: texture_2d_array<f32>;
@group(0) @binding(1) var rsDst: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> rsP: TexResampleParams;
@group(0) @binding(3) var<uniform> rsSrgb: array<vec4u, 16>;   // 1 bit per destination layer (≤ 2048)

fn rs_fetch(p: vec2i, layer: u32, srgb: bool) -> vec4f {
  let q = clamp(p, vec2i(0), vec2i(rsP.srcSize) - vec2i(1));
  let c = textureLoad(rsSrc, q, layer, 0);
  return select(c, tex_srgb_to_linear4(c), srgb);
}

fn rs_bilinear(x: vec2f, layer: u32, srgb: bool) -> vec4f {   // x in texel units, centres at i + 0.5
  let p = x - vec2f(0.5);
  let i = floor(p);
  let f = p - i;
  let ii = vec2i(i);
  let a = rs_fetch(ii, layer, srgb);
  let b = rs_fetch(ii + vec2i(1, 0), layer, srgb);
  let c = rs_fetch(ii + vec2i(0, 1), layer, srgb);
  let d = rs_fetch(ii + vec2i(1, 1), layer, srgb);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

@compute @workgroup_size(8, 8, 1)
fn tex_resample(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= rsP.dstSize.x || gid.y >= rsP.dstSize.y) { return; }
  let dl = rsP.dstLayerBase + gid.z;
  let sl = rsP.srcLayerBase + gid.z;
  let srgb = ((rsSrgb[dl >> 7u][(dl >> 5u) & 3u] >> (dl & 31u)) & 1u) != 0u;
  let scale = vec2f(rsP.srcSize) / vec2f(rsP.dstSize);
  let taps = vec2f(f32(rsP.tapsX), f32(rsP.tapsY));
  var acc = vec4f(0.0);
  for (var ty = 0u; ty < rsP.tapsY; ty++) {
    for (var tx = 0u; tx < rsP.tapsX; tx++) {
      let pos = (vec2f(gid.xy) + (vec2f(f32(tx), f32(ty)) + vec2f(0.5)) / taps) * scale;
      acc += rs_bilinear(pos, sl, srgb);
    }
  }
  acc /= taps.x * taps.y;
  var outc = acc;
  if (srgb) { outc = vec4f(tex_linear_to_srgb(acc.r), tex_linear_to_srgb(acc.g), tex_linear_to_srgb(acc.b), acc.a); }
  textureStore(rsDst, vec2i(gid.xy), dl, outc);
}
#endif

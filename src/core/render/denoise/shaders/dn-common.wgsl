// A-SVGF-lite denoiser, shared declarations (docs/decisions/denoiser.md). Mirror: render/denoise/layout.ts
// (DN_PARAMS_SIZE, DNF_*, DN_VIEW, packDnParams). These shaders live outside src/core/shaders on purpose (DN10: the
// M4/M5 validation cache keys hash every WGSL file there); they include the shared sources by their usual keys.
//   G0  0 FrameUniforms (cam, prevCam, resolution, flags) · 1 DnParams
//   G1  pass bindings (each pass file)
//   G2  empty
//   G3  debug (DebugResources.layout): views 520–535
#include "common/frame.wgsl"
#include "common/nan.wgsl"
#include "debug/debug-common.wgsl"

struct DnParams {          // 96 B
  size: vec2u,             //  0  W, H (internal resolution)
  tiles: vec2u,            //  8  ⌈W/8⌉, ⌈H/8⌉
  flags: u32,              // 16  DNF_*
  nMax: f32,               // 20  history length cap (64)
  alphaMin: f32,           // 24  α_min (0.2)
  lambda0: f32,            // 28  λ₀ of the ramp
  lambda1: f32,            // 32  λ₁ of the ramp
  sigmaZ: f32,             // 36  σ_z (1)
  sigmaN: f32,             // 40  σ_n (128)
  sigmaL: f32,             // 44  σ_l (4)
  tsBase: u32,             // 48  global arena word of tState[0] (64 + 6·P·NS_alloc)
  resPlanes: u32,          // 52  reservoir planes per record (10)
  sigmaA: f32,             // 56  σ_a of the albedo edge stop (Changelog DN-5; 0 = off)
  varCorr: f32,            // 60  K of the variance of the integrated colour (Changelog DN-4; 0 = sample variance)
  nMaxT: f32,              // 64  history cap of the output resolve (Changelog DN-6)
  sinceChange: u32,        // 68  frames since the last lighting change (0 = this frame; the resolve's dynamic window)
  invRadius: u32,          // 72  tile radius of the inverse family's window (DN-10; forward: 1 = 3×3)
  lumMinN: f32,            // 76  the luminance stop applies from this colour-history length on (DN-12; 0 = always)
  lumPre: u32,             // 80  luminance-guide prefilter radius (DN-13: 0 = the tap's own value, 1 = 3×3, 2 = 5×5)
  pad3: u32, pad4: u32, pad5: u32,
}
@group(0) @binding(1) var<uniform> dn: DnParams;

const DNF_RESET: u32 = 1u;         // full history reset this frame
const DNF_LAMBDA: u32 = 2u;        // the change bits open the gate: a light or env change between t−1 and t
const DNF_HAS_L1: u32 = 4u;        // the input has a separate length-1 term (ReSTIR rsL1); PT: L1 = 0
const DNF_FW: u32 = 8u;            // write lum(F·W) of the final reservoir into dnMom.a (ReSTIR)
const DNF_INVERSE: u32 = 16u;      // inverse gradient pairs (res[w] still holds the temporal output)
const DNF_GRADIENT: u32 = 32u;     // the gradient passes ran this frame (dnLambda holds this frame's λ)
const DNF_LAMBDA_CAM: u32 = 64u;   // option gradientOnCamera: also use λ on camera-only frames
const DNF_NO_RESOLVE: u32 = 128u;  // the output resolve is off (dn_resolve copies; Changelog DN-6)
const DNF_GUIDE: u32 = 256u;       // à-trous luminance stop on the converged previous output (static ≥ 8 frames; DN-9)

// Debug view ids (render/denoise/layout.ts DN_VIEW)
const DNV_VARIANCE: u32 = 520u;
const DNV_HISTORY: u32 = 521u;
const DNV_ALPHA: u32 = 522u;
const DNV_LAMBDA: u32 = 523u;
const DNV_DEMOD: u32 = 524u;
const DNV_INTEGRATED: u32 = 525u;
const DNV_REPROJ: u32 = 526u;
const DNV_PAIRS: u32 = 527u;
const DNV_LEVEL0: u32 = 530u;
const DNV_DEMOD_FACTOR: u32 = 540u;
const DNV_ALB_ACCUM: u32 = 541u;
const DNV_DEMOD_CHECK: u32 = 542u;

const DN_FP16_MAX: f32 = 65504.0;

/// Demodulation factor a′ (denoiser.md §3, Changelog DN-3): max(albedo, 0.02) + F0 per channel (F0 = 0.04, the
/// dielectric Fresnel at normal incidence: the white specular term of a dielectric is not inflated in its low-albedo
/// channels), 1 for (near-)black albedo.
const DN_F0: f32 = 0.04;
fn dn_demod_factor(albedo: vec3f) -> vec3f {
  if (!(max(albedo.x, max(albedo.y, albedo.z)) >= 0.02)) { return vec3f(1.0); }
  return max(albedo, vec3f(0.02)) + vec3f(DN_F0);
}

/// Octahedral encoding of a unit vector to [-1, 1]² (stored as 2 × 16 snorm).
fn dn_oct_encode(n: vec3f) -> vec2f {
  let a = abs(n.x) + abs(n.y) + abs(n.z);
  var p = n.xy / max(a, 1e-20);
  if (n.z < 0.0) {
    let s = select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
    p = (vec2f(1.0) - abs(p.yx)) * s;
  }
  return p;
}
fn dn_oct_decode(e: vec2f) -> vec3f {
  var v = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
  let t = max(-v.z, 0.0);
  v = vec3f(v.x + select(t, -t, v.x >= 0.0), v.y + select(t, -t, v.y >= 0.0), v.z);
  return normalize(v);
}
/// Guide texel (rg32uint): x = f32 bits of the distance to the frame's camera (0 = background), y = oct 2×16 snorm n_s.
fn dn_guide_pack(dist: f32, n: vec3f) -> vec2u { return vec2u(bitcast<u32>(dist), pack2x16snorm(dn_oct_encode(n))); }
fn dn_guide_dist(g: vec2u) -> f32 { return bitcast<f32>(g.x); }
fn dn_guide_normal(g: vec2u) -> vec3f { return dn_oct_decode(unpack2x16snorm(g.y)); }

fn dn_in_image(c: vec2i) -> bool { return c.x >= 0 && c.y >= 0 && c.x < i32(dn.size.x) && c.y < i32(dn.size.y); }
fn dn_flag(f: u32) -> bool { return (dn.flags & f) != 0u; }
/// x ≥ 0 rounded to the nearest binary16 value (ties to even), returned as f32 (Changelog DN-7): Metal converts f32 to
/// f16 on rgba16float stores toward zero, which biased every history stored in fp16 low (≈ −ulp/2 per store; a 1/n
/// accumulation drifts by ≈ −(n/2)·ulp/2). A value that is already binary16 converts exactly under any rounding mode.
/// Exact: round-half-to-even on the f32 bit pattern (no log2 / exp2, which fast math does not keep exact).
fn dn_rn16(x: f32) -> f32 {
  if (!(x > 0.0)) { return 0.0; }
  if (x < 6.1035156e-5) { return round(x * 16777216.0) * 5.9604645e-8; }   // binary16 subnormals: k·2^-24 (exact scaling)
  let b = bitcast<u32>(x);                               // normal: keep 10 of the 23 mantissa bits, ties to even
  return bitcast<f32>((b + 0xFFFu + ((b >> 13u) & 1u)) & 0xFFFFE000u);
}

/// A finite non-negative f32 clamped to the fp16 range and rounded to nearest (rgba16float storage), else 0.
fn dn_fp16(x: f32) -> f32 { return dn_rn16(select(0.0, clamp(x, 0.0, DN_FP16_MAX), is_finite(x))); }
fn dn_fp16v(v: vec3f) -> vec3f { return vec3f(dn_fp16(v.x), dn_fp16(v.y), dn_fp16(v.z)); }

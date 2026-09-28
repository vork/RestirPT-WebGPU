// Resolve: exposure + view transform (display encoding) and the debug-view false-colour resolve.
// Tonemap functions return DISPLAY-ENCODED values in [0,1] for a non-sRGB (bgra8unorm) target.
// With RESOLVE_DEBUG_ENTRY this module also provides the `debug_resolve` compute entry (AOV plane -> debugOut).
#include "common/math.wgsl"
#include "common/nan.wgsl"

const TONEMAP_STANDARD: u32 = 0u;  // Blender 'Standard' view transform: clamp + sRGB piecewise OETF (exact)
const TONEMAP_AGX: u32 = 1u;       // AgX-style approximation (NOT Blender's AgX LUT; eyeballing only)
const TONEMAP_ACES: u32 = 2u;      // ACES filmic fit (Narkowicz 2015) + sRGB OETF (approximation)
const TONEMAP_RAW: u32 = 3u;       // no view transform: linear values written as-is (clamped)

/// IEC 61966-2-1 sRGB OETF (inverse EOTF), per channel on [0,1].
fn srgb_oetf(x: vec3f) -> vec3f {
  let c = clamp(x, vec3f(0.0), vec3f(1.0));
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

fn agx_contrast(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn tonemap_agx(c: vec3f) -> vec3f {
  let inset = mat3x3f(
    vec3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
    vec3f(0.0784335999999992, 0.878468636469772, 0.0784336),
    vec3f(0.0792237451477643, 0.0791661274605434, 0.879142973793104));
  let outset = mat3x3f(
    vec3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
    vec3f(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
    vec3f(-0.0990297440797205, -0.0989611768448433, 1.15107367264116));
  let min_ev = -12.47393;
  let max_ev = 4.026069;
  var v = inset * max(c, vec3f(1e-10));
  v = (clamp(log2(v), vec3f(min_ev), vec3f(max_ev)) - min_ev) / (max_ev - min_ev);
  return clamp(outset * agx_contrast(v), vec3f(0.0), vec3f(1.0));
}

fn tonemap_aces(c: vec3f) -> vec3f {
  let x = max(c, vec3f(0.0));
  return srgb_oetf((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

/// Scene-linear radiance -> display-encoded colour. `exposure` is the linear multiplier 2^EV.
fn resolve_display(linear: vec3f, exposure: f32, tonemap: u32) -> vec3f {
  let c = linear * exposure;
  switch (tonemap) {
    case 1u: { return tonemap_agx(c); }
    case 2u: { return tonemap_aces(c); }
    case 3u: { return clamp(c, vec3f(0.0), vec3f(1.0)); }
    default: { return srgb_oetf(c); }
  }
}

#if RESOLVE_DEBUG_ENTRY
#include "debug/debug-common.wgsl"

// AOV plane -> false colour -> debugOut (display-encoded). Also records the raw AOV value at the probe pixel.
@compute @workgroup_size(8, 8, 1)
fn debug_resolve(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.xy;
  if (any(p >= dbg.size)) { return; }
  let v = dbgBuf.aov[p.y * dbg.size.x + p.x];
  let c = debug_false_colour(v, dbg.kind, dbg.rangeMin, dbg.rangeMax, dbg.flags);
  textureStore(debugOut, p, vec4f(c, 1.0));
  probe_record(p, PROBE_TAG_AOV, v);
}
#endif

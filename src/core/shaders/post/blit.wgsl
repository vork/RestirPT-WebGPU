// Present: fullscreen triangle that upscales the internal-resolution image to the canvas (plan §1.8, §1.10).
// Beauty: linear 'color' -> resolve_display (exposure + view transform) per source texel, then nearest / bilinear /
// bicubic (M8: Catmull-Rom with an anti-ringing clamp, the app default).
// Debug: debugOut already holds display-encoded false colour. Output target: bgra8unorm (no hardware sRGB).
// perf2 WP-7g (BLIT_PRETONED, the app default): the beauty is tonemapped once per internal texel by post/tonemap.wgsl
// into dispTex; bicubic then takes 9 bilinear taps of it (Catmull-Rom's middle weight pairs merged into one tap per
// axis) plus the 2 × 2 anti-ringing clamp, and a 1:1 present reads one texel (every filter is the identity there).
#include "post/resolve.wgsl"

struct BlitParams {
  srcSize: vec2u,
  dstSize: vec2u,
  exposure: f32,       // 2^EV
  tonemap: u32,        // TONEMAP_*
  flags: u32,          // BLIT_* bits
  split: f32,          // A|B split position in [0,1] (left = debug)
  probePixel: vec2u,
  _pad: vec2u,
}

const BLIT_BILINEAR: u32 = 1u;
const BLIT_DEBUG: u32 = 2u;
const BLIT_SPLIT: u32 = 4u;
const BLIT_NONFINITE: u32 = 8u;
const BLIT_PROBE: u32 = 16u;
const BLIT_BICUBIC: u32 = 32u;
const BLIT_PRETONED: u32 = 64u;   // perf2 WP-7g: the beauty comes from dispTex (tonemapped once)

@group(0) @binding(0) var colorTex: texture_2d<f32>;
@group(0) @binding(1) var debugTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> P: BlitParams;
@group(0) @binding(3) var dispTex: texture_2d<f32>;   // WP-7g: display-encoded beauty (internal resolution)
@group(0) @binding(4) var dispSmp: sampler;           // WP-7g: bilinear, clamp-to-edge

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn beauty_texel(p: vec2i) -> vec3f {
  if ((P.flags & BLIT_PRETONED) != 0u) { return textureLoad(dispTex, p, 0).rgb; }
  let c = textureLoad(colorTex, p, 0).rgb;
  if ((P.flags & BLIT_NONFINITE) != 0u && !all_finite3(c)) { return vec3f(1.0, 0.0, 1.0); }
  return resolve_display(c, P.exposure, P.tonemap);
}

fn debug_texel(p: vec2i) -> vec3f { return textureLoad(debugTex, p, 0).rgb; }

fn fetch_src(p: vec2i, dbgView: bool) -> vec3f {
  let q = clamp(p, vec2i(0), vec2i(P.srcSize) - 1);
  if (dbgView) { return debug_texel(q); }
  return beauty_texel(q);
}

/// M8 (m8-perf.md §10): Catmull-Rom (4 × 4 display-space texels) clamped to the range of the 2 × 2 nearest texels
/// (FSR-style anti-ringing: no halos at HDR edges or fireflies), sharper than bilinear on upscaled internal resolution.
fn catmull_rom_w(t: f32) -> vec4f {
  let t2 = t * t;
  let t3 = t2 * t;
  return vec4f(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}
fn sample_bicubic(s: vec2f, dbgView: bool) -> vec3f {
  let t = s - 0.5;
  let i0 = vec2i(floor(t)) - vec2i(1);
  let f = t - floor(t);
  let wx = catmull_rom_w(f.x);
  let wy = catmull_rom_w(f.y);
  var acc = vec3f(0.0);
  var lo = vec3f(1e30);
  var hi = vec3f(-1e30);
  for (var j = 0; j < 4; j++) {
    var row = vec3f(0.0);
    for (var i = 0; i < 4; i++) {
      let c = fetch_src(i0 + vec2i(i, j), dbgView);
      row += wx[i] * c;
      if ((i == 1 || i == 2) && (j == 1 || j == 2)) { lo = min(lo, c); hi = max(hi, c); }
    }
    acc += wy[j] * row;
  }
  return clamp(acc, lo, hi);
}

/// WP-7g: the same Catmull-Rom from 9 bilinear taps of dispTex: per axis the outer texels b − 1 and b + 2 (b = floor(s −
/// ½)) at their centres and one tap between b and b + 1 at offset w2 / (w1 + w2) (w1, w2 > 0 for Catmull-Rom), weighted
/// w1 + w2. Clamped to the 2 × 2 nearest texels as sample_bicubic.
fn sample_bicubic9(s: vec2f) -> vec3f {
  let t = s - 0.5;
  let b = floor(t);
  let f = t - b;
  let wx = catmull_rom_w(f.x);
  let wy = catmull_rom_w(f.y);
  let mx = wx.y + wx.z;
  let my = wy.y + wy.z;
  let inv = 1.0 / vec2f(P.srcSize);
  let x = vec3f(b.x - 0.5, b.x + 0.5 + wx.z / mx, b.x + 2.5) * inv.x;   // texel centre i at i + 0.5
  let y = vec3f(b.y - 0.5, b.y + 0.5 + wy.z / my, b.y + 2.5) * inv.y;
  let ax = vec3f(wx.x, mx, wx.w);
  let ay = vec3f(wy.x, my, wy.w);
  var acc = vec3f(0.0);
  for (var j = 0; j < 3; j++) {
    var row = vec3f(0.0);
    for (var i = 0; i < 3; i++) { row += ax[i] * textureSampleLevel(dispTex, dispSmp, vec2f(x[i], y[j]), 0.0).rgb; }
    acc += ay[j] * row;
  }
  let i0 = vec2i(b);
  let hiC = vec2i(P.srcSize) - 1;
  let c00 = textureLoad(dispTex, clamp(i0, vec2i(0), hiC), 0).rgb;
  let c10 = textureLoad(dispTex, clamp(i0 + vec2i(1, 0), vec2i(0), hiC), 0).rgb;
  let c01 = textureLoad(dispTex, clamp(i0 + vec2i(0, 1), vec2i(0), hiC), 0).rgb;
  let c11 = textureLoad(dispTex, clamp(i0 + vec2i(1, 1), vec2i(0), hiC), 0).rgb;
  return clamp(acc, min(min(c00, c10), min(c01, c11)), max(max(c00, c10), max(c01, c11)));
}

fn sample_src(s: vec2f, dbgView: bool) -> vec3f {
  if (all(P.srcSize == P.dstSize) && (P.flags & BLIT_PRETONED) != 0u) { return fetch_src(vec2i(floor(s)), dbgView); }   // WP-7g 1:1
  if ((P.flags & BLIT_BICUBIC) != 0u) {
    if (!dbgView && (P.flags & BLIT_PRETONED) != 0u) { return sample_bicubic9(s); }
    return sample_bicubic(s, dbgView);
  }
  if ((P.flags & BLIT_BILINEAR) == 0u) { return fetch_src(vec2i(floor(s)), dbgView); }
  let t = s - 0.5;
  let i0 = vec2i(floor(t));
  let f = t - floor(t);
  let a = mix(fetch_src(i0, dbgView), fetch_src(i0 + vec2i(1, 0), dbgView), f.x);
  let b = mix(fetch_src(i0 + vec2i(0, 1), dbgView), fetch_src(i0 + vec2i(1, 1), dbgView), f.x);
  return mix(a, b, f.y);
}

@fragment
fn fs_blit(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let dst = vec2f(P.dstSize);
  let src = vec2f(P.srcSize);
  let uv = pos.xy / dst;
  var dbgView = (P.flags & BLIT_DEBUG) != 0u;
  if (dbgView && (P.flags & BLIT_SPLIT) != 0u) {
    let sx = P.split * dst.x;
    if (abs(pos.x - sx) < 1.0) { return vec4f(1.0, 1.0, 1.0, 1.0); }
    dbgView = pos.x < sx;
  }
  var c = sample_src(uv * src, dbgView);
  if ((P.flags & BLIT_PROBE) != 0u) {
    // outline the probe pixel's footprint (at least 7 canvas px wide)
    let scale = dst / src;
    let centre = (vec2f(P.probePixel) + 0.5) * scale;
    let hw = max(0.5 * scale, vec2f(3.5));
    let d = abs(pos.xy - centre);
    let inside = all(d <= hw + 1.0);
    let edge = inside && (d.x > hw.x || d.y > hw.y);
    if (edge) { c = vec3f(1.0, 0.0, 1.0); }
  }
  return vec4f(c, 1.0);
}

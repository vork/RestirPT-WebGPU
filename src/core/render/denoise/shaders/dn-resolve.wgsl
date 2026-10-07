// dn_resolve (docs/decisions/denoiser.md Changelog DN-6): temporal resolve of the denoised, remodulated output. The
// per-frame à-trous output still follows the frame's jittered primary hits (its guides and its α_min share of new
// samples), so silhouettes and texels moved by up to ±½ px from frame to frame ("fizzy edges", R2 "wobble") while the
// app's progressive mean averaged the jitter away. The resolve accumulates the output itself:
//   static camera and no lighting change in the last 8 frames: identity reprojection of every pixel (hit or background,
//       as the progressive mean: the footprint is the same), α_t = 1/n_t, n_t ≤ nMaxT;
//   static camera, a lighting change within the last 8 frames: identity, n_t ≤ taaLightMax (32), clamped to the current
//       3×3 neighbourhood (YCoCg mean ± γσ, variance clipping, γ = taaGammaLight = 4), α_t = max(1/n_t, λ′) (the gradient
//       detects real changes; Changelog DN-16: with the former n_t ≤ 8, γ = 1 an animated light kept every edge pixel at
//       ≈ ¼ of its per-frame jitter flicker);
//   camera motion: the history reprojected by the motion vector of the closest hit in the 3×3 neighbourhood (TAA, no
//       geometric test), fetched with a Catmull-Rom filter (DN-16: bilinear blurred the history every frame, and the
//       jittered hit's own motion made silhouette pixels fetch from the foreground or the background side at random),
//       n_t ≤ taaCamMax (8), variance-clipped with γ = taaGammaCam (1), α_t = max(1/n_t, λ′);
//   reset / disocclusion: the current output.
// Background pixels accumulate too (identity with a static camera; in motion by the direction-based motion vector).
// G1: 0 dnOut (the remodulated output) · 1 dnTaa[prev] · 2 dnGeo[cur] · 3 dnGeo[prev] · 4 gbuf (ro) · 5 dnLambda
//     · 6 dnMom[cur] · 7 dnTaa[cur] (w) · 8 colour (w)
#include "denoise/dn-common.wgsl"
#include "passes/gbuffer.wgsl"

@group(1) @binding(0) var outTex: texture_2d<f32>;
@group(1) @binding(1) var taaPrev: texture_2d<f32>;
@group(1) @binding(2) var geoCur: texture_2d<u32>;
@group(1) @binding(3) var geoPrev: texture_2d<u32>;
@group(1) @binding(4) var<storage, read> gbuf: array<GBufTexel>;
@group(1) @binding(5) var lambdaTex: texture_2d<f32>;
@group(1) @binding(6) var momCur: texture_2d<f32>;
@group(1) @binding(7) var taaOut: texture_storage_2d<rgba32float, write>;   // f32: 1/n_t accumulation (DN-7)
@group(1) @binding(8) var colourOut: texture_storage_2d<$COLOR_FORMAT, write>;

const DN_TAA_DYN_MAX: f32 = 8.0;   // λ′ cuts the effective history length to ≤ 8 frames (DN-8)
const DNV_TAA_N: u32 = 528u;

fn rgb2ycocg(c: vec3f) -> vec3f { return vec3f(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b); }
fn ycocg2rgb(c: vec3f) -> vec3f { return vec3f(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

/// The output history at p: identity with a static camera; in motion the pixel centre reprojected by the G-buffer motion
/// vector (the jittered hit's prev − cur image position: hits by position, background by direction; DN-16: of the closest
/// hit in the 3×3 neighbourhood, so a silhouette follows the foreground), bilinear or Catmull-Rom (DN-16), without
/// a geometric test (TAA: a silhouette pixel's jittered hit flips between the two surfaces, so a depth / normal test
/// would reject half its history every frame; disocclusions are handled by the variance clipping). w = 0: none.
fn taa_history(p: vec2u, hit: bool, idx: u32) -> vec4f {
  if (dn_flag(DNF_RESET)) { return vec4f(0.0); }
  if ((frame.flags & FRAME_CAMERA_MOVED) == 0u) { return textureLoad(taaPrev, p, 0); }
  var mi = idx;
  if ((dn.taaFlags & DNT_DILATE) != 0u) {   // DN-16: the closest hit of the 3×3 neighbourhood decides the motion
    var zBest = 1e30;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let q = vec2i(p) + vec2i(dx, dy);
        if (!dn_in_image(q)) { continue; }
        let z = dn_guide_dist(textureLoad(geoCur, q, 0).xy);
        if (z > 0.0 && z < zBest) { zBest = z; mi = u32(q.y) * dn.size.x + u32(q.x); }
      }
    }
  }
  let g = gbuf[mi];
  if ((g.flags & GB_MOTION_VALID) == 0u) { return vec4f(0.0); }
  let sp = vec2f(p) + g.motion;
  let b = floor(sp);
  let f = sp - b;
  var acc = vec4f(0.0);
  var ws = 0.0;
  for (var j = 0u; j < 4u; j++) {
    let o = vec2i(i32(j & 1u), i32(j >> 1u));
    let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
    let c = vec2i(b) + o;
    if (dn_in_image(c) && w > 0.0) { acc += w * textureLoad(taaPrev, c, 0); ws += w; }
  }
  if (!(ws > 1e-3)) { return vec4f(0.0); }
  var h = acc / ws;
  if ((dn.taaFlags & DNT_CUBIC) != 0u && ws > 0.999) {
    // DN-16: Catmull-Rom (B = 0, C = ½) on the 4×4 texels around sp (edges clamped): a bilinear fetch blurs the history
    // by f(1 − f) px² per frame, which accumulates over the history length (soft silhouettes in motion).
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let wx = array<f32, 4>(w0.x, w1.x, w2.x, w3.x);
    let wy = array<f32, 4>(w0.y, w1.y, w2.y, w3.y);
    let mx = vec2i(dn.size) - vec2i(1);
    var cr = vec3f(0.0);
    for (var y = 0; y < 4; y++) {
      for (var x = 0; x < 4; x++) {
        let c = clamp(vec2i(b) + vec2i(x - 1, y - 1), vec2i(0), mx);
        cr += wx[x] * wy[y] * textureLoad(taaPrev, c, 0).rgb;
      }
    }
    h = vec4f(max(cr, vec3f(0.0)), h.a);
  }
  return h;
}

fn taa_lambda(p: vec2u) -> f32 {
  if (!dn_flag(DNF_GRADIENT)) { return 0.0; }
  let camMoved = (frame.flags & FRAME_CAMERA_MOVED) != 0u;
  if (!(dn_flag(DNF_LAMBDA) || (dn_flag(DNF_LAMBDA_CAM) && camMoved))) { return 0.0; }
  let tc = (vec2f(p) + vec2f(0.5)) / 8.0 - vec2f(0.5);
  let b = floor(tc);
  let f = tc - b;
  let mx = vec2i(dn.tiles) - vec2i(1);
  let c0 = clamp(vec2i(b), vec2i(0), mx);
  let c1 = clamp(vec2i(b) + vec2i(1), vec2i(0), mx);
  return mix(mix(textureLoad(lambdaTex, c0, 0).x, textureLoad(lambdaTex, vec2i(c1.x, c0.y), 0).x, f.x),
             mix(textureLoad(lambdaTex, vec2i(c0.x, c1.y), 0).x, textureLoad(lambdaTex, c1, 0).x, f.x), f.y);
}

@compute @workgroup_size(8, 8, 1)
fn dn_resolve(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.xy;
  if (any(p >= dn.size)) { return; }
  let idx = p.y * dn.size.x + p.x;
  let cur = textureLoad(outTex, p, 0).rgb;
  let hit = dn_guide_dist(textureLoad(geoCur, p, 0).xy) > 0.0;
  let camMoved = (frame.flags & FRAME_CAMERA_MOVED) != 0u;
  let dynamic = camMoved || dn.sinceChange < 8u;
  var h = taa_history(p, hit, idx);
  let lp = clamp((taa_lambda(p) - dn.lambda0) / max(dn.lambda1 - dn.lambda0, 1e-6), 0.0, 1.0);
  var nMax = dn.nMaxT;
  if (camMoved) { nMax = min(nMax, dn.taaCamMax); } else if (dynamic) { nMax = min(nMax, dn.taaLightMax); }   // DN-16
  let hn = select(h.a, min(h.a, DN_TAA_DYN_MAX), lp > 0.0);   // λ′ cuts the effective length (DN-8)
  let nT = select(min(1.0 + (1.0 - lp) * hn, max(nMax, 1.0)), 1.0, dn_flag(DNF_NO_RESOLVE));
  var res = cur;
  if (h.a > 0.0 && nT > 1.0) {
    var hist = h.rgb;
    let gamma = select(dn.taaGammaLight, dn.taaGammaCam, camMoved);
    if (dynamic && gamma > 0.0) {                        // variance clipping against the current 3×3 neighbourhood
      var m1 = vec3f(0.0);
      var m2 = vec3f(0.0);
      var k = 0.0;
      for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
          let q = vec2i(p) + vec2i(dx, dy);
          if (!dn_in_image(q)) { continue; }
          let y = rgb2ycocg(textureLoad(outTex, q, 0).rgb);
          m1 += y; m2 += y * y; k += 1.0;
        }
      }
      let mu = m1 / k;
      let sg = sqrt(max(m2 / k - mu * mu, vec3f(0.0)));
      hist = ycocg2rgb(clamp(rgb2ycocg(hist), mu - gamma * sg, mu + gamma * sg));
    }
    res = mix(hist, cur, max(1.0 / nT, lp));
  }
  res = max(res, vec3f(0.0));
  textureStore(taaOut, p, vec4f(res, nT));
  textureStore(colourOut, p, vec4f(res, 1.0));
  debug_write1(p, DNV_TAA_N, nT);
}

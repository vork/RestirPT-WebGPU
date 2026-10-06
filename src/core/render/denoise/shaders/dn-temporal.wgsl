// dn_temporal (docs/decisions/denoiser.md §3, §4, §6): demodulation, reprojection with TD12's predicate (bilinear taps,
// 3×3 ring fallback), the gradient-driven history length / α, temporal accumulation of the demodulated colour and of
// the luminance moments (EMA mean and standard deviation), the guide of this frame, and lum(F·W) of the final reservoir
// for the next frame's gradient.
// G1: 0 gbuf (ro) · 1 input (rsFrame | the PT sample) · 2 L1 (rsL1 | zero) · 3 dnGeo[prev] · 4 dnHist[prev] · 5 dnMom[prev]
//     · 6 dnLambda (tile res) · 7 res[final] (ro, ReSTIR; a dummy otherwise) · 8 dnAtrous[0] (w) · 9 dnHist[cur] (w)
//     · 10 dnMom[cur] (w) · 11 dnGeo[cur] (w) · 12 dnAlb[prev] · 13 dnAlb[cur] (w) · 14 dnL1[prev] · 15 dnL1[cur] (w)
// Changelog DN-2: ā and L1 have their own history length n_a (dnAlb.a), independent of the gradient (albedo does not
// change with the lighting): α_a = 1/n_a with a static camera (the progressive mean of the jitter), max(1/n_a, 0.1) in
// motion; L1 blends with max(α_a, λ′).
// Changelog DN-1: a static camera reprojects by identity (static geometry: the pixel footprint is the same, only the
// jitter moves the sample, as for the progressive mean); the demodulation factor a′ is accumulated like the colour
// (dnAlb) and remodulates the output, so silhouettes and texture detail are anti-aliased over the jitter.
#include "denoise/dn-common.wgsl"
#include "passes/gbuffer.wgsl"

@group(1) @binding(0) var<storage, read> gbuf: array<GBufTexel>;
@group(1) @binding(1) var inputTex: texture_2d<f32>;
@group(1) @binding(2) var l1Tex: texture_2d<f32>;
@group(1) @binding(3) var geoPrev: texture_2d<u32>;
@group(1) @binding(4) var histPrev: texture_2d<f32>;
@group(1) @binding(5) var momPrev: texture_2d<f32>;
@group(1) @binding(6) var lambdaTex: texture_2d<f32>;
@group(1) @binding(7) var<storage, read> resFinal: array<vec4u>;
@group(1) @binding(8) var atrousOut: texture_storage_2d<rgba16float, write>;
@group(1) @binding(9) var histOut: texture_storage_2d<rgba16float, write>;
@group(1) @binding(10) var momOut: texture_storage_2d<rgba16float, write>;
@group(1) @binding(11) var geoOut: texture_storage_2d<rg32uint, write>;
@group(1) @binding(12) var albPrev: texture_2d<f32>;
@group(1) @binding(13) var albOut: texture_storage_2d<rgba32float, write>;   // f32: 1/n_a accumulation (DN-7)
@group(1) @binding(14) var l1Prev: texture_2d<f32>;
@group(1) @binding(15) var l1Out: texture_storage_2d<rgba32float, write>;     // f32: 1/n_a accumulation (DN-7)

const DN_REPROJ_BG: u32 = 0u;
const DN_REPROJ_FULL: u32 = 1u;
const DN_REPROJ_PARTIAL: u32 = 2u;
const DN_REPROJ_RING: u32 = 3u;
const DN_REPROJ_NONE: u32 = 4u;
const DN_REPROJ_RESET: u32 = 5u;

struct DnHistory { colour: vec3f, mu: f32, m2: f32, n: f32, w: f32, code: u32, alb: vec4f, l1: vec3f }

/// TD12 predicate on a previous-frame tap (denoiser.md §4): a previous hit, dot(n, n′) ≥ 0.5, |z − z′| ≤ 0.1·z′.
fn dn_tap_valid(c: vec2i, n: vec3f, z: f32) -> bool {
  if (!dn_in_image(c)) { return false; }
  let g = textureLoad(geoPrev, c, 0).xy;
  let zp = dn_guide_dist(g);
  if (!(zp > 0.0)) { return false; }
  if (!(dot(n, dn_guide_normal(g)) >= 0.5)) { return false; }
  return abs(z - zp) <= 0.1 * zp;
}

fn dn_accum_tap(h: ptr<function, DnHistory>, c: vec2i, w: f32) {
  let hc = textureLoad(histPrev, c, 0);
  let m = textureLoad(momPrev, c, 0);
  (*h).colour += w * hc.rgb;
  (*h).alb += w * textureLoad(albPrev, c, 0);
  (*h).l1 += w * textureLoad(l1Prev, c, 0).rgb;
  (*h).mu += w * m.x;
  (*h).m2 += w * (m.y * m.y + m.x * m.x);
  (*h).n += w * m.z;
  (*h).w += w;
}

/// Reprojected history of a hit pixel: x₁ back-projected with prevCam, bilinear taps around sp (pixel centres at
/// integers, tpick.wgsl's convention), else the 3×3 ring around round(sp), else none. Normalised.
fn dn_reproject(p: vec2u, pos: vec3f, n: vec3f) -> DnHistory {
  var h = DnHistory(vec3f(0.0), 0.0, 0.0, 0.0, 0.0, DN_REPROJ_NONE, vec4f(0.0), vec3f(0.0));
  if (dn_flag(DNF_RESET)) { h.code = DN_REPROJ_RESET; return h; }
  if ((frame.flags & FRAME_CAMERA_MOVED) == 0u) {        // static camera: identity (only the previous frame's hit is required)
    if (dn_guide_dist(textureLoad(geoPrev, p, 0).xy) > 0.0) { dn_accum_tap(&h, vec2i(p), 1.0); h.code = DN_REPROJ_FULL; }
    return h;
  }
  if (!(frame_view_depth(pos, frame.prevCam) >= 1e-12)) { return h; }
  let z = length(pos - frame.prevCam.camToWorld[3].xyz);
  let sp = frame_project(pos, frame.prevCam) - vec2f(0.5);
  let lim = vec2f(dn.size) + vec2f(2.0);
  if (!(sp.x >= -2.0 && sp.y >= -2.0 && sp.x <= lim.x && sp.y <= lim.y)) { return h; }
  let b = floor(sp);
  let f = sp - b;
  let c0 = vec2i(b);
  var valid = 0u;
  for (var j = 0u; j < 4u; j++) {
    let o = vec2i(i32(j & 1u), i32(j >> 1u));
    let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
    let c = c0 + o;
    if (dn_tap_valid(c, n, z)) { dn_accum_tap(&h, c, w); valid++; }
  }
  if (h.w >= 1e-3) {
    h.code = select(DN_REPROJ_PARTIAL, DN_REPROJ_FULL, valid == 4u);
  } else {
    h = DnHistory(vec3f(0.0), 0.0, 0.0, 0.0, 0.0, DN_REPROJ_NONE, vec4f(0.0), vec3f(0.0));
    let cr = vec2i(floor(sp + vec2f(0.5)));
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let c = cr + vec2i(dx, dy);
        if (dn_tap_valid(c, n, z)) { dn_accum_tap(&h, c, 1.0); }
      }
    }
    if (h.w > 0.0) { h.code = DN_REPROJ_RING; } else { return h; }
  }
  let inv = 1.0 / h.w;
  h.colour *= inv; h.mu *= inv; h.m2 *= inv; h.n *= inv; h.alb *= inv; h.l1 *= inv;
  return h;
}

/// λ at pixel p: bilinear between tile centres (8·t + 3.5) of the 3×3-tile λ texture.
fn dn_lambda(p: vec2u) -> f32 {
  if (!dn_flag(DNF_GRADIENT)) { return 0.0; }
  let camMoved = (frame.flags & FRAME_CAMERA_MOVED) != 0u;
  if (!(dn_flag(DNF_LAMBDA) || (dn_flag(DNF_LAMBDA_CAM) && camMoved))) { return 0.0; }
  let tc = (vec2f(p) + vec2f(0.5)) / 8.0 - vec2f(0.5);
  let b = floor(tc);
  let f = tc - b;
  let mx = vec2i(dn.tiles) - vec2i(1);
  let c0 = clamp(vec2i(b), vec2i(0), mx);
  let c1 = clamp(vec2i(b) + vec2i(1), vec2i(0), mx);
  let l00 = textureLoad(lambdaTex, c0, 0).x;
  let l10 = textureLoad(lambdaTex, vec2i(c1.x, c0.y), 0).x;
  let l01 = textureLoad(lambdaTex, vec2i(c0.x, c1.y), 0).x;
  let l11 = textureLoad(lambdaTex, c1, 0).x;
  return mix(mix(l00, l10, f.x), mix(l01, l11, f.x), f.y);
}

@compute @workgroup_size(8, 8, 1)
fn dn_temporal(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.xy;
  if (any(p >= dn.size)) { return; }
  let idx = p.y * dn.size.x + p.x;
  let g = gbuf[idx];
  var fw = 0.0;
  if (dn_flag(DNF_FW)) {
    let p0 = resFinal[idx * dn.resPlanes];
    fw = dn_fp16(luminance(bitcast<vec3f>(p0.yzw)) * bitcast<f32>(p0.x));
  }
  if ((g.flags & GB_HIT) == 0u) {     // background: never a destination; passed through by the last à-trous pass
    textureStore(geoOut, p, vec4u(0u));
    textureStore(momOut, p, vec4f(0.0, 0.0, 0.0, fw));
    textureStore(histOut, p, vec4f(0.0));
    textureStore(atrousOut, p, vec4f(0.0));
    textureStore(albOut, p, vec4f(0.0));
    textureStore(l1Out, p, vec4f(0.0));
    debug_write_code(p, DNV_REPROJ, DN_REPROJ_BG);
    return;
  }
  let L = textureLoad(inputTex, p, 0).rgb;
  let L1 = select(vec3f(0.0), textureLoad(l1Tex, p, 0).rgb, dn_flag(DNF_HAS_L1));
  let ad = dn_demod_factor(g.albedo);
  let c = dn_fp16v((L - L1) / ad);
  let l = luminance(c);
  let n = g.ns;
  let h = dn_reproject(p, g.pos, n);
  let lambda = dn_lambda(p);
  let lp = clamp((lambda - dn.lambda0) / max(dn.lambda1 - dn.lambda0, 1e-6), 0.0, 1.0);
  // Changelog DN-8: λ′ cuts the EFFECTIVE history length: beyond 1/α_min a longer n does not change α, so a partial cut
  // of n = 64 left α at α_min and the stale share decayed only at the α_min rate.
  let nIn = select(0.0, h.n, h.w > 0.0);
  let nCut = select(nIn, min(nIn, 1.0 / dn.alphaMin), lp > 0.0);
  let nNew = min(1.0 + (1.0 - lp) * nCut, dn.nMax);
  let alpha = max(max(dn.alphaMin, 1.0 / nNew), lp);
  let camMoved = (frame.flags & FRAME_CAMERA_MOVED) != 0u;
  let nA = min(1.0 + select(0.0, h.alb.a, h.w > 0.0), dn.nMax);
  let alphaA = select(1.0 / nA, max(1.0 / nA, 0.1), camMoved);
  var colour = c;
  var alb = ad;
  var l1 = L1;
  var mu = l;
  var vr = 0.0;
  if (h.w > 0.0) {
    colour = mix(h.colour, c, alpha);
    alb = mix(h.alb.rgb, ad, alphaA);
    l1 = mix(h.l1, L1, max(alphaA, lp));
    let mu0 = h.mu;
    let v0 = max(h.m2 - mu0 * mu0, 0.0);
    let d = l - mu0;
    mu = mu0 + alpha * d;
    vr = (1.0 - alpha) * (v0 + alpha * d * d);
  }
  colour = dn_fp16v(colour);
  let sigma = sqrt(max(vr, 0.0));
  let dist = length(g.pos - frame.cam.camToWorld[3].xyz);
  textureStore(geoOut, p, vec4u(dn_guide_pack(dist, n), 0u, 0u));
  textureStore(momOut, p, vec4f(dn_fp16(mu), dn_fp16(sigma), nNew, fw));
  textureStore(histOut, p, vec4f(colour, 0.0));
  // Changelog DN-4: the à-trous edge stops see the variance of the integrated colour, not of the input samples (SVGF):
  // an EMA of weight α keeps α/(2 − α) of the variance of independent samples; ReSTIR's temporal reuse correlates
  // successive frames, so K (varCorr, ≈ frames per independent sample) scales it back up: min(1, K·α/(2 − α)).
  let vScale = select(1.0, min(1.0, dn.varCorr * alpha / (2.0 - alpha)), dn.varCorr > 0.0);
  textureStore(atrousOut, p, vec4f(colour, dn_fp16(vr * vScale)));
  textureStore(albOut, p, vec4f(alb, nA));
  textureStore(l1Out, p, vec4f(max(l1, vec3f(0.0)), 0.0));
  debug_write3(p, DNV_DEMOD, c);
  debug_write3(p, DNV_INTEGRATED, colour);
  debug_write1(p, DNV_HISTORY, nNew);
  debug_write1(p, DNV_ALPHA, alpha);
  debug_write1(p, DNV_LAMBDA, lambda);
  debug_write_code(p, DNV_REPROJ, h.code);
  debug_write3(p, DNV_DEMOD_FACTOR, ad);
  debug_write3(p, DNV_ALB_ACCUM, alb);
  // the per-frame round trip: lum(c · a′) / lum(L − L1) = 1 wherever c is not clamped (the albedo is this sample's)
  let li = luminance(L - L1);
  debug_write1(p, DNV_DEMOD_CHECK, select(1.0, luminance(c * ad) / li, abs(li) > 1e-12));
}

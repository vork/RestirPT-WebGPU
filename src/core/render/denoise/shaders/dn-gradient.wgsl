// dn_gradient + dn_grad_filter (docs/decisions/denoiser.md §5, DN6): the ReSTIR temporal gradient. Per pixel, the
// same path evaluated under the lighting of frame t (a) and of frame t−1 (b), read from the temporal stage's tState
// (restir-temporal-api.md §2.8) after the ReSTIR frame:
//   forward  (history sample X_p at q′):  a = w̃_p / c_p = lum F_t(Y_p)·W_p·J_p (0 if the path lost its light, was
//            shadowed or became zero), b = lum(F·W) of the final reservoir of t−1 at q′ (stored by dn_temporal in dnMom.a)
//   inverse  (canonical X_c, s = c pixels after T4): a = w̃_c = lum F_c·W_c, b = w̃_c·π_p(X_c)/π_c (0 if X_c's light did
//            not exist / it was shadowed or zero at t−1); only while res[w] still holds the temporal output (≤ 1 round)
// Geometric failures (camera motion, footprint, lobe) give no pair. Per family, Δ = Σ(a − b), M = Σ max(a, b) (a firefly
// is scaled to M = 10⁴); the inverse pairs exist only on s = c pixels and are weighted by 1/P(s = c) = (w̃_c + w̃_p)/w̃_c
// (Changelog DN-2). Summed over 8×8 tiles in workgroup memory; dn_grad_filter turns 3×3 tiles into
// λ = max(|ΣΔ_f|/ΣM_f, |ΣΔ_i|/ΣM_i): forward pairs see the change on the support of p̂_{t−1} (removed / changed light),
// inverse pairs on the support of p̂_t (added light).
// DN-11: a third, sample-independent family: the change of the tile mean between this frame's raw estimate L − L1 and the
// remodulated history at q′ (dnHist[prev]·ā[prev]), less 3 standard errors of the window mean, so it ignores noise:
// λ_c = max(0, |m_cur − m_old| − 3·se)/max(m_cur, m_old). It sees any lighting change (an added light's indirect share,
// which the inverse family samples sparsely), and also a history that still lags.
//   dn_gradient     G1: 0 arena (ro words) · 1 res[w] (ro) · 2 dnMom[prev] · 3 dnGradTile (rgba32float, write) · 4 input
//                   · 5 L1 · 6 dnHist[prev] · 7 dnAlb[prev] · 8 dnGradTile2 (rgba32float, write: Σcur, Σold, Σcur², N)
//   dn_grad_filter  G1: 0 dnGradTile · 1 dnLambda (r32float, write)
#include "denoise/dn-common.wgsl"

#if DN_GRADIENT
@group(1) @binding(0) var<storage, read> arenaWords: array<u32>;
@group(1) @binding(1) var<storage, read> resW: array<vec4u>;
@group(1) @binding(2) var momPrev: texture_2d<f32>;
@group(1) @binding(3) var gradTile: texture_storage_2d<rgba32float, write>;
@group(1) @binding(4) var inputTex: texture_2d<f32>;
@group(1) @binding(5) var l1Tex: texture_2d<f32>;
@group(1) @binding(6) var histPrev: texture_2d<f32>;
@group(1) @binding(7) var albPrev: texture_2d<f32>;
@group(1) @binding(8) var gradTile2: texture_storage_2d<rgba32float, write>;

// tState words (restir/tframe.wgsl TSW_*; layout.ts TS_CONSTS) and flags, slot codes (restir/types.wgsl SC_*)
const TSW_QPRIME: u32 = 8u;  const TSW_CP: u32 = 9u;  const TSW_FWDCODE: u32 = 10u;  const TSW_FLAGS: u32 = 11u;
const TSW_WC: u32 = 13u;  const TSW_WP: u32 = 14u;  const TSW_INVCODE: u32 = 15u;  const TSW_PIRECOMP: u32 = 17u;
const TS_WORDS: u32 = 20u;
const TS_QVALID: u32 = 1u;  const TS_SEL_C: u32 = 32u;  const TS_INV_DONE: u32 = 128u;  const TS_EMPTY_OUT: u32 = 256u;
const TS_QPRIME_NONE: u32 = 0xFFFFFFFFu;
const SC_OK: u32 = 0u;  const SC_O0_LIGHT: u32 = 6u;  const SC_OCCLUDED: u32 = 10u;  const SC_ZERO: u32 = 11u;
const DN_M_CAP: f32 = 1e4;

var<workgroup> wgSum: array<vec4f, 64>;
var<workgroup> wgSum2: array<vec4f, 64>;

#if RS_TSTATE_SOA
// perf2 WP-6: word-major tState; DnParams word 26 (pad6) = the tState word stride P (denoiser.ts tsStride)
fn ts(ai: u32, w: u32) -> u32 { return arenaWords[dn.tsBase + w * dn.pad6 + ai]; }
#else
fn ts(ai: u32, w: u32) -> u32 { return arenaWords[dn.tsBase + TS_WORDS * ai + w]; }
#endif
fn tsf(ai: u32, w: u32) -> f32 { return bitcast<f32>(ts(ai, w)); }
/// The path lost its contribution for a lighting reason (removed light / zero pmf, newly occluded, zero).
fn dn_light_zero(code: u32) -> bool { let sc = code & 0xFFu; return sc == SC_O0_LIGHT || sc == SC_OCCLUDED || sc == SC_ZERO; }
fn dn_pos(x: f32) -> f32 { return select(0.0, x, is_finite(x) && x > 0.0); }

struct DnPair { df: f32, mf: f32, di: f32, mi: f32, bits: u32 }

/// DN-11 colour family of pixel p: (cur, old, cur², 1) or 0 (no q′ / no history at q′).
fn dn_colour(p: vec2u) -> vec4f {
  let ai = p.y * dn.size.x + p.x;
  if ((ts(ai, TSW_FLAGS) & TS_QVALID) == 0u) { return vec4f(0.0); }
  let qP = ts(ai, TSW_QPRIME);
  if (qP == TS_QPRIME_NONE || qP >= dn.size.x * dn.size.y) { return vec4f(0.0); }
  let qpx = vec2u(qP % dn.size.x, qP / dn.size.x);
  let a = textureLoad(albPrev, qpx, 0);
  if (!(a.a > 0.0)) { return vec4f(0.0); }               // no history there (background / reset)
  let cur = max(luminance(textureLoad(inputTex, p, 0).rgb - textureLoad(l1Tex, p, 0).rgb), 0.0);
  let old = luminance(textureLoad(histPrev, qpx, 0).rgb * a.rgb);
  let c = min(cur, DN_M_CAP);
  if (!(is_finite(c) && is_finite(old))) { return vec4f(0.0); }
  return vec4f(c, old, c * c, 1.0);
}

/// Gradient pairs of pixel p (atlas index = image index: E = 1, interactive).
fn dn_pairs(p: vec2u) -> DnPair {
  var r = DnPair(0.0, 0.0, 0.0, 0.0, 0u);
  let ai = p.y * dn.size.x + p.x;
  let flags = ts(ai, TSW_FLAGS);
  if ((flags & TS_QVALID) == 0u) { return r; }
  let qP = ts(ai, TSW_QPRIME);
  if (qP == TS_QPRIME_NONE || qP >= dn.size.x * dn.size.y) { return r; }
  // forward: X_p (the history sample) re-evaluated in frame t
  let fc = ts(ai, TSW_FWDCODE);
  let cP = tsf(ai, TSW_CP);
  let qpx = vec2u(qP % dn.size.x, qP / dn.size.x);
  let b = dn_pos(textureLoad(momPrev, qpx, 0).a);
  let wp = dn_pos(tsf(ai, TSW_WP));
  if ((fc & 0xFFu) == SC_OK && cP > 0.0) {
    let a = dn_pos(wp / cP);
    r.df = a - b; r.mf = max(a, b); r.bits |= 1u;
  } else if (dn_light_zero(fc)) {
    r.df = -b; r.mf = b; r.bits |= 1u;
  }
  // inverse: X_c (the canonical sample, s = c) evaluated in frame t−1
  if (dn_flag(DNF_INVERSE) && (flags & (TS_SEL_C | TS_INV_DONE)) == (TS_SEL_C | TS_INV_DONE) && (flags & TS_EMPTY_OUT) == 0u) {
    let ic = ts(ai, TSW_INVCODE);
    let a = dn_pos(tsf(ai, TSW_WC));
    var bi = -1.0;
    if ((ic & 0xFFu) == SC_OK) {
      let piC = luminance(bitcast<vec3f>(resW[ai * dn.resPlanes].yzw));
      if (piC > 0.0 && is_finite(piC)) { bi = dn_pos(a * tsf(ai, TSW_PIRECOMP) / piC); }
    } else if (dn_light_zero(ic)) {
      bi = 0.0;
    }
    if (bi >= 0.0 && a > 0.0) {
      let wsel = (a + wp) / a;                           // 1/P(s = c) of contribution MIS
      r.di = wsel * (a - bi); r.mi = wsel * max(a, bi); r.bits |= 2u;
    }
  }
  if (r.mf > DN_M_CAP) { let s = DN_M_CAP / r.mf; r.df *= s; r.mf = DN_M_CAP; }
  if (r.mi > DN_M_CAP) { let s = DN_M_CAP / r.mi; r.di *= s; r.mi = DN_M_CAP; }
  if (!(is_finite(r.df) && is_finite(r.mf))) { r.df = 0.0; r.mf = 0.0; }
  if (!(is_finite(r.di) && is_finite(r.mi))) { r.di = 0.0; r.mi = 0.0; }
  return r;
}

@compute @workgroup_size(8, 8, 1)
fn dn_gradient(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3u) {
  var v = vec4f(0.0);
  var v2 = vec4f(0.0);
  let p = gid.xy;
  if (all(p < dn.size)) {
    let r = dn_pairs(p);
    v = vec4f(r.df, r.mf, r.di, r.mi);
    v2 = dn_colour(p);
    debug_write_code(p, DNV_PAIRS, r.bits);
  }
  wgSum[li] = v;
  wgSum2[li] = v2;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (li < s) { wgSum[li] = wgSum[li] + wgSum[li + s]; wgSum2[li] = wgSum2[li] + wgSum2[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { textureStore(gradTile, wid.xy, wgSum[0]); textureStore(gradTile2, wid.xy, wgSum2[0]); }
}
#endif

#if DN_GRAD_FILTER
@group(1) @binding(0) var gradTileIn: texture_2d<f32>;
@group(1) @binding(1) var lambdaOut: texture_storage_2d<r32float, write>;
@group(1) @binding(2) var gradTile2In: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn dn_grad_filter(@builtin(global_invocation_id) gid: vec3u) {
  let t = vec2i(gid.xy);
  if (any(gid.xy >= dn.tiles)) { return; }
  // DN-10: the forward family over 3×3 tiles; the inverse family (sparse: only s = c pixels whose canonical sample uses
  // the changed light) over (2·invRadius + 1)² tiles
  var s = vec4f(0.0);
  var s2 = vec4f(0.0);
  let R = i32(max(dn.invRadius, 1u));
  for (var dy = -R; dy <= R; dy++) {
    for (var dx = -R; dx <= R; dx++) {
      let c = t + vec2i(dx, dy);
      if (c.x < 0 || c.y < 0 || c.x >= i32(dn.tiles.x) || c.y >= i32(dn.tiles.y)) { continue; }
      let v = textureLoad(gradTileIn, c, 0);
      if (abs(dx) <= 1 && abs(dy) <= 1) { s.x += v.x; s.y += v.y; s2 += textureLoad(gradTile2In, c, 0); }
      s.z += v.z; s.w += v.w;
    }
  }
  let lf = select(0.0, min(abs(s.x) / s.y, 1.0), s.y > 1e-8);
  let lv = select(0.0, min(abs(s.z) / s.w, 1.0), s.w > 1e-8);
  var lc = 0.0;
  if (s2.w >= 16.0) {                                    // DN-11 colour family
    let mc = s2.x / s2.w;
    let mo = s2.y / s2.w;
    let se = sqrt(max(s2.z / s2.w - mc * mc, 0.0) / s2.w);
    lc = select(0.0, min(max(abs(mc - mo) - 3.0 * se, 0.0) / max(mc, mo), 1.0), max(mc, mo) > 1e-8);
  }
  let lambda = max(max(lf, lv), lc);
  textureStore(lambdaOut, gid.xy, vec4f(lambda, 0.0, 0.0, 0.0));
}
#endif

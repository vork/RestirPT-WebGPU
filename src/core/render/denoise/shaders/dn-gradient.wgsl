// dn_gradient + dn_grad_filter (docs/decisions/denoiser.md §5, DN6): the ReSTIR temporal gradient. Per pixel, the
// same path evaluated under the lighting of frame t (a) and of frame t−1 (b), read from the temporal stage's tState
// (restir-temporal-api.md §2.8) after the ReSTIR frame:
//   forward  (history sample X_p at q′):  a = w̃_p / c_p = lum F_t(Y_p)·W_p·J_p (0 if the path lost its light, was
//            shadowed or became zero), b = lum(F·W) of the final reservoir of t−1 at q′ (stored by dn_temporal in dnMom.a)
//   inverse  (canonical X_c, s = c pixels after T4): a = w̃_c = lum F_c·W_c, b = w̃_c·π_p(X_c)/π_c (0 if X_c's light did
//            not exist / it was shadowed or zero at t−1); only while res[w] still holds the temporal output (≤ 1 round)
// Geometric failures (camera motion, footprint, lobe) give no pair. Δ = Σ(a − b), M = Σ max(a, b) (a firefly is scaled to
// M = 10⁴), summed over 8×8 tiles in workgroup memory; dn_grad_filter turns 3×3 tiles into λ = |ΣΔ| / ΣM.
//   dn_gradient     G1: 0 arena (ro words) · 1 res[w] (ro) · 2 dnMom[prev] · 3 dnGradTile (rg32float, write)
//   dn_grad_filter  G1: 0 dnGradTile · 1 dnLambda (r32float, write)
#include "denoise/dn-common.wgsl"

#if DN_GRADIENT
@group(1) @binding(0) var<storage, read> arenaWords: array<u32>;
@group(1) @binding(1) var<storage, read> resW: array<vec4u>;
@group(1) @binding(2) var momPrev: texture_2d<f32>;
@group(1) @binding(3) var gradTile: texture_storage_2d<rg32float, write>;

// tState words (restir/tframe.wgsl TSW_*; layout.ts TS_CONSTS) and flags, slot codes (restir/types.wgsl SC_*)
const TSW_QPRIME: u32 = 8u;  const TSW_CP: u32 = 9u;  const TSW_FWDCODE: u32 = 10u;  const TSW_FLAGS: u32 = 11u;
const TSW_WC: u32 = 13u;  const TSW_WP: u32 = 14u;  const TSW_INVCODE: u32 = 15u;  const TSW_PIRECOMP: u32 = 17u;
const TS_WORDS: u32 = 20u;
const TS_QVALID: u32 = 1u;  const TS_SEL_C: u32 = 32u;  const TS_INV_DONE: u32 = 128u;  const TS_EMPTY_OUT: u32 = 256u;
const TS_QPRIME_NONE: u32 = 0xFFFFFFFFu;
const SC_OK: u32 = 0u;  const SC_O0_LIGHT: u32 = 6u;  const SC_OCCLUDED: u32 = 10u;  const SC_ZERO: u32 = 11u;
const DN_M_CAP: f32 = 1e4;

var<workgroup> wgSum: array<vec2f, 64>;

fn ts(ai: u32, w: u32) -> u32 { return arenaWords[dn.tsBase + TS_WORDS * ai + w]; }
fn tsf(ai: u32, w: u32) -> f32 { return bitcast<f32>(ts(ai, w)); }
/// The path lost its contribution for a lighting reason (removed light / zero pmf, newly occluded, zero).
fn dn_light_zero(code: u32) -> bool { let sc = code & 0xFFu; return sc == SC_O0_LIGHT || sc == SC_OCCLUDED || sc == SC_ZERO; }
fn dn_pos(x: f32) -> f32 { return select(0.0, x, is_finite(x) && x > 0.0); }

struct DnPair { d: f32, m: f32, bits: u32 }

/// Gradient pairs of pixel p (atlas index = image index: E = 1, interactive).
fn dn_pairs(p: vec2u) -> DnPair {
  var r = DnPair(0.0, 0.0, 0u);
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
  if ((fc & 0xFFu) == SC_OK && cP > 0.0) {
    let a = dn_pos(tsf(ai, TSW_WP) / cP);
    r.d += a - b; r.m += max(a, b); r.bits |= 1u;
  } else if (dn_light_zero(fc)) {
    r.d -= b; r.m += b; r.bits |= 1u;
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
    if (bi >= 0.0) { r.d += a - bi; r.m += max(a, bi); r.bits |= 2u; }
  }
  if (r.m > DN_M_CAP) { let s = DN_M_CAP / r.m; r.d *= s; r.m = DN_M_CAP; }
  if (!(is_finite(r.d) && is_finite(r.m))) { r = DnPair(0.0, 0.0, 0u); }
  return r;
}

@compute @workgroup_size(8, 8, 1)
fn dn_gradient(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3u) {
  var v = vec2f(0.0);
  let p = gid.xy;
  if (all(p < dn.size)) {
    let r = dn_pairs(p);
    v = vec2f(r.d, r.m);
    debug_write_code(p, DNV_PAIRS, r.bits);
  }
  wgSum[li] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (li < s) { wgSum[li] = wgSum[li] + wgSum[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { textureStore(gradTile, wid.xy, vec4f(wgSum[0], 0.0, 0.0)); }
}
#endif

#if DN_GRAD_FILTER
@group(1) @binding(0) var gradTileIn: texture_2d<f32>;
@group(1) @binding(1) var lambdaOut: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn dn_grad_filter(@builtin(global_invocation_id) gid: vec3u) {
  let t = vec2i(gid.xy);
  if (any(gid.xy >= dn.tiles)) { return; }
  var s = vec2f(0.0);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let c = t + vec2i(dx, dy);
      if (c.x < 0 || c.y < 0 || c.x >= i32(dn.tiles.x) || c.y >= i32(dn.tiles.y)) { continue; }
      s += textureLoad(gradTileIn, c, 0).xy;
    }
  }
  let lambda = select(0.0, min(abs(s.x) / s.y, 1.0), s.y > 1e-8);
  textureStore(lambdaOut, gid.xy, vec4f(lambda, 0.0, 0.0, 0.0));
}
#endif

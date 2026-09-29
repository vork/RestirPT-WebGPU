// ReSTIR reservoir record, validation layout (restir-api.md §2.2 and appendix B.2, normative; math.md#reservoir-fields).
// array<vec4<u32>>, record i = planes i·10 … i·10+9 (160 B). The accessors below are the ONLY way passes read or write
// records; writers always write all ten planes (unused words 0, jDen 1). Mirror: render/restir/layout.ts (U-RES-1).
//   P0 (W, F.rgb) · P1 (seed.xy, flags, c) · P2 (rcA, rcB, rcC, jDen) · P3 (rcWi.xyz, aux) · P4 (rcRad.rgb, wSum) ·
//   P5 (endA, endB, endC, lobeHist) · P6 (sfxA, sfxB, sfxC, sfxFlags) · P7 (sfxDir.xyz, sfxT) · P8 (betaS.rgb, sfxP2) ·
//   P9 (nCand, selId, kMargin, endpointId)
// Bindings by define (group 2): RS_RES_IN_BINDING → resIn (read), RS_RES_OUT_BINDING → resOut (read_write); values are
// '<n>u' strings from resources.ts restirPassDefines() (Changelog A4).
#include "restir/types.wgsl"

// Plane indices of a record (§2.2)
const RP_WF: u32 = 0u;  const RP_SEED: u32 = 1u;  const RP_RC: u32 = 2u;  const RP_WI: u32 = 3u;  const RP_RAD: u32 = 4u;
const RP_END: u32 = 5u;  const RP_SFX0: u32 = 6u;  const RP_SFX1: u32 = 7u;  const RP_SFX2: u32 = 8u;  const RP_DIAG: u32 = 9u;
#if RS_RES_IN_BINDING
@group(2) @binding($RS_RES_IN_BINDING) var<storage, read> resIn: array<vec4u>;
fn resin_plane(i: u32, p: u32) -> vec4u { return resIn[i * RS_RES_PLANES + p]; }
#endif
#if RS_RES_OUT_BINDING
@group(2) @binding($RS_RES_OUT_BINDING) var<storage, read_write> resOut: array<vec4u>;
fn resout_plane(i: u32, p: u32) -> vec4u { return resOut[i * RS_RES_PLANES + p]; }
fn resout_set(i: u32, p: u32, v: vec4u) { resOut[i * RS_RES_PLANES + p] = v; }
#endif
fn rf_d(f: u32) -> u32 { return f & 0xFu; }
fn rf_k(f: u32) -> u32 { return (f >> RF_K_SHIFT) & 0xFu; }
fn rf_tech(f: u32) -> u32 { return (f >> RF_TECH_SHIFT) & 3u; }
fn rf_ep(f: u32) -> u32 { return (f >> RF_EP_SHIFT) & 7u; }
fn rf_lkm1(f: u32) -> u32 { return (f >> RF_LKM1_SHIFT) & 7u; }
fn rf_lk(f: u32) -> u32 { return (f >> RF_LK_SHIFT) & 7u; }
fn rf_pack(d: u32, k: u32, tech: u32, ep: u32, isDelta: bool, lkm1: u32, dkm1: bool, lk: u32, dk: bool, forced: bool) -> u32 {
  var f = (d & 0xFu) | ((k & 0xFu) << RF_K_SHIFT) | ((tech & 3u) << RF_TECH_SHIFT) | ((ep & 7u) << RF_EP_SHIFT)
        | ((lkm1 & 7u) << RF_LKM1_SHIFT) | ((lk & 7u) << RF_LK_SHIFT);   // mode bits 22–23 = 0 (Mode A, M4)
  if (isDelta) { f |= RF_ISDELTA; }
  if (dkm1) { f |= RF_DKM1; }
  if (dk) { f |= RF_DK; }
  if (forced) { f |= RF_FORCED; }
  return f;
}
fn res_empty(f: u32) -> bool { return rf_d(f) == 0u; }
/// Streaming RIS update (math.md#path-tree): true if the candidate with weight w replaces the selection.
fn ris_update(wSum: ptr<function, f32>, w: f32, u: f32) -> bool {
  if (!rs_pos_finite(w)) { return false; }
  *wSum = *wSum + w;
  return u * *wSum < w;
}

// ---- small accessors on plane values ----------------------------------------------------------------------------
fn rp_W(p0: vec4u) -> f32 { return bitcast<f32>(p0.x); }
fn rp_F(p0: vec4u) -> vec3f { return bitcast<vec3f>(p0.yzw); }
fn rp_flags(p1: vec4u) -> u32 { return p1.z; }
fn rp_c(p1: vec4u) -> f32 { return bitcast<f32>(p1.w); }

#if RS_RES_OUT_BINDING
/// Empty record (d = 0, W = 0, F = 0; c = 1 on a hit pixel) or background record (RF_BG, c = 0), all ten planes.
/// `seed` is the pixel's frame key (tree 0), kept for the dupmap identity / debugging.
fn res_write_empty(i: u32, seed: vec2u, bg: bool) {
  let one = bitcast<u32>(1.0);
  resout_set(i, RP_WF, vec4u(0u));
  resout_set(i, RP_SEED, vec4u(seed, select(0u, RF_BG, bg), select(one, 0u, bg)));
  resout_set(i, RP_RC, vec4u(RC_NONE, 0u, 0u, one));
  resout_set(i, RP_WI, vec4u(0u));
  resout_set(i, RP_RAD, vec4u(0u));
  resout_set(i, RP_END, vec4u(RC_NONE, 0u, 0u, RS_HIST_NONE));
  resout_set(i, RP_SFX0, vec4u(0u));
  resout_set(i, RP_SFX1, vec4u(0u));
  resout_set(i, RP_SFX2, vec4u(0u));
  resout_set(i, RP_DIAG, vec4u(0u, 0u, 0u, RC_NONE));
}
#endif

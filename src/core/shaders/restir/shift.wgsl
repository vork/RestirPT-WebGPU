// Hybrid shift T_{src→dst} (restir-api.md §3.7; math.md#jacobian, #offset-checks, #visibility, plan §2 rules 3–7).
// OWNER WP-B. P0 STUB (restir-api.md §1.4): shift_hybrid returns FAILED (SC_EMPTY_SRC for an empty source, else
// SC_O0_MISS), so the paired spatial reuse keeps the canonical sample (m_c = 1 for FAILED pairs): unbiased.
// Loaders are real. Compile-time switches: RS_REPLAY (replay compiled in), RS_SHIFT_TRACE (tests).
#include "restir/reservoir.wgsl"
#include "restir/endpoint.wgsl"
#include "restir/rc.wgsl"
#include "path/replay.wgsl"

struct ShiftSrc { empty: bool, flags: u32, seed: vec2u, F: vec3f, rc: vec3u, jDen: f32, rcWi: vec3f, aux: f32,
                  rcRad: vec3f, end: vec3u }
struct ShiftDst { valid: bool, prim: u32, bary: vec2f, thr: f32, camPos: vec3f }
struct ShiftOut { FJ: vec3f, J: f32, jNum: f32, code: u32 }   // code packed as §2.6; FJ, J = 0 unless SC_OK

/// Non-empty ∧ (k > 2 ∨ k = ∅): the offset needs its prefix replayed (y_{k−1} ≠ y₁).
fn res_needs_replay(flags: u32) -> bool {
  let k = rf_k(flags);
  return !res_empty(flags) && (k > 2u || k == 0u);
}

#if RS_RES_IN_BINDING
fn shift_src_load(ai: u32) -> ShiftSrc {
  var s: ShiftSrc;
  let p0 = resin_plane(ai, RP_WF);
  let p1 = resin_plane(ai, RP_SEED);
  let p2 = resin_plane(ai, RP_RC);
  let p3 = resin_plane(ai, RP_WI);
  let p4 = resin_plane(ai, RP_RAD);
  let p5 = resin_plane(ai, RP_END);
  s.flags = p1.z;
  s.empty = res_empty(s.flags);
  s.seed = p1.xy;
  s.F = rp_F(p0);
  s.rc = p2.xyz;
  s.jDen = bitcast<f32>(p2.w);
  s.rcWi = bitcast<vec3f>(p3.xyz);
  s.aux = bitcast<f32>(p3.w);
  s.rcRad = bitcast<vec3f>(p4.xyz);
  s.end = p5.xyz;
  return s;
}
#endif

#if RS_VBUF_BINDING && RS_GEO_BINDING
fn shift_dst_load(px: vec2u) -> ShiftDst {
  let vb = rs_vbuf(px);
  var d: ShiftDst;
  d.valid = vb.x != 0xFFFFFFFFu;
  d.prim = vb.x;
  d.bary = bitcast<vec2f>(vb.yz);
  d.thr = rs_geo(px).w;
  d.camPos = rs_cam_pos();
  return d;
}
#endif

/// STUB (P0): FAILED for every source.
fn shift_hybrid(src: ShiftSrc, dst: ShiftDst) -> ShiftOut {
  var o: ShiftOut;
  o.code = rs_slot_code(select(SC_O0_MISS, SC_EMPTY_SRC, src.empty), RCT_NONE, 0u, 0.0);
  return o;
}

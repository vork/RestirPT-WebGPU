// Temporal shift wrappers, loaders and MIS (restir-temporal-api.md §3.4, §3.6; math.md#temporal, #jacobian [M5]).
// OWNER T-B. tsrc_load / tdst_cur / tdst_prev / temporal_shift wrap the ONE hybrid shift (TD10) for T (t−1 → t) and
// T⁻¹ (t → t−1); J = J_rc·J_P (TD8); tmis_* live in restir/tmis.wgsl.
// P0 STUB: temporal_shift returns FAILED (SC_O0_MISS); tsrc_load marks every record undefined on TF_REFRESH frames
// (the refresh stub, §1.4). Both unbiased: every temporal candidate is FAILED (w̃_p = 0, π_p = 0).
#include "restir/tframe.wgsl"
#include "restir/tmis.wgsl"
#include "restir/shift.wgsl"

struct TSrc { base: ShiftSrc, undefinedLight: bool, jp: f32, xpEntry: u32 }
struct TShiftOut { F: vec3f, J: f32, jP: f32, code: u32 }   // F = F_dst(ȳ) (not F·J); J = J_rc·J_P; code §2.6 format

/// Plane p of record ai of resIn (fromBuf 0) or resOut (fromBuf 1), whichever the pass binds.
fn tres_plane(fromBuf: u32, ai: u32, p: u32) -> vec4u {
#if RS_RES_IN_BINDING && RS_RES_OUT_BINDING
  if (fromBuf == 0u) { return resin_plane(ai, p); }
  return resout_plane(ai, p);
#elif RS_RES_IN_BINDING
  return resin_plane(ai, p);
#elif RS_RES_OUT_BINDING
  return resout_plane(ai, p);
#else
  return vec4u(0u);
#endif
}

fn tsrc_load(ai: u32, fromBuf: u32, sfxDir: u32, fsTo: u32) -> TSrc {
  var t: TSrc;
  let p0 = tres_plane(fromBuf, ai, RP_WF);
  let p1 = tres_plane(fromBuf, ai, RP_SEED);
  t.base.flags = p1.z;
  t.base.empty = res_empty(t.base.flags);
  t.base.seed = p1.xy;
  t.base.F = rp_F(p0);
  let p2 = tres_plane(fromBuf, ai, RP_RC);
  t.base.rc = p2.xyz;
  t.base.jDen = bitcast<f32>(p2.w);
  let p3 = tres_plane(fromBuf, ai, RP_WI);
  t.base.rcWi = bitcast<vec3f>(p3.xyz);
  t.base.aux = bitcast<f32>(p3.w);
  t.base.rcRad = bitcast<vec3f>(tres_plane(fromBuf, ai, RP_RAD).xyz);
  t.base.end = tres_plane(fromBuf, ai, RP_END).xyz;
  t.jp = 1.0;
  t.xpEntry = RC_NONE;
  t.undefinedLight = rs_tf(TF_REFRESH);                  // P0 stub: refresh_record ⇒ SXS_UNDEF on every record
  return t;
}

#if RS_VBUF_BINDING && RS_GEO_BINDING
fn tdst_cur(q: RsPix) -> ShiftDst { return shift_dst_load(q.px); }
#endif

#if RS_VBUF_PREV_BINDING && RS_GEO_PREV_BINDING
fn tdst_prev(qPrime: u32) -> ShiftDst {
  let px = vec2u(qPrime % rsParams.atlasSize.x, qPrime / rsParams.atlasSize.x);
  let vb = rs_vbuf_prev(px);
  var d: ShiftDst;
  d.valid = vb.x != 0xFFFFFFFFu;
  d.prim = vb.x;
  d.bary = bitcast<vec2f>(vb.yz);
  d.thr = rs_geo_prev(px).w;
  d.camPos = lf_cam_pos(RS_FS_PREV);
  return d;
}
#endif

fn temporal_shift(s: TSrc, dst: ShiftDst) -> TShiftOut {
  var o: TShiftOut;
  o.code = rs_slot_code(select(SC_O0_MISS, SC_O0_LIGHT, s.undefinedLight), RCT_NONE, 0u, 0.0);
  return o;
}

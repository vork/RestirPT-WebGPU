// Temporal shift wrappers, loaders and tState stores (restir-temporal-api.md §3.4, §3.6; math.md#temporal,
// #jacobian, #light-changes [M5 additions]). OWNER T-B.
//   tsrc_load       a record of resIn (the history res[h], forward) or resOut (res[w], inverse) as the source of the ONE
//                   hybrid shift (TD10), loaded plane by plane (Metal Q1); on TF_REFRESH frames the refresh record of
//                   the direction (sfxOut, T-C) supplies the translated entry, J_P, the deep suffix radiance and the N1
//                   end visibility (§3.4 normative); a stale refresh record (gen ≠ frameGen) is a bug
//                   (RSC_T_PENDING_LEFT) and the record is treated as undefined (unbiased).
//   tdst_cur/prev   destination domains: (q, t) from the current G-buffer, fs = CUR; (q′, t−1) from the previous
//                   jittered V-buffer, thr_{t−1}[q′] and the previous camera, fs = PREV (plant N6: current camera).
//   temporal_shift  T and T⁻¹: undefinedLight ⇒ SC_O0_LIGHT; else shift_hybrid; for SC_OK, F = F_dst(ȳ) (never FJ/J)
//                   and J = J_rc·J_P (TD8: J_P applies to NEE-terminated records only, jp = 1 otherwise).
// tmis_* live in restir/tmis.wgsl (restir-temporal-api.md Changelog A-2).
#include "restir/tframe.wgsl"
#include "restir/tmis.wgsl"
#include "restir/shift.wgsl"

struct TSrc { base: ShiftSrc, undefinedLight: bool, jp: f32, xpEntry: u32 }
struct TShiftOut { F: vec3f, J: f32, jP: f32, code: u32 }   // F = F_dst(ȳ) (not F·J); J = J_rc·J_P; code §2.6 format

/// T-B local tState flag (restir-temporal-api.md Changelog B-3): T4's robust ID check failed (the translated-back
/// endpoint entry of Y_p differs from X_p's entry xpEntry); read by T3 phase B.
const TS_ROBUST_IDMIS: u32 = 32768u;

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

/// Path classes the refresh serves: deep (k ≤ d−2, k ≠ ∅) and N1 (k = d−1, NEE end) (math.md#reservoir-fields).
fn tcls_deep(f: u32) -> bool { let k = rf_k(f); return k != 0u && k + 2u <= rf_d(f); }
fn tcls_n1(f: u32) -> bool { let k = rf_k(f); return k != 0u && k + 1u == rf_d(f) && rf_tech(f) == RS_TECH_NEE; }

/// Record ai as a temporal shift source (§3.4 normative). sfxDir SFX_FWD ⇔ fsTo = CUR (history record under frame t),
/// SFX_INV ⇔ fsTo = PREV (current record under frame t−1).
fn tsrc_load(ai: u32, fromBuf: u32, sfxDir: u32, fsTo: u32) -> TSrc {
  var t: TSrc;
  let p1 = tres_plane(fromBuf, ai, RP_SEED);
  let f = p1.z;
  t.base.flags = f;
  t.base.empty = res_empty(f);
  t.base.seed = p1.xy;
  t.base.F = rp_F(tres_plane(fromBuf, ai, RP_WF));
  let p2 = tres_plane(fromBuf, ai, RP_RC);
  t.base.rc = p2.xyz;
  t.base.jDen = bitcast<f32>(p2.w);
  let p3 = tres_plane(fromBuf, ai, RP_WI);
  t.base.rcWi = bitcast<vec3f>(p3.xyz);
  t.base.aux = bitcast<f32>(p3.w);
  t.base.rcRad = bitcast<vec3f>(tres_plane(fromBuf, ai, RP_RAD).xyz);
  t.base.end = tres_plane(fromBuf, ai, RP_END).xyz;
  t.base.endOcc = false;
  t.jp = 1.0;
  t.undefinedLight = false;
  let nee = !t.base.empty && rf_tech(f) == RS_TECH_NEE;
  t.xpEntry = select(RC_NONE, t.base.end.x & RC_ENTRY_MASK, nee);
  if (t.base.empty || !rs_tf(TF_REFRESH)) { return t; }  // no refresh: entries unchanged, J_P = 1, V_end = 1, stored rcRad
  let r = sfx_load(sfxDir, ai);
  if (r.gen != rsTemporal.frameGen || (r.status & SXS_DONE) == 0u) {
#if RS_ARENA_RW
    rs_count(RSC_T_PENDING_LEFT, 1u);                    // a record the refresh should have served (bug; unbiased here)
#endif
    t.undefinedLight = true;
    return t;
  }
  t.undefinedLight = (r.status & SXS_UNDEF) != 0u;
  if (nee) {
    let e = RC_TAG_NEE | (r.entryTo & RC_ENTRY_MASK);
    t.base.end.x = e;
    if (rf_k(f) == rf_d(f)) { t.base.rc.x = e; }
    t.jp = r.jp;
  }
  if (tcls_deep(f)) { t.base.rcRad = r.rad; }              // N3 stale values come from the refresh (C-5, B-10)
  if (tcls_n1(f)) { t.base.endOcc = (r.status & SXS_VIS) == 0u; }
  return t;
}

#if RS_VBUF_BINDING && RS_GEO_BINDING
/// Destination (q, t): the M4 destination of the current G-buffer, fs = CUR.
fn tdst_cur(q: RsPix) -> ShiftDst { return shift_dst_load(q.px); }
#endif

#if RS_VBUF_PREV_BINDING && RS_GEO_PREV_BINDING
/// Destination (q′, t−1): the previous frame's jittered V-buffer hit and thr at atlas pixel qPrime, the previous camera
/// (plant TP_N6_CUR_CAM: the current camera), the previous light / env state (fs = PREV). Plant TP_N5_PIXEL_CENTRE: the
/// primary of q′ is re-traced through the pixel centre with the previous camera instead of the stored jittered hit.
fn tdst_prev(qPrime: u32) -> ShiftDst {
  let px = vec2u(qPrime % rsParams.atlasSize.x, qPrime / rsParams.atlasSize.x);
  let vb = rs_vbuf_prev(px);
  var d: ShiftDst;
  d.valid = vb.x != 0xFFFFFFFFu;
  d.prim = vb.x;
  d.bary = bitcast<vec2f>(vb.yz);
  d.thr = rs_geo_prev(px).w;
  d.camPos = lf_cam_pos(select(RS_FS_PREV, RS_FS_CUR, rs_tplant(TP_N6_CUR_CAM)));
  d.fs = RS_FS_PREV;
  if (rs_tplant(TP_N5_PIXEL_CENTRE)) {
    let local = vec2u(px.x % rsParams.memberSize.x, px.y % rsParams.memberSize.y);
    let m = frame.prevCam.camToWorld;
    let dc = frame_dir_cam(local, vec2f(0.5), frame.prevCam);
    let h = trace_closest(m[3].xyz, normalize(m[0].xyz * dc.x + m[1].xyz * dc.y + m[2].xyz * dc.z), FLT_MAX);
    d.valid = h.primId != BVH_MISS;
    d.prim = h.primId;
    d.bary = vec2f(h.u, h.v);
    if (d.valid) {
      let s = vertex_from_ids(h.primId, h.u, h.v, m[3].xyz);
      d.thr = primaryThreshold(m[3].xyz, s.pos, s.ng, rsParams.tau);
    }
  }
  return d;
}
#endif

/// T (t−1 → t) or T⁻¹ (t → t−1) of a loaded source into dst (§3.4).
fn temporal_shift(s: TSrc, dst: ShiftDst) -> TShiftOut {
  var o: TShiftOut;
  o.jP = s.jp;
  if (s.undefinedLight) {
    o.code = rs_slot_code(SC_O0_LIGHT, RCT_NONE, 0u, 0.0);
    return o;
  }
  let r = shift_hybrid(s.base, dst);
  o.code = r.code;
  if (rs_slot_code_sc(r.code) == SC_OK) {
    let J = r.J * s.jp;
    if (!rs_pos_finite(J) || !all_finite3(r.F)) {
      o.code = rs_slot_code(SC_NONFINITE, RCT_NONE, 0u, 0.0);
      return o;
    }
    o.F = r.F;
    o.J = J;
  }
  return o;
}

/// The refresh record of (dir, ai) refused the record for a light reason (Changelog B-9: entry missing in the target
/// frame or its realized pmf ≤ 0), not E2 class zeroing and not a stale record. Endpoint class changes by light edits
/// reduce to this in Mode A: a type or topology change is remove + add (§24), and emissive triangles are static.
fn tsfx_light_refused(dir: u32, ai: u32) -> bool {
  if (!rs_tf(TF_REFRESH)) { return false; }
  let r = sfx_load(dir, ai);
  return r.gen == rsTemporal.frameGen && (r.status & SXS_DONE) != 0u && (r.status & SXS_UNDEF) != 0u && (r.status & SXS_E2) == 0u;
}

/// An undefined shift code (outside the producibility set): not OK, ZERO, OCCLUDED or NONFINITE.
fn tshift_undefined(code: u32) -> bool {
  let sc = rs_slot_code_sc(code);
  return sc != SC_OK && sc != SC_ZERO && sc != SC_OCCLUDED && sc != SC_NONFINITE;
}

#if RS_ARENA_BINDING && RS_ARENA_RW
/// Store the forward shift of pixel q (tState TS0, TS2.z, TS3.x); FAILED J word unless SC_OK. Returns SC_*.
fn ts_store_fwd(q: u32, o: TShiftOut) -> u32 {
  let ok = rs_slot_code_sc(o.code) == SC_OK;
  let F = select(vec3f(0.0), o.F, ok);
  ts_storef(q, TSW_FWDF, F.x);
  ts_storef(q, TSW_FWDF + 1u, F.y);
  ts_storef(q, TSW_FWDF + 2u, F.z);
  ts_store(q, TSW_FWDJ, select(JW_FAILED, bitcast<u32>(o.J), ok));
  ts_store(q, TSW_FWDCODE, o.code);
  ts_storef(q, TSW_JP, o.jP);
  return rs_slot_code_sc(o.code);
}
/// Store the inverse shift of pixel q (tState TS1, TS3.w, TS4.y = π_p = lum(F)·J); returns π_p.
fn ts_store_inv(q: u32, o: TShiftOut) -> f32 {
  let ok = rs_slot_code_sc(o.code) == SC_OK;
  var pi = 0.0;
  if (ok) { pi = luminance(o.F) * o.J; }
  if (!rs_pos_finite(pi)) { pi = 0.0; }                  // bit test (Metal folds NaN compares)
  let F = select(vec3f(0.0), o.F, ok);
  ts_storef(q, TSW_INVF, F.x);
  ts_storef(q, TSW_INVF + 1u, F.y);
  ts_storef(q, TSW_INVF + 2u, F.z);
  ts_store(q, TSW_INVJ, select(JW_FAILED, bitcast<u32>(o.J), ok));
  ts_store(q, TSW_INVCODE, o.code);
  ts_storef(q, TSW_PIRECOMP, pi);
  return pi;
}
/// Counters of a forward shift (T1 inline and T2): FWD_OK; LIGHT_UNDEF (refresh: missing / zero-pmf entry, E2);
/// CLASS_UNDEF (an undefined shift code on a refresh frame, §9.3-6′).
fn tcount_fwd(code: u32) {
  let sc = rs_slot_code_sc(code);
  if (sc == SC_OK) { rs_count(RSC_T_FWD_OK, 1u); }
  if (sc == SC_O0_LIGHT) { rs_count(RSC_T_LIGHT_UNDEF, 1u); }
  else if (rs_tf(TF_REFRESH) && tshift_undefined(code) && sc != SC_EMPTY_SRC) { rs_count(RSC_T_CLASS_UNDEF, 1u); }
}
#endif

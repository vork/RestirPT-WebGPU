// T1 rs_t_classify (restir-temporal-api.md §3.6, §4.1; math.md#temporal [M5 addition]): per pixel q over the atlas.
// Clears tState[q] (every word, every pixel incl. background), picks q′ (restir/tpick.wgsl temporal_pixel: G-buffers
// and the pick stream only), reads c_prev of the history record res[h][q′] (c_p = min(cCap, c_prev), TD14) and runs the
// forward temporal shift T(X_p) inline for sources that need no replay (k = 2, or class L with d = 2); replay sources
// are queued on Q_f (q1) for T2. A valid q′ with an empty or undefined source still counts (TD13: c_p is kept, w̃_p = 0).
// OWNER T-B. G2: 0 resIn = res[h] ro · 1 arena rw · 2 rsVbuf · 3 rsGeo · 4 rsVbufPrev · 5 rsGeoPrev (§4.2). RS_REPLAY = 0.
#include "restir/tshift.wgsl"
#include "restir/tpick.wgsl"
#include "debug/restir-views.wgsl"
#if RS_DUPMAP
#include "restir/m6-types.wgsl"
#endif

@compute @workgroup_size(8, 8, 1)
fn rs_t_classify(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
  if (rs_vbuf(p.px).x == 0xFFFFFFFFu) { ts_clear(p.ai, TS_BG); return; }
  let key = rs_frame_key(p.member, rs_t(), p.localIdx);
  let pk = temporal_pixel(p, key);
  rsdbg_tpick(p.px, pk.sp, pk.tap, pk.valid);
  if (!pk.valid) {
    ts_clear(p.ai, TS_DISOCC | select(TS_NO_HIST, 0u, rs_tf(TF_HIST_VALID)));
    rs_count(RSC_T_DISOCC, 1u);
#if RS_BOOST_GATE
    // perf2 WP-5: open the boost gate (a load first: one atomic per pixel only until the word is set)
    if (atomicLoad(&rsArena.hdr[RS_HDR_BOOST_GATE]) == 0u) { atomicOr(&rsArena.hdr[RS_HDR_BOOST_GATE], 1u); }
#endif
    return;
  }
  let qP = pk.ai;
  let cPrev = rp_c(resin_plane(qP, RP_SEED));
#if RS_DUPMAP
  // M6 MD10 (BIASED, interactive only): c_Cap = cCap − (cCap − 1)·D^α, D = duplicates of q′ in the 17×17 window of the
  // previous frame's final reservoirs / 288, α = 0.1 (math.md#dupmap)
  let dupD = f32(rsArena.words[rs_dup_base() + qP]) / DUP_DENOM;
  let cCapD = rsParams.cCap - (rsParams.cCap - 1.0) * pow(min(dupD, 1.0), 0.1);
  let cP = min(max(cCapD, 1.0), cPrev);
#else
  let cP = min(rsParams.cCap, cPrev);
#endif
  var flags = TS_QVALID | select(0u, TS_PICK_RING, pk.tap != 0u);
#if RS_TSTATE_SOA
  // perf2 WP-6 (RS_TSTATE_SOA): every tState word written once (the words T1 knows now here, the forward-shift words
  // and the flags after the shift); the final words equal the #else text's
#if RS_DUAL_MV
  let dual = pk.tap >= 10u;
  if (dual) { flags |= TS_DUAL_PICK; }
  ts_storef(p.ai, TSW_CP, select(cP, min(cP, DMV_C_CAP), dual));
#else
  ts_storef(p.ai, TSW_CP, cP);
#endif
  ts_store(p.ai, TSW_QPRIME, qP);
  ts_storef(p.ai, TSW_CPREV, cPrev);
  ts_clear_inv(p.ai);
  rs_count(RSC_T_QVALID, 1u);
  let src = tsrc_load(qP, 0u, SFX_FWD, RS_FS_CUR);
  ts_store(p.ai, TSW_XPENTRY, src.xpEntry);
  var fF = vec3f(0.0);
  var fJ = JW_FAILED;
  var fCode = rs_slot_code(SC_EMPTY_SRC, RCT_NONE, 0u, 0.0);
  var fJP = 0.0;
  if (src.base.empty) {
  } else if (src.undefinedLight) {
    fCode = rs_slot_code(SC_O0_LIGHT, RCT_NONE, 0u, 0.0);
    fJP = src.jp;
    tcount_fwd(fCode);
    if (tsfx_light_refused(SFX_FWD, qP)) { rs_count(RSC_T_LIGHT_CLASS, 1u); }
  } else if (res_needs_replay(src.base.flags)) {
    fJ = JW_PENDING;
    fCode = rs_slot_code(SC_PENDING, RCT_NONE, 0u, 0.0);
    queue_append(RS_Q_FWD, queue_item_word(p.ai, 0u));
    flags |= TS_FWD_QUEUED;
    rs_count(RSC_T_FWD_QUEUED, 1u);
  } else {
    let o = temporal_shift(src, tdst_cur(p));
    // = ts_store_fwd
    let ok = rs_slot_code_sc(o.code) == SC_OK;
    fF = select(vec3f(0.0), o.F, ok);
    fJ = select(JW_FAILED, bitcast<u32>(o.J), ok);
    fCode = o.code;
    fJP = o.jP;
    tcount_fwd(o.code);
    flags |= TS_FWD_DONE;
  }
  ts_storef(p.ai, TSW_FWDF, fF.x);
  ts_storef(p.ai, TSW_FWDF + 1u, fF.y);
  ts_storef(p.ai, TSW_FWDF + 2u, fF.z);
  ts_store(p.ai, TSW_FWDJ, fJ);
  ts_store(p.ai, TSW_FWDCODE, fCode);
  ts_storef(p.ai, TSW_JP, fJP);
  ts_store(p.ai, TSW_FLAGS, flags);
#else
#if RS_DUAL_MV
  // MD11 amendment DMV-1: a dual-MV q′ carries c_p = min(DMV_C_CAP, c_prev) (its history belongs to another surface
  // point; with c_p up to cCap its importance ratio p̂_q·J/p̂_q′ was amplified up to cCap-fold, the ix-d f40 tail)
  let dual = pk.tap >= 10u;
  if (dual) { flags |= TS_DUAL_PICK; }
  ts_clear(p.ai, flags);
  ts_store(p.ai, TSW_QPRIME, qP);
  ts_storef(p.ai, TSW_CP, select(cP, min(cP, DMV_C_CAP), dual));
#else
  ts_clear(p.ai, flags);
  ts_store(p.ai, TSW_QPRIME, qP);
  ts_storef(p.ai, TSW_CP, cP);
#endif
  ts_storef(p.ai, TSW_CPREV, cPrev);
  rs_count(RSC_T_QVALID, 1u);
  let src = tsrc_load(qP, 0u, SFX_FWD, RS_FS_CUR);
  ts_store(p.ai, TSW_XPENTRY, src.xpEntry);
  if (src.base.empty) {
    ts_store(p.ai, TSW_FWDCODE, rs_slot_code(SC_EMPTY_SRC, RCT_NONE, 0u, 0.0));
  } else if (src.undefinedLight) {
    ts_store(p.ai, TSW_FWDCODE, rs_slot_code(SC_O0_LIGHT, RCT_NONE, 0u, 0.0));
    ts_storef(p.ai, TSW_JP, src.jp);
    tcount_fwd(rs_slot_code(SC_O0_LIGHT, RCT_NONE, 0u, 0.0));
    if (tsfx_light_refused(SFX_FWD, qP)) { rs_count(RSC_T_LIGHT_CLASS, 1u); }
  } else if (res_needs_replay(src.base.flags)) {
    ts_store(p.ai, TSW_FWDJ, JW_PENDING);
    ts_store(p.ai, TSW_FWDCODE, rs_slot_code(SC_PENDING, RCT_NONE, 0u, 0.0));
    queue_append(RS_Q_FWD, queue_item_word(p.ai, 0u));
    flags |= TS_FWD_QUEUED;
    rs_count(RSC_T_FWD_QUEUED, 1u);
  } else {
    let o = temporal_shift(src, tdst_cur(p));
    ts_store_fwd(p.ai, o);
    tcount_fwd(o.code);
    flags |= TS_FWD_DONE;
  }
  ts_store(p.ai, TSW_FLAGS, flags);
#endif
}

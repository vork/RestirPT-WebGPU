// T1 rs_t_classify (restir-temporal-api.md §3.6, §4.1; math.md#temporal [M5 addition]): per pixel q over the atlas.
// Clears tState[q] (every word, every pixel incl. background), picks q′ (restir/tpick.wgsl temporal_pixel: G-buffers
// and the pick stream only), reads c_prev of the history record res[h][q′] (c_p = min(cCap, c_prev), TD14) and runs the
// forward temporal shift T(X_p) inline for sources that need no replay (k = 2, or class L with d = 2); replay sources
// are queued on Q_f (q1) for T2. A valid q′ with an empty or undefined source still counts (TD13: c_p is kept, w̃_p = 0).
// OWNER T-B. G2: 0 resIn = res[h] ro · 1 arena rw · 2 rsVbuf · 3 rsGeo · 4 rsVbufPrev · 5 rsGeoPrev (§4.2). RS_REPLAY = 0.
#include "restir/tshift.wgsl"
#include "restir/tpick.wgsl"
#include "debug/restir-views.wgsl"

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
    return;
  }
  let qP = pk.ai;
  let cPrev = rp_c(resin_plane(qP, RP_SEED));
  let cP = min(rsParams.cCap, cPrev);
  var flags = TS_QVALID | select(0u, TS_PICK_RING, pk.tap != 0u);
  ts_clear(p.ai, flags);
  ts_store(p.ai, TSW_QPRIME, qP);
  ts_storef(p.ai, TSW_CP, cP);
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
}

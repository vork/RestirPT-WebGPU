// T4 rs_t_inverse (restir-temporal-api.md §3.6, §4.1; math.md#temporal): 64/wg, 2D indirect over Q_i (q2): π_p of the
// record y at res[w][q] (X_c; Y_p in recompute / robust mode) = lum F_{t−1}(T⁻¹ y)·|∂T⁻¹/∂y| through the inverse
// temporal shift into (q′, t−1) under E_{t−1}: the previous jittered V-buffer hit and thr at q′, the previous camera,
// the previous light / env state (fs = PREV; entry translated cur → prev and J_P⁻¹ by the inverse refresh record).
// Robust mode: the translated-back endpoint entry of Y_p must equal X_p's (xpEntry). res[w] is bound rw as resOut and
// never written here. OWNER T-B.
// G2: 0 resOut = res[w] rw · 1 arena rw · 2 rsVbuf · 3 rsGeo · 4 rsVbufPrev · 5 rsGeoPrev (§4.2). RS_REPLAY = 1.
#include "restir/tshift.wgsl"
#include "debug/restir-views.wgsl"

@compute @workgroup_size(64)
fn rs_t_inverse(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
  let i = queue_item_chunk(RS_Q_INV, wid, nwg, lid, rsDispatch.treeBase, rsDispatch.treeCount);
  if (i == 0xFFFFFFFFu) { return; }
  let q = queue_item_ai(arena_word(arena_item_word(queue_item_base(RS_Q_INV) + i)));
  let qP = ts_load(q, TSW_QPRIME);
  var flags = ts_load(q, TSW_FLAGS) | TS_INV_DONE;
  let src = tsrc_load(q, 1u, SFX_INV, RS_FS_PREV);
  if ((flags & TS_ROBUST) != 0u && src.xpEntry != RC_NONE) {
    // tsrc_load put the translated-back (prev-numbered) entry into end.x
    let back = src.base.end.x & RC_ENTRY_MASK;
    if (src.undefinedLight || back != ts_load(q, TSW_XPENTRY)) { flags |= TS_ROBUST_IDMIS; }
  }
  if (rs_tf(TF_REFRESH) && (sfx_load(SFX_INV, q).status & SXS_E2) != 0u) {
    flags |= TS_E2_ZERO;
    rs_count(RSC_T_E2_ZEROED, 1u);
  }
  if (src.undefinedLight && tsfx_light_refused(SFX_INV, q)) { rs_count(RSC_T_LIGHT_CLASS, 1u); }
  let o = temporal_shift(src, tdst_prev(qP));
  ts_store_inv(q, o);
  if (rs_slot_code_sc(o.code) == SC_OK) { rs_count(RSC_T_INV_OK, 1u); }
  ts_store(q, TSW_FLAGS, flags);
}

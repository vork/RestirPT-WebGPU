// T4 rs_t_inverse (restir-temporal-api.md §3.6, §4.1): 64/wg, 2D indirect over Q_i: π_p of the record at res[w][q]
// (X_c, or Y_p in recompute / robust mode) through T⁻¹ into (q′, t−1) under the previous frame's state.
// OWNER T-B (P0 stub body by T-A). res[w] is bound rw as resOut and never written here.
// G2: 0 resOut = res[w] rw · 1 arena rw · 2 rsVbuf · 3 rsGeo · 4 rsVbufPrev · 5 rsGeoPrev (§4.2). RS_REPLAY = 1.
#include "restir/tshift.wgsl"
#include "debug/restir-views.wgsl"

@compute @workgroup_size(64)
fn rs_t_inverse(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
  let i = queue_item_chunk(RS_Q_INV, wid, nwg, lid, rsDispatch.treeBase, rsDispatch.treeCount);
  if (i == 0xFFFFFFFFu) { return; }
  let q = queue_item_ai(arena_word(arena_item_word(queue_item_base(RS_Q_INV) + i)));
  let qP = ts_load(q, TSW_QPRIME);
  let o = temporal_shift(tsrc_load(q, 1u, SFX_INV, RS_FS_PREV), tdst_prev(qP));
  ts_store(q, TSW_INVJ, JW_FAILED);
  ts_store(q, TSW_INVCODE, o.code);
  ts_storef(q, TSW_PIRECOMP, 0.0);
}

// T2 rs_t_forward (restir-temporal-api.md §3.6, §4.1): 64/wg, 2D indirect over Q_f (item chunks as Changelog C7): the
// forward temporal shift of queued (replay) sources. OWNER T-B (P0 stub body by T-A).
// G2: as rs_t_classify. RS_REPLAY = 1.
#include "restir/tshift.wgsl"
#include "debug/restir-views.wgsl"

@compute @workgroup_size(64)
fn rs_t_forward(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
  let i = queue_item_chunk(RS_Q_FWD, wid, nwg, lid, rsDispatch.treeBase, rsDispatch.treeCount);
  if (i == 0xFFFFFFFFu) { return; }
  let q = queue_item_ai(arena_word(arena_item_word(queue_item_base(RS_Q_FWD) + i)));
  let qP = ts_load(q, TSW_QPRIME);
  let px = vec2u(q % rsParams.atlasSize.x, q / rsParams.atlasSize.x);
  let o = temporal_shift(tsrc_load(qP, 0u, SFX_FWD, RS_FS_CUR), tdst_cur(rs_pix(px)));
  ts_store(q, TSW_FWDJ, JW_FAILED);
  ts_store(q, TSW_FWDCODE, o.code);
}

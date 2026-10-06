// Suffix refresh passes (restir-temporal-api.md §3.5, §4.1, TD9). OWNER T-C.
//   rs_refresh_fwd  8×8 per pixel over the atlas (TF_REFRESH ∧ TF_HIST_VALID): record res[h][ai] (fsFrom = PREV) under
//                   the current frame → sfxOut.fwd
//   rs_refresh_inv  64/wg, 2D indirect over Q_i (item chunk RsDispatch.treeBase/treeCount): record res[w][q] (fsFrom =
//                   CUR) under the previous frame → sfxOut.inv
// Both return without writing on frames without TF_REFRESH (the loaders read sfxOut only on refresh frames), so an
// unconditional encode is harmless. G2: 0 resIn ro · 1 arena rw · 2 rsVbuf · 3 rsGeo (restir-temporal-api.md §4.2).
#include "restir/refresh.wgsl"
#include "debug/restir-views.wgsl"

@compute @workgroup_size(8, 8, 1)
fn rs_refresh_fwd(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid || !rs_tf(TF_REFRESH) || !rs_tf(TF_HIST_VALID)) { return; }
  let r = refresh_record(p.ai, RS_FS_PREV, RS_FS_CUR);
  sfx_store(SFX_FWD, p.ai, r);
  rsdbg_refresh(p.ai, SFX_FWD, r);
}

@compute @workgroup_size(64)
fn rs_refresh_inv(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
  if (!rs_tf(TF_REFRESH)) { return; }
  let i = queue_item_chunk(RS_Q_INV, wid, nwg, lid, rsDispatch.treeBase, rsDispatch.treeCount);
  if (i == 0xFFFFFFFFu) { return; }
  let q = queue_item_ai(arena_word(arena_item_word(queue_item_base(RS_Q_INV) + i)));
  let r = refresh_record(q, RS_FS_CUR, RS_FS_PREV);
  sfx_store(SFX_INV, q, r);
  rsdbg_refresh(q, SFX_INV, r);
}

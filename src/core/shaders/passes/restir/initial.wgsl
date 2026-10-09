// rs_initial (restir-api.md §4.1 step 2, §3.5; math.md#path-tree): the path tree(s) of each atlas pixel streamed into
// resOut[ai] with streaming RIS. Tree chunks: RsDispatch.treeBase/treeCount, RSD_FIRST_CHUNK initialises the record,
// RSD_FINAL_CHUNK finalises W; Σw persists in the reservoir between chunks (D10). Background pixels (vbuf miss) get
// the background record on the first chunk.
// G2: 0 resOut rw · 1 shiftArena rw (counters) · 2 rsVbuf · 3 rsGeo [· 4 candDump rw, RS_DUMP_CANDIDATES].
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"
#include "restir/queue.wgsl"
#include "path/pathtree.wgsl"
#include "debug/restir-views.wgsl"

@compute @workgroup_size(8, 8, 1)
fn rs_initial(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
  let key = rs_frame_key(p.member, rs_t(), p.localIdx);
  let first = (rsDispatch.flags & RSD_FIRST_CHUNK) != 0u;
  let lastChunk = (rsDispatch.flags & RSD_FINAL_CHUNK) != 0u;
  if (rs_vbuf(p.px).x == 0xFFFFFFFFu) {
    if (first) { res_write_empty(p.ai, key, true); }
    if (lastChunk) { rsdbg_reservoir(p.px, p.ai, 1u); }
    return;
  }
  pathtree_run(p, key, rsDispatch.treeBase, rsDispatch.treeCount, first, lastChunk);
}

#if RS_RIS_PREPASS && RS_RIS_NEE
// perf2 WP-2d (RS_RIS_PREPASS; interactive, trees = 1, no candidate dump / test text): the RIS-NEE selection at x₁ of
// tree treeBase as its own lean pass before rs_initial (schedule rs_primary → rs_light_tiles → rs_ris_nee → rs_initial).
// x₁, V and the MatEval are rs_initial's own (vertex_from_ids, D3 direction, material_eval); the loop always runs (never
// gated on this pass's MatEval flags: rs_initial decides with its own). Record → resOut[ai] plane RP_DIAG
// (ris-nee.wgsl ris_prepass_record); background pixels write nothing (rs_initial gives them the background record).
@compute @workgroup_size(8, 8, 1)
fn rs_ris_nee(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
  let vb = rs_vbuf(p.px);
  if (vb.x == 0xFFFFFFFFu) { return; }
  let key = rs_frame_key(p.member, rs_t(), p.localIdx);
  let s = rsDispatch.treeBase;
  let camPos = rs_cam_pos();
  let cur = vertex_from_ids(vb.x, bitcast<f32>(vb.y), bitcast<f32>(vb.z), camPos);
  let V = normalize(camPos - cur.pos);
  let m = material_eval(cur, V);
  resout_set(p.ai, RP_DIAG, ris_prepass_record(ris_nee_select(p, key, rs_tree_seed(key, s), s, cur, vb.x, m, V)));
}
#endif

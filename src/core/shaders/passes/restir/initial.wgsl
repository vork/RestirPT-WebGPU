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

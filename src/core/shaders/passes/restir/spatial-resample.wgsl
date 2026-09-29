// rs_spatial_resample (restir-api.md §3.8, §4.1; math.md#paired-mis; PLAN §2 rule 8, §3 step 5): thread per atlas
// pixel; paired defensive-pairwise MIS over the canonical sample and the accepted slots, streaming RIS, write-back
// into resOut (ping-pong: reads resIn only), final round (RSD_FINAL_ROUND): RGB shade into rsShade and the
// DBG_TAP_SPATIAL reservoir views (rsdbg_reservoir). The body is
// restir/mis.wgsl spatial_resample; counters (RSC_PENDING_LEFT, RSC_SLOT_MISMATCH, RSC_W_NONFINITE,
// RSC_SELECTED_SHIFTED, RSC_EMPTY_CANON) are aggregated per workgroup (Changelog C1: the arena is bound read_write).
// G2: 0 resIn ro · 1 resOut rw · 2 shiftArena rw · 3 pairTex · 4 rsShade (st w) · 5 rsVbuf.
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"
#include "restir/queue.wgsl"
#include "restir/mis.wgsl"

@compute @workgroup_size(8, 8, 1)
fn rs_spatial_resample(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  let finalRound = (rsDispatch.flags & RSD_FINAL_ROUND) != 0u;
  if (p.valid) {
    spatial_resample(p, rsDispatch.round, finalRound);
    if (finalRound) { rsdbg_reservoir(p.px, p.ai, DBG_TAP_SPATIAL); }
  }
  workgroupBarrier();
  mis_flush_counters(li);
}

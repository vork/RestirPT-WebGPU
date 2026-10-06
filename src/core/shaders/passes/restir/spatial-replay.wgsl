// rs_spatial_replay (restir-api.md §3.9, §2.7; PLAN §1.8 "prefix replay is its own compacted indirect pass, with the
// reconnection fused in"; gap-webgpu-pass-cost): one thread per queue-0 item (ai << 3 | s), 2D indirect dispatch of
// rs_args, @workgroup_size(64); optionally one chunk [treeBase, treeBase + treeCount) of the queue per dispatch
// (Changelog C7, submit budget). Recomputes the partner of (ai, s) and runs shift_hybrid with replay compiled in
// (RS_REPLAY = 1): source = the reservoir of ai, destination = its partner's domain; writes slot (ai, s).
// Counters: SC histogram and RSC_SHIFT_NONFINITE, aggregated per workgroup.
// G2: 0 resIn ro · 1 shiftArena rw · 2 rsVbuf · 3 rsGeo · 4 pairTex.
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"
#include "restir/queue.wgsl"
#include "restir/pairing.wgsl"
#include "restir/shift.wgsl"
#include "debug/restir-views.wgsl"

var<workgroup> srHist: array<atomic<u32>, 16>;

fn sr_item(i: u32) {
  let w = arena_word(arena_item_word(i));
  let ai = queue_item_ai(w);
  let s = queue_item_slot(w);
  let p = rs_pix(vec2u(ai % rsParams.atlasSize.x, ai / rsParams.atlasSize.x));
  let pr = pair_partner(p.local, p.member, rs_t(), rsDispatch.round, s);
  var o: ShiftOut;
  if (p.valid && pr.valid) {
#if RS_PLANT_T2
    rsT2SrcAi = ai;                                      // U8-4 plant: the source record of this shift
#endif
    o = shift_hybrid(shift_src_load(ai), shift_dst_load(pair_atlas_px(p, pr.partner)));
  } else {
    o.code = rs_slot_code(SC_NOT_ACCEPTED, RCT_NONE, 0u, 0.0);
  }
  let sc = arena_slot_store_shift(ai, s, o.FJ, o.J, o.code);
  atomicAdd(&srHist[sc & 15u], 1u);
  rsdbg_slot(p.px, s, o.code, o.J, true);
}

@compute @workgroup_size(64)
fn rs_spatial_replay(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) li: u32) {
  let i = queue_item_chunk(0u, wid, nwg, li, rsDispatch.treeBase, rsDispatch.treeCount);
  if (i != 0xFFFFFFFFu) { sr_item(i); }
  workgroupBarrier();
  if (li < 16u) {
    let n = atomicLoad(&srHist[li]);
    rs_count(RSC_CODE_BASE + li, n);
    if (li == SC_NONFINITE) { rs_count(RSC_SHIFT_NONFINITE, n); }
  }
}

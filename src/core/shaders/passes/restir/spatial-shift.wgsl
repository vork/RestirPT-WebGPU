// rs_spatial_shift (restir-api.md §3.9, §2.6; math.md#paired-mis "Shift pass (S1)"; PLAN §3 step 5): thread per atlas
// pixel p; for every slot s whose J word is PENDING and whose source (p's own reservoir) does NOT need replay
// (res_needs_replay false: k = 2, so y_{k−1} = y₁), shift p's sample into the partner's domain with
// shift_hybrid (RS_REPLAY = 0) and write slot (p, s). Slots of replay sources were handled by rs_spatial_replay
// before this pass; a still-PENDING one is counted as RSC_PENDING_LEFT by the resample. Also reports the accepted
// slots (rsdbg_accept). Counters: SC histogram and RSC_SHIFT_NONFINITE, aggregated per workgroup.
// G2: 0 resIn ro · 1 shiftArena rw · 2 rsVbuf · 3 rsGeo · 4 pairTex.
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"
#include "restir/queue.wgsl"
#include "restir/pairing.wgsl"
#include "restir/shift.wgsl"
#include "debug/restir-views.wgsl"

var<workgroup> ssHist: array<atomic<u32>, 16>;

#if RS_DENSE_SLOTS || RS_BOOST_GATE || RS_PAIR_TABLE
// perf2 WP-5 variants (the #else branch is the unchanged validation text).
fn ss_partner(local: vec2u, member: u32, t: u32, r: u32, s: u32) -> PairResult {
#if RS_PAIR_TABLE
  return pair_partner_wg(local, member, t, r, s);
#else
  return pair_partner(local, member, t, r, s);
#endif
}

fn ss_flush(li: u32) {
  if (li < 16u) {
    let n = atomicLoad(&ssHist[li]);
    rs_count(RSC_CODE_BASE + li, n);
    if (li == SC_NONFINITE) { rs_count(RSC_SHIFT_NONFINITE, n); }
  }
}

#if RS_DENSE_SLOTS
// RS_DENSE_SLOTS: one thread per q3 item (ai << 3 | s) of rs_pair_accept, 2D indirect dispatch of rs_args (q3),
// @workgroup_size(64), optionally one chunk [treeBase, treeBase + treeCount) per dispatch. Same shift, store, counters
// and probe records as the per-pixel loop. The replay predicate stays as a safety net: a replay source is left PENDING
// (counted by the resample as RSC_PENDING_LEFT), exactly as the per-pixel pass skipped it. rsdbg_accept (a no-op since
// Changelog D1: view 439 is written from the J words) is not called.
fn sd_item(i: u32) {
  let w = arena_word(arena_dense_item_word(i));
  let ai = queue_item_ai(w);
  let s = queue_item_slot(w);
  if (res_needs_replay(resin_plane(ai, RP_SEED).z)) { return; }
  let p = rs_pix(vec2u(ai % rsParams.atlasSize.x, ai / rsParams.atlasSize.x));
  let pr = ss_partner(p.local, p.member, rs_t(), rsDispatch.round, s);
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
  atomicAdd(&ssHist[sc & 15u], 1u);
  rsdbg_slot(p.px, s, o.code, o.J, false);
}

@compute @workgroup_size(64)
fn rs_spatial_shift(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) li: u32) {
#if RS_PAIR_TABLE
  pair_xf_prepare(li, rsDispatch.round);
#endif
  let i = queue_item_chunk(RS_Q_DENSE, wid, nwg, li, rsDispatch.treeBase, rsDispatch.treeCount);
  if (i != 0xFFFFFFFFu) { sd_item(i); }
  workgroupBarrier();
  ss_flush(li);
}
#else
fn ss_pixel(p: RsPix) {
  let t = rs_t();
  let r = rsDispatch.round;
  let flags = resin_plane(p.ai, RP_SEED).z;
  let own = !res_needs_replay(flags);
#if RS_BOOST_GATE
  let NS = pair_ns_eff();                                // gated boost slots are NOT_ACCEPTED (pair_accept wrote them)
#else
  let NS = rsParams.numSlots;
#endif
  for (var s = 0u; s < RS_MAX_SLOTS; s++) {
    if (s >= NS) { break; }
    let jw = arena_slot_jword(p.ai, s);
    if (jw != JW_PENDING || !own) { continue; }
    let pr = ss_partner(p.local, p.member, t, r, s);
    var o: ShiftOut;
    if (pr.valid) {
#if RS_PLANT_T2
      rsT2SrcAi = p.ai;                                  // U8-4 plant: the source record of this shift
#endif
      o = shift_hybrid(shift_src_load(p.ai), shift_dst_load(pair_atlas_px(p, pr.partner)));
    } else {
      o.code = rs_slot_code(SC_NOT_ACCEPTED, RCT_NONE, 0u, 0.0);
    }
    let sc = arena_slot_store_shift(p.ai, s, o.FJ, o.J, o.code);
    atomicAdd(&ssHist[sc & 15u], 1u);
    rsdbg_slot(p.px, s, o.code, o.J, false);
  }
}

@compute @workgroup_size(8, 8, 1)
fn rs_spatial_shift(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
#if RS_PAIR_TABLE
  pair_xf_prepare(li, rsDispatch.round);
#endif
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (p.valid) { ss_pixel(p); }
  workgroupBarrier();
  ss_flush(li);
}
#endif
#else
fn ss_pixel(p: RsPix) {
  let t = rs_t();
  let r = rsDispatch.round;
  let flags = resin_plane(p.ai, RP_SEED).z;
  var mask = 0u;
  let own = !res_needs_replay(flags);
  for (var s = 0u; s < rsParams.numSlots; s++) {
    let jw = arena_slot_jword(p.ai, s);
    if (jw_accepted(jw)) { mask |= 1u << s; }
    if (jw != JW_PENDING || !own) { continue; }
    let pr = pair_partner(p.local, p.member, t, r, s);
    // An accepted slot always has a valid partner (pair_accept wrote it); guard anyway.
    var o: ShiftOut;
    if (pr.valid) {
#if RS_PLANT_T2
      rsT2SrcAi = p.ai;                                  // U8-4 plant: the source record of this shift
#endif
      o = shift_hybrid(shift_src_load(p.ai), shift_dst_load(pair_atlas_px(p, pr.partner)));
    } else {
      o.code = rs_slot_code(SC_NOT_ACCEPTED, RCT_NONE, 0u, 0.0);
    }
    let sc = arena_slot_store_shift(p.ai, s, o.FJ, o.J, o.code);
    atomicAdd(&ssHist[sc & 15u], 1u);
    rsdbg_slot(p.px, s, o.code, o.J, false);
  }
  rsdbg_accept(p.px, mask);
}

@compute @workgroup_size(8, 8, 1)
fn rs_spatial_shift(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (p.valid) { ss_pixel(p); }
  workgroupBarrier();
  if (li < 16u) {
    let n = atomicLoad(&ssHist[li]);
    rs_count(RSC_CODE_BASE + li, n);
    if (li == SC_NONFINITE) { rs_count(RSC_SHIFT_NONFINITE, n); }
  }
}
#endif

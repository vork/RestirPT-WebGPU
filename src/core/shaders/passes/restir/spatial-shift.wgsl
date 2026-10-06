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

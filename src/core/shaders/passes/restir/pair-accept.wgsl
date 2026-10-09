// rs_pair_accept (restir-api.md §3.9, §2.6, D8; math.md#paired-mis "Acceptance"; PLAN §2 rule 8, §3 step 5): the
// acceptance of every pair of every slot of this round, evaluated ONCE per pair by the thread of the smaller atlas index
// from the G-buffer only (A never depends on reservoir contents), writing the J words and codes of BOTH slots:
//   no partner / not reciprocal → own slot NOT_ACCEPTED;  A0 false → both NOT_ACCEPTED;
//   A0 true → per side: empty source → FAILED (SC_EMPTY_SRC); source needs replay → PENDING + queue item (ai << 3 | s);
//   else PENDING (handled by rs_spatial_shift).
// Every slot is written exactly once per round (the map is an involution). One thread per atlas pixel, background
// pixels included (A0 rejects them). Counters are aggregated per workgroup (RSC_ACCEPTED, RSC_QUEUED, SC histogram).
// M5 boost (OWNER T-D; restir-temporal-api.md TD21, §3.7): slots s ≥ numSlots − boostSlots accept a pair iff
// A0 ∧ (dis(p) ∨ dis(q)) (restir/pairing.wgsl pair_boost_accept), evaluated by the same min thread; without the boost
// (boostSlots = 0, always with temporal off) the loop is the M4 one.
// G2: 0 resIn ro · 1 shiftArena rw · 2 rsVbuf · 3 rsGeo · 4 pairTex.
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"
#include "restir/queue.wgsl"
#include "restir/pairing.wgsl"

/// Same predicate as shift.wgsl res_needs_replay (non-empty ∧ (k > 2 ∨ k = ∅)); shift.wgsl cannot be included here
/// (no scene group). A divergence leaves PENDING slots (RSC_PENDING_LEFT) and is covered by restir-spatial's
/// replay-predicate test.
fn pa_needs_replay(flags: u32) -> bool {
  let k = rf_k(flags);
  return !res_empty(flags) && (k > 2u || k == 0u);
}

// workgroup counter aggregation: 0 accepted, 1 queued, 2 SC_NOT_ACCEPTED, 3 SC_EMPTY_SRC, 4 slot mismatch
var<workgroup> paCnt: array<atomic<u32>, 5>;

/// Accepted slot (ai, s) of a source with reservoir flags `flags`.
fn pa_accept_slot(ai: u32, s: u32, flags: u32) {
  if (res_empty(flags)) {
    arena_slot_write(ai, s, vec3f(0.0), JW_FAILED, rs_slot_code(SC_EMPTY_SRC, RCT_NONE, 0u, 0.0));
    atomicAdd(&paCnt[3], 1u);
    return;
  }
  arena_slot_write(ai, s, vec3f(0.0), JW_PENDING, rs_slot_code(SC_PENDING, RCT_NONE, 0u, 0.0));
  if (pa_needs_replay(flags)) {
    queue_append(0u, queue_item_word(ai, s));
    atomicAdd(&paCnt[1], 1u);
  }
}

fn pa_not_accepted(ai: u32, s: u32) {
  arena_slot_write(ai, s, vec3f(0.0), JW_NOT_ACCEPTED, rs_slot_code(SC_NOT_ACCEPTED, RCT_NONE, 0u, 0.0));
  atomicAdd(&paCnt[2], 1u);
}

#if RS_DENSE_SLOTS || RS_BOOST_GATE || RS_PAIR_TABLE
// perf2 WP-5 variants (the #else branch is the unchanged validation text).
/// Partner of slot s (RS_PAIR_TABLE: through the workgroup transform table).
fn pa_partner(local: vec2u, member: u32, t: u32, r: u32, s: u32) -> PairResult {
#if RS_PAIR_TABLE
  return pair_partner_wg(local, member, t, r, s);
#else
  return pair_partner(local, member, t, r, s);
#endif
}

/// NS_eff (RS_BOOST_GATE: the gated boost slots are NOT_ACCEPTED without partner / A0 work; each side writes its own).
fn pa_ns_eff() -> u32 {
#if RS_BOOST_GATE
  return pair_ns_eff();
#else
  return rsParams.numSlots;
#endif
}

fn pa_flush(li: u32) {
  if (li == 0u) {
    rs_count(RSC_ACCEPTED, atomicLoad(&paCnt[0]));
    rs_count(RSC_QUEUED, atomicLoad(&paCnt[1]));
    rs_count(RSC_CODE_BASE + SC_NOT_ACCEPTED, atomicLoad(&paCnt[2]));
    rs_count(RSC_CODE_BASE + SC_EMPTY_SRC, atomicLoad(&paCnt[3]));
    rs_count(RSC_SLOT_MISMATCH, atomicLoad(&paCnt[4]));
  }
}

#if RS_DENSE_SLOTS
// RS_DENSE_SLOTS: every thread writes only its OWN slots. A0 is evaluated by both threads of a pair with the operands
// in canonical order (smaller atlas index first), so both sides agree bit for bit; the counters are per side
// (RSC_ACCEPTED +1 per accepted side = +2 per pair, SC_NOT_ACCEPTED +1 per slot, as before). Non-replay PENDING slots
// become q3 items, appended workgroup-aggregated (one global atomic per workgroup; a pixel's items stay adjacent);
// replay items go to q0 exactly as before (RSC_QUEUED counts them only).
var<workgroup> paDenseN: atomic<u32>;
var<workgroup> paDenseBase: u32;

fn pa_pixel_dense(p: RsPix, items: ptr<function, array<u32, RS_MAX_SLOTS>>) -> u32 {
  let t = rs_t();
  let r = rsDispatch.round;
  let vbP = rs_vbuf(p.px);
  let geoP = rs_geo(p.px);
  let flags = resin_plane(p.ai, RP_SEED).z;
  let firstBoost = pair_first_boost_slot();
  let nsEff = pa_ns_eff();
  var n = 0u;
  for (var s = 0u; s < RS_MAX_SLOTS; s++) {
    if (s >= rsParams.numSlots) { break; }
    if (s >= nsEff) { pa_not_accepted(p.ai, s); continue; }
    let pr = pa_partner(p.local, p.member, t, r, s);
    if (!pr.valid) { pa_not_accepted(p.ai, s); continue; }
    // Reciprocity (true by construction for an involution map; checked so that every slot is still written once).
    let back = pa_partner(pr.partner, p.member, t, r, s);
    if (!back.valid || back.partner.x != p.local.x || back.partner.y != p.local.y) {
      pa_not_accepted(p.ai, s);
      atomicAdd(&paCnt[4], 1u);
      continue;
    }
    let qpx = pair_atlas_px(p, pr.partner);
    let qai = pair_atlas_index(qpx);
    let vbQ = rs_vbuf(qpx);
    let geoQ = rs_geo(qpx);
    let pMin = p.ai < qai;                        // canonical order: G[min], G[max]
    var a = pair_A0(select(vbQ, vbP, pMin), select(geoQ, geoP, pMin), select(vbP, vbQ, pMin), select(geoP, geoQ, pMin));
    if (s >= firstBoost) { a = pair_boost_accept(a, pair_disoccluded(select(qai, p.ai, pMin)), pair_disoccluded(select(p.ai, qai, pMin))); }
    if (!a) { pa_not_accepted(p.ai, s); continue; }
    atomicAdd(&paCnt[0], 1u);
    if (res_empty(flags)) {
      arena_slot_write(p.ai, s, vec3f(0.0), JW_FAILED, rs_slot_code(SC_EMPTY_SRC, RCT_NONE, 0u, 0.0));
      atomicAdd(&paCnt[3], 1u);
      continue;
    }
    arena_slot_write(p.ai, s, vec3f(0.0), JW_PENDING, rs_slot_code(SC_PENDING, RCT_NONE, 0u, 0.0));
    if (pa_needs_replay(flags)) {
      queue_append(0u, queue_item_word(p.ai, s));
      atomicAdd(&paCnt[1], 1u);
    } else {
      (*items)[n] = queue_item_word(p.ai, s);
      n++;
    }
  }
  return n;
}

@compute @workgroup_size(8, 8, 1)
fn rs_pair_accept(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
#if RS_PAIR_TABLE
  pair_xf_prepare(li, rsDispatch.round);
#endif
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  var items: array<u32, RS_MAX_SLOTS>;
  var n = 0u;
  if (p.valid) { n = pa_pixel_dense(p, &items); }
  let off = atomicAdd(&paDenseN, n);
  workgroupBarrier();
  if (li == 0u) {
    let total = atomicLoad(&paDenseN);
    var base = 0u;
    if (total != 0u) { base = atomicAdd(&rsArena.hdr[4u * RS_Q_DENSE], total); }
    paDenseBase = base;
  }
  workgroupBarrier();
  let cap = queue_capacity();
  let b = paDenseBase + off;
  for (var j = 0u; j < RS_MAX_SLOTS; j++) {
    if (j >= n) { break; }
    if (b + j >= cap) { atomicStore(&rsArena.hdr[4u * RS_Q_DENSE + 3u], 1u); break; }
    rsArena.words[arena_dense_item_word(b + j)] = items[j];
  }
  pa_flush(li);
}
#else
fn pa_pixel(p: RsPix) {
  let t = rs_t();
  let r = rsDispatch.round;
  let vbP = rs_vbuf(p.px);
  let geoP = rs_geo(p.px);
  let firstBoost = pair_first_boost_slot();
  let nsEff = pa_ns_eff();
  for (var s = 0u; s < RS_MAX_SLOTS; s++) {
    if (s >= rsParams.numSlots) { break; }
    if (s >= nsEff) { pa_not_accepted(p.ai, s); continue; }
    let pr = pa_partner(p.local, p.member, t, r, s);
    if (!pr.valid) { pa_not_accepted(p.ai, s); continue; }
    let back = pa_partner(pr.partner, p.member, t, r, s);
    if (!back.valid || back.partner.x != p.local.x || back.partner.y != p.local.y) {
      pa_not_accepted(p.ai, s);
      atomicAdd(&paCnt[4], 1u);
      continue;
    }
    let qpx = pair_atlas_px(p, pr.partner);
    let qai = pair_atlas_index(qpx);
    if (p.ai > qai) { continue; }                 // the partner's thread owns the pair
    var a = pair_A0(vbP, geoP, rs_vbuf(qpx), rs_geo(qpx));   // canonical order: G[min], G[max]
    if (s >= firstBoost) { a = pair_boost_accept(a, pair_disoccluded(p.ai), pair_disoccluded(qai)); }
    if (!a) {
      pa_not_accepted(p.ai, s);
      pa_not_accepted(qai, s);
      continue;
    }
    atomicAdd(&paCnt[0], 2u);
    pa_accept_slot(p.ai, s, resin_plane(p.ai, RP_SEED).z);
    pa_accept_slot(qai, s, resin_plane(qai, RP_SEED).z);
  }
}

@compute @workgroup_size(8, 8, 1)
fn rs_pair_accept(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
#if RS_PAIR_TABLE
  pair_xf_prepare(li, rsDispatch.round);
#endif
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (p.valid) { pa_pixel(p); }
  workgroupBarrier();
  pa_flush(li);
}
#endif
#else
fn pa_pixel(p: RsPix) {
  let t = rs_t();
  let r = rsDispatch.round;
  let vbP = rs_vbuf(p.px);
  let geoP = rs_geo(p.px);
  let firstBoost = pair_first_boost_slot();
  for (var s = 0u; s < rsParams.numSlots; s++) {
    let pr = pair_partner(p.local, p.member, t, r, s);
    if (!pr.valid) { pa_not_accepted(p.ai, s); continue; }
    // Reciprocity (true by construction for an involution map; checked so that every slot is still written once).
    let back = pair_partner(pr.partner, p.member, t, r, s);
    if (!back.valid || back.partner.x != p.local.x || back.partner.y != p.local.y) {
      pa_not_accepted(p.ai, s);
      atomicAdd(&paCnt[4], 1u);
      continue;
    }
    let qpx = pair_atlas_px(p, pr.partner);
    let qai = pair_atlas_index(qpx);
    if (p.ai > qai) { continue; }                 // the partner's thread owns the pair
    var a = pair_A0(vbP, geoP, rs_vbuf(qpx), rs_geo(qpx));   // canonical order: G[min], G[max]
    if (s >= firstBoost) { a = pair_boost_accept(a, pair_disoccluded(p.ai), pair_disoccluded(qai)); }
    if (!a) {
      pa_not_accepted(p.ai, s);
      pa_not_accepted(qai, s);
      continue;
    }
    atomicAdd(&paCnt[0], 2u);
    pa_accept_slot(p.ai, s, resin_plane(p.ai, RP_SEED).z);
    pa_accept_slot(qai, s, resin_plane(qai, RP_SEED).z);
  }
}

@compute @workgroup_size(8, 8, 1)
fn rs_pair_accept(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (p.valid) { pa_pixel(p); }
  workgroupBarrier();
  if (li == 0u) {
    rs_count(RSC_ACCEPTED, atomicLoad(&paCnt[0]));
    rs_count(RSC_QUEUED, atomicLoad(&paCnt[1]));
    rs_count(RSC_CODE_BASE + SC_NOT_ACCEPTED, atomicLoad(&paCnt[2]));
    rs_count(RSC_CODE_BASE + SC_EMPTY_SRC, atomicLoad(&paCnt[3]));
    rs_count(RSC_SLOT_MISMATCH, atomicLoad(&paCnt[4]));
  }
}
#endif

// ReSTIR debug passes (restir-api.md §2.11, Changelog D1; PLAN §6 "Shifts" row). OWNER WP-D. Dispatched by
// render/restir/debug.ts only while a ReSTIR view or the probe is active; production outputs are never written.
//   rs_debug_fill   AOV := DBG_CODE_NONE before the ReSTIR passes when a code view is active (pixels no hook writes,
//                   e.g. background pixels of the spatial tap, render black instead of code 0).
//   rs_debug_views  after the spatial stage: shift views 420–446 from the shift arena of the LAST executed round
//                   (RsDispatch.round = executed rounds, as rs_finalize, Changelog A6) and the probe's final slot
//                   records (tags 71 outgoing, 72 incoming) plus the anchor vertices of the shifted paths for the
//                   inspector overlay (tag 67, path 16+s: the partner's primary hit y₁ of p→partner; path 24+s: the
//                   probe's y₁, the partner's rc vertex x_k and surface endpoint of partner→p; the rest of each path
//                   is the base path's own vertices, joined in TS, render/restir/debug.ts shiftedPolylines).
// M5 (T-D; restir-temporal-api.md §2.11, Changelog D-1): view 497 `s.boost` (accepted boost slots of the last round,
// bit s − firstBoost) and, at the probe pixel, the temporal anchor ids (tag 79, recorded by rsdbg_temporal in T3, which
// has no scene group) turned into vertices (tag 67): path 23 = the forward shift T(X_p) into this pixel (y₁ of this
// pixel, X_p's x_k and surface endpoint), path 31 = the inverse shift into q′ at t−1 (b = 0: the previous camera, y₁′ of
// q′ from the previous V-buffer, then the source's x_k and surface endpoint). The temporal views 480–496 themselves
// come from the hooks (debug/restir-views.wgsl); rs_debug_fill starts the temporal code views as NONE (480: 0 = bg).
// G2: 0 resSrc ro (the reservoir buffer the last round read) · 1 shiftArena ro · 2 rsVbuf · 3 rsGeo · 4 pairTex ·
// 5 rsVbufPrev (M5; = rsVbuf without temporal). G1 scene (positions of the anchor vertices). G3 debug.
// Storage buffers: records + scene 5 + 2 + debug = 9.
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"
#include "restir/queue.wgsl"
#include "restir/pairing.wgsl"
#include "restir/tframe.wgsl"
#include "scene/scene-data.wgsl"
#include "debug/restir-views.wgsl"

@group(2) @binding(0) var<storage, read> dbgResSrc: array<vec4u>;

fn rsdbg_src_plane(i: u32, p: u32) -> vec4u { return dbgResSrc[i * RS_RES_PLANES + p]; }

/// Position of the surface point (primId, bary u, v) (render frame).
fn rsdbg_prim_pos(prim: u32, u: f32, v: f32) -> vec3f {
  let t = sceneTris[prim];
  return (1.0 - u - v) * scene_vertex_pos(t.x) + u * scene_vertex_pos(t.y) + v * scene_vertex_pos(t.z);
}
fn rsdbg_anchor(px: vec2u, path: u32, b: u32, ids: vec3u) {
  if (ids.x >= RC_TAG_NEE) { return; }             // light / env / none: no surface position
  rsdbg_vertex(px, path, b, rsdbg_prim_pos(ids.x, bitcast<f32>(ids.y), bitcast<f32>(ids.z)), 0xFu);
}

/// = shift.wgsl res_needs_replay (non-empty ∧ (k > 2 ∨ k = ∅)); restated here because shift.wgsl needs the scene group.
fn rsdbg_needs_replay(f: u32) -> bool { return !res_empty(f) && (rf_k(f) > 2u || rf_k(f) == 0u); }

/// Atlas index of the partner of slot s of member-local pixel p in round r (pair_partner of the pairing module).
fn rsdbg_partner_ai(p: RsPix, t: u32, round: u32, s: u32) -> u32 {
  let pr = pair_partner(p.local, p.member, t, round, s);
  if (!pr.valid) { return 0xFFFFFFFFu; }
  let m = p.member - rsParams.memberBase;
  let o = vec2u((m % rsParams.memberCols) * rsParams.memberSize.x, (m / rsParams.memberCols) * rsParams.memberSize.y);
  let q = o + pr.partner;
  return q.y * rsParams.atlasSize.x + q.x;
}

fn rsdbg_is_code_view(mode: u32) -> bool {
  return (mode >= 404u && mode <= 409u) || (mode >= RSV_SHIFT_CODE && mode < RSV_SHIFT_LOGJ)
    || (mode >= RSV_SHIFT_TERM && mode <= RSV_ACCEPT_MASK) || mode == RSV_MIS_K || mode == RSV_MIS_SEL
    || mode == RSV_T_QVALID || (mode >= RSV_T_RF_FWD && mode <= RSV_T_INVCODE) || mode == RSV_T_SEL || mode == RSV_T_LCHG
    || mode == RSV_S_BOOST;
}

@compute @workgroup_size(8, 8, 1)
fn rs_debug_fill(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= dbg.size) || !rsdbg_is_code_view(dbg.mode)) { return; }
  // View 480: background pixels (never written by T3) show code 0 = bg while the temporal stage runs.
  let bgCode = select(DBG_CODE_NONE, 0u, dbg.mode == RSV_T_QVALID && (rsParams.flags & RSF_TEMPORAL) != 0u);
  dbgBuf.aov[gid.y * dbg.size.x + gid.x] = vec4f(rsdbg_bits(bgCode), 0.0, 0.0, 0.0);
}

/// Probe pixel: turn the temporal anchor ids of this frame (tag 79, rsdbg_temporal) into tag-67 vertices of paths 23
/// (forward T(X_p) into this pixel) and 31 (inverse into q′ at t−1); y₁ / y₁′ and the previous camera from the
/// G-buffers and the frame uniforms.
fn rsdbg_temporal_anchors(p: RsPix) {
  let n = min(atomicLoad(&dbgBuf.counters[DBGC_PROBE_COUNT]), PROBE_CAPACITY);
  var header = false;
  var tflags = 0u;
  var qP = 0xFFFFFFFFu;
  for (var i = 0u; i < n; i++) {
    let r = dbgBuf.probe[i];
    if (any(r.pixel != p.px)) { continue; }
    if (r.tag == RSP_T_HEADER) { header = true; qP = bitcast<u32>(r.value.y); tflags = bitcast<u32>(r.value.w); }
    if (r.tag != RSP_T_ANCHOR) { continue; }
    let w = bitcast<vec4u>(r.value);
    let role = w.x & 0xFFu;
    rsdbg_anchor(p.px, select(31u, 23u, role < 2u), w.x >> 8u, w.yzw);
  }
  if (!header) { return; }
  let vp = rs_vbuf(p.px);
  if ((tflags & TS_QVALID) != 0u && vp.x != 0xFFFFFFFFu) { rsdbg_anchor(p.px, 23u, 1u, vp.xyz); }
  if ((tflags & TS_INV_QUEUED) != 0u && qP < rs_atlas_pixels()) {
    rsdbg_vertex(p.px, 31u, 0u, frame.prevCam.camToWorld[3].xyz, 0xFu);
#if RS_VBUF_PREV_BINDING
    let vq = rs_vbuf_prev(vec2u(qP % rsParams.atlasSize.x, qP / rsParams.atlasSize.x));
    if (vq.x != 0xFFFFFFFFu) { rsdbg_anchor(p.px, 31u, 1u, vq.xyz); }
#endif
  }
}

@compute @workgroup_size(8, 8, 1)
fn rs_debug_views(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
  let probe = debug_is_probe(p.px);
  let boostView = dbg.mode == RSV_S_BOOST;
  let view = dbg.mode >= RSV_SHIFT_CODE && dbg.mode <= RSV_THR;
  if (!view && !probe && !boostView) { return; }
  if (dbg.mode == RSV_THR) { debug_write1(p.px, RSV_THR, rs_geo(p.px).w); }
  if (probe) { rsdbg_temporal_anchors(p); }
  let rounds = rsDispatch.round;
  if (rounds == 0u) { return; }             // no spatial stage ran this frame: the arena is stale (fill keeps NONE)
  let NS = min(rsParams.numSlots, RS_MAX_SLOTS);
  let f = rp_flags(rsdbg_src_plane(p.ai, RP_SEED));
  let replaySrc = rsdbg_needs_replay(f);
  let t = rs_t();
  var acceptMask = 0u;
  var replayMask = 0u;
  for (var s = 0u; s < NS; s++) {
    let Jw = arena_word(arena_slot_word(p.ai, s) + 3u);
    let code = arena_word(arena_code_word(p.ai, s));
    let sc = code & 0xFFu;
    let accepted = Jw != JW_NOT_ACCEPTED;
    let queued = accepted && sc != SC_EMPTY_SRC && replaySrc;
    acceptMask |= select(0u, 1u << s, accepted);
    replayMask |= select(0u, 1u << s, queued);
    if (view) {
      let valid = sc == SC_OK && Jw != JW_FAILED && Jw != JW_PENDING && Jw != JW_NOT_ACCEPTED;
      debug_write_code(p.px, RSV_SHIFT_CODE + s, sc);
      debug_write1(p.px, RSV_SHIFT_LOGJ + s, select(0.0, log2(bitcast<f32>(Jw)), valid));
      debug_write_code(p.px, RSV_SHIFT_TERM + s, ((code >> 8u) & 0xFu) | (((code >> 12u) & 0xFu) << 4u));
      debug_write1(p.px, RSV_SHIFT_MARGIN + s, unpack2x16float(code >> 16u).x);
    }
    if (probe) {
      let q = rsdbg_partner_ai(p, t, rounds - 1u, s);
      let flagsOut = s | select(0u, 0x100u, queued) | select(0u, 0x200u, q != 0xFFFFFFFFu);
      probe_record(p.px, RSP_SLOT_OUT, vec4f(rsdbg_bits(code), rsdbg_bits(Jw), rsdbg_bits(flagsOut), rsdbg_bits(q)));
      if (q != 0xFFFFFFFFu) {
        let pq = vec2u(q % rsParams.atlasSize.x, q / rsParams.atlasSize.x);
        let vq = rs_vbuf(pq);
        let vp = rs_vbuf(p.px);
        if (vq.x != 0xFFFFFFFFu) { rsdbg_anchor(p.px, 16u + s, 1u, vq.xyz); }
        if (vp.x != 0xFFFFFFFFu) { rsdbg_anchor(p.px, 24u + s, 1u, vp.xyz); }
        let fq = rp_flags(rsdbg_src_plane(q, RP_SEED));
        if (!res_empty(fq)) {
          if (rf_k(fq) >= 2u) { rsdbg_anchor(p.px, 24u + s, rf_k(fq), rsdbg_src_plane(q, RP_RC).xyz); }
          if (rf_tech(fq) == RS_TECH_BSDF_TRI && rf_k(fq) != rf_d(fq)) { rsdbg_anchor(p.px, 24u + s, rf_d(fq), rsdbg_src_plane(q, RP_END).xyz); }
        }
        let qw = arena_slot_word(q, s);
        let FJ = vec3f(bitcast<f32>(arena_word(qw)), bitcast<f32>(arena_word(qw + 1u)), bitcast<f32>(arena_word(qw + 2u)));
        probe_record(p.px, RSP_SLOT_IN, vec4f(rsdbg_bits(arena_word(arena_code_word(q, s))), rsdbg_bits(arena_word(qw + 3u)), rsdbg_bits(s), rsdbg_lum(FJ)));
      }
    }
  }
  if (boostView) {
    let fb = min(pair_first_boost_slot(), NS);
    debug_write_code(p.px, RSV_S_BOOST, (acceptMask >> fb) & ((1u << (NS - fb)) - 1u));
  }
  if (view) {
    for (var s = NS; s < RS_MAX_SLOTS; s++) {
      debug_write_code(p.px, RSV_SHIFT_CODE + s, DBG_CODE_NONE);
      debug_write_code(p.px, RSV_SHIFT_TERM + s, DBG_CODE_NONE);
    }
    debug_write_code(p.px, RSV_REPLAY_MASK, replayMask);
    debug_write_code(p.px, RSV_ACCEPT_MASK, acceptMask);
  }
}

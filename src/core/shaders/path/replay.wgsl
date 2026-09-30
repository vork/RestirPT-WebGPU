// Prefix replay of an offset path (restir-api.md §3.6; math.md#offset-checks O0/O1/O3, #rng-layout, #rr; gap-rc §6.2,
// §9.1). OWNER WP-B. Random replay from the destination primary y₁ with the source path's seed: vertex index b of the
// offset equals the base's b (y₁ ↔ x₁) and each replayed vertex consumes exactly the base's BSDF dims (slots 0–3 of
// vertex b through rs_path_hash); no NEE, no RR, no crossings, no RIS (math.md#rr: replay never applies RR). The loop
// is the base loop's steps (3)–(4) (restir-api.md §3.5): material_eval with V from positions, bsdf_sample,
// closest-hit trace, next vertex from ids (vertex_from_ids), wOut = normalize(pos(y_{b+1}) − pos(y_b)) (D3), the event
// e_b = (sampled lobe, p̄ of wOut), then the shared pair (y_{b−1}, y_b | EV_BSDF at y_b) through the one rcPairTest with
// the destination thr (O1 in prefix mode, O3 in ∅ mode).
//
//   prefix mode (kIdx = k ≥ 3): replays b = 1 … k−2; returns y_{k−1}, V at y_{k−1}, Tp = ∏ weights over y₁…y_{k−2}
//     and the pre-rc pair data (y_{k−2}, e_{k−2}); the pair (y_{k−2}, y_{k−1}) is tested by the shift with
//     EV_RECONNECT (never with the direction replay would sample at y_{k−1}).
//   ∅ mode (kIdx = 0, d ≥ 2): replays b = 1 … d−1; the last sample must end with the base's technique at the same d
//     (BSDF_ENV: escape; BSDF_TRI: hit on a TRI_EMISSIVE triangle; else SC_O0_TECH); the terminal pair (y_{d−1},
//     env | emitter as RCK_LIGHT) must fail too (O3, pair d); F = ω2·Tp·L with the PT's ω2 (1 after a delta lobe).
// An invalid sample, or a miss before the last vertex, is SC_O0_MISS (pair field = b). A zero throughput is NOT a
// decision here: the replay continues so that every "undefined" decision precedes any "zero" decision (restir-api.md
// §3.7); F = 0 becomes SC_ZERO in the shift. Inline budget (§4.5): one material_eval, bsdf_sample, bsdf_query site.
// RS_SHIFT_TRACE (tests): the includer defines rs_trace_vertex / rs_trace_pair / rs_trace_escape (restir-shift.gpu.test.ts).
#include "restir/rc.wgsl"
#include "restir/tframe.wgsl"
#include "path/path-weight.wgsl"
#include "restir/frame.wgsl"
#include "lights/env-sample.wgsl"

struct ReplayOut {
  code: u32,             // SC_OK | SC_O0_MISS | SC_O0_TECH | SC_O1 | SC_O3 (+ term, pair, margin packed as §2.6)
  margin: f32,
  yLast: SurfaceHit,     // prefix mode: y_{k−1};   ∅ mode: unused
  yLastPrim: u32,
  VLast: vec3f,          // incoming direction at y_{k−1} (from positions)
  Tp: vec3f,             // ∏ over replayed samples (prefix: y₁…y_{k−2}; ∅: y₁…y_{d−1}), RR-free
  preValid: bool,        // a pre-rc pair (y_{k−2}, y_{k−1}) exists (k−1 ≥ 2)
  preV: RcVertex, preE: RcEvent,   // y_{k−2} and its replayed event
  F: vec3f,              // ∅ mode: the replayed path's own integrand incl. ω2
}

/// kIdx = k (prefix mode, 3 ≤ k ≤ d) or 0 (∅ mode, d ≥ 2). tech = the source's technique (∅ mode). fs = the frame
/// whose light / env state evaluates the ∅ end term (M5, restir-temporal-api.md §3.4; RS_FS_CUR spatially).
fn replay_prefix(seed: vec2u, y1: SurfaceHit, y1Prim: u32, camPos: vec3f, thr: f32, kIdx: u32, d: u32, tech: u32, fs: u32) -> ReplayOut {
  var r: ReplayOut;
  r.code = rs_slot_code(SC_OK, RCT_NONE, 0u, 0.0);
  var cur = y1;
  var curPrim = y1Prim;
  var V = normalize(camPos - y1.pos);
  var Tp = vec3f(1.0);
  var prevV = RcVertex(camPos, vec3f(0.0), RCK_SURFACE, 0u);
  var prevE = rc_event_none();
  let empty = kIdx == 0u;
  let nB = select(kIdx - 2u, d - 1u, empty);            // replayed BSDF samples
  for (var b = 1u; b <= nB; b++) {
    let last = b == nB;
    let m = material_eval(cur, V);
    let curV = rc_vertex(cur, m);
    let bs = bsdf_sample(m, V, rs_path_bsdf_u4(seed, b));
    if (!bs.valid) {
#if RS_SHIFT_TRACE
      rs_trace_vertex(b, BVH_MISS, 0.0, 0.0, bs.lobe | 0x80u);
#endif
      r.code = rs_slot_code(SC_O0_MISS, RCT_NONE, b, 0.0);
      return r;
    }
    let org = offset_ray(cur.pos, select(-cur.ng, cur.ng, dot(cur.ng, bs.L) >= 0.0));
    let h = trace_closest_ex(org, bs.L, FLT_MAX, curPrim, BVH_MISS);
    let isHit = h.primId != BVH_MISS;
#if RS_SHIFT_TRACE
    rs_trace_vertex(b, h.primId, h.u, h.v, bs.lobe | select(0u, 8u, bs.is_delta));
    if (!isHit) { rs_trace_escape(bs.L); }
#endif
    if (!isHit && !(empty && last)) {
      r.code = rs_slot_code(SC_O0_MISS, RCT_NONE, b, 0.0);
      return r;
    }
    var nxt = cur;
    var wOut = bs.L;
    if (isHit) {
      nxt = vertex_from_ids(h.primId, h.u, h.v, cur.pos);
      wOut = normalize(nxt.pos - cur.pos);              // D3: same-formula direction
    }
    let qb = bsdf_query(m, V, wOut, bs.lobe);
    let eB = rc_event_bsdf(m, bs.lobe, bs.is_delta, qb.p_marg);
    // shared pair b = (y_{b−1}, y_b | EV_BSDF at y_b): must fail (O1 in prefix mode, O3 in ∅ mode)
    if (b >= 2u) {
      let pr = rcPairTest(prevV, prevE, curV, eB, thr);
#if RS_SHIFT_TRACE
      rs_trace_pair(b, pr.ok, pr.margin, pr.term);
#endif
      if (pr.ok) {
        r.code = rs_slot_code(select(SC_O1, SC_O3, empty), pr.term, b, pr.margin);
        r.margin = pr.margin;
        return r;
      }
    }
    Tp *= rs_path_weight(qb, bs.weight, bs.is_delta);
    if (empty && last) {
      // ∅: the same technique at the same d, then the terminal pair (y_{d−1}, env | emitter) must fail as well
      var Le = vec3f(0.0);
      var w2 = 1.0;
      var endV = RcVertex(cur.pos + bs.L, vec3f(0.0), RCK_ENV, 0u);
      if (tech == RS_TECH_BSDF_ENV) {
        let er = lf_env(fs);
        if (isHit || (er.flags & ENV_FLAG_PRESENT) == 0u) { r.code = rs_slot_code(SC_O0_TECH, RCT_NONE, d, 0.0); return r; }
        w2 = env_bsdf_mis_weight_s(bs.L, qb.p_marg, b, bs.is_delta, lf_slot(fs), er);
        Le = envRadiance_s(envUV(bs.L, er.cg, er.sg), er);
      } else {
        if (tech != RS_TECH_BSDF_TRI || !isHit || (nxt.triFlags & TRI_EMISSIVE) == 0u) {
          r.code = rs_slot_code(SC_O0_TECH, RCT_NONE, d, 0.0);
          return r;
        }
        Le = tri_emission(h.primId, h.u, h.v);
        if (!bs.is_delta) { w2 = mis_w2(tri_light_p1_s(cur.pos, nxt.pos, nxt.ng, h.primId, lf_slot(fs)), qb.p_marg, b); }
        endV = RcVertex(nxt.pos, nxt.ng, RCK_LIGHT, 0u);
      }
      let pt = rcPairTest(curV, eB, endV, rc_event_none(), thr);
#if RS_SHIFT_TRACE
      rs_trace_pair(d, pt.ok, pt.margin, pt.term);
#endif
      if (pt.ok) {
        r.code = rs_slot_code(SC_O3, pt.term, d, pt.margin);
        r.margin = pt.margin;
        return r;
      }
      r.F = w2 * Tp * Le;
    }
    prevV = curV;
    prevE = eB;
    cur = nxt;
    curPrim = h.primId;
    V = -wOut;
  }
  r.Tp = Tp;
  r.yLast = cur;
  r.yLastPrim = curPrim;
  r.VLast = V;
  r.preValid = !empty;
  r.preV = prevV;
  r.preE = prevE;
  return r;
}

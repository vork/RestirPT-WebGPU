// Hybrid shift T_{src→dst} (restir-api.md §3.7; math.md#jacobian, #offset-checks, #support-indicator, #visibility,
// #mis; PLAN §2 rules 3–7). OWNER WP-B.
//
// shift_hybrid(src, dst): the offset path ȳ of the source path x̄ in the destination pixel domain (primary y₁ from
// dst's rsVbuf ids, dst's thr, the frame camera). Order of evaluation (normative, §3.7; every "undefined" decision
// precedes any "zero" decision, so T and T⁻¹ classify identically):
//   1. empty source ⇒ SC_EMPTY_SRC.  y₁ = vertex_from_ids(dst ids, camPos).
//   2. replay (RS_REPLAY = 1, sources with k > 2 or k = ∅; path/replay.wgsl): O0 miss / technique, O1 on the replayed
//      pairs (prefix mode), O3 (∅ mode). ∅ ⇒ F = the replayed path's own integrand, J = 1.
//   3. reconnection at y_{k−1} (and x_k): the case of (k, d, technique) — letters of math.md#jacobian:
//        (a)/(f) forced NEE, k = d:  y_{d−1} must have a non-delta lobe (O0); O1 on the pre-rc pair with
//                EV_RECONNECT_NEE; no O2; J = 1.
//        (d) emitter rc, (e) env rc, k = d: EV_RECONNECT (copied ℓ_{d−1}, ω' toward z / the stored world ω); O1; O2
//                against a LIGHT / ENV vertex; O0 (joint pdf > 0, support indicator).
//        (b) N1 (k = d−1, NEE), (c) B1 (k = d−1, BSDF end), (deep) (k ≤ d−2): x_k from ids, V_k = −ω'; the x_k event
//                is EV_NEE of the re-evaluated stored endpoint (b) or EV_BSDF of the stored ω_k with the copied ℓ_k;
//                O1, O2 (recomputed p̄ at y_{k−1} and x_k), O0 at y_{k−1} and (c)/(deep) at x_k.
//   4. J = jNum / jDen (Eq. 2 with joint pdfs; 1 for (a), (f), ∅); not finite-positive ⇒ SC_J_INVALID; plants.
//   5. F non-finite ⇒ SC_NONFINITE; lum(F) = 0 ⇒ SC_ZERO; reconnection segment occluded ⇒ SC_OCCLUDED; else SC_OK.
// End terms of (b)/(c) are re-evaluated at x_k from the stored endpoint (D6); p1/p2 of every MIS weight are recomputed
// at the offset's own x_{d−1} (math.md#mis). Directions between vertices come from positions rebuilt from ids (D3).
// Inline budget (§4.5): the reconnection loop has one material_eval and one bsdf_query call site (≤ 2 iterations:
// y_{k−1}, then x_k), replay one of each plus the only bsdf_sample.
// Compile-time switches: RS_REPLAY (replay compiled in), RS_SHIFT_TRACE (tests: the includer defines rs_trace_vertex,
// rs_trace_pair, rs_trace_light and rs_trace_recon).
#include "restir/reservoir.wgsl"
#include "restir/endpoint.wgsl"
#include "restir/rc.wgsl"
#include "path/replay.wgsl"

struct ShiftSrc { empty: bool, flags: u32, seed: vec2u, F: vec3f, rc: vec3u, jDen: f32, rcWi: vec3f, aux: f32,
                  rcRad: vec3f, end: vec3u }
struct ShiftDst { valid: bool, prim: u32, bary: vec2f, thr: f32, camPos: vec3f }
struct ShiftOut { FJ: vec3f, J: f32, jNum: f32, code: u32 }   // code packed as §2.6; FJ, J = 0 unless SC_OK

/// Non-empty ∧ (k > 2 ∨ k = ∅): the offset needs its prefix replayed (y_{k−1} ≠ y₁).
fn res_needs_replay(flags: u32) -> bool {
  let k = rf_k(flags);
  return !res_empty(flags) && (k > 2u || k == 0u);
}

#if RS_RES_IN_BINDING
fn shift_src_load(ai: u32) -> ShiftSrc {
  var s: ShiftSrc;
  let p0 = resin_plane(ai, RP_WF);
  let p1 = resin_plane(ai, RP_SEED);
  let p2 = resin_plane(ai, RP_RC);
  let p3 = resin_plane(ai, RP_WI);
  let p4 = resin_plane(ai, RP_RAD);
  let p5 = resin_plane(ai, RP_END);
  s.flags = p1.z;
  s.empty = res_empty(s.flags);
  s.seed = p1.xy;
  s.F = rp_F(p0);
  s.rc = p2.xyz;
  s.jDen = bitcast<f32>(p2.w);
  s.rcWi = bitcast<vec3f>(p3.xyz);
  s.aux = bitcast<f32>(p3.w);
  s.rcRad = bitcast<vec3f>(p4.xyz);
  s.end = p5.xyz;
  return s;
}
#endif

#if RS_VBUF_BINDING && RS_GEO_BINDING
fn shift_dst_load(px: vec2u) -> ShiftDst {
  let vb = rs_vbuf(px);
  var d: ShiftDst;
  d.valid = vb.x != 0xFFFFFFFFu;
  d.prim = vb.x;
  d.bary = bitcast<vec2f>(vb.yz);
  d.thr = rs_geo(px).w;
  d.camPos = rs_cam_pos();
  return d;
}
#endif

// Reconnection cases (letters of math.md#jacobian)
const SH_FORCED: u32 = 0u;   // (a)/(f)
const SH_EMIT: u32 = 1u;     // (d)
const SH_ENV: u32 = 2u;      // (e)
const SH_N1: u32 = 3u;       // (b)
const SH_B1: u32 = 4u;       // (c)
const SH_DEEP: u32 = 5u;     // (deep)
const SH_NONE: u32 = 6u;     // ∅
const SH_BAD: u32 = 7u;      // not an M4 configuration (e.g. BSDF_ANALYTIC)

fn shift_case(f: u32) -> u32 {
  let d = rf_d(f);
  let k = rf_k(f);
  let tech = rf_tech(f);
  if (k == 0u) { return select(SH_BAD, SH_NONE, tech == RS_TECH_BSDF_TRI || tech == RS_TECH_BSDF_ENV); }
  if (k < 2u || k > d || tech == RS_TECH_BSDF_ANALYTIC) { return SH_BAD; }
  if (k == d) {
    if (tech == RS_TECH_NEE) { return SH_FORCED; }
    return select(SH_ENV, SH_EMIT, tech == RS_TECH_BSDF_TRI);
  }
  if (k == d - 1u) { return select(SH_B1, SH_N1, tech == RS_TECH_NEE); }
  return SH_DEEP;
}

fn shift_fail(sc: u32, term: u32, pair: u32, margin: f32) -> ShiftOut {
  var o: ShiftOut;
  o.code = rs_slot_code(sc, term, pair, margin);
  return o;
}

/// The hybrid shift (restir-api.md §3.7). FJ = F_dst(ȳ)·J and J only for SC_OK.
fn shift_hybrid(src: ShiftSrc, dst: ShiftDst) -> ShiftOut {
  if (src.empty) { return shift_fail(SC_EMPTY_SRC, RCT_NONE, 0u, 0.0); }
  if (!dst.valid) { return shift_fail(SC_O0_MISS, RCT_NONE, 1u, 0.0); }
  let f = src.flags;
  let d = rf_d(f);
  let k = rf_k(f);
  let cs = shift_case(f);
  if (cs == SH_BAD) { return shift_fail(SC_O0_TECH, RCT_NONE, 0u, 0.0); }
  let camPos = dst.camPos;
  let thr = dst.thr;
  let B = d - 1u;                                        // vertex index of the offset's x_{d−1}
  var yPrev = vertex_from_ids(dst.prim, dst.bary.x, dst.bary.y, camPos);
  var yPrevPrim = dst.prim;
  var VPrev = normalize(camPos - yPrev.pos);
  var Tp = vec3f(1.0);
  var preValid = false;
  var preV = RcVertex(camPos, vec3f(0.0), RCK_SURFACE, 0u);
  var preE = rc_event_none();
  // ---- 2. replay -------------------------------------------------------------------------------------------------------
  if (k > 2u || k == 0u) {
#if RS_REPLAY
    let rp = replay_prefix(src.seed, yPrev, yPrevPrim, camPos, thr, k, d, rf_tech(f));
    if (rs_slot_code_sc(rp.code) != SC_OK) { var o: ShiftOut; o.code = rp.code; return o; }
    if (cs == SH_NONE) { return shift_finish(rp.F, 1.0, 1.0, true); }
    yPrev = rp.yLast;
    yPrevPrim = rp.yLastPrim;
    VPrev = rp.VLast;
    Tp = rp.Tp;
    preValid = rp.preValid;
    preV = rp.preV;
    preE = rp.preE;
#else
    return shift_fail(SC_O0_MISS, RCT_NONE, 0u, 0.0);    // caller bug: a replay source in a non-replay pipeline
#endif
  }
  // ---- 3. reconnection -------------------------------------------------------------------------------------------------
  let lkm1 = rf_lkm1(f);
  let lk = rf_lk(f);
  let dk = (f & RF_DK) != 0u;
  let neeAtY = cs == SH_FORCED;
  let surfK = cs == SH_N1 || cs == SH_B1 || cs == SH_DEEP;
  let ep = nee_endpoint_from_words(src.end);
  // x_k (surface rc) or z (emitter rc), flipped toward y_{k−1}; ω' = normalize(x_k − y_{k−1}) (D3)
  var xk = yPrev;
  var xkPrim = BVH_MISS;
  var wP = src.rcWi;                                     // (e): the stored world escape direction
  if (surfK || cs == SH_EMIT) {
    xkPrim = src.rc.x;
    xk = vertex_from_ids(src.rc.x, bitcast<f32>(src.rc.y), bitcast<f32>(src.rc.z), yPrev.pos);
    wP = normalize(xk.pos - yPrev.pos);
  }
  var ls = light_sample_none();
  var yV = RcVertex(yPrev.pos, yPrev.ng, RCK_SURFACE, 0u);
  var eY = rc_event_none();
  var fY = vec3f(0.0);  var pY = 0.0;  var pYm = 0.0;  var suppY = true;
  var fK = vec3f(1.0);  var pK = 1.0;  var pKm = 0.0;  var suppK = true;
  let nIt = select(1u, 2u, surfK);
  for (var it = 0u; it < nIt; it++) {
    let atY = it == 0u;
    var s = yPrev;
    var V = VPrev;
    var L = wP;
    var lobe = lkm1;
    if (!atY) { s = xk; V = -wP; L = src.rcWi; lobe = lk; }
    if ((atY && neeAtY) || (!atY && cs == SH_N1)) {
      ls = nee_eval(s.pos, ep);
      L = ls.dir;
      lobe = LOBE_NEE;
#if RS_SHIFT_TRACE
      rs_trace_light(ls.dir, ls.pos, ls.isInf);
#endif
    }
    let m = material_eval(s, V);
    let q = bsdf_query(m, V, L, lobe);
    if (atY) {
      if (neeAtY && (m.flags & MATEVAL_HAS_NON_DELTA) == 0u) { return shift_fail(SC_O0_LOBE, RCT_NONE, k - 1u, 0.0); }
      yV = rc_vertex(yPrev, m);
      if (neeAtY) { eY = rc_event_nee(m, q.p_marg); fY = q.f_all; }
      else { eY = rc_event_bsdf(m, lkm1, false, q.p_marg); fY = q.f_lobe; }
      pY = q.p_joint; pYm = q.p_marg; suppY = q.supp;
      // O1: the last pre-rc pair (y_{k−2}, y_{k−1}) with EV_RECONNECT(_NEE) at y_{k−1}
      if (preValid) {
        let pr = rcPairTest(preV, preE, yV, eY, thr);
#if RS_SHIFT_TRACE
        rs_trace_pair(k - 1u, pr.ok, pr.margin, pr.term);
#endif
        if (pr.ok) { return shift_fail(SC_O1, pr.term, k - 1u, pr.margin); }
      }
      // O2 of the single-iteration cases: (y_{d−1}, z | LIGHT) and (y_{d−1}, env | ENV)
      if (cs == SH_EMIT || cs == SH_ENV) {
        var zV = RcVertex(yPrev.pos + wP, vec3f(0.0), RCK_ENV, 0u);
        if (cs == SH_EMIT) { zV = RcVertex(xk.pos, xk.ng, RCK_LIGHT, 0u); }
        let pr = rcPairTest(yV, eY, zV, rc_event_none(), thr);
#if RS_SHIFT_TRACE
        rs_trace_pair(k, pr.ok, pr.margin, pr.term);
#endif
        if (!pr.ok) { return shift_fail(SC_O2, pr.term, k, pr.margin); }
      }
    } else {
      let xV = rc_vertex(xk, m);
      var eK = rc_event_nee(m, q.p_marg);
      fK = q.f_all;
      pK = 1.0;
      if (cs != SH_N1) { eK = rc_event_bsdf(m, lk, dk, q.p_marg); fK = q.f_lobe; pK = q.p_joint; }
      pKm = q.p_marg; suppK = q.supp;
      // O2: the rc pair (y_{k−1}, x_k) with recomputed p̄^y_{k−1}(ω') and p̄^y_k(ω_k | from y_{k−1})
      let pr = rcPairTest(yV, eY, xV, eK, thr);
#if RS_SHIFT_TRACE
      rs_trace_pair(k, pr.ok, pr.margin, pr.term);
#endif
      if (!pr.ok) { return shift_fail(SC_O2, pr.term, k, pr.margin); }
    }
  }
  // O0: copied-lobe joint pdfs > 0 and the sampler-support indicator (never on NEE segments)
  if (!neeAtY) {
    if (!rs_pos_finite(pY)) { return shift_fail(SC_O0_LOBE, RCT_NONE, k - 1u, 0.0); }
    if (!suppY) { return shift_fail(SC_O0_SUPPORT, RCT_NONE, k - 1u, 0.0); }
  }
  if (cs == SH_B1 || cs == SH_DEEP) {
    if (!rs_pos_finite(pK)) { return shift_fail(SC_O0_LOBE, RCT_NONE, k, 0.0); }
    if (!suppK) { return shift_fail(SC_O0_SUPPORT, RCT_NONE, k, 0.0); }
  }
  // ---- 4. Jacobian (Eq. 2, joint pdfs; RSF_PLANT_MARGINAL_J: marginal pdfs, U-11 negative control) --------------------
  let marg = (rsParams.flags & RSF_PLANT_MARGINAL_J) != 0u;
  let pYj = select(pY, pYm, marg);
  let pKj = select(pK, pKm, marg);
  var jNum = 1.0;
  if (cs == SH_ENV) { jNum = pYj; }
  else if (cs == SH_EMIT || cs == SH_N1) { jNum = pYj * rc_G(yPrev.pos, xk.pos, xk.ng); }
  else if (cs != SH_FORCED) { jNum = pYj * rc_G(yPrev.pos, xk.pos, xk.ng) * pKj; }
  var J = 1.0;
  if (cs != SH_FORCED) { J = jNum / src.jDen; }
#if RS_SHIFT_TRACE
  rs_trace_recon(abs(dot(yPrev.ng, wP)), select(1.0, abs(dot(xk.ng, src.rcWi)), cs == SH_B1 || cs == SH_DEEP));
#endif
  // ---- 5. integrand F(ȳ) in the destination domain ---------------------------------------------------------------------
  var F = vec3f(0.0);
  switch cs {
    case SH_FORCED: {                                    // (a)/(f): ω1 and p2 recomputed at y_{d−1}
      if (ls.valid && ls.prim != yPrevPrim && any(ls.Lambda > vec3f(0.0))) {
        let w1 = nee_mis_w1(ls, pYm, B);
        F = Tp * ((w1 / ls.q) * (fY * ls.Lambda));
      }
    }
    case SH_EMIT: {                                      // (d): ω2 with p1 recomputed at y_{d−1}
      let w2 = mis_w2(tri_light_p1(yPrev.pos, xk.pos, xk.ng, xkPrim), pYm, B);
      F = Tp * (fY / pY) * (w2 * tri_emission(xkPrim, bitcast<f32>(src.rc.y), bitcast<f32>(src.rc.z)));
    }
    case SH_ENV: {                                       // (e)
      let w2 = env_bsdf_mis_weight(wP, pYm, B, false);
      F = Tp * (fY / pY) * (w2 * envRadiance(envUV(wP, envParams.cg, envParams.sg)));
    }
    case SH_N1: {                                        // (b): NEE end term re-evaluated at x_{d−1} (D6), p2 changes
      if (ls.valid && any(ls.Lambda > vec3f(0.0))) {
        let w1 = nee_mis_w1(ls, pKm, B);
        F = Tp * (fY / pY) * ((w1 / ls.q) * (fK * ls.Lambda));
      }
    }
    case SH_B1: {                                        // (c): emitter / env end term re-evaluated at x_{d−1} (D6)
      var endT = vec3f(0.0);
      if (src.end.x == RC_ENV_DIR) {
        endT = env_bsdf_mis_weight(src.rcWi, pKm, B, false) * envRadiance(envUV(src.rcWi, envParams.cg, envParams.sg));
      } else {
        let z = vertex_from_ids(src.end.x, bitcast<f32>(src.end.y), bitcast<f32>(src.end.z), xk.pos);
        endT = mis_w2(tri_light_p1(xk.pos, z.pos, z.ng, src.end.x), pKm, B) * tri_emission(src.end.x, bitcast<f32>(src.end.y), bitcast<f32>(src.end.z));
      }
      F = Tp * (fY / pY) * (fK / pK) * endT;
    }
    default: {                                           // (deep): the stored post-rc suffix (MIS included)
      F = Tp * (fY / pY) * (fK / pK) * src.rcRad;
    }
  }
  return shift_finish_vis(F, J, jNum, cs, yPrev, yPrevPrim, xk, xkPrim, wP, ls);
}

/// Steps 4–5 for ∅ (no reconnection segment): J guard, plants, NONFINITE, ZERO.
fn shift_finish(F: vec3f, J: f32, jNum: f32, visible: bool) -> ShiftOut {
  var o: ShiftOut;
  o.jNum = jNum;
  if (!rs_pos_finite(J)) { o.code = rs_slot_code(SC_J_INVALID, RCT_NONE, 0u, 0.0); return o; }
  let Jp = select(J, 1.0, (rsParams.flags & RSF_PLANT_NO_J) != 0u);
  if (!all_finite3(F)) { o.code = rs_slot_code(SC_NONFINITE, RCT_NONE, 0u, 0.0); return o; }
  if (!(luminance(F) > 0.0)) { o.code = rs_slot_code(SC_ZERO, RCT_NONE, 0u, 0.0); return o; }
  if (!visible) { o.code = rs_slot_code(SC_OCCLUDED, RCT_NONE, 0u, 0.0); return o; }
  o.code = rs_slot_code(SC_OK, RCT_NONE, 0u, 0.0);
  o.J = Jp;
  o.FJ = F * Jp;
  return o;
}

/// Steps 4–5 with the reconnection visibility (traced only when F > 0 and the shift is defined).
fn shift_finish_vis(F: vec3f, J: f32, jNum: f32, cs: u32, y: SurfaceHit, yPrim: u32, xk: SurfaceHit, xkPrim: u32, wP: vec3f,
                    ls: LightSample) -> ShiftOut {
  var o = shift_finish(F, J, jNum, true);
  if (rs_slot_code_sc(o.code) != SC_OK) { return o; }
  var vis = true;
  if (cs == SH_FORCED) { vis = nee_visible(y, yPrim, ls); }
  else if (cs == SH_ENV) { vis = visibleInf(y.pos, y.ng, yPrim, wP); }
  else { vis = visible(y.pos, y.ng, yPrim, xk.pos, xk.ng, xkPrim); }
  if (!vis) { return shift_finish(F, J, jNum, false); }
  return o;
}

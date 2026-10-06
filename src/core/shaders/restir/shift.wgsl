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
// M5 (restir-temporal-api.md §3.4, TD10; OWNER T-B in M5): ONE shift for spatial, T and T⁻¹. ShiftDst.fs selects the
// light / env state of the destination domain (every light and env term goes through the `_s` evaluators with
// lf_slot(dst.fs) / lf_env(dst.fs), fetched from the uniforms at the call site: Metal Q1); ShiftSrc.endVis is the N1
// end visibility under the destination frame, stored inverted as ShiftSrc.endOcc so that a zero-initialised source is
// the M4 source (restir-temporal-api.md Changelog B-1; case (b): vis = visible(y_{k−1}, x_k) ∧ ¬endOcc; an occluded end
// is a defined zero, SC_OCCLUDED, after every undefined test); ShiftOut.F is the destination integrand before the J
// multiply (SC_OK only). Spatial callers pass fs = CUR, endOcc = false: bitwise M4 (U-M4-BITS).
#include "restir/tframe.wgsl"
#include "restir/reservoir.wgsl"
#include "restir/endpoint.wgsl"
#include "restir/rc.wgsl"
#include "path/replay.wgsl"
#if RS_RIS_NEE
#include "restir/m6-types.wgsl"
#endif
#if RS_MODE_B
#include "restir/cross.wgsl"
#endif
#if RS_PLANT_T2
#include "restir/m6-types.wgsl"
/// U8-4 plant (validation only): atlas index of the source record of the current spatial shift (set by the caller).
var<private> rsT2SrcAi: u32 = 0xFFFFFFFFu;
#endif

struct ShiftSrc { empty: bool, flags: u32, seed: vec2u, F: vec3f, rc: vec3u, jDen: f32, rcWi: vec3f, aux: f32,
                  rcRad: vec3f, end: vec3u, endOcc: bool }                // M5: N1 end occluded (¬endVis; false spatially)
struct ShiftDst { valid: bool, prim: u32, bary: vec2f, thr: f32, camPos: vec3f, fs: u32 }   // M5: fs (CUR spatially)
struct ShiftOut { FJ: vec3f, J: f32, jNum: f32, code: u32, F: vec3f }   // code packed as §2.6; FJ, J, F = 0 unless SC_OK

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
  s.endOcc = false;
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
  d.fs = RS_FS_CUR;
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

#if RS_MODE_B
const SH_XEMIT: u32 = 8u;    // (d-ana): the crossed analytic-light point is the rc vertex (restir-m6-api.md MD7)
fn shift_case(f: u32) -> u32 {
  let d = rf_d(f);
  let k = rf_k(f);
  let tech = rf_tech(f);
  if (k == 0u) { return SH_NONE; }
  if (k < 2u || k > d) { return SH_BAD; }
  if (k == d) {
    if (tech == RS_TECH_NEE) { return SH_FORCED; }
    if (tech == RS_TECH_BSDF_ANALYTIC) { return SH_XEMIT; }
    return select(SH_ENV, SH_EMIT, tech == RS_TECH_BSDF_TRI);
  }
  if (k == d - 1u) { return select(SH_B1, SH_N1, tech == RS_TECH_NEE); }
  return SH_DEEP;
}
#else
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
#endif

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
#if RS_MODE_B
    rsReplayCrossEntry = src.end.x & RC_ENTRY_MASK;
#endif
    let rp = replay_prefix(src.seed, yPrev, yPrevPrim, camPos, thr, k, d, rf_tech(f), dst.fs);
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
#if RS_MODE_B
  if (cs == SH_XEMIT) {                                  // (d-ana): z = the stored crossing of the frame's light record
    let slotX = lf_slot(dst.fs);
    let eX = src.rc.x & RC_ENTRY_MASK;
    if (eX >= slotX.nAnalytic) { return shift_fail(SC_O0_LIGHT, RCT_NONE, 0u, 0.0); }
    let rX = light_load(slotX, eX);
    xk.pos = cross_point(rX, bitcast<vec2f>(src.rc.yz));
    xk.ng = rX.normal;
    xkPrim = LIGHT_NONE;
    wP = normalize(xk.pos - yPrev.pos);
  }
#endif
  // NEE end term and its visibility data, reduced to a few registers inside the iteration that evaluates the light
  // sample (the full LightSample is not kept live across the rest of the shift and the visibility traversal:
  // docs/decisions/platform-lanes.md "Metal quirks", M4 T3 control-flow fault)
  var neeT = vec3f(0.0);                                 // (ω1/q)·(f_all ⊙ Λ); 0 for an invalid / same-triangle / Λ = 0 sample
  var visPos = vec3f(0.0);
  var visN = vec3f(0.0);
  var visPrim = BVH_MISS;
  var visInf = false;
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
    let neeHere = (atY && neeAtY) || (!atY && cs == SH_N1);
    var ls = light_sample_none();
    if (neeHere) {
      ls = nee_eval_s(s.pos, ep, lf_slot(dst.fs), lf_env(dst.fs));
      if (shift_u8_light_plants_on(dst.fs)) { ls = shift_u8_light_plant(ls, s.pos, ep.entry, dst.fs); }
      L = ls.dir;
      lobe = LOBE_NEE;
#if RS_SHIFT_TRACE
      rs_trace_light(ls.dir, ls.pos, ls.isInf);
#endif
    }
    let m = material_eval(s, V);
    let q = bsdf_query(m, V, L, lobe);
    if (neeHere) {                                       // (a)/(f) at y_{d−1}, (b) at x_{d−1}: p2 = this vertex's p̄
      if (ls.valid && (!atY || ls.prim != yPrevPrim) && any(ls.Lambda > vec3f(0.0))) {
#if RS_RIS_NEE
        var w1 = nee_mis_w1(ls, rs_p2m(q.p_marg, B), B);   // M6 MD5: M(B) as p2/M (B = d−1; M = 1 unless B = 1)
#else
        var w1 = nee_mis_w1(ls, q.p_marg, B);
#endif
        if (shift_u8_plants_on()) { w1 = shift_u8_w1(ls, q.p_marg, B, w1, !atY && cs == SH_N1, src.aux); }
        neeT = (w1 / ls.q) * (q.f_all * ls.Lambda);
      }
      visPos = ls.pos; visN = ls.nz; visPrim = ls.prim; visInf = ls.isInf; wP = select(wP, ls.dir, atY);
    }
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
#if RS_MODE_B
      if (cs == SH_EMIT || cs == SH_ENV || cs == SH_XEMIT) {
        var zV = RcVertex(yPrev.pos + wP, vec3f(0.0), RCK_ENV, 0u);
        if (cs == SH_EMIT || cs == SH_XEMIT) { zV = RcVertex(xk.pos, xk.ng, RCK_LIGHT, 0u); }
#else
      if (cs == SH_EMIT || cs == SH_ENV) {
        var zV = RcVertex(yPrev.pos + wP, vec3f(0.0), RCK_ENV, 0u);
        if (cs == SH_EMIT) { zV = RcVertex(xk.pos, xk.ng, RCK_LIGHT, 0u); }
#endif
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
  // PLANT U8-3 (RSF_PLANT_U8_NO_PK): J without the p^y_k factor (cases (c) and deep). A select on the factor, not a
  // branch on jNum: the branch changed the Metal code of the default path (U-M4-BITS, restir-temporal-api.md B-8).
  let pKj = select(select(pK, pKm, marg), 1.0, (rsParams.flags & RSF_PLANT_U8_NO_PK) != 0u);
  var jNum = 1.0;
  if (cs == SH_ENV) { jNum = pYj; }
#if RS_MODE_B
  else if (cs == SH_EMIT || cs == SH_N1 || cs == SH_XEMIT) { jNum = pYj * rc_G(yPrev.pos, xk.pos, xk.ng); }
#else
  else if (cs == SH_EMIT || cs == SH_N1) { jNum = pYj * rc_G(yPrev.pos, xk.pos, xk.ng); }
#endif
  else if (cs != SH_FORCED) { jNum = pYj * rc_G(yPrev.pos, xk.pos, xk.ng) * pKj; }
  var J = 1.0;
  if (cs != SH_FORCED) { J = jNum / src.jDen; }
#if RS_PLANT_T2
  // PLANT U8-4 (validation only, spatial shifts): J = t_x²/t_y² for a forced point / spot endpoint (Cycles' pseudo-pdf
  // t² used in a ratio; the correct light-vertex J is 1, gap-light §5.7)
  if (cs == SH_FORCED && rs_m6_flag(RSF_PLANT_U8_T2) && rsT2SrcAi != 0xFFFFFFFFu && !visInf && visPrim == BVH_MISS) {
    let lt = rf_ep(f);
    if (lt == LT_POINT || lt == LT_SPOT) {
      let sx = resin_plane(rsT2SrcAi, RP_SFX0);
      let xs = scene_surface(sx.x, bitcast<f32>(sx.y), bitcast<f32>(sx.z), vec3f(0.0)).pos;
      let tx = xs - visPos;
      let ty = yPrev.pos - visPos;
      J = dot(tx, tx) / max(dot(ty, ty), 1e-30);
    }
  }
#endif
#if RS_SHIFT_TRACE
  rs_trace_recon(min(abs(dot(yPrev.ng, wP)), select(1.0, abs(dot(xk.ng, wP)), xkPrim != BVH_MISS)),
                 select(1.0, abs(dot(xk.ng, src.rcWi)), cs == SH_B1 || cs == SH_DEEP), length(xk.pos - yPrev.pos),
                 min(min(bitcast<f32>(src.rc.y), bitcast<f32>(src.rc.z)), 1.0 - bitcast<f32>(src.rc.y) - bitcast<f32>(src.rc.z)));
#endif
  // ---- 5. integrand F(ȳ) in the destination domain ---------------------------------------------------------------------
  var F = vec3f(0.0);
  switch cs {
    case SH_FORCED: {                                    // (a)/(f): ω1 and p2 recomputed at y_{d−1}
      F = Tp * neeT;
    }
    case SH_EMIT: {                                      // (d): ω2 with p1 recomputed at y_{d−1}
#if RS_RIS_NEE
      let w2 = mis_w2(tri_light_p1_s(yPrev.pos, xk.pos, xk.ng, xkPrim, lf_slot(dst.fs)), rs_p2m(pYm, B), B);
#else
      let w2 = mis_w2(tri_light_p1_s(yPrev.pos, xk.pos, xk.ng, xkPrim, lf_slot(dst.fs)), pYm, B);
#endif
      F = Tp * (fY / pY) * (w2 * tri_emission(xkPrim, bitcast<f32>(src.rc.y), bitcast<f32>(src.rc.z)));
    }
    case SH_ENV: {                                       // (e)
#if RS_RIS_NEE
      let w2 = env_bsdf_mis_weight_s(wP, rs_p2m(pYm, B), B, false, lf_slot(dst.fs), lf_env(dst.fs));
#else
      let w2 = env_bsdf_mis_weight_s(wP, pYm, B, false, lf_slot(dst.fs), lf_env(dst.fs));
#endif
      let er = lf_env(dst.fs);
      F = Tp * (fY / pY) * (w2 * envRadiance_s(envUV(wP, er.cg, er.sg), er));
    }
    case SH_N1: {                                        // (b): NEE end term re-evaluated at x_{d−1} (D6), p2 changes
      F = Tp * (fY / pY) * neeT;
    }
#if RS_MODE_B
    case SH_XEMIT: {                                     // (d-ana): one-sided L_e of the copied point, ω2 with p1 at y_{d−1}
      let slotX = lf_slot(dst.fs);
      let eX = src.rc.x & RC_ENTRY_MASK;
      let rX = light_load(slotX, eX);
      var w2 = 1.0;
#if RS_RIS_NEE
      if (cross_mis(false)) { w2 = mis_w2(analytic_area_p1_s(yPrev.pos, eX, xk.pos, slotX), rs_p2m(pYm, B), B); }
#else
      if (cross_mis(false)) { w2 = mis_w2(analytic_area_p1_s(yPrev.pos, eX, xk.pos, slotX), pYm, B); }
#endif
      F = Tp * (fY / pY) * (w2 * area_radiance(rX, -wP));
    }
#endif
    case SH_B1: {                                        // (c): emitter / env end term re-evaluated at x_{d−1} (D6)
      var endT = vec3f(0.0);
      if (src.end.x == RC_ENV_DIR) {
        let er = lf_env(dst.fs);
        var w2 = env_bsdf_mis_weight_s(src.rcWi, pKm, B, false, lf_slot(dst.fs), er);
        if (rs_tplant(TP_U8_STALE_AUX)) { w2 = mis_w2(src.aux, pKm, B); }                  // PLANT U8-2t: stale p1
        endT = w2 * envRadiance_s(envUV(src.rcWi, er.cg, er.sg), er);
#if RS_MODE_B
      } else if (rs_is_cross_words(src.end)) {          // (c-ana): the crossing re-intersected along ω_k (MD7)
        let ce = cross_end(xk.pos, src.rcWi, src.end.x & RC_ENTRY_MASK, lf_slot(dst.fs));
        var p1 = ce.p1;
        if (rs_tplant(TP_U8_STALE_AUX)) { p1 = src.aux; }                                 // PLANT U8-2t: stale p1
        var w2 = 1.0;
        if (cross_mis(false)) { w2 = mis_w2(p1, pKm, B); }
        endT = select(vec3f(0.0), w2 * ce.Le, ce.ok);
#endif
      } else {
        let z = vertex_from_ids(src.end.x, bitcast<f32>(src.end.y), bitcast<f32>(src.end.z), xk.pos);
        var p1 = tri_light_p1_s(xk.pos, z.pos, z.ng, src.end.x, lf_slot(dst.fs));
        if (rs_tplant(TP_U8_STALE_AUX)) { p1 = src.aux; }                                 // PLANT U8-2t: stale p1
        endT = mis_w2(p1, pKm, B) * tri_emission(src.end.x, bitcast<f32>(src.end.y), bitcast<f32>(src.end.z));
      }
      F = Tp * (fY / pY) * (fK / pK) * endT;
    }
    default: {                                           // (deep): the stored post-rc suffix (MIS included)
      F = Tp * (fY / pY) * (fK / pK) * src.rcRad;
    }
  }
  return shift_finish_vis(F, J, jNum, cs, yPrev, yPrevPrim, xk, xkPrim, wP, visPos, visN, visPrim, visInf, src.endOcc);
}

// ---- U8 planted controls (restir-temporal-api.md TD30, §6.5; validation only, uniform switches) -----------------------
fn shift_u8_plants_on() -> bool {
  return (rsParams.flags & RSF_PLANT_U8_W1DELTA) != 0u || rs_tplant(TP_U8_STALE_AUX);
}
fn shift_u8_light_plants_on(fs: u32) -> bool {
  return (rsParams.flags & RSF_PLANT_U8_ONESIDED) != 0u
    || (rs_tplant(TP_U8_SPOT_PREV_AXIS) && fs == RS_FS_CUR && !rs_tf(TF_LIGHTS_SAME) && rs_tf(TF_HIST_VALID));
}
/// U8-1 (ω1 < 1 for a delta light: 1/(1 + p2)) and U8-2t (case (b): ω1 with the stored, possibly stale p1 = aux).
fn shift_u8_w1(ls: LightSample, p2: f32, B: u32, w1: f32, isN1: bool, aux: f32) -> f32 {
  var w = w1;
  if ((rsParams.flags & RSF_PLANT_U8_W1DELTA) != 0u && ls.isDelta) { w = 1.0 / (1.0 + max(p2, 0.0)); }
  if (rs_tplant(TP_U8_STALE_AUX) && isN1 && !ls.isDelta) { var l2 = ls; l2.p1 = aux; w = nee_mis_w1(l2, p2, B); }
  return w;
}
/// U8-6 (one-sidedness of rect / disk lights ignored at the offset: back-side samples emit with |cos|) and U8-5t (the
/// spot profile of the current frame evaluated with the spot axis of frame t−1, forward shifts on light-change frames).
fn shift_u8_light_plant(ls0: LightSample, x: vec3f, entry: u32, fs: u32) -> LightSample {
  var ls = ls0;
  let slot = lf_slot(fs);
  if (!ls.valid || entry >= slot.nAnalytic) { return ls; }
  let r = light_load(slot, entry);
  if ((rsParams.flags & RSF_PLANT_U8_ONESIDED) != 0u && (r.kind == LT_RECT || r.kind == LT_DISK)) {
    let c = dot(-ls.dir, r.normal);
    if (c < 0.0) {
      let ac = -c;
      let d2 = ls.dist * ls.dist;
      ls.cosZ = ac;
      ls.Lambda = r.emit * (area_spread(r, ac) * ac / d2);
      ls.q = light_pmf(slot, entry) * r.invArea * d2 / ac;
      ls.p1 = ls.q;
    }
  }
  if (rs_tplant(TP_U8_SPOT_PREV_AXIS) && fs == RS_FS_CUR && r.kind == LT_SPOT) {
    let e0 = lt_translate(entry, RS_FS_CUR, RS_FS_PREV);
    if (e0 != LIGHT_NONE) {
      let rp = light_load(lf_slot(RS_FS_PREV), e0);
      ls.Lambda = r.emit * (spot_profile(r, dot(-ls.dir, rp.normal)) / (ls.dist * ls.dist));
    }
  }
  return ls;
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
  o.F = F;
  return o;
}

/// Steps 4–5 with the reconnection visibility (traced only when F > 0 and the shift is defined).
fn shift_finish_vis(F: vec3f, J: f32, jNum: f32, cs: u32, y: SurfaceHit, yPrim: u32, xk: SurfaceHit, xkPrim: u32, wP: vec3f,
                    visPos: vec3f, visN: vec3f, visPrim: u32, visInf: bool, endOcc: bool) -> ShiftOut {
  var o = shift_finish(F, J, jNum, true);
  if (rs_slot_code_sc(o.code) != SC_OK) { return o; }
  var vis = true;
  if (cs == SH_FORCED) {                                 // = nee_visible(y, yPrim, ls) (endpoint.wgsl), from the kept fields
    if (visInf) { vis = visibleInf(y.pos, y.ng, yPrim, wP); }
    else { vis = visible(y.pos, y.ng, yPrim, visPos, visN, visPrim); }
  }
  else if (cs == SH_ENV) { vis = visibleInf(y.pos, y.ng, yPrim, wP); }
#if RS_MODE_B
  else if ((cs == SH_N1 || cs == SH_B1) && endOcc) { vis = false; }   // N1 / B1-ana end occluded under frame dst.fs (refresh)
#else
  else if (cs == SH_N1 && endOcc) { vis = false; }      // M5 (§3.4): N1 end occluded under frame dst.fs (refresh)
#endif
  else { vis = visible(y.pos, y.ng, yPrim, xk.pos, xk.ng, xkPrim); }
  if (!vis) { return shift_finish(F, J, jNum, false); }
  return o;
}

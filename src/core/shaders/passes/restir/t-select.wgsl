// T3 rs_t_select (restir-temporal-api.md §3.6, §4.1; math.md#temporal [M5 addition], #confidence): contribution MIS
// with the exact E_{t−1} between the canonical X_c = res[w][q] (c_c = 1) and the shifted temporal candidate
// Y_p = T(X_p), X_p = res[h][q′] (c_p = min(cCap, c_prev)). Pixel-local and in place into res[w] (TD2, I1: res[h] is
// read-only; phase B is its last reader in the frame).
//   phase A (RsDispatch.round = 0): no q′ ⇒ res[w][q] bitwise untouched (TD13). w̃_c = lum(F_c)·W_c, w̃_p =
//     c_p·lum(F_t(Y_p))·W_p·J_p (0 unless the forward J word is a value); both 0 ⇒ empty record with c = 1 + c_p;
//     streaming RIS (canonical counter 0, temporal counter 1, RS_PASS_TEMPORAL). s = p: π_c = lum F_t(Y_p), π_p =
//     lum(F_p^st)/J_p (stored route), W_Y = π_p/(c_c π_c + c_p π_p)·Σw/π_c, write-back of Y_p (res_select_temporal),
//     final (robust: also queued on Q_i for the check; recompute: W in phase B). s = c: queued on Q_i (π_p(X_c) by T4).
//     Talbot: every valid-q′ pixel is queued; the selection happens in phase B.
//   phase B (RSD_PHASE_B, round = 1): the Q_i pixels after T4: s = c W_Y = π_c/(c_c π_c + c_p π_p(X_c))·Σw/π_c;
//     recompute s = p with π_p(Y_p) of T4; robust compare (RSC_T_ROBUST_MISMATCH); Talbot m_c / m_p and selection.
// Write-back of Y_p (math.md#jacobian [M5 addition]): all ten planes of res[h][q′], then F ← F_t(Y_p), jDen ← (J_p/J_P)·jDen
// (J_rc only), NEE entry words (end.x; rc.x when k = d) and endpointId renumbered to frame t, deep rcRad and the (b)/(c)
// cache (rcRad, aux) from the frame-t refresh (TF_REFRESH), W·wScale, c. Scene-free (restir-temporal-api.md A-2).
// OWNER T-B. G2: 0 resIn = res[h] ro · 1 resOut = res[w] rw · 2 arena rw · 3 rsVbuf (§4.2). No scene group.
#include "restir/tframe.wgsl"
#include "restir/tmis.wgsl"
#include "restir/reservoir.wgsl"
#include "debug/restir-views.wgsl"

const TS_ROBUST_IDMIS: u32 = 32768u;                     // = restir/tshift.wgsl (T4 sets it; Changelog B-3)
/// T3-2 round-trip harness only (restir-temporal.gpu.test.ts substitutes `true` through the kernel's
/// instrumentation.extraSources; Changelog B-7): every defined forward shift is selected (s = p), so robust mode runs
/// T⁻¹(T(X_p)) on every pixel with a valid q′. Never true in a production build.
const TSEL_TRACE_FORCE_P: bool = false;

fn tsel_fwdF(q: u32) -> vec3f { return vec3f(ts_loadf(q, TSW_FWDF), ts_loadf(q, TSW_FWDF + 1u), ts_loadf(q, TSW_FWDF + 2u)); }

/// Write-back of the temporal candidate Y_p into res[w][q] (see the header).
fn res_select_temporal(q: u32, qP: u32, W: f32, c: f32) {
  for (var pl = 0u; pl < RS_RES_PLANES; pl++) { resout_set(q, pl, resin_plane(qP, pl)); }
  let Jp = bitcast<f32>(ts_load(q, TSW_FWDJ));
  let jP = ts_loadf(q, TSW_JP);
  resout_set(q, RP_WF, vec4u(bitcast<u32>(W * rsParams.wScale), bitcast<vec3u>(tsel_fwdF(q))));
  let p1 = resin_plane(qP, RP_SEED);
  resout_set(q, RP_SEED, vec4u(p1.xyz, bitcast<u32>(c)));
  var p2 = resin_plane(qP, RP_RC);
  p2.w = bitcast<u32>((Jp / jP) * bitcast<f32>(p2.w));   // jDen ← J_rc·jDen_src (J_P excluded, TD8)
  if (!rs_tf(TF_REFRESH)) { resout_set(q, RP_RC, p2); return; }
  let f = p1.z;
  let k = rf_k(f);
  let d = rf_d(f);
  let r = sfx_load(SFX_FWD, qP);
  if (rf_tech(f) == RS_TECH_NEE) {
    let e = r.entryTo & RC_ENTRY_MASK;
    var p5 = resin_plane(qP, RP_END);
    p5.x = RC_TAG_NEE | e;
    resout_set(q, RP_END, p5);
    if (k == d) { p2.x = RC_TAG_NEE | e; }
    // endpointId (D5) = nee_endpoint_id_s(entry, cur slot): analytic lights are renumbered to the frame-t alias entry;
    // emissive-triangle primIds and RS_ENV_ID are frame-independent and stay as copied.
    if (e < lightsParams.cur.nAnalytic) {
      var p9 = resin_plane(qP, RP_DIAG);
      p9.w = e;
      resout_set(q, RP_DIAG, p9);
    }
  }
#if RS_MODE_B
  if (rf_tech(f) == RS_TECH_BSDF_ANALYTIC) {             // M6 MD8: crossing entries and endpointId renumbered to frame t
    let e = r.entryTo & RC_ENTRY_MASK;
    var p5 = resin_plane(qP, RP_END);
    p5.x = RC_TAG_CROSS | e;
    resout_set(q, RP_END, p5);
    if (k == d) { p2.x = RC_TAG_CROSS | e; }
    var p9 = resin_plane(qP, RP_DIAG);
    p9.w = e;
    resout_set(q, RP_DIAG, p9);
  }
#endif
  resout_set(q, RP_RC, p2);
  let deep = k != 0u && k + 2u <= d;
  let bc = k != 0u && k + 1u == d;
  if ((deep && (r.status & SXS_DEEP) != 0u) || (bc && (r.status & (SXS_N1 | SXS_B1)) != 0u)) {   // N3: stale values from the refresh (C-5)
    let p4 = resin_plane(qP, RP_RAD);
    resout_set(q, RP_RAD, vec4u(bitcast<vec3u>(r.rad), p4.w));
  }
  if (bc && (r.status & (SXS_N1 | SXS_B1)) != 0u && !rs_tplant(TP_U8_STALE_AUX)) {
    var p3 = resin_plane(qP, RP_WI);
    p3.w = bitcast<u32>(r.aux);
    resout_set(q, RP_WI, p3);
  }
}

/// W (× wScale) and c of res[w][q] (planes P0.x and P1.w only).
fn tsel_write_wc(q: u32, W: f32, c: f32) {
  var p0 = resout_plane(q, RP_WF);
  p0.x = bitcast<u32>(W * rsParams.wScale);
  resout_set(q, RP_WF, p0);
  var p1 = resout_plane(q, RP_SEED);
  p1.w = bitcast<u32>(c);
  resout_set(q, RP_SEED, p1);
}

/// A resampling weight made safe: non-finite or negative ⇒ 0 (counted, must stay 0). Bit tests (Metal Q: folded NaN).
fn tsel_w(w: f32) -> f32 {
  if (is_finite(w) && (bitcast<u32>(w) & 0x80000000u) == 0u) { return w; }
  if (bitcast<u32>(w) == 0x80000000u) { return 0.0; }    // −0
  rs_count(RSC_T_NONFINITE, 1u);
  return 0.0;
}

fn tsel_empty(p: RsPix, c: f32) {
  res_write_empty_c(p.ai, rs_frame_key(p.member, rs_t(), p.localIdx), c);
  rs_count(RSC_T_EMPTY_OUT, 1u);
}

/// c_p in the MIS denominators (plant TP_CP_PLUS1: c_p + 1, detected by rung 3.4).
fn tsel_cpw(cP: f32) -> f32 { return select(cP, cP + 1.0, rs_tplant(TP_CP_PLUS1)); }

fn tsel_phase_a(p: RsPix, flags0: u32) {
  let q = p.ai;
  var flags = flags0;
  let qP = ts_load(q, TSW_QPRIME);
  let cP = ts_loadf(q, TSW_CP);
  let pc0 = resout_plane(q, RP_WF);
  let Wc = rp_W(pc0);
  let pic = luminance(rp_F(pc0));
  var fwdJ = ts_load(q, TSW_FWDJ);
  if (fwdJ == JW_PENDING) { rs_count(RSC_T_PENDING_LEFT, 1u); fwdJ = JW_FAILED; ts_store(q, TSW_FWDJ, JW_FAILED); }
  let Jp = bitcast<f32>(fwdJ);
  let fwdOk = jw_valid(fwdJ);
  let pp0 = resin_plane(qP, RP_WF);
  let wc = tsel_w(pic * Wc);
  var wp = 0.0;
  if (fwdOk) { wp = tsel_w(cP * luminance(tsel_fwdF(q)) * rp_W(pp0) * Jp); }
  ts_storef(q, TSW_WC, wc);
  ts_storef(q, TSW_WP, wp);
  if (rs_tmode(TM_TALBOT)) {
    queue_append(RS_Q_INV, queue_item_word(q, 0u));
    rs_count(RSC_T_INV_QUEUED, 1u);
    ts_store(q, TSW_FLAGS, flags | TS_INV_QUEUED);
    return;
  }
  if (!rs_pos_finite(wc + wp)) {
    tsel_empty(p, 1.0 + cP);
    ts_store(q, TSW_FLAGS, flags | TS_EMPTY_OUT | TS_FINAL);
    rsdbg_temporal(p.px, q, 0u);
    return;
  }
  let key = rs_frame_key(p.member, rs_t(), p.localIdx);
  var wSum = 0.0;
  var selP = false;
  ris_update(&wSum, wc, rs_rand(key, RS_PASS_TEMPORAL, 0u));
  if (ris_update(&wSum, wp, rs_rand(key, RS_PASS_TEMPORAL, 1u))) { selP = true; }
  if (TSEL_TRACE_FORCE_P) { selP = wp > 0.0; }
  if (selP) {
    let piC = luminance(tsel_fwdF(q));                   // π_c(Y_p) = p̂_t(Y_p)
    let piP = luminance(rp_F(pp0)) / Jp;                 // stored route π_p(Y_p) = lum F_p^st / J_p
    ts_storef(q, TSW_PISTORED, piP);
    rs_count(RSC_T_SEL_P, 1u);
    if (rs_tmode(TM_PP_RECOMPUTE)) {
      res_select_temporal(q, qP, 0.0, 1.0 + cP);         // W written in phase B from π_p(Y_p) of T4
      queue_append(RS_Q_INV, queue_item_word(q, 0u));
      rs_count(RSC_T_INV_QUEUED, 1u);
      ts_store(q, TSW_FLAGS, flags | TS_SEL_P | TS_INV_QUEUED);
      return;
    }
    let W = tmis_contrib_W(piP, piC, piP, 1.0, tsel_cpw(cP), wSum);
    flags |= TS_SEL_P | TS_FINAL;
    if (is_finite(W)) {
      res_select_temporal(q, qP, W, 1.0 + cP);
    } else {
      rs_count(RSC_T_NONFINITE, 1u);
      tsel_empty(p, 1.0 + cP);
      flags |= TS_EMPTY_OUT;
    }
    if (rs_tmode(TM_ROBUST)) {
      queue_append(RS_Q_INV, queue_item_word(q, 0u));
      rs_count(RSC_T_INV_QUEUED, 1u);
      flags |= TS_INV_QUEUED | TS_ROBUST;
    }
    ts_store(q, TSW_FLAGS, flags);
    if ((flags & TS_INV_QUEUED) == 0u) { rsdbg_temporal(p.px, q, 0u); }
    return;
  }
  queue_append(RS_Q_INV, queue_item_word(q, 0u));
  rs_count(RSC_T_INV_QUEUED, 1u);
  ts_store(q, TSW_FLAGS, flags | TS_SEL_C | TS_INV_QUEUED);
}

fn tsel_phase_b(p: RsPix, flags0: u32) {
  let q = p.ai;
  var flags = flags0 | TS_FINAL;
  let cP = ts_loadf(q, TSW_CP);
  let cOut = 1.0 + cP;
  let piR = ts_loadf(q, TSW_PIRECOMP);
  if ((flags & TS_INV_DONE) == 0u) { rs_count(RSC_T_PENDING_LEFT, 1u); }
  if (rs_tmode(TM_TALBOT)) {
    // Talbot-exact: m_c(X_c) with π_p(X_c) of T4, m_p(Y_p) with the stored route; the phase-A counters 0 / 1.
    let qP = ts_load(q, TSW_QPRIME);
    let pc0 = resout_plane(q, RP_WF);
    let lumFc = luminance(rp_F(pc0));
    let fwdJ = ts_load(q, TSW_FWDJ);
    let Jp = bitcast<f32>(fwdJ);
    let pp0 = resin_plane(qP, RP_WF);
    let lumFY = luminance(tsel_fwdF(q));
    let wc = tsel_w(tmis_talbot_mc(1.0, lumFc, cP, piR) * lumFc * rp_W(pc0));
    var wp = 0.0;
    if (jw_valid(fwdJ)) {
      let piPY = luminance(rp_F(pp0)) / Jp;
      ts_storef(q, TSW_PISTORED, piPY);
      wp = tsel_w(tmis_talbot_mp(1.0, lumFY, cP, piPY) * lumFY * rp_W(pp0) * Jp);
    }
    ts_storef(q, TSW_WC, wc);
    ts_storef(q, TSW_WP, wp);
    let key = rs_frame_key(p.member, rs_t(), p.localIdx);
    var wSum = 0.0;
    var selP = false;
    ris_update(&wSum, wc, rs_rand(key, RS_PASS_TEMPORAL, 0u));
    if (ris_update(&wSum, wp, rs_rand(key, RS_PASS_TEMPORAL, 1u))) { selP = true; }
    if (!rs_pos_finite(wSum)) {
      tsel_empty(p, cOut);
      flags |= TS_EMPTY_OUT;
    } else if (selP) {
      let W = wSum / lumFY;
      flags |= TS_SEL_P;
      rs_count(RSC_T_SEL_P, 1u);
      if (is_finite(W)) { res_select_temporal(q, qP, W, cOut); } else { rs_count(RSC_T_NONFINITE, 1u); tsel_empty(p, cOut); flags |= TS_EMPTY_OUT; }
    } else {
      let W = wSum / lumFc;
      flags |= TS_SEL_C;
      if (is_finite(W)) { tsel_write_wc(q, W, cOut); } else { rs_count(RSC_T_NONFINITE, 1u); tsel_empty(p, cOut); flags |= TS_EMPTY_OUT; }
    }
  } else if ((flags & TS_SEL_C) != 0u) {
    // contribution MIS, s = c: π_c = p̂_t(X_c), π_p(X_c) by T4
    let piC = luminance(rp_F(resout_plane(q, RP_WF)));
    let wSum = ts_loadf(q, TSW_WC) + ts_loadf(q, TSW_WP);
    let W = tmis_contrib_W(piC, piC, piR, 1.0, tsel_cpw(cP), wSum);
    if (is_finite(W)) { tsel_write_wc(q, W, cOut); } else { rs_count(RSC_T_NONFINITE, 1u); tsel_empty(p, cOut); flags |= TS_EMPTY_OUT; }
  } else if ((flags & TS_ROBUST) != 0u) {
    // robust (s = p): the stored route was used in phase A; compare it with π_p(Y_p) recomputed by E_{t−1}
    let a = ts_loadf(q, TSW_PISTORED);
    let ok = rs_slot_code_sc(ts_load(q, TSW_INVCODE)) == SC_OK;
    if (!ok || abs(a - piR) > 1e-3 * max(a, piR) || (flags & TS_ROBUST_IDMIS) != 0u) { rs_count(RSC_T_ROBUST_MISMATCH, 1u); }
  } else {
    // recompute (s = p): π_p(Y_p) recomputed by E_{t−1} instead of the stored route
    let piC = luminance(tsel_fwdF(q));
    let wSum = ts_loadf(q, TSW_WC) + ts_loadf(q, TSW_WP);
    let W = tmis_contrib_W(piR, piC, piR, 1.0, tsel_cpw(cP), wSum);
    if (is_finite(W)) { tsel_write_wc(q, W, cOut); } else { rs_count(RSC_T_NONFINITE, 1u); tsel_empty(p, cOut); flags |= TS_EMPTY_OUT; }
  }
  ts_store(q, TSW_FLAGS, flags);
  rsdbg_temporal(p.px, q, 1u);
}

@compute @workgroup_size(8, 8, 1)
fn rs_t_select(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
  if (rs_vbuf(p.px).x == 0xFFFFFFFFu) { return; }
  let flags = ts_load(p.ai, TSW_FLAGS);
  if ((rsDispatch.flags & RSD_PHASE_B) == 0u) {
    if ((flags & TS_QVALID) == 0u) { rsdbg_temporal(p.px, p.ai, 0u); return; }   // TD13: res[w][q] untouched
    tsel_phase_a(p, flags);
  } else if ((flags & TS_INV_QUEUED) != 0u) {
#if RS_TSEL_FOLD
    // RSD_TFOLD: T4 finished the pixel (phase B folded into T4); this dispatch only records the debug views / probe
    if ((rsDispatch.flags & RSD_TFOLD) != 0u) { rsdbg_temporal(p.px, p.ai, 1u); return; }
#endif
    tsel_phase_b(p, flags);
  }
}

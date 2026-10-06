// ReSTIR debug-view and probe hooks (restir-api.md §2.11; PLAN §6 M4 rows): view ids 400–499, probe tags 64–95.
// OWNER WP-D. Every hook is a no-op unless its view is active (dbg.mode) or the pixel is the probe pixel, so the
// debug-off path does no work beyond a uniform compare and never touches a production output (bit-identical
// validation). With DEBUG_NO_BINDINGS (test pipelines without G3) every hook compiles to an empty body.
//
// Who writes what (restir-api.md Changelog D1–D4):
//   400–410 reservoir views   rsdbg_reservoir (rs_initial: tap INITIAL; M5 T3: tap TEMPORAL; final-round resample: tap
//                             SPATIAL). The FINAL tap accepts every tap: the later stage of the frame overwrites (last
//                             writer wins).
//   480–497 temporal views    M5, below (rsdbg_temporal / rsdbg_tpick) and rs_debug_views (497).
//   420–446 shift views       passes/restir/debug.wgsl (rs_debug_views, after the spatial stage) from the arena, so slots
//                             written only by rs_pair_accept (NOT_ACCEPTED, EMPTY_SRC) are covered and the per-pixel
//                             masks need no cross-thread read-modify-write.
//   460–470 MIS views         rsdbg_mis (spatial resample; the last round overwrites earlier rounds).
// Probe records (value = vec4f; "bits" = bitcast<f32>(u32)):
//   64 reservoir plane (the plane's 4 words as bits), preceded by 65 header (bits(pass), bits(round), bits(ai), bits(tap));
//      header + 10 planes are reserved as one block of 11 consecutive records.
//   66 candidate  (bits(d | tech<<4 | k<<8 | selected<<12), w, lumF, bits(counter))
//   67 vertex     (pos.xyz in the render frame (recentred), bits(path<<8 | b<<4 | lobeCode))
//   68 slot event (bits(code), J, bits(s), bits(replayed))                      rsdbg_slot (live, shift/replay)
//   69 MIS record (bits(s), m, w, 0); s = 0xFF: canonical (m = m_c, w = w_c)   rsdbg_mis
//   70 MIS canonical (bits(k | sel<<8), m_c, Σm−1, lumRel)                       rsdbg_mis
//   71 slot final, outgoing p→partner (bits(code), bits(Jword), bits(s | queued<<8 | partnerValid<<9), bits(partner ai))
//   72 slot final, incoming partner→p (bits(code), bits(Jword), bits(s), lum(FJ))   (71/72: rs_debug_views)
#include "debug/debug-common.wgsl"
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"

// View ids (render/restir/layout.ts RS_VIEW; render/restir/debug.ts RESTIR_VIEWS).
const RSV_C: u32 = 400u;  const RSV_W: u32 = 401u;  const RSV_PHAT: u32 = 402u;  const RSV_FW: u32 = 403u;
const RSV_D: u32 = 404u;  const RSV_K: u32 = 405u;  const RSV_TECH: u32 = 406u;  const RSV_EP: u32 = 407u;
const RSV_LOBES: u32 = 408u;  const RSV_CLASS: u32 = 409u;  const RSV_KMARGIN: u32 = 410u;
const RSV_SHIFT_CODE: u32 = 420u;  const RSV_SHIFT_LOGJ: u32 = 426u;  const RSV_SHIFT_TERM: u32 = 432u;
const RSV_REPLAY_MASK: u32 = 438u;  const RSV_ACCEPT_MASK: u32 = 439u;  const RSV_SHIFT_MARGIN: u32 = 440u;
const RSV_THR: u32 = 446u;
const RSV_MIS_MC: u32 = 460u;  const RSV_MIS_SUMM: u32 = 461u;  const RSV_MIS_LUML: u32 = 462u;  const RSV_MIS_MJ: u32 = 463u;
const RSV_MIS_K: u32 = 469u;  const RSV_MIS_SEL: u32 = 470u;

// Probe tags.
const RSP_PLANE: u32 = 64u;  const RSP_HEADER: u32 = 65u;  const RSP_CAND: u32 = 66u;  const RSP_VERTEX: u32 = 67u;
const RSP_SLOT: u32 = 68u;  const RSP_MIS: u32 = 69u;  const RSP_MIS_CANON: u32 = 70u;
const RSP_SLOT_OUT: u32 = 71u;  const RSP_SLOT_IN: u32 = 72u;
// M5 (T-D; restir-temporal-api.md §2.11, Changelog D-1): temporal views 480–497 and probe tags 73–79 (formats below).
const RSV_T_QVALID: u32 = 480u;  const RSV_T_MOTION: u32 = 481u;  const RSV_T_RF_FWD: u32 = 482u;  const RSV_T_RF_INV: u32 = 483u;
const RSV_T_FWDCODE: u32 = 484u;  const RSV_T_INVCODE: u32 = 485u;  const RSV_T_LOGJ: u32 = 486u;  const RSV_T_LOGJP: u32 = 487u;
const RSV_T_PIC: u32 = 488u;  const RSV_T_PIP: u32 = 489u;  const RSV_T_CPREV: u32 = 490u;  const RSV_T_COUT: u32 = 491u;
const RSV_T_SEL: u32 = 492u;  const RSV_T_PHATREL: u32 = 493u;  const RSV_T_ROBUST: u32 = 494u;  const RSV_T_WP: u32 = 495u;
const RSV_T_LCHG: u32 = 496u;
const RSP_T_HEADER: u32 = 73u;  const RSP_T_FWD: u32 = 74u;  const RSP_T_INV: u32 = 75u;  const RSP_T_SEL: u32 = 76u;
const RSP_T_REFRESH: u32 = 77u;  const RSP_T_PICK: u32 = 78u;  const RSP_T_ANCHOR: u32 = 79u;
const RSV_S_BOOST: u32 = 497u;

fn rsdbg_bits(u: u32) -> f32 { return bitcast<f32>(u); }
fn rsdbg_lum(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

/// Path class of a record (math.md#reservoir-fields; layout.ts pathClass): 0 L, 1 N1, 2 B1, 3 E, 4 D-NEE, 5 D-BSDF,
/// 6 R (k = ∅); DBG_CODE_NONE when empty or background.
fn rsdbg_class(f: u32) -> u32 {
  let d = rf_d(f);
  let k = rf_k(f);
  if (d == 0u) { return DBG_CODE_NONE; }
  if (k == 0u) { return 6u; }
  let nee = rf_tech(f) == RS_TECH_NEE;
  if (k == d) { return select(3u, 0u, nee); }
  if (k + 1u == d) { return select(2u, 1u, nee); }
  return select(5u, 4u, nee);
}

#if !DEBUG_NO_BINDINGS
/// Reserve n consecutive probe records (one atomicAdd, so a block is never interleaved with other records); returns the
/// first index or 0xFFFFFFFF (the n records are counted as overflow).
fn rsdbg_probe_block(n: u32) -> u32 {
  let i = atomicAdd(&dbgBuf.counters[DBGC_PROBE_COUNT], n);
  if (i + n > PROBE_CAPACITY) { atomicAdd(&dbgBuf.counters[DBGC_PROBE_OVERFLOW], n); return 0xFFFFFFFFu; }
  return i;
}
fn rsdbg_probe_put(i: u32, px: vec2u, tag: u32, v: vec4f) { dbgBuf.probe[i] = ProbeRecord(px, tag, i, v); }
fn rsdbg_view_in(lo: u32, hi: u32) -> bool { return dbg.mode >= lo && dbg.mode <= hi; }
#endif

/// After a reservoir is final (rs_initial after the final chunk: tap DBG_TAP_INITIAL; M5: T3 rs_t_select after the
/// temporal output: DBG_TAP_TEMPORAL; spatial resample of the final round: DBG_TAP_SPATIAL). Reads the record back
/// from resOut (every pass that calls it binds resOut).
fn rsdbg_reservoir(px: vec2u, ai: u32, tap: u32) {
#if !DEBUG_NO_BINDINGS && RS_RES_OUT_BINDING
  let probe = debug_is_probe(px);
  let view = rsdbg_view_in(RSV_C, RSV_KMARGIN) && (dbg.tap == tap || dbg.tap == DBG_TAP_FINAL);
  if (!view && !probe) { return; }
  if (probe) {
    let i = rsdbg_probe_block(1u + RS_RES_PLANES);
    if (i != 0xFFFFFFFFu) {
      let passId = select(select(RS_PASS_INITIAL, RS_PASS_TEMPORAL, tap == DBG_TAP_TEMPORAL), RS_PASS_SPATIAL + rsDispatch.round, tap == DBG_TAP_SPATIAL);
      rsdbg_probe_put(i, px, RSP_HEADER, bitcast<vec4f>(vec4u(passId, rsDispatch.round, ai, tap)));
      for (var pl = 0u; pl < RS_RES_PLANES; pl++) {
        rsdbg_probe_put(i + 1u + pl, px, RSP_PLANE, bitcast<vec4f>(resout_plane(ai, pl)));
      }
    }
  }
  if (!view) { return; }
  let p0 = resout_plane(ai, RP_WF);
  let f = rp_flags(resout_plane(ai, RP_SEED));
  let none = (f & RF_BG) != 0u || res_empty(f);
  let W = rp_W(p0);
  let F = rp_F(p0);
  switch (dbg.mode) {
    case 400u: { debug_write1(px, RSV_C, rp_c(resout_plane(ai, RP_SEED))); }
    case 401u: { debug_write1(px, RSV_W, W); }
    case 402u: { debug_write1(px, RSV_PHAT, rsdbg_lum(F)); }
    case 403u: { debug_write3(px, RSV_FW, select(F * W, vec3f(0.0), W == 0.0)); }
    case 404u: { debug_write_code(px, RSV_D, select(rf_d(f), DBG_CODE_NONE, (f & RF_BG) != 0u)); }
    case 405u: { debug_write_code(px, RSV_K, select(rf_k(f), DBG_CODE_NONE, none)); }
    case 406u: { debug_write_code(px, RSV_TECH, select(rf_tech(f), DBG_CODE_NONE, none)); }
    case 407u: { debug_write_code(px, RSV_EP, select(rf_ep(f), DBG_CODE_NONE, none)); }
    case 408u: {
      let dm = select(0u, 1u, (f & RF_DKM1) != 0u) | select(0u, 2u, (f & RF_DK) != 0u);
      debug_write_code(px, RSV_LOBES, select(rf_lkm1(f) * 16u + rf_lk(f) + dm * 256u, DBG_CODE_NONE, none));
    }
    case 409u: { debug_write_code(px, RSV_CLASS, rsdbg_class(f)); }
    case 410u: { debug_write1(px, RSV_KMARGIN, bitcast<f32>(resout_plane(ai, RP_DIAG).z)); }
    default: { }
  }
#endif
}

/// One streamed candidate of the initial pass (probe pixel only).
fn rsdbg_candidate(px: vec2u, d: u32, tech: u32, k: u32, w: f32, lumF: f32, counter: u32, selected: bool) {
#if !DEBUG_NO_BINDINGS
  if (!debug_is_probe(px)) { return; }
  let code = (d & 0xFu) | ((tech & 0xFu) << 4u) | ((k & 0xFu) << 8u) | (select(0u, 1u, selected) << 12u);
  probe_record(px, RSP_CAND, vec4f(rsdbg_bits(code), w, lumF, rsdbg_bits(counter)));
#endif
}

/// One vertex of a path for the inspector's 3D overlay (probe pixel only). path 0 = base tree (b = 1 … d; b = 0 may be
/// the camera), 1+s = the probe pixel's path shifted into its partner of slot s, 8+s = the partner's path shifted into
/// the probe pixel (the caller passes px = the probe pixel in both cases). lobeCode = lobe | delta<<3 (0xF none).
fn rsdbg_vertex(px: vec2u, path: u32, b: u32, pos: vec3f, lobeCode: u32) {
#if !DEBUG_NO_BINDINGS
  if (!debug_is_probe(px)) { return; }
  probe_record(px, RSP_VERTEX, vec4f(pos, rsdbg_bits(((path & 0xFFFFFFu) << 8u) | ((b & 0xFu) << 4u) | (lobeCode & 0xFu))));
#endif
}

/// A slot was shifted (spatial shift / replay; px = the source pixel whose path was shifted). Probe record only: the
/// shift views are written from the arena by rs_debug_views (Changelog D1).
fn rsdbg_slot(px: vec2u, s: u32, code: u32, J: f32, replayed: bool) {
#if !DEBUG_NO_BINDINGS
  if (!debug_is_probe(px)) { return; }
  probe_record(px, RSP_SLOT, vec4f(rsdbg_bits(code), J, rsdbg_bits(s), rsdbg_bits(select(0u, 1u, replayed))));
#endif
}

/// Accepted-slot mask of a pixel (spatial shift). Kept for signature stability: view 439 is written by rs_debug_views
/// from the J words (Changelog D1), which is the same mask.
fn rsdbg_accept(px: vec2u, mask: u32) { }

/// Spatial resample records. Called once with s = 0xFF for the canonical sample (mj = m_c, wj = w_c) and once per
/// partner slot s ∈ S_c (mj = m_j, wj = w_j). k = |S_c|, sumM = m_c(X_c) + Σ_j m_j(X_c) − 1 at y = X_c,
/// lumRel = (lum(L_rgb) − Σw)/max(Σw, 1e-30), sel = 0 canonical / 1+s partner.
fn rsdbg_mis(px: vec2u, k: u32, mc: f32, sumM: f32, lumRel: f32, s: u32, mj: f32, wj: f32, sel: u32) {
#if !DEBUG_NO_BINDINGS
  if (debug_is_probe(px)) {
    probe_record(px, RSP_MIS, vec4f(rsdbg_bits(s), mj, wj, 0.0));
    if (s == 0xFFu) { probe_record(px, RSP_MIS_CANON, vec4f(rsdbg_bits((k & 0xFFu) | ((sel & 0xFFu) << 8u)), mc, sumM, lumRel)); }
  }
  if (!rsdbg_view_in(RSV_MIS_MC, RSV_MIS_SEL)) { return; }
  if (s == 0xFFu) {
    debug_write1(px, RSV_MIS_MC, mc);
    debug_write1(px, RSV_MIS_SUMM, sumM);
    debug_write1(px, RSV_MIS_LUML, lumRel);
    debug_write_code(px, RSV_MIS_K, k);
    debug_write_code(px, RSV_MIS_SEL, sel);
  } else if (s < RS_MAX_SLOTS) {
    debug_write1(px, RSV_MIS_MJ + s, mj);
  }
#endif
}

// ---- M5 temporal views and probes (restir-temporal-api.md §2.11; OWNER T-D; Changelog D-1). Compiled only into the
// temporal / refresh passes (define RS_TEMPORAL), which include restir/tframe.wgsl (SfxRec, tState accessors).
//   rsdbg_temporal (T3 rs_t_select, after the pixel's temporal output is final: phase A for no-q′ / s = p / empty
//                   pixels, phase B for the Q_i pixels): reservoir views 400–410 at tap DBG_TAP_TEMPORAL, temporal views
//                   480 and 482–496 from tState / sfxOut / res[h] / res[w], probe tags 73–77 and the anchor ids (79)
//   rsdbg_tpick    (T1): view 481 (s′ − q, pixel centres at integers, Changelog A-7) and probe tag 78
//   rsdbg_refresh  (refresh passes): probe tag 77 of the record at the probe pixel itself (status bit 17 set)
// View 497 (boost mask) and the anchor → vertex conversion (tag 79 → tag 67, paths 23 / 31) are written by
// rs_debug_views (passes/restir/debug.wgsl), which has the scene group.
// Probe formats ("bits" = bitcast<f32>(u32)):
//   73 header   (bits(q ai), bits(q′ ai or NONE), c_prev, bits(tState flags))
//   74 forward  (bits(fwdCode), J_p (0 unless VALID), J_P, lum F_t(T(X_p)))
//   75 inverse  (bits(invCode), J_inv (0 unless VALID), lum F_{t−1}(T⁻¹ y), π_p recomputed)
//   76 select   (w̃_c, w̃_p, π_c, bits(sel | phase << 8))            sel as view 492
//   77 refresh  (bits(status | dir << 16 | fromPass << 17), lum rad, aux, bits(entryTo)); dir 0 fwd (the history record
//               X_p at q′), 1 inv (the current record at q); fromPass = recorded by the refresh pass for the record at
//               the probe pixel itself (rsdbg_refresh) instead of by T3 for the probe pixel's q′ / q
//   78 pick     (s′.x, s′.y, bits(tap), bits(valid))
//   79 anchor   (bits(role | b << 8), bits(prim or RC tag word), bits(u), bits(v)): role 0 X_p's x_k, 1 X_p's surface
//               endpoint, 2 the inverse source's x_k, 3 its surface endpoint (internal: rs_debug_views turns them into
//               tag-67 vertices of path 23 = forward T(X_p) into q and path 31 = inverse into q′ at t−1)
#if RS_TEMPORAL
#include "restir/tframe.wgsl"


/// View 480 code of a tState flag word: 0 background, 1 centre tap, 2 ring tap, 3 no q′ (disoccluded), 4 no history.
fn rsdbg_qvalid_code(f: u32) -> u32 {
  if ((f & TS_BG) != 0u) { return 0u; }
  if ((f & TS_QVALID) != 0u) { return select(1u, 2u, (f & TS_PICK_RING) != 0u); }
  return select(3u, 4u, (f & TS_NO_HIST) != 0u);
}
/// View 482/483 class of a refresh record: 0 not refreshed this frame, 1 analytic (no ray), 2 ray traced,
/// 3 undefined, 4 E2 zeroed, 5 zero.
fn rsdbg_refresh_class(r: SfxRec) -> u32 {
  if (!rs_tf(TF_REFRESH) || r.gen != rsTemporal.frameGen || (r.status & SXS_DONE) == 0u) { return 0u; }
  if ((r.status & SXS_E2) != 0u) { return 4u; }
  if ((r.status & SXS_UNDEF) != 0u) { return 3u; }
  if ((r.status & SXS_ZERO) != 0u) { return 5u; }
  return select(1u, 2u, (r.status & SXS_RAY) != 0u);
}
/// View 492 / tag 76 selection code: 0 canonical kept (no q′), 1 canonical selected, 2 temporal selected, 3 empty.
fn rsdbg_sel_code(f: u32) -> u32 {
  if ((f & TS_QVALID) == 0u) { return 0u; }
  if ((f & TS_EMPTY_OUT) != 0u) { return 3u; }
  return select(1u, 2u, (f & TS_SEL_P) != 0u);
}
/// f32 value of a J word, 0 unless VALID.
fn rsdbg_jval(jw: u32) -> f32 { return select(0.0, bitcast<f32>(jw), jw_valid(jw)); }
fn rsdbg_rel(a: f32, b: f32) -> f32 {
  let m = max(abs(a), abs(b));
  return select(0.0, (a - b) / m, m > 0.0);
}
fn rsdbg_log2_pos(x: f32) -> f32 { return select(0.0, log2(x), x > 0.0 && x < 3.0e38); }

#if !DEBUG_NO_BINDINGS && RS_ARENA_BINDING
/// View 496: X_p's light changed between t−1 and t: bit0 moved, bit1 radiometric, bit2 undefined (removed / missing).
/// NEE ends use the forward refresh record of q′ (translated entry, SXS_UNDEF); BSDF env ends the env flags.
fn rsdbg_lights_changed(qP: u32, fp: u32) -> u32 {
  if (!rs_tf(TF_REFRESH) || res_empty(fp)) { return 0u; }
  let tech = rf_tech(fp);
  if (tech == RS_TECH_BSDF_ENV) {
    return select(0u, 1u, rs_tf(TF_ENV_MOVED)) | select(0u, 2u, rs_tf(TF_ENV_RADIO));
  }
  if (tech != RS_TECH_NEE) { return 0u; }
  let r = sfx_load(SFX_FWD, qP);
  if (r.gen != rsTemporal.frameGen || (r.status & SXS_UNDEF) != 0u || r.entryTo == LIGHT_NONE || r.entryTo == RC_NONE) { return 4u; }
  let b = lt_change_bits(r.entryTo);
  return select(0u, 1u, (b & LCB_MOVED) != 0u) | select(0u, 2u, (b & LCB_RADIO) != 0u);
}
fn rsdbg_refresh_probe(px: vec2u, dir: u32, fromPass: bool, r: SfxRec) {
  let st = (r.status & 0xFFFFu) | (dir << 16u) | select(0u, 1u << 17u, fromPass);
  probe_record(px, RSP_T_REFRESH, vec4f(rsdbg_bits(st), rsdbg_lum(r.rad), r.aux, rsdbg_bits(r.entryTo)));
}
#endif

/// After T3 phase A/B finalised the pixel (reads resOut + tState). Phase A skips the Q_i pixels (phase B records them).
fn rsdbg_temporal(px: vec2u, ai: u32, phase: u32) {
#if !DEBUG_NO_BINDINGS && RS_ARENA_BINDING && RS_RES_OUT_BINDING
  let probe = debug_is_probe(px);
  let tview = rsdbg_view_in(RSV_T_QVALID, RSV_T_LCHG) && dbg.mode != RSV_T_MOTION;
  let rview = rsdbg_view_in(RSV_C, RSV_KMARGIN);
  if (!probe && !tview && !rview) { return; }
  let f = ts_load(ai, TSW_FLAGS);
  if (phase == 0u && (f & TS_INV_QUEUED) != 0u) { return; }
  rsdbg_reservoir(px, ai, DBG_TAP_TEMPORAL);
  if (!probe && !tview) { return; }
  let qvalid = (f & TS_QVALID) != 0u;
  let invq = (f & TS_INV_QUEUED) != 0u;
  let qP = ts_load(ai, TSW_QPRIME);
  let fwdJ = ts_load(ai, TSW_FWDJ);
  let fwdOk = qvalid && jw_valid(fwdJ);
  let fwdF = vec3f(ts_loadf(ai, TSW_FWDF), ts_loadf(ai, TSW_FWDF + 1u), ts_loadf(ai, TSW_FWDF + 2u));
  let jP = ts_loadf(ai, TSW_JP);
  let wc = ts_loadf(ai, TSW_WC);
  let wp = ts_loadf(ai, TSW_WP);
  let piStored = ts_loadf(ai, TSW_PISTORED);
  let piRecomp = ts_loadf(ai, TSW_PIRECOMP);
  let o0 = resout_plane(ai, RP_WF);
  let piC = rsdbg_lum(rp_F(o0));
  let piP = select(select(0.0, piRecomp, invq), piStored, (f & TS_SEL_P) != 0u);
  let sel = rsdbg_sel_code(f);
#if RS_RES_IN_BINDING
  let fp = select(0u, rp_flags(resin_plane(min(qP, rs_atlas_pixels() - 1u), RP_SEED)), qvalid);
  let lumSt = select(0.0, rsdbg_lum(rp_F(resin_plane(min(qP, rs_atlas_pixels() - 1u), RP_WF))), qvalid);
#else
  let fp = 0u;
  let lumSt = 0.0;
#endif
  if (tview) {
    switch (dbg.mode) {
      case 480u: { debug_write_code(px, RSV_T_QVALID, rsdbg_qvalid_code(f)); }
      case 482u: { debug_write_code(px, RSV_T_RF_FWD, select(0u, rsdbg_refresh_class(sfx_load(SFX_FWD, min(qP, rs_atlas_pixels() - 1u))), qvalid)); }
      case 483u: { debug_write_code(px, RSV_T_RF_INV, select(0u, rsdbg_refresh_class(sfx_load(SFX_INV, ai)), invq)); }
      case 484u: { debug_write_code(px, RSV_T_FWDCODE, select(DBG_CODE_NONE, rs_slot_code_sc(ts_load(ai, TSW_FWDCODE)), qvalid)); }
      case 485u: { debug_write_code(px, RSV_T_INVCODE, select(DBG_CODE_NONE, rs_slot_code_sc(ts_load(ai, TSW_INVCODE)), invq)); }
      case 486u: { debug_write1(px, RSV_T_LOGJ, select(0.0, rsdbg_log2_pos(rsdbg_jval(fwdJ)), fwdOk)); }
      case 487u: { debug_write1(px, RSV_T_LOGJP, select(0.0, rsdbg_log2_pos(jP), fwdOk)); }
      case 488u: { debug_write1(px, RSV_T_PIC, piC); }
      case 489u: { debug_write1(px, RSV_T_PIP, piP); }
      case 490u: { debug_write1(px, RSV_T_CPREV, select(0.0, ts_loadf(ai, TSW_CPREV), qvalid)); }
      case 491u: { debug_write1(px, RSV_T_COUT, rp_c(resout_plane(ai, RP_SEED))); }
      case 492u: { debug_write_code(px, RSV_T_SEL, sel); }
      case 493u: { debug_write1(px, RSV_T_PHATREL, select(0.0, rsdbg_rel(rsdbg_lum(fwdF), lumSt), fwdOk)); }
      case 494u: { debug_write1(px, RSV_T_ROBUST, select(0.0, rsdbg_rel(piRecomp, piStored), (f & TS_ROBUST) != 0u)); }
      case 495u: { debug_write1(px, RSV_T_WP, select(0.0, wp / (wc + wp), wc + wp > 0.0)); }
      case 496u: { debug_write_code(px, RSV_T_LCHG, select(0u, rsdbg_lights_changed(min(qP, rs_atlas_pixels() - 1u), fp), qvalid)); }
      default: { }
    }
  }
  if (!probe) { return; }
  probe_record(px, RSP_T_HEADER, vec4f(rsdbg_bits(ai), rsdbg_bits(qP), ts_loadf(ai, TSW_CPREV), rsdbg_bits(f)));
  if (qvalid) {
    probe_record(px, RSP_T_FWD, vec4f(rsdbg_bits(ts_load(ai, TSW_FWDCODE)), rsdbg_jval(fwdJ), jP, rsdbg_lum(fwdF)));
    if (rs_tf(TF_REFRESH)) { rsdbg_refresh_probe(px, SFX_FWD, false, sfx_load(SFX_FWD, qP)); }
#if RS_RES_IN_BINDING
    if (!res_empty(fp)) {
      let rk = rf_k(fp);
      if (rk >= 2u) { probe_record(px, RSP_T_ANCHOR, bitcast<vec4f>(vec4u(0u | (rk << 8u), resin_plane(qP, RP_RC).xyz))); }
      if (rf_tech(fp) == RS_TECH_BSDF_TRI && rk != rf_d(fp)) {
        probe_record(px, RSP_T_ANCHOR, bitcast<vec4f>(vec4u(1u | (rf_d(fp) << 8u), resin_plane(qP, RP_END).xyz)));
      }
    }
#endif
  }
  if (invq) {
    let invF = vec3f(ts_loadf(ai, TSW_INVF), ts_loadf(ai, TSW_INVF + 1u), ts_loadf(ai, TSW_INVF + 2u));
    probe_record(px, RSP_T_INV, vec4f(rsdbg_bits(ts_load(ai, TSW_INVCODE)), rsdbg_jval(ts_load(ai, TSW_INVJ)), rsdbg_lum(invF), piRecomp));
    if (rs_tf(TF_REFRESH)) { rsdbg_refresh_probe(px, SFX_INV, false, sfx_load(SFX_INV, ai)); }
    let fc = rp_flags(resout_plane(ai, RP_SEED));
    if (!res_empty(fc)) {
      let ck = rf_k(fc);
      if (ck >= 2u) { probe_record(px, RSP_T_ANCHOR, bitcast<vec4f>(vec4u(2u | (ck << 8u), resout_plane(ai, RP_RC).xyz))); }
      if (rf_tech(fc) == RS_TECH_BSDF_TRI && ck != rf_d(fc)) {
        probe_record(px, RSP_T_ANCHOR, bitcast<vec4f>(vec4u(3u | (rf_d(fc) << 8u), resout_plane(ai, RP_END).xyz)));
      }
    }
  }
  probe_record(px, RSP_T_SEL, vec4f(wc, wp, piC, rsdbg_bits(sel | (phase << 8u))));
#endif
}

/// T1, after temporal_pixel: view 481 (s′ − q in pixels; 0 without history or without a projection) and probe tag 78.
fn rsdbg_tpick(px: vec2u, sp: vec2f, tap: u32, valid: bool) {
#if !DEBUG_NO_BINDINGS
  let probe = debug_is_probe(px);
  if (!probe && dbg.mode != RSV_T_MOTION) { return; }
  let projected = rs_tf(TF_HIST_VALID) && !(sp.x == -1.0 && sp.y == -1.0);
  let m = select(vec2f(0.0), sp - vec2f(rs_pix(px).local), projected);
  debug_write3(px, RSV_T_MOTION, vec3f(m, 0.0));
  if (probe) { probe_record(px, RSP_T_PICK, vec4f(sp, rsdbg_bits(tap), rsdbg_bits(select(0u, 1u, valid)))); }
#endif
}

/// Refresh passes, probe only: the refresh record of the record stored at the probe pixel itself (tag 77, bit 17).
fn rsdbg_refresh(ai: u32, dir: u32, rec: SfxRec) {
#if !DEBUG_NO_BINDINGS && RS_ARENA_BINDING
  if ((dbg.flags & DBGF_PROBE) == 0u) { return; }
  let px = vec2u(ai % rsParams.atlasSize.x, ai / rsParams.atlasSize.x);
  if (debug_is_probe(px)) { rsdbg_refresh_probe(px, dir, true, rec); }
#endif
}
#endif

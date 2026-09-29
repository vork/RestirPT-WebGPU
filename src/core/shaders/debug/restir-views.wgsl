// ReSTIR debug-view and probe hooks (restir-api.md §2.11; PLAN §6 M4 rows): view ids 400–499, probe tags 64–95.
// OWNER WP-D. Every hook is a no-op unless its view is active (dbg.mode) or the pixel is the probe pixel, so the
// debug-off path does no work beyond a uniform compare and never touches a production output (bit-identical
// validation). With DEBUG_NO_BINDINGS (test pipelines without G3) every hook compiles to an empty body.
//
// Who writes what (restir-api.md Changelog D1–D4):
//   400–410 reservoir views   rsdbg_reservoir (rs_initial: tap INITIAL; final-round resample: tap SPATIAL). The FINAL tap
//                             accepts both: the later stage of the frame overwrites (last writer wins).
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

/// After a reservoir is final (rs_initial after the final chunk: tap DBG_TAP_INITIAL; spatial resample of the final
/// round: DBG_TAP_SPATIAL). Reads the record back from resOut (every pass that calls it binds resOut).
fn rsdbg_reservoir(px: vec2u, ai: u32, tap: u32) {
#if !DEBUG_NO_BINDINGS && RS_RES_OUT_BINDING
  let probe = debug_is_probe(px);
  let view = rsdbg_view_in(RSV_C, RSV_KMARGIN) && (dbg.tap == tap || dbg.tap == DBG_TAP_FINAL);
  if (!view && !probe) { return; }
  if (probe) {
    let i = rsdbg_probe_block(1u + RS_RES_PLANES);
    if (i != 0xFFFFFFFFu) {
      let passId = select(RS_PASS_INITIAL, RS_PASS_SPATIAL + rsDispatch.round, tap == DBG_TAP_SPATIAL);
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

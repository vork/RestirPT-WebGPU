// RIS-NEE at x₁ over the frame's light tiles (restir-m6-api.md MD4; PLAN §2 rule 2; math.md#measure [M6 addition];
// gap-light §3.8; Enhanced S-§5). Pipeline variant RS_RIS_NEE (rs_initial only; shifts, replay and the refresh never
// draw from tiles, PLAN rule 10).
//   candidates  j = 0 … M−1: slot = h.x & 1023 of the pixel's light tile (one of 128, picked per 8×8 member-local screen
//               tile and frame), entry = tile[slot] (marginal pmf[L]), light-local words by nee_draw's per-entry rule
//               from (h.y, h.z, h.w), h = pcg4d(seed.x, seed.y, j, STREAM_RIS_NEE) of the TREE seed
//   target      p̂ = lum(f_all ⊙ Λ) at x (visibility excluded; an invalid sample, a sample on the shading triangle
//               itself (same-triangle skip, the PT's convention) or Λ = 0 has p̂ = 0)
//   ratio       r = p̂ / q (both in μ: Λ with q = P/A | P | pmf·pdf_σ)
//   selection   streaming, rs_rand(key, RS_PASS_RIS_NEE, (s << 20) | j) (resampling stream, never the path stream)
//   W_NEE       = (1/M)·Σr / r_Y   (the Enhanced "W^RIS·p1" with p1 = q in μ); 0 when every r is 0
// Plants (validation only, restir-m6-api.md §5.3): U8-8 RSF_PLANT_U8_RIS_MIXED: W ← W^RIS·p1_σ (= W_NEE·p1/q for area /
// triangle selections, the UCW in mixed measures); U8-10 RSF_PLANT_U8_TILE_PMF: the selection also reports the tile
// multiplicity of its entry (the caller replaces pmf by it inside ω1).
#include "restir/m6-types.wgsl"
#include "restir/endpoint.wgsl"
#include "restir/queue.wgsl"

#if RS_RIS_PREPASS
// perf2 WP-2d (RS_RIS_PREPASS, interactive, trees = 1): the selection is made by the lean pre-pass rs_ris_nee
// (passes/restir/initial.wgsl) and handed to rs_initial in the pixel's own RP_DIAG plane of resOut (dead until
// res_write_empty): (j | RIS_PRE_NONE, bits(W_NEE), entry, bits(mult)). rs_initial rebuilds the endpoint from (j, entry)
// with nee_draw_entry, exactly as the loop built it; W_NEE and the U8-10 multiplicity are the stored values.
const RIS_PRE_NONE: u32 = 0xFFFFFFFFu;
struct RisSel { ep: NeeEndpoint, W: f32, mult: f32, j: u32 }
#else
struct RisSel { ep: NeeEndpoint, W: f32, mult: f32 }   // mult: tile multiplicity / 1024 of the selection (U8-10 plant)
#endif

/// Light tile (0 … 127) of member-local pixel `local` of member `member` at frame t.
fn ris_tile_of(local: vec2u, member: u32) -> u32 {
  let tiles = (rsParams.memberSize.x + RS_SCREEN_TILE - 1u) / RS_SCREEN_TILE;
  let st = (local.y / RS_SCREEN_TILE) * tiles + local.x / RS_SCREEN_TILE;
  return pcg4d(vec4u(frame.runSeed ^ (member * 0x9e3779b9u), rs_t(), st, STREAM_LIGHT_TILE ^ 0xa511e9b3u)).x & (RS_TILES - 1u);
}

/// NEE endpoint of a GIVEN alias entry from three hashes (nee_draw's per-entry rule without the alias draw).
fn nee_draw_entry(slot: LightSlot, entry: u32, hL: vec3u) -> NeeEndpoint {
  if (entry == slot.envEntry) {
    let ij = env_draw_cell(hL.x, hL.y);
    return NeeEndpoint(entry, (ij.x << 16u) | ij.y, hL.z);
  }
  if (entry < slot.nAnalytic) {
    let kind = records[slot.lightOff + entry * LIGHT_REC_WORDS + 3u];
    if (kind == LT_POINT || kind == LT_SPOT || kind == LT_SUN) { return NeeEndpoint(entry, 0u, 0u); }
  }
  return NeeEndpoint(entry, bitcast<u32>(u32_to_unit(hL.x)), bitcast<u32>(u32_to_unit(hL.y)));
}

/// RIS over M tile candidates at shading point x (primId xPrim) with MatEval m and outgoing V. p is the pixel (member and
/// member-local position for the tile), atlas member index = p.member − memberBase.
fn ris_nee_select(p: RsPix, key: vec2u, seed: vec2u, s: u32, x: SurfaceHit, xPrim: u32, m: MatEval, V: vec3f) -> RisSel {
#if RS_RIS_PREPASS
  var out = RisSel(NeeEndpoint(LIGHT_NONE, 0u, 0u), 0.0, 0.0, RIS_PRE_NONE);
#else
  var out = RisSel(NeeEndpoint(LIGHT_NONE, 0u, 0u), 0.0, 0.0);
#endif
  let slot = lightsParams.cur;
  if (slot.nEntries == 0u) { return out; }
  let M = rs_ris_m();
  let mIdx = p.member - rsParams.memberBase;
  let tb = rs_tiles_base() + (mIdx * RS_TILES + ris_tile_of(p.local, p.member)) * RS_TILE_SIZE;
  var wSum = 0.0;
  var rSel = 0.0;
#if RS_RIS_HOIST
  let bctx = bsdf_prepare(m, V);   // perf2 WP-2a: loop-invariant (m, V), after the nEntries == 0 early return
#endif
  for (var j = 0u; j < M; j++) {
    let h = pcg4d(vec4u(seed.x, seed.y, j, STREAM_RIS_NEE));
    let entry = rsArena.words[tb + (h.x & (RS_TILE_SIZE - 1u))];
    if (entry == LIGHT_NONE) { continue; }
    let ep = nee_draw_entry(slot, entry, h.yzw);
    let ls = nee_eval(x.pos, ep);
    var r = 0.0;
    if (ls.valid && ls.prim != xPrim && any(ls.Lambda > vec3f(0.0))) {
#if RS_RIS_HOIST
      r = luminance(bsdf_f_all_ctx(bctx, V, ls.dir) * ls.Lambda) / ls.q;
#else
      let q = bsdf_query(m, V, ls.dir, LOBE_NEE);
      r = luminance(q.f_all * ls.Lambda) / ls.q;
#endif
    }
    if (ris_update(&wSum, r, rs_rand(key, RS_PASS_RIS_NEE, (s << 20u) | j))) {
      out.ep = ep;
      rSel = r;
#if RS_RIS_PREPASS
      out.j = j;
#endif
    }
  }
  if (rs_pos_finite(rSel) && rs_pos_finite(wSum)) { out.W = (wSum / f32(M)) / rSel; }
  if (rs_m6_flag(RSF_PLANT_U8_TILE_PMF) && out.ep.entry != LIGHT_NONE) {   // U8-10 plant: tile multiplicity of the pick
    var n = 0u;
    for (var i = 0u; i < RS_TILE_SIZE; i++) { n += select(0u, 1u, rsArena.words[tb + i] == out.ep.entry); }
    out.mult = f32(n) / f32(RS_TILE_SIZE);
  }
  return out;
}

#if RS_RIS_PREPASS
/// perf2 WP-2d: the pre-pass record of a selection (RP_DIAG plane of resOut).
fn ris_prepass_record(r: RisSel) -> vec4u { return vec4u(r.j, bitcast<u32>(r.W), r.ep.entry, bitcast<u32>(r.mult)); }

/// perf2 WP-2d: the selection of a pre-pass record for tree seed `seed`: the endpoint of candidate j rebuilt with
/// nee_draw_entry from the stored entry and the candidate's hashes (the loop's own rule), W_NEE and mult as stored.
fn ris_prepass_sel(rec: vec4u, seed: vec2u) -> RisSel {
  var out = RisSel(NeeEndpoint(LIGHT_NONE, 0u, 0u), bitcast<f32>(rec.y), bitcast<f32>(rec.w), rec.x);
  if (rec.x != RIS_PRE_NONE) {
    let h = pcg4d(vec4u(seed.x, seed.y, rec.x, STREAM_RIS_NEE));
    out.ep = nee_draw_entry(lightsParams.cur, rec.z, h.yzw);
  }
  return out;
}
#endif

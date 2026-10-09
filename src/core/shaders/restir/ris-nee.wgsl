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

struct RisSel { ep: NeeEndpoint, W: f32, mult: f32 }   // mult: tile multiplicity / 1024 of the selection (U8-10 plant)

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
#if RS_LIGHT_REC4
    let kind = light_kind(slot, entry);
#else
    let kind = records[slot.lightOff + entry * LIGHT_REC_WORDS + 3u];
#endif
    if (kind == LT_POINT || kind == LT_SPOT || kind == LT_SUN) { return NeeEndpoint(entry, 0u, 0u); }
  }
  return NeeEndpoint(entry, bitcast<u32>(u32_to_unit(hL.x)), bitcast<u32>(u32_to_unit(hL.y)));
}

/// RIS over M tile candidates at shading point x (primId xPrim) with MatEval m and outgoing V. p is the pixel (member and
/// member-local position for the tile), atlas member index = p.member − memberBase.
fn ris_nee_select(p: RsPix, key: vec2u, seed: vec2u, s: u32, x: SurfaceHit, xPrim: u32, m: MatEval, V: vec3f) -> RisSel {
  var out = RisSel(NeeEndpoint(LIGHT_NONE, 0u, 0u), 0.0, 0.0);
  let slot = lightsParams.cur;
  if (slot.nEntries == 0u) { return out; }
  let M = rs_ris_m();
  let mIdx = p.member - rsParams.memberBase;
  let tb = rs_tiles_base() + (mIdx * RS_TILES + ris_tile_of(p.local, p.member)) * RS_TILE_SIZE;
  var wSum = 0.0;
  var rSel = 0.0;
  for (var j = 0u; j < M; j++) {
    let h = pcg4d(vec4u(seed.x, seed.y, j, STREAM_RIS_NEE));
    let entry = rsArena.words[tb + (h.x & (RS_TILE_SIZE - 1u))];
    if (entry == LIGHT_NONE) { continue; }
    let ep = nee_draw_entry(slot, entry, h.yzw);
    let ls = nee_eval(x.pos, ep);
    var r = 0.0;
    if (ls.valid && ls.prim != xPrim && any(ls.Lambda > vec3f(0.0))) {
      let q = bsdf_query(m, V, ls.dir, LOBE_NEE);
      r = luminance(q.f_all * ls.Lambda) / ls.q;
    }
    if (ris_update(&wSum, r, rs_rand(key, RS_PASS_RIS_NEE, (s << 20u) | j))) {
      out.ep = ep;
      rSel = r;
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

// Paired-spatial MIS, resampling and write-back (restir-api.md §3.8; math.md#paired-mis, #confidence).
// OWNER WP-C. P0 STUB (restir-api.md §1.4): spatial_resample copies resIn → resOut and shades F·W (no reuse).
// mis_canonical_term / mis_partner_weight are the real formulas.
#include "restir/reservoir.wgsl"
#include "restir/queue.wgsl"

/// a·pc/(a·pc + cj·lumH), := 1 if the denominator is 0.
fn mis_canonical_term(a: f32, pc: f32, cj: f32, lumH: f32) -> f32 {
  let den = a * pc + cj * lumH;
  return select(1.0, a * pc / den, den > 0.0);
}
/// [cj·pj/(cj·pj + a·lumG)]/(k+1), := 0 if the denominator is 0.
fn mis_partner_weight(a: f32, cj: f32, pj: f32, lumG: f32, k: u32) -> f32 {
  let den = cj * pj + a * lumG;
  return select(0.0, cj * pj / den, den > 0.0) / f32(k + 1u);
}

#if RS_RES_IN_BINDING && RS_RES_OUT_BINDING
/// Write-back of a partner sample (math.md#jacobian, PLAN rule 4): copy all ten planes of resIn[src], then F = G/J,
/// jDen = J·jDen_src, W, c.
fn res_select_shifted(dst: u32, src: u32, G: vec3f, J: f32, W: f32, c: f32) {
  for (var pl = 0u; pl < RS_RES_PLANES; pl++) { resout_set(dst, pl, resin_plane(src, pl)); }
  resout_set(dst, RP_WF, vec4u(bitcast<u32>(W), bitcast<vec3u>(G / J)));
  let p1 = resin_plane(src, RP_SEED);
  resout_set(dst, RP_SEED, vec4u(p1.xyz, bitcast<u32>(c)));
  let p2 = resin_plane(src, RP_RC);
  resout_set(dst, RP_RC, vec4u(p2.xyz, bitcast<u32>(J * bitcast<f32>(p2.w))));
}

/// STUB (P0): canonical copy; the final round shades F·W.
fn spatial_resample(p: RsPix, round: u32, finalRound: bool) {
  for (var pl = 0u; pl < RS_RES_PLANES; pl++) { resout_set(p.ai, pl, resin_plane(p.ai, pl)); }
#if RS_SHADE_W_BINDING
  if (finalRound) {
    let p0 = resin_plane(p.ai, RP_WF);
    textureStore(rsShadeOut, p.px, vec4f(rp_F(p0) * rp_W(p0), 0.0));
  }
#endif
}
#endif

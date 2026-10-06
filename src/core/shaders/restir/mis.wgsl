// Paired-spatial MIS, resampling and write-back (restir-api.md §3.8; math.md#paired-mis, #confidence, #jacobian;
// gap-rc §7.3–7.4; PLAN §2 rule 8). OWNER WP-C.
// Defensive pairwise MIS (GRIS Eq. 38 with |R| = 1 and confidences), computed from the slots only, for canonical
// pixel c with S_c = accepted slots (VALID or FAILED; PENDING counts as FAILED), k = |S_c|, a = c_c / k:
//   m_c = [1 + Σ_j a·p̂_c / (a·p̂_c + c_j·lum H_j)] / (k+1)      H_j = slot (c, s): c's sample shifted into j
//   m_j = [c_j·p̂_j / (c_j·p̂_j + a·lum G_j)] / (k+1)            G_j = slot (j, s): j's sample shifted into c (0 FAILED)
//   w_c = m_c·p̂_c·W_c (counter 0), w_j = m_j·lum(G_j)·W_j (counter 1 + s), streamed with rs_rand(key_c,
//   RS_PASS_SPATIAL + round, ·); W_Y = Σw / p̂_c(Y), p̂_c(Y_j) = lum(G_j / J_j); c_out = c_c + Σ_{S_c} c_j (uncapped).
//   RGB shading L = m_c·F_c·W_c + Σ m_j·G_j·W_j (lum L = Σw). W and L are multiplied by RestirParams.wScale (plant,
//   1 = off; the final-round shade carries it so that the W × s plant reaches the image).
// Do NOT copy EvanLuo42's MIS (arguments swapped). Write-back of a partner sample: copy all ten planes of resIn[j],
// then F = G/J, jDen = J·jDen_j, W, c (res_select_shifted).
// M5 (OWNER T-D): boost slots (TD21) are ordinary slots here (the loops run over numSlots = slots + boostSlots);
// U8 plant 9 (RSF_PLANT_U8_FAILED_K) drops failed neighbours from S_c.
#include "common/math.wgsl"
#include "restir/reservoir.wgsl"
#include "restir/queue.wgsl"
#include "restir/pairing.wgsl"
#include "debug/restir-views.wgsl"

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

// Resample counters, aggregated per workgroup (the pass flushes them after a barrier with mis_flush_counters):
// 0 RSC_PENDING_LEFT, 1 RSC_SLOT_MISMATCH, 2 RSC_W_NONFINITE, 3 RSC_SELECTED_SHIFTED, 4 RSC_EMPTY_CANON.
var<workgroup> misCnt: array<atomic<u32>, 5>;

#if RS_ARENA_BINDING && RS_ARENA_RW
fn mis_flush_counters(li: u32) {
  if (li == 0u) {
    rs_count(RSC_PENDING_LEFT, atomicLoad(&misCnt[0]));
    rs_count(RSC_SLOT_MISMATCH, atomicLoad(&misCnt[1]));
    rs_count(RSC_W_NONFINITE, atomicLoad(&misCnt[2]));
    rs_count(RSC_SELECTED_SHIFTED, atomicLoad(&misCnt[3]));
    rs_count(RSC_EMPTY_CANON, atomicLoad(&misCnt[4]));
  }
}
#endif

#if RS_RES_IN_BINDING && RS_RES_OUT_BINDING && RS_ARENA_BINDING
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

/// Copy of the canonical record of `dst` (resIn → resOut) with W and c replaced.
fn res_copy_canonical(dst: u32, W: f32, c: f32) {
  for (var pl = 0u; pl < RS_RES_PLANES; pl++) { resout_set(dst, pl, resin_plane(dst, pl)); }
  let p0 = resin_plane(dst, RP_WF);
  resout_set(dst, RP_WF, vec4u(bitcast<u32>(W), p0.yzw));
  let p1 = resin_plane(dst, RP_SEED);
  resout_set(dst, RP_SEED, vec4u(p1.xyz, bitcast<u32>(c)));
}

fn mis_store_shade(p: RsPix, finalRound: bool, L: vec3f) {
#if RS_SHADE_W_BINDING
  if (finalRound) { textureStore(rsShadeOut, p.px, vec4f(L, 0.0)); }
#endif
}

/// Paired spatial resampling of canonical pixel p for round `round` (§3.8). Reads resIn and the slots, writes
/// resOut[p.ai] (+ rsShade on the final round).
fn spatial_resample(p: RsPix, round: u32, finalRound: bool) {
  let c = p.ai;
  let p0 = resin_plane(c, RP_WF);
  let p1 = resin_plane(c, RP_SEED);
  let Fc = rp_F(p0);
  let Wc = rp_W(p0);
  let cc = rp_c(p1);
  let pc = luminance(Fc);
  let bg = (p1.z & RF_BG) != 0u;
  if (!bg && res_empty(p1.z)) { atomicAdd(&misCnt[4], 1u); }
  let t = rs_t();
  let NS = rsParams.numSlots;

  // S_c: partners of accepted slots (0xFFFFFFFF = not in S_c); consistency counters.
  var qs: array<u32, 6>;
  var k = 0u;
  var cOut = cc;
  for (var s = 0u; s < NS; s++) {
    qs[s] = 0xFFFFFFFFu;
    let jw = arena_slot_jword(c, s);
    let pr = pair_partner(p.local, p.member, t, round, s);
    var q = 0xFFFFFFFFu;
    var jwq = JW_NOT_ACCEPTED;
    if (pr.valid) {
      q = pair_atlas_index(pair_atlas_px(p, pr.partner));
      jwq = arena_slot_jword(q, s);
    }
    if (jw_accepted(jw) != jw_accepted(jwq)) { atomicAdd(&misCnt[1], 1u); }
    if (!jw_accepted(jw) || !pr.valid) { continue; }
    if (jw == JW_PENDING) { atomicAdd(&misCnt[0], 1u); }
    // U8 plant 9 (RSF_PLANT_U8_FAILED_K, validation only; restir-temporal-api.md §6.5): failed neighbours (G_j not VALID)
    // are removed from S_c, i.e. from k, the normaliser 1/(k+1) and c_out (biased: energy loss).
    if ((rsParams.flags & RSF_PLANT_U8_FAILED_K) != 0u && !jw_valid(jwq)) { continue; }
    qs[s] = q;
    k++;
    cOut += rp_c(resin_plane(q, RP_SEED));
  }
  let wScale = rsParams.wScale;
  if (k == 0u) {
    // No partner: the canonical sample as is (background: F = W = 0, shade 0).
    for (var pl = 0u; pl < RS_RES_PLANES; pl++) { resout_set(c, pl, resin_plane(c, pl)); }
    var W0 = Wc * wScale;
    if (!is_finite(W0)) { atomicAdd(&misCnt[2], 1u); W0 = 0.0; }
    resout_set(c, RP_WF, vec4u(bitcast<u32>(W0), p0.yzw));
    mis_store_shade(p, finalRound, select(Fc * Wc * wScale, vec3f(0.0), Wc == 0.0));
    rsdbg_mis(p.px, 0u, 1.0, 0.0, 0.0, 0xFFu, 1.0, pc * Wc, 0u);
    return;
  }
  let a = cc / f32(k);

  // m_c(X_c) and, for the diagnostic view 461, Σ_j m_j(X_c) (p̂_{←j}(X_c) = lum H_j).
  var sumT = 1.0;
  var sumMj = 0.0;
  for (var s = 0u; s < NS; s++) {
    let q = qs[s];
    if (q == 0xFFFFFFFFu) { continue; }
    let h = arena_slot(c, s);
    let lumH = select(0.0, luminance(bitcast<vec3f>(h.xyz)), jw_valid(h.w));
    let cj = rp_c(resin_plane(q, RP_SEED));
    sumT += mis_canonical_term(a, pc, cj, lumH);
    sumMj += mis_partner_weight(a, cj, lumH, pc, k);
  }
  let mc = sumT / f32(k + 1u);
  let sumM = mc + sumMj - 1.0;

  // Streaming RIS over {canonical, S_c}.
  let key = rs_frame_key(p.member, t, p.localIdx);
  let passId = RS_PASS_SPATIAL + round;
  var wSum = 0.0;
  var sel = 0u;                   // 0 canonical, 1 + s partner
  var selQ = c;
  var selG = vec3f(0.0);
  var selJ = 1.0;
  let wc = mc * pc * Wc;
  if (ris_update(&wSum, wc, rs_rand(key, passId, 0u))) { sel = 0u; }
  var L = select(mc * Fc * Wc, vec3f(0.0), Wc == 0.0);
  for (var s = 0u; s < NS; s++) {
    let q = qs[s];
    if (q == 0xFFFFFFFFu) { continue; }
    let g = arena_slot(q, s);
    let valid = jw_valid(g.w);
    let G = select(vec3f(0.0), bitcast<vec3f>(g.xyz), valid);
    let J = bitcast<f32>(g.w);
    let pq0 = resin_plane(q, RP_WF);
    let pj = luminance(rp_F(pq0));
    let Wj = rp_W(pq0);
    let cj = rp_c(resin_plane(q, RP_SEED));
    let lumG = luminance(G);
    let mj = select(0.0, mis_partner_weight(a, cj, pj, lumG, k), valid);
    let wj = mj * lumG * Wj;
    if (ris_update(&wSum, wj, rs_rand(key, passId, 1u + s))) { sel = 1u + s; selQ = q; selG = G; selJ = J; }
    if (valid && Wj != 0.0) { L += mj * G * Wj; }
    rsdbg_mis(p.px, k, mc, sumM, 0.0, s, mj, wj, sel);
  }

  // Write-back.
  var W = 0.0;
  if (sel == 0u) {
    if (wSum > 0.0) { W = wSum / pc; }
    W *= wScale;
    if (!is_finite(W)) { atomicAdd(&misCnt[2], 1u); W = 0.0; }
    res_copy_canonical(c, W, cOut);
  } else {
    W = wSum / luminance(selG / selJ) * wScale;
    if (!is_finite(W)) { atomicAdd(&misCnt[2], 1u); W = 0.0; }
    res_select_shifted(c, selQ, selG, selJ, W, cOut);
    atomicAdd(&misCnt[3], 1u);
  }
  mis_store_shade(p, finalRound, L * wScale);
  let lumRel = (luminance(L) - wSum) / max(wSum, 1e-30);
  rsdbg_mis(p.px, k, mc, sumM, lumRel, 0xFFu, mc, wc, sel);
}
#endif

// Unified path tree with streaming RIS (restir-api.md §3.5; math.md#path-tree, #rc-predicate, #rr, #reservoir-fields,
// #jacobian table; PLAN §3 pass 2). One tree per (pixel, s): the PT's own path (same seed, same slot layout, the PT's
// NEE / MIS / RR formulas); each NEE sample and each BSDF ending (emissive-triangle hit, env escape) is a candidate
// with m = 1 (disjoint domains by (d, technique, endpoint)). Every candidate carries its own deferred k* (no lobe
// revocation), the case fields of §2.2 and the suffix cache (§2.4), and is streamed with w = lum(F)·W_src
// (W_src = 1/∏q_RR of the RR tests it survived, §3.10). The selection is written to resOut[ai] immediately (no
// register-held candidate); Σw and the candidate count persist in the record between tree chunks (D10); the final
// chunk sets W = Σw/(S·lum F_Y).
//
// Directions between stored vertices are normalize(pos(b) − pos(a)) from positions rebuilt from ids (D3); the sampled
// direction is only traced (and is the escape direction of a BSDF_ENV ending). Throughput uses the sampler's weight.
// F is RR-free. Background pixels never reach here (rs_initial writes their record).
//
// RS_PT_DIRECTIONS (tests only, U-RIS-1 part a): the PT's directions instead of D3 (V₁ = −camera ray, ω = the
// sampled direction, vertices oriented against the ray), so the tree reproduces pt_trace's contributions up to
// expression order. Never used by production pipelines.
// RS_DUMP_CANDIDATES (tests, §2.12): every streamed candidate of tree 0 (≤ RS_DUMP_CAP per pixel) is also written to
// candDump as a full record (W = its RIS weight w_i, wSum = Σw before it, nCand = its ordinal) + the primIds of the
// base path's x₁…x₈ (BVH_MISS beyond x_{d−1}); the per-pixel count follows the records.
#include "restir/reservoir.wgsl"
#include "restir/endpoint.wgsl"
#include "restir/rc.wgsl"
#include "path/path-weight.wgsl"
#include "restir/queue.wgsl"
#include "debug/restir-views.wgsl"

#if RS_DUMP_CANDIDATES
@group(2) @binding(4) var<storage, read_write> candDump: array<u32>;
var<private> ptDumpPrims: array<u32, 8>;
#endif

/// Joint pdf used in the stored Jacobian denominators (RSF_PLANT_MARGINAL_J: the marginal, U-11 negative control).
fn pt_jpdf(q: BsdfQuery) -> f32 {
  return select(q.p_joint, q.p_marg, (rsParams.flags & RSF_PLANT_MARGINAL_J) != 0u);
}

fn pt_hist_set(h: u32, b: u32, code: u32) -> u32 {
  if (b < 1u || b > 8u) { return h; }
  let s = 4u * (b - 1u);
  return (h & ~(0xFu << s)) | ((code & 0xFu) << s);
}

/// One plane of a candidate: to resOut[ai] when selected, to dump slot `di` when dumping (di ≠ 0xFFFFFFFF).
fn pt_put(ai: u32, sel: bool, di: u32, i: u32, v: vec4u) {
  if (sel) { resout_set(ai, i, v); }
#if RS_DUMP_CANDIDATES
  if (di != 0xFFFFFFFFu) {
    let base = (ai * RS_DUMP_CAP + di) * RS_DUMP_WORDS + 4u * i;
    candDump[base] = v.x; candDump[base + 1u] = v.y; candDump[base + 2u] = v.z; candDump[base + 3u] = v.w;
  }
#endif
}

fn pt_emit(ai: u32, sel: bool, di: u32, w: f32, F: vec3f, seed: vec2u, flags: u32, rc: vec3u, jDen: f32, rcWi: vec3f, aux: f32,
           rcRad: vec3f, wBefore: f32, end: vec3u, hist: u32, sfx: vec3u, sfxFlags: u32, sfxDir: vec3f, sfxT: f32, betaS: vec3f,
           sfxP2: f32, ordinal: u32, selId: u32, kMargin: f32, endpointId: u32) {
  pt_put(ai, sel, di, RP_WF, vec4u(bitcast<u32>(w), bitcast<vec3u>(F)));
  pt_put(ai, sel, di, RP_SEED, vec4u(seed, flags, bitcast<u32>(1.0)));
  pt_put(ai, sel, di, RP_RC, vec4u(rc, bitcast<u32>(jDen)));
  pt_put(ai, sel, di, RP_WI, vec4u(bitcast<vec3u>(rcWi), bitcast<u32>(aux)));
  pt_put(ai, sel, di, RP_RAD, vec4u(bitcast<vec3u>(rcRad), bitcast<u32>(wBefore)));
  pt_put(ai, sel, di, RP_END, vec4u(end, hist));
  pt_put(ai, sel, di, RP_SFX0, vec4u(sfx, sfxFlags));
  pt_put(ai, sel, di, RP_SFX1, vec4u(bitcast<vec3u>(sfxDir), bitcast<u32>(sfxT)));
  pt_put(ai, sel, di, RP_SFX2, vec4u(bitcast<vec3u>(betaS), bitcast<u32>(sfxP2)));
  pt_put(ai, sel, di, RP_DIAG, vec4u(ordinal, selId, bitcast<u32>(kMargin), endpointId));
#if RS_DUMP_CANDIDATES
  if (di != 0xFFFFFFFFu) {
    let base = (ai * RS_DUMP_CAP + di) * RS_DUMP_WORDS + 40u;
    for (var b = 0u; b < 8u; b++) { candDump[base + b] = ptDumpPrims[b]; }
  }
#endif
}

fn pathtree_run(p: RsPix, key: vec2u, treeBase: u32, treeCount: u32, firstChunk: bool, finalChunk: bool) {
  let ai = p.ai;
  if (firstChunk) { res_write_empty(ai, key, false); }
  var wSum = bitcast<f32>(resout_plane(ai, RP_RAD).w);
  var nCand = resout_plane(ai, RP_DIAG).x;
#if RS_DUMP_CANDIDATES
  var nDump = 0u;
#endif
  let vb = rs_vbuf(p.px);
  let thr = rs_geo(p.px).w;
  let camPos = rs_cam_pos();
  let maxB = rsParams.maxBounces + 1u;
  let rr = (rsParams.flags & RSF_RR) != 0u;
  let envPresent = (envParams.flags & ENV_FLAG_PRESENT) != 0u;
  bvh_stats_reset();
  var nonFinite = 0u;
  var jdenBad = 0u;

  for (var s = treeBase; s < treeBase + treeCount; s++) {
    let seed = rs_tree_seed(key, s);
    var curPrim = vb.x;
    var curIds = vb.xyz;
#if RS_PT_DIRECTIONS
    let ray0 = frame_camera_ray(p.local, rs_jitter(key, p));
    var cur = scene_surface(vb.x, bitcast<f32>(vb.y), bitcast<f32>(vb.z), ray0.d);
    var V = -ray0.d;
#else
    var cur = vertex_from_ids(vb.x, bitcast<f32>(vb.y), bitcast<f32>(vb.z), camPos);
    var V = normalize(camPos - cur.pos);
#endif
    var beta = vec3f(1.0);
    var rrInv = 1.0;
    var hist = RS_HIST_NONE;
    // shared prefix rc of the tree: the first passing pair (x_{k−1}, x_k) with EV_BSDF at x_k (gap-rc §3.4)
    var treeRc = 0u;
    var treeIds = vec3u(RC_NONE, 0u, 0u);
    var treeWi = vec3f(0.0);
    var treeJDen = 1.0;
    var treeL = vec2u(LOBE_NONE, LOBE_NONE);       // (ℓ_{k−1} | δ_{k−1} << 3, ℓ_k | δ_k << 3)
    var treeMargin = 0.0;
    var betaPost = vec3f(1.0);                     // ∏ of the weights at x_{k+1} … (post-rc suffix)
    var prevV = RcVertex(camPos, vec3f(0.0), RCK_SURFACE, 0u);
    var prevE = rc_event_none();
    var prevPJoint = 1.0;
    var prevPrim = BVH_MISS;
    // Changelog B-5: the rc segment of a candidate is also tested with the shift's visible() (the closest-hit ray along
    // the SAMPLED direction and the shadow segment between the stored vertices can disagree on a measure ~1e-6 set);
    // a candidate whose rc segment fails it is not streamed (F := 0), so base and shifts share one visibility term
    var treeVis = true;
#if RS_DUMP_CANDIDATES
    for (var b = 0u; b < 8u; b++) { ptDumpPrims[b] = 0xFFFFFFFFu; }
#endif
    for (var B = 1u; B <= maxB; B++) {
#if RS_DUMP_CANDIDATES
      if (B <= 8u) { ptDumpPrims[B - 1u] = curPrim; }
#endif
      rsdbg_vertex(p.px, 0u, B, cur.pos, (hist >> (4u * min(B - 1u, 7u))) & 0xFu);
      let m = material_eval(cur, V);
      let curV = rc_vertex(cur, m);
      // ---- (1) NEE candidate, d = B + 1 (not at a delta-only vertex) ------------------------------------------------
      if ((m.flags & MATEVAL_HAS_NON_DELTA) != 0u) {
        let ep = nee_draw(lightsParams.cur, rs_path_hash(seed, B, SLOT_SEL), rs_path_hash(seed, B, SLOT_SEL2),
                          vec3u(rs_path_hash(seed, B, SLOT_L0), rs_path_hash(seed, B, SLOT_L1), rs_path_hash(seed, B, SLOT_L2)));
        let ls = nee_eval(cur.pos, ep);
        // same-triangle skip (Cycles shade_surface.h:345-351), as the PT
        if (ls.valid && ls.prim != curPrim && any(ls.Lambda > vec3f(0.0))) {
          let qn = bsdf_query(m, V, ls.dir, LOBE_NEE);
          let w1 = nee_mis_w1(ls, qn.p_marg, B);
          let F = (w1 / ls.q) * (beta * qn.f_all * ls.Lambda);
          if (any(F > vec3f(0.0)) && nee_visible(cur, curPrim, ls)) {
            let km = kstar_nee(treeRc, B, prevV, prevE, curV, rc_event_nee(m, qn.p_marg), thr);
            let k = km.x;
            var rc = nee_endpoint_words(ep);           // (a)/(f) forced: rc = the NEE light vertex
            var rcWi = vec3f(0.0);
            var jDen = 1.0;
            var aux = 0.0;
            var rcRad = vec3f(0.0);
            var lkm1 = LOBE_NEE;
            var lk = LOBE_NONE;
            var margin = bitcast<f32>(km.y);
            if (k == B) {                              // (b) N1: rc at x_{d−1}
              rc = curIds; rcWi = ls.dir; aux = ls.p1; rcRad = ls.Lambda;
              jDen = prevPJoint * rc_G(prevV.pos, cur.pos, cur.ng);
              lkm1 = prevE.lobe | (prevE.delta << 3u); lk = LOBE_NEE;
            } else if (k <= B - 1u) {                  // D-NEE: the tree's rc (k ≤ d − 2)
              rc = treeIds; rcWi = treeWi; jDen = treeJDen; lkm1 = treeL.x; lk = treeL.y; margin = treeMargin;
              rcRad = betaPost * (w1 / ls.q) * qn.f_all * ls.Lambda;
            }
            var visOk = true;
            if (k == B) { visOk = visible(prevV.pos, prevV.ng, prevPrim, cur.pos, cur.ng, curPrim); }
            else if (k <= B - 1u) { visOk = treeVis; }
            let w = select(0.0, luminance(F) * rrInv, visOk);
            let counter = (s << 20u) | (B << 12u);
            let wBefore = wSum;
            var sel = false;
            if (!rs_pos_finite(w)) {
              if (!(w == 0.0)) { nonFinite++; }
            } else {
              sel = ris_update(&wSum, w, rs_rand(key, RS_PASS_INITIAL, counter));
              var di = 0xFFFFFFFFu;
#if RS_DUMP_CANDIDATES
              if (s == 0u && nDump < RS_DUMP_CAP) { di = nDump; nDump++; }
#endif
              let flags = rf_pack(B + 1u, k, RS_TECH_NEE, ls.kind, ls.isDelta, lkm1 & 7u, (lkm1 & 8u) != 0u, lk & 7u, (lk & 8u) != 0u, k == B + 1u);
              pt_emit(ai, sel, di, w, F, seed, flags, rc, jDen, rcWi, aux, rcRad, wBefore, nee_endpoint_words(ep),
                      pt_hist_set(hist, B, LOBE_NEE), curIds, select(0u, SFX_VALID, k <= B), V, 0.0,
                      select(vec3f(1.0), betaPost, k <= B - 1u), qn.p_marg, nCand, counter, margin, nee_endpoint_id(ep));
              nCand++;
            }
            rsdbg_candidate(p.px, B + 1u, RS_TECH_NEE, k, w, luminance(F), counter, sel);
          }
        }
      }
      // ---- (2) Russian roulette (D11: the PT's rule on the RR-free β, after NEE, before the continuation) -----------
      if (rr && B > rsParams.rrMinBounces) {
        let q = min(sqrt(max(max(beta.x, beta.y), beta.z)), 1.0);
        if (!(rs_path_u01(seed, B, SLOT_RR) < q)) { break; }
        rrInv /= q;
      }
      // ---- (3) BSDF continuation ------------------------------------------------------------------------------------
      let bs = bsdf_sample(m, V, rs_path_bsdf_u4(seed, B));
      if (!bs.valid) { break; }
      hist = pt_hist_set(hist, B, bs.lobe | select(0u, 8u, bs.is_delta));
      let org = offset_ray(cur.pos, select(-cur.ng, cur.ng, dot(cur.ng, bs.L) >= 0.0));
      let h = trace_closest_ex(org, bs.L, FLT_MAX, curPrim, BVH_MISS);
      let isHit = h.primId != BVH_MISS;
      var nxt = cur;
      var wOut = bs.L;
      if (isHit) {
#if RS_PT_DIRECTIONS
        nxt = scene_surface(h.primId, h.u, h.v, bs.L);
#else
        nxt = vertex_from_ids(h.primId, h.u, h.v, cur.pos);
        wOut = normalize(nxt.pos - cur.pos);                // D3: same-formula direction
#endif
      }
      let qb = bsdf_query(m, V, wOut, bs.lobe);
      let eB = rc_event_bsdf(m, bs.lobe, bs.is_delta, qb.p_marg);
      if (!bs.is_delta && !rs_pos_finite(qb.p_joint)) { jdenBad++; }
      // tree pair B: (x_{B−1}, x_B), EV_BSDF at x_B
      if (treeRc == 0u && B >= 2u) {
        let r = kstar_tree_pair(prevV, prevE, curV, eB, thr);
        if (r.ok) {
          treeRc = B; treeIds = curIds; treeWi = wOut; treeMargin = r.margin;
          treeL = vec2u(prevE.lobe | (prevE.delta << 3u), eB.lobe | (eB.delta << 3u));
          treeJDen = prevPJoint * rc_G(prevV.pos, cur.pos, cur.ng) * pt_jpdf(qb);
          betaPost = vec3f(1.0);
          treeVis = visible(prevV.pos, prevV.ng, prevPrim, cur.pos, cur.ng, curPrim);
        }
      }
      let wq = rs_path_weight(qb, bs.weight, bs.is_delta);   // D3 / Changelog B-2: the shift's own factor formula
      beta *= wq;
      if (treeRc != 0u && treeRc < B) { betaPost *= wq; }
      if (!any(beta > vec3f(0.0))) { break; }
      // (4) Mode-B crossings: RS_MODE_B = 0 in M4 (D1)
      // ---- (5) BSDF endings at x_{B+1}, d = B + 1 -------------------------------------------------------------------
      var endF = vec3f(0.0);
      var endLe = vec3f(0.0);
      var endP1 = 0.0;
      var endW2 = 1.0;
      var isEnd = false;
      var endV = RcVertex(cur.pos + bs.L, vec3f(0.0), RCK_ENV, 0u);
      if (!isHit) {
        if (envPresent) {
          endW2 = env_bsdf_mis_weight(bs.L, qb.p_marg, B, bs.is_delta);
          endLe = envRadiance(envUV(bs.L, envParams.cg, envParams.sg));
          endF = endW2 * beta * endLe;
          endP1 = p1Env(bs.L);
          isEnd = true;
        }
      } else {
        endLe = tri_emission(h.primId, h.u, h.v);
        if (any(endLe > vec3f(0.0))) {
          endP1 = tri_light_p1(cur.pos, nxt.pos, nxt.ng, h.primId);
          if (!bs.is_delta) { endW2 = mis_w2(endP1, qb.p_marg, B); }
          endF = endW2 * beta * endLe;
          endV = RcVertex(nxt.pos, nxt.ng, RCK_LIGHT, 0u);
          isEnd = true;
        }
      }
      if (isEnd) {
        let km = kstar_bsdf_end(treeRc, B, curV, eB, endV, thr);
        let k = km.x;
        let tech = select(RS_TECH_BSDF_ENV, RS_TECH_BSDF_TRI, isHit);
        let endW = select(vec3u(RC_ENV_DIR, 0u, 0u), vec3u(h.primId, bitcast<u32>(h.u), bitcast<u32>(h.v)), isHit);
        var rc = vec3u(RC_NONE, 0u, 0u);               // ∅: full replay
        var rcWi = vec3f(0.0);
        var jDen = 1.0;
        var aux = 0.0;
        var rcRad = vec3f(0.0);
        var lkm1 = LOBE_NONE;
        var lk = LOBE_NONE;
        var margin = bitcast<f32>(km.y);
        if (k == B + 1u) {                             // (d) emitter rc / (e) env rc
          rc = endW;
          lkm1 = eB.lobe | (eB.delta << 3u);
          if (isHit) { jDen = pt_jpdf(qb) * rc_G(cur.pos, nxt.pos, nxt.ng); }
          else { jDen = pt_jpdf(qb); rcWi = bs.L; }
        } else if (k != 0u) {                          // (c) B1 (tree rc at x_B = x_{d−1}) or deep (k ≤ d − 2)
          rc = treeIds; rcWi = treeWi; jDen = treeJDen; lkm1 = treeL.x; lk = treeL.y; margin = treeMargin;
          if (k == B) { rcRad = endLe; aux = endP1; }
          else { rcRad = betaPost * endW2 * endLe; }
        }
        var visOk = true;                            // B-5 (env rc: the same ray as visibleInf, no extra test)
        if (k == B + 1u && isHit) { visOk = visible(cur.pos, cur.ng, curPrim, nxt.pos, nxt.ng, h.primId); }
        else if (k != 0u && k <= B) { visOk = treeVis; }
        let w = select(0.0, luminance(endF) * rrInv, visOk);
        let counter = (s << 20u) | (B << 12u) | 1u;
        let wBefore = wSum;
        var sel = false;
        if (!rs_pos_finite(w)) {
          if (!(w == 0.0)) { nonFinite++; }
        } else {
          sel = ris_update(&wSum, w, rs_rand(key, RS_PASS_INITIAL, counter));
          var di = 0xFFFFFFFFu;
#if RS_DUMP_CANDIDATES
          if (s == 0u && nDump < RS_DUMP_CAP) { di = nDump; nDump++; }
#endif
          let flags = rf_pack(B + 1u, k, tech, select(LT_ENV, LT_TRI, isHit), false, lkm1 & 7u, (lkm1 & 8u) != 0u, lk & 7u, (lk & 8u) != 0u, false);
          let sfxFlags = SFX_BSDF_END | select(SFX_ESCAPE, 0u, isHit) | select(0u, SFX_VALID, k != 0u && k <= B);
          pt_emit(ai, sel, di, w, endF, seed, flags, rc, jDen, rcWi, aux, rcRad, wBefore, endW, hist, curIds, sfxFlags, wOut,
                  select(FLT_MAX, h.t, isHit), select(vec3f(1.0), betaPost, k != 0u && k <= B - 1u), qb.p_marg, nCand, counter,
                  margin, select(RS_ENV_ID, h.primId, isHit));
          nCand++;
        }
        rsdbg_candidate(p.px, B + 1u, tech, k, w, luminance(endF), counter, sel);
      }
      if (!isHit) { break; }
      prevV = curV;
      prevE = eB;
      prevPJoint = pt_jpdf(qb);
      prevPrim = curPrim;
      cur = nxt;
      curPrim = h.primId;
      curIds = vec3u(h.primId, bitcast<u32>(h.u), bitcast<u32>(h.v));
      V = -wOut;
    }
  }

  // persist Σw and the candidate count (D10); finalise W on the last chunk
  resOut[ai * RS_RES_PLANES + RP_RAD].w = bitcast<u32>(wSum);
  resOut[ai * RS_RES_PLANES + RP_DIAG].x = nCand;
#if RS_DUMP_CANDIDATES
  if (treeBase == 0u) { candDump[rs_atlas_pixels() * RS_DUMP_CAP * RS_DUMP_WORDS + ai] = nDump; }
#endif
  if (finalChunk) {
    let lumF = luminance(rp_F(resout_plane(ai, RP_WF)));
    if (!(wSum > 0.0) || res_empty(resout_plane(ai, RP_SEED).z) || !(lumF > 0.0)) {
      res_write_empty(ai, key, false);
      resOut[ai * RS_RES_PLANES + RP_RAD].w = bitcast<u32>(wSum);
      resOut[ai * RS_RES_PLANES + RP_DIAG].x = nCand;
    } else {
      let W = wSum / (f32(rsParams.numTrees) * lumF);
      if (!rs_pos_finite(W)) { rs_count(RSC_W_NONFINITE, 1u); }
      resOut[ai * RS_RES_PLANES + RP_WF].x = bitcast<u32>(W);
    }
    rsdbg_reservoir(p.px, ai, 1u);
  }
  rs_count(RSC_CAND_NONFINITE, nonFinite);
  rs_count(RSC_BASE_JDEN_INVALID, jdenBad);
  let fl = bvh_stats().w;
  if ((fl & BVH_FLAG_OVERFLOW) != 0u) { rs_count(RSC_BVH_OVERFLOW, 1u); }
  if ((fl & BVH_FLAG_ITERCAP) != 0u) { rs_count(RSC_BVH_ITERCAP, 1u); }
}

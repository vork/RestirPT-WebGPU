// Reconnection predicate and vertex reconstruction (restir-api.md §3.4; math.md#rc-predicate, gap-rc §3).
// OWNER WP-B. P0 state (restir-api.md §1.4): rcPairTest is a STUB that fails every pair (margin 0), so every NEE
// candidate is forced (k = d) and every BSDF ending is k = ∅ — valid and unbiased with maximal replay. The trivial
// helpers (vertex_from_ids, rc_vertex, rc_event_*, rc_G, primaryThreshold) and the k* drivers in terms of rcPairTest
// are real so the path tree already has its final shape.
#include "restir/types.wgsl"
#include "material/material-eval.wgsl"

struct RcVertex { pos: vec3f, ng: vec3f, kind: u32, diffuseOnly: u32 }   // kind RCK_*
struct RcEvent  { lobe: u32, delta: u32, alpha: f32, pMarg: f32 }         // alpha = perceptual roughness of the event
struct RcResult { ok: bool, margin: f32, term: u32 }                       // term RCT_* of the first failing sub-test

/// Surface at (prim, u, v) with ng/ns flipped toward fromPos: bitwise scene_surface(prim, u, v, pos − fromPos).
fn vertex_from_ids(prim: u32, u: f32, v: f32, fromPos: vec3f) -> SurfaceHit {
  var s = scene_surface(prim, u, v, vec3f(0.0));
  if (dot(s.ng, fromPos - s.pos) < 0.0) {
    s.ng = -s.ng;
    s.ns = -s.ns;
    s.backfacing = true;
  }
  return s;
}

fn rc_vertex(s: SurfaceHit, m: MatEval) -> RcVertex {
  return RcVertex(s.pos, s.ng, RCK_SURFACE, select(0u, 1u, (m.flags & MATEVAL_DIFFUSE_ONLY) != 0u));
}
fn rc_event_bsdf(m: MatEval, lobe: u32, isDelta: bool, pMarg: f32) -> RcEvent {
  return RcEvent(lobe, select(0u, 1u, isDelta), select(lobe_roughness(m, lobe), 0.0, isDelta), pMarg);
}
fn rc_event_nee(m: MatEval, pMarg: f32) -> RcEvent { return RcEvent(LOBE_NEE, 0u, lobe_roughness(m, LOBE_NEE), pMarg); }
fn rc_event_none() -> RcEvent { return RcEvent(LOBE_NONE, 0u, FLT_MAX, 0.0); }

/// STUB (P0): every pair fails. WP-B implements gap-rc §3.6 (Enhanced) and D19 (2022 mode, RSF_CRIT_2022).
fn rcPairTest(a: RcVertex, ea: RcEvent, b: RcVertex, eb: RcEvent, thr: f32) -> RcResult {
  return RcResult(false, 0.0, RCT_NONE);
}

/// thr = τ·R²_pri, R²_pri = ‖x₁ − x₀‖²·4π / max(|⟨n^g_{x₁}, normalize(x₀ − x₁)⟩|, 1e-6) (math.md#rc-predicate).
fn primaryThreshold(camPos: vec3f, x1: vec3f, ng1: vec3f, tau: f32) -> f32 {
  let d = camPos - x1;
  let dist2 = dot(d, d);
  return tau * dist2 * 4.0 * PI / max(abs(dot(ng1, normalize(d))), 1e-6);
}

/// Geometry term toward the receiving vertex b: |n_b·(a − b)| / ‖a − b‖³.
fn rc_G(a: vec3f, b: vec3f, ngB: vec3f) -> f32 {
  let d = a - b;
  let t2 = dot(d, d);
  return abs(dot(ngB, d)) / (t2 * sqrt(t2));
}

/// Deferred k* of an NEE candidate at x_B (d = B+1): the tree's rc if set; else B if B ≥ 2 and the pair
/// (x_{B−1}, x_B | EV_NEE) passes (case b); else B+1 (forced). Returns (k, bits(margin of the deciding pair)).
fn kstar_nee(treeRc: u32, B: u32, prevV: RcVertex, prevE: RcEvent, curV: RcVertex, eNee: RcEvent, thr: f32) -> vec2u {
  if (treeRc != 0u) { return vec2u(treeRc, bitcast<u32>(0.0)); }
  if (B >= 2u) {
    let r = rcPairTest(prevV, prevE, curV, eNee, thr);
    if (r.ok) { return vec2u(B, bitcast<u32>(r.margin)); }
  }
  return vec2u(B + 1u, bitcast<u32>(0.0));
}

/// Shared prefix pair B = (x_{B−1}, x_B), EV_BSDF at x_B.
fn kstar_tree_pair(prevV: RcVertex, prevE: RcEvent, curV: RcVertex, eB: RcEvent, thr: f32) -> RcResult {
  return rcPairTest(prevV, prevE, curV, eB, thr);
}

/// Deferred k* of a BSDF ending at x_{B+1}: the tree's rc if set; else B+1 if the terminal pair (x_B, emitter | env)
/// passes (case d / e); else 0 (∅).
fn kstar_bsdf_end(treeRc: u32, B: u32, curV: RcVertex, eB: RcEvent, endV: RcVertex, thr: f32) -> vec2u {
  if (treeRc != 0u) { return vec2u(treeRc, bitcast<u32>(0.0)); }
  let r = rcPairTest(curV, eB, endV, rc_event_none(), thr);
  if (r.ok) { return vec2u(B + 1u, bitcast<u32>(r.margin)); }
  return vec2u(0u, bitcast<u32>(0.0));
}

// Reconnection predicate and vertex reconstruction (restir-api.md §3.4; math.md#rc-predicate, gap-rc §3.3–§3.6, §8.3).
// OWNER WP-B. The ONLY place the pair predicate P_k = D ∧ R ∧ F ∧ I, the primary threshold and the k* drivers exist
// (same-formula principle, gap-rc §8.3): the path tree (deferred k*), replay (O1/O3) and the shift (O1/O2) all call
// rcPairTest with inputs rebuilt from ids (vertex_from_ids), position-derived directions (D3), the stored per-pixel
// thr bits and the runtime uniforms τ (via thr) and α_min. Decisions are integer compares on the bit patterns of
// non-negative finite floats (rs_pos_finite / rs_geq_pos, types.wgsl), comparison operator ≥ everywhere.
//
// Margin (debug, T3 violation classes, slot code bits 16–31): min over the evaluated sub-tests of log2(value/thr)
// (F, I; 2022: log2(t/d_min)) or α − α_min (R). A DISCRETE failure (delta event, G_T lobe, failed guard) has
// margin RC_MARGIN_DISCRETE = −1024: it can never be an FP-boundary flip (gap-rc §10.3).
#include "restir/types.wgsl"
#include "restir/frame.wgsl"
#include "material/material-eval.wgsl"

const RC_MARGIN_DISCRETE: f32 = -1024.0;
const RC_MARGIN_MAX: f32 = 1024.0;

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

/// log2(num/den) for non-negative finite num, den, clamped to ±RC_MARGIN_MAX (den = 0 ⇒ +max, num = 0 ⇒ −max).
fn rc_log2_ratio(num: f32, den: f32) -> f32 {
  if (!rs_pos_finite(den)) { return RC_MARGIN_MAX; }
  if (!rs_pos_finite(num)) { return -RC_MARGIN_MAX; }
  return clamp(log2(num) - log2(den), -RC_MARGIN_MAX, RC_MARGIN_MAX);
}

/// Pair predicate P_k for (a, b) = (x_{k−1}, x_k) with the events leaving a and b (gap-rc §3.3 verbatim; math.md
/// #rc-predicate). Enhanced (default) or the 2022 criteria (RSF_CRIT_2022, D19). thr = τ·R²_pri of the domain.
///   D  e_a not delta (v1: ℓ_{k−1} = G_T never passes, PLAN rule 5)
///   R  r(ℓ_{k−1}) ≥ α_min (perceptual; D = 1, delta = 0)
///   F  kind(b) = ENV ⇒ pass; else t²/(p̄_a·|cos_b|) ≥ thr   (tested as t² ≥ thr·(p̄_a·|cos_b|), same decision)
///   I  skipped for LIGHT / ENV / diffuseOnly(b); else e_b not delta and not G_T, t²/(p̄_b·|cos_a|) ≥ thr
/// Guards t² > 1e-12, p̄ finite > 0, cos finite > 0: a failed guard fails the pair (term RCT_GUARD).
fn rcPairTest(a: RcVertex, ea: RcEvent, b: RcVertex, eb: RcEvent, thr: f32) -> RcResult {
  if ((rsParams.flags & RSF_CRIT_2022) != 0u) { return rc_pair_2022(a, ea, b, eb); }
  var r = RcResult(false, RC_MARGIN_DISCRETE, RCT_D);
  if (ea.delta != 0u || ea.lobe == LOBE_GT) { return r; }                                   // D (at x_{k−1})
  let aMin = rsParams.alphaMin;
  r.margin = clamp(ea.alpha - aMin, -RC_MARGIN_MAX, RC_MARGIN_MAX);
  if (!rs_geq_pos(ea.alpha, aMin)) { r.term = RCT_R; return r; }                            // R
  if (b.kind == RCK_ENV) { r.ok = true; r.term = RCT_NONE; return r; }                     // F = +∞, I skipped
  let dl = b.pos - a.pos;
  let t2 = dot(dl, dl);
  r.term = RCT_GUARD;
  if (!rs_pos_finite(t2) || bitcast<u32>(t2) <= bitcast<u32>(1e-12)) {                     // t² > 1e-12
    r.margin = RC_MARGIN_DISCRETE; return r;
  }
  let t = sqrt(t2);
  let cb = abs(dot(b.ng, dl)) / t;                                                          // cos at the receiving b
  if (!rs_pos_finite(cb) || !rs_pos_finite(ea.pMarg)) { r.margin = RC_MARGIN_DISCRETE; return r; }
  let rhsF = thr * (ea.pMarg * cb);
  r.margin = min(r.margin, rc_log2_ratio(t2, rhsF));
  if (!rs_geq_pos(t2, rhsF)) { r.term = RCT_F; return r; }                                  // F
  if (b.kind == RCK_LIGHT || b.diffuseOnly != 0u) { r.ok = true; r.term = RCT_NONE; return r; }   // fn. 6
  if (eb.delta != 0u || eb.lobe == LOBE_GT) { r.margin = RC_MARGIN_DISCRETE; r.term = RCT_D; return r; }
  let ca = abs(dot(a.ng, dl)) / t;                                                          // cos at a
  if (!rs_pos_finite(ca) || !rs_pos_finite(eb.pMarg)) { r.margin = RC_MARGIN_DISCRETE; return r; }
  let rhsI = thr * (eb.pMarg * ca);
  r.margin = min(r.margin, rc_log2_ratio(t2, rhsI));
  if (!rs_geq_pos(t2, rhsI)) { r.term = RCT_I; return r; }                                  // I
  r.ok = true;
  r.term = RCT_NONE;
  return r;
}

/// 2022 criteria (D19; GRIS §7.5, enhanced-paper §3.1): D ∧ min(r(e_a), r(e_b)) ≥ α_min ∧ (ENV ∨ ‖b − a‖ ≥ d_min),
/// light / env vertices rough (rc_event_none: α = FLT_MAX), r(EV_NEE) = lobe_roughness(m, LOBE_NEE).
fn rc_pair_2022(a: RcVertex, ea: RcEvent, b: RcVertex, eb: RcEvent) -> RcResult {
  var r = RcResult(false, RC_MARGIN_DISCRETE, RCT_D);
  if (ea.delta != 0u || eb.delta != 0u || ea.lobe == LOBE_GT || eb.lobe == LOBE_GT) { return r; }
  let aMin = rsParams.alphaMin;
  let rmin = min(ea.alpha, eb.alpha);
  r.margin = clamp(rmin - aMin, -RC_MARGIN_MAX, RC_MARGIN_MAX);
  if (!rs_geq_pos(rmin, aMin)) { r.term = RCT_R; return r; }
  if (b.kind == RCK_ENV) { r.ok = true; r.term = RCT_NONE; return r; }
  let dl = b.pos - a.pos;
  let t2 = dot(dl, dl);
  let dmin = rsParams.crit2022MinDist;
  let d2 = dmin * dmin;
  if (!rs_pos_finite(t2)) { r.term = RCT_GUARD; return r; }
  r.margin = min(r.margin, 0.5 * rc_log2_ratio(t2, d2));
  if (!rs_geq_pos(t2, d2)) { r.term = RCT_F; return r; }
  r.ok = true;
  r.term = RCT_NONE;
  return r;
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

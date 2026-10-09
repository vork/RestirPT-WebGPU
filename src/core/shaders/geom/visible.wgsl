// The single visibility module (plan §1.3; math.md#visibility): every segment test (NEE shadow, reconnection
// forward/inverse, refresh, T3 re-test) goes through visible(); every direction test through visibleInf().
// Triangles only (analytic lights never occlude), alpha-MASK via traverse.wgsl's alpha_pass hook, glass and
// emissive triangles occlude. Endpoint primitives are excluded (pass BVH_MISS for non-triangle endpoints).
#include "bvh/traverse.wgsl"
#include "geom/offset.wgsl"

// One shared ε (math.md#visibility "t_max = ‖b'−a'‖ − ε"): the segment is traced along the UNNORMALIZED
// d = b' − a' with t_max = 1 − VIS_EPS, i.e. ε is relative to the offset-segment length.
const VIS_EPS: f32 = 1e-4;

// Offset p along ±ng toward the side of direction w (w on the plane counts as the +ng side).
fn vis_offset(p: vec3f, ng: vec3f, w: vec3f) -> vec3f {
  return offset_ray(p, select(-ng, ng, dot(ng, w) >= 0.0));
}

// a, b: endpoints; na, nb: unit geometric normals (0 for point endpoints); primA, primB: endpoint primIds.
fn visible(a: vec3f, na: vec3f, primA: u32, b: vec3f, nb: vec3f, primB: u32) -> bool {
  let w = b - a;
  let a2 = vis_offset(a, na, w);   // a' = offset(a, sign(n_a·ω)·n_a)
  let b2 = vis_offset(b, nb, -w);  // b' = offset(b, sign(n_b·(−ω))·n_b)
  return !trace_any_ex(a2, b2 - a2, 1.0 - VIS_EPS, primA, primB);
}

// Direction test to infinity (sun, env NEE / rc / refresh): t_max = FLT_MAX, never +Inf.
fn visibleInf(a: vec3f, na: vec3f, primA: u32, dir: vec3f) -> bool {
  return !trace_any_ex(vis_offset(a, na, dir), dir, FLT_MAX, primA, BVH_MISS);
}

#if RS_VIS_MERGE || RS_REFRESH_VIS
// perf2 WP-1 shared visibility ray (perf2-plan.md §2 WP-1 step 1; users: RS_VIS_MERGE shift.wgsl, RS_REFRESH_VIS
// refresh.wgsl; RS_NEE_SITE / WP-2b adds `|| RS_NEE_SITE` to this gate when it lands). Each inlined trace_any_ex copy carries its own traversal stack and code, so
// a caller that picks between several visibility tests per lane builds ONE VisRay with selects and traces it once:
//   inf = false: the segment test visible(a, na, primA, b, nb, primB)   (dir ignored)
//   inf = true:  the direction test visibleInf(a, na, primA, dir)       (b, nb, primB ignored)
// The operands are exactly visible() / visibleInf()'s (bitwise: same offsets, same unnormalised d, same t_max, BVH_MISS
// as the far endpoint of a direction test). On inf lanes the finite b' is computed from (b, nb, −dir) and discarded.
// vis_trace(vis_ray(...), primA) == visible(...) / visibleInf(...). A caller with an "occluded without tracing" case keeps
// that case outside (no trace), as shift_finish_vis does for endOcc.
struct VisRay { o: vec3f, d: vec3f, tmax: f32, primB: u32 }
fn vis_ray(a: vec3f, na: vec3f, b: vec3f, nb: vec3f, primB: u32, dir: vec3f, inf: bool) -> VisRay {
  let w = select(b - a, dir, inf);
  let a2 = vis_offset(a, na, w);
  let b2 = vis_offset(b, nb, -w);
  return VisRay(a2, select(b2 - a2, dir, inf), select(1.0 - VIS_EPS, FLT_MAX, inf), select(primB, BVH_MISS, inf));
}
// true = unoccluded (the visible() convention); primA = the origin endpoint's primId (BVH_MISS for non-triangles).
fn vis_trace(r: VisRay, primA: u32) -> bool { return !trace_any_ex(r.o, r.d, r.tmax, primA, r.primB); }
#endif

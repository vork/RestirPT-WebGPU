// BVH2 traversal, Aila–Laine layout (plan §1.3; layout contract in src/core/bvh/layout.ts). math.md#visibility
//
// API
//   fn trace_closest(o, d, tmax) -> Hit            closest triangle hit with t in (0, tmax); miss: primId = BVH_MISS, t = tmax
//   fn trace_any(o, d, tmax) -> bool                true = occluded (any accepted hit in (0, tmax))
//   fn trace_closest_ex / trace_any_ex(o, d, tmax, skipA, skipB)   same, ignoring hits on primIds skipA / skipB
//   fn bvh_stats() -> vec4u   (steps, boxTests, triTests, flags); flags bit0 = stack overflow, bit1 = iteration
//                             cap, bits 8..15 = max stack depth. Counters are private (per invocation) and
//                             accumulate across calls until bvh_stats_reset(); steps/tests/depth need BVH_STATS.
//   d need not be normalized (t is in units of |d|). Barycentrics: (1−u−v, u, v) weight (v0, v1, v2) of the
//   ORIGINAL index order; primId indexes SceneGeometry.indices/3.
//
// Bindings: the includer declares
//   var<storage, read> bvh_nodes: array<vec4f>;   var<storage, read> bvh_tris: array<vec4f>;
// (BVH_CWBVH: bvh_nodes: array<vec4u>)
// (bvh_tris = the MT layout, or the Woop layout + primId tail with WATERTIGHT; bind the whole buffer),
// or sets BVH_DECLARE_BINDINGS with BVH_GROUP, BVH_BINDING_NODES, BVH_BINDING_TRIS.
//
// Defines: WATERTIGHT (Woop et al. 2013, canonical edge order), BVH_STATS, CUSTOM_ALPHA (the includer then
// provides fn alpha_pass(primId: u32, u: f32, v: f32) -> bool; true = the hit counts. MASK cutout, plan §1.3),
// BVH_CWBVH (M8: bvh_nodes / bvh_tris hold the CWBVH of src/core/bvh/cwbvh.ts; bvh/traverse-cwbvh.wgsl replaces
// bvh_trace, everything else here is shared).
#include "common/math.wgsl"
#include "geom/intersect.wgsl"

#if BVH_DECLARE_BINDINGS
#if BVH_CWBVH
// CWBVH node words are arbitrary bit patterns (quantized bytes, imask): read as u32, never through f32 (NaN payloads)
@group($BVH_GROUP) @binding($BVH_BINDING_NODES) var<storage, read> bvh_nodes: array<vec4u>;
#else
@group($BVH_GROUP) @binding($BVH_BINDING_NODES) var<storage, read> bvh_nodes: array<vec4f>;
#endif
@group($BVH_GROUP) @binding($BVH_BINDING_TRIS) var<storage, read> bvh_tris: array<vec4f>;
#endif

struct Hit { t: f32, u: f32, v: f32, primId: u32 }

const BVH_MISS: u32 = 0xffffffffu;
const BVH_STACK_SIZE: u32 = 32u;
const BVH_ITER_CAP: u32 = 65536u;
const BVH_FLAG_OVERFLOW: u32 = 1u;
const BVH_FLAG_ITERCAP: u32 = 2u;
const BVH_LEAF_BIT: u32 = 0x80000000u;
// Ize 2013 (JCGT 2(2)) robust slab test: far distances scaled by 1 + 2·γ3, γn = nε/(1−nε), ε = 2^-24.
const BVH_ROBUST_FAR: f32 = 1.0000003576279;

var<private> bvh_flags: u32;
#if BVH_STATS
var<private> bvh_st_steps: u32;
var<private> bvh_st_box: u32;
var<private> bvh_st_tri: u32;
#endif

#ifndef CUSTOM_ALPHA
fn alpha_pass(primId: u32, u: f32, v: f32) -> bool { return true; }
#endif

fn bvh_stats() -> vec4u {
#if BVH_STATS
  return vec4u(bvh_st_steps, bvh_st_box, bvh_st_tri, bvh_flags);
#else
  return vec4u(0u, 0u, 0u, bvh_flags);
#endif
}

fn bvh_stats_reset() {
  bvh_flags = 0u;
#if BVH_STATS
  bvh_st_steps = 0u; bvh_st_box = 0u; bvh_st_tri = 0u;
#endif
}

// Reciprocal that never produces Inf (|x| < 1e-30 → ±1e30, sign of x with −0 → +): avoids 0·∞ = NaN in the
// slab test when the origin lies on a slab plane; scene extents stay ≪ 1e8 so products stay finite.
fn bvh_safe_rcp(x: f32) -> f32 {
  return 1.0 / select(x, select(-1e-30, 1e-30, x >= 0.0), abs(x) < 1e-30);
}

// Entry distance of the ray into [bmin, bmax] if it overlaps [0, tlim], else -1.
fn bvh_slab(bmin: vec3f, bmax: vec3f, o: vec3f, rd: vec3f, tlim: f32) -> f32 {
  let t0 = (bmin - o) * rd;
  let t1 = (bmax - o) * rd;
  let tn = min(t0, t1);
  let tf = max(t0, t1);
  let tnear = max(max(tn.x, tn.y), max(tn.z, 0.0));
  let tfar = min(min(min(tf.x, tf.y), tf.z) * BVH_ROBUST_FAR, tlim);
  return select(-1.0, tnear, tnear <= tfar);
}

#if BVH_CWBVH
#include "bvh/traverse-cwbvh.wgsl"
#else
#if BVH2_PRIV_STACK
// perf2 WP-3d: the traversal stack at module scope (BVH2 only: on CWBVH this measured +10 ms, perf2-plan.md §1)
var<private> bvh_pstack: array<u32, 32>;
#endif
fn bvh_trace(o: vec3f, d: vec3f, tmax: f32, any_hit: bool, skipA: u32, skipB: u32) -> Hit {
  var hit = Hit(tmax, 0.0, 0.0, BVH_MISS);
  let rd = vec3f(bvh_safe_rcp(d.x), bvh_safe_rcp(d.y), bvh_safe_rcp(d.z));
#if WATERTIGHT
  let wr = woop_setup(d);
  let nTris4 = arrayLength(&bvh_tris);
#endif
#if !BVH2_PRIV_STACK
  var stack: array<u32, 32>;
#endif
  var sp = 0u;
  var node = 0u; // child ref: interior index, or leaf (LEAF_BIT set)
#if BVH_CONST_LOOPS
  // perf2 WP-3c: constant loop bounds (no Tint loop guards); iteration iter + 1 of the old loop, same cap and flag
  for (var iter = 0u; iter <= BVH_ITER_CAP; iter += 1u) {
    if (iter == BVH_ITER_CAP) { bvh_flags |= BVH_FLAG_ITERCAP; break; }
#else
  var iter = 0u;
  loop {
    iter += 1u;
    if (iter > BVH_ITER_CAP) { bvh_flags |= BVH_FLAG_ITERCAP; break; }
#endif
#if BVH_STATS
    bvh_st_steps += 1u;
#endif
    if ((node & BVH_LEAF_BIT) != 0u) {
      let cnt = (node >> 24u) & 127u;
      let first = node & 0xffffffu;
#if BVH_CONST_LOOPS
      for (var li = 0u; li < 127u; li += 1u) {
        if (li >= cnt) { break; }
        let i = first + li;
#else
      for (var i = first; i < first + cnt; i += 1u) {
#endif
#if BVH_STATS
        bvh_st_tri += 1u;
#endif
        let base = 3u * i;
#if WATERTIGHT
        let p0 = bvh_tris[base];
        let p1 = bvh_tris[base + 1u];
        let p2 = bvh_tris[base + 2u];
        let r = isect_woop(o, wr, p0.xyz, p1.xyz, p2.xyz, vec3u(bitcast<u32>(p0.w), bitcast<u32>(p1.w), bitcast<u32>(p2.w)), hit.t);
        if (!r.ok) { continue; }
        let prim = bitcast<u32>(bvh_tris[nTris4 - 1u - (i >> 2u)][i & 3u]); // primId tail, read from the end
#elif BVH_ALPHA_BIT
        let v0 = bvh_tris[base];
        let e1 = bvh_tris[base + 1u];   // e1.w: the MASK bit (bit 0) written by scene-gpu.ts (perf2 WP-3b)
        let r = isect_mt(o, d, v0.xyz, e1.xyz, bvh_tris[base + 2u].xyz, hit.t);
        if (!r.ok) { continue; }
        let prim = bitcast<u32>(v0.w);
#else
        let v0 = bvh_tris[base];
        let r = isect_mt(o, d, v0.xyz, bvh_tris[base + 1u].xyz, bvh_tris[base + 2u].xyz, hit.t);
        if (!r.ok) { continue; }
        let prim = bitcast<u32>(v0.w);
#endif
        if (prim == skipA || prim == skipB || prim == BVH_MISS) { continue; }
#if BVH_ALPHA_BIT && BVH_NO_ALPHA
        // perf2 WP-3b: no MASK triangle in the scene (scene-gpu.ts): alpha_pass is constant true
#elif BVH_ALPHA_BIT && !WATERTIGHT
        if ((bitcast<u32>(e1.w) & 1u) != 0u && !alpha_pass(prim, r.u, r.v)) { continue; }
#else
        if (!alpha_pass(prim, r.u, r.v)) { continue; }
#endif
#if GLASS_PLANT == 5
        if (any_hit && glass_plant_transparent(prim)) { continue; }   // Gate-1 plant B-shadow (validation only)
#endif
        hit = Hit(r.t, r.u, r.v, prim);
        if (any_hit) { return hit; }
      }
    } else {
      let nb = 4u * node;
      let n0 = bvh_nodes[nb];
      let n1 = bvh_nodes[nb + 1u];
      let n2 = bvh_nodes[nb + 2u];
      let n3 = bvh_nodes[nb + 3u];
#if BVH_STATS
      bvh_st_box += 2u;
#endif
      let tl = bvh_slab(n0.xyz, n1.xyz, o, rd, hit.t);
      let tr = bvh_slab(n2.xyz, n3.xyz, o, rd, hit.t);
      let left = bitcast<u32>(n0.w);
      let right = bitcast<u32>(n1.w);
      let hl = tl >= 0.0;
      let hr = tr >= 0.0;
      if (hl && hr) {
        let leftFirst = tl <= tr;
        node = select(right, left, leftFirst);
        if (sp < BVH_STACK_SIZE) {
#if BVH2_PRIV_STACK
          bvh_pstack[sp] = select(left, right, leftFirst);
#else
          stack[sp] = select(left, right, leftFirst);
#endif
          sp += 1u;
#if BVH_STATS
          bvh_flags = max(bvh_flags, (bvh_flags & 0xffu) | (sp << 8u));
#endif
        } else {
          bvh_flags |= BVH_FLAG_OVERFLOW;
        }
        continue;
      }
      if (hl) { node = left; continue; }
      if (hr) { node = right; continue; }
    }
    if (sp == 0u) { break; }
    sp -= 1u;
#if BVH2_PRIV_STACK
    node = bvh_pstack[sp];
#else
    node = stack[sp];
#endif
  }
  return hit;
}
#endif

fn trace_closest_ex(o: vec3f, d: vec3f, tmax: f32, skipA: u32, skipB: u32) -> Hit {
  return bvh_trace(o, d, tmax, false, skipA, skipB);
}
fn trace_any_ex(o: vec3f, d: vec3f, tmax: f32, skipA: u32, skipB: u32) -> bool {
  return bvh_trace(o, d, tmax, true, skipA, skipB).primId != BVH_MISS;
}
fn trace_closest(o: vec3f, d: vec3f, tmax: f32) -> Hit { return bvh_trace(o, d, tmax, false, BVH_MISS, BVH_MISS); }
fn trace_any(o: vec3f, d: vec3f, tmax: f32) -> bool { return bvh_trace(o, d, tmax, true, BVH_MISS, BVH_MISS).primId != BVH_MISS; }

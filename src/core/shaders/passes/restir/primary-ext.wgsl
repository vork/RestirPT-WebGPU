// perf2 WP-7e (RS_PRIMARY_EXT; perf2-plan.md §2 WP-7, docs/decisions/perf2-api.md §1): rs_primary without its own
// camera ray trace. The jittered primary hit of the atlas pixel comes from a V-buffer texel
//   rsVsrc = (primId | BVH_MISS, bits(u), bits(v), bits(t))
// written by the renderer's M1 primary pass (passes/primary.wgsl with RS_PRIMARY_EXT: the same camera ray, seed and jitter
// for member 0) or, where no M1 V-buffer exists (validation kernels, tests), by rs_vtrace below. Everything else is
// rs_primary's text (passes/restir/primary.wgsl): rsVbuf / rsGeo / rsL1 from the same hit, in the same order. The only
// difference to rs_primary is the hit itself: a separately compiled intersection may move the barycentrics by ulps and
// flip exact edge ties (unbiased: either is the closest hit of the same ray to f32 precision).
// The BVH flags of the primary trace are counted by the trace's own pass (M1: DBGC_BVH_*; rs_vtrace: RSC_BVH_*).
//   rs_primary_ext  G2: 0 rsVbuf (st w) · 1 rsGeo (st w) · 2 rsL1 (st w) · 3 shiftArena rw · 4 rsVsrc (texture_2d<u32>)
//   rs_vtrace       G2: 0 rsVsrcOut (st w) · 1 shiftArena rw
#include "restir/frame.wgsl"
#include "restir/queue.wgsl"
#include "scene/scene-data.wgsl"
#include "bvh/traverse.wgsl"
#include "lights/env.wgsl"
#include "path/length1.wgsl"
#include "restir/rc.wgsl"

#if RS_VSRC_BINDING
@group(2) @binding($RS_VSRC_BINDING) var rsVsrc: texture_2d<u32>;

@compute @workgroup_size(8, 8, 1)
fn rs_primary_ext(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
  let key = rs_frame_key(p.member, rs_t(), p.localIdx);
  let ray = frame_camera_ray(p.local, rs_jitter(key, p));
  let src = textureLoad(rsVsrc, p.px, 0);
  let primId = src.x;
  let hu = bitcast<f32>(src.y);
  let hv = bitcast<f32>(src.z);
  var L = l1_camera_lights(ray.o, ray.d, select(FLT_MAX, bitcast<f32>(src.w), primId != BVH_MISS));
  var vb = vec4u(BVH_MISS, 0u, 0u, 0u);
  var geo = vec4f(0.0);
  if (primId == BVH_MISS) {
    L += envBackground(ray.d);
  } else {
    let camPos = rs_cam_pos();
    let s = vertex_from_ids(primId, hu, hv, camPos);
    L += tri_emission(primId, hu, hv);
    vb = vec4u(primId, src.y, src.z, bitcast<u32>(length(s.pos - camPos)));
    geo = vec4f(s.ng, primaryThreshold(camPos, s.pos, s.ng, rsParams.tau));
  }
  textureStore(rsVbufOut, p.px, vb);
  textureStore(rsGeoOut, p.px, geo);
  textureStore(rsL1Out, p.px, vec4f(L, 0.0));
}
#endif

#if RS_VSRC_W_BINDING
@group(2) @binding($RS_VSRC_W_BINDING) var rsVsrcOut: texture_storage_2d<rgba32uint, write>;

/// The M1 primary's V-buffer texel for kernels without an M1 pass: rs_primary's camera ray and trace.
@compute @workgroup_size(8, 8, 1)
fn rs_vtrace(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
  let key = rs_frame_key(p.member, rs_t(), p.localIdx);
  let ray = frame_camera_ray(p.local, rs_jitter(key, p));
  bvh_stats_reset();
  let hit = trace_closest(ray.o, ray.d, FLT_MAX);
  var vb = vec4u(BVH_MISS, 0u, 0u, 0u);
  if (hit.primId != BVH_MISS) { vb = vec4u(hit.primId, bitcast<u32>(hit.u), bitcast<u32>(hit.v), bitcast<u32>(hit.t)); }
  let fl = bvh_stats().w;
  if ((fl & BVH_FLAG_OVERFLOW) != 0u) { rs_count(RSC_BVH_OVERFLOW, 1u); }
  if ((fl & BVH_FLAG_ITERCAP) != 0u) { rs_count(RSC_BVH_ITERCAP, 1u); }
  textureStore(rsVsrcOut, p.px, vb);
}
#endif

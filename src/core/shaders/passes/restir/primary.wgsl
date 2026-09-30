// rs_primary (restir-api.md §4.1 step 1, D12; math.md#raster, #path-tree "Length-1 terms", #rc-predicate thr):
// one jittered camera ray per atlas pixel (member-local raster, the PT's seed and jitter), closest triangle x₁.
//   rsVbuf = (primId | BVH_MISS, bits(u), bits(v), bits(‖x₁ − x₀‖))       rsGeo = (n^g of x₁ toward the camera, thr)
//   rsL1   = length-1 radiance: camera-visible analytic area lights + L_e(x₁) | visibleToCamera·L_env on a miss
// (evaluated in the PT's order, so rsL1 is bitwise the PT's length-1 part for the same seed). BVH flags → arena
// counters RSC_BVH_*. G2: 0 rsVbuf (st w) · 1 rsGeo (st w) · 2 rsL1 (st w) · 3 shiftArena rw.
#include "restir/frame.wgsl"
#include "restir/queue.wgsl"
#include "scene/scene-data.wgsl"
#include "bvh/traverse.wgsl"
#include "lights/env.wgsl"
#include "path/length1.wgsl"
#include "restir/rc.wgsl"

@compute @workgroup_size(8, 8, 1)
fn rs_primary(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
  let key = rs_frame_key(p.member, rs_t(), p.localIdx);
  let ray = frame_camera_ray(p.local, rs_jitter(key, p));
  bvh_stats_reset();
  let hit = trace_closest(ray.o, ray.d, FLT_MAX);
  var L = l1_camera_lights(ray.o, ray.d, select(FLT_MAX, hit.t, hit.primId != BVH_MISS));
  var vb = vec4u(BVH_MISS, 0u, 0u, 0u);
  var geo = vec4f(0.0);
  if (hit.primId == BVH_MISS) {
    L += envBackground(ray.d);
  } else {
    let camPos = rs_cam_pos();
    let s = vertex_from_ids(hit.primId, hit.u, hit.v, camPos);
    L += tri_emission(hit.primId, hit.u, hit.v);
    vb = vec4u(hit.primId, bitcast<u32>(hit.u), bitcast<u32>(hit.v), bitcast<u32>(length(s.pos - camPos)));
    geo = vec4f(s.ng, primaryThreshold(camPos, s.pos, s.ng, rsParams.tau));
  }
  let fl = bvh_stats().w;
  if ((fl & BVH_FLAG_OVERFLOW) != 0u) { rs_count(RSC_BVH_OVERFLOW, 1u); }
  if ((fl & BVH_FLAG_ITERCAP) != 0u) { rs_count(RSC_BVH_ITERCAP, 1u); }
  textureStore(rsVbufOut, p.px, vb);
  textureStore(rsGeoOut, p.px, geo);
  textureStore(rsL1Out, p.px, vec4f(L, 0.0));
}

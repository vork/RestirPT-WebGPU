// Length-1 terms (math.md#path-tree "Length-1 terms", plan §3): radiance that reaches the camera without a scattering
// vertex. Outside every reservoir, weight 1, shared by the reference PT (passes/pt.wgsl) and ReSTIR rs_primary:
//   Σ camera-visible analytic area lights (rect/disk, LF_VISIBLE_CAMERA) crossed front-facing before x₁ (or before
//     FLT_MAX on a miss)                                          l1_camera_lights
//   + L_e of an emissive x₁ (tri_emission)                        (caller)
//   + on a miss visibleToCamera·L_env(d_cam) (envBackground)      (caller)
// Moved verbatim from pt.wgsl's pt_camera_lights (restir-api.md §1.2 WP-A; U-PT-BITS proves the PT bit-identical).
#include "lights/measure.wgsl"

/// Σ radiance of camera-visible analytic area lights crossed by the camera ray before tMax (length-1, weight 1).
fn l1_camera_lights(o: vec3f, d: vec3f, tMax: f32) -> vec3f {
  var L = vec3f(0.0);
  let slot = lightsParams.cur;
  for (var i = 0u; i < slot.lightCount; i++) {
    let r = light_load(slot, i);
    if ((r.flags & LF_VISIBLE_CAMERA) != 0u && (r.kind == LT_RECT || r.kind == LT_DISK)) { L += area_light_crossing(r, o, d, tMax); }
  }
  return L;
}

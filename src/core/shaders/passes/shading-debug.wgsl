// M7 shading-normal debug views (plan §6; docs/decisions/m7-api.md §7), drawn through the debug AOV plane (group 3,
// debug_write) so the resolve pass maps them to false colour. Host: src/core/render/shading-debug.ts. One camera ray
// through the pixel centre, the production scene_surface (so the views show exactly what the BSDF receives):
//   320 SH_N         closure normal N (normal-mapped where the material has a normal texture, else = Ns)
//   321 SH_NS        unmapped shading normal Ns (interpolated smooth normal; Ng on flat faces)
//   322 SH_T         MikkTSpace tangent T (interpolated, normalised; black without tangents / NORMAL_MAP)
//   323 SH_B         bitangent B = sign·(Ns_u × T) (normalised), the tangent-space +y of the normal map
//   324 SH_NM        normal-map decode c = 2(rgb − ½) with the Cycles strength applied (xy·s, z = mix(1, z, s))
//   325 SH_NM_ANGLE  angle(N, Ns) in degrees
//   326 SH_NS_ANGLE  angle(Ns, Ng) in degrees (smooth-shading deviation; 0 on flat faces)
//   327 SH_SIGN      bitangent sign code: 1 = +1, 2 = −1, none = no tangent / no normal map
// Bindings: G0 frame; G1 scene; G3 debug.
#include "common/frame.wgsl"
#include "common/nan.wgsl"
#include "scene/scene-data.wgsl"
#include "bvh/traverse.wgsl"
#include "debug/debug-common.wgsl"

const DBG_SH_N: u32 = 320u;
const DBG_SH_NS: u32 = 321u;
const DBG_SH_T: u32 = 322u;
const DBG_SH_B: u32 = 323u;
const DBG_SH_NM: u32 = 324u;
const DBG_SH_NM_ANGLE: u32 = 325u;
const DBG_SH_NS_ANGLE: u32 = 326u;
const DBG_SH_SIGN: u32 = 327u;

fn sh_angle_deg(a: vec3f, b: vec3f) -> f32 { return degrees(atan2(length(cross(a, b)), dot(a, b))); }

@compute @workgroup_size(8, 8, 1)
fn shading_view(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = gid.xy;
  if (any(pixel >= frame.resolution)) { return; }
  let mode = dbg.mode;
  let ray = frame_camera_ray(pixel, vec2f(0.5));
  let hit = trace_closest(ray.o, ray.d, FLT_MAX);
  if (hit.primId == BVH_MISS) {
    if (mode == DBG_SH_SIGN) { debug_write_code(pixel, mode, DBG_CODE_NONE); } else { debug_write(pixel, mode, vec4f(0.0)); }
    return;
  }
  let s = scene_surface(hit.primId, hit.u, hit.v, ray.d);
#if NORMAL_MAP
  let nsm = s.nsm;
#else
  let nsm = s.ns;
#endif
  var T = vec3f(0.0);
  var B = vec3f(0.0);
  var c = vec3f(0.0, 0.0, 1.0);
  var sign = DBG_CODE_NONE;
#if NORMAL_MAP
  let t = sceneTris[hit.primId];
  let w = 1.0 - hit.u - hit.v;
  let tg = w * scene_vertex_tangent(t.x) + hit.u * scene_vertex_tangent(t.y) + hit.v * scene_vertex_tangent(t.z);
  let nU = select(nsm, -nsm, s.backfacing);           // winding orientation (the Normal Map node's frame)
  if (dot(tg.xyz, tg.xyz) > 0.0) {
    T = normalize(tg.xyz);
    B = normalize(tg.w * cross(nU, tg.xyz));
    sign = select(2u, 1u, tg.w >= 0.0);
  }
  let mat = sceneMaterials[s.matId];
  if (tex_slot_valid(mat.texNormal)) {
    c = 2.0 * (tex_sample(mat.texNormal, s.uv).rgb - vec3f(0.5));
    c.x *= mat.normalScale;
    c.y *= mat.normalScale;
    c.z = mix(1.0, c.z, saturate(mat.normalScale));
  }
#endif
  switch mode {
    case DBG_SH_N: { debug_write3(pixel, mode, s.ns); }
    case DBG_SH_NS: { debug_write3(pixel, mode, nsm); }
    case DBG_SH_T: { debug_write3(pixel, mode, T); }
    case DBG_SH_B: { debug_write3(pixel, mode, B); }
    case DBG_SH_NM: { debug_write3(pixel, mode, c); }
    case DBG_SH_NM_ANGLE: { debug_write1(pixel, mode, sh_angle_deg(s.ns, nsm)); }
    case DBG_SH_NS_ANGLE: { debug_write1(pixel, mode, sh_angle_deg(nsm, s.ng)); }
    case DBG_SH_SIGN: { debug_write_code(pixel, mode, sign); }
    default: { }
  }
}

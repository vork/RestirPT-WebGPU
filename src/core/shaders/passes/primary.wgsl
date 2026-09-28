// Primary pass (plan §3 step 1, §5 M1). One jittered camera ray per pixel, 8×8 workgroups.
// - Camera ray and jitter: math.md#raster (frame.wgsl frame_camera_ray / frame_pixel_jitter; JITTER_IID =
//   rand2(hash(runSeed, seedIndex, pixel), 0, STREAM_JITTER), JITTER_NONE = pixel centre).
// - Closest TRIANGLE hit through the BVH with the alpha-MASK cutout (scene-data.wgsl alpha_pass); t_max = FLT_MAX,
//   never Inf (math.md#visibility). Analytic lights are not in the BVH.
// - Outputs: V-buffer (primId, f32 barycentrics), linear view depth, oriented Ng/Ns, material id, albedo AOV,
//   motion vectors and thr (math.md#rc-predicate) into the G-buffer. A miss writes primId = BVH_MISS and the
//   camera-miss env term visibleToCamera·L_env (env.wgsl envBackground; math.md#path-tree length-1 terms).
// - M1 placeholder beauty: the progressive mean of (albedo on hits | env background on misses). M3a replaces it.
// Defines: COLOR_FORMAT, SCENE_GROUP (=1), BVH_* / TEX_* / ENV_* (renderer.ts), CUSTOM_ALPHA, WATERTIGHT, BVH_STATS.
#include "common/frame.wgsl"
#include "debug/debug-common.wgsl"
#include "scene/scene-data.wgsl"
#include "bvh/traverse.wgsl"
#include "debug/env-grid.wgsl"
#include "passes/gbuffer.wgsl"
#include "debug/gbuffer-views.wgsl"

struct PrimaryParams {
  flags: u32,     // PRIM_* bits
  thrTau: f32,    // τ in thr = τ·R²_pri (math.md#rc-predicate: 2·10⁻⁴)
  pad0: u32,
  pad1: u32,
}
const PRIM_ACCUMULATE: u32 = 1u;   // progressive mean on; else the output is this frame's sample
const PRIM_ADVANCED: u32 = 2u;     // the frame advanced (not paused): add this frame's sample to the mean

@group(0) @binding(4) var<uniform> prim: PrimaryParams;

@group(2) @binding(0) var colorOut: texture_storage_2d<$COLOR_FORMAT, write>;
@group(2) @binding(1) var depthOut: texture_storage_2d<r32float, write>;
@group(2) @binding(2) var vbufOut: texture_storage_2d<rgba32uint, write>;
@group(2) @binding(3) var<storage, read_write> gbuf: array<GBufTexel>;
@group(2) @binding(4) var<storage, read_write> accum: array<vec4f>;   // rgb sum, w = sample count

const PROBE_TAG_HIT: u32 = 16u;     // (t, u, v, bits(primId))
const PROBE_TAG_POS: u32 = 17u;     // (pos.xyz internal, viewZ)
const PROBE_TAG_NS: u32 = 18u;      // (ns.xyz, bits(matId))
const PROBE_TAG_BVH: u32 = 19u;     // (steps, box, tri, bits(flags))
const PROBE_TAG_COLOR: u32 = 20u;   // (sample rgb, count)

// Image position of `p` (continuous, row 0 = top) under `cam`, plus validity (in front of the camera).
// w = 1: a point (frame.wgsl frame_project); w = 0: a direction at infinity (background, rotation only).
fn project_h(p: vec3f, w: f32, cam: CameraFrame) -> vec3f {
  let pc = (cam.worldToCam * vec4f(p, w)).xyz;
  if (!(pc.z < -1e-12)) { return vec3f(0.0); }
  let ndc = vec2f(pc.x / (-pc.z * cam.tanHalfY * cam.aspect), pc.y / (-pc.z * cam.tanHalfY));
  let res = vec2f(frame.resolution);
  return vec3f((ndc.x + 1.0) * 0.5 * res.x, res.y - (ndc.y + 1.0) * 0.5 * res.y, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn primary(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = gid.xy;
  if (any(pixel >= frame.resolution)) { return; }
  let idx = pixel.y * frame.resolution.x + pixel.x;

  let jit = frame_pixel_jitter(pixel);
  let ray = frame_camera_ray(pixel, jit);
  bvh_stats_reset();
  let hit = trace_closest(ray.o, ray.d, FLT_MAX);
  let stats = bvh_stats();
  if ((stats.w & BVH_FLAG_OVERFLOW) != 0u) { debug_count(DBGC_BVH_OVERFLOW, 1u); }
  if ((stats.w & BVH_FLAG_ITERCAP) != 0u) { debug_count(DBGC_BVH_ITERCAP, 1u); }

  var g: GBufTexel;   // zero-initialised
  var color: vec3f;
  var vb = vec4u(BVH_MISS, 0u, 0u, DBG_CODE_NONE);
  var uv = vec2f(0.0);
  var bary = vec3f(0.0);
  let isHit = hit.primId != BVH_MISS;
  g.matId = DBG_CODE_NONE;
  if (isHit) {
    let s = scene_surface(hit.primId, hit.u, hit.v, ray.d);
    let m = sceneMaterials[s.matId];
    g.ng = s.ng;
    g.ns = s.ns;
    g.pos = s.pos;
    g.matId = s.matId;
    g.albedo = material_albedo(m, s.uv, s.color);
    g.viewZ = frame_view_depth(s.pos, frame.cam);
    let dist2 = dot(s.pos - ray.o, s.pos - ray.o);
    g.thr = prim.thrTau * dist2 * 4.0 * PI / max(abs(dot(s.ng, ray.d)), 1e-6);   // math.md#rc-predicate R²_pri
    let cur = project_h(s.pos, 1.0, frame.cam);
    let prv = project_h(s.pos, 1.0, frame.prevCam);
    g.flags = GB_HIT | select(0u, GB_BACKFACING, s.backfacing) | select(0u, GB_MOTION_VALID, cur.z * prv.z > 0.0);
    g.motion = select(vec2f(0.0), prv.xy - cur.xy, cur.z * prv.z > 0.0);
    color = g.albedo;
    uv = s.uv;
    bary = vec3f(1.0 - hit.u - hit.v, hit.u, hit.v);
    vb = vec4u(hit.primId, bitcast<u32>(hit.u), bitcast<u32>(hit.v), s.matId);
  } else {
    color = envBackground(ray.d);
    let cur = project_h(ray.d, 0.0, frame.cam);
    let prv = project_h(ray.d, 0.0, frame.prevCam);
    g.flags = select(0u, GB_MOTION_VALID, cur.z * prv.z > 0.0);
    g.motion = select(vec2f(0.0), prv.xy - cur.xy, cur.z * prv.z > 0.0);
  }

  // NaN/Inf bit tests (plan §1.1, T15): the beauty sample AND every G-buffer field later passes read are counted
  // (DBGC_NAN / DBGC_INF, one count per offending group); the mask view shows where.
  var chk = 0u;
  if (!debug_check_finite(color)) { chk |= CHK_COLOR; }
  if (!(all_finite3(g.ng) && all_finite3(g.ns) && all_finite3(g.pos) && all_finite3(g.albedo))) {
    chk |= CHK_VECTORS;
    _ = debug_check_finite(g.ng + g.ns + g.pos + g.albedo);   // NaN dominates Inf in the sum; either is counted
  }
  if (!(is_finite(g.viewZ) && is_finite(g.thr) && is_finite(g.motion.x) && is_finite(g.motion.y))) {
    chk |= CHK_SCALARS;
    _ = debug_check_finite(vec3f(g.viewZ + g.thr, g.motion));
  }

  // Progressive mean (placeholder beauty until M3a): restart on history reset / camera motion.
  let restart = (frame.flags & (FRAME_RESET_HISTORY | FRAME_CAMERA_MOVED)) != 0u || (prim.flags & PRIM_ACCUMULATE) == 0u;
  var a = select(accum[idx], vec4f(0.0), restart);
  if (restart || (prim.flags & PRIM_ADVANCED) != 0u) { a += vec4f(color, 1.0); }
  accum[idx] = a;

  gbuf[idx] = g;
  textureStore(vbufOut, pixel, vb);
  textureStore(depthOut, pixel, vec4f(g.viewZ, 0.0, 0.0, 0.0));
  textureStore(colorOut, pixel, vec4f(a.rgb / max(a.w, 1.0), 1.0));

  gbuffer_debug_write(pixel, g, hit.primId, bary, uv, select(0.0, hit.t, isHit), stats, chk);
  if (debug_active(DBG_ENV_GRID)) {
    let bg = color / (vec3f(1.0) + color);   // Reinhard, display-only
    let grid = env_grid_overlay(ray.d, envParams.cg, envParams.sg);
    debug_write3(pixel, DBG_ENV_GRID, select(mix(bg, grid.rgb, grid.a), 0.35 * g.albedo, isHit));
  }
  if (debug_is_probe(pixel)) {
    probe_record(pixel, PROBE_TAG_HIT, vec4f(hit.t, hit.u, hit.v, bitcast<f32>(hit.primId)));
    probe_record(pixel, PROBE_TAG_POS, vec4f(g.pos, g.viewZ));
    probe_record(pixel, PROBE_TAG_NS, vec4f(g.ns, bitcast<f32>(g.matId)));
    probe_record(pixel, PROBE_TAG_BVH, vec4f(f32(stats.x), f32(stats.y), f32(stats.z), bitcast<f32>(stats.w)));
    probe_record(pixel, PROBE_TAG_COLOR, vec4f(color, a.w));
  }
}

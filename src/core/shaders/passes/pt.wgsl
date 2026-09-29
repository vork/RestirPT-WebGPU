// Reference path tracer (plan §3 mode PT, §5 M3a; math.md#bounces, #raster, #rng-layout, #measure, #mis, #path-tree,
// #visibility, #env-mapping, #env-sampling). Mode A (analytic lights NEE-only), no glass (M3b). The env is one entry of
// the global alias table (NEE_ENV) and BSDF escapes use MIS against it (lights/env-sample.wgsl); with env NEE off (no
// entry) escapes have ω2 = 1 (≡ Cycles world sampling_method NONE).
//
// Per sample (pixel p, sample/frame index t):
//   initSeed = pcg3d(runSeed ⊕ member·φ, t, p).xy (64-bit PathSeed, rng-path.wgsl); jitter = rand2(initSeed.x, 0, STREAM_JITTER)
//   camera ray (frame.wgsl frame_camera_ray, pixel ↔ Cycles raster, BOX 1 px) → closest TRIANGLE x₁ (alpha MASK in
//   traversal). Length-1 terms outside the path loop, weight 1: camera-visible analytic area lights crossed before x₁
//   (or before FLT_MAX on a miss), L_e of an emissive x₁, and on a miss visibleToCamera·L_env.
//   Bounce loop, Cycles max_bounces = N semantics: scattering vertices x_B, B = 1 … N+1 (≤ N+1 of them), NEE at each;
//   emission (BSDF hit / escape) collected at x₂ … x_{N+2}; no NEE and no continuation at x_{N+2}.
//   At x_B:  NEE      F = β ⊙ ω1·f_cos ⊙ Λ·V/q     (nee_sample in μ; ω1 ≡ 1 for delta / Mode-A analytic)
//            RR       optional (PT_RR): q = min(sqrt(max_c β_c), 1) on the RR-free β, only for B > rrMinBounces
//            BSDF     bsdf_sample(u_lobe, u_h1, u_h2, u_rt) → β ⊙= weight (joint pdf), trace the closest triangle
//            hit on an emissive triangle: + β ⊙ L_e·ω2, ω2 = p2/(M p1 + p2) with p1 recomputed from x_B (1 after a
//            delta lobe); escape: + β ⊙ L_env·ω2_env, ω2_env = p2/(M p1Env + p2) (1 after a delta lobe) and stop.
//   Analytic lights are never hit by BSDF rays in Mode A and never occlude (math.md#visibility).
// Accumulation in f32 per batch (accum[p] += Σ samples); NaN/Inf samples are counted and dropped (T15).
//
// Entry points: pt_batch (validation batches, G2 = accum + counters, row bands) and, with PT_INTERACTIVE, pt_frame
// (one sample per pixel per frame, progressive mean, writes the colour target).
// Defines: SCENE_GROUP (=1), BVH_* / TEX_* (scene-gpu.ts), ENV_* (env-gpu.ts), LIGHTS_GROUP/LIGHTS_BINDING,
// LUT_* (luts/lut-layout.ts lutDefines, records kind u32; the BSDF LUTs live in `records`), PT_INTERACTIVE + COLOR_FORMAT.
#include "common/frame.wgsl"
#include "common/nan.wgsl"
#include "common/rng-path.wgsl"
#include "scene/scene-data.wgsl"
#include "bvh/traverse.wgsl"
#include "geom/visible.wgsl"
#include "lights/env.wgsl"
#include "lights/measure.wgsl"
#include "lights/env-sample.wgsl"
#include "material/material-eval.wgsl"


struct PtParams {
  sampleBase: u32,     // batch: run-global index of this dispatch's first sample
  sampleCount: u32,    // batch: samples per pixel in this dispatch
  rowBase: u32,        // batch: first image row of this dispatch (row-band sub-dispatches, plan §1.8)
  rowEnd: u32,         // batch: one past the last row
  maxBounces: u32,     // Cycles max_bounces N
  flags: u32,          // PT_*
  rrMinBounces: u32,   // RR only at vertices B > rrMinBounces
  member: u32,         // ensemble member id (0)
  // Gate-1 planted biases (plan §7.3 calibration; validation only, identity by default):
  emitScale: f32,      // every emitter ×emitScale (≡ L×emitScale by linearity; 1 = off) — "light ×1.01"
  dropProb: f32,       // terminate the path with probability dropProb at vertex dropBounce, NO compensation (0 = off)
  dropBounce: u32,     // vertex index B of the drop plant ("drop 1% of paths at bounce 2")
  _pad: u32,
}

const PT_RR: u32 = 1u;           // Russian roulette (off by default; interactive option)
const PT_NEE_ONLY: u32 = 2u;     // T9d estimator: NEE with ω1 = 1, BSDF-hit emission dropped (d ≥ 2)
const PT_BSDF_ONLY: u32 = 4u;    // T9d estimator: no NEE, BSDF-hit emission with ω2 = 1
const PT_ACCUMULATE: u32 = 8u;   // interactive: progressive mean on
const PT_ADVANCED: u32 = 16u;    // interactive: the frame advanced (add this frame's sample)

const SLOT_DBG: u32 = 13u;       // path slot of the drop plant (unused by the estimator itself)

const PT_CNT_NONFINITE: u32 = 0u;
const PT_CNT_BVH_OVERFLOW: u32 = 1u;
const PT_CNT_BVH_ITERCAP: u32 = 2u;
const PT_CNT_NEGATIVE: u32 = 3u;

@group(0) @binding(4) var<uniform> pt: PtParams;

struct PtResult { L: vec3f, bvhFlags: u32 }

/// Σ radiance of camera-visible analytic area lights crossed by the camera ray before tMax (length-1, weight 1).
fn pt_camera_lights(o: vec3f, d: vec3f, tMax: f32) -> vec3f {
  var L = vec3f(0.0);
  let slot = lightsParams.cur;
  for (var i = 0u; i < slot.lightCount; i++) {
    let r = light_load(slot, i);
    if ((r.flags & LF_VISIBLE_CAMERA) != 0u && (r.kind == LT_RECT || r.kind == LT_DISK)) { L += area_light_crossing(r, o, d, tMax); }
  }
  return L;
}

fn pt_env_present() -> bool { return (envParams.flags & ENV_FLAG_PRESENT) != 0u; }

/// One path sample for camera ray (o, d) with path seed `seed`.
fn pt_trace(o: vec3f, d: vec3f, seed: PathSeed) -> PtResult {
  bvh_stats_reset();
  let hit0 = trace_closest(o, d, FLT_MAX);
  var L = pt_camera_lights(o, d, select(FLT_MAX, hit0.t, hit0.primId != BVH_MISS));
  if (hit0.primId == BVH_MISS) {
    L += envBackground(d);
    return PtResult(L, bvh_stats().w);
  }
  var s = scene_surface(hit0.primId, hit0.u, hit0.v, d);
  var prim = hit0.primId;
  L += tri_emission(prim, hit0.u, hit0.v);                         // length-1 emitter term, weight 1
  var beta = vec3f(1.0);
  var rrScale = 1.0;                                               // 1/∏q_RR (kept apart from the RR-free β)
  var V = -d;
  let lastB = pt.maxBounces + 1u;
  let neeOnly = (pt.flags & PT_NEE_ONLY) != 0u;
  let bsdfOnly = (pt.flags & PT_BSDF_ONLY) != 0u;
  for (var B = 1u; B <= lastB; B++) {
    let m = material_eval(s, V);
    // ---- NEE at x_B (not at a delta-only vertex) ------------------------------------------------------------------
    if ((m.flags & 1u) != 0u && !bsdfOnly) {
      let ls = nee_sample(s.pos, path_hash(seed, B, SLOT_SEL), path_hash(seed, B, SLOT_SEL2),
                          vec3u(path_hash(seed, B, SLOT_L0), path_hash(seed, B, SLOT_L1), path_hash(seed, B, SLOT_L2)));
      // same-triangle skip (Cycles shade_surface.h:345-351): a sample on the shading triangle itself has cosθ_z = 0
      if (ls.valid && ls.prim != prim && any(ls.Lambda > vec3f(0.0))) {
        let ev = bsdf_eval(m, V, ls.dir);
        if (any(ev.f_cos > vec3f(0.0))) {
          let w1 = select(nee_mis_w1(ls, ev.pdf_marginal, B), 1.0, neeOnly);
          var vis: bool;
          if (ls.isInf) { vis = visibleInf(s.pos, s.ng, prim, ls.dir); }
          else { vis = visible(s.pos, s.ng, prim, ls.pos, ls.nz, ls.prim); }
          if (vis) { L += (rrScale * w1 / ls.q) * (beta * ev.f_cos * ls.Lambda); }
        }
      }
    }
    // ---- planted bias (Gate 1 calibration only): drop the continuation without compensation -------------------------
    if (pt.dropProb > 0.0 && B == pt.dropBounce && path_u01(seed, B, SLOT_DBG) < pt.dropProb) { break; }
    // ---- Russian roulette (initial sampling only; after NEE, before the continuation) ------------------------------
    if ((pt.flags & PT_RR) != 0u && B > pt.rrMinBounces) {
      let q = min(sqrt(max(max(beta.x, beta.y), beta.z)), 1.0);
      if (!(path_u01(seed, B, SLOT_RR) < q)) { break; }
      rrScale /= q;
    }
    // ---- BSDF continuation ----------------------------------------------------------------------------------------
    let bs = bsdf_sample(m, V, path_bsdf_u4(seed, B));
    if (!bs.valid) { break; }
    beta *= bs.weight;
    if (!any(beta > vec3f(0.0))) { break; }
    let org = offset_ray(s.pos, select(-s.ng, s.ng, dot(s.ng, bs.L) >= 0.0));
    let h = trace_closest_ex(org, bs.L, FLT_MAX, prim, BVH_MISS);
    if (h.primId == BVH_MISS) {
      if (pt_env_present() && !neeOnly) {
        let w2 = select(env_bsdf_mis_weight(bs.L, bs.pdf_marginal, B, bs.is_delta), 1.0, bsdfOnly);
        L += (rrScale * w2) * beta * envRadiance(envUV(bs.L, envParams.cg, envParams.sg));
      }
      break;
    }
    let s2 = scene_surface(h.primId, h.u, h.v, bs.L);
    let Le = tri_emission(h.primId, h.u, h.v);
    if (any(Le > vec3f(0.0)) && !neeOnly) {
      var w2 = 1.0;
      if (!bs.is_delta && !bsdfOnly) {
        w2 = mis_w2(tri_light_p1(s.pos, s2.pos, s2.ng, h.primId), bs.pdf_marginal, B);
      }
      L += (rrScale * w2) * beta * Le;
    }
    s = s2;
    prim = h.primId;
    V = -bs.L;
  }
  return PtResult(L * pt.emitScale, bvh_stats().w);
}

#if !PT_INTERACTIVE
@group(2) @binding(0) var<storage, read_write> accum: array<vec4f>;            // rgb sum over samples
@group(2) @binding(1) var<storage, read_write> counters: array<atomic<u32>, 4>; // PT_CNT_*

@compute @workgroup_size(8, 8, 1)
fn pt_batch(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = vec2u(gid.x, gid.y + pt.rowBase);
  if (any(pixel >= frame.resolution) || pixel.y >= pt.rowEnd) { return; }
  let idx = pixel.y * frame.resolution.x + pixel.x;
  var sum = vec3f(0.0);
  var bad = 0u;
  var neg = 0u;
  var flags = 0u;
  for (var k = 0u; k < pt.sampleCount; k++) {
    let t = pt.sampleBase + k;
    let seed = path_init_seed(frame.runSeed, pt.member, t, idx);
    var jit = frame.jitter;
    if (frame.jitterMode == JITTER_IID) { jit = rand2(seed.x, 0u, STREAM_JITTER); }
    let ray = frame_camera_ray(pixel, jit);
    let r = pt_trace(ray.o, ray.d, seed);
    flags |= r.bvhFlags;
    if (all_finite3(r.L)) {
      sum += r.L;
      if (any(r.L < vec3f(0.0))) { neg++; }
    } else { bad++; }
  }
  accum[idx] += vec4f(sum, 0.0);
  if (bad != 0u) { atomicAdd(&counters[PT_CNT_NONFINITE], bad); }
  if (neg != 0u) { atomicAdd(&counters[PT_CNT_NEGATIVE], neg); }
  if ((flags & BVH_FLAG_OVERFLOW) != 0u) { atomicAdd(&counters[PT_CNT_BVH_OVERFLOW], 1u); }
  if ((flags & BVH_FLAG_ITERCAP) != 0u) { atomicAdd(&counters[PT_CNT_BVH_ITERCAP], 1u); }
}
#else
@group(2) @binding(0) var colorOut: texture_storage_2d<$COLOR_FORMAT, write>;
@group(2) @binding(1) var<storage, read_write> accum: array<vec4f>;            // rgb sum, w = sample count
@group(2) @binding(2) var<storage, read_write> counters: array<atomic<u32>, 4>;

@compute @workgroup_size(8, 8, 1)
fn pt_frame(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = gid.xy;
  if (any(pixel >= frame.resolution)) { return; }
  let idx = pixel.y * frame.resolution.x + pixel.x;
  // Same jitter and seed as the primary pass (frame_pixel_seed = path_init_seed(runSeed, 0, seedIndex, p).x).
  let seed = path_init_seed(frame.runSeed, pt.member, frame.seedIndex, idx);
  let ray = frame_camera_ray(pixel, frame_pixel_jitter(pixel));
  let r = pt_trace(ray.o, ray.d, seed);
  var c = r.L;
  if (!all_finite3(c)) { c = vec3f(0.0); atomicAdd(&counters[PT_CNT_NONFINITE], 1u); }
  if ((r.bvhFlags & BVH_FLAG_OVERFLOW) != 0u) { atomicAdd(&counters[PT_CNT_BVH_OVERFLOW], 1u); }
  if ((r.bvhFlags & BVH_FLAG_ITERCAP) != 0u) { atomicAdd(&counters[PT_CNT_BVH_ITERCAP], 1u); }
  let restart = (frame.flags & (FRAME_RESET_HISTORY | FRAME_CAMERA_MOVED)) != 0u || (pt.flags & PT_ACCUMULATE) == 0u;
  var a = select(accum[idx], vec4f(0.0), restart);
  if (restart || (pt.flags & PT_ADVANCED) != 0u) { a += vec4f(c, 1.0); }
  accum[idx] = a;
  textureStore(colorOut, pixel, vec4f(a.rgb / max(a.w, 1.0), 1.0));
}
#endif

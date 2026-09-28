// Emission-only kernel (plan §5 M2, harness calibration C0a/C0b/C0p; gap-light §U11). Per pixel and sample: one
// jittered camera ray (math.md#raster, frame.wgsl), and its length-1 radiance:
//   - L_e of the first triangle hit (TRI_EMISSIVE: material emission = emissiveFactor·strength ⊙ emissive texture;
//     two-sided, math.md#units-lights "Emissive triangle"), 0 for non-emissive hits;
//   - + every camera-visible analytic area light (rect/disk) crossed before that hit: one-sided, pass-through, weight 1,
//     L = Φ/(πA)·spread(θ) (math.md#units-lights). Mode A forbids them in validation scenes; Mode B tests use them;
//   - on a miss: + visibleToCamera·L_env (env.wgsl envBackground; math.md#env-mapping, length-1 env term).
// Point/spot/sun lights are never hit. Alpha-MASK cutouts are skipped inside traversal (no vertex, no RNG).
// Samples accumulate in f32 into accum[pixel] (rgb sum); the CPU divides by the batch's spp (batch-accumulator.ts).
// Jitter: sample k of the run uses u = rand2(pcg3d(runSeed, k, pixelIndex).x, 0, STREAM_JITTER) — i.i.d. per run and
// per sample (the per-sample index plays the role of the frame index t in math.md#raster).
// Defines: SCENE_GROUP (=1), BVH_* / TEX_* / ENV_* (emission-kernel.ts), CUSTOM_ALPHA, WATERTIGHT.
#include "common/frame.wgsl"
#include "common/nan.wgsl"
#include "scene/scene-data.wgsl"
#include "bvh/traverse.wgsl"
#include "lights/env.wgsl"

struct EmissionParams {
  sampleBase: u32,    // run-global index of this dispatch's first sample
  sampleCount: u32,   // samples per pixel in this dispatch
  lightCount: u32,    // entries in camLights
  rowBase: u32,       // first image row of this dispatch (row-band sub-dispatches, plan §1.8 submit budget)
  rowEnd: u32,        // one past the last row of this dispatch
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

// Camera-visible area light, recentred (emission-kernel.ts packCamLights). 80 B.
struct CamLight {
  center: vec3f,
  shape: u32,         // 0 rect, 1 disk (ellipse with half axes halfU, halfV)
  axisU: vec3f,       // unit X_obj
  halfU: f32,
  axisV: vec3f,       // unit Y_obj
  halfV: f32,
  radiance: vec3f,    // color · power · 2^exposure / (π A)
  spreadNorm: f32,    // N_s = 1/(tan a − a) (or 3/a³); < 0 ⇒ spread = π (factor 1)
  normal: vec3f,      // emission axis a_L = −Z_obj
  tanHalfSpread: f32, // tan a, a = spread/2
}

@group(0) @binding(4) var<uniform> ep: EmissionParams;
@group(0) @binding(5) var<storage, read> camLights: array<CamLight>;

@group(2) @binding(0) var<storage, read_write> accum: array<vec4f>;          // rgb sum over samples, w unused
@group(2) @binding(1) var<storage, read_write> counters: array<atomic<u32>, 4>; // 0 NaN/Inf samples, 1 BVH overflow, 2 BVH iteration cap

const EMI_CNT_NONFINITE: u32 = 0u;
const EMI_CNT_BVH_OVERFLOW: u32 = 1u;
const EMI_CNT_BVH_ITERCAP: u32 = 2u;

/// Radiance of camera-visible area light `li` along the ray (o, d) if crossed at t ∈ (0, tMax). math.md#units-lights
fn cam_light_radiance(li: CamLight, o: vec3f, d: vec3f, tMax: f32) -> vec3f {
  let dn = dot(d, li.normal);
  if (!(dn < 0.0)) { return vec3f(0.0); }                 // one-sided: the ray must travel against a_L
  let t = dot(li.center - o, li.normal) / dn;
  if (!(t > 0.0 && t < tMax)) { return vec3f(0.0); }
  let p = o + t * d - li.center;
  let x = dot(p, li.axisU) / li.halfU;
  let y = dot(p, li.axisV) / li.halfV;
  let inside = select(abs(x) <= 1.0 && abs(y) <= 1.0, x * x + y * y <= 1.0, li.shape == 1u);
  if (!inside) { return vec3f(0.0); }
  if (li.spreadNorm < 0.0) { return li.radiance; }
  let cosT = -dn;                                           // angle between the emitted direction −d and a_L
  let tanT = sqrt(max(1.0 - cosT * cosT, 0.0)) / cosT;
  return li.radiance * max((li.tanHalfSpread - tanT) * li.spreadNorm, 0.0);   // Cycles area.h:106-118
}

struct EmissionSample { L: vec3f, bvhFlags: u32 }

fn emission_sample(o: vec3f, d: vec3f) -> EmissionSample {
  bvh_stats_reset();
  let hit = trace_closest(o, d, FLT_MAX);
  var L = vec3f(0.0);
  var tHit = FLT_MAX;
  if (hit.primId != BVH_MISS) {
    tHit = hit.t;
    let tri = sceneTris[hit.primId];
    if ((tri_flags(tri) & TRI_EMISSIVE) != 0u) {
      let m = sceneMaterials[tri_material(tri)];
      L = m.emission * tex_sample(m.texEmissive, scene_uv0(tri, hit.u, hit.v)).rgb;   // two-sided, no π
    }
  } else {
    L = envBackground(d);
  }
  for (var i = 0u; i < ep.lightCount; i++) { L += cam_light_radiance(camLights[i], o, d, tHit); }
  return EmissionSample(L, bvh_stats().w);
}

@compute @workgroup_size(8, 8, 1)
fn emission(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = vec2u(gid.x, gid.y + ep.rowBase);
  if (any(pixel >= frame.resolution) || pixel.y >= ep.rowEnd) { return; }
  let idx = pixel.y * frame.resolution.x + pixel.x;
  var sum = vec3f(0.0);
  var bad = 0u;
  var flags = 0u;
  for (var s = 0u; s < ep.sampleCount; s++) {
    let k = ep.sampleBase + s;
    var jit = frame.jitter;
    if (frame.jitterMode == JITTER_IID) { jit = rand2(pcg3d(vec3u(frame.runSeed, k, idx)).x, 0u, STREAM_JITTER); }
    let ray = frame_camera_ray(pixel, jit);
    let r = emission_sample(ray.o, ray.d);
    flags |= r.bvhFlags;
    if (all_finite3(r.L)) { sum += r.L; } else { bad++; }   // NaN/Inf: bit tests (plan §1.1), counted, fails validation
  }
  accum[idx] += vec4f(sum, 0.0);
  if (bad != 0u) { atomicAdd(&counters[EMI_CNT_NONFINITE], bad); }
  if ((flags & BVH_FLAG_OVERFLOW) != 0u) { atomicAdd(&counters[EMI_CNT_BVH_OVERFLOW], 1u); }
  if ((flags & BVH_FLAG_ITERCAP) != 0u) { atomicAdd(&counters[EMI_CNT_BVH_ITERCAP], 1u); }
}

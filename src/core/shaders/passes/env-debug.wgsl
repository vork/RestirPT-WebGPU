// Env sampling debug views (plan §6 "M3c Env sampling"): production env-sample.wgsl functions, drawn through the debug
// AOV plane (group 3, debug_write) so the resolve pass maps them to false colour. Host: src/core/render/env-debug.ts.
//   ENV_IMPORTANCE (310)  pdfUV of the env cell seen along the camera ray (log scale; env space, geometry ignored)
//   ENV_RATIO      (311)  realized / target density of that cell − 1 (u16 alias quantization; ≈ 0)
//   ENV_SPLAT      (312)  splat histogram of NEE_ENV samples (env_splat) / expected count from the realized pdf − 1 (≈ 0);
//                         the χ² of the same histogram is computed on the CPU (HUD)
//   ENV_ESCAPE     (313)  fraction of BSDF rays from the primary hit that escape to the env (running mean)
//   ENV_W1         (314)  ω1 of NEE_ENV samples at the primary hit (running mean over samples with f > 0)
//   ENV_W2         (315)  ω2 of BSDF_ENV escapes at the primary hit (running mean over escapes)
// Bindings: G0 frame | env | params | LightsParams | records; G1 scene; G2 0 `envDbg` (u32 atomics: splat histogram
// [cells] then per-pixel running sums (escape n, escape hits, ω1 sum, ω1 n, ω2 sum, ω2 n) as f32 bits), 1 targetUV (f32);
// G3 debug.
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
#include "debug/debug-common.wgsl"

const DBG_ENV_IMPORTANCE: u32 = 310u;
const DBG_ENV_RATIO: u32 = 311u;
const DBG_ENV_SPLAT: u32 = 312u;
const DBG_ENV_ESCAPE: u32 = 313u;
const DBG_ENV_W1: u32 = 314u;
const DBG_ENV_W2: u32 = 315u;
const ENV_DBG_PIXEL_WORDS: u32 = 6u;

struct EnvDbgParams {
  splatCount: u32,     // env_splat: threads
  splatK: u32,         // env_splat: samples per thread
  splatSeed: u32,
  reset: u32,          // 1: restart the per-pixel running means this frame
  splatTotal: f32,     // Σ samples in the histogram (for the expected counts)
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(4) var<uniform> ed: EnvDbgParams;
@group(2) @binding(0) var<storage, read_write> envDbg: array<atomic<u32>>;
@group(2) @binding(1) var<storage, read> targetUV: array<f32>;

fn env_cells() -> u32 { return select(0u, env_Wm() * (env_Wm() >> 1u), env_tables_present()); }

@compute @workgroup_size(256)
fn env_splat(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= ed.splatCount || !env_tables_present()) { return; }
  let W = env_Wm();
  for (var k = 0u; k < ed.splatK; k++) {
    let h = pcg3d(vec3u(ed.splatSeed, gid.x, k));
    let c = env_sample_cell(h.x, h.y, h.z);
    atomicAdd(&envDbg[c.i * W + c.j], 1u);
  }
}

fn acc_load(i: u32) -> f32 { return bitcast<f32>(atomicLoad(&envDbg[i])); }
fn acc_store(i: u32, v: f32) { atomicStore(&envDbg[i], bitcast<u32>(v)); }

@compute @workgroup_size(8, 8, 1)
fn env_view(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = gid.xy;
  if (any(pixel >= frame.resolution)) { return; }
  let ray = frame_camera_ray(pixel, vec2f(0.5));
  let mode = dbg.mode;
  if (mode == DBG_ENV_IMPORTANCE || mode == DBG_ENV_RATIO || mode == DBG_ENV_SPLAT) {
    if (!env_tables_present()) { debug_write(pixel, mode, vec4f(0.0)); return; }
    let c = env_cell_of(envUV(ray.d, envParams.cg, envParams.sg));
    let k = c.y * env_Wm() + c.x;
    let p = env_pdf_uv(c.y, c.x);
    var v = p;
    // ratio views are written as (ratio − 1) for the diverging colormap (centred on 0)
    if (mode == DBG_ENV_RATIO) { v = p / max(targetUV[k], 1e-30) - 1.0; }
    if (mode == DBG_ENV_SPLAT) {
      let expect = ed.splatTotal * p / f32(env_cells());
      v = select(0.0, f32(atomicLoad(&envDbg[k])) / expect - 1.0, expect > 0.0);
    }
    debug_write(pixel, mode, vec4f(v, 0.0, 0.0, 0.0));
    return;
  }
  // Primary-hit views: one BSDF sample and one env NEE sample per pixel per frame, running means.
  let base = env_cells() + (pixel.y * frame.resolution.x + pixel.x) * ENV_DBG_PIXEL_WORDS;
  if (ed.reset != 0u || (frame.flags & (FRAME_RESET_HISTORY | FRAME_CAMERA_MOVED)) != 0u) { for (var w = 0u; w < ENV_DBG_PIXEL_WORDS; w++) { acc_store(base + w, 0.0); } }
  let hit = trace_closest(ray.o, ray.d, FLT_MAX);
  if (hit.primId == BVH_MISS) { debug_write(pixel, mode, vec4f(0.0)); return; }
  let s = scene_surface(hit.primId, hit.u, hit.v, ray.d);
  let V = -ray.d;
  let m = material_eval(s, V);
  let seed = path_init_seed(frame.runSeed ^ 0x5eedu, 0u, frame.seedIndex, pixel.y * frame.resolution.x + pixel.x);
  let bs = bsdf_sample(m, V, path_bsdf_u4(seed, 1u));
  var escaped = false;
  if (bs.valid) {
    escaped = visibleInf(s.pos, s.ng, hit.primId, bs.L);
    acc_store(base, acc_load(base) + 1.0);
    if (escaped) {
      acc_store(base + 1u, acc_load(base + 1u) + 1.0);
      acc_store(base + 4u, acc_load(base + 4u) + env_bsdf_mis_weight(bs.L, bs.pdf_marginal, 1u, bs.is_delta));
      acc_store(base + 5u, acc_load(base + 5u) + 1.0);
    }
  }
  let slot = lightsParams.cur;
  if (slot.envEntry != LIGHT_NONE && (m.flags & 1u) != 0u) {
    let ls = env_light_sample(slot, slot.envEntry, vec3u(path_hash(seed, 1u, SLOT_L0), path_hash(seed, 1u, SLOT_L1), path_hash(seed, 1u, SLOT_L2)));
    if (ls.valid) {
      let ev = bsdf_eval(m, V, ls.dir);
      if (any(ev.f_cos > vec3f(0.0))) {
        acc_store(base + 2u, acc_load(base + 2u) + nee_mis_w1(ls, ev.pdf_marginal, 1u));
        acc_store(base + 3u, acc_load(base + 3u) + 1.0);
      }
    }
  }
  var v = 0.0;
  if (mode == DBG_ENV_ESCAPE) { v = acc_load(base + 1u) / max(acc_load(base), 1.0); }
  if (mode == DBG_ENV_W1) { v = acc_load(base + 2u) / max(acc_load(base + 3u), 1.0); }
  if (mode == DBG_ENV_W2) { v = acc_load(base + 4u) / max(acc_load(base + 5u), 1.0); }
  debug_write(pixel, mode, vec4f(v, 0.0, 0.0, 0.0));
}

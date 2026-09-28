// M1 G-buffer / BVH / env debug-view writers for the primary pass (plan §6 M1 rows). Ids 100-109, 200-204 and
// 300-301 are the framework's (debug-common.wgsl); the extra views below are registered by renderer.ts
// (EXTRA_VIEWS) and must keep these ids.
#include "debug/debug-common.wgsl"
#include "passes/gbuffer.wgsl"

const DBG_GB_MOTION_HUE: u32 = 110u;   // vec3: hue = direction, value = 1 − exp(−|mv|/4 px)
const DBG_CHK_NONFINITE: u32 = 111u;   // code: bit0 colour, bit1 G-buffer vectors, bit2 depth/thr non-finite; black = ok
const DBG_GB_HITDIST: u32 = 112u;      // scalar: ray t to the primary hit (m)

// NaN/Inf check bits for DBG_CHK_NONFINITE.
const CHK_COLOR: u32 = 1u;
const CHK_VECTORS: u32 = 2u;
const CHK_SCALARS: u32 = 4u;

fn dbg_hsv(h: f32, s: f32, v: f32) -> vec3f {
  let k = vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0);
  let p = abs(fract(vec3f(h) + k) * 6.0 - 3.0);
  return v * mix(vec3f(1.0), clamp(p - 1.0, vec3f(0.0), vec3f(1.0)), s);
}

/// Motion vector colour: hue = atan2 direction (image y down), value saturates smoothly with length.
fn dbg_motion_colour(mv: vec2f) -> vec3f {
  let len = length(mv);
  let h = fract(atan2(-mv.y, mv.x) * INV_TWO_PI + 1.0);
  return dbg_hsv(h, 1.0, 1.0 - exp(-len * 0.25));
}

/// All G-buffer / BVH views of one pixel. Cheap: every write is a no-op unless its view is active.
fn gbuffer_debug_write(pixel: vec2u, g: GBufTexel, primId: u32, bary: vec3f, uv: vec2f, hitT: f32, stats: vec4u, chk: u32) {
  if (dbg.mode == DBG_OFF) { return; }
  let hit = (g.flags & GB_HIT) != 0u;
  debug_write3(pixel, DBG_GB_ALBEDO, g.albedo);
  debug_write3(pixel, DBG_GB_NS, g.ns);
  debug_write3(pixel, DBG_GB_NG, g.ng);
  debug_write1(pixel, DBG_GB_DEPTH, g.viewZ);
  debug_write_code(pixel, DBG_GB_PRIM, primId);
  debug_write_code(pixel, DBG_GB_MATERIAL, g.matId);
  debug_write3(pixel, DBG_GB_UV, select(vec3f(0.0), vec3f(fract(uv), 0.0), hit));
  debug_write1(pixel, DBG_GB_THR, g.thr);
  debug_write3(pixel, DBG_GB_MOTION, vec3f(g.motion, 0.0));
  debug_write3(pixel, DBG_GB_MOTION_HUE, select(vec3f(0.0), dbg_motion_colour(g.motion), (g.flags & GB_MOTION_VALID) != 0u));
  debug_write3(pixel, DBG_GB_BARY, bary);
  debug_write1(pixel, DBG_GB_HITDIST, select(0.0, hitT, hit));
  debug_write1(pixel, DBG_BVH_STEPS, f32(stats.x));
  debug_write1(pixel, DBG_BVH_BOX, f32(stats.y));
  debug_write1(pixel, DBG_BVH_TRI, f32(stats.z));
  debug_write1(pixel, DBG_BVH_STACK, f32((stats.w >> 8u) & 0xffu));
  debug_write_code(pixel, DBG_BVH_FLAGS, select(DBG_CODE_NONE, stats.w & 3u, (stats.w & 3u) != 0u));
  debug_write_code(pixel, DBG_CHK_NONFINITE, select(DBG_CODE_NONE, chk, chk != 0u));
  debug_write_code(pixel, DBG_ENV_BGMASK, select(1u, DBG_CODE_NONE, hit));
}

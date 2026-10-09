// G-buffer texel written by the primary pass (plan §3 step 1) and read by later passes (M3a+). One per internal
// pixel, row-major (row 0 = top). The V-buffer proper (primId, f32 barycentrics) is the rgba32uint texture
// `vbuf`: (primId | BVH_MISS, bitcast(u), bitcast(v), matId | 0xffffffff). Mirror: GBUF_TEXEL_BYTES in renderer.ts.
struct GBufTexel {
  ng: vec3f,       // unit geometric normal, oriented toward the camera side (0 on background)
  thr: f32,        // footprint threshold τ·R²_pri of this jittered primary hit (math.md#rc-predicate); 0 = background
  ns: vec3f,       // shading normal, flipped consistently with ng
  viewZ: f32,      // linear view depth −z_cam (0 = background)
  pos: vec3f,      // hit point (internal frame) from barycentrics
  matId: u32,      // 0xffffffff on background
  albedo: vec3f,   // base colour × texture × COLOR_0
  flags: u32,      // GB_* bits
  motion: vec2f,   // previous − current continuous image position (pixels); background: rotation-only
  pad: vec2f,
}

#if GBUF_48
// perf2 WP-7d: the stored texel (48 B, lossless for every reader: the denoiser reads pos, flags, ns, albedo, motion; ng,
// thr, viewZ and matId stay in the primary pass, which writes their debug views, the depth texture and vbuf.w).
// Mirror: gbufTexelBytes() in renderer.ts.
struct GBufStore {
  pos: vec3f,
  flags: u32,
  ns: vec3f,
  motionX: f32,
  albedo: vec3f,
  motionY: f32,
}
fn gbuf_store(g: GBufTexel) -> GBufStore { return GBufStore(g.pos, g.flags, g.ns, g.motion.x, g.albedo, g.motion.y); }
#endif

const GB_HIT: u32 = 1u;
const GB_BACKFACING: u32 = 2u;
const GB_MOTION_VALID: u32 = 4u;

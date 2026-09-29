// Path-stream RNG with the fixed per-vertex slot layout (plan §1.7; math.md#rng-layout, open items 4/5).
//   slot value = h(initSeed, vertex·D + slot),  D = 16,  vertex = index b ≥ 1 of the scattering vertex x_b
//   h(seed, i) = pcg4d(seed.x, seed.y, i, STREAM_PATH).x (rng.wgsl); u01 = f32(h >> 8)·2⁻²⁴ ∈ [0, 1)
//   initSeed   = pcg3d(runSeed ⊕ member·φ, frame t, pixel p).xy — a 64-bit PathSeed. initSeed.x equals frame.wgsl
//                frame_pixel_seed for member 0, so the jitter stream (STREAM_JITTER, keyed by .x) and the path stream
//                (STREAM_PATH) share the seed but never a hash input.
// Why 64 bits (M3c finding, math.md#rng-layout): with a 32-bit initSeed every path of every pixel and frame is one of
// only 2^32 dimension vectors, so the estimator's expectation is a fixed 2^32-point quadrature, not the integral. Its
// error is ~σ/√2^32 per unit of single-sample relative σ — 0.05–0.13% for BSDF-only sampling of a 1e-4 sr texel
// (C0s NONE, reproducible across seeds, opposite sign for other 32-bit hash choices); 64 bits remove it.
// All four BSDF dims (u_lobe, u_h1, u_h2, u_rt) are consumed at EVERY bounce, NEE dims are never replayed.
#include "common/rng.wgsl"

const RNG_D: u32 = 16u;
const SLOT_LOBE: u32 = 0u;   // lobe-class pick                      (replayed)
const SLOT_H1: u32 = 1u;     // VNDF / cosine sample dim 1          (replayed)
const SLOT_H2: u32 = 2u;     // VNDF / cosine sample dim 2          (replayed)
const SLOT_RT: u32 = 3u;     // glass R/T decision                  (replayed)
const SLOT_SEL: u32 = 4u;    // NEE light-alias bucket (top bits) + threshold (low 16 bits)
const SLOT_L0: u32 = 5u;     // light (u,v).u / triangle u1 / env row hash h0
const SLOT_L1: u32 = 6u;     // light (u,v).v / triangle u2 / env column hash h1
const SLOT_L2: u32 = 7u;     // env in-cell offsets h2
const SLOT_SEL2: u32 = 8u;   // second alias threshold hash (padded table > 2^16 entries)
const SLOT_RR: u32 = 9u;     // Russian roulette (initial sampling only)
const SLOT_TAU: u32 = 14u;   // optional rc-threshold jitter

/// Per-pixel, per-frame (or per-sample) path seed. math.md#rng-layout initSeed
alias PathSeed = vec2u;

fn path_init_seed(runSeed: u32, member: u32, t: u32, pixelIndex: u32) -> PathSeed {
  return pcg3d(vec3u(runSeed ^ (member * 0x9e3779b9u), t, pixelIndex)).xy;
}

/// Raw u32 of slot `slot` at scattering vertex `vertex` (b ≥ 1).
fn path_hash(initSeed: PathSeed, vertex: u32, slot: u32) -> u32 {
  return pcg4d(vec4u(initSeed.x, initSeed.y, vertex * RNG_D + slot, STREAM_PATH)).x;
}

fn path_u01(initSeed: PathSeed, vertex: u32, slot: u32) -> f32 {
  return u32_to_unit(path_hash(initSeed, vertex, slot));
}

/// (u_lobe, u_h1, u_h2, u_rt) of vertex b: the BSDF sampler input (docs/decisions/bsdf-api.md bsdf_sample).
fn path_bsdf_u4(initSeed: PathSeed, vertex: u32) -> vec4f {
  return vec4f(path_u01(initSeed, vertex, SLOT_LOBE), path_u01(initSeed, vertex, SLOT_H1),
               path_u01(initSeed, vertex, SLOT_H2), path_u01(initSeed, vertex, SLOT_RT));
}

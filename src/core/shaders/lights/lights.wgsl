// Light records and the shared `records` buffer (plan §1.4 "Storage", §1.8 "records"; math.md#units-lights,
// math.md#light-selection). CPU packer + layout: src/core/render/lights-gpu.ts (LIGHT_REC, LightsState).
//
// Bindings (the includer defines LIGHTS_GROUP and LIGHTS_BINDING): uniform LightsParams at LIGHTS_BINDING, the
// read-only `records: array<u32>` at LIGHTS_BINDING + 1. Define RECORDS_EXTERNAL when another module declares
// `records` (shared with the BSDF LUTs) — then only the uniform is declared here.
//
// Analytic lights are never in the BVH and never occlude (math.md#visibility). Frame: a_u = X_obj (axisU),
// a_v = Y_obj (axisV), emission axis a_L = −Z_obj (normal); the direction toward the sun is −normal = +Z_obj.
#include "common/math.wgsl"

// math.md#path-tree endpoint types
const LT_TRI: u32 = 0u;
const LT_POINT: u32 = 1u;
const LT_SPOT: u32 = 2u;
const LT_RECT: u32 = 3u;
const LT_DISK: u32 = 4u;
const LT_SUN: u32 = 5u;
const LT_ENV: u32 = 7u;

const LF_VISIBLE_CAMERA: u32 = 1u;
const LF_DELTA: u32 = 2u;

const LP_MODE_A: u32 = 1u;    // analytic area lights are NEE-only: ω1 ≡ 1, never hit by BSDF rays
const LP_ENV_NEE: u32 = 2u;   // the env is an alias entry (env NEE on); BSDF escapes use MIS with p1Env

const LIGHT_REC_WORDS: u32 = 28u;
const LIGHT_NONE: u32 = 0xffffffffu;

struct LightRec {
  pos: vec3f,          // centre / position (render-internal recentred frame)
  kind: u32,           // LT_*
  axisU: vec3f,        // unit X_obj
  halfU: f32,          // rect: sizeX/2, disk: diameter/2
  axisV: vec3f,        // unit Y_obj
  halfV: f32,
  normal: vec3f,       // emission axis a_L = −Z_obj
  area: f32,           // A (rect/disk), 0 otherwise
  emit: vec3f,         // point/spot: I = Φ/(4π) [W/sr]; rect/disk: L_e = Φ/(πA); sun: E = Φ [W/m²]
  flags: u32,          // LF_*
  cosHalf: f32,        // spot: cos(spot_size/2)
  spotSmooth: f32,     // spot: 1/((1 − cosH)·blend); < 0 ⇒ blend = 0 (hard step guard)
  spreadNorm: f32,     // rect/disk: N_s = 1/(tan a − a) (or 3/a³); < 0 ⇒ spread = π (factor 1)
  tanHalfSpread: f32,  // tan a, a = spread/2
  stableId: u32,
  invArea: f32,        // 1/A (rect/disk)
}

// Mirrors LightSlotCpu (48 B).
struct LightSlot {
  lightOff: u32,       // word offset of light record 0
  lightCount: u32,
  aliasOff: u32,       // (q, alias) u32 pairs, 2^aliasLog2 entries
  aliasLog2: u32,
  pmfOff: u32,         // realized pmf per alias entry (f32 bits)
  nAnalytic: u32,      // entries [0, nAnalytic) are analytic lights, then emissive triangles
  nEntries: u32,       // 0 ⇒ nothing to sample (no lights, or all zero power)
  envEntry: u32,       // alias entry of ENV_ID (env NEE on), LIGHT_NONE when absent
  curToPrevOff: u32,
  prevToCurOff: u32,
  pad0: u32,
  pad1: u32,
}

struct LightsParams {
  cur: LightSlot,
  prev: LightSlot,
  triOff: u32,         // emissive-triangle entries: (primId, bits(area)) pairs
  triCount: u32,
  primMapOff: u32,     // primId → emissive-triangle entry index or LIGHT_NONE
  flags: u32,          // LP_*
  envRowOff: u32,      // env importance tables (lights/env-sample.wgsl): row alias (H_m entries alias<<16 | q)
  envColOff: u32,      //   column alias per row (H_m·W_m)
  envPdfOff: u32,      //   realized pdfUV per cell (f32 bits)
  envLog2W: u32,       //   log2 W_m (0: no tables); H_m = W_m/2
}

@group($LIGHTS_GROUP) @binding($LIGHTS_BINDING) var<uniform> lightsParams: LightsParams;
#if !RECORDS_EXTERNAL
@group($LIGHTS_GROUP) @binding($LIGHTS_BINDING + 1) var<storage, read> records: array<u32>;
#endif

fn rec_f32(w: u32) -> f32 { return bitcast<f32>(records[w]); }
fn rec_vec3(w: u32) -> vec3f { return vec3f(rec_f32(w), rec_f32(w + 1u), rec_f32(w + 2u)); }

fn light_load(slot: LightSlot, i: u32) -> LightRec {
  let b = slot.lightOff + i * LIGHT_REC_WORDS;
  var r: LightRec;
  r.pos = rec_vec3(b);            r.kind = records[b + 3u];
  r.axisU = rec_vec3(b + 4u);     r.halfU = rec_f32(b + 7u);
  r.axisV = rec_vec3(b + 8u);     r.halfV = rec_f32(b + 11u);
  r.normal = rec_vec3(b + 12u);   r.area = rec_f32(b + 15u);
  r.emit = rec_vec3(b + 16u);     r.flags = records[b + 19u];
  r.cosHalf = rec_f32(b + 20u);   r.spotSmooth = rec_f32(b + 21u);
  r.spreadNorm = rec_f32(b + 22u); r.tanHalfSpread = rec_f32(b + 23u);
  r.stableId = records[b + 24u];  r.invArea = rec_f32(b + 25u);
  return r;
}

fn light_is_delta(r: LightRec) -> bool { return (r.flags & LF_DELTA) != 0u; }

/// Emissive-triangle entry i: (primId, area).
fn emissive_tri(i: u32) -> vec2u {
  let b = lightsParams.triOff + 2u * i;
  return vec2u(records[b], records[b + 1u]);
}

/// Emissive-triangle entry index of primId, LIGHT_NONE if the triangle is not an NEE entry (emission_sampling NONE).
fn emissive_entry_of_prim(primId: u32) -> u32 {
  if (lightsParams.triCount == 0u) { return LIGHT_NONE; }
  return records[lightsParams.primMapOff + primId];
}

/// One NEE light sample in the product measure μ (math.md#measure). Filled by the per-type samplers
/// (point/spot/area/sun/emissive.wgsl) and completed (entry, q, p1) by measure.wgsl light_sample().
struct LightSample {
  valid: bool,         // false: nothing sampled (no lights) or a zero-probability sample
  entry: u32,          // alias entry
  kind: u32,           // LT_*
  pos: vec3f,          // z on the light (sun: unused)
  nz: vec3f,           // geometric normal at z for the visibility offset (point/spot/sun: 0)
  prim: u32,           // emissive triangle primId (excluded by visible()), else 0xffffffff
  dir: vec3f,          // ω_L = (z − x)/r (sun: toward the sun)
  dist: f32,           // r (sun: FLT_MAX)
  cosZ: f32,           // |cosθ_z| at the light (area/triangle), 1 otherwise
  Lambda: vec3f,       // Λ_L(z; x): L_e|cosθ_z|/r² (area/tri), I/r² (point/spot), E (sun)
  q: f32,              // source density in μ: P(L)/A (area/tri), P(L) (delta, sun)
  p1: f32,             // solid-angle pdf q·r²/|cosθ_z| (area/tri); 0 (undefined) for delta lights and the sun
  isDelta: bool,       // point, spot, sun: ω1 ≡ 1 by flag (never by pdf arithmetic)
  isInf: bool,         // sun: visibility to infinity (visibleInf)
  analytic: bool,      // analytic light (Mode A: ω1 ≡ 1)
}

fn light_sample_none() -> LightSample {
  var s: LightSample;
  s.valid = false;
  s.prim = LIGHT_NONE;
  s.entry = LIGHT_NONE;
  s.cosZ = 1.0;
  return s;
}

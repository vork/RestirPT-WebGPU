// Debug framework (plan §6): DebugParams uniform, AOV plane, probe records, counters and false-colour maps.
// Mirror: src/core/render/debug-views.ts (ids, layout, flag bits). debugMode/debugTap are UNIFORMS (plan §1.8):
// switching views never recompiles. Bindings live in group 3:
//   @binding(0) dbg: DebugParams (uniform)
//   @binding(1) dbgBuf: DebugBuffer (storage, read_write): counters | probe records | AOV plane
//   @binding(2) debugOut: rgba32float storage texture (written only by the debug resolve pass)
// Define DEBUG_NO_BINDINGS to get only the types/colormaps (e.g. for the blit), DEBUG_OUT_BINDING to declare
// debugOut (only the resolve pass needs it; other passes keep the storage-texture budget free).
#include "common/math.wgsl"
#include "common/nan.wgsl"
#include "common/rng.wgsl"

struct DebugParams {
  mode: u32,            // view id (DBG_* below or registered in TS); 0 = beauty
  tap: u32,             // DBG_TAP_*: which stage writes stage-dependent views
  probePixel: vec2u,    // internal-resolution pixel (row 0 = top)
  rangeMin: f32,
  rangeMax: f32,
  flags: u32,           // DBGF_* bits
  frame: u32,           // frame index (for animated/debug hashes)
  kind: u32,            // DBG_KIND_* of the active view (set by TS from the registry)
  split: f32,           // A|B split position in [0,1] (left = debug view, right = beauty)
  size: vec2u,          // AOV plane dimensions (internal resolution)
}

const DBG_KIND_SCALAR: u32 = 0u;
const DBG_KIND_VEC3: u32 = 1u;
const DBG_KIND_CODE: u32 = 2u;

const DBGF_LOG: u32 = 1u;
const DBGF_CMAP_SHIFT: u32 = 1u;      // 2 bits: 0 viridis, 1 turbo, 2 signed (diverging), 3 grey
const DBGF_CMAP_MASK: u32 = 3u;
const DBGF_PROBE: u32 = 8u;           // probe recording enabled
const DBGF_SPLIT: u32 = 16u;          // A|B split enabled
const DBGF_ABS: u32 = 32u;            // map |v|
const DBGF_NONFINITE: u32 = 64u;      // highlight NaN/Inf in magenta/cyan

const CMAP_VIRIDIS: u32 = 0u;
const CMAP_TURBO: u32 = 1u;
const CMAP_SIGNED: u32 = 2u;
const CMAP_GREY: u32 = 3u;

// Stage taps (plan §6).
const DBG_TAP_FINAL: u32 = 0u;
const DBG_TAP_INITIAL: u32 = 1u;
const DBG_TAP_TEMPORAL: u32 = 2u;
const DBG_TAP_SPATIAL: u32 = 3u;
const DBG_TAP_DENOISED: u32 = 4u;

// View ids: stable, allocated per group (debug-views.ts BUILTIN_VIEWS must match).
const DBG_OFF: u32 = 0u;
const DBG_TEST_UV: u32 = 1u;
const DBG_TEST_DEPTH: u32 = 2u;
const DBG_TEST_NORMAL: u32 = 3u;
const DBG_TEST_CELL: u32 = 4u;
const DBG_GB_ALBEDO: u32 = 100u;
const DBG_GB_NS: u32 = 101u;
const DBG_GB_NG: u32 = 102u;
const DBG_GB_DEPTH: u32 = 103u;
const DBG_GB_PRIM: u32 = 104u;
const DBG_GB_MATERIAL: u32 = 105u;
const DBG_GB_UV: u32 = 106u;
const DBG_GB_THR: u32 = 107u;
const DBG_GB_MOTION: u32 = 108u;
const DBG_GB_BARY: u32 = 109u;
const DBG_BVH_STEPS: u32 = 200u;
const DBG_BVH_BOX: u32 = 201u;
const DBG_BVH_TRI: u32 = 202u;
const DBG_BVH_STACK: u32 = 203u;
const DBG_BVH_FLAGS: u32 = 204u;
const DBG_ENV_GRID: u32 = 300u;
const DBG_ENV_BGMASK: u32 = 301u;

// Counter slots in DebugBuffer.counters (cleared every frame; HUD reads them through the probe ring).
const DBGC_PROBE_COUNT: u32 = 0u;
const DBGC_PROBE_OVERFLOW: u32 = 1u;
const DBGC_NAN: u32 = 2u;
const DBGC_INF: u32 = 3u;
const DBGC_BVH_OVERFLOW: u32 = 4u;
const DBGC_BVH_ITERCAP: u32 = 5u;
const DBGC_QUEUE_OVERFLOW: u32 = 6u;
const DBGC_NEGATIVE: u32 = 7u;
const DBG_COUNTERS: u32 = 16u;
const PROBE_CAPACITY: u32 = 256u;

// Probe tags 0..15 are reserved for the framework; passes use 16+ (names registered in probe.ts).
const PROBE_TAG_AOV: u32 = 1u;        // raw AOV value at the probe pixel (debug resolve pass)
const PROBE_TAG_USER: u32 = 16u;

const DBG_CODE_NONE: u32 = 0xffffffffu;

struct ProbeRecord {
  pixel: vec2u,
  tag: u32,
  seq: u32,
  value: vec4f,
}

struct DebugBuffer {
  counters: array<atomic<u32>, 16>,       // 64 B
  probe: array<ProbeRecord, 256>,         // 8192 B
  aov: array<vec4f>,                      // offset 8256: one vec4f per internal pixel
}

#if !DEBUG_NO_BINDINGS
@group(3) @binding(0) var<uniform> dbg: DebugParams;
@group(3) @binding(1) var<storage, read_write> dbgBuf: DebugBuffer;
#if DEBUG_OUT_BINDING
@group(3) @binding(2) var debugOut: texture_storage_2d<rgba32float, write>;
#endif

/// True when view `id` is displayed: passes may skip computing debug values otherwise.
fn debug_active(id: u32) -> bool { return dbg.mode == id; }
fn debug_tap_active(id: u32, tap: u32) -> bool { return dbg.mode == id && dbg.tap == tap; }

/// Write the raw value of view `id` for `pixel` (no-op unless the view is active).
fn debug_write(pixel: vec2u, id: u32, v: vec4f) {
  if (dbg.mode != id || any(pixel >= dbg.size)) { return; }
  dbgBuf.aov[pixel.y * dbg.size.x + pixel.x] = v;
}
fn debug_write1(pixel: vec2u, id: u32, v: f32) { debug_write(pixel, id, vec4f(v, 0.0, 0.0, 0.0)); }
fn debug_write3(pixel: vec2u, id: u32, v: vec3f) { debug_write(pixel, id, vec4f(v, 0.0)); }
/// Integer code views (ids, flags, outcome codes): stored bit-exact in .x.
fn debug_write_code(pixel: vec2u, id: u32, code: u32) { debug_write(pixel, id, vec4f(bitcast<f32>(code), 0.0, 0.0, 0.0)); }
fn debug_write_tap(pixel: vec2u, id: u32, tap: u32, v: vec4f) { if (dbg.tap == tap) { debug_write(pixel, id, v); } }

fn debug_is_probe(pixel: vec2u) -> bool {
  return (dbg.flags & DBGF_PROBE) != 0u && all(pixel == dbg.probePixel);
}

/// Append a probe record when `pixel` is the probe pixel. Records beyond PROBE_CAPACITY count as overflow.
fn probe_record(pixel: vec2u, tag: u32, v: vec4f) {
  if (!debug_is_probe(pixel)) { return; }
  let i = atomicAdd(&dbgBuf.counters[DBGC_PROBE_COUNT], 1u);
  if (i >= PROBE_CAPACITY) { atomicAdd(&dbgBuf.counters[DBGC_PROBE_OVERFLOW], 1u); return; }
  dbgBuf.probe[i] = ProbeRecord(pixel, tag, i, v);
}

fn debug_count(slot: u32, n: u32) { atomicAdd(&dbgBuf.counters[slot], n); }

/// Count NaN/Inf components (bit tests, plan §1.1). Returns true if all finite.
fn debug_check_finite(v: vec3f) -> bool {
  var nan = 0u;
  var inf = 0u;
  for (var i = 0u; i < 3u; i++) {
    if (is_nan(v[i])) { nan++; } else if (is_inf(v[i])) { inf++; }
  }
  if (nan != 0u) { atomicAdd(&dbgBuf.counters[DBGC_NAN], 1u); }
  if (inf != 0u) { atomicAdd(&dbgBuf.counters[DBGC_INF], 1u); }
  return nan == 0u && inf == 0u;
}
#endif

// ---- false colour (outputs are display-encoded sRGB values in [0,1]) ------------------------------------------

/// Viridis, 6th-order polynomial fit to matplotlib's table (max error ~0.01).
fn cmap_viridis(t_in: f32) -> vec3f {
  let t = clamp(t_in, 0.0, 1.0);
  let c0 = vec3f(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  let c1 = vec3f(0.1050930431085774, 1.404613529898575, 1.384590162594685);
  let c2 = vec3f(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
  let c3 = vec3f(-4.634230498983486, -5.799100973351585, -19.33244095627987);
  let c4 = vec3f(6.228269936347081, 14.17993336680509, 56.69055260068105);
  let c5 = vec3f(4.776384997670288, -13.74514537774601, -65.35303263337234);
  let c6 = vec3f(-5.435455855934631, 4.645852612178535, 26.3124352495832);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), vec3f(0.0), vec3f(1.0));
}

/// Turbo (Mikhailov 2019), polynomial approximation.
fn cmap_turbo(t_in: f32) -> vec3f {
  let x = clamp(t_in, 0.0, 1.0);
  let v4 = vec4f(1.0, x, x * x, x * x * x);
  let v2 = v4.zw * v4.z;
  let r = dot(v4, vec4f(0.13572138, 4.61539260, -42.66032258, 132.13108234)) + dot(v2, vec2f(-152.94239396, 59.28637943));
  let g = dot(v4, vec4f(0.09140261, 2.19418839, 4.84296658, -14.18503333)) + dot(v2, vec2f(4.27729857, 2.82956604));
  let b = dot(v4, vec4f(0.10667330, 12.64194608, -60.58204836, 110.36276771)) + dot(v2, vec2f(-89.90310912, 27.34824973));
  return clamp(vec3f(r, g, b), vec3f(0.0), vec3f(1.0));
}

/// Diverging map for signed values, s in [-1, 1]: blue (neg) - white (0) - red (pos).
fn cmap_signed(s: f32) -> vec3f {
  let a = clamp(abs(s), 0.0, 1.0);
  let neg = vec3f(0.230, 0.299, 0.754);
  let pos = vec3f(0.706, 0.016, 0.150);
  return mix(vec3f(0.865), select(neg, pos, s >= 0.0), a);
}

fn cmap_apply(cmap: u32, t: f32) -> vec3f {
  switch (cmap) {
    case 1u: { return cmap_turbo(t); }
    case 2u: { return cmap_signed(2.0 * t - 1.0); }
    case 3u: { return vec3f(clamp(t, 0.0, 1.0)); }
    default: { return cmap_viridis(t); }
  }
}

/// Categorical colour for integer codes; DBG_CODE_NONE -> black.
fn cmap_code(code: u32) -> vec3f {
  if (code == DBG_CODE_NONE) { return vec3f(0.0); }
  let h = pcg3d(vec3u(code, 0x2545f491u, 0x9e3779b9u));
  return vec3f(0.15) + 0.85 * vec3f(f32(h.x & 255u), f32(h.y & 255u), f32(h.z & 255u)) / 255.0;
}

/// Normalise a scalar to [0,1] with range/log/abs/signed handling. Signed maps put rangeMin..rangeMax
/// symmetric around 0 (t = 0.5 at v = 0) using max(|min|, |max|).
fn debug_normalize(v_in: f32, rmin: f32, rmax: f32, flags: u32) -> f32 {
  var v = v_in;
  if ((flags & DBGF_ABS) != 0u) { v = abs(v); }
  let cmap = (flags >> DBGF_CMAP_SHIFT) & DBGF_CMAP_MASK;
  if (cmap == CMAP_SIGNED) {
    let m = max(max(abs(rmin), abs(rmax)), 1e-30);
    if ((flags & DBGF_LOG) != 0u) {
      // symmetric log: sign(v) * log10(1 + |v|/lo) / log10(1 + m/lo), lo = max(|rmin| min, m*1e-6)
      let lo = max(m * 1e-6, 1e-30);
      return 0.5 + 0.5 * sign(v) * log2(1.0 + abs(v) / lo) / log2(1.0 + m / lo);
    }
    return 0.5 + 0.5 * clamp(v / m, -1.0, 1.0);
  }
  if ((flags & DBGF_LOG) != 0u) {
    let lo = log2(max(rmin, 1e-30));
    let hi = log2(max(rmax, 1e-30));
    return (log2(max(v, 1e-30)) - lo) / max(hi - lo, 1e-30);
  }
  return (v - rmin) / select(rmax - rmin, 1e-30, rmax == rmin);
}

/// Map a raw AOV value to a display-encoded colour for the given kind.
fn debug_false_colour(v: vec4f, kind: u32, rmin: f32, rmax: f32, flags: u32) -> vec3f {
  let cmap = (flags >> DBGF_CMAP_SHIFT) & DBGF_CMAP_MASK;
  if (kind == DBG_KIND_CODE) { return cmap_code(bitcast<u32>(v.x)); }
  if ((flags & DBGF_NONFINITE) != 0u) {
    let n = select(3u, 1u, kind == DBG_KIND_SCALAR);
    for (var i = 0u; i < n; i++) {
      if (is_nan(v[i])) { return vec3f(1.0, 0.0, 1.0); }
      if (is_inf(v[i])) { return vec3f(0.0, 1.0, 1.0); }
    }
  }
  if (kind == DBG_KIND_VEC3) {
    var o = vec3f(0.0);
    // per-channel remap (signed colormap -> symmetric around 0); no colormap for vectors
    for (var i = 0u; i < 3u; i++) { o[i] = clamp(debug_normalize(v[i], rmin, rmax, flags), 0.0, 1.0); }
    return o;
  }
  return cmap_apply(cmap, debug_normalize(v.x, rmin, rmax, flags));
}

// Prefix replay of an offset path (restir-api.md §3.6; math.md#offset-checks O0/O1/O3, gap-rc §6.2/§9.1).
// OWNER WP-B. P0 STUB (restir-api.md §1.4): replay_prefix returns SC_O0_MISS, so every replayed shift is FAILED.
#include "restir/rc.wgsl"
#include "restir/frame.wgsl"

struct ReplayOut {
  code: u32,             // SC_OK | SC_O0_MISS | SC_O0_TECH | SC_O1 | SC_O3 | SC_ZERO (+ term, pair, margin packed as §2.6)
  margin: f32,
  yLast: SurfaceHit,     // prefix mode: y_{k−1};   ∅ mode: unused
  yLastPrim: u32,
  VLast: vec3f,          // incoming direction at y_{k−1} (from positions)
  Tp: vec3f,             // ∏ over replayed samples (prefix: y₁…y_{k−2}; ∅: y₁…y_{d−1}), RR-free
  preValid: bool,        // a pre-rc pair (y_{k−2}, y_{k−1}) exists (k−1 ≥ 2)
  preV: RcVertex, preE: RcEvent,   // y_{k−2} and its replayed event
  F: vec3f,              // ∅ mode: the replayed path's own integrand incl. ω2
}

/// kIdx = k (prefix mode, 3 ≤ k ≤ d) or 0 (∅ mode, d ≥ 2). STUB: always SC_O0_MISS.
fn replay_prefix(seed: vec2u, y1: SurfaceHit, y1Prim: u32, camPos: vec3f, thr: f32, kIdx: u32, d: u32, tech: u32) -> ReplayOut {
  var r: ReplayOut;
  r.code = rs_slot_code(SC_O0_MISS, RCT_NONE, 0u, 0.0);
  r.yLast = y1;
  r.yLastPrim = y1Prim;
  return r;
}

// Temporal pixel q′ (restir-temporal-api.md §3.3, TD12; math.md#temporal [M5 addition] "q′ rule"). OWNER T-A.
// Back-projection of the current jittered primary hit x₁(q) (rebuilt from the V-buffer ids) with the previous camera,
// stochastic rounding with ξ from STREAM_TEMPORAL_PICK, then the centre tap and the 8-ring (fixed order rotated by the
// pick hash); first tap with a previous hit, dot(n^g, n^g′) ≥ 0.5 and |‖x₁ − o_{t−1}‖ − z′| ≤ 0.1·z′ in the same
// member wins. A function of the G-buffers and the pick stream only (sample-independent).
// P0 STUB: always invalid (tap 9) ⇒ no temporal candidate (T3 leaves the canonical unchanged; unbiased).
#include "restir/tframe.wgsl"

struct TPick { valid: bool, ai: u32, local: vec2u, sp: vec2f, tap: u32 }   // tap 0 centre, 1…8 ring, 9 none

fn tpick_none() -> TPick { return TPick(false, TS_QPRIME_NONE, vec2u(0u), vec2f(-1.0), 9u); }

fn temporal_pixel(p: RsPix, key: vec2u) -> TPick {
  return tpick_none();
}

// Global light selection: integer alias table with u16 thresholds and the REALIZED pmf (plan §1.4 "Selection";
// math.md#light-selection, math.md open item 5; CPU mirror alias.ts aliasSample / realizedPmf).
//   bucket b = hSel >> (32 − m)                  m = log2 n ≥ 1 (never a shift by 32: WGSL shifts are mod 32)
//   t        = (m ≤ 16 ? hSel : hSel2) & 0xFFFF   (for m ≤ 16 the bucket bits and the threshold bits are disjoint)
//   entry    = (t < q_b) ? b : alias_b
// The pmf used anywhere is pmf[entry], computed on the CPU from the stored integers; it never depends on the shading
// point, the pixel or the path.
#include "lights/lights.wgsl"

fn alias_sample(slot: LightSlot, hSel: u32, hSel2: u32) -> u32 {
  let m = slot.aliasLog2;
  let b = hSel >> (32u - m);
  let t = select(hSel, hSel2, m > 16u) & 0xffffu;
  let e = slot.aliasOff + 2u * b;
#if RS_LIGHT_REC4
  let q = records[e >> 2u];                           // perf2 WP-4a: aliasOff is 16 B-aligned, so (q_b, alias_b) is
  let qa = select(q.xy, q.zw, (e & 2u) != 0u);        // one half of one vec4
  return select(qa.y, b, t < qa.x);
#else
  return select(records[e + 1u], b, t < records[e]);
#endif
}

/// Realized selection probability of alias entry `entry`.
fn light_pmf(slot: LightSlot, entry: u32) -> f32 {
#if RS_LIGHT_REC4
  return bitcast<f32>(rec_u32(slot.pmfOff + entry));
#else
  return bitcast<f32>(records[slot.pmfOff + entry]);
#endif
}

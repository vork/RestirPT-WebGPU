// Shift arena binding, arena counters and the replay work queue (restir-api.md §2.6, §2.7, §3.9 and appendix B.4;
// PLAN §1.8 "queues"). P0 real module (owner WP-C, who may refine it). Defines: RS_ARENA_BINDING ('<n>u' string,
// Changelog A4) declares the arena; RS_ARENA_RW selects the read_write (atomic header) view.
//   header words 0–15: 4 queue headers {counter (atomic), n, capacity, overflow (atomic)} (q0 spatial replay;
//   q1/q2 M5; q3 reserved); 16–63 RSC_* counters. words[] (after the header): slots 4·(ai·NS + s) … +3,
//   codes 4·P·NS + ai·NS + s, items 5·P·NS + i.
#include "restir/frame.wgsl"

struct ShiftArenaRW { hdr: array<atomic<u32>, 64>, words: array<u32> }
struct ShiftArenaRO { hdr: array<u32, 64>, words: array<u32> }

fn rs_atlas_pixels() -> u32 { return rsParams.atlasSize.x * rsParams.atlasSize.y; }
/// Queue capacity (every queue is worst-case sized: one item per (pixel, slot)).
fn queue_capacity() -> u32 { return rs_atlas_pixels() * rsParams.numSlots; }
fn slot_index(ai: u32, s: u32) -> u32 { return ai * rsParams.numSlots + s; }
fn arena_slot_word(ai: u32, s: u32) -> u32 { return 4u * slot_index(ai, s); }
fn arena_code_word(ai: u32, s: u32) -> u32 { return 4u * queue_capacity() + slot_index(ai, s); }
fn arena_item_word(i: u32) -> u32 { return 5u * queue_capacity() + i; }

#if RS_ARENA_BINDING
#if RS_ARENA_RW
@group(2) @binding($RS_ARENA_BINDING) var<storage, read_write> rsArena: ShiftArenaRW;
fn rs_hdr(w: u32) -> u32 { return atomicLoad(&rsArena.hdr[w]); }
/// Add n to arena counter c (RSC_*).
fn rs_count(c: u32, n: u32) { if (n != 0u) { atomicAdd(&rsArena.hdr[c], n); } }
/// Append one item to queue q (one atomicAdd per item, D18); past capacity only the overflow flag is set.
fn queue_append(q: u32, item: u32) {
  let i = atomicAdd(&rsArena.hdr[4u * q], 1u);
  if (i >= queue_capacity()) { atomicStore(&rsArena.hdr[4u * q + 3u], 1u); return; }
  rsArena.words[arena_item_word(i)] = item;
}
#else
@group(2) @binding($RS_ARENA_BINDING) var<storage, read> rsArena: ShiftArenaRO;
fn rs_hdr(w: u32) -> u32 { return rsArena.hdr[w]; }
#endif
fn arena_word(w: u32) -> u32 { return rsArena.words[w]; }
/// Item index of a queue consumer thread (2D indirect args, §2.7), or 0xFFFFFFFF past hdr.n.
fn queue_item(q: u32, wid: vec3u, nwg: vec3u, lid: u32) -> u32 {
  let item = (wid.y * nwg.x + wid.x) * RS_WG + lid;
  return select(0xFFFFFFFFu, item, item < rs_hdr(4u * q + 1u));
}
#endif

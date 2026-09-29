// Shift arena binding, arena counters and the replay work queue (restir-api.md §2.6, §2.7, §3.9 and appendix B.4;
// PLAN §1.8 "queues"). OWNER WP-C. Defines: RS_ARENA_BINDING ('<n>u' string, Changelog A4) declares the arena;
// RS_ARENA_RW selects the read_write (atomic header) view.
//   header words 0–15: 4 queue headers {counter (atomic), n, capacity, overflow (atomic)} (q0 spatial replay;
//   q1/q2 M5; q3 reserved); 16–63 RSC_* counters. words[] (after the header): slots 4·(ai·NS + s) … +3,
//   codes 4·P·NS + ai·NS + s, items 5·P·NS + i.
// Queue protocol (§2.7, Changelog C1): per use, clear {counter, n} (8 bytes; overflow is sticky until the counters
// are read with reset); producers queue_append (one atomicAdd per item, D18; past capacity only the overflow flag is
// set); rs_args writes the 2D indirect args of n = min(counter, capacity), hdr.n, hdr.capacity and the overflow flag;
// consumers (@workgroup_size(64), 2D indirect dispatch) take item index queue_item(...) and exit past hdr.n.
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

/// Items of the chunk [base, base + count) of a queue holding n items (count 0 = unbounded).
fn queue_chunk_n(n: u32, base: u32, count: u32) -> u32 {
  let rest = select(0u, n - base, n > base);
  return select(min(rest, count), rest, count == 0u);
}
/// 2D indirect args of n items (§2.7): (min(g, 65535), ceil(g / 65535), 1), g = ceil(n / 64); n = 0 → (0, 1, 1).
/// A 1D dispatch would silently no-op above 65535·64 = 4.19 M items (PLAN §1.8).
fn queue_args(n: u32) -> vec3u {
  let g = n / RS_WG + select(0u, 1u, (n % RS_WG) != 0u);
  return vec3u(min(g, 65535u), max(g / 65535u + select(0u, 1u, (g % 65535u) != 0u), 1u), 1u);
}
/// Queue item word = (ai << 3) | slot (§2.7).
fn queue_item_word(ai: u32, s: u32) -> u32 { return (ai << 3u) | s; }
fn queue_item_ai(w: u32) -> u32 { return w >> 3u; }
fn queue_item_slot(w: u32) -> u32 { return w & 7u; }

/// J-word predicates (D7, §2.6; integer compares only).
fn jw_accepted(jw: u32) -> bool { return jw != JW_NOT_ACCEPTED; }
/// VALID: the f32 bits of a finite J > 0 (not FAILED, PENDING or NOT_ACCEPTED).
fn jw_valid(jw: u32) -> bool { return jw != JW_FAILED && jw < 0x7f800000u; }

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
/// Write slot (ai, s): (FJ.rgb, J word) and its packed code word (§2.6).
fn arena_slot_write(ai: u32, s: u32, FJ: vec3f, jw: u32, code: u32) {
  let w = arena_slot_word(ai, s);
  rsArena.words[w] = bitcast<u32>(FJ.x);
  rsArena.words[w + 1u] = bitcast<u32>(FJ.y);
  rsArena.words[w + 2u] = bitcast<u32>(FJ.z);
  rsArena.words[w + 3u] = jw;
  rsArena.words[arena_code_word(ai, s)] = code;
}
/// Store a shift result (shift.wgsl ShiftOut fields) into slot (ai, s): VALID (FJ, bits(J)) for SC_OK with a finite
/// J > 0 and finite FJ, else FAILED with FJ = 0. An SC_OK result with non-finite values becomes SC_NONFINITE.
/// Returns the final SC_* code.
fn arena_slot_store_shift(ai: u32, s: u32, FJ: vec3f, J: f32, code: u32) -> u32 {
  var c = code;
  var sc = rs_slot_code_sc(code);
  if (sc == SC_OK && !(rs_pos_finite(J) && all_finite3(FJ))) {
    sc = SC_NONFINITE;
    c = rs_slot_code(SC_NONFINITE, RCT_NONE, 0u, 0.0);
  }
  if (sc == SC_OK) { arena_slot_write(ai, s, FJ, bitcast<u32>(J), c); }
  else { arena_slot_write(ai, s, vec3f(0.0), JW_FAILED, c); }
  return sc;
}
#else
@group(2) @binding($RS_ARENA_BINDING) var<storage, read> rsArena: ShiftArenaRO;
fn rs_hdr(w: u32) -> u32 { return rsArena.hdr[w]; }
#endif
fn arena_word(w: u32) -> u32 { return rsArena.words[w]; }
/// Slot (ai, s) as (bits(FJ.r), bits(FJ.g), bits(FJ.b), J word).
fn arena_slot(ai: u32, s: u32) -> vec4u {
  let w = arena_slot_word(ai, s);
  return vec4u(rsArena.words[w], rsArena.words[w + 1u], rsArena.words[w + 2u], rsArena.words[w + 3u]);
}
fn arena_slot_jword(ai: u32, s: u32) -> u32 { return rsArena.words[arena_slot_word(ai, s) + 3u]; }
/// Item index of a queue consumer thread (2D indirect args, §2.7), or 0xFFFFFFFF past hdr.n.
fn queue_item(q: u32, wid: vec3u, nwg: vec3u, lid: u32) -> u32 {
  let item = (wid.y * nwg.x + wid.x) * RS_WG + lid;
  return select(0xFFFFFFFFu, item, item < rs_hdr(4u * q + 1u));
}
/// Chunked consumer (Changelog C7): item base + local index of the chunk [base, base + count) (count 0 = unbounded),
/// or 0xFFFFFFFF past the chunk or past hdr.n. The chunk's args come from rs_args with the same (base, count).
fn queue_item_chunk(q: u32, wid: vec3u, nwg: vec3u, lid: u32, base: u32, count: u32) -> u32 {
  let local = (wid.y * nwg.x + wid.x) * RS_WG + lid;
  let item = base + local;
  let inChunk = count == 0u || local < count;
  return select(0xFFFFFFFFu, item, inChunk && item < rs_hdr(4u * q + 1u));
}
#endif

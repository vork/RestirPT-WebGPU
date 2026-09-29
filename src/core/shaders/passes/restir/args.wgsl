// rs_args (restir-api.md §2.7, §4.1; PLAN §1.8 "2D indirect args"): one thread turns the replay queue q0's counter
// into 2D indirect dispatch args (min(g, 65535), ceil(g / 65535), 1), g = ceil(n / 64), n = min(counter, capacity),
// and writes hdr.n, hdr.capacity and the (sticky) overflow flag (Changelog C1: the arena is bound read_write here).
// Chunked replay (Changelog C7): RsDispatch.treeBase / treeCount = item base / count of the consumer dispatch that
// follows (count 0 = the whole queue); the args then cover that chunk only, hdr.n stays the whole queue.
// G2: 0 shiftArena rw · 1 rsArgs rw (STORAGE | INDIRECT; never bound as storage by the consumer).
#include "restir/frame.wgsl"
#include "restir/queue.wgsl"

@group(2) @binding($RS_ARGS_BINDING) var<storage, read_write> rsArgs: array<u32, 16>;

@compute @workgroup_size(1)
fn rs_args() {
  let q = 0u;
  let counter = atomicLoad(&rsArena.hdr[4u * q]);
  let cap = queue_capacity();
  let n = min(counter, cap);
  atomicStore(&rsArena.hdr[4u * q + 1u], n);
  atomicStore(&rsArena.hdr[4u * q + 2u], cap);
  if (counter > cap) { atomicStore(&rsArena.hdr[4u * q + 3u], 1u); }
  let a = queue_args(queue_chunk_n(n, rsDispatch.treeBase, rsDispatch.treeCount));
  rsArgs[4u * q] = a.x;
  rsArgs[4u * q + 1u] = a.y;
  rsArgs[4u * q + 2u] = a.z;
  rsArgs[4u * q + 3u] = 0u;
}

// Unified path tree with streaming RIS (restir-api.md §3.5; math.md#path-tree, #rc-predicate, #rr).
// P0 STUB (restir-api.md §1.4): writes an empty reservoir (rung 3.1 = L1 only). Replaced by WP-A (A1).
#include "restir/reservoir.wgsl"
#include "restir/endpoint.wgsl"
#include "restir/rc.wgsl"
#include "restir/queue.wgsl"
#include "debug/restir-views.wgsl"

#if RS_DUMP_CANDIDATES
// Candidate dump (tests only, §2.12): per atlas pixel RS_DUMP_CAP records of RS_DUMP_WORDS words, then one count per
// pixel at word RS_DUMP_CAP·RS_DUMP_WORDS·P + ai.
@group(2) @binding(4) var<storage, read_write> candDump: array<u32>;
#endif

fn pathtree_run(p: RsPix, key: vec2u, treeBase: u32, treeCount: u32, firstChunk: bool, finalChunk: bool) {
  if (firstChunk) { res_write_empty(p.ai, key, false); }
  if (finalChunk) { rsdbg_reservoir(p.px, p.ai, 1u); }
}

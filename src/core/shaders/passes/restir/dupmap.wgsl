// rs_dupmap (restir-m6-api.md MD10, §3; PLAN §3 pass 6, rule 13; math.md#dupmap; Enhanced §5): per atlas pixel p, the
// number of pixels q ≠ p in the 17×17 window (same ensemble member, inside its tile) whose final reservoir carries the
// same 64-bit path seed as p's, both non-empty (empty / background reservoirs carry no sample id). The count is stored
// in the arena's M6 region (words[] rs_dup_base() + ai) and read by the NEXT frame's T1 at q′ (RS_DUPMAP variant), which
// lowers the temporal confidence cap: c_Cap = cCap − (cCap − 1)·(count/288)^0.1. BIASED; off in every unbiasedness gate.
// 16×16 workgroups; a 32×32 shared tile of (seed.x, seed.y, valid) covers the window of every thread. Scene-free.
// G2: 0 resIn = res[final] ro · 1 shiftArena rw.
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"
#include "restir/queue.wgsl"
#include "restir/m6-types.wgsl"

var<workgroup> dupSeeds: array<vec2u, 1024>;
var<workgroup> dupOk: array<u32, 1024>;

/// rs_pix without the row-band test (the window reads neighbours outside the dispatch's band).
fn rs_pix_any(px: vec2u) -> RsPix {
  var p: RsPix;
  p.px = px;
  let W = rsParams.memberSize.x;
  let H = rsParams.memberSize.y;
  let mcol = px.x / W;
  let mrow = px.y / H;
  let m = mrow * rsParams.memberCols + mcol;
  p.valid = px.x < rsParams.atlasSize.x && px.y < rsParams.atlasSize.y && mcol < rsParams.memberCols && m < rsParams.memberCount;
  p.member = rsParams.memberBase + m;
  p.local = vec2u(px.x - mcol * W, px.y - mrow * H);
  p.localIdx = p.local.y * W + p.local.x;
  p.ai = px.y * rsParams.atlasSize.x + px.x;
  return p;
}

@compute @workgroup_size(16, 16, 1)
fn rs_dupmap(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let px = gid.xy;
  let pix = rs_pix_any(px);
  let org = vec2i(wid.xy * 16u) - vec2i(i32(DUP_HALF));     // tile texel (0, 0) = atlas (16·wid − 8)
  // each thread loads 2×2 texels of the 32×32 tile (the member test uses the CENTRE pixel's member below)
  for (var k = 0u; k < 4u; k++) {
    let tx = lid.x * 2u + (k & 1u);
    let ty = lid.y * 2u + (k >> 1u);
    let a = org + vec2i(i32(tx), i32(ty));
    var v = vec3u(0u);
    if (a.x >= 0 && a.y >= 0 && u32(a.x) < rsParams.atlasSize.x && u32(a.y) < rsParams.atlasSize.y) {
      let q = rs_pix_any(vec2u(a));
      if (q.valid) {
        let p1 = resin_plane(q.ai, RP_SEED);
        if (!res_empty(p1.z) && (p1.z & RF_BG) == 0u) { v = vec3u(p1.xy, q.member + 1u); }
      }
    }
    dupSeeds[ty * 32u + tx] = v.xy;
    dupOk[ty * 32u + tx] = v.z;                                 // member id + 1 (0 = no sample id)
  }
  workgroupBarrier();
  if (!pix.valid) { return; }
  let c = vec2u(lid.x + DUP_HALF, lid.y + DUP_HALF);
  let me = dupOk[c.y * 32u + c.x];
  var n = 0u;
  if (me != 0u) {
    let sd = dupSeeds[c.y * 32u + c.x];
    for (var dy = 0u; dy <= 2u * DUP_HALF; dy++) {
      for (var dx = 0u; dx <= 2u * DUP_HALF; dx++) {
        if (dx == DUP_HALF && dy == DUP_HALF) { continue; }
        let i = (lid.y + dy) * 32u + lid.x + dx;
        n += select(0u, 1u, dupOk[i] == me && all(dupSeeds[i] == sd));
      }
    }
  }
  rsArena.words[rs_dup_base() + pix.ai] = n;
}

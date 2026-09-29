// Ensemble statistics (restir-api.md §2.10, D14/D15; PLAN §3 "Ensemble mode"): per-run (member) tile sums of this
// frame's rsFrame and per-pixel Σx / Σx² over members and frames. Deterministic fixed-order reductions only (no float
// atomics), so equal inputs give bitwise-equal sums on every run.
//   rs_ensemble_stats  @workgroup_size(16, 16), dispatch (⌈W/16⌉, ⌈H/16⌉·E): one workgroup per (member, 16² tile):
//                      shared-memory tree reduction → tiles16[m][ty][tx]; workgroups of member 0 also add, per
//                      member-local pixel, Σ_m x and Σ_m x² (members in index order) to ensPixel.
//   rs_ensemble_reduce @workgroup_size(64), 1D over E·(n32 + n64 + 1 + 16) entries (≪ 65535·64): tiles32/64 from the
//                      16² tile sums by fixed 2×2 trees, global[m] by pairwise (cascade) summation of the 64² tiles,
//                      mask[m][·] = 0 (mask regions are not used in M4: M = 0, layout reserved).
// ensStats (vec4, w = 0) = [ℓ ∈ {16, 32, 64}][m][ty][tx] · global[m] · mask[m][16]; ensPixel = per member-local pixel
// (Σx.rgb, 0), (Σx².rgb, 0), accumulated over the frames of a batch (the runner reads and clears it per batch).
// G2: 0 ensStats rw · 1 ensPixel rw · 2 rsFrame · 3 maskTex.
#include "restir/frame.wgsl"

@group(2) @binding($RS_ENS_STATS_BINDING) var<storage, read_write> ensStats: array<vec4f>;
@group(2) @binding($RS_ENS_PIXEL_BINDING) var<storage, read_write> ensPixel: array<vec4f>;
@group(2) @binding($RS_MASK_BINDING) var maskTex: texture_2d<u32>;

const ENS_MASK_REGIONS: u32 = 16u;

fn ens_tiles(l: u32) -> vec2u {
  let W = rsParams.memberSize.x;
  let H = rsParams.memberSize.y;
  return vec2u((W + l - 1u) / l, (H + l - 1u) / l);
}
fn ens_count(l: u32) -> u32 { let t = ens_tiles(l); return rsParams.memberCount * t.x * t.y; }
fn ens_off16() -> u32 { return 0u; }
fn ens_off32() -> u32 { return ens_count(16u); }
fn ens_off64() -> u32 { return ens_off32() + ens_count(32u); }
fn ens_off_global() -> u32 { return ens_off64() + ens_count(64u); }
fn ens_off_mask() -> u32 { return ens_off_global() + rsParams.memberCount; }

/// Atlas pixel of member m's local pixel.
fn ens_atlas_px(m: u32, local: vec2u) -> vec2u {
  let W = rsParams.memberSize.x;
  let H = rsParams.memberSize.y;
  return vec2u((m % rsParams.memberCols) * W + local.x, (m / rsParams.memberCols) * H + local.y);
}

var<workgroup> ensRed: array<vec3f, 256>;

@compute @workgroup_size(16, 16, 1)
fn rs_ensemble_stats(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let W = rsParams.memberSize.x;
  let H = rsParams.memberSize.y;
  let E = rsParams.memberCount;
  let th = ens_tiles(16u).y;
  let m = wid.y / th;
  let tile = vec2u(wid.x, wid.y % th);
  let local = tile * 16u + lid.xy;
  let inside = local.x < W && local.y < H && m < E;
  var x = vec3f(0.0);
  if (inside) { x = textureLoad(rsFrame, ens_atlas_px(m, local), 0).rgb; }
  ensRed[li] = x;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (li < stride) { ensRed[li] = ensRed[li] + ensRed[li + stride]; }
    workgroupBarrier();
  }
  if (li == 0u && m < E) {
    let t = ens_tiles(16u);
    ensStats[ens_off16() + (m * t.y + tile.y) * t.x + tile.x] = vec4f(ensRed[0], 0.0);
  }
  // Per-pixel moments over members (index order) of this frame, added to the batch accumulators.
  if (m == 0u && inside) {
    var s1 = vec3f(0.0);
    var s2 = vec3f(0.0);
    for (var mm = 0u; mm < E; mm++) {
      let v = textureLoad(rsFrame, ens_atlas_px(mm, local), 0).rgb;
      s1 += v;
      s2 += v * v;
    }
    let i = 2u * (local.y * W + local.x);
    ensPixel[i] = ensPixel[i] + vec4f(s1, 0.0);
    ensPixel[i + 1u] = ensPixel[i + 1u] + vec4f(s2, 0.0);
  }
}

fn ens_t16(m: u32, ty: u32, tx: u32) -> vec3f {
  let t = ens_tiles(16u);
  if (ty >= t.y || tx >= t.x) { return vec3f(0.0); }
  return ensStats[ens_off16() + (m * t.y + ty) * t.x + tx].xyz;
}
/// 32² tile (ty, tx) of member m from its 2×2 block of 16² tiles, fixed tree ((00 + 01) + (10 + 11)).
fn ens_t32(m: u32, ty: u32, tx: u32) -> vec3f {
  return (ens_t16(m, 2u * ty, 2u * tx) + ens_t16(m, 2u * ty, 2u * tx + 1u))
       + (ens_t16(m, 2u * ty + 1u, 2u * tx) + ens_t16(m, 2u * ty + 1u, 2u * tx + 1u));
}
/// 64² tile from its 2×2 block of 32² tiles (the same tree as the stored 32² sums).
fn ens_t64(m: u32, ty: u32, tx: u32) -> vec3f {
  return (ens_t32(m, 2u * ty, 2u * tx) + ens_t32(m, 2u * ty, 2u * tx + 1u))
       + (ens_t32(m, 2u * ty + 1u, 2u * tx) + ens_t32(m, 2u * ty + 1u, 2u * tx + 1u));
}

@compute @workgroup_size(64)
fn rs_ensemble_reduce(@builtin(global_invocation_id) gid: vec3u) {
  let E = rsParams.memberCount;
  var i = gid.x;
  let t32 = ens_tiles(32u);
  let t64 = ens_tiles(64u);
  let n32 = E * t32.x * t32.y;
  let n64 = E * t64.x * t64.y;
  if (i < n32) {
    let per = t32.x * t32.y;
    let m = i / per;
    let r = i % per;
    ensStats[ens_off32() + i] = vec4f(ens_t32(m, r / t32.x, r % t32.x), 0.0);
    return;
  }
  i -= n32;
  if (i < n64) {
    let per = t64.x * t64.y;
    let m = i / per;
    let r = i % per;
    ensStats[ens_off64() + i] = vec4f(ens_t64(m, r / t64.x, r % t64.x), 0.0);
    return;
  }
  i -= n64;
  if (i < E) {
    // Cascade (pairwise) summation of the member's 64² tiles in row-major order: a fixed binary tree.
    var stack: array<vec3f, 24>;
    var sp = 0u;
    let n = t64.x * t64.y;
    for (var j = 0u; j < n; j++) {
      var v = ens_t64(i, j / t64.x, j % t64.x);
      var c = j + 1u;
      while ((c & 1u) == 0u && sp > 0u) { sp -= 1u; v = stack[sp] + v; c = c >> 1u; }
      stack[sp] = v;
      sp += 1u;
    }
    var g = vec3f(0.0);
    while (sp > 0u) { sp -= 1u; g = stack[sp] + g; }
    ensStats[ens_off_global() + i] = vec4f(g, 0.0);
    return;
  }
  i -= E;
  if (i < E * ENS_MASK_REGIONS) { ensStats[ens_off_mask() + i] = vec4f(0.0); }
}

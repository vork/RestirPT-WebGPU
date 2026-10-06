// rs_light_tiles (restir-m6-api.md MD4, §3; PLAN §2 rule 2; Enhanced §6.1; gap-light §3.9): per frame and ensemble member,
// RS_TILES × RS_TILE_SIZE (128 × 1024) i.i.d. draws of the frame's global alias table (the env entry included), the
// presampled candidate pool of RIS-NEE at x₁. Entry (m, tile, slot) at words[] tilesBase + (m·128 + tile)·1024 + slot =
// alias_sample(lightsParams.cur, h.x, h.y), h = pcg4d(runSeed ⊕ member·φ, t, tile·1024 + slot, STREAM_LIGHT_TILE);
// LIGHT_NONE when the frame has no light entries. Only the MARGINAL pmf[L] of a tile entry is ever used (rule 2).
// Dispatch (RS_TILE_SIZE / 64, RS_TILES·E) workgroups of 64 (gid.y = tile + 128·member). G1 empty, G2: 0 shiftArena rw.
#include "restir/frame.wgsl"
#include "restir/queue.wgsl"
#include "restir/m6-types.wgsl"
#include "lights/select.wgsl"

@compute @workgroup_size(64)
fn rs_light_tiles(@builtin(global_invocation_id) gid: vec3u) {
  let slot = gid.x;
  let tile = gid.y % RS_TILES;
  let m = gid.y / RS_TILES;
  if (slot >= RS_TILE_SIZE || m >= rsParams.memberCount) { return; }
  let member = rsParams.memberBase + m;
  let h = pcg4d(vec4u(frame.runSeed ^ (member * 0x9e3779b9u), rs_t(), tile * RS_TILE_SIZE + slot, STREAM_LIGHT_TILE));
  let ls = lightsParams.cur;
  var e = LIGHT_NONE;
  if (ls.nEntries != 0u) { e = alias_sample(ls, h.x, h.y); }
  rsArena.words[rs_tiles_base() + (m * RS_TILES + tile) * RS_TILE_SIZE + slot] = e;
}

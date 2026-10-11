#include "denoise/dn-common.wgsl"
#include "passes/gbuffer.wgsl"
#include "denoise/dn-world-common.wgsl"
#if GBUF_48
@group(1) @binding(0) var<storage, read> gbuf: array<GBufStore>;
#else
@group(1) @binding(0) var<storage, read> gbuf: array<GBufTexel>;
#endif
@group(1) @binding(1) var localCache: texture_2d<f32>;
@group(1) @binding(2) var guide: texture_2d<f32>;
@group(1) @binding(3) var<storage, read> previous: array<GiWorldEntry>;
@group(1) @binding(4) var<storage, read_write> next: array<GiWorldEntry>;
@group(1) @binding(5) var<storage, read_write> winners: array<atomic<u32>>;
#if DN_WORLD_SELECT
@compute @workgroup_size(8, 8)
fn dn_world_select(@builtin(global_invocation_id) id: vec3u) {
  let p = id.xy;
  if (any(p >= dn.size)) { return; }
  let i = p.y * dn.size.x + p.x;
  let g = gbuf[i];
  if ((g.flags & GB_HIT) == 0u || textureLoad(localCache, p, 0).a < 8.0) { return; }
  if (any(textureDimensions(guide) != vec2u(1u)) && !any(textureLoad(guide, p, 0).rgb > vec3f(0.02))) { return; }
  // Deterministic selection: no floating atomics and no simultaneous reads/writes of an entry.
  atomicMax(&winners[gi_world_index(g.pos + frame.origin, g.ns)], i + 1u);
}
#else
@compute @workgroup_size(64)
fn dn_world_update(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= 32768u) { return; }
  let selected = atomicLoad(&winners[id.x]);
  if (selected == 0u) { return; }
  let i = selected - 1u;
  let p = vec2u(i % dn.size.x, i / dn.size.x);
  let g = gbuf[i];
  let pos = g.pos + frame.origin;
  let old = previous[id.x];
  var value = textureLoad(localCache, p, 0).rgb;
  var count = 1.0;
  if (gi_world_valid(old, pos, g.ns)) {
    count = min(old.count + 1.0, 16.0);
    value = mix(old.irradiance, value, max(1.0 / count, 0.15));
  }
  next[id.x] = GiWorldEntry(gi_world_cell(pos), dn.worldEpoch, pos, dn.worldStep, g.ns, count, value, 0.0);
}
#endif

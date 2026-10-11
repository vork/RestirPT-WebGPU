// Bounded, direct-mapped surface irradiance cache. Full tags and geometry checks reject hash collisions.
// A world position (including the renderer's floating origin) and normal, not a screen pixel, own each entry.
struct GiWorldEntry {
  cell: vec3i, epoch: u32,
  position: vec3f, step: u32,
  normal: vec3f, count: f32,
  irradiance: vec3f, pad: f32,
}
fn gi_world_size() -> f32 { return clamp(frame.sceneDiag / 128.0, 0.02, 0.5); }
fn gi_world_cell(pos: vec3f) -> vec3i { return vec3i(floor(pos / gi_world_size())); }
fn gi_world_index(pos: vec3f, n: vec3f) -> u32 {
  let an = abs(n);
  var axis = 2u;
  if (an.x >= an.y && an.x >= an.z) { axis = 0u; } else if (an.y >= an.z) { axis = 1u; }
  let face = 2u * axis + select(0u, 1u, n[axis] < 0.0);
  let cell = bitcast<vec3u>(gi_world_cell(pos));
  return pcg3d(cell + vec3u(face * 0x9e3779b9u)).x & 32767u;
}
fn gi_world_valid(e: GiWorldEntry, pos: vec3f, n: vec3f) -> bool {
  let delta = e.position - pos;
  return e.epoch == dn.worldEpoch && e.count > 0.0 && dn.worldStep - e.step <= 120u
    && all(e.cell == gi_world_cell(pos)) && dot(e.normal, n) > 0.95
    && max(abs(dot(delta, n)), abs(dot(delta, e.normal))) < gi_world_size() * 0.1;
}

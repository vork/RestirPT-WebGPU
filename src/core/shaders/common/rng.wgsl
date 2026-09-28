// Counter-based RNG (plan §1.7): u = hash(seed, dim) so random replay can skip work without drifting.
// pcg3d (Jarzynski & Olano, JCGT 2020). Floats from the top 24 bits -> [0, 1) exactly.
fn pcg3d(v_in: vec3u) -> vec3u {
  var v = v_in * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> vec3u(16u);
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}
fn u32_to_unit(x: u32) -> f32 { return f32(x >> 8u) * (1.0 / 16777216.0); }
/// One uniform number for (seed, dim, stream). `stream` separates path / resampling / jitter streams.
fn rand1(seed: u32, dim: u32, stream: u32) -> f32 { return u32_to_unit(pcg3d(vec3u(seed, dim, stream)).x); }
fn rand2(seed: u32, dim: u32, stream: u32) -> vec2f {
  let h = pcg3d(vec3u(seed, dim, stream));
  return vec2f(u32_to_unit(h.x), u32_to_unit(h.y));
}
const STREAM_PATH: u32 = 0x9e3779b9u;
const STREAM_RESAMPLE: u32 = 0x85ebca6bu;
const STREAM_JITTER: u32 = 0xc2b2ae35u;

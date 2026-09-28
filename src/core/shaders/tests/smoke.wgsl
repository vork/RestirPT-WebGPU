#include "common/math.wgsl"
#include "common/nan.wgsl"
#include "common/rng.wgsl"

struct Params { a: f32, b: f32, c: f32, zero: f32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> outU: array<u32>;
@group(0) @binding(2) var<storage, read_write> outF: array<f32>;

@compute @workgroup_size($WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= $N) { return; }
  // Deterministic RNG stream values
  outU[i] = pcg3d(vec3u(i, 7u, STREAM_PATH)).x;
  outF[i] = rand1(i, 3u, STREAM_RESAMPLE);
  if (i == 0u) {
    let nanv = P.zero / P.zero;       // runtime NaN
    let infv = 1.0 / P.zero;          // runtime +Inf
    outU[$N + 0u] = select(0u, 1u, is_nan(nanv));
    outU[$N + 1u] = select(0u, 1u, is_inf(infv));
    outU[$N + 2u] = select(0u, 1u, nanv != nanv);   // probe: folded under relaxed math?
    outU[$N + 3u] = select(0u, 1u, is_finite(FLT_MAX));
    outF[$N + 0u] = P.a * P.b + P.c;                // FMA contraction probe
  }
}

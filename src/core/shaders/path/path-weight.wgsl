// Throughput factor of a BSDF-sampled path vertex, shared by the path tree (path/pathtree.wgsl) and replay
// (path/replay.wgsl) (restir-api.md D3, Changelog B-2; math.md#path-tree "same-formula directions"). OWNER WP-B.
// A module of its own (not rc.wgsl) so tests that substitute rc.wgsl keep it.
#include "restir/types.wgsl"
#include "material/bsdf.wgsl"

/// Throughput factor of a BSDF-sampled vertex (restir-api.md D3, Changelog B-2): f_ℓ·|cos|/p(ω, ℓ) evaluated by
/// bsdf_query at the POSITION-DERIVED direction ω = normalize(pos(next) − pos(cur)) (the escape direction for a miss),
/// exactly the factor the shift recomputes at a reconnection vertex, so F is one function of the stored vertices in the
/// path tree, in replay and in every shift. Delta lobes (no density) and a non-positive joint pdf at ω (the sampled
/// direction was valid but ω, off by the ray offset, is not: measure-small) keep the sampler's weight.
fn rs_path_weight(q: BsdfQuery, sampleWeight: vec3f, isDelta: bool) -> vec3f {
  if (isDelta || !rs_pos_finite(q.p_joint)) { return sampleWeight; }
  return q.f_lobe / q.p_joint;
}


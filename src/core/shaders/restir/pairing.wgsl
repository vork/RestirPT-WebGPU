// Pairing maps: dihedral transforms of the involution textures and the partner of a pixel (restir-api.md §2.8, §3.9;
// math.md#paired-mis "pairing textures" [M4 addition]). OWNER WP-C. P0 STUB (restir-api.md §1.4): pair_partner
// returns invalid, so every slot is NOT_ACCEPTED. dihedral_apply(_t) and pair_A0 are real.
#include "restir/frame.wgsl"

struct PairResult { valid: bool, partner: vec2u }            // member-local pixel

/// Dihedral code M (3 bits): bit0 swap axes, bit1 negate x, bit2 negate y, applied in that order.
fn dihedral_apply(code: u32, v: vec2i) -> vec2i {
  var r = select(v, v.yx, (code & 1u) != 0u);
  if ((code & 2u) != 0u) { r.x = -r.x; }
  if ((code & 4u) != 0u) { r.y = -r.y; }
  return r;
}
/// Mᵀ = M⁻¹ (orthogonal): undo the steps in reverse order.
fn dihedral_apply_t(code: u32, v: vec2i) -> vec2i {
  var r = v;
  if ((code & 4u) != 0u) { r.y = -r.y; }
  if ((code & 2u) != 0u) { r.x = -r.x; }
  return select(r, r.yx, (code & 1u) != 0u);
}

/// STUB (P0): no partner.
fn pair_partner(local: vec2u, member: u32, t: u32, round: u32, slot: u32) -> PairResult {
  return PairResult(false, local);
}

/// A0 (D8, §3.9): both hits, n_g·n_g ≥ 0.5, |z_a − z_b| ≤ 0.1·min(z_a, z_b), z = camera distance (rsVbuf.w).
fn pair_A0(a: vec4u, ga: vec4f, b: vec4u, gb: vec4f) -> bool {
  if (a.x == 0xFFFFFFFFu || b.x == 0xFFFFFFFFu) { return false; }
  let za = bitcast<f32>(a.w);
  let zb = bitcast<f32>(b.w);
  return dot(ga.xyz, gb.xyz) >= 0.5 && abs(za - zb) <= 0.1 * min(za, zb);
}

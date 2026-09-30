// Pairing maps: dihedral transforms of the involution textures and the partner of a pixel (restir-api.md §2.8, §3.9,
// D8/D9; math.md#paired-mis "pairing textures" [M4 addition]). OWNER WP-C. TS mirror: render/restir/pairing.ts.
//   pairTex (RS_PAIRTEX_BINDING): texture_2d_array<i32> rg8sint 256² × 8; layer s = slot s holds a W_s-torus involution
//   of partner deltas (d = 0: no partner). Per (frame t, round r, member m, slot s):
//     h = pcg4d(runSeed ^ m·φ, t, (r << 8) | s, STREAM_PAIRING);  M = h.x & 7;  o = (h.y % W_s, h.z % W_s)
//     q = (M·p + o) mod W_s (positive),  partner = p + Mᵀ·d(q),  valid iff d ≠ 0 and the partner lies in the member tile.
//   Reciprocity survives the transform, off-tile partners are lost symmetrically: no cross-member pairs.
// Without RS_PAIRTEX_BINDING, pair_partner compiles to "no partner" (modules that include this file for types only).
// M5 (OWNER T-D; restir-temporal-api.md TD21, §3.7): the reciprocal disocclusion boost. Boost slots
// s ∈ [numSlots − boostSlots, numSlots) use pairing layers s ≤ 5 like ordinary slots and accept a pair iff
// A0 ∧ (dis(p) ∨ dis(q)), dis = the pixel has no valid q′ this frame (tState flag TS_DISOCC, written by T1 for every
// pixel before the spatial stage), evaluated once per pair by the thread of the smaller atlas index (rs_pair_accept).
#include "restir/frame.wgsl"
#include "restir/queue.wgsl"

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

/// Logical size W_s of layer s (RestirParams.pairTexSize / pairTexSize2; 0 = unused layer).
fn pair_tex_size(s: u32) -> u32 {
  if (s < 4u) { return rsParams.pairTexSize[s]; }
  return rsParams.pairTexSize2[s & 3u];
}

struct PairXf { code: u32, off: vec2i, size: i32 }
/// Transform of (frame t, round r, member m, slot s) (§2.8; m includes RestirParams.memberBase, Changelog A5).
fn pair_transform(member: u32, t: u32, round: u32, slot: u32) -> PairXf {
  let Ws = pair_tex_size(slot);
  let h = pcg4d(vec4u(frame.runSeed ^ (member * 0x9e3779b9u), t, (round << 8u) | slot, STREAM_PAIRING));
  let W = max(Ws, 1u);
  return PairXf(h.x & 7u, vec2i(i32(h.y % W), i32(h.z % W)), i32(Ws));
}

#if RS_PAIRTEX_BINDING
/// Partner of member-local pixel `local` in slot `slot` of round `round` (§2.8).
fn pair_partner(local: vec2u, member: u32, t: u32, round: u32, slot: u32) -> PairResult {
  var r = PairResult(false, local);
  let x = pair_transform(member, t, round, slot);
  if (x.size <= 0) { return r; }
  let mp = dihedral_apply(x.code, vec2i(local)) + x.off;
  let q = ((mp % x.size) + x.size) % x.size;
  let d = textureLoad(pairTex, q, i32(slot), 0).xy;
  if (d.x == 0 && d.y == 0) { return r; }
  let pp = vec2i(local) + dihedral_apply_t(x.code, d);
  if (pp.x < 0 || pp.y < 0 || pp.x >= i32(rsParams.memberSize.x) || pp.y >= i32(rsParams.memberSize.y)) { return r; }
  r.valid = true;
  r.partner = vec2u(pp);
  return r;
}
#else
/// Without the pairing texture: no partner.
fn pair_partner(local: vec2u, member: u32, t: u32, round: u32, slot: u32) -> PairResult {
  return PairResult(false, local);
}
#endif

/// Atlas pixel of a member-local partner of p (same member tile).
fn pair_atlas_px(p: RsPix, partnerLocal: vec2u) -> vec2u { return p.px - p.local + partnerLocal; }
fn pair_atlas_index(px: vec2u) -> u32 { return px.y * rsParams.atlasSize.x + px.x; }

/// A0 (D8, §3.9): both hits, n_g·n_g ≥ 0.5, |z_a − z_b| ≤ 0.1·min(z_a, z_b), z = camera distance (rsVbuf.w).
/// Callers pass the texels in canonical order (smaller atlas index first) and evaluate it once per pair.
fn pair_A0(a: vec4u, ga: vec4f, b: vec4u, gb: vec4f) -> bool {
  if (a.x == 0xFFFFFFFFu || b.x == 0xFFFFFFFFu) { return false; }
  let za = bitcast<f32>(a.w);
  let zb = bitcast<f32>(b.w);
  return dot(ga.xyz, gb.xyz) >= 0.5 && abs(za - zb) <= 0.1 * min(za, zb);
}

// ---- M5 reciprocal disocclusion boost (restir-temporal-api.md TD21, §3.7; OWNER T-D) ---------------------------------
// The tState flag word is read without restir/tframe.wgsl: that module declares the RsTemporal uniform, which M4 modules
// (rs_pair_accept) must not (tests/restir/tframe.test.ts). PAIR_TS_* mirror tframe.wgsl ts_word(ai, TSW_FLAGS) and
// TS_DISOCC; tests/restir/boost.test.ts checks them against layout.ts (tsWord, TSW.flags, TS_CONSTS.TS_DISOCC).
const PAIR_TS_WORDS: u32 = 20u;
const PAIR_TSW_FLAGS: u32 = 11u;
const PAIR_TS_DISOCC: u32 = 2u;

/// First boost slot (= numSlots when the boost is off: RestirParams.boostSlots is 0 without temporal).
fn pair_first_boost_slot() -> u32 { return rsParams.numSlots - min(rsParams.boostSlots, rsParams.numSlots); }
/// A_boost = A0(p, q) ∧ (dis(p) ∨ dis(q)) (TD21): symmetric in (p, q), G-buffer / q′ only (sample-independent).
fn pair_boost_accept(a0: bool, disP: bool, disQ: bool) -> bool { return a0 && (disP || disQ); }
#if RS_ARENA_BINDING
/// dis(ai): no valid q′ this frame (T1's tState flags; TS_DISOCC also covers the no-history frames, TS_NO_HIST).
fn pair_disoccluded(ai: u32) -> bool {
  let w = 6u * rs_atlas_pixels() * rs_ns_alloc() + PAIR_TS_WORDS * ai + PAIR_TSW_FLAGS;
  return (arena_word(w) & PAIR_TS_DISOCC) != 0u;
}
#endif

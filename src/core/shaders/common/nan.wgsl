// Bit-level float classification. `x != x` is folded away under Chrome's relaxed Metal math (measured, M0),
// so NaN/Inf checks MUST be bit tests (plan §1.1, test T15).
fn is_nan(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }
fn is_inf(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u; }
fn is_finite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }
fn all_finite3(v: vec3f) -> bool { return is_finite(v.x) && is_finite(v.y) && is_finite(v.z); }

// Wächter & Binder, "A Fast and Robust Method for Avoiding Self-Intersection" (Ray Tracing Gems ch. 6, 2019).
// math.md#visibility (offsets). Constants as in the chapter (= Cycles kernel/bvh/util.h).
const WB_ORIGIN: f32 = 1.0 / 32.0;        // below this |p| component use a float offset
const WB_FLOAT_SCALE: f32 = 1.0 / 65536.0;
const WB_INT_SCALE: f32 = 256.0;          // offset in ULPs along the normal

// p: surface point (computed from barycentrics on the vertices, not o + t·d); n: unit GEOMETRIC normal, already
// flipped to the side the ray leaves. n = 0 returns p unchanged (point endpoints, e.g. analytic lights).
fn offset_ray(p: vec3f, n: vec3f) -> vec3f {
  let of_i = vec3i(WB_INT_SCALE * n);
  let p_i = bitcast<vec3f>(bitcast<vec3i>(p) + select(of_i, -of_i, p < vec3f(0.0)));
  return select(p_i, p + WB_FLOAT_SCALE * n, abs(p) < vec3f(WB_ORIGIN));
}

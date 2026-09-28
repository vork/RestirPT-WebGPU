// Ray–triangle intersection (plan §1.3): Möller–Trumbore (default) and Woop–Benthin–Wald 2013 watertight with a
// canonical edge order. Both are two-sided, accept t in (0, tmax) and return barycentrics (1−u−v, u, v) of
// (v0, v1, v2). d need not be normalized.
#include "common/math.wgsl"
#include "common/nan.wgsl"

struct TriIsect { t: f32, u: f32, v: f32, ok: bool }

// Möller–Trumbore on precomputed edges e1 = v1 − v0, e2 = v2 − v0 (the BVH `tris` layout).
fn isect_mt(o: vec3f, d: vec3f, v0: vec3f, e1: vec3f, e2: vec3f, tmax: f32) -> TriIsect {
  let h = cross(d, e2);
  let a = dot(e1, h);
  if (a == 0.0) { return TriIsect(0.0, 0.0, 0.0, false); }
  let f = 1.0 / a;
  let s = o - v0;
  let u = f * dot(s, h);
  let q = cross(s, e1);
  let v = f * dot(d, q);
  let t = f * dot(e2, q);
  return TriIsect(t, u, v, u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > 0.0 && t < tmax);
}

// Woop et al. 2013 (JCGT 2(1)). The per-ray shear is three row vectors, so every vertex is transformed by
// identical branch-free code: a vertex shared by two triangles yields identical bits in both.
struct WoopRay { rx: vec3f, ry: vec3f, rz: vec3f }

fn isect_unit(k: u32) -> vec3f { return vec3f(select(0.0, 1.0, k == 0u), select(0.0, 1.0, k == 1u), select(0.0, 1.0, k == 2u)); }

fn woop_setup(d: vec3f) -> WoopRay {
  let ad = abs(d);
  var kz = 2u;
  if (ad.x >= ad.y && ad.x >= ad.z) { kz = 0u; } else if (ad.y >= ad.z) { kz = 1u; }
  var kx = (kz + 1u) % 3u;
  var ky = (kx + 1u) % 3u;
  let ez = isect_unit(kz);
  let dz = dot(d, ez);
  if (dz < 0.0) { let tmp = kx; kx = ky; ky = tmp; } // preserve winding
  let sx = dot(d, isect_unit(kx)) / dz;
  let sy = dot(d, isect_unit(ky)) / dz;
  return WoopRay(isect_unit(kx) - sx * ez, isect_unit(ky) - sy * ez, (1.0 / dz) * ez);
}

// 2D edge function with endpoints in canonical (vertex-id) order, negated as needed: the two triangles sharing
// an edge compute bit-identical magnitudes, so FMA contraction cannot open a crack between them.
fn woop_edge(p: vec2f, vp: u32, q: vec2f, vq: u32) -> f32 {
  let swap = vq < vp;
  let a = select(p, q, swap);
  let b = select(q, p, swap);
  let e = a.x * b.y - a.y * b.x;
  return select(e, -e, swap);
}

// p0..p2: raw vertices; vid: their position-welded global vertex ids (layout.ts).
fn isect_woop(o: vec3f, wr: WoopRay, p0: vec3f, p1: vec3f, p2: vec3f, vid: vec3u, tmax: f32) -> TriIsect {
  let A = p0 - o;
  let B = p1 - o;
  let C = p2 - o;
  let a2 = vec2f(dot(wr.rx, A), dot(wr.ry, A));
  let b2 = vec2f(dot(wr.rx, B), dot(wr.ry, B));
  let c2 = vec2f(dot(wr.rx, C), dot(wr.ry, C));
  let U = woop_edge(c2, vid.z, b2, vid.y);
  let V = woop_edge(a2, vid.x, c2, vid.z);
  let W = woop_edge(b2, vid.y, a2, vid.x);
  let neg = U < 0.0 || V < 0.0 || W < 0.0;
  let pos = U > 0.0 || V > 0.0 || W > 0.0;
  let det = U + V + W;
  if ((neg && pos) || det == 0.0) { return TriIsect(0.0, 0.0, 0.0, false); }
  let T = U * dot(wr.rz, A) + V * dot(wr.rz, B) + W * dot(wr.rz, C);
  let rdet = 1.0 / det;
  let t = T * rdet;
  return TriIsect(t, V * rdet, W * rdet, t > 0.0 && t < tmax && is_finite(t));
}

// Surface point from barycentrics on the vertices (Wächter–Binder: never o + t·d) and the geometric normal
// (right-handed winding, unnormalized cross → normalized; zero for degenerate triangles).
fn tri_point(p0: vec3f, p1: vec3f, p2: vec3f, u: f32, v: f32) -> vec3f {
  return p0 + u * (p1 - p0) + v * (p2 - p0);
}
fn tri_geom_normal(p0: vec3f, p1: vec3f, p2: vec3f) -> vec3f {
  let n = cross(p1 - p0, p2 - p0);
  let l = length(n);
  return select(vec3f(0.0), n / l, l > 0.0);
}

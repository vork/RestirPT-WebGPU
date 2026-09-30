// Scene data on the GPU (packers + binding table: src/core/render/scene-gpu.ts). All positions are in the
// render-internal recentred frame (p_int = p_world − O, math.md#raster).
//
// Group $SCENE_GROUP:
//   0 bvh_nodes, 1 bvh_tris      traverse.wgsl (the includer sets BVH_DECLARE_BINDINGS, BVH_GROUP = SCENE_GROUP)
//   2 sceneVerts                  vertex arena, array<vec4u> (src/core/gpu/vertex-format.ts; data-formats.md §B1–§B6)
//   3 sceneTris                   vec4u per triangle: (i0, i1, i2, material | triFlags << 24), indexed by primId
//   4 sceneMaterials              MaterialGpu per material (336 B)
//   8.. textures                  material/textures.wgsl (TEX_GROUP = SCENE_GROUP, TEX_BINDING_BASE = 8)
//
// VERTEX_FORMAT 1 (quantized, 16 B/vertex): P21 position on the global 2^k lattice (header posBase/posScale), oct
// 2 × 16 snorm normal, UV as a 2 × 16 bit code on the material's dyadic lattice (or an index into the wide-UV f32
// section), COLOR_0 in its own rgba8/rgba16 section. VERTEX_FORMAT 0 (lossless): f32 [p, uv.x] [n, uv.y] [rgba].
// Positions and UVs decode as (integer) × 2^k: bit-identical to the CPU mirror under fast-math/FMA. Normals and
// colours decode through the exact f32 product q·f32(1/(2^b − 1)) (never unpack*: implementation-defined precision).
//
// CUSTOM_ALPHA: provides traverse.wgsl's alpha_pass hook = the MASK cutout (plan §1.3; math.md#visibility):
// α = baseColorFactor.a × baseColorTexture.a (bilinear, LOD 0) × COLOR_0.a, the hit counts iff α ≥ alphaCutoff.
// Only triangles flagged TRI_ALPHA_MASK fetch anything; it creates no vertex and consumes no random numbers.
#include "common/nan.wgsl"
#include "material/textures.wgsl"

// Must match packMaterials() in scene-gpu.ts (MATERIAL_LAYOUT).
struct MaterialGpu {
  baseColor: vec4f,          // baseColorFactor (linear rgba)
  emission: vec3f,           // emissiveFactor · emissiveStrength
  alphaCutoff: f32,
  metallic: f32,
  roughness: f32,
  ior: f32,
  flags: u32,                // MAT_* bits
  specularColor: vec3f,
  specularFactor: f32,
  v1Diffuse: vec3f,          // V1 validation model (model == 'v1')
  transmission: f32,
  v1Glossy: vec3f,
  v1Roughness: f32,
  v1Mix: f32,
  normalScale: f32,
  uvBaseU: i32,              // UV lattice base (VERTEX_FORMAT 1): uv = (q + base) · 2^k, k in flags bits 16..31
  uvBaseV: i32,
  texBaseColor: TexSlot,     // sRGB
  texMetalRough: TexSlot,    // linear (B metallic, G roughness)
  texNormal: TexSlot,        // linear
  texEmissive: TexSlot,      // sRGB
  texTransmission: TexSlot,  // linear
  texSpecular: TexSlot,      // linear (A)
  texSpecularColor: TexSlot, // sRGB
}

const MAT_ALPHA_MASK: u32 = 1u;
const MAT_DOUBLE_SIDED: u32 = 2u;
const MAT_V1: u32 = 4u;
const MAT_GLASS_NODE: u32 = 8u;        // Cycles Glass BSDF node (M3b): Color = baseColor, Roughness, IOR
const MAT_REFRACTION_NODE: u32 = 16u;  // Cycles Refraction BSDF node (M3b)
const MAT_UV_WIDE: u32 = 32u;          // VERTEX_FORMAT 1: this material's UVs are f32 (wide-UV section)

// SceneGeometry.triFlags bits (src/core/scene/types.ts).
const TRI_ALPHA_MASK: u32 = 1u;
const TRI_EMISSIVE: u32 = 2u;
const TRI_FLIPPED: u32 = 4u;
const TRI_FLAT: u32 = 8u;              // flat face: ns = ng (the vertex normal is not read)
const TRI_MAT_MASK: u32 = 0xffffffu;
const TRI_FLAGS_SHIFT: u32 = 24u;

@group($SCENE_GROUP) @binding(2) var<storage, read> sceneVerts: array<vec4u>;
@group($SCENE_GROUP) @binding(3) var<storage, read> sceneTris: array<vec4u>;
@group($SCENE_GROUP) @binding(4) var<storage, read> sceneMaterials: array<MaterialGpu>;

fn tri_material(t: vec4u) -> u32 { return t.w & TRI_MAT_MASK; }
fn tri_flags(t: vec4u) -> u32 { return t.w >> TRI_FLAGS_SHIFT; }

// ---- vertex arena (vertex-format.ts) ----
const VQ_HEADER: u32 = 2u;                   // header vec4s before the records
const VQ_C_OCT16: u32 = 0x38000100u;         // f32(1/32767)
const VQ_C_UNORM8: u32 = 0x3b808081u;        // f32(1/255)
const VQ_C_UNORM16: u32 = 0x37800080u;       // f32(1/65535)

fn vq_word(w: u32) -> u32 { return sceneVerts[w >> 2u][w & 3u]; }
/// 2^k as f32, k in [−126, 127] (exact bit construction).
fn vq_pow2(k: i32) -> f32 { return bitcast<f32>(u32(k + 127) << 23u); }

/// Octahedral decode (Stubbe fold) of snorm codes q with step c; CPU mirror quantize.ts octDecode.
fn vq_oct(q: vec2i, c: f32) -> vec3f {
  let e = vec2f(q) * c;
  var v = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
  let t = max(-v.z, 0.0);
  v.x += select(t, -t, e.x >= 0.0);
  v.y += select(t, -t, e.y >= 0.0);
  return normalize(v);
}

#if VERTEX_FORMAT == 1
/// Vertex record i (16 B): (P21 lo, P21 hi, oct16 normal, uv word).
fn vq_rec(i: u32) -> vec4u { return sceneVerts[VQ_HEADER + i]; }
/// Recentred position: (21-bit offset + posBase) · 2^k — integer × power of two, exact.
fn vq_pos(r: vec4u) -> vec3f {
  let h = sceneVerts[0];
  let x = r.x & 0x1FFFFFu;
  let y = (r.x >> 21u) | ((r.y & 0x3FFu) << 11u);
  let z = (r.y >> 10u) & 0x1FFFFFu;
  return vec3f(bitcast<vec3i>(vec3u(x, y, z)) + bitcast<vec3i>(h.xyz)) * bitcast<f32>(h.w);
}
fn vq_normal(r: vec4u) -> vec3f {
  let q = vec2i(bitcast<i32>(r.z << 16u) >> 16u, bitcast<i32>(r.z) >> 16u);
  return vq_oct(q, bitcast<f32>(VQ_C_OCT16));
}
/// UV of record r under material mi's lattice (or the wide f32 section).
fn vq_uv(r: vec4u, mi: u32) -> vec2f {
  let f = sceneMaterials[mi].flags;
  if ((f & MAT_UV_WIDE) != 0u) {
    let o = sceneVerts[1].y + 2u * r.w;
    return vec2f(bitcast<f32>(vq_word(o)), bitcast<f32>(vq_word(o + 1u)));
  }
  let q = vec2i(vec2u(r.w & 0xFFFFu, r.w >> 16u));
  let base = vec2i(sceneMaterials[mi].uvBaseU, sceneMaterials[mi].uvBaseV);
  let k = vec2i(i32((f >> 16u) & 0xFFu) - 128, i32(f >> 24u) - 128);
  return vec2f(q + base) * vec2f(vq_pow2(k.x), vq_pow2(k.y));
}
fn vq_color(i: u32) -> vec4f {
  let h = sceneVerts[1];
  if (h.w == 1u) {
    let c = vq_word(h.z + i);
    return vec4f(vec4u(c & 0xFFu, (c >> 8u) & 0xFFu, (c >> 16u) & 0xFFu, c >> 24u)) * bitcast<f32>(VQ_C_UNORM8);
  }
  if (h.w == 2u) {
    let c0 = vq_word(h.z + 2u * i);
    let c1 = vq_word(h.z + 2u * i + 1u);
    return vec4f(vec4u(c0 & 0xFFFFu, c0 >> 16u, c1 & 0xFFFFu, c1 >> 16u)) * bitcast<f32>(VQ_C_UNORM16);
  }
  return vec4f(1.0);
}
fn scene_vertex_pos(i: u32) -> vec3f { return vq_pos(vq_rec(i)); }
fn scene_vertex_normal(i: u32) -> vec3f { return vq_normal(vq_rec(i)); }
fn scene_vertex_uv(i: u32, mi: u32) -> vec2f { return vq_uv(vq_rec(i), mi); }
fn scene_vertex_color(i: u32) -> vec4f { return vq_color(i); }
#else
fn vf_rec(i: u32, k: u32) -> vec4u { return sceneVerts[VQ_HEADER + 3u * i + k]; }
fn scene_vertex_pos(i: u32) -> vec3f { return bitcast<vec3f>(vf_rec(i, 0u).xyz); }
fn scene_vertex_normal(i: u32) -> vec3f { return bitcast<vec3f>(vf_rec(i, 1u).xyz); }
fn scene_vertex_uv(i: u32, mi: u32) -> vec2f { return bitcast<vec2f>(vec2u(vf_rec(i, 0u).w, vf_rec(i, 1u).w)); }
fn scene_vertex_color(i: u32) -> vec4f { return bitcast<vec4f>(vf_rec(i, 2u)); }
#endif

/// Barycentric interpolation with (1−u−v, u, v) on (v0, v1, v2) of the ORIGINAL index order (traverse.wgsl Hit).
fn scene_uv0(t: vec4u, u: f32, v: f32) -> vec2f {
  let mi = tri_material(t);
  return (1.0 - u - v) * scene_vertex_uv(t.x, mi) + u * scene_vertex_uv(t.y, mi) + v * scene_vertex_uv(t.z, mi);
}
fn scene_color0(t: vec4u, u: f32, v: f32) -> vec4f {
  return (1.0 - u - v) * scene_vertex_color(t.x) + u * scene_vertex_color(t.y) + v * scene_vertex_color(t.z);
}

/// MASK alpha (math.md#visibility): baseColorFactor.a × texture.a (LOD 0) × COLOR_0.a.
fn material_alpha(m: MaterialGpu, uv: vec2f, col: vec4f) -> f32 {
  return m.baseColor.a * tex_sample(m.texBaseColor, uv).a * col.a;
}

/// Albedo AOV: base colour factor × texture × COLOR_0 (V1 materials: the Lambert albedo).
fn material_albedo(m: MaterialGpu, uv: vec2f, col: vec4f) -> vec3f {
  if ((m.flags & MAT_V1) != 0u) { return m.v1Diffuse; }
  return m.baseColor.rgb * tex_sample(m.texBaseColor, uv).rgb * col.rgb;
}

#if CUSTOM_ALPHA
fn alpha_pass(primId: u32, u: f32, v: f32) -> bool {
  let t = sceneTris[primId];
  if ((tri_flags(t) & TRI_ALPHA_MASK) == 0u) { return true; }
  let m = sceneMaterials[tri_material(t)];
  return material_alpha(m, scene_uv0(t, u, v), scene_color0(t, u, v)) >= m.alphaCutoff;
}
#endif

#if GLASS_PLANT == 5
/// Gate-1 plant B-shadow (glass.wgsl): glass triangles (Glass / Refraction node, Principled transmission > 0) do not
/// occlude any-hit rays.
fn glass_plant_transparent(primId: u32) -> bool {
  let m = sceneMaterials[tri_material(sceneTris[primId])];
  return (m.flags & (MAT_GLASS_NODE | MAT_REFRACTION_NODE)) != 0u || m.transmission > 0.0;
}
#endif

struct SurfaceHit {
  pos: vec3f,        // from barycentrics on the vertices (never o + t·d; Wächter–Binder)
  ng: vec3f,         // unit geometric normal, flipped toward the incoming ray (Cycles two-sided)
  ns: vec3f,         // interpolated shading normal, flipped together with ng
  uv: vec2f,
  color: vec4f,
  matId: u32,
  triFlags: u32,
  backfacing: bool,  // the ray hit the side opposite to the winding normal
}

/// Surface attributes at hit (primId, u, v) for a ray with direction d (need not be normalized).
fn scene_surface(primId: u32, u: f32, v: f32, d: vec3f) -> SurfaceHit {
  let t = sceneTris[primId];
  let w = 1.0 - u - v;
  var s: SurfaceHit;
  s.matId = tri_material(t);
  s.triFlags = tri_flags(t);
#if VERTEX_FORMAT == 1
  let ra = vq_rec(t.x);
  let rb = vq_rec(t.y);
  let rc = vq_rec(t.z);
  let pa = vq_pos(ra);
  let pb = vq_pos(rb);
  let pc = vq_pos(rc);
#else
  let pa = scene_vertex_pos(t.x);
  let pb = scene_vertex_pos(t.y);
  let pc = scene_vertex_pos(t.z);
#endif
  s.pos = w * pa + u * pb + v * pc;
  var ng = normalize(cross(pb - pa, pc - pa));
  s.backfacing = dot(ng, d) > 0.0;
  ng = select(ng, -ng, s.backfacing);
  s.ng = ng;
  if ((s.triFlags & TRI_FLAT) != 0u) {
    s.ns = ng;                                        // flat face: exactly ng (Cycles Ng on flat faces)
  } else {
#if VERTEX_FORMAT == 1
    var ns = w * vq_normal(ra) + u * vq_normal(rb) + v * vq_normal(rc);
#else
    var ns = w * scene_vertex_normal(t.x) + u * scene_vertex_normal(t.y) + v * scene_vertex_normal(t.z);
#endif
    let l2 = dot(ns, ns);
    s.ns = select(ng, select(ns, -ns, s.backfacing) * inverseSqrt(l2), l2 > 1e-24 && all_finite3(ns));
  }
#if VERTEX_FORMAT == 1
  s.uv = w * vq_uv(ra, s.matId) + u * vq_uv(rb, s.matId) + v * vq_uv(rc, s.matId);
#else
  s.uv = scene_uv0(t, u, v);
#endif
  s.color = scene_color0(t, u, v);
  return s;
}

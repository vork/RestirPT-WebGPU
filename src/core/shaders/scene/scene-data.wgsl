// Scene data on the GPU (packers + binding table: src/core/render/scene-gpu.ts). All positions are in the
// render-internal recentred frame (p_int = p_world − O, math.md#raster).
//
// Group $SCENE_GROUP:
//   0 bvh_nodes, 1 bvh_tris      traverse.wgsl (the includer sets BVH_DECLARE_BINDINGS, BVH_GROUP = SCENE_GROUP)
//   2 sceneVerts                  SceneVertex per vertex (48 B)
//   3 sceneTris                   vec4u per triangle: (i0, i1, i2, material | triFlags << 24), indexed by primId
//   4 sceneMaterials              MaterialGpu per material (336 B)
//   8.. textures                  material/textures.wgsl (TEX_GROUP = SCENE_GROUP, TEX_BINDING_BASE = 8)
//
// CUSTOM_ALPHA: provides traverse.wgsl's alpha_pass hook = the MASK cutout (plan §1.3; math.md#visibility):
// α = baseColorFactor.a × baseColorTexture.a (bilinear, LOD 0) × COLOR_0.a, the hit counts iff α ≥ alphaCutoff.
// Only triangles flagged TRI_ALPHA_MASK fetch anything; it creates no vertex and consumes no random numbers.
#include "common/nan.wgsl"
#include "material/textures.wgsl"

struct SceneVertex {
  p: vec3f,       // recentred position
  uvx: f32,       // TEXCOORD_0.x (glTF convention, origin top-left)
  n: vec3f,       // shading normal (world, unit)
  uvy: f32,       // TEXCOORD_0.y
  color: vec4f,   // COLOR_0 (1 when absent)
}

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
  pad0: f32,
  pad1: f32,
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

// SceneGeometry.triFlags bits (src/core/scene/types.ts).
const TRI_ALPHA_MASK: u32 = 1u;
const TRI_EMISSIVE: u32 = 2u;
const TRI_FLIPPED: u32 = 4u;
const TRI_MAT_MASK: u32 = 0xffffffu;
const TRI_FLAGS_SHIFT: u32 = 24u;

@group($SCENE_GROUP) @binding(2) var<storage, read> sceneVerts: array<SceneVertex>;
@group($SCENE_GROUP) @binding(3) var<storage, read> sceneTris: array<vec4u>;
@group($SCENE_GROUP) @binding(4) var<storage, read> sceneMaterials: array<MaterialGpu>;

fn tri_material(t: vec4u) -> u32 { return t.w & TRI_MAT_MASK; }
fn tri_flags(t: vec4u) -> u32 { return t.w >> TRI_FLAGS_SHIFT; }

/// Barycentric interpolation with (1−u−v, u, v) on (v0, v1, v2) of the ORIGINAL index order (traverse.wgsl Hit).
fn scene_uv0(t: vec4u, u: f32, v: f32) -> vec2f {
  let a = sceneVerts[t.x];
  let b = sceneVerts[t.y];
  let c = sceneVerts[t.z];
  return (1.0 - u - v) * vec2f(a.uvx, a.uvy) + u * vec2f(b.uvx, b.uvy) + v * vec2f(c.uvx, c.uvy);
}
fn scene_color0(t: vec4u, u: f32, v: f32) -> vec4f {
  return (1.0 - u - v) * sceneVerts[t.x].color + u * sceneVerts[t.y].color + v * sceneVerts[t.z].color;
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
  let a = sceneVerts[t.x];
  let b = sceneVerts[t.y];
  let c = sceneVerts[t.z];
  let w = 1.0 - u - v;
  var s: SurfaceHit;
  s.pos = w * a.p + u * b.p + v * c.p;
  var ng = normalize(cross(b.p - a.p, c.p - a.p));
  s.backfacing = dot(ng, d) > 0.0;
  ng = select(ng, -ng, s.backfacing);
  var ns = w * a.n + u * b.n + v * c.n;
  let l2 = dot(ns, ns);
  ns = select(ng, select(ns, -ns, s.backfacing) * inverseSqrt(l2), l2 > 1e-24 && all_finite3(ns));
  s.ng = ng;
  s.ns = ns;
  s.uv = w * vec2f(a.uvx, a.uvy) + u * vec2f(b.uvx, b.uvy) + v * vec2f(c.uvx, c.uvy);
  s.color = w * a.color + u * b.color + v * c.color;
  s.matId = tri_material(t);
  s.triFlags = tri_flags(t);
  return s;
}

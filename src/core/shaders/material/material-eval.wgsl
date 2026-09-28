// material_eval: MatEval at a surface hit from the material table + textures + COLOR_0
// (plan §1.5/§1.6; gap-bsdf §9 (textures), §10 (glTF → Principled mapping); math.md#bsdf-v1, #bsdf-v2).
// Includes scene/scene-data.wgsl (MaterialGpu table, textures, SurfaceHit) and material/bsdf.wgsl (LUT defines).
//
// V2 (glTF metallic-roughness → Cycles Principled, Blender-importer-equivalent, gap-bsdf §10.1):
//   base      = baseColorFactor.rgb × sRGB⁻¹(baseColorTexture.rgb) × COLOR_0.rgb
//   metallic  = metallicFactor × MR.B        roughness = roughnessFactor × MR.G       (linear)
//   ior       = KHR_materials_ior (1.5)
//   spec lvl  = 0.5 × specularFactor × specularTexture.A
//   spec tint = specularColorFactor × sRGB⁻¹(specularColorTexture.rgb)
//   trans     = transmissionFactor × transmissionTexture.R   (> 1e-5 ⇒ model 2, the M3b glass stub)
//   emission  = emissiveFactor·emissiveStrength × sRGB⁻¹(emissiveTexture.rgb)
// V1 (MAT_V1): Lambert albedo, glossy colour / roughness, mix from the V1 fields; emission as above.
// Textures: validation lookup tex_sample() = bilinear at LOD 0, decoded after filtering (textures.wgsl).
// Normal maps are M7 (TODO): ns is the interpolated shading normal of the hit.
#include "scene/scene-data.wgsl"
#include "material/bsdf.wgsl"

const MATEVAL_TRANSMISSION_CUTOFF: f32 = 1e-5;   // Principled transmission_weight > CLOSURE_WEIGHT_CUTOFF

/// MatEval from a material record at (uv, COLOR_0) with normals already flipped to the V side.
fn material_eval_record(mat: MaterialGpu, uv: vec2f, color: vec4f, ns: vec3f, ng: vec3f, V: vec3f) -> MatEval {
  var m: MatEval;
  m.ns = ns;
  m.ng = ng;
  m.emission = mat.emission * tex_sample(mat.texEmissive, uv).rgb;
  if ((mat.flags & MAT_V1) != 0u) {
    m.model = BSDF_MODEL_V1;
    m.v1_diffuse = mat.v1Diffuse;
    m.v1_glossy = mat.v1Glossy;
    m.v1_mix = mat.v1Mix;
    m.roughness = mat.v1Roughness;
    m.base_color = mat.v1Diffuse;       // albedo AOV convenience; unused by the V1 BSDF
    m.ior = 1.5;
    m.specular_level = 0.5;
    m.specular_tint = vec3f(1.0);
  } else {
    let mr = tex_sample(mat.texMetalRough, uv);
    m.base_color = mat.baseColor.rgb * tex_sample(mat.texBaseColor, uv).rgb * color.rgb;
    m.metallic = mat.metallic * mr.b;
    m.roughness = mat.roughness * mr.g;
    m.ior = mat.ior;
    m.specular_level = 0.5 * mat.specularFactor * tex_sample(mat.texSpecular, uv).a;
    m.specular_tint = mat.specularColor * tex_sample(mat.texSpecularColor, uv).rgb;
    m.transmission = mat.transmission * tex_sample(mat.texTransmission, uv).r;
    m.model = select(BSDF_MODEL_V2, BSDF_MODEL_GLASS, saturate(m.transmission) > MATEVAL_TRANSMISSION_CUTOFF);
  }
  m.flags = bsdf_flags(m, V);
  return m;
}

/// MatEval at a hit (scene_surface()) for the view direction V (unit, toward the previous vertex).
fn material_eval(hit: SurfaceHit, V: vec3f) -> MatEval {
  return material_eval_record(sceneMaterials[hit.matId], hit.uv, hit.color, hit.ns, hit.ng, V);
}

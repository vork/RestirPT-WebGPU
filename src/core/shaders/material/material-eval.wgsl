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
//   trans     = transmissionFactor × transmissionTexture.R   (> 1e-5 ⇒ model 2: Principled + glass closure, lobe G)
// Glass / Refraction BSDF nodes (MAT_GLASS_NODE / MAT_REFRACTION_NODE, validation): Color = baseColor (× texture ×
// COLOR_0), Roughness = roughness × MR.G, IOR = ior (models 3 / 4, material/glass.wgsl).
// MATEVAL_BACKFACING is set from the hit (the normals were flipped away from the winding normal) so the glass
// η_side can be recomputed for any evaluating V (bsdf_prepare).
//   emission  = emissiveFactor·emissiveStrength × sRGB⁻¹(emissiveTexture.rgb)
// V1 (MAT_V1): Lambert albedo, glossy colour / roughness, mix from the V1 fields; emission as above.
// Textures: validation lookup tex_sample() = bilinear at LOD 0, decoded after filtering (textures.wgsl).
// Normal maps (M7, NORMAL_MAP; m7-api.md §1): hit.ns is the Cycles Normal Map node normal for materials with a normal
// texture (scene_surface), hit.nsm the unmapped smooth / flat normal; both go to MatEval (ns = closure normal of every
// lobe, nsm = the bump-shadowing reference normal, bsdf.wgsl bsdf_bump_ok).
#include "scene/scene-data.wgsl"
#include "material/bsdf.wgsl"

const MATEVAL_TRANSMISSION_CUTOFF: f32 = 1e-5;   // Principled transmission_weight > CLOSURE_WEIGHT_CUTOFF

/// MatEval from a material record at (uv, COLOR_0) with normals already flipped to the V side (front-facing hit).
fn material_eval_record(mat: MaterialGpu, uv: vec2f, color: vec4f, ns: vec3f, ng: vec3f, V: vec3f) -> MatEval {
  return material_eval_record_ex(mat, uv, color, ns, ng, V, false);
}

/// As material_eval_record; `backfacing`: ns/ng are the negated winding normals (the hit was on the back side).
fn material_eval_record_ex(mat: MaterialGpu, uv: vec2f, color: vec4f, ns: vec3f, ng: vec3f, V: vec3f, backfacing: bool) -> MatEval {
  var m: MatEval;
  m.ns = ns;
  m.ng = ng;
#if NORMAL_MAP
  m.nsm = ns;                         // no normal map: the bump reference is the closure normal itself (test passes)
#endif
  m.flags = select(0u, MATEVAL_BACKFACING, backfacing);
  if ((mat.flags & MAT_V1) != 0u) {
    m.model = BSDF_MODEL_V1;
    m.base_color = mat.v1Diffuse;       // V1 slots (bsdf.wgsl MatEval): Diffuse, Glossy colour, mix
    m.specular_tint = mat.v1Glossy;
    m.metallic = mat.v1Mix;
    m.roughness = mat.v1Roughness;
    m.ior = 1.5;
    m.specular_level = 0.5;
  } else if ((mat.flags & (MAT_GLASS_NODE | MAT_REFRACTION_NODE)) != 0u) {
    let mr = tex_sample(mat.texMetalRough, uv);
    m.model = select(BSDF_MODEL_REFRACTION_NODE, BSDF_MODEL_GLASS_NODE, (mat.flags & MAT_GLASS_NODE) != 0u);
    m.base_color = mat.baseColor.rgb * tex_sample(mat.texBaseColor, uv).rgb * color.rgb;
    m.roughness = mat.roughness * mr.g;
    m.ior = mat.ior;
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
  m.flags = bsdf_flags(m, V);   // keeps MATEVAL_BACKFACING
  return m;
}

/// MatEval at a hit (scene_surface()) for the view direction V (unit, toward the previous vertex).
fn material_eval(hit: SurfaceHit, V: vec3f) -> MatEval {
#if NORMAL_MAP
  var m = material_eval_record_ex(sceneMaterials[hit.matId], hit.uv, hit.color, hit.ns, hit.ng, V, hit.backfacing);
  m.nsm = hit.nsm;
  return m;
#else
  return material_eval_record_ex(sceneMaterials[hit.matId], hit.uv, hit.color, hit.ns, hit.ng, V, hit.backfacing);
#endif
}

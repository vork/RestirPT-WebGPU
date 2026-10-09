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

#if MAT_VARIANTS
// ---- perf2 WP-8 (MAT_VARIANTS, interactive only). material_eval loads only the fields the material's branch reads
// (field-wise, never the whole 336 B record by value), and the scene keys of SceneGpu.materialVariantDefines() fold
// what the material table cannot contain: MAT_NO_V1 / MAT_ONLY_V1 / MAT_NO_NODES / MAT_NO_GLASS_NODE /
// MAT_NO_REFRACTION / MAT_NO_PGLASS (model tests), TEX_NO_<slot> (no material has a valid slot of that kind: the
// sample is the invalid-slot 1), TEX_NO_XFORM (every valid slot is the identity transform: uv' = uv exactly).
// The sRGB decode is static per slot (packMaterials: base colour / specular colour are the sRGB slots). Arithmetic
// is the expression of material_eval_record_ex, operand for operand.

/// tex_sample(sceneMaterials[i].texBaseColor, uv): baseColorTexture (sRGB).
fn mv_tex_base(i: u32, uv: vec2f) -> vec4f {
#if TEX_NO_BASE
  return vec4f(1.0);
#else
  let info = sceneMaterials[i].texBaseColor.info;
  if ((info & TEX_VALID) == 0u) { return vec4f(1.0); }
#if TEX_NO_XFORM
  let st = uv;
#else
  let st = tex_xf_uv(sceneMaterials[i].texBaseColor.xf0, sceneMaterials[i].texBaseColor.xf1, uv);
#endif
  return tex_srgb_to_linear4(tex_fetch(info, st, 0.0));
#endif
}
/// tex_sample(sceneMaterials[i].texMetalRough, uv): metallicRoughnessTexture (linear).
fn mv_tex_mr(i: u32, uv: vec2f) -> vec4f {
#if TEX_NO_MR
  return vec4f(1.0);
#else
  let info = sceneMaterials[i].texMetalRough.info;
  if ((info & TEX_VALID) == 0u) { return vec4f(1.0); }
#if TEX_NO_XFORM
  let st = uv;
#else
  let st = tex_xf_uv(sceneMaterials[i].texMetalRough.xf0, sceneMaterials[i].texMetalRough.xf1, uv);
#endif
  return tex_fetch(info, st, 0.0);
#endif
}
/// tex_sample(sceneMaterials[i].texSpecular, uv): specularTexture (linear).
fn mv_tex_spec(i: u32, uv: vec2f) -> vec4f {
#if TEX_NO_SPEC
  return vec4f(1.0);
#else
  let info = sceneMaterials[i].texSpecular.info;
  if ((info & TEX_VALID) == 0u) { return vec4f(1.0); }
#if TEX_NO_XFORM
  let st = uv;
#else
  let st = tex_xf_uv(sceneMaterials[i].texSpecular.xf0, sceneMaterials[i].texSpecular.xf1, uv);
#endif
  return tex_fetch(info, st, 0.0);
#endif
}
/// tex_sample(sceneMaterials[i].texSpecularColor, uv): specularColorTexture (sRGB).
fn mv_tex_speccol(i: u32, uv: vec2f) -> vec4f {
#if TEX_NO_SPECCOL
  return vec4f(1.0);
#else
  let info = sceneMaterials[i].texSpecularColor.info;
  if ((info & TEX_VALID) == 0u) { return vec4f(1.0); }
#if TEX_NO_XFORM
  let st = uv;
#else
  let st = tex_xf_uv(sceneMaterials[i].texSpecularColor.xf0, sceneMaterials[i].texSpecularColor.xf1, uv);
#endif
  return tex_srgb_to_linear4(tex_fetch(info, st, 0.0));
#endif
}
/// tex_sample(sceneMaterials[i].texTransmission, uv): transmissionTexture (linear).
fn mv_tex_trans(i: u32, uv: vec2f) -> vec4f {
#if TEX_NO_TRANS
  return vec4f(1.0);
#else
  let info = sceneMaterials[i].texTransmission.info;
  if ((info & TEX_VALID) == 0u) { return vec4f(1.0); }
#if TEX_NO_XFORM
  let st = uv;
#else
  let st = tex_xf_uv(sceneMaterials[i].texTransmission.xf0, sceneMaterials[i].texTransmission.xf1, uv);
#endif
  return tex_fetch(info, st, 0.0);
#endif
}

fn mv_mat_v1(i: u32) -> bool {
#if MAT_NO_V1
  return false;
#elif MAT_ONLY_V1
  return true;
#else
  return (sceneMaterials[i].flags & MAT_V1) != 0u;
#endif
}
fn mv_mat_node(i: u32) -> bool {
#if MAT_NO_NODES
  return false;
#else
  return (sceneMaterials[i].flags & (MAT_GLASS_NODE | MAT_REFRACTION_NODE)) != 0u;
#endif
}
fn mv_node_model(i: u32) -> u32 {
#if MAT_NO_REFRACTION
  return BSDF_MODEL_GLASS_NODE;
#elif MAT_NO_GLASS_NODE
  return BSDF_MODEL_REFRACTION_NODE;
#else
  return select(BSDF_MODEL_REFRACTION_NODE, BSDF_MODEL_GLASS_NODE, (sceneMaterials[i].flags & MAT_GLASS_NODE) != 0u);
#endif
}

/// material_eval_record_ex(sceneMaterials[i], uv, color, ns, ng, V, backfacing) from field-wise loads.
fn material_eval_mv(i: u32, uv: vec2f, color: vec4f, ns: vec3f, ng: vec3f, V: vec3f, backfacing: bool) -> MatEval {
  var m: MatEval;
  m.ns = ns;
  m.ng = ng;
#if NORMAL_MAP
  m.nsm = ns;
#endif
  m.flags = select(0u, MATEVAL_BACKFACING, backfacing);
  if (mv_mat_v1(i)) {
    m.model = BSDF_MODEL_V1;
    m.base_color = sceneMaterials[i].v1Diffuse;
    m.specular_tint = sceneMaterials[i].v1Glossy;
    m.metallic = sceneMaterials[i].v1Mix;
    m.roughness = sceneMaterials[i].v1Roughness;
    m.ior = 1.5;
    m.specular_level = 0.5;
  } else if (mv_mat_node(i)) {
    let mr = mv_tex_mr(i, uv);
    m.model = mv_node_model(i);
    m.base_color = sceneMaterials[i].baseColor.rgb * mv_tex_base(i, uv).rgb * color.rgb;
    m.roughness = sceneMaterials[i].roughness * mr.g;
    m.ior = sceneMaterials[i].ior;
    m.specular_level = 0.5;
    m.specular_tint = vec3f(1.0);
  } else {
    let mr = mv_tex_mr(i, uv);
    m.base_color = sceneMaterials[i].baseColor.rgb * mv_tex_base(i, uv).rgb * color.rgb;
    m.metallic = sceneMaterials[i].metallic * mr.b;
    m.roughness = sceneMaterials[i].roughness * mr.g;
    m.ior = sceneMaterials[i].ior;
    m.specular_level = 0.5 * sceneMaterials[i].specularFactor * mv_tex_spec(i, uv).a;
    m.specular_tint = sceneMaterials[i].specularColor * mv_tex_speccol(i, uv).rgb;
#if MAT_NO_PGLASS
    m.model = BSDF_MODEL_V2;         // transmission factor ≤ 0 everywhere: model 2 is unreachable (m.transmission is read
                                     // only by the model-2 closure, v2.wgsl, so it keeps its zero initialiser)
#else
    m.transmission = sceneMaterials[i].transmission * mv_tex_trans(i, uv).r;
    m.model = select(BSDF_MODEL_V2, BSDF_MODEL_GLASS, saturate(m.transmission) > MATEVAL_TRANSMISSION_CUTOFF);
#endif
  }
  m.flags = bsdf_flags(m, V);   // keeps MATEVAL_BACKFACING
  return m;
}
#endif

/// MatEval at a hit (scene_surface()) for the view direction V (unit, toward the previous vertex).
fn material_eval(hit: SurfaceHit, V: vec3f) -> MatEval {
#if MAT_VARIANTS
  var m = material_eval_mv(hit.matId, hit.uv, hit.color, hit.ns, hit.ng, V, hit.backfacing);
#if NORMAL_MAP
  m.nsm = hit.nsm;
#endif
  return m;
#elif NORMAL_MAP
  var m = material_eval_record_ex(sceneMaterials[hit.matId], hit.uv, hit.color, hit.ns, hit.ng, V, hit.backfacing);
  m.nsm = hit.nsm;
  return m;
#else
  return material_eval_record_ex(sceneMaterials[hit.matId], hit.uv, hit.color, hit.ns, hit.ng, V, hit.backfacing);
#endif
}

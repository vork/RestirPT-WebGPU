// Cached Potato direct-light specialization. Rough opaque surfaces use only light samples;
// sharp reflections and transmission retain BSDF/light MIS. No visibility is reused.
fn potato_nee_only(m: MatEval) -> bool {
  // When environment NEE is disabled, BSDF escapes are its only sampling technique.
  let envBsdfOnly = (envParams.flags & ENV_FLAG_PRESENT) != 0u && lightsParams.cur.envEntry == LIGHT_NONE;
  return m.roughness >= 0.5 && m.model != BSDF_MODEL_GLASS && !envBsdfOnly;
}

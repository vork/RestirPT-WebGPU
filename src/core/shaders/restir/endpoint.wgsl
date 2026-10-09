// NEE endpoints and emitter end terms for ReSTIR (restir-api.md §3.3, D4/D5; math.md#measure, #mis, #path-tree
// [M4 addition] "Base NEE endpoint"). An NEE endpoint is stored as the sampler's own light-local coordinates:
//   alias entry + (bits(u01(h_l0)), bits(u01(h_l1)))   area lights and emissive triangles ((u₁,u₂) of the area-uniform
//                                                       map, NOT barycentrics)
//   alias entry + (0, 0)                                delta lights (point, spot) and the sun (u unused)
//   env entry + ((i << 16) | j, h2)                     env: drawn cell and in-cell lattice offsets
// and re-evaluated with the PT's own per-entry functions (light_sample_entry / env_light_sample_cell), so
// nee_eval(x, nee_draw(h)) is bitwise nee_sample(x, h) (U-EP-1) and every shift reproduces z, Λ, q, p1 exactly.
// Emitter end terms (BSDF_TRI / BSDF_ENV) use the PT's tri_emission, tri_light_p1, envRadiance(envUV(ω)), p1Env,
// env_bsdf_mis_weight, mis_w2 and nee_mis_w1 directly (no copies).
#include "restir/types.wgsl"
#include "lights/env-sample.wgsl"
#include "geom/visible.wgsl"

struct NeeEndpoint { entry: u32, a: u32, b: u32 }        // alias entry + light-local words (§2.3); entry LIGHT_NONE: none

/// Draw an NEE endpoint (alias selection + light-local coordinates) from the path hashes of slots u_sel, u_sel2 and
/// (h_l0, h_l1, h_l2). No shading point is involved: the draw is a pure function of the hashes and the light slot.
fn nee_draw(slot: LightSlot, hSel: u32, hSel2: u32, hL: vec3u) -> NeeEndpoint {
  if (slot.nEntries == 0u) { return NeeEndpoint(LIGHT_NONE, 0u, 0u); }
  let entry = alias_sample(slot, hSel, hSel2);
  if (entry == slot.envEntry) {
    let ij = env_draw_cell(hL.x, hL.y);
    return NeeEndpoint(entry, (ij.x << 16u) | ij.y, hL.z);
  }
  if (entry < slot.nAnalytic) {
#if RS_LIGHT_REC4
    let kind = light_kind(slot, entry);
#else
    let kind = records[slot.lightOff + entry * LIGHT_REC_WORDS + 3u];
#endif
    if (kind == LT_POINT || kind == LT_SPOT || kind == LT_SUN) { return NeeEndpoint(entry, 0u, 0u); }
  }
  return NeeEndpoint(entry, bitcast<u32>(u32_to_unit(hL.x)), bitcast<u32>(u32_to_unit(hL.y)));
}

/// Evaluate a stored NEE endpoint at shading point x (z, dir, dist, cosZ, Λ, q, p1, isDelta, isInf, analytic, prim).
fn nee_eval(x: vec3f, ep: NeeEndpoint) -> LightSample {
  return nee_eval_s(x, ep, lightsParams.cur, envParams);
}

/// nee_eval under light slot `slot` and env record `er` (frame-selected, restir-temporal-api.md §3.1): the entry is in
/// the numbering of that slot (translate first with lt_translate).
fn nee_eval_s(x: vec3f, ep: NeeEndpoint, slot: LightSlot, er: EnvParams) -> LightSample {
  if (slot.nEntries == 0u || ep.entry == LIGHT_NONE) { return light_sample_none(); }
  if (ep.entry == slot.envEntry) { return env_light_sample_cell_s(slot, er, ep.entry, ep.a >> 16u, ep.a & 0xFFFFu, ep.b); }
  return light_sample_entry(x, slot, ep.entry, vec2f(bitcast<f32>(ep.a), bitcast<f32>(ep.b)));
}

/// Reservoir words of an NEE endpoint (rc triple of forced NEE, endpoint triple of every NEE path).
fn nee_endpoint_words(ep: NeeEndpoint) -> vec3u { return vec3u(RC_TAG_NEE | ep.entry, ep.a, ep.b); }
fn nee_endpoint_from_words(w: vec3u) -> NeeEndpoint { return NeeEndpoint(w.x & RC_ENTRY_MASK, w.y, w.z); }
fn rs_is_nee_words(w: vec3u) -> bool { return (w.x & RC_TAG_MASK) == RC_TAG_NEE && w.x != RC_ENV_DIR && w.x != RC_NONE; }

/// endpointId (D5) of an NEE endpoint: analytic alias entry | emissive-triangle primId | RS_ENV_ID.
fn nee_endpoint_id(ep: NeeEndpoint) -> u32 {
  return nee_endpoint_id_s(ep, lightsParams.cur);
}

/// endpointId of an NEE endpoint whose entry is in the numbering of `slot`.
fn nee_endpoint_id_s(ep: NeeEndpoint, slot: LightSlot) -> u32 {
  if (ep.entry == LIGHT_NONE) { return RC_NONE; }
  if (ep.entry == slot.envEntry) { return RS_ENV_ID; }
  if (ep.entry < slot.nAnalytic) { return ep.entry; }
  return emissive_tri(ep.entry - slot.nAnalytic).x;
}

/// NEE visibility exactly as the PT: visibleInf toward the sun / env, else the segment to z excluding both primitives.
fn nee_visible(x: SurfaceHit, xPrim: u32, ls: LightSample) -> bool {
  if (ls.isInf) { return visibleInf(x.pos, x.ng, xPrim, ls.dir); }
  return visible(x.pos, x.ng, xPrim, ls.pos, ls.nz, ls.prim);
}

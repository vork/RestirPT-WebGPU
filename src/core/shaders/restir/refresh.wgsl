// Suffix refresh of a stored record under a frame's light / env state (restir-temporal-api.md §3.5, TD9, TD23, TD31;
// math.md#light-changes [M5 addition]). OWNER T-C. refresh_record(ai, fsFrom, fsTo) evaluates the prefix-INDEPENDENT
// end terms of record resIn[ai] (its own frame fsFrom) under frame fsTo; the passes store the result as SfxRec
// (tframe.wgsl sfx_store). Everything prefix-dependent (classes L, E, B1, ∅, and f / ω1 at x_k of N1) is re-evaluated
// by the shift itself (TD10).
//
//   every NEE end    entry translated (lt_translate), J_P = pmf_to(e_to)/pmf_from(e), undefined iff e_to is missing or
//                    a pmf is ≤ 0 (the same predicate in T and T⁻¹; math.md#jacobian [M5 addition])
//   D-NEE  (k ≤ d−2) rad = β_s ⊙ (ω1/q)·f_all·Λ at the cached x_{d−1} (the path tree's own functions and grouping), with
//                    the shadow ray V(x_{d−1}, Φ_fs) only when the endpoint's own light MOVED (visibility rule)
//   D-BSDF (k ≤ d−2) rad = β_s ⊙ ω2·L: escape L_env,fs(envUV(ω)) or the static triangle L_e, ω2 from p1 of the frame's
//                    pmf and the cached p2 (ω2 = 1 after a delta end, SFX_DELTA_END)
//   N1     (k = d−1) rad = Λ_fs, aux = p1_fs at x_k (write-back cache), SXS_VIS = V(x_k, Φ_fs) (ray iff moved)
//   B1     (k = d−1) rad = L_e (triangle) / L_env,fs (escape along rcWi), aux = p1_fs of the end (write-back cache)
//   L, E, R          rad = 0 (the shift re-evaluates everything)
// Deep records under TM_E2 (TD23) are undefined without evaluation (SXS_UNDEF | SXS_E2). Plants (validation only):
//   TP_N3_STALE        no refresh: stored rcRad / aux, stale end visibility (entry and J_P are still translated)
//   TP_N7_PER_LIGHT    records whose own light neither MOVED nor changed radiometrically keep the stored values
//                      (N3 / N7 are ignored by the robust check's inverse refresh, TM_ROBUST with fsTo = PREV: C-8)
//   TP_N4_RIS          forward only (frame-t random numbers): the D-NEE endpoint is re-drawn by RIS over 8 alias
//                      candidates (target lum(f_all·Λ), source q) and rad = the single-sample value of the pick
//                      (restir-temporal-api.md Changelog C-3)
//   TP_ENV_NO_ROT_VIS  no shadow ray for env NEE ends on env rotation (V := 1)
// (TP_NO_JP / TP_NO_JP_ENV: J_P := 1 here; TP_ENV_GAMMA_T and TP_N1_MIXED act in tframe.wgsl lf_env / lf_slot.)
// Test-only define RS_REFRESH_FORCE_MOVED: every NEE end counts as moved (exactness test of the visibility rule).
// Metal Q1: planes are read one at a time where they are used; fs is a u32 and the frame state comes from uniforms.
#include "restir/tframe.wgsl"
#include "restir/reservoir.wgsl"
#include "restir/endpoint.wgsl"
#include "restir/rc.wgsl"

// Path classes (math.md#reservoir-fields; layout.ts pathClass, restir-views.wgsl rsdbg_class)
const RFC_L: u32 = 0u;  const RFC_N1: u32 = 1u;  const RFC_B1: u32 = 2u;  const RFC_E: u32 = 3u;
const RFC_DNEE: u32 = 4u;  const RFC_DBSDF: u32 = 5u;  const RFC_R: u32 = 6u;

fn refresh_class(d: u32, k: u32, tech: u32) -> u32 {
  if (k == 0u) { return RFC_R; }
  let nee = tech == RS_TECH_NEE;
  if (k == d) { return select(RFC_E, RFC_L, nee); }
  if (k + 1u == d) { return select(RFC_B1, RFC_N1, nee); }
  return select(RFC_DBSDF, RFC_DNEE, nee);
}

fn refresh_count(c: u32, n: u32) {
#if RS_ARENA_BINDING && RS_ARENA_RW
  rs_count(c, n);
#endif
}

/// Translated entry and J_P of an NEE endpoint `entry` of frame fsFrom in frame fsTo (restir-temporal-api.md §3.2).
struct RfEntry { eTo: u32, jp: f32, undef: bool }
fn refresh_entry(entry: u32, fsFrom: u32, fsTo: u32) -> RfEntry {
  var r: RfEntry;
  r.eTo = lt_translate(entry, fsFrom, fsTo);
  let pTo = lt_pmf(r.eTo, fsTo);
  let pFrom = lt_pmf(entry, fsFrom);
  let jp = pTo / pFrom;
  // undefined iff the entry is missing or has pmf ≤ 0 in either frame (bit tests: NaN, Inf and ≤ 0 all fail)
  r.undef = !rs_pos_finite(pTo) || !rs_pos_finite(pFrom) || !rs_pos_finite(jp);
  r.jp = select(jp, 1.0, r.undef);
  let isEnv = entry == lf_slot(fsFrom).envEntry;
  if (rs_tplant(TP_NO_JP) || (isEnv && rs_tplant(TP_NO_JP_ENV))) { r.jp = 1.0; }
  return r;
}

/// Change bits (LCB_*) of the env as a BSDF-escape endpoint (independent of an env NEE entry).
fn refresh_env_bits() -> u32 {
  return select(0u, LCB_MOVED, rs_tf(TF_ENV_MOVED)) | select(0u, LCB_RADIO, rs_tf(TF_ENV_RADIO));
}

/// Frame key of atlas pixel ai in the current frame (N4 plant stream, restir-temporal-api.md §5).
fn refresh_key(ai: u32) -> vec2u {
  let W = rsParams.memberSize.x;
  let H = rsParams.memberSize.y;
  let px = vec2u(ai % rsParams.atlasSize.x, ai / rsParams.atlasSize.x);
  let mcol = px.x / W;
  let mrow = px.y / H;
  let local = vec2u(px.x - mcol * W, px.y - mrow * H);
  return rs_frame_key(rsParams.memberBase + mrow * rsParams.memberCols + mcol, rs_t(), local.y * W + local.x);
}
/// Four hashes of counter j (< 16) of the N4 plant stream (rs_rand's formula with pass RS_PASS_T_PLANT, all lanes).
fn refresh_plant_hash(key: vec2u, j: u32) -> vec4u {
  return pcg4d(vec4u(key.x, key.y ^ (RS_PASS_T_PLANT * 0x9e3779b9u), j, STREAM_RESAMPLE));
}

struct RfOut { rad: vec3f, aux: f32, status: u32 }

/// Class D-NEE: β_s ⊙ (ω1/q)·f_all·Λ at the cached x_{d−1} under frame fsTo (endpoint eTo, light-local words ab).
/// The shadow ray is traced iff `moved` (or the N4 plant re-drew the endpoint).
fn refresh_deep_nee(ai: u32, d: u32, eTo: u32, ab: vec2u, fsTo: u32, moved: bool) -> RfOut {
  var o: RfOut;
  o.status = SXS_DEEP;
  let sfx = resin_plane(ai, RP_SFX0);
  let V = bitcast<vec3f>(resin_plane(ai, RP_SFX1).xyz);
  let betaS = bitcast<vec3f>(resin_plane(ai, RP_SFX2).xyz);
  let x = scene_surface(sfx.x, bitcast<f32>(sfx.y), bitcast<f32>(sfx.z), -V);   // oriented toward x_{d−2} (§2.9)
  let m = material_eval(x, V);
  let slot = lf_slot(fsTo);
  let er = lf_env(fsTo);
  let n4 = rs_tplant(TP_N4_RIS) && fsTo == RS_FS_CUR;
  var key = vec2u(0u);
  if (n4) { key = refresh_key(ai); }
  var wSum = 0.0;
  var lsSel = light_sample_none();
  var rad = vec3f(0.0);
  // one call site each of nee_eval_s / bsdf_query (§4.5); one iteration unless the N4 plant draws 8 RIS candidates
  for (var j = 0u; j < select(1u, 8u, n4); j++) {
    var ep = NeeEndpoint(eTo, ab.x, ab.y);
    var uSel = 0.0;
    if (n4) {
      let h = refresh_plant_hash(key, j);
      let h2 = refresh_plant_hash(key, 8u + j);
      ep = nee_draw(slot, h.x, h.y, vec3u(h.z, h.w, h2.x));
      uSel = u32_to_unit(h2.y);
    }
    let ls = nee_eval_s(x.pos, ep, slot, er);
    let qn = bsdf_query(m, V, ls.dir, LOBE_NEE);
    let w1 = nee_mis_w1(ls, qn.p_marg, d - 1u);
    let ok = ls.valid && ls.prim != sfx.x && any(ls.Lambda > vec3f(0.0));
    var take = ok;
    if (n4) { take = ok && ris_update(&wSum, luminance(qn.f_all * ls.Lambda) / ls.q, uSel); }
    if (take) {
      rad = betaS * (w1 / ls.q) * qn.f_all * ls.Lambda;    // the path tree's grouping (pathtree.wgsl, D-NEE rcRad)
      lsSel = ls;
    }
  }
  var trace = moved || n4;
  if (lsSel.kind == LT_ENV && rs_tplant(TP_ENV_NO_ROT_VIS)) { trace = false; }
  if (trace && any(rad > vec3f(0.0))) {
    o.status |= SXS_RAY;
    refresh_count(RSC_T_REFRESH_RAYS, 1u);
    if (!nee_visible(x, sfx.x, lsSel)) { rad = vec3f(0.0); }
  }
  if (n4) { o.status |= SXS_PLANT; }
  o.rad = rad;
  return o;
}

/// Class D-BSDF: β_s ⊙ ω2·L of the cached BSDF end (escape along ω_{d−1}, or the stored emissive-triangle hit).
fn refresh_deep_bsdf(ai: u32, d: u32, tech: u32, fsTo: u32) -> vec3f {
  let sfx = resin_plane(ai, RP_SFX0);
  let s2 = resin_plane(ai, RP_SFX2);
  let betaS = bitcast<vec3f>(s2.xyz);
  let p2 = bitcast<f32>(s2.w);
  let delta = sfx_delta_end(sfx.w);
  let slot = lf_slot(fsTo);
  let er = lf_env(fsTo);
  if (tech == RS_TECH_BSDF_ENV) {
    let dir = bitcast<vec3f>(resin_plane(ai, RP_SFX1).xyz);
    let L = envRadiance_s(envUV(dir, er.cg, er.sg), er);
    let w2 = env_bsdf_mis_weight_s(dir, p2, d - 1u, delta, slot, er);
    return betaS * w2 * L;                                  // the path tree's grouping (betaPost * endW2 * endLe)
  }
  let xPos = scene_surface(sfx.x, bitcast<f32>(sfx.y), bitcast<f32>(sfx.z), vec3f(0.0)).pos;
  let end = resin_plane(ai, RP_END).xyz;
  let z = vertex_from_ids(end.x, bitcast<f32>(end.y), bitcast<f32>(end.z), xPos);
  let L = tri_emission(end.x, bitcast<f32>(end.y), bitcast<f32>(end.z));
  var w2 = 1.0;
  if (!delta) { w2 = mis_w2(tri_light_p1_s(xPos, z.pos, z.ng, end.x, slot), p2, d - 1u); }
  return betaS * w2 * L;
}

/// Class N1: Λ_fs and p1_fs at x_k = x_{d−1}, and the end visibility (ray iff moved; V = 1 otherwise, which is exact).
fn refresh_n1(ai: u32, eTo: u32, ab: vec2u, fsTo: u32, moved: bool) -> RfOut {
  var o: RfOut;
  o.status = SXS_N1;
  let sfx = resin_plane(ai, RP_SFX0);
  let V = bitcast<vec3f>(resin_plane(ai, RP_SFX1).xyz);
  let x = scene_surface(sfx.x, bitcast<f32>(sfx.y), bitcast<f32>(sfx.z), -V);
  let ls = nee_eval_s(x.pos, NeeEndpoint(eTo, ab.x, ab.y), lf_slot(fsTo), lf_env(fsTo));
  o.rad = ls.Lambda;
  o.aux = ls.p1;
  var vis = true;
  var trace = moved;
  if (ls.kind == LT_ENV && rs_tplant(TP_ENV_NO_ROT_VIS)) { trace = false; }
  if (trace) {
    vis = false;                                            // an invalid / zero end term has F = 0 either way
    if (ls.valid && any(ls.Lambda > vec3f(0.0))) {
      o.status |= SXS_RAY;
      refresh_count(RSC_T_REFRESH_RAYS, 1u);
      vis = nee_visible(x, sfx.x, ls);
    }
  }
  if (vis) { o.status |= SXS_VIS; }
  return o;
}

/// Class B1: L of the BSDF end (triangle: static L_e; escape: L_env,fs along rcWi) and its p1 under frame fsTo.
fn refresh_b1(ai: u32, tech: u32, fsTo: u32) -> RfOut {
  var o: RfOut;
  o.status = SXS_B1;
  let slot = lf_slot(fsTo);
  let er = lf_env(fsTo);
  if (tech == RS_TECH_BSDF_ENV) {
    let dir = bitcast<vec3f>(resin_plane(ai, RP_WI).xyz);
    o.rad = envRadiance_s(envUV(dir, er.cg, er.sg), er);
    o.aux = p1Env_s(dir, slot, er);
    return o;
  }
  let rc = resin_plane(ai, RP_RC).xyz;
  let xPos = scene_surface(rc.x, bitcast<f32>(rc.y), bitcast<f32>(rc.z), vec3f(0.0)).pos;
  let end = resin_plane(ai, RP_END).xyz;
  let z = vertex_from_ids(end.x, bitcast<f32>(end.y), bitcast<f32>(end.z), xPos);
  o.rad = tri_emission(end.x, bitcast<f32>(end.y), bitcast<f32>(end.z));
  o.aux = tri_light_p1_s(xPos, z.pos, z.ng, end.x, slot);
  return o;
}

fn refresh_record(ai: u32, fsFrom: u32, fsTo: u32) -> SfxRec {
  var r: SfxRec;
  r.rad = vec3f(0.0);
  r.aux = 0.0;
  r.status = 0u;
  r.entryTo = RC_NONE;
  r.jp = 1.0;
  r.gen = rsTemporal.frameGen;
  let flags = resin_plane(ai, RP_SEED).z;
  if (res_empty(flags) || (flags & RF_BG) != 0u) { return r; }       // empty / background: status 0
  refresh_count(RSC_T_REFRESH_RECS, 1u);
  r.status = SXS_DONE;
  let d = rf_d(flags);
  let tech = rf_tech(flags);
  let cls = refresh_class(d, rf_k(flags), tech);
  let end = resin_plane(ai, RP_END).xyz;
  var eTo = RC_NONE;
  var bits = 0u;                                                       // LCB_* of the endpoint's own light
  if (tech == RS_TECH_NEE) {
    let entry = end.x & RC_ENTRY_MASK;
    let te = refresh_entry(entry, fsFrom, fsTo);
    r.entryTo = te.eTo;
    r.jp = te.jp;
    if (te.undef) { r.status |= SXS_UNDEF; return r; }
    eTo = te.eTo;
    bits = lt_change_bits(select(entry, eTo, fsTo == RS_FS_CUR));    // the frame-t index of the endpoint
#if RS_REFRESH_FORCE_MOVED
    bits |= LCB_MOVED;
#endif
  } else if (tech == RS_TECH_BSDF_ENV) {
    bits = refresh_env_bits();
  } else if (tech == RS_TECH_BSDF_ANALYTIC) {
    r.status |= SXS_UNDEF;                                             // Mode-B crossings: not in M5 (D1)
    return r;
  }
  let deep = cls == RFC_DNEE || cls == RFC_DBSDF;
  if (deep && rs_tmode(TM_E2)) { r.status |= SXS_UNDEF | SXS_E2; return r; }
  if (!(deep || cls == RFC_N1 || cls == RFC_B1)) { return r; }        // L, E, R: the shift re-evaluates everything
  // stale plants: the stored cache values (N3 always; N7 when the endpoint's own light is unchanged). Not in the robust
  // check's inverse (TM_ROBUST, fsTo = PREV): T6(b) compares the stored π_p(Y_p) with a FRESH inverse evaluation
  // (gap-temporal §9.2; Changelog C-8, B-10)
  let stale = rs_tplant(TP_N3_STALE) || (rs_tplant(TP_N7_PER_LIGHT) && (bits & (LCB_MOVED | LCB_RADIO)) == 0u);
  if (stale && !(rs_tmode(TM_ROBUST) && fsTo == RS_FS_PREV)) {
    r.rad = bitcast<vec3f>(resin_plane(ai, RP_RAD).xyz);
    var cs = SXS_B1;
    if (deep) { cs = SXS_DEEP; } else if (cls == RFC_N1) { cs = SXS_N1 | SXS_VIS; }
    if (!deep) { r.aux = bitcast<f32>(resin_plane(ai, RP_WI).w); }
    r.status |= SXS_PLANT | cs;
    return r;
  }
  let moved = (bits & LCB_MOVED) != 0u;
  switch (cls) {
    case RFC_DNEE: {
      let o = refresh_deep_nee(ai, d, eTo, end.yz, fsTo, moved);
      r.rad = o.rad;
      r.status |= o.status;
    }
    case RFC_DBSDF: {
      r.rad = refresh_deep_bsdf(ai, d, tech, fsTo);
      r.status |= SXS_DEEP;
    }
    case RFC_N1: {
      let o = refresh_n1(ai, eTo, end.yz, fsTo, moved);
      r.rad = o.rad;
      r.aux = o.aux;
      r.status |= o.status;
    }
    default: {
      let o = refresh_b1(ai, tech, fsTo);
      r.rad = o.rad;
      r.aux = o.aux;
      r.status |= o.status;
    }
  }
  if (deep && !any(r.rad > vec3f(0.0))) { r.status |= SXS_ZERO; }
  return r;
}

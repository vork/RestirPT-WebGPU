// V2 "Cycles Principled Tier 1" closure setup, distribution = 'GGX' (math.md#bsdf-v2; gap-bsdf §6;
// Cycles svm/closure.h:169-509, closure/bsdf_microfacet.h:423-500 (albedo estimate), :832-921 (Fresnel setups),
// closure/bsdf_util.h:488-492 (closure_layering_weight), closure/alloc.h:55-80 (cutoff, sample_weight)).
//
// Input clamps: C = max(Base, 0), Cc = min(C, 1), r = saturate(Roughness), α = r², m = saturate(Metallic),
// η = max(IOR, 1e-5), L_s = max(Specular IOR Level, 0), T_s = max(Specular Tint, 0); weight = mix_weight·alpha = 1.
// Closure order (each closure allocated iff |avg(weight_i)| ≥ 1e-5):
//   1. metal     (m > 1e-5): w_M = m·weight, F82-tint with F0 = Cc, f82 = min(T_s, 1); then weight *= (1 − m)
//   2. glass     (model 2, t = saturate(Transmission Weight) > 1e-5): w_G = t·weight, generalized Schlick with the
//                NODE IOR (η_side = 1/ior when backfacing), f0 = saturate(F0(ior)·T_s), f90 = 1, reflection tint 1,
//                transmission tint √Cc (per interface) — glass.wgsl; then weight *= (1 − t) (no albedo layering)
//   3. IOR level: f0 = F0(η); if L_s ≠ 0.5: f0 *= 2L_s, η' = ior_from_F0(f0), η' = 1/η' if η < 1; else η' = η
//   4. specular  (η' ≠ 1): w_S = weight, generalized Schlick f0_S = saturate(f0·T_s), f90 = 1, exponent = −η';
//                then layering weight ← weight·saturate(1 − max_c E_S,c(μ)),  E_S = mix(f0_S, 1, S_ior(r, μ, z_S))
//   5. diffuse   w_D = C·weight (unclamped C, normal N)
// Albedo-scaled sample weights (Cycles-exact, noise only; cyc-verify omission 1):
//   sw_M = m·avg(mix(Cc, 1, S5(r, μ))),  sw_S = (1 − m)·avg(E_S(μ)),  sw_D = |avg(C)|·(1 − m)·Λ_S(μ)
//   (each estimate is 0 when the closure's Fresnel at μ is exactly 0 — Cycles' is_zero(reflectance) short-cut)
// q(·|V) depends on V ⇒ V2 is NON-reciprocal (gap-bsdf §6.9): always evaluate with V toward the camera-side vertex.
// Tier 2 (MULTI_GGX) is not implemented (validation forces distribution = 'GGX').

fn bsdf_setup_v2(m: MatEval, c_in: BsdfCtx, mu: f32) -> BsdfCtx {
  var c = c_in;
  let C = max(m.base_color, vec3f(0.0));
  let Cc = min(C, vec3f(1.0));
  let met = saturate(m.metallic);
  let ior = max(m.ior, 1e-5);
  let level = max(m.specular_level, 0.0);
  let tint = max(m.specular_tint, vec3f(0.0));
  var weight = 1.0;   // mix_weight · alpha (grey; alpha = 1 — MASK cut-outs are handled by traversal)

  // 1. metal [closure.h:344-375]
  if (met > BSDF_WEIGHT_CUTOFF) {
    let wM = met * weight;
    if (abs(wM) >= BSDF_WEIGHT_CUTOFF) {
      c.bits |= BC_HAS_S | BC_HAS_METAL;
      c.w_m = wM;
      c.s_a = Cc;                                           // F82 F0
      c.s_b = fresnel_f82_tint_B(Cc, min(tint, vec3f(1.0)));
      // bsdf_microfacet_estimate_albedo, F82_TINT branch (B ignored: Cycles TODO)
      var est = vec3f(0.0);
      if (any(fresnel_f82(mu, c.s_a, c.s_b) != vec3f(0.0))) {
        est = mix(Cc, vec3f(1.0), lut_ggx_gen_schlick_s(bctx_rough(c), mu, 0.5));
      }
      c.q_s += abs(wM) * bsdf_avg3(est);                   // raw sample weight (normalised in bsdf_prepare)
    }
    weight *= (1.0 - met);
  }

  // 2. transmission: glass closure (lobe class G) [closure.h:377-415]
  if (m.model == BSDF_MODEL_GLASS) {
    let t = saturate(m.transmission);
    if (t > BSDF_WEIGHT_CUTOFF) {
#if GLASS_PLANT == 3
      c = bsdf_setup_glass_schlick(c, t * weight, ior, fresnel_F0_from_ior(ior) * tint, Cc, false, mu);   // plant B-tint
#else
      c = bsdf_setup_glass_schlick(c, t * weight, ior, fresnel_F0_from_ior(ior) * tint, sqrt(Cc), false, mu);
#endif
      weight *= (1.0 - t);
    }
  }

  // 3. IOR level [closure.h:417-426]
  var eta = ior;
  var f0 = fresnel_F0_from_ior(eta);
  if (level != 0.5) {
    f0 *= 2.0 * level;
    eta = fresnel_ior_from_F0(f0);
    if (ior < 1.0) { eta = 1.0 / eta; }
  }

  // 4. dielectric specular + layering [closure.h:428-462]
  if (eta != 1.0 && abs(weight) >= BSDF_WEIGHT_CUTOFF) {
    c.bits |= BC_HAS_S | BC_HAS_SPEC;
    c.w_s = weight;
    c.eta_s = eta;
    c.f0_s = saturate(f0 * tint);                           // bsdf_microfacet_setup_fresnel_generalized_schlick
    var est = vec3f(0.0);                                    // E_S / weight (reflection_tint = 1)
    if (any(fresnel_gen_schlick_ior(mu, eta, c.f0_s) != vec3f(0.0))) {
      let z = sqrt(abs((eta - 1.0) / (eta + 1.0)));
      est = mix(c.f0_s, vec3f(1.0), lut_ggx_gen_schlick_ior_s(bctx_rough(c), mu, z));
    }
    c.q_s += abs(weight) * bsdf_avg3(est);
    // closure_layering_weight(albedo = weight·est, weight): safe_divide_color → est where weight ≠ 0 (grey weight > 0 here)
    weight = weight * saturate(1.0 - bsdf_max3(est));
  }

  // 5. diffuse (Lambert: Diffuse Roughness = 0) [closure.h:494-509]
  let wD = max(C * weight, vec3f(0.0));
  let swD = abs(bsdf_avg3(wD));
  if (swD >= BSDF_WEIGHT_CUTOFF) {
    c.bits |= BC_HAS_D;
    c.w_d = wD;
    c.q_d = swD;
  }
  return c;
}

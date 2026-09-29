// V1 "validation" BSDF closure setup (math.md#bsdf-v1; gap-bsdf §5; Cycles svm/closure.h:531 (Diffuse),
// :686-741 (Glossy, fresnel NONE), :1543-1566 (Mix), closure/alloc.h:55-80 (cutoff, sample_weight)).
//   f    = saturate(Fac)
//   w_D  = max((1 − f)·ρ, 0)     allocated iff |avg(w_D)| ≥ 1e-5
//   w_G  = max(f·k, 0)           allocated iff |avg(w_G)| ≥ 1e-5,  α = saturate(r)², F ≡ 1
//   q(D) = |avg(w_D)| / (|avg(w_D)| + |avg(w_G)|)  (plain sample weights, V-independent; GGX Glossy has no albedo scaling)
// V1 is reciprocal.

fn bsdf_setup_v1(m: MatEval, c_in: BsdfCtx) -> BsdfCtx {
  var c = c_in;
  let f = saturate(m.metallic);                        // MatEval V1 slots: metallic = mix
  let wD = max((1.0 - f) * m.base_color, vec3f(0.0));   // base_color = Diffuse
  let wG = max(f * m.specular_tint, vec3f(0.0));        // specular_tint = Glossy
  let swD = abs(bsdf_avg3(wD));
  let swG = abs(bsdf_avg3(wG));
  if (swD >= BSDF_WEIGHT_CUTOFF) { c.bits |= BC_HAS_D; c.w_d = wD; c.q_d = swD; }
  if (swG >= BSDF_WEIGHT_CUTOFF) { c.bits |= BC_HAS_S; c.s_a = wG; c.q_s = swG; }   // q_*: raw sample weights
  return c;
}

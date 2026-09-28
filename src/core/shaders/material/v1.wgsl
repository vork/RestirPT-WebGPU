// V1 "validation" BSDF closure setup (math.md#bsdf-v1; gap-bsdf §5; Cycles svm/closure.h:531 (Diffuse),
// :686-741 (Glossy, fresnel NONE), :1543-1566 (Mix), closure/alloc.h:55-80 (cutoff, sample_weight)).
//   f    = saturate(Fac)
//   w_D  = max((1 − f)·ρ, 0)     allocated iff |avg(w_D)| ≥ 1e-5
//   w_G  = max(f·k, 0)           allocated iff |avg(w_G)| ≥ 1e-5,  α = saturate(r)², F ≡ 1
//   q(D) = |avg(w_D)| / (|avg(w_D)| + |avg(w_G)|)  (plain sample weights, V-independent; GGX Glossy has no albedo scaling)
// V1 is reciprocal.

fn bsdf_setup_v1(m: MatEval, c_in: BsdfCtx) -> BsdfCtx {
  var c = c_in;
  let f = saturate(m.v1_mix);
  let wD = max((1.0 - f) * m.v1_diffuse, vec3f(0.0));
  let wG = max(f * m.v1_glossy, vec3f(0.0));
  let swD = abs(bsdf_avg3(wD));
  let swG = abs(bsdf_avg3(wG));
  c.has_d = swD >= BSDF_WEIGHT_CUTOFF;
  c.has_s = swG >= BSDF_WEIGHT_CUTOFF;
  if (c.has_d) { c.w_d = wD; c.sw_d = swD; }
  if (c.has_s) { c.w_g = wG; c.sw_s = swG; }
  return c;
}

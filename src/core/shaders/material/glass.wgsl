// Lobe class G: Cycles' GGX glass / refraction closure, distribution 'GGX', Tier 1 (math.md#glass; gap-glass §2, §5,
// §9; Cycles closure/bsdf_microfacet.h:226-357 (microfacet_fresnel), :423-499 (estimate_albedo), :586-675 (eval),
// :677-820 (sample); svm/closure.h:377-415 (Principled transmission), :745-777 (Refraction node), :779-832 (Glass node)).
//
// One class with two sub-events decided by the sign of Ns·L: G_R (reflection, L above) and G_T (transmission, L below).
//   η_side   = back ? 1/ior : ior      back: the EVALUATING V lies on the backfacing side of the winding normal
//                                       (stateless: recomputed for every V, never stored, no medium stack)
//   F        = mix(f0, 1, saturate((F_diel(H·V, η_side) − F0(η_side)) / (1 − F0(η_side))))   (real Fresnel remapped)
//   R = F·rt,  T = (1 − F)·tt;   Refraction node (Fresnel NONE): R = 0, T = (F_diel == 1 ? 0 : 1)
//   P_R      = avg(R)/avg(R + T) at the microfacet (TIR ⇒ F = 1 ⇒ T = 0 ⇒ always reflect)
//   H        = isT ? −(η_side·L + V) : V + L  (normalised, NOT re-oriented: every use is sign invariant)
//   common   = D(N·H)/cNI · (isT ? (η_side·invLen)²·|cHI·(H·L)| : 1/4)    (refraction half-vector Jacobian included)
//   f·cos    = w_G·(isT ? T : R)·common/(1 + Λ_I + Λ_O)      Cycles eval — the spurious refraction region INCLUDED
//   pdf_C    = common·(isT ? 1 − P_R : P_R)/(1 + Λ_I)         Cycles' eval pdf (diagnostics only: ∫ pdf_C can exceed 1)
//   pdf_V    = pdf_C·1[valid]                                  TRUE sampler density (MIS, footprints, Jacobians, χ²)
//   valid    : G_R: Ng·L ≥ 0;  G_T: Ng·L < 0 ∧ Hn·V > 0 ∧ Hn·L < 0 ∧ T(Hn) > 0 ∧ |η_side − 1| ≥ 1e-4,
//              Hn = H oriented to N·Hn ≥ 0 (for η < 1 the physical H_t points below N)
//   delta    : G_R iff α² ≤ 2e-10; G_T iff α² ≤ 2e-10 ∨ |η_side − 1| < 1e-4 (bsdf_microfacet.h:771)
// NO 1/η² radiance scaling anywhere (glass §1 item 3): a smooth dielectric has albedo exactly 1 from both sides.
// Requires the BsdfCtx of bsdf.wgsl (BC_HAS_G / BC_BACK / BC_G_* bits, g_w, g_eta, g_f0, g_tt), ggx.wgsl, fresnel.wgsl,
// lut.wgsl. The closure parameters are stored compactly (bsdf.wgsl BsdfCtx): closure weight g_w (grey), transmission
// tint g_tt, reflection tint = g_tt (Glass node, BC_G_RT_TT) or 1; the Refraction node keeps its Colour in g_tt.

const GLASS_ETA_SINGULAR_EPS: f32 = 1e-4;   // bsdf_microfacet.h:771

// Gate-1 planted biases (validation only; compile-time define GLASS_PLANT, absent = 0 = off):
//   1 "B-η"     the BTDF gets a 1/η_side² radiance scaling (eval and sampled weights)
//   2 "P_R 0.5" the R/T decision uses 0.5 instead of P_R, the weights still divide by P_R (no pdf compensation)
//   3 "B-tint"  the Principled glass closure takes C instead of √C per interface (v2.wgsl)
//   4 "B-side"  η is not inverted on backfaces (η_side = ior on both sides)
//   5 "B-shadow" shadow / any-hit rays pass through glass materials (traverse.wgsl any-hit, scene-data.wgsl)

/// η_side from the node IOR (Cycles clamps max(ior, 1e-5)) and the backfacing test of the evaluating V.
fn glass_eta_side(ior: f32, back: bool) -> f32 {
  let i = max(ior, 1e-5);
#if GLASS_PLANT == 4
  return i;
#else
  return select(i, 1.0 / i, back);
#endif
}

struct GlassFresnel {
  R: vec3f,
  T: vec3f,
  cosT: f32,   // signed cosine of the refracted direction w.r.t. the microfacet (< 0), 0 under TIR
}

/// microfacet_fresnel at the microfacet cosine cos_hi = H·V (sign-free; bsdf_util.h:47-99 fresnel_dielectric).
/// G_T singular: α² ≤ 2e-10 or |η_side − 1| < 1e-4.
fn glass_delta_t(c: BsdfCtx) -> bool { return bctx_singular(c) || abs(c.g_eta - 1.0) < GLASS_ETA_SINGULAR_EPS; }

fn glass_fresnel(c: BsdfCtx, cos_hi: f32) -> GlassFresnel {
  var o: GlassFresnel;
  let eta = c.g_eta;
  let g = eta * eta - (1.0 - cos_hi * cos_hi);
  var Freal = 1.0;
  if (g > 0.0) {
    let ci = abs(cos_hi);
    let ct = -sqrt(g) / eta;
    o.cosT = ct;
    let rs = (ci + eta * ct) / (ci - eta * ct);
    let rp = (ct + eta * ci) / (eta * ci - ct);
    Freal = 0.5 * (rs * rs + rp * rp);
  }
  if (bctx_has(c, BC_G_FNONE)) {                     // Refraction node: no reflection; TIR kills the path
    o.T = select(c.g_tt, vec3f(0.0), Freal == 1.0);
    return o;
  }
  let F0r = fresnel_F0_from_ior(eta);
  let s = saturate((Freal - F0r) / (1.0 - F0r));
  let F = mix(c.g_f0, vec3f(1.0), s);
  o.R = select(F, F * c.g_tt, bctx_has(c, BC_G_RT_TT));
  o.T = (vec3f(1.0) - F) * c.g_tt;
  return o;
}

struct GlassEval {
  f: vec3f,      // w_G·(R|T)·common/(1+Λ_I+Λ_O): Cycles eval (includes |N·L|), spurious part included
  pdfC: f32,     // Cycles' eval pdf (P_R / 1 − P_R included, class pmf q(G) NOT included)
  pdfV: f32,     // valid-only = the true sampler density of the G sampler (q(G) not included)
  isT: bool,     // sub-event: Ns·L < 0
  valid: bool,   // (V, L) lies in the sampler support (math.md#support-indicator G_R / G_T)
}

/// Eval of the glass closure for (V, L), V toward the previous vertex (Cycles bsdf_microfacet_eval + validity).
fn glass_eval(c: BsdfCtx, V: vec3f, L: vec3f) -> GlassEval {
  var e: GlassEval;
  if (!bctx_has(c, BC_HAS_G) || bctx_singular(c)) { return e; }   // singular: no eval (no SD_BSDF_HAS_EVAL)
  let cNI = dot(c.ns, V);
  let cNO = dot(c.ns, L);
  if (!(cNI > 0.0)) { return e; }
  e.isT = cNO < 0.0;
  if (!e.isT && !bctx_has(c, BC_G_HAS_R)) { return e; }
  let Hu = select(V + L, -(c.g_eta * L + V), e.isT);
  let len2 = dot(Hu, Hu);
  if (!(len2 > 0.0)) { return e; }
  let invLen = inverseSqrt(len2);
  let H = Hu * invLen;
  let cHI = dot(H, V);
  let fr = glass_fresnel(c, cHI);
  let aR = bsdf_avg3(fr.R);
  let aT = bsdf_avg3(fr.T);
  if (!(aR + aT > 0.0)) { return e; }
  let cNH = dot(c.ns, H);
  let nxh = cross(c.ns, H);
  let a2 = bctx_a2(c);
  let D = ggx_D(a2, cNH * cNH, dot(nxh, nxh));
  let lI = ggx_lambda(a2, cNI);
  let lO = ggx_lambda(a2, cNO);
  let k = c.g_eta * invLen;
  let jac = D / cNI * select(0.25, k * k * abs(cHI * dot(H, L)), e.isT);   // Cycles "common"
  let pR = aR / (aR + aT);
  e.pdfC = jac * select(pR, 1.0 - pR, e.isT) / (1.0 + lI);
  e.f = select(fr.R, fr.T, e.isT) * (c.g_w * jac / (1.0 + lI + lO));
#if GLASS_PLANT == 1
  if (e.isT) { e.f /= c.g_eta * c.g_eta; }
#endif
  if (e.isT) {
    let Hn = select(-H, H, cNH >= 0.0);
    e.valid = !glass_delta_t(c) && dot(c.ng, L) < 0.0 && dot(Hn, V) > 0.0 && dot(Hn, L) < 0.0 && aT > 0.0;
  } else {
    e.valid = dot(c.ng, L) >= 0.0 && aR > 0.0;
  }
  e.pdfV = select(0.0, e.pdfC, e.valid);
  return e;
}

struct GlassSample {
  L: vec3f,
  ok: bool,             // false: rejected (no Fresnel energy, wrong Ng/Ns side, cNI ≤ 0)
  isT: bool,
  isDelta: bool,        // singular sub-event (α² ≤ 2e-10, or refraction with |η_side − 1| < 1e-4)
  weightDelta: vec3f,   // delta: w_G·(R|T)/((P_R | 1 − P_R)·q(G))  (Cycles: eval·1e6 / pdf·1e6)
}

/// Direction sampler of the glass closure: u = (u_h1, u_h2, u_rt) (math.md#rng-layout); the class pick used u_lobe.
fn glass_sample(c: BsdfCtx, V: vec3f, u: vec3f) -> GlassSample {
  var s: GlassSample;
  let cNI = dot(c.ns, V);
  if (!(cNI > 0.0)) { return s; }
  let H = ggx_sample_h(c.alpha, bctx_singular(c), c.ns, V, u.xy);
  let cHI = dot(H, V);
  let fr = glass_fresnel(c, cHI);
  let aR = bsdf_avg3(fr.R);
  let aT = bsdf_avg3(fr.T);
  if (!(aR + aT > 0.0)) { return s; }
  let pR = aR / (aR + aT);
#if GLASS_PLANT == 2
  let refr = u.z >= 0.5 && aT > 0.0;
#else
  let refr = u.z >= pR;
#endif
  let inv = 1.0 / c.g_eta;
  s.L = normalize(select(2.0 * cHI * H - V, (inv * cHI + fr.cosT) * H - inv * V, refr));   // refract_angle / reflect
  s.isT = refr;                                     // the attempted sub-event (also reported for a rejected sample)
  if ((dot(c.ng, s.L) < 0.0) != refr || (dot(c.ns, s.L) < 0.0) != refr) { return s; }        // :757-761
  s.ok = true;
  s.isDelta = bctx_singular(c) || (refr && abs(c.g_eta - 1.0) < GLASS_ETA_SINGULAR_EPS);
  if (s.isDelta) {
    s.weightDelta = select(fr.R, fr.T, refr) * (c.g_w / (select(pR, 1.0 - pR, refr) * c.q_g));
#if GLASS_PLANT == 1
    if (refr) { s.weightDelta /= c.g_eta * c.g_eta; }
#endif
  }
  return s;
}

// ------------------------------------------------------------------------------------------------ closure setup

/// Generalized-Schlick glass closure (Principled transmission, Glass node): grey closure weight w, node IOR, f0
/// (saturated here, bsdf_microfacet.h:869), transmission tint tt (reflection tint tt if rtIsTt, else 1); leaves the Cycles
/// albedo-scaled sample weight at μ = Ns·V in q_g (bsdf_microfacet_estimate_albedo: smooth-surface Fresnel at N, the
/// reflection part from the gen-Schlick LUT; normalised by bsdf_prepare).
fn bsdf_setup_glass_schlick(c_in: BsdfCtx, w: f32, ior: f32, f0: vec3f, tt: vec3f, rtIsTt: bool, mu: f32) -> BsdfCtx {
  var c = c_in;
  if (abs(w) < BSDF_WEIGHT_CUTOFF) { return c; }     // bsdf_alloc cutoff (alloc.h)
  c.bits |= BC_HAS_G | BC_G_HAS_R | select(0u, BC_G_RT_TT, rtIsTt);
  c.g_w = w;
  c.g_eta = glass_eta_side(ior, bctx_has(c, BC_BACK));
  c.g_f0 = saturate(f0);
  c.g_tt = tt;
  let fr = glass_fresnel(c, mu);
  var R = fr.R;
  if (any(R != vec3f(0.0))) {
    let z = sqrt(abs((c.g_eta - 1.0) / (c.g_eta + 1.0)));
    let Rn = mix(c.g_f0, vec3f(1.0), lut_ggx_gen_schlick_ior_s(bctx_rough(c), mu, z));
    R = select(Rn, Rn * c.g_tt, rtIsTt);
  }
  c.q_g = abs(w) * bsdf_avg3(R + fr.T);
  return c;
}

/// Cycles Glass BSDF node: weight = mix_weight (1), f0 = F0(ior), reflection_tint = transmission_tint = max(Color, 0).
fn bsdf_setup_glass_node(m: MatEval, c_in: BsdfCtx, mu: f32) -> BsdfCtx {
  return bsdf_setup_glass_schlick(c_in, 1.0, m.ior, vec3f(fresnel_F0_from_ior(max(m.ior, 1e-5))), max(m.base_color, vec3f(0.0)), true, mu);
}

/// Cycles Refraction BSDF node: weight = Color·mix_weight (kept in g_tt, g_w = 1), Fresnel NONE (R = 0, T = 1 except
/// TIR), no albedo scaling of the sample weight (sw = |avg(Color)|).
fn bsdf_setup_refraction_node(m: MatEval, c_in: BsdfCtx) -> BsdfCtx {
  var c = c_in;
  let sw0 = abs(bsdf_avg3(m.base_color));
  if (sw0 < BSDF_WEIGHT_CUTOFF) { return c; }
  c.bits |= BC_HAS_G | BC_G_FNONE;
  c.g_w = 1.0;
  c.g_tt = m.base_color;
  c.g_eta = glass_eta_side(m.ior, bctx_has(c, BC_BACK));
  c.q_g = sw0;
  return c;
}

// Environment lighting: importance sampling (NEE_ENV), p1Env and the env MIS weights (plan §1.4b, §2 rules 2 and 15;
// math.md#env-sampling, #measure, #mis; env §2.4, §3.1). CPU mirror of the tables: scene/env/env-importance.ts.
//
// Tables (in `records`, offsets in LightsParams): rowAlias[H_m], colAlias[H_m·W_m] (entry = alias << 16 | q, u16
// thresholds), pdfUV[H_m·W_m] = the REALIZED density in (u, v) of that two-level alias sampler (f32 bits). The env is
// alias entry slot.envEntry (LIGHT_NONE when env NEE is off ≡ Cycles world sampling_method NONE: BSDF-only, ω2 = 1).
//   NEE_ENV   (i, j) from the alias (hashes h0, h1), in-cell lattice offsets (du16, dv16) = h2, uv CLAMPED inside the
//             cell (env-verify 1), s = sin(π·min(v, 1 − v));  q = p1 = pmf[ENV]·pdfUV[i,j]/(2π² s);  Λ = L_env(uv)
//             (no r², no cos); visibility to infinity (visibleInf, t_max = FLT_MAX). s < 1e-6 (pole cap) ⇒ rejected.
//   BSDF_ENV  escape along ω: ω2 = p2/(M(B)·p1Env(ω) + p2), p1Env(ω) = pmf[ENV]·pdfUV[cell(envUV(ω))]/(2π² s),
//             s = ‖b.xy‖ of the direction (env-verify 2), pole cap s < 1e-6 ⇒ p1 = 0 ⇒ ω2 = 1; ω2 = 1 after a delta lobe.
// The env is not an analytic light: it ALWAYS uses MIS (modes A, A′, B), because escaping BSDF rays always see it.
//
// Validation-only defines (default off; identity otherwise):
//   ENV_PLANT = 1  planted bias "sinθ missing in pdf_σ" (both techniques)
//             = 2  planted bias "ω2 without pmf[ENV]"
//             = 3  planted bias "NEE and BSDF both weight 1" (double count)
//   ENV_MIS_POWER  power heuristic for the env (negative control: any partition of unity is unbiased)
// ("strength ×1.0075", "pdf from target weights", "no floors" and "importance-resolution change" are CPU-side: env
//  params / env-importance.ts options.)
#include "lights/measure.wgsl"
#include "lights/env.wgsl"

const ENV_POLE_CAP: f32 = 1e-6;
const ENV_INV_2PI2: f32 = 0.050660591821168885;   // 1/(2π²): dω = 2π² sinθ du dv

fn env_tables_present() -> bool { return lightsParams.envLog2W != 0u; }
fn env_Wm() -> u32 { return 1u << lightsParams.envLog2W; }
fn env_Hm() -> u32 { return 1u << (lightsParams.envLog2W - 1u); }

/// Realized density in (u, v) of cell (i, j) (f32 as stored).
fn env_pdf_uv(i: u32, j: u32) -> f32 {
  return bitcast<f32>(records[lightsParams.envPdfOff + i * env_Wm() + j]);
}

/// Cell (j, i) of a texture coordinate (u, v) ∈ [0, 1]² (u = 1 only from envUV at −X; clamped).
fn env_cell_of(uv: vec2f) -> vec2u {
  let W = env_Wm();
  let H = W >> 1u;
  return vec2u(min(u32(max(uv.x, 0.0) * f32(W)), W - 1u), min(u32(max(uv.y, 0.0) * f32(H)), H - 1u));
}

/// Solid-angle density given "env chosen": pdfUV[i,j]/(2π² s), 0 inside the pole cap (math.md#env-sampling).
fn env_pdf_sa_cell(i: u32, j: u32, s: f32) -> f32 {
  if (!(s >= ENV_POLE_CAP)) { return 0.0; }
#if ENV_PLANT == 1
  return env_pdf_uv(i, j) * ENV_INV_2PI2;               // PLANTED: sinθ missing
#else
  return env_pdf_uv(i, j) * ENV_INV_2PI2 / s;
#endif
}

/// Env solid-angle density of a world DIRECTION d (escapes, camera misses, refresh) under rotation (cg, sg):
/// s = ‖b.xy‖ = sinθ of the true direction (well conditioned at the poles; env-verify 2).
fn envPdfSA(d: vec3f, cg: f32, sg: f32) -> f32 {
  if (!env_tables_present()) { return 0.0; }
  let b = envToBlender(d, cg, sg);
  let c = env_cell_of(envUV(d, cg, sg));
  return env_pdf_sa_cell(c.y, c.x, length(b.xy));
}

/// p1Env(ω) = pmf[ENV]·pdf_σ(ω): the ONE env light pdf (NEE q, BSDF-escape MIS, later RIS-NEE, shifts, refresh).
/// 0 when env NEE is off (no alias entry).
fn p1Env(d: vec3f) -> f32 {
  let slot = lightsParams.cur;
  if (slot.envEntry == LIGHT_NONE) { return 0.0; }
  return light_pmf(slot, slot.envEntry) * envPdfSA(d, envParams.cg, envParams.sg);
}

/// One draw of the two-level alias + in-cell lattice offsets. uv is clamped inside cell (i, j) (prevFloat of the upper
/// edge), so the pdf of the sampled point is exactly pdfUV[i, j] (math.md#env-sampling "CELL CLAMP").
struct EnvCellSample { i: u32, j: u32, uv: vec2f, s: f32 }

fn env_prev_float(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) - 1u); }

fn env_sample_cell(h0: u32, h1: u32, h2: u32) -> EnvCellSample {
  let log2W = lightsParams.envLog2W;
  let log2H = log2W - 1u;
  let W = 1u << log2W;
  let H = 1u << log2H;
  let i0 = h0 >> (32u - log2H);                       // log2H ≥ 1 (W_m ≥ 4): never a shift by 32
  let er = records[lightsParams.envRowOff + i0];
  let i = select(er >> 16u, i0, (h0 & 0xffffu) < (er & 0xffffu));
  let j0 = h1 >> (32u - log2W);
  let ec = records[lightsParams.envColOff + i * W + j0];
  let j = select(ec >> 16u, j0, (h1 & 0xffffu) < (ec & 0xffffu));
  let du16 = h2 >> 16u;
  let dv16 = h2 & 0xffffu;
  var u = (f32(j) + (f32(du16) + 0.5) / 65536.0) / f32(W);
  var v = (f32(i) + (f32(dv16) + 0.5) / 65536.0) / f32(H);
  u = min(u, env_prev_float(f32(j + 1u) / f32(W)));
  v = min(v, env_prev_float(f32(i + 1u) / f32(H)));
  return EnvCellSample(i, j, vec2f(u, v), sin(PI * min(v, 1.0 - v)));
}

/// NEE sample of the env alias entry (math.md#measure: q = p1 = pmf[ENV]·pdf_σ(uv), Λ = L_env(uv), isInf).
fn env_light_sample(slot: LightSlot, entry: u32, h: vec3u) -> LightSample {
  var ls = light_sample_none();
  let c = env_sample_cell(h.x, h.y, h.z);
  let q = light_pmf(slot, entry) * env_pdf_sa_cell(c.i, c.j, c.s);
  ls.entry = entry;
  ls.kind = LT_ENV;
  ls.dir = envDir(c.uv, envParams.cg, envParams.sg);
  ls.pos = vec3f(0.0);
  ls.nz = vec3f(0.0);
  ls.dist = FLT_MAX;
  ls.cosZ = 1.0;
  ls.Lambda = envRadiance(c.uv);                        // evaluated at the sampled uv (no direction round trip)
  ls.q = q;
  ls.p1 = q;                                            // already a solid-angle density (no r²/cos)
  ls.isDelta = false;
  ls.isInf = true;
  ls.analytic = false;                                  // the env always uses MIS (never Mode-A NEE-only)
  ls.valid = q > 0.0;                                   // pole cap: rejected, F = 0 (BSDF covers it with ω2 = 1)
  return ls;
}

/// One NEE sample at x over the whole global alias table (analytic lights, emissive triangles, ENV).
/// hSel/hSel2: slots u_sel/u_sel2; hL = (h_l0, h_l1, h_l2): area lights use u01(h_l0), u01(h_l1) (identical to the
/// M3a path_u01 values), the env uses the three raw hashes.
fn nee_sample(x: vec3f, hSel: u32, hSel2: u32, hL: vec3u) -> LightSample {
  let slot = lightsParams.cur;
  if (slot.nEntries == 0u) { return light_sample_none(); }
  let entry = alias_sample(slot, hSel, hSel2);
  if (entry == slot.envEntry) { return env_light_sample(slot, entry, hL); }
  return light_sample_entry(x, slot, entry, vec2f(u32_to_unit(hL.x), u32_to_unit(hL.y)));
}

fn env_power_w(a: f32, b: f32) -> f32 {                 // a²/(a² + b²), 0/0 guarded
  let a2 = a * a;
  let d = a2 + b * b;
  return select(1.0, a2 / d, d > 0.0);
}

/// ω1 of an NEE sample (env: balance heuristic with M(B), like every non-delta light; see mis_w1).
fn nee_mis_w1(ls: LightSample, p2: f32, B: u32) -> f32 {
  if (ls.kind != LT_ENV) { return mis_w1(ls, p2, B); }
#if ENV_PLANT == 3
  return 1.0;                                           // PLANTED: double count
#elif ENV_MIS_POWER
  return env_power_w(mis_M(B) * ls.p1, max(p2, 0.0));
#else
  return mis_w1(ls, p2, B);
#endif
}

/// ω2 of a BSDF ray that escapes along `dir` from a vertex with BSDF marginal p2 (math.md#mis rule 5):
/// p2/(M(B)·p1Env + p2); 1 after a delta lobe (MIS skip) and when env NEE is off (BSDF is the only technique).
fn env_bsdf_mis_weight(dir: vec3f, p2: f32, B: u32, afterDelta: bool) -> f32 {
  if (afterDelta) { return 1.0; }
  let slot = lightsParams.cur;
  if (slot.envEntry == LIGHT_NONE) { return 1.0; }
#if ENV_PLANT == 3
  return 1.0;                                           // PLANTED: double count
#else
#if ENV_PLANT == 2
  let p1 = envPdfSA(dir, envParams.cg, envParams.sg);   // PLANTED: ω2 without pmf[ENV]
#else
  let p1 = p1Env(dir);
#endif
#if ENV_MIS_POWER
  return select(1.0, env_power_w(max(p2, 0.0), mis_M(B) * p1), p1 > 0.0);
#else
  return mis_w2(p1, p2, B);
#endif
#endif
}

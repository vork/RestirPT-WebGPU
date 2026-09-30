# BSDF / material WGSL API (contract for M3a+, glass since M3b)

Owner modules: `src/core/shaders/material/*.wgsl` (`bsdf.wgsl` dispatch, `v1.wgsl`, `v2.wgsl`, `glass.wgsl`, `ggx.wgsl`,
`fresnel.wgsl`, `lambert.wgsl`, `lut.wgsl`, `material-eval.wgsl`). Consumers: path tracer (M3a), NEE, shifts (M4+).
Normative math: `docs/math.md#bsdf-v1`, `#bsdf-v2`, `#glass`, `#pdf-conventions`, `#support-indicator`, `#rng-layout`.
Direction convention (Cycles): `V` points toward the previous vertex (camera side), `L` toward the next vertex.
**"eval" (`f_cos`) includes |Ns·L|.**

```wgsl
// Lobe codes (math.md#rng-layout / plan §1.5): 3 bits + delta flag carried separately.
const LOBE_D: u32 = 0u;   const LOBE_S: u32 = 1u;   const LOBE_GR: u32 = 2u;   const LOBE_GT: u32 = 3u;
const LOBE_NEE: u32 = 4u; const LOBE_NONE: u32 = 5u;

// Everything the BSDF needs at a hit, built once per vertex by material_eval(...) in material/material-eval.wgsl.
struct MatEval {
  model: u32,            // 0 V1 validation, 1 V2 Principled (Tier 1), 2 V2 + glass closure (Transmission Weight > 1e-5),
                         // 3 Cycles Glass BSDF node, 4 Cycles Refraction BSDF node
  base_color: vec3f,     // V2: Base Color (texture × factor × COLOR_0, linear) | V1: Lambert albedo | 3/4: Colour
  metallic: f32,         // V2: metallic | V1: mix factor ((1−mix)·Diffuse + mix·Glossy)
  roughness: f32,        // perceptual r (GGX α = r²); V1: glossy roughness; 3/4: node Roughness
  ior: f32,              // V2 / glass: node IOR (η_side = ior, or 1/ior on the backfacing side)
  specular_level: f32,   // Principled "Specular IOR Level" (0.5 neutral)
  specular_tint: vec3f,  // V2: Specular Tint | V1: glossy colour
  transmission: f32,     // Principled Transmission Weight (model 2)
  ns: vec3f,             // shading normal, flipped to the V side (two-sided shading frame)
  ng: vec3f,             // geometric normal, flipped to the V side
  flags: u32,            // MATEVAL_*: bit0 has non-delta lobe; bit1 diffuse-only; bits 2–6 hasD/hasS/S-delta/hasG/G-delta;
                         // bit7 MATEVAL_BACKFACING (INPUT: ns/ng are the negated winding normal)
}

struct BsdfEval {       // all-lobe evaluation (NEE) — Cycles eval semantics
  f_cos: vec3f,
  pdf_marginal: f32,     // TRUE sampler density of the one-sample mixture (support-consistent), solid angle
}

struct BsdfSample {
  L: vec3f,
  weight: vec3f,         // f_ℓ·cos / (q(ℓ|V)·p_ℓ(L)) using the JOINT pdf (lobe-indexed throughput); delta: see below
  pdf_joint: f32,        // q(ℓ|V)·p_ℓ(L); 0 for delta
  pdf_marginal: f32,     // Σ_ℓ q(ℓ|V)·p_ℓ(L) over non-delta lobes (MIS, footprints); 0 for delta
  lobe: u32,             // LOBE_*: D, S, G_R or G_T (the attempted sub-event also for a rejected glass sample)
  is_delta: bool,
  valid: bool,           // false if the sampler rejected (Ng/Ns side, no Fresnel energy, TIR of the Refraction node)
}

fn bsdf_eval(m: MatEval, V: vec3f, L: vec3f) -> BsdfEval;                       // NEE: all lobes
fn bsdf_sample(m: MatEval, V: vec3f, u: vec4f) -> BsdfSample;                    // u = (u_lobe, u_h1, u_h2, u_rt)
fn bsdf_eval_lobe(m: MatEval, V: vec3f, L: vec3f, lobe: u32) -> vec4f;          // (f_cos.rgb, pdf_joint) for shifts
fn bsdf_pdf_marginal(m: MatEval, V: vec3f, L: vec3f) -> f32;                    // MIS for BSDF-hit emitters
fn bsdf_sample_support(m: MatEval, V: vec3f, L: vec3f, lobe: u32) -> bool;      // math.md#support-indicator
fn lobe_roughness(m: MatEval, lobe: u32) -> f32;                                 // perceptual r: D=1, S/G_R=r, G_T=r or 0
fn bsdf_query(m: MatEval, V: vec3f, L: vec3f, lobe: u32) -> BsdfQuery;          // ReSTIR (M4): all of the above for one (V, L)
// diagnostics / tests: bsdf_lobe_probs (q_D, q_S, q_G, η_side), bsdf_sample_weights (Cycles sw_ℓ), bsdf_glass_diag,
// bsdf_flags (MatEval.flags for a V; preserves MATEVAL_BACKFACING)
```

Rules:
- `bsdf_eval` = Cycles eval (no Ng test on L), including the V2 albedo-LUT layering weight `(1−m)(1−t)·(1−E_spec(V))`
  on the diffuse lobe **and the glass closure's spurious rough-refraction region** (`bsdf_microfacet.h:621` TODO; NEE
  must count it for parity). `bsdf_sample` rejects directions whose Ng/Ns side mismatches the lobe exactly as Cycles
  (math.md#support-indicator), and **every pdf** (marginal, joint, `bsdf_eval_lobe`) is the true sampler density:
  for glass the *valid-only* density `pdf_C·1[valid]` (G_T: Ng·L < 0 ∧ Hn·V > 0 ∧ Hn·L < 0 ∧ T(Hn) > 0 ∧
  |η_side − 1| ≥ 1e-4). The spurious region therefore has p2 = 0 and ω1 = 1 (support-consistent MIS).
- Lobe selection q(ℓ|V) = Cycles' albedo-scaled sample weights (V2, glass: `|avg(w_G)|·avg(R_est + T_est)`, R_est from
  the gen-Schlick LUT, T_est the smooth Fresnel at N) / plain weights (V1, Refraction node), normalized; the class pick
  is `u_lobe` against the cumulative q(D), q(S), q(G), the glass R/T decision `u_rt ≥ P_R(H)`.
- Glass (`glass.wgsl`, math.md#glass): η_side = 1/ior iff the **evaluating** V lies on the backfacing side of the winding
  normal (bsdf_prepare: `(Ng·V < 0) ⊕ MATEVAL_BACKFACING`), never stored, no medium stack. Principled transmission tint
  √min(C, 1) per interface, reflection untinted, f0 = saturate(F0(ior)·Specular Tint); Glass node: Colour on R and T;
  Refraction node: R = 0, TIR kills the path. **No 1/η² radiance scaling** (the BTDF is Cycles', non-reciprocal:
  f_T(V,L)/|N·L| = η_side(V)²·f_T(L,V)/|N·V|).
- Delta: a lobe is singular iff α_x·α_y ≤ 2e-10 → `is_delta`, weight = F/q(S) (S), `w_G·(R|T)/((P_R | 1−P_R)·q(G))`
  (G), pdfs 0, never evaluated by NEE; G_T is also delta when |η_side − 1| < 1e-4 (G_R stays rough). The next emitter
  hit after a delta lobe has MIS weight 1.
- `bsdf_eval_lobe(G_R | G_T)` returns the glass eval restricted to its sub-event side (Ns·L ≥ 0 | < 0) with the
  valid-only joint pdf; f carries no support indicator (multiply by `bsdf_sample_support` for BSDF-sampled segments).
- `bsdf_query` (M4, restir-api.md §3.1) returns `{f_lobe, p_joint, f_all, p_marg, supp}` from **one** `bsdf_prepare` +
  **one** `bsdf_eval_ctx`: `f_lobe, p_joint` ≡ `bsdf_eval_lobe`, `f_all, p_marg` ≡ `bsdf_eval` (= `bsdf_pdf_marginal`),
  `supp` ≡ `bsdf_sample_support` (true for LOBE_NEE / LOBE_NONE). U-PT-BITS part 2 checks the agreement (measured bitwise
  on 4.2·10⁶ random queries). ReSTIR modules call only `bsdf_query`, `bsdf_sample`, `material_eval` and `lobe_roughness`
  (inline budget per pipeline: ≤ 1 `bsdf_sample`, ≤ 2 `bsdf_query`, ≤ 2 `material_eval`).
- The LUT tables live in the `records` storage buffer at offsets provided by `$LUT_*` defines (material/lut.wgsl).
  The Tier-2 glass tables (`cycles-glass-luts.ts`) are extracted for the CPU tests only (Tier 2 is M7).

## BsdfCtx / MatEval size (a platform constraint, found in M3b)

The BSDF library is inlined at every call site. On this platform (Apple M5 Pro, macOS 26, Chrome 154 **and** dawn.node)
the Metal compiler **silently miscompiles** compute kernels whose per-thread state crosses a size threshold: no
validation error, no device loss, but wrong values (u32 counters holding float bit patterns) or no writes at all. The
M3a BSDF χ² harness (one `bsdf_sample` + `bsdf_sample_support` + two `bsdf_eval_lobe` + `bsdf_pdf_marginal` per loop
iteration) broke when the context struct grew by ~4 vec3 (measured with dummy fields on the M3a code: +1 ok, +4 broken;
the first glass version added 4 vec3 + 12 scalars/bools). Therefore:
- `BsdfCtx` is 8 × 16 B: booleans are bits (`BC_*`), α² / √α / F0(η) / the glass delta flags are derived on use, the
  raw sample weights live in the `q_*` fields until `bsdf_prepare` normalises them, and fields of mutually exclusive
  models are unions (`s_a` = V1 glossy colour | V2 metal F0; the glass tint `g_tt` doubles as the Refraction colour).
- `MatEval` reuses the V2 slots for V1 and carries no emission (the PT reads `tri_emission()`).
- New GPU tests / shifts that call several BSDF entry points per loop iteration must be checked on the Chrome lane for
  this failure mode (a kernel that "runs" in ~0 s or whose counters are implausible); prefer one `bsdf_prepare` per
  vertex and the ctx-level functions (`bsdf_eval_ctx`) over repeated public calls.

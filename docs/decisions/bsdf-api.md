# BSDF / material WGSL API (contract for M3a+)

Owner modules: `src/core/shaders/material/*.wgsl`. Consumers: path tracer (M3a), NEE, shifts (M4+).
Normative math: `docs/math.md#bsdf-v1`, `#bsdf-v2`, `#glass` (M3b), `#pdf-conventions`, `#support-indicator`,
`#rng-layout`. Direction convention (Cycles): `V` points toward the previous vertex (camera side), `L` toward the
next vertex. **"eval" (`f_cos`) includes |Ns·L|.**

```wgsl
// Lobe codes (math.md#rng-layout / plan §1.5): 3 bits + delta flag carried separately.
const LOBE_D: u32 = 0u;   const LOBE_S: u32 = 1u;   const LOBE_GR: u32 = 2u;   const LOBE_GT: u32 = 3u;
const LOBE_NEE: u32 = 4u; const LOBE_NONE: u32 = 5u;

// Everything the BSDF needs at a hit, built once per vertex by material_eval(...) in material/material-eval.wgsl.
struct MatEval {
  model: u32,            // 0 = V1 validation, 1 = V2 Principled (Tier 1), 2 = glass-capable V2 (M3b)
  base_color: vec3f,     // after texture × factor × COLOR_0 (linear)
  metallic: f32,
  roughness: f32,        // perceptual r (GGX α = r²); V1: glossy roughness
  ior: f32,
  specular_level: f32,   // Principled "Specular IOR Level" (0.5 neutral)
  specular_tint: vec3f,
  transmission: f32,     // M3b
  emission: vec3f,       // L_e (two-sided), radiance units
  v1_diffuse: vec3f,     // V1 only
  v1_glossy: vec3f,      // V1 only
  v1_mix: f32,           // V1 only: (1-mix)·Diffuse + mix·Glossy
  ns: vec3f,             // shading normal, flipped to the V side (two-sided shading frame)
  ng: vec3f,             // geometric normal, flipped to the V side
  flags: u32,            // bit0: has non-delta lobe; bit1: diffuse-only (Lambert is the only allocated lobe)
}

struct BsdfEval {       // all-lobe evaluation (NEE) — Cycles eval semantics
  f_cos: vec3f,
  pdf_marginal: f32,     // TRUE sampler density of the one-sample mixture (support-consistent), solid angle
}

struct BsdfSample {
  L: vec3f,
  weight: vec3f,         // f_ℓ·cos / (q(ℓ|V)·p_ℓ(L)) using the JOINT pdf (lobe-indexed throughput)
  pdf_joint: f32,        // q(ℓ|V)·p_ℓ(L)
  pdf_marginal: f32,     // Σ_ℓ q(ℓ|V)·p_ℓ(L) over non-delta lobes (MIS, footprints)
  lobe: u32,             // LOBE_*
  is_delta: bool,
  valid: bool,           // false if the sampler rejected (e.g. Ng·L on the wrong side, TIR in M3b)
}

fn bsdf_eval(m: MatEval, V: vec3f, L: vec3f) -> BsdfEval;                       // NEE: all lobes
fn bsdf_sample(m: MatEval, V: vec3f, u: vec4f) -> BsdfSample;                    // u = (u_lobe, u_h1, u_h2, u_rt)
fn bsdf_eval_lobe(m: MatEval, V: vec3f, L: vec3f, lobe: u32) -> vec4f;          // (f_cos.rgb, pdf_joint) for shifts
fn bsdf_pdf_marginal(m: MatEval, V: vec3f, L: vec3f) -> f32;                    // MIS for BSDF-hit emitters
fn bsdf_sample_support(m: MatEval, V: vec3f, L: vec3f, lobe: u32) -> bool;      // math.md#support-indicator
fn lobe_roughness(m: MatEval, lobe: u32) -> f32;                                 // perceptual r: D=1, S=roughness
```

Rules:
- `bsdf_eval` = Cycles eval (no Ng test on L), including the V2 albedo-LUT layering weight `(1−m)·(1−E_spec(V))` on
  the diffuse lobe. `bsdf_sample` rejects directions whose Ng/Ns side mismatches the lobe exactly as Cycles
  (math.md#support-indicator), and its pdfs are the true sampler densities.
- Lobe selection q(ℓ|V) = Cycles' albedo-scaled sample weights (V2) / plain weights (V1), normalized.
- Singular GGX (αx·αy ≤ 2e-10) → delta lobe: `is_delta`, weight = F/q(S), pdfs 0, never evaluated by NEE.
- The LUT tables live in the `records` storage buffer at offsets provided by `$LUT_*` defines (material/lut.wgsl).

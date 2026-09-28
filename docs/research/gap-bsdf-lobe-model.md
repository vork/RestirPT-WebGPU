# Gap-fill `bsdf-lobe-model`: one normative BSDF / lobe spec for the WebGPU ReSTIR PT renderer (V1 "validation" and V2 "Cycles Principled Tier 1/2")

> **Where this file is.** Plan mode was active in this session, and plan mode only allows writing this plan file. The requested path `/private/tmp/claude-502/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-ea73-491f-b7b7-6d0aab1ffce1/scratchpad/research/gap-bsdf-lobe-model.md` was **not** written. Copy this file there if the pipeline needs it. Nothing was cloned or saved: GitHub sources and `shader.tables` were streamed to stdout, and numeric checks ran in Blender's bundled Python 3.13 (numpy) reading from stdin.

---

## 0. Provenance, tags, notation

**Primary sources read in this session.**

| Key | Source | How |
|---|---|---|
| **[CY]** | Cycles kernel shipped with the installed Blender 5.1.2: `/Applications/Blender.app/Contents/Resources/5.1/scripts/addons_core/cycles/source/kernel/...` (this is the exact kernel the Metal reference renders compile) | Read in place. `file:line` refers to this tree. |
| **[CYH]** | Cycles host code at tag `v5.1.2`: `intern/cycles/scene/shader.tables`, `app/cycles_precompute.cpp`, `scene/shader.cpp`, `scene/shader_nodes.cpp` | `curl` of raw.githubusercontent to stdout |
| **[GLTF]** | Installed glTF importer `.../addons_core/io_scene_gltf2/blender/imp/{pbrMetallicRoughness,material,mesh}.py` (add-on 5.1.20) | Read in place |
| **[FAL]** | Falcor in `DQLin/ReSTIR_PT` @ `8d12332228eb64bc234e27c6f7e0913a926285ab`: `Source/Falcor/Rendering/Materials/BxDF.slang`, `IBxDF.slang`, `Source/RenderPasses/ReSTIRPTPass/PathTracer.slang` | `curl` to stdout |
| **[EVL]** | `EvanLuo42/ReSTIR-PT-Enhanced` @ `b6d30d90d71e0482b1ce73c8e8a7b65bdb077ee8` (2026-08-21), `Source/RenderPasses/ReSTIRPTPass/{Common/EnhancedPolicy,Common/HybridShift,Path/PathSampling,Common/SurfaceLoad}.slang` | `curl` to stdout |
| **[GRIS]** | GRIS paper, `scratchpad/pages/gris/p-14.png` (§7.5–§8, Eqs. 49–51) | Page image read |
| **[ENH]** | Enhanced paper, `scratchpad/pages/enhanced/p-04.png` (Eq. 2), `p-07.png` (Eqs. 5–6), `p-08.png` (§4.1–4.2, footnotes 5–6) | Page images read |
| **[BPY]** | Headless read-only introspection, `Blender --background --factory-startup --python-expr …` | Run in this session |

**Prior reports consulted** (all under `/Users/mark.boss/.claude/plans/`, prefix `do-a-deep-dive-shimmering-ritchie-agent-`): cycles-conventions `a2d6cc1a462b20a80` §4 and its verification `ac3ddf11ffb75ef90` (C2, C4, omissions 1–5); validation-harness `a26c6df2c3f5e4af0` §7.4 and T8; scene-io `a597df4a1dd101994` §3.2–3.3 and §4.3; gris-math `a7f08dec9968f0324` §5.2 and §6.5–6.6, verification `a147516e5d2fb9558` C9; enhanced-paper `a91ab1d861819b276` §3.5–3.6, verification `acb243dbea3ce2a29` O1, O2, O8; reference-code `a3b89657b22d15f34` §1.5.2 steps 8–9 and §2.3; critique `a40267d04641507fa` X5, X6.

**Tags.**
- **[SOURCE]**: stated by a paper or code at the cited location.
- **[VERIFIED]**: I checked it in this session, by reading code or by running a computation.
- **[INFERENCE]**: my own derivation or design decision.
- **[UNVERIFIED]**: plausible, not checked. A suggested check is given.
- **NORMATIVE**: a rule the implementation must follow. Changing it requires changing every consumer listed with it.

**Direction convention (Cycles, used throughout).**
- `V` (Cycles `sd->wi`) is the unit vector from the shading point toward the *previous* vertex, i.e. toward the camera side.
- `L` (Cycles `wo`) is the unit vector toward the *next* vertex or the light.
- **"eval" always includes the cosine of L**: `eval(V,L) = f_r(V,L)·max(N·L,0)`. This is the Cycles convention: the diffuse eval is `max(N·L,0)/π` ([CY] `closure/bsdf_diffuse.h:38`) and the microfacet eval is `F·D·G2/(4·N·V)` ([CY] `bsdf_microfacet.h:666-674`).
- Paths are only ever traced from the camera, so every vertex has a well-defined V. This matters because V2 is **non-reciprocal** (§6.9).

Symbols: `N` shading normal, `Ng` geometric normal, `Ns` normal used by the GGX lobes, `r` perceptual roughness, `α = r²` GGX width, `a2 = α_x·α_y`, `μ = Ns·V`, `q(ℓ|V)` lobe-selection probability, `p_ℓ(L|V)` lobe-conditional solid-angle pdf, `p(L,ℓ|V) = q(ℓ|V)·p_ℓ(L|V)` joint pdf, `p_all(L|V) = Σ_ℓ q(ℓ|V)·p_ℓ(L|V)` marginal pdf.

---

## 1. TL;DR: the normative decisions

1. **Two BSDF variants, one code path.**
   - **V1 "validation"** = Lambert + single-scatter GGX Glossy (F ≡ 1) + constant Mix + Emission. Blender: Diffuse BSDF (Roughness 0), Glossy BSDF `distribution='GGX'`, Mix Shader, Add Shader + Emission.
   - **V2 "Cycles Principled Tier 1"** = Principled BSDF with `distribution='GGX'`: Lambert diffuse layered under a generalized-Schlick dielectric GGX specular (port of the 4096-float LUT `table_ggx_gen_schlick_ior_s`), plus an F82-tint metal GGX lobe.
   - **Tier 2** = V2 plus MULTI_GGX energy compensation (`table_ggx_E[1024]`, `table_ggx_Eavg[32]`).
   - All three are exactly matchable to Cycles in expectation (§2.1).
2. **Lobe set is two classes: `D` (Lambert) and `S` (GGX reflection).** In V2, the metal closure and the dielectric-specular closure share Ns, α and the VNDF sampler, so they merge exactly into one class `S` (§6.4). Two more lobe codes, `NEE` and `NONE`, mark NEE-sampled and light vertices. A lobe code is 2 bits.
3. **Path space is lobe-extended (GRIS §7.6, Falcor `separatePathBSDF`).** NORMATIVE:
   - A BSDF-sampled vertex contributes `f_ℓ·cos / p(L,ℓ)` with the **joint** pdf.
   - An NEE vertex evaluates all lobes.
   - NEE/BSDF MIS uses the **marginal** p_all.
   - The PSS reconnection Jacobian uses **joint** pdfs.
   - The Enhanced footprint tests use the **marginal** p_all, as in the paper.
   - Mixing conventions, for example a marginal throughput with a joint Jacobian, is **biased** (§4.4).
4. **Lobe-selection probabilities.** Any strictly positive `q(ℓ|V)` is unbiased, and q only changes noise (proof in §4.3). **V2 uses Cycles' albedo-scaled `sample_weight` exactly**; V1 uses Cycles' plain weights. This gives MIS weights identical to Cycles at no extra cost, because the LUT is needed anyway (§6.6).
5. **Stored per reservoir:** a 2-bit lobe code plus a delta bit for the vertex *before* the reconnection vertex (x_{k−1}) and for the reconnection vertex itself (x_k). Prefix lobes are regenerated by replay and are not stored. Suffix lobes are baked into the cached radiance (§7.3).
6. **Per-lobe roughness for the reconnection predicate** is perceptual: r_D = 1, r_S = Roughness input (= `sqrt(sqrt(α_x α_y))`), r_delta = 0. An NEE vertex takes the max over allocated lobes, and a light vertex counts as rough. The threshold is `r ≥ 0.2` (§7.4). Do **not** copy EvanLuo42, which compares GGX α = r² against 0.2 (§7.4).
7. **Delta lobe** iff `α_x·α_y ≤ 2e-10`, i.e. r ≤ 0.003761. It gets no NEE eval, its throughput is `F/q(S)`, and an emitter hit after it has MIS weight 1 (§5.6, §6.8).
8. **Enhanced footnote 6 ("diffuse x_k skips the inverse footprint test").** The skip is exact iff `p_all(·|x_k, ω_in)` does not depend on ω_in. For V1/V2 that means **x_k is diffuse-only**, i.e. the Lambert lobe is the only allocated lobe, or x_k is a light vertex. A diffuse *lobe sampled* on a mixed material does **not** qualify (§7.5).
9. **Normals.** Jacobian and footprint cosines use **Ng**. Sampling rejects directions below Ng; eval does not. Under smooth shading this makes Cycles' MIS a non-partition, so **validate with flat shading first**. For smooth shading, shifted BSDF-sampled segments must carry the sampler-support indicator (§8).
10. **Textures.** For sRGB colour textures use `rgba8unorm` (**not** `-srgb`), `textureSampleLevel(...,0)`, bilinear, no mips, then apply the exact sRGB EOTF in the shader. Cycles filters the bytes first and decodes afterwards ([CY] `svm/image.h:28-38`) (§9).
11. **Tier 2 is required for GLBs imported with Blender's defaults.** The glTF importer never sets `distribution`, so it stays MULTI_GGX. The difference is up to **3.26×** for a white metal at roughness 1 and 2.8× for coloured rough metals, and +0.3…+2.9 % on dielectric diffuse. A GLB with no MR data defaults to a rough metal (metallicFactor = roughnessFactor = 1). For the validation harness, forcing `distribution='GGX'` makes Tier 1 sufficient (§6.10).

---

## 2. How the contradictions resolve

### 2.1 X5: can Principled match Cycles? Yes, exactly, in expectation.

- validation-harness §0.7/§7.4 says Principled "cannot be matched". **That is wrong for `distribution='GGX'`.**
- Everything Principled Tier 1 does with the validation subset of inputs is either closed-form or a lookup into one 16×16×16 float table:
  - Coat, sheen, transmission, SSS, thin film, anisotropy and Diffuse Roughness are 0; Alpha is 1.
  - Code: [CY] `svm/closure.h:98-529`, `bsdf_microfacet.h:423-499`, `util/lookup_table.h`.
- I re-derived the table's meaning numerically [VERIFIED]: `S(r, μ, z) = E_VNDF[ s(V·H) · 1(N·L > 0) ]`, with `s = saturate((F_diel(V·H, η) − F0(η))/(1 − F0(η)))` and `η = ior_from_F0(z⁴)`.
  - My quadrature matched the table to ≤ 2e-4 at 5 grid points (e.g. r = 8/15, μ = 0.4, z = 7/15: numeric 0.06131 vs LUT 0.06128).
  - This confirms the precompute ([CYH] `app/cycles_precompute.cpp:89-133`: mean of `saturate(eval.x/eval.y)` from `bsdf_microfacet_ggx_sample`, 0 when the sample is rejected).
- The same check shows `table_ggx_E` is the single-scatter GGX albedo `E_VNDF[G2/G1·1(N·L>0)]` [VERIFIED: 0.91569 numeric vs 0.91528 table at r = 0.5, μ = 1; 0.30685 vs 0.30685 at r = 1].
- So Tier 2 is also exactly matchable.

### 2.2 X6: `rgba8unorm-srgb` is wrong for parity.

scene-io §3.2, webgpu-platform §3.2 and validation-harness §7.4 recommend hardware sRGB decode. Cycles does this instead [VERIFIED]:
- On GPU it samples the 8-bit texture through a hardware `filter::linear` sampler with no mip filter ([CY] `device/gpu/image.h:85-117`, `device/metal/compat.h:388-395`).
- Only *after* interpolation does it call `color_srgb_to_linear_v4` ([CY] `svm/image.h:36-38`).
- On Metal (no `__KERNEL_SSE2__`) that function uses the exact piecewise EOTF ([CY] `util/color.h:63-69, 347-358`).

Details in §9.

### 2.3 O2 (joint vs marginal) and O8 (footnote 6), and gris-verify C9 (lobe classes, perceptual roughness)

- **O2.** The joint pdf goes into J. The marginal pdf goes into MIS and the footprint. This is codified in the §7.1 table and applies to both variants.
- **O8.** "Diffuse-only" is a property of the whole BSDF at x_k (§7.5). It is not a property of the sampled lobe.
- **C9.** Falcor's lobe classes are `SampledBSDFFlags` 0x3 = {DiffuseReflection 1, DiffuseTransmission 2} and 0xC = {SpecularReflection 4, SpecularReflectionTransmission 8} ([FAL] `IBxDF.slang:60-66`). `linearRoughness` is perceptual (`alpha = sd.linearRoughness * sd.linearRoughness`, [FAL] `BxDF.slang:840`). Our D maps to Falcor 0x1 ⊂ 0x3, and our S maps to Falcor 0x4 ⊂ 0xC.

### 2.4 cycles-verify C2 and C4 (carried into this spec)

- **C2.** The singular microfacet case returns `eval = F·1e6` and `pdf = lobe_prob·1e6` ([CY] `bsdf_microfacet.h:767-782`). For a reflection-only closure lobe_prob = 1, so the closure's sample weight is F. We implement this as a true delta lobe (§5.6).
- **C4.** `F_gs = F_diel` only for η ≤ 2+√3 ≈ 3.73. Port the `saturate` literally (it is in the WGSL, §11).

### 2.5 cycles-verify omissions 1–5, as they affect this spec

1. **Albedo-scaled sample weights** are fully specified in §6.6. They are noise only.
2. **NEE skips closures with pdf = 0.** Automatic in our `eval_all` (§11).
3. **Smooth-shading MIS non-partition.** §8.2.
4. **Normal-map node math.** §8.4. This is post-flat-stage work.
5. **Alpha / transparent BSDF.** Out of scope for V1/V2. Validation requires Alpha = 1 (§10.3).

---

## 3. Shared building blocks (both variants)

All formulas below are [SOURCE] from the cited Cycles lines unless tagged.

### 3.1 Two-sided shading frame

- [SOURCE] If `Ng·V < 0`, Cycles negates both Ng and N before any closure is built (`geom/shader_data.h:108-116`, cited by cycles-conventions §4.11 and verified there). It has no one-sided BSDFs.
- NORMATIVE: do this flip in `bsdf_prepare` from the *actual* incoming direction of the path being evaluated.
- For a shifted path, the reconnection vertex x_k is re-prepared with `V = normalize(y_{k−1} − x_k)`.
- If y_{k−1} lies behind x_k's surface, the flip changes. Then `N·ω_k < 0`, f = 0, and the shift fails naturally.

### 3.2 GGX (isotropic in V1/V2)

```
a2         = α_x·α_y                   (α_x = α_y = α = r², r = saturate(Roughness))       [closure.h:219, 694; bsdf_microfacet.h:961-964]
D(cosNH)   = a2 / (π·((1−c²) + a2·c²)²),   c² = min(cosNH², 1)                            [bsdf_microfacet.h:553-563]
Λ(cosN)    = ½·(sqrt(1 + a2·max(1/cosN² − 1, 0)) − 1)                                    [:507-528]
G1(V)      = 1/(1+Λ(N·V));   G2 = 1/(1 + Λ(N·V) + Λ(N·L))   (height-correlated Smith)      [:539-549]
eval_S     = F(V·H) · D · G2 / (4·N·V)          with H = normalize(V+L)                  [:620-674]
p_VNDF(L)  = D · G1(V) / (4·N·V) = D / (4·N·V·(1+Λ(N·V)))                                 [:673]
valid iff  N·V > 0, a2 > 2e-10, N·L ≥ 0 (N·L < 0 is "transmission", rejected for reflection closures)   [:614-618]
```

**VNDF sampler** (Heitz 2018), [CY] `bsdf_microfacet.h:189-220`:
1. Stretch: `V_s = normalize(α_x V.x, α_y V.y, V.z)`.
2. Build T1/T2. If `lensq > 1e-7`, `T1 = (−V_s.y, V_s.x, 0)/√lensq` and `T2 = V_s × T1`; otherwise T1 = x, T2 = y.
3. `t = concentric_disk(u)`, then `t.y = mix(√(1−t.x²), t.y, ½(1+V_s.z))`.
4. `H_s = t.x T1 + t.y T2 + √(1−|t|²) V_s`.
5. Unstretch: `H = normalize(α_x H_s.x, α_y H_s.y, max(0, H_s.z))`.
6. `L = 2(V·H)H − V`.

The sampler **rejects** the sample if `Ng·L < 0` or `N·L < 0` for reflection (`:755-765`). Eval does **not** test Ng (the comment is at `:756-760`). Rejection returns `LABEL_NONE`, which terminates the path.

### 3.3 Lambert

- Eval: `eval_D = max(N·L,0)/π`, `pdf = max(N·L,0)/π` ([CY] `bsdf_diffuse.h:30-41`).
- Sampling: cosine hemisphere about N using the **concentric** disk map ([CY] `sample/mapping.h:71-83` and `sample_uniform_disk`).
- The sample is rejected (pdf = 0) unless **`Ng·L > 0` (strict)** (`bsdf_diffuse.h:57-63`).

### 3.4 Fresnel models

```
F0(η)            = ((η−1)/(η+1))²                                                         [bsdf_util.h:338-341]
ior_from_F0(f)   = (1+√f)/(1−√f),  f clamped to [0, 0.99]                                  [:332-336]
F_diel(c, η)     = ½(r_s² + r_p²),  exact unpolarised; 1 under TIR                          [:47-99]
                   g = η² − (1 − c²); if g ≤ 0 → 1; ci = |c|; ct = −√g/η;
                   r_s = (ci + η ct)/(ci − η ct);  r_p = (ct + η ci)/(η ci − ct)
F_gs(c; η, f0)   = mix(f0, 1, saturate((F_diel(c,η) − F0(η))/(1 − F0(η))))   ("exponent<0" generalized Schlick)  [bsdf_microfacet.h:313-320]
F82(c; F0, B)    = saturate(F0 + (1−F0)s⁵ − B·c·s⁶),  s = saturate(1−c)                    [bsdf_util.h:178-184]
B(F0, tint)      = 0 if tint ≡ (1,1,1) else (F0 + (1−F0)f⁵)·(7/f⁶)·(1 − tint),  f = 6/7      [bsdf_util.h:146-158; bsdf_microfacet.h:908-913]
Fss_diel(η)      = (η−1)/(4.08567 + 1.00071η) for η ≥ 1                                   [bsdf_util.h:131-137]   (Tier 2 only)
Fss_F82(F0,B)    = mix(F0, 1, 1/21) − B/126                                              [:140-143]              (Tier 2 only)
```

### 3.5 LUT reader: exact port of `util/lookup_table.h`

Per axis, the reader does:
```
x' = saturate(x)·(n−1);  i = min(trunc(x'), n−1);  j = min(i+1, n−1);  t = x' − i
if t == 0 return d[i]
return (1−t)·d[i] + t·d[j]
```
- Axes are nested z (outer) → y → x (inner).
- The layout is `data[z·n² + y·n + x]`.
- `x = rough = sqrt(sqrt(α_x α_y))`, which is the perceptual roughness r for isotropic lobes. `y = μ = dot(V, Ns)`. `z` depends on the table.

NORMATIVE rules:
- Store the tables in a read-only **storage buffer** of f32 and port the reader literally.
- Do not use `textureSampleLevel` on a 3D texture. `r32float` is not filterable without `float32-filterable`, and hardware weights are quantised (cycles-verify omission 6).

The tables are copied verbatim from [CYH] `intern/cycles/scene/shader.tables` at v5.1.2. Checksums I computed [VERIFIED]:

| Table | Floats | Σ over table | Used by |
|---|---|---|---|
| `table_ggx_gen_schlick_ior_s` | 4096 | 184.247126 | V2 dielectric layering + sample weight |
| `table_ggx_gen_schlick_s` | 4096 | 764.768581 | V2 metal sample weight (noise only). Only slices z = 7, 8 are read (z = 0.5). |
| `table_ggx_E` | 1024 | 849.736018 | Tier 2 |
| `table_ggx_Eavg` | 32 | 25.879163 | Tier 2 |

Total is 9248 floats = 36,992 B. Put them in one buffer, or append them to an existing static-scene storage buffer to save one of the 10 storage-buffer bindings the M5 Pro allows (webgpu-platform measured limit, critique X10).

---

## 4. The lobe-indexed path space (math, for both variants)

### 4.1 What the sources say

- **[SOURCE] GRIS §7.6, Eqs. 49–51 (p.14).**
  - Paths are paired with lobe sequences `ℓ = (ℓ_1..ℓ_{d−1})`, with `ℓ_{d−1} = 𝒩` if the last vertex is NEE-sampled.
  - `f_ℓ` evaluates only lobe `ℓ_j` at each BSDF-sampled vertex, and **all lobes** at an NEE-sampled vertex.
  - The balance-heuristic MIS weights use NEE and BSDF path pdfs "and sum over all BSDF lobes".
  - The integral becomes `I = ∫_Ω̃ ω_{n(ℓ)}(x̄) f_ℓ(x̄) dx̃`.
- **[SOURCE] GRIS §7.5–7.6, p.14.**
  - Connectability examines "the roughness of the lobe ℓ_j chosen to sample vertex x_{j+1}".
  - "We treat NEE-sampled vertices as rough if at least one of their BRDF lobes is sufficiently rough. All light vertices are treated as rough."
  - "A reconnection copies the lobe index from the base path vertex."
- **[SOURCE] Enhanced Eq. 2 (p.4)** gives the PSS Jacobian. It is `p^y_{k−1}(ω'_{k−1})G(y_{k−1}→x_k)p^y_k(ω_k)` over the x-equivalent, "p^x_k(ω_k) is replaced with 1 for k = d". The supplement's S-Eq.4 writes these as joint `p(ω_k, ℓ_k | x_k, −ω_{k−1})` (enhanced-verify O2).
- **[SOURCE] Falcor** ([FAL] `BxDF.slang:943-1051`, `PathTracer.slang:298-345`):
  - `sample()` returns `pdfSingle = pLobe·pdf_lobe` (joint) and `pdf = pdfAll` (marginal), with `weight = f_lobe/(pLobe·pdf_lobe)`.
  - `generateScatterRay` uses `pdfSingle` for the throughput and the cached Jacobian, and `pdfAll` for the next hit's MIS.

### 4.2 Definitions [INFERENCE, formalised from the above]

At a vertex with incoming V and lobes `ℓ ∈ Λ(V)`, the allocated closures:

```
f(V,L)          = Σ_ℓ f_ℓ(V,L)                      (full BSDF; f_ℓ includes its closure weight)
p(L,ℓ|V)        = q(ℓ|V) · p_ℓ(L|V)                  (joint; q = discrete lobe pmf, Σ_ℓ q = 1)
p_all(L|V)      = Σ_{ℓ non-delta} q(ℓ|V) · p_ℓ(L|V)  (marginal density of the BSDF sampler; delta lobes carry probability mass, no density)
```

Per-vertex integrand factors in PSS (what the throughput multiplies by):

| Vertex type | Factor |
|---|---|
| BSDF-sampled, lobe ℓ (non-delta) | `f_ℓ(V,L)·cos / p(L,ℓ|V)` |
| BSDF-sampled, delta lobe | `F_ℓ(V·N_s) / q(ℓ|V)` (no density) |
| NEE-sampled last vertex | `f(V,L_nee)·cos · L_e · ω_NEE / p_light` (all lobes) |
| BSDF-sampled emitter hit (next vertex) | multiply by `L_e · ω_BSDF` with `ω_BSDF` from `p_all(L|V)` (1 after a delta lobe) |

### 4.3 Any strictly positive q is unbiased; q only changes noise [INFERENCE, proof]

**One-bounce estimator.** Take ℓ ~ q(·|V) and L ~ p_ℓ(·|V), and let `Y = f_ℓ(V,L)cos · g(L) / (q(ℓ|V) p_ℓ(L|V))` for any test function g, e.g. the incident radiance times the rest of the path. Then

```
E[Y] = Σ_ℓ q(ℓ|V) ∫ p_ℓ(L|V) · f_ℓ cos g /(q(ℓ|V) p_ℓ(L|V)) dL = Σ_ℓ ∫_{supp p_ℓ} f_ℓ cos g dL = ∫ f cos g dL
```

This holds provided two conditions are met:
- (i) `q(ℓ|V) > 0` whenever `f_ℓ(V,·) ≢ 0`;
- (ii) `supp f_ℓ ⊆ supp p_ℓ`, i.e. every lobe's sampler covers its own lobe.

Otherwise the expectation does not depend on q.

**MIS.** Both techniques use the same `p_all = Σ q p_ℓ`, so `ω_NEE + ω_BSDF = 1` for every q. That covers balance, power, and Enhanced's RIS-NEE `M p1/(M p1 + p2)`. So q changes the MIS weights but not the expectation.

**GRIS / ReSTIR.**
- The target `p̂ = lum(F)` in PSS contains q through `F = f/p`. The target is a free choice in GRIS (gris-math §3), so this is unbiased.
- The PSS Jacobian contains the ratio `q(ℓ|V^y)/q(ℓ|V^x)` explicitly (§7.2), so it is exact.

**The two places where q *does* change the expectation:**
- (a) A lobe with `f_ℓ ≠ 0` but `q = 0`. Guard: if a lobe is allocated but its sample weight is 0, give it `q = 1e-6` and renormalise. In Cycles this only happens on measure-zero sets, e.g. black F0 at μ = 1 exactly ([CY] `bsdf_microfacet.h:437-440`).
- (b) Directions where the MIS is not a partition of unity, i.e. smooth shading, §8.2. There the weight value, and so q, the heuristic and the light pdf, enters the expectation. **This is the only reason to prefer flat shading for exact parity.**

### 4.4 Two self-consistent conventions; never mix them

| Convention | Throughput at a BSDF vertex | Jacobian pdfs | Lobe-specific connectability? | Used by |
|---|---|---|---|---|
| **A. Lobe-extended (NORMATIVE here)** | `f_ℓ cos/(q_ℓ p_ℓ)` | joint `q_ℓ p_ℓ`, lobe copied at reconnection | yes | GRIS §7.6, Falcor (`separatePathBSDF=true`) |
| B. Vertex / marginal (Cycles-style one-sample MIS) | `f cos / p_all` | marginal `p_all` | no; only "any lobe rough" | Cycles itself; EvanLuo42 in effect (below) |

- **Mixing A's throughput with B's Jacobian, or the reverse, is biased.** The PSS density of a reconnected direction is not the density that generated the throughput (enhanced-verify O2).
- **Why A.**
  - Plastic-like materials (any V2 dielectric) are two-lobe.
  - With B, a glossy-sampled bounce on a plastic cannot be distinguished from a diffuse-sampled one, so a rough-enough plastic reconnects even when its specular lobe produced the direction.
  - GRIS Fig. 12 reports that separating lobes increases hybrid-shift efficiency significantly [SOURCE GRIS p.14].
- **What EvanLuo42 does.**
  - [SOURCE EVL] `hasExactSingleLobe` (`EnhancedPolicy.slang:184-190`) requires `concreteLobeCount(lobeTypes) == 1`.
  - `cacheInternalReconnection` (`PathSampling.slang:113-128`) refuses to reconnect unless **both** `y_{k−1}` and `x_k` carry `kWalkExactLobePdf`.
  - Replay fails if a replayed lobe differs from the stored one (`HybridShift.slang:62-64`).
  - [INFERENCE] With Falcor-7 StandardBSDF this means **no reconnection ever happens at a two-lobe (plastic) vertex**. For glTF/Principled scenes most vertices are two-lobe, so the hybrid shift degenerates towards random replay there.
  - Do not adopt that restriction. Convention A with joint pdfs handles multi-lobe vertices exactly.

---

## 5. Variant V1: "validation" BSDF

### 5.1 Blender node recipe (the reference must be built exactly like this)

```python
# Blender 5.1.2. Defaults verified by [BPY]: Glossy distribution default is MULTI_GGX, so it MUST be set to 'GGX'.
m  = bpy.data.materials.new(name); nt = m.node_tree; nt.nodes.clear()
out = nt.nodes.new('ShaderNodeOutputMaterial')
d  = nt.nodes.new('ShaderNodeBsdfDiffuse');  d.inputs['Color'].default_value = (*rho, 1); d.inputs['Roughness'].default_value = 0.0
g  = nt.nodes.new('ShaderNodeBsdfGlossy');   g.distribution = 'GGX'
g.inputs['Color'].default_value = (*k, 1);   g.inputs['Roughness'].default_value = r;      g.inputs['Anisotropy'].default_value = 0.0
mx = nt.nodes.new('ShaderNodeMixShader');    mx.inputs['Fac'].default_value = f            # Fac=0 -> all diffuse
nt.links.new(d.outputs[0], mx.inputs[1]); nt.links.new(g.outputs[0], mx.inputs[2]); s = mx.outputs[0]
if Es > 0:                                                                                  # optional emission
    e = nt.nodes.new('ShaderNodeEmission'); e.inputs['Color'].default_value = (*E, 1); e.inputs['Strength'].default_value = Es
    a = nt.nodes.new('ShaderNodeAddShader'); nt.links.new(s, a.inputs[0]); nt.links.new(e.outputs[0], a.inputs[1]); s = a.outputs[0]
nt.links.new(s, out.inputs['Surface'])
m.cycles.use_bump_map_correction = False; m.cycles.emission_sampling = 'FRONT_BACK'
```

### 5.2 Closures and lobe set

[SOURCE] The Mix Shader saturates Fac and writes `in·(1−f)` / `in·f` ([CY] `svm/closure.h:1543-1566`). Diffuse and Glossy use `weight = closure_weight(=Color)·mix_weight` (`:531, :686`).

| Lobe | Closure weight (RGB) | Allocated iff |
|---|---|---|
| **D** (Lambert) | `w_D = (1−f)·ρ` | `|avg(w_D)| ≥ 1e-5` ([CY] `closure/alloc.h:64-72`) |
| **S** (GGX, F ≡ 1) | `w_G = f·k`, `α = saturate(r)²` | `|avg(w_G)| ≥ 1e-5` |

Emission `L_e = Strength·Color` is not a lobe. It is two-sided and has no π or cosine ([CY] `surface_shader.h:1102-1108`).

### 5.3 eval·cos and pdfs

```
f_D·cos(V,L) = w_D · max(N·L,0)/π                           p_D(L)   = max(N·L,0)/π
f_S·cos(V,L) = w_G · D(Ns·H)·G2/(4 Ns·V)   (valid per §3.2)   p_S(L|V) = D(Ns·H)/(4 Ns·V (1+Λ(Ns·V)))
```

### 5.4 Selection probabilities (Cycles-exact)

- [SOURCE] `sample_weight = |avg(weight)|` ([CY] `alloc.h:64`). The GGX Glossy closure calls no `setup_fresnel_*`, so it gets **no albedo scaling** ([CY] `closure.h:726-741`, `bsdf_microfacet.h:961-971`).
- [SOURCE] The closure is picked ∝ `sample_weight` ([CY] `surface_shader.h:408-448`).

```
q(D) = |avg(w_D)| / (|avg(w_D)| + |avg(w_G)|),     q(S) = 1 − q(D)        (independent of V)
```

### 5.5 Marginal and joint pdfs

```
p(L,D|V) = q(D)·max(N·L,0)/π
p(L,S|V) = q(S)·p_S(L|V)
p_all(L|V) = p(L,D|V) + p(L,S|V)       (depends on V through p_S)
```

Cycles' NEE/forward MIS pdf is exactly this p_all: `Σ_i sw_i pdf_i / Σ_i sw_i` over closures with `pdf_i ≠ 0` ([CY] `surface_shader.h:270-306`).

### 5.6 Delta (singular) GGX

- **NORMATIVE:** `isDelta = !(α_x·α_y > 2e-10)` ([CY] `bsdf_microfacet.h:581-584`, `svm/types.h:545`). For isotropic α = r², this is r ≤ (2e-10)^{1/4} = 0.0037606.
- **Sampling:**
  - `H = Ns`, `L = 2(Ns·V)Ns − V`.
  - Reject if `Ns·V ≤ 0`, `Ng·L < 0` or `Ns·L < 0` (`:693, :763`).
  - Throughput factor: `w_G/q(S)` in V1, `F_S(Ns·V)/q(S)` in V2.
- **Cycles equivalence.** Cycles returns `eval = F·1e6` and `pdf = 1e6` for the picked singular closure (`:778-782`). The mixture throughput is then `F·w_i·Σsw/sw_i` up to an O(1e-6) contamination from the other closures' evals, which we drop [INFERENCE: negligible].
- **NEE:** the delta lobe contributes 0 to eval and to p_all. Its q still counts in the normalisation, because Cycles keeps its `sample_weight` in `sum_sample_weight` (`surface_shader.h:301`).
- **Emitter hit after a delta bounce:** the MIS weight is 1, because the singular label sets `PATH_RAY_MIS_SKIP` ([CY] `integrator/path_state.h:215-216`; `light/sample.h:515`). The bounce counts as a **glossy** bounce.

### 5.7 Stored lobe, roughness, footnote 6 for V1

- **Stored code:** D or S (2 bits) plus a delta bit, per §7.3.
- **Roughness:** r_D = 1; r_S = r (the Glossy Roughness socket, perceptual); r_S = 0 if delta.
- **Diffuse-only** iff S is not allocated (`f == 0` or `avg(f·k) < 1e-5`).
- [INFERENCE] V1 has V-independent q, so the *joint* pdf of lobe D, `q(D)·cos/π`, is independent of ω_in. Under a joint-pdf footprint variant, "ℓ_k = D" would therefore be an exact skip condition for V1. It is not exact under the paper's marginal convention, which we adopt. See §7.5.
- V1 **is reciprocal**: `f_r = (1−f)ρ/π + f k D G2/(4 NV NL)` is symmetric. V2 is not.

---

## 6. Variant V2: "Cycles Principled Tier 1" (plus Tier 2)

### 6.1 Validation input subset and Blender setup

- **Validation input subset:**
  - `distribution='GGX'` (Tier 1) or `'MULTI_GGX'` (Tier 2);
  - Coat, Sheen, Transmission, Subsurface, Thin Film Thickness, Anisotropic and Diffuse Roughness all 0;
  - Alpha 1;
  - any Base Color, Metallic, Roughness, IOR, Specular IOR Level and Specular Tint;
  - Emission Color × Strength.
- **Blender defaults** [BPY, VERIFIED]: Base 0.8, Metallic 0, Roughness 0.5, IOR 1.5, Specular IOR Level 0.5, Specular Tint white, distribution MULTI_GGX, enum `['GGX','MULTI_GGX']`.
- **Cycles-internal node default** is the same, with `distribution` defaulting to `CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID` ([CYH] `scene/shader_nodes.cpp:2725-2729`).
- **Distribution enum mapping:** `"ggx" → CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID`, `"multi_ggx" → CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID`. The kernel tests `is_multiggx = (distribution == …MULTI_GGX_GLASS_ID)` ([CY] `closure.h:368-370, 454`).

### 6.2 Closure construction (exact order, [CY] `svm/closure.h:169-509`)

Inputs after clamping:
- `C = max(Base,0)` (unclamped above);
- `Cc = min(C,1)`;
- `r = saturate(Roughness)`, `α = r²`;
- `m = saturate(Metallic)`;
- `η = max(IOR, 1e-5)`;
- `L = max(Specular IOR Level, 0)`;
- `T = max(Specular Tint, 0)`.

`weight` starts at `mix_weight` = 1, then `weight *= alpha`, which is 1 here.

1. **Metal** (if `m > 1e-5`, `:344-375`).
   - Closure weight `w_M = m·weight` (grey).
   - `Ns = valid_reflection_N`.
   - F82-tint with `F0_M = Cc` and `f82 = min(T,1)`, so `B_M = B(Cc, f82)`.
   - Then `weight *= (1−m)`. This happens even if the closure was not allocated.
2. Transmission is skipped (0).
3. **IOR level** (`:417-426`).
   - `f0 = F0(η)`.
   - If `L ≠ 0.5`: `f0 *= 2L`, `η' = ior_from_F0(f0)`, and `η' = 1/η'` if `η < 1`.
   - Otherwise `η' = η`.
4. **Dielectric specular** (if `η' ≠ 1`, `:428-462`).
   - Closure weight `w_S = weight`.
   - `Ns = valid_reflection_N`.
   - Generalized Schlick with `f0_S = saturate(f0·T)` (the saturate is at setup, `bsdf_microfacet.h:869`), `f90 = 1`, `exponent = −η'`, `ior = η'`.
   - Then **layering**:
     ```
     E_S(μ)  = mix(f0_S, 1, S_ior(r, μ, z_S))   per channel,  z_S = sqrt(|(η'−1)/(η'+1)|),  μ = dot(V, Ns)   [bsdf_microfacet.h:441-463]
     weight ← weight · saturate(1 − max_c E_S,c(μ))                                                      [bsdf_util.h:488-492]
     ```
     The general form is `albedo/weight = E_S` per channel (Tier 1), or `darkening⊙E_S` (Tier 2, §6.10). `safe_divide_color` gives 0 where `weight_c = 0`.
5. **Diffuse** (`:494-509`).
   - Closure weight `w_D = C·(1−sss)·weight = C·(1−m)·Λ(μ)` with `Λ(μ) = saturate(1 − max_c E_S,c(μ))`.
   - It uses the **unclamped** C and normal **N** (not Ns).
   - It is Lambert because Diffuse Roughness < 1e-5.
6. **Emission** `L_e = EmissionColor·EmissionStrength·weight_at_emission`, where the weight is α = 1 here (`:337-339`). It is two-sided.

Each closure is allocated only if `|avg(weight_i)| ≥ 1e-5` (`alloc.h:64-72`). NORMATIVE: replicate the cutoff. It is a V-dependent part of the BSDF definition for w_D.

### 6.3 Lobe set, eval·cos and lobe-conditional pdfs

| Lobe code | Members (Cycles closures) | eval·cos | p_ℓ(L|V) |
|---|---|---|---|
| **D** | diffuse | `w_D(V) · max(N·L,0)/π` | `max(N·L,0)/π` |
| **S** | metal ⊕ dielectric specular | `[w_M·F82(V·H; Cc, B_M) + w_S·F_gs(V·H; η', f0_S)] · D(Ns·H)·G2/(4 Ns·V)` | `p_VNDF(L|V)` (§3.2) |

- `F_gs` uses the real Fresnel of **η'** remapped to `[f0_S, 1]` (`bsdf_microfacet.h:313-320`).
- For `1 < η' ≤ 3.73` with `L = 0.5` and `T = 1` it equals `F_diel(V·H, η)` exactly (cycles-verify C4).

### 6.4 Why merging metal + specular into one class S is exact [INFERENCE, proof]

- Both closures have `N = valid_reflection_N`, the same `α_x, α_y, T`, and no transmission (`closure.h:354-358, 438-442`). So both sample H from the identical VNDF, and both have `pdf_reflect = 1`.
- Cycles picks the metal closure with prob `sw_M/Σ` and the specular closure with prob `sw_S/Σ`. Conditioned on "some GGX closure picked", L has density `p_VNDF` either way.
- Our class sampler picks S with `q(S) = (sw_M + sw_S)/Σ` and samples `p_VNDF` once. That is the **same distribution** of L.
- Evaluating the class integrand `f_S = f_M + f_S'` then gives the lobe-extended estimator of §4.3 with class pdf `q(S)·p_VNDF`.
- The marginal p_all is identical to Cycles' `Σ sw_i pdf_i / Σ sw_i`.
- Delta case: Cycles' two singular closures never see each other's eval (both are 0 off-sample). Its expectation is `(sw_M/Σ)·F82 w_M Σ/sw_M + (sw_S/Σ)·F_gs w_S Σ/sw_S = F82 w_M + F_gs w_S`, which is the same as our `q(S)·(F82 w_M + F_gs w_S)/q(S)`.
- The same argument is how Falcor gets one specular lobe for metal + dielectric: `pSpecularReflection = specularWeight·(metallicBRDF + dielectricBSDF)` ([FAL] `BxDF.slang:863-873`).

### 6.5 Lobe weights as functions of V (Tier 1)

```
w_M        = m                                  (grey; V-independent)
w_S        = (1−m)                              (grey; V-independent)
w_D(V)     = C · (1−m) · Λ(μ_V),     Λ(μ) = saturate(1 − max_c mix(f0_S,c, 1, S_ior(r, μ, z_S)))
μ_V        = dot(V, Ns)   (saturated to [0,1] inside the LUT reader)
```

The *effective* specular reflectance also depends on V through F(V·H), but the closure weights w_M and w_S do not.

### 6.6 Selection probabilities: Cycles-exact albedo-scaled sample weights

[SOURCE] `sample_weight = |avg(weight)|` at allocation (`alloc.h:64`). Then every Fresnel setup multiplies by `avg(bsdf_microfacet_estimate_albedo(...))` (`bsdf_microfacet.h:833, 849, 872, 917, 942`), and MULTI_GGX also multiplies by `avg(darkening)` (`:408`) (cycles-verify omission 1).

```
sw_M(V) = m · avg( mix(Cc, 1, S5(r, μ_V)) )                 S5 = lut3(r, μ, 0.5, table_ggx_gen_schlick_s)   [:465-479; B ignored: "TODO"]
sw_S(V) = (1−m) · avg( E_S(μ_V) )                           (same E_S as layering)                           [:441-463]
sw_D(V) = |avg(C)| · (1−m) · Λ(μ_V)
q(D|V)  = sw_D/(sw_D+sw_M+sw_S),     q(S|V) = (sw_M+sw_S)/(sw_D+sw_M+sw_S)
```

Notes:
- `S5` uses exponent 5 ⇔ z = `1/(0.2·5+1)` = 0.5 ([CY] `:458`; [CYH] precompute maps `exponent = 5(1−z)/z`).
- `table_ggx_gen_schlick_s` is needed **only** for this noise-only sample weight. An implementation may drop it and use any positive q (§4.3), at the cost of MIS weights that differ from Cycles.
- **Recommendation:** port it (4096 floats, 16 KB).
  - Identical MIS weights make per-technique debugging and Cycles light-pass comparisons (§12, T-C3) meaningful.
  - It keeps noise levels comparable when planning spp.
- Falcor makes the analogous V-dependent choice with `luminance(Schlick(F0, 1, V·N))` ([FAL] `BxDF.slang:867-873`).

### 6.7 Marginal and joint pdfs (V2)

```
p(L,D|V)   = q(D|V)·max(N·L,0)/π
p(L,S|V)   = q(S|V)·p_VNDF(L|V)
p_all(L|V) = p(L,D|V) + p(L,S|V)        (the S term only if non-delta and valid; §3.2)
```

Both q and p_VNDF depend on V. That matters for §7.5.

### 6.8 Delta, stored lobe, roughness

- **Delta:** as §5.6, with throughput `(w_M·F82(Ns·V) + w_S·F_gs(Ns·V))/q(S)`. In Tier 2, multiply each term by its compensation multiplier. Cycles applies `energy_scale` to singular samples too (`bsdf_microfacet.h:1029`).
- **Stored lobe:** D or S (2 bits) plus a delta bit (§7.3).
- **Roughness:** r_D = 1; r_S = r (Principled Roughness, perceptual; both GGX closures share it); 0 if delta.

### 6.9 Non-reciprocity and what it implies [SOURCE + INFERENCE]

- `Λ(μ_V)`, `q(·|V)` and the Tier-2 factors depend on `μ_V = Ns·V` only, so `f(V,L) ≠ f(L,V)` (cycles-conventions §4.4 inference, confirmed by code).
- **NORMATIVE:** always evaluate with V = toward the camera-side predecessor.
  - At a reconnection vertex x_k of an offset path, `V = normalize(y_{k−1} − x_k)`.
  - For NEE, V is toward the predecessor and L toward the light.
- validation-harness **T8(e) "reciprocity f(ωi,ωo)=f(ωo,ωi)" must be removed for V2.** Replace it with an *asymmetry* test (§12, U-9).
- Enhanced footnote 5 assumes BSDF **pdf** reciprocity, which is approximate for VNDF. That is only a heuristic justification for the inverse footprint. Nothing in our estimator relies on reciprocity.

### 6.10 Tier 2 (MULTI_GGX): formulas, magnitude, and whether it is needed

**Formulas** ([CY] `bsdf_microfacet.h:359-410`), per GGX closure with its own Fss:

```
E     = lut2(r, μ, table_ggx_E, 32, 32);   Eavg = lut1(r, table_ggx_Eavg, 32);   missing = (1−E)/E
Fss_M = mix(Cc, 1, 1/21) − B_M/126                    (metal)
Fss_S = mix(f0_S, 1, saturate((Fss_diel(η') − F0(η'))/(1 − F0(η'))))   (dielectric spec)   [:874-898]
Fms   = Fss·Eavg/(1 − Fss(1−Eavg))
eval multiplier       mult = 1 + Fms·missing           (= energy_scale·darkening)
darkening             dark = (1 + Fms·missing)/(1 + missing) = E + Fms(1−E);   dark ≡ 1 if Fss ≡ 1 (skipped by isequal)
layering albedo ratio → dark ⊙ E_S(μ)          sample weight → × avg(dark)
```

**Magnitude**, computed in this session from the v5.1.2 tables [VERIFIED numerically]:

| Case | r | μ | E | Eavg | Tier-2 eval multiplier | Effect |
|---|---|---|---|---|---|---|
| white metal (Fss = 1) | 0.5 | 1.0 | 0.91528 | 0.88204 | 1.0926 | +9.3 % |
| white metal | 1.0 | 1.0 | 0.30685 | 0.40914 | **3.2589** | ×3.26 (single-scatter albedo 0.307 → ≈1.0) |
| metal F0 = 0.9 (R channel) | 1.0 | 1.0 | 0.30685 | 0.40914 | 2.7966 | albedo_R 0.276 → 0.772 |
| metal F0 = 0.3 (B channel) | 1.0 | 1.0 | 0.30685 | 0.40914 | 1.3836 | |
| dielectric η = 1.5, C = 0.8 | 0.5 | 1.0 | | | spec ×1.0074 | diffuse Λ 0.9595 → 0.9626 (×1.0033) |
| dielectric | 0.5 | 0.1 | | | spec ×1.0097 | diffuse ×1.0255 |
| dielectric | 1.0 | 1.0 | | | spec ×1.0873 | diffuse Λ 0.9587 → 0.9862 (×1.0287) |

**Answer: yes, Tier 2 is needed** to match a GLB imported with Blender's defaults.

- [VERIFIED] `io_scene_gltf2/blender/imp` never assigns `distribution` (grep finds no occurrence). Every imported Principled node therefore stays MULTI_GGX.
- The Tier-1/Tier-2 gap (up to 3.3× on rough metals, ~3 % on dielectric diffuse) is far above the harness tolerances of 0.5 % global and 2 % per tile.
- It is aggravated by the glTF default `metallicFactor = roughnessFactor = 1` (the importer's `metallic_roughness()` uses 1.0 when absent, [GLTF] `pbrMetallicRoughness.py:734-747`). Any material that specifies base colour but not MR is a **roughness-1 metal**, the worst case in the table.

Recommendation:
- (i) The **validation harness forces `distribution='GGX'`** on every Principled node (cycles-conventions §4.8 post-processing), so Tier 1 is enough for stage-A correctness.
- (ii) Implement Tier 2 behind a per-material flag. It is 1056 more floats and ~40 WGSL lines.
- (iii) Default **imported** glTF/USD materials to Tier 2, because the scene-io unit contract is "our render = Cycles(BlenderImporter(file))".
- (iv) Validate Tier 2 separately against MULTI_GGX references. It is exactly matchable too (§2.1).

---

## 7. ReSTIR integration rules (both variants)

### 7.1 Which pdf goes where (NORMATIVE)

| Consumer | Quantity | Source / rationale |
|---|---|---|
| Path-tracer throughput at a BSDF-sampled vertex (initial candidates, random replay) | `f_ℓ cos / p(L,ℓ)` (joint) | Convention A, §4.4; Falcor `result.pdf = pdfSingle` |
| NEE contribution at the last vertex | `f(V,L)cos`, all lobes | GRIS §7.6 |
| NEE and BSDF-hit MIS weights | `p_all` (marginal) | GRIS Eq. 49 "sum over all BSDF lobes"; Falcor `path.pdf = pdfAll`; Cycles `_surface_shader_bsdf_eval_mis` |
| PSS reconnection Jacobian factors at x_{k−1}/y_{k−1} and x_k | `p(ω,ℓ)` joint, ℓ copied from the base | Enhanced Eq. 2 / S-Eq.4; Falcor `pdfSingle` in `cachedJacobian`; enhanced-verify O2 |
| Enhanced ray footprint `(p_{k−1}(ω_{k−1})·G(x_{k−1}→x_k))⁻¹` | `p_all(ω_{k−1}|x_{k−1}, V_{k−1})` | Enhanced S-§3: marginal used "in practice" (enhanced-paper §3.5) |
| Enhanced inverse footprint `(p_k(ω_k)·G(x_k→x_{k−1}))⁻¹` | `p_all(ω_k|x_k, −ω_{k−1})`, or skipped per §7.5 | same |
| Enhanced single-vertex roughness at x_{k−1} | `r(ℓ_{k−1})` per §7.4 | Enhanced §4.2 |
| Target `p̂` | `lum(F)` with the lobe-extended F | gris-math §5.1 |

[INFERENCE] Enhanced S-Fig.1 reports marginal FLIP 0.171 vs conditional 0.169, so the footprint pdf choice is a small quality knob. Keep the paper's marginal, because c = 0.02 was tuned with it. Whatever is chosen must be **bit-identical** in base-path rc selection and in the offset-path invertibility check.

### 7.2 Jacobian with lobes (hybrid shift, reconnection at x_k)

```
|∂T/∂ū| = [ q(ℓ_{k−1}|V^y_{k−1}) p_{ℓ_{k−1}}(ω'_{k−1}|V^y_{k−1}) · G(y_{k−1}→x_k) · q(ℓ_k|V^y_k) p_{ℓ_k}(ω_k|V^y_k) ]
        / [ q(ℓ_{k−1}|V^x_{k−1}) p_{ℓ_{k−1}}(ω_{k−1}|V^x_{k−1})  · G(x_{k−1}→x_k) · q(ℓ_k|V^x_k) p_{ℓ_k}(ω_k|V^x_k) ]
G(a→b) = |n_g(b)·(a−b)/‖a−b‖| / ‖a−b‖²           (geometric normal at the RECEIVING vertex; enhanced-verify C8/O1)
V^y_k  = normalize(y_{k−1} − x_k),  V^x_k = normalize(x_{k−1} − x_k),  ω'_{k−1} = normalize(x_k − y_{k−1})
```

- The last factor pair becomes 1 when ω_k is NEE-sampled (ℓ_k = NEE) or x_k is a light vertex (ℓ_k = NONE) [SOURCE ENH Eq. 2: "replaced with 1 for k = d"].
- It is **not** conditional on MIS being on (gris-verify C3).
- **Shift failure** (f = 0) rules that come from the BSDF:
  - ℓ_{k−1} or ℓ_k is delta;
  - `q(ℓ|V^y) = 0` (lobe not allocated for the new V);
  - `p_ℓ = 0`;
  - `Ns·V^y ≤ 0` for an S lobe;
  - the sampler-support indicator of §8.3 fails for a BSDF-sampled segment.
- **Cached denominator.** For Falcor-style caching, store `p(ω_{k−1}, ℓ_{k−1})`, `p(ω_k, ℓ_k)` and `G(x_{k−1}→x_k)` for the base. When a shifted sample is selected, overwrite them with the offset values (enhanced-verify C4).

### 7.3 Per-vertex lobe storage (NORMATIVE)

| Field (bits) | Meaning | Mirrors Falcor |
|---|---|---|
| `lobeBefore` (2) | lobe that generated `ω_{k−1}` at x_{k−1}: D, S, or NEE (forced NEE-light reconnection, where x_k is the light vertex) | `pathFlags` bit 26 (specular before rc) |
| `lobeAfter` (2) | lobe that generated ω_k at x_k: D, S, NEE (x_k is the last scattering vertex and its outgoing direction was NEE-sampled), NONE (x_k is a light/env vertex) | bit 27 |
| `deltaBefore`, `deltaAfter` (1 + 1) | delta flags, kept for replay-termination bookkeeping; a delta lobe is never a reconnection lobe | bits 8/9 |

Lobe codes: `D = 0`, `S = 1`, `NEE = 2`, `NONE = 3`. This leaves room: transmission and coat will need 3 bits later, so reserve a third bit per field if the packing allows [INFERENCE].

- **Prefix vertices y_1..y_{k−2}:** nothing is stored. Random replay regenerates their lobes from the same random numbers with the **offset's own** q(·|V^y). The lobe may therefore differ from the base's. The PSS Jacobian is still 1, and the inverse shift reproduces the base exactly, so no bias [INFERENCE; this is Falcor's behaviour].
- **Optional EvanLuo-style rule** "replayed lobe must equal stored lobe" ([EVL] `HybridShift.slang:62-64`). It is symmetric, so it remains unbiased, but it needs a per-vertex lobe record (8 bits × 8 vertices in EvanLuo). **Not adopted**; keep it only as an experiment flag.
- **Suffix vertices after x_k:** baked into the cached `rcVertexIrradiance`, and never re-evaluated.
- **RNG layout:** each bounce consumes a fixed 3 dims (u_sel, u1, u2), including delta bounces. The lobe choice therefore never shifts the random stream (validation-harness T1/T2).

### 7.4 Per-lobe roughness for connectability (NORMATIVE)

```
r(D)    = 1
r(S)    = sqrt(sqrt(α_x α_y))  (= perceptual Roughness for isotropic; the Glossy/Principled "Roughness" socket, glTF roughnessFactor·G)
r(S)    = 0                    if delta
r(NEE)  = max( hasD ? 1 : 0,  hasS ? r(S) : 0 )     ("at least one lobe rough", GRIS §7.6)
r(NONE) = +∞ (light / env vertex is always rough; GRIS §7.6)
rough(ℓ) ⇔ r(ℓ) ≥ 0.2          (Enhanced "α_{x_{k−1}} ≥ α_min", α_min = 0.2, P-§7.2; Falcor uses '>' — pick '≥' and use it everywhere)
```

Justification that 0.2 is **perceptual**:
- Falcor's `specularRoughnessThreshold = 0.2` is compared against `linearRoughness`, and `alpha = linearRoughness²` ([FAL] `BxDF.slang:840, 913-921`; gris-verify C9).
- Enhanced reuses "ReSTIR PT's default α_min = 0.2".

Pitfalls found in the reference implementations:
- **EvanLuo42 compares GGX α, not perceptual roughness** [SOURCE EVL `SurfaceLoad.slang:104`: `roughnessAlpha = roughness * roughness`; `EnhancedPolicy.slang:202-210` `sampledLobeAlpha`; gate `roughness >= minReconnectionRoughness (0.2)`]. [INFERENCE] If Falcor-7 `properties.roughness` is perceptual [UNVERIFIED; check `StandardMaterialInstance::getProperties`], EvanLuo's gate is r ≥ 0.447, not r ≥ 0.2. Do not copy it.
- **Falcor's k = 1 primary vertex** uses `hasRoughComponent(sd, 1.0)`, i.e. "has a diffuse lobe" (gris-verify O5). NORMATIVE for us: use the same `r(ℓ) ≥ 0.2` rule at every vertex including x_1, so that base-path selection and offset invertibility use one function.
- **Delta thresholds differ.** Falcor's delta threshold is `α < kMinGGXAlpha = ROUGHNESS_THRESHOLD²` ([FAL] `BxDF.slang:51-55, 840-846`); the comment says historically 0.08, so α < 0.0064 [UNVERIFIED value]. Cycles uses `α_xα_y ≤ 2e-10`. **Use Cycles' threshold.** With Falcor's, surfaces with 0.0038 < r < 0.08 would become delta, lose NEE, and diverge from Cycles.

### 7.5 Enhanced footnote 6: the precise meaning of "diffuse-only x_k" (NORMATIVE)

- **[SOURCE ENH p.8 fn. 6]** "For diffuse or emissive x_k, reconnection does not affect p^x_k(ω_k) and the inverse ray footprint test is skipped."
- **[INFERENCE, definition]** The skip is justified exactly iff the density entering the inverse-footprint test, `p_all(ω | x_k, ω_in)` in our convention, is independent of the incoming direction ω_in for all ω. For V1 and V2:

  ```
  skipInverse(x_k) ⇔ lobeAfter == NONE                         (light / emissive-only vertex: no ω_k)
                    ∨ ( hasD(x_k) ∧ ¬hasS(x_k) )               ("diffuse-only": the Lambert lobe is the only allocated lobe)
  ```
  - **V1 diffuse-only:** Mix Fac = 0, or `avg(f·k) < 1e-5`.
  - **V2 diffuse-only:** metallic ≤ 1e-5, transmission, coat and sheen ≤ 1e-5, **and no specular closure**. The latter means `η' == 1` (Specular IOR Level = 0, or IOR = 1 at level 0.5) with no thin film. In that case `w_D = C·(1−m)` is V-independent.
- **Why a sampled diffuse lobe on a mixed material does not qualify:**
  - Under the marginal pdf, `p_all = q(D)cos/π + q(S)p_VNDF(ω|ω_in)` depends on ω_in through the VNDF term. This holds even in V1.
  - V2 adds V-dependent `q(D|V)` (§6.6). So even the *joint* pdf of lobe D changes by `q(D|V^y)/q(D|V^x)`. At grazing angles this ratio is far from 1 (e.g. q(S) = 0.050 at μ = 1 vs 0.242 at μ = 0.1, for r = 0.5, C = 0.8; §13.4).
  - V2 is also non-reciprocal, so the premise of footnote 5 (pdf reciprocity) is doubly broken.
  - Cycles Principled is therefore exactly the case enhanced-verify O8 and critique Gap 3 flag.
- **Emissive:** a *light vertex* (no outgoing direction). A V1/V2 surface that is emissive **and** has a BSDF is treated by its BSDF lobes, not as "emissive x_k".
- **NEE-sampled ω_k** (`lobeAfter == NEE`, x_k is the last scattering vertex):
  - Eq. 2's factor (4b) is identically 1 (p_k := 1), so skipping is also defensible.
  - RTXDI instead evaluates the inverse footprint with the BSDF pdf of the NEE direction (enhanced-verify O4). That guards against large integrand changes at glossy x_k.
  - **Adopt RTXDI's choice:** use `p_all(ω_nee|x_k, −ω'_{k−1})`, unless x_k is diffuse-only. This choice belongs to the rc-predicate spec (critique Gap 4) and must match there.
- **Bias statement [INFERENCE]:**
  - Skipping, or not skipping, never biases the estimator, provided base-path rc selection and offset-path invertibility evaluate the identical predicate.
  - A wrong skip only makes the (4b) Jacobian factor, and the p̂ ratio, deviate from 1. That costs variance and robustness, not correctness.

---

## 8. Geometric vs shading normals

### 8.1 Rules (NORMATIVE)

1. **Jacobian and footprint cosines use Ng** at the receiving vertex: GRIS Eq. 52 "geometric surface normal at x_{i+1}"; Falcor `faceN`; enhanced-verify O1/C8. A shading-normal cosine in J is biased.
2. **Sampling** rejects below-Ng directions:
   - Lambert requires `Ng·L > 0`;
   - GGX requires `Ng·L ≥ 0` and `Ns·L ≥ 0` (reflection), and `Ns·V > 0`.
3. **Eval** tests only the lobe normal: Lambert `max(N·L,0)`, GGX `Ns·V > 0` and `Ns·L ≥ 0`. There is **no Ng test** ([CY] `bsdf_microfacet.h:756-760`).
4. **NEE same-triangle skip** as Cycles: if the light sample lies on the shading triangle itself, skip it when `ls.D` is in the Ng hemisphere ([CY] `shade_surface.h:345-351`).
5. **Flat shading first.** Set `N = Ng` in both renderers, using Blender `shade_flat()` or `import_shading='FLAT'`, and have our loader ignore vertex normals in validation mode. Then items 2 and 3 coincide and §8.2 disappears.

### 8.2 Smooth shading: Cycles' own MIS is not a partition of unity [SOURCE + INFERENCE]

- For directions with `N·L > 0` but `Ng·L ≤ 0`, three things hold:
  - NEE evaluates f > 0 and weights it with `p_nee²/(p_nee² + p_bsdf²)`, where `p_bsdf` comes from eval pdfs, so it is > 0;
  - BSDF sampling can never produce the direction;
  - the combined weight is therefore < 1 (cycles-verify omission 3).
- **Consequence.** In that region Cycles' *expectation* depends on the MIS heuristic (power, β = 2, [CY] `light/sample.h:322-356`), on q, and on **Cycles' light-selection pdf**, which comes from the light tree.
- A renderer with a different light sampler, such as Enhanced's RIS light tiles, cannot match it exactly.
- The discrepancy is confined to below-Ng-but-above-N directions. These are usually self-occluded on closed meshes and matter on open, smooth-shaded surfaces.
- **Policy:**
  - flat shading for exact parity;
  - smooth-shaded scenes get a "model-approximate" tolerance tier;
  - use the power heuristic and Cycles-matched q to minimise the gap.

### 8.3 Keeping GRIS consistent with the PT under smooth shading [INFERENCE, NORMATIVE]

- **The problem.** The canonical sampler of a pixel domain cannot produce BSDF-sampled segments with `Ng·L ≤ 0`, so the PT integrand is `f̃ = ω·f_ℓ·1[segment ∈ supp(sampler)]`. If a shift produced such a segment with nonzero `f_ℓ` from eval, p̂ would be positive outside the canonical support. That violates Def. 5.2, and pairwise/Talbot MIS would then add energy the PT never integrates.
- **Rule.** When re-evaluating a **BSDF-sampled** segment in a shift, multiply by the sampler-support indicator:
  - `ω'_{k−1}` at y_{k−1}: D needs `Ng·ω > 0`; S needs `Ns·V > 0 ∧ Ng·ω ≥ 0 ∧ Ns·ω ≥ 0`;
  - `ω_k` at x_k with the new V.
- **NEE segments do not get the indicator,** because Cycles' NEE has no Ng test.
- `bsdf_sample_support()` in §11 implements this. With flat shading it is redundant but harmless.

### 8.4 Normal maps and the bump helpers (later stage)

- **`bump_shadowing_term`** ([CY] `closure/bsdf.h:65-125`).
  - It applies only when the closure normal ≠ smooth normal `sd->N`, i.e. normal or bump maps.
  - Eval, all closures: returns 0 if `(Ns_smooth·L)(Ns_smooth·N)(N·L) < 0`.
  - Sampling: rejects only for diffuse (`is_eval || is_diffuse`). Glossy samples are not rejected, and the pdf is not zeroed.
  - With `SD_USE_BUMP_MAP_CORRECTION`, diffuse is additionally multiplied by `G1_GGX(α² = saturate(0.125·tan²θ_d), |Ns_smooth·L|)`.
  - **[INFERENCE] With flat geometry the reject test is redundant.** Here `Ns_smooth = Ng`. Case `Ng·L > 0`, `N·L < 0`: reflection closures already give 0. Case `Ng·L < 0`, `N·L > 0`: the sampler rejects (Ng test) and eval returns 0 via the bump term, so eval and sampling agree.
  - So flat + normal map + `use_bump_map_correction = False` needs **no** bump helper port.
- **`ensure_valid_specular_reflection`** ([CY] `bsdf_util.h:370-457`) bends Ns toward Ng whenever `N ≠ Ng` and bump correction is on. That includes plain smooth shading. **Set `material.cycles.use_bump_map_correction = False`** in the reference; our `Ns = N`. Port it only if default-settings parity is wanted.
- **Normal Map node math** (cycles-verify omission 4, [CY] `svm/tex_coord.h:324-420`):
  ```
  c = 2(rgb − 0.5);  c.xy *= s;  c.z = mix(1, c.z, saturate(s))
  N = normalize(c.x·t_ + c.y·sign·cross(n_, t_) + c.z·n_)
  ```
  - `t_` is the un-normalised MikkTSpace tangent, `n_` the un-normalised interpolated normal, and `s = normalTexture.scale`.
  - Flip N if backfacing.
  - glTF `TANGENT` is ignored by Blender (scene-io §3.2).
- **Shadow-terminator geometry offset:** set `Object.shadow_terminator_geometry_offset = 0` (default 0.1).

---

## 9. Texture sampling for parity (NORMATIVE in validation mode)

1. **Format.** sRGB-encoded colour textures (base colour, emissive, specular colour, sheen colour) use `rgba8unorm`, **not** `rgba8unorm-srgb`. Non-colour data (metallic-roughness, normal, occlusion, specular factor in A) also use `rgba8unorm`, with no decode.
2. **Sample with** `textureSampleLevel(t, s, uv[, layer], 0.0)`.
   - Level 0 always. Create validation textures with `mipLevelCount = 1`.
   - `textureSample` is illegal in compute anyway.
   - Sampler `magFilter = minFilter = 'linear'` for Blender "Linear", and `'nearest'` for "Closest".
   - Address modes from glTF: REPEAT → `repeat`, CLAMP_TO_EDGE → `clamp-to-edge`, MIRRORED_REPEAT → `mirror-repeat`. These are the Metal samplers Cycles uses ([CY] `device/metal/compat.h:388-395`).
3. **Decode after filtering, per RGB channel, with the exact EOTF** Cycles uses on Metal ([CY] `util/color.h:63-69`). Alpha is not decoded.
   ```
   srgb_to_linear(c) = c < 0.04045 ? (c < 0 ? 0 : c/12.92) : ((c + 0.055)/1.055)^2.4
   ```
4. **Magnitude of the error you avoid.** A texel pair (0, 255) sampled at the midpoint:
   - decode-after-filter gives `srgb_to_linear(0.5) = 0.2140`;
   - `-srgb` hardware decode gives 0.5.
   - That is a 2.3× difference at every magnified high-contrast edge, e.g. a checker test.
5. **Texel convention.** WebGPU and Metal both use normalised coordinates with texel centres at `(i+0.5)/W`.
   - [INFERENCE] Chrome/Dawn on this Mac drives the same Apple GPU sampler hardware that Cycles-Metal uses, so the bilinear weights should agree bit-for-bit [UNVERIFIED; test U-T1 below].
   - A manual `textureLoad` bilinear with f32 weights is the fallback. It differs only by the hardware weight-quantisation error.
6. **Atlas pitfall.** Hardware `repeat`/`mirror-repeat` wraps at the **layer or texture edge**. Bucketed `texture_2d_array`s are only safe when every layer has exactly the image's size. A padded atlas needs manual wrap plus manual bilinear.
7. **Upload bytes unmodified.**
   - Use `createImageBitmap(blob, {colorSpaceConversion: 'none', premultiplyAlpha: 'none'})`, then `copyExternalImageToTexture(..., premultipliedAlpha: false)` into `rgba8unorm`.
   - Alternatively decode PNG in WASM, which is what the harness's raw sidecar does (validation-harness §5.4).
   - Use **PNG** for validation textures. JPEG decoders (browser vs OIIO) may differ in chroma upsampling [INFERENCE].
   - Block-compressed (BC7/ASTC via KTX2) textures cannot match a PNG reference.
8. **Alpha association.** Cycles un-associates alpha after interpolation when the node flag asks for it, *before* the sRGB decode ([CY] `svm/image.h:28-38`). Whether byte images are premultiplied at load is host-side and [UNVERIFIED]. Validation textures must have A = 1.
9. **UVs.** glTF UV origin is top-left and images are uploaded top row first, so use the glTF uv directly. Blender flips both v and image rows, which cancel [INFERENCE]. For USD, `v' = 1 − v` (scene-io §5.2). `KHR_texture_transform` is applied in glTF uv space.

---

## 10. glTF metallic-roughness → V2 mapping (Blender-importer-equivalent)

### 10.1 Table ([GLTF] verified in this session unless tagged)

| Principled input | Value | Colour space | Importer source |
|---|---|---|---|
| Base Color C | `baseColorFactor.rgb × sRGB⁻¹(baseColorTexture.rgb) × COLOR_0` | tex sRGB | Mix MULTIPLY; COLOR_0 rule [UNVERIFIED] |
| Metallic m | `metallicFactor × MR.B` (factor default **1.0**) | linear | `pbrMetallicRoughness.py:734-747, ~801` (`outputs['Blue']`) |
| Roughness r | `roughnessFactor × MR.G` (factor default **1.0**) | linear | `:~802` (`outputs['Green']`) |
| IOR η | `KHR_materials_ior.ior`, else `GLTF_IOR = 1.5` | – | `:120-122`; `io/com/constants.py:155` |
| Specular IOR Level L | `0.5 × specularFactor × specularTexture.A` (factor default 1 ⇒ 0.5) | linear | `:318-333` |
| Specular Tint T | `specularColorFactor × sRGB⁻¹(specularColorTexture.rgb)` (default white) | tex sRGB | `:343-350` |
| Emission | `emissiveFactor × sRGB⁻¹(emissiveTexture)`, Strength = `emissiveStrength` (greyscale factor folded in) | sRGB | scene-io §4.3 |
| Normal | Normal Map node, strength = `normalTexture.scale` | linear | §8.4 |
| Alpha | OPAQUE → 1; MASK/BLEND → out of scope for V2 | | |
| Occlusion | ignored by Cycles | | |
| distribution | **not set, stays MULTI_GGX** | | grep: no occurrence in `blender/imp` |

### 10.2 Semantics that differ from the glTF reference BRDF

Our renderer follows Cycles, not the glTF spec [INFERENCE from the formulas in §6.2]:
- glTF `KHR_materials_specular`: `dielectric_f0 = min(F0(ior)·specularColor,1)·specular` and **F90 = specular**.
- Cycles: `f0 = F0(η)·2L`, `η' = ior_from_F0(f0)`, `f0_S = saturate(f0·T)`, and **F90 = 1**, with the curve shape from the real Fresnel of η'.
- Specular Tint also becomes the **F82 edge tint of metals** (`f82 = min(T,1)`, `closure.h:361`). A glTF `specularColorFactor` therefore tints metal edges in Blender, although the glTF spec says the extension does not affect metals.

Example: `specularFactor = 2` (L = 1) at IOR 1.5 gives f0 = 0.08 and η' = 1.7888. `specularFactor = 0.5` (L = 0.25) gives f0 = 0.02 and η' = 1.3294 [VERIFIED numerically].

### 10.3 Missing material

- [VERIFIED] A primitive **without** a material but **with** `COLOR_0` gets a `DefaultMaterial` Principled with Blender node defaults: base = vertex colour, m = 0, r = 0.5, IOR 1.5, MULTI_GGX ([GLTF] `material.py:19-35`, `mesh.py:440-469`).
- Without `COLOR_0`, it has no material, so Cycles' `default_surface` applies. That is a `PrincipledBsdfNode` with Cycles defaults (Base 0.8, m 0, r 0.5, IOR 1.5, MULTI_GGX) ([CYH] `scene/shader.cpp:748-762`; `shader_nodes.cpp:2725-2743`).
- **This is not the glTF spec default material** (metallic 1, roughness 1). Map "no material" to Principled(0.8, 0, 0.5, 1.5, L 0.5) under the Blender-equivalent contract [the no-COLOR_0 branch is UNVERIFIED end-to-end; confirm with a one-triangle GLB spike].

---

## 11. WGSL reference module (normative)

The module is written against WGSL as shipped in Chrome 154 / Tint. Conventions follow §0. One `bsdf_prepare` and one `bsdf_sample` must be used by **every** pipeline: initial sampling, replay, shift and NEE. That keeps lobe decisions identical under Metal fast-math (validation-harness §5.6, T2).

```wgsl
// ============================ bsdf.wgsl ============================
const PI     : f32 = 3.14159265358979323846;
const INV_PI : f32 = 0.31830988618379067154;
const ROUGH_SQ_THRESH : f32 = 2e-10;  // BSDF_ROUGHNESS_SQ_THRESH  [svm/types.h:545]
const WEIGHT_CUTOFF   : f32 = 1e-5;   // CLOSURE_WEIGHT_CUTOFF     [svm/types.h:543]
const ROUGHNESS_MIN   : f32 = 0.2;    // perceptual; Enhanced alpha_min / Falcor specularRoughnessThreshold

const LOBE_D    : u32 = 0u;  // Lambert reflection            (Falcor SampledBSDFFlags 0x1 in class 0x3)
const LOBE_S    : u32 = 1u;  // GGX reflection class          (Falcor 0x4 in class 0xC)
const LOBE_NEE  : u32 = 2u;  // outgoing direction NEE-sampled; all lobes evaluated (GRIS "N")
const LOBE_NONE : u32 = 3u;  // light / env vertex; no outgoing direction

const KIND_V1 : u32 = 1u;    // Diffuse + Glossy(GGX, F=1) + Mix + Emission
const KIND_V2 : u32 = 2u;    // Principled, distribution GGX (Tier 1) or MULTI_GGX (Tier 2 flag)

// LUTs copied verbatim from intern/cycles/scene/shader.tables @ v5.1.2 (float32)
const LUT_IOR_S : u32 = 0u;     // table_ggx_gen_schlick_ior_s[4096], x=rough, y=mu, z=sqrt|(eta-1)/(eta+1)|
const LUT_S     : u32 = 4096u;  // table_ggx_gen_schlick_s[4096],     z = 1/(0.2*exponent+1)  (metal: z=0.5)
const LUT_E     : u32 = 8192u;  // table_ggx_E[1024],   32x32   (Tier 2)
const LUT_EAVG  : u32 = 9216u;  // table_ggx_Eavg[32]           (Tier 2)
@group(0) @binding(0) var<storage, read> lut : array<f32>;   // binding is a placeholder

// ---------- exact port of kernel/util/lookup_table.h ----------
fn lut1(x_in: f32, off: u32, n: u32) -> f32 {
  let x = saturate(x_in) * f32(n - 1u);
  let i = min(u32(x), n - 1u);            // truncation, x >= 0
  let j = min(i + 1u, n - 1u);
  let t = x - f32(i);
  let d0 = lut[off + i];
  if (t == 0.0) { return d0; }
  return (1.0 - t) * d0 + t * lut[off + j];
}
fn lut2(x: f32, y_in: f32, off: u32, nx: u32, ny: u32) -> f32 {
  let y = saturate(y_in) * f32(ny - 1u);
  let i = min(u32(y), ny - 1u); let j = min(i + 1u, ny - 1u); let t = y - f32(i);
  let d0 = lut1(x, off + nx * i, nx);
  if (t == 0.0) { return d0; }
  return (1.0 - t) * d0 + t * lut1(x, off + nx * j, nx);
}
fn lut3(x: f32, y: f32, z_in: f32, off: u32, n: u32) -> f32 {
  let z = saturate(z_in) * f32(n - 1u);
  let i = min(u32(z), n - 1u); let j = min(i + 1u, n - 1u); let t = z - f32(i);
  let d0 = lut2(x, y, off + n * n * i, n, n);
  if (t == 0.0) { return d0; }
  return (1.0 - t) * d0 + t * lut2(x, y, off + n * n * j, n, n);
}

// ---------- small helpers ----------
fn avg3(v: vec3f) -> f32 { return (v.x + v.y + v.z) * (1.0 / 3.0); }
fn max3(v: vec3f) -> f32 { return max(v.x, max(v.y, v.z)); }
fn F0_from_ior(eta: f32) -> f32 { let t = (eta - 1.0) / (eta + 1.0); return t * t; }
fn ior_from_F0(f0: f32) -> f32 { let s = sqrt(clamp(f0, 0.0, 0.99)); return (1.0 + s) / (1.0 - s); }

fn sample_uniform_disk(u: vec2f) -> vec2f {          // concentric map  [kernel/sample/mapping.h]
  let a = 2.0 * u.x - 1.0; let b = 2.0 * u.y - 1.0;
  if (a == 0.0 && b == 0.0) { return vec2f(0.0); }
  var r: f32; var phi: f32;
  if (a * a > b * b) { r = a; phi = (PI / 4.0) * (b / a); }
  else               { r = b; phi = (PI / 2.0) - (PI / 4.0) * (a / b); }
  return vec2f(r * cos(phi), r * sin(phi));
}
fn make_orthonormals(N: vec3f) -> mat3x3f {           // [util/math_float3.h:718-741]; columns (T, B, N)
  var a: vec3f;
  if (N.x != N.y || N.x != N.z) { a = vec3f(N.z - N.y, N.x - N.z, N.y - N.x); }
  else                          { a = vec3f(N.z - N.y, N.x + N.z, -N.y - N.x); }
  a = normalize(a);
  return mat3x3f(a, cross(N, a), N);
}

// ---------- microfacet GGX ----------
fn ggx_D(a2: f32, cosNH: f32) -> f32 {
  let c2 = min(cosNH * cosNH, 1.0);
  let t = (1.0 - c2) + a2 * c2;
  return a2 / (PI * t * t);
}
fn ggx_lambda(a2: f32, cosN: f32) -> f32 {
  let c = max(cosN, 1e-7);   // [INFERENCE] guard: Cycles relies on inf -> G=0; fast-math inf is unreliable
  return 0.5 * (sqrt(1.0 + a2 * max(1.0 / (c * c) - 1.0, 0.0)) - 1.0);
}
fn ggx_sample_vndf_local(Vl: vec3f, ax: f32, ay: f32, u: vec2f) -> vec3f {   // [bsdf_microfacet.h:189-220]
  let Vs = normalize(vec3f(ax * Vl.x, ay * Vl.y, Vl.z));
  let lensq = Vs.x * Vs.x + Vs.y * Vs.y;
  var T1 = vec3f(1.0, 0.0, 0.0);
  var T2 = vec3f(0.0, 1.0, 0.0);
  if (lensq > 1e-7) { T1 = vec3f(-Vs.y, Vs.x, 0.0) * inverseSqrt(lensq); T2 = cross(Vs, T1); }
  var t = sample_uniform_disk(u);
  t.y = mix(sqrt(max(1.0 - t.x * t.x, 0.0)), t.y, 0.5 * (1.0 + Vs.z));
  let Hs = t.x * T1 + t.y * T2 + sqrt(max(1.0 - dot(t, t), 0.0)) * Vs;
  return normalize(vec3f(ax * Hs.x, ay * Hs.y, max(0.0, Hs.z)));
}

// ---------- Fresnel ----------
fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {  // [bsdf_util.h:47-99]
  let g = eta * eta - (1.0 - cos_i * cos_i);
  if (g <= 0.0) { return 1.0; }
  let ci = abs(cos_i);
  let ct = -sqrt(g) / eta;
  let rs = (ci + eta * ct) / (ci - eta * ct);
  let rp = (ct + eta * ci) / (eta * ci - ct);
  return 0.5 * (rs * rs + rp * rp);
}
fn fresnel_gen_schlick_ior(cosHI: f32, eta: f32, f0: vec3f) -> vec3f {   // exponent<0 mode [bsdf_microfacet.h:313-320]
  let F0r = F0_from_ior(eta);
  let s = saturate((fresnel_dielectric(cosHI, eta) - F0r) / (1.0 - F0r));
  return mix(f0, vec3f(1.0), s);
}
fn fresnel_f82(c: f32, F0: vec3f, B: vec3f) -> vec3f {                   // [bsdf_util.h:178-184]
  let s = saturate(1.0 - c); let s5 = (s * s) * (s * s) * s;
  return saturate(mix(F0, vec3f(1.0), s5) - B * c * s5 * s);
}
fn f82tint_B(F0: vec3f, tint: vec3f) -> vec3f {                          // [bsdf_util.h:146-158]
  let f = 6.0 / 7.0; let f5 = (f * f) * (f * f) * f;
  return mix(F0, vec3f(1.0), f5) * (7.0 / (f5 * f)) * (vec3f(1.0) - tint);
}
fn fresnel_dielectric_Fss(eta: f32) -> f32 {                             // [bsdf_util.h:131-137] (Tier 2)
  if (eta < 1.0) { return 0.997118 + eta * (0.1014 - eta * (0.965241 + eta * 0.130607)); }
  return (eta - 1.0) / (4.08567 + 1.00071 * eta);
}
struct Tier2 { mult: vec3f, dark: vec3f };
fn tier2(rough: f32, mu: f32, Fss: vec3f) -> Tier2 {                     // [bsdf_microfacet.h:359-410]
  let E = lut2(rough, mu, LUT_E, 32u, 32u);
  let Eavg = lut1(rough, LUT_EAVG, 32u);
  let missing = (1.0 - E) / E;
  var t: Tier2;
  if (all(Fss == vec3f(1.0))) { t.mult = vec3f(1.0 + missing); t.dark = vec3f(1.0); return t; }
  let Fms = Fss * Eavg / (vec3f(1.0) - Fss * (1.0 - Eavg));
  t.mult = vec3f(1.0) + Fms * missing;            // eval multiplier = energy_scale * darkening
  t.dark = t.mult / (1.0 + missing);              // darkening (layering + sample weight)
  return t;
}

// ---------- material input (after texture fetch; all linear) ----------
struct MatIn {
  kind: u32,
  v1_diffuse: vec3f, v1_glossy: vec3f, v1_mix: f32,                 // V1
  base: vec3f, metallic: f32, ior: f32, spec_level: f32, spec_tint: vec3f, multi_ggx: bool,   // V2
  roughness: f32,                                                   // perceptual (both)
  alpha: f32,                                                       // must be 1 in validation
};

// ---------- per-vertex prepared BSDF (function of material AND V) ----------
struct BsdfCtx {
  kind: u32,
  N: vec3f, Ns: vec3f, Ng: vec3f,   // all flipped to the V side
  a: f32, a2: f32, rough: f32, isDelta: bool,
  hasD: bool, hasS: bool,
  wD: vec3f,                                         // Lambert closure weight
  wG: vec3f,                                         // V1 glossy weight (F == 1)
  wM: vec3f, F0m: vec3f, Bm: vec3f, mulM: vec3f,     // V2 metal (F82-tint); mulM = Tier-2 multiplier (1 in Tier 1)
  wS: vec3f, f0s: vec3f, etaS: f32, mulS: vec3f,     // V2 dielectric specular (gen. Schlick, eta')
  pD: f32, pS: f32,                                  // q(D|V), q(S|V)
};

fn bsdf_prepare(m: MatIn, V: vec3f, Nin: vec3f, NgIn: vec3f) -> BsdfCtx {
  var c: BsdfCtx;
  c.kind = m.kind;
  let back = dot(NgIn, V) < 0.0;                    // two-sided flip [geom/shader_data.h]
  c.Ng = select(NgIn, -NgIn, back);
  c.N  = select(Nin,  -Nin,  back);
  c.Ns = c.N;                                       // use_bump_map_correction = False
  let r = saturate(m.roughness);
  c.a = r * r; c.a2 = c.a * c.a; c.rough = r;       // rough == sqrt(sqrt(a2))
  c.isDelta = !(c.a2 > ROUGH_SQ_THRESH);
  c.mulM = vec3f(1.0); c.mulS = vec3f(1.0);
  let mu = dot(V, c.Ns);
  var swD = 0.0; var swS = 0.0;

  if (m.kind == KIND_V1) {
    let f  = saturate(m.v1_mix);
    let wD = max((1.0 - f) * m.v1_diffuse, vec3f(0.0));
    let wG = max(f * m.v1_glossy, vec3f(0.0));
    c.hasD = abs(avg3(wD)) >= WEIGHT_CUTOFF;
    c.hasS = abs(avg3(wG)) >= WEIGHT_CUTOFF;
    if (c.hasD) { c.wD = wD; swD = abs(avg3(wD)); }
    if (c.hasS) { c.wG = wG; swS = abs(avg3(wG)); }  // GGX: no albedo scaling of sample_weight
  } else {
    var weight = vec3f(saturate(m.alpha));            // mix_weight(=1) * alpha
    let base = max(m.base, vec3f(0.0));
    let met  = saturate(m.metallic);
    let tint = max(m.spec_tint, vec3f(0.0));
    let ior  = max(m.ior, 1e-5);
    // metal  [svm/closure.h:344-375]
    if (met > WEIGHT_CUTOFF) {
      let wMv = met * weight;
      if (abs(avg3(wMv)) >= WEIGHT_CUTOFF) {
        c.hasS = true; c.wM = wMv;
        c.F0m = min(base, vec3f(1.0));
        let f82 = min(tint, vec3f(1.0));
        c.Bm = select(f82tint_B(c.F0m, f82), vec3f(0.0), all(f82 == vec3f(1.0)));
        let est = mix(c.F0m, vec3f(1.0), lut3(c.rough, mu, 0.5, LUT_S, 16u));  // F82 albedo estimate (B ignored)
        var sw = abs(avg3(wMv)) * avg3(est);
        if (m.multi_ggx) {
          let t2 = tier2(c.rough, mu, mix(c.F0m, vec3f(1.0), 1.0 / 21.0) - c.Bm * (1.0 / 126.0));
          c.mulM = t2.mult; sw *= avg3(t2.dark);
        }
        swS += sw;
      }
      weight *= (1.0 - met);
    }
    // IOR level  [closure.h:417-426]
    var eta = ior; var f0 = F0_from_ior(eta);
    if (max(m.spec_level, 0.0) != 0.5) {
      f0 *= 2.0 * max(m.spec_level, 0.0);
      eta = ior_from_F0(f0);
      if (ior < 1.0) { eta = 1.0 / eta; }
    }
    // dielectric specular + layering  [closure.h:428-462]
    if (eta != 1.0 && abs(avg3(weight)) >= WEIGHT_CUTOFF) {
      c.hasS = true; c.wS = weight; c.etaS = eta;
      c.f0s = saturate(f0 * tint);
      let z = sqrt(abs((eta - 1.0) / (eta + 1.0)));
      var ratio = mix(c.f0s, vec3f(1.0), lut3(c.rough, mu, z, LUT_IOR_S, 16u));   // albedo / weight
      var sw = abs(avg3(weight)) * avg3(ratio);
      if (m.multi_ggx) {
        let F0r = F0_from_ior(eta);
        let s = saturate((fresnel_dielectric_Fss(eta) - F0r) / (1.0 - F0r));
        let t2 = tier2(c.rough, mu, mix(c.f0s, vec3f(1.0), s));
        c.mulS = t2.mult; sw *= avg3(t2.dark); ratio *= t2.dark;
      }
      swS += sw;
      let rr = select(vec3f(0.0), ratio, weight != vec3f(0.0));   // safe_divide_color
      weight = weight * saturate(1.0 - max3(rr));                // closure_layering_weight [bsdf_util.h:488-492]
    }
    // diffuse (Lambert: Diffuse Roughness == 0)  [closure.h:494-509]
    let wD = base * weight;
    c.hasD = abs(avg3(wD)) >= WEIGHT_CUTOFF;
    if (c.hasD) { c.wD = wD; swD = abs(avg3(wD)); }
  }
  // lobe pmf; guard against allocated-but-zero-weight lobes (measure-zero in Cycles, bias for us)
  if (c.hasD) { swD = max(swD, 1e-12); }
  if (c.hasS) { swS = max(swS, 1e-12); }
  let sum = swD + swS;
  if (sum > 0.0) { c.pD = swD / sum; c.pS = swS / sum; }
  return c;
}

// ---------- evaluation ----------
struct LobeEval { f: vec3f, pdf: f32 };                 // f = f_l*cos (closure weights included); pdf = p_l(L|V)
fn eval_D(c: BsdfCtx, L: vec3f) -> LobeEval {
  var e: LobeEval;
  if (!c.hasD) { return e; }
  let k = max(dot(c.N, L), 0.0) * INV_PI;
  e.f = c.wD * k; e.pdf = k;
  return e;
}
fn F_S(c: BsdfCtx, cosHI: f32) -> vec3f {
  if (c.kind == KIND_V1) { return c.wG; }
  return c.wM * c.mulM * fresnel_f82(cosHI, c.F0m, c.Bm)
       + c.wS * c.mulS * fresnel_gen_schlick_ior(cosHI, c.etaS, c.f0s);
}
fn eval_S(c: BsdfCtx, V: vec3f, L: vec3f) -> LobeEval {
  var e: LobeEval;
  if (!c.hasS || c.isDelta) { return e; }
  let cosNI = dot(c.Ns, V); let cosNO = dot(c.Ns, L);
  if (cosNI <= 0.0 || cosNO <= 0.0) { return e; }   // [bsdf_microfacet.h:614-618]; cosNO==0 has f=0 in Cycles
  let Hu = V + L; let len = length(Hu);
  if (!(len > 0.0)) { return e; }
  let H = Hu / len;
  let common = ggx_D(c.a2, dot(c.Ns, H)) / cosNI * 0.25;
  let lI = ggx_lambda(c.a2, cosNI); let lO = ggx_lambda(c.a2, cosNO);
  e.pdf = common / (1.0 + lI);                        // VNDF pdf
  e.f = F_S(c, dot(H, V)) * (common / (1.0 + lI + lO));
  return e;
}
struct AllEval { f: vec3f, fD: vec3f, fS: vec3f, pAll: f32 };
fn bsdf_eval_all(c: BsdfCtx, V: vec3f, L: vec3f) -> AllEval {   // NEE: all lobes, marginal pdf, no Ng test
  let d = eval_D(c, L); let s = eval_S(c, V, L);
  var r: AllEval;
  r.fD = d.f; r.fS = s.f; r.f = d.f + s.f;
  r.pAll = c.pD * d.pdf + c.pS * s.pdf;
  return r;
}
struct JointEval { f: vec3f, pJoint: f32, pAll: f32 };
fn bsdf_eval_lobe(c: BsdfCtx, V: vec3f, L: vec3f, lobe: u32) -> JointEval {   // shifts: f_l and joint pdf
  let d = eval_D(c, L); let s = eval_S(c, V, L);
  var r: JointEval;
  r.pAll = c.pD * d.pdf + c.pS * s.pdf;
  if (lobe == LOBE_D) { r.f = d.f; r.pJoint = c.pD * d.pdf; }
  else if (lobe == LOBE_S) { r.f = s.f; r.pJoint = c.pS * s.pdf; }
  else { r.f = d.f + s.f; r.pJoint = 1.0; }          // LOBE_NEE: p_k := 1 in the Jacobian
  return r;
}
fn bsdf_sample_support(c: BsdfCtx, V: vec3f, L: vec3f, lobe: u32) -> bool {  // Section 8.3 indicator
  if (lobe == LOBE_D) { return dot(c.Ng, L) > 0.0; }
  if (lobe == LOBE_S) { return dot(c.Ns, V) > 0.0 && dot(c.Ng, L) >= 0.0 && dot(c.Ns, L) >= 0.0; }
  return true;                                       // NEE segments: no Ng test in Cycles
}

// ---------- sampling (u = (u_sel, u1, u2); always consumes 3 dims) ----------
struct BsdfSample {
  L: vec3f, lobe: u32, valid: bool, isDelta: bool,
  weight: vec3f,   // f_l*cos / p(L,l)  (delta: F/q(S))
  pJoint: f32,     // q(l|V) * p_l(L|V)             (0 for delta)
  pAll: f32,       // marginal, for MIS at the next hit and the footprint (0 for delta => MIS weight 1)
};
fn bsdf_sample(c: BsdfCtx, V: vec3f, u: vec3f) -> BsdfSample {
  var s: BsdfSample;
  if (!(c.pD + c.pS > 0.0)) { return s; }
  if (u.x < c.pD) {
    s.lobe = LOBE_D;
    let t = sample_uniform_disk(u.yz);
    let B = make_orthonormals(c.N);
    s.L = t.x * B[0] + t.y * B[1] + sqrt(max(1.0 - dot(t, t), 0.0)) * c.N;
    if (!(dot(c.Ng, s.L) > 0.0)) { return s; }       // [bsdf_diffuse.h:57-63]
  } else {
    s.lobe = LOBE_S;
    let cosNI = dot(c.Ns, V);
    if (cosNI <= 0.0) { return s; }                  // [bsdf_microfacet.h:693-696]
    var H = c.Ns;
    if (!c.isDelta) {
      let B = make_orthonormals(c.Ns);
      let Hl = ggx_sample_vndf_local(vec3f(dot(B[0], V), dot(B[1], V), cosNI), c.a, c.a, u.yz);
      H = Hl.x * B[0] + Hl.y * B[1] + Hl.z * c.Ns;
    }
    let cosHI = dot(H, V);
    s.L = 2.0 * cosHI * H - V;
    if (dot(c.Ng, s.L) < 0.0 || dot(c.Ns, s.L) < 0.0) { return s; }   // [bsdf_microfacet.h:761-765]
    if (c.isDelta) {
      s.isDelta = true; s.valid = true;
      s.weight = F_S(c, cosHI) / c.pS;               // Cycles: F*1e6 / (1e6 * sw/sum)
      return s;
    }
  }
  let d = eval_D(c, s.L); let g = eval_S(c, V, s.L);
  s.pAll = c.pD * d.pdf + c.pS * g.pdf;
  let fl = select(g.f, d.f, s.lobe == LOBE_D);
  let pl = select(c.pS * g.pdf, c.pD * d.pdf, s.lobe == LOBE_D);
  s.pJoint = pl;
  if (!(pl > 0.0)) { return s; }
  s.weight = fl / pl;
  s.valid = true;
  return s;
}

// ---------- reconnection helpers ----------
fn lobe_roughness(c: BsdfCtx, lobe: u32) -> f32 {
  let rS = select(c.rough, 0.0, c.isDelta);
  switch lobe {
    case LOBE_D:   { return 1.0; }
    case LOBE_S:   { return rS; }
    case LOBE_NEE: { return max(select(0.0, 1.0, c.hasD), select(0.0, rS, c.hasS)); }
    default:       { return 1e30; }                  // light vertex: always rough
  }
}
fn is_rough(c: BsdfCtx, lobe: u32) -> bool { return lobe_roughness(c, lobe) >= ROUGHNESS_MIN; }
fn is_diffuse_only(c: BsdfCtx) -> bool { return c.hasD && !c.hasS; }   // Enhanced fn. 6 skip condition

// ---------- textures ----------
fn srgb_to_linear(c: f32) -> f32 {                   // [util/color.h:63-69], exact form used on Metal
  if (c < 0.04045) { return select(c * (1.0 / 12.92), 0.0, c < 0.0); }
  return pow((c + 0.055) * (1.0 / 1.055), 2.4);
}
fn sample_srgb8(t: texture_2d_array<f32>, smp: sampler, uv: vec2f, layer: i32) -> vec4f {
  let raw = textureSampleLevel(t, smp, uv, layer, 0.0);    // rgba8unorm, bilinear on encoded bytes
  return vec4f(srgb_to_linear(raw.r), srgb_to_linear(raw.g), srgb_to_linear(raw.b), raw.a);
}
```

**Usage notes.**

- **Emitter-hit MIS weight after a BSDF bounce:**
  - `s.isDelta` → 1;
  - otherwise `power_heuristic(s.pAll, p_light)`, matching Cycles (§8.2), where p_light is the solid-angle light pdf at the hit including selection;
  - delta lights → NEE weight 1 (gris-verify C1).
- **NEE contribution:** `bsdf_eval_all(c,V,L).f · Le · w_NEE / p_light`, with `w_NEE = power_heuristic(p_light, pAll)`, or 1 for delta lights.
- **Emission** is `Strength·Color` (V1) or `EmissionColor·Strength·alpha` (V2). It is added at hits, two-sided, and independent of `c`.
- **Cost.** `lut3` is 8 f32 loads. Per vertex: 8 for S_ior, plus 8 for S5 if the material has a metal, plus 6 for Tier 2.

---

## 12. Unit-test list (extends validation-harness T8/T10)

All GPU tests run the production `bsdf.wgsl` in a compute kernel, in both the Chrome and dawn.node lanes, and once with `strictMath` (validation-harness §5.6).

**LUT and Fresnel (exact).**
- **U-1 LUT reader.** WGSL `lut3/lut2/lut1` equals a TS/CPU port on 10⁵ random (x, y, z) in [−0.1, 1.1]³, within 1e-6. Hit the table vectors in §13.1–13.3 to ±5e-5, e.g. `S_ior(0.5, 0.4, IOR 1.5) = 0.0639`, `S5(0.5, 0.1) = 0.1718`, `ggx_E(0.5,1) = 0.91528`, `ggx_Eavg(1) = 0.40914`. Also verify the 4 table checksums (§3.5) on the uploaded buffer.
- **U-2 Fresnel.**
  - `fresnel_dielectric(1, 1.5) = 0.04`; TIR returns 1 for η < 1 at grazing.
  - `F_gs(c; η, F0(η)) == F_diel(c, η)` for η ∈ (1, 3.7], within 1e-6. It must differ (clip) at η = 4 (cycles-verify C4).
  - F82 with tint = 1 equals Schlick.
  - `F82(1/7; F0, B(F0,tint)) == Schlick(1/7)·tint`, within 1e-5 (the definition of B).
  - `ior_from_F0(F0(η)) == η` for η < 19.

**Sampling correctness (χ², Mitsuba `chi2.py` / pbrt `bsdfs_test.cpp` methodology, validation-harness T8).**
- **U-3 χ² per lobe.**
  - Lambert `p_D`, and VNDF `p_S(·|V)` on the lobe's sampler alone.
  - Grid: θ_V ∈ {0, 30, 60, 80, 89}°; r ∈ {0.004, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0}.
  - 10⁶ samples, 101×(2·101) bins in (cosθ, φ), expected-frequency pooling < 5, Šidák α = 0.01.
  - Include the rejected mass: VNDF loses the below-horizon part, so compare against `∫_bin p` and assert that the histogram total equals `∫ p ≤ 1`.
- **U-4 χ² of the joint sampler.**
  - `bsdf_sample` over (L, ℓ): per lobe, the histogram of samples with lobe ℓ must match `q(ℓ|V)·p_ℓ`.
  - The pooled histogram must match `p_all`.
  - Run for V1 (f ∈ {0.2, 0.5, 0.9}) and V2 (m ∈ {0, 0.5, 1}, C ∈ {0.8 grey, (0.9,0.6,0.3)}, L ∈ {0.25, 0.5, 1}, T ∈ {white, (1,0.5,0.25)}).
- **U-5 Weight consistency.**
  - For every valid sample: `weight == eval_lobe(L).f / (q_ℓ·p_ℓ(L))` to 1e-4 relative.
  - `pJoint == bsdf_eval_lobe(c,V,L,ℓ).pJoint`.
  - `pAll == bsdf_eval_all(c,V,L).pAll` exactly (same code path).
  - `pD + pS == 1` to 1e-6.
- **U-6 q against Cycles.** `pS` equals the §13.4 vectors to 1e-4 (V2). For V1, `pS = |avg(f k)|/(|avg((1−f)ρ)|+|avg(f k)|)` exactly.

**Energy, albedo and furnace.**
- **U-7 Directional-albedo furnace (unit level).**
  - `A_est(μ) = mean over 10⁶ samples of avg(weight)` (with delta handled) must equal the quadrature values of §13.5 within 3σ + 1e-3.
  - Separately check `C·Λ(μ)` for the diffuse part and `spec_true(μ)` for the S part by forcing q.
  - Tier 1 must satisfy `A ≤ 1 + 1e-3`. A Tier-2 white metal must reach A ≈ 1 ± 5e-3 at all μ (LUT interpolation error).
- **U-8 Lobe-extended unbiasedness.** For random V and 3 different q choices (Cycles q, uniform 0.5/0.5, and q(S) = 0.9), the mean of `weight` is identical within noise (§4.3). This is a direct regression test for "any positive q".
- **U-9 Reciprocity / asymmetry.**
  - V1: `eval(V,L)/(N·L) == eval(L,V)/(N·V)` to 1e-5.
  - V2: assert the **expected** asymmetry. With C = 0.8, r = 0.5, `f_D(V,L)/f_D(L,V) = Λ(μ_V)/Λ(μ_L)`, e.g. at μ_V = 1, μ_L = 0.1 it is 0.9595/0.7963 = 1.2049. This catches V/L swaps.
- **U-10 Delta.** For r = 0.0037 (isDelta) and r = 0.0038 (not delta):
  - isDelta: mirror direction exactly; `weight == F_S/q(S)`; `eval_all` has S-part 0; `pAll` excludes S; the next-hit MIS weight is 1.
  - Just above the threshold: finite VNDF pdf, NEE non-zero.
  - The delta flag is identical in `bsdf_prepare` for base and offset V.

**ReSTIR-specific.**
- **U-11 Jacobian lobe consistency.** For random reconnection configurations: `J·J⁻¹ = 1` (validation-harness T4) with joint pdfs. Also the planted-bias control: replace joint by marginal pdfs in J and assert that the ensemble test (validation-harness §3.6) **fails**. This is the O2 negative control.
- **U-12 Predicate symmetry.** `is_rough`, `is_diffuse_only`, `skipInverse` and the footprint test are computed by the same functions in base-path construction and in offset invertibility. A census like validation-harness T3 must give zero violations. Include V2 materials where `hasD` depends on V (Λ near the 1e-5 cutoff).
- **U-13 Support indicator (smooth shading).** With `N ≠ Ng`, a shifted BSDF segment with `Ng·L ≤ 0 < N·L` must yield f = 0. An NEE segment in the same configuration must not.

**Textures.**
- **U-T1** 2×1 `rgba8unorm` texture (0, 255): sampling at the texel-pair midpoint returns 0.21404 ± 1e-3 after decode. The same data in `rgba8unorm-srgb` returns 0.5, so the test proves the wrong format is detectable. Compare against a Cycles render of a magnified checker plane: pixel-exact within noise.
- **U-T2** Wrap modes: repeat, clamp and mirror at uv = −0.25, 1.25, per glTF sampler.
- **U-T3** Upload path: a PNG with a gAMA/iCCP chunk uploads bit-identical bytes (`colorSpaceConversion: 'none'`).

**Cycles parity scenes** (validation-harness §3, stage A).
- **T-C1 BSDF furnace render** (validation-harness T10).
  - A flat-shaded UV sphere of V1/V2 material inside a closed, two-sided emissive sphere (`L_e = 1`, `FRONT_BACK`); Cycles `max_bounces = 0`, all caps equal.
  - Each pixel equals `A(μ)` for its μ. Compare the radial profile against §13.5 and against our renderer (TOST δ = 0.5 %).
- **T-C2** Tier-2 variant of T-C1 with `distribution='MULTI_GGX'`. White metal r = 1 must render ≈ 1.0, while Tier 1 renders 0.307 at μ = 1. This proves the flag works.
- **T-C3** (optional, [UNVERIFIED semantics]) Cycles light passes (Diffuse/Glossy Direct/Indirect) vs our first-bounce lobe split.
  - Expectations should coincide, because Cycles' pass weight is `eval_D/eval_total` at the first vertex and ours is the lobe choice.
  - Cycles' Direct/Indirect passes are divided by the colour pass at film conversion; replicate that division before comparing.

---

## 13. Test vectors

All values below were computed in this session from `shader.tables` @ v5.1.2 using the exact Cycles reader [VERIFIED]. Quadrature uses a 700×700 midpoint grid on the VNDF sampler, so its error is about 1e-4.

### 13.1 `S_ior(r, μ, z)` (dielectric layering; E_S = F0 + (1−F0)·S when tint = 1)

| IOR (F0) | r \ μ | 1.0 | 0.7 | 0.4 | 0.1 |
|---|---|---|---|---|---|
| 1.5 (0.04), z = 0.4472 | 0.0 | 0.0000 | 0.0117 | 0.0955 | 0.5628 |
| | 0.25 | 0.0001 | 0.0126 | 0.0927 | 0.4092 |
| | 0.5 | 0.0006 | 0.0138 | 0.0639 | 0.1705 |
| | 1.0 | 0.0014 | 0.0058 | 0.0161 | 0.0403 |
| 1.33 (0.02006) | 0.25 | 0.0000 | 0.0092 | 0.0763 | 0.3857 |
| | 0.5 | 0.0004 | 0.0105 | 0.0529 | 0.1539 |
| | 1.0 | 0.0010 | 0.0044 | 0.0130 | 0.0344 |
| 2.0 (0.11111) | 0.25 | 0.0001 | 0.0148 | 0.0969 | 0.4013 |
| | 0.5 | 0.0007 | 0.0158 | 0.0669 | 0.1700 |
| | 1.0 | 0.0017 | 0.0066 | 0.0172 | 0.0417 |

(IOR 1.5 matches cycles-conventions §4.5 and cycles-verify.)

### 13.2 `S5(r, μ) = lut3(r, μ, 0.5, table_ggx_gen_schlick_s)` (metal sample weight only)

| r \ μ | 1.0 | 0.7 | 0.4 | 0.1 |
|---|---|---|---|---|
| 0.0 | 0.0000 | 0.0034 | 0.0805 | 0.5974 |
| 0.25 | 0.0000 | 0.0044 | 0.0790 | 0.4330 |
| 0.5 | 0.0001 | 0.0068 | 0.0540 | 0.1718 |
| 1.0 | 0.0002 | 0.0030 | 0.0119 | 0.0354 |

### 13.3 `ggx_E(r, μ)` and `ggx_Eavg(r)` (Tier 2; also equals the V1 glossy albedo with F = 1)

| r | μ = 1.0 | 0.7 | 0.4 | 0.1 | Eavg |
|---|---|---|---|---|---|
| 0.25 | 0.99560 | 0.99302 | 0.98244 | 0.89780 | 0.98710 |
| 0.5 | 0.91528 | 0.88498 | 0.84441 | 0.89170 | 0.88204 |
| 1.0 | 0.30685 | 0.37894 | 0.49905 | 0.76056 | 0.40914 |

Independent quadrature of the V1 glossy (F = 1) albedo: r = 0.25 → 0.9942/0.9925/0.9821/0.8962; r = 0.5 → 0.9157/0.8853/0.8443/0.8916; r = 1 → 0.3068/0.3789/0.4989/0.7602. This agrees with the table to ≤ 5e-4.

### 13.4 V2 closure weights and q(S) (Tier 1)

Weights are per unit `mix_weight`; μ = Ns·V.

| C, m, r, IOR, L, T | μ | η' | w_D | Λ | sw_M | sw_S | sw_D | q(S) |
|---|---|---|---|---|---|---|---|---|
| 0.8 grey, 0, 0.5, 1.5, 0.5, white | 1.0 | 1.5 | 0.7676 | 0.9595 | 0 | 0.0405 | 0.7676 | 0.0502 |
| | 0.4 | | 0.7189 | 0.8987 | 0 | 0.1013 | 0.7189 | 0.1236 |
| | 0.1 | | 0.6371 | 0.7963 | 0 | 0.2037 | 0.6371 | 0.2422 |
| 0.8 grey, 0, 0.5, 1.5, **1.0**, white | 1.0 | 1.7888 | 0.7355 | 0.9193 | 0 | 0.0807 | 0.7355 | 0.0988 |
| | 0.1 | | 0.6061 | 0.7576 | 0 | 0.2424 | 0.6061 | 0.2857 |
| 0.8 grey, 0, 0.5, 1.5, **0.25**, white | 1.0 | 1.3294 | 0.7837 | 0.9796 | 0 | 0.0204 | 0.7837 | 0.0254 |
| (0.9,0.6,0.3), **0.5**, 0.5, 1.5, 0.5, white | 1.0 | 1.5 | (0.4318,0.2878,0.1439) | 0.9595 | 0.3000 | 0.0203 | 0.2878 | 0.5267 |
| | 0.4 | | (0.4044,0.2696,0.1348) | 0.8987 | 0.3108 | 0.0507 | 0.2696 | 0.5728 |
| | 0.1 | | (0.3584,0.2389,0.1195) | 0.7963 | 0.3344 | 0.1018 | 0.2389 | 0.6461 |
| (0.9,0.6,0.3), 0, 0.3, 1.5, 0.5, **(1,0.5,0.25)** | 1.0 | 1.5 | (0.8639,0.5759,0.2880) | 0.9599 | 0 | 0.0234 | 0.5759 | 0.0391 |
| | 0.1 | | (0.5615,0.3743,0.1872) | 0.6239 | 0 | 0.3653 | 0.3743 | 0.4939 |

In the last row, the coloured tint makes `max_c` pick the red channel, so Λ = 1 − E_S,red.

### 13.5 Directional albedo A(μ) for the furnace (Tier 1, single scatter)

**Dielectric** C = 0.8 grey, IOR 1.5, L = 0.5, T white, m = 0. `A = spec_true + C·Λ`.

| r | μ | E_S (LUT estimate) | Λ | spec_true (quadrature) | C·Λ | **A** | q(S) |
|---|---|---|---|---|---|---|---|
| 0.25 | 1.0 | 0.0401 | 0.9599 | 0.0398 | 0.7680 | **0.8078** | 0.0496 |
| 0.25 | 0.7 | 0.0520 | 0.9480 | 0.0514 | 0.7584 | **0.8098** | 0.0642 |
| 0.25 | 0.4 | 0.1290 | 0.8710 | 0.1278 | 0.6968 | **0.8247** | 0.1562 |
| 0.25 | 0.1 | 0.4329 | 0.5671 | 0.3913 | 0.4537 | **0.8450** | 0.4882 |
| 0.5 | 1.0 | 0.0405 | 0.9595 | 0.0370 | 0.7676 | **0.8046** | 0.0502 |
| 0.5 | 0.7 | 0.0533 | 0.9467 | 0.0466 | 0.7574 | **0.8040** | 0.0657 |
| 0.5 | 0.4 | 0.1013 | 0.8987 | 0.0835 | 0.7189 | **0.8024** | 0.1236 |
| 0.5 | 0.1 | 0.2037 | 0.7963 | 0.1718 | 0.6371 | **0.8089** | 0.2422 |
| 1.0 | 1.0 | 0.0413 | 0.9587 | 0.0127 | 0.7669 | **0.7796** | 0.0511 |
| 1.0 | 0.7 | 0.0456 | 0.9544 | 0.0175 | 0.7635 | **0.7810** | 0.0563 |
| 1.0 | 0.4 | 0.0554 | 0.9446 | 0.0279 | 0.7557 | **0.7836** | 0.0683 |
| 1.0 | 0.1 | 0.0787 | 0.9213 | 0.0585 | 0.7371 | **0.7956** | 0.0964 |

[INFERENCE] `spec_true < E_S`: the LUT estimate assumes an energy-preserving microfacet, as Cycles' own TODO at `bsdf_microfacet.h:420-422` says. So a Tier-1 Principled dielectric loses energy at high roughness. Our renderer must reproduce that loss, not "fix" it.

**Metal** m = 1, base (0.9, 0.6, 0.3), tint white (B = 0). `A = spec_true` (RGB).

| r | μ = 1.0 | 0.7 | 0.4 | 0.1 |
|---|---|---|---|---|
| 0.25 | (0.8948, 0.5965, 0.2983) | (0.8935, 0.5968, 0.3000) | (0.8914, 0.6194, 0.3474) | (0.8457, 0.6943, 0.5429) |
| 0.5 | (0.8241, 0.5494, 0.2747) | (0.7972, 0.5330, 0.2688) | (0.7639, 0.5227, 0.2814) | (0.8164, 0.5909, 0.3653) |
| 1.0 | (0.2762, 0.1841, 0.0921) | (0.3411, 0.2277, 0.1143) | (0.4495, 0.3014, 0.1532) | (0.6866, 0.4657, 0.2448) |

Metal `sw_M = avg(mix(F0, 1, S5))`: 0.6000 at μ = 1 for all r. At μ = 0.1 it is 0.7732 (r = 0.25), 0.6687 (r = 0.5) and 0.6142 (r = 1).

### 13.6 Tier-2 multipliers

See the §6.10 table. Additional values:
- White metal at μ = 0.4: ×1.1843 (r = 0.5) and ×2.0038 (r = 1).
- Dielectric IOR 1.5 at r = 0.5, μ = 0.4: spec ×1.0147, darkening 0.8568, Λ 0.8987 → 0.9132.

---

## 14. Risks and open questions

**Risks.**
1. **Silent bias from mixed conventions** (marginal throughput with joint Jacobian, or the reverse). Mitigated by U-11's planted negative control.
2. **Predicate drift under fast-math.** `q(D|V)` near a lobe boundary, or `hasD` near the 1e-5 cutoff, evaluated by differently compiled pipelines, can flip a lobe or rc decision between the base and the inverse shift. This is measure-small, not zero. Keep one call site per function and compare discrete IDs (validation-harness T2/T3).
3. **Smooth shading cannot match Cycles exactly** (§8.2), because the light sampler differs. Keep a separate tolerance tier.
4. **Tier-1 vs Tier-2 mismatch** is large (up to 3.3×) whenever a Blender reference is built with the stock glTF importer and no post-processing. The harness must force GGX, or the renderer must run Tier 2.
5. **Textures.** `-srgb` formats, `createImageBitmap` colour conversion, JPEG, block compression, padded atlases, and mip/LOD all break parity (§9).
6. **The EvanLuo42 code is not a safe template** for lobes: single-lobe-only reconnection (§4.4), α-vs-perceptual threshold (§7.4), and the swapped MIS arguments noted in reference-code §2.3.

**Open questions.**
1. Is Falcor-7 `BSDFProperties.roughness` perceptual? This decides whether EvanLuo42's α_min gate is r ≥ 0.447 [UNVERIFIED].
2. Do Apple-GPU bilinear weights through Dawn equal those Cycles-Metal gets, bit for bit? U-T1 will tell [UNVERIFIED].
3. Does Cycles premultiply byte-image alpha at load (host `scene/image.cpp`)? Only relevant for alpha textures [UNVERIFIED].
4. Can Cycles' light-pass semantics (division by the colour pass) be used for T-C3? [UNVERIFIED]
5. **Rc-predicate spec (critique Gap 4)** must confirm the NEE-final inverse-footprint pdf choice (§7.5), and whether the optional "replayed lobe == stored lobe" rule is enabled.
6. **Coat, sheen, transmission, alpha** need lobe codes and Cycles-exact layering (coat and sheen `closure_layering_weight`, coat tint, `sheen_ltc` table). Out of scope for V1/V2. Reserve lobe-code space.

---

## 15. Sources

**Cycles kernel (installed Blender 5.1.2):** `/Applications/Blender.app/Contents/Resources/5.1/scripts/addons_core/cycles/source/kernel/`

| Area | File:lines |
|---|---|
| Microfacet | `closure/bsdf_microfacet.h`: VNDF 189-220; Fresnel dispatch 226-357; `preserve_energy` 359-410; `estimate_albedo` 423-499; Λ/G/D 506-578; eval flag 581-584; eval 586-675; sample 677-820; Fresnel setups 825-946; `ggx_setup` 961-971 |
| Diffuse | `closure/bsdf_diffuse.h` 24-65 |
| Allocation | `closure/alloc.h` 55-84 |
| BSDF dispatch | `closure/bsdf.h`: `bump_shadowing_term` 65-125; `bsdf_sample` 144-295; `bsdf_eval` 512-615; `bsdf_albedo` 648-681 |
| Utilities | `closure/bsdf_util.h`: 47-99, 131-192, 332-341, 370-457, 488-492 |
| Closures | `svm/closure.h`: Principled 98-529; Diffuse 530-550; Glossy 677-744; mix 1543-1566 |
| Surface shader | `integrator/surface_shader.h`: 270-306, 365-405, 408-448, 891-923, 1102-1108 |
| Surface integrator | `integrator/shade_surface.h` 341-421, 469-603 |
| Path state | `integrator/path_state.h` 142-217 |
| Light MIS | `light/sample.h` 322-356 |
| Types | `svm/types.h` 440-545 |
| LUT reader | `util/lookup_table.h` |
| Images | `svm/image.h` 20-40; `device/gpu/image.h` 85-117; `device/metal/compat.h` 388-395 |
| Colour and maths | `util/color.h` 63-69, 347-358 (under `source/util`); `sample/mapping.h` 71-83; `util/math_float3.h` 718-741 |

**Cycles host (v5.1.2, raw.githubusercontent):**
- `intern/cycles/scene/shader.tables` (tables at lines 11, 46, 737, 1011);
- `app/cycles_precompute.cpp` 89-133 and the term registry;
- `scene/shader.cpp` 748-762;
- `scene/shader_nodes.cpp` 2725-2752.

**Blender glTF importer (5.1.20):** `io_scene_gltf2/blender/imp/pbrMetallicRoughness.py` 120-126, 316-350, 734-802; `material.py` 19-35; `mesh.py` 399-470; `io/com/constants.py:155`.

**Falcor / ReSTIR_PT** @ 8d12332: `Source/Falcor/Rendering/Materials/BxDF.slang` 40-55, 794-1053; `IBxDF.slang` 32-72; `Source/RenderPasses/ReSTIRPTPass/PathTracer.slang` 298-345.

**EvanLuo42/ReSTIR-PT-Enhanced** @ b6d30d9: `Common/EnhancedPolicy.slang` 117-215; `Path/PathSampling.slang` 30-60, 113-150, 490-570; `Common/HybridShift.slang` 55-70, 278-279, 352-353; `Common/SurfaceLoad.slang:104`.

**Papers:** GRIS §7.5–7.6 and §8 (p.14, Eqs. 49–51); Enhanced Eq. 2 (p.4), Eqs. 5–6 (p.7), §4.1–4.2 and footnotes 5–6 (p.8).

**Prior reports:** as listed in §0.

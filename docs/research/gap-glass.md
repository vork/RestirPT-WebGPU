# Gap report `glass-transmission`: exact Cycles 5.1.2 glass/transmission spec, and its integration into ReSTIR PT (hybrid shift)

> **Where this file is.** Plan mode was active; this plan file is the only writable path, so the report is here. Nothing else was written except throw-away numerical scripts in the session scratchpad (`scratchpad/glassref.py`, `check_*.py`, `slab.py`, `vectors.py`, `principled_q.py`, and a parsed copy of `shader.tables` as `tables.npz`). Nothing was installed. Blender was run headless, read-only, for RNA introspection only.

---

## 0. Provenance, tags, notation

**Primary sources read in this session**

| Key | Source |
|---|---|
| **[CY]** | Cycles kernel shipped in the app (the kernel the Metal references compile): `/Applications/Blender.app/Contents/Resources/5.1/scripts/addons_core/cycles/source/kernel/`. Files: `closure/bsdf_microfacet.h` (all 1095 lines), `closure/bsdf_util.h` 1–140, `closure/bsdf.h`, `closure/alloc.h`, `svm/closure.h` 20–529, 530–833, 1208–1252, `svm/types.h` 511–546, `integrator/path_state.h` 110–235, `integrator/shade_surface.h` 40–95, 290–460, 470–600, `integrator/surface_shader.h` 265–310, 408–448, 891–923, 948–966, `integrator/shade_shadow.h` 18–70, `bvh/intersect_filter.h` 130–185, `light/light.h` 255–300, `geom/triangle.h` 30–43, `geom/shader_data.h` 100–120, 220–235, `geom/object.h` 350–365. `file:line` below refers to this tree. |
| **[CYH]** | Cycles host code at tag `v5.1.2`, streamed read-only from raw.githubusercontent: `scene/shader.cpp` (flags 600–660, `has_surface_shadow_transparency` 311–324, tables 702–710), `scene/shader_nodes.{h,cpp}` (`has_surface_transparent`), `scene/shader.tables` (all tables, checksummed), `app/cycles_precompute.cpp` (all), `scene/object.cpp` 494–540. |
| **[GLTF]** | Installed Blender glTF importer 5.1.20: `io_scene_gltf2/blender/imp/pbrMetallicRoughness.py` 1–135, 218–314. |
| **[USD]** | `source/blender/io/usd/intern/usd_reader_material.cc` @ v5.1.2 (lines 605–645, 692–771, 1075–1160), streamed read-only. |
| **[BPY]** | `Blender --background --factory-startup --python-expr …` introspection in this session. |
| **[NUM]** | Numerical checks in this session with Blender's bundled Python 3.13 + numpy: a line-by-line numpy port of Cycles' GGX glass `eval`/`sample` (`scratchpad/glassref.py`), checked against Cycles' own precomputed albedo tables. |
| **[GRIS] / [ENH]** | `scratchpad/pages/gris/p-13.png` (§7.1–7.5); `restirpt_enhanced_2026.txt` (Fig. 1, §7.3, Fig. 11, §8). |
| prior reports | cycles-conventions `a2d6cc1a462b20a80` §4 (+verify `ac3ddf11ffb75ef90` C2/C4/omissions), gap-bsdf `aa7778fb0ac69ed41` (all), gris-math `a7f08dec9968f0324` §6.2–6.8, reference-code `a3b89657b22d15f34` §1.3–1.6, gap-rc `ad74911edeeb27e44` §0–§5, gap-light `a40054d87d055e569` §0–§2, scene-io `a597df4a1dd101994` (glTF extension table), the draft plan `do-a-deep-dive-shimmering-ritchie.md`. |

**Tags.** **[SOURCE]** stated by code/paper at the cited location. **[VERIFIED]** checked in this session (code read, introspection, or computation). **[INFERENCE]** my derivation/decision. **[UNVERIFIED]** plausible, not checked. **NORMATIVE** = implementation rule.

**Direction convention (Cycles, as in gap-bsdf).** `V` = `sd->wi` points from the shading point toward the previous vertex (camera side). `L` = `wo` points to the next vertex/light. "eval" includes `|N·L|`. `N`, `Ng` are flipped to V's side when backfacing. `η` (Cycles `bsdf->ior`) is the **relative IOR seen from V's side**: `ior` if front-facing, `1/ior` if backfacing.

---

## 1. TL;DR (normative decisions)

1. **One new lobe class `G` = Cycles' GGX glass/refraction closure** (reflection + refraction sub-events, one roughness, one η). It covers the Principled transmission closure, the Glass BSDF node and the Refraction BSDF node. The R/T sub-event is a function of `L` (sign of `N·L`), not a separate lobe. Lobe codes become 3 bits: `D=0, S=1, G=2, TRANSP=3 (reserved, M7 alpha), NEE=4, NONE=5`.
2. **Exact Cycles glass BSDF** (§2): Walter-2007 GGX BTDF/BRDF with Cycles' own normalisation; Fresnel = real unpolarised dielectric Fresnel remapped to `[f0, 1]` ("generalized Schlick, exponent < 0"), which equals the real Fresnel for white Specular Tint and `1/3.73 < η < 3.73`; R/T chosen per sample with `P_R = avg(R)/avg(R+T)` at the **sampled microfacet**; TIR ⇒ `T = 0`, always reflect. Principled tints transmission by **`sqrt(min(BaseColor,1))` per interface**; the Glass node tints R and T by `Color`; the Refraction node has no reflection and no Fresnel. **No "Transmission Roughness" exists in 5.1** [VERIFIED BPY]. No thin-walled mode exists in Cycles.
3. **Cycles applies no 1/η² radiance scaling** [VERIFIED code + NUM]: the sampling weight of refraction is `T/P_T · G2/G1` for any η. Our PT and every shift must use exactly Cycles' `f_G(V,L)` (non-reciprocal in η). Jacobians carry **no η² factor**; η enters only through the evaluated pdfs.
4. **Glass is opaque to every visibility query** (NEE shadow rays, reconnection rays, refresh rays) [VERIFIED: glass has no transparent closure, `surface_shader_transparency = 0`]. Consequences (§4): through **smooth** glass, analytic lights in mode A and delta lights in every mode contribute **exactly zero** to what lies behind; through **rough** glass light arrives via *exit-face NEE* (NEE from the inner surface through the exit interface); with MIS-on (mode B) area lights, smooth glass passes light only via BSDF-sampled caustic paths.
5. **The plan's "Mode A ≡ Mode B in expectation" is false for scenes with singular (delta) BSDFs** (mirrors with r ≤ 0.00376, smooth glass): mode A drops every light-ending path whose last scattering vertex is delta; mode B keeps them via BSDF sampling. The A≡B invariance test must use scenes without singular lobes; add a negative control.
6. **Cycles' refraction `eval` has a spurious region** [VERIFIED NUM, SOURCE `bsdf_microfacet.h:621-623` TODO]: it never checks the generalized half-vector configuration, so `eval`/`pdf` are nonzero for transmission directions the sampler can never produce. The integral of Cycles' eval-pdf exceeds 1 (1.11 at r=0.53, μ=0.4, η=1.557; 1.92 at r=1, μ=0.2). NORMATIVE for us:
   - `f_G` (NEE integrand) = Cycles eval **including** the spurious part (needed for parity: in mode A NEE weight is 1, so Cycles counts it fully);
   - **every pdf used for MIS, footprints, Jacobians and χ² tests = the true sampler density ("valid-only")**, and every *BSDF-sampled* segment in a shift is multiplied by the sampler-support indicator. This "support-consistent MIS" makes MIS a partition of unity for any heuristic and any RIS-NEE M, so ReSTIR ≡ PT exactly.
   - Parity consequence: exact in mode A and for emitters with `emission_sampling='NONE'`; **approximate** (documented tolerance tier) only when an MIS-sampled emitter (mode-B area light or FRONT/BACK emissive mesh) is NEE-reached through a rough transmission lobe.
7. **η is stateless and local**: `η_side = backfacing ? 1/ior : ior`, with backfacing from `Ng·V < 0` and `Ng = normalize(cross(v1−v0, v2−v0))`, flipped for negative-scale objects [SOURCE `triangle.h:39-42`]. No medium stack, **no nested-dielectric priority logic** (Falcor's `handleNestedDielectrics` must not be ported). Triangle winding and negative-determinant transforms must match Blender exactly, or η inverts.
8. **Delta for G is per sub-event**: R is delta iff `α² ≤ 2e-10` (r ≤ 0.0037606); T is delta iff `α² ≤ 2e-10` **or** `|η_side − 1| < 1e-4` [SOURCE `bsdf_microfacet.h:771`].
9. **Reconnection with glass** (§6): delta glass never participates in a passing pair (D_k); rough glass may be `x_{k−1}` (lobe G, `r_G ≥ 0.2`) and `x_k` (inverse footprint always evaluated; G is never "diffuse-only"). Random replay through any glass (smooth or rough, R/T/TIR) is the PSS identity with J = 1; **TIR never makes a hybrid shift undefined** (only half-vector copy has that problem); at a reconnection vertex TIR just gives f = 0.
10. **RNG: 4 dimensions per bounce** `(u_lobe, u_h1, u_h2, u_rt)` (Cycles instead rescales `u_lobe` for the R/T decision; both are unbiased; a separate dimension is cleaner for replay).
11. **Validation must force `distribution='GGX'` on Glass BSDF nodes too** (default is **MULTI_GGX**) and on Refraction nodes (default **BECKMANN**) [VERIFIED BPY]. Tier 2 (MULTI_GGX glass) needs 4 more tables (8704 floats) and matters for imported rough glass (×1.008…×2.37).
12. **Existing WGSL bug to fix before glass**: gap-bsdf §11 `ggx_lambda` clamps `max(cosN, 1e-7)`; for refraction `cos_NO < 0`, so it must be `max(abs(cosN), 1e-7)`, or G2 → 0 for every transmission.
13. **New closed-form validation scene** (§7): smooth glass slab in an L_e = 1 enclosure, per pixel `E(θ) = F + (1−F)·C·(1−F^N)` (Principled, grey C, `max_bounces = N`) [VERIFIED against an MC emulation of Cycles' sampling]; plus "emitter inside glass" `E = (1−F)·√C·L_e`, which detects any η² scaling bug (a 2.25× error).

---

## 2. Cycles 5.1.2 glass: exact formulas

### 2.1 Where glass closures come from

**Principled BSDF** [SOURCE `svm/closure.h:169-509`]. The order in which closures are built, with `weight` starting at `mix_weight` (1):

| Step | Closure | Closure weight | Notes |
|---|---|---|---|
| 0 | clamps | – | `C = max(Base,0)`, `Cc = min(C,1)`, `ior = max(IOR,1e-5)`, `r = sat(Roughness)`, `m = sat(Metallic)`, `t = sat(Transmission Weight)`, `T_s = max(SpecTint,0)` |
| 1 | transparent (alpha < 1) | `(1−α)·weight`; then `weight *= α` | out of scope (validation α = 1) |
| 2 | sheen, coat | layering | out of scope (0) |
| 3 | emission | `E_c·E_s·weight` | not attenuated by transmission |
| 4 | **metal** (m > 1e-5) | `w_M = m·weight` | F82-tint, then `weight *= (1−m)` |
| 5 | **glass** (t > 1e-5) | `w_G = t·weight = t(1−m)α` | GGX glass closure (below), then `weight *= (1−t)` (**no albedo layering** after glass) |
| 6 | IOR level | – | `η' = ior_from_F0(F0(ior)·2L)` if L ≠ 0.5 |
| 7 | dielectric specular | `w_S = weight = (1−t)(1−m)α` | reflection only, η' (**not** inverted on backfacing hits), then `weight = layering(E_S, weight)` |
| 8 | diffuse | `w_D = C·(1−t)(1−m)α·Λ_S(μ)` | unclamped C, Lambert if Diffuse Roughness < 1e-5 |

**Principled glass closure parameters** [SOURCE `closure.h:378-411`]:
- `N = valid_reflection_N` (= N with `use_bump_map_correction = False`), `T = 0`; **isotropic** `α_x = α_y = r²` even when Anisotropic > 0.
- `η_side = SD_BACKFACING ? 1/ior : ior`. Note: the IOR here is the node IOR. The Specular IOR Level η' affects only step 7.
- Fresnel = generalized Schlick with `f0 = saturate(F0(ior)·T_s)` (saturate at setup, `bsdf_microfacet.h:869`), `f90 = 1`, `exponent = −ior` (<0 ⇒ "real Fresnel remapped" mode), `reflection_tint = 1`, **`transmission_tint = sqrt(Cc)`**, thin film = node Thin Film (0 in validation).
- Setup: `bsdf_microfacet_ggx_glass_setup` then `bsdf_microfacet_setup_fresnel_generalized_schlick(…, is_multiggx)`.

**Glass BSDF node** [SOURCE `closure.h:779-832`]:
- Closure weight = `mix_weight` (grey). **`Color` goes into both tints**: `reflection_tint = transmission_tint = max(Color,0)`.
- `f0 = F0(ior)` (no tint), `f90 = 1`, `exponent = −ior`, `η_side` as above, `α = sat(Roughness)²`.
- Distribution enum `BECKMANN | GGX | MULTI_GGX`, **default MULTI_GGX** [VERIFIED BPY]. Inputs: Color (1,1,1), Roughness 0, IOR 1.5, Thin Film Thickness 0, Thin Film IOR 1.33.

**Refraction BSDF node** [SOURCE `closure.h:745-777`]:
- Closure weight = `Color·mix_weight`. Fresnel `NONE`: `R = 0`; `T = 1` except under TIR, where `T = 0` and the path dies (energy loss, no reflection).
- `α = r²` (saturated in setup).
- Distribution enum `BECKMANN | GGX`, **default BECKMANN** [VERIFIED BPY]. IOR default 1.45.

**Principled inputs in 5.1.2** [VERIFIED BPY]: `Transmission Weight` (default 0) exists. **There is no `Transmission Roughness` input**: glass uses the shared `Roughness`. Thin Film Thickness 0 / IOR 1.33 exist and also feed the glass Fresnel when > 0.1 nm (`THINFILM_THICKNESS_CUTOFF = 0.1`). Exclude them.

### 2.2 Fresnel for glass [SOURCE `bsdf_microfacet.h:226-357`, `bsdf_util.h:47-99`]

```
F_diel(c, η): g = η² − (1 − c²); if g ≤ 0 → 1 (TIR), cosT = unset
              ci = |c|; cosT = −sqrt(g)/η  (signed, opposite side of H)
              r_s = (ci + η cosT)/(ci − η cosT); r_p = (cosT + η ci)/(η ci − cosT); F = (r_s² + r_p²)/2
Gen.Schlick (exponent<0):  s = saturate((F_diel(c, η_side) − F0(η_side)) / (1 − F0(η_side)));  F = mix(f0, f90=1, s)
                            R = F·reflection_tint,  T = (1 − F)·transmission_tint
Refraction node (NONE):     R = 0;  T = (F_diel == 1 ? 0 : 1)
```

- `F0(η) = ((η−1)/(η+1))²` is the same for η and 1/η.
- With `T_s = white`, `F = F_diel` exactly for **both sides** whenever `1/3.73 < η_side < 3.73` [VERIFIED NUM: `min(F_diel − F0) = 0` for η and 1/η up to 3.73; −1.6e-4 at 3.8, −2.2e-3 at 4.0]. This extends cycles-verify C4 to the inside.
- Under TIR, F = 1, so T = 0 and P_R = 1: always reflect.
- It is the real dielectric Fresnel (not Schlick) that shapes the curve. `f0·T_s` only rescales its floor.

### 2.3 eval·cos and pdf, exactly as Cycles writes them [SOURCE `bsdf_microfacet.h:586-675`]

With `cNI = N·V`, `cNO = N·L`, `a2 = α_x α_y` (isotropic for glass):

```
return 0 if cNI ≤ 0, or a2 ≤ 2e-10 (singular: no eval), or (isT && !has_T), or (!isT && !has_R)
isT      = cNO < 0
H        = isT ? −(η·L + V) : (V + L);  invLen = safe 1/|H|;  H *= invLen          (NOT re-oriented; sign-invariant below)
(R, T)   = fresnel(cHI = H·V)                    ; return 0 if both are zero
D        = a2 / (π·((1 − cNH²) + a2·cNH²)²),  cNH² = min((N·H)², 1)
Λ(c)     = ½(sqrt(1 + a2·max(1/c² − 1, 0)) − 1)   (uses c², so the sign of cNO does not matter)
common   = D/cNI · ( isT ? (η·invLen)²·|cHI·(H·L)|  :  1/4 )
P_R      = avg(R) / avg(R + T);   lobe = isT ? 1 − P_R : P_R
pdf      = common · lobe / (1 + Λ(cNI))
eval·cos = (isT ? T : R) · common / (1 + Λ(cNI) + Λ(cNO))    [× energy_scale (Tier 2); × closure weight]
```

The refraction half-vector Jacobian `|dω_h/dω_o| = η²|o·h| / (i·h + η·o·h)²` appears as `(η·invLen)²·|H·L|`, because `|V + ηL|² = (cHI + η cHO)²` (V+ηL ∥ H). With the VNDF term `G1(V)|V·H|D/|N·V|` this gives the Walter-2007 form above. Cycles writes the same term in `sample()` as `|cHI·cHO| / (cHO + cHI/η)²` (`:807-809`). The two are identical on valid configurations.

**The spurious region** [VERIFIED NUM + SOURCE TODO at `:621-623`]. `eval` never checks that the refraction configuration is physical. After orienting `Hn = H·sign(N·H)`, the sampler can produce `L` only if `Hn·V > 0`, `Hn·L < 0`, `N·L < 0`, `Ng·L < 0` and there is no TIR at `Hn`. `eval` returns nonzero values outside that set:

| r | μ=N·V | η_side | ∫ eval (all) | ∫ eval (valid only) = sampler albedo | ∫ pdf (all) | ∫ pdf (valid) | spurious eval |
|---|---|---|---|---|---|---|---|
| 0.533 | 0.4 | 1.5568 | 1.04343 | 0.93690 (MC 0.93703) | **1.1126** | 0.9758 | 0.1065 |
| 0.533 | 1.0 | 1.5568 | 1.00060 | 0.98417 | 1.0129 | 0.9866 | 0.0164 |
| 1.0 | 0.2 | 1.5568 | 1.30700 | 0.70644 (MC 0.70687) | **1.9213** | 0.9751 | 0.6006 |
| 0.533 | 0.4 | 0.6423 | 0.82299 | 0.78911 (MC 0.78927) | 0.9421 | 0.9029 | 0.0339 |
| 1.0 | 0.2 | 0.6423 | 0.88241 | 0.50593 (MC 0.50597) | 1.2388 | 0.7640 | 0.3765 |
| 0.2 | 0.7 | 1.5 | 1.00060 | 0.99951 | 1.0008 | 0.9997 | 0.0011 |

(1600² quadrature over the sphere vs 2×10⁶-sample MC of the Cycles sampler port.) Reflection has no spurious region: `normalize(V+L)` is always valid for two upper-hemisphere directions. A second spurious source is rough refraction with `|η−1| < 1e-4`: sampling makes T singular (straight through), while `eval` still returns a rough BTDF lobe.

### 2.4 Sampling [SOURCE `bsdf_microfacet.h:677-820`]

```
if cNI ≤ 0 → LABEL_NONE
singular = !(a2 > 2e-10)
H = singular ? N : VNDF_GGX(V_local, α, (u.x, u.y))                    (Heitz 2018; same as gap-bsdf §3.2)
(R, T, cHO) = fresnel(cHI)                  ; NONE if both are zero
P_R = avg(R)/avg(R+T);  refract = (u.z ≥ P_R)                         (u.z = rescaled closure-pick number in Cycles)
L = refract ? (cHI/η + cHO)·H − V/η  :  2cHI·H − V
reject (LABEL_NONE) if (Ng·L < 0) ≠ refract  or  (N·L < 0) ≠ refract
if refract: singular |= |η − 1| < 1e-4
eval = refract ? T : R;  pdf = refract ? 1 − P_R : P_R
singular: eval·1e6, pdf·1e6          → weight = eval/pdf = T/(1−P_R) or R/P_R  (then × energy_scale)
else:     common as in §2.3 (sample form);  pdf *= common/(1+Λ_I);  eval *= common/(1+Λ_I+Λ_O)
          weight = (R or T)/(P_R or 1−P_R) · (1+Λ_I)/(1+Λ_I+Λ_O)     ← no η² anywhere
label: REFLECT|TRANSMIT  ×  GLOSSY|SINGULAR
```

Consequences:
- **Glass node**: `P_R = F` exactly (`avg(F·c)/avg(c)`), so the refraction weight is `c·G2/G1`.
- **Principled**: `P_R = F/(F + (1−F)·avg(√Cc))`. The refraction weight is `√Cc·(F + (1−F)a)/a · G2/G1` with `a = avg(√Cc)`.
- **Singular glass**: throughput `(1−F)√Cc/(1−P_R)` or `F/P_R`. NEE is skipped (no `SD_BSDF_HAS_EVAL`), and the next emitter hit gets MIS weight 1 (`PATH_RAY_MIS_SKIP`, `path_state.h:212-216`).

### 2.5 Integrator semantics around glass [SOURCE]

- **Bounces** (`path_state.h:110-231`). A transmission (rough or singular) increments `bounce` **and** `transmission_bounce`, capped by `max_transmission_bounce` (default 12). A reflection off glass is a glossy bounce. The total `max_bounces` counts every glass interface. Keep all per-type caps equal to `max_bounces` (plan M2 already does).
- **No radiance scaling.** `bsdf_eta` goes only to path guiding (`shade_surface.h:588-600`). The throughput is `bsdf_eval/bsdf_pdf` (`:563-565`).
- **Throughput convention.** Cycles uses one-sample MIS over closures (`surface_shader.h:891-923`: throughput = Σ_i eval_i / Σ_i sw_i pdf_i). That is convention B of gap-bsdf §4.4, and unbiased like our convention A. With a singular pick, the other closures add only O(1e-6).
- **Shadow rays** (`intersect_filter.h:151-160`, `shade_shadow.h:60-66`, `surface_shader.h:948-957`). A hit on a shader without `SD_HAS_TRANSPARENT_SHADOW` blocks immediately. That flag is set only by a Transparent BSDF, a Ray Portal, Principled with Alpha < 1 or linked, **or a volume shader** ([CYH] `shader.cpp:628-640`). Even then the surface transparency is the transparent closure's weight, which is **0 for glass**. So glass **always blocks** NEE and AO rays, including glass with a volume (the volume flag only makes the shadow ray record the hit).
- **MNEE is off by default** [VERIFIED BPY: `object.cycles.is_caustics_caster/receiver = False`, `light.cycles.is_caustics_light = False`]. Keep it off.
- **Backfacing** (`shader_data.h:107-113`, `triangle.h:30-43`): `Ng = normalize(cross(v1−v0, v2−v0))`, reversed if `object_negative_scale_applied`. If `Ng·wi < 0`, Ng and N flip and `SD_BACKFACING` is set, so the glass η becomes 1/ior.
- **Caustics tricks**: `caustics_reflective/refractive = True` by default. Keep them on (False drops glass after diffuse bounces: biased).
- **Filter Glossy** also widens glass α (`bsdf.h` `bsdf_blur` includes the GLASS IDs). Keep `blur_glossy = 0`.
- **`film_transparent_glass`** (default False) matters only with `film_transparent`, which is off. Irrelevant.

### 2.6 Lobe-selection weights (noise-only, but needed for Cycles-identical MIS) [SOURCE `alloc.h:64`, `bsdf_microfacet.h:423-499, 862-899`]

```
sw_G = |avg(w_G)| · avg( R_est + T_est )          [× avg(darkening) in Tier 2 if Fss ≠ 1]
  μ = N·V (flipped N);  z = sqrt(|(η_side − 1)/(η_side + 1)|)   (same z for η and 1/η)
  R_est = mix(f0, 1, lut3(r, μ, z, table_ggx_gen_schlick_ior_s))·reflection_tint     (LUT, as for the specular layer)
  T_est = (1 − F_gs(μ; η_side))·transmission_tint                                     (smooth-surface estimate at N, NOT at H)
Refraction node: no Fresnel setup ⇒ sw = |avg(Color·mix_weight)|, with no albedo scaling
```

- Inside and beyond the critical angle, `T_est = 0` while `R_est` comes from a table built for η > 1. That is only a noise-level approximation.
- Test vectors [VERIFIED NUM], grey C = 0.8, IOR 1.5, r = 0.5:

| Case | sw_G | q(G) |
|---|---|---|
| t = 1, μ = 1, front-facing | 0.8992 | 1 |
| t = 1, μ = 0.4, front-facing | 0.8773 | 1 |
| t = 1, μ = 0.4, **backfacing** (TIR at N) | 0.1013 | 1 |
| t = 0.5, μ = 1 | 0.4496 | 0.5267 (q(S) 0.0237, q(D) 0.4496) |
| t = 0.5, μ = 0.1 | 0.2934 | 0.4111 (q(S) 0.1427, q(D) 0.4463) |
| C = (0.9, 0.6, 0.3), m = 0.25, t = 0.5, r = 0.3, μ = 0.7 | sw_M 0.1505, sw_G 0.2891, sw_S 0.0197, sw_D 0.2132 | q(G) 0.4299 |

### 2.7 Tier 2 (MULTI_GGX) glass [SOURCE `bsdf_microfacet.h:359-410, 862-899`; CYH precompute]

```
ior' = η_side;  tables = (glass_E, glass_Eavg);  if ior' < 1: ior' = 1/ior', tables = (glass_inv_E, glass_inv_Eavg)
z = sqrt(|(ior'−1)/(ior'+1)|);  E = lut3(r, μ, z, *_E, 16);  Eavg = lut2(r, z, *_Eavg, 16, 16)   (x = rough, y = z)
missing = (1−E)/E;  energy_scale = 1 + missing
Fss = transmission_tint (√Cc for Principled, Color for the Glass node)
if Fss ≡ 1: multiplier = 1/E on R and T
else:       Fms = Fss·Eavg/(1 − Fss(1−Eavg));  darkening = (1 + Fms·missing)/energy_scale;
            weight ×= darkening;  sample_weight ×= avg(darkening);  eval·cos multiplier = 1 + Fms·missing
Applied to singular samples too (E = 1 at r = 0 ⇒ no effect on smooth glass).
```

**Tables** (copy verbatim from `shader.tables` @ v5.1.2) [VERIFIED checksums]:

| Table | Floats | Σ over table |
|---|---|---|
| `table_ggx_glass_E` | 4096 | 3758.664856 |
| `table_ggx_glass_Eavg` | 256 | 235.910746 |
| `table_ggx_glass_inv_E` | 4096 | 3477.751955 |
| `table_ggx_glass_inv_Eavg` | 256 | 216.090299 |

Appending them to gap-bsdf's LUT buffer gives offsets `LUT_GLASS_E = 9248`, `LUT_GLASS_EAVG = 13344`, `LUT_GLASS_INV_E = 13600`, `LUT_GLASS_INV_EAVG = 17696`; the buffer grows from 9248 to **17952 floats (71,808 B)**. No new binding.

**Magnitudes** (energy_scale = 1/E; white glass):

| η_side | r | μ = 1 | μ = 0.4 |
|---|---|---|---|
| 1.5 | 0.25 | 1.0005 | 1.0057 |
| 1.5 | 0.5 | 1.0083 | 1.0568 |
| 1.5 | 1.0 | 1.1200 | 1.3917 |
| 1/1.5 | 0.25 | 1.0044 | 1.0191 |
| 1/1.5 | 0.5 | 1.0762 | 1.2132 |
| 1/1.5 | 1.0 | **2.3706** | 1.9477 |

- **Observation** [VERIFIED NUM; cause UNVERIFIED]. `glass_E` (η > 1) matches the 5.1.2 sampler's single-scatter albedo to MC precision at every grid point tested, e.g. (r, μ, z) = (8/15, 1, 7/15): MC 0.98866 vs table 0.98860.
- `glass_inv_E` matches at μ = 1 but **not in the TIR regime**:
  - (8/15, 0.4, 7/15): MC 0.7893 vs table 0.8041;
  - (1, 0.2, 7/15): MC 0.5060 vs table 0.5649.
  - The tables probably predate a sampler change. Tier 2 must **port the tables verbatim, not regenerate them**. Unit tests must expect the table values, not "albedo × multiplier = 1", for inside hits.

---

## 3. Single-scatter albedo test vectors (Tier 1, valid-only sampler) [VERIFIED NUM, 10⁶ samples; `(R…)` = reflection part]

| η_side | r | μ = 1 | μ = 0.7 | μ = 0.4 | μ = 0.1 |
|---|---|---|---|---|---|
| 1.5 | 0.1 | 1.0000 (R 0.0396) | 1.0000 (0.0510) | 0.9999 (0.1324) | 0.9959 (0.5610) |
| 1.5 | 0.25 | 0.9995 (0.0398) | 0.9988 (0.0514) | 0.9948 (0.1276) | 0.9295 (0.3915) |
| 1.5 | 0.5 | 0.9920 (0.0370) | 0.9793 (0.0467) | 0.9468 (0.0833) | 0.9163 (0.1717) |
| 1.5 | 1.0 | 0.8932 (0.0127) | 0.8002 (0.0177) | 0.7190 (0.0279) | 0.7413 (0.0583) |
| 1/1.5 | 0.1 | 0.9999 (0.0399) | 0.9998 (0.9957) | 0.9996 (0.9994) | 0.9939 (0.9938) |
| 1/1.5 | 0.25 | 0.9960 (0.0412) | 0.9879 (0.8843) | 0.9807 (0.9745) | 0.8885 (0.8855) |
| 1/1.5 | 0.5 | 0.9290 (0.0481) | 0.8626 (0.5403) | 0.8112 (0.7412) | 0.8105 (0.7911) |
| 1/1.5 | 1.0 | 0.4168 (0.0258) | 0.4964 (0.1121) | 0.5053 (0.2220) | 0.5112 (0.4312) |
| 1.33 | 0.25 | 0.9997 (0.0202) | 0.9989 (0.0287) | 0.9951 (0.0940) | 0.9318 (0.3589) |
| 1.33 | 0.5 | 0.9944 (0.0186) | 0.9816 (0.0261) | 0.9480 (0.0583) | 0.9087 (0.1423) |
| 1.33 | 1.0 | 0.9233 (0.0065) | 0.8181 (0.0094) | 0.7211 (0.0164) | 0.7197 (0.0395) |

A smooth dielectric has albedo exactly 1 on both sides in Cycles' convention; with 1/η² scaling it would be F + (1−F)/η². The T-part at η = 1.557 is 0.946, not 0.39, which proves there is no η² in the weights.

**Eval/pdf test vectors** (Glass node, Color 1; N = +z; directions as (θ, φ); `valid` = sampler support) [VERIFIED NUM]:

| r | η_side | V | L | eval·cos | pdf (Cycles) | valid |
|---|---|---|---|---|---|---|
| 0.5 | 1.5 | (30°, 0°) | (40°, 180°, below) | 2.466877e-1 | 2.493586e-1 | yes |
| 0.5 | 1.5 | (30°, 0°) | (15°, 180°, below) | 1.032414e+1 | 1.033565e+1 | yes |
| 0.5 | 1.5 | (30°, 0°) | (30°, 0°, above) | 2.579733e-3 | 2.593031e-3 | yes (R) |
| 0.3 | 1.5 | (60°, 0°) | (20°, 180°, below) | 4.155928e-2 | 4.157036e-2 | yes |
| 0.5 | 1/1.5 | (20°, 0°) | (35°, 180°, below) | 7.401114e+0 | 7.457269e+0 | yes |
| 0.5 | 1/1.5 | (60°, 0°) | (70°, 180°, below) | 4.183677e-1 | 4.610455e-1 | yes |
| 1.0 | 1.5 | (80°, 0°) | (80°, 0°, below) | 1.097268e-1 | 1.869842e-1 | **no (spurious)** |
| 0.5 | 1.5 | (30°, 0°) | (60°, 0°, below) | 6.523885e-3 | 6.815053e-3 | **no (spurious)** |

---

## 4. What NEE does at glass, and what Cycles renders behind glass

### 4.1 Mechanics [SOURCE, §2.5]

1. **At a singular glass vertex** there is no NEE (no `SD_BSDF_HAS_EVAL`). Emission found by the next BSDF ray has MIS weight 1.
2. **At a rough glass vertex** NEE happens. The shadow ray is blocked by any glass surface it crosses. It succeeds only if the segment from the vertex to the light crosses no other triangle. For closed objects that means:
   - **Entry face, light on the far side:** blocked by the exit face.
   - **Exit face (an interior hit, `V` inside, light outside):** *unoccluded*. NEE uses the transmission eval (η_side = 1/ior), including the spurious region. "Exit-face NEE" is the only way NEE carries light *through* glass.
   - **Light inside the glass** (an emissive mesh embedded in it): entry-face NEE works.
   - **Open, single-sided panes:** NEE through the pane works directly.
3. **Delta lights** (point/spot r = 0, sun angle 0) are never hit by any ray. **Mode-A analytic lights** (MIS off) are never hit by BSDF rays, even after singular bounces (`light.h:283-286` has no MIS-skip exception) [VERIFIED]. **Mode-B area lights** and **emissive meshes** are hit by BSDF rays, with weight 1 after a singular bounce.

### 4.2 What Cycles renders (MNEE off, defaults otherwise)

"Floor behind glass" means a diffuse floor that sees the light **only through** the glass object.

| Light | Smooth glass (r ≤ 0.00376), closed | Rough glass, closed slab | Rough glass, single-quad pane |
|---|---|---|---|
| Point/spot r = 0 (any MIS setting) | **exactly 0** | > 0 via `floor → x2 (T, BSDF) → x3 (exit-face NEE, weight 1)`; needs ≥ 3 scattering vertices (max_bounces ≥ 2) | > 0 via `floor → pane (NEE through pane)` |
| Area light, MIS off (**mode A**) | **exactly 0** | same as point (NEE weight 1) | same as point |
| Area light, MIS on (**mode B**, Blender default) | > 0 only as a BSDF-sampled caustic `floor → x2(T,δ) → x3(T,δ) → light` with weight 1 (noisy) | exit-face NEE with power heuristic (p_bsdf includes spurious pdf) + BSDF hits × w_bsdf | pane NEE + BSDF hits |
| Emissive mesh (always MIS) | BSDF-only caustic, weight 1 | exit-face NEE + BSDF hits (MIS) | pane NEE + BSDF hits |

The same logic applies to **perfect mirrors**: in mode A a mirror never shows a lamp and never reflects lamp light onto the floor. In mode B it does, through BSDF sampling.

**Camera looking through glass at a lit floor** works in every mode. The path is camera → δ → δ → floor, with NEE at the floor.

### 4.3 Consequences for the plan [INFERENCE]

1. **Mode A ≡ Mode B is false with singular BSDFs.**
   - The gap-light §2.7 proof assumed NEE covers every analytic light point. It does not when `x_{d−1}` is delta.
   - Amend plan §1 Lights to read: *"identical expectation iff no light-ending path has a singular lobe at x_{d−1}"*.
   - Restrict the A≡B unit test to scenes without singular lobes.
   - Add a negative control: a smooth-glass or mirror scene where A < B, as predicted.
2. **Parity is exact in both modes**, as long as the Blender reference uses the same mode. Our mode A drops the same paths Cycles-mode-A drops.
3. **Look of glass scenes.**
   - In mode A, smooth glass casts pitch-black shadows from all analytic lights, and mirrors never show lamps.
   - For the interactive app, recommend mode B for area lights once validated, so smooth glass passes area light as (noisy) caustics.
   - Point and spot lights never pass through smooth glass in Cycles either. Document this as expected behaviour, not a bug.
   - A "shadow-transparent glass" hack would break parity. Do not add it.
4. **ReSTIR efficiency.** Delta→light caustic paths have no passing pair: D fails at every glass vertex. They are always fully replayed (k* = ∅), so reuse helps little. The Enhanced paper itself names caustics as an open problem ([ENH] §8).

---

## 5. Lobe model extension (extends gap-bsdf; all NORMATIVE unless tagged)

### 5.1 Classes and codes

| Code | Class | Members | Sub-events |
|---|---|---|---|
| 0 `D` | Lambert | Principled diffuse, Diffuse node | R |
| 1 `S` | GGX reflection-only | Principled metal + dielectric specular, Glossy node | R |
| 2 `G` | **GGX glass** | Principled transmission closure, Glass node, Refraction node (`hasR = false`) | R and T, decided by `sign(N·L)` |
| 3 `TRANSP` | reserved (M7 alpha) | | |
| 4 `NEE` | outgoing direction NEE-sampled (all lobes evaluated) | | |
| 5 `NONE` | light/env vertex | | |

- **Why G is not merged with S.** S merges exactly because metal and specular share one VNDF density (gap-bsdf §6.4). G's density includes P_R/P_T and refraction, so a merged class would be a mixture. That is legal, but unnecessary.
- **Why R/T are not separate lobes.** Cycles decides them inside one closure. The joint density `q(G|V)·p_G(L|V)` is a single density over the sphere (R above, T below), exactly as in Falcor's "specular reflection + transmission" class `0xC` ([FAL] via gap-bsdf §2.3).
- A **copied-side restriction** is an optional, symmetric (hence unbiased) experiment flag: "the event side at y_{k−1} must equal the base's side at x_{k−1}". Default **off**.

### 5.2 pdfs (two kinds; never mix)

```
p_G^C(L|V)    = Cycles eval pdf (§2.3), spurious included      → used ONLY for Cycles-parity diagnostics
p_G(L|V)      = p_G^C(L|V) · 1[L ∈ supp(sampler_G | V)]         → the "valid-only" density = true sampler density
supp(sampler_G|V): R: Ng·L ≥ 0 ∧ N·L ≥ 0;  T: Ng·L < 0 ∧ N·L < 0 ∧ Hn·V > 0 ∧ Hn·L < 0 ∧ no TIR at Hn ∧ |η_side−1| ≥ 1e-4
                   (Hn = normalize(−(η L + V)) oriented to N·Hn > 0; with η≈1 the T sub-event is delta)
joint  p(L,G|V)  = q(G|V)·p_G(L|V)
marginal p_all(L|V) = q(D)p_D + q(S)p_S + q(G)p_G                (for L below the surface only G contributes: marginal = joint)
```

| Consumer | Uses |
|---|---|
| Throughput at a G-sampled vertex | `f_G/(q_G·p_G)` (automatically valid-only) |
| NEE contribution at a vertex | `f_all(V,L) = f_D + f_S + f_G` with **Cycles eval (spurious included)**, no support indicator |
| NEE MIS `ω1 = M p1/(M p1 + p2)` and emitter-hit MIS | `p2 = p_all` (**valid-only**) |
| Enhanced footprints (F_k, I_k) | `p̄ = p_all` (**valid-only**) |
| PSS Jacobian factors | joint `q·p_G` (**valid-only**) |
| Shifted **BSDF-sampled** segments (`ω'` at y_{k−1}; `ω_k` at x_k) | `f_ℓ · 1[supp]` (extends gap-bsdf §8.3 to transmission) |
| χ² tests | valid-only density (Cycles' eval pdf integrates to > 1) |

**Why valid-only p2 ("support-consistent MIS") is required** [INFERENCE, proof sketch]:
- In a spurious configuration only NEE can generate the path.
- With `p2 = p_G^C > 0`, the NEE weight is `M p1/(M p1 + p2) < 1`. The total weight is then < 1 and depends on M, the heuristic and the light pdf.
- Consequences:
  - our PT (M = 1) and our ReSTIR initial sampling (M = RIS-NEE count) would have **different expectations**, and Gate 3 fails;
  - Cycles' own value (power heuristic, light-tree pdf) is unreproducible anyway.
- With valid-only `p2 = 0` there, `ω1 = 1`, and the MIS is a partition of unity for every heuristic and every M.
- The same rule also resolves gap-bsdf §8.2 (smooth-shading Ng rejection) inside our renderer.

### 5.3 Parity consequence and the validation rule [INFERENCE]

| Emitter NEE-reached through a rough-transmission spurious configuration | Cycles | Ours | Parity |
|---|---|---|---|
| Mode-A analytic light (MIS off) | `f_spur · 1` | `f_spur · 1` | **exact** |
| Emissive mesh with `emission_sampling = 'NONE'` | 0 (no NEE) | 0 | **exact** |
| Mode-B area light or FRONT/BACK emissive mesh | `w_C · f_spur`, `w_C < 1` (power heuristic, light-tree pdf) | `f_spur` | approximate: excess `(1−w_C)·f_spur` |

**Validation rule.** Rough-glass scenes use only mode-A analytic lights and/or `emission_sampling = 'NONE'` emissive meshes (for example the furnace enclosure). Mode-B + rough glass goes to a separate "model-approximate" tolerance tier.

How much can the spurious region add? It is bounded by the "spurious eval" column of §2.3: about 0.1 % at r = 0.2, a few % at r = 0.5, and up to 60 % of the lobe energy at r = 1 grazing. It only matters where the light direction falls in the spurious set.

### 5.4 Roughness and delta for the predicate

```
r(G)     = sqrt(sqrt(α_x α_y)) = Roughness input (Principled shared roughness; Glass/Refraction node Roughness)
delta(G, R) = α² ≤ 2e-10
delta(G, T) = α² ≤ 2e-10  ∨  |η_side − 1| < 1e-4
r(event) = 0 if delta(event) else r(G)
r(NEE)   = max over allocated, non-delta lobes of {D:1, S:r_S, G:r_G}      (GRIS §7.5 "at least one lobe rough")
diffuseOnly(x) = hasD ∧ ¬hasS ∧ ¬hasG                                        (never true with glass)
```

`α_min = 0.2` (perceptual) is unchanged. The footprint tests take care of the narrower refraction lobes, because p̄ for T events is large. No special threshold is needed [INFERENCE].

### 5.5 Reservoir / pathFlags (extends gap-light §2.2, gap-bsdf §7.3)

- `ℓ_{k−1}`, `ℓ_k`: **3 bits each** (bits 22–24, 25–27).
- `side_{k−1}`, `side_k`: 1 bit each (bits 28–29; 0 = R, 1 = T). Used for debug views, the T3 invertibility signature and the optional side restriction. The maths recomputes side from geometry.
- Bits 30–31 stay reserved (frame parity).
- The `isDelta` bit stays **per event**. Never infer delta from `pdf == 0`.

### 5.6 RNG

- Every bounce consumes 4 dims `(u_lobe, u_h1, u_h2, u_rt)`, including delta and TIR bounces and non-glass materials (fixed layout; plan §1 "fixed per-vertex dimension layout").
- Cycles instead reuses the rescaled `u_lobe` for the R/T decision (`surface_shader.h:425-438`). Both are measure-preserving. A separate dimension avoids rescale round-off at class boundaries under relaxed Metal math.

---

## 6. Hybrid shift across glass

### 6.1 Random replay through glass [INFERENCE from GRIS §6.6 Eq. 53–54 and gap-rc]

- **Valid for every glass event, with J = 1 in PSS.** The replayed vertex uses the **offset's own** incident direction:
  - its own backfacing, so its own η_side;
  - its own Fresnel, P_R and TIR;
  - its own lobe pmf `q(·|V^y)`.
- Replay may therefore change R↔T, hit TIR (forcing R) or pick a different class. None of these makes the shift undefined, because random replay is the identity on the random numbers.
- **"TIR ⇒ undefined" belongs to half-vector copy** (GRIS §7.1), which we do not use.
- Requirements:
  - one shared `bsdf_sample` for initial sampling and replay;
  - fixed RNG dims;
  - V computed from reconstructed positions in both base and replay (gap-rc §3.1 same-formula principle);
  - backfacing computed from that same V.
- Under relaxed Metal math, `u_rt ≈ P_R` or near-critical-angle cases can flip R/T between pipelines. That is measure-small; classify it as an FP-boundary flip (gap-rc §8), not a logic violation.

### 6.2 Which vertices can take part in a passing pair `P_k` (gap-rc §3.3 with §5.4 above)

| Vertex role | Smooth (delta) glass | Rough glass |
|---|---|---|
| `x_{k−1}` (event ℓ_{k−1} = G) | never (D_k fails) | allowed if `r_G ≥ 0.2`, the event is non-delta (T with η≈1 is delta), and F_k passes with `p̄_{k−1}` valid-only |
| `x_k`, path continues (ℓ_k = G) | never (D fails for e_b) | allowed. I_k is always evaluated (not diffuse-only), with `p̄_k(ω_k | from y_{k−1})` valid-only |
| `x_k`, NEE-final (ℓ_k = NEE) | – (no NEE at delta) | allowed. I_k uses the valid-only `p̄` of the NEE direction; in a spurious NEE direction `p̄ = 0`, so the pair fails and forced light reconnection applies |
| Emitter after a delta vertex | – | – (the pair (x_{d−1}, x_d) fails D; BSDF-emitter candidate ⇒ k* = ∅, full replay) |

Typical outcomes:
- **Camera → δ → δ → diffuse x3 → NEE light:** no passing pair, so k* = d (forced light reconnection). Replay goes through the glass, then reconnects y3 → light. Reuse is good for coherent refraction.
- **Diffuse x1 → rough glass x2 → rough glass x3 → floor x4:** pairs (x2, x3) and (x3, x4) may pass. Reconnecting across the glass interior (y2 → x3) is legal. The visibility ray runs inside the glass, and glass faces in between occlude.

### 6.3 Reconnection at or into glass vertices

- **Jacobian** (plan §2 rule 4; gap-bsdf §7.2):
  `J = [q(G|V^y_{k−1})·p_G(ω'|V^y_{k−1}) · G(y_{k−1}→x_k) · q(ℓ_k|V^y_k)·p_{ℓ_k}(ω_k|V^y_k)] / [same for x]`
  - solid-angle pdfs, valid-only, each evaluated with **that vertex's own V and η_side**;
  - `G(a→b) = |n_g(b)·(a−b)|/‖a−b‖³` (absolute value: arrival may be from either side);
  - **no η² factor anywhere**. Solid angle is the same measure on both sides of an interface; η enters only inside p_G.
- **Side flips at x_k.** If `y_{k−1}` lies on the other side of x_k's surface than `x_{k−1}`:
  - x_k's backfacing flips, so η_side at x_k inverts and N flips;
  - the fixed `ω_k` changes sub-event (T↔R) relative to the flipped N.
  This is still a valid bijection: x_k and ω_k are copied, and f and p are re-evaluated with the true geometry. The cached suffix (after x_k) is unchanged, because its geometry is unchanged. If `(V^y, ω_k)` falls outside the G support, then `f·1[supp] = 0`: a zero contribution, not undefined.
- **TIR at the reconnection half-vector** gives `T = 0`, so `f = 0` (a zero contribution).
- **Radiance convention.** `f_G` is Cycles' (η² in the BTDF numerator for the V-side-relative η; no radiance scaling) in the PT, in initial sampling, and in every shift re-evaluation. Any 1/η² factor anywhere is a bias. A consistent one everywhere would still mismatch Cycles (planted-bias test B-η in §7.3).

### 6.4 What does not change

- Suffix caching, forced NEE reconnection, and the light-vertex Jacobian (1 spatially; pmf ratio temporally).
- Light-local endpoints and suffix refresh. Refresh shadow rays are blocked by glass, exactly as NEE is.
- Paired acceptance, the MIS formulas, temporal E_{t−1}.
- No per-path medium state is needed. The only exception is volume absorption (§8.3), deferred.

### 6.5 Reference-code notes

- **Falcor** keeps delta/transmission bits before and after the rc vertex, plus nested-dielectric handling, `insideDielectricVolume`, and `LDeltaDirect` for transmission/delta at x1, because its ReSTIR DI cannot do those ([REF] §1.3, §1.5.2).
  - We need **none** of these. The unified Enhanced reservoir handles d ≥ 2 through delta vertices.
  - Cycles has no nested-dielectric logic, so porting Falcor's would break parity.
- **EvanLuo42's** single-lobe-only reconnection (gap-bsdf §4.4) would forbid reconnection at any Principled vertex with t > 0. Do not copy it.

---

## 7. Validation scenes and tests

### 7.1 Blender setup (add to `cycles_settings.py` / `build_scene.py`)

```python
# Glass node (Tier 1)
g = nt.nodes.new('ShaderNodeBsdfGlass'); g.distribution = 'GGX'          # default MULTI_GGX!
g.inputs['Color'].default_value = (*c, 1); g.inputs['Roughness'].default_value = r; g.inputs['IOR'].default_value = ior
g.inputs['Thin Film Thickness'].default_value = 0.0
# Refraction node (optional)
f = nt.nodes.new('ShaderNodeBsdfRefraction'); f.distribution = 'GGX'     # default BECKMANN!
# Principled glass (Tier 1)
p.distribution = 'GGX'; p.inputs['Transmission Weight'].default_value = t
p.inputs['Base Color'].default_value = (*C, 1)        # transmission tint = sqrt(min(C,1)) per interface
p.inputs['Specular IOR Level'].default_value = 0.5; p.inputs['Specular Tint'].default_value = (1,1,1,1)
for s in ('Coat Weight','Sheen Weight','Subsurface Weight','Thin Film Thickness','Anisotropic','Diffuse Roughness','Emission Strength'):
    p.inputs[s].default_value = 0.0
p.inputs['Alpha'].default_value = 1.0
m.cycles.use_bump_map_correction = False
# enclosure emitter used as a furnace: BSDF-only in both renderers
enc_mat.cycles.emission_sampling = 'NONE'
# scene: all per-type caps = max_bounces (already), caustics_* True (default), blur_glossy 0 (already)
assert not any(o.cycles.is_caustics_caster or o.cycles.is_caustics_receiver for o in scene.objects)
assert not any(l.cycles.is_caustics_light for l in bpy.data.lights)
```

Two requirements on our side:
- **`emission_sampling = 'NONE'` support** (currently implied only as "noise-only" in cycles-conventions §3): exclude the triangles from the alias table and give forward hits MIS weight 1. It becomes necessary for exact rough-glass parity (§5.3).
- **Flat-shaded glass meshes** first. With smooth shading the Ng-rejection non-partition of gap-bsdf §8.2 also applies to the refraction sub-event.

### 7.2 Scenes (Gate 2 (vi) "glass and mirror", expanded)

| ID | Scene | Expected value | What it tests |
|---|---|---|---|
| **G1** | Smooth glass slab (e.g. 20×20×0.02 m box, flat) in an L_e = 1 enclosure (`emission_sampling = 'NONE'`), camera at 0–80° incidence, `max_bounces = N` ∈ {0,1,2,4} | **Closed form per ray**: Principled (grey C): `E = F + (1−F)·C·(1−F^N)`; Glass node (colour c): `E = cF + c²(1−F)²(1−(cF)^N)/(1−cF)`; with F = F_diel(cos θ_i, ior). [VERIFIED vs MC emulation of Cycles' sampling, e.g. θ = 80°, C = 1, N = 1: MC 0.76299 vs 0.76261; θ = 80°, C = 0.5, N = 2: 0.64782 vs 0.64783; θ = 45°, η = 1.33, C = 0.25, N = 3: 0.27067 vs 0.27064]. Average over the pixel footprint (16² sub-samples); mask pixels within `(N+1)·h·tan θ_t` of the slab edges. | Fresnel, P_R/weights, √C vs C tint, singular MIS skip, bounce semantics of transmission, η inversion on backfaces (N ≥ 1) |
| **G2** | Emissive quad (L_e = 1, two-sided) **inside** a smooth glass box, black world | `E = (1−F(θ_i))·√C·L_e` (Principled) or `(1−F)·c` (Glass node), for pixels whose refracted ray hits the quad | **η² convention detector**: a 1/η² bug gives 0.444× |
| **G3** | Rough glass slab (r ∈ {0.1, 0.25, 0.5, 1}), furnace enclosure, `max_bounces = 0` | Semi-analytic `E_R(r, μ, η)` = reflection part of the glass albedo (§3), by quadrature with the numpy reference | Reflection sub-lobe, P_R per microfacet, rough eval/sample consistency |
| **G4** | Same slab, `max_bounces` ∈ {1,2,3} | Independent CPU MC of the infinite rough slab (numpy port of §2) | Rough refraction, inside η = 1/ior, TIR, energy loss (Tier 1 loses energy; do not "fix" it) |
| **G5** | Diffuse floor + point light (r = 0) behind a pane: (a) smooth closed slab, (b) rough closed slab, (c) rough single-quad pane; **mode A** | (a) **exactly 0** in the pane's shadow; (b), (c) statistical parity | Glass blocks shadow rays; no NEE at delta vertices; exit-face and pane NEE through transmission (spurious counted with weight 1 in both) |
| **G6** | Area light (MIS on, **mode B**, `transparent_max_bounces = 1024`) behind a smooth glass sphere (flat icosphere) over a floor | Statistical parity with Cycles MIS-on | Caustics through delta glass via BSDF-only paths |
| **G6-neg** | G6 rendered in mode A | Predicted A < B (caustic missing) | Negative control for the corrected A≡B claim |
| **G7** | Principled mixtures: t ∈ {0.25, 0.5}, m ∈ {0, 0.25}, C coloured, L ∈ {0.25, 0.5}, T_s coloured; `max_bounces = 0` (semi-analytic albedo sum) and N = 4 (parity) | §2.1 weights | Closure order, weight products, `f0·T_s` in the glass Fresnel, η' not used by glass, q(·) |
| **G8** | Cornell box + flat rough glass icosphere (r = 0.3) + smooth glass cube; mode-A lights; ceiling = analytic area light or emissive mesh with `emission_sampling = 'NONE'`; `max_bounces = 8` | Parity (Gate 2), then the full Gate-3 ladder | End-to-end glass, replay-heavy ReSTIR |
| **G9** | "Bubble": Principled IOR 0.75 object in air | Parity | η_side < 1 from the front, TIR from outside |
| **G10** | Glass node coloured + Refraction node (GGX) | Parity | Colour in R and T; refraction TIR energy loss |

### 7.3 Unit and GPU tests (extend gap-bsdf §12)

- **U-G1 Fresnel.**
  - `F_diel(1, 1.5) = 0.04`;
  - symmetry `F_gs = F_diel` for `1/3.73 < η < 3.73` (1e-6), with a clip at η = 4 and 1/4;
  - TIR returns F = 1, and `cosT` is negative and correct;
  - the Refraction node gives T = 0 under TIR.
- **U-G2 Eval/pdf vectors.** The §3 table to 1e-5 relative, both the Cycles pdf and the valid flag.
- **U-G3 χ² of the G sampler** over the full sphere against the **valid-only** pdf:
  - grid: η ∈ {1.5, 1/1.5, 1.33, 2.4, 1.00005}, r ∈ {0.004, 0.01, 0.05, 0.2, 0.5, 1}, θ_V ∈ {0, 30, 60, 80, 89}°;
  - the rejected mass must equal `1 − ∫p_valid`;
  - also assert that the **Cycles** eval pdf integrates to > 1 in the spurious cases (the §2.3 table), so nobody "fixes" the test by switching pdfs.
- **U-G4 Albedo.** MC albedo equals `table_ggx_glass_E` at grid points for η > 1 (3σ + 2e-4). For η < 1, compare against the §3 table (valid-only MC), not `glass_inv_E` (§2.7 observation).
- **U-G5 Weight consistency.**
  - `weight == f_G/(q_G p_G)` on sampled directions;
  - the sample-form and eval-form `common` agree to 1e-4 relative;
  - delta weights are `(1−F)√C/(1−P_R)/q_G` and `F/P_R/q_G`.
- **U-G6 q(·).** The §2.6 vectors to 1e-4.
- **U-G7 Delta per side.** At r = 0.0037 vs 0.0038; T delta at |η−1| < 1e-4 with r = 0.3; no NEE at delta vertices; next-hit MIS weight 1.
- **U-G8 Backfacing / winding.**
  - η_side = 1/ior iff `Ng·V < 0`;
  - a negative-determinant glTF node flips winding at flatten time, so Ng matches Cycles' `object_negative_scale_applied` rule;
  - a one-triangle GLB spike renders identically in Blender.
- **U-G9 Tier 2.** Table checksums (§2.7); multipliers at the §2.7 grid; the darkening path for Fss ≠ 1.
- **U-G10 ggx_lambda with negative cosine.** `Λ(a2, −c) == Λ(a2, c)`. This is the regression test for the gap-bsdf WGSL fix.
- **ReSTIR tests.**
  - T2 replay determinism through delta chains (IDs, lobe codes, side bits);
  - T3 invertibility census on G5/G8 with R/T flips and TIR (zero logic violations);
  - T4 Jacobian reciprocity with a rough-glass x_{k−1} and x_k (including side-flip cases);
  - T6 MIS partition `|Σm − 1| < 1e-5`, including spurious NEE directions (ω1 must be exactly 1 there).
- **Planted biases** (harness calibration, Gate 1):
  - **B-η**: multiply refraction by 1/η². G2 detects it (0.444×).
  - **B-shadow**: let shadow rays pass through glass. G5a detects it (nonzero in the shadow).
  - **B-tint**: C instead of √C. G1 with C = 0.5 detects it.
  - **B-side**: do not invert η on backfaces. G1 with N ≥ 1, and G9, detect it.
  - **B-spur**: use the Cycles eval pdf for p2. ReSTIR(M = 8 RIS-NEE) vs PT(M = 1) on G5b must **fail**.

---

## 8. glTF / USD → our glass material (Blender-importer-equivalent)

### 8.1 glTF (importer 5.1.20) [SOURCE GLTF, verified this session]

| glTF | Blender Principled input | Our material |
|---|---|---|
| `KHR_materials_transmission.transmissionFactor × transmissionTexture.R` (linear) | Transmission Weight | `t` |
| `KHR_materials_ior.ior`, else 1.5 (`ior` set even without transmission) | IOR (no clamp other than Cycles' `max(ior, 1e-5)`) | η for the glass closure and the specular layer (η' via Specular IOR Level) |
| `baseColor` (× tex × COLOR_0) | Base Color | transmission tint `sqrt(min(C,1))` per interface; reflection untinted |
| `metallicRoughness` G / B | Roughness / Metallic | shared α for glass; glass weight `(1−m)t` |
| `KHR_materials_specular.specularFactor` | Specular IOR Level = 0.5·factor | affects **only** the dielectric specular layer (η'), not glass |
| `KHR_materials_specular.specularColorFactor` | Specular Tint | multiplies the glass `f0` (`saturate(F0(ior)·T_s)`), so it changes the R/T split; also the metal F82 tint |
| `KHR_materials_volume`, **only if `thicknessFactor ≠ 0`** (or animated) | Volume Absorption: Color = `attenuationColor`, Density = `1/attenuationDistance` (∞ → 0); `thickness` goes to the glTF Settings group, which Cycles ignores | homogeneous absorption **σ_a = (1 − attenuationColor)·/attenuationDistance** [SOURCE `closure.h:1239-1243`: absorption = (1 − colour)·density; mesh `volume_density = 1`]. **This is not the glTF spec's `−ln(c)/d`** |
| (no thin-walled mode) | — | **Always refract with η at every interface**, even for single-quad "windows" (glTF would treat them as thin-walled). Follow Blender. |
| distribution | **not set ⇒ MULTI_GGX** | imported glass defaults to Tier 2 (§2.7). Validation forces GGX. |
| `KHR_materials_dispersion`, `_diffuse_transmission` | not imported | ignore (warn) |

### 8.2 USD (Blender 5.1.2 importer) [SOURCE USD `usd_reader_material.cc:614-642, 735-737`]

UsdPreviewSurface `opacity` handling:
- **`opacityThreshold == 0`, not connected to a texture's `a` output:** Transmission Weight = **1 − opacity** (constant or through a one-minus node). A semi-transparent USD surface therefore becomes **glass** in Blender.
- **Otherwise:** it goes to Alpha (with a threshold node).

`ior` maps to IOR. Our `UsdSceneSource` adapter (M7) must do the same. This is new relative to scene-io, which listed only the exporter direction.

### 8.3 Volume absorption (deferred; phase G2) [INFERENCE]

**Phase G1: no volumes.**
- Warn when a glTF has `thicknessFactor ≠ 0` with a finite attenuation distance, because parity breaks.
- Validation scenes use no volume.

**Phase G2: homogeneous absorption with a local rule, for closed, non-overlapping, consistently wound meshes.**
- A segment is inside medium O iff it leaves `y` through O's surface into O's interior (backfacing side of O) and arrives at the next vertex on O's backfacing side. Cycles' volume stack gives the same result for such meshes.
- **ReSTIR:** the new reconnection segment `y_{k−1} → x_k` gets `exp(−σ_a·d)` if that rule says it is inside O. Replayed segments get it naturally. The cached suffix already contains its own absorption, since its geometry is unchanged.
- Inconsistent classifications (the two ends disagree, e.g. nested or overlapping meshes) ⇒ shift failure. This is symmetric.

---

## 9. Normative WGSL additions (extends gap-bsdf §11)

**Fix first:** `fn ggx_lambda(a2, cosN) { let c = max(abs(cosN), 1e-7); … }`.

```wgsl
const LOBE_D: u32 = 0u; const LOBE_S: u32 = 1u; const LOBE_G: u32 = 2u; const LOBE_TRANSP: u32 = 3u;
const LOBE_NEE: u32 = 4u; const LOBE_NONE: u32 = 5u;          // NOTE: renumbered from gap-bsdf (NEE 2→4, NONE 3→5)
const ETA_SINGULAR_EPS: f32 = 1e-4;                            // bsdf_microfacet.h:771
const LUT_GLASS_E: u32 = 9248u;  const LUT_GLASS_EAVG: u32 = 13344u;
const LUT_GLASS_INV_E: u32 = 13600u; const LUT_GLASS_INV_EAVG: u32 = 17696u;   // 9248 + 4096 + 256 + 4096

// BsdfCtx additions
//   back: bool; hasG: bool; hasRG: bool; fresnelNone: bool;
//   wG: vec3f; etaG: f32 (η as seen from V); F0rG: f32 (= F0(ior)); f0G: vec3f; rtG: vec3f; ttG: vec3f;
//   mulG: vec3f (Tier-2 eval multiplier, 1 in Tier 1); deltaG: bool (α² ≤ 2e-10); deltaGT: bool (deltaG || |η−1|<1e-4); pG: f32

struct GlassFresnel { R: vec3f, T: vec3f, cosT: f32 };
fn glass_fresnel(c: BsdfCtx, cosHI: f32) -> GlassFresnel {        // bsdf_microfacet.h:226-357, bsdf_util.h:47-99
  var o: GlassFresnel;
  let eta = c.etaG;
  let g = eta * eta - (1.0 - cosHI * cosHI);
  var Freal = 1.0;
  if (g > 0.0) {
    let ci = abs(cosHI); let ct = -sqrt(g) / eta; o.cosT = ct;
    let rs = (ci + eta * ct) / (ci - eta * ct); let rp = (ct + eta * ci) / (eta * ci - ct);
    Freal = 0.5 * (rs * rs + rp * rp);
  }
  if (c.fresnelNone) { o.T = select(vec3f(0.0), vec3f(1.0), g > 0.0); return o; }   // Refraction node
  let s = saturate((Freal - c.F0rG) / (1.0 - c.F0rG));
  let F = mix(c.f0G, vec3f(1.0), s);
  o.R = F * c.rtG; o.T = (vec3f(1.0) - F) * c.ttG;
  return o;
}

struct GEval { f: vec3f, pdfC: f32, pdfV: f32, isT: bool, valid: bool };
fn eval_G(c: BsdfCtx, V: vec3f, L: vec3f) -> GEval {             // bsdf_microfacet.h:586-675 (+ validity)
  var e: GEval;
  if (!c.hasG || c.deltaG) { return e; }
  let cNI = dot(c.Ns, V); let cNO = dot(c.Ns, L);
  if (cNI <= 0.0) { return e; }
  e.isT = cNO < 0.0;
  if (!e.isT && !c.hasRG) { return e; }
  let Hu = select(V + L, -(c.etaG * L + V), e.isT);
  let len = length(Hu); let invLen = select(0.0, 1.0 / len, len > 0.0);
  let H = Hu * invLen; let cHI = dot(H, V); let cNH = dot(c.Ns, H);
  let fr = glass_fresnel(c, cHI);
  let aR = avg3(fr.R); let aT = avg3(fr.T);
  if (!(aR + aT > 0.0)) { return e; }
  let lI = ggx_lambda(c.a2, cNI); let lO = ggx_lambda(c.a2, cNO);
  let k = c.etaG * invLen;
  let common = ggx_D(c.a2, cNH) / cNI * select(0.25, k * k * abs(cHI * dot(H, L)), e.isT);
  let pR = aR / (aR + aT);
  e.pdfC = common * select(pR, 1.0 - pR, e.isT) / (1.0 + lI);
  e.f = c.wG * c.mulG * select(fr.R, fr.T, e.isT) * (common / (1.0 + lI + lO));   // Cycles eval (spurious included)
  if (e.isT) {
    let Hn = H * select(-1.0, 1.0, cNH >= 0.0);
    e.valid = !c.deltaGT && dot(c.Ng, L) < 0.0 && dot(Hn, V) > 0.0 && dot(Hn, L) < 0.0 && aT > 0.0;
  } else {
    e.valid = dot(c.Ng, L) >= 0.0;                                // cNO >= 0 already
  }
  e.pdfV = select(0.0, e.pdfC, e.valid);                          // valid-only density
  return e;
}

// u = (u_h1, u_h2, u_rt); the class pick used u_lobe before this call
fn sample_G(c: BsdfCtx, V: vec3f, u: vec3f) -> BsdfSample {       // bsdf_microfacet.h:677-820
  var s: BsdfSample; s.lobe = LOBE_G;
  let cNI = dot(c.Ns, V);
  if (cNI <= 0.0) { return s; }
  var H = c.Ns; var cNH = 1.0;
  if (!c.deltaG) {
    let B = make_orthonormals(c.Ns);
    let Hl = ggx_sample_vndf_local(vec3f(dot(B[0], V), dot(B[1], V), cNI), c.a, c.a, u.xy);
    H = Hl.x * B[0] + Hl.y * B[1] + Hl.z * c.Ns; cNH = Hl.z;
  }
  let cHI = dot(H, V);
  let fr = glass_fresnel(c, cHI);
  let aR = avg3(fr.R); let aT = avg3(fr.T);
  if (!(aR + aT > 0.0)) { return s; }
  let pR = aR / (aR + aT);
  let refr = u.z >= pR;
  let inv = 1.0 / c.etaG;
  s.L = select(2.0 * cHI * H - V, (inv * cHI + fr.cosT) * H - inv * V, refr);
  if ((dot(c.Ng, s.L) < 0.0) != refr || (dot(c.Ns, s.L) < 0.0) != refr) { return s; }
  s.isT = refr;
  let ev = select(fr.R, fr.T, refr); let pl = select(pR, 1.0 - pR, refr);
  if (c.deltaG || (refr && abs(c.etaG - 1.0) < ETA_SINGULAR_EPS)) {
    s.isDelta = true; s.valid = true;
    s.weight = c.wG * c.mulG * ev / (pl * c.pG);                  // Cycles: F·1e6 / (pl·1e6·sw_G/Σsw)
    return s;                                                      // pJoint = pAll = 0 ⇒ next-hit MIS weight 1
  }
  let cNO = dot(c.Ns, s.L);
  let lI = ggx_lambda(c.a2, cNI); let lO = ggx_lambda(c.a2, cNO);
  let den = fr.cosT + cHI * inv;
  let common = ggx_D(c.a2, cNH) / cNI * select(0.25, abs(cHI * fr.cosT) / (den * den), refr);
  let pdfG = pl * common / (1.0 + lI);
  s.pJoint = c.pG * pdfG;
  s.weight = c.wG * c.mulG * ev * (common / (1.0 + lI + lO)) / s.pJoint;
  s.valid = pdfG > 0.0;
  // s.pAll = pD*p_D(L) + pS*p_S(L) + pG*pdfG   (valid-only; computed by the caller via eval_all)
  return s;
}
```

**Integration.**
- **`bsdf_prepare`.** Keep `back`. Build G per §2.1 (Principled: after metal, before the IOR level; Glass node; Refraction node with `hasRG = false`, `fresnelNone = true`, `sw = |avg(weight)|`). Compute `sw_G` per §2.6 and `mulG` per §2.7.
- **`bsdf_eval_all` (NEE).** `f += eval_G(...).f` (spurious included); `pAll += pG * eval_G(...).pdfV`.
- **`bsdf_eval_lobe(…, LOBE_G)` (shifts, BSDF-sampled segments).** `f = eval_G.f · f32(eval_G.valid)`, `pJoint = pG · pdfV`.
- **`bsdf_sample_support(LOBE_G)`** = `eval_G.valid`.
- **`lobe_roughness(LOBE_G)`** = `select(c.rough, 0, delta(side))`.
- **`is_diffuse_only`** = `hasD && !hasS && !hasG`.
- **Offsets.** A transmitted ray or shadow ray leaves on the side of its direction: offset along `sign(dot(Ng, L))·Ng`, plus self-primitive skip.

---

## 10. Plan edits (for the orchestrator)

1. **§1 Materials.** Add "Glass: lobe class G (§2, §5 of the glass gap report); `distribution='GGX'` forced on Principled, Glass and Refraction nodes; Tier-2 glass tables."
2. **§1 Lights.**
   - Replace "Modes A and B have identical expectation, which is itself a unit test" with "identical expectation iff no light-ending path has a singular lobe at x_{d−1}. Test on scenes without singular BSDFs; negative control G6-neg."
   - Recommend mode B for area lights in the interactive app when glass or mirrors are present (after the gates pass).
3. **§1 Path space.** RNG uses 4 dims per bounce.
4. **§1 Reservoir.** Lobe codes 3 bits (D, S, G, TRANSP, NEE, NONE); side bits for x_{k−1} and x_k.
5. **§2 new correctness rules.**
   - (15) Support-consistent MIS: every pdf in MIS/footprint/Jacobian is the true sampler density; shifted BSDF-sampled segments carry the support indicator; NEE uses Cycles' eval unchanged.
   - (16) No η² radiance scaling; the BTDF is exactly Cycles' (non-reciprocal); no η² in Jacobians.
   - (17) η_side from local backfacing; winding and negative scale identical to Blender; glass occludes every visibility query; no nested-dielectric logic.
   - (18) G delta is per sub-event (T also delta for |η−1| < 1e-4).
6. **§2 rule 6.** D_k and R_k extended to G (§5.4); diffuse-only excludes G.
7. **M2.**
   - `cycles_settings.py`: force GGX on Glass and Refraction nodes; assert MNEE off; support per-material `emission_sampling = 'NONE'` in the scene bridge.
   - `compare.py`: add a closed-form mode for G1/G2 and a quadrature/CPU-MC reference mode for G3/G4.
8. **M3.**
   - Glass after V2 Tier 1: WGSL §9; fix `ggx_lambda`; tests U-G1…U-G10.
   - Our renderer supports `emission_sampling = 'NONE'`.
   - Tier-2 glass tables with Tier 2 (M7).
9. **Gate 2 (vi).** Scenes G1–G10 (§7.2). **Gate 1.** Planted biases B-η, B-shadow, B-tint, B-side, B-spur.
10. **Gate 3.** Add G5b and G8 to the ladder (steps 1–4 and 7), plus B-spur as a negative control.
11. **Debug views (§6).** G-buffer: backfacing/η_side. Path tracer: lobe class incl. G, R/T side, TIR events, delta-chain length, spurious-NEE mask (NEE directions with valid = false).
12. **Loaders (M1/M7).** glTF transmission/ior/volume per §8.1 (volume: warn in phase G1); USD opacity → Transmission Weight = 1 − opacity per §8.2; flip winding for negative-determinant transforms.
13. **Interactive defaults.**
    - With the total-cap-only rule, glass interfaces consume `max_bounces`. For glass scenes use N ≥ 4.
    - Cycles-style per-type caps (total 8, diffuse 2, glossy 2, transmission 8) are representable. For ReSTIR they need the per-type counts of the cached suffix stored and re-checked in shifts, because the side at y_{k−1} and x_k can flip glossy ↔ transmission. Defer (M8+).
14. **Denoiser note (M8).** Demodulation albedo and normal guides should follow delta chains to the first non-delta vertex. Otherwise content behind glass is blurred. This is a quality issue, not correctness.

---

## 11. Risks and open questions

1. **Mode-B rough-glass parity is approximate** (spurious region × (1 − w_C)). Mitigated by the validation rule in §5.3 and a separate tolerance tier. The size depends on scene geometry. [UNVERIFIED] Measure it once on G5b rendered in mode B.
2. **`glass_inv_E` does not reproduce the current sampler** in the TIR regime (§2.7). Harmless if ported verbatim. The cause is [UNVERIFIED] (likely a sampler change after table generation).
3. **FP-boundary flips** at `u_rt ≈ P_R` and near the critical angle, between pipelines under relaxed Metal math. Measure-small; classify per gap-rc §8. Keep one shared `glass_fresnel`.
4. **Winding / negative scale.** A mismatch silently inverts η on whole objects. U-G8 plus a GLB spike.
5. **Smooth-shaded glass** (common in GLBs): the Ng-rejection non-partition applies to refraction too. Use flat shading for exact parity and a tolerance tier for smooth shading.
6. **Replay cost.** Delta glass forces replay (k* = ∅ or late rc). Glass-heavy views raise replay-queue occupancy. The perf budget (gap-perf) assumed mostly rough scenes. [UNVERIFIED] Re-measure on G8.
7. **IOR ≈ 1 rough glass** (some glTF "thin" hacks use ior = 1): Cycles samples it as delta but evaluates a rough spurious lobe in NEE. Replicated as specified, but avoid it in validation.
8. **glTF `ior = 0`** (spec special value) becomes Blender IOR 0, then Cycles `max(ior, 1e-5)`: a degenerate glass. Mirror Blender and warn. [UNVERIFIED] Blender's handling beyond the importer.
9. **Volume absorption** is deferred. Phase G1 mismatches Blender on GLBs with `thicknessFactor ≠ 0` and a finite attenuation distance.
10. **Open question.** Should the optional side-equality restriction (§5.1) be on by default for variance? Decide empirically in M4/M6 (unbiased either way).

---

## 12. Source index (5.1.2)

- `closure/bsdf_microfacet.h`:
  - VNDF 189–220; `microfacet_fresnel` 226–357; `preserve_energy` 359–410; `estimate_albedo` 423–499;
  - Λ/G/D 506–578; eval flag 581–584; **eval 586–675 (TODO 621–623)**; **sample 677–820**;
  - Fresnel setups 825–946; `ggx_setup` / `refraction_setup` / `glass_setup` 961–995; `ggx_eval` / `ggx_sample` 1005–1031.
- `closure/bsdf_util.h`: `fresnel_dielectric_polarized` 47–91, `fresnel_dielectric` 94–100, `refract_angle` 104–110, `fresnel_dielectric_Fss` 131–137.
- `svm/closure.h`: Principled 98–529 (transmission 377–415, IOR level 417–426, specular 428–462); Refraction 745–777; Glass 779–832; volume absorption 1208–1252.
- `svm/types.h`: 511–546 (`CLOSURE_IS_REFRACTION/GLASS`, `CLOSURE_WEIGHT_CUTOFF = 1e-5`, `BSDF_ROUGHNESS_SQ_THRESH = 2e-10`, `THINFILM_THICKNESS_CUTOFF = 0.1`).
- `closure/alloc.h`: 55–84.
- `closure/bsdf.h`: `bsdf_sample` 144–295 (no bump term for transmission), `bsdf_eval` 512–615, `bsdf_albedo` 648–681.
- `integrator/path_state.h`: 110–231.
- `integrator/shade_surface.h`: ray offset 44–95, NEE 305–460, bounce 470–600.
- `integrator/surface_shader.h`: `_bsdf_eval_mis` 269–306, closure pick 408–448, `sample_closure` 891–923, transparency 948–966.
- `integrator/shade_shadow.h`: 18–70.
- `bvh/intersect_filter.h`: 130–185.
- `light/light.h`: 258–300.
- `geom/triangle.h`: 30–43.
- `geom/shader_data.h`: 100–120, 220–235.
- Host (raw.githubusercontent, v5.1.2):
  - `scene/shader.cpp` 311–324, 600–660, 702–710;
  - `scene/shader_nodes.{h,cpp}` (`has_surface_transparent`: Transparent/RayPortal true, Principled Alpha only);
  - `scene/shader.tables` (glass tables at lines 50, 324, 343, 617);
  - `app/cycles_precompute.cpp` (glass E/Eavg definitions, `ior_parametrization`);
  - `scene/object.cpp` 494–540.
- glTF importer 5.1.20: `blender/imp/pbrMetallicRoughness.py` 15–135 (volume node, IOR), 218–264 (transmission), 266–313 (volume).
- USD importer: `io/usd/intern/usd_reader_material.cc` 614–642 (opacity → Transmission Weight, inverted), 735–737.
- Blender RNA [BPY]:
  - Glass `distribution` MULTI_GGX (BECKMANN/GGX/MULTI_GGX); Refraction BECKMANN (BECKMANN/GGX); Principled MULTI_GGX (GGX/MULTI_GGX);
  - Principled inputs (no Transmission Roughness; Thin Film 0 / 1.33); Volume Absorption Color 0.8, Density 1;
  - bounces 12/4/4/12/0/8; caustics True/True; `blur_glossy` 1.0;
  - object `cycles.is_caustics_caster/receiver` False; light `cycles.is_caustics_light` False; `film_transparent_glass` False.

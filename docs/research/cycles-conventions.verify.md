# Adversarial verification: "Blender 5.1 Cycles conventions" report

**Report under review:** `/Users/mark.boss/.claude/plans/do-a-deep-dive-shimmering-ritchie-agent-a2d6cc1a462b20a80.md` (topic `cycles-conventions`)

> **Where this file is.** Plan mode was active, so the only file I could write was this plan file. The requested path `…/scratchpad/research/cycles-conventions.verify.md` was **not** written. Copy this file there if the pipeline needs it.

---

## 0. Method and primary sources

I did not trust the report. I re-read every load-bearing claim against these sources.

1. **Cycles kernel source that ships inside the installed Blender 5.1.2.** The path is `/Applications/Blender.app/Contents/Resources/5.1/scripts/addons_core/cycles/source/kernel/...` (271 files, used for Metal runtime compilation).
   - This is the exact kernel the reference renders will run.
   - All `kernel/...` line numbers below refer to this tree.
2. **Host-side Cycles and Blender sources at tag `v5.1.2`**, streamed read-only from `raw.githubusercontent.com/blender/blender/v5.1.2/...` to stdout. Nothing was saved. Files read:
   - `intern/cycles/scene/{light,integrator,film,shader,camera,object}.cpp`, `scene/shader.tables`
   - `intern/cycles/blender/{light,sync,camera,shader,object,session}.cpp`
   - `intern/cycles/util/{math_cdf,transform}.cpp`, `app/cycles_precompute.cpp`
   - `source/blender/io/usd/intern/usd_reader_light.cc`, `makesrna/intern/rna_light.cc`
3. **The installed glTF importer** at `/Applications/Blender.app/Contents/Resources/5.1/scripts/addons_core/io_scene_gltf2/`.
4. **Headless, read-only bpy introspection** with `Blender --background --factory-startup --python-expr …`. Nothing was saved and nothing was rendered.
5. **Independent numeric checks** in plain `python3` on stdin/stdout:
   - the LUT test vectors, re-computed from `shader.tables` using Cycles' exact `lookup_table_read_*` logic;
   - the Fresnel monotonicity;
   - the t-distribution tail probabilities.

Tags: **[VERIFIED]** means checked against code or execution by me. **[INFERENCE]** means my derivation. **[UNVERIFIED]** means I could not check it.

---

## 1. Verdict

**The report is accurate on everything that decides whether the renderer is biased or matches Cycles.** I checked each of these against the code and they are correct:

- light normalisation (point, spot, area, sun, mesh emission);
- the MIS structure (power heuristic, mixture BSDF pdf, delta and MIS-off lights);
- the `max_bounces` path-length semantics;
- the GGX D/Λ/G2/eval/pdf formulas and the Fresnel models (dielectric, generalized Schlick with `exponent<0`, F82-tint);
- the Principled lobe and layering structure, and the LUT interpolation;
- Russian roulette;
- the filter, raster convention and camera FOV formulas;
- the glTF light and camera conversions;
- the render-setting defaults.

I re-computed the report's LUT test vectors from the actual table and they match to 4 decimals. The same is true of `ggx_E(0.5,1) = 0.91528` and `ggx_Eavg(0.5) = 0.88204`.

The errors I found are listed below. **One of them (C1, the camera `angle_x` example) is labelled "verified" but is misleading.** If it were used literally, every comparison image would be wrong.

The rest are wrong line or file citations, imprecise wording, and missing detail. None of them creates a silent bias in the recommended validation subset (Tier 0/1, flat shading, r = 0 lights, MIS mode A).

Overall verdict: **minor_issues**, with one item of **major** severity (C1).

---

## 2. Corrections

Severity:
- **critical**: silently biases the renderer or makes it mismatch Cycles.
- **major**: produces a visibly wrong reference or setup if followed.
- **minor**: wording, citation, or edge case.

### C1 [MAJOR, high confidence]: §6 camera example misstates the horizontal FOV

**Report says:** "Setting an exact vertical FOV: `cam.sensor_fit = 'VERTICAL'` and `cam.angle_y = vfov` (verified: `angle_y = 40°` with `sensor_height = 24` gives `lens = 32.9697` and `angle_x = 57.265°` at 16:9)."

**Problem.** The RNA property `Camera.angle_x` is always `2·atan(sensor_width/(2·lens))`. It does not depend on `sensor_fit` or on the render aspect ratio. 57.265° is therefore **not** the horizontal FOV of the 16:9 render.

With VERTICAL fit, Cycles uses:
- `aspectratio = H/W`
- viewplane x ∈ [−1, 1], y ∈ [−H/W, H/W]
- `fov = 2·atan(0.5·sensor_height/lens/aspectratio)` (blender/camera.cpp:363-372 and :615)

So the rendered FOVs are:
- `tan(hfov/2) = tan(vfov/2)·W/H` gives **hfov = 65.81°** for vfov = 40° at 1920×1080;
- vfov = 40.00°.

An implementer who reads `cam.angle_x` (or this sentence) as the horizontal FOV gets a camera about 15 % too narrow horizontally.

**Evidence** [VERIFIED by execution]: `Object.calc_matrix_camera(depsgraph, x=1920, y=1080)` after `sensor_fit='VERTICAL'` and `angle_y=40°` gives:
- `P[0][0] = 1.545456`, so hfov = 65.8104°;
- `P[1][1] = 2.747477`, so vfov = 40.0000°;
- `cam.lens = 32.96973`, `cam.angle_x = 57.2651°`.

**Corrected version:**
```
Blender side:  cam.sensor_fit = 'VERTICAL';  cam.angle_y = vfov      (lens = sensor_height / (2·tan(vfov/2)))
WGSL side:     ty = tan(vfov/2);  tx = ty · (W/H)                      (hfov = 2·atan(tx))
Never use Camera.angle_x as the render's horizontal FOV unless sensor_fit = 'HORIZONTAL',
or sensor_fit = 'AUTO' with W > H.
Only in those two cases does angle_x = hfov (sensor_width drives it).
```

The general formulas in §6 ("Vertical fit: `tan(vfov/2) = sensor/(2·lens)`, `tan(hfov/2) = tan(vfov/2)·W/H`") are correct. Only the parenthetical example is wrong.

### C2 [minor, high]: §4.1, singular GGX case, "eval = pdf = F·1e6"

**Code** (`kernel/closure/bsdf_microfacet.h:766-781`):
```
*eval = reflectance (or transmittance);   *pdf = pdf_reflect (or 1 − pdf_reflect)
if (m_singular) { *pdf *= 1e6f; *eval *= 1e6f; }
```

**Corrected version:**
- `pdf = lobe_prob · 1e6`, where `lobe_prob = avg(R)/avg(R+T)`, which equals 1 for reflection-only closures;
- `eval = F · 1e6`.

They are equal only when F ≡ 1. The sample weight is `eval/pdf = F/lobe_prob`.

The singular threshold is:
- `alpha_x·alpha_y ≤ BSDF_ROUGHNESS_SQ_THRESH = 2e-10` (`kernel/svm/types.h:545`);
- equivalently, roughness ≤ (2e-10)^(1/4) = 0.003761.

The report states the threshold correctly.

### C3 [minor, high]: §5.3, Filter Glossy formula acts on α, not on roughness, and is conditional

**Report says:** `roughness = max(r, sqrt(1 − filter·min_ray_pdf)/2)`.

**Code** (`kernel/integrator/surface_shader.h:184-213`; `bsdf_microfacet.h:997-1003`):
```
blur_pdf = filter_glossy · min_ray_pdf            (filter_glossy = 1/blur_glossy; FLT_MAX if blur_glossy == 0)
if (blur_pdf < 1):  b = 0.5·sqrt(1 − blur_pdf);  alpha_x = max(b, alpha_x);  alpha_y = max(b, alpha_y)
```

`min_ray_pdf` starts at FLT_MAX, so there is no blur at the camera hit. It becomes the minimum BSDF pdf along the path. The blur is on **α**, not on roughness (r = √α).

The report's conclusion (biased, set it to 0) stands. The factory default is `blur_glossy = 1.0` [VERIFIED introspection].

### C4 [minor, high]: §4.4 closed form, "`F_d = F_diel(c, η)` exactly when Specular IOR Level = 0.5 and T_s = 1"

`F_d = mix(F0(η), 1, saturate((F_diel(c,η) − F0(η))/(1 − F0(η))))` reduces to `F_diel` only if `F_diel(c, η) ≥ F0(η)` for all c.

The unpolarised dielectric Fresnel dips below F0 for η > 2+√3 ≈ 3.73 (the Brewster dip dominates):
- numerically, the minimum of F − F0 is −1.6e-4 at η = 3.8 and −2.2e-3 at η = 4.0 [VERIFIED numerically];
- `saturate` then clips.

**Corrected version:** "exactly `F_diel(c, η)` for 1 < η ≤ 2+√3 (all practical IORs). For η > 3.73 port the `saturate` literally."

### C5 [minor, high]: §2.7, "lights with MIS disabled … are never hit"

**Code** (`kernel/light/light.h:278-286`, `lights_intersect_impl`):
- For **camera** rays only `SHADER_EXCLUDE_CAMERA` (object `visible_camera`) is tested. `SHADER_USE_MIS` is not.
- Camera rays start with `PATH_RAY_MIS_SKIP` (`kernel/integrator/path_state.h:58-59`), so the forward weight is 1.

**Corrected version:** a light with MIS off (or a delta light) is never hit by **BSDF/indirect** rays. An **area** light with MIS off and `visible_camera = True` is still visible to camera rays, and camera rays still pass through it. Mode (A) in §2.8 correctly sets both flags, so the recommendation itself is fine.

### C6 [minor, high]: §4.9 USD lights, unnormalised sun, and SphereLight geometry

1. **"`eval_fac` is then `1/π` without the area term" is right for point, spot and area lights but wrong for the sun.** For `LIGHT_DISTANT` the code has no 1/π (`scene/light.cpp`, device_update_lights):
   ```
   eval_fac = normalize ? 1/(π·sin²(angle/2)) : 1
   ```
   Consequences for an unnormalised sun:
   - disk radiance = `4·intensity` (energy = intensity·4, from `usd_reader_light.cc:137-139`);
   - perpendicular irradiance = `4·intensity·π·sin²(half)` for angle > 0;
   - for angle = 0, `Light::area` = 1, so normalisation does not matter and E = `4·intensity`.
2. **Omission.** A UsdLux `SphereLight` imports as a Blender POINT light (`usd_reader_light.cc:84-100`) with:
   - `radius` = the USD radius (UsdLux fallback 0.5 [UNVERIFIED fallback value]) unless `treatAsPoint`;
   - `use_soft_falloff` left at the Blender default **True** [VERIFIED default].

   Blender therefore renders it as the camera-facing **oriented disk** ("soft falloff", §2.2(b)), **not as a sphere**. Consequences:
   - irradiance is `L·πr²/(d²+r²)` instead of the sphere's `L·πr²/d²`;
   - a WGSL USD loader that implements a true sphere will not match.

**Corrected version:** for USD → Blender parity, either set `use_soft_falloff = False` after import in Blender, or implement the oriented-disk model in WGSL.

### C7 [minor, medium]: §5.5, default `sampling_pattern`

"The default `sampling_pattern` is TABULATED_SOBOL in 5.1.2 (introspected)" is true **only for the scene inside factory-startup**. That scene stores raw value 1 [VERIFIED: `scene.cycles.get('sampling_pattern') == 1`, also after `read_factory_settings(use_empty=True)`].

- The RNA default is `5` = AUTOMATIC (`addons_core/cycles/properties.py:524-529`).
- A scene created with `bpy.data.scenes.new()` gets AUTOMATIC [VERIFIED].
- In background renders, AUTOMATIC becomes `BLUE_NOISE_PURE` (`blender/sync.cpp:409-420`), and the seed is then hashed (`scene/integrator.cpp:304-313`).

This is noise structure only, not bias. Set `cycles.sampling_pattern` explicitly in the reference script for reproducibility. With the developer UI off, only AUTOMATIC and TABULATED_SOBOL/BLUE_NOISE_PURE survive (`sync.cpp:421-430`).

### C8 [minor, high]: §10, statistical threshold for the per-pixel z-test

"Check `P(|z| > 3) ≈ 0.27%`" assumes known variances. With σ² estimated from K = 16 independent renders, the statistic is Student-t with about 15 dof (Welch):
- P(|t₁₅| > 3) ≈ **0.90 %**;
- P(|t₃₀| > 3) ≈ 0.54 %;
- normal: 0.27 % [VERIFIED numerically].

**Corrected version:** compare against the t-distribution with the Welch–Satterthwaite dof, or use region-averaged means. The report also suggests region averaging.

### C9 [minor, high]: wrong citations

| Report citation | Correct location (v5.1.2) |
|---|---|
| `util/math_float.h:485-495` (`smoothstepf`) | `intern/cycles/util/math_base.h:485-495` |
| `blender/session.cpp:1006-1015` (`use_sample_subset`) | `intern/cycles/blender/sync.cpp:1006-1022`. `session.cpp` contains no `sample_subset` reference. |
| `kernel/light/sample.h … (line ~356)` for `light_sample_mis_weight_nee/forward` | `sample.h:322` (forward) and `:339` (nee). `_forward_surface` is at 464; `_forward_lamp` is at 508. |

The other spot-checked citations are correct within a few lines:
- `scene/light.cpp:151-176, 202-226, 1264-1283, 1329-1391`, `integrator.cpp:192-193/240/281-285/318-323`, `sync.cpp:585-590`, `camera.h:440-445`;
- `bsdf_util.h:94/131/140-190/332-341/370-458/488-492`, `bsdf.h:65-125`;
- `area.h:106-121/313/336/405`, `point.h:139/152-154`, `svm/types.h:543-545`;
- `scene/object.cpp:659-661`, `blender/object.cpp:288-290`, `scene/shader.cpp:236-278/709`, `blender/camera.cpp:334`, `rna_light.cc:358-359`.

### C10 [minor, low impact]: §2.2(a), NEE-only wording

"so pdf = t²" is correct: `invarea = 1` and `light_pdf_area_to_solid_angle(lightN, −D, t) = t²/1`. For completeness:
- the MNEE path uses `pdf = eval_fac·4π` (= 1 for r = 0), which is irrelevant here;
- `ls->Ng = −ls->D`, so the emission shader's `|Ng·wi| > 0` test always passes.

No change is needed. This entry records that it was verified.

---

## 3. Omissions

None of these creates bias in the recommended Tier-0/1 flat-shaded validation. They matter for later tiers and for matching noise.

1. **Microfacet `sample_weight` is scaled by the Fresnel albedo estimate.** `bsdf_alloc` sets `sample_weight = |average(weight)|` (`kernel/closure/alloc.h:64`). For microfacet closures, every Fresnel setup then multiplies it by `average(bsdf_microfacet_estimate_albedo(...))`, and MULTI_GGX also multiplies it by `average(darkening)` (`bsdf_microfacet.h:408, 833, 849, 872, 917, 942`).

   This changes:
   - the closure pick probability for BSDF sampling;
   - the mixture pdf `Σ sw_i·pdf_i / Σ sw_i` used in **both** MIS weights.

   It is noise only (expectation unchanged). The report defines `sw_i` as "closure sample_weight" without this detail, so a WGSL port that uses the plain average weight gets Cycles' expectation but different MIS weights.

2. **NEE BSDF evaluation skips closures whose pdf is 0.** `_surface_shader_bsdf_eval_mis` adds `eval·weight` only `if (bsdf_pdf != 0.0f)` (`surface_shader.h:325-331`). For all closures in the subset, eval and pdf vanish together, so this is harmless. Mirror it anyway.

3. **MIS is not a partition of unity under smooth shading.** For directions with `N·L > 0` but `Ng·L ≤ 0`:
   - evaluation accepts them (NEE weight `p_nee²/(p_nee²+p_bsdf²)` with the pdf computed from the evaluation);
   - every sampler rejects them and terminates the path (`LABEL_NONE`, or pdf = 0 for diffuse);
   - the BSDF-sampling side therefore never contributes there, and the combined weight is < 1.

   Such directions usually point into the object and are occluded, so the effect is small. Port this exact asymmetry or stay flat-shaded, as the report recommends.

4. **The Normal Map node math is not documented** (`kernel/svm/tex_coord.h:324-420`, tangent space). Needed for textured or glTF tiers:
   ```
   c = 2·(rgb − 0.5);  if DirectX: c.y = −c.y
   c.x *= s;  c.y *= s;  c.z = mix(1, c.z, saturate(s))            (s = Strength)
   n_ = unnormalized interpolated object-space normal (smooth) or Ng (flat, un-flipped)
   t_ = unnormalized interpolated tangent attribute (MikkTSpace from the UV map), sign = tangent sign attribute
   B  = sign·cross(n_, t_);  N = normalize(c.x·t_ + c.y·B + c.z·n_) → world;  if backfacing: N = −N
   ```
   - Differences from glTF: glTF scales only XY by `normalTexture.scale`. Cycles also mixes z toward 1 for s < 1.
   - Per-pixel details: the bitangent is computed per pixel from unnormalised vectors.

5. **Alpha, transparent BSDF and transparent shadows are not specified.** They matter for glTF MASK/BLEND materials.
   - `alpha < 1` adds `bsdf_transparent` with weight (1−α) (`svm/closure.h:242-246`).
   - Shadow rays through transparent surfaces are attenuated when `SD_HAS_TRANSPARENT_SHADOW`.
   - Transparent passes count toward `transparent_max_bounces` and are subject to transparent RR (`transparent_min_bounce = min_transparent_bounces + 1`, `integrator.cpp:200`).

6. **WebGPU porting detail for the LUTs** [INFERENCE].
   - `r32float` textures are not filterable in WebGPU unless the optional `float32-filterable` feature is enabled.
   - Hardware trilinear filtering also has limited sub-texel weight precision.
   - The report's advice to port `lookup_table_read_*` literally from a storage buffer is the right one. Use it rather than `textureSampleLevel`.

7. **The box-filter table has an unset last entry** [INFERENCE, negligible].
   - `util_cdf_invert(make_symmetric=true)` fills entries 0…1022 of the 1024-entry table (`half_size = 511`), so entry 1023 is not written by that loop.
   - With a linear box CDF, the tail interpolation over u ∈ (1022/1023, 1) spreads its 1/1023 of the mass uniformly over [0, 1] as well.
   - The pixel average stays uniform over the pixel. The report's "negligible" conclusion holds.

8. **Spot lights are one-sided for ray hits.** `spot_light_intersect` rejects rays with `dot(D, P − co) ≥ 0` (`kernel/light/spot.h`). This only matters for r > 0 spots in MIS mode (B).

9. **glTF import details worth recording:**
   - The spot cone depends on which fields are present (`blender/imp/light.py:103-110`):
     - if the `spot` object is present but `outerConeAngle` or `innerConeAngle` is missing, the glTF defaults apply (outer = π/4, inner = 0), giving `spot_size = 90°` and `spot_blend = 1`;
     - if the `spot` object is missing entirely, the importer leaves Blender's new-light defaults (45°, blend 0.15) [VERIFIED introspection of those defaults].
   - glTF `TANGENT` attributes are not used by Cycles (Blender computes MikkTSpace from the UVs).
   - UVs are flipped on import: v_blender = 1 − v_gltf [INFERENCE from the standard importer; not re-read].

---

## 4. Independent re-derivations (all agree with the report)

Notation:
- `P` = Blender power, `E_n` = light node emission (colour × strength), `e = 2^exposure`;
- estimator per NEE sample: `C = Le · eval_fac · f_cos(V,L) · w_nee / (p_sel · pdf_ω)` (`kernel/integrator/shade_surface.h:402-404`).

1. **Point light, r = 0, normalised.**
   - `eval_fac = (1/4)·(1/π)`, `pdf_ω = t²`.
   - `C = P·e·E_n/(4π) · f_r·cosθ / t² / p_sel`.
   - So I = P/(4π) and E_⊥ = I/t².
   - Analytic test: P = 4π, h = 1, Lambert ρ gives `L(θ) = ρ·cos³θ/(π h²)`. ✓
2. **Soft-falloff disk, r > 0.**
   - Radiance `L = P/(4π²r²)` (`eval_fac = 1/(4πr²·π)`).
   - On-axis irradiance of a facing uniform disk: `E = πL·r²/(d²+r²) = P/(4π(d²+r²))`. ✓
3. **Sphere, r > 0, `use_soft_falloff = False`.**
   - Same L.
   - Irradiance from a uniform sphere wholly above the horizon, facing: `E = πL·sin²α = πL·r²/d² = P/(4πd²)`. ✓
   - This is exact, which is why the sphere mode matches a point light outside the sphere.
4. **Area light.**
   - `L = P/(π A)`, one-sided.
   - Φ = L·π·A = P. ✓
   - With spread a = spread/2: `L(θ) = P/(πA) · (tan a − tan θ)/(tan a − a)` for θ < a.
   - Φ = A·2π·∫₀ᵃ L(θ) cosθ sinθ dθ. Using ∫₀ᵃ(tan a − tanθ)cosθ sinθ dθ = (tan a − a)/2, this gives Φ = P. ✓ (power preserved)
   - At spread = 180°, `normalize_spread = 1/(FLT_MAX − π/2) ≈ 2.94e-39` (denormal). Two cases:
     - on FTZ hardware it becomes 0 and the `> 0` gate skips the attenuation;
     - otherwise the factor is `(FLT_MAX − tanθ)·2.94e-39 ≈ 1`.

     Either way the factor is 1. ✓
5. **Sun.**
   - Normalised disk radiance `L = E/(π sin² h)`, with h = angle/2 (`Light::area`, `scene/light.cpp:219-223`).
   - Irradiance `E_⊥ = L·π sin² h = strength`. ✓
   - For angle = 0, pdf = 1 (`sample_uniform_cone` zero-angle branch, `kernel/sample/mapping.h:171-174`) and eval_fac = 1. ✓
6. **Mesh emission.**
   - `L = strength·color·[|Ng·wi| > 0]` (`closure/emissive.h:43-60`, `surface_shader.h:1102-1106`), two-sided.
   - An emissive quad emits Φ = π·A·S per side. ✓
7. **`max_bounces = N`.**
   - `kintegrator.max_bounce = N+1` (`integrator.cpp:193`).
   - `path_state_next` sets `TERMINATE_AFTER_TRANSPARENT` once the bounce count reaches N+1 (`path_state.h:141-145`).
   - The next hit keeps 0 closures (`surface_shader.h:1161-1163`), so there is emission but no NEE and no scattering.
   - NEE happens at x₁…x_{N+1}; the longest path has N+2 segments. ✓
8. **Russian roulette** (`path_state.h:271-299`, `intersect_closest.h:61-80`, `shade_surface.h:621-635`).
   - Continuation probability `q = min(√(max_c|β_c|), 1)`, applied only when `bounce > min_light_bounces + 1`.
   - The decision is made before shading, but emission is still collected on termination.
   - Survivors get `β /= q`. ✓
9. **LUT test vectors.** Re-computed from `table_ggx_gen_schlick_ior_s` at z = √((η−1)/(η+1)) = 0.4472, with S at μ = {1, 0.7, 0.4, 0.1}:
   - r = 0: S = {0, 0.0117, 0.0955, 0.5628}
   - r = 0.5: S = {0.0006, 0.0138, 0.0639, 0.1705}
   - r = 1: S = {0.0014, 0.0058, 0.0161, 0.0403}

   Identical to the report's table. ✓
10. **glTF light units.**
    - `P_point/spot = cd·4π/683` (`io_scene_gltf2/blender/imp/light.py:69-71`), so I = cd/683 W/sr.
    - Sun: `lux/683` (`:58-60`).
    - Spot: `spot_size = 2·outer`, `spot_blend = 1 − inner/outer` (`:138-143`).
    - `shadow_soft_size` is not set, and the new-light default is 0.0 [VERIFIED]. ✓
11. **glTF axes.**
    - `convert_loc(x) = (x₀, −x₂, x₁)` is R_x(+90°).
    - The camera/light correction quaternion (√2/2, √2/2, 0, 0) is also R_x(+90°).
    - M_b = C·M_g·C⁻¹·R_x(90°) = C·M_g. ✓ (the report's inference is correct)

---

## 5. Claim-by-claim status

### Render settings table (§1)

**Defaults** [VERIFIED by introspection]:

| Setting | Default |
|---|---|
| `render.engine` | `BLENDER_EEVEE` |
| `device` | `CPU` |
| `samples` | 4096 |
| adaptive sampling | True |
| denoising | True |
| `blur_glossy` | 1.0 |
| `sample_clamp_direct` / `sample_clamp_indirect` | 0 / 10 |
| `light_sampling_threshold` | 0.01 |
| `use_light_tree` | True |
| caustics (reflective / refractive) | True / True |
| Fast GI / guiding | False / False |
| pixel filter | BLACKMAN_HARRIS, width 1.5 |
| bounces (max / diffuse / glossy / transmission / volume / transparent) | 12 / 4 / 4 / 12 / 0 / 8 |
| `min_light_bounces` | 0 |
| `seed` / `use_animated_seed` | 0 / False |
| `film_exposure` | 1 |
| `scrambling_distance` | 1.0, auto off |
| `use_layer_samples` | USE |
| `texture_limit_render` | OFF |
| world colour | 0.050876 |
| view transform | AgX |
| image format | PNG 8-bit |
| camera | 50 mm, 36×24 sensor, AUTO fit, clip 0.1–100 |
| light objects `visible_camera` | False (other ray visibilities True) |
| `material.cycles.emission_sampling` | AUTO |
| `material.cycles.use_bump_map_correction` | True |
| `Object.shadow_terminator_geometry_offset` | 0.1 |
| `Object.shadow_terminator_shading_offset` | 0.0 |
| Principled and Glossy distribution | MULTI_GGX |
| new-light `shadow_soft_size` | 0.0 |
| new-light `use_soft_falloff` | True |
| new-light `normalize` | True |
| new-light `energy` | 10 |
| new-light node tree | Emission (1,1,1), strength 1 |
| `spot_size` range | [1°, 180°] |
| `exr_codec` enum | contains ZIP, DWAA, DWAB, PXR24, B44, B44A, HTJ2K, ZIPS, RLE, PIZ, NONE |
| EXR `color_depth` after switching format | '32' |
| EXR `linear_colorspace_settings.name` | 'Linear Rec.709' |

`Object.shadow_terminator_normal_offset` also exists (default 0). It is not read by the Cycles sync (`blender/object.cpp:288-290` reads only the shading and geometry offsets).

`read_factory_settings(use_empty=True)` gives `scene.world = None`. With no world, the final render gets an empty background graph, which is black (`blender/shader.cpp:1647-1676`, `scene/shader.cpp:800-810`). ✓

### Lights (§2)

- **§2.1** `strength = color·energy·exp2(exposure)` (× temperature colour) ✓. Normalise flag ✓. Sphere flag ✓. Area size mapping ✓. Implicit Emission (1, 1) when there is no node tree (`blender/shader.cpp:1825-1837`) ✓. The Cycles-internal `default_light` has colour 0.8 and strength 0, but Blender never uses it for lights. `has_contribution` ✓.
- **§2.2 / §2.3** ✓. The spot attenuation uses the inverse object transform and then negates z ✓. `smoothstepf` ✓ (see C9 for the file).
- **§2.4** ✓. Ellipse `invarea < 0` is only a flag (`fabsf` in eval) ✓. MIS for an area light additionally requires `spread > 0` (`scene/light.cpp`), which does not matter here.
- **§2.5** ✓. `co = −Z_obj`, and the NEE direction is `D = −Ng` from a cone around `co`, so the direction toward the sun is +Z_obj ✓.
- **§2.6 / §2.7** ✓, except C5. The power heuristic `a²/(a²+b²)` is in `kernel/sample/mis.h:26-29`. The BSDF pdf is forced to 0 when the light is not MIS (`surface_shader.h:379-381`).
- **§2.8** ✓. `shade_light.h`:
  - advances `tmin`;
  - `transparent_bounce++`;
  - terminates at `transparent_max_bounce`, which gets no +1 (`integrator.cpp:204`).

### Emission (§3)

✓ in full, including the AUTO emission-sampling rule (`scene/shader.cpp:236-278`: threshold 0.5, scale 0.1 for auto-converted Emission nodes).

### BSDFs (§4)

- **§4.1** D, Λ, G1, G2 (height-correlated), eval, pdf and the α clamp ✓ (`bsdf_microfacet.h:500-640, 960-969`). VNDF with the `(1+V'_z)/2` warp ✓ (`:188-219`). Sample rejection on `sign(Ng·L)` and `sign(N·L)` ✓ (`:757-761`). The singular case is covered in C2.
- **§4.2** Every Fresnel formula ✓ (`bsdf_util.h`).
- **§4.3** Lambert ✓. The EON Oren–Nayar formulas ✓ (`bsdf_oren_nayar.h`).
- **§4.4** The order and weights of the Principled layers ✓ (`svm/closure.h:98-527`):
  - alpha → sheen → coat → emission (× the current weight) → metal (F82, f0 = clamped base, f82 = min(tint, 1)) → transmission → IOR level → specular (`f0·tint`, saturated at setup, `exponent = −η'`) → layering → diffuse (unclamped base, N not N_spec).
- **§4.5 / §4.6** LUT layout, interpolation, precompute, `preserve_energy` and the `darkening` factor ✓ (`bsdf_microfacet.h:359-410, 423-498`). `lookup_table.h` is exact as quoted.
- **§4.8** glTF importer mapping:
  - distribution not set (stays MULTI_GGX) ✓;
  - B → Metallic, G → Roughness ✓;
  - `Specular IOR Level = 0.5·specularFactor` ✓;
  - IOR default 1.5 ✓;
  - greyscale emissive factor folded into Strength ✓;
  - `use_backface_culling = not double_sided`, ignored by Cycles (no reference in the Blender sync) ✓;
  - `import_shading` default 'NORMALS' ✓.
- **§4.9** The USD importer also does not set the distribution ✓. Light mapping ✓, with the exceptions in C6.
- **§4.10** sRGB decode happens after interpolation (`svm/image.h:36-38`) ✓. Metal samplers are `filter::linear` with no mip filter (`device/metal/compat.h:387-395`), so there is no mipmapping on GPU ✓.
- **§4.11**
  - `ensure_valid_specular_reflection` threshold `min(0.9·Iz, 0.01)` ✓;
  - `bump_shadowing_term` (reject rule for eval and diffuse sampling; Conty-Estévez G1 with α² = saturate(0.125·tan²θ_d) for diffuse when bump correction is on) ✓;
  - shadow terminator offsets ✓;
  - `ray_offset` constants ✓.

### Integrator (§5)

- **§5.1–5.3** ✓, except C3.
- The clamp compares the RGB **sum** `reduce_add(|L|)` against `3·value` (`film/light_passes.h:141-146`). This explains the ×3.
- `ensure_finite` is always applied.

### Pixel filter (§5.4)

✓:
- box width forced to 1 (`blender/sync.cpp:587-589`);
- `raster = (x + tbl(u), y + tbl(v))` (`camera.h:441-444`);
- raster y = 0 at the viewplane bottom (`transform_from_viewplane` in `util/transform.cpp:197-203`).

### Camera (§6)

✓, except C1. Also:
- Z flip at `blender/camera.cpp:334`;
- AUTO fit uses `sensor_width` (`:356-359`);
- clipping by planes (`camera.h:166-170`).

### Output (§7)

The EXR conclusion matches the introspection:
- after `file_format = 'OPEN_EXR'`, `linear_colorspace_settings.name` is 'Linear Rec.709';
- the display device is sRGB.

I did not re-read `colormanagement.cc` [UNVERIFIED line numbers].

### Headless rendering and timing (§8)

- The Open Data score (3746) and the timing estimate are [UNVERIFIED] by me.
- The script is consistent with the RNA names I introspected:
  - `use_sample_subset`, `sample_offset` and `sample_subset_length` (default 2048) exist;
  - `time_limit` exists.
- It should also set `c.sampling_pattern` explicitly (C7).

---

## 6. Recommended edits, in priority order

1. Replace the §6 example (C1) with:

   > `angle_y = 40°` gives `lens = 32.9697`, and the rendered hfov at 16:9 is 65.81°. `Camera.angle_x` (57.27°) is sensor_width-based and is not the render's hfov under VERTICAL fit.

2. §4.1 singular case: `pdf = lobe_prob·1e6`, `eval = F·1e6` (C2).
3. §5.3: the Filter Glossy clamp acts on α, and only when `blur_pdf < 1` (C3).
4. §4.9: the unnormalised sun has `eval_fac = 1`. A USD SphereLight becomes a soft-falloff **disk** unless `use_soft_falloff` is turned off (C6).
5. §2.7: MIS-off area lights are still camera-visible if `visible_camera = True` (C5).
6. §5.5: set `sampling_pattern` explicitly. The factory scene stores TABULATED_SOBOL; new scenes use AUTOMATIC, which becomes BLUE_NOISE_PURE in background renders (C7).
7. §10: use a t or Welch threshold instead of 0.27 % (C8).
8. Fix the citations in C9.
9. Add omissions 1, 3, 4 and 5 before the textured, smooth-shaded and alpha tiers.

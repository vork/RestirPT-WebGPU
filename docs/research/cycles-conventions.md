# Blender 5.1 Cycles conventions: how to make a custom WebGPU path tracer match Cycles to within noise

Target: a WebGPU/WGSL ReSTIR PT renderer validated against Cycles reference renders made with Blender 5.1.2 (`/Applications/Blender.app`, build hash `ec6e62d40fa9`, branch `blender-v5.1-release`, confirmed via `Blender --version`).

> **Where this file is.** Plan mode was active in this session, so the only file I could write was this plan file. The requested path `…/scratchpad/research/cycles-conventions.md` was **not** written. Copy this file there if needed.

## 0. Provenance, notation and conventions used in this report

**Sources.**
- Cycles/Blender C++ sources were read at git tag `v5.1.2`, which matches the installed build, from `https://raw.githubusercontent.com/blender/blender/v5.1.2/<path>`. File paths below are relative to that root. Line numbers are for that tag.
- The glTF importer was read from the installed bundle at `/Applications/Blender.app/Contents/Resources/5.1/scripts/addons_core/io_scene_gltf2/` (version 5.1.20).
- Property names and defaults were confirmed by headless, read-only `bpy` introspection (`Blender --background --factory-startup --python-expr …`). No files were saved and no renders were run.

**Tags.**
- **[INFERENCE]**: my own derivation or reasoning, not stated by a source.
- **[UNVERIFIED]**: plausible but not checked against code or execution.
- Everything else is read directly from code or from bpy introspection.

**Cycles direction naming, which matters when you port formulas.**
- `sd->wi` (called **V** here) points from the shading point toward the previous vertex, i.e. toward the camera.
- `wo` (called **L** here) points toward the next vertex, i.e. toward the light.
- **BSDF "eval" values in Cycles already include the cosine foreshortening of L.** For example, the diffuse eval is `max(N·L,0)/π`, and the microfacet eval is `F·D·G2/(4·N·V)`, which equals `f_r·(N·L)`.
- Every "eval" formula below follows this convention, so `f_cos(V,L) = f_r(V,L)·max(N·L,0)`.

**Units.** "Blender units" of radiance are arbitrary but consistent:
- an Emission shader of strength 1 and white colour has radiance 1;
- a point light of power P "watts" has radiant intensity P/(4π).

The film stores the plain average of radiance samples times `film_exposure`, which defaults to 1.

---

## 1. TL;DR checklist

### Render settings (defaults are from bpy introspection of 5.1.2 factory startup)

| Property | 5.1.2 default | Required for reference | Effect if left at default |
|---|---|---|---|
| `scene.render.engine` | `BLENDER_EEVEE` | `'CYCLES'` | wrong renderer |
| `scene.cycles.device` | `CPU` | `'GPU'` (Metal) | only slower |
| `cycles.samples` | 4096 | N | – |
| `cycles.use_adaptive_sampling` | True | **False** | per-pixel stopping rule, slightly biased / uneven noise |
| `cycles.use_denoising` | True | **False** | **biased** |
| `cycles.blur_glossy` (Filter Glossy) | **1.0** | **0.0** | **biased** (roughness widened along paths) |
| `cycles.sample_clamp_direct` | 0.0 | 0.0 | – |
| `cycles.sample_clamp_indirect` | **10.0** | **0.0** | **biased** (energy loss) |
| `cycles.light_sampling_threshold` | 0.01 | 0.0 | unbiased Russian roulette on shadow rays; ignored when the light tree is on |
| `cycles.use_light_tree` | True | either | unbiased; noise only [INFERENCE] |
| `cycles.caustics_reflective` / `caustics_refractive` | True / True | True / True | False is **biased** |
| `cycles.use_fast_gi` | False | False | True is **biased** (AO replacement) |
| `cycles.use_guiding` | False | False | unbiased but different noise |
| `cycles.pixel_filter_type` | **BLACKMAN_HARRIS** (width 1.5) | **'BOX'** (width forced to 1.0) | blurs the image |
| `cycles.max_bounces` | 12 | N (see §5.1) | different path length |
| `cycles.diffuse_bounces`, `glossy_bounces`, `transmission_bounces` | 4, 4, 12 | = `max_bounces` | per-type caps truncate paths |
| `cycles.volume_bounces` | 0 | 0 | – |
| `cycles.transparent_max_bounces` | 8 | ≥ 8 (keep) | Blender lights consume transparent bounces (§2.8) |
| `cycles.min_light_bounces` | 0 | = `max_bounces` to disable RR (optional) | RR is unbiased; noise only |
| `cycles.seed` / `use_animated_seed` | 0 / False | a distinct seed per independent run | – |
| `cycles.film_exposure` | 1.0 | 1.0 | scales EXR values |
| `cycles.scrambling_distance` / `auto_scrambling_distance` | 1.0 / False | 1.0 / False | <1 correlates pixels |
| `cycles.use_layer_samples` + `view_layer.samples` | USE + 0 | leave (0 means use scene samples) | – |
| `scene.render.film_transparent` | False | False | – |
| `scene.world` | grey world, colour 0.0509, strength 1 | **None** (black) or black background | adds environment light |
| `scene.render.use_motion_blur` | False | False | – |
| `scene.render.use_compositing` / `use_sequencer` | True / True | False / False | safety |
| `scene.render.dither_intensity` | 1.0 | 0 (affects 8-bit outputs only) | – |
| `view_settings.view_transform` | **AgX** | `'Standard'`, look `'None'`, exposure 0, gamma 1 | **does not affect EXR** (§7) but does affect PNG previews |
| `image_settings` | PNG 8-bit | `OPEN_EXR`, `color_depth='32'`, `exr_codec='ZIP'` | – |
| Camera `clip_start` / `clip_end` | 0.1 / 100 | e.g. 1e-3 / 1e5 | far geometry clipped |
| Light objects `visible_camera` | **False** in 5.1 (verified for `bpy.data.objects.new` and `ops.object.light_add`) | choose deliberately | – |
| `material.cycles.emission_sampling` | AUTO (NEE only if the emission estimate is > 0.5) | `'FRONT_BACK'` | noise only |
| `material.cycles.use_bump_map_correction` | True | False, unless you port §4.13 | changes specular normals and diffuse bump shadowing when a normal map is present |
| `Object.shadow_terminator_geometry_offset` | 0.1 | 0.0 | shifts shadow-ray origins on smooth-shaded triangles |
| `Object.shadow_terminator_shading_offset` | 0.0 | 0.0 | – |

### Scene conventions to reproduce in WGSL (details in §2–§6)

1. **Point and spot lights**: `I = P·color·2^exposure·nodeEmission / (4π)` [W/sr]. With radius 0 they are true delta lights and NEE-only.
2. **Spot lights** use the same I as point lights (**not** renormalised by the cone), multiplied by `smoothstep01((cosθ − cos(size/2)) / ((1 − cos(size/2))·blend))`.
3. **Area lights**: `L = P·color / (π·A)`. They are **one-sided** (emit toward local −Z) and **non-occluding / "transparent"**: rays pass through them after collecting emission. Spread < 180° reshapes the emission while preserving total power.
4. **Sun**: irradiance `E = strength·color` on a surface perpendicular to the sun; the disk radiance is `E/(π sin²(angle/2))`.
5. **Mesh emission**: `L = strength·color`, **two-sided**, and the mesh is opaque.
6. **MIS**: power heuristic with β = 2. The BSDF pdf is the sample-weight mixture over closures. Delta lights and lights with MIS disabled are NEE weight 1.
7. **`max_bounces = N`** allows at most **N+1 scattering vertices** (N indirect bounces). `max_bounces = 0` means direct lighting only.
8. **Box filter** over [x, x+1]×[y, y+1] in raster space, with raster y measured from the bottom.
9. **Principled BSDF**: validate with `distribution='GGX'` (single scatter, no energy compensation). You must still port the 16×16×16 albedo LUT `table_ggx_gen_schlick_ior_s`, because the diffuse lobe is weighted by `1 − E_spec(N·V)`. The alternative is a Diffuse + Glossy(GGX) mix, which needs no LUT.

---

## 2. Lights

### 2.1 Common pipeline: from Blender properties to Cycles `strength` and `eval_fac`

`intern/cycles/blender/light.cpp`:

- **Power (line 80):**
  `strength = light_color · b_light.energy · exp2f(b_light.exposure)`,
  where `light_color = (r,g,b)`, multiplied by `temperature_color` if `use_temperature`.
  RNA properties: `Light.color`, `Light.energy` (UI "Power", W; "Strength", W/m², for sun), `Light.exposure` (default 0, range ±32), `Light.use_temperature`, `Light.temperature`.
- **Normalisation (line 84):** `normalize = !(mode & LA_UNNORMALIZED)`. RNA `Light.normalize`, default True.
- **Point (lines 23–35):** `size = b_light.radius` (RNA `shadow_soft_size`, DNA `radius`, confirmed in `source/blender/makesrna/intern/rna_light.cc:358-359`).
- **Spot:** additionally `spot_angle = spotsize` (RNA `spot_size`, the **full** cone angle, range 1°…180°) and `spot_smooth = spotblend` (RNA `spot_blend`, range 0…1).
- **Soft falloff:** `is_sphere = !(mode & LA_USE_SOFT_FALLOFF)`. RNA `use_soft_falloff` defaults to **True** for point and spot, so by default the light is **not** a sphere.
- **Sun:** `angle = sun_angle` (RNA `angle`, full angular diameter).
- **Area:** `size = 1`, `sizeu = area_size` (RNA `size`), `spread = area_spread` (RNA `spread`, 0…180°).
  - SQUARE: `sizev = sizeu`, not an ellipse.
  - RECTANGLE: `sizev = area_sizey` (RNA `size_y`).
  - DISK: `sizev = sizeu`, ellipse.
  - ELLIPSE: `sizev = size_y`, ellipse.
- **Other Cycles settings:** `cast_shadow = LA_SHADOW` (RNA `use_shadow`); `use_mis = light.cycles.use_multiple_importance_sampling` (default True); `max_bounces = light.cycles.max_bounces` (default 1024); `is_portal` (area only); `is_caustics_light` (MNEE, default False).
- **Not read by Cycles sync:** `specular_factor`, `diffuse_factor`, `transmission_factor`, `volume_factor`, `use_square`, `cutoff_distance`, and the shadow-buffer settings. These are EEVEE-only [INFERENCE from their absence in `blender/light.cpp`].

**Light shader.** `blender/shader.cpp:1825-1833`: if the light has a node tree (every new light in 5.1 gets one, containing Emission with Color (1,1,1) and Strength 1.0, per introspection), that tree is compiled. Otherwise Cycles uses an implicit Emission with colour 1 and strength 1. `Light.use_nodes` is deprecated ("expected to be removed in 6.0").

The light's emitted value is:

`emission_shader_value · klight.strength · eval_fac`

This comes from `kernel/light/sample.h`, `light_sample_shader_eval_nee_constant` and `light_sample_shader_eval_forward`, which multiply by `klight->strength`, and from the integrator, which multiplies by `ls.eval_fac`.

**`eval_fac` for point and spot** (`scene/light.cpp:1267-1283`):

```
radius  = light.size
invarea = normalize ? 1/area : 1
area    = 4π·radius²  if radius > 0,  else 4          (Light::area, line 202-207)
eval_fac = invarea · (1/π)
is_sphere(kernel) = is_sphere && radius != 0
```

So with `normalize = True`:
- r = 0: `eval_fac = 1/(4π)`.
- r > 0: `eval_fac = 1/(4π²r²)`.

With `normalize = False`: `eval_fac = 1/π`. Avoid this; USD import produces it (§4.11).

**`has_contribution`** (`scene/light.cpp:151-176`): a light is removed if its strength is 0, if it is a portal, if it is an area light with `sizeu·sizev·size == 0` or a degenerate transform, or if the shader's emission estimate is 0.

### 2.2 Point light (`kernel/light/point.h`)

**(a) Radius 0: pure delta light.** This is the recommended validation configuration.
- `point_light_intersect` returns false when `radius == 0` (point.h:139), so the light can never be hit by camera or BSDF rays.
- `SHADER_USE_MIS` is set only if `use_mis && radius > 0` (light.cpp:1276). In `surface_shader_bsdf_eval` the BSDF pdf is then forced to 0 (surface_shader.h:379), so the NEE MIS weight is exactly 1.
- NEE: `ls.P = co`, `ls.pdf = invarea(=1) · t²/cos` with cos = 1, so pdf = t². `ls.eval_fac = 1/(4π)`.
- Contribution per NEE sample:
  `L_direct = [P·color·2^exposure·E_node] / (4π t²) · f_cos(V, L) · V(x, co) / p_select`,
  where `p_select` is the light-selection probability. This is the textbook isotropic point light with **radiant intensity `I = P/(4π)` W/sr**.

**(b) Radius > 0 with `use_soft_falloff = True` (the Blender default).** This is an "oriented disk" of radius r, always facing the shading point (point.h:64-77).
- The disk normal is `lightN = normalize(P_shade − co)`.
- NEE samples the disk uniformly by area. `pdf_ω = (1/(πr²)) · t²/cos_l` with `cos_l = lightN·(−D)`.
- Emitted radiance is uniform: `L = P/(4π²r²)`. When hit by BSDF rays, the disk is again oriented toward the ray origin (`ray_disk_intersect` with `diskN = normalize(ray.P − co)`, point.h:152-154).
- [INFERENCE, derived] On-axis irradiance at distance d is `E = L·π r²/(d²+r²) = P/(4π(d²+r²))`. This is the "soft falloff": 1/(d²+r²) instead of 1/d².

**(c) Radius > 0 with `use_soft_falloff = False`.** This is a true sphere of radius r with uniform radiance `L = P/(4π²r²)`. Outside the sphere it is solid-angle sampled with a uniform cone of half-angle `asin(r/d)`. [INFERENCE] Irradiance at a facing receiver fully outside is exactly `P/(4π d²)`. Inside the sphere it uses cosine or uniform-sphere sampling.

**Camera visibility.** A point light is hit by camera rays only if r > 0, the light *object* has `visible_camera` = True, and the ray actually hits it (light.h:278-281). Light objects default to `visible_camera = False` in 5.1.

**Recipe for a pure point light.**
```
ld = bpy.data.lights.new(name, 'POINT')
ld.shadow_soft_size = 0.0; ld.use_soft_falloff = False   # use_soft_falloff irrelevant when r = 0
ld.normalize = True; ld.exposure = 0.0; ld.use_temperature = False
```
Leave the node tree as its default (Emission 1.0, white), or set its Strength to 1.

### 2.3 Spot light (`kernel/light/spot.h`, `scene/light.cpp:1371-1391`)

All of §2.2 applies: the same `eval_fac` and the same sphere/disk logic. **Power is not normalised by the cone.**
- The RNA description of `energy` for spots: "The energy this light would emit over its entire area if it wasn't limited by the spot angle" (`rna_light.cc`).
- So the on-axis intensity equals that of a point light of the same power: `I = P/(4π)`.

Attenuation (spot.h:15-31, light.cpp:1372-1384):

```
cos_half    = cos(spot_size/2)
spot_smooth = 1 / ((1 - cos_half) * spot_blend)          // +inf when blend = 0
local = normalize(inverse(lightObjectMatrix) * dir_light_to_point)   // transform_direction, then safe_normalize
local.z = -local.z                                       // cone axis = local -Z of the light object
att = smoothstepf((local.z - cos_half) * spot_smooth)
smoothstepf(f) = 0 if f<=0 ; 1 if f>=1 ; 3f² - 2f³ otherwise   (util/math_float.h:485-495)
```

Notes:
- `dir_light_to_point = −ls.D`, i.e. from the sampled light point to the shading point.
- The inverse **object** transform is used, so object scale deforms the cone; non-uniform scale gives an elliptical cone. **Keep light objects unscaled.**
- `spot_blend = 0` gives a hard edge (`0·∞` is NaN exactly on the boundary; that set has measure zero).
- The blend region is `cosθ ∈ [cos_half, cos_half + (1 − cos_half)·blend]`.
- Radius > 0 behaves as in §2.2, with attenuation evaluated along each sampled direction. Inside the sphere there is no attenuation.
- `use_square` and `show_cone` are ignored by Cycles.

**glTF → Blender** (importer `blender/imp/light.py`):
- `spot_size = 2·outerConeAngle`, `spot_blend = 1 − inner/outer` (angle ratio).
- This is **not** the glTF reference falloff, which is linear or squared in cos. To match Blender, use Blender's formula with the converted parameters.

### 2.4 Area light (`kernel/light/area.h`, `scene/light.cpp:1329-1369`)

**Geometry.**
- `len_u = |X_obj|·size`, `len_v = |Y_obj|·size_y` (SQUARE and DISK use `size` for both). Object scale multiplies these.
- `size` is the **full** edge length, or the **diameter** for disk/ellipse.
- Area: `A = len_u·len_v`, or `(π/4)·len_u·len_v` for disk/ellipse.
- Emission direction: `dir = −Z_obj` (light.cpp:1210, 1367).

**Radiance.**
- `eval_fac = (1/π)·invarea` (area.h:313), with `invarea = normalize ? 1/A : 1`.
- So `L = P·color·2^exp·E_node / (π·A)` for every emitted direction in the front hemisphere when spread = 180°.

**One-sided.** NEE returns no sample if the shading point is behind the light plane (area.h:338). Ray intersection is rejected when `dot(ray.D, Ng) ≥ 0` (area.h:405). There is **no back emission**.

**Spread** (`area_light_spread_attenuation`, area.h:106-118):

```
a = spread/2 ; θ = angle between emission direction (−D) and light normal Ng
tan_half_spread = (spread == π) ? FLT_MAX : tan(a)
normalize_spread = (a > 0.05) ? 1/(tan a − a) : 3/a³          (Taylor, light.cpp:1351-1355)
att(θ) = max((tan a − tan θ)·normalize_spread, 0)      for spread ∈ (0, π)
att    = π if tan θ ≤ 1e-5 else 0                      for spread == 0 (collimated; avoid)
```

[INFERENCE, derived from the code comment at light.cpp:1347-1349]
- `∫_0^a cos x (1 − tan x/tan a) sin x dx = (tan a − a)/(2 tan a)`, so `att = (1 − tan θ/tan a)·tan a/(tan a − a)` **preserves total power** for any spread.
- At spread = 180°, att ≡ 1. `normalize_spread = 1/(FLT_MAX − π/2) ≈ 2.9e-39` is a denormal. On FTZ GPUs it flushes to 0 and the spread code is skipped entirely. Either way the factor is effectively 1.

**Sampling.**
- Rectangles use Ureña spherical-rectangle sampling (area.h:17-100).
- Disk/ellipse use uniform area sampling converted to solid angle.
- With spread < 180° the sample region is clamped to the visible sub-rectangle or circle.
- None of this changes the expectation.

**Portal** (`light.cycles.is_portal`): it is not an emitter (`has_contribution` returns false) and only guides world sampling. Ignore portals when the world is black.

**Camera visibility:** only if `visible_camera` = True (default False in 5.1). When visible, the camera sees L and **also whatever is behind the light** (§2.8).

### 2.5 Sun / distant light (`kernel/light/distant.h`, renamed `kernel/light/sun.h` in 5.2 with identical math; `scene/light.cpp:1285-1306`)

```
half = angle/2 ;  A_disk = π sin²(half) if half>0 else 1     (Light::area, DISTANT)
eval_fac = normalize ? 1/A_disk : 1
pdf = 1/(2π(1−cos half)) if half>0 else 1
```

- With angle = 0 it is a delta directional light: irradiance on a perpendicular surface is `E = strength·color` (RNA "Sunlight strength in W/m²").
- With angle > 0 it is a uniform-radiance disk: `L = E/(π sin²(angle/2))`, cone-sampled for NEE.
- It can be hit only by rays that escape to infinity (`shade_background.h`, `integrate_distant_lights`), and only if `SHADER_USE_MIS` (MIS on and angle > 0).
- Direction: the light travels along local −Z, so the direction toward the sun is +Z_local.
- **Recommendation:** use `angle = 0` for validation.

### 2.6 Per-light and per-object visibility

- **Ray visibility for lights** comes from the *object* flags `visible_camera`, `visible_diffuse`, `visible_glossy`, `visible_transmission`, `visible_volume_scatter` (light.cpp:309-337, `common.h: is_light_shader_visible_to_path`).
  - For NEE, an excluded type removes that closure type's contribution (`_surface_shader_exclude`).
  - Light objects in 5.1 default to `visible_camera = False`; the others default to True.
- **Shadows:** `Light.use_shadow = False` stops the light's shadow rays from being blocked.
- **Per-light bounce limit:** `light.cycles.max_bounces` (default 1024). NEE is skipped when `bounce > max_bounces` (light.h:25-29).

### 2.7 MIS in Cycles

- **Heuristic.** NEE weight `w = p_nee²/(p_nee² + p_bsdf²)`; forward-hit weight `w = p_bsdf²/(p_bsdf² + p_nee²)`.
  Sources: `kernel/light/sample.h`, `light_sample_mis_weight_nee/forward` (line ~356); `kernel/sample/mis.h`, `power_heuristic`.
- **`p_nee`** is the light-selection pdf times the solid-angle pdf (from the light tree or flat distribution).
- **`p_bsdf`** is the one-sample mixture `Σ_i sw_i·pdf_i / Σ_i sw_i` over all closures (surface_shader.h:270-305), where `sw_i` is the closure `sample_weight`.
- **Delta lights (r = 0) and lights with MIS disabled** get NEE weight 1 and are never hit (surface_shader.h:379-381; light.h:284).
- **Singular BSDFs** (microfacet with `α_x·α_y ≤ BSDF_ROUGHNESS_SQ_THRESH = 2e-10`, from `kernel/svm/types.h:545`, i.e. roughness ≲ 0.00376) lose `SD_BSDF_HAS_EVAL`. They get no NEE, and set `PATH_RAY_MIS_SKIP`, so a forward hit has weight 1.
  - Consequence: a perfect mirror never shows a delta light. This is the same in any correct renderer.
- **Mesh emitters with `emission_sampling = NONE`**, or the unsampled side of FRONT/BACK, get forward weight 1 (`light_sample_mis_weight_forward_surface`). Expectation is unchanged.
- **The heuristic and pdfs do not affect the expected value.** They matter only for (i) which lights are hittable at all (see above) and (ii) noise.

### 2.8 Blender lights are non-occluding and transparent

This one cannot be disabled.

- `kernel/integrator/shade_light.h:18-100`: when a camera or BSDF ray hits an analytic light (point/spot sphere or disk, area), Cycles adds its MIS-weighted emission and then **advances the same ray past the light** (`tmin = intersection_t_offset(t)`). The path continues to whatever lies behind.
- Each such pass increments `transparent_bounce`. The path terminates if it reaches `transparent_max_bounces` (default 8).
- Shadow rays never test lights, so **lights cast no shadows on anything**.

To match this in WGSL, choose one of two modes:
- **(A) Simplest.** Set `light.cycles.use_multiple_importance_sampling = False` and `visible_camera = False` for all analytic lights in Blender. Cycles then treats them as NEE-only emitters invisible to all rays. The WGSL renderer implements lights as NEE-only as well, with no intersection.
- **(B) Full.** MIS on. The WGSL renderer intersects area lights one-sidedly, adds `w_mis·L` and **continues the ray** without creating a scattering vertex.

### 2.9 Recommended validation light setup

- Point and spot: r = 0, normalize = True, exposure = 0, default node tree.
- Area: RECTANGLE or SQUARE, spread = 180°, unscaled object, `visible_camera = False`. Use MIS mode (A) first, then (B).
- Sun: angle = 0.
- Use no `temperature`.
- If lights are imported from glTF `KHR_lights_punctual` with the importer's default `export_import_convert_lighting_mode='SPEC'`: `P_point/spot = cd·4π/683` W, `E_sun = lux/683` W/m². Radiometric intensity is therefore `I = cd/683`.
- glTF-imported lights get `shadow_soft_size = 0` (new-light default in 5.1, confirmed by introspection), so they are delta lights.

---

## 3. Emission on meshes

- **Emission node** (`kernel/svm/closure.h`, `svm_node_emission_weight` + `svm_node_closure_emission`):
  `weight = color·strength·mix_weight` → `emission_setup`.
- **Principled emission** (closure.h:188-192, 337-338):
  `emission = EmissionColor·EmissionStrength·weight`, where `weight` already includes `alpha` and the sheen/coat attenuation.
- **Radiance** (`surface_shader.h:1102-1108`, `closure/emissive.h:43-47`):
  `L = emissive_simple_eval · closure_emission = (|Ng·wi| > 0 ? 1 : 0)·strength·color`.
  - So **L = strength × color** with **no π factor**. An emission-1 plane renders as pixel value 1.
  - It is **two-sided**. Additionally, `shader_setup_from_ray` flips `Ng` and `N` toward the ray for back-facing hits (`geom/shader_data.h:108-116`).
- A mesh with an Emission-only material has **no BSDF**, so the path ends there. It is **opaque** and occludes both shadow rays and BSDF rays.
- **Equivalence** [INFERENCE]: an emissive quad of area A and strength S emits `Φ = π·A·S` per side. To reproduce a one-sided area light of power P, use `S = P/(πA)` and hide the back side, for example by facing it into a wall.

**`material.cycles.emission_sampling`** (`scene/shader.cpp:236-278`): values NONE, AUTO, FRONT, BACK, FRONT_BACK.
- AUTO resolves to FRONT_BACK if `max(|estimate|·scale) > 0.5`, else NONE. `scale = 0.1` for Emission nodes auto-converted from colour links.
- This **only affects NEE sampling**: forward hits always see both sides, with MIS weight 1 for unsampled sides. It does not change the expectation.
- Set it to `'FRONT_BACK'` for lower noise in references.

---

## 4. BSDFs and the material subset to match

### 4.1 Microfacet GGX exactly as in Cycles (`kernel/closure/bsdf_microfacet.h`)

Definitions:
- `α_x, α_y` are clamped to [0,1] (`bsdf_microfacet_ggx_setup`, line 961).
- `α² := α_x·α_y`.
- Principled and Glossy use `α = roughness²` (closure.h:219; Glossy: `roughness = sqr(saturatef(param1))`, closure.h ~689).

Isotropic formulas (lines 500-563):

```
D(h)      = α² / (π · ((1 − c_h²) + α² c_h²)²),   c_h = N·H   (cos_NH² clamped ≤ 1)
Λ(c)      = ½ (sqrt(1 + α²·max(1/c² − 1, 0)) − 1)
G1(c)     = 1/(1 + Λ(c));   G2 = 1/(1 + Λ(N·V) + Λ(N·L))           (height-correlated Smith)
H         = normalize(V + L)                              (reflection)
eval_cos  = F(V·H) · D · G2 / (4 · N·V)                   ( = f_r · N·L )
pdf_ω     = D · G1(N·V) / (4 · N·V)  × lobe_pdf           (VNDF pdf; lobe_pdf = 1 for reflection-only)
```

Validity rules:
- Returns 0 if `N·V ≤ 0`, if the closure is singular, or if the transmission/reflection side does not match. `N·L < 0` means transmission and is rejected for reflection-only closures (lines 614-640).
- **Sampling** (lines 686-825) uses **Heitz 2018 VNDF**, `microfacet_ggx_sample_vndf` at line 189: stretch V → sample the projected disk with the `(1+V_z)/2` warp → unstretch. It then **rejects the sample if `sign(Ng·L)` or `sign(N·L)` disagrees with reflect/refract**.
- The eval path does **not** test Ng; that asymmetry is deliberate (comment at line 758).
- **Singular case** (`α_x α_y ≤ 2e-10`): H = N, `eval = pdf = F·1e6` (a "high number for MIS"). Treat it as a perfect mirror in WGSL with the same threshold.
- **Anisotropy** (Principled): `aspect = sqrt(1 − 0.9·aniso)`, `α_x = r²/aspect`, `α_y = r²·aspect`, with the tangent rotated by `rotation·2π`. Exclude it from the validation subset.

### 4.2 Fresnel models (`kernel/closure/bsdf_util.h`, `bsdf_microfacet.h:221-357`)

- **`fresnel_dielectric(cosθ, η)`** (bsdf_util.h:94) is the exact unpolarised dielectric Fresnel, `(R_s + R_p)/2`, and returns 1 under TIR.
- **`F0_from_ior(η) = ((η−1)/(η+1))²`** (line 338).
- **`ior_from_F0(f0) = (1+√f)/(1−√f)`** with `f = clamp(f0, 0, 0.99)` (line 332).
- **Generalized Schlick, `exponent < 0` mode** (Principled dielectric; bsdf_microfacet.h:313-319):
  `s = saturate((F_diel(V·H, η) − F0(η)) / (1 − F0(η)))`, `F = mix(f0, f90, s)` with `f90 = 1`.
  So F equals the real Fresnel curve remapped from `[F0_real, 1]` to `[f0, 1]`. `f0` is saturated at setup (line 856).
- **F82-tint** (metallic; bsdf_util.h:146-184):
  ```
  s = saturate(1 − c); F = saturate(F0 + (1−F0)s⁵ − b·c·s⁶)      (c = V·H)
  b = 0 if tint == 1, else  b = (F0 + (1−F0)f⁵)·(7/f⁶)·(1 − tint),  f = 6/7
  ```
- **Glossy BSDF node**: `fresnel_type = NONE`, so F ≡ 1 and the node colour is the closure weight.

### 4.3 Diffuse

- **Lambert** (`bsdf_diffuse.h`): `eval_cos = max(N·L, 0)/π`, `pdf = same`.
  - Sampling is cosine-hemisphere about N, **rejected if `Ng·L ≤ 0`**; eval has no Ng test.
  - The Diffuse BSDF node uses Lambert when `Roughness < 1e-5` (closure.h:530-545).
  - Principled uses Lambert when `Diffuse Roughness < 1e-5` (`CLOSURE_WEIGHT_CUTOFF`). The default is 0.
- **Oren–Nayar**, used only if roughness > 0, is the energy-preserving Fujii/OpenPBR "EON" variant (`bsdf_oren_nayar.h`):
  ```
  σ = sat(rough);  A = 1/(π + σ(π/2 − 2/3));  B = σA
  G(c) = sinθ(θ − 2/3 − sinθ c) + (2/3)(sinθ/c)(1 − sin³θ)   (Taylor for c<1e-6: (π/2 − 2/3) − c)
  t = L·V − (N·L)(N·V); if t>0: t /= max(N·L, N·V)
  E(c) = Aπ + B·G(c);  Eavg = Aπ + ((2π − 5.6)/3)B
  Ems  = (1/π)·ρ²·(Eavg/(1−Eavg)) / (1 − ρ(1−Eavg)),  ρ = saturate(color)
  eval_cos = (N·L)·[A + B t + Ems (1−E(N·V))(1−E(N·L))]
  ```
  Keep diffuse roughness at 0 for validation.

### 4.4 Principled BSDF: exact lobe structure (Blender 5.1)

Source: `kernel/svm/closure.h:98-527`.

**Defaults** (introspected): Base Color 0.8, Metallic 0, Roughness 0.5, IOR 1.5, Alpha 1, Diffuse Roughness 0, Subsurface Weight 0, Specular IOR Level 0.5, Specular Tint white, Anisotropic 0, Transmission 0, Coat 0 (roughness 0.03, IOR 1.5), Sheen 0, Emission Color white, Emission Strength 0, Thin Film 0 (IOR 1.33). **`distribution = 'MULTI_GGX'`** (enum: GGX, MULTI_GGX). `subsurface_method = 'RANDOM_WALK'`.

Algorithm, in order. `weight` starts at `mix_weight`.

1. **Clamps:** `base_color = max(C, 0)`, `Cc = min(base_color, 1)`, `η = max(IOR, 1e-5)`, `r = sat(roughness)`, `m = sat(metallic)`, `α = r²`.
2. **Alpha < 1:** add a transparent BSDF with weight `(1−alpha)`, then `weight *= alpha`.
3. **Sheen** (weight > 1e-5): add the sheen closure and `weight = layering(albedo_sheen, weight)`. Excluded from the subset.
4. **Coat** (weight > 1e-5): add GGX (dielectric, `α = coat_rough²`, `ior = coat_ior`) with weight `coat·weight`; `weight = layering(albedo_coat, weight)`; apply the coat tint `mix(1, tint^(1/cosNT), coat)`. Excluded.
5. **Emission:** `emission_setup(E_c·E_s·weight)`.
6. **Metal** (m > 1e-5): GGX with weight `m·weight`, `N = N_spec`, F82-tint with `F0 = Cc`, `f82 = min(SpecularTint, 1)`, energy compensation iff MULTI_GGX. Then `weight *= (1 − m)`.
7. **Transmission** (> 1e-5): GGX glass. Excluded. Then `weight *= (1 − trans)`.
8. **IOR level:** `f0 = F0(η)`. If `level ≠ 0.5`: `f0 *= 2·level`, `η' = ior_from_F0(f0)`, and `η' = 1/η'` if η < 1. Otherwise `η' = η`.
9. **Dielectric specular** (if `η' ≠ 1` or thin film): GGX with weight `weight`, `ior = η'`, `N = N_spec`, `f0 := f0·SpecularTint`, `f90 = 1`, `exponent = −η'`, energy compensation iff MULTI_GGX. Then:
   `weight = closure_layering_weight(albedo_spec, weight)`, which is
   `weight · saturate(1 − max_c(albedo_c/weight_c))` (bsdf_util.h:488-492).
10. **Subsurface** (> 1e-5). Excluded.
11. **Diffuse:** weight `base_color·(1 − sss)·weight` (**unclamped base colour**), `N = N` (not `N_spec`), Lambert if diffuse roughness < 1e-5.

**Specular albedo used in step 9** (`bsdf_albedo` → `bsdf_microfacet_estimate_albedo`, bsdf_microfacet.h:423-500):

```
albedo_spec / weight = darkening · mix(f0·T_s, 1, S(r, μ, z))
μ = V·N_spec,   z = sqrt(|(η'−1)/(η'+1)|),   r = sqrt(sqrt(α_x α_y)) (= roughness)
S = lookup_table_read_3D(table_ggx_gen_schlick_ior_s, x=r, y=μ, z, 16,16,16)
darkening = 1 for GGX; see §4.6 for MULTI_GGX
```

**Closed form for the validation subset** (GGX, no coat/sheen/transmission/SSS/thin film, alpha 1, isotropic, Diffuse Roughness 0):

```
F_m(c)  = saturate(Cc + (1−Cc)(1−c)⁵ − b·c(1−c)⁶)        (b = 0 when Specular Tint = white)
F_d(c)  = mix(saturate(f0·T_s), 1, saturate((F_diel(c,η') − F0(η'))/(1 − F0(η'))))
          (= F_diel(c, η) exactly when Specular IOR Level = 0.5 and T_s = 1)
w_d     = (1−m) · saturate(1 − max_c[ mix(f0·T_s, 1, S(r, V·N_spec, z)) ])
f_cos(V,L) = [ m·F_m(V·H) + (1−m)·F_d(V·H) ] · D·G2/(4 N_spec·V)  +  w_d · base_color · max(N·L,0)/π
L_e     = EmissionColor · EmissionStrength     (two-sided; × alpha)
```

- If `N_spec·V ≤ 0`, the specular part is 0.
- If roughness ≤ ~0.00376, the specular lobes are singular.
- [INFERENCE] **The diffuse weight depends only on V.** The BSDF is therefore **non-reciprocal**. For ReSTIR PT shift maps, always evaluate `f(V = toward camera, L = toward light)` with the full formula at every vertex. The integrand stays well defined.

### 4.5 The albedo LUT to port

The layering LUT is always on; there is no switch.

**Location.** `intern/cycles/scene/shader.tables`, array `static const float table_ggx_gen_schlick_ior_s[4096]` (lines 737–1010). Layout: `data[z·256 + y·16 + x]`, with x = rough (0…1), y = μ (0…1), z (0…1). Registered in `scene/shader.cpp:709`.

**How it was generated** (`app/cycles_precompute.cpp`):
- Mean over 2^20 Sobol samples of `saturate(eval.x/eval.y)` from `bsdf_microfacet_ggx_sample`.
- Uses a generalized Schlick with `f0 = (0,1,·)`, `f90 = (1,1,·)`, exponent −1, `ior = ior_from_F0(z⁴)`.
- Grid point i maps to `clamp(i/15, 1e-4, 1)` for rough and μ, and `clamp(i/15, 1e-4, 0.99)` for z.

**Interpolation** (`kernel/util/lookup_table.h`), per axis:
```
x' = saturate(x)·(n−1); i = min(int(x'), n−1); j = min(i+1, n−1); t = x' − i; lerp(data[i], data[j], t)
```
applied in nested order: z outer, then y, then x. `float_to_int` truncates. Port this exactly: WGSL `textureSampleLevel` on a 3D texture uses different addressing at the edges.

**Test vectors** (computed from the table with the formula above; IOR 1.5, so z = 0.4472; `E_spec = F0 + (1−F0)·S`, F0 = 0.04):

| roughness | μ=1.0 | μ=0.7 | μ=0.4 | μ=0.1 |
|---|---|---|---|---|
| 0.0 | S=0.0000, E=0.0400 | 0.0117, 0.0512 | 0.0955, 0.1317 | 0.5628, 0.5803 |
| 0.25 | 0.0001, 0.0401 | 0.0126, 0.0520 | 0.0927, 0.1290 | 0.4092, 0.4329 |
| 0.5 | 0.0006, 0.0405 | 0.0138, 0.0533 | 0.0639, 0.1013 | 0.1705, 0.2037 |
| 1.0 | 0.0014, 0.0413 | 0.0058, 0.0456 | 0.0161, 0.0554 | 0.0403, 0.0787 |

The diffuse weight is `1 − E`, for example 0.7963 at r = 0.5, μ = 0.1.

### 4.6 MULTI_GGX energy compensation

This is Tier 2: port it only if you want to match unmodified glTF or Principled imports.

`microfacet_ggx_preserve_energy` (bsdf_microfacet.h:359-410):

```
E    = lookup_table_read_2D(table_ggx_E[1024], x=r, y=μ=V·N, 32, 32)
Eavg = lookup_table_read(table_ggx_Eavg[32], x=r, 32)
Fss  = dielectric Principled: mix(f0·T_s, 1, saturate((Fss_diel(η') − F0(η'))/(1 − F0(η'))))
         Fss_diel(η) = (η−1)/(4.08567 + 1.00071η) for η ≥ 1 (Kulla–Conty fit, bsdf_util.h)
       metal (F82):  mix(F0, 1, 1/21) − b/126
       Glossy node:  node color
Fms  = Fss·Eavg / (1 − Fss(1 − Eavg))
multiplier on the single-scatter eval  = 1 + Fms·(1−E)/E        (per channel)
layering "darkening" (§4.4)            = E + Fms·(1 − E)
```

Test values: `ggx_E(r=0.5, μ=1) = 0.91528`, `ggx_Eavg(0.5) = 0.88204`.

### 4.7 Recommended material subset and node setups

- **Tier 0 (no LUTs, fully analytic).**
  - Diffuse BSDF (Roughness 0) with Color.
  - Glossy BSDF with **distribution 'GGX'** (the 5.1 default is MULTI_GGX), Roughness r, Color k: `f_cos = k·D·G2/(4N·V)`, F ≡ 1.
  - Mix Shader with a constant factor f: `(1−f)·A + f·B`.
  - Emission: `L = strength·color`.
  - Good for validating the integrator, the light normalisations and ReSTIR PT shifts.
- **Tier 1 (Principled, "GGX").** Set `node.distribution = 'GGX'`; keep Coat, Sheen, Transmission, Subsurface, Thin Film, Anisotropic and Diffuse Roughness at 0; Alpha 1. Any Metallic in [0,1], Roughness, IOR, Specular IOR Level and Specular Tint are then exact, using §4.4 plus the LUT from §4.5. This is what glTF metallic-roughness maps onto.
- **Tier 2.** Tier 1 plus MULTI_GGX (§4.6), so that imported GLBs match without post-processing.

### 4.8 What the Blender glTF importer produces for a metallic-roughness material

Importer 5.1.20, `blender/imp/pbrMetallicRoughness.py`, `material.py`, `mesh.py`.

**Nodes.** One `ShaderNodeBsdfPrincipled` feeding `ShaderNodeOutputMaterial`. The **distribution is not set**, so it stays at the default **MULTI_GGX**.
- **Base Color** = `baseColorFactor × baseColorTexture × COLOR_0` (via Mix node MULTIPLY). The texture uses sRGB colour space.
- **Alpha:**
  - OPAQUE → 1.
  - MASK → a `1 − (a < cutoff)` math node, or a constant.
  - BLEND → the alpha value.
- **Metallic / Roughness:** `metallicRoughnessTexture` → Separate Color. **B → Metallic, G → Roughness**, each × factor (Math MULTIPLY when the factor ≠ 1). Without a texture, the factor is set directly (default 1.0 each).
- **Normal:** Normal Map node (tangent space) → Principled Normal.
- **Emission:** Emission Color = `emissiveFactor × emissiveTexture`, Emission Strength = `KHR_materials_emissive_strength` (default 1). A **greyscale** emissive factor is folded into Strength.
- **IOR:** `KHR_materials_ior` or **1.5**.
- **Specular IOR Level** = `0.5 × KHR_materials_specular.specularFactor` (default 0.5); **Specular Tint** = `specularColorFactor`.
- **Extensions:** clearcoat → Coat Weight/Roughness/Normal; transmission → Transmission Weight; sheen → Sheen Weight 1 + Tint + Roughness; volume → Volume Absorption; anisotropy → Anisotropic.
- **Occlusion** goes to a glTF Settings group and is **ignored by Cycles**.
- **Double-sided:** `mat.use_backface_culling = not double_sided`. **Cycles ignores backface culling** (no reference in `intern/cycles/blender/*.cpp`), so all surfaces are two-sided in Cycles.

**Shading.** The operator option `import_shading` is 'NORMALS' (default), 'FLAT' or 'SMOOTH'.
- In 'NORMALS' mode, faces whose three corner normals equal the face normal are marked flat.
- Other faces are smooth with custom split normals (`normals_split_custom_set_from_vertices`).

**Axes** (`blender_gltf.py:64-100`):
- `p_blender = (x, −z, y)·u`, where u = 1/unit scale (1 by default).
- Quaternions `(w, x, −z, y)`. Mesh vertices and normals are converted the same way.
- Cameras and lights get a local correction of R_x(+90°).
- [INFERENCE] Net result: `M_blender_world = C · M_gltf_world` for cameras and lights, and `world_b(v) = C·world_g(v)` for mesh points, with `C = R_x(+90°) = [[1,0,0],[0,0,−1],[0,1,0]]`.

**Cameras** (`blender/imp/camera.py`): `angle_y = yfov`, `sensor_fit = 'VERTICAL'`, `clip_start = znear`, `clip_end = zfar` (1e12 if infinite). The aspect ratio is ignored.

**Lights:** see §2.9. The glTF `range` is ignored.

**Validation post-processing script** (after `bpy.ops.import_scene.gltf(filepath=…)`):
- set `n.distribution = 'GGX'` for every Principled node (Tier 1);
- `mat.cycles.use_bump_map_correction = False`;
- `mat.cycles.emission_sampling = 'FRONT_BACK'`;
- `ob.shadow_terminator_geometry_offset = 0`.

### 4.9 USD import in Blender

Source: `source/blender/io/usd/intern/usd_reader_light.cc`, `usd_reader_material.cc`.

**UsdPreviewSurface → Principled.** No distribution is set, so it stays MULTI_GGX. Force GGX as for glTF.

**UsdLux lights:**

| UsdLux | Blender |
|---|---|
| RectLight | AREA, RECT: `size = width`, `size_y = height` |
| DiskLight | AREA, DISK: `size = 2·radius` |
| SphereLight | POINT: `radius`, or 0 if `treatAsPoint` |
| SphereLight + ShapingAPI.coneAngle | SPOT: `spot_size = 2·coneAngle`, `spot_blend = coneSoftness` |
| DistantLight | SUN: `angle = 2·angle` (degrees) |

- **Energy:** `intensity·π` (sun: `intensity·4`, commented "approximately matches Karma"), then × `light_intensity_scale`. Exposure, colour and temperature are copied.
- **Normalize:** `normalize = inputs:normalize`, whose USD fallback is **false**, so imported lights are usually **UNNORMALIZED**. `eval_fac` is then `1/π` without the area term: area radiance `L = intensity`, point intensity `I = intensity`.
- If the WGSL renderer imports USD itself, apply the same mapping. Otherwise set `normalize = True` after import.

### 4.10 Textures

These behaviours matter only for textured validation.

- **8-bit sRGB images are stored as sRGB bytes, interpolated, and only then decoded to linear.**
  - `util/image_metadata.cpp:79-91` sets `is_compressible_as_srgb`.
  - `kernel/svm/image.h:36-38` applies `color_srgb_to_linear_v4` *after* `kernel_image_interp`.
  - To match exactly in WebGPU, use `rgba8unorm` (not `-srgb`), filter manually or in hardware, and decode afterwards.
  - Float images and non-sRGB colour spaces are converted to linear (half) at load time.
- **Filtering:** Image Texture interpolation 'Linear' is bilinear with **no mipmapping** [INFERENCE: no mip use in the GPU image path; UNVERIFIED for the CPU texture cache, which is off by default]. The glTF importer maps NEAREST samplers to 'Closest'.
- `cycles.texture_limit_render` defaults to 'OFF', so textures are not downscaled.

### 4.11 Shading normals and the hacks that come with them

- **Backfacing** (`shader_data.h:108`): if `Ng·V < 0`, both Ng and N are negated. Cycles has no single-sided BSDFs.
- **Smooth normals:** barycentric interpolation of corner/vertex normals, normalised.
  - **Sampling** rejects directions below Ng (diffuse, Oren–Nayar, microfacet). **Evaluation** does not. For flat shading these are identical.
  - **NEE same-triangle skip** (shade_surface.h ~360-366): for an emissive triangle sampling itself, the sample is skipped when `ls.D` lies in the Ng hemisphere.
- **`ensure_valid_specular_reflection`** (bsdf_util.h:370-460): used for specular lobes when `SD_USE_BUMP_MAP_CORRECTION` is set (`material.cycles.use_bump_map_correction`, default True) **and** `N ≠ Ng`. That includes plain smooth shading. It bends N toward Ng so that `R = 2(N·V)N − V` satisfies `Ng·R ≥ min(0.9·Ng·V, 0.01)`. Disable it in Blender or port it.
- **`bump_shadowing_term`** (bsdf.h:65-125): applies when the closure N ≠ smooth N (normal/bump maps).
  - **Always active:** reject if `(Ns·L)(Ns·N)(N·L) < 0` (for eval, and for diffuse sampling).
  - **Diffuse only, when bump correction is on:** multiply by the GGX `G1(α² = 0.125·tan²θ_d, |Ns·L|)`.
- **Shadow terminator:**
  - `Object.shadow_terminator_shading_offset` (default 0): if > 0, multiply by `shift_cos_in`, where `frequency_multiplier = 1/(1 − 0.5·offset)` (`scene/object.cpp:659`, bsdf.h:127-137, 282, 603).
  - `Object.shadow_terminator_geometry_offset` (default **0.1**): on smooth-shaded triangles, shadow and AO ray origins are displaced by a parabolic smooth-surface offset (`kernel/light/sample.h:218-262`). Set it to **0**.
  - These RNA properties exist at the top level of `Object`. `ob.cycles.shadow_terminator_offset` / `ob.cycles.shadow_terminator_geometry_offset` also exist as Python properties, but the sync reads the DNA fields (`blender/object.cpp:288-290`) [INFERENCE: the `cycles.*` ones are legacy].
- **Ray offsets** (cannot be disabled):
  - Self-intersection is avoided by skipping the same primitive/object (`ray.self`).
  - `integrate_surface_ray_offset` (shade_surface.h) applies `ray_offset(P, Ng)` only if a watertight re-test says the same triangle would be missed. That is the Wächter–Binder integer-ulp offset: `int_scale = 256`, `origin = 1/32`, `float_scale = 1/65536` (`kernel/bvh/util.h`).
  - Lights advance with `intersection_t_offset` (nextafter).
  - Porting the prim-id skip plus this offset to WGSL gives the closest match.

**Recommendation.** For the first validation round, use **flat-shaded geometry** (Blender `mesh.shade_flat()` or `import_shading='FLAT'`) with no normal maps. Then add smooth normals with `use_bump_map_correction = False` and `shadow_terminator_geometry_offset = 0`.

---

## 5. Integrator semantics and render settings

### 5.1 What `max_bounces = N` means in path-vertex terms

- `scene/integrator.cpp:192-193`: `kintegrator.max_bounce = max_bounces + 1` (comment: "so that a bounce of 0 indicates no GI, only direct"). The same +1 applies to the diffuse, glossy, transmission and volume limits.
- `path_state.h:141-145` (`path_state_next`, after each BSDF sample): `bounce += 1`; if `bounce ≥ N+1`, set `PATH_RAY_TERMINATE_AFTER_TRANSPARENT`. The ray is still traced.
- At the next hit, `surface_shader_eval` keeps **0 closures** (surface_shader.h:1161-1163). Emission is still added with its MIS weight, completing the previous vertex's MIS pair, but there is no NEE and no further scattering.

**Result.**
- Camera hit x₁ arrives with bounce 0. NEE happens at x₁ … x_{N+1}. Emission is collected at x₁ … x_{N+2}.
- **At most N+1 scattering vertices** (surface interactions strictly between camera and emitter), i.e. at most N+2 segments from camera to light.
- `max_bounces = 0` is direct lighting only.
- If the WGSL renderer uses "max path length k = number of segments", then `max_bounces = k − 2`. Direct emission seen by the camera (k = 1) is always included.
- The diffuse, glossy and transmission caps each also terminate. Set them all equal to N so that only the total counts.
- Passes through analytic lights count as **transparent** bounces (§2.8), not regular ones.

### 5.2 Russian roulette

Source: `path_state.h:275-299`, `intersect_closest.h`.
- No RR while `bounce ≤ min_light_bounces + 1`.
- After that, `q = min(sqrt(max_c |throughput_c|), 1)`, and throughput is divided by q.
- This is unbiased. Set `min_light_bounces = max_bounces` to disable it and reduce noise.

### 5.3 Other biased or unbiased switches

- **`blur_glossy`**: `filter_glossy = (blur == 0) ? FLT_MAX : 1/blur` (integrator.cpp:240). The blur is applied in `surface_shader_prepare_closures` (surface_shader.h:189-212) as `roughness = max(r, sqrt(1 − filter·min_ray_pdf)/2)`. **Biased.** Set 0.
- **Clamps:** `sample_clamp_* = 0` → FLT_MAX (integrator.cpp:281-285). Otherwise the limit is `3·value`. **Biased.** `film_clamp_light` additionally **zeroes non-finite contributions**; this cannot be disabled and is harmless.
- **Caustics tricks:** with `caustics_reflective/refractive` False, glossy and refraction closures are dropped on paths with a diffuse ancestor (closure.h `__CAUSTICS_TRICKS__`). **Biased.** Keep True.
- **`light_sampling_threshold`**: when > 0 **and** the light tree is off, `light_inv_rr_threshold = exposure/threshold` (integrator.cpp:318-323) and shadow rays are RR-terminated with reweighting. Unbiased. Set 0 anyway.
- **Light tree:** unbiased importance sampling [INFERENCE]. For extra safety, validate one scene with it off.
- **Denoiser, adaptive sampling, Fast GI, path guiding:** all off.
- **Filter:** see §5.4.

### 5.4 Pixel filter

Sources: `scene/film.cpp:26-80`, `blender/sync.cpp:585-590`, `kernel/camera/camera.h:440-445`.
- `pixel_filter_type` is one of BOX, GAUSSIAN, BLACKMAN_HARRIS.
- **For BOX, Blender forces `filter_width = 1.0`** regardless of the property (sync.cpp:587).
- The filter is importance-sampled through a 1024-entry inverse-CDF table (`FILTER_TABLE_SIZE`). For a box of width 1 the table maps u ∈ [0,1] to an offset in [0,1] (symmetric construction in `util/math_cdf.cpp`), so `raster = (x + u', y + v')`.
- **Each pixel is the uniform average over [x, x+1]×[y, y+1] in raster space.** That matches a renderer that jitters uniformly in the pixel. The table discretisation error is negligible [INFERENCE].
- Raster y = 0 is the **bottom** of the viewplane (`transform_from_viewplane` maps bottom to 0). Saved image files are top-down, as usual, so the WGSL row r corresponds to raster `y = H − 1 − r`.

### 5.5 Seeds and splitting samples

- `cycles.seed` is hashed into the sampler (blue-noise patterns hash it again, integrator.cpp:303-313). The default `sampling_pattern` is TABULATED_SOBOL in 5.1.2 (introspected).
- **Independent references:** use different seeds. Averaging K renders with seeds 0…K−1 is unbiased.
- **Exact splitting of one long render:** `cycles.use_sample_subset = True`, `sample_offset = k·M`, `sample_subset_length = M`, `samples = total` (`blender/session.cpp:1006-1015`; property description "Typically used for distributed rendering"). Averaging the subsets gives a single total-spp render.
- `use_animated_seed` hashes the frame number into the seed.

### 5.6 World

- Set `scene.world = None`. With no world, Cycles keeps the empty `default_background` graph, so the background is black (`blender/shader.cpp`, `sync_world`).
- Alternatively, use a Background node with Strength 0.
- A constant world is fine too if the WGSL renderer implements a constant environment. The world then also lights the scene, and camera rays see it unless `film_transparent`.

---

## 6. Camera

**Coordinate conventions.**
- Blender world is right-handed, Z-up. A Blender camera looks down its local **−Z**, with +Y up and +X right.
- Cycles internally flips Z (`blender/camera.cpp:334`, `tfm·scale(1,1,−1)`) so that its camera space looks down +Z.
- glTF is Y-up, with the camera also looking down −Z and +Y up. See §4.8 for the conversion `C = R_x(+90°)`.

**FOV** (`blender/camera.cpp:340-400, 615`). Let W, H be the pixel dimensions with pixel aspect 1.
- `AUTO`: horizontal fit if `W > H`, else vertical. **The sensor size is always `sensor_width`** in AUTO.
- `HORIZONTAL`: horizontal fit, `sensor = sensor_width`.
- `VERTICAL`: vertical fit, `sensor = sensor_height`.
- Horizontal fit: `tan(hfov/2) = sensor/(2·lens)`, `tan(vfov/2) = tan(hfov/2)·H/W`.
- Vertical fit: `tan(vfov/2) = sensor/(2·lens)`, `tan(hfov/2) = tan(vfov/2)·W/H`.
- **Setting an exact vertical FOV:** `cam.sensor_fit = 'VERTICAL'` and `cam.angle_y = vfov` (verified: `angle_y = 40°` with `sensor_height = 24` gives `lens = 32.9697` and `angle_x = 57.265°` at 16:9). Equivalently, `cam.lens = sensor_height/(2·tan(vfov/2))`.
- Factory camera defaults: 50 mm, sensor 36×24, AUTO, shift 0, clip 0.1/100, DOF off.

**Ray generation that matches Cycles** (no DOF, no shift) [INFERENCE from the matrices above, including viewplane → NDC → raster]:

```
tx = tan(hfov/2), ty = tan(vfov/2)
(u,v) ~ U[0,1)²;   rx = x + u;   ry = (H − 1 − row) + v          // ry measured from bottom
d_cam = normalize( (2·rx/W − 1)·tx,  (2·ry/H − 1)·ty,  −1 )    // Blender camera space
d_world = R_cam · d_cam ;  origin = camera position
```

- **Clipping** (`kernel/camera/camera.h:160-170`, perspective) is against **planes**: `origin += (near/z)·d` and `tmax = (far − near)/z`, with z the camera-space depth of d. It applies to camera rays only.
- **Shift:** `viewplane.offset(2·aspectratio·shift)` (the larger dimension equals shift 1). Keep it at 0.

---

## 7. Colour management and output

**EXR files get no view transform** (verified in `source/blender/imbuf/intern/colormanagement.cc:2710-2800`, `IMB_colormanagement_imbuf_for_write`).
- For formats that require linear float (`BKE_imtype_requires_linear_float`, i.e. OpenEXR), the display-space branch (`save_as_render && !linear_float_output`) is skipped.
- The data is instead converted from scene_linear to `image_settings.linear_colorspace_settings.name`, which defaults to **'Linear Rec.709'**, the scene-linear role in the default config. No conversion happens [INFERENCE: equal colour spaces].
- **So `view_transform`, `look`, view `exposure` and `gamma` do not affect EXR.** `cycles.film_exposure` does.
- There is a "100 vs 203 nits" adjustment hook (`get_color_space_for_hdr_image`) that does not apply to Linear Rec.709 [UNVERIFIED].

**Recommended output settings** (enums verified):

```python
im = scene.render.image_settings
im.media_type = 'IMAGE'            # 5.x; 'MULTI_LAYER_IMAGE' for multilayer
im.file_format = 'OPEN_EXR'        # single-layer EXR contains the Combined pass only
im.color_depth = '32'              # float32; '16' = half
im.exr_codec = 'ZIP'               # lossless. Avoid DWAA/DWAB/B44/B44A (lossy) and PXR24 (24-bit float)
im.color_mode = 'RGB'              # RGBA also fine; alpha = 1 with an opaque film
im.color_management = 'FOLLOW_SCENE'   # linear_colorspace_settings.name == 'Linear Rec.709'
scene.view_settings.view_transform = 'Standard'; scene.view_settings.look = 'None'
scene.view_settings.exposure = 0.0; scene.view_settings.gamma = 1.0   # PNG previews only
```

Other notes:
- Non-RGBA output composites alpha **under black** (`colormanagement.cc`).
- Byte formats get `dither_intensity` noise.
- Stored pixel = `film_exposure × (1/N)·Σ samples`.
- For PNG previews, 'Standard' is the plain sRGB OETF.

---

## 8. Headless command line, Metal and timing

**Command** (arguments after `--` go to the script via `sys.argv`):

```
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python-exit-code 1 \
    -P render_ref.py -- --scene scene.glb --out out.exr --spp 65536 --seed 0
```

**Metal on Apple Silicon.** Introspected: `compute_device_type = 'METAL'`, devices `Apple M5 Pro (GPU - 16 cores)` (METAL, `use = True`) and CPU. Preference methods available: `refresh_devices`, `get_devices`, `get_devices_for_type`, `has_active_device`; `metalrt = 'AUTO'`.

```python
import bpy, sys, math
argv = sys.argv[sys.argv.index('--')+1:]
bpy.ops.wm.read_factory_settings(use_empty=True)          # empty scene; do BEFORE setting prefs
cp = bpy.context.preferences.addons['cycles'].preferences
cp.compute_device_type = 'METAL'
cp.refresh_devices()
for d in cp.devices: d.use = (d.type == 'METAL')
s = bpy.context.scene
s.render.engine = 'CYCLES'; c = s.cycles
c.device = 'GPU'
c.samples = SPP; c.use_adaptive_sampling = False; c.use_denoising = False; c.time_limit = 0
c.max_bounces = c.diffuse_bounces = c.glossy_bounces = c.transmission_bounces = NB
c.volume_bounces = 0; c.transparent_max_bounces = 8; c.min_light_bounces = NB   # (no RR; optional)
c.caustics_reflective = c.caustics_refractive = True
c.blur_glossy = 0.0; c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0
c.light_sampling_threshold = 0.0; c.use_light_tree = True
c.use_fast_gi = False; c.use_guiding = False
c.pixel_filter_type = 'BOX'; c.seed = SEED; c.use_animated_seed = False; c.film_exposure = 1.0
s.render.film_transparent = False; s.render.use_motion_blur = False
s.render.use_compositing = False; s.render.use_sequencer = False; s.render.dither_intensity = 0
s.render.resolution_x, s.render.resolution_y, s.render.resolution_percentage = W, H, 100
s.world = None
# ... build or import scene, lights (§2.9), camera (§6), output (§7) ...
s.render.filepath = OUT
bpy.ops.render.render(write_still=True)
```

- `--factory-startup` does not load user preferences, so all device settings must happen in the script. Preference changes made in background mode are not saved [INFERENCE].
- The first GPU render compiles Metal kernels, which takes minutes; later runs use the cache [UNVERIFIED cache location].

**Timing estimate** [UNVERIFIED; no renders were run because only introspection was permitted].
- The Blender Open Data median score for "Apple M5 Pro (GPU - 16 cores)" is **3746** (opendata.blender.org), roughly 0.3× an RTX 4090.
- A 512×512 Cornell box at 65,536 spp is `2.6e5 × 65,536 ≈ 1.7e10` camera samples, each ~5–10 rays on average with NEE.
- **Estimate: about 5–15 minutes** of GPU time on the M5 Pro, after kernel compilation. Add ~2× with RR disabled and 8+ bounces.
- **Calibrate:** render 1024 spp, time it, multiply by 64 (time scales linearly in spp). Use `use_sample_subset` to split into e.g. 16 × 4096 spp chunks.

---

## 9. Cycles behaviours that cannot be disabled, and how to account for them

1. **Analytic lights are additive and transparent** (§2.8): no occlusion, rays continue, and each pass costs one transparent bounce. *Account:* mode (A) (MIS off, NEE-only) or mode (B) (intersect, add, continue) in WGSL. Keep `transparent_max_bounces` ≥ 8.
2. **Mesh emission is always two-sided and opaque** (§3). *Account:* two-sided emitters in WGSL, or design emitters whose back side is hidden.
3. **Principled layering uses a view-dependent albedo LUT** (`1 − E_spec(N·V)`), and the result is non-reciprocal (§4.4–4.5). *Account:* port the 4096-float LUT and its interpolation exactly.
4. **Default MULTI_GGX energy compensation** (§4.6), on both Principled and Glossy nodes. *Account:* force GGX in validation scenes, or port `ggx_E` / `ggx_Eavg`.
5. **Singular-roughness threshold** `α_x α_y ≤ 2e-10`: roughness ≲ 0.00376 becomes a perfect mirror with no NEE. *Account:* same threshold in WGSL, or avoid such roughness values.
6. **Shading-normal handling** (§4.11): Ng-based sample rejection in all BSDF samplers, the `bump_shadowing_term` hemisphere test when normal maps are present, and `ensure_valid_specular_reflection` (unless bump correction is off). *Account:* flat geometry first, then port these.
7. **Self-intersection strategy** (prim-id skip plus ulp offset only when needed). *Account:* port it, or expect tiny differences at contact regions and very thin geometry.
8. **8-bit sRGB textures are decoded after bilinear interpolation**, with no mips. *Account:* use `rgba8unorm` and decode after filtering.
9. **Spot cone uses the inverse object transform**, so scale changes the cone. *Account:* unscaled light objects.
10. **Area-light size and normalisation include object scale.** *Account:* unscaled objects, or multiply by scale.
11. **Camera clipping is by planes, for primary rays only.** *Account:* identical near/far in WGSL, or tiny near and huge far.
12. **Box filter via a 1024-entry table.** Error is negligible.
13. **NaN/Inf contributions are silently dropped** (`film_clamp_light`). Harmless.
14. **`CLOSURE_WEIGHT_CUTOFF = 1e-5`**: lobes whose weight is below this are skipped (e.g. metallic 1e-6 → 0). Negligible.
15. **Principled clamping:** base colour is clamped only for metal/transmission, not for diffuse; roughness, metallic and alpha are saturated; IOR ≥ 1e-5.
16. **Denormal `normalize_spread` at spread = 180°**: CPU and GPU both give factor ≈ 1.
17. **Emission sampling heuristics and the light tree** change noise, not the expectation.

---

## 10. Suggested validation protocol

This section is a recommendation [INFERENCE].

1. **Analytic sanity tests** on both renderers:
   - Point light: `P = 4π` W, r = 0, h = 1 m above a white Lambert plane with albedo ρ. Expected radiance at the nadir point is `ρ/π`, and `ρ·cos³θ/(π h²)` off-axis.
   - Sun with E = 1 on a Lambert plane: `ρ·cosθ/π`.
   - Emission plane with strength S seen directly: pixel = S.
2. **Tier-0 Cornell box** (Lambert + GGX Glossy, one rectangle area light in MIS mode A, flat shading, `max_bounces` = 0, 1 and 8).
   - Cycles: K = 16 independent seeds × 4096 spp.
   - WGSL: many accumulated frames.
   - Per-pixel `z = (μ_W − μ_C)/sqrt(σ²_W/n_W + σ²_C/n_C)`, with σ² estimated from the per-render spread. Check `mean(z) ≈ 0`, `P(|z| > 3) ≈ 0.27%`, and region-averaged relative error < 0.5%.
3. Then Tier 1 (Principled GGX with the LUT), spot and point lights, MIS mode B, smooth normals, textures, Tier 2.

---

## 11. Open items

- Exact Metal kernel-cache location and first-compile time on M5 Pro [UNVERIFIED].
- Actual M5 Pro throughput for the Cornell reference [UNVERIFIED]; calibrate as in §8.
- Whether `ob.cycles.shadow_terminator_*` Python properties are used anywhere. The sync reads the `Object` DNA fields [INFERENCE].
- Exact light-tree unbiasedness for extreme spot or spread configurations [INFERENCE: unbiased]. Cross-check once with `use_light_tree = False`.
- USD importer material-to-Principled details beyond the distribution; not inspected.

## Sources

- Blender v5.1.2 sources (raw GitHub, tag `v5.1.2`):
  - Lights: `intern/cycles/scene/light.cpp`, `intern/cycles/blender/light.cpp`, `intern/cycles/kernel/light/{point,spot,area,distant,common,light,sample}.h`
  - Closures: `intern/cycles/kernel/closure/{bsdf_microfacet,bsdf_util,bsdf_diffuse,bsdf_oren_nayar,bsdf,emissive}.h`, `intern/cycles/kernel/svm/closure.h`, `intern/cycles/kernel/svm/image.h`
  - Integrator: `intern/cycles/kernel/integrator/{path_state,shade_surface,shade_light,shade_background,surface_shader,intersect_closest}.h`, `intern/cycles/scene/integrator.cpp`, `intern/cycles/blender/sync.cpp`
  - Film and camera: `intern/cycles/scene/film.cpp`, `intern/cycles/util/math_cdf.cpp`, `intern/cycles/kernel/camera/camera.h`, `intern/cycles/scene/camera.cpp`, `intern/cycles/blender/camera.cpp`
  - Shaders and tables: `intern/cycles/scene/shader.cpp`, `intern/cycles/scene/shader.tables`, `intern/cycles/app/cycles_precompute.cpp`, `intern/cycles/kernel/util/lookup_table.h`, `intern/cycles/blender/shader.cpp`, `intern/cycles/blender/object.cpp`, `intern/cycles/scene/object.cpp`, `intern/cycles/util/image_metadata.cpp`, `intern/cycles/kernel/bvh/util.h`
  - Blender side: `source/blender/makesrna/intern/rna_light.cc`, `source/blender/imbuf/intern/colormanagement.cc`, `source/blender/io/usd/intern/usd_reader_light.cc`
- Installed glTF importer: `/Applications/Blender.app/Contents/Resources/5.1/scripts/addons_core/io_scene_gltf2/blender/imp/{pbrMetallicRoughness,light,camera,material,mesh,blender_gltf}.py`
- bpy introspection of Blender 5.1.2 (factory startup, read-only)
- Blender Open Data, Apple M5 Pro (GPU - 16 cores): https://opendata.blender.org/devices/Apple%20M5%20Pro%20(GPU%20-%2016%20cores)/

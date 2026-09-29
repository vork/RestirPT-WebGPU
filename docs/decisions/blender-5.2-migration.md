# Blender 5.1.2 → 5.2.2 Cycles migration audit

**Status: complete (source-level).** Nothing was rendered, because the GPU lock is shared. Every verdict below comes from
reading both tags' source and from headless RNA introspection in the installed Blender 5.2.2. A verdict of "no radiance
change" still has to be confirmed by re-running the gates (§E).

## Sources and method
- A shallow, blobless git fetch of tags `v5.1.2` (ec6e62d) and `v5.2.2` (d13f752), with sparse worktrees at
  `scratchpad/bl/wt5.1.2` and `scratchpad/bl/wt5.2.2`. They hold `intern/cycles`, `scripts/addons_core/io_scene_gltf2`,
  `source/blender/{makesrna,nodes/shader,blenloader,makesdna}`.
- Installed build: Blender 5.2.2 LTS, hash d13f752e3b9c, built 2026-09-15 (matches the tag).
- `git diff --stat v5.1.2 v5.2.2 -- intern/cycles`: **484 files changed, +28320 / −16425** (list in `scratchpad/bl/changed.txt`).
- File and line references are written as `A:n` (v5.1.2) and `B:n` (v5.2.2), relative to `intern/cycles/` unless stated.
- Tags: [SOURCE] means read in code; [RNA] means introspected in 5.2.2 with `-b --factory-startup` and no render;
  [INFERENCE] means derived and not directly verified.
- Scratch evidence:
  - `scratchpad/bl/rna/` (`probe522.json`, `apply522.json`: the repo's `apply_settings` run in 12 configurations);
  - `bl/imgtex/`, `bl/bsdf/`, `bl/lights/`, `bl/integ/`, `bl/svmutil/` (per-area diffs and probes).

## Summary

For our pinned configuration, no audited change alters rendered radiance, **provided `scene.render.use_texture_cache`
is pinned False**. That setting is new in 5.2.2 and **defaults to True**.

Three pins are recommended. Some changes are noise-only or latent:
- one noise-only change (the glass lobe-selection weight);
- one new Principled input to pin (`Thin Wall`);
- one latent Blender-side change to normals.

Most of the 484-file diff is refactoring:
- SVM bytecode moved from `uint4` records to typed structs;
- `dual1/2/3` derivative types replace the `*_BUMP_DX/DY` variants;
- the path-flag split (`PATH_RAY_VISIBILITY_*`);
- per-object vertex position offsets and a packed float3;
- `distant.h` renamed to `sun.h`;
- the `Light` class was split into subclasses;
- texture-cache miss/resume plumbing;
- MNEE moved into its own kernel.

---

## A. Textures and the texture cache (blocking item): resolved

### A1. What is new [SOURCE]
- A tiled, mip-mapped texture cache:
  - `scene/image_cache.{h,cpp}`, `util/image_maketx.{h,cpp}`, `blender/texture_cache.cpp`, `blender/addon/maketx.py`;
  - `session/cache_eviction.cpp`, `util/cache_limiter.h`;
  - an operator `render.generate_texture_cache` and the CLI command `blender --command maketx`.
- Kernel entry point:
  - `svm_image_texture(kg,id,x,y,flags)` (A `kernel/svm/image.h`) became `svm_image_texture(kg,sd,id,dual2 uv,flags)` →
    `kernel_image_interp_with_udim` (B `svm/image.h:22-39`).
  - The UDIM search moved to `kernel/util/image_2d.h:17-46`.
- **LOD selection exists only on the tiled path**, in `kernel_image_tile_map` (B `kernel/util/image_2d.h:75-190`):
  - `level = 0.5·log2(max(min(|∂u|²W²,|∂v|²H²), max(…)/256))`;
  - plus 0.5 and a stochastic ±0.25 jitter when `sd->lcg_state≠0` (B `surface_shader.h:1257` now always seeds it);
  - plus `mip_bias = −log2(texture_resolution)`;
  - software wrap, Y flipped, bilinear inside a padded tile atlas;
  - while a tile is still loading, the GPU returns `tex.average_color`.
- The tiled path is taken only when `tex.tile_descriptor_offset ≠ KERNEL_TILE_LOAD_NONE` (B
  `kernel/device/gpu/image.h:109-128`). The host sets that only when all of the following hold (B `scene/image.cpp:493`):
  1. `scene.render.use_texture_cache` is True;
  2. the image comes through `OIIOImageLoader`, i.e. it is file-backed and not packed, generated, movie or dirty
     (B `blender/util.h:326-342`, `blender/shader.cpp:928-1005`);
  3. `resolve_tx` finds a valid tiled + full-mip file (B `util/image_maketx.cpp:380-441`), searching in this order:
     - the preferences directory `texture_cache_directory`;
     - `<image dir>/blender_tx/<basename>.<ver>-<md5>.tx`;
     - **the source file itself**, if it is already a tiled EXR/TIFF with a full mip chain.

  `set_need_derivatives()` is requested only under the same condition (B `scene/shader_nodes.cpp:427-429` for images,
  `:619-621` for the environment texture). Otherwise the float3 SVM variant runs and the `dual2` has zero derivatives.

### A2. Full-image path (cache off or no tx): identical to 5.1.2 [SOURCE]
- The GPU reads `ccl_gpu_image_object_read_2D<float4>(tex, uv.x, uv.y)` at the raw uv (B `gpu/image.h:129-150` vs A `:86-102`).
- Metal sampler index is `extension (+4 if not Closest)` (B `device/metal/device_impl.mm:1123-1126` vs A `:1017-1020`).
- `kernel/device/metal/compat.h` (`metal_samplers`) and `metal/context_begin.h` are **unchanged**.
- Textures are still not mip-mapped (`mipmapped:NO`). FLOAT4 maps to RGBA32F and HALF4 to RGBA16F.
- Result: the same hardware bilinear or nearest filtering at LOD 0, the same texel centres, the same REPEAT/EXTEND/CLIP
  wrap, and the same storage.
- The loader is unchanged:
  - file-backed images use `OIIOImageLoader` in both tags;
  - `oiio_load_pixels` and `conform_pixels`, colour-space detection, alpha handling and the half/float choice are unchanged;
  - no float texture compression was added.
- New in 5.2.2: a **dirty** image now goes through `BlenderImageLoader` (B `blender/util.h:338-339`).
- Verdict: **no radiance change** on this path.

### A3. Answers to the coordinator's questions
- **Settings that force LOD-0 bilinear with no mips, no ray differentials, no .tx and no resolution limit**, for both
  Image Texture nodes and the world Environment Texture on Metal in `-b` mode:

  | RNA path | Value | 5.2.2 default | Effect |
  |---|---|---|---|
  | `scene.render.use_texture_cache` | **False** | **True** [RNA] | The only switch that guarantees the full-image path: no `resolve_tx`, no tiles, no LOD (`image_oiio.cpp:35`, `image.cpp:493`). **Required.** |
  | `scene.render.use_auto_generate_texture_cache` | False | False [RNA] | Nothing is ever written to disk. |
  | `scene.cycles.texture_resolution_render` | 1.0 | 1.0 [RNA] | New. Acts only when `use_simplify` is on (B `blender/sync.cpp:1008-1015`). |
  | `scene.cycles.texture_limit_render` | `'OFF'` | `'OFF'` | Already pinned. Same items; acts only under Simplify (A `sync.cpp:943-948`). |
  | `scene.render.use_simplify` | False | — | Already pinned. |
  | `scene.cycles.blur_glossy` | 0.0 | 1.0 | Already pinned (see below). |

  - Cycles add-on preferences have no texture options [RNA].
  - The Image and Environment nodes have no filter property [RNA].
  - `ShaderNodeTexEnvironment` has no `extension` property. The environment texture is always `EXTENSION_REPEAT`
    (B `shader_nodes.cpp:592`), so the documented pole-wrap quirk remains.
  - Also assert, in `build_scene.py`, for every image: `img.source=='FILE' and img.packed_file is None and not img.is_dirty`.
    This keeps the 5.1.2 loader. Checked [RNA]: `images.load`, then setting colour space and `alpha_mode`, then
    `pixels.foreach_get` leaves `is_dirty` False.
- **Where `.tx` files go:**
  - With the default empty preference: `<image dir>/blender_tx/<basename>.0-<md5>.tx`.
  - A relative preference is resolved against the image directory; an absolute one is used as is.
  - They are written only with auto-generate on, or by the operator or CLI. The write is atomic, and the mtime is set
    equal to the source's.
  - **With the cache on, existing `.tx` files are reused even when auto-generate is off.** A tx counts as fresh when its
    mtime equals the source's. The md5 does not cover file content, so a source replaced with the same mtime (`cp -p`,
    `rsync -a`, tar) would pick up a stale tx [INFERENCE].
  - Any input that is itself a tiled + mipped EXR/TIFF would silently switch to LOD sampling.
  - Current inputs are safe even at the default. `validation/scenes/*/env.exr` and `assets/downloaded/hdri/*` are all
    scanline with 1 mip (checked with OIIO), textures are PNG, and there are no `blender_tx/` directories. Pin the cache
    off anyway.
- **`blur_glossy = 0`:**
  - It still disables the closure blur (`filter_glossy = FLT_MAX`, B `scene/integrator.cpp:269`).
  - It now also zeroes the new roughness-based ray-differential widening: `differential_widen_scale = min(1, blur_glossy)`
    (B `:270`), used by `bsdf_widen_dD` (B `closure/bsdf.h:77-87`) at `shade_surface.h:436,589`.
  - It does **not** remove camera or propagated differentials. On the tiled path those would still select mips above 0.
  - So `blur_glossy=0` plus `use_texture_cache=False` fully disables texture filtering. Neither is sufficient alone.
- **`use_pixel_jitter`:** see §E1.

### A4. Environment and background [SOURCE]
- **`kernel/light/background.h`:** the only change is the new `background_light_clamp_dD(kg,dD) = min(dD, map_dD)`
  (B `:20-27`).
  - `map_dD = min(π/res_y, 2π/res_x)` when the map is in use, otherwise FLT_MAX (B `scene/light.cpp:1321-1327`).
  - It is applied to the dD passed to `shader_setup_from_background`:
    - on BSDF escapes (not on camera-visible rays), `shade_background.h:69-83`;
    - in NEE, `shade_light.h:143-151`.
  - The importance-map bake now passes a per-texel dD (B `kernel/bake/bake.h`).
  - dD feeds only the texture-LOD derivatives, so there is **no effect with the cache off**.
  - `background_map_sample`, `background_map_pdf`, `background_light_sample` and `background_light_pdf` are unchanged
    (A `:20,109,324,427` = B `:29,118,333,436`).
- **Background MIS:** `light_sample_mis_weight_forward_background` has the same logic (A `light/sample.h:556-587` vs
  B `:575-615`). The only changes are a `path_visibility` argument and `light_index` renamed to `object_index`.
- **Importance map:**
  - `background_cdf` weight `average(|rgb|)·sin(π(i+0.5)/H)` is byte-identical (A `light.cpp:927`, B `:1156`).
  - Pixel-centre bake, enable rule `has_portal || (use_mis && has_surface_spatial_varying)`, and the resolution rule
    are all unchanged.
  - `scene/background.cpp` only renames flags.
- **Equirect mapping:**
  - The float `direction_to_equirectangular` is unchanged; only dual overloads were added (`kernel/camera/projection.h`).
  - The SVM node still does `safe_normalize` and then equirect (B `svm/image.h:176-184`).
  - Texture Coordinate → Generated on the world is still the ray direction (`NODE_GEOM_P`; B `svm/geometry.h`
    `shading_position` returns `sd->P`).
  - Mapping POINT is unchanged (`svm/mapping_util.h` is only templated).
  - The OSL `node_environment_texture.osl` dropped a `1−v` flip. It is not used (SVM), so this is a note only.
- **World settings:**
  - New `world.cycles.use_shadows` (default True) maps to `BackgroundLight.cast_shadow` (B `blender/light.cpp:142`).
    False would make environment NEE unshadowed. In 5.1.2 it was implicitly true. **Pin True.**
  - `sampling_method`, `sample_map_resolution` (1024), `max_bounces`, `is_caustics_light` and `cycles_visibility` are
    unchanged [RNA].
- **Verdict:** no radiance change for math.md §5 or §6.
- Two pre-existing doc errors, the same in both tags (not a 5.2 change):
  - AUTOMATIC map resolution is `max(env res, 4096×2048)` unless there is exactly one sun-disc Sky node
    (A `light.cpp:1062-1068`, B `:1292-1298`). `docs/research/gap-env-hdri.md:107` and `:558` say "= image res".
    Variance-only; fix the text.
  - `gap-env-hdri.md:43,78` say the environment loads via `BlenderImageLoader`. It is `OIIOImageLoader` in both tags.
    Values are unaffected; ENV-U9 already hashes against the OIIO read.
- The bundled OIIO is now **3.1.13.1** [RNA]. gap-env-hdri cites 3.1.7.0. Re-run ENV-U1 (the bitwise decode check)
  on 5.2.2.

---

## B. Lights, light selection and light MIS: no radiance change

- **`sun.h` vs `distant.h`:**
  - This is a rename only. There are 122 lines each, and the diff is `distant_*`→`sun_*`, `LIGHT_DISTANT`→`LIGHT_SUN`
    (same enum value 1), `KernelDistantLight`→`KernelSunLight`.
  - `angle==0` still gives no hit and no eval (B `sun.h:63,79`).
  - `half_angle`, `one_minus_cosangle`, `pdf` and `eval_fac` are identical (A `light.cpp:1285-1306` → B
    `SunLight::copy_to_kernel :301-320`).
  - MIS flag: `use_mis && half_angle>0` in both.
- **`scene/light.cpp` (the 741-line diff):**
  - The monolithic `device_update_lights` (A `:1152-1400`) became per-class `copy_to_kernel`.
  - Every formula is equal:
    - power `color·energy·2^exposure`;
    - `normalize`, `cast_shadow`;
    - point/spot area `4π r²` and `eval_fac` `1/(area·π)`;
    - spot `cos_half`, `spot_smooth = 1/((1−cos_half)·blend)` (smoothstep blend), `half_cot`;
    - area `|u·sizeu|·|v·sizev|` (·π/4 for ellipse), invarea;
    - spread `N_s = a>0.05 ? 1/(tan a − a) : 3/a³` (B `:258-268`);
    - the MIS flags;
    - shape mapping;
    - visibility → `SHADER_EXCLUDE_*`.
  - The old generic `size` socket is gone. Area lights had `size ≡ 1` from Blender (A `blender/light.cpp` `LA_AREA`),
    so the values are equal.
  - `rna_light.cc` has no diff. Light RNA defaults are unchanged [RNA].
- **Light tree and distribution:**
  - Refactor only. `light_to_tree` is re-keyed by object index and absorbs `object_to_tree`.
  - Strength factors (point/spot 0.25/π, area 1/π, background `average_radiance·π`, `!normalize → ×area`) are unchanged.
  - Traversal and pdf are unchanged (`K/light/tree.h` only has renames and the position-offset read).
  - `use_light_tree` default is True [RNA]; it is pinned.
  - `emission_sampling` enum, default AUTO and the AUTO threshold `>0.5` are unchanged (`scene/shader.cpp:274`).
- **NEE and forward MIS:**
  - `light_sample_mis_weight_nee` (A `sample.h:339` / B `:344`) and the forward weights only gained arguments.
  - "Light without `SHADER_USE_MIS` → bsdf_pdf 0" is still there (A `surface_shader.h:377-381` / B `:405-409`).
  - The `integrate_surface_direct_light` formula is unchanged (B `shade_surface.h:395,414`).
  - Light pass-through `transparent_bounce+=1` is unchanged (B `shade_light.h:97-100`).
- **Mode-A camera-visibility quirk: still present, same rule.**
  - The gate `if (kernel_data.integrator.use_light_mis && !integrator_intersect_skip_lights(...))` is still the only
    `lights_intersect` call, including for camera rays (A `intersect_closest.h:426` → **B `:424`**).
  - `use_light_mis` is now `Scene::use_light_mis()` (B `scene/scene.cpp:851-865`): enabled && use_mis &&
    `is_traceable()`. That is the same set of lights as A `light.cpp:1159-1195`.
  - `_assert_invariants` stays valid.
  - A pre-existing imprecision: Cycles also requires the MIS light to be traceable (point/spot radius>0, area size>0,
    never sun). A config where only point/spot/sun lights have MIS=True still gives `use_light_mis=0`. Optional
    tightening: count only AREA lights with MIS.
- **Numeric changes (neutral for us):**
  - `shadow_ray_setup` for lights with `use_shadow=False` now computes P/D and sets `tmin=FLT_MAX` instead of zeroing
    the ray (A `sample.h:258-285` → B `:265-290`). It fixes evaluation of textured emission on no-shadow lights. We pin
    `use_shadow=True` and use constant emission.
  - `world.cycles.use_shadows` (see §A4).
- **Actions (docs only):**
  - `cycles_settings.py:485` and gap-light §1.1: `intersect_closest.h:426` → `:424`.
  - math.md:205 and cycles-conventions.md:215: `light.cpp:1351-1355` → `:264-268` (`AreaLight::copy_to_kernel`).
  - cycles-conventions.md:234 and cycles-conventions.verify.md:150: `distant.h`/`LIGHT_DISTANT` → `sun.h`/`LIGHT_SUN`.
  - Other `scene/light.cpp` citations now map to `{Point,Spot,Area,Sun}Light::copy_to_kernel` (B `:138-320`).
  - `area.h`, `point.h`, `spot.h`, `emissive.h` and `sample/mapping.h` are unchanged.
- **math.md §4, §7, §8 and §9 need no formula edits.**

---

## C. BSDFs and closures: one noise-only change, one input to pin

### C1. GGX, glass and η: numerically identical [SOURCE]
- These functions are unchanged apart from comments and the refactor below:
  - `microfacet_ggx_sample_vndf` (A:189/B:230);
  - `bsdf_lambda*` (A:507-530/B:602-625);
  - `bsdf_microfacet_eval` (A:587/B:688);
  - `bsdf_microfacet_sample` (A:678/B:779). This includes glass R/T selection, refraction, the TIR path and
    `m_singular ||= |η−1|<1e-4` (A:771/B:872).
  - `bsdf_microfacet_ggx_setup` and `ggx_glass_setup` (A:961/985 → B:1062/1086).
- Specular threshold: `roughness_is_almost_specular(α)` is `α_xα_y ≤ 2e-10`. That is the exact complement of the old
  `> BSDF_ROUGHNESS_SQ_THRESH (2e-10)` (B:676; A:581-584).
- `microfacet_fresnel` (A:226/B:335):
  - generalized Schlick moved into its own function;
  - the TIR `*has_reflection` factor moved to the common tail (B:427-428), which is equivalent;
  - DIELECTRIC, DIELECTRIC_TINT and F82 are unchanged;
  - thin film became per-channel but still runs only above `THINFILM_THICKNESS_CUTOFF 0.1`, so it is dead code at
    thickness 0.
- `bsdf_util.h`: `fresnel_dielectric`, `F0_from_ior`, `ior_from_F0`, `fresnel_f82_B`, `ensure_valid_specular_reflection`
  and `closure_layering_weight` have no hunks.
- **math.md §10, §12 (eval, sample, pdf, delta rules) and §13 stay valid.**

### C2. `svm/closure.h` setup: refactor [SOURCE]
- **Principled:**
  - The order (alpha → sheen → coat → emission → metal → transmission → IOR level → specular with layering → SSS →
    diffuse) is unchanged.
  - F82 is unchanged: `f0 = clamped C`, `f82 = min(spec_tint,1)` (A:343-376/B:320-352).
  - Transmission is unchanged:
    - `f0=F0(ior)·spec_tint`, `f90=1`, `exponent=−ior`;
    - `transmission_tint = sqrt(C)`, gated by caustics;
    - B builds it via `generalized_schlick_setup` (bsdf_microfacet.h B:41-55), with identical fields.
  - `weight·(1−t)`, IOR level `f0·=2L_s`, and the specular layering with `closure_layering_weight` are unchanged
    (A:417-462/B:425-472).
  - Diffuse is `C(1−ssw)·weight`, and the zero-roughness test is equivalent.
  - Thin film at thickness 0: not evaluated.
- **Glass node** (A:779-830/B:779-836), **Refraction** (`α=r²`, no saturate), **Glossy**, **Diffuse** (Lambert at
  roughness 0; Oren-Nayar params only refactored), and **Mix/Emission/Transparent**: refactors.
- The caustics gating now reads `ray_visibility & PATH_RAY_VISIBILITY_DIFFUSE`. It is irrelevant with
  `caustics_* = True`.
- **Not reachable for us:**
  - Emission under Coat/Sheen is now attenuated during emission-only evaluation (B `alloc.h:96-118`).
  - Subsurface with Scale 0 now scales the diffuse by (1−ssw).
  - New thin-glass closure `CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID` (B `bsdf_microfacet.h:1195-1405`).
- **New Principled input `Thin Wall`** (Bool, default False, socket index 5; B
  `nodes/shader/nodes/node_shader_bsdf_principled.cc:77`). When True, transmission becomes thin glass, which acts as
  transparent for non-camera rays when smooth. **Pin False and reject a link.**
- Blender-side default changes (all neutral while SSS and thin film are 0):
  - Subsurface Scale 0.05 → 0.005;
  - Cycles Subsurface Radius default → (1, 0.2, 0.1);
  - Thin Film IOR 1.3 → 1.33;
  - new `RANDOM_WALK_LEGACY` enum item. We set BURLEY.
- `distribution='GGX'` is still valid on Principled {GGX, MULTI_GGX}, Glass {BECKMANN, GGX, MULTI_GGX}, Refraction
  {BECKMANN, GGX} and Glossy/Anisotropic {BECKMANN, GGX, ASHIKHMIN_SHIRLEY, MULTI_GGX} [RNA]. Defaults are unchanged.
  The Blender→Cycles mapping is unchanged.

### C3. Lobe-selection weight of glass closures: a numeric change, noise-only [SOURCE, verified]
- `bsdf_microfacet_estimate_albedo`, old version (A:423-499):
  - it calls `microfacet_fresnel` first and skips the LUT if `is_zero(reflectance)`;
  - in the generalized-Schlick branch it replaces **only** the reflectance by `mix(f0,f90,s)·reflection_tint`;
  - so T_est is the smooth-surface `(1−F_gs(μ;η))⊙transmission_tint`.
- New version (B:498-596):
  - the `is_zero` shortcut is gone;
  - the generalized-Schlick branch returns `F·reflection_tint·[eval_R] + (1−F)·transmission_tint·[eval_T]` with
    `F = mix(f0,f90,lut3(r,μ,z,ggx_gen_schlick_ior_s))`;
  - backfacing beyond the critical angle gives T_est ≠ 0.
- Effect:
  - It changes `sample_weight` of glass-type closures (the Principled transmission lobe and the Glass node).
  - Principled specular (transmission_tint 0), coat, F82 (except the measure-zero all-zero F82 case) and the Refraction
    node are unchanged.
  - The Principled layering value `bsdf_albedo(…,true,false)` is unchanged, so **math.md §11 Λ_S is unchanged**.
- `sample_weight` feeds only closure picking (`surface_shader_bsdf_bssrdf_pick`, A:408/B:436) and the one-sample MIS
  normalisation `sum_pdf/sum_sample_weight`. So the **expectation is unchanged; only Cycles' variance moves**.
- It matters only when a glass closure coexists with others: Principled with 0<t<1, metal plus transmission, or Glass
  in a Mix.
  - Worked example [INFERENCE, hand-computed]: grey 0.8, IOR 1.5, r 0.5, μ 0.4 front gives sw_G 0.8773 → ≈0.905.
    Backfacing gives 0.1013 → ≈0.905.
- A side effect on documented deviation **D5** (rough-transmission NEE, approximate tier): Cycles' `w_C` in the
  spurious region depends on this normalisation, so the size of the documented excess shifts slightly [INFERENCE]. G5b
  stays in the approximate tier.
- **Action.** Choose one:
  - **(a) Keep our q at 5.1.2** and add deviation D6: "sw_G follows 5.1.2; noise-only". Our MIS already uses the
    balance form, not Cycles' power heuristic (math.md Open inconsistency 6), so bit-exact q was never required for
    expectation.
  - **(b) Follow 5.2.2.** Change math.md §12 "Sample weight" (`T_est` line, math.md:832) to
    `T_est = (1 − mix(f0,1,lut3(r,μ,z,table_ggx_gen_schlick_ior_s))) ⊙ transmission_tint`, and drop the TIR zero. Update
    gap-glass §2.6 and its test vectors, the `sw_G` code in `src/core/shaders/material/bsdf.wgsl:370` and glass.wgsl,
    the `v2.wgsl:18` `is_zero` comment, and the U-G6 vectors.
  - Recommendation: **(a)**. It is zero-risk for the gates.

### C4. `surface_shader.h` [SOURCE]
- Picking and `sum_pdf/sum_sample_weight` are unchanged (A:305/B:315, A:356/B:375).
- New `r_avg_roughness_squared` feeds only `bsdf_widen_dD`, a no-op at `blur_glossy 0`.
- `lcg_state` is now always seeded. Only the texture LOD jitter reads it, and that is off with the cache off.

### C5. LUTs
- `scene/shader.tables` has sha256 `fbde5fea…3af5175` in both tags, equal to `LUT_SOURCE_SHA256` in
  `cycles-glass-luts.ts` and `extract_luts.py`. `kernel/util/lookup_table.h` is unchanged.
- No action. Keep `SOURCE_URL` at v5.1.2, because `tests/material/luts.test.ts:17` asserts `/v5.1.2/`.

---

## D. SVM utility nodes: refactor, no radiance change [SOURCE]

- **Math (WRAP, PINGPONG, MULTIPLY, LESS_THAN, SUBTRACT):** `svm_math` body unchanged.
  - `wrapf`: `range = max−min; range≠0 ? v − range·floor((v−min)/range) : min` (A `util/math_base.h:435` / B `:449`).
  - `pingpongf`: `b≠0 ? |fract((a−b)/(2b))·2b − b| : 0` (A:441/B:455).
- **Vector Math MULTIPLY, Mix RGBA MULTIPLY** (`color_util.h` 0-line diff), **Mapping POINT**, **Separate/Combine XYZ**,
  **Separate Color**, **Vertex Color** (float4 POINT, now shareable without copy but same precision), **UV Map**
  (float2 CORNER, `(u,v,0)`), **TexCoord Generated on the world**, **Normal Map TANGENT** (same math; MikkTSpace sign
  unchanged), **Mix/Add Shader, Emission, Background**, **Convert/Value**: encoding or template refactors with the same
  `.val`.
- Attribute interpolation and storage: same types and elements, no half or compression.
- Derivative (`*_DERIVATIVE`) variants are compiled only when a texture node requests derivatives, i.e. only on the
  tiled cache path (B `scene/svm.cpp:999`, `shader_nodes.cpp:428,620`).
- Real changes that we don't reach:
  1. Blender Mix `Factor_Float` default 0.5 → 1.0 (`node_shader_mix.cc:45-46`). We set 1.0 explicitly
     (`build_scene.py:335,343`).
  2. `NORMAL_UNDISPLACED` now uses corner normals and is transformed. Only Normal Map `base=ORIGINAL` or displacement
     uses it; we use `DISPLACED` [RNA].
  3. Non-finite unlinked constants are zeroed (B `svm.cpp:310`). All our constants are finite.
  4. Normals domain; see §E4.
- glTF MASK (`1 − (α<cutoff)`), `texture.py` wrap, and `conversion.py` `texture_transform_gltf_to_blender` are
  unchanged. The `build_scene.py` copy is still verbatim.

---

## E. Camera, sampling, integrator and geometry: no radiance change with the pins

### E1. `use_pixel_jitter` [SOURCE]
- 5.1.2: `raster = (x,y) + filter_table(filter_uv)` with a per-sample `filter_uv` (A `camera.h:441-444`,
  `init_from_camera.h:27`).
- 5.2.2 (`camera.h:461-468`): the same path runs **only when `pixel_jitter.x == FLT_MAX`**.
- With `use_pixel_jitter=True`:
  - it uses one Halton(2,3) − 0.5 offset per `device_update`, shared by every sample and pixel (or
    `scene["override_pixel_jitter_sample"]`), i.e. a TAA mode with no per-sample anti-aliasing (B
    `scene/integrator.cpp:29-50,391-402`, `blender/sync.cpp:387-406`);
  - it also re-hashes the seed (`integrator.cpp:335-343`).
- With False (the default): `pixel_jitter = FLT_MAX` (`:402`), which is exactly the 5.1.2 per-sample box-filter jitter
  and the unhashed seed. The `render_reference.py` seed rule still holds.
- **Pin `scene.cycles.use_pixel_jitter=False`** (guard with `hasattr` for 5.1.2).

### E2. Camera, film and sampling
- Sensor fit, viewplane and the raster↔camera matrices are unchanged.
- New `differential_scale = 0.5(w/full_w + h/full_h)` equals 1 for final renders and only scales ray differentials
  (LOD only).
- The orthographic `dP.dy` bug fix does not affect us.
- Film: `scene/film.cpp` filter table is untouched; combined-pass accumulation is the same (B `light_passes.h:344-363`,
  `film/read.h:75-85`).
- `kernel/sample/*` is **identical** (diff -r).
- The `PathTraceDimension` enum (PRNG layout) is identical (checked).
- `sampling_pattern` handling is unchanged:
  - SOBOL_BURLEY needs `use_cycles_debug` and developer UI [RNA]; the default is AUTOMATIC;
  - scrambling is forced to 1.0 for non-TABULATED patterns.
- So **D1 and D4 carry over unchanged** [INFERENCE from the unchanged sampler and dimensions; re-confirm in the gates].

### E3. Bounces, Russian roulette and shadows
- The `PATH_RAY_*` bits were split into `path.visibility` plus a flag. The cleared set in `path_state_next` is equal
  (A:147 vs B:148-150).
- `path_state_continuation_probability` is identical (A:271-299/B:277-305). Bounce sockets and `+1` offsets are
  unchanged.
- math.md §2 and §25 need no change.
- Ranges for `transparent_max_bounces`, `min_light_bounces` and similar are still [0,1024] [RNA].
- Shadows:
  - glass is still opaque to shadow rays;
  - the GPU `num_hits` is now a 12-bit field, and 1024 < 4095;
  - the no-shadow light case is covered in §B.
- MNEE: the gate still needs a caster, a receiver and a light (A `scene.cpp:582-586`/B `:608-613`). Dispatch moved to
  the new `INTERSECT_MNEE` kernel. We keep all `is_caustics_* = False`, so it is off.

### E4. Intersection, offsets and normals
- Triangles:
  - vertices are now read via a per-object `position_offset` with mesh-local indices, giving the same values;
  - the intersection and `P = a + u(b−a) + v(c−a)` refinement are unchanged;
  - the self-intersection skip is unchanged;
  - Metal BLAS vertex stride is now 12 B (`packed_float3`), with the same values [INFERENCE: bit-identical HW
    intersection not tested].
- Offsets: `shadow_ray_offset` and the shadow terminator are unchanged; `ray_valid` gains `tmin<FLT_MAX`.
- Metal instance mask: the shadow bits now fall inside the 8-bit mask. This only matters for objects with shadow
  visibility off; we set none.
- `shader_setup_from_ray` and `triangle_smooth_normal`: renames only.
- **Latent Blender-side change:**
  - `blender/mesh.cpp` A:594 `normals_domain(true)` became B:628 `normals_domain()`, and `support_sharp_face` was
    removed.
  - Consequence: a mesh that **mixes flat and smooth faces without custom normals** now exports corner normals, so the
    smooth faces next to flat ones get split normals. Blender 5.2.2 reports `CORNER` [RNA]. The 5.1.2 result was POINT
    [INFERENCE; blenkernel 5.1.2 was not read].
  - Our meshes are all-flat (FACE) or all-smooth with a POINT `custom_normal` (asserted in `build_scene.py:194`), so
    they are unaffected.
  - **Action:** make "all-flat or all-smooth with custom POINT normals" an explicit contract in scene-bridge.md, or
    assert it.

---

## F. Settings table, glTF importer and crash fix (verified earlier)

### F1. The settings table reads back in 5.2.2 [RNA]
- `cycles_settings.apply_settings` ran in 12 configurations (CPU/GPU × black/constant/HDRI world × MIS off/on) with
  0 mismatches.
- No property we set was renamed, removed or re-defaulted. `rna_light.cc` and `rna_world.cc` are byte-identical.
- New properties that we don't set yet:
  - `use_pixel_jitter`
  - `texture_resolution_render`
  - `render.use_texture_cache`
  - `render.use_auto_generate_texture_cache`
  - `world.cycles.use_shadows`
  - the Principled `Thin Wall` input
  - the debug texture-cache eviction preferences, which have no effect with the cache off

### F2. glTF importer 5.1.20 → 5.2.40 [SOURCE]
- `com/conversion.py` and `imp/texture.py` are unchanged. The MASK convention is unchanged.
- **New `KHR_materials_iridescence` import** writes the Principled `Thin Film Thickness` (iridescenceThickness, default
  400 nm) and `Thin Film IOR`. The factor goes only to the glTF settings node group, which Cycles does not render.
  - **Action for E2E:** zero Thin Film Thickness, or reject assets that use the extension. `glass_rows` would otherwise
    fail loudly.
- `KHR_materials_dispersion` writes only to the settings group.
- Other changes: point-cloud import (opt-in), `KHR_*` custom attributes, exporter-only files. No UV, normal or COLOR_0
  change was found in the hunks read. The `mesh.py` diff was not read line by line.

### F3. Crash diagnosis: confirmed and still present in 5.2.2 [SOURCE]
- **The race.** `util/path.cpp` B:63 `static string cached_xdg_cache_path` and B:379-386 `path_cache_get` still
  initialise the path lazily without a lock. This code is not in the v5.1.2→v5.2.2 diff.
- **Where it is reached.** `MetalKernelPipeline::compile()` → `path_cache_get` (`device/metal/kernel.mm:697`), inside
  `if (use_binary_archive)`. It runs on `max(2, maximumConcurrentCompilationTaskCount−1)` compile threads (`kernel.mm:309-323`).
- **The other Metal caller.** `device_impl.mm:596` runs only with `CYCLES_METAL_PROFILING` or `CYCLES_METAL_DEBUG`
  set. CUDA and HIP callers are irrelevant; OSL uses `path_user_get`, and OSL is off.
- **Why the env var fixes it.** `should_use_binary_archive()` (`kernel.mm:402-435`, the same in both tags) returns
  false when `CYCLES_METAL_DISABLE_BINARY_ARCHIVES` is non-zero on macOS ≥ 15.4 (this machine: macOS 26). That skips
  the `path_cache_get` call and both `fileURLWithPath` calls (`:675-720`, `:826-837`).
- **When it is read.** The variable is read per pipeline compile, not at device init: the Metal device constructor
  reads other variables, and `refresh_devices()` does not compile. So setting `os.environ` before the first
  `bpy.ops.render.render` is early enough. Exporting it in the launching shell is also valid. Never change it while
  renders are running.
- **`kernel_optimization_level='OFF'`.** It is still valid (OFF/INTERSECT/FULL, default FULL), and
  `blender/device.cpp` is identical. It maps to `PSO_GENERIC`, but generic kernels are still archived. **On its own it
  does not avoid the race.**
- `grep` finds neither setting in `validation/`.

### F4. Cache keys and stale 5.1.2 text
- `render_reference.py` includes `bpy.app.version_string` in the cache key, so every reference re-renders.
- Stale text:
  - `docs/math.md:27` (`K/` is now the 5.2 kernel that ships in the app);
  - `docs/PLAN.md:20,210,220`;
  - `cycles_settings.py:4,485`;
  - the `build_scene.py` `_gltf_to_blender_transform` docstring;
  - `tests/scene/blender-blackbody.json` and `usd-loader.test.ts:43`, which were generated with 5.1.2. We pin
    `use_temperature=False`; blackbody code was not audited.

---

## G. Prioritized migration checklist

1. **Crash fix.**
   - Set `CYCLES_METAL_DISABLE_BINARY_ARCHIVES=1` before the first render in every Blender process:
     `render_reference.py`, `smoke_render.py`, `calib_scenes.py`, or the launching shell.
   - Keep `CYCLES_METAL_PROFILING` and `CYCLES_METAL_DEBUG` unset.
   - `kernel_optimization_level='OFF'` is optional.
   - Record both in the manifest.
2. **Texture cache off (blocking for M3c and textured scenes).** Add rows in `cycles_settings.py`, guarded with
   `hasattr` so 5.1.2 still works:
   - `scene.render.use_texture_cache = False`
   - `scene.render.use_auto_generate_texture_cache = False`
   - `scene.cycles.texture_resolution_render = 1.0`

   Keep `use_simplify=False`, `texture_limit_render='OFF'` and `blur_glossy=0`. Add
   `preferences.filepaths.texture_cache_directory` to the manifest. In `build_scene.py`, assert that every image is
   `source=='FILE'`, not packed and not dirty.
3. **Other new pins** (guarded):
   - `scene.cycles.use_pixel_jitter = False`
   - `worlds[..].cycles.use_shadows = True` in `_world_rows`
   - Principled `inputs["Thin Wall"].default_value = False`, rejecting a link (in `glass_rows` or the material rows),
     and set it in `build_scene.py`
4. **Glass sample weight (C3):** add deviation **D6** to `cycles-deviations.md` ("sw_G/q follow 5.1.2; 5.2.2 changed
   T_est to the LUT form; noise-only"), and note its small effect on D5. Alternatively take option (b) in C3.
5. **Mesh contract (E4):** document or assert "all-flat or all-smooth with custom POINT normals".
6. **glTF E2E (F2):** zero Thin Film Thickness, or reject `KHR_materials_iridescence`.
7. **Re-render and re-run every gate from M2 to M3c on 5.2.2.** The cache key forces the re-render.
   - Re-confirm D1 and D4. The sampler and PRNG layout are unchanged, so they are expected to hold.
   - Re-run ENV-U1 against OIIO 3.1.13.1 and ENV-U9.
8. **Doc updates, no formula changes:**
   - math.md:27 (`K/` = 5.2.2) and math.md:205 citation (`light.cpp:264-268`);
   - `cycles_settings.py:485` (`intersect_closest.h:424`);
   - `distant.h` → `sun.h` in cycles-conventions*;
   - gap-env-hdri.md:43,78 (OIIOImageLoader) and :107,:558 (AUTOMATIC res = max(img, 4096×2048));
   - the gap-env-hdri settings table (add the texture-cache pin);
   - PLAN.md version strings;
   - the blackbody fixture: regenerate it or label it 5.1.2.
9. Optional:
   - tighten `_assert_invariants` to count only AREA lights with MIS (a pre-existing imprecision);
   - assert Normal Map `base=='DISPLACED'`.

## H. Could not determine
- Render-level confirmation of every "no radiance change" verdict. Source-level only; no renders were run (GPU lock).
- Whether hardware triangle intersection on Metal is bit-identical after the 12-byte vertex stride change.
- The exact 5.1.2 behaviour of Blender's `normals_domain(true)` (blenkernel not in the checkout). This matters only for
  mixed flat/smooth meshes, which we don't produce.
- Whether the tiled path at LOD 0 would equal full-image bilinear (padding, tile borders). Moot with the cache off.
- The exact variance effect of the new glass `sw_G`. The 5.2.2 values are hand-computed.
- The bundled OIIO version of 5.1.2 (only 5.2.2 is installed), and the blackbody code (pinned off).
- The OSL and Hydra paths, which are unused.
- Commit intent. The clone is shallow; conclusions come from code and comments.

## Result: gates re-run on Blender 5.2.2 (2026-09-29)

Every Cycles reference was re-rendered with 5.2.2, using the pins above plus D7. Every gate passes:

| Gate | Result | Notes |
|---|---|---|
| M2 | 24/24 | Second run. The first run failed one unit on a 3600 s GPU-lock timeout (no render) and hit the old `Resources/5.1` python path, which is fixed in e63367b. |
| M3a | 175/175 | |
| M3b | 96/96 | |
| M3c | 173/173 | |

No Blender crash occurred in about 9 h of rendering. The source-level verdict "no radiance change under our pins" is therefore confirmed at gate level.

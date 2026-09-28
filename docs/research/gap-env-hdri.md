# HDRI environment map for v1: implementation-ready spec (Cycles 5.1.2 parity, PT, ReSTIR PT Enhanced, WebGPU)

Tag legend: **[SOURCE]** read in code this session (file:line), **[RNA]** headless read-only Blender 5.1.2 introspection this session, **[INFERENCE]** derived, **[UNVERIFIED]** plausible but not checked.

Sources read: Cycles kernel shipped in `/Applications/Blender.app/Contents/Resources/5.1/scripts/addons_core/cycles/source/` (below `K/` = `…/source/kernel/`, `U/` = `…/source/util/`); host code `raw.githubusercontent.com/blender/blender/v5.1.2/intern/cycles/…` (below `H/`); OIIO v3.1.7.0 `src/hdr.imageio/hdrinput.cpp`; three.js r186.1 `HDRLoader.js`, `EXRLoader.js`, `common.glsl.js`; Poly Haven API + license page. Plan and reports as listed in the task.

---

## 0. Decisions (read first)

1. **Mapping (exact, Cycles-identical).** For a unit glTF-world direction `d`, `b = R_z(γ)·C·d` with `C = R_x(+90°)`, i.e. `b = (cγ·d.x + sγ·d.z, sγ·d.x − cγ·d.z, d.y)`; `u = (atan2(b.y,b.x) − π)/(−2π)`, `v = (acos(b.z) − π)/(−π)`. Image centre `u = 0.5` looks toward **glTF +X**; `u = 0.25` → glTF −Z; `u = 0.75` → glTF +Z; `v = 1` zenith (+Y), `v = 0` nadir. `γ` is the Blender **Mapping node rotation Z (type POINT)**; the bridge writes γ unchanged (C is inside our formula; the env itself is not axis-converted). This equals three.js `equirectUv` for γ = 0.
2. **Texels = Cycles texels.** Cycles stores Blender float images as **RGBA32F** and samples them with Metal hardware bilinear, **address mode repeat on both axes**, no mips. We use `rgba32float` (`float32-filterable`), rows uploaded **bottom-up** (row 0 = nadir, Blender ImBuf order), sampler `repeat/repeat/linear/linear`, `textureSampleLevel(…, 0)`, sampled at the identical `(u, v)`. Consequence (quirk, replicate it): within half a texel of a pole the lookup blends the top and bottom rows.
3. **The bridge exports the exact texel array** we render with (float32 ZIP EXR, written by OIIO in Blender's Python), and asserts `image.pixels` hash equality in Blender. Decoders are tested separately (bitwise vs OIIO). Never use three.js `HDRLoader` values: it decodes RGBE as `byte·2^(e−128)/255`, **+0.39 %** vs OIIO/Blender (`byte·2^(e−136)`), and its half path clamps at 65504.
4. **Importance sampling (ours).** 2-D piecewise-constant over env `(u,v)` cells, power-of-two `W_m × H_m` (`H_m = W_m/2`), cell weight = exact integral of the bilinear-reconstructed `avg(|rgb|)` over the cell × row-average `sin θ`, with a **defensive floor** so every cell has positive probability. Sampled by a **two-level integer (u16-threshold) alias table**; the stored pdf table is the **realized** probability of that sampler computed in f64 from the stored integers, so `pdf_eval ≡ sampling density` exactly (not "approximately"). `pdf_σ(ω) = pdf_uv(u,v)/(2π² sin θ)`.
5. **Selection.** The env is one entry (stable id `ENV_ID`) of the global power alias table. Power proxy `Φ_env = 4π²·R_s²·L̄_env` (flux through the scene bounding sphere, same family as the plan's sun proxy), then `P(env)` clamped to `[0.1, 0.9]` when other emitters exist. `p1_env(ω) = pmf[ENV]·pdf_σ(ω)` is **one WGSL function** used by NEE, BSDF-escape MIS, RIS-NEE, shifts and refresh.
6. **Always MIS.** The env is not an analytic light: NEE_ENV and BSDF_ENV candidates exist in every mode (A, A′, B), like Cycles (`SHADER_USE_MIS` forced on the background light). `ω1 = M(B)p1/(M(B)p1+p2)`, `ω2 = p2/(M(B)p1+p2)`, `ω2 = 1` after a delta lobe.
7. **Measure.** μ gains a solid-angle component: `μ += δ_ENV ⊗ σ_env` (solid angle on S², env-local). For env candidates `Λ = L_env(ω)`, `q = pmf[ENV]·pdf_σ(ω)`, `p1 = q` (no r²/cos conversion).
8. **Reservoir.** Technique code 3 = `BSDF_ENV`; endpoint type 7 = `ENV`. NEE-env endpoint stored **losslessly** as the sampler's integers `(i, j, du16, dv16)` (env-local, follows env rotation). BSDF_ENV endpoint stored as the **world** direction `ω` (it is a BSDF sample; f32×3 in validation).
9. **Jacobians.** NEE-env forced reconnection: `J = 1` spatially, `J = J_P = pmf_t(ENV)/pmf_{t−1}(ENV)` temporally (env-local copy; the (u,v) density is unchanged by rotation/strength). BSDF_ENV as rc (direction copy): `J = p^y_{d−1}(ω,ℓ)/jDen`, `G ≡ 1`, no pmf factor, no J_P.
10. **Predicate.** Kind ENV: `F_k = +∞`, `I_k` skipped, **`R_k` (α ≥ 0.2 at x_{k−1}) mandatory**, `D_k` as usual. NEE-env is forced like any NEE light vertex.
11. **Temporal changes.** Rotation = "moved" (1 shadow ray to ∞ per NEE-env sample; BSDF-env: re-lookup only). Strength/tint = radiometric (no rays; pmf may change → J_P for all NEE-terminated samples). **Map swap or importance-resolution change = config change → history reset.** The importance table is never rebuilt for rotation/strength/tint.
12. **Background pixels** (no primary hit): no reservoir; `L = [visibleToCamera]·L_env(ω_cam)` added outside the reservoir with weight 1; never accepted as spatial partner or temporal source.
13. **Milestones.** Env display in M1, bridge + C0p in M2, **new M3c "Environment lighting (PT)"**, env cases in M4 (spatial), M5 (temporal), M6 (RIS-NEE tiles, Mode B), M8 (production encoding).

---

## 1. Cycles parity (what Cycles 5.1.2 computes)

### 1.1 Equirectangular mapping of the Environment Texture node

**Code [SOURCE].**
- `K/svm/image.h:234-268` `svm_node_tex_environment`: `co = safe_normalize(co)`; projection 0 → `uv = direction_to_equirectangular(co)`; `svm_image_texture(kg, id, uv.x, uv.y, flags)`. No `1−v`.
- `K/camera/projection.h:19-29,40-48`: `direction_to_equirectangular(dir) = direction_to_equirectangular_range(dir, (−2π, π, −π, π))` with `u = (atan2f(dir.y, dir.x) − range.y)/range.x`, `v = (acosf(dir.z/len(dir)) − range.w)/range.z`. Hence
  ```
  u = (atan2(b.y, b.x) − π) / (−2π) = 0.5 − atan2(b.y, b.x)/(2π)
  v = (acos(b.z) − π) / (−π)        = 1 − acos(b.z)/π
  ```
- Inverse `equirectangular_to_direction(u,v)`: `φ = −2πu + π`, `θ = −πv + π`, `spherical_to_direction(θ, φ) = (sinθ cosφ, sinθ sinφ, cosθ)` (`U/projection.h:19-22`).
- Default Vector: `EnvironmentTextureNode` input `SOCKET_IN_POINT(vector, …, LINK_POSITION)` (`H/scene/shader_nodes.cpp:529`); for the world, `shader_setup_from_background` sets `sd->P = ray_D` (`K/geom/shader_data.h:376`). `TextureCoordinate.Generated` in a background shader compiles to `NODE_GEOM_P`, i.e. also `ray_D` (`H/scene/shader_nodes.cpp:4165-4170`). So the lookup vector is the **world-space ray direction** (Blender Z-up).

**Axes [INFERENCE from the formulas].** In Blender world: `u = 0.5` ↔ +X_b, `u = 0.25` ↔ +Y_b, `u = 0.75` ↔ −Y_b, `u ∈ {0,1}` ↔ −X_b; `v = 1` ↔ +Z_b (zenith), `v = 0` ↔ −Z_b. Moving right in the image from the centre turns from +X_b toward −Y_b, which is "to the right" for an observer facing +X_b with Z up: the image is not mirrored.

**Image v orientation [SOURCE + INFERENCE].** Blender ImBuf rows are bottom-up; Cycles uploads them unchanged (`H/blender/image.cpp:92-129` memcpy) and Metal samples `float2(x,y)` with y = 0 at row 0 (`K/device/metal/context_begin.h:42`). So `v = 0` is the bottom row of the picture (nadir), as `equirectangular_to_direction(·, 0)` → θ = π requires.

**Rotation (Mapping node) [SOURCE].** `K/svm/mapping_util.h:15-35`: POINT: `transform_direction(euler_to_transform(rot), vector·scale) + location`. With only `rot.z = γ` (`U/transform.h:201-225`): `R_z(γ)·v`, i.e. `(cγ vx − sγ vy, sγ vx + cγ vy, vz)`. The lookup vector is rotated by +γ, so the **visible environment rotates by −γ** about +Z (clockwise seen from above). TEXTURE type would use the transpose. The bridge uses POINT, `location = 0`, `scale = 1`.

**Composite glTF → texture coordinate [INFERENCE, exact].** `C = R_x(+90°)`: `(x,y,z)_g → (x, −z, y)_b` (plan §1.2). Then `b = R_z(γ)·C·d`:
```
b.x = cγ·d.x + sγ·d.z
b.y = sγ·d.x − cγ·d.z
b.z = d.y
```
Equivalently `b = C·R_y(γ)·d` (`C R_y(γ) C⁻¹ = R_z(γ)`), i.e. a rotation about glTF +Y. Inverse: `d = C⁻¹ R_z(−γ) b`, with `C⁻¹(X,Y,Z)_b = (X, Z, −Y)_g`. For γ = 0: `u = 0.5 + atan2(d.z, d.x)/(2π)`, `v = 1 − acos(d.y)/π`, identical to three.js `equirectUv` (`common.glsl.js:91-99` [SOURCE]).

**Blender node setup the bridge writes [SOURCE for node/RNA names, RNA for defaults].**
```
world = bpy.data.worlds.new("ENV"); scene.world = world;  nt = world.node_tree; nt.nodes.clear()
tc  = nt.nodes.new("ShaderNodeTexCoord")
map = nt.nodes.new("ShaderNodeMapping");  map.vector_type = 'POINT'
      map.inputs["Location"].default_value = (0,0,0); map.inputs["Rotation"].default_value = (0,0,γ); map.inputs["Scale"].default_value = (1,1,1)
env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = img; env.projection = 'EQUIRECTANGULAR'; env.interpolation = 'Linear'
      # env.texture_mapping must stay identity (translation 0, rotation 0, scale 1, mapping X/Y/Z, use_min/max False): assert
[tint: vm = ShaderNodeVectorMath(operation='MULTIPLY', inputs[1] = tint)]      # only when tint != (1,1,1)
bg  = nt.nodes.new("ShaderNodeBackground"); bg.inputs["Strength"].default_value = s
out = nt.nodes.new("ShaderNodeOutputWorld"); out.target = 'ALL'; out.is_active_output = True
links: tc.Generated→map.Vector; map.Vector→env.Vector; env.Color→(vm→)bg.Color; bg.Background→out.Surface
img = bpy.data.images.load(env_exr_path, check_existing=False)
img.colorspace_settings.name = 'Linear Rec.709'; img.alpha_mode = 'NONE'
assert bpy.data.colorspace.working_space == 'Linear Rec.709'
assert sha256(np.asarray(img.pixels, f32).reshape(H,W,4)[..., :3]) == manifest.envTexelHash
```
Background output = `Color × Strength` (`K/svm/closure.h:1530-1541` [SOURCE]). Animated γ and s are set per frame in the render loop (plan: no Blender interpolation), or keyed CONSTANT.

### 1.2 Image handling

| Item | Cycles 5.1.2 behaviour | Source |
|---|---|---|
| Storage of float images | `BlenderImageLoader`: any ImBuf float buffer → `IMAGE_DATA_TYPE_FLOAT4` (1-channel → FLOAT); "Float images are already converted on the Blender side" → colorspace scene_linear | `H/blender/image.cpp:44-90` [SOURCE] |
| Metal format | FLOAT4 → `MTLPixelFormatRGBA32Float`, no mipmaps, max 16384² | `H/device/metal/device_impl.mm:1005-1011,1045-1082` [SOURCE] |
| Half | Cycles half path only for byte→half upsampling. `Image.use_half_precision` (default True [RNA]) only affects Blender GPU textures (EEVEE/viewport: `image_gpu.cc:212,470`, `util_gpu.cc:59-74`), **not Cycles** | [SOURCE] |
| .hdr decode | Blender reads .hdr via OIIO (`imbuf/intern/format_hdr.cc`); OIIO `rgbe2float`: `e ≠ 0 → byte·2^(e−136)` (table `ldexpf(1, e−136)`), `e = 0 → 0`; comment: deliberately not Ward's `(byte+0.5)` | OIIO `hdrinput.cpp:119-202` [SOURCE] |
| three.js HDRLoader | `scale = 2^(e−128)/255` → ×256/255 = **+0.392 %**; HalfType clamps to 65504 | `HDRLoader.js:358-376` [SOURCE] |
| Colorspace | Env node default colorspace auto; bridge sets `'Linear Rec.709'` (= working space `'Linear Rec.709'` [RNA]) → identity. `'Non-Color'` is also identity (is_data) | [RNA], `image.cpp:79-80` |
| Interpolation | Env node default `'Linear'` [RNA]; `image_params()` sets `extension = EXTENSION_REPEAT` always | `shader_nodes.cpp:520,550-558` [SOURCE] |
| Filtering | GPU: non-cubic → hardware `tex.sample(metal_samplers[sid], (x,y))`; `sid = extension + 4` for non-closest → `sampler(address::repeat, filter::linear)` **on both axes** | `K/device/gpu/image.h:86-117`, `K/device/metal/compat.h:387-396`, `device_impl.mm:1017-1021` [SOURCE] |
| Seams | u: periodic (repeat) — correct wrap. v: also repeat → at `v ∈ [0, 0.5/H)` the bilinear blends nadir row with **zenith** row, and vice versa. This is a Cycles quirk; replicate it | [INFERENCE from the sampler] |
| Poles | `atan2(0,0) = 0` → u = 0.5 exactly at the poles; `acosf` without clamp (NaN if `|z|>1`; practically never after `safe_normalize`) | [SOURCE] |
| Texture limit | `texture_limit_render` default `'OFF'` [RNA]; applies only with simplify; pin `use_simplify = False` | `H/scene/image.cpp:474-544` [SOURCE] |
| NaN scrub | None in the Blender loader path (`file_load_image` has none) | [SOURCE] |

### 1.3 Radiance, camera rays, BSDF escapes, NEE

**Camera rays that miss [SOURCE].** Camera state flag `PATH_RAY_CAMERA | PATH_RAY_MIS_SKIP | PATH_RAY_TRANSPARENT_BACKGROUND` (`K/integrator/path_state.h:59`). `integrate_background` (`K/integrator/shade_background.h:64-125`) evaluates the world shader and multiplies by `light_sample_mis_weight_forward_background`, which returns 1 when `PATH_RAY_MIS_SKIP` (`K/light/sample.h:556-563`). With `film_transparent = False` the background is written to the combined pass. Transparent (alpha) crossings keep all flags (`path_state.h:118-139`), so the camera still sees the env through cutouts with weight 1. Camera visibility: `world.cycles_visibility.camera` → `SHADER_EXCLUDE_CAMERA` (`H/blender/shader.cpp:1728-1740`; `K/light/common.h:74-91`).

**BSDF rays that escape [SOURCE].** Same function; if `kernel_data.background.use_mis` and not MIS_SKIP:
```
pdf  = background_light_pdf(kg, ray_P, ray_D)                       // solid angle, map method only here
pdf *= use_light_tree ? light_tree_pdf(kg, P, N, dt, path_flag, 0, light, …) : distribution_pdf_lights
w    = power_heuristic(mis_ray_pdf, pdf)                             // sample.h:322-337
```
`mis_ray_pdf` is the all-closure BSDF pdf of the sampled direction. After a singular bounce `PATH_RAY_MIS_SKIP` is set (`path_state.h:216`) → weight 1. Bounce accounting: the miss after the last allowed bounce still shades the background (`intersect_closest.h:292-296`), consistent with plan `d ≤ maxBounces+2`.

**NEE of the background [SOURCE].** `K/light/light.h:149-157`: `D = −background_light_sample(kg, P, rand, &ls->pdf)`, `ls->t = FLT_MAX`, `eval_fac = 1`. With no portals and no Sky-Texture sun, `map_weight = 1`, `portal_weight = sun_weight = 0` (`H/scene/light.cpp:995-1072`), so `background_light_sample = background_map_sample`. NEE contribution `bsdf_eval · L_bg · eval_fac/ls.pdf · power_heuristic(ls.pdf, bsdf_pdf)` (`K/integrator/shade_surface.h:398-402`), `ls.pdf` including the selection pdf. Shadow ray to FLT_MAX; world `max_bounces` gates only NEE selection (`light.h:25-29,240`).

**Importance map [SOURCE].**
- Enable rule (`H/scene/light.cpp:280-300`): the background light is enabled iff `has_portal || (use_mis && shader->has_surface_spatial_varying)`. A world with only a constant Background colour is **not** spatially varying → **no NEE, BSDF-only with weight 1**. An Environment Texture fed by TexCoord/Geometry (or its implicit position input) is spatially varying (`TextureCoordinateNode/GeometryNode::has_spatial_varying() = true`, `shader_nodes.h:979-997`; `svm.cpp:463-466`) [SOURCE + INFERENCE for the implicit-input case].
- Resolution (`light.cpp:1074-1089`): MANUAL → `(R, R/2)` with `R = sample_map_resolution`; AUTOMATIC (`map_resolution = 0`, `blender/light.cpp:156-161`) → max width/height over Environment Texture nodes; else 1024×512.
- Bake (`light.cpp:29-65`, `K/bake/bake.h:55-77`): the world shader is evaluated at `u = (x+0.5)/W_m, v = (y+0.5)/H_m`, direction `equirectangular_to_direction(u,v)`, i.e. at map-cell centres (with Linear filtering, at a texel centre when `W_m = W`).
- `background_cdf` (`light.cpp:927-969`): `f_ij = average(|rgb|)·sin(π(i+0.5)/H_m)` (row-centre sinθ; `average` = (r+g+b)/3), conditional CDFs per row (`.x` = f, `.y` = CDF, last entry `.x` = row total `Σ_j f_ij/W_m`), marginal over row totals (`light.cpp:1097-1143`).
- Sampling (`K/light/background.h:20-104`): lower-bound search on the marginal, `v = (i + inverse_lerp(cdf_i, cdf_{i+1}, ξ_y))/H_m`; same for u in row i; within-cell uniform. **pdf**
  ```
  pdf_σ = f_ij / (2π²·sin(π v) · rowTotal_i · total) · rowTotal_i  =  f_ij / (2π² sin(πv) · mean(f))
  ```
  i.e. `p_uv = f_ij/mean(f)` piecewise constant and `pdf_σ = p_uv/(2π² sinθ)` with sinθ **at the continuous v** (`dω = 2π² sinθ du dv`). `sin θ == 0 → pdf = 0` (`:95-100,118-120`).
- Average radiance for selection: `map_average_radiance = cdf_total·π/2` (`light.cpp:1125-1135`), used by the light tree as `strength·avg_radiance·π` (`H/scene/light_tree.cpp:223-232`).

**Selection probability [SOURCE].** Light tree on by default (`use_light_tree = True` [RNA]); the background sits in the distant-light subtree, and its pmf depends on the shading point. Without the tree: `distribution_pdf_lights = 1/num_lights` (×0.5 if emissive triangles exist) (`light.cpp:495-512`).

**Expectation invariance [INFERENCE].** Map resolution, weights, selection pmf, power vs balance heuristic, NEE on/off (sampling_method NONE) and our different importance map all change only variance, provided (a) the MIS weights of NEE and BSDF sum to 1 wherever the integrand is non-zero, and (b) each pdf used is the true density of its sampler. Cycles satisfies both (NONE: no NEE, BSDF weight 1). Our design must satisfy both (§2.4, §2.5).

### 1.4 Settings to pin (Blender side)

| Setting | Value | Why |
|---|---|---|
| `scene.world` | None when `world.type == 'none'`; else the node setup of §1.1 | plan §7.5 currently "world None" |
| `world.cycles.sampling_method` | `'AUTOMATIC'` (map res = image res). Variant runs: `'NONE'` = BSDF-only env (still unbiased) | [RNA] enum NONE/AUTOMATIC/MANUAL |
| `world.cycles.sample_map_resolution` | leave 1024 (unused with AUTOMATIC); record resolved map res = image res in manifest | |
| `world.cycles.max_bounces` | 1024 | NEE gating only |
| `world.cycles.is_caustics_light` | False | MNEE off |
| `world.cycles_visibility` | camera = `visibleToCamera`; diffuse = glossy = transmission = scatter = shadow = True | [RNA] fields |
| `scene.render.film_transparent` | False | |
| portals | none (`light.cycles.is_portal` False on all lights; assert) | |
| `scene.cycles.use_light_tree` | True (pin explicitly; only noise) | [RNA] default True |
| `scene.cycles.texture_limit_render` / `render.use_simplify` | `'OFF'` / False | |
| `bpy.data.colorspace.working_space` | assert `'Linear Rec.709'` | [RNA] Blender 5 working space |
| Env node | projection EQUIRECTANGULAR, interpolation Linear, texture_mapping identity | |
| Image | loaded from the exported float32 ZIP RGB EXR; colorspace `'Linear Rec.709'`; alpha_mode `'NONE'`; pixel hash asserted | |
| `light_sampling_threshold` | 0 (already in plan) | with the light tree, `light_inv_rr_threshold = 0` anyway (`H/scene/integrator.cpp:317-323`) |

### 1.5 A sun inside an HDRI

A very bright small region is handled by importance sampling in both renderers; any residual is **variance, not bias**. Cycles' map at full image res concentrates NEE on the sun; BSDF hits on the sun get power-heuristic weight ≈ 0; glossy reflections of the sun are BSDF-dominated. The expensive paths are sun → specular chain → diffuse (caustics), which NEE cannot sample: heavy-tailed replicate distributions in **both** renderers. Validation consequence: put sun-HDRIs in a separate heavy-tail tier (§5.4), not in the tight gates.

---

## 2. Our renderer

### 2.1 Loader (Worker, transfer lists)

- **.hdr (Radiance RGBE): own decoder** (~100 lines TS): header `#?RADIANCE`/`#?RGBE`, `FORMAT=32-bit_rle_rgbe` only (reject `xyze`), resolution line **`-Y H +X` only** (reject the other 7 orientations), new-style RLE, old-style/flat fallback like OIIO. Pixel: `e = 0 → 0`, else `byte · 2^(e−136)` computed in f64 then stored into `Float32Array` (exact; the product is representable, including subnormals). `EXPOSURE=` headers are ignored (as OIIO), logged.
- **.exr: three.js `EXRLoader` as parser** with `type = FloatType` (half→float exact). Accept lossless codecs only in validation (NONE, RLE, ZIPS, ZIP, PIZ); B44/DWA allowed interactively with a "lossy codec" badge. Reject multi-part/deep/tiled-mip EXRs, data window ≠ display window. Warn if `chromaticities` ≠ Rec.709. EXRLoader emits rows **bottom-up** (`EXRLoader.js:2453,2493`: `outLineOffset = (height − 1 − y)·width`) [SOURCE].
- **Canonical in-memory form:** `Float32Array` RGB, W×H, **rows bottom-up** (row 0 = nadir), alpha dropped (validation requires alpha ≡ 1 or no alpha). Negative/NaN/Inf texels: NaN/Inf → reject in validation, replace by 0 with a warning interactively; negatives kept (Cycles renders them), importance uses `|·|`.
- Aspect: any; equirect implied. Warn when W ≠ 2H.

### 2.2 GPU format, memory, caps

| Map | rgba32float | rgba16float | importance tables at `W_m = min(pow2 ≤ W, cap)` (8 B/cell) |
|---|---|---|---|
| 1k (1024×512) | 8 MiB | 4 MiB | 4 MiB |
| 2k | 32 MiB | 16 MiB | 16 MiB |
| 4k | 128 MiB | 64 MiB | 64 MiB (cap 4096) / 16 MiB (cap 2048) |
| 8k (8192×4096) | 512 MiB | 256 MiB | 64 MiB (cap 4096) |
| 16k | 2 GiB | 1 GiB | — (reject; also the 16384 texture limit) |

Policy:
- **Validation:** `rgba32float`, native resolution, no downsampling, **W ≤ 8192**; importance cap `W_m ≤ 4096`. rgba16float is **forbidden** in validation (overflow above 65504 for suns, 2⁻¹¹ relative rounding) and asserted by T16.
- **Interactive:** `rgba32float` up to 4k; inputs > 4k are 2×2 box-downsampled in linear float (energy-preserving mean) with a warning; optional "low memory" `rgba16float` (clamp 65504, badge "biased"). Importance cap `W_m ≤ 2048`.
- A downsampled texture is a different environment. If a downsampled scene is ever exported, the bridge exports the downsampled texels (the export is always "what we render with").
- Upload: `queue.writeTexture` in row bands ≤ 64 MB, inside `pushErrorScope('out-of-memory')`.

### 2.3 Mapping and radiance in WGSL (normative)

```wgsl
// frame uniform (cur and prev copies): env.cs = vec2(cos γ, sin γ), γ = Blender Mapping rotation Z (POINT);
// env.scale = strength * tint (rgb);  texEnv: rgba32float, rows bottom-up;  sEnv: repeat/repeat, linear/linear, 1 mip
fn envBlender(d: vec3f, cs: vec2f) -> vec3f {                 // b = R_z(γ)·C·d, C = R_x(+90°)
  return vec3f(cs.x * d.x + cs.y * d.z, cs.y * d.x - cs.x * d.z, d.y);
}
fn envUV(d: vec3f, cs: vec2f) -> vec2f {                        // == Cycles direction_to_equirectangular
  let b = envBlender(d, cs);
  let u = (atan2(b.y, b.x) - PI) / (-2.0 * PI);
  let v = (acos(clamp(b.z, -1.0, 1.0)) - PI) / (-PI);           // d unit ⇒ |b| = 1 (clamp only for NaN safety)
  return vec2f(u, v);
}
fn envDir(uv: vec2f, cs: vec2f) -> vec3f {                      // == Cycles equirectangular_to_direction, then C⁻¹·R_z(−γ)
  let phi = -2.0 * PI * uv.x + PI;
  let theta = -PI * uv.y + PI;
  let st = sin(theta);
  let b = vec3f(st * cos(phi), st * sin(phi), cos(theta));
  let rx = cs.x * b.x + cs.y * b.y;                             // R_z(−γ)·b
  let ry = -cs.y * b.x + cs.x * b.y;
  return vec3f(rx, b.z, -ry);                                   // C⁻¹(X,Y,Z) = (X, Z, −Y)
}
fn envRadiance(uv: vec2f, scale: vec3f) -> vec3f {              // the ONLY env radiance evaluation
  return scale * textureSampleLevel(texEnv, sEnv, uv, 0.0).rgb;
}
```
- NEE-env samples evaluate `envRadiance(uv_sampled)` directly (no direction round trip). BSDF escapes, camera misses and refresh of BSDF-ended samples evaluate `envRadiance(envUV(ω))`. Same texture, sampler and function everywhere.
- The coordinate floats fed to the Metal sampler equal Cycles' (up to atan2/acos ulps), so the hardware bilinear weights, including their fixed-point subtexel quantization, are identical [INFERENCE].

### 2.4 Importance sampling structure (normative)

**Grid.** `W_m = largest power of two ≤ min(W, cap)`, `H_m = W_m/2`, rows `i` bottom-up in v (`v ∈ [i/H_m, (i+1)/H_m)`), columns `j` in u. Requires `log2 W_m ≤ 16`.

**Cell weights (Worker, f64, deterministic, fixed loop order).**
1. Per texel `a[r][c] = (|R|+|G|+|B|)/3` of the **untinted, unscaled** texels (strength and tint excluded, so they never change the table).
2. Exact bilinear-integral kernel per texel footprint: `B[r][c] = Σ_{dr,dc∈{−1,0,1}} k(dr)k(dc)·a[r+dr][c+dc]` with `k = (1/8, 6/8, 1/8)`; u periodic; **v periodic too** (Cycles repeat quirk: row −1 ≡ row H−1). `B` is the integral of the bilinear reconstruction over the texel footprint, so `B > 0` wherever the reconstruction is positive in that footprint.
3. Cell sum: if `W` is a multiple of `W_m` (all power-of-two HDRIs), `B̂_ij = mean of B over the k×k block`; otherwise area-weighted overlap of texel footprints with the cell (conservative: any footprint that overlaps the cell and has positive integral contributes).
4. Row solid-angle factor: `s̄_i = (H_m/π)·(cos(π i/H_m) − cos(π(i+1)/H_m))` (row mean of sin(πv) = sinθ; > 0 for every row including pole rows).
5. `w_ij = s̄_i·B̂_ij`, then **floors**: `w_ij ← max(w_ij, 2⁻¹⁰·mean_all(w))`; row totals `R_i = Σ_j w_ij`; within each row `w_ij ← max(w_ij, 2⁻¹⁰·R_i/W_m)` (recompute `R_i`); marginal `R_i ← max(R_i, 2⁻¹⁰·mean(R))`. Every cell then has bucket-relative mass ≥ 2⁻¹⁰, hence a u16 threshold ≥ 64: **every cell has positive realized probability**, so NEE alone covers the sphere.

**Two-level integer alias (Vose), per table of n = 2^m entries.**
- Scaled `p_k = w_k·n/Σw`. Vose small/large lists in f64; for a small bucket `s` set `q_s = clamp(round(p_s·65536), 0, 65535)`, `alias_s = l`, and debit the large bucket by exactly `(65536 − q_s)/65536`. Leftover buckets: `alias = self`, `q = 65535` (either branch returns self).
- Packed entry `u32 = alias(16) | q(16)`. Tables: `rowAlias[H_m]`, `colAlias[H_m·W_m]`.
- **Realized probabilities (exact rationals, f64):** `P(k) = [Σ_b (b == k)·q_b + (alias_b == k)·(65536 − q_b)] / (65536·n)`. Then `pdfUV[i·W_m + j] = P_row(i)·P_col(j|i)·W_m·H_m` stored as f32. **pdfUV is derived from the stored integers, never from the target weights.**

**Sampling (WGSL, 3 u32 hashes h0, h1, h2 from the vertex's NEE dims).**
```wgsl
let i0 = h0 >> (32u - log2Hm);  let er = rowAlias[i0];
let i  = select(er >> 16u, i0, (h0 & 0xFFFFu) < (er & 0xFFFFu));
let j0 = h1 >> (32u - log2Wm);  let ec = colAlias[i * Wm + j0];
let j  = select(ec >> 16u, j0, (h1 & 0xFFFFu) < (ec & 0xFFFFu));
let du16 = h2 >> 16u;  let dv16 = h2 & 0xFFFFu;                      // stored losslessly in the reservoir
let uv = vec2f((f32(j) + (f32(du16) + 0.5) / 65536.0) / f32(Wm),
               (f32(i) + (f32(dv16) + 0.5) / 65536.0) / f32(Hm));
```
Top bits (bucket) and low 16 bits (threshold) of the same hash are independent because `log2 n ≤ 16`. The in-cell offsets lie on a 2⁻¹⁶ lattice (1/65536 of a cell), which never touches a cell edge or a pole.

**pdf (one function, env-conditional; `p1` multiplies by `pmf[ENV]`).**
```wgsl
fn envPdfSA(uv: vec2f) -> f32 {                  // solid-angle density given "env chosen"
  let s = sin(PI * min(uv.y, 1.0 - uv.y));       // sinθ; min() keeps precision near v = 1
  if (s < 1e-6) { return 0.0; }                  // pole cap: NEE samples there are rejected (F = 0), BSDF gets ω2 = 1
  let j = min(u32(uv.x * f32(Wm)), Wm - 1u);
  let i = min(u32(uv.y * f32(Hm)), Hm - 1u);
  return pdfUV[i * Wm + j] / (2.0 * PI * PI * s);
}
fn p1Env(uv: vec2f) -> f32 { return pmf[ENV_IDX] * envPdfSA(uv); }
```
- NEE rejects its own sample when `s < 1e-6` (the same test), so the pole cap (≈ 6·10⁻¹² sr) is covered by BSDF sampling only, and the partition stays exact.
- **Bilinear vs piecewise-constant mismatch:** irrelevant for unbiasedness, because the pdf is exactly the sampler's density and is positive everywhere (floor). The kernel only makes the density track the bilinear radiance (lower variance).
- Residual inexactness: the direction round trip `envUV(envDir(uv))` for BSDF escapes may land in the neighbouring cell on cell edges, or differ near the poles because `acos` is ill-conditioned (≈ 3·10⁻⁴ rad at |z|→1 in f32). The MIS partition then fails only on a set of measure ≈ 10⁻⁷ of the sphere [INFERENCE]. Accepted and documented; the test ENV-U6 bounds it.

**Build cost [INFERENCE, measure in M3c].** Kernel pass O(W·H) (8k: 33 M texels, ≈ 0.3 s JS), weights/alias/realized pass O(W_m·H_m) (2k: 2.1 M cells ≈ 50–100 ms; 4k: 8.4 M ≈ 0.2–0.4 s). Single Worker, f64, typed arrays; results transferred. A GPU build is unnecessary.

### 2.5 Selection probability of the env

- `L̄_env = strength·(1/4π)·Σ_{r,c} lum(tint ⊙ texel_rc)·ΔΩ_r`, with `ΔΩ_r = (2π/W)(cos(πr/H) − cos(π(r+1)/H))` (rows bottom-up).
- `Φ_env = 4π²·R_s²·L̄_env`, with `R_s` = bounding-sphere radius of the static triangles (fixed per scene load, so camera motion never changes the pmf). The same family as the plan's sun proxy `lum(E)·π·R_s²`: both are flux through the scene's bounding sphere. pbrt-v4 uses the same `4π²R²L̄` for infinite lights [UNVERIFIED detail].
- Normalise all proxies in stable-id order (f64). If other emitters exist: `P(env) = clamp(Φ_env/ΣΦ, 0.1, 0.9)`, and the others are rescaled by `(1 − P(env))/Σ_others`. If the env is the only emitter, `P(env) = 1`. Build the alias from the final pmf. Rebuild only when lights or env strength/tint change; rotation never changes the pmf.
- **Consistency requirement:** the exact same `pmf[ENV_IDX]` (the frame's f32 value) appears in NEE `q`, `ω1`, `ω2` (BSDF escape), RIS-NEE ratios, shift re-evaluation, refresh, and `J_P`. Any function of the frame's state is unbiased; mixing two functions is not.

### 2.6 Cost of evaluating the env (for gap-perf)

Per NEE-env sample: 2 packed alias loads + 1 pdf load + 1 bilinear RGBA32F fetch + 1 shadow ray to ∞. Per BSDF escape: 2 atan2/acos + 1 fetch + 1 pdf load (MIS). Env radiance fetches are incoherent (4 × 16 B per lookup) [INFERENCE].

---

## 3. ReSTIR PT integration

### 3.1 Path space, measure and candidates

- **Env vertex at infinity.** `x_d = (ENV, ω)` with `ω ∈ S²` parameterised in **env-local** coordinates `(u,v)`, measure = solid angle. No position, no normal, no geometry term.
- **Measure extension (gap-light §3.1):** `μ = Σ_{area∪tri} δ_L⊗A_L + Σ_delta δ_{(L,c_L)} + δ_ENV⊗σ_env`. At a shading vertex x, `g(ENV, ω) = f_cos(ω_o, ω)·V_∞(x, ω)·Λ_ENV(ω)` with `Λ_ENV(ω) = L_env(ω)`. `∫ g dμ` includes `∫_{S²} f_cos L_env V dω`.
- **NEE_ENV candidate** (technique NEE, endpoint ENV) at `x_{d−1}`:
  ```
  q     = pmf[ENV]·pdf_σ(uv)                       // = p1 (already solid angle)
  F     = T·ω1·f_cos(ω_o, ω)·L_env(uv)·V_∞(x_{d−1}, ω) / q ,   ω = envDir(uv, cs)
  ω1    = M(B)·p1 / (M(B)·p1 + p2(ω))              // p2 = marginal BSDF pdf over non-delta lobes; ω1 = 1 if p2 ≡ 0
  RIS-NEE ratio  r_i = lum(f_cos·L_env(uv_i)) / (pmf[ENV]·pdf_σ(uv_i))   (same measure for p̂ and q)
  ```
- **BSDF_ENV candidate** (technique 3) when the BSDF ray from `x_{d−1}` escapes (after any Mode-B crossings, which are their own candidates):
  ```
  F  = T_prefix·(f_ℓ cos/p(ω,ℓ))·L_env(envUV(ω))·ω2,   ω2 = p2/(M(B)·p1Env(envUV(ω)) + p2)   (ω2 = 1 after a delta lobe)
  ```
  RR: survived tests 1..d−1 (as BSDF-hit candidates). NEE_ENV: 1..d−2 (plan rule 11).
- **Disjoint domains** `(d, NEE, ENV)` and `(d, BSDF_ENV)` join the streaming RIS with m = 1 (plan §3.2). Env and analytic lights never share a domain.
- **Length-1** (camera sees env, also through alpha cutouts and Mode-B light crossings of the camera ray): outside the reservoir, weight 1, times `visibleToCamera`.

### 3.2 Reservoir encoding

| Field | NEE_ENV endpoint (class L/N1/D-NEE) | BSDF_ENV endpoint as rc, k = d (class E) | BSDF escape after rc, k < d (class B1/D-BSDF) |
|---|---|---|---|
| flags | technique 0, endpoint 7, isDelta 0 | technique 3, endpoint 7 | technique 3, endpoint 7 |
| endpointId | `ENV_ID` (stable) | `ENV_ID` | `ENV_ID` |
| rcA / rcB / rcBary | when k = d: `rcA = LIGHT\|ENV_IDX`, `rcB = (i<<16)\|j`, `rcBary = (du16<<16)\|dv16` (**lossless**) | `rcA = ENV_RC` sentinel | x_k (inst, prim, bary) |
| rcWi | unused (k = d); for N1 (rc = x_{d−1}) the NEE point is stored in the suffix cache (below) | world `ω_{d−1}` (f32×3 validation; oct16×2 production, quantised **before** F and jDen) | `ω_k` (world) |
| jDen | k = d: unused (J = 1 / J_P) | `p^x_{d−1}(ω, ℓ_{d−1})` (joint) | Eq. 2 product at k |
| rcRadiance | k = d: none (recomputed); N1: `Λ_x = L_env(uv)` with MIS and 1/q divided out | none (recomputed) | full suffix incl. `L_env·ω2` |
| suffix cache (§1.9) | `(ENV_IDX, i, j, du16, dv16)` for refresh | — | `x_{d−1}`, `ω_{d−1}`, `β_s`, `(t_occ = +∞, p2)` (escape flag) |
| aux p1 | not needed (shading-point independent; recompute) | not needed | not needed |

Quantisation: the NEE_ENV encoding is exact (these are the sampler's own integers). Only the BSDF_ENV production direction (oct16) is quantised, before computing F and jDen (plan rule 14), and is covered by the M8 Gate-3 δ check.

### 3.3 Shifts, Jacobians and predicate

**Visibility to infinity.** `visibleInf(a, ω)`: Wächter–Binder offset of `a` toward ω, `t_max = +∞`, triangles only, MASK cutout, glass occludes, analytic lights never occlude, a's primitive excluded. Same module as `visible(a,b)`.

**Case (f): forced NEE reconnection to the env (k = d).** `y_{d−1}` connects to the *same env-local* `(i,j,du,dv)`; within a frame that is the same world direction. Evaluate
`F(ȳ) = T^y·ω1^y·f_cos(y_{d−1}; ω_o^y, ω)·L_env(uv)·V_∞(y_{d−1}, ω)/q`, with `p1 = pmf[ENV]·pdf_σ(uv)` (unchanged) and `p2^y` recomputed at y_{d−1}.
**J = 1 spatially** [INFERENCE, verified]: the NEE dims map `(u_sel, h0, h1, h2) → (ENV, i, j, du, dv)` through a sampler that does not depend on the shading vertex, so copying the endpoint is the identity in PSS (same argument as area-uniform lights, gap-light §5.7; gap-rc (f)). Pre-rc pair (y_{d−2}, y_{d−1}) uses `EV_RECONNECT_NEE` with `p̄^y_{d−1}(ω)`.

**Case (e): BSDF escape as rc (k = d), direction copy.** `y_{d−1}` (end of replay) continues along the stored **world** ω with the copied lobe ℓ_{d−1}:
```
F(ȳ) = T^y_prefix · 1_supp(ℓ, V^y, ω)·f_ℓ(y_{d−1}; ω_o^y, ω)|cos|/p^y(ω,ℓ) · L_env(envUV(ω)) · ω2^y · V_∞(y_{d−1}, ω)
ω2^y = p2^y(ω)/(M(B)·p1Env(envUV(ω)) + p2^y(ω))
J    = p^y_{d−1}(ω, ℓ_{d−1}) / jDen ,   jDen = p^x_{d−1}(ω, ℓ_{d−1})      // joint pdfs; G ≡ 1; p_k := 1; no pmf; no J_P
```
This is plan rule 3 Eq. 2 with k = d, `G(y_{k−1}→x_k) := 1` and `ω′ := ω` (no `normalize(x_k − y_{k−1})`). Falcor `Shift.slang:403-432` and gris §6.8 agree (`p_{y}(rcWi)/p_{x}(rcWi)`, no geometry term); C2-Renderer sets G = 1 because the infinite distances cancel (reference-code §1.9). Write-back on selection: `jDen ← p^y_{d−1}(ω,ℓ)`.

**Deeper cases (k < d) with an env end.** Standard Eq. 2 at x_k. Suffix directions are copied in world space, so the suffix is identical. N1 with env NEE at x_k: `J = p_{k−1}·G/jDen` with `p_k := 1`; F re-evaluates `f(x_k; ω_in^y, ω_env)` and `ω1(p1_env, p2^y(x_k; ω_in^y → ω_env))`.

**Predicate `rc.wgsl` (gap-rc §3.3, K_ENV).** For the terminal pair `(x_{d−1}, env)` of a BSDF_ENV candidate: `D` (ℓ_{d−1} non-delta), **`R` α(x_{d−1}, ℓ_{d−1}) ≥ 0.2 mandatory** (the only guard: rayFP = +∞, P-§4.2), `F` = pass, `I` skipped. v1: `G_T` never passes, so the env seen through rough glass always goes to full replay. The pair `(x_{d−2}, x_{d−1})` of a BSDF_ENV candidate is identical to the continuing path's pair d−1. NEE_ENV light vertex: forced, never tested. Precedence: k* = first passing pair; if none: NEE_ENV → k = d (forced), BSDF_ENV → k = ∅ (full replay, J = 1, same d, same technique required).

**Offset checks.**
- O0: replay reaches y_{k−1}; copied-lobe joint pdf `p^y(ω,ℓ) > 0` and `1_supp(ℓ, V^y, ω) = 1`.
- O1: every earlier pair of ȳ fails. The last pre-rc pair uses `EV_RECONNECT` with `p̄^y_{k−1}(ω)` for the env direction (case e), or `EV_RECONNECT_NEE` (case f).
- O2 for case (e): `D ∧ R` at y_{k−1} with the copied lobe (roughness may be texture-varying, so this can fail).
- O3 for k = ∅: no pair passes, **including the terminal env pair**, and the replay ends at the same d by escaping (BSDF_ENV). Hitting geometry means a mismatched technique: fail.
- An occluded `visibleInf(y_{k−1}, ω)`: a defined shift with F = 0 (status "occluded"), like the occluded-reconnection case.

**Primary vertex.** k = 2 means `x_{k−1} = x₁`: env rc straight from the primary hit (the ReSTIR-DI-for-env case) when x₁'s sampled lobe is rough. A glossy floor with α < 0.2 → BSDF_ENV goes to full replay; NEE_ENV is still forced at the env.

### 3.4 Temporal reuse with a changing env

**Which frame the stored direction lives in.**
- NEE_ENV: **env-local** `(i,j,du,dv)`. The sample follows a rotating env, like `(lightId,u,v)` follows a moving area light.
- BSDF_ENV and deep BSDF escapes: **world** ω. It is a BSDF-sampled direction, and rotating the env does not move BSDF samples; only `L_env(ω)` and `p1_env(ω)` change.

**Jacobians.**
- NEE-terminated path ending on the env: `J = J_rc(Eq. 2 at x_k, unchanged) × J_P`, with `J_P = [pmf_t(ENV)·pdf^t_uv(cell)] / [pmf_{t−1}(ENV)·pdf^{t−1}_uv(cell)] · J_M`. Here `J_M = 1` (solid angle in env-local coordinates is rotation-invariant, and env-local sinθ is unchanged), and `pdf_uv` is the same table (map swaps reset history). So **`J_P = pmf_t(ENV)/pmf_{t−1}(ENV)`**, which is exactly plan rule 3 ("× J_P for every NEE-terminated path").
- BSDF-ended paths: no J_P (plan rule 3); J unchanged.

**Change taxonomy (CPU diff of env cur vs prev → flags).**

| Event | Classification | Forward shift / refresh under S_t | Inverse under S_{t−1} | Rays |
|---|---|---|---|---|
| Rotation γ changed | env **moved** | NEE_ENV: `ω_t = envDir(uv, cs_t)`, new `V_∞`, `L_env(uv)` unchanged, `p1` unchanged. BSDF_ENV/escapes: `L = envRadiance(envUV(ω, cs_t))`, `ω2_t` with `p1Env(envUV(ω, cs_t))` | same with `cs_{t−1}` | 1 shadow ray to ∞ per NEE_ENV sample; 0 for BSDF escapes (`t_occ = ∞` is static) |
| Strength or tint changed | radiometric | `L` scaled; pmf rebuilt, so `J_P ≠ 1` for **all** NEE-terminated samples (all lights) | with `scale_{t−1}`, `pmf_{t−1}` | 0 |
| Map swap, importance resolution change, projection change | **config change → history reset** (plan rule 9) | — | — | — |
| visibleToCamera toggled | length-1 only | nothing | nothing | 0 |
| Env added/removed | light add/remove with `ENV_ID` (gap-temporal §7) | removed: NEE_ENV history undefined (w̃_p = 0); BSDF escapes defined (L = 0) | added: canonical NEE_ENV π_p = 0; BSDF escapes use L_{t−1} = 0 | 0 |

- **E_{t−1} additions:** `envPrev = {cs_{t−1}, scale_{t−1}, pmf_{t−1}[ENV]}` in `lightsPrev` (records); the texture and tables are shared because they cannot change without a reset.
- **Refresh (`suffix_refresh`):** classes L/N1/D-NEE with endpoint ENV use the stored `(i,j,du,dv)` (gap-temporal option (ii)); classes B1/D-BSDF/E with the escape flag need no BVH ray. Idempotence test: 3-refresh with env unchanged → ΔF ≤ 1e-6.

### 3.5 Mode A / A′ / B interplay

The env is not an analytic light. In every mode BSDF rays that escape evaluate the env with MIS (ω2), and NEE samples it (ω1); it is never NEE-only. The Blender side is the same in all modes: world `sampling_method = AUTOMATIC`, and per-light MIS flags do not affect the world. In Mode B, an escaping ray that crosses analytic area lights yields one `BSDF_ANALYTIC` candidate per crossing plus one `BSDF_ENV` candidate; lights never attenuate the env.

### 3.6 Background pixels, motion vectors, disocclusion

- `primary`: miss → V-buffer `primId = NONE`, `thr` undefined, length-1 buffer `+= visibleToCamera·envRadiance(envUV(ω_cam))` (jittered ray, same as Cycles' box filter).
- `initial`: skip (no reservoir; W = 0, c = 0).
- Temporal (T1): `q′` on a previous-frame miss → no temporal candidate; a current miss → skip.
- Spatial `pair_accept`: a partner or canonical without a hit is never accepted (G-buffer-only rule; not "FAILED").
- Dual MV for misses: rotation-only reprojection of the direction (for the denoiser only).
- Denoiser: background passed through untouched (albedo = 1, excluded from à-trous and history).

---

## 4. WebGPU specifics

- **Bindings.** `texEnv` (rgba32float, `sampleType: 'float'`, legal with `float32-filterable`) and `sEnv` in G0 (static per scene), visible to `primary`, `initial`, `suffix_refresh`, T1–T4 and all spatial passes. Tables `rowAlias`, `colAlias`, `pdfUV` in the **records** arena (B3) with section offsets in the frame uniform. No extra storage-buffer binding in any pass (plan's ≤ 9/10 budget is unchanged). The env light record (type ENV, `cs`, `scale`, table offsets, `W_m`, `H_m`, `log2`s) sits in `lights cur/prev`.
- **Sampler.** `{addressModeU:'repeat', addressModeV:'repeat', magFilter:'linear', minFilter:'linear', mipmapFilter:'nearest', lodMinClamp:0, lodMaxClamp:0}`. Texture `mipLevelCount = 1`. Always `textureSampleLevel(texEnv, sEnv, uv, 0.0)` (legal in compute).
- **Memory.** §2.2 table. At 1024² validation (0.57 GB reservoirs, ensemble E = 16) plus a 4k env (128 MiB) plus 64 MiB tables ≈ 0.77 GB; 8k: ≈ 1.15 GB. Record in the M0 allocation probe.
- **Build time.** §2.4; the Worker holds the f64 intermediates, transfers `Uint32Array` + `Float32Array`. The map-load UI shows progress; env changes of map/resolution reset history.
- **Numerics.**
  - `sinθ` from `sin(π·min(v, 1−v))` (never `sqrt(1−z²)` mixed with acos).
  - Pole cap `s < 1e-6` on both sampling and pdf.
  - NaN tests are bit tests (plan §1.1).
  - `acos` clamps input.
  - `pdf_σ` max ≈ `pdfUV_max/(2π²·1e-6)` ≈ 5e10 for `pdfUV ≤ 1e6`: finite in f32.
  - F for such samples is tiny, not Inf.
  - `u32(uv.x·W_m)` clamped to `W_m − 1` (u = 1.0 only reachable from `envUV` at −X).
- **RNG layout.** NEE block per vertex becomes `{u_sel, h_l0, h_l1, h_l2}` (4 u32 dims; area lights use l0/l1 as before). NEE dims are never replayed (forced reconnection), so adding l2 only changes the fixed per-vertex stride D (plan §1.7 / T1 update).

---

## 5. Validation

### 5.1 Unit tests (Gate 0)

| ID | Milestone | Test | Pass |
|---|---|---|---|
| ENV-U1 | M1 | RGBE decode of 3 Poly Haven .hdr vs Blender-Python OIIO `ImageInput` read; EXR (ZIP/PIZ, half/float) vs OIIO | SHA-256 of f32 RGB equal (bitwise); row order equal after flip |
| ENV-U2 | M1 | `envUV/envDir` vs an f64 Python port of Cycles `direction_to_equirectangular` + `R_z(γ)C` for 10⁶ dirs × γ ∈ {0, 0.3, −π/2, π} | ≤ 2e-6 in u,v (away from seam/poles); round trip `envUV(envDir(uv))` ≤ 2e-6 |
| ENV-U3 | M3c | integer alias: CPU recompute of realized pmf from stored u16s vs stored pdfUV; `Σ pdfUV/(W_m H_m) = 1 ± 1e-6`; GPU χ² of 10⁸ samples per cell (cells merged to ≥ 50 expected) | exact to 1 ulp; χ² p ≥ 1e-3 in 99/100 seeds |
| ENV-U4 | M3c | support: random dirs incl. seam, pole caps, pole-wrap quirk rows, and a single-bright-texel map: `L_env(ω) > 0 ⇒ p1Env > 0` | 0 violations outside the 1e-6 pole cap |
| ENV-U5 | M3c | `∫ pdf_σ dω` by f64 quadrature (64× supersampled per cell) | 1 ± 1e-5 |
| ENV-U6 | M3c | NEE/BSDF partition at fixed shading points (Lambert, GGX α ∈ {0.2, 0.5}), with `pmf[ENV] ∈ {1, 0.3}`: `E[ω1-NEE] + E[ω2-BSDF] = ∫ f_cos L_env dω` (quadrature) | rel ≤ 1e-4 (z ≤ 4) |
| ENV-U7 | M0/M1 | GPU `textureSampleLevel` vs f64 CPU bilinear-repeat emulation at 10⁵ (u,v) incl. v ∈ [0, 0.5/H] (pole wrap) | records the hardware weight precision; wrap behaviour asserted |
| ENV-U8 | M3c | pmf determinism: rotation-only edit → pmf bitwise unchanged; strength edit → proxy/clamp as specified | bitwise |
| ENV-U9 | M2 | bridge round trip: uploaded texels → EXR → `bpy.data.images.load` → `image.pixels` hash | equal |
| T3-ENV | M4 | invertibility for cases (e), (f), k = ∅ with terminal env pair, k = 2 env-from-primary, N1-env, O0–O3 env branches | LOGIC = 0 over ≥ 10⁷ per case; f64 dual agrees |
| T-ENV-temporal | M5 | refresh idempotence with env; forward/inverse with γ change; J_P with pmf change from env strength; PSS change-of-variables z-test (gap-temporal §9.3-2) with env | as gap-temporal §9 |

### 5.2 Scenes (Stage A: PT vs Cycles and vs analytic; Stage B: ReSTIR vs PT)

- **C0p env-view (orientation calibration), M2 Gate 1-lite with the emission kernel.** Synthetic 512×256 float EXR:
  - 8 colours for the octants (4 longitude quadrants × 2 hemispheres);
  - bright asymmetric markers (value 100): an "L" glyph at `u=0.5, v=0.5` (+X_g), a dot at u=0.25 (−Z_g), two dots at u=0.75 (+Z_g), a bar crossing the seam u=0/1, and markers at v = 0.95 and v = 0.03 (near poles).
  - Cameras: looking +X_g, −Z_g, +Y_g (zenith) at 512², 640×360 and 360×640. γ ∈ {0, +30°, −90°}. visibleToCamera ∈ {true, false}.
  - Pass: marker centroids within 0.1 px of Cycles; TOST as C0b.
  - Plants that must fail: u + 0.5/W, v flip, γ sign flip, C omitted (Z-up used as Y-up).
- **C0q constant-env furnace (M3c).** Constant env L = 1, both as a 64×32 constant texture (NEE on) and as a Cycles constant Background (Cycles BSDF-only). The two Cycles variants must agree (A/A).
  - (a) Lambert sphere ρ = 0.8, b ∈ {0,1,3}: sphere pixels = ρL exactly (convex, no self-view); background = L.
  - (b) Lambert quad seen from above: ρL (upper hemisphere fully visible). Cycles' cos-sampled BSDF-only estimate has zero variance here.
  - (c) V1 GGX sphere F ≡ 1, α ∈ {0.2, 0.5}: pixel = `L·E_ss(μ_o, α)` from an f64 quadrature table, pixel-footprint supersampled.
  - (d) ρ = 1 open box, b = 13 (d ≤ 15): no closed form; PT vs Cycles, ReSTIR vs PT.
- **C0r irradiance/mirror sphere under a real HDRI (M3c), independent of Cycles.**
  - Lambert sphere under `overcast_soil_puresky_1k`: `L_o(x) = ρ/π ∫ L_env(ω) max(0, n·ω) dω`, by f64 quadrature over the bilinear reconstruction (8× texel density). Exact for any b because the sphere is convex.
  - Mirror sphere (V1 GGX α = 0, F ≡ 1): `L_o = L_env(reflect)`, supersampled per pixel.
  - Tests orientation, MIS partition and the delta path.
- **C0s single-texel sun (M3c).** 512×256 map, upper hemisphere 1, lower 0, one texel 10⁴ at elevation 45°. Variants: the texel on the seam (u ≈ 0), and in the top row (pole wrap).
  - Lambert plane (ρ = 0.5): `L_o = ρ/π·E`, with E by quadrature of the bilinear hat × cos⁺ (≤ 1e-6).
  - Run env-NEE on and off (sampling_method NONE on the Cycles side); all four (ours/Cycles × NEE on/off) must match. Record the variance ratio.
- **(xiii) glossy/mirror spheres + HDRI (M3c Stage A; M4 Stage B).** `studio_small_09_1k`, V1 GGX α ∈ {0, 0.05, 0.15, 0.2, 0.3, 0.5}, plus a V2 metal. This exercises:
  - direction copy (α ≥ 0.2, J = p^y/p^x);
  - the roughness guard (α < 0.2 → full replay);
  - delta handling (α = 0, MIS skip).
- **(xiv) Cornell open to sky (ceiling removed), b ∈ {1, 3, 7}.**
  - HDRIs: `overcast_soil_puresky_1k` (tight tier) and `kloofendal_48d_partly_cloudy_puresky_1k` (heavy-tail tier).
  - With and without an interior rect light, which exercises the P(env) clamp and consistency. Modes A and B.
  - After M3b: plus a glass sphere (env caustics, heavy-tail tier only).
- **(v)** is (xiv). **(vi) dynamic**:
  - **ix-h** env rotating 1°/frame (γ track), Lambert + GGX α = 0.3 objects, test frames 1, 10, 25, 40.
  - **ix-i** env strength ×2 at frame 16, tint change at 24, interior rect-light power ×2 at 30 (each changes the pmf → J_P for env), test frames 15, 16, 17, 24, 25, 30, 31.
  - **ix-j** env rotation + fly camera + moving rect light.
  - **ix-k** map swap at frame 20 (asserts the history reset: c_prev = 0 everywhere at 20).
- **E2E-HDR (M7).** Blender loads the *original* Poly Haven .hdr and .exr files (not the exported texels) and we load the same files: (xiv)-lite must pass Stage A. The decoder guard is ENV-U1 (the +0.39 % three.js error is below the 0.5 % global δ).

### 5.3 Planted controls

**Stage A (must fail ≥ 9/10):**
- env strength × 1.0075;
- u + 0.5/W;
- v flip;
- γ sign;
- **sinθ missing in pdf_σ** (C0s/C0r);
- **pdf from target weights (texel-centre `avg·sinθ_row`) while sampling the kernel+floor alias** (C0s, bright-texel edges);
- ω2 without `pmf[ENV]`;
- NEE and BSDF both weight 1 (double count);
- three.js RGBE decode (caught by ENV-U1, bitwise).

**Negative controls (must pass):** remove the floor (still unbiased: BSDF covers p1 = 0 cells with ω2 = 1); power ↔ balance heuristic; importance res 256 vs 4096.

**Stage B:**
- env rc with J := 1 (xiii α ≥ 0.2 → biased);
- J with a spurious `1/t²` or G term → invalid/NaN counters;
- `R` guard removed → must NOT bias (domain restriction applied symmetrically), only more variance: negative control;
- ix-h: skip the NEE_ENV shadow-ray refresh on rotation → bias at shadow edges;
- ix-h: E_{t−1} evaluated with `cs_t` → N1-like bias;
- ix-i: omit J_P on env → N2-like sign (darkening/brightening per pmf ratio).

### 5.4 Tiers and sizing

- Tight tier (δ as plan §7.3): HDRIs with ≤ 12 EV (Poly Haven `evs_cap`), synthetic maps, no specular-to-diffuse sun caustics.
- Heavy-tail tier: sun HDRIs (≥ 19 EV) and glass/mirror + sun. K doubled, a pilot-based SE, and a median-of-means cross-check logged. Failures there are reported but gate only on global δ = 1 %.
- Budget: add Cycles s/4096 spp for C0q, C0r, (xiii), (xiv) to `budget.json` in M3c.

### 5.5 Assets (Poly Haven, **CC0**, no attribution required: polyhaven.com/license [SOURCE])

| Id | 1k .hdr | Size | EV range | Role |
|---|---|---|---|---|
| `overcast_soil_puresky` | `https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/overcast_soil_puresky_1k.hdr` | 1.20 MB | 12 | tight tier (C0r, xiv) |
| `studio_small_09` | `https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_09_1k.hdr` | 1.62 MB | 18 | glossy spheres (xiii), softbox highlights |
| `kloofendal_48d_partly_cloudy_puresky` | `https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/kloofendal_48d_partly_cloudy_puresky_1k.hdr` | 1.44 MB | 21 | sun, heavy-tail tier |

2k/EXR variants exist, e.g. `…/exr/1k/studio_small_09_1k.exr` (1.32 MB), used for EXR decode tests. Sizes and URLs are from `api.polyhaven.com/files/<id>` [SOURCE]. A fetch script pins SHA-256; the files are gitignored.

---

## 6. Plan edits (exact)

**Header / User decisions.** Add: "v1 includes a full HDRI environment map (equirectangular .hdr/.exr, importance sampled, rotation about up, strength, tint, camera visibility), validated against Cycles world lighting." Add the report tag `env` = this report (id `aaf24370367ba6bcc`) to §4.2.

**§0 Scope.**
- v1 goals, Lights: add "environment map (equirect, §1.4b)".
- Non-goals: replace the env line with "Procedural/Sky-Texture worlds, mirror-ball projection, portals, world volumes, multiple env maps, image-sequence envs, USD DomeLight import (stretch: orientation convention unverified; the loader warns)".
- Delete "The background is black; the codes for env techniques stay reserved."
- Stretch: add "USD DomeLight → env".

**§1.2.** Append: "Env direction mapping (§1.4b): `b = R_z(γ)·C·d`; C lives inside `envUV`; the bridge writes Blender Mapping rotation Z = γ unchanged."

**§1.4 Lights.** Add subsection **1.4b Environment**:
- storage (one texture, tables in records, env record in lights cur/prev with stable `ENV_ID`);
- emission `L = strength·tint·bilinear(texel)`;
- selection (alias entry, `Φ_env = 4π²R_s²L̄`, clamp [0.1, 0.9]);
- sampling (2-level u16 alias, realized pdf, floor, `pdf_σ = pdf_uv/(2π² sinθ)`);
- modes: always MIS.
- Amend "Solid-angle samplers … are forbidden" → "Shading-point-dependent samplers (e.g. the spherical rectangle) are forbidden. The env's direction sampler is shading-point independent, so J = 1."
- Light-editor properties: add an Env panel (below, UI).

**§1.6 Textures.** Add: "Env texture: rgba32float (validation always), rows bottom-up, sampler repeat/repeat linear, 1 mip, `textureSampleLevel(…,0)`, sampled at Cycles' (u,v). Interactive > 4k: 2×2 downsample; optional rgba16f low-memory (biased, never in validation)."

**§1.7.** Add length-1 "camera sees env". NEE RNG block = `{u_sel, h_l0, h_l1, h_l2}` (u32 hashes).

**§1.8 Bindings.** `records` += `rowAlias`, `colAlias`, `pdfUV`, env record cur/prev; G0 += `texEnv`, `sEnv`.

**§1.9 Reservoir.**
- technique code 3 = `BSDF_ENV`; endpoint type 7 = `ENV`;
- NEE_ENV rc fields `(ENV_IDX, (i<<16)|j, (du16<<16)|dv16)` lossless;
- BSDF_ENV rcWi = world ω (f32×3 validation; oct16 production, quantised before F);
- suffix cache escape flag `t_occ = +∞`;
- memory: + texture + tables (§2.2).

**§1.10.** Add a row: "+ env: est. +3–8 % frame time for outdoor HDRI scenes; +150–200 MiB at 4k" [INFERENCE; measure in M3c].

**§2 Normative rules.**
- **Rule 2:** add the env to q/Λ/p1: "`q = pmf[ENV]·pdf_σ(uv)`, `Λ = L_env(uv)`, `p1 = q` (no r²/cos). RIS-NEE ratio formed in the same measure. `p1Env()` is one function used by NEE, BSDF-escape MIS, RIS-NEE, shifts and refresh."
- **Rule 3:** add cases "(e) BSDF escape as rc: `J = p^y_{d−1}(ω,ℓ)/jDen`, G ≡ 1, no pmf, no J_P"; "(f) forced NEE rc on the env: J = 1 spatial; temporal × J_P = pmf ratio (the pdf_uv table is shared)".
- **Rule 5:** add "Kind ENV: F = +∞, I skipped, R mandatory at x_{k−1}".
- **Rule 6:** add "O3 includes the terminal env pair; k = ∅ requires the replay to escape at the same d".
- **Rule 7:** "support indicator applies to the copied env direction at y_{d−1}".
- **Rule 10:** add the env change taxonomy (rotation = moved; strength/tint = radiometric; map/importance-res change = config → history reset).
- **Add rule 15, Env mapping and sampling:** the §2.3 formulas verbatim; realized-pdf rule; floors; pole cap `sinθ < 1e-6` on both sides; strength/tint excluded from the importance weights.
- **Add rule 16, Background pixels:** no reservoir; length-1 only; never a partner/temporal source.

**§3 Pass graph.**
- CPU: env cur/prev record, env flags (moved/radiometric), config hash incl. env map id + importance res → reset.
- `primary`: miss → length-1 env.
- `initial`: NEE_ENV/BSDF_ENV candidates.
- `suffix_refresh`: env classes.
- T1–T4 / spatial: skip misses.
- `accumulate`: + length-1 env.

**§4.1 Layout.**
- `src/scene/env/{hdr.ts (RGBE), exr.ts (EXRLoader wrapper), importance.worker.ts, proxy.ts}`;
- `shaders/lights/env.wgsl` (envUV, envDir, envRadiance, envSample, envPdfSA, p1Env);
- `validation/blender/build_scene.py` world builder + OIIO EXR writer;
- `validation/tools/env_quadrature.py` (C0q–C0s references);
- `tests/env/*`;
- `validation/assets/fetch_hdris.ts` (pinned SHA-256).

**§5 Milestones.**
- **M1:** env loader (Worker), texture upload, camera-sees-env background, orientation-grid debug view, ENV-U1/U2/U7.
- **M2:** bridge world export (EXR writer, node setup, pixel-hash assert, ENV-U9), `cycles_settings.py` world block, **C0p** in Gate 1-lite (emission kernel includes length-1 env).
- **New M3c "Environment lighting (PT)"** (after M3a; parallel to M3b; its glass variants after M3b).
  - Build: importance tables, `p1Env`, NEE_ENV + BSDF_ENV + MIS, power proxy/clamp in the alias table, env debug views, env UI and tracks (γ, strength).
  - Exit: ENV-U3–U6/U8 green; Gate 2 on C0q, C0r, C0s, (xiii), (xiv) in both modes; env plants detected; negative controls pass; `budget.json` env rows.
- **M4:** env candidates in the path tree/RIS; rc cases (e)/(f); ENV predicate; O0–O3 env; T3-ENV; Gate 3 rungs 3.1–3.2 add C0q(d), C0r, (xiii), (xiv).
- **M5:** env cur/prev in E_{t−1}; refresh; J_P; ix-h/i/j/k; env temporal plants.
- **M5.5:** background pass-through.
- **M6:** env entries in RIS-NEE light tiles (`(ENV, i, j, du, dv)`); Mode-B + env (crossings on escaping rays); Gate 3.11 includes (xiv) Mode B.
- **M7:** E2E-HDR.
- **M8:** oct16 rcWi for BSDF_ENV (Gate 3 δ), optional rgba16f low-memory mode, perf rows with env.

**§6 Debug views.**
- M1: "Env: orientation grid (lat-long lines, axes labelled +X/−Z/+Z/−X glTF, horizon), background mask."
- M3c: "Env: importance map (log pdf_uv equirect; realized/target ratio), env sample splat histogram (equirect, χ² per bin vs pdf), escape fraction per pixel, NEE-env ω1 / BSDF-env ω2, P(env) in HUD."
- M4: "technique view shows NEE_ENV/BSDF_ENV; failing-predicate view shows ENV-R fails."

**§7.1/7.2.** Add scenes C0p–C0s, (xiii), (xiv), ix-h…k, E2E-HDR, and the tiers of §5.4. Coverage table: env properties (strength, tint, γ, visibleToCamera, map res) each map to ≥ 1 Stage-A and ≥ 1 Stage-B scene.

**§7.4.** Add ENV-U1–U9 (M1/M2/M3c), T3-ENV (M4), T-ENV-temporal (M5). **T16** asserts: env rgba32float, no downsample in validation (or exported-texel hash equality), importance floors on.

**§7.5 Cycles settings.** Replace "world None" by the §1.4 table above (world node setup, sampling_method AUTOMATIC with a NONE variant, max_bounces 1024, is_caustics_light False, cycles_visibility, film_transparent False, no portals, use_light_tree True pinned, texture_limit OFF, use_simplify False, colorspace working space assert, image colorspace/alpha, pixel-hash assert). Manifest records the resolved importance-map resolution (= image res).

**§7.6 Manual E2E.** Add "load an HDRI (drag-drop), rotate it, change strength; Export for Cycles with the env".

**UI (Tweakpane "Environment" folder).**
- Controls: Load HDRI (file/URL/drop, .hdr/.exr); preset list (the three CC0 assets); Strength (log slider); Rotation (deg = Blender Mapping Z, POINT); Tint; Visible to camera.
- Camera background: env / black (validated) / colour (interactive-only badge; exportable later via a Light Path "Is Camera Ray" mix with the env on the fac = 0 input).
- Importance res (auto / 512 … 4096); Env NEE on/off (= Cycles sampling_method NONE); Texture format (32F / 16F low-memory, biased badge).
- Info: res, memory, peak radiance, EV range, P(env).
- γ and strength are animatable timeline tracks.

---

## 7. Risks and open items

1. **Hardware-filter parity** relies on identical (u,v) floats reaching the same Metal sampler; atan2/acos ulp differences are negligible but unproven. ENV-U7 plus C0p confirm.
2. **Pole-wrap quirk** (repeat in v) is Cycles behaviour pinned to 5.1.2; if we "fix" it (clamp) we diverge within half a texel of the poles.
3. **Memory at 8k** (512 MiB texture + tables + 0.57 GB reservoirs): OOM risk on the ensemble; validation cap 8k, recommend ≤ 4k.
4. **Heavy tails** with sun HDRIs make TOST sizing expensive in both renderers; handled by tiers, not by clamping.
5. **Decoder bias:** three.js HDRLoader +0.39 % is below the 0.5 % global δ, so only the bitwise unit test catches it. EXR lossy codecs (B44/DWA) may not decode bit-identically to OpenEXR: validation uses lossless or exported texels only.
6. **P(env) proxy** may be badly tuned (variance only); the clamp keeps ≥ 10 % for other emitters.
7. **Measure-zero partition gaps** (cell edges, acos near poles) give bias ≈ 10⁻⁷ relative [INFERENCE]; ENV-U6 bounds it.
8. **World vs env-local direction** must be applied exactly as §3.4 (a swap is a silent bias); ix-h and T-ENV-temporal detect it.
9. **Map swap resets history:** visible reconvergence for ~20 frames; acceptable.
10. **Colour management:** Blender 5 working space, EXR chromaticities and image colorspace are all pinned or asserted via the pixel hash (ENV-U9).
11. **Mode B + env:** escaping rays loop over all area lights for crossings (existing O(#lights) cost).
12. **Production quantisation** of BSDF_ENV rcWi (oct16) introduces a small bias, gated by M8 Gate 3 δ.
13. **Open:** measured build time and frame cost (M3c); USD DomeLight convention (stretch); whether to export the "camera background colour" to Cycles (optional).

---

## 8. Sources (this session)

- Cycles kernel (shipped): `K/light/background.h:20-142,324-468`; `K/light/light.h:25-29,149-157`; `K/light/sample.h:322-357,556-587`; `K/light/distribution.h:45-58`; `K/integrator/shade_background.h:24-125`; `K/integrator/shade_surface.h:398-402`; `K/integrator/path_state.h:59,110-216`; `K/svm/image.h:21-41,234-268`; `K/svm/mapping_util.h:15-35`; `K/svm/closure.h:1530-1541`; `K/geom/shader_data.h:366-405`; `K/camera/projection.h:19-48`; `K/device/gpu/image.h:86-117`; `K/device/metal/compat.h:387-396`; `K/device/metal/context_begin.h:40-50`; `K/bake/bake.h:55-77`; `U/projection.h:11-22`; `U/transform.h:201-225,322-327`.
- Cycles host v5.1.2: `scene/light.cpp:29-65,238-310,495-512,927-1150,1296-1330`; `scene/light_tree.cpp:223-232,305-316`; `scene/shader_nodes.cpp:74-134,498-606,1916-1970,4147-4187`; `scene/shader_nodes.h:440-453,970-997`; `scene/svm.cpp:462-477`; `scene/image.cpp:474-662`; `scene/integrator.cpp:317-323`; `blender/light.cpp:107-176`; `blender/shader.cpp:1728-1740`; `blender/image.cpp:44-129`; `device/metal/device_impl.mm:1005-1150`.
- Blender v5.1.2: `imbuf/intern/format_hdr.cc` (OIIO path); `makesrna/intern/rna_image.cc:1401-1406`; `blenkernel/intern/image_gpu.cc:212,470`; `imbuf/intern/util_gpu.cc:59-74`.
- OIIO v3.1.7.0 `src/hdr.imageio/hdrinput.cpp:119-202,335-365`.
- three.js r186.1: `examples/jsm/loaders/HDRLoader.js:358-376`; `EXRLoader.js:86,2188-2230,2453-2563`; `src/renderers/shaders/ShaderChunk/common.glsl.js:91-99`.
- RNA introspection (Blender 5.1.2 headless, factory startup, nothing saved):
  - `World.cycles`: sampling_method {NONE, AUTOMATIC, MANUAL} = AUTOMATIC, sample_map_resolution 1024, max_bounces 1024, is_caustics_light False.
  - `World.cycles_visibility`: {camera, diffuse, glossy, transmission, shadow, scatter} all True.
  - `ShaderNodeTexEnvironment`: projection {EQUIRECTANGULAR, MIRROR_BALL}, interpolation {Linear, Closest, Cubic, Smart} = Linear, texture_mapping identity.
  - `ShaderNodeMapping.vector_type` = POINT. Background default colour 0.05.
  - `film_transparent` False, `use_simplify` False, `texture_limit_render` OFF, `use_light_tree` True.
  - `bpy.data.colorspace.working_space` 'Linear Rec.709'; float image default colorspace 'Linear Rec.709'; `Image.use_half_precision` True (GPU only).
- Poly Haven API `api.polyhaven.com/files|info/<id>`; license page (CC0).
- Plan reports: gris §6.8 (env rc row), ref §1.5.2/§1.5.5/§1.6.3 (handleMiss, env rc J), enh §3.11/§5.4, gap-rc §3.3–§3.5 (e)/(f), gap-light §2.2/§3.1/§3.9/§5.7, gap-temporal §1/§4/§5/§6/§7, scene-io §7.6 (proxies) and open question 746 (answered here), webgpu-platform §3.2, gap-perf §4.3–§4.4.

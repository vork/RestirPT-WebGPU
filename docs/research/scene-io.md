# Scene I/O for a WebGPU ReSTIR PT renderer: glTF/GLB, USD, lights, internal representation, shared scene JSON

Topic report for the planning phase. Audience: an expert graphics engineer implementing the loader, light system and validation harness in TypeScript + WGSL (Chrome, Metal backend, Apple M5 Pro). No hardware ray tracing: BVH traversal in compute.

> **Where this file is.** The task asked for `.../scratchpad/research/scene-io.md`. Plan mode was active in this session, and plan mode only lets me write this plan file. So the report is here instead: `/Users/mark.boss/.claude/plans/do-a-deep-dive-shimmering-ritchie-agent-a597df4a1dd101994.md`. The orchestrator can copy it to the requested path. Nothing else was written. No repos were cloned. Sources were read in place: the Blender app bundle, `curl` to stdout, WebFetch, npm registry JSON, and `tar -tz`/`-xzO` streamed from registry tarballs. Blender was run headless only for read-only RNA introspection.
>
> **Versions checked on 2026-09-28:**
> - Blender 5.1.2 (hash ec6e62d40fa9). Bundled glTF add-on `io_scene_gltf2` 5.1.20.
> - Cycles and USD IO sources at git tag `v5.1.2`.
> - three.js r186 (npm 0.186.1).
> - @gltf-transform/* 4.5.1.
> - @loaders.gl/gltf 4.5.2.
> - @needle-tools/usd 1.1.2.
> - lightusd 1.0.0-rc4 (TinyUSDZ was renamed LightUSD).
> - OpenUSD `dev` schema docs.
>
> **Tags:** `[INFERENCE]` marks my reasoning, not something a source states. `[UNVERIFIED]` marks claims I could not confirm from a primary source in this session.

---

## 0. TL;DR: decisions I recommend

1. **One unit contract: "Blender-equivalent radiometric units".**
   - Internally the renderer uses exactly the quantities Cycles uses: W, W/sr, W/(m² sr) as Blender defines them.
   - Rule for any file: our render of it equals **Cycles(BlenderImporter(file))**.
   - glTF `KHR_lights_punctual` intensities are divided by **683** (cd→"W/sr", lux→"W/m²"). glTF emissive (nits) is taken **1:1**. This is exactly what Blender's own glTF importer does in its default `SPEC` mode. Blender's *exporter* is photometrically inconsistent: it multiplies lights by 683 but leaves emission as is (§4). So this contract is the only one that makes Blender round-trips and Cycles references agree.
   - UsdLux is taken 1:1 (Blender's USD exporter/importer treat 1 Blender radiance unit = 1 nit). There are two Blender-specific `DistantLight` quirks (§6).
2. **Lights for validation live in our own scene JSON, not in the asset files.**
   - glTF has no ratified area light. Blender's glTF exporter drops area lights.
   - Spot falloff models differ between glTF, UsdLux and Cycles by up to ~0.4 in absolute attenuation (§3.5).
   - The JSON uses Blender's native light parameterization (power in W, `spot_size`, `spot_blend`, area `size`/`size_y`, sun `angle`). Cycles light math (§2.3) is implemented exactly in WGSL.
3. **glTF parsing: `@gltf-transform/core` + `@gltf-transform/extensions` (+ `meshoptimizer`, `draco3dgltf`, a Basis transcoder).**
   - It is a complete, MIT, renderer-agnostic document model.
   - It can do `dequantize`/`unweld`/`tangents` (MikkTSpace WASM).
   - It is the best fit for converting into flat GPU buffers.
   - Fallback: three.js `GLTFLoader` used as a parser.
4. **USD in the browser: `lightusd` (formerly TinyUSDZ).**
   - Apache-2.0 OR MIT. ~1.7 MB "next" wasm (0.56 MB zstd). Single-threaded, so no COOP/COEP.
   - Reads USDA/USDC/USDZ. Complete UsdLux incl. ShapingAPI. UsdPreviewSurface. GeomSubset. PointInstancer.
   - Fallbacks: (a) three.js r186 `USDLoader` (pure JS, MIT, weaker lights/instancing); (b) offline conversion USD→GLB+JSON with Blender or `usd-core`.
   - Avoid `@needle-tools/usd` unless licensing is sorted: **PolyForm-Noncommercial-1.0.0**, 35 MB threaded wasm, requires cross-origin isolation.
5. **Internal representation.**
   - Phase 1: world-space flattened triangles + one BVH. Static geometry; lights are analytic and move freely.
   - Phase 2: two-level (instances/TLAS + BLAS) for moving or instanced geometry.
   - Stable `(instanceId, primId)` and stable `lightId` across frames.
   - Material table plus bucketed `texture_2d_array`s.
   - One light table: point/sphere, spot, rect/ellipse area, sun, plus emissive triangles.
   - **Power-based alias table + per-frame light tiles** (the scheme the ReSTIR PT Enhanced paper uses, §7.6). Light BVH later.
6. **Reference workflow.**
   - The Blender script imports *the exact file the web app loaded* with the stock importer.
   - It deletes or ignores file lights and builds lights, camera and world from the scene JSON's *resolved per-frame states*.
   - It sets unbiased Cycles settings (clamp off, filter-glossy off, denoise off, Box 1 px filter, Standard view) and writes linear EXR.

---

## 1. Why a unit contract is needed

Measured facts in the bundled add-on source, summarized:

| Quantity | Blender → glTF export (`SPEC`, default) | glTF → Blender import (`SPEC`) | Blender → USD export | USD → Blender import |
|---|---|---|---|---|
| Point/spot power P [W] | `I_cd = P/(4π) · 683 · 2^exposure` | `P = I_cd · 4π / 683` | `intensity = P/π`, `normalize = true` | `P = intensity · π` |
| Sun irradiance E [W/m²] | `E_lux = E · 683 · 2^exposure` | `E = E_lux / 683` | `intensity = E/4` (!) | `E = intensity · 4` (!) |
| Area light | **dropped** (warning "Unsupported light source AREA") | n/a | RectLight/DiskLight, `intensity = P/π`, `normalize=true` | Rect/Disk, `P = intensity·π` |
| Mesh emission (Principled Emission Color × Strength) | `emissiveFactor` (+ `KHR_materials_emissive_strength`), **no 683** | Emission Strength = `emissiveStrength` (× grey factor), **no 683** | `emissiveColor = color × strength` (constant only) | n/a |

Sources:
- `.../io_scene_gltf2/blender/com/conversion.py:10` (`PBR_WATTS_TO_LUMENS = 683`)
- `blender/exp/lights.py:39-44, 96-179`
- `blender/imp/light.py:58-77`
- `blender/exp/material/extensions/emission.py:16-114`
- `blender/imp/pbrMetallicRoughness.py:483-520`
- `source/blender/io/usd/intern/usd_writer_light.cc:133-140`
- `usd_reader_light.cc:136-145`
- `usd_writer_material.cc:199-212`

The glTF core spec says emissive is in cd/m² (nits) (`specification/2.0/Specification.adoc:2542`). Lights are in cd and lux. Photometrically those are consistent, so a "physically consistent" glTF renderer would leave both unscaled. Blender's exporter instead applies ×683 to lights but not to emission. So in a Blender-exported GLB, lights are 683× too bright *relative to emission* under a strictly photometric reading.

Blender's **importer** applies the exact inverse. Therefore:

> [INFERENCE] Defining the renderer as "what Cycles renders after Blender imports the file" is the only definition under which (a) Blender→glTF→web and (b) Cycles on the original scene agree. It also makes third-party glTF render the same way it would after importing into Blender. Internal units = Blender units. glTF lights ÷683. glTF emissive ×1. UsdLux ×1.

Absolute scale only matters up to camera exposure. Keep a global `exposureEV` in the JSON, applied as `2^EV` to the final radiance in both renderers: Blender `view_settings.exposure` is also `2^EV`.

---

## 2. Cycles light model (what the WGSL must reproduce)

All from Blender `v5.1.2` sources:
- `intern/cycles/blender/light.cpp`: Blender→Cycles sync.
- `intern/cycles/scene/light.cpp`: host-side packing.
- `intern/cycles/kernel/light/{point,spot,area,distant}.h`.
- `intern/cycles/kernel/closure/emissive.h`.
- `intern/cycles/util/math_base.h`.

### 2.1 Common factors
- `strength = color × energy × 2^exposure` (`blender/light.cpp:80`). `color` is additionally multiplied by `temperature_color` when `use_temperature` is on (lines ~74-79).
- `normalize = !(mode & LA_UNNORMALIZED)` (`blender/light.cpp:84`). The RNA property `normalize` defaults to **True** (RNA introspection).
- Light frame: `dir = -column2(tfm)`, i.e. lights emit along **local −Z**. `axisu = column0`, `axisv = column1`, `co = column3` (`scene/light.cpp:1211-1215`). The object's **scale enters these columns**. It changes area size and spot shape.
- Default object ray visibility for light objects is `visible_camera = False`, and diffuse/glossy/transmission/shadow = True (introspected). **Lights are invisible to camera rays by default but visible to BSDF rays (MIS).**
- Cycles ignores `use_custom_distance`/`cutoff_distance` (EEVEE-only): no range windowing. [UNVERIFIED that no Cycles code reads it; grep of `blender/light.cpp` shows no use.]

### 2.2 `Light::area()` (`scene/light.cpp:202-226`)
```
point/spot: A = 4π r²            (if A == 0 → A := 4)        r = shadow_soft_size ("radius")
area:       A = |axisu·sizeu| · |axisv·sizev|   (× π/4 if ellipse; sizeu/v = size, size_y, with object scale)
distant:    A = π sin²(angle/2)  (if angle == 0 → 1)          angle = sun angular diameter
```

### 2.3 Per-type emission (normalize = true, the default)

Notation: `P` = `strength` (RGB). `d` = distance. `θ` = angle between the light axis (−Z) and the direction light→shading point.

**Point, radius r = 0** (`scene/light.cpp:1269-1284`):
- `eval_fac = invarea/π` with `invarea = 1/4`, so `eval_fac = 1/(4π)`.
- Sampled "pdf" is `t²` in solid angle.
- Result: radiant intensity **I = P/(4π) [W/sr]**. Contribution `f·I·|cos θ_x|/d²`.

**Point, r > 0, `use_soft_falloff = False` → `is_sphere = true`** (`blender/light.cpp:27`, `point.h:34-63`):
- True Lambertian **sphere** light with radiance **L = P/(π·4πr²) = P/(4π²r²)**.
- Sampling: uniform cone with `one_minus_cos = 1−sqrt(1−r²/d²)`, pdf_ω = `1/(2π(1−cosθmax))`.
- Inside the sphere: cosine-hemisphere sampling (or uniform sphere if the BSDF has transmission).

**Point, r > 0, `use_soft_falloff = True` (Blender DEFAULT)** (`point.h:64-76`):
- Not a sphere. It is a **disk of radius r always facing the shading point**, with the same `eval_fac`.
- Effective intensity toward x is `P/(4π)` × (disk cos ≈ 1).
- This gives soft shadows but no sphere near-field. It only differs from the sphere model near or inside the light.
- **For validation, set `use_soft_falloff = False`** (or r = 0). [INFERENCE]

**Spot** (`blender/light.cpp:30-36`, `scene/light.cpp:1371-1390`, `spot.h:27-31`):
- Same emitter as a point/sphere light (energy is "as if not limited by the cone", RNA description).
- Multiplied by
  ```
  cosHalf  = cos(spot_size/2)
  smoothInv = 1 / ((1 − cosHalf) · spot_blend)
  att(θ) = smoothstepf( (cosθ' − cosHalf) · smoothInv ),   smoothstepf(x)=0 (x≤0), 1 (x≥1), 3x²−2x³ otherwise
  ```
- `cosθ'` is the z of the **normalized direction in light-local space after dividing by object scale** (`spot_light_to_local`, `spot.h:15-24`, with z negated so that +z means along −Z_light). With unit scale it is plain `cosθ`.
- `spot_blend = 0` gives `smoothInv = +inf`, i.e. a hard step. Guard this in WGSL.
- Defaults: `spot_size = 45°` (0.785398), `spot_blend = 0.15`.
- For r > 0 with `is_sphere`, the attenuation is skipped for shading points inside the sphere (`spot.h:244`).

**Area (rect/square/disk/ellipse)** (`scene/light.cpp:1329-1370`, `area.h:106-121, 313-319`):
- One-sided, emits along −Z.
- `L = P · invarea / π`, i.e. **L = P/(π·A)** (A as §2.2).
- Spread: `half = spread/2`. If `spread == π` the factor is effectively 1 (tan = FLT_MAX). Otherwise:
  ```
  factor = max((tan(half) − tan α) · N, 0),   N = 1/(tan(half) − half)  (half > 0.05)   or 3/half³ (small)
  α = angle between −D and light normal;  spread == 0 → factor = π only exactly on-axis
  ```
- Default `spread = π` (180°) and `size = size_y = 0.25`.

**Sun / distant** (`scene/light.cpp:1286-1306`, `distant.h`):
- `h = angle/2`. **E = P (W/m²)** at normal incidence. When `h > 0`, the radiance within the cone is `L = P/(π sin² h)`.
- Sampling is a uniform cone with pdf `1/(2π(1−cos h))`.
- `angle = 0` gives a delta directional light with irradiance P.
- **New lights get `angle = 0.00918 rad` (0.526°)** (introspected). This matters for glTF-imported directional lights (§4.3).

**Mesh emission** (`kernel/closure/emissive.h:40-60`):
- `emissive_pdf` uses `fabsf(dot(Ng, wi))`, so **mesh emission is two-sided** with **L = Strength × Color** (no π, no area normalization).
- The material `emission_sampling` setting (AUTO/FRONT/BACK/FRONT_BACK/NONE) only changes NEE sampling. [INFERENCE] It does not change the expectation, because BSDF-sampled hits still see both sides.
- Cycles does **not** cull backfaces for rendering: `blender/*.cpp` contains no use of `MA_BL_CULL_BACKFACE`. The glTF importer's `use_backface_culling = not doubleSided` (`blender/imp/material.py:84-85`) is therefore irrelevant to Cycles. **To match Cycles, treat every triangle as double-sided and every emitter as two-sided.**

### 2.4 Light defaults that affect references (Blender 5.1.2, introspected)
- `bpy.data.lights.new(...)`: `energy = 10`, `shadow_soft_size = 0.0`, `normalize = True`, `use_soft_falloff = True`, `exposure = 0`, `cycles.use_multiple_importance_sampling = True`, `max_bounces = 1024`.
- The factory startup scene's `Light`: POINT, 1000 W, radius 0.1, soft falloff on.
- The factory world background is `(0.0509, 0.0509, 0.0509)`, strength 1. **Set it explicitly** (black or a JSON value).

---

## 3. glTF / GLB

### 3.1 Parser options

| Option | Package / version / license | What you get | Pros | Cons |
|---|---|---|---|---|
| Hand-written | none | GLB container (12-B header, JSON chunk, BIN chunk), accessors | zero deps; stream straight into GPU layouts | you still need WASM decoders (Draco, meshopt, Basis); many edge cases (sparse accessors, `byteStride`, normalized ints/`KHR_mesh_quantization`, primitive modes, texture transforms, `.gltf` external URIs) |
| **glTF-Transform** | `@gltf-transform/core`, `/extensions`, `/functions` **4.5.1** (published 2026-09-28), MIT | typed Document graph; `WebIO.read(url)`, `readBinary(u8)`, `readJSON({json,resources})`; all Khronos extensions (`KHRONOS_EXTENSIONS`/`ALL_EXTENSIONS`): lights_punctual, emissive_strength, ior, specular, transmission, volume, clearcoat, sheen, anisotropy, iridescence, dispersion, diffuse_transmission, texture_transform, texture_basisu, mesh_quantization, draco, meshopt (EXT), gpu_instancing, variants, node_visibility, accessor_float16/64 | lossless access to raw glTF values (factors, texCoord, transforms); decoders via `registerDependencies({'draco3d.decoder', 'meshopt.decoder'})`; functions `dequantize`, `unweld`, `weld`, `tangents` (needs `mikktspace` WASM, MIT, 1.1.1), `flatten`, `dedup`, `uninstance`, `transformPrimitive`, `getBounds` | not a renderer; KTX2 stays encoded (you transcode); images are encoded bytes (decode with `createImageBitmap`) |
| three.js `GLTFLoader` as parser | `three` **0.186.1 (r186)**, MIT | three.js scene graph, `BufferGeometry`, `MeshPhysicalMaterial`, three lights | battle-tested; DRACOLoader/KTX2Loader/MeshoptDecoder; extensions (r186 `EXTENSIONS` list): draco, lights_punctual, clearcoat, dispersion, ior, sheen, specular, transmission, iridescence, anisotropy, unlit, volume, basisu, texture_transform, mesh_quantization, emissive_strength, EXT_materials_bump, webp, avif, EXT/KHR_meshopt_compression, EXT_mesh_gpu_instancing | lossy remapping into three's material model (e.g. spot `penumbra = 1 − inner/outer`, `GLTFLoader.js:733-736`); pulls in three.js; texture transforms become three.js `Texture` offset/repeat/rotation conventions |
| loaders.gl | `@loaders.gl/gltf` **4.5.2**, MIT | post-processed JSON + typed arrays | framework-independent; Draco, meshopt (EXT/KHR), basisu, texture_transform, webp/avif | material/light extensions are passed through as raw JSON (`KHR_lights_punctual` lives in `lib/extensions/deprecated/`) |

**Recommendation: glTF-Transform.** It exposes every value we need exactly as authored, so we can apply our own Blender-equivalent semantics. The decoders are small, and it runs in a Web Worker. Keep three.js `GLTFLoader` as a debugging cross-check (e.g. against three's WebGPU path-tracing examples). [INFERENCE]

Loader sketch (TypeScript, in a Worker):
```ts
import { WebIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { dequantize, unweld } from '@gltf-transform/functions';
import { MeshoptDecoder } from 'meshoptimizer';          // 1.3.0, MIT
import draco3d from 'draco3dgltf';                       // 1.5.7, Apache-2.0
await MeshoptDecoder.ready;
const io = new WebIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
  'draco3d.decoder': await draco3d.createDecoderModule(),
  'meshopt.decoder': MeshoptDecoder,
});
const doc = await io.readBinary(glbBytes);          // or io.read(url) / io.readJSON({json, resources})
await doc.transform(dequantize());                   // KHR_mesh_quantization → float32
// traverse doc.getRoot().getDefaultScene() nodes; node.getWorldMatrix(); prim.getAttribute('POSITION')...
```
KTX2/Basis: transcode with Binomial's `basis_transcoder` (three.js ships `examples/jsm/libs/basis/basis_transcoder.{js,wasm}`, the wasm is 527 kB). Target `bc7-rgba-unorm(-srgb)` when `adapter.features.has('texture-compression-bc')`, else `astc-4x4`, else RGBA8. Apple-silicon Macs support BC and ASTC natively.

### 3.2 Core glTF facts that matter for a path tracer
- Units are meters. Right-handed, **+Y up**. Front of asset faces +Z. Spec §"Coordinate System and Units" (`Specification.adoc:703-713`).
- **UV origin is the top-left** of the image (restated in the `KHR_texture_transform` README). Blender and USD use bottom-left, so importers flip v.
- Winding: a negative-determinant world transform flips the front face (core spec). For geometric normals used in one-sided emission or shading-normal fixups, flip the triangle when `det(M) < 0`.
- Missing `NORMAL`: the spec requires flat normals. Missing `TANGENT` with a normal map: the spec says to use MikkTSpace. **Blender's importer never reads `TANGENT`** (no occurrence in `blender/imp/mesh.py`); Cycles computes MikkTSpace from the UV map. So always generate MikkTSpace per-vertex tangents for matching (glTF-Transform `tangents({generateTangents})` with `mikktspace`).
- Alpha:
  - `OPAQUE`: α = 1.
  - `MASK`: α' = (α ≥ alphaCutoff ? 1 : 0).
  - `BLEND`: α' = α.
  - The path tracer treats α' as a stochastic "transparent BSDF" mix (pass-through with probability 1−α'), which is what Cycles does with Principled Alpha. Cycles counts these as *transparent bounces* (`transparent_max_bounces` default 8). Match or raise it. [INFERENCE on exact equivalence of the importer's MASK node setup]
- `doubleSided`: ignored in "Cycles mode" (§2.3).
- Samplers: Blender Image Texture default interpolation is Linear (bilinear, **no mip-mapping** in Cycles). For validation mode sample **LOD 0 bilinear** (`textureSampleLevel(t, s, uv, layer, 0.0)`); `textureSample` is fragment-only in WGSL anyway. Wrap: REPEAT/CLAMP/MIRROR → WebGPU `repeat`/`clamp-to-edge`/`mirror-repeat`.
- sRGB decode: base color, emissive, sheen color and specular color textures are sRGB; everything else is linear. Use `rgba8unorm-srgb` (hardware decode before filtering), which matches Cycles' linearize-at-load then filter.
- Multiple UV sets: `textureInfo.texCoord`; `KHR_texture_transform.texCoord` overrides it.
- `COLOR_0` multiplies base color (Blender importer wires it via "vertex color" only when used by a material; [UNVERIFIED] exact rule).
- Skins, morphs and animations: out of scope for v1. Load the bind pose (static); optionally bake a pose at load.

### 3.3 Extensions, prioritized for PBR path tracing

| Priority | Extension | Parameters (defaults) | Path-tracer semantics |
|---|---|---|---|
| P0 | core metallic-roughness | baseColorFactor(1), metallic(1), roughness(1), textures (MR: B=metal, G=rough), normal(scale), occlusion (ignore in PT), emissive(0) | α_GGX = roughness² (same convention as Cycles Principled) |
| P0 | `KHR_materials_emissive_strength` | `emissiveStrength` = 1 | `L = emissiveFactor · emissiveTex · emissiveStrength`; "unitless multiplier, does not alter units" (README "Physical Units") |
| P0 | `KHR_texture_transform` | offset [0,0], rotation 0 (rad, CCW of UVs), scale [1,1], texCoord | `uv' = T·R·S·[uv,1]` with GLSL `mat3 rotation = mat3(cos, sin, 0, −sin, cos, 0, 0,0,1)` (column-major), i.e. `R = [[c,−s],[s,c]]` (README lines 27-45). Precompute a 2×3 per texture slot. |
| P0 | `KHR_mesh_quantization` | normalized/int attributes | `dequantize()` at load |
| P0 | `EXT_meshopt_compression`, `KHR_draco_mesh_compression` | — | decode at load (meshoptimizer / draco3dgltf) |
| P0 | `KHR_texture_basisu`, `EXT_texture_webp`, `EXT_texture_avif` | — | KTX2 → BC7/ASTC/RGBA8; webp/avif via `createImageBitmap` |
| P0 | `KHR_lights_punctual` | §3.4 | analytic lights (÷683 contract) |
| P1 | `KHR_materials_ior` | `ior` = 1.5; `ior = 0` special: F = 1, spec-gloss compatibility | `F0 = ((ior−1)/(ior+1))²` (0.04 at 1.5) |
| P1 | `KHR_materials_specular` | specularFactor 1 (tex A), specularColorFactor [1,1,1] (tex RGB, sRGB) | `dielectric_f0 = min(F0(ior)·specularColor, 1)·specular`, F90 = specular. Blender exports `specularFactor = 2 × "Specular IOR Level"` and `specularColor = "Specular Tint"` (`blender/exp/material/extensions/specular.py` ~l.14-93) |
| P1 | `KHR_materials_transmission` | transmissionFactor 0 (tex R) | specular BTDF replacing diffuse in the dielectric base; thin-walled unless `volume` |
| P1 | `KHR_materials_volume` | thicknessFactor 0 (tex G), attenuationDistance +∞, attenuationColor [1,1,1] | closed mesh bounds a homogeneous absorbing medium, `σ_a = −ln(attenuationColor)/attenuationDistance`. Thickness is a raster hint; a path tracer ignores it except `thickness == 0` ⇒ thin-walled |
| P1 | `KHR_materials_clearcoat` | clearcoatFactor 0 (R), clearcoatRoughness 0 (G), clearcoatNormalTexture | GGX layer, IOR 1.5 |
| P2 | `KHR_materials_sheen` | sheenColor [0,0,0] (RGB sRGB), sheenRoughness 0 (A) | Charlie sheen + albedo scaling |
| P2 | `KHR_materials_anisotropy`, `_iridescence`, `_dispersion`, `_diffuse_transmission` (RC), `_unlit`, `_variants`, `EXT_mesh_gpu_instancing`, `KHR_node_visibility` | — | later; instancing matters for phase-2 BLAS reuse |

Blender 5.1 exporter emits (grep of `blender/exp`): `KHR_materials_{specular, volume, clearcoat, sheen, anisotropy, transmission, ior, emissive_strength, unlit, variants}`, `KHR_texture_transform`, `KHR_lights_punctual`, `KHR_draco_mesh_compression`, `EXT_mesh_gpu_instancing`, `EXT_texture_webp`, `KHR_animation_pointer`. That is the minimum set for Blender round-trips.

> Material-model caveat for validation, to hand to the BSDF owner. Cycles' Principled BSDF differs from the glTF reference BRDF: its default distribution is `MULTI_GGX` (energy-compensated), `Diffuse Roughness 0` = Lambert, "Specular IOR Level" 0.5 is neutral, and metals use an F82-tint model. [INFERENCE] Either implement Cycles-equivalent lobes, or have the reference script force `distribution = 'GGX'` and restrict validation materials to a common subset (Diffuse BSDF; Principled with coat/sheen/subsurface = 0).

### 3.4 `KHR_lights_punctual`, exact semantics (ratified; `KHR_lights_punctual/README.md`)
- Types are `directional`, `point` and `spot`. A node references them via `extensions.KHR_lights_punctual.light`. Lights inherit the node's world transform.
- Direction is **local −Z**. For point and spot, position is the node's world location.
- "The light's transform is affected by the node's world scale, but all properties of the light (such as `range` and `intensity`) are unaffected."
- `color` is linear RGB, default [1,1,1]. `intensity` default 1.
  - point/spot: **luminous intensity in candela (lm/sr)**.
  - directional: **illuminance in lux (lm/m²)** (README line 90).
  - Spot intensity is the intensity "inside the innerConeAngle"; it is *not* normalized by cone size.
- `range` (point/spot only, > 0; undefined = ∞): "rendering engines ignore the light beyond this range". The recommended window is
  `attenuation = max(min(1 − (d/range)^4, 1), 0) / d²` (line 106).
- `spot.innerConeAngle` default 0, must be ≥ 0 and < outer. `spot.outerConeAngle` default π/4, must be > inner and ≤ π/2 (line 127).
- Recommended (informative) smooth falloff, lines 160-167:
  ```
  lightAngleScale  = 1 / max(0.001, cos(inner) − cos(outer))
  lightAngleOffset = −cos(outer) · lightAngleScale
  a = saturate(dot(spotDir, L) · lightAngleScale + lightAngleOffset);   a = a²
  ```
- **Contract mapping (Blender-equivalent):**
  - `I_W/sr = intensity/683 · color`.
  - `E_W/m² = intensity/683 · color`, with a sun angle of 0.00918 rad (what Blender assigns on import).
  - Spot falloff uses **Blender's model** with `spot_size = 2·outer` and `spot_blend = 1 − inner/outer` (importer, `blender/imp/light.py:137-143`).
  - `range` is ignored (importer: `# TODO range`, line 35).
  - Offer a per-scene switch `lightModel: "blender" | "gltf-spec"` for non-validation use. [INFERENCE]

### 3.5 Spot falloff: three models compared (numbers computed in this session)

Setup: Blender `spot_size = 45°`, `spot_blend = 0.15`. glTF export gives inner 19.125°, outer 22.5°. USD export gives cone:angle 22.5°, softness 0.15.

| θ (deg) | Cycles (Blender) | glTF recommended | UsdLux spec (softness = blend) |
|---|---|---|---|
| 19.5 | 1.000 | 0.804 | 0.966 |
| 20.5 | 1.000 | 0.374 | 0.637 |
| 21.0 | 0.939 | 0.215 | 0.417 |
| 21.5 | 0.608 | 0.098 | 0.211 |
| 22.0 | 0.203 | 0.025 | 0.059 |

With `spot_size = 90°` and `blend = 0.5`, at 35° the three give 0.860 / 0.267 / 0.417.

UsdLux formula (`schema.usda` ShapingAPI, lines 765-797):
`θ_smoothStart = lerp(softness, θ_cutoff, 0) = θ_cutoff(1−softness)` and `L *= 1 − smoothstep(θ_offAxis; θ_smoothStart, θ_cutoff)`. This is smoothstep in the **angle** domain; Cycles uses smoothstep in the **cosine** domain over `[cos h, cos h + (1−cos h)·blend]`.

⇒ **Lights must be specified in Blender terms for validation.** The implementation must support at least the Blender model. It should also support the glTF and UsdLux models, selected per light by a `falloffModel` enum.

### 3.6 Area lights in glTF
- **No ratified area-light extension.**
  - `KHR_lights_area` (PR #1948, bhouston, rect/disk/sphere, intensity in nits) was **closed 2023-10-17**.
  - The successor **`EXT_lights_area` (PR #2525, MiiBond) is open**; latest activity 2026-07-22.
  - Its draft, which may change: shapes `rect` and `disk`; common `color`, `intensity` in **nits** (default 1000), `size` (default 1); `rect.aspect`. Emission is one-sided along −Z and Lambertian, with `flux = intensity·area·π`. Uniform scale is the max |scale| component; rect width = `scale·size·aspect`, height = `scale·size`; disk diameter = `scale·size`. [UNVERIFIED — draft text as summarized from the PR diff; re-read before implementing.]
  - Registry "in-progress" table (`extensions/README.md`) lists neither.
  - Vendor light extensions that do exist: `EXT_lights_ies`, `EXT_lights_image_based`.
- Blender's glTF exporter **drops AREA lights** (`blender/exp/lights.py:39-44`). The light's node is still exported without an extension.
  - To carry them anyway, post-process the exported JSON: match node names and inject `EXT_lights_area`.
  - Or write a `glTF2ExportUserExtension` with `gather_node_hook`. The exporter discovers these only from *enabled add-ons* in `preferences.addons` (`__init__.py:1326-1340`).
  - Our scene JSON makes this unnecessary for validation.
- Supporting `EXT_lights_area` read-only in our loader is cheap. Map it to our rect/disk area light, with `L = intensity·color` (nits → 1:1 contract).

### 3.7 Emissive meshes as area lights
- Any triangle whose material has non-zero `emissiveFactor·emissiveStrength` becomes an emitter. Radiance is `L(uv) = emissiveFactor · sRGB→lin(emissiveTex(uv)) · emissiveStrength` (1:1 contract), **two-sided in Cycles mode**.
- For light selection you need per-triangle flux. Falcor pre-integrates textured emission over each triangle in UV space (`Source/Falcor/Scene/Lights/EmissiveIntegrator.3d.slang`, `FinalizeIntegration.cs.slang:58-111`):
  ```
  averageRadiance = avgTexel(tri) · emissiveFactor
  flux = luminance(averageRadiance) · area · π        // "diffuse emitters, integrate per side (hemisphere)"
  ```
  Double it for two-sided emission. Cheaper approximations: sample the emissive texture at a mip whose texel size ≈ triangle UV area, or average the 3 vertices (Falcor's fallback).
- A rect area light is **not** equivalent to an emissive quad in Cycles:
  - area lights are one-sided and camera-invisible by default;
  - emissive quads are two-sided and camera-visible.
  Do not substitute one for the other in validation scenes.

---

## 4. Blender 5.1 glTF I/O light and material conversion (source-verified)

Add-on path: `/Applications/Blender.app/Contents/Resources/5.1/scripts/addons_core/io_scene_gltf2/` (version 5.1.20). In 5.1 the files are `blender/exp/lights.py`, `blender/exp/light_spots.py`, `blender/imp/light.py` and `blender/com/conversion.py`. The old `gltf2_blender_*.py` names no longer exist.

### 4.1 `export_import_convert_lighting_mode` (`__init__.py:185-197`, shared by import and export)
Enum items:
- `SPEC` "Standard": "Physically-based glTF lighting units (cd, lx, nt)". **Default.**
- `COMPAT` "Unitless": "Non-physical, unitless lighting".
- `RAW` "Raw (Deprecated)": "Blender lighting strengths with no conversion".

The description says it "Applies to lights" (the emissive TODO is commented out).

**Export** (`blender/exp/lights.py:96-179`). `S` is the strength resolved first:
1. `S = energy`. If the light uses nodes with an Emission node, `S = LightFalloff.Strength/(4π)` or the Emission Strength.
2. If `normalize == False`, `S *= light.area(matrix_world)` (lines 111-113, 133-135, 148-150).

Then:
- `RAW`: `intensity = S · 2^exposure`.
- Otherwise:
  - SUN: `lum = S`.
  - POINT/SPOT: `lum = S/(4π)`.
  - SPEC: `lum *= 683` (`PBR_WATTS_TO_LUMENS`, `conversion.py:10`).
  - COMPAT: no factor.
  - Then `intensity = lum · 2^exposure`.

So in SPEC mode, a point light gives **cd = P·683/(4π) = 54.35·P**, and a sun gives **lux = 683·E**.
- Color = `light.color × emission-node color × temperature_color` (the emission node is ignored under EEVEE) (lines 47-90).
- `range` = `cutoff_distance` only if `use_custom_distance` (lines 192-201).
- `type` map: POINT→point, SUN→directional, SPOT→spot (`blender/com/blender_default.py:11-15`). **AREA and HEMI are filtered out** with a warning (lines 39-44).
- Orientation: with `export_yup` (default True), light and camera nodes get `matrix_world @= Quaternion((√2/2, −√2/2, 0, 0))` (`blender/exp/tree.py:286-298`). Blender lights and glTF lights both point down local −Z; the correction compensates for the Y-up basis change.
- **`export_lights` defaults to False, and so does `export_cameras`** (operator introspection). The reference and asset pipeline must pass `export_lights=True` explicitly if lights are wanted in the file.

**Import** (`blender/imp/light.py`):
- directional: `energy = lux/683` (SPEC); `lux` (COMPAT/RAW) (lines 58-66).
- point/spot: `energy = cd/683 · 4π` (SPEC); `cd·4π` (COMPAT); `cd` (RAW) (lines 69-77).
- Spot: `spot_size = 2·outer`, `spot_blend = 1 − inner/outer`. Missing values use the glTF defaults (outer π/4, inner 0) (lines 102-112, 137-143).
- `range` is ignored (line 35).
- Lights are created with `bpy.data.lights.new`, so radius 0 (true point) and sun `angle` 0.00918 rad.
- Axis conversion: glTF (x,y,z) → Blender (x,−z,y), plus a camera/light correction quaternion `(√2/2, √2/2, 0, 0)` (`blender/imp/blender_gltf.py:85-102`).

### 4.2 Spot blend ↔ inner cone
- Export: `outer = spot_size/2`, `inner = outer − outer·spot_blend = outer·(1 − blend)` (`light_spots.py:28-47`).
- Import: exact inverse.
- The *parameters* round-trip exactly, but the *falloff shape* does not match the glTF recommended curve (§3.5).

### 4.3 Emission and materials
- Export (`emission.py:16-114`): `factor = EmissionColor × EmissionStrength`. If any component is > 1, `emissiveStrength = max(factor)` and `factor /= max` (`materials.py:88-92`). **No unit conversion.**
- Import (`pbrMetallicRoughness.py:483-520`): Strength = `emissiveStrength`; a grey factor is folded into strength.
- `KHR_texture_transform` ↔ Mapping-node conversion is implemented in `blender/com/conversion.py:37-70`. It includes the v-flip.

---

## 5. USD in the browser

### 5.1 Options (facts gathered 2026-09-28)

| Option | Distribution | License | Size | Threads / COOP-COEP | Formats | Composition | UsdLux | Materials | Maintenance |
|---|---|---|---|---|---|---|---|---|---|
| **LightUSD** (formerly TinyUSDZ, Light Transport) | npm `lightusd` **1.0.0-rc4** (2026-09-03), repo `github.com/lighttransport/LightUSD` (`web/npm`); older npm `tinyusdz` (dist-tag latest 0.9.1 from 2025-06-27; `rc` 0.9.10 from 2026-07-14) | **Apache-2.0 OR MIT** | `lightusd_next.wasm` 1.67 MB (**0.56 MB .zst**), legacy `lightusd.wasm` 6.9 MB (1.45 MB .zst), wasm64 variants | no SharedArrayBuffer in the JS glue (grep) → **no cross-origin isolation needed**; has a Worker loader | USDA, USDC, USDZ | LIVRPS layer flattening + a PCP-like DAG engine (`doc/composition.md`) | `doc/api-status.md`: Sphere/Disk/Rect/Cylinder/Distant/Dome/Dome_1/Geometry/Portal lights, LightAPI, MeshLightAPI, **ShapingAPI: done**; Tydra `RenderLight` has type, color, intensity, exposure, normalize, colorTemperature, radius, width, height, angle, shaping cone angle/softness/focus, world transform (`src/tydra/render-data.hh:1630-1720` on `dev`; JS `getLight()` in `web/binding.cc:4285-4455`) | UsdPreviewSurface, OpenPBR, MaterialX; GeomSubset; PointInstancer | very active; renamed and 1.0 RC in 2026 |
| three.js `USDLoader` | `three` r186 `examples/jsm/loaders/USDLoader.js` + `usd/USDAParser.js`, `USDCParser.js`, `USDComposer.js` (112 kB); `USDZLoader` deprecated since r179 | MIT | pure JS (~180 kB src) | none | USDA, USDC, USDZ | references, payloads, variantSets (grep); no `instanceable`/PointInstancer handling found | Sphere/Rect/Disk/Distant → three.js lights, **raw `inputs:intensity`**; ignores exposure/normalize; `shaping:cone:softness` → penumbra; disk → square RectAreaLight (`USDComposer.js:1523-1580`) | UsdPreviewSurface → three materials; GeomSubset; UsdTransform2d | maintained with three.js |
| Needle `@needle-tools/usd` (usd-viewer) | npm 1.1.2 (2026-07-17); upstream **OpenUSD 26.05** + Hydra + MaterialX 1.39.5 + OpenSubdiv 3.6.1 + usdGltf/usdDraco (`src/bindings/openusd-build-info.json`) | **PolyForm-Noncommercial-1.0.0** since 1.0.0 (2026-06); earlier 0.0.x releases had no license field; "For commercial use, please contact" Needle | **35.4 MB** `emHdBindings.wasm` | **threaded**: requires `Cross-Origin-Embedder-Policy: require-corp` + `Cross-Origin-Opener-Policy: same-origin` | USD/USDA/USDC/USDZ (+glTF via usdGltf) | full OpenUSD | full, but surfaced through a Hydra→three.js render delegate | full (Hydra material networks) | active |
| Upstream OpenUSD wasm | OpenUSD **v26.03** added emscripten wasm32/wasm64 builds (PRs #3832, #3833) + `wasmFetchResolver` example (AOUSD blog "Announcing OpenUSD v26.03") | Tomorrow Open Source Technology License (modified Apache-2.0) | self-built, large (≥ tens of MB) [UNVERIFIED] | threading depends on your build | all | full | full | full | you own the build and the JS bindings |
| Autodesk fork | `autodesk-forks/USD` (`adsk/feature/webgpu`), the origin of the Needle bindings | Apache-style | large | threaded | all | full | via Hydra | via Hydra | superseded by upstream 26.03 support [INFERENCE] |
| Offline conversion | Blender headless (`wm.usd_import` → our JSON + `export_scene.gltf`) or `usd-core` Python | — | 0 in browser | — | — | full | full (we control the conversion) | lossy to glTF | robust fallback |

**Recommendation.**
- **Primary: LightUSD (`lightusd`)**, used only to get composed, triangulated render data (meshes, materials, lights, cameras) via its Tydra RenderScene API. We never use its three.js builders. Reasons:
  - permissive license;
  - small single-threaded wasm, so no COOP/COEP headers, which keeps hosting simple;
  - complete UsdLux incl. `normalize` and ShapingAPI;
  - output already in "render-ready" form (triangulated, per-vertex, material IDs).
- **Fallback 1: three.js `USDLoader`** for simple USDZ/USDA, when the wasm fails or for comparison. We must re-derive the light semantics ourselves from raw attributes, because it maps lights naively.
- **Fallback 2: offline conversion** through Blender, which is always available on the dev machine.
- **Pin versions and write an adapter.** [INFERENCE] LightUSD is an RC that was renamed in 2026. Pin the exact version and put an adapter interface (`UsdSceneSource`) in front of it so it can be swapped. Spike first: load Blender-exported USD with lights, Kitchen_set (non-commercial test only) and a PointInstancer asset, and check the `getLight()` fields in the rc4 build. [UNVERIFIED that every field listed in `binding.cc` on `dev` is exposed in rc4.]

### 5.2 USD geometry → internal mapping
- **Stage metadata.** `upAxis` is `"Y"` or `"Z"`. Blender exports **Z** unless `convert_orientation=True` (`usd_capi_export.cc:545-555`). Convert Z-up→Y-up with R_x(−90°): (x,y,z) → (x, z, −y). `metersPerUnit`: scale to meters. `timeCodesPerSecond`. The root layer documentation string is `"Blender v…"` for Blender exports (`usd_capi_export.cc:529`); use it for quirk detection (§6).
- **Transforms.**
  - `xformOpOrder` composes as `M_local = op[0]·op[1]·…·op[n−1]` (column-vector convention; `[translate, rotateXYZ, scale]` gives T·R·S).
  - `!invert!` prefix; `!resetXformStack!` drops parent transforms.
  - `rotateXYZ` is in **degrees**: X is applied first, then Y, then Z.
  - `orient` is a quaternion. `transform` is a matrix4d.
  - USD matrices are row-major and act on row vectors, with translation in elements 12-14. This is **the same memory layout** as glTF/WebGPU column-major matrices for column vectors, so a straight copy works.
- **UsdGeomMesh.**
  - `points`, `faceVertexCounts`, `faceVertexIndices`, `holeIndices`.
  - `orientation`: `leftHanded` reverses winding.
  - `doubleSided` (irrelevant in Cycles mode).
  - `subdivisionScheme`: default **catmullClark**. Authored normals are then ignored per spec. We render the cage with computed smooth normals (flag it). Blender exports `none` for meshes without a subsurf modifier (`usd_writer_mesh.cc:453-503`).
  - Normals: Blender writes faceVarying.
  - Primvars have interpolation `constant | uniform | varying | vertex | faceVarying`; indexed primvars use `primvars:X:indices`.
  - Strategy: unweld to face-corners, triangulate (fan for convex polygons, ear-clip for concave, as three.js does with earcut), then weld identical corner tuples.
  - `primvars:st` has its origin at the bottom-left, so `v_internal = 1 − v`.
- **Materials.**
  - `material:binding` (UsdShadeMaterialBindingAPI; purposes `full`/`preview`, prefer `full` then all-purpose).
  - `UsdGeomSubset` with `familyName = "materialBind"` and `elementType = "face"` gives per-face material IDs. Faces not in a subset fall back to the mesh binding. Blender writes subsets via `CreateMaterialBindSubset` (`usd_writer_mesh.cc:691`).
- **Instancing.**
  - `instanceable = true` prims share a prototype, which maps to one BLAS with many instances in phase 2.
  - `UsdGeomPointInstancer` has `protoIndices`, `positions`, `orientations` (quath), `scales`, `invisibleIds`, and `prototypes`.
  - Phase 1 flattens everything.
- `visibility`, and `purpose` (`render` preferred; skip `guide`/`proxy`).

### 5.3 UsdPreviewSurface → internal material (spec v2.5, openusd.org `spec_usdpreviewsurface.html`)

| Input | Default | Mapping |
|---|---|---|
| diffuseColor | 0.18 grey | baseColor |
| emissiveColor | 0 | emission radiance (1:1 contract) |
| useSpecularWorkflow | 0 | if 1: `specularColor` = F0 color (edge white); map to `KHR_materials_specular`-like F0 with metallic 0 |
| metallic, roughness | 0, 0.5 | same meaning; roughness "usually squared before use with GGX" |
| clearcoat, clearcoatRoughness | 0, 0.01 | clearcoat layer |
| opacity, opacityThreshold, opacityMode | 1, 0, `transparent` | `threshold > 0` → MASK; else stochastic α; `presence` scales all lighting |
| ior | 1.5 | metallic workflow: F0 = ((1−ior)/(1+ior))² when metallic = 0 |
| normal | (0,0,1) | tangent space; 8-bit textures need scale (2,2,2,1) / bias (−1,−1,−1,0) authored on UsdUVTexture |
| occlusion, displacement | — | ignore |

- `UsdUVTexture`:
  - `file`, `st`, `wrapS`/`wrapT` (black/clamp/repeat/mirror/useMetadata), `fallback`;
  - `scale`/`bias` (applied as `value·scale + bias`);
  - `sourceColorSpace` (raw/sRGB/auto);
  - outputs r, g, b, a, rgb.
- `UsdTransform2d`: `in·scale·rotate + translation`, rotation in degrees CCW.
- `UsdPrimvarReader_float2` with `varname` selects the UV set.
- Blender writes a **non-standard float input `specular`** (from "Specular IOR Level"), so treat it as optional (`usd_writer_material.cc:73-76, 613-636`).

### 5.4 UsdLux → internal lights (OpenUSD `pxr/usd/usdLux/schema.usda`, `dev`)
- **Base (LightAPI).** Radiance `L = intensity · 2^exposure · color (· colorTemperature white point) / sizeFactor`.
  - "luminance of the default light will be 1 nit (cd/m²)" (schema lines 56-73, 195-234).
  - `sizeFactor = 1` if `normalize = false` (the default is **false**).
- **normalize = true** (lines 258-375):
  - Area family (Rect/Disk/Sphere/Cylinder/MeshLightAPI): `sizeFactor = world-space surface area` (including transform scale).
  - Dome: ignored.
  - Distant, with `θmax = clamp(rad(angle)/2, 0, π)`:
    - `θmax = 0` → 1;
    - `0 < θmax ≤ π/2` → `π sin²θmax`;
    - `> π/2` → `(2 − sin²θmax)·π`.
  - With normalize on, distant `intensity` becomes illuminance in lux.
- **Geometry.**
  - RectLight: `width`, `height` (defaults 1), one-sided along −Z.
  - DiskLight: `radius` (0.5), one-sided along −Z.
  - SphereLight: `radius` (0.5), plus `treatAsPoint`, a hint that zero radius is acceptable.
  - CylinderLight: `length`, `radius`, along X; `treatAsLine`.
  - DistantLight: `angle` = **angular diameter** in degrees (default 0.53), `intensity` default **50000**.
- **ShapingAPI.**
  - `shaping:cone:angle` (default 90°) is a hard cutoff beyond the angle.
  - `shaping:cone:softness` (default 0): formula in §3.5.
  - `shaping:focus` (0) and `focusTint` (black): `focusFactor = |emitDir·axis|^focus`, `L *= lerp(focusFactor, focusTint, 1)`.
  - IES profiles are out of scope.
- **Mapping to our types** [INFERENCE for the point limit]:
  - SphereLight with `treatAsPoint` or radius 0 → point light.
    - With `normalize = true`, the limit of `I = L·πr²` with `L = i/(4πr²)` is **I = i·2^e/4**, independent of r.
    - This equals Cycles' convention (`area := 4` when r = 0) and Blender's export `i = P/π` → `I = P/(4π)`. ✓
    - With `normalize = false`: I = i (Blender import `P = iπ`, Cycles unnormalized `I = P/π`).
  - SphereLight r > 0 → sphere light with `L = i·2^e/(4πr²)` (normalized) or `L = i·2^e`.
  - SphereLight + ShapingAPI cone → spot. Falloff model is `usd` (spec) or `blender` (if a Blender-authored file, see §6).
  - RectLight → rect area light `L = i·2^e/(w·h)`. DiskLight → ellipse area light with `A = πr²`.
  - DistantLight → sun: `E = i·2^e` (normalized); angle_full = `angle`.
- **Unit caveat [UNVERIFIED].** With `metersPerUnit ≠ 1`, "world-space surface area" is ambiguous (scene units² vs m²). Compute it in authored stage units *before* the meters conversion, and log a warning.

---

## 6. Blender 5.1 USD export/import of lights and materials (source-verified)

`source/blender/io/usd/intern/usd_writer_light.cc` (v5.1.2):

| Blender light | USD prim | Attributes written |
|---|---|---|
| AREA SQUARE / RECTANGLE | `RectLight` | width = size; height = size (square) or size_y |
| AREA DISK | `DiskLight` | radius = size/2 |
| AREA ELLIPSE | `DiskLight` | radius = (size + size_y)/4 (lossy) |
| POINT | `SphereLight` | radius = `radius`; `treatAsPoint = (radius == 0)` |
| SPOT | `SphereLight` + `ShapingAPI` | `shaping:cone:angle = deg(spot_size)/2`; `shaping:cone:softness = spot_blend` (lines 102-110) |
| SUN | `DistantLight` | `angle = deg(sun_angle/2)` (**half-angle written into a diameter attribute**, line 121) |
| all | LightAPI | `intensity = energy/4` (SUN, "Unclear why, but approximately matches Karma", l.133-135) or `energy/π` (others, l.137-140); `exposure`; `color`; `enableColorTemperature`, `colorTemperature`; `diffuse = diff_fac`; `specular = spec_fac`; **`normalize = !LA_UNNORMALIZED`** (l.171) |

- **Consistency with Cycles** [INFERENCE, derived]:
  - Rect: `L_usd = (P/π)/A = P/(πA)` = Cycles. ✓
  - Sphere: `(P/π)/(4πr²)` = Cycles sphere. ✓
  - Point limit `P/(4π)`. ✓
  - **DistantLight:** `E_usd = energy/4` vs Cycles `E = energy` gives a **4× discrepancy for any spec-compliant USD renderer**. The angle is also half the true diameter.
  - Area `spread`, `use_soft_falloff`, `use_square`, and non-uniform scale are not representable.
  - Spot softness semantics differ (§3.5).
- **Importer (`usd_reader_light.cc`)** is the exact inverse: `spot_size = 2·rad(cone_angle)` (l.107), `spot_blend = softness` (l.114), `sun_angle = rad(2·angle)` (l.127), `energy = 4·intensity` (sun, l.139) or `π·intensity` (l.143), `× light_intensity_scale` (l.145). `treatAsPoint` → radius 0 (l.90).
- **Our USD loader rule** [INFERENCE]:
  - default = UsdLux spec;
  - if the root-layer documentation starts with `"Blender v"`: `DistantLight` → `E = 4·intensity·2^e`, `angle_full = 2·angle`, and spot softness uses the Blender cos-domain model.
- **Materials** (`usd_writer_material.cc:613-636`), Principled socket → UsdPreviewSurface:
  - Base Color→diffuseColor, Emission Color→emissiveColor (× Emission Strength, **constant values only**: the texture branch ignores the strength, l.199-212 vs the texture path) [INFERENCE from code].
  - Roughness, Metallic.
  - Specular IOR Level→`specular` (non-standard).
  - Alpha→opacity.
  - Transmission Weight→opacity (inverted via scale/bias).
  - IOR→ior. Normal→normal (scale 2 / bias −1 for raw textures).
  - Coat Weight→clearcoat, Coat Roughness→clearcoatRoughness.
  - Height→displacement.
  - Optional `generate_materialx_network` (default False).
- **Mesh:** `doubleSided` comes from the first material's backface-culling flag, defaulting to true (`usd_writer_mesh.cc:650-668`).
- **Operator defaults** (introspected `wm.usd_export`): `export_lights=True`, `export_materials=True`, `generate_preview_surface=True`, `convert_orientation=False` (so Z-up), `export_global_up_selection='Y'` (used only when converting), `xform_op_mode='TRS'`, `triangulate_meshes=False`, `export_subdivision='BEST_MATCH'`, `use_instancing=False`, `convert_world_material=True` (world→DomeLight), `root_prim_path='/root'`, `convert_scene_units='METERS'`, `meters_per_unit=1.0`.
- **Import defaults** (`wm.usd_import`): `import_lights=True`, `light_intensity_scale=1.0`, `apply_unit_conversion_scale=True`, `create_world_material=True`, `import_subdivision=False`, `import_usd_preview=True`, `mtl_purpose='MTL_FULL'`.

**Blender-authored scenes → GLB + USD for the web app.**
- Export GLB with `export_lights=True`. Area lights are lost; the JSON carries them anyway.
- Export USD with defaults: it keeps area lights, is Z-up, and has the sun quirk.
- Author validation materials in the common subset.
- The reference is rendered from a fresh Blender scene that re-imports the same file (§8.5), not from the authoring .blend. That removes exporter losses from the comparison.

---

## 7. Internal scene representation (recommendation)

### 7.1 Geometry
**Phase 1: flattened world-space triangle soup + one BVH.** Justification: the stated requirements are moving camera and moving lights; geometry is static. Lights are analytic (not in the BVH), so moving them never touches the BVH. This gives the simplest traversal and best performance.
- Buffers:
  - `triPos: array<vec4f>`, 3 per triangle (or `v0` + edges `e1`, `e2` for Möller–Trumbore: 48 B/tri).
  - `triIdx: array<vec3u>` into shading-vertex arrays (or a separate index buffer).
  - `triMaterial: array<u32>`: material id + flags (bit: emissive, bit: alpha-tested).
  - `triEmissiveIndex: array<u32>` (0xFFFFFFFF if not emissive; needed for MIS pdf lookup when a BSDF ray hits an emitter).
- Keep **BVH leaf order separate from triangle IDs**: the BVH stores references into a stable `primId` space. ReSTIR reservoirs and reconnection vertices store `(instanceId, primId, barycentrics)`, and those must survive BVH rebuilds. [INFERENCE]

**Phase 2: two-level (TLAS over instances, BLAS per mesh/prototype)** for glTF `EXT_mesh_gpu_instancing`, USD prototypes/PointInstancers and moving objects.
- Per-instance `objectToWorld`, `worldToObject` and **previous-frame `objectToWorld`** (needed for motion vectors and temporal shift re-evaluation).
- Rebuild the TLAS per frame (cheap). BLAS is static, built once in a Worker/WASM or on the GPU.

WebGPU limits: the spec defaults are small (`maxStorageBufferBindingSize` 128 MiB, `maxBufferSize` 256 MiB). Chrome on Apple M-series reports ~4 GiB (4,294,967,292 B) for both on an M3 Pro [UNVERIFIED for M5 Pro; request `adapter.limits` values explicitly in `requestDevice`]. Budget ~100 B/triangle incl. BVH, so 10 M triangles ≈ 1 GB. Also mind `maxStorageBuffersPerShaderStage` (default 8): pack arrays accordingly.

### 7.2 Vertex attributes (shading data)
- `normal` (oct-encoded 2×16 snorm in a u32).
- `tangent` (oct u32 + sign bit in material flags or a separate u8; MikkTSpace).
- `uv0` (2×f32; f16 is risky for tiled UVs).
- `uv1` (optional).
- `color0` (optional, unorm8×4).
- Positions are needed only for intersection.
- Geometric normal from the cross product (flip by `det(M) < 0`, §3.2).

### 7.3 Material table (one struct per material, ~160 B)
```
baseColor: vec4f; emissive: vec3f; emissiveStrength: f32;
metallic, roughness, ior, transmission: f32;
specularFactor: f32; specularColor: vec3f;
clearcoat, clearcoatRoughness, sheenRoughness: f32; sheenColor: vec3f;
attenuationColor: vec3f; attenuationDistance: f32; thicknessFactor: f32;
alphaCutoff: f32; normalScale: f32; flags: u32  (alphaMode, doubleSided, unlit, thinWalled, falloff/validation bits)
tex[N]: u32  (bucket:4 | layer:12 | uvSet:1 | hasXform:1 | xformIndex:14)   N ≈ 10 slots
```
Texture transforms go in a side table of `mat2x3f`.

### 7.4 Textures
- There is no bindless in core WebGPU/WGSL. Use **bucketed `texture_2d_array`s** by size (e.g. 512², 1K², 2K², 4K²) and format (`rgba8unorm-srgb` for color, `rgba8unorm` for data; BC7/ASTC variants when KTX2). Resample non-square or odd sizes into the bucket (keep a per-layer UV scale if padding).
- This stays within `maxSampledTexturesPerShaderStage` (16 default).
- Validation mode: LOD 0 bilinear (matches Cycles). Real-time mode: ray-cone LOD with `textureSampleLevel`.

### 7.5 Lights table
Analytic lights: one 96-B struct. `lightId` stays stable across frames (free-list + generation counter for UI add/delete).
```
p0: vec4f  xyz = position (or unit direction TOWARD light for sun); w = bitcast type {POINT, SPHERE, SPOT, RECT, ELLIPSE, SUN}
p1: vec4f  xyz = axisU (unit); w = halfSizeU | radius
p2: vec4f  xyz = axisV (unit); w = halfSizeV
p3: vec4f  xyz = emission axis n = −Z_world; w = invArea (0 = delta)
p4: vec4f  xyz = radiometric scale in contract units:
                  POINT/SPOT r=0: I = P·c·2^e/(4π)      [W/sr]
                  SPHERE:        L = P·c·2^e/(4π²r²)
                  RECT/ELLIPSE:  L = P·c·2^e/(π·A)
                  SUN:           L = E·c·2^e/(π sin²h)   (or E if h = 0)
           w = flags (visibleToCamera, softFalloffDisk, falloffModel, twoSided)
p5: vec4f  SPOT: cosHalf, smoothInv (or −1 = hard), cosInnerGltf, gltfScale; RECT: tanHalfSpread, normalizeSpread; SUN: cosH, pdfCone
```
- Keep a **previous-frame copy** of the light buffer (temporal reuse needs `p̂` in the prior domain and motion of light-attached samples). [INFERENCE]
- Emissive triangles: a separate world-space buffer (`v0`, `e1`, `e2`, `area`, `primId`, `instanceId`, `avgRadiance`, `twoSided`). Refresh it per frame only for emitters on moving instances.

### 7.6 Light sampling
The Enhanced paper uses power-based light selection and **light tiles**: "each frame precomputes 128 tiles with 1024 lights, from which each 8 × 8 screen tile draws a light tile" (Wyman & Panteleev 2021). It uses **32 NEE candidates at the primary hit** and `32/k²` (min 1) at bounce k, with RIS for NEE and visibility only for the selected light (§6.1 and §7, text lines ~504-509, ~600-603). "Forced NEE light reconnection" relies on replayed random numbers "usually selecting the same light (e.g., with power-based sampling)" (§6.2.3).

Recommendation:
1. **Alias table (Vose) over all lights** (analytic + emissive triangles), with probability ∝ power Φ. Store the normalized `pmf[lightId]` for O(1) pdf lookup; MIS needs it when a BSDF ray hits an emitter.
   - Build it on the CPU. **Rebuild only when intensities, sizes or emitter sets change.** Rigid motion changes neither power nor area.
   - Power proxies:
     - point: `4π·lum(I)`.
     - sphere: `lum(L)·π·4πr²` (= P).
     - spot: `lum(I)·2π[(1 − c_full) + (c_full − c_outer)/2]` with `c_outer = cos(spot_size/2)` and `c_full = c_outer + (1 − c_outer)·blend`. [INFERENCE: exact for a cos-domain smoothstep, whose mean is 1/2]
     - rect/ellipse: `lum(L)·π·A` (× spread factor if spread < π).
     - emissive triangle: `lum(avgL)·A·π·(2 if two-sided)`.
     - sun: `lum(E)·π·R_scene²` (pbrt-style proxy) [UNVERIFIED pbrt detail], or give distant lights a fixed selection probability.
2. **Light tiles** per frame: a compute pass draws 128×1024 i.i.d. samples from the alias table. The effective per-candidate pdf is still `pmf[lightId]·p(point | light)`. [INFERENCE: i.i.d. draws, so unbiased; tile reuse only introduces correlation]
3. **Light BVH later** (Conty Estevez & Kulla 2018; Moreau, Pharr & Clarberg 2019, cited in GRIS; Cycles light tree; Falcor `LightBVHSampler.h`), for scenes with ≥10⁴ emissive triangles and strong locality. For comparison, Falcor offers `EmissiveUniformSampler`, `EmissivePowerSampler` and `LightBVHSampler` (`Source/Falcor/Rendering/Lights/`), and analytic types Point (spot if opening < 2π), Directional, Distant, Rect, Disc, Sphere (`Scene/Lights/LightData.slang:36-44`), which closely matches our type set.

### 7.7 ReSTIR-specific implications [INFERENCE]
- Store light-vertex samples in **light-local parameterization**: `(lightId, u, v)` for analytic lights and `(primId, barycentrics)` for emissive triangles, not world positions. Then a temporal shift of a path ending on a *moving* light follows the light, as in ReSTIR DI.
- Surface reconnection vertices: `(instanceId, primId, barycentrics)`, re-transformed with the *current* instance transform. The previous-frame transform is used where the MIS weight needs the prior domain.
- The GRIS paper says "All light vertices are treated as rough" for reconnection decisions (GRIS text line ~1083). Delta lights (r = 0 point/spot, h = 0 sun) must be flagged: they cannot be reconnected to by area sampling. They can only be reached via NEE (no MIS from BSDF hits).

---

## 8. Shared scene-description JSON (web app ⇄ Blender script)

### 8.1 Principles
1. **Canonical frame = glTF.** Right-handed, +Y up, meters. Camera and lights look down local −Z with +Y up.
2. **Light semantics = Blender's.** Power in W; sun irradiance in W/m²; `spot_size` = full angle; `spot_blend`; area `shape`/`size`/`size_y`/`spread`; `radius` + `softFalloff`; `normalize` fixed to true. The WGSL implements §2 exactly. **No object scale** on lights or camera: Cycles bakes scale into area/spot shape (§2.1).
3. **Resolve animation in one place.** The web app owns a tiny keyframe evaluator (linear position, slerp rotation, linear scalars; step optional). Blender never interpolates: the export includes `frames[]` with *resolved* per-frame world matrices and parameters for every validation frame, and the Blender script keys exactly those values with CONSTANT interpolation. This removes any F-curve or quaternion-interpolation mismatch.
4. **The file is data, versioned** (`"version": 1`), with a JSON Schema kept in the repo.

### 8.2 Schema sketch (valid JSON)
```json
{
  "version": 1,
  "frame": { "upAxis": "Y", "unitMeters": 1.0, "handedness": "right" },
  "assets": [
    { "id": "sponza", "uri": "assets/Sponza/glTF/Sponza.gltf", "format": "gltf",
      "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
      "importFileLights": false, "importFileCameras": false }
  ],
  "world": { "type": "constant", "color": [0.0, 0.0, 0.0], "strength": 1.0 },
  "camera": { "type": "perspective", "yfov": 0.6911, "near": 0.0001,
              "resolution": [1280, 720], "exposureEV": 0.0 },
  "lights": [
    { "id": "key",  "type": "point", "color": [1,1,1], "power": 100.0, "radius": 0.0, "softFalloff": false, "exposure": 0.0, "visibleToCamera": false },
    { "id": "spot", "type": "spot",  "color": [1,0.9,0.8], "power": 500.0, "radius": 0.0, "softFalloff": false,
      "spotSize": 0.785398, "spotBlend": 0.15, "falloffModel": "blender" },
    { "id": "box",  "type": "area",  "shape": "RECTANGLE", "size": 1.0, "sizeY": 0.5, "spread": 3.14159265,
      "color": [1,1,1], "power": 200.0 },
    { "id": "sun",  "type": "sun",   "color": [1,1,1], "irradiance": 3.0, "angle": 0.0 }
  ],
  "tracks": {
    "camera": { "interp": "linear", "keys": [ { "t": 0.0, "position": [0,1.6,4], "rotation": [0,0,0,1] },
                                               { "t": 5.0, "position": [2,1.6,1], "rotation": [0,0.3826834,0,0.9238795] } ] },
    "spot":   { "interp": "linear", "keys": [ { "t": 0.0, "position": [0,3,0], "rotation": [-0.7071068,0,0,0.7071068] } ] }
  },
  "timeline": { "fps": 24, "frameStart": 0, "frameEnd": 120 },
  "render": { "maxBounces": 8, "spp": 16384, "seed": 1, "pixelFilter": { "type": "BOX", "width": 1.0 },
              "transparentMaxBounces": 64, "lightModel": "blender" },
  "frames": [
    { "frame": 0, "camera": { "matrix": [/* 16 floats, column-major, camera-to-world */] },
      "lights": { "spot": { "matrix": [/* 16 */], "power": 500.0 } } }
  ]
}
```
Rotations are glTF-order quaternions `[x,y,z,w]`. Matrices are column-major (glTF/WebGPU). The web app renders from `tracks` live. For validation it evaluates `tracks` at `t = frame/fps`, writes `frames[]`, and renders the same frames.

### 8.3 Mapping to Blender (script side)
- **Axis conversion.** `C = Matrix.Rotation(radians(90), 4, 'X')` maps glTF world (x,y,z) → Blender (x,−z,y), matching the glTF importer (`blender_gltf.py:85-95`).
  - Cameras and lights: **`obj.matrix_world = C @ M_gltf`** (local axes unchanged).
  - Check: local −Z maps to C·(0,0,−1) = (0,1,0), which is glTF −Z. Local +Y maps to (0,0,1), Blender up. ✓
  - Meshes come in through the importer, which does the same conversion.
- **Camera.** `cam.sensor_fit = 'VERTICAL'`; `cam.angle_y = yfov`; `render.resolution_x/y`; `resolution_percentage = 100`; `pixel_aspect = 1`; `clip_start = near` (Cycles clips camera rays at `clip_start`); DOF off; motion blur off.
- **Lights.**
  - point: `type='POINT'`, `energy = power`, `shadow_soft_size = radius`, `use_soft_falloff = softFalloff`, `normalize = True`.
  - spot: add `spot_size`, `spot_blend`, `use_square = False`, `show_cone` irrelevant.
  - area: `shape`, `size`, `size_y`, `spread`.
  - sun: `energy = irradiance`, `angle`.
  - `color`, `exposure`, `use_temperature = False` (bake temperature to RGB in the JSON).
  - Object `visible_camera = visibleToCamera` (Blender default False).
  - `cycles.use_multiple_importance_sampling = True`; `max_bounces` high.
- **World.** Background node color/strength from `world` (the factory default grey 0.0509 must be overwritten).

### 8.4 Cycles settings for an unbiased reference

Defaults as introspected in 5.1.2:

| Setting | Default | Reference value | Why |
|---|---|---|---|
| `scene.render.engine` | BLENDER_EEVEE | `CYCLES` | |
| `cycles.sample_clamp_indirect` | **10.0** | 0 | clamping is biased |
| `cycles.sample_clamp_direct` | 0 | 0 | |
| `cycles.blur_glossy` | **1.0** | 0 | "Filter Glossy" is biased |
| `cycles.caustics_reflective/refractive` | True | True | |
| `cycles.use_denoising` | True | False | |
| `cycles.use_adaptive_sampling` | True | False (fixed spp) | |
| `cycles.pixel_filter_type`, `filter_width` | BLACKMAN_HARRIS, 1.5 | BOX, 1.0 | equals uniform in-pixel jitter |
| `cycles.max_bounces` / diffuse / glossy / transmission / transparent | 12 / 4 / 4 / 12 / 8 | all = JSON `maxBounces`; transparent = JSON value | Cycles `max_bounces = B` ⇒ ≤ B+1 scattering vertices incl. the primary hit (B = 0 is direct lighting only) [INFERENCE from Cycles semantics] |
| `cycles.light_sampling_threshold` | 0.01 | 0 | RR-style culling; unbiased but noisier [UNVERIFIED unbiasedness]; 0 removes doubt |
| `cycles.use_light_tree` | True | either | unbiased |
| `cycles.seed`, `use_animated_seed` | 0, False | fixed | |
| `view_settings.view_transform` | **AgX** | Standard, exposure = `exposureEV`, look None | compare linear data |
| Output | — | OpenEXR float32 | linear radiance |
| `cycles.device` | CPU | GPU (Metal) or CPU | both unbiased |

### 8.5 Reference-generation workflow (import-the-same-file principle)
1. Start Blender with `--factory-startup --background`. Delete the default cube, light and camera.
2. Import each asset with the stock importer:
   - glTF: `bpy.ops.import_scene.gltf(filepath=…, export_import_convert_lighting_mode='SPEC', import_shading='NORMALS', merge_vertices=False)`.
   - USD: `bpy.ops.wm.usd_import(filepath=…, import_lights=False, apply_unit_conversion_scale=True)`.
   Apply `assets[].matrix` via C.
3. If `importFileLights == false`, remove the imported light objects.
4. Create camera, lights and world from the JSON. For each validation frame, set the resolved state and render to `ref_<frame>.exr`.
5. Optional material normalization (e.g. set Principled `distribution='GGX'`), mirrored by a flag the web renderer honors.

This makes the reference **Cycles(BlenderImport(file) + JSON)**, the same definition our loader implements.

---

## 9. Test assets and licenses

glTF, from `KhronosGroup/glTF-Sample-Assets` (license per `Models/<name>/metadata.json`):

| Asset | License | Use |
|---|---|---|
| PointLightIntensityTest | CC0-1.0 | KHR_lights_punctual intensity vs color |
| DirectionalLight | CC0-1.0 | light math verification |
| LightsPunctualLamp | CC-BY-4.0 | point + spot |
| EmissiveStrengthTest, CompareEmissiveStrength | CC-BY-4.0 | emissive_strength |
| MetalRoughSpheres(+NoTextures) | CC-BY-4.0 | BSDF sweep |
| TextureTransformTest / TextureTransformMultiTest | (check metadata) | KHR_texture_transform |
| NormalTangentTest / NormalTangentMirrorTest, OrientationTest | (check) | tangent space, winding, orientation |
| Compare{BaseColor, Metallic, Roughness, IOR, Specular, Transmission, Volume, Clearcoat, Sheen, …} | (check) | per-extension checks |
| TransmissionTest, TransmissionRoughnessTest, IORTestGrid, AttenuationTest, DragonAttenuation (Stanford license + CC0) | mixed | refraction/volume |
| FlightHelmet | CC0-1.0 | textured, KTX2 variant |
| ToyCar | CC0-1.0 | transmission/clearcoat/sheen |
| Sponza (Khronos, Crytek) | **LicenseRef-CRYENGINE-Agreement** (licensing issue #172) | lighting test; do not redistribute publicly |
| PlaysetLightTest | **CC-BY-NC-SA-4.0** (IKEA) | non-commercial only |

Other sources:
- **Intel Sponza** (GPU Research Samples, CC-BY 4.0; glTF/FBX/USD, 4K textures, ~3.6 GB). Better Sponza, and it ships USD too.
- Amazon Lumberyard Bistro (CC-BY 4.0) [UNVERIFIED license text].
- **Cornell box:** there is none in the Khronos assets. **Generate it procedurally with the Blender script**, using Cornell's published geometry/reflectances, and export to GLB + USD. This has zero license risk and is the best first validation scene. Alternative: McGuire's Computer Graphics Archive CornellBox (CC-BY 3.0) [UNVERIFIED].

USD:
- **Pixar Kitchen_set** (openusd.org `dl_kitchen_set.html`): license is "personal, non-commercial testing of Pixar's USD technology" only. Test-only, never ship.
- `usd-wg/assets` (Apache-2.0 repo; per-asset licenses, mostly CC0/permissive): `test_assets` has schema-specific cases (lights, UsdPreviewSurface), `full_assets` has production-like assets (e.g. StandardShaderBall) [UNVERIFIED exact paths].
- Blender-exported USD of our own Cornell box, the primary UsdLux validation.

---

## 10. Risks
1. **Unit inconsistency in Blender's glTF export** (lights ×683, emission ×1). Mitigated by the Blender-equivalent contract. If someone later switches to a "physically consistent photometric" interpretation, Blender scenes will mismatch by 683×.
2. **Spot falloff and point-light geometry mismatches** (glTF vs UsdLux vs Cycles; Cycles default soft-falloff disk). Must be parameterized per light; up to ~0.4 absolute attenuation error otherwise.
3. **Blender USD DistantLight quirks** (intensity/4, half-angle written as diameter) and dropped area-light `spread`/ellipse. Handle with generator detection; keep the sun out of USD validation initially.
4. **Material model mismatch** (Cycles Principled multiscatter GGX / F82 metals vs glTF BRDF). This is the largest risk to "matches Cycles"; it needs a subset or Cycles-equivalent lobes.
5. **LightUSD is an RC and was just renamed.** API churn is possible. Pin the version, use an adapter, run a spike. Needle is technically strongest but has a **non-commercial license**, 35 MB wasm, and COOP/COEP (which also blocks cross-origin assets without CORP headers).
6. **Texture memory.** No bindless; bucketed arrays waste memory on non-power-of-two textures. KTX2/BC7 helps.
7. **Backface/two-sided semantics.** Cycles renders everything two-sided, including emission. Rasterizer-oriented glTF content may rely on backface culling (e.g. single-sided planes intended to be invisible from behind).
8. **Subdivision surfaces in USD** (default `catmullClark`): rendering the cage differs from usdview/Blender (if subdivided).
9. **Validation noise floor.** High-spp Cycles EXRs at 1280×720 with many frames are slow on CPU. Use Metal GPU Cycles, which is unbiased.
10. **Stable IDs.** Any loader or BVH step that reorders triangles or lights breaks temporal reservoirs. Enforce the ID indirection from day one.

## 11. Open questions
- Is the environment only a constant color, or are HDRI domes needed? Blender world equirect orientation (Z-up, u = atan2 convention) vs DomeLight `poleAxis` vs our Y-up needs a verified mapping.
- Does any *geometry* move (skinning, rigid animation)? That decides when phase 2 (TLAS/BLAS) is required.
- Should the UI expose physical units (cd, lm, lux) with conversion to the internal Blender units (÷683), or Blender units directly?
- Which BSDF feature set is the validation target (Diffuse-only first? Principled subset?)
- Is commercial use intended? That rules out Needle USD and Kitchen_set/Sponza-Crytek/IKEA assets for distribution.
- `metersPerUnit ≠ 1` + `normalize`: which area units does Blender's USD importer assume? Needs a test file.
- Exact field set of `getLight()`/`getMeshCopy()` in `lightusd@1.0.0-rc4` (next backend) vs the `dev` branch binding.

## 12. Sources
- Blender 5.1.2 bundled glTF add-on (5.1.20), `/Applications/Blender.app/Contents/Resources/5.1/scripts/addons_core/io_scene_gltf2/`: `__init__.py:185-197,1320-1345`; `blender/exp/lights.py`; `blender/exp/light_spots.py`; `blender/exp/tree.py:286-298`; `blender/exp/nodes.py:203-245`; `blender/exp/material/extensions/{emission,specular}.py`; `blender/imp/light.py`; `blender/imp/blender_gltf.py:85-116`; `blender/imp/pbrMetallicRoughness.py:483-520`; `blender/imp/material.py:84-85`; `blender/com/conversion.py:10,37-70`; `blender/com/blender_default.py:11-15`.
- Blender headless RNA introspection (5.1.2): Light/PointLight/SpotLight/AreaLight/SunLight properties and defaults; `export_scene.gltf`, `import_scene.gltf`, `wm.usd_export`, `wm.usd_import` operator properties; Cycles scene defaults; object ray-visibility defaults; Principled BSDF defaults.
- Blender source at tag v5.1.2 (raw.githubusercontent.com/blender/blender/v5.1.2/…): `intern/cycles/blender/light.cpp`; `intern/cycles/scene/light.cpp` (`Light::area` l.202, packing l.1211-1399); `intern/cycles/kernel/light/{point,spot,area,distant}.h`; `intern/cycles/kernel/closure/emissive.h`; `intern/cycles/util/math_base.h:485-495`; `source/blender/io/usd/intern/usd_writer_light.cc`, `usd_reader_light.cc`, `usd_writer_material.cc`, `usd_writer_mesh.cc`, `usd_capi_export.cc`.
- Khronos: `extensions/2.0/Khronos/KHR_lights_punctual/README.md`; `KHR_texture_transform`, `KHR_materials_emissive_strength`, `KHR_materials_ior`, `KHR_materials_specular` READMEs; `extensions/README.md` (registry); `specification/2.0/Specification.adoc` (l.703-713, 2540-2549); PR #1948 `KHR_lights_area` (closed 2023-10-17); PR #2525 `EXT_lights_area` (open); `glTF-Sample-Assets/Models/model-index.json` and per-model `metadata.json`.
- OpenUSD: `pxr/usd/usdLux/schema.usda` (dev); openusd.org `class_usd_lux_light_a_p_i.html`, `usd_lux_page_front.html`, `spec_usdpreviewsurface.html` (v2.5); AOUSD blog "Announcing OpenUSD v26.03" (wasm32/wasm64, PRs #3832/#3833, wasmFetchResolver); openusd.org `dl_kitchen_set.html` (license).
- npm registry JSON + tarballs: `@gltf-transform/{core,extensions,functions}@4.5.1`, `three@0.186.1`, `@loaders.gl/gltf@4.5.2`, `meshoptimizer@1.3.0`, `draco3dgltf@1.5.7`, `mikktspace@1.1.1`, `ktx-parse@2.0.0`, `@needle-tools/usd@1.1.2` (README, build-info, license history), `tinyusdz@0.9.10`, `lightusd@1.0.0-rc4`, `three-usdz-loader@1.0.9`.
- three.js r186: `examples/jsm/loaders/{GLTFLoader,USDLoader,USDZLoader}.js`, `usd/USDComposer.js`, `libs/basis/basis_transcoder.wasm`.
- glTF-Transform `packages/extensions/src/index.ts`, `packages/functions/src/index.ts`, `packages/core/src/io/web-io.ts`, `ext-meshopt-compression`.
- loaders.gl `modules/gltf/src/lib/extensions/` listing.
- LightUSD/TinyUSDZ: `README.md`, `doc/composition.md`, `doc/api-status.md`, `src/tydra/render-data.hh` (dev), `web/binding.cc` (dev); npm package READMEs.
- needle-tools/usd-viewer GitHub README; Autodesk forum post and Khronos slides on USD/MaterialX on the Web (autodesk-forks/USD `adsk/feature/webgpu`).
- Falcor (NVIDIAGameWorks/Falcor master): `Source/Falcor/Scene/Lights/{LightData.slang,LightCollection.h,FinalizeIntegration.cs.slang,EmissiveIntegrator.3d.slang}`, `Source/Falcor/Rendering/Lights/{EmissivePowerSampler,LightBVHSampler,EmissiveUniformSampler}.h`.
- Papers (local): `restirpt_enhanced_2026.txt` §6.1 (unified direct+indirect, RIS NEE, light tiles 128×1024), §6.2.3 (forced NEE reconnection, power-based sampling), §7 (32 NEE candidates, 32/k² decay); `gris_sig22.txt` (light vertices treated as rough; Falcor implementation; Moreau et al. 2019 citation).
- WebGPU limits (webgpufundamentals limits page; Hack-Log note on 4 GiB limits, M3 Pro Chrome); webgpu.report `texture-compression-bc`; Aras P., "Texture Compression on Apple M1".
- Intel GPU Research Samples (Sponza, CC-BY 4.0).

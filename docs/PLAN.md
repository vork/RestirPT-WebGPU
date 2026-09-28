# ReSTIR PT (Enhanced) in WebGPU, validated against Blender Cycles: implementation plan

## Context

`/Users/mark.boss/Dev/WebGPURestirPT` is empty and is not a git repository.

**Goal.** Build a browser renderer that implements ReSTIR PT:
- **GRIS / ReSTIR PT**: Lin et al., SIGGRAPH 2022.
- **ReSTIR PT Enhanced**: Lin, Kettunen & Wyman, 2026. This is the version we implement.

**Platform.** WebGPU in Chrome 154 on an Apple M5 Pro.
- Metal backend.
- No hardware ray tracing, so BVH traversal runs in compute shaders.

**Requirements.**
- Load GLB and USD in the website.
- Place point, spot and area lights, and animate them.
- Fly camera: WASD + mouse; **E = up, Q = down**.
- Debug visualizations of every pipeline stage.
- Statistical proof of correctness against **Blender 5.1.2 Cycles**, run headless.

**User decisions.**
- Static geometry; only the lights and the camera move.
- glTF metallic-roughness materials, plus glass/transmission.
- A simple denoiser that can be toggled.
- **A full HDRI environment map**: equirectangular .hdr/.exr, importance-sampled, with rotation, strength, tint and camera visibility. It must match Cycles world lighting.
- **Correctness first**, speed second.

**Research basis.** About 11 M agent tokens and 28 agents produced these documents (mapping in §4.2):
- 8 research reports.
- 3 adversarial verifications.
- 1 completeness critique.
- 7 gap-fill specs: lights, temporal, BSDF, rc-predicate, performance, glass, and the HDRI environment (adversarially verified).
- A 4-lens adversarial plan review, with its serious findings independently verified.

They are the normative formula reference: **M0 copies all of them into `docs/research/`**. The "(tag §x)" references below point into them.

---

## 0. Scope

### v1 goals
Everything in Context, plus:
- **Lights:**
  - point and spot, radius 0;
  - rect and disk area lights, spread 0 < s ≤ 180°;
  - sun, angle 0 (glTF directional lights need it);
  - emissive triangles;
  - **HDRI environment map** (§1.4b).
- **Materials:**
  - V1 validation BSDF;
  - V2 Cycles-Principled Tier 1 (GGX);
  - GGX glass.
- **Alpha:** MASK only.

### Non-goals (v1)
- World-lighting variants beyond one equirect HDRI:
  - procedural or Sky-Texture worlds;
  - mirror-ball projection;
  - portals;
  - world volumes;
  - multiple env maps;
  - image-sequence envs.
- Moving or deforming geometry.
- Volumes, including `KHR_materials_volume`. The loader warns about them.
- Alpha BLEND. It is rendered as MASK with cutoff 0.5, and a warning is logged.
- Coat, sheen, SSS, anisotropy, thin film.
- Lights with radius > 0, sun angle > 0, or spread = 0.
- USD Cylinder, Dome and Geometry lights.
- IES profiles.
- A light BVH.

### Stretch (after M8, each with its own gate)
- Ellipse area lights.
- Tier-2 MULTI_GGX, including the glass tables, plus a stock-imported-GLB end-to-end test.
- KTX2/Basis textures.
- A sample gallery.
- USD DomeLight → env. Its orientation convention is unverified.
- An exact 2022 gather-style spatial mode, with 2 shifts per neighbour.
- Rough-transmission reconnection.

---

## 1. Architecture decisions

### 1.1 Stack and platform
**Toolchain.**
- TypeScript + Vite 8 + plain WebGPU. three.js is used only as a parser (EXRLoader, USDLoader fallback).
- WGSL is built by a small in-repo composer (`#include`/`#define`, line map). It has a **feature profile** with fallbacks:
  - immediates → a uniform ring buffer;
  - subgroup ballot → per-item `atomicAdd`;
  - tier-2 storage textures → storage-buffer accumulators;
  - `linear_indexing` → manual index computation.
- UI: Tweakpane 4.
- Analysis: Python 3.13 in a `uv` venv (numpy, scipy, matplotlib, openimageio, flip-evaluator).

**Test lanes.**
- Vitest in two lanes:
  - Node + `webgpu@0.6.1` (dawn.node): a fast pre-check only;
  - **Chrome 154** via Playwright: **authoritative** for every gate.
- M0 diffs the two lanes (limits, features, WGSL language features, FMA/NaN probes) into `docs/platform-lanes.md`.

**Device.**
- `device.ts` requests, for each `max*` limit, min(adapter, **CHROME154_M5PRO profile**):
  - 10 storage buffers (hard limit), 8 storage textures, 48 sampled textures, 16 samplers, 4 bind groups;
  - 64 B immediates, 32 KiB workgroup storage;
  - 4 GiB−4 binding and buffer size;
  - 2048 array layers, 16384 max 2D size, 65535 workgroups per dimension.
- Features: timestamp-query, subgroups (size 32), shader-f16, texture-formats-tier2, float32-filterable.
- Headless runs must reject fallback or software adapters: require vendor `apple`, architecture `metal*`, and `!isFallbackAdapter`.

**Browser constraints.**
- Relaxed Metal math: NaN and Inf checks are **bit tests**, and FMA contraction happens.
- Device loss requires calling `requestAdapter()` again.
- **Never require cross-origin isolation.**
  - No SharedArrayBuffer; single-threaded WASM only.
  - Workers hand data back with transfer lists.
  - Uploads use `mappedAtCreation`, or `writeBuffer` in chunks of at most 64 MB.

### 1.2 Units, frames, pixels, bounces
**Units** (io §1).
- Blender/Cycles radiometric units.
- glTF `KHR_lights_punctual` intensity is divided by 683 (cd, lux). glTF emissive is taken 1:1.
- **scene.json** uses the glTF canonical frame: right-handed, +Y up, metres, **un-recentred** world coordinates. Light *semantics* follow Blender: W, `spot_size`, `spot_blend`, `size`, `spread`.
- Render-internal recentring is applied identically to triangles, camera, lights and tracks.
- `build_scene.py` applies `C = R_x(+90°)` exactly once.

**USD lights, v1 rules.**
- SphereLight becomes a point light with r := 0 and preserved radiant intensity:
  - `normalize` true: I = i·2^e/4;
  - `normalize` false with r > 0: I = i·2^e·πr².
- DistantLight gets angle := 0, with the Blender-author quirk: E = 4·intensity.
- Simplified lights get a UI badge.

**Pixel/raster mapping** (cyc §5.4/§6).
- Image pixel (c, r), with row 0 at the top, covers Cycles raster [c, c+1]×[H−1−r, H−r].
- Camera ray: `d_cam = ((2(c+u)/W−1)·tan(vfov/2)·W/H, (2(H−1−r+v)/H−1)·tan(vfov/2), −1)`.
- The camera uses `sensor_fit='VERTICAL'` with `angle_y`, never `angle_x`.

**Jitter.**
- Uniform per-frame subpixel jitter with a fresh V-buffer each frame. This equals Cycles `BOX`, whose width Blender forces to 1.0.
- Validation jitter is **i.i.d. per run and per frame**: `hash(runSeed, t, p, JITTER)`, separate from the path and resampling streams.
- Interactive mode may use Halton/R2 with a per-run Cranley–Patterson rotation.

**Env mapping** (env §2; Cycles-identical, verified against `projection.h` and `svm/image.h`).
- Let d be a unit glTF-world direction and γ the Blender Mapping-node rotation Z (POINT).
- b = R_z(γ)·C·d = (cγ·d.x + sγ·d.z, sγ·d.x − cγ·d.z, d.y).
- u = (atan2(b.y, b.x) − π)/(−2π), v = (acos(clamp(b.z, −1, 1)) − π)/(−π).
- Orientation: u = 0.5 looks toward +X_glTF, u = 0.25 toward −Z, u = 0.75 toward +Z, and v = 1 is the zenith (+Y).
- The bridge writes γ unchanged; C exists only inside `envUV`.

**Bounces.** `maxBounces` has Cycles semantics everywhere: at most maxBounces+1 scattering vertices, and d ≤ maxBounces+2.
- Each glass interface counts as a bounce.
- Alpha cutouts and Mode-B light pass-throughs consume **no** bounce, RNG dimension or RR.

### 1.3 Geometry, BVH, visibility
**Flattening.**
- World-space triangles with a **stable primId** indirection.
- Degenerate and non-finite triangles are dropped at flatten time and logged.
- Scenes above 2²⁴ triangles are rejected.
- Negative-determinant transforms flip the winding.

**BVH2.**
- Binned SAH in the Aila–Laine layout, built in a TS Worker. Measured: 428 ms for 262k triangles; ~600 / ~200 Mrays/s coherent / incoherent.
- Leaves hold 3–4 triangles; depth is capped at 30.
- Traversal counts stack overflows and iteration-cap exits (cap 1<<16) in the probe buffer. **A non-zero count fails any validation run.**
- CWBVH in M8 gives about 1.2× per frame (gap-perf §2.7).

**Intersection.**
- Möller–Trumbore by default.
- Woop watertight, with canonical global-vertex edge order, for T12, glass and contact scenes.

**One visibility function `visible(a, b)`** for every segment: NEE shadow rays, reconnection (forward and inverse), refresh, and T3 re-tests.
- It tests **triangles only**, with MASK cutout. Glass occludes. Analytic lights never occlude (gap-light §2.5–2.6).
- Wächter–Binder offsets per endpoint, toward the side the segment arrives from.
- `t_max` is the distance between the two offset points, with one shared ε.
- Both endpoint primitives are excluded.

**Alpha MASK** is a geometric cutout inside traversal from M1.
- α = baseColorFactor.a × texture.a × COLOR_0.a, compared against `alphaCutoff` (default 0.5).
- Sampled bilinear at LOD 0 in every pass.
- It creates no vertex and consumes no random numbers.

No previous-frame BVH is needed (gap-temporal §4.2).

### 1.4 Lights (cyc §2, gap-light, glass)
**Storage.**
- Analytic lights live in a separate buffer, **cur/prev double-buffered**, with stable `lightId` and `curToPrev`/`prevToCur` maps. They are never in the BVH.
- Emissive triangles stay in the BVH. They are two-sided, opaque and area-sampled.

**Cycles-exact emission.**
- Point/spot: `I = P/(4π)`. Spot adds the cos-domain smoothstep, which is **not** cone-normalized.
- Rect/disk: `L = P/(πA)`, one-sided, times the spread factor `(tan a − tan θ)/(tan a − a)` with a = s/2 (area.h:106-118).
- Sun: E = P.

**Selection.**
- A global **power alias table**, rebuilt deterministically in stable-id order **only when lights change**. That keeps pmf_t ≡ pmf_{t−1} bitwise otherwise.
- Its marginal pmf never depends on the shading point.
- Every light is sampled area-uniformly in light-local (u,v):
  - rect: uniform;
  - disk: concentric mapping;
  - triangle: b = (1−√u₁, √u₁(1−u₂), √u₁u₂).
- Samplers that depend on the shading point, such as the spherical rectangle, are forbidden, because they break J = 1. The env direction sampler does not depend on the shading point, so it is allowed.
- **Every alias table stores integer u16 thresholds.** That covers the global light table and both env tables. The pmf/pdf used anywhere is the **realized** probability computed from those stored integers, never the target weights.

**Modes.**

| Mode | Behaviour | Blender side | Notes |
|---|---|---|---|
| **A** (validation start) | Analytic lights are NEE-only; **area lights must have visibleToCamera = False** (Cycles 5.1.2 only intersects lights for camera rays when some light has MIS on — M0 finding) | per-light MIS = False | **Not** equivalent to B when x_{d−1} is singular: smooth mirrors and smooth glass never show area lights, and there are no area-light caustics. The UI states this. |
| **B** (product default once Gate 3.11 passes) | Area lights are hittable by BSDF rays, pass through, and use MIS. Each crossed light is its own candidate `(d, BSDF_ANALYTIC, lightId)` | `transparent_max_bounces = 1024` | |
| **A′** (optional) | Area lights are hittable only by rays leaving a singular vertex, with MIS weight 1 | | Same expectation as B |

- The U9 check "A ≡ B" runs **only on scenes without singular lobes**.
- Positive control: a roughness-0 mirror plus a rect light must show A < B, and B ≡ Cycles with MIS on.

**Light-editor properties** (no radius field): type, power (W), colour, exposure, spot size and blend, area size and shape and spread, visibleToCamera.
- Changing the type means remove plus add with a new id.

### 1.4b Environment map (env; verified against the Cycles 5.1.2 kernel)
**Loading.**
- A **custom RGBE decoder** computes `byte·2^(e−136)`, bit-identical to OIIO/Blender.
  - three.js HDRLoader is 0.39% brighter, so it is never used.
- EXR goes through the EXRLoader parser, with lossless codecs only in validation.
- Negative texels are rejected in validation and clamped to 0 in interactive mode.

**GPU texture.**
- `rgba32float`, rows bottom-up, 1 mip.
- Sampler: `repeat`/`repeat` with linear filtering. This replicates Cycles' EXTENSION_REPEAT, including the blend of the top and bottom rows at the poles.
- Lookup is `textureSampleLevel(…, 0)` at Cycles' exact (u, v).
- Env radiance: `L_env = strength·tint ⊙ texel`.
- A 2×2 downsample is allowed only in interactive mode above 4k. `rgba16float` is interactive-only and marked biased.

**Importance sampling.**
- A two-level integer alias table (rows, then columns within a row) over power-of-two cells, with W_m ≥ 4.
- Cell weight = the bilinear-integral kernel (1/8, 6/8, 1/8) of avg(|rgb|) × the row's mean sin θ.
- Defensive floors at 2⁻¹⁰ of the mean.
- `pdfUV` comes from the realized integers.
- NEE samples are clamped inside their own cell.
- `pdf_σ = pdfUV/(2π²·s)`.
  - For directions (escapes, camera misses, refresh), s = ‖b.xy‖.
  - For NEE lattice samples, s = sin(π·min(v, 1−v)).
- The pole cap `s < 1e-6` is applied on both sides.
- Tables are built in a Worker and rebuilt only when the map or the importance resolution changes.

**Selection.**
- The env is one entry, `ENV_ID`, in the global alias table.
- Φ_env = 4π²R_s²L̄_env.
- With other emitters present, P(env) is clamped to [0.1, 0.9].
- One WGSL function, `p1Env(uv) = pmf[ENV]·pdf_σ(uv)`, serves NEE, BSDF-escape MIS, RIS-NEE, the shifts and refresh.

**MIS.** The env **always uses MIS**, in modes A, A′ and B, because escaping BSDF rays always see it.

**Camera misses.** A missed camera ray adds `visibleToCamera·L_env` outside the reservoir with weight 1. This includes rays passing through alpha cutouts and through any camera-visible analytic light, in every mode.

**UI.** An Environment panel with:
- load (file, URL, drop) and CC0 presets;
- strength (log slider), rotation γ, tint;
- visible to camera;
- importance resolution;
- Env NEE on/off (equivalent to Cycles `sampling_method` NONE);
- info readout (resolution, memory, P(env)).

γ and strength are timeline tracks.

### 1.5 Materials (gap-bsdf, glass)
**BSDF variants.**
- **V1 (validation):** Lambert + single-scatter GGX glossy (F ≡ 1) + constant Mix + Emission.
- **V2 (Cycles Principled Tier 1, `distribution='GGX'`):**
  - generalized-Schlick dielectric specular with **LUT layering** over Lambert. Port `table_ggx_gen_schlick_ior_s` (4096 floats) and the exact `lookup_table_read` from a storage buffer;
  - F82-tint metal;
  - Cycles **albedo-scaled sample weights** for lobe selection.
- **Glass** (lobe class G: the Walter/Cycles GGX glass closure):
  - real dielectric Fresnel remapped to [f0, 1];
  - R/T is chosen per microfacet with P_R = avg(R)/avg(R+T);
  - Principled transmission tint is √min(BaseColor, 1) per interface;
  - **no 1/η² radiance scaling**;
  - `η_side` comes from the local backfacing test (1/ior when backfacing). It is never stored and there is no medium stack.
- The glTF `transmission`/`ior` mapping follows Blender's importer (`pbrMetallicRoughness.py:218-254`).

**Lobe codes** (3 bits per vertex, plus a delta bit): `D, S, G_R, G_T, NEE, NONE`.
- A lobe is **delta** when α_x·α_y ≤ 2e-10, and G_T is also delta when |η_side−1| < 1e-4.
- Metal and specular merge exactly into S.

**pdf conventions.**
- **Joint** pdf `p(ω,ℓ)` for throughput and PSS Jacobians.
- **Marginal** pdf for NEE/BSDF MIS and footprints.
- The two are never mixed.

**Support-consistent pdfs.**
- NEE uses Cycles' eval exactly, including the spurious rough-refraction region (bsdf_microfacet.h:621 TODO).
- Every pdf used for MIS, footprints, Jacobians and χ² tests is the **true sampler density**.

**Shading.** Validation uses **flat shading first**. Under smooth shading Cycles' own MIS is not a partition of unity.

### 1.6 Textures
**Validation mode** (every Cycles comparison) **never resamples.**
- One `texture_2d_array` per distinct (w, h), at most 16 per scene, with `mipLevelCount = 1`, format `rgba8unorm`.
- Hardware bilinear filtering with the matching address mode.
- **Decode sRGB after filtering**, in the `svm/image.h` order.
- Blender image interpolation is set to 'Linear'.
- Validation textures have α ≡ 1 in v1.
- `createImageBitmap(…, {colorSpaceConversion:'none', premultiplyAlpha:'none'})`.

**Interactive mode.**
- Size-bucketed arrays (256–2048) with linear-space mips.
- Ray-cone LOD at x₁ only. E_{t−1} uses the previous LOD plane.
- A texture memory budget with downscale warnings.

MikkTSpace tangents. Normal maps use the Cycles node math, in M7.

### 1.7 Path space and RNG
- PSS, lobe-indexed; target `p̂ = luminance(F)`.
- **Index convention**, pasted into `docs/math.md` from gap-rc §1:
  - x₀ = camera, x₁ = primary hit, x_d = light vertex, d ≥ 2 in the reservoir;
  - k ∈ {2..d} ∪ {∅};
  - Falcor `rcVertexLength = k−1`, `pathLength = d−1`;
  - needs-replay ⇔ k > 2 ∨ k = ∅;
  - B = index of x_{d−1}.
- **Counter-based RNG** `u = hash(seed, vertex·D + slot)` with a fixed per-vertex layout:
  - 4 BSDF dimensions (u_lobe, u_h1, u_h2, u_rt);
  - light select + light uv;
  - RR;
  - alpha/crossing dimensions per crossing.
- The resampling RNG is a separate stream.
- Length-1 paths (camera sees an emitter, a camera-visible light, or the env) are added **outside** the reservoir, with weight 1.
- **Background pixels** (no primary hit) have no reservoir. They are never accepted as a spatial partner or used as a temporal source, and the denoiser passes them through.
- **Never compute with IEEE Inf.** Use FLT_MAX (`0x7f7fffff`) or explicit flag bits for "escaped" and for t_max to infinity.

### 1.8 Kernels, queues, pipelines, bindings, submits
**Passes.**
- One compute pass per stage; timestamps exist only at pass boundaries.
- A megakernel inside each stage.
- **Prefix replay is its own compacted indirect pass, with the reconnection fused in.** Measured 1.39× faster, and it needs no replay state.
- The suffix refresh is its own pass and writes scratch only.

**Queues.**
- One `queue` buffer with per-queue headers `{counter, n, capacity, overflowFlag}`. The item regions are sized for the worst case.
- `append()` sets `overflowFlag` instead of writing past the end. **Any overflow fails validation.**
- **2D indirect args**: `(min(g, 65535), ceil(g/65535), 1)`. Kernels compute the item from `workgroup_index` and exit early past `n`. A 1D dispatch silently no-ops above 4.19 M items.

**Compile time.**
- Avoid multiple inlined copies of the BSDF library per pipeline (70–160 ms each). Route every BSDF eval through one call site.
- Traversal call sites are cheap. **Do not** force a single site through a state machine in production (−39%).

**Pipeline variants.**
- Variant axes: reservoir layout, BSDF tier, BVH2/CWBVH, MT/Woop, instrumentation, and single-engine build.
- Only the active variant compiles at startup, in parallel (2–4 s). Others compile lazily on toggle behind a progress indicator, cached by variant hash.
- `debugMode`/`debugTap` are uniforms, not overrides.

**Bindings.**
- A per-pass binding table (extends gap-perf §4.4) keeps every pass ≤ 9 of 10 storage buffers.
- `records` holds:
  - the LUTs;
  - lights cur/prev, including the env record (γ, strength, tint);
  - alias/pmf cur/prev and the id maps;
  - the env `rowAlias`/`colAlias`/`pdfUV` tables.
- `texEnv` + `sEnv` are bound in G0. Env support adds no storage-buffer binding to any pass.
- Arenas are split by write set, because binding the same buffer both read-only and read-write is a validation error.
- Buffers and textures that swap between cur and prev use 2 prebuilt bind-group variants.
- `debugOut` is an `rgba32float` storage texture.
- Present through a fullscreen render pass.

**Submit budget.**
- ≤ 100 ms of GPU time per `queue.submit`, hard cap 200 ms. Row-band or member-group sub-dispatches keep within it.
- PT batches are split into k-spp sub-submits and read back per batch.
- Offline mode splits its 32 trees across dispatches, and persists streaming-RIS Σw in the validation reservoir. It is bitwise equal to a single dispatch (unit test).
- Validation runs **never overlap with Blender Metal renders** (orchestrator GPU lock).

### 1.9 Reservoir and memory
**Validation layout** (uncompressed f32):
- W; F (rgb); initSeed; `endpointId`;
- flags: d, k, technique, endpoint type, isDelta, lobe codes ℓ_{k−1} and ℓ_k (3+1 bits each), mode;
- **c (f32)**;
- rc vertex, one of:
  - `(primId, bary)`;
  - light-local `(lightId, u, v)`;
  - **env NEE**: `(ENV, (i<<16)|j, (du16<<16)|dv16)`, lossless and env-local;
  - **BSDF_ENV escape**: the world ω in rcWi, with `jDen = p^x_{d−1}(ω,ℓ)`.
- technique codes: NEE, BSDF_TRI, BSDF_ANALYTIC, **BSDF_ENV**. Endpoint type `ENV` = 7.
- rcWi, rcRadiance;
- `jDen`, the lobe-joint denominator product;
- `aux p1`, a case-(b)/(c) light pdf measured at x_{d−1} and valid within a frame only;
- the **suffix cache**: x_{d−1} hit, ω_o, β_s, and either (lightIdx, uv) or (t_occ, p2).

**Production layout** (M8): 64 B core plus 32 B suffix cache = 96 B.
- SoA `vec4<u32>` planes in 8×8-tiled order.
- An 8-bit c is allowed only when c_max ≤ 255, saturating. Offline mode needs f32.
- Quantize (u,v) and barycentrics **before** computing F.

**Memory.**
- About 540 B per pixel in the validation layout: roughly 0.57 GB at 1024² (ensemble E = 16).
- Production: about 273 MB at 720p (gap-perf §5).
- Env: about 128 MiB for a 4k `rgba32float` map plus 16–64 MiB of tables. 8k is the cap and costs 512 MiB.
- All allocations are wrapped in `pushErrorScope('out-of-memory')`.

### 1.10 Performance targets
These are **mock-kernel estimates** (gap-perf) at a 10% replay fraction. They exclude alpha, glass and large textures, so expect +10–25%.

| Internal resolution | Role | Estimated frame time |
|---|---|---|
| **960×540** | Interactive default | 28–35 ms, N=3 pairs, CWBVH |
| 1280×720 | Quality mode | 37–51 ms |
| 1920×1080 | Progressive/validation only | — |

- The **internal resolution is independent of devicePixelRatio** from M1 onward. A dropdown offers 540p / 720p / 1080p / native, and a blit upscales to the canvas.
- Interactive defaults: 2 bounces, raised to ≥ 4 when glass is present. RR at initial sampling only.
- An HDRI env adds an estimated 3–8% (measured in M3c).
- M8 adds a dynamic-resolution controller targeting 33 ms.
- Replay fraction f_r is logged from M4 on.

---

## 2. Normative correctness rules

Each rule is implemented once, in a shared WGSL module, and has tests (§7.4). Derivations are in docs/research.

1. **Delta and Mode-A lights.**
   - ω₁ ≡ 1 for delta lights and for Mode-A analytic lights. The `isDelta` and mode flags decide this, never `pdf == 0`.
   - ω₂ = 1 after a delta lobe (MIS skip).
2. **NEE measure and RIS-NEE** (gap-light §3; unbiasedness review R7).
   - Source density: q = P(L)/A_L for area and triangle lights; q = P(L) for delta lights and the sun.
   - **Env:** μ gains a solid-angle component. q = p₁ = pmf[ENV]·pdf_σ(uv), and Λ = L_env(uv), with no r²/cos term. There is also a BSDF_ENV technique with ω₂ = p₂/(M(B)·p1Env + p₂).
   - Λ = L_e|cosθ_z|/r² (area/triangle), I(ω)/r² (delta), E (sun).
   - RIS ratio: r_i = lum(f_cos·Λ_i)/q_i.
   - Stored weight: W_NEE = (1/M)·Σr_i/r_Y.
   - Integrand: F_NEE = T·ω₁·f_cos·Λ·V/q.
   - The solid-angle pdf p₁ = q·r²/|cosθ_z| appears **only** in ω₁ = M(B)p₁/(M(B)p₁+p₂) and ω₂ = p₂/(M(B)p₁+p₂).
   - M(B) is one shared function (32 at B=1 with RIS-NEE, else 1).
   - p₂ is the marginal over non-delta lobes.
   - Light pdfs are recomputed at the offset path's own previous vertex.
   - Never use tile-conditional pmfs.
3. **PSS Jacobian** (enh Eq. 2).
   - `J = p^y_{k−1}(ω′,ℓ_{k−1})·G(y_{k−1}→x_k)·p^y_k(ω_k,ℓ_k) / jDen`.
   - G(a→b) = |n^g_b·ω̂|/‖a−b‖², with the cosine taken at the receiving vertex x_k.
   - Pdfs are joint.
   - p_k := 1 for an NEE-sampled ω_k and for light vertices.
   - The BSDF ratio applies **whether or not MIS is on**.
   - Light-vertex cases:
     - (a) forced NEE rc: J = 1;
     - (d) BSDF-hit emitter as rc: J = p^y_{d−1}·G/jDen, with no pmf factor;
     - temporal: **× J_P = pmf_t(L)/pmf_{t−1}(L) for every NEE-terminated path**, never for BSDF-hit endings, and never folded into jDen;
     - delta lights and sun: never a t² ratio;
     - (e) **BSDF_ENV escape as rc**, a direction copy: J = p^y_{d−1}(ω,ℓ)/jDen with G ≡ 1, p_k := 1, and no pmf or J_P factor;
     - (f) **forced NEE on env**: J = 1 spatially, and × J_P temporally.
4. **Selecting a shifted sample** writes back:
   - F ← F_c(Y) (slot F·J/J);
   - jDen ← the offset path's own product (= J·jDen_src);
   - temporal only: light indices renumbered to frame t, and the refreshed rcRadiance and end-term cache.
5. **Reconnection predicate**, `rc.wgsl` only (gap-rc §0/§3). P_k = D_k ∧ R_k ∧ F_k ∧ I_k for the pair (x_{k−1}, x_k), k ≥ 2.
   - **D_k:** no delta lobe at x_{k−1}, or on the continuing lobe at x_k.
   - **R_k:** r(ℓ_{k−1}) ≥ 0.2 in perceptual roughness. D counts as 1; S uses Roughness. **v1: ℓ ∈ {G_T} never passes**, as ℓ_{k−1} or ℓ_k.
   - **F_k:** t²/(p̄_{k−1}(ω_{k−1})·|cos^g_{x_k}|) ≥ thr.
   - **I_k:** t²/(p̄_k(ω_k)·|cos^g_{x_{k−1}}|) ≥ thr. Skipped for light vertices and for Lambert-only x_k.
   - **Env endpoint** (`kind ENV`): F = +∞ (pass), I is skipped, and **R at x_{k−1} is mandatory**.
     - A BSDF_ENV path with no passing pair gets k* = ∅: full replay, which must escape again at the same d.
     - An occluded `visibleInf` is a defined shift with F = 0.
   - Roughness is always **perceptual** r (GGX α = r²). Test scenes avoid r = 0.2 exactly.
   - thr = 2·10⁻⁴·R²_pri, with R²_pri = ‖x₁−x₀‖²·4π/|⟨n^g_{x₁}, ω̂_{x₁→x₀}⟩|, **of the destination domain**. It is stored per pixel for the current and previous frame.
   - For NEE-final candidates, p̄_k is the all-lobe BSDF pdf of the NEE direction.
   - k* is the first passing pair. With no passing pair, an NEE-ended path is **forced** to reconnect at the light; otherwise k* = ∅ (full replay).
   - There is no lobe revocation.
   - **k* is evaluated in the destination frame's light state.** The refresh never relabels k or class.
6. **Offset checks, O0–O3.** All go through the same `rcPairTest`.
   - **O0:** the replay reaches y_{k−1}; the copied-lobe joint pdfs are > 0; for k = ∅, same d and technique.
   - **O1:** every pair j < k fails. The last pre-rc pair uses **EV_RECONNECT** at y_{k−1}: the copied ℓ_{k−1}, ω′ → x_k, and p̄^y(ω′). Never use the direction replay would sample there. Forced NEE uses EV_RECONNECT_NEE.
   - **O2:** (y_{k−1}, x_k) passes with recomputed pdfs, except for forced NEE.
   - **O3:** for k = ∅, no pair passes.
   - Record margins. Violations are classified LOGIC vs FP-BOUNDARY (gap-rc §10.3).
7. **Sampler-support indicator.** When a shift evaluates a BSDF-sampled segment, multiply f and the joint pdf by 1_supp(ℓ, V, L) from `bsdf_sample_support()`. This applies to ω′ at y_{k−1}, to ω_k at x_k under the new V, and to cases (c)/(d).
   - D: Ng·L > 0.
   - S and G_R: Ns·V > 0 ∧ Ng·L ≥ 0 ∧ Ns·L ≥ 0.
   - G_T: the side tests plus a valid half-vector, H·V > 0 ∧ H·L < 0.
   - NEE segments get no indicator.
8. **Paired spatial reuse** (gap-rc §7). A(p,q) is evaluated **once per pair in canonical index order**, from the G-buffer only. Per-slot status codes distinguish NOT_ACCEPTED (excluded) from FAILED (counts in k, w = 0).
   - With S_c the accepted partners, k = |S_c|, and a_c = c_c/k.
   - Own slot j holds G′_j = F_j(Z_j)·J_{c→j}. The partner's slot holds G_j = F_c(Y_j)·J_{j→c}.
   - `m_j = c_j p̂_j(X_j)/(c_j p̂_j(X_j) + a_c·lum(G_j)) / (k+1)`
   - `m_c = [1 + Σ_j a_c p̂_c(X_c)/(a_c p̂_c(X_c) + c_j·lum(G′_j))] / (k+1)`
   - w_j = m_j·lum(G_j)·W_j, and w_c = m_c·p̂_c(X_c)·W_c.
   - W_Y = Σw/p̂_c(Y), and c_out = c_c + Σ_{j∈S_c} c_j.
   - **RGB shading:** L = m_c·F_c·W_c + Σ m_j·G_j·W_j. Gate-0 check: lum(L) = Σw.
   - The disocclusion boost adds extra pairing textures only as reciprocal pairs.
   - Do **not** copy EvanLuo42's MIS or its shift.
9. **Temporal step: contribution MIS** (gap-temporal §3.6).
   - q′ is a function of the G-buffers, motion vectors and a sample-independent RNG **only**.
   - c_c = 1 and c_p = min(20, c_prev).
   - w̃_c = c_c·lum(F_c)·W_c; w̃_p = c_p·lum(F_t(Y_p))·W_p·J_p. Select s.
   - If s = p: π_c = lum F_t(Y_p) and π_p = lum F_p^stored/J_p.
   - Otherwise: π_c = lum F_c and π_p = lum F_{t−1}(T⁻¹X_c)·J_inv, via the **exact E_{t−1}**.
   - W_Y = π_s/(c_c π_c + c_p π_p)·(w̃_c + w̃_p)/π_c.
   - E_{t−1} consists of: previous camera, the stored jittered previous V-buffer, thr_{t−1}, lightsPrev, pmf_{t−1}, the id maps, the previous LOD plane, and frame config. Any config change resets history.
   - **reservoir_prev is read-only** (storage invariant I1).
   - Talbot-exact is kept as a debug cross-check.
   - Never Falcor's stored/recomputed mix, which is −4.3% biased at a 2× intensity step.
10. **Light changes** (gap-temporal §5; unbiasedness review R2).
    - The refresh scope is **per frame**. If pmf_t ≠ pmf_{t−1}, every light-terminated sample's end term is re-evaluated analytically, with no rays.
    - Shadow rays are needed **only for samples whose own light moved**. Each such refresh uses ≤ 1 ray via the suffix cache.
    - **Never replay RIS-NEE over per-frame light tiles.** It is not a shift, and it is +64% biased.
    - A removed light makes the shift undefined. An added light gets π_p = 0.
    - **Env changes:**
      - **Rotation** counts as "moved". NEE_ENV samples are stored env-local and follow the rotation, at 1 shadow ray to infinity each. BSDF_ENV samples are stored in world space and only re-look-up L and ω₂.
      - **Strength or tint** is radiometric. It needs no rays, and gives J_P on every NEE-terminated sample.
      - **Map swap, importance-resolution change, or Env-NEE toggle** is a config change and resets history.
    - E_{t−1} also holds the previous env record: (cγ, sγ, scale, pmf[ENV]).
11. **Russian roulette** happens only at initial sampling.
    - An NEE candidate ending at x_d survived tests i = 1..d−2. A BSDF-hit candidate survived i = 1..d−1.
    - The UCW is divided by ∏q over exactly those tests.
    - Replay never applies RR.
12. **Glass.**
    - Recompute η and the N/Ng flip at every evaluation, from the evaluating V.
    - Joint pdfs include P_R(H) or (1−P_R(H)) and the refraction half-vector Jacobian η²|H·L|/(V·H+η L·H)².
    - Random replay through glass (including R/T flips and TIR) is the PSS identity, with J = 1.
    - No η² factors in J.
13. **Confidence.**
    - f32 in validation.
    - Spatial confidence is uncapped. The temporal cap is 20.
    - The duplication-map adaptive cap is **biased**. It is off in all unbiasedness gates. Use the corrected n_σ = ⌊σ²/2 + 1.46/σ − 1.76/σ² + 0.656/σ³ + 0.5⌋ (= 128 at σ = 16).
14. **Always:**
    - re-evaluate f at x_k (RTXDI's shortcut is biased);
    - never mix occlusion semantics;
    - quantize lossy fields before computing F.
15. **Env exactness** (env §2, env-verify).
    - The pdf is always taken from the realized alias integers.
    - NEE uv is clamped inside the sampled cell, so `p1Env(uv(i,j,du,dv)) ≡ pdfUV[i,j]`.
    - The pole cap is applied identically on both techniques.
    - One function each for `envUV`, `envRadiance`, `p1Env`, and for `visibleInf(a, ω)` (triangles only, t_max = FLT_MAX).

---

## 3. Per-frame pass graph and modes

\* marks buffers that are cur/prev double-buffered.

0. **CPU.**
   - Tracks.
   - Camera cur/prev.
   - `records`: lights* + maps + pmf*.
   - `lightsChanged` / `pmfChanged` / env moved/radiometric flags.
   - The config hash covers the env map id, the importance resolution, env NEE and the BSDF tier. Any change resets history.
   - Pairing-texture dihedral transform + offset, drawn per round, and per member in ensemble mode.
   - Uniforms.
1. **`primary`:** jittered rays. Outputs:
   - V-buffer* (primId, f32 bary);
   - depth, Ng/Ns, material;
   - `thr*`;
   - motion vectors;
   - dual MVs (M6).
   - On a miss: primId = NONE plus the length-1 env term.
2. **`initial`:** the unified DI+GI path tree.
   - NEE at every vertex, including NEE_ENV.
   - BSDF escapes produce BSDF_ENV candidates.
   - Mode-B crossings.
   - Streaming RIS; the domains are disjoint, so m = 1.
   - Deferred k*.
   - Fills the suffix cache.
   - Writes `reservoir_cur`.
3. **`suffix_refresh`** (only on light- or pmf-change frames). Writes `suffixOut` scratch:
   - fwd: prev samples under S_t;
   - inv: canonicals under S_{t−1}.
   - Class L and N1 end terms are evaluated inside the shifts.
4. **Temporal.**
   - **T1 `temporal_classify`:** choose q′, translate light ids, run inline reconnection-only forward shifts, append replays to Q_f.
   - **T2 `temporal_forward`:** indirect dispatch over Q_f.
   - **T3 `temporal_select`:** choose s. If s = p, finalize. If s = c, append to Q_i.
   - **T4 `temporal_inverse`:** indirect dispatch over Q_i. Computes T⁻¹(X_c) under E_{t−1}, then finalizes.
5. **Spatial (paired).**
   - `pair_accept` → args.
   - `spatial_replay_reconnect`: indirect.
   - `spatial_shift`: shifts the pixel's own path into each partner and writes the slot.
   - `spatial_resample`: rule 8. Writes `reservoir_out` and radiance.
6. **`dupmap`** (optional, biased) → r32uint.
7. **`accumulate` | `denoise`** → `resolve` (exposure/tonemap) → `debug_view` → raster overlay (gizmos, probe paths) → present.

**Modes.**
- **`PT`:** the reference path tracer, using the same modules.
- **`ReSTIR-unbiased`:** validation layout, exact E_{t−1}, no biased options.
- **`ReSTIR-interactive`:** duplication map, dual MV, boost, denoiser, and later the 96 B layout.
- **`ReSTIR-2022-criteria`:** paired uniform-disk maps, the 2022 roughness/distance criteria, fixed cap, no Enhanced extras. This is an A|B baseline.
- **`Offline`:** temporal off, 32 candidates, 3 rounds × 6 paired partners with fresh maps per round, f32 c.

**Ensemble mode** (validation only).
- An atlas of 256² member tiles: E = 16 by default, E = 64 after an M0 allocation probe.
- The member id is included in the seed.
- Pairing uses a per-member transform. A(p,q) requires the same member; cross-member partners are NOT_ACCEPTED.
- Temporal search is rejected outside the member.
- An `ensembleStats` pass outputs per-run tile sums (16², 32², 64²), mask-region sums and global sums, plus per-pixel Σx and Σx² across runs. Only these are read back.

---

## 4. Repository and documentation

### 4.1 Layout (M0)

```
docs/research/        all reports (§4.2), papers + supplemental text, plan-review results
docs/math.md          normative formulas with anchors (index conventions, §2 rules, gap-* formulas); WGSL cites `math.md#anchor`
docs/decisions/       usd.md (spike result), platform-lanes.md, validation.md (coverage table)
src/core/  gpu/ scene/ (gltf, usd adapter, flatten, textures, lights, alias, animation) bvh/ render/ shaders/
           shaders/: common/ bvh/ geom/{intersect,offset,visible} material/{lambert,ggx,fresnel,lut,v1,v2,glass,support}
                     lights/{point,spot,area,sun,emissive,select,measure} path/{pathtree,replay} restir/{rc,reservoir,shift,mis,
                     temporal,spatial,refresh,dupmap,queue} post/{accumulate,atrous,svgf,resolve} debug/
src/app/   main, fly-camera, ui/, light-editor, gizmos, timeline, pixel-inspector, debug-views, reference-compare, export-for-cycles
validation/ blender/{cycles_settings.py (table-driven, read-back asserted), build_scene.py, render_reference.py, make_cornell.py}
            harness/{harness.html, harness.ts, run-chrome.ts, upload-middleware.ts, reference-endpoint.ts (dev only)}
            gpu-tests/  tools/{compare.py, stats.py, report/}  scenes/*.scene.json  budget.json
tests/     CPU tests (alias, pairing involution, BVH vs brute force, loaders, units, IO orientation, f64 dual of k*/O0–O3)
```

### 4.2 Report map

All reports live in `~/.claude/plans/do-a-deep-dive-shimmering-ritchie-agent-<id>.md`. M0 copies each to `docs/research/<name>.md`.

| Tag | Report | id |
|---|---|---|
| gris | GRIS math | a7f08dec9968f0324 |
| gris-verify | Verification of gris | a147516e5d2fb9558 |
| enh | Enhanced paper | a91ab1d861819b276 |
| enh-verify | Verification of enh | acb243dbea3ce2a29 |
| ref | Reference code | a3b89657b22d15f34 |
| dyn | ReSTIR under motion | a668fe13fb61070e8 |
| wgpu | WebGPU platform | a64ceeae6903be301 |
| io | Scene IO | a597df4a1dd101994 |
| cyc | Cycles conventions | a2d6cc1a462b20a80 |
| cyc-verify | Verification of cyc | ac3ddf11ffb75ef90 |
| val | Validation harness | a26c6df2c3f5e4af0 |
| critique | Completeness critique | a40267d04641507fa |
| gap-light | Light integration | a40054d87d055e569 |
| gap-temporal | Temporal with moving lights | a5c2e93ddd1bccd67 |
| gap-bsdf | BSDF / lobe model | aa7778fb0ac69ed41 |
| gap-rc | Reconnection predicate and invertibility | ad74911edeeb27e44 |
| gap-perf | WebGPU pass cost budget | a92ec6ad3bfa4472f |
| glass | Glass / transmission | a4ce50b9899e17ebc |
| env | HDRI environment | aaf24370367ba6bcc |

The env verification notes are in the workflow journal `wf_872e9192-545`. M0 exports them to `docs/research/env-verify.json`.

Also copy:
- the plan-review output: `~/.claude/projects/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-…/tool-results/b27hqv1tz.txt`;
- the Enhanced supplemental text: `…/tool-results/bkusnhdet.txt`;
- the papers from the session scratchpad (`restirpt_enhanced_2026.pdf`, `gris_sig22.pdf`). They can be gitignored and fetched by a script instead.

**Precedence:** this plan §0–§2 > gap-* and glass > *-verify > critique > original reports.

**Superseded statements** (do not follow):
- io: `rgba8unorm-srgb`, import-the-same-file as Stage A, light MIS True by default, SphereLight → true sphere.
- wgpu: area lights as BVH geometry.
- gris: tiny pixel filter.
- val: `+1.76σ⁻²`; "Principled cannot be matched".
- gap-perf: temporal Talbot bindings, which must be re-derived for T1–T4.
- Any claim that "Mode A ≡ Mode B" holds unconditionally.

---

## 5. Milestones

Each milestone ends with `npm run validate -- --milestone Mx` green.

### M0: Bootstrap and spikes
**Build.**
- Scaffold and `git init`.
- Copy the research into docs (§4.2). Seed `docs/math.md` with anchors.
- `device.ts` with the profile clamp. The WGSL composer with its feature profile.
- The upload middleware. The `uv` venv.

**Smoke tests.**
- One WGSL test passes in both lanes. The lane diff is recorded.
- Headless Chrome gives an Apple Metal adapter, not a fallback. Try without `--enable-unsafe-webgpu` first.
- Blender renders a Cornell box to a 32-bit EXR on Metal. **Record s/4096 spp at 512²** (Cornell b=3, b=7, and a Sponza-class scene) in `budget.json`.
- IO orientation test: upload (c, r, c+r) → PFM, which is bottom-to-top → compare.py asserts the orientation. Also check a Cycles EXR.
- An allocation probe for ensemble E = 64.
- **LightUSD rc4 spike (2 days)** in Chrome 154 via its Worker loader, on the Blender-exported Cornell USD. Output: `docs/decisions/usd.md`, which is go / fallback-1 (three.js USDLoader) / fallback-2 (offline Blender conversion).

**Exit.** All smoke tests pass, `budget.json` is populated, and the USD decision is recorded.

### M1: Scene, BVH, camera, debug framework
**Build.**
- glTF loader (gltf-transform + meshopt + draco + MikkTSpace) in a Worker, with transfer lists.
- Flatten; validation and interactive texture paths; MASK cutout.
- SAH BVH2 plus traversal, with counters.
- `visible()`; the `primary` pass.
- Internal resolution + blit. DPR handling.
- **Fly camera:**
  - mouse-look by **RMB-drag** (pointer capture); `F` toggles optional pointer-lock fly mode;
  - LMB reserved for picking and gizmos;
  - WASD moves in the horizontal plane, **E/Q along world ±Y**, using `KeyboardEvent.code` (layout-independent);
  - speed scaled to the scene diagonal; wheel, Shift and Ctrl modifiers;
  - sensitivity and invert-Y stored in localStorage;
  - pitch clamp; Home key; bookmarks; "use file camera";
  - frame-rate independent;
  - record and playback into `tracks.camera`.
- Drag-and-drop and URL loading.
- **Debug framework** (§6):
  - **Env display:**
  - own RGBE decoder, EXR parser, Worker loading;
  - camera sees the env in the background;
  - rotation, strength and tint controls;
  - orientation-grid debug view.
- `debugTap`/`debugMode` uniforms;
  - AOV plane and `debugOut`;
  - probe ring;
  - HUD with NaN/Inf counters and timings;
  - pause, step and freeze controls;
  - raster overlay pass.

**Exit.**
- T12 passes: brute force over 10⁶ rays, watertight Woop, self-intersection, zero overflow count.
- Sponza (MASK foliage) flies at the internal resolution.
- The G-buffer and BVH views work.
- ENV-U1 passes: RGBE decode bit-identical to OIIO.
- ENV-U2 passes: mapping matches an f64 port of Cycles.
- ENV-U7 passes: hardware-bilinear and pole-wrap probe.

### M2: Validation harness and Cycles references
**Build.**
- `make_cornell.py` (GLB + USD).
- The scene.json schema and the **scene bridge**: our parsed scene → `build_scene.py`, with exact normals and V1/V2/glass node trees.
- A minimal `UsdSceneSource`: Mesh, xformOps, upAxis/metersPerUnit, PreviewSurface constants, UsdLux Rect/Disk/Sphere(r=0)/Shaping/Distant.
- `cycles_settings.py`, table-driven; §7.5 lists the settings. The values are read back and written to `manifest.json`.
- `render_reference.py`: K seeds, per-frame renders, caching.
  - Cache key: scene hash + config hash + `validation/blender/*.py` hashes + texture bytes + Blender version.
- The harness page and the Playwright runner.
- `compare.py` with the §7.3 statistics and an HTML report.
- An **emission-only kernel** (primary emission plus camera-visible lights plus the length-1 env).
- **Bridge world export:**
  - the exact GPU texel array is written as a float32 ZIP EXR through Blender's OIIO;
  - node setup: TexCoord.Generated → Mapping(POINT, rot Z = γ) → Environment Texture(EQUIRECTANGULAR, Linear, 'Linear Rec.709', alpha NONE) → tint → Background(strength);
  - a pixel-hash assert (ENV-U9), computed with `foreach_get`.

**Exit (Gate 1-lite).**
- A/A tests on Cycles seed halves pass at the nominal false-positive rate (C0b, (i)).
- Cycles-vs-Cycles planted biases are detected at their MDB: light ×1.0075, and spot blend 0.15→0.16.
- The emission kernel passes C0a (≤ 0.1 px, for all three aspect ratios) and C0b (≤ 1e-4).
- The emission kernel passes **C0p**, the env orientation calibration: markers ≤ 0.1 px, γ ∈ {0, 30°, −90°}, three aspect ratios.
- The C0p planted errors are detected: a u offset of half a texel, a v flip, a γ sign flip, and a missing C.

### M3a: Reference path tracer, lights, editor
**Build.**
- All v1 light types (Mode A), emissive triangles, alias table and `measure.wgsl`.
- V1, then V2 Tier 1 BSDFs.
- NEE + MIS (balance heuristic).
- Bounce semantics; progressive accumulation; a spatial-only à-trous toggle.
- **Light editor:**
  - add point / spot / rect / disk / sun;
  - place on a surface via V-buffer readback;
  - CPU proxy picking;
  - translate and rotate gizmos (no scale);
  - panel in Blender units;
  - delete, duplicate, undo/redo;
  - file lights become editable; emissive meshes are shown as static.
- **cur/prev light buffers + id maps + deterministic pmf rebuild**, as defined in §1.4.
- **Animation:**
  - tracks for the camera and each light (position, rotation, power);
  - keyframes (linear, slerp, step), presets (orbit, bob, sweep), and a timeline;
  - interactive mode runs on wall-clock time; validation uses frame/fps;
  - export resolved `frames[]` with CONSTANT keys.
- **"Export for Cycles"** (dev only): scene.json + parsed scene → `/api/reference` → headless Blender → EXR, auto-loaded in the **compare view** (three.js EXRLoader as the parser).

**Exit.**
- Gate 2 passes on:
  - C0a–C0g, C0l–C0o;
  - (i)–(v), (vii) flat-shaded, (viii) minimal USD, (x), (xi), (xii) alpha;
  - the keyframes of ix-a…g.
- our-PT-vs-Cycles planted biases pass, and the PT A/A test passes.
- scene.json round-trips losslessly.
- gap-temporal §9.3-1 (light maps) passes.

### M3b: Glass and Mode B in the path tracer
**Build.**
- The G lobe (glass spec), with support-consistent pdfs.
- Mode B crossings and Mode A′.
- Cycles settings for glass: `distribution=GGX` on the Glass, Refraction and Principled nodes; pinned caustics flags; MNEE off.

**Exit.**
- Gate 2 passes on C0h–C0k, G1–G10, and (vi) in both modes: Mode A vs Cycles MIS-off, Mode B vs Cycles MIS-on.
- U9 (A ≡ B) passes on delta-free scenes. The mirror positive control shows A < B.
- Rough-glass Mode-B scenes run in the documented **approximate tier**.

### M3c: Environment lighting in the PT
Runs after M3a, in parallel with M3b. Its glass variants follow M3b.

**Build.**
- Importance tables with realized pdfs.
- `p1Env`.
- NEE_ENV, BSDF_ENV and MIS.
- The P(env) power proxy and clamp in the global alias table.
- Env debug views.
- Env timeline tracks (γ, strength).

**Exit.**
- ENV-U3 through ENV-U6 and ENV-U8 pass.
- Gate 2 passes in both modes on:
  - C0q, the constant-env furnace;
  - C0r, HDRI irradiance and mirror spheres checked by f64 quadrature (independent of Cycles);
  - C0s, a single bright "sun" texel, including seam and pole variants, with env NEE both on and off;
  - (xiii), glossy and mirror spheres under an HDRI;
  - (xiv), a Cornell box open to the sky.
- The env planted biases are detected, and the negative controls pass.
- `budget.json` has env rows.

### M4: ReSTIR PT core (spatial, unbiased)
**Build.**
- Unified path tree plus streaming RIS, using the validation layout.
- Hybrid shift with `rc.wgsl`, O0–O3, the support indicator and the write-back.
- Queues with 2D indirect args. Compacted replay + reconnect.
- **The paired pass structure from the start**, using uniform-disk involution maps.
- Defensive pairwise MIS plus RGB weights.
- Offline mode; the `ReSTIR-2022-criteria` mode; ensemble mode + `ensembleStats`.
- The pixel inspector with a 3D path overlay.

**Exit.**
- Gate 0 for M4 tests (§7.4): **LOGIC = 0** in ≥ 10⁷ round trips per case, FP-BOUNDARY ≤ 1e-5, and the T3-D f64 dual agrees.
- Gate 3 rungs 3.1, 3.1b and 3.2 pass on (i)–(vi), (x), (xi), (xii), and on the env scenes C0q(d), C0r, (xiii), (xiv).
- T3-ENV passes with LOGIC = 0 on reconnection cases (e) and (f), on k = ∅, and on N1-env.
- Planted control: an omitted spatial Jacobian is detected.

### M5: Temporal reuse and dynamics
**Build.**
- Temporal passes T1–T4 and `suffix_refresh`/`suffixOut`.
- The exact E_{t−1} and J_P.
- Reciprocal disocclusion boost; dual MVs (or in M6).

**Exit.**
- Gate 3 rungs 3.3–3.6, including the per-frame ensembles for **ix-a…k**:
  - ix-h: env rotating;
  - ix-i: env strength and tint steps, plus a rect-light power step;
  - ix-j: env rotation combined with a flying camera and a moving light;
  - ix-k: env map swap, which must reset history.
- gap-temporal §9 tests.
- Planted controls N1–N7 are detected with their predicted signs.

### M5.5: Denoiser
**Build.** A-SVGF-lite (demodulation, moments, à-trous, α from the ReSTIR temporal gradient) and its debug views.

**Exit.**
- FLIP vs a 64k-spp PT on (i), (v), (vii) and ix-d is ≥ 2× lower than without denoising.
- It recovers 95% within ≤ 8 frames after the ix-e steps.
- ≤ 3 ms at 540p.
- The harness asserts it is off in validation modes.

### M6: Enhanced features and Mode-B ReSTIR
**Build.**
- σ = 16 pairing textures: 254/230/210 px, `rg8sint`, corrected n_σ. These replace the M4 maps without kernel changes.
- RR at initial sampling only.
- RIS-NEE light tiles at x₁.
- Duplication map + adaptive cap (biased, toggle).
- **Mode-B ReSTIR**: BSDF_ANALYTIC candidates, replay crossings, rc case (d). Also covers crossings on rays that escape to the env.
- Env entries in the RIS-NEE light tiles.
- Glass in ReSTIR.

**Exit.**
- Gate 3 rungs 3.7 (per-feature), 3.9 (glass), 3.10 (alpha), 3.11 (Mode B).
- Gate 5 bias budget for the duplication map, measured against Enhanced's 3.25%.
- The interactive default switches to Mode B.

### M7: USD completion, textures, normal maps, fidelity
**Build.**
- Full USD per `usd.md`: PointInstancer, instanceable prims, UsdUVTexture.
- Normal maps with the Cycles node math.
- Smooth shading in ReSTIR.

**Exit.**
- Gate 3 rung 3.8 (smooth + normal-mapped: ReSTIR ≡ PT).
- (vii-N) in the model-approximate tier.
- (vii-L)/(viii-L) loader fidelity.
- **E2E-GLB and E2E-USD** stock-import tests (§7.2).
- **E2E-HDR**: Blender loads the original Poly Haven .hdr and .exr files directly; (xiv)-lite must pass Stage A.

### M8: Performance
**Build.**
- The 96 B production layout; it must pass Gate 3 δ plus T3 round trips on quantized fields, including the oct16 rcWi for BSDF_ENV.
- Branch-to-select micro-optimisations.
- CWBVH (tinybvh WASM, or the verified port in gap-perf §10.2).
- f16 path state.
- Dynamic resolution; a better upscaler.

**Exit.**
- `validate --all` is green in unbiased mode (CI regression).
- Reported performance targets: ≤ 35 ms at 540p with N=3, ≤ 32 ms with N=1 and moving lights, ≤ 51 ms at 720p. Measured with `--enable-webgpu-developer-features` for unquantized timestamps, averaged over ≥ 32 frames. Misses are documented deviations.

---

## 6. Debug visualizations (user requirement)

**Framework (M1).**
- **Stage taps** via the `debugTap` uniform (no recompiles): after initial, temporal and spatial; final; denoised. A|B split.
- A per-pixel **AOV plane** inside the debug binding (~8 MB at 540p), plus the `debugOut` texture.
- A **probe pixel** backed by a 3-deep MAP_READ ring.
- A **raster overlay** depth-tested against the V-buffer.
- **Time controls:** pause, step, freeze seed/frame, freeze or reset history.
- A HUD with per-pass timings (≥ 32-frame averages) and NaN/Inf counters.

| Milestone | Group | Views |
|---|---|---|
| M1 | G-buffer | albedo; Ns/Ng; depth; primId/material; UV; thr; MVs |
| M1 | BVH | steps; box and triangle tests; stack depth and overflow |
| M1 | Env | orientation grid (lat-long lines, axes labelled +X/−Z/+Z/−X glTF, horizon); background mask |
| M3c | Env sampling | importance map (log pdf_uv, realized/target ratio); env sample splat histogram with χ²; escape fraction; NEE-env ω₁ and BSDF-env ω₂; P(env) in the HUD |
| M3a | Path tracer | per-bounce radiance; NEE vs BSDF; ω₁/ω₂; path length; light selection; light gizmos (spot cones, area frames) |
| M4 | Reservoir | c; W; p̂; F·W; d; k; technique; endpoint type; lobe codes |
| M4 | Shifts | outcome codes per slot (ok, NOT_ACCEPTED, FAILED-O0/O1/O2/O3, occluded, support-0, J invalid); log\|J\|; replay mask and queue occupancy and f_r; failing predicate term and margin; footprints vs thr |
| M4 | MIS | m_c; Σm−1 (must be 0); lum(L_rgb)−Σw; per-partner weights |
| M4 | Inspector | reservoir dump; per-candidate m/w/J; 3D overlay of base and shifted paths |
| M5 | Temporal | q′ validity; disocclusion; lightsChanged/refresh mask; π_c/π_p; c_prev/c_out; forward-vs-stored p̂ |
| M6 | Enhanced | pairing offsets (hue = angle, value = length) and reciprocity; duplication map and c_cap |
| M5.5 | Denoiser | variance; history; α |
| M3a | Compare | Cycles EXR split/flip; relative error; t-map |

---

## 7. Verification

### 7.1 Gates and ladder
**Gates.**
- **Gate 0:** unit/property tests (§7.4). LOGIC = 0.
- **Gate 1:** harness calibration (M2 lite, M3a full).
- **Gate 2 (Stage A):** our PT ≡ Cycles.
- **Gate 3 (Stage B):** our ReSTIR ≡ our PT. The ladder below stops at the first failure.
- **Gate 4:** convergence. Slope in [−1.1, −0.9]; no-cap run as a negative control.
- **Gate 5:** bias budgets for the biased options (duplication map, interactive options).

**Gate 3 ladder.**
- 3.1: initial RIS only.
- 3.1b: + RR.
- 3.2: + spatial (offline).
- 3.3: + temporal, static (ensemble).
- 3.4: full.
- 3.5: time-average.
- 3.6: dynamic ix-a…g:
  - ix-a moving point light; ix-b moving area light; ix-c rotating spot; ix-d moving camera; ix-e add/remove/intensity steps;
  - ix-f camera and lights together, a teleport and an FOV ramp;
  - ix-g rotating sun;
  - ix-h env rotation; ix-i env strength/tint and light power steps; ix-j env + camera + light combined; ix-k env map swap.
- 3.7: each unbiased Enhanced feature toggled, plus the 96 B layout (M8).
- 3.8: smooth shading and normal maps.
- 3.9: glass.
- 3.10: alpha.
- 3.11: Mode B.
- ReSTIR vs Cycles is also run as a transitivity check.

### 7.2 Scenes
- **Calibration:**
  - C0a: asymmetric coloured squares at 512², 640×360 and 360×640, plus C0a-far (recentring);
  - C0b: emission units;
  - C0c point, C0d spot, C0e rect, C0m disk, C0n spread 30°/90°, C0l sun;
  - C0f: furnace, `L_e(1−ρ^{b+2})/(1−ρ)` for b ∈ {0,1,3,7};
  - C0g: BSDF furnace;
  - C0o: visibleToCamera;
  - glass: C0h slab `L_e(1−R)²Σ R^{2j}`, C0i immersed emitter (η² convention), C0j rough-glass furnace, C0k glass shadow = 0;
  - glass spec scenes G1–G10;
  - **env:**
    - C0p: orientation calibration with a synthetic octant/marker EXR;
    - C0q: constant-env furnace (ρL; GGX albedo by f64 quadrature). Both variants must agree: constant Background (BSDF-only) and constant texture (NEE);
    - C0r: HDRI irradiance sphere and mirror sphere, checked by f64 quadrature;
    - C0s: single-texel sun at 45°, plus seam and pole variants, with env NEE on and off.
- **Validation:**
  - (i) diffuse Cornell; (ii) point; (iii) spot; (iv) emissive mesh; (v) glossy sweep;
  - (vi) glass/mirror, in both modes; (vi-B) area light in a roughness-0 mirror;
  - (vii) textures, flat; (vii-N) normal-mapped; (vii-L) GLB loader fidelity;
  - (viii) USD; (viii-L) USD loader fidelity vs a `pxr` dump;
  - (x) many lights; (xi) contact geometry; (xii) alpha foliage card;
  - (xiii) glossy and mirror spheres under `studio_small_09`, with perceptual r ∈ {0, 0.05, 0.15, 0.19, 0.21, 0.3, 0.5} plus a V2 metal (heavy-tail tier);
  - (xiv) Cornell box open to the sky under `overcast_soil_puresky` (tight tier) and `kloofendal_48d_partly_cloudy_puresky` (heavy-tail tier), with and without an interior rect light, in modes A and B, plus a glass variant.
- **HDRI assets:** CC0 Poly Haven 1k .hdr files, fetched by `fetch_hdris.ts` with pinned SHA-256.
- **Env tiers:**
  - tight: ≤ 18 EV and no sun or specular-to-diffuse caustic chains;
  - heavy-tail: pilot-sized, with K doubled;
  - the tier is recorded per scene in `report.json`;
  - never clamp to make a scene pass.
- **E2E-GLB:** stock `import_scene.gltf` (SPEC; FLAT, then NORMALS), with post-processing to GGX and the §7.5 settings. Assets: Cornell with point + spot (inner ≠ 0), MetalRoughSpheresNoTextures, TextureTransformTest, NormalTangentMirrorTest, AlphaBlendModeTest (MASK), EmissiveStrengthTest, TransmissionTest/IORTestGrid.
- **E2E-USD:** stock `wm.usd_import`. Inputs:
  - the Blender Cornell;
  - a hand-authored .usda covering upAxis Y and Z, metersPerUnit 0.01, `normalize` true and false, SphereLight treatAsPoint and r = 0.5, ShapingAPI, DistantLight, and a UV flip.
  - Compared against our loader in Blender-compatible mode.
- `docs/decisions/validation.md` holds a **coverage table** mapping every light type and editor property to ≥ 1 Stage-A scene and ≥ 1 Stage-B scene.

### 7.3 Statistics (compare.py)
- **Uncertainty** comes from **replicates of the statistic**: Cycles K seeds, PT batches, ReSTIR runs. Never sum per-pixel variances.
- **Tests:**
  - Welch t with Welch–Satterthwaite degrees of freedom;
  - **TOST** with α = 0.01 per side:
    - PT vs Cycles: δ 0.5% global, 2% per tile;
    - ReSTIR vs PT: δ 0.2% global, 1% per tile;
    - dynamic: global 0.2%, 64² tiles 2%, mask regions 3%;
  - χ²_red, mean-t, KS/AD, Šidák tiles, BH-FDR maps;
  - relMSE_corr, N·MSE, BNR, MAPE, HDR-FLIP.
- **Sizing rule:** choose spp/K/R/B so that SE_Δ ≤ δ/(t₁₋α,ν + z₁₋₀.₀₀₅/ₘ), about δ/6.4 for 256 tiles.
  - Take SE from a ~5% pilot.
  - If this is unaffordable, enlarge the aggregates and record that in `report.json`. Never loosen δ silently.
- **Suite FWER:** α_u = 1 − 0.99^{1/n_units} over scenes × configs × gates × {Y, R, G, B}. A failed unit is re-run once on disjoint seeds.
- **Dark tiles:** use an absolute margin δ·0.05·R̄_image when R̄_tile < 0.05·R̄_image. Zero-variance tiles must match to 1e-6. Guard every 0/0.
- **Calibration:**
  - A/A tests via disjoint splits of existing replicates, with ≥ 20 re-splits;
  - δ-scale plants must fail in ≥ 9/10 repeats:
    - Stage A: light ×1.0075, and one 32² region +3%;
    - Stage B: W ×1.003, and c_p+1 in the MIS denominator.
- **Compute tiers**, using `budget.json` anchors: per-commit ≤ 10 min; nightly ≈ 4–5 h; weekly ≈ 12 h.
  - PT references for dynamic scenes are rendered only at the test frames and at t−1 of each.

### 7.4 Test registry (ID → milestone)
- **M1:** T12, ENV-U1, ENV-U2, ENV-U7.
- **M2:** ENV-U9, the bridge pixel-hash round trip.
- **M3c:**
  - ENV-U3: realized-pdf identity plus GPU χ²;
  - ENV-U4: support;
  - ENV-U5: ∫pdf_σ = 1 ± 1e-5;
  - ENV-U6: NEE/BSDF partition vs quadrature at pmf[ENV] ∈ {1, 0.3};
  - ENV-U8: pmf determinism;
  - env planted biases (strength ×1.0075, missing sin θ, pdf from target weights, ω₂ without pmf[ENV], NEE and BSDF both weight 1);
  - negative controls (no floors, power vs balance heuristic, importance-resolution change).
- **M4:** T3-ENV.
- **M5:** T-ENV-temporal, plus the plants "skip the rotation refresh", "E_{t−1} with γ_t" and "omit J_P on env".
- **M3a:**
  - T1 RNG;
  - T8 BSDF χ²/weights/LUT vectors = gap-bsdf U-1…U-10;
  - T9 light pdf and NEE/BSDF partition = gap-light U1–U4;
  - T10 furnaces;
  - T13 camera, and T13b jitter i.i.d. with positive and negative controls;
  - U11 primary emission;
  - the Offline split-dispatch equality test.
- **M3b:** glass U-G1…U-G10 (including the `ggx_lambda` abs fix), U9 on delta-free scenes plus the mirror positive control, U10 pass-through replay.
- **M4:**
  - T2 replay (compare IDs);
  - T3-0…T3-5 invertibility and T3-D f64 dual;
  - T4 reciprocity; T5 change of variables;
  - T6(a, b) MIS partition and stored-vs-recomputed checks;
  - T7 GRIS toys;
  - U5–U7; gap-bsdf U-11…U-13;
  - dense PSS sweeps (d ≤ 4, 2²⁰ ū per pair);
  - T17 queue overflow and 2D-args coverage (> 4.19 M items).
- **M5:** gap-temporal §9.2 T6(a–c) and §9.3 tests 1–6; N1–N7; the U8 ladder.
- **M6:** T14 pairing involution and σ; T3-3 paired-acceptance symmetry.
- **All milestones:**
  - T15 NaN/Inf = 0;
  - T16 config assertions: validation mode is unbiased, internal scale 1, no upscaler or denoiser, readback of the linear accumulation buffer, jitterMode `iid-per-run`, maxBounces equal to the reference;
  - BVH overflow = 0; queue overflow = 0.

### 7.5 Cycles settings (`cycles_settings.py`, table-driven, read back into manifest.json)
**Sampling and integrator.**
- samples = spp; adaptive sampling off; denoising off (scene and view layer).
- `sample_clamp_direct` and `sample_clamp_indirect` = 0; `blur_glossy` = 0; `light_sampling_threshold` = 0.
- `caustics_reflective` and `caustics_refractive` = True; fast GI and guiding off.
- `max_bounces` = diffuse = glossy = transmission = b; volume = 0; **`transparent_max_bounces` = 1024 always**.
- BOX filter.
- `sampling_pattern = TABULATED_SOBOL`; scrambling distance 1, auto off.
- seed = hash(rep[, frame]); animated seed off.
- film exposure 1, not transparent.
- motion blur, compositing and sequencer off.
- resolution 100%, pixel aspect 1, time limit 0.

**World** (env scenes; world None otherwise).
- The node setup from M2.
- `world.cycles.sampling_method` AUTOMATIC, with a NONE variant for the env-NEE-off test.
- `world.cycles.max_bounces` = 1024, asserted ≥ max_bounces+1. A lower value makes Cycles biased low.
- `is_caustics_light` False.
- World visibility: camera = visibleToCamera, all other ray types True.
- `film_transparent` False; no portals.
- `use_light_tree` True, pinned.
- `texture_limit_render` OFF; `use_simplify` False.
- Assert the working space is 'Linear Rec.709'.
- Assert the world has no Light Path or ray-type nodes.

**Camera.** `sensor_fit` VERTICAL, `angle_y` = vfov, clip 1e-4 / 1e5, no DOF, no shift.

**Lights.**
- `normalize` True; exposure from JSON; temperature off; scale 1.
- `visible_camera` from JSON; per-light MIS from the mode; spread from JSON.
- `shadow_soft_size` 0, `use_soft_falloff` False, sun angle 0.

**Materials and objects.**
- `distribution = 'GGX'` on every node that has the property (Principled, Glass, Refraction, Glossy); Tier 2 is a stretch goal.
- `use_bump_map_correction` False; `emission_sampling` FRONT_BACK, or NONE where the test requires it.
- `shadow_terminator_geometry_offset` 0; MNEE off.
- Image interpolation Linear.

**Output.** EXR 32-bit ZIP, RGB, Standard view.

**Orchestration.** A GPU lock: never run Blender and Chrome GPU jobs at the same time.

### 7.6 Manual end-to-end
1. `npm run dev`.
2. Load Sponza (GLB) and the Cornell USD.
3. Fly with WASD/QE (RMB look).
4. Place a spot light and a rect light on surfaces, and animate them on the timeline.
5. Drop in an HDRI, then rotate it and change its strength.
6. Toggle PT / ReSTIR / Mode A-B / denoiser.
7. Step through the stage taps and debug views.
8. Probe a pixel's paths.
9. Use "Export for Cycles" (with the env) and compare against the auto-loaded Cycles EXR.

# WebGPU/WGSL platform design for a compute-shader ReSTIR PT path tracer
## Target: Chrome 154 on an Apple M5 Pro (16-core GPU, Metal backend)

> Research report, platform track. Written for the engineer implementing the renderer in WGSL/TypeScript.
> Tags: **[MEASURED]** = measured on this Mac during this research; **[SOURCE]** = what a cited source says;
> **[INFERENCE]** = my own reasoning; **[UNVERIFIED]** = plausible but not confirmed.
>
> Where this file lives: the orchestrator asked for `.../scratchpad/research/webgpu-platform.md`, but the session is in
> plan mode, which only allows writing this plan file. Copy it to the research path if you need it there.

---

## 0. Summary of the key numbers

| Question | Answer |
|---|---|
| Installed Chrome | `Google Chrome 154.0.8037.58` (`--version`) |
| Where the measurements ran | Claude desktop's built-in browser (Chromium **152**.0.7977.130, Electron), same M5 Pro, same Dawn/Metal stack. The 153→154 differences are only two WGSL language features, `buffer_view` and `swizzle_assignment` (see §2.4) |
| Adapter | vendor `apple`, architecture `metal-3`, `subgroupMinSize = subgroupMaxSize = 32`, preferred canvas format `bgra8unorm` [MEASURED] |
| Limits you must request | `maxStorageBufferBindingSize` and `maxBufferSize`: 128/256 MiB up to **4 GiB − 4**. `maxStorageBuffersPerShaderStage` 8 → **10**. `maxStorageTexturesPerShaderStage` 4 → **8**. `maxSampledTexturesPerShaderStage` 16 → **48**. `maxComputeWorkgroupStorageSize` 16 KiB → **32 KiB**. `maxComputeInvocationsPerWorkgroup` 256 → **1024**. `maxTextureArrayLayers` 256 → **2048**. `maxTextureDimension2D` 8192 → **16384** [MEASURED] |
| Hard ceilings that cannot be raised | `maxBindGroups = 4`, `maxSamplersPerShaderStage = 16`, `maxUniformBufferBindingSize = 64 KiB`, `maxImmediateSize = 64 B`, `maxComputeWorkgroupsPerDimension = 65535` [MEASURED] |
| Features available | `timestamp-query`, `subgroups`, `subgroup-size-control`, `shader-f16`, `float32-filterable`, `float32-blendable`, `bgra8unorm-storage`, `texture-formats-tier1`/`tier2`, `texture-compression-bc`/`astc`/`etc2` (+ sliced-3d), `primitive-index`, `dual-source-blending`, `clip-distances`, `rg11b10ufloat-renderable`, `depth32float-stencil8`, `depth-clip-control`, `indirect-first-instance`, `texture-component-swizzle`, `core-features-and-limits` [MEASURED] |
| No bindless | `binding_array` is gated behind the `sized_binding_array` language feature, which is "unsafe experimental". Resource tables are experimental (`chromium_experimental_resource_table`). **Plan for texture arrays.** [MEASURED + SOURCE] |
| BVH2 (Aila–Laine layout) throughput in WGSL, Sponza 262k tris, 1920×1080 | Primary rays **~600 Mrays/s**. Diffuse secondary rays **~190 Mrays/s**. Point-light any-hit shadow rays **~610 Mrays/s**. Full 3-bounce path with NEE **~212 Mrays/s**, **77 ms per 1080p frame** [MEASURED] |
| JS binned-SAH build, 16 bins, 262k tris | **428 ms** single-threaded [MEASURED] |
| Reservoir bandwidth | Read + write of a 64 B reservoir at 1080p takes **~1.1 ms**, an effective **~240–250 GB/s** [MEASURED] |
| Pipeline compile time (Tint → MSL → Metal) | Roughly **4 ms per KB of WGSL**: 5 KB ≈ 60 ms, 37 KB ≈ 155 ms, 103 KB ≈ 420 ms. Each distinct `override` set is a separate full compile [MEASURED] |
| Timestamp resolution | Quantized to **65.536 µs** (the Dawn mask `0xFFFF0000` on ns) unless the WebGPU Developer Features flag is on [MEASURED + SOURCE] |
| Floating-point semantics | Chrome compiles MSL with `#pragma METAL fp math_mode(relaxed)`. **`x != x` does not detect NaN** (it was folded to false); use bit tests. `a*b+c` **is contracted to FMA** [MEASURED + SOURCE] |
| Real-time budget estimate | Software BVH2 in WebGPU on the M5 Pro is several times slower than the RTX 5880 Ada (hardware RT) in the Enhanced paper, which runs at 13–15.5 ms at 1080p. Plan on **960×540 to 1280×720 internal resolution** with temporal upscaling, or on CWBVH to recover about 1.5–2× [INFERENCE] |

---

## 1. Environment and measurement method

- The Mac is an **Apple M5 Pro** with a **16-core GPU** and Metal 4 (`system_profiler SPDisplaysDataType`). Apple quotes M5 Pro memory bandwidth at up to ~307 GB/s [SOURCE: flopper.io/Wikipedia, UNVERIFIED for this SKU].
- The installed Chrome is **154.0.8037.58**.
- All GPU measurements ran in the Claude desktop app's built-in browser. Its user agent is `... Claude/2.9939.2 Chrome/152.0.7977.130`, the origin was `https://webgpureport.org`, and the backend is Dawn on Metal, as in Chrome.
  - I did not touch the user's own Chrome profile.
  - Because the harness allowed read-only work only, I wrote no local files. All test code ran in-page and cached nothing to disk.
- **Benchmark scene.** Khronos glTF-Sample-Assets Sponza (262,267 triangles), fetched from raw.githubusercontent.com (CORS `*`).
  - The loader was a minimal inline glTF parser.
  - The BVH builder was an inline JS binned SAH: 16 bins, max leaf 4, SAH with C_trav = C_isect = 1, SAH cost 76.7.
  - The BVH was converted to the tinybvh/Aila–Laine "BVH_GPU" layout: 64 B interior nodes holding both child AABBs, leaves encoded in the child pointer, and triangles as `(v0, e1, e2)` in 48 B.
- **Timing.** GPU time is `timestampWrites` on each compute pass: median of 8–10 passes after 2–3 warm-up passes. Camera inside the atrium, 60° vertical FOV, 1920×1080; every primary ray hits.

---

## 2. Limits, features and language extensions

### 2.1 Adapter limits vs. default device limits [MEASURED]

| Limit | Default device (no `requiredLimits`) | Adapter max (M5 Pro, Chromium 152) | Notes |
|---|---|---|---|
| maxStorageBufferBindingSize | 134,217,728 (128 MiB) | **4,294,967,292** (4 GiB − 4) | Needed: a 1080p reservoir buffer is 132.7 MB at 64 B/px and 182 MB at 88 B/px |
| maxBufferSize | 268,435,456 (256 MiB) | **4,294,967,292** | |
| maxStorageBuffersPerShaderStage | 8 | **10** | Hard budget for the pipeline layout; see §5.2 |
| maxStorageTexturesPerShaderStage | 4 | **8** | |
| maxSampledTexturesPerShaderStage | 16 | **48** | Texture-array design; see §3 |
| maxSamplersPerShaderStage | 16 | 16 | |
| maxUniformBuffersPerShaderStage | 12 | 12 | |
| maxUniformBufferBindingSize | 65,536 | 65,536 | Reported value is tier-capped even though Metal allows more |
| maxBindGroups | 4 | 4 | |
| maxBindingsPerBindGroup | 1000 | 1000 | |
| maxDynamicStorageBuffersPerPipelineLayout | 4 | 8 | |
| maxDynamicUniformBuffersPerPipelineLayout | 8 | 10 | |
| maxComputeWorkgroupStorageSize | 16,384 | **32,768** | Requesting a WG stack array bigger than 16 KiB without raising this fails at pipeline creation (observed error text) |
| maxComputeInvocationsPerWorkgroup | 256 | **1024** | |
| maxComputeWorkgroupSizeX / Y / Z | 256 / 256 / 64 | 1024 / 1024 / 64 | |
| maxComputeWorkgroupsPerDimension | 65,535 | 65,535 | A 1D dispatch of 64-wide groups caps at 4.19 M invocations; use 2D |
| maxTextureDimension1D / 2D / 3D | 8192 / 8192 / 2048 | **16384 / 16384** / 2048 | |
| maxTextureArrayLayers | 256 | **2048** | |
| maxImmediateSize | 64 | 64 | Immediates (Chrome 149+) |
| maxColorAttachmentBytesPerSample | 32 | 128 | |
| maxInterStageShaderVariables | 16 | 28 | |
| maxVertexAttributes | 16 | 30 | |
| min{Uniform,Storage}BufferOffsetAlignment | 256 | 256 | |

**Why these particular numbers.** Chrome/Dawn exposes **tiered limits** for fingerprinting reasons.
- `src/dawn/native/Limits.cpp` defines the tiers. For example:
  - `maxSampledTexturesPerShaderStage` tiers: 16, 16, 16, **48**.
  - `maxStorageBuffersPerShaderStage` tiers: 8, 8, **10**, 16.
  - `maxStorageBufferBindingSize` tiers up to `4 GiB − 4`.
- The Metal backend derives raw limits in `src/dawn/native/metal/PhysicalDeviceMTL.mm` (`InitializeSupportedLimitsImpl`):
  - Storage buffers: 31 Metal buffer slots, minus 1 reserved for buffer lengths, minus 12 uniform and 8 vertex slots, gives 10 storage buffers.
  - Textures: 128 texture slots, split between sampled and storage textures, then tier-capped to 48 and 8 [SOURCE].
- Metal argument buffers (a TODO in Dawn) could raise these later, but do not design for that.

### 2.2 Recommended `requestDevice` (TypeScript)

```ts
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) throw new Error('WebGPU unavailable');
const want: (keyof GPUSupportedLimits)[] = [
  'maxStorageBufferBindingSize', 'maxBufferSize', 'maxStorageBuffersPerShaderStage',
  'maxStorageTexturesPerShaderStage', 'maxSampledTexturesPerShaderStage',
  'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupSizeY', 'maxTextureArrayLayers',
  'maxTextureDimension2D', 'maxDynamicStorageBuffersPerPipelineLayout',
];
const requiredLimits: Record<string, number> = {};
for (const k of want) requiredLimits[k] = adapter.limits[k] as number;
const optional: GPUFeatureName[] = ['timestamp-query', 'subgroups', 'shader-f16', 'float32-filterable',
  'texture-formats-tier2', 'bgra8unorm-storage', 'texture-compression-bc', 'primitive-index'] as GPUFeatureName[];
const requiredFeatures = optional.filter(f => adapter.features.has(f));
const device = await adapter.requestDevice({ requiredLimits, requiredFeatures });
device.lost.then(info => { /* recreate: request a NEW adapter (Chrome 140: requestDevice consumes the adapter) */ });
device.addEventListener('uncapturederror', e => console.error((e as GPUUncapturedErrorEvent).error.message));
```

- Chrome 140 "Device requests consume adapter": after `requestDevice`, the same adapter cannot give you a second device. Device-loss recovery must call `requestAdapter()` again [SOURCE: Chrome 140 notes].
- Chrome 133: unknown limit names can be requested with value `undefined` [SOURCE].

### 2.3 Optional features: what to use them for

| Feature | Present | Use in this renderer |
|---|---|---|
| `timestamp-query` (Chrome 121 inside passes via `timestampWrites`) | yes | Per-pass GPU timing. Apple GPUs only sample at stage boundaries: Dawn enables `chromium-experimental-timestamp-query-inside-passes` only when dispatch-boundary sampling exists, and it is not exposed here. **Put one compute pass around each thing you want to time.** [MEASURED + SOURCE Dawn `PhysicalDeviceMTL.mm` l.683–687] |
| `subgroups` (shipped Chrome 134) | yes, fixed size 32 | Ballot-based stream compaction for the random-replay queue (Enhanced §6.2.2), warp-level prefix sums, `subgroupBroadcastFirst` for per-subgroup atomics. Built-ins must sit in **subgroup-uniform control flow** or compilation fails (observed: "'subgroupBallot' must only be called from subgroup uniform control flow"); use `diagnostic(off, subgroup_uniformity)` deliberately. `subgroup_uniformity` (Chrome 145) makes more values count as uniform |
| `subgroup-size-control` (Chrome 151–152) | yes | Pointless on Apple (min = max = 32), but harmless: `@subgroup_size(32)` |
| `shader-f16` (Chrome 120) | yes | M5 **doubles FP16 ALU throughput** and halves register footprint [SOURCE: Apple M5/A19 tech talk]. Use it for BSDF colour math and packed intermediates in performance builds. The `pack2x16float`/`unpack2x16float` storage packing works without this feature. f16 overflow → inf [MEASURED] |
| `float32-filterable` | yes | Linear filtering of `rgba32float` HDR environment maps (alternatively use `rgba16float`, which is always filterable) |
| `texture-formats-tier2` (Chrome 142) | yes | **`read_write` storage textures** for r8unorm/uint/sint, rgba8unorm/uint/sint, r16uint/sint/float, rgba16uint/sint/float and **rgba32uint/sint/float** [SOURCE: WebGPU spec §25.20]. Lets the reference accumulator do in-place `rgba32float` read-modify-write. Compile tests of rw rgba16float, rgba32float and rgba8unorm all pass [MEASURED]. Implies tier1 (snorm/unorm16 storage) |
| `bgra8unorm-storage` | yes | Write-only compute output straight into the canvas texture (`bgra8unorm` is the preferred format). Read-only bgra8 storage usage was removed in Chrome 143 |
| `texture-compression-bc` (+ `-sliced-3d`) | yes | BC7/BC6H material and environment textures from KTX2/Basis transcoding (§3.3) |
| `primitive-index` (Chrome 142) | yes | Optional raster visibility buffer: `@builtin(primitive_index)` in the fragment stage, followed by exact barycentrics recomputed by a ray/triangle test in compute (§5.2) |
| `float32-blendable`, `dual-source-blending`, `clip-distances` | yes | Not needed for a compute path tracer |
| `texture-component-swizzle` (Chrome 143) | yes | Not needed |

### 2.4 WGSL language extensions (`navigator.gpu.wgslLanguageFeatures`)

[MEASURED] on Chromium 152: `immediate_address_space`, `linear_indexing`, `packed_4x8_integer_dot_product`, `pointer_composite_access`, `readonly_and_readwrite_storage_textures`, `subgroup_id`, `subgroup_uniformity`, `texture_and_sampler_let`, `uniform_buffer_standard_layout`, `unrestricted_pointer_parameters`.

[SOURCE: Chrome 153–154 notes] Chrome 153–154 adds:
- **`buffer_view`**: `bufferView<T>(&buf, byteOffset)`, `bufferArrayView<T>(...)`, `bufferLength()` over an untyped `var<storage> x: buffer`. Useful to pack several logical arrays into one binding under the 10-storage-buffer budget.
- **`swizzle_assignment`** (`v.xy = ...`).
- `texture_formats_tier1` now reported correctly.

Tint's `src/tint/lang/wgsl/feature_status.cc` [SOURCE] classifies them as follows:

| Status | Language features |
|---|---|
| Shipped | `packed_4x8_integer_dot_product`, `pointer_composite_access`, `readonly_and_readwrite_storage_textures`, `unrestricted_pointer_parameters`, `immediate_address_space` |
| Shipped with killswitch | `uniform_buffer_standard_layout`, `subgroup_id`, `subgroup_uniformity`, `texture_and_sampler_let`, `texture_formats_tier1`, `linear_indexing`, `buffer_view`, `fragment_depth`, `swizzle_assignment` |
| **Unsafe experimental** (needs `--enable-unsafe-webgpu`; not for users) | `sized_binding_array`, `texel_buffers`, `chromium_print` |

Practical value of each extension for this project:
- `unrestricted_pointer_parameters`: pass `ptr<storage, ...>` into helpers such as reservoir load/store on different buffers.
- `pointer_composite_access`: `p.field` sugar.
- `immediate_address_space`: `var<immediate>`, up to 64 B set by `pass.setImmediates(offset, typedArray)`. **Works in compute passes**; the test read back 543 [MEASURED]. Use it for per-dispatch constants (pass index, pairing-texture transform, frame seed) instead of `writeBuffer` plus a bind group.
- `linear_indexing`: `@builtin(global_invocation_index)` and `workgroup_index`.
- `uniform_buffer_standard_layout`: std430-like uniform layout.
- `subgroup_id` / `num_subgroups`.

Shader modules that use a language extension should start with `requires <name>;`.

### 2.5 Not available (design around these) [MEASURED compile tests]

- No `binding_array` / bindless (§3).
- No `u64` or `f64`: "unresolved type 'u64'". PCG32 with a 64-bit state must be emulated, so use 32-bit-state generators (§8.5).
- No float atomics: "'atomic' only supports 'i32', 'u32' or 'vec2u'". Accumulate with per-pixel ownership or fixed-point `u32`.
- No recursion ("cyclic dependency found"), function pointers or callable shaders. Materials become a `switch`.
- `textureSample` is not allowed in compute ("built-in cannot be used by compute pipeline stage"). **`textureSampleLevel` and `textureSampleGrad` work in compute**, including on `texture_2d_array` [MEASURED].
- Constant-expression NaN or Inf is a compile error: `bitcast<f32>(0xffffffffu)` gives "value nan cannot be represented as 'f32'", and `0.0/0.0` gives "cannot be represented as 'abstract-float'". Produce sentinels at runtime or use large finite values.
- No hardware ray tracing and no ray queries in browser WebGPU. Bevy Solari's `enable wgpu_ray_query;` is a wgpu-native extension.
- No `writeTimestamp` inside passes on Apple (see above).
- The `mapSync` experiment for workers needs `--enable-features=WebGPUMapSyncOnWorkers`; it was not present here.

### 2.6 Chrome WebGPU timeline for features this project touches [SOURCE: developer.chrome.com "What's New in WebGPU"]

| Chrome | Feature |
|---|---|
| 113 | WebGPU ships (macOS/Metal) |
| 117 | Caching of pipelines with auto layout |
| 119 | float32-filterable |
| 120 | shader-f16; timestamp quantization |
| 121 | Timestamp queries in passes (`timestampWrites`) |
| 123 | DP4a, unrestricted pointer parameters, pointer composite access |
| 124 | Read-only and read-write storage textures |
| 126 | maxTextureArrayLayers increase |
| 130 | **Tint IR: up to 10× faster WGSL→MSL** |
| 131 | `strictMath` shader module option (behind the developer-features flag), subgroup inclusive scans |
| 132 | float32-blendable |
| 134 | **subgroups shipped** |
| 137 | `workgroupUniformLoad` on atomics |
| 139 | `core-features-and-limits` |
| 140 | Device requests consume the adapter |
| 141 | Tint IR complete; integer range analysis to drop robustness clamps |
| 142 | texture-formats-tier1/2, primitive-index |
| 143 | Texture component swizzle; bgra8 read-only storage removed |
| 144 | subgroup_id, uniform_buffer_standard_layout, up to 2× faster writeBuffer/writeTexture |
| 145 | subgroup_uniformity |
| 146 | Transient attachments, texture_and_sampler_let |
| 147–148 | linear_indexing |
| 149–150 | **Immediates** |
| 151–152 | subgroup-size-control |
| 153–154 | buffer_view, swizzle_assignment. Dawn now requires macOS 13+ |

---

## 3. Many material textures without bindless

### 3.1 Constraints

- 48 sampled-texture bindings and 16 samplers per stage.
- Up to 2048 layers per array and 16384² per 2D texture.
- Arrays of texture bindings **cannot be dynamically indexed** in shipping WGSL.
- `texture_and_sampler_let` (Chrome 146) lets you bind a texture to a `let`, but gives no dynamic selection.
- Every texture in a `texture_2d_array` must share size, format and mip count.
- Chrome 154 has no bindless path. The gpuweb bindless/resource-table proposal is still being assessed with experimental implementations [SOURCE: gpuweb proposals/bindless.md, issue #5517].

### 3.2 Options

| Option | Pros | Cons |
|---|---|---|
| **A. A few `texture_2d_array`s by size class** (e.g. 256, 512, 1024, 2048 [, 4096]) × {sRGB rgba8, linear rgba8} (+ BC7 variants) | Hardware wrap, mip and trilinear/anisotropic filtering all work. One `switch` over ≤ 5 classes. Trivial material record `{class, layer}` | Textures must be resampled to the class size (memory waste for odd aspect ratios). A few bindings per format |
| B. Single atlas | One binding | Manual wrap and filtering; mip bleeding across borders; no hardware repeat. Painful with ray cones |
| C. N separate `texture_2d` bindings plus a `switch` | No resampling | ≤ 48 textures; huge `switch`; divergence; code bloat and compile time |
| D. Virtual texturing / software sampling from storage buffers | Unlimited | Manual filtering in shader. Only worth it for very large scenes |

**Recommendation: A.** [INFERENCE]
1. At load, resample each image to a **square** class size `S = clamp(nextPow2(max(w, h)), 256, cap)`, with `cap` = 2048 (1024 on low memory).
   - Stretching a non-square image to a square layer keeps UVs unchanged. Repeat wrap still works because the full image fills the layer. The only cost is anisotropic texel density, which filtering handles.
2. Generate mips with a compute or render pass. WebGPU has no `generateMipmap`; `webgpu-utils` (MIT) has a helper.
3. The material record stores `{classIndex: u8, layer: u16}` per texture slot (baseColor, normal, ORM, emissive, alpha).
4. WGSL: one `switch classIndex` with a `textureSampleLevel(arr_k, samp, uv, layer, lod)` in each case.

**Memory per layer**, including the full mip chain (×4/3):

| Size | rgba8 | BC7 |
|---|---|---|
| 512² | 1.4 MB | 0.35 MB |
| 1024² | 5.6 MB | 1.4 MB |
| 2048² | 22.4 MB | 5.6 MB |

For example, 200 textures at 2048² in rgba8 is ≈ 4.5 GB, which is too much. Cap at 1024 (1.1 GB) or use BC7 (1.1 GB at 2048²).

**Formats.**
- Base colour and emissive: `rgba8unorm-srgb`, so the sampler decodes to linear. Blender/Cycles likewise treats base-colour images as sRGB.
- Normal, ORM and alpha: `rgba8unorm`.
- With `texture-compression-bc`: `bc7-rgba-unorm(-srgb)` via Basis Universal / KTX2 transcoding (KHR_texture_basisu) in a worker (Basis transcoder WASM, Apache-2.0). JPEG/PNG sources would need a runtime BC7 encoder, which is not worth it in v1.
- Apple M5 note: Metal now compresses compute-written textures automatically. Irrelevant for sampled material textures [SOURCE: Apple M5 tech talk].

**Environment map.**
- Store as `rgba16float` (always filterable) or `rgba32float` with `float32-filterable`.
- Importance sampling: a 2D piecewise-constant marginal/conditional CDF, or an alias table, in a storage buffer (pbrt-v4 `PiecewiseConstant2D`; C2-Renderer ports it to WGSL).

**Binding budget.** 4 classes × 2 formats = 8 array bindings, plus env map, blue noise and 3 pairing textures (Enhanced §3.2). That is ≈ 13 of 48 sampled textures and 1–2 of 16 samplers.

### 3.3 Texture LOD in compute

- **Validation mode (vs. Cycles): force LOD 0.** Use `textureSampleLevel(..., 0.0)`. As far as I know Cycles does not mip-map image textures on the GPU [UNVERIFIED, confirm with the Blender agent]. Any LOD filtering changes the integrand and shows up as bias in the comparison.
- **Interactive mode: ray cones** [SOURCE: Akenine-Möller et al., Ray Tracing Gems ch. 20 (2019); JCGT 10(1) "Improved Shader and Texture LOD Using Ray Cones" (2021); formulas from memory, verify against the papers]:
  - Pixel spread angle: `β0 ≈ atan(2·tan(ψ/2) / H)`, with ψ the vertical FOV and H the image height in pixels.
  - Cone width along a segment: `w_i = w_{i−1} + β_{i−1}·t_i`, with `w_0 = 0` at a pinhole camera.
  - At a hit: `β_i = β_{i−1} + 2·β_c`, where β_c is a curvature term (0 for flat).
  - Per-triangle constant: `Δ = ½·log2(t_a / p_a)`, where
    - `t_a = W_tex·H_tex·|(u1−u0)(v2−v0) − (u2−u0)(v1−v0)|`
    - `p_a = ‖(P1−P0)×(P2−P0)‖`, with P in **world** space (apply the instance scale).
  - LOD: `λ = Δ + log2(|w_i|) − log2(|n̂·d̂|)`. Then `textureSampleLevel(arr, s, uv, layer, λ)`. W_tex and H_tex are the **class size**.
  - Anisotropic alternative: ray differentials → `textureSampleGrad(arr, s, uv, layer, ddx, ddy)`. This compiles in compute [MEASURED], and anisotropy then comes from the sampler (`maxAnisotropy` ≤ 16, linear filters only).
- **Interaction with ReSTIR PT** [INFERENCE]:
  - A shifted path's cone differs from the base path's, so LOD-filtered shading makes the integrand depend on the path prefix.
  - That is still a deterministic function of the path, so reuse stays consistent. But the stored reconnection radiance `L_o(x_k)` was computed with the *base* path's LODs.
  - Simplest consistent choice: cone LOD only at the primary hit, LOD 0 (or a fixed coarse LOD for rough bounces) at secondary vertices and in all shift/replay code.
- **The Enhanced paper's "ray footprint" (§4, Eq. 5) is not a texture LOD.** It is the new reconnection criterion [SOURCE: Enhanced p.7]:

  `min( (p^x_{k−1}(ω_{k−1}) G(x_{k−1}→x_k))^{−1}, (p^x_k(ω_k) G(x_k→x_{k−1}))^{−1} ) ≥ (c/100) · ‖x0 − x1‖² / (⟨n_{x1}, x̂1x̂0⟩ / (4π))`,

  with c = 0.02, plus a single-vertex roughness threshold `α_{x_{k−1}} ≥ α_min`.
  - Platform implication: the path tracer must keep, for the primary hit, `‖x0−x1‖²` and `⟨n_{x1}, dir(x1→x0)⟩` (camera position and primary normal; cheap, G-buffer).
  - At each vertex it needs the solid-angle BSDF pdf and the geometry term, which are already available during path construction.

---

## 4. Ray tracing in compute shaders

### 4.1 Measured throughput on the M5 Pro [MEASURED]

Sponza, 262,267 triangles, 1920×1080 (2,073,600 px), BVH2 Aila–Laine layout (131,415 interior nodes; 8.0 MB nodes + 12.0 MB triangles), Möller–Trumbore, workgroup 8×8, traversal stack of 32 entries.

| Kernel | Median GPU time | Rays | Throughput |
|---|---|---|---|
| Primary only (coherent) | 3.47 ms (private stack), 3.67 ms (workgroup stack) | 2.07 M | **~600 Mrays/s** |
| Primary + 1 cosine-diffuse bounce | 14.29 ms (private), 14.88 ms (WG) | 4.15 M | Secondary alone: 10.8 ms → **~192 Mrays/s** |
| Primary + point-light any-hit shadow ray | 6.88 ms | 4.15 M | Shadow alone: 3.4 ms → **~610 Mrays/s** (coherent: all rays go to one light) |
| Path, 1 bounce + NEE at each vertex | 29.0 ms | 8.27 M | 285 Mrays/s |
| Path, 3 bounces + NEE | **77.3 ms** | 16.38 M | **212 Mrays/s** |
| Path, 5 bounces + NEE | 125.8 ms | 24.27 M | 193 Mrays/s |

Workgroup shape (primary / primary+diffuse, ms): 8×4 3.34/14.29; **8×8 3.21/14.29**; 16×8 3.47/14.48; 16×16 3.47/14.75; 32×1 4.13/15.01; 64×1 4.00/15.27; 32×4 3.93/15.20.
- **Use 8×8 (or 8×4) 2D tiles.** 1D rows are about 25% slower for coherent rays.
- A private-memory stack (`var<private> array<u32, 32>`) is slightly faster than a workgroup-memory stack (about 3–6%).

Comparison points [SOURCE]:
- **strahl** (WebGPU OpenPBR path tracer, three-mesh-bvh BVH), M1 Max: 512×512, 100 spp, depth 5 → 2319 ms for 1.07 M triangles (≈ 23 ms/spp). BVH build 328 ms for 1.07 M triangles (arXiv 2407.19977).
- Will Usher's M1 Metal RT dive (2020, pre-hardware-RT): Sponza 20.1 Mrays/s, San Miguel 11.7 Mrays/s. Different workload; historical only.
- The tinybvh README claims up to 5 Grays/s on an RTX 5080 laptop with plain BVH2 compute traversal.

### 4.2 BVH layout choice

1. **Start with BVH2 in the Aila & Laine (2009) layout** (tinybvh `BVH_GPU`) [SOURCE: tinybvh README; tiny.comp].
   - 64 B nodes: `lmin.xyz | left`, `lmax.xyz | right`, `rmin.xyz | pad`, `rmax.xyz | pad`.
   - Child reference: `0x80000000 | (triCount << 24) | firstTri` for leaves. That allows ≤ 127 triangles per leaf and 16.7 M triangles per BLAS; widen the encoding if needed.
   - Triangles are stored in leaf order as `(v0.xyz | primId, e1, e2)`, 48 B.
   - Measured above: simple, ~250 lines of WGSL, directly portable from `tinybvh/external/vulkanrt/shaders/tiny.comp`.
2. **Upgrade to CWBVH** (Ylitie, Karras & Laine 2017) when traversal dominates.
   - 80 B 8-wide nodes, child AABBs quantized to 8 bits relative to a per-node origin and power-of-two scale, and an octant-ordered traversal mask.
   - The paper reports **1.9–2.1× faster incoherent traversal** than the state of the art and **35–60% of the memory** of an uncompressed BVH [SOURCE].
   - tinybvh builds `BVH8_CWBVH`, and its GLSL traversal `tiny8.comp` (≈ 250 lines: `bitCount`, `findMSB`, `floatBitsToUint`, byte unpacking) ports mechanically to WGSL (`countOneBits`, `firstLeadingBit`, `bitcast`, `unpack4xU8`/shifts). Its group stack is only 8 entries deep: "Measured peak: 6 at 450k primitives".
   - Speed-up on Apple GPUs is **[UNVERIFIED]**. Expect a large gain on the diffuse/secondary rays that dominate ReSTIR PT.
3. BVH4_GPU (tinybvh `tiny4.comp`) is an intermediate step.
4. **Triangle formats.**
   - Möller–Trumbore on precomputed `(v0, e1, e2)`: 48 B, fastest to write.
   - Baldwin–Weber (JCGT 5(3) 2016): 3 × vec4 affine transform per triangle; tinybvh's `CWBVH_COMPRESSED_TRIS` uses 4 × vec4 including the primitive index.
   - Woop watertight (JCGT 2(1) 2013): raw vertices plus a per-ray shear (§8.2).
   - Keep shading attributes (normals, UVs, tangents) in a separate buffer indexed by the original primitive ID.
5. **Leaf size** 1–4 triangles; SAH with `C_trav ≈ C_isect`. three-mesh-bvh's default `strategy: CENTER, targetLeafSize: 10` is tuned for CPU raycasting; for path tracing use SAH with 2–4 triangles per leaf.

### 4.3 Traversal kernel skeleton (WGSL, compiled and benchmarked above)

```wgsl
struct Node { lmin: vec4f, lmax: vec4f, rmin: vec4f, rmax: vec4f } // .w = bitcast child refs
@group(1) @binding(0) var<storage, read> nodes: array<Node>;
@group(1) @binding(1) var<storage, read> tris: array<vec4f>;      // v0|primId, e1, e2
const STACK_SIZE: u32 = 32u;
var<private> pstack: array<u32, 32>;
fn safe_rcp(x: f32) -> f32 { let s = select(1.0, -1.0, x < 0.0); return 1.0 / (s * max(abs(x), 1e-8)); }
fn traverse(O: vec3f, D: vec3f, tmaxIn: f32, anyHit: bool) -> vec4f {        // (t, u, v, triIdx bits)
  let rD = vec3f(safe_rcp(D.x), safe_rcp(D.y), safe_rcp(D.z)); let Ord = O * rD;
  var hit = vec4f(tmaxIn, 0.0, 0.0, 0.0); var node = 0u; var sp = 0u;
  loop {
    if ((node & 0x80000000u) != 0u) {                                          // leaf
      let cnt = (node >> 24u) & 127u; var v = (node & 0xffffffu) * 3u;
      for (var i = 0u; i < cnt; i++) {
        let v0 = tris[v]; let e1 = tris[v + 1u].xyz; let e2 = tris[v + 2u].xyz;
        let h = cross(D, e2); let a = dot(e1, h); let f = 1.0 / a;
        let s = O - v0.xyz; let u = f * dot(s, h); let q = cross(s, e1);
        let vv = f * dot(D, q); let d = f * dot(e2, q);
        if (u >= 0.0 && vv >= 0.0 && u + vv <= 1.0 && d > 0.0 && d < hit.x) {
          hit = vec4f(d, u, vv, bitcast<f32>(v / 3u)); if (anyHit) { return hit; } }
        v += 3u;
      }
      if (sp == 0u) { break; } sp -= 1u; node = pstack[sp]; continue;
    }
    let n = nodes[node];
    let t1a = n.lmin.xyz * rD - Ord; let t2a = n.lmax.xyz * rD - Ord;
    let t1b = n.rmin.xyz * rD - Ord; let t2b = n.rmax.xyz * rD - Ord;
    let tmina = max(max(max(min(t1a, t2a).x, min(t1a, t2a).y), min(t1a, t2a).z), 0.0);
    let tminb = max(max(max(min(t1b, t2b).x, min(t1b, t2b).y), min(t1b, t2b).z), 0.0);
    let tmaxa = min(min(min(max(t1a, t2a).x, max(t1a, t2a).y), max(t1a, t2a).z), hit.x);
    let tmaxb = min(min(min(max(t1b, t2b).x, max(t1b, t2b).y), max(t1b, t2b).z), hit.x);
    let hitL = tmina <= tmaxa; let hitR = tminb <= tmaxb;
    let left = bitcast<u32>(n.lmin.w); let right = bitcast<u32>(n.lmax.w);
    if (hitL && hitR) { let lf = tmina <= tminb; node = select(right, left, lf);
      if (sp < STACK_SIZE) { pstack[sp] = select(left, right, lf); sp += 1u; } }
    else if (hitL) { node = left; } else if (hitR) { node = right; }
    else { if (sp == 0u) { break; } sp -= 1u; node = pstack[sp]; }
  }
  return hit;
}
```

Notes:
- **Stack overflow.** tinybvh silently drops pushes when the stack is full. That is incorrect but rare with SAH trees (depth ≲ 30 for millions of triangles). In debug builds, count overflows into a debug buffer.
- **Loop guard** [INFERENCE]: add an iteration cap (e.g. 1 << 16) in debug builds. A NaN ray or corrupt BVH otherwise spins until the Chrome GPU watchdog kills the device; on macOS a runaway shader can even freeze the desktop ("Deathray", auberon.xyz).
- **Stackless / short-stack** traversal (restart trail, parent pointers) mainly saves memory. With M5 dynamic caching, a 32-entry private stack was measured faster than LDS; keep it.
- **Robust slab test** (Ize 2013, "Robust BVH Ray Traversal", JCGT 2(2)): scale `tmax` in the box test by `1 + 2·γ3`, with `γn = nε/(1−nε)` and `ε = 2⁻²⁴`, to be conservative. `safe_rcp` avoids `0·∞ = NaN` when the origin lies on a slab plane.

### 4.4 BVH builders available to a TS/JS app

| Builder | Language / license | Quality | Build speed | GPU layouts | Notes |
|---|---|---|---|---|---|
| **Own TS binned SAH** | TS, yours | SAH 16–32 bins (SAH 76.7 here) | **428 ms / 262k tris** single thread [MEASURED]; parallelize over subtrees in workers | whatever you emit | Full control of layout and leaf encoding; easy TLAS |
| **three-mesh-bvh 0.9.15** | JS, MIT (npm, updated 2026-09-09) | `SAH` with 32 bins (`splitUtils.js` `BIN_COUNT = 32`); default `CENTER`, `targetLeafSize: 10`, `maxDepth: 40` | 1.07 M tris in 328 ms on M1 Max (strahl, strategy unknown) | 32 B BVH2 nodes (bounds + `rightChildOrTriangleOffset`, `splitAxisOrTriangleCount`, depth-first, left = i+1); WGSL traversal in `src/webgpu/*.wgsl.js` (BVHComputeData, ClusteredBVH/TLAS) | Mature, worker-parallel builds (`ParallelMeshBVHWorker`), integrates with three.js GLTFLoader. Node layout is not the fastest for incoherent GPU rays |
| **tinybvh 1.9.x** (Jacco Bikker) | C++ header-only, MIT; README: "Support for WASM / Emscripten" | Binned SAH, SBVH spatial splits, pre-splitting, Bittner optimizer, full-sweep | Natively fastest (README: faster than Embree); WASM scalar path, speed [UNVERIFIED] | BVH_GPU, BVH4_GPU, **BVH8_CWBVH**; TLAS/BLAS | **Not on npm**: build your own WASM (Emscripten, SIMD128). Multi-threading needs cross-origin isolation (COOP/COEP) for SharedArrayBuffer. EA SEED Gigi's WebGPU demo uses tinybvh (BVH_GPU layout) |
| GPU LBVH / PLOC in WebGPU | e.g. AddisonPrairie/WebGPU-LVBH-demo, Skyepulse/Fast-Parallel-BVH-Construction-WebGPU (MIT) | Lower (Morton-order). PLOC is better than LBVH | ms-scale | own | For deforming meshes; v1 does not need it |

**Recommendation** [INFERENCE]:
- **v1:** own TS binned-SAH builder in a Web Worker, emitting the Aila–Laine layout. Or use three-mesh-bvh's SAH build and convert its nodes.
- **v2:** tinybvh compiled to WASM for SBVH plus CWBVH, with the traversal ported from `tiny8.comp`.
- Cache built BLASes per GLB, e.g. in IndexedDB keyed by a mesh hash.

### 4.5 TLAS/BLAS for instanced and moving geometry and lights [INFERENCE unless noted]

- **BLAS** per unique mesh: static, built once, all concatenated in **one nodes buffer and one triangle buffer** with per-BLAS base offsets. This matters for the 10-storage-buffer budget.
- **Instances** record:
  - `objectToWorld` 3×4 (48 B) and `worldToObject` 3×4 (48 B);
  - BLAS root node offset, triangle offset, material base, flags (opaque, alpha-tested, emissive, visible-to-camera);
  - **previous-frame** `objectToWorld` for motion vectors and temporal reuse.
- **TLAS**: a BVH2 over instance world AABBs, rebuilt on the CPU every frame. It is tiny: 1–10k instances build in ≲ a few ms with binned SAH. Upload with `writeBuffer`, which is up to 2× faster since Chrome 144.
- **Two-level traversal** (see tinybvh `kernels/traverse_tlas.cl`):
  - At a TLAS leaf, transform the ray into object space: `O' = W2O·O`, `D' = W2O·D`. Do not renormalize D', so `t` stays comparable across levels.
  - Push a sentinel on the shared stack and continue with the BLAS root. When the sentinel pops, restore the world ray.
- **Rigid animation** (moving meshes and moving emissive area lights) only changes instance transforms and needs a TLAS rebuild, not BLAS rebuilds. **Deforming meshes** need a GPU refit (bottom-up with atomics); defer that.
- **ReSTIR temporal reuse needs the previous frame's scene.**
  - The MIS weights of the temporal shift evaluate target functions and visibility in the previous frame's domain.
  - Keep two TLAS buffers (current and previous) plus previous instance transforms and the previous light buffer.
  - Bevy Solari does exactly this: `trace_visibility_previous_frame`, plus `previous_frame_light_id_translations` for lights added or removed between frames [SOURCE: bevy `crates/bevy_solari/src/realtime/restir.wesl`].
  - Store reconnection vertices as **(instanceId, primId, barycentrics)**, never as world positions, so a reconnection target on a moving object is re-evaluated at its current (or previous) transform. Falcor ReSTIR PT stores `TriMeshHitInfo { instanceID, primitiveIndex, float2 barycentrics }` [SOURCE: `PathReservoir.slang`].
- **Light types.**
  - Point and spot lights are delta lights: NEE only, never hit by BSDF rays. Keep them in a light buffer, not the BVH.
  - Area lights (quad or disk, as in Blender) should be real emissive geometry (triangles in a BLAS, or analytic shapes tested in a small extra loop). BSDF-sampled paths must be able to hit them for MIS and for ReSTIR PT's BSDF-terminated paths.
  - Enhanced §6.1 light tiles: 128 tiles × 1024 presampled lights per frame; each 8×8 screen tile picks one tile; 32 NEE candidates at x1 and 32/k² at bounce k [SOURCE]. That is a 4 MB buffer at 32 B per light sample: a cheap compute pass.

### 4.6 Any-hit (alpha masks), shadow rays, visibility

- Flag alpha-tested triangles with a bit in the triangle record (e.g. the top bit of the primId word) so opaque triangles never pay for UV fetches.
- For a flagged candidate: fetch the 3 UVs, interpolate, `textureSampleLevel(arr, s, uv, layer, 0)`, compare with `alphaCutoff`, and reject the hit if below. The same code runs in closest-hit and any-hit traversal.
- glTF `BLEND` materials: treat as stochastic opacity using a hash of `(pathSeed, bounce, primId)`. It must be **deterministic per path** so random replay reproduces it [INFERENCE].
- Shadow and reconnection visibility: any-hit with early `return` (the `anyHit` flag above). For a segment between two surface points, **offset both ends** (§8.3). Trace with an unnormalized direction `d = p1' − p0'` and `tmax = 1`; slabs and Möller–Trumbore both work with unnormalized directions.

---

## 5. Kernel architecture

### 5.1 Megakernel vs. wavefront in WebGPU

**Wavefront** (Laine, Karras & Aila 2013) needs:
- queues in global memory, atomics, and one pass per stage with `dispatchWorkgroupsIndirect`;
- about 64–128 B of path state per ray, written and read per stage.

At the measured ~250 GB/s, 1 M paths × 4 bounces × 5 stages × 256 B round-trip ≈ 5 GB → ~20 ms of pure traffic. In addition, each indirect dispatch gets a hidden validation dispatch: Dawn's `ComputePassEncoder::TransformIndirectDispatchBuffer` injects a small compute pass that zeroes counts above `maxComputeWorkgroupsPerDimension` [SOURCE: `ComputePassEncoder.cpp` l.98, 311]. **Not worth it for the main path tracer.** [INFERENCE]

**Megakernel per ReSTIR stage** is the natural fit:
- WGSL has no function pointers, so materials are a `switch`.
- Apple M3+/M5 **dynamic caching** allocates registers and private/threadgroup memory from the on-chip cache according to live usage, which softens the classic megakernel occupancy penalty [SOURCE: Apple M3 and M5 tech talks].
- Apple still warns: "If your shaders have too many live registers staying resident for too long ... The GPU lowers occupancy". It recommends expressing repeated calls as **loops** so registers are released per iteration, and avoiding large stack spills [SOURCE: M5 tech talk].
- Keep the bounce loop a real loop; do not manually unroll or duplicate BSDF code for the replay and shift variants.

**Hybrid** [SOURCE Enhanced §6.2.2 + INFERENCE]: use a megakernel per stage plus **stream compaction only where divergence is extreme**. The Enhanced paper compacts pixel–neighbour pairs that need random replay (Table 1: 29.75 → 26.81 ms), which maps directly to the subgroup-ballot compaction plus indirect dispatch in §5.6.

### 5.2 Proposed pass graph for ReSTIR PT Enhanced [INFERENCE, stage content from the Enhanced paper and GRIS]

One command encoder per frame, one compute pass per stage (also gives per-stage timestamps):

0. **CPU**: camera, TLAS rebuild, light buffer (current and previous), pairing-texture transform (random flip/mirror/transpose/offset per frame, Enhanced §3.2), immediates.
1. `presample_lights`: light tiles (128 × 1024).
2. `initial`: primary visibility, unified DI+GI path-tree sampling (Enhanced §6.1), Russian roulette at initial sampling only (§6.2.4), and RIS to one path. Writes `reservoir_cur`, the G-buffer (primary instance/prim/bary, normal, material, footprint data) and seeds.
   - Optional faster variant: a raster visibility buffer (`primitive-index` feature) followed by exact barycentrics from a camera-ray/triangle test in compute.
3. `temporal`: reprojection (camera and instance motion; dual motion vectors, §6.4), adaptive M-cap from the previous duplication map (§5), shift prev→cur and cur→prev (needs the previous-frame TLAS). Writes `reservoir_tmp`.
4. `spatial_prepass`: for each pixel p and each of K = 3 pairing textures, shift p's path to its partner q and store the result.
   - Pixels that need random replay are compacted into a queue, and `dispatchWorkgroupsIndirect` then processes that queue.
5. `spatial_resample`: pairwise-MIS resampling of the canonical path plus K neighbours using the prepass results (each shift computed once, Enhanced §3). Vector-valued weights for colour (§6.3). Writes `reservoir_out` and the radiance.
6. `duplication_map`: per pixel, count reservoirs sharing its random seed in a 17×17 window, divided by 288 (Enhanced §5).
   - Use a workgroup tile of (8+16)×(8+16) = 576 seeds (2.3 KB); 16×16 tiles with 32×32 aprons are 4 KB.
7. `resolve`: accumulate, denoise, tonemap, present. Present with a fullscreen render pass or a write-only `bgra8unorm` storage texture.

Then swap roles: `reservoir_out` becomes next frame's `prev`.

**Storage-buffer budget, ≤ 10 per stage (hard).**
- Merge scene data into ~5 buffers: (a) all BVH nodes (TLAS + BLASes), (b) all triangles, (c) vertex attributes, (d) instances + materials + lights + light tiles (one "scene records" buffer via manual offsets, or `bufferView` in Chrome 153+), (e) env CDF.
- ReSTIR buffers: `reservoir_in`, `reservoir_out`, prepass shift results, compaction list + counters (one buffer with the counter at offset 0).
- Debug buffer: 1.

That fits in 10. Uniforms: frame constants. Immediates: per-dispatch constants. Use sampled textures for the G-buffer and pairing textures.

**Bind groups (4 max)** [INFERENCE]:
- group 0: frame uniforms + env;
- group 1: scene (static per scene, rebound rarely);
- group 2: ReSTIR I/O (two prebuilt bind groups for ping-pong);
- group 3: debug and output.

### 5.3 Compile time and pipeline management [MEASURED + SOURCE]

- **Measured** `createComputePipelineAsync`, cold (source salted):

| WGSL size | Pipeline time |
|---|---|
| 4.6 KB | 58 ms |
| 12.8 KB | 67 ms |
| 37.5 KB | 155 ms |
| 103 KB | 420 ms |

  `createShaderModule` itself takes ≤ 14 ms. **The same module with a different `override` value costs the same again (32–431 ms)**, because Dawn substitutes overrides before the Tint → MSL → `newLibraryWithSource` compile (`ShaderModuleMTL.mm`, `substitute_overrides_config`).
- A full ReSTIR PT stage with BSDF, light sampling, shift and replay code in the 50–150 KB range should compile in about 0.2–0.6 s per pipeline. Expect ~2–4 s total for all stages on first load [INFERENCE].
- Therefore:
  - Create all pipelines with `createComputePipelineAsync` at startup behind a loading screen. Never create them inside the frame loop.
  - Use `override` only for things that change code shape: workgroup size, max bounces, feature on/off switches that remove large code, and debug instrumentation. Use uniforms or immediates for tunables (c = 0.02, α_min, M_cap, radius, K).
  - Chrome caches compiled pipelines in its GPU disk cache (Dawn blob cache; auto-layout caching since Chrome 117), so second loads are faster [SOURCE partially; UNVERIFIED how long entries survive].
- Symbol renaming: Tint strips names (`strip_all_names`). For readable dumps use `--enable-dawn-features=dump_shaders,disable_symbol_renaming`.
- Shader composition: WGSL has no `#include`. Options:
  - template literals (as C2 and Web-RTRT do);
  - **WESL** (`wesl` on npm, MIT; imports and conditional compilation; used by Bevy Solari);
  - `wgsl-linker`.

  Keep a line-map so `getCompilationInfo()` errors (`lineNum`, `linePos`) map back to source chunks.

### 5.4 Register pressure and precision on M5 [SOURCE + INFERENCE]

- M5 doubles FP16 and "complex" ALU throughput.
- M5 exposes occupancy-throttling counters in Xcode/Instruments: `occupancy_target`, `occupancy_target_influence_registers`, `..._l1_cache_pressure`, `..._memory_stalls` [SOURCE: M5 tech talk]. These are only visible with native Metal tools (§7).
- Keep path state compact: throughput `vec3<f16>` (performance builds), pdfs and Jacobians in f32, positions in f32.
- Avoid large private arrays other than the traversal stack. Avoid keeping full `ShadingData` structs live across trace calls; re-fetch material data after the trace.
- **For validation builds keep everything in f32**, so precision loss is not confused with algorithmic bias.

### 5.5 Reservoir storage layout

Sources:
- GRIS/ReSTIR PT reservoir: **88 B per path** [SOURCE: GRIS §8.2 p.75:15; Falcor `PathReservoir.slang` comment "88/128 B"]. Fields:
  - `M` (f32), `weight` (f32)
  - `pathFlags` (u32: path length 4 bits, rc-vertex length 4 bits, delta/transmission/specular bits for two vertices, lastVertexNEE, light type)
  - `rcRandomSeed` (u32), `F` (float3), `lightPdf` (f32)
  - `cachedJacobian` (float3), `initRandomSeed` (u32)
  - `rcVertexHit` {instanceID, primitiveIndex, float2 barycentrics} = 16 B
  - `rcVertexWi` (float3), `rcVertexIrradiance` (float3)
- Enhanced: "reduce reservoir storage from 88 bytes to 64 bytes by removing unnecessary fields and applying lossy compression to selected quantities (details in the supplemental material)" [SOURCE Enhanced §6.2.1]. The supplemental's exact layout is **[UNVERIFIED]**; not read.

**Proposed 64 B layout as 4 × `vec4<u32>`** [INFERENCE; all packing functions compile-tested]:

```
a: W (f32 bits) | pathFlags (u32) | initSeed (u32) | rcSeed (u32, droppable with counter-based RNG §8.5)
b: rcInstanceId (u32) | rcPrimId (u32) | pack2x16unorm(rcBary) | lightPdf (f32)
c: cachedJacobian.x | .y | .z (f32 ×3) | octEncode(rcVertexWi) as pack2x16snorm
d: pack2x16float(F.xy) | pack2x16float(F.z, M) | pack2x16float(Lo.xy) | pack2x16float(Lo.z, spare)
```

Compression cost: F and Lo in f16 (6 B each instead of 12), Wi octahedral (4 B instead of 12), barycentrics unorm16 (4 B instead of 8), M in f16 (exact up to 2048).

**f16 range caveat:** F and L_o above 65504 overflow to **inf** [MEASURED f16 overflow → inf]. Either:
- pre-scale by a per-scene power of two (exact) and unscale on load; or
- store luminance as f32 plus 2 × unorm16 chroma (8 B).

**Lossy fields introduce small bias.** Provide an `override`-selected **uncompressed 128 B debug layout** so the Blender comparison can separate compression bias from algorithm bias.

**AoS vs. SoA** [MEASURED]: read + write of all 64 B per pixel at 1080p:

| Layout | Time | Effective bandwidth |
|---|---|---|
| AoS (four consecutive vec4) | 1.114 ms | 238 GB/s |
| SoA (four planes) | 1.049 ms | 253 GB/s |

The gap is small when all fields are touched. SoA wins for partial reads: the duplication map needs only seeds (plane a); temporal validation and the M-cap need only plane a and part of b.
- **Recommendation:** 4 planes of `array<vec4u>`, indexed with **8×8-tiled pixel order**, `idx = ((y>>3)*(W>>3) + (x>>3))*64 + (y&7)*8 + (x&7)`. This improves cache locality for random spatial neighbours within a 16–30 px radius [INFERENCE].
- An alternative with free 2D tiling: store each plane in an `rgba32uint` texture. Tier 2 allows `read_write`, but that costs storage-texture slots (8 max).

**Memory totals at 64 B** (current + temporal-out + previous = 3 buffers):

| Resolution | Per buffer | 3 buffers |
|---|---|---|
| 960×540 | 33 MB | 100 MB |
| 1280×720 | 59 MB | 177 MB |
| 1920×1080 | 133 MB | 398 MB |

- Paired-spatial prepass results (K = 3 × 16 B per pixel): 100 MB at 1080p.
- A single 1080p buffer at 88 B or at the 128 B debug layout exceeds the default 128 MiB binding limit, which is why raising `maxStorageBufferBindingSize` is mandatory.

### 5.6 Compaction, indirect dispatch, immediates (code compiled and run) [MEASURED]

```wgsl
enable subgroups;
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> list: array<u32>;
@group(0) @binding(3) var<storage, read_write> args: array<u32>;   // usage STORAGE|INDIRECT
fn append_compacted(needs: bool, item: u32) {       // call from subgroup-uniform control flow!
  let ballot = subgroupBallot(needs);
  let cnt = countOneBits(ballot.x) + countOneBits(ballot.y) + countOneBits(ballot.z) + countOneBits(ballot.w);
  let prefix = subgroupExclusiveAdd(select(0u, 1u, needs));
  var base = 0u;
  if (subgroupElect()) { base = atomicAdd(&counter, cnt); }
  base = subgroupBroadcastFirst(base);
  if (needs) { list[base + prefix] = item; }
}
@compute @workgroup_size(1)
fn write_args() { let n = atomicLoad(&counter); args[0] = (n + 63u) / 64u; args[1] = 1u; args[2] = 1u; }
```

Sequence:
1. `encoder.clearBuffer(counter)`.
2. Classify pass (calls `append_compacted`).
3. `write_args` pass.
4. `dispatchWorkgroupsIndirect(args, 0)` for the replay pass.

The test compacted ~30% of 4096 items to 20 indirect workgroups, verified by readback. **Immediates in compute passes** also verified: `pass.setImmediates(0, new Uint32Array([...]))` with `var<immediate> imm: vec4u` and `requires immediate_address_space;`.

### 5.7 Frame budget estimate on the M5 Pro [INFERENCE from §4.1]

Measured anchor: 3 bounces with NEE = 77.3 ms at 1080p (≈ 212 Mrays/s average).

| Stage | 1280×720 (0.92 M px) | 960×540 (0.52 M px) |
|---|---|---|
| Initial path tree (≤ 3 bounces + NEE, RR) | 25–35 ms | 14–20 ms |
| Temporal (≈ 2 incoherent visibility rays/px + shading; replays) | 6–10 ms | 3–6 ms |
| Paired spatial, K = 3 (≈ 3 shifts/px, visibility rays + compacted replays) | 8–14 ms | 5–8 ms |
| Light tiles, duplication map, resolve, reservoir I/O (~1 ms per 64 B read+write at 1080p) | 3–5 ms | 2–3 ms |
| **Total** | **~45–65 ms (15–22 fps)** | **~25–37 ms (27–40 fps)** |

- CWBVH could plausibly cut traversal-bound stages by 1.5–2×.
- Reference point [SOURCE Enhanced Table 1]: total 13.04–15.53 ms at 1920×1080 on an RTX 5880 Ada with hardware RT.
- Recommendation: an internal resolution slider plus temporal upscaling (Web-RTRT ships a WGSL FSR2 port, MIT). The **reference/validation mode** is progressive and resolution-independent.

---

## 6. Open-source WebGPU path tracers and ReSTIR implementations to learn from

| Project | License | What it is | Architecture / BVH | Reusable pieces |
|---|---|---|---|---|
| **Domenicobrz/C2-Renderer** (112★, MIT, last push 2025-07) | MIT | **ReSTIR PT in WebGPU**: hybrid shift, GBH variants (pairwise MIS defensive / complete GBH / biased), env-map importance sampling (pbrt PiecewiseConstant2D), EON diffuse, anisotropic Torrance–Sparrow, multi-scatter LUTs | Svelte + TS. **Single megakernel pipeline** reused for initial and spatial passes; progressive tile-based (not real-time); ping-pong `restirPassBuffer1/2`; checks buffer size against `maxStorageBufferBindingSize`. BVH: JS midpoint split, 64 B nodes, 2 tris/leaf (slow). Reservoir struct ≈ 144 B (my layout computation) | `src/lib/shaders/integrators/ReSTIR-PT/*` (pathConstruction, rrPathConstruction = random replay, gbhVariants, reservoir), `docs/integrators/ReSTIR-PT/randoms.md` (PSS random-number consumption rules), `debuggingReSTIR.md` (debug workflow: 1 candidate, 1 spatial sample, no temporal; compare with the plain PT), a debug buffer plus debug-pixel uniform, `passPerformance.ts` (timestamps) |
| **C-none/Web-RTRT** (MIT, 2024) | MIT | ReSTIR **DI + GI** (not PT) real-time in WebGPU | Raster visibility buffer + compute; BVH built in a worker; **64 B packed reservoir** `array<u32,16>` with `pack2x16float` and oct normals | WGSL **FSR2 port**, SVGF-style à-trous denoiser, TAA, packing helpers |
| **gkjohnson/three-mesh-bvh** (MIT) | MIT | BVH library | SAH-32 bins, worker builds, 32 B BVH2 nodes; WebGPU WGSL traversal via three TSL (`BVHComputeData`, TLAS `ClusteredBVH`) | Builder; GLTFLoader integration |
| **jbikker/tinybvh** (MIT) | MIT | Best-in-class BVH builders and GPU layouts | BVH_GPU / BVH4_GPU / CWBVH; GLSL `tiny.comp`, `tiny4.comp`, `tiny8.comp`; OpenCL `traverse_tlas.cl`, `wavefront*.cl` | Traversal kernels to port to WGSL; WASM build for SBVH + CWBVH |
| **bevyengine/bevy `bevy_solari`** | MIT / Apache-2.0 | ReSTIR DI + GI in WESL (WGSL) with **hardware ray queries** (wgpu-native, not browser) | Passes `initial_and_temporal`, `spatial_and_shade`; light-tile presampling; world cache | Temporal validation (`pixel_dissimilar`, `permute_pixel`), previous-frame visibility, light ID translation across frames, confidence cap. Good WGSL style reference |
| **StuckiSimon/strahl** (MIT) | MIT | OpenPBR path tracer on WebGPU | three-mesh-bvh; published perf on M1 Max | OpenPBR WGSL, oidn-web denoise integration |
| **DQLin/ReSTIR_PT** (Falcor, BSD-3) | BSD-3 | The official GRIS/ReSTIR PT code | Slang, DXR | Ground truth for reservoir fields, seeds and replay: `PathReservoir.slang`, `PathTracer.slang` (`generateRandomReplayPath`), `Shift.slang`, `SpatialReuse.cs.slang`, `TemporalReuse.cs.slang` |
| gnikoloff/webgpu-raytracer, JamesRandall/webgpu-doom-pathtracer, re-ovo/wgpu-path-tracing | mostly MIT | Small WebGPU path tracers | BVH2 + compute | Blog-level references only |

Nothing was cloned; files were read through the GitHub raw/API endpoints (the read-only constraint). If you want local copies, shallow-clone C2-Renderer, tinybvh and three-mesh-bvh into `scratchpad/sources/`.

---

## 7. Profiling and debugging

**GPU timing**
- `timestampWrites: {querySet, beginningOfPassWriteIndex, endOfPassWriteIndex}` on each compute pass, then `resolveQuerySet` into a `QUERY_RESOLVE|COPY_SRC` buffer, then copy to a `MAP_READ` ring buffer and read 2–3 frames later.
- Resolution is quantized to **65.536 µs**: `kTimestampQuantizationMask = 0xFFFF0000` in `src/dawn/common/Constants.h:110`. Chrome's docs round this to "100 microseconds".
- Enable `chrome://flags/#enable-webgpu-developer-features` for full precision. It also adds `GPUAdapterInfo.backend/driver/memoryHeaps` and the `strictMath` shader-module option.
- **Only pass-boundary timestamps** on Apple, so split stages into separate passes. Overhead per pass on Metal is a new compute encoder: µs-level [INFERENCE].
- Wall time: `queue.onSubmittedWorkDone()`. CPU time: `performance.now()`, coarsened unless the page is cross-origin isolated.

**Chrome command-line / flags** [SOURCE: Dawn `docs/dawn/debugging.md`, `Toggles.cpp`]
- `--enable-dawn-features=dump_shaders,disable_symbol_renaming`: the WGSL and **generated MSL** are printed to the DevTools console.
- `dump_shaders_on_failure`.
- `use_user_defined_labels_in_backend`: labels visible in Instruments / the Metal debugger.
- `disable_robustness`: **only to measure the cost of bounds clamping**. Never ship assumptions based on it.
- `enable_immediate_error_handling`.
- `disable_blob_cache`: measure cold compiles.
- `--enable-unsafe-webgpu` (`chrome://flags/#enable-unsafe-webgpu`): unlocks unsafe-experimental WGSL such as `requires chromium_print;`.
  - `print()` becomes Metal `os_log` via Tint `convert_print_to_log.cc`, prefixed `[ comp <entry>:L<line> global_invocation_id(x,y,z) ]`.
  - Needs toggle `enable_shader_print` and macOS 15+ (`MTLCompileOptions.enableLogging`). End-to-end use in Chrome is [UNVERIFIED]; view with Console.app or `log stream`.
- Metal capture: Dawn documents `DAWN_TRACE_FILE_BASE` + `MTL_CAPTURE_ENABLED=1`, which saves a `.gputrace` for Xcode. Inside Chrome's sandboxed GPU process it may need `--disable-gpu-sandbox` [UNVERIFIED].
- **Instruments → Metal System Trace** attached to the "Google Chrome Helper (GPU)" process gives the M5 occupancy counters [UNVERIFIED but standard].

**Print-debugging without printf** (works everywhere) [INFERENCE; C2 uses the same pattern]
- A `debug` storage buffer: `atomic<u32>` counter plus a record array `{pixel, tag, vec4f}`, written only when `pixel == debugPixel` (immediate or uniform set from a mouse click).
- Read it back with `mapAsync` each frame.
- Add AOV visualizations through a `debugMode` uniform: M, W, path length, reconnection index k, Jacobian, replay-needed flag, duplication score, NaN/inf mask.

**Error handling**
- Label every object. Labels appear in validation errors.
- `device.pushErrorScope('validation' | 'out-of-memory' | 'internal')` / `popErrorScope()` around resource and pipeline creation; `device.onuncapturederror`.
- `shaderModule.getCompilationInfo()` for line and column messages.
- `device.lost` → rebuild from CPU-side scene copies. Remember: a **new adapter** is required.

**Watchdog and hangs.** Chrome's GPU watchdog loses the device on very long GPU work: a few seconds, figures vary [SOURCE: toji.dev device-loss best practices; UNVERIFIED exact timeout]. On macOS a runaway shader can hang the desktop [SOURCE: "Deathray", auberon.xyz].
- Keep each `submit` ≲ 100–200 ms.
- For reference accumulation, submit N spp or tiles per `requestAnimationFrame`, as C2 does with tiles.
- Guard loops with iteration caps.

**Extensions**
- **WebGPU Inspector** (brendan-duncan, MIT): live object inspection, frame capture with buffer/texture contents, an experimental **CPU-interpreted WGSL shader debugger** (single-invocation stepping), and a modeled "shader flame graph".
- greggman's **WebGPU Dev Extension** / webgpu-debug-helper: makes encoder errors throw immediately.

**Robustness pitfalls**
- Robust buffer access clamps out-of-bounds indices, so reads return wrong data silently instead of crashing.
- Add debug-build asserts that write to the debug buffer.
- Integer range analysis (Chrome 141, toggle `enable_integer_range_analysis_in_robustness`) removes clamps when it can prove bounds. Simple, provable index math (`min`, masks) helps.

---

## 8. Precision and robustness

### 8.1 Floating-point semantics in Chrome/Metal [MEASURED + SOURCE]

- Dawn prepends `#pragma METAL fp math_mode(relaxed)` on macOS 15+ (`ShaderModuleMTL.mm` l.462–467). It uses `fastMathEnabled = !strictMath` on older macOS.
- `strictMath` is only reachable with the developer-features flag, so **users run relaxed math**. Chrome's description of disabled strict math: may ignore NaN/Inf, treat −0 as +0, replace division with reciprocal multiplication, and reassociate.
- Measured in a compute shader on this Mac:

| Test | Result |
|---|---|
| `0/0` produces a NaN value (`0x7fc00000`); `1/0` produces +inf | yes |
| `nan != nan` | **false** (the compiler folded the self-comparison) |
| Bit-test `isnan` / `isinf` | work |
| `max(NaN, 1.0)` | `1.0` |
| `min(inf, 5.0)` | `5.0` |
| `3e38 * 10` | inf |
| `a*b + c` with a = b = 1.0000001, c = −1.0000002 | `1.42e-14` (the fused result; separate rounding gives 0), so **FMA contraction happens** |

- Consequences:
  - Always use the bit-level helpers below. Falcor's `isnan(w)` guards in `PathReservoir.add/merge` must become these.
  - Never rely on NaN propagation to reject samples; test explicitly and early.

```wgsl
fn is_nan(x: f32) -> bool    { return (bitcast<u32>(x) & 0x7fffffffu) >  0x7f800000u; }
fn is_inf(x: f32) -> bool    { return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u; }
fn is_finite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }
```

### 8.2 Watertight ray/triangle intersection (Woop, Benthin & Wald, JCGT 2(1) 2013) [SOURCE algorithm; WGSL compile-tested]

Algorithm:
1. Per ray: `kz = argmax|d|`, `kx = (kz+1) % 3`, `ky = (kx+1) % 3`; if `d[kz] < 0`, swap kx and ky (preserves winding).
2. Shear constants: `S = (d_kx/d_kz, d_ky/d_kz, 1/d_kz)`.
3. Per triangle: `A = v0−o`, `B = v1−o`, `C = v2−o`; `Ax = A_kx − Sx·A_kz`, `Ay = A_ky − Sy·A_kz`, and the same for B and C.
4. Edge functions: `U = Cx·By − Cy·Bx`, `V = Ax·Cy − Ay·Cx`, `W = Bx·Ay − By·Ax`. Reject if the signs are mixed.
5. `det = U+V+W`; reject if 0.
6. `T = U·(Sz·A_kz) + V·(Sz·B_kz) + W·(Sz·C_kz)`. Sign-consistent range test: `0 < T·sgn(det) ≤ tmax·det·sgn(det)`.
7. `t = T/det`; barycentrics `(U, V, W)/det`, the weights of v0, v1, v2.

WebGPU caveats [INFERENCE]:
- The paper falls back to **double precision** when U, V or W is exactly 0. WGSL has no f64, so exact edge and vertex hits may be missed or double-counted.
- **FMA contraction** can make the same shared edge evaluate to different values in the two adjacent triangles. Mitigation: evaluate each edge function with a **canonical endpoint order** (e.g. by global vertex index) and negate as needed, so both triangles compute bit-identical values.
- Cost is higher than Möller–Trumbore on precomputed edges. Offer it as an `override` for the validation build.
- Expected image-level impact of Möller–Trumbore leaks is tiny, but light leaks through closed geometry show up in Cornell-box-type validation scenes.

### 8.3 Avoiding self-intersection (Wächter & Binder, Ray Tracing Gems ch. 6, 2019) [SOURCE constants; WGSL compile-tested]

```wgsl
const ORIGIN: f32 = 1.0 / 32.0;       // below this |p| use a float offset
const FLOAT_SCALE: f32 = 1.0 / 65536.0;
const INT_SCALE: f32 = 256.0;         // offset in ULPs along the normal
fn offset_ray(p: vec3f, n: vec3f) -> vec3f {  // n = GEOMETRIC normal, flipped to the side the ray leaves
  let of_i = vec3i(INT_SCALE * n);
  let p_i = bitcast<vec3f>(bitcast<vec3i>(p) + select(of_i, -of_i, p < vec3f(0.0)));
  return select(p_i, p + FLOAT_SCALE * n, abs(p) < vec3f(ORIGIN));
}
```

- Compute `p` from **barycentrics on the object-space vertices**, then transform to world space. Do not use `o + t·d`; that is the paper's recommendation for accuracy. With instancing, the transform adds error; NVIDIA's 2022 "Solving Self-Intersection Artifacts in DirectX Raytracing" extends the bound with object-to-world error terms [SOURCE].
- For **reconnection visibility** between y_{k−1} and x_k, offset **both** endpoints along their geometric normals, each toward the other point's side. Then trace `[0, 1)` along `p1' − p0'`.
- **Recentre the scene** at load (subtract the bounding-box centre). The offsets scale with `|p|`, as does f32 error in general [INFERENCE].

### 8.4 Other numerics [INFERENCE]

- Convert random `u32` to float as `f32(x >> 8) * 2^-24`, giving `[0, 1)` exactly. This avoids C2's `clamp(u, 0, 0.9999999)` hack.
- Compute Jacobian ratios such as `p^y(ω)·G(y→x) / (p^x(ω)·G(x→x))` as products of ratios, clamp to finite values, and treat non-finite values as an invalid shift (weight 0) using `is_finite`.
- Keep the reconnection vertex as (instance, prim, bary) and recompute its position.
- Accumulation for the reference: `rgba32float` read_write texture (tier 2) or an `array<vec4f>` buffer with one owner thread per pixel. No float atomics are needed.
- Kahan summation is unnecessary for ≲ 10⁵ spp in f32 [INFERENCE]. Hold samples per batch and add batch means if you go further.

### 8.5 RNG choices and replayable RNG

- **What ReSTIR PT does** [SOURCE: Falcor `ReSTIRPTPass.h` l.83, `PathTracer.slang` l.220–250, `TinyUniformSampleGenerator.slang`, `LCG.slang`, `HashUtils.slang`]:
  - Default `SAMPLE_GENERATOR_TINY_UNIFORM`: a **32-bit-state LCG** (A = 1664525, C = 1013904223).
  - Seeded per pixel/sample by `blockCipherTEA(interleave_32bit(pixel), sampleNumber).x` (16 TEA rounds).
  - The reservoir stores the generator's state at the first bounce (`initRandomSeed = sg.getCurrentSeed()`) and after the rc vertex (`rcRandomSeed`).
  - Random replay recreates the generator from that 32-bit state: `path.sg = SampleGenerator(randomSeed)` in `generateRandomReplayPath`.
- **Enhanced §5** detects duplicated samples "by comparing the random seeds already stored for (potential) random replay" [SOURCE]. So seeds **must be unique per initial candidate** (pixel, frame, candidate). A collision of 32-bit seeds inside a 17×17 window has probability ≈ 288/2³², which is negligible.
- **Recommended in WGSL** [INFERENCE]:
  1. **Stream generator with 32-bit state:** PCG-RXS-M-XS-32. This is the `pcg` hash of Jarzynski & Olano (JCGT 9(3) 2020) iterated as an LCG with an output permutation. The state *is* the seed, so replay just stores a `u32`, exactly like Falcor, with much better quality than a bare LCG.

     ```wgsl
     fn rng_next(state: ptr<function, u32>) -> f32 {
       let s = *state * 747796405u + 2891336453u; *state = s;
       let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
       return f32(((w >> 22u) ^ w) >> 8u) * (1.0 / 16777216.0);
     }
     ```

  2. **Counter-based alternative (preferred):** `u(seed, dim) = pcg3d(vec3u(seed, dim, salt)).x · 2⁻²⁴` (after `>> 8`), with `dim = bounce·D + slot`.
     - Replay from any vertex needs only the path seed and the bounce index.
     - Skipped work (e.g. NEE at non-final vertices during replay; C2's `randoms.md` explains the pitfall) needs no manual stream advancing.
     - `rcRandomSeed` becomes redundant (−4 B per reservoir).
     - The ~2 extra hash evaluations per sample are cheap ALU.
  3. **Seeding:** `seed = pcg3d(vec3u(px, py, frameIndex ^ candidateIndex·K)).x`, or TEA as in Falcor.
  4. **Quasi-random:** Owen-scrambled Sobol (Burley, "Practical Hash-based Owen Scrambling", JCGT 9(4) 2020: `reverseBits`, then a Laine–Karras-style hash, then `reverseBits`; WGSL has `reverseBits`) can drive *initial* sampling if you seed the scramble from the replay seed.
     - Cross-pixel reuse destroys per-pixel low-discrepancy structure anyway, so the benefit is limited.
     - Keep a plain PRNG for correctness validation. Blue noise (spatiotemporal blue noise, Wolfe et al. 2022) is optional for resampling and selection decisions.
  - WGSL `u32` arithmetic wraps mod 2³², and shifts use the shift amount mod 32 for runtime values. Both are fine for hashing.

---

## 9. Risks and open issues (platform)

1. **Performance gap.** Software BVH2 gives ~190–210 Mrays/s for incoherent rays on the M5 Pro. Real-time 1080p ReSTIR PT is unrealistic without a lower internal resolution and/or CWBVH [MEASURED + INFERENCE].
2. **Storage-buffer budget of 10** forces merged buffers and careful binding design. Chrome 153+ `buffer_view` helps.
3. **Relaxed math** (no NaN self-compare, FMA contraction, reassociation) can silently break NaN guards and watertightness, and makes WebGPU results differ numerically from Cycles. Validate with statistical tolerances, not bit-exactness.
4. **No bindless:** texture resampling to size classes costs load time and memory; BC7 needs a KTX2/Basis pipeline.
5. **Compile times** of 0.2–0.6 s per large pipeline, multiplied by every `override` combination.
6. **Temporal reuse needs previous-frame scene data** (TLAS, transforms, lights) for correct MIS in dynamic scenes; that means extra memory and code paths.
7. **Device loss / watchdog** during long reference renders. Batch the submits.
8. **Lossy reservoir compression** adds bias. Keep the uncompressed debug layout for validation.
9. **Pairing-texture generation.** The Enhanced paper's CUDA index-table method is racy by design. Generate pairing textures on the CPU in JS once (254², 230², 210²) and upload them as `rg16sint` or packed `r32uint` [INFERENCE].

---

## 10. Sources

**Chrome, Dawn and Tint**
- Chrome WebGPU release notes index: https://developer.chrome.com/docs/web-platform/webgpu/news
- Chrome 153–154: https://developer.chrome.com/blog/new-in-webgpu-153-154
- Chrome 151–152: https://developer.chrome.com/blog/new-in-webgpu-151-152
- Chrome 149–150: https://developer.chrome.com/blog/new-in-webgpu-149-150
- Chrome 147–148: https://developer.chrome.com/blog/new-in-webgpu-147-148
- Chrome 145: https://developer.chrome.com/blog/new-in-webgpu-145
- Chrome 144: https://developer.chrome.com/blog/new-in-webgpu-144
- Chrome 142: https://developer.chrome.com/blog/new-in-webgpu-142
- Chrome 141: https://developer.chrome.com/blog/new-in-webgpu-141
- Chrome 131: https://developer.chrome.com/blog/new-in-webgpu-131
- Chrome 130: https://developer.chrome.com/blog/new-in-webgpu-130
- Chrome developer features: https://developer.chrome.com/docs/web-platform/webgpu/developer-features
- Chrome troubleshooting: https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips
- Dawn source (main): `src/dawn/native/metal/ShaderModuleMTL.mm`, `src/dawn/native/metal/PhysicalDeviceMTL.mm`, `src/dawn/native/Limits.cpp`, `src/dawn/native/Toggles.cpp`, `src/dawn/native/ComputePassEncoder.cpp`, `src/dawn/common/Constants.h`, `src/tint/lang/wgsl/feature_status.cc`, `src/tint/lang/wgsl/wgsl.def`, `src/tint/lang/msl/writer/raise/convert_print_to_log.cc`, `docs/dawn/debugging.md`, `docs/tint/extensions/chromium_experimental_resource_table.md` (https://github.com/google/dawn)

**Specifications and proposals**
- WebGPU spec (texture-formats tiers): https://gpuweb.github.io/gpuweb/
- Bindless proposal: https://github.com/gpuweb/gpuweb/blob/main/proposals/bindless.md and https://github.com/gpuweb/gpuweb/issues/5517

**Apple GPU guidance**
- Apple M5/A19 GPU tech talk: https://developer.apple.com/videos/play/tech-talks/111431/
- M3 dynamic caching: https://developer.apple.com/videos/play/tech-talks/111375/

**BVH libraries and GPU traversal**
- tinybvh: https://github.com/jbikker/tinybvh (README; `external/vulkanrt/shaders/tiny.comp`, `tiny8.comp`, `README_vulkan.md`)
- Gigi tinybvh WebGPU demo: https://electronicarts.github.io/gigi/Demos/tinybvh/index.html
- three-mesh-bvh: https://github.com/gkjohnson/three-mesh-bvh (`src/core/Constants.js`, `src/core/build/splitUtils.js`, `src/webgpu/*`)

**Reference implementations**
- C2-Renderer: https://github.com/Domenicobrz/C2-Renderer
- Web-RTRT: https://github.com/C-none/Web-RTRT
- Bevy Solari: https://github.com/bevyengine/bevy/tree/main/crates/bevy_solari
- Falcor ReSTIR PT: https://github.com/DQLin/ReSTIR_PT (`Source/RenderPasses/ReSTIRPTPass/PathReservoir.slang`, `PathTracer.slang`, `ReSTIRPTPass.h`; `Source/Falcor/Utils/Sampling/*`)
- strahl paper: https://arxiv.org/html/2407.19977v1
- M1 ray tracing performance: https://www.willusher.io/graphics/2020/12/20/rt-dive-m1/

**Debugging tools and device loss**
- WebGPU Inspector: https://github.com/brendan-duncan/webgpu_inspector
- WebGPU Dev Extension: https://github.com/greggman/webgpu-dev-extension
- Device loss best practices: https://toji.dev/webgpu-best-practices/device-loss.html
- Deathray (macOS WebGPU hang): https://auberon.xyz/blog/posts/deathray/

**Papers**
- Ylitie, Karras, Laine 2017, CWBVH: https://research.nvidia.com/publication/2017-07_efficient-incoherent-ray-traversal-gpus-through-compressed-wide-bvhs
- Wächter & Binder 2019: https://link.springer.com/chapter/10.1007/978-1-4842-4427-2_6
- Woop, Benthin, Wald 2013, JCGT 2(1): https://jcgt.org/published/0002/01/05/
- Baldwin & Weber 2016, JCGT 5(3): https://jcgt.org/published/0005/03/03/
- Jarzynski & Olano 2020, JCGT 9(3); Burley 2020, JCGT 9(4); Ize 2013, JCGT 2(2); Akenine-Möller et al. 2019 (RTG ch. 20) and 2021 (JCGT 10(1)); Laine et al. 2013 HPG; Aila & Laine 2009 HPG (cited from memory, not re-fetched)

**Local paper copies**
- Enhanced paper: `restirpt_enhanced_2026.pdf`, pp. 13:4–13:12 (Eq. 5 on p.7; §6.2 and Table 1 on pp.10–12)
- GRIS: `gris_sig22.pdf` §8.2–8.3 (p.75:15)

**Package metadata**
- npm registry: three-mesh-bvh 0.9.15 (MIT), three-gpu-pathtracer 0.0.25 (MIT), @gltf-transform/core 4.5.1 (MIT), tinyusdz 0.9.1 (Apache-2.0 / MIT), @needle-tools/usd 1.1.2 (**PolyForm-Noncommercial**, license risk), wesl 0.7.31 (MIT), webgpu-utils 2.1.1 (MIT), typegpu 0.12.6 (MIT)
